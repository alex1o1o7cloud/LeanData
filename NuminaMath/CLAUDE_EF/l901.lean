import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l901_90109

/-- The area of a triangle given its vertices' coordinates -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- The area of a rectangle given its width and height -/
def rectangle_area (width height : ℝ) : ℝ :=
  width * height

theorem triangle_fraction_of_grid :
  let a_x := (2 : ℝ)
  let a_y := (4 : ℝ)
  let b_x := (7 : ℝ)
  let b_y := (2 : ℝ)
  let c_x := (6 : ℝ)
  let c_y := (6 : ℝ)
  let grid_width := (8 : ℝ)
  let grid_height := (6 : ℝ)
  (triangle_area a_x a_y b_x b_y c_x c_y) / (rectangle_area grid_width grid_height) = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_fraction_of_grid_l901_90109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l901_90140

-- Define the function g(x) with parameter c
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + c) / (x^2 - 2*x - 24)

-- Define what it means for a function to have a vertical asymptote at a point
def has_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε

-- Theorem statement
theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, has_vertical_asymptote (g c) x) ↔ (c = -48 ∨ c = -8) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l901_90140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_coin_toss_probability_l901_90182

theorem three_coin_toss_probability : 
  (7 : ℚ) / 8 = 1 - (1 : ℚ) / (2^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_coin_toss_probability_l901_90182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l901_90131

/-- Parabola with equation y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1}

/-- Circle with equation (x-4)² + (y-1)² = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4) ^ 2 + (p.2 - 1) ^ 2 = 1}

/-- Focus of the parabola y² = 4x -/
def FocusOfParabola : ℝ × ℝ := (1, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem min_distance_sum :
  ∃ (c : ℝ), c = 4 ∧
  ∀ (M : ℝ × ℝ) (A : ℝ × ℝ),
  M ∈ Parabola → A ∈ Circle →
  c ≤ distance M A + distance M FocusOfParabola := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l901_90131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l901_90170

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sum of first n terms of an arithmetic sequence
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_9 (a : ℕ → ℚ) :
  arithmetic_sequence a →
  2 * a 3 + a 9 = 33 →
  sum_arithmetic a 9 = 99 :=
by
  sorry

#check arithmetic_sequence_sum_9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_9_l901_90170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l901_90169

/-- Represents a parallelogram with given dimensions -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  slant : ℝ

/-- Calculates the area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  p.base * p.height

/-- Calculates the length of the diagonal from origin to (base + slant, height) -/
noncomputable def diagonalLength (p : Parallelogram) : ℝ :=
  Real.sqrt ((p.base + p.slant)^2 + p.height^2)

theorem parallelogram_properties (p : Parallelogram) 
    (h1 : p.base = 10) 
    (h2 : p.height = 4) 
    (h3 : p.slant = 6) : 
    area p = 40 ∧ diagonalLength p = Real.sqrt 272 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l901_90169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_rotation_translation_l901_90145

/-- The transformation matrix M represents a 90-degree counterclockwise rotation followed by a 2-unit translation to the right. -/
theorem transformation_matrix_rotation_translation :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 2; 1, 0, 0; 0, 0, 1]
  ∀ (x y : ℝ), 
    let v : Fin 3 → ℝ := ![x, y, 1]
    let w : Fin 3 → ℝ := M.mulVec v
  (w 0 = -y + 2 ∧ w 1 = x ∧ w 2 = 1) := by
  sorry

#check transformation_matrix_rotation_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_matrix_rotation_translation_l901_90145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l901_90155

open Real

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define N as the midpoint of AC
noncomputable def N (A C : ℝ × ℝ) : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define Q as a point on AC between A and N
variable (Q : ℝ × ℝ)

-- Define the condition that Q is between A and N
def Q_between_A_and_N (A C Q : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ Q = (A.1 + t * (N A C).1 - A.1, A.2 + t * (N A C).2 - A.2)

-- Define the area ratio s
noncomputable def s (A B C Q : ℝ × ℝ) : ℝ := 
  (abs ((Q.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (Q.2 - A.2))) / 
  (abs ((C.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2)))

-- State the theorem
theorem area_ratio_bounds {A B C Q : ℝ × ℝ} (h : Q_between_A_and_N A C Q) : 
  0 < s A B C Q ∧ s A B C Q ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l901_90155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l901_90120

def S (n : ℕ) : ℕ := n^2 + n + 1

def a : ℕ → ℕ
| 0 => 3  -- Add this case to handle n = 0
| 1 => 3
| (n + 2) => 2 * (n + 2)

theorem sequence_formula (n : ℕ) : 
  (∀ k, S k = k^2 + k + 1) → 
  a n = if n ≤ 1 then 3 else 2 * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l901_90120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_of_each_expected_prize_money_profitable_promotion_l901_90184

-- Define the number of models for each product type
def washing_machines : Nat := 2
def televisions : Nat := 2
def air_conditioners : Nat := 3

-- Define the number of models to be selected
def selected_models : Nat := 4

-- Define the price increase for the promotion
noncomputable def price_increase : ℝ := 150

-- Define the number of lottery chances per purchase
def lottery_chances : Nat := 3

-- Define the probability of winning a prize
noncomputable def win_probability : ℝ := 1/2

-- Define the prize money as a positive real number
variable (m : ℝ) (hm : m > 0)

-- Theorem 1: Probability of selecting at least one model of each type
theorem probability_at_least_one_of_each :
  (Nat.choose washing_machines 1 * Nat.choose televisions 1 * Nat.choose air_conditioners 2 +
   Nat.choose washing_machines 1 * Nat.choose televisions 1 * Nat.choose air_conditioners 1) /
  Nat.choose (washing_machines + televisions + air_conditioners) selected_models = 24/35 := by
  sorry

-- Theorem 2: Expected value of total prize money
theorem expected_prize_money :
  (lottery_chances : ℝ) * m * win_probability = 1.5 * m := by
  sorry

-- Theorem 3: Condition for profitable promotion
theorem profitable_promotion :
  (lottery_chances : ℝ) * m * win_probability < price_increase → m < 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_of_each_expected_prize_money_profitable_promotion_l901_90184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_location_l901_90146

theorem reciprocal_location (a b : ℝ) (ha : a > 0) (hb : b > 0) (hmag : a^2 + b^2 > 1) :
  let F : ℂ := -a + b * Complex.I
  let recip : ℂ := 1 / F
  (Complex.re recip < 0 ∧ Complex.im recip < 0) ∧ Complex.abs recip < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_location_l901_90146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90118

noncomputable def f (θ : Real) (x : Real) : Real :=
  Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

theorem function_properties (θ : Real) :
  (∀ x, f θ (-x) = -(f θ x)) →
  (∀ x y, -π/4 ≤ x ∧ x < y ∧ y ≤ 0 → f θ y < f θ x) →
  θ = 5*π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l901_90166

theorem roots_of_equation (x : ℝ) : 
  3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) = 9 ↔ 
    x = ((9 + 3 * Real.sqrt 5) / 6)^2 ∨ 
    x = ((9 - 3 * Real.sqrt 5) / 6)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l901_90166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_weighted_average_l901_90177

/-- Calculates the weighted average of marks given marks and weights -/
noncomputable def weighted_average (marks : List ℝ) (weights : List ℝ) : ℝ :=
  (List.sum (List.zipWith (· * ·) marks weights)) / (List.sum weights)

theorem david_weighted_average :
  let marks := [76, 65, 82, 67, 85]
  let weights := [0.20, 0.25, 0.25, 0.15, 0.15]
  weighted_average marks weights = 74.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_weighted_average_l901_90177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_property_l901_90180

def n : ℕ := 987654321

def first_digit (x : ℕ) : ℕ := 
  if x < 10 then x else first_digit (x / 10)

def last_digit (x : ℕ) : ℕ := x % 10

def last_two_digits (x : ℕ) : ℕ := x % 100

theorem number_property :
  (∀ m : ℕ, m ∈ ({18, 27, 36, 45, 54, 63, 72, 81, 99} : Set ℕ) → 
    first_digit (n * m) = first_digit m ∧ 
    last_digit (n * m) = last_digit m) ∧
  last_two_digits (n * 90) = 90 := by
  sorry

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_property_l901_90180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l901_90112

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := sorry

/-- Represents the original speed of the current -/
def current_speed : ℝ := sorry

/-- The time it takes to sail from Port A to Port B against the original current -/
def time_original : ℝ := sorry

/-- The time it takes to sail from Port A to Port B after the current speed doubles -/
def time_doubled : ℝ := sorry

/-- The distance between Port A and Port B -/
def distance : ℝ := sorry

theorem boat_travel_time :
  (time_original = 2) →
  (time_doubled = 3) →
  (distance = 2 * (boat_speed - current_speed)) →
  (distance = 3 * (boat_speed - 2 * current_speed)) →
  (distance / (boat_speed + 2 * current_speed) = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_travel_time_l901_90112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_false_l901_90100

theorem triangle_inequality_false (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_order : a > b ∧ b > c) : 
  ¬(b + c > 2*a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_false_l901_90100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_inverse_difference_l901_90127

theorem real_part_of_inverse_difference (z : ℂ) (h1 : z ≠ 2) (h2 : Complex.abs z = 2) :
  (1 / (2 - z)).re = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_inverse_difference_l901_90127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l901_90152

/-- Given a conic section x^2 + my^2 = 1 with eccentricity 2, prove that m = -1/3 -/
theorem conic_section_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →  -- Condition 1: Equation of the conic section
  (Real.sqrt (1 + (-1/m)) = 2) →  -- Condition 2: Eccentricity is 2
  m = -1/3 :=                     -- Conclusion: m = -1/3
by
  intro h1 h2
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_eccentricity_l901_90152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l901_90101

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.sqrt 3 * Real.cos x, 1)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin (Real.pi / 2 + x), Real.sqrt 3 / 2)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  a = 3 →
  f (A / 2 + Real.pi / 12) = 1 / 2 →
  Real.sin C = 2 * Real.sin B →
  A = Real.pi / 3 ∧ b = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l901_90101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l901_90171

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (a r : ℝ) :
  (geometric_sequence a r 3 = 13) →
  (geometric_sequence a r 5 = 121) →
  (geometric_sequence a r 8 = 3280) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l901_90171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l901_90105

-- Define the function f(x) = x / (x^2 + 1)
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Define the domain (-1, 1)
def domain : Set ℝ := Set.Ioo (-1) 1

theorem f_properties :
  -- Part 1: f is increasing on (-1, 1)
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) ∧
  -- Part 2: The solution set of f(2x-1) + f(x) < 0 is (0, 1/3)
  {x : ℝ | f (2*x - 1) + f x < 0} = Set.Ioo 0 (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l901_90105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABDF_is_500_l901_90117

/-- Rectangle ACDE with given properties -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Point B on AC -/
noncomputable def trisection_point (rect : Rectangle) : ℝ := rect.length / 3

/-- Point F on AE -/
noncomputable def midpoint_width (rect : Rectangle) : ℝ := rect.width / 2

/-- Area of the quadrilateral ABDF -/
noncomputable def area_ABDF (rect : Rectangle) : ℝ :=
  rect.length * rect.width - (2 * rect.length / 3 * rect.width) / 2 - (rect.width / 2 * rect.length) / 2

/-- Theorem stating the area of ABDF is 500 -/
theorem area_ABDF_is_500 (rect : Rectangle) 
  (h1 : rect.length = 40) 
  (h2 : rect.width = 30) : 
  area_ABDF rect = 500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ABDF_is_500_l901_90117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_properties_l901_90153

/-- Given a triangle with sides 8, 10, and 12, prove properties of a similar triangle with perimeter 150 -/
theorem similar_triangle_properties :
  ∃ (s a : ℝ → ℝ → ℝ → ℝ) (x : ℝ),
    (s 8 10 12 = (8 + 10 + 12) / 2) ∧  -- Definition of semiperimeter
    (a 8 10 12 = Real.sqrt (s 8 10 12 * (s 8 10 12 - 8) * (s 8 10 12 - 10) * (s 8 10 12 - 12))) ∧  -- Definition of area using Heron's formula
    8 * x + 10 * x + 12 * x = 150 ∧  -- Perimeter condition
    12 * x = 60 ∧  -- Longest side
    a (8 * x) (10 * x) (12 * x) = Real.sqrt 985875 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_properties_l901_90153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_parabola_circle_l901_90172

/-- The parabola equation -/
noncomputable def parabola (p q x : ℝ) : ℝ := -p/2 * x^2 + q

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The area enclosed by the parabola and x-axis -/
noncomputable def enclosed_area (p : ℝ) : ℝ :=
  2 * (p^2 + 1)^(3/2) / (3 * p^2)

/-- The theorem statement -/
theorem min_area_parabola_circle (p q : ℝ) :
  p > 0 → q > 0 →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    circle_eq x₁ (parabola p q x₁) ∧ 
    circle_eq x₂ (parabola p q x₂)) →
  ∃ p_min : ℝ, p_min > 0 ∧ 
    ∀ p' : ℝ, p' > 0 → enclosed_area p_min ≤ enclosed_area p' ∧ 
    enclosed_area p_min = Real.sqrt 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_parabola_circle_l901_90172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_f_l901_90142

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 - 3*x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≠ 1 ∧ x ≠ 2}

-- Theorem stating that the complement of the domain is {1, 2}
theorem complement_of_domain_f : 
  (domain_f)ᶜ = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_f_l901_90142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_pyramid_rows_l901_90116

/-- Represents a pyramid display of cans -/
structure CanPyramid where
  topRowCans : ℕ
  rowIncrease : ℕ
  totalCans : ℕ

/-- Calculates the number of rows in a can pyramid -/
def numRows (p : CanPyramid) : ℕ :=
  let a : Int := 2
  let b : Int := 1
  let c : Int := -p.totalCans
  (((-b) + (b * b - 4 * a * c).sqrt) / (2 * a)).toNat

/-- Theorem stating the number of rows in the specific can pyramid -/
theorem can_pyramid_rows (p : CanPyramid) 
  (h1 : p.topRowCans = 3) 
  (h2 : p.rowIncrease = 4) 
  (h3 : p.totalCans = 225) : 
  numRows p = 10 := by
  sorry

#eval numRows { topRowCans := 3, rowIncrease := 4, totalCans := 225 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_pyramid_rows_l901_90116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_chinese_athletes_adjacent_probability_l901_90163

/-- The probability of two specific athletes being adjacent in a random arrangement of athletes -/
theorem adjacent_probability (n : ℕ) (k : ℕ) : 
  2 ≤ n → k = 2 → 
  (n - k + 1 : ℚ) / n = 1 / 4 ↔ n = 8 :=
by
  sorry

/-- The probability of two Chinese athletes being adjacent in a random arrangement of 8 athletes -/
theorem chinese_athletes_adjacent_probability :
  let total_athletes : ℕ := 8
  let chinese_athletes : ℕ := 2
  let foreign_athletes : ℕ := 6
  (total_athletes - chinese_athletes + 1 : ℚ) / total_athletes = 1 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_probability_chinese_athletes_adjacent_probability_l901_90163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unsold_tomatoes_percentage_l901_90123

def initial_harvest : ℝ := 340.2
def sold_to_maxwell : ℝ := 125.5
def sold_to_wilson : ℝ := 78.25
def sold_to_brown : ℝ := 43.8
def sold_to_johnson : ℝ := 56.65

theorem unsold_tomatoes_percentage :
  let total_sold := sold_to_maxwell + sold_to_wilson + sold_to_brown + sold_to_johnson
  let unsold := initial_harvest - total_sold
  let percentage := (unsold / initial_harvest) * 100
  ∃ ε > 0, |percentage - 10.58| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unsold_tomatoes_percentage_l901_90123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_probability_theorem_l901_90126

/-- Represents the population of Unlucky Island -/
structure Population where
  hare_ratio : ℝ
  rabbit_ratio : ℝ
  hare_mislead_rate : ℝ
  rabbit_mislead_rate : ℝ

/-- The probability of an animal being a hare given it made both statements -/
noncomputable def probability_hare (pop : Population) : ℝ :=
  let p_hare := pop.hare_ratio
  let p_rabbit := pop.rabbit_ratio
  let p_hare_mislead := pop.hare_mislead_rate
  let p_rabbit_mislead := pop.rabbit_mislead_rate
  let p_hare_correct := 1 - p_hare_mislead
  let p_rabbit_correct := 1 - p_rabbit_mislead
  let numerator := p_hare * p_hare_mislead * p_hare_correct
  let denominator := numerator + p_rabbit * p_rabbit_correct * p_rabbit_mislead
  numerator / denominator

/-- The theorem to be proved -/
theorem hare_probability_theorem (pop : Population) 
  (h1 : pop.hare_ratio = 1/2)
  (h2 : pop.rabbit_ratio = 1/2)
  (h3 : pop.hare_mislead_rate = 1/4)
  (h4 : pop.rabbit_mislead_rate = 1/3) :
  probability_hare pop = (3/32) / ((3/32) + (1/9)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hare_probability_theorem_l901_90126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90151

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - a + 1) * x^(a + 2)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + x

-- State the theorem
theorem function_properties :
  -- f is a power function and an odd function
  (∃ (k : ℝ), ∀ x, f 1 x = k * x^(1 + 2)) ∧ 
  (∀ x, f 1 (-x) = -(f 1 x)) →
  -- The value of a is 1
  (1 : ℝ) = 1 ∧ 
  -- The only zero of g(x) is 0
  (∀ x : ℝ, g 1 x = 0 ↔ x = 0) ∧
  -- There does not exist a natural number n such that g(n) = 900
  ¬∃ n : ℕ, g 1 n = 900 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_angle_properties_l901_90106

theorem unit_circle_angle_properties (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1^2 + P.2^2 = 1 ∧ P.1 = 4/5 ∧ P.2 = -3/5) →
  (Real.cos α = 4/5 ∧ Real.tan α = -3/4 ∧ Real.sin (α + π) = 3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_circle_angle_properties_l901_90106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subsequence_divides_l901_90114

/-- A sequence of positive integers -/
def PositiveIntegerSequence := ℕ → ℕ+

/-- Predicate for one positive integer dividing another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Predicate for a subsequence where no term divides another -/
def noDivides (s : PositiveIntegerSequence) (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → ¬(divides (s (f i)) (s (f j))) ∧ ¬(divides (s (f j)) (s (f i)))

/-- Predicate for a subsequence where for every pair of terms, one divides the other -/
def allDivides (s : PositiveIntegerSequence) (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → (divides (s (f i)) (s (f j))) ∨ (divides (s (f j)) (s (f i)))

/-- The main theorem -/
theorem infinite_subsequence_divides (s : PositiveIntegerSequence) 
  (h : ∀ i j : ℕ, i ≠ j → s i ≠ s j) : 
  (∃ f : ℕ → ℕ, StrictMono f ∧ noDivides s f) ∨ 
  (∃ f : ℕ → ℕ, StrictMono f ∧ allDivides s f) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_subsequence_divides_l901_90114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_course_assessment_probabilities_l901_90110

/-- Represents a student in the course -/
inductive Student
| A
| B
| C

/-- Probability of passing the theory part for a given student -/
def theory_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.6
  | Student.B => 0.5
  | Student.C => 0.4

/-- Probability of passing the experiment part for a given student -/
def experiment_prob (s : Student) : ℝ :=
  match s with
  | Student.A => 0.5
  | Student.B => 0.6
  | Student.C => 0.75

/-- Probability of passing the course for a given student -/
def course_prob (s : Student) : ℝ :=
  theory_prob s * experiment_prob s

/-- Theorem stating the probabilities of passing -/
theorem course_assessment_probabilities :
  (((theory_prob Student.A * theory_prob Student.B * (1 - theory_prob Student.C)) +
    (theory_prob Student.A * (1 - theory_prob Student.B) * theory_prob Student.C) +
    ((1 - theory_prob Student.A) * theory_prob Student.B * theory_prob Student.C)) +
   (theory_prob Student.A * theory_prob Student.B * theory_prob Student.C) = 0.5) ∧
  (course_prob Student.A * course_prob Student.B * course_prob Student.C = 0.027) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_course_assessment_probabilities_l901_90110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_age_calculation_l901_90168

theorem group_age_calculation (num_girls num_boys : ℕ) (age_diff youngest_boy_age : ℕ) : 
  num_girls = 3 →
  num_boys = 6 →
  age_diff = 2 →
  youngest_boy_age = 5 →
  (List.sum (List.range num_boys |>.map (λ i => youngest_boy_age + i)) +
   List.sum (List.range num_girls |>.map (λ i => youngest_boy_age + i + age_diff))) = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_age_calculation_l901_90168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_fourth_power_l901_90148

theorem quadratic_roots_fourth_power (a b : ℝ) (r₁ r₂ : ℂ) :
  (r₁^2 + a*r₁ + b = 0) →
  (r₂^2 + a*r₂ + b = 0) →
  (r₁^4)^2 + (-a^4 + 4*a*b^2 - 2*b^2)*(r₁^4) + b^4 = 0 ∧
  (r₂^4)^2 + (-a^4 + 4*a*b^2 - 2*b^2)*(r₂^4) + b^4 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_fourth_power_l901_90148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_set_m_range_l901_90108

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x + 5/3

-- Statement 1: f is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

-- Statement 2: Set of a that satisfy f(2a - a^2) + f(3) > 0
theorem a_set : ∀ a : ℝ, f (2*a - a^2) + f 3 > 0 ↔ -1 < a ∧ a < 3 := by sorry

-- Statement 3: Range of m for which ∀ x₁ ∈ [-1,1], ∃ x₂ ∈ [-1,1] such that f(x₁) = g(x₂)
theorem m_range : 
  ∀ m : ℝ, (∀ x₁ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ 1 → 
    ∃ x₂ : ℝ, -1 ≤ x₂ ∧ x₂ ≤ 1 ∧ f x₁ = g m x₂) ↔ 
  (m ≤ -3/2 ∨ m ≥ 3/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_a_set_m_range_l901_90108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l901_90121

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x ∈ Set.Ioo 0 (π/2), f x * tan x > deriv f x) →
  sqrt 3 * f (π/6) > f (π/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l901_90121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l901_90102

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem problem_solution (n : ℕ+) : 
  floor (floor (91 / (n : ℝ)) / (n : ℝ)) = 1 → n.val ∈ ({7, 8, 9} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l901_90102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_pyramid_volume_l901_90162

/-- Represents a pyramid with a rectangular base -/
structure Pyramid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := (1 / 3) * p.length * p.width * p.height

/-- Represents the transformation applied to the pyramid -/
noncomputable def transform (p : Pyramid) : Pyramid :=
  { length := 3 * p.length,
    width := (1 / 2) * p.width,
    height := p.height }

theorem modified_pyramid_volume 
  (p : Pyramid) 
  (h_volume : volume p = 60) :
  volume (transform p) = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_modified_pyramid_volume_l901_90162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_water_percentage_is_70_percent_l901_90137

-- Define the initial volumes and percentages
noncomputable def initial_volume : ℝ := 300
noncomputable def initial_water_percentage : ℝ := 60
noncomputable def initial_acid_percentage : ℝ := 40
noncomputable def added_water : ℝ := 100

-- Define the final volume
noncomputable def final_volume : ℝ := initial_volume + added_water

-- Define the initial water volume
noncomputable def initial_water_volume : ℝ := (initial_water_percentage / 100) * initial_volume

-- Define the final water volume
noncomputable def final_water_volume : ℝ := initial_water_volume + added_water

-- Theorem statement
theorem final_water_percentage_is_70_percent :
  (final_water_volume / final_volume) * 100 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_water_percentage_is_70_percent_l901_90137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_for_sum_inequality_l901_90193

open BigOperators

theorem exists_k_for_sum_inequality (m : ℕ) : ∃ k : ℕ,
  1 ≤ (∑ i in Finset.range (k - 1), (i + 1) ^ m) / k ^ m ∧
  (∑ i in Finset.range (k - 1), (i + 1) ^ m) / k ^ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_for_sum_inequality_l901_90193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_negative_420_deg_l901_90136

-- Define the cosecant function
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

-- Convert degrees to radians
noncomputable def deg_to_rad (θ : ℝ) : ℝ := θ * (Real.pi / 180)

-- State the theorem
theorem csc_negative_420_deg :
  csc (deg_to_rad (-420)) = -(2 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_csc_negative_420_deg_l901_90136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l901_90175

def count_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ i ↦ n % i = 0) (Finset.range (n + 1))).card

theorem smallest_with_eight_factors : 
  ∀ n : ℕ, n > 0 → (count_factors n = 8 → n ≥ 24) ∧ count_factors 24 = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_eight_factors_l901_90175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_diameter_approximation_l901_90139

/-- The volume of a cylinder given its radius and height -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The diameter of a cylinder given its radius -/
def cylinderDiameter (r : ℝ) : ℝ := 2 * r

/-- Approximation relation for real numbers -/
def approx (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cylinder_diameter_approximation (h V : ℝ) (h_pos : h > 0) (V_pos : V > 0) 
  (h_val : h = 14) (V_val : V = 1099.5574287564277) :
  ∃ d : ℝ, cylinderVolume (d/2) h = V ∧ approx d 9.99528 0.00001 := by
  sorry

#check cylinder_diameter_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_diameter_approximation_l901_90139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l901_90115

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating that if the difference between compound and simple interest
    is 1 for a principal P, rate 4%, and time 2 years, then P = 625 -/
theorem interest_difference_implies_principal :
  ∀ P : ℝ,
  compoundInterest P 4 2 - simpleInterest P 4 2 = 1 →
  P = 625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l901_90115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tier_level_is_10000_l901_90174

/-- Represents the two-tiered tax system for imported cars -/
structure TaxSystem where
  firstTierRate : ℚ
  secondTierRate : ℚ
  firstTierLevel : ℚ

/-- Calculates the tax for a given car price under the tax system -/
def calculateTax (system : TaxSystem) (carPrice : ℚ) : ℚ :=
  if carPrice ≤ system.firstTierLevel then
    system.firstTierRate * carPrice
  else
    system.firstTierRate * system.firstTierLevel + 
    system.secondTierRate * (carPrice - system.firstTierLevel)

/-- Theorem: The first tier's price level is $10,000 -/
theorem first_tier_level_is_10000 
  (taxSystem : TaxSystem)
  (carPrice : ℚ)
  (taxPaid : ℚ)
  (h1 : taxSystem.firstTierRate = 1/4)
  (h2 : taxSystem.secondTierRate = 3/20)
  (h3 : carPrice = 30000)
  (h4 : taxPaid = 5500)
  (h5 : calculateTax taxSystem carPrice = taxPaid) :
  taxSystem.firstTierLevel = 10000 := by
  sorry

#eval calculateTax ⟨1/4, 3/20, 10000⟩ 30000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tier_level_is_10000_l901_90174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_ratio_l901_90133

-- Define the parabola
noncomputable def parabola (a : ℝ) (x y : ℝ) : Prop := y = a * x^2 ∧ a > 0

-- Define the directrix distance
noncomputable def directrix_distance (a : ℝ) : ℝ := 1 / (4 * a)

-- Define the point M
def M : ℝ × ℝ := (3, 2)

-- Define the point N
noncomputable def N (l : ℝ) : ℝ × ℝ := (l, l)

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem parabola_minimum_ratio :
  ∃ (a : ℝ) (F : ℝ × ℝ) (l : ℝ),
    parabola a M.1 M.2 ∧
    directrix_distance a = 4 ∧
    (∀ (P : ℝ × ℝ), line_l P.1 P.2 →
      (distance P (N l) - 1) / distance P F ≥ (2 - Real.sqrt 2) / 4) ∧
    (∃ (P : ℝ × ℝ), line_l P.1 P.2 ∧
      (distance P (N l) - 1) / distance P F = (2 - Real.sqrt 2) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_minimum_ratio_l901_90133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l901_90154

def sequenceA (n : ℕ+) : ℚ := (-1)^n.val * (n.val^2 : ℚ) / (n.val + 1)

theorem sequence_correct : 
  (sequenceA 1 = -1/2) ∧ 
  (sequenceA 2 = 4/3) ∧ 
  (sequenceA 3 = -9/4) ∧ 
  (sequenceA 4 = 16/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l901_90154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_points_l901_90198

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x + 4 - 4*m^2 = 0

-- Define point N
def point_N : ℝ × ℝ := (2, 0)

-- Define the ellipse C
def ellipse_C (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m^2 + y^2 / (m^2 - 4) = 1

-- Helper function to check if a point is on the perpendicular bisector of a line segment
def is_on_perp_bisector (P Q N : ℝ × ℝ) : Prop :=
  (Q.1 - (P.1 + N.1) / 2)^2 + (Q.2 - (P.2 + N.2) / 2)^2 = 
  ((P.1 - N.1) / 2)^2 + ((P.2 - N.2) / 2)^2

-- Helper function to check if three points are collinear
def are_collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

-- Main theorem
theorem ellipse_and_fixed_points (m : ℝ) (h : m > 2) :
  (∀ x y : ℝ, (∃ P : ℝ × ℝ, circle_M m P.1 P.2 ∧ 
    (∃ Q : ℝ × ℝ, Q = (x, y) ∧ 
      is_on_perp_bisector P Q point_N ∧ 
      are_collinear (0, 0) P Q)) 
    ↔ ellipse_C m x y) ∧
  (m = Real.sqrt 5 → 
    ∃ E : ℝ × ℝ, E.1 = Real.sqrt 30 / 3 ∧ E.2 = 0 ∧
      ∀ A B : ℝ × ℝ, ellipse_C (Real.sqrt 5) A.1 A.2 → 
        ellipse_C (Real.sqrt 5) B.1 B.2 →
        (B.2 - E.2) * (A.1 - E.1) = (B.1 - E.1) * (A.2 - E.2) →
        1 / ((A.1 - E.1)^2 + (A.2 - E.2)^2) + 
        1 / ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 6) ∧
    (∃ E : ℝ × ℝ, E.1 = -Real.sqrt 30 / 3 ∧ E.2 = 0 ∧
      ∀ A B : ℝ × ℝ, ellipse_C (Real.sqrt 5) A.1 A.2 → 
        ellipse_C (Real.sqrt 5) B.1 B.2 →
        (B.2 - E.2) * (A.1 - E.1) = (B.1 - E.1) * (A.2 - E.2) →
        1 / ((A.1 - E.1)^2 + (A.2 - E.2)^2) + 
        1 / ((B.1 - E.1)^2 + (B.2 - E.2)^2) = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_points_l901_90198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l901_90119

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, (1/2)^(abs (x-1)) ≥ a → False
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + (a-2)*x + 9/8 > 0

-- Define the range of a
def range_a (a : ℝ) : Prop := (1/2 < a ∧ a ≤ 1) ∨ (a ≥ 8)

-- Theorem statement
theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_a a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l901_90119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l901_90157

/-- A function that generates all valid five-digit numbers under the given conditions -/
def validNumbers : List (Fin 5 → Fin 5) := sorry

/-- A predicate that checks if a five-digit number satisfies all conditions -/
def isValid (n : Fin 5 → Fin 5) : Bool :=
  let adjacent := ∃ i, i < 4 ∧ ((n i = 1 ∧ n (i+1) = 3) ∨ 
                                (n i = 1 ∧ n (i+1) = 5) ∨ 
                                (n i = 3 ∧ n (i+1) = 5) ∨
                                (n i = 3 ∧ n (i+1) = 1) ∨
                                (n i = 5 ∧ n (i+1) = 1) ∨
                                (n i = 5 ∧ n (i+1) = 3))
  (n 0 ≠ 4 && n 4 ≠ 4) &&  -- 4 is not in first or last position
  (∀ i j, i ≠ j → n i ≠ n j) &&  -- no repeating digits
  adjacent  -- exactly two among 1, 3, 5 are adjacent

theorem count_valid_numbers : 
  (validNumbers.filter isValid).length = 48 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l901_90157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_alpha_l901_90104

theorem tan_value_for_special_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : (Real.cos α) ^ 2 - Real.sin α = 1 / 4) : 
  Real.tan α = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_alpha_l901_90104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_fence_problem_l901_90122

theorem yard_fence_problem (area : ℝ) (fence_length : ℝ) :
  area = 480 ∧ fence_length = 64 →
  ∃ (length width : ℝ),
    length * width = area ∧
    2 * width + length = fence_length ∧
    length = 40 ∧ width = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yard_fence_problem_l901_90122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l901_90167

/-- Given a function f: ℝ → ℝ with a tangent line y = (1/2)x + 2 at the point (1, f(1)),
    prove that f(1) + f'(1) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : (fun x => (1/2)*x + 2) = fun x => f 1 + (deriv f) 1 * (x - 1)) :
  f 1 + (deriv f) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l901_90167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_mod_1000_l901_90144

open BigOperators

def binomial_sum (n : ℕ) : ℕ := ∑ k in Finset.filter (fun i => i % 4 = 0) (Finset.range (n + 1)), n.choose k

theorem binomial_sum_mod_1000 : binomial_sum 2011 % 1000 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_mod_1000_l901_90144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_interest_rate_l901_90124

theorem true_discount_interest_rate 
  (face_value : ℝ) 
  (time_months : ℝ) 
  (true_discount : ℝ) 
  (h1 : face_value = 1400)
  (h2 : time_months = 9)
  (h3 : true_discount = 150) :
  let time_years := time_months / 12
  let interest_rate := (true_discount * (100 + 100 * time_years)) / (face_value * time_years)
  interest_rate = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_interest_rate_l901_90124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90128

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- Theorem statement
theorem function_properties :
  (f (-4) = -2) ∧
  (f 3 = 6) ∧
  (f (f (-2)) = 0) ∧
  (f 0 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l901_90128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_still_water_speed_theorem_l901_90183

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  downstream : ℝ

/-- Calculates the speed in still water given upstream and downstream speeds -/
noncomputable def stillWaterSpeed (s : RowingSpeed) : ℝ := (s.upstream + s.downstream) / 2

/-- Theorem stating that for the given upstream and downstream speeds, 
    the still water speed is 45 kmph -/
theorem still_water_speed_theorem (s : RowingSpeed) 
  (h1 : s.upstream = 25) (h2 : s.downstream = 65) : 
  stillWaterSpeed s = 45 := by
  unfold stillWaterSpeed
  rw [h1, h2]
  norm_num
  -- The proof is complete, so we don't need 'sorry' here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_still_water_speed_theorem_l901_90183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bruce_time_l901_90181

/-- Represents a competitor in the sponsored run -/
structure Competitor where
  name : String
  walkSpeed : ℝ
  runSpeed : ℝ

/-- Calculates the time taken by a competitor who walks half the distance -/
noncomputable def timeForHalfDistanceWalker (c : Competitor) (totalDistance : ℝ) : ℝ :=
  (totalDistance / 2) / c.walkSpeed + (totalDistance / 2) / c.runSpeed

/-- Calculates the distance covered by a competitor who walks half the time -/
noncomputable def distanceForHalfTimeWalker (c : Competitor) (totalTime : ℝ) : ℝ :=
  c.walkSpeed * (totalTime / 2) + c.runSpeed * (totalTime / 2)

theorem bruce_time (angus bruce : Competitor) (angusTime : ℝ) :
  angus.walkSpeed = 3 →
  angus.runSpeed = 6 →
  bruce.walkSpeed = 3 →
  bruce.runSpeed = 6 →
  angusTime = 2/3 →
  timeForHalfDistanceWalker bruce (distanceForHalfTimeWalker angus angusTime) = 3/4 := by
  sorry

#eval (3/4 * 60 : Float)  -- Converts to minutes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bruce_time_l901_90181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l901_90192

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (1, -1) to the line 3x - 4y + 3 = 0 is 2 -/
theorem distance_point_to_line_example : distance_point_to_line 1 (-1) 3 (-4) 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l901_90192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l901_90178

-- Define the function
noncomputable def f (x : ℝ) : ℝ := -2 / (x - 1)

-- Define the domain
def domain (x : ℝ) : Prop := (0 ≤ x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2)

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain x ∧ f x = y) ↔ (y ≤ -2 ∨ y ≥ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l901_90178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_per_child_l901_90199

/-- Given 4.0 oranges and 3.0 children, prove that the number of oranges per child
    is approximately equal to 1.33. -/
theorem oranges_per_child (total_oranges : ℝ) (num_children : ℝ) 
    (h1 : total_oranges = 4) (h2 : num_children = 3) :
  abs ((total_oranges / num_children) - 1.33333333) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_per_child_l901_90199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_dilation_l901_90158

-- Define the dilation transformation
noncomputable def dilation (x y : ℝ) : ℝ × ℝ :=
  (1/2 * x, 3 * y)

-- Define the original sine curve
noncomputable def original_curve (x : ℝ) : ℝ :=
  Real.sin x

-- Define the transformed curve
noncomputable def transformed_curve (x' : ℝ) : ℝ :=
  3 * Real.sin (2 * x')

-- Theorem statement
theorem sine_curve_dilation :
  ∀ x y, y = original_curve x →
    let (x', y') := dilation x y
    y' = transformed_curve x' :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_curve_dilation_l901_90158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_4x10_impossible_5x10_l901_90173

/-- Represents a rectangular table with crosses -/
structure CrossTable (m n : ℕ) where
  crosses : Fin m → Fin n → Bool

/-- Checks if a row contains an odd number of crosses -/
def rowIsOdd (t : CrossTable m n) (i : Fin m) : Prop :=
  Odd (Finset.card (Finset.filter (fun j => t.crosses i j) (Finset.univ : Finset (Fin n))))

/-- Checks if a column contains an odd number of crosses -/
def colIsOdd (t : CrossTable m n) (j : Fin n) : Prop :=
  Odd (Finset.card (Finset.filter (fun i => t.crosses i j) (Finset.univ : Finset (Fin m))))

/-- Checks if all rows and columns contain an odd number of crosses -/
def allOdd (t : CrossTable m n) : Prop :=
  (∀ i, rowIsOdd t i) ∧ (∀ j, colIsOdd t j)

/-- The total number of crosses in the table -/
def totalCrosses (t : CrossTable m n) : ℕ :=
  Finset.card (Finset.filter (fun p => t.crosses p.1 p.2) (Finset.univ : Finset (Fin m × Fin n)))

theorem max_crosses_4x10 :
  ∃ (t : CrossTable 4 10), allOdd t ∧ totalCrosses t = 30 ∧
  ∀ (t' : CrossTable 4 10), allOdd t' → totalCrosses t' ≤ 30 := by sorry

theorem impossible_5x10 :
  ¬∃ (t : CrossTable 5 10), allOdd t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_crosses_4x10_impossible_5x10_l901_90173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l901_90196

noncomputable def projection_matrix (v : Fin 2 → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_v := Real.sqrt (v 0 ^ 2 + v 1 ^ 2)
  let u := fun i => v i / norm_v
  Matrix.of fun i j => u i * u j

noncomputable def Q : Matrix (Fin 2) (Fin 2) ℝ :=
  projection_matrix fun i => if i = 0 then 4 else -1

theorem projection_matrix_not_invertible :
  ¬(IsUnit Q) ∧ Q⁻¹ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_matrix_not_invertible_l901_90196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l901_90187

/-- Represents the total number of marbles Marcy has -/
def n : ℕ := 56

/-- The number of red marbles Marcy has -/
def red_marbles : ℕ := n / 2

/-- The number of blue marbles Marcy has -/
def blue_marbles : ℕ := 3 * n / 8

/-- The number of green marbles Marcy has -/
def green_marbles : ℕ := 7

/-- The number of yellow marbles Marcy has -/
def yellow_marbles : ℤ := n - (red_marbles + blue_marbles + green_marbles)

theorem smallest_yellow_marbles :
  (∀ k : ℕ, k < n → k % 8 = 0 → (k : ℤ) - ((k / 2) + (3 * k / 8) + 7) < 0) ∧
  yellow_marbles = 0 := by
  sorry

#check smallest_yellow_marbles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_yellow_marbles_l901_90187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_product_invariance_l901_90186

/-- A parabola in a 2D plane. -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → ℝ

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A diameter of a parabola. -/
def Diameter := Line

/-- Distance between a point and a line. -/
noncomputable def distance_point_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Intersection points of a line and a parabola. -/
noncomputable def intersection_points (l : Line) (p : Parabola) : Set (ℝ × ℝ) := sorry

/-- Diameter passing through a point on a parabola. -/
noncomputable def diameter_through_point (para : Parabola) (p : ℝ × ℝ) : Diameter := sorry

/-- Statement of the theorem. -/
theorem parabola_distance_product_invariance (para : Parabola) (P : ℝ × ℝ) 
  (l₁ l₂ : Line) : 
  let pts₁ := intersection_points l₁ para
  let pts₂ := intersection_points l₂ para
  ∀ (A B : ℝ × ℝ), A ∈ pts₁ → B ∈ pts₁ → A ≠ B →
  ∀ (C D : ℝ × ℝ), C ∈ pts₂ → D ∈ pts₂ → C ≠ D →
  let d₁₁ := diameter_through_point para A
  let d₁₂ := diameter_through_point para B
  let d₂₁ := diameter_through_point para C
  let d₂₂ := diameter_through_point para D
  (distance_point_line P d₁₁ * distance_point_line P d₁₂) = 
  (distance_point_line P d₂₁ * distance_point_line P d₂₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_product_invariance_l901_90186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l901_90191

/-- The time it takes to complete a work given two workers' individual completion times and their initial collaboration period -/
noncomputable def total_completion_time (a_time b_time collab_time : ℝ) : ℝ :=
  let combined_rate := 1 / a_time + 1 / b_time
  let work_done_together := combined_rate * collab_time
  let remaining_work := 1 - work_done_together
  collab_time + remaining_work * a_time

/-- Theorem stating that under the given conditions, the total completion time is 12 days -/
theorem work_completion_time :
  total_completion_time 15 10 2 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l901_90191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_volume_calculation_l901_90132

noncomputable section

/-- The volume of a sphere with radius r -/
def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The volume of the region between two concentric spheres, excluding a cylinder -/
def regionVolume (r₁ r₂ r_cyl h_cyl : ℝ) : ℝ :=
  sphereVolume r₂ - sphereVolume r₁ - cylinderVolume r_cyl h_cyl

theorem region_volume_calculation :
  regionVolume 5 8 5 5 = 391 * Real.pi := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_volume_calculation_l901_90132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_equation_result_l901_90190

theorem prime_equation_result (p q : ℕ) (hp : Prime p) (hq : Prime q) (heq : p + 5 * q = 97) :
  (p : ℤ)^2 - q = -15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_equation_result_l901_90190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l901_90113

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- Sum of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  (∀ n : ℕ, n > 0 → arithmetic_seq d (n + 1) > arithmetic_seq d n) →
  (∀ n : ℕ, n > 0 → geometric_seq r (n + 1) > geometric_seq r n) →
  c_seq d r (k - 1) = 150 →
  c_seq d r (k + 1) = 1500 →
  c_seq d r k = 263 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l901_90113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l901_90159

/-- Represents the problem of determining maximum allowable deviation and original liquid densities in a mixture scenario. -/
theorem mixture_problem
  (a b c d e f g h i j : ℝ)
  (h_positive : 0 < h)
  (i_positive : 0 < i)
  (g_positive : 0 < g)
  (e_positive : 0 < e)
  (mixture_ratio : a + b = e * (c + d))
  (first_mixture_density : ∃ x₁ x₂, (a + b) / ((a / x₁) + (b / x₂)) = g * (1000 - h) / 1000)
  (second_mixture_density : ∃ x₁ x₂ y, (c + d) / ((c / x₁) + (d / x₂)) = g * (1000 + y + i) / 1000)
  (final_mixture_density : ∃ y,
    ((a + b - f - j) + (c + d - f)) /
    (((a + b - f - j) / (g * (1000 - h) / 1000)) + ((c + d - f) / (g * (1000 + y + i) / 1000)))
    = g * (1000 + y) / 1000) :
  ∃ y x₁ x₂,
    y = Real.sqrt ((500 + (i - h) / 2)^2 + ((c + d - f) / (a + b - f - j)) * (1000 - h) * i) -
        (500 + (i + h) / 2) ∧
    x₁ = (g / 1000) * ((a * d - b * c) * (1000000 + 1000 * (y + i - h) - h * (y + i))) /
         (1000 * (a * d - b * c) + (y + i) * (a + b) * d + h * b * (c + d)) ∧
    x₂ = (g / 1000) * ((a * d - b * c) * (1000000 + 1000 * (y + i - h) - h * (y + i))) /
         (1000 * (a * d - b * c) + (y + i) * (c + d) * a + h * c * (a + b)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_problem_l901_90159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_size_l901_90141

def is_valid_subset (M : Finset ℕ) : Prop :=
  ∀ x y z, x ∈ M → y ∈ M → z ∈ M → x < y ∧ y < z → ¬(x + y ∣ z)

theorem largest_valid_subset_size :
  ∃ (M : Finset ℕ), M ⊆ Finset.range 2007 ∧ is_valid_subset M ∧ M.card = 1004 ∧
  ∀ (N : Finset ℕ), N ⊆ Finset.range 2007 → is_valid_subset N → N.card ≤ 1004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_size_l901_90141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_equals_sqrt_two_l901_90179

theorem sec_seven_pi_fourth_equals_sqrt_two : 1 / Real.cos (7 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_seven_pi_fourth_equals_sqrt_two_l901_90179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_years_to_reach_target_l901_90149

/-- The population growth function -/
noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℝ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- Theorem stating that it takes at least 5 years for the population to reach 2.1 million -/
theorem population_growth_years_to_reach_target : ∃ (min_years : ℕ), 
  (∀ (y : ℝ), y < min_years → population_growth 200 0.01 y < 210) ∧
  population_growth 200 0.01 (min_years : ℝ) ≥ 210 ∧
  min_years = 5 := by
  sorry

#check population_growth_years_to_reach_target

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_years_to_reach_target_l901_90149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l901_90129

-- Define the inverse proportion function
noncomputable def f (x : ℝ) : ℝ := 3 / x

-- Define the y-coordinates
noncomputable def y₁ : ℝ := f (-2)
noncomputable def y₂ : ℝ := f (-1)
noncomputable def y₃ : ℝ := f 1

-- Theorem statement
theorem inverse_proportion_inequality : y₂ < y₁ ∧ y₁ < y₃ := by
  -- Expand the definitions
  unfold y₁ y₂ y₃ f
  -- Simplify the fractions
  simp
  -- Split the conjunction
  apply And.intro
  -- Prove y₂ < y₁
  · norm_num
  -- Prove y₁ < y₃
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l901_90129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_index_l901_90156

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: For an arithmetic sequence where S₁₁ > 0 and S₁₂ < 0,
    the maximum value of n for which Sn is the largest is 6 -/
theorem max_sum_index (seq : ArithmeticSequence) 
    (h1 : S seq 11 > 0) (h2 : S seq 12 < 0) :
    ∃ (n : ℕ), n = 6 ∧ ∀ (m : ℕ), S seq m ≤ S seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_index_l901_90156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l901_90185

/-- The time taken for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  total_length / relative_speed

/-- Theorem: The time taken for two trains to cross each other is approximately 15 seconds -/
theorem train_crossing_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_crossing_time 200 300 70 50 - 15| < ε := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 200 300 70 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l901_90185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_at_seven_yuan_price_for_max_sales_at_target_profit_l901_90125

/-- Represents the souvenir sales scenario -/
structure SouvenirSales where
  cost_price : ℚ
  initial_price : ℚ
  initial_sales : ℚ
  price_decrease : ℚ
  sales_increase : ℚ

/-- Calculate the new sales volume based on price decrease -/
def new_sales_volume (s : SouvenirSales) (price_drop : ℚ) : ℚ :=
  s.initial_sales + (price_drop / s.price_decrease) * s.sales_increase

/-- Calculate the profit based on new price and sales volume -/
def profit (s : SouvenirSales) (new_price : ℚ) : ℚ :=
  let price_drop := s.initial_price - new_price
  let volume := new_sales_volume s price_drop
  (new_price - s.cost_price) * volume

/-- Theorem for the first part of the problem -/
theorem sales_at_seven_yuan (s : SouvenirSales) 
  (h1 : s.cost_price = 5)
  (h2 : s.initial_price = 8)
  (h3 : s.initial_sales = 100)
  (h4 : s.price_decrease = 1/10)
  (h5 : s.sales_increase = 10) :
  new_sales_volume s 1 = 200 := by
  sorry

/-- Theorem for the second part of the problem -/
theorem price_for_max_sales_at_target_profit (s : SouvenirSales) 
  (h1 : s.cost_price = 5)
  (h2 : s.initial_price = 8)
  (h3 : s.initial_sales = 100)
  (h4 : s.price_decrease = 1/10)
  (h5 : s.sales_increase = 10) :
  ∃ x : ℚ, profit s x = 375 ∧ 
  ∀ y : ℚ, profit s y = 375 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_at_seven_yuan_price_for_max_sales_at_target_profit_l901_90125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l901_90161

theorem log_inequality (a b c : ℝ) 
  (ha : a = Real.log 5 / Real.log 2)
  (hb : b = Real.log 11 / Real.log 3)
  (hc : c = 5 / 2) : 
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l901_90161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l901_90147

theorem imaginary_part_of_z : ∃ z : ℂ, z.im = 1 := by
  let z := Complex.I * (1 + 2 * Complex.I)
  have h : z.im = 1 := by
    -- Proof steps would go here
    sorry
  exact ⟨z, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l901_90147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_spending_l901_90134

/-- The total cost of Teresa's purchases at the local shop -/
def total_cost (sandwich_price : ℚ) (sandwich_quantity : ℕ) (salami_price : ℚ) 
  (olive_price_per_pound : ℚ) (olive_quantity : ℚ) (feta_price_per_pound : ℚ) 
  (feta_quantity : ℚ) (bread_price : ℚ) : ℚ :=
  sandwich_price * sandwich_quantity + 
  salami_price + 
  3 * salami_price + 
  olive_price_per_pound * olive_quantity + 
  feta_price_per_pound * feta_quantity + 
  bread_price

/-- Theorem stating that Teresa's total spending is $40.00 -/
theorem teresa_spending : 
  total_cost (7.75) 2 4 10 (1/4) 8 (1/2) 2 = 40 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teresa_spending_l901_90134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_equals_sixteen_ninths_l901_90135

noncomputable def A : ℝ :=
  ∫ x in (3/4)..(4/3), (2*x^2 + x + 1) / (x^3 + x^2 + x + 1)

theorem integral_exp_equals_sixteen_ninths : Real.exp A = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_equals_sixteen_ninths_l901_90135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l901_90143

/-- Represents a rectangular parallelepiped -/
structure RectangularParallelepiped where
  a : ℝ  -- length
  b : ℝ  -- width
  c : ℝ  -- height
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- 
Represents the angle between the plane through BD₁ parallel to AC
and the base plane ABCD of the parallelepiped.
This function is not implemented and serves as a placeholder.
-/
noncomputable def angle_between_plane_and_base (p : RectangularParallelepiped) : ℝ :=
  sorry

/-- 
Given a rectangular parallelepiped with dimensions a, b, and c,
the angle between a plane through BD₁ parallel to AC and the base plane ABCD
is equal to arctan(√(a² + b²) / c).
-/
theorem angle_between_planes (p : RectangularParallelepiped) :
  ∃ θ : ℝ, θ = Real.arctan (Real.sqrt (p.a^2 + p.b^2) / p.c) ∧
    θ = angle_between_plane_and_base p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l901_90143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_grids_l901_90176

/-- A 2x2 grid filled with 0's and 1's -/
def Grid := Fin 2 → Fin 2 → Fin 2

/-- Check if a row has no duplicate entries -/
def row_no_duplicates (g : Grid) (i : Fin 2) : Prop :=
  g i 0 ≠ g i 1

/-- Check if a column has no duplicate entries -/
def col_no_duplicates (g : Grid) (j : Fin 2) : Prop :=
  g 0 j ≠ g 1 j

/-- Check if the grid has no duplicate entries in any row or column -/
def valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 2, row_no_duplicates g i) ∧
  (∀ j : Fin 2, col_no_duplicates g j)

/-- The set of all valid grids -/
def valid_grids : Set Grid :=
  {g : Grid | valid_grid g}

/-- Prove that valid_grids is finite -/
instance : Fintype valid_grids := by
  sorry

theorem two_valid_grids : Fintype.card valid_grids = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_valid_grids_l901_90176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_500x_l901_90189

/-- Represents the setup of a cube illuminated by a point source -/
structure CubeShadow where
  cube_edge : ℝ
  light_height : ℝ
  shadow_area : ℝ

/-- Calculates the value of x based on the cube shadow setup -/
noncomputable def calculate_x (setup : CubeShadow) : ℝ :=
  Real.sqrt (setup.shadow_area + setup.cube_edge ^ 2) - setup.cube_edge

/-- Theorem stating the largest integer not exceeding 500x for the given setup -/
theorem largest_integer_below_500x (setup : CubeShadow) 
  (h1 : setup.cube_edge = 2)
  (h2 : setup.light_height = 3)
  (h3 : setup.shadow_area = 98) :
  ⌊500 * calculate_x setup⌋ = 4050 := by
  sorry

-- Uncomment the following line if you want to evaluate the result
-- #eval ⌊500 * (Real.sqrt 102 - 2)⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_integer_below_500x_l901_90189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_add_two_floor_sum_condition_floor_double_not_always_double_l901_90111

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

-- Statement I
theorem floor_add_two (x : ℝ) : floor (x + 2) = floor x + 2 := by sorry

-- Statement II
theorem floor_sum_condition (x y : ℝ) (h : frac x + frac y ≥ 1) :
  floor (x + y) = floor x + floor y + 1 := by sorry

-- Statement III (counterexample)
theorem floor_double_not_always_double :
  ∃ x : ℝ, floor (2 * x) ≠ 2 * floor x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_add_two_floor_sum_condition_floor_double_not_always_double_l901_90111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l901_90103

theorem tan_double_angle_special_case (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π/2) π) 
  (h2 : Real.sin (2*α) = -Real.sin α) : 
  Real.tan (2*α) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l901_90103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pizza_slices_l901_90138

/-- Given an 8-slice pizza, if a friend eats 2 slices and James eats half of the remaining slices,
    then James eats 3 slices. -/
theorem james_pizza_slices : ℕ := by
  let total_slices : ℕ := 8
  let friend_eats : ℕ := 2
  let remaining_slices : ℕ := total_slices - friend_eats
  let james_eats : ℕ := remaining_slices / 2
  have : james_eats = 3 := by
    -- Proof steps would go here
    sorry
  exact james_eats

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pizza_slices_l901_90138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l901_90130

-- Define the function as noncomputable due to dependency on Real.log
noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 2*x)

-- Theorem statement
theorem f_monotone_decreasing :
  StrictMonoOn f (Set.Iio 1) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l901_90130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_point_inequality_l901_90160

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point P
def Point := ℝ × ℝ

-- Define a function to calculate distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a point is inside a triangle
def isInside (P : Point) (t : Triangle) : Prop := sorry

-- Define functions to calculate side lengths of the triangle
noncomputable def a (t : Triangle) : ℝ := distance t.B t.C
noncomputable def b (t : Triangle) : ℝ := distance t.C t.A
noncomputable def c (t : Triangle) : ℝ := distance t.A t.B

-- State the theorem
theorem interior_point_inequality (t : Triangle) (P : Point) 
  (h : isInside P t) : 
  (distance P t.A) / (a t) + (distance P t.B) / (b t) + (distance P t.C) / (c t) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_point_inequality_l901_90160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l901_90194

-- Define the circle
def circle_m (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 3 = 0

-- Define the symmetry line
def symmetry_line (m x y : ℝ) : Prop := m * x + y + 1 = 0

-- Define the condition for point P
def point_p_condition (x y : ℝ) : Prop :=
  ((x + 2)^2 + y^2) * ((x - 2)^2 + y^2) = (x^2 + y^2)^2

-- Main theorem
theorem circle_properties :
  -- 1. Circle equation
  (∀ x y : ℝ, tangent_line x y → circle_m x y) ∧
  -- 2. Symmetry line property
  (∃ m : ℝ, ∀ x y : ℝ, circle_m x y ∧ symmetry_line m x y → m = 1) ∧
  -- 3. Range of dot product
  (∀ x y : ℝ, circle_m x y → point_p_condition x y →
    -2 ≤ (x^2 - 4 + y^2) ∧ (x^2 - 4 + y^2) < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l901_90194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l901_90150

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ k : ℝ, ∀ x y : ℝ, y = (Real.sqrt 3/2) * x → x^2/a^2 - y^2/b^2 = 1) →
  (∃ f : ℝ × ℝ, f.1 = -(Real.sqrt 7) ∧ f ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1}) →
  a^2 = 4 ∧ b^2 = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l901_90150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_element_set_l901_90165

/-- An arithmetic sequence with common difference 2π/3 -/
noncomputable def arithmetic_sequence (a₁ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (2 * Real.pi / 3) * (n - 1)

/-- The set S of cosines of the arithmetic sequence -/
noncomputable def S (a₁ : ℝ) : Set ℝ :=
  {x | ∃ n : ℕ, n ≥ 1 ∧ x = Real.cos (arithmetic_sequence a₁ n)}

theorem product_of_two_element_set (a₁ : ℝ) :
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a ≠ b) → 
  (∃ a b : ℝ, S a₁ = {a, b} ∧ a * b = -1/2) :=
by
  intro h
  rcases h with ⟨a, b, hS, hab⟩
  use a, b
  constructor
  · exact hS
  · sorry  -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_two_element_set_l901_90165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_zero_one_intersection_function_g_is_zero_one_intersection_function_l901_90188

/-- Definition of a [0,1] intersection function -/
def is_zero_one_intersection_function (f : ℝ → ℝ) : Prop :=
  (Set.Icc 0 1 : Set ℝ) = {x | ∃ y, f y = x} ∩ {x | ∃ y, f x = y}

/-- Function f(x) = √(1-x) -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x)

/-- Function g(x) = 2√x - x -/
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt x - x

/-- Theorem: f is a [0,1] intersection function -/
theorem f_is_zero_one_intersection_function :
  is_zero_one_intersection_function f := by sorry

/-- Theorem: g is a [0,1] intersection function -/
theorem g_is_zero_one_intersection_function :
  is_zero_one_intersection_function g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_zero_one_intersection_function_g_is_zero_one_intersection_function_l901_90188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_redistribution_theorem_l901_90164

/-- Represents the income distribution in the city -/
structure IncomeDistribution where
  poor : ℝ
  middle : ℝ
  rich : ℝ

/-- Calculates the tax rate for the rich group -/
noncomputable def taxRate (x : ℝ) : ℝ :=
  x^2 / 5 + x

/-- Applies the tax and redistributes income -/
noncomputable def applyTax (initial : IncomeDistribution) : IncomeDistribution :=
  let x := initial.poor
  let taxAmount := initial.rich * (taxRate x / 100)
  { poor := initial.poor + 2/3 * taxAmount,
    middle := initial.middle + 1/3 * taxAmount,
    rich := initial.rich - taxAmount }

/-- The main theorem to prove -/
theorem income_redistribution_theorem :
  ∀ x : ℝ,
  x > 0 →
  x + 3*x + 6*x = 100 →
  let initial := { poor := x, middle := 3*x, rich := 6*x : IncomeDistribution }
  let final := applyTax initial
  final.poor = 22 ∧ final.middle = 36 ∧ final.rich = 42 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_redistribution_theorem_l901_90164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_unique_l901_90197

def is_arithmetic_sequence (seq : List Nat) : Prop :=
  seq.length > 1 ∧ ∃ d, ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = d

def is_valid_sampling (n : Nat) (k : Nat) (seq : List Nat) : Prop :=
  seq.length = k ∧
  (∀ x ∈ seq, x ≤ n) ∧
  is_arithmetic_sequence seq ∧
  ∃ d, d * k = n ∧ ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = d

theorem systematic_sampling_unique :
  is_valid_sampling 20 4 [3, 8, 13, 18] ∧
  ∀ seq : List Nat, is_valid_sampling 20 4 seq → seq = [3, 8, 13, 18] := by
  sorry

#check systematic_sampling_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_unique_l901_90197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_path_length_l901_90195

theorem inscribed_circle_path_length 
  (a b c : ℝ) 
  (h_triangle : a = 6 ∧ b = 8 ∧ c = 10) 
  (h_radius : ℝ) 
  (h_radius_val : h_radius = 2) : 
  (a - 2 * h_radius) + (b - 2 * h_radius) + (c - 2 * h_radius) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_path_length_l901_90195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chalk_packages_sold_l901_90107

/-- Represents the quantity of chalk packages of each type -/
structure ChalkQuantities where
  regular : ℕ
  unusual : ℕ
  excellent : ℕ

/-- Represents the ratio of chalk packages of each type -/
structure ChalkRatio where
  regular : ℕ
  unusual : ℕ
  excellent : ℕ

def initial_ratio : ChalkRatio := ⟨2, 3, 6⟩
def final_ratio : ChalkRatio := ⟨5, 7, 4⟩

def max_added_packages : ℕ := 100
def excellent_sold_percentage : ℚ := 40 / 100

theorem chalk_packages_sold
  (initial : ChalkQuantities)
  (final : ChalkQuantities)
  (added_regular : ℕ)
  (added_unusual : ℕ)
  (h1 : initial.regular * final_ratio.regular = final.regular * initial_ratio.regular)
  (h2 : initial.unusual * final_ratio.unusual = final.unusual * initial_ratio.unusual)
  (h3 : initial.excellent * final_ratio.excellent = final.excellent * initial_ratio.excellent)
  (h4 : final.regular = initial.regular + added_regular)
  (h5 : final.unusual = initial.unusual + added_unusual)
  (h6 : added_regular + added_unusual ≤ max_added_packages)
  (h7 : final.excellent = initial.excellent - Nat.floor (excellent_sold_percentage * ↑initial.excellent))
  : Nat.floor (excellent_sold_percentage * ↑initial.excellent) = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chalk_packages_sold_l901_90107
