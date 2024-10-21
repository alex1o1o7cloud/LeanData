import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1098_109820

open Real Set

noncomputable def f (x : ℝ) : ℝ := x * sin x + cos x

theorem f_increasing_on_interval :
  ∀ x ∈ Ioo (3 * π / 2) (5 * π / 2), 
    ∃ ε > 0, ∀ y ∈ Ioo x (x + ε), f y > f x :=
by
  intro x hx
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1098_109820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l1098_109872

/-- An arithmetic sequence with common difference d -/
structure ArithmeticSequence (α : Type*) [AddGroup α] where
  a₀ : α
  d : α

variable {α : Type*} [LinearOrderedField α]

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.a (seq : ArithmeticSequence α) (n : ℕ) : α :=
  seq.a₀ + n • seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.S (seq : ArithmeticSequence α) (n : ℕ) : α :=
  n * (2 * seq.a₀ + (n - 1) • seq.d) / 2

/-- The main theorem -/
theorem arithmetic_sequence_max_sum 
    (seq : ArithmeticSequence α) 
    (hd : seq.d < 0) 
    (h_eq : seq.a 4 ^ 2 = seq.a 12 ^ 2) : 
    ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ 
    ∀ m : ℕ, seq.S m ≤ seq.S n :=
  sorry

#check arithmetic_sequence_max_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l1098_109872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_partition_l1098_109893

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

def has_arithmetic_progression (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a < b ∧ b < c ∧ is_arithmetic_progression a b c

def valid_partition (n : ℕ) : Prop :=
  ∃ A B : Set ℕ,
    A ∪ B = Finset.range (n + 1) ∧
    A ∩ B = ∅ ∧
    ¬ has_arithmetic_progression A ∧
    ¬ has_arithmetic_progression B

theorem largest_valid_partition : 
  (∀ m, m ≤ 8 → valid_partition m) ∧ 
  (∀ m, m > 8 → ¬ valid_partition m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_partition_l1098_109893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_likeable_count_l1098_109803

/-- A number is Julia-likeable if it's divisible by 4 -/
def is_julia_likeable (n : ℕ) : Prop := n % 4 = 0

/-- The set of two-digit numbers (from 00 to 99) -/
def two_digit_numbers : Set ℕ := {n | 0 ≤ n ∧ n ≤ 99}

/-- The count of unique two-digit numbers that are divisible by 4 -/
def count_julia_likeable_two_digit : ℕ :=
  Finset.card (Finset.filter (λ n => n % 4 = 0) (Finset.range 100))

theorem julia_likeable_count : count_julia_likeable_two_digit = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_likeable_count_l1098_109803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1098_109823

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4*y*f x) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_solution_l1098_109823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_conveyance_percent_l1098_109860

/-- Rohan's monthly expenses and savings --/
structure RohanFinances where
  salary : ℚ
  food_percent : ℚ
  rent_percent : ℚ
  entertainment_percent : ℚ
  savings : ℚ

/-- Calculate the percentage of salary spent on conveyance --/
def conveyance_percent (r : RohanFinances) : ℚ :=
  100 - r.food_percent - r.rent_percent - r.entertainment_percent - (r.savings / r.salary * 100)

/-- Theorem stating that Rohan spends 10% of his salary on conveyance --/
theorem rohan_conveyance_percent :
  let r : RohanFinances := {
    salary := 7500,
    food_percent := 40,
    rent_percent := 20,
    entertainment_percent := 10,
    savings := 1500
  }
  conveyance_percent r = 10 := by
  -- Proof goes here
  sorry

#eval conveyance_percent {
  salary := 7500,
  food_percent := 40,
  rent_percent := 20,
  entertainment_percent := 10,
  savings := 1500
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rohan_conveyance_percent_l1098_109860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_in_ranges_l1098_109873

theorem even_count_in_ranges : 
  let S : Finset ℕ := {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16}
  (S.filter (fun x => Even x)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_count_in_ranges_l1098_109873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_minimum_on_interval_l1098_109836

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) - 2

noncomputable def α : ℝ := Real.arcsin (-1/2)

theorem f_at_alpha : f α = -3 := by sorry

theorem f_minimum_on_interval :
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -2 ∧ ∃ y ∈ Set.Icc 0 (Real.pi / 2), f y = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_alpha_f_minimum_on_interval_l1098_109836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1098_109827

def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 9)

def is_on_segment (C A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

def twice_distance (C A B : ℝ × ℝ) : Prop :=
  ‖C - A‖ = 2 * ‖C - B‖

theorem point_C_coordinates :
  ∃ C : ℝ × ℝ, is_on_segment C A B ∧ twice_distance C A B ∧ C = (2, 19/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coordinates_l1098_109827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_1_range_of_a_when_f_less_than_g_l1098_109888

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -(1 + a) / x

-- Theorem 1: Minimum value of f when a = 1
theorem min_value_f_when_a_1 :
  ∀ x > 0, f 1 x ≥ 1 := by
  sorry

-- Theorem 2: Range of a when f(x₀) < g(x₀) for some x₀ ∈ [1,e]
theorem range_of_a_when_f_less_than_g (a : ℝ) :
  (a > 0) →
  (∃ x₀ : ℝ, x₀ ∈ [1, exp 1] ∧ f a x₀ < g a x₀) →
  a > (exp 2 + 1) / (exp 1 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_when_a_1_range_of_a_when_f_less_than_g_l1098_109888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1098_109846

noncomputable def series_term (n : ℕ) : ℝ := (3 ^ n) / ((3 ^ (2 ^ n)) + 2)

theorem series_sum : ∑' (n : ℕ), series_term n = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l1098_109846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_l1098_109874

noncomputable def cost_price : ℝ := 200
noncomputable def market_price : ℝ := 300
noncomputable def min_profit_margin : ℝ := 0.05

noncomputable def discount_rate (x : ℝ) : ℝ := x / 100

noncomputable def selling_price (x : ℝ) : ℝ := market_price * (1 - discount_rate x)

noncomputable def profit (x : ℝ) : ℝ := selling_price x - cost_price

noncomputable def profit_margin (x : ℝ) : ℝ := profit x / cost_price

theorem max_discount_rate :
  ∀ x : ℝ, x ≤ 30 ∧ x > 0 →
    profit_margin x ≥ min_profit_margin ∧
    ∀ y : ℝ, y > x → profit_margin y < min_profit_margin :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_discount_rate_l1098_109874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l1098_109817

theorem integer_solutions_count : 
  ∃ (S : Finset Int), 
    (∀ y : Int, y ∈ S ↔ (-3 * y ≥ y + 9 ∧ -2 * y ≤ 18 ∧ -4 * y ≥ 2 * y + 20)) ∧ 
    Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l1098_109817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_to_height_ratio_l1098_109871

/-- A regular triangular pyramid is a pyramid with an equilateral triangle as its base and all edges equal -/
structure RegularTriangularPyramid where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- The inscribed sphere of a regular triangular pyramid -/
structure InscribedSphere (p : RegularTriangularPyramid) where
  radius : ℝ
  radius_pos : radius > 0

/-- The height of a regular triangular pyramid -/
noncomputable def pyramid_height (p : RegularTriangularPyramid) : ℝ :=
  p.edge_length * Real.sqrt 6 / 3

/-- The theorem stating that the ratio of the radius of the inscribed sphere
    to the height of a regular triangular pyramid is 1:4 -/
theorem inscribed_sphere_radius_to_height_ratio
  (p : RegularTriangularPyramid) (s : InscribedSphere p) :
  s.radius / pyramid_height p = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_to_height_ratio_l1098_109871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_eight_l1098_109821

def fibonacci_like_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like_sequence n + fibonacci_like_sequence (n + 1)

theorem sixth_term_is_eight :
  fibonacci_like_sequence 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_eight_l1098_109821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_l1098_109840

theorem divisors_of_factorial : 
  (Finset.filter (λ d ↦ d > Nat.factorial 9) (Nat.divisors (Nat.factorial 10))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_l1098_109840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_behind_yoongi_l1098_109859

/-- Given a line of students, calculates the number of students behind a specific position. -/
def studentsBehind (total : ℕ) (position : ℕ) : ℕ :=
  total - position

theorem students_behind_yoongi (total : ℕ) (jungkookPosition yoongiPosition : ℕ) :
  total = 20 →
  jungkookPosition = 1 →
  yoongiPosition = jungkookPosition + 1 →
  studentsBehind total yoongiPosition = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_behind_yoongi_l1098_109859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_l1098_109807

-- Define the constants
noncomputable def path_distance : ℝ := 300
noncomputable def jenny_speed : ℝ := 2
noncomputable def kenny_speed : ℝ := 4
noncomputable def building_diameter : ℝ := 150
noncomputable def building_distance : ℝ := 200
noncomputable def initial_distance : ℝ := 250

-- Define the positions of Jenny and Kenny at time t
noncomputable def jenny_position (t : ℝ) : ℝ × ℝ := (-150 + 2*t, 150)
noncomputable def kenny_position (t : ℝ) : ℝ × ℝ := (-150 + 4*t, -150)

-- Define the equation of the line connecting Jenny and Kenny at time t
noncomputable def line_equation (t : ℝ) (x : ℝ) : ℝ := 
  -150 / t * x + 300 - 7500 / t

-- Define the equation of the second building (circle)
def building_equation (x y : ℝ) : Prop :=
  (x - 200)^2 + y^2 = 75^2

-- State the theorem
theorem visibility_time :
  ∃ t : ℝ, t = 130/3 ∧
  ∃ x y : ℝ, 
    building_equation x y ∧
    y = line_equation t x ∧
    (x - 200) / y = -150 / t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visibility_time_l1098_109807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1098_109811

-- Define the function f(x) = e^x + 4x - 3
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

-- Theorem statement
theorem root_in_interval :
  ∃ x ∈ Set.Ioo 0 (1/2), f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1098_109811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shaded_is_four_ninths_l1098_109804

/-- Represents a triangle in the diagram -/
structure Triangle where
  shaded : Bool

/-- The set of all triangles in the diagram -/
def triangles : Finset Triangle := sorry

/-- The number of triangles is greater than 5 -/
axiom more_than_five : 5 < triangles.card

/-- Some triangles are shaded -/
axiom some_shaded : ∃ t ∈ triangles, t.shaded

/-- The number of shaded triangles -/
def num_shaded : Nat := (triangles.filter (·.shaded)).card

/-- The probability of selecting a shaded triangle -/
noncomputable def prob_shaded : ℚ := num_shaded / triangles.card

/-- The probability of selecting a shaded triangle is 4/9 -/
theorem prob_shaded_is_four_ninths : prob_shaded = 4/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_shaded_is_four_ninths_l1098_109804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l1098_109880

-- Define the power function
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(2*m + 1)

-- Define the derivative of y with respect to x
noncomputable def y_deriv (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * (2*m + 1) * x^(2*m)

theorem decreasing_power_function (m : ℝ) :
  (∀ x > 0, (y_deriv m x < 0)) ↔ m = -1 := by
  sorry

#check decreasing_power_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_l1098_109880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1098_109833

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ 
    {x : ℝ | f x > 0} ≠ Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_set_l1098_109833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1098_109878

/-- The circle with equation x^2 + 2x + y^2 - 3 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 - 3 = 0

/-- The line with equation y = x + 3 -/
def line_equation (x y : ℝ) : Prop := y = x + 3

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  (|a*x + b*y + c|) / Real.sqrt (a^2 + b^2)

theorem distance_from_circle_center_to_line :
  distance_point_to_line (circle_center.1) (circle_center.2) (-1) 1 (-3) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1098_109878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bons_win_probability_l1098_109835

/-- The probability of rolling a six -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a six -/
def prob_not_six : ℚ := 1 - prob_six

/-- The probability that B. Bons wins the game -/
noncomputable def prob_bons_wins : ℚ :=
  (prob_not_six * prob_six) / (1 - prob_not_six * prob_not_six)

theorem bons_win_probability :
  prob_bons_wins = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bons_win_probability_l1098_109835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_area_l1098_109861

/-- Calculates the original area of a bathroom given its extended dimensions -/
theorem bathroom_area (original_width : ℝ) (extension : ℝ) (new_area : ℝ) : 
  original_width = 8 →
  extension = 2 →
  new_area = 140 →
  ∃ (original_length : ℝ),
    original_length * (original_width + 2 * extension) = new_area ∧
    abs (original_length * original_width - 93.36) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_area_l1098_109861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_unique_solution_l1098_109800

theorem range_of_a_for_unique_solution :
  ∀ (a : ℝ),
  (∀ x, x ∈ Set.Icc 0 1 → ∃! y, y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y - a = 0) ↔
  a ∈ Set.Ioo (1 + Real.exp (-1)) (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_unique_solution_l1098_109800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_mean_difference_l1098_109822

/-- The number of families --/
def num_families : ℕ := 1200

/-- The correct highest income --/
noncomputable def correct_highest_income : ℝ := 110000

/-- The incorrectly recorded highest income --/
noncomputable def incorrect_highest_income : ℝ := 220000

/-- The sum of all other incomes (excluding the highest) --/
noncomputable def sum_other_incomes : ℝ := 0  -- We don't know this value, but it's not needed for the proof

/-- The mean of the actual data --/
noncomputable def actual_mean : ℝ := (sum_other_incomes + correct_highest_income) / num_families

/-- The mean of the incorrect data --/
noncomputable def incorrect_mean : ℝ := (sum_other_incomes + incorrect_highest_income) / num_families

/-- The theorem to be proved --/
theorem income_mean_difference :
  ∃ (ε : ℝ), abs (incorrect_mean - actual_mean - 91.67) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_mean_difference_l1098_109822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1098_109826

/-- Hyperbola C₁ -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Parabola C₂ -/
structure Parabola where
  p : ℝ
  h_pos_p : p > 0

/-- The focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- The directrix of a parabola -/
noncomputable def directrix (c : Parabola) : ℝ := -c.p / 2

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = h.b / h.a * x ∨ y = -h.b / h.a * x}

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Equilateral triangle property -/
def IsEquilateralTriangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (c : Parabola)
  (h_focus : focus c ∈ {(x, y) | x^2 / h.a^2 - y^2 / h.b^2 = 1})
  (h_triangle : ∃ (t : Set (ℝ × ℝ)), t ⊆ asymptotes h ∧ 
    (directrix c, 0) ∈ t ∧ IsEquilateralTriangle t) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1098_109826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1098_109886

def point1 : ℝ × ℝ := (1, -3)
def point2 : ℝ × ℝ := (-4, 7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1098_109886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1098_109828

noncomputable def line_l (m t : ℝ) : ℝ × ℝ := (m - Real.sqrt 2 * t, Real.sqrt 5 + Real.sqrt 2 * t)

def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2 * Real.sqrt 5 * y

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_circle_intersection (m : ℝ) :
  (∃ t1 t2 : ℝ, 
    circle_C (line_l m t1).1 (line_l m t1).2 ∧
    circle_C (line_l m t2).1 (line_l m t2).2 ∧
    distance (line_l m t1) (line_l m t2) = Real.sqrt 2 ∧
    m > 0) →
  m = 3 ∧
  (let p := (m, Real.sqrt 5);
   let a := line_l m t1;
   let b := line_l m t2;
   distance p a + distance p b = 3 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1098_109828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_prime_factors_l1098_109809

theorem existence_of_three_prime_factors (n : ℕ) : 
  ∃ N : ℕ, ∀ n ≥ N, ∃ k ∈ (Set.range (fun i => n + i) ∩ Set.Icc n (n + 9)), 
    (Finsupp.support (Nat.factorization k)).card ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_three_prime_factors_l1098_109809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l1098_109837

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem f_negative (a : ℝ) (h : f a = Real.sqrt 3) : f (-a) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_l1098_109837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_P_complement_Q_l1098_109816

def P : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x : ℝ | x^2 ≥ 4}

theorem union_P_complement_Q : P ∪ (Set.univ \ Q) = Set.Ioc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_P_complement_Q_l1098_109816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1098_109834

noncomputable section

-- Define f as a function from ℝ to ℝ
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 6}

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (2 * x) / (x - 2)

-- Define the domain of g
def dom_g : Set ℝ := {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)}

-- Theorem statement
theorem domain_of_g (x : ℝ) : x ∈ dom_g ↔ (2 * x ∈ dom_f ∧ x ≠ 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1098_109834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_integer_point_l1098_109841

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define a convex quadrilateral with integer-coordinate vertices
structure ConvexQuadrilateral where
  P : IntPoint
  Q : IntPoint
  R : IntPoint
  S : IntPoint
  is_convex : Bool  -- Assume this property is true for our quadrilateral

-- Define the diagonal intersection point
noncomputable def diagonalIntersection (quad : ConvexQuadrilateral) : IntPoint := sorry

-- Define a function to check if a point is inside or on the boundary of a triangle
def isInTriangle (A B C : IntPoint) (P : IntPoint) : Bool := sorry

-- Define a function to calculate the angle between two points
noncomputable def angle (A B C : IntPoint) : ℝ := sorry

-- Main theorem
theorem quadrilateral_integer_point 
  (quad : ConvexQuadrilateral) 
  (h_angle_sum : angle quad.P quad.Q quad.R + angle quad.S quad.Q quad.P < 180) :
  ∃ (M : IntPoint), M ≠ quad.P ∧ M ≠ quad.Q ∧ 
    isInTriangle quad.P quad.Q (diagonalIntersection quad) M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_integer_point_l1098_109841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1098_109885

def U : Set ℕ := {1,2,3,4,5,6,7,8}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | 1 ≤ x ∧ x ≤ 5}

def C : Set ℕ := {x ∈ U | 2 < x ∧ x < 9}

theorem set_operations :
  (A ∪ (B ∩ C) = {1,2,3,4,5}) ∧
  ((U \ B) ∪ (U \ C) = {1,2,6,7,8}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1098_109885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_quadratic_l1098_109832

theorem negation_of_existential_quadratic :
  (¬ ∃ x : ℝ, x^2 - x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existential_quadratic_l1098_109832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1098_109818

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (3*x^3 + 4*x^2 + 6*x) / ((x^2 + 2)*(x^2 + 2*x + 2))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := log (abs (x^2 + 2)) + (1/2) * log (abs (x^2 + 2*x + 2)) - arctan (x + 1)

-- State the theorem
theorem integral_equality (x : ℝ) : deriv F x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l1098_109818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_book_price_is_nine_l1098_109868

/-- Calculates the highest possible whole dollar price per book given the conditions --/
def highest_book_price (total_budget : ℕ) (num_books : ℕ) (entry_fee : ℕ) (tax_rate : ℚ) : ℕ :=
  let remaining_budget : ℚ := (total_budget - entry_fee : ℚ)
  let pre_tax_budget : ℚ := remaining_budget / (1 + tax_rate)
  let max_price_per_book : ℚ := pre_tax_budget / num_books
  (Int.floor max_price_per_book).toNat

/-- Theorem stating that the highest possible whole dollar price per book is 9 --/
theorem highest_book_price_is_nine :
  highest_book_price 200 20 5 (8/100) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_highest_book_price_is_nine_l1098_109868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_target_sum_equals_60_l1098_109831

/-- Triangle ABC with vertices A(0,0), B(8,0), and C(1,7), and point P(3,4) -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (1, 7)
def P : ℝ × ℝ := (3, 4)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances from P to A, B, and C -/
noncomputable def sum_distances : ℝ :=
  distance P A + distance P B + distance P C

/-- Theorem: The sum of distances from P to A, B, and C is 5 + √41 + √13 -/
theorem sum_distances_equals_target : sum_distances = 5 + Real.sqrt 41 + Real.sqrt 13 := by
  sorry

/-- The sum m + n + p + q where the distances are expressed as m√p + n√q -/
def result : Nat := 60

/-- Theorem: The sum m + n + p + q equals 60 -/
theorem sum_equals_60 : result = 60 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_equals_target_sum_equals_60_l1098_109831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l1098_109842

/-- The atomic mass of aluminum in g/mol -/
noncomputable def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of sulfur in g/mol -/
noncomputable def atomic_mass_S : ℝ := 32.07

/-- The atomic mass of oxygen in g/mol -/
noncomputable def atomic_mass_O : ℝ := 16.00

/-- The molar mass of Al₂(SO₄)₃ in g/mol -/
noncomputable def molar_mass_Al2SO4_3 : ℝ :=
  2 * atomic_mass_Al + 3 * atomic_mass_S + 12 * atomic_mass_O

/-- The mass of Al in Al₂(SO₄)₃ in g/mol -/
noncomputable def mass_Al_in_Al2SO4_3 : ℝ := 2 * atomic_mass_Al

/-- The mass percentage of Al in Al₂(SO₄)₃ -/
noncomputable def mass_percentage_Al : ℝ :=
  (mass_Al_in_Al2SO4_3 / molar_mass_Al2SO4_3) * 100

theorem mass_percentage_Al_approx :
  abs (mass_percentage_Al - 15.77) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l1098_109842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_case1_part_ii_case2_part_ii_case3_l1098_109808

noncomputable section

-- Define the function f(x) = ax - ln x
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Part I
theorem part_i (a : ℝ) (x : ℝ) (h1 : a ≤ 1) (h2 : x ≥ 1) :
  x^2 ≥ f a x := by
  sorry

-- Part II
theorem part_ii_case1 (a : ℝ) (h : a > Real.exp (-1)) :
  ∀ x > 0, f a x > 0 := by
  sorry

theorem part_ii_case2 (a : ℝ) (h : a = Real.exp (-1)) :
  ∃! x, x > 0 ∧ f a x = 0 := by
  sorry

theorem part_ii_case3 (a : ℝ) (h1 : a > 0) (h2 : a < Real.exp (-1)) :
  ∃ x1 x2, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ 
  (∀ x, x > 0 → f a x = 0 → x = x1 ∨ x = x2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_i_part_ii_case1_part_ii_case2_part_ii_case3_l1098_109808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_shopping_l1098_109864

def children_home (total : ℕ) (toddlers : ℕ) (newborn_cost : ℝ) (discount : ℝ) : Prop :=
  ∃ (teenagers school_age newborns : ℕ),
    teenagers = 5 * toddlers ∧
    school_age = 3 * newborns ∧
    total = teenagers + school_age + toddlers + newborns ∧
    newborns = 1 ∧
    newborn_cost * (1 - discount) * (newborns : ℝ) = 35

theorem diana_shopping :
  children_home 40 6 50 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diana_shopping_l1098_109864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1098_109857

/-- A triangle with two sides of lengths 3 and 5, and the third side being a root of x^2 - 5x + 6 = 0 that satisfies the triangle inequality, has a perimeter of 11. -/
theorem triangle_perimeter : ∃ x : ℝ, x^2 - 5*x + 6 = 0 ∧ 
                                      x > 0 ∧
                                      x < 3 + 5 ∧ 
                                      3 < x + 5 ∧ 
                                      5 < x + 3 ∧
                                      x + 3 + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1098_109857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1098_109895

/-- The time taken for two trains to cross each other completely -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) : ℝ :=
  (2 * train_length) / (2 * train_speed * (1000 / 3600))

/-- Theorem stating that the train crossing time is approximately 4.5 seconds
    under the given conditions -/
theorem train_crossing_theorem (ε : ℝ) (hε : ε > 0) :
  ∃ (δ : ℝ), δ > 0 ∧ 
  ∀ (train_length train_speed : ℝ),
  train_length > 0 ∧ train_speed > 0 →
  |train_length - 100| < δ ∧ |train_speed - 80| < δ →
  |train_crossing_time train_length train_speed - 4.5| < ε := by
  sorry

#check train_crossing_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1098_109895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1098_109858

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≤ 2}
def N : Set ℝ := {x | Real.exp (x * Real.log 2) > 1}

-- Define the complement of N relative to ℝ
def notN : Set ℝ := {x | x ≤ 0}

-- State the theorem
theorem set_operations :
  (M ∩ N = {x | 0 < x ∧ x ≤ 3}) ∧
  (M ∪ notN = {x | x ≤ 3}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1098_109858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_pi_over_4_l1098_109855

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_angle_pi_over_4 (a b : ℝ × ℝ) :
  Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 2 →
  Real.sqrt (b.1^2 + b.2^2) = 2 →
  (a.1 - b.1) * a.1 + (a.2 - b.2) * a.2 = 0 →
  angle_between_vectors a b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_pi_over_4_l1098_109855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_isosceles_trapezoid_l1098_109887

/-- An isosceles trapezoid -/
structure IsoscelesTrapezoid where
  /-- The longer base of the trapezoid -/
  longer_base : Real
  /-- The shorter base of the trapezoid -/
  shorter_base : Real
  /-- The height of the trapezoid -/
  height : Real
  /-- The condition that makes it an isosceles trapezoid -/
  isosceles : longer_base > shorter_base

/-- The solid formed by rotating an isosceles trapezoid around its longer base -/
structure RotatedSolid (t : IsoscelesTrapezoid) where
  cylinder : Unit
  cones : Fin 2 → Unit

/-- Theorem stating that rotating an isosceles trapezoid around its longer base
    forms a solid consisting of one cylinder and two cones -/
theorem rotate_isosceles_trapezoid (t : IsoscelesTrapezoid) :
  ∃ (s : RotatedSolid t), True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_isosceles_trapezoid_l1098_109887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1098_109866

theorem parallel_vectors_x_value 
  (e₁ e₂ : ℝ × ℝ) 
  (h_non_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂) 
  (x : ℝ) 
  (a b : ℝ × ℝ) 
  (h_a : a = x • e₁ - 3 • e₂) 
  (h_b : b = 2 • e₁ + e₂) 
  (h_parallel : ∃ (l : ℝ), a = l • b) : 
  x = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l1098_109866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1098_109853

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 3)

def C : Set (ℝ × ℝ) := {(x, y) | y = f x}

theorem function_properties :
  (∀ x y : ℝ, (x, y) ∈ C → (11 * Real.pi / 6 - x, y) ∈ C) ∧ 
  (∀ x ∈ Set.Ioo (5 * Real.pi / 12) (11 * Real.pi / 12), 
   ∀ y ∈ Set.Ioo (5 * Real.pi / 12) (11 * Real.pi / 12), 
   x < y → f x > f y) ∧
  (∃ x : ℝ, f (x + Real.pi / 3) ≠ 3 * Real.sin (2 * x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1098_109853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1098_109854

/-- Given a point M(x₀, y₀) on the circle x² + y² = r² (r > 0),
    prove that the line x₀x + y₀y = r² is tangent to the circle. -/
theorem line_tangent_to_circle (r x₀ y₀ : ℝ) (hr : r > 0) (hm : x₀^2 + y₀^2 = r^2) :
  let line := {p : ℝ × ℝ | x₀ * p.1 + y₀ * p.2 = r^2}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  ∃! p, p ∈ line ∩ circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1098_109854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1098_109812

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from (1, 1) to the line x-y+1=0 is √2/2 -/
theorem distance_point_to_line :
  distancePointToLine 1 1 1 (-1) 1 = Real.sqrt 2 / 2 := by
  sorry

#check distance_point_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l1098_109812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_c_is_smallest_l1098_109829

/-- The smallest positive real number satisfying the inequality -/
noncomputable def c : ℝ := 1/2

/-- The inequality holds for all nonnegative real numbers x and y -/
theorem inequality_holds (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  Real.sqrt (x * y) + c * |x^2 - y^2| ≥ (x + y) / 2 := by
  sorry

/-- c is the smallest positive real number satisfying the inequality -/
theorem c_is_smallest (c' : ℝ) (hc' : c' > 0) 
  (h : ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c' * |x^2 - y^2| ≥ (x + y) / 2) :
  c' ≥ c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_c_is_smallest_l1098_109829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_range_l1098_109891

/-- Given a test with scores, prove that the range is 75 -/
theorem test_score_range :
  ∀ (scores : Finset ℕ) (mark_score : ℕ),
    98 ∈ scores →  -- highest score is 98
    46 ∈ scores →  -- Mark's score is 46
    (∀ s ∈ scores, s ≤ 98) →  -- 98 is the highest score
    (∀ s ∈ scores, s ≥ 23) →  -- 23 is the least score (half of Mark's score)
    46 = 2 * (Finset.min scores) →  -- Mark's score is twice the least score
    98 - 23 = 75 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_range_l1098_109891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_divide_f_values_l1098_109875

def is_function_satisfying_condition (f : ℕ+ → ℕ+) : Prop :=
  ∀ a b : ℕ+, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinitely_many_primes_divide_f_values
  (f : ℕ+ → ℕ+)
  (hf : is_function_satisfying_condition f)
  (hf_non_constant : ∃ x y : ℕ+, f x ≠ f y) :
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ c : ℕ+, p ∣ f c} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_divide_f_values_l1098_109875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gerbil_sale_profit_l1098_109876

/-- Calculates the profit from selling gerbils given the specified conditions --/
theorem gerbil_sale_profit
  (initial_stock : ℕ)
  (purchase_price selling_price : ℚ)
  (sales_percentage : ℚ)
  (tax_rate : ℚ)
  (h1 : initial_stock = 450)
  (h2 : purchase_price = 8)
  (h3 : selling_price = 12)
  (h4 : sales_percentage = 35 / 100)
  (h5 : tax_rate = 5 / 100) :
  let sold := ⌊(initial_stock : ℚ) * sales_percentage⌋
  let revenue := (sold : ℚ) * selling_price
  let cost := (sold : ℚ) * purchase_price
  let profit := revenue - cost
  profit = 628 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gerbil_sale_profit_l1098_109876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_calculation_l1098_109806

theorem one_correct_calculation : ∃! n : ℕ, n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  (n = 4 → ∀ (a b : ℝ), a ≠ 0 → 4 * a^3 * b / (-2 * a^2 * b) = -2 * a) ∧
  (n = 1 → ∀ (a b : ℝ), 3 * a + 2 * b ≠ 5 * a * b) ∧
  (n = 2 → ∀ (m n : ℝ), 4 * m^3 * n - 5 * m * n^3 ≠ -m^3 * n) ∧
  (n = 3 → ∀ (x : ℝ), 3 * x^2 * (-2 * x^2) ≠ -6 * x^5) ∧
  (n = 5 → ∀ (a : ℝ), (a^3)^2 ≠ a^5) ∧
  (n = 6 → ∀ (a : ℝ), a ≠ 0 → (-a)^3 / (-a) ≠ -a^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_correct_calculation_l1098_109806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1098_109838

-- Define the points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (-2, 0)

-- Define the trajectory C
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define the point P on trajectory C
def P (x y : ℝ) : Prop := C x y

-- Define the slope product condition
def slope_product (x y : ℝ) : Prop := 
  (y / (x - 2)) * (y / (x + 2)) = -3/4

-- Define the line l
def l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1/2

-- Define the midpoint M of E and F
def M (x₀ y₀ : ℝ) (m : ℝ) : Prop := 
  x₀ = 2 / (3 * m^2 + 4) ∧ 
  y₀ = -3 * m / (2 * (3 * m^2 + 4))

-- Define the slope k of line MA
noncomputable def k (m : ℝ) : ℝ := m / (4 * m^2 + 4)

-- State the theorem
theorem slope_range : 
  ∀ x y m x₀ y₀, 
    P x y → 
    slope_product x y → 
    l m x y → 
    M x₀ y₀ m → 
    -1/8 ≤ k m ∧ k m ≤ 1/8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_l1098_109838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_two_std_dev_below_mean_l1098_109839

/-- Represents a normal distribution of heights -/
structure HeightDistribution where
  mean : ℝ
  std_dev : ℝ

/-- Converts centimeters to inches -/
noncomputable def cm_to_inches (cm : ℝ) : ℝ := cm / 2.54

/-- Calculates the height that is n standard deviations from the mean -/
def height_n_std_dev (d : HeightDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

theorem height_two_std_dev_below_mean (d : HeightDistribution) 
  (h_mean : d.mean = 14.5)
  (h_std_dev : d.std_dev = 1.7) : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (cm_to_inches (height_n_std_dev d 2) - 4.37) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_two_std_dev_below_mean_l1098_109839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_3250_l1098_109865

/-- Represents the grazing details of a milkman -/
structure GrazingDetails where
  cows : Nat
  months : Nat
deriving Inhabited

/-- Calculates the total rent of a field given the grazing details of milkmen -/
def calculateTotalRent (milkmen : List GrazingDetails) (aRent : Nat) : Nat :=
  let totalCowMonths := milkmen.foldl (fun acc x => acc + x.cows * x.months) 0
  let aCowMonths := (milkmen.head!).cows * (milkmen.head!).months
  let rentPerCowMonth := aRent / aCowMonths
  rentPerCowMonth * totalCowMonths

/-- Theorem stating that the total rent of the field is 3250 -/
theorem total_rent_is_3250 :
  let milkmen := [
    GrazingDetails.mk 24 3,  -- A
    GrazingDetails.mk 10 5,  -- B
    GrazingDetails.mk 35 4,  -- C
    GrazingDetails.mk 21 3   -- D
  ]
  calculateTotalRent milkmen 720 = 3250 := by
  sorry

#eval calculateTotalRent [
  GrazingDetails.mk 24 3,
  GrazingDetails.mk 10 5,
  GrazingDetails.mk 35 4,
  GrazingDetails.mk 21 3
] 720

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_rent_is_3250_l1098_109865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1098_109852

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/3
  | n+1 => (-1)^(n+1) * 2 * my_sequence n

theorem fifth_term_value : my_sequence 4 = 16/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1098_109852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_formula_l1098_109881

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  -- Area of the trapezoid
  area : ℝ
  -- Acute angle at the base
  base_angle : ℝ
  -- Assumption that the base angle is acute (0 < α < π/2)
  acute_angle : 0 < base_angle ∧ base_angle < Real.pi / 2
  -- Assumption that the area is positive
  positive_area : 0 < area

/-- The midline of a circumscribed isosceles trapezoid -/
noncomputable def midline (t : CircumscribedIsoscelesTrapezoid) : ℝ :=
  Real.sqrt (t.area / Real.sin t.base_angle)

/-- Theorem: The midline of a circumscribed isosceles trapezoid is √(S/sin(α)) -/
theorem midline_formula (t : CircumscribedIsoscelesTrapezoid) :
  midline t = Real.sqrt (t.area / Real.sin t.base_angle) := by
  -- Proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_formula_l1098_109881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_arithmetic_sequence_l1098_109843

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_of_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem max_sum_of_arithmetic_sequence :
  let a₁ : ℚ := 5
  let d : ℚ := -5/7
  ∀ k : ℕ, k ≠ 0 →
    (sum_of_arithmetic_sequence a₁ d 7 ≥ sum_of_arithmetic_sequence a₁ d k ∧
     sum_of_arithmetic_sequence a₁ d 8 ≥ sum_of_arithmetic_sequence a₁ d k) ∧
    (∃ n : ℕ, n ∈ ({7, 8} : Set ℕ) ∧ sum_of_arithmetic_sequence a₁ d n > sum_of_arithmetic_sequence a₁ d k) ∨
    k ∈ ({7, 8} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_arithmetic_sequence_l1098_109843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_third_l1098_109884

theorem tan_a_pi_third (a : ℝ) (h : 9 = Real.exp (a * Real.log 3)) : 
  Real.tan (a * π / 3) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_pi_third_l1098_109884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_c_l1098_109830

/-- The smallest possible value of c given the conditions -/
noncomputable def smallest_c : ℝ := (5 + Real.sqrt 5) / 2

theorem smallest_possible_c (b c : ℝ) 
  (h1 : 1 < b) (h2 : b < c)
  (h3 : ¬ (1 + b > c ∧ 1 + c > b ∧ b + c > 1))
  (h4 : ¬ (1/c + 1/b > 1 ∧ 1/c + 1 > 1/b ∧ 1/b + 1 > 1/c)) : 
  c ≥ smallest_c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_c_l1098_109830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_floor_N_div_3_l1098_109899

/-- Given a bag with N balls of three colors (red, white, and blue),
    if the probability of picking one ball of each color when drawing
    three balls without replacement is greater than 23%,
    then the maximum value of ⌊N/3⌋ is 29. -/
theorem max_floor_N_div_3 (N : ℕ) (red white blue : ℕ) :
  red + white + blue = N →
  (red * white * blue : ℚ) / ((N * (N - 1) * (N - 2) : ℚ) / 6) > 23 / 100 →
  ⌊(N : ℚ) / 3⌋ ≤ 29 ∧ ∃ (N' : ℕ) (red' white' blue' : ℕ),
    red' + white' + blue' = N' ∧
    (red' * white' * blue' : ℚ) / ((N' * (N' - 1) * (N' - 2) : ℚ) / 6) > 23 / 100 ∧
    ⌊(N' : ℚ) / 3⌋ = 29 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_floor_N_div_3_l1098_109899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1098_109847

noncomputable section

-- Define the vertices of triangle ABC
def B : ℝ × ℝ := (0, -1)
def C : ℝ × ℝ := (0, 1)

-- Define the trajectory E of vertex A
def E (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1 ∧ x ≠ 0

-- Define point F
def F : ℝ × ℝ := (Real.sqrt 2, 0)

-- Define the centroid P and circumcenter Q
def P (A : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
def Q (A : ℝ × ℝ) : ℝ × ℝ := (A.1 / 3, 0)

-- Define the area of quadrilateral A1A2B1B2
def area (A1 A2 B1 B2 : ℝ × ℝ) : ℝ := sorry

-- Define the line MN
def MN (M N : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem triangle_ABC_properties :
  ∀ A : ℝ × ℝ,
  E A.1 A.2 →
  (∃ l1 l2 : Set (ℝ × ℝ),
    -- l1 and l2 are perpendicular lines passing through F
    (∀ x y : ℝ, (x, y) ∈ l1 ↔ sorry) ∧
    (∀ x y : ℝ, (x, y) ∈ l2 ↔ sorry) ∧
    (∀ x y : ℝ, (x, y) ∈ l1 ∩ l2 ↔ x = F.1 ∧ y = F.2) ∧
    -- A1B1 and A2B2 are chords formed by intersections of l1, l2 with E
    (∃ A1 B1 A2 B2 : ℝ × ℝ,
      E A1.1 A1.2 ∧ E B1.1 B1.2 ∧ E A2.1 A2.2 ∧ E B2.1 B2.2 ∧
      (A1 ∈ l1 ∨ A1 ∈ l2) ∧ (B1 ∈ l1 ∨ B1 ∈ l2) ∧
      (A2 ∈ l1 ∨ A2 ∈ l2) ∧ (B2 ∈ l1 ∨ B2 ∈ l2) ∧
      -- M and N are midpoints of A1B1 and A2B2
      let M := ((A1.1 + B1.1) / 2, (A1.2 + B1.2) / 2);
      let N := ((A2.1 + B2.1) / 2, (A2.2 + B2.2) / 2);
      -- Minimum area of A1A2B1B2 is 3/2
      (∀ A1' B1' A2' B2' : ℝ × ℝ,
        E A1'.1 A1'.2 ∧ E B1'.1 B1'.2 ∧ E A2'.1 A2'.2 ∧ E B2'.1 B2'.2 →
        (A1' ∈ l1 ∨ A1' ∈ l2) ∧ (B1' ∈ l1 ∨ B1' ∈ l2) ∧
        (A2' ∈ l1 ∨ A2' ∈ l2) ∧ (B2' ∈ l1 ∨ B2' ∈ l2) →
        area A1' A2' B1' B2' ≥ 3/2) ∧
      -- Line MN always passes through (3√2/4, 0)
      (3 * Real.sqrt 2 / 4, 0) ∈ MN M N)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l1098_109847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1098_109802

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (4, 4)

-- Define the function to calculate the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Define the vertical line x = a
def verticalLine (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = a}

-- Theorem statement
theorem equal_area_division :
  ∃ a : ℝ, a = 2 ∧
  triangleArea A B (a, 4) = triangleArea (a, 4) B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1098_109802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coloring_schemes_l1098_109894

/-- The number of ways to color n sectors of a circle using 5 colors, 
    such that no two adjacent sectors have the same color. -/
def coloringSchemes (n : ℕ) : ℤ :=
  4^n + 4 * (-1:ℤ)^n

/-- The actual number of valid colorings for n sectors -/
noncomputable def number_of_valid_colorings (n : ℕ) : ℤ :=
  sorry

/-- Theorem stating the number of coloring schemes for n sectors (n ≥ 4) -/
theorem correct_coloring_schemes (n : ℕ) (h : n ≥ 4) :
  coloringSchemes n = number_of_valid_colorings n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coloring_schemes_l1098_109894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_m_l1098_109805

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (lineX : ℝ) : ℝ :=
  |p.x - lineX|

/-- The focus point F(4, 0) -/
def F : Point := ⟨4, 0⟩

/-- The theorem stating the trajectory of point M -/
theorem trajectory_of_point_m (M : Point) :
  distance M F = distanceToVerticalLine M (-4) →
  M.y^2 = 16 * M.x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_point_m_l1098_109805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_bound_l1098_109882

/-- A three-digit number with no zeros in its digits -/
def ThreeDigitNoZero : Type :=
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n / 100 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0) }

/-- The sum of reciprocals of digits for a three-digit number -/
def sumReciprocalsOfDigits (n : ThreeDigitNoZero) : ℚ :=
  1 / (n.val / 100 : ℚ) + 1 / ((n.val / 10) % 10 : ℚ) + 1 / (n.val % 10 : ℚ)

/-- The product of a three-digit number and the sum of reciprocals of its digits -/
def productWithSumReciprocals (n : ThreeDigitNoZero) : ℚ :=
  (n.val : ℚ) * sumReciprocalsOfDigits n

/-- Theorem stating that the maximum value of the product is less than or equal to 1923.222 -/
theorem max_product_bound :
  ∀ n : ThreeDigitNoZero, productWithSumReciprocals n ≤ 1923.222 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_bound_l1098_109882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l1098_109845

/-- The distance from a point (x, y) to the line x + 2y - 10 = 0 --/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x + 2*y - 10| / Real.sqrt 5

/-- The ellipse x²/9 + y²/4 = 1 --/
def on_ellipse (x y : ℝ) : Prop :=
  x^2/9 + y^2/4 = 1

theorem minimum_distance_point :
  ∃ (x y : ℝ),
    on_ellipse x y ∧
    (∀ (x' y' : ℝ), on_ellipse x' y' →
      distance_to_line x y ≤ distance_to_line x' y') ∧
    distance_to_line x y = Real.sqrt 5 ∧
    x = 9/5 ∧ y = 8/5 := by
  sorry

#check minimum_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_distance_point_l1098_109845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1098_109819

theorem solve_exponential_equation (s : ℝ) : (9 : ℝ) = (3 : ℝ)^(2*s + 2) → s = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1098_109819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_amount_l1098_109890

theorem fred_amount (amy ben carla dave elaine fred : ℤ) : 
  amy + ben + carla + dave + elaine + fred = 75 →
  |amy - ben| = 15 →
  |ben - carla| = 10 →
  |carla - dave| = 6 →
  |dave - elaine| = 8 →
  |elaine - fred| = 9 →
  |fred - amy| = 13 →
  fred = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_amount_l1098_109890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1098_109851

-- Define the ellipse and circle
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def my_circle (x y : ℝ) : Prop := x^2 + (y + 2)^2 = 1

-- Define a point P on the ellipse
def P : Type := {p : ℝ × ℝ // ellipse p.1 p.2 ∧ p.2 ≠ -1}

-- Define points M and N on the circle
def M : Type := {m : ℝ × ℝ // my_circle m.1 m.2}
def N : Type := {n : ℝ × ℝ // my_circle n.1 n.2}

-- Define the dot product of CM and CN
def dotProduct (c m n : ℝ × ℝ) : ℝ :=
  (m.1 - c.1) * (n.1 - c.1) + (m.2 - c.2) * (n.2 - c.2)

-- The theorem to prove
theorem min_dot_product (p : P) (m : M) (n : N) :
  ∃ (c : ℝ × ℝ), c = (0, -2) → 
    ∀ (p' : P) (m' : M) (n' : N), dotProduct c m'.val n'.val ≥ -11/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1098_109851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_naturals_12_to_53_l1098_109898

-- Define the start and end of the sequence
def start : ℕ := 12
def finish : ℕ := 53

-- Define the function to calculate the average
def average_of_naturals (a b : ℕ) : ℚ :=
  let n : ℕ := b - a + 1
  let sum : ℕ := n * (a + b) / 2
  (sum : ℚ) / n

-- Theorem statement
theorem average_of_naturals_12_to_53 :
  average_of_naturals start finish = 32.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_naturals_12_to_53_l1098_109898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1098_109889

theorem subset_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (x₀ q : ℝ), {x | ∃ n : ℕ, x = x₀ * q ^ n} ⊆ {x | ∃ n : ℕ, x = a * -n + b}) ↔
  ∃ (p q : ℤ), a / b = p / q := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1098_109889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l1098_109810

def number_list : List ℕ := [34, 37, 39, 41, 43]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factorial (n - 1) + 1) % n == 1

def prime_numbers : List ℕ := number_list.filter is_prime

def arithmetic_mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem arithmetic_mean_of_primes :
  arithmetic_mean prime_numbers = 121 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l1098_109810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_negative_eight_l1098_109870

theorem x_plus_y_equals_negative_eight (x y : ℝ) 
  (h1 : (4 : ℝ)^x = (16 : ℝ)^(y + 2))
  (h2 : (27 : ℝ)^y = (9 : ℝ)^(x - 6)) : 
  x + y = -8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_negative_eight_l1098_109870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1098_109801

/-- The focal length of a hyperbola -/
noncomputable def focal_length (a b c : ℝ) : ℝ := 2 * c

/-- The equation of the hyperbola -/
def hyperbola_equation (x y a b : ℝ) : Prop := x^2 / (4 : ℝ) - y^2 / b^2 = 1

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a b c : ℝ) : ℝ := c / a

theorem hyperbola_focal_length (b : ℝ) (hb : b > 0) :
  ∃ (a c x y : ℝ), 
    a = 2 ∧ 
    hyperbola_equation x y a b ∧
    eccentricity a b c = (Real.sqrt 3 / 3) * b ∧
    focal_length a b c = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l1098_109801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l1098_109844

/-- Given two trains traveling between two points A and B, this theorem proves
    the speed of the slower train given the conditions of the problem. -/
theorem second_train_speed 
  (distance : ℝ) 
  (speed_first : ℝ) 
  (time_diff : ℝ) 
  (h1 : distance = 65)
  (h2 : speed_first = 65)
  (h3 : time_diff = 5) :
  distance / (distance / speed_first + time_diff) = 65 / 6 := by
  sorry

#eval (65 : Float) / 6 -- To display the approximate result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_l1098_109844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_third_quadrant_l1098_109848

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (-α + 3*Real.pi/2)) /
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_value_third_quadrant (α : Real)
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_third_quadrant_l1098_109848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crows_eating_time_l1098_109892

/-- The time for two crows to eat a quarter of the nuts -/
noncomputable def time_to_eat_quarter (N : ℝ) : ℝ :=
  let first_crow_rate := N / (5 * 8)
  let second_crow_rate := N / (3 * 12)
  let combined_rate := first_crow_rate + second_crow_rate
  (1 / 4) * N / combined_rate

theorem crows_eating_time :
  ∀ N : ℝ, N > 0 → time_to_eat_quarter N = 90 / 19 :=
by
  sorry

#eval (90 : ℚ) / 19

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crows_eating_time_l1098_109892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1098_109877

theorem school_students_count : ∃ (total_students : ℕ),
  let blue_percent : ℚ := 45/100
  let red_percent : ℚ := 23/100
  let green_percent : ℚ := 15/100
  let other_count : ℕ := 136
  (blue_percent + red_percent + green_percent + (other_count : ℚ) / total_students = 1) ∧
  total_students = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_students_count_l1098_109877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_five_l1098_109897

theorem three_digit_numbers_divisible_by_five :
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 900)).card + 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_divisible_by_five_l1098_109897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_sqrt2_minus_1_l1098_109849

open Real

/-- The function f(x) = 1/x + 2/(1-x) for 0 < x < 1 -/
noncomputable def f (x : ℝ) : ℝ := 1/x + 2/(1-x)

/-- The theorem stating that f(x) reaches its minimum when x = √2 - 1 -/
theorem f_min_at_sqrt2_minus_1 :
  ∀ x : ℝ, 0 < x → x < 1 → f x ≥ f (sqrt 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_sqrt2_minus_1_l1098_109849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s₁_less_than_s₂_l1098_109867

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Sum of distances from centroid to vertices -/
noncomputable def s₁ (t : Triangle) : ℝ := distance t.G t.A + distance t.G t.B + distance t.G t.C

/-- Perimeter of the triangle -/
noncomputable def s₂ (t : Triangle) : ℝ := distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: s₁ is less than s₂ for any triangle -/
theorem s₁_less_than_s₂ (t : Triangle) : s₁ t < s₂ t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s₁_less_than_s₂_l1098_109867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1098_109896

/-- Given points M(1, 0) and N(-1, 0), and point P(x, y) on the line 2x - y - 1 = 0,
    the minimum value of PM^2 + PN^2 is 2/5, achieved when P has coordinates (1/5, -3/5). -/
theorem min_distance_sum (x y : ℝ) : 
  let M : ℝ × ℝ := (1, 0)
  let N : ℝ × ℝ := (-1, 0)
  let P : ℝ × ℝ := (x, y)
  2 * x - y - 1 = 0 → 
  (∃ (min : ℝ), min = (P.1 - M.1)^2 + (P.2 - M.2)^2 + (P.1 - N.1)^2 + (P.2 - N.2)^2 ∧
    (∀ (x' y' : ℝ), 2 * x' - y' - 1 = 0 → 
      (x' - M.1)^2 + (y' - M.2)^2 + (x' - N.1)^2 + (y' - N.2)^2 ≥ min) ∧
    min = 2/5 ∧ x = 1/5 ∧ y = -3/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1098_109896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_35gon_subset_l1098_109863

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A set of 35 points forming a convex 35-gon -/
def ConvexPolygon (points : Finset Point) : Prop :=
  points.card = 35 ∧ points.card ≥ 3 -- Simplified convexity condition

theorem convex_35gon_subset (points : Finset Point) 
  (h_convex : ConvexPolygon points)
  (h_distance : ∀ p q : Point, p ∈ points → q ∈ points → p ≠ q → distance p q ≥ Real.sqrt 3) :
  ∃ subset : Finset Point, subset ⊆ points ∧ subset.card = 5 ∧
    ∀ p q : Point, p ∈ subset → q ∈ subset → p ≠ q → distance p q ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_35gon_subset_l1098_109863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabelles_card_value_l1098_109856

open Real

theorem isabelles_card_value (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  (∀ y, 0 < y → y < π / 2 → 
    (sin y < 1 ∧ tan y < 1 ∧ 1 / cos y > 1) ∧
    (1 / cos y ≠ sin y ∧ 1 / cos y ≠ tan y) ∧
    (sin y < 1 / cos y ∧ tan y < 1 / cos y)) →
  1 / cos x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabelles_card_value_l1098_109856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_polygons_area_ratio_l1098_109824

-- Define Polygon type
def Polygon := ℝ → ℝ → Prop

-- Define Similar relation
def Similar (P Q : Polygon) : Prop := sorry

-- Define SimilarityRatio function
def SimilarityRatio (P Q : Polygon) : ℝ := sorry

-- Define AreaRatio function
def AreaRatio (P Q : Polygon) : ℝ := sorry

theorem similar_polygons_area_ratio 
  (P Q : Polygon) 
  (h_similar : Similar P Q) 
  (h_ratio : SimilarityRatio P Q = 1 / 5) : 
  AreaRatio P Q = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_polygons_area_ratio_l1098_109824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_approx_l1098_109814

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in degrees -/
  face_angle : ℝ

/-- Volume of tetrahedron ABCD in cm³ -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the tetrahedron with given properties -/
theorem tetrahedron_volume_approx (t : Tetrahedron) 
  (h1 : t.ab_length = 5)
  (h2 : t.abc_area = 18)
  (h3 : t.abd_area = 24)
  (h4 : t.face_angle = 45) :
  abs (tetrahedron_volume t - 43.2) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_approx_l1098_109814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_copper_ratio_l1098_109862

def total_weight : ℕ := 60
def zinc_weight : ℕ := 27

def copper_weight : ℕ := total_weight - zinc_weight

theorem zinc_copper_ratio :
  let gcd := Nat.gcd zinc_weight copper_weight
  (zinc_weight / gcd) = 9 ∧ (copper_weight / gcd) = 11 :=
by
  -- Define gcd locally to avoid ambiguity
  let gcd := Nat.gcd zinc_weight copper_weight
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zinc_copper_ratio_l1098_109862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l1098_109883

/-- Calculates the time needed for a train to pass a platform. -/
noncomputable def time_to_pass_platform (train_length : ℝ) (pole_passing_time : ℝ) (platform_length : ℝ) : ℝ :=
  let train_speed := train_length / pole_passing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem: A train 250 m long that passes a pole in 10 sec will take 60 sec to pass a 1250 m long platform. -/
theorem train_platform_passing_time :
  time_to_pass_platform 250 10 1250 = 60 := by
  -- Unfold the definition of time_to_pass_platform
  unfold time_to_pass_platform
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passing_time_l1098_109883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l1098_109879

/-- The cost price of an article given its selling price and discount information -/
noncomputable def cost_price (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  (selling_price * (1 - discount_rate)) / (1 + profit_rate)

theorem article_cost_price :
  let selling_price : ℝ := 27000
  let discount_rate : ℝ := 0.1
  let profit_rate : ℝ := 0.08
  cost_price selling_price discount_rate profit_rate = 22500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l1098_109879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_petal_count_l1098_109825

def flowers_in_garden : Fin 4 → ℕ
  | 0 => 8  -- lilies
  | 1 => 5  -- tulips
  | 2 => 4  -- roses
  | 3 => 3  -- daisies

def petals_per_flower : Fin 4 → ℕ
  | 0 => 6  -- lilies
  | 1 => 3  -- tulips
  | 2 => 5  -- roses
  | 3 => 12 -- daisies

def flowers_in_bouquet (i : Fin 4) : ℕ := (flowers_in_garden i) / 2

theorem bouquet_petal_count : 
  (Finset.sum (Finset.range 4) (fun i => flowers_in_bouquet i * petals_per_flower i)) = 52 := by
  sorry

#eval Finset.sum (Finset.range 4) (fun i => flowers_in_bouquet i * petals_per_flower i)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouquet_petal_count_l1098_109825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_gravitational_force_l1098_109813

/-- The gravitational force at a given distance from Earth's center -/
noncomputable def gravitational_force (distance : ℝ) : ℝ :=
  14400000000 / (distance ^ 2)

theorem moon_gravitational_force :
  gravitational_force 6000 = 400 →
  gravitational_force 180000 = 4 / 9 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moon_gravitational_force_l1098_109813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_growth_l1098_109815

/-- Sequence of plate lengths -/
def plate_length : ℕ → ℝ := sorry

/-- Sequence of covered lengths -/
def S : ℕ → ℝ := sorry

/-- The first plate length is not 1 -/
axiom first_plate_not_one : plate_length 1 ≠ 1

/-- The covered length after n plates -/
axiom covered_length (n : ℕ) : S (n + 1) = S n + plate_length (n + 1)

/-- The similarity property of plates -/
axiom similarity_property (n : ℕ) : plate_length (n + 1) = 1 / S n

/-- The main theorem: For any real M, there exists n such that S n > M -/
theorem unbounded_growth (M : ℝ) : ∃ n : ℕ, S n > M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unbounded_growth_l1098_109815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_sine_l1098_109869

-- Define the rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the dihedral angle
noncomputable def dihedralAngle (prism : RectangularPrism) : ℝ :=
  Real.arcsin ((prism.a^2 * prism.b * prism.c) / 
    (Real.sqrt (prism.b^2 * prism.c^2 + prism.c^2 * prism.a^2 + prism.a^2 * prism.b^2) *
     Real.sqrt (prism.a^2 * prism.b^2 + 4 * prism.b^2 * prism.c^2 + 4 * prism.c^2 * prism.a^2)))

-- State the theorem
theorem dihedral_angle_sine (prism : RectangularPrism) :
  Real.sin (dihedralAngle prism) = 
    (prism.a^2 * prism.b * prism.c) / 
    (Real.sqrt (prism.b^2 * prism.c^2 + prism.c^2 * prism.a^2 + prism.a^2 * prism.b^2) *
     Real.sqrt (prism.a^2 * prism.b^2 + 4 * prism.b^2 * prism.c^2 + 4 * prism.c^2 * prism.a^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_sine_l1098_109869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l1098_109850

open Real
open BigOperators

-- Define the series term
noncomputable def a (n : ℕ) : ℝ := n / (n^2 + 2*n + 3)

-- State the theorem
theorem series_diverges : ¬ (Summable a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l1098_109850
