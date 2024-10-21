import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l848_84899

open Real

theorem problem_solution (x : ℝ) (p q r : ℕ+) (h1 : (1 + sin x) * (1 + cos x) = 9/4)
  (h2 : (1 - sin x) * (1 - cos x) = (p : ℝ)/(q : ℝ) - Real.sqrt r)
  (h3 : Nat.Coprime p q) : r + p + q = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l848_84899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l848_84884

theorem equation_solution :
  ∃ (x y : ℚ), (Real.sqrt (2 * Real.sqrt 3 - 3) = Real.sqrt (x * Real.sqrt 3) - Real.sqrt (y * Real.sqrt 3)) ∧ 
  x = 3/2 ∧ y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l848_84884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l848_84829

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Theorem: The sum of the infinite geometric series with first term 3 and common ratio -3/4 is 12/7 -/
theorem geometric_series_sum :
  let a : ℝ := 3
  let r : ℝ := -3/4
  geometric_sum a r = 12/7 := by
  -- Proof goes here
  sorry

#eval (12 : ℚ) / 7 -- This will evaluate the fraction 12/7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l848_84829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_on_interval_l848_84807

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x)

theorem function_range_on_interval 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_period : ∀ x, f ω (x + π) = f ω x) :
  Set.range (fun x => f ω x) ∩ Set.Icc (-π/3) (π/6) = Set.Icc (-1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_on_interval_l848_84807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_arrangement_exists_l848_84832

/-- Represents a grid figure --/
structure GridFigure where
  /-- Indicates whether the figure is asymmetric --/
  asymmetric : Bool

/-- Represents an arrangement of grid figures --/
structure Arrangement where
  figures : List GridFigure
  symmetric : Bool

/-- A function that attempts to create a symmetric arrangement from three identical asymmetric figures --/
def create_symmetric_arrangement (figure : GridFigure) : Option Arrangement :=
  sorry

/-- Theorem stating that it's possible to create a symmetric arrangement from three identical asymmetric figures --/
theorem symmetric_arrangement_exists :
  ∀ (figure : GridFigure),
    figure.asymmetric →
    ∃ (arr : Arrangement),
      arr.figures.length = 3 ∧
      (∀ f ∈ arr.figures, f = figure) ∧
      arr.symmetric :=
by
  sorry

#check symmetric_arrangement_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_arrangement_exists_l848_84832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_closed_interval_l848_84869

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 + 2^x

-- State the theorem
theorem f_range_closed_interval :
  {y : ℝ | ∃ x ∈ Set.Icc 0 1, f x = y} = Set.Icc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_closed_interval_l848_84869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_99_terms_l848_84826

def sequenceA (n : ℕ) (a : ℚ) : ℚ :=
  match n with
  | 0 => a
  | n+1 => sequenceA n ((a - 1) / (a + 1))

def sumFirstNTerms (n : ℕ) (a : ℚ) : ℚ :=
  (Finset.range n).sum (λ i => sequenceA i a)

theorem sum_of_99_terms :
  ∃ a₁₀ : ℚ, a₁₀ = 1/3 ∧ sumFirstNTerms 99 (sequenceA 9 a₁₀) = -193/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_99_terms_l848_84826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alireza_winning_strategy_l848_84883

/-- Ω(n) is the largest prime factor of n -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

/-- ω(n) is the smallest prime factor of n -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluate a polynomial at a given value -/
def eval_poly (p : IntPolynomial) (x : ℕ) : ℤ := sorry

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

theorem alireza_winning_strategy :
  ∃ (B : Finset IntPolynomial),
    B.card = 1400 ∧
    ∀ (A : Finset IntPolynomial),
      A ⊆ B →
      A.card = 700 →
      ∃ (n : ℕ),
        (∀ p ∈ A, eval_poly p n = 1) ∧
        ∃ q ∈ B, q ∉ A ∧ ∃ prime : ℕ, is_prime prime ∧ eval_poly q n = prime :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alireza_winning_strategy_l848_84883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_equation_solutions_l848_84811

-- Define the left-hand side of the equation
def f (x : ℝ) : ℝ := abs (abs (abs (abs (abs x - 2) - 2) - 2))

-- Define the right-hand side of the equation
def g (x : ℝ) : ℝ := abs (abs (abs (abs (abs x - 3) - 3) - 3))

-- Theorem statement
theorem nested_absolute_value_equation_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = g x) ∧ (Finset.card S = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_absolute_value_equation_solutions_l848_84811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_l848_84818

/-- The distance from a point (x, x^2) on the parabola y = x^2 to the line 2x - y = 4 -/
noncomputable def distance_to_line (x : ℝ) : ℝ := 
  |x^2 - 2*x + 4| / Real.sqrt 5

/-- The point (1, 1) on the parabola y = x^2 -/
def closest_point : ℝ × ℝ := (1, 1)

theorem closest_point_on_parabola : 
  ∀ x : ℝ, distance_to_line (closest_point.1) ≤ distance_to_line x := by
  sorry

#check closest_point_on_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_l848_84818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_celsius_is_zero_celsius_l848_84848

/-- Represents temperature in Celsius --/
structure Temperature where
  value : ℝ

/-- Converts a temperature to its representation on the scale --/
def temperature_representation (t : Temperature) : ℝ := t.value

/-- Defines the zero point on the temperature scale --/
def zero_celsius : Temperature := ⟨0⟩

instance : OfNat Temperature (nat_lit 0) where
  ofNat := ⟨0⟩

instance : OfNat Temperature (nat_lit 3) where
  ofNat := ⟨3⟩

instance : LT Temperature where
  lt a b := a.value < b.value

/-- States that temperatures below zero are represented as negative numbers --/
axiom below_zero_negative (t : Temperature) :
  t < zero_celsius → temperature_representation t < 0

/-- The temperature we're considering --/
def three_celsius : Temperature := 3

/-- The theorem to be proved --/
theorem three_celsius_is_zero_celsius :
  three_celsius = zero_celsius := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_celsius_is_zero_celsius_l848_84848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l848_84892

noncomputable def g (x : ℝ) : ℝ := Real.log (x^2)

theorem g_is_even : ∀ x : ℝ, x ≠ 0 → g (-x) = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l848_84892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_digit_stability_l848_84853

def a (n : ℕ) : ℕ := Nat.factorial (n + 7) / Nat.factorial (n - 1)

theorem rightmost_digit_stability :
  ∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n ≥ k →
      (a n) % 10 = (a k) % 10) ∧
    k = 1 ∧
    (a k) % 10 = 0 :=
by
  -- The proof goes here
  sorry

#eval a 1 % 10  -- This will evaluate the result for k = 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightmost_digit_stability_l848_84853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l848_84815

/-- Represents a line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Represents a curve in parametric form -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line l -/
noncomputable def line_l : ParametricLine :=
  { x := λ l => l * Real.cos (60 * Real.pi / 180)
  , y := λ l => -1 + l * Real.sin (60 * Real.pi / 180) }

/-- The given curve C -/
def curve_C (a : ℝ) : ParametricCurve :=
  { x := λ t => 2 * a * t^2
  , y := λ t => 2 * a * t }

/-- Predicate to check if a line and curve intersect at two distinct points -/
def intersect_at_two_points (l : ParametricLine) (c : ParametricCurve) : Prop := sorry

theorem intersection_condition (a : ℝ) (h : a ≠ 0) :
  intersect_at_two_points line_l (curve_C a) ↔ a > 0 ∨ a < -2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l848_84815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implication_second_quadrant_range_l848_84836

-- Part 1
theorem complex_real_implication (m : ℝ) : 
  (Complex.I * (m + 1) + (m^2 - 1) : ℂ).im = 0 → m = -1 := by sorry

-- Part 2
theorem second_quadrant_range (x : ℝ) :
  (Real.sqrt x - 1 < 0 ∧ x^2 - 3*x + 2 > 0 ∧ x ≥ 0) → 0 ≤ x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_real_implication_second_quadrant_range_l848_84836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_product_l848_84825

theorem absolute_value_equation_solution_product : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, |x - 4| - 5 = -3) ∧ 
  (∀ x : ℝ, |x - 4| - 5 = -3 → x ∈ S) ∧
  (S.prod id) = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_equation_solution_product_l848_84825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_general_term_l848_84854

def my_sequence (n : ℕ) : ℚ := 1 / (n + 1)

theorem my_sequence_general_term : ∀ n : ℕ, n ≥ 1 → my_sequence n = 1 / (n + 1) := by
  intro n hn
  unfold my_sequence
  rfl

#check my_sequence_general_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_general_term_l848_84854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l848_84816

def leftmost_digit (n : ℕ) : ℕ :=
  n / (10 ^ (Nat.log 10 n))

def remove_leftmost_digit (n : ℕ) : ℕ :=
  n % (10 ^ (Nat.log 10 n))

def satisfies_condition (n : ℕ) : Prop :=
  n > 0 ∧ remove_leftmost_digit n = n / 19

theorem smallest_satisfying_number :
  satisfies_condition 95 ∧ ∀ m : ℕ, m < 95 → ¬satisfies_condition m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_satisfying_number_l848_84816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_on_ellipse_l848_84819

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Area of triangle PF₁F₂ on an ellipse -/
theorem area_triangle_on_ellipse 
  (e : Ellipse) 
  (F₁ F₂ P : Point) 
  (h_on_ellipse : (P.x^2 / e.a^2) + (P.y^2 / e.b^2) = 1)
  (h_foci : distance F₁ F₂ = 2 * Real.sqrt (e.a^2 - e.b^2))
  (h_distance : distance P F₁ = 10) :
  let PF₂ := distance P F₂
  (1/2) * 10 * PF₂ * Real.sqrt (1 - ((PF₂^2 - 100) / (2 * PF₂ * 10))^2) = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_on_ellipse_l848_84819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l848_84893

/-- Given positive integers x and y, where x / y = 96.16 and y = 50.000000000001066,
    the remainder when x is divided by y is approximately 8.00000000041. -/
theorem division_remainder_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.16)
  (h2 : (y : ℝ) = 50.000000000001066) :
  ∃ (ε : ℝ), ε > 0 ∧ |((x : ℝ) % (y : ℝ) - 8.00000000041)| < ε := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l848_84893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_building_steps_l848_84875

/-- The number of steps between each floor in the apartment building -/
def steps_per_floor : ℕ := sorry

/-- The total number of steps from the 1st to the 3rd floor -/
def petya_steps : ℕ := 2 * steps_per_floor

/-- The total number of steps from the 1st to the 6th floor -/
def vasya_steps : ℕ := 5 * steps_per_floor

/-- The total number of steps from the 1st to the 9th floor -/
def kolya_steps : ℕ := 8 * steps_per_floor

theorem apartment_building_steps :
  (steps_per_floor < 22) →
  (petya_steps % 4 = 0) →
  (vasya_steps % 9 = 0) →
  (kolya_steps = 144) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_building_steps_l848_84875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_article_l848_84895

/-- The cost of an article given specific selling prices and gain difference -/
noncomputable def article_cost (selling_price_1 : ℝ) (selling_price_2 : ℝ) (gain_difference_percent : ℝ) : ℝ :=
  let gain := (selling_price_2 - selling_price_1) / (gain_difference_percent / 100)
  selling_price_1 - gain

/-- Theorem stating the cost of the article under given conditions -/
theorem cost_of_article : 
  article_cost 340 350 5 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_of_article_l848_84895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_nerds_count_l848_84806

theorem rainbow_nerds_count (purple yellow green : ℕ) : 
  purple = 10 → 
  yellow = purple + 4 → 
  green = yellow - 2 → 
  purple + yellow + green = 36 := by
  intro h_purple h_yellow h_green
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainbow_nerds_count_l848_84806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_point_l848_84813

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 3

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 3)

/-- The given point -/
def given_point : ℝ × ℝ := (10, 5)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_from_circle_center_to_point :
  distance circle_center given_point = 2 * Real.sqrt 17 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_point_l848_84813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l848_84850

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

-- State the theorem
theorem function_satisfies_conditions :
  -- Condition 1: Smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
   (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → q ≥ p) ∧ p = Real.pi) ∧
  -- Condition 2: Symmetric about x = π/3
  (∀ (x : ℝ), f (Real.pi / 3 + x) = f (Real.pi / 3 - x)) ∧
  -- Condition 3: Increasing on [-π/6, π/3]
  (∀ (x y : ℝ), -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 → f x < f y) ∧
  -- Condition 4: Center of symmetry at (π/12, 0)
  f (Real.pi / 12) = 0 ∧ (∀ (x : ℝ), f (Real.pi / 6 + x) = -f (Real.pi / 6 - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_conditions_l848_84850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_locus_is_circle_ellipse_eccentricity_sqrt2_div_2_hyperbola_eccentricity_l848_84833

-- Define the distance ratio condition
def distanceRatio (lambda : ℝ) (A B M : ℝ × ℝ) : Prop :=
  lambda > 0 ∧ lambda ≠ 1 ∧ dist M A = lambda * dist M B

-- Define a circle
def isCircle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (C : ℝ × ℝ) (r : ℝ), S = {P | dist P C = r}

-- Define an ellipse
def isEllipse (a b : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  a > b ∧ b > 0 ∧ S = {(x, y) | x^2/a^2 + y^2/b^2 = 1}

-- Define a hyperbola
def isHyperbola (a b : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  a > b ∧ b > 0 ∧ S = {(x, y) | x^2/a^2 - y^2/b^2 = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Theorem 1
theorem distance_ratio_locus_is_circle (lambda : ℝ) (A B : ℝ × ℝ) :
  let S := {M | distanceRatio lambda A B M}
  isCircle S :=
sorry

-- Theorem 2
theorem ellipse_eccentricity_sqrt2_div_2 (a b c : ℝ) (S : Set (ℝ × ℝ)) :
  isEllipse a b S → eccentricity a b = Real.sqrt 2 / 2 → b = c :=
sorry

-- Theorem 3
theorem hyperbola_eccentricity (a b theta : ℝ) (S : Set (ℝ × ℝ)) :
  isHyperbola a b S → theta = Real.arctan (b/a) → eccentricity a b = 1 / Real.cos theta :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_locus_is_circle_ellipse_eccentricity_sqrt2_div_2_hyperbola_eccentricity_l848_84833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_tax_rate_is_four_percent_l848_84820

/-- Calculates the tax rate on clothing given spending percentages and tax rates -/
noncomputable def tax_rate_on_clothing (clothing_percent : ℝ) (food_percent : ℝ) (other_percent : ℝ)
  (other_tax_rate : ℝ) (total_tax_rate : ℝ) : ℝ :=
  let total_spending := 100
  let clothing_spending := clothing_percent * total_spending / 100
  let other_spending := other_percent * total_spending / 100
  let other_tax := other_tax_rate * other_spending / 100
  let total_tax := total_tax_rate * total_spending / 100
  ((total_tax - other_tax) / clothing_spending) * 100

/-- The tax rate on clothing is 4% given the specified conditions -/
theorem clothing_tax_rate_is_four_percent :
  tax_rate_on_clothing 50 10 40 8 5.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clothing_tax_rate_is_four_percent_l848_84820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_and_regression_equation_l848_84847

-- Define the data points
def data : List (ℝ × ℝ) := [(1, 2.6), (2, 3.1), (3, 4.5), (4, 6.8), (5, 8.0)]

-- Define the correlation coefficient function
noncomputable def correlation_coefficient (data : List (ℝ × ℝ)) : ℝ :=
  let n := data.length
  let sum_t := data.map Prod.fst |>.sum
  let sum_y := data.map Prod.snd |>.sum
  let sum_ty := data.map (λ (t, y) => t * y) |>.sum
  let sum_t_sq := data.map (λ (t, _) => t ^ 2) |>.sum
  let sum_y_sq := data.map (λ (_, y) => y ^ 2) |>.sum
  let mean_t := sum_t / n
  let mean_y := sum_y / n
  (sum_ty - n * mean_t * mean_y) / 
    ((sum_t_sq - n * mean_t ^ 2) * (sum_y_sq - n * mean_y ^ 2)).sqrt

-- Define the regression coefficients
noncomputable def regression_coefficients (data : List (ℝ × ℝ)) : ℝ × ℝ :=
  let n := data.length
  let sum_t := data.map Prod.fst |>.sum
  let sum_y := data.map Prod.snd |>.sum
  let sum_ty := data.map (λ (t, y) => t * y) |>.sum
  let sum_t_sq := data.map (λ (t, _) => t ^ 2) |>.sum
  let mean_t := sum_t / n
  let mean_y := sum_y / n
  let b := (sum_ty - n * mean_t * mean_y) / (sum_t_sq - n * mean_t ^ 2)
  let a := mean_y - b * mean_t
  (a, b)

-- Theorem statement
theorem correlation_and_regression_equation :
  abs (correlation_coefficient data - 0.98) < 0.01 ∧
  let (a, b) := regression_coefficients data
  abs (a - 0.65) < 0.01 ∧ abs (b - 1.45) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_and_regression_equation_l848_84847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l848_84831

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem x_range (x : ℝ) : floor ((x + 3) / 2) = 3 → 3 ≤ x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_l848_84831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l848_84896

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = -1/3) : Real.cos (3*θ) = 23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l848_84896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l848_84861

open Set

noncomputable section

variable (f : ℝ → ℝ)

axiom f_differentiable : Differentiable ℝ f

axiom f_domain : ∀ x, x < 0 → f x ≠ 0

axiom f_condition : ∀ x, x < 0 → 2 * f x + x * deriv f x > x^2

theorem f_inequality (x : ℝ) :
  (x + 2014)^2 * f (x + 2014) - 4 * f (-2) > 0 ↔ x < -2016 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l848_84861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l848_84824

-- Define the function f(x) = x² + 1/x
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

-- Theorem statement
theorem f_has_one_zero :
  ∃! x : ℝ, x ≠ 0 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_l848_84824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_equilateral_triangle_l848_84839

/-- The area of the shaded regions in an equilateral triangle with inscribed circles -/
theorem shaded_area_in_equilateral_triangle (side_length : ℝ) (h : side_length = 16) :
  let triangle_height := side_length * Real.sqrt 3 / 2
  let central_circle_radius := side_length / 2
  let inscribed_circle_radius := triangle_height / 3
  let sector_area := π * central_circle_radius^2 / 6
  let inscribed_circle_area := π * inscribed_circle_radius^2
  let shaded_area := 2 * (sector_area - inscribed_circle_area / 6)
  shaded_area = 128 * π / 9 := by
  sorry

#eval (128 : ℚ) / 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_equilateral_triangle_l848_84839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_at_least_once_l848_84857

/-- A fair six-sided die -/
def Die : Type := Fin 6

/-- The possible outcomes of rolling two dice -/
def TwoDiceRoll : Type := Die × Die

/-- A winning roll occurs when one die shows three times the number on the other -/
def is_winning_roll (roll : TwoDiceRoll) : Prop :=
  (roll.1.val + 1 = 3 * (roll.2.val + 1)) ∨ (roll.2.val + 1 = 3 * (roll.1.val + 1))

/-- The probability of a winning roll -/
def prob_winning_roll : ℚ := 1 / 9

/-- The number of times the game is played -/
def num_games : ℕ := 3

/-- Theorem: The probability of winning at least once in three games is 217/729 -/
theorem prob_win_at_least_once :
  (1 : ℚ) - (1 - prob_winning_roll) ^ num_games = 217 / 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_at_least_once_l848_84857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l848_84879

def vector1 : Fin 3 → ℝ := ![3, 4, 5]
def vector2 (k : ℝ) : Fin 3 → ℝ := ![2, k, 3]
def vector3 (k : ℝ) : Fin 3 → ℝ := ![2, 3, k + 1]

def parallelepiped_volume (k : ℝ) : ℝ :=
  |Matrix.det (Matrix.of ![vector1, vector2 k, vector3 k])|

theorem parallelepiped_volume_solution (k : ℝ) :
  k > 0 → parallelepiped_volume k = 20 → k = (15 + Real.sqrt 237) / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_solution_l848_84879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_good_point_in_isosceles_triangle_l848_84862

/-- A point inside a triangle is considered "good" if the three cevians passing through it are equal -/
def is_good_point (A B C P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- The number of good points in a triangle -/
def num_good_points (A B C : EuclideanSpace ℝ (Fin 2)) : ℕ := sorry

theorem unique_good_point_in_isosceles_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_isosceles : dist A B = dist B C) 
  (h_odd : Odd (num_good_points A B C)) : 
  num_good_points A B C = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_good_point_in_isosceles_triangle_l848_84862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_square_volume_and_area_l848_84830

/-- Represents a square rotating around an axis through one of its vertices -/
structure RotatingSquare where
  a : ℝ  -- Side length of the square
  φ : ℝ  -- Angle between axis and side
  h1 : a > 0
  h2 : 0 ≤ φ ∧ φ ≤ Real.pi / 4

/-- Volume of the solid of revolution -/
noncomputable def volume (s : RotatingSquare) : ℝ :=
  Real.pi * s.a^3 * Real.sqrt 2 * Real.sin (s.φ + Real.pi / 4)

/-- Surface area of the solid of revolution -/
noncomputable def surfaceArea (s : RotatingSquare) : ℝ :=
  4 * Real.pi * s.a^2 * Real.sqrt 2 * Real.sin (s.φ + Real.pi / 4)

/-- Theorem stating the volume and surface area of the solid of revolution -/
theorem rotating_square_volume_and_area (s : RotatingSquare) :
  volume s = Real.pi * s.a^3 * Real.sqrt 2 * Real.sin (s.φ + Real.pi / 4) ∧
  surfaceArea s = 4 * Real.pi * s.a^2 * Real.sqrt 2 * Real.sin (s.φ + Real.pi / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_square_volume_and_area_l848_84830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_33_over_2_l848_84868

-- Define the expression
noncomputable def expr : ℝ := (1/4)^(-2 : ℤ) + (1/2) * (Real.log 6 / Real.log 3) - (Real.log 2 / Real.log 3) / 2

-- State the theorem
theorem expr_equals_33_over_2 : expr = 33/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_equals_33_over_2_l848_84868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l848_84803

noncomputable section

open Real

-- Define the angle function
def angle (A B C : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define IsIsoscelesRight
def IsIsoscelesRight (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  angle B A C = π / 2 ∧ ‖B - A‖ = ‖C - A‖

-- Main theorem
theorem isosceles_right_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖B - A + (C - A)‖ = ‖B - A - (C - A)‖)
  (h2 : sin (angle B A C) = 2 * sin (angle A B C) * cos (angle B C A)) :
  IsIsoscelesRight A B C := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_l848_84803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_N_l848_84885

/-- Definition of matrix N -/
noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := sorry

/-- First condition -/
axiom cond1 : N.mulVec ![3, -2] = ![4, 1]

/-- Second condition -/
axiom cond2 : N.mulVec ![-2, 3] = ![1, 2]

/-- Theorem to prove -/
theorem compute_N : N.mulVec ![7, 0] = ![14, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_N_l848_84885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_makeup_exam_average_score_l848_84801

/-- Proves that the average score of students who took the exam on the make-up date was 80% -/
theorem makeup_exam_average_score
  (total_students : ℕ)
  (assigned_day_percentage : ℚ)
  (assigned_day_average : ℚ)
  (class_average : ℚ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 70 / 100)
  (h3 : assigned_day_average = 60 / 100)
  (h4 : class_average = 66 / 100) :
  let makeup_day_percentage : ℚ := 1 - assigned_day_percentage
  let makeup_day_average : ℚ := 
    (class_average * total_students - assigned_day_average * (assigned_day_percentage * total_students)) /
    (makeup_day_percentage * total_students)
  makeup_day_average = 80 / 100 := by
  sorry

#check makeup_exam_average_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_makeup_exam_average_score_l848_84801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_x5_l848_84855

theorem largest_coefficient_x5 (n : ℕ) :
  (n > 0 ∧ ∀ k : ℕ, k ≤ n → Nat.choose n 5 ≥ Nat.choose n k) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_x5_l848_84855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l848_84817

/-- The sequence a_n defined recursively -/
def a : ℕ → ℚ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 2 => 2 * a (n + 1) / (a (n + 1) + 2)

/-- The theorem stating the general term of the sequence -/
theorem a_general_term (n : ℕ) (h : n > 0) : a n = 2 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l848_84817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l848_84889

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x - 3)) / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x > 3/2 ∧ x ≠ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l848_84889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_five_fourths_l848_84814

theorem log_expression_equals_five_fourths :
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * (Real.log 2 / Real.log 3 + Real.log 2 / Real.log 9) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_equals_five_fourths_l848_84814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_l848_84835

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := 2 - 1 / x

-- Theorem statement
theorem tangent_line_at_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (λ x ↦ m * (x - x₀) + y₀) = (λ x ↦ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_two_l848_84835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_sum_exists_one_seventh_subsequence_sum_not_exists_one_fifth_l848_84810

/-- The geometric progression with first term 1 and common ratio 1/2 -/
def geometric_progression (n : ℕ) : ℚ := (1/2) ^ n

/-- A subsequence of the geometric progression -/
def is_subsequence (s : ℕ → ℚ) : Prop :=
  ∃ f : ℕ → ℕ, Monotone f ∧ StrictMono f ∧ ∀ n, s n = geometric_progression (f n)

/-- The sum of a subsequence -/
noncomputable def subsequence_sum (s : ℕ → ℚ) : ℚ := ∑' n, s n

theorem subsequence_sum_exists_one_seventh :
  ∃ s, is_subsequence s ∧ subsequence_sum s = 1/7 := by
  sorry

theorem subsequence_sum_not_exists_one_fifth :
  ¬ ∃ s, is_subsequence s ∧ subsequence_sum s = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsequence_sum_exists_one_seventh_subsequence_sum_not_exists_one_fifth_l848_84810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_l848_84834

/-- Triangle with sides a, b, c satisfying the given conditions -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≤ b
  h2 : b ≤ c
  h3 : b = 2008
  h4 : a + b > c  -- Triangle inequality
  h5 : a + c > b
  h6 : b + c > a

/-- The set of all valid triangles satisfying the conditions -/
def ValidTriangles : Set Triangle := {t : Triangle | True}

/-- The number of valid triangles -/
noncomputable def NumTriangles : ℕ := (Finset.range 2008).sum (fun a =>
  (Finset.range (4016 - 2008)).sum (fun c' =>
    let c := c' + 2008
    if a ≤ 2008 ∧ 2008 ≤ c ∧ a + 2008 > c ∧ a + c > 2008 ∧ 2008 + c > a
    then 1
    else 0))

theorem count_triangles : NumTriangles = 2017036 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_l848_84834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l848_84888

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- State the theorem
theorem f_properties :
  (f 2 = -4/3 ∧ (deriv f) 2 = 0) ∧
  (∀ k : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = k ∧ f x₂ = k ∧ f x₃ = k) ↔ -4/3 < k ∧ k < 8/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l848_84888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_l848_84890

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Define the orthocenter
def orthocenter (a b c : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the triangle OAB
structure TriangleOAB where
  A : PointOnParabola
  B : PointOnParabola
  orthocenter_is_focus : orthocenter origin (A.x, A.y) (B.x, B.y) = focus

-- Define the area of a triangle
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_OAB (t : TriangleOAB) :
  triangleArea origin (t.A.x, t.A.y) (t.B.x, t.B.y) = 10 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_l848_84890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l848_84827

def sequence_b (b₁ b₂ : ℕ+) : ℕ → ℕ+
  | 0 => b₁
  | 1 => b₂
  | (n + 2) => ⟨(sequence_b b₁ b₂ n + 3001) / (1 + sequence_b b₁ b₂ (n + 1)), sorry⟩

theorem min_sum_first_two_terms :
  ∀ b₁ b₂ : ℕ+, (∀ n : ℕ, (sequence_b b₁ b₂ (n + 2)).val * (1 + (sequence_b b₁ b₂ (n + 1)).val) = (sequence_b b₁ b₂ n).val + 3001) →
  b₁.val + b₂.val ≥ 3002 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_first_two_terms_l848_84827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deeper_end_depth_l848_84828

/-- Represents a swimming pool with a trapezoidal cross-section -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  shallowDepth : ℝ
  volume : ℝ

/-- Calculates the depth at the deeper end of the pool -/
noncomputable def deeperDepth (pool : SwimmingPool) : ℝ :=
  (2 * pool.volume) / (pool.width * pool.length) - pool.shallowDepth

/-- Theorem stating that for a specific swimming pool, the deeper end is 4 meters deep -/
theorem deeper_end_depth :
  let pool : SwimmingPool := {
    width := 9,
    length := 12,
    shallowDepth := 1,
    volume := 270
  }
  deeperDepth pool = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_deeper_end_depth_l848_84828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l848_84872

-- Define the marked price and cost price as real numbers
variable (MP CP : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.13

-- Define the gain rate
def gain_rate : ℝ := 0.359375

-- Define the selling price after discount
def selling_price (MP : ℝ) : ℝ := MP * (1 - discount_rate)

-- Theorem statement
theorem cost_price_percentage (h : selling_price MP = CP * (1 + gain_rate)) :
  ∃ ε > 0, |CP / MP - 0.64| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_percentage_l848_84872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hyperbola_asymptotes_l848_84842

/-- The distance from a point to the asymptotes of a hyperbola --/
noncomputable def distance_to_asymptotes (a b x₀ y₀ : ℝ) : ℝ :=
  |x₀| / Real.sqrt (1 + (a / b)^2)

/-- Theorem: The distance from (4, 0) to the asymptotes of x² - y² = 1 is 2√2 --/
theorem distance_to_hyperbola_asymptotes :
  distance_to_asymptotes 1 1 4 0 = 2 * Real.sqrt 2 := by
  -- Unfold the definition of distance_to_asymptotes
  unfold distance_to_asymptotes
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The proof steps would go here, but for now we use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_hyperbola_asymptotes_l848_84842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_trend_l848_84812

/-- Represents the percentage of working adults in Beatrice City working from home for a given year -/
def WorkFromHomePercentage : ℕ → ℚ
  | 1985 => 4
  | 1995 => 9
  | 2005 => 14
  | 2015 => 12
  | 2025 => 20
  | _ => 0  -- For years not specified, we return 0

/-- Represents the trend of the graph between two years -/
inductive GraphTrend
  | Increasing
  | Decreasing
  | Steady

/-- Determines the trend of the graph between two years -/
def getTrend (year1 year2 : ℕ) : GraphTrend :=
  if WorkFromHomePercentage year2 > WorkFromHomePercentage year1 then
    GraphTrend.Increasing
  else if WorkFromHomePercentage year2 < WorkFromHomePercentage year1 then
    GraphTrend.Decreasing
  else
    GraphTrend.Steady

/-- Theorem stating that the graph increases, then slightly declines, and then increases strongly -/
theorem work_from_home_trend :
  getTrend 1985 1995 = GraphTrend.Increasing ∧
  getTrend 1995 2005 = GraphTrend.Increasing ∧
  getTrend 2005 2015 = GraphTrend.Decreasing ∧
  getTrend 2015 2025 = GraphTrend.Increasing ∧
  (WorkFromHomePercentage 2025 - WorkFromHomePercentage 2015 > 
   WorkFromHomePercentage 2005 - WorkFromHomePercentage 1995) :=
by
  sorry

/-- The correct answer is option C -/
def correct_answer : String := "C"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_from_home_trend_l848_84812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_coins_for_all_amounts_l848_84822

inductive Coin
| penny
| nickel
| dime
| quarter
| half_dollar

def coin_value : Coin → Nat
| Coin.penny => 1
| Coin.nickel => 5
| Coin.dime => 10
| Coin.quarter => 25
| Coin.half_dollar => 50

def can_make_amount (coins : List Coin) (amount : Nat) : Prop :=
  ∃ (subset : List Coin), 
    (subset.map coin_value).sum = amount

theorem minimum_coins_for_all_amounts :
  ∃ (coins : List Coin),
    coins.length = 9 ∧
    (∀ (amount : Nat), amount > 0 ∧ amount < 100 → can_make_amount coins amount) ∧
    (∀ (other_coins : List Coin),
      other_coins.length < 9 →
      ∃ (amount : Nat), amount > 0 ∧ amount < 100 ∧ ¬can_make_amount other_coins amount) :=
by
  sorry

#check minimum_coins_for_all_amounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_coins_for_all_amounts_l848_84822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_equals_four_l848_84858

-- Define the rotation angle
noncomputable def θ : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Define the rotation matrix
noncomputable def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

-- Define the scaling factor
def s : ℝ := 2

-- Define the scaling matrix
def T : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![s, 0],
    ![0, s]]

-- Define the combined transformation matrix S
noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ := T * R

-- Theorem statement
theorem det_S_equals_four :
  Matrix.det S = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_equals_four_l848_84858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l848_84867

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + 2

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := (f x)^2 + f (x^2)

-- Theorem statement
theorem max_value_of_y :
  ∃ (max_val : ℝ), max_val = 13 ∧
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → y x ≤ max_val :=
by
  -- We'll use 13 as our maximum value
  use 13
  constructor
  · -- Prove that max_val = 13
    rfl
  · -- Prove that for all x in [1, 3], y x ≤ 13
    intro x ⟨hx_lower, hx_upper⟩
    sorry -- The actual proof would go here

-- The proof is omitted (using sorry) as requested

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l848_84867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomial_l848_84864

/-- A polynomial function over the integers -/
def IntPolynomial (f : ℤ → ℤ) : Prop :=
  ∃ (m : ℕ), ∀ (x y : ℤ), (x - y) ^ (m + 1) ∣ (f x - f y)

/-- Main theorem: If a polynomial takes integer values at m+1 consecutive integers, 
    it takes integer values for all integers -/
theorem integer_valued_polynomial 
  (f : ℤ → ℤ) (m : ℕ) (a : ℤ) 
  (h1 : IntPolynomial f) 
  (h2 : ∀ i : Fin (m + 1), f (a + i) ∈ Set.range f) : 
  ∀ x : ℤ, f x ∈ Set.range f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_valued_polynomial_l848_84864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_knights_probability_sum_numerator_denominator_l848_84886

def total_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (total_knights - chosen_knights * 3) chosen_knights : ℚ) /
      (Nat.choose total_knights chosen_knights : ℚ)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 541 / 609 := by
  sorry

theorem sum_numerator_denominator :
  (probability_adjacent_knights.num.natAbs) +
  (probability_adjacent_knights.den) = 1150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_knights_probability_sum_numerator_denominator_l848_84886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kw_price_to_combined_assets_ratio_l848_84878

/-- The price of company KW at the start of the year -/
def price_KW (a : ℝ) : ℝ := 1.9 * a

/-- Assets of company A at the end of the year -/
def assets_A_end (a : ℝ) : ℝ := 1.2 * a

/-- Assets of company B at the end of the year -/
def assets_B_end (b : ℝ) : ℝ := 0.9 * b

/-- Combined assets of companies A and B at the end of the year -/
def combined_assets_end (a b : ℝ) : ℝ := assets_A_end a + assets_B_end b

/-- Theorem stating the relationship between KW's price and combined assets -/
theorem kw_price_to_combined_assets_ratio (a b : ℝ) 
  (h1 : price_KW a = 2 * b) : 
  ∃ ε > 0, |price_KW a / combined_assets_end a b - 0.9246| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kw_price_to_combined_assets_ratio_l848_84878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l848_84897

/-- The circle equation -/
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + 2*x + y^2 - 6*y + 6 = 0

/-- The line equation -/
def line_eq (a x y : ℝ) : Prop := x + a*y + 1 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (a x y : ℝ) : ℝ := 
  |x + a*y + 1| / Real.sqrt (1 + a^2)

/-- The theorem stating the relationship between the circle, line, and the value of a -/
theorem circle_line_intersection (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ circle_eq x₃ y₃ ∧
    distance_to_line a x₁ y₁ = 1 ∧ distance_to_line a x₂ y₂ = 1 ∧ distance_to_line a x₃ y₃ = 1 ∧
    (∀ (x y : ℝ), circle_eq x y ∧ distance_to_line a x y = 1 → 
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃))) →
  a = Real.sqrt 2 / 4 ∨ a = -Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l848_84897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l848_84880

theorem consecutive_non_prime_powers (k : ℕ+) :
  ∃ (N : ℕ), ∀ (i : ℕ), i < k → ¬ (∃ (p : ℕ) (n : ℕ), Nat.Prime p ∧ N + i = p ^ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_non_prime_powers_l848_84880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_perfect_square_l848_84877

theorem smallest_base_perfect_square :
  ∀ b : ℕ, b > 3 → (∀ k : ℕ, 3 < k ∧ k < b → ¬∃ n : ℕ, 4 * k + 5 = n ^ 2) →
  (∃ n : ℕ, 4 * b + 5 = n ^ 2) → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_base_perfect_square_l848_84877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_13_l848_84887

theorem three_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 900)).card + 1 = 69 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_divisible_by_13_l848_84887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_11_and_12_l848_84891

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Main theorem
theorem x_plus_y_between_11_and_12 
  (x y : ℝ) 
  (h1 : y = 4 * (floor x) + 1) 
  (h2 : y = 2 * (floor (x - 1)) + 7) 
  (h3 : x ≠ ↑(floor x)) : 
  11 < x + y ∧ x + y < 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_between_11_and_12_l848_84891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l848_84859

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C ∧
  t.a = Real.sqrt 7 ∧
  t.b + t.c = 4

-- Helper function to calculate triangle area
noncomputable def area_triangle (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = Real.pi / 3 ∧ 
  area_triangle t = 3 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l848_84859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l848_84894

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + bridge_length
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long takes 27 seconds to cross a 170-meter bridge at 36 kmph -/
theorem train_crossing_bridge : train_crossing_time 100 170 36 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l848_84894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_division_l848_84852

/-- Given a line segment AB extended to Q such that AQ:QB = 7:2, 
    prove that vector Q can be expressed as -2/5 * vector A + 7/5 * vector B -/
theorem vector_division (A B Q : ℝ × ℝ) (h : (dist A Q) / (dist Q B) = 7 / 2) :
  Q = (-2/5 : ℝ) • A + (7/5 : ℝ) • B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_division_l848_84852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l848_84802

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

-- State the theorem
theorem solution_set (x : ℝ) : 
  |f x| ≥ 1/3 ↔ x ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l848_84802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l848_84808

theorem exponential_equation_solution (x : ℝ) : 
  (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x + (3 : ℝ)^x = 1458 → (x + 2) * (x - 2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l848_84808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonicity_condition_l848_84845

-- Define the function f(x) = x^2 + 2ax + 3
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the interval [-4, 6]
def I : Set ℝ := Set.Icc (-4) 6

-- Theorem 1: Maximum and minimum values when a = -2
theorem max_min_values :
  (∀ y ∈ I, f (-2) y ≤ f (-2) (-4)) ∧
  (∀ y ∈ I, f (-2) 2 ≤ f (-2) y) ∧
  f (-2) (-4) = 35 ∧ f (-2) 2 = -1 :=
sorry

-- Theorem 2: Conditions for monotonicity
theorem monotonicity_condition (a : ℝ) :
  (∀ x y, x ∈ I → y ∈ I → x < y → f a x < f a y) ↔ (a ≥ 4 ∨ a ≤ -6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_monotonicity_condition_l848_84845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l848_84837

/-- Given points A(1,0) and B(4,0) in the Cartesian coordinate system,
    and a point P on the line x-y+m=0 such that PA = 1/2 PB,
    prove that the range of the real number m is [-2√2, 2√2]. -/
theorem range_of_m (m : ℝ) : 
  ∃ (x y : ℝ), 
    (x - y + m = 0) ∧ 
    ((x - 1)^2 + y^2 = (1/4) * ((x - 4)^2 + y^2)) → 
    m ∈ Set.Icc (-2 * Real.sqrt 2) (2 * Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l848_84837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_f_increasing_l848_84843

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Theorem 1
theorem f_composition_value : f (f 0 + 4) = 15/17 := by
  sorry

-- Theorem 2
theorem f_increasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_f_increasing_l848_84843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l848_84881

theorem existence_of_special_set (n : ℕ) (h : n ≥ 3) :
  ∃ (S : Finset ℕ), 
    (Finset.card S = n) ∧ 
    (∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ ∅ → B ≠ ∅ → A ≠ B →
      let sum_A := (A.sum id)
      let card_B := (Finset.card B)
      (¬ Nat.Prime (sum_A / card_B)) ∧ 
      (Nat.Coprime (sum_A / card_B) card_B)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l848_84881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l848_84800

/-- Calculates the speed of a car given distance and time -/
noncomputable def car_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem: The speed of a car traveling 585 miles in 9 hours is 65 miles per hour -/
theorem car_speed_problem : car_speed 585 9 = 65 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_l848_84800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l848_84874

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 243 / Real.log x = 5/3 → x = 27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l848_84874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probability_l848_84898

/-- The number of applicants -/
def n : ℕ := 4

/-- The number of positions to be filled -/
def k : ℕ := 2

/-- The number of applicants in the specific group -/
def m : ℕ := 2

/-- The probability of at least one person from the specific group being hired -/
def p : ℚ := 5 / 6

theorem hiring_probability (hn : n = 4) (hk : k = 2) (hm : m = 2) :
  1 - (Nat.choose (n - m) k / Nat.choose n k : ℚ) = p := by
  sorry

#check hiring_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiring_probability_l848_84898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_sin_plus_cos_sqrt3_l848_84873

theorem no_solutions_sin_plus_cos_sqrt3 :
  ∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → Real.sin x + Real.cos x ≠ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_sin_plus_cos_sqrt3_l848_84873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_control_indicator_l848_84838

def meets_criteria (infections : List ℕ) : Prop :=
  infections.length = 7 ∧ ∀ x ∈ infections, x ≤ 5

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

def range (lst : List ℕ) : ℕ :=
  match lst.maximum, lst.minimum with
  | some max, some min => max - min
  | _, _ => 0

def mode (lst : List ℕ) : ℕ :=
  lst.foldl (λ acc x => if lst.count x > lst.count acc then x else acc) 0

theorem epidemic_control_indicator 
  (infections : List ℕ) :
  (average infections ≤ 3 ∧ range infections ≤ 2) ∨
  (mode infections = 1 ∧ range infections ≤ 1) →
  meets_criteria infections := by
  sorry

#eval range [1, 2, 3, 4, 5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_epidemic_control_indicator_l848_84838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l848_84882

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (Real.pi - x) * Real.cos x - 1 + 2 * (Real.cos x)^2

theorem f_monotone_increasing : 
  StrictMonoOn f (Set.Icc (-Real.pi/3) (Real.pi/6)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l848_84882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l848_84841

/-- Given a total spent amount, tax rate, and cost of tax-free items, 
    calculate the sales tax paid on taxable purchases. -/
noncomputable def sales_tax_paid (total_spent : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) : ℝ :=
  let taxable_cost := total_spent - tax_free_cost
  let tax_paid := taxable_cost * tax_rate / (1 + tax_rate)
  tax_paid

/-- Theorem stating that given the problem conditions, 
    the sales tax paid is 0.3 -/
theorem sales_tax_calculation :
  sales_tax_paid 25 0.06 19.7 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_tax_calculation_l848_84841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kingdom_tour_l848_84840

/-- A custom graph type with three types of edges -/
structure TriGraph (V : Type) where
  highway : V → V
  railway : V → V
  dirt : V → V

/-- A path in the graph -/
def GraphPath (V : Type) := List V

/-- Predicate to check if a path is valid (no repeated edges in same direction) -/
def ValidPath {V : Type} (g : TriGraph V) (p : GraphPath V) : Prop := sorry

/-- Predicate to check if a path visits all vertices -/
def VisitsAllVertices {V : Type} (p : GraphPath V) (vertices : Set V) : Prop := sorry

/-- Main theorem -/
theorem kingdom_tour {V : Type} (g : TriGraph V) (vertices : Set V) :
  (∀ v w : V, ∃ p : GraphPath V, p.head? = some v ∧ p.getLast? = some w) →  -- Connected
  (∀ v : V, v ∈ vertices) →                             -- All vertices are in the set
  (∀ v : V, g.highway (g.dirt v) = v) →                 -- Highway-dirt subgraph forms cycles
  ∃ p : GraphPath V, ValidPath g p ∧ VisitsAllVertices p vertices := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kingdom_tour_l848_84840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l848_84809

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) - (Real.cos (ω * x))^2 + 1/2

-- State the theorem
theorem problem_solution (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  f ω (π/3) = f ω (-π/3) →  -- Axis of symmetry
  f ω (π/12) = 0 →  -- Zero adjacent to axis of symmetry
  c = Real.sqrt 3 →
  f ω C = 1 →
  ∃ (k : ℝ), (k • (1, Real.sin A) : ℝ × ℝ) = (2, Real.sin B) →  -- Collinearity condition
  ω = 1 ∧ C = π/3 ∧ a = 1 ∧ b = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l848_84809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_equation_solution_problem_l848_84866

-- Part 1: Factorization
theorem factorization_problem (m a b : ℝ) : 
  m * a^2 - 4 * m * b^2 = m * (a + 2*b) * (a - 2*b) := by sorry

-- Part 2: Equation solving
theorem equation_solution_problem : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 2 → (1 / (x - 2) = 3 / x) ↔ x = 3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_equation_solution_problem_l848_84866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_square_angle_l848_84823

/-- Theorem: The sides of a square circumscribing an ellipse form a 45° angle with the coordinate axes. -/
theorem circumscribing_square_angle (a b : ℝ) (h : a > b) (h' : b > 0) :
  let ellipse := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let square_sides := {l : Set (ℝ × ℝ) | ∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ x + y = k ∨ x - y = k ∧
    ∃ (x₀ y₀ : ℝ), (x₀, y₀) ∈ ellipse ∧ x₀ * x / a^2 + y₀ * y / b^2 = 1}
  ∀ (l : Set (ℝ × ℝ)), l ∈ square_sides → 
    ∃ (m : ℝ), abs m = 1 ∧ ∀ (x y : ℝ), (x, y) ∈ l → y = m * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_square_angle_l848_84823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l848_84870

-- Define the curves
noncomputable def curve1 (x : ℝ) : ℝ := x^2
noncomputable def curve2 (x : ℝ) : ℝ := Real.sqrt x

-- Define the area function
noncomputable def area_function (x : ℝ) : ℝ := Real.sqrt x - x^2

-- Theorem statement
theorem area_enclosed_by_curves :
  ∃ (S : ℝ), S = ∫ x in Set.Icc 0 1, area_function x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l848_84870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_of_given_triangle_l848_84804

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
noncomputable def givenTriangle : Triangle where
  a := 1
  B := Real.pi / 4  -- 45° in radians
  b := sorry   -- We don't know b yet
  c := sorry   -- We don't know c yet
  A := sorry   -- We don't know A yet
  C := sorry   -- We don't know C yet

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.c * Real.sin t.B

-- Define the diameter of the circumcircle
noncomputable def circumcircleDiameter (t : Triangle) : ℝ := 
  t.b / Real.sin t.B

-- State the theorem
theorem circumcircle_diameter_of_given_triangle : 
  triangleArea givenTriangle = 2 → 
  circumcircleDiameter givenTriangle = 5 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircle_diameter_of_given_triangle_l848_84804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_journey_time_l848_84860

/-- Represents the time taken for Cole's journey to work in hours -/
noncomputable def time_to_work (distance : ℝ) : ℝ := distance / 75

/-- Represents the time taken for Cole's journey back home in hours -/
noncomputable def time_to_home (distance : ℝ) : ℝ := distance / 105

/-- Theorem stating that Cole's journey to work takes 70 minutes -/
theorem cole_journey_time : ∃ (distance : ℝ), 
  time_to_work distance + time_to_home distance = 2 ∧ 
  time_to_work distance * 60 = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cole_journey_time_l848_84860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_75_l848_84821

/-- Calculates the upstream distance traveled by a boat given downstream distance, downstream time, upstream time, and stream speed. -/
noncomputable def upstreamDistance (downstreamDistance : ℝ) (downstreamTime : ℝ) (upstreamTime : ℝ) (streamSpeed : ℝ) : ℝ :=
  let downstreamSpeed := downstreamDistance / downstreamTime
  let boatSpeed := downstreamSpeed - streamSpeed
  let upstreamSpeed := boatSpeed - streamSpeed
  upstreamSpeed * upstreamTime

/-- Theorem stating that under given conditions, the upstream distance is 75 km. -/
theorem upstream_distance_is_75 :
  upstreamDistance 100 8 15 3.75 = 75 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_75_l848_84821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_form_isosceles_triangle_l848_84805

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point :=
  { x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope),
    y := l1.slope * ((l2.intercept - l1.intercept) / (l1.slope - l2.slope)) + l1.intercept }

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a triangle is isosceles -/
def isIsosceles (p1 p2 p3 : Point) : Prop :=
  let d12 := distance p1 p2
  let d13 := distance p1 p3
  let d23 := distance p2 p3
  d12 = d13 ∨ d12 = d23 ∨ d13 = d23

/-- The main theorem -/
theorem lines_form_isosceles_triangle (l1 l2 l3 : Line)
  (h1 : l1 = { slope := 4, intercept := 3 })
  (h2 : l2 = { slope := -4, intercept := 3 })
  (h3 : l3 = { slope := 0, intercept := -3 }) :
  let p1 := intersection l1 l2
  let p2 := intersection l1 l3
  let p3 := intersection l2 l3
  isIsosceles p1 p2 p3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_form_isosceles_triangle_l848_84805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_base4_157_l848_84871

/-- Converts a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  digits.filter (fun d => d % 2 = 1) |>.length

theorem odd_digits_base4_157 :
  countOddDigits (toBase4 157) = 3 := by
  -- Evaluate toBase4 157
  have h1 : toBase4 157 = [2, 1, 3, 1] := by rfl
  -- Count odd digits
  have h2 : countOddDigits [2, 1, 3, 1] = 3 := by rfl
  -- Combine the results
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_digits_base4_157_l848_84871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l848_84863

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_c : c^2 = a^2 + b^2

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

def Hyperbola.vertex_left (h : Hyperbola) : Point := ⟨-h.a, 0⟩
def Hyperbola.vertex_right (h : Hyperbola) : Point := ⟨h.a, 0⟩
def Hyperbola.focus_left (h : Hyperbola) : Point := ⟨-h.c, 0⟩
def Hyperbola.focus_right (h : Hyperbola) : Point := ⟨h.c, 0⟩

/-- A point is on the hyperbola if it satisfies the equation -/
def Hyperbola.on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Vector between two points -/
def vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

/-- Dot product of two vectors -/
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Main theorem -/
theorem hyperbola_dot_product (h : Hyperbola) (c m n : Point)
  (h_c : c ≠ h.vertex_left ∧ c ≠ h.vertex_right)
  (h_on : h.on_hyperbola c)
  (h_m : ∃ t : ℝ, m = ⟨h.a^2 / h.c, t * c.y⟩)
  (h_n : ∃ t : ℝ, n = ⟨h.a^2 / h.c, t * c.y⟩) :
  dot_product (vector (h.focus_left) m) (vector (h.focus_right) n) = -2 * h.b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_dot_product_l848_84863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_inverse_x_in_expansion_l848_84849

/-- The coefficient of 1/x in the expansion of (2/x^2 - x)^5 -/
def coefficientOfInverseX : ℤ := -40

/-- The binomial expansion of (2/x^2 - x)^5 -/
noncomputable def expansion (x : ℝ) : ℝ := (2 / x^2 - x)^5

/-- Theorem stating the existence of the coefficient of 1/x in the expansion -/
theorem coefficient_of_inverse_x_in_expansion :
  ∃ (f : ℝ → ℝ), ∀ x, x ≠ 0 → expansion x = coefficientOfInverseX / x + x * f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_inverse_x_in_expansion_l848_84849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_linear_function_between_range_of_k_for_linear_function_between_l848_84851

-- Part 1
theorem unique_linear_function_between (f h : ℝ → ℝ) :
  (∀ x, f x = x^2 + x) →
  (∀ x, h x = -x^2 + x) →
  ∃! g : ℝ → ℝ, ∃ k b : ℝ,
    (∀ x, g x = k * x + b) ∧
    (∀ x, f x ≥ g x ∧ g x ≥ h x) ∧
    g = λ x ↦ x :=
by sorry

-- Part 2
theorem range_of_k_for_linear_function_between (f h : ℝ → ℝ) (b : ℝ) :
  (∀ x > 0, f x = x^2 + x + 2) →
  (∀ x > 0, h x = x - 1/x) →
  b = 1 →
  {k : ℝ | ∀ x > 0, f x ≥ k * x + b ∧ k * x + b ≥ h x} = Set.Icc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_linear_function_between_range_of_k_for_linear_function_between_l848_84851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l848_84844

/-- The length of the major axis of an ellipse with given foci and tangent line -/
theorem ellipse_major_axis_length : 
  let f₁ : ℝ × ℝ := (5, 10)
  let f₂ : ℝ × ℝ := (35, 40)
  let tangent_y : ℝ := -5
  let major_axis_length : ℝ := 30 * Real.sqrt 5
  major_axis_length = 30 * Real.sqrt 5 := by
  -- Introduce the variables
  intro f₁ f₂ tangent_y major_axis_length
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l848_84844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l848_84846

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) - 1

theorem f_max_min_on_interval :
  ∃ (max min : ℝ), max = 1 ∧ min = -2 ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ max ∧ f x ≥ min) ∧
  (∃ x₁ ∈ Set.Icc 0 (Real.pi / 2), f x₁ = max) ∧
  (∃ x₂ ∈ Set.Icc 0 (Real.pi / 2), f x₂ = min) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_on_interval_l848_84846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l848_84876

noncomputable def f (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 2)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l848_84876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_most_popular_l848_84865

def zongzi_types := Fin 4

def preference_list : List (Fin 4) := [2, 3, 3, 0, 0, 1, 0, 1, 1, 1, 0,
                                       2, 2, 0, 0, 1, 0, 0, 2, 3, 2, 3]

def count (l : List (Fin 4)) (t : Fin 4) : Nat :=
  l.filter (· == t) |>.length

theorem a_most_popular :
  ∀ t : Fin 4, t ≠ 0 → count preference_list 0 ≥ count preference_list t :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_most_popular_l848_84865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_time_four_units_min_b_value_l848_84856

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then (4 + x) / (4 - x)
  else if 2 < x ∧ x ≤ 5 then 5 - x
  else 0

-- Define the concentration function y
noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * f x

-- Define the effectiveness condition
def is_effective (c : ℝ) : Prop := c ≥ 4

-- Theorem 1: Effective time for 4 units is 4 days
theorem effective_time_four_units :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → is_effective (y 4 x) :=
by
  sorry

-- Theorem 2: Minimum value of b
theorem min_b_value :
  ∃ b : ℝ, b = 24 - 16 * Real.sqrt 2 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 →
    is_effective (y 2 (x + 3) + y b x)) ∧
  (∀ b' : ℝ, b' < b →
    ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧
      ¬is_effective (y 2 (x + 3) + y b' x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effective_time_four_units_min_b_value_l848_84856
