import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_sqrt_481_l924_92470

/-- An isosceles trapezoid with specific side lengths -/
structure IsoscelesTrapezoid where
  -- Base lengths
  AB : ℝ
  CD : ℝ
  -- Leg length
  AD : ℝ
  -- Conditions
  AB_positive : AB > 0
  CD_positive : CD > 0
  AD_positive : AD > 0
  AB_greater_CD : AB > CD
  isosceles : AD = BC

/-- The length of the diagonal AC in the isosceles trapezoid -/
noncomputable def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt 481

/-- Theorem stating that for an isosceles trapezoid with given side lengths, 
    the length of diagonal AC is √481 -/
theorem diagonal_length_is_sqrt_481 (t : IsoscelesTrapezoid) 
    (h1 : t.AB = 26) (h2 : t.CD = 12) (h3 : t.AD = 13) : 
    diagonal_length t = Real.sqrt 481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_is_sqrt_481_l924_92470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l924_92466

/-- The length of the chord intercepted by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (d e f : ℝ) : ℝ :=
  let line := fun x : ℝ => a * x + b
  let circle := fun (x y : ℝ) => x^2 + y^2 + d*x + e*y + f
  2 * Real.sqrt (25 - 5)  -- This is a placeholder calculation

/-- The main theorem -/
theorem chord_length_specific_case :
  chord_length 2 3 (-6) (-8) 0 0 = 4 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l924_92466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l924_92413

/-- 
Given that the terminal side of angle α is in the fourth quadrant 
and 2sin(2α) + 1 = cos(2α), prove that tan(α - π/4) = 3.
-/
theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0 ∧ Real.cos α > 0) -- Terminal side in fourth quadrant
  (h2 : 2 * Real.sin (2 * α) + 1 = Real.cos (2 * α)) : 
  Real.tan (α - Real.pi/4) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l924_92413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l924_92458

-- Define the daily output in 10,000 pieces
variable (x : ℝ)

-- Define the daily defect rate as a function of x
noncomputable def p (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then (x^2 + 60) / 540
  else if 12 < x ∧ x ≤ 20 then 1/2
  else 0

-- Define the daily profit function in 10,000 yuan
noncomputable def y (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then 5*x/3 - x^3/180
  else if 12 < x ∧ x ≤ 20 then x/2
  else 0

-- State the theorem
theorem max_profit :
  ∀ z, 0 < z ∧ z ≤ 20 → y z ≤ y 10 ∧ y 10 = 100/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l924_92458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l924_92450

theorem congruence_solutions_count :
  let solutions := {x : ℕ | x > 0 ∧ x < 150 ∧ (x + 17) % 46 = 72 % 46}
  Finset.card (Finset.filter (λ x => x > 0 ∧ x < 150 ∧ (x + 17) % 46 = 72 % 46) (Finset.range 150)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruence_solutions_count_l924_92450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_and_inequality_l924_92479

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 + 4*m + 4) * x^(m + 2)

-- State the theorem
theorem power_function_decreasing_and_inequality (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ > f m x₂) →
  (∀ a : ℝ, (2*a - 1)^(-m) < (a + 3)^(-m) → a < 4) →
  m = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_and_inequality_l924_92479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_consecutive_divisible_by_digit_sum_l924_92488

noncomputable def sum_of_digits : ℕ → ℕ
| 0 => 0
| n+1 => (n+1) % 10 + sum_of_digits (n / 10)

theorem eighteen_consecutive_divisible_by_digit_sum (n : ℕ) (h : 100 ≤ n ∧ n ≤ 982) :
  ∃ k : ℕ, k ≤ 17 ∧ (∃ d : ℕ, n + k = d * (sum_of_digits (n + k))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_consecutive_divisible_by_digit_sum_l924_92488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_l924_92414

/-- The total surface area of a prism with specific properties -/
theorem prism_surface_area (a : ℝ) (h : a > 0) : 
  2 * a^2 + 2 * a^2 + 2 * (a^2 * (Real.sqrt 3 / 2)) = a^2 * (4 + Real.sqrt 3) :=
by
  -- Expand the left side of the equation
  have h1 : 2 * a^2 + 2 * a^2 + 2 * (a^2 * (Real.sqrt 3 / 2)) = 4 * a^2 + a^2 * Real.sqrt 3 := by
    ring
  
  -- Rewrite the right side of the equation
  have h2 : a^2 * (4 + Real.sqrt 3) = 4 * a^2 + a^2 * Real.sqrt 3 := by
    ring
  
  -- Use transitivity to prove the equality
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_surface_area_l924_92414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l924_92410

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.cos x ^ 2 - 3) * Real.sin x

-- State the theorem
theorem function_minimum_implies_a_range :
  (∀ a : ℝ, (∃ m : ℝ, ∀ x : ℝ, f a x ≥ m) → m = -3) →
  (∃ a_min a_max : ℝ, a_min = -3/2 ∧ a_max = 12 ∧
    ∀ a : ℝ, (∀ x : ℝ, f a x ≥ -3) ↔ a_min ≤ a ∧ a ≤ a_max) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_implies_a_range_l924_92410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_formula_l924_92489

/-- A cylinder where the front view and lateral development are similar rectangles -/
structure SimilarRectangleCylinder where
  radius : ℝ
  height : ℝ
  similar_rectangles : (2 * radius) / height = height / (2 * Real.pi * radius)

/-- The ratio of total surface area to lateral surface area for a SimilarRectangleCylinder -/
noncomputable def surface_area_ratio (c : SimilarRectangleCylinder) : ℝ :=
  (2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius^2) / (2 * Real.pi * c.radius * c.height)

theorem surface_area_ratio_formula (c : SimilarRectangleCylinder) :
  surface_area_ratio c = 1 + 1 / (2 * Real.sqrt Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_formula_l924_92489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_line_tangent_to_circle_l924_92426

-- Define the lines and circle
def line1 (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def lineL (x y m : ℝ) : Prop := x - 2 * y + m = 0
def circleC (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1/5

-- Define the distance function
noncomputable def distance (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

-- Theorem 1: Find the value of m
theorem find_m : ∃ m : ℝ, 
  distance 4 (-2) 5 = (1/2) * distance 1 (-2) m ∧ 
  lineL 0 0 m ∧ 
  m > 0 := by sorry

-- Theorem 2: Determine the positional relationship
theorem line_tangent_to_circle : 
  ∃ x y : ℝ, lineL x y 5 ∧ circleC x y ∧
  distance 1 (-2) 1 = Real.sqrt (1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_m_line_tangent_to_circle_l924_92426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l924_92409

theorem sin_plus_cos_value (α : ℝ) (h : Real.tan (α/2) = 1/2) :
  Real.sin α + Real.cos α = 7/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l924_92409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_formula_l924_92404

/-- Represents a saturated monohydric alcohol -/
structure Alcohol where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ
  mass : ℝ
  gas_volume : ℝ

/-- The molar volume of an ideal gas at standard temperature and pressure (STP) -/
def molar_volume_stp : ℝ := 22.4

/-- Calculates the number of moles of gas produced -/
noncomputable def moles_of_gas (a : Alcohol) : ℝ := a.gas_volume / molar_volume_stp

/-- Calculates the molar mass of the alcohol -/
noncomputable def molar_mass (a : Alcohol) : ℝ := a.mass / (moles_of_gas a)

/-- Theorem stating that the given alcohol has the formula C₉H₁₉OH -/
theorem alcohol_formula (a : Alcohol) 
  (h_mass : a.mass = 28.8)
  (h_gas : a.gas_volume = 4.48)
  (h_saturated : a.hydrogen = 2 * a.carbon + 2)
  (h_monohydric : a.oxygen = 1) :
  a.carbon = 9 ∧ a.hydrogen = 19 ∧ a.oxygen = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_formula_l924_92404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximation_l924_92447

noncomputable def segment1_speed : ℝ := 60
noncomputable def segment1_distance : ℝ := 30
noncomputable def segment2_speed : ℝ := 70
noncomputable def segment2_distance : ℝ := 35
noncomputable def segment3_speed : ℝ := 80
noncomputable def segment3_time : ℝ := 1
noncomputable def segment4_speed : ℝ := 55
noncomputable def segment4_time : ℝ := 1/3

noncomputable def total_distance : ℝ := segment1_distance + segment2_distance + segment3_speed * segment3_time + segment4_speed * segment4_time
noncomputable def total_time : ℝ := segment1_distance / segment1_speed + segment2_distance / segment2_speed + segment3_time + segment4_time

theorem average_speed_approximation :
  abs ((total_distance / total_time) - 70) < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_approximation_l924_92447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_value_after_four_bailouts_l924_92428

/-- The value of the dollar in gold after n bailouts -/
def dollar_value (n : ℕ) : ℚ :=
  let initial_value := 1 / 980
  let rec bailout_value (k : ℕ) : ℚ :=
    match k with
    | 0 => initial_value
    | k + 1 => bailout_value k * (1 + 1 / (2^(2^k)))
  bailout_value n

/-- The theorem stating the value of the dollar after 4 bailouts -/
theorem dollar_value_after_four_bailouts :
  dollar_value 4 = (1 / 490) * (1 - 1 / (2^16)) := by
  sorry

/-- Calculate b + c where b is the denominator and c is the exponent in the result -/
def result : ℕ :=
  let b := 490
  let c := 16
  b + c

#eval result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dollar_value_after_four_bailouts_l924_92428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l924_92499

-- Define the function f(x) = log₂x + x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x

-- State the theorem
theorem solution_interval :
  ∃! x : ℝ, 1/2 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_interval_l924_92499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pump_half_time_is_half_hour_l924_92490

-- Define the time it takes for the second pump to drain the whole pond
noncomputable def second_pump_time : ℝ := 1.090909090909091

-- Define the time it takes for both pumps to drain half the pond together
noncomputable def both_pumps_half_time : ℝ := 1/2

-- Theorem to prove
theorem first_pump_half_time_is_half_hour :
  ∃ x : ℝ, x > 0 ∧ 
    (1 / (2 * x) + 1 / second_pump_time = 1 / both_pumps_half_time) ∧ 
    x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_pump_half_time_is_half_hour_l924_92490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_integrality_l924_92497

theorem binomial_fraction_integrality (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ (m : ℕ), (n + 2*k - 3) * Nat.factorial n = m * (k + 2) * Nat.factorial k * Nat.factorial (n - k) ↔ 
  ∃ (q : ℕ), n = (k + 2) * q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_fraction_integrality_l924_92497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_game_results_l924_92443

/-- A regular tetrahedron with faces numbered 1, 2, 3, 4 -/
structure Tetrahedron :=
  (faces : Finset Nat)
  (regular : faces = {1, 2, 3, 4})

/-- The result of throwing the tetrahedron 5 times -/
def ThrowResult := Fin 5 → Nat

/-- The probability of getting each number is 1/4 -/
noncomputable def prob_each_face (n : Nat) : ℝ := 1 / 4

/-- The reward function: 200 yuan for 3, -100 yuan otherwise -/
def reward (n : Nat) : ℤ :=
  if n = 3 then 200 else -100

/-- The theorem to be proved -/
theorem tetrahedron_game_results (t : Tetrahedron) :
  /- 1. Probability that product is divisible by 4 -/
  (57 : ℚ) / 64 = 57 / 64 ∧
  /- 2. Expected value of money after the game -/
  (875 : ℝ) = 875 ∧
  /- 3. Probability of winning at least 300 yuan in at most 3 throws -/
  (5 : ℚ) / 32 = 5 / 32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_game_results_l924_92443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l924_92467

/-- In triangle ABC, prove that the maximum value of (a+b)/c is √2, given b*cos(C) + c*cos(B) = c*sin(A) --/
theorem triangle_side_ratio_max (a b c A B C : ℝ) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) →
  -- Given condition
  b * Real.cos C + c * Real.cos B = c * Real.sin A →
  -- Conclusion: Maximum value of (a+b)/c is √2
  (∀ x, (a + b) / c ≤ x → x ≤ Real.sqrt 2) ∧ 
  (∃ x, (a + b) / c = x ∧ x = Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l924_92467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_symmetric_origin_l924_92459

-- Define the function f(x) = x - 2/x
noncomputable def f (x : ℝ) : ℝ := x - 2/x

-- Theorem stating that f is an odd function
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  intro x hx
  simp [f]
  field_simp
  ring
  sorry

-- Theorem stating that f is symmetric with respect to the origin
theorem f_symmetric_origin : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  exact f_is_odd


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_symmetric_origin_l924_92459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_tax_income_distribution_l924_92440

/-- Represents the income distribution and tax system in a city --/
structure CityEconomy where
  /-- The percentage of total income for the poor group (x%) --/
  poor_income : ℝ
  /-- The percentage of total income for the middle group (4x%) --/
  middle_income : ℝ
  /-- The percentage of total income for the rich group (5x%) --/
  rich_income : ℝ
  /-- The tax rate on the rich group's income --/
  tax_rate : ℝ
  /-- Assumption that the groups are equal in size --/
  equal_groups : poor_income + middle_income + rich_income = 100
  /-- Relationship between incomes --/
  income_relation : middle_income = 4 * poor_income ∧ rich_income = 5 * poor_income
  /-- Definition of the tax rate --/
  tax_rate_def : tax_rate = poor_income^2 / 4 + poor_income

/-- Calculates the post-tax income distribution --/
noncomputable def post_tax_incomes (e : CityEconomy) : ℝ × ℝ × ℝ :=
  let tax_amount := e.rich_income * e.tax_rate / 100
  let poor_new := e.poor_income + 0.75 * tax_amount
  let middle_new := e.middle_income + 0.25 * tax_amount
  let rich_new := e.rich_income - tax_amount
  (poor_new, middle_new, rich_new)

/-- The main theorem stating the post-tax income distribution --/
theorem post_tax_income_distribution (e : CityEconomy) :
  post_tax_incomes e = (23.125, 44.375, 32.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_tax_income_distribution_l924_92440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_l924_92402

/-- Represents a six-digit number in the form A15B94 --/
structure SixDigitNumber where
  A : Nat
  B : Nat
  h1 : A < 10
  h2 : B < 10

/-- Checks if a natural number is divisible by another natural number --/
def isDivisibleBy (n m : Nat) : Prop := ∃ k, n = m * k

/-- Converts a SixDigitNumber to its numerical value --/
def SixDigitNumber.toNat (n : SixDigitNumber) : Nat :=
  n.A * 100000 + 15000 + n.B * 100 + 94

/-- The main theorem --/
theorem six_digit_divisibility (n : SixDigitNumber) :
  isDivisibleBy (n.toNat) 99 ↔ n.B = 3 ∧ n.A = 5 := by
  sorry

#check six_digit_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_l924_92402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l924_92462

/-- A rectangle ABCD with a point P on side CD -/
structure RectangleWithPoint where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Distance of point P from vertex D -/
  pd : ℝ
  /-- ab > 0 (non-degenerate rectangle) -/
  ab_pos : ab > 0
  /-- ad > 0 (non-degenerate rectangle) -/
  ad_pos : ad > 0
  /-- 0 ≤ pd ≤ ab (P is on side CD) -/
  pd_range : 0 ≤ pd ∧ pd ≤ ab

/-- The probability that AB is the longest side of triangle APB -/
noncomputable def prob_longest_side (r : RectangleWithPoint) : ℝ :=
  r.pd / r.ab

theorem rectangle_ratio (r : RectangleWithPoint) 
    (h : prob_longest_side r = 1/3) : 
    r.ad / r.ab = Real.sqrt 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ratio_l924_92462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_cosine_l924_92485

-- Define the original function
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

-- Define the shifted function
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6)

-- Theorem statement
theorem horizontal_shift_cosine :
  ∀ x : ℝ, f x = g (x - Real.pi / 12) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_cosine_l924_92485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_isosceles_right_triangle_l924_92455

/-- Line equation -/
def line (a x y : ℝ) : Prop := a * x + y - 1 = 0

/-- Circle equation -/
def circle_eq (a x y : ℝ) : Prop := (x - 1)^2 + (y + a)^2 = 1

/-- Distance from point (x,y) to line ax + by + c = 0 -/
noncomputable def distancePointToLine (a b c x y : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

/-- Radius of the circle -/
def circleRadius : ℝ := 1

theorem intersection_isosceles_right_triangle (a : ℝ) :
  (∃ x y : ℝ, line a x y ∧ circle_eq a x y) →
  (distancePointToLine a 1 (-1) 1 (-a) = circleRadius * Real.sqrt 2 / 2) →
  a = 1 ∨ a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_isosceles_right_triangle_l924_92455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_negative_one_l924_92473

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log (1 - x)

-- State the theorem
theorem derivative_f_at_negative_one :
  deriv f (-1) = -5/2 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_negative_one_l924_92473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_increase_approx_27_percent_l924_92486

/-- Calculates the percentage increase from sale price to post-sale price -/
noncomputable def percentage_increase (original_price : ℝ) : ℝ :=
  let first_week_discount := 0.13
  let second_week_discount := 0.08
  let original_tax_rate := 0.07
  let new_tax_rate := 0.09
  
  let price_after_first_discount := original_price * (1 - first_week_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_week_discount)
  let final_sale_price := price_after_second_discount * (1 + original_tax_rate)
  let post_sale_price := original_price * (1 + new_tax_rate)
  
  (post_sale_price - final_sale_price) / final_sale_price * 100

/-- The percentage increase from sale price to post-sale price is approximately 27% -/
theorem sale_price_increase_approx_27_percent (original_price : ℝ) (hp : original_price > 0) :
  ∃ ε > 0, abs (percentage_increase original_price - 27) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_increase_approx_27_percent_l924_92486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_products_difference_le_one_l924_92469

/-- An infinite arithmetic progression --/
def ArithmeticProgression (a b : ℝ) : Set ℝ := {x : ℝ | ∃ n : ℤ, x = a + n * b}

/-- The set of products of pairs of members in an arithmetic progression --/
def ProductsOfPairs (ap : Set ℝ) : Set ℝ := {p : ℝ | ∃ x y, x ∈ ap ∧ y ∈ ap ∧ p = x * y}

/-- Theorem: For any infinite arithmetic progression, there exist two products of pairs
    of its members whose difference is less than or equal to 1 --/
theorem products_difference_le_one (a b : ℝ) :
  ∃ p q, p ∈ ProductsOfPairs (ArithmeticProgression a b) ∧ 
         q ∈ ProductsOfPairs (ArithmeticProgression a b) ∧ 
         |p - q| ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_products_difference_le_one_l924_92469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l924_92484

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem system_solution :
  ∃ (x₁ x₂ : ℝ),
    (2 * floor x₁ + x₂ = 3/2) ∧
    (3 * floor x₁ - 2 * x₂ = 4) ∧
    (1 ≤ x₁) ∧ (x₁ < 2) ∧
    (x₂ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l924_92484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l924_92483

theorem modulus_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.abs (-i * (1 + 2 * i)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l924_92483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l924_92422

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3) / Real.log (1/2)

-- State the theorem
theorem f_monotone_decreasing :
  StrictMonoOn f (Set.Ioi 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l924_92422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_palindromic_primes_l924_92480

def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    let sqrt_n := (n.sqrt : ℕ)
    (List.range (sqrt_n - 1)).all (fun m => n % (m + 2) ≠ 0)

def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def isPalindromeDigitsPrime (n : ℕ) : Bool :=
  10 ≤ n && n < 100 && 
  isPrime n && 
  isPrime (reverseDigits n) && 
  isPrime (n / 10) && 
  isPrime (n % 10)

def palindromicPrimes : List ℕ :=
  List.filter isPalindromeDigitsPrime (List.range 90 ++ [90, 91, 92, 93, 94, 95, 96, 97, 98, 99])

theorem sum_palindromic_primes : 
  List.sum palindromicPrimes = 110 := by
  sorry

#eval List.sum palindromicPrimes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_palindromic_primes_l924_92480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_is_sqrt_two_l924_92454

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 3 else Real.sqrt x

-- State the theorem
theorem f_of_a_is_sqrt_two (a : ℝ) (h : f (a - 3) = f (a + 2)) : f a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_a_is_sqrt_two_l924_92454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l924_92411

theorem counterexample_exists : ∃ f : ℝ → ℝ,
  (∀ x, x ∈ Set.Ioo 0 2 → f x > f 0) ∧
  ¬(∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterexample_exists_l924_92411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_cost_per_minute_l924_92419

/-- Calculates the labor cost per minute given the total cost, part costs, and labor time. -/
theorem labor_cost_per_minute 
  (total_cost : ℝ)
  (part_cost : ℝ)
  (num_parts : ℕ)
  (labor_hours : ℝ)
  (h1 : total_cost = 220)
  (h2 : part_cost = 20)
  (h3 : num_parts = 2)
  (h4 : labor_hours = 6) :
  (total_cost - part_cost * (num_parts : ℝ)) / (labor_hours * 60) = 0.5 := by
  sorry

#check labor_cost_per_minute

end NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_cost_per_minute_l924_92419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l924_92423

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, 
    if a^2 * tan(B) = b^2 * tan(A), then the triangle is either isosceles or right-angled -/
theorem triangle_shape (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin B = b * Real.sin A →  -- Law of Sines
  a^2 * Real.tan B = b^2 * Real.tan A →
  (A = B) ∨ (A + B = π / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_shape_l924_92423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l924_92457

-- Define the conversion factor
noncomputable def pi_to_deg : ℝ := 180

-- Define the conversion functions
noncomputable def rad_to_deg (x : ℝ) : ℝ := x * pi_to_deg / Real.pi
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / pi_to_deg

-- State the theorem
theorem angle_conversion :
  (rad_to_deg (5/3 * Real.pi) = -300) ∧
  (deg_to_rad (-135) = -3/4 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_l924_92457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_distance_l924_92407

/-- The distance by which runner A beats runner B -/
noncomputable def distance_a_beats_b (distance : ℝ) (time_a : ℝ) (time_b : ℝ) : ℝ :=
  distance * (time_b / time_a - 1)

/-- Theorem stating the approximate distance by which A beats B -/
theorem a_beats_b_distance :
  let distance := (160 : ℝ)
  let time_a := (28 : ℝ)
  let time_b := (32 : ℝ)
  ∃ ε > 0, |distance_a_beats_b distance time_a time_b - 22.848| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_distance_l924_92407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_discount_price_l924_92452

/-- The final price of a laptop after applying a percentage discount. -/
noncomputable def final_price (original_price : ℝ) (discount_percentage : ℝ) : ℝ :=
  original_price * (1 - discount_percentage / 100)

/-- Theorem stating that a laptop originally priced at $800 with a 15% discount will cost $680. -/
theorem laptop_discount_price :
  final_price 800 15 = 680 := by
  -- Unfold the definition of final_price
  unfold final_price
  -- Simplify the arithmetic
  simp [mul_sub, mul_div_cancel']
  -- Check that the result is equal to 680
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_discount_price_l924_92452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l924_92464

theorem problem_statement (p q r : ℝ) 
  (h1 : ((p+q)*(q+r)*(r+p)) / (p*q*r) = 24)
  (h2 : ((p-2*q)*(q-2*r)*(r-2*p)) / (p*q*r) = 10) :
  ∃ (m n : ℕ), Int.gcd m n = 1 ∧ (p/q + q/r + r/p : ℝ) = m/n ∧ m = 35 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l924_92464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l924_92493

/-- A triangle with an incircle that evenly bisects a median -/
structure SpecialTriangle where
  -- Side lengths
  pq : ℝ
  qr : ℝ
  rp : ℝ
  -- Median length
  pm : ℝ
  -- Incircle radius
  r : ℝ
  -- Area components
  k : ℕ
  p : ℕ
  -- Conditions
  qr_eq : qr = 30
  incircle_bisects : pm = 2 * (pq - r)
  area_eq : pq * qr * rp = 4 * (k : ℝ) * (p : ℝ)
  p_not_square_divisible : ∀ (prime : ℕ), Nat.Prime prime → ¬(prime ^ 2 ∣ p)

/-- The main theorem -/
theorem special_triangle_sum (t : SpecialTriangle) : t.k + t.p = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_sum_l924_92493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_red_than_white_l924_92439

-- Define the number of red and white balls
def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_white_balls

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define the probability space
def Ω : Type := Fin die_sides × Fin total_balls

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define the event of drawing more red balls than white balls
def more_red_than_white : Set Ω := sorry

-- Theorem statement
theorem probability_more_red_than_white :
  P more_red_than_white = 19 / 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_more_red_than_white_l924_92439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l924_92460

/-- Calculates the simple interest rate given principal, time, and interest amount -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (interest : ℝ) : ℝ :=
  (interest * 100) / (principal * time)

/-- Theorem stating that the calculated interest rate is approximately 12.99% -/
theorem interest_rate_calculation (principal time interest : ℝ) 
  (h_principal : principal = 13846.153846153846)
  (h_time : time = 3)
  (h_interest : interest = 5400) :
  ∃ ε > 0, |calculate_interest_rate principal time interest - 12.99| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l924_92460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_circulation_ratio_l924_92444

/-- The ratio of magazine P's circulation in 1971 to its total circulation from 1971-1980 -/
theorem magazine_circulation_ratio : 
  ∀ (A : ℝ), A > 0 → 
  (4 * A) / ((4 * A) + (9 * A)) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magazine_circulation_ratio_l924_92444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_proof_l924_92476

/-- Given points P₁, P₂, and P in ℝ², prove that P has coordinates (-2, 11) -/
theorem point_coordinates_proof (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) →
  P₂ = (0, 5) →
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) →
  ‖P - P₁‖ = 2 * ‖P₂ - P‖ →
  P = (-2, 11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_proof_l924_92476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_combination_sum_l924_92421

theorem alternating_combination_sum (n : ℕ) (hn : 0 < n) :
  (Finset.range (n + 1)).sum (fun k => 
    ((-1 : ℤ) ^ (k + 1)) * (Nat.choose n k) * (Nat.choose (n * (n - k)) n)) = 
  ((-1 : ℤ) ^ (n + 1)) * (n ^ n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_combination_sum_l924_92421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l924_92453

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the theorem
theorem distance_to_y_axis 
  (M : ℝ × ℝ) 
  (h1 : parabola M.1 M.2) 
  (h2 : distance M focus = 10) : 
  M.1 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_y_axis_l924_92453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_difference_l924_92472

noncomputable def f (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g (x : ℝ) : ℝ := x / 2 + 3

noncomputable def h (x : ℝ) : ℝ := x^2 - 4

theorem function_composition_difference (x : ℝ) : 
  f (g (h x)) - h (g (f x)) = -2 * x^2 - 2 * x + 11 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_difference_l924_92472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speeds_satisfy_conditions_l924_92434

/-- The distance both cars travel -/
noncomputable def distance : ℝ := 120

/-- The time difference between the two cars -/
noncomputable def time_diff : ℝ := 18 / 60

/-- The speed reduction of the first car -/
noncomputable def speed_reduction : ℝ := 12

/-- The speed increase factor of the second car -/
noncomputable def speed_increase : ℝ := 1.1

/-- The speed of the first car -/
noncomputable def speed1 : ℝ := 100

/-- The speed of the second car -/
noncomputable def speed2 : ℝ := 80

/-- Theorem stating that the given speeds satisfy the problem conditions -/
theorem car_speeds_satisfy_conditions :
  (distance / speed1 = distance / speed2 - time_diff) ∧
  (distance / (speed1 - speed_reduction) = distance / (speed2 * speed_increase)) := by
  apply And.intro
  · sorry -- Proof for the first condition
  · sorry -- Proof for the second condition

#check car_speeds_satisfy_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speeds_satisfy_conditions_l924_92434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_distance_50_l924_92495

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the sequence of points A_n -/
def A : ℕ → Point
  | 0 => ⟨0, 0⟩
  | n + 1 => sorry  -- Definition based on the problem conditions

/-- Defines the sequence of points B_n -/
def B : ℕ → Point
  | n => sorry  -- x-coordinate to be determined, y = x²

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Checks if a triangle is equilateral -/
def isEquilateral (p q r : Point) : Prop :=
  distance p q = distance q r ∧ distance q r = distance r p

/-- The main theorem to be proved -/
theorem least_n_for_distance_50 :
  (∀ n : ℕ, n > 0 → isEquilateral (A (n-1)) (B n) (A n)) →
  (∀ n : ℕ, n > 0 → (B n).y = (B n).x^2) →
  (∀ n : ℕ, n > 0 → (A n).y = 0) →
  (∀ n : ℕ, n > 0 → ∀ m : ℕ, m > 0 → m ≠ n → A n ≠ A m) →
  (∀ n : ℕ, n > 0 → ∀ m : ℕ, m > 0 → m ≠ n → B n ≠ B m) →
  (∀ n : ℕ, n < 10 → distance (A 0) (A n) < 50) ∧
  distance (A 0) (A 10) ≥ 50 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_distance_50_l924_92495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_bounds_l924_92471

-- Define the parameter c
variable (c : ℝ)

-- Define the roots a and b of the quadratic equation
noncomputable def a (c : ℝ) : ℝ := 
  (-1 + Real.sqrt (1 - 4*c)) / 2

noncomputable def b (c : ℝ) : ℝ := 
  (-1 - Real.sqrt (1 - 4*c)) / 2

-- Define the distance function between the lines
noncomputable def distance (c : ℝ) : ℝ := 
  |a c - b c| / Real.sqrt 2

-- State the theorem
theorem parallel_lines_distance_bounds (h1 : 0 ≤ c) (h2 : c ≤ 1/8) :
  (distance c ≤ Real.sqrt 2 / 2) ∧ (distance c ≥ 1/2) ∧
  (∃ c', 0 ≤ c' ∧ c' ≤ 1/8 ∧ distance c' = Real.sqrt 2 / 2) ∧
  (∃ c'', 0 ≤ c'' ∧ c'' ≤ 1/8 ∧ distance c'' = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_bounds_l924_92471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l924_92482

noncomputable def a (x : ℝ) : ℝ := Real.log (Real.log x)
noncomputable def b (x : ℝ) : ℝ := Real.log (Real.log x) / Real.log 10
noncomputable def c (x : ℝ) : ℝ := Real.log (Real.log x / Real.log 10)
noncomputable def d (x : ℝ) : ℝ := (Real.log (Real.log x)) / Real.log 10

theorem order_of_logarithms (x : ℝ) (h : Real.exp 1 < x ∧ x < 10) :
  c x < d x ∧ d x < a x ∧ a x < b x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l924_92482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l924_92475

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem hyperbola_focus_distance 
  (h : Hyperbola) 
  (p f1 f2 : Point) 
  (h_eq : h.a = 3 ∧ h.b = 4) 
  (h_on : isOnHyperbola h p) 
  (h_f1 : distance p f1 = 3) 
  (h_left : f1.x < f2.x) : 
  distance p f2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l924_92475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_speed_l924_92429

/-- Calculates the highway speed given the total distance, local distance, local speed, and average speed of a trip -/
noncomputable def highway_speed (total_distance local_distance local_speed avg_speed : ℝ) : ℝ :=
  let total_time := total_distance / avg_speed
  let local_time := local_distance / local_speed
  let highway_distance := total_distance - local_distance
  let highway_time := total_time - local_time
  highway_distance / highway_time

/-- Theorem stating that given the specific conditions of the problem, the highway speed is 65 mph -/
theorem car_highway_speed :
  let total_distance : ℝ := 125
  let local_distance : ℝ := 60
  let local_speed : ℝ := 30
  let avg_speed : ℝ := 41.67
  highway_speed total_distance local_distance local_speed avg_speed = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_highway_speed_l924_92429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l924_92487

-- Define the ellipse parameters
noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the area of the rhombus
noncomputable def rhombus_area (a b : ℝ) : ℝ :=
  4 * a * b

-- Define the slope angle
noncomputable def slope_angle (k : ℝ) : ℝ :=
  Real.arctan k

-- Theorem statement
theorem ellipse_properties (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2) 
  (h4 : rhombus_area a b = 4) :
  ∃ (k : ℝ), 
    (ellipse_equation 2 1 = ellipse_equation a b) ∧ 
    ((slope_angle k = π / 4) ∨ (slope_angle k = 3 * π / 4)) ∧
    (∃ (x y : ℝ), 
      ellipse_equation a b x y ∧ 
      y = k * (x + a) ∧ 
      (x + a)^2 + y^2 = (4 * Real.sqrt 2 / 5)^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l924_92487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_event_total_l924_92408

def ticket_price (t : Char) : ℚ :=
  match t with
  | 'A' => 5/2
  | 'B' => 9/2
  | 'C' => 8
  | 'D' => 14
  | _ => 0

def tickets_sold (t : Char) : ℕ :=
  match t with
  | 'A' => 120
  | 'B' => 80
  | 'C' => 40
  | 'D' => 15
  | _ => 0

def donation_amounts : List ℚ := [20, 20, 20, 55, 55, 75, 95, 150]

def total_ticket_sales : ℚ :=
  List.sum (List.map (λ t => (ticket_price t) * (tickets_sold t)) ['A', 'B', 'C', 'D'])

def total_donations : ℚ :=
  List.sum donation_amounts

theorem charity_event_total :
  total_ticket_sales + total_donations = 1680 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charity_event_total_l924_92408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2048_is_identity_l924_92436

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.cos (Real.pi / 4),     0, -Real.sin (Real.pi / 4)],
    ![                    0,     1,                       0],
    ![Real.sin (Real.pi / 4),     0,  Real.cos (Real.pi / 4)]]

theorem B_power_2048_is_identity :
  B ^ 2048 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_power_2048_is_identity_l924_92436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l924_92433

/-- 
Given:
- P is the initial share price at the beginning of the year
- The share price increases by 30% in the first quarter
- The share price increases by 34.61538461538463% from the end of the first quarter to the end of the second quarter

Prove that the total percent increase from the beginning of the year to the end of the second quarter is approximately 75%.
-/
theorem share_price_increase (P : ℝ) : 
  let first_quarter_price := P * (1 + 0.30)
  let second_quarter_price := first_quarter_price * (1 + 0.3461538461538463)
  let total_increase_percent := (second_quarter_price / P - 1) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ |total_increase_percent - 75| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l924_92433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_lung_disease_relation_l924_92435

theorem smoking_lung_disease_relation (K_squared : Real) 
  (critical_value_95 : Real) (critical_value_99 : Real) :
  K_squared = 5.231 →
  critical_value_95 = 3.841 →
  critical_value_99 = 6.635 →
  ∃ confidence_level : Real,
    confidence_level > 0.95 ∧
    (K_squared ≥ critical_value_95 ∧ K_squared < critical_value_99) →
    True := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smoking_lung_disease_relation_l924_92435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_2012_l924_92468

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers in a range -/
def sumOfDigitsInRange (start finish : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of all numbers from 0 to 2012 is 28077 -/
theorem sum_of_digits_0_to_2012 :
  sumOfDigitsInRange 0 2012 = 28077 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_0_to_2012_l924_92468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_five_l924_92416

def sequenceList : List ℕ := [15, 17, 51, 53, 159, 161]

def sequence_rule (a b : ℕ) : Prop :=
  b = 3 * a ∨ b = 3 * a + 2

theorem first_number_is_five :
  ∃ (a : ℕ), a = 5 ∧
  (∀ i, i < sequenceList.length - 1 →
    sequence_rule (if i = 0 then a else sequenceList.get ⟨i, by sorry⟩)
                  (sequenceList.get ⟨i + 1, by sorry⟩)) :=
by sorry

#check first_number_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_five_l924_92416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_zeros_l924_92403

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3 / 2

theorem symmetry_and_zeros (ω : ℝ) (h_ω : ω > 0) :
  (∃ k : ℤ, ∀ x : ℝ, f ω x = f ω ((5 / 12) * Real.pi + k * Real.pi - x)) ∧
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi ∧ 
    f ω x₁ = 1 / 3 ∧ f ω x₂ = 1 / 3 ∧ 
    Real.cos (x₁ - x₂) = 1 / 3) := by
  sorry

#check symmetry_and_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_zeros_l924_92403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_roots_l924_92401

def quadraticEquation (l : ℝ) (x : ℂ) := (1 - Complex.I) * x^2 + (l + Complex.I) * x + (1 + Complex.I * l)

theorem two_complex_roots (l : ℝ) : 
  (∃ x y : ℂ, x ≠ y ∧ quadraticEquation l x = 0 ∧ quadraticEquation l y = 0) ↔ l ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_roots_l924_92401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_on_line_l924_92418

/-- Given two points P(a, b) and Q(c, d) on a line y = mx + k, 
    the magnitude of vector PQ is |a-c| √(1 + m²) -/
theorem vector_magnitude_on_line 
  (a b c d m k : ℝ) 
  (hP : b = m * a + k) 
  (hQ : d = m * c + k) : 
  ‖(c - a, d - b)‖ = |c - a| * Real.sqrt (1 + m^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_on_line_l924_92418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l924_92474

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.tan (Real.pi / 2 + α)) /
  (Real.cos (Real.pi - α) * Real.sin (3 * Real.pi - α) * (1 / Real.tan (-α))) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l924_92474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_roots_in_T_l924_92405

noncomputable def T : Set ℂ := {z : ℂ | Real.sqrt 3 / 2 ≤ z.re ∧ z.re ≤ 2 / Real.sqrt 3}

theorem smallest_m_for_nth_roots_in_T : 
  ∀ m : ℕ, (∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = 1) ↔ m ≥ 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_nth_roots_in_T_l924_92405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l924_92461

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  asymptote : ∀ x y : ℝ, y = 2 * x
  passes_through : (Real.sqrt 6)^2 / a^2 - 4^2 / b^2 = 1

/-- The equation of the hyperbola is x^2/2 - y^2/8 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, x^2 / 2 - y^2 / 8 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l924_92461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l924_92431

/-- Represents Mary's walking trip -/
structure MaryTrip where
  home_to_park_dist : ℝ
  park_to_school_dist : ℝ
  home_to_park_time : ℝ
  park_to_school_time : ℝ
  school_to_home_time : ℝ

/-- Calculate the average speed for Mary's round trip -/
noncomputable def averageSpeed (trip : MaryTrip) : ℝ :=
  let totalDist := 2 * (trip.home_to_park_dist + trip.park_to_school_dist)
  let totalTime := trip.home_to_park_time + trip.park_to_school_time + trip.school_to_home_time
  totalDist / (totalTime / 60)

/-- Theorem stating that Mary's average speed for the round trip is 3 km/hr -/
theorem mary_average_speed :
  ∀ (trip : MaryTrip),
    trip.home_to_park_dist = 1.5 →
    trip.park_to_school_dist = 0.5 →
    trip.home_to_park_time = 45 →
    trip.park_to_school_time = 15 →
    trip.school_to_home_time = 20 →
    averageSpeed trip = 3 := by
  intro trip h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l924_92431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l924_92481

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  h1 : base1 = 1
  h2 : base2 = 4

/-- The area of the isosceles trapezoid -/
noncomputable def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area (t : IsoscelesTrapezoid) :
  area t = (15 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l924_92481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_in_triangle_l924_92496

/-- Given a triangle ABC and a point P inside it, if the sum of vectors AP, BP, and CP is zero,
    and BD is one-third of BC, then AD + AP equals AB + (2/3)AC. -/
theorem vector_sum_in_triangle (A B C P D : EuclideanSpace ℝ (Fin 2)) :
  (P - A) + (P - B) + (P - C) = 0 →
  D - B = (1 / 3 : ℝ) • (C - B) →
  (D - A) + (P - A) = (B - A) + (2 / 3 : ℝ) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_in_triangle_l924_92496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_magnitude_l924_92442

/-- Sequence of complex numbers defined by z₁ = 0 and z_{n+1} = z_n^2 + i for all n ≥ 1 -/
noncomputable def z : ℕ → ℂ
  | 0 => 0
  | n + 1 => (z n)^2 + Complex.I

/-- The magnitude of z₁₁₁ is √2 -/
theorem z_111_magnitude : Complex.abs (z 111) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_111_magnitude_l924_92442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_surface_area_l924_92432

noncomputable def sphere_volume : ℝ := (256 * Real.pi) / 3

noncomputable def cube_edge_from_sphere_radius (r : ℝ) : ℝ := (2 * r) / Real.sqrt 3

theorem inscribed_cube_surface_area (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = sphere_volume) :
  6 * (cube_edge_from_sphere_radius r)^2 = 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_surface_area_l924_92432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l924_92420

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := { x : ℤ | -2 < x ∧ x ≤ 2 }

theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l924_92420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_inequality_l924_92427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x + a) / x

theorem f_max_value_and_inequality (a : ℝ) (n : ℕ) (hn : n ≥ 2) :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f a y ≤ f a x) ∧
  f a (Real.exp (1 - a)) = Real.exp (a - 1) ∧
  (List.range (n - 1)).foldr (λ i acc => acc + (List.range (i + 2)).foldl (λ prod j => prod * Real.log (j + 1)) 1 / (i + 3).factorial) 0 < (n - 1 : ℝ) / (2 * n + 2) := by
  sorry

#check f_max_value_and_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_and_inequality_l924_92427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_prime_floor_linear_l924_92492

open Nat Real

theorem no_always_prime_floor_linear (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ n : ℕ, ¬ Nat.Prime (⌊a * (n : ℝ) + b⌋).toNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_always_prime_floor_linear_l924_92492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l924_92448

-- Define the hyperbola parameters
variable (a b : ℝ)

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the condition for the circle passing through the right vertex
def circle_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) ∧
               (x + a)^2 + y^2 = (2*a)^2

-- State the theorem
theorem hyperbola_eccentricity (ha : a > 0) (hb : b > 0) 
  (h_circle : circle_condition a b) : eccentricity a b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l924_92448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_sum_l924_92465

theorem factory_output_sum (r : ℝ) (n : ℕ) (h1 : r = 1.1) (h2 : n = 5) :
  (Finset.range n).sum (fun i => r^(i + 1)) = (r / (r - 1)) * (r^n - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factory_output_sum_l924_92465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l924_92412

theorem solve_exponential_equation (x : ℝ) : 
  (27 : ℝ)^x * (27 : ℝ)^x * (27 : ℝ)^x = (243 : ℝ)^3 → x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l924_92412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_slices_per_container_l924_92437

theorem orange_slices_per_container : ∃ x : ℕ, 
  x > 0 ∧
  (332 % x = 0) ∧ 
  (329 < x * (332 / x)) ∧ 
  (x * (332 / x) ≤ 332) ∧
  (∀ y : ℕ, y > 0 → (332 % y = 0) ∧ (329 < y * (332 / y)) ∧ (y * (332 / y) ≤ 332) → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_slices_per_container_l924_92437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_over_n_l924_92400

def sequence_a : ℕ → ℚ
| 0 => 33
| n + 1 => sequence_a n + 2 * (n + 1)

theorem min_value_a_over_n :
  ∀ n : ℕ, n > 0 → sequence_a n / n ≥ 21/2 ∧ ∃ m : ℕ, m > 0 ∧ sequence_a m / m = 21/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_over_n_l924_92400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kristyna_number_l924_92406

theorem kristyna_number (k : ℕ) : 
  k % 2 = 1 →  -- k is odd
  k % 3 = 0 →  -- k is divisible by 3
  (∃ (a b c : ℕ), a + b + c = k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) →  -- k is the perimeter of a triangle with distinct integer sides
  (k / 2 : ℕ) + (k / 3 : ℕ) = 1681 →  -- sum of max longest and max shortest sides
  k = 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kristyna_number_l924_92406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_quadrants_I_II_IV_l924_92446

/-- A linear function f(x) = kx + b -/
structure LinearFunction where
  k : ℚ
  b : ℚ

/-- Quadrants of the Cartesian plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determine if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℚ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I  => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- The given linear function -/
def f : LinearFunction := { k := -1/2, b := 2 }

/-- Theorem: The linear function f(x) = -1/2x + 2 passes through Quadrants I, II, and IV -/
theorem passes_through_quadrants_I_II_IV :
  (∃ x y : ℚ, y = f.k * x + f.b ∧ inQuadrant x y Quadrant.I) ∧
  (∃ x y : ℚ, y = f.k * x + f.b ∧ inQuadrant x y Quadrant.II) ∧
  (∃ x y : ℚ, y = f.k * x + f.b ∧ inQuadrant x y Quadrant.IV) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_passes_through_quadrants_I_II_IV_l924_92446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l924_92424

def is_valid_number (n : ℕ) : Prop :=
  (∀ d, d ∈ n.digits 10 → d = 2 ∨ d = 3) ∧
  (n.digits 10).sum = 13

theorem largest_valid_number : 
  (is_valid_number 33332) ∧ 
  (∀ m : ℕ, is_valid_number m → m ≤ 33332) := by
  sorry

#check largest_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_number_l924_92424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_theorem_l924_92449

noncomputable def common_roots_product (A B : ℝ) : ℝ :=
  let eq1 := fun x => x^3 + 2*A*x + 20
  let eq2 := fun x => x^3 + 3*B*x^2 + 100
  let roots := {x : ℝ | eq1 x = 0 ∧ eq2 x = 0}
  Real.sqrt (Real.sqrt 2000)

theorem common_roots_theorem (A B : ℝ) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (x^3 + 2*A*x + 20 = 0) ∧ (x^3 + 3*B*x^2 + 100 = 0) ∧
    (y^3 + 2*A*y + 20 = 0) ∧ (y^3 + 3*B*y^2 + 100 = 0) →
  ∃ (k d m : ℕ), k > 0 ∧ d > 0 ∧ m > 0 ∧
    common_roots_product A B = k * (m ^ (1 / d : ℝ)) ∧
    k = 10 ∧ d = 3 ∧ m = 2 ∧
    k + d + m = 15 := by
  sorry

#eval Nat.succ 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_theorem_l924_92449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_makemyday_max_diff_invariant_problem_solution_l924_92441

-- Define the "Make-My-Day" procedure
def makemyday : ℤ × ℤ × ℤ → ℤ × ℤ × ℤ
| (a, b, c) => (b + c, a + c, a + b)

-- Define the maximum difference function
def max_diff : ℤ × ℤ × ℤ → ℤ
| (a, b, c) => max (max (abs (b - a)) (abs (c - b))) (abs (a - c))

-- Theorem statement
theorem makemyday_max_diff_invariant (triple : ℤ × ℤ × ℤ) (n : ℕ) :
  max_diff ((makemyday^[n]) triple) = max_diff triple :=
by sorry

-- Apply the theorem to the specific problem
theorem problem_solution (triple : ℤ × ℤ × ℤ) (h : triple = (20, 1, 8)) :
  max_diff ((makemyday^[2018]) triple) = 19 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_makemyday_max_diff_invariant_problem_solution_l924_92441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l924_92438

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

-- Theorem statement
theorem g_is_odd : ∀ x : ℝ, g x = -g (-x) := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l924_92438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_arrangement_l924_92451

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 49}

def IsValidArrangement (arr : List ℕ) : Prop :=
  arr.length > 0 ∧
  (∀ n, n ∈ arr → n ∈ S) ∧
  (∀ i, i < arr.length → (arr[i]! * arr[(i + 1) % arr.length]!) < 100)

theorem max_circle_arrangement :
  (∃ arr : List ℕ, IsValidArrangement arr ∧ arr.length = 18) ∧
  (∀ arr : List ℕ, IsValidArrangement arr → arr.length ≤ 18) := by
  sorry

#check max_circle_arrangement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circle_arrangement_l924_92451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l924_92494

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.cos α = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l924_92494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l924_92463

noncomputable def f (x : ℝ) := Real.sqrt ((x - 1) * (x - 2)) + Real.sqrt (x - 1)

theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | x = 1 ∨ x ≥ 2} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l924_92463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_same_exponents_l924_92456

/-- A Term is represented by its letters and exponents -/
structure Term where
  letters : Set Char
  exponents : Char → ℕ

/-- Definition of like terms -/
def like_terms (term1 term2 : Term) : Prop :=
  (term1.letters = term2.letters) ∧ (term1.exponents = term2.exponents)

/-- Theorem: For terms to be like terms, given the same letters, the exponents must be the same -/
theorem like_terms_same_exponents (t1 t2 : Term) :
  t1.letters = t2.letters → (like_terms t1 t2 ↔ t1.exponents = t2.exponents) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_same_exponents_l924_92456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l924_92445

-- Define the function f(x) = 2sin(x) - 1
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x - 1

-- State the theorem about the range of f
theorem f_range :
  Set.range f = Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l924_92445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_hyperbola_to_line_l924_92425

/-- The maximum distance between any point on the right branch of the hyperbola x^2 - y^2 = 1
    and the line x - y + 1 = 0 is √2/2 -/
theorem max_distance_hyperbola_to_line :
  ∃ (c : ℝ), c = Real.sqrt 2 / 2 ∧
    (∀ (x y : ℝ), x^2 - y^2 = 1 → x > 0 →
      ∀ (x' y' : ℝ), x' - y' + 1 = 0 →
        Real.sqrt ((x - x')^2 + (y - y')^2) ≤ c) ∧
    (∃ (x y x' y' : ℝ), x^2 - y^2 = 1 ∧ x > 0 ∧ x' - y' + 1 = 0 ∧
      Real.sqrt ((x - x')^2 + (y - y')^2) = c) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_hyperbola_to_line_l924_92425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l924_92478

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 20 + y^2 / 36 = 1

-- Define the focal length
noncomputable def focal_length : ℝ :=
  2 * Real.sqrt (36 - 20)

-- Theorem statement
theorem ellipse_focal_length :
  focal_length = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l924_92478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_126_l924_92491

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := 
  λ n ↦ a₁ * q^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_126 :
  ∀ n : ℕ, 
  (geometric_sequence 2 2 = λ k ↦ 2^k) →
  (geometric_sum 2 2 n = 126) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_126_l924_92491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_women_ratio_l924_92498

/-- Represents a co-ed softball team -/
structure Team where
  total_players : ℕ
  men : ℕ
  women : ℕ
  women_more_than_men : women = men + 4
  total_is_sum : total_players = men + women

/-- The ratio of men to women on the team is 2:3 -/
theorem men_to_women_ratio (team : Team) (h : team.total_players = 20) :
  (team.men : ℚ) / team.women = 2 / 3 := by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_to_women_ratio_l924_92498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_df_length_l924_92417

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  angle : ℝ

-- Define the problem setup
def triangle_ABC : Triangle := { a := 5, b := 4, angle := 1 }
def triangle_DEF : Triangle := { a := 2.5, b := 0, angle := 1 }

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.angle

-- Theorem statement
theorem equal_area_implies_df_length :
  area triangle_ABC = area triangle_DEF →
  triangle_DEF.b = 8 := by
  intro h
  -- The proof goes here, but we'll use sorry to skip it for now
  sorry

#check equal_area_implies_df_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_df_length_l924_92417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l924_92415

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := Real.cos θ + Real.sin θ

-- Define the line l in polar coordinates
noncomputable def line_l (θ : ℝ) : ℝ := (2 * Real.sqrt 2) / Real.cos (θ + Real.pi / 4)

-- Define the Cartesian equation of circle C
def circle_C_cartesian (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2

-- Define the Cartesian equation of line l
def line_l_cartesian (x y : ℝ) : Prop :=
  x - y = 4

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), circle_C_cartesian x y →
    ∀ (x' y' : ℝ), line_l_cartesian x' y' →
      Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l924_92415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_2_l924_92477

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x * k

-- State the theorem
theorem f_second_derivative_at_2 (k : ℝ) : 
  (deriv^[2] (fun x => f x k)) 2 = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_second_derivative_at_2_l924_92477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_notebook_purchase_l924_92430

/-- The greatest possible price of a notebook (in whole dollars) that Carl can afford --/
def max_notebook_price (total_budget : ℚ) (num_notebooks : ℕ) (entrance_fee : ℚ) (tax_rate : ℚ) : ℕ :=
  let remaining_budget := total_budget - entrance_fee
  let effective_budget := remaining_budget / (1 + tax_rate)
  (effective_budget / num_notebooks).floor.toNat

/-- Theorem stating the maximum notebook price Carl can afford --/
theorem carl_notebook_purchase :
  max_notebook_price 200 20 5 (3/100) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_notebook_purchase_l924_92430
