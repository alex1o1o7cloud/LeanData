import Mathlib

namespace NUMINAMATH_CALUDE_triangle_range_theorem_l2496_249628

theorem triangle_range_theorem (a b x : ℝ) (B : ℝ) (has_two_solutions : Prop) :
  a = x →
  b = 2 →
  B = π / 3 →
  has_two_solutions →
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_range_theorem_l2496_249628


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l2496_249641

theorem initial_alcohol_percentage
  (initial_volume : Real)
  (added_alcohol : Real)
  (final_percentage : Real)
  (h1 : initial_volume = 6)
  (h2 : added_alcohol = 1.2)
  (h3 : final_percentage = 50)
  (h4 : (initial_percentage / 100) * initial_volume + added_alcohol = 
        (final_percentage / 100) * (initial_volume + added_alcohol)) :
  initial_percentage = 40 := by
  sorry

#check initial_alcohol_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l2496_249641


namespace NUMINAMATH_CALUDE_infimum_of_expression_l2496_249687

theorem infimum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / (2 * a)) + (2 / b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ (1 / (2 * a₀)) + (2 / b₀) = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_infimum_of_expression_l2496_249687


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2496_249690

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2496_249690


namespace NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_intersections_l2496_249620

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a parabola y = x^2 -/
def Parabola := {p : Point | p.y = p.x^2}

/-- Checks if a point is on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Checks if a point is on the parabola y = x^2 -/
def onParabola (p : Point) : Prop := p.y = p.x^2

/-- Checks if two curves have coinciding tangents at a point -/
def coincidingTangents (p : Point) : Prop := sorry

/-- The main theorem -/
theorem tangents_not_necessarily_coincide_at_both_intersections
  (c : Circle) (A B : Point) :
  onCircle A c → onCircle B c →
  onParabola A → onParabola B →
  A ≠ B →
  coincidingTangents A →
  ¬ ∀ (c : Circle) (A B : Point),
    onCircle A c → onCircle B c →
    onParabola A → onParabola B →
    A ≠ B →
    coincidingTangents A →
    coincidingTangents B :=
by sorry

end NUMINAMATH_CALUDE_tangents_not_necessarily_coincide_at_both_intersections_l2496_249620


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2496_249661

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p / (q*r - 1) + q / (p*r - 1) + r / (p*q - 1) = 17/29 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2496_249661


namespace NUMINAMATH_CALUDE_A_intersect_B_l2496_249696

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2496_249696


namespace NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2496_249677

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (product_eq : x * y = 120) :
  x^2 - y^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_sum_and_product_l2496_249677


namespace NUMINAMATH_CALUDE_custom_op_solution_l2496_249644

/-- Custom operation "*" for positive integers -/
def custom_op (k n : ℕ+) : ℕ := (n : ℕ) * (2 * k + n - 1) / 2

/-- Theorem stating that if 3 * n = 150 using the custom operation, then n = 15 -/
theorem custom_op_solution :
  ∃ (n : ℕ+), custom_op 3 n = 150 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l2496_249644


namespace NUMINAMATH_CALUDE_preceding_binary_l2496_249601

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def M : List Bool := [true, false, true, false, true, false]

theorem preceding_binary (M : List Bool) : 
  M = [true, false, true, false, true, false] → 
  decimal_to_binary (binary_to_decimal M - 1) = [true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l2496_249601


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l2496_249666

theorem initial_number_of_girls :
  ∀ (n : ℕ) (A : ℝ),
  n > 0 →
  (n * (A + 3) - n * A = 94 - 70) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l2496_249666


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2496_249686

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 2 > 0} = {x : ℝ | x < -1 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2496_249686


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2496_249693

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : 
  (x + 1)^3 - 27 = 0 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2496_249693


namespace NUMINAMATH_CALUDE_six_balls_removal_ways_l2496_249604

/-- Represents the number of ways to remove n balls from a box, removing at least one at a time. -/
def removalWays (n : ℕ) : ℕ :=
  if n = 0 then 1
  else sorry  -- The actual implementation would go here

/-- The number of ways to remove 6 balls is 32. -/
theorem six_balls_removal_ways : removalWays 6 = 32 := by
  sorry  -- The proof would go here

end NUMINAMATH_CALUDE_six_balls_removal_ways_l2496_249604


namespace NUMINAMATH_CALUDE_prob_at_least_two_correct_l2496_249659

def num_questions : ℕ := 30
def num_guessed : ℕ := 5
def num_choices : ℕ := 6

def prob_correct : ℚ := 1 / num_choices
def prob_incorrect : ℚ := 1 - prob_correct

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem prob_at_least_two_correct :
  (1 : ℚ) - (binomial num_guessed 0 : ℚ) * prob_incorrect ^ num_guessed
          - (binomial num_guessed 1 : ℚ) * prob_correct * prob_incorrect ^ (num_guessed - 1)
  = 1526 / 7776 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_correct_l2496_249659


namespace NUMINAMATH_CALUDE_relationship_abc_l2496_249624

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.7 0.4
  let b : ℝ := Real.rpow 0.4 0.7
  let c : ℝ := Real.rpow 0.4 0.4
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2496_249624


namespace NUMINAMATH_CALUDE_min_value_trig_function_l2496_249618

theorem min_value_trig_function :
  let f : ℝ → ℝ := λ x ↦ 2 * (Real.cos x)^2 - Real.sin (2 * x)
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = 1 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l2496_249618


namespace NUMINAMATH_CALUDE_valid_regression_equation_l2496_249646

-- Define the linear regression equation
def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define the theorem
theorem valid_regression_equation :
  -- Conditions
  ∀ (x_mean y_mean : ℝ),
  x_mean = 3 →
  y_mean = 3.5 →
  -- The regression equation
  ∃ (a b : ℝ),
  -- Positive correlation
  a > 0 ∧
  -- Equation passes through (x_mean, y_mean)
  linear_regression a b x_mean = y_mean ∧
  -- Specific coefficients
  a = 0.4 ∧
  b = 2.3 :=
by
  sorry


end NUMINAMATH_CALUDE_valid_regression_equation_l2496_249646


namespace NUMINAMATH_CALUDE_max_profit_at_110_unique_max_profit_at_110_l2496_249691

/-- Represents the profit function for a new energy company -/
def profit (x : ℕ+) : ℚ :=
  if x < 100 then
    -1/2 * x^2 + 90 * x - 600
  else
    -2 * x - 24200 / x + 4100

/-- Theorem stating the maximum profit occurs at x = 110 -/
theorem max_profit_at_110 :
  ∀ x : ℕ+, profit x ≤ profit 110 ∧ profit 110 = 3660 := by
  sorry

/-- Theorem stating that 110 is the unique maximizer of the profit function -/
theorem unique_max_profit_at_110 :
  ∀ x : ℕ+, x ≠ 110 → profit x < profit 110 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_110_unique_max_profit_at_110_l2496_249691


namespace NUMINAMATH_CALUDE_village_cats_l2496_249613

theorem village_cats (total_cats : ℕ) 
  (striped_ratio : ℚ) (spotted_ratio : ℚ) 
  (fluffy_striped_ratio : ℚ) (fluffy_spotted_ratio : ℚ)
  (h_total : total_cats = 180)
  (h_striped : striped_ratio = 1/2)
  (h_spotted : spotted_ratio = 1/3)
  (h_fluffy_striped : fluffy_striped_ratio = 1/8)
  (h_fluffy_spotted : fluffy_spotted_ratio = 3/7) :
  ⌊striped_ratio * total_cats * fluffy_striped_ratio⌋ + 
  ⌊spotted_ratio * total_cats * fluffy_spotted_ratio⌋ = 36 := by
sorry

end NUMINAMATH_CALUDE_village_cats_l2496_249613


namespace NUMINAMATH_CALUDE_triangle_problem_l2496_249676

theorem triangle_problem (A B C : Real) (a b c : Real) :
  (a + c = 6) →
  (b = 2) →
  (Real.cos B = 7/9) →
  (a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A))) →
  (b = Real.sqrt (a^2 + c^2 - 2*a*c*(Real.cos B))) →
  (c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C))) →
  (Real.sin A / a = Real.sin B / b) →
  (Real.sin B / b = Real.sin C / c) →
  (A + B + C = Real.pi) →
  (a = 3 ∧ c = 3 ∧ Real.sin (A - B) = (10 * Real.sqrt 2) / 27) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2496_249676


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2496_249627

theorem greatest_three_digit_number : ∃ n : ℕ,
  n = 793 ∧
  100 ≤ n ∧ n < 1000 ∧
  ∃ k₁ : ℕ, n = 9 * k₁ + 1 ∧
  ∃ k₂ : ℕ, n = 5 * k₂ + 3 ∧
  ∃ k₃ : ℕ, n = 7 * k₃ + 2 ∧
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧
    ∃ l₁ : ℕ, m = 9 * l₁ + 1 ∧
    ∃ l₂ : ℕ, m = 5 * l₂ + 3 ∧
    ∃ l₃ : ℕ, m = 7 * l₃ + 2) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2496_249627


namespace NUMINAMATH_CALUDE_dictionary_cost_l2496_249662

theorem dictionary_cost (total_cost dinosaur_cost cookbook_cost : ℕ) 
  (h1 : total_cost = 37)
  (h2 : dinosaur_cost = 19)
  (h3 : cookbook_cost = 7) :
  total_cost - dinosaur_cost - cookbook_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_dictionary_cost_l2496_249662


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2496_249606

theorem complex_product_magnitude : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2496_249606


namespace NUMINAMATH_CALUDE_gift_wrapping_l2496_249616

theorem gift_wrapping (total_rolls total_gifts first_roll_gifts second_roll_gifts : ℕ) :
  total_rolls = 3 →
  total_gifts = 12 →
  first_roll_gifts = 3 →
  second_roll_gifts = 5 →
  total_gifts = first_roll_gifts + second_roll_gifts + (total_gifts - (first_roll_gifts + second_roll_gifts)) →
  (total_gifts - (first_roll_gifts + second_roll_gifts)) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_l2496_249616


namespace NUMINAMATH_CALUDE_monomial_combination_l2496_249625

theorem monomial_combination (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), ∃ (m n : ℤ), 3 * a^(7*x) * b^(y+7) = k * a^m * b^n ∧ 
                           -7 * a^(2-4*y) * b^(2*x) = k * a^m * b^n) → 
  x + y = -1 := by sorry

end NUMINAMATH_CALUDE_monomial_combination_l2496_249625


namespace NUMINAMATH_CALUDE_line_circle_intersection_m_values_l2496_249681

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter m in the line equation x - y + m = 0 -/
  m : ℝ
  /-- The line intersects the circle x^2 + y^2 = 4 at two points -/
  intersects : ∃ (A B : ℝ × ℝ), A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4 ∧
                                 A.1 - A.2 + m = 0 ∧ B.1 - B.2 + m = 0
  /-- The length of the chord AB is 2√3 -/
  chord_length : ∃ (A B : ℝ × ℝ), (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12

/-- The theorem stating the possible values of m -/
theorem line_circle_intersection_m_values (lci : LineCircleIntersection) :
  lci.m = Real.sqrt 2 ∨ lci.m = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_m_values_l2496_249681


namespace NUMINAMATH_CALUDE_sqrt_ab_equals_18_l2496_249631

theorem sqrt_ab_equals_18 (a b : ℝ) : 
  a = Real.log 9 / Real.log 4 → 
  b = 108 * (Real.log 8 / Real.log 3) → 
  Real.sqrt (a * b) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ab_equals_18_l2496_249631


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_16_adults_l2496_249650

-- Define the problem parameters
def total_cans : ℕ := 8
def adults_per_can : ℕ := 4
def children_per_can : ℕ := 6
def children_fed : ℕ := 24

-- Theorem statement
theorem remaining_soup_feeds_16_adults :
  ∃ (cans_for_children : ℕ) (remaining_cans : ℕ),
    cans_for_children * children_per_can = children_fed ∧
    remaining_cans = total_cans - cans_for_children ∧
    remaining_cans * adults_per_can = 16 :=
by sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_16_adults_l2496_249650


namespace NUMINAMATH_CALUDE_tangent_lines_at_P_l2496_249603

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point P
def P : ℝ × ℝ := (2, -6)

-- Define the two potential tangent lines
def line1 (x y : ℝ) : Prop := 3*x + y = 0
def line2 (x y : ℝ) : Prop := 24*x - y - 54 = 0

-- Theorem statement
theorem tangent_lines_at_P :
  (∃ t : ℝ, f t = P.2 ∧ f' t * (P.1 - t) = P.2 - f t ∧ (line1 P.1 P.2 ∨ line2 P.1 P.2)) ∧
  (∀ x y : ℝ, (line1 x y ∨ line2 x y) → ∃ t : ℝ, f t = y ∧ f' t * (x - t) = y - f t) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_at_P_l2496_249603


namespace NUMINAMATH_CALUDE_expression_value_l2496_249651

theorem expression_value
  (x y z w : ℝ)
  (eq1 : 4 * x * z + y * w = 3)
  (eq2 : x * w + y * z = 6) :
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2496_249651


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2496_249669

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  (a = 4 ∧ b = 8 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 8 ∧ c = 4) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ a + c > b →  -- triangle inequality
  a + b + c = 20 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2496_249669


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2496_249671

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 42*x + 400 ≤ 16 ↔ 16 ≤ x ∧ x ≤ 24 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2496_249671


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l2496_249698

/-- Represents a hyperbola with equation ax² - by² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → a * x^2 - b * y^2 = 1

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def has_focus (h : Hyperbola) (p : Point) : Prop :=
  ∃ c : ℝ, c^2 = 1 / h.a + 1 / h.b ∧ p.y^2 = c^2

def has_same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  h1.a / h1.b = h2.a / h2.b

theorem hyperbola_equation_correct (h1 h2 : Hyperbola) (p : Point) :
  h1.a = 1/24 ∧ h1.b = 1/12 ∧
  h2.a = 1/2 ∧ h2.b = 1 ∧
  p.x = 0 ∧ p.y = 6 ∧
  has_focus h1 p ∧
  has_same_asymptotes h1 h2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l2496_249698


namespace NUMINAMATH_CALUDE_expected_length_is_six_l2496_249674

/-- The probability of appending H -/
def pH : ℚ := 1/4

/-- The probability of appending M -/
def pM : ℚ := 1/2

/-- The probability of appending T -/
def pT : ℚ := 1/4

/-- The expected length of the string until MM appears -/
noncomputable def expectedLength : ℚ := 6

/-- Theorem stating that the expected length until MM appears is 6 -/
theorem expected_length_is_six :
  pH + pM + pT = 1 →
  expectedLength = 6 := by sorry

end NUMINAMATH_CALUDE_expected_length_is_six_l2496_249674


namespace NUMINAMATH_CALUDE_trouser_cost_calculation_final_cost_is_correct_l2496_249663

/-- Calculate the final cost in GBP for three trousers with given prices, discounts, taxes, and fees -/
theorem trouser_cost_calculation (price1 price2 price3 : ℝ) 
  (discount1 discount2 discount3 : ℝ) (global_discount : ℝ) 
  (sales_tax handling_fee conversion_rate : ℝ) : ℝ :=
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := price2 * (1 - discount2)
  let discounted_price3 := price3 * (1 - discount3)
  let total_discounted := discounted_price1 + discounted_price2 + discounted_price3
  let after_global_discount := total_discounted * (1 - global_discount)
  let after_tax := after_global_discount * (1 + sales_tax)
  let final_usd := after_tax + 3 * handling_fee
  let final_gbp := final_usd * conversion_rate
  final_gbp

/-- The final cost in GBP for the given trouser prices and conditions is £271.87 -/
theorem final_cost_is_correct : 
  trouser_cost_calculation 100 150 200 0.20 0.15 0.25 0.10 0.08 5 0.75 = 271.87 := by
  sorry


end NUMINAMATH_CALUDE_trouser_cost_calculation_final_cost_is_correct_l2496_249663


namespace NUMINAMATH_CALUDE_football_team_progress_l2496_249611

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 8 = 3 := by sorry

end NUMINAMATH_CALUDE_football_team_progress_l2496_249611


namespace NUMINAMATH_CALUDE_trig_identity_l2496_249617

theorem trig_identity (α : Real) 
  (h : Real.sin α - Real.cos α = -7/5) : 
  (Real.sin α * Real.cos α = -12/25) ∧ 
  ((Real.tan α = -3/4) ∨ (Real.tan α = -4/3)) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2496_249617


namespace NUMINAMATH_CALUDE_trapezium_height_l2496_249655

theorem trapezium_height (a b area : ℝ) (ha : a = 20) (hb : b = 18) (harea : area = 247) :
  (2 * area) / (a + b) = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l2496_249655


namespace NUMINAMATH_CALUDE_fifth_employee_speed_correct_l2496_249653

/-- Calculates the typing speed of the 5th employee given the team's average and the typing speeds of the other 4 employees. -/
def calculate_fifth_employee_speed (team_size : Nat) (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) : Nat :=
  team_size * team_average - (employee1_speed + employee2_speed + employee3_speed + employee4_speed)

/-- Theorem stating that the calculated speed of the 5th employee is correct given the team's average and the speeds of the other 4 employees. -/
theorem fifth_employee_speed_correct (team_average : Nat) (employee1_speed : Nat) (employee2_speed : Nat) (employee3_speed : Nat) (employee4_speed : Nat) :
  let team_size : Nat := 5
  let fifth_employee_speed := calculate_fifth_employee_speed team_size team_average employee1_speed employee2_speed employee3_speed employee4_speed
  (employee1_speed + employee2_speed + employee3_speed + employee4_speed + fifth_employee_speed) / team_size = team_average :=
by
  sorry

#eval calculate_fifth_employee_speed 5 80 64 76 91 80

end NUMINAMATH_CALUDE_fifth_employee_speed_correct_l2496_249653


namespace NUMINAMATH_CALUDE_sum_of_integers_l2496_249647

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2496_249647


namespace NUMINAMATH_CALUDE_sum_abc_l2496_249685

theorem sum_abc (a b c : ℝ) 
  (h1 : a - (b + c) = 16)
  (h2 : a^2 - (b + c)^2 = 1664) : 
  a + b + c = 104 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_l2496_249685


namespace NUMINAMATH_CALUDE_club_members_count_l2496_249609

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 8

/-- The total cost of apparel for all members in dollars -/
def total_cost : ℕ := 4440

/-- The number of pairs of socks each member needs -/
def socks_per_member : ℕ := 1

/-- The number of T-shirts each member needs -/
def tshirts_per_member : ℕ := 2

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + tshirt_additional_cost

/-- The cost of apparel for one member in dollars -/
def cost_per_member : ℕ := sock_cost * socks_per_member + tshirt_cost * tshirts_per_member

/-- The number of members in the club -/
def club_members : ℕ := total_cost / cost_per_member

theorem club_members_count : club_members = 130 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l2496_249609


namespace NUMINAMATH_CALUDE_min_ab_in_triangle_l2496_249622

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if 2c cos B = 2a + b and the area S = √3 c, then ab ≥ 48. -/
theorem min_ab_in_triangle (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 * c →
  ab ≥ 48 := by
sorry

end NUMINAMATH_CALUDE_min_ab_in_triangle_l2496_249622


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2496_249630

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h * c = 2 * (a * b / 2) →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2496_249630


namespace NUMINAMATH_CALUDE_function_property_l2496_249695

theorem function_property (f : ℕ → ℤ) (k : ℤ) 
  (h1 : f 2006 = 2007)
  (h2 : ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)) :
  k = 0 ∨ k = -1 := by
sorry

end NUMINAMATH_CALUDE_function_property_l2496_249695


namespace NUMINAMATH_CALUDE_largest_in_set_l2496_249634

theorem largest_in_set (a : ℝ) (h : a = -3) :
  let S : Set ℝ := {-2*a, 3*a, 18/a, a^3, 2}
  ∀ x ∈ S, -2*a ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_in_set_l2496_249634


namespace NUMINAMATH_CALUDE_sqrt_17_minus_1_gt_3_l2496_249679

theorem sqrt_17_minus_1_gt_3 : Real.sqrt 17 - 1 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_17_minus_1_gt_3_l2496_249679


namespace NUMINAMATH_CALUDE_circumscribed_polygon_has_triangle_l2496_249672

/-- A polygon circumscribed about a circle. -/
structure CircumscribedPolygon where
  /-- The number of sides in the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- All side lengths are positive. -/
  sides_pos : ∀ i, 0 < sides i

/-- Theorem: In any polygon circumscribed about a circle, 
    there exist three sides that can form a triangle. -/
theorem circumscribed_polygon_has_triangle (P : CircumscribedPolygon) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    P.sides i + P.sides j > P.sides k ∧
    P.sides j + P.sides k > P.sides i ∧
    P.sides k + P.sides i > P.sides j :=
sorry

end NUMINAMATH_CALUDE_circumscribed_polygon_has_triangle_l2496_249672


namespace NUMINAMATH_CALUDE_complement_of_union_equals_set_l2496_249688

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3}
def N : Set Nat := {1, 2}

theorem complement_of_union_equals_set (U M N : Set Nat) 
  (hU : U = {1, 2, 3, 4, 5}) 
  (hM : M = {1, 3}) 
  (hN : N = {1, 2}) : 
  (M ∪ N)ᶜ = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_set_l2496_249688


namespace NUMINAMATH_CALUDE_larger_number_proof_l2496_249660

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 5382) →
  (∃ (x y : ℕ+), x * y = 234 ∧ (x = 13 ∨ x = 18) ∧ (y = 13 ∨ y = 18)) →
  (max a b = 414) := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2496_249660


namespace NUMINAMATH_CALUDE_water_flow_fraction_l2496_249680

/-- Given a water flow problem with the following conditions:
  * The original flow rate is 5 gallons per minute
  * The reduced flow rate is 2 gallons per minute
  * The reduced flow rate is 1 gallon per minute less than a fraction of the original flow rate
  Prove that the fraction of the original flow rate is 3/5 -/
theorem water_flow_fraction (original_rate reduced_rate : ℚ) 
  (h1 : original_rate = 5)
  (h2 : reduced_rate = 2) :
  ∃ f : ℚ, f * original_rate - 1 = reduced_rate ∧ f = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_fraction_l2496_249680


namespace NUMINAMATH_CALUDE_line_symmetry_l2496_249643

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a third line -/
def symmetric (l1 l2 ls : Line) : Prop :=
  ∀ x y : ℝ, l1.contains x y → 
    ∃ x' y' : ℝ, l2.contains x' y' ∧
      (x + x') / 2 = (y + y') / 2 ∧ ls.contains ((x + x') / 2) ((y + y') / 2)

theorem line_symmetry :
  let l1 : Line := ⟨-2, 1, 1⟩  -- y = 2x + 1
  let l2 : Line := ⟨1, -2, 0⟩  -- x - 2y = 0
  let ls : Line := ⟨1, 1, 1⟩  -- x + y + 1 = 0
  symmetric l1 l2 ls := by sorry

end NUMINAMATH_CALUDE_line_symmetry_l2496_249643


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2496_249648

def original_earnings : ℚ := 60
def new_earnings : ℚ := 80

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2496_249648


namespace NUMINAMATH_CALUDE_lori_marble_sharing_l2496_249600

def total_marbles : ℕ := 30
def marbles_per_friend : ℕ := 6

theorem lori_marble_sharing :
  total_marbles / marbles_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_lori_marble_sharing_l2496_249600


namespace NUMINAMATH_CALUDE_gp_sum_equality_l2496_249615

/-- Given two geometric progressions (GPs) where the sum of 3n terms of the first GP
    equals the sum of n terms of the second GP, prove that the first term of the second GP
    equals the sum of the first three terms of the first GP. -/
theorem gp_sum_equality (a b q : ℝ) (n : ℕ) (h_q_ne_one : q ≠ 1) :
  a * (q^(3*n) - 1) / (q - 1) = b * (q^(3*n) - 1) / (q^3 - 1) →
  b = a * (1 + q + q^2) :=
by sorry

end NUMINAMATH_CALUDE_gp_sum_equality_l2496_249615


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l2496_249610

/-- The number of green lava lamps -/
def green_lamps : ℕ := 4

/-- The number of purple lava lamps -/
def purple_lamps : ℕ := 4

/-- The total number of lamps -/
def total_lamps : ℕ := green_lamps + purple_lamps

/-- The number of lamps in each row -/
def lamps_per_row : ℕ := 4

/-- The number of rows -/
def num_rows : ℕ := 2

/-- The number of lamps turned on -/
def lamps_on : ℕ := 4

/-- The probability of the specific arrangement -/
def specific_arrangement_probability : ℚ := 1 / 7

theorem lava_lamp_probability :
  (green_lamps = 4) →
  (purple_lamps = 4) →
  (total_lamps = green_lamps + purple_lamps) →
  (lamps_per_row = 4) →
  (num_rows = 2) →
  (lamps_on = 4) →
  (specific_arrangement_probability = 1 / 7) := by
  sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l2496_249610


namespace NUMINAMATH_CALUDE_sports_league_games_l2496_249633

/-- The number of games in a complete season for a sports league -/
def total_games (n : ℕ) (d : ℕ) (t : ℕ) (s : ℕ) (c : ℕ) : ℕ :=
  (n * (d - 1) * s + n * t * c) / 2

/-- Theorem: The total number of games in the given sports league is 296 -/
theorem sports_league_games :
  total_games 8 8 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l2496_249633


namespace NUMINAMATH_CALUDE_work_completion_time_l2496_249640

theorem work_completion_time (a b : ℕ) (h1 : a + b = 1/12) (h2 : a = 1/20) : b = 1/30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2496_249640


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l2496_249675

theorem circle_radius_three_inches 
  (r : ℝ) 
  (h : r > 0) 
  (h_eq : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : 
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l2496_249675


namespace NUMINAMATH_CALUDE_derivative_limit_theorem_l2496_249683

open Real

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define x₀ as a real number
variable (x₀ : ℝ)

-- State the theorem
theorem derivative_limit_theorem (h : HasDerivAt f (-3) x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, |h| < δ →
    |((f (x₀ + h) - f (x₀ - 3 * h)) / h) - (-12)| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_limit_theorem_l2496_249683


namespace NUMINAMATH_CALUDE_composite_iff_on_line_count_lines_l2496_249639

def S (a b : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + (a * b + a) * p.2 - b - 1 = 0}

def A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ a : ℕ, p = (0, 1 / (a : ℝ))}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ b : ℕ, p = ((b + 1 : ℝ), 0)}

def M (m : ℕ) : ℝ × ℝ := (m, -1)

def τ (m : ℕ) : ℕ := (Nat.divisors m).card

theorem composite_iff_on_line (m : ℕ) :
  (∃ a b : ℕ, m = (a + 1) * (b + 1)) ↔
  (∃ a b : ℕ, M m ∈ S a b) :=
sorry

theorem count_lines (m : ℕ) :
  (Nat.card {p : ℕ × ℕ | m = (p.1 + 1) * (p.2 + 1)}) = τ m - 2 :=
sorry

end NUMINAMATH_CALUDE_composite_iff_on_line_count_lines_l2496_249639


namespace NUMINAMATH_CALUDE_total_berries_picked_l2496_249678

theorem total_berries_picked (total : ℕ) : 
  (total / 2 : ℚ) + (total / 3 : ℚ) + 7 = total → total = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_total_berries_picked_l2496_249678


namespace NUMINAMATH_CALUDE_circular_sector_angle_l2496_249607

/-- Given a circular sector with arc length 30 and diameter 16, 
    prove that its central angle in radians is 15/4 -/
theorem circular_sector_angle (arc_length : ℝ) (diameter : ℝ) 
  (h1 : arc_length = 30) (h2 : diameter = 16) :
  arc_length / (diameter / 2) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circular_sector_angle_l2496_249607


namespace NUMINAMATH_CALUDE_truffle_price_is_one_fifty_l2496_249623

/-- Represents the revenue and sales data of a candy store --/
structure CandyStore where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  pretzelsDozens : ℕ
  pretzelPrice : ℚ
  totalRevenue : ℚ

/-- Calculates the price of a single chocolate truffle --/
def trufflePrice (store : CandyStore) : ℚ :=
  let fudgeRevenue := store.fudgePounds * store.fudgePrice
  let pretzelsCount := store.pretzelsDozens * 12
  let pretzelsRevenue := pretzelsCount * store.pretzelPrice
  let trufflesRevenue := store.totalRevenue - fudgeRevenue - pretzelsRevenue
  let trufflesCount := store.trufflesDozens * 12
  trufflesRevenue / trufflesCount

/-- Theorem stating that the price of each chocolate truffle is $1.50 --/
theorem truffle_price_is_one_fifty (store : CandyStore)
  (h1 : store.fudgePounds = 20)
  (h2 : store.fudgePrice = 5/2)
  (h3 : store.trufflesDozens = 5)
  (h4 : store.pretzelsDozens = 3)
  (h5 : store.pretzelPrice = 2)
  (h6 : store.totalRevenue = 212) :
  trufflePrice store = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_truffle_price_is_one_fifty_l2496_249623


namespace NUMINAMATH_CALUDE_cube_of_difference_l2496_249699

theorem cube_of_difference (a b : ℝ) 
  (h1 : a - b = 8) 
  (h2 : a^2 + b^2 = 98) : 
  (a - b)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_difference_l2496_249699


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2496_249612

/-- The function f(x) = |x - a| is increasing on [-3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ici (-3) → y ∈ Set.Ici (-3) → x ≤ y → |x - a| ≤ |y - a|

/-- "a = -3" is a sufficient condition -/
theorem sufficient_condition (a : ℝ) (h : a = -3) : is_increasing_on_interval a :=
sorry

/-- "a = -3" is not a necessary condition -/
theorem not_necessary_condition : ∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a :=
sorry

/-- "a = -3" is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -3 → is_increasing_on_interval a) ∧
  (∃ a : ℝ, a ≠ -3 ∧ is_increasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2496_249612


namespace NUMINAMATH_CALUDE_largest_of_three_l2496_249697

theorem largest_of_three (a b c : ℝ) : 
  let x₁ := a
  let x₂ := if b > x₁ then b else x₁
  let x₃ := if c > x₂ then c else x₂
  x₃ = max a (max b c) := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_l2496_249697


namespace NUMINAMATH_CALUDE_positive_solution_of_equation_l2496_249657

theorem positive_solution_of_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1/x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_of_equation_l2496_249657


namespace NUMINAMATH_CALUDE_last_installment_value_is_3300_l2496_249664

/-- Represents the payment structure for a TV set purchase -/
structure TVPayment where
  price : ℕ               -- Price of the TV set in Rupees
  num_installments : ℕ    -- Number of installments
  installment_amount : ℕ  -- Amount of each installment in Rupees
  interest_rate : ℚ       -- Annual interest rate as a rational number
  processing_fee : ℕ      -- Processing fee in Rupees

/-- Calculates the value of the last installment for a TV payment plan -/
def last_installment_value (payment : TVPayment) : ℕ :=
  payment.installment_amount + payment.processing_fee

/-- Theorem stating that the last installment value for the given TV payment plan is 3300 Rupees -/
theorem last_installment_value_is_3300 (payment : TVPayment) 
  (h1 : payment.price = 35000)
  (h2 : payment.num_installments = 36)
  (h3 : payment.installment_amount = 2300)
  (h4 : payment.interest_rate = 9 / 100)
  (h5 : payment.processing_fee = 1000) :
  last_installment_value payment = 3300 := by
  sorry

#eval last_installment_value { 
  price := 35000, 
  num_installments := 36, 
  installment_amount := 2300, 
  interest_rate := 9 / 100, 
  processing_fee := 1000 
}

end NUMINAMATH_CALUDE_last_installment_value_is_3300_l2496_249664


namespace NUMINAMATH_CALUDE_equation_solution_l2496_249629

theorem equation_solution (a b : ℝ) (x₁ x₂ x₃ : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₃ = b →
  a + b = 144 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2496_249629


namespace NUMINAMATH_CALUDE_min_value_of_a_l2496_249656

theorem min_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (ha : a > 0) (hxy : x ≠ y)
  (h : (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)) :
  a ≥ 1 / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l2496_249656


namespace NUMINAMATH_CALUDE_sand_art_total_grams_l2496_249658

-- Define constants for the dimensions and conversion rate
def rect_length : ℝ := 6
def rect_width : ℝ := 7
def rect_depth : ℝ := 2

def square_side : ℝ := 5
def square_depth : ℝ := 3

def circle_diameter : ℝ := 4
def circle_depth : ℝ := 1.5

def conversion_rate : ℝ := 3

-- Define the theorem
theorem sand_art_total_grams :
  let rect_volume := rect_length * rect_width * rect_depth
  let square_volume := square_side * square_side * square_depth
  let circle_area := π * (circle_diameter / 2) ^ 2
  let circle_volume := circle_area * circle_depth
  let total_volume := rect_volume + square_volume + circle_volume
  let total_grams := total_volume * conversion_rate
  ∃ ε > 0, |total_grams - 533.55| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_sand_art_total_grams_l2496_249658


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l2496_249605

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_composition_fixed_points :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l2496_249605


namespace NUMINAMATH_CALUDE_evaluate_expression_l2496_249689

theorem evaluate_expression (x y z : ℚ) : 
  x = 1/4 → y = 3/4 → z = 3 → x^2 * y^3 * z = 81/1024 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2496_249689


namespace NUMINAMATH_CALUDE_collinear_vectors_k_value_l2496_249667

/-- Given two non-collinear vectors in a real vector space, 
    if certain conditions are met, then k = -8. -/
theorem collinear_vectors_k_value 
  (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e₁ e₂ : V) (k : ℝ) 
  (h_non_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂) 
  (AB CB CD : V) 
  (h_AB : AB = 2 • e₁ + k • e₂) 
  (h_CB : CB = e₁ + 3 • e₂) 
  (h_CD : CD = 2 • e₁ - e₂) 
  (h_collinear : ∃ (t : ℝ), AB = t • (CD - CB)) : 
  k = -8 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_k_value_l2496_249667


namespace NUMINAMATH_CALUDE_max_toys_theorem_l2496_249670

def max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (tax_rate : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_game_cost := game_cost * (1 + tax_rate)
  let remaining_money := initial_amount - total_game_cost
  (remaining_money / toy_cost).floor.toNat

theorem max_toys_theorem :
  max_toys_purchasable 57 27 (8/100) 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_toys_theorem_l2496_249670


namespace NUMINAMATH_CALUDE_heptagon_angle_sums_l2496_249652

/-- A heptagon is a polygon with 7 sides -/
def Heptagon : Nat := 7

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : Nat) : ℝ := (n - 2) * 180

/-- The sum of exterior angles of any polygon -/
def sum_exterior_angles : ℝ := 360

theorem heptagon_angle_sums :
  (sum_interior_angles Heptagon = 900) ∧ (sum_exterior_angles = 360) := by
  sorry

#check heptagon_angle_sums

end NUMINAMATH_CALUDE_heptagon_angle_sums_l2496_249652


namespace NUMINAMATH_CALUDE_units_digit_34_pow_30_l2496_249608

theorem units_digit_34_pow_30 : (34^30) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_34_pow_30_l2496_249608


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2496_249637

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 15 ∧ r₁ * r₂ = 6 ∧ r₁^2 + r₂^2 = 213 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2496_249637


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l2496_249626

theorem larger_solution_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 → x ≤ 11 ∧ (∃ y, y^2 - 9*y - 22 = 0 ∧ y ≠ x) :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l2496_249626


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2496_249621

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + 4*x - 5 > 0} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2496_249621


namespace NUMINAMATH_CALUDE_blue_balls_removed_l2496_249649

theorem blue_balls_removed (initial_total : Nat) (initial_blue : Nat) (final_probability : Rat) :
  initial_total = 25 →
  initial_blue = 9 →
  final_probability = 1/5 →
  ∃ (removed : Nat), 
    removed ≤ initial_blue ∧
    (initial_blue - removed : Rat) / (initial_total - removed : Rat) = final_probability ∧
    removed = 5 :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l2496_249649


namespace NUMINAMATH_CALUDE_parallelogram_height_l2496_249665

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) 
    (h1 : area = 384) 
    (h2 : base = 24) 
    (h3 : area = base * height) : height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l2496_249665


namespace NUMINAMATH_CALUDE_perfect_square_addition_l2496_249638

theorem perfect_square_addition : ∃ x : ℤ,
  (∃ a : ℤ, 100 + x = a^2) ∧
  (∃ b : ℤ, 164 + x = b^2) ∧
  x = 125 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_l2496_249638


namespace NUMINAMATH_CALUDE_sequence_expression_l2496_249684

theorem sequence_expression (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^(n - 1)) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_expression_l2496_249684


namespace NUMINAMATH_CALUDE_laura_biathlon_l2496_249635

/-- Laura's biathlon training problem -/
theorem laura_biathlon (x : ℝ) : x > 0 → (25 / (3*x + 2) + 4 / x + 8/60 = 140/60) → (6.6*x^2 - 32.6*x - 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_laura_biathlon_l2496_249635


namespace NUMINAMATH_CALUDE_rectangle_area_l2496_249619

/-- Given a rectangle where the length is four times the width and the perimeter is 250 cm,
    prove that its area is 2500 square centimeters. -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 250 → l * w = 2500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2496_249619


namespace NUMINAMATH_CALUDE_subtract_inequality_preserves_order_l2496_249668

theorem subtract_inequality_preserves_order (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_preserves_order_l2496_249668


namespace NUMINAMATH_CALUDE_combined_population_theorem_l2496_249654

def wellington_population : ℕ := 900

def port_perry_population (wellington : ℕ) : ℕ := 7 * wellington

def lazy_harbor_population (port_perry : ℕ) : ℕ := port_perry - 800

theorem combined_population_theorem (wellington : ℕ) (port_perry : ℕ) (lazy_harbor : ℕ) :
  wellington = wellington_population →
  port_perry = port_perry_population wellington →
  lazy_harbor = lazy_harbor_population port_perry →
  port_perry + lazy_harbor = 11800 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_population_theorem_l2496_249654


namespace NUMINAMATH_CALUDE_least_days_to_double_l2496_249673

/-- The least number of days for a loan to double with daily compound interest -/
theorem least_days_to_double (principal : ℝ) (rate : ℝ) (n : ℕ) : 
  principal > 0 → 
  rate > 0 → 
  principal * (1 + rate) ^ n ≥ 2 * principal → 
  principal * (1 + rate) ^ (n - 1) < 2 * principal → 
  principal = 20 → 
  rate = 0.1 → 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_least_days_to_double_l2496_249673


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2496_249614

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2496_249614


namespace NUMINAMATH_CALUDE_P_lower_bound_and_equality_l2496_249682

/-- The number of 4k-digit numbers composed of digits 2 and 0 (not starting with 0) that are divisible by 2020 -/
def P (k : ℕ+) : ℕ := sorry

/-- The theorem stating the inequality and the condition for equality -/
theorem P_lower_bound_and_equality (k : ℕ+) :
  P k ≥ Nat.choose (2 * k - 1) k ^ 2 ∧
  (P k = Nat.choose (2 * k - 1) k ^ 2 ↔ k ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_P_lower_bound_and_equality_l2496_249682


namespace NUMINAMATH_CALUDE_min_value_and_nonexistence_l2496_249602

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 1 / b = Real.sqrt (a * b)) :
  (∀ x y, x > 0 → y > 0 → 1 / x + 1 / y = Real.sqrt (x * y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = Real.sqrt (x * y) ∧ 2 * x + 3 * y = 6) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_nonexistence_l2496_249602


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l2496_249632

theorem base_10_to_base_8 : 
  (3 * 8^3 + 4 * 8^2 + 1 * 8^1 + 1 * 8^0 : ℕ) = 1801 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l2496_249632


namespace NUMINAMATH_CALUDE_other_triangle_area_ratio_l2496_249645

/-- Represents a right triangle with a point on its hypotenuse and parallel lines dividing it -/
structure DividedRightTriangle where
  /-- The area of one small right triangle -/
  smallTriangleArea : ℝ
  /-- The area of the rectangle -/
  rectangleArea : ℝ
  /-- The ratio of the small triangle area to the rectangle area -/
  n : ℝ
  /-- The ratio of the longer side to the shorter side of the rectangle -/
  k : ℝ
  /-- The small triangle area is n times the rectangle area -/
  area_relation : smallTriangleArea = n * rectangleArea
  /-- The sides of the rectangle are in the ratio 1:k -/
  rectangle_ratio : k > 0

/-- The ratio of the area of the other small right triangle to the area of the rectangle is n -/
theorem other_triangle_area_ratio (t : DividedRightTriangle) :
    ∃ otherTriangleArea : ℝ, otherTriangleArea / t.rectangleArea = t.n := by
  sorry

end NUMINAMATH_CALUDE_other_triangle_area_ratio_l2496_249645


namespace NUMINAMATH_CALUDE_linear_function_composition_l2496_249642

-- Define a linear function
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 9 * x + 8) →
  (∀ x, f x = 3 * x + 2) ∨ (∀ x, f x = -3 * x - 4) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l2496_249642


namespace NUMINAMATH_CALUDE_royal_family_children_count_l2496_249692

/-- Represents the number of years that have passed -/
def n : ℕ := sorry

/-- Represents the number of daughters -/
def d : ℕ := sorry

/-- The total number of children -/
def total_children : ℕ := d + 3

/-- The initial age of the king and queen -/
def initial_royal_age : ℕ := 35

/-- The initial total age of the children -/
def initial_children_age : ℕ := 35

/-- The combined age of the king and queen after n years -/
def royal_age_after_n_years : ℕ := 2 * initial_royal_age + 2 * n

/-- The total age of the children after n years -/
def children_age_after_n_years : ℕ := initial_children_age + total_children * n

theorem royal_family_children_count :
  (royal_age_after_n_years = children_age_after_n_years) ∧
  (total_children ≤ 20) →
  (total_children = 7 ∨ total_children = 9) :=
by sorry

end NUMINAMATH_CALUDE_royal_family_children_count_l2496_249692


namespace NUMINAMATH_CALUDE_certain_number_proof_l2496_249636

theorem certain_number_proof (x : ℝ) : 0.15 * x + 0.12 * 45 = 9.15 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2496_249636


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2496_249694

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, x^2 + 2*x - (k - 1) = 0) ↔ k ≥ 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2496_249694
