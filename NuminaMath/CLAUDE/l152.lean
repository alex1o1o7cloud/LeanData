import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_proposition_l152_15228

theorem negation_of_proposition :
  ¬(∀ x y : ℤ, Even (x + y) → (Even x ∧ Even y)) ↔
  (∀ x y : ℤ, ¬Even (x + y) → ¬(Even x ∧ Even y)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l152_15228


namespace NUMINAMATH_CALUDE_swimmers_passing_theorem_l152_15254

/-- Represents the number of times two swimmers pass each other in a pool -/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  -- The actual implementation is not provided, as per the instructions
  sorry

/-- Theorem stating the number of times the swimmers pass each other under given conditions -/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 120
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 53 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_passing_theorem_l152_15254


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l152_15202

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∀ x, x ∈ Set.Ioo (-3 : ℝ) 6 ↔ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l152_15202


namespace NUMINAMATH_CALUDE_lattice_point_in_triangle_l152_15290

/-- A point in a 2D integer lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A convex quadrilateral in a 2D integer lattice -/
structure ConvexLatticeQuadrilateral where
  P : LatticePoint
  Q : LatticePoint
  R : LatticePoint
  S : LatticePoint
  is_convex : Bool  -- Assume this is true for a convex quadrilateral

/-- The angle between two vectors -/
def angle (v1 v2 : LatticePoint → LatticePoint) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def is_in_triangle (X P Q E : LatticePoint) : Prop := sorry

theorem lattice_point_in_triangle
  (PQRS : ConvexLatticeQuadrilateral)
  (E : LatticePoint)
  (h_diagonals_intersect : E = sorry)  -- E is the intersection of diagonals
  (h_angle_sum : angle (λ p => PQRS.P) (λ p => PQRS.Q) < 180) :
  ∃ X : LatticePoint, X ≠ PQRS.P ∧ X ≠ PQRS.Q ∧ is_in_triangle X PQRS.P PQRS.Q E :=
sorry

end NUMINAMATH_CALUDE_lattice_point_in_triangle_l152_15290


namespace NUMINAMATH_CALUDE_negation_of_implication_l152_15213

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 2 → a^2 + b^2 ≥ 4) ↔ (ab ≠ 2 → a^2 + b^2 < 4) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l152_15213


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l152_15240

theorem complex_absolute_value_product : 
  Complex.abs ((3 * Real.sqrt 5 - 3 * Complex.I) * (2 * Real.sqrt 2 + 2 * Complex.I)) = 18 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l152_15240


namespace NUMINAMATH_CALUDE_point_on_or_outside_circle_l152_15257

-- Define the circle C
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 2}

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (a, 2 - a)

-- Theorem statement
theorem point_on_or_outside_circle : 
  ∀ a : ℝ, (P a) ∈ C ∨ (P a) ∉ interior C :=
sorry

end NUMINAMATH_CALUDE_point_on_or_outside_circle_l152_15257


namespace NUMINAMATH_CALUDE_lower_rent_is_40_l152_15282

/-- Represents the motel rental scenario -/
structure MotelRental where
  lower_rent : ℕ  -- The lower rent amount
  higher_rent : ℕ := 60  -- The higher rent amount, fixed at $60
  total_rent : ℕ := 1000  -- The total rent for the night
  reduction_percent : ℕ := 20  -- The reduction percentage if 10 rooms switch to lower rent

/-- Theorem stating that the lower rent amount is $40 -/
theorem lower_rent_is_40 (m : MotelRental) : m.lower_rent = 40 := by
  sorry

#check lower_rent_is_40

end NUMINAMATH_CALUDE_lower_rent_is_40_l152_15282


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l152_15219

-- Define the number of balls of each color
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l152_15219


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l152_15238

theorem product_purely_imaginary (x : ℝ) :
  (∃ b : ℝ, (x + 2*Complex.I) * ((x + 1) + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 3) + 2*Complex.I) = b * Complex.I) ↔
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l152_15238


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l152_15236

theorem quadratic_inequality_equivalence (m : ℝ) : 
  (∀ x > 1, x^2 + (m - 2) * x + 3 - m ≥ 0) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l152_15236


namespace NUMINAMATH_CALUDE_reflection_line_sum_l152_15229

/-- Given a line y = mx + b, if the reflection of point (1, 2) across this line is (7, 6), then m + b = 8.5 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x - 1)^2 + (y - 2)^2 = (7 - x)^2 + (6 - y)^2 ∧ 
    (x + 7) / 2 = (y + 6) / 2 / m + b ∧
    (y + 6) / 2 = m * (x + 7) / 2 + b) → 
  m + b = 8.5 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l152_15229


namespace NUMINAMATH_CALUDE_max_value_quadratic_expression_l152_15223

/-- Given a system of equations, prove that the maximum value of a quadratic expression is 11 -/
theorem max_value_quadratic_expression (x y z : ℝ) 
  (eq1 : x - y + z - 1 = 0)
  (eq2 : x * y + 2 * z^2 - 6 * z + 1 = 0) :
  ∃ (max : ℝ), max = 11 ∧ ∀ (x' y' z' : ℝ), 
    x' - y' + z' - 1 = 0 → 
    x' * y' + 2 * z'^2 - 6 * z' + 1 = 0 → 
    (x' - 1)^2 + (y' + 1)^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_expression_l152_15223


namespace NUMINAMATH_CALUDE_m_range_l152_15220

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 3

-- State the theorem
theorem m_range :
  (∀ x : ℝ, x ≥ 1 → f x + m^2 * f x ≥ f (x - 1) + 3 * f m) ↔
  (m ≤ -1 ∨ m ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_m_range_l152_15220


namespace NUMINAMATH_CALUDE_circumcenter_outside_l152_15250

/-- An isosceles trapezoid with specific angle measurements -/
structure IsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  base_angle : ℝ
  /-- The angle between the diagonals adjacent to the lateral side -/
  diagonal_angle : ℝ
  /-- Condition that the base angle is 50 degrees -/
  base_angle_eq : base_angle = 50
  /-- Condition that the diagonal angle is 40 degrees -/
  diagonal_angle_eq : diagonal_angle = 40

/-- The location of the circumcenter relative to the trapezoid -/
inductive CircumcenterLocation
  | Inside
  | Outside

/-- Theorem stating that the circumcenter is outside the trapezoid -/
theorem circumcenter_outside (t : IsoscelesTrapezoid) : 
  CircumcenterLocation.Outside = 
    CircumcenterLocation.Outside := by sorry

end NUMINAMATH_CALUDE_circumcenter_outside_l152_15250


namespace NUMINAMATH_CALUDE_final_result_proof_l152_15204

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 990) : 
  (chosen_number / 9 : ℚ) - 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_final_result_proof_l152_15204


namespace NUMINAMATH_CALUDE_polynomial_simplification_l152_15285

theorem polynomial_simplification (x : ℝ) : 
  (3*x^2 - 2*x + 5)*(x - 2) - (x - 2)*(2*x^2 - 5*x + 42) + (2*x - 7)*(x - 2)*(x + 3) = 
  3*x^3 - 4*x^2 - 62*x + 116 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l152_15285


namespace NUMINAMATH_CALUDE_product_simplification_l152_15293

theorem product_simplification : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 = 81 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l152_15293


namespace NUMINAMATH_CALUDE_nested_radical_equation_l152_15296

theorem nested_radical_equation (x : ℝ) : 
  x = 34 → 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_nested_radical_equation_l152_15296


namespace NUMINAMATH_CALUDE_cubic_inequality_l152_15283

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 + 3*a*b*c ≥ a*b*(a+b) + b*c*(b+c) + c*a*(c+a) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l152_15283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l152_15261

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 1 + a 5 = 8 →                                       -- given condition
  a 3 = 4 :=                                            -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l152_15261


namespace NUMINAMATH_CALUDE_selling_price_with_equal_loss_l152_15244

/-- Given an article with cost price 59 and selling price 66 resulting in a profit of 7,
    prove that the selling price resulting in the same loss as the profit is 52. -/
theorem selling_price_with_equal_loss (cost_price selling_price_profit : ℕ) 
  (h1 : cost_price = 59)
  (h2 : selling_price_profit = 66)
  (h3 : selling_price_profit - cost_price = 7) : 
  ∃ (selling_price_loss : ℕ), 
    selling_price_loss = 52 ∧ 
    cost_price - selling_price_loss = selling_price_profit - cost_price :=
by sorry

end NUMINAMATH_CALUDE_selling_price_with_equal_loss_l152_15244


namespace NUMINAMATH_CALUDE_quadratic_decreasing_after_vertex_l152_15224

def f (x : ℝ) : ℝ := -(x - 2)^2 - 7

theorem quadratic_decreasing_after_vertex :
  ∀ (x1 x2 : ℝ), x1 > 2 → x2 > x1 → f x2 < f x1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_after_vertex_l152_15224


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l152_15275

theorem binomial_expansion_sum_zero (n k : ℕ) (b : ℝ) 
  (h1 : n ≥ 2)
  (h2 : b ≠ 0)
  (h3 : k > 0)
  (h4 : n.choose 1 * b^(n-1) * (k+1) + n.choose 2 * b^(n-2) * (k+1)^2 = 0) :
  n = 2*k + 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l152_15275


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l152_15276

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*x + m - 1 = 0 ∧ 
    y^2 - 2*y + m - 1 = 0) →
  m < 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l152_15276


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l152_15245

theorem quadratic_form_equivalence (d : ℕ) (h : d > 0) (h_div : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2*x^2 + 2*x*y + 3*y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l152_15245


namespace NUMINAMATH_CALUDE_fourth_root_power_eight_l152_15286

theorem fourth_root_power_eight : (2^6)^(1/4)^8 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_power_eight_l152_15286


namespace NUMINAMATH_CALUDE_cubic_root_sum_l152_15255

theorem cubic_root_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l152_15255


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l152_15292

theorem largest_x_sqrt_3x_eq_5x :
  (∃ x : ℝ, x > 0 ∧ Real.sqrt (3 * x) = 5 * x) →
  (∀ x : ℝ, Real.sqrt (3 * x) = 5 * x → x ≤ 3/25) ∧
  Real.sqrt (3 * (3/25)) = 5 * (3/25) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l152_15292


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l152_15235

theorem sum_of_extreme_prime_factors_of_1365 : ∃ (min max : ℕ), 
  (min.Prime ∧ max.Prime ∧ 
   min ∣ 1365 ∧ max ∣ 1365 ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → min ≤ p) ∧
   (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ max)) ∧
  min + max = 16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l152_15235


namespace NUMINAMATH_CALUDE_power_of_two_six_l152_15288

theorem power_of_two_six : 2^3 * 2^3 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_six_l152_15288


namespace NUMINAMATH_CALUDE_optimization_scheme_sales_l152_15207

/-- Given a sequence of three terms forming an arithmetic progression with a sum of 2.46 million,
    prove that the middle term (second term) is equal to 0.82 million. -/
theorem optimization_scheme_sales (a₁ a₂ a₃ : ℝ) : 
  a₁ + a₂ + a₃ = 2.46 ∧ 
  a₂ - a₁ = a₃ - a₂ → 
  a₂ = 0.82 := by
sorry

end NUMINAMATH_CALUDE_optimization_scheme_sales_l152_15207


namespace NUMINAMATH_CALUDE_classroom_ratio_l152_15252

/-- Represents a classroom with two portions of students with different GPAs -/
structure Classroom where
  portion_a : ℝ  -- Size of portion A (GPA 15)
  portion_b : ℝ  -- Size of portion B (GPA 18)
  gpa_a : ℝ      -- GPA of portion A
  gpa_b : ℝ      -- GPA of portion B
  gpa_total : ℝ  -- Total GPA of the class

/-- The ratio of portion A to the whole class is 1:3 given the conditions -/
theorem classroom_ratio (c : Classroom) 
  (h1 : c.gpa_a = 15)
  (h2 : c.gpa_b = 18)
  (h3 : c.gpa_total = 17)
  (h4 : c.gpa_a * c.portion_a + c.gpa_b * c.portion_b = c.gpa_total * (c.portion_a + c.portion_b)) :
  c.portion_a / (c.portion_a + c.portion_b) = 1 / 3 := by
  sorry

#check classroom_ratio

end NUMINAMATH_CALUDE_classroom_ratio_l152_15252


namespace NUMINAMATH_CALUDE_lcm_plus_hundred_l152_15280

theorem lcm_plus_hundred (a b : ℕ) (h1 : a = 1056) (h2 : b = 792) :
  Nat.lcm a b + 100 = 3268 := by sorry

end NUMINAMATH_CALUDE_lcm_plus_hundred_l152_15280


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l152_15270

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + x - 1

theorem polynomial_sum_theorem (g : ℝ → ℝ) 
  (h1 : ∀ x, f x + g x = 3*x^2 - 2) :
  g = λ x => -x^4 + 6*x^2 - x - 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l152_15270


namespace NUMINAMATH_CALUDE_fourth_root_of_256000000_l152_15289

theorem fourth_root_of_256000000 : (400 : ℕ) ^ 4 = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_256000000_l152_15289


namespace NUMINAMATH_CALUDE_jerry_remaining_money_l152_15271

/-- Calculates the remaining money after expenses --/
def remaining_money (initial_amount video_games_cost snack_cost toy_original_cost toy_discount_percent : ℚ) : ℚ :=
  let toy_discount := toy_original_cost * (toy_discount_percent / 100)
  let toy_final_cost := toy_original_cost - toy_discount
  let total_spent := video_games_cost + snack_cost + toy_final_cost
  initial_amount - total_spent

/-- Theorem stating that Jerry's remaining money is $6 --/
theorem jerry_remaining_money :
  remaining_money 18 6 3 4 25 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerry_remaining_money_l152_15271


namespace NUMINAMATH_CALUDE_exist_special_numbers_l152_15233

/-- Sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of two numbers satisfying the given conditions -/
theorem exist_special_numbers : 
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A = 2016 * B ∧ sum_of_digits A = sum_of_digits B / 2016 := by
  sorry

end NUMINAMATH_CALUDE_exist_special_numbers_l152_15233


namespace NUMINAMATH_CALUDE_even_integers_between_fractions_l152_15239

theorem even_integers_between_fractions :
  let lower_bound : ℚ := 23/5
  let upper_bound : ℚ := 47/3
  (Finset.filter (fun n => n % 2 = 0) (Finset.Icc ⌈lower_bound⌉ ⌊upper_bound⌋)).card = 5 :=
by sorry

end NUMINAMATH_CALUDE_even_integers_between_fractions_l152_15239


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l152_15256

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l152_15256


namespace NUMINAMATH_CALUDE_arrange_balls_count_l152_15284

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeColoredBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.choose (red + yellow + white) white *
  Nat.choose (red + yellow) red *
  Nat.choose yellow yellow

/-- Theorem stating that arranging 2 red, 3 yellow, and 4 white balls results in 1260 arrangements -/
theorem arrange_balls_count : arrangeColoredBalls 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_count_l152_15284


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l152_15211

theorem product_of_successive_numbers :
  let n : ℝ := 51.49757275833493
  let product := n * (n + 1)
  ∃ ε > 0, |product - 2703| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l152_15211


namespace NUMINAMATH_CALUDE_hearts_ratio_equals_half_l152_15279

-- Define the ♥ operation
def hearts (n m : ℕ) : ℕ := n^4 * m^3

-- Theorem statement
theorem hearts_ratio_equals_half : 
  (hearts 2 4) / (hearts 4 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hearts_ratio_equals_half_l152_15279


namespace NUMINAMATH_CALUDE_dice_labeling_exists_l152_15215

/-- Represents a 6-sided die with integer labels -/
def Die := Fin 6 → ℕ

/-- Checks if a given labeling of two dice produces all sums from 1 to 36 -/
def valid_labeling (d1 d2 : Die) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 36 → ∃ (i j : Fin 6), d1 i + d2 j = n

/-- There exists a labeling for two dice that produces all sums from 1 to 36 with equal probabilities -/
theorem dice_labeling_exists : ∃ (d1 d2 : Die), valid_labeling d1 d2 := by
  sorry

end NUMINAMATH_CALUDE_dice_labeling_exists_l152_15215


namespace NUMINAMATH_CALUDE_nonnegative_solutions_count_l152_15241

theorem nonnegative_solutions_count : 
  ∃! (n : ℕ), ∃ (x : ℝ), x ≥ 0 ∧ x^2 = -6*x ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_nonnegative_solutions_count_l152_15241


namespace NUMINAMATH_CALUDE_savings_period_is_four_months_l152_15205

-- Define the savings and stock parameters
def wife_weekly_savings : ℕ := 100
def husband_monthly_savings : ℕ := 225
def stock_price : ℕ := 50
def shares_bought : ℕ := 25

-- Define the function to calculate the number of months saved
def months_saved : ℕ :=
  let total_investment := stock_price * shares_bought
  let total_savings := total_investment * 2
  let monthly_savings := wife_weekly_savings * 4 + husband_monthly_savings
  total_savings / monthly_savings

-- Theorem statement
theorem savings_period_is_four_months :
  months_saved = 4 :=
sorry

end NUMINAMATH_CALUDE_savings_period_is_four_months_l152_15205


namespace NUMINAMATH_CALUDE_rosie_circles_count_l152_15209

/-- Proves that given a circular track of 1/4 mile length, if person A runs 3 miles
    and person B runs at twice the speed of person A, then person B circles the track 24 times. -/
theorem rosie_circles_count (track_length : ℝ) (lou_distance : ℝ) (speed_ratio : ℝ) : 
  track_length = 1/4 →
  lou_distance = 3 →
  speed_ratio = 2 →
  (lou_distance * speed_ratio) / track_length = 24 := by
sorry

end NUMINAMATH_CALUDE_rosie_circles_count_l152_15209


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l152_15272

theorem max_value_sum_of_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l152_15272


namespace NUMINAMATH_CALUDE_min_absolute_difference_l152_15234

theorem min_absolute_difference (a b c d : ℝ) 
  (hab : |a - b| = 5)
  (hbc : |b - c| = 8)
  (hcd : |c - d| = 10) :
  ∃ (m : ℝ), (∀ x, |a - d| ≥ x → m ≤ x) ∧ |a - d| ≥ m ∧ m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_absolute_difference_l152_15234


namespace NUMINAMATH_CALUDE_oplus_2_3_4_l152_15253

-- Define the operation ⊕
def oplus (a b c : ℝ) : ℝ := a * b - 4 * a + c^2

-- Theorem statement
theorem oplus_2_3_4 : oplus 2 3 4 = 14 := by sorry

end NUMINAMATH_CALUDE_oplus_2_3_4_l152_15253


namespace NUMINAMATH_CALUDE_root_shrinking_method_l152_15281

theorem root_shrinking_method (a b c p α β : ℝ) (ha : a ≠ 0) (hp : p ≠ 0) 
  (hα : a * α^2 + b * α + c = 0) (hβ : a * β^2 + b * β + c = 0) :
  (p^2 * a) * (α/p)^2 + (p * b) * (α/p) + c = 0 ∧
  (p^2 * a) * (β/p)^2 + (p * b) * (β/p) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_shrinking_method_l152_15281


namespace NUMINAMATH_CALUDE_empty_seats_calculation_l152_15200

/-- Calculates the number of empty seats in a theater -/
def empty_seats (total_seats people_watching : ℕ) : ℕ :=
  total_seats - people_watching

/-- Theorem: The number of empty seats is the difference between total seats and people watching -/
theorem empty_seats_calculation (total_seats people_watching : ℕ) 
  (h1 : total_seats ≥ people_watching) :
  empty_seats total_seats people_watching = total_seats - people_watching :=
by sorry

end NUMINAMATH_CALUDE_empty_seats_calculation_l152_15200


namespace NUMINAMATH_CALUDE_average_percent_increase_per_year_l152_15266

def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def time_period : ℕ := 10

theorem average_percent_increase_per_year :
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / time_period
  let percent_increase : ℚ := (average_annual_increase / initial_population) * 100
  percent_increase = 7 := by sorry

end NUMINAMATH_CALUDE_average_percent_increase_per_year_l152_15266


namespace NUMINAMATH_CALUDE_circle_tangent_to_three_lines_l152_15299

-- Define the types for lines and circles
variable (Line Circle : Type)

-- Define the tangent relation between a circle and a line
variable (tangent_to : Circle → Line → Prop)

-- Define the intersection angle between two lines
variable (intersection_angle : Line → Line → ℝ)

-- Define the main theorem
theorem circle_tangent_to_three_lines 
  (C : Circle) (l m n : Line) :
  (tangent_to C l ∧ tangent_to C m ∧ tangent_to C n) →
  (∃ (C' : Circle), 
    tangent_to C' l ∧ tangent_to C' m ∧ tangent_to C' n) ∧
  (intersection_angle l m = π/3 ∧ 
   intersection_angle m n = π/3 ∧ 
   intersection_angle n l = π/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_three_lines_l152_15299


namespace NUMINAMATH_CALUDE_min_distance_to_line_l152_15269

/-- Given the line x + 2y = 1, the minimum value of x^2 + y^2 is 1/5 -/
theorem min_distance_to_line (x y : ℝ) (h : x + 2*y = 1) : 
  ∃ (min : ℝ), min = 1/5 ∧ ∀ (x' y' : ℝ), x' + 2*y' = 1 → x'^2 + y'^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l152_15269


namespace NUMINAMATH_CALUDE_circle_x_axis_intersection_l152_15278

/-- A circle with diameter endpoints at (0,0) and (10, -6) -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 3)^2 = 34}

/-- The x-axis -/
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

/-- The point (0,0) -/
def Origin : ℝ × ℝ := (0, 0)

theorem circle_x_axis_intersection :
  ∃ p : ℝ × ℝ, p ∈ Circle ∩ XAxis ∧ p ≠ Origin ∧ p.1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_x_axis_intersection_l152_15278


namespace NUMINAMATH_CALUDE_angle_beta_properties_l152_15264

theorem angle_beta_properties (β : Real) 
  (h1 : π/2 < β ∧ β < π)  -- β is in the second quadrant
  (h2 : (2 * Real.tan β ^ 2) / (3 * Real.tan β + 2) = 1) :  -- β satisfies the given equation
  (Real.sin (β + 3 * π / 2) = 2 * Real.sqrt 5 / 5) ∧ 
  ((2 / 3) * Real.sin β ^ 2 + Real.cos β * Real.sin β = -1 / 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_beta_properties_l152_15264


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l152_15210

/-- A function f: ℝ → ℝ is monotonically increasing -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x^3 + 2x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The condition p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The condition q: m ≥ 8x / (x^2 + 4) holds for any x > 0 -/
def q (m : ℝ) : Prop := ∀ x > 0, m ≥ 8*x / (x^2 + 4)

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ m : ℝ, q m → p m) ∧ (∃ m : ℝ, p m ∧ ¬q m) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l152_15210


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l152_15295

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 6x + 3 and y = 3cx + 1 are parallel -/
theorem parallel_lines_c_value : 
  (∀ x y : ℝ, y = 6 * x + 3 ↔ y = 3 * c * x + 1) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l152_15295


namespace NUMINAMATH_CALUDE_smallest_n_same_divisors_l152_15262

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Checks if three consecutive natural numbers have the same number of divisors -/
def same_num_divisors (n : ℕ) : Prop :=
  num_divisors n = num_divisors (n + 1) ∧ num_divisors n = num_divisors (n + 2)

/-- 33 is the smallest natural number n such that n, n+1, and n+2 have the same number of divisors -/
theorem smallest_n_same_divisors :
  (∀ m : ℕ, m < 33 → ¬ same_num_divisors m) ∧ same_num_divisors 33 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_same_divisors_l152_15262


namespace NUMINAMATH_CALUDE_max_k_for_exp_inequality_l152_15225

theorem max_k_for_exp_inequality : 
  (∃ k : ℝ, ∀ x : ℝ, Real.exp x ≥ k * x) ∧ 
  (∀ k : ℝ, (∀ x : ℝ, Real.exp x ≥ k * x) → k ≤ Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_max_k_for_exp_inequality_l152_15225


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l152_15217

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  ∃ p : ℤ, 2 * a^4 + 2 * b^4 + 2 * c^4 = 4 * p^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l152_15217


namespace NUMINAMATH_CALUDE_rectangle_inequality_l152_15267

/-- Represents a rectangle with side lengths 3b and b -/
structure Rectangle (b : ℝ) where
  length : ℝ := 3 * b
  width : ℝ := b

/-- Represents a point P on the longer side of the rectangle -/
structure PointP (b : ℝ) where
  x : ℝ
  y : ℝ := 0
  h1 : 0 ≤ x ∧ x ≤ 3 * b

/-- Represents a point T inside the rectangle -/
structure PointT (b : ℝ) where
  x : ℝ
  y : ℝ
  h1 : 0 < x ∧ x < 3 * b
  h2 : 0 < y ∧ y < b
  h3 : y = b / 2

/-- The theorem to be proved -/
theorem rectangle_inequality (b : ℝ) (h : b > 0) (R : Rectangle b) (P : PointP b) (T : PointT b) :
  let s := (2 * b)^2 + b^2
  let rt := (T.x - 0)^2 + (T.y - 0)^2
  s > 2 * rt := by sorry

end NUMINAMATH_CALUDE_rectangle_inequality_l152_15267


namespace NUMINAMATH_CALUDE_johns_age_theorem_l152_15259

theorem johns_age_theorem :
  ∀ (age : ℕ),
  (∃ (s : ℕ), (age - 2) = s^2) ∧ 
  (∃ (c : ℕ), (age + 2) = c^3) →
  age = 6 ∨ age = 123 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_age_theorem_l152_15259


namespace NUMINAMATH_CALUDE_polynomial_property_l152_15208

/-- A polynomial of the form 2x^3 - 30x^2 + cx -/
def P (c : ℤ) (x : ℤ) : ℤ := 2 * x^3 - 30 * x^2 + c * x

/-- The property that P(x) yields consecutive integers for consecutive integer inputs -/
def consecutive_values (c : ℤ) : Prop :=
  ∀ a : ℤ, ∃ k : ℤ, P c (a - 1) = k - 1 ∧ P c a = k ∧ P c (a + 1) = k + 1

theorem polynomial_property :
  ∀ c : ℤ, consecutive_values c → c = 149 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l152_15208


namespace NUMINAMATH_CALUDE_spratilish_word_count_mod_1000_l152_15260

/-- Represents a Spratilish letter -/
inductive SpratilishLetter
| M
| P
| Z
| O

/-- Checks if a SpratilishLetter is a consonant -/
def isConsonant (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.M => true
  | SpratilishLetter.P => true
  | _ => false

/-- Checks if a SpratilishLetter is a vowel -/
def isVowel (l : SpratilishLetter) : Bool :=
  match l with
  | SpratilishLetter.Z => true
  | SpratilishLetter.O => true
  | _ => false

/-- Represents a Spratilish word as a list of SpratilishLetters -/
def SpratilishWord := List SpratilishLetter

/-- Checks if a SpratilishWord is valid (at least three consonants between any two vowels) -/
def isValidSpratilishWord (w : SpratilishWord) : Bool :=
  sorry

/-- Counts the number of valid 9-letter Spratilish words -/
def countValidSpratilishWords : Nat :=
  sorry

/-- The main theorem: The number of valid 9-letter Spratilish words is congruent to 704 modulo 1000 -/
theorem spratilish_word_count_mod_1000 :
  countValidSpratilishWords % 1000 = 704 := by sorry

end NUMINAMATH_CALUDE_spratilish_word_count_mod_1000_l152_15260


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l152_15242

-- Define the set of letters in BIOLOGY
def biology : Finset Char := {'B', 'I', 'O', 'L', 'G', 'Y'}

-- Define the set of vowels in BIOLOGY
def vowels : Finset Char := {'I', 'O'}

-- Define the set of consonants in BIOLOGY
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

-- Define the number of vowels to be selected
def num_vowels : ℕ := 2

-- Define the number of consonants to be selected
def num_consonants : ℕ := 4

-- Define a function to count distinct collections
def count_distinct_collections : ℕ := sorry

-- Theorem statement
theorem distinct_collections_biology :
  count_distinct_collections = 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l152_15242


namespace NUMINAMATH_CALUDE_probability_second_draw_given_first_l152_15248

/-- The probability of drawing a high-quality item on the second draw, given that the first draw was a high-quality item, when there are 5 high-quality items and 3 defective items in total. -/
theorem probability_second_draw_given_first (total_items : ℕ) (high_quality : ℕ) (defective : ℕ) :
  total_items = high_quality + defective →
  high_quality = 5 →
  defective = 3 →
  (high_quality - 1 : ℚ) / (total_items - 1) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_draw_given_first_l152_15248


namespace NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_l152_15298

theorem tan_fifteen_degree_fraction : 
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_l152_15298


namespace NUMINAMATH_CALUDE_unique_positive_integers_sum_l152_15212

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 73) / 2 + 5 / 2)

theorem unique_positive_integers_sum (a b c : ℕ+) :
  x^80 = 3*x^78 + 18*x^74 + 15*x^72 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 →
  a + b + c = 265 := by sorry

end NUMINAMATH_CALUDE_unique_positive_integers_sum_l152_15212


namespace NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l152_15287

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (regularHours : Float) (regularRate : Float) (totalHours : Float) (totalEarnings : Float) : Float :=
  let regularEarnings := regularHours * regularRate
  let excessHours := totalHours - regularHours
  let excessEarnings := totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / regularRate

/-- Proves that given Lloyd's work conditions, the multiple of his regular rate for excess hours is 2.5 --/
theorem lloyd_excess_rate_multiple :
  let regularHours : Float := 7.5
  let regularRate : Float := 4.5
  let totalHours : Float := 10.5
  let totalEarnings : Float := 67.5
  excessRateMultiple regularHours regularRate totalHours totalEarnings = 2.5 := by
  sorry

#eval excessRateMultiple 7.5 4.5 10.5 67.5

end NUMINAMATH_CALUDE_lloyd_excess_rate_multiple_l152_15287


namespace NUMINAMATH_CALUDE_inequality_satisfied_iff_m_in_range_l152_15247

theorem inequality_satisfied_iff_m_in_range (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8*x + 20) / (m*x^2 + 2*(m+1)*x + 9*m + 4) < 0) ↔ 
  m < -1/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_satisfied_iff_m_in_range_l152_15247


namespace NUMINAMATH_CALUDE_fourth_player_win_probability_l152_15291

/-- The probability of the fourth player winning in a coin-flipping game with four players -/
theorem fourth_player_win_probability :
  let p : ℕ → ℝ := λ n => (1 / 2) ^ (4 * n + 1)
  let total_prob := (∑' n, p n)
  total_prob = 1 / 30
  := by sorry

end NUMINAMATH_CALUDE_fourth_player_win_probability_l152_15291


namespace NUMINAMATH_CALUDE_additional_interest_rate_proof_l152_15274

/-- Proves that given specific investment conditions, the additional interest rate must be 8% --/
theorem additional_interest_rate_proof (initial_investment : ℝ) (initial_rate : ℝ) 
  (total_rate : ℝ) (additional_investment : ℝ) : 
  initial_investment = 2400 →
  initial_rate = 0.04 →
  total_rate = 0.06 →
  additional_investment = 2400 →
  (initial_investment * initial_rate + additional_investment * 0.08) / 
    (initial_investment + additional_investment) = total_rate :=
by sorry

end NUMINAMATH_CALUDE_additional_interest_rate_proof_l152_15274


namespace NUMINAMATH_CALUDE_miles_traveled_l152_15268

/-- Represents the efficiency of a car in miles per gallon -/
def miles_per_gallon : ℝ := 25

/-- Represents the cost of gas in dollars per gallon -/
def dollars_per_gallon : ℝ := 5

/-- Represents the amount of money spent on gas in dollars -/
def money_spent : ℝ := 25

/-- Theorem stating that given the efficiency of the car and the cost of gas,
    $25 worth of gas will allow the car to travel 125 miles -/
theorem miles_traveled (mpg : ℝ) (dpg : ℝ) (spent : ℝ) :
  mpg = miles_per_gallon →
  dpg = dollars_per_gallon →
  spent = money_spent →
  (spent / dpg) * mpg = 125 := by
  sorry

end NUMINAMATH_CALUDE_miles_traveled_l152_15268


namespace NUMINAMATH_CALUDE_point_on_x_axis_l152_15265

/-- A point P with coordinates (m+3, m-2) lies on the x-axis if and only if its coordinates are (5,0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 2 = 0 ∧ (m + 3, m - 2).1 = m + 3 ∧ (m + 3, m - 2).2 = m - 2) ↔ 
  (m + 3, m - 2) = (5, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l152_15265


namespace NUMINAMATH_CALUDE_unique_intersection_l152_15216

/-- Three lines in the 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ
  line3 : ℝ → ℝ → ℝ

/-- The intersection point of three lines -/
def intersection (lines : ThreeLines) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | lines.line1 p.1 p.2 = 0 ∧ lines.line2 p.1 p.2 = 0 ∧ lines.line3 p.1 p.2 = 0}

/-- The theorem stating that k = -1/2 is the unique value for which the given lines intersect at a single point -/
theorem unique_intersection : ∃! k : ℝ, 
  let lines := ThreeLines.mk
    (fun x y => x + k * y)
    (fun x y => 2 * x + 3 * y + 8)
    (fun x y => x - y - 1)
  (∃! p : ℝ × ℝ, p ∈ intersection lines k) ∧ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l152_15216


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l152_15258

-- Define the log function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for lgm < 1
def condition (m : ℝ) : Prop := log m < 1

-- Define the set {1, 2}
def set_B : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m ∈ set_B, condition m) ∧
  (∃ m : ℝ, condition m ∧ m ∉ set_B) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l152_15258


namespace NUMINAMATH_CALUDE_sum_of_powers_l152_15263

theorem sum_of_powers (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) :
  (2 * a) ^ (2 * b ^ 2) = (a ^ b + y ^ b) ^ 2 → y = 4 * a ^ 2 - a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l152_15263


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l152_15221

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±x is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 1) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l152_15221


namespace NUMINAMATH_CALUDE_sin_transformation_l152_15273

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (2 * (x - π / 12) + π / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l152_15273


namespace NUMINAMATH_CALUDE_solutions_periodic_l152_15227

/-- A system of differential equations with given initial conditions -/
structure DiffSystem where
  f : ℝ → ℝ  -- y = f(x)
  g : ℝ → ℝ  -- z = g(x)
  eqn1 : ∀ x, deriv f x = -(g x)^3
  eqn2 : ∀ x, deriv g x = (f x)^3
  init1 : f 0 = 1
  init2 : g 0 = 0
  unique : ∀ f' g', (∀ x, deriv f' x = -(g' x)^3) →
                    (∀ x, deriv g' x = (f' x)^3) →
                    f' 0 = 1 → g' 0 = 0 →
                    f' = f ∧ g' = g

/-- Definition of a periodic function -/
def Periodic (f : ℝ → ℝ) :=
  ∃ k : ℝ, k > 0 ∧ ∀ x, f (x + k) = f x

/-- The main theorem stating that solutions are periodic with the same period -/
theorem solutions_periodic (sys : DiffSystem) :
  ∃ k : ℝ, k > 0 ∧ Periodic sys.f ∧ Periodic sys.g ∧
  ∀ x, sys.f (x + k) = sys.f x ∧ sys.g (x + k) = sys.g x :=
sorry

end NUMINAMATH_CALUDE_solutions_periodic_l152_15227


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l152_15277

theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (3 * a * Real.cos A = c * Real.cos B + b * Real.cos C) →
  (Real.cos A = 1 / 3) ∧
  (a = 2 * Real.sqrt 3 ∧ Real.cos B + Real.cos C = 2 * Real.sqrt 3 / 3 → c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l152_15277


namespace NUMINAMATH_CALUDE_min_dot_product_of_tangents_l152_15246

-- Define a circle with radius 1
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point outside the circle
def PointOutside (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 > 1

-- Define tangent points
def TangentPoints (p a b : ℝ × ℝ) : Prop :=
  a ∈ Circle ∧ b ∈ Circle ∧
  ((p.1 - a.1) * a.1 + (p.2 - a.2) * a.2 = 0) ∧
  ((p.1 - b.1) * b.1 + (p.2 - b.2) * b.2 = 0)

-- Define dot product of vectors
def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem min_dot_product_of_tangents :
  ∀ p : ℝ × ℝ, PointOutside p →
  ∀ a b : ℝ × ℝ, TangentPoints p a b →
  ∃ m : ℝ, m = -3 + 2 * Real.sqrt 2 ∧
  ∀ x y : ℝ × ℝ, TangentPoints p x y →
  DotProduct (x.1 - p.1, x.2 - p.2) (y.1 - p.1, y.2 - p.2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_of_tangents_l152_15246


namespace NUMINAMATH_CALUDE_tangent_and_max_chord_length_l152_15206

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point M
def point_M (a : ℝ) : ℝ × ℝ := (1, a)

theorem tangent_and_max_chord_length :
  -- Part I: Point M is on the circle if and only if a = ±√3
  (∃ a : ℝ, circle_O (point_M a).1 (point_M a).2 ↔ a = Real.sqrt 3 ∨ a = -Real.sqrt 3) ∧
  -- Part II: Maximum value of |AC| + |BD| is 2√10
  (let a : ℝ := Real.sqrt 2
   ∀ A B C D : ℝ × ℝ,
   circle_O A.1 A.2 →
   circle_O B.1 B.2 →
   circle_O C.1 C.2 →
   circle_O D.1 D.2 →
   (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0 →  -- AC ⊥ BD
   (point_M a).1 = (A.1 + C.1) / 2 →  -- M is midpoint of AC
   (point_M a).1 = (B.1 + D.1) / 2 →  -- M is midpoint of BD
   (point_M a).2 = (A.2 + C.2) / 2 →  -- M is midpoint of AC
   (point_M a).2 = (B.2 + D.2) / 2 →  -- M is midpoint of BD
   Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) ≤ 2 * Real.sqrt 10) := by
sorry

end NUMINAMATH_CALUDE_tangent_and_max_chord_length_l152_15206


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l152_15249

theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_8_12_l152_15249


namespace NUMINAMATH_CALUDE_building_floors_l152_15226

theorem building_floors (floors_B floors_C : ℕ) : 
  (floors_C = 5 * floors_B - 6) →
  (floors_C = 59) →
  (∃ floors_A : ℕ, floors_A = floors_B - 9 ∧ floors_A = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_building_floors_l152_15226


namespace NUMINAMATH_CALUDE_merchant_tea_cups_l152_15251

theorem merchant_tea_cups (S O P : ℕ) 
  (h1 : S + O = 11) 
  (h2 : P + O = 15) 
  (h3 : S + P = 14) : 
  S + O + P = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_tea_cups_l152_15251


namespace NUMINAMATH_CALUDE_probability_no_empty_boxes_l152_15222

/-- The number of distinct balls -/
def num_balls : ℕ := 3

/-- The number of distinct boxes -/
def num_boxes : ℕ := 3

/-- The probability of placing balls into boxes with no empty boxes -/
def prob_no_empty_boxes : ℚ := 2/9

/-- Theorem stating that the probability of placing 3 distinct balls into 3 distinct boxes
    with no empty boxes is 2/9 -/
theorem probability_no_empty_boxes :
  (num_balls = 3 ∧ num_boxes = 3) →
  prob_no_empty_boxes = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_empty_boxes_l152_15222


namespace NUMINAMATH_CALUDE_polynomial_roots_l152_15297

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem polynomial_roots (P : ℝ → ℝ) (h_nonzero : P ≠ 0) 
  (h_form : ∀ x, P x = P 0 + P 1 * x + P 2 * x^2) :
  (∃ c ≠ 0, ∀ x, P x = c * (x^2 - x - 1)) ∧
  (∀ x, P x = 0 ↔ x = golden_ratio ∨ x = 1 - golden_ratio) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l152_15297


namespace NUMINAMATH_CALUDE_simplify_radical_fraction_l152_15230

theorem simplify_radical_fraction :
  (3 * Real.sqrt 10) / (Real.sqrt 5 + 2) = 15 * Real.sqrt 2 - 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_fraction_l152_15230


namespace NUMINAMATH_CALUDE_car_price_calculation_l152_15237

/-- Represents the price of a car given loan terms and payments. -/
def car_price (loan_years : ℕ) (interest_rate : ℚ) (down_payment : ℚ) (monthly_payment : ℚ) : ℚ :=
  down_payment + (loan_years * 12 : ℕ) * monthly_payment

/-- Theorem stating the price of the car under given conditions. -/
theorem car_price_calculation :
  car_price 5 (4/100) 5000 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_car_price_calculation_l152_15237


namespace NUMINAMATH_CALUDE_largest_pot_cost_largest_pot_cost_is_1_92_l152_15294

/-- The cost of the largest pot given specific conditions -/
theorem largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_diff : ℚ) (smallest_pot_odd_cents : Bool) : ℚ :=
  let smallest_pot_cost : ℚ := (total_cost - price_diff * (num_pots * (num_pots - 1) / 2)) / num_pots
  let rounded_smallest_pot_cost : ℚ := if smallest_pot_odd_cents then ⌊smallest_pot_cost * 100⌋ / 100 else ⌈smallest_pot_cost * 100⌉ / 100
  rounded_smallest_pot_cost + price_diff * (num_pots - 1)

/-- The main theorem proving the cost of the largest pot -/
theorem largest_pot_cost_is_1_92 :
  largest_pot_cost 6 (39/5) (1/4) true = 96/50 := by
  sorry

end NUMINAMATH_CALUDE_largest_pot_cost_largest_pot_cost_is_1_92_l152_15294


namespace NUMINAMATH_CALUDE_ellipse_equation_through_points_l152_15218

/-- The standard equation of an ellipse passing through (-3, 0) and (0, -2) -/
theorem ellipse_equation_through_points :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ 
    (x^2 / 9 + y^2 / 4 = 1)) ∧
  (-3^2 / a^2 + 0^2 / b^2 = 1) ∧
  (0^2 / a^2 + (-2)^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_through_points_l152_15218


namespace NUMINAMATH_CALUDE_robot_price_ratio_l152_15243

/-- The ratio of the price Tom should pay to the original price of the robot -/
theorem robot_price_ratio (original_price tom_price : ℚ) 
  (h1 : original_price = 3)
  (h2 : tom_price = 9) :
  tom_price / original_price = 3 := by
sorry

end NUMINAMATH_CALUDE_robot_price_ratio_l152_15243


namespace NUMINAMATH_CALUDE_class_size_problem_l152_15232

theorem class_size_problem (x : ℕ) : 
  (40 * x + 50 * 90) / (x + 50 : ℝ) = 71.25 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_class_size_problem_l152_15232


namespace NUMINAMATH_CALUDE_triangle_max_area_l152_15201

theorem triangle_max_area (a b c : ℝ) (h1 : a + b = 10) (h2 : c = 6) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  S ≤ 12 ∧ ∃ a b, a + b = 10 ∧ S = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l152_15201


namespace NUMINAMATH_CALUDE_judge_court_cases_judge_court_cases_proof_l152_15214

theorem judge_court_cases : ℕ → Prop :=
  fun total_cases =>
    let dismissed := 2
    let remaining := total_cases - dismissed
    let innocent := (2 * remaining) / 3
    let delayed := 1
    let guilty := 4
    remaining - innocent - delayed = guilty ∧ total_cases = 17

-- The proof
theorem judge_court_cases_proof : ∃ n : ℕ, judge_court_cases n := by
  sorry

end NUMINAMATH_CALUDE_judge_court_cases_judge_court_cases_proof_l152_15214


namespace NUMINAMATH_CALUDE_hexagon_minus_rhombus_area_l152_15203

-- Define the regular hexagon
def regular_hexagon (area : ℝ) : Prop :=
  area > 0 ∧ ∃ (side : ℝ), area = (3 * Real.sqrt 3 / 2) * side^2

-- Define the rhombus inside the hexagon
def rhombus_in_hexagon (hexagon_area : ℝ) (rhombus_area : ℝ) : Prop :=
  ∃ (side : ℝ), 
    rhombus_area = 2 * (Real.sqrt 3 / 4) * (4 / 3 * 30 * Real.sqrt 3)

-- The theorem to be proved
theorem hexagon_minus_rhombus_area 
  (hexagon_area : ℝ) (rhombus_area : ℝ) (remaining_area : ℝ) :
  regular_hexagon hexagon_area →
  rhombus_in_hexagon hexagon_area rhombus_area →
  hexagon_area = 135 →
  remaining_area = hexagon_area - rhombus_area →
  remaining_area = 75 := by
sorry

end NUMINAMATH_CALUDE_hexagon_minus_rhombus_area_l152_15203


namespace NUMINAMATH_CALUDE_ratio_200_percent_l152_15231

theorem ratio_200_percent (x : ℝ) : (6 : ℝ) / x = 2 → x = 3 :=
  sorry

end NUMINAMATH_CALUDE_ratio_200_percent_l152_15231
