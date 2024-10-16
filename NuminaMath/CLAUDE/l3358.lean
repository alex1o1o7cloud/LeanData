import Mathlib

namespace NUMINAMATH_CALUDE_f_2011_equals_2011_l3358_335804

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the main property of f
variable (h : ∀ a b : ℝ, f (a * f b) = a * b)

-- Theorem statement
theorem f_2011_equals_2011 : f 2011 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_equals_2011_l3358_335804


namespace NUMINAMATH_CALUDE_complex_equality_l3358_335808

theorem complex_equality (z : ℂ) : z = -1 + I ↔ Complex.abs (z - 2) = Complex.abs (z + 4) ∧ Complex.abs (z - 2) = Complex.abs (z - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3358_335808


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3358_335817

/-- Given an arithmetic sequence where a₇ = 10 and a₂₁ = 34, prove that a₅₀ = 682/7 -/
theorem arithmetic_sequence_50th_term :
  ∀ (a : ℕ → ℚ), 
    (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
    a 7 = 10 →                                        -- 7th term is 10
    a 21 = 34 →                                       -- 21st term is 34
    a 50 = 682 / 7 :=                                 -- 50th term is 682/7
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l3358_335817


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3358_335895

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ = -n² + 4n,
    prove that the common difference d is -2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- The given sum formula
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum function
  (h3 : ∀ n, a (n+1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3358_335895


namespace NUMINAMATH_CALUDE_lilies_count_l3358_335865

/-- The cost of a single chrysanthemum in yuan -/
def chrysanthemum_cost : ℕ := 3

/-- The cost of a single lily in yuan -/
def lily_cost : ℕ := 4

/-- The total amount of money Mom wants to spend in yuan -/
def total_money : ℕ := 100

/-- The number of chrysanthemums Mom wants to buy -/
def chrysanthemums_to_buy : ℕ := 16

/-- The number of lilies that can be bought with the remaining money -/
def lilies_to_buy : ℕ := (total_money - chrysanthemum_cost * chrysanthemums_to_buy) / lily_cost

theorem lilies_count : lilies_to_buy = 13 := by
  sorry

end NUMINAMATH_CALUDE_lilies_count_l3358_335865


namespace NUMINAMATH_CALUDE_seven_x_minus_three_y_equals_thirteen_l3358_335859

theorem seven_x_minus_three_y_equals_thirteen 
  (x y : ℝ) 
  (h1 : 4 * x + y = 8) 
  (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 13 := by
sorry

end NUMINAMATH_CALUDE_seven_x_minus_three_y_equals_thirteen_l3358_335859


namespace NUMINAMATH_CALUDE_circumscribed_diagonals_center_implies_rhombus_l3358_335819

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Check if a quadrilateral is circumscribed around a circle -/
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

/-- Check if the diagonals of a quadrilateral intersect at a given point -/
def diagonalsIntersectAt (q : Quadrilateral) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a quadrilateral is a rhombus -/
def isRhombus (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem circumscribed_diagonals_center_implies_rhombus (q : Quadrilateral) (c : Circle) :
  isCircumscribed q c → diagonalsIntersectAt q c.center → isRhombus q := by sorry

end NUMINAMATH_CALUDE_circumscribed_diagonals_center_implies_rhombus_l3358_335819


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3358_335805

theorem product_sum_theorem (a b c : ℤ) : 
  a * b * c = -13 → (a + b + c = -11 ∨ a + b + c = 13) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3358_335805


namespace NUMINAMATH_CALUDE_candy_bar_cost_l3358_335821

/-- The cost of a candy bar given initial amount and change --/
theorem candy_bar_cost (initial_amount change : ℕ) (h1 : initial_amount = 50) (h2 : change = 5) :
  initial_amount - change = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l3358_335821


namespace NUMINAMATH_CALUDE_store_charge_with_interest_l3358_335801

/-- Proves that a principal amount of $35 with 7% simple annual interest results in a total debt of $37.45 after one year -/
theorem store_charge_with_interest (P : ℝ) (interest_rate : ℝ) (total_debt : ℝ) : 
  interest_rate = 0.07 →
  total_debt = 37.45 →
  P * (1 + interest_rate) = total_debt →
  P = 35 := by
sorry

end NUMINAMATH_CALUDE_store_charge_with_interest_l3358_335801


namespace NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l3358_335890

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem solution_to_logarithmic_equation :
  ∃ x : ℝ, lg (3 * x + 4) = 1 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_to_logarithmic_equation_l3358_335890


namespace NUMINAMATH_CALUDE_missing_files_l3358_335824

theorem missing_files (total : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  total = 60 → 
  morning = total / 2 → 
  afternoon = 15 → 
  total - (morning + afternoon) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_files_l3358_335824


namespace NUMINAMATH_CALUDE_total_payment_after_discounts_l3358_335807

def shirt_price : ℝ := 80
def pants_price : ℝ := 100
def shirt_discount : ℝ := 0.15
def pants_discount : ℝ := 0.10
def coupon_discount : ℝ := 0.05

theorem total_payment_after_discounts :
  let discounted_shirt := shirt_price * (1 - shirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let total_before_coupon := discounted_shirt + discounted_pants
  let final_amount := total_before_coupon * (1 - coupon_discount)
  final_amount = 150.10 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_after_discounts_l3358_335807


namespace NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3358_335858

theorem factorial_sum_unique_solution :
  ∀ w x y z : ℕ+,
  w.val.factorial = x.val.factorial + y.val.factorial + z.val.factorial →
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_unique_solution_l3358_335858


namespace NUMINAMATH_CALUDE_park_pathway_width_l3358_335837

/-- Represents a rectangular park with pathways -/
structure Park where
  length : ℝ
  width : ℝ
  lawn_area : ℝ

/-- Calculates the total width of all pathways in the park -/
def total_pathway_width (p : Park) : ℝ :=
  -- Define the function here, but don't implement it
  sorry

/-- Theorem stating the total pathway width for the given park specifications -/
theorem park_pathway_width :
  let p : Park := { length := 60, width := 40, lawn_area := 2109 }
  total_pathway_width p = 2.91 := by
  sorry

end NUMINAMATH_CALUDE_park_pathway_width_l3358_335837


namespace NUMINAMATH_CALUDE_final_state_is_blue_l3358_335876

/-- Represents the colors of sheep -/
inductive Color
| Red
| Green
| Blue

/-- Represents the state of sheep on the island -/
structure SheepState :=
  (red : Nat)
  (green : Nat)
  (blue : Nat)

/-- Represents the rules of color change -/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Red, Color.Green => Color.Blue
  | Color.Red, Color.Blue => Color.Green
  | Color.Green, Color.Blue => Color.Red
  | Color.Green, Color.Red => Color.Blue
  | Color.Blue, Color.Red => Color.Green
  | Color.Blue, Color.Green => Color.Red
  | _, _ => c1

/-- The initial state of sheep -/
def initialState : SheepState :=
  { red := 18, green := 15, blue := 22 }

/-- Checks if all sheep are of the same color -/
def allSameColor (state : SheepState) : Bool :=
  (state.red = 0 ∧ state.green = 0) ∨
  (state.red = 0 ∧ state.blue = 0) ∨
  (state.green = 0 ∧ state.blue = 0)

/-- Theorem: The final state of all sheep is blue -/
theorem final_state_is_blue :
  ∃ (finalState : SheepState),
    allSameColor finalState ∧
    finalState.blue = initialState.red + initialState.green + initialState.blue ∧
    finalState.red = 0 ∧
    finalState.green = 0 :=
  sorry

end NUMINAMATH_CALUDE_final_state_is_blue_l3358_335876


namespace NUMINAMATH_CALUDE_log_319_approximation_l3358_335861

-- Define the logarithm values for 0.317 and 0.318
def log_317 : ℝ := 0.33320
def log_318 : ℝ := 0.3364

-- Define the approximation function for log 0.319
def approx_log_319 : ℝ := log_318 + (log_318 - log_317)

-- Theorem statement
theorem log_319_approximation : 
  abs (approx_log_319 - 0.3396) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_log_319_approximation_l3358_335861


namespace NUMINAMATH_CALUDE_interest_calculation_period_l3358_335835

theorem interest_calculation_period (P n : ℝ) 
  (h1 : P * n / 20 = 40)
  (h2 : P * ((1 + 0.05)^n - 1) = 41) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_calculation_period_l3358_335835


namespace NUMINAMATH_CALUDE_correct_statements_count_l3358_335836

/-- Represents a programming statement --/
inductive Statement
  | Output (cmd : String) (vars : List String)
  | Input (var : String) (value : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if a statement is correct --/
def is_correct (s : Statement) : Bool :=
  match s with
  | Statement.Output cmd vars => cmd = "PRINT"
  | Statement.Input var value => true  -- Simplified for this problem
  | Statement.Assignment lhs rhs => true  -- Simplified for this problem

/-- The list of statements to evaluate --/
def statements : List Statement :=
  [ Statement.Output "INPUT" ["a", "b", "c"]
  , Statement.Input "x" "3"
  , Statement.Assignment "3" "A"
  , Statement.Assignment "A" "B=C"
  ]

/-- Counts the number of correct statements --/
def count_correct (stmts : List Statement) : Nat :=
  stmts.filter is_correct |>.length

theorem correct_statements_count :
  count_correct statements = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l3358_335836


namespace NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l3358_335849

theorem abs_neg_2023_eq_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_eq_2023_l3358_335849


namespace NUMINAMATH_CALUDE_star_sum_larger_than_emilio_sum_l3358_335891

def star_numbers : List ℕ := List.range 50

def emilio_numbers : List ℕ :=
  star_numbers.map (fun n => 
    let tens := n / 10
    let ones := n % 10
    if tens = 2 ∨ tens = 3 then
      (if tens = 2 then 5 else 5) * 10 + ones
    else if ones = 2 ∨ ones = 3 then
      tens * 10 + 5
    else
      n
  )

theorem star_sum_larger_than_emilio_sum :
  (star_numbers.sum - emilio_numbers.sum) = 550 := by
  sorry

end NUMINAMATH_CALUDE_star_sum_larger_than_emilio_sum_l3358_335891


namespace NUMINAMATH_CALUDE_thousand_worries_conforms_to_cognitive_movement_l3358_335852

-- Define cognitive movement
structure CognitiveMovement where
  repetitive : Bool
  infinite : Bool

-- Define a phrase
structure Phrase where
  text : String
  conformsToCognitiveMovement : Bool

-- Define the specific phrase
def thousandWorries : Phrase where
  text := "A thousand worries yield one insight"
  conformsToCognitiveMovement := true -- This is what we want to prove

-- Theorem statement
theorem thousand_worries_conforms_to_cognitive_movement 
  (cm : CognitiveMovement) 
  (h1 : cm.repetitive = true) 
  (h2 : cm.infinite = true) : 
  thousandWorries.conformsToCognitiveMovement = true := by
  sorry


end NUMINAMATH_CALUDE_thousand_worries_conforms_to_cognitive_movement_l3358_335852


namespace NUMINAMATH_CALUDE_f_inequality_implies_a_greater_than_one_l3358_335843

def f (x : ℝ) : ℝ := x * |x|

theorem f_inequality_implies_a_greater_than_one (a : ℝ) : 
  (∀ x ∈ Set.Icc a (a + 1), f (x + 2 * a) > 4 * f x) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_implies_a_greater_than_one_l3358_335843


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3358_335838

theorem cube_volume_ratio (s₂ : ℝ) (h : s₂ > 0) : 
  let s₁ := s₂ * Real.sqrt 3
  (s₁^3) / (s₂^3) = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3358_335838


namespace NUMINAMATH_CALUDE_x_to_y_value_l3358_335879

theorem x_to_y_value (x y : ℝ) (h : (x - 2)^2 + Real.sqrt (y + 1) = 0) : x^y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_value_l3358_335879


namespace NUMINAMATH_CALUDE_supplement_bottles_sum_l3358_335846

/-- Given 5 supplement bottles, where 2 bottles have 30 pills each, and after using 70 pills,
    350 pills remain, prove that the sum of pills in the other 3 bottles is 360. -/
theorem supplement_bottles_sum (total_bottles : Nat) (small_bottles : Nat) (pills_per_small_bottle : Nat)
  (pills_used : Nat) (pills_remaining : Nat) :
  total_bottles = 5 →
  small_bottles = 2 →
  pills_per_small_bottle = 30 →
  pills_used = 70 →
  pills_remaining = 350 →
  ∃ (a b c : Nat), a + b + c = 360 :=
by sorry

end NUMINAMATH_CALUDE_supplement_bottles_sum_l3358_335846


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3358_335829

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 10) ↔
  (x^2 / 25 + y^2 / 21 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l3358_335829


namespace NUMINAMATH_CALUDE_inequality_condition_l3358_335887

theorem inequality_condition (a b : ℝ) : 
  (a < b ∧ b < 0 → 1/a > 1/b) ∧ 
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(a < b ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_condition_l3358_335887


namespace NUMINAMATH_CALUDE_count_five_digit_even_divisible_by_five_l3358_335806

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

theorem count_five_digit_even_divisible_by_five : 
  (Finset.filter (λ n : Nat => 
    10000 ≤ n ∧ n ≤ 99999 ∧ 
    has_only_even_digits n ∧ 
    n % 5 = 0
  ) (Finset.range 100000)).card = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_five_digit_even_divisible_by_five_l3358_335806


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l3358_335871

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, f_domain x → f_inv (f x) = x) ∧
    (∀ y, (∃ x, f_domain x ∧ f x = y) → f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l3358_335871


namespace NUMINAMATH_CALUDE_leonardo_earnings_l3358_335877

/-- Calculates the total earnings for Leonardo over two weeks given the following conditions:
  * Leonardo worked 18 hours in the second week
  * Leonardo worked 13 hours in the first week
  * Leonardo earned $65.70 more in the second week than in the first week
  * His hourly wage remained the same throughout both weeks
-/
def total_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) : ℚ :=
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1 : ℚ)
  (hours_week1 + hours_week2 : ℚ) * hourly_wage

/-- The theorem states that given the specific conditions in the problem,
    Leonardo's total earnings for the two weeks is $407.34. -/
theorem leonardo_earnings :
  total_earnings 13 18 65.70 = 407.34 := by
  sorry

end NUMINAMATH_CALUDE_leonardo_earnings_l3358_335877


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3358_335867

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3358_335867


namespace NUMINAMATH_CALUDE_solution_of_f_1001_l3358_335810

def f₁ (x : ℚ) : ℚ := 2/3 - 3/(3*x+1)

def f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => f₁ x
  | n+1 => f₁ (f n x)

theorem solution_of_f_1001 :
  ∃ x : ℚ, f 1001 x = x - 3 ∧ x = 5/3 := by sorry

end NUMINAMATH_CALUDE_solution_of_f_1001_l3358_335810


namespace NUMINAMATH_CALUDE_f_exp_negative_range_l3358_335813

open Real

theorem f_exp_negative_range (e : ℝ) (h : e = exp 1) :
  let f : ℝ → ℝ := λ x => x - 1 - (e - 1) * log x
  ∀ x : ℝ, f (exp x) < 0 ↔ 0 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_exp_negative_range_l3358_335813


namespace NUMINAMATH_CALUDE_johns_outfit_cost_l3358_335862

theorem johns_outfit_cost (pants_cost : ℝ) (h1 : pants_cost + 1.6 * pants_cost = 130) : pants_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_johns_outfit_cost_l3358_335862


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l3358_335856

theorem sin_cos_sum_identity : 
  Real.sin (70 * π / 180) * Real.sin (10 * π / 180) + 
  Real.cos (10 * π / 180) * Real.cos (70 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l3358_335856


namespace NUMINAMATH_CALUDE_fraction_equality_l3358_335884

theorem fraction_equality (x y z m : ℝ) 
  (h1 : 5 / (x + y) = m / (x + z)) 
  (h2 : m / (x + z) = 13 / (z - y)) : 
  m = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3358_335884


namespace NUMINAMATH_CALUDE_money_bounds_l3358_335803

theorem money_bounds (a b : ℝ) 
  (h1 : 4 * a + b < 60) 
  (h2 : 6 * a - b = 30) : 
  a < 9 ∧ b < 24 := by
  sorry

end NUMINAMATH_CALUDE_money_bounds_l3358_335803


namespace NUMINAMATH_CALUDE_monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l3358_335814

noncomputable section

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Part 1: Monotonicity of f when a = 1
theorem monotonicity_f_when_a_is_1 :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f 1 x₁ > f 1 x₂ ∧
  ∀ x₃ x₄, 2 ≤ x₃ ∧ x₃ < x₄ → f 1 x₃ < f 1 x₄ := by sorry

-- Part 2: Minimum value of a when f has no zeros in (0, 1/2)
theorem min_a_when_f_has_no_zeros :
  (∀ x, 0 < x ∧ x < 1/2 → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_f_when_a_is_1_min_a_when_f_has_no_zeros_l3358_335814


namespace NUMINAMATH_CALUDE_vehicle_value_depreciation_l3358_335816

theorem vehicle_value_depreciation (last_year_value : ℝ) (depreciation_factor : ℝ) (this_year_value : ℝ) :
  last_year_value = 20000 →
  depreciation_factor = 0.8 →
  this_year_value = last_year_value * depreciation_factor →
  this_year_value = 16000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_depreciation_l3358_335816


namespace NUMINAMATH_CALUDE_function_properties_l3358_335894

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (10 + x) = f (10 - x))
  (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
  is_odd f ∧ has_period f 40 := by sorry

end NUMINAMATH_CALUDE_function_properties_l3358_335894


namespace NUMINAMATH_CALUDE_restaurant_bill_change_l3358_335832

/-- Calculates the change received after a restaurant bill payment --/
theorem restaurant_bill_change
  (salmon_price truffled_mac_price chicken_katsu_price seafood_pasta_price black_burger_price wine_price : ℝ)
  (discount_rate service_charge_rate additional_tip_rate : ℝ)
  (payment : ℝ)
  (h_salmon : salmon_price = 40)
  (h_truffled_mac : truffled_mac_price = 20)
  (h_chicken_katsu : chicken_katsu_price = 25)
  (h_seafood_pasta : seafood_pasta_price = 30)
  (h_black_burger : black_burger_price = 15)
  (h_wine : wine_price = 50)
  (h_discount : discount_rate = 0.1)
  (h_service : service_charge_rate = 0.12)
  (h_tip : additional_tip_rate = 0.05)
  (h_payment : payment = 300) :
  let food_cost := salmon_price + truffled_mac_price + chicken_katsu_price + seafood_pasta_price + black_burger_price
  let total_cost := food_cost + wine_price
  let service_charge := service_charge_rate * total_cost
  let bill_before_discount := total_cost + service_charge
  let discount := discount_rate * food_cost
  let bill_after_discount := bill_before_discount - discount
  let additional_tip := additional_tip_rate * bill_after_discount
  let final_bill := bill_after_discount + additional_tip
  payment - final_bill = 101.97 := by sorry

end NUMINAMATH_CALUDE_restaurant_bill_change_l3358_335832


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3358_335873

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, a * x^2 + b * x + c < 0) : 
  b / a < c / a + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3358_335873


namespace NUMINAMATH_CALUDE_tangent_line_power_l3358_335841

/-- Given a curve y = x^2 + ax + b with a tangent line at (0, b) of equation x - y + 1 = 0, prove a^b = 1 -/
theorem tangent_line_power (a b : ℝ) : 
  (∀ x, x^2 + a*x + b = 0 → x - (x^2 + a*x + b) + 1 = 0) → 
  a^b = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_power_l3358_335841


namespace NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3358_335864

theorem min_sum_of_distances (x y z : ℝ) :
  Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) ≥ Real.sqrt 6 :=
by sorry

theorem min_sum_of_distances_achievable :
  ∃ (x y z : ℝ), Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2) = Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_min_sum_of_distances_achievable_l3358_335864


namespace NUMINAMATH_CALUDE_undeclared_majors_fraction_l3358_335860

/-- Represents the distribution of students across different years -/
structure StudentDistribution where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Represents the proportion of students who have not declared a major in each year -/
structure UndeclaredMajors where
  firstYear : Rat
  secondYear : Rat
  thirdYear : Rat
  fourthYear : Rat
  postgraduate : Rat

/-- Calculates the fraction of all students who have not declared a major -/
def fractionUndeclaredMajors (dist : StudentDistribution) (undeclared : UndeclaredMajors) : Rat :=
  dist.firstYear * undeclared.firstYear +
  dist.secondYear * undeclared.secondYear +
  dist.thirdYear * undeclared.thirdYear +
  dist.fourthYear * undeclared.fourthYear +
  dist.postgraduate * undeclared.postgraduate

theorem undeclared_majors_fraction 
  (dist : StudentDistribution)
  (undeclared : UndeclaredMajors)
  (h1 : dist.firstYear = 1/5)
  (h2 : dist.secondYear = 2/5)
  (h3 : dist.thirdYear = 1/5)
  (h4 : dist.fourthYear = 1/10)
  (h5 : dist.postgraduate = 1/10)
  (h6 : undeclared.firstYear = 4/5)
  (h7 : undeclared.secondYear = 3/4)
  (h8 : undeclared.thirdYear = 1/3)
  (h9 : undeclared.fourthYear = 1/6)
  (h10 : undeclared.postgraduate = 1/12) :
  fractionUndeclaredMajors dist undeclared = 14/25 := by
  sorry


end NUMINAMATH_CALUDE_undeclared_majors_fraction_l3358_335860


namespace NUMINAMATH_CALUDE_integer_fraction_pairs_l3358_335815

theorem integer_fraction_pairs : 
  ∀ a b : ℕ+, 
    (∃ k l : ℤ, (a.val^2 + b.val : ℤ) = k * (b.val^2 - a.val) ∧ 
                (b.val^2 + a.val : ℤ) = l * (a.val^2 - b.val)) →
    ((a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_pairs_l3358_335815


namespace NUMINAMATH_CALUDE_student_distribution_l3358_335822

/-- The number of ways to distribute n students between two cities --/
def distribute (n : ℕ) (min1 min2 : ℕ) : ℕ :=
  (Finset.range (n - min1 - min2 + 1)).sum (λ k => Nat.choose n (min1 + k))

/-- The theorem stating the number of arrangements for 6 students --/
theorem student_distribution : distribute 6 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_l3358_335822


namespace NUMINAMATH_CALUDE_fraction_modification_result_l3358_335823

theorem fraction_modification_result (a b : ℤ) (h1 : a.gcd b = 1) 
  (h2 : (a - 1) / (b - 2) = (a + 1) / b) : (a - 1) / (b - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_modification_result_l3358_335823


namespace NUMINAMATH_CALUDE_problem_statement_l3358_335840

theorem problem_statement (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  a^2 * b + a * b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3358_335840


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_zero_l3358_335847

theorem min_sum_squares (x y s : ℝ) (h : x + y + s = 0) : 
  ∀ a b c : ℝ, a + b + c = 0 → x^2 + y^2 + s^2 ≤ a^2 + b^2 + c^2 :=
by
  sorry

theorem min_sum_squares_zero (x y s : ℝ) (h : x + y + s = 0) : 
  ∃ a b c : ℝ, a + b + c = 0 ∧ a^2 + b^2 + c^2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_zero_l3358_335847


namespace NUMINAMATH_CALUDE_simplify_expression_l3358_335812

theorem simplify_expression (a b : ℝ) : 4*a + 5*b - a - 7*b = 3*a - 2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3358_335812


namespace NUMINAMATH_CALUDE_original_length_is_one_meter_l3358_335845

/-- The length of the line after erasing part of it, in centimeters -/
def remaining_length : ℝ := 76

/-- The length that was erased from the line, in centimeters -/
def erased_length : ℝ := 24

/-- The number of centimeters in one meter -/
def cm_per_meter : ℝ := 100

/-- The theorem stating that the original length of the line was 1 meter -/
theorem original_length_is_one_meter : 
  (remaining_length + erased_length) / cm_per_meter = 1 := by sorry

end NUMINAMATH_CALUDE_original_length_is_one_meter_l3358_335845


namespace NUMINAMATH_CALUDE_water_transfer_problem_l3358_335833

theorem water_transfer_problem (initial_volume : ℝ) (loss_percentage : ℝ) (hemisphere_volume : ℝ) : 
  initial_volume = 10936 →
  loss_percentage = 2.5 →
  hemisphere_volume = 4 →
  ⌈(initial_volume * (1 - loss_percentage / 100)) / hemisphere_volume⌉ = 2666 := by
  sorry

end NUMINAMATH_CALUDE_water_transfer_problem_l3358_335833


namespace NUMINAMATH_CALUDE_bed_fraction_of_plot_l3358_335857

/-- Given a square plot of land with side length 8 units, prove that the fraction
    of the plot occupied by 13 beds (12 in an outer band and 1 central square)
    is 15/32 of the total area. -/
theorem bed_fraction_of_plot (plot_side : ℝ) (total_beds : ℕ) 
  (outer_beds : ℕ) (inner_bed_side : ℝ) :
  plot_side = 8 →
  total_beds = 13 →
  outer_beds = 12 →
  inner_bed_side = 4 →
  (outer_beds * (plot_side - inner_bed_side) + inner_bed_side ^ 2 / 2) / plot_side ^ 2 = 15 / 32 := by
  sorry

#check bed_fraction_of_plot

end NUMINAMATH_CALUDE_bed_fraction_of_plot_l3358_335857


namespace NUMINAMATH_CALUDE_sqrt_9x_lt_3x_squared_iff_x_gt_1_l3358_335888

theorem sqrt_9x_lt_3x_squared_iff_x_gt_1 :
  ∀ x : ℝ, x > 0 → (Real.sqrt (9 * x) < 3 * x^2 ↔ x > 1) := by
sorry

end NUMINAMATH_CALUDE_sqrt_9x_lt_3x_squared_iff_x_gt_1_l3358_335888


namespace NUMINAMATH_CALUDE_alan_cd_purchase_cost_l3358_335834

theorem alan_cd_purchase_cost :
  let avnPrice : ℝ := 12
  let darkPrice : ℝ := 2 * avnPrice
  let darkTotal : ℝ := 2 * darkPrice
  let otherTotal : ℝ := darkTotal + avnPrice
  let ninetyPrice : ℝ := 0.4 * otherTotal
  darkTotal + avnPrice + ninetyPrice = 84 := by
  sorry

end NUMINAMATH_CALUDE_alan_cd_purchase_cost_l3358_335834


namespace NUMINAMATH_CALUDE_circle_equation_l3358_335885

/-- The equation of a circle passing through point P(2,5) with center C(8,-3) -/
theorem circle_equation (x y : ℝ) : 
  let P : ℝ × ℝ := (2, 5)
  let C : ℝ × ℝ := (8, -3)
  (x - C.1)^2 + (y - C.2)^2 = (P.1 - C.1)^2 + (P.2 - C.2)^2 ↔ 
  (x - 8)^2 + (y + 3)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3358_335885


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3358_335850

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m → p ∣ n)

def is_multiple_of_five (n : ℕ) : Prop :=
  5 ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ),
    has_all_prime_factors bob_number alice_number ∧
    is_multiple_of_five bob_number ∧
    (∀ k : ℕ, k < bob_number →
      ¬(has_all_prime_factors k alice_number ∧ is_multiple_of_five k)) ∧
    bob_number = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3358_335850


namespace NUMINAMATH_CALUDE_sqrt_75_plus_30sqrt3_form_l3358_335809

def is_square_free (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m.sqrt ^ 2 ∣ n → m.sqrt ^ 2 = 1

theorem sqrt_75_plus_30sqrt3_form :
  ∃ (a b c : ℤ), (c : ℝ) > 0 ∧ is_square_free c.toNat ∧
  Real.sqrt (75 + 30 * Real.sqrt 3) = a + b * Real.sqrt c ∧
  a + b + c = 12 :=
sorry

end NUMINAMATH_CALUDE_sqrt_75_plus_30sqrt3_form_l3358_335809


namespace NUMINAMATH_CALUDE_a_plus_b_value_l3358_335892

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_value (a b : ℝ) : 
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = Set.Ioc 3 4) →
  a + b = -7 := by
  sorry


end NUMINAMATH_CALUDE_a_plus_b_value_l3358_335892


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3358_335870

-- Define the equation
def equation (x : ℝ) : Prop := |x - 1| = 3 * |x + 3|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- State the theorem
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = -7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3358_335870


namespace NUMINAMATH_CALUDE_lavender_bouquet_cost_l3358_335896

/-- The cost of a bouquet is directly proportional to the number of lavenders it contains. -/
def is_proportional (cost : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → cost n / n = cost m / m

/-- Given that a bouquet of 15 lavenders costs $25 and the price is directly proportional
    to the number of lavenders, prove that a bouquet of 50 lavenders costs $250/3. -/
theorem lavender_bouquet_cost (cost : ℕ → ℚ)
    (h_prop : is_proportional cost)
    (h_15 : cost 15 = 25) :
    cost 50 = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lavender_bouquet_cost_l3358_335896


namespace NUMINAMATH_CALUDE_cos_seven_pi_fourth_l3358_335842

theorem cos_seven_pi_fourth : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_fourth_l3358_335842


namespace NUMINAMATH_CALUDE_sector_central_angle_l3358_335878

theorem sector_central_angle (circumference : ℝ) (area : ℝ) :
  circumference = 6 →
  area = 2 →
  ∃ (r l : ℝ),
    l + 2*r = 6 ∧
    (1/2) * l * r = 2 ∧
    (l / r = 1 ∨ l / r = 4) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3358_335878


namespace NUMINAMATH_CALUDE_definite_integral_x_squared_l3358_335851

theorem definite_integral_x_squared : ∫ x in (0 : ℝ)..1, x^2 = 1/3 := by sorry

end NUMINAMATH_CALUDE_definite_integral_x_squared_l3358_335851


namespace NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_one_third_l3358_335868

theorem sqrt_three_minus_sqrt_one_third : 
  Real.sqrt 3 - Real.sqrt (1/3) = (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_sqrt_one_third_l3358_335868


namespace NUMINAMATH_CALUDE_leftmost_digit_in_base9_is_5_l3358_335893

/-- Represents a number in base-3 as a list of digits -/
def Base3Number := List Nat

/-- Converts a base-3 number to its decimal (base-10) representation -/
def toDecimal (n : Base3Number) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base-9 representation -/
def toBase9 (n : Nat) : List Nat :=
  sorry

/-- Gets the leftmost digit of a list of digits -/
def leftmostDigit (digits : List Nat) : Nat :=
  digits.head!

/-- The given base-3 number -/
def givenNumber : Base3Number :=
  [1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2]

theorem leftmost_digit_in_base9_is_5 :
  leftmostDigit (toBase9 (toDecimal givenNumber)) = 5 :=
sorry

end NUMINAMATH_CALUDE_leftmost_digit_in_base9_is_5_l3358_335893


namespace NUMINAMATH_CALUDE_greatest_base8_digit_sum_l3358_335830

/-- Represents a positive integer in base 8 --/
structure Base8Int where
  digits : List Nat
  positive : digits ≠ []
  valid : ∀ d ∈ digits, d < 8

/-- Converts a Base8Int to its decimal representation --/
def toDecimal (n : Base8Int) : Nat :=
  sorry

/-- Computes the sum of digits of a Base8Int --/
def digitSum (n : Base8Int) : Nat :=
  sorry

/-- The theorem to be proved --/
theorem greatest_base8_digit_sum :
  (∃ (n : Base8Int), toDecimal n < 1728 ∧
    ∀ (m : Base8Int), toDecimal m < 1728 → digitSum m ≤ digitSum n) ∧
  (∀ (n : Base8Int), toDecimal n < 1728 → digitSum n ≤ 23) :=
sorry

end NUMINAMATH_CALUDE_greatest_base8_digit_sum_l3358_335830


namespace NUMINAMATH_CALUDE_f_composition_one_ninth_l3358_335825

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^3 + 1 else Real.log x / Real.log 3

theorem f_composition_one_ninth : f (f (1/9)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_one_ninth_l3358_335825


namespace NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l3358_335818

/-- Represents a circular pizza with pepperoni toppings -/
structure PizzaWithPepperoni where
  pizzaDiameter : ℝ
  pepperoniAcrossDiameter : ℕ
  totalPepperoni : ℕ

/-- Calculates the fraction of the pizza covered by pepperoni -/
def pepperoniCoverage (pizza : PizzaWithPepperoni) : ℚ :=
  sorry

/-- Theorem stating that the fraction of the pizza covered by pepperoni is 4/9 -/
theorem pepperoni_coverage_is_four_ninths (pizza : PizzaWithPepperoni) 
  (h1 : pizza.pizzaDiameter = 18)
  (h2 : pizza.pepperoniAcrossDiameter = 9)
  (h3 : pizza.totalPepperoni = 36) : 
  pepperoniCoverage pizza = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_coverage_is_four_ninths_l3358_335818


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3358_335869

/-- Represents a card color -/
inductive CardColor
| Red
| Black
| Blue
| White

/-- Represents a person -/
inductive Person
| A
| B
| C
| D

/-- Represents the distribution of cards to people -/
def Distribution := Person → CardColor

/-- The event "A receives the red card" -/
def event_A_red (d : Distribution) : Prop := d Person.A = CardColor.Red

/-- The event "B receives the red card" -/
def event_B_red (d : Distribution) : Prop := d Person.B = CardColor.Red

/-- The set of all possible distributions -/
def all_distributions : Set Distribution :=
  {d | ∀ c : CardColor, ∃! p : Person, d p = c}

theorem events_mutually_exclusive_but_not_opposite :
  (∀ d : Distribution, d ∈ all_distributions →
    ¬(event_A_red d ∧ event_B_red d)) ∧
  (∃ d : Distribution, d ∈ all_distributions ∧
    ¬event_A_red d ∧ ¬event_B_red d) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_opposite_l3358_335869


namespace NUMINAMATH_CALUDE_max_red_points_is_13_l3358_335820

/-- Represents a point on the circle -/
inductive Point
| Red : ℕ → Point  -- Red point with number of connections
| Blue : Point

/-- The configuration of points on the circle -/
structure CircleConfig where
  points : Finset Point
  red_count : ℕ
  blue_count : ℕ
  total_count : ℕ
  total_is_25 : total_count = 25
  total_is_sum : total_count = red_count + blue_count
  unique_connections : ∀ p q : Point, p ∈ points → q ∈ points → 
    p ≠ q → (∃ n m : ℕ, p = Point.Red n ∧ q = Point.Red m) → n ≠ m

/-- The maximum number of red points possible -/
def max_red_points : ℕ := 13

/-- Theorem stating that the maximum number of red points is 13 -/
theorem max_red_points_is_13 (config : CircleConfig) : 
  config.red_count ≤ max_red_points :=
sorry

end NUMINAMATH_CALUDE_max_red_points_is_13_l3358_335820


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3358_335863

theorem election_winner_percentage (winner_votes loser_votes total_votes : ℕ) 
  (h1 : winner_votes = 899)
  (h2 : winner_votes - loser_votes = 348)
  (h3 : total_votes = winner_votes + loser_votes) :
  (winner_votes : ℝ) / (total_votes : ℝ) * 100 = 899 / 1450 * 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3358_335863


namespace NUMINAMATH_CALUDE_standing_arrangements_l3358_335848

def number_of_people : ℕ := 5

-- Function to calculate the number of ways person A and B can stand next to each other
def ways_next_to_each_other (n : ℕ) : ℕ := sorry

-- Function to calculate the total number of ways n people can stand
def total_ways (n : ℕ) : ℕ := sorry

-- Function to calculate the number of ways person A and B can stand not next to each other
def ways_not_next_to_each_other (n : ℕ) : ℕ := sorry

theorem standing_arrangements :
  (ways_next_to_each_other number_of_people = 48) ∧
  (ways_not_next_to_each_other number_of_people = 72) := by sorry

end NUMINAMATH_CALUDE_standing_arrangements_l3358_335848


namespace NUMINAMATH_CALUDE_egg_order_problem_l3358_335897

theorem egg_order_problem (total : ℚ) : 
  (total > 0) →
  (total * (1 - 1/4) * (1 - 2/3) = 9) →
  total = 18 := by
sorry

end NUMINAMATH_CALUDE_egg_order_problem_l3358_335897


namespace NUMINAMATH_CALUDE_money_distribution_l3358_335839

/-- The problem of distributing money among boys -/
theorem money_distribution (total_amount : ℕ) (extra_per_boy : ℕ) : 
  (total_amount = 5040) →
  (extra_per_boy = 80) →
  ∃ (x : ℕ), 
    (x * (total_amount / 18 + extra_per_boy) = total_amount) ∧
    (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3358_335839


namespace NUMINAMATH_CALUDE_always_true_inequality_l3358_335802

theorem always_true_inequality (a b x : ℝ) (h : a > b) : a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l3358_335802


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3358_335828

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c = 4) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 / 12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3358_335828


namespace NUMINAMATH_CALUDE_anita_apples_l3358_335811

/-- The number of apples Anita has, given the number of students and apples per student -/
def total_apples (num_students : ℕ) (apples_per_student : ℕ) : ℕ :=
  num_students * apples_per_student

/-- Theorem: Anita has 360 apples -/
theorem anita_apples : total_apples 60 6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_anita_apples_l3358_335811


namespace NUMINAMATH_CALUDE_range_of_m_l3358_335855

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3358_335855


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l3358_335844

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 12) (h2 : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l3358_335844


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3358_335853

theorem no_function_satisfies_conditions :
  ¬∃ (f : ℝ → ℝ) (a b : ℝ),
    (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
    (a > 0 ∧ b > 0) ∧
    (∀ x : ℝ, f (x^2) - (f (a * x + b))^2 ≥ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l3358_335853


namespace NUMINAMATH_CALUDE_min_value_expression_l3358_335831

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ y₀ / x₀ + 16 * x₀ / (2 * x₀ + y₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3358_335831


namespace NUMINAMATH_CALUDE_parallelogram_area_theorem_l3358_335899

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is the midpoint of a line segment -/
def isMidpoint (M A B : Point) : Prop := sorry

/-- Checks if three points are collinear -/
def collinear (A B C : Point) : Prop := sorry

theorem parallelogram_area_theorem (ABCD : Parallelogram) (E F : Point) :
  collinear C E F →
  isMidpoint F ABCD.A ABCD.B →
  areaTriangle ⟨ABCD.B, E, C⟩ = 100 →
  areaQuadrilateral ⟨ABCD.A, F, E, ABCD.D⟩ = 250 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_theorem_l3358_335899


namespace NUMINAMATH_CALUDE_check_cashing_error_l3358_335800

theorem check_cashing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  x > y →
  100 * y + x - (100 * x + y) = 2187 →
  x - y = 22 := by
sorry

end NUMINAMATH_CALUDE_check_cashing_error_l3358_335800


namespace NUMINAMATH_CALUDE_inheritance_calculation_l3358_335827

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 33000

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The additional fee in dollars -/
def additional_fee : ℝ := 50

/-- The total amount paid for taxes and fee in dollars -/
def total_paid : ℝ := 12000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + additional_fee +
  state_tax_rate * ((1 - federal_tax_rate) * inheritance - additional_fee) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l3358_335827


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3358_335898

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 ∧ (-2 + y) / 2 = 5 → x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l3358_335898


namespace NUMINAMATH_CALUDE_line_circle_properties_l3358_335826

-- Define the line l and circle C
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - m - 3 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 4 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem line_circle_properties (m : ℝ) :
  (∀ x y, line_l m x y → (x = 1 ∧ y = 1)) ∧
  (∃ x y, line_l m x y ∧ circle_C x y) ∧
  (∃ x y, line_l m x y ∧ 
    Real.sqrt ((x - circle_center.1)^2 + (y - circle_center.2)^2) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_properties_l3358_335826


namespace NUMINAMATH_CALUDE_final_concentration_is_correct_l3358_335854

/-- Represents the volume of saline solution in the cup -/
def initial_volume : ℝ := 1

/-- Represents the initial concentration of the saline solution -/
def initial_concentration : ℝ := 0.16

/-- Represents the volume ratio of the large ball -/
def large_ball_ratio : ℝ := 10

/-- Represents the volume ratio of the medium ball -/
def medium_ball_ratio : ℝ := 4

/-- Represents the volume ratio of the small ball -/
def small_ball_ratio : ℝ := 3

/-- Represents the percentage of solution that overflows when the small ball is immersed -/
def overflow_percentage : ℝ := 0.1

/-- Calculates the final concentration of the saline solution after the process -/
def final_concentration : ℝ := sorry

/-- Theorem stating that the final concentration is approximately 10.7% -/
theorem final_concentration_is_correct : 
  ∀ ε > 0, |final_concentration - 0.107| < ε := by sorry

end NUMINAMATH_CALUDE_final_concentration_is_correct_l3358_335854


namespace NUMINAMATH_CALUDE_equation_solutions_l3358_335880

theorem equation_solutions :
  (∀ x : ℝ, 9 * x^2 - 25 = 0 ↔ x = 5/3 ∨ x = -5/3) ∧
  (∀ x : ℝ, (x + 1)^3 - 27 = 0 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3358_335880


namespace NUMINAMATH_CALUDE_min_odd_correct_answers_l3358_335886

/-- Represents the number of correct answers a student can give -/
inductive CorrectAnswers
  | zero
  | one
  | two
  | three
  | four

/-- Represents the distribution of correct answers among students -/
structure AnswerDistribution where
  total : Nat
  zero : Nat
  one : Nat
  two : Nat
  three : Nat
  four : Nat
  sum_constraint : total = zero + one + two + three + four

/-- Checks if a distribution satisfies the problem constraints -/
def satisfies_constraints (d : AnswerDistribution) : Prop :=
  d.total = 50 ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.three)).card ≥ 1) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.two)).card ≥ 2) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.one)).card ≥ 3) ∧
  (∀ (s : Finset Nat), s.card = 40 → s.filter (λ i => i < d.total) ≠ ∅ → 
    (s.filter (λ i => i < d.zero)).card ≥ 4)

/-- The main theorem to prove -/
theorem min_odd_correct_answers (d : AnswerDistribution) 
  (h : satisfies_constraints d) : d.one + d.three ≥ 23 := by
  sorry


end NUMINAMATH_CALUDE_min_odd_correct_answers_l3358_335886


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l3358_335882

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 6*x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line
def line (x y : ℝ) (a : ℝ) : Prop := x - y + a = 0

-- Define the perpendicularity condition
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem circle_and_line_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    -- The curve intersects the coordinate axes at points on circle C
    (curve 0 y1 ∧ circle_C 0 y1) ∧
    (curve x1 0 ∧ circle_C x1 0) ∧
    (curve x2 0 ∧ circle_C x2 0) ∧
    -- Circle C intersects the line at A(x1, y1) and B(x2, y2)
    (circle_C x1 y1 ∧ line x1 y1 (-1)) ∧
    (circle_C x2 y2 ∧ line x2 y2 (-1)) ∧
    -- OA ⊥ OB
    perpendicular x1 y1 x2 y2 :=
  sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l3358_335882


namespace NUMINAMATH_CALUDE_fair_attendance_difference_l3358_335889

theorem fair_attendance_difference : 
  ∀ (last_year : ℕ) (this_year : ℕ) (next_year : ℕ),
    this_year = 600 →
    next_year = 2 * this_year →
    last_year + this_year + next_year = 2800 →
    last_year < next_year →
    next_year - last_year = 200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_difference_l3358_335889


namespace NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l3358_335874

/-- Represents the craft selling scenario -/
structure CraftSelling where
  cost_price : ℕ
  base_price : ℕ
  base_volume : ℕ
  price_volume_ratio : ℕ
  max_price : ℕ

/-- Calculates the daily sales volume based on the selling price -/
def daily_volume (cs : CraftSelling) (price : ℕ) : ℤ :=
  cs.base_volume - cs.price_volume_ratio * (price - cs.base_price)

/-- Calculates the daily profit based on the selling price -/
def daily_profit (cs : CraftSelling) (price : ℕ) : ℤ :=
  (price - cs.cost_price) * daily_volume cs price

/-- The craft selling scenario for the given problem -/
def craft_scenario : CraftSelling := {
  cost_price := 30
  base_price := 40
  base_volume := 80
  price_volume_ratio := 2
  max_price := 55
}

/-- Theorem for the daily sales profit at 45 yuan -/
theorem profit_at_45 : daily_profit craft_scenario 45 = 1050 := by sorry

/-- Theorem for the selling price that achieves 1200 yuan daily profit -/
theorem price_for_1200_profit :
  ∃ (price : ℕ), price ≤ craft_scenario.max_price ∧ daily_profit craft_scenario price = 1200 ∧
  ∀ (p : ℕ), p ≤ craft_scenario.max_price → daily_profit craft_scenario p = 1200 → p = price := by sorry

end NUMINAMATH_CALUDE_profit_at_45_price_for_1200_profit_l3358_335874


namespace NUMINAMATH_CALUDE_complex_sum_as_polar_l3358_335881

open Complex

theorem complex_sum_as_polar : ∃ (r θ : ℝ),
  7 * exp (3 * π * I / 14) - 7 * exp (10 * π * I / 21) = r * exp (θ * I) ∧
  r = Real.sqrt (2 - Real.sqrt 3 / 2) ∧
  θ = 29 * π / 84 + Real.arctan (-2 / (Real.sqrt 3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_as_polar_l3358_335881


namespace NUMINAMATH_CALUDE_extreme_points_and_inequality_l3358_335866

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - x

theorem extreme_points_and_inequality (a : ℝ) (h : a > 1/2) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    (∀ (y : ℝ), |y - x₁| < δ → f a y ≤ f a x₁ + ε) ∧
    (∀ (y : ℝ), |y - x₂| < δ → f a y ≥ f a x₂ - ε)) ∧
  f a x₂ < 1 + (Real.sin x₂ - x₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_and_inequality_l3358_335866


namespace NUMINAMATH_CALUDE_segment_length_line_circle_l3358_335883

/-- The length of the segment cut by a line from a circle -/
theorem segment_length_line_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : 
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    2 * Real.sqrt (r^2 - (a*x₀ + b*y₀ + c)^2 / (a^2 + b^2)) = Real.sqrt 3) →
  x₀ = 1 ∧ y₀ = 0 ∧ r = 1 ∧ a = 1 ∧ b = Real.sqrt 3 ∧ c = -2 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_line_circle_l3358_335883


namespace NUMINAMATH_CALUDE_parallel_planes_lines_l3358_335872

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the contained relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Define the intersection relation for lines
variable (line_intersect : Line → Line → Prop)

-- Theorem statement
theorem parallel_planes_lines
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  (¬ line_intersect a b) ∧
  (line_parallel a b ∨ line_skew a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_l3358_335872


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3358_335875

theorem simplify_polynomial (a : ℝ) : (1 : ℝ) * (3 * a) * (5 * a^2) * (7 * a^3) * (9 * a^4) = 945 * a^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3358_335875
