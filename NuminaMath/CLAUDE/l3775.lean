import Mathlib

namespace NUMINAMATH_CALUDE_final_sum_is_correct_l3775_377597

/-- Represents the state of the three calculators -/
structure CalculatorState where
  calc1 : ℤ
  calc2 : ℤ
  calc3 : ℤ

/-- Applies the operations to the calculator state -/
def applyOperations (state : CalculatorState) : CalculatorState :=
  { calc1 := 2 * state.calc1,
    calc2 := state.calc2 ^ 2,
    calc3 := -state.calc3 }

/-- Iterates the operations n times -/
def iterateOperations (n : ℕ) (state : CalculatorState) : CalculatorState :=
  match n with
  | 0 => state
  | n + 1 => applyOperations (iterateOperations n state)

/-- The initial state of the calculators -/
def initialState : CalculatorState :=
  { calc1 := 2, calc2 := 0, calc3 := -2 }

/-- The main theorem to prove -/
theorem final_sum_is_correct :
  let finalState := iterateOperations 51 initialState
  finalState.calc1 + finalState.calc2 + finalState.calc3 = 2^52 + 2 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_is_correct_l3775_377597


namespace NUMINAMATH_CALUDE_total_files_deleted_l3775_377574

def initial_files : ℕ := 24
def final_files : ℕ := 21

def deletions : List ℕ := [5, 10]
def additions : List ℕ := [7, 5]

theorem total_files_deleted :
  (initial_files + additions.sum - deletions.sum = final_files) →
  deletions.sum = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_files_deleted_l3775_377574


namespace NUMINAMATH_CALUDE_expression_simplification_l3775_377509

theorem expression_simplification :
  (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt (1/2) - |-1/2| = 1 + (Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3775_377509


namespace NUMINAMATH_CALUDE_polynomial_composition_l3775_377569

theorem polynomial_composition (f g : ℝ → ℝ) :
  (∀ x, f x = x^2) →
  (∃ a b c : ℝ, ∀ x, g x = a * x^2 + b * x + c) →
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_l3775_377569


namespace NUMINAMATH_CALUDE_eight_pencils_l3775_377525

/-- Represents Sam's pen and pencil collection -/
structure SamsCollection where
  pencils : ℕ
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ

/-- The conditions of Sam's collection -/
def valid_collection (c : SamsCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.red_pens = c.pencils - 2 ∧
  c.black_pens + c.blue_pens + c.red_pens = 48

/-- Theorem stating that in a valid collection, there are 8 pencils -/
theorem eight_pencils (c : SamsCollection) (h : valid_collection c) : c.pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_pencils_l3775_377525


namespace NUMINAMATH_CALUDE_negate_all_men_are_good_drivers_l3775_377502

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Man : U → Prop)
variable (GoodDriver : U → Prop)

-- Define the statements
def AllMenAreGoodDrivers : Prop := ∀ x, Man x → GoodDriver x
def AtLeastOneManIsBadDriver : Prop := ∃ x, Man x ∧ ¬GoodDriver x

-- Theorem to prove
theorem negate_all_men_are_good_drivers :
  AtLeastOneManIsBadDriver U Man GoodDriver ↔ ¬(AllMenAreGoodDrivers U Man GoodDriver) :=
sorry

end NUMINAMATH_CALUDE_negate_all_men_are_good_drivers_l3775_377502


namespace NUMINAMATH_CALUDE_horner_method_f_neg_four_l3775_377568

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 - 8x^2 + 6x^4 + 5x^5 + 3x^6 -/
def f (x : ℤ) : ℤ := 12 - 8*x^2 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_f_neg_four :
  horner_eval [3, 5, 6, 0, -8, 0, 12] (-4) = f (-4) ∧ f (-4) = -845 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_neg_four_l3775_377568


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3775_377553

/-- A quadratic function passing through (1,0) and (-3,0) with minimum value 25 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_one : a + b + c = 0
  passes_through_neg_three : 9*a - 3*b + c = 0
  has_minimum_25 : ∀ x, a*x^2 + b*x + c ≥ 25

/-- The sum of coefficients a + b + c equals -75/4 for the given quadratic function -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = -75/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3775_377553


namespace NUMINAMATH_CALUDE_article_sale_loss_percentage_l3775_377511

theorem article_sale_loss_percentage 
  (cost : ℝ) 
  (original_price : ℝ) 
  (discounted_price : ℝ) 
  (h1 : original_price = cost * 1.35)
  (h2 : discounted_price = original_price * (2/3)) :
  (cost - discounted_price) / cost * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_article_sale_loss_percentage_l3775_377511


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3775_377516

theorem reciprocal_of_negative_one_third :
  let x : ℚ := -1/3
  let y : ℚ := -3
  (x * y = 1) → (∀ z : ℚ, x * z = 1 → z = y) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_third_l3775_377516


namespace NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_is_one_l3775_377543

/-- The probability that at least 4 people stay for the entire game, given that
    8 people come to a basketball game, 4 are certain to stay, and 4 have a
    1/3 probability of staying. -/
theorem basketball_game_probability : Real :=
  let total_people : ℕ := 8
  let certain_stayers : ℕ := 4
  let uncertain_stayers : ℕ := 4
  let stay_probability : Real := 1/3
  1

/-- Proof that the probability is indeed 1. -/
theorem basketball_game_probability_is_one :
  basketball_game_probability = 1 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_probability_basketball_game_probability_is_one_l3775_377543


namespace NUMINAMATH_CALUDE_boat_travel_time_difference_l3775_377571

def distance : ℝ := 90
def downstream_time : ℝ := 2.5191640969412834

theorem boat_travel_time_difference (v : ℝ) : 
  v > 3 →
  distance / (v - 3) - distance / (v + 3) = downstream_time →
  ∃ (diff : ℝ), abs (diff - 0.5088359030587166) < 1e-10 ∧ 
                 distance / (v - 3) - downstream_time = diff :=
by sorry

end NUMINAMATH_CALUDE_boat_travel_time_difference_l3775_377571


namespace NUMINAMATH_CALUDE_product_upper_bound_l3775_377515

theorem product_upper_bound (x y z t : ℝ) 
  (h_order : x ≤ y ∧ y ≤ z ∧ z ≤ t) 
  (h_sum : x*y + x*z + x*t + y*z + y*t + z*t = 1) : 
  x*t < 1/3 ∧ ∀ C, (∀ a b c d, a ≤ b ∧ b ≤ c ∧ c ≤ d → 
    a*b + a*c + a*d + b*c + b*d + c*d = 1 → a*d < C) → 1/3 ≤ C :=
by sorry

end NUMINAMATH_CALUDE_product_upper_bound_l3775_377515


namespace NUMINAMATH_CALUDE_multiply_decimals_l3775_377593

theorem multiply_decimals : 0.9 * 0.007 = 0.0063 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l3775_377593


namespace NUMINAMATH_CALUDE_natural_number_equation_solutions_l3775_377575

theorem natural_number_equation_solutions :
  ∀ (a b c d : ℕ), 
    a * b = c + d ∧ a + b = c * d →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2) ∨
     (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 5 ∧ d = 1) ∨
     (a = 2 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∧
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 5) ∨
     (a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 5)) :=
by sorry

end NUMINAMATH_CALUDE_natural_number_equation_solutions_l3775_377575


namespace NUMINAMATH_CALUDE_bin_drawing_probability_l3775_377538

def bin_probability : ℚ :=
  let total_balls : ℕ := 20
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let drawn_balls : ℕ := 4
  let favorable_outcomes : ℕ := (Nat.choose black_balls 2) * (Nat.choose white_balls 2)
  let total_outcomes : ℕ := Nat.choose total_balls drawn_balls
  (favorable_outcomes : ℚ) / total_outcomes

theorem bin_drawing_probability : bin_probability = 135 / 323 := by
  sorry

end NUMINAMATH_CALUDE_bin_drawing_probability_l3775_377538


namespace NUMINAMATH_CALUDE_inequality_proof_l3775_377518

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3775_377518


namespace NUMINAMATH_CALUDE_egg_box_count_l3775_377526

theorem egg_box_count (total_eggs : Real) (eggs_per_box : Real) (h1 : total_eggs = 3.0) (h2 : eggs_per_box = 1.5) :
  (total_eggs / eggs_per_box : Real) = 2 := by
  sorry

end NUMINAMATH_CALUDE_egg_box_count_l3775_377526


namespace NUMINAMATH_CALUDE_product_as_difference_of_squares_l3775_377520

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ((a + b) / 2)^2 - ((a - b) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_as_difference_of_squares_l3775_377520


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3775_377528

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_sequence (a : ℕ) : Prop :=
  fib (a + 4) = 2 * fib (a + 2) - fib a

-- Define the sum condition
def sum_condition (a : ℕ) : Prop :=
  a + (a + 2) + (a + 4) = 2500

-- Theorem statement
theorem fibonacci_arithmetic_sequence :
  ∃ a : ℕ, is_arithmetic_sequence a ∧ sum_condition a ∧ a = 831 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3775_377528


namespace NUMINAMATH_CALUDE_correct_equation_l3775_377523

theorem correct_equation (x y : ℝ) : 3 * x^2 * y - 4 * y * x^2 = -x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3775_377523


namespace NUMINAMATH_CALUDE_simplify_expression_l3775_377552

theorem simplify_expression : (6 + 6 + 12) / 3 - 2 * 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3775_377552


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3775_377539

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3775_377539


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3775_377504

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 2*x^9 - x^6) =
  15*x^13 - 19*x^12 - 6*x^11 + 6*x^10 - 4*x^9 - 3*x^7 + 2*x^6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3775_377504


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3775_377573

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = 180 ∧ -- Angle sum theorem
  a / Real.sin A = b / Real.sin B ∧ -- Sine rule
  a / Real.sin A = c / Real.sin C -- Sine rule

-- State the theorem
theorem longest_side_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle_ABC A B C a b c →
  B = 135 →
  C = 15 →
  a = 5 →
  b = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3775_377573


namespace NUMINAMATH_CALUDE_computable_logarithms_l3775_377501

def is_computable (n : ℕ) : Prop :=
  ∃ (m n p : ℕ), n = 2^m * 3^n * 5^p ∧ n ≤ 100

def computable_set : Set ℕ :=
  {n : ℕ | n ≥ 1 ∧ n ≤ 100 ∧ is_computable n}

theorem computable_logarithms :
  computable_set = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100} :=
by sorry

end NUMINAMATH_CALUDE_computable_logarithms_l3775_377501


namespace NUMINAMATH_CALUDE_price_reduction_effect_l3775_377598

theorem price_reduction_effect (P S : ℝ) (P_reduced : ℝ) (S_increased : ℝ) :
  P_reduced = 0.8 * P →
  S_increased = 1.8 * S →
  P_reduced * S_increased = 1.44 * P * S :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_effect_l3775_377598


namespace NUMINAMATH_CALUDE_cube_side_ratio_l3775_377544

theorem cube_side_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (6 * a^2) / (6 * b^2) = 36 → a / b = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_ratio_l3775_377544


namespace NUMINAMATH_CALUDE_exists_range_sum_and_even_count_l3775_377527

/-- Sum of integers from n to m, inclusive -/
def sum_range (n m : ℤ) : ℤ := (m - n + 1) * (n + m) / 2

/-- Number of even integers from n to m, inclusive -/
def count_even (n m : ℤ) : ℤ :=
  if (n % 2 = m % 2) then (m - n) / 2 + 1 else (m - n + 1) / 2

/-- Theorem stating the existence of a range satisfying the given conditions -/
theorem exists_range_sum_and_even_count :
  ∃ (n m : ℤ), n ≤ m ∧ sum_range n m + count_even n m = 641 :=
sorry

end NUMINAMATH_CALUDE_exists_range_sum_and_even_count_l3775_377527


namespace NUMINAMATH_CALUDE_point_four_units_from_origin_l3775_377596

theorem point_four_units_from_origin (x : ℝ) : 
  |x| = 4 → x = 4 ∨ x = -4 := by
sorry

end NUMINAMATH_CALUDE_point_four_units_from_origin_l3775_377596


namespace NUMINAMATH_CALUDE_range_of_a_l3775_377535

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 3, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0

-- Define the set of a values that satisfy the conditions
def A : Set ℝ := {a | (p a ∨ q a) ∧ ¬(p a ∧ q a)}

-- Theorem statement
theorem range_of_a : A = Set.Icc (-1) 1 ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3775_377535


namespace NUMINAMATH_CALUDE_min_a_plus_b_l3775_377536

theorem min_a_plus_b (x y a b : ℝ) : 
  2*x - y + 2 ≥ 0 →
  8*x - y - 4 ≤ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 2*x' - y' + 2 ≥ 0 → 8*x' - y' - 4 ≤ 0 → x' ≥ 0 → y' ≥ 0 → a*x' + y' ≤ 8) →
  a*x + y = 8 →
  a + b ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_a_plus_b_l3775_377536


namespace NUMINAMATH_CALUDE_no_primes_divisible_by_56_l3775_377519

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0 ∨ m = 1

theorem no_primes_divisible_by_56 :
  ∀ p : ℕ, is_prime p → p % 56 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_primes_divisible_by_56_l3775_377519


namespace NUMINAMATH_CALUDE_kopeck_payment_l3775_377572

theorem kopeck_payment (n : ℕ) (h : n > 7) : ∃ a b : ℕ, n = 3 * a + 5 * b := by
  sorry

end NUMINAMATH_CALUDE_kopeck_payment_l3775_377572


namespace NUMINAMATH_CALUDE_unique_angle_solution_l3775_377595

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 130 := by
sorry

end NUMINAMATH_CALUDE_unique_angle_solution_l3775_377595


namespace NUMINAMATH_CALUDE_fayes_remaining_money_fayes_remaining_money_is_30_l3775_377554

/-- Calculates the remaining money for Faye after receiving money from her mother and making purchases. -/
theorem fayes_remaining_money (initial_money : ℝ) (cupcake_price : ℝ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℝ) (cookie_box_quantity : ℕ) : ℝ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Proves that Faye's remaining money is $30 given the initial conditions. -/
theorem fayes_remaining_money_is_30 : 
  fayes_remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_fayes_remaining_money_fayes_remaining_money_is_30_l3775_377554


namespace NUMINAMATH_CALUDE_intersection_A_B_l3775_377555

def set_A : Set ℝ := {x | x^2 - 11*x - 12 < 0}

def set_B : Set ℝ := {x | ∃ n : ℤ, x = 3*n + 1}

theorem intersection_A_B :
  set_A ∩ set_B = {1, 4, 7, 10} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3775_377555


namespace NUMINAMATH_CALUDE_square_equals_self_l3775_377503

theorem square_equals_self (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_square_equals_self_l3775_377503


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3775_377524

/-- The equation of the tangent line to y = x^3 + 2x at (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + 2*x
  let P : ℝ × ℝ := (1, 3)
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧ 
    (f P.1 = P.2) ∧
    (∀ x, x ≠ P.1 → (f x - P.2) / (x - P.1) ≠ m) ∧
    m*P.1 - P.2 + b = 0 ∧
    m = 5 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3775_377524


namespace NUMINAMATH_CALUDE_acute_angle_range_l3775_377587

theorem acute_angle_range (α : Real) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α < Real.cos α) : 
  α < π / 4 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l3775_377587


namespace NUMINAMATH_CALUDE_minimum_days_to_exceed_500_l3775_377584

def bacteria_count (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem minimum_days_to_exceed_500 :
  ∃ (n : ℕ), n = 6 ∧
  (∀ (k : ℕ), k < n → bacteria_count 4 3 k ≤ 500) ∧
  bacteria_count 4 3 n > 500 :=
sorry

end NUMINAMATH_CALUDE_minimum_days_to_exceed_500_l3775_377584


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l3775_377594

theorem complex_magnitude_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 5 → m = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l3775_377594


namespace NUMINAMATH_CALUDE_prob_second_unqualified_given_first_is_one_fifth_l3775_377517

/-- A box containing disinfectant bottles -/
structure DisinfectantBox where
  total : ℕ
  qualified : ℕ
  unqualified : ℕ

/-- The probability of drawing an unqualified bottle for the second time,
    given that an unqualified bottle was drawn for the first time -/
def prob_second_unqualified_given_first (box : DisinfectantBox) : ℚ :=
  (box.unqualified - 1 : ℚ) / (box.total - 1)

/-- The main theorem -/
theorem prob_second_unqualified_given_first_is_one_fifth
  (box : DisinfectantBox)
  (h_total : box.total = 6)
  (h_qualified : box.qualified = 4)
  (h_unqualified : box.unqualified = 2) :
  prob_second_unqualified_given_first box = 1/5 :=
sorry

end NUMINAMATH_CALUDE_prob_second_unqualified_given_first_is_one_fifth_l3775_377517


namespace NUMINAMATH_CALUDE_unique_solution_system_l3775_377500

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3775_377500


namespace NUMINAMATH_CALUDE_sphere_tangent_plane_distance_l3775_377583

/-- Given three spheres where two smaller spheres touch each other externally and
    each touches a larger sphere internally, with radii as specified,
    the distance from the center of the largest sphere to the tangent plane
    at the touching point of the smaller spheres is R/5. -/
theorem sphere_tangent_plane_distance (R : ℝ) : ℝ := by
  -- Define the radii of the smaller spheres
  let r₁ := R / 2
  let r₂ := R / 3
  
  -- Define the distance from the center of the largest sphere
  -- to the tangent plane at the touching point of the smaller spheres
  let d : ℝ := R / 5
  
  -- The proof would go here
  sorry

#check sphere_tangent_plane_distance

end NUMINAMATH_CALUDE_sphere_tangent_plane_distance_l3775_377583


namespace NUMINAMATH_CALUDE_melanie_dimes_l3775_377563

/-- Calculates the total number of dimes Melanie has after receiving dimes from her parents. -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proves that Melanie has 19 dimes in total. -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3775_377563


namespace NUMINAMATH_CALUDE_fraction_begins_with_0_239_l3775_377534

/-- The infinite decimal a --/
def a : ℝ := 0.1234567891011

/-- The infinite decimal b --/
def b : ℝ := 0.51504948

/-- Theorem stating that the fraction a/b begins with 0.239 --/
theorem fraction_begins_with_0_239 (h1 : 0.515 < b) (h2 : b < 0.516) :
  0.239 * b ≤ a ∧ a < 0.24 * b := by sorry

end NUMINAMATH_CALUDE_fraction_begins_with_0_239_l3775_377534


namespace NUMINAMATH_CALUDE_center_of_mass_position_l3775_377505

/-- A system of disks with specific properties -/
structure DiskSystem where
  -- The ratio of radii of two adjacent disks
  ratio : ℝ
  -- The radius of the largest disk
  largest_radius : ℝ
  -- Assertion that the ratio is 1/2
  ratio_is_half : ratio = 1/2
  -- Assertion that the largest radius is 2 meters
  largest_radius_is_two : largest_radius = 2

/-- The center of mass of the disk system -/
noncomputable def center_of_mass (ds : DiskSystem) : ℝ := sorry

/-- Theorem stating that the center of mass is at 6/7 meters from the largest disk's center -/
theorem center_of_mass_position (ds : DiskSystem) : 
  center_of_mass ds = 6/7 := by sorry

end NUMINAMATH_CALUDE_center_of_mass_position_l3775_377505


namespace NUMINAMATH_CALUDE_delivery_driver_boxes_l3775_377522

/-- Calculates the total number of boxes a delivery driver has -/
def total_boxes (num_stops : ℕ) (boxes_per_stop : ℕ) : ℕ :=
  num_stops * boxes_per_stop

/-- Proves that a delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total -/
theorem delivery_driver_boxes :
  total_boxes 3 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_delivery_driver_boxes_l3775_377522


namespace NUMINAMATH_CALUDE_broken_line_length_formula_l3775_377541

/-- Given an acute angle α and a point A₁ on one of its sides, we repeatedly drop perpendiculars
    to form an infinite broken line. This function represents the length of that line. -/
noncomputable def broken_line_length (α : Real) (m : Real) : Real :=
  m / (1 - Real.cos α)

/-- Theorem stating that the length of the infinite broken line formed by repeatedly dropping
    perpendiculars in an acute angle is equal to m / (1 - cos(α)), where m is the length of
    the first perpendicular and α is the magnitude of the angle. -/
theorem broken_line_length_formula (α : Real) (m : Real) 
    (h_acute : 0 < α ∧ α < Real.pi / 2) 
    (h_positive : m > 0) : 
  broken_line_length α m = m / (1 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_formula_l3775_377541


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3775_377599

-- Problem 1
theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (-2 * a^2)^2 * (-b^2) / (4 * a^3 * b^2) = -a := by sorry

-- Problem 2
theorem problem_2 : 2023^2 - 2021 * 2025 = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3775_377599


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l3775_377592

theorem arctan_sum_equals_pi_over_four (n : ℕ+) :
  Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/5) + Real.arctan (1/n) = π/4 →
  n = 47 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l3775_377592


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3775_377547

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3775_377547


namespace NUMINAMATH_CALUDE_no_x_squared_term_l3775_377514

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (x + 2) * (x^2 - 5*a*x + 1) = x^3 + (-9*a)*x + 2) → a = 2/5 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l3775_377514


namespace NUMINAMATH_CALUDE_volume_equals_target_l3775_377542

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of points inside or within one unit of a parallelepiped -/
def volume_with_buffer (p : Parallelepiped) : ℝ := sorry

/-- The specific parallelepiped in the problem -/
def problem_parallelepiped : Parallelepiped :=
  { length := 2,
    width := 3,
    height := 4 }

theorem volume_equals_target : 
  volume_with_buffer problem_parallelepiped = (456 + 31 * Real.pi) / 6 := by sorry

end NUMINAMATH_CALUDE_volume_equals_target_l3775_377542


namespace NUMINAMATH_CALUDE_amount_division_l3775_377545

/-- Given an amount divided into 3 parts proportional to 1/2 : 2/3 : 3/4, 
    with the first part being 204, prove the total amount is 782. -/
theorem amount_division (amount : ℕ) 
  (h1 : amount > 0)
  (h2 : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a * 3 * 4 = 1 * 2 * 4 ∧ 
    b * 2 * 4 = 2 * 3 * 4 ∧ 
    c * 2 * 3 = 3 * 2 * 4 ∧
    a + b + c = amount)
  (h3 : a = 204) : 
  amount = 782 := by
  sorry

end NUMINAMATH_CALUDE_amount_division_l3775_377545


namespace NUMINAMATH_CALUDE_sequence_sum_l3775_377566

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = 2 * a n

def arithmetic_sequence (b : ℕ → ℚ) := ∃ d, ∀ n, b (n + 1) = b n + d

theorem sequence_sum (a : ℕ → ℚ) (b : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 3 * a 4 = 27 / 64 →
  arithmetic_sequence b →
  b 7 = a 5 →
  b 3 + b 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l3775_377566


namespace NUMINAMATH_CALUDE_fraction_sum_between_extremes_l3775_377580

theorem fraction_sum_between_extremes 
  (a b c d n p x y : ℚ) 
  (h_pos : b > 0 ∧ d > 0 ∧ p > 0 ∧ y > 0)
  (h_order : a/b > c/d ∧ c/d > n/p ∧ n/p > x/y) : 
  x/y < (a + c + n + x) / (b + d + p + y) ∧ 
  (a + c + n + x) / (b + d + p + y) < a/b := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_between_extremes_l3775_377580


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l3775_377510

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x^3 + x + 1) :
  ∀ x < 0, f x = x^3 + x - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l3775_377510


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3775_377576

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  volume = (surface_area / 6) ^ (3/2) →
  volume = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3775_377576


namespace NUMINAMATH_CALUDE_decreasing_interval_of_f_decreasing_interval_is_open_interval_l3775_377513

-- Define the function
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (f' x < 0) ↔ (-1 < x ∧ x < 1) :=
sorry

-- Main theorem
theorem decreasing_interval_is_open_interval :
  {x : ℝ | ∀ y : ℝ, -1 < y ∧ y < x → f y > f x} = Set.Ioo (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_f_decreasing_interval_is_open_interval_l3775_377513


namespace NUMINAMATH_CALUDE_mold_diameter_l3775_377582

/-- The diameter of a circular mold with radius 2 inches is 4 inches. -/
theorem mold_diameter (r : ℝ) (h : r = 2) : 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_mold_diameter_l3775_377582


namespace NUMINAMATH_CALUDE_water_amount_l3775_377537

/-- Represents the recipe ratios and quantities -/
structure Recipe where
  water : ℝ
  sugar : ℝ
  cranberry : ℝ
  water_sugar_ratio : water = 5 * sugar
  sugar_cranberry_ratio : sugar = 3 * cranberry
  cranberry_amount : cranberry = 4

/-- Proves that the amount of water needed is 60 cups -/
theorem water_amount (r : Recipe) : r.water = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_amount_l3775_377537


namespace NUMINAMATH_CALUDE_initial_breads_l3775_377512

/-- The number of thieves -/
def num_thieves : ℕ := 8

/-- The number of breads remaining after all thieves -/
def remaining_breads : ℕ := 5

/-- The function representing how many breads remain after each thief -/
def breads_after_thief (n : ℕ) (b : ℚ) : ℚ :=
  if n = 0 then b else (1/2) * (breads_after_thief (n-1) b) - (1/2)

/-- The theorem stating the initial number of breads -/
theorem initial_breads :
  ∃ (b : ℚ), breads_after_thief num_thieves b = remaining_breads ∧ b = 1535 := by
  sorry

end NUMINAMATH_CALUDE_initial_breads_l3775_377512


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3775_377506

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), 
    (∀ θ ∈ s, 0 < θ ∧ θ ≤ π ∧ 4 - 2 * Real.sin θ + 3 * Real.cos (2 * θ) = 0) ∧
    s.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3775_377506


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l3775_377529

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 5 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 2) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 9) = 10 / 4) -- New ratio after removing 9 pounds of clothes
  : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l3775_377529


namespace NUMINAMATH_CALUDE_customers_who_left_l3775_377586

theorem customers_who_left (initial_customers : ℕ) (new_customers : ℕ) (final_customers : ℕ) :
  initial_customers = 19 →
  new_customers = 36 →
  final_customers = 41 →
  initial_customers - (initial_customers - new_customers - final_customers) + new_customers = final_customers :=
by
  sorry

end NUMINAMATH_CALUDE_customers_who_left_l3775_377586


namespace NUMINAMATH_CALUDE_two_true_propositions_l3775_377560

theorem two_true_propositions : 
  let original := ∀ a : ℝ, a > 2 → a > 1
  let converse := ∀ a : ℝ, a > 1 → a > 2
  let inverse := ∀ a : ℝ, a ≤ 2 → a ≤ 1
  let contrapositive := ∀ a : ℝ, a ≤ 1 → a ≤ 2
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l3775_377560


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3775_377521

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (1 + 2 * Complex.I) / (a + b * Complex.I) = 1 + Complex.I →
  a = 3/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3775_377521


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l3775_377546

/-- Proves that the minimum bailing rate to reach shore without sinking is 10.75 gallons per minute --/
theorem minimum_bailing_rate 
  (distance : ℝ) 
  (intake_rate : ℝ) 
  (max_water : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance = 2) 
  (h2 : intake_rate = 12) 
  (h3 : max_water = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 10.75 ∧ 
    bailing_rate < intake_rate ∧
    (distance / rowing_speed) * 60 * (intake_rate - bailing_rate) ≤ max_water :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l3775_377546


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_triangle_perimeter_l3775_377577

theorem smallest_whole_number_above_triangle_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 8 > 25 →
  s + 25 > 8 →
  8 + 25 > s →
  (∃ n : ℕ, n = 67 ∧ ∀ m : ℕ, (m : ℝ) > 8 + 25 + s → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_triangle_perimeter_l3775_377577


namespace NUMINAMATH_CALUDE_kth_level_associated_point_coordinates_l3775_377531

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the k-th level associated point -/
def kth_level_associated_point (A : Point) (k : ℝ) : Point :=
  { x := k * A.x + A.y,
    y := A.x + k * A.y }

/-- Theorem: The k-th level associated point B of A(x,y) has coordinates (kx+y, x+ky) -/
theorem kth_level_associated_point_coordinates (A : Point) (k : ℝ) (h : k ≠ 0) :
  let B := kth_level_associated_point A k
  B.x = k * A.x + A.y ∧ B.y = A.x + k * A.y :=
by sorry

end NUMINAMATH_CALUDE_kth_level_associated_point_coordinates_l3775_377531


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3775_377585

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 1

-- State the theorem
theorem fixed_point_on_line 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a (-1) = 2) 
  (b : ℝ) 
  (h4 : b * (-1) + 2 + 1 = 0) :
  b = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3775_377585


namespace NUMINAMATH_CALUDE_prism_volume_proof_l3775_377558

/-- The volume of a right rectangular prism with face areas 28, 45, and 63 square centimeters -/
def prism_volume : ℝ := 282

theorem prism_volume_proof (x y z : ℝ) 
  (face1 : x * y = 28)
  (face2 : x * z = 45)
  (face3 : y * z = 63) :
  x * y * z = prism_volume := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_proof_l3775_377558


namespace NUMINAMATH_CALUDE_males_in_band_not_orchestra_l3775_377559

/-- Represents the number of students in a group -/
structure GroupCount where
  female : ℕ
  male : ℕ

/-- Represents the counts for band, orchestra, and choir -/
structure MusicGroups where
  band : GroupCount
  orchestra : GroupCount
  choir : GroupCount
  all_three : GroupCount
  total : ℕ

def music_groups : MusicGroups := {
  band := { female := 120, male := 90 },
  orchestra := { female := 90, male := 120 },
  choir := { female := 50, male := 40 },
  all_three := { female := 30, male := 20 },
  total := 250
}

theorem males_in_band_not_orchestra (g : MusicGroups) (h : g = music_groups) :
  g.band.male - (g.band.male + g.orchestra.male + g.choir.male - g.total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_males_in_band_not_orchestra_l3775_377559


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3775_377508

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l3775_377508


namespace NUMINAMATH_CALUDE_trapezoid_reconstruction_l3775_377591

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- Checks if two line segments are parallel -/
def parallel (p q r s : Point) : Prop :=
  (q.y - p.y) * (s.x - r.x) = (s.y - r.y) * (q.x - p.x)

/-- Checks if a point divides two line segments proportionally -/
def divides_proportionally (o p q r s : Point) : Prop :=
  (o.x - p.x) * (s.y - q.y) = (o.y - p.y) * (s.x - q.x)

/-- Theorem: Given three points A, B, C, and a point O, 
    there exists a point D such that ABCD forms a trapezoid 
    with O as the intersection of its diagonals -/
theorem trapezoid_reconstruction 
  (A B C O : Point) 
  (h1 : collinear A O C) 
  (h2 : ¬ collinear A B C) : 
  ∃ D : Point, 
    parallel A B C D ∧ 
    collinear B O D ∧
    divides_proportionally O A C B D :=
sorry

end NUMINAMATH_CALUDE_trapezoid_reconstruction_l3775_377591


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3775_377567

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Represents the configuration of a cylinder inscribed in a cone -/
structure InscribedCylinder where
  cone : Cone
  cylinder : Cylinder
  height_radius_ratio : ℝ
  axes_coincide : Bool

/-- Theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (ic : InscribedCylinder) 
  (h1 : ic.cone.diameter = 12) 
  (h2 : ic.cone.altitude = 15) 
  (h3 : ic.height_radius_ratio = 3) 
  (h4 : ic.axes_coincide = true) : 
  ic.cylinder.radius = 30 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l3775_377567


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3775_377564

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) (hxa : x^2 ≠ a) :
  ((x^2 + a) / x^2 + (x^2 - a) / x^2) / 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l3775_377564


namespace NUMINAMATH_CALUDE_sum_of_remaining_segments_l3775_377548

/-- Represents a rectangular figure with some interior segments -/
structure RectFigure where
  left : ℝ
  right : ℝ
  bottomLeft : ℝ
  topLeft : ℝ
  topRight : ℝ

/-- Calculates the sum of remaining segments after removing four sides -/
def remainingSum (f : RectFigure) : ℝ :=
  f.left + f.right + (f.bottomLeft + f.topLeft + f.topRight) + f.topRight

/-- Theorem stating that for the given measurements, the sum of remaining segments is 23 -/
theorem sum_of_remaining_segments :
  let f : RectFigure := {
    left := 10,
    right := 7,
    bottomLeft := 3,
    topLeft := 1,
    topRight := 1
  }
  remainingSum f = 23 := by sorry

end NUMINAMATH_CALUDE_sum_of_remaining_segments_l3775_377548


namespace NUMINAMATH_CALUDE_initial_number_proof_l3775_377507

theorem initial_number_proof : ∃ x : ℤ, x - 10 * 2 * 5 = 10011 ∧ x = 10111 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l3775_377507


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3775_377590

/-- The probability of selecting 2 red, 1 blue, and 1 green marble when choosing 4 marbles
    without replacement from a bag containing 3 red, 3 blue, and 3 green marbles. -/
theorem marble_selection_probability :
  let total_marbles : ℕ := 9
  let red_marbles : ℕ := 3
  let blue_marbles : ℕ := 3
  let green_marbles : ℕ := 3
  let selected_marbles : ℕ := 4
  let favorable_outcomes : ℕ := Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1
  let total_outcomes : ℕ := Nat.choose total_marbles selected_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 14 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3775_377590


namespace NUMINAMATH_CALUDE_chord_convex_quadrilateral_probability_l3775_377570

/-- Given six points on a circle, the probability that four randomly chosen chords
    form a convex quadrilateral is 1/91. -/
theorem chord_convex_quadrilateral_probability (n : ℕ) (h : n = 6) :
  (Nat.choose n 4 : ℚ) / (Nat.choose (Nat.choose n 2) 4) = 1 / 91 :=
sorry

end NUMINAMATH_CALUDE_chord_convex_quadrilateral_probability_l3775_377570


namespace NUMINAMATH_CALUDE_carnival_earnings_value_l3775_377549

/-- The total earnings from two ring toss games at a carnival -/
def carnival_earnings : ℕ :=
  let game1_period1 := 88
  let game1_rate1 := 761
  let game1_period2 := 20
  let game1_rate2 := 487
  let game2_period1 := 66
  let game2_rate1 := 569
  let game2_period2 := 15
  let game2_rate2 := 932
  let game1_earnings := game1_period1 * game1_rate1 + game1_period2 * game1_rate2
  let game2_earnings := game2_period1 * game2_rate1 + game2_period2 * game2_rate2
  game1_earnings + game2_earnings

theorem carnival_earnings_value : carnival_earnings = 128242 := by
  sorry

end NUMINAMATH_CALUDE_carnival_earnings_value_l3775_377549


namespace NUMINAMATH_CALUDE_f_properties_l3775_377589

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi) * Real.cos (Real.pi - x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/2) → f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/2) ∧ f x = -Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3775_377589


namespace NUMINAMATH_CALUDE_max_teams_is_six_l3775_377551

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := n.choose 2 * games_between_teams

/-- The theorem stating that 6 is the maximum number of teams that can participate -/
theorem max_teams_is_six :
  ∀ n : ℕ, n > 6 → total_games n > max_games ∧
  total_games 6 ≤ max_games :=
sorry

end NUMINAMATH_CALUDE_max_teams_is_six_l3775_377551


namespace NUMINAMATH_CALUDE_rectangle_coverage_l3775_377578

/-- A shape composed of 6 unit squares -/
structure Shape :=
  (area : ℕ)
  (h_area : area = 6)

/-- A rectangle with dimensions m × n -/
structure Rectangle (m n : ℕ) :=
  (width : ℕ)
  (height : ℕ)
  (h_width : width = m)
  (h_height : height = n)

/-- Predicate for a rectangle that can be covered by shapes -/
def is_coverable (m n : ℕ) : Prop :=
  (3 ∣ m ∧ 4 ∣ n) ∨ (3 ∣ n ∧ 4 ∣ m) ∨ (12 ∣ m ∧ 12 ∣ n)

theorem rectangle_coverage (m n : ℕ) (hm : m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 5) (hn : n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 5) :
  ∃ (s : Shape), ∃ (r : Rectangle m n), is_coverable m n ↔ 
    (∃ (arrangement : ℕ → ℕ → Shape), 
      (∀ i j, i < m ∧ j < n → (arrangement i j).area = 6) ∧
      (∀ i j, i < m ∧ j < n → ∃ k l, k < m ∧ l < n ∧ arrangement i j = arrangement k l) ∧
      (∀ i j k l, i < m ∧ j < n ∧ k < m ∧ l < n → 
        (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_coverage_l3775_377578


namespace NUMINAMATH_CALUDE_polygon_area_bound_l3775_377561

/-- A polygon with n vertices -/
structure Polygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ

/-- The area of a polygon -/
def area (P : Polygon) : ℝ := sorry

/-- The length of a line segment between two points -/
def distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Area of a polygon with constrained sides and diagonals -/
theorem polygon_area_bound (P : Polygon) 
  (h1 : ∀ (i j : Fin P.n), distance (P.vertices i) (P.vertices j) ≤ 1) : 
  area P < Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_polygon_area_bound_l3775_377561


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l3775_377550

theorem factor_implies_m_value (m : ℤ) : 
  (∃ a : ℤ, ∀ x : ℤ, x^2 - m*x - 15 = (x + 3) * (x - a)) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l3775_377550


namespace NUMINAMATH_CALUDE_remaining_payment_proof_l3775_377540

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  (deposit_amount / deposit_percentage) - deposit_amount

/-- Proves that the remaining amount to be paid is 990, given a 10% deposit of 110 -/
theorem remaining_payment_proof : 
  remaining_payment (1/10) 110 = 990 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_proof_l3775_377540


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l3775_377530

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure PolygonConfiguration where
  P1 : ConvexPolygon
  P2 : ConvexPolygon
  no_common_segment : Bool
  h : P1.sides ≤ P2.sides

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.P1.sides * config.P2.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  config.P1.convex ∧ config.P2.convex ∧ config.no_common_segment →
  max_intersections config = config.P1.sides * config.P2.sides :=
by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l3775_377530


namespace NUMINAMATH_CALUDE_vincent_stickers_l3775_377579

theorem vincent_stickers (yesterday : ℕ) (extra_today : ℕ) : 
  yesterday = 15 → extra_today = 10 → yesterday + (yesterday + extra_today) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vincent_stickers_l3775_377579


namespace NUMINAMATH_CALUDE_forty_percent_relation_l3775_377581

theorem forty_percent_relation (x : ℝ) (v : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * x = v → (40/100 : ℝ) * x = 12 * v := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_relation_l3775_377581


namespace NUMINAMATH_CALUDE_watch_selling_price_l3775_377533

/-- Calculates the selling price of an item given its cost price and profit percentage. -/
def selling_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (cost_price * profit_percentage) / 100

/-- Proves that for a watch with a cost price of 90 rupees, 
    if the profit percentage is equal to the cost price, 
    then the selling price is 180 rupees. -/
theorem watch_selling_price : 
  let cost_price : ℕ := 90
  let profit_percentage : ℕ := 100
  selling_price cost_price profit_percentage = 180 := by
sorry


end NUMINAMATH_CALUDE_watch_selling_price_l3775_377533


namespace NUMINAMATH_CALUDE_three_reflection_theorem_l3775_377588

/-- A circular billiard table -/
structure BilliardTable where
  R : ℝ
  R_pos : R > 0

/-- A point on the billiard table -/
structure Point (bt : BilliardTable) where
  x : ℝ
  y : ℝ
  on_table : x^2 + y^2 ≤ bt.R^2

/-- Predicate for a valid starting point A -/
def valid_start_point (bt : BilliardTable) (A : Point bt) : Prop :=
  A.x^2 + A.y^2 > (bt.R/3)^2 ∧ A.x^2 + A.y^2 < bt.R^2

/-- Predicate for a valid reflection path -/
def valid_reflection_path (bt : BilliardTable) (A : Point bt) : Prop :=
  ∃ (B C : Point bt),
    B ≠ A ∧ C ≠ A ∧ B ≠ C ∧
    (A.x^2 + A.y^2 = B.x^2 + B.y^2) ∧
    (B.x^2 + B.y^2 = C.x^2 + C.y^2) ∧
    (C.x^2 + C.y^2 = A.x^2 + A.y^2)

theorem three_reflection_theorem (bt : BilliardTable) (A : Point bt) :
  valid_start_point bt A ↔ valid_reflection_path bt A :=
sorry

end NUMINAMATH_CALUDE_three_reflection_theorem_l3775_377588


namespace NUMINAMATH_CALUDE_inverse_sum_product_l3775_377556

theorem inverse_sum_product (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hsum : 3*x + 4*y ≠ 0) :
  (3*x + 4*y)⁻¹ * ((3*x)⁻¹ + (4*y)⁻¹) = (12*x*y)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_product_l3775_377556


namespace NUMINAMATH_CALUDE_period_of_symmetric_function_l3775_377557

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- A real number p is a period of a function f if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem period_of_symmetric_function (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : SymmetricAbout (fun x ↦ f (2 * x)) (a / 2)) 
    (h2 : SymmetricAbout (fun x ↦ f (2 * x)) (b / 2)) 
    (h3 : b > a) : 
    IsPeriod f (4 * (b - a)) := by
  sorry

end NUMINAMATH_CALUDE_period_of_symmetric_function_l3775_377557


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l3775_377562

theorem smallest_yellow_marbles (n : ℕ) (h1 : n > 0) 
  (h2 : n % 2 = 0) (h3 : n % 3 = 0) (h4 : 4 ≤ n) : 
  ∃ (y : ℕ), y = n - (n / 2 + n / 3 + 4) ∧ 
  (∀ (m : ℕ), m > 0 → m % 2 = 0 → m % 3 = 0 → 4 ≤ m → 
    m - (m / 2 + m / 3 + 4) ≥ 0 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l3775_377562


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l3775_377565

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 8BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 5/8. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line segment
  B - A = 4 * (D - A) →    -- AB = 4AD
  B - A = 8 * (C - B) →    -- AB = 8BC
  (D - C) / (B - A) = 5/8 := by sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l3775_377565


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3775_377532

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def discarded_numbers : List ℚ := [50, 60, 70]

theorem arithmetic_mean_after_removal :
  let original_sum : ℚ := original_mean * original_set_size
  let remaining_sum : ℚ := original_sum - (discarded_numbers.sum)
  let remaining_set_size : ℕ := original_set_size - discarded_numbers.length
  (remaining_sum / remaining_set_size : ℚ) = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3775_377532
