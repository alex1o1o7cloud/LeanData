import Mathlib

namespace NUMINAMATH_CALUDE_function_problem_l720_72038

/-- Given a function f(x) = x / (ax + b) where a ≠ 0, f(4) = 4/3, and f(x) = x has a unique solution,
    prove that f(x) = 2x / (x + 2) and f[f(-3)] = 3/2 -/
theorem function_problem (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ x / (a * x + b)
  (f 4 = 4 / 3) →
  (∃! x, f x = x) →
  (∀ x, f x = 2 * x / (x + 2)) ∧
  (f (f (-3)) = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_problem_l720_72038


namespace NUMINAMATH_CALUDE_fraction_product_equals_twelve_l720_72060

theorem fraction_product_equals_twelve :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_twelve_l720_72060


namespace NUMINAMATH_CALUDE_faye_pencil_count_l720_72058

/-- The number of rows of pencils and crayons --/
def num_rows : ℕ := 30

/-- The number of pencils in each row --/
def pencils_per_row : ℕ := 24

/-- The total number of pencils --/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencil_count : total_pencils = 720 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_count_l720_72058


namespace NUMINAMATH_CALUDE_new_lamp_height_is_correct_l720_72010

/-- The height of the old lamp in feet -/
def old_lamp_height : ℝ := 1

/-- The difference in height between the new and old lamp in feet -/
def height_difference : ℝ := 1.3333333333333333

/-- The height of the new lamp in feet -/
def new_lamp_height : ℝ := old_lamp_height + height_difference

theorem new_lamp_height_is_correct : new_lamp_height = 2.3333333333333333 := by
  sorry

end NUMINAMATH_CALUDE_new_lamp_height_is_correct_l720_72010


namespace NUMINAMATH_CALUDE_class_A_student_count_l720_72032

/-- The number of students who like social studies -/
def social_studies_count : ℕ := 25

/-- The number of students who like music -/
def music_count : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_count : ℕ := 27

/-- The total number of students in class (A) -/
def total_students : ℕ := social_studies_count + music_count - both_count

theorem class_A_student_count :
  total_students = 30 :=
sorry

end NUMINAMATH_CALUDE_class_A_student_count_l720_72032


namespace NUMINAMATH_CALUDE_quadratic_minimum_l720_72039

theorem quadratic_minimum (f : ℝ → ℝ) (h : f = λ x => (x - 1)^2 + 3) : 
  ∀ x, f x ≥ 3 ∧ ∃ x₀, f x₀ = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l720_72039


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l720_72077

/-- The height of a tree after n years, given its initial height and growth factor --/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * growth_factor ^ years

/-- Theorem stating that a tree with initial height h quadrupling every year for 4 years
    reaches 256 feet if and only if h = 1 foot --/
theorem tree_growth_theorem (h : ℝ) : 
  tree_height h 4 4 = 256 ↔ h = 1 := by
  sorry

#check tree_growth_theorem

end NUMINAMATH_CALUDE_tree_growth_theorem_l720_72077


namespace NUMINAMATH_CALUDE_unique_triple_gcd_sum_l720_72037

theorem unique_triple_gcd_sum (m n l : ℕ) : 
  (m + n = (Nat.gcd m n)^2) ∧ 
  (m + l = (Nat.gcd m l)^2) ∧ 
  (n + l = (Nat.gcd n l)^2) →
  m = 2 ∧ n = 2 ∧ l = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_gcd_sum_l720_72037


namespace NUMINAMATH_CALUDE_susan_peaches_in_knapsack_l720_72087

/-- The number of peaches Susan bought -/
def total_peaches : ℕ := 5 * 12

/-- The number of cloth bags Susan has -/
def num_cloth_bags : ℕ := 2

/-- Represents the relationship between peaches in cloth bags and knapsack -/
def knapsack_ratio : ℚ := 1 / 2

/-- The number of peaches in the knapsack -/
def peaches_in_knapsack : ℕ := 12

theorem susan_peaches_in_knapsack :
  ∃ (x : ℕ), 
    (x : ℚ) * num_cloth_bags + (x : ℚ) * knapsack_ratio = total_peaches ∧
    peaches_in_knapsack = (x : ℚ) * knapsack_ratio := by
  sorry

end NUMINAMATH_CALUDE_susan_peaches_in_knapsack_l720_72087


namespace NUMINAMATH_CALUDE_prime_count_in_range_l720_72059

theorem prime_count_in_range (n : ℕ) (h : n > 2) :
  (n > 3 → ∀ p, Nat.Prime p → ¬((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) ∧
  (n = 3 → ∃! p, Nat.Prime p ∧ ((n - 1).factorial + 2 < p ∧ p < (n - 1).factorial + n)) :=
sorry

end NUMINAMATH_CALUDE_prime_count_in_range_l720_72059


namespace NUMINAMATH_CALUDE_intersection_A_B_l720_72052

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l720_72052


namespace NUMINAMATH_CALUDE_one_fifth_of_seven_x_plus_three_l720_72003

theorem one_fifth_of_seven_x_plus_three (x : ℝ) : 
  (1 / 5) * (7 * x + 3) = (7 / 5) * x + 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_seven_x_plus_three_l720_72003


namespace NUMINAMATH_CALUDE_min_amount_lost_l720_72091

/-- Represents the denomination of a bill -/
inductive Bill
  | ten  : Bill
  | fifty : Bill

/-- Calculates the value of a bill -/
def billValue (b : Bill) : Nat :=
  match b with
  | Bill.ten  => 10
  | Bill.fifty => 50

/-- Represents the cash transaction and usage -/
structure CashTransaction where
  totalCashed : Nat
  billsUsed : Nat
  tenBills : Nat
  fiftyBills : Nat

/-- Conditions of the problem -/
def transactionConditions (t : CashTransaction) : Prop :=
  t.totalCashed = 1270 ∧
  t.billsUsed = 15 ∧
  (t.tenBills = t.fiftyBills + 1 ∨ t.tenBills = t.fiftyBills - 1) ∧
  t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty ≤ t.totalCashed

/-- Theorem stating the minimum amount lost -/
theorem min_amount_lost (t : CashTransaction) 
  (h : transactionConditions t) : 
  t.totalCashed - (t.tenBills * billValue Bill.ten + t.fiftyBills * billValue Bill.fifty) = 800 := by
  sorry

end NUMINAMATH_CALUDE_min_amount_lost_l720_72091


namespace NUMINAMATH_CALUDE_multiply_subtract_equation_l720_72072

theorem multiply_subtract_equation : ∃ x : ℝ, 12 * x - 3 = (12 - 7) * 9 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_equation_l720_72072


namespace NUMINAMATH_CALUDE_abs_ratio_sum_l720_72015

theorem abs_ratio_sum (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a| / a + |b| / b : ℚ) = 2 ∨ (|a| / a + |b| / b : ℚ) = -2 ∨ (|a| / a + |b| / b : ℚ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_abs_ratio_sum_l720_72015


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l720_72094

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_f_at_one : 
  deriv f 1 = 2 + Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l720_72094


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_120_by_75_percent_l720_72033

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_120_by_75_percent :
  120 * (1 + 75 / 100) = 210 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_120_by_75_percent_l720_72033


namespace NUMINAMATH_CALUDE_rahuls_share_l720_72063

/-- Calculates the share of payment for a worker in a joint work scenario -/
def calculate_share (days_worker1 days_worker2 total_payment : ℚ) : ℚ :=
  let worker1_rate := 1 / days_worker1
  let worker2_rate := 1 / days_worker2
  let combined_rate := worker1_rate + worker2_rate
  let share_ratio := worker1_rate / combined_rate
  share_ratio * total_payment

/-- Theorem stating that Rahul's share of the payment is $68 -/
theorem rahuls_share :
  calculate_share 3 2 170 = 68 := by
  sorry

#eval calculate_share 3 2 170

end NUMINAMATH_CALUDE_rahuls_share_l720_72063


namespace NUMINAMATH_CALUDE_max_value_of_f_l720_72086

theorem max_value_of_f (α : ℝ) :
  ∃ M : ℝ, M = (Real.sqrt 2 + 1) / 2 ∧
  (∀ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) ≤ M) ∧
  (∃ x : ℝ, 1 - Real.sin (x + α)^2 + Real.cos (x + α) * Real.sin (x + α) = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l720_72086


namespace NUMINAMATH_CALUDE_Q_sufficient_not_necessary_l720_72093

open Real

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define proposition P
def P (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → |(f x₁ - f x₂) / (x₁ - x₂)| < 2018

-- Define proposition Q
def Q (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, |deriv f x| < 2018

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (Q f → P f) ∧ ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ P g ∧ ¬(Q g) := by
  sorry

end NUMINAMATH_CALUDE_Q_sufficient_not_necessary_l720_72093


namespace NUMINAMATH_CALUDE_standard_colony_conditions_l720_72065

/-- Represents the type of culture medium -/
inductive CultureMedium
| Liquid
| Solid

/-- Represents a bacterial colony -/
structure BacterialColony where
  initialBacteria : ℕ
  medium : CultureMedium

/-- Defines what constitutes a standard bacterial colony -/
def isStandardColony (colony : BacterialColony) : Prop :=
  colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid

/-- Theorem stating the conditions for a standard bacterial colony -/
theorem standard_colony_conditions :
  ∀ (colony : BacterialColony),
    isStandardColony colony ↔
      colony.initialBacteria = 1 ∧ colony.medium = CultureMedium.Solid :=
by
  sorry

end NUMINAMATH_CALUDE_standard_colony_conditions_l720_72065


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l720_72083

/-- A biased coin with probability of heads and tails, and corresponding gains/losses -/
structure BiasedCoin where
  prob_heads : ℝ
  prob_tails : ℝ
  gain_heads : ℝ
  loss_tails : ℝ

/-- The expected value of a coin flip -/
def expected_value (c : BiasedCoin) : ℝ :=
  c.prob_heads * c.gain_heads + c.prob_tails * (-c.loss_tails)

/-- Theorem: The expected value of the specific biased coin is 0 -/
theorem biased_coin_expected_value :
  let c : BiasedCoin := {
    prob_heads := 2/3,
    prob_tails := 1/3,
    gain_heads := 5,
    loss_tails := 10
  }
  expected_value c = 0 := by sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l720_72083


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l720_72002

theorem number_of_students_in_class : 
  ∀ (N : ℕ) (avg_age_all avg_age_5 avg_age_9 last_student_age : ℚ),
    avg_age_all = 15 →
    avg_age_5 = 13 →
    avg_age_9 = 16 →
    last_student_age = 16 →
    N * avg_age_all = 5 * avg_age_5 + 9 * avg_age_9 + last_student_age →
    N = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l720_72002


namespace NUMINAMATH_CALUDE_multiply_cube_by_negative_l720_72070

/-- For any real number y, 2y³ * (-y) = -2y⁴ -/
theorem multiply_cube_by_negative (y : ℝ) : 2 * y^3 * (-y) = -2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_cube_by_negative_l720_72070


namespace NUMINAMATH_CALUDE_polynomial_value_at_one_l720_72080

theorem polynomial_value_at_one (a b c : ℝ) : 
  (-a - b - c + 1 = 6) → (a + b + c + 1 = -4) := by sorry

end NUMINAMATH_CALUDE_polynomial_value_at_one_l720_72080


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l720_72036

/-- The length of the generatrix of a cone with lateral area 6π and base radius 2 is 3. -/
theorem cone_generatrix_length :
  ∀ (l : ℝ), 
    (l > 0) →
    (2 * Real.pi * l = 6 * Real.pi) →
    l = 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l720_72036


namespace NUMINAMATH_CALUDE_circle_integer_points_l720_72045

theorem circle_integer_points 
  (center : ℝ × ℝ) 
  (h_center : center = (Real.sqrt 2, Real.sqrt 3)) :
  ∀ (A B : ℤ × ℤ), A ≠ B →
  ¬(∃ (r : ℝ), r > 0 ∧ 
    ((A.1 - center.1)^2 + (A.2 - center.2)^2 = r^2) ∧
    ((B.1 - center.1)^2 + (B.2 - center.2)^2 = r^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_integer_points_l720_72045


namespace NUMINAMATH_CALUDE_problem_statement_l720_72013

theorem problem_statement :
  (∃ (x₁ x₂ x₃ x₄ x₅ : ℚ), x₁ < 0 ∧ x₂ < 0 ∧ x₃ < 0 ∧ x₄ * x₅ > 0 ∧ x₁ * x₂ * x₃ * x₄ * x₅ < 0) ∧
  (∀ m : ℝ, abs m + m = 0 → m ≤ 0) ∧
  (∃ a b : ℝ, 1 / a < 1 / b ∧ (a < b ∨ a > b)) ∧
  (∀ a : ℝ, 5 - abs (a - 5) ≤ 5) ∧ (∃ a : ℝ, 5 - abs (a - 5) = 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l720_72013


namespace NUMINAMATH_CALUDE_unique_sum_of_squares_125_l720_72082

/-- A function that returns the number of ways to write a given number as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
def countWaysToSum (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there is exactly one way to write 125 as the sum of three positive perfect squares,
    where the order doesn't matter and at least one square appears twice. -/
theorem unique_sum_of_squares_125 : countWaysToSum 125 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_sum_of_squares_125_l720_72082


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l720_72088

theorem other_root_of_quadratic (m : ℝ) : 
  ((-4 : ℝ)^2 + m * (-4) - 20 = 0) → (5^2 + m * 5 - 20 = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l720_72088


namespace NUMINAMATH_CALUDE_prob_rain_theorem_l720_72048

/-- The probability of rain on at least one day during a three-day period -/
def prob_rain_at_least_once (p1 p2 p3 : ℝ) : ℝ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

/-- Theorem stating the probability of rain on at least one day is 86% -/
theorem prob_rain_theorem :
  prob_rain_at_least_once 0.3 0.6 0.5 = 0.86 := by
  sorry

#eval prob_rain_at_least_once 0.3 0.6 0.5

end NUMINAMATH_CALUDE_prob_rain_theorem_l720_72048


namespace NUMINAMATH_CALUDE_remainder_4672_div_34_l720_72067

theorem remainder_4672_div_34 : 4672 % 34 = 14 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4672_div_34_l720_72067


namespace NUMINAMATH_CALUDE_sonya_falls_count_l720_72040

/-- The number of times Sonya fell while ice skating --/
def sonya_falls (steven_falls stephanie_falls : ℕ) : ℕ :=
  (stephanie_falls / 2) - 2

/-- Proof that Sonya fell 6 times given the conditions --/
theorem sonya_falls_count :
  ∀ (steven_falls stephanie_falls : ℕ),
    steven_falls = 3 →
    stephanie_falls = steven_falls + 13 →
    sonya_falls steven_falls stephanie_falls = 6 := by
  sorry

end NUMINAMATH_CALUDE_sonya_falls_count_l720_72040


namespace NUMINAMATH_CALUDE_alcohol_concentration_problem_l720_72046

theorem alcohol_concentration_problem (vessel1_capacity vessel2_capacity total_liquid final_vessel_capacity : ℝ)
  (vessel2_concentration final_concentration : ℝ) :
  vessel1_capacity = 2 →
  vessel2_capacity = 6 →
  vessel2_concentration = 55 / 100 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  final_concentration = 37 / 100 →
  ∃ initial_concentration : ℝ,
    initial_concentration = 20 / 100 ∧
    initial_concentration * vessel1_capacity + vessel2_concentration * vessel2_capacity =
    final_concentration * final_vessel_capacity :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_problem_l720_72046


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l720_72024

theorem no_valid_box_dimensions : ¬∃ (b c : ℤ), b ≤ c ∧ 2 * b * c + 2 * (2 * b + 2 * c + b * c) = 120 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l720_72024


namespace NUMINAMATH_CALUDE_digits_of_multiples_of_3_l720_72064

/-- The number of multiples of 3 from 1 to 100 -/
def multiplesOf3 : ℕ := 33

/-- The number of single-digit multiples of 3 from 1 to 100 -/
def singleDigitMultiples : ℕ := 3

/-- The number of two-digit multiples of 3 from 1 to 100 -/
def twoDigitMultiples : ℕ := multiplesOf3 - singleDigitMultiples

/-- The total number of digits written when listing all multiples of 3 from 1 to 100 -/
def totalDigits : ℕ := singleDigitMultiples * 1 + twoDigitMultiples * 2

theorem digits_of_multiples_of_3 : totalDigits = 63 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_multiples_of_3_l720_72064


namespace NUMINAMATH_CALUDE_correct_calculation_l720_72008

theorem correct_calculation (a : ℝ) : 3 * a^2 + 2 * a^2 = 5 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l720_72008


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l720_72026

theorem ceiling_floor_difference (x : ℤ) :
  let y : ℚ := 1/2
  (⌈(x : ℚ) + y⌉ - ⌊(x : ℚ) + y⌋ : ℤ) = 1 ∧ 
  (⌈(x : ℚ) + y⌉ - ((x : ℚ) + y) : ℚ) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l720_72026


namespace NUMINAMATH_CALUDE_total_guppies_per_day_l720_72007

/-- The number of guppies eaten by a moray eel per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta : ℕ := 5

/-- The number of guppies eaten by each betta fish per day -/
def betta_guppies : ℕ := 7

/-- The number of angelfish Jason has -/
def num_angelfish : ℕ := 3

/-- The number of guppies eaten by each angelfish per day -/
def angelfish_guppies : ℕ := 4

/-- The number of lionfish Jason has -/
def num_lionfish : ℕ := 2

/-- The number of guppies eaten by each lionfish per day -/
def lionfish_guppies : ℕ := 10

/-- Theorem stating the total number of guppies Jason needs to buy per day -/
theorem total_guppies_per_day :
  moray_eel_guppies +
  num_betta * betta_guppies +
  num_angelfish * angelfish_guppies +
  num_lionfish * lionfish_guppies = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_guppies_per_day_l720_72007


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l720_72053

/-- Geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_prop : a 2 * a 4 = a 5)
  (h_a4 : a 4 = 8) :
  ∃ (q : ℝ) (S : ℕ → ℝ),
    (q = 2) ∧
    (∀ n : ℕ, S n = 2^n - 1) ∧
    (∀ n : ℕ, S n = (a 1) * (1 - q^n) / (1 - q)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l720_72053


namespace NUMINAMATH_CALUDE_james_current_age_l720_72021

-- Define the ages as natural numbers
def james_age : ℕ := sorry
def john_age : ℕ := sorry
def tim_age : ℕ := sorry

-- State the given conditions
axiom age_difference : john_age = james_age + 12
axiom tim_age_relation : tim_age = 2 * john_age - 5
axiom tim_age_value : tim_age = 79

-- Theorem to prove
theorem james_current_age : james_age = 25 := by sorry

end NUMINAMATH_CALUDE_james_current_age_l720_72021


namespace NUMINAMATH_CALUDE_andrew_steps_to_meet_ben_l720_72034

/-- The distance between Andrew's and Ben's houses in feet -/
def distance : ℝ := 21120

/-- The ratio of Ben's speed to Andrew's speed -/
def speed_ratio : ℝ := 3

/-- The length of Andrew's step in feet -/
def step_length : ℝ := 3

/-- The number of steps Andrew takes before meeting Ben -/
def steps : ℕ := 1760

theorem andrew_steps_to_meet_ben :
  (distance / (1 + speed_ratio)) / step_length = steps := by
  sorry

end NUMINAMATH_CALUDE_andrew_steps_to_meet_ben_l720_72034


namespace NUMINAMATH_CALUDE_union_equals_A_iff_m_in_range_l720_72068

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - (2*m + 1)*x + 2*m < 0}

-- State the theorem
theorem union_equals_A_iff_m_in_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_iff_m_in_range_l720_72068


namespace NUMINAMATH_CALUDE_locus_of_centers_l720_72085

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₃ with equation (x - 3)² + y² = 25 -/
def C₃ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₃ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₃ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and internally tangent to C₃ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₃ a b r) → 
  12 * a^2 + 16 * b^2 - 36 * a - 81 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l720_72085


namespace NUMINAMATH_CALUDE_tower_surface_area_l720_72030

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the surface area of a cube given its visible faces -/
def surfaceArea (cube : Cube) (visibleFaces : ℕ) : ℕ :=
  visibleFaces * cube.sideLength * cube.sideLength

/-- Represents a tower of cubes -/
def CubeTower := List Cube

/-- Calculates the total surface area of a tower of cubes -/
def totalSurfaceArea (tower : CubeTower) : ℕ :=
  match tower with
  | [] => 0
  | [c] => surfaceArea c 6  -- Top cube has all 6 faces visible
  | c :: rest => surfaceArea c 5 + (rest.map (surfaceArea · 4)).sum

theorem tower_surface_area :
  let tower : CubeTower := [
    { sideLength := 1 },
    { sideLength := 2 },
    { sideLength := 3 },
    { sideLength := 4 },
    { sideLength := 5 },
    { sideLength := 6 },
    { sideLength := 7 }
  ]
  totalSurfaceArea tower = 610 := by sorry

end NUMINAMATH_CALUDE_tower_surface_area_l720_72030


namespace NUMINAMATH_CALUDE_angle_problem_l720_72099

theorem angle_problem (x : ℝ) :
  (x > 0) →
  (x - 30 > 0) →
  (2 * x + (x - 30) = 360) →
  (x = 130) := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l720_72099


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l720_72078

/-- Represents the number of people to be seated. -/
def num_people : ℕ := 4

/-- Represents the total number of chairs in a row. -/
def total_chairs : ℕ := 8

/-- Represents the number of consecutive empty seats required. -/
def consecutive_empty_seats : ℕ := 3

/-- Calculates the number of seating arrangements for the given conditions. -/
def seating_arrangements (p : ℕ) (c : ℕ) (e : ℕ) : ℕ :=
  (Nat.factorial (p + 1)) * (c - p - e + 1)

/-- Theorem stating the number of seating arrangements for the given conditions. -/
theorem seating_arrangements_count :
  seating_arrangements num_people total_chairs consecutive_empty_seats = 600 :=
by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_count_l720_72078


namespace NUMINAMATH_CALUDE_fraction_addition_l720_72062

theorem fraction_addition : (8 : ℚ) / 12 + (7 : ℚ) / 15 = (17 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l720_72062


namespace NUMINAMATH_CALUDE_inequality_implies_absolute_value_order_l720_72020

theorem inequality_implies_absolute_value_order 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (h : a^2 / (b^2 + c^2) < b^2 / (c^2 + a^2) ∧ b^2 / (c^2 + a^2) < c^2 / (a^2 + b^2)) : 
  |a| < |b| ∧ |b| < |c| := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_absolute_value_order_l720_72020


namespace NUMINAMATH_CALUDE_root_sum_ratio_l720_72042

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ p q : ℝ, 
    (∀ m : ℝ, m * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    p / q + q / p = 2 ∧
    (m₁ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₁ * (q^2 - 3*q) + 2*q + 7 = 0) ∧
    (m₂ * (p^2 - 3*p) + 2*p + 7 = 0 ∧ m₂ * (q^2 - 3*q) + 2*q + 7 = 0)) →
  m₁ / m₂ + m₂ / m₁ = 85/2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l720_72042


namespace NUMINAMATH_CALUDE_current_speed_l720_72073

/-- 
Given a man's speed with and against a current, this theorem proves 
the speed of the current.
-/
theorem current_speed 
  (speed_with_current : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : speed_against_current = 8) :
  ∃ (man_speed current_speed : ℝ),
    man_speed + current_speed = speed_with_current ∧
    man_speed - current_speed = speed_against_current ∧
    current_speed = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_current_speed_l720_72073


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l720_72095

/-- Definition of an O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49,16,x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l720_72095


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l720_72069

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, (x + 3)^2 + 2*(y - 2)^2 + 4*(x - 7)^2 + (y + 4)^2 ≥ 104 ∧
  ∃ x₀ y₀ : ℝ, (x₀ + 3)^2 + 2*(y₀ - 2)^2 + 4*(x₀ - 7)^2 + (y₀ + 4)^2 = 104 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l720_72069


namespace NUMINAMATH_CALUDE_rayman_workout_hours_l720_72012

/-- Represents the workout hours of Rayman, Junior, and Wolverine in a week --/
structure WorkoutHours where
  rayman : ℝ
  junior : ℝ
  wolverine : ℝ

/-- Defines the relationship between Rayman's, Junior's, and Wolverine's workout hours --/
def valid_workout_hours (h : WorkoutHours) : Prop :=
  h.rayman = h.junior / 2 ∧
  h.wolverine = 2 * (h.rayman + h.junior) ∧
  h.wolverine = 60

/-- Theorem stating that Rayman works out for 10 hours in a week --/
theorem rayman_workout_hours (h : WorkoutHours) (hvalid : valid_workout_hours h) : 
  h.rayman = 10 := by
  sorry

#check rayman_workout_hours

end NUMINAMATH_CALUDE_rayman_workout_hours_l720_72012


namespace NUMINAMATH_CALUDE_randy_blocks_left_l720_72061

/-- Calculates the number of blocks Randy has left after a series of actions. -/
def blocks_left (initial : ℕ) (used : ℕ) (given_away : ℕ) (bought : ℕ) : ℕ :=
  initial - used - given_away + bought

/-- Proves that Randy has 70 blocks left after his actions. -/
theorem randy_blocks_left : 
  blocks_left 78 19 25 36 = 70 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_left_l720_72061


namespace NUMINAMATH_CALUDE_exists_rank_with_profit_2016_l720_72047

/-- The profit of a firm given its rank -/
def profit : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 1) => profit n + (n + 1)

/-- The theorem stating that there exists a rank with profit 2016 -/
theorem exists_rank_with_profit_2016 : ∃ n : ℕ, profit n = 2016 := by
  sorry

end NUMINAMATH_CALUDE_exists_rank_with_profit_2016_l720_72047


namespace NUMINAMATH_CALUDE_modular_home_cost_modular_home_cost_proof_l720_72066

/-- Calculates the total cost of a modular home given specific conditions. -/
theorem modular_home_cost (kitchen_area : ℕ) (kitchen_cost : ℕ) 
  (bathroom_area : ℕ) (bathroom_cost : ℕ) (other_cost_per_sqft : ℕ) 
  (total_area : ℕ) (num_bathrooms : ℕ) : ℕ :=
  let total_module_area := kitchen_area + num_bathrooms * bathroom_area
  let remaining_area := total_area - total_module_area
  kitchen_cost + num_bathrooms * bathroom_cost + remaining_area * other_cost_per_sqft

/-- Proves that the total cost of the specified modular home is $174,000. -/
theorem modular_home_cost_proof : 
  modular_home_cost 400 20000 150 12000 100 2000 2 = 174000 := by
  sorry

end NUMINAMATH_CALUDE_modular_home_cost_modular_home_cost_proof_l720_72066


namespace NUMINAMATH_CALUDE_garden_table_bench_ratio_l720_72016

theorem garden_table_bench_ratio :
  ∀ (table_cost bench_cost : ℕ),
    bench_cost = 150 →
    table_cost + bench_cost = 450 →
    ∃ (k : ℕ), table_cost = k * bench_cost →
    (table_cost : ℚ) / (bench_cost : ℚ) = 2 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_table_bench_ratio_l720_72016


namespace NUMINAMATH_CALUDE_subtraction_of_fractions_l720_72057

theorem subtraction_of_fractions : (12 : ℚ) / 30 - 1 / 7 = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_fractions_l720_72057


namespace NUMINAMATH_CALUDE_orange_balls_count_l720_72098

theorem orange_balls_count (black white : ℕ) (p : ℚ) (orange : ℕ) : 
  black = 7 → 
  white = 6 → 
  p = 38095238095238093 / 100000000000000000 →
  (black : ℚ) / (orange + black + white : ℚ) = p →
  orange = 5 :=
by sorry

end NUMINAMATH_CALUDE_orange_balls_count_l720_72098


namespace NUMINAMATH_CALUDE_sum_of_max_min_M_l720_72090

/-- The set T of points (x, y) satisfying |x+1| + |y-2| ≤ 3 -/
def T : Set (ℝ × ℝ) := {p | |p.1 + 1| + |p.2 - 2| ≤ 3}

/-- The set M of values x + 2y for (x, y) in T -/
def M : Set ℝ := {z | ∃ p ∈ T, z = p.1 + 2 * p.2}

theorem sum_of_max_min_M : (⨆ z ∈ M, z) + (⨅ z ∈ M, z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_M_l720_72090


namespace NUMINAMATH_CALUDE_function_lower_bound_l720_72027

/-- Given a function f(x) = (1/2)x^4 - 2x^3 + 3m for all real x,
    if f(x) + 9 ≥ 0 for all real x, then m ≥ 3/2 --/
theorem function_lower_bound (m : ℝ) : 
  (∀ x : ℝ, (1/2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) → m ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l720_72027


namespace NUMINAMATH_CALUDE_C_power_50_l720_72084

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

theorem C_power_50 : C^50 = !![101, 50; -200, -99] := by sorry

end NUMINAMATH_CALUDE_C_power_50_l720_72084


namespace NUMINAMATH_CALUDE_angle_A_measure_max_area_l720_72054

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Condition that sides are positive
  ha : a > 0
  hb : b > 0
  hc : c > 0
  -- Condition that angles are between 0 and π
  hA : 0 < A ∧ A < π
  hB : 0 < B ∧ B < π
  hC : 0 < C ∧ C < π
  -- Condition that angles sum to π
  hsum : A + B + C = π
  -- Law of cosines
  hlawA : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  hlawB : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  hlawC : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Part 1
theorem angle_A_measure (t : Triangle) (h : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) :
  t.A = 2*π/3 := by sorry

-- Part 2
theorem max_area (t : Triangle) (h1 : t.b^2 + t.c^2 - t.a^2 + t.b*t.c = 0) (h2 : t.a = Real.sqrt 3) :
  (t.b * t.c * Real.sin t.A / 2) ≤ Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_A_measure_max_area_l720_72054


namespace NUMINAMATH_CALUDE_bobs_weight_l720_72018

theorem bobs_weight (j b : ℝ) 
  (sum_condition : j + b = 180)
  (diff_condition : b - j = b / 2) : 
  b = 120 := by sorry

end NUMINAMATH_CALUDE_bobs_weight_l720_72018


namespace NUMINAMATH_CALUDE_apple_sale_percentage_l720_72029

theorem apple_sale_percentage (total_apples : ℝ) (first_batch_percentage : ℝ) 
  (first_batch_profit : ℝ) (second_batch_profit : ℝ) (total_profit : ℝ) :
  first_batch_percentage > 0 ∧ first_batch_percentage < 100 →
  first_batch_profit = second_batch_profit →
  first_batch_profit = total_profit →
  (100 - first_batch_percentage) = (100 - first_batch_percentage) := by
sorry

end NUMINAMATH_CALUDE_apple_sale_percentage_l720_72029


namespace NUMINAMATH_CALUDE_sum_in_base_nine_l720_72019

/-- Represents a number in base 9 --/
def BaseNine : Type := List Nat

/-- Converts a base 9 number to its decimal representation --/
def to_decimal (n : BaseNine) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Adds two base 9 numbers --/
def add_base_nine (a b : BaseNine) : BaseNine :=
  sorry

/-- Theorem: The sum of 254₉, 367₉, and 142₉ is 774₉ in base 9 --/
theorem sum_in_base_nine :
  let a : BaseNine := [4, 5, 2]
  let b : BaseNine := [7, 6, 3]
  let c : BaseNine := [2, 4, 1]
  let result : BaseNine := [4, 7, 7]
  add_base_nine (add_base_nine a b) c = result :=
sorry

end NUMINAMATH_CALUDE_sum_in_base_nine_l720_72019


namespace NUMINAMATH_CALUDE_optimal_distribution_part1_optimal_distribution_part2_l720_72049

/-- Represents the types of vegetables -/
inductive VegetableType
| A
| B
| C

/-- Properties of each vegetable type -/
def tons_per_truck (v : VegetableType) : ℚ :=
  match v with
  | .A => 2
  | .B => 1
  | .C => 2.5

def profit_per_ton (v : VegetableType) : ℚ :=
  match v with
  | .A => 5
  | .B => 7
  | .C => 4

/-- Theorem for part 1 -/
theorem optimal_distribution_part1 :
  ∃ (b c : ℕ),
    b + c = 14 ∧
    b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 17 ∧
    b = 12 ∧ c = 2 := by sorry

/-- Theorem for part 2 -/
theorem optimal_distribution_part2 :
  ∃ (a b c : ℕ) (max_profit : ℚ),
    a + b + c = 30 ∧
    1 ≤ a ∧ a ≤ 10 ∧
    a * tons_per_truck VegetableType.A + b * tons_per_truck VegetableType.B + c * tons_per_truck VegetableType.C = 48 ∧
    a = 9 ∧ b = 15 ∧ c = 6 ∧
    max_profit = 255 ∧
    (∀ (a' b' c' : ℕ),
      a' + b' + c' = 30 →
      1 ≤ a' ∧ a' ≤ 10 →
      a' * tons_per_truck VegetableType.A + b' * tons_per_truck VegetableType.B + c' * tons_per_truck VegetableType.C = 48 →
      a' * tons_per_truck VegetableType.A * profit_per_ton VegetableType.A +
      b' * tons_per_truck VegetableType.B * profit_per_ton VegetableType.B +
      c' * tons_per_truck VegetableType.C * profit_per_ton VegetableType.C ≤ max_profit) := by sorry

end NUMINAMATH_CALUDE_optimal_distribution_part1_optimal_distribution_part2_l720_72049


namespace NUMINAMATH_CALUDE_min_value_product_l720_72025

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x / y + y / z + z / x + y / x + z / y + x / z = 10) :
  (x / y + y / z + z / x) * (y / x + z / y + x / z) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l720_72025


namespace NUMINAMATH_CALUDE_soccer_tournament_equation_l720_72014

theorem soccer_tournament_equation (x : ℕ) (h : x > 1) : 
  (x.choose 2 = 28) ↔ (x * (x - 1) / 2 = 28) := by sorry

end NUMINAMATH_CALUDE_soccer_tournament_equation_l720_72014


namespace NUMINAMATH_CALUDE_inequality_system_solution_l720_72022

theorem inequality_system_solution (x : ℝ) :
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l720_72022


namespace NUMINAMATH_CALUDE_store_profit_is_33_percent_l720_72009

/-- Calculates the store's profit percentage given the markups, discount, and shipping cost -/
def store_profit_percentage (first_markup : ℝ) (second_markup : ℝ) (discount : ℝ) (shipping_cost : ℝ) : ℝ :=
  let price_after_first_markup := 1 + first_markup
  let price_after_second_markup := price_after_first_markup + second_markup * price_after_first_markup
  let price_after_discount := price_after_second_markup * (1 - discount)
  let total_cost := 1 + shipping_cost
  price_after_discount - total_cost

/-- Theorem stating that the store's profit is 33% of the original cost price -/
theorem store_profit_is_33_percent :
  store_profit_percentage 0.20 0.25 0.08 0.05 = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_is_33_percent_l720_72009


namespace NUMINAMATH_CALUDE_range_of_m_l720_72031

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ 1 - (x - 1) / 3 ∧ 1 - (x - 1) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- "not p" is sufficient for "not q"
  (∃ x, ¬(q x m) ∧ p x) ∧     -- "not p" is not necessary for "not q"
  (m > 0) →                   -- given condition m > 0
  m ≤ 3 :=                    -- prove that m ≤ 3
sorry

end NUMINAMATH_CALUDE_range_of_m_l720_72031


namespace NUMINAMATH_CALUDE_stream_speed_l720_72074

/-- Proves that the speed of the stream is 4 km/hr, given the boat's speed in still water
    and the time and distance traveled downstream. -/
theorem stream_speed (boat_speed : ℝ) (time : ℝ) (distance : ℝ) :
  boat_speed = 16 →
  time = 3 →
  distance = 60 →
  ∃ (stream_speed : ℝ), stream_speed = 4 ∧ distance = (boat_speed + stream_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l720_72074


namespace NUMINAMATH_CALUDE_blind_cave_scorpion_diet_l720_72000

/-- The number of segments in the first millipede eaten by a blind cave scorpion -/
def first_millipede_segments : ℕ := 60

/-- The total number of segments the scorpion needs to eat daily -/
def total_required_segments : ℕ := 800

/-- The number of additional 50-segment millipedes the scorpion needs to eat -/
def additional_millipedes : ℕ := 10

/-- The number of segments in each additional millipede -/
def segments_per_additional_millipede : ℕ := 50

theorem blind_cave_scorpion_diet (x : ℕ) :
  x = first_millipede_segments ↔
    x + 2 * (2 * x) + additional_millipedes * segments_per_additional_millipede = total_required_segments :=
by
  sorry

end NUMINAMATH_CALUDE_blind_cave_scorpion_diet_l720_72000


namespace NUMINAMATH_CALUDE_ten_player_tournament_matches_l720_72043

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches :
  num_matches 10 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ten_player_tournament_matches_l720_72043


namespace NUMINAMATH_CALUDE_squares_sequence_correct_l720_72055

/-- Represents the number of nonoverlapping unit squares in figure n -/
def squares (n : ℕ) : ℕ :=
  3 * n^2 + n + 1

theorem squares_sequence_correct : 
  squares 0 = 1 ∧ 
  squares 1 = 5 ∧ 
  squares 2 = 15 ∧ 
  squares 3 = 29 ∧ 
  squares 4 = 49 ∧ 
  squares 100 = 30101 :=
by sorry

end NUMINAMATH_CALUDE_squares_sequence_correct_l720_72055


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l720_72035

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l720_72035


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l720_72023

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 :
  ∃ (p : ℕ), Prime p ∧ p ∣ (4^17 - 2^29) ∧ ∀ (q : ℕ), Prime q → q ∣ (4^17 - 2^29) → q ≤ p ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l720_72023


namespace NUMINAMATH_CALUDE_unique_intersection_l720_72076

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-1, 1-a, 9}

theorem unique_intersection (a : ℝ) : A a ∩ B a = {9} → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l720_72076


namespace NUMINAMATH_CALUDE_square_remainder_mod_16_l720_72004

theorem square_remainder_mod_16 (n : ℤ) : ∃ k : ℤ, 0 ≤ k ∧ k < 4 ∧ (n^2) % 16 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_16_l720_72004


namespace NUMINAMATH_CALUDE_outfit_choices_count_l720_72089

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 5

/-- The number of shirts available -/
def num_shirts : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfit combinations where all items are the same color -/
def same_color_combinations : ℕ := num_colors

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - same_color_combinations

theorem outfit_choices_count : valid_outfit_choices = 120 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_count_l720_72089


namespace NUMINAMATH_CALUDE_certain_number_proof_l720_72081

theorem certain_number_proof : ∃ x : ℝ, (1/4 * x + 15 = 27) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l720_72081


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l720_72041

theorem gcd_of_three_numbers :
  Nat.gcd 45321 (Nat.gcd 76543 123456) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l720_72041


namespace NUMINAMATH_CALUDE_f_composition_value_l720_72096

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then
    if z.re = 0 then 2 * z else -z^2 + 1
  else z^2 + 1

theorem f_composition_value : f (f (f (f (1 + I)))) = 378 + 336 * I := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l720_72096


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l720_72097

theorem weekend_rain_probability
  (p_saturday : ℝ)
  (p_sunday : ℝ)
  (p_sunday_given_saturday : ℝ)
  (h1 : p_saturday = 0.3)
  (h2 : p_sunday = 0.6)
  (h3 : p_sunday_given_saturday = 0.8) :
  1 - ((1 - p_saturday) * (1 - p_sunday) + p_saturday * (1 - p_sunday_given_saturday)) = 0.66 := by
  sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l720_72097


namespace NUMINAMATH_CALUDE_least_valid_number_l720_72017

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * m + d ∧ 
    1 ≤ d ∧ d ≤ 9 ∧ 
    m = n / 25

theorem least_valid_number : 
  (∀ k < 3125, ¬(is_valid_number k)) ∧ is_valid_number 3125 :=
sorry

end NUMINAMATH_CALUDE_least_valid_number_l720_72017


namespace NUMINAMATH_CALUDE_shaded_area_of_square_l720_72006

theorem shaded_area_of_square (r : ℝ) (h1 : r = 1/4) :
  (∑' n, r^n) * r = 1/3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_l720_72006


namespace NUMINAMATH_CALUDE_parallelogram_count_l720_72092

/-- Given a triangle ABC with each side divided into n equal segments and connected by parallel lines,
    f(n) represents the total number of parallelograms formed within the network. -/
def f (n : ℕ) : ℕ := 3 * (Nat.choose (n + 2) 4)

/-- Theorem stating that f(n) correctly counts the number of parallelograms in the described configuration. -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * (Nat.choose (n + 2) 4) := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_l720_72092


namespace NUMINAMATH_CALUDE_baxter_peanut_purchase_l720_72071

/-- Calculates the pounds of peanuts purchased over the minimum -/
def peanuts_over_minimum (cost_per_pound : ℚ) (minimum_pounds : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent / cost_per_pound) - minimum_pounds

theorem baxter_peanut_purchase : 
  let cost_per_pound : ℚ := 3
  let minimum_pounds : ℚ := 15
  let total_spent : ℚ := 105
  peanuts_over_minimum cost_per_pound minimum_pounds total_spent = 20 := by
sorry

end NUMINAMATH_CALUDE_baxter_peanut_purchase_l720_72071


namespace NUMINAMATH_CALUDE_surface_area_of_combined_solid_l720_72050

/-- Calculates the surface area of a rectangular solid -/
def surfaceAreaRect (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Represents the combined solid formed by attaching a rectangular prism to a rectangular solid -/
structure CombinedSolid where
  mainLength : ℝ
  mainWidth : ℝ
  mainHeight : ℝ
  attachedLength : ℝ
  attachedWidth : ℝ
  attachedHeight : ℝ

/-- Calculates the total surface area of the combined solid -/
def totalSurfaceArea (s : CombinedSolid) : ℝ :=
  surfaceAreaRect s.mainLength s.mainWidth s.mainHeight +
  surfaceAreaRect s.attachedLength s.attachedWidth s.attachedHeight -
  2 * (s.attachedLength * s.attachedWidth)

/-- The specific combined solid from the problem -/
def problemSolid : CombinedSolid :=
  { mainLength := 4
    mainWidth := 3
    mainHeight := 2
    attachedLength := 2
    attachedWidth := 1
    attachedHeight := 1 }

theorem surface_area_of_combined_solid :
  totalSurfaceArea problemSolid = 58 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_combined_solid_l720_72050


namespace NUMINAMATH_CALUDE_meeting_participants_count_l720_72001

theorem meeting_participants_count :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 750 :=
by sorry

end NUMINAMATH_CALUDE_meeting_participants_count_l720_72001


namespace NUMINAMATH_CALUDE_beast_sports_meeting_l720_72079

theorem beast_sports_meeting (total : ℕ) (tigers lions leopards : ℕ) : 
  total = 220 →
  lions = 2 * tigers + 5 →
  leopards = 2 * lions - 5 →
  total = tigers + lions + leopards →
  leopards - tigers = 95 :=
by
  sorry

end NUMINAMATH_CALUDE_beast_sports_meeting_l720_72079


namespace NUMINAMATH_CALUDE_cubic_equation_sum_l720_72028

theorem cubic_equation_sum (p q r : ℝ) : 
  (p^3 - 6*p^2 + 11*p = 14) → 
  (q^3 - 6*q^2 + 11*q = 14) → 
  (r^3 - 6*r^2 + 11*r = 14) → 
  (p*q/r + q*r/p + r*p/q = -47/14) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_sum_l720_72028


namespace NUMINAMATH_CALUDE_cubic_sum_divided_by_quadratic_sum_l720_72075

theorem cubic_sum_divided_by_quadratic_sum (a b c : ℚ) 
  (ha : a = 7) (hb : b = 5) (hc : c = -2) : 
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 + c^2) = 460 / 43 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_divided_by_quadratic_sum_l720_72075


namespace NUMINAMATH_CALUDE_total_fudge_eaten_l720_72044

-- Define the conversion rate from pounds to ounces
def pounds_to_ounces : ℝ → ℝ := (· * 16)

-- Define the amount of fudge eaten by each person in pounds
def tomas_fudge : ℝ := 1.5
def katya_fudge : ℝ := 0.5
def boris_fudge : ℝ := 2

-- Theorem statement
theorem total_fudge_eaten :
  pounds_to_ounces tomas_fudge +
  pounds_to_ounces katya_fudge +
  pounds_to_ounces boris_fudge = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_fudge_eaten_l720_72044


namespace NUMINAMATH_CALUDE_intersection_line_equation_l720_72011

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 6*x - 7*y - 4*z - 2 = 0
def plane2 (x y z : ℝ) : Prop := x + 7*y - z - 5 = 0

-- Define the line equation
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 35 = (y - 4/7) / 2 ∧ (y - 4/7) / 2 = z / 49

-- Theorem statement
theorem intersection_line_equation :
  ∀ x y z : ℝ, plane1 x y z ∧ plane2 x y z → line_equation x y z :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l720_72011


namespace NUMINAMATH_CALUDE_smallestDualPalindromeCorrect_l720_72005

def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

def smallestDualPalindrome : ℕ := 15

theorem smallestDualPalindromeCorrect :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallestDualPalindromeCorrect_l720_72005


namespace NUMINAMATH_CALUDE_kobe_initial_order_proof_l720_72056

/-- Represents the number of pieces of fried chicken Kobe initially ordered -/
def kobe_initial_order : ℕ := 5

/-- Represents the number of pieces of fried chicken Pau initially ordered -/
def pau_initial_order : ℕ := 2 * kobe_initial_order

/-- Represents the total number of pieces of fried chicken Pau ate -/
def pau_total : ℕ := 20

theorem kobe_initial_order_proof :
  pau_initial_order + pau_initial_order = pau_total :=
by sorry

end NUMINAMATH_CALUDE_kobe_initial_order_proof_l720_72056


namespace NUMINAMATH_CALUDE_f_extrema_and_inequality_l720_72051

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

noncomputable def g (x : ℝ) : ℝ := (2/3) * x^3 + (1/2) * x^2

theorem f_extrema_and_inequality :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ 1 + (Real.exp 1)^2) ∧
  (∀ x ∈ Set.Ioi 1, f x < g x) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_and_inequality_l720_72051
