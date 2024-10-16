import Mathlib

namespace NUMINAMATH_CALUDE_boat_speed_l2890_289021

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream upstream : ℝ) (h1 : downstream = 15) (h2 : upstream = 7) :
  (downstream + upstream) / 2 = 11 :=
by
  sorry

#check boat_speed

end NUMINAMATH_CALUDE_boat_speed_l2890_289021


namespace NUMINAMATH_CALUDE_solution_of_equation_l2890_289085

theorem solution_of_equation (x : ℝ) :
  x ≠ 3 →
  ((2 - x) / (x - 3) = 0) ↔ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2890_289085


namespace NUMINAMATH_CALUDE_daily_evaporation_rate_l2890_289049

/-- Calculates the daily evaporation rate given initial water amount, time period, and evaporation percentage. -/
theorem daily_evaporation_rate 
  (initial_water : ℝ) 
  (days : ℕ) 
  (evaporation_percentage : ℝ) : 
  initial_water * evaporation_percentage / 100 / days = 0.1 :=
by
  -- Assuming initial_water = 10, days = 20, and evaporation_percentage = 2
  sorry

#check daily_evaporation_rate

end NUMINAMATH_CALUDE_daily_evaporation_rate_l2890_289049


namespace NUMINAMATH_CALUDE_fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l2890_289055

def dvd_sequence : Nat → Nat
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem fifth_rack_dvds : dvd_sequence 4 = 32 :=
by
  sorry

theorem sixth_rack_dvds : dvd_sequence 5 = 64 :=
by
  sorry

theorem prove_fifth_rack (h : dvd_sequence 5 = 64) : dvd_sequence 4 = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l2890_289055


namespace NUMINAMATH_CALUDE_harold_adrienne_speed_difference_l2890_289002

/-- Prove that Harold walks 1 mile per hour faster than Adrienne --/
theorem harold_adrienne_speed_difference :
  ∀ (total_distance : ℝ) (adrienne_speed : ℝ) (harold_catch_up_distance : ℝ),
    total_distance = 60 →
    adrienne_speed = 3 →
    harold_catch_up_distance = 12 →
    ∃ (harold_speed : ℝ),
      harold_speed > adrienne_speed ∧
      harold_speed - adrienne_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_harold_adrienne_speed_difference_l2890_289002


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_param_range_l2890_289005

/-- A function f(x) = -x^3 + ax^2 - x - 1 is monotonic on (-∞, +∞) if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_param_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_param_range_l2890_289005


namespace NUMINAMATH_CALUDE_odd_prime_factor_form_l2890_289097

theorem odd_prime_factor_form (p q : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) (hq : Nat.Prime q) (h_div : q ∣ 2^p - 1) :
  ∃ k : ℕ, q = 2*k*p + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_factor_form_l2890_289097


namespace NUMINAMATH_CALUDE_tan_n_theta_rational_odd_denominator_l2890_289072

/-- Given that tan(θ) = 2, prove that for all n ≥ 1, tan(nθ) is a rational number with an odd denominator -/
theorem tan_n_theta_rational_odd_denominator (θ : ℝ) (h : Real.tan θ = 2) :
  ∀ n : ℕ+, ∃ (p q : ℤ), Odd q ∧ Real.tan (n * θ) = p / q :=
sorry

end NUMINAMATH_CALUDE_tan_n_theta_rational_odd_denominator_l2890_289072


namespace NUMINAMATH_CALUDE_correlation_coefficient_formula_correlation_coefficient_problem_l2890_289042

/-- Given a linear regression equation ŷ = bx + a, where b is the slope,
    Sy^2 is the variance of y, and Sx^2 is the variance of x,
    prove that the correlation coefficient r = b * (√(Sx^2) / √(Sy^2)) -/
theorem correlation_coefficient_formula 
  (b : ℝ) (Sy_squared : ℝ) (Sx_squared : ℝ) (h1 : Sy_squared > 0) (h2 : Sx_squared > 0) :
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

/-- Given the specific values from the problem, prove that the correlation coefficient is 0.94 -/
theorem correlation_coefficient_problem :
  let b := 4.7
  let Sy_squared := 50
  let Sx_squared := 2
  let r := b * (Real.sqrt Sx_squared / Real.sqrt Sy_squared)
  ∀ ε > 0, |r - 0.94| < ε := by
sorry

end NUMINAMATH_CALUDE_correlation_coefficient_formula_correlation_coefficient_problem_l2890_289042


namespace NUMINAMATH_CALUDE_least_months_to_triple_debt_l2890_289084

theorem least_months_to_triple_debt (interest_rate : ℝ) (n : ℕ) : 
  interest_rate = 0.03 →
  n = 37 →
  (∀ m : ℕ, m < n → (1 + interest_rate)^m ≤ 3) ∧
  (1 + interest_rate)^n > 3 :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_debt_l2890_289084


namespace NUMINAMATH_CALUDE_light_bulb_resistance_l2890_289024

theorem light_bulb_resistance (U I R : ℝ) (hU : U = 220) (hI : I ≤ 0.11) (hOhm : I = U / R) : R ≥ 2000 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_resistance_l2890_289024


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2890_289054

/-- Represents a repeating decimal with an integer part, a non-repeating fractional part, and a repeating part -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ
  nonRepeatingPartLessThanOne : nonRepeatingPart < 1
  repeatingPartLessThanOne : repeatingPart < 1

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.nonRepeatingPart + d.repeatingPart / (1 - (1/10)^(d.repeatingPart.den))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (n d : ℤ) : Prop :=
  Nat.gcd n.natAbs d.natAbs = 1

theorem repeating_decimal_equiv_fraction :
  let d : RepeatingDecimal := ⟨0, 4/10, 37/100, by norm_num, by norm_num⟩
  d.toRational = 433 / 990 ∧ isLowestTerms 433 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2890_289054


namespace NUMINAMATH_CALUDE_root_preservation_l2890_289031

/-- Given a polynomial P(x) = x^3 + ax^2 + bx + c with three distinct real roots,
    the polynomial Q(x) = x^3 + ax^2 + (1/4)(a^2 + b)x + (1/8)(ab - c) also has three distinct real roots. -/
theorem root_preservation (a b c : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  ∃ (y₁ y₂ y₃ : ℝ), y₁ ≠ y₂ ∧ y₂ ≠ y₃ ∧ y₁ ≠ y₃ ∧
    (∀ x, x^3 + a*x^2 + (1/4)*(a^2 + b)*x + (1/8)*(a*b - c) = 0 ↔ x = y₁ ∨ x = y₂ ∨ x = y₃) :=
by sorry

end NUMINAMATH_CALUDE_root_preservation_l2890_289031


namespace NUMINAMATH_CALUDE_square_sum_and_product_l2890_289029

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l2890_289029


namespace NUMINAMATH_CALUDE_blue_face_cubes_count_l2890_289079

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with more than one blue face in a painted block -/
def count_multi_blue_face_cubes (b : Block) : Nat :=
  sorry

/-- The main theorem stating that a 5x3x1 block has 10 cubes with more than one blue face -/
theorem blue_face_cubes_count :
  let block := Block.mk 5 3 1
  count_multi_blue_face_cubes block = 10 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_cubes_count_l2890_289079


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2890_289073

/-- A geometric sequence of four numbers satisfying specific conditions -/
structure GeometricSequence where
  a : ℝ
  b : ℝ
  isGeometric : (a + 6) / a = (b + 5) / b
  sumOfSquares : (a + 6)^2 + a^2 + (b + 5)^2 + b^2 = 793

/-- The theorem stating the only possible solutions for the geometric sequence -/
theorem geometric_sequence_solution (seq : GeometricSequence) :
  (seq.a = 12 ∧ seq.b = 10) ∨ (seq.a = -18 ∧ seq.b = -15) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2890_289073


namespace NUMINAMATH_CALUDE_apples_per_box_is_10_l2890_289092

/-- The number of apples in each box. -/
def apples_per_box : ℕ := 10

/-- The number of boxes Merry had on Saturday. -/
def saturday_boxes : ℕ := 50

/-- The number of boxes Merry had on Sunday. -/
def sunday_boxes : ℕ := 25

/-- The total number of apples Merry sold. -/
def sold_apples : ℕ := 720

/-- The number of boxes Merry has left. -/
def remaining_boxes : ℕ := 3

/-- Theorem stating that the number of apples in each box is 10. -/
theorem apples_per_box_is_10 :
  apples_per_box * (saturday_boxes + sunday_boxes) - sold_apples = apples_per_box * remaining_boxes :=
by sorry

end NUMINAMATH_CALUDE_apples_per_box_is_10_l2890_289092


namespace NUMINAMATH_CALUDE_kannon_apples_difference_kannon_apples_difference_proof_l2890_289068

theorem kannon_apples_difference : ℕ → Prop :=
  fun x => 
    let apples_last_night : ℕ := 3
    let bananas_last_night : ℕ := 1
    let oranges_last_night : ℕ := 4
    let apples_today : ℕ := x
    let bananas_today : ℕ := 10 * bananas_last_night
    let oranges_today : ℕ := 2 * apples_today
    let total_fruits : ℕ := 39
    (apples_last_night + bananas_last_night + oranges_last_night + 
     apples_today + bananas_today + oranges_today = total_fruits) →
    (apples_today > apples_last_night) →
    (apples_today - apples_last_night = 4)

-- Proof
theorem kannon_apples_difference_proof : kannon_apples_difference 7 := by
  sorry

end NUMINAMATH_CALUDE_kannon_apples_difference_kannon_apples_difference_proof_l2890_289068


namespace NUMINAMATH_CALUDE_milly_fold_count_l2890_289089

/-- Represents the croissant-making process with given time constraints. -/
structure CroissantProcess where
  fold_time : ℕ         -- Time to fold dough once (in minutes)
  rest_time : ℕ         -- Time to rest dough once (in minutes)
  mix_time : ℕ          -- Time to mix ingredients (in minutes)
  bake_time : ℕ         -- Time to bake (in minutes)
  total_time : ℕ        -- Total time for the whole process (in minutes)

/-- Calculates the number of times the dough needs to be folded. -/
def fold_count (process : CroissantProcess) : ℕ :=
  ((process.total_time - process.mix_time - process.bake_time) / 
   (process.fold_time + process.rest_time))

/-- Theorem stating that for the given process, the dough needs to be folded 4 times. -/
theorem milly_fold_count : 
  let process : CroissantProcess := {
    fold_time := 5,
    rest_time := 75,
    mix_time := 10,
    bake_time := 30,
    total_time := 6 * 60  -- 6 hours in minutes
  }
  fold_count process = 4 := by
  sorry

end NUMINAMATH_CALUDE_milly_fold_count_l2890_289089


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2890_289053

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2890_289053


namespace NUMINAMATH_CALUDE_base_prime_repr_294_l2890_289013

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of natural numbers is a valid base prime representation -/
def is_valid_base_prime_repr (repr : List ℕ) : Prop :=
  sorry

theorem base_prime_repr_294 :
  let repr := base_prime_repr 294
  is_valid_base_prime_repr repr ∧ repr = [2, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_prime_repr_294_l2890_289013


namespace NUMINAMATH_CALUDE_number_set_properties_l2890_289017

/-- A set of natural numbers excluding 1 -/
def NumberSet : Set ℕ :=
  {n : ℕ | n > 1}

/-- Predicate for a number being prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Predicate for a number being composite -/
def isComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

theorem number_set_properties (S : Set ℕ) (h : S = NumberSet) :
  (¬∀ n ∈ S, isComposite n) →
  (∃ n ∈ S, isPrime n) ∧
  (∀ n ∈ S, ¬isComposite n) ∧
  (∀ n ∈ S, isPrime n) ∧
  (∃ n ∈ S, isComposite n ∧ ∃ m ∈ S, isPrime m) ∧
  (∃ n ∈ S, isPrime n ∧ ∃ m ∈ S, isComposite m) :=
by sorry

end NUMINAMATH_CALUDE_number_set_properties_l2890_289017


namespace NUMINAMATH_CALUDE_certain_number_proof_l2890_289086

theorem certain_number_proof : ∃ x : ℝ, x * 16 = 3408 ∧ x * 1.6 = 340.8 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2890_289086


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l2890_289094

theorem shirt_cost_problem (x : ℝ) : 
  (3 * x + 2 * 20 = 85) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l2890_289094


namespace NUMINAMATH_CALUDE_max_value_theorem_l2890_289059

theorem max_value_theorem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  ∃ (M : ℝ), M = 100 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 → x^4 + y^2 + z^2 + w^2 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2890_289059


namespace NUMINAMATH_CALUDE_fish_pond_population_l2890_289019

-- Define the parameters
def initial_tagged : ℕ := 40
def second_catch : ℕ := 50
def tagged_in_second : ℕ := 2

-- Define the theorem
theorem fish_pond_population :
  let total_fish : ℕ := (initial_tagged * second_catch) / tagged_in_second
  total_fish = 1000 := by
  sorry

end NUMINAMATH_CALUDE_fish_pond_population_l2890_289019


namespace NUMINAMATH_CALUDE_total_defective_rate_l2890_289071

/-- Given two workers x and y who check products, with known defective rates and
    the fraction of products checked by worker y, prove the total defective rate. -/
theorem total_defective_rate 
  (defective_rate_x : ℝ) 
  (defective_rate_y : ℝ) 
  (fraction_checked_by_y : ℝ) 
  (h1 : defective_rate_x = 0.005) 
  (h2 : defective_rate_y = 0.008) 
  (h3 : fraction_checked_by_y = 0.5) 
  (h4 : fraction_checked_by_y ≥ 0 ∧ fraction_checked_by_y ≤ 1) : 
  defective_rate_x * (1 - fraction_checked_by_y) + defective_rate_y * fraction_checked_by_y = 0.0065 := by
  sorry

#check total_defective_rate

end NUMINAMATH_CALUDE_total_defective_rate_l2890_289071


namespace NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l2890_289016

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

theorem fib_gcd_consecutive (n : ℕ) : Nat.gcd (fib n) (fib (n + 1)) = 1 := by sorry

theorem fib_gcd_identity (m n : ℕ) : 
  fib (Nat.gcd m n) = Nat.gcd (fib m) (fib n) := by sorry

theorem fib_sum_identity (m n : ℕ) :
  fib (n + m) = fib m * fib (n + 1) + fib (m - 1) * fib n := by sorry

end NUMINAMATH_CALUDE_fib_gcd_consecutive_fib_gcd_identity_fib_sum_identity_l2890_289016


namespace NUMINAMATH_CALUDE_tank_insulation_problem_l2890_289082

/-- Proves that for a rectangular tank with given dimensions and insulation cost, 
    the third dimension is 2 feet. -/
theorem tank_insulation_problem (x : ℝ) : 
  x > 0 → 
  (2 * 3 * 5 + 2 * 3 * x + 2 * 5 * x) * 20 = 1240 → 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_tank_insulation_problem_l2890_289082


namespace NUMINAMATH_CALUDE_interest_rate_increase_l2890_289070

theorem interest_rate_increase (initial_rate : ℝ) (increase_percentage : ℝ) (final_rate : ℝ) : 
  initial_rate = 8.256880733944953 →
  increase_percentage = 10 →
  final_rate = initial_rate * (1 + increase_percentage / 100) →
  final_rate = 9.082568807339448 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_increase_l2890_289070


namespace NUMINAMATH_CALUDE_triangle_operation_result_l2890_289040

-- Define the triangle operation
def triangle (P Q : ℚ) : ℚ := (P + Q) / 3

-- State the theorem
theorem triangle_operation_result :
  triangle 3 (triangle 6 9) = 8 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_operation_result_l2890_289040


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_9_l2890_289011

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_9 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 9 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 9 → n ≤ m :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_9_l2890_289011


namespace NUMINAMATH_CALUDE_max_trailing_zeros_product_l2890_289045

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_product (a b c : ℕ) (h_sum : a + b + c = 1003) :
  (∃ n : ℕ, a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧
  ¬(∃ m : ℕ, a * b * c = m * 10^8) :=
by sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_product_l2890_289045


namespace NUMINAMATH_CALUDE_combination_equality_l2890_289032

theorem combination_equality (n : ℕ) : 
  Nat.choose 5 2 = Nat.choose 5 n → n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l2890_289032


namespace NUMINAMATH_CALUDE_triangle_inequality_l2890_289030

theorem triangle_inequality (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2890_289030


namespace NUMINAMATH_CALUDE_base_k_is_seven_l2890_289067

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := 
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Converts a number from base k to base 10 -/
def baseKToBase10 (n : ℕ) (k : ℕ) : ℕ := 
  (n / 100) * k^2 + ((n / 10) % 10) * k + (n % 10)

/-- The theorem stating that 7 is the base k where (524)₈ = (664)ₖ -/
theorem base_k_is_seven : 
  ∃ k : ℕ, k > 1 ∧ base8ToBase10 524 = baseKToBase10 664 k → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_base_k_is_seven_l2890_289067


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2890_289081

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = (k * b.1, k * b.2)

/-- The theorem states that if vectors (4,2) and (x,3) are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, 2) (x, 3) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2890_289081


namespace NUMINAMATH_CALUDE_calculation_proof_l2890_289063

theorem calculation_proof : (27 * 0.92 * 0.85) / (23 * 1.7 * 1.8) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2890_289063


namespace NUMINAMATH_CALUDE_B_is_largest_l2890_289035

/-- A is defined as the sum of 2023/2022 and 2023/2024 -/
def A : ℚ := 2023/2022 + 2023/2024

/-- B is defined as the sum of 2024/2023 and 2026/2023 -/
def B : ℚ := 2024/2023 + 2026/2023

/-- C is defined as the sum of 2025/2024 and 2025/2026 -/
def C : ℚ := 2025/2024 + 2025/2026

/-- Theorem stating that B is the largest among A, B, and C -/
theorem B_is_largest : B > A ∧ B > C := by
  sorry

end NUMINAMATH_CALUDE_B_is_largest_l2890_289035


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2890_289033

theorem smallest_prime_dividing_sum : Nat.minFac (7^7 + 3^14) = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l2890_289033


namespace NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2890_289069

theorem negation_of_square_positive_equals_zero :
  (¬ ∀ m : ℝ, m > 0 → m^2 = 0) ↔ (∀ m : ℝ, m ≤ 0 → m^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_square_positive_equals_zero_l2890_289069


namespace NUMINAMATH_CALUDE_expression_simplification_l2890_289047

theorem expression_simplification (a b c d x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) 
  (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  ∃ k : ℝ, 
    (x + a)^4 / ((a - b) * (a - c) * (a - d)) +
    (x + b)^4 / ((b - a) * (b - c) * (b - d)) +
    (x + c)^4 / ((c - a) * (c - b) * (c - d)) +
    (x + d)^4 / ((d - a) * (d - b) * (d - c)) =
    k * (x + a) * (x + b) * (x + c) * (x + d) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2890_289047


namespace NUMINAMATH_CALUDE_quadratic_root_geometric_sequence_l2890_289056

theorem quadratic_root_geometric_sequence (a b c : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- Condition: a ≥ b ≥ c ≥ 0
  (∃ r : ℝ, b = a * r ∧ c = a * r^2) →  -- Condition: a, b, c form a geometric sequence
  (∃! x : ℝ, a * x^2 + b * x + c = 0) →  -- Condition: quadratic has exactly one root
  (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1/8) :=  -- Conclusion: the root is -1/8
by sorry

end NUMINAMATH_CALUDE_quadratic_root_geometric_sequence_l2890_289056


namespace NUMINAMATH_CALUDE_decimal_division_multiplication_l2890_289022

theorem decimal_division_multiplication : (0.08 / 0.005) * 2 = 32 := by sorry

end NUMINAMATH_CALUDE_decimal_division_multiplication_l2890_289022


namespace NUMINAMATH_CALUDE_roots_of_equation_l2890_289075

def f (x : ℝ) : ℝ := x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {-2, -1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2890_289075


namespace NUMINAMATH_CALUDE_probability_of_overlap_l2890_289039

/-- Represents the duration of the entire time frame in minutes -/
def totalDuration : ℝ := 60

/-- Represents the waiting time of the train in minutes -/
def waitingTime : ℝ := 10

/-- Represents the area of the triangle in the graphical representation -/
def triangleArea : ℝ := 50

/-- Calculates the area of the parallelogram in the graphical representation -/
def parallelogramArea : ℝ := totalDuration * waitingTime

/-- Calculates the total area of overlap (favorable outcomes) -/
def overlapArea : ℝ := triangleArea + parallelogramArea

/-- Calculates the total area of all possible outcomes -/
def totalArea : ℝ := totalDuration * totalDuration

/-- Theorem stating the probability of Alex arriving while the train is at the station -/
theorem probability_of_overlap : overlapArea / totalArea = 11 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_overlap_l2890_289039


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2890_289050

/-- A right circular cylinder inscribed in a right circular cone --/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  h_diameter_height : cylinder_radius * 2 = cylinder_radius * 2
  h_cone_cylinder_axes : True  -- This condition is implicit and cannot be directly expressed

/-- The radius of the inscribed cylinder is 90/19 --/
theorem inscribed_cylinder_radius 
  (c : InscribedCylinder) 
  (h_cone_diameter : c.cone_diameter = 18) 
  (h_cone_altitude : c.cone_altitude = 20) : 
  c.cylinder_radius = 90 / 19 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2890_289050


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_l2890_289015

theorem smallest_solution_quartic (x : ℝ) : 
  (x^4 - 50*x^2 + 625 = 0) → (∃ y : ℝ, y^4 - 50*y^2 + 625 = 0 ∧ y ≤ x) → x ≥ -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_l2890_289015


namespace NUMINAMATH_CALUDE_differential_equation_classification_l2890_289080

-- Define a type for equations
inductive Equation
| A : Equation  -- y' + 3x = 0
| B : Equation  -- y² + x² = 5
| C : Equation  -- y = e^x
| D : Equation  -- y = ln|x| + C
| E : Equation  -- y'y - x = 0
| F : Equation  -- 2dy + 3xdx = 0

-- Define a predicate for differential equations
def isDifferentialEquation : Equation → Prop
| Equation.A => True
| Equation.B => False
| Equation.C => False
| Equation.D => False
| Equation.E => True
| Equation.F => True

-- Theorem statement
theorem differential_equation_classification :
  (isDifferentialEquation Equation.A ∧
   isDifferentialEquation Equation.E ∧
   isDifferentialEquation Equation.F) ∧
  (¬isDifferentialEquation Equation.B ∧
   ¬isDifferentialEquation Equation.C ∧
   ¬isDifferentialEquation Equation.D) :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_classification_l2890_289080


namespace NUMINAMATH_CALUDE_impossible_exact_usage_l2890_289078

theorem impossible_exact_usage (p q r : ℕ) : 
  ¬∃ (x y z : ℤ), (2*x + 2*z = 2*p + 2*r + 2) ∧ 
                   (2*x + y = 2*p + q + 1) ∧ 
                   (y + z = q + r) :=
sorry

end NUMINAMATH_CALUDE_impossible_exact_usage_l2890_289078


namespace NUMINAMATH_CALUDE_happy_island_parrots_l2890_289093

theorem happy_island_parrots (total_birds : ℕ) (yellow_fraction : ℚ) (red_parrots : ℕ) :
  total_birds = 120 →
  yellow_fraction = 2/3 →
  red_parrots = total_birds - (yellow_fraction * total_birds).floor →
  red_parrots = 40 := by
sorry

end NUMINAMATH_CALUDE_happy_island_parrots_l2890_289093


namespace NUMINAMATH_CALUDE_probability_three_colors_l2890_289091

/-- The probability of picking at least one ball of each color when selecting 3 balls from a jar
    containing 8 black, 5 white, and 3 red balls is 3/14. -/
theorem probability_three_colors (black white red : ℕ) (total : ℕ) (h1 : black = 8) (h2 : white = 5) (h3 : red = 3) 
    (h4 : total = black + white + red) : 
  (black * white * red : ℚ) / (total * (total - 1) * (total - 2) / 6) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_colors_l2890_289091


namespace NUMINAMATH_CALUDE_multiplication_distributive_property_l2890_289025

theorem multiplication_distributive_property (m : ℝ) : (4*m + 1) * (2*m) = 8*m^2 + 2*m := by
  sorry

end NUMINAMATH_CALUDE_multiplication_distributive_property_l2890_289025


namespace NUMINAMATH_CALUDE_show_revenue_calculation_l2890_289099

def first_showing_attendance : ℕ := 200
def second_showing_multiplier : ℕ := 3
def ticket_price : ℕ := 25

theorem show_revenue_calculation :
  let second_showing_attendance := first_showing_attendance * second_showing_multiplier
  let total_attendance := first_showing_attendance + second_showing_attendance
  let total_revenue := total_attendance * ticket_price
  total_revenue = 20000 := by
  sorry

end NUMINAMATH_CALUDE_show_revenue_calculation_l2890_289099


namespace NUMINAMATH_CALUDE_medal_winners_combinations_l2890_289052

theorem medal_winners_combinations (semifinalists : ℕ) (advance : ℕ) (finalists : ℕ) (medals : ℕ) :
  semifinalists = 8 →
  advance = semifinalists - 2 →
  finalists = advance →
  medals = 3 →
  Nat.choose finalists medals = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_medal_winners_combinations_l2890_289052


namespace NUMINAMATH_CALUDE_four_row_lattice_triangles_l2890_289038

/-- Represents a modified triangular lattice with n rows -/
structure ModifiedTriangularLattice (n : ℕ) where
  -- Each row i has i dots, with the base row having n dots
  rows : Fin n → ℕ
  rows_def : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a modified triangular lattice -/
def countTriangles (n : ℕ) : ℕ :=
  let lattice := ModifiedTriangularLattice n
  -- The actual counting logic would go here
  0 -- Placeholder

/-- The theorem stating that a 4-row modified triangular lattice contains 22 triangles -/
theorem four_row_lattice_triangles :
  countTriangles 4 = 22 := by
  sorry

#check four_row_lattice_triangles

end NUMINAMATH_CALUDE_four_row_lattice_triangles_l2890_289038


namespace NUMINAMATH_CALUDE_power_product_equality_l2890_289066

theorem power_product_equality : (0.125^8 * (-8)^7) = -0.125 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2890_289066


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2890_289076

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, (z : ℚ) / 4 + 3 / 7 > 2 / 3 → y ≤ z :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2890_289076


namespace NUMINAMATH_CALUDE_division_problem_l2890_289057

theorem division_problem (n : ℕ) : 
  n / 20 = 10 ∧ n % 20 = 10 → n = 210 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2890_289057


namespace NUMINAMATH_CALUDE_sum_equals_270_l2890_289062

/-- The sum of the arithmetic sequence with first term a, common difference d, and n terms -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

/-- The sum of two arithmetic sequences, each with 5 terms and common difference 10 -/
def two_sequence_sum (a₁ a₂ : ℕ) : ℕ := arithmetic_sum a₁ 10 5 + arithmetic_sum a₂ 10 5

theorem sum_equals_270 : two_sequence_sum 3 11 = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_270_l2890_289062


namespace NUMINAMATH_CALUDE_max_value_cube_ratio_l2890_289044

theorem max_value_cube_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cube_ratio_l2890_289044


namespace NUMINAMATH_CALUDE_intersection_empty_condition_l2890_289077

theorem intersection_empty_condition (a : ℝ) : 
  let A : Set ℝ := Set.Iio (2 * a)
  let B : Set ℝ := Set.Ioi (3 - a^2)
  A ∩ B = ∅ → 2 * a ≤ 3 - a^2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_condition_l2890_289077


namespace NUMINAMATH_CALUDE_function_extrema_sum_l2890_289046

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_extrema_sum (a : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = 12) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l2890_289046


namespace NUMINAMATH_CALUDE_zoo_elephants_l2890_289020

theorem zoo_elephants (giraffes : ℕ) (penguins : ℕ) (total : ℕ) (elephants : ℕ) : 
  giraffes = 5 →
  penguins = 2 * giraffes →
  penguins = (20 : ℚ) / 100 * total →
  elephants = (4 : ℚ) / 100 * total →
  elephants = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_elephants_l2890_289020


namespace NUMINAMATH_CALUDE_jim_has_220_buicks_l2890_289098

/-- Represents the number of model cars of each brand Jim has. -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- The conditions of Jim's model car collection. -/
def jim_collection (cars : ModelCars) : Prop :=
  cars.ford + cars.buick + cars.chevy = 301 ∧
  cars.buick = 4 * cars.ford ∧
  cars.ford = 2 * cars.chevy + 3

/-- Theorem stating that Jim has 220 Buicks. -/
theorem jim_has_220_buicks :
  ∃ (cars : ModelCars), jim_collection cars ∧ cars.buick = 220 := by
  sorry

end NUMINAMATH_CALUDE_jim_has_220_buicks_l2890_289098


namespace NUMINAMATH_CALUDE_one_third_of_1206_percent_of_200_l2890_289028

theorem one_third_of_1206_percent_of_200 : (1206 / 3) / 200 * 100 = 201 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_1206_percent_of_200_l2890_289028


namespace NUMINAMATH_CALUDE_rotation_of_point_l2890_289001

def rotate90ClockwiseAboutOrigin (x y : ℝ) : ℝ × ℝ := (y, -x)

theorem rotation_of_point :
  let D : ℝ × ℝ := (-3, 2)
  rotate90ClockwiseAboutOrigin D.1 D.2 = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_point_l2890_289001


namespace NUMINAMATH_CALUDE_pattern_circle_area_ratio_l2890_289064

-- Define the circle
def circle_radius : ℝ := 3

-- Define the rectangle
def rectangle_length : ℝ := 12
def rectangle_width : ℝ := 6

-- Define the number of arcs
def num_arcs : ℕ := 6

-- Theorem statement
theorem pattern_circle_area_ratio :
  let circle_area := π * circle_radius^2
  let pattern_area := circle_area  -- Assumption: rearranged arcs preserve total area
  pattern_area / circle_area = 1 := by sorry

end NUMINAMATH_CALUDE_pattern_circle_area_ratio_l2890_289064


namespace NUMINAMATH_CALUDE_manhattan_to_bronx_travel_time_l2890_289060

/-- The total travel time from Manhattan to the Bronx -/
def total_travel_time (subway_time train_time bike_time : ℕ) : ℕ :=
  subway_time + train_time + bike_time

/-- Theorem stating that the total travel time is 38 hours -/
theorem manhattan_to_bronx_travel_time :
  ∃ (subway_time train_time bike_time : ℕ),
    subway_time = 10 ∧
    train_time = 2 * subway_time ∧
    bike_time = 8 ∧
    total_travel_time subway_time train_time bike_time = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_manhattan_to_bronx_travel_time_l2890_289060


namespace NUMINAMATH_CALUDE_count_solutions_l2890_289087

def positive_integer_solutions : Nat :=
  let n := 25
  let k := 5
  let min_values := [2, 3, 1, 2, 4]
  let remaining := n - (min_values.sum)
  Nat.choose (remaining + k - 1) (k - 1)

theorem count_solutions :
  positive_integer_solutions = 1190 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_l2890_289087


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2890_289051

/-- An isosceles triangle with sides of length 3 and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 → b = 6 → c = 3 →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2890_289051


namespace NUMINAMATH_CALUDE_half_area_triangle_l2890_289083

/-- A square in a 2D plane -/
structure Square where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ
  w : ℝ × ℝ

/-- The area of a triangle given three points -/
def triangleArea (a b c : ℝ × ℝ) : ℝ := sorry

/-- The area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Theorem: The coordinates (3, 3) for point T' result in the area of triangle YZT' 
    being half the area of square XYZW, given that XYZW is a square with X at (0,0) 
    and Z at (3,3) -/
theorem half_area_triangle (xyzw : Square) 
  (h1 : xyzw.x = (0, 0))
  (h2 : xyzw.z = (3, 3))
  (t' : ℝ × ℝ)
  (h3 : t' = (3, 3)) : 
  triangleArea xyzw.y xyzw.z t' = (1/2) * squareArea xyzw := by
  sorry

end NUMINAMATH_CALUDE_half_area_triangle_l2890_289083


namespace NUMINAMATH_CALUDE_total_fish_count_l2890_289006

/-- The number of tuna in the sea -/
def num_tuna : ℕ := 5

/-- The number of spearfish in the sea -/
def num_spearfish : ℕ := 2

/-- The total number of fish in the sea -/
def total_fish : ℕ := num_tuna + num_spearfish

theorem total_fish_count : total_fish = 7 := by sorry

end NUMINAMATH_CALUDE_total_fish_count_l2890_289006


namespace NUMINAMATH_CALUDE_all_grids_have_uniform_subgrid_l2890_289007

def Grid := Fin 5 → Fin 6 → Bool

def hasUniformSubgrid (g : Grid) : Prop :=
  ∃ (i j : Fin 4), 
    (g i j = g (i + 1) j ∧ 
     g i j = g i (j + 1) ∧ 
     g i j = g (i + 1) (j + 1))

theorem all_grids_have_uniform_subgrid :
  ∀ (g : Grid), hasUniformSubgrid g :=
sorry

end NUMINAMATH_CALUDE_all_grids_have_uniform_subgrid_l2890_289007


namespace NUMINAMATH_CALUDE_stratified_sampling_example_l2890_289088

/-- Represents a population divided into two strata --/
structure Population :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Represents a sample taken from a population --/
structure Sample :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Defines a stratified sampling method --/
def is_stratified_sampling (pop : Population) (samp : Sample) : Prop :=
  (pop.male_count : ℚ) / (pop.male_count + pop.female_count) =
  (samp.male_count : ℚ) / (samp.male_count + samp.female_count)

/-- The theorem to be proved --/
theorem stratified_sampling_example :
  let pop := Population.mk 500 400
  let samp := Sample.mk 25 20
  is_stratified_sampling pop samp :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_example_l2890_289088


namespace NUMINAMATH_CALUDE_fraction_calculation_l2890_289090

theorem fraction_calculation : 
  (8 / 17) / (7 / 5) + (5 / 7) * (9 / 17) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l2890_289090


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_of_zeros_l2890_289010

/-- Represents a parabola and its transformations -/
structure Parabola where
  a : ℝ  -- coefficient of x^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Apply transformations to the parabola -/
def transform (p : Parabola) : Parabola :=
  { a := -p.a,  -- 180-degree rotation
    h := p.h + 4,  -- 4 units right shift
    k := p.k + 4 }  -- 4 units up shift

/-- Calculate the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_sum_of_zeros :
  let original := Parabola.mk 1 2 3
  let transformed := transform original
  sumOfZeros transformed = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_sum_of_zeros_l2890_289010


namespace NUMINAMATH_CALUDE_function_equality_implies_m_zero_l2890_289027

/-- Given two functions f and g, prove that m = 0 when 3f(3) = g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + m
  let g := fun (x : ℝ) => x^2 - 3*x + 5*m
  3 * f 3 = g 3 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_zero_l2890_289027


namespace NUMINAMATH_CALUDE_concert_ticket_price_l2890_289096

/-- The price of each ticket in dollars -/
def ticket_price : ℚ := 4

/-- The total number of tickets bought -/
def total_tickets : ℕ := 8

/-- The total amount spent in dollars -/
def total_spent : ℚ := 32

theorem concert_ticket_price : 
  ticket_price * total_tickets = total_spent :=
sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l2890_289096


namespace NUMINAMATH_CALUDE_inequality_count_l2890_289009

theorem inequality_count (x y a b : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_x_lt_a : x < a) (h_y_lt_b : y < b) : 
  (((x + y < a + b) ∧ (x * y < a * b) ∧ (x / y < a / b)) ∧ 
   ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x < a → y < b → x - y < a - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_count_l2890_289009


namespace NUMINAMATH_CALUDE_volume_formula_correct_l2890_289014

/-- A solid formed by the union of a sphere and a truncated cone -/
structure SphereConeUnion where
  R : ℝ  -- radius of the sphere
  S : ℝ  -- total surface area of the solid

/-- The sphere is tangent to one base of the truncated cone -/
axiom sphere_tangent_base (solid : SphereConeUnion) : True

/-- The sphere is tangent to the lateral surface of the cone along a circle -/
axiom sphere_tangent_lateral (solid : SphereConeUnion) : True

/-- The circle of tangency coincides with the other base of the cone -/
axiom tangency_coincides_base (solid : SphereConeUnion) : True

/-- The volume of the solid formed by the union of the cone and the sphere -/
noncomputable def volume (solid : SphereConeUnion) : ℝ :=
  (1 / 3) * solid.S * solid.R

/-- Theorem stating that the volume formula is correct -/
theorem volume_formula_correct (solid : SphereConeUnion) :
  volume solid = (1 / 3) * solid.S * solid.R := by sorry

end NUMINAMATH_CALUDE_volume_formula_correct_l2890_289014


namespace NUMINAMATH_CALUDE_line_segment_representation_l2890_289043

/-- Represents the scale factor of the drawing -/
def scale_factor : ℝ := 800

/-- Represents the length of the line segment in the drawing (in inches) -/
def line_segment_length : ℝ := 4.75

/-- Calculates the actual length in feet represented by a given length in the drawing -/
def actual_length (drawing_length : ℝ) : ℝ := drawing_length * scale_factor

/-- Theorem stating that a 4.75-inch line segment on the scale drawing represents 3800 feet -/
theorem line_segment_representation : 
  actual_length line_segment_length = 3800 := by sorry

end NUMINAMATH_CALUDE_line_segment_representation_l2890_289043


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2890_289012

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 7 and a₇ = 3, prove that a₁₀ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 7) 
  (h_a7 : a 7 = 3) : 
  a 10 = 0 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2890_289012


namespace NUMINAMATH_CALUDE_homework_time_distribution_l2890_289003

theorem homework_time_distribution (total_time : ℕ) (math_percent : ℚ) (science_percent : ℚ) 
  (h1 : total_time = 150)
  (h2 : math_percent = 30 / 100)
  (h3 : science_percent = 40 / 100) :
  total_time - (math_percent * total_time + science_percent * total_time) = 45 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_distribution_l2890_289003


namespace NUMINAMATH_CALUDE_cubic_function_coefficient_l2890_289074

/-- Given a cubic function f(x) = ax³ - 2x that passes through the point (-1, 4),
    prove that the coefficient a equals -2. -/
theorem cubic_function_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2 * x) (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_coefficient_l2890_289074


namespace NUMINAMATH_CALUDE_monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l2890_289041

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement for part 1
theorem monotonicity_when_a_is_neg_one :
  let f₁ := f (-1)
  ∀ x y, x < y →
    (x ≤ 1/2 → y ≤ 1/2 → f₁ y ≤ f₁ x) ∧
    (1/2 ≤ x → 1/2 ≤ y → f₁ x ≤ f₁ y) :=
sorry

-- Statement for part 2
theorem monotonicity_condition_on_interval :
  ∀ a : ℝ, (∀ x y, -5 ≤ x → x < y → y ≤ 5 → 
    (f a x < f a y ∨ f a y < f a x)) ↔ 
    (a < -10 ∨ a > 10) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_when_a_is_neg_one_monotonicity_condition_on_interval_l2890_289041


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2890_289048

theorem purely_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 3*x + 2) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2890_289048


namespace NUMINAMATH_CALUDE_complex_product_simplification_l2890_289036

theorem complex_product_simplification (x y : ℝ) :
  let i := Complex.I
  (x + i * y + 1) * (x - i * y + 1) = (x + 1)^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l2890_289036


namespace NUMINAMATH_CALUDE_min_female_participants_l2890_289000

theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = (male_students + female_students) * 60 / 100) :
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by
  sorry

end NUMINAMATH_CALUDE_min_female_participants_l2890_289000


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2890_289004

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2890_289004


namespace NUMINAMATH_CALUDE_altitude_segment_length_l2890_289065

/-- An acute triangle with two altitudes dividing two sides -/
structure AcuteTriangleWithAltitudes where
  -- Sides
  AC : ℝ
  BC : ℝ
  -- Segments created by altitudes
  AD : ℝ
  DC : ℝ
  CE : ℝ
  EB : ℝ
  -- Conditions
  acute : AC > 0 ∧ BC > 0  -- Simplification for acute triangle
  altitude_division : AD + DC = AC ∧ CE + EB = BC
  given_lengths : AD = 6 ∧ DC = 4 ∧ CE = 3

/-- The theorem stating that y (EB) equals 11/3 -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.EB = 11/3 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l2890_289065


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2890_289008

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 5 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 5 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2890_289008


namespace NUMINAMATH_CALUDE_semicircle_radius_l2890_289026

theorem semicircle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_area : π * a^2 / 8 = 12.5 * π) (h_arc : π * b / 2 = 11 * π) :
  c / 2 = Real.sqrt 584 / 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l2890_289026


namespace NUMINAMATH_CALUDE_runners_meeting_time_l2890_289023

/-- The time (in seconds) after which two runners meet at the starting point -/
def meetingTime (p_time q_time : ℕ) : ℕ :=
  Nat.lcm p_time q_time

/-- Theorem stating that two runners with given lap times meet after a specific time -/
theorem runners_meeting_time :
  meetingTime 252 198 = 2772 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l2890_289023


namespace NUMINAMATH_CALUDE_rectangleChoicesIs54_l2890_289058

/-- The number of ways to choose 4 lines (2 horizontal and 2 vertical) from 5 horizontal
    and 4 vertical lines to form a rectangle, without selecting the first and fifth
    horizontal lines together. -/
def rectangleChoices : ℕ := by
  -- Define the number of horizontal and vertical lines
  let horizontalLines : ℕ := 5
  let verticalLines : ℕ := 4

  -- Calculate the total number of ways to choose 2 horizontal lines
  let totalHorizontalChoices : ℕ := Nat.choose horizontalLines 2

  -- Calculate the number of choices that include both first and fifth horizontal lines
  let invalidHorizontalChoices : ℕ := 1

  -- Calculate the number of valid horizontal line choices
  let validHorizontalChoices : ℕ := totalHorizontalChoices - invalidHorizontalChoices

  -- Calculate the number of ways to choose 2 vertical lines
  let verticalChoices : ℕ := Nat.choose verticalLines 2

  -- Calculate the total number of valid choices
  exact validHorizontalChoices * verticalChoices

/-- Theorem stating that the number of valid rectangle choices is 54 -/
theorem rectangleChoicesIs54 : rectangleChoices = 54 := by sorry

end NUMINAMATH_CALUDE_rectangleChoicesIs54_l2890_289058


namespace NUMINAMATH_CALUDE_general_term_formula_l2890_289095

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℝ := 3^n - 1

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

/-- Theorem stating that the given general term formula is correct -/
theorem general_term_formula (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n - 1) := by sorry

end NUMINAMATH_CALUDE_general_term_formula_l2890_289095


namespace NUMINAMATH_CALUDE_maurice_current_age_l2890_289061

/-- Given Ron's current age and the relation between Ron and Maurice's ages after 5 years,
    prove Maurice's current age. -/
theorem maurice_current_age :
  ∀ (ron_current_age : ℕ) (maurice_current_age : ℕ),
    ron_current_age = 43 →
    ron_current_age + 5 = 4 * (maurice_current_age + 5) →
    maurice_current_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_maurice_current_age_l2890_289061


namespace NUMINAMATH_CALUDE_work_done_stretching_spring_l2890_289034

/-- Work done by stretching a spring -/
theorem work_done_stretching_spring
  (force : ℝ) (compression : ℝ) (stretch : ℝ)
  (hf : force = 10)
  (hc : compression = 0.1)
  (hs : stretch = 0.06)
  : (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_work_done_stretching_spring_l2890_289034


namespace NUMINAMATH_CALUDE_max_value_plus_cos_squared_l2890_289018

theorem max_value_plus_cos_squared (x : ℝ) (M : ℝ) : 
  0 ≤ x → x ≤ π / 2 → 
  (∀ y, 0 ≤ y ∧ y ≤ π / 2 → 
    3 * Real.sin y ^ 2 + 8 * Real.sin y * Real.cos y + 9 * Real.cos y ^ 2 ≤ M) →
  (3 * Real.sin x ^ 2 + 8 * Real.sin x * Real.cos x + 9 * Real.cos x ^ 2 = M) →
  M + 100 * Real.cos x ^ 2 = 91 := by
sorry

end NUMINAMATH_CALUDE_max_value_plus_cos_squared_l2890_289018


namespace NUMINAMATH_CALUDE_sum_f_positive_l2890_289037

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l2890_289037
