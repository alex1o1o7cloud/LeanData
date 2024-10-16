import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l678_67871

theorem quadratic_equation_solution :
  ∀ x : ℝ, x * (x - 3) = 0 ↔ x = 0 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l678_67871


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l678_67802

theorem sugar_consumption_reduction (initial_price new_price : ℚ) 
  (h1 : initial_price = 10)
  (h2 : new_price = 13) :
  let reduction_percentage := (1 - initial_price / new_price) * 100
  reduction_percentage = 300 / 13 := by
sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l678_67802


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l678_67888

theorem two_digit_numbers_problem (A B : ℕ) : 
  A ≥ 10 ∧ A ≤ 99 ∧ B ≥ 10 ∧ B ≤ 99 →
  (100 * A + B) / B = 121 →
  (100 * B + A) / A = 84 ∧ (100 * B + A) % A = 14 →
  A = 42 ∧ B = 35 := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l678_67888


namespace NUMINAMATH_CALUDE_lisa_walking_distance_l678_67853

/-- Lisa's walking problem -/
theorem lisa_walking_distance
  (walking_speed : ℕ)  -- Lisa's walking speed in meters per minute
  (daily_duration : ℕ)  -- Lisa's daily walking duration in minutes
  (days : ℕ)  -- Number of days
  (h1 : walking_speed = 10)  -- Lisa walks 10 meters each minute
  (h2 : daily_duration = 60)  -- Lisa walks for an hour (60 minutes) every day
  (h3 : days = 2)  -- We're considering two days
  : walking_speed * daily_duration * days = 1200 :=
by
  sorry

#check lisa_walking_distance

end NUMINAMATH_CALUDE_lisa_walking_distance_l678_67853


namespace NUMINAMATH_CALUDE_legs_per_chair_correct_l678_67812

/-- The number of legs per office chair in Kenzo's company -/
def legs_per_chair : ℕ := 5

/-- The initial number of office chairs -/
def initial_chairs : ℕ := 80

/-- The number of round tables -/
def round_tables : ℕ := 20

/-- The number of legs per round table -/
def legs_per_table : ℕ := 3

/-- The percentage of chairs that remain after damage (as a rational number) -/
def remaining_chair_ratio : ℚ := 3/5

/-- The total number of furniture legs remaining after disposal -/
def total_remaining_legs : ℕ := 300

/-- Theorem stating that the number of legs per chair is correct given the conditions -/
theorem legs_per_chair_correct : 
  (remaining_chair_ratio * initial_chairs : ℚ).num * legs_per_chair + 
  round_tables * legs_per_table = total_remaining_legs :=
by sorry

end NUMINAMATH_CALUDE_legs_per_chair_correct_l678_67812


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l678_67823

/-- Sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Main sequence -/
def a (n : ℕ) : ℝ := 2*n - 1

/-- Arithmetic subsequence -/
def isArithmeticSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric subsequence -/
def isGeometricSubsequence (f : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, f (n + 1) = q * f n

/-- Arithmetic sequence -/
def isArithmeticSequence (f : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, f (n + 1) - f n = d

/-- Geometric sequence -/
def isGeometricSequence (f : ℕ → ℝ) (q : ℝ) : Prop :=
  q ≠ 1 ∧ ∀ n : ℕ, f (n + 1) = q * f n

theorem problem_1 : isArithmeticSubsequence (λ n => a (3*n)) := by sorry

theorem problem_2 (a : ℕ → ℤ) (d : ℤ) (h1 : d ≠ 0) (h2 : isArithmeticSequence a d) 
  (h3 : a 5 = 6) (h4 : isGeometricSubsequence (λ n => a (2*n + 1))) :
  ∃ n1 : ℕ, n1 ∈ ({6, 8, 11} : Set ℕ) := by sorry

theorem problem_3 (a : ℕ → ℝ) (q : ℝ) (h1 : isGeometricSequence a q) 
  (h2 : ∃ f : ℕ → ℕ, Infinite {n : ℕ | n ∈ Set.range f} ∧ isArithmeticSubsequence (λ n => a (f n))) :
  q = -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l678_67823


namespace NUMINAMATH_CALUDE_nanguo_pear_profit_l678_67881

/-- Represents the weight difference from standard and the number of boxes for each difference -/
structure WeightDifference :=
  (difference : ℚ)
  (numBoxes : ℕ)

/-- Calculates the total profit from selling Nanguo pears -/
def calculateProfit (
  numBoxes : ℕ)
  (standardWeight : ℚ)
  (weightDifferences : List WeightDifference)
  (purchasePrice : ℚ)
  (highSellPrice : ℚ)
  (lowSellPrice : ℚ)
  (highSellProportion : ℚ) : ℚ :=
  sorry

theorem nanguo_pear_profit :
  let numBoxes : ℕ := 50
  let standardWeight : ℚ := 10
  let weightDifferences : List WeightDifference := [
    ⟨-2/10, 12⟩, ⟨-1/10, 3⟩, ⟨0, 3⟩, ⟨1/10, 7⟩, ⟨2/10, 15⟩, ⟨3/10, 10⟩
  ]
  let purchasePrice : ℚ := 4
  let highSellPrice : ℚ := 10
  let lowSellPrice : ℚ := 3/2
  let highSellProportion : ℚ := 3/5
  calculateProfit numBoxes standardWeight weightDifferences purchasePrice highSellPrice lowSellPrice highSellProportion = 27216/10
  := by sorry

end NUMINAMATH_CALUDE_nanguo_pear_profit_l678_67881


namespace NUMINAMATH_CALUDE_inequality_solution_l678_67850

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l678_67850


namespace NUMINAMATH_CALUDE_youngest_child_age_l678_67832

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem youngest_child_age (children : Fin 6 → ℕ) : 
  (∀ i : Fin 6, is_prime (children i)) →
  (∃ y : ℕ, children 0 = y ∧ 
            children 1 = y + 2 ∧
            children 2 = y + 6 ∧
            children 3 = y + 8 ∧
            children 4 = y + 12 ∧
            children 5 = y + 14) →
  children 0 = 5 :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l678_67832


namespace NUMINAMATH_CALUDE_parking_lot_bikes_l678_67842

/-- The number of bikes in a parking lot with cars and bikes. -/
def numBikes (numCars : ℕ) (totalWheels : ℕ) (wheelsPerCar : ℕ) (wheelsPerBike : ℕ) : ℕ :=
  (totalWheels - numCars * wheelsPerCar) / wheelsPerBike

theorem parking_lot_bikes :
  numBikes 14 76 4 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_bikes_l678_67842


namespace NUMINAMATH_CALUDE_sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l678_67886

/-- The molecular weight of 3 moles of Na2SO4·10H2O -/
theorem sodium_sulfate_decahydrate_weight : ℝ → ℝ → ℝ → ℝ → ℝ := 
  fun (na_weight : ℝ) (s_weight : ℝ) (o_weight : ℝ) (h_weight : ℝ) =>
  let mw := 2 * na_weight + s_weight + 14 * o_weight + 20 * h_weight
  3 * mw

/-- The molecular weight of 3 moles of Na2SO4·10H2O is 966.75 grams -/
theorem sodium_sulfate_decahydrate_weight_is_966_75 :
  sodium_sulfate_decahydrate_weight 22.99 32.07 16.00 1.01 = 966.75 := by
  sorry

end NUMINAMATH_CALUDE_sodium_sulfate_decahydrate_weight_sodium_sulfate_decahydrate_weight_is_966_75_l678_67886


namespace NUMINAMATH_CALUDE_power_product_equality_l678_67834

theorem power_product_equality (a : ℝ) : a * a^2 * (-a)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l678_67834


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l678_67861

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l678_67861


namespace NUMINAMATH_CALUDE_x_condition_l678_67883

theorem x_condition (x : ℝ) : |x - 1| + |x - 5| = 4 → 1 ≤ x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_condition_l678_67883


namespace NUMINAMATH_CALUDE_class_size_l678_67804

theorem class_size (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 3) : n = 21 ∨ n = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l678_67804


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l678_67844

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ (k ≥ -5 ∧ k ≠ -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l678_67844


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l678_67854

/-- A function f that represents the inequality to be proven -/
noncomputable def f (a b : ℝ) : ℝ := sorry

/-- Theorem stating the inequality and equality conditions -/
theorem inequality_and_equality_conditions (a b : ℝ) (ha : a ≥ 3) (hb : b ≥ 3) :
  f a b ≥ 0 ∧ (f a b = 0 ↔ (a = 3 ∧ b ≥ 3) ∨ (b = 3 ∧ a ≥ 3)) := by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l678_67854


namespace NUMINAMATH_CALUDE_sin_cos_sum_10_50_l678_67820

theorem sin_cos_sum_10_50 : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (50 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_10_50_l678_67820


namespace NUMINAMATH_CALUDE_power_six_2045_mod_13_l678_67833

theorem power_six_2045_mod_13 : 6^2045 ≡ 2 [ZMOD 13] := by sorry

end NUMINAMATH_CALUDE_power_six_2045_mod_13_l678_67833


namespace NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l678_67841

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) := by
sorry

end NUMINAMATH_CALUDE_negation_of_existential_absolute_value_l678_67841


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l678_67894

theorem simplify_sqrt_expression (t : ℝ) : 
  Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l678_67894


namespace NUMINAMATH_CALUDE_not_all_greater_than_one_l678_67896

theorem not_all_greater_than_one (a b c : Real) 
  (ha : 0 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 2) 
  (hc : 0 < c ∧ c < 2) : 
  ¬((2 - a) * b > 1 ∧ (2 - b) * c > 1 ∧ (2 - c) * a > 1) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_one_l678_67896


namespace NUMINAMATH_CALUDE_square_area_on_xz_l678_67829

/-- A right-angled triangle with squares on each side -/
structure RightTriangleWithSquares where
  /-- Length of side XZ -/
  x : ℝ
  /-- The sum of areas of squares on all sides is 500 -/
  area_sum : x^2 / 2 + x^2 + 5 * x^2 / 4 = 500

/-- The area of the square on side XZ is 2000/11 -/
theorem square_area_on_xz (t : RightTriangleWithSquares) : t.x^2 = 2000 / 11 := by
  sorry

#check square_area_on_xz

end NUMINAMATH_CALUDE_square_area_on_xz_l678_67829


namespace NUMINAMATH_CALUDE_sum_in_base5_l678_67806

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem sum_in_base5 :
  toBase5 (toDecimal [2, 1, 3] + toDecimal [3, 2, 4] + toDecimal [1, 4, 1]) = [1, 3, 3, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base5_l678_67806


namespace NUMINAMATH_CALUDE_parallelogram_area_l678_67827

def v : Fin 2 → ℝ
| 0 => 7
| 1 => -5

def w : Fin 2 → ℝ
| 0 => 14
| 1 => -4

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 42 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l678_67827


namespace NUMINAMATH_CALUDE_thomas_weight_vest_cost_l678_67898

/-- Calculates the total cost for Thomas to increase his weight vest weight --/
def calculate_total_cost (initial_weight : ℕ) (increase_percentage : ℕ) (ingot_weight : ℕ) (ingot_cost : ℕ) : ℚ :=
  let additional_weight := initial_weight * increase_percentage / 100
  let num_ingots := (additional_weight + ingot_weight - 1) / ingot_weight
  let base_cost := num_ingots * ingot_cost
  let discounted_cost := 
    if num_ingots ≤ 10 then base_cost
    else if num_ingots ≤ 20 then base_cost * 80 / 100
    else if num_ingots ≤ 30 then base_cost * 75 / 100
    else base_cost * 70 / 100
  let taxed_cost :=
    if num_ingots ≤ 20 then discounted_cost * 105 / 100
    else if num_ingots ≤ 30 then discounted_cost * 103 / 100
    else discounted_cost * 101 / 100
  let shipping_fee :=
    if num_ingots * ingot_weight ≤ 20 then 10
    else if num_ingots * ingot_weight ≤ 40 then 15
    else 20
  taxed_cost + shipping_fee

/-- Theorem stating that the total cost for Thomas is $90.60 --/
theorem thomas_weight_vest_cost :
  calculate_total_cost 60 60 2 5 = 9060 / 100 :=
sorry

end NUMINAMATH_CALUDE_thomas_weight_vest_cost_l678_67898


namespace NUMINAMATH_CALUDE_gloria_cypress_trees_l678_67864

def cabin_price : ℕ := 129000
def initial_cash : ℕ := 150
def final_cash : ℕ := 350
def pine_trees : ℕ := 600
def maple_trees : ℕ := 24
def pine_price : ℕ := 200
def maple_price : ℕ := 300
def cypress_price : ℕ := 100

theorem gloria_cypress_trees :
  ∃ (cypress_trees : ℕ),
    cypress_trees * cypress_price + 
    pine_trees * pine_price + 
    maple_trees * maple_price = 
    cabin_price + final_cash - initial_cash ∧
    cypress_trees = 20 := by
  sorry

end NUMINAMATH_CALUDE_gloria_cypress_trees_l678_67864


namespace NUMINAMATH_CALUDE_correct_result_l678_67889

def add_subtract_round (a b c : ℕ) : ℕ :=
  let sum := a + b - c
  (sum + 5) / 10 * 10

theorem correct_result : add_subtract_round 53 28 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l678_67889


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l678_67856

-- Define the function f(x) = x³ + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0 ∨ p.1 = -1 ∧ p.2 = -4} =
  {p : ℝ × ℝ | f p.1 = p.2 ∧ f' p.1 = 4} :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l678_67856


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l678_67814

theorem triangle_angle_sum (a b c : ℝ) (h1 : b = 30)
    (h2 : c = 3 * b) : a = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l678_67814


namespace NUMINAMATH_CALUDE_shortest_distance_is_one_l678_67859

/-- Curve C₁ parameterized by θ -/
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Curve C₂ parameterized by t -/
noncomputable def C₂ (t : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Distance function between points on C₁ and C₂ -/
noncomputable def D (θ t : ℝ) : ℝ :=
  let (x₁, y₁, z₁) := C₁ θ
  let (x₂, y₂, z₂) := C₂ t
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)

/-- The shortest distance between C₁ and C₂ is 1 -/
theorem shortest_distance_is_one : ∃ θ₀ t₀, ∀ θ t, D θ₀ t₀ ≤ D θ t ∧ D θ₀ t₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_is_one_l678_67859


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l678_67803

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_2_to_20 :
  arithmetic_sequence_sum 2 2 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_2_to_20_l678_67803


namespace NUMINAMATH_CALUDE_power_equation_l678_67800

theorem power_equation (k m n : ℕ) 
  (h1 : 3^(k - 1) = 81) 
  (h2 : 4^(m + 2) = 256) 
  (h3 : 5^(n - 3) = 625) : 
  2^(4*k - 3*m + 5*n) = 2^49 := by
sorry

end NUMINAMATH_CALUDE_power_equation_l678_67800


namespace NUMINAMATH_CALUDE_last_digit_is_two_l678_67810

/-- Represents a 2000-digit integer as a list of natural numbers -/
def LongInteger := List Nat

/-- Checks if two consecutive digits are divisible by 17 or 23 -/
def validPair (a b : Nat) : Prop := (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

/-- Defines the properties of our specific 2000-digit integer -/
def SpecialInteger (n : LongInteger) : Prop :=
  n.length = 2000 ∧
  n.head? = some 3 ∧
  ∀ i, i < 1999 → validPair (n.get! i) (n.get! (i + 1))

theorem last_digit_is_two (n : LongInteger) (h : SpecialInteger n) : 
  n.getLast? = some 2 := by
  sorry

#check last_digit_is_two

end NUMINAMATH_CALUDE_last_digit_is_two_l678_67810


namespace NUMINAMATH_CALUDE_consecutive_sum_largest_l678_67884

theorem consecutive_sum_largest (n : ℕ) : 
  (n + (n+1) + (n+2) + (n+3) + (n+4) = 180) → (n+4 = 38) :=
by
  sorry

#check consecutive_sum_largest

end NUMINAMATH_CALUDE_consecutive_sum_largest_l678_67884


namespace NUMINAMATH_CALUDE_cost_price_calculation_l678_67801

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 21000 →
  discount_rate = 0.10 →
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 17500 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l678_67801


namespace NUMINAMATH_CALUDE_fraction_simplification_l678_67852

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

theorem fraction_simplification : (numerator : ℚ) / denominator = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l678_67852


namespace NUMINAMATH_CALUDE_exists_valid_set_l678_67863

def is_valid_set (S : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (n ∈ S ↔ 
    (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = n) ∨
    (∃ a b : ℕ, a ∉ S ∧ b ∉ S ∧ a ≠ b ∧ a > 0 ∧ b > 0 ∧ a + b = n))

theorem exists_valid_set : ∃ S : Set ℕ, is_valid_set S := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_set_l678_67863


namespace NUMINAMATH_CALUDE_aarti_work_completion_time_l678_67815

/-- If Aarti can complete three times a piece of work in 24 days, 
    then she can complete one piece of work in 8 days. -/
theorem aarti_work_completion_time : 
  ∀ (work_time : ℝ), work_time > 0 → 3 * work_time = 24 → work_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_time_l678_67815


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l678_67837

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem stating the total number of houses after the housing boom -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l678_67837


namespace NUMINAMATH_CALUDE_xy_value_l678_67818

theorem xy_value (x y : ℝ) (h : x * (x + 2*y) = x^2 + 12) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l678_67818


namespace NUMINAMATH_CALUDE_max_expected_games_max_at_half_l678_67890

/-- The expected number of games in a best-of-five series -/
def f (p : ℝ) : ℝ := 6 * p^4 - 12 * p^3 + 3 * p^2 + 3 * p + 3

/-- The theorem stating the maximum value of f(p) -/
theorem max_expected_games :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → f p ≤ 33/8 :=
by
  sorry

/-- The theorem stating that the maximum is achieved at p = 1/2 -/
theorem max_at_half :
  f (1/2) = 33/8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expected_games_max_at_half_l678_67890


namespace NUMINAMATH_CALUDE_base_prime_representation_of_140_l678_67838

/-- Base prime representation of a natural number -/
def BasePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Check if a list represents the correct base prime representation of a number -/
def IsCorrectBasePrimeRepresentation (n : ℕ) (l : List ℕ) : Prop :=
  BasePrimeRepresentation n = l

theorem base_prime_representation_of_140 :
  IsCorrectBasePrimeRepresentation 140 [2, 1, 0, 2] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_representation_of_140_l678_67838


namespace NUMINAMATH_CALUDE_other_root_is_one_l678_67824

theorem other_root_is_one (a : ℝ) : 
  (2^2 - a*2 + 2 = 0) → 
  ∃ x, x ≠ 2 ∧ x^2 - a*x + 2 = 0 ∧ x = 1 := by
sorry

end NUMINAMATH_CALUDE_other_root_is_one_l678_67824


namespace NUMINAMATH_CALUDE_f_5_equals_56_l678_67860

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_5_equals_56 :
  f 5 = horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 ∧
  horner_eval [2, -9, 5, -49, -5, 2, 1, 1] 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_56_l678_67860


namespace NUMINAMATH_CALUDE_baseball_cards_difference_l678_67899

theorem baseball_cards_difference (jorge matias carlos : ℕ) : 
  jorge = matias → 
  carlos = 20 → 
  jorge + matias + carlos = 48 → 
  carlos - matias = 6 := by
sorry

end NUMINAMATH_CALUDE_baseball_cards_difference_l678_67899


namespace NUMINAMATH_CALUDE_ellipse_parameter_sum_l678_67826

-- Define the ellipse parameters
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (8, 0)
def distance_sum : ℝ := 10

-- Define the ellipse equation
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  ∃ (h k a b : ℝ), 
    (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ∧
    (x - F₁.1)^2 + (y - F₁.2)^2 + (x - F₂.1)^2 + (y - F₂.2)^2 = distance_sum^2

-- Theorem statement
theorem ellipse_parameter_sum :
  ∃ (h k a b : ℝ),
    (∀ P, is_on_ellipse P → 
      (P.1 - h)^2 / a^2 + (P.2 - k)^2 / b^2 = 1) ∧
    h + k + a + b = 12 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameter_sum_l678_67826


namespace NUMINAMATH_CALUDE_second_question_correct_percentage_l678_67819

theorem second_question_correct_percentage
  (first_correct : Real)
  (neither_correct : Real)
  (both_correct : Real)
  (h1 : first_correct = 0.63)
  (h2 : neither_correct = 0.20)
  (h3 : both_correct = 0.32) :
  ∃ (second_correct : Real),
    second_correct = 0.49 ∧
    first_correct + second_correct - both_correct = 1 - neither_correct :=
by sorry

end NUMINAMATH_CALUDE_second_question_correct_percentage_l678_67819


namespace NUMINAMATH_CALUDE_sum_of_xyz_l678_67878

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l678_67878


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l678_67825

theorem divisibility_by_twelve (n : Nat) : n < 10 → (3150 + n) % 12 = 0 ↔ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l678_67825


namespace NUMINAMATH_CALUDE_decreasing_even_shifted_function_property_l678_67868

/-- A function that is decreasing on (8, +∞) and f(x+8) is even -/
def DecreasingEvenShiftedFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > 8 ∧ y > 8 ∧ x > y → f x < f y) ∧
  (∀ x, f (x + 8) = f (-x + 8))

theorem decreasing_even_shifted_function_property
  (f : ℝ → ℝ) (h : DecreasingEvenShiftedFunction f) :
  f 7 > f 10 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_even_shifted_function_property_l678_67868


namespace NUMINAMATH_CALUDE_optimal_allocation_l678_67874

/-- Represents the farming problem with given conditions -/
structure FarmingProblem where
  totalLand : ℝ
  riceYield : ℝ
  peanutYield : ℝ
  riceCost : ℝ
  peanutCost : ℝ
  ricePrice : ℝ
  peanutPrice : ℝ
  availableInvestment : ℝ

/-- Calculates the profit for a given allocation of land -/
def profit (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : ℝ :=
  (p.ricePrice * p.riceYield - p.riceCost) * riceLand +
  (p.peanutPrice * p.peanutYield - p.peanutCost) * peanutLand

/-- Checks if a land allocation is valid according to the problem constraints -/
def isValidAllocation (p : FarmingProblem) (riceLand : ℝ) (peanutLand : ℝ) : Prop :=
  riceLand ≥ 0 ∧ peanutLand ≥ 0 ∧
  riceLand + peanutLand ≤ p.totalLand ∧
  p.riceCost * riceLand + p.peanutCost * peanutLand ≤ p.availableInvestment

/-- The main theorem stating that the given allocation maximizes profit -/
theorem optimal_allocation (p : FarmingProblem) 
  (h : p = {
    totalLand := 2,
    riceYield := 6000,
    peanutYield := 1500,
    riceCost := 3600,
    peanutCost := 1200,
    ricePrice := 3,
    peanutPrice := 5,
    availableInvestment := 6000
  }) :
  ∀ x y, isValidAllocation p x y → profit p x y ≤ profit p (3/2) (1/2) :=
sorry


end NUMINAMATH_CALUDE_optimal_allocation_l678_67874


namespace NUMINAMATH_CALUDE_bakery_theft_l678_67845

/-- The number of breads remaining after a thief takes their share -/
def breads_after_thief (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => (breads_after_thief initial n - 1) / 2

/-- The proposition that given 5 thieves and 3 breads remaining at the end, 
    the initial number of breads was 127 -/
theorem bakery_theft (initial : ℕ) :
  breads_after_thief initial 5 = 3 → initial = 127 := by
  sorry

#check bakery_theft

end NUMINAMATH_CALUDE_bakery_theft_l678_67845


namespace NUMINAMATH_CALUDE_new_home_library_capacity_l678_67882

theorem new_home_library_capacity 
  (M : ℚ) -- Millicent's total number of books
  (H : ℚ) -- Harold's total number of books
  (harold_ratio : H = (1/2) * M) -- Harold has 1/2 as many books as Millicent
  (harold_brings : ℚ) -- Number of books Harold brings
  (millicent_brings : ℚ) -- Number of books Millicent brings
  (harold_brings_def : harold_brings = (1/3) * H) -- Harold brings 1/3 of his books
  (millicent_brings_def : millicent_brings = (1/2) * M) -- Millicent brings 1/2 of her books
  : harold_brings + millicent_brings = (2/3) * M := by
  sorry

end NUMINAMATH_CALUDE_new_home_library_capacity_l678_67882


namespace NUMINAMATH_CALUDE_angle_triple_complement_l678_67879

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l678_67879


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l678_67892

open Real

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (x - 1) / (Real.exp x))
  (h2 : ∀ t ∈ Set.Icc (1/2) 2, f t > t) :
  ∃ a, a > Real.exp 2 + 1/2 ∧ ∀ x ∈ Set.Icc (1/2) 2, (a - 1) / (Real.exp x) > x :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l678_67892


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l678_67809

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x ∈ Set.Iio (-3) ∪ Set.Ioi 1) →
  (∀ x, a*x^2 + b*x - 2 < 0 ↔ x ∈ Set.Ioo (-1/2) 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l678_67809


namespace NUMINAMATH_CALUDE_removed_ball_number_l678_67858

theorem removed_ball_number (n : ℕ) (h1 : n > 0) :
  (n * (n + 1)) / 2 - 5048 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_removed_ball_number_l678_67858


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l678_67839

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l678_67839


namespace NUMINAMATH_CALUDE_monica_cookies_problem_l678_67887

/-- The number of cookies Monica's father ate -/
def father_cookies : ℕ := 6

/-- The total number of cookies Monica made -/
def total_cookies : ℕ := 30

/-- The number of cookies Monica has left -/
def remaining_cookies : ℕ := 8

theorem monica_cookies_problem :
  (∃ (f : ℕ),
    f = father_cookies ∧
    total_cookies = f + (f / 2) + (f / 2 + 2) + remaining_cookies) :=
by
  sorry

end NUMINAMATH_CALUDE_monica_cookies_problem_l678_67887


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l678_67848

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 10 = -2) :
  a 4 * a 7 = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l678_67848


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l678_67813

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the given conditions
variable (h1 : B → A)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem to prove
theorem sufficient_not_necessary : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l678_67813


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_neg_one_l678_67867

/-- Two lines in the plane are perpendicular if and only if the product of their slopes is -1 -/
theorem lines_perpendicular_iff_slope_product_neg_one 
  (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ) (hB₁ : B₁ ≠ 0) (hB₂ : B₂ ≠ 0) :
  (∀ x y : ℝ, A₁ * x + B₁ * y + C₁ = 0 → A₂ * x + B₂ * y + C₂ = 0 → 
    (A₁ * x + B₁ * y + C₁ = 0 ∧ A₂ * x + B₂ * y + C₂ = 0) → 
    (A₁ * A₂) / (B₁ * B₂) = -1) ↔
  (A₁ * A₂) / (B₁ * B₂) = -1 :=
by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_slope_product_neg_one_l678_67867


namespace NUMINAMATH_CALUDE_system_solution_existence_l678_67835

/-- The system of equations has at least one solution for some b iff a ≥ -√2 - 1/4 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 - a ∧ x^2 + y^2 + 8*b^2 = 4*b*(y - x) + 1) ↔ 
  a ≥ -Real.sqrt 2 - 1/4 := by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l678_67835


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l678_67843

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and y-coordinates are equal -/
def symmetric_about_y_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = -x2 ∧ y1 = y2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a + 1) 3 (-2) (b + 2) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l678_67843


namespace NUMINAMATH_CALUDE_right_angle_point_coordinates_l678_67849

/-- Given points A, B, and P, where P is on the y-axis and forms a right angle with AB, 
    prove that P has coordinates (0, -11) -/
theorem right_angle_point_coordinates 
  (A B P : ℝ × ℝ)
  (hA : A = (-3, -2))
  (hB : B = (6, 1))
  (hP_y_axis : P.1 = 0)
  (h_right_angle : (P.2 - A.2) * (B.2 - A.2) = -(P.1 - A.1) * (B.1 - A.1)) :
  P = (0, -11) := by
  sorry

end NUMINAMATH_CALUDE_right_angle_point_coordinates_l678_67849


namespace NUMINAMATH_CALUDE_smaller_number_l678_67870

theorem smaller_number (a b d x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = 2 * a / (3 * b) → x + 2 * y = d →
  min x y = a * d / (2 * a + 3 * b) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_l678_67870


namespace NUMINAMATH_CALUDE_car_wash_goal_proof_l678_67816

def car_wash_goal (families_10 : ℕ) (amount_10 : ℕ) (families_5 : ℕ) (amount_5 : ℕ) (more_needed : ℕ) : Prop :=
  let earned_10 := families_10 * amount_10
  let earned_5 := families_5 * amount_5
  let total_earned := earned_10 + earned_5
  let goal := total_earned + more_needed
  goal = 150

theorem car_wash_goal_proof :
  car_wash_goal 3 10 15 5 45 := by
  sorry

end NUMINAMATH_CALUDE_car_wash_goal_proof_l678_67816


namespace NUMINAMATH_CALUDE_no_nontrivial_solution_for_4n_plus_3_prime_l678_67817

theorem no_nontrivial_solution_for_4n_plus_3_prime (a : ℕ) (x y z : ℤ) :
  Prime a →
  (∃ n : ℕ, a = 4 * n + 3) →
  x^2 + y^2 = a * z^2 →
  x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_nontrivial_solution_for_4n_plus_3_prime_l678_67817


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l678_67875

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_f_at_one :
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l678_67875


namespace NUMINAMATH_CALUDE_inequality_proof_l678_67840

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l678_67840


namespace NUMINAMATH_CALUDE_linear_system_solution_l678_67847

/-- The system of linear equations ax + by = 10 -/
def linear_system (a b : ℝ) (x y : ℝ) : Prop := a * x + b * y = 10

theorem linear_system_solution :
  ∃ (a b : ℝ),
    (linear_system a b 2 4 ∧ linear_system a b 3 1) ∧
    (a = 3 ∧ b = 1) ∧
    (∀ x : ℝ, x > 10 / 3 → linear_system a b x 0 → linear_system a b x y → y < 0) :=
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l678_67847


namespace NUMINAMATH_CALUDE_triangle_base_length_l678_67822

/-- Given a triangle with area 16 m² and height 8 m, prove its base length is 4 m -/
theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 16 → height = 8 → area = (base * height) / 2 → base = 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_base_length_l678_67822


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l678_67805

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 25 →
  a + b = -25 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l678_67805


namespace NUMINAMATH_CALUDE_hyperbola_condition_l678_67869

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (3 - k) + y^2 / (k - 1) = 1

-- Theorem statement
theorem hyperbola_condition (k : ℝ) :
  k > 3 → is_hyperbola k ∧ ¬(∀ k', is_hyperbola k' → k' > 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l678_67869


namespace NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l678_67895

/-- Represents the student population at Greendale College --/
structure StudentPopulation where
  total : ℝ
  likesSwimming : ℝ
  dislikesSwimming : ℝ
  likesSayLike : ℝ
  likesSayDislike : ℝ
  dislikesSayLike : ℝ
  dislikesSayDislike : ℝ

/-- Conditions of the problem --/
def greendaleCollege : StudentPopulation where
  total := 100
  likesSwimming := 70
  dislikesSwimming := 30
  likesSayLike := 0.75 * 70
  likesSayDislike := 0.25 * 70
  dislikesSayLike := 0.15 * 30
  dislikesSayDislike := 0.85 * 30

/-- The main theorem to prove --/
theorem fraction_who_say_dislike_but_like (ε : ℝ) (hε : ε > 0) :
  let totalSayDislike := greendaleCollege.likesSayDislike + greendaleCollege.dislikesSayDislike
  let fraction := greendaleCollege.likesSayDislike / totalSayDislike
  abs (fraction - 0.407) < ε := by
  sorry


end NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l678_67895


namespace NUMINAMATH_CALUDE_integral_x_squared_zero_to_one_l678_67851

theorem integral_x_squared_zero_to_one :
  ∫ x in (0 : ℝ)..(1 : ℝ), x^2 = (1 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_zero_to_one_l678_67851


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l678_67897

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_last_year_l678_67897


namespace NUMINAMATH_CALUDE_f_difference_l678_67891

/-- The function f(x) = x^4 + 3x^3 + 2x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + 2*x^2 + 7*x

/-- Theorem: f(6) - f(-6) = 1380 -/
theorem f_difference : f 6 - f (-6) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l678_67891


namespace NUMINAMATH_CALUDE_average_of_p_and_q_l678_67876

theorem average_of_p_and_q (p q : ℝ) (h : (5 / 4) * (p + q) = 15) : (p + q) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_p_and_q_l678_67876


namespace NUMINAMATH_CALUDE_cakes_sold_l678_67862

/-- Given the initial number of cakes, the remaining number of cakes,
    and the fact that some cakes were sold, prove that the number of cakes sold is 10. -/
theorem cakes_sold (initial_cakes remaining_cakes : ℕ) 
  (h1 : initial_cakes = 149)
  (h2 : remaining_cakes = 139)
  (h3 : remaining_cakes < initial_cakes) :
  initial_cakes - remaining_cakes = 10 := by
  sorry

#check cakes_sold

end NUMINAMATH_CALUDE_cakes_sold_l678_67862


namespace NUMINAMATH_CALUDE_three_digit_numbers_from_five_cards_l678_67873

theorem three_digit_numbers_from_five_cards : 
  let n : ℕ := 5  -- number of cards
  let r : ℕ := 3  -- number of digits in the formed number
  Nat.factorial n / Nat.factorial (n - r) = 60 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_from_five_cards_l678_67873


namespace NUMINAMATH_CALUDE_cube_root_of_64_l678_67836

theorem cube_root_of_64 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 64) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l678_67836


namespace NUMINAMATH_CALUDE_games_last_month_l678_67866

def games_this_month : ℕ := 9
def games_next_month : ℕ := 7
def total_games : ℕ := 24

theorem games_last_month : total_games - (games_this_month + games_next_month) = 8 := by
  sorry

end NUMINAMATH_CALUDE_games_last_month_l678_67866


namespace NUMINAMATH_CALUDE_estimate_red_balls_l678_67808

/-- Represents the result of drawing a ball -/
inductive BallColor
| Red
| White

/-- Represents a bag of balls -/
structure BallBag where
  totalBalls : Nat
  redBalls : Nat
  whiteBalls : Nat
  totalBalls_eq : totalBalls = redBalls + whiteBalls

/-- Represents the result of multiple draws -/
structure DrawResult where
  totalDraws : Nat
  redDraws : Nat
  whiteDraws : Nat
  totalDraws_eq : totalDraws = redDraws + whiteDraws

/-- Theorem stating the estimated number of red balls -/
theorem estimate_red_balls 
  (bag : BallBag) 
  (draws : DrawResult) 
  (h1 : bag.totalBalls = 8) 
  (h2 : draws.totalDraws = 100) 
  (h3 : draws.redDraws = 75) :
  (bag.totalBalls : ℚ) * (draws.redDraws : ℚ) / (draws.totalDraws : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l678_67808


namespace NUMINAMATH_CALUDE_distance_is_49_l678_67857

/-- Represents a sign at a kilometer marker -/
structure Sign :=
  (to_yolkino : Nat)
  (to_palkino : Nat)

/-- Calculates the sum of digits of a natural number -/
def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- The distance between Yolkino and Palkino -/
def distance_yolkino_palkino : Nat := sorry

theorem distance_is_49 :
  ∀ (k : Nat), k < distance_yolkino_palkino →
    ∃ (sign : Sign),
      sign.to_yolkino = k ∧
      sign.to_palkino = distance_yolkino_palkino - k ∧
      digit_sum sign.to_yolkino + digit_sum sign.to_palkino = 13 →
  distance_yolkino_palkino = 49 :=
sorry

end NUMINAMATH_CALUDE_distance_is_49_l678_67857


namespace NUMINAMATH_CALUDE_sin_double_minus_cos_half_squared_l678_67831

theorem sin_double_minus_cos_half_squared 
  (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : Real.sin (Real.pi - α) = 4 / 5) : 
  Real.sin (2 * α) - Real.cos (α / 2) ^ 2 = 4 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_minus_cos_half_squared_l678_67831


namespace NUMINAMATH_CALUDE_uncovered_side_length_l678_67885

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 20 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 80)
  (h_fencing : field.fencing = 28)
  (h_rect_area : field.area = field.length * field.width)
  (h_fencing_sum : field.fencing = 2 * field.width + field.length) :
  field.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_side_length_l678_67885


namespace NUMINAMATH_CALUDE_probability_at_most_one_incorrect_l678_67828

/-- The probability of at most one incorrect result in 10 hemoglobin tests -/
def prob_at_most_one_incorrect (p : ℝ) : ℝ :=
  p^9 * (10 - 9*p)

/-- Theorem: Given the accuracy of a hemoglobin test is p, 
    the probability of at most one incorrect result out of 10 tests 
    is equal to p^9 * (10 - 9p) -/
theorem probability_at_most_one_incorrect 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) : 
  (p^10 + 10 * (1 - p) * p^9) = prob_at_most_one_incorrect p :=
sorry

end NUMINAMATH_CALUDE_probability_at_most_one_incorrect_l678_67828


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l678_67880

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 0.9 * original_price
  let second_reduction := 0.9 * first_reduction
  second_reduction = 0.81 * original_price :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l678_67880


namespace NUMINAMATH_CALUDE_movie_sale_price_is_10000_l678_67807

/-- The sale price of a movie given costs and profit -/
def movie_sale_price (actor_cost food_cost_per_person equipment_cost_multiplier num_people profit : ℕ) : ℕ :=
  let food_cost := food_cost_per_person * num_people
  let equipment_cost := equipment_cost_multiplier * (actor_cost + food_cost)
  let total_cost := actor_cost + food_cost + equipment_cost
  total_cost + profit

/-- Theorem stating the sale price of the movie is $10000 -/
theorem movie_sale_price_is_10000 :
  movie_sale_price 1200 3 2 50 5950 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_movie_sale_price_is_10000_l678_67807


namespace NUMINAMATH_CALUDE_twins_age_today_twins_age_proof_l678_67893

/-- The age of twin brothers today, given that the product of their ages today is smaller by 11 from the product of their ages a year from today. -/
theorem twins_age_today : ℕ :=
  let age_today : ℕ → Prop := fun x => (x + 1) ^ 2 = x ^ 2 + 11
  5

theorem twins_age_proof (age : ℕ) : (age + 1) ^ 2 = age ^ 2 + 11 → age = 5 := by
  sorry

end NUMINAMATH_CALUDE_twins_age_today_twins_age_proof_l678_67893


namespace NUMINAMATH_CALUDE_cream_cake_problem_l678_67830

def creamPerCake : ℕ := 75
def totalCream : ℕ := 500
def totalCakes : ℕ := 50
def cakesPerBox : ℕ := 6

theorem cream_cake_problem :
  (totalCream / creamPerCake : ℕ) = 6 ∧
  (totalCakes + cakesPerBox - 1) / cakesPerBox = 9 := by
  sorry

end NUMINAMATH_CALUDE_cream_cake_problem_l678_67830


namespace NUMINAMATH_CALUDE_garden_length_l678_67846

/-- Proves that a rectangular garden with length twice its width and perimeter 900 yards has a length of 300 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- The length is twice the width
  2 * length + 2 * width = 900 →  -- The perimeter is 900 yards
  length = 300 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l678_67846


namespace NUMINAMATH_CALUDE_exponent_division_l678_67877

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l678_67877


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l678_67855

/-- A function that returns the number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 8 zeros -/
def ends_with_8_zeros (n : ℕ) : Prop := sorry

/-- The theorem to be proved -/
theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    ends_with_8_zeros a ∧
    ends_with_8_zeros b ∧
    num_divisors a = 90 ∧
    num_divisors b = 90 ∧
    a + b = 700000000 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l678_67855


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l678_67872

-- Define the workday duration in minutes
def workday_minutes : ℕ := 10 * 60

-- Define the duration of the first meeting
def first_meeting_duration : ℕ := 30

-- Define the duration of the second meeting
def second_meeting_duration : ℕ := 2 * first_meeting_duration

-- Define the duration of the third meeting
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

-- Define the total time spent in meetings
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

-- Theorem to prove
theorem workday_meeting_percentage : 
  (total_meeting_time : ℚ) / workday_minutes * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l678_67872


namespace NUMINAMATH_CALUDE_min_a_value_l678_67865

theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l678_67865


namespace NUMINAMATH_CALUDE_largest_term_is_a10_l678_67821

def a (n : ℕ) : ℚ := (2 * n - 17) / (2 * n - 19)

theorem largest_term_is_a10 : 
  ∀ k ∈ ({1, 9, 10, 12} : Set ℕ), a 10 ≥ a k :=
by sorry

end NUMINAMATH_CALUDE_largest_term_is_a10_l678_67821


namespace NUMINAMATH_CALUDE_residual_analysis_characteristics_l678_67811

/-- Represents a residual in a statistical model. -/
structure Residual where
  value : ℝ

/-- Represents a statistical analysis method. -/
structure AnalysisMethod where
  name : String
  uses_residuals : Bool
  judges_model_fitting : Bool
  identifies_suspicious_data : Bool

/-- Definition of residual analysis based on its characteristics. -/
def residual_analysis : AnalysisMethod :=
  { name := "residual analysis",
    uses_residuals := true,
    judges_model_fitting := true,
    identifies_suspicious_data := true }

/-- Theorem stating that the analysis method using residuals to judge model fitting
    and identify suspicious data is residual analysis. -/
theorem residual_analysis_characteristics :
  ∀ (method : AnalysisMethod),
    method.uses_residuals ∧
    method.judges_model_fitting ∧
    method.identifies_suspicious_data →
    method = residual_analysis :=
by sorry

end NUMINAMATH_CALUDE_residual_analysis_characteristics_l678_67811
