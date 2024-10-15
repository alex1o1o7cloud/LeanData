import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l4089_408918

def sequence_first_term : ℕ := 103
def sequence_last_term : ℕ := 443
def sequence_common_difference : ℕ := 10

def sequence_length : ℕ := (sequence_last_term - sequence_first_term) / sequence_common_difference + 1

theorem sum_of_integers_ending_in_3 :
  (sequence_length : ℕ) * (sequence_first_term + sequence_last_term) / 2 = 9555 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_integers_ending_in_3_l4089_408918


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_280_l4089_408993

theorem sum_of_two_smallest_prime_factors_of_280 : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ 
  p ∣ 280 ∧ q ∣ 280 ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ 280 → r = p ∨ r ≥ q) ∧
  p + q = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_prime_factors_of_280_l4089_408993


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l4089_408913

def selling_price : ℝ := 900
def profit : ℝ := 100

theorem profit_percentage_calculation :
  (profit / (selling_price - profit)) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l4089_408913


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l4089_408953

theorem smallest_five_digit_divisible_by_53 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_53_l4089_408953


namespace NUMINAMATH_CALUDE_equation_describes_cylinder_l4089_408909

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define a cylinder
def IsCylinder (S : Set CylindricalCoord) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ p : CylindricalCoord, p ∈ S ↔ p.r = c

-- Theorem statement
theorem equation_describes_cylinder (c : ℝ) :
  IsCylinder {p : CylindricalCoord | p.r = c} c :=
by
  sorry

end NUMINAMATH_CALUDE_equation_describes_cylinder_l4089_408909


namespace NUMINAMATH_CALUDE_distance_between_points_l4089_408939

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, 2)
  let pointB : ℝ × ℝ := (5, 7)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4089_408939


namespace NUMINAMATH_CALUDE_ratio_theorem_l4089_408958

theorem ratio_theorem (a b c d r : ℝ) 
  (h1 : (b + c + d) / a = r)
  (h2 : (a + c + d) / b = r)
  (h3 : (a + b + d) / c = r)
  (h4 : (a + b + c) / d = r)
  : r = 3 ∨ r = -1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_theorem_l4089_408958


namespace NUMINAMATH_CALUDE_students_in_canteen_l4089_408942

theorem students_in_canteen (total : ℕ) (absent_fraction : ℚ) (classroom_fraction : ℚ) :
  total = 40 →
  absent_fraction = 1 / 10 →
  classroom_fraction = 3 / 4 →
  (total : ℚ) * (1 - absent_fraction) * (1 - classroom_fraction) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_students_in_canteen_l4089_408942


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l4089_408981

theorem triangle_angle_measurement (D E F : ℝ) : 
  D = 85 →                  -- Measure of ∠D is 85 degrees
  E = 4 * F + 15 →          -- Measure of ∠E is 15 degrees more than four times the measure of ∠F
  D + E + F = 180 →         -- Sum of angles in a triangle is 180 degrees
  F = 16                    -- Measure of ∠F is 16 degrees
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l4089_408981


namespace NUMINAMATH_CALUDE_weight_replacement_l4089_408992

theorem weight_replacement (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 77 := by
  sorry

end NUMINAMATH_CALUDE_weight_replacement_l4089_408992


namespace NUMINAMATH_CALUDE_pattern_1005th_row_l4089_408979

/-- Represents the number of items in the nth row of the pattern -/
def num_items (n : ℕ) : ℕ := n

/-- Represents the sum of items in the nth row of the pattern -/
def sum_items (n : ℕ) : ℕ := n * (n + 1) / 2 + (n - 1) * n / 2

/-- Theorem stating that the 1005th row is the one where the number of items
    and their sum equals 20092 -/
theorem pattern_1005th_row :
  num_items 1005 + sum_items 1005 = 20092 := by sorry

end NUMINAMATH_CALUDE_pattern_1005th_row_l4089_408979


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l4089_408944

theorem modular_inverse_of_5_mod_26 :
  ∃! x : ℕ, x ∈ Finset.range 26 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_26_l4089_408944


namespace NUMINAMATH_CALUDE_equilateral_triangle_solutions_l4089_408959

/-- A complex number z satisfies the equilateral triangle property if 0, z, and z^4 
    form the distinct vertices of an equilateral triangle in the complex plane. -/
def has_equilateral_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ z ≠ z^4 ∧ Complex.abs z = Complex.abs (z^4 - z) ∧ Complex.abs z = Complex.abs z^4

/-- There are exactly two nonzero complex numbers that satisfy 
    the equilateral triangle property. -/
theorem equilateral_triangle_solutions :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, has_equilateral_triangle_property z :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_solutions_l4089_408959


namespace NUMINAMATH_CALUDE_bahs_equal_to_yahs_l4089_408931

/-- The number of bahs equal to 30 rahs -/
def bahs_to_30_rahs : ℕ := 20

/-- The number of rahs equal to 20 yahs -/
def rahs_to_20_yahs : ℕ := 12

/-- The number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1200

/-- The theorem stating the equivalence between bahs and yahs -/
theorem bahs_equal_to_yahs : ∃ (n : ℕ), n * bahs_to_30_rahs * rahs_to_20_yahs = yahs_to_convert * 30 * 20 :=
sorry

end NUMINAMATH_CALUDE_bahs_equal_to_yahs_l4089_408931


namespace NUMINAMATH_CALUDE_total_cost_calculation_l4089_408985

def vegetable_price : ℝ := 2
def beef_price_multiplier : ℝ := 3
def vegetable_weight : ℝ := 6
def beef_weight : ℝ := 4

theorem total_cost_calculation : 
  (vegetable_price * vegetable_weight) + (vegetable_price * beef_price_multiplier * beef_weight) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l4089_408985


namespace NUMINAMATH_CALUDE_divisibility_property_l4089_408910

theorem divisibility_property (m : ℤ) (n : ℕ) :
  (10 ∣ (3^n + m)) → (10 ∣ (3^(n+4) + m)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l4089_408910


namespace NUMINAMATH_CALUDE_prob_win_series_4_1_l4089_408901

/-- Represents the location of a game -/
inductive GameLocation
  | Home
  | Away

/-- Represents the schedule of games for Team A -/
def schedule : List GameLocation :=
  [GameLocation.Home, GameLocation.Home, GameLocation.Away, GameLocation.Away, 
   GameLocation.Home, GameLocation.Away, GameLocation.Home]

/-- Probability of Team A winning a home game -/
def probWinHome : ℝ := 0.6

/-- Probability of Team A winning an away game -/
def probWinAway : ℝ := 0.5

/-- Calculates the probability of Team A winning a game based on its location -/
def probWin (loc : GameLocation) : ℝ :=
  match loc with
  | GameLocation.Home => probWinHome
  | GameLocation.Away => probWinAway

/-- Calculates the probability of a specific game outcome for Team A -/
def probOutcome (outcomes : List Bool) : ℝ :=
  List.zipWith (fun o l => if o then probWin l else 1 - probWin l) outcomes schedule
  |> List.prod

/-- Theorem: The probability of Team A winning the series with a 4:1 score is 0.18 -/
theorem prob_win_series_4_1 : 
  (probOutcome [false, true, true, true, true] +
   probOutcome [true, false, true, true, true] +
   probOutcome [true, true, false, true, true] +
   probOutcome [true, true, true, false, true]) = 0.18 := by
  sorry


end NUMINAMATH_CALUDE_prob_win_series_4_1_l4089_408901


namespace NUMINAMATH_CALUDE_shop_profit_per_tshirt_l4089_408977

/-- The amount the shop makes off each t-shirt -/
def T : ℝ := 25

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 115

/-- The number of t-shirts sold -/
def t_shirts_sold : ℕ := 113

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 78

/-- The price difference between a jersey and a t-shirt -/
def price_difference : ℝ := 90

theorem shop_profit_per_tshirt :
  T = 25 ∧
  jersey_profit = 115 ∧
  t_shirts_sold = 113 ∧
  jerseys_sold = 78 ∧
  jersey_profit = T + price_difference ∧
  price_difference = 90 →
  T = 25 := by sorry

end NUMINAMATH_CALUDE_shop_profit_per_tshirt_l4089_408977


namespace NUMINAMATH_CALUDE_no_integer_both_roots_finite_decimal_l4089_408927

theorem no_integer_both_roots_finite_decimal (n : ℤ) (hn : n ≠ 0) :
  ¬(∃ (x₁ x₂ : ℚ), 
    (x₁ ≠ x₂) ∧
    ((4 * n^2 - 1) * x₁^2 - 4 * n^2 * x₁ + n^2 = 0) ∧
    ((4 * n^2 - 1) * x₂^2 - 4 * n^2 * x₂ + n^2 = 0) ∧
    (∃ (a b c d : ℤ), x₁ = (a : ℚ) / (2^b * 5^c) ∧ x₂ = (d : ℚ) / (2^b * 5^c))) :=
sorry

end NUMINAMATH_CALUDE_no_integer_both_roots_finite_decimal_l4089_408927


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l4089_408908

/-- 
For a quadratic equation x^2 - 4x + m - 1 = 0, 
if it has two distinct real roots, then m < 5.
-/
theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 4*x + m - 1 = 0 ∧ 
    y^2 - 4*y + m - 1 = 0) → 
  m < 5 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_distinct_roots_l4089_408908


namespace NUMINAMATH_CALUDE_box_sum_remainder_l4089_408932

theorem box_sum_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ (a i + i.val) % (2 * n) = (a j + j.val) % (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_box_sum_remainder_l4089_408932


namespace NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l4089_408975

/-- Given three positive integers with HCF 37 and LCM with additional prime factors 17, 19, 23, and 29,
    the largest of these numbers is 7,976,237 -/
theorem largest_number_with_given_hcf_and_lcm_factors
  (a b c : ℕ+)
  (hcf_abc : Nat.gcd a b.val = 37 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 37)
  (lcm_factors : ∃ (k : ℕ+), Nat.lcm (Nat.lcm a b.val) c.val = 37 * 17 * 19 * 23 * 29 * k) :
  max a (max b c) = 7976237 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_given_hcf_and_lcm_factors_l4089_408975


namespace NUMINAMATH_CALUDE_class_average_approximation_l4089_408935

/-- Represents the class data for a test --/
structure ClassData where
  total_students : ℕ
  section1_percent : ℝ
  section1_average : ℝ
  section2_percent : ℝ
  section2_average : ℝ
  section3_percent : ℝ
  section3_average : ℝ
  section4_average : ℝ
  weight1 : ℝ
  weight2 : ℝ
  weight3 : ℝ
  weight4 : ℝ

/-- Calculates the weighted overall class average --/
def weightedAverage (data : ClassData) : ℝ :=
  data.section1_average * data.weight1 +
  data.section2_average * data.weight2 +
  data.section3_average * data.weight3 +
  data.section4_average * data.weight4

/-- Theorem stating that the weighted overall class average is approximately 86% --/
theorem class_average_approximation (data : ClassData) 
  (h1 : data.total_students = 120)
  (h2 : data.section1_percent = 0.187)
  (h3 : data.section1_average = 0.965)
  (h4 : data.section2_percent = 0.355)
  (h5 : data.section2_average = 0.784)
  (h6 : data.section3_percent = 0.258)
  (h7 : data.section3_average = 0.882)
  (h8 : data.section4_average = 0.647)
  (h9 : data.weight1 = 0.35)
  (h10 : data.weight2 = 0.25)
  (h11 : data.weight3 = 0.30)
  (h12 : data.weight4 = 0.10)
  (h13 : data.section1_percent + data.section2_percent + data.section3_percent + 
         (1 - data.section1_percent - data.section2_percent - data.section3_percent) = 1) :
  abs (weightedAverage data - 0.86) < 0.005 := by
  sorry


end NUMINAMATH_CALUDE_class_average_approximation_l4089_408935


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l4089_408922

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l4089_408922


namespace NUMINAMATH_CALUDE_fraction_division_five_sixths_divided_by_nine_tenths_l4089_408970

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem five_sixths_divided_by_nine_tenths :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_five_sixths_divided_by_nine_tenths_l4089_408970


namespace NUMINAMATH_CALUDE_chelsea_cupcake_time_l4089_408925

/-- Calculates the total time Chelsea spent making and decorating cupcakes --/
def total_cupcake_time (num_batches : ℕ) 
                       (bake_time_per_batch : ℕ) 
                       (ice_time_per_batch : ℕ)
                       (cupcakes_per_batch : ℕ)
                       (decor_time_per_cupcake : List ℕ) : ℕ :=
  let base_time := num_batches * (bake_time_per_batch + ice_time_per_batch)
  let decor_time := (List.map (· * cupcakes_per_batch) decor_time_per_cupcake).sum
  base_time + decor_time

/-- Theorem stating that Chelsea's total time making and decorating cupcakes is 542 minutes --/
theorem chelsea_cupcake_time : 
  total_cupcake_time 4 20 30 6 [10, 15, 12, 20] = 542 := by
  sorry


end NUMINAMATH_CALUDE_chelsea_cupcake_time_l4089_408925


namespace NUMINAMATH_CALUDE_calculate_expression_l4089_408911

theorem calculate_expression : (1/2)⁻¹ + (Real.pi - 3.14)^0 - |-3| + Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4089_408911


namespace NUMINAMATH_CALUDE_ralph_tv_hours_l4089_408965

/-- The number of hours Ralph watches TV each day from Monday to Friday -/
def weekday_hours : ℝ := sorry

/-- The number of hours Ralph watches TV each day on Saturday and Sunday -/
def weekend_hours : ℝ := 6

/-- The total number of hours Ralph watches TV in one week -/
def total_weekly_hours : ℝ := 32

/-- Theorem stating that Ralph watches TV for 4 hours each day from Monday to Friday -/
theorem ralph_tv_hours : weekday_hours = 4 := by
  have h1 : 5 * weekday_hours + 2 * weekend_hours = total_weekly_hours := sorry
  sorry

end NUMINAMATH_CALUDE_ralph_tv_hours_l4089_408965


namespace NUMINAMATH_CALUDE_total_accidents_l4089_408916

/-- Represents the accident rate and total traffic for a highway -/
structure HighwayData where
  accidents : ℕ
  per_vehicles : ℕ
  total_vehicles : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculate_accidents (data : HighwayData) : ℕ :=
  (data.accidents * data.total_vehicles) / data.per_vehicles

/-- The given data for the three highways -/
def highway_A : HighwayData := ⟨75, 100, 2500⟩
def highway_B : HighwayData := ⟨50, 80, 1600⟩
def highway_C : HighwayData := ⟨90, 200, 1900⟩

/-- The theorem stating the total number of accidents across all three highways -/
theorem total_accidents :
  calculate_accidents highway_A +
  calculate_accidents highway_B +
  calculate_accidents highway_C = 3730 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l4089_408916


namespace NUMINAMATH_CALUDE_middle_digit_zero_l4089_408983

theorem middle_digit_zero (N : ℕ) (a b c : ℕ) : 
  (N = 49*a + 7*b + c) →  -- N in base 7
  (N = 81*c + 9*b + a) →  -- N in base 9
  (0 ≤ a ∧ a < 7) →       -- a is a valid digit in base 7
  (0 ≤ b ∧ b < 7) →       -- b is a valid digit in base 7
  (0 ≤ c ∧ c < 7) →       -- c is a valid digit in base 7
  (b = 0) :=              -- middle digit is 0
by sorry

end NUMINAMATH_CALUDE_middle_digit_zero_l4089_408983


namespace NUMINAMATH_CALUDE_acid_dilution_l4089_408987

/-- Given n ounces of n% acid solution, to obtain a (n-20)% solution by adding y ounces of water, 
    where n > 30, y must equal 20n / (n-20). -/
theorem acid_dilution (n : ℝ) (y : ℝ) (h : n > 30) :
  (n * n / 100 = (n - 20) * (n + y) / 100) → y = 20 * n / (n - 20) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l4089_408987


namespace NUMINAMATH_CALUDE_platform_length_l4089_408995

/-- Given a train of length l traveling at constant velocity, if it passes a pole in t seconds
    and a platform in 6t seconds, then the length of the platform is 5l. -/
theorem platform_length (l t : ℝ) (h1 : l > 0) (h2 : t > 0) : 
  (∃ v : ℝ, v > 0 ∧ v = l / t ∧ v = (l + 5 * l) / (6 * t)) := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l4089_408995


namespace NUMINAMATH_CALUDE_nested_square_root_value_l4089_408973

theorem nested_square_root_value (y : ℝ) :
  y = Real.sqrt (2 + y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l4089_408973


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l4089_408961

/-- Represents a car model with its production volume -/
structure CarModel where
  name : String
  volume : Nat

/-- Calculates the number of cars to be sampled from a given model -/
def sampleSize (model : CarModel) (totalProduction : Nat) (totalSample : Nat) : Nat :=
  (model.volume * totalSample) / totalProduction

/-- Theorem stating that the stratified sampling produces the correct sample sizes -/
theorem stratified_sampling_correct 
  (emgrand kingKong freedomShip : CarModel)
  (h1 : emgrand.volume = 1600)
  (h2 : kingKong.volume = 6000)
  (h3 : freedomShip.volume = 2000)
  (h4 : emgrand.volume + kingKong.volume + freedomShip.volume = 9600)
  (h5 : 48 ≤ 9600) :
  let totalProduction := 9600
  let totalSample := 48
  (sampleSize emgrand totalProduction totalSample = 8) ∧
  (sampleSize kingKong totalProduction totalSample = 30) ∧
  (sampleSize freedomShip totalProduction totalSample = 10) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l4089_408961


namespace NUMINAMATH_CALUDE_smaller_integer_problem_l4089_408988

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end NUMINAMATH_CALUDE_smaller_integer_problem_l4089_408988


namespace NUMINAMATH_CALUDE_fraction_equivalence_l4089_408945

theorem fraction_equivalence (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∀ x y : ℝ, (x + a) / (y + c) = a / c) ↔ (∀ x y : ℝ, x = (a / c) * y) :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l4089_408945


namespace NUMINAMATH_CALUDE_sum_and_product_identities_l4089_408971

theorem sum_and_product_identities (a b : ℝ) 
  (sum_eq : a + b = 4) 
  (product_eq : a * b = 1) : 
  a^2 + b^2 = 14 ∧ (a - b)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_identities_l4089_408971


namespace NUMINAMATH_CALUDE_negative_two_fourth_power_l4089_408921

theorem negative_two_fourth_power :
  ∀ (x : ℤ) (n : ℕ), x = -2 ∧ n = 4 → x^n = (-2)^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_fourth_power_l4089_408921


namespace NUMINAMATH_CALUDE_equation_solutions_l4089_408933

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 9 = 0 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, 2*x*(x - 3) + (x - 3) = 0 ↔ x = 3 ∨ x = -1/2) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = 1 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 16 = 0 ↔ x = 8 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4089_408933


namespace NUMINAMATH_CALUDE_trapezoid_area_three_squares_l4089_408967

/-- The area of the trapezoid formed by three squares with sides 3, 5, and 7 units -/
theorem trapezoid_area_three_squares :
  let square1 : ℝ := 3
  let square2 : ℝ := 5
  let square3 : ℝ := 7
  let total_base : ℝ := square1 + square2 + square3
  let height_ratio : ℝ := square3 / total_base
  let trapezoid_height : ℝ := square2
  let trapezoid_base1 : ℝ := square1 * height_ratio
  let trapezoid_base2 : ℝ := (square1 + square2) * height_ratio
  let trapezoid_area : ℝ := (trapezoid_base1 + trapezoid_base2) * trapezoid_height / 2
  trapezoid_area = 12.825 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_three_squares_l4089_408967


namespace NUMINAMATH_CALUDE_parabola_theorem_l4089_408969

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  eq : (a * x^2 : ℝ) + (b * x * y : ℝ) + (c * y^2 : ℝ) + (d * x : ℝ) + (e * y : ℝ) + (f : ℝ) = 0

/-- The parabola passes through the point (2,6) -/
def passes_through (p : Parabola) : Prop :=
  (p.a * 2^2 : ℝ) + (p.b * 2 * 6 : ℝ) + (p.c * 6^2 : ℝ) + (p.d * 2 : ℝ) + (p.e * 6 : ℝ) + (p.f : ℝ) = 0

/-- The y-coordinate of the focus is 4 -/
def focus_y_coord (p : Parabola) : Prop :=
  ∃ x : ℝ, (p.a * x^2 : ℝ) + (p.b * x * 4 : ℝ) + (p.c * 4^2 : ℝ) + (p.d * x : ℝ) + (p.e * 4 : ℝ) + (p.f : ℝ) = 0

/-- The axis of symmetry is parallel to the x-axis -/
def axis_parallel_x (p : Parabola) : Prop :=
  p.b = 0 ∧ p.c ≠ 0

/-- The vertex lies on the y-axis -/
def vertex_on_y_axis (p : Parabola) : Prop :=
  ∃ y : ℝ, (p.c * y^2 : ℝ) + (p.e * y : ℝ) + (p.f : ℝ) = 0

/-- The coefficients satisfy the required conditions -/
def coeff_conditions (p : Parabola) : Prop :=
  p.c > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The main theorem stating that the given equation represents a parabola satisfying all conditions -/
theorem parabola_theorem : ∃ p : Parabola, 
  p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -2 ∧ p.e = -8 ∧ p.f = 16 ∧
  passes_through p ∧
  focus_y_coord p ∧
  axis_parallel_x p ∧
  vertex_on_y_axis p ∧
  coeff_conditions p :=
sorry

end NUMINAMATH_CALUDE_parabola_theorem_l4089_408969


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l4089_408974

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_f_on_interval :
  ∃ (max min : ℝ), max = 5 ∧ min = -15 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
  (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
  (∃ x ∈ Set.Icc 0 3, f x = max) ∧
  (∃ x ∈ Set.Icc 0 3, f x = min) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l4089_408974


namespace NUMINAMATH_CALUDE_snooker_tournament_tickets_l4089_408947

theorem snooker_tournament_tickets (total_tickets : ℕ) (vip_price gen_price : ℚ) 
  (total_revenue : ℚ) (h1 : total_tickets = 320) (h2 : vip_price = 40) 
  (h3 : gen_price = 15) (h4 : total_revenue = 7500) : 
  ∃ (vip_tickets gen_tickets : ℕ), 
    vip_tickets + gen_tickets = total_tickets ∧ 
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue ∧ 
    gen_tickets - vip_tickets = 104 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_tickets_l4089_408947


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4089_408997

theorem inequality_equivalence (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) > 1 ↔ 1 < x ∧ x < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4089_408997


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l4089_408990

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = a + 3 ∧ n = 3 * b + 1 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 3 ∧ d > 3 ∧ m = c + 3 ∧ m = 3 * d + 1 → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l4089_408990


namespace NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l4089_408937

/-- The number of points on the circumference of the circle -/
def n : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def k : ℕ := 4

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 495 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l4089_408937


namespace NUMINAMATH_CALUDE_victoria_gym_schedule_l4089_408930

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a gym schedule -/
structure GymSchedule where
  startDay : DayOfWeek
  sessionsPlanned : ℕ
  publicHolidays : ℕ
  personalEvents : ℕ

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry

/-- Calculates the number of Sundays in a given number of days -/
def sundaysInDays (days : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of days needed to complete the gym schedule -/
def totalDays (schedule : GymSchedule) : ℕ :=
  sorry

/-- Theorem: Victoria completes her 30th gym session on a Wednesday -/
theorem victoria_gym_schedule (schedule : GymSchedule) 
  (h1 : schedule.startDay = DayOfWeek.Monday)
  (h2 : schedule.sessionsPlanned = 30)
  (h3 : schedule.publicHolidays = 3)
  (h4 : schedule.personalEvents = 2) :
  dayAfter schedule.startDay (totalDays schedule) = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_victoria_gym_schedule_l4089_408930


namespace NUMINAMATH_CALUDE_banks_revenue_is_500_l4089_408950

/-- Represents the revenue structure for Mr. Banks and Ms. Elizabeth -/
structure RevenueStructure where
  banks_investments : ℕ
  elizabeth_investments : ℕ
  elizabeth_revenue_per_investment : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Mr. Banks' revenue per investment given the revenue structure -/
def banks_revenue_per_investment (rs : RevenueStructure) : ℕ :=
  ((rs.elizabeth_investments * rs.elizabeth_revenue_per_investment) - rs.elizabeth_total_revenue_difference) / rs.banks_investments

/-- Theorem stating that Mr. Banks' revenue per investment is $500 given the specific conditions -/
theorem banks_revenue_is_500 (rs : RevenueStructure) 
  (h1 : rs.banks_investments = 8)
  (h2 : rs.elizabeth_investments = 5)
  (h3 : rs.elizabeth_revenue_per_investment = 900)
  (h4 : rs.elizabeth_total_revenue_difference = 500) :
  banks_revenue_per_investment rs = 500 := by
  sorry


end NUMINAMATH_CALUDE_banks_revenue_is_500_l4089_408950


namespace NUMINAMATH_CALUDE_inequality_proof_l4089_408914

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_pos : 0 < q) 
  (ha : p ≤ a ∧ a ≤ q) 
  (hb : p ≤ b ∧ b ≤ q) 
  (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) 
  (he : p ≤ e ∧ e ≤ q) : 
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4089_408914


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l4089_408919

/-- Represents the elevator scenario with people and their weights -/
structure ElevatorScenario where
  initial_people : ℕ
  initial_avg_weight : ℝ
  new_avg_weights : List ℝ

/-- Calculates the combined weight of new people entering the elevator -/
def combined_weight_of_new_people (scenario : ElevatorScenario) : ℝ :=
  sorry

/-- Theorem stating the combined weight of new people in the given scenario -/
theorem combined_weight_theorem (scenario : ElevatorScenario) :
  scenario.initial_people = 6 →
  scenario.initial_avg_weight = 152 →
  scenario.new_avg_weights = [154, 153, 151] →
  combined_weight_of_new_people scenario = 447 := by
  sorry

#check combined_weight_theorem

end NUMINAMATH_CALUDE_combined_weight_theorem_l4089_408919


namespace NUMINAMATH_CALUDE_spider_plant_theorem_l4089_408946

def spider_plant_problem (baby_plants_per_time : ℕ) (times_per_year : ℕ) (total_baby_plants : ℕ) : Prop :=
  let baby_plants_per_year := baby_plants_per_time * times_per_year
  let years_passed := total_baby_plants / baby_plants_per_year
  years_passed = 4

theorem spider_plant_theorem :
  spider_plant_problem 2 2 16 := by
  sorry

end NUMINAMATH_CALUDE_spider_plant_theorem_l4089_408946


namespace NUMINAMATH_CALUDE_time_to_finish_game_l4089_408900

/-- Calculates the time to finish a game given initial and increased play times --/
theorem time_to_finish_game 
  (initial_hours_per_day : ℝ)
  (initial_days : ℝ)
  (completion_percentage : ℝ)
  (increased_hours_per_day : ℝ) :
  initial_hours_per_day = 4 →
  initial_days = 14 →
  completion_percentage = 0.4 →
  increased_hours_per_day = 7 →
  (initial_days * initial_hours_per_day * (1 / completion_percentage) - 
   initial_days * initial_hours_per_day) / increased_hours_per_day = 12 := by
sorry

end NUMINAMATH_CALUDE_time_to_finish_game_l4089_408900


namespace NUMINAMATH_CALUDE_NaHSO3_moles_required_l4089_408906

/-- Represents the balanced chemical equation for the reaction -/
structure ChemicalEquation :=
  (reactants : List String)
  (products : List String)

/-- Represents the stoichiometric coefficient of a substance in a reaction -/
def stoichiometricCoefficient (equation : ChemicalEquation) (substance : String) : ℕ :=
  if substance ∈ equation.reactants ∨ substance ∈ equation.products then 1 else 0

/-- The chemical equation for the given reaction -/
def reactionEquation : ChemicalEquation :=
  { reactants := ["NaHSO3", "HCl"],
    products := ["SO2", "H2O", "NaCl"] }

/-- Theorem stating the number of moles of NaHSO3 required to form 2 moles of SO2 -/
theorem NaHSO3_moles_required :
  let NaHSO3_coeff := stoichiometricCoefficient reactionEquation "NaHSO3"
  let SO2_coeff := stoichiometricCoefficient reactionEquation "SO2"
  let SO2_moles_formed := 2
  NaHSO3_coeff * SO2_moles_formed / SO2_coeff = 2 := by
  sorry

end NUMINAMATH_CALUDE_NaHSO3_moles_required_l4089_408906


namespace NUMINAMATH_CALUDE_toothpicks_43_10_l4089_408989

/-- The number of toothpicks used in a 1 × 10 grid -/
def toothpicks_1_10 : ℕ := 31

/-- The number of toothpicks used in an n × 10 grid -/
def toothpicks_n_10 (n : ℕ) : ℕ := 21 * n + 10

/-- Theorem: The number of toothpicks in a 43 × 10 grid is 913 -/
theorem toothpicks_43_10 :
  toothpicks_n_10 43 = 913 :=
sorry

end NUMINAMATH_CALUDE_toothpicks_43_10_l4089_408989


namespace NUMINAMATH_CALUDE_joseph_running_distance_l4089_408951

/-- Calculates the daily running distance given the total distance and number of days -/
def dailyDistance (totalDistance : ℕ) (days : ℕ) : ℕ :=
  totalDistance / days

theorem joseph_running_distance :
  let totalDistance : ℕ := 2700
  let days : ℕ := 3
  dailyDistance totalDistance days = 900 := by
  sorry

end NUMINAMATH_CALUDE_joseph_running_distance_l4089_408951


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l4089_408941

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l4089_408941


namespace NUMINAMATH_CALUDE_linear_system_solution_l4089_408924

/-- A system of linear equations with a parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ∀ x y z : ℝ, x + m*y + 5*z = 0
  eq2 : ∀ x y z : ℝ, 4*x + m*y - 3*z = 0
  eq3 : ∀ x y z : ℝ, 3*x + 6*y - 4*z = 0

/-- The solution to the system exists and is nontrivial -/
def has_nontrivial_solution (m : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    x + m*y + 5*z = 0 ∧
    4*x + m*y - 3*z = 0 ∧
    3*x + 6*y - 4*z = 0

theorem linear_system_solution :
  ∃ m : ℝ, has_nontrivial_solution m ∧ m = 11.5 ∧
    ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 →
      x + m*y + 5*z = 0 →
      4*x + m*y - 3*z = 0 →
      3*x + 6*y - 4*z = 0 →
      x*z / (y^2) = -108/169 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l4089_408924


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l4089_408917

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 63 ∧ 
  (∀ (m : ℕ), (2023 : ℕ)^m ∣ factorial 2023 → m ≤ k) ∧
  (2023 : ℕ)^k ∣ factorial 2023 :=
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l4089_408917


namespace NUMINAMATH_CALUDE_walking_distance_l4089_408960

/-- If a person walks 1.5 miles in 45 minutes, they will travel 3 miles in 90 minutes at the same rate. -/
theorem walking_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
  (h1 : distance = 1.5)
  (h2 : time = 45)
  (h3 : new_time = 90) :
  (distance / time) * new_time = 3 := by
  sorry

#check walking_distance

end NUMINAMATH_CALUDE_walking_distance_l4089_408960


namespace NUMINAMATH_CALUDE_optimal_water_tank_design_l4089_408923

/-- Represents the dimensions and costs of a rectangular water tank -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank -/
def totalCost (tank : WaterTank) (length width : ℝ) : ℝ :=
  tank.bottomCost * length * width + 
  tank.wallCost * (2 * length * tank.depth + 2 * width * tank.depth)

/-- Theorem stating the optimal dimensions and minimum cost of the water tank -/
theorem optimal_water_tank_design (tank : WaterTank) 
  (h_volume : tank.volume = 4800)
  (h_depth : tank.depth = 3)
  (h_bottom_cost : tank.bottomCost = 150)
  (h_wall_cost : tank.wallCost = 120) :
  ∃ (cost : ℝ),
    (∀ length width, 
      length * width * tank.depth = tank.volume → 
      totalCost tank length width ≥ cost) ∧
    totalCost tank 40 40 = cost ∧
    cost = 297600 := by
  sorry

end NUMINAMATH_CALUDE_optimal_water_tank_design_l4089_408923


namespace NUMINAMATH_CALUDE_waiter_new_customers_l4089_408963

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (remaining_customers : ℕ) 
  (final_total_customers : ℕ) : 
  initial_customers = 8 →
  customers_left = 3 →
  remaining_customers = 5 →
  final_total_customers = 104 →
  final_total_customers - remaining_customers = 99 :=
by sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l4089_408963


namespace NUMINAMATH_CALUDE_total_distance_calculation_l4089_408956

/-- Calculates the total distance traveled given the distances and number of trips for each mode of transportation -/
def total_distance (plane_distance : Float) (train_distance : Float) (bus_distance : Float)
                   (plane_trips : Nat) (train_trips : Nat) (bus_trips : Nat) : Float :=
  plane_distance * plane_trips.toFloat +
  train_distance * train_trips.toFloat +
  bus_distance * bus_trips.toFloat

/-- Theorem stating that the total distance traveled is 11598.4 miles -/
theorem total_distance_calculation :
  total_distance 256.0 120.5 35.2 32 16 42 = 11598.4 := by
  sorry

#eval total_distance 256.0 120.5 35.2 32 16 42

end NUMINAMATH_CALUDE_total_distance_calculation_l4089_408956


namespace NUMINAMATH_CALUDE_complex_equality_l4089_408948

theorem complex_equality (z : ℂ) : 
  Complex.abs (1 + Complex.I * z) = Complex.abs (3 + 4 * Complex.I) →
  Complex.abs (z - Complex.I) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_l4089_408948


namespace NUMINAMATH_CALUDE_shelter_cats_l4089_408980

theorem shelter_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) : 
  total_animals = 1212 → dogs = 567 → total_animals = cats + dogs → cats = 645 := by
  sorry

end NUMINAMATH_CALUDE_shelter_cats_l4089_408980


namespace NUMINAMATH_CALUDE_distance_to_canada_is_360_l4089_408955

/-- Calculates the distance traveled given speed, total time, and stop time. -/
def distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) : ℝ :=
  speed * (total_time - stop_time)

/-- Proves that the distance to Canada is 360 miles under the given conditions. -/
theorem distance_to_canada_is_360 :
  distance_to_canada 60 7 1 = 360 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_canada_is_360_l4089_408955


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4089_408943

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 10 ∧ |r₁ - r₂| = 12) → (a = 1 ∧ b = -10 ∧ c = -11) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4089_408943


namespace NUMINAMATH_CALUDE_ratio_and_linear_equation_l4089_408957

theorem ratio_and_linear_equation (x y : ℚ) : 
  x / y = 4 → x = 18 - 3 * y → y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_and_linear_equation_l4089_408957


namespace NUMINAMATH_CALUDE_soccer_games_played_l4089_408968

theorem soccer_games_played (total_players : ℕ) (total_goals : ℕ) (goals_by_others : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  goals_by_others = 30 →
  ∃ (games_played : ℕ),
    games_played = 15 ∧
    games_played * (total_players / 3) + goals_by_others = total_goals :=
by sorry

end NUMINAMATH_CALUDE_soccer_games_played_l4089_408968


namespace NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l4089_408938

/-- The curve C: x^2 - y^2 = 1 (x > 0) -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1 ∧ x > 0

/-- The dot product function f -/
def f (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem min_dot_product_on_hyperbola :
  ∀ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ → C x₂ y₂ → 
  ∃ m : ℝ, m = 1 ∧ ∀ a b c d : ℝ, C a b → C c d → f x₁ y₁ x₂ y₂ ≥ m ∧ f a b c d ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l4089_408938


namespace NUMINAMATH_CALUDE_triangle_area_l4089_408934

/-- Given a triangle with perimeter 20 cm and inradius 2.5 cm, its area is 25 cm². -/
theorem triangle_area (p r A : ℝ) : 
  p = 20 → r = 2.5 → A = r * (p / 2) → A = 25 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4089_408934


namespace NUMINAMATH_CALUDE_smallest_m_value_l4089_408904

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_m_value :
  ∃ (m x y : ℕ),
    m = x * y * (10 * x + y) ∧
    100 ≤ m ∧ m < 1000 ∧
    x < 10 ∧ y < 10 ∧
    x ≠ y ∧
    is_prime (10 * x + y) ∧
    is_prime (x + y) ∧
    (∀ (m' x' y' : ℕ),
      m' = x' * y' * (10 * x' + y') →
      100 ≤ m' ∧ m' < 1000 →
      x' < 10 ∧ y' < 10 →
      x' ≠ y' →
      is_prime (10 * x' + y') →
      is_prime (x' + y') →
      m ≤ m') ∧
    m = 138 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_value_l4089_408904


namespace NUMINAMATH_CALUDE_conference_teams_l4089_408926

/-- The number of games played in a conference where each team plays every other team twice -/
def games_played (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 12 teams in the conference -/
theorem conference_teams : ∃ n : ℕ, n > 0 ∧ games_played n = 132 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_conference_teams_l4089_408926


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l4089_408999

/-- Given three positive real numbers that form a geometric sequence,
    their sum is 21, and subtracting 9 from the third number results
    in an arithmetic sequence, prove that the numbers are either
    (1, 4, 16) or (16, 4, 1). -/
theorem geometric_arithmetic_sequence (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence
  a + b + c = 21 →  -- sum is 21
  ∃ d : ℝ, b - a = d ∧ (c - 9) - b = d →  -- arithmetic sequence after subtracting 9
  ((a = 1 ∧ b = 4 ∧ c = 16) ∨ (a = 16 ∧ b = 4 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l4089_408999


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l4089_408998

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l4089_408998


namespace NUMINAMATH_CALUDE_line_slope_problem_l4089_408982

theorem line_slope_problem (k : ℚ) : 
  (∃ line : ℝ → ℝ, 
    (line (-1) = -4) ∧ 
    (line 3 = k) ∧ 
    (∀ x y : ℝ, x ≠ -1 → (line y - line x) / (y - x) = k)) → 
  k = 4/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_problem_l4089_408982


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l4089_408976

theorem shortest_side_of_right_triangle (a b c : ℝ) (ha : a = 5) (hb : b = 12) 
  (hright : a^2 + b^2 = c^2) : 
  min a (min b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l4089_408976


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l4089_408929

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (a, b) := f n
      ∃ x y : ℝ,
        x ≠ y ∧
        x * y = 1 ∧
        x^2012 = a * x + b ∧
        y^2012 = a * y + b :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l4089_408929


namespace NUMINAMATH_CALUDE_squared_inequality_condition_l4089_408903

theorem squared_inequality_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_squared_inequality_condition_l4089_408903


namespace NUMINAMATH_CALUDE_room_freezer_temp_difference_l4089_408964

/-- The temperature difference between room and freezer --/
def temperature_difference (room_temp freezer_temp : ℤ) : ℤ :=
  room_temp - freezer_temp

/-- Theorem stating the temperature difference between room and freezer --/
theorem room_freezer_temp_difference :
  temperature_difference 10 (-6) = 16 := by
  sorry

end NUMINAMATH_CALUDE_room_freezer_temp_difference_l4089_408964


namespace NUMINAMATH_CALUDE_incenter_distance_l4089_408912

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 15 ∧ AC = 17 ∧ BC = 16

-- Define the incenter
def Incenter (I : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) = r) ∧
  (Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = r) ∧
  (Real.sqrt ((I.1 - C.1)^2 + (I.2 - C.2)^2) = r)

theorem incenter_distance (A B C I : ℝ × ℝ) :
  Triangle A B C → Incenter I A B C →
  Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = Real.sqrt 85 :=
by sorry

end NUMINAMATH_CALUDE_incenter_distance_l4089_408912


namespace NUMINAMATH_CALUDE_chord_length_l4089_408954

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l4089_408954


namespace NUMINAMATH_CALUDE_functional_equation_equivalence_l4089_408902

theorem functional_equation_equivalence (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x + y + x * y) = f x + f y + f (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_equivalence_l4089_408902


namespace NUMINAMATH_CALUDE_apple_production_formula_l4089_408940

/-- Represents an apple orchard with additional trees planted -/
structure Orchard where
  initial_trees : ℕ
  initial_avg_apples : ℕ
  decrease_per_tree : ℕ
  additional_trees : ℕ

/-- Calculates the total number of apples produced in an orchard -/
def total_apples (o : Orchard) : ℕ :=
  (o.initial_trees + o.additional_trees) * (o.initial_avg_apples - o.decrease_per_tree * o.additional_trees)

/-- Theorem stating the relationship between additional trees and total apples -/
theorem apple_production_formula (x : ℕ) :
  let o : Orchard := {
    initial_trees := 10,
    initial_avg_apples := 200,
    decrease_per_tree := 5,
    additional_trees := x
  }
  total_apples o = (10 + x) * (200 - 5 * x) := by
  sorry

end NUMINAMATH_CALUDE_apple_production_formula_l4089_408940


namespace NUMINAMATH_CALUDE_odd_even_array_parity_l4089_408996

/-- Represents an n × n array where each entry is either 1 or -1 -/
def OddEvenArray (n : ℕ) := Fin n → Fin n → Int

/-- Counts the number of rows with an odd number of -1s -/
def oddRowCount (A : OddEvenArray n) : ℕ := sorry

/-- Counts the number of columns with an odd number of -1s -/
def oddColumnCount (A : OddEvenArray n) : ℕ := sorry

/-- The main theorem -/
theorem odd_even_array_parity (n : ℕ) (hn : Odd n) (A : OddEvenArray n) :
  Even (oddRowCount A + oddColumnCount A) := by sorry

end NUMINAMATH_CALUDE_odd_even_array_parity_l4089_408996


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4089_408986

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4089_408986


namespace NUMINAMATH_CALUDE_six_digit_number_puzzle_l4089_408928

theorem six_digit_number_puzzle :
  ∀ P Q R S T U : ℕ,
    P ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    Q ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
    Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
    R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
    S ≠ T ∧ S ≠ U ∧
    T ≠ U →
    (100 * P + 10 * Q + R) % 9 = 0 →
    (100 * Q + 10 * R + S) % 4 = 0 →
    (100 * R + 10 * S + T) % 3 = 0 →
    (P + Q + R + S + T + U) % 5 = 0 →
    U = 4 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_puzzle_l4089_408928


namespace NUMINAMATH_CALUDE_equal_utility_at_two_l4089_408952

/-- Utility function -/
def utility (swimming : ℝ) (coding : ℝ) : ℝ := 2 * swimming * coding + 1

/-- Saturday's utility -/
def saturday_utility (t : ℝ) : ℝ := utility t (10 - 2*t)

/-- Sunday's utility -/
def sunday_utility (t : ℝ) : ℝ := utility (4 - t) (2*t + 2)

/-- Theorem: The value of t that results in equal utility for both days is 2 -/
theorem equal_utility_at_two :
  ∃ t : ℝ, saturday_utility t = sunday_utility t ∧ t = 2 := by
sorry

end NUMINAMATH_CALUDE_equal_utility_at_two_l4089_408952


namespace NUMINAMATH_CALUDE_angle_A_range_triangle_area_l4089_408949

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.a * t.c = t.b^2

-- Theorem I
theorem angle_A_range (t : Triangle) (h : triangle_condition t) :
  0 < t.A ∧ t.A < π/3 := by sorry

-- Theorem II
theorem triangle_area (t : Triangle) (h : triangle_condition t) 
  (h_a : t.a = 2) (h_A : t.A = π/6) :
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_range_triangle_area_l4089_408949


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4089_408936

/-- The coordinates of a point (-1, 2) with respect to the origin in a Cartesian coordinate system are (-1, 2) -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-1, 2)
  P = P :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4089_408936


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_144_l4089_408962

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_144_l4089_408962


namespace NUMINAMATH_CALUDE_only_solution_is_three_l4089_408907

def sum_of_digits (n : ℕ) : ℕ :=
  sorry

theorem only_solution_is_three :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 :=
sorry

end NUMINAMATH_CALUDE_only_solution_is_three_l4089_408907


namespace NUMINAMATH_CALUDE_daisies_bought_l4089_408991

theorem daisies_bought (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : ℕ :=
  let daisies : ℕ := (total_spent - roses_bought * flower_price) / flower_price
  by
    -- Proof goes here
    sorry

#check daisies_bought 6 7 60 = 3

end NUMINAMATH_CALUDE_daisies_bought_l4089_408991


namespace NUMINAMATH_CALUDE_xyz_sum_root_l4089_408978

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 16) 
  (h2 : z + x = 18) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = 9 * Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l4089_408978


namespace NUMINAMATH_CALUDE_prob_sum_le_5_is_correct_l4089_408972

/-- The probability of the sum of two fair six-sided dice being less than or equal to 5 -/
def prob_sum_le_5 : ℚ :=
  5 / 18

/-- The set of possible outcomes when rolling two dice -/
def dice_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6)

/-- The set of favorable outcomes (sum ≤ 5) when rolling two dice -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  dice_outcomes.filter (fun p => p.1 + p.2 + 2 ≤ 5)

/-- Theorem stating that the probability of the sum of two fair six-sided dice
    being less than or equal to 5 is 5/18 -/
theorem prob_sum_le_5_is_correct :
  (favorable_outcomes.card : ℚ) / dice_outcomes.card = prob_sum_le_5 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_le_5_is_correct_l4089_408972


namespace NUMINAMATH_CALUDE_cyclist_rejoining_time_l4089_408966

/-- Prove that the time taken for a cyclist to break away from a group, travel 10 km ahead, 
    turn back, and rejoin the group is 1/4 hours. -/
theorem cyclist_rejoining_time 
  (group_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (separation_distance : ℝ) 
  (h1 : group_speed = 35) 
  (h2 : cyclist_speed = 45) 
  (h3 : separation_distance = 20) : 
  (separation_distance / (cyclist_speed - group_speed) = 1/4) := by
sorry

end NUMINAMATH_CALUDE_cyclist_rejoining_time_l4089_408966


namespace NUMINAMATH_CALUDE_equation_solutions_l4089_408920

theorem equation_solutions :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 - 9*y₁^2 = 18 ∧
    x₂^2 - 9*y₂^2 = 18 ∧
    x₁ = 19/2 ∧ y₁ = 17/6 ∧
    x₂ = 11/2 ∧ y₂ = 7/6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4089_408920


namespace NUMINAMATH_CALUDE_pandas_minus_lions_l4089_408984

/-- The number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- The conditions of John's zoo --/
def validZoo (zoo : ZooAnimals) : Prop :=
  zoo.snakes = 15 ∧
  zoo.monkeys = 2 * zoo.snakes ∧
  zoo.lions = zoo.monkeys - 5 ∧
  zoo.dogs = zoo.pandas / 3 ∧
  zoo.snakes + zoo.monkeys + zoo.lions + zoo.pandas + zoo.dogs = 114

/-- The theorem to prove --/
theorem pandas_minus_lions (zoo : ZooAnimals) (h : validZoo zoo) : 
  zoo.pandas - zoo.lions = 8 := by
  sorry


end NUMINAMATH_CALUDE_pandas_minus_lions_l4089_408984


namespace NUMINAMATH_CALUDE_pirate_digging_time_pirate_digging_time_proof_l4089_408905

/-- Calculates the time needed to dig up a buried treasure after natural events --/
theorem pirate_digging_time (initial_depth : ℝ) (initial_time : ℝ) 
  (storm_factor : ℝ) (tsunami_sand : ℝ) (earthquake_sand : ℝ) (mudslide_sand : ℝ) 
  (speed_change : ℝ) : ℝ :=
  let initial_speed := initial_depth / initial_time
  let new_speed := initial_speed * (1 - speed_change)
  let final_depth := initial_depth * storm_factor + tsunami_sand + earthquake_sand + mudslide_sand
  final_depth / new_speed

/-- Proves that the time to dig up the treasure is approximately 6.56 hours --/
theorem pirate_digging_time_proof :
  ∃ ε > 0, |pirate_digging_time 8 4 0.5 2 1.5 3 0.2 - 6.56| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_digging_time_pirate_digging_time_proof_l4089_408905


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l4089_408915

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l4089_408915


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l4089_408994

/-- Given a geometric sequence where the third term is 12 and the fourth term is 16,
    prove that the first term is 27/4. -/
theorem geometric_sequence_first_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^2 = 12) -- Third term is 12
  (h2 : a * r^3 = 16) -- Fourth term is 16
  : a = 27 / 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l4089_408994
