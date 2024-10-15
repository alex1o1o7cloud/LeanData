import Mathlib

namespace NUMINAMATH_CALUDE_male_gerbil_fraction_l1792_179202

theorem male_gerbil_fraction (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) :
  total_pets = 90 →
  total_gerbils = 66 →
  total_males = 25 →
  (total_pets - total_gerbils) / 3 + (total_males - (total_pets - total_gerbils) / 3) = total_males →
  (total_males - (total_pets - total_gerbils) / 3) / total_gerbils = 17 / 66 := by
  sorry

end NUMINAMATH_CALUDE_male_gerbil_fraction_l1792_179202


namespace NUMINAMATH_CALUDE_b_initial_investment_l1792_179295

/-- Represents the business scenario with two partners A and B -/
structure BusinessScenario where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment (unknown)
  a_withdraw : ℕ  -- Amount A withdraws after 8 months
  b_add : ℕ  -- Amount B adds after 8 months
  total_profit : ℕ  -- Total profit at the end of the year
  a_profit : ℕ  -- A's share of the profit

/-- Calculates the investment value for a partner -/
def investment_value (initial : ℕ) (change : ℕ) (is_withdraw : Bool) : ℕ :=
  if is_withdraw then
    8 * initial + 4 * (initial - change)
  else
    8 * initial + 4 * (initial + change)

/-- Theorem stating that B's initial investment was 4000 -/
theorem b_initial_investment
  (scenario : BusinessScenario)
  (h1 : scenario.a_initial = 6000)
  (h2 : scenario.a_withdraw = 1000)
  (h3 : scenario.b_add = 1000)
  (h4 : scenario.total_profit = 630)
  (h5 : scenario.a_profit = 357)
  : scenario.b_initial = 4000 := by
  sorry

end NUMINAMATH_CALUDE_b_initial_investment_l1792_179295


namespace NUMINAMATH_CALUDE_limit_of_sequence_a_l1792_179286

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | (n + 2) => (4/7) * a (n + 1) + (3/7) * a n

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - 17/10| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_of_sequence_a_l1792_179286


namespace NUMINAMATH_CALUDE_parallelogram_area_l1792_179231

/-- The area of a parallelogram with given dimensions -/
theorem parallelogram_area (h : ℝ) (angle : ℝ) (s : ℝ) : 
  h = 30 → angle = 60 * π / 180 → s = 15 → h * s * Real.cos angle = 225 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1792_179231


namespace NUMINAMATH_CALUDE_probability_first_box_given_defective_l1792_179251

def box1_total : ℕ := 5
def box1_defective : ℕ := 2
def box2_total : ℕ := 10
def box2_defective : ℕ := 3

def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

def prob_defective_given_box1 : ℚ := box1_defective / box1_total
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

theorem probability_first_box_given_defective :
  (prob_select_box1 * prob_defective_given_box1) /
  (prob_select_box1 * prob_defective_given_box1 + prob_select_box2 * prob_defective_given_box2) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_probability_first_box_given_defective_l1792_179251


namespace NUMINAMATH_CALUDE_last_toggled_locker_l1792_179287

theorem last_toggled_locker (n : Nat) (h : n = 2048) :
  (Nat.sqrt n) ^ 2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_last_toggled_locker_l1792_179287


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1792_179284

theorem polynomial_factorization : 
  ∀ x : ℝ, x^2 - x + (1/4 : ℝ) = (x - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1792_179284


namespace NUMINAMATH_CALUDE_sweet_potato_sharing_l1792_179254

theorem sweet_potato_sharing (total : ℝ) (per_person : ℝ) (h1 : total = 52.5) (h2 : per_person = 5) :
  total - (⌊total / per_person⌋ * per_person) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_sharing_l1792_179254


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1792_179230

theorem correct_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 20 ∧ original_mean = 150 ∧ incorrect_value = 135 ∧ correct_value = 160 →
  (n * original_mean - incorrect_value + correct_value) / n = 151.25 := by
sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1792_179230


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1792_179204

/-- The area of the circle described by the polar equation r = -4 cos θ + 8 sin θ is equal to 20π. -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := fun θ ↦ -4 * Real.cos θ + 8 * Real.sin θ
  ∃ c : ℝ × ℝ, ∃ radius : ℝ,
    (∀ θ : ℝ, (r θ * Real.cos θ - c.1)^2 + (r θ * Real.sin θ - c.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l1792_179204


namespace NUMINAMATH_CALUDE_power_of_1307_squared_cubed_l1792_179213

theorem power_of_1307_squared_cubed : (1307 * 1307)^3 = 4984209203082045649 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1307_squared_cubed_l1792_179213


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1792_179273

/-- The number of players on the basketball team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def starting_lineup_size : ℕ := 6

/-- The number of predetermined players in the starting lineup -/
def predetermined_players : ℕ := 3

/-- The number of different possible starting lineups -/
def different_lineups : ℕ := 220

theorem basketball_lineup_combinations :
  Nat.choose (total_players - predetermined_players) (starting_lineup_size - predetermined_players) = different_lineups := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1792_179273


namespace NUMINAMATH_CALUDE_inequality_proof_l1792_179258

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1792_179258


namespace NUMINAMATH_CALUDE_train_speed_l1792_179233

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 12) :
  (train_length + bridge_length) / crossing_time = 400 / 12 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l1792_179233


namespace NUMINAMATH_CALUDE_cover_room_with_tiles_l1792_179201

/-- The length of the room -/
def room_length : ℝ := 8

/-- The width of the room -/
def room_width : ℝ := 12

/-- The length of a tile -/
def tile_length : ℝ := 1.5

/-- The width of a tile -/
def tile_width : ℝ := 2

/-- The number of tiles needed to cover the room -/
def tiles_needed : ℕ := 32

theorem cover_room_with_tiles : 
  (room_length * room_width) / (tile_length * tile_width) = tiles_needed := by
  sorry

end NUMINAMATH_CALUDE_cover_room_with_tiles_l1792_179201


namespace NUMINAMATH_CALUDE_eight_digit_non_decreasing_remainder_l1792_179208

/-- The number of ways to arrange n indistinguishable objects into k distinguishable boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of 8-digit positive integers with non-decreasing digits -/
def M : ℕ := stars_and_bars 8 10

theorem eight_digit_non_decreasing_remainder :
  M % 1000 = 310 := by sorry

end NUMINAMATH_CALUDE_eight_digit_non_decreasing_remainder_l1792_179208


namespace NUMINAMATH_CALUDE_boy_age_proof_l1792_179265

/-- Given a group of boys with specific average ages, prove the age of the boy not in either subgroup -/
theorem boy_age_proof (total_boys : ℕ) (total_avg : ℚ) (first_six_avg : ℚ) (last_six_avg : ℚ) :
  total_boys = 13 ∧ 
  total_avg = 50 ∧ 
  first_six_avg = 49 ∧ 
  last_six_avg = 52 →
  ∃ (middle_boy_age : ℚ), middle_boy_age = 50 :=
by sorry


end NUMINAMATH_CALUDE_boy_age_proof_l1792_179265


namespace NUMINAMATH_CALUDE_fraction_integer_values_l1792_179225

theorem fraction_integer_values (a : ℤ) :
  (∃ k : ℤ, (a^3 + 1) / (a - 1) = k) ↔ a = -1 ∨ a = 0 ∨ a = 2 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_values_l1792_179225


namespace NUMINAMATH_CALUDE_inequality_proof_l1792_179228

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1792_179228


namespace NUMINAMATH_CALUDE_henrikhs_commute_l1792_179270

def blocks_to_office (x : ℕ) : Prop :=
  let walking_time := 60 * x
  let bicycle_time := 20 * x
  let skateboard_time := 40 * x
  (walking_time = bicycle_time + 480) ∧ 
  (walking_time = skateboard_time + 240)

theorem henrikhs_commute : ∃ (x : ℕ), blocks_to_office x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_henrikhs_commute_l1792_179270


namespace NUMINAMATH_CALUDE_negative_sqrt_seven_greater_than_negative_sqrt_eleven_l1792_179267

theorem negative_sqrt_seven_greater_than_negative_sqrt_eleven :
  -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_seven_greater_than_negative_sqrt_eleven_l1792_179267


namespace NUMINAMATH_CALUDE_odd_function_extension_l1792_179276

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then Real.exp (-x) + 2 * x - 1
  else -Real.exp x + 2 * x + 1

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x < 0, f x = Real.exp (-x) + 2 * x - 1) →
  (∀ x ≥ 0, f x = -Real.exp x + 2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1792_179276


namespace NUMINAMATH_CALUDE_log_7_18_l1792_179207

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : Real.log 2 / Real.log 10 = a)
variable (h2 : Real.log 3 / Real.log 10 = b)

-- State the theorem to be proved
theorem log_7_18 : Real.log 18 / Real.log 7 = (a + 2*b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_7_18_l1792_179207


namespace NUMINAMATH_CALUDE_golden_ratio_equivalences_l1792_179240

open Real

theorem golden_ratio_equivalences :
  let φ : ℝ := 2 * sin (18 * π / 180)
  (sin (102 * π / 180) + Real.sqrt 3 * cos (102 * π / 180) = φ) ∧
  (sin (36 * π / 180) / sin (108 * π / 180) = φ) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_equivalences_l1792_179240


namespace NUMINAMATH_CALUDE_point_set_equivalence_l1792_179291

theorem point_set_equivalence (x y : ℝ) : 
  y^2 - y = x^2 - x ↔ y = x ∨ y = 1 - x := by sorry

end NUMINAMATH_CALUDE_point_set_equivalence_l1792_179291


namespace NUMINAMATH_CALUDE_divisible_by_seven_l1792_179277

theorem divisible_by_seven (k : ℕ) : 
  7 ∣ (2^(6*k+1) + 3^(6*k+1) + 5^(6*k+1)) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l1792_179277


namespace NUMINAMATH_CALUDE_quadratic_vertex_l1792_179264

/-- Represents a quadratic function of the form y = ax^2 + bx - 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.contains (f : QuadraticFunction) (x y : ℝ) : Prop :=
  y = f.a * x^2 + f.b * x - 3

theorem quadratic_vertex (f : QuadraticFunction) :
  f.contains (-2) 5 →
  f.contains (-1) 0 →
  f.contains 0 (-3) →
  f.contains 1 (-4) →
  f.contains 2 (-3) →
  ∃ (a b : ℝ), f = ⟨a, b⟩ ∧ (1, -4) = (- b / (2 * a), - (b^2 - 4*a*3) / (4 * a)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l1792_179264


namespace NUMINAMATH_CALUDE_quadratic_inequality_max_value_l1792_179227

theorem quadratic_inequality_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ M : ℝ, M = 2/3 ∧ ∀ a b c : ℝ, b^2 / (3 * a^2 + c^2) ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_max_value_l1792_179227


namespace NUMINAMATH_CALUDE_discount_order_difference_l1792_179210

theorem discount_order_difference (initial_price : ℝ) (flat_discount : ℝ) (percentage_discount : ℝ) : 
  initial_price = 30 ∧ 
  flat_discount = 5 ∧ 
  percentage_discount = 0.25 →
  (initial_price - flat_discount) * (1 - percentage_discount) - 
  (initial_price * (1 - percentage_discount) - flat_discount) = 1.25 := by
sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1792_179210


namespace NUMINAMATH_CALUDE_root_sum_absolute_value_l1792_179238

theorem root_sum_absolute_value (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2027*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 98 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_absolute_value_l1792_179238


namespace NUMINAMATH_CALUDE_sample_size_is_thirteen_l1792_179224

/-- Represents a workshop with its production quantity -/
structure Workshop where
  quantity : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleFromSmallest : ℕ

/-- Calculates the total sample size for a stratified sampling scenario -/
def totalSampleSize (s : StratifiedSampling) : ℕ :=
  sorry

/-- The main theorem stating that for the given scenario, the total sample size is 13 -/
theorem sample_size_is_thirteen :
  let scenario := StratifiedSampling.mk
    [Workshop.mk 120, Workshop.mk 80, Workshop.mk 60]
    3
  totalSampleSize scenario = 13 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_thirteen_l1792_179224


namespace NUMINAMATH_CALUDE_simple_interest_difference_l1792_179205

/-- Calculate the simple interest and prove that it's Rs. 306 less than the principal -/
theorem simple_interest_difference (principal rate time : ℝ) : 
  principal = 450 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 306 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l1792_179205


namespace NUMINAMATH_CALUDE_quiche_egg_volume_l1792_179263

/-- Given the initial volume of raw spinach, the percentage it reduces to when cooked,
    the volume of cream cheese added, and the total volume of the quiche,
    calculate the volume of eggs used. -/
theorem quiche_egg_volume
  (raw_spinach : ℝ)
  (cooked_spinach_percentage : ℝ)
  (cream_cheese : ℝ)
  (total_quiche : ℝ)
  (h1 : raw_spinach = 40)
  (h2 : cooked_spinach_percentage = 0.20)
  (h3 : cream_cheese = 6)
  (h4 : total_quiche = 18) :
  total_quiche - (raw_spinach * cooked_spinach_percentage + cream_cheese) = 4 := by
  sorry

end NUMINAMATH_CALUDE_quiche_egg_volume_l1792_179263


namespace NUMINAMATH_CALUDE_sally_picked_peaches_l1792_179297

/-- Calculates the number of peaches Sally picked at the orchard. -/
def peaches_picked (initial_peaches final_peaches : ℕ) : ℕ :=
  final_peaches - initial_peaches

/-- Theorem stating that Sally picked 55 peaches at the orchard. -/
theorem sally_picked_peaches : peaches_picked 13 68 = 55 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_peaches_l1792_179297


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1792_179293

theorem contradiction_assumption (x y z : ℝ) :
  (¬ (x > 0 ∨ y > 0 ∨ z > 0)) ↔ (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1792_179293


namespace NUMINAMATH_CALUDE_consecutive_product_sum_l1792_179235

theorem consecutive_product_sum (a b c : ℤ) : 
  b = a + 1 → c = b + 1 → a * b * c = 210 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_sum_l1792_179235


namespace NUMINAMATH_CALUDE_equal_money_in_five_weeks_l1792_179262

/-- Represents the number of weeks it takes for two people to have the same amount of money -/
def weeks_to_equal_money (carol_initial : ℕ) (carol_weekly : ℕ) (mike_initial : ℕ) (mike_weekly : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 5 weeks for Carol and Mike to have the same amount of money -/
theorem equal_money_in_five_weeks :
  weeks_to_equal_money 60 9 90 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_money_in_five_weeks_l1792_179262


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1792_179292

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1792_179292


namespace NUMINAMATH_CALUDE_equal_fractions_from_given_numbers_l1792_179212

theorem equal_fractions_from_given_numbers : 
  let numbers : Finset ℕ := {2, 4, 5, 6, 12, 15}
  ∃ (a b c d e f : ℕ), 
    a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (a : ℚ) / b = (c : ℚ) / d ∧ (c : ℚ) / d = (e : ℚ) / f :=
by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_from_given_numbers_l1792_179212


namespace NUMINAMATH_CALUDE_linear_function_property_l1792_179266

theorem linear_function_property (k b : ℝ) : 
  (3 = k + b) → (2 = -k + b) → k^2 - b^2 = -6 := by sorry

end NUMINAMATH_CALUDE_linear_function_property_l1792_179266


namespace NUMINAMATH_CALUDE_crayons_lost_l1792_179288

theorem crayons_lost (initial : ℕ) (given_away : ℕ) (final : ℕ) 
  (h1 : initial = 440)
  (h2 : given_away = 111)
  (h3 : final = 223) :
  initial - given_away - final = 106 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_l1792_179288


namespace NUMINAMATH_CALUDE_power_sum_sequence_l1792_179226

/-- Given a sequence of sums of powers of a and b, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_sequence_l1792_179226


namespace NUMINAMATH_CALUDE_abs_frac_inequality_l1792_179271

theorem abs_frac_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| < 3 ↔ 4/3 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_abs_frac_inequality_l1792_179271


namespace NUMINAMATH_CALUDE_sugar_calculation_l1792_179278

theorem sugar_calculation (recipe_sugar : ℕ) (additional_sugar : ℕ) 
  (h1 : recipe_sugar = 7)
  (h2 : additional_sugar = 3) :
  recipe_sugar - additional_sugar = 4 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l1792_179278


namespace NUMINAMATH_CALUDE_regular_soda_count_l1792_179236

/-- The number of bottles of diet soda -/
def diet_soda : ℕ := 60

/-- The number of bottles of lite soda -/
def lite_soda : ℕ := 60

/-- The difference between regular and diet soda bottles -/
def regular_diet_difference : ℕ := 21

/-- The number of bottles of regular soda -/
def regular_soda : ℕ := diet_soda + regular_diet_difference

theorem regular_soda_count : regular_soda = 81 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l1792_179236


namespace NUMINAMATH_CALUDE_percentage_equality_l1792_179220

theorem percentage_equality : ∃ x : ℝ, (x / 100) * 75 = (2.5 / 100) * 450 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1792_179220


namespace NUMINAMATH_CALUDE_sunday_price_calculation_l1792_179247

def original_price : ℝ := 250
def regular_discount : ℝ := 0.4
def sunday_discount : ℝ := 0.25

theorem sunday_price_calculation : 
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_price_calculation_l1792_179247


namespace NUMINAMATH_CALUDE_tylers_meal_combinations_l1792_179259

/-- The number of meat options available --/
def meat_options : ℕ := 4

/-- The number of vegetable options available --/
def vegetable_options : ℕ := 5

/-- The number of dessert options available --/
def dessert_options : ℕ := 5

/-- The number of vegetables Tyler must choose --/
def vegetables_to_choose : ℕ := 3

/-- The number of desserts Tyler must choose --/
def desserts_to_choose : ℕ := 2

/-- The number of unique meal combinations Tyler can choose --/
def unique_meals : ℕ := meat_options * (Nat.choose vegetable_options vegetables_to_choose) * (Nat.choose dessert_options desserts_to_choose)

theorem tylers_meal_combinations :
  unique_meals = 400 :=
sorry

end NUMINAMATH_CALUDE_tylers_meal_combinations_l1792_179259


namespace NUMINAMATH_CALUDE_min_value_problem_l1792_179206

theorem min_value_problem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  ∃ (min : ℝ), min = 27 ∧ ∀ z, z = (x + 1) * (y + 2) → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1792_179206


namespace NUMINAMATH_CALUDE_total_fruits_is_78_l1792_179283

-- Define the number of fruits for Louis
def louis_oranges : ℕ := 5
def louis_apples : ℕ := 3

-- Define the number of fruits for Samantha
def samantha_oranges : ℕ := 8
def samantha_apples : ℕ := 7

-- Define the number of fruits for Marley
def marley_oranges : ℕ := 2 * louis_oranges
def marley_apples : ℕ := 3 * samantha_apples

-- Define the number of fruits for Edward
def edward_oranges : ℕ := 3 * louis_oranges
def edward_apples : ℕ := 3 * louis_apples

-- Define the total number of fruits
def total_fruits : ℕ := 
  louis_oranges + louis_apples + 
  samantha_oranges + samantha_apples + 
  marley_oranges + marley_apples + 
  edward_oranges + edward_apples

-- Theorem statement
theorem total_fruits_is_78 : total_fruits = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_78_l1792_179283


namespace NUMINAMATH_CALUDE_jeff_running_schedule_l1792_179275

/-- Jeff's running schedule problem -/
theorem jeff_running_schedule (x : ℕ) : 
  (3 * x + (x - 20) + (x + 10) = 290) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_jeff_running_schedule_l1792_179275


namespace NUMINAMATH_CALUDE_sum_of_bottom_circles_l1792_179269

-- Define the type for circle positions
inductive Position
| Top | UpperLeft | UpperMiddle | UpperRight | Middle | LowerLeft | LowerMiddle | LowerRight

-- Define the function type for number placement
def Placement := Position → Nat

-- Define the conditions of the problem
def validPlacement (p : Placement) : Prop :=
  (∀ i : Position, p i ∈ Finset.range 9 \ {0}) ∧ 
  (∀ i j : Position, i ≠ j → p i ≠ p j) ∧
  p Position.Top * p Position.UpperLeft * p Position.UpperMiddle = 30 ∧
  p Position.Top * p Position.UpperMiddle * p Position.UpperRight = 40 ∧
  p Position.UpperLeft * p Position.Middle * p Position.LowerLeft = 28 ∧
  p Position.UpperRight * p Position.Middle * p Position.LowerRight = 35 ∧
  p Position.LowerLeft * p Position.LowerMiddle * p Position.LowerRight = 20

-- State the theorem
theorem sum_of_bottom_circles (p : Placement) (h : validPlacement p) : 
  p Position.LowerLeft + p Position.LowerMiddle + p Position.LowerRight = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bottom_circles_l1792_179269


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l1792_179294

open Real

theorem sin_shift_equivalence (x : ℝ) :
  sin (2 * (x + π / 6)) = sin (2 * x + π / 3) := by sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l1792_179294


namespace NUMINAMATH_CALUDE_share_price_increase_l1792_179203

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter := P * 1.25
  let second_quarter := first_quarter * 1.24
  (second_quarter - P) / P * 100 = 55 := by
sorry

end NUMINAMATH_CALUDE_share_price_increase_l1792_179203


namespace NUMINAMATH_CALUDE_least_number_of_pennies_l1792_179248

theorem least_number_of_pennies : ∃ (a : ℕ), a > 0 ∧ 
  a % 7 = 3 ∧ 
  a % 5 = 4 ∧ 
  a % 3 = 2 ∧ 
  ∀ (b : ℕ), b > 0 → b % 7 = 3 → b % 5 = 4 → b % 3 = 2 → a ≤ b :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_pennies_l1792_179248


namespace NUMINAMATH_CALUDE_base_10_to_base_8_l1792_179223

theorem base_10_to_base_8 : 
  (3 * 8^3 + 4 * 8^2 + 1 * 8^1 + 1 * 8^0 : ℕ) = 1801 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_8_l1792_179223


namespace NUMINAMATH_CALUDE_extra_bottles_eq_three_l1792_179299

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 9

/-- The difference between Donald's daily juice consumption and twice Paul's daily juice consumption -/
def extra_bottles : ℕ := donald_bottles - 2 * paul_bottles

theorem extra_bottles_eq_three : extra_bottles = 3 := by
  sorry

end NUMINAMATH_CALUDE_extra_bottles_eq_three_l1792_179299


namespace NUMINAMATH_CALUDE_A_subset_B_l1792_179285

def A : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}

theorem A_subset_B : A ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_A_subset_B_l1792_179285


namespace NUMINAMATH_CALUDE_shelbys_scooter_problem_l1792_179260

/-- Shelby's scooter problem -/
theorem shelbys_scooter_problem 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 25)
  (h2 : speed_rain = 15)
  (h3 : total_distance = 18)
  (h4 : total_time = 36)
  : ∃ (time_no_rain : ℝ), 
    time_no_rain = 6 ∧ 
    speed_no_rain * (time_no_rain / 60) + speed_rain * ((total_time - time_no_rain) / 60) = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shelbys_scooter_problem_l1792_179260


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1792_179216

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2
def divisor (x : ℝ) : ℝ := (x^2 - 1) * (x - 2)

-- Define the remainder
def remainder (x : ℝ) : ℝ := 9 * x^2 - 8

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = divisor x * q x + remainder x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1792_179216


namespace NUMINAMATH_CALUDE_problem_solution_l1792_179282

theorem problem_solution : ∃ x : ℝ, (0.15 * 40 = 0.25 * x + 2) ∧ (x = 16) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1792_179282


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1792_179261

theorem fahrenheit_to_celsius (C F : ℝ) : C = (4 / 7) * (F - 40) → C = 35 → F = 101.25 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l1792_179261


namespace NUMINAMATH_CALUDE_average_not_1380_l1792_179253

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1200]

def average (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem average_not_1380 : average numbers ≠ 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_not_1380_l1792_179253


namespace NUMINAMATH_CALUDE_factory_max_profit_l1792_179237

/-- The annual profit function for the factory -/
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 80 then
    -(1/3) * x^2 + 40 * x - 250
  else if x ≥ 80 then
    50 * x - 10000 / x + 1200
  else
    0

/-- The maximum profit and corresponding production level -/
theorem factory_max_profit :
  (∃ (x : ℝ), L x = 1000 ∧ x = 100) ∧
  (∀ (y : ℝ), L y ≤ 1000) := by
  sorry

end NUMINAMATH_CALUDE_factory_max_profit_l1792_179237


namespace NUMINAMATH_CALUDE_power_of_64_l1792_179209

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l1792_179209


namespace NUMINAMATH_CALUDE_unique_N_value_l1792_179250

theorem unique_N_value (a b N : ℕ) (h1 : N = (a^2 + b^2) / (a*b - 1)) : N = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_N_value_l1792_179250


namespace NUMINAMATH_CALUDE_parabola_values_l1792_179229

/-- A parabola passing through specific points -/
structure Parabola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ := λ x => x^2 + a * x + b
  point1 : eq 2 = 20
  point2 : eq (-2) = 0
  point3 : eq 0 = b

/-- The values of a and b for the given parabola -/
theorem parabola_values (p : Parabola) : p.a = 5 ∧ p.b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_values_l1792_179229


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l1792_179214

/-- The daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 300

/-- The daily profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 10) * sales_quantity x

theorem profit_and_max_profit :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit x₁ = 750 ∧ profit x₂ = 750 ∧ 
    ((∀ x : ℝ, profit x = 750 → x = x₁ ∨ x = x₂) ∧ 
    (x₁ = 15 ∨ x₁ = 25) ∧ (x₂ = 15 ∨ x₂ = 25))) ∧
  (∃ max_profit : ℝ, max_profit = 1000 ∧ ∀ x : ℝ, profit x ≤ max_profit) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l1792_179214


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1792_179274

theorem pure_imaginary_complex_number (x : ℝ) :
  (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).re = 0 ∧ (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).im ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1792_179274


namespace NUMINAMATH_CALUDE_x_formula_l1792_179244

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (x (n + 1))^2 / (x n + 2 * x (n + 1))

theorem x_formula (n : ℕ) : x n = 1 / (double_factorial (2 * n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_x_formula_l1792_179244


namespace NUMINAMATH_CALUDE_largest_indecomposable_amount_l1792_179217

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (fun k => 3^(n - k) * 5^k)

/-- Predicate to check if a number is decomposable using given coin denominations -/
def is_decomposable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), 
    coeffs.length = n + 1 ∧ 
    (List.zip coeffs (coin_denominations n) |> List.map (fun (c, d) => c * d) |> List.sum) = s

/-- The main theorem stating the largest indecomposable amount -/
theorem largest_indecomposable_amount (n : ℕ) : 
  ¬(is_decomposable (5^(n+1) - 2 * 3^(n+1)) n) ∧ 
  ∀ m : ℕ, m > (5^(n+1) - 2 * 3^(n+1)) → is_decomposable m n :=
by sorry

end NUMINAMATH_CALUDE_largest_indecomposable_amount_l1792_179217


namespace NUMINAMATH_CALUDE_smoothie_servings_l1792_179218

/-- The number of servings that can be made from a given volume of smoothie mix -/
def number_of_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree and 100 ml of cream, 
    the number of 150 ml servings that can be made is equal to 4 -/
theorem smoothie_servings : 
  number_of_servings 500 100 150 = 4 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_servings_l1792_179218


namespace NUMINAMATH_CALUDE_tangent_line_properties_l1792_179252

/-- Parabola C: x^2 = 4y with focus F(0, 1) -/
structure Parabola where
  C : ℝ → ℝ
  F : ℝ × ℝ
  h : C = fun x ↦ (x^2) / 4
  focus : F = (0, 1)

/-- Line through P(a, -2) forming tangents to C at A(x₁, y₁) and B(x₂, y₂) -/
structure TangentLine (C : Parabola) where
  a : ℝ
  P : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : P = (a, -2)
  h₂ : A.2 = C.C A.1
  h₃ : B.2 = C.C B.1

/-- Circumcenter of triangle PAB -/
def circumcenter (C : Parabola) (L : TangentLine C) : ℝ × ℝ := sorry

/-- Main theorem -/
theorem tangent_line_properties (C : Parabola) (L : TangentLine C) :
  let (x₁, y₁) := L.A
  let (x₂, y₂) := L.B
  let M := circumcenter C L
  (x₁ * x₂ + y₁ * y₂ = -4) ∧
  (∃ r : ℝ, (M.1 - C.F.1)^2 + (M.2 - C.F.2)^2 = r^2 ∧
            (L.P.1 - C.F.1)^2 + (L.P.2 - C.F.2)^2 = r^2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l1792_179252


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1792_179257

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p * q = k

theorem inverse_proportion_problem (p q : ℝ) :
  InverselyProportional p q →
  p + q = 40 →
  p - q = 10 →
  p = 7 →
  q = 375 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1792_179257


namespace NUMINAMATH_CALUDE_divisible_by_seven_l1792_179246

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(12*n + 1) + 2^(6*n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seven_l1792_179246


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1792_179219

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through the point (1, -2) -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  (-1 = (k + b) / 2) →
  ∀ x y : ℝ, y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1792_179219


namespace NUMINAMATH_CALUDE_coefficient_x5y3_in_expansion_l1792_179239

def binomial_expansion (a b : ℤ) (n : ℕ) : Polynomial ℤ := sorry

def coefficient_of_term (p : Polynomial ℤ) (x_power y_power : ℕ) : ℤ := sorry

theorem coefficient_x5y3_in_expansion :
  let p := binomial_expansion 2 (-3) 6
  coefficient_of_term (p - Polynomial.C (-1) * Polynomial.X ^ 6) 5 3 = 720 := by sorry

end NUMINAMATH_CALUDE_coefficient_x5y3_in_expansion_l1792_179239


namespace NUMINAMATH_CALUDE_pirate_treasure_chests_l1792_179243

theorem pirate_treasure_chests : ∀ (gold silver bronze chests : ℕ),
  gold = 3500 →
  silver = 500 →
  bronze = 2 * silver →
  (gold + silver + bronze) / 1000 = chests →
  chests * 1000 = gold + silver + bronze →
  chests = 5 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_chests_l1792_179243


namespace NUMINAMATH_CALUDE_min_value_theorem_l1792_179268

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a + 8) * x + a^2 + a - 12

theorem min_value_theorem (a : ℝ) (h1 : a < 0) 
  (h2 : f a (a^2 - 4) = f a (2*a - 8)) :
  ∀ n : ℕ+, (f a n - 4*a) / (n + 1) ≥ 37/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1792_179268


namespace NUMINAMATH_CALUDE_plane_division_theorem_l1792_179281

/-- Represents the number of regions formed by lines in a plane -/
def num_regions (h s : ℕ) : ℕ := h * (s + 1) + 1 + s * (s + 1) / 2

/-- Checks if a pair (h, s) satisfies the problem conditions -/
def is_valid_pair (h s : ℕ) : Prop :=
  h > 0 ∧ s > 0 ∧ num_regions h s = 1992

theorem plane_division_theorem :
  ∀ h s : ℕ, is_valid_pair h s ↔ (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by sorry

end NUMINAMATH_CALUDE_plane_division_theorem_l1792_179281


namespace NUMINAMATH_CALUDE_max_common_roots_and_coefficients_l1792_179215

/-- A polynomial of degree 2020 with non-zero coefficients -/
def Polynomial2020 : Type := { p : Polynomial ℝ // p.degree = some 2020 ∧ ∀ i, p.coeff i ≠ 0 }

/-- The number of common real roots (counting multiplicity) of two polynomials -/
noncomputable def commonRoots (P Q : Polynomial2020) : ℕ := sorry

/-- The number of common coefficients of two polynomials -/
def commonCoefficients (P Q : Polynomial2020) : ℕ := sorry

/-- The main theorem: the maximum possible value of r + s is 3029 -/
theorem max_common_roots_and_coefficients (P Q : Polynomial2020) (h : P ≠ Q) :
  commonRoots P Q + commonCoefficients P Q ≤ 3029 := by sorry

end NUMINAMATH_CALUDE_max_common_roots_and_coefficients_l1792_179215


namespace NUMINAMATH_CALUDE_attendance_difference_l1792_179221

/-- Calculates the total attendance for a week of football games given the conditions. -/
def totalAttendance (saturdayAttendance : ℕ) (expectedTotal : ℕ) : ℕ :=
  let mondayAttendance := saturdayAttendance - 20
  let wednesdayAttendance := mondayAttendance + 50
  let fridayAttendance := saturdayAttendance + mondayAttendance
  saturdayAttendance + mondayAttendance + wednesdayAttendance + fridayAttendance

/-- Theorem stating that the actual attendance exceeds the expected attendance by 40 people. -/
theorem attendance_difference (saturdayAttendance : ℕ) (expectedTotal : ℕ) 
  (h1 : saturdayAttendance = 80) 
  (h2 : expectedTotal = 350) : 
  totalAttendance saturdayAttendance expectedTotal - expectedTotal = 40 := by
  sorry

#eval totalAttendance 80 350 - 350

end NUMINAMATH_CALUDE_attendance_difference_l1792_179221


namespace NUMINAMATH_CALUDE_even_monotone_inequality_l1792_179289

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_monotone_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  f (-2) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_monotone_inequality_l1792_179289


namespace NUMINAMATH_CALUDE_shower_tiles_l1792_179241

/-- Calculates the total number of tiles in a shower --/
def total_tiles (sides : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  sides * width * height

/-- Theorem: The total number of tiles in a 3-sided shower with 8 tiles in width and 20 tiles in height is 480 --/
theorem shower_tiles : total_tiles 3 8 20 = 480 := by
  sorry

end NUMINAMATH_CALUDE_shower_tiles_l1792_179241


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l1792_179290

-- Define the number of people in the main committee
def n : ℕ := 8

-- Define the number of people to be selected for each sub-committee
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem three_person_subcommittees_from_eight :
  combination n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l1792_179290


namespace NUMINAMATH_CALUDE_sale_price_is_63_percent_l1792_179255

/-- The sale price of an item after two successive discounts -/
def sale_price (original_price : ℝ) : ℝ :=
  let first_discount := 0.1
  let second_discount := 0.3
  let price_after_first_discount := original_price * (1 - first_discount)
  price_after_first_discount * (1 - second_discount)

/-- Theorem stating that the sale price is 63% of the original price -/
theorem sale_price_is_63_percent (x : ℝ) : sale_price x = 0.63 * x := by
  sorry

end NUMINAMATH_CALUDE_sale_price_is_63_percent_l1792_179255


namespace NUMINAMATH_CALUDE_sequence_increasing_k_bound_l1792_179245

theorem sequence_increasing_k_bound (k : ℝ) :
  (∀ n : ℕ+, (2 * n^2 + k * n) < (2 * (n + 1)^2 + k * (n + 1))) →
  k > -6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_k_bound_l1792_179245


namespace NUMINAMATH_CALUDE_teacher_school_arrangements_l1792_179234

theorem teacher_school_arrangements :
  let n : ℕ := 4  -- number of teachers and schools
  let arrangements := {f : Fin n → Fin n | Function.Surjective f}  -- surjective functions represent valid arrangements
  Fintype.card arrangements = 24 := by
sorry

end NUMINAMATH_CALUDE_teacher_school_arrangements_l1792_179234


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_of_f_l1792_179232

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

theorem monotonicity_and_extrema_of_f :
  (∀ x y, x < -1 → y < -1 → f x < f y) ∧ 
  (∀ x y, 3 < x → 3 < y → f x < f y) ∧
  (∀ x y, -1 < x → x < y → y < 3 → f x > f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - (-1)| → |x - (-1)| < δ → f x < f (-1)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 3| → |x - 3| < δ → f x > f 3) ∧
  f (-1) = 6 ∧
  f 3 = -26 :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_of_f_l1792_179232


namespace NUMINAMATH_CALUDE_f_composition_fixed_points_l1792_179298

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_composition_fixed_points :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_fixed_points_l1792_179298


namespace NUMINAMATH_CALUDE_f_periodicity_and_smallest_a_l1792_179249

def is_valid_f (f : ℕ+ → ℝ) (a : ℕ+) : Prop :=
  f a = f 1995 ∧
  f (a + 1) = f 1996 ∧
  f (a + 2) = f 1997 ∧
  ∀ n : ℕ+, f (n + a) = (f n - 1) / (f n + 1)

theorem f_periodicity_and_smallest_a :
  ∃ (f : ℕ+ → ℝ) (a : ℕ+),
    is_valid_f f a ∧
    (∀ n : ℕ+, f (n + 4 * a) = f n) ∧
    (∀ a' : ℕ+, a' < a → ¬ is_valid_f f a') :=
  sorry

end NUMINAMATH_CALUDE_f_periodicity_and_smallest_a_l1792_179249


namespace NUMINAMATH_CALUDE_four_digit_sum_l1792_179272

theorem four_digit_sum (A B C D : Nat) : 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A < 10 → B < 10 → C < 10 → D < 10 →
  (A + B + C) % 9 = 0 →
  (B + C + D) % 9 = 0 →
  A + B + C + D = 18 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l1792_179272


namespace NUMINAMATH_CALUDE_system_solution_l1792_179211

theorem system_solution (x y z t : ℤ) : 
  (x * y + z * t = 1 ∧ 
   x * z + y * t = 1 ∧ 
   x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (0, 1, 1, 1) ∨
   (x, y, z, t) = (1, 0, 1, 1) ∨
   (x, y, z, t) = (1, 1, 0, 1) ∨
   (x, y, z, t) = (1, 1, 1, 0) ∨
   (x, y, z, t) = (0, -1, -1, -1) ∨
   (x, y, z, t) = (-1, 0, -1, -1) ∨
   (x, y, z, t) = (-1, -1, 0, -1) ∨
   (x, y, z, t) = (-1, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1792_179211


namespace NUMINAMATH_CALUDE_ellipse_intersection_l1792_179280

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-4)^2) + Real.sqrt ((x-6)^2 + y^2) = 10

-- Define the foci
def F1 : ℝ × ℝ := (0, 4)
def F2 : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem ellipse_intersection :
  ∃ (x : ℝ), x ≠ 0 ∧ ellipse x 0 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l1792_179280


namespace NUMINAMATH_CALUDE_prob_all_blue_is_one_twelfth_l1792_179279

/-- The number of balls in the urn -/
def total_balls : ℕ := 10

/-- The number of blue balls in the urn -/
def blue_balls : ℕ := 5

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

/-- Combination function -/
def C (n k : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The probability of drawing all blue balls -/
def prob_all_blue : ℚ := C blue_balls drawn_balls / C total_balls drawn_balls

theorem prob_all_blue_is_one_twelfth : 
  prob_all_blue = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_prob_all_blue_is_one_twelfth_l1792_179279


namespace NUMINAMATH_CALUDE_second_student_marks_l1792_179242

/-- Proves that given two students' marks satisfying specific conditions, 
    the student with the lower score obtained 33 marks. -/
theorem second_student_marks : 
  ∀ (x y : ℝ), 
  x = y + 9 →  -- First student scored 9 marks more
  x = 0.56 * (x + y) →  -- Higher score is 56% of sum
  y = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_second_student_marks_l1792_179242


namespace NUMINAMATH_CALUDE_remaining_sales_to_goal_l1792_179200

def goal : ℕ := 100

def grandmother_sales : ℕ := 5
def uncle_initial_sales : ℕ := 12
def neighbor_initial_sales : ℕ := 8
def mother_friend_sales : ℕ := 25
def cousin_initial_sales : ℕ := 3
def uncle_additional_sales : ℕ := 10
def neighbor_returns : ℕ := 4
def cousin_additional_sales : ℕ := 5

def total_sales : ℕ := 
  grandmother_sales + 
  (uncle_initial_sales + uncle_additional_sales) + 
  (neighbor_initial_sales - neighbor_returns) + 
  mother_friend_sales + 
  (cousin_initial_sales + cousin_additional_sales)

theorem remaining_sales_to_goal : goal - total_sales = 36 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sales_to_goal_l1792_179200


namespace NUMINAMATH_CALUDE_remaining_money_is_24_l1792_179222

/-- Given an initial amount of money, calculates the remaining amount after a series of transactions. -/
def remainingMoney (initialAmount : ℚ) : ℚ :=
  let afterIceCream := initialAmount - 5
  let afterTShirt := afterIceCream / 2
  let afterDeposit := afterTShirt * (4/5)
  afterDeposit

/-- Proves that given an initial amount of $65, the remaining money after transactions is $24. -/
theorem remaining_money_is_24 :
  remainingMoney 65 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_is_24_l1792_179222


namespace NUMINAMATH_CALUDE_milk_carton_delivery_l1792_179296

theorem milk_carton_delivery (total_cartons : ℕ) (damaged_per_customer : ℕ) (total_accepted : ℕ) :
  total_cartons = 400 →
  damaged_per_customer = 60 →
  total_accepted = 160 →
  ∃ (num_customers : ℕ),
    num_customers > 0 ∧
    num_customers * (total_cartons / num_customers - damaged_per_customer) = total_accepted ∧
    num_customers = 4 :=
by sorry

end NUMINAMATH_CALUDE_milk_carton_delivery_l1792_179296


namespace NUMINAMATH_CALUDE_production_days_calculation_l1792_179256

theorem production_days_calculation (n : ℕ) : 
  (∀ k : ℕ, k > 0 → (60 * n + 90) / (n + 1) = 65) → n = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l1792_179256
