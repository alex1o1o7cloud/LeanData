import Mathlib

namespace NUMINAMATH_CALUDE_batch_size_calculation_l443_44375

theorem batch_size_calculation (sample_size : ℕ) (probability : ℚ) (total : ℕ) : 
  sample_size = 30 →
  probability = 1/4 →
  (sample_size : ℚ) / probability = total →
  total = 120 := by
sorry

end NUMINAMATH_CALUDE_batch_size_calculation_l443_44375


namespace NUMINAMATH_CALUDE_aunt_gemma_feeding_times_l443_44302

/-- Calculates the number of times Aunt Gemma feeds her dogs per day -/
def feeding_times_per_day (num_dogs : ℕ) (food_per_meal : ℕ) (num_sacks : ℕ) (sack_weight : ℕ) (days : ℕ) : ℕ :=
  let total_food := num_sacks * sack_weight * 1000
  let food_per_day := total_food / days
  let food_per_dog_per_day := food_per_day / num_dogs
  food_per_dog_per_day / food_per_meal

theorem aunt_gemma_feeding_times : 
  feeding_times_per_day 4 250 2 50 50 = 2 := by sorry

end NUMINAMATH_CALUDE_aunt_gemma_feeding_times_l443_44302


namespace NUMINAMATH_CALUDE_ab_squared_equals_twelve_l443_44323

theorem ab_squared_equals_twelve (a b : ℝ) : (a + 2)^2 + |b - 3| = 0 → a^2 * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_equals_twelve_l443_44323


namespace NUMINAMATH_CALUDE_polynomial_remainder_l443_44326

theorem polynomial_remainder (q : ℝ → ℝ) (h1 : ∃ r1 : ℝ → ℝ, ∀ x, q x = (x - 1) * (r1 x) + 10)
  (h2 : ∃ r2 : ℝ → ℝ, ∀ x, q x = (x + 3) * (r2 x) - 8) :
  ∃ r : ℝ → ℝ, ∀ x, q x = (x - 1) * (x + 3) * (r x) + 4.5 * x + 5.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l443_44326


namespace NUMINAMATH_CALUDE_new_failing_grades_saturday_is_nine_l443_44311

/-- The number of new failing grades that appear on Saturday -/
def new_failing_grades_saturday : ℕ :=
  let group1_students : ℕ := 7
  let group2_students : ℕ := 9
  let days_mon_to_sat : ℕ := 6
  let failing_grades_mon_to_fri : ℕ := 30
  let group1_failing_grades := group1_students * (days_mon_to_sat / 2)
  let group2_failing_grades := group2_students * (days_mon_to_sat / 3)
  let total_failing_grades := group1_failing_grades + group2_failing_grades
  total_failing_grades - failing_grades_mon_to_fri

theorem new_failing_grades_saturday_is_nine :
  new_failing_grades_saturday = 9 := by
  sorry

end NUMINAMATH_CALUDE_new_failing_grades_saturday_is_nine_l443_44311


namespace NUMINAMATH_CALUDE_expression_evaluation_l443_44347

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (x - 1) / (x - 2) * ((x^2 - 4) / (x^2 - 2*x + 1)) - 2 / (x - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l443_44347


namespace NUMINAMATH_CALUDE_first_platform_length_l443_44349

/-- Given a train and two platforms, calculates the length of the first platform. -/
theorem first_platform_length 
  (train_length : ℝ) 
  (first_platform_time : ℝ) 
  (second_platform_length : ℝ) 
  (second_platform_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : first_platform_time = 15)
  (h3 : second_platform_length = 500)
  (h4 : second_platform_time = 20) :
  ∃ L : ℝ, (L + train_length) / first_platform_time = 
           (second_platform_length + train_length) / second_platform_time ∧ 
           L = 350 :=
by sorry

end NUMINAMATH_CALUDE_first_platform_length_l443_44349


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l443_44322

theorem mary_baseball_cards (x : ℕ) : 
  x - 8 + 26 + 40 = 84 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l443_44322


namespace NUMINAMATH_CALUDE_homework_problem_l443_44336

theorem homework_problem (total : ℕ) (ratio_incomplete : ℕ) (ratio_complete : ℕ) 
  (h_total : total = 15)
  (h_ratio : ratio_incomplete = 3 ∧ ratio_complete = 2) :
  ∃ (completed : ℕ), completed = 6 ∧ 
    ratio_incomplete * completed = ratio_complete * (total - completed) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l443_44336


namespace NUMINAMATH_CALUDE_netPopulationIncreaseIs345600_l443_44360

/-- Calculates the net population increase in one day given birth and death rates -/
def netPopulationIncreaseInOneDay (birthRate : ℕ) (deathRate : ℕ) : ℕ :=
  let netIncreasePerTwoSeconds := birthRate - deathRate
  let netIncreasePerSecond := netIncreasePerTwoSeconds / 2
  let secondsInDay : ℕ := 24 * 60 * 60
  netIncreasePerSecond * secondsInDay

/-- Theorem stating that the net population increase in one day is 345,600 given the specified birth and death rates -/
theorem netPopulationIncreaseIs345600 :
  netPopulationIncreaseInOneDay 10 2 = 345600 := by
  sorry

#eval netPopulationIncreaseInOneDay 10 2

end NUMINAMATH_CALUDE_netPopulationIncreaseIs345600_l443_44360


namespace NUMINAMATH_CALUDE_max_ones_in_table_l443_44395

/-- Represents a table with rows and columns -/
structure Table :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the constraints for the table -/
structure TableConstraints :=
  (table : Table)
  (row_sum_mod_3 : ℕ)
  (col_sum_mod_3 : ℕ)

/-- The maximum number of 1's that can be placed in the table -/
def max_ones (constraints : TableConstraints) : ℕ :=
  sorry

/-- The specific constraints for our problem -/
def our_constraints : TableConstraints :=
  { table := { rows := 2005, cols := 2006 },
    row_sum_mod_3 := 0,
    col_sum_mod_3 := 0 }

/-- The theorem to be proved -/
theorem max_ones_in_table :
  max_ones our_constraints = 1336 :=
sorry

end NUMINAMATH_CALUDE_max_ones_in_table_l443_44395


namespace NUMINAMATH_CALUDE_circle_center_l443_44321

/-- The equation of a circle in the x-y plane --/
def CircleEquation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 80 = 0

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with the given equation is (1, -2) --/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 1 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l443_44321


namespace NUMINAMATH_CALUDE_backyard_area_l443_44392

/-- The area of a rectangular backyard given specific walking conditions -/
theorem backyard_area (length width : ℝ) : 
  (20 * length = 800) →
  (8 * (2 * length + 2 * width) = 800) →
  (length * width = 400) := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l443_44392


namespace NUMINAMATH_CALUDE_probability_of_draw_l443_44305

/-- Given two players A and B playing chess, this theorem proves the probability of a draw. -/
theorem probability_of_draw 
  (p_win : ℝ) 
  (p_not_lose : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_not_lose = 0.9) : 
  p_not_lose - p_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_draw_l443_44305


namespace NUMINAMATH_CALUDE_triple_consecutive_primes_l443_44356

theorem triple_consecutive_primes (p : ℤ) : 
  (Nat.Prime p.natAbs ∧ Nat.Prime (p + 2).natAbs ∧ Nat.Prime (p + 4).natAbs) ↔ p = 3 :=
sorry

end NUMINAMATH_CALUDE_triple_consecutive_primes_l443_44356


namespace NUMINAMATH_CALUDE_renovation_theorem_l443_44396

/-- Represents a city grid --/
structure CityGrid where
  rows : Nat
  cols : Nat

/-- Calculates the minimum number of buildings after renovation --/
def minBuildingsAfterRenovation (grid : CityGrid) : Nat :=
  sorry

theorem renovation_theorem :
  (minBuildingsAfterRenovation ⟨20, 20⟩ = 25) ∧
  (minBuildingsAfterRenovation ⟨50, 90⟩ = 282) := by
  sorry

end NUMINAMATH_CALUDE_renovation_theorem_l443_44396


namespace NUMINAMATH_CALUDE_abs_neg_two_l443_44327

theorem abs_neg_two : |(-2 : ℝ)| = 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_two_l443_44327


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l443_44357

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (8*x + 4) * (8*x + 8) * (4*x + 2) = 384 * k) ∧
  (∀ (d : ℤ), d > 384 → ¬(∀ (y : ℤ), Odd y → ∃ (m : ℤ), (8*y + 4) * (8*y + 8) * (4*y + 2) = d * m)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l443_44357


namespace NUMINAMATH_CALUDE_triangle_angle_equality_l443_44398

open Real

theorem triangle_angle_equality (A B : ℝ) (a b : ℝ) 
  (h1 : sin A / a = cos B / b) 
  (h2 : a = b) : 
  B = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_equality_l443_44398


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_97_minus_97_l443_44340

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating that the sum of digits of 10^97 - 97 is 858 -/
theorem sum_of_digits_of_10_pow_97_minus_97 : sum_of_digits (10^97 - 97) = 858 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_97_minus_97_l443_44340


namespace NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l443_44312

theorem inverse_trig_sum_equals_pi : 
  Real.arctan (Real.sqrt 3) - Real.arcsin (-1/2) + Real.arccos 0 = π := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_equals_pi_l443_44312


namespace NUMINAMATH_CALUDE_log_32_2_l443_44342

theorem log_32_2 : Real.log 2 / Real.log 32 = 1 / 5 := by
  have h : 32 = 2^5 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_32_2_l443_44342


namespace NUMINAMATH_CALUDE_darcy_folded_shirts_darcy_problem_l443_44388

theorem darcy_folded_shirts (total_shirts : ℕ) (total_shorts : ℕ) (folded_shorts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  let total_clothing := total_shirts + total_shorts
  let folded_clothing := total_clothing - folded_shorts - remaining_to_fold
  let folded_shirts := folded_clothing - folded_shorts
  folded_shirts

theorem darcy_problem :
  darcy_folded_shirts 20 8 5 11 = 7 := by
  sorry

end NUMINAMATH_CALUDE_darcy_folded_shirts_darcy_problem_l443_44388


namespace NUMINAMATH_CALUDE_unique_positive_solution_l443_44351

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ Real.sqrt ((7 * x) / 3) = x :=
  ⟨7/3, by sorry⟩

end NUMINAMATH_CALUDE_unique_positive_solution_l443_44351


namespace NUMINAMATH_CALUDE_mr_A_net_gain_l443_44365

def initial_cash_A : ℕ := 20000
def initial_house_value : ℕ := 20000
def initial_car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000

def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500

def house_buyback_price : ℕ := 19000
def car_depreciation_rate : ℚ := 1/10
def car_buyback_price : ℕ := 4050

theorem mr_A_net_gain :
  let first_transaction_cash_A := initial_cash_A + house_sale_price + car_sale_price
  let second_transaction_cash_A := first_transaction_cash_A - house_buyback_price - car_buyback_price
  second_transaction_cash_A - initial_cash_A = 2000 := by sorry

end NUMINAMATH_CALUDE_mr_A_net_gain_l443_44365


namespace NUMINAMATH_CALUDE_inequality_solution_l443_44341

theorem inequality_solution (x : ℝ) : 
  ((1 + x) / 2 - (2 * x + 1) / 3 ≤ 1) ↔ (x ≥ -5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l443_44341


namespace NUMINAMATH_CALUDE_bcm_hens_percentage_l443_44331

/-- Given a farm with chickens, calculate the percentage of Black Copper Marans hens -/
theorem bcm_hens_percentage 
  (total_chickens : ℕ) 
  (bcm_percentage : ℚ) 
  (bcm_hens : ℕ) 
  (h1 : total_chickens = 100) 
  (h2 : bcm_percentage = 1/5) 
  (h3 : bcm_hens = 16) : 
  (bcm_hens : ℚ) / (bcm_percentage * total_chickens) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_bcm_hens_percentage_l443_44331


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l443_44344

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 10/21) 
  (h2 : x - y = 1/63) : 
  x^2 - y^2 = 10/1323 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l443_44344


namespace NUMINAMATH_CALUDE_specific_ohara_triple_l443_44309

/-- O'Hara triple condition -/
def is_ohara_triple (a b x : ℝ) : Prop := Real.sqrt a + Real.sqrt b = x

/-- Proof of the specific O'Hara triple -/
theorem specific_ohara_triple :
  let a : ℝ := 49
  let b : ℝ := 16
  ∃ x : ℝ, is_ohara_triple a b x ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_specific_ohara_triple_l443_44309


namespace NUMINAMATH_CALUDE_bruce_pizza_production_l443_44381

/-- The number of batches of pizza dough Bruce can make in a week -/
def pizzas_per_week (batches_per_sack : ℕ) (sacks_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  batches_per_sack * sacks_per_day * days_in_week

/-- Theorem stating that Bruce can make 525 batches of pizza dough in a week -/
theorem bruce_pizza_production :
  pizzas_per_week 15 5 7 = 525 := by
  sorry

end NUMINAMATH_CALUDE_bruce_pizza_production_l443_44381


namespace NUMINAMATH_CALUDE_calculation_proof_l443_44330

theorem calculation_proof :
  (13 + (-7) + (-6) = 0) ∧
  ((-8) * (-4/3) * (-0.125) * (5/4) = -5/3) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l443_44330


namespace NUMINAMATH_CALUDE_wire_circle_square_area_l443_44370

/-- The area of a square formed by a wire that can also form a circle of radius 56 cm is 784π² cm² -/
theorem wire_circle_square_area :
  let r : ℝ := 56  -- radius of the circle in cm
  let circle_circumference : ℝ := 2 * Real.pi * r
  let square_side : ℝ := circle_circumference / 4
  let square_area : ℝ := square_side * square_side
  square_area = 784 * Real.pi ^ 2 := by
    sorry

end NUMINAMATH_CALUDE_wire_circle_square_area_l443_44370


namespace NUMINAMATH_CALUDE_quadratic_downward_solution_nonempty_l443_44345

/-- A quadratic function f(x) = ax² + bx + c opens downwards if a < 0 -/
def opens_downwards (a b c : ℝ) : Prop := a < 0

/-- The solution set of ax² + bx + c < 0 is not empty -/
def solution_set_nonempty (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c < 0

/-- If a quadratic function opens downwards, its solution set for f(x) < 0 is not empty -/
theorem quadratic_downward_solution_nonempty (a b c : ℝ) :
  opens_downwards a b c → solution_set_nonempty a b c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_downward_solution_nonempty_l443_44345


namespace NUMINAMATH_CALUDE_fraction_product_cube_l443_44363

theorem fraction_product_cube (x : ℝ) (hx : x ≠ 0) : 
  (8 / 9)^3 * (x / 3)^3 * (3 / x)^3 = 512 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_cube_l443_44363


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l443_44300

/-- The number of squares in the nth figure -/
def f (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern -/
axiom pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37

/-- The number of squares in figure 50 is 7651 -/
theorem squares_in_figure_50 : f 50 = 7651 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_figure_50_l443_44300


namespace NUMINAMATH_CALUDE_winningScoresCount_is_nineteen_l443_44319

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- Number of runners in each team -/
  runnersPerTeam : Nat
  /-- Total number of runners -/
  totalRunners : Nat
  /-- The sum of all positions -/
  totalPositionSum : Nat
  /-- Condition that there are 2 teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam
  /-- Condition that the total sum of positions is correct -/
  validTotalSum : totalPositionSum = totalRunners * (totalRunners + 1) / 2

/-- The number of different winning scores possible in a cross country meet -/
def winningScoresCount (meet : CrossCountryMeet) : Nat :=
  meet.totalPositionSum / 2 - (meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2) + 1

/-- Theorem stating that the number of different winning scores is 19 -/
theorem winningScoresCount_is_nineteen :
  ∀ (meet : CrossCountryMeet),
    meet.runnersPerTeam = 6 →
    winningScoresCount meet = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_winningScoresCount_is_nineteen_l443_44319


namespace NUMINAMATH_CALUDE_square_of_negative_product_l443_44308

theorem square_of_negative_product (x y : ℝ) : (-x * y^2)^2 = x^2 * y^4 := by sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l443_44308


namespace NUMINAMATH_CALUDE_pushups_sum_is_350_l443_44354

/-- The number of push-ups done by Zachary, David, and Emily -/
def total_pushups (zachary_pushups : ℕ) (david_extra : ℕ) : ℕ :=
  let david_pushups := zachary_pushups + david_extra
  let emily_pushups := 2 * david_pushups
  zachary_pushups + david_pushups + emily_pushups

/-- Theorem stating that the total number of push-ups is 350 -/
theorem pushups_sum_is_350 : total_pushups 44 58 = 350 := by
  sorry

end NUMINAMATH_CALUDE_pushups_sum_is_350_l443_44354


namespace NUMINAMATH_CALUDE_youngest_child_age_l443_44358

theorem youngest_child_age (age1 age2 age3 : ℕ) : 
  age1 < age2 ∧ age2 < age3 →
  6 + (0.60 * (age1 + age2 + age3 : ℝ)) + (3 * 0.90) = 15.30 →
  age1 = 1 :=
sorry

end NUMINAMATH_CALUDE_youngest_child_age_l443_44358


namespace NUMINAMATH_CALUDE_recipe_total_cups_l443_44328

/-- The total number of cups needed for a recipe with cereal, milk, and nuts. -/
def total_cups (cereal_servings milk_servings nuts_servings : ℝ)
               (cereal_cups_per_serving milk_cups_per_serving nuts_cups_per_serving : ℝ) : ℝ :=
  cereal_servings * cereal_cups_per_serving +
  milk_servings * milk_cups_per_serving +
  nuts_servings * nuts_cups_per_serving

/-- Theorem stating that the total cups needed for the given recipe is 57.0 cups. -/
theorem recipe_total_cups :
  total_cups 18.0 12.0 6.0 2.0 1.5 0.5 = 57.0 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l443_44328


namespace NUMINAMATH_CALUDE_bench_cost_proof_l443_44334

/-- The cost of the bench in dollars -/
def bench_cost : ℝ := 150

/-- The cost of the garden table in dollars -/
def table_cost : ℝ := 2 * bench_cost

/-- The combined cost of the bench and garden table in dollars -/
def combined_cost : ℝ := 450

theorem bench_cost_proof : bench_cost = 150 := by sorry

end NUMINAMATH_CALUDE_bench_cost_proof_l443_44334


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l443_44355

theorem binomial_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l443_44355


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l443_44379

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l443_44379


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l443_44315

/-- Given a parabola with equation x = (1/(4*m)) * y^2, prove that its focus has coordinates (m, 0) -/
theorem parabola_focus_coordinates (m : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | x = (1 / (4 * m)) * y^2}
  let focus := (m, 0)
  focus ∈ parabola ∧ ∀ p ∈ parabola, ∃ d : ℝ, (p.1 - focus.1)^2 + (p.2 - focus.2)^2 = d^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l443_44315


namespace NUMINAMATH_CALUDE_min_d_value_l443_44314

theorem min_d_value (a b d : ℕ+) (h1 : a < b) (h2 : b < d + 1)
  (h3 : ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
       p.2 = |p.1 - a| + |p.1 - b| + |p.1 - (d + 1)|) :
  (∀ d' : ℕ+, d' ≥ d → 
    ∃ a' b' : ℕ+, a' < b' ∧ b' < d' + 1 ∧
    ∃! p : ℝ × ℝ, p.1 * 2 + p.2 = 2023 ∧ 
    p.2 = |p.1 - a'| + |p.1 - b'| + |p.1 - (d' + 1)|) →
  d = 2020 := by
sorry

end NUMINAMATH_CALUDE_min_d_value_l443_44314


namespace NUMINAMATH_CALUDE_pentagon_area_half_octagon_l443_44318

/-- Regular octagon with vertices labeled CHILDREN -/
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : sorry)

/-- Pentagon formed by 5 consecutive vertices of the octagon -/
def Pentagon (o : RegularOctagon) : Set (ℝ × ℝ) :=
  {p | ∃ i : Fin 5, p = o.vertices i}

/-- Area of a shape in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

theorem pentagon_area_half_octagon (o : RegularOctagon) 
  (h : area {p | ∃ i : Fin 8, p = o.vertices i} = 1) : 
  area (Pentagon o) = 1/2 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_half_octagon_l443_44318


namespace NUMINAMATH_CALUDE_smallest_valid_n_l443_44303

def is_valid_arrangement (n : ℕ) (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i j, (i.val - j.val) % 9 ≥ 2 → n ∣ a i * a j) ∧
  (∀ i, ¬(n ∣ a i * a (i + 1)))

theorem smallest_valid_n : 
  (∃ (a : Fin 9 → ℕ), is_valid_arrangement 485100 a) ∧
  (∀ n < 485100, ¬∃ (a : Fin 9 → ℕ), is_valid_arrangement n a) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l443_44303


namespace NUMINAMATH_CALUDE_exists_special_function_l443_44307

def I : Set ℝ := Set.Icc (-1) 1

def is_piecewise_continuous (f : ℝ → ℝ) : Prop :=
  ∃ (s : Set ℝ), Set.Finite s ∧
  ∀ x ∈ I, x ∉ s → ∃ ε > 0, ∀ y ∈ I, |y - x| < ε → f y = f x

theorem exists_special_function :
  ∃ f : ℝ → ℝ,
    (∀ x ∈ I, f (f x) = -x) ∧
    (∀ x ∉ I, f x = 0) ∧
    is_piecewise_continuous f :=
sorry

end NUMINAMATH_CALUDE_exists_special_function_l443_44307


namespace NUMINAMATH_CALUDE_square_side_equals_pi_l443_44374

theorem square_side_equals_pi : ∃ x : ℝ, 
  4 * x = 2 * π * 2 ∧ x = π := by
  sorry

end NUMINAMATH_CALUDE_square_side_equals_pi_l443_44374


namespace NUMINAMATH_CALUDE_rectangle_diagonal_rectangle_diagonal_proof_l443_44366

/-- The length of the diagonal of a rectangle with length 100 and width 100√2 is 100√3 -/
theorem rectangle_diagonal : Real → Prop :=
  fun d =>
    let length := 100
    let width := 100 * Real.sqrt 2
    d = 100 * Real.sqrt 3 ∧ d^2 = length^2 + width^2

/-- Proof of the theorem -/
theorem rectangle_diagonal_proof : ∃ d, rectangle_diagonal d := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_rectangle_diagonal_proof_l443_44366


namespace NUMINAMATH_CALUDE_greatest_divisor_of_consecutive_multiples_of_four_l443_44304

theorem greatest_divisor_of_consecutive_multiples_of_four : ∃ (k : ℕ), 
  k > 0 ∧ 
  (∀ (n : ℕ), 
    (4*n * 4*(n+1) * 4*(n+2)) % k = 0) ∧
  (∀ (m : ℕ), 
    m > k → 
    ∃ (n : ℕ), (4*n * 4*(n+1) * 4*(n+2)) % m ≠ 0) ∧
  k = 768 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_consecutive_multiples_of_four_l443_44304


namespace NUMINAMATH_CALUDE_certain_amount_less_than_twice_l443_44310

theorem certain_amount_less_than_twice (n : ℤ) (x : ℤ) : n = 16 ∧ 2 * n - x = 20 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_less_than_twice_l443_44310


namespace NUMINAMATH_CALUDE_divisible_by_three_divisible_by_nine_l443_44361

/-- Represents the decimal digits of a non-negative integer -/
def DecimalDigits : Type := List Nat

/-- Returns the sum of digits in a DecimalDigits representation -/
def sum_of_digits (digits : DecimalDigits) : Nat :=
  digits.sum

/-- Converts a non-negative integer to its DecimalDigits representation -/
def to_decimal_digits (n : Nat) : DecimalDigits :=
  sorry

/-- Converts a DecimalDigits representation back to the original number -/
def from_decimal_digits (digits : DecimalDigits) : Nat :=
  sorry

/-- Theorem: A number is divisible by 3 iff the sum of its digits is divisible by 3 -/
theorem divisible_by_three (n : Nat) :
  n % 3 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 3 = 0 :=
  sorry

/-- Theorem: A number is divisible by 9 iff the sum of its digits is divisible by 9 -/
theorem divisible_by_nine (n : Nat) :
  n % 9 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 9 = 0 :=
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_divisible_by_nine_l443_44361


namespace NUMINAMATH_CALUDE_box_ball_count_l443_44301

theorem box_ball_count (red_balls : ℕ) (red_prob : ℚ) (total_balls : ℕ) : 
  red_balls = 12 → red_prob = 3/5 → (red_balls : ℚ) / total_balls = red_prob → total_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_box_ball_count_l443_44301


namespace NUMINAMATH_CALUDE_regions_99_lines_l443_44337

/-- The number of regions created by lines in a plane -/
def num_regions (num_lines : ℕ) (all_parallel : Bool) (all_intersect_one_point : Bool) : ℕ :=
  if all_parallel then
    num_lines + 1
  else if all_intersect_one_point then
    2 * num_lines
  else
    0  -- This case is not used in our theorem, but included for completeness

/-- Theorem stating the possible number of regions created by 99 lines in a plane -/
theorem regions_99_lines :
  ∀ (n : ℕ), n < 199 →
  (∃ (all_parallel all_intersect_one_point : Bool),
    num_regions 99 all_parallel all_intersect_one_point = n) →
  (n = 100 ∨ n = 198) :=
by
  sorry

#check regions_99_lines

end NUMINAMATH_CALUDE_regions_99_lines_l443_44337


namespace NUMINAMATH_CALUDE_length_A_l443_44391

-- Define the points
def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' on the line y = x
def A' : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry

axiom A'_on_line : line_y_eq_x A'
axiom B'_on_line : line_y_eq_x B'

-- Define the lines AA' and BB'
def line_AA' (x : ℝ) : ℝ := sorry
def line_BB' (x : ℝ) : ℝ := sorry

-- C is on both lines AA' and BB'
axiom C_on_AA' : line_AA' C.1 = C.2
axiom C_on_BB' : line_BB' C.1 = C.2

-- Theorem: The length of A'B' is 2√2
theorem length_A'B'_is_2_sqrt_2 : 
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_length_A_l443_44391


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_condition_l443_44316

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ q = a / b ∧ ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

theorem finite_decimal_fraction_condition (n : ℕ) :
  n > 0 → (is_finite_decimal (1 / (n * (n + 1))) ↔ n = 1 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_condition_l443_44316


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l443_44317

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 3/x ≥ (3/2) * Real.rpow 18 (1/3) :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, x^2 + 3/x = (3/2) * Real.rpow 18 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l443_44317


namespace NUMINAMATH_CALUDE_inequality_impossibility_l443_44394

theorem inequality_impossibility (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(a > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_impossibility_l443_44394


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l443_44393

-- Define the doubling time in seconds
def doubling_time : ℕ := 15

-- Define the total time in seconds
def total_time : ℕ := 4 * 60

-- Define the final number of bacteria
def final_bacteria : ℕ := 2097152

-- Theorem statement
theorem initial_bacteria_count :
  ∃ (initial : ℕ), 
    initial * (2 ^ (total_time / doubling_time)) = final_bacteria ∧
    initial = 32 := by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l443_44393


namespace NUMINAMATH_CALUDE_range_of_2x_minus_y_l443_44373

theorem range_of_2x_minus_y (x y : ℝ) (hx : 2 < x ∧ x < 4) (hy : -1 < y ∧ y < 3) :
  1 < 2 * x - y ∧ 2 * x - y < 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2x_minus_y_l443_44373


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l443_44383

theorem geometric_progression_proof (x : ℝ) (r : ℝ) : 
  (((30 + x) / (10 + x) = r) ∧ ((90 + x) / (30 + x) = r)) ↔ (x = 0 ∧ r = 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l443_44383


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l443_44362

theorem candidate_vote_percentage
  (total_votes : ℝ)
  (vote_difference : ℝ)
  (h_total : total_votes = 25000.000000000007)
  (h_diff : vote_difference = 5000) :
  let candidate_percentage := (total_votes - vote_difference) / (2 * total_votes) * 100
  candidate_percentage = 40 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l443_44362


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_plus_c_l443_44386

theorem value_of_a_minus_b_plus_c 
  (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_plus_c_l443_44386


namespace NUMINAMATH_CALUDE_girls_boys_acquaintance_l443_44380

theorem girls_boys_acquaintance (n : ℕ) :
  n > 1 →
  (∃ (girls_know : Fin (n + 1) → Fin (n + 1)) (boys_know : Fin n → ℕ),
    Function.Injective girls_know ∧
    (∀ i : Fin n, boys_know i = (n + 1) / 2) ∧
    (∀ i : Fin (n + 1), girls_know i ≤ n)) →
  Odd n :=
by sorry

end NUMINAMATH_CALUDE_girls_boys_acquaintance_l443_44380


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_16_l443_44350

/-- A regular six-pointed star -/
structure RegularSixPointedStar :=
  (area : ℝ)
  (star_formed_by_two_triangles : Bool)
  (each_triangle_area : ℝ)

/-- The area of a quadrilateral formed by two adjacent points and the center of the star -/
def quadrilateral_area (star : RegularSixPointedStar) : ℝ := sorry

/-- Theorem stating the area of the quadrilateral is 16 cm² -/
theorem quadrilateral_area_is_16 (star : RegularSixPointedStar) 
  (h1 : star.star_formed_by_two_triangles = true) 
  (h2 : star.each_triangle_area = 72) : quadrilateral_area star = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_16_l443_44350


namespace NUMINAMATH_CALUDE_y_work_time_l443_44377

/-- Given workers x, y, and z, and their work rates, prove that y alone takes 24 hours to complete the work. -/
theorem y_work_time (x y z : ℝ) (hx : x = 1 / 8) (hyz : y + z = 1 / 6) (hxz : x + z = 1 / 4) :
  1 / y = 24 := by
  sorry

end NUMINAMATH_CALUDE_y_work_time_l443_44377


namespace NUMINAMATH_CALUDE_total_passengers_is_120_l443_44338

/-- The total number of passengers on the flight -/
def total_passengers : ℕ := 120

/-- The proportion of female passengers -/
def female_proportion : ℚ := 55 / 100

/-- The proportion of passengers in first class -/
def first_class_proportion : ℚ := 10 / 100

/-- The proportion of male passengers in first class -/
def male_first_class_proportion : ℚ := 1 / 3

/-- The number of females in coach class -/
def females_in_coach : ℕ := 58

/-- Theorem stating that the total number of passengers is 120 -/
theorem total_passengers_is_120 : 
  total_passengers = 120 ∧
  female_proportion * total_passengers = 
    (females_in_coach : ℚ) + 
    (1 - male_first_class_proportion) * first_class_proportion * total_passengers :=
by sorry

end NUMINAMATH_CALUDE_total_passengers_is_120_l443_44338


namespace NUMINAMATH_CALUDE_inequalities_representation_l443_44376

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a shape in 2D space -/
class Shape where
  contains : Point2D → Prop

/-- Diamond shape defined by |x| + |y| ≤ r -/
def Diamond (r : ℝ) : Shape where
  contains p := abs p.x + abs p.y ≤ r

/-- Circle shape defined by x² + y² ≤ r² -/
def Circle (r : ℝ) : Shape where
  contains p := p.x^2 + p.y^2 ≤ r^2

/-- Hexagon shape defined by 3Max(|x|, |y|) ≤ r -/
def Hexagon (r : ℝ) : Shape where
  contains p := 3 * max (abs p.x) (abs p.y) ≤ r

/-- Theorem stating that the inequalities represent the described geometric shapes -/
theorem inequalities_representation (r : ℝ) (p : Point2D) :
  (Diamond r).contains p → (Circle r).contains p → (Hexagon r).contains p :=
by sorry

end NUMINAMATH_CALUDE_inequalities_representation_l443_44376


namespace NUMINAMATH_CALUDE_count_valid_formations_l443_44352

/-- The total number of musicians in the marching band -/
def total_musicians : ℕ := 420

/-- The minimum number of musicians per row -/
def min_per_row : ℕ := 12

/-- The maximum number of musicians per row -/
def max_per_row : ℕ := 50

/-- A function that checks if a pair of positive integers (s, t) forms a valid rectangular formation -/
def is_valid_formation (s t : ℕ+) : Prop :=
  s * t = total_musicians ∧ min_per_row ≤ t ∧ t ≤ max_per_row

/-- The theorem stating that there are exactly 8 valid rectangular formations -/
theorem count_valid_formations :
  ∃! (formations : Finset (ℕ+ × ℕ+)),
    formations.card = 8 ∧
    ∀ pair : ℕ+ × ℕ+, pair ∈ formations ↔ is_valid_formation pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_count_valid_formations_l443_44352


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l443_44306

theorem sin_negative_1740_degrees : Real.sin (-(1740 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l443_44306


namespace NUMINAMATH_CALUDE_trees_left_unwatered_l443_44390

theorem trees_left_unwatered :
  let total_trees : ℕ := 29
  let boys_watering : ℕ := 9
  let trees_watered_per_boy : List ℕ := [2, 3, 1, 3, 2, 4, 3, 2, 5]
  let total_watered : ℕ := trees_watered_per_boy.sum
  total_trees - total_watered = 4 := by
sorry

end NUMINAMATH_CALUDE_trees_left_unwatered_l443_44390


namespace NUMINAMATH_CALUDE_eight_teams_twentyeight_games_unique_solution_eight_teams_l443_44329

/-- The number of games played when each team in a conference plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 8 teams results in 28 games played -/
theorem eight_teams_twentyeight_games :
  ∃ (n : ℕ), n > 0 ∧ games_played n = 28 ∧ n = 8 := by
  sorry

/-- The theorem proving that 8 is the only positive integer satisfying the conditions -/
theorem unique_solution_eight_teams :
  ∀ (n : ℕ), n > 0 → games_played n = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_teams_twentyeight_games_unique_solution_eight_teams_l443_44329


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l443_44324

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a * b / 2 = 18) →  -- Area of smaller triangle
  (c * d / 2 = 288) →  -- Area of larger triangle
  (a^2 + b^2 = 10^2) →  -- Pythagorean theorem for smaller triangle
  (c / a = d / b) →  -- Similar triangles condition
  (c + d = 52) := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l443_44324


namespace NUMINAMATH_CALUDE_lonely_island_turtles_l443_44399

theorem lonely_island_turtles : 
  ∀ (happy_island lonely_island : ℕ),
  happy_island = 60 →
  happy_island = 2 * lonely_island + 10 →
  lonely_island = 25 := by
sorry

end NUMINAMATH_CALUDE_lonely_island_turtles_l443_44399


namespace NUMINAMATH_CALUDE_inequality_solution_set_l443_44346

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 0 → 
  2 - x ≥ 0 → 
  (((Real.sqrt (2 - x) + 4 * x - 3) / x ≥ 2) ↔ (x < 0 ∨ (1 ≤ x ∧ x ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l443_44346


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l443_44333

/-- Given a geometric sequence {a_n} with sum S_n = 2010^n + t, prove a_1 = 2009 -/
theorem geometric_sequence_first_term (n : ℕ) (t : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ k, S k = 2010^k + t) →
  (a 1 * a 3 = (a 2)^2) →
  a 1 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l443_44333


namespace NUMINAMATH_CALUDE_intersection_of_sets_l443_44378

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l443_44378


namespace NUMINAMATH_CALUDE_quadratic_function_property_l443_44332

/-- Given a quadratic function f(x) = ax^2 + bx + c with a ≠ 0,
    if f(r) = f(s) = k for two distinct points r and s,
    then f(r + s) = c -/
theorem quadratic_function_property
  (a b c r s k : ℝ)
  (h_a : a ≠ 0)
  (h_distinct : r ≠ s)
  (h_fr : a * r^2 + b * r + c = k)
  (h_fs : a * s^2 + b * s + c = k) :
  a * (r + s)^2 + b * (r + s) + c = c :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l443_44332


namespace NUMINAMATH_CALUDE_class_average_score_l443_44382

theorem class_average_score (total_questions : ℕ) 
  (score_3_percent : ℝ) (score_2_percent : ℝ) (score_1_percent : ℝ) (score_0_percent : ℝ) :
  total_questions = 3 →
  score_3_percent = 0.3 →
  score_2_percent = 0.5 →
  score_1_percent = 0.1 →
  score_0_percent = 0.1 →
  score_3_percent + score_2_percent + score_1_percent + score_0_percent = 1 →
  3 * score_3_percent + 2 * score_2_percent + 1 * score_1_percent + 0 * score_0_percent = 2 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l443_44382


namespace NUMINAMATH_CALUDE_polynomial_sum_l443_44397

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 160 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l443_44397


namespace NUMINAMATH_CALUDE_factory_growth_rate_l443_44325

theorem factory_growth_rate (P a b x : ℝ) (h1 : P > 0) (h2 : a > -1) (h3 : b > -1)
  (h4 : (1 + x)^2 = (1 + a) * (1 + b)) : x ≤ max a b := by
  sorry

end NUMINAMATH_CALUDE_factory_growth_rate_l443_44325


namespace NUMINAMATH_CALUDE_angle_three_measure_l443_44371

def minutes_to_degrees (m : ℕ) : ℚ := m / 60

def angle_measure (degrees : ℕ) (minutes : ℕ) : ℚ := degrees + minutes_to_degrees minutes

theorem angle_three_measure 
  (angle1 angle2 angle3 : ℚ) 
  (h1 : angle1 + angle2 = 90) 
  (h2 : angle2 + angle3 = 180) 
  (h3 : angle1 = angle_measure 67 12) : 
  angle3 = angle_measure 157 12 := by
sorry

end NUMINAMATH_CALUDE_angle_three_measure_l443_44371


namespace NUMINAMATH_CALUDE_circle_c_equation_l443_44385

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- Define the line l: x - y - 1 = 0
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_c_equation :
  ∃ (c : Circle),
    (circle_equation c A.1 A.2) ∧
    (line_l B.1 B.2) ∧
    (∀ (x y : ℝ), line_l x y → (x - B.1)^2 + (y - B.2)^2 ≤ c.radius^2) ∧
    (circle_equation c x y ↔ (x - 3)^2 + y^2 = 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_c_equation_l443_44385


namespace NUMINAMATH_CALUDE_plate_color_probability_l443_44364

/-- The probability of selecting two plates of the same color -/
def same_color_probability (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let same_color_combinations := (red.choose 2) + (blue.choose 2) + (yellow.choose 2)
  let total_combinations := total.choose 2
  same_color_combinations / total_combinations

/-- Theorem: The probability of selecting two plates of the same color
    given 7 red plates, 5 blue plates, and 3 yellow plates is 34/105 -/
theorem plate_color_probability :
  same_color_probability 7 5 3 = 34 / 105 := by
  sorry

end NUMINAMATH_CALUDE_plate_color_probability_l443_44364


namespace NUMINAMATH_CALUDE_prob_not_all_same_value_l443_44320

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 8-sided dice won't all show the same number -/
def prob_not_all_same : ℚ :=
  1 - (sides : ℚ) / sides ^ num_dice

theorem prob_not_all_same_value :
  prob_not_all_same = 4095 / 4096 :=
sorry

end NUMINAMATH_CALUDE_prob_not_all_same_value_l443_44320


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l443_44369

def geometric_series (n : ℕ) : ℚ := (7 / 3) * (7 / 3) ^ n

theorem common_ratio_of_geometric_series :
  ∀ n : ℕ, geometric_series (n + 1) / geometric_series n = 7 / 3 :=
by
  sorry

#check common_ratio_of_geometric_series

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l443_44369


namespace NUMINAMATH_CALUDE_cars_cannot_meet_l443_44313

-- Define the network structure
structure TriangleNetwork where
  -- Assume the network is infinite and regular
  -- Each vertex has exactly 6 edges connected to it
  vertex_degree : ℕ
  vertex_degree_eq : vertex_degree = 6

-- Define a car's position and movement
structure Car where
  position : ℕ × ℕ  -- Represent position as discrete coordinates
  direction : ℕ     -- 0, 1, or 2 representing the three possible directions

-- Define the movement options
inductive Move
  | straight
  | left
  | right

-- Function to update car position based on move
def update_position (c : Car) (m : Move) : Car :=
  sorry  -- Implementation details omitted for brevity

-- Theorem statement
theorem cars_cannot_meet 
  (network : TriangleNetwork) 
  (car1 car2 : Car) 
  (start_same_edge : car1.position.1 = car2.position.1 ∧ car1.direction = car2.direction)
  (t : ℕ) :
  ∀ (moves1 moves2 : List Move),
  moves1.length = t ∧ moves2.length = t →
  (moves1.foldl update_position car1).position ≠ (moves2.foldl update_position car2).position :=
sorry

end NUMINAMATH_CALUDE_cars_cannot_meet_l443_44313


namespace NUMINAMATH_CALUDE_minimum_guests_l443_44348

theorem minimum_guests (total_food : ℝ) (max_individual_food : ℝ) (h1 : total_food = 520) (h2 : max_individual_food = 1.5) :
  ∃ n : ℕ, n * max_individual_food ≥ total_food ∧ ∀ m : ℕ, m * max_individual_food ≥ total_food → m ≥ n ∧ n = 347 :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_l443_44348


namespace NUMINAMATH_CALUDE_binomial_divisibility_l443_44359

theorem binomial_divisibility (n k : ℕ) (h : k ≤ n - 1) :
  (∀ k ≤ n - 1, n ∣ Nat.choose n k) ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l443_44359


namespace NUMINAMATH_CALUDE_power_of_1024_is_16_l443_44368

theorem power_of_1024_is_16 :
  (1024 : ℝ) ^ (2/5 : ℝ) = 16 :=
by
  have h : 1024 = 2^10 := by norm_num
  sorry

end NUMINAMATH_CALUDE_power_of_1024_is_16_l443_44368


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l443_44387

/-- The area of a circle with diameter endpoints C(5,9) and D(13,17) is 32π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (5, 9)
  let d : ℝ × ℝ := (13, 17)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius_squared := diameter_squared / 4
  π * radius_squared = 32 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l443_44387


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l443_44343

theorem geometric_sequence_eighth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^5 = 32) 
  (h3 : r > 0) : 
  a * r^7 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l443_44343


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l443_44389

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x < -2 ∨ x > 2}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.compl B) = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l443_44389


namespace NUMINAMATH_CALUDE_geometric_series_sum_l443_44367

theorem geometric_series_sum :
  let a : ℕ := 1  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let series_sum := (a * (r^n - 1)) / (r - 1)
  series_sum = 3280 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l443_44367


namespace NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l443_44353

theorem unique_m_satisfying_lcm_conditions : 
  ∃! m : ℕ+, (Nat.lcm 36 m.val = 180) ∧ (Nat.lcm m.val 45 = 225) ∧ (m.val = 25) :=
by sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_lcm_conditions_l443_44353


namespace NUMINAMATH_CALUDE_problem_statement_l443_44335

theorem problem_statement :
  ((-2023)^0 : ℝ) - 4 * Real.sin (π/4) + |(-Real.sqrt 8)| = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l443_44335


namespace NUMINAMATH_CALUDE_carly_job_applications_l443_44384

/-- The number of job applications Carly sent to companies in her state -/
def in_state_applications : ℕ := 200

/-- The number of job applications Carly sent to companies in other states -/
def out_state_applications : ℕ := 2 * in_state_applications

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := in_state_applications + out_state_applications

theorem carly_job_applications : total_applications = 600 := by
  sorry

end NUMINAMATH_CALUDE_carly_job_applications_l443_44384


namespace NUMINAMATH_CALUDE_average_page_count_l443_44372

theorem average_page_count (n : ℕ) (g1 g2 g3 : ℕ) (p1 p2 p3 : ℕ) :
  n = g1 + g2 + g3 →
  g1 = g2 →
  g2 = g3 →
  g1 = 5 →
  p1 = 2 →
  p2 = 3 →
  p3 = 1 →
  (g1 * p1 + g2 * p2 + g3 * p3) / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_page_count_l443_44372


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l443_44339

theorem sin_pi_minus_alpha (α : Real) : 
  (∃ (x y : Real), x = Real.sqrt 3 ∧ y = 1 ∧ x = Real.tan α * y) →
  Real.sin (Real.pi - α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l443_44339
