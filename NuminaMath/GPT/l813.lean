import Mathlib

namespace NUMINAMATH_GPT_percentage_less_than_l813_81374

theorem percentage_less_than (p j t : ℝ) (h1 : j = 0.75 * p) (h2 : j = 0.80 * t) : 
  t = (1 - 0.0625) * p := 
by 
  sorry

end NUMINAMATH_GPT_percentage_less_than_l813_81374


namespace NUMINAMATH_GPT_cube_sum_l813_81367

theorem cube_sum (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_GPT_cube_sum_l813_81367


namespace NUMINAMATH_GPT_range_of_a_l813_81331

noncomputable def f (a x : ℝ) := a * Real.log x + x - 1

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x → f a x ≥ 0) : a ≥ -1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l813_81331


namespace NUMINAMATH_GPT_christine_savings_l813_81345

/-- Christine's commission rate as a percentage. -/
def commissionRate : ℝ := 0.12

/-- Total sales made by Christine this month in dollars. -/
def totalSales : ℝ := 24000

/-- Percentage of commission allocated to personal needs. -/
def personalNeedsRate : ℝ := 0.60

/-- The amount Christine saved this month. -/
def amountSaved : ℝ := 1152

/--
Given the commission rate, total sales, and personal needs rate,
prove the amount saved is correctly calculated.
-/
theorem christine_savings :
  (1 - personalNeedsRate) * (commissionRate * totalSales) = amountSaved :=
by
  sorry

end NUMINAMATH_GPT_christine_savings_l813_81345


namespace NUMINAMATH_GPT_chocolate_candy_cost_l813_81318

-- Define the constants and conditions
def cost_per_box : ℕ := 5
def candies_per_box : ℕ := 30
def discount_rate : ℝ := 0.1

-- Define the total number of candies to buy
def total_candies : ℕ := 450

-- Define the threshold for applying discount
def discount_threshold : ℕ := 300

-- Calculate the number of boxes needed
def boxes_needed (total_candies : ℕ) (candies_per_box : ℕ) : ℕ :=
  total_candies / candies_per_box

-- Calculate the total cost without discount
def total_cost (boxes_needed : ℕ) (cost_per_box : ℕ) : ℝ :=
  boxes_needed * cost_per_box

-- Calculate the discounted cost
def discounted_cost (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  if total_candies > discount_threshold then
    total_cost * (1 - discount_rate)
  else
    total_cost

-- Statement to be proved
theorem chocolate_candy_cost :
  discounted_cost 
    (total_cost (boxes_needed total_candies candies_per_box) cost_per_box) 
    discount_rate = 67.5 :=
by
  -- Proof is needed here, using the correct steps from the solution.
  sorry

end NUMINAMATH_GPT_chocolate_candy_cost_l813_81318


namespace NUMINAMATH_GPT_fraction_of_90_l813_81353

theorem fraction_of_90 : (1 / 2) * (1 / 3) * (1 / 6) * (90 : ℝ) = (5 / 2) := by
  sorry

end NUMINAMATH_GPT_fraction_of_90_l813_81353


namespace NUMINAMATH_GPT_percentage_increase_l813_81399

theorem percentage_increase (X Y Z : ℝ) (h1 : X = 1.25 * Y) (h2 : Z = 100) (h3 : X + Y + Z = 370) :
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l813_81399


namespace NUMINAMATH_GPT_travel_options_l813_81321

-- Define the conditions
def trains_from_A_to_B := 3
def ferries_from_B_to_C := 2

-- State the proof problem
theorem travel_options (t : ℕ) (f : ℕ) (h1 : t = trains_from_A_to_B) (h2 : f = ferries_from_B_to_C) : t * f = 6 :=
by
  rewrite [h1, h2]
  sorry

end NUMINAMATH_GPT_travel_options_l813_81321


namespace NUMINAMATH_GPT_raspberry_pie_degrees_l813_81382

def total_students : ℕ := 48
def chocolate_preference : ℕ := 18
def apple_preference : ℕ := 10
def blueberry_preference : ℕ := 8
def remaining_students : ℕ := total_students - chocolate_preference - apple_preference - blueberry_preference
def raspberry_preference : ℕ := remaining_students / 2
def pie_chart_degrees : ℕ := (raspberry_preference * 360) / total_students

theorem raspberry_pie_degrees :
  pie_chart_degrees = 45 := by
  sorry

end NUMINAMATH_GPT_raspberry_pie_degrees_l813_81382


namespace NUMINAMATH_GPT_total_children_is_11_l813_81312

noncomputable def num_of_children (b g : ℕ) := b + g

theorem total_children_is_11 (b g : ℕ) :
  (∃ c : ℕ, b * c + g * (c + 1) = 47) ∧
  (∃ m : ℕ, b * (m + 1) + g * m = 74) → 
  num_of_children b g = 11 :=
by
  -- The proof steps would go here to show that b + g = 11
  sorry

end NUMINAMATH_GPT_total_children_is_11_l813_81312


namespace NUMINAMATH_GPT_find_x_l813_81315

theorem find_x (x : ℝ) (h : (40 / 80) = Real.sqrt (x / 80)) : x = 20 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l813_81315


namespace NUMINAMATH_GPT_units_digit_8421_1287_l813_81369

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_8421_1287 :
  units_digit (8421 ^ 1287) = 1 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_8421_1287_l813_81369


namespace NUMINAMATH_GPT_donny_total_spending_l813_81372

noncomputable def total_saving_mon : ℕ := 15
noncomputable def total_saving_tue : ℕ := 28
noncomputable def total_saving_wed : ℕ := 13
noncomputable def total_saving_fri : ℕ := 22

noncomputable def total_savings_mon_to_wed : ℕ := total_saving_mon + total_saving_tue + total_saving_wed
noncomputable def thursday_spending : ℕ := total_savings_mon_to_wed / 2
noncomputable def remaining_savings_after_thursday : ℕ := total_savings_mon_to_wed - thursday_spending
noncomputable def total_savings_before_sat : ℕ := remaining_savings_after_thursday + total_saving_fri
noncomputable def saturday_spending : ℕ := total_savings_before_sat * 40 / 100

theorem donny_total_spending : thursday_spending + saturday_spending = 48 := by sorry

end NUMINAMATH_GPT_donny_total_spending_l813_81372


namespace NUMINAMATH_GPT_paula_aunt_money_l813_81352

theorem paula_aunt_money
  (shirts_cost : ℕ := 2 * 11)
  (pants_cost : ℕ := 13)
  (money_left : ℕ := 74) : 
  shirts_cost + pants_cost + money_left = 109 :=
by
  sorry

end NUMINAMATH_GPT_paula_aunt_money_l813_81352


namespace NUMINAMATH_GPT_train_length_l813_81344

noncomputable section

-- Define the variables involved in the problem.
def train_length_cross_signal (V : ℝ) : ℝ := V * 18
def train_speed_cross_platform (L : ℝ) (platform_length : ℝ) : ℝ := (L + platform_length) / 40

-- Define the main theorem to prove the length of the train.
theorem train_length (V L : ℝ) (platform_length : ℝ) (h1 : L = V * 18)
(h2 : L + platform_length = V * 40) (h3 : platform_length = 366.67) :
L = 300 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l813_81344


namespace NUMINAMATH_GPT_village_population_l813_81332

variable (Px : ℕ) (t : ℕ) (dX dY : ℕ)
variable (Py : ℕ := 42000) (rateX : ℕ := 1200) (rateY : ℕ := 800) (timeYears : ℕ := 15)

theorem village_population : (Px - rateX * timeYears = Py + rateY * timeYears) → Px = 72000 :=
by
  sorry

end NUMINAMATH_GPT_village_population_l813_81332


namespace NUMINAMATH_GPT_sector_properties_l813_81357

variables (r : ℝ) (alpha l S : ℝ)

noncomputable def arc_length (r alpha : ℝ) : ℝ := alpha * r
noncomputable def sector_area (l r : ℝ) : ℝ := (1/2) * l * r

theorem sector_properties
  (h_r : r = 2)
  (h_alpha : alpha = π / 6) :
  arc_length r alpha = π / 3 ∧ sector_area (arc_length r alpha) r = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_properties_l813_81357


namespace NUMINAMATH_GPT_perpendicular_slopes_l813_81333

theorem perpendicular_slopes {m : ℝ} (h : (1 : ℝ) * -m = -1) : m = 1 :=
by sorry

end NUMINAMATH_GPT_perpendicular_slopes_l813_81333


namespace NUMINAMATH_GPT_greatest_divisor_same_remainder_l813_81343

theorem greatest_divisor_same_remainder (a b c : ℕ) (h₁ : a = 54) (h₂ : b = 87) (h₃ : c = 172) : 
  ∃ d, (d ∣ (b - a)) ∧ (d ∣ (c - b)) ∧ (d ∣ (c - a)) ∧ (∀ e, (e ∣ (b - a)) ∧ (e ∣ (c - b)) ∧ (e ∣ (c - a)) → e ≤ d) ∧ d = 1 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_divisor_same_remainder_l813_81343


namespace NUMINAMATH_GPT_milk_purchase_maximum_l813_81308

theorem milk_purchase_maximum :
  let num_1_liter_bottles := 6
  let num_half_liter_bottles := 6
  let value_per_1_liter_bottle := 20
  let value_per_half_liter_bottle := 15
  let price_per_liter := 22
  let total_value := num_1_liter_bottles * value_per_1_liter_bottle + num_half_liter_bottles * value_per_half_liter_bottle
  total_value / price_per_liter = 5 :=
by
  sorry

end NUMINAMATH_GPT_milk_purchase_maximum_l813_81308


namespace NUMINAMATH_GPT_work_completion_time_l813_81381

-- Define work rates for workers p, q, and r
def work_rate_p : ℚ := 1 / 12
def work_rate_q : ℚ := 1 / 9
def work_rate_r : ℚ := 1 / 18

-- Define time they work in respective phases
def time_p : ℚ := 2
def time_pq : ℚ := 3

-- Define the total time taken to complete the work
def total_time : ℚ := 6

-- Prove that the total time to complete the work is 6 days
theorem work_completion_time :
  (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq + (1 - (work_rate_p * time_p + (work_rate_p + work_rate_q) * time_pq)) / (work_rate_p + work_rate_q + work_rate_r)) = total_time :=
by sorry

end NUMINAMATH_GPT_work_completion_time_l813_81381


namespace NUMINAMATH_GPT_graph_does_not_pass_first_quadrant_l813_81341

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end NUMINAMATH_GPT_graph_does_not_pass_first_quadrant_l813_81341


namespace NUMINAMATH_GPT_infinite_primes_of_form_2px_plus_1_l813_81354

theorem infinite_primes_of_form_2px_plus_1 (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) : 
  ∃ᶠ (n : ℕ) in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end NUMINAMATH_GPT_infinite_primes_of_form_2px_plus_1_l813_81354


namespace NUMINAMATH_GPT_valid_outfits_l813_81301

-- Let's define the conditions first:
variable (shirts colors pairs : ℕ)

-- Suppose we have the following constraints according to the given problem:
def totalShirts : ℕ := 6
def totalPants : ℕ := 6
def totalHats : ℕ := 6
def totalShoes : ℕ := 6
def numOfColors : ℕ := 6

-- We refuse to wear an outfit in which all 4 items are the same color, or in which the shoes match the color of any other item.
theorem valid_outfits : 
  (totalShirts * totalPants * totalHats * (totalShoes - 1) + (totalShirts * 5 - totalShoes)) = 1104 :=
by sorry

end NUMINAMATH_GPT_valid_outfits_l813_81301


namespace NUMINAMATH_GPT_students_doing_at_least_one_hour_of_homework_l813_81362

theorem students_doing_at_least_one_hour_of_homework (total_angle : ℝ) (less_than_one_hour_angle : ℝ) 
  (h1 : total_angle = 360) (h2 : less_than_one_hour_angle = 90) :
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  at_least_one_hour_percentage = 75 :=
by
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  sorry

end NUMINAMATH_GPT_students_doing_at_least_one_hour_of_homework_l813_81362


namespace NUMINAMATH_GPT_number_of_divisors_23232_l813_81376

theorem number_of_divisors_23232 : ∀ (n : ℕ), 
    n = 23232 → 
    (∃ k : ℕ, k = 42 ∧ (∀ d : ℕ, (d > 0 ∧ d ∣ n) → (↑d < k + 1))) :=
by
  sorry

end NUMINAMATH_GPT_number_of_divisors_23232_l813_81376


namespace NUMINAMATH_GPT_g_h_of_2_eq_2340_l813_81339

def g (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3
def h (x : ℝ) : ℝ := 4 * x^3 + 1

theorem g_h_of_2_eq_2340 : g (h 2) = 2340 := 
  sorry

end NUMINAMATH_GPT_g_h_of_2_eq_2340_l813_81339


namespace NUMINAMATH_GPT_taxi_fare_ride_distance_l813_81309

theorem taxi_fare_ride_distance (fare_first: ℝ) (first_mile: ℝ) (additional_fare_rate: ℝ) (additional_distance: ℝ) (total_amount: ℝ) (tip: ℝ) (x: ℝ) :
  fare_first = 3.00 ∧ first_mile = 0.75 ∧ additional_fare_rate = 0.25 ∧ additional_distance = 0.1 ∧ total_amount = 15 ∧ tip = 3 ∧
  (total_amount - tip) = fare_first + additional_fare_rate * (x - first_mile) / additional_distance → x = 4.35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_taxi_fare_ride_distance_l813_81309


namespace NUMINAMATH_GPT_course_gender_relationship_expected_value_X_l813_81302

-- Define the data based on the problem statement
def total_students := 450
def total_boys := 250
def total_girls := 200
def boys_course_b := 150
def girls_course_a := 50
def boys_course_a := total_boys - boys_course_b -- 100
def girls_course_b := total_girls - girls_course_a -- 150

-- Test statistic for independence (calculated)
def chi_squared := 22.5
def critical_value := 10.828

-- Null hypothesis for independence
def H0 := "The choice of course is independent of gender"

-- part 1: proving independence rejection based on chi-squared value
theorem course_gender_relationship : chi_squared > critical_value :=
  by sorry

-- For part 2, stratified sampling and expected value
-- Define probabilities and expected value
def P_X_0 := 1/6
def P_X_1 := 1/2
def P_X_2 := 3/10
def P_X_3 := 1/30

def expected_X := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3

-- part 2: proving expected value E(X) calculation
theorem expected_value_X : expected_X = 6/5 :=
  by sorry

end NUMINAMATH_GPT_course_gender_relationship_expected_value_X_l813_81302


namespace NUMINAMATH_GPT_cone_volume_from_half_sector_l813_81335

theorem cone_volume_from_half_sector (R : ℝ) (V : ℝ) : 
  R = 6 →
  V = (1/3) * Real.pi * (R / 2)^2 * (R * Real.sqrt 3) →
  V = 9 * Real.pi * Real.sqrt 3 := by sorry

end NUMINAMATH_GPT_cone_volume_from_half_sector_l813_81335


namespace NUMINAMATH_GPT_sin_25_over_6_pi_l813_81304

noncomputable def sin_value : ℝ :=
  Real.sin (25 / 6 * Real.pi)

theorem sin_25_over_6_pi : sin_value = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_25_over_6_pi_l813_81304


namespace NUMINAMATH_GPT_eight_mul_eleven_and_one_fourth_l813_81358

theorem eight_mul_eleven_and_one_fourth : 8 * (11 + (1 / 4 : ℝ)) = 90 := by
  sorry

end NUMINAMATH_GPT_eight_mul_eleven_and_one_fourth_l813_81358


namespace NUMINAMATH_GPT_original_workers_l813_81303

theorem original_workers (x y : ℝ) (h : x = (65 / 100) * y) : y = (20 / 13) * x :=
by sorry

end NUMINAMATH_GPT_original_workers_l813_81303


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l813_81377

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^(m - 1) > 0 → m = 2) →
  (|m - 2| < 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l813_81377


namespace NUMINAMATH_GPT_find_value_of_m_l813_81364

theorem find_value_of_m : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 :=
by
  use -12
  sorry

end NUMINAMATH_GPT_find_value_of_m_l813_81364


namespace NUMINAMATH_GPT_larry_win_probability_correct_l813_81356

/-- Define the probabilities of knocking off the bottle for both players in the first four turns. -/
structure GameProb (turns : ℕ) :=
  (larry_prob : ℚ)
  (julius_prob : ℚ)

/-- Define the probabilities of knocking off the bottle for both players from the fifth turn onwards. -/
def subsequent_turns_prob : ℚ := 1 / 2
/-- Initial probabilities for the first four turns -/
def initial_prob : GameProb 4 := { larry_prob := 2 / 3, julius_prob := 1 / 3 }
/-- The probability that Larry wins the game -/
def larry_wins (prob : GameProb 4) (subsequent_prob : ℚ) : ℚ :=
  -- Calculation logic goes here resulting in the final probability
  379 / 648

theorem larry_win_probability_correct :
  larry_wins initial_prob subsequent_turns_prob = 379 / 648 :=
sorry

end NUMINAMATH_GPT_larry_win_probability_correct_l813_81356


namespace NUMINAMATH_GPT_acres_of_flax_l813_81346

-- Let F be the number of acres of flax
variable (F : ℕ)

-- Condition: The total farm size is 240 acres
def total_farm_size (F : ℕ) := F + (F + 80) = 240

-- Proof statement
theorem acres_of_flax (h : total_farm_size F) : F = 80 :=
sorry

end NUMINAMATH_GPT_acres_of_flax_l813_81346


namespace NUMINAMATH_GPT_mark_collects_money_l813_81392

variable (households_per_day : Nat)
variable (days : Nat)
variable (pair_amount : Nat)
variable (half_factor : Nat)

theorem mark_collects_money
  (h1 : households_per_day = 20)
  (h2 : days = 5)
  (h3 : pair_amount = 40)
  (h4 : half_factor = 2) :
  (households_per_day * days / half_factor) * pair_amount = 2000 :=
by
  sorry

end NUMINAMATH_GPT_mark_collects_money_l813_81392


namespace NUMINAMATH_GPT_brothers_travel_distance_l813_81355

theorem brothers_travel_distance
  (x : ℝ)
  (hb_x : (120 : ℝ) / (x : ℝ) - 4 = (120 : ℝ) / (x + 40))
  (total_time : 2 = 2) :
  x = 20 ∧ (x + 40) = 60 :=
by
  -- we need to prove the distances
  sorry

end NUMINAMATH_GPT_brothers_travel_distance_l813_81355


namespace NUMINAMATH_GPT_problem_divisibility_l813_81395

theorem problem_divisibility 
  (m n : ℕ) 
  (a : Fin (mn + 1) → ℕ)
  (h_pos : ∀ i, 0 < a i)
  (h_order : ∀ i j, i < j → a i < a j) :
  (∃ (b : Fin (m + 1) → Fin (mn + 1)), ∀ i j, i ≠ j → ¬(a (b i) ∣ a (b j))) ∨
  (∃ (c : Fin (n + 1) → Fin (mn + 1)), ∀ i, i < n → a (c i) ∣ a (c i.succ)) :=
sorry

end NUMINAMATH_GPT_problem_divisibility_l813_81395


namespace NUMINAMATH_GPT_triangular_number_30_l813_81384

theorem triangular_number_30 : 
  (∃ (T : ℕ), T = 30 * (30 + 1) / 2 ∧ T = 465) :=
by 
  sorry

end NUMINAMATH_GPT_triangular_number_30_l813_81384


namespace NUMINAMATH_GPT_solve_for_x_l813_81322

def condition (x : ℝ) : Prop := (x - 5)^3 = (1 / 27)⁻¹

theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 8 := by
  use 8
  unfold condition
  sorry

end NUMINAMATH_GPT_solve_for_x_l813_81322


namespace NUMINAMATH_GPT_solve_quadratic_eqn_l813_81320

theorem solve_quadratic_eqn :
  ∃ x₁ x₂ : ℝ, (x - 6) * (x + 2) = 0 ↔ (x = x₁ ∨ x = x₂) ∧ x₁ = 6 ∧ x₂ = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eqn_l813_81320


namespace NUMINAMATH_GPT_meal_cost_l813_81386

theorem meal_cost (total_paid change tip_rate : ℝ)
  (h_total_paid : total_paid = 20 - change)
  (h_change : change = 5)
  (h_tip_rate : tip_rate = 0.2) :
  ∃ x, x + tip_rate * x = total_paid ∧ x = 12.5 := 
by
  sorry

end NUMINAMATH_GPT_meal_cost_l813_81386


namespace NUMINAMATH_GPT_num_zeros_of_g_l813_81310

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 - 2 * x
else -(x^2 - 2 * -x)

noncomputable def g (x : ℝ) : ℝ := f x + 1

theorem num_zeros_of_g : ∃! x : ℝ, g x = 0 := sorry

end NUMINAMATH_GPT_num_zeros_of_g_l813_81310


namespace NUMINAMATH_GPT_proctoring_arrangements_l813_81396

/-- Consider 4 teachers A, B, C, D each teaching their respective classes a, b, c, d.
    Each teacher must not proctor their own class.
    Prove that there are exactly 9 ways to arrange the proctoring as required. -/
theorem proctoring_arrangements : 
  ∃ (arrangements : Finset ((Fin 4) → (Fin 4))), 
    (∀ (f : (Fin 4) → (Fin 4)), f ∈ arrangements → ∀ i : Fin 4, f i ≠ i) 
    ∧ arrangements.card = 9 :=
sorry

end NUMINAMATH_GPT_proctoring_arrangements_l813_81396


namespace NUMINAMATH_GPT_smallest_n_is_60_l813_81375

def smallest_n (n : ℕ) : Prop :=
  ∃ (n : ℕ), (n > 0) ∧ (24 ∣ n^2) ∧ (450 ∣ n^3) ∧ ∀ m : ℕ, 24 ∣ m^2 → 450 ∣ m^3 → m ≥ n

theorem smallest_n_is_60 : smallest_n 60 :=
  sorry

end NUMINAMATH_GPT_smallest_n_is_60_l813_81375


namespace NUMINAMATH_GPT_youngest_child_cakes_l813_81383

theorem youngest_child_cakes : 
  let total_cakes := 60
  let oldest_cakes := (1 / 4 : ℚ) * total_cakes
  let second_oldest_cakes := (3 / 10 : ℚ) * total_cakes
  let middle_cakes := (1 / 6 : ℚ) * total_cakes
  let second_youngest_cakes := (1 / 5 : ℚ) * total_cakes
  let distributed_cakes := oldest_cakes + second_oldest_cakes + middle_cakes + second_youngest_cakes
  let youngest_cakes := total_cakes - distributed_cakes
  youngest_cakes = 5 := 
by
  exact sorry

end NUMINAMATH_GPT_youngest_child_cakes_l813_81383


namespace NUMINAMATH_GPT_distance_covered_l813_81360

theorem distance_covered (t : ℝ) (s_kmph : ℝ) (distance : ℝ) (h1 : t = 180) (h2 : s_kmph = 18) : 
  distance = 900 :=
by 
  sorry

end NUMINAMATH_GPT_distance_covered_l813_81360


namespace NUMINAMATH_GPT_expression_parity_l813_81391

variable (o n c : ℕ)

def is_odd (x : ℕ) : Prop := ∃ k, x = 2 * k + 1

theorem expression_parity (ho : is_odd o) (hc : is_odd c) : 
  (o^2 + n * o + c) % 2 = 0 :=
  sorry

end NUMINAMATH_GPT_expression_parity_l813_81391


namespace NUMINAMATH_GPT_find_pairs_l813_81387

theorem find_pairs (n k : ℕ) : (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l813_81387


namespace NUMINAMATH_GPT_inverse_function_log_l813_81373

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a^x

theorem inverse_function_log (a : ℝ) (g : ℝ → ℝ) (x : ℝ) (y : ℝ) :
  (a > 0) → (a ≠ 1) → 
  (f 2 a = 4) → 
  (f y a = x) → 
  (g x = y) → 
  g x = Real.logb 2 x := 
by
  intros ha hn hfx hfy hg
  sorry

end NUMINAMATH_GPT_inverse_function_log_l813_81373


namespace NUMINAMATH_GPT_candy_pieces_given_l813_81334

theorem candy_pieces_given (initial total : ℕ) (h1 : initial = 68) (h2 : total = 93) :
  total - initial = 25 :=
by
  sorry

end NUMINAMATH_GPT_candy_pieces_given_l813_81334


namespace NUMINAMATH_GPT_sin_minus_cos_value_complex_trig_value_l813_81300

noncomputable def sin_cos_equation (x : Real) :=
  -Real.pi / 2 < x ∧ x < Real.pi / 2 ∧ Real.sin x + Real.cos x = -1 / 5

theorem sin_minus_cos_value (x : Real) (h : sin_cos_equation x) :
  Real.sin x - Real.cos x = 7 / 5 :=
sorry

theorem complex_trig_value (x : Real) (h : sin_cos_equation x) :
  (Real.sin (Real.pi + x) + Real.sin (3 * Real.pi / 2 - x)) / 
  (Real.tan (Real.pi - x) + Real.sin (Real.pi / 2 - x)) = 3 / 11 :=
sorry

end NUMINAMATH_GPT_sin_minus_cos_value_complex_trig_value_l813_81300


namespace NUMINAMATH_GPT_fractional_part_lawn_remainder_l813_81349

def mary_mowing_time := 3 -- Mary can mow the lawn in 3 hours
def tom_mowing_time := 6  -- Tom can mow the lawn in 6 hours
def mary_working_hours := 1 -- Mary works for 1 hour alone

theorem fractional_part_lawn_remainder : 
  (1 - mary_working_hours / mary_mowing_time) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_fractional_part_lawn_remainder_l813_81349


namespace NUMINAMATH_GPT_fraction_meaningful_condition_l813_81325

theorem fraction_meaningful_condition (m : ℝ) : (m + 3 ≠ 0) → (m ≠ -3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_meaningful_condition_l813_81325


namespace NUMINAMATH_GPT_tan_angle_equiv_tan_1230_l813_81368

theorem tan_angle_equiv_tan_1230 : ∃ n : ℤ, -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (1230 * Real.pi / 180) :=
sorry

end NUMINAMATH_GPT_tan_angle_equiv_tan_1230_l813_81368


namespace NUMINAMATH_GPT_novels_next_to_each_other_l813_81311

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem novels_next_to_each_other (n_essays n_novels : Nat) (condition_novels : n_novels = 2) (condition_essays : n_essays = 3) :
  let total_units := (n_novels - 1) + n_essays
  factorial total_units * factorial n_novels = 48 :=
by
  sorry

end NUMINAMATH_GPT_novels_next_to_each_other_l813_81311


namespace NUMINAMATH_GPT_base8_subtraction_and_conversion_l813_81338

-- Define the base 8 numbers
def num1 : ℕ := 7463 -- 7463 in base 8
def num2 : ℕ := 3254 -- 3254 in base 8

-- Define the subtraction in base 8 and conversion to base 10
def result_base8 : ℕ := 4207 -- Expected result in base 8
def result_base10 : ℕ := 2183 -- Expected result in base 10

-- Helper function to convert from base 8 to base 10
def convert_base8_to_base10 (n : ℕ) : ℕ := 
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8 + (n % 10)
 
-- Main theorem statement
theorem base8_subtraction_and_conversion :
  (num1 - num2 = result_base8) ∧ (convert_base8_to_base10 result_base8 = result_base10) :=
by
  sorry

end NUMINAMATH_GPT_base8_subtraction_and_conversion_l813_81338


namespace NUMINAMATH_GPT_fill_boxes_l813_81371

theorem fill_boxes (a b c d e f g : ℤ) 
  (h1 : a + (-1) + 2 = 4)
  (h2 : 2 + 1 + b = 3)
  (h3 : c + (-4) + (-3) = -2)
  (h4 : b - 5 - 4 = -9)
  (h5 : f = d - 3)
  (h6 : g = d + 3)
  (h7 : -8 = 4 + 3 - 9 - 2 + (d - 3) + (d + 3)) : 
  a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_fill_boxes_l813_81371


namespace NUMINAMATH_GPT_original_number_of_friends_l813_81306

theorem original_number_of_friends (F : ℕ) (h₁ : 5000 / F - 125 = 5000 / (F + 8)) : F = 16 :=
sorry

end NUMINAMATH_GPT_original_number_of_friends_l813_81306


namespace NUMINAMATH_GPT_pen_case_cost_l813_81317

noncomputable def case_cost (p i c : ℝ) : Prop :=
  p + i + c = 2.30 ∧
  p = 1.50 + i ∧
  c = 0.5 * i →
  c = 0.1335

theorem pen_case_cost (p i c : ℝ) : case_cost p i c :=
by
  sorry

end NUMINAMATH_GPT_pen_case_cost_l813_81317


namespace NUMINAMATH_GPT_pyramid_max_volume_height_l813_81342

-- Define the conditions and the theorem
theorem pyramid_max_volume_height
  (a h V : ℝ)
  (SA : ℝ := 2 * Real.sqrt 3)
  (h_eq : h = Real.sqrt (SA^2 - (Real.sqrt 2 * a / 2)^2))
  (V_eq : V = (1 / 3) * a^2 * h)
  (derivative_at_max : ∀ a, (48 * a^3 - 3 * a^5 = 0) → (a = 0 ∨ a = 4))
  (max_a_value : a = 4):
  h = 2 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_max_volume_height_l813_81342


namespace NUMINAMATH_GPT_permutation_combination_example_l813_81351

-- Definition of permutation (A) and combination (C) in Lean
def permutation (n k : ℕ): ℕ := Nat.factorial n / Nat.factorial (n - k)
def combination (n k : ℕ): ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The Lean statement of the proof problem
theorem permutation_combination_example : 
3 * permutation 3 2 + 2 * combination 4 2 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_permutation_combination_example_l813_81351


namespace NUMINAMATH_GPT_tiered_water_pricing_l813_81389

theorem tiered_water_pricing (x : ℝ) (y : ℝ) : 
  (∀ z, 0 ≤ z ∧ z ≤ 12 → y = 3 * z ∨
        12 < z ∧ z ≤ 18 → y = 36 + 6 * (z - 12) ∨
        18 < z → y = 72 + 9 * (z - 18)) → 
  y = 54 → 
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_tiered_water_pricing_l813_81389


namespace NUMINAMATH_GPT_distinct_real_numbers_condition_l813_81314

theorem distinct_real_numbers_condition (a b c : ℝ) (h_abc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : (a / (b - c)) + (b / (c - a)) + (c / (a - b)) = 1) :
  (a / (b - c)^2) + (b / (c - a)^2) + (c / (a - b)^2) = 1 := 
by sorry

end NUMINAMATH_GPT_distinct_real_numbers_condition_l813_81314


namespace NUMINAMATH_GPT_geometric_sequence_a4_l813_81323

noncomputable def a (n : ℕ) : ℝ := sorry -- placeholder for the geometric sequence

axiom a_2 : a 2 = -2
axiom a_6 : a 6 = -32
axiom geom_seq (n : ℕ) : a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem geometric_sequence_a4 : a 4 = -8 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l813_81323


namespace NUMINAMATH_GPT_y_relationship_l813_81363

theorem y_relationship :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = (-2)^2 - 4*(-2) - 3) ∧ 
  (y2 = 1^2 - 4*1 - 3) ∧ 
  (y3 = 4^2 - 4*4 - 3) → 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end NUMINAMATH_GPT_y_relationship_l813_81363


namespace NUMINAMATH_GPT_range_of_b_l813_81324

theorem range_of_b (b : ℝ) (hb : b > 0) : (∃ x : ℝ, |x - 5| + |x - 10| > b) ↔ (0 < b ∧ b < 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l813_81324


namespace NUMINAMATH_GPT_direction_vector_of_line_l813_81350

theorem direction_vector_of_line : ∃ Δx Δy : ℚ, y = - (1/2) * x + 1 → Δx = 2 ∧ Δy = -1 :=
sorry

end NUMINAMATH_GPT_direction_vector_of_line_l813_81350


namespace NUMINAMATH_GPT_least_three_digit_divisible_3_4_7_is_168_l813_81366

-- Define the function that checks the conditions
def is_least_three_digit_divisible_by_3_4_7 (x : ℕ) : Prop :=
  100 ≤ x ∧ x < 1000 ∧ x % 3 = 0 ∧ x % 4 = 0 ∧ x % 7 = 0

-- Define the target value
def least_three_digit_number_divisible_by_3_4_7 : ℕ := 168

-- The theorem we want to prove
theorem least_three_digit_divisible_3_4_7_is_168 :
  ∃ x : ℕ, is_least_three_digit_divisible_by_3_4_7 x ∧ x = least_three_digit_number_divisible_by_3_4_7 := by
  sorry

end NUMINAMATH_GPT_least_three_digit_divisible_3_4_7_is_168_l813_81366


namespace NUMINAMATH_GPT_smallest_period_of_f_is_pi_div_2_l813_81348

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 4 + (Real.sin x) ^ 2

theorem smallest_period_of_f_is_pi_div_2 : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_smallest_period_of_f_is_pi_div_2_l813_81348


namespace NUMINAMATH_GPT_highest_score_l813_81347

variables (H L : ℕ) (average_46 : ℕ := 61) (innings_46 : ℕ := 46) 
                (difference : ℕ := 150) (average_44 : ℕ := 58) (innings_44 : ℕ := 44)

theorem highest_score:
  (H - L = difference) →
  (average_46 * innings_46 = average_44 * innings_44 + H + L) →
  H = 202 :=
by
  intros h_diff total_runs_eq
  sorry

end NUMINAMATH_GPT_highest_score_l813_81347


namespace NUMINAMATH_GPT_harish_ganpat_paint_wall_together_l813_81379

theorem harish_ganpat_paint_wall_together :
  let r_h := 1 / 3 -- Harish's rate of work (walls per hour)
  let r_g := 1 / 6 -- Ganpat's rate of work (walls per hour)
  let combined_rate := r_h + r_g -- Combined rate of work when both work together
  let time_to_paint_one_wall := 1 / combined_rate -- Time to paint one wall together
  time_to_paint_one_wall = 2 :=
by
  sorry

end NUMINAMATH_GPT_harish_ganpat_paint_wall_together_l813_81379


namespace NUMINAMATH_GPT_ratio_shorter_to_longer_l813_81380

-- Constants for the problem
def total_length : ℝ := 49
def shorter_piece_length : ℝ := 14

-- Definition of longer piece length based on the given conditions
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- The theorem to be proved
theorem ratio_shorter_to_longer : 
  shorter_piece_length / longer_piece_length = 2 / 5 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_ratio_shorter_to_longer_l813_81380


namespace NUMINAMATH_GPT_grape_juice_percentage_after_addition_l813_81397

def initial_mixture_volume : ℝ := 40
def initial_grape_juice_percentage : ℝ := 0.10
def added_grape_juice_volume : ℝ := 10

theorem grape_juice_percentage_after_addition :
  ((initial_mixture_volume * initial_grape_juice_percentage + added_grape_juice_volume) /
  (initial_mixture_volume + added_grape_juice_volume)) * 100 = 28 :=
by 
  sorry

end NUMINAMATH_GPT_grape_juice_percentage_after_addition_l813_81397


namespace NUMINAMATH_GPT_triangle_tangent_identity_l813_81307

theorem triangle_tangent_identity (A B C : ℝ) (h : A + B + C = Real.pi) : 
  (Real.tan (A / 2) * Real.tan (B / 2)) + (Real.tan (B / 2) * Real.tan (C / 2)) + (Real.tan (C / 2) * Real.tan (A / 2)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_tangent_identity_l813_81307


namespace NUMINAMATH_GPT_value_of_f_neg_11_over_2_l813_81330

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodicity (x : ℝ) : f (x + 2) = - (f x)⁻¹
axiom interval_value (h : 2 ≤ 5 / 2 ∧ 5 / 2 ≤ 3) : f (5 / 2) = 5 / 2

theorem value_of_f_neg_11_over_2 : f (-11 / 2) = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_neg_11_over_2_l813_81330


namespace NUMINAMATH_GPT_volume_is_correct_l813_81378

noncomputable def volume_of_rectangular_parallelepiped (a b : ℝ) (h_diag : (2 * a^2 + b^2 = 1)) (h_surface_area : (4 * a * b + 2 * a^2 = 1)) : ℝ :=
  a^2 * b

theorem volume_is_correct (a b : ℝ)
  (h_diag : 2 * a^2 + b^2 = 1)
  (h_surface_area : 4 * a * b + 2 * a^2 = 1) :
  volume_of_rectangular_parallelepiped a b h_diag h_surface_area = (Real.sqrt 2) / 27 :=
sorry

end NUMINAMATH_GPT_volume_is_correct_l813_81378


namespace NUMINAMATH_GPT_harry_average_sleep_l813_81313

-- Conditions
def sleep_time_monday : ℕ × ℕ := (8, 15)
def sleep_time_tuesday : ℕ × ℕ := (7, 45)
def sleep_time_wednesday : ℕ × ℕ := (8, 10)
def sleep_time_thursday : ℕ × ℕ := (10, 25)
def sleep_time_friday : ℕ × ℕ := (7, 50)

-- Total sleep time calculation
def total_sleep_time : ℕ × ℕ :=
  let (h1, m1) := sleep_time_monday
  let (h2, m2) := sleep_time_tuesday
  let (h3, m3) := sleep_time_wednesday
  let (h4, m4) := sleep_time_thursday
  let (h5, m5) := sleep_time_friday
  (h1 + h2 + h3 + h4 + h5, m1 + m2 + m3 + m4 + m5)

-- Convert minutes to hours and minutes
def convert_minutes (mins : ℕ) : ℕ × ℕ :=
  (mins / 60, mins % 60)

-- Final total sleep time
def final_total_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := total_sleep_time
  let (extra_hours, remaining_minutes) := convert_minutes total_minutes
  (total_hours + extra_hours, remaining_minutes)

-- Average calculation
def average_sleep_time : ℕ × ℕ :=
  let (total_hours, total_minutes) := final_total_time
  (total_hours / 5, (total_hours % 5) * 60 / 5 + total_minutes / 5)

-- The proof statement
theorem harry_average_sleep :
  average_sleep_time = (8, 29) :=
  by
    sorry

end NUMINAMATH_GPT_harry_average_sleep_l813_81313


namespace NUMINAMATH_GPT_problem_statement_l813_81361

noncomputable def f (m x : ℝ) := (m-1) * Real.log x + m * x^2 + 1

theorem problem_statement (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → f m x₁ - f m x₂ > 2 * (x₁ - x₂)) ↔ 
  m ≥ (1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l813_81361


namespace NUMINAMATH_GPT_purely_imaginary_complex_l813_81336

theorem purely_imaginary_complex (a : ℝ) 
  (h₁ : a^2 + 2 * a - 3 = 0)
  (h₂ : a + 3 ≠ 0) : a = 1 := by
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_l813_81336


namespace NUMINAMATH_GPT_paving_cost_l813_81337

-- Definitions based on conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sqm : ℝ := 600
def expected_cost : ℝ := 12375

-- The problem statement
theorem paving_cost :
  (length * width * rate_per_sqm = expected_cost) :=
sorry

end NUMINAMATH_GPT_paving_cost_l813_81337


namespace NUMINAMATH_GPT_total_hours_proof_l813_81370

-- Conditions
def half_hour_show_episodes : ℕ := 24
def one_hour_show_episodes : ℕ := 12
def half_hour_per_episode : ℝ := 0.5
def one_hour_per_episode : ℝ := 1.0

-- Define the total hours Tim watched
def total_hours_watched : ℝ :=
  half_hour_show_episodes * half_hour_per_episode + one_hour_show_episodes * one_hour_per_episode

-- Prove that the total hours watched is 24
theorem total_hours_proof : total_hours_watched = 24 := by
  sorry

end NUMINAMATH_GPT_total_hours_proof_l813_81370


namespace NUMINAMATH_GPT_hours_per_day_l813_81385

theorem hours_per_day
  (num_warehouse : ℕ := 4)
  (num_managers : ℕ := 2)
  (rate_warehouse : ℝ := 15)
  (rate_manager : ℝ := 20)
  (tax_rate : ℝ := 0.10)
  (days_worked : ℕ := 25)
  (total_cost : ℝ := 22000) :
  ∃ h : ℝ, 6 * h * days_worked * (rate_warehouse + rate_manager) * (1 + tax_rate) = total_cost ∧ h = 8 :=
by
  sorry

end NUMINAMATH_GPT_hours_per_day_l813_81385


namespace NUMINAMATH_GPT_total_students_correct_l813_81305

-- Define the given conditions
variables (A B C : ℕ)

-- Number of students in class B
def B_def : ℕ := 25

-- Number of students in class A (B is 8 fewer than A)
def A_def : ℕ := B_def + 8

-- Number of students in class C (C is 5 times B)
def C_def : ℕ := 5 * B_def

-- The total number of students
def total_students : ℕ := A_def + B_def + C_def

-- The proof statement
theorem total_students_correct : total_students = 183 := by
  sorry

end NUMINAMATH_GPT_total_students_correct_l813_81305


namespace NUMINAMATH_GPT_jane_dolls_l813_81316

theorem jane_dolls (jane_dolls jill_dolls : ℕ) (h1 : jane_dolls + jill_dolls = 32) (h2 : jill_dolls = jane_dolls + 6) : jane_dolls = 13 := 
by {
  sorry
}

end NUMINAMATH_GPT_jane_dolls_l813_81316


namespace NUMINAMATH_GPT_power_problem_l813_81329

theorem power_problem (k : ℕ) (h : 6 ^ k = 4) : 6 ^ (2 * k + 3) = 3456 := 
by 
  sorry

end NUMINAMATH_GPT_power_problem_l813_81329


namespace NUMINAMATH_GPT_evaluate_expression_l813_81359

theorem evaluate_expression : (2^3001 * 3^3003) / 6^3002 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l813_81359


namespace NUMINAMATH_GPT_average_monthly_growth_rate_l813_81394

-- Define the conditions
variables (P : ℝ) (r : ℝ)
-- The condition that output in December is P times that of January
axiom growth_rate_condition : (1 + r)^11 = P

-- Define the goal to prove the average monthly growth rate
theorem average_monthly_growth_rate : r = (P^(1/11) - 1) :=
by
  sorry

end NUMINAMATH_GPT_average_monthly_growth_rate_l813_81394


namespace NUMINAMATH_GPT_probability_of_selection_of_X_l813_81390

theorem probability_of_selection_of_X 
  (P_Y : ℝ)
  (P_X_and_Y : ℝ) :
  P_Y = 2 / 7 →
  P_X_and_Y = 0.05714285714285714 →
  ∃ P_X : ℝ, P_X = 0.2 :=
by
  intro hY hXY
  sorry

end NUMINAMATH_GPT_probability_of_selection_of_X_l813_81390


namespace NUMINAMATH_GPT_right_triangle_sides_l813_81365

theorem right_triangle_sides {a b c : ℕ} (h1 : a * (b + 2) = 150) (h2 : a^2 + b^2 = c^2) (h3 : a + (1 / 2 : ℤ) * (a * b) = 75) :
  (a = 6 ∧ b = 23 ∧ c = 25) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
sorry

end NUMINAMATH_GPT_right_triangle_sides_l813_81365


namespace NUMINAMATH_GPT_pentagon_area_l813_81388

-- Definitions of the vertices of the pentagon
def vertices : List (ℝ × ℝ) :=
  [(0, 0), (1, 2), (3, 3), (4, 1), (2, 0)]

-- Definition of the number of interior points
def interior_points : ℕ := 7

-- Definition of the number of boundary points
def boundary_points : ℕ := 5

-- Pick's theorem: Area = Interior points + Boundary points / 2 - 1
noncomputable def area : ℝ :=
  interior_points + boundary_points / 2 - 1

-- Theorem to be proved
theorem pentagon_area :
  area = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_area_l813_81388


namespace NUMINAMATH_GPT_ratio_of_areas_of_circles_l813_81398

theorem ratio_of_areas_of_circles (C_A C_B C_C : ℝ) (h1 : (60 / 360) * C_A = (40 / 360) * C_B) (h2 : (30 / 360) * C_B = (90 / 360) * C_C) : 
  (C_A / (2 * Real.pi))^2 / (C_C / (2 * Real.pi))^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_circles_l813_81398


namespace NUMINAMATH_GPT_nate_current_age_l813_81393

open Real

variables (E N : ℝ)

/-- Ember is half as old as Nate, so E = 1/2 * N. -/
def ember_half_nate (h1 : E = 1/2 * N) : Prop := True

/-- The age difference of 7 years remains constant, so 21 - 14 = N - E. -/
def age_diff_constant (h2 : 7 = N - E) : Prop := True

/-- Prove that Nate is currently 14 years old given the conditions. -/
theorem nate_current_age (h1 : E = 1/2 * N) (h2 : 7 = N - E) : N = 14 :=
by sorry

end NUMINAMATH_GPT_nate_current_age_l813_81393


namespace NUMINAMATH_GPT_N_is_composite_l813_81340

def N : ℕ := 2011 * 2012 * 2013 * 2014 + 1

theorem N_is_composite : ¬ Prime N := by
  sorry

end NUMINAMATH_GPT_N_is_composite_l813_81340


namespace NUMINAMATH_GPT_evaluate_expression_l813_81327

theorem evaluate_expression : 6 - 8 * (5 - 2^3) / 2 = 18 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l813_81327


namespace NUMINAMATH_GPT_number_of_girls_l813_81326

theorem number_of_girls (classes : ℕ) (students_per_class : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : classes = 4) 
  (h2 : students_per_class = 25) 
  (h3 : boys = 56) 
  (h4 : girls = (classes * students_per_class) - boys) : 
  girls = 44 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l813_81326


namespace NUMINAMATH_GPT_households_subscribing_B_and_C_l813_81328

/-- Each household subscribes to 2 different newspapers.
Residents only subscribe to newspapers A, B, and C.
There are 30 subscriptions for newspaper A.
There are 34 subscriptions for newspaper B.
There are 40 subscriptions for newspaper C.
Thus, the number of households that subscribe to both
newspaper B and newspaper C is 22. -/
theorem households_subscribing_B_and_C (subs_A subs_B subs_C households : ℕ) 
    (hA : subs_A = 30) (hB : subs_B = 34) (hC : subs_C = 40) (h_total : households = (subs_A + subs_B + subs_C) / 2) :
  (households - subs_A) = 22 :=
by
  -- Substitute the values to demonstrate equality based on the given conditions.
  sorry

end NUMINAMATH_GPT_households_subscribing_B_and_C_l813_81328


namespace NUMINAMATH_GPT_maximize_winning_probability_l813_81319

def ahmet_wins (n : ℕ) : Prop :=
  n = 13

theorem maximize_winning_probability :
  ∃ n ∈ {x : ℕ | x ≥ 1 ∧ x ≤ 25}, ahmet_wins n :=
by
  sorry

end NUMINAMATH_GPT_maximize_winning_probability_l813_81319
