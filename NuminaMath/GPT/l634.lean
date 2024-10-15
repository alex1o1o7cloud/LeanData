import Mathlib

namespace NUMINAMATH_GPT_number_one_seventh_equals_five_l634_63466

theorem number_one_seventh_equals_five (n : ℕ) (h : n / 7 = 5) : n = 35 :=
sorry

end NUMINAMATH_GPT_number_one_seventh_equals_five_l634_63466


namespace NUMINAMATH_GPT_no_integer_solution_l634_63436

theorem no_integer_solution :
  ∀ (x : ℤ), ¬ (x^2 + 3 < 2 * x) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_integer_solution_l634_63436


namespace NUMINAMATH_GPT_find_d_l634_63445

-- Define the conditions
variables (x₀ y₀ c : ℝ)

-- Define the system of equations
def system_of_equations : Prop :=
  x₀ * y₀ = 6 ∧ x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2

-- Define the target proof problem
theorem find_d (h : system_of_equations x₀ y₀ c) : x₀^2 + y₀^2 = 69 :=
sorry

end NUMINAMATH_GPT_find_d_l634_63445


namespace NUMINAMATH_GPT_carnival_days_l634_63480

theorem carnival_days (d : ℕ) (h : 50 * d + 3 * (50 * d) - 30 * d - 75 = 895) : d = 5 :=
by
  sorry

end NUMINAMATH_GPT_carnival_days_l634_63480


namespace NUMINAMATH_GPT_abs_sum_inequality_for_all_x_l634_63404

theorem abs_sum_inequality_for_all_x (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ (m ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_abs_sum_inequality_for_all_x_l634_63404


namespace NUMINAMATH_GPT_animals_on_stump_l634_63482

def possible_n_values (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 15

theorem animals_on_stump (n : ℕ) (h1 : n ≥ 3) (h2 : n ≤ 20)
  (h3 : 11 ≥ (n + 1) / 3) (h4 : 9 ≥ n - (n + 1) / 3) : possible_n_values n :=
by {
  sorry
}

end NUMINAMATH_GPT_animals_on_stump_l634_63482


namespace NUMINAMATH_GPT_timothy_read_pages_l634_63429

theorem timothy_read_pages 
    (mon_tue_pages : Nat) (wed_pages : Nat) (thu_sat_pages : Nat) 
    (sun_read_pages : Nat) (sun_review_pages : Nat) : 
    mon_tue_pages = 45 → wed_pages = 50 → thu_sat_pages = 40 → sun_read_pages = 25 → sun_review_pages = 15 →
    (2 * mon_tue_pages + wed_pages + 3 * thu_sat_pages + sun_read_pages + sun_review_pages = 300) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_timothy_read_pages_l634_63429


namespace NUMINAMATH_GPT_triangle_angle_bisector_segment_length_l634_63428

theorem triangle_angle_bisector_segment_length
  (DE DF EF DG EG : ℝ)
  (h_ratio : DE / 12 = 1 ∧ DF / DE = 4 / 3 ∧ EF / DE = 5 / 3)
  (h_angle_bisector : DG / EG = DE / DF ∧ DG + EG = EF) :
  EG = 80 / 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_bisector_segment_length_l634_63428


namespace NUMINAMATH_GPT_scientific_notation_of_30067_l634_63487

theorem scientific_notation_of_30067 : ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 30067 = a * 10^n := by
  use 3.0067
  use 4
  sorry

end NUMINAMATH_GPT_scientific_notation_of_30067_l634_63487


namespace NUMINAMATH_GPT_sin_neg_60_eq_neg_sqrt_3_div_2_l634_63496

theorem sin_neg_60_eq_neg_sqrt_3_div_2 : 
  Real.sin (-π / 3) = - (Real.sqrt 3) / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_neg_60_eq_neg_sqrt_3_div_2_l634_63496


namespace NUMINAMATH_GPT_solve_fraction_eq_l634_63483

theorem solve_fraction_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : (1 / (x - 1) = 3 / (x - 3)) ↔ x = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_fraction_eq_l634_63483


namespace NUMINAMATH_GPT_highest_score_is_151_l634_63459

-- Definitions for the problem conditions
def total_runs : ℕ := 2704
def total_runs_excluding_HL : ℕ := 2552

variables (H L : ℕ) 

-- Problem conditions as hypotheses
axiom h1 : H - L = 150
axiom h2 : H + L = 152
axiom h3 : 2704 = 2552 + H + L

-- Proof statement
theorem highest_score_is_151 (H L : ℕ) (h1 : H - L = 150) (h2 : H + L = 152) (h3 : 2704 = 2552 + H + L) : H = 151 :=
by sorry

end NUMINAMATH_GPT_highest_score_is_151_l634_63459


namespace NUMINAMATH_GPT_smallest_three_digit_divisible_by_4_and_5_l634_63489

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 4 = 0) ∧ (m % 5 = 0) → m ≥ n →
n = 100 :=
sorry

end NUMINAMATH_GPT_smallest_three_digit_divisible_by_4_and_5_l634_63489


namespace NUMINAMATH_GPT_greatest_number_divisible_by_11_and_3_l634_63463

namespace GreatestNumberDivisibility

theorem greatest_number_divisible_by_11_and_3 : 
  ∃ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (2 * A - 2 * B + C) % 11 = 0 ∧ 
    (2 * A + 2 * C + B) % 3 = 0 ∧
    (10000 * A + 1000 * C + 100 * C + 10 * B + A) = 95695 :=
by
  -- The proof here is omitted.
  sorry

end GreatestNumberDivisibility

end NUMINAMATH_GPT_greatest_number_divisible_by_11_and_3_l634_63463


namespace NUMINAMATH_GPT_Lin_trip_time_l634_63408

theorem Lin_trip_time
  (v : ℕ) -- speed on the mountain road in miles per minute
  (h1 : 80 = d_highway) -- Lin travels 80 miles on the highway
  (h2 : 20 = d_mountain) -- Lin travels 20 miles on the mountain road
  (h3 : v_highway = 2 * v) -- Lin drives twice as fast on the highway
  (h4 : 40 = 20 / v) -- Lin spent 40 minutes driving on the mountain road
  : 40 + 80 = 120 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_Lin_trip_time_l634_63408


namespace NUMINAMATH_GPT_range_of_m_l634_63457

def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B (m : ℝ) : Set ℝ := {x | abs (x - 3) ≤ m}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (m : ℝ) : Prop := x ∈ B m

theorem range_of_m (m : ℝ) (hm : m > 0):
  (∀ x, p x → q x m) ↔ (6 ≤ m) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l634_63457


namespace NUMINAMATH_GPT_front_view_length_l634_63488

-- Define the conditions of the problem
variables (d_body : ℝ) (d_side : ℝ) (d_top : ℝ)
variables (d_front : ℝ)

-- The given conditions
def conditions :=
  d_body = 5 * Real.sqrt 2 ∧
  d_side = 5 ∧
  d_top = Real.sqrt 34

-- The theorem to be proved
theorem front_view_length : 
  conditions d_body d_side d_top →
  d_front = Real.sqrt 41 :=
sorry

end NUMINAMATH_GPT_front_view_length_l634_63488


namespace NUMINAMATH_GPT_trig_second_quadrant_l634_63405

theorem trig_second_quadrant (α : ℝ) (h1 : α > π / 2) (h2 : α < π) :
  (|Real.sin α| / Real.sin α) - (|Real.cos α| / Real.cos α) = 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_second_quadrant_l634_63405


namespace NUMINAMATH_GPT_value_of_expression_l634_63462

theorem value_of_expression : (4 * 3) + 2 = 14 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l634_63462


namespace NUMINAMATH_GPT_ratio_closest_to_one_l634_63485

-- Define the entrance fee for adults and children.
def adult_fee : ℕ := 20
def child_fee : ℕ := 15

-- Define the total collected amount.
def total_collected : ℕ := 2400

-- Define the number of adults and children.
variables (a c : ℕ)

-- The main theorem to prove:
theorem ratio_closest_to_one 
  (h1 : a > 0) -- at least one adult
  (h2 : c > 0) -- at least one child
  (h3 : adult_fee * a + child_fee * c = total_collected) : 
  a / (c : ℚ) = 69 / 68 := 
sorry

end NUMINAMATH_GPT_ratio_closest_to_one_l634_63485


namespace NUMINAMATH_GPT_trig_triple_angle_l634_63407

theorem trig_triple_angle (θ : ℝ) (h : Real.tan θ = 5) :
  Real.tan (3 * θ) = 55 / 37 ∧
  Real.sin (3 * θ) = 55 * Real.sqrt 1369 / (37 * Real.sqrt 4394) ∨ Real.sin (3 * θ) = -(55 * Real.sqrt 1369 / (37 * Real.sqrt 4394)) ∧
  Real.cos (3 * θ) = Real.sqrt (1369 / 4394) ∨ Real.cos (3 * θ) = -Real.sqrt (1369 / 4394) :=
by
  sorry

end NUMINAMATH_GPT_trig_triple_angle_l634_63407


namespace NUMINAMATH_GPT_complex_quadrant_l634_63451

open Complex

theorem complex_quadrant
  (z1 z2 z : ℂ) (h1 : z1 = 2 + I) (h2 : z2 = 1 - I) (h3 : z = z1 / z2) :
  0 < z.re ∧ 0 < z.im :=
by
  -- sorry to skip the proof steps
  sorry

end NUMINAMATH_GPT_complex_quadrant_l634_63451


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l634_63443

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a = 1) (h2 : |a| = 1) : 
  (a = 1 → |a| = 1) ∧ ¬(|a| = 1 → a = 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l634_63443


namespace NUMINAMATH_GPT_inequality_solution_set_l634_63444

theorem inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ x ≠ 2 ∧ (2 * x / (x - 2) + (x + 3) / (3 * x) ≥ 4)} 
  = {x : ℝ | (0 < x ∧ x ≤ 1/5) ∨ (2 < x ∧ x ≤ 6)} := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_set_l634_63444


namespace NUMINAMATH_GPT_calculate_otimes_l634_63430

def otimes (x y : ℝ) : ℝ := x^3 - y^2 + x

theorem calculate_otimes (k : ℝ) : 
  otimes k (otimes k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end NUMINAMATH_GPT_calculate_otimes_l634_63430


namespace NUMINAMATH_GPT_equivalent_expr_l634_63411

theorem equivalent_expr (a y : ℝ) (ha : a ≠ 0) (hy : y ≠ a ∧ y ≠ -a) :
  ( (a / (a + y) + y / (a - y)) / ( y / (a + y) - a / (a - y)) ) = -1 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_expr_l634_63411


namespace NUMINAMATH_GPT_find_number_l634_63460

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l634_63460


namespace NUMINAMATH_GPT_factorize_expression_l634_63434

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l634_63434


namespace NUMINAMATH_GPT_part_A_part_B_l634_63481

-- Definitions for the setup
variables (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0)

-- Part (A): Specific distance 5d
theorem part_A (d : ℝ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = 25 * d^2 ∧ |y - d| = 5 * d → 
  (x = 3 * d ∧ y = -4 * d) ∨ (x = -3 * d ∧ y = -4 * d)) :=
sorry

-- Part (B): General distance nd
theorem part_B (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d → ∃ x y, (x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d)) :=
sorry

end NUMINAMATH_GPT_part_A_part_B_l634_63481


namespace NUMINAMATH_GPT_length_of_DE_l634_63425

theorem length_of_DE 
  (area_ABC : ℝ) 
  (area_trapezoid : ℝ) 
  (altitude_ABC : ℝ) 
  (h1 : area_ABC = 144) 
  (h2 : area_trapezoid = 96)
  (h3 : altitude_ABC = 24) :
  ∃ (DE_length : ℝ), DE_length = 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_length_of_DE_l634_63425


namespace NUMINAMATH_GPT_solve_for_x_l634_63473

theorem solve_for_x (x : ℚ) :
  (4 * x - 12) / 3 = (3 * x + 6) / 5 → 
  x = 78 / 11 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l634_63473


namespace NUMINAMATH_GPT_jackson_total_souvenirs_l634_63474

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end NUMINAMATH_GPT_jackson_total_souvenirs_l634_63474


namespace NUMINAMATH_GPT_food_duration_l634_63468

theorem food_duration (mom_meals_per_day : ℕ) (mom_cups_per_meal : ℚ)
                      (puppy_count : ℕ) (puppy_meals_per_day : ℕ) (puppy_cups_per_meal : ℚ)
                      (total_food : ℚ)
                      (H_mom : mom_meals_per_day = 3) 
                      (H_mom_cups : mom_cups_per_meal = 3/2)
                      (H_puppies : puppy_count = 5) 
                      (H_puppy_meals : puppy_meals_per_day = 2) 
                      (H_puppy_cups : puppy_cups_per_meal = 1/2) 
                      (H_total_food : total_food = 57) : 
  (total_food / ((mom_meals_per_day * mom_cups_per_meal) + (puppy_count * puppy_meals_per_day * puppy_cups_per_meal))) = 6 := 
by
  sorry

end NUMINAMATH_GPT_food_duration_l634_63468


namespace NUMINAMATH_GPT_contradictory_statement_of_p_l634_63433

-- Given proposition p
def p : Prop := ∀ (x : ℝ), x + 3 ≥ 0 → x ≥ -3

-- Contradictory statement of p
noncomputable def contradictory_p : Prop := ∀ (x : ℝ), x + 3 < 0 → x < -3

-- Proof statement
theorem contradictory_statement_of_p : contradictory_p :=
sorry

end NUMINAMATH_GPT_contradictory_statement_of_p_l634_63433


namespace NUMINAMATH_GPT_fractional_exponent_representation_of_sqrt_l634_63439

theorem fractional_exponent_representation_of_sqrt (a : ℝ) : 
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3 / 4) := 
sorry

end NUMINAMATH_GPT_fractional_exponent_representation_of_sqrt_l634_63439


namespace NUMINAMATH_GPT_tony_combined_lift_weight_l634_63497

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end NUMINAMATH_GPT_tony_combined_lift_weight_l634_63497


namespace NUMINAMATH_GPT_remainder_when_divided_by_17_l634_63471

theorem remainder_when_divided_by_17 (N : ℤ) (k : ℤ) 
  (h : N = 221 * k + 43) : N % 17 = 9 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_17_l634_63471


namespace NUMINAMATH_GPT_updated_mean_of_decremented_observations_l634_63455

theorem updated_mean_of_decremented_observations (n : ℕ) (initial_mean decrement : ℝ)
  (h₀ : n = 50) (h₁ : initial_mean = 200) (h₂ : decrement = 6) :
  ((n * initial_mean) - (n * decrement)) / n = 194 := by
  sorry

end NUMINAMATH_GPT_updated_mean_of_decremented_observations_l634_63455


namespace NUMINAMATH_GPT_max_distance_proof_l634_63461

def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline_gallons : ℝ := 21
def maximum_distance : ℝ := highway_mpg * gasoline_gallons

theorem max_distance_proof : maximum_distance = 256.2 := by
  sorry

end NUMINAMATH_GPT_max_distance_proof_l634_63461


namespace NUMINAMATH_GPT_perfect_square_factors_450_l634_63470

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_factors_450_l634_63470


namespace NUMINAMATH_GPT_students_taking_art_l634_63499

def total_students : ℕ := 500
def students_taking_music : ℕ := 20
def students_taking_both : ℕ := 10
def students_taking_neither : ℕ := 470

theorem students_taking_art :
  ∃ (A : ℕ), A = 20 ∧ total_students = 
             (students_taking_music - students_taking_both) + (A - students_taking_both) + students_taking_both + students_taking_neither :=
by
  sorry

end NUMINAMATH_GPT_students_taking_art_l634_63499


namespace NUMINAMATH_GPT_find_5_digit_number_l634_63403

theorem find_5_digit_number {A B C D E : ℕ} 
  (hA_even : A % 2 = 0) 
  (hB_even : B % 2 = 0) 
  (hA_half_B : A = B / 2) 
  (hC_sum : C = A + B) 
  (hDE_prime : Prime (10 * D + E)) 
  (hD_3B : D = 3 * B) : 
  10000 * A + 1000 * B + 100 * C + 10 * D + E = 48247 := 
sorry

end NUMINAMATH_GPT_find_5_digit_number_l634_63403


namespace NUMINAMATH_GPT_sum_of_all_possible_N_l634_63423

theorem sum_of_all_possible_N
  (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c = a + b)
  (h3 : N = a * b * c)
  (h4 : N = 6 * (a + b + c)) :
  N = 156 ∨ N = 96 ∨ N = 84 ∧
  (156 + 96 + 84 = 336) :=
by {
  -- proof will go here
  sorry
}

end NUMINAMATH_GPT_sum_of_all_possible_N_l634_63423


namespace NUMINAMATH_GPT_matt_profit_trade_l634_63426

theorem matt_profit_trade
  (total_cards : ℕ := 8)
  (value_per_card : ℕ := 6)
  (traded_cards_count : ℕ := 2)
  (trade_value_per_card : ℕ := 6)
  (received_cards_count_1 : ℕ := 3)
  (received_value_per_card_1 : ℕ := 2)
  (received_cards_count_2 : ℕ := 1)
  (received_value_per_card_2 : ℕ := 9)
  (profit : ℕ := 3) :
  profit = (received_cards_count_1 * received_value_per_card_1 
           + received_cards_count_2 * received_value_per_card_2) 
           - (traded_cards_count * trade_value_per_card) :=
  by
  sorry

end NUMINAMATH_GPT_matt_profit_trade_l634_63426


namespace NUMINAMATH_GPT_motorists_with_tickets_l634_63464

section SpeedingTickets

variables
  (total_motorists : ℕ)
  (percent_speeding : ℝ) -- percent_speeding is 25% (given)
  (percent_not_ticketed : ℝ) -- percent_not_ticketed is 60% (given)

noncomputable def percent_ticketed : ℝ :=
  let speeding_motorists := percent_speeding * total_motorists / 100
  let ticketed_motorists := speeding_motorists * ((100 - percent_not_ticketed) / 100)
  ticketed_motorists / total_motorists * 100

theorem motorists_with_tickets (total_motorists : ℕ) 
  (h1 : percent_speeding = 25)
  (h2 : percent_not_ticketed = 60) :
  percent_ticketed total_motorists percent_speeding percent_not_ticketed = 10 := 
by
  unfold percent_ticketed
  rw [h1, h2]
  sorry

end SpeedingTickets

end NUMINAMATH_GPT_motorists_with_tickets_l634_63464


namespace NUMINAMATH_GPT_shop_owner_percentage_profit_l634_63454

theorem shop_owner_percentage_profit
  (cp : ℝ)  -- cost price of 1 kg
  (cheat_buy : ℝ) -- cheat percentage when buying
  (cheat_sell : ℝ) -- cheat percentage when selling
  (h_cp : cp = 100) -- cost price is $100
  (h_cheat_buy : cheat_buy = 15) -- cheat by 15% when buying
  (h_cheat_sell : cheat_sell = 20) -- cheat by 20% when selling
  :
  let weight_bought := 1 + (cheat_buy / 100)
  let weight_sold := 1 - (cheat_sell / 100)
  let real_selling_price_per_kg := cp / weight_sold
  let total_selling_price := weight_bought * real_selling_price_per_kg
  let profit := total_selling_price - cp
  let percentage_profit := (profit / cp) * 100
  percentage_profit = 43.75 := 
by
  sorry

end NUMINAMATH_GPT_shop_owner_percentage_profit_l634_63454


namespace NUMINAMATH_GPT_robbers_divide_and_choose_l634_63479

/-- A model of dividing loot between two robbers who do not trust each other -/
def divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) : Prop :=
  ∀ (B : ℕ → ℕ), B (max P1 P2) ≥ B P1 ∧ B (max P1 P2) ≥ B P2

theorem robbers_divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) :
  divide_and_choose P1 P2 A :=
sorry

end NUMINAMATH_GPT_robbers_divide_and_choose_l634_63479


namespace NUMINAMATH_GPT_kayla_score_fourth_level_l634_63431

theorem kayla_score_fourth_level 
  (score1 score2 score3 score5 score6 : ℕ) 
  (h1 : score1 = 2) 
  (h2 : score2 = 3) 
  (h3 : score3 = 5) 
  (h5 : score5 = 12) 
  (h6 : score6 = 17)
  (h_diff : ∀ n : ℕ, score2 - score1 + n = score3 - score2 + n + 1 ∧ score3 - score2 + n + 2 = score5 - score3 + n + 3 ∧ score5 - score3 + n + 4 = score6 - score5 + n + 5) :
  ∃ score4 : ℕ, score4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_kayla_score_fourth_level_l634_63431


namespace NUMINAMATH_GPT_part1_purchase_price_part2_minimum_A_l634_63442

section
variables (x y m : ℝ)

-- Part 1: Purchase price per piece
theorem part1_purchase_price (h1 : 10 * x + 15 * y = 3600) (h2 : 25 * x + 30 * y = 8100) :
  x = 180 ∧ y = 120 :=
sorry

-- Part 2: Minimum number of model A bamboo mats
theorem part2_minimum_A (h3 : x = 180) (h4 : y = 120) 
    (h5 : (260 - x) * m + (180 - y) * (60 - m) ≥ 4400) : 
  m ≥ 40 :=
sorry
end

end NUMINAMATH_GPT_part1_purchase_price_part2_minimum_A_l634_63442


namespace NUMINAMATH_GPT_min_value_of_a_k_l634_63484

-- Define the conditions for our proof in Lean

-- a_n is a positive arithmetic sequence
def is_positive_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ m, a (m + 1) = a m + d

-- Given inequality condition for the sequence
def inequality_condition (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

-- Prove the minimum value of a_k
theorem min_value_of_a_k (a : ℕ → ℝ) (k : ℕ) (h_arith : is_positive_arithmetic_seq a) (h_ineq : inequality_condition a k) :
  a k = 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_k_l634_63484


namespace NUMINAMATH_GPT_system_of_equations_solution_l634_63492

theorem system_of_equations_solution (x y z : ℕ) :
  x + y + z = 6 ∧ xy + yz + zx = 11 ∧ xyz = 6 ↔
  (x, y, z) = (1, 2, 3) ∨ (x, y, z) = (1, 3, 2) ∨ 
  (x, y, z) = (2, 1, 3) ∨ (x, y, z) = (2, 3, 1) ∨ 
  (x, y, z) = (3, 1, 2) ∨ (x, y, z) = (3, 2, 1) := by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l634_63492


namespace NUMINAMATH_GPT_sum_of_digits_of_N_plus_2021_is_10_l634_63469

-- The condition that N is the smallest positive integer whose digits add to 41.
def smallest_integer_with_digit_sum_41 (N : ℕ) : Prop :=
  (N > 0) ∧ ((N.digits 10).sum = 41)

-- The Lean 4 statement to prove the problem.
theorem sum_of_digits_of_N_plus_2021_is_10 :
  ∃ N : ℕ, smallest_integer_with_digit_sum_41 N ∧ ((N + 2021).digits 10).sum = 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_plus_2021_is_10_l634_63469


namespace NUMINAMATH_GPT_no_integer_roots_l634_63490

  theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 4 * x + 24 ≠ 0 :=
  by
    sorry
  
end NUMINAMATH_GPT_no_integer_roots_l634_63490


namespace NUMINAMATH_GPT_triangle_DEF_rotate_180_D_l634_63440

def rotate_180_degrees_clockwise (E D : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let ED := (D.1 - E.1, D.2 - E.2)
  (E.1 - ED.1, E.2 - ED.2)

theorem triangle_DEF_rotate_180_D (D E F : (ℝ × ℝ))
  (hD : D = (3, 2)) (hE : E = (6, 5)) (hF : F = (6, 2)) :
  rotate_180_degrees_clockwise E D = (9, 8) :=
by
  rw [hD, hE, rotate_180_degrees_clockwise]
  sorry

end NUMINAMATH_GPT_triangle_DEF_rotate_180_D_l634_63440


namespace NUMINAMATH_GPT_weight_of_second_piece_l634_63498

-- Given conditions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

def weight (density : ℚ) (area : ℕ) : ℚ := density * area

-- Given dimensions and weight of the first piece
def length1 : ℕ := 4
def width1 : ℕ := 3
def area1 : ℕ := area length1 width1
def weight1 : ℚ := 18

-- Given dimensions of the second piece
def length2 : ℕ := 6
def width2 : ℕ := 4
def area2 : ℕ := area length2 width2

-- Uniform density implies a proportional relationship between area and weight
def density1 : ℚ := weight1 / area1

-- The main theorem to prove
theorem weight_of_second_piece :
  weight density1 area2 = 36 :=
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_weight_of_second_piece_l634_63498


namespace NUMINAMATH_GPT_graph_single_point_l634_63467

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end NUMINAMATH_GPT_graph_single_point_l634_63467


namespace NUMINAMATH_GPT_sin_value_l634_63419

theorem sin_value (α : ℝ) 
  (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_value_l634_63419


namespace NUMINAMATH_GPT_message_forwarding_time_l634_63456

theorem message_forwarding_time :
  ∃ n : ℕ, (∀ m : ℕ, (∀ p : ℕ, (∀ q : ℕ, 1 + (2 * (2 ^ n)) - 1 = 2047)) ∧ n = 10) :=
sorry

end NUMINAMATH_GPT_message_forwarding_time_l634_63456


namespace NUMINAMATH_GPT_dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l634_63420

-- Question 1
theorem dual_expr_result (m n : ℝ) (h1 : m = 2 - Real.sqrt 3) (h2 : n = 2 + Real.sqrt 3) :
  m * n = 1 :=
sorry

-- Question 2
theorem solve_sqrt_eq_16 (x : ℝ) (h : Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) :
  x = 39 :=
sorry

-- Question 3
theorem solve_sqrt_rational_eq_4x (x : ℝ) (h : Real.sqrt (4 * x^2 + 6 * x - 5) + Real.sqrt (4 * x^2 - 2 * x - 5) = 4 * x) :
  x = 3 :=
sorry

end NUMINAMATH_GPT_dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l634_63420


namespace NUMINAMATH_GPT_fraction_spent_on_raw_material_l634_63406

variable (C : ℝ)
variable (x : ℝ)

theorem fraction_spent_on_raw_material :
  C - x * C - (1/10) * (C * (1 - x)) = 0.675 * C → x = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_raw_material_l634_63406


namespace NUMINAMATH_GPT_temperature_relationship_l634_63447

def temperature (t : ℕ) (T : ℕ) :=
  ∀ t < 10, T = 7 * t + 30

-- Proof not required, hence added sorry.
theorem temperature_relationship (t : ℕ) (T : ℕ) (h : t < 10) :
  temperature t T :=
by {
  sorry
}

end NUMINAMATH_GPT_temperature_relationship_l634_63447


namespace NUMINAMATH_GPT_equilateral_triangle_circumradius_ratio_l634_63438

variables (B b S s : ℝ)

-- Given two equilateral triangles with side lengths B and b, and respectively circumradii S and s
-- B and b are not equal
-- Prove that S / s = B / b
theorem equilateral_triangle_circumradius_ratio (hBneqb : B ≠ b)
  (hS : S = B * Real.sqrt 3 / 3)
  (hs : s = b * Real.sqrt 3 / 3) : S / s = B / b :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_circumradius_ratio_l634_63438


namespace NUMINAMATH_GPT_mary_picked_nine_lemons_l634_63449

def num_lemons_sally := 7
def total_num_lemons := 16
def num_lemons_mary := total_num_lemons - num_lemons_sally

theorem mary_picked_nine_lemons :
  num_lemons_mary = 9 := by
  sorry

end NUMINAMATH_GPT_mary_picked_nine_lemons_l634_63449


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_negative_root_l634_63432

def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem sufficient_but_not_necessary_condition_for_negative_root 
  (a : ℝ) (h : a < 0) : 
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) ∧ 
  (∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0 ∧ x < 0) → a ≤ 0) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_negative_root_l634_63432


namespace NUMINAMATH_GPT_problem_y_eq_l634_63491

theorem problem_y_eq (y : ℝ) (h : y^3 - 3*y = 9) : y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
  sorry

end NUMINAMATH_GPT_problem_y_eq_l634_63491


namespace NUMINAMATH_GPT_evaluate_i_powers_sum_l634_63465

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end NUMINAMATH_GPT_evaluate_i_powers_sum_l634_63465


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l634_63422

noncomputable def eq1_solution1 := -2 + Real.sqrt 5
noncomputable def eq1_solution2 := -2 - Real.sqrt 5

noncomputable def eq2_solution1 := 3
noncomputable def eq2_solution2 := 1

theorem solve_eq1 (x : ℝ) :
  x^2 + 4 * x - 1 = 0 → (x = eq1_solution1 ∨ x = eq1_solution2) :=
by
  sorry

theorem solve_eq2 (x : ℝ) :
  (x - 3)^2 + 2 * x * (x - 3) = 0 → (x = eq2_solution1 ∨ x = eq2_solution2) :=
by 
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l634_63422


namespace NUMINAMATH_GPT_volume_of_prism_in_cubic_feet_l634_63486

theorem volume_of_prism_in_cubic_feet:
  let length_yd := 1
  let width_yd := 2
  let height_yd := 3
  let yard_to_feet := 3
  let length_ft := length_yd * yard_to_feet
  let width_ft := width_yd * yard_to_feet
  let height_ft := height_yd * yard_to_feet
  let volume := length_ft * width_ft * height_ft
  volume = 162 := by
  sorry

end NUMINAMATH_GPT_volume_of_prism_in_cubic_feet_l634_63486


namespace NUMINAMATH_GPT_hawkeye_remaining_money_l634_63458

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end NUMINAMATH_GPT_hawkeye_remaining_money_l634_63458


namespace NUMINAMATH_GPT_solution_to_problem_l634_63450

theorem solution_to_problem (x y : ℕ) (h : (2*x - 5) * (2*y - 5) = 25) : x + y = 10 ∨ x + y = 18 := by
  sorry

end NUMINAMATH_GPT_solution_to_problem_l634_63450


namespace NUMINAMATH_GPT_cannot_be_six_l634_63453

theorem cannot_be_six (n r : ℕ) (h_n : n = 6) : 3 * n ≠ 4 * r :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_six_l634_63453


namespace NUMINAMATH_GPT_cone_volume_filled_88_8900_percent_l634_63437

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_filled_88_8900_percent_l634_63437


namespace NUMINAMATH_GPT_farmer_apples_count_l634_63421

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end NUMINAMATH_GPT_farmer_apples_count_l634_63421


namespace NUMINAMATH_GPT_find_C_l634_63477

theorem find_C (A B C D : ℕ) (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_eq : 4000 + 100 * A + 50 + B + (1000 * C + 200 + 10 * D + 7) = 7070) : C = 2 :=
sorry

end NUMINAMATH_GPT_find_C_l634_63477


namespace NUMINAMATH_GPT_simplify_evaluate_expr_l634_63400

theorem simplify_evaluate_expr (x : ℕ) (h : x = 2023) : (x + 1) ^ 2 - x * (x + 1) = 2024 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expr_l634_63400


namespace NUMINAMATH_GPT_fraction_meaningful_l634_63452

theorem fraction_meaningful (x : ℝ) : (x ≠ -1) ↔ (∃ k : ℝ, k = 1 / (x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l634_63452


namespace NUMINAMATH_GPT_triangle_ABC_no_common_factor_l634_63441

theorem triangle_ABC_no_common_factor (a b c : ℕ) (h_coprime: Nat.gcd (Nat.gcd a b) c = 1)
  (h_angleB_eq_2angleC : True) (h_b_lt_600 : b < 600) : False :=
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_no_common_factor_l634_63441


namespace NUMINAMATH_GPT_tangent_line_m_value_l634_63427

theorem tangent_line_m_value : 
  (∀ m : ℝ, ∃ (x y : ℝ), (x = my + 2) ∧ (x + one)^2 + (y + one)^2 = 2) → 
  (m = 1 ∨ m = -7) :=
  sorry

end NUMINAMATH_GPT_tangent_line_m_value_l634_63427


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l634_63410

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + 4 * b^2 + 4 * b - 4 * a + 5 ≥ 0 ∧ (a^2 + 4 * b^2 + 4 * b - 4 * a + 5 = 0 ↔ (a = 2 ∧ b = -1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l634_63410


namespace NUMINAMATH_GPT_percentage_of_women_in_study_group_l634_63424

variable (W : ℝ) -- W is the percentage of women in the study group in decimal form

-- Given conditions as hypotheses
axiom h1 : 0 < W ∧ W <= 1         -- W represents a percentage, so it must be between 0 and 1.
axiom h2 : 0.40 * W = 0.28         -- 40 percent of women are lawyers, and the probability of selecting a woman lawyer is 0.28.

-- The statement to prove
theorem percentage_of_women_in_study_group : W = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_women_in_study_group_l634_63424


namespace NUMINAMATH_GPT_problem_correct_answer_l634_63409

theorem problem_correct_answer :
  (∀ (P L : Type) (passes_through_point : P → L → Prop) (parallel_to : L → L → Prop),
    (∀ (l₁ l₂ : L) (p : P), passes_through_point p l₁ ∧ ¬ passes_through_point p l₂ → (∃! l : L, passes_through_point p l ∧ parallel_to l l₂)) ->
  (∃ (l₁ l₂ : L) (A : P), passes_through_point A l₁ ∧ ¬ passes_through_point A l₂ ∧ ∃ l : L, passes_through_point A l ∧ parallel_to l l₂) ) :=
sorry

end NUMINAMATH_GPT_problem_correct_answer_l634_63409


namespace NUMINAMATH_GPT_min_bn_of_arithmetic_sequence_l634_63413

theorem min_bn_of_arithmetic_sequence :
  (∃ n : ℕ, 1 ≤ n ∧ b_n = n + 1 + 7 / n ∧ (∀ m : ℕ, 1 ≤ m → b_m ≥ b_n)) :=
sorry

def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

def S_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else n * (n + 1) / 2

def b_n (n : ℕ) : ℕ :=
  if n = 0 then 0 else (2 * S_n n + 7) / n

end NUMINAMATH_GPT_min_bn_of_arithmetic_sequence_l634_63413


namespace NUMINAMATH_GPT_exists_base_and_digit_l634_63415

def valid_digit_in_base (B : ℕ) (V : ℕ) : Prop :=
  V^2 % B = V ∧ V ≠ 0 ∧ V ≠ 1

theorem exists_base_and_digit :
  ∃ B V, valid_digit_in_base B V :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_base_and_digit_l634_63415


namespace NUMINAMATH_GPT_time_to_fill_bottle_l634_63418

-- Definitions
def flow_rate := 500 / 6 -- mL per second
def volume := 250 -- mL

-- Target theorem
theorem time_to_fill_bottle (r : ℝ) (v : ℝ) (t : ℝ) (h : r = flow_rate) (h2 : v = volume) : t = 3 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_bottle_l634_63418


namespace NUMINAMATH_GPT_find_a_l634_63446

noncomputable def f (a x : ℝ) : ℝ := a * x * Real.log x

theorem find_a (a : ℝ) (h : (deriv (f a)) e = 3) : a = 3 / 2 :=
by
-- placeholder for the proof
sorry

end NUMINAMATH_GPT_find_a_l634_63446


namespace NUMINAMATH_GPT_minimum_distance_after_9_minutes_l634_63414

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end NUMINAMATH_GPT_minimum_distance_after_9_minutes_l634_63414


namespace NUMINAMATH_GPT_ellipse_hyperbola_foci_l634_63472

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_foci_l634_63472


namespace NUMINAMATH_GPT_simplify_expression_solve_fractional_eq_l634_63478

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_solve_fractional_eq_l634_63478


namespace NUMINAMATH_GPT_solve_problem_l634_63402

theorem solve_problem
    (product_trailing_zeroes : ∃ (x y z w v u p q r : ℕ), (10 ∣ (x * y * z * w * v * u * p * q * r)) ∧ B = 0)
    (digit_sequences : (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9) % 10 = 8 ∧
                       (11 * 12 * 13 * 14 * 15 * 16 * 17 * 18 * 19) % 10 = 4 ∧
                       (21 * 22 * 23 * 24 * 25 * 26 * 27 * 28 * 29) % 10 = 4 ∧
                       (31 * 32 * 33 * 34 * 35) % 10 = 4 ∧
                       A = 2 ∧ B = 0)
    (divisibility_rule_11 : ∀ C D, (71 + C) - (68 + D) = 11 → C - D = -3 ∨ C - D = 8)
    (divisibility_rule_9 : ∀ C D, (139 + C + D) % 9 = 0 → C + D = 5 ∨ C + D = 14)
    (system_of_equations : ∀ C D, (C - D = -3 ∧ C + D = 5) → (C = 1 ∧ D = 4)) :
  A = 2 ∧ B = 0 ∧ C = 1 ∧ D = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l634_63402


namespace NUMINAMATH_GPT_larger_number_is_eight_l634_63416

variable {x y : ℝ}

theorem larger_number_is_eight (h1 : x - y = 3) (h2 : x^2 - y^2 = 39) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_eight_l634_63416


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l634_63494

theorem area_of_triangle_ABC (AB CD : ℝ) (height : ℝ) (h1 : CD = 3 * AB) (h2 : AB * height + CD * height = 48) :
  (1/2) * AB * height = 6 :=
by
  have trapezoid_area : AB * height + CD * height = 48 := h2
  have length_relation : CD = 3 * AB := h1
  have area_triangle_ABC := 6
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l634_63494


namespace NUMINAMATH_GPT_radius_ratio_l634_63493

noncomputable def ratio_of_radii (V1 V2 : ℝ) (R : ℝ) : ℝ := 
  (V2 / V1)^(1/3) * R 

theorem radius_ratio (V1 V2 : ℝ) (π : ℝ) (R r : ℝ) :
  V1 = 450 * π → 
  V2 = 36 * π → 
  (4 / 3) * π * R^3 = V1 →
  (4 / 3) * π * r^3 = V2 →
  r / R = 1 / (12.5)^(1/3) :=
by {
  sorry
}

end NUMINAMATH_GPT_radius_ratio_l634_63493


namespace NUMINAMATH_GPT_range_of_t_l634_63476

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_t_l634_63476


namespace NUMINAMATH_GPT_original_inhabitants_l634_63448

theorem original_inhabitants (X : ℝ) (h : 0.75 * 0.9 * X = 5265) : X = 7800 :=
by
  sorry

end NUMINAMATH_GPT_original_inhabitants_l634_63448


namespace NUMINAMATH_GPT_total_people_ball_l634_63401

theorem total_people_ball (n m : ℕ) (h1 : n + m < 50) (h2 : 3 * n = 20 * m) : n + m = 41 := 
sorry

end NUMINAMATH_GPT_total_people_ball_l634_63401


namespace NUMINAMATH_GPT_rhombus_diagonal_l634_63495

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 20) (h2 : area = 170) :
  (area = (d1 * d2) / 2) → d2 = 17 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l634_63495


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l634_63412

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Conditions
def common_difference := d = 2
def value_a_2007 := a 2007 = 2007

-- Question to be proved
theorem arithmetic_sequence_term :
  common_difference d →
  value_a_2007 a →
  a 2009 = 2011 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l634_63412


namespace NUMINAMATH_GPT_reimbursement_correct_l634_63435

-- Define the days and miles driven each day
def miles_monday : ℕ := 18
def miles_tuesday : ℕ := 26
def miles_wednesday : ℕ := 20
def miles_thursday : ℕ := 20
def miles_friday : ℕ := 16

-- Define the mileage rate
def mileage_rate : ℝ := 0.36

-- Define the total miles driven
def total_miles_driven : ℕ := miles_monday + miles_tuesday + miles_wednesday + miles_thursday + miles_friday

-- Define the total reimbursement
def reimbursement : ℝ := total_miles_driven * mileage_rate

-- Prove that the reimbursement is $36
theorem reimbursement_correct : reimbursement = 36 := by
  sorry

end NUMINAMATH_GPT_reimbursement_correct_l634_63435


namespace NUMINAMATH_GPT_sqrt_expression_l634_63475

theorem sqrt_expression (x : ℝ) : 2 - x ≥ 0 ↔ x ≤ 2 := sorry

end NUMINAMATH_GPT_sqrt_expression_l634_63475


namespace NUMINAMATH_GPT_S_30_zero_l634_63417

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {n : ℕ} 

-- Definitions corresponding to the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a_n n = a1 + d * n

def sum_arithmetic_sequence (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
  
-- The given conditions
axiom S_eq (S_10 S_20 : ℝ) : S 10 = S 20

-- The theorem we need to prove
theorem S_30_zero (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_arithmetic_sequence S a_n)
  (h_eq : S 10 = S 20) :
  S 30 = 0 :=
sorry

end NUMINAMATH_GPT_S_30_zero_l634_63417
