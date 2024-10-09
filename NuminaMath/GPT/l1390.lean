import Mathlib

namespace boundary_points_distance_probability_l1390_139068

theorem boundary_points_distance_probability
  (a b c : ℕ)
  (h1 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → (|x - y| ≥ 1 / 2 → True))
  (h2 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → True)
  (h3 : ∃ a b c : ℕ, a - b * Real.pi = 2 ∧ c = 4 ∧ Int.gcd (Int.ofNat a) (Int.gcd (Int.ofNat b) (Int.ofNat c)) = 1) :
  (a + b + c = 62) := sorry

end boundary_points_distance_probability_l1390_139068


namespace worker_savings_l1390_139011

theorem worker_savings (P : ℝ) (f : ℝ) (h : 12 * f * P = 4 * (1 - f) * P) : f = 1 / 4 :=
by
  have h1 : 12 * f * P = 4 * (1 - f) * P := h
  have h2 : P ≠ 0 := sorry  -- P should not be 0 for the worker to have a meaningful income.
  field_simp [h2] at h1
  linarith

end worker_savings_l1390_139011


namespace arithmetic_sequence_common_difference_l1390_139002

variable (a₁ d : ℝ)

def sum_odd := 5 * a₁ + 20 * d
def sum_even := 5 * a₁ + 25 * d

theorem arithmetic_sequence_common_difference 
  (h₁ : sum_odd a₁ d = 15) 
  (h₂ : sum_even a₁ d = 30) :
  d = 3 := 
by
  sorry

end arithmetic_sequence_common_difference_l1390_139002


namespace angles_in_first_or_third_quadrant_l1390_139072

noncomputable def angles_first_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}

noncomputable def angles_third_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}

theorem angles_in_first_or_third_quadrant :
  ∃ S1 S2 : Set ℝ, 
    (S1 = {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}) ∧
    (S2 = {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}) ∧
    (angles_first_quadrant_set = S1 ∧ angles_third_quadrant_set = S2)
  :=
sorry

end angles_in_first_or_third_quadrant_l1390_139072


namespace no_positive_integer_triples_l1390_139075

theorem no_positive_integer_triples (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : ¬ (x^2 + y^2 + 41 = 2^n) :=
  sorry

end no_positive_integer_triples_l1390_139075


namespace sum_of_coordinates_A_l1390_139093

-- Define the points A, B, and C and the given conditions
variables (A B C : ℝ × ℝ)
variables (h_ratio1 : dist A C / dist A B = 1 / 3)
variables (h_ratio2 : dist B C / dist A B = 1 / 3)
variables (h_B : B = (2, 8))
variables (h_C : C = (0, 2))

-- Lean 4 statement to prove the sum of the coordinates of A is -14
theorem sum_of_coordinates_A : (A.1 + A.2) = -14 :=
sorry

end sum_of_coordinates_A_l1390_139093


namespace find_base_b_l1390_139045

theorem find_base_b : ∃ b : ℕ, (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 ∧ b = 7 := 
by {
  sorry
}

end find_base_b_l1390_139045


namespace minimal_volume_block_l1390_139070

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end minimal_volume_block_l1390_139070


namespace power_difference_of_squares_l1390_139014

theorem power_difference_of_squares : (((7^2 - 3^2) : ℤ)^4) = 2560000 := by
  sorry

end power_difference_of_squares_l1390_139014


namespace simplest_quadratic_radical_l1390_139095

noncomputable def optionA := Real.sqrt 7
noncomputable def optionB := Real.sqrt 9
noncomputable def optionC := Real.sqrt 12
noncomputable def optionD := Real.sqrt (2 / 3)

theorem simplest_quadratic_radical :
  optionA = Real.sqrt 7 ∧
  optionB = Real.sqrt 9 ∧
  optionC = Real.sqrt 12 ∧
  optionD = Real.sqrt (2 / 3) ∧
  (optionB = 3 ∧ optionC = 2 * Real.sqrt 3 ∧ optionD = Real.sqrt 6 / 3) ∧
  (optionA < 3 ∧ optionA < 2 * Real.sqrt 3 ∧ optionA < Real.sqrt 6 / 3) :=
  by {
    sorry
  }

end simplest_quadratic_radical_l1390_139095


namespace problem_a_l1390_139043

theorem problem_a (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  Int.floor (5 * x) + Int.floor (5 * y) ≥ Int.floor (3 * x + y) + Int.floor (3 * y + x) :=
sorry

end problem_a_l1390_139043


namespace find_positive_number_l1390_139098

theorem find_positive_number (x : ℕ) (h_pos : 0 < x) (h_equation : x * x / 100 + 6 = 10) : x = 20 :=
by
  sorry

end find_positive_number_l1390_139098


namespace airplane_fraction_l1390_139096

noncomputable def driving_time : ℕ := 195

noncomputable def airport_drive_time : ℕ := 10

noncomputable def waiting_time : ℕ := 20

noncomputable def get_off_time : ℕ := 10

noncomputable def faster_by : ℕ := 90

theorem airplane_fraction :
  ∃ x : ℕ, 195 = 40 + x + 90 ∧ x = 65 ∧ x = driving_time / 3 := sorry

end airplane_fraction_l1390_139096


namespace range_of_a_in_third_quadrant_l1390_139030

def pointInThirdQuadrant (x y : ℝ) := x < 0 ∧ y < 0

theorem range_of_a_in_third_quadrant (a : ℝ) (M : ℝ × ℝ) 
  (hM : M = (-1, a-1)) (hThirdQuad : pointInThirdQuadrant M.1 M.2) : 
  a < 1 :=
by
  sorry

end range_of_a_in_third_quadrant_l1390_139030


namespace channel_width_at_top_l1390_139089

theorem channel_width_at_top 
  (area : ℝ) (bottom_width : ℝ) (depth : ℝ) 
  (H1 : bottom_width = 6) 
  (H2 : area = 630) 
  (H3 : depth = 70) : 
  ∃ w : ℝ, (∃ H : w + 6 > 0, area = 1 / 2 * (w + bottom_width) * depth) ∧ w = 12 :=
by
  sorry

end channel_width_at_top_l1390_139089


namespace lcm_220_504_l1390_139067

/-- The least common multiple of 220 and 504 is 27720. -/
theorem lcm_220_504 : Nat.lcm 220 504 = 27720 :=
by
  -- This is the final statement of the theorem. The proof is not provided and marked with 'sorry'.
  sorry

end lcm_220_504_l1390_139067


namespace sum_of_interior_diagonals_l1390_139061

theorem sum_of_interior_diagonals (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 50) (h2 : x * y + y * z + z * x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 :=
by 
  sorry

end sum_of_interior_diagonals_l1390_139061


namespace conversion_rates_l1390_139057

noncomputable def teamADailyConversionRate (a b : ℝ) := 1.2 * b
noncomputable def teamBDailyConversionRate (a b : ℝ) := b

theorem conversion_rates (total_area : ℝ) (b : ℝ) (h1 : total_area = 1500) (h2 : b = 50) 
    (h3 : teamADailyConversionRate 1500 b * b = 1.2) 
    (h4 : teamBDailyConversionRate 1500 b = b) 
    (h5 : (1500 / teamBDailyConversionRate 1500 b) - 5 = 1500 / teamADailyConversionRate 1500 b) :
  teamADailyConversionRate 1500 b = 60 ∧ teamBDailyConversionRate 1500 b = 50 := 
by
  sorry

end conversion_rates_l1390_139057


namespace pencil_eraser_cost_l1390_139029

/-- Oscar buys 13 pencils and 3 erasers for 100 cents. A pencil costs more than an eraser, 
    and both items cost a whole number of cents. 
    We need to prove that the total cost of one pencil and one eraser is 10 cents. -/
theorem pencil_eraser_cost (p e : ℕ) (h1 : 13 * p + 3 * e = 100) (h2 : p > e) : p + e = 10 :=
sorry

end pencil_eraser_cost_l1390_139029


namespace batsman_average_19th_inning_l1390_139085

theorem batsman_average_19th_inning (initial_avg : ℝ) 
    (scored_19th_inning : ℝ) 
    (new_avg : ℝ) 
    (h1 : scored_19th_inning = 100) 
    (h2 : new_avg = initial_avg + 2)
    (h3 : new_avg = (18 * initial_avg + 100) / 19) :
    new_avg = 64 :=
by
  have h4 : initial_avg = 62 := by
    sorry
  sorry

end batsman_average_19th_inning_l1390_139085


namespace sector_area_is_correct_l1390_139099

noncomputable def area_of_sector (r : ℝ) (α : ℝ) : ℝ := 1/2 * α * r^2

theorem sector_area_is_correct (circumference : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) 
  (h1 : circumference = 8) 
  (h2 : central_angle = 2) 
  (h3 : circumference = central_angle * r + 2 * r)
  (h4 : r = 2) : area = 4 :=
by
  have h5: area = 1/2 * central_angle * r^2 := sorry
  exact sorry

end sector_area_is_correct_l1390_139099


namespace graph_eq_pair_of_straight_lines_l1390_139009

theorem graph_eq_pair_of_straight_lines (x y : ℝ) :
  x^2 - 9*y^2 = 0 ↔ (x = 3*y ∨ x = -3*y) :=
by
  sorry

end graph_eq_pair_of_straight_lines_l1390_139009


namespace avg_daily_production_l1390_139039

theorem avg_daily_production (x y : ℕ) (h1 : x + y = 350) (h2 : 2 * x - y = 250) : x = 200 ∧ y = 150 := 
by
  sorry

end avg_daily_production_l1390_139039


namespace cl_mass_percentage_in_ccl4_l1390_139040

noncomputable def mass_percentage_of_cl_in_ccl4 : ℝ :=
  let mass_C : ℝ := 12.01
  let mass_Cl : ℝ := 35.45
  let num_Cl : ℝ := 4
  let total_mass_Cl : ℝ := num_Cl * mass_Cl
  let total_mass_CCl4 : ℝ := mass_C + total_mass_Cl
  (total_mass_Cl / total_mass_CCl4) * 100

theorem cl_mass_percentage_in_ccl4 :
  abs (mass_percentage_of_cl_in_ccl4 - 92.19) < 0.01 := 
sorry

end cl_mass_percentage_in_ccl4_l1390_139040


namespace total_paintable_area_l1390_139074

-- Define the dimensions of a bedroom
def bedroom_length : ℕ := 10
def bedroom_width : ℕ := 12
def bedroom_height : ℕ := 9

-- Define the non-paintable area per bedroom
def non_paintable_area_per_bedroom : ℕ := 74

-- Number of bedrooms
def number_of_bedrooms : ℕ := 4

-- The total paintable area that we need to prove
theorem total_paintable_area : 
  4 * (2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height) - non_paintable_area_per_bedroom) = 1288 := 
by
  sorry

end total_paintable_area_l1390_139074


namespace percentage_error_l1390_139077

-- Define the conditions
def actual_side (a : ℝ) := a
def measured_side (a : ℝ) := 1.05 * a
def actual_area (a : ℝ) := a^2
def calculated_area (a : ℝ) := (1.05 * a)^2

-- Define the statement that we need to prove
theorem percentage_error (a : ℝ) (h : a > 0) :
  (calculated_area a - actual_area a) / actual_area a * 100 = 10.25 :=
by
  -- Proof goes here
  sorry

end percentage_error_l1390_139077


namespace eden_initial_bears_l1390_139020

theorem eden_initial_bears (d_total : ℕ) (d_favorite : ℕ) (sisters : ℕ) (eden_after : ℕ) (each_share : ℕ)
  (h1 : d_total = 20)
  (h2 : d_favorite = 8)
  (h3 : sisters = 3)
  (h4 : eden_after = 14)
  (h5 : each_share = (d_total - d_favorite) / sisters)
  : (eden_after - each_share) = 10 :=
by
  sorry

end eden_initial_bears_l1390_139020


namespace fib_100_mod_5_l1390_139086

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end fib_100_mod_5_l1390_139086


namespace troy_needs_more_money_to_buy_computer_l1390_139015

theorem troy_needs_more_money_to_buy_computer :
  ∀ (price_new_computer savings sale_old_computer : ℕ),
  price_new_computer = 80 →
  savings = 50 →
  sale_old_computer = 20 →
  (price_new_computer - (savings + sale_old_computer)) = 10 :=
by
  intros price_new_computer savings sale_old_computer Hprice Hsavings Hsale
  sorry

end troy_needs_more_money_to_buy_computer_l1390_139015


namespace rods_in_one_mile_l1390_139065

-- Define the conditions as assumptions in Lean

-- 1. 1 mile = 8 furlongs
def mile_to_furlong : ℕ := 8

-- 2. 1 furlong = 220 paces
def furlong_to_pace : ℕ := 220

-- 3. 1 pace = 0.2 rods
def pace_to_rod : ℝ := 0.2

-- Define the statement to be proven
theorem rods_in_one_mile : (mile_to_furlong * furlong_to_pace * pace_to_rod) = 352 := by
  sorry

end rods_in_one_mile_l1390_139065


namespace claire_crafting_hours_l1390_139044

theorem claire_crafting_hours (H1 : 24 = 24) (H2 : 8 = 8) (H3 : 4 = 4) (H4 : 2 = 2):
  let total_hours_per_day := 24
  let sleep_hours := 8
  let cleaning_hours := 4
  let cooking_hours := 2
  let working_hours := total_hours_per_day - sleep_hours
  let remaining_hours := working_hours - (cleaning_hours + cooking_hours)
  let crafting_hours := remaining_hours / 2
  crafting_hours = 5 :=
by
  sorry

end claire_crafting_hours_l1390_139044


namespace player_A_wins_iff_n_is_odd_l1390_139082

-- Definitions of the problem conditions
structure ChessboardGame (n : ℕ) :=
  (stones : ℕ := 99)
  (playerA_first : Prop := true)
  (turns : ℕ := n * 99)

-- Statement of the problem
theorem player_A_wins_iff_n_is_odd (n : ℕ) (g : ChessboardGame n) : 
  PlayerA_has_winning_strategy ↔ n % 2 = 1 := 
sorry

end player_A_wins_iff_n_is_odd_l1390_139082


namespace cab_driver_income_l1390_139078

theorem cab_driver_income (x : ℕ) 
  (h₁ : (45 + x + 60 + 65 + 70) / 5 = 58) : x = 50 := 
by
  -- Insert the proof here
  sorry

end cab_driver_income_l1390_139078


namespace find_common_ratio_l1390_139003

noncomputable def geom_series_common_ratio (q : ℝ) : Prop :=
  ∃ (a1 : ℝ), a1 > 0 ∧ (a1 * q^2 = 18) ∧ (a1 * (1 + q + q^2) = 26)

theorem find_common_ratio (q : ℝ) :
  geom_series_common_ratio q → q = 3 :=
sorry

end find_common_ratio_l1390_139003


namespace man_l1390_139017

theorem man's_present_age (P : ℝ) 
  (h1 : P = (4/5) * P + 10)
  (h2 : P = (3/2.5) * P - 10) :
  P = 50 :=
sorry

end man_l1390_139017


namespace john_boxes_l1390_139018

theorem john_boxes
  (stan_boxes : ℕ)
  (joseph_boxes : ℕ)
  (jules_boxes : ℕ)
  (john_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : joseph_boxes = stan_boxes - 80 * stan_boxes / 100)
  (h3 : jules_boxes = joseph_boxes + 5)
  (h4 : john_boxes = jules_boxes + 20 * jules_boxes / 100) :
  john_boxes = 30 :=
by
  -- Proof will go here
  sorry

end john_boxes_l1390_139018


namespace fred_current_dimes_l1390_139037

-- Definitions based on the conditions
def original_dimes : ℕ := 7
def borrowed_dimes : ℕ := 3

-- The theorem to prove
theorem fred_current_dimes : original_dimes - borrowed_dimes = 4 := by
  sorry

end fred_current_dimes_l1390_139037


namespace sacks_per_day_l1390_139055

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (harvest_rate : ℕ)
  (h1 : total_sacks = 498)
  (h2 : days = 6)
  (h3 : harvest_rate = total_sacks / days) :
  harvest_rate = 83 := by
  sorry

end sacks_per_day_l1390_139055


namespace total_formula_portions_l1390_139046

def puppies : ℕ := 7
def feedings_per_day : ℕ := 3
def days : ℕ := 5

theorem total_formula_portions : 
  (feedings_per_day * days * puppies = 105) := 
by
  sorry

end total_formula_portions_l1390_139046


namespace geom_seq_sum_eq_six_l1390_139034

theorem geom_seq_sum_eq_six 
    (a : ℕ → ℝ) 
    (r : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * r) 
    (h_pos : ∀ n, a n > 0)
    (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) 
    : a 5 + a 7 = 6 :=
sorry

end geom_seq_sum_eq_six_l1390_139034


namespace minimum_value_on_line_l1390_139080

theorem minimum_value_on_line : ∃ (x y : ℝ), (x + y = 4) ∧ (∀ x' y', (x' + y' = 4) → (x^2 + y^2 ≤ x'^2 + y'^2)) ∧ (x^2 + y^2 = 8) :=
sorry

end minimum_value_on_line_l1390_139080


namespace range_of_a_l1390_139059

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (curve_y a) x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) 2, 0 ≤ deriv (function_f a) x) → a ≤ 0 :=
by sorry

end range_of_a_l1390_139059


namespace customers_left_l1390_139025

theorem customers_left (initial_customers remaining_tables people_per_table customers_left : ℕ)
    (h_initial : initial_customers = 62)
    (h_tables : remaining_tables = 5)
    (h_people : people_per_table = 9)
    (h_left : customers_left = initial_customers - remaining_tables * people_per_table) : 
    customers_left = 17 := 
    by 
        -- Provide the proof here 
        sorry

end customers_left_l1390_139025


namespace length_of_BC_l1390_139084

theorem length_of_BC (BD CD : ℝ) (h1 : BD = 3 + 3 * BD) (h2 : CD = 2 + 2 * CD) (h3 : 4 * BD + 3 * CD + 5 = 20) : 2 * CD + 2 = 4 :=
by {
  sorry
}

end length_of_BC_l1390_139084


namespace right_triangle_second_arm_square_l1390_139076

theorem right_triangle_second_arm_square :
  ∀ (k : ℤ) (a : ℤ) (c : ℤ) (b : ℤ),
  a = 2 * k + 1 → 
  c = 2 * k + 3 → 
  a^2 + b^2 = c^2 → 
  b^2 ≠ a * c ∧ b^2 ≠ (c / a) ∧ b^2 ≠ (a + c) ∧ b^2 ≠ (c - a) :=
by sorry

end right_triangle_second_arm_square_l1390_139076


namespace binary_add_sub_l1390_139010

theorem binary_add_sub:
  let a := 0b10110
  let b := 0b1010
  let c := 0b11100
  let d := 0b1110
  a + b - c + d = 0b01110 := by
  sorry

end binary_add_sub_l1390_139010


namespace line_equation_l1390_139032

theorem line_equation (P A B : ℝ × ℝ) (h1 : P = (-1, 3)) (h2 : A = (1, 2)) (h3 : B = (3, 1)) :
  ∃ c : ℝ, (x - 2*y + c = 0) ∧ (4*x - 2*y - 5 = 0) :=
by
  sorry

end line_equation_l1390_139032


namespace gcd_lcm_product_eq_prod_l1390_139028

theorem gcd_lcm_product_eq_prod (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := 
sorry

end gcd_lcm_product_eq_prod_l1390_139028


namespace total_players_must_be_square_l1390_139007

variables (k m : ℕ)
def n : ℕ := k + m

theorem total_players_must_be_square (h: (k*(k-1) / 2) + (m*(m-1) / 2) = k * m) :
  ∃ (s : ℕ), n = s^2 :=
by sorry

end total_players_must_be_square_l1390_139007


namespace part1_l1390_139054

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 :=
by
  sorry

end part1_l1390_139054


namespace john_marble_choices_l1390_139047

open Nat

theorem john_marble_choices :
  (choose 4 2) * (choose 12 3) = 1320 :=
by
  sorry

end john_marble_choices_l1390_139047


namespace average_speed_of_Car_X_l1390_139097

noncomputable def average_speed_CarX (V_x : ℝ) : Prop :=
  let head_start_time := 1.2
  let distance_traveled_by_CarX := 98
  let speed_CarY := 50
  let time_elapsed := distance_traveled_by_CarX / speed_CarY
  (distance_traveled_by_CarX / time_elapsed) = V_x

theorem average_speed_of_Car_X : average_speed_CarX 50 :=
  sorry

end average_speed_of_Car_X_l1390_139097


namespace lisa_investment_in_stocks_l1390_139071

-- Definitions for the conditions
def total_investment (r : ℝ) : Prop := r + 7 * r = 200000
def stock_investment (r : ℝ) : ℝ := 7 * r

-- Given the conditions, we need to prove the amount invested in stocks
theorem lisa_investment_in_stocks (r : ℝ) (h : total_investment r) : stock_investment r = 175000 :=
by
  -- proof goes here
  sorry

end lisa_investment_in_stocks_l1390_139071


namespace cannot_cover_chessboard_with_one_corner_removed_l1390_139049

theorem cannot_cover_chessboard_with_one_corner_removed :
  ¬ (∃ (f : Fin (8*8 - 1) → Fin (64-1) × Fin (64-1)), 
        (∀ (i j : Fin (64-1)), 
          i ≠ j → f i ≠ f j) ∧ 
        (∀ (i : Fin (8 * 8 - 1)), 
          (f i).fst + (f i).snd = 2)) :=
by
  sorry

end cannot_cover_chessboard_with_one_corner_removed_l1390_139049


namespace programmer_debugging_hours_l1390_139053

theorem programmer_debugging_hours
    (total_hours : ℕ)
    (flow_chart_fraction : ℚ)
    (coding_fraction : ℚ)
    (meeting_fraction : ℚ)
    (flow_chart_hours : ℚ)
    (coding_hours : ℚ)
    (meeting_hours : ℚ)
    (debugging_hours : ℚ)
    (H1 : total_hours = 192)
    (H2 : flow_chart_fraction = 3 / 10)
    (H3 : coding_fraction = 3 / 8)
    (H4 : meeting_fraction = 1 / 5)
    (H5 : flow_chart_hours = flow_chart_fraction * total_hours)
    (H6 : coding_hours = coding_fraction * total_hours)
    (H7 : meeting_hours = meeting_fraction * total_hours)
    (H8 : debugging_hours = total_hours - (flow_chart_hours + coding_hours + meeting_hours))
    :
    debugging_hours = 24 :=
by 
  sorry

end programmer_debugging_hours_l1390_139053


namespace colby_mangoes_l1390_139033

def mangoes_still_have (t m k : ℕ) : ℕ :=
  let r1 := t - m
  let r2 := r1 / 2
  let r3 := r1 - r2
  r3 * k

theorem colby_mangoes (t m k : ℕ) (h_t : t = 60) (h_m : m = 20) (h_k : k = 8) :
  mangoes_still_have t m k = 160 :=
by
  sorry

end colby_mangoes_l1390_139033


namespace percentage_of_orange_and_watermelon_juice_l1390_139048

-- Define the total volume of the drink
def total_volume := 150

-- Define the volume of grape juice in the drink
def grape_juice_volume := 45

-- Define the percentage calculation for grape juice
def grape_juice_percentage := (grape_juice_volume / total_volume) * 100

-- Define the remaining percentage that is made of orange and watermelon juices
def remaining_percentage := 100 - grape_juice_percentage

-- Define the percentage of orange and watermelon juice being the same
def orange_and_watermelon_percentage := remaining_percentage / 2

theorem percentage_of_orange_and_watermelon_juice : 
  orange_and_watermelon_percentage = 35 :=
by
  -- The proof steps would go here
  sorry

end percentage_of_orange_and_watermelon_juice_l1390_139048


namespace min_focal_length_of_hyperbola_l1390_139008

theorem min_focal_length_of_hyperbola
  (a b k : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_area : k * b = 8) :
  2 * Real.sqrt (a^2 + b^2) = 8 :=
sorry -- proof to be completed

end min_focal_length_of_hyperbola_l1390_139008


namespace trigonometric_identity_solution_l1390_139021

open Real

theorem trigonometric_identity_solution (k n l : ℤ) (x : ℝ) 
  (h : 2 * cos x ≠ sin x) : 
  (sin x ^ 3 + cos x ^ 3) / (2 * cos x - sin x) = cos (2 * x) ↔
  (∃ k : ℤ, x = (π / 2) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (π / 4) * (4 * n - 1)) ∨
  (∃ l : ℤ, x = arctan (1 / 2) + π * l) :=
sorry

end trigonometric_identity_solution_l1390_139021


namespace find_k_l1390_139005

-- Define the matrix M
def M (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2, -1, 3], ![0, 4, -k], ![3, -1, 2]]

-- Define the problem statement
theorem find_k (k : ℝ) (h : Matrix.det (M k) = -20) : k = 0 := by
  sorry

end find_k_l1390_139005


namespace amc_proposed_by_Dorlir_Ahmeti_Albania_l1390_139090

-- Define the problem statement, encapsulating the conditions and the final inequality.
theorem amc_proposed_by_Dorlir_Ahmeti_Albania
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cond : a * b + b * c + c * a = 3) :
  (a / Real.sqrt (a^3 + 5) + b / Real.sqrt (b^3 + 5) + c / Real.sqrt (c^3 + 5) ≤ Real.sqrt 6 / 2) := 
by 
  sorry -- Proof steps go here, which are omitted as per the requirement.

end amc_proposed_by_Dorlir_Ahmeti_Albania_l1390_139090


namespace sum_eq_3_or_7_l1390_139022

theorem sum_eq_3_or_7 {x y z : ℝ} 
  (h1 : x + y / z = 2)
  (h2 : y + z / x = 2)
  (h3 : z + x / y = 2) : 
  x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_eq_3_or_7_l1390_139022


namespace sum_of_cos_series_l1390_139051

theorem sum_of_cos_series :
  6 * Real.cos (18 * Real.pi / 180) + 2 * Real.cos (36 * Real.pi / 180) + 
  4 * Real.cos (54 * Real.pi / 180) + 6 * Real.cos (72 * Real.pi / 180) + 
  8 * Real.cos (90 * Real.pi / 180) + 10 * Real.cos (108 * Real.pi / 180) + 
  12 * Real.cos (126 * Real.pi / 180) + 14 * Real.cos (144 * Real.pi / 180) + 
  16 * Real.cos (162 * Real.pi / 180) + 18 * Real.cos (180 * Real.pi / 180) + 
  20 * Real.cos (198 * Real.pi / 180) + 22 * Real.cos (216 * Real.pi / 180) + 
  24 * Real.cos (234 * Real.pi / 180) + 26 * Real.cos (252 * Real.pi / 180) + 
  28 * Real.cos (270 * Real.pi / 180) + 30 * Real.cos (288 * Real.pi / 180) + 
  32 * Real.cos (306 * Real.pi / 180) + 34 * Real.cos (324 * Real.pi / 180) + 
  36 * Real.cos (342 * Real.pi / 180) + 38 * Real.cos (360 * Real.pi / 180) = 10 :=
by
  sorry

end sum_of_cos_series_l1390_139051


namespace convert_300_degree_to_radian_l1390_139092

theorem convert_300_degree_to_radian : (300 : ℝ) * π / 180 = 5 * π / 3 :=
by
  sorry

end convert_300_degree_to_radian_l1390_139092


namespace factorization_mn_l1390_139035

variable (m n : ℝ) -- Declare m and n as arbitrary real numbers.

theorem factorization_mn (m n : ℝ) : m^2 - m * n = m * (m - n) := by
  sorry

end factorization_mn_l1390_139035


namespace diana_age_l1390_139073

open Classical

theorem diana_age :
  ∃ (D : ℚ), (∃ (C E : ℚ), C = 4 * D ∧ E = D + 5 ∧ C = E) ∧ D = 5/3 :=
by
  -- Definitions and conditions are encapsulated in the existential quantifiers and the proof concludes with D = 5/3.
  sorry

end diana_age_l1390_139073


namespace gcd_lcm_product_l1390_139036

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 1350 :=
by
  rw [h1, h2]
  sorry

end gcd_lcm_product_l1390_139036


namespace min_value_quadratic_l1390_139079

theorem min_value_quadratic (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) :
  (∀ x, (a * x^2 + 2 * x + b = 0) → x = -1 / a) →
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ (∀ a b, a > b → b > 0 → a * b = 1 →
     c ≤ (a^2 + b^2) / (a - b)) :=
by
  sorry

end min_value_quadratic_l1390_139079


namespace optimal_play_winner_l1390_139038

-- Definitions for the conditions
def chessboard_size (K N : ℕ) : Prop := True
def rook_initial_position (K N : ℕ) : (ℕ × ℕ) :=
  (K, N)
def move (r : ℕ × ℕ) (direction : ℕ) : (ℕ × ℕ) :=
  if direction = 0 then (r.1 - 1, r.2)
  else (r.1, r.2 - 1)
def rook_cannot_move (r : ℕ × ℕ) : Prop :=
  r.1 = 0 ∨ r.2 = 0

-- Theorem to prove the winner given the conditions
theorem optimal_play_winner (K N : ℕ) :
  (K = N → ∃ player : ℕ, player = 2) ∧ (K ≠ N → ∃ player : ℕ, player = 1) :=
by
  sorry

end optimal_play_winner_l1390_139038


namespace quadratic_distinct_real_roots_l1390_139056

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end quadratic_distinct_real_roots_l1390_139056


namespace find_m_for_line_passing_through_circle_center_l1390_139042

theorem find_m_for_line_passing_through_circle_center :
  ∀ (m : ℝ), (∀ (x y : ℝ), 2 * x + y + m = 0 ↔ (x - 1)^2 + (y + 2)^2 = 5) → m = 0 :=
by
  intro m
  intro h
  -- Here we construct that the center (1, -2) must lie on the line 2x + y + m = 0
  -- using the given condition of the circle center.
  have center := h 1 (-2)
  -- solving for the equation at the point (1, -2) must yield m = 0
  sorry

end find_m_for_line_passing_through_circle_center_l1390_139042


namespace elevator_stop_time_l1390_139004

def time_to_reach_top (stories time_per_story : Nat) : Nat := stories * time_per_story

def total_time_with_stops (stories time_per_story stop_time : Nat) : Nat :=
  stories * time_per_story + (stories - 1) * stop_time

theorem elevator_stop_time (stories : Nat) (lola_time_per_story elevator_time_per_story total_elevator_time_to_top stop_time_per_floor : Nat)
  (lola_total_time : Nat) (is_slower : Bool)
  (h_lola: lola_total_time = time_to_reach_top stories lola_time_per_story)
  (h_slower: total_elevator_time_to_top = if is_slower then lola_total_time else 220)
  (h_no_stops: time_to_reach_top stories elevator_time_per_story + (stories - 1) * stop_time_per_floor = total_elevator_time_to_top) :
  stop_time_per_floor = 3 := 
  sorry

end elevator_stop_time_l1390_139004


namespace triangle_perimeter_l1390_139019

theorem triangle_perimeter (r AP PB x : ℕ) (h_r : r = 14) (h_AP : AP = 20) (h_PB : PB = 30) (h_BC_gt_AC : ∃ BC AC : ℝ, BC > AC)
: ∃ s : ℕ, s = (25 + x) → 2 * s = 50 + 2 * x :=
by
  sorry

end triangle_perimeter_l1390_139019


namespace work_completed_together_l1390_139052

theorem work_completed_together (A_days B_days : ℕ) (hA : A_days = 40) (hB : B_days = 60) : 
  1 / (1 / (A_days: ℝ) + 1 / (B_days: ℝ)) = 24 :=
by
  sorry

end work_completed_together_l1390_139052


namespace find_first_number_l1390_139016

theorem find_first_number
  (x y : ℝ)
  (h1 : y = 3.0)
  (h2 : x * y + 4 = 19) : x = 5 := by
  sorry

end find_first_number_l1390_139016


namespace find_current_l1390_139000

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current_l1390_139000


namespace initial_amount_l1390_139081

variable (X : ℝ)

/--
An individual deposited 20% of 25% of 30% of their initial amount into their bank account.
If the deposited amount is Rs. 750, prove that their initial amount was Rs. 50000.
-/
theorem initial_amount (h : (0.2 * 0.25 * 0.3 * X) = 750) : X = 50000 :=
by
  sorry

end initial_amount_l1390_139081


namespace bethany_age_l1390_139013

theorem bethany_age : ∀ (B S R : ℕ),
  (B - 3 = 2 * (S - 3)) →
  (B - 3 = R - 3 + 4) →
  (S + 5 = 16) →
  (R + 5 = 21) →
  B = 19 :=
by
  intros B S R h1 h2 h3 h4
  sorry

end bethany_age_l1390_139013


namespace estimate_probability_concave_l1390_139066

noncomputable def times_thrown : ℕ := 1000
noncomputable def frequency_convex : ℝ := 0.44

theorem estimate_probability_concave :
  (1 - frequency_convex) = 0.56 := by
  sorry

end estimate_probability_concave_l1390_139066


namespace fewer_females_than_males_l1390_139062

theorem fewer_females_than_males 
  (total_students : ℕ)
  (female_students : ℕ)
  (h_total : total_students = 280)
  (h_female : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end fewer_females_than_males_l1390_139062


namespace length_imaginary_axis_hyperbola_l1390_139087

theorem length_imaginary_axis_hyperbola : 
  ∀ (a b : ℝ), (a = 2) → (b = 1) → 
  (∀ x y : ℝ, (y^2 / a^2 - x^2 = 1) → 2 * b = 2) :=
by intros a b ha hb x y h; sorry

end length_imaginary_axis_hyperbola_l1390_139087


namespace total_days_stayed_l1390_139091

-- Definitions of given conditions as variables
def cost_first_week := 18
def days_first_week := 7
def cost_additional_week := 13
def total_cost := 334

-- Formulation of the target statement in Lean
theorem total_days_stayed :
  (days_first_week + 
  ((total_cost - (days_first_week * cost_first_week)) / cost_additional_week)) = 23 :=
by
  sorry

end total_days_stayed_l1390_139091


namespace quadratic_roots_condition_l1390_139060

theorem quadratic_roots_condition (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 0) :
  ¬ ((∃ x y : ℝ, ax^2 + 2*x + 1 = 0 ∧ ax^2 + 2*y + 1 = 0 ∧ x*y < 0) ↔
     (a > 0 ∧ a ≠ 0)) :=
by
  sorry

end quadratic_roots_condition_l1390_139060


namespace paint_required_for_small_statues_l1390_139001

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_required_for_small_statues_l1390_139001


namespace sum_of_last_two_digits_l1390_139024

theorem sum_of_last_two_digits (a b : ℕ) (ha: a = 6) (hb: b = 10) :
  ((a^15 + b^15) % 100) = 0 :=
by
  -- ha, hb represent conditions given.
  sorry

end sum_of_last_two_digits_l1390_139024


namespace usual_time_to_cover_distance_l1390_139094

variable (S T : ℝ)

-- Conditions:
-- 1. The man walks at 40% of his usual speed.
-- 2. He takes 24 minutes more to cover the same distance at this reduced speed.
-- 3. Usual speed is S.
-- 4. Usual time to cover the distance is T.

def usual_speed := S
def usual_time := T
def reduced_speed := 0.4 * S
def extra_time := 24

-- Question: Prove the man's usual time to cover the distance is 16 minutes.
theorem usual_time_to_cover_distance : T = 16 := 
by
  have speed_relation : S / (0.4 * S) = (T + 24) / T :=
    sorry
  have simplified_speed_relation : 2.5 = (T + 24) / T :=
    sorry
  have cross_multiplication_step : 2.5 * T = T + 24 :=
    sorry
  have solve_for_T_step : 1.5 * T = 24 :=
    sorry
  have final_step : T = 16 :=
    sorry
  exact final_step

end usual_time_to_cover_distance_l1390_139094


namespace max_trading_cards_l1390_139058

variable (money : ℝ) (cost_per_card : ℝ) (max_cards : ℕ)

theorem max_trading_cards (h_money : money = 9) (h_cost : cost_per_card = 1) : max_cards ≤ 9 :=
sorry

end max_trading_cards_l1390_139058


namespace paint_time_l1390_139006

theorem paint_time (n1 t1 n2 : ℕ) (k : ℕ) (h : n1 * t1 = k) (h1 : 5 * 4 = k) (h2 : n2 = 6) : (k / n2) = 10 / 3 :=
by {
  -- Proof would go here
  sorry
}

end paint_time_l1390_139006


namespace find_rectangle_length_l1390_139023

theorem find_rectangle_length (L W : ℕ) (h_area : L * W = 300) (h_perimeter : 2 * L + 2 * W = 70) : L = 20 :=
by
  sorry

end find_rectangle_length_l1390_139023


namespace analyze_quadratic_function_l1390_139026

variable (x : ℝ)

def quadratic_function : ℝ → ℝ := λ x => x^2 - 4 * x + 6

theorem analyze_quadratic_function :
  (∃ y : ℝ, quadratic_function y = (x - 2)^2 + 2) ∧
  (∃ x0 : ℝ, quadratic_function x0 = (x0 - 2)^2 + 2 ∧ x0 = 2 ∧ (∀ y : ℝ, quadratic_function y ≥ 2)) :=
by
  sorry

end analyze_quadratic_function_l1390_139026


namespace minimum_value_of_y_l1390_139050

theorem minimum_value_of_y : ∀ x : ℝ, ∃ y : ℝ, (y = 3 * x^2 + 6 * x + 9) → y ≥ 6 :=
by
  intro x
  use (3 * (x + 1)^2 + 6)
  intro h
  sorry

end minimum_value_of_y_l1390_139050


namespace problem_travel_time_with_current_l1390_139088

theorem problem_travel_time_with_current
  (D r c : ℝ) (t : ℝ)
  (h1 : (r - c) ≠ 0)
  (h2 : D / (r - c) = 60 / 7)
  (h3 : D / r = t - 7)
  (h4 : D / (r + c) = t)
  : t = 3 + 9 / 17 := 
sorry

end problem_travel_time_with_current_l1390_139088


namespace temperature_decrease_time_l1390_139041

theorem temperature_decrease_time
  (T_initial T_final T_per_hour : ℤ)
  (h_initial : T_initial = -5)
  (h_final : T_final = -25)
  (h_decrease : T_per_hour = -5) :
  (T_final - T_initial) / T_per_hour = 4 := by
sorry

end temperature_decrease_time_l1390_139041


namespace f_properties_l1390_139064

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l1390_139064


namespace option_C_is_proposition_l1390_139083

def is_proposition (s : Prop) : Prop := ∃ p : Prop, s = p

theorem option_C_is_proposition : is_proposition (4 + 3 = 8) := sorry

end option_C_is_proposition_l1390_139083


namespace cute_angle_of_isosceles_cute_triangle_l1390_139069

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle_l1390_139069


namespace triangle_obtuse_l1390_139031

-- We need to set up the definitions for angles and their relationships in triangles.

variable {A B C : ℝ} -- representing the angles of the triangle in radians

structure Triangle (A B C : ℝ) : Prop where
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  sum_to_pi : A + B + C = Real.pi -- representing the sum of angles in a triangle

-- Definition to state the condition in the problem
def triangle_condition (A B C : ℝ) : Prop :=
  Triangle A B C ∧ (Real.cos A * Real.cos B - Real.sin A * Real.sin B > 0)

-- Theorem to prove the triangle is obtuse under the given condition
theorem triangle_obtuse {A B C : ℝ} (h : triangle_condition A B C) : ∃ C', C' = C ∧ C' > Real.pi / 2 :=
sorry

end triangle_obtuse_l1390_139031


namespace value_of_a_l1390_139012

theorem value_of_a (a : ℝ) : (∀ x : ℝ, |x - a| < 1 ↔ 2 < x ∧ x < 4) → a = 3 :=
by
  intro h
  have h1 : 2 = a - 1 := sorry
  have h2 : 4 = a + 1 := sorry
  have h3 : a = 3 := sorry
  exact h3

end value_of_a_l1390_139012


namespace squares_count_correct_l1390_139063

-- Assuming basic setup and coordinate system.
def is_valid_point (x y : ℕ) : Prop :=
  x ≤ 8 ∧ y ≤ 8

-- Checking if a point (a, b) in the triangle as described.
def is_in_triangle (a b : ℕ) : Prop :=
  0 ≤ b ∧ b ≤ a ∧ a ≤ 4

-- Function derived from the solution detailing the number of such squares.
def count_squares (a b : ℕ) : ℕ :=
  -- Placeholder to represent the derived formula - to be replaced with actual derivation function
  (9 - a + b) * (a + b + 1) - 1

-- Statement to prove
theorem squares_count_correct (a b : ℕ) (h : is_in_triangle a b) :
  ∃ n, n = count_squares a b := 
sorry

end squares_count_correct_l1390_139063


namespace increasing_interval_of_f_l1390_139027

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2)

theorem increasing_interval_of_f :
  f x = (1/2)^(x^2 - 2) →
  ∀ x, f (x) ≤ f (x + 0.0001) :=
by
  sorry

end increasing_interval_of_f_l1390_139027
