import Mathlib

namespace NUMINAMATH_GPT_find_side_c_l440_44085

noncomputable def triangle_side_c (A b S : ℝ) (c : ℝ) : Prop :=
  S = 0.5 * b * c * Real.sin A

theorem find_side_c :
  ∀ (c : ℝ), triangle_side_c (Real.pi / 3) 16 (64 * Real.sqrt 3) c → c = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_side_c_l440_44085


namespace NUMINAMATH_GPT_round_balloons_burst_l440_44060

theorem round_balloons_burst :
  let round_balloons := 5 * 20
  let long_balloons := 4 * 30
  let total_balloons := round_balloons + long_balloons
  let balloons_left := 215
  ((total_balloons - balloons_left) = 5) :=
by 
  sorry

end NUMINAMATH_GPT_round_balloons_burst_l440_44060


namespace NUMINAMATH_GPT_number_of_pairs_satisfying_equation_l440_44055

theorem number_of_pairs_satisfying_equation :
  ∃ n : ℕ, n = 4998 ∧ (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → (x, y) ≠ (0, 0)) ∧
  (∀ x y : ℤ, x^2 + 7 * x * y + 6 * y^2 = 15^50 → ((x + 6 * y) = (3 * 5) ^ a ∧ (x + y) = (3 ^ (50 - a) * 5 ^ (50 - b)) ∨
        (x + 6 * y) = -(3 * 5) ^ a ∧ (x + y) = -(3 ^ (50 - a) * 5 ^ (50 - b)) → (a + b = 50))) :=
sorry

end NUMINAMATH_GPT_number_of_pairs_satisfying_equation_l440_44055


namespace NUMINAMATH_GPT_largest_number_value_l440_44035

theorem largest_number_value (x : ℕ) (h : 7 * x - 3 * x = 40) : 7 * x = 70 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_value_l440_44035


namespace NUMINAMATH_GPT_positive_distinct_solutions_of_system_l440_44099

variables {a b x y z : ℝ}

theorem positive_distinct_solutions_of_system
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : xy = z^2) :
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by
  sorry

end NUMINAMATH_GPT_positive_distinct_solutions_of_system_l440_44099


namespace NUMINAMATH_GPT_average_tree_height_is_800_l440_44011

def first_tree_height : ℕ := 1000
def other_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200
def total_height : ℕ := first_tree_height + other_tree_height + other_tree_height + last_tree_height
def average_height : ℕ := total_height / 4

theorem average_tree_height_is_800 :
  average_height = 800 := by
  sorry

end NUMINAMATH_GPT_average_tree_height_is_800_l440_44011


namespace NUMINAMATH_GPT_milan_billed_minutes_l440_44078

-- Variables corresponding to the conditions
variables (f r b : ℝ) (m : ℕ)

-- The conditions of the problem
def conditions : Prop :=
  f = 2 ∧ r = 0.12 ∧ b = 23.36 ∧ b = f + r * m

-- The theorem based on given conditions and aiming to prove that m = 178
theorem milan_billed_minutes (h : conditions f r b m) : m = 178 :=
sorry

end NUMINAMATH_GPT_milan_billed_minutes_l440_44078


namespace NUMINAMATH_GPT_abs_inequality_solution_l440_44006

theorem abs_inequality_solution (x : ℝ) (h : |x - 4| ≤ 6) : -2 ≤ x ∧ x ≤ 10 := 
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l440_44006


namespace NUMINAMATH_GPT_value_of_quotient_l440_44008

variable (a b c d : ℕ)

theorem value_of_quotient 
  (h1 : a = 3 * b)
  (h2 : b = 2 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_quotient_l440_44008


namespace NUMINAMATH_GPT_abs_sum_eq_3_given_condition_l440_44054

theorem abs_sum_eq_3_given_condition (m n p : ℤ)
  (h : |m - n|^3 + |p - m|^5 = 1) :
  |p - m| + |m - n| + 2 * |n - p| = 3 :=
sorry

end NUMINAMATH_GPT_abs_sum_eq_3_given_condition_l440_44054


namespace NUMINAMATH_GPT_solve_system_l440_44068

theorem solve_system (x1 x2 x3 : ℝ) :
  (x1 - 2 * x2 + 3 * x3 = 5) ∧ 
  (2 * x1 + 3 * x2 - x3 = 7) ∧ 
  (3 * x1 + x2 + 2 * x3 = 12) 
  ↔ (x1, x2, x3) = (7 - 5 * x3, 1 - x3, x3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l440_44068


namespace NUMINAMATH_GPT_average_t_value_is_15_l440_44061

noncomputable def average_of_distinct_t_values (t_vals : List ℤ) : ℤ :=
t_vals.sum / t_vals.length

theorem average_t_value_is_15 :
  average_of_distinct_t_values [8, 14, 18, 20] = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_t_value_is_15_l440_44061


namespace NUMINAMATH_GPT_Ram_Shyam_weight_ratio_l440_44044

theorem Ram_Shyam_weight_ratio :
  ∃ (R S : ℝ), 
    (1.10 * R + 1.21 * S = 82.8) ∧ 
    (1.15 * (R + S) = 82.8) ∧ 
    (R / S = 1.20) :=
by {
  sorry
}

end NUMINAMATH_GPT_Ram_Shyam_weight_ratio_l440_44044


namespace NUMINAMATH_GPT_solution_set_of_inequality_minimum_value_2a_plus_b_l440_44024

noncomputable def f (x : ℝ) : ℝ := x + 1 + |3 - x|

theorem solution_set_of_inequality :
  {x : ℝ | x ≥ -1 ∧ f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem minimum_value_2a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 8 * a * b = a + 2 * b) :
  2 * a + b = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_minimum_value_2a_plus_b_l440_44024


namespace NUMINAMATH_GPT_original_amount_l440_44030

theorem original_amount (x : ℝ) (h : 0.25 * x = 200) : x = 800 := 
by
  sorry

end NUMINAMATH_GPT_original_amount_l440_44030


namespace NUMINAMATH_GPT_min_max_values_in_interval_l440_44053

def func (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

theorem min_max_values_in_interval :
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≥ -1/3) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = -1/3) ∧
  (∀ x y, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → func x y ≤ 9/8) ∧
  (∃ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ func x y = 9/8) :=
by
  sorry

end NUMINAMATH_GPT_min_max_values_in_interval_l440_44053


namespace NUMINAMATH_GPT_find_m_for_split_l440_44063

theorem find_m_for_split (m : ℕ) (h1 : m > 1) (h2 : ∃ k, k < m ∧ 2023 = (m^2 - m + 1) + 2*k) : m = 45 :=
sorry

end NUMINAMATH_GPT_find_m_for_split_l440_44063


namespace NUMINAMATH_GPT_compute_f_g_f_l440_44096

def f (x : ℤ) : ℤ := 2 * x + 4
def g (x : ℤ) : ℤ := 5 * x + 2

theorem compute_f_g_f (x : ℤ) : f (g (f 3)) = 108 := 
  by 
  sorry

end NUMINAMATH_GPT_compute_f_g_f_l440_44096


namespace NUMINAMATH_GPT_sum_of_numbers_l440_44026

theorem sum_of_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000)
(h_eq : 100 * x + y = 7 * x * y) : x + y = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l440_44026


namespace NUMINAMATH_GPT_factor_expression_l440_44003

theorem factor_expression (x : ℝ) : 72 * x ^ 5 - 162 * x ^ 9 = -18 * x ^ 5 * (9 * x ^ 4 - 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l440_44003


namespace NUMINAMATH_GPT_cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l440_44043

-- Define the production cost function
def production_cost (n : ℕ) : ℕ := 4000 + 50 * n

-- Define the profit function
def profit (n : ℕ) : ℤ := 90 * n - 4000 - 50 * n

-- 1. Prove that the cost for producing 1000 pairs of shoes is 54,000 yuan
theorem cost_of_1000_pairs : production_cost 1000 = 54000 := 
by sorry

-- 2. Prove that if the production cost is 48,000 yuan, then 880 pairs of shoes were produced
theorem pairs_for_48000_yuan (n : ℕ) (h : production_cost n = 48000) : n = 880 := 
by sorry

-- 3. Prove that at least 100 pairs of shoes must be produced each day to avoid a loss
theorem minimum_pairs_to_avoid_loss (n : ℕ) : profit n ≥ 0 ↔ n ≥ 100 := 
by sorry

end NUMINAMATH_GPT_cost_of_1000_pairs_pairs_for_48000_yuan_minimum_pairs_to_avoid_loss_l440_44043


namespace NUMINAMATH_GPT_range_of_x_l440_44081

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem range_of_x (x : ℝ) (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) 2) :
  f (m * x - 3) + f x < 0 → -3 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_range_of_x_l440_44081


namespace NUMINAMATH_GPT_rate_per_square_meter_l440_44036

-- Define the conditions
def length (L : ℝ) := L = 8
def width (W : ℝ) := W = 4.75
def total_cost (C : ℝ) := C = 34200
def area (A : ℝ) (L W : ℝ) := A = L * W
def rate (R C A : ℝ) := R = C / A

-- The theorem to prove
theorem rate_per_square_meter (L W C A R : ℝ) 
  (hL : length L) (hW : width W) (hC : total_cost C) (hA : area A L W) : 
  rate R C A :=
by
  -- By the conditions, length is 8, width is 4.75, and total cost is 34200.
  simp [length, width, total_cost, area, rate] at hL hW hC hA ⊢
  -- It remains to calculate the rate and use conditions
  have hA : A = L * W := hA
  rw [hL, hW] at hA
  have hA' : A = 8 * 4.75 := by simp [hA]
  rw [hA']
  simp [rate]
  sorry -- The detailed proof is omitted.

end NUMINAMATH_GPT_rate_per_square_meter_l440_44036


namespace NUMINAMATH_GPT_determine_x_l440_44066

theorem determine_x (x : ℝ) :
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 8 * x + 15) / (x^2 - 10 * x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_x_l440_44066


namespace NUMINAMATH_GPT_proposition_D_l440_44017

-- Definitions extracted from the conditions
variables {a b : ℝ} (c d : ℝ)

-- Proposition D to be proven
theorem proposition_D (ha : a < b) (hb : b < 0) : a^2 > b^2 := sorry

end NUMINAMATH_GPT_proposition_D_l440_44017


namespace NUMINAMATH_GPT_files_remaining_l440_44086

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h_music : music_files = 13) 
  (h_video : video_files = 30) 
  (h_deleted : deleted_files = 10) : 
  (music_files + video_files - deleted_files) = 33 :=
by
  sorry

end NUMINAMATH_GPT_files_remaining_l440_44086


namespace NUMINAMATH_GPT_jack_keeps_deers_weight_is_correct_l440_44004

-- Define conditions
def monthly_hunt_count : Float := 7.5
def fraction_of_year_hunting_season : Float := 1 / 3
def deers_per_hunt : Float := 2.5
def weight_per_deer : Float := 600
def weight_kept_per_deer : Float := 0.65

-- Prove the total weight of the deer Jack keeps
theorem jack_keeps_deers_weight_is_correct :
  (12 * fraction_of_year_hunting_season) * monthly_hunt_count * deers_per_hunt * weight_per_deer * weight_kept_per_deer = 29250 :=
by
  sorry

end NUMINAMATH_GPT_jack_keeps_deers_weight_is_correct_l440_44004


namespace NUMINAMATH_GPT_monthly_rent_requirement_l440_44093

noncomputable def initial_investment : Float := 200000
noncomputable def annual_return_rate : Float := 0.06
noncomputable def annual_insurance_cost : Float := 4500
noncomputable def maintenance_percentage : Float := 0.15
noncomputable def required_monthly_rent : Float := 1617.65

theorem monthly_rent_requirement :
  let annual_return := initial_investment * annual_return_rate
  let annual_cost_with_insurance := annual_return + annual_insurance_cost
  let monthly_required_net := annual_cost_with_insurance / 12
  let rental_percentage_kept := 1 - maintenance_percentage
  let monthly_rental_full := monthly_required_net / rental_percentage_kept
  monthly_rental_full = required_monthly_rent := 
by
  sorry

end NUMINAMATH_GPT_monthly_rent_requirement_l440_44093


namespace NUMINAMATH_GPT_total_cookies_l440_44088

   -- Define the conditions
   def cookies_per_bag : ℕ := 41
   def number_of_bags : ℕ := 53

   -- Define the problem: Prove that the total number of cookies is 2173
   theorem total_cookies : cookies_per_bag * number_of_bags = 2173 :=
   by sorry
   
end NUMINAMATH_GPT_total_cookies_l440_44088


namespace NUMINAMATH_GPT_numbers_composite_l440_44014

theorem numbers_composite (a b c d : ℕ) (h : a * b = c * d) : ∃ x y : ℕ, (x > 1 ∧ y > 1) ∧ a^2000 + b^2000 + c^2000 + d^2000 = x * y := 
sorry

end NUMINAMATH_GPT_numbers_composite_l440_44014


namespace NUMINAMATH_GPT_cannot_buy_same_number_of_notebooks_l440_44052

theorem cannot_buy_same_number_of_notebooks
  (price_softcover : ℝ)
  (price_hardcover : ℝ)
  (notebooks_ming : ℝ)
  (notebooks_li : ℝ)
  (h1 : price_softcover = 12)
  (h2 : price_hardcover = 21)
  (h3 : price_hardcover = price_softcover + 1.2) :
  notebooks_ming = 12 / price_softcover ∧
  notebooks_li = 21 / price_hardcover →
  ¬ (notebooks_ming = notebooks_li) :=
by
  sorry

end NUMINAMATH_GPT_cannot_buy_same_number_of_notebooks_l440_44052


namespace NUMINAMATH_GPT_expression_value_l440_44037

theorem expression_value (y : ℤ) (h : y = 5) : (y^2 - y - 12) / (y - 4) = 8 :=
by
  rw[h]
  sorry

end NUMINAMATH_GPT_expression_value_l440_44037


namespace NUMINAMATH_GPT_stuffed_animal_cost_l440_44049

variable (S : ℝ)  -- Cost of the stuffed animal
variable (total_cost_after_discount_gave_30_dollars : S * 0.10 = 3.6) 
-- Condition: cost of stuffed animal = $4.44
theorem stuffed_animal_cost :
  S = 4.44 :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animal_cost_l440_44049


namespace NUMINAMATH_GPT_total_tickets_sold_is_336_l440_44094

-- Define the costs of the tickets
def cost_vip_ticket : ℕ := 45
def cost_ga_ticket : ℕ := 20

-- Define the total cost collected
def total_cost_collected : ℕ := 7500

-- Define the difference in the number of tickets sold
def vip_less_ga : ℕ := 276

-- Define the main theorem to be proved
theorem total_tickets_sold_is_336 (V G : ℕ) 
  (h1 : cost_vip_ticket * V + cost_ga_ticket * G = total_cost_collected)
  (h2 : V = G - vip_less_ga) : V + G = 336 :=
  sorry

end NUMINAMATH_GPT_total_tickets_sold_is_336_l440_44094


namespace NUMINAMATH_GPT_T_n_bounds_l440_44095

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

noncomputable def b_n (n : ℕ) : ℚ := 
if n ≤ 4 then 2 * n + 1
else 1 / (n * (n + 2))

noncomputable def T_n (n : ℕ) : ℚ := 
if n ≤ 4 then S_n n
else (24 : ℚ) + (1 / 2) * (1 / 5 + 1 / 6 - 1 / (n + 1 : ℚ) - 1 / (n + 2 : ℚ))

theorem T_n_bounds (n : ℕ) : 3 ≤ T_n n ∧ T_n n < 24 + 11 / 60 := by
  sorry

end NUMINAMATH_GPT_T_n_bounds_l440_44095


namespace NUMINAMATH_GPT_parallel_lines_condition_l440_44062

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y - 4 = 0 → x + (a + 1) * y + 2 = 0) ↔ a = 1 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_condition_l440_44062


namespace NUMINAMATH_GPT_xyz_value_l440_44019

-- Define the real numbers x, y, and z
variables (x y z : ℝ)

-- Condition 1
def condition1 := (x + y + z) * (x * y + x * z + y * z) = 49

-- Condition 2
def condition2 := x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19

-- Main theorem statement
theorem xyz_value (h1 : condition1 x y z) (h2 : condition2 x y z) : x * y * z = 10 :=
sorry

end NUMINAMATH_GPT_xyz_value_l440_44019


namespace NUMINAMATH_GPT_ny_mets_fans_count_l440_44084

variable (Y M R : ℕ) -- Variables representing number of fans
variable (k j : ℕ)   -- Helper variables for ratios

theorem ny_mets_fans_count :
  (Y = 3 * k) →
  (M = 2 * k) →
  (M = 4 * j) →
  (R = 5 * j) →
  (Y + M + R = 330) →
  (∃ (k j : ℕ), k = 2 * j) →
  M = 88 := sorry

end NUMINAMATH_GPT_ny_mets_fans_count_l440_44084


namespace NUMINAMATH_GPT_average_episodes_per_year_is_16_l440_44070

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end NUMINAMATH_GPT_average_episodes_per_year_is_16_l440_44070


namespace NUMINAMATH_GPT_wendy_total_glasses_l440_44092

noncomputable def small_glasses : ℕ := 50
noncomputable def large_glasses : ℕ := small_glasses + 10
noncomputable def total_glasses : ℕ := small_glasses + large_glasses

theorem wendy_total_glasses : total_glasses = 110 :=
by
  sorry

end NUMINAMATH_GPT_wendy_total_glasses_l440_44092


namespace NUMINAMATH_GPT_P_at_6_l440_44016

noncomputable def P (x : ℕ) : ℚ := (720 * x) / (x^2 - 1)

theorem P_at_6 : P 6 = 48 :=
by
  -- Definitions and conditions derived from the problem.
  -- Establishing given condition and deriving P(6) value.
  sorry

end NUMINAMATH_GPT_P_at_6_l440_44016


namespace NUMINAMATH_GPT_old_record_was_300_points_l440_44064

theorem old_record_was_300_points :
  let touchdowns_per_game := 4
  let points_per_touchdown := 6
  let games_in_season := 15
  let conversions := 6
  let points_per_conversion := 2
  let points_beat := 72
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + conversions * points_per_conversion
  total_points - points_beat = 300 := 
by
  sorry

end NUMINAMATH_GPT_old_record_was_300_points_l440_44064


namespace NUMINAMATH_GPT_trains_meet_480_km_away_l440_44025

-- Define the conditions
def bombay_express_speed : ℕ := 60 -- speed in km/h
def rajdhani_express_speed : ℕ := 80 -- speed in km/h
def bombay_express_start_time : ℕ := 1430 -- 14:30 in 24-hour format
def rajdhani_express_start_time : ℕ := 1630 -- 16:30 in 24-hour format

-- Define the function to calculate the meeting point distance
noncomputable def meeting_distance (bombay_speed rajdhani_speed : ℕ) (bombay_start rajdhani_start : ℕ) : ℕ :=
  let t := 6 -- time taken for Rajdhani to catch up in hours, derived from the solution
  rajdhani_speed * t

-- The statement we need to prove:
theorem trains_meet_480_km_away :
  meeting_distance bombay_express_speed rajdhani_express_speed bombay_express_start_time rajdhani_express_start_time = 480 := by
  sorry

end NUMINAMATH_GPT_trains_meet_480_km_away_l440_44025


namespace NUMINAMATH_GPT_total_travel_distance_l440_44034

noncomputable def total_distance_traveled (DE DF : ℝ) : ℝ :=
  let EF := Real.sqrt (DE^2 - DF^2)
  DE + EF + DF

theorem total_travel_distance
  (DE DF : ℝ)
  (hDE : DE = 4500)
  (hDF : DF = 4000)
  : total_distance_traveled DE DF = 10560.992 :=
by
  rw [hDE, hDF]
  unfold total_distance_traveled
  norm_num
  sorry

end NUMINAMATH_GPT_total_travel_distance_l440_44034


namespace NUMINAMATH_GPT_round_time_of_A_l440_44050

theorem round_time_of_A (T_a T_b : ℝ) 
  (h1 : 4 * T_b = 5 * T_a) 
  (h2 : 4 * T_b = 4 * T_a + 10) : T_a = 10 :=
by
  sorry

end NUMINAMATH_GPT_round_time_of_A_l440_44050


namespace NUMINAMATH_GPT_proof_expr_l440_44027

theorem proof_expr (a b c : ℤ) (h1 : a - b = 3) (h2 : b - c = 2) : (a - c)^2 + 3 * a + 1 - 3 * c = 41 := by {
  sorry
}

end NUMINAMATH_GPT_proof_expr_l440_44027


namespace NUMINAMATH_GPT_spinsters_count_l440_44075

variable (S C : ℕ)

-- defining the conditions
def ratio_condition (S C : ℕ) : Prop := 9 * S = 2 * C
def difference_condition (S C : ℕ) : Prop := C = S + 63

-- theorem to prove
theorem spinsters_count 
  (h1 : ratio_condition S C) 
  (h2 : difference_condition S C) : 
  S = 18 :=
sorry

end NUMINAMATH_GPT_spinsters_count_l440_44075


namespace NUMINAMATH_GPT_units_digit_fib_cycle_length_60_l440_44097

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Define the function to get the units digit (mod 10)
def units_digit_fib (n : ℕ) : ℕ :=
  (fib n) % 10

-- State the theorem about the cycle length of the units digits in Fibonacci sequence
theorem units_digit_fib_cycle_length_60 :
  ∃ k, k = 60 ∧ ∀ n, units_digit_fib (n + k) = units_digit_fib n := sorry

end NUMINAMATH_GPT_units_digit_fib_cycle_length_60_l440_44097


namespace NUMINAMATH_GPT_find_x_l440_44091

-- We define the given condition in Lean
theorem find_x (x : ℝ) (h : 6 * x - 12 = -(4 + 2 * x)) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l440_44091


namespace NUMINAMATH_GPT_cone_height_circular_sector_l440_44083

theorem cone_height_circular_sector (r : ℝ) (n : ℕ) (h : ℝ)
  (hr : r = 10)
  (hn : n = 3)
  (hradius : r > 0)
  (hcircumference : 2 * Real.pi * r / n = 2 * Real.pi * r / 3)
  : h = (20 * Real.sqrt 2) / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_cone_height_circular_sector_l440_44083


namespace NUMINAMATH_GPT_sequence_number_pair_l440_44089

theorem sequence_number_pair (n m : ℕ) (h : m ≤ n) : (m, n - m + 1) = (m, n - m + 1) :=
by sorry

end NUMINAMATH_GPT_sequence_number_pair_l440_44089


namespace NUMINAMATH_GPT_material_needed_l440_44071

-- Define the required conditions
def feet_per_tee_shirt : ℕ := 4
def number_of_tee_shirts : ℕ := 15

-- State the theorem and the proof obligation
theorem material_needed : feet_per_tee_shirt * number_of_tee_shirts = 60 := 
by 
  sorry

end NUMINAMATH_GPT_material_needed_l440_44071


namespace NUMINAMATH_GPT_total_money_before_spending_l440_44029

-- Define the amounts for each friend
variables (J P Q A: ℝ)

-- Define the conditions from the problem
def condition1 := P = 2 * J
def condition2 := Q = P + 20
def condition3 := A = 1.15 * Q
def condition4 := J + P + Q + A = 1211
def cost_of_item : ℝ := 1200

-- The total amount before buying the item
theorem total_money_before_spending (J P Q A : ℝ)
  (h1 : condition1 J P)
  (h2 : condition2 P Q)
  (h3 : condition3 Q A)
  (h4 : condition4 J P Q A) : 
  J + P + Q + A - cost_of_item = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_money_before_spending_l440_44029


namespace NUMINAMATH_GPT_max_value_of_quadratic_l440_44007

theorem max_value_of_quadratic : ∃ x : ℝ, (∀ y : ℝ, (-3 * y^2 + 9 * y - 1) ≤ (-3 * (3/2)^2 + 9 * (3/2) - 1)) ∧ x = 3/2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l440_44007


namespace NUMINAMATH_GPT_initial_interest_rate_l440_44073

theorem initial_interest_rate 
  (r P : ℝ)
  (h1 : 20250 = P * r)
  (h2 : 22500 = P * (r + 5)) :
  r = 45 :=
by
  sorry

end NUMINAMATH_GPT_initial_interest_rate_l440_44073


namespace NUMINAMATH_GPT_minimum_area_convex_quadrilateral_l440_44069

theorem minimum_area_convex_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  (∀ S_BOC S_AOD : ℝ, S_AOB * S_COD = S_BOC * S_AOD → 
    (S_AOB + S_BOC + S_COD + S_AOD) ≥ 25) := sorry

end NUMINAMATH_GPT_minimum_area_convex_quadrilateral_l440_44069


namespace NUMINAMATH_GPT_geometric_series_sum_l440_44046

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1/3) :
  (∑' n : ℕ, a * r^n) = 3/2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l440_44046


namespace NUMINAMATH_GPT_line_through_point_with_slope_l440_44038

theorem line_through_point_with_slope (x y : ℝ) (h : y - 2 = -3 * (x - 1)) : 3 * x + y - 5 = 0 :=
sorry

example : 3 * 1 + 2 - 5 = 0 := by sorry

end NUMINAMATH_GPT_line_through_point_with_slope_l440_44038


namespace NUMINAMATH_GPT_num_values_of_n_l440_44072

theorem num_values_of_n (a b c : ℕ) (h : 7 * a + 77 * b + 7777 * c = 8000) : 
  ∃ n : ℕ, (n = a + 2 * b + 4 * c) ∧ (110 * n ≤ 114300) ∧ ((8000 - 7 * a) % 70 = 7 * (10 * b + 111 * c) % 70) := 
sorry

end NUMINAMATH_GPT_num_values_of_n_l440_44072


namespace NUMINAMATH_GPT_function_is_quadratic_l440_44031

-- Definitions for the conditions
def is_quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0) ∧ ∀ (x : ℝ), f x = a * x^2 + b * x + c

-- The function to be proved as a quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- The theorem statement: f must be a quadratic function
theorem function_is_quadratic : is_quadratic_function f :=
  sorry

end NUMINAMATH_GPT_function_is_quadratic_l440_44031


namespace NUMINAMATH_GPT_johns_yearly_grass_cutting_cost_l440_44021

-- Definitions of the conditions
def initial_height : ℝ := 2.0
def growth_rate : ℝ := 0.5
def cutting_height : ℝ := 4.0
def cost_per_cut : ℝ := 100.0
def months_per_year : ℝ := 12.0

-- Formulate the statement
theorem johns_yearly_grass_cutting_cost :
  let months_to_grow : ℝ := (cutting_height - initial_height) / growth_rate
  let cuts_per_year : ℝ := months_per_year / months_to_grow
  let total_cost_per_year : ℝ := cuts_per_year * cost_per_cut
  total_cost_per_year = 300.0 :=
by
  sorry

end NUMINAMATH_GPT_johns_yearly_grass_cutting_cost_l440_44021


namespace NUMINAMATH_GPT_find_the_number_l440_44087

-- Define the variables and conditions
variable (x z : ℝ)
variable (the_number : ℝ)

-- Condition: given that x = 1
axiom h1 : x = 1

-- Condition: given the equation
axiom h2 : 14 * (-x + z) + 18 = -14 * (x - z) - the_number

-- The theorem to prove
theorem find_the_number : the_number = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l440_44087


namespace NUMINAMATH_GPT_distance_home_to_school_l440_44032

def speed_walk := 5
def speed_car := 15
def time_difference := 2

variable (d : ℝ) -- distance from home to school
variable (T1 T2 : ℝ) -- T1: time to school, T2: time back home

-- Conditions
axiom h1 : T1 = d / speed_walk / 2 + d / speed_car / 2
axiom h2 : d = speed_car * T2 / 3 + speed_walk * 2 * T2 / 3
axiom h3 : T1 = T2 + time_difference

-- Theorem to prove
theorem distance_home_to_school : d = 150 :=
by
  sorry

end NUMINAMATH_GPT_distance_home_to_school_l440_44032


namespace NUMINAMATH_GPT_g_symmetric_l440_44002

noncomputable def g (x : ℝ) : ℝ := |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem g_symmetric : ∀ x : ℝ, g x = g (1 - x) := by
  sorry

end NUMINAMATH_GPT_g_symmetric_l440_44002


namespace NUMINAMATH_GPT_muffins_equation_l440_44000

def remaining_muffins : ℕ := 48
def total_muffins : ℕ := 83
def initially_baked_muffins : ℕ := 35

theorem muffins_equation : initially_baked_muffins + remaining_muffins = total_muffins :=
  by
    -- Skipping the proof here
    sorry

end NUMINAMATH_GPT_muffins_equation_l440_44000


namespace NUMINAMATH_GPT_initial_cheerleaders_count_l440_44079

theorem initial_cheerleaders_count (C : ℕ) 
  (initial_football_players : ℕ := 13) 
  (quit_football_players : ℕ := 10) 
  (quit_cheerleaders : ℕ := 4) 
  (remaining_people : ℕ := 15) 
  (initial_total : ℕ := initial_football_players + C) 
  (final_total : ℕ := (initial_football_players - quit_football_players) + (C - quit_cheerleaders)) :
  remaining_people = final_total → C = 16 :=
by intros h; sorry

end NUMINAMATH_GPT_initial_cheerleaders_count_l440_44079


namespace NUMINAMATH_GPT_grocer_sales_l440_44039

theorem grocer_sales 
  (s1 s2 s3 s4 s5 s6 s7 s8 sales : ℝ)
  (h_sales_1 : s1 = 5420)
  (h_sales_2 : s2 = 5660)
  (h_sales_3 : s3 = 6200)
  (h_sales_4 : s4 = 6350)
  (h_sales_5 : s5 = 6500)
  (h_sales_6 : s6 = 6780)
  (h_sales_7 : s7 = 7000)
  (h_sales_8 : s8 = 7200)
  (h_avg : (5420 + 5660 + 6200 + 6350 + 6500 + 6780 + 7000 + 7200 + 2 * sales) / 10 = 6600) :
  sales = 9445 := 
  by 
  sorry

end NUMINAMATH_GPT_grocer_sales_l440_44039


namespace NUMINAMATH_GPT_simplify_expression_l440_44067

variable (a b c d x y z : ℝ)

theorem simplify_expression :
  (cx * (b^2 * x^3 + 3 * a^2 * y^3 + c^2 * z^3) + dz * (a^2 * x^3 + 3 * c^2 * y^3 + b^2 * z^3)) / (cx + dz) =
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l440_44067


namespace NUMINAMATH_GPT_berry_circle_properties_l440_44074

theorem berry_circle_properties :
  ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 12 = 2 * x + 4 * y → r = Real.sqrt 17)
    ∧ (π * Real.sqrt 17 ^ 2 > 30) :=
by
  sorry

end NUMINAMATH_GPT_berry_circle_properties_l440_44074


namespace NUMINAMATH_GPT_intersection_complement_A_B_l440_44090

def Universe : Set ℝ := Set.univ

def A : Set ℝ := {x | abs (x - 1) > 2}

def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_complement_A_B :
  (Universe \ A) ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_l440_44090


namespace NUMINAMATH_GPT_binary_ternary_conversion_l440_44076

theorem binary_ternary_conversion (a b : ℕ) (h_b : b = 0 ∨ b = 1) (h_a : a = 0 ∨ a = 1 ∨ a = 2)
  (h_eq : 8 + 2 * b + 1 = 9 * a + 2) : 2 * a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_binary_ternary_conversion_l440_44076


namespace NUMINAMATH_GPT_radius_of_sphere_is_approximately_correct_l440_44001

noncomputable def radius_of_sphere_in_cylinder_cone : ℝ :=
  let radius_cylinder := 12
  let height_cylinder := 30
  let radius_sphere := 21 - 0.5 * Real.sqrt (30^2 + 12^2)
  radius_sphere

theorem radius_of_sphere_is_approximately_correct : abs (radius_of_sphere_in_cylinder_cone - 4.84) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_is_approximately_correct_l440_44001


namespace NUMINAMATH_GPT_gym_monthly_cost_l440_44047

theorem gym_monthly_cost (down_payment total_cost total_months : ℕ) (h_down_payment : down_payment = 50) (h_total_cost : total_cost = 482) (h_total_months : total_months = 36) : 
  (total_cost - down_payment) / total_months = 12 := by 
  sorry

end NUMINAMATH_GPT_gym_monthly_cost_l440_44047


namespace NUMINAMATH_GPT_part_1_part_3_500_units_part_3_1000_units_l440_44051

/-- Define the pricing function P as per the given conditions -/
def P (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x <= 550 then 62 - 0.02 * x
  else 51

/-- Verify that ordering 550 units results in a per-unit price of 51 yuan -/
theorem part_1 : P 550 = 51 := sorry

/-- Compute profit for given order quantities -/
def profit (x : ℕ) : ℝ :=
  x * (P x - 40)

/-- Verify that an order of 500 units results in a profit of 6000 yuan -/
theorem part_3_500_units : profit 500 = 6000 := sorry

/-- Verify that an order of 1000 units results in a profit of 11000 yuan -/
theorem part_3_1000_units : profit 1000 = 11000 := sorry

end NUMINAMATH_GPT_part_1_part_3_500_units_part_3_1000_units_l440_44051


namespace NUMINAMATH_GPT_triangle_vertex_y_coordinate_l440_44013

theorem triangle_vertex_y_coordinate (h : ℝ) :
  let A := (0, 0)
  let C := (8, 0)
  let B := (4, h)
  (1/2) * (8) * h = 32 → h = 8 :=
by
  intro h
  intro H
  sorry

end NUMINAMATH_GPT_triangle_vertex_y_coordinate_l440_44013


namespace NUMINAMATH_GPT_math_problem_l440_44018

noncomputable def canA_red_balls := 3
noncomputable def canA_black_balls := 4
noncomputable def canB_red_balls := 2
noncomputable def canB_black_balls := 3

noncomputable def prob_event_A := canA_red_balls / (canA_red_balls + canA_black_balls) -- P(A)
noncomputable def prob_event_B := 
  (canA_red_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls + 1) / (6) +
  (canA_black_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls) / (6) -- P(B)

theorem math_problem : 
  (prob_event_A = 3 / 7) ∧ 
  (prob_event_B = 17 / 42) ∧
  (¬ (prob_event_A * prob_event_B = (3 / 7) * (17 / 42))) ∧
  ((prob_event_A * (canB_red_balls + 1) / 6) / prob_event_A = 1 / 2) := by
  repeat { sorry }

end NUMINAMATH_GPT_math_problem_l440_44018


namespace NUMINAMATH_GPT_new_trailer_homes_added_l440_44020

theorem new_trailer_homes_added (n : ℕ) (h1 : (20 * 20 + 2 * n)/(20 + n) = 14) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_new_trailer_homes_added_l440_44020


namespace NUMINAMATH_GPT_find_S_9_l440_44028

-- Conditions
def aₙ (n : ℕ) : ℕ := sorry  -- arithmetic sequence

def Sₙ (n : ℕ) : ℕ := sorry  -- sum of the first n terms of the sequence

axiom condition_1 : 2 * aₙ 8 = 6 + aₙ 11

-- Proof goal
theorem find_S_9 : Sₙ 9 = 54 :=
sorry

end NUMINAMATH_GPT_find_S_9_l440_44028


namespace NUMINAMATH_GPT_medians_form_right_triangle_medians_inequality_l440_44012

variable {α : Type*}
variables {a b c : ℝ}
variables {m_a m_b m_c : ℝ}
variable (orthogonal_medians : m_a * m_b = 0)

-- Part (a)
theorem medians_form_right_triangle
  (orthogonal_medians : m_a * m_b = 0) :
  m_a^2 + m_b^2 = m_c^2 :=
sorry

-- Part (b)
theorem medians_inequality
  (orthogonal_medians : m_a * m_b = 0)
  (triangle_sides : a^2 + b^2 = 5 * c^2): 
  5 * (a^2 + b^2 - c^2) ≥ 8 * a * b :=
sorry

end NUMINAMATH_GPT_medians_form_right_triangle_medians_inequality_l440_44012


namespace NUMINAMATH_GPT_find_x_l440_44077

theorem find_x (x : ℝ) (h : 70 + 60 / (x / 3) = 71) : x = 180 :=
sorry

end NUMINAMATH_GPT_find_x_l440_44077


namespace NUMINAMATH_GPT_find_integer_pairs_l440_44033

theorem find_integer_pairs (x y : ℤ) (h : x^3 - y^3 = 2 * x * y + 8) : 
  (x = 0 ∧ y = -2) ∨ (x = 2 ∧ y = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_integer_pairs_l440_44033


namespace NUMINAMATH_GPT_min_sum_of_consecutive_natural_numbers_l440_44057

theorem min_sum_of_consecutive_natural_numbers (a b c : ℕ) 
  (h1 : a + 1 = b)
  (h2 : a + 2 = c)
  (h3 : a % 9 = 0)
  (h4 : b % 8 = 0)
  (h5 : c % 7 = 0) :
  a + b + c = 1488 :=
sorry

end NUMINAMATH_GPT_min_sum_of_consecutive_natural_numbers_l440_44057


namespace NUMINAMATH_GPT_rectangle_side_ratio_l440_44005

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_side_ratio_l440_44005


namespace NUMINAMATH_GPT_triangle_third_side_range_l440_44022

variable (a b c : ℝ)

theorem triangle_third_side_range 
  (h₁ : |a + b - 4| + (a - b + 2)^2 = 0)
  (h₂ : a + b > c)
  (h₃ : a + c > b)
  (h₄ : b + c > a) : 2 < c ∧ c < 4 := 
sorry

end NUMINAMATH_GPT_triangle_third_side_range_l440_44022


namespace NUMINAMATH_GPT_sum_of_three_consecutive_integers_is_21_l440_44080

theorem sum_of_three_consecutive_integers_is_21 : 
  ∃ (n : ℤ), 3 * n = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_integers_is_21_l440_44080


namespace NUMINAMATH_GPT_nickels_count_l440_44041

theorem nickels_count (N Q : ℕ) 
  (h_eq : N = Q) 
  (h_total_value : 5 * N + 25 * Q = 1200) :
  N = 40 := 
by 
  sorry

end NUMINAMATH_GPT_nickels_count_l440_44041


namespace NUMINAMATH_GPT_find_fourth_vertex_l440_44010

structure Point :=
  (x : ℝ)
  (y : ℝ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  is_midpoint ({x := 0, y := -9}) A C ∧ is_midpoint ({x := 2, y := 6}) B D ∧
  is_midpoint ({x := 4, y := 5}) C D ∧ is_midpoint ({x := 0, y := -9}) A D

theorem find_fourth_vertex :
  ∃ D : Point,
    (is_parallelogram ({x := 0, y := -9}) ({x := 2, y := 6}) ({x := 4, y := 5}) D)
    ∧ ((D = {x := 2, y := -10}) ∨ (D = {x := -2, y := -8}) ∨ (D = {x := 6, y := 20})) :=
sorry

end NUMINAMATH_GPT_find_fourth_vertex_l440_44010


namespace NUMINAMATH_GPT_central_student_coins_l440_44098

theorem central_student_coins (n_students: ℕ) (total_coins : ℕ)
  (equidistant_same : Prop)
  (coin_exchange : Prop):
  (n_students = 16) →
  (total_coins = 3360) →
  (equidistant_same) →
  (coin_exchange) →
  ∃ coins_in_center: ℕ, coins_in_center = 280 :=
by
  intros
  sorry

end NUMINAMATH_GPT_central_student_coins_l440_44098


namespace NUMINAMATH_GPT_problem_arithmetic_sequence_l440_44058

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) := a + d * (n - 1)

theorem problem_arithmetic_sequence (a d : ℝ) (h₁ : d < 0) (h₂ : (arithmetic_sequence a d 1)^2 = (arithmetic_sequence a d 9)^2):
  (arithmetic_sequence a d 5) = 0 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_problem_arithmetic_sequence_l440_44058


namespace NUMINAMATH_GPT_solve_system_l440_44023

variable {x y z : ℝ}

theorem solve_system :
  (y + z = 16 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 13 - 4 * z) →
  2 * x + 2 * y + 2 * z = 11 / 3 :=
by
  intros h1 h2 h3
  -- proof skips, to be completed
  sorry

end NUMINAMATH_GPT_solve_system_l440_44023


namespace NUMINAMATH_GPT_max_value_OP_OQ_l440_44009

def circle_1_polar_eq (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

def circle_2_polar_eq (rho theta : ℝ) : Prop :=
  rho = 2 * Real.sin theta

theorem max_value_OP_OQ (alpha : ℝ) :
  (∃ rho1 rho2 : ℝ, circle_1_polar_eq rho1 alpha ∧ circle_2_polar_eq rho2 alpha) ∧
  (∃ max_OP_OQ : ℝ, max_OP_OQ = 4) :=
sorry

end NUMINAMATH_GPT_max_value_OP_OQ_l440_44009


namespace NUMINAMATH_GPT_distance_ratio_l440_44056

variables (dw dr : ℝ)

theorem distance_ratio (h1 : 4 * (dw / 4) + 8 * (dr / 8) = 8)
  (h2 : dw + dr = 8)
  (h3 : (dw / 4) + (dr / 8) = 1.5) :
  dw / dr = 1 :=
by
  sorry

end NUMINAMATH_GPT_distance_ratio_l440_44056


namespace NUMINAMATH_GPT_compare_xyz_l440_44048

open Real

theorem compare_xyz (x y z : ℝ) : x = Real.log π → y = log 2 / log 5 → z = exp (-1 / 2) → y < z ∧ z < x := by
  intros h_x h_y h_z
  sorry

end NUMINAMATH_GPT_compare_xyz_l440_44048


namespace NUMINAMATH_GPT_solve_fractional_equation_l440_44015

theorem solve_fractional_equation (x : ℝ) (h : (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : 
  x = 1 :=
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l440_44015


namespace NUMINAMATH_GPT_notebook_problem_l440_44059

theorem notebook_problem :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 2 * x + 5 * y + 6 * z = 62 ∧ x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_notebook_problem_l440_44059


namespace NUMINAMATH_GPT_algebraic_identity_l440_44042

theorem algebraic_identity (theta : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < theta) (h2 : theta < π) (h3 : x + x⁻¹ = 2 * Real.cos theta) : 
  x^n + (x⁻¹)^n = 2 * Real.cos (n * theta) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l440_44042


namespace NUMINAMATH_GPT_negation_of_P_equiv_l440_44082

-- Define the proposition P
def P : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of P equivalently
theorem negation_of_P_equiv :
  ¬ P ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := 
sorry

end NUMINAMATH_GPT_negation_of_P_equiv_l440_44082


namespace NUMINAMATH_GPT_jake_not_drop_coffee_percentage_l440_44065

-- Definitions for the conditions
def trip_probability : ℝ := 0.40
def drop_when_trip_probability : ℝ := 0.25

-- The question and proof statement
theorem jake_not_drop_coffee_percentage :
  100 * (1 - trip_probability * drop_when_trip_probability) = 90 :=
by
  sorry

end NUMINAMATH_GPT_jake_not_drop_coffee_percentage_l440_44065


namespace NUMINAMATH_GPT_f_neg_l440_44045

variable (f : ℝ → ℝ)

-- Given condition that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The form of f for x ≥ 0
def f_pos (x : ℝ) (h : 0 ≤ x) : f x = -x^2 + 2 * x := sorry

-- Objective to prove f(x) for x < 0
theorem f_neg {x : ℝ} (h : x < 0) (hf_odd : odd_function f) (hf_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x) : f x = x^2 + 2 * x := 
by 
  sorry

end NUMINAMATH_GPT_f_neg_l440_44045


namespace NUMINAMATH_GPT_sum_of_sequence_l440_44040

def a (n : ℕ) : ℕ := 2 * n + 1 + 2^n

def S (n : ℕ) : ℕ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_of_sequence (n : ℕ) : S n = n^2 + 2 * n + 2^(n + 1) - 2 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l440_44040
