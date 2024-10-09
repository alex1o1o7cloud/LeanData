import Mathlib

namespace tan_alpha_value_sin_cos_expression_l903_90393

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_value (α : ℝ) (h1 : Real.tan (α + Real.pi / 4) = 2) : tan_alpha α = 1 / 3 :=
by
  sorry

theorem sin_cos_expression (α : ℝ) (h2 : tan_alpha α = 1 / 3) :
  (Real.sin (2 * α) - Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 5 / 18 :=
by
  sorry

end tan_alpha_value_sin_cos_expression_l903_90393


namespace calculate_train_speed_l903_90327

def speed_train_excluding_stoppages (distance_per_hour_including_stoppages : ℕ) (stoppage_minutes_per_hour : ℕ) : ℕ :=
  let effective_running_time_per_hour := 60 - stoppage_minutes_per_hour
  let effective_running_time_in_hours := effective_running_time_per_hour / 60
  distance_per_hour_including_stoppages / effective_running_time_in_hours

theorem calculate_train_speed :
  speed_train_excluding_stoppages 42 4 = 45 :=
by
  sorry

end calculate_train_speed_l903_90327


namespace jane_change_l903_90370

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l903_90370


namespace base_conversion_sum_correct_l903_90398

theorem base_conversion_sum_correct :
  (253 / 8 / 13 / 3 + 245 / 7 / 35 / 6 : ℚ) = 339 / 23 := sorry

end base_conversion_sum_correct_l903_90398


namespace incorrect_square_root_0_2_l903_90301

theorem incorrect_square_root_0_2 :
  (0.45)^2 = 0.2 ∧ (0.02)^2 ≠ 0.2 :=
by
  sorry

end incorrect_square_root_0_2_l903_90301


namespace find_range_of_a_l903_90375

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (2 * x) - a * x

theorem find_range_of_a (a : ℝ) :
  (∀ x > 0, f x a > a * x^2 + 1) → a ≤ 2 :=
by
  sorry

end find_range_of_a_l903_90375


namespace a_gt_b_iff_a_ln_a_gt_b_ln_b_l903_90324

theorem a_gt_b_iff_a_ln_a_gt_b_ln_b {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (a > b) ↔ (a + Real.log a > b + Real.log b) :=
by sorry

end a_gt_b_iff_a_ln_a_gt_b_ln_b_l903_90324


namespace find_k_value_l903_90377

theorem find_k_value (x k : ℝ) (hx : Real.logb 9 3 = x) (hk : Real.logb 3 81 = k * x) : k = 8 :=
by sorry

end find_k_value_l903_90377


namespace new_job_larger_than_original_l903_90300

theorem new_job_larger_than_original (original_workers original_days new_workers new_days : ℕ) 
  (h_original_workers : original_workers = 250)
  (h_original_days : original_days = 16)
  (h_new_workers : new_workers = 600)
  (h_new_days : new_days = 20) :
  (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry

end new_job_larger_than_original_l903_90300


namespace first_year_after_2023_with_digit_sum_8_l903_90321

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2023_with_digit_sum_8 : ∃ (y : ℕ), y > 2023 ∧ sum_of_digits y = 8 ∧ ∀ z, (z > 2023 ∧ sum_of_digits z = 8) → y ≤ z :=
by sorry

end first_year_after_2023_with_digit_sum_8_l903_90321


namespace A_work_days_l903_90333

theorem A_work_days {total_wages B_share : ℝ} (B_work_days : ℝ) (total_wages_eq : total_wages = 5000) 
    (B_share_eq : B_share = 3333) (B_rate : ℝ) (correct_rate : B_rate = 1 / B_work_days) :
    ∃x : ℝ, B_share / (total_wages - B_share) = B_rate / (1 / x) ∧ total_wages - B_share = 5000 - B_share ∧ B_work_days = 10 -> x = 20 :=
by
  sorry

end A_work_days_l903_90333


namespace average_leaves_per_hour_l903_90303

theorem average_leaves_per_hour :
  let leaves_first_hour := 7
  let leaves_second_hour := 4
  let leaves_third_hour := 4
  let total_hours := 3
  let total_leaves := leaves_first_hour + leaves_second_hour + leaves_third_hour
  let average_leaves_per_hour := total_leaves / total_hours
  average_leaves_per_hour = 5 := by
  sorry

end average_leaves_per_hour_l903_90303


namespace cyclic_path_1310_to_1315_l903_90369

theorem cyclic_path_1310_to_1315 :
  ∀ (n : ℕ), (n % 6 = 2 → (n + 5) % 6 = 3) :=
by
  sorry

end cyclic_path_1310_to_1315_l903_90369


namespace number_of_officers_l903_90312

theorem number_of_officers
  (avg_all : ℝ := 120)
  (avg_officer : ℝ := 420)
  (avg_non_officer : ℝ := 110)
  (num_non_officer : ℕ := 450) :
  ∃ O : ℕ, avg_all * (O + num_non_officer) = avg_officer * O + avg_non_officer * num_non_officer ∧ O = 15 :=
by
  sorry

end number_of_officers_l903_90312


namespace paint_remaining_after_two_days_l903_90328

-- Define the conditions
def original_paint_amount := 1
def paint_used_day1 := original_paint_amount * (1/4)
def remaining_paint_after_day1 := original_paint_amount - paint_used_day1
def paint_used_day2 := remaining_paint_after_day1 * (1/2)
def remaining_paint_after_day2 := remaining_paint_after_day1 - paint_used_day2

-- Theorem to be proved
theorem paint_remaining_after_two_days :
  remaining_paint_after_day2 = (3/8) * original_paint_amount := sorry

end paint_remaining_after_two_days_l903_90328


namespace train_length_calculation_l903_90355

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end train_length_calculation_l903_90355


namespace total_bouncy_balls_l903_90386

-- Definitions of the given quantities
def r : ℕ := 4 -- number of red packs
def y : ℕ := 8 -- number of yellow packs
def g : ℕ := 4 -- number of green packs
def n : ℕ := 10 -- number of balls per pack

-- Proof statement to show the correct total number of balls
theorem total_bouncy_balls : r * n + y * n + g * n = 160 := by
  sorry

end total_bouncy_balls_l903_90386


namespace tangent_line_eq_l903_90310

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

def M : ℝ×ℝ := (2, -3)

theorem tangent_line_eq (x y : ℝ) (h : y = f x) (h' : (x, y) = M) :
  2 * x - y - 7 = 0 :=
sorry

end tangent_line_eq_l903_90310


namespace train_crossing_time_l903_90356

noncomputable def time_to_cross_platform
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph / 3.6
  let total_distance := length_train + length_platform
  total_distance / speed_ms

theorem train_crossing_time
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ)
  (h_speed : speed_kmph = 72)
  (h_train_length : length_train = 280.0416)
  (h_platform_length : length_platform = 240) :
  time_to_cross_platform speed_kmph length_train length_platform = 26.00208 := by
  sorry

end train_crossing_time_l903_90356


namespace complex_number_sum_l903_90357

noncomputable def x : ℝ := 3 / 5
noncomputable def y : ℝ := -3 / 5

theorem complex_number_sum :
  (x + y) = -2 / 5 := 
by
  sorry

end complex_number_sum_l903_90357


namespace valve_rate_difference_l903_90311

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end valve_rate_difference_l903_90311


namespace proof_problem_l903_90365

variable (a b c m : ℝ)

-- Condition
def condition : Prop := m = (c * a * b) / (a + b)

-- Question
def question : Prop := b = (m * a) / (c * a - m)

-- Proof statement
theorem proof_problem (h : condition a b c m) : question a b c m := 
sorry

end proof_problem_l903_90365


namespace common_ratio_of_arithmetic_seq_l903_90346

theorem common_ratio_of_arithmetic_seq (a_1 q : ℝ) 
  (h1 : a_1 + a_1 * q^2 = 10) 
  (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) : 
  q = 1 / 2 := 
by 
  sorry

end common_ratio_of_arithmetic_seq_l903_90346


namespace harry_took_5_eggs_l903_90342

theorem harry_took_5_eggs (initial : ℕ) (left : ℕ) (took : ℕ) 
  (h1 : initial = 47) (h2 : left = 42) (h3 : left = initial - took) : 
  took = 5 :=
sorry

end harry_took_5_eggs_l903_90342


namespace pages_per_hour_l903_90385

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l903_90385


namespace minimum_fruits_l903_90383

open Nat

theorem minimum_fruits (n : ℕ) :
    (n % 3 = 2) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 6 = 5) →
    n = 59 := by
  sorry

end minimum_fruits_l903_90383


namespace sampling_is_systematic_l903_90343

-- Defining the conditions
def mock_exam (rooms students_per_room seat_selected: ℕ) : Prop :=
  rooms = 80 ∧ students_per_room = 30 ∧ seat_selected = 15

-- Theorem statement
theorem sampling_is_systematic 
  (rooms students_per_room seat_selected: ℕ)
  (h: mock_exam rooms students_per_room seat_selected) : 
  sampling_method = "Systematic sampling" :=
sorry

end sampling_is_systematic_l903_90343


namespace present_ages_l903_90368

theorem present_ages
  (R D K : ℕ) (x : ℕ)
  (H1 : R = 4 * x)
  (H2 : D = 3 * x)
  (H3 : K = 5 * x)
  (H4 : R + 6 = 26)
  (H5 : (R + 8) + (D + 8) = K) :
  D = 15 ∧ K = 51 :=
sorry

end present_ages_l903_90368


namespace min_distance_A_D_l903_90336

theorem min_distance_A_D (A B C E D : Type) 
  (d_AB d_BC d_CE d_ED : ℝ) 
  (h1 : d_AB = 12) 
  (h2 : d_BC = 7) 
  (h3 : d_CE = 2) 
  (h4 : d_ED = 5) : 
  ∃ d_AD : ℝ, d_AD = 2 := 
by
  sorry

end min_distance_A_D_l903_90336


namespace shakes_indeterminable_l903_90317

variable {B S C x : ℝ}

theorem shakes_indeterminable (h1 : 3 * B + x * S + C = 130) (h2 : 4 * B + 10 * S + C = 164.5) : 
  ¬ (∃ x, 3 * B + x * S + C = 130 ∧ 4 * B + 10 * S + C = 164.5) :=
by
  sorry

end shakes_indeterminable_l903_90317


namespace smallest_square_value_l903_90392

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h₁ : ∃ r : ℕ, 15 * a + 16 * b = r^2) (h₂ : ∃ s : ℕ, 16 * a - 15 * b = s^2) :
  ∃ (m : ℕ), m = 481^2 ∧ (15 * a + 16 * b = m ∨ 16 * a - 15 * b = m) :=
  sorry

end smallest_square_value_l903_90392


namespace Nathan_daily_hours_l903_90387

theorem Nathan_daily_hours (x : ℝ) 
  (h1 : 14 * x + 35 = 77) : 
  x = 3 := 
by 
  sorry

end Nathan_daily_hours_l903_90387


namespace basketball_three_point_shots_l903_90361

theorem basketball_three_point_shots (t h f : ℕ) 
  (h1 : 2 * t = 6 * h)
  (h2 : f = h - 4)
  (h3: 2 * t + 3 * h + f = 76)
  (h4: t + h + f = 40) : h = 8 :=
sorry

end basketball_three_point_shots_l903_90361


namespace european_savings_correct_l903_90390

noncomputable def movie_ticket_price : ℝ := 8
noncomputable def popcorn_price : ℝ := 8 - 3
noncomputable def drink_price : ℝ := popcorn_price + 1
noncomputable def candy_price : ℝ := drink_price / 2
noncomputable def hotdog_price : ℝ := 5

noncomputable def monday_discount_popcorn : ℝ := 0.15 * popcorn_price
noncomputable def wednesday_discount_candy : ℝ := 0.10 * candy_price
noncomputable def friday_discount_drink : ℝ := 0.05 * drink_price

noncomputable def monday_price : ℝ := 22
noncomputable def wednesday_price : ℝ := 20
noncomputable def friday_price : ℝ := 25
noncomputable def weekend_price : ℝ := 25
noncomputable def monday_exchange_rate : ℝ := 0.85
noncomputable def wednesday_exchange_rate : ℝ := 0.85
noncomputable def friday_exchange_rate : ℝ := 0.83
noncomputable def weekend_exchange_rate : ℝ := 0.81

noncomputable def total_cost_monday : ℝ := movie_ticket_price + (popcorn_price - monday_discount_popcorn) + drink_price + candy_price + hotdog_price
noncomputable def savings_monday_usd : ℝ := total_cost_monday - monday_price
noncomputable def savings_monday_eur : ℝ := savings_monday_usd * monday_exchange_rate

noncomputable def total_cost_wednesday : ℝ := movie_ticket_price + popcorn_price + drink_price + (candy_price - wednesday_discount_candy) + hotdog_price
noncomputable def savings_wednesday_usd : ℝ := total_cost_wednesday - wednesday_price
noncomputable def savings_wednesday_eur : ℝ := savings_wednesday_usd * wednesday_exchange_rate

noncomputable def total_cost_friday : ℝ := movie_ticket_price + popcorn_price + (drink_price - friday_discount_drink) + candy_price + hotdog_price
noncomputable def savings_friday_usd : ℝ := total_cost_friday - friday_price
noncomputable def savings_friday_eur : ℝ := savings_friday_usd * friday_exchange_rate

noncomputable def total_cost_weekend : ℝ := movie_ticket_price + popcorn_price + drink_price + candy_price + hotdog_price
noncomputable def savings_weekend_usd : ℝ := total_cost_weekend - weekend_price
noncomputable def savings_weekend_eur : ℝ := savings_weekend_usd * weekend_exchange_rate

theorem european_savings_correct :
  savings_monday_eur = 3.61 ∧ 
  savings_wednesday_eur = 5.70 ∧ 
  savings_friday_eur = 1.41 ∧ 
  savings_weekend_eur = 1.62 :=
by
  sorry

end european_savings_correct_l903_90390


namespace Q_has_negative_root_l903_90308

def Q (x : ℝ) : ℝ := x^7 + 2 * x^5 + 5 * x^3 - x + 12

theorem Q_has_negative_root : ∃ x : ℝ, x < 0 ∧ Q x = 0 :=
by
  sorry

end Q_has_negative_root_l903_90308


namespace alcohol_percentage_in_second_vessel_l903_90381

open Real

theorem alcohol_percentage_in_second_vessel (x : ℝ) (h : (0.2 * 2) + (0.01 * x * 6) = 8 * 0.28) : 
  x = 30.666666666666668 :=
by 
  sorry

end alcohol_percentage_in_second_vessel_l903_90381


namespace ratio_of_dogs_to_cats_l903_90394

-- Definition of conditions
def total_animals : Nat := 21
def cats_to_spay : Nat := 7
def dogs_to_spay : Nat := total_animals - cats_to_spay

-- Ratio of dogs to cats
def dogs_to_cats_ratio : Nat := dogs_to_spay / cats_to_spay

-- Statement to prove
theorem ratio_of_dogs_to_cats : dogs_to_cats_ratio = 2 :=
by
  -- Proof goes here
  sorry

end ratio_of_dogs_to_cats_l903_90394


namespace range_of_k_l903_90344

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ (x, y) = (1, 2)) →
  (3 < k ∧ k < 7) :=
by
  intros hxy
  sorry

end range_of_k_l903_90344


namespace problem_1_problem_2_problem_3_l903_90335

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_1 :
  (∀ x : ℝ, f 1 x ≥ f 1 1) :=
by sorry

theorem problem_2 (x e : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) (hf : f a x = 1) :
  0 ≤ a ∧ a ≤ 1 :=
by sorry

theorem problem_3 (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 1 → f a x ≥ f a (1 / x)) → 1 ≤ a :=
by sorry

end problem_1_problem_2_problem_3_l903_90335


namespace eliminating_y_l903_90331

theorem eliminating_y (x y : ℝ) (h1 : y = x + 3) (h2 : 2 * x - y = 5) : 2 * x - x - 3 = 5 :=
by {
  sorry
}

end eliminating_y_l903_90331


namespace sqrt_product_l903_90397

theorem sqrt_product (h54 : Real.sqrt 54 = 3 * Real.sqrt 6)
                     (h32 : Real.sqrt 32 = 4 * Real.sqrt 2)
                     (h6 : Real.sqrt 6 = Real.sqrt 6) :
    Real.sqrt 54 * Real.sqrt 32 * Real.sqrt 6 = 72 * Real.sqrt 2 := by
  sorry

end sqrt_product_l903_90397


namespace find_preimage_l903_90391

def mapping (x y : ℝ) : ℝ × ℝ :=
  (x + y, x - y)

theorem find_preimage :
  mapping 2 1 = (3, 1) :=
by
  sorry

end find_preimage_l903_90391


namespace Ellen_won_17_legos_l903_90329

theorem Ellen_won_17_legos (initial_legos : ℕ) (current_legos : ℕ) (h₁ : initial_legos = 2080) (h₂ : current_legos = 2097) : 
  current_legos - initial_legos = 17 := 
  by 
    sorry

end Ellen_won_17_legos_l903_90329


namespace complex_quadrant_l903_90373

theorem complex_quadrant 
  (z : ℂ) 
  (h : (2 + 3 * Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end complex_quadrant_l903_90373


namespace scrambled_eggs_count_l903_90376

-- Definitions based on the given conditions
def num_sausages := 3
def time_per_sausage := 5
def time_per_egg := 4
def total_time := 39

-- Prove that Kira scrambled 6 eggs
theorem scrambled_eggs_count : (total_time - num_sausages * time_per_sausage) / time_per_egg = 6 := by
  sorry

end scrambled_eggs_count_l903_90376


namespace magnitude_product_complex_l903_90315

theorem magnitude_product_complex :
  let z1 := Complex.mk 7 (-4)
  let z2 := Complex.mk 3 11
  Complex.abs (z1 * z2) = Real.sqrt 8450 :=
by
  sorry

end magnitude_product_complex_l903_90315


namespace total_clothes_count_l903_90374

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end total_clothes_count_l903_90374


namespace range_of_a_l903_90307

noncomputable def line_eq (a : ℝ) (x y : ℝ) : ℝ := 3 * x - 2 * y + a 

def pointA : ℝ × ℝ := (3, 1)
def pointB : ℝ × ℝ := (-4, 6)

theorem range_of_a :
  (line_eq a pointA.1 pointA.2) * (line_eq a pointB.1 pointB.2) < 0 ↔ -7 < a ∧ a < 24 := sorry

end range_of_a_l903_90307


namespace multiple_of_four_l903_90347

theorem multiple_of_four (n : ℕ) (h1 : ∃ k : ℕ, 12 + 4 * k = n) (h2 : 21 = (n - 12) / 4 + 1) : n = 96 := 
sorry

end multiple_of_four_l903_90347


namespace number_of_people_prefer_soda_l903_90339

-- Given conditions
def total_people : ℕ := 600
def central_angle_soda : ℝ := 198
def full_circle_angle : ℝ := 360

-- Problem statement
theorem number_of_people_prefer_soda : 
  (total_people : ℝ) * (central_angle_soda / full_circle_angle) = 330 := by
  sorry

end number_of_people_prefer_soda_l903_90339


namespace number_of_clerks_l903_90388

theorem number_of_clerks 
  (num_officers : ℕ) 
  (num_clerks : ℕ) 
  (avg_salary_staff : ℕ) 
  (avg_salary_officers : ℕ) 
  (avg_salary_clerks : ℕ)
  (h1 : avg_salary_staff = 90)
  (h2 : avg_salary_officers = 600)
  (h3 : avg_salary_clerks = 84)
  (h4 : num_officers = 2)
  : num_clerks = 170 :=
sorry

end number_of_clerks_l903_90388


namespace intersection_M_N_eq_l903_90320

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_l903_90320


namespace enclosed_area_l903_90337

noncomputable def calculateArea : ℝ :=
  ∫ (x : ℝ) in (1 / 2)..2, 1 / x

theorem enclosed_area : calculateArea = 2 * Real.log 2 :=
by
  sorry

end enclosed_area_l903_90337


namespace expected_value_shorter_gentlemen_l903_90380

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l903_90380


namespace mike_spent_total_l903_90334

-- Define the prices of the items
def price_trumpet : ℝ := 145.16
def price_song_book : ℝ := 5.84

-- Define the total amount spent
def total_spent : ℝ := price_trumpet + price_song_book

-- The theorem to be proved
theorem mike_spent_total :
  total_spent = 151.00 :=
sorry

end mike_spent_total_l903_90334


namespace moles_of_NaCl_formed_l903_90338

-- Define the balanced chemical reaction and quantities
def chemical_reaction :=
  "NaOH + HCl → NaCl + H2O"

-- Define the initial moles of sodium hydroxide (NaOH) and hydrochloric acid (HCl)
def moles_NaOH : ℕ := 2
def moles_HCl : ℕ := 2

-- The stoichiometry from the balanced equation: 1 mole NaOH reacts with 1 mole HCl to produce 1 mole NaCl.
def stoichiometry_NaOH_to_NaCl : ℕ := 1
def stoichiometry_HCl_to_NaCl : ℕ := 1

-- Given the initial conditions, prove that 2 moles of NaCl are formed.
theorem moles_of_NaCl_formed :
  (moles_NaOH = 2) → (moles_HCl = 2) → 2 = 2 :=
by 
  sorry

end moles_of_NaCl_formed_l903_90338


namespace tan_ratio_l903_90341

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l903_90341


namespace uniqueSumEqualNumber_l903_90372

noncomputable def sumPreceding (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem uniqueSumEqualNumber :
  ∃! n : ℕ, sumPreceding n = n := by
  sorry

end uniqueSumEqualNumber_l903_90372


namespace moles_of_hcl_l903_90348

-- Definitions according to the conditions
def methane := 1 -- 1 mole of methane (CH₄)
def chlorine := 2 -- 2 moles of chlorine (Cl₂)
def hcl := 1 -- The expected number of moles of Hydrochloric acid (HCl)

-- The Lean 4 statement (no proof required)
theorem moles_of_hcl (methane chlorine : ℕ) : hcl = 1 :=
by sorry

end moles_of_hcl_l903_90348


namespace coordinate_sum_of_point_on_graph_l903_90353

theorem coordinate_sum_of_point_on_graph (g : ℕ → ℕ) (h : ℕ → ℕ)
  (h1 : g 2 = 8)
  (h2 : ∀ x, h x = 3 * (g x) ^ 2) :
  2 + h 2 = 194 :=
by
  sorry

end coordinate_sum_of_point_on_graph_l903_90353


namespace exists_f_condition_l903_90384

open Nat

-- Define the function φ from ℕ to ℕ
variable (ϕ : ℕ → ℕ)

-- The formal statement capturing the given math proof problem
theorem exists_f_condition (ϕ : ℕ → ℕ) : 
  ∃ (f : ℕ → ℤ), (∀ x : ℕ, f x > f (ϕ x)) :=
  sorry

end exists_f_condition_l903_90384


namespace f_is_increasing_l903_90371

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 3 * x

theorem f_is_increasing : ∀ (x : ℝ), (deriv f x) > 0 :=
by
  intro x
  calc
    deriv f x = 2 * Real.exp (2 * x) + 3 := by sorry
    _ > 0 := by sorry

end f_is_increasing_l903_90371


namespace area_PCD_eq_l903_90399

/-- Define the points P, D, and C as given in the conditions. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 18⟩
def D : Point := ⟨3, 18⟩
def C (q : ℝ) : Point := ⟨0, q⟩

/-- Define the function to compute the area of triangle PCD given q. -/
noncomputable def area_triangle_PCD (q : ℝ) : ℝ :=
  1 / 2 * (D.x - P.x) * (P.y - q)

theorem area_PCD_eq (q : ℝ) : 
  area_triangle_PCD q = 27 - 3 / 2 * q := 
by 
  sorry

end area_PCD_eq_l903_90399


namespace sum_abc_of_quadrilateral_l903_90306

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_abc_of_quadrilateral :
  let p1 := (0, 0)
  let p2 := (4, 3)
  let p3 := (5, 2)
  let p4 := (4, -1)
  let perimeter := 
    distance p1 p2 + distance p2 p3 + distance p3 p4 + distance p4 p1
  let a : ℤ := 1    -- corresponding to the equivalent simplified distances to √5 parts
  let b : ℤ := 2    -- corresponding to the equivalent simplified distances to √2 parts
  let c : ℤ := 9    -- rest constant integer simplified part
  a + b + c = 12 :=
by
  sorry

end sum_abc_of_quadrilateral_l903_90306


namespace range_of_a_l903_90305

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 64 > 0) → -16 < a ∧ a < 16 :=
by
  -- The proof steps will go here
  sorry

end range_of_a_l903_90305


namespace geometric_sequence_sum_l903_90304

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₁ : a 3 = 4) (h₂ : a 2 + a 4 = -10) (h₃ : |q| > 1) : 
  (a 0 + a 1 + a 2 + a 3 = -5) := 
by 
  sorry

end geometric_sequence_sum_l903_90304


namespace fraction_relation_l903_90349

-- Definitions for arithmetic sequences and their sums
noncomputable def a_n (a₁ d₁ n : ℕ) := a₁ + (n - 1) * d₁
noncomputable def b_n (b₁ d₂ n : ℕ) := b₁ + (n - 1) * d₂

noncomputable def A_n (a₁ d₁ n : ℕ) := n * a₁ + n * (n - 1) * d₁ / 2
noncomputable def B_n (b₁ d₂ n : ℕ) := n * b₁ + n * (n - 1) * d₂ / 2

-- Theorem statement
theorem fraction_relation (a₁ d₁ b₁ d₂ : ℕ) (h : ∀ n : ℕ, B_n a₁ d₁ n ≠ 0 → A_n a₁ d₁ n / B_n b₁ d₂ n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b_n b₁ d₂ n ≠ 0 → a_n a₁ d₁ n / b_n b₁ d₂ n = (4 * n - 3) / (6 * n - 2) :=
sorry

end fraction_relation_l903_90349


namespace number_of_20_paise_coins_l903_90319

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 336) (h2 : (20 / 100 : ℚ) * x + (25 / 100 : ℚ) * y = 71) :
    x = 260 :=
by
  sorry

end number_of_20_paise_coins_l903_90319


namespace max_not_expressed_as_linear_comb_l903_90360

theorem max_not_expressed_as_linear_comb {a b c : ℕ} (h_coprime_ab : Nat.gcd a b = 1)
                                        (h_coprime_bc : Nat.gcd b c = 1)
                                        (h_coprime_ca : Nat.gcd c a = 1) :
    Nat := sorry

end max_not_expressed_as_linear_comb_l903_90360


namespace simplify_expression_l903_90362

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x ^ 2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_expression_l903_90362


namespace Lisa_days_l903_90358

theorem Lisa_days (L : ℝ) (h : 1/4 + 1/2 + 1/L = 1/1.09090909091) : L = 2.93333333333 :=
by sorry

end Lisa_days_l903_90358


namespace transformed_conic_symmetric_eq_l903_90396

def conic_E (x y : ℝ) := x^2 + 2 * x * y + y^2 + 3 * x + y
def line_l (x y : ℝ) := 2 * x - y - 1

def transformed_conic_equation (x y : ℝ) := x^2 + 14 * x * y + 49 * y^2 - 21 * x + 103 * y + 54

theorem transformed_conic_symmetric_eq (x y : ℝ) :
  (∀ x y, conic_E x y = 0 → 
    ∃ x' y', line_l x' y' = 0 ∧ conic_E x' y' = 0 ∧ transformed_conic_equation x y = 0) :=
sorry

end transformed_conic_symmetric_eq_l903_90396


namespace factor_expression_l903_90345

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) :=
by
  sorry

end factor_expression_l903_90345


namespace weight_of_replaced_person_l903_90364

theorem weight_of_replaced_person (avg_weight : ℝ) (new_person_weight : ℝ)
  (h1 : new_person_weight = 65)
  (h2 : ∀ (initial_avg_weight : ℝ), 8 * (initial_avg_weight + 2.5) - 8 * initial_avg_weight = new_person_weight - avg_weight) :
  avg_weight = 45 := 
by
  -- Proof goes here
  sorry

end weight_of_replaced_person_l903_90364


namespace triangle_perimeter_l903_90326

-- Let the lengths of the sides of the triangle be a, b, c.
variables (a b c : ℕ)
-- To represent the sides with specific lengths as stated in the problem.
def side1 := 2
def side2 := 5

-- The condition that the third side must be an odd integer.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Setting up the third side based on the given conditions.
def third_side_odd (c : ℕ) : Prop := 3 < c ∧ c < 7 ∧ is_odd c

-- The perimeter of the triangle.
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The main theorem to prove.
theorem triangle_perimeter (c : ℕ) (h_odd : third_side_odd c) : perimeter side1 side2 c = 12 :=
by
  sorry

end triangle_perimeter_l903_90326


namespace gwendolyn_read_time_l903_90395

theorem gwendolyn_read_time :
  let rate := 200 -- sentences per hour
  let paragraphs_per_page := 30
  let sentences_per_paragraph := 15
  let pages := 100
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  let total_time := total_sentences / rate
  total_time = 225 :=
by
  sorry

end gwendolyn_read_time_l903_90395


namespace linear_equation_a_is_minus_one_l903_90379

theorem linear_equation_a_is_minus_one (a : ℝ) (x : ℝ) :
  ((a - 1) * x ^ (2 - |a|) + 5 = 0) → (2 - |a| = 1) → (a ≠ 1) → a = -1 :=
by
  intros h1 h2 h3
  sorry

end linear_equation_a_is_minus_one_l903_90379


namespace find_three_digit_number_l903_90314

-- Define the function that calculates the total number of digits required
def total_digits (x : ℕ) : ℕ :=
  (if x >= 1 then 9 else 0) +
  (if x >= 10 then 90 * 2 else 0) +
  (if x >= 100 then 3 * (x - 99) else 0)

theorem find_three_digit_number : ∃ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ 2 * x = total_digits x := by
  sorry

end find_three_digit_number_l903_90314


namespace sam_wins_probability_l903_90359

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l903_90359


namespace simplify_expression_l903_90313

theorem simplify_expression (x : ℝ) : (3 * x + 8) + (50 * x + 25) = 53 * x + 33 := 
by sorry

end simplify_expression_l903_90313


namespace quadratic_inequality_range_of_k_l903_90382

theorem quadratic_inequality_range_of_k :
  ∀ k : ℝ , (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) ↔ (-1 < k ∧ k ≤ 0) :=
sorry

end quadratic_inequality_range_of_k_l903_90382


namespace bookstore_earnings_difference_l903_90318

def base_price_TOP := 8.0
def base_price_ABC := 23.0
def discount_TOP := 0.10
def discount_ABC := 0.05
def sales_tax := 0.07
def num_TOP_sold := 13
def num_ABC_sold := 4

def discounted_price (base_price discount : Float) : Float :=
  base_price * (1.0 - discount)

def final_price (discounted_price tax : Float) : Float :=
  discounted_price * (1.0 + tax)

def total_earnings (final_price : Float) (quantity : Nat) : Float :=
  final_price * (quantity.toFloat)

theorem bookstore_earnings_difference :
  let discounted_price_TOP := discounted_price base_price_TOP discount_TOP
  let discounted_price_ABC := discounted_price base_price_ABC discount_ABC
  let final_price_TOP := final_price discounted_price_TOP sales_tax
  let final_price_ABC := final_price discounted_price_ABC sales_tax
  let total_earnings_TOP := total_earnings final_price_TOP num_TOP_sold
  let total_earnings_ABC := total_earnings final_price_ABC num_ABC_sold
  total_earnings_TOP - total_earnings_ABC = 6.634 :=
by
  sorry

end bookstore_earnings_difference_l903_90318


namespace ratio_jerky_l903_90389

/-
  Given conditions:
  1. Janette camps for 5 days.
  2. She has an initial 40 pieces of beef jerky.
  3. She eats 4 pieces of beef jerky per day.
  4. She will have 10 pieces of beef jerky left after giving some to her brother.

  Prove that the ratio of the pieces of beef jerky she gives to her brother 
  to the remaining pieces is 1:1.
-/

theorem ratio_jerky (days : ℕ) (initial_jerky : ℕ) (jerky_per_day : ℕ) (jerky_left_after_trip : ℕ)
  (h1 : days = 5) (h2 : initial_jerky = 40) (h3 : jerky_per_day = 4) (h4 : jerky_left_after_trip = 10) :
  (initial_jerky - days * jerky_per_day - jerky_left_after_trip) = jerky_left_after_trip :=
by
  sorry

end ratio_jerky_l903_90389


namespace remainder_when_dividing_l903_90332

theorem remainder_when_dividing (c d : ℕ) (p q : ℕ) :
  c = 60 * p + 47 ∧ d = 45 * q + 14 → (c + d) % 15 = 1 :=
by
  sorry

end remainder_when_dividing_l903_90332


namespace correct_answer_l903_90354

theorem correct_answer (A B C D : String) (sentence : String)
  (h1 : A = "us")
  (h2 : B = "we")
  (h3 : C = "our")
  (h4 : D = "ours")
  (h_sentence : sentence = "To save class time, our teacher has _ students do half of the exercise in class and complete the other half for homework.") :
  sentence = "To save class time, our teacher has " ++ A ++ " students do half of the exercise in class and complete the other half for homework." :=
by
  sorry

end correct_answer_l903_90354


namespace least_positive_integer_x_l903_90351

theorem least_positive_integer_x : ∃ x : ℕ, ((2 * x)^2 + 2 * 43 * (2 * x) + 43^2) % 53 = 0 ∧ 0 < x ∧ (∀ y : ℕ, ((2 * y)^2 + 2 * 43 * (2 * y) + 43^2) % 53 = 0 → 0 < y → x ≤ y) := 
by
  sorry

end least_positive_integer_x_l903_90351


namespace common_chord_of_circles_l903_90325

theorem common_chord_of_circles :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x = 0) ∧ (x^2 + y^2 - 4 * y = 0) → (x + 2 * y = 0) :=
by
  sorry

end common_chord_of_circles_l903_90325


namespace shaded_area_of_squares_is_20_l903_90302

theorem shaded_area_of_squares_is_20 :
  ∀ (a b : ℝ), a = 2 → b = 6 → 
    (1/2) * a * a + (1/2) * b * b = 20 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end shaded_area_of_squares_is_20_l903_90302


namespace exists_negative_fraction_lt_four_l903_90352

theorem exists_negative_fraction_lt_four : 
  ∃ (x : ℚ), x < 0 ∧ |x| < 4 := 
sorry

end exists_negative_fraction_lt_four_l903_90352


namespace distance_between_riya_and_priya_l903_90378

theorem distance_between_riya_and_priya (speed_riya speed_priya : ℝ) (time_hours : ℝ)
  (h1 : speed_riya = 21) (h2 : speed_priya = 22) (h3 : time_hours = 1) :
  speed_riya * time_hours + speed_priya * time_hours = 43 := by
  sorry

end distance_between_riya_and_priya_l903_90378


namespace find_sum_of_coefficients_l903_90350

theorem find_sum_of_coefficients (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -(1/2) ∨ x > 1/3)) :
  a + b = -14 := 
sorry

end find_sum_of_coefficients_l903_90350


namespace number_of_outfits_l903_90330

theorem number_of_outfits (num_shirts : ℕ) (num_pants : ℕ) (num_shoe_types : ℕ) (shoe_styles_per_type : ℕ) (h_shirts : num_shirts = 4) (h_pants : num_pants = 4) (h_shoes : num_shoe_types = 2) (h_styles : shoe_styles_per_type = 2) :
  num_shirts * num_pants * (num_shoe_types * shoe_styles_per_type) = 64 :=
by {
  sorry
}

end number_of_outfits_l903_90330


namespace p_squared_plus_13_mod_n_eq_2_l903_90367

theorem p_squared_plus_13_mod_n_eq_2 (p : ℕ) (prime_p : Prime p) (h : p > 3) (n : ℕ) :
  (∃ (k : ℕ), p ^ 2 + 13 = k * n + 2) → n = 2 :=
by
  sorry

end p_squared_plus_13_mod_n_eq_2_l903_90367


namespace ben_owes_rachel_l903_90366

theorem ben_owes_rachel :
  let dollars_per_lawn := (13 : ℚ) / 3
  let lawns_mowed := (8 : ℚ) / 5
  let total_owed := (104 : ℚ) / 15
  dollars_per_lawn * lawns_mowed = total_owed := 
by 
  sorry

end ben_owes_rachel_l903_90366


namespace additional_toothpicks_needed_l903_90309

theorem additional_toothpicks_needed 
  (t : ℕ → ℕ)
  (h1 : t 1 = 4)
  (h2 : t 2 = 10)
  (h3 : t 3 = 18)
  (h4 : t 4 = 28)
  (h5 : t 5 = 40)
  (h6 : t 6 = 54) :
  t 6 - t 4 = 26 :=
by
  sorry

end additional_toothpicks_needed_l903_90309


namespace range_of_m_l903_90323

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 4^x - m * 2^x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l903_90323


namespace square_side_length_l903_90322

-- Definition of the problem (statements)
theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s * s) (hA : A = 49) : s = 7 := 
by 
  sorry

end square_side_length_l903_90322


namespace determine_x_l903_90316

theorem determine_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  have : ∀ y : ℝ, (5 * y + 1) * (2 * x - 3) = 0 := 
    sorry
  have : (2 * x - 3) = 0 := 
    sorry
  show x = 3 / 2
  sorry

end determine_x_l903_90316


namespace smallest_x_l903_90340

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l903_90340


namespace tim_watched_total_hours_tv_l903_90363

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l903_90363
