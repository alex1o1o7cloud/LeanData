import Mathlib

namespace train_B_speed_l75_7584

theorem train_B_speed (V_B : ℝ) : 
  (∀ t meet_A meet_B, 
     meet_A = 9 ∧
     meet_B = 4 ∧
     t = 70 ∧
     (t * meet_A) = (V_B * meet_B)) →
     V_B = 157.5 :=
by
  intros h
  sorry

end train_B_speed_l75_7584


namespace cloth_sold_l75_7535

theorem cloth_sold (C S M : ℚ) (P : ℚ) (hP : P = 1 / 3) (hG : 10 * S = (1 / 3) * (M * C)) (hS : S = (4 / 3) * C) : M = 40 := by
  sorry

end cloth_sold_l75_7535


namespace probability_sequence_correct_l75_7589

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l75_7589


namespace min_commission_deputies_l75_7590

theorem min_commission_deputies 
  (members : ℕ) 
  (brawls : ℕ) 
  (brawl_participants : brawls = 200) 
  (member_count : members = 200) :
  ∃ minimal_commission_members : ℕ, minimal_commission_members = 67 := 
sorry

end min_commission_deputies_l75_7590


namespace emilia_strCartons_l75_7548

theorem emilia_strCartons (total_cartons_needed cartons_bought cartons_blueberries : ℕ) (h1 : total_cartons_needed = 42) (h2 : cartons_blueberries = 7) (h3 : cartons_bought = 33) :
  (total_cartons_needed - (cartons_bought + cartons_blueberries)) = 2 :=
by
  sorry

end emilia_strCartons_l75_7548


namespace baseball_card_decrease_l75_7507

theorem baseball_card_decrease (x : ℝ) (h : (1 - x / 100) * (1 - x / 100) = 0.64) : x = 20 :=
by
  sorry

end baseball_card_decrease_l75_7507


namespace sequence_general_formula_l75_7550

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) 
    (h₂ : ∀ n : ℕ, 1 < n → a n = (n / (n - 1)) * a (n - 1)) : 
    ∀ n : ℕ, 1 ≤ n → a n = 3 * n :=
by
  -- Proof description here
  sorry

end sequence_general_formula_l75_7550


namespace find_sum_due_l75_7528

variable (BD TD FV : ℝ)

-- given conditions
def condition_1 : Prop := BD = 80
def condition_2 : Prop := TD = 70
def condition_3 : Prop := BD = TD + (TD * BD / FV)

-- goal statement
theorem find_sum_due (h1 : condition_1 BD) (h2 : condition_2 TD) (h3 : condition_3 BD TD FV) : FV = 560 :=
by
  sorry

end find_sum_due_l75_7528


namespace max_poly_l75_7514

noncomputable def poly (a b : ℝ) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_poly (a b : ℝ) (h : a + b = 4) :
  ∃ (a b : ℝ) (h : a + b = 4), poly a b = (7225 / 56) :=
sorry

end max_poly_l75_7514


namespace correct_regression_line_l75_7536

theorem correct_regression_line (h_neg_corr: ∀ x: ℝ, ∀ y: ℝ, y = -10*x + 200 ∨ y = 10*x + 200 ∨ y = -10*x - 200 ∨ y = 10*x - 200) 
(h_slope_neg : ∀ a b: ℝ, a < 0) 
(h_y_intercept: ∀ x: ℝ, x = 0 → 200 > 0 → y = 200) : 
∃ y: ℝ, y = -10*x + 200 :=
by
-- the proof will go here
sorry

end correct_regression_line_l75_7536


namespace correct_sampling_method_l75_7565

structure SchoolPopulation :=
  (senior : ℕ)
  (intermediate : ℕ)
  (junior : ℕ)

-- Define the school population
def school : SchoolPopulation :=
  { senior := 10, intermediate := 50, junior := 75 }

-- Define the condition for sampling method
def total_school_teachers (s : SchoolPopulation) : ℕ :=
  s.senior + s.intermediate + s.junior

-- The desired sample size
def sample_size : ℕ := 30

-- The correct sampling method based on the population strata
def stratified_sampling (s : SchoolPopulation) : Prop :=
  s.senior + s.intermediate + s.junior > 0

theorem correct_sampling_method : stratified_sampling school :=
by { sorry }

end correct_sampling_method_l75_7565


namespace product_mod_7_l75_7598

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l75_7598


namespace possible_values_for_a_l75_7527

def setM : Set ℝ := {x | x^2 + x - 6 = 0}
def setN (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_for_a (a : ℝ) : (∀ x, x ∈ setN a → x ∈ setM) ↔ (a = -1 ∨ a = 0 ∨ a = 2 / 3) := 
by
  sorry

end possible_values_for_a_l75_7527


namespace slices_leftover_is_9_l75_7511

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l75_7511


namespace find_valid_ns_l75_7557

theorem find_valid_ns (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, k^2 = (n^2 + 7 * n + 136) / (n-1)) : n = 5 ∨ n = 37 :=
sorry

end find_valid_ns_l75_7557


namespace find_number_l75_7566

theorem find_number (x : ℝ) (h : (x - 5) / 3 = 4) : x = 17 :=
by {
  sorry
}

end find_number_l75_7566


namespace arnaldo_bernaldo_distribute_toys_l75_7560

noncomputable def num_ways_toys_distributed (total_toys remaining_toys : ℕ) : ℕ :=
  if total_toys = 10 ∧ remaining_toys = 8 then 6561 - 256 else 0

theorem arnaldo_bernaldo_distribute_toys : num_ways_toys_distributed 10 8 = 6305 :=
by 
  -- Lean calculation for 3^8 = 6561 and 2^8 = 256 can be done as follows
  -- let three_power_eight := 3^8
  -- let two_power_eight := 2^8
  -- three_power_eight - two_power_eight = 6305
  sorry

end arnaldo_bernaldo_distribute_toys_l75_7560


namespace selling_price_to_achieve_profit_l75_7522

theorem selling_price_to_achieve_profit (num_pencils : ℝ) (cost_per_pencil : ℝ) (desired_profit : ℝ) (selling_price : ℝ) :
  num_pencils = 1800 →
  cost_per_pencil = 0.15 →
  desired_profit = 100 →
  selling_price = 0.21 :=
by
  sorry

end selling_price_to_achieve_profit_l75_7522


namespace percent_not_crust_l75_7552

-- Definitions as conditions
def pie_total_weight : ℕ := 200
def crust_weight : ℕ := 50

-- The theorem to be proven
theorem percent_not_crust : (pie_total_weight - crust_weight) / pie_total_weight * 100 = 75 := 
by
  sorry

end percent_not_crust_l75_7552


namespace find_mn_expression_l75_7510

-- Define the conditions
variables (m n : ℤ)
axiom abs_m_eq_3 : |m| = 3
axiom abs_n_eq_2 : |n| = 2
axiom m_lt_n : m < n

-- State the problem
theorem find_mn_expression : m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end find_mn_expression_l75_7510


namespace count_two_digit_numbers_with_digit_8_l75_7543

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_digit_8 (n : ℕ) : Prop :=
  n / 10 = 8 ∨ n % 10 = 8

theorem count_two_digit_numbers_with_digit_8 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, is_two_digit n ∧ has_digit_8 n) ∧ S.card = 18) :=
sorry

end count_two_digit_numbers_with_digit_8_l75_7543


namespace xiangshan_port_investment_scientific_notation_l75_7545

-- Definition of scientific notation
def in_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

-- Theorem stating the equivalence of the investment in scientific notation
theorem xiangshan_port_investment_scientific_notation :
  in_scientific_notation 7.7 9 7.7e9 :=
by {
  sorry
}

end xiangshan_port_investment_scientific_notation_l75_7545


namespace bottle_caps_given_l75_7504

variable (initial_caps : ℕ) (final_caps : ℕ) (caps_given_by_rebecca : ℕ)

theorem bottle_caps_given (h1: initial_caps = 7) (h2: final_caps = 9) : caps_given_by_rebecca = 2 :=
by
  -- The proof will be filled here
  sorry

end bottle_caps_given_l75_7504


namespace cricketer_average_score_l75_7553

theorem cricketer_average_score
  (avg1 : ℕ)
  (matches1 : ℕ)
  (avg2 : ℕ)
  (matches2 : ℕ)
  (total_matches : ℕ)
  (total_avg : ℕ)
  (h1 : avg1 = 20)
  (h2 : matches1 = 2)
  (h3 : avg2 = 30)
  (h4 : matches2 = 3)
  (h5 : total_matches = 5)
  (h6 : total_avg = 26)
  (h_total_runs : total_avg * total_matches = avg1 * matches1 + avg2 * matches2) :
  total_avg = 26 := 
sorry

end cricketer_average_score_l75_7553


namespace feeding_amount_per_horse_per_feeding_l75_7524

-- Define the conditions as constants
def num_horses : ℕ := 25
def feedings_per_day : ℕ := 2
def half_ton_in_pounds : ℕ := 1000
def bags_needed : ℕ := 60
def days : ℕ := 60

-- Statement of the problem
theorem feeding_amount_per_horse_per_feeding :
  (bags_needed * half_ton_in_pounds / days / feedings_per_day) / num_horses = 20 := by
  -- Assume conditions are satisfied
  sorry

end feeding_amount_per_horse_per_feeding_l75_7524


namespace work_completion_days_l75_7519

noncomputable def A_days : ℝ := 20
noncomputable def B_days : ℝ := 35
noncomputable def C_days : ℝ := 50

noncomputable def A_work_rate : ℝ := 1 / A_days
noncomputable def B_work_rate : ℝ := 1 / B_days
noncomputable def C_work_rate : ℝ := 1 / C_days

noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate + C_work_rate
noncomputable def total_days : ℝ := 1 / combined_work_rate

theorem work_completion_days : total_days = 700 / 69 :=
by
  -- Proof steps would go here
  sorry

end work_completion_days_l75_7519


namespace number_of_cookies_first_friend_took_l75_7537

-- Definitions of given conditions:
def initial_cookies : ℕ := 22
def eaten_by_Kristy : ℕ := 2
def given_to_brother : ℕ := 1
def taken_by_second_friend : ℕ := 5
def taken_by_third_friend : ℕ := 5
def cookies_left : ℕ := 6

noncomputable def cookies_after_Kristy_ate_and_gave_away : ℕ :=
  initial_cookies - eaten_by_Kristy - given_to_brother

noncomputable def cookies_after_second_and_third_friends : ℕ :=
  taken_by_second_friend + taken_by_third_friend

noncomputable def cookies_before_second_and_third_friends_took : ℕ :=
  cookies_left + cookies_after_second_and_third_friends

theorem number_of_cookies_first_friend_took :
  cookies_after_Kristy_ate_and_gave_away - cookies_before_second_and_third_friends_took = 3 := by
  sorry

end number_of_cookies_first_friend_took_l75_7537


namespace winning_candidate_percentage_l75_7523

/-- 
In an election, a candidate won by a majority of 1040 votes out of a total of 5200 votes.
Prove that the winning candidate received 60% of the votes.
-/
theorem winning_candidate_percentage {P : ℝ} (h_majority : (P * 5200) - ((1 - P) * 5200) = 1040) : P = 0.60 := 
by
  sorry

end winning_candidate_percentage_l75_7523


namespace total_ticket_count_is_59_l75_7577

-- Define the constants and variables
def price_adult : ℝ := 4
def price_student : ℝ := 2.5
def total_revenue : ℝ := 222.5
def student_tickets_sold : ℕ := 9

-- Define the equation representing the total revenue and solve for the number of adult tickets
noncomputable def total_tickets_sold (adult_tickets : ℕ) :=
  adult_tickets + student_tickets_sold

theorem total_ticket_count_is_59 (A : ℕ) 
  (h : price_adult * A + price_student * (student_tickets_sold : ℝ) = total_revenue) :
  total_tickets_sold A = 59 :=
by
  sorry

end total_ticket_count_is_59_l75_7577


namespace sheets_per_pack_l75_7540

theorem sheets_per_pack (p d t : Nat) (total_sheets : Nat) (sheets_per_pack : Nat) 
  (h1 : p = 2) (h2 : d = 80) (h3 : t = 6) 
  (h4 : total_sheets = d * t)
  (h5 : sheets_per_pack = total_sheets / p) : sheets_per_pack = 240 := 
  by 
    sorry

end sheets_per_pack_l75_7540


namespace rose_joined_after_six_months_l75_7599

noncomputable def profit_shares (m : ℕ) : ℕ :=
  12000 * (12 - m) - 9000 * 8

theorem rose_joined_after_six_months :
  ∃ (m : ℕ), profit_shares m = 370 :=
by
  use 6
  unfold profit_shares
  norm_num
  sorry

end rose_joined_after_six_months_l75_7599


namespace SamLastPage_l75_7564

theorem SamLastPage (total_pages : ℕ) (Sam_read_time : ℕ) (Lily_read_time : ℕ) (last_page : ℕ) :
  total_pages = 920 ∧ Sam_read_time = 30 ∧ Lily_read_time = 50 → last_page = 575 :=
by
  intros h
  sorry

end SamLastPage_l75_7564


namespace jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l75_7582

theorem jake_first_week_sales :
  let initial_pieces := 80
  let monday_sales := 15
  let tuesday_sales := 2 * monday_sales
  let remaining_pieces := 7
  monday_sales + tuesday_sales + (initial_pieces - (monday_sales + tuesday_sales) - remaining_pieces) = 73 :=
by
  sorry

theorem jake_second_week_sales :
  let monday_sales := 12
  let tuesday_sales := 18
  let wednesday_sales := 20
  let thursday_sales := 11
  let friday_sales := 25
  monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales = 86 :=
by
  sorry

theorem jake_highest_third_week_sales :
  let highest_sales := 40
  highest_sales = 40 :=
by
  sorry

end jake_first_week_sales_jake_second_week_sales_jake_highest_third_week_sales_l75_7582


namespace donuts_Niraek_covers_l75_7502

/- Define the radii of the donut holes -/
def radius_Niraek : ℕ := 5
def radius_Theo : ℕ := 9
def radius_Akshaj : ℕ := 10
def radius_Lily : ℕ := 7

/- Define the surface areas of the donut holes -/
def surface_area (r : ℕ) : ℕ := 4 * r * r

/- Compute the surface areas -/
def sa_Niraek := surface_area radius_Niraek
def sa_Theo := surface_area radius_Theo
def sa_Akshaj := surface_area radius_Akshaj
def sa_Lily := surface_area radius_Lily

/- Define a function to compute the LCM of a list of natural numbers -/
def lcm_of_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

/- Compute the lcm of the surface areas -/
def lcm_surface_areas := lcm_of_list [sa_Niraek, sa_Theo, sa_Akshaj, sa_Lily]

/- Compute the answer -/
def num_donuts_Niraek_covers := lcm_surface_areas / sa_Niraek

/- Prove the statement -/
theorem donuts_Niraek_covers : num_donuts_Niraek_covers = 63504 :=
by
  /- Skipping the proof for now -/
  sorry

end donuts_Niraek_covers_l75_7502


namespace problem1_problem2_l75_7521

-- Problem (1)
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^3 + b^3 >= a*b^2 + a^2*b := 
sorry

-- Problem (2)
theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
sorry

end problem1_problem2_l75_7521


namespace economical_shower_heads_l75_7518

theorem economical_shower_heads (x T : ℕ) (x_pos : 0 < x)
    (students : ℕ := 100)
    (preheat_time_per_shower : ℕ := 3)
    (shower_time_per_group : ℕ := 12) :
  (T = preheat_time_per_shower * x + shower_time_per_group * (students / x)) →
  (students * preheat_time_per_shower + shower_time_per_group * students / x = T) →
  x = 20 := by
  sorry

end economical_shower_heads_l75_7518


namespace train_pass_tree_in_time_l75_7531

-- Definitions from the given conditions
def train_length : ℚ := 270  -- length in meters
def train_speed_km_per_hr : ℚ := 108  -- speed in km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℚ) : ℚ := v * (5 / 18)

-- Speed of the train in m/s
def train_speed_m_per_s : ℚ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Question translated into a proof problem
theorem train_pass_tree_in_time :
  train_length / train_speed_m_per_s = 9 :=
by
  sorry

end train_pass_tree_in_time_l75_7531


namespace total_meters_built_l75_7596

/-- Define the length of the road -/
def road_length (L : ℕ) := L = 1000

/-- Define the average meters built per day -/
def average_meters_per_day (A : ℕ) := A = 120

/-- Define the number of days worked from July 29 to August 2 -/
def number_of_days_worked (D : ℕ) := D = 5

/-- The total meters built by the time they finished -/
theorem total_meters_built
  (L A D : ℕ)
  (h1 : road_length L)
  (h2 : average_meters_per_day A)
  (h3 : number_of_days_worked D)
  : L / D * A = 600 := by
  sorry

end total_meters_built_l75_7596


namespace shortest_distance_between_circles_l75_7573

def circle_eq1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 15 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 + 12*y + 21 = 0

theorem shortest_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), circle_eq1 x1 y1 → circle_eq2 x2 y2 → 
  (abs ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) - (15^(1/2) + 82^(1/2))) =
  2 * 41^(1/2) - 97^(1/2) :=
by sorry

end shortest_distance_between_circles_l75_7573


namespace minimize_expression_l75_7588

theorem minimize_expression (x y z : ℝ) (h : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 2 / 3) :
  x^2 + 6 * x * y + 18 * y^2 + 12 * y * z + 4 * z^2 = 18 :=
sorry

end minimize_expression_l75_7588


namespace six_sin6_cos6_l75_7551

theorem six_sin6_cos6 (A : ℝ) (h : Real.cos (2 * A) = - Real.sqrt 5 / 3) : 
  6 * Real.sin (A) ^ 6 + 6 * Real.cos (A) ^ 6 = 4 := 
sorry

end six_sin6_cos6_l75_7551


namespace fraction_value_l75_7534

theorem fraction_value
  (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 :=
by
  sorry

end fraction_value_l75_7534


namespace will_buy_5_toys_l75_7580

theorem will_buy_5_toys (initial_money spent_money toy_cost money_left toys : ℕ) 
  (h1 : initial_money = 57) 
  (h2 : spent_money = 27) 
  (h3 : toy_cost = 6) 
  (h4 : money_left = initial_money - spent_money) 
  (h5 : toys = money_left / toy_cost) : 
  toys = 5 := 
by
  sorry

end will_buy_5_toys_l75_7580


namespace not_divisible_by_1998_l75_7539

theorem not_divisible_by_1998 (n : ℕ) :
  ∀ k : ℕ, ¬ (2^(k+1) * n + 2^k - 1) % 2 = 0 → ¬ (2^(k+1) * n + 2^k - 1) % 1998 = 0 :=
by
  intros _ _
  sorry

end not_divisible_by_1998_l75_7539


namespace product_of_roots_l75_7581

theorem product_of_roots (x : ℝ) (h : (x - 1) * (x + 4) = 22) : ∃ a b, (x^2 + 3*x - 26 = 0) ∧ a * b = -26 :=
by
  -- Given the equation (x - 1) * (x + 4) = 22,
  -- We want to show that the roots of the equation when simplified are such that
  -- their product is -26.
  sorry

end product_of_roots_l75_7581


namespace find_m_l75_7542

theorem find_m {x1 x2 m : ℝ} 
  (h_eqn : ∀ x, x^2 - (m+3)*x + (m+2) = 0) 
  (h_cond : x1 / (x1 + 1) + x2 / (x2 + 1) = 13 / 10) : 
  m = 2 := 
sorry

end find_m_l75_7542


namespace kim_gets_change_of_5_l75_7587

noncomputable def meal_cost : ℝ := 10
noncomputable def drink_cost : ℝ := 2.5
noncomputable def tip_rate : ℝ := 0.20
noncomputable def payment : ℝ := 20
noncomputable def total_cost_before_tip := meal_cost + drink_cost
noncomputable def tip := tip_rate * total_cost_before_tip
noncomputable def total_cost_with_tip := total_cost_before_tip + tip
noncomputable def change := payment - total_cost_with_tip

theorem kim_gets_change_of_5 : change = 5 := by
  sorry

end kim_gets_change_of_5_l75_7587


namespace mailman_junk_mail_l75_7594

/-- 
  Given:
    - n = 640 : total number of pieces of junk mail for the block
    - h = 20 : number of houses in the block
  
  Prove:
    - The number of pieces of junk mail given to each house equals 32, when the total number of pieces of junk mail is divided by the number of houses.
--/
theorem mailman_junk_mail (n h : ℕ) (h_total : n = 640) (h_houses : h = 20) :
  n / h = 32 :=
by
  sorry

end mailman_junk_mail_l75_7594


namespace slope_intercept_parallel_l75_7544

theorem slope_intercept_parallel (A : ℝ × ℝ) (x y : ℝ) (hA : A = (3, 2))
(hparallel : 4 * x + y - 2 = 0) :
  ∃ b : ℝ, y = -4 * x + b ∧ b = 14 :=
by
  sorry

end slope_intercept_parallel_l75_7544


namespace find_y_l75_7597

noncomputable def imaginary_unit : ℂ := Complex.I

noncomputable def z1 (y : ℝ) : ℂ := 3 + y * imaginary_unit

noncomputable def z2 : ℂ := 2 - imaginary_unit

theorem find_y (y : ℝ) (h : z1 y / z2 = 1 + imaginary_unit) : y = 1 :=
by
  sorry

end find_y_l75_7597


namespace determine_x_y_l75_7558

-- Definitions from the conditions
def cond1 (x y : ℚ) : Prop := 12 * x + 198 = 12 * y + 176
def cond2 (x y : ℚ) : Prop := x + y = 29

-- Statement to prove
theorem determine_x_y : ∃ x y : ℚ, cond1 x y ∧ cond2 x y ∧ x = 163 / 12 ∧ y = 185 / 12 := 
by 
  sorry

end determine_x_y_l75_7558


namespace total_tiles_in_room_l75_7508

theorem total_tiles_in_room (s : ℕ) (hs : 6 * s - 5 = 193) : s^2 = 1089 :=
by sorry

end total_tiles_in_room_l75_7508


namespace variance_cows_l75_7574

-- Define the number of cows and incidence rate.
def n : ℕ := 10
def p : ℝ := 0.02

-- The variance of the binomial distribution, given n and p.
def variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Statement to prove
theorem variance_cows : variance n p = 0.196 :=
by
  sorry

end variance_cows_l75_7574


namespace correct_option_C_l75_7570

variable (x : ℝ)
variable (hx : 0 < x ∧ x < 1)

theorem correct_option_C : 0 < 1 - x^2 ∧ 1 - x^2 < 1 :=
by
  sorry

end correct_option_C_l75_7570


namespace coords_of_P_max_PA_distance_l75_7503

open Real

noncomputable def A : (ℝ × ℝ) := (0, -5)

def on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, x = P.1 ∧ y = P.2 ∧ (x - 2)^2 + (y + 3)^2 = 2

def max_PA_distance (P : (ℝ × ℝ)) : Prop :=
  dist P A = max (dist (3, -2) A) (dist (1, -4) A)

theorem coords_of_P_max_PA_distance (P : (ℝ × ℝ)) :
  on_circle P →
  max_PA_distance P →
  P = (3, -2) :=
  sorry

end coords_of_P_max_PA_distance_l75_7503


namespace find_n_l75_7595

theorem find_n (n : ℤ) (h : Real.sqrt (10 + n) = 9) : n = 71 :=
sorry

end find_n_l75_7595


namespace relationship_between_first_and_third_numbers_l75_7591

variable (A B C : ℕ)

theorem relationship_between_first_and_third_numbers
  (h1 : A + B + C = 660)
  (h2 : A = 2 * B)
  (h3 : B = 180) :
  C = A - 240 :=
by
  sorry

end relationship_between_first_and_third_numbers_l75_7591


namespace b_days_work_alone_l75_7555

theorem b_days_work_alone 
  (W_b : ℝ)  -- Work done by B in one day
  (W_a : ℝ)  -- Work done by A in one day
  (D_b : ℝ)  -- Number of days for B to complete the work alone
  (h1 : W_a = 2 * W_b)  -- A is twice as good a workman as B
  (h2 : 7 * (W_a + W_b) = D_b * W_b)  -- A and B took 7 days together to do the work
  : D_b = 21 :=
sorry

end b_days_work_alone_l75_7555


namespace intersection_of_A_and_B_l75_7515

open Set

def A := {x : ℝ | 2 + x ≥ 4}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := sorry

end intersection_of_A_and_B_l75_7515


namespace average_speed_second_day_l75_7563

theorem average_speed_second_day
  (t v : ℤ)
  (h1 : 2 * t + 2 = 18)
  (h2 : (v + 5) * (t + 2) + v * t = 680) :
  v = 35 :=
by
  sorry

end average_speed_second_day_l75_7563


namespace collinear_points_x_value_l75_7530

theorem collinear_points_x_value :
  (∀ A B C : ℝ × ℝ, A = (-1, 1) → B = (2, -4) → C = (x, -9) → 
                    (∃ x : ℝ, x = 5)) :=
by sorry

end collinear_points_x_value_l75_7530


namespace simplify_and_evaluate_expression_l75_7533

theorem simplify_and_evaluate_expression (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -2) (hx3 : x ≠ 2) :
  ( ( (x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = x - 2 ) ∧ 
  ( (x = 1) → ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = -1 ) :=
by
  sorry

end simplify_and_evaluate_expression_l75_7533


namespace ellipse_hyperbola_proof_l75_7512

noncomputable def ellipse_and_hyperbola_condition (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (a^2 - b^2 = 5) ∧ (a^2 = 11 * b^2)

theorem ellipse_hyperbola_proof : 
  ∀ (a b : ℝ), ellipse_and_hyperbola_condition a b → b^2 = 0.5 :=
by
  intros a b h
  sorry

end ellipse_hyperbola_proof_l75_7512


namespace c_minus_3_eq_neg3_l75_7529

variable (g : ℝ → ℝ)
variable (c : ℝ)

-- defining conditions
axiom invertible_g : Function.Injective g
axiom g_c_eq_3 : g c = 3
axiom g_3_eq_5 : g 3 = 5

-- The goal is to prove that c - 3 = -3
theorem c_minus_3_eq_neg3 : c - 3 = -3 :=
by
  sorry

end c_minus_3_eq_neg3_l75_7529


namespace pyramid_volume_of_unit_cube_l75_7517

noncomputable def volume_of_pyramid : ℝ :=
  let s := (Real.sqrt 2) / 2
  let base_area := (Real.sqrt 3) / 8
  let height := 1
  (1 / 3) * base_area * height

theorem pyramid_volume_of_unit_cube :
  volume_of_pyramid = (Real.sqrt 3) / 24 := by
  sorry

end pyramid_volume_of_unit_cube_l75_7517


namespace index_card_area_l75_7579

theorem index_card_area :
  ∀ (length width : ℕ), length = 5 → width = 7 →
  (length - 2) * width = 21 →
  length * (width - 1) = 30 :=
by
  intros length width h_length h_width h_condition
  sorry

end index_card_area_l75_7579


namespace find_number_of_clerks_l75_7568

-- Define the conditions 
def avg_salary_per_head_staff : ℝ := 90
def avg_salary_officers : ℝ := 600
def avg_salary_clerks : ℝ := 84
def number_of_officers : ℕ := 2

-- Define the variable C (number of clerks)
def number_of_clerks : ℕ := sorry   -- We will prove that this is 170

-- Define the total salary equations based on the conditions
def total_salary_officers := number_of_officers * avg_salary_officers
def total_salary_clerks := number_of_clerks * avg_salary_clerks
def total_number_of_staff := number_of_officers + number_of_clerks
def total_salary := total_salary_officers + total_salary_clerks

-- Define the average salary per head equation 
def avg_salary_eq : Prop := avg_salary_per_head_staff = total_salary / total_number_of_staff

theorem find_number_of_clerks (h : avg_salary_eq) : number_of_clerks = 170 :=
sorry

end find_number_of_clerks_l75_7568


namespace geometric_seq_a8_l75_7546

noncomputable def geometric_seq_term (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

noncomputable def geometric_seq_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_seq_a8
  (a₁ r : ℝ)
  (h1 : geometric_seq_sum a₁ r 3 = 7/4)
  (h2 : geometric_seq_sum a₁ r 6 = 63/4)
  (h3 : r ≠ 1) :
  geometric_seq_term a₁ r 8 = 32 :=
by
  sorry

end geometric_seq_a8_l75_7546


namespace range_of_a_l75_7513

theorem range_of_a (a : ℝ) (x y : ℝ) (hxy : x * y > 0) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + a / y) ≥ 9 → a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l75_7513


namespace additional_savings_if_purchase_together_l75_7585

theorem additional_savings_if_purchase_together :
  let price_per_window := 100
  let windows_each_offer := 4
  let free_each_offer := 1
  let dave_windows := 7
  let doug_windows := 8

  let cost_without_offer (windows : Nat) := windows * price_per_window
  let cost_with_offer (windows : Nat) := 
    if windows % (windows_each_offer + free_each_offer) = 0 then
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window
    else
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window 
      + (windows % (windows_each_offer + free_each_offer)) * price_per_window

  (cost_without_offer (dave_windows + doug_windows) 
  - cost_with_offer (dave_windows + doug_windows)) 
  - ((cost_without_offer dave_windows - cost_with_offer dave_windows)
  + (cost_without_offer doug_windows - cost_with_offer doug_windows)) = price_per_window := 
  sorry

end additional_savings_if_purchase_together_l75_7585


namespace find_B_value_l75_7571

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l75_7571


namespace derivative_at_zero_l75_7593
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem derivative_at_zero : (deriv f 0) = -120 :=
by
  -- The proof is omitted
  sorry

end derivative_at_zero_l75_7593


namespace union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l75_7509

open Set

variables {α : Type*} (A B C : Set α)

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := sorry
theorem inter_comm : A ∩ B = B ∩ A := sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := sorry

-- Idempotence
theorem union_idem : A ∪ A = A := sorry
theorem inter_idem : A ∩ A = A := sorry

-- De Morgan's Laws
theorem de_morgan_union : compl (A ∪ B) = compl A ∩ compl B := sorry
theorem de_morgan_inter : compl (A ∩ B) = compl A ∪ compl B := sorry

end union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l75_7509


namespace rose_bush_cost_correct_l75_7559

-- Definitions of the given conditions
def total_rose_bushes : ℕ := 20
def gardener_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def gardener_cost : ℕ := gardener_rate * gardener_hours_per_day * gardener_days
def soil_cubic_feet : ℕ := 100
def soil_cost_per_cubic_foot : ℕ := 5
def soil_cost : ℕ := soil_cubic_feet * soil_cost_per_cubic_foot
def total_cost : ℕ := 4100

-- Result computed given the conditions
def rose_bush_cost : ℕ := 150

-- The proof goal (statement only, no proof)
theorem rose_bush_cost_correct : 
  total_cost - gardener_cost - soil_cost = total_rose_bushes * rose_bush_cost :=
by
  sorry

end rose_bush_cost_correct_l75_7559


namespace marsha_remainder_l75_7578

-- Definitions based on problem conditions
def a (n : ℤ) : ℤ := 90 * n + 84
def b (m : ℤ) : ℤ := 120 * m + 114
def c (p : ℤ) : ℤ := 150 * p + 144

-- Proof statement
theorem marsha_remainder (n m p : ℤ) : ((a n + b m + c p) % 30) = 12 :=
by 
  -- Notice we need to add the proof steps here
  sorry 

end marsha_remainder_l75_7578


namespace points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l75_7586

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem points_in_first_quadrant (x y : ℝ) (h : x > 0 ∧ y > 0) : first_quadrant x y :=
by {
  sorry
}

theorem points_in_fourth_quadrant (x y : ℝ) (h : x > 0 ∧ y < 0) : fourth_quadrant x y :=
by {
  sorry
}

theorem points_in_second_quadrant (x y : ℝ) (h : x < 0 ∧ y > 0) : second_quadrant x y :=
by {
  sorry
}

theorem points_in_third_quadrant (x y : ℝ) (h : x < 0 ∧ y < 0) : third_quadrant x y :=
by {
  sorry
}

end points_in_first_quadrant_points_in_fourth_quadrant_points_in_second_quadrant_points_in_third_quadrant_l75_7586


namespace cost_price_correct_l75_7538

noncomputable def cost_price (selling_price marked_price_ratio cost_profit_ratio : ℝ) : ℝ :=
  (selling_price * marked_price_ratio) / cost_profit_ratio

theorem cost_price_correct : 
  abs (cost_price 63.16 0.94 1.25 - 50.56) < 0.01 :=
by 
  sorry

end cost_price_correct_l75_7538


namespace leaked_before_fixing_l75_7592

def total_leaked_oil := 6206
def leaked_while_fixing := 3731

theorem leaked_before_fixing :
  total_leaked_oil - leaked_while_fixing = 2475 := by
  sorry

end leaked_before_fixing_l75_7592


namespace hypotenuse_length_l75_7520

theorem hypotenuse_length (x y h : ℝ)
  (hx : (1 / 3) * π * y * x^2 = 1620 * π)
  (hy : (1 / 3) * π * x * y^2 = 3240 * π) :
  h = Real.sqrt 507 :=
by
  sorry

end hypotenuse_length_l75_7520


namespace problem1_problem2_l75_7569

-- Definitions and assumptions
def p (m : ℝ) : Prop := ∀x y : ℝ, (x^2)/(4 - m) + (y^2)/m = 1 → ∃ c : ℝ, c^2 < (4 - m) ∧ c^2 < m
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ m ≥ 1 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l75_7569


namespace minimum_value_f_range_a_l75_7576

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem minimum_value_f :
  ∃ x : ℝ, f x = -(1 / Real.exp 1) :=
sorry

theorem range_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ a ∈ Set.Iic 1 :=
sorry

end minimum_value_f_range_a_l75_7576


namespace find_multiple_of_q_l75_7567

variable (p q m : ℚ)

theorem find_multiple_of_q (h1 : p / q = 3 / 4) (h2 : 3 * p + m * q = 6.25) :
  m = 4 :=
sorry

end find_multiple_of_q_l75_7567


namespace total_students_is_100_l75_7525

-- Definitions of the conditions
def largest_class_students : Nat := 24
def decrement : Nat := 2

-- Let n be the number of classes, which is given by 5
def num_classes : Nat := 5

-- The number of students in each class
def students_in_class (n : Nat) : Nat := 
  if n = 1 then largest_class_students
  else largest_class_students - decrement * (n - 1)

-- Total number of students in the school
def total_students : Nat :=
  List.sum (List.map students_in_class (List.range num_classes))

-- Theorem to prove that total_students equals 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end total_students_is_100_l75_7525


namespace maximum_median_soda_shop_l75_7501

noncomputable def soda_shop_median (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) : ℝ :=
  if total_cans = 300 ∧ total_customers = 120 ∧ min_cans_per_customer = 1 then 3.5 else sorry

theorem maximum_median_soda_shop : soda_shop_median 300 120 1 = 3.5 :=
by
  sorry

end maximum_median_soda_shop_l75_7501


namespace regular_polygon_sides_l75_7549

theorem regular_polygon_sides (n : ℕ) (h₁ : 1 < n)
  (interior_angle_sum : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end regular_polygon_sides_l75_7549


namespace cash_price_of_television_l75_7526

variable (DownPayment : ℕ := 120)
variable (MonthlyPayment : ℕ := 30)
variable (NumberOfMonths : ℕ := 12)
variable (Savings : ℕ := 80)

-- Define the total installment cost
def TotalInstallment := DownPayment + MonthlyPayment * NumberOfMonths

-- The main statement to prove
theorem cash_price_of_television : (TotalInstallment - Savings) = 400 := by
  sorry

end cash_price_of_television_l75_7526


namespace project_selection_l75_7505

noncomputable def binomial : ℕ → ℕ → ℕ 
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binomial n k + binomial n (k+1)

theorem project_selection :
  (binomial 5 2 * binomial 3 2) + (binomial 3 1 * binomial 5 1) = 45 := 
sorry

end project_selection_l75_7505


namespace tan_alpha_eq_neg_one_l75_7554

theorem tan_alpha_eq_neg_one (alpha : ℝ) (h1 : Real.tan alpha = -1) (h2 : 0 ≤ alpha ∧ alpha < Real.pi) :
  alpha = (3 * Real.pi) / 4 :=
sorry

end tan_alpha_eq_neg_one_l75_7554


namespace mother_age_4times_daughter_l75_7583

-- Conditions
def Y := 12
def M := 42

-- Proof statement: Prove that 2 years ago, the mother's age was 4 times Yujeong's age.
theorem mother_age_4times_daughter (X : ℕ) (hY : Y = 12) (hM : M = 42) : (42 - X) = 4 * (12 - X) :=
by
  intros
  sorry

end mother_age_4times_daughter_l75_7583


namespace scientific_notation_of_population_l75_7532

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l75_7532


namespace units_digit_of_n_l75_7516

theorem units_digit_of_n
  (m n : ℕ)
  (h1 : m * n = 23^7)
  (h2 : m % 10 = 9) : n % 10 = 3 :=
sorry

end units_digit_of_n_l75_7516


namespace train_crossing_time_l75_7541

-- Define the problem conditions in Lean 4
def train_length : ℕ := 130
def bridge_length : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := (speed_kmph * 1000 / 3600)

-- The statement to prove
theorem train_crossing_time : (train_length + bridge_length) / speed_mps = 28 :=
by
  -- The proof starts here
  sorry

end train_crossing_time_l75_7541


namespace exists_inequality_l75_7575

theorem exists_inequality (n : ℕ) (x : Fin (n + 1) → ℝ) 
  (hx1 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) 
  (h_n : 2 ≤ n) : 
  ∃ i : Fin n, x i * (1 - x (i + 1)) ≥ (1 / 4) * x 0 * (1 - x n) :=
sorry

end exists_inequality_l75_7575


namespace molecular_weight_calculation_l75_7556

-- Define the atomic weights of each element
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms of each element in the compound
def num_atoms_C : ℕ := 7
def num_atoms_H : ℕ := 6
def num_atoms_O : ℕ := 2

-- The molecular weight calculation
def molecular_weight : ℝ :=
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_H * atomic_weight_H) +
  (num_atoms_O * atomic_weight_O)

theorem molecular_weight_calculation : molecular_weight = 122.118 :=
by
  -- Proof
  sorry

end molecular_weight_calculation_l75_7556


namespace annual_avg_growth_rate_export_volume_2023_l75_7561

variable (V0 V2 V3 : ℕ) (r : ℝ)
variable (h1 : V0 = 200000) (h2 : V2 = 450000) (h3 : V3 = 675000)

-- Definition of the exponential growth equation
def growth_exponential (V0 Vn: ℕ) (n : ℕ) (r : ℝ) : Prop :=
  Vn = V0 * ((1 + r) ^ n)

-- The Lean statement to prove the annual average growth rate
theorem annual_avg_growth_rate (x : ℝ) (h : growth_exponential V0 V2 2 x) : 
  x = 0.5 :=
by
  sorry

-- The Lean statement to prove the export volume in 2023
theorem export_volume_2023 (h_growth : growth_exponential V2 V3 1 0.5) :
  V3 = 675000 :=
by
  sorry

end annual_avg_growth_rate_export_volume_2023_l75_7561


namespace problem1_problem2_problem3_l75_7547

noncomputable def a_n (n : ℕ) : ℕ := 3 * (2 ^ n) - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * a_n n - 3 * n

-- 1. Prove a_1 = 3 and a_2 = 9 given S_n = 2a_n - 3n
theorem problem1 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    a_n 1 = 3 ∧ a_n 2 = 9 :=
  sorry

-- 2. Prove that the sequence {a_n + 3} is a geometric sequence and find the general term formula for the sequence {a_n}.
theorem problem2 (n : ℕ) (h : ∀ n > 0, S_n n = 2 * (a_n n) - 3 * n) :
    ∀ n, (a_n (n + 1) + 3) / (a_n n + 3) = 2 ∧ a_n n = 3 * (2 ^ n) - 3 :=
  sorry

-- 3. Prove {S_{n_k}} is not an arithmetic sequence given S_n = 2a_n - 3n and {n_k} is an arithmetic sequence
theorem problem3 (n_k : ℕ → ℕ) (h_arithmetic : ∃ d, ∀ k, n_k (k + 1) - n_k k = d) :
    ¬ ∃ d, ∀ k, S_n (n_k (k + 1)) - S_n (n_k k) = d :=
  sorry

end problem1_problem2_problem3_l75_7547


namespace remainder_when_expression_divided_l75_7562

theorem remainder_when_expression_divided 
  (x y u v : ℕ) 
  (h1 : x = u * y + v) 
  (h2 : 0 ≤ v) 
  (h3 : v < y) :
  (x - u * y + 3 * v) % y = (4 * v) % y :=
by
  sorry

end remainder_when_expression_divided_l75_7562


namespace tom_is_15_l75_7572

theorem tom_is_15 (T M : ℕ) (h1 : T + M = 21) (h2 : T + 3 = 2 * (M + 3)) : T = 15 :=
by {
  sorry
}

end tom_is_15_l75_7572


namespace Tina_profit_l75_7500

variables (x : ℝ) (profit_per_book : ℝ) (number_of_people : ℕ) (cost_per_book : ℝ)
           (books_per_customer : ℕ) (total_profit : ℝ) (total_cost : ℝ) (total_books_sold : ℕ)

theorem Tina_profit :
  (number_of_people = 4) →
  (cost_per_book = 5) →
  (books_per_customer = 2) →
  (total_profit = 120) →
  (books_per_customer * number_of_people = total_books_sold) →
  (cost_per_book * total_books_sold = total_cost) →
  (total_profit = total_books_sold * x - total_cost) →
  x = 20 :=
by
  intros
  sorry


end Tina_profit_l75_7500


namespace proof_strictly_increasing_sequence_l75_7506

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence_l75_7506
