import Mathlib

namespace subtraction_of_decimals_l2354_235426

theorem subtraction_of_decimals : 58.3 - 0.45 = 57.85 := by
  sorry

end subtraction_of_decimals_l2354_235426


namespace Greg_PPO_Obtained_90_Percent_l2354_235485

theorem Greg_PPO_Obtained_90_Percent :
  let max_procgen_reward := 240
  let max_coinrun_reward := max_procgen_reward / 2
  let greg_reward := 108
  (greg_reward / max_coinrun_reward * 100) = 90 := by
  sorry

end Greg_PPO_Obtained_90_Percent_l2354_235485


namespace simplify_expression_l2354_235473

variable (a b : ℝ)

theorem simplify_expression (a b : ℝ) :
  (6 * a^5 * b^2) / (3 * a^3 * b^2) + ((2 * a * b^3)^2) / ((-b^2)^3) = -2 * a^2 :=
by 
  sorry

end simplify_expression_l2354_235473


namespace negation_of_exists_leq_zero_l2354_235416

theorem negation_of_exists_leq_zero (x : ℝ) : ¬(∃ x ≥ 1, 2^x ≤ 0) ↔ ∀ x ≥ 1, 2^x > 0 :=
by
  sorry

end negation_of_exists_leq_zero_l2354_235416


namespace problem_statement_l2354_235450

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end problem_statement_l2354_235450


namespace length_PR_l2354_235488

noncomputable def circle_radius : ℝ := 10
noncomputable def distance_PQ : ℝ := 12
noncomputable def midpoint_minor_arc_length_PR : ℝ :=
  let PS : ℝ := distance_PQ / 2
  let OS : ℝ := Real.sqrt (circle_radius^2 - PS^2)
  let RS : ℝ := circle_radius - OS
  Real.sqrt (PS^2 + RS^2)

theorem length_PR :
  midpoint_minor_arc_length_PR = 2 * Real.sqrt 10 :=
by
  sorry

end length_PR_l2354_235488


namespace vincent_total_packs_l2354_235419

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l2354_235419


namespace sqrt_ratio_simplify_l2354_235420

theorem sqrt_ratio_simplify :
  ( (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 ) :=
by
  let sqrt27 := Real.sqrt 27
  let sqrt243 := Real.sqrt 243
  let sqrt75 := Real.sqrt 75
  have h_sqrt27 : sqrt27 = Real.sqrt (3^2 * 3) := by sorry
  have h_sqrt243 : sqrt243 = Real.sqrt (3^5) := by sorry
  have h_sqrt75 : sqrt75 = Real.sqrt (3 * 5^2) := by sorry
  have h_simplified : (sqrt27 + sqrt243) / sqrt75 = 12 / 5 := by sorry
  exact h_simplified

end sqrt_ratio_simplify_l2354_235420


namespace baking_time_correct_l2354_235476

/-- Mark lets the bread rise for 120 minutes twice. -/
def rising_time : ℕ := 120 * 2

/-- Mark spends 10 minutes kneading the bread. -/
def kneading_time : ℕ := 10

/-- Total time taken to finish making the bread. -/
def total_time : ℕ := 280

/-- Calculate the baking time based on the given conditions. -/
def baking_time (rising kneading total : ℕ) : ℕ := total - (rising + kneading)

theorem baking_time_correct :
  baking_time rising_time kneading_time total_time = 30 := 
by 
  -- Proof is omitted
  sorry

end baking_time_correct_l2354_235476


namespace light_ray_total_distance_l2354_235412

theorem light_ray_total_distance 
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (M_eq : M = (2, 1))
  (N_eq : N = (4, 5)) :
  dist M N = 2 * Real.sqrt 10 := 
sorry

end light_ray_total_distance_l2354_235412


namespace x_gt_zero_sufficient_but_not_necessary_l2354_235466

theorem x_gt_zero_sufficient_but_not_necessary (x : ℝ): 
  (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬ (x > 0)) → 
  ((x > 0 ↔ x ≠ 0) = false) :=
by
  intro h
  sorry

end x_gt_zero_sufficient_but_not_necessary_l2354_235466


namespace average_distance_per_day_l2354_235449

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_distance_per_day :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 :=
by
  sorry

end average_distance_per_day_l2354_235449


namespace shipment_cost_l2354_235418

-- Define the conditions
def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def shipping_cost_per_crate : ℝ := 1.5
def surcharge_per_crate : ℝ := 0.5
def flat_fee : ℝ := 10

-- Define the question as a theorem
theorem shipment_cost : 
  let crates := total_weight / weight_per_crate
  let cost_per_crate := shipping_cost_per_crate + surcharge_per_crate
  let total_cost_crates := crates * cost_per_crate
  let total_cost := total_cost_crates + flat_fee
  total_cost = 46 := by
  -- Proof omitted
  sorry

end shipment_cost_l2354_235418


namespace modulus_of_complex_l2354_235430

open Complex

theorem modulus_of_complex : ∀ (z : ℂ), z = 3 - 2 * I → Complex.abs z = Real.sqrt 13 :=
by
  intro z
  intro h
  rw [h]
  simp [Complex.abs]
  sorry

end modulus_of_complex_l2354_235430


namespace positive_integers_N_segment_condition_l2354_235452

theorem positive_integers_N_segment_condition (N : ℕ) (x : ℕ) (n : ℕ)
  (h1 : 10 ≤ N ∧ N ≤ 10^20)
  (h2 : N = x * (10^n - 1) / 9) (h3 : 1 ≤ n ∧ n ≤ 20) : 
  N + 1 = (x + 1) * (9 + 1)^n ∧ x < 10 :=
by {
  sorry
}

end positive_integers_N_segment_condition_l2354_235452


namespace difference_in_squares_l2354_235439

noncomputable def radius_of_circle (x y h R : ℝ) : Prop :=
  5 * x^2 - 4 * x * h + h^2 = R^2 ∧ 5 * y^2 + 4 * y * h + h^2 = R^2

theorem difference_in_squares (x y h R : ℝ) (h_radius : radius_of_circle x y h R) :
  2 * x - 2 * y = (8/5 : ℝ) * h :=
by
  sorry

end difference_in_squares_l2354_235439


namespace solution_for_4_minus_c_l2354_235499

-- Define the conditions as Lean hypotheses
theorem solution_for_4_minus_c (c d : ℚ) (h1 : 4 + c = 5 - d) (h2 : 5 + d = 9 + c) : 4 - c = 11 / 2 :=
by
  sorry

end solution_for_4_minus_c_l2354_235499


namespace number_of_cakes_l2354_235443

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l2354_235443


namespace binom_18_6_mul_smallest_prime_gt_10_eq_80080_l2354_235425

theorem binom_18_6_mul_smallest_prime_gt_10_eq_80080 :
  (Nat.choose 18 6) * 11 = 80080 := sorry

end binom_18_6_mul_smallest_prime_gt_10_eq_80080_l2354_235425


namespace find_a_l2354_235428

variable (a x : ℝ)

noncomputable def curve1 (x : ℝ) := x + Real.log x
noncomputable def curve2 (a x : ℝ) := a * x^2 + (a + 2) * x + 1

theorem find_a : (curve1 1 = 1 ∧ curve1 1 = curve2 a 1) → a = 8 :=
by
  sorry

end find_a_l2354_235428


namespace range_of_M_l2354_235400

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
    ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) )  ≥ 8 := 
  sorry

end range_of_M_l2354_235400


namespace find_ordered_pair_l2354_235417

theorem find_ordered_pair :
  ∃ x y : ℚ, 
  (x + 2 * y = (7 - x) + (7 - 2 * y)) ∧
  (3 * x - 2 * y = (x + 2) - (2 * y + 2)) ∧
  x = 0 ∧ 
  y = 7 / 2 :=
by
  sorry

end find_ordered_pair_l2354_235417


namespace min_value_y_l2354_235481

theorem min_value_y (x y : ℝ) (h : x^2 + y^2 = 14 * x + 48 * y) : y = -1 := 
sorry

end min_value_y_l2354_235481


namespace remainder_8347_div_9_l2354_235440
-- Import all necessary Mathlib modules

-- Define the problem and conditions
theorem remainder_8347_div_9 : (8347 % 9) = 4 :=
by
  -- To ensure the code builds successfully and contains a placeholder for the proof
  sorry

end remainder_8347_div_9_l2354_235440


namespace geometric_sequence_a3_l2354_235445

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 
  (a : ℕ → ℝ) (h1 : a 1 = -2) (h5 : a 5 = -8)
  (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 3 = -4 :=
sorry

end geometric_sequence_a3_l2354_235445


namespace proportion_equivalence_l2354_235446

variable {x y : ℝ}

theorem proportion_equivalence (h : 3 * x = 5 * y) (hy : y ≠ 0) : 
  x / 5 = y / 3 :=
by
  -- Proof goes here
  sorry

end proportion_equivalence_l2354_235446


namespace problem_21_sum_correct_l2354_235409

theorem problem_21_sum_correct (A B C D E : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
    (h_eq : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 21 :=
sorry

end problem_21_sum_correct_l2354_235409


namespace initial_cookies_l2354_235422

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end initial_cookies_l2354_235422


namespace books_fill_shelf_l2354_235423

theorem books_fill_shelf
  (A H S M E : ℕ)
  (h1 : A ≠ H) (h2 : S ≠ M) (h3 : M ≠ H) (h4 : E > 0)
  (Eq1 : A > 0) (Eq2 : H > 0) (Eq3 : S > 0) (Eq4 : M > 0)
  (h5 : A ≠ S) (h6 : E ≠ A) (h7 : E ≠ H) (h8 : E ≠ S) (h9 : E ≠ M) :
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end books_fill_shelf_l2354_235423


namespace positive_integer_root_k_l2354_235421

theorem positive_integer_root_k (k : ℕ) :
  (∃ x : ℕ, x > 0 ∧ x * x - 34 * x + 34 * k - 1 = 0) ↔ k = 1 :=
by
  sorry

end positive_integer_root_k_l2354_235421


namespace john_spent_l2354_235472

/-- John bought 9.25 meters of cloth at a cost price of $44 per meter.
    Prove that the total amount John spent on the cloth is $407. -/
theorem john_spent :
  let length_of_cloth := 9.25
  let cost_per_meter := 44
  let total_cost := length_of_cloth * cost_per_meter
  total_cost = 407 := by
  sorry

end john_spent_l2354_235472


namespace largest_multiple_of_9_less_than_75_is_72_l2354_235482

theorem largest_multiple_of_9_less_than_75_is_72 : 
  ∃ n : ℕ, 9 * n < 75 ∧ ∀ m : ℕ, 9 * m < 75 → 9 * m ≤ 9 * n :=
sorry

end largest_multiple_of_9_less_than_75_is_72_l2354_235482


namespace train_crossing_pole_time_l2354_235434

theorem train_crossing_pole_time :
  ∀ (length_of_train : ℝ) (speed_km_per_hr : ℝ) (t : ℝ),
    length_of_train = 45 →
    speed_km_per_hr = 108 →
    t = 1.5 →
    t = length_of_train / (speed_km_per_hr * 1000 / 3600) := 
  sorry

end train_crossing_pole_time_l2354_235434


namespace colleen_paid_more_l2354_235451

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l2354_235451


namespace max_min_sum_l2354_235483

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log (x + 1) / Real.log 2

theorem max_min_sum : 
  (f 0 + f 1) = 4 := 
by
  sorry

end max_min_sum_l2354_235483


namespace overall_profit_is_600_l2354_235453

def grinder_cp := 15000
def mobile_cp := 10000
def laptop_cp := 20000
def camera_cp := 12000

def grinder_loss_percent := 4 / 100
def mobile_profit_percent := 10 / 100
def laptop_loss_percent := 8 / 100
def camera_profit_percent := 15 / 100

def grinder_sp := grinder_cp * (1 - grinder_loss_percent)
def mobile_sp := mobile_cp * (1 + mobile_profit_percent)
def laptop_sp := laptop_cp * (1 - laptop_loss_percent)
def camera_sp := camera_cp * (1 + camera_profit_percent)

def total_cp := grinder_cp + mobile_cp + laptop_cp + camera_cp
def total_sp := grinder_sp + mobile_sp + laptop_sp + camera_sp

def overall_profit_or_loss := total_sp - total_cp

theorem overall_profit_is_600 : overall_profit_or_loss = 600 := by
  sorry

end overall_profit_is_600_l2354_235453


namespace rob_nickels_count_l2354_235480

noncomputable def value_of_quarters (num_quarters : ℕ) : ℝ := num_quarters * 0.25
noncomputable def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10
noncomputable def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
noncomputable def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05

theorem rob_nickels_count :
  let quarters := 7
  let dimes := 3
  let pennies := 12
  let total := 2.42
  let nickels := 5
  value_of_quarters quarters + value_of_dimes dimes + value_of_pennies pennies + value_of_nickels nickels = total :=
by
  sorry

end rob_nickels_count_l2354_235480


namespace chess_tournament_games_l2354_235477

theorem chess_tournament_games (n : ℕ) (h : n = 17) (k : n - 1 = 16) :
  (n * (n - 1)) / 2 = 136 := by
  sorry

end chess_tournament_games_l2354_235477


namespace gas_cost_is_4_l2354_235402

theorem gas_cost_is_4
    (mileage_rate : ℝ)
    (truck_efficiency : ℝ)
    (profit : ℝ)
    (trip_distance : ℝ)
    (trip_cost : ℝ)
    (gallons_used : ℝ)
    (cost_per_gallon : ℝ) :
  mileage_rate = 0.5 →
  truck_efficiency = 20 →
  profit = 180 →
  trip_distance = 600 →
  trip_cost = mileage_rate * trip_distance - profit →
  gallons_used = trip_distance / truck_efficiency →
  cost_per_gallon = trip_cost / gallons_used →
  cost_per_gallon = 4 :=
by
  sorry

end gas_cost_is_4_l2354_235402


namespace calculation_correct_l2354_235405

def calculation : ℝ := 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123

theorem calculation_correct : calculation = 172.20 := by
  sorry

end calculation_correct_l2354_235405


namespace correct_point_on_hyperbola_l2354_235497

-- Given condition
def hyperbola_condition (x y : ℝ) : Prop := x * y = -4

-- Question (translated to a mathematically equivalent proof)
theorem correct_point_on_hyperbola :
  hyperbola_condition (-2) 2 :=
sorry

end correct_point_on_hyperbola_l2354_235497


namespace fixed_point_l2354_235471

noncomputable def f (a : ℝ) (x : ℝ) := a^(x - 2) - 3

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = -2 :=
by
  sorry

end fixed_point_l2354_235471


namespace find_diameter_endpoint_l2354_235404

def circle_center : ℝ × ℝ := (4, 1)
def diameter_endpoint_1 : ℝ × ℝ := (1, 5)

theorem find_diameter_endpoint :
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  (2 * h - x1, 2 * k - y1) = (7, -3) :=
by
  let (h, k) := circle_center
  let (x1, y1) := diameter_endpoint_1
  sorry

end find_diameter_endpoint_l2354_235404


namespace xiao_ming_reading_plan_l2354_235438

-- Define the number of pages in the book
def total_pages : Nat := 72

-- Define the total number of days to finish the book
def total_days : Nat := 10

-- Define the number of pages read per day for the first two days
def pages_first_two_days : Nat := 5

-- Define the variable x to represent the number of pages read per day for the remaining days
variable (x : Nat)

-- Define the inequality representing the reading plan
def reading_inequality (x : Nat) : Prop :=
  10 + 8 * x ≥ total_pages

-- The statement to be proved
theorem xiao_ming_reading_plan (x : Nat) : reading_inequality x := sorry

end xiao_ming_reading_plan_l2354_235438


namespace total_parts_in_order_l2354_235415

theorem total_parts_in_order (total_cost : ℕ) (cost_20 : ℕ) (cost_50 : ℕ) (num_50_dollar_parts : ℕ) (num_20_dollar_parts : ℕ) :
  total_cost = 2380 → cost_20 = 20 → cost_50 = 50 → num_50_dollar_parts = 40 → (total_cost = num_50_dollar_parts * cost_50 + num_20_dollar_parts * cost_20) → (num_50_dollar_parts + num_20_dollar_parts = 59) :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_parts_in_order_l2354_235415


namespace count_arithmetic_sequence_l2354_235432

theorem count_arithmetic_sequence :
  let a1 := 2.5
  let an := 68.5
  let d := 6.0
  let offset := 0.5
  let adjusted_a1 := a1 + offset
  let adjusted_an := an + offset
  let n := (adjusted_an - adjusted_a1) / d + 1
  n = 12 :=
by {
  sorry
}

end count_arithmetic_sequence_l2354_235432


namespace pos_numbers_equal_l2354_235495

theorem pos_numbers_equal (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eq : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end pos_numbers_equal_l2354_235495


namespace initial_people_employed_l2354_235462

-- Definitions from the conditions
def initial_work_days : ℕ := 25
def total_work_days : ℕ := 50
def work_done_percentage : ℕ := 40
def additional_people : ℕ := 30

-- Defining the statement to be proved
theorem initial_people_employed (P : ℕ) 
  (h1 : initial_work_days = 25) 
  (h2 : total_work_days = 50)
  (h3 : work_done_percentage = 40)
  (h4 : additional_people = 30) 
  (work_remaining_percentage := 60) : 
  (P * 25 / 10 = 100) -> (P + 30) * 50 = P * 625 / 10 -> P = 120 :=
by
  sorry

end initial_people_employed_l2354_235462


namespace inequality_2_inequality_1_9_l2354_235424

variables {a : ℕ → ℝ}

-- Conditions
def non_negative (a : ℕ → ℝ) : Prop := ∀ n, a n ≥ 0
def boundary_zero (a : ℕ → ℝ) : Prop := a 1 = 0 ∧ a 9 = 0
def non_zero_interior (a : ℕ → ℝ) : Prop := ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i ≠ 0

-- Proof problems
theorem inequality_2 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 2 * a i := sorry

theorem inequality_1_9 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 1.9 * a i := sorry

end inequality_2_inequality_1_9_l2354_235424


namespace extreme_values_l2354_235441

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 3

theorem extreme_values (a b : ℝ) : 
  (f a b (-1) = 10) ∧ (f a b 2 = -17) →
  (6 * (-1)^2 + 2 * a * (-1) + b = 0) ∧ (6 * 2^2 + 2 * (a * 2) + b = 0) →
  a = -3 ∧ b = -12 :=
by 
  sorry

end extreme_values_l2354_235441


namespace estimate_total_balls_l2354_235436

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l2354_235436


namespace max_right_angles_in_triangle_l2354_235490

theorem max_right_angles_in_triangle (a b c : ℝ) (h : a + b + c = 180) (ha : a = 90 ∨ b = 90 ∨ c = 90) : a = 90 ∧ b ≠ 90 ∧ c ≠ 90 ∨ b = 90 ∧ a ≠ 90 ∧ c ≠ 90 ∨ c = 90 ∧ a ≠ 90 ∧ b ≠ 90 :=
sorry

end max_right_angles_in_triangle_l2354_235490


namespace smallest_solution_for_quartic_eq_l2354_235459

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l2354_235459


namespace value_of_4_Y_3_eq_neg23_l2354_235493

def my_operation (a b : ℝ) (c : ℝ) : ℝ := a^2 - 2 * a * b * c + b^2

theorem value_of_4_Y_3_eq_neg23 : my_operation 4 3 2 = -23 := by
  sorry

end value_of_4_Y_3_eq_neg23_l2354_235493


namespace sam_gave_fraction_l2354_235474

/-- Given that Mary bought 1500 stickers and shared them between Susan, Andrew, 
and Sam in the ratio 1:1:3. After Sam gave some stickers to Andrew, Andrew now 
has 900 stickers. Prove that the fraction of Sam's stickers given to Andrew is 2/3. -/
theorem sam_gave_fraction (total_stickers : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
    (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (final_B : ℕ) (given_stickers : ℕ) :
    total_stickers = 1500 → ratio_A = 1 → ratio_B = 1 → ratio_C = 3 →
    initial_A = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_B = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_C = 3 * (total_stickers / (ratio_A + ratio_B + ratio_C)) →
    final_B = 900 →
    initial_B + given_stickers = final_B →
    given_stickers / initial_C = 2 / 3 :=
by
  intros
  sorry

end sam_gave_fraction_l2354_235474


namespace manager_salary_4200_l2354_235407

theorem manager_salary_4200
    (avg_salary_employees : ℕ → ℕ → ℕ) 
    (total_salary_employees : ℕ → ℕ → ℕ)
    (new_avg_salary : ℕ → ℕ → ℕ)
    (total_salary_with_manager : ℕ → ℕ → ℕ) 
    (n_employees : ℕ)
    (employee_salary : ℕ) 
    (n_total : ℕ)
    (total_salary_before : ℕ)
    (avg_increase : ℕ)
    (new_employee_salary : ℕ) 
    (total_salary_after : ℕ) 
    (manager_salary : ℕ) :
    n_employees = 15 →
    employee_salary = 1800 →
    avg_increase = 150 →
    avg_salary_employees n_employees employee_salary = 1800 →
    total_salary_employees n_employees employee_salary = 27000 →
    new_avg_salary employee_salary avg_increase = 1950 →
    new_employee_salary = 1950 →
    total_salary_with_manager (n_employees + 1) new_employee_salary = 31200 →
    total_salary_before = 27000 →
    total_salary_after = 31200 →
    manager_salary = total_salary_after - total_salary_before →
    manager_salary = 4200 := 
by 
  intros 
  sorry

end manager_salary_4200_l2354_235407


namespace percentage_increase_in_side_of_square_l2354_235489

theorem percentage_increase_in_side_of_square (p : ℝ) : 
  (1 + p / 100) ^ 2 = 1.3225 → 
  p = 15 :=
by
  sorry

end percentage_increase_in_side_of_square_l2354_235489


namespace discount_is_five_l2354_235469
-- Importing the needed Lean Math library

-- Defining the problem conditions
def costPrice : ℝ := 100
def profit_percent_with_discount : ℝ := 0.2
def profit_percent_without_discount : ℝ := 0.25

-- Calculating the respective selling prices
def sellingPrice_with_discount := costPrice * (1 + profit_percent_with_discount)
def sellingPrice_without_discount := costPrice * (1 + profit_percent_without_discount)

-- Calculating the discount 
def calculated_discount := sellingPrice_without_discount - sellingPrice_with_discount

-- Proving that the discount is $5
theorem discount_is_five : calculated_discount = 5 := by
  -- Proof omitted
  sorry

end discount_is_five_l2354_235469


namespace tens_digit_of_23_pow_2023_l2354_235461

theorem tens_digit_of_23_pow_2023 : (23 ^ 2023 % 100 / 10) = 6 :=
by
  sorry

end tens_digit_of_23_pow_2023_l2354_235461


namespace triceratops_count_l2354_235457

theorem triceratops_count (r t : ℕ) 
  (h_legs : 4 * r + 4 * t = 48) 
  (h_horns : 2 * r + 3 * t = 31) : 
  t = 7 := 
by 
  hint

/- The given conditions are:
1. Each rhinoceros has 2 horns.
2. Each triceratops has 3 horns.
3. Each animal has 4 legs.
4. There is a total of 31 horns.
5. There is a total of 48 legs.

Using these conditions and the equations derived from them, we need to prove that the number of triceratopses (t) is 7.
-/

end triceratops_count_l2354_235457


namespace max_arithmetic_sequence_of_primes_less_than_150_l2354_235475

theorem max_arithmetic_sequence_of_primes_less_than_150 : 
  ∀ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x) ∧ (∀ x ∈ S, x < 150) ∧ (∃ d, ∀ x ∈ S, ∃ n : ℕ, x = S.min' (by sorry) + n * d) → S.card ≤ 5 := 
by
  sorry

end max_arithmetic_sequence_of_primes_less_than_150_l2354_235475


namespace perpendicular_condition_parallel_condition_opposite_direction_l2354_235460

/-- Conditions definitions --/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def k_vector_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
def vector_a_minus_3b : ℝ × ℝ := (10, -4)

/-- Problem 1: Prove the perpendicular condition --/
theorem perpendicular_condition (k : ℝ) : (k_vector_a_plus_b k).fst * vector_a_minus_3b.fst + (k_vector_a_plus_b k).snd * vector_a_minus_3b.snd = 0 → k = 19 :=
by
  sorry

/-- Problem 2: Prove the parallel condition --/
theorem parallel_condition (k : ℝ) : (-(k - 3) / 10 = (2 * k + 2) / (-4)) → k = -1/3 :=
by
  sorry

/-- Determine if the vectors are in opposite directions --/
theorem opposite_direction (k : ℝ) (hk : k = -1/3) : k_vector_a_plus_b k = (-(1/3):ℝ) • vector_a_minus_3b :=
by
  sorry

end perpendicular_condition_parallel_condition_opposite_direction_l2354_235460


namespace smallest_k_for_bisectors_l2354_235487

theorem smallest_k_for_bisectors (a b c l_a l_b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c))
  (h5 : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)) :
  (l_a + l_b) / (a + b) ≤ 4 / 3 :=
by
  sorry

end smallest_k_for_bisectors_l2354_235487


namespace solution_is_D_l2354_235491

-- Definitions of the equations
def eqA (x : ℝ) := 3 * x + 6 = 0
def eqB (x : ℝ) := 2 * x + 4 = 0
def eqC (x : ℝ) := (1 / 2) * x = -4
def eqD (x : ℝ) := 2 * x - 4 = 0

-- Theorem stating that only eqD has a solution x = 2
theorem solution_is_D : 
  ¬ eqA 2 ∧ ¬ eqB 2 ∧ ¬ eqC 2 ∧ eqD 2 := 
by
  sorry

end solution_is_D_l2354_235491


namespace team_e_speed_l2354_235431

-- Definitions and conditions
variables (v t : ℝ)
def distance_team_e := 300 = v * t
def distance_team_a := 300 = (v + 5) * (t - 3)

-- The theorem statement: Prove that given the conditions, Team E's speed is 20 mph
theorem team_e_speed (h1 : distance_team_e v t) (h2 : distance_team_a v t) : v = 20 :=
by
  sorry -- proof steps are omitted as requested

end team_e_speed_l2354_235431


namespace solutions_count_l2354_235468

noncomputable def number_of_solutions (x y z : ℚ) : ℕ :=
if (x^2 - y * z = 1) ∧ (y^2 - x * z = 1) ∧ (z^2 - x * y = 1)
then 6
else 0

theorem solutions_count : number_of_solutions x y z = 6 :=
sorry

end solutions_count_l2354_235468


namespace op_7_3_eq_70_l2354_235429

noncomputable def op (x y : ℝ) : ℝ := sorry

axiom ax1 : ∀ x : ℝ, op x 0 = x
axiom ax2 : ∀ x y : ℝ, op x y = op y x
axiom ax3 : ∀ x y : ℝ, op (x + 1) y = (op x y) + y + 2

theorem op_7_3_eq_70 : op 7 3 = 70 := by
  sorry

end op_7_3_eq_70_l2354_235429


namespace least_number_l2354_235498

theorem least_number (n : ℕ) : 
  (n % 45 = 2) ∧ (n % 59 = 2) ∧ (n % 77 = 2) → n = 205517 :=
by
  sorry

end least_number_l2354_235498


namespace remainder_when_divided_by_8_l2354_235410

theorem remainder_when_divided_by_8:
  ∀ (n : ℕ), (∃ (q : ℕ), n = 7 * q + 5) → n % 8 = 1 :=
by
  intro n h
  rcases h with ⟨q, hq⟩
  sorry

end remainder_when_divided_by_8_l2354_235410


namespace sin_theta_value_l2354_235458

theorem sin_theta_value (a : ℝ) (h : a ≠ 0) (h_tan : Real.tan θ = -a) (h_point : P = (a, -1)) : Real.sin θ = -Real.sqrt 2 / 2 :=
sorry

end sin_theta_value_l2354_235458


namespace simplify_polynomial_l2354_235437

theorem simplify_polynomial (x : ℝ) :
  (14 * x ^ 12 + 8 * x ^ 9 + 3 * x ^ 8) + (2 * x ^ 14 - x ^ 12 + 2 * x ^ 9 + 5 * x ^ 5 + 7 * x ^ 2 + 6) =
  2 * x ^ 14 + 13 * x ^ 12 + 10 * x ^ 9 + 3 * x ^ 8 + 5 * x ^ 5 + 7 * x ^ 2 + 6 :=
by
  sorry

end simplify_polynomial_l2354_235437


namespace binom_20_5_l2354_235447

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l2354_235447


namespace amalie_coins_proof_l2354_235479

def coins_proof : Prop :=
  ∃ (E A : ℕ),
    (E / A = 10 / 45) ∧
    (E + A = 440) ∧
    ((3 / 4) * A = 270) ∧
    (A - 270 = 90)

theorem amalie_coins_proof : coins_proof :=
  sorry

end amalie_coins_proof_l2354_235479


namespace ratio_w_y_l2354_235496

-- Define the necessary variables
variables (w x y z : ℚ)

-- Define the conditions as hypotheses
axiom h1 : w / x = 4 / 3
axiom h2 : y / z = 5 / 3
axiom h3 : z / x = 1 / 6

-- State the proof problem
theorem ratio_w_y : w / y = 24 / 5 :=
by sorry

end ratio_w_y_l2354_235496


namespace part1_part2_l2354_235401

theorem part1 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin α - Real.cos α = 7 / 5 := sorry

theorem part2 (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi) (h_trig : Real.sin α + Real.cos α = 1 / 5) :
  Real.sin (2 * α + Real.pi / 3) = -12 / 25 - 7 * Real.sqrt 3 / 50 := sorry

end part1_part2_l2354_235401


namespace sum_of_n_values_l2354_235403

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end sum_of_n_values_l2354_235403


namespace distance_from_origin_l2354_235444

theorem distance_from_origin (A : ℝ) (h : |A - 0| = 4) : A = 4 ∨ A = -4 :=
by {
  sorry
}

end distance_from_origin_l2354_235444


namespace total_cost_is_correct_l2354_235463

noncomputable def total_cost_of_gifts : ℝ :=
  let polo_shirts := 3 * 26
  let necklaces := 2 * 83
  let computer_game := 90
  let socks := 4 * 7
  let books := 3 * 15
  let scarves := 2 * 22
  let mugs := 5 * 8
  let sneakers := 65

  let cost_before_discounts := polo_shirts + necklaces + computer_game + socks + books + scarves + mugs + sneakers

  let discount_books := 0.10 * books
  let discount_sneakers := 0.15 * sneakers
  let cost_after_discounts := cost_before_discounts - discount_books - discount_sneakers

  let sales_tax := 0.065 * cost_after_discounts
  let cost_after_tax := cost_after_discounts + sales_tax

  let final_cost := cost_after_tax - 12

  final_cost

theorem total_cost_is_correct :
  total_cost_of_gifts = 564.96 := by
sorry

end total_cost_is_correct_l2354_235463


namespace ratio_dog_to_hamster_l2354_235494

noncomputable def dog_lifespan : ℝ := 10
noncomputable def hamster_lifespan : ℝ := 2.5

theorem ratio_dog_to_hamster : dog_lifespan / hamster_lifespan = 4 :=
by
  sorry

end ratio_dog_to_hamster_l2354_235494


namespace solution_positive_iff_k_range_l2354_235411

theorem solution_positive_iff_k_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (k / (2 * x - 4) - 1 = x / (x - 2))) ↔ (k > -4 ∧ k ≠ 4) := 
sorry

end solution_positive_iff_k_range_l2354_235411


namespace max_product_is_2331_l2354_235470

open Nat

noncomputable def max_product (a b : ℕ) : ℕ :=
  if a + b = 100 ∧ a % 5 = 2 ∧ b % 6 = 3 then a * b else 0

theorem max_product_is_2331 (a b : ℕ) (h_sum : a + b = 100) (h_mod_a : a % 5 = 2) (h_mod_b : b % 6 = 3) :
  max_product a b = 2331 :=
  sorry

end max_product_is_2331_l2354_235470


namespace find_angle_x_l2354_235414

-- Define the angles and parallel lines conditions
def parallel_lines (k l : Prop) (angle1 : Real) (angle2 : Real) : Prop :=
  k ∧ l ∧ angle1 = 30 ∧ angle2 = 90

-- Statement of the problem in Lean syntax
theorem find_angle_x (k l : Prop) (angle1 angle2 : Real) (x : Real) : 
  parallel_lines k l angle1 angle2 → x = 150 :=
by
  -- Assuming conditions are given, prove x = 150
  sorry

end find_angle_x_l2354_235414


namespace reciprocal_inequality_l2354_235455

theorem reciprocal_inequality {a b c : ℝ} (hab : a < b) (hbc : b < c) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_l2354_235455


namespace combined_cost_is_450_l2354_235427

-- Given conditions
def bench_cost : ℕ := 150
def table_cost : ℕ := 2 * bench_cost

-- The statement we want to prove
theorem combined_cost_is_450 : bench_cost + table_cost = 450 :=
by
  sorry

end combined_cost_is_450_l2354_235427


namespace four_distinct_numbers_are_prime_l2354_235464

-- Lean 4 statement proving the conditions
theorem four_distinct_numbers_are_prime : 
  ∃ (a b c d : ℕ), 
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧ 
    (Prime (a * b + c * d)) ∧ 
    (Prime (a * c + b * d)) ∧ 
    (Prime (a * d + b * c)) := 
sorry

end four_distinct_numbers_are_prime_l2354_235464


namespace quadrangular_prism_volume_l2354_235467

theorem quadrangular_prism_volume
  (perimeter : ℝ)
  (side_length : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (H1 : perimeter = 32)
  (H2 : side_length = perimeter / 4)
  (H3 : height = side_length)
  (H4 : volume = side_length * side_length * height) :
  volume = 512 := by
    sorry

end quadrangular_prism_volume_l2354_235467


namespace probability_two_dice_same_l2354_235448

def fair_dice_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  1 - ((sides.factorial / (sides - dice).factorial) / sides^dice)

theorem probability_two_dice_same (dice : ℕ) (sides : ℕ) (h1 : dice = 5) (h2 : sides = 10) :
  fair_dice_probability dice sides = 1744 / 2500 := by
  sorry

end probability_two_dice_same_l2354_235448


namespace smallest_five_digit_perfect_square_and_cube_l2354_235456

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ n = 15625 :=
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l2354_235456


namespace students_not_taking_math_or_physics_l2354_235435

theorem students_not_taking_math_or_physics (total_students math_students phys_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 75)
  (h3 : phys_students = 50)
  (h4 : both_students = 15) :
  total_students - (math_students + phys_students - both_students) = 10 :=
by
  sorry

end students_not_taking_math_or_physics_l2354_235435


namespace find_constant_a_find_ordinary_equation_of_curve_l2354_235484

open Real

theorem find_constant_a (a t : ℝ) (h1 : 1 + 2 * t = 3) (h2 : a * t^2 = 1) : a = 1 :=
by
  -- Proof goes here
  sorry

theorem find_ordinary_equation_of_curve (x y t : ℝ) (h1 : x = 1 + 2 * t) (h2 : y = t^2) :
  (x - 1)^2 = 4 * y :=
by
  -- Proof goes here
  sorry

end find_constant_a_find_ordinary_equation_of_curve_l2354_235484


namespace fraction_equivalence_l2354_235478

theorem fraction_equivalence :
  ( (3 / 7 + 2 / 3) / (5 / 11 + 3 / 8) ) = (119 / 90) :=
by
  sorry

end fraction_equivalence_l2354_235478


namespace no_perfect_square_in_range_l2354_235442

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end no_perfect_square_in_range_l2354_235442


namespace mean_goals_is_correct_l2354_235454

theorem mean_goals_is_correct :
  let goals5 := 5
  let players5 := 4
  let goals6 := 6
  let players6 := 3
  let goals7 := 7
  let players7 := 2
  let goals8 := 8
  let players8 := 1
  let total_goals := goals5 * players5 + goals6 * players6 + goals7 * players7 + goals8 * players8
  let total_players := players5 + players6 + players7 + players8
  (total_goals / total_players : ℝ) = 6 :=
by
  -- The proof is omitted.
  sorry

end mean_goals_is_correct_l2354_235454


namespace rotations_per_block_l2354_235486

/--
If Greg's bike wheels have already rotated 600 times and need to rotate 
1000 more times to reach his goal of riding at least 8 blocks,
then the number of rotations per block is 200.
-/
theorem rotations_per_block (r1 r2 n b : ℕ) (h1 : r1 = 600) (h2 : r2 = 1000) (h3 : n = 8) :
  (r1 + r2) / n = 200 := by
  sorry

end rotations_per_block_l2354_235486


namespace large_jars_count_l2354_235413

theorem large_jars_count (S L : ℕ) (h1 : S + L = 100) (h2 : S = 62) (h3 : 3 * S + 5 * L = 376) : L = 38 :=
by
  sorry

end large_jars_count_l2354_235413


namespace find_set_M_l2354_235492

variable (U : Set ℕ) (M : Set ℕ)

def isUniversalSet : Prop := U = {1, 2, 3, 4, 5, 6}
def isComplement : Prop := U \ M = {1, 2, 4}

theorem find_set_M (hU : isUniversalSet U) (hC : isComplement U M) : M = {3, 5, 6} :=
  sorry

end find_set_M_l2354_235492


namespace nth_derivative_correct_l2354_235465

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (3 * x + 1) + Real.cos (5 * x)

noncomputable def n_th_derivative (n : ℕ) (x : ℝ) : ℝ :=
  3^n * Real.sin ((3 * Real.pi / 2) * n + 3 * x + 1) + 5^n * Real.cos ((3 * Real.pi / 2) * n + 5 * x)

theorem nth_derivative_correct (x : ℝ) (n : ℕ) :
  derivative^[n] y x = n_th_derivative n x :=
by
  sorry

end nth_derivative_correct_l2354_235465


namespace intersection_complement_eq_singleton_l2354_235406

def U : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) }
def M : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ (y - 3) / (x - 2) = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 }
def complement_U (M : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := { p | p ∈ U ∧ p ∉ M }

theorem intersection_complement_eq_singleton :
  N ∩ complement_U M = {(2,3)} :=
by
  sorry

end intersection_complement_eq_singleton_l2354_235406


namespace range_of_a_l2354_235408

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

noncomputable def q (a : ℝ) : Prop :=
  a < 1 ∧ a ≠ 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0 :=
by sorry

end range_of_a_l2354_235408


namespace sqrt_approximation_l2354_235433

theorem sqrt_approximation :
  (2^2 < 5) ∧ (5 < 3^2) ∧ 
  (2.2^2 < 5) ∧ (5 < 2.3^2) ∧ 
  (2.23^2 < 5) ∧ (5 < 2.24^2) ∧ 
  (2.236^2 < 5) ∧ (5 < 2.237^2) →
  (Float.ceil (Float.sqrt 5 * 100) / 100) = 2.24 := 
by
  intro h
  sorry

end sqrt_approximation_l2354_235433
