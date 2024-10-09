import Mathlib

namespace garden_perimeter_l775_77551

theorem garden_perimeter (width_garden length_playground width_playground : ℕ) 
  (h1 : width_garden = 12) 
  (h2 : length_playground = 16) 
  (h3 : width_playground = 12) 
  (area_playground : ℕ)
  (h4 : area_playground = length_playground * width_playground) 
  (area_garden : ℕ) 
  (h5 : area_garden = area_playground) 
  (length_garden : ℕ) 
  (h6 : area_garden = length_garden * width_garden) :
  2 * length_garden + 2 * width_garden = 56 := by
  sorry

end garden_perimeter_l775_77551


namespace pencils_per_row_cannot_be_determined_l775_77529

theorem pencils_per_row_cannot_be_determined
  (rows : ℕ)
  (total_crayons : ℕ)
  (crayons_per_row : ℕ)
  (h_total_crayons: total_crayons = 210)
  (h_rows: rows = 7)
  (h_crayons_per_row: crayons_per_row = 30) :
  ∀ (pencils_per_row : ℕ), false :=
by
  sorry

end pencils_per_row_cannot_be_determined_l775_77529


namespace rowing_upstream_speed_l775_77510

theorem rowing_upstream_speed (V_down V_m : ℝ) (h_down : V_down = 35) (h_still : V_m = 31) : ∃ V_up, V_up = V_m - (V_down - V_m) ∧ V_up = 27 := by
  sorry

end rowing_upstream_speed_l775_77510


namespace abe_age_is_22_l775_77570

-- Define the conditions of the problem
def abe_age_condition (A : ℕ) : Prop := A + (A - 7) = 37

-- State the theorem
theorem abe_age_is_22 : ∃ A : ℕ, abe_age_condition A ∧ A = 22 :=
by
  sorry

end abe_age_is_22_l775_77570


namespace find_a7_a8_l775_77555

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (g : geometric_sequence a q)

def sum_1_2 : ℝ := a 1 + a 2
def sum_3_4 : ℝ := a 3 + a 4

theorem find_a7_a8
  (h1 : sum_1_2 = 30)
  (h2 : sum_3_4 = 60)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 7 + a 8 = (a 1 + a 2) * (q ^ 6) := 
sorry

end find_a7_a8_l775_77555


namespace parabola_passes_through_points_and_has_solution_4_l775_77512

theorem parabola_passes_through_points_and_has_solution_4 
  (a h k m: ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k → 
    (y = 0 → (x = -1 → x = 5))) → 
  (∃ m, ∀ x, (a * (x - h + m) ^ 2 + k = 0) → x = 4) → 
  m = -5 ∨ m = 1 :=
sorry

end parabola_passes_through_points_and_has_solution_4_l775_77512


namespace problem1_problem2_l775_77503

-- Definitions for the number of combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1
theorem problem1 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 4) + (C r 3 * C w 1) + (C r 2 * C w 2) = 115 := 
sorry

-- Problem 2
theorem problem2 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 2 * C w 3) + (C r 3 * C w 2) + (C r 4 * C w 1) = 186 := 
sorry

end problem1_problem2_l775_77503


namespace original_population_l775_77573

variable (n : ℝ)

theorem original_population
  (h1 : n + 1500 - 0.15 * (n + 1500) = n - 45) :
  n = 8800 :=
sorry

end original_population_l775_77573


namespace find_number_l775_77506

-- Define the given condition
def number_div_property (num : ℝ) : Prop :=
  num / 0.3 = 7.3500000000000005

-- State the theorem to prove
theorem find_number (num : ℝ) (h : number_div_property num) : num = 2.205 :=
by sorry

end find_number_l775_77506


namespace geometric_sequence_problem_l775_77513

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Geometric sequence definition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
def condition_1 : Prop := a 5 * a 8 = 6
def condition_2 : Prop := a 3 + a 10 = 5

-- Concluded value of q^7
def q_seven (q : ℝ) (a : ℕ → ℝ) : Prop := 
  q^7 = a 20 / a 13

theorem geometric_sequence_problem
  (h1 : is_geometric_sequence a q)
  (h2 : condition_1 a)
  (h3 : condition_2 a) :
  q_seven q a = (q = 3/2) ∨ (q = 2/3) :=
sorry

end geometric_sequence_problem_l775_77513


namespace speed_of_first_train_l775_77516

-- Define the conditions
def distance_pq := 110 -- km
def speed_q := 25 -- km/h
def meet_time := 10 -- hours from midnight
def start_p := 7 -- hours from midnight
def start_q := 8 -- hours from midnight

-- Define the total travel time for each train
def travel_time_p := meet_time - start_p -- hours
def travel_time_q := meet_time - start_q -- hours

-- Define the distance covered by each train
def distance_covered_p (V_p : ℕ) : ℕ := V_p * travel_time_p
def distance_covered_q := speed_q * travel_time_q

-- Theorem to prove the speed of the first train
theorem speed_of_first_train (V_p : ℕ) : distance_covered_p V_p + distance_covered_q = distance_pq → V_p = 20 :=
sorry

end speed_of_first_train_l775_77516


namespace fraction_checked_by_worker_y_l775_77550

-- Definitions of conditions given in the problem
variable (P Px Py : ℝ)
variable (h1 : Px + Py = P)
variable (h2 : 0.005 * Px = defective_x)
variable (h3 : 0.008 * Py = defective_y)
variable (defective_x defective_y : ℝ)
variable (total_defective : ℝ)
variable (h4 : defective_x + defective_y = total_defective)
variable (h5 : total_defective = 0.0065 * P)

-- The fraction of products checked by worker y
theorem fraction_checked_by_worker_y (h : Px + Py = P) (h2 : 0.005 * Px = 0.0065 * P) (h3 : 0.008 * Py = 0.0065 * P) :
  Py / P = 1 / 2 := 
  sorry

end fraction_checked_by_worker_y_l775_77550


namespace seth_initial_boxes_l775_77549

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end seth_initial_boxes_l775_77549


namespace negation_exists_l775_77517

theorem negation_exists {x : ℝ} (h : ∀ x, x > 0 → x^2 - x ≤ 0) : ∃ x, x > 0 ∧ x^2 - x > 0 :=
sorry

end negation_exists_l775_77517


namespace grandpa_uncle_ratio_l775_77596

def initial_collection := 150
def dad_gift := 10
def mum_gift := dad_gift + 5
def auntie_gift := 6
def uncle_gift := auntie_gift - 1
def final_collection := 196
def total_cars_needed := final_collection - initial_collection
def other_gifts := dad_gift + mum_gift + auntie_gift + uncle_gift
def grandpa_gift := total_cars_needed - other_gifts

theorem grandpa_uncle_ratio : grandpa_gift = 2 * uncle_gift := by
  sorry

end grandpa_uncle_ratio_l775_77596


namespace sum_remainder_l775_77542

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 5) (h3 : c % 30 = 18) : 
  (a + b + c) % 30 = 7 :=
by
  sorry

end sum_remainder_l775_77542


namespace percentage_increase_bears_with_assistant_l775_77583

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l775_77583


namespace jane_reading_speed_second_half_l775_77535

-- Definitions from the problem's conditions
def total_pages : ℕ := 500
def first_half_pages : ℕ := total_pages / 2
def first_half_speed : ℕ := 10
def total_days : ℕ := 75

-- The number of days spent reading the first half
def first_half_days : ℕ := first_half_pages / first_half_speed

-- The number of days spent reading the second half
def second_half_days : ℕ := total_days - first_half_days

-- The number of pages in the second half
def second_half_pages : ℕ := total_pages - first_half_pages

-- The actual theorem stating that Jane's reading speed for the second half was 5 pages per day
theorem jane_reading_speed_second_half :
  second_half_pages / second_half_days = 5 :=
by
  sorry

end jane_reading_speed_second_half_l775_77535


namespace youngsville_population_l775_77531

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end youngsville_population_l775_77531


namespace probability_of_neighboring_points_l775_77539

theorem probability_of_neighboring_points (n : ℕ) (h : n ≥ 3) : 
  (2 / (n - 1) : ℝ) = (n / (n * (n - 1) / 2) : ℝ) :=
by sorry

end probability_of_neighboring_points_l775_77539


namespace average_after_17th_inning_l775_77567

theorem average_after_17th_inning (A : ℝ) (total_runs_16th_inning : ℝ) 
  (average_before_17th : A * 16 = total_runs_16th_inning) 
  (increased_average_by_3 : (total_runs_16th_inning + 83) / 17 = A + 3) :
  (A + 3) = 35 := 
sorry

end average_after_17th_inning_l775_77567


namespace jack_pages_l775_77521

theorem jack_pages (pages_per_booklet : ℕ) (num_booklets : ℕ) (h1 : pages_per_booklet = 9) (h2 : num_booklets = 49) : num_booklets * pages_per_booklet = 441 :=
by {
  sorry
}

end jack_pages_l775_77521


namespace g_s_difference_l775_77589

def g (n : ℤ) : ℤ := n^3 + 3 * n^2 + 3 * n + 1

theorem g_s_difference (s : ℤ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end g_s_difference_l775_77589


namespace necessary_but_not_sufficient_condition_l775_77565

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  {x | 1 / x ≤ 1} ⊆ {x | Real.log x ≥ 0} ∧ 
  ¬ ({x | Real.log x ≥ 0} ⊆ {x | 1 / x ≤ 1}) :=
by
  sorry

end necessary_but_not_sufficient_condition_l775_77565


namespace minimize_y_l775_77541

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, x = (3 * a + b) / 4 ∧ 
  ∀ y : ℝ, (3 * (y - a) ^ 2 + (y - b) ^ 2) ≥ (3 * ((3 * a + b) / 4 - a) ^ 2 + ((3 * a + b) / 4 - b) ^ 2) :=
sorry

end minimize_y_l775_77541


namespace find_C_work_rate_l775_77568

-- Conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 6

-- Combined work rate of A and B
def AB_work_rate := A_work_rate + B_work_rate

-- Total work rate when C is assisting, completing in 2 days
def total_work_rate_of_ABC := 1 / 2

theorem find_C_work_rate : ∃ c : ℕ, (AB_work_rate + 1 / c = total_work_rate_of_ABC) ∧ c = 12 :=
by
  -- To complete the proof, we solve the equation for c
  sorry

end find_C_work_rate_l775_77568


namespace arc_length_l775_77598

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 7) : (α * r) = 2 * π / 7 := by
  sorry

end arc_length_l775_77598


namespace find_x_l775_77585

theorem find_x (x : ℝ) (h : (3 * x) / 4 = 24) : x = 32 :=
by
  sorry

end find_x_l775_77585


namespace lucy_withdrawal_l775_77566

-- Given conditions
def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

-- Define balance before withdrawal
def balance_before_withdrawal := initial_balance + deposit

-- Theorem to prove
theorem lucy_withdrawal : balance_before_withdrawal - final_balance = 4 :=
by sorry

end lucy_withdrawal_l775_77566


namespace boxwoods_shaped_into_spheres_l775_77575

theorem boxwoods_shaped_into_spheres :
  ∀ (total_boxwoods : ℕ) (cost_trimming : ℕ) (cost_shaping : ℕ) (total_charge : ℕ) (x : ℕ),
    total_boxwoods = 30 →
    cost_trimming = 5 →
    cost_shaping = 15 →
    total_charge = 210 →
    30 * 5 + x * 15 = 210 →
    x = 4 :=
by
  intros total_boxwoods cost_trimming cost_shaping total_charge x
  rintro rfl rfl rfl rfl h
  sorry

end boxwoods_shaped_into_spheres_l775_77575


namespace negation_of_sine_bound_l775_77500

theorem negation_of_sine_bound (p : ∀ x : ℝ, Real.sin x ≤ 1) : ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x₀ : ℝ, Real.sin x₀ > 1 := 
by 
  sorry

end negation_of_sine_bound_l775_77500


namespace merchant_discount_percentage_l775_77564

theorem merchant_discount_percentage
  (CP MP SP : ℝ)
  (h1 : MP = CP + 0.40 * CP)
  (h2 : SP = CP + 0.26 * CP)
  : ((MP - SP) / MP) * 100 = 10 := by
  sorry

end merchant_discount_percentage_l775_77564


namespace rhombus_min_rotation_l775_77522

theorem rhombus_min_rotation (α : ℝ) (h1 : α = 60) : ∃ θ, θ = 180 := 
by 
  -- The proof here will show that the minimum rotation angle is 180°
  sorry

end rhombus_min_rotation_l775_77522


namespace rectangle_perimeter_change_l775_77509

theorem rectangle_perimeter_change :
  ∀ (a b : ℝ), 
  (2 * (a + b) = 2 * (1.3 * a + 0.8 * b)) →
  ((2 * (0.8 * a + 1.95 * b) - 2 * (a + b)) / (2 * (a + b)) = 0.1) :=
by
  intros a b h
  sorry

end rectangle_perimeter_change_l775_77509


namespace opposite_and_reciprocal_numbers_l775_77515

theorem opposite_and_reciprocal_numbers (a b c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1) :
  2019 * a + (7 / (c * d)) + 2019 * b = 7 :=
sorry

end opposite_and_reciprocal_numbers_l775_77515


namespace probability_coin_covers_black_region_l775_77587

open Real

noncomputable def coin_cover_black_region_probability : ℝ :=
  let side_length_square := 10
  let triangle_leg := 3
  let diamond_side_length := 3 * sqrt 2
  let smaller_square_side := 1
  let coin_diameter := 1
  -- The derived probability calculation
  (32 + 9 * sqrt 2 + π) / 81

theorem probability_coin_covers_black_region :
  coin_cover_black_region_probability = (32 + 9 * sqrt 2 + π) / 81 :=
by
  -- Proof goes here
  sorry

end probability_coin_covers_black_region_l775_77587


namespace probability_not_snowing_l775_77559

variable (P_snowing : ℚ)
variable (h : P_snowing = 2/5)

theorem probability_not_snowing (P_not_snowing : ℚ) : 
  P_not_snowing = 3 / 5 :=
by 
  -- sorry to skip the proof
  sorry

end probability_not_snowing_l775_77559


namespace mike_initial_games_l775_77534

theorem mike_initial_games (v w: ℕ)
  (h_non_working : v - w = 8)
  (h_earnings : 7 * w = 56)
  : v = 16 :=
by
  sorry

end mike_initial_games_l775_77534


namespace granddaughter_fraction_l775_77582

noncomputable def betty_age : ℕ := 60
def fraction_younger (p : ℕ) : ℕ := (p * 40) / 100
noncomputable def daughter_age : ℕ := betty_age - fraction_younger betty_age
def granddaughter_age : ℕ := 12
def fraction (a b : ℕ) : ℚ := a / b

theorem granddaughter_fraction :
  fraction granddaughter_age daughter_age = 1 / 3 := 
by
  sorry

end granddaughter_fraction_l775_77582


namespace factorization_of_expression_l775_77572

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l775_77572


namespace line_slope_l775_77530

theorem line_slope (t : ℝ) : 
  (∃ (t : ℝ), x = 1 + 2 * t ∧ y = 2 - 3 * t) → 
  (∃ (m : ℝ), m = -3 / 2) :=
sorry

end line_slope_l775_77530


namespace induction_divisibility_l775_77511

theorem induction_divisibility (k x y : ℕ) (h : k > 0) :
  (x^(2*k-1) + y^(2*k-1)) ∣ (x + y) → 
  (x^(2*k+1) + y^(2*k+1)) ∣ (x + y) :=
sorry

end induction_divisibility_l775_77511


namespace cost_of_swim_trunks_is_14_l775_77576

noncomputable def cost_of_swim_trunks : Real :=
  let flat_rate_shipping := 5.00
  let shipping_rate := 0.20
  let price_shirt := 12.00
  let price_socks := 5.00
  let price_shorts := 15.00
  let cost_known_items := 3 * price_shirt + price_socks + 2 * price_shorts
  let total_bill := 102.00
  let x := (total_bill - 0.20 * cost_known_items - cost_known_items) / 1.20
  x

theorem cost_of_swim_trunks_is_14 : cost_of_swim_trunks = 14 := by
  -- sorry is used to skip the proof
  sorry

end cost_of_swim_trunks_is_14_l775_77576


namespace Marley_fruit_count_l775_77536

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l775_77536


namespace solve_for_b_l775_77578

theorem solve_for_b (b : ℝ) (hb : b + ⌈b⌉ = 17.8) : b = 8.8 := 
by sorry

end solve_for_b_l775_77578


namespace actual_cost_of_article_l775_77523

theorem actual_cost_of_article {x : ℝ} (h : 0.76 * x = 760) : x = 1000 :=
by
  sorry

end actual_cost_of_article_l775_77523


namespace problem_l775_77518

def a (x : ℕ) : ℕ := 2005 * x + 2006
def b (x : ℕ) : ℕ := 2005 * x + 2007
def c (x : ℕ) : ℕ := 2005 * x + 2008

theorem problem (x : ℕ) : (a x)^2 + (b x)^2 + (c x)^2 - (a x) * (b x) - (a x) * (c x) - (b x) * (c x) = 3 :=
by sorry

end problem_l775_77518


namespace base7_number_l775_77528

theorem base7_number (A B C : ℕ) (h1 : 1 ≤ A ∧ A ≤ 6) (h2 : 1 ≤ B ∧ B ≤ 6) (h3 : 1 ≤ C ∧ C ≤ 6)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_condition1 : B + C = 7)
  (h_condition2 : A + 1 = C)
  (h_condition3 : A + B = C) :
  A = 5 ∧ B = 1 ∧ C = 6 :=
sorry

end base7_number_l775_77528


namespace value_of_ab_l775_77508

theorem value_of_ab (a b : ℝ) (x : ℝ) 
  (h : ∀ x, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : a * b = 0 :=
sorry

end value_of_ab_l775_77508


namespace linear_function_common_quadrants_l775_77532

theorem linear_function_common_quadrants {k b : ℝ} (h : k * b < 0) :
  (exists (q1 q2 : ℕ), q1 = 1 ∧ q2 = 4) := 
sorry

end linear_function_common_quadrants_l775_77532


namespace relatively_prime_m_n_l775_77577

noncomputable def probability_of_distinct_real_solutions : ℝ :=
  let b := (1 : ℝ)
  if 1 ≤ b ∧ b ≤ 25 then 1 else 0

theorem relatively_prime_m_n : ∃ m n : ℕ, 
  Nat.gcd m n = 1 ∧ 
  (1 : ℝ) = (m : ℝ) / (n : ℝ) ∧ m + n = 2 := 
by
  sorry

end relatively_prime_m_n_l775_77577


namespace avg_hamburgers_per_day_l775_77537

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 49) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 7 :=
by {
  sorry
}

end avg_hamburgers_per_day_l775_77537


namespace b_plus_d_over_a_l775_77554

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l775_77554


namespace domain_of_sqrt_quadratic_l775_77595

open Set

def domain_of_f : Set ℝ := {x : ℝ | 2*x - x^2 ≥ 0}

theorem domain_of_sqrt_quadratic :
  domain_of_f = Icc 0 2 :=
by
  sorry

end domain_of_sqrt_quadratic_l775_77595


namespace sum_of_coefficients_l775_77548

noncomputable def problem_expr (d : ℝ) := (16 * d + 15 + 18 * d^2 + 3 * d^3) + (4 * d + 2 + d^2 + 2 * d^3)
noncomputable def simplified_expr (d : ℝ) := 5 * d^3 + 19 * d^2 + 20 * d + 17

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : 
  problem_expr d = simplified_expr d ∧ (5 + 19 + 20 + 17 = 61) := 
by
  sorry

end sum_of_coefficients_l775_77548


namespace mass_of_compound_l775_77588

-- Constants as per the conditions
def molecular_weight : ℕ := 444           -- The molecular weight in g/mol.
def number_of_moles : ℕ := 6             -- The number of moles.

-- Defining the main theorem we want to prove.
theorem mass_of_compound : (number_of_moles * molecular_weight) = 2664 := by 
  sorry

end mass_of_compound_l775_77588


namespace max_f_value_l775_77571

open Real

noncomputable def problem (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) : Prop :=
  x1 * x2 * x3 = ((12 - x1) * (12 - x2) * (12 - x3))^2

theorem max_f_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) (h : problem x1 x2 x3 h1 h2 h3) : 
  x1 * x2 * x3 ≤ 729 :=
sorry

end max_f_value_l775_77571


namespace probability_of_continuous_stripe_pattern_l775_77546

def tetrahedron_stripes := 
  let faces := 4
  let configurations_per_face := 2
  2 ^ faces

def continuous_stripe_probability := 
  let total_configurations := tetrahedron_stripes
  1 / total_configurations * 4 -- Since final favorable outcomes calculation is already given and inferred to be 1/4.
  -- or any other logic that follows here based on problem description but this matches problem's derivation

theorem probability_of_continuous_stripe_pattern : continuous_stripe_probability = 1 / 4 := by
  sorry

end probability_of_continuous_stripe_pattern_l775_77546


namespace trigonometric_expression_evaluation_l775_77505

theorem trigonometric_expression_evaluation (θ : ℝ) (hθ : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 :=
by
  sorry

end trigonometric_expression_evaluation_l775_77505


namespace number_of_real_solutions_eq_2_l775_77520

theorem number_of_real_solutions_eq_2 :
  ∃! (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3 :=
sorry

end number_of_real_solutions_eq_2_l775_77520


namespace Marissa_has_21_more_marbles_than_Jonny_l775_77525

noncomputable def Mara_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Markus_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Jonny_marbles (total_marbles : ℕ) (bags : ℕ) : ℕ :=
total_marbles

noncomputable def Marissa_marbles (bags1 : ℕ) (marbles1 : ℕ) (bags2 : ℕ) (marbles2 : ℕ) : ℕ :=
(bags1 * marbles1) + (bags2 * marbles2)

noncomputable def Jonny : ℕ := Jonny_marbles 18 3

noncomputable def Marissa : ℕ := Marissa_marbles 3 5 3 8

theorem Marissa_has_21_more_marbles_than_Jonny : (Marissa - Jonny) = 21 :=
by
  sorry

end Marissa_has_21_more_marbles_than_Jonny_l775_77525


namespace find_non_negative_integer_solutions_l775_77533

theorem find_non_negative_integer_solutions :
  ∃ (x y z w : ℕ), 2 ^ x * 3 ^ y - 5 ^ z * 7 ^ w = 1 ∧
  ((x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
   (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1)) := by
  sorry

end find_non_negative_integer_solutions_l775_77533


namespace range_of_first_person_l775_77581

variable (R1 R2 R3 : ℕ)
variable (min_range : ℕ)
variable (condition1 : min_range = 25)
variable (condition2 : R2 = 25)
variable (condition3 : R3 = 30)
variable (condition4 : min_range ≤ R1 ∧ min_range ≤ R2 ∧ min_range ≤ R3)

theorem range_of_first_person : R1 = 25 :=
by
  sorry

end range_of_first_person_l775_77581


namespace exists_neg_monomial_l775_77526

theorem exists_neg_monomial (a : ℤ) (x y : ℤ) (m n : ℕ) (hq : a < 0) (hd : m + n = 5) :
  ∃ a m n, a < 0 ∧ m + n = 5 ∧ a * x^m * y^n = -x^2 * y^3 :=
by
  sorry

end exists_neg_monomial_l775_77526


namespace average_rainfall_per_hour_in_June_1882_l775_77594

open Real

theorem average_rainfall_per_hour_in_June_1882 
  (total_rainfall : ℝ) (days_in_June : ℕ) (hours_per_day : ℕ)
  (H1 : total_rainfall = 450) (H2 : days_in_June = 30) (H3 : hours_per_day = 24) :
  total_rainfall / (days_in_June * hours_per_day) = 5 / 8 :=
by
  sorry

end average_rainfall_per_hour_in_June_1882_l775_77594


namespace arithmetic_progressions_count_l775_77553

theorem arithmetic_progressions_count (d : ℕ) (h_d : d = 2) (S : ℕ) (h_S : S = 200) : 
  ∃ n : ℕ, n = 6 := sorry

end arithmetic_progressions_count_l775_77553


namespace frequency_of_second_group_l775_77593

theorem frequency_of_second_group (total_capacity : ℕ) (freq_percentage : ℝ)
    (h_capacity : total_capacity = 80)
    (h_percentage : freq_percentage = 0.15) :
    total_capacity * freq_percentage = 12 :=
by
  sorry

end frequency_of_second_group_l775_77593


namespace vacant_seats_l775_77574

theorem vacant_seats (total_seats filled_percentage : ℕ) (h_filled_percentage : filled_percentage = 62) (h_total_seats : total_seats = 600) : 
  (total_seats - total_seats * filled_percentage / 100) = 228 :=
by
  sorry

end vacant_seats_l775_77574


namespace each_nap_duration_l775_77552

-- Definitions based on the problem conditions
def BillProjectDurationInDays : ℕ := 4
def HoursPerDay : ℕ := 24
def TotalProjectHours : ℕ := BillProjectDurationInDays * HoursPerDay
def WorkHours : ℕ := 54
def NapsTaken : ℕ := 6

-- Calculate the time spent on naps and the duration of each nap
def NapHoursTotal : ℕ := TotalProjectHours - WorkHours
def DurationEachNap : ℕ := NapHoursTotal / NapsTaken

-- The theorem stating the expected answer
theorem each_nap_duration :
  DurationEachNap = 7 := by
  sorry

end each_nap_duration_l775_77552


namespace rohan_food_percentage_l775_77563

noncomputable def rohan_salary : ℝ := 7500
noncomputable def rohan_savings : ℝ := 1500
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def entertainment_percentage : ℝ := 0.10
noncomputable def conveyance_percentage : ℝ := 0.10
noncomputable def total_spent : ℝ := rohan_salary - rohan_savings
noncomputable def known_percentage : ℝ := house_rent_percentage + entertainment_percentage + conveyance_percentage

theorem rohan_food_percentage (F : ℝ) :
  total_spent = rohan_salary * (1 - known_percentage - F) →
  F = 0.20 :=
sorry

end rohan_food_percentage_l775_77563


namespace division_result_l775_77591

def n : ℕ := 16^1024

theorem division_result : n / 8 = 2^4093 :=
by sorry

end division_result_l775_77591


namespace next_term_in_geom_sequence_l775_77556

   /- Define the given geometric sequence as a function in Lean -/

   def geom_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ n

   theorem next_term_in_geom_sequence (x : ℤ) (n : ℕ) 
     (h₁ : geom_sequence 3 (-3*x) 0 = 3)
     (h₂ : geom_sequence 3 (-3*x) 1 = -9*x)
     (h₃ : geom_sequence 3 (-3*x) 2 = 27*(x^2))
     (h₄ : geom_sequence 3 (-3*x) 3 = -81*(x^3)) :
     geom_sequence 3 (-3*x) 4 = 243*(x^4) := 
   sorry
   
end next_term_in_geom_sequence_l775_77556


namespace gamma_max_two_day_success_ratio_l775_77590

theorem gamma_max_two_day_success_ratio :
  ∃ (e g f h : ℕ), 0 < e ∧ 0 < g ∧
  e + g = 335 ∧ 
  e < f ∧ g < h ∧ 
  f + h = 600 ∧ 
  (e : ℚ) / f < (180 : ℚ) / 360 ∧ 
  (g : ℚ) / h < (150 : ℚ) / 240 ∧ 
  (e + g) / 600 = 67 / 120 :=
by
  sorry

end gamma_max_two_day_success_ratio_l775_77590


namespace Tim_gave_kittens_to_Jessica_l775_77502

def Tim_original_kittens : ℕ := 6
def kittens_given_to_Jessica := 3
def kittens_given_by_Sara : ℕ := 9 
def Tim_final_kittens : ℕ := 12

theorem Tim_gave_kittens_to_Jessica :
  (Tim_original_kittens + kittens_given_by_Sara - kittens_given_to_Jessica = Tim_final_kittens) :=
by sorry

end Tim_gave_kittens_to_Jessica_l775_77502


namespace determine_digit_phi_l775_77519

theorem determine_digit_phi (Φ : ℕ) (h1 : Φ > 0) (h2 : Φ < 10) (h3 : 504 / Φ = 40 + 3 * Φ) : Φ = 8 :=
by
  sorry

end determine_digit_phi_l775_77519


namespace fields_fertilized_in_25_days_l775_77584

-- Definitions from conditions
def fertilizer_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def fertilizer_needed_per_acre : ℕ := 400
def number_of_acres : ℕ := 20
def acres_fertilized_per_day : ℕ := 4

-- Total fertilizer produced per day
def total_fertilizer_per_day : ℕ := fertilizer_per_horse_per_day * number_of_horses

-- Total fertilizer needed
def total_fertilizer_needed : ℕ := fertilizer_needed_per_acre * number_of_acres

-- Days to collect enough fertilizer
def days_to_collect_fertilizer : ℕ := total_fertilizer_needed / total_fertilizer_per_day

-- Days to spread fertilizer
def days_to_spread_fertilizer : ℕ := number_of_acres / acres_fertilized_per_day

-- Calculate the total time until all fields are fertilized
def total_days : ℕ := days_to_collect_fertilizer + days_to_spread_fertilizer

-- Theorem statement
theorem fields_fertilized_in_25_days : total_days = 25 :=
by
  sorry

end fields_fertilized_in_25_days_l775_77584


namespace proof_value_g_expression_l775_77599

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom g_invertible : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : ∀ x, (x = 1 → g x = 4) ∧ (x = 2 → g x = 5) ∧ (x = 3 → g x = 7) ∧ (x = 4 → g x = 9) ∧ (x = 5 → g x = 10)

theorem proof_value_g_expression :
  g (g 2) + g (g_inv 9) + g_inv (g_inv 7) = 21 :=
by
  sorry

end proof_value_g_expression_l775_77599


namespace mary_number_l775_77558

-- Definitions for conditions
def has_factor_150 (m : ℕ) : Prop := 150 ∣ m
def is_multiple_of_45 (m : ℕ) : Prop := 45 ∣ m
def in_range (m : ℕ) : Prop := 1000 < m ∧ m < 3000

-- Theorem stating that Mary's number is one of {1350, 1800, 2250, 2700} given the conditions
theorem mary_number 
  (m : ℕ) 
  (h1 : has_factor_150 m)
  (h2 : is_multiple_of_45 m)
  (h3 : in_range m) :
  m = 1350 ∨ m = 1800 ∨ m = 2250 ∨ m = 2700 :=
sorry

end mary_number_l775_77558


namespace part1_inequality_part2_range_l775_77540

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

-- Part 1: Prove that f(x) ≥ f(0) for all x
theorem part1_inequality : ∀ x : ℝ, f x ≥ f 0 :=
sorry

-- Part 2: Prove that the range of a satisfying 2f(x) ≥ f(a+1) for all x is -4.5 ≤ a ≤ 1.5
theorem part2_range (a : ℝ) (h : ∀ x : ℝ, 2 * f x ≥ f (a + 1)) : -4.5 ≤ a ∧ a ≤ 1.5 :=
sorry

end part1_inequality_part2_range_l775_77540


namespace sale_in_fifth_month_l775_77501

theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 : ℕ)
  (avg : ℕ)
  (h1 : s1 = 5435)
  (h2 : s2 = 5927)
  (h3 : s3 = 5855)
  (h4 : s4 = 6230)
  (h6 : s6 = 3991)
  (hav : avg = 5500) :
  ∃ s5 : ℕ, s1 + s2 + s3 + s4 + s5 + s6 = avg * 6 ∧ s5 = 5562 := 
by
  sorry

end sale_in_fifth_month_l775_77501


namespace cone_volume_half_sector_rolled_l775_77538

theorem cone_volume_half_sector_rolled {r slant_height h V : ℝ}
  (radius_given : r = 3)
  (height_calculated : h = 3 * Real.sqrt 3)
  (slant_height_given : slant_height = 6)
  (arc_length : 2 * Real.pi * r = 6 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * (r^2) * h) :
  V = 9 * Real.pi * Real.sqrt 3 :=
by {
  sorry
}

end cone_volume_half_sector_rolled_l775_77538


namespace marbles_shared_equally_l775_77569

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l775_77569


namespace right_angled_triangle_not_axisymmetric_l775_77524

-- Define a type for geometric figures
inductive Figure
| Angle : Figure
| EquilateralTriangle : Figure
| LineSegment : Figure
| RightAngledTriangle : Figure

open Figure

-- Define a function to determine if a figure is axisymmetric
def is_axisymmetric: Figure -> Prop
| Angle => true
| EquilateralTriangle => true
| LineSegment => true
| RightAngledTriangle => false

-- Statement of the problem
theorem right_angled_triangle_not_axisymmetric : 
  is_axisymmetric RightAngledTriangle = false :=
by
  sorry

end right_angled_triangle_not_axisymmetric_l775_77524


namespace inequality_proof_l775_77545

variables {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l775_77545


namespace find_k_l775_77507

-- Define the conditions
variables (x y k : ℕ)
axiom part_sum : x + y = 36
axiom first_part : x = 19
axiom value_eq : 8 * x + k * y = 203

-- Prove that k is 3
theorem find_k : k = 3 :=
by
  -- Insert your proof here
  sorry

end find_k_l775_77507


namespace initial_card_distribution_l775_77557

variables {A B C D : ℕ}

theorem initial_card_distribution 
  (total_cards : A + B + C + D = 32)
  (alfred_final : ∀ c, c = A → ((c / 2) + (c / 2)) + B + C + D = 8)
  (bruno_final : ∀ c, c = B → ((c / 2) + (c / 2)) + A + C + D = 8)
  (christof_final : ∀ c, c = C → ((c / 2) + (c / 2)) + A + B + D = 8)
  : A = 7 ∧ B = 7 ∧ C = 10 ∧ D = 8 :=
by sorry

end initial_card_distribution_l775_77557


namespace max_product_production_l775_77580

theorem max_product_production (C_mats A_mats C_ship A_ship B_mats B_ship : ℝ)
  (cost_A cost_B ship_A ship_B : ℝ) (prod_A prod_B max_cost_mats max_cost_ship prod_max : ℝ)
  (h_prod_A : prod_A = 90)
  (h_cost_A : cost_A = 1000)
  (h_ship_A : ship_A = 500)
  (h_prod_B : prod_B = 100)
  (h_cost_B : cost_B = 1500)
  (h_ship_B : ship_B = 400)
  (h_max_cost_mats : max_cost_mats = 6000)
  (h_max_cost_ship : max_cost_ship = 2000)
  (h_prod_max : prod_max = 440)
  (H_C_mats : C_mats = cost_A * A_mats + cost_B * B_mats)
  (H_C_ship : C_ship = ship_A * A_ship + ship_B * B_ship)
  (H_A_mats_ship : A_mats = A_ship)
  (H_B_mats_ship : B_mats = B_ship)
  (H_C_mats_le : C_mats ≤ max_cost_mats)
  (H_C_ship_le : C_ship ≤ max_cost_ship) :
  prod_A * A_mats + prod_B * B_mats ≤ prod_max :=
by {
  sorry
}

end max_product_production_l775_77580


namespace shortest_chord_intercepted_by_line_l775_77514

theorem shortest_chord_intercepted_by_line (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 3 = 0 → y = k*x + 1 → (x - y + 1 = 0)) :=
sorry

end shortest_chord_intercepted_by_line_l775_77514


namespace find_functional_l775_77527

noncomputable def functional_equation_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = 2 * x + f (f y - x)

theorem find_functional (f : ℝ → ℝ) :
  functional_equation_solution f → ∃ c : ℝ, ∀ x, f x = x + c := 
by
  sorry

end find_functional_l775_77527


namespace intersection_of_A_and_B_l775_77592

open Set

def A : Set ℝ := { x | 3 * x + 2 > 0 }
def B : Set ℝ := { x | (x + 1) * (x - 3) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | x > 3 } :=
by 
  sorry

end intersection_of_A_and_B_l775_77592


namespace optimal_solution_for_z_is_1_1_l775_77544

def x := 1
def y := 1
def z (x y : ℝ) := 2 * x + y

theorem optimal_solution_for_z_is_1_1 :
  ∀ (x y : ℝ), z x y ≥ z 1 1 := 
by
  simp [z]
  sorry

end optimal_solution_for_z_is_1_1_l775_77544


namespace part1_part2_l775_77579

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l775_77579


namespace mulch_price_per_pound_l775_77560

noncomputable def price_per_pound (total_cost : ℝ) (total_tons : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  total_cost / (total_tons * pounds_per_ton)

theorem mulch_price_per_pound :
  price_per_pound 15000 3 2000 = 2.5 :=
by
  sorry

end mulch_price_per_pound_l775_77560


namespace set_intersection_example_l775_77586

theorem set_intersection_example :
  let A := { y | ∃ x, y = Real.log x / Real.log 2 ∧ x ≥ 3 }
  let B := { x | x^2 - 4 * x + 3 = 0 }
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l775_77586


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l775_77561

-- Case 1
theorem quadratic_function_expression 
  (a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = 3) : 
  by {exact (a = -2 ∧ b = 3)} := sorry

theorem quadratic_function_range 
  (x : ℝ) 
  (h : -1 ≤ x ∧ x ≤ 2) : 
  (-3 ≤ -2*x^2 + 3*x + 2 ∧ -2*x^2 + 3*x + 2 ≤ 25/8) := sorry

-- Case 2
theorem quadratic_function_m_range 
  (m a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = m) 
  (h₃ : a > 0) : 
  m < 1 := sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l775_77561


namespace evaluate_expression_at_three_l775_77504

-- Define the evaluation of the expression (x^x)^(x^x) at x=3
theorem evaluate_expression_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_three_l775_77504


namespace maximize_ab_l775_77597

theorem maximize_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ab + a + b = 1) : 
  ab ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end maximize_ab_l775_77597


namespace solve_y_l775_77543

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l775_77543


namespace calc_result_l775_77562

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l775_77562


namespace functional_equation_solution_l775_77547

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2) →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 :=
by
  sorry

end functional_equation_solution_l775_77547
