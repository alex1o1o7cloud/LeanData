import Mathlib

namespace arithmetic_mean_l2197_219736

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 6/11) :
  (a + b) / 2 = 75 / 154 :=
by
  sorry

end arithmetic_mean_l2197_219736


namespace max_geometric_progression_terms_l2197_219733

theorem max_geometric_progression_terms :
  ∀ a0 q : ℕ, (∀ k, a0 * q^k ≥ 100 ∧ a0 * q^k < 1000) →
  (∃ r s : ℕ, r > s ∧ q = r / s) →
  (∀ n, ∃ r s : ℕ, (r^n < 1000) ∧ ((r / s)^n < 10)) →
  n ≤ 5 :=
sorry

end max_geometric_progression_terms_l2197_219733


namespace probability_2x_less_y_equals_one_over_eight_l2197_219715

noncomputable def probability_2x_less_y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 3 * 1.5
  let area_rectangle : ℚ := 6 * 3
  area_triangle / area_rectangle

theorem probability_2x_less_y_equals_one_over_eight :
  probability_2x_less_y_in_rectangle = 1 / 8 :=
by
  sorry

end probability_2x_less_y_equals_one_over_eight_l2197_219715


namespace quadratic_roots_identity_l2197_219799

theorem quadratic_roots_identity (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) (hmn : m * n = -5) (hm_plus_n : m + n = -2) : m^2 + m * n + 2 * m = 0 :=
by {
    sorry
}

end quadratic_roots_identity_l2197_219799


namespace product_of_four_integers_negative_l2197_219706

theorem product_of_four_integers_negative {a b c d : ℤ}
  (h : a * b * c * d < 0) :
  (∃ n : ℕ, n ≤ 3 ∧ (n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0))) :=
sorry

end product_of_four_integers_negative_l2197_219706


namespace bhanu_spends_on_petrol_l2197_219744

-- Define the conditions as hypotheses
variable (income : ℝ)
variable (spend_on_rent : income * 0.7 * 0.14 = 98)

-- Define the theorem to prove
theorem bhanu_spends_on_petrol : (income * 0.3 = 300) :=
by
  sorry

end bhanu_spends_on_petrol_l2197_219744


namespace divisible_by_eight_l2197_219751

def expr (n : ℕ) : ℕ := 3^(4*n + 1) + 5^(2*n + 1)

theorem divisible_by_eight (n : ℕ) : expr n % 8 = 0 :=
  sorry

end divisible_by_eight_l2197_219751


namespace problem_statement_l2197_219757

theorem problem_statement
  (a b m n c : ℝ)
  (h1 : a = -b)
  (h2 : m * n = 1)
  (h3 : |c| = 3)
  : a + b + m * n - |c| = -2 := by
  sorry

end problem_statement_l2197_219757


namespace t_shirts_sold_l2197_219730

theorem t_shirts_sold (total_money : ℕ) (money_per_tshirt : ℕ) (n : ℕ) 
  (h1 : total_money = 2205) (h2 : money_per_tshirt = 9) (h3 : total_money = n * money_per_tshirt) : 
  n = 245 :=
by
  sorry

end t_shirts_sold_l2197_219730


namespace eggs_in_each_basket_l2197_219780

theorem eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 42 % n = 0) (h3 : n ≥ 5) :
  n = 6 :=
by sorry

end eggs_in_each_basket_l2197_219780


namespace hiker_displacement_l2197_219716

theorem hiker_displacement :
  let start_point := (0, 0)
  let move_east := (24, 0)
  let move_north := (0, 20)
  let move_west := (-7, 0)
  let move_south := (0, -9)
  let final_position := (start_point.1 + move_east.1 + move_west.1, start_point.2 + move_north.2 + move_south.2)
  let distance_from_start := Real.sqrt (final_position.1^2 + final_position.2^2)
  distance_from_start = Real.sqrt 410
:= by 
  sorry

end hiker_displacement_l2197_219716


namespace immortal_flea_can_visit_every_natural_l2197_219753

theorem immortal_flea_can_visit_every_natural :
  ∀ (k : ℕ), ∃ (jumps : ℕ → ℤ), (∀ n : ℕ, ∃ m : ℕ, jumps m = n) :=
by
  -- proof goes here
  sorry

end immortal_flea_can_visit_every_natural_l2197_219753


namespace problem_part1_problem_part2_l2197_219701

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2_l2197_219701


namespace score_analysis_l2197_219727

open Real

noncomputable def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]
def benchmark : ℝ := 85

theorem score_analysis :
  let highest_score := benchmark + List.maximum deviations
  let lowest_score := benchmark + List.minimum deviations
  let sum_deviations := List.sum deviations
  let average_deviation := sum_deviations / List.length deviations
  let average_score := benchmark + average_deviation
  highest_score = 97 ∧ lowest_score = 75 ∧ average_score = 84.9 :=
by
  sorry -- This is the placeholder for the proof

end score_analysis_l2197_219727


namespace calc_result_l2197_219718

theorem calc_result : (-2 * -3 + 2) = 8 := sorry

end calc_result_l2197_219718


namespace mos_to_ory_bus_encounter_l2197_219738

def encounter_buses (departure_time : Nat) (encounter_bus_time : Nat) (travel_time : Nat) : Nat := sorry

theorem mos_to_ory_bus_encounter :
  encounter_buses 0 30 5 = 10 :=
sorry

end mos_to_ory_bus_encounter_l2197_219738


namespace inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l2197_219756

theorem inequality_d_over_c_lt_d_plus_4_over_c_plus_4
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : c > d)
  (h3 : d > 0) :
  (d / c) < ((d + 4) / (c + 4)) :=
by
  sorry

end inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l2197_219756


namespace solution_set_of_inequality_l2197_219786

theorem solution_set_of_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ ((0 < x ∧ x < 1) ∨ (x > 1)) := 
by
  sorry

end solution_set_of_inequality_l2197_219786


namespace gear_ratio_l2197_219767

variable (a b c : ℕ) (ωG ωH ωI : ℚ)

theorem gear_ratio :
  (a * ωG = b * ωH) ∧ (b * ωH = c * ωI) ∧ (a * ωG = c * ωI) →
  ωG / ωH = bc / ac ∧ ωH / ωI = ac / ab ∧ ωG / ωI = bc / ab :=
by
  sorry

end gear_ratio_l2197_219767


namespace op_value_l2197_219719

noncomputable def op (a b c : ℝ) (k : ℤ) : ℝ :=
  b^2 - k * a^2 * c

theorem op_value : op 2 5 3 3 = -11 := by
  sorry

end op_value_l2197_219719


namespace find_cost_price_l2197_219747

-- Condition 1: The owner charges his customer 15% more than the cost price.
def selling_price (C : Real) : Real := C * 1.15

-- Condition 2: A customer paid Rs. 8325 for the computer table.
def paid_amount : Real := 8325

-- Define the cost price and its expected value
def cost_price : Real := 7239.13

-- The theorem to prove that the cost price matches the expected value
theorem find_cost_price : 
  ∃ C : Real, selling_price C = paid_amount ∧ C = cost_price :=
by
  sorry

end find_cost_price_l2197_219747


namespace find_k_multiple_l2197_219720

theorem find_k_multiple (a b k : ℕ) (h1 : a = b + 5) (h2 : a + b = 13) 
  (h3 : 3 * (a + 7) = k * (b + 7)) : k = 4 := sorry

end find_k_multiple_l2197_219720


namespace problem_l2197_219778

def f (x : ℝ) (a b : ℝ) := x^5 + a * x^3 + b * x - 2

-- We are given f(-2) = m
variables (a b m : ℝ)
theorem problem (h : f (-2) a b = m) : f 2 a b + f (-2) a b = -4 :=
by sorry

end problem_l2197_219778


namespace gcd_of_powers_l2197_219726

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2007 - 1) : 
  Nat.gcd m n = 131071 :=
by
  sorry

end gcd_of_powers_l2197_219726


namespace sum_cubes_eq_power_l2197_219721

/-- Given the conditions, prove that 1^3 + 2^3 + 3^3 + 4^3 = 10^2 -/
theorem sum_cubes_eq_power : 1 + 2 + 3 + 4 = 10 → 1^3 + 2^3 + 3^3 + 4^3 = 10^2 :=
by
  intro h
  sorry

end sum_cubes_eq_power_l2197_219721


namespace slope_parallel_l2197_219768

theorem slope_parallel (x y : ℝ) (m : ℝ) : (3:ℝ) * x - (6:ℝ) * y = (9:ℝ) → m = (1:ℝ) / (2:ℝ) :=
by
  sorry

end slope_parallel_l2197_219768


namespace ratio_of_colored_sheets_l2197_219792

theorem ratio_of_colored_sheets
    (total_sheets : ℕ)
    (num_binders : ℕ)
    (sheets_colored_by_justine : ℕ)
    (sheets_per_binder : ℕ)
    (h1 : total_sheets = 2450)
    (h2 : num_binders = 5)
    (h3 : sheets_colored_by_justine = 245)
    (h4 : sheets_per_binder = total_sheets / num_binders) :
    (sheets_colored_by_justine / Nat.gcd sheets_colored_by_justine sheets_per_binder) /
    (sheets_per_binder / Nat.gcd sheets_colored_by_justine sheets_per_binder) = 1 / 2 := by
  sorry

end ratio_of_colored_sheets_l2197_219792


namespace ufo_convention_attendees_l2197_219728

theorem ufo_convention_attendees 
  (F M : ℕ) 
  (h1 : F + M = 450) 
  (h2 : M = F + 26) : 
  M = 238 := 
sorry

end ufo_convention_attendees_l2197_219728


namespace gcd_g50_g52_l2197_219795

def g (x : ℤ) := x^2 - 2*x + 2022

theorem gcd_g50_g52 : Int.gcd (g 50) (g 52) = 2 := by
  sorry

end gcd_g50_g52_l2197_219795


namespace find_smallest_k_l2197_219789

theorem find_smallest_k : ∃ (k : ℕ), 64^k > 4^20 ∧ ∀ (m : ℕ), (64^m > 4^20) → m ≥ k := sorry

end find_smallest_k_l2197_219789


namespace solve_m_range_l2197_219739

-- Define the propositions
def p (m : ℝ) := m + 1 ≤ 0

def q (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Provide the Lean statement for the problem
theorem solve_m_range (m : ℝ) (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) :
  m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
sorry

end solve_m_range_l2197_219739


namespace line_symmetric_y_axis_eqn_l2197_219735

theorem line_symmetric_y_axis_eqn (x y : ℝ) : 
  (∀ x y : ℝ, x - y + 1 = 0 → x + y - 1 = 0) := 
sorry

end line_symmetric_y_axis_eqn_l2197_219735


namespace nancy_yearly_payment_l2197_219713

open Real

-- Define the monthly cost of the car insurance
def monthly_cost : ℝ := 80

-- Nancy's percentage contribution
def percentage : ℝ := 0.40

-- Calculate the monthly payment Nancy will make
def monthly_payment : ℝ := percentage * monthly_cost

-- Calculate the yearly payment Nancy will make
def yearly_payment : ℝ := 12 * monthly_payment

-- State the proof problem
theorem nancy_yearly_payment : yearly_payment = 384 :=
by
  -- Proof goes here
  sorry

end nancy_yearly_payment_l2197_219713


namespace min_sum_l2197_219790

namespace MinimumSum

theorem min_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hc : 98 * m = n^3) : m + n = 42 :=
sorry

end MinimumSum

end min_sum_l2197_219790


namespace no_day_income_is_36_l2197_219742

theorem no_day_income_is_36 : ∀ (n : ℕ), 3 * 3^(n-1) ≠ 36 :=
by
  intro n
  sorry

end no_day_income_is_36_l2197_219742


namespace max_gcd_b_n_b_n_plus_1_l2197_219708

noncomputable def b (n : ℕ) : ℚ := (2 ^ n - 1) / 3

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, Int.gcd (b n).num (b (n + 1)).num = 1 :=
by
  sorry

end max_gcd_b_n_b_n_plus_1_l2197_219708


namespace price_of_cheese_cookie_pack_l2197_219725

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end price_of_cheese_cookie_pack_l2197_219725


namespace average_percentage_score_is_71_l2197_219700

-- Define the number of students.
def number_of_students : ℕ := 150

-- Define the scores and their corresponding frequencies.
def scores_and_frequencies : List (ℕ × ℕ) :=
  [(100, 10), (95, 20), (85, 45), (75, 30), (65, 25), (55, 15), (45, 5)]

-- Define the total points scored by all students.
def total_points_scored : ℕ := 
  scores_and_frequencies.foldl (λ acc pair => acc + pair.1 * pair.2) 0

-- Define the average percentage score.
def average_score : ℚ := total_points_scored / number_of_students

-- Statement of the proof problem.
theorem average_percentage_score_is_71 :
  average_score = 71.0 := by
  sorry

end average_percentage_score_is_71_l2197_219700


namespace value_of_f_750_l2197_219750

theorem value_of_f_750 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y^2)
    (hf500 : f 500 = 4) :
    f 750 = 16 / 9 :=
sorry

end value_of_f_750_l2197_219750


namespace fixed_cost_is_50000_l2197_219766

-- Definition of conditions
def fixed_cost : ℕ := 50000
def books_sold : ℕ := 10000
def revenue_per_book : ℕ := 9 - 4

-- Theorem statement: Proving that the fixed cost of making books is $50,000
theorem fixed_cost_is_50000 (F : ℕ) (h : revenue_per_book * books_sold = F) : 
  F = fixed_cost :=
by sorry

end fixed_cost_is_50000_l2197_219766


namespace Lisa_income_percentage_J_M_combined_l2197_219797

variables (T M J L : ℝ)

-- Conditions as definitions
def Mary_income_eq_1p6_T (M T : ℝ) : Prop := M = 1.60 * T
def Tim_income_eq_0p5_J (T J : ℝ) : Prop := T = 0.50 * J
def Lisa_income_eq_1p3_M (L M : ℝ) : Prop := L = 1.30 * M
def Lisa_income_eq_0p75_J (L J : ℝ) : Prop := L = 0.75 * J

-- Theorem statement
theorem Lisa_income_percentage_J_M_combined (M T J L : ℝ)
  (h1 : Mary_income_eq_1p6_T M T)
  (h2 : Tim_income_eq_0p5_J T J)
  (h3 : Lisa_income_eq_1p3_M L M)
  (h4 : Lisa_income_eq_0p75_J L J) :
  (L / (M + J)) * 100 = 41.67 := 
sorry

end Lisa_income_percentage_J_M_combined_l2197_219797


namespace product_eval_at_3_l2197_219798

theorem product_eval_at_3 : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) = 720 := by
  sorry

end product_eval_at_3_l2197_219798


namespace sum_of_extreme_values_eq_four_l2197_219717

-- Given conditions in problem statement
variables (x y z : ℝ)
variables (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8)

-- Statement to be proved: sum of smallest and largest possible values of x is 4
theorem sum_of_extreme_values_eq_four : m + M = 4 :=
sorry

end sum_of_extreme_values_eq_four_l2197_219717


namespace least_number_to_subtract_l2197_219743

theorem least_number_to_subtract 
  (n : ℤ) 
  (h1 : 7 ∣ (90210 - n + 12)) 
  (h2 : 11 ∣ (90210 - n + 12)) 
  (h3 : 13 ∣ (90210 - n + 12)) 
  (h4 : 17 ∣ (90210 - n + 12)) 
  (h5 : 19 ∣ (90210 - n + 12)) : 
  n = 90198 :=
sorry

end least_number_to_subtract_l2197_219743


namespace arithmetic_sequence_sum_l2197_219704

-- Definitions based on conditions from step a
def first_term : ℕ := 1
def last_term : ℕ := 36
def num_terms : ℕ := 8

-- The problem statement in Lean 4
theorem arithmetic_sequence_sum :
  (num_terms / 2) * (first_term + last_term) = 148 := by
  sorry

end arithmetic_sequence_sum_l2197_219704


namespace length_of_wall_l2197_219785

theorem length_of_wall (side_mirror length_wall width_wall : ℕ) 
  (mirror_area wall_area : ℕ) (H1 : side_mirror = 54) 
  (H2 : mirror_area = side_mirror * side_mirror) 
  (H3 : wall_area = 2 * mirror_area) 
  (H4 : width_wall = 68) 
  (H5 : wall_area = length_wall * width_wall) : 
  length_wall = 86 :=
by
  sorry

end length_of_wall_l2197_219785


namespace original_square_side_length_l2197_219759

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l2197_219759


namespace ratio_P_to_A_l2197_219732

variable (M P A : ℕ) -- Define variables for Matthew, Patrick, and Alvin's egg rolls

theorem ratio_P_to_A (hM : M = 6) (hM_to_P : M = 3 * P) (hA : A = 4) : P / A = 1 / 2 := by
  sorry

end ratio_P_to_A_l2197_219732


namespace unique_line_intercept_l2197_219763

noncomputable def is_positive_integer (n : ℕ) : Prop := n > 0
noncomputable def is_prime (n : ℕ) : Prop := n = 2 ∨ (n > 2 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem unique_line_intercept (a b : ℕ) :
  ((is_positive_integer a) ∧ (is_prime b) ∧ (6 * b + 5 * a = a * b)) ↔ (a = 11 ∧ b = 11) :=
by
  sorry

end unique_line_intercept_l2197_219763


namespace probability_5800_in_three_spins_l2197_219783

def spinner_labels : List String := ["Bankrupt", "$600", "$1200", "$4000", "$800", "$2000", "$150"]

def total_outcomes (spins : Nat) : Nat :=
  let segments := spinner_labels.length
  segments ^ spins

theorem probability_5800_in_three_spins :
  (6 / total_outcomes 3 : ℚ) = 6 / 343 :=
by
  sorry

end probability_5800_in_three_spins_l2197_219783


namespace time_to_shovel_snow_l2197_219712

noncomputable def initial_rate : ℕ := 30
noncomputable def decay_rate : ℕ := 2
noncomputable def driveway_width : ℕ := 6
noncomputable def driveway_length : ℕ := 15
noncomputable def snow_depth : ℕ := 2

noncomputable def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

def snow_shoveling_time (initial_rate decay_rate total_volume : ℕ) : ℕ :=
-- Function to compute the time needed, assuming definition provided
sorry

theorem time_to_shovel_snow 
  : snow_shoveling_time initial_rate decay_rate total_snow_volume = 8 :=
sorry

end time_to_shovel_snow_l2197_219712


namespace bridge_length_at_least_200_l2197_219705

theorem bridge_length_at_least_200 :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_secs : ℝ),
  length_train = 200 ∧ speed_kmph = 32 ∧ time_secs = 20 →
  ∃ l : ℝ, l ≥ length_train :=
by
  sorry

end bridge_length_at_least_200_l2197_219705


namespace tracy_two_dogs_food_consumption_l2197_219793

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l2197_219793


namespace percentage_employed_females_is_16_l2197_219746

/- 
  In Town X, the population is divided into three age groups: 18-34, 35-54, and 55+.
  For each age group, the percentage of the employed population is 64%, and the percentage of employed males is 48%.
  We need to prove that the percentage of employed females in each age group is 16%.
-/

theorem percentage_employed_females_is_16
  (percentage_employed_population : ℝ)
  (percentage_employed_males : ℝ)
  (h1 : percentage_employed_population = 0.64)
  (h2 : percentage_employed_males = 0.48) :
  percentage_employed_population - percentage_employed_males = 0.16 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end percentage_employed_females_is_16_l2197_219746


namespace total_stoppage_time_l2197_219779

theorem total_stoppage_time (stop1 stop2 stop3 : ℕ) (h1 : stop1 = 5)
  (h2 : stop2 = 8) (h3 : stop3 = 10) : stop1 + stop2 + stop3 = 23 :=
sorry

end total_stoppage_time_l2197_219779


namespace remainder_division_l2197_219770

/-- A number when divided by a certain divisor left a remainder, 
when twice the number was divided by the same divisor, the remainder was 112. 
The divisor is 398.
Prove that the remainder when the original number is divided by the divisor is 56. -/
theorem remainder_division (N R : ℤ) (D : ℕ) (Q Q' : ℤ)
  (hD : D = 398)
  (h1 : N = D * Q + R)
  (h2 : 2 * N = D * Q' + 112) :
  R = 56 :=
sorry

end remainder_division_l2197_219770


namespace fried_chicken_total_l2197_219769

-- The Lean 4 statement encapsulates the problem conditions and the correct answer
theorem fried_chicken_total :
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  pau_initial * another_set = 20 :=
by
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  show pau_initial * another_set = 20
  sorry

end fried_chicken_total_l2197_219769


namespace apples_difference_l2197_219752

theorem apples_difference
    (adam_apples : ℕ)
    (jackie_apples : ℕ)
    (h_adam : adam_apples = 10)
    (h_jackie : jackie_apples = 2) :
    adam_apples - jackie_apples = 8 :=
by
    sorry

end apples_difference_l2197_219752


namespace find_a_if_y_is_even_l2197_219765

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l2197_219765


namespace count_multiples_of_4_between_300_and_700_l2197_219796

noncomputable def num_multiples_of_4_in_range (a b : ℕ) : ℕ :=
  (b - (b % 4) - (a - (a % 4) + 4)) / 4 + 1

theorem count_multiples_of_4_between_300_and_700 : 
  num_multiples_of_4_in_range 301 699 = 99 := by
  sorry

end count_multiples_of_4_between_300_and_700_l2197_219796


namespace number_of_sides_l2197_219781

-- Define the given conditions as Lean definitions

def exterior_angle := 72
def sum_of_exterior_angles := 360

-- Now state the theorem based on these conditions

theorem number_of_sides (n : ℕ) (h1 : exterior_angle = 72) (h2 : sum_of_exterior_angles = 360) : n = 5 :=
by
  sorry

end number_of_sides_l2197_219781


namespace sequence_a_b_10_l2197_219772

theorem sequence_a_b_10 (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : 
  a^10 + b^10 = 123 := 
sorry

end sequence_a_b_10_l2197_219772


namespace brian_expenses_l2197_219782

def cost_apples_per_bag : ℕ := 14
def cost_kiwis : ℕ := 10
def cost_bananas : ℕ := cost_kiwis / 2
def subway_fare_one_way : ℕ := 350
def maximum_apples : ℕ := 24

theorem brian_expenses : 
  cost_kiwis + cost_bananas + (cost_apples_per_bag * (maximum_apples / 12)) + (subway_fare_one_way * 2) = 50 := by
sorry

end brian_expenses_l2197_219782


namespace find_a_bi_c_l2197_219729

theorem find_a_bi_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_eq : (a - (b : ℤ)*I)^2 + c = 13 - 8*I) :
  a = 2 ∧ b = 2 ∧ c = 13 :=
by
  sorry

end find_a_bi_c_l2197_219729


namespace expand_and_simplify_l2197_219745

theorem expand_and_simplify : ∀ x : ℝ, (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 :=
by
  intro x
  sorry

end expand_and_simplify_l2197_219745


namespace geometric_sequence_sum_l2197_219773

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_common_ratio : ∀ n, a (n + 1) = 2 * a n)
    (h_sum : a 1 + a 2 + a 3 = 21) : a 3 + a 4 + a 5 = 84 :=
sorry

end geometric_sequence_sum_l2197_219773


namespace yuna_average_score_l2197_219724

theorem yuna_average_score (avg_may_june : ℕ) (score_july : ℕ) (h1 : avg_may_june = 84) (h2 : score_july = 96) :
  (avg_may_june * 2 + score_july) / 3 = 88 := by
  sorry

end yuna_average_score_l2197_219724


namespace max_value_correct_l2197_219731

noncomputable def max_value (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_correct (x y : ℝ) (h : x + y = 5) : max_value x y h ≤ 22884 :=
  sorry

end max_value_correct_l2197_219731


namespace total_logs_combined_l2197_219723

theorem total_logs_combined 
  (a1 l1 a2 l2 : ℕ) 
  (n1 n2 : ℕ) 
  (S1 S2 : ℕ) 
  (h1 : a1 = 15) 
  (h2 : l1 = 10) 
  (h3 : n1 = 6) 
  (h4 : S1 = n1 * (a1 + l1) / 2) 
  (h5 : a2 = 9) 
  (h6 : l2 = 5) 
  (h7 : n2 = 5) 
  (h8 : S2 = n2 * (a2 + l2) / 2) : 
  S1 + S2 = 110 :=
by {
  sorry
}

end total_logs_combined_l2197_219723


namespace geometric_series_sum_l2197_219775

noncomputable def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum (2/3) (2/3) 10 = 116050 / 59049 :=
by
  sorry

end geometric_series_sum_l2197_219775


namespace eqD_is_linear_l2197_219748

-- Definitions for the given equations
def eqA (x y : ℝ) : Prop := 3 * x - 2 * y = 1
def eqB (x : ℝ) : Prop := 1 + (1 / x) = x
def eqC (x : ℝ) : Prop := x^2 = 9
def eqD (x : ℝ) : Prop := 2 * x - 3 = 5

-- Definition of a linear equation in one variable
def isLinear (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x : ℝ, eq x ↔ a * x + b = c)

-- Theorem stating that eqD is a linear equation
theorem eqD_is_linear : isLinear eqD :=
  sorry

end eqD_is_linear_l2197_219748


namespace problem_l2197_219788

noncomputable def a : ℝ := (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3)
noncomputable def b : ℝ := (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3)

theorem problem :
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end problem_l2197_219788


namespace eq_exponents_l2197_219771

theorem eq_exponents (m n : ℤ) : ((5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n) → (m = 0 ∧ n = 0) :=
by
  sorry

end eq_exponents_l2197_219771


namespace solution_set_of_inequality_l2197_219755

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
-- f(x) is symmetric about the origin
variable (symmetric_f : ∀ x, f (-x) = -f x)
-- f(2) = 2
variable (f_at_2 : f 2 = 2)
-- For any 0 < x2 < x1, the slope condition holds
variable (slope_cond : ∀ x1 x2, 0 < x2 ∧ x2 < x1 → (f x1 - f x2) / (x1 - x2) < 1)

theorem solution_set_of_inequality :
  {x : ℝ | f x - x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end solution_set_of_inequality_l2197_219755


namespace initial_pens_l2197_219777

-- Conditions as definitions
def initial_books := 108
def books_after_sale := 66
def books_sold := 42
def pens_after_sale := 59

-- Theorem statement proving the initial number of pens
theorem initial_pens:
  initial_books - books_after_sale = books_sold →
  ∃ (P : ℕ), P - pens_sold = pens_after_sale ∧ (P = 101) :=
by
  sorry

end initial_pens_l2197_219777


namespace max_pieces_from_cake_l2197_219711

theorem max_pieces_from_cake (large_cake_area small_piece_area : ℕ) 
  (h_large_cake : large_cake_area = 15 * 15) 
  (h_small_piece : small_piece_area = 5 * 5) :
  large_cake_area / small_piece_area = 9 := 
by
  sorry

end max_pieces_from_cake_l2197_219711


namespace find_a_l2197_219760

-- Define sets A and B
def A : Set ℕ := {1, 2, 5}
def B (a : ℕ) : Set ℕ := {2, a}

-- Given condition: A ∪ B = {1, 2, 3, 5}
def union_condition (a : ℕ) : Prop := A ∪ B a = {1, 2, 3, 5}

-- Theorem we want to prove
theorem find_a (a : ℕ) : union_condition a → a = 3 :=
by
  intro h
  sorry

end find_a_l2197_219760


namespace common_tangent_and_inequality_l2197_219784

noncomputable def f (x : ℝ) := Real.log (1 + x)
noncomputable def g (x : ℝ) := x - (1 / 2) * x^2 + (1 / 3) * x^3

theorem common_tangent_and_inequality :
  -- Condition: common tangent at (0, 0)
  (∀ x, deriv f x = deriv g x) →
  -- Condition: values of a and b found to be 0 and 1 respectively
  (∀ x, f x ≤ g x) :=
by
  intro h
  sorry

end common_tangent_and_inequality_l2197_219784


namespace vanessa_score_l2197_219749

-- Define the total score of the team
def total_points : ℕ := 60

-- Define the score of the seven other players
def other_players_points : ℕ := 7 * 4

-- Mathematics statement for proof
theorem vanessa_score : total_points - other_players_points = 32 :=
by
    sorry

end vanessa_score_l2197_219749


namespace fraction_money_left_zero_l2197_219703

-- Defining variables and conditions
variables {m c : ℝ} -- m: total money, c: total cost of CDs

-- Condition under the problem statement
def uses_one_fourth_of_money_to_buy_one_fourth_of_CDs (m c : ℝ) := (1 / 4) * m = (1 / 4) * c

-- The conjecture to be proven
theorem fraction_money_left_zero 
  (h: uses_one_fourth_of_money_to_buy_one_fourth_of_CDs m c) 
  (h_eq: c = m) : 
  (m - c) / m = 0 := 
by
  sorry

end fraction_money_left_zero_l2197_219703


namespace problem_f_17_l2197_219734

/-- Assume that f(1) = 0 and f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1) for all natural numbers m and n.
    Prove that f(17) = 4832.
-/
theorem problem_f_17 (f : ℕ → ℤ) 
  (h1 : f 1 = 0) 
  (h_func : ∀ m n : ℕ, f (m + n) = f m + f n + 4 * (9 * m * n - 1)) 
  : f 17 = 4832 := 
sorry

end problem_f_17_l2197_219734


namespace circle_radius_l2197_219754

theorem circle_radius (m : ℝ) (h : 2 * 1 + (-m / 2) = 0) :
  let radius := 1 / 2 * Real.sqrt (4 + m ^ 2 + 16)
  radius = 3 :=
by
  sorry

end circle_radius_l2197_219754


namespace part1_part2_axis_of_symmetry_part2_center_of_symmetry_l2197_219762

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : perpendicular (m x) (n x)) :
  x = π / 2 ∨ x = 3 * π / 4 :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = f (2 * c - x) ∧ 
    ((2 * x + π / 4) = k * π + π / 2 → x = k * π / 2 + π / 8) :=
sorry

theorem part2_center_of_symmetry (k : ℤ) :
  ∃ x c : ℝ, f x = 1 ∧ ((2 * x + π / 4) = k * π → x = k * π / 2 - π / 8) :=
sorry

end part1_part2_axis_of_symmetry_part2_center_of_symmetry_l2197_219762


namespace solution_l2197_219774

-- Definitions for perpendicular and parallel relations
def perpendicular (a b : Type) : Prop := sorry -- Abstraction for perpendicularity
def parallel (a b : Type) : Prop := sorry -- Abstraction for parallelism

-- Here we define x, y, z as variables
variables {x y : Type} {z : Type}

-- Conditions for Case 2
def case2_lines_plane (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Conditions for Case 3
def case3_planes_line (x y : Type) (z : Type) := 
  (perpendicular x z) ∧ (perpendicular y z) → (parallel x y)

-- Theorem statement combining both cases
theorem solution : case2_lines_plane x y z ∧ case3_planes_line x y z := 
sorry

end solution_l2197_219774


namespace solve_quadratic_eq_l2197_219707

theorem solve_quadratic_eq {x : ℝ} (h : x^2 - 5*x + 6 = 0) : x = 2 ∨ x = 3 :=
sorry

end solve_quadratic_eq_l2197_219707


namespace zeros_of_f_x_minus_1_l2197_219722

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_f_x_minus_1 :
  (f (0 - 1) = 0) ∧ (f (2 - 1) = 0) :=
by
  sorry

end zeros_of_f_x_minus_1_l2197_219722


namespace length_of_field_l2197_219710

-- Define the known conditions
def width := 50
def total_distance_run := 1800
def num_laps := 6

-- Define the problem statement
theorem length_of_field :
  ∃ L : ℕ, 6 * (2 * (L + width)) = total_distance_run ∧ L = 100 :=
by
  sorry

end length_of_field_l2197_219710


namespace fraction_irreducible_l2197_219737

theorem fraction_irreducible (n : ℤ) : gcd (2 * n ^ 2 + 9 * n - 17) (n + 6) = 1 := by
  sorry

end fraction_irreducible_l2197_219737


namespace naomi_total_time_l2197_219740

-- Definitions
def time_to_parlor : ℕ := 60
def speed_ratio : ℕ := 2 -- because her returning speed is half of the going speed
def first_trip_delay : ℕ := 15
def coffee_break : ℕ := 10
def second_trip_delay : ℕ := 20
def detour_time : ℕ := 30

-- Calculate total round trip times
def first_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + first_trip_delay + coffee_break
def second_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + second_trip_delay + detour_time

-- Hypothesis
def total_round_trip_time : ℕ := first_round_trip_time + second_round_trip_time

-- Main theorem statement
theorem naomi_total_time : total_round_trip_time = 435 := by
  sorry

end naomi_total_time_l2197_219740


namespace isabella_hair_growth_l2197_219791

def initial_hair_length : ℝ := 18
def final_hair_length : ℝ := 24
def hair_growth : ℝ := final_hair_length - initial_hair_length

theorem isabella_hair_growth : hair_growth = 6 := by
  sorry

end isabella_hair_growth_l2197_219791


namespace isabella_haircut_length_l2197_219741

-- Define the original length of Isabella's hair.
def original_length : ℕ := 18

-- Define the length of hair cut off.
def cut_off_length : ℕ := 9

-- The length of Isabella's hair after the haircut.
def length_after_haircut : ℕ := original_length - cut_off_length

-- Statement of the theorem we want to prove.
theorem isabella_haircut_length : length_after_haircut = 9 :=
by
  sorry

end isabella_haircut_length_l2197_219741


namespace max_boxes_in_warehouse_l2197_219787

def warehouse_length : ℕ := 50
def warehouse_width : ℕ := 30
def warehouse_height : ℕ := 5
def box_edge_length : ℕ := 2

theorem max_boxes_in_warehouse : (warehouse_length / box_edge_length) * (warehouse_width / box_edge_length) * (warehouse_height / box_edge_length) = 750 := 
by
  sorry

end max_boxes_in_warehouse_l2197_219787


namespace open_box_volume_l2197_219794

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l2197_219794


namespace ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l2197_219714

theorem ab_cd_ge_ac_bd_squared (a b c d : ℝ) : ((a^2 + b^2) * (c^2 + d^2)) ≥ (a * c + b * d)^2 := 
by sorry

theorem eq_condition_ad_eq_bc (a b c d : ℝ) (h : a * d = b * c) : ((a^2 + b^2) * (c^2 + d^2)) = (a * c + b * d)^2 := 
by sorry

end ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l2197_219714


namespace stormi_cars_washed_l2197_219764

-- Definitions based on conditions
def cars_earning := 10
def lawns_number := 2
def lawn_earning := 13
def bicycle_cost := 80
def needed_amount := 24

-- Auxiliary calculations
def lawns_total_earning := lawns_number * lawn_earning
def already_earning := bicycle_cost - needed_amount
def cars_total_earning := already_earning - lawns_total_earning

-- Main problem statement
theorem stormi_cars_washed : (cars_total_earning / cars_earning) = 3 :=
  by sorry

end stormi_cars_washed_l2197_219764


namespace min_value_of_objective_function_l2197_219776

theorem min_value_of_objective_function : 
  ∃ (x y : ℝ), 
    (2 * x + y - 2 ≥ 0) ∧ 
    (x - 2 * y + 4 ≥ 0) ∧ 
    (x - 1 ≤ 0) ∧ 
    (∀ (u v: ℝ), 
      (2 * u + v - 2 ≥ 0) → 
      (u - 2 * v + 4 ≥ 0) → 
      (u - 1 ≤ 0) → 
      (3 * u + 2 * v ≥ 3)) :=
  sorry

end min_value_of_objective_function_l2197_219776


namespace age_ratio_l2197_219702

/-- 
Axiom: Kareem's age is 42 and his son's age is 14. 
-/
axiom Kareem_age : ℕ
axiom Son_age : ℕ

/-- 
Conditions: 
  - Kareem's age after 10 years plus his son's age after 10 years equals 76.
  - Kareem's current age is 42.
  - His son's current age is 14.
-/
axiom age_condition : Kareem_age + 10 + Son_age + 10 = 76
axiom Kareem_current_age : Kareem_age = 42
axiom Son_current_age : Son_age = 14

/-- 
Theorem: The ratio of Kareem's age to his son's age is 3:1.
-/
theorem age_ratio : Kareem_age / Son_age = 3 / 1 := by {
  -- Proof skipped
  sorry 
}

end age_ratio_l2197_219702


namespace number_of_true_propositions_is_one_l2197_219758

theorem number_of_true_propositions_is_one :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (¬ (∀ (p q : Prop), ¬ (p ∧ q) → (¬ p ∧ ¬ q))) ∧
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) →
  1 = 1 :=
by
  sorry

end number_of_true_propositions_is_one_l2197_219758


namespace maximize_area_l2197_219709

noncomputable def max_area : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area (l w : ℝ) (h1 : 2 * l + 2 * w = 400) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  (l * w ≤ 10000) :=
sorry

end maximize_area_l2197_219709


namespace common_sale_days_in_july_l2197_219761

def BookstoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (d % 4 = 0)

def ShoeStoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (∃ k : ℕ, d = 2 + k * 7)

theorem common_sale_days_in_july : ∃! d, (BookstoreSaleDays d) ∧ (ShoeStoreSaleDays d) :=
by {
  sorry
}

end common_sale_days_in_july_l2197_219761
