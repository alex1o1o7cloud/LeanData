import Mathlib

namespace solve_for_x_l2112_211228

theorem solve_for_x (x : ℕ) : x * 12 = 173 * 240 → x = 3460 :=
by
  sorry

end solve_for_x_l2112_211228


namespace cylinder_sphere_ratio_l2112_211233

theorem cylinder_sphere_ratio (r R : ℝ) (h : 8 * r^2 = 4 * R^2) : R / r = Real.sqrt 2 :=
by
  sorry

end cylinder_sphere_ratio_l2112_211233


namespace A_serves_on_50th_week_is_Friday_l2112_211244

-- Define the people involved in the rotation
inductive Person
| A | B | C | D | E | F

open Person

-- Define the function that computes the day A serves on given the number of weeks
def day_A_serves (weeks : ℕ) : ℕ :=
  let days := weeks * 7
  (days % 6 + 0) % 7 -- 0 is the offset for the initial day when A serves (Sunday)

theorem A_serves_on_50th_week_is_Friday :
  day_A_serves 50 = 5 :=
by
  -- We provide the proof here
  sorry

end A_serves_on_50th_week_is_Friday_l2112_211244


namespace macy_hit_ball_50_times_l2112_211274

-- Definitions and conditions
def token_pitches : ℕ := 15
def macy_tokens : ℕ := 11
def piper_tokens : ℕ := 17
def piper_hits : ℕ := 55
def missed_pitches : ℕ := 315

-- Calculation based on conditions
def total_pitches : ℕ := (macy_tokens + piper_tokens) * token_pitches
def total_hits : ℕ := total_pitches - missed_pitches
def macy_hits : ℕ := total_hits - piper_hits

-- Prove that Macy hit 50 times
theorem macy_hit_ball_50_times : macy_hits = 50 := 
by
  sorry

end macy_hit_ball_50_times_l2112_211274


namespace minimum_value_f_l2112_211236

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem minimum_value_f :
  ∃ x > 0, (∀ y > 0, f x ≤ f y) ∧ f x = 1 :=
sorry

end minimum_value_f_l2112_211236


namespace solve_problem_l2112_211260

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem solve_problem : spadesuit 3 (spadesuit 5 (spadesuit 8 11)) = 1 :=
by
  -- Proof is omitted
  sorry

end solve_problem_l2112_211260


namespace difference_of_squares_l2112_211254

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 8) : x^2 - y^2 = 80 :=
by
  sorry

end difference_of_squares_l2112_211254


namespace three_digit_solutions_exist_l2112_211248

theorem three_digit_solutions_exist :
  ∃ (x y z : ℤ), 100 ≤ x ∧ x ≤ 999 ∧ 
                 100 ≤ y ∧ y ≤ 999 ∧
                 100 ≤ z ∧ z ≤ 999 ∧
                 17 * x + 15 * y - 28 * z = 61 ∧
                 19 * x - 25 * y + 12 * z = 31 :=
by
    sorry

end three_digit_solutions_exist_l2112_211248


namespace sequence_may_or_may_not_be_arithmetic_l2112_211277

theorem sequence_may_or_may_not_be_arithmetic (a : ℕ → ℕ) 
  (h1 : a 0 = 1) (h2 : a 1 = 2) (h3 : a 2 = 3) 
  (h4 : a 3 = 4) (h5 : a 4 = 5) : 
  ¬(∀ n, a (n + 1) - a n = 1) → 
  (∀ n, a (n + 1) - a n = 1) ∨ ¬(∀ n, a (n + 1) - a n = 1) :=
by
  sorry

end sequence_may_or_may_not_be_arithmetic_l2112_211277


namespace dice_roll_probability_is_correct_l2112_211290

/-- Define the probability calculation based on conditions of the problem. --/
def dice_rolls_probability_diff_by_two (successful_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  successful_outcomes / total_outcomes

/-- Given the problem conditions, there are 8 successful outcomes and 36 total outcomes. --/
theorem dice_roll_probability_is_correct :
  dice_rolls_probability_diff_by_two 8 36 = 2 / 9 :=
by
  sorry

end dice_roll_probability_is_correct_l2112_211290


namespace num_tickets_bought_l2112_211241

-- Defining the cost and discount conditions
def ticket_cost : ℝ := 40
def discount_rate : ℝ := 0.05
def total_paid : ℝ := 476
def base_tickets : ℕ := 10

-- Definition to calculate the cost of the first 10 tickets
def cost_first_10_tickets : ℝ := base_tickets * ticket_cost
-- Definition of the discounted price for tickets exceeding 10
def discounted_ticket_cost : ℝ := ticket_cost * (1 - discount_rate)
-- Definition of the total cost for the tickets exceeding 10
def cost_discounted_tickets (num_tickets_exceeding_10 : ℕ) : ℝ := num_tickets_exceeding_10 * discounted_ticket_cost
-- Total amount spent on the tickets exceeding 10
def amount_spent_on_discounted_tickets : ℝ := total_paid - cost_first_10_tickets

-- Main theorem statement proving the total number of tickets Mr. Benson bought
theorem num_tickets_bought : ∃ x : ℕ, x = base_tickets + (amount_spent_on_discounted_tickets / discounted_ticket_cost) ∧ x = 12 := 
by
  sorry

end num_tickets_bought_l2112_211241


namespace machines_complete_order_l2112_211286

theorem machines_complete_order (h1 : ℝ) (h2 : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ)
  (h1_def : h1 = 9)
  (h2_def : h2 = 8)
  (rate1_def : rate1 = 1 / h1)
  (rate2_def : rate2 = 1 / h2)
  (combined_rate : ℝ := rate1 + rate2) :
  time = 72 / 17 :=
by
  sorry

end machines_complete_order_l2112_211286


namespace trivia_team_average_points_l2112_211225

noncomputable def average_points_per_member (total_members didn't_show_up total_points : ℝ) : ℝ :=
  total_points / (total_members - didn't_show_up)

@[simp]
theorem trivia_team_average_points :
  let total_members := 8.0
  let didn't_show_up := 3.5
  let total_points := 12.5
  ∃ avg_points, avg_points = 2.78 ∧ avg_points = average_points_per_member total_members didn't_show_up total_points :=
by
  sorry

end trivia_team_average_points_l2112_211225


namespace vertices_form_vertical_line_l2112_211264

theorem vertices_form_vertical_line (a b k d : ℝ) (ha : 0 < a) (hk : 0 < k) :
  ∃ x, ∀ t : ℝ, ∃ y, (x = -b / (2 * a) ∧ y = - (b^2) / (4 * a) + k * t + d) :=
sorry

end vertices_form_vertical_line_l2112_211264


namespace average_weight_of_all_girls_l2112_211282

theorem average_weight_of_all_girls 
    (avg_weight_group1 : ℝ) (avg_weight_group2 : ℝ) 
    (num_girls_group1 : ℕ) (num_girls_group2 : ℕ) 
    (h1 : avg_weight_group1 = 50.25) 
    (h2 : avg_weight_group2 = 45.15) 
    (h3 : num_girls_group1 = 16) 
    (h4 : num_girls_group2 = 8) : 
    (avg_weight_group1 * num_girls_group1 + avg_weight_group2 * num_girls_group2) / (num_girls_group1 + num_girls_group2) = 48.55 := 
by 
    sorry

end average_weight_of_all_girls_l2112_211282


namespace num_partition_sets_correct_l2112_211252

noncomputable def num_partition_sets (n : ℕ) : ℕ :=
  2^(n-1) - 1

theorem num_partition_sets_correct (n : ℕ) (hn : n ≥ 2) : 
  num_partition_sets n = 2^(n-1) - 1 := 
by sorry

end num_partition_sets_correct_l2112_211252


namespace savings_after_four_weeks_l2112_211215

noncomputable def hourly_wage (name : String) : ℝ :=
  match name with
  | "Robby" | "Jaylen" | "Miranda" => 10
  | "Alex" => 12
  | "Beth" => 15
  | "Chris" => 20
  | _ => 0

noncomputable def daily_hours (name : String) : ℝ :=
  match name with
  | "Robby" | "Miranda" => 10
  | "Jaylen" => 8
  | "Alex" => 6
  | "Beth" => 4
  | "Chris" => 3
  | _ => 0

noncomputable def saving_rate (name : String) : ℝ :=
  match name with
  | "Robby" => 2/5
  | "Jaylen" => 3/5
  | "Miranda" => 1/2
  | "Alex" => 1/3
  | "Beth" => 1/4
  | "Chris" => 3/4
  | _ => 0

noncomputable def weekly_earning (name : String) : ℝ :=
  hourly_wage name * daily_hours name * 5

noncomputable def weekly_saving (name : String) : ℝ :=
  weekly_earning name * saving_rate name

noncomputable def combined_savings : ℝ :=
  4 * (weekly_saving "Robby" + 
       weekly_saving "Jaylen" + 
       weekly_saving "Miranda" + 
       weekly_saving "Alex" + 
       weekly_saving "Beth" + 
       weekly_saving "Chris")

theorem savings_after_four_weeks :
  combined_savings = 4440 :=
by
  sorry

end savings_after_four_weeks_l2112_211215


namespace solveNumberOfWaysToChooseSeats_l2112_211250

/--
Define the problem of professors choosing their seats among 9 chairs with specific constraints.
-/
noncomputable def numberOfWaysToChooseSeats : ℕ :=
  let totalChairs := 9
  let endChairChoices := 2 * (7 * (7 - 2))  -- (2 end chairs, 7 for 2nd prof, 5 for 3rd prof)
  let middleChairChoices := 7 * (6 * (6 - 2))  -- (7 non-end chairs, 6 for 2nd prof, 4 for 3rd prof)
  endChairChoices + middleChairChoices

/--
The final result should be 238
-/
theorem solveNumberOfWaysToChooseSeats : numberOfWaysToChooseSeats = 238 := by
  sorry

end solveNumberOfWaysToChooseSeats_l2112_211250


namespace arithmetic_mean_25_41_50_l2112_211229

theorem arithmetic_mean_25_41_50 :
  (25 + 41 + 50) / 3 = 116 / 3 := by
  sorry

end arithmetic_mean_25_41_50_l2112_211229


namespace value_of_x_l2112_211224

theorem value_of_x (x : ℚ) (h : (3 * x + 4) / 7 = 15) : x = 101 / 3 :=
by
  sorry

end value_of_x_l2112_211224


namespace work_done_by_force_l2112_211234

noncomputable def displacement (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem work_done_by_force :
  let F := (5, 2)
  let A := (-1, 3)
  let B := (2, 6)
  let AB := displacement A B
  dot_product F AB = 21 := by
  sorry

end work_done_by_force_l2112_211234


namespace problem_solution_l2112_211289

theorem problem_solution (N : ℚ) (h : (4/5) * (3/8) * N = 24) : 2.5 * N = 200 :=
by {
  sorry
}

end problem_solution_l2112_211289


namespace quadratic_factorization_sum_l2112_211267

theorem quadratic_factorization_sum (d e f : ℤ) (h1 : ∀ x, x^2 + 18 * x + 80 = (x + d) * (x + e)) 
                                     (h2 : ∀ x, x^2 - 20 * x + 96 = (x - e) * (x - f)) : 
                                     d + e + f = 30 :=
by
  sorry

end quadratic_factorization_sum_l2112_211267


namespace value_of_expression_l2112_211205

theorem value_of_expression (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 9 - 4 * x^2 - 6 * x = 7 := by
  sorry

end value_of_expression_l2112_211205


namespace twice_x_minus_3_l2112_211291

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l2112_211291


namespace find_y_l2112_211275

theorem find_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end find_y_l2112_211275


namespace number_of_n_l2112_211271

theorem number_of_n (n : ℕ) (h1 : n ≤ 1000) (h2 : ∃ k : ℕ, 18 * n = k^2) : 
  ∃ K : ℕ, K = 7 :=
sorry

end number_of_n_l2112_211271


namespace avg_weight_b_c_l2112_211270

variables (A B C : ℝ)

-- Given Conditions
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := B = 37

-- Statement to prove
theorem avg_weight_b_c 
  (h1 : condition1 A B C)
  (h2 : condition2 A B)
  (h3 : condition3 B) : 
  (B + C) / 2 = 46 :=
sorry

end avg_weight_b_c_l2112_211270


namespace total_rent_of_pasture_l2112_211298

theorem total_rent_of_pasture 
  (oxen_A : ℕ) (months_A : ℕ) (oxen_B : ℕ) (months_B : ℕ)
  (oxen_C : ℕ) (months_C : ℕ) (share_C : ℕ) (total_rent : ℕ) :
  oxen_A = 10 →
  months_A = 7 →
  oxen_B = 12 →
  months_B = 5 →
  oxen_C = 15 →
  months_C = 3 →
  share_C = 72 →
  total_rent = 280 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2 hC3
  sorry

end total_rent_of_pasture_l2112_211298


namespace probability_both_selected_l2112_211204

theorem probability_both_selected (p_ram : ℚ) (p_ravi : ℚ) (h_ram : p_ram = 5/7) (h_ravi : p_ravi = 1/5) : 
  (p_ram * p_ravi = 1/7) := 
by
  sorry

end probability_both_selected_l2112_211204


namespace noemi_initial_amount_l2112_211297

theorem noemi_initial_amount : 
  ∀ (rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount : ℕ), 
    rouletteLoss = 600 → 
    blackjackLoss = 800 → 
    pokerLoss = 400 → 
    baccaratLoss = 700 → 
    remainingAmount = 1500 → 
    initialAmount = rouletteLoss + blackjackLoss + pokerLoss + baccaratLoss + remainingAmount →
    initialAmount = 4000 :=
by
  intros rouletteLoss blackjackLoss pokerLoss baccaratLoss remainingAmount initialAmount
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  exact h6

end noemi_initial_amount_l2112_211297


namespace number_composite_l2112_211220

theorem number_composite (n : ℕ) : 
  n = 10^(2^1974 + 2^1000 - 1) + 1 →
  ∃ a b : ℕ, 1 < a ∧ a < n ∧ n = a * b :=
by sorry

end number_composite_l2112_211220


namespace circle_radius_l2112_211283

theorem circle_radius (x y d : ℝ) (h₁ : x = π * r^2) (h₂ : y = 2 * π * r) (h₃ : d = 2 * r) (h₄ : x + y + d = 164 * π) : r = 10 :=
by sorry

end circle_radius_l2112_211283


namespace sum_circumferences_of_small_circles_l2112_211240

theorem sum_circumferences_of_small_circles (R : ℝ) (n : ℕ) (hR : R > 0) (hn : n > 0) :
  let original_circumference := 2 * Real.pi * R
  let part_length := original_circumference / n
  let small_circle_radius := part_length / Real.pi
  let small_circle_circumference := 2 * Real.pi * small_circle_radius
  let total_circumference := n * small_circle_circumference
  total_circumference = 2 * Real.pi ^ 2 * R :=
by {
  sorry
}

end sum_circumferences_of_small_circles_l2112_211240


namespace isosceles_triangle_base_length_l2112_211223

theorem isosceles_triangle_base_length (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 5) (h₂ : a + b + c = 17) : c = 7 :=
by
  -- proof would go here
  sorry

end isosceles_triangle_base_length_l2112_211223


namespace garden_area_increase_l2112_211296

theorem garden_area_increase :
  let length_rect := 60
  let width_rect := 20
  let area_rect := length_rect * width_rect
  
  let perimeter := 2 * (length_rect + width_rect)
  
  let side_square := perimeter / 4
  let area_square := side_square * side_square

  area_square - area_rect = 400 := by
    sorry

end garden_area_increase_l2112_211296


namespace dabbies_turkey_cost_l2112_211268

noncomputable def first_turkey_weight : ℕ := 6
noncomputable def second_turkey_weight : ℕ := 9
noncomputable def third_turkey_weight : ℕ := 2 * second_turkey_weight
noncomputable def cost_per_kg : ℕ := 2

noncomputable def total_cost : ℕ :=
  first_turkey_weight * cost_per_kg +
  second_turkey_weight * cost_per_kg +
  third_turkey_weight * cost_per_kg

theorem dabbies_turkey_cost : total_cost = 66 :=
by
  sorry

end dabbies_turkey_cost_l2112_211268


namespace find_x_plus_one_over_x_l2112_211272

open Real

theorem find_x_plus_one_over_x (x : ℝ) (h : x ^ 3 + 1 / x ^ 3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_plus_one_over_x_l2112_211272


namespace greatest_value_q_minus_r_l2112_211280

theorem greatest_value_q_minus_r : ∃ q r : ℕ, 1043 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 37) :=
by {
  sorry
}

end greatest_value_q_minus_r_l2112_211280


namespace product_of_solutions_l2112_211249

theorem product_of_solutions :
  let a := 2
  let b := 4
  let c := -6
  let discriminant := b^2 - 4*a*c
  ∃ (x₁ x₂ : ℝ), 2*x₁^2 + 4*x₁ - 6 = 0 ∧ 2*x₂^2 + 4*x₂ - 6 = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -3 :=
sorry

end product_of_solutions_l2112_211249


namespace phosphorus_atoms_l2112_211261

theorem phosphorus_atoms (x : ℝ) : 122 = 26.98 + 30.97 * x + 64 → x = 1 := by
sorry

end phosphorus_atoms_l2112_211261


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_4_l2112_211256

-- Define the function representing the quadratic equation
def quadratic_eq (m x : ℝ) := x^2 + (2 - m) * x + 1 - m

-- Part 1
theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
sorry

-- Part 2
theorem find_m_for_roots_difference_4 (m : ℝ) (H : m < 0) :
  (∃ (x1 x2 : ℝ), quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 ∧ x1 - x2 = 4) → m = -4 :=
sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_4_l2112_211256


namespace driver_net_rate_of_pay_is_25_l2112_211211

noncomputable def net_rate_of_pay_per_hour (hours_traveled : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (fuel_cost_per_gallon : ℝ) : ℝ :=
  let total_distance := speed * hours_traveled
  let total_fuel_used := total_distance / fuel_efficiency
  let total_earnings := pay_per_mile * total_distance
  let total_fuel_cost := fuel_cost_per_gallon * total_fuel_used
  let net_earnings := total_earnings - total_fuel_cost
  net_earnings / hours_traveled

theorem driver_net_rate_of_pay_is_25 :
  net_rate_of_pay_per_hour 3 50 25 0.6 2.5 = 25 := sorry

end driver_net_rate_of_pay_is_25_l2112_211211


namespace vector_dot_product_correct_l2112_211230

-- Definitions of the vectors
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ :=
  let x := 4 - 2 * vector_a.1
  let y := 1 - 2 * vector_a.2
  (x, y)

-- Theorem to prove the dot product is correct
theorem vector_dot_product_correct :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 4 := by
  sorry

end vector_dot_product_correct_l2112_211230


namespace workers_not_worked_days_l2112_211284

theorem workers_not_worked_days (W N : ℤ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 := 
by
  sorry

end workers_not_worked_days_l2112_211284


namespace yola_past_weight_l2112_211258

-- Definitions based on the conditions
def current_weight_yola : ℕ := 220
def weight_difference_current (D : ℕ) : ℕ := 30
def weight_difference_past (D : ℕ) : ℕ := D

-- Main statement
theorem yola_past_weight (D : ℕ) :
  (250 - D) = (current_weight_yola + weight_difference_current D - weight_difference_past D) :=
by
  sorry

end yola_past_weight_l2112_211258


namespace find_AD_l2112_211226

-- Given conditions as definitions
def AB := 5 -- given length in meters
def angle_ABC := 85 -- given angle in degrees
def angle_BCA := 45 -- given angle in degrees
def angle_DBC := 20 -- given angle in degrees

-- Lean theorem statement to prove the result
theorem find_AD : AD = AB := by
  -- The proof will be filled in afterwards; currently, we leave it as sorry.
  sorry

end find_AD_l2112_211226


namespace total_distance_l2112_211219

theorem total_distance (x : ℝ) (h : (1/2) * (x - 1) = (1/3) * x + 1) : x = 9 := 
by 
  sorry

end total_distance_l2112_211219


namespace part1_part2_part3_l2112_211216

namespace Problem

-- Definitions and conditions for problem 1
def f (m x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + (m - 1)

theorem part1 (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m < -5/3 := sorry

-- Definitions and conditions for problem 2
theorem part2 (m : ℝ) (h : m < 0) :
  ((-1 < m ∧ m < 0) → ∀ x : ℝ, x ≤ 1 ∨ x ≥ 1 / (m + 1)) ∧
  (m = -1 → ∀ x : ℝ, x ≤ 1) ∧
  (m < -1 → ∀ x : ℝ, 1 / (m + 1) ≤ x ∧ x ≤ 1) := sorry

-- Definitions and conditions for problem 3
theorem part3 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f m x ≥ x^2 + 2 * x) ↔ m ≥ (2 * Real.sqrt 3) / 3 + 1 := sorry

end Problem

end part1_part2_part3_l2112_211216


namespace original_number_of_employees_l2112_211213

theorem original_number_of_employees (E : ℝ) :
  (E - 0.125 * E) - 0.09 * (E - 0.125 * E) = 12385 → E = 15545 := 
by  -- Start the proof
  sorry  -- Placeholder for the proof, which is not required

end original_number_of_employees_l2112_211213


namespace max_g_equals_sqrt3_l2112_211232

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.pi / 9) + Real.sin (5 * Real.pi / 9 - x)

noncomputable def g (x : ℝ) : ℝ :=
  f (f x)

theorem max_g_equals_sqrt3 : ∀ x, g x ≤ Real.sqrt 3 :=
by
  sorry

end max_g_equals_sqrt3_l2112_211232


namespace johnny_marbles_l2112_211243

noncomputable def choose_at_least_one_red : ℕ :=
  let total_marbles := 8
  let red_marbles := 1
  let other_marbles := 7
  let choose_4_out_of_8 := Nat.choose total_marbles 4
  let choose_3_out_of_7 := Nat.choose other_marbles 3
  let choose_4_with_at_least_1_red := choose_3_out_of_7
  choose_4_with_at_least_1_red

theorem johnny_marbles : choose_at_least_one_red = 35 :=
by
  -- Sorry, proof is omitted
  sorry

end johnny_marbles_l2112_211243


namespace percentage_volume_occupied_is_100_l2112_211209

-- Define the dimensions of the box and cube
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 2

-- Define the volumes
def box_volume : ℕ := box_length * box_width * box_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the number of cubes that fit in each dimension
def cubes_along_length : ℕ := box_length / cube_side
def cubes_along_width : ℕ := box_width / cube_side
def cubes_along_height : ℕ := box_height / cube_side

-- Define the total number of cubes and the volume they occupy
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height
def volume_occupied_by_cubes : ℕ := total_cubes * cube_volume

-- Define the percentage of the box volume occupied by the cubes
def percentage_volume_occupied : ℕ := (volume_occupied_by_cubes * 100) / box_volume

-- Statement to prove
theorem percentage_volume_occupied_is_100 : percentage_volume_occupied = 100 := by
  sorry

end percentage_volume_occupied_is_100_l2112_211209


namespace rabbit_wins_race_l2112_211295

theorem rabbit_wins_race :
  ∀ (rabbit_speed1 rabbit_speed2 snail_speed rest_time total_distance : ℕ)
  (rabbit_time1 rabbit_time2 : ℚ),
  rabbit_speed1 = 20 →
  rabbit_speed2 = 30 →
  snail_speed = 2 →
  rest_time = 3 →
  total_distance = 100 →
  rabbit_time1 = (30 : ℚ) / rabbit_speed1 →
  rabbit_time2 = (70 : ℚ) / rabbit_speed2 →
  (rabbit_time1 + rest_time + rabbit_time2 < total_distance / snail_speed) :=
by
  intros
  sorry

end rabbit_wins_race_l2112_211295


namespace max_stamps_l2112_211293

theorem max_stamps (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 45) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, n ≤ total_cents / price_per_stamp ∧ n = 111 :=
by
  sorry

end max_stamps_l2112_211293


namespace cosine_identity_l2112_211288

theorem cosine_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) :
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end cosine_identity_l2112_211288


namespace fibonacci_invariant_abs_difference_l2112_211245

-- Given the sequence defined by the recurrence relation
def mArithmetical_fibonacci (u_n : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u_n n = u_n (n - 2) + u_n (n - 1)

theorem fibonacci_invariant_abs_difference (u : ℕ → ℤ) 
  (h : mArithmetical_fibonacci u) :
  ∃ c : ℤ, ∀ n : ℕ, |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c := 
sorry

end fibonacci_invariant_abs_difference_l2112_211245


namespace storyteller_friends_house_number_l2112_211218

theorem storyteller_friends_house_number
  (x y : ℕ)
  (htotal : 50 < x ∧ x < 500)
  (hsum : 2 * y = x * (x + 1)) :
  y = 204 :=
by
  sorry

end storyteller_friends_house_number_l2112_211218


namespace percentage_of_number_l2112_211206

variable (N P : ℝ)

theorem percentage_of_number 
  (h₁ : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) 
  (h₂ : (P / 100) * N = 120) : 
  P = 40 := 
by 
  sorry

end percentage_of_number_l2112_211206


namespace bottle_capacity_l2112_211247

theorem bottle_capacity
  (num_boxes : ℕ)
  (bottles_per_box : ℕ)
  (fill_fraction : ℚ)
  (total_volume : ℚ)
  (total_bottles : ℕ)
  (filled_volume : ℚ) :
  num_boxes = 10 →
  bottles_per_box = 50 →
  fill_fraction = 3 / 4 →
  total_volume = 4500 →
  total_bottles = num_boxes * bottles_per_box →
  filled_volume = (total_bottles : ℚ) * (fill_fraction * (12 : ℚ)) →
  12 = 4500 / (total_bottles * fill_fraction) := 
by 
  intros h1 h2 h3 h4 h5 h6
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end bottle_capacity_l2112_211247


namespace hot_dog_cost_l2112_211265

theorem hot_dog_cost : 
  ∃ h d : ℝ, (3 * h + 4 * d = 10) ∧ (2 * h + 3 * d = 7) ∧ (d = 1) := 
by 
  sorry

end hot_dog_cost_l2112_211265


namespace find_ratio_l2112_211217

theorem find_ratio (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 10 * b) / (b + 10 * a) = 2) : a / b = 0.8 :=
  sorry

end find_ratio_l2112_211217


namespace gcd_result_is_two_l2112_211202

theorem gcd_result_is_two
  (n m k j: ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) (hj : j > 0) :
  Nat.gcd (Nat.gcd (16 * n) (20 * m)) (Nat.gcd (18 * k) (24 * j)) = 2 := 
by
  sorry

end gcd_result_is_two_l2112_211202


namespace pentagon_coloring_count_l2112_211231

-- Define the three colors
inductive Color
| Red
| Yellow
| Green

open Color

-- Define the pentagon coloring problem
def adjacent_different (color1 color2 : Color) : Prop :=
color1 ≠ color2

-- Define a coloring for the pentagon
structure PentagonColoring :=
(A B C D E : Color)
(adjAB : adjacent_different A B)
(adjBC : adjacent_different B C)
(adjCD : adjacent_different C D)
(adjDE : adjacent_different D E)
(adjEA : adjacent_different E A)

-- The main statement to prove
theorem pentagon_coloring_count :
  ∃ (colorings : Finset PentagonColoring), colorings.card = 30 := sorry

end pentagon_coloring_count_l2112_211231


namespace a_minus_3d_eq_zero_l2112_211257

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x - 3 * d)

theorem a_minus_3d_eq_zero (a b c d : ℝ) (h : f a b c d ≠ 0)
  (h1 : ∀ x, f a b c d x = x) :
  a - 3 * d = 0 :=
sorry

end a_minus_3d_eq_zero_l2112_211257


namespace last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l2112_211221

theorem last_digit_of_sum_1_to_5 : 
  (1 ^ 2012 + 2 ^ 2012 + 3 ^ 2012 + 4 ^ 2012 + 5 ^ 2012) % 10 = 9 :=
  sorry

theorem last_digit_of_sum_1_to_2012 : 
  (List.sum (List.map (λ k => k ^ 2012) (List.range 2012).tail)) % 10 = 0 :=
  sorry

end last_digit_of_sum_1_to_5_last_digit_of_sum_1_to_2012_l2112_211221


namespace least_subtraction_for_divisibility_l2112_211285

/-- 
  Theorem: The least number that must be subtracted from 9857621 so that 
  the result is divisible by 17 is 8.
-/
theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, 9857621 % 17 = k ∧ k = 8 :=
by
  sorry

end least_subtraction_for_divisibility_l2112_211285


namespace boxes_left_to_sell_l2112_211238

def sales_goal : ℕ := 150
def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def total_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer

theorem boxes_left_to_sell : sales_goal - total_sold = 75 := by
  sorry

end boxes_left_to_sell_l2112_211238


namespace part_I_part_II_part_III_l2112_211279

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) - (1 / (2^x + 1))

theorem part_I :
  ∃ a : ℝ, ∀ x : ℝ, f x = a - (1 / (2^x + 1)) → a = (1 / 2) :=
by sorry

theorem part_II :
  ∀ y : ℝ, y = f x → (-1 / 2) < y ∧ y < (1 / 2) :=
by sorry

theorem part_III :
  ∀ m n : ℝ, m + n ≠ 0 → (f m + f n) / (m^3 + n^3) > f 0 :=
by sorry

end part_I_part_II_part_III_l2112_211279


namespace diagonal_crosses_768_unit_cubes_l2112_211299

-- Defining the dimensions of the rectangular prism
def a : ℕ := 150
def b : ℕ := 324
def c : ℕ := 375

-- Computing the gcd values
def gcd_ab : ℕ := Nat.gcd a b
def gcd_ac : ℕ := Nat.gcd a c
def gcd_bc : ℕ := Nat.gcd b c
def gcd_abc : ℕ := Nat.gcd (Nat.gcd a b) c

-- Using the formula to compute the number of unit cubes the diagonal intersects
def num_unit_cubes : ℕ := a + b + c - gcd_ab - gcd_ac - gcd_bc + gcd_abc

-- Stating the theorem to prove
theorem diagonal_crosses_768_unit_cubes : num_unit_cubes = 768 := by
  sorry

end diagonal_crosses_768_unit_cubes_l2112_211299


namespace more_movies_than_books_l2112_211253

-- Conditions
def books_read := 15
def movies_watched := 29

-- Question: How many more movies than books have you watched?
theorem more_movies_than_books : (movies_watched - books_read) = 14 := sorry

end more_movies_than_books_l2112_211253


namespace jenny_money_l2112_211251

theorem jenny_money (x : ℝ) (h : (4 / 7) * x = 24) : (x / 2) = 21 := 
sorry

end jenny_money_l2112_211251


namespace number_of_possible_m_values_l2112_211266

theorem number_of_possible_m_values :
  ∃ m_set : Finset ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 40 → (x1 + x2) ∈ m_set) ∧ m_set.card = 8 :=
sorry

end number_of_possible_m_values_l2112_211266


namespace baron_munchausen_claim_l2112_211255

-- Given conditions and question:
def weight_partition_problem (weights : Finset ℕ) (h_card : weights.card = 50) (h_distinct : ∀ w ∈ weights,  1 ≤ w ∧ w ≤ 100) (h_sum_even : weights.sum id % 2 = 0) : Prop :=
  ¬(∃ (s1 s2 : Finset ℕ), s1 ∪ s2 = weights ∧ s1 ∩ s2 = ∅ ∧ s1.sum id = s2.sum id)

-- We need to prove that the above statement is true.
theorem baron_munchausen_claim :
  ∀ (weights : Finset ℕ), weights.card = 50 ∧ (∀ w ∈ weights, 1 ≤ w ∧ w ≤ 100) ∧ weights.sum id % 2 = 0 → weight_partition_problem weights (by sorry) (by sorry) (by sorry) :=
sorry

end baron_munchausen_claim_l2112_211255


namespace zinc_in_combined_mass_l2112_211227

def mixture1_copper_zinc_ratio : ℕ × ℕ := (13, 7)
def mixture2_copper_zinc_ratio : ℕ × ℕ := (5, 3)
def mixture1_mass : ℝ := 100
def mixture2_mass : ℝ := 50

theorem zinc_in_combined_mass :
  let zinc1 := (mixture1_copper_zinc_ratio.2 : ℝ) / (mixture1_copper_zinc_ratio.1 + mixture1_copper_zinc_ratio.2) * mixture1_mass
  let zinc2 := (mixture2_copper_zinc_ratio.2 : ℝ) / (mixture2_copper_zinc_ratio.1 + mixture2_copper_zinc_ratio.2) * mixture2_mass
  zinc1 + zinc2 = 53.75 :=
by
  sorry

end zinc_in_combined_mass_l2112_211227


namespace net_profit_calculation_l2112_211276

def original_purchase_price : ℝ := 80000
def annual_property_tax_rate : ℝ := 0.012
def annual_maintenance_cost : ℝ := 1500
def annual_mortgage_interest_rate : ℝ := 0.04
def selling_profit_rate : ℝ := 0.20
def broker_commission_rate : ℝ := 0.05
def years_of_ownership : ℕ := 5

noncomputable def net_profit : ℝ :=
  let selling_price := original_purchase_price * (1 + selling_profit_rate)
  let brokers_commission := original_purchase_price * broker_commission_rate
  let total_property_tax := original_purchase_price * annual_property_tax_rate * years_of_ownership
  let total_maintenance_cost := annual_maintenance_cost * years_of_ownership
  let total_mortgage_interest := original_purchase_price * annual_mortgage_interest_rate * years_of_ownership
  let total_costs := brokers_commission + total_property_tax + total_maintenance_cost + total_mortgage_interest
  (selling_price - original_purchase_price) - total_costs

theorem net_profit_calculation : net_profit = -16300 := by
  sorry

end net_profit_calculation_l2112_211276


namespace minimize_cost_l2112_211200

-- Define the prices at each salon
def GustranSalonHaircut : ℕ := 45
def GustranSalonFacial : ℕ := 22
def GustranSalonNails : ℕ := 30

def BarbarasShopHaircut : ℕ := 30
def BarbarasShopFacial : ℕ := 28
def BarbarasShopNails : ℕ := 40

def FancySalonHaircut : ℕ := 34
def FancySalonFacial : ℕ := 30
def FancySalonNails : ℕ := 20

-- Define the total cost at each salon
def GustranSalonTotal : ℕ := GustranSalonHaircut + GustranSalonFacial + GustranSalonNails
def BarbarasShopTotal : ℕ := BarbarasShopHaircut + BarbarasShopFacial + BarbarasShopNails
def FancySalonTotal : ℕ := FancySalonHaircut + FancySalonFacial + FancySalonNails

-- Prove that the minimum total cost is $84
theorem minimize_cost : min GustranSalonTotal (min BarbarasShopTotal FancySalonTotal) = 84 := by
  -- proof goes here
  sorry

end minimize_cost_l2112_211200


namespace find_x_l2112_211214

theorem find_x (x y : ℤ) (some_number : ℤ) (h1 : y = 2) (h2 : some_number = 14) (h3 : 2 * x - y = some_number) : x = 8 :=
by 
  sorry

end find_x_l2112_211214


namespace quad_relation_l2112_211235

theorem quad_relation
  (α AI BI CI DI : ℝ)
  (h1 : AB = α * (AI / CI + BI / DI))
  (h2 : BC = α * (BI / DI + CI / AI))
  (h3 : CD = α * (CI / AI + DI / BI))
  (h4 : DA = α * (DI / BI + AI / CI)) :
  AB + CD = AD + BC := by
  sorry

end quad_relation_l2112_211235


namespace calculate_value_l2112_211237

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

variable (f : ℝ → ℝ)

axiom h : odd_function f
axiom h1 : increasing_on_interval f 3 7
axiom h2 : f 3 = -1
axiom h3 : f 6 = 8

theorem calculate_value : 2 * f (-6) + f (-3) = -15 := by
  sorry

end calculate_value_l2112_211237


namespace monica_study_ratio_l2112_211278

theorem monica_study_ratio :
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  weekend = wednesday + thursday + friday :=
by
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  sorry

end monica_study_ratio_l2112_211278


namespace circle_ratio_new_diameter_circumference_l2112_211222

theorem circle_ratio_new_diameter_circumference (r : ℝ) :
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := 
by
  sorry

end circle_ratio_new_diameter_circumference_l2112_211222


namespace min_nS_n_eq_neg32_l2112_211262

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (d : ℤ) (a_1 : ℤ)

-- Conditions
axiom arithmetic_sequence_def : ∀ n : ℕ, a n = a_1 + (n - 1) * d
axiom sum_first_n_def : ∀ n : ℕ, S n = n * a_1 + (n * (n - 1) / 2) * d

axiom a5_eq_3 : a 5 = 3
axiom S10_eq_40 : S 10 = 40

theorem min_nS_n_eq_neg32 : ∃ n : ℕ, n * S n = -32 :=
sorry

end min_nS_n_eq_neg32_l2112_211262


namespace cost_price_of_book_l2112_211239

theorem cost_price_of_book 
  (C : ℝ) 
  (h1 : 1.10 * C = sp10) 
  (h2 : 1.15 * C = sp15)
  (h3 : sp15 - sp10 = 90) : 
  C = 1800 := 
sorry

end cost_price_of_book_l2112_211239


namespace remainder_div_eq_4_l2112_211207

theorem remainder_div_eq_4 {x y : ℕ} (h1 : y = 25) (h2 : (x / y : ℝ) = 96.16) : x % y = 4 := 
sorry

end remainder_div_eq_4_l2112_211207


namespace remainder_1234_5678_9012_div_5_l2112_211203

theorem remainder_1234_5678_9012_div_5 : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end remainder_1234_5678_9012_div_5_l2112_211203


namespace minimum_value_l2112_211287

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (y : ℝ), y = (c / (a + b)) + (b / c) ∧ y ≥ (Real.sqrt 2) - (1 / 2) :=
sorry

end minimum_value_l2112_211287


namespace candy_left_l2112_211292

-- Definitions according to the conditions
def initialCandy : ℕ := 15
def candyGivenToHaley : ℕ := 6

-- Theorem statement formalizing the proof problem
theorem candy_left (c : ℕ) (h₁ : c = initialCandy - candyGivenToHaley) : c = 9 :=
by
  -- The proof is omitted as instructed.
  sorry

end candy_left_l2112_211292


namespace percent_of_whole_is_fifty_l2112_211281

theorem percent_of_whole_is_fifty (part whole : ℝ) (h1 : part = 180) (h2 : whole = 360) : 
  ((part / whole) * 100) = 50 := 
by 
  rw [h1, h2] 
  sorry

end percent_of_whole_is_fifty_l2112_211281


namespace weight_of_second_piece_l2112_211242

-- Define the uniform density of the metal.
def density : ℝ := 0.5  -- ounces per square inch

-- Define the side lengths of the two pieces of metal.
def side_length1 : ℝ := 4  -- inches
def side_length2 : ℝ := 7  -- inches

-- Define the weights of the first piece of metal.
def weight1 : ℝ := 8  -- ounces

-- Define the areas of the pieces of metal.
def area1 : ℝ := side_length1^2  -- square inches
def area2 : ℝ := side_length2^2  -- square inches

-- The theorem to prove: the weight of the second piece of metal.
theorem weight_of_second_piece : (area2 * density) = 24.5 :=
by
  sorry

end weight_of_second_piece_l2112_211242


namespace speed_on_local_roads_l2112_211210

theorem speed_on_local_roads (v : ℝ) (h1 : 60 + 120 = 180) (h2 : (60 + 120) / (60 / v + 120 / 60) = 36) : v = 20 :=
by
  sorry

end speed_on_local_roads_l2112_211210


namespace ladder_slip_l2112_211246

theorem ladder_slip 
  (ladder_length : ℝ) 
  (initial_base : ℝ) 
  (slip_height : ℝ) 
  (h_length : ladder_length = 30) 
  (h_base : initial_base = 11) 
  (h_slip : slip_height = 6) 
  : ∃ (slide_distance : ℝ), abs (slide_distance - 9.49) < 0.01 :=
by
  let initial_height := Real.sqrt (ladder_length^2 - initial_base^2)
  let new_height := initial_height - slip_height
  let new_base := Real.sqrt (ladder_length^2 - new_height^2)
  let slide_distance := new_base - initial_base
  use slide_distance
  have h_approx : abs (slide_distance - 9.49) < 0.01 := sorry
  exact h_approx

end ladder_slip_l2112_211246


namespace motorboat_speed_l2112_211212

theorem motorboat_speed 
  (c : ℝ) (h_c : c = 2.28571428571)
  (t_up : ℝ) (h_t_up : t_up = 20 / 60)
  (t_down : ℝ) (h_t_down : t_down = 15 / 60) :
  ∃ v : ℝ, v = 16 :=
by
  sorry

end motorboat_speed_l2112_211212


namespace simplify_expr_l2112_211201

-- Define the terms
def a : ℕ := 2 ^ 10
def b : ℕ := 5 ^ 6

-- Define the expression we need to simplify
def expr := (a * b : ℝ)^(1/3)

-- Define the simplified form
def c : ℕ := 200
def d : ℕ := 2
def simplified_expr := (c : ℝ) * (d : ℝ)^(1/3)

-- The statement we need to prove
theorem simplify_expr : expr = simplified_expr ∧ (c + d = 202) := by
  sorry

end simplify_expr_l2112_211201


namespace possible_values_y_l2112_211273

theorem possible_values_y (x : ℝ) (h : x^2 + 4 * (x / (x - 2))^2 = 45) : 
  ∃ y : ℝ, y = 2 ∨ y = 16 :=
sorry

end possible_values_y_l2112_211273


namespace complement_angle_l2112_211294

theorem complement_angle (A : ℝ) (hA : A = 35) : 90 - A = 55 := by
  sorry

end complement_angle_l2112_211294


namespace number_of_real_solutions_l2112_211208

theorem number_of_real_solutions :
  (∃ (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) →
  (∃! (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) :=
sorry

end number_of_real_solutions_l2112_211208


namespace iron_ii_sulfate_moles_l2112_211269

/-- Given the balanced chemical equation for the reaction between iron (Fe) and sulfuric acid (H2SO4)
    to form Iron (II) sulfate (FeSO4) and hydrogen gas (H2) and the 1:1 molar ratio between iron and
    sulfuric acid, determine the number of moles of Iron (II) sulfate formed when 3 moles of Iron and
    2 moles of Sulfuric acid are combined. This is a limiting reactant problem with the final 
    product being 2 moles of Iron (II) sulfate (FeSO4). -/
theorem iron_ii_sulfate_moles (Fe moles_H2SO4 : Nat) (reaction_ratio : Nat) (FeSO4 moles_formed : Nat) :
  Fe = 3 → moles_H2SO4 = 2 → reaction_ratio = 1 → moles_formed = 2 :=
by
  intros hFe hH2SO4 hRatio
  apply sorry

end iron_ii_sulfate_moles_l2112_211269


namespace sum_arithmetic_sequence_l2112_211263

theorem sum_arithmetic_sequence :
  let n := 21
  let a := 100
  let l := 120
  (n / 2) * (a + l) = 2310 :=
by
  -- define n, a, and l based on the conditions
  let n := 21
  let a := 100
  let l := 120
  -- state the goal
  have h : (n / 2) * (a + l) = 2310 := sorry
  exact h

end sum_arithmetic_sequence_l2112_211263


namespace smallest_z_value_l2112_211259

theorem smallest_z_value :
  ∀ w x y z : ℤ, (∃ k : ℤ, w = 2 * k - 1 ∧ x = 2 * k + 1 ∧ y = 2 * k + 3 ∧ z = 2 * k + 5) ∧
    w^3 + x^3 + y^3 = z^3 →
    z = 9 :=
sorry

end smallest_z_value_l2112_211259
