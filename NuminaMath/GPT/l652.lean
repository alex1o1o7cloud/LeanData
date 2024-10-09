import Mathlib

namespace arithmetic_sequence_condition_l652_65296

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (h : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) : 
a 6 = 3 := 
by 
  sorry

end arithmetic_sequence_condition_l652_65296


namespace fraction_people_over_65_l652_65280

theorem fraction_people_over_65 (T : ℕ) (F : ℕ) : 
  (3:ℚ) / 7 * T = 24 ∧ 50 < T ∧ T < 100 → T = 56 ∧ ∃ F : ℕ, (F / 56 : ℚ) = F / (T : ℚ) :=
by 
  sorry

end fraction_people_over_65_l652_65280


namespace common_tangents_l652_65209

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

theorem common_tangents :
  ∃ (n : ℕ), n = 4 ∧ 
    (∀ (L : ℝ → ℝ → Prop), 
      (∀ x y, L x y → circle1_eqn x y ∧ circle2_eqn x y) → n = 4) := 
sorry

end common_tangents_l652_65209


namespace find_all_pos_integers_l652_65282

theorem find_all_pos_integers (M : ℕ) (h1 : M > 0) (h2 : M < 10) :
  (5 ∣ (1989^M + M^1989)) ↔ (M = 1) ∨ (M = 4) :=
by
  sorry

end find_all_pos_integers_l652_65282


namespace elvins_fixed_charge_l652_65239

theorem elvins_fixed_charge (F C : ℝ) 
  (h1 : F + C = 40) 
  (h2 : F + 2 * C = 76) : F = 4 := 
by 
  sorry

end elvins_fixed_charge_l652_65239


namespace circle_radius_of_complex_roots_l652_65297

theorem circle_radius_of_complex_roots (z : ℂ) (hz : (z - 1)^3 = 8 * z^3) : 
  ∃ r : ℝ, r = 1 / Real.sqrt 3 :=
by
  sorry

end circle_radius_of_complex_roots_l652_65297


namespace bottles_of_regular_soda_l652_65203

theorem bottles_of_regular_soda (R : ℕ) : 
  let apples := 36 
  let diet_soda := 54
  let total_bottles := apples + 98 
  R + diet_soda = total_bottles → R = 80 :=
by
  sorry

end bottles_of_regular_soda_l652_65203


namespace min_value_x_squared_plus_6x_l652_65202

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l652_65202


namespace jill_vs_jack_arrival_time_l652_65233

def distance_to_park : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem jill_vs_jack_arrival_time :
  let jill_time := distance_to_park / jill_speed
  let jack_time := distance_to_park / jack_speed
  let jill_time_minutes := jill_time * 60
  let jack_time_minutes := jack_time * 60
  jill_time_minutes < jack_time_minutes ∧ jack_time_minutes - jill_time_minutes = 5.4 :=
by
  sorry

end jill_vs_jack_arrival_time_l652_65233


namespace ratio_of_solving_linear_equations_to_algebra_problems_l652_65227

theorem ratio_of_solving_linear_equations_to_algebra_problems:
  let total_problems := 140
  let algebra_percentage := 0.40
  let solving_linear_equations := 28
  let total_algebra_problems := algebra_percentage * total_problems
  let ratio := solving_linear_equations / total_algebra_problems
  ratio = 1 / 2 := by
  sorry

end ratio_of_solving_linear_equations_to_algebra_problems_l652_65227


namespace brandy_used_0_17_pounds_of_chocolate_chips_l652_65240

def weight_of_peanuts : ℝ := 0.17
def weight_of_raisins : ℝ := 0.08
def total_weight_of_trail_mix : ℝ := 0.42

theorem brandy_used_0_17_pounds_of_chocolate_chips :
  total_weight_of_trail_mix - (weight_of_peanuts + weight_of_raisins) = 0.17 :=
by
  sorry

end brandy_used_0_17_pounds_of_chocolate_chips_l652_65240


namespace isabelle_weeks_needed_l652_65268

def total_ticket_cost : ℕ := 20 + 10 + 10
def total_savings : ℕ := 5 + 5
def weekly_earnings : ℕ := 3
def amount_needed : ℕ := total_ticket_cost - total_savings
def weeks_needed : ℕ := amount_needed / weekly_earnings

theorem isabelle_weeks_needed 
  (ticket_cost_isabelle : ℕ := 20)
  (ticket_cost_brother : ℕ := 10)
  (savings_brothers : ℕ := 5)
  (savings_isabelle : ℕ := 5)
  (earnings_weekly : ℕ := 3)
  (total_cost := ticket_cost_isabelle + 2 * ticket_cost_brother)
  (total_savings := savings_brothers + savings_isabelle)
  (needed_amount := total_cost - total_savings)
  (weeks := needed_amount / earnings_weekly) :
  weeks = 10 :=
  by
  sorry

end isabelle_weeks_needed_l652_65268


namespace inequality_solution_range_l652_65231

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℤ, 6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) ∧
  (∃ x1 x2 x3 : ℤ, (x1 = 3 ∧ x2 = 4 ∧ x3 = 5) ∧
   (6 - 3 * (x1 : ℝ) < 0 ∧ 2 * (x1 : ℝ) ≤ a) ∧
   (6 - 3 * (x2 : ℝ) < 0 ∧ 2 * (x2 : ℝ) ≤ a) ∧
   (6 - 3 * (x3 : ℝ) < 0 ∧ 2 * (x3 : ℝ) ≤ a) ∧
   (∀ x : ℤ, (6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) → 
     (x = 3 ∨ x = 4 ∨ x = 5)))
  → 10 ≤ a ∧ a < 12 :=
sorry

end inequality_solution_range_l652_65231


namespace state_tax_percentage_l652_65275

theorem state_tax_percentage (weekly_salary federal_percent health_insurance life_insurance parking_fee final_paycheck : ℝ)
  (h_weekly_salary : weekly_salary = 450)
  (h_federal_percent : federal_percent = 1/3)
  (h_health_insurance : health_insurance = 50)
  (h_life_insurance : life_insurance = 20)
  (h_parking_fee : parking_fee = 10)
  (h_final_paycheck : final_paycheck = 184) :
  (36 / 450) * 100 = 8 :=
by
  sorry

end state_tax_percentage_l652_65275


namespace a5_value_l652_65286

def seq (a : ℕ → ℤ) (a1 : a 1 = 2) (rec : ∀ n, a (n + 1) = 2 * a n - 1) : Prop := True

theorem a5_value : 
  ∀ (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, a (n + 1) = 2 * a n - 1),
  seq a h1 recurrence → a 5 = 17 :=
by
  intros a h1 recurrence seq_a
  sorry

end a5_value_l652_65286


namespace find_age_l652_65277

theorem find_age (A : ℤ) (h : 4 * (A + 4) - 4 * (A - 4) = A) : A = 32 :=
by sorry

end find_age_l652_65277


namespace problem_statement_l652_65242

noncomputable def f : ℝ → ℝ := sorry

variable (α : ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 3) = -f x
axiom tan_alpha : Real.tan α = 2

theorem problem_statement : f (15 * Real.sin α * Real.cos α) = 0 := 
by {
  sorry
}

end problem_statement_l652_65242


namespace michael_initial_money_l652_65208

theorem michael_initial_money (M : ℝ) 
  (half_give_away_to_brother : ∃ (m_half : ℝ), M / 2 = m_half)
  (brother_initial_money : ℝ := 17)
  (candy_cost : ℝ := 3)
  (brother_ends_up_with : ℝ := 35) :
  brother_initial_money + M / 2 - candy_cost = brother_ends_up_with ↔ M = 42 :=
sorry

end michael_initial_money_l652_65208


namespace number_of_primary_schools_l652_65262

theorem number_of_primary_schools (A B total : ℕ) (h1 : A = 2 * 400)
  (h2 : B = 2 * 340) (h3 : total = 1480) (h4 : total = A + B) :
  2 + 2 = 4 :=
by
  sorry

end number_of_primary_schools_l652_65262


namespace find_x_l652_65218

variable (a b x : ℝ)

def condition1 : Prop := a / b = 5 / 4
def condition2 : Prop := (4 * a + x * b) / (4 * a - x * b) = 4

theorem find_x (h1 : condition1 a b) (h2 : condition2 a b x) : x = 3 :=
  sorry

end find_x_l652_65218


namespace compare_magnitudes_l652_65219

theorem compare_magnitudes (a b c d e : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) (h₅ : e < 0) :
  (e / (a - c)) > (e / (b - d)) :=
  sorry

end compare_magnitudes_l652_65219


namespace race_time_l652_65224

theorem race_time 
    (v_A v_B t_A t_B : ℝ)
    (h1 : v_A = 1000 / t_A) 
    (h2 : v_B = 940 / t_A)
    (h3 : v_B = 1000 / (t_A + 15)) 
    (h4 : t_B = t_A + 15) :
    t_A = 235 := 
  by
    sorry

end race_time_l652_65224


namespace remainder_2023_mul_7_div_45_l652_65215

/-- The remainder when the product of 2023 and 7 is divided by 45 is 31. -/
theorem remainder_2023_mul_7_div_45 : 
  (2023 * 7) % 45 = 31 := 
by
  sorry

end remainder_2023_mul_7_div_45_l652_65215


namespace number_of_clients_l652_65272

-- Definitions from the problem
def cars : ℕ := 18
def selections_per_client : ℕ := 3
def selections_per_car : ℕ := 3

-- Theorem statement: Prove that the number of clients is 18
theorem number_of_clients (total_cars : ℕ) (cars_selected_by_each_client : ℕ) (each_car_selected : ℕ)
  (h_cars : total_cars = cars)
  (h_select_each : cars_selected_by_each_client = selections_per_client)
  (h_selected_car : each_car_selected = selections_per_car) :
  (total_cars * each_car_selected) / cars_selected_by_each_client = 18 :=
by
  rw [h_cars, h_select_each, h_selected_car]
  sorry

end number_of_clients_l652_65272


namespace simplify_expression_l652_65285

variable (x y z : ℝ)

-- Statement of the problem to be proved.
theorem simplify_expression :
  (15 * x + 45 * y - 30 * z) + (20 * x - 10 * y + 5 * z) - (5 * x + 35 * y - 15 * z) = 
  (30 * x - 10 * z) :=
by
  -- Placeholder for the actual proof
  sorry

end simplify_expression_l652_65285


namespace sand_problem_l652_65263

-- Definitions based on conditions
def initial_sand := 1050
def sand_lost_first := 32
def sand_lost_second := 67
def sand_lost_third := 45
def sand_lost_fourth := 54

-- Total sand lost
def total_sand_lost := sand_lost_first + sand_lost_second + sand_lost_third + sand_lost_fourth

-- Sand remaining
def sand_remaining := initial_sand - total_sand_lost

-- Theorem stating the proof problem
theorem sand_problem : sand_remaining = 852 :=
by
-- Skipping proof as per instructions
sorry

end sand_problem_l652_65263


namespace count_real_solutions_l652_65201

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l652_65201


namespace zinc_percentage_in_1_gram_antacid_l652_65289

theorem zinc_percentage_in_1_gram_antacid :
  ∀ (z1 z2 : ℕ → ℤ) (total_zinc : ℤ),
    z1 0 = 2 ∧ z2 0 = 2 ∧ z1 1 = 1 ∧ total_zinc = 650 ∧
    (z1 0) * 2 * 5 / 100 + (z2 1) * 3 = total_zinc / 100 →
    (z2 1) * 100 = 15 :=
by
  sorry

end zinc_percentage_in_1_gram_antacid_l652_65289


namespace equal_roots_of_quadratic_l652_65290

theorem equal_roots_of_quadratic (k : ℝ) : 
  ( ∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0 → x = x ) ↔ k = 16 / 49 :=
by
  sorry

end equal_roots_of_quadratic_l652_65290


namespace function_not_strictly_decreasing_l652_65245

theorem function_not_strictly_decreasing (b : ℝ)
  (h : ¬ ∀ x1 x2 : ℝ, x1 < x2 → (-x1^3 + b*x1^2 - (2*b + 3)*x1 + 2 - b > -x2^3 + b*x2^2 - (2*b + 3)*x2 + 2 - b)) : 
  b < -1 ∨ b > 3 :=
by
  sorry

end function_not_strictly_decreasing_l652_65245


namespace train_speed_platform_man_l652_65200

theorem train_speed_platform_man (t_man t_platform : ℕ) (platform_length : ℕ) (v_train_mps : ℝ) (v_train_kmph : ℝ) 
  (h1 : t_man = 18) 
  (h2 : t_platform = 32) 
  (h3 : platform_length = 280)
  (h4 : v_train_mps = (platform_length / (t_platform - t_man)))
  (h5 : v_train_kmph = v_train_mps * 3.6) :
  v_train_kmph = 72 := 
sorry

end train_speed_platform_man_l652_65200


namespace determine_b_l652_65207

theorem determine_b (N a b c : ℤ) (h1 : a > 1 ∧ b > 1 ∧ c > 1) (h2 : N ≠ 1)
  (h3 : (N : ℝ) ^ (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c ^ 2)) = N ^ (49 / 60)) :
  b = 4 :=
sorry

end determine_b_l652_65207


namespace verify_z_relationship_l652_65274

variable {x y z : ℝ}

theorem verify_z_relationship (h1 : x > y) (h2 : y > 1) :
  z = (x + 3) - 2 * (y - 5) → z = x - 2 * y + 13 :=
by
  intros
  sorry

end verify_z_relationship_l652_65274


namespace masha_can_pay_with_5_ruble_coins_l652_65260

theorem masha_can_pay_with_5_ruble_coins (p c n : ℤ) (h : 2 * p + c + 7 * n = 100) : (p + 3 * c + n) % 5 = 0 :=
  sorry

end masha_can_pay_with_5_ruble_coins_l652_65260


namespace ratio_of_profits_is_2_to_3_l652_65221

-- Conditions
def Praveen_initial_investment := 3220
def Praveen_investment_duration := 12
def Hari_initial_investment := 8280
def Hari_investment_duration := 7

-- Effective capital contributions
def Praveen_effective_capital : ℕ := Praveen_initial_investment * Praveen_investment_duration
def Hari_effective_capital : ℕ := Hari_initial_investment * Hari_investment_duration

-- Theorem statement to be proven
theorem ratio_of_profits_is_2_to_3 : (Praveen_effective_capital : ℚ) / Hari_effective_capital = 2 / 3 :=
by sorry

end ratio_of_profits_is_2_to_3_l652_65221


namespace terminating_decimals_count_l652_65254

theorem terminating_decimals_count :
  (∃ count : ℕ, count = 166 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → (∃ m : ℕ, n = 3 * m)) :=
sorry

end terminating_decimals_count_l652_65254


namespace avg_growth_rate_eq_l652_65251

variable (x : ℝ)

theorem avg_growth_rate_eq :
  (560 : ℝ) * (1 + x)^2 = 830 :=
sorry

end avg_growth_rate_eq_l652_65251


namespace num_valid_four_digit_numbers_l652_65225

theorem num_valid_four_digit_numbers :
  let N (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d
  ∃ (a b c d : ℕ), 5000 ≤ N a b c d ∧ N a b c d < 7000 ∧ (N a b c d % 5 = 0) ∧ (2 ≤ b ∧ b < c ∧ c ≤ 7) ∧
                   (60 = (if a = 5 ∨ a = 6 then (if d = 0 ∨ d = 5 then 15 else 0) else 0)) :=
sorry

end num_valid_four_digit_numbers_l652_65225


namespace train_length_correct_l652_65235

noncomputable def length_of_train (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let total_distance := speed_m_per_s * time_s
  total_distance - platform_length_m

theorem train_length_correct :
  length_of_train 55 520 43.196544276457885 = 140 :=
by
  unfold length_of_train
  -- The conversion and calculations would be verified here
  sorry

end train_length_correct_l652_65235


namespace remaining_pictures_l652_65204

theorem remaining_pictures (first_book : ℕ) (second_book : ℕ) (third_book : ℕ) (colored_pictures : ℕ) :
  first_book = 23 → second_book = 32 → third_book = 45 → colored_pictures = 44 →
  (first_book + second_book + third_book - colored_pictures) = 56 :=
by
  sorry

end remaining_pictures_l652_65204


namespace a_n_is_perfect_square_l652_65241

theorem a_n_is_perfect_square :
  ∀ (a b : ℕ → ℤ), a 0 = 1 → b 0 = 0 →
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∀ n, ∃ k : ℤ, a n = k * k :=
by
  sorry

end a_n_is_perfect_square_l652_65241


namespace minimum_sum_of_distances_squared_l652_65288

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -2, y := 0 }
def B : Point := { x := 2, y := 0 }

-- Define the moving point P on the circle
def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

-- Distance squared between two points
def dist_squared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the sum of squared distances from P to points A and B
def sum_distances_squared (P : Point) : ℝ :=
  dist_squared P A + dist_squared P B

-- Statement of the proof problem
theorem minimum_sum_of_distances_squared :
  ∃ P : Point, on_circle P ∧ sum_distances_squared P = 26 :=
sorry

end minimum_sum_of_distances_squared_l652_65288


namespace weight_of_B_l652_65247

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l652_65247


namespace smallest_solution_floor_equation_l652_65253

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l652_65253


namespace solve_for_x_l652_65256

theorem solve_for_x (x : ℝ) (h : 3 * x + 20 = (1 / 3) * (7 * x + 45)) : x = -7.5 :=
sorry

end solve_for_x_l652_65256


namespace even_function_implies_f2_eq_neg5_l652_65211

def f (x a : ℝ) : ℝ := (x - a) * (x + 3)

theorem even_function_implies_f2_eq_neg5 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) :
  f 2 a = -5 :=
by
  sorry

end even_function_implies_f2_eq_neg5_l652_65211


namespace min_value_x_plus_4_div_x_plus_1_l652_65287

theorem min_value_x_plus_4_div_x_plus_1 (x : ℝ) (h : x > -1) : ∃ m, m = 3 ∧ ∀ y, y = x + 4 / (x + 1) → y ≥ m :=
sorry

end min_value_x_plus_4_div_x_plus_1_l652_65287


namespace least_number_division_remainder_4_l652_65257

theorem least_number_division_remainder_4 : 
  ∃ n : Nat, (n % 6 = 4) ∧ (n % 130 = 4) ∧ (n % 9 = 4) ∧ (n % 18 = 4) ∧ n = 2344 :=
by
  sorry

end least_number_division_remainder_4_l652_65257


namespace enclosed_area_correct_l652_65213

noncomputable def enclosed_area : ℝ :=
  ∫ x in (1/2)..2, (-x + 5/2 - 1/x)

theorem enclosed_area_correct :
  enclosed_area = (15/8) - 2 * Real.log 2 :=
by
  sorry

end enclosed_area_correct_l652_65213


namespace max_min_magnitude_of_sum_l652_65299

open Real

-- Define the vectors a and b and their magnitudes
variables {a b : ℝ × ℝ}
variable (h_a : ‖a‖ = 5)
variable (h_b : ‖b‖ = 2)

-- Define the constant 7 and 3 for the max and min values
noncomputable def max_magnitude : ℝ := 7
noncomputable def min_magnitude : ℝ := 3

-- State the theorem
theorem max_min_magnitude_of_sum (h_a : ‖a‖ = 5) (h_b : ‖b‖ = 2) :
  ‖a + b‖ ≤ max_magnitude ∧ ‖a + b‖ ≥ min_magnitude :=
by {
  sorry -- Proof goes here
}

end max_min_magnitude_of_sum_l652_65299


namespace exists_circle_with_exactly_n_integer_points_l652_65248

noncomputable def circle_with_n_integer_points (n : ℕ) : Prop :=
  ∃ r : ℤ, ∃ (xs ys : List ℤ), 
    xs.length = n ∧ ys.length = n ∧
    ∀ (x y : ℤ), x ∈ xs → y ∈ ys → x^2 + y^2 = r^2

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  circle_with_n_integer_points n := 
sorry

end exists_circle_with_exactly_n_integer_points_l652_65248


namespace rank_from_left_l652_65283

theorem rank_from_left (total_students : ℕ) (rank_from_right : ℕ) (h1 : total_students = 20) (h2 : rank_from_right = 13) : 
  (total_students - rank_from_right + 1 = 8) :=
by
  sorry

end rank_from_left_l652_65283


namespace cos_225_eq_neg_sqrt2_div_2_l652_65238

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l652_65238


namespace age_difference_is_24_l652_65228

theorem age_difference_is_24 (d f : ℕ) (h1 : d = f / 9) (h2 : f + 1 = 7 * (d + 1)) : f - d = 24 := sorry

end age_difference_is_24_l652_65228


namespace inequality_solution_nonempty_l652_65267

theorem inequality_solution_nonempty (a : ℝ) :
  (∃ x : ℝ, x ^ 2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end inequality_solution_nonempty_l652_65267


namespace proof_by_contradiction_conditions_l652_65265

theorem proof_by_contradiction_conditions:
  (∃ (neg_conclusion known_conditions ax_thms_defs original_conclusion : Prop),
    (neg_conclusion ∧ known_conditions ∧ ax_thms_defs) → False)
:= sorry

end proof_by_contradiction_conditions_l652_65265


namespace find_N_l652_65237

/-- Given a row: [a, b, c, d, 2, f, g], 
    first column: [15, h, i, 14, j, k, l, 10],
    second column: [N, m, n, o, p, q, r, -21],
    where h=i+4 and i=j+4,
    b=15 and d = (2 - 15) / 3.
    The common difference c_n = -2.5.
    Prove N = -13.5.
-/
theorem find_N (a b c d f g h i j k l m n o p q r : ℝ) (N : ℝ) :
  b = 15 ∧ j = 14 ∧ l = 10 ∧ r = -21 ∧
  h = i + 4 ∧ i = j + 4 ∧
  c = (2 - 15) / 3 ∧
  g = b + 6 * c ∧
  N = g + 1 * (-2.5) →
  N = -13.5 :=
by
  intros h1
  sorry

end find_N_l652_65237


namespace number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l652_65269

variable (A B C D : ℕ)
variable (dice : ℕ → ℕ)

-- Conditions
axiom dice_faces : ∀ {i : ℕ}, 1 ≤ i ∧ i ≤ 6 → ∃ j, dice i = j
axiom opposite_faces_sum : ∀ {i j : ℕ}, dice i + dice j = 7
axiom configuration : True -- Placeholder for the specific arrangement configuration

-- Questions and Proof Statements
theorem number_of_dots_on_A :
  A = 3 := sorry

theorem number_of_dots_on_B :
  B = 5 := sorry

theorem number_of_dots_on_C :
  C = 6 := sorry

theorem number_of_dots_on_D :
  D = 5 := sorry

end number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l652_65269


namespace computer_program_X_value_l652_65255

theorem computer_program_X_value : 
  ∃ (n : ℕ), (let X := 5 + 3 * (n - 1) 
               let S := (3 * n^2 + 7 * n) / 2 
               S ≥ 10500) ∧ X = 251 :=
sorry

end computer_program_X_value_l652_65255


namespace transformation_1_transformation_2_l652_65234

theorem transformation_1 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq1 : 5 * x + 2 * y = 0) : 
  5 * x' + 3 * y' = 0 := 
sorry

theorem transformation_2 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq2 : x^2 + y^2 = 1) : 
  4 * x' ^ 2 + 9 * y' ^ 2 = 1 := 
sorry

end transformation_1_transformation_2_l652_65234


namespace sum_of_remainders_mod_500_l652_65278

theorem sum_of_remainders_mod_500 : 
  (5 ^ (5 ^ (5 ^ 5)) + 2 ^ (2 ^ (2 ^ 2))) % 500 = 49 := by
  sorry

end sum_of_remainders_mod_500_l652_65278


namespace sin_300_eq_neg_sqrt_three_div_two_l652_65258

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l652_65258


namespace solution_set_f_l652_65210

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2) - 1

theorem solution_set_f (x : ℝ) (hx_pos : x > 0) : 
  f x > f (2 * x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end solution_set_f_l652_65210


namespace area_of_triangle_ABC_circumcenter_of_triangle_ABC_l652_65223

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨8, 3⟩

def triangle_area (A B C : Point) : ℚ := by
  -- area calculation will be filled here
  sorry

def circumcenter (A B C : Point) : Point := by
  -- circumcenter calculation will be filled here
  sorry

theorem area_of_triangle_ABC : triangle_area A B C = 16 :=
  sorry

theorem circumcenter_of_triangle_ABC : circumcenter A B C = ⟨9/2, 7/2⟩ :=
  sorry

end area_of_triangle_ABC_circumcenter_of_triangle_ABC_l652_65223


namespace rope_cutting_impossible_l652_65273

/-- 
Given a rope initially cut into 5 pieces, and then some of these pieces were each cut into 
5 parts, with this process repeated several times, it is not possible for the total 
number of pieces to be exactly 2019.
-/ 
theorem rope_cutting_impossible (n : ℕ) : 5 + 4 * n ≠ 2019 := 
sorry

end rope_cutting_impossible_l652_65273


namespace find_number_l652_65229

theorem find_number (x : ℝ) (h : (1/3) * x = 12) : x = 36 :=
sorry

end find_number_l652_65229


namespace probability_of_at_least_two_same_rank_approx_l652_65252

noncomputable def probability_at_least_two_same_rank (cards_drawn : ℕ) (total_cards : ℕ) : ℝ :=
  let ranks := 13
  let different_ranks_comb := Nat.choose ranks cards_drawn
  let rank_suit_combinations := different_ranks_comb * (4 ^ cards_drawn)
  let total_combinations := Nat.choose total_cards cards_drawn
  let p_complement := rank_suit_combinations / total_combinations
  1 - p_complement

theorem probability_of_at_least_two_same_rank_approx (h : 5 ≤ 52) : 
  abs (probability_at_least_two_same_rank 5 52 - 0.49) < 0.01 := 
by
  sorry

end probability_of_at_least_two_same_rank_approx_l652_65252


namespace hyperbola_A_asymptote_l652_65214

-- Define the hyperbola and asymptote conditions
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def asymptote_eq (y x : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Statement of the proof problem in Lean 4
theorem hyperbola_A_asymptote :
  ∀ (x y : ℝ), hyperbola_A x y → asymptote_eq y x :=
sorry

end hyperbola_A_asymptote_l652_65214


namespace least_number_to_subtract_l652_65266

-- Define the problem and prove that this number, when subtracted, makes the original number divisible by 127.
theorem least_number_to_subtract (n : ℕ) (h₁ : n = 100203) (h₂ : 127 > 0) : 
  ∃ k : ℕ, (100203 - 72) = 127 * k :=
by
  sorry

end least_number_to_subtract_l652_65266


namespace sheila_picnic_probability_l652_65220

theorem sheila_picnic_probability :
  let P_rain := 0.5
  let P_go_given_rain := 0.3
  let P_go_given_sunny := 0.9
  let P_remember := 0.9  -- P(remember) = 1 - P(forget)
  let P_sunny := 1 - P_rain
  
  P_rain * P_go_given_rain * P_remember + P_sunny * P_go_given_sunny * P_remember = 0.54 :=
by
  sorry

end sheila_picnic_probability_l652_65220


namespace coefficient_x2_in_expansion_l652_65236

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the problem: Given (2x + 1)^5, find the coefficient of x^2 term
theorem coefficient_x2_in_expansion : 
  binomial 5 3 * (2 ^ 2) = 40 := by 
  sorry

end coefficient_x2_in_expansion_l652_65236


namespace find_a_l652_65244

theorem find_a (a : ℕ) (h_pos : a > 0) (h_quadrant : 2 - a > 0) : a = 1 := by
  sorry

end find_a_l652_65244


namespace minimum_difference_l652_65270

def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem minimum_difference (x y z : ℤ) 
  (hx : even x) (hy : odd y) (hz : odd z)
  (hxy : x < y) (hyz : y < z) (hzx : z - x = 9) : y - x = 1 := 
sorry

end minimum_difference_l652_65270


namespace percentage_difference_l652_65276

variable (x y z : ℝ)

theorem percentage_difference (h1 : y = 1.75 * x) (h2 : z = 0.60 * y) :
  (1 - x / z) * 100 = 4.76 :=
by
  sorry

end percentage_difference_l652_65276


namespace smaller_solid_volume_l652_65295

noncomputable def cube_edge_length : ℝ := 2

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def D := point 0 0 0
def M := point 1 2 0
def N := point 2 0 1

-- Define the condition for the plane that passes through D, M, and N
def plane (p r q : ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  let (px, py, pz) := p
  let (rx, ry, rz) := r
  let (qx, qy, qz) := q
  2 * x - 4 * y - 8 * z = 0

-- Predicate to test if point is on a plane
def on_plane (pt : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := pt
  plane D M N x y z

-- Volume of the smaller solid
theorem smaller_solid_volume :
  ∃ V : ℝ, V = 1 / 6 :=
by
  sorry

end smaller_solid_volume_l652_65295


namespace equal_numbers_l652_65249

theorem equal_numbers {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + da) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end equal_numbers_l652_65249


namespace crayons_per_row_correct_l652_65284

-- Declare the given conditions
def total_crayons : ℕ := 210
def num_rows : ℕ := 7

-- Define the expected number of crayons per row
def crayons_per_row : ℕ := 30

-- The desired proof statement: Prove that dividing total crayons by the number of rows yields the expected crayons per row.
theorem crayons_per_row_correct : total_crayons / num_rows = crayons_per_row :=
by sorry

end crayons_per_row_correct_l652_65284


namespace largest_c_value_l652_65205

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 3 * x + c

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, f x c = -2) ↔ c ≤ 1/4 := by
sorry

end largest_c_value_l652_65205


namespace complex_fraction_value_l652_65243

theorem complex_fraction_value :
  1 + (1 / (2 + (1 / (2 + 2)))) = 13 / 9 :=
by
  sorry

end complex_fraction_value_l652_65243


namespace fernanda_total_time_eq_90_days_l652_65217

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l652_65217


namespace smallest_integer_square_l652_65271

theorem smallest_integer_square (x : ℤ) (h : x^2 = 2 * x + 75) : x = -7 :=
  sorry

end smallest_integer_square_l652_65271


namespace second_year_associates_l652_65216

theorem second_year_associates (not_first_year : ℝ) (more_than_two_years : ℝ) 
  (h1 : not_first_year = 0.75) (h2 : more_than_two_years = 0.5) : 
  (not_first_year - more_than_two_years) = 0.25 :=
by 
  sorry

end second_year_associates_l652_65216


namespace central_angle_measure_l652_65293

-- Given the problem definitions
variables (A : ℝ) (x : ℝ)

-- Condition: The probability of landing in the region is 1/8
def probability_condition : Prop :=
  (1 / 8 : ℝ) = (x / 360)

-- The final theorem to prove
theorem central_angle_measure (h : probability_condition x) : x = 45 := 
  sorry

end central_angle_measure_l652_65293


namespace number_of_raised_beds_l652_65292

def length_feed := 8
def width_feet := 4
def height_feet := 1
def cubic_feet_per_bag := 4
def total_bags_needed := 16

theorem number_of_raised_beds :
  ∀ (length_feed width_feet height_feet : ℕ) (cubic_feet_per_bag total_bags_needed : ℕ),
    (length_feed * width_feet * height_feet) / cubic_feet_per_bag = 8 →
    total_bags_needed / (8 : ℕ) = 2 :=
by sorry

end number_of_raised_beds_l652_65292


namespace difference_between_min_and_max_l652_65298

noncomputable 
def minValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 0

noncomputable
def maxValue (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ := 1.5

theorem difference_between_min_and_max (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  maxValue x z hx hz - minValue x z hx hz = 1.5 :=
by
  sorry

end difference_between_min_and_max_l652_65298


namespace exist_indices_l652_65212

-- Define the sequence and the conditions.
variable (x : ℕ → ℤ)
variable (H1 : x 1 = 1)
variable (H2 : ∀ n : ℕ, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n)

theorem exist_indices (k : ℕ) (hk : 0 < k) :
  ∃ r s : ℕ, x r - x s = k := 
sorry

end exist_indices_l652_65212


namespace cylindrical_container_depth_l652_65250

theorem cylindrical_container_depth :
    ∀ (L D A : ℝ), 
      L = 12 ∧ D = 8 ∧ A = 48 → (∃ h : ℝ, h = 4 - 2 * Real.sqrt 3) :=
by
  intros L D A h_cond
  obtain ⟨hL, hD, hA⟩ := h_cond
  sorry

end cylindrical_container_depth_l652_65250


namespace multiplication_with_mixed_number_l652_65230

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l652_65230


namespace combined_tax_rate_35_58_l652_65222

noncomputable def combined_tax_rate (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  (total_tax / total_income) * 100

theorem combined_tax_rate_35_58
  (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h1 : john_income = 57000) (h2 : john_tax_rate = 0.3)
  (h3 : ingrid_income = 72000) (h4 : ingrid_tax_rate = 0.4) :
  combined_tax_rate john_income john_tax_rate ingrid_income ingrid_tax_rate = 35.58 :=
by
  simp [combined_tax_rate, h1, h2, h3, h4]
  sorry

end combined_tax_rate_35_58_l652_65222


namespace cleaner_used_after_30_minutes_l652_65291

-- Define function to calculate the total amount of cleaner used
def total_cleaner_used (time: ℕ) (rate1: ℕ) (time1: ℕ) (rate2: ℕ) (time2: ℕ) (rate3: ℕ) (time3: ℕ) : ℕ :=
  (rate1 * time1) + (rate2 * time2) + (rate3 * time3)

-- The main theorem statement
theorem cleaner_used_after_30_minutes : total_cleaner_used 30 2 15 3 10 4 5 = 80 := by
  -- insert proof here
  sorry

end cleaner_used_after_30_minutes_l652_65291


namespace PJ_approx_10_81_l652_65232

noncomputable def PJ_length (P Q R J : Type) (PQ PR QR : ℝ) : ℝ :=
  if PQ = 30 ∧ PR = 29 ∧ QR = 27 then 10.81 else 0

theorem PJ_approx_10_81 (P Q R J : Type) (PQ PR QR : ℝ):
  PQ = 30 ∧ PR = 29 ∧ QR = 27 → PJ_length P Q R J PQ PR QR = 10.81 :=
by sorry

end PJ_approx_10_81_l652_65232


namespace muffin_banana_costs_l652_65294

variable (m b : ℕ) -- Using natural numbers for non-negativity

theorem muffin_banana_costs (h : 3 * (3 * m + 5 * b) = 4 * m + 10 * b) : m = b :=
by
  sorry

end muffin_banana_costs_l652_65294


namespace smallest_nat_div3_and_5_rem1_l652_65246

theorem smallest_nat_div3_and_5_rem1 : ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ ∀ M : ℕ, M > 1 ∧ (M % 3 = 1) ∧ (M % 5 = 1) → N ≤ M := 
by
  sorry

end smallest_nat_div3_and_5_rem1_l652_65246


namespace length_of_second_train_l652_65206

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_seconds : ℝ)
  (same_direction : Bool) : 
  length_first_train = 380 ∧ 
  speed_first_train_kmph = 72 ∧ 
  speed_second_train_kmph = 36 ∧ 
  time_seconds = 91.9926405887529 ∧ 
  same_direction = tt → 
  ∃ L2 : ℝ, L2 = 539.93 := by
  intro h
  rcases h with ⟨hf, sf, ss, ts, sd⟩
  use 539.926405887529 -- exact value obtained in the solution
  sorry

end length_of_second_train_l652_65206


namespace unique_coprime_solution_l652_65281

theorem unique_coprime_solution 
  (p : ℕ) (a b m r : ℕ) 
  (hp : Nat.Prime p) 
  (hp_odd : p % 2 = 1)
  (hp_nmid_ab : ¬ (p ∣ a * b))
  (hab_gt_m2 : a * b > m^2) :
  ∃! (x y : ℕ), Nat.Coprime x y ∧ (a * x^2 + b * y^2 = m * p ^ r) := 
sorry

end unique_coprime_solution_l652_65281


namespace circle_value_l652_65264

theorem circle_value (c d s : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y + d)^2 = s^2)
  (h2 : c = 4)
  (h3 : d = -8)
  (h4 : s = 2 * Real.sqrt 5) :
  c + d + s = -4 + 2 * Real.sqrt 5 := 
sorry

end circle_value_l652_65264


namespace parallel_transitivity_l652_65259

variable (Line Plane : Type)
variable (m n : Line)
variable (α : Plane)

-- Definitions for parallelism
variable (parallel : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Conditions
variable (m_n_parallel : parallel m n)
variable (m_alpha_parallel : parallelLinePlane m α)
variable (n_outside_alpha : ¬ parallelLinePlane n α)

-- Proposition to be proved
theorem parallel_transitivity (m n : Line) (α : Plane) 
  (h1 : parallel m n) 
  (h2 : parallelLinePlane m α) 
  : parallelLinePlane n α :=
sorry 

end parallel_transitivity_l652_65259


namespace fraction_of_boys_among_attendees_l652_65261

def boys : ℕ := sorry
def girls : ℕ := boys
def teachers : ℕ := boys / 2

def boys_attending : ℕ := (4 * boys) / 5
def girls_attending : ℕ := girls / 2
def teachers_attending : ℕ := teachers / 10

theorem fraction_of_boys_among_attendees :
  (boys_attending : ℚ) / (boys_attending + girls_attending + teachers_attending) = 16 / 27 := sorry

end fraction_of_boys_among_attendees_l652_65261


namespace major_axis_double_minor_axis_l652_65279

-- Define the radius of the right circular cylinder.
def cylinder_radius := 2

-- Define the minor axis length based on the cylinder's radius.
def minor_axis_length := 2 * cylinder_radius

-- Define the major axis length as double the minor axis length.
def major_axis_length := 2 * minor_axis_length

-- State the theorem to prove the major axis length.
theorem major_axis_double_minor_axis : major_axis_length = 8 := by
  sorry

end major_axis_double_minor_axis_l652_65279


namespace rectangle_distances_sum_l652_65226

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem rectangle_distances_sum :
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (3, 0)
  let C : (ℝ × ℝ) := (3, 4)
  let D : (ℝ × ℝ) := (0, 4)

  let M : (ℝ × ℝ) := ((B.1 + A.1) / 2, (B.2 + A.2) / 2)
  let N : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : (ℝ × ℝ) := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : (ℝ × ℝ) := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  distance A.1 A.2 M.1 M.2 + distance A.1 A.2 N.1 N.2 + distance A.1 A.2 O.1 O.2 + distance A.1 A.2 P.1 P.2 = 7.77 + Real.sqrt 13 :=
sorry

end rectangle_distances_sum_l652_65226
