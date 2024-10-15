import Mathlib

namespace NUMINAMATH_GPT_airport_exchange_rate_frac_l758_75855

variable (euros_received : ℕ) (euros : ℕ) (official_exchange_rate : ℕ) (dollars_received : ℕ)

theorem airport_exchange_rate_frac (h1 : euros = 70) (h2 : official_exchange_rate = 5) (h3 : dollars_received = 10) :
  (euros_received * dollars_received) = (euros * official_exchange_rate) →
  euros_received = 5 / 7 :=
  sorry

end NUMINAMATH_GPT_airport_exchange_rate_frac_l758_75855


namespace NUMINAMATH_GPT_find_a_l758_75851

theorem find_a (a : ℝ) :
  (∃ b : ℝ, 4 * b + 3 = 7 ∧ 5 * (-b) - 1 = 2 * (-b) + a) → a = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l758_75851


namespace NUMINAMATH_GPT_find_number_l758_75897

theorem find_number (x : ℝ) : 
  0.05 * x = 0.20 * 650 + 190 → x = 6400 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l758_75897


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l758_75809

theorem convert_to_scientific_notation :
  (26.62 * 10^9) = 2.662 * 10^9 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l758_75809


namespace NUMINAMATH_GPT_simplify_fraction_l758_75843

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l758_75843


namespace NUMINAMATH_GPT_max_area_rectangle_l758_75846

-- Define the conditions using Lean
def is_rectangle (length width : ℕ) : Prop :=
  2 * (length + width) = 34

-- Define the problem as a theorem in Lean
theorem max_area_rectangle : ∃ (length width : ℕ), is_rectangle length width ∧ length * width = 72 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l758_75846


namespace NUMINAMATH_GPT_painting_colors_area_l758_75871

theorem painting_colors_area
  (B G Y : ℕ)
  (h_total_blue : B + (1 / 3 : ℝ) * G = 38)
  (h_total_yellow : Y + (2 / 3 : ℝ) * G = 38)
  (h_grass_sky_relation : G = B + 6) :
  B = 27 ∧ G = 33 ∧ Y = 16 :=
by
  sorry

end NUMINAMATH_GPT_painting_colors_area_l758_75871


namespace NUMINAMATH_GPT_max_value_of_expression_max_value_achieved_l758_75882

theorem max_value_of_expression (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
    8 * x + 3 * y + 10 * z ≤ Real.sqrt 173 :=
sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1)
    (hx : x = Real.sqrt 173 / 30)
    (hy : y = Real.sqrt 173 / 20)
    (hz : z = Real.sqrt 173 / 50) :
    8 * x + 3 * y + 10 * z = Real.sqrt 173 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_max_value_achieved_l758_75882


namespace NUMINAMATH_GPT_measure_of_angle_F_l758_75817

theorem measure_of_angle_F (angle_D angle_E angle_F : ℝ) (h1 : angle_D = 80)
  (h2 : angle_E = 4 * angle_F + 10)
  (h3 : angle_D + angle_E + angle_F = 180) : angle_F = 18 := 
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_F_l758_75817


namespace NUMINAMATH_GPT_josh_money_left_l758_75836

def initial_amount : ℝ := 20
def cost_hat : ℝ := 10
def cost_pencil : ℝ := 2
def number_of_cookies : ℝ := 4
def cost_per_cookie : ℝ := 1.25

theorem josh_money_left : initial_amount - cost_hat - cost_pencil - (number_of_cookies * cost_per_cookie) = 3 := by
  sorry

end NUMINAMATH_GPT_josh_money_left_l758_75836


namespace NUMINAMATH_GPT_loss_percentage_l758_75854

theorem loss_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 1500) (h_sell : selling_price = 1260) : 
  (cost_price - selling_price) / cost_price * 100 = 16 := 
by
  sorry

end NUMINAMATH_GPT_loss_percentage_l758_75854


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l758_75863

noncomputable def solve_inequality (a b : ℝ) : Prop :=
  (∀ x : ℝ, (x > -1/2 ∧ x < 1/3) → (a * x^2 + b * x + 2 > 0)) →
  (a = -12) ∧ (b = -2)

theorem quadratic_inequality_solution :
   solve_inequality (-12) (-2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l758_75863


namespace NUMINAMATH_GPT_integer_solutions_l758_75896

-- Define the polynomial equation as a predicate
def polynomial (n : ℤ) : Prop := n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0

-- The theorem statement
theorem integer_solutions :
  {n : ℤ | polynomial n} = {-1, 3} :=
by 
  sorry

end NUMINAMATH_GPT_integer_solutions_l758_75896


namespace NUMINAMATH_GPT_volleyball_team_math_count_l758_75870

theorem volleyball_team_math_count (total_players taking_physics taking_both : ℕ) 
  (h1 : total_players = 30) 
  (h2 : taking_physics = 15) 
  (h3 : taking_both = 6) 
  (h4 : total_players = 30 ∧ total_players = (taking_physics + (total_players - taking_physics - taking_both))) 
  : (total_players - (taking_physics - taking_both) + taking_both) = 21 := 
by
  sorry

end NUMINAMATH_GPT_volleyball_team_math_count_l758_75870


namespace NUMINAMATH_GPT_white_sox_wins_l758_75834

theorem white_sox_wins 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (games_lost : ℕ)
  (win_loss_difference : ℤ) 
  (total_games_condition : total_games = 162) 
  (lost_games_condition : games_lost = 63) 
  (win_loss_diff_condition : (games_won : ℤ) - games_lost = win_loss_difference) 
  (win_loss_difference_value : win_loss_difference = 36) 
  : games_won = 99 :=
by
  sorry

end NUMINAMATH_GPT_white_sox_wins_l758_75834


namespace NUMINAMATH_GPT_workshops_participation_l758_75819

variable (x y z a b c d : ℕ)
variable (A B C : Finset ℕ)

theorem workshops_participation:
  (A.card = 15) →
  (B.card = 14) →
  (C.card = 11) →
  (25 = x + y + z + a + b + c + d) →
  (12 = a + b + c + d) →
  (A.card = x + a + c + d) →
  (B.card = y + a + b + d) →
  (C.card = z + b + c + d) →
  d = 0 :=
by
  intro hA hB hC hTotal hAtLeastTwo hAkA hBkA hCkA
  -- The proof will go here
  -- Parsing these inputs shall lead to establishing d = 0
  sorry

end NUMINAMATH_GPT_workshops_participation_l758_75819


namespace NUMINAMATH_GPT_correct_calculation_l758_75816

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end NUMINAMATH_GPT_correct_calculation_l758_75816


namespace NUMINAMATH_GPT_complex_unit_circle_sum_l758_75815

theorem complex_unit_circle_sum :
  let z1 := (1 + Complex.I * Real.sqrt 3) / 2
  let z2 := (1 - Complex.I * Real.sqrt 3) / 2
  (z1 ^ 8 + z2 ^ 8 = -1) :=
by
  sorry

end NUMINAMATH_GPT_complex_unit_circle_sum_l758_75815


namespace NUMINAMATH_GPT_find_OP_l758_75862

variable (a b c d e f : ℝ)
variable (P : ℝ)

-- Given conditions
axiom AP_PD_ratio : (a - P) / (P - d) = 2 / 3
axiom BP_PC_ratio : (b - P) / (P - c) = 3 / 4

-- Conclusion to prove
theorem find_OP : P = (3 * a + 2 * d) / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_OP_l758_75862


namespace NUMINAMATH_GPT_bill_has_six_times_more_nuts_l758_75899

-- Definitions for the conditions
def sue_has_nuts : ℕ := 48
def harry_has_nuts (sueNuts : ℕ) : ℕ := 2 * sueNuts
def combined_nuts (harryNuts : ℕ) (billNuts : ℕ) : ℕ := harryNuts + billNuts
def bill_has_nuts (totalNuts : ℕ) (harryNuts : ℕ) : ℕ := totalNuts - harryNuts

-- Statement to prove
theorem bill_has_six_times_more_nuts :
  ∀ sueNuts billNuts harryNuts totalNuts,
    sueNuts = sue_has_nuts →
    harryNuts = harry_has_nuts sueNuts →
    totalNuts = 672 →
    combined_nuts harryNuts billNuts = totalNuts →
    billNuts = bill_has_nuts totalNuts harryNuts →
    billNuts = 6 * harryNuts :=
by
  intros sueNuts billNuts harryNuts totalNuts hsueNuts hharryNuts htotalNuts hcombinedNuts hbillNuts
  sorry

end NUMINAMATH_GPT_bill_has_six_times_more_nuts_l758_75899


namespace NUMINAMATH_GPT_axis_of_symmetry_l758_75835

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 5

-- Define the statement that we need to prove
theorem axis_of_symmetry : (∃ (a : ℝ), ∀ x, parabola (x) = (x - a) ^ 2 + 4) ∧ 
                           (∃ (b : ℝ), b = 1) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l758_75835


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_by_ten_million_l758_75810

theorem smallest_n_for_divisibility_by_ten_million 
  (a₁ a₂ : ℝ) 
  (a₁_eq : a₁ = 5 / 6) 
  (a₂_eq : a₂ = 30) 
  (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : ∀ (k : ℕ), T k = a₁ * (36 ^ (k - 1))) :
  (∃ n, T n = T 9 ∧ (∃ m : ℤ, T n = m * 10^7)) := 
sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_by_ten_million_l758_75810


namespace NUMINAMATH_GPT_percentage_l_75_m_l758_75813

theorem percentage_l_75_m
  (j k l m : ℝ)
  (x : ℝ)
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : (x / 100) * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 175 :=
by
  sorry

end NUMINAMATH_GPT_percentage_l_75_m_l758_75813


namespace NUMINAMATH_GPT_Kishore_education_expense_l758_75890

theorem Kishore_education_expense
  (rent milk groceries petrol misc saved : ℝ) -- expenses
  (total_saved_salary : ℝ) -- percentage of saved salary
  (saving_amount : ℝ) -- actual saving
  (total_salary total_expense_children_education : ℝ) -- total salary and expense on children's education
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : petrol = 2000)
  (H5 : misc = 3940)
  (H6 : total_saved_salary = 0.10)
  (H7 : saving_amount = 2160)
  (H8 : total_salary = saving_amount / total_saved_salary)
  (H9 : total_expense_children_education = total_salary - (rent + milk + groceries + petrol + misc) - saving_amount) :
  total_expense_children_education = 2600 :=
by 
  simp only [H1, H2, H3, H4, H5, H6, H7] at *
  norm_num at *
  sorry

end NUMINAMATH_GPT_Kishore_education_expense_l758_75890


namespace NUMINAMATH_GPT_value_of_f_10_l758_75849

def f (n : ℕ) : ℕ := n^2 - n + 17

theorem value_of_f_10 : f 10 = 107 := by
  sorry

end NUMINAMATH_GPT_value_of_f_10_l758_75849


namespace NUMINAMATH_GPT_at_least_two_of_three_equations_have_solutions_l758_75874

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end NUMINAMATH_GPT_at_least_two_of_three_equations_have_solutions_l758_75874


namespace NUMINAMATH_GPT_index_card_area_l758_75812

theorem index_card_area (a b : ℕ) (new_area : ℕ) (reduce_length reduce_width : ℕ)
  (original_length : a = 3) (original_width : b = 7)
  (reduced_area_condition : a * (b - reduce_width) = new_area)
  (reduce_width_2 : reduce_width = 2) 
  (new_area_correct : new_area = 15) :
  (a - reduce_length) * b = 7 := by
  sorry

end NUMINAMATH_GPT_index_card_area_l758_75812


namespace NUMINAMATH_GPT_positive_divisors_multiple_of_5_l758_75841

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end NUMINAMATH_GPT_positive_divisors_multiple_of_5_l758_75841


namespace NUMINAMATH_GPT_determine_values_l758_75814

theorem determine_values (x y : ℝ) (h1 : x - y = 25) (h2 : x * y = 36) : (x^2 + y^2 = 697) ∧ (x + y = Real.sqrt 769) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_determine_values_l758_75814


namespace NUMINAMATH_GPT_sequence_term_1000_l758_75856

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end NUMINAMATH_GPT_sequence_term_1000_l758_75856


namespace NUMINAMATH_GPT_cos_6theta_l758_75825

theorem cos_6theta (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (6 * θ) = -3224/4096 := 
by
  sorry

end NUMINAMATH_GPT_cos_6theta_l758_75825


namespace NUMINAMATH_GPT_sin_330_degree_l758_75802

theorem sin_330_degree : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_330_degree_l758_75802


namespace NUMINAMATH_GPT_minimum_discount_percentage_l758_75852

theorem minimum_discount_percentage (cost_price marked_price : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  cost_price = 400 ∧ marked_price = 600 ∧ profit_margin = 0.05 ∧ 
  (marked_price * (1 - discount / 100) - cost_price) / cost_price ≥ profit_margin → discount ≤ 30 := 
by
  intros h
  rcases h with ⟨hc, hm, hp, hineq⟩
  sorry

end NUMINAMATH_GPT_minimum_discount_percentage_l758_75852


namespace NUMINAMATH_GPT_find_unknown_number_l758_75880

theorem find_unknown_number (x : ℕ) :
  (x + 30 + 50) / 3 = ((20 + 40 + 6) / 3 + 8) → x = 10 := by
    sorry

end NUMINAMATH_GPT_find_unknown_number_l758_75880


namespace NUMINAMATH_GPT_intercept_sum_mod_7_l758_75832

theorem intercept_sum_mod_7 :
  ∃ (x_0 y_0 : ℤ), (2 * x_0 ≡ 3 * y_0 + 1 [ZMOD 7]) ∧ (0 ≤ x_0) ∧ (x_0 < 7) ∧ (0 ≤ y_0) ∧ (y_0 < 7) ∧ (x_0 + y_0 = 6) :=
by
  sorry

end NUMINAMATH_GPT_intercept_sum_mod_7_l758_75832


namespace NUMINAMATH_GPT_yellow_sweets_l758_75826

-- Definitions
def green_sweets : Nat := 212
def blue_sweets : Nat := 310
def sweets_per_person : Nat := 256
def people : Nat := 4

-- Proof problem statement
theorem yellow_sweets : green_sweets + blue_sweets + x = sweets_per_person * people → x = 502 := by
  sorry

end NUMINAMATH_GPT_yellow_sweets_l758_75826


namespace NUMINAMATH_GPT_inequality_least_n_l758_75821

theorem inequality_least_n (n : ℕ) (h : (1 : ℝ) / n - (1 : ℝ) / (n + 2) < 1 / 15) : n = 5 :=
sorry

end NUMINAMATH_GPT_inequality_least_n_l758_75821


namespace NUMINAMATH_GPT_mobile_purchase_price_l758_75806

theorem mobile_purchase_price (M : ℝ) 
  (P_grinder : ℝ := 15000)
  (L_grinder : ℝ := 0.05 * P_grinder)
  (SP_grinder : ℝ := P_grinder - L_grinder)
  (SP_mobile : ℝ := 1.1 * M)
  (P_overall : ℝ := P_grinder + M)
  (SP_overall : ℝ := SP_grinder + SP_mobile)
  (profit : ℝ := 50)
  (h : SP_overall = P_overall + profit) :
  M = 8000 :=
by 
  sorry

end NUMINAMATH_GPT_mobile_purchase_price_l758_75806


namespace NUMINAMATH_GPT_inequality_holds_l758_75884

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * z) + y^3 / (y^3 + 2 * z^2 * x) + z^3 / (z^3 + 2 * x^2 * y)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l758_75884


namespace NUMINAMATH_GPT_green_duck_percentage_l758_75803

noncomputable def smaller_pond_ducks : ℕ := 45
noncomputable def larger_pond_ducks : ℕ := 55
noncomputable def green_percentage_small_pond : ℝ := 0.20
noncomputable def green_percentage_large_pond : ℝ := 0.40

theorem green_duck_percentage :
  let total_ducks := smaller_pond_ducks + larger_pond_ducks
  let green_ducks_smaller := green_percentage_small_pond * (smaller_pond_ducks : ℝ)
  let green_ducks_larger := green_percentage_large_pond * (larger_pond_ducks : ℝ)
  let total_green_ducks := green_ducks_smaller + green_ducks_larger
  (total_green_ducks / total_ducks) * 100 = 31 :=
by {
  -- The proof is omitted.
  sorry
}

end NUMINAMATH_GPT_green_duck_percentage_l758_75803


namespace NUMINAMATH_GPT_platform_length_l758_75889

theorem platform_length
    (train_length : ℕ)
    (time_to_cross_tree : ℕ)
    (speed : ℕ)
    (time_to_pass_platform : ℕ)
    (platform_length : ℕ) :
    train_length = 1200 →
    time_to_cross_tree = 120 →
    speed = train_length / time_to_cross_tree →
    time_to_pass_platform = 150 →
    speed * time_to_pass_platform = train_length + platform_length →
    platform_length = 300 :=
by
  intros h_train_length h_time_to_cross_tree h_speed h_time_to_pass_platform h_pass_platform_eq
  sorry

end NUMINAMATH_GPT_platform_length_l758_75889


namespace NUMINAMATH_GPT_percentage_orange_juice_in_blend_l758_75848

theorem percentage_orange_juice_in_blend :
  let pear_juice_per_pear := 10 / 2
  let orange_juice_per_orange := 8 / 2
  let pear_juice := 2 * pear_juice_per_pear
  let orange_juice := 3 * orange_juice_per_orange
  let total_juice := pear_juice + orange_juice
  (orange_juice / total_juice) = (6 / 11) := 
by
  sorry

end NUMINAMATH_GPT_percentage_orange_juice_in_blend_l758_75848


namespace NUMINAMATH_GPT_john_expenditure_l758_75830

theorem john_expenditure (X : ℝ) (h : (1/2) * X + (1/3) * X + (1/10) * X + 8 = X) : X = 120 :=
by
  sorry

end NUMINAMATH_GPT_john_expenditure_l758_75830


namespace NUMINAMATH_GPT_find_value_of_X_l758_75865

theorem find_value_of_X :
  let X_initial := 5
  let S_initial := 0
  let X_increment := 3
  let target_sum := 15000
  let X := X_initial + X_increment * 56
  2 * target_sum ≥ 3 * 57 * 57 + 7 * 57 →
  X = 173 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_X_l758_75865


namespace NUMINAMATH_GPT_painting_cost_in_cny_l758_75859

theorem painting_cost_in_cny (usd_to_nad : ℝ) (usd_to_cny : ℝ) (painting_cost_nad : ℝ) :
  usd_to_nad = 8 → usd_to_cny = 7 → painting_cost_nad = 160 →
  painting_cost_nad / usd_to_nad * usd_to_cny = 140 :=
by
  intros
  sorry

end NUMINAMATH_GPT_painting_cost_in_cny_l758_75859


namespace NUMINAMATH_GPT_sequence_sum_S6_l758_75860

theorem sequence_sum_S6 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - 3) :
  S_n 6 = 189 :=
by
  sorry

end NUMINAMATH_GPT_sequence_sum_S6_l758_75860


namespace NUMINAMATH_GPT_ondra_homework_problems_l758_75829

theorem ondra_homework_problems (a b c d : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (-a) * (-b) ≠ -a - b ∧ 
  (-c) * (-d) = -182 * (1 / (-c - d)) →
  ((a = 2 ∧ b = 2) 
  ∨ (c = 1 ∧ d = 13) 
  ∨ (c = 13 ∧ d = 1)) :=
sorry

end NUMINAMATH_GPT_ondra_homework_problems_l758_75829


namespace NUMINAMATH_GPT_number_of_boys_l758_75824

theorem number_of_boys (x : ℕ) (y : ℕ) (h1 : x + y = 8) (h2 : y > x) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l758_75824


namespace NUMINAMATH_GPT_no_intersection_points_l758_75867

theorem no_intersection_points :
  ∀ x y : ℝ, y = abs (3 * x + 6) ∧ y = -2 * abs (2 * x - 1) → false :=
by
  intros x y h
  cases h
  sorry

end NUMINAMATH_GPT_no_intersection_points_l758_75867


namespace NUMINAMATH_GPT_hypotenuse_longer_side_difference_l758_75842

theorem hypotenuse_longer_side_difference
  (x : ℝ)
  (h1 : 17^2 = x^2 + (x - 7)^2)
  (h2 : x = 15)
  : 17 - x = 2 := by
  sorry

end NUMINAMATH_GPT_hypotenuse_longer_side_difference_l758_75842


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l758_75844

-- Define the main problem as a Lean theorem statement
theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 5| + |x + 3| ≥ 10 ↔ (x ≤ -4 ∨ x ≥ 6)) :=
by {
  sorry
}

end NUMINAMATH_GPT_abs_inequality_solution_set_l758_75844


namespace NUMINAMATH_GPT_veronica_initial_marbles_l758_75877

variable {D M P V : ℕ}

theorem veronica_initial_marbles (hD : D = 14) (hM : M = 20) (hP : P = 19)
  (h_total : D + M + P + V = 60) : V = 7 :=
by
  sorry

end NUMINAMATH_GPT_veronica_initial_marbles_l758_75877


namespace NUMINAMATH_GPT_sue_necklace_total_beads_l758_75864

theorem sue_necklace_total_beads :
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  (purple + blue + green = 46) :=
by
  intros purple blue green h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_sue_necklace_total_beads_l758_75864


namespace NUMINAMATH_GPT_anthony_solve_l758_75805

def completing_square (a b c : ℤ) : ℤ :=
  let d := Int.sqrt a
  let e := b / (2 * d)
  let f := (d * e * e - c)
  d + e + f

theorem anthony_solve (d e f : ℤ) (h_d_pos : d > 0)
  (h_eqn : 25 * d * d + 30 * d * e - 72 = 0)
  (h_form : (d * x + e)^2 = f) : 
  d + e + f = 89 :=
by
  have d : ℤ := 5
  have e : ℤ := 3
  have f : ℤ := 81
  sorry

end NUMINAMATH_GPT_anthony_solve_l758_75805


namespace NUMINAMATH_GPT_minimum_value_l758_75881

-- Given conditions
variables (a b c d : ℝ)
variables (h_a : a > 0) (h_b : b = 0) (h_a_eq : a = 1)

-- Define the function
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- The statement to prove
theorem minimum_value (h_c : c = 0) : ∃ x : ℝ, f a b c d x = d :=
by
  -- Given the conditions a=1, b=0, and c=0, we need to show that the minimum value is d
  sorry

end NUMINAMATH_GPT_minimum_value_l758_75881


namespace NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_problem_E_l758_75827

-- Definitions and assumptions based on the problem statement
def eqI (x y z : ℕ) := x + y + z = 45
def eqII (x y z w : ℕ) := x + y + z + w = 50
def consecutive_odd_integers (x y z : ℕ) := y = x + 2 ∧ z = x + 4
def multiples_of_five (x y z w : ℕ) := (∃ a b c d : ℕ, x = 5 * a ∧ y = 5 * b ∧ z = 5 * c ∧ w = 5 * d)
def consecutive_integers (x y z w : ℕ) := y = x + 1 ∧ z = x + 2 ∧ w = x + 3
def prime_integers (x y z : ℕ) := Prime x ∧ Prime y ∧ Prime z

-- Lean theorem statements
theorem problem_A : ∃ x y z : ℕ, eqI x y z ∧ consecutive_odd_integers x y z := 
sorry

theorem problem_B : ¬ (∃ x y z : ℕ, eqI x y z ∧ prime_integers x y z) := 
sorry

theorem problem_C : ¬ (∃ x y z w : ℕ, eqII x y z w ∧ consecutive_odd_integers x y z) :=
sorry

theorem problem_D : ∃ x y z w : ℕ, eqII x y z w ∧ multiples_of_five x y z w := 
sorry

theorem problem_E : ∃ x y z w : ℕ, eqII x y z w ∧ consecutive_integers x y z w := 
sorry

end NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_problem_E_l758_75827


namespace NUMINAMATH_GPT_find_C_coordinates_l758_75838

noncomputable def maximize_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (x : ℝ) : Prop :=
  ∀ C : ℝ × ℝ, C = (x, 0) → x = Real.sqrt (a * b)

theorem find_C_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  maximize_angle a b ha hb hab (Real.sqrt (a * b)) :=
by  sorry

end NUMINAMATH_GPT_find_C_coordinates_l758_75838


namespace NUMINAMATH_GPT_find_n_eq_5_l758_75895

variable {a_n b_n : ℕ → ℤ}

def a (n : ℕ) : ℤ := 2 + 3 * (n - 1)
def b (n : ℕ) : ℤ := -2 + 4 * (n - 1)

theorem find_n_eq_5 :
  ∃ n : ℕ, a n = b n ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_eq_5_l758_75895


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l758_75868

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l758_75868


namespace NUMINAMATH_GPT_repeating_fraction_equality_l758_75850

theorem repeating_fraction_equality : (0.5656565656 : ℚ) = 56 / 99 :=
by
  sorry

end NUMINAMATH_GPT_repeating_fraction_equality_l758_75850


namespace NUMINAMATH_GPT_find_c1_minus_c2_l758_75800

theorem find_c1_minus_c2 (c1 c2 : ℝ) (h1 : 2 * 3 + 3 * 5 = c1) (h2 : 5 = c2) : c1 - c2 = 16 := by
  sorry

end NUMINAMATH_GPT_find_c1_minus_c2_l758_75800


namespace NUMINAMATH_GPT_positive_integers_square_of_sum_of_digits_l758_75892

-- Define the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem positive_integers_square_of_sum_of_digits :
  ∀ (n : ℕ), (n > 0) → (n = sum_of_digits n ^ 2) → (n = 1 ∨ n = 81) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_square_of_sum_of_digits_l758_75892


namespace NUMINAMATH_GPT_find_eccentricity_l758_75837

noncomputable def ellipse_eccentricity (m : ℝ) (c : ℝ) (a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity
  (m : ℝ) (c := Real.sqrt 2) (a := 3 * Real.sqrt 2 / 2)
  (h1 : 2 * m^2 - (m + 1) = 2)
  (h2 : m > 0) :
  ellipse_eccentricity m c a = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_find_eccentricity_l758_75837


namespace NUMINAMATH_GPT_large_doll_cost_is_8_l758_75873

-- Define the cost of the large monkey doll
def cost_large_doll : ℝ := 8

-- Define the total amount spent
def total_spent : ℝ := 320

-- Define the price difference between large and small dolls
def price_difference : ℝ := 4

-- Define the count difference between buying small dolls and large dolls
def count_difference : ℝ := 40

theorem large_doll_cost_is_8 
    (h1 : total_spent = 320)
    (h2 : ∀ L, L - price_difference = 4)
    (h3 : ∀ L, (total_spent / (L - 4)) = (total_spent / L) + count_difference) :
    cost_large_doll = 8 := 
by 
  sorry

end NUMINAMATH_GPT_large_doll_cost_is_8_l758_75873


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l758_75833

theorem remainder_when_divided_by_6 (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 6 = 4 :=
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l758_75833


namespace NUMINAMATH_GPT_max_log_value_l758_75853

noncomputable def max_log_product (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a * b = 8 then (Real.logb 2 a) * (Real.logb 2 (2 * b)) else 0

theorem max_log_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 8) :
  max_log_product a b ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_log_value_l758_75853


namespace NUMINAMATH_GPT_not_always_possible_repaint_all_white_l758_75872

-- Define the conditions and the problem
def equilateral_triangle_division (n: ℕ) : Prop := 
  ∀ m, m > 1 → m = n^2

def line_parallel_repaint (triangles : List ℕ) : Prop :=
  -- Definition of how the repaint operation affects the triangle colors
  sorry

theorem not_always_possible_repaint_all_white (n : ℕ) (h: equilateral_triangle_division n) :
  ¬∀ triangles, line_parallel_repaint triangles → (∀ t ∈ triangles, t = 0) := 
sorry

end NUMINAMATH_GPT_not_always_possible_repaint_all_white_l758_75872


namespace NUMINAMATH_GPT_trapezoid_diagonals_l758_75808

theorem trapezoid_diagonals (a c b d e f : ℝ) (h1 : a ≠ c):
  e^2 = a * c + (a * d^2 - c * b^2) / (a - c) ∧ 
  f^2 = a * c + (a * b^2 - c * d^2) / (a - c) := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_diagonals_l758_75808


namespace NUMINAMATH_GPT_find_k_l758_75878

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (hk : k ≠ 1) (h3 : 2 * a + b = a * b) : 
  k = 18 :=
sorry

end NUMINAMATH_GPT_find_k_l758_75878


namespace NUMINAMATH_GPT_max_non_overlapping_triangles_l758_75811

variable (L : ℝ) (n : ℕ)
def equilateral_triangle (L : ℝ) := true   -- Placeholder definition for equilateral triangle 
def non_overlapping_interior := true        -- Placeholder definition for non-overlapping condition
def unit_triangle_orientation_shift := true -- Placeholder for orientation condition

theorem max_non_overlapping_triangles (L_pos : 0 < L)
                                    (h1 : equilateral_triangle L)
                                    (h2 : ∀ i, i < n → non_overlapping_interior)
                                    (h3 : ∀ i, i < n → unit_triangle_orientation_shift) :
                                    n ≤ (2 : ℝ) / 3 * L^2 := 
by 
  sorry

end NUMINAMATH_GPT_max_non_overlapping_triangles_l758_75811


namespace NUMINAMATH_GPT_percent_of_percent_l758_75861

theorem percent_of_percent (y : ℝ) : 0.3 * (0.6 * y) = 0.18 * y :=
sorry

end NUMINAMATH_GPT_percent_of_percent_l758_75861


namespace NUMINAMATH_GPT_no_cube_sum_of_three_consecutive_squares_l758_75876

theorem no_cube_sum_of_three_consecutive_squares :
  ¬∃ x y : ℤ, x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by
  sorry

end NUMINAMATH_GPT_no_cube_sum_of_three_consecutive_squares_l758_75876


namespace NUMINAMATH_GPT_smallest_x_satisfying_equation_l758_75879

theorem smallest_x_satisfying_equation :
  ∀ x : ℝ, (2 * x ^ 2 + 24 * x - 60 = x * (x + 13)) → x = -15 ∨ x = 4 ∧ ∃ y : ℝ, y = -15 ∨ y = 4 ∧ y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_satisfying_equation_l758_75879


namespace NUMINAMATH_GPT_vendor_throws_away_8_percent_l758_75858

theorem vendor_throws_away_8_percent (total_apples: ℕ) (h₁ : total_apples > 0) :
    let apples_after_first_day := total_apples * 40 / 100
    let thrown_away_first_day := apples_after_first_day * 10 / 100
    let apples_after_second_day := (apples_after_first_day - thrown_away_first_day) * 30 / 100
    let thrown_away_second_day := apples_after_second_day * 20 / 100
    let apples_after_third_day := (apples_after_second_day - thrown_away_second_day) * 60 / 100
    let thrown_away_third_day := apples_after_third_day * 30 / 100
    total_apples > 0 → (8 : ℕ) * total_apples = (thrown_away_first_day + thrown_away_second_day + thrown_away_third_day) * 100 := 
by
    -- Placeholder proof
    sorry

end NUMINAMATH_GPT_vendor_throws_away_8_percent_l758_75858


namespace NUMINAMATH_GPT_bag_contains_n_black_balls_l758_75840

theorem bag_contains_n_black_balls (n : ℕ) : (5 / (n + 5) = 1 / 3) → n = 10 := by
  sorry

end NUMINAMATH_GPT_bag_contains_n_black_balls_l758_75840


namespace NUMINAMATH_GPT_oliver_final_amount_l758_75804

def initial_amount : ℤ := 33
def spent : ℤ := 4
def received : ℤ := 32

def final_amount (initial_amount spent received : ℤ) : ℤ :=
  initial_amount - spent + received

theorem oliver_final_amount : final_amount initial_amount spent received = 61 := 
by sorry

end NUMINAMATH_GPT_oliver_final_amount_l758_75804


namespace NUMINAMATH_GPT_visitors_on_monday_l758_75894

theorem visitors_on_monday (M : ℕ) (h : M + 2 * M + 100 = 250) : M = 50 :=
by
  sorry

end NUMINAMATH_GPT_visitors_on_monday_l758_75894


namespace NUMINAMATH_GPT_negation_of_existence_l758_75866

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l758_75866


namespace NUMINAMATH_GPT_reflect_parabola_y_axis_l758_75875

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end NUMINAMATH_GPT_reflect_parabola_y_axis_l758_75875


namespace NUMINAMATH_GPT_solve_for_x_l758_75820

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x = 600 - (4 * x + 6 * x) → x = 40 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l758_75820


namespace NUMINAMATH_GPT_product_of_real_roots_eq_one_l758_75847

theorem product_of_real_roots_eq_one:
  ∀ x : ℝ, (x ^ (Real.log x / Real.log 5) = 25) → (∀ x1 x2 : ℝ, (x1 ^ (Real.log x1 / Real.log 5) = 25) → (x2 ^ (Real.log x2 / Real.log 5) = 25) → x1 * x2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_product_of_real_roots_eq_one_l758_75847


namespace NUMINAMATH_GPT_range_of_m_l758_75801

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l758_75801


namespace NUMINAMATH_GPT_square_units_digit_l758_75857

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end NUMINAMATH_GPT_square_units_digit_l758_75857


namespace NUMINAMATH_GPT_expenditure_ratio_l758_75818

variable (P1 P2 : Type)
variable (I1 I2 E1 E2 : ℝ)
variable (R_incomes : I1 / I2 = 5 / 4)
variable (S1 S2 : ℝ)
variable (S_equal : S1 = S2)
variable (I1_fixed : I1 = 4000)
variable (Savings : S1 = 1600)

theorem expenditure_ratio :
  (I1 - E1 = 1600) → 
  (I2 * 4 / 5 - E2 = 1600) →
  I2 = 3200 →
  E1 / E2 = 3 / 2 :=
by
  intro P1_savings P2_savings I2_calc
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_expenditure_ratio_l758_75818


namespace NUMINAMATH_GPT_find_a_c_area_A_90_area_B_90_l758_75839

variable (a b c : ℝ)
variable (C : ℝ)

def triangle_condition1 := a + 1/a = 4 * Real.cos C
def triangle_condition2 := b = 1
def sin_C := Real.sin C = Real.sqrt 21 / 7

-- Proof problem for (1)
theorem find_a_c (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h3 : sin_C C) :
  (a = Real.sqrt 7 ∧ c = 2) ∨ (a = Real.sqrt 7 / 7 ∧ c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Conditions for (2) when A=90°
def right_triangle_A := C = Real.pi / 2

-- Proof problem for (2) when A=90°
theorem area_A_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h4 : right_triangle_A C) :
  ((a = Real.sqrt 3) → area = Real.sqrt 2 / 2) :=
sorry

-- Conditions for (2) when B=90°
def right_triangle_B := b = 1 ∧ C = Real.pi / 2

-- Proof problem for (2) when B=90°
theorem area_B_90 (h1 : triangle_condition1 a C) (h2 : triangle_condition2 b) (h5 : right_triangle_B b C) :
  ((a = Real.sqrt 3 / 3) → area = Real.sqrt 2 / 6) :=
sorry

end NUMINAMATH_GPT_find_a_c_area_A_90_area_B_90_l758_75839


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l758_75823

variable (a b c e : ℝ)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (hyperbola_eq : c = Real.sqrt (a^2 + b^2))
variable (y_B : ℝ)
variable (slope_eq : 3 = (y_B - 0) / (c - a))
variable (y_B_on_hyperbola : y_B = b^2 / a)

theorem hyperbola_eccentricity (h : a > 0) (h' : b > 0) (c_def : c = Real.sqrt (a^2 + b^2))
    (slope_cond : 3 = (y_B - 0) / (c - a)) (y_B_cond : y_B = b^2 / a) :
    e = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l758_75823


namespace NUMINAMATH_GPT_leftover_grass_seed_coverage_l758_75828

/-
Question: How many extra square feet could the leftover grass seed cover after Drew reseeds his lawn?

Conditions:
- One bag of grass seed covers 420 square feet of lawn.
- The lawn consists of a rectangular section and a triangular section.
- Rectangular section:
    - Length: 32 feet
    - Width: 45 feet
- Triangular section:
    - Base: 25 feet
    - Height: 20 feet
- Triangular section requires 1.5 times the standard coverage rate.
- Drew bought seven bags of seed.

Answer: The leftover grass seed coverage is 1125 square feet.
-/

theorem leftover_grass_seed_coverage
  (bag_coverage : ℕ := 420)
  (rect_length : ℕ := 32)
  (rect_width : ℕ := 45)
  (tri_base : ℕ := 25)
  (tri_height : ℕ := 20)
  (coverage_multiplier : ℕ := 15)  -- Using 15 instead of 1.5 for integer math
  (bags_bought : ℕ := 7) :
  (bags_bought * bag_coverage - 
    (rect_length * rect_width + tri_base * tri_height * coverage_multiplier / 20) = 1125) :=
  by {
    -- Placeholder for proof steps
    sorry
  }

end NUMINAMATH_GPT_leftover_grass_seed_coverage_l758_75828


namespace NUMINAMATH_GPT_find_8th_result_l758_75898

theorem find_8th_result 
  (S_17 : ℕ := 17 * 24) 
  (S_7 : ℕ := 7 * 18) 
  (S_5_1 : ℕ := 5 * 23) 
  (S_5_2 : ℕ := 5 * 32) : 
  S_17 - S_7 - S_5_1 - S_5_2 = 7 := 
by
  sorry

end NUMINAMATH_GPT_find_8th_result_l758_75898


namespace NUMINAMATH_GPT_reflection_of_C_over_y_eq_x_l758_75893

def point_reflection_over_yx := ∀ (A B C : (ℝ × ℝ)), 
  A = (6, 2) → 
  B = (2, 5) → 
  C = (2, 2) → 
  (reflect_y_eq_x C) = (2, 2)
where reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_C_over_y_eq_x :
  point_reflection_over_yx :=
by 
  sorry

end NUMINAMATH_GPT_reflection_of_C_over_y_eq_x_l758_75893


namespace NUMINAMATH_GPT_production_cost_per_performance_l758_75883

def overhead_cost := 81000
def income_per_performance := 16000
def performances_needed := 9

theorem production_cost_per_performance :
  ∃ P, 9 * income_per_performance = overhead_cost + 9 * P ∧ P = 7000 :=
by
  sorry

end NUMINAMATH_GPT_production_cost_per_performance_l758_75883


namespace NUMINAMATH_GPT_sin_cos_product_l758_75831

open Real

theorem sin_cos_product (θ : ℝ) (h : sin θ + cos θ = 3 / 4) : sin θ * cos θ = -7 / 32 := 
  by 
    sorry

end NUMINAMATH_GPT_sin_cos_product_l758_75831


namespace NUMINAMATH_GPT_tan_sin_cos_l758_75888

theorem tan_sin_cos (θ : ℝ) (h : Real.tan θ = 1 / 2) : 
  Real.sin (2 * θ) - 2 * Real.cos θ ^ 2 = - 4 / 5 := by 
  sorry

end NUMINAMATH_GPT_tan_sin_cos_l758_75888


namespace NUMINAMATH_GPT_faster_cow_days_to_eat_one_bag_l758_75869

-- Conditions as assumptions
def num_cows : ℕ := 60
def num_husks : ℕ := 150
def num_days : ℕ := 80
def faster_cows : ℕ := 20
def normal_cows : ℕ := num_cows - faster_cows
def faster_rate : ℝ := 1.3

-- The question translated to Lean 4 statement
theorem faster_cow_days_to_eat_one_bag :
  (faster_cows * faster_rate + normal_cows) / num_cows * (num_husks / num_days) = 1 / 27.08 :=
sorry

end NUMINAMATH_GPT_faster_cow_days_to_eat_one_bag_l758_75869


namespace NUMINAMATH_GPT_dice_impossible_divisible_by_10_l758_75845

theorem dice_impossible_divisible_by_10 :
  ¬ ∃ n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ), n % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_dice_impossible_divisible_by_10_l758_75845


namespace NUMINAMATH_GPT_red_balls_removal_l758_75887

theorem red_balls_removal (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (x : ℕ) :
  total_balls = 600 →
  red_balls = 420 →
  blue_balls = 180 →
  (red_balls - x) / (total_balls - x : ℚ) = 3 / 5 ↔ x = 150 :=
by 
  intros;
  sorry

end NUMINAMATH_GPT_red_balls_removal_l758_75887


namespace NUMINAMATH_GPT_exists_integers_m_n_l758_75886

theorem exists_integers_m_n (a b c p q r : ℝ) (h_a : a ≠ 0) (h_p : p ≠ 0) :
  ∃ (m n : ℤ), ∀ (x : ℝ), (a * x^2 + b * x + c = m * (p * x^2 + q * x + r) + n) := sorry

end NUMINAMATH_GPT_exists_integers_m_n_l758_75886


namespace NUMINAMATH_GPT_tan_subtraction_l758_75885

variable {α β : ℝ}

theorem tan_subtraction (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) : Real.tan (α - β) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_subtraction_l758_75885


namespace NUMINAMATH_GPT_circle_center_radius_l758_75822

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 4 → (x - h)^2 + (y - k)^2 = r^2) ∧
    h = -1 ∧ k = 1 ∧ r = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_radius_l758_75822


namespace NUMINAMATH_GPT_four_digit_sum_10_divisible_by_9_is_0_l758_75891

theorem four_digit_sum_10_divisible_by_9_is_0 : 
  ∀ (N : ℕ), (1000 * ((N / 1000) % 10) + 100 * ((N / 100) % 10) + 10 * ((N / 10) % 10) + (N % 10) = 10) ∧ (N % 9 = 0) → false :=
by
  sorry

end NUMINAMATH_GPT_four_digit_sum_10_divisible_by_9_is_0_l758_75891


namespace NUMINAMATH_GPT_hannahs_son_cuts_three_strands_per_minute_l758_75807

variable (x : ℕ)

theorem hannahs_son_cuts_three_strands_per_minute
  (total_strands : ℕ)
  (hannah_rate : ℕ)
  (total_time : ℕ)
  (total_strands_cut : ℕ := hannah_rate * total_time)
  (son_rate := (total_strands - total_strands_cut) / total_time)
  (hannah_rate := 8)
  (total_time := 2)
  (total_strands := 22) :
  son_rate = 3 := 
by
  sorry

end NUMINAMATH_GPT_hannahs_son_cuts_three_strands_per_minute_l758_75807
