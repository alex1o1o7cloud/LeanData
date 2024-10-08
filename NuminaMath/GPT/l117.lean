import Mathlib

namespace distance_missouri_to_new_york_by_car_l117_117791

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l117_117791


namespace discount_is_20_percent_l117_117272

noncomputable def discount_percentage 
  (puppy_cost : ℝ := 20.0)
  (dog_food_cost : ℝ := 20.0)
  (treat_cost : ℝ := 2.5)
  (num_treats : ℕ := 2)
  (toy_cost : ℝ := 15.0)
  (crate_cost : ℝ := 20.0)
  (bed_cost : ℝ := 20.0)
  (collar_leash_cost : ℝ := 15.0)
  (total_spent : ℝ := 96.0) : ℝ := 
  let total_cost_before_discount := dog_food_cost + (num_treats * treat_cost) + toy_cost + crate_cost + bed_cost + collar_leash_cost
  let spend_at_store := total_spent - puppy_cost
  let discount_amount := total_cost_before_discount - spend_at_store
  (discount_amount / total_cost_before_discount) * 100

theorem discount_is_20_percent : discount_percentage = 20 := sorry

end discount_is_20_percent_l117_117272


namespace solve_for_x_l117_117418

theorem solve_for_x : 
  ∀ x : ℝ, 
    (x ≠ 2) ∧ (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 → 
    x = -11 / 6 :=
by
  intro x
  intro h 
  sorry

end solve_for_x_l117_117418


namespace team_points_l117_117349

theorem team_points (wins losses ties : ℕ) (points_per_win points_per_loss points_per_tie : ℕ) :
  wins = 9 → losses = 3 → ties = 4 → points_per_win = 2 → points_per_loss = 0 → points_per_tie = 1 →
  (points_per_win * wins + points_per_loss * losses + points_per_tie * ties = 22) :=
by
  intro h_wins h_losses h_ties h_points_per_win h_points_per_loss h_points_per_tie
  sorry

end team_points_l117_117349


namespace largest_common_term_l117_117008

theorem largest_common_term (b : ℕ) (h1 : b ≡ 1 [MOD 3]) (h2 : b ≡ 2 [MOD 10]) (h3 : b < 300) : b = 290 :=
sorry

end largest_common_term_l117_117008


namespace geometric_sequence_term_l117_117232

theorem geometric_sequence_term :
  ∃ (a_n : ℕ → ℕ),
    -- common ratio condition
    (∀ n, a_n (n + 1) = 2 * a_n n) ∧
    -- sum of first 4 terms condition
    (a_n 1 + a_n 2 + a_n 3 + a_n 4 = 60) ∧
    -- conclusion: value of the third term
    (a_n 3 = 16) :=
by
  sorry

end geometric_sequence_term_l117_117232


namespace triangle_area_DEF_l117_117011

def point : Type := ℝ × ℝ

def D : point := (-2, 2)
def E : point := (8, 2)
def F : point := (6, -4)

theorem triangle_area_DEF :
  let base : ℝ := abs (D.1 - E.1)
  let height : ℝ := abs (F.2 - 2)
  let area := 1/2 * base * height
  area = 30 := 
by 
  sorry

end triangle_area_DEF_l117_117011


namespace calculate_expression_l117_117730

/-- Calculate the expression 2197 + 180 ÷ 60 × 3 - 197. -/
theorem calculate_expression : 2197 + (180 / 60) * 3 - 197 = 2009 := by
  sorry

end calculate_expression_l117_117730


namespace find_S2019_l117_117470

-- Conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Definitions and conditions extracted: conditions for sum of arithmetic sequence
axiom arithmetic_sum (n : ℕ) : S n = n * a (n / 2)
axiom OB_condition : a 3 + a 2017 = 1

-- Lean statement to prove S2019
theorem find_S2019 : S 2019 = 2019 / 2 := by
  sorry

end find_S2019_l117_117470


namespace shifted_parabola_expression_l117_117603

theorem shifted_parabola_expression (x y x' y' : ℝ) 
  (h_initial : y = (x + 2)^2 + 3)
  (h_shift_right : x' = x - 3)
  (h_shift_down : y' = y - 2)
  : y' = (x' - 1)^2 + 1 := 
sorry

end shifted_parabola_expression_l117_117603


namespace solve_fractional_equation_l117_117299

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 3) : (2 / (x - 3) = 3 / x) → x = 9 :=
by
  sorry

end solve_fractional_equation_l117_117299


namespace vans_hold_people_per_van_l117_117070

theorem vans_hold_people_per_van (students adults vans total_people people_per_van : ℤ) 
    (h1: students = 12) 
    (h2: adults = 3) 
    (h3: vans = 3) 
    (h4: total_people = students + adults) 
    (h5: people_per_van = total_people / vans) :
    people_per_van = 5 := 
by
    -- Steps will go here
    sorry

end vans_hold_people_per_van_l117_117070


namespace find_m_l117_117441

theorem find_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (4 / x) + (9 / y) = m) (h4 : ∃ x y , x + y = 5/6) : m = 30 :=
sorry

end find_m_l117_117441


namespace hawksbill_to_green_turtle_ratio_l117_117863

theorem hawksbill_to_green_turtle_ratio (total_turtles : ℕ) (green_turtles : ℕ) (hawksbill_turtles : ℕ) (h1 : green_turtles = 800) (h2 : total_turtles = 3200) (h3 : hawksbill_turtles = total_turtles - green_turtles) :
  hawksbill_turtles / green_turtles = 3 :=
by {
  sorry
}

end hawksbill_to_green_turtle_ratio_l117_117863


namespace difference_in_pencil_buyers_l117_117846

theorem difference_in_pencil_buyers :
  ∀ (cost_per_pencil : ℕ) (total_cost_eighth_graders : ℕ) (total_cost_fifth_graders : ℕ), 
  cost_per_pencil = 13 →
  total_cost_eighth_graders = 234 →
  total_cost_fifth_graders = 325 →
  (total_cost_fifth_graders / cost_per_pencil) - (total_cost_eighth_graders / cost_per_pencil) = 7 :=
by
  intros cost_per_pencil total_cost_eighth_graders total_cost_fifth_graders 
         hcpe htc8 htc5
  sorry

end difference_in_pencil_buyers_l117_117846


namespace even_and_odd_implies_zero_l117_117059

theorem even_and_odd_implies_zero (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = -f x) (h2 : ∀ x : ℝ, f (-x) = f x) : ∀ x : ℝ, f x = 0 :=
by
  sorry

end even_and_odd_implies_zero_l117_117059


namespace parallel_line_slope_l117_117047

theorem parallel_line_slope {x y : ℝ} (h : 3 * x + 6 * y = -24) : 
  ∀ m b : ℝ, (y = m * x + b) → m = -1 / 2 :=
sorry

end parallel_line_slope_l117_117047


namespace part_a_part_b_part_c_l117_117628

def transformable (w1 w2 : String) : Prop :=
∀ q : String → String → Prop,
  (q "xy" "yyx") →
  (q "xt" "ttx") →
  (q "yt" "ty") →
  (q w1 w2)

theorem part_a : ¬ transformable "xy" "xt" :=
sorry

theorem part_b : ¬ transformable "xytx" "txyt" :=
sorry

theorem part_c : transformable "xtxyy" "ttxyyyyx" :=
sorry

end part_a_part_b_part_c_l117_117628


namespace intersection_of_lines_l117_117538

theorem intersection_of_lines :
  ∃ (x y : ℚ), (8 * x - 3 * y = 24) ∧ (10 * x + 2 * y = 14) ∧ x = 45 / 23 ∧ y = -64 / 23 :=
by
  sorry

end intersection_of_lines_l117_117538


namespace total_rabbits_and_chickens_l117_117501

theorem total_rabbits_and_chickens (r c : ℕ) (h₁ : r = 64) (h₂ : r = c + 17) : r + c = 111 :=
by {
  sorry
}

end total_rabbits_and_chickens_l117_117501


namespace sum_coords_B_l117_117292

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end sum_coords_B_l117_117292


namespace sum_2019_l117_117483

noncomputable def a : ℕ → ℝ := sorry
def S (n : ℕ) : ℝ := sorry

axiom prop_1 : (a 2 - 1)^3 + (a 2 - 1) = 2019
axiom prop_2 : (a 2018 - 1)^3 + (a 2018 - 1) = -2019
axiom arithmetic_sequence : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom sum_formula : S 2019 = (2019 * (a 1 + a 2019)) / 2

theorem sum_2019 : S 2019 = 2019 :=
by sorry

end sum_2019_l117_117483


namespace value_of_y_l117_117298

theorem value_of_y :
  ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end value_of_y_l117_117298


namespace marie_messages_days_l117_117637

theorem marie_messages_days (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) (days : ℕ) :
  initial_messages = 98 ∧ read_per_day = 20 ∧ new_per_day = 6 → days = 7 :=
by
  sorry

end marie_messages_days_l117_117637


namespace max_side_length_l117_117372

theorem max_side_length (a b c : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 24)
  (h4 : b + c > a) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) : a ≤ 11 :=
by
  sorry

end max_side_length_l117_117372


namespace solve_triangle_l117_117972

theorem solve_triangle (a b : ℝ) (A B : ℝ) : ((A + B < π ∧ A > 0 ∧ B > 0 ∧ a > 0) ∨ (a > 0 ∧ b > 0 ∧ (π > A) ∧ (A > 0))) → ∃ c C, c > 0 ∧ (π > C) ∧ C > 0 :=
sorry

end solve_triangle_l117_117972


namespace compare_two_sqrt_three_l117_117409

theorem compare_two_sqrt_three : 2 > Real.sqrt 3 :=
by {
  sorry
}

end compare_two_sqrt_three_l117_117409


namespace characters_per_day_l117_117872

-- Definitions based on conditions
def chars_total_older : ℕ := 8000
def chars_total_younger : ℕ := 6000
def chars_per_day_diff : ℕ := 100

-- Define the main theorem
theorem characters_per_day (x : ℕ) :
  chars_total_older / x = chars_total_younger / (x - chars_per_day_diff) := 
sorry

end characters_per_day_l117_117872


namespace no_solution_abs_eq_2_l117_117797

theorem no_solution_abs_eq_2 (x : ℝ) :
  |x - 5| = |x + 3| + 2 → false :=
by sorry

end no_solution_abs_eq_2_l117_117797


namespace ellipse_standard_equation_l117_117040

theorem ellipse_standard_equation (c a : ℝ) (h1 : 2 * c = 8) (h2 : 2 * a = 10) : 
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ ( ( ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) ∨ ( ∀ x y : ℝ, x^2 / b^2 + y^2 / a^2 = 1 ) )) :=
by
  sorry

end ellipse_standard_equation_l117_117040


namespace equation_of_hyperbola_l117_117734

-- Definitions for conditions

def center_at_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def focus_point (focus : ℝ × ℝ) : Prop :=
  focus = (Real.sqrt 2, 0)

def distance_to_asymptote (focus : ℝ × ℝ) (distance : ℝ) : Prop :=
  -- Placeholder for the actual distance calculation
  distance = 1 -- The given distance condition in the problem

-- The mathematical proof problem statement

theorem equation_of_hyperbola :
  center_at_origin (0,0) ∧
  focus_point (Real.sqrt 2, 0) ∧
  distance_to_asymptote (Real.sqrt 2, 0) 1 → 
    ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
    (a^2 + b^2 = 2) ∧ (a^2 = 1) ∧ (b^2 = 1) ∧ 
    (∀ x y : ℝ, b^2*y^2 = x^2 - a^2*y^2 → (y = 0 ∧ x^2 = 1)) :=
sorry

end equation_of_hyperbola_l117_117734


namespace tan_alpha_value_l117_117504

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 := 
by
  sorry

end tan_alpha_value_l117_117504


namespace gcd_of_ten_digit_same_five_digit_l117_117670

def ten_digit_same_five_digit (n : ℕ) : Prop :=
  n > 9999 ∧ n < 100000 ∧ ∃ k : ℕ, k = n * (10^10 + 10^5 + 1)

theorem gcd_of_ten_digit_same_five_digit :
  (∀ n : ℕ, ten_digit_same_five_digit n → ∃ d : ℕ, d = 10000100001 ∧ ∀ m : ℕ, m ∣ d) := 
sorry

end gcd_of_ten_digit_same_five_digit_l117_117670


namespace multiply_binomials_l117_117100

theorem multiply_binomials :
  ∀ (x : ℝ), 
  (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 :=
by
  sorry

end multiply_binomials_l117_117100


namespace minnie_mounts_time_period_l117_117646

theorem minnie_mounts_time_period (M D : ℕ) 
  (mickey_daily_mounts_eq : 2 * M - 6 = 14)
  (minnie_mounts_per_day_eq : M = D + 3) : 
  D = 7 := 
by
  sorry

end minnie_mounts_time_period_l117_117646


namespace food_needed_for_vacation_l117_117952

-- Define the conditions
def daily_food_per_dog := 250 -- in grams
def number_of_dogs := 4
def number_of_days := 14

-- Define the proof problem
theorem food_needed_for_vacation :
  (daily_food_per_dog * number_of_dogs * number_of_days / 1000) = 14 :=
by
  sorry

end food_needed_for_vacation_l117_117952


namespace smallest_n_div_75_has_75_divisors_l117_117632

theorem smallest_n_div_75_has_75_divisors :
  ∃ n : ℕ, (n % 75 = 0) ∧ (n.factors.length = 75) ∧ (n / 75 = 432) :=
by
  sorry

end smallest_n_div_75_has_75_divisors_l117_117632


namespace expand_expression_l117_117383

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l117_117383


namespace adjacent_abby_bridget_probability_l117_117277
open Nat

-- Define the conditions
def total_kids := 6
def grid_rows := 3
def grid_cols := 2
def middle_row := 2
def abby_and_bridget := 2

-- Define the probability calculation
theorem adjacent_abby_bridget_probability :
  let total_arrangements := 6!
  let num_ways_adjacent :=
    (2 * abby_and_bridget) * (total_kids - abby_and_bridget)!
  let total_outcomes := total_arrangements
  (num_ways_adjacent / total_outcomes : ℚ) = 4 / 15
:= sorry

end adjacent_abby_bridget_probability_l117_117277


namespace range_of_a_l117_117002

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - |x + 1| + 2 * a ≥ 0) ↔ a ∈ (Set.Ici ((Real.sqrt 3 + 1) / 4)) := by
  sorry

end range_of_a_l117_117002


namespace integer_solution_l117_117486

theorem integer_solution (n : ℤ) (hneq : n ≠ -2) :
  ∃ (m : ℤ), (n^3 + 8) = m * (n^2 - 4) ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end integer_solution_l117_117486


namespace diamond_cut_1_3_loss_diamond_max_loss_ratio_l117_117677

noncomputable def value (w : ℝ) : ℝ := 6000 * w^2

theorem diamond_cut_1_3_loss (a : ℝ) :
  (value a - (value (1/4 * a) + value (3/4 * a))) / value a = 0.375 :=
by sorry

theorem diamond_max_loss_ratio :
  ∀ (m n : ℝ), (m > 0) → (n > 0) → 
  (1 - (value (m/(m + n)) + value (n/(m + n))) ≤ 0.5) :=
by sorry

end diamond_cut_1_3_loss_diamond_max_loss_ratio_l117_117677


namespace solve_system_of_equations_l117_117816

variable {x : Fin 15 → ℤ}

theorem solve_system_of_equations (h : ∀ i : Fin 15, 1 - x i * x ((i + 1) % 15) = 0) :
  (∀ i : Fin 15, x i = 1) ∨ (∀ i : Fin 15, x i = -1) :=
by
  -- Here we put the proof, but it's omitted for now.
  sorry

end solve_system_of_equations_l117_117816


namespace orchestra_members_l117_117699

theorem orchestra_members : ∃ (x : ℕ), (130 < x) ∧ (x < 260) ∧ (x % 6 = 1) ∧ (x % 5 = 2) ∧ (x % 7 = 3) ∧ (x = 241) :=
by
  sorry

end orchestra_members_l117_117699


namespace rr_sr_sum_le_one_l117_117099

noncomputable def rr_sr_le_one (r s : ℝ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : Prop :=
  r^r * s^s + r^s * s^r ≤ 1

theorem rr_sr_sum_le_one {r s : ℝ} (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_sum : r + s = 1) : rr_sr_le_one r s h_pos_r h_pos_s h_sum :=
  sorry

end rr_sr_sum_le_one_l117_117099


namespace tiffany_total_score_l117_117888

-- Definitions based on conditions
def points_per_treasure : ℕ := 6
def treasures_first_level : ℕ := 3
def treasures_second_level : ℕ := 5

-- The statement we want to prove
theorem tiffany_total_score : (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level) = 48 := by
  sorry

end tiffany_total_score_l117_117888


namespace max_area_rectangle_l117_117038

theorem max_area_rectangle (p : ℝ) (a b : ℝ) (h : p = 2 * (a + b)) : 
  ∃ S : ℝ, S = a * b ∧ (∀ (a' b' : ℝ), p = 2 * (a' + b') → S ≥ a' * b') → a = b :=
by
  sorry

end max_area_rectangle_l117_117038


namespace exists_element_x_l117_117769

open Set

theorem exists_element_x (n : ℕ) (S : Finset (Fin n)) (A : Fin n → Finset (Fin n)) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j) : 
  ∃ x ∈ S, ∀ i j : Fin n, i ≠ j → (A i \ {x}) ≠ (A j \ {x}) :=
sorry

end exists_element_x_l117_117769


namespace find_a_l117_117996

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = -f (-x) a) → a = 1 :=
by
  sorry

end find_a_l117_117996


namespace algebraic_expression_zero_l117_117570

theorem algebraic_expression_zero (a b : ℝ) (h : a^2 + 2 * a * b + b^2 = 0) : 
  a * (a + 4 * b) - (a + 2 * b) * (a - 2 * b) = 0 :=
by
  sorry

end algebraic_expression_zero_l117_117570


namespace lewis_weekly_earning_l117_117361

def total_amount_earned : ℕ := 178
def number_of_weeks : ℕ := 89
def weekly_earning (total : ℕ) (weeks : ℕ) : ℕ := total / weeks

theorem lewis_weekly_earning : weekly_earning total_amount_earned number_of_weeks = 2 :=
by
  -- The proof will go here
  sorry

end lewis_weekly_earning_l117_117361


namespace pictures_total_l117_117778

theorem pictures_total (peter_pics : ℕ) (quincy_extra_pics : ℕ) (randy_pics : ℕ) (quincy_pics : ℕ) (total_pics : ℕ) 
  (h1 : peter_pics = 8)
  (h2 : quincy_extra_pics = 20)
  (h3 : randy_pics = 5)
  (h4 : quincy_pics = peter_pics + quincy_extra_pics)
  (h5 : total_pics = randy_pics + peter_pics + quincy_pics) :
  total_pics = 41 :=
by sorry

end pictures_total_l117_117778


namespace large_pizzas_sold_l117_117515

variables (num_small_pizzas num_large_pizzas : ℕ) (price_small price_large total_revenue revenue_from_smalls revenue_from_larges : ℕ)

theorem large_pizzas_sold
  (price_small := 2)
  (price_large := 8)
  (total_revenue := 40)
  (num_small_pizzas := 8)
  (revenue_from_smalls := num_small_pizzas * price_small)
  (revenue_from_larges := total_revenue - revenue_from_smalls)
  (large_pizza_count := revenue_from_larges / price_large) :
  large_pizza_count = 3 :=
sorry

end large_pizzas_sold_l117_117515


namespace range_of_x_coordinate_l117_117141

theorem range_of_x_coordinate (x : ℝ) : 
  (0 ≤ 2*x + 2 ∧ 2*x + 2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ -1/2) := 
sorry

end range_of_x_coordinate_l117_117141


namespace polynomial_root_sum_eq_48_l117_117592

theorem polynomial_root_sum_eq_48 {r s t : ℕ} (h1 : r * s * t = 2310) 
  (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) : r + s + t = 48 :=
sorry

end polynomial_root_sum_eq_48_l117_117592


namespace units_digit_of_square_l117_117452

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 :=
by
  sorry

end units_digit_of_square_l117_117452


namespace intersection_of_M_and_P_l117_117391

def M : Set ℝ := { x | x^2 = x }
def P : Set ℝ := { x | |x - 1| = 1 }

theorem intersection_of_M_and_P : M ∩ P = {0} := by
  sorry

end intersection_of_M_and_P_l117_117391


namespace radius_of_circle_l117_117680

theorem radius_of_circle
  (d PQ QR : ℝ) (h1 : d = 15) (h2 : PQ = 7) (h3 : QR = 8) :
  ∃ r : ℝ, r = 2 * Real.sqrt 30 ∧ (PQ * (PQ + QR) = (d - r) * (d + r)) :=
by
  -- All necessary non-proof related statements
  sorry

end radius_of_circle_l117_117680


namespace inequality_solution_set_inequality_proof_2_l117_117032

theorem inequality_solution_set : 
  { x : ℝ | |x + 1| + |x + 3| < 4 } = { x : ℝ | -4 < x ∧ x < 0 } :=
sorry

theorem inequality_proof_2 (a b : ℝ) (ha : -4 < a) (ha' : a < 0) (hb : -4 < b) (hb' : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| :=
sorry

end inequality_solution_set_inequality_proof_2_l117_117032


namespace range_of_m_l117_117499

def P (m : ℝ) : Prop :=
  9 - m > 2 * m ∧ 2 * m > 0

def Q (m : ℝ) : Prop :=
  m > 0 ∧ (Real.sqrt (6) / 2 < Real.sqrt (5 + m) / Real.sqrt (5)) ∧ (Real.sqrt (5 + m) / Real.sqrt (5) < Real.sqrt (2))

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) → (0 < m ∧ m ≤ 5 / 2) ∨ (3 ≤ m ∧ m < 5) :=
sorry

end range_of_m_l117_117499


namespace sixth_power_sum_l117_117140

theorem sixth_power_sum (a b c d e f : ℤ) :
  a^6 + b^6 + c^6 + d^6 + e^6 + f^6 = 6 * a * b * c * d * e * f + 1 → 
  (a = 1 ∨ a = -1 ∨ b = 1 ∨ b = -1 ∨ c = 1 ∨ c = -1 ∨ 
   d = 1 ∨ d = -1 ∨ e = 1 ∨ e = -1 ∨ f = 1 ∨ f = -1) ∧
  ((a = 1 ∨ a = -1 ∨ a = 0) ∧ 
   (b = 1 ∨ b = -1 ∨ b = 0) ∧ 
   (c = 1 ∨ c = -1 ∨ c = 0) ∧ 
   (d = 1 ∨ d = -1 ∨ d = 0) ∧ 
   (e = 1 ∨ e = -1 ∨ e = 0) ∧ 
   (f = 1 ∨ f = -1 ∨ f = 0)) ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0 ∨ f ≠ 0) ∧
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0 ∨ f = 0) := 
sorry

end sixth_power_sum_l117_117140


namespace determine_constants_l117_117542

theorem determine_constants
  (C D : ℝ)
  (h1 : 3 * C + D = 7)
  (h2 : 4 * C - 2 * D = -15) :
  C = -0.1 ∧ D = 7.3 :=
by
  sorry

end determine_constants_l117_117542


namespace cos_double_angle_identity_l117_117763

theorem cos_double_angle_identity (x : ℝ) (h : Real.sin (Real.pi / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 :=
sorry

end cos_double_angle_identity_l117_117763


namespace partial_fraction_product_l117_117303

theorem partial_fraction_product (A B C : ℚ)
  (h_eq : ∀ x, (x^2 - 13) / ((x-2) * (x+2) * (x-3)) = A / (x-2) + B / (x+2) + C / (x-3))
  (h_A : A = 9 / 4)
  (h_B : B = -9 / 20)
  (h_C : C = -4 / 5) :
  A * B * C = 81 / 100 := 
by
  sorry

end partial_fraction_product_l117_117303


namespace integral_cos_plus_one_l117_117579

theorem integral_cos_plus_one :
  ∫ x in - (Real.pi / 2).. (Real.pi / 2), (1 + Real.cos x) = Real.pi + 2 :=
by
  sorry

end integral_cos_plus_one_l117_117579


namespace harry_drank_last_mile_l117_117423

theorem harry_drank_last_mile :
  ∀ (T D start_water end_water leak_rate drink_rate leak_time first_miles : ℕ),
    start_water = 10 →
    end_water = 2 →
    leak_rate = 1 →
    leak_time = 2 →
    drink_rate = 1 →
    first_miles = 3 →
    T = leak_rate * leak_time →
    D = drink_rate * first_miles →
    start_water - end_water = T + D + (start_water - end_water - T - D) →
    start_water - end_water - T - D = 3 :=
by
  sorry

end harry_drank_last_mile_l117_117423


namespace circle_diameter_from_area_l117_117757

theorem circle_diameter_from_area (A r d : ℝ) (hA : A = 64 * Real.pi) (h_area : A = Real.pi * r^2) : d = 16 :=
by
  sorry

end circle_diameter_from_area_l117_117757


namespace buying_beams_l117_117553

/-- Problem Statement:
Given:
1. The total money for beams is 6210 wen.
2. The transportation cost per beam is 3 wen.
3. Removing one beam means the remaining beams' total transportation cost equals the price of one beam.

Prove: 3 * (x - 1) = 6210 / x
-/
theorem buying_beams (x : ℕ) (h₁ : x > 0) (h₂ : 6210 % x = 0) :
  3 * (x - 1) = 6210 / x :=
sorry

end buying_beams_l117_117553


namespace sufficient_but_not_necessary_l117_117624

variable (p q : Prop)

theorem sufficient_but_not_necessary (h : p ∧ q) : ¬¬p :=
  by sorry -- Proof not required

end sufficient_but_not_necessary_l117_117624


namespace triangle_inequality_l117_117054

theorem triangle_inequality (a b c : ℝ) (habc_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > (a^4 + b^4 + c^4) :=
by
  sorry

end triangle_inequality_l117_117054


namespace percent_daffodils_is_57_l117_117634

-- Condition 1: Four-sevenths of the flowers are yellow
def fraction_yellow : ℚ := 4 / 7

-- Condition 2: Two-thirds of the red flowers are daffodils
def fraction_red_daffodils_given_red : ℚ := 2 / 3

-- Condition 3: Half of the yellow flowers are tulips
def fraction_yellow_tulips_given_yellow : ℚ := 1 / 2

-- Calculate fractions of yellow and red flowers
def fraction_red : ℚ := 1 - fraction_yellow

-- Calculate fractions of daffodils
def fraction_yellow_daffodils : ℚ := fraction_yellow * (1 - fraction_yellow_tulips_given_yellow)
def fraction_red_daffodils : ℚ := fraction_red * fraction_red_daffodils_given_red

-- Total fraction of daffodils
def fraction_daffodils : ℚ := fraction_yellow_daffodils + fraction_red_daffodils

-- Proof statement
theorem percent_daffodils_is_57 :
  fraction_daffodils * 100 = 57 := by
  sorry

end percent_daffodils_is_57_l117_117634


namespace principal_amount_l117_117762

theorem principal_amount (P R T SI : ℝ) (hR : R = 4) (hT : T = 5) (hSI : SI = P - 2240) 
    (h_formula : SI = (P * R * T) / 100) : P = 2800 :=
by 
  sorry

end principal_amount_l117_117762


namespace water_formed_on_combining_l117_117591

theorem water_formed_on_combining (molar_mass_water : ℝ) (n_NaOH : ℝ) (n_HCl : ℝ) :
  n_NaOH = 1 ∧ n_HCl = 1 ∧ molar_mass_water = 18.01528 → 
  n_NaOH * molar_mass_water = 18.01528 :=
by sorry

end water_formed_on_combining_l117_117591


namespace total_books_l117_117127

-- Define the number of books Stu has
def Stu_books : ℕ := 9

-- Define the multiplier for Albert's books
def Albert_multiplier : ℕ := 4

-- Define the number of books Albert has
def Albert_books : ℕ := Albert_multiplier * Stu_books

-- Prove that the total number of books is 45
theorem total_books:
  Stu_books + Albert_books = 45 :=
by 
  -- This is where the proof steps would go, but we skip it for now 
  sorry

end total_books_l117_117127


namespace base_length_of_isosceles_triangle_l117_117386

-- Definitions for the problem
def isosceles_triangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] :=
  ∃ (AB BC : ℝ), AB = BC

-- The problem to prove
theorem base_length_of_isosceles_triangle
  {A B C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
  (AB BC : ℝ) (AC x : ℝ)
  (height_base : ℝ) (height_side : ℝ) 
  (h1 : AB = BC)
  (h2 : height_base = 10)
  (h3 : height_side = 12)
  (h4 : AC = x)
  (h5 : ∀ AE BD : ℝ, AE = height_side → BD = height_base) :
  x = 15 := by sorry

end base_length_of_isosceles_triangle_l117_117386


namespace equality_conditions_l117_117752

theorem equality_conditions (a b c d : ℝ) :
  a + bcd = (a + b) * (a + c) * (a + d) ↔ a = 0 ∨ a^2 + a * (b + c + d) + bc + bd + cd = 1 :=
by
  sorry

end equality_conditions_l117_117752


namespace geom_seq_a_n_l117_117967

theorem geom_seq_a_n (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -1) 
  (h_a7 : a 7 = -9) :
  a 5 = -3 :=
sorry

end geom_seq_a_n_l117_117967


namespace no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l117_117704

-- Part (1): Prove that there do not exist positive integers m and n such that m(m+2) = n(n+1)
theorem no_solutions_m_m_plus_2_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) :=
sorry

-- Part (2): Given k ≥ 3,
-- Case (a): Prove that for k=3, there do not exist positive integers m and n such that m(m+3) = n(n+1)
theorem no_solutions_m_m_plus_3_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 3) = n * (n + 1) :=
sorry

-- Case (b): Prove that for k ≥ 4, there exist positive integers m and n such that m(m+k) = n(n+1)
theorem solutions_exist_m_m_plus_k_eq_n_n_plus_1 (k : ℕ) (h : k ≥ 4) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1) :=
sorry

end no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l117_117704


namespace unique_solution_l117_117170

noncomputable def unique_solution_exists : Prop :=
  ∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    (a + b = (c + d + e) / 7) ∧
    (a + d = (b + c + e) / 5) ∧
    (a + b + c + d + e = 24) ∧
    (a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 3 ∧ e = 9)

theorem unique_solution : unique_solution_exists :=
sorry

end unique_solution_l117_117170


namespace problem_statement_l117_117743

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement_l117_117743


namespace percentage_neither_language_l117_117900

noncomputable def total_diplomats : ℝ := 120
noncomputable def latin_speakers : ℝ := 20
noncomputable def russian_non_speakers : ℝ := 32
noncomputable def both_languages : ℝ := 0.10 * total_diplomats

theorem percentage_neither_language :
  let D := total_diplomats
  let L := latin_speakers
  let R := D - russian_non_speakers
  let LR := both_languages
  ∃ P, P = 100 * (D - (L + R - LR)) / D :=
by
  existsi ((total_diplomats - (latin_speakers + (total_diplomats - russian_non_speakers) - both_languages)) / total_diplomats * 100)
  sorry

end percentage_neither_language_l117_117900


namespace square_area_problem_l117_117989

theorem square_area_problem 
  (BM : ℝ) 
  (ABCD_is_divided : Prop)
  (hBM : BM = 4)
  (hABCD_is_divided : ABCD_is_divided) : 
  ∃ (side_length : ℝ), side_length * side_length = 144 := 
by
-- We skip the proof part for this task
sorry

end square_area_problem_l117_117989


namespace describe_graph_l117_117416

noncomputable def points_satisfying_equation (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

theorem describe_graph : {p : ℝ × ℝ | points_satisfying_equation p.1 p.2} = {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} :=
by
  sorry

end describe_graph_l117_117416


namespace vector_solution_l117_117664

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_solution (a x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a :=
by sorry

end vector_solution_l117_117664


namespace prime_base_representation_of_360_l117_117919

theorem prime_base_representation_of_360 :
  ∃ (exponents : List ℕ), exponents = [3, 2, 1, 0]
  ∧ (2^exponents.head! * 3^(exponents.tail!.head!) * 5^(exponents.tail!.tail!.head!) * 7^(exponents.tail!.tail!.tail!.head!)) = 360 := by
sorry

end prime_base_representation_of_360_l117_117919


namespace find_x_l117_117613

theorem find_x {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1/y = 10) (h2 : y + 1/x = 5/12) : x = 4 ∨ x = 6 :=
by
  sorry

end find_x_l117_117613


namespace sum_reciprocals_transformed_roots_l117_117715

theorem sum_reciprocals_transformed_roots (a b c : ℝ) (h : ∀ x, (x^3 - 2 * x - 5 = 0) → (x = a) ∨ (x = b) ∨ (x = c)) : 
  (1 / (a - 2)) + (1 / (b - 2)) + (1 / (c - 2)) = 10 := 
by sorry

end sum_reciprocals_transformed_roots_l117_117715


namespace parallelogram_base_length_l117_117738

theorem parallelogram_base_length (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_eq_2b : h = 2 * b) (area_eq_98 : area = 98) 
  (area_def : area = b * h) : b = 7 :=
by
  sorry

end parallelogram_base_length_l117_117738


namespace solve_B_share_l117_117093

def ratio_shares (A B C : ℚ) : Prop :=
  A = 1/2 ∧ B = 1/3 ∧ C = 1/4

def initial_capitals (total_capital : ℚ) (A_s B_s C_s : ℚ) : Prop :=
  A_s = 1/2 * total_capital ∧ B_s = 1/3 * total_capital ∧ C_s = 1/4 * total_capital

def total_capital_contribution (A_contrib B_contrib C_contrib : ℚ) : Prop :=
  A_contrib = 42 ∧ B_contrib = 48 ∧ C_contrib = 36

def B_share (B_contrib total_contrib profit : ℚ) : ℚ := 
  (B_contrib / total_contrib) * profit

theorem solve_B_share : 
  ∀ (A_s B_s C_s total_capital profit A_contrib B_contrib C_contrib total_contrib : ℚ),
  ratio_shares (1/2) (1/3) (1/4) →
  initial_capitals total_capital A_s B_s C_s →
  total_capital_contribution A_contrib B_contrib C_contrib →
  total_contrib = A_contrib + B_contrib + C_contrib →
  profit = 378 →
  B_s = (1/3) * total_capital →
  B_contrib = 48 →
  B_share B_contrib total_contrib profit = 108 := by 
    sorry

end solve_B_share_l117_117093


namespace roots_sum_of_squares_l117_117962

noncomputable def proof_problem (p q r : ℝ) : Prop :=
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 598

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h1 : p + q + r = 18)
  (h2 : p * q + q * r + r * p = 25)
  (h3 : p * q * r = 6) :
  proof_problem p q r :=
by {
  -- Solution steps here (omitted; not needed for the task)
  sorry
}

end roots_sum_of_squares_l117_117962


namespace product_of_last_two_digits_l117_117589

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 11) (h2 : ∃ (n : ℕ), 10 * A + B = 6 * n) : A * B = 24 :=
sorry

end product_of_last_two_digits_l117_117589


namespace slope_parallel_to_original_line_l117_117208

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l117_117208


namespace factorize_x_squared_minus_1_l117_117021

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by
  -- Proof goes here
  sorry

end factorize_x_squared_minus_1_l117_117021


namespace range_of_a_l117_117139

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end range_of_a_l117_117139


namespace function_characterization_l117_117198

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end function_characterization_l117_117198


namespace intersection_A_B_l117_117832

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end intersection_A_B_l117_117832


namespace smallest_palindromic_primes_l117_117991

def is_palindromic (n : ℕ) : Prop :=
  ∀ a b : ℕ, n = 1001 * a + 1010 * b → 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem smallest_palindromic_primes :
  ∃ n1 n2 : ℕ, 
  is_palindromic n1 ∧ is_palindromic n2 ∧ is_prime n1 ∧ is_prime n2 ∧ n1 < n2 ∧
  ∀ m : ℕ, (is_palindromic m ∧ is_prime m ∧ m < n2 → m = n1) ∧
           (is_palindromic m ∧ is_prime m ∧ m < n1 → m ≠ n2) ∧ n1 = 1221 ∧ n2 = 1441 := 
sorry

end smallest_palindromic_primes_l117_117991


namespace evaluate_f_at_3_l117_117742

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end evaluate_f_at_3_l117_117742


namespace certain_number_is_166_l117_117890

theorem certain_number_is_166 :
  ∃ x : ℕ, x - 78 =  (4 - 30) + 114 ∧ x = 166 := by
  sorry

end certain_number_is_166_l117_117890


namespace ratio_of_areas_l117_117774

noncomputable def large_square_side : ℝ := 4
noncomputable def large_square_area : ℝ := large_square_side ^ 2
noncomputable def inscribed_square_side : ℝ := 1  -- As it fits in the definition from the problem description
noncomputable def inscribed_square_area : ℝ := inscribed_square_side ^ 2

theorem ratio_of_areas :
  (inscribed_square_area / large_square_area) = 1 / 16 :=
by
  sorry

end ratio_of_areas_l117_117774


namespace find_a_l117_117547

theorem find_a (b c : ℤ) 
  (vertex_condition : ∀ (x : ℝ), x = -1 → (ax^2 + b*x + c) = -2)
  (point_condition : ∀ (x : ℝ), x = 0 → (a*x^2 + b*x + c) = -1) :
  ∃ (a : ℤ), a = 1 :=
by
  sorry

end find_a_l117_117547


namespace solution_exists_for_100_100_l117_117796

def exists_positive_integers_sum_of_cubes (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a^3 + b^3 + c^3 + d^3 = x

theorem solution_exists_for_100_100 : exists_positive_integers_sum_of_cubes (100 ^ 100) :=
by
  sorry

end solution_exists_for_100_100_l117_117796


namespace isabella_more_than_giselle_l117_117063

variables (I S G : ℕ)

def isabella_has_more_than_sam : Prop := I = S + 45
def giselle_amount : Prop := G = 120
def total_amount : Prop := I + S + G = 345

theorem isabella_more_than_giselle
  (h1 : isabella_has_more_than_sam I S)
  (h2 : giselle_amount G)
  (h3 : total_amount I S G) :
  I - G = 15 :=
by
  sorry

end isabella_more_than_giselle_l117_117063


namespace cos_210_eq_neg_sqrt_3_div_2_l117_117596

theorem cos_210_eq_neg_sqrt_3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by
  -- leave proof as sorry
  sorry

end cos_210_eq_neg_sqrt_3_div_2_l117_117596


namespace smallest_prime_divisor_of_sum_first_100_is_5_l117_117622

-- Conditions: The sum of the first 100 natural numbers
def sum_first_n_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Prime checking function to identify the smallest prime divisor
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  if n % 5 = 0 then 5 else
  n -- Such a simplification works because we know the answer must be within the first few primes.

-- Proof statement
theorem smallest_prime_divisor_of_sum_first_100_is_5 : smallest_prime_divisor (sum_first_n_numbers 100) = 5 :=
by
  -- Proof steps would follow here.
  sorry

end smallest_prime_divisor_of_sum_first_100_is_5_l117_117622


namespace new_solid_edges_l117_117619

-- Definitions based on conditions
def original_vertices : ℕ := 8
def original_edges : ℕ := 12
def new_edges_per_vertex : ℕ := 3
def number_of_vertices : ℕ := original_vertices

-- Conclusion to prove
theorem new_solid_edges : 
  (original_edges + new_edges_per_vertex * number_of_vertices) = 36 := 
by
  sorry

end new_solid_edges_l117_117619


namespace badge_counts_l117_117087

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l117_117087


namespace last_number_of_ratio_l117_117309

theorem last_number_of_ratio (A B C : ℕ) (h1 : 5 * B = A) (h2 : 4 * B = C) (h3 : A + B + C = 1000) : C = 400 :=
by
  sorry

end last_number_of_ratio_l117_117309


namespace select_from_companyA_l117_117098

noncomputable def companyA_representatives : ℕ := 40
noncomputable def companyB_representatives : ℕ := 60
noncomputable def total_representatives : ℕ := companyA_representatives + companyB_representatives
noncomputable def sample_size : ℕ := 10
noncomputable def sampling_ratio : ℚ := sample_size / total_representatives
noncomputable def selected_from_companyA : ℚ := companyA_representatives * sampling_ratio

theorem select_from_companyA : selected_from_companyA = 4 := by
  sorry


end select_from_companyA_l117_117098


namespace satellite_modular_units_l117_117537

variables (N S T U : ℕ)
variable (h1 : N = S / 3)
variable (h2 : S / T = 1 / 9)
variable (h3 : U * N = 8 * T / 9)

theorem satellite_modular_units :
  U = 24 :=
by sorry

end satellite_modular_units_l117_117537


namespace jenny_hours_left_l117_117121

theorem jenny_hours_left
  (hours_research : ℕ)
  (hours_proposal : ℕ)
  (hours_total : ℕ)
  (h1 : hours_research = 10)
  (h2 : hours_proposal = 2)
  (h3 : hours_total = 20) :
  (hours_total - (hours_research + hours_proposal) = 8) :=
by
  sorry

end jenny_hours_left_l117_117121


namespace dice_sum_eight_dice_l117_117050

/--
  Given 8 fair 6-sided dice, prove that the number of ways to obtain
  a sum of 11 on the top faces of these dice, is 120.
-/
theorem dice_sum_eight_dice :
  (∃ n : ℕ, ∀ (dices : List ℕ), (dices.length = 8 ∧ (∀ d ∈ dices, 1 ≤ d ∧ d ≤ 6) 
   ∧ dices.sum = 11) → n = 120) :=
sorry

end dice_sum_eight_dice_l117_117050


namespace quadratic_transform_l117_117942

theorem quadratic_transform : ∀ (x : ℝ), x^2 = 3 * x + 1 ↔ x^2 - 3 * x - 1 = 0 :=
by
  sorry

end quadratic_transform_l117_117942


namespace min_alpha_beta_l117_117398

theorem min_alpha_beta (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1)
  (alpha : ℝ := a + 1 / a) (beta : ℝ := b + 1 / b) :
  alpha + beta ≥ 10 := by
  sorry

end min_alpha_beta_l117_117398


namespace tradesman_gain_l117_117918

-- Let's define a structure representing the tradesman's buying and selling operation.
structure Trade where
  true_value : ℝ
  defraud_rate : ℝ
  buy_price : ℕ
  sell_price : ℕ

theorem tradesman_gain (T : Trade) (H1 : T.defraud_rate = 0.2) (H2 : T.true_value = 100)
  (H3 : T.buy_price = T.true_value * (1 - T.defraud_rate))
  (H4 : T.sell_price = T.true_value * (1 + T.defraud_rate)) :
  ((T.sell_price - T.buy_price) / T.buy_price) * 100 = 50 := 
by
  sorry

end tradesman_gain_l117_117918


namespace pennies_thrown_total_l117_117066

theorem pennies_thrown_total (rachelle_pennies gretchen_pennies rocky_pennies : ℕ) 
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) : 
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 := 
by 
  sorry

end pennies_thrown_total_l117_117066


namespace prob_two_girls_l117_117786

variable (Pboy Pgirl : ℝ)

-- Conditions
def prob_boy : Prop := Pboy = 1 / 2
def prob_girl : Prop := Pgirl = 1 / 2

-- The theorem to be proven
theorem prob_two_girls (h₁ : prob_boy Pboy) (h₂ : prob_girl Pgirl) : (Pgirl * Pgirl) = 1 / 4 :=
by
  sorry

end prob_two_girls_l117_117786


namespace largest_divisor_of_expression_l117_117449

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 + n^3 - n - 1) :=
sorry

end largest_divisor_of_expression_l117_117449


namespace number_subtracted_from_10000_l117_117254

theorem number_subtracted_from_10000 (x : ℕ) (h : 10000 - x = 9001) : x = 999 := by
  sorry

end number_subtracted_from_10000_l117_117254


namespace radio_show_play_song_duration_l117_117230

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end radio_show_play_song_duration_l117_117230


namespace divisors_large_than_8_fact_count_l117_117027

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem divisors_large_than_8_fact_count :
  let n := 9
  let factorial_n := factorial n
  let factorial_n_minus_1 := factorial (n - 1)
  ∃ (num_divisors : ℕ), num_divisors = 8 ∧
    (∀ d, d ∣ factorial_n → d > factorial_n_minus_1 ↔ ∃ k, k ∣ factorial_n ∧ k < 9) :=
by
  sorry

end divisors_large_than_8_fact_count_l117_117027


namespace triangle_shortest_side_condition_l117_117151

theorem triangle_shortest_side_condition
  (A B C : Type) 
  (r : ℝ) (AF FB : ℝ)
  (P : ℝ)
  (h_AF : AF = 7)
  (h_FB : FB = 9)
  (h_r : r = 5)
  (h_P : P = 46) 
  : (min (min (7 + 9) (2 * 14)) ((7 + 9) - 14)) = 2 := 
by sorry

end triangle_shortest_side_condition_l117_117151


namespace angle_D_in_pentagon_l117_117798

theorem angle_D_in_pentagon (A B C D E : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : D = E) (h4 : A + 40 = D) 
  (h5 : A + B + C + D + E = 540) : D = 132 :=
by
  -- Add proof here if needed
  sorry

end angle_D_in_pentagon_l117_117798


namespace angle_of_inclination_l117_117671

-- The statement of the mathematically equivalent proof problem in Lean 4
theorem angle_of_inclination
  (k: ℝ)
  (α: ℝ)
  (line_eq: ∀ x, ∃ y, y = (k-1) * x + 2)
  (circle_eq: ∀ x y, x^2 + y^2 + k * x + 2 * y + k^2 = 0) :
  α = 3 * Real.pi / 4 :=
sorry -- Proof to be provided

end angle_of_inclination_l117_117671


namespace possible_values_for_p_t_l117_117512

theorem possible_values_for_p_t (p q r s t : ℝ)
(h₁ : |p - q| = 3)
(h₂ : |q - r| = 4)
(h₃ : |r - s| = 5)
(h₄ : |s - t| = 6) :
  ∃ (v : Finset ℝ), v = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ v :=
sorry

end possible_values_for_p_t_l117_117512


namespace area_triangle_MDA_l117_117466

noncomputable def area_of_triangle_MDA (r : ℝ) : ℝ := 
  let AM := r / 3
  let OM := (r ^ 2 - (AM ^ 2)).sqrt
  let AD := AM / 2
  let DM := AD / (1 / 2)
  1 / 2 * AD * DM

theorem area_triangle_MDA (r : ℝ) : area_of_triangle_MDA r = r ^ 2 / 36 := by
  sorry

end area_triangle_MDA_l117_117466


namespace smallest_possible_value_l117_117434

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l117_117434


namespace linear_term_coefficient_l117_117317

-- Define the given equation
def equation (x : ℝ) : ℝ := x^2 - 2022*x - 2023

-- The goal is to prove that the coefficient of the linear term in equation is -2022
theorem linear_term_coefficient : ∀ x : ℝ, equation x = x^2 - 2022*x - 2023 → -2022 = -2022 :=
by
  intros x h
  sorry

end linear_term_coefficient_l117_117317


namespace geometric_sum_is_correct_l117_117575

theorem geometric_sum_is_correct : 
  let a := 1
  let r := 5
  let n := 6
  a * (r^n - 1) / (r - 1) = 3906 := by
  sorry

end geometric_sum_is_correct_l117_117575


namespace quadratic_function_range_l117_117270

noncomputable def quadratic_range : Set ℝ := {y | -2 ≤ y ∧ y < 2}

theorem quadratic_function_range :
  ∀ y : ℝ, 
    (∃ x : ℝ, -2 < x ∧ x < 1 ∧ y = x^2 + 2 * x - 1) ↔ (y ∈ quadratic_range) :=
by
  sorry

end quadratic_function_range_l117_117270


namespace parallelogram_area_l117_117474

variable (base height : ℝ) (tripled_area_factor original_area new_area : ℝ)

theorem parallelogram_area (h_base : base = 6) (h_height : height = 20)
    (h_tripled_area_factor : tripled_area_factor = 9)
    (h_original_area_calc : original_area = base * height)
    (h_new_area_calc : new_area = original_area * tripled_area_factor) :
    original_area = 120 ∧ tripled_area_factor = 9 ∧ new_area = 1080 := by
  sorry

end parallelogram_area_l117_117474


namespace solve_equation_simplify_expression_l117_117428

-- Part 1: Solving the equation
theorem solve_equation (x : ℝ) : 9 * (x - 3) ^ 2 - 121 = 0 ↔ x = 20 / 3 ∨ x = -2 / 3 :=
by 
    sorry

-- Part 2: Simplifying the expression
theorem simplify_expression (x y : ℝ) : (x - 2 * y) * (x ^ 2 + 2 * x * y + 4 * y ^ 2) = x ^ 3 - 8 * y ^ 3 :=
by 
    sorry

end solve_equation_simplify_expression_l117_117428


namespace evaluate_expression_l117_117131

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end evaluate_expression_l117_117131


namespace increasing_geometric_progression_l117_117683

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem increasing_geometric_progression (a : ℝ) (ha : 0 < a)
  (h1 : ∃ b c q : ℝ, b = Int.floor a ∧ c = a - b ∧ a = b + c ∧ c = b * q ∧ a = c * q ∧ 1 < q) : 
  a = golden_ratio :=
sorry

end increasing_geometric_progression_l117_117683


namespace tent_cost_solution_l117_117995

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end tent_cost_solution_l117_117995


namespace Kendra_weekly_words_not_determined_without_weeks_l117_117030

def Kendra_goal : Nat := 60
def Kendra_already_learned : Nat := 36
def Kendra_needs_to_learn : Nat := 24

theorem Kendra_weekly_words_not_determined_without_weeks (weeks : Option Nat) : weeks = none → Kendra_needs_to_learn / weeks.getD 1 = 24 -> False := by
  sorry

end Kendra_weekly_words_not_determined_without_weeks_l117_117030


namespace smallest_w_l117_117494

theorem smallest_w (x y w : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 ^ x) ∣ (3125 * w)) (h4 : (3 ^ y) ∣ (3125 * w)) 
  (h5 : (5 ^ (x + y)) ∣ (3125 * w)) (h6 : (7 ^ (x - y)) ∣ (3125 * w))
  (h7 : (13 ^ 4) ∣ (3125 * w))
  (h8 : x + y ≤ 10) (h9 : x - y ≥ 2) :
  w = 33592336 :=
by
  sorry

end smallest_w_l117_117494


namespace area_ratio_of_similar_polygons_l117_117572

theorem area_ratio_of_similar_polygons (similarity_ratio: ℚ) (hratio: similarity_ratio = 1/5) : (similarity_ratio ^ 2 = 1/25) := 
by 
  sorry

end area_ratio_of_similar_polygons_l117_117572


namespace base8_subtraction_correct_l117_117598

noncomputable def base8_subtraction (x y : Nat) : Nat :=
  if y > x then 0 else x - y

theorem base8_subtraction_correct :
  base8_subtraction 546 321 - 105 = 120 :=
by
  -- Given the condition that all arithmetic is in base 8
  sorry

end base8_subtraction_correct_l117_117598


namespace units_digit_of_product_of_first_three_positive_composite_numbers_l117_117914

theorem units_digit_of_product_of_first_three_positive_composite_numbers :
  (4 * 6 * 8) % 10 = 2 :=
by sorry

end units_digit_of_product_of_first_three_positive_composite_numbers_l117_117914


namespace find_z_l117_117378

variable (x y z : ℝ)

-- Define x, y as given in the problem statement
def x_def : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) := by
  sorry

def y_def : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) := by
  sorry

-- Define the equation relating z to x and y
def z_eq : 192 * z = x^4 + y^4 + (x + y)^4 := by 
  sorry

-- Theorem stating the value of z
theorem find_z (h1 : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3))
               (h2 : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3))
               (h3 : 192 * z = x^4 + y^4 + (x + y)^4) :
  z = 6 := by 
  sorry

end find_z_l117_117378


namespace binary_multiplication_l117_117897

theorem binary_multiplication : (10101 : ℕ) * (101 : ℕ) = 1101001 :=
by sorry

end binary_multiplication_l117_117897


namespace polynomial_factorization_l117_117358

-- Definitions from conditions
def p (x : ℝ) : ℝ := x^6 - 2 * x^4 + 6 * x^3 + x^2 - 6 * x + 9
def q (x : ℝ) : ℝ := (x^3 - x + 3)^2

-- The theorem statement proving question == answer given conditions
theorem polynomial_factorization : ∀ x : ℝ, p x = q x :=
by
  sorry

end polynomial_factorization_l117_117358


namespace arithmetic_sequence_Sn_l117_117322

noncomputable def S (n : ℕ) : ℕ := sorry -- S is the sequence function

theorem arithmetic_sequence_Sn {n : ℕ} (h1 : S n = 2) (h2 : S (3 * n) = 18) : S (4 * n) = 26 :=
  sorry

end arithmetic_sequence_Sn_l117_117322


namespace product_of_solutions_l117_117350

theorem product_of_solutions (x : ℝ) :
  let a := -2
  let b := -8
  let c := -49
  ∀ x₁ x₂, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) → 
  x₁ * x₂ = 49/2 :=
sorry

end product_of_solutions_l117_117350


namespace christian_sue_need_more_money_l117_117980

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end christian_sue_need_more_money_l117_117980


namespace equal_number_of_boys_and_girls_l117_117958

theorem equal_number_of_boys_and_girls
  (m d M D : ℝ)
  (hm : m ≠ 0)
  (hd : d ≠ 0)
  (avg1 : M / m ≠ D / d)
  (avg2 : (M / m + D / d) / 2 = (M + D) / (m + d)) :
  m = d :=
by
  sorry

end equal_number_of_boys_and_girls_l117_117958


namespace octagon_perimeter_l117_117183

/-- 
  Represents the side length of the regular octagon
-/
def side_length : ℕ := 12

/-- 
  Represents the number of sides of a regular octagon
-/
def number_of_sides : ℕ := 8

/-- 
  Defines the perimeter of the regular octagon
-/
def perimeter (side_length : ℕ) (number_of_sides : ℕ) : ℕ :=
  side_length * number_of_sides

/-- 
  Proof statement: asserting that the perimeter of a regular octagon
  with a side length of 12 meters is 96 meters
-/
theorem octagon_perimeter :
  perimeter side_length number_of_sides = 96 :=
  sorry

end octagon_perimeter_l117_117183


namespace compound_interest_correct_amount_l117_117573

-- Define constants and conditions
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

def compound_interest (P R T : ℕ) : ℕ := P * ((1 + R / 100) ^ T - 1)

-- Given values and conditions
def P₁ : ℕ := 1750
def R₁ : ℕ := 8
def T₁ : ℕ := 3
def R₂ : ℕ := 10
def T₂ : ℕ := 2

def SI : ℕ := simple_interest P₁ R₁ T₁
def CI : ℕ := 2 * SI

def P₂ : ℕ := 4000

-- The statement to be proven
theorem compound_interest_correct_amount : 
  compound_interest P₂ R₂ T₂ = CI := 
by 
  sorry

end compound_interest_correct_amount_l117_117573


namespace larger_exceeds_smaller_times_l117_117608

theorem larger_exceeds_smaller_times {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_diff : a ≠ b)
  (h_eq : a^3 - b^3 = 3 * (2 * a^2 * b - 3 * a * b^2 + b^3)) : a = 4 * b :=
sorry

end larger_exceeds_smaller_times_l117_117608


namespace find_heaviest_or_lightest_l117_117907

theorem find_heaviest_or_lightest (stones : Fin 10 → ℝ)
  (h_distinct: ∀ i j : Fin 10, i ≠ j → stones i ≠ stones j)
  (h_pairwise_sums_distinct : ∀ i j k l : Fin 10, 
    i ≠ j → k ≠ l → stones i + stones j ≠ stones k + stones l) :
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≥ stones j) ∨ 
  (∃ i : Fin 10, ∀ j : Fin 10, stones i ≤ stones j) :=
sorry

end find_heaviest_or_lightest_l117_117907


namespace total_cakes_served_l117_117903

-- Define the conditions
def cakes_lunch_today := 5
def cakes_dinner_today := 6
def cakes_yesterday := 3

-- Define the theorem we want to prove
theorem total_cakes_served : (cakes_lunch_today + cakes_dinner_today + cakes_yesterday) = 14 :=
by
  -- The proof is not required, so we use sorry to skip it
  sorry

end total_cakes_served_l117_117903


namespace quadratic_to_square_form_l117_117402

theorem quadratic_to_square_form (x m n : ℝ) (h : x^2 + 6 * x - 1 = 0) 
  (hm : m = 3) (hn : n = 10) : m - n = -7 :=
by 
  -- Proof steps (skipped, as per instructions)
  sorry

end quadratic_to_square_form_l117_117402


namespace symmetric_shading_additional_squares_l117_117367

theorem symmetric_shading_additional_squares :
  let initial_shaded : List (ℕ × ℕ) := [(1, 1), (2, 4), (4, 3)]
  let required_horizontal_symmetry := [(4, 1), (1, 6), (4, 6)]
  let required_vertical_symmetry := [(2, 3), (1, 3)]
  let total_additional_squares := required_horizontal_symmetry ++ required_vertical_symmetry
  let final_shaded := initial_shaded ++ total_additional_squares
  ∀ s ∈ total_additional_squares, s ∉ initial_shaded →
    final_shaded.length - initial_shaded.length = 5 :=
by
  sorry

end symmetric_shading_additional_squares_l117_117367


namespace steps_in_staircase_using_210_toothpicks_l117_117144

-- Define the conditions
def first_step : Nat := 3
def increment : Nat := 2
def total_toothpicks_5_steps : Nat := 55

-- Define required theorem
theorem steps_in_staircase_using_210_toothpicks : ∃ (n : ℕ), (n * (n + 2) = 210) ∧ n = 13 :=
by
  sorry

end steps_in_staircase_using_210_toothpicks_l117_117144


namespace closest_integer_to_cube_root_of_500_l117_117006

theorem closest_integer_to_cube_root_of_500 :
  ∃ n : ℤ, (∀ m : ℤ, |m^3 - 500| ≥ |8^3 - 500|) := 
sorry

end closest_integer_to_cube_root_of_500_l117_117006


namespace find_b_l117_117545

-- Define the quadratic equation
def quadratic_eq (b : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x - 15

-- Prove that b = 49/8 given -8 is a solution to the quadratic equation
theorem find_b (b : ℝ) : quadratic_eq b (-8) = 0 -> b = 49 / 8 :=
by
  intro h
  sorry

end find_b_l117_117545


namespace astroid_arc_length_l117_117226

theorem astroid_arc_length (a : ℝ) (h_a : a > 0) :
  ∃ l : ℝ, (l = 6 * a) ∧ 
  ((a = 1 → l = 6) ∧ (a = 2/3 → l = 4)) := 
by
  sorry

end astroid_arc_length_l117_117226


namespace find_m_l117_117696

open Set

def A (m: ℝ) := {x : ℝ | x^2 - m * x + m^2 - 19 = 0}

def B := {x : ℝ | x^2 - 5 * x + 6 = 0}

def C := ({2, -4} : Set ℝ)

theorem find_m (m : ℝ) (ha : A m ∩ B ≠ ∅) (hb : A m ∩ C = ∅) : m = -2 :=
  sorry

end find_m_l117_117696


namespace distance_between_points_l117_117238

theorem distance_between_points : ∀ (A B : ℤ), A = 5 → B = -3 → |A - B| = 8 :=
by
  intros A B hA hB
  rw [hA, hB]
  norm_num

end distance_between_points_l117_117238


namespace ratio_induction_l117_117149

theorem ratio_induction (k : ℕ) (hk : k > 0) :
    (k + 2) * (k + 3) / (2 * (2 * k + 1)) = 1 := by
sorry

end ratio_induction_l117_117149


namespace find_special_three_digit_numbers_l117_117173

theorem find_special_three_digit_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 
  (100 * a + 10 * b + (c + 3)) % 10 + (100 * a + 10 * (b + 1) + c).div 10 % 10 + (100 * (a + 1) + 10 * b + c).div 100 % 10 + 3 = 
  (a + b + c) / 3)} → n = 117 ∨ n = 207 ∨ n = 108 :=
by
  sorry

end find_special_three_digit_numbers_l117_117173


namespace functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l117_117264

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

def profit_maximized (x : ℝ) : Prop :=
  daily_sales_profit x = -5 * (80 - x)^2 + 4500

def sufficient_profit_range (x : ℝ) : Prop :=
  daily_sales_profit x >= 4000 ∧ (x - 50) * (500 - 5 * x) <= 7000

theorem functional_relationship (x : ℝ) : daily_sales_profit x = -5 * x^2 + 800 * x - 27500 :=
  sorry

theorem profit_maximized_at (x : ℝ) : profit_maximized x → x = 80 ∧ daily_sales_profit x = 4500 :=
  sorry

theorem sufficient_profit_range_verified (x : ℝ) : sufficient_profit_range x → 82 ≤ x ∧ x ≤ 90 :=
  sorry

end functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l117_117264


namespace pentagon_angle_sum_l117_117384

theorem pentagon_angle_sum
  (a b c d : ℝ) (Q : ℝ)
  (sum_angles : 180 * (5 - 2) = 540)
  (given_angles : a = 130 ∧ b = 80 ∧ c = 105 ∧ d = 110) :
  Q = 540 - (a + b + c + d) := by
  sorry

end pentagon_angle_sum_l117_117384


namespace integral_curves_l117_117672

theorem integral_curves (y x : ℝ) : 
  (∃ k : ℝ, (y - x) / (y + x) = k) → 
  (∃ c : ℝ, y = x * (c + 1) / (c - 1)) ∨ (y = 0) ∨ (y = x) ∨ (x = 0) :=
by
  sorry

end integral_curves_l117_117672


namespace cost_per_set_l117_117911

variable (C : ℝ)

theorem cost_per_set :
  let total_manufacturing_cost := 10000 + 500 * C
  let revenue := 500 * 50
  let profit := revenue - total_manufacturing_cost
  profit = 5000 → C = 20 := 
by
  sorry

end cost_per_set_l117_117911


namespace find_fx_l117_117159

variable {e : ℝ} {a : ℝ} (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (hodd : odd_function f)
variable (hdef : ∀ x, -e ≤ x → x < 0 → f x = a * x + Real.log (-x))

theorem find_fx (x : ℝ) (hx : 0 < x ∧ x ≤ e) : f x = a * x - Real.log x :=
by
  sorry

end find_fx_l117_117159


namespace simplify_and_evaluate_l117_117291

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) :
  ( ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3 * a) / (a^2 - 1)) = -1/2 ) :=
by
  sorry

end simplify_and_evaluate_l117_117291


namespace covered_area_of_strips_l117_117505

/-- Four rectangular strips of paper, each 16 cm long and 2 cm wide, overlap on a table. 
    We need to prove that the total area of the table surface covered by these strips is 112 cm². --/

theorem covered_area_of_strips (length width : ℝ) (number_of_strips : ℕ) (intersections : ℕ) 
    (area_of_strip : ℝ) (total_area_without_overlap : ℝ) (overlap_area : ℝ) 
    (actual_covered_area : ℝ) :
  length = 16 →
  width = 2 →
  number_of_strips = 4 →
  intersections = 4 →
  area_of_strip = length * width →
  total_area_without_overlap = number_of_strips * area_of_strip →
  overlap_area = intersections * (width * width) →
  actual_covered_area = total_area_without_overlap - overlap_area →
  actual_covered_area = 112 := 
by
  intros
  sorry

end covered_area_of_strips_l117_117505


namespace two_point_question_count_l117_117037

/-- Define the number of questions and points on the test,
    and prove that the number of 2-point questions is 30. -/
theorem two_point_question_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 := by
  sorry

end two_point_question_count_l117_117037


namespace largest_ordered_pair_exists_l117_117870

-- Define the condition for ordered pairs (a, b)
def ordered_pair_condition (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ ∃ (k : ℤ), (a + b) * (a + b + 1) = k * a * b

-- Define the specific ordered pair to be checked
def specific_pair (a b : ℤ) : Prop :=
  a = 35 ∧ b = 90

-- The main statement to be proven
theorem largest_ordered_pair_exists : specific_pair 35 90 ∧ ordered_pair_condition 35 90 :=
by
  sorry

end largest_ordered_pair_exists_l117_117870


namespace tan_alpha_minus_pi_over_4_l117_117618

noncomputable def alpha : ℝ := sorry
axiom alpha_in_range : -Real.pi / 2 < alpha ∧ alpha < 0
axiom cos_alpha : Real.cos alpha = (Real.sqrt 5) / 5

theorem tan_alpha_minus_pi_over_4 : Real.tan (alpha - Real.pi / 4) = 3 := by
  sorry

end tan_alpha_minus_pi_over_4_l117_117618


namespace max_zeros_in_product_l117_117807

theorem max_zeros_in_product (a b c : ℕ) (h_sum : a + b + c = 1003) : ∃ N, N = 7 ∧ ∀ p : ℕ, (a * b * c = p) → (∃ k, p = 10^k ∧ k ≤ N) ∧ (∀ k, p = 10^k → k ≤ 7) :=
by
  sorry

end max_zeros_in_product_l117_117807


namespace Matilda_age_is_35_l117_117772

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l117_117772


namespace smallest_n_identity_matrix_l117_117910

noncomputable def rotation_45_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_45_matrix ^ n = 1) ∧ ∀ m : ℕ, m > 0 → (rotation_45_matrix ^ m = 1 → n ≤ m) := sorry

end smallest_n_identity_matrix_l117_117910


namespace y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l117_117078

def y : ℕ := 54 + 108 + 162 + 216 + 648 + 810 + 972

theorem y_is_multiple_of_2 : 2 ∣ y :=
sorry

theorem y_is_multiple_of_3 : 3 ∣ y :=
sorry

theorem y_is_multiple_of_6 : 6 ∣ y :=
sorry

theorem y_is_multiple_of_9 : 9 ∣ y :=
sorry

end y_is_multiple_of_2_y_is_multiple_of_3_y_is_multiple_of_6_y_is_multiple_of_9_l117_117078


namespace sphere_surface_area_l117_117237

theorem sphere_surface_area (edge_length : ℝ) (diameter_eq_edge_length : (diameter : ℝ) = edge_length) :
  (edge_length = 2) → (diameter = 2) → (surface_area : ℝ) = 8 * Real.pi :=
by
  sorry

end sphere_surface_area_l117_117237


namespace terminal_sides_positions_l117_117790

def in_third_quadrant (θ : ℝ) (k : ℤ) : Prop :=
  (180 + k * 360 : ℝ) < θ ∧ θ < (270 + k * 360 : ℝ)

theorem terminal_sides_positions (θ : ℝ) (k : ℤ) :
  in_third_quadrant θ k →
  ((2 * θ > 360 + 2 * k * 360 ∧ 2 * θ < 540 + 2 * k * 360) ∨
   (90 + k * 180 < θ / 2 ∧ θ / 2 < 135 + k * 180) ∨
   (2 * θ = 360 + 2 * k * 360) ∨ (2 * θ = 540 + 2 * k * 360) ∨ 
   (θ / 2 = 90 + k * 180) ∨ (θ / 2 = 135 + k * 180)) :=
by
  intro h
  sorry

end terminal_sides_positions_l117_117790


namespace find_constant_l117_117830

variable (constant : ℝ)

theorem find_constant (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 1 - 2 * t)
  (h2 : y = constant * t - 2)
  (h3 : x = y) : constant = 2 :=
by
  sorry

end find_constant_l117_117830


namespace jasmine_pies_l117_117607

-- Definitions based on the given conditions
def total_pies : Nat := 30
def raspberry_part : Nat := 2
def peach_part : Nat := 5
def plum_part : Nat := 3
def total_parts : Nat := raspberry_part + peach_part + plum_part

-- Calculate pies per part
def pies_per_part : Nat := total_pies / total_parts

-- Prove the statement
theorem jasmine_pies :
  (plum_part * pies_per_part = 9) :=
by
  -- The statement and proof will go here, but we are skipping the proof part.
  sorry

end jasmine_pies_l117_117607


namespace investment_duration_p_l117_117749

-- Given the investments ratio, profits ratio, and time period for q,
-- proving the time period of p's investment is 7 months.
theorem investment_duration_p (T_p T_q : ℕ) 
  (investment_ratio : 7 * T_p = 5 * T_q) 
  (profit_ratio : 7 * T_p / T_q = 7 / 10)
  (T_q_eq : T_q = 14) : T_p = 7 :=
by
  sorry

end investment_duration_p_l117_117749


namespace michael_remaining_books_l117_117604

theorem michael_remaining_books (total_books : ℕ) (read_percentage : ℚ) 
  (H1 : total_books = 210) (H2 : read_percentage = 0.60) : 
  (total_books - (read_percentage * total_books) : ℚ) = 84 :=
by
  sorry

end michael_remaining_books_l117_117604


namespace wall_building_time_l117_117500

variable (r : ℝ) -- rate at which one worker can build the wall
variable (W : ℝ) -- the wall in units, let’s denote one whole wall as 1 unit

theorem wall_building_time:
  (∀ (w t : ℝ), W = (60 * r) * t → W = (30 * r) * 6) :=
by
  sorry

end wall_building_time_l117_117500


namespace area_of_sandbox_is_correct_l117_117509

-- Define the length and width of the sandbox
def length_sandbox : ℕ := 312
def width_sandbox : ℕ := 146

-- Define the area calculation
def area_sandbox (length width : ℕ) : ℕ := length * width

-- The theorem stating that the area of the sandbox is 45552 cm²
theorem area_of_sandbox_is_correct : area_sandbox length_sandbox width_sandbox = 45552 := sorry

end area_of_sandbox_is_correct_l117_117509


namespace number_multiply_increase_l117_117544

theorem number_multiply_increase (x : ℕ) (h : 25 * x = 25 + 375) : x = 16 := by
  sorry

end number_multiply_increase_l117_117544


namespace greatest_measure_length_l117_117766

theorem greatest_measure_length :
  let l1 := 18000
  let l2 := 50000
  let l3 := 1520
  ∃ d, d = Int.gcd (Int.gcd l1 l2) l3 ∧ d = 40 :=
by
  sorry

end greatest_measure_length_l117_117766


namespace ben_daily_spending_l117_117915

variable (S : ℕ)

def daily_savings (S : ℕ) : ℕ := 50 - S

def total_savings (S : ℕ) : ℕ := 7 * daily_savings S

def final_amount (S : ℕ) : ℕ := 2 * total_savings S + 10

theorem ben_daily_spending :
  final_amount 15 = 500 :=
by
  unfold final_amount
  unfold total_savings
  unfold daily_savings
  sorry

end ben_daily_spending_l117_117915


namespace smallest_zarks_l117_117010

theorem smallest_zarks (n : ℕ) : (n^2 > 15 * n) → (n ≥ 16) := sorry

end smallest_zarks_l117_117010


namespace line_intersects_circle_and_focus_condition_l117_117811

variables {x y k : ℝ}

/-- The line l intersects the circle x^2 + y^2 + 2x - 4y + 1 = 0 at points A and B. If the midpoint of the chord AB is the focus of the parabola x^2 = 4y, then prove that the equation of the line l is x - y + 1 = 0. -/
theorem line_intersects_circle_and_focus_condition :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, l x = y) ∧
  (∀ A B : ℝ × ℝ, ∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ M = (0, 1)) ∧
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧
  x^2 = 4*y ) → 
  (∀ x y : ℝ, x - y + 1 = 0) :=
sorry

end line_intersects_circle_and_focus_condition_l117_117811


namespace average_productivity_l117_117616

theorem average_productivity (T : ℕ) (total_words : ℕ) (increased_time_fraction : ℚ) (increased_productivity_fraction : ℚ) :
  T = 100 →
  total_words = 60000 →
  increased_time_fraction = 0.2 →
  increased_productivity_fraction = 1.5 →
  (total_words / T : ℚ) = 600 :=
by
  sorry

end average_productivity_l117_117616


namespace polynomial_value_at_2_l117_117405

def f (x : ℝ) : ℝ := 2 * x^5 + 4 * x^4 - 2 * x^3 - 3 * x^2 + x

theorem polynomial_value_at_2 : f 2 = 102 := by
  sorry

end polynomial_value_at_2_l117_117405


namespace intersection_eq_zero_l117_117620

def M := { x : ℤ | abs (x - 3) < 4 }
def N := { x : ℤ | x^2 + x - 2 < 0 }

theorem intersection_eq_zero : M ∩ N = {0} := 
  by
    sorry

end intersection_eq_zero_l117_117620


namespace lattice_points_in_region_l117_117847

theorem lattice_points_in_region :
  ∃ n : ℕ, n = 12 ∧ 
  ( ∀ x y : ℤ, (y = x ∨ y = -x ∨ y = -x^2 + 4) → n = 12) :=
by
  sorry

end lattice_points_in_region_l117_117847


namespace ten_unique_positive_odd_integers_equality_l117_117079

theorem ten_unique_positive_odd_integers_equality {x : ℕ} (h1: x = 3):
  ∃ S : Finset ℕ, S.card = 10 ∧ 
    (∀ n ∈ S, n < 100 ∧ n % 2 = 1 ∧ 
      ∃ k : ℕ, k % 2 = 1 ∧ n = k * x) :=
by
  sorry

end ten_unique_positive_odd_integers_equality_l117_117079


namespace value_of_f_nine_halves_l117_117263

noncomputable def f : ℝ → ℝ := sorry  -- Define f with noncomputable since it's not explicitly given

axiom even_function (x : ℝ) : f x = f (-x)  -- Define the even function property
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0 -- Define the property that f is not identically zero
axiom functional_equation (x : ℝ) : x * f (x + 1) = (x + 1) * f x -- Define the given functional equation

theorem value_of_f_nine_halves : f (9 / 2) = 0 := by
  sorry

end value_of_f_nine_halves_l117_117263


namespace find_number_l117_117661

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number_l117_117661


namespace first_ring_time_l117_117094

-- Define the properties of the clock
def rings_every_three_hours : Prop := ∀ n : ℕ, 3 * n < 24
def rings_eight_times_a_day : Prop := ∀ n : ℕ, n = 8 → 3 * n = 24

-- The theorem statement
theorem first_ring_time : rings_every_three_hours → rings_eight_times_a_day → (∀ n : ℕ, n = 1 → 3 * n = 3) := 
    sorry

end first_ring_time_l117_117094


namespace code_length_is_4_l117_117289

-- Definitions based on conditions provided
def code_length : ℕ := 4 -- Each code consists of 4 digits
def total_codes_with_leading_zeros : ℕ := 10^code_length -- Total possible codes allowing leading zeros
def total_codes_without_leading_zeros : ℕ := 9 * 10^(code_length - 1) -- Total possible codes disallowing leading zeros
def codes_lost_if_no_leading_zeros : ℕ := total_codes_with_leading_zeros - total_codes_without_leading_zeros -- Codes lost if leading zeros are disallowed
def manager_measured_codes_lost : ℕ := 10000 -- Manager's incorrect measurement

-- Theorem to be proved based on the problem
theorem code_length_is_4 : code_length = 4 :=
by
  sorry

end code_length_is_4_l117_117289


namespace largest_number_l117_117981

theorem largest_number (a b c : ℝ) (h1 : a + b + c = 67) (h2 : c - b = 7) (h3 : b - a = 5) : c = 86 / 3 := 
by sorry

end largest_number_l117_117981


namespace longer_string_length_l117_117754

theorem longer_string_length 
  (total_length : ℕ) 
  (length_diff : ℕ)
  (h_total_length : total_length = 348)
  (h_length_diff : length_diff = 72) :
  ∃ (L S : ℕ), 
  L - S = length_diff ∧
  L + S = total_length ∧ 
  L = 210 :=
by
  sorry

end longer_string_length_l117_117754


namespace series_proof_l117_117867

noncomputable def series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / (b ^ (n + 1))

noncomputable def transformed_series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / ((a + 2 * b) ^ (n + 1))

theorem series_proof (a b : ℝ)
  (h1 : series_sum a b = 7)
  (h2 : a = 7 * (b - 1)) :
  transformed_series_sum a b = 7 * (b - 1) / (9 * b - 8) :=
by sorry

end series_proof_l117_117867


namespace gain_percent_of_articles_l117_117184

theorem gain_percent_of_articles (C S : ℝ) (h : 50 * C = 15 * S) : (S - C) / C * 100 = 233.33 :=
by
  sorry

end gain_percent_of_articles_l117_117184


namespace four_digit_numbers_using_0_and_9_l117_117876

theorem four_digit_numbers_using_0_and_9 :
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d, d ∈ Nat.digits 10 n → (d = 0 ∨ d = 9)} = {9000, 9009, 9090, 9099, 9900, 9909, 9990, 9999} :=
by
  sorry

end four_digit_numbers_using_0_and_9_l117_117876


namespace one_over_x_plus_one_over_y_eq_two_l117_117546

theorem one_over_x_plus_one_over_y_eq_two 
  (x y : ℝ)
  (h1 : 3^x = Real.sqrt 12)
  (h2 : 4^y = Real.sqrt 12) : 
  1 / x + 1 / y = 2 := 
by 
  sorry

end one_over_x_plus_one_over_y_eq_two_l117_117546


namespace value_of_expression_l117_117376

theorem value_of_expression (x : ℤ) (h : x = 3) : x^6 - 3 * x = 720 := by
  sorry

end value_of_expression_l117_117376


namespace number_of_2_dollar_socks_l117_117854

-- Given conditions
def total_pairs (a b c : ℕ) := a + b + c = 15
def total_cost (a b c : ℕ) := 2 * a + 4 * b + 5 * c = 41
def min_each_pair (a b c : ℕ) := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- To be proved
theorem number_of_2_dollar_socks (a b c : ℕ) (h1 : total_pairs a b c) (h2 : total_cost a b c) (h3 : min_each_pair a b c) : 
  a = 11 := 
  sorry

end number_of_2_dollar_socks_l117_117854


namespace correct_calculation_l117_117323

theorem correct_calculation (x : ℕ) (h : 637 = x + 238) : x - 382 = 17 :=
by
  sorry

end correct_calculation_l117_117323


namespace number_of_sweet_potatoes_sold_to_mrs_adams_l117_117025

def sweet_potatoes_harvested := 80
def sweet_potatoes_sold_to_mr_lenon := 15
def sweet_potatoes_unsold := 45

def sweet_potatoes_sold_to_mrs_adams :=
  sweet_potatoes_harvested - sweet_potatoes_sold_to_mr_lenon - sweet_potatoes_unsold

theorem number_of_sweet_potatoes_sold_to_mrs_adams :
  sweet_potatoes_sold_to_mrs_adams = 20 := by
  sorry

end number_of_sweet_potatoes_sold_to_mrs_adams_l117_117025


namespace jason_flame_time_l117_117857

-- Define firing interval and flame duration
def firing_interval := 15
def flame_duration := 5

-- Define the function to calculate seconds per minute
def seconds_per_minute (interval : ℕ) (duration : ℕ) : ℕ :=
  (60 / interval) * duration

-- Theorem to state the problem
theorem jason_flame_time : 
  seconds_per_minute firing_interval flame_duration = 20 := 
by
  sorry

end jason_flame_time_l117_117857


namespace greatest_possible_a_l117_117064

theorem greatest_possible_a (a : ℤ) (x : ℤ) (h_pos : 0 < a) (h_eq : x^3 + a * x^2 = -30) : 
  a ≤ 29 :=
sorry

end greatest_possible_a_l117_117064


namespace percentage_entree_cost_l117_117711

-- Conditions
def total_spent : ℝ := 50.0
def num_appetizers : ℝ := 2
def cost_per_appetizer : ℝ := 5.0
def total_appetizer_cost : ℝ := num_appetizers * cost_per_appetizer
def total_entree_cost : ℝ := total_spent - total_appetizer_cost

-- Proof Problem
theorem percentage_entree_cost :
  (total_entree_cost / total_spent) * 100 = 80 :=
sorry

end percentage_entree_cost_l117_117711


namespace a_in_A_l117_117735

def A := {x : ℝ | x ≥ 2 * Real.sqrt 2}
def a : ℝ := 3

theorem a_in_A : a ∈ A :=
by 
  sorry

end a_in_A_l117_117735


namespace tom_found_dimes_l117_117662

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes_l117_117662


namespace perpendicular_line_x_intercept_l117_117253

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, ∃ y : ℝ, 4 * x + 5 * y = 10) →
  (∃ y : ℝ, y = (5/4) * x - 3) →
  (∃ x : ℝ, y = 0) →
  x = 12 / 5 :=
by
  sorry

end perpendicular_line_x_intercept_l117_117253


namespace algebraic_expression_value_l117_117558

theorem algebraic_expression_value (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (  ( ((x + 2)^2 * (x^2 - 2 * x + 4)^2) / ( (x^3 + 8)^2 ))^2
   * ( ((x - 2)^2 * (x^2 + 2 * x + 4)^2) / ( (x^3 - 8)^2 ))^2 ) = 1 :=
by
  sorry

end algebraic_expression_value_l117_117558


namespace june_ride_time_l117_117171

theorem june_ride_time (dist1 time1 dist2 time2 : ℝ) (h : dist1 = 2 ∧ time1 = 8 ∧ dist2 = 5 ∧ time2 = 20) :
  (dist2 / (dist1 / time1) = time2) := by
  -- using the defined conditions
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  -- simplifying the expression
  sorry

end june_ride_time_l117_117171


namespace james_total_matches_l117_117531

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l117_117531


namespace pyramid_bottom_right_value_l117_117155

theorem pyramid_bottom_right_value (a x y z b : ℕ) (h1 : 18 = (21 + x) / 2)
  (h2 : 14 = (21 + y) / 2) (h3 : 16 = (15 + z) / 2) (h4 : b = (21 + y) / 2) :
  a = 6 := 
sorry

end pyramid_bottom_right_value_l117_117155


namespace number_of_girls_l117_117721

theorem number_of_girls (d c : ℕ) (h1 : c = 2 * (d - 15)) (h2 : d - 15 = 5 * (c - 45)) : d = 40 := 
by
  sorry

end number_of_girls_l117_117721


namespace consecutive_numbers_sum_l117_117584

theorem consecutive_numbers_sum (n : ℤ) (h1 : (n - 1) * n * (n + 1) = 210) (h2 : ∀ m, (m - 1) * m * (m + 1) = 210 → (m - 1)^2 + m^2 + (m + 1)^2 ≥ (n - 1)^2 + n^2 + (n + 1)^2) :
  (n - 1) + n = 11 :=
by 
  sorry

end consecutive_numbers_sum_l117_117584


namespace arithmetic_sequence_l117_117901

variable (a : ℕ → ℕ)
variable (h : a 1 + 3 * a 8 + a 15 = 120)

theorem arithmetic_sequence (h : a 1 + 3 * a 8 + a 15 = 120) : a 2 + a 14 = 48 :=
sorry

end arithmetic_sequence_l117_117901


namespace subset_iff_l117_117602

open Set

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | 0 < x ∧ x < a}

theorem subset_iff (a : ℝ) : A ⊆ B a ↔ 2 ≤ a :=
by sorry

end subset_iff_l117_117602


namespace license_plate_count_l117_117560

def num_license_plates : Nat :=
  let letters := 26 -- choices for each of the first two letters
  let primes := 4 -- choices for prime digits
  let composites := 4 -- choices for composite digits
  letters * letters * (primes * composites * 2)

theorem license_plate_count : num_license_plates = 21632 :=
  by
  sorry

end license_plate_count_l117_117560


namespace rice_mixing_ratio_l117_117377

theorem rice_mixing_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4.5 * x + 8.75 * y) / (x + y) = 7.5 → y / x = 2.4 :=
by
  sorry

end rice_mixing_ratio_l117_117377


namespace jill_arrives_before_jack_l117_117859

def pool_distance : ℝ := 2
def jill_speed : ℝ := 12
def jack_speed : ℝ := 4
def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

theorem jill_arrives_before_jack
    (d : ℝ) (v_jill : ℝ) (v_jack : ℝ) (convert : ℝ → ℝ)
    (h_d : d = pool_distance)
    (h_vj : v_jill = jill_speed)
    (h_vk : v_jack = jack_speed)
    (h_convert : convert = hours_to_minutes) :
  convert (d / v_jack) - convert (d / v_jill) = 20 := by
  sorry

end jill_arrives_before_jack_l117_117859


namespace edge_length_of_prism_l117_117794

-- Definitions based on conditions
def rectangular_prism_edges : ℕ := 12
def total_edge_length : ℕ := 72

-- Proof problem statement
theorem edge_length_of_prism (num_edges : ℕ) (total_length : ℕ) (h1 : num_edges = rectangular_prism_edges) (h2 : total_length = total_edge_length) : 
  (total_length / num_edges) = 6 :=
by {
  -- The proof is omitted here as instructed
  sorry
}

end edge_length_of_prism_l117_117794


namespace cds_total_l117_117601

theorem cds_total (dawn_cds : ℕ) (h1 : dawn_cds = 10) (h2 : ∀ kristine_cds : ℕ, kristine_cds = dawn_cds + 7) :
  dawn_cds + (dawn_cds + 7) = 27 :=
by
  sorry

end cds_total_l117_117601


namespace smallest_number_l117_117709

theorem smallest_number (n : ℕ) : 
  (∀ k ∈ [12, 16, 18, 21, 28, 35, 39], ∃ m : ℕ, (n - 3) = k * m) → 
  n = 65517 := by
  sorry

end smallest_number_l117_117709


namespace arithmetic_sequence_sum_l117_117052

noncomputable def Sn (a d n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a d : ℕ) (h1 : a = 3 * d) (h2 : Sn a d 5 = 50) : Sn a d 8 = 104 :=
by
/-
  From the given conditions:
  - \(a_4\) is the geometric mean of \(a_2\) and \(a_7\) implies \(a = 3d\)
  - Sum of first 5 terms is 50 implies \(S_5 = 50\)
  We need to prove \(S_8 = 104\)
-/
  sorry

end arithmetic_sequence_sum_l117_117052


namespace tangency_splits_segments_l117_117005

def pentagon_lengths (a b c d e : ℕ) (h₁ : a = 1) (h₃ : c = 1) (x1 x2 : ℝ) :=
x1 + x2 = b ∧ x1 = 1/2 ∧ x2 = 1/2

theorem tangency_splits_segments {a b c d e : ℕ} (h₁ : a = 1) (h₃ : c = 1) :
    ∃ x1 x2 : ℝ, pentagon_lengths a b c d e h₁ h₃ x1 x2 :=
    by 
    sorry

end tangency_splits_segments_l117_117005


namespace product_of_xy_l117_117462

theorem product_of_xy (x y : ℝ) : 
  (1 / 5 * (x + y + 4 + 5 + 6) = 5) ∧ 
  (1 / 5 * ((x - 5) ^ 2 + (y - 5) ^ 2 + (4 - 5) ^ 2 + (5 - 5) ^ 2 + (6 - 5) ^ 2) = 2) 
  → x * y = 21 :=
by sorry

end product_of_xy_l117_117462


namespace range_of_m_satisfies_inequality_l117_117595

theorem range_of_m_satisfies_inequality (m : ℝ) :
  ((∀ x : ℝ, (1 - m^2) * x^2 - (1 + m) * x - 1 < 0) ↔ (m ≤ -1 ∨ m > 5/3)) :=
sorry

end range_of_m_satisfies_inequality_l117_117595


namespace total_pages_read_l117_117347

theorem total_pages_read (days : ℕ)
  (deshaun_books deshaun_pages_per_book lilly_percent ben_extra eva_factor sam_pages_per_day : ℕ)
  (lilly_percent_correct : lilly_percent = 75)
  (ben_extra_correct : ben_extra = 25)
  (eva_factor_correct : eva_factor = 2)
  (total_break_days : days = 80)
  (deshaun_books_correct : deshaun_books = 60)
  (deshaun_pages_per_book_correct : deshaun_pages_per_book = 320)
  (sam_pages_per_day_correct : sam_pages_per_day = 150) :
  deshaun_books * deshaun_pages_per_book +
  (lilly_percent * deshaun_books * deshaun_pages_per_book / 100) +
  (deshaun_books * (100 + ben_extra) / 100) * 280 +
  (eva_factor * (deshaun_books * (100 + ben_extra) / 100 * 280)) +
  (sam_pages_per_day * days) = 108450 := 
sorry

end total_pages_read_l117_117347


namespace absent_children_l117_117633

/-- On a school's annual day, sweets were to be equally distributed amongst 112 children. 
But on that particular day, some children were absent. Thus, the remaining children got 6 extra sweets. 
Each child was originally supposed to get 15 sweets. Prove that 32 children were absent. -/
theorem absent_children (A : ℕ) 
  (total_children : ℕ := 112) 
  (sweets_per_child : ℕ := 15) 
  (extra_sweets : ℕ := 6)
  (absent_eq : (total_children - A) * (sweets_per_child + extra_sweets) = total_children * sweets_per_child) : 
  A = 32 := 
by
  sorry

end absent_children_l117_117633


namespace cyclist_time_to_climb_and_descend_hill_l117_117482

noncomputable def hill_length : ℝ := 400 -- hill length in meters
noncomputable def ascent_speed_kmh : ℝ := 7.2 -- ascent speed in km/h
noncomputable def ascent_speed_ms : ℝ := ascent_speed_kmh * 1000 / 3600 -- ascent speed converted in m/s
noncomputable def descent_speed_ms : ℝ := 2 * ascent_speed_ms -- descent speed in m/s

noncomputable def time_to_climb : ℝ := hill_length / ascent_speed_ms -- time to climb in seconds
noncomputable def time_to_descend : ℝ := hill_length / descent_speed_ms -- time to descend in seconds
noncomputable def total_time : ℝ := time_to_climb + time_to_descend -- total time in seconds

theorem cyclist_time_to_climb_and_descend_hill : total_time = 300 :=
by
  sorry

end cyclist_time_to_climb_and_descend_hill_l117_117482


namespace trapezoid_area_calculation_l117_117219

noncomputable def trapezoid_area : ℝ :=
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2

theorem trapezoid_area_calculation :
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2 = 75 := 
by
  -- Validation of the translation to Lean 4. Proof steps are omitted.
  sorry

end trapezoid_area_calculation_l117_117219


namespace min_value_expression_l117_117700

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) :
  ∃ z : ℝ, z = 16 / 7 ∧ ∀ u > 0, ∀ v > 0, u + v = 4 → ((u^2 / (u + 1)) + (v^2 / (v + 2))) ≥ z :=
by
  sorry

end min_value_expression_l117_117700


namespace shaded_fraction_l117_117679

noncomputable def fraction_shaded (l w : ℝ) : ℝ :=
  1 - (1 / 8)

theorem shaded_fraction (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  fraction_shaded l w = 7 / 8 :=
by
  sorry

end shaded_fraction_l117_117679


namespace fraction_zero_l117_117663

theorem fraction_zero (x : ℝ) (h₁ : 2 * x = 0) (h₂ : x + 2 ≠ 0) : (2 * x) / (x + 2) = 0 :=
by {
  sorry
}

end fraction_zero_l117_117663


namespace naturals_less_than_10_l117_117285

theorem naturals_less_than_10 :
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end naturals_less_than_10_l117_117285


namespace table_tennis_matches_l117_117858

theorem table_tennis_matches (n : ℕ) :
  ∃ x : ℕ, 3 * 2 - x + n * (n - 1) / 2 = 50 ∧ x = 1 :=
by
  sorry

end table_tennis_matches_l117_117858


namespace markup_percentage_l117_117156

variable (W R : ℝ)

-- Condition: When sold at a 40% discount, a sweater nets the merchant a 30% profit on the wholesale cost.
def discount_condition : Prop := 0.6 * R = 1.3 * W

-- Theorem: The percentage markup of the sweater from wholesale to normal retail price is 116.67%
theorem markup_percentage (h : discount_condition W R) : (R - W) / W * 100 = 116.67 :=
by sorry

end markup_percentage_l117_117156


namespace square_area_l117_117084

theorem square_area (y : ℝ) (x₁ x₂ : ℝ) (s : ℝ) (A : ℝ) :
  y = 7 → 
  (y = x₁^2 + 4 * x₁ + 3) →
  (y = x₂^2 + 4 * x₂ + 3) →
  x₁ ≠ x₂ →
  s = |x₂ - x₁| → 
  A = s^2 →
  A = 32 :=
by
  intros hy intersection_x1 intersection_x2 hx1x2 hs ha
  sorry

end square_area_l117_117084


namespace cost_per_load_is_25_cents_l117_117615

def washes_per_bottle := 80
def price_per_bottle_on_sale := 20
def bottles := 2
def total_cost := bottles * price_per_bottle_on_sale -- 2 * 20 = 40
def total_loads := bottles * washes_per_bottle -- 2 * 80 = 160
def cost_per_load_in_dollars := total_cost / total_loads -- 40 / 160 = 0.25
def cost_per_load_in_cents := cost_per_load_in_dollars * 100

theorem cost_per_load_is_25_cents :
  cost_per_load_in_cents = 25 :=
by 
  sorry

end cost_per_load_is_25_cents_l117_117615


namespace sum_of_digits_is_2640_l117_117222

theorem sum_of_digits_is_2640 (x : ℕ) (h_cond : (1 + 3 + 4 + 6 + x) * (Nat.factorial 5) = 2640) : x = 8 := by
  sorry

end sum_of_digits_is_2640_l117_117222


namespace number_of_kittens_l117_117057

-- Definitions for the given conditions.
def total_animals : ℕ := 77
def hamsters : ℕ := 15
def birds : ℕ := 30

-- The proof problem statement.
theorem number_of_kittens : total_animals - hamsters - birds = 32 := by
  sorry

end number_of_kittens_l117_117057


namespace unique_triad_l117_117587

theorem unique_triad (x y z : ℕ) 
  (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) 
  (h_gcd: Nat.gcd (Nat.gcd x y) z = 1)
  (h_div_properties: (z ∣ x + y) ∧ (x ∣ y + z) ∧ (y ∣ z + x)) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end unique_triad_l117_117587


namespace value_of_clothing_piece_eq_l117_117102

def annual_remuneration := 10
def work_months := 7
def received_silver_coins := 2

theorem value_of_clothing_piece_eq : 
  ∃ x : ℝ, (x + received_silver_coins) * 12 = (x + annual_remuneration) * work_months → x = 9.2 :=
by
  sorry

end value_of_clothing_piece_eq_l117_117102


namespace interval_of_a_l117_117276

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  f a n.succ  -- since ℕ in Lean includes 0, use n.succ to start from 1

-- The main theorem to prove
theorem interval_of_a (a : ℝ) : (∀ n : ℕ, n ≠ 0 → a_n a n < a_n a (n + 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end interval_of_a_l117_117276


namespace smallest_number_collected_l117_117158

-- Define the numbers collected by each person according to the conditions
def jungkook : ℕ := 6 * 3
def yoongi : ℕ := 4
def yuna : ℕ := 5

-- The statement to prove
theorem smallest_number_collected : yoongi = min (min jungkook yoongi) yuna :=
by sorry

end smallest_number_collected_l117_117158


namespace sufficient_but_not_necessary_condition_for_subset_l117_117046

variable {A B : Set ℕ}
variable {a : ℕ}

theorem sufficient_but_not_necessary_condition_for_subset (hA : A = {1, a}) (hB : B = {1, 2, 3}) :
  (a = 3 → A ⊆ B) ∧ (A ⊆ B → (a = 3 ∨ a = 2)) ∧ ¬(A ⊆ B → a = 3) := by
sorry

end sufficient_but_not_necessary_condition_for_subset_l117_117046


namespace monotonic_increasing_interval_l117_117034

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ (Real.sqrt (2 * x - x ^ 2))

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 1 ≤ x ∧ x < 2 → ∀ x1 x2, x1 < x2 → f x1 ≤ f x2 :=
by
  sorry

end monotonic_increasing_interval_l117_117034


namespace power_of_2_l117_117229

theorem power_of_2 (n : ℕ) (h1 : n ≥ 1) (h2 : ∃ m : ℕ, m ≥ 1 ∧ (2^n - 1) ∣ (m^2 + 9)) : ∃ k : ℕ, n = 2^k := 
sorry

end power_of_2_l117_117229


namespace cost_of_projector_and_whiteboard_l117_117324

variable (x : ℝ)

def cost_of_projector : ℝ := x
def cost_of_whiteboard : ℝ := x + 4000
def total_cost_eq_44000 : Prop := 4 * (x + 4000) + 3 * x = 44000

theorem cost_of_projector_and_whiteboard 
  (h : total_cost_eq_44000 x) : 
  cost_of_projector x = 4000 ∧ cost_of_whiteboard x = 8000 :=
by
  sorry

end cost_of_projector_and_whiteboard_l117_117324


namespace sequence_divisible_by_11_l117_117468

theorem sequence_divisible_by_11 
  (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
  (∀ n, n = 4 ∨ n = 8 ∨ n ≥ 10 → 11 ∣ a n) := sorry

end sequence_divisible_by_11_l117_117468


namespace div_by_eleven_l117_117210

theorem div_by_eleven (n : ℤ) : 11 ∣ ((n + 11)^2 - n^2) :=
by
  sorry

end div_by_eleven_l117_117210


namespace jordan_more_novels_than_maxime_l117_117370

theorem jordan_more_novels_than_maxime :
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  jordan_novels - maxime_novels = 51 :=
by
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  sorry

end jordan_more_novels_than_maxime_l117_117370


namespace sum_of_remainders_l117_117194

-- Definitions of the given problem
def a : ℕ := 1234567
def b : ℕ := 123

-- First remainder calculation
def r1 : ℕ := a % b

-- Second remainder calculation with the power
def r2 : ℕ := (2 ^ r1) % b

-- The proof statement
theorem sum_of_remainders : r1 + r2 = 29 := by
  sorry

end sum_of_remainders_l117_117194


namespace probability_first_die_l117_117442

theorem probability_first_die (n : ℕ) (n_pos : n = 4025) (m : ℕ) (m_pos : m = 2012) : 
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  (favorable_outcomes / total_outcomes : ℚ) = 1006 / 4025 :=
by
  have h_n : n = 4025 := n_pos
  have h_m : m = 2012 := m_pos
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  sorry

end probability_first_die_l117_117442


namespace sum_of_numbers_l117_117726

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : a^2 + b^2 + c^2 = 62) 
  (h₂ : ab + bc + ca = 131) : 
  a + b + c = 18 :=
sorry

end sum_of_numbers_l117_117726


namespace packs_bought_l117_117934

theorem packs_bought (total_uncommon : ℕ) (cards_per_pack : ℕ) (fraction_uncommon : ℚ) 
  (total_packs : ℕ) (uncommon_per_pack : ℕ)
  (h1 : cards_per_pack = 20)
  (h2 : fraction_uncommon = 1/4)
  (h3 : uncommon_per_pack = fraction_uncommon * cards_per_pack)
  (h4 : total_uncommon = 50)
  (h5 : total_packs = total_uncommon / uncommon_per_pack)
  : total_packs = 10 :=
by 
  sorry

end packs_bought_l117_117934


namespace total_gulbis_l117_117022

theorem total_gulbis (dureums fish_per_dureum : ℕ) (h1 : dureums = 156) (h2 : fish_per_dureum = 20) : dureums * fish_per_dureum = 3120 :=
by
  sorry

end total_gulbis_l117_117022


namespace yuebao_scientific_notation_l117_117623

-- Definition of converting a number to scientific notation
def scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10 ^ n

-- The specific problem statement
theorem yuebao_scientific_notation :
  scientific_notation (1853 * 10 ^ 9) 1.853 11 :=
by
  sorry

end yuebao_scientific_notation_l117_117623


namespace P_lt_Q_l117_117399

noncomputable def P (a : ℝ) : ℝ := (Real.sqrt (a + 41)) - (Real.sqrt (a + 40))
noncomputable def Q (a : ℝ) : ℝ := (Real.sqrt (a + 39)) - (Real.sqrt (a + 38))

theorem P_lt_Q (a : ℝ) (h : a > -38) : P a < Q a := by sorry

end P_lt_Q_l117_117399


namespace pythagorean_triple_B_l117_117055

def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_B : isPythagoreanTriple 3 4 5 :=
by
  sorry

end pythagorean_triple_B_l117_117055


namespace gain_percentage_l117_117969

theorem gain_percentage (CP SP : ℕ) (h_sell : SP = 10 * CP) : 
  (10 * CP / 25 * CP) * 100 = 40 := by
  sorry

end gain_percentage_l117_117969


namespace integers_within_range_l117_117192

def is_within_range (n : ℤ) : Prop :=
  (-1.3 : ℝ) < (n : ℝ) ∧ (n : ℝ) < 2.8

theorem integers_within_range :
  { n : ℤ | is_within_range n } = {-1, 0, 1, 2} :=
by
  sorry

end integers_within_range_l117_117192


namespace jinho_initial_money_l117_117248

variable (M : ℝ)

theorem jinho_initial_money :
  (M / 2 + 300) + (((M / 2 - 300) / 2) + 400) = M :=
by
  -- This proof is yet to be completed.
  sorry

end jinho_initial_money_l117_117248


namespace building_houses_200_people_l117_117436

-- Define number of floors, apartments per floor, and people per apartment as constants
def numFloors := 25
def apartmentsPerFloor := 4
def peoplePerApartment := 2

-- Define the total number of apartments
def totalApartments := numFloors * apartmentsPerFloor

-- Define the total number of people
def totalPeople := totalApartments * peoplePerApartment

theorem building_houses_200_people : totalPeople = 200 :=
by
  sorry

end building_houses_200_people_l117_117436


namespace inequality_proof_l117_117880

noncomputable def a : Real := (1 / 3) ^ Real.pi
noncomputable def b : Real := (1 / 3) ^ (1 / 2 : Real)
noncomputable def c : Real := Real.pi ^ (1 / 2 : Real)

theorem inequality_proof : a < b ∧ b < c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l117_117880


namespace find_numbers_l117_117873

theorem find_numbers :
  ∃ (x y z : ℕ), x = y + 75 ∧ 
                 (x * y = z + 1000) ∧
                 (z = 227 * y + 113) ∧
                 (x = 234) ∧ 
                 (y = 159) := by
  sorry

end find_numbers_l117_117873


namespace find_x_l117_117095

noncomputable def S (x : ℝ) : ℝ := 1 + 3 * x + 5 * x^2 + 7 * x^3 + ∑' n, (2 * n - 1) * x^n

theorem find_x (x : ℝ) (h : S x = 16) : x = 3/4 :=
sorry

end find_x_l117_117095


namespace vanya_correct_answers_l117_117510

theorem vanya_correct_answers (x : ℕ) (y : ℕ) (h1 : y = 50 - x) (h2 : 7 * x = 3 * y) : x = 15 :=
by
  sorry

end vanya_correct_answers_l117_117510


namespace center_of_circle_l117_117694

theorem center_of_circle :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 1 → (x = -2 ∧ y = 1) :=
by
  intros x y hyp
  -- Here, we would perform the steps of comparing to the standard form and proving the center.
  sorry

end center_of_circle_l117_117694


namespace jared_annual_earnings_l117_117167

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l117_117167


namespace Mary_more_than_Tim_l117_117490

-- Define the incomes
variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.80 * J
def Mary_income : Prop := M = 1.28 * J

-- Theorem statement to prove
theorem Mary_more_than_Tim (J T M : ℝ) (h1 : Tim_income J T)
  (h2 : Mary_income J M) : ((M - T) / T) * 100 = 60 :=
by
  -- Including sorry to skip the proof
  sorry

end Mary_more_than_Tim_l117_117490


namespace bike_riders_count_l117_117821

variables (B H : ℕ)

theorem bike_riders_count
  (h₁ : H = B + 178)
  (h₂ : H + B = 676) :
  B = 249 :=
sorry

end bike_riders_count_l117_117821


namespace abc_zero_l117_117842

theorem abc_zero {a b c : ℝ} 
(h1 : (a + b) * (b + c) * (c + a) = a * b * c)
(h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = a^3 * b^3 * c^3) : 
a * b * c = 0 := 
by sorry

end abc_zero_l117_117842


namespace least_pos_integer_with_8_factors_l117_117746

theorem least_pos_integer_with_8_factors : 
  ∃ k : ℕ, (k > 0 ∧ ((∃ m n p q : ℕ, k = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, k = p^7 ∧ Prime p)) ∧ 
            ∀ l : ℕ, (l > 0 ∧ ((∃ m n p q : ℕ, l = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, l = p^7 ∧ Prime p)) → k ≤ l)) ∧ k = 24 :=
sorry

end least_pos_integer_with_8_factors_l117_117746


namespace prime_p_squared_plus_71_divisors_l117_117581

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def num_distinct_divisors (n : ℕ) : ℕ :=
  (factors n).toFinset.card

theorem prime_p_squared_plus_71_divisors (p : ℕ) (hp : is_prime p) 
  (hdiv : num_distinct_divisors (p ^ 2 + 71) ≤ 10) : p = 2 ∨ p = 3 :=
sorry

end prime_p_squared_plus_71_divisors_l117_117581


namespace evaluate_f_l117_117866

def f (x : ℝ) : ℝ := x^2 + 4*x - 3

theorem evaluate_f (x : ℝ) : f (x + 1) = x^2 + 6*x + 2 :=
by 
  -- The proof is omitted
  sorry

end evaluate_f_l117_117866


namespace number_of_solutions_l117_117049

open Real

theorem number_of_solutions :
  ∀ x : ℝ, (0 < x ∧ x < 3 * π) → (3 * cos x ^ 2 + 2 * sin x ^ 2 = 2) → 
  ∃ (L : Finset ℝ), L.card = 3 ∧ ∀ y ∈ L, 0 < y ∧ y < 3 * π ∧ 3 * cos y ^ 2 + 2 * sin y ^ 2 = 2 :=
by 
  sorry

end number_of_solutions_l117_117049


namespace find_x_if_opposites_l117_117678

theorem find_x_if_opposites (x : ℝ) (h : 2 * (x - 3) = - 4 * (1 - x)) : x = -1 := 
by
  sorry

end find_x_if_opposites_l117_117678


namespace max_value_g_l117_117777

def g : ℕ → ℕ
| n => if n < 7 then n + 8 else g (n - 3)

theorem max_value_g : ∃ m, (∀ n, g n ≤ m) ∧ m = 14 := by
  sorry

end max_value_g_l117_117777


namespace ratio_black_white_extended_pattern_l117_117103

def originalBlackTiles : ℕ := 8
def originalWhiteTiles : ℕ := 17
def originalSquareSide : ℕ := 5
def extendedSquareSide : ℕ := 7
def newBlackTiles : ℕ := (extendedSquareSide * extendedSquareSide) - (originalSquareSide * originalSquareSide)
def totalBlackTiles : ℕ := originalBlackTiles + newBlackTiles
def totalWhiteTiles : ℕ := originalWhiteTiles

theorem ratio_black_white_extended_pattern : totalBlackTiles / totalWhiteTiles = 32 / 17 := sorry

end ratio_black_white_extended_pattern_l117_117103


namespace factorizable_trinomial_l117_117659

theorem factorizable_trinomial (k : ℤ) : (∃ a b : ℤ, a + b = k ∧ a * b = 5) ↔ (k = 6 ∨ k = -6) :=
by
  sorry

end factorizable_trinomial_l117_117659


namespace determine_n_l117_117767

theorem determine_n (n : ℕ) (h : 3^n = 3^2 * 9^4 * 81^3) : n = 22 := 
by
  sorry

end determine_n_l117_117767


namespace number_description_l117_117884

theorem number_description :
  4 * 10000 + 3 * 1000 + 7 * 100 + 5 * 10 + 2 + 8 / 10 + 4 / 100 = 43752.84 :=
by
  sorry

end number_description_l117_117884


namespace commutativity_associativity_l117_117559

variables {α : Type*} (op : α → α → α)

-- Define conditions as hypotheses
axiom cond1 : ∀ a b c : α, op a (op b c) = op b (op c a)
axiom cond2 : ∀ a b c : α, op a b = op a c → b = c
axiom cond3 : ∀ a b c : α, op a c = op b c → a = b

-- Commutativity statement
theorem commutativity (a b : α) : op a b = op b a := sorry

-- Associativity statement
theorem associativity (a b c : α) : op (op a b) c = op a (op b c) := sorry

end commutativity_associativity_l117_117559


namespace additional_money_needed_l117_117943

/-- Mrs. Smith needs to calculate the additional money required after a discount -/
theorem additional_money_needed
  (initial_amount : ℝ) (ratio_more : ℝ) (discount_rate : ℝ) (final_amount_needed : ℝ) (additional_needed : ℝ)
  (h_initial : initial_amount = 500)
  (h_ratio : ratio_more = 2/5)
  (h_discount : discount_rate = 15/100)
  (h_total_needed : final_amount_needed = initial_amount * (1 + ratio_more) * (1 - discount_rate))
  (h_additional : additional_needed = final_amount_needed - initial_amount) :
  additional_needed = 95 :=
by 
  sorry

end additional_money_needed_l117_117943


namespace inequality_holds_for_a_l117_117043

theorem inequality_holds_for_a (a : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < -1 → (x + 1)^2 < Real.logb a (|x|)) ↔ 1 < a ∧ a ≤ 2 := by
  sorry

end inequality_holds_for_a_l117_117043


namespace find_n_l117_117643

theorem find_n :
  ∃ n : ℕ, 50 ≤ n ∧ n ≤ 150 ∧
          n % 7 = 0 ∧
          n % 9 = 3 ∧
          n % 6 = 3 ∧
          n = 75 :=
by
  sorry

end find_n_l117_117643


namespace problem_l117_117906

theorem problem (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a + b = 11 :=
by {
  sorry
}

end problem_l117_117906


namespace find_f_of_neg3_l117_117424

noncomputable def f : ℚ → ℚ := sorry 

theorem find_f_of_neg3 (h : ∀ (x : ℚ) (hx : x ≠ 0), 5 * f (x⁻¹) + 3 * (f x) * x⁻¹ = 2 * x^2) :
  f (-3) = -891 / 22 :=
sorry

end find_f_of_neg3_l117_117424


namespace isabella_hair_length_l117_117673

-- Define the conditions and the question in Lean
def current_length : ℕ := 9
def length_cut_off : ℕ := 9

-- Main theorem statement
theorem isabella_hair_length 
  (current_length : ℕ) 
  (length_cut_off : ℕ) 
  (H1 : current_length = 9) 
  (H2 : length_cut_off = 9) : 
  current_length + length_cut_off = 18 :=
  sorry

end isabella_hair_length_l117_117673


namespace time_for_six_visits_l117_117101

noncomputable def time_to_go_n_times (total_time : ℕ) (total_visits : ℕ) (n_visits : ℕ) : ℕ :=
  (total_time / total_visits) * n_visits

theorem time_for_six_visits (h : time_to_go_n_times 20 8 6 = 15) : time_to_go_n_times 20 8 6 = 15 :=
by
  exact h

end time_for_six_visits_l117_117101


namespace tooth_extraction_cost_l117_117301

noncomputable def cleaning_cost : ℕ := 70
noncomputable def filling_cost : ℕ := 120
noncomputable def root_canal_cost : ℕ := 400
noncomputable def crown_cost : ℕ := 600
noncomputable def bridge_cost : ℕ := 800

noncomputable def crown_discount : ℕ := (crown_cost * 20) / 100
noncomputable def bridge_discount : ℕ := (bridge_cost * 10) / 100

noncomputable def total_cost_without_extraction : ℕ := 
  cleaning_cost + 
  3 * filling_cost + 
  root_canal_cost + 
  (crown_cost - crown_discount) + 
  (bridge_cost - bridge_discount)

noncomputable def root_canal_and_one_filling : ℕ := 
  root_canal_cost + filling_cost

noncomputable def dentist_bill : ℕ := 
  11 * root_canal_and_one_filling

theorem tooth_extraction_cost : 
  dentist_bill - total_cost_without_extraction = 3690 :=
by
  -- The proof would go here
  sorry

end tooth_extraction_cost_l117_117301


namespace tickets_left_l117_117478

-- Define the number of tickets won by Dave
def tickets_won : ℕ := 14

-- Define the number of tickets lost by Dave
def tickets_lost : ℕ := 2

-- Define the number of tickets used to buy toys
def tickets_used : ℕ := 10

-- The theorem to prove that the number of tickets left is 2
theorem tickets_left : tickets_won - tickets_lost - tickets_used = 2 := by
  -- Initial computation of tickets left after losing some
  let tickets_after_lost := tickets_won - tickets_lost
  -- Computation of tickets left after using some
  let tickets_after_used := tickets_after_lost - tickets_used
  show tickets_after_used = 2
  sorry

end tickets_left_l117_117478


namespace value_of_f_is_negative_l117_117295

theorem value_of_f_is_negative {a b c : ℝ} (h1 : a + b < 0) (h2 : b + c < 0) (h3 : c + a < 0) :
  2 * a ^ 3 + 4 * a + 2 * b ^ 3 + 4 * b + 2 * c ^ 3 + 4 * c < 0 := by
sorry

end value_of_f_is_negative_l117_117295


namespace eggs_per_meal_l117_117719

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l117_117719


namespace fill_tanker_time_l117_117313

/-- Given that pipe A can fill the tanker in 60 minutes and pipe B can fill the tanker in 40 minutes,
    prove that the time T to fill the tanker if pipe B is used for half the time and both pipes 
    A and B are used together for the other half is equal to 30 minutes. -/
theorem fill_tanker_time (T : ℝ) (hA : ∀ (a : ℝ), a = 1/60) (hB : ∀ (b : ℝ), b = 1/40) :
  (T / 2) * (1 / 40) + (T / 2) * (1 / 24) = 1 → T = 30 :=
by
  sorry

end fill_tanker_time_l117_117313


namespace zahra_kimmie_money_ratio_l117_117960

theorem zahra_kimmie_money_ratio (KimmieMoney ZahraMoney : ℕ) (hKimmie : KimmieMoney = 450)
  (totalSavings : ℕ) (hSaving : totalSavings = 375)
  (h : KimmieMoney / 2 + ZahraMoney / 2 = totalSavings) :
  ZahraMoney / KimmieMoney = 2 / 3 :=
by
  -- Conditions to be used in the proof, but skipped for now
  sorry

end zahra_kimmie_money_ratio_l117_117960


namespace range_of_m_l117_117162

noncomputable def f (x m : ℝ) : ℝ := -x^2 - 4 * m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → f x1 m ≥ f x2 m) ↔ m ≥ -1 := 
sorry

end range_of_m_l117_117162


namespace below_zero_notation_l117_117202

def celsius_above (x : ℤ) : String := "+" ++ toString x ++ "°C"
def celsius_below (x : ℤ) : String := "-" ++ toString x ++ "°C"

theorem below_zero_notation (h₁ : celsius_above 5 = "+5°C")
  (h₂ : ∀ x : ℤ, x > 0 → celsius_above x = "+" ++ toString x ++ "°C")
  (h₃ : ∀ x : ℤ, x > 0 → celsius_below x = "-" ++ toString x ++ "°C") :
  celsius_below 3 = "-3°C" :=
sorry

end below_zero_notation_l117_117202


namespace total_revenue_from_selling_snakes_l117_117363

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end total_revenue_from_selling_snakes_l117_117363


namespace calc_3a2008_minus_5b2008_l117_117199

theorem calc_3a2008_minus_5b2008 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : 3 * a ^ 2008 - 5 * b ^ 2008 = -5 :=
by
  sorry

end calc_3a2008_minus_5b2008_l117_117199


namespace value_of_5y_l117_117808

-- Define positive integers
variables {x y z : ℕ}

-- Define the conditions
def conditions (x y z : ℕ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (5 * y = 6 * z) ∧ (x + y + z = 26)

-- The theorem statement
theorem value_of_5y (x y z : ℕ) (h : conditions x y z) : 5 * y = 30 :=
by
  -- proof skipped (proof goes here)
  sorry

end value_of_5y_l117_117808


namespace pure_imaginary_real_zero_l117_117864

theorem pure_imaginary_real_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) (h : a * i = 0 + a * i) : a = 0 := by
  sorry

end pure_imaginary_real_zero_l117_117864


namespace part_a_part_b_l117_117365

section
-- Definitions based on the conditions
variable (n : ℕ)  -- Variable n representing the number of cities

-- Given a condition function T_n that returns an integer (number of ways to build roads)
def T_n (n : ℕ) : ℕ := sorry  -- Definition placeholder for T_n function

-- Part (a): For all odd n, T_n(n) is divisible by n
theorem part_a (hn : n % 2 = 1) : T_n n % n = 0 := sorry

-- Part (b): For all even n, T_n(n) is divisible by n / 2
theorem part_b (hn : n % 2 = 0) : T_n n % (n / 2) = 0 := sorry

end

end part_a_part_b_l117_117365


namespace trigonometric_identity_proof_l117_117609

open Real

theorem trigonometric_identity_proof (x y : ℝ) (hx : sin x / sin y = 4) (hy : cos x / cos y = 1 / 3) :
  (sin (2 * x) / sin (2 * y)) + (cos (2 * x) / cos (2 * y)) = 169 / 381 :=
by
  sorry

end trigonometric_identity_proof_l117_117609


namespace general_term_formula_l117_117844

-- Define the problem parameters
variables (a : ℤ)

-- Definitions based on the conditions
def first_term : ℤ := a - 1
def second_term : ℤ := a + 1
def third_term : ℤ := 2 * a + 3

-- Define the theorem to prove the general term formula
theorem general_term_formula :
  2 * (first_term a + 1) = first_term a + third_term a → a = 0 →
  ∀ n : ℕ, a_n = 2 * n - 3 := 
by
  intro h1 h2
  sorry

end general_term_formula_l117_117844


namespace find_solution_to_inequality_l117_117015

open Set

noncomputable def inequality_solution : Set ℝ := {x : ℝ | 0.5 ≤ x ∧ x < 2 ∨ 3 ≤ x}

theorem find_solution_to_inequality :
  {x : ℝ | (x^2 + 1) / (x - 2) + (2 * x + 3) / (2 * x - 1) ≥ 4} = inequality_solution := 
sorry

end find_solution_to_inequality_l117_117015


namespace simplify_expression_l117_117329

theorem simplify_expression :
  ( ∀ (a b c : ℕ), c > 0 ∧ (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) →
  (a - b * Real.sqrt c = (28 - 16 * Real.sqrt 3) * 2 ^ (-2 - Real.sqrt 5))) :=
sorry

end simplify_expression_l117_117329


namespace total_gold_is_100_l117_117297

-- Definitions based on conditions
def GregsGold : ℕ := 20
def KatiesGold : ℕ := GregsGold * 4
def TotalGold : ℕ := GregsGold + KatiesGold

-- Theorem to prove
theorem total_gold_is_100 : TotalGold = 100 := by
  sorry

end total_gold_is_100_l117_117297


namespace solution_of_fractional_equation_l117_117651

theorem solution_of_fractional_equation :
  (∃ x, x ≠ 3 ∧ (x / (x - 3) - 2 = (m - 1) / (x - 3))) → m = 4 := by
  sorry

end solution_of_fractional_equation_l117_117651


namespace range_of_m_for_circle_l117_117963

theorem range_of_m_for_circle (m : ℝ) :
  (∃ x y, x^2 + y^2 - 4 * x - 2 * y + m = 0) → m < 5 :=
by
  sorry

end range_of_m_for_circle_l117_117963


namespace min_value_x_plus_y_l117_117396

open Real

noncomputable def xy_plus_x_minus_y_minus_10_eq_zero (x y: ℝ) := x * y + x - y - 10 = 0

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : xy_plus_x_minus_y_minus_10_eq_zero x y) : 
  x + y ≥ 6 :=
by
  sorry

end min_value_x_plus_y_l117_117396


namespace gcd_polynomial_example_l117_117058

theorem gcd_polynomial_example (b : ℕ) (h : ∃ k : ℕ, b = 2 * 7784 * k) : 
  gcd (5 * b ^ 2 + 68 * b + 143) (3 * b + 14) = 25 :=
by 
  sorry

end gcd_polynomial_example_l117_117058


namespace cuboid_surface_area_500_l117_117020

def surface_area (w l h : ℝ) : ℝ :=
  2 * l * w + 2 * l * h + 2 * w * h

theorem cuboid_surface_area_500 :
  ∀ (w l h : ℝ), w = 4 → l = w + 6 → h = l + 5 →
  surface_area w l h = 500 :=
by
  intros w l h hw hl hh
  unfold surface_area
  rw [hw, hl, hh]
  norm_num
  sorry

end cuboid_surface_area_500_l117_117020


namespace teresa_ahmad_equation_l117_117076

theorem teresa_ahmad_equation (b c : ℝ) :
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 7 ∨ x = 1) →
  (∀ x : ℝ, x^2 + b * x + c = 0 ↔ x = 7 ∨ x = 1) →
  (b, c) = (-8, 7) :=
by
  intro h1 h2
  sorry

end teresa_ahmad_equation_l117_117076


namespace work_finished_earlier_due_to_additional_men_l117_117485

-- Define the conditions as given facts in Lean
def original_men := 10
def original_days := 12
def additional_men := 10

-- State the theorem to be proved
theorem work_finished_earlier_due_to_additional_men :
  let total_men := original_men + additional_men
  let original_work := original_men * original_days
  let days_earlier := original_days - x
  original_work = total_men * days_earlier → x = 6 :=
by
  sorry

end work_finished_earlier_due_to_additional_men_l117_117485


namespace households_with_at_least_one_appliance_l117_117548

theorem households_with_at_least_one_appliance (total: ℕ) (color_tvs: ℕ) (refrigerators: ℕ) (both: ℕ) :
  total = 100 → color_tvs = 65 → refrigerators = 84 → both = 53 →
  (color_tvs + refrigerators - both) = 96 :=
by
  intros
  sorry

end households_with_at_least_one_appliance_l117_117548


namespace distinct_roots_polynomial_l117_117951

theorem distinct_roots_polynomial (a b : ℂ) (h₁ : a ≠ b) (h₂: a^3 + 3*a^2 + a + 1 = 0) (h₃: b^3 + 3*b^2 + b + 1 = 0) :
  a^2 * b + a * b^2 + 3 * a * b = 1 :=
sorry

end distinct_roots_polynomial_l117_117951


namespace sasha_kolya_distance_l117_117949

theorem sasha_kolya_distance
  (v_S v_L v_K : ℝ) 
  (h1 : Lesha_dist = 100 - 10) 
  (h2 : Kolya_dist = 100 - 10) 
  (h3 : v_L = (90 / 100) * v_S) 
  (h4 : v_K = (90 / 100) * v_L) 
  : v_S * (100/v_S - 10/v_S) = 19 :=
by
  sorry

end sasha_kolya_distance_l117_117949


namespace multiply_seven_l117_117588

variable (x : ℕ)

theorem multiply_seven (h : 8 * x = 64) : 7 * x = 56 := by
  sorry


end multiply_seven_l117_117588


namespace tan_diff_sin_double_l117_117041

theorem tan_diff (α : ℝ) (h : Real.tan α = 2) : 
  Real.tan (α - Real.pi / 4) = 1 / 3 := 
by 
  sorry

theorem sin_double (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end tan_diff_sin_double_l117_117041


namespace last_part_length_l117_117840

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end last_part_length_l117_117840


namespace sum_distinct_vars_eq_1716_l117_117234

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l117_117234


namespace exists_line_intersecting_circle_and_passing_origin_l117_117693

theorem exists_line_intersecting_circle_and_passing_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -4) ∧ 
  ∃ (x y : ℝ), 
    ((x - 1) ^ 2 + (y + 2) ^ 2 = 9) ∧ 
    ((x - y + m = 0) ∧ 
     ∃ (x' y' : ℝ),
      ((x' - 1) ^ 2 + (y' + 2) ^ 2 = 9) ∧ 
      ((x' - y' + m = 0) ∧ ((x + x') / 2 = 0 ∧ (y + y') / 2 = 0))) :=
by 
  sorry

end exists_line_intersecting_circle_and_passing_origin_l117_117693


namespace find_numbers_l117_117380

theorem find_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : a + b = 8) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
sorry

end find_numbers_l117_117380


namespace sandy_friday_hours_l117_117446

-- Define the conditions
def hourly_rate := 15
def saturday_hours := 6
def sunday_hours := 14
def total_earnings := 450

-- Define the proof problem
theorem sandy_friday_hours (F : ℝ) (h1 : F * hourly_rate + saturday_hours * hourly_rate + sunday_hours * hourly_rate = total_earnings) : F = 10 :=
sorry

end sandy_friday_hours_l117_117446


namespace sum_of_n_for_perfect_square_l117_117165

theorem sum_of_n_for_perfect_square (n : ℕ) (Sn : ℕ) 
  (hSn : Sn = n^2 + 20 * n + 12) 
  (hn : n > 0) :
  ∃ k : ℕ, k^2 = Sn → (sum_of_possible_n = 16) :=
by
  sorry

end sum_of_n_for_perfect_square_l117_117165


namespace sum_of_roots_zero_l117_117341

theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : ∀ x, x^2 + p * x + q = 0) : p + q = 0 := 
by {
  sorry 
}

end sum_of_roots_zero_l117_117341


namespace fraction_lost_l117_117878

-- Definitions of the given conditions
def initial_pencils : ℕ := 30
def lost_pencils_initially : ℕ := 6
def current_pencils : ℕ := 16

-- Statement of the proof problem
theorem fraction_lost (initial_pencils lost_pencils_initially current_pencils : ℕ) :
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  (lost_remaining_pencils : ℚ) / remaining_pencils = 1 / 3 :=
by
  let remaining_pencils := initial_pencils - lost_pencils_initially
  let lost_remaining_pencils := remaining_pencils - current_pencils
  sorry

end fraction_lost_l117_117878


namespace exists_ab_negated_l117_117612

theorem exists_ab_negated :
  ¬ (∀ a b : ℝ, (a + b = 0 → a^2 + b^2 = 0)) ↔ 
  ∃ a b : ℝ, (a + b = 0 ∧ a^2 + b^2 ≠ 0) :=
by
  sorry

end exists_ab_negated_l117_117612


namespace max_gcd_seq_l117_117185

theorem max_gcd_seq (a : ℕ → ℕ) (d : ℕ → ℕ) :
  (∀ n : ℕ, a n = 121 + n^2) →
  (∀ n : ℕ, d n = Nat.gcd (a n) (a (n + 1))) →
  ∃ m : ℕ, ∀ n : ℕ, d n ≤ d m ∧ d m = 99 :=
by
  sorry

end max_gcd_seq_l117_117185


namespace distance_james_rode_l117_117181

def speed : ℝ := 80.0
def time : ℝ := 16.0
def distance : ℝ := speed * time

theorem distance_james_rode :
  distance = 1280.0 :=
by
  -- to show the theorem is sane
  sorry

end distance_james_rode_l117_117181


namespace geometric_sequence_ratio_l117_117525

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (S3 : ℝ) 
  (h1 : a 1 = 1) (h2 : S3 = 3 / 4) 
  (h3 : S3 = a 1 + a 1 * q + a 1 * q^2) :
  q = -1 / 2 := 
by
  sorry

end geometric_sequence_ratio_l117_117525


namespace Bianca_pictures_distribution_l117_117569

theorem Bianca_pictures_distribution 
(pictures_total : ℕ) 
(pictures_in_one_album : ℕ) 
(albums_remaining : ℕ) 
(h1 : pictures_total = 33)
(h2 : pictures_in_one_album = 27)
(h3 : albums_remaining = 3)
: (pictures_total - pictures_in_one_album) / albums_remaining = 2 := 
by 
  sorry

end Bianca_pictures_distribution_l117_117569


namespace expand_binomials_l117_117970

variable (x y : ℝ)

theorem expand_binomials: (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 :=
by sorry

end expand_binomials_l117_117970


namespace chord_length_of_circle_l117_117994

theorem chord_length_of_circle (x y : ℝ) :
  (x^2 + y^2 - 4 * x - 4 * y - 1 = 0) ∧ (y = x + 2) → 
  2 * Real.sqrt 7 = 2 * Real.sqrt 7 :=
by sorry

end chord_length_of_circle_l117_117994


namespace arithmetic_sequence_a1_a7_a3_a5_l117_117120

noncomputable def arithmetic_sequence_property (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_a1_a7_a3_a5 (a : ℕ → ℝ) (h_arith : arithmetic_sequence_property a)
  (h_cond : a 1 + a 7 = 10) : a 3 + a 5 = 10 :=
by
  sorry

end arithmetic_sequence_a1_a7_a3_a5_l117_117120


namespace value_of_a_l117_117543

theorem value_of_a (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x - 1 = 0) ∧ (∀ x y : ℝ, a * x^2 - 2 * x - 1 = 0 ∧ a * y^2 - 2 * y - 1 = 0 → x = y) ↔ (a = 0 ∨ a = -1) :=
by
  sorry

end value_of_a_l117_117543


namespace correct_equation_l117_117369

variable (x : ℕ)

def three_people_per_cart_and_two_empty_carts (x : ℕ) :=
  x / 3 + 2

def two_people_per_cart_and_nine_walking (x : ℕ) :=
  (x - 9) / 2

theorem correct_equation (x : ℕ) :
  three_people_per_cart_and_two_empty_carts x = two_people_per_cart_and_nine_walking x :=
by
  sorry

end correct_equation_l117_117369


namespace max_value_ab_bc_cd_l117_117235

theorem max_value_ab_bc_cd (a b c d : ℝ) (h1 : 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd ≤ 2500 :=
by
  sorry

end max_value_ab_bc_cd_l117_117235


namespace option_a_option_b_l117_117815

theorem option_a (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  -- Proof goes here
  sorry

theorem option_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : a * b ≤ (a + b)^2 / 4 :=
by
  -- Proof goes here
  sorry

end option_a_option_b_l117_117815


namespace number_of_males_in_village_l117_117242

-- Given the total population is 800 and it is divided into four equal groups.
def total_population : ℕ := 800
def num_groups : ℕ := 4

-- Proof statement
theorem number_of_males_in_village : (total_population / num_groups) = 200 := 
by sorry

end number_of_males_in_village_l117_117242


namespace find_angle_x_l117_117782

noncomputable def angle_x (angle_ABC angle_ACB angle_CDE : ℝ) : ℝ :=
  let angle_BAC := 180 - angle_ABC - angle_ACB
  let angle_ADE := 180 - angle_CDE
  let angle_EAD := angle_BAC
  let angle_AED := 180 - angle_ADE - angle_EAD
  180 - angle_AED

theorem find_angle_x (angle_ABC angle_ACB angle_CDE : ℝ) :
  angle_ABC = 70 → angle_ACB = 90 → angle_CDE = 42 → angle_x angle_ABC angle_ACB angle_CDE = 158 :=
by
  intros hABC hACB hCDE
  simp [angle_x, hABC, hACB, hCDE]
  sorry

end find_angle_x_l117_117782


namespace football_team_lineup_ways_l117_117233

theorem football_team_lineup_ways :
  let members := 12
  let offensive_lineman_options := 4
  let remaining_after_linemen := members - offensive_lineman_options
  let quarterback_options := remaining_after_linemen
  let remaining_after_qb := remaining_after_linemen - 1
  let wide_receiver_options := remaining_after_qb
  let remaining_after_wr := remaining_after_qb - 1
  let tight_end_options := remaining_after_wr
  let lineup_ways := offensive_lineman_options * quarterback_options * wide_receiver_options * tight_end_options
  lineup_ways = 3960 :=
by
  sorry

end football_team_lineup_ways_l117_117233


namespace athlete_total_heartbeats_l117_117186

theorem athlete_total_heartbeats (h : ℕ) (p : ℕ) (d : ℕ) (r : ℕ) : (h = 150) ∧ (p = 6) ∧ (d = 30) ∧ (r = 15) → (p * d + r) * h = 29250 :=
by
  sorry

end athlete_total_heartbeats_l117_117186


namespace total_tickets_sold_l117_117747

-- Definitions of the conditions as given in the problem
def price_adult : ℕ := 7
def price_child : ℕ := 4
def total_revenue : ℕ := 5100
def child_tickets_sold : ℕ := 400

-- The main statement (theorem) to prove
theorem total_tickets_sold:
  ∃ (A C : ℕ), C = child_tickets_sold ∧ price_adult * A + price_child * C = total_revenue ∧ (A + C = 900) :=
by
  sorry

end total_tickets_sold_l117_117747


namespace probability_none_solve_l117_117713

theorem probability_none_solve (a b c : ℕ) (ha : 0 < a ∧ a < 10)
                               (hb : 0 < b ∧ b < 10)
                               (hc : 0 < c ∧ c < 10)
                               (P_A : ℚ := 1 / a)
                               (P_B : ℚ := 1 / b)
                               (P_C : ℚ := 1 / c)
                               (H : (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15) :
                               -- Conclusion: The probability that none of them solve the problem is 8/15
                               (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15 :=
sorry

end probability_none_solve_l117_117713


namespace infinite_solutions_xyz_l117_117536

theorem infinite_solutions_xyz : ∀ k : ℕ, 
  (∃ n : ℕ, n > k ∧ ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008) →
  ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008 := 
sorry

end infinite_solutions_xyz_l117_117536


namespace watch_hands_angle_120_l117_117971

theorem watch_hands_angle_120 (n : ℝ) (h₁ : 0 ≤ n ∧ n ≤ 60) 
    (h₂ : abs ((210 + n / 2) - 6 * n) = 120) : n = 43.64 := sorry

end watch_hands_angle_120_l117_117971


namespace g_neg501_l117_117318

noncomputable def g : ℝ → ℝ := sorry

axiom g_eq (x y : ℝ) : g (x * y) + 2 * x = x * g y + g x

axiom g_neg1 : g (-1) = 7

theorem g_neg501 : g (-501) = 507 :=
by
  sorry

end g_neg501_l117_117318


namespace crystal_barrette_sets_l117_117385

-- Definitional and situational context
def cost_of_barrette : ℕ := 3
def cost_of_comb : ℕ := 1
def kristine_total_cost : ℕ := 4
def total_spent : ℕ := 14

-- The Lean 4 theorem statement to prove that Crystal bought 3 sets of barrettes
theorem crystal_barrette_sets (x : ℕ) 
  (kristine_cost : kristine_total_cost = cost_of_barrette + cost_of_comb + 1)
  (total_cost_eq : kristine_total_cost + (x * cost_of_barrette + cost_of_comb) = total_spent) 
  : x = 3 := 
sorry

end crystal_barrette_sets_l117_117385


namespace no_two_right_angles_in_triangle_l117_117381

theorem no_two_right_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90) (h3 : B = 90): false :=
by
  -- we assume A = 90 and B = 90,
  -- then A + B + C > 180, which contradicts h1,
  sorry
  
example : (3 = 3) := by sorry  -- Given the context of the multiple-choice problem.

end no_two_right_angles_in_triangle_l117_117381


namespace chess_tournament_l117_117134

theorem chess_tournament (n games : ℕ) 
  (h_games : games = 81)
  (h_equation : (n - 2) * (n - 3) = 156) :
  n = 15 :=
sorry

end chess_tournament_l117_117134


namespace bart_trees_needed_l117_117600

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end bart_trees_needed_l117_117600


namespace speed_ratio_of_runners_l117_117529

theorem speed_ratio_of_runners (v_A v_B : ℝ) (c : ℝ)
  (h1 : 0 < v_A ∧ 0 < v_B) -- They run at constant, but different speeds
  (h2 : (v_B / v_A) = (2 / 3)) -- Distance relationship from meeting points
  : v_B / v_A = 2 :=
by
  sorry

end speed_ratio_of_runners_l117_117529


namespace abe_family_total_yen_l117_117756

theorem abe_family_total_yen (yen_checking : ℕ) (yen_savings : ℕ) (h₁ : yen_checking = 6359) (h₂ : yen_savings = 3485) : yen_checking + yen_savings = 9844 :=
by
  sorry

end abe_family_total_yen_l117_117756


namespace max_PA_PB_l117_117617

noncomputable def max_distance (PA PB : ℝ) : ℝ :=
  PA + PB

theorem max_PA_PB {A B : ℝ × ℝ} (m : ℝ) :
  A = (0, 0) ∧
  B = (1, 3) ∧
  dist A B = 10 →
  max_distance (dist A B) (dist (1, 3) B) = 2 * Real.sqrt 5 :=
by
  sorry

end max_PA_PB_l117_117617


namespace quadratic_function_range_l117_117805

theorem quadratic_function_range (a b c : ℝ) (x y : ℝ) :
  (∀ x, x = -4 → y = a * (-4)^2 + b * (-4) + c → y = 3) ∧
  (∀ x, x = -3 → y = a * (-3)^2 + b * (-3) + c → y = -2) ∧
  (∀ x, x = -2 → y = a * (-2)^2 + b * (-2) + c → y = -5) ∧
  (∀ x, x = -1 → y = a * (-1)^2 + b * (-1) + c → y = -6) ∧
  (∀ x, x = 0 → y = a * 0^2 + b * 0 + c → y = -5) →
  (∀ x, x < -2 → y > -5) :=
sorry

end quadratic_function_range_l117_117805


namespace largest_possible_number_of_neither_l117_117739

theorem largest_possible_number_of_neither
  (writers : ℕ)
  (editors : ℕ)
  (attendees : ℕ)
  (x : ℕ)
  (N : ℕ)
  (h_writers : writers = 45)
  (h_editors_gt : editors > 38)
  (h_attendees : attendees = 90)
  (h_both : N = 2 * x)
  (h_equation : writers + editors - x + N = attendees) :
  N = 12 :=
by
  sorry

end largest_possible_number_of_neither_l117_117739


namespace lindas_savings_l117_117935

theorem lindas_savings :
  ∃ S : ℝ, (3 / 4 * S) + 150 = S ∧ (S - 150) = 3 / 4 * S := 
sorry

end lindas_savings_l117_117935


namespace racetrack_circumference_diff_l117_117308

theorem racetrack_circumference_diff (d_inner d_outer width : ℝ) 
(h1 : d_inner = 55) (h2 : width = 15) (h3 : d_outer = d_inner + 2 * width) : 
  (π * d_outer - π * d_inner) = 30 * π :=
by
  sorry

end racetrack_circumference_diff_l117_117308


namespace paint_rate_5_l117_117784
noncomputable def rate_per_sq_meter (L : ℝ) (total_cost : ℝ) (B : ℝ) : ℝ :=
  let Area := L * B
  total_cost / Area

theorem paint_rate_5 : 
  ∀ (L B total_cost rate : ℝ),
    L = 19.595917942265423 →
    total_cost = 640 →
    L = 3 * B →
    rate = rate_per_sq_meter L total_cost B →
    rate = 5 :=
by
  intros L B total_cost rate hL hC hR hRate
  -- Proof goes here
  sorry

end paint_rate_5_l117_117784


namespace fraction_product_l117_117565

theorem fraction_product : 
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := 
by
  sorry

end fraction_product_l117_117565


namespace intersection_of_M_and_N_l117_117817

def set_M : Set ℝ := {x | -1 < x}
def set_N : Set ℝ := {x | x * (x + 2) ≤ 0}

theorem intersection_of_M_and_N : (set_M ∩ set_N) = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_of_M_and_N_l117_117817


namespace equal_costs_at_60_minutes_l117_117447

-- Define the base rates and the per minute rates for each company
def base_rate_united : ℝ := 9.00
def rate_per_minute_united : ℝ := 0.25
def base_rate_atlantic : ℝ := 12.00
def rate_per_minute_atlantic : ℝ := 0.20

-- Define the total cost functions
def cost_united (m : ℝ) : ℝ := base_rate_united + rate_per_minute_united * m
def cost_atlantic (m : ℝ) : ℝ := base_rate_atlantic + rate_per_minute_atlantic * m

-- State the theorem to be proved
theorem equal_costs_at_60_minutes : 
  ∃ (m : ℝ), cost_united m = cost_atlantic m ∧ m = 60 :=
by
  -- Pending proof
  sorry

end equal_costs_at_60_minutes_l117_117447


namespace probability_gather_info_both_workshops_l117_117374

theorem probability_gather_info_both_workshops :
  ∃ (p : ℚ), p = 56 / 62 :=
by
  sorry

end probability_gather_info_both_workshops_l117_117374


namespace buffy_whiskers_l117_117974

/-- Definition of whisker counts for the cats --/
def whiskers_of_juniper : ℕ := 12
def whiskers_of_puffy : ℕ := 3 * whiskers_of_juniper
def whiskers_of_scruffy : ℕ := 2 * whiskers_of_puffy
def whiskers_of_buffy : ℕ := (whiskers_of_juniper + whiskers_of_puffy + whiskers_of_scruffy) / 3

/-- Proof statement for the number of whiskers of Buffy --/
theorem buffy_whiskers : whiskers_of_buffy = 40 := 
by
  -- Proof is omitted
  sorry

end buffy_whiskers_l117_117974


namespace triangle_area_ratio_l117_117841

theorem triangle_area_ratio (x y : ℝ) (n m : ℕ) (hn : n > 0) (hm : m > 0) :
  let A_area := (1/2) * (y/n) * (x/2)
  let B_area := (1/2) * (x/m) * (y/2)
  A_area / B_area = m / n := by
  sorry

end triangle_area_ratio_l117_117841


namespace cube_distance_l117_117944

-- The Lean 4 statement
theorem cube_distance (side_length : ℝ) (h1 h2 h3 : ℝ) (r s t : ℕ) 
  (h1_eq : h1 = 18) (h2_eq : h2 = 20) (h3_eq : h3 = 22) (side_length_eq : side_length = 15) :
  r = 57 ∧ s = 597 ∧ t = 3 ∧ r + s + t = 657 :=
by
  sorry

end cube_distance_l117_117944


namespace perpendicular_vectors_m_eq_half_l117_117220

theorem perpendicular_vectors_m_eq_half (m : ℝ) (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (-1, m)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : m = 1 / 2 :=
sorry

end perpendicular_vectors_m_eq_half_l117_117220


namespace number_of_ordered_quadruples_l117_117753

theorem number_of_ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h_sum : x1 + x2 + x3 + x4 = 100) : 
  ∃ n : ℕ, n = 156849 := 
by 
  sorry

end number_of_ordered_quadruples_l117_117753


namespace shadow_area_greatest_integer_l117_117148

theorem shadow_area_greatest_integer (x : ℝ)
  (h1 : ∀ (a : ℝ), a = 1)
  (h2 : ∀ (b : ℝ), b = 48)
  (h3 : ∀ (c: ℝ), x = 1 / 6):
  ⌊1000 * x⌋ = 166 := 
by sorry

end shadow_area_greatest_integer_l117_117148


namespace fraction_problem_l117_117638

theorem fraction_problem (N D : ℚ) (h1 : 1.30 * N / (0.85 * D) = 25 / 21) : 
  N / D = 425 / 546 :=
sorry

end fraction_problem_l117_117638


namespace coins_donated_l117_117255

theorem coins_donated (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (coins_left : ℕ) : 
  pennies = 42 ∧ nickels = 36 ∧ dimes = 15 ∧ coins_left = 27 → (pennies + nickels + dimes - coins_left) = 66 :=
by
  intros h
  sorry

end coins_donated_l117_117255


namespace rope_costs_purchasing_plans_minimum_cost_l117_117732

theorem rope_costs (x y m : ℕ) :
  (10 * x + 5 * y = 175) →
  (15 * x + 10 * y = 300) →
  x = 10 ∧ y = 15 :=
sorry

theorem purchasing_plans (m : ℕ) :
  (10 * 10 + 15 * 15 = 300) →
  23 ≤ m ∧ m ≤ 25 :=
sorry

theorem minimum_cost (m : ℕ) :
  (23 ≤ m ∧ m ≤ 25) →
  m = 25 →
  10 * m + 15 * (45 - m) = 550 :=
sorry

end rope_costs_purchasing_plans_minimum_cost_l117_117732


namespace fully_filled_boxes_l117_117682

theorem fully_filled_boxes (total_cards : ℕ) (cards_per_box : ℕ) (h1 : total_cards = 94) (h2 : cards_per_box = 8) : total_cards / cards_per_box = 11 :=
by {
  sorry
}

end fully_filled_boxes_l117_117682


namespace correct_average_l117_117924

theorem correct_average (S' : ℝ) (a a' b b' c c' : ℝ) (n : ℕ) 
  (incorrect_avg : S' / n = 22) 
  (a_eq : a = 52) (a'_eq : a' = 32)
  (b_eq : b = 47) (b'_eq : b' = 27) 
  (c_eq : c = 68) (c'_eq : c' = 45)
  (n_eq : n = 12) 
  : ((S' - (a' + b' + c') + (a + b + c)) / 12 = 27.25) := 
by
  sorry

end correct_average_l117_117924


namespace junk_mail_per_red_or_white_house_l117_117908

noncomputable def pieces_per_house (total_pieces : ℕ) (total_houses : ℕ) : ℕ := 
  total_pieces / total_houses

noncomputable def total_pieces_for_type (pieces_per_house : ℕ) (houses_of_type : ℕ) : ℕ := 
  pieces_per_house * houses_of_type

noncomputable def total_pieces_for_red_or_white 
  (total_pieces : ℕ)
  (total_houses : ℕ)
  (white_houses : ℕ)
  (red_houses : ℕ) : ℕ :=
  let pieces_per_house := pieces_per_house total_pieces total_houses
  let pieces_for_white := total_pieces_for_type pieces_per_house white_houses
  let pieces_for_red := total_pieces_for_type pieces_per_house red_houses
  pieces_for_white + pieces_for_red

theorem junk_mail_per_red_or_white_house :
  ∀ (total_pieces : ℕ) (total_houses : ℕ) (white_houses : ℕ) (red_houses : ℕ),
    total_pieces = 48 →
    total_houses = 8 →
    white_houses = 2 →
    red_houses = 3 →
    total_pieces_for_red_or_white total_pieces total_houses white_houses red_houses / (white_houses + red_houses) = 6 :=
by
  intros
  sorry

end junk_mail_per_red_or_white_house_l117_117908


namespace initial_boys_down_slide_l117_117453

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end initial_boys_down_slide_l117_117453


namespace inequality_with_means_l117_117412

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l117_117412


namespace rise_in_water_level_correct_l117_117075

noncomputable def volume_of_rectangular_solid (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def area_of_circular_base (d : ℝ) : ℝ :=
  Real.pi * (d / 2) ^ 2

noncomputable def rise_in_water_level (solid_volume base_area : ℝ) : ℝ :=
  solid_volume / base_area

theorem rise_in_water_level_correct :
  let l := 10
  let w := 12
  let h := 15
  let d := 18
  let solid_volume := volume_of_rectangular_solid l w h
  let base_area := area_of_circular_base d
  let expected_rise := 7.07
  abs (rise_in_water_level solid_volume base_area - expected_rise) < 0.01 
:= 
by {
  sorry
}

end rise_in_water_level_correct_l117_117075


namespace k_cannot_be_zero_l117_117611

theorem k_cannot_be_zero (k : ℝ) (h₁ : k ≠ 0) (h₂ : 4 - 2 * k > 0) : k ≠ 0 :=
by 
  exact h₁

end k_cannot_be_zero_l117_117611


namespace triangle_BC_length_l117_117088

theorem triangle_BC_length (A B C X : Type) 
  (AB AC : ℕ) (BX CX BC : ℕ)
  (h1 : AB = 100)
  (h2 : AC = 121)
  (h3 : ∃ x y : ℕ, x = BX ∧ y = CX ∧ AB = 100 ∧ x + y = BC)
  (h4 : x * y = 31 * 149 ∧ x + y = 149) :
  BC = 149 := 
by
  sorry

end triangle_BC_length_l117_117088


namespace assign_roles_l117_117408

def maleRoles : ℕ := 3
def femaleRoles : ℕ := 3
def eitherGenderRoles : ℕ := 4
def menCount : ℕ := 7
def womenCount : ℕ := 8

theorem assign_roles : 
  (menCount.choose maleRoles) * 
  (womenCount.choose femaleRoles) * 
  ((menCount + womenCount - maleRoles - femaleRoles).choose eitherGenderRoles) = 213955200 := 
  sorry

end assign_roles_l117_117408


namespace cube_surface_area_including_inside_l117_117315

theorem cube_surface_area_including_inside 
  (original_edge_length : ℝ) 
  (hole_side_length : ℝ) 
  (original_cube_surface_area : ℝ)
  (removed_hole_area : ℝ)
  (newly_exposed_internal_area : ℝ) 
  (total_surface_area : ℝ) 
  (h1 : original_edge_length = 3)
  (h2 : hole_side_length = 1)
  (h3 : original_cube_surface_area = 6 * (original_edge_length * original_edge_length))
  (h4 : removed_hole_area = 6 * (hole_side_length * hole_side_length))
  (h5 : newly_exposed_internal_area = 6 * 4 * (hole_side_length * hole_side_length))
  (h6 : total_surface_area = original_cube_surface_area - removed_hole_area + newly_exposed_internal_area) : 
  total_surface_area = 72 :=
by
  sorry

end cube_surface_area_including_inside_l117_117315


namespace range_of_f_2x_le_1_l117_117028

-- Given conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def cond_f_neg_2_eq_1 (f : ℝ → ℝ) : Prop :=
  f (-2) = 1

-- Main theorem
theorem range_of_f_2x_le_1 (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_monotonically_decreasing f (Set.Iic 0))
  (h3 : cond_f_neg_2_eq_1 f) :
  Set.Icc (-1 : ℝ) 1 = { x | |f (2 * x)| ≤ 1 } :=
sorry

end range_of_f_2x_le_1_l117_117028


namespace paint_replacement_fractions_l117_117785

variables {r b g : ℚ}

/-- Given the initial and replacement intensities and the final intensities of red, blue,
and green paints respectively, prove the fractions of the original amounts of each paint color
that were replaced. -/
theorem paint_replacement_fractions :
  (0.6 * (1 - r) + 0.3 * r = 0.4) ∧
  (0.4 * (1 - b) + 0.15 * b = 0.25) ∧
  (0.25 * (1 - g) + 0.1 * g = 0.18) →
  (r = 2/3) ∧ (b = 3/5) ∧ (g = 7/15) :=
by
  sorry

end paint_replacement_fractions_l117_117785


namespace width_of_property_l117_117916

theorem width_of_property (W : ℝ) 
  (h1 : ∃ w l, (w = W / 8) ∧ (l = 2250 / 10) ∧ (w * l = 28125)) : W = 1000 :=
by
  -- Formal proof here
  sorry

end width_of_property_l117_117916


namespace integer_values_count_l117_117216

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l117_117216


namespace probability_three_one_l117_117567

-- Definitions based on the conditions
def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4

-- Defining the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the total number of ways to draw 4 balls from 18
def total_ways_to_draw : ℕ := binom total_balls drawn_balls

-- Definition of the number of favorable ways to draw 3 black and 1 white ball
def favorable_black_white : ℕ := binom black_balls 3 * binom white_balls 1

-- Definition of the number of favorable ways to draw 1 black and 3 white balls
def favorable_white_black : ℕ := binom black_balls 1 * binom white_balls 3

-- Total favorable outcomes
def total_favorable_ways : ℕ := favorable_black_white + favorable_white_black

-- The probability of drawing 3 one color and 1 other color
def probability : ℚ := total_favorable_ways / total_ways_to_draw

-- Prove that the probability is 19/38
theorem probability_three_one :
  probability = 19 / 38 :=
sorry

end probability_three_one_l117_117567


namespace blue_candy_count_l117_117668

def total_pieces : ℕ := 3409
def red_pieces : ℕ := 145
def blue_pieces : ℕ := total_pieces - red_pieces

theorem blue_candy_count :
  blue_pieces = 3264 := by
  sorry

end blue_candy_count_l117_117668


namespace part1_part2_l117_117284

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 4 ≤ x ∧ x < 8 }
def B : Set ℝ := { x | 3 < x ∧ x < 7 }

theorem part1 :
  (A ∩ B = { x | 4 ≤ x ∧ x < 7 }) ∧
  ((U \ A) ∪ B = { x | x < 7 ∨ x ≥ 8 }) :=
by
  sorry
  
def C (t : ℝ) : Set ℝ := { x | x < t + 1 }

theorem part2 (t : ℝ) :
  (A ∩ C t = ∅) → (t ≤ 3 ∨ t ≥ 7) :=
by
  sorry

end part1_part2_l117_117284


namespace check_bag_correct_l117_117430

-- Define the conditions as variables and statements
variables (uber_to_house : ℕ) (uber_to_airport : ℕ) (check_bag : ℕ)
          (security : ℕ) (wait_for_boarding : ℕ) (wait_for_takeoff : ℕ) (total_time : ℕ)

-- Assign the given conditions
def given_conditions : Prop :=
  uber_to_house = 10 ∧
  uber_to_airport = 5 * uber_to_house ∧
  security = 3 * check_bag ∧
  wait_for_boarding = 20 ∧
  wait_for_takeoff = 2 * wait_for_boarding ∧
  total_time = 180

-- Define the question as a statement
def check_bag_time (check_bag : ℕ) : Prop :=
  check_bag = 15

-- The Lean theorem based on the problem, conditions, and answer
theorem check_bag_correct :
  given_conditions uber_to_house uber_to_airport check_bag security wait_for_boarding wait_for_takeoff total_time →
  check_bag_time check_bag :=
by
  intros h
  sorry

end check_bag_correct_l117_117430


namespace min_x_plus_2y_l117_117820

theorem min_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 :=
sorry

end min_x_plus_2y_l117_117820


namespace students_present_l117_117360

theorem students_present (total_students : ℕ) (absent_percent : ℝ) (total_absent : ℝ) (total_present : ℝ) :
  total_students = 50 → absent_percent = 0.12 → total_absent = total_students * absent_percent →
  total_present = total_students - total_absent →
  total_present = 44 :=
by
  intros _ _ _ _; sorry

end students_present_l117_117360


namespace find_k_l117_117069

theorem find_k 
  (x y: ℝ) 
  (h1: y = 5 * x + 3) 
  (h2: y = -2 * x - 25) 
  (h3: y = 3 * x + k) : 
  k = -5 :=
sorry

end find_k_l117_117069


namespace mineral_sample_ages_l117_117550

/--
We have a mineral sample with digits {2, 2, 3, 3, 5, 9}.
Given the condition that the age must start with an odd number,
we need to prove that the total number of possible ages is 120.
-/
theorem mineral_sample_ages : 
  ∀ (l : List ℕ), l = [2, 2, 3, 3, 5, 9] → 
  (l.filter odd).length > 0 →
  ∃ n : ℕ, n = 120 :=
by
  intros l h_digits h_odd
  sorry

end mineral_sample_ages_l117_117550


namespace find_share_of_A_l117_117096

variable (A B C : ℝ)
variable (h1 : A = (2/3) * B)
variable (h2 : B = (1/4) * C)
variable (h3 : A + B + C = 510)

theorem find_share_of_A : A = 60 :=
by
  sorry

end find_share_of_A_l117_117096


namespace blocks_found_l117_117636

def initial_blocks : ℕ := 2
def final_blocks : ℕ := 86

theorem blocks_found : (final_blocks - initial_blocks) = 84 :=
by
  sorry

end blocks_found_l117_117636


namespace gcd_of_180_and_450_l117_117860

theorem gcd_of_180_and_450 : Int.gcd 180 450 = 90 := 
  sorry

end gcd_of_180_and_450_l117_117860


namespace slope_at_two_l117_117861

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2
noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem slope_at_two (a b : ℝ) (h1 : f' 1 a b = 0) (h2 : f 1 a b = 10) :
  f' 2 4 (-11) = 17 :=
sorry

end slope_at_two_l117_117861


namespace boat_distance_downstream_l117_117893

theorem boat_distance_downstream 
    (boat_speed_still : ℝ) 
    (stream_speed : ℝ) 
    (time_downstream : ℝ) 
    (distance_downstream : ℝ) 
    (h_boat_speed_still : boat_speed_still = 13) 
    (h_stream_speed : stream_speed = 6) 
    (h_time_downstream : time_downstream = 3.6315789473684212) 
    (h_distance_downstream : distance_downstream = 19 * 3.6315789473684212): 
    distance_downstream = 69 := 
by 
  have h_effective_speed : boat_speed_still + stream_speed = 19 := by 
    rw [h_boat_speed_still, h_stream_speed]; norm_num 
  rw [h_distance_downstream]; norm_num 
  sorry

end boat_distance_downstream_l117_117893


namespace probability_personA_not_personB_l117_117843

theorem probability_personA_not_personB :
  let n := Nat.choose 5 3
  let m := Nat.choose 1 1 * Nat.choose 3 2
  (m / n : ℚ) = 3 / 10 :=
by
  -- Proof omitted
  sorry

end probability_personA_not_personB_l117_117843


namespace Alchemerion_is_3_times_older_than_his_son_l117_117224

-- Definitions of Alchemerion's age, his father's age and the sum condition
def Alchemerion_age : ℕ := 360
def Father_age (A : ℕ) := 2 * A + 40
def age_sum (A S F : ℕ) := A + S + F

-- Main theorem statement
theorem Alchemerion_is_3_times_older_than_his_son (S : ℕ) (h1 : Alchemerion_age = 360)
    (h2 : Father_age Alchemerion_age = 2 * Alchemerion_age + 40)
    (h3 : age_sum Alchemerion_age S (Father_age Alchemerion_age) = 1240) :
    Alchemerion_age / S = 3 :=
sorry

end Alchemerion_is_3_times_older_than_his_son_l117_117224


namespace find_c8_l117_117240

-- Definitions of arithmetic sequences and their products
def arithmetic_seq (a d : ℤ) (n : ℕ) := a + n * d

def c_n (a d1 b d2 : ℤ) (n : ℕ) := arithmetic_seq a d1 n * arithmetic_seq b d2 n

-- Given conditions
variables (a1 d1 a2 d2 : ℤ)
variables (c1 c2 c3 : ℤ)
variables (h1 : c_n a1 d1 a2 d2 1 = 1440)
variables (h2 : c_n a1 d1 a2 d2 2 = 1716)
variables (h3 : c_n a1 d1 a2 d2 3 = 1848)

-- The goal is to prove c_8 = 348
theorem find_c8 : c_n a1 d1 a2 d2 8 = 348 :=
sorry

end find_c8_l117_117240


namespace reversed_digit_multiple_of_sum_l117_117338

variable (u v k : ℕ)

theorem reversed_digit_multiple_of_sum (h1 : 10 * u + v = k * (u + v)) :
  10 * v + u = (11 - k) * (u + v) :=
sorry

end reversed_digit_multiple_of_sum_l117_117338


namespace Josiah_spent_on_cookies_l117_117283

theorem Josiah_spent_on_cookies :
  let cookies_per_day := 2
  let cost_per_cookie := 16
  let days_in_march := 31
  2 * days_in_march * cost_per_cookie = 992 := 
by
  sorry

end Josiah_spent_on_cookies_l117_117283


namespace exist_integers_xy_divisible_by_p_l117_117332

theorem exist_integers_xy_divisible_by_p (p : ℕ) [Fact (Nat.Prime p)] : ∃ x y : ℤ, (x^2 + y^2 + 2) % p = 0 := by
  sorry

end exist_integers_xy_divisible_by_p_l117_117332


namespace number_of_four_digit_numbers_l117_117722

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end number_of_four_digit_numbers_l117_117722


namespace votes_cast_l117_117508

theorem votes_cast (V : ℝ) (candidate_votes : ℝ) (rival_margin : ℝ)
  (h1 : candidate_votes = 0.30 * V)
  (h2 : rival_margin = 4000)
  (h3 : 0.30 * V + (0.30 * V + rival_margin) = V) :
  V = 10000 := 
by 
  sorry

end votes_cast_l117_117508


namespace jenny_original_amount_half_l117_117215

-- Definitions based on conditions
def original_amount (x : ℝ) := x
def spent_fraction := 3 / 7
def left_after_spending (x : ℝ) := x * (1 - spent_fraction)

theorem jenny_original_amount_half (x : ℝ) (h : left_after_spending x = 24) : original_amount x / 2 = 21 :=
by
  -- Indicate the intention to prove the statement by sorry
  sorry

end jenny_original_amount_half_l117_117215


namespace j_mod_2_not_zero_l117_117471

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 :=
sorry

end j_mod_2_not_zero_l117_117471


namespace minimum_a_l117_117169

theorem minimum_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - (x - a) * |x - a| - 2 ≥ 0) → a ≥ Real.sqrt 3 := 
by 
  sorry

end minimum_a_l117_117169


namespace find_two_digit_number_l117_117533

-- Define the problem conditions and statement
theorem find_two_digit_number (a b n : ℕ) (h1 : a = 2 * b) (h2 : 10 * a + b + a^2 = n^2) : 
  10 * a + b = 21 :=
sorry

end find_two_digit_number_l117_117533


namespace prime_sum_and_difference_l117_117223

theorem prime_sum_and_difference (m n p : ℕ) (hmprime : Nat.Prime m) (hnprime : Nat.Prime n) (hpprime: Nat.Prime p)
  (h1: m > n)
  (h2: n > p)
  (h3 : m + n + p = 74) 
  (h4 : m - n - p = 44) : 
  m = 59 ∧ n = 13 ∧ p = 2 :=
by
  sorry

end prime_sum_and_difference_l117_117223


namespace find_m_l117_117645

theorem find_m (m : ℤ) (x y : ℤ) (h1 : x = 1) (h2 : y = m) (h3 : 3 * x - 4 * y = 7) : m = -1 :=
by
  sorry

end find_m_l117_117645


namespace eval_expr1_eval_expr2_l117_117787

theorem eval_expr1 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^3 + b^3) / (a^2 - a*b + b^2) = 10 := 
by
  sorry

theorem eval_expr2 (a b : ℝ) (h₁ : a = 7) (h₂ : b = 3) : 
  (a^2 + b^2) / (a + b) = 5.8 :=
by
  sorry

end eval_expr1_eval_expr2_l117_117787


namespace flowchart_output_is_minus_nine_l117_117337

-- Given initial state and conditions
def initialState : ℤ := 0

-- Hypothetical function representing the sequence of operations in the flowchart
-- (hiding the exact operations since they are speculative)
noncomputable def flowchartOperations (S : ℤ) : ℤ := S - 9  -- Assuming this operation represents the described flowchart

-- The proof problem
theorem flowchart_output_is_minus_nine : flowchartOperations initialState = -9 :=
by
  sorry

end flowchart_output_is_minus_nine_l117_117337


namespace arithmetic_sequence_minimization_l117_117658

theorem arithmetic_sequence_minimization (a b : ℕ) (h_range : 1 ≤ a ∧ b ≤ 17) (h_seq : a + b = 18) (h_min : ∀ x y, (1 ≤ x ∧ y ≤ 17 ∧ x + y = 18) → (1 / x + 25 / y) ≥ (1 / a + 25 / b)) : ∃ n : ℕ, n = 9 :=
by
  -- We'd usually follow by proving the conditions and defining the sequence correctly.
  -- Definitions and steps leading to finding n = 9 will be elaborated here.
  -- This placeholder is to satisfy the requirement only.
  sorry

end arithmetic_sequence_minimization_l117_117658


namespace complement_intersection_l117_117373

open Set

variable (U A B : Set ℕ)

theorem complement_intersection (U : Set ℕ) (A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 6}) (hB : B = {1, 2}) :
  ((U \ A) ∩ B) = {2} :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l117_117373


namespace find_jordana_and_james_age_l117_117760

variable (current_age_of_Jennifer : ℕ) (current_age_of_Jordana : ℕ) (current_age_of_James : ℕ)

-- Conditions
axiom jennifer_40_in_twenty_years : current_age_of_Jennifer + 20 = 40
axiom jordana_twice_jennifer_in_twenty_years : current_age_of_Jordana + 20 = 2 * (current_age_of_Jennifer + 20)
axiom james_ten_years_younger_in_twenty_years : current_age_of_James + 20 = 
  (current_age_of_Jennifer + 20) + (current_age_of_Jordana + 20) - 10

-- Prove that Jordana is currently 60 years old and James is currently 90 years old
theorem find_jordana_and_james_age : current_age_of_Jordana = 60 ∧ current_age_of_James = 90 :=
  sorry

end find_jordana_and_james_age_l117_117760


namespace find_first_offset_l117_117278

theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) 
  (h_area : area = 210) 
  (h_diagonal : diagonal = 28)
  (h_offset2 : offset2 = 6) :
  offset1 = 9 :=
by
  sorry

end find_first_offset_l117_117278


namespace trapezoid_diagonal_intersection_l117_117874

theorem trapezoid_diagonal_intersection (PQ RS PR : ℝ) (h1 : PQ = 3 * RS) (h2 : PR = 15) :
  ∃ RT : ℝ, RT = 15 / 4 :=
by
  have RT := 15 / 4
  use RT
  sorry

end trapezoid_diagonal_intersection_l117_117874


namespace x_less_than_y_by_35_percent_l117_117172

noncomputable def percentage_difference (x y : ℝ) : ℝ :=
  ((y / x) - 1) * 100

theorem x_less_than_y_by_35_percent (x y : ℝ) (h : y = 1.5384615384615385 * x) :
  percentage_difference x y = 53.846153846153854 :=
by
  sorry

end x_less_than_y_by_35_percent_l117_117172


namespace ratio_equivalence_l117_117454

theorem ratio_equivalence (m n s u : ℚ) (h1 : m / n = 5 / 4) (h2 : s / u = 8 / 15) :
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 :=
by
  sorry

end ratio_equivalence_l117_117454


namespace simplify_expression_l117_117114

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l117_117114


namespace tg_half_angle_inequality_l117_117311

variable (α β γ : ℝ)

theorem tg_half_angle_inequality 
  (h : α + β + γ = 180) : 
  (Real.tan (α / 2)) * (Real.tan (β / 2)) * (Real.tan (γ / 2)) ≤ (Real.sqrt 3) / 9 := 
sorry

end tg_half_angle_inequality_l117_117311


namespace find_specific_M_in_S_l117_117940

section MatrixProgression

variable {R : Type*} [CommRing R]

-- Definition of arithmetic progression in a 2x2 matrix.
def is_arithmetic_progression (a b c d : R) : Prop :=
  ∃ r : R, b = a + r ∧ c = a + 2 * r ∧ d = a + 3 * r

-- Definition of set S.
def S : Set (Matrix (Fin 2) (Fin 2) R) :=
  { M | ∃ a b c d : R, M = ![![a, b], ![c, d]] ∧ is_arithmetic_progression a b c d }

-- Main problem statement
theorem find_specific_M_in_S (M : Matrix (Fin 2) (Fin 2) ℝ) (k : ℕ) :
  k > 1 → M ∈ S → ∃ (α : ℝ), (M = α • ![![1, 1], ![1, 1]] ∨ (M = α • ![![ -3, -1], ![1, 3]] ∧ Odd k)) :=
by
  sorry

end MatrixProgression

end find_specific_M_in_S_l117_117940


namespace opposite_meaning_for_option_C_l117_117930

def opposite_meaning (a b : Int) : Bool :=
  (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0)

theorem opposite_meaning_for_option_C :
  (opposite_meaning 300 (-500)) ∧ 
  ¬ (opposite_meaning 5 (-5)) ∧ 
  ¬ (opposite_meaning 180 90) ∧ 
  ¬ (opposite_meaning 1 (-1)) :=
by
  unfold opposite_meaning
  sorry

end opposite_meaning_for_option_C_l117_117930


namespace domain_of_sqrt_one_minus_ln_l117_117686

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end domain_of_sqrt_one_minus_ln_l117_117686


namespace Leonard_is_11_l117_117698

def Leonard_age (L N J P T: ℕ) : Prop :=
  (L = N - 4) ∧
  (N = J / 2) ∧
  (P = 2 * L) ∧
  (T = P - 3) ∧
  (L + N + J + P + T = 75)

theorem Leonard_is_11 (L N J P T : ℕ) (h : Leonard_age L N J P T) : L = 11 :=
by {
  sorry
}

end Leonard_is_11_l117_117698


namespace wire_cut_problem_l117_117764

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ :=
  let x := total_length / (1 + ratio)
  x

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 35 → ratio = 5/2 → shorter_length = 10 → shorter_piece_length total_length ratio = shorter_length := by
  intros h1 h2 h3
  unfold shorter_piece_length
  rw [h1, h2, h3]
  sorry

end wire_cut_problem_l117_117764


namespace total_golf_balls_l117_117356

theorem total_golf_balls :
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  dan + gus + chris = 132 :=
by
  let dozen := 12
  let dan := 5 * dozen
  let gus := 2 * dozen
  let chris := 48
  sorry

end total_golf_balls_l117_117356


namespace distance_second_day_l117_117522

theorem distance_second_day 
  (total_distance : ℕ)
  (a1 : ℕ)
  (n : ℕ)
  (r : ℚ)
  (hn : n = 6)
  (htotal : total_distance = 378)
  (hr : r = 1 / 2)
  (geo_sum : a1 * (1 - r^n) / (1 - r) = total_distance) :
  a1 * r = 96 :=
by
  sorry

end distance_second_day_l117_117522


namespace arc_length_of_sector_l117_117334

theorem arc_length_of_sector (r α : ℝ) (hα : α = Real.pi / 5) (hr : r = 20) : r * α = 4 * Real.pi :=
by
  sorry

end arc_length_of_sector_l117_117334


namespace sum_lent_l117_117532

theorem sum_lent (P : ℝ) (r t : ℝ) (I : ℝ) (h1 : r = 6) (h2 : t = 6) (h3 : I = P - 672) (h4 : I = P * r * t / 100) :
  P = 1050 := by
  sorry

end sum_lent_l117_117532


namespace g_increasing_on_minus_infty_one_l117_117413

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 + (2 * x) / (1 - x)

theorem g_increasing_on_minus_infty_one : (∀ x y : ℝ, x < y → x < 1 → y ≤ 1 → g x < g y) :=
sorry

end g_increasing_on_minus_infty_one_l117_117413


namespace sqrt_diff_inequality_l117_117017

open Real

theorem sqrt_diff_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 1) < sqrt (a - 2) - sqrt (a - 3) :=
sorry

end sqrt_diff_inequality_l117_117017


namespace women_in_room_l117_117411

theorem women_in_room (M W : ℕ) 
  (h1 : 9 * M = 7 * W) 
  (h2 : M + 5 = 23) : 
  3 * (W - 4) = 57 :=
by
  sorry

end women_in_room_l117_117411


namespace radius_tangent_circle_l117_117496

theorem radius_tangent_circle (r r1 r2 : ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 5)
    (h_concentric : true) : r = 1 := by
  -- Definitions are given as conditions
  have h1 := r1 -- radius of smaller concentric circle
  have h2 := r2 -- radius of larger concentric circle
  have h3 := h_concentric -- the circles are concentric
  have h4 := h_r1 -- r1 = 3
  have h5 := h_r2 -- r2 = 5
  sorry

end radius_tangent_circle_l117_117496


namespace total_points_scored_l117_117345

-- Define the variables
def games : ℕ := 10
def points_per_game : ℕ := 12

-- Formulate the proposition to prove
theorem total_points_scored : games * points_per_game = 120 :=
by
  sorry

end total_points_scored_l117_117345


namespace shekar_average_marks_l117_117351

-- Define the scores for each subject
def mathematics := 76
def science := 65
def social_studies := 82
def english := 67
def biology := 55
def computer_science := 89
def history := 74
def geography := 63
def physics := 78
def chemistry := 71

-- Define the total number of subjects
def number_of_subjects := 10

-- State the theorem to prove the average marks
theorem shekar_average_marks :
  (mathematics + science + social_studies + english + biology +
   computer_science + history + geography + physics + chemistry) 
   / number_of_subjects = 72 := 
by
  -- Proof is omitted
  sorry

end shekar_average_marks_l117_117351


namespace upstream_distance_18_l117_117241

theorem upstream_distance_18 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) : 
  upstream_distance = 18 :=
by
  have v := (downstream_distance / downstream_time) - still_water_speed
  have upstream_distance := (still_water_speed - v) * upstream_time
  sorry

end upstream_distance_18_l117_117241


namespace tree_F_height_l117_117789

variable (A B C D E F : ℝ)

def height_conditions : Prop :=
  A = 150 ∧ -- Tree A's height is 150 feet
  B = (2 / 3) * A ∧ -- Tree B's height is 2/3 of Tree A's height
  C = (1 / 2) * B ∧ -- Tree C's height is 1/2 of Tree B's height
  D = C + 25 ∧ -- Tree D's height is 25 feet more than Tree C's height
  E = 0.40 * A ∧ -- Tree E's height is 40% of Tree A's height
  F = (B + D) / 2 -- Tree F's height is the average of Tree B's height and Tree D's height

theorem tree_F_height : height_conditions A B C D E F → F = 87.5 :=
by
  intros
  sorry

end tree_F_height_l117_117789


namespace find_number_l117_117829

theorem find_number (x : ℝ) (h : 0.30 * x - 70 = 20) : x = 300 :=
sorry

end find_number_l117_117829


namespace pascal_sum_of_squares_of_interior_l117_117712

theorem pascal_sum_of_squares_of_interior (eighth_row_interior : List ℕ) 
    (h : eighth_row_interior = [7, 21, 35, 35, 21, 7]) : 
    (eighth_row_interior.map (λ x => x * x)).sum = 3430 := 
by
  sorry

end pascal_sum_of_squares_of_interior_l117_117712


namespace ratio_of_length_to_perimeter_is_one_over_four_l117_117518

-- We define the conditions as given in the problem.
def room_length_1 : ℕ := 23 -- length of the rectangle in feet
def room_width_1 : ℕ := 15  -- width of the rectangle in feet
def room_width_2 : ℕ := 8   -- side of the square in feet

-- Total dimensions after including the square
def total_length : ℕ := room_length_1  -- total length remains the same
def total_width : ℕ := room_width_1 + room_width_2  -- width is sum of widths

-- Defining the perimeter
def perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width

-- Calculate the ratio
def length_to_perimeter_ratio (length perimeter : ℕ) : ℚ := length / perimeter

-- Theorem to prove the desired ratio is 1:4
theorem ratio_of_length_to_perimeter_is_one_over_four : 
  length_to_perimeter_ratio total_length (perimeter total_length total_width) = 1 / 4 :=
by
  -- Proof code would go here
  sorry

end ratio_of_length_to_perimeter_is_one_over_four_l117_117518


namespace find_b_value_l117_117213

theorem find_b_value :
  (∀ x : ℝ, (x < 0 ∨ x > 4) → -x^2 + 4*x - 4 < 0) ↔ b = 4 := by
sorry

end find_b_value_l117_117213


namespace integer_solution_n_l117_117417

theorem integer_solution_n 
  (n : Int) 
  (h1 : n + 13 > 15) 
  (h2 : -6 * n > -18) : 
  n = 2 := 
sorry

end integer_solution_n_l117_117417


namespace expand_product_l117_117706

theorem expand_product (x : ℝ) : (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 :=
by
  sorry

end expand_product_l117_117706


namespace range_of_T_l117_117720

open Real

theorem range_of_T (x y z : ℝ) (h : x^2 + 2 * y^2 + 3 * z^2 = 4) : 
    - (2 * sqrt 6) / 3 ≤ x * y + y * z ∧ x * y + y * z ≤ (2 * sqrt 6) / 3 := 
by 
    sorry

end range_of_T_l117_117720


namespace max_value_expression_l117_117089

theorem max_value_expression : ∃ (max_val : ℝ), max_val = (1 / 16) ∧ ∀ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → (a - b^2) * (b - a^2) ≤ max_val :=
by
  sorry

end max_value_expression_l117_117089


namespace total_dinners_l117_117751

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l117_117751


namespace gcd_lcm_of_300_105_l117_117355

theorem gcd_lcm_of_300_105 :
  ∃ g l : ℕ, g = Int.gcd 300 105 ∧ l = Nat.lcm 300 105 ∧ g = 15 ∧ l = 2100 :=
by
  let g := Int.gcd 300 105
  let l := Nat.lcm 300 105
  have g_def : g = 15 := sorry
  have l_def : l = 2100 := sorry
  exact ⟨g, l, ⟨g_def, ⟨l_def, ⟨g_def, l_def⟩⟩⟩⟩

end gcd_lcm_of_300_105_l117_117355


namespace each_mouse_not_visit_with_every_other_once_l117_117684

theorem each_mouse_not_visit_with_every_other_once : 
    (∃ mice: Finset ℕ, mice.card = 24 ∧ (∀ f : ℕ → Finset ℕ, 
    (∀ n, (f n).card = 4) ∧ 
    (∀ i j, i ≠ j → (f i ∩ f j ≠ ∅) → (f i ∩ f j).card ≠ (mice.card - 1)))
    ) → false := 
by
  sorry

end each_mouse_not_visit_with_every_other_once_l117_117684


namespace strawberry_unit_prices_l117_117862

theorem strawberry_unit_prices (x y : ℝ) (h1 : x = 1.5 * y) (h2 : 2 * x - 2 * y = 10) : x = 15 ∧ y = 10 :=
by
  sorry

end strawberry_unit_prices_l117_117862


namespace inequality_am_gm_l117_117802

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
by sorry

end inequality_am_gm_l117_117802


namespace value_of_t_l117_117146

theorem value_of_t (x y t : ℝ) (hx : 2^x = t) (hy : 7^y = t) (hxy : 1/x + 1/y = 2) : t = Real.sqrt 14 :=
by
  sorry

end value_of_t_l117_117146


namespace list_of_21_numbers_l117_117939

theorem list_of_21_numbers (numbers : List ℝ) (n : ℝ) (h_length : numbers.length = 21) 
  (h_mem : n ∈ numbers) 
  (h_n_avg : n = 4 * (numbers.sum - n) / 20) 
  (h_n_sum : n = (numbers.sum) / 6) : numbers.length - 1 = 20 :=
by
  -- We provide the statement with the correct hypotheses
  -- the proof is yet to be filled in
  sorry

end list_of_21_numbers_l117_117939


namespace minimum_value_expression_l117_117836

variable (a b c k : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a = k ∧ b = k ∧ c = k)

theorem minimum_value_expression : 
  (a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) = 9 / 2 :=
by
  sorry

end minimum_value_expression_l117_117836


namespace compute_value_condition_l117_117576

theorem compute_value_condition (x : ℝ) (h : x + (1 / x) = 3) :
  (x - 2) ^ 2 + 25 / (x - 2) ^ 2 = -x + 5 := by
  sorry

end compute_value_condition_l117_117576


namespace polynomial_at_most_one_integer_root_l117_117258

theorem polynomial_at_most_one_integer_root (n : ℤ) :
  ∀ x1 x2 : ℤ, (x1 ≠ x2) → 
  (x1 ^ 4 - 1993 * x1 ^ 3 + (1993 + n) * x1 ^ 2 - 11 * x1 + n = 0) → 
  (x2 ^ 4 - 1993 * x2 ^ 3 + (1993 + n) * x2 ^ 2 - 11 * x2 + n = 0) → 
  false :=
by
  sorry

end polynomial_at_most_one_integer_root_l117_117258


namespace probability_condition_l117_117736

namespace SharedPowerBank

def P (event : String) : ℚ :=
  match event with
  | "A" => 3 / 4
  | "B" => 1 / 2
  | _   => 0 -- Default case for any other event

def probability_greater_than_1000_given_greater_than_500 : ℚ :=
  P "B" / P "A"

theorem probability_condition :
  probability_greater_than_1000_given_greater_than_500 = 2 / 3 :=
by 
  sorry

end SharedPowerBank

end probability_condition_l117_117736


namespace candidates_count_l117_117803

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 := 
sorry

end candidates_count_l117_117803


namespace number_of_combinations_of_planets_is_1141_l117_117947

def number_of_combinations_of_planets : ℕ :=
  (if 7 ≥ 7 ∧ 8 ≥2 then Nat.choose 7 7 * Nat.choose 8 2 else 0) + 
  (if 7 ≥ 6 ∧ 8 ≥ 4 then Nat.choose 7 6 * Nat.choose 8 4 else 0) + 
  (if 7 ≥ 5 ∧ 8 ≥ 6 then Nat.choose 7 5 * Nat.choose 8 6 else 0) +
  (if 7 ≥ 4 ∧ 8 ≥ 8 then Nat.choose 7 4 * Nat.choose 8 8 else 0)

theorem number_of_combinations_of_planets_is_1141 :
  number_of_combinations_of_planets = 1141 :=
by
  sorry

end number_of_combinations_of_planets_is_1141_l117_117947


namespace area_enclosed_by_sin_l117_117379

/-- The area of the figure enclosed by the curve y = sin(x), the lines x = -π/3, x = π/2, and the x-axis is 3/2. -/
theorem area_enclosed_by_sin (x y : ℝ) (h : y = Real.sin x) (a b : ℝ) 
(h1 : a = -Real.pi / 3) (h2 : b = Real.pi / 2) :
  ∫ x in a..b, |Real.sin x| = 3 / 2 := 
sorry

end area_enclosed_by_sin_l117_117379


namespace functions_increase_faster_l117_117516

-- Define the functions
def y₁ (x : ℝ) : ℝ := 100 * x
def y₂ (x : ℝ) : ℝ := 1000 + 100 * x
def y₃ (x : ℝ) : ℝ := 10000 + 99 * x

-- Restate the problem in Lean
theorem functions_increase_faster :
  (∀ (x : ℝ), deriv y₁ x = 100) ∧
  (∀ (x : ℝ), deriv y₂ x = 100) ∧
  (∀ (x : ℝ), deriv y₃ x = 99) ∧
  (100 > 99) :=
by
  sorry

end functions_increase_faster_l117_117516


namespace correct_choice_l117_117265

theorem correct_choice : 2 ∈ ({0, 1, 2} : Set ℕ) :=
sorry

end correct_choice_l117_117265


namespace probability_of_distance_less_than_8000_l117_117771

-- Define distances between cities

noncomputable def distances : List (String × String × ℕ) :=
  [("Bangkok", "Cape Town", 6300),
   ("Bangkok", "Honolulu", 7609),
   ("Bangkok", "London", 5944),
   ("Bangkok", "Tokyo", 2870),
   ("Cape Town", "Honolulu", 11535),
   ("Cape Town", "London", 5989),
   ("Cape Town", "Tokyo", 13400),
   ("Honolulu", "London", 7240),
   ("Honolulu", "Tokyo", 3805),
   ("London", "Tokyo", 5950)]

-- Define the total number of pairs and the pairs with distances less than 8000 miles

noncomputable def total_pairs : ℕ := 10
noncomputable def pairs_less_than_8000 : ℕ := 7

-- Define the statement of the probability being 7/10
theorem probability_of_distance_less_than_8000 :
  pairs_less_than_8000 / total_pairs = 7 / 10 :=
by
  sorry

end probability_of_distance_less_than_8000_l117_117771


namespace charles_travel_time_l117_117834

theorem charles_travel_time (D S T : ℕ) (hD : D = 6) (hS : S = 3) : T = D / S → T = 2 :=
by
  intros h
  rw [hD, hS] at h
  simp at h
  exact h

end charles_travel_time_l117_117834


namespace intersection_A_B_l117_117200

open Set

def f (x : ℕ) : ℕ := x^2 - 12 * x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a, a ∈ A ∧ b = f a}

theorem intersection_A_B : A ∩ B = {1, 4, 9} :=
by
  -- Proof skipped
  sorry

end intersection_A_B_l117_117200


namespace cassidy_total_grounding_days_l117_117578

-- Define the initial grounding days
def initial_grounding_days : ℕ := 14

-- Define the grounding days per grade below a B
def extra_days_per_grade : ℕ := 3

-- Define the number of grades below a B
def grades_below_B : ℕ := 4

-- Define the total grounding days calculation
def total_grounding_days : ℕ := initial_grounding_days + grades_below_B * extra_days_per_grade

-- The theorem statement
theorem cassidy_total_grounding_days :
  total_grounding_days = 26 := 
sorry

end cassidy_total_grounding_days_l117_117578


namespace value_of_a_b_squared_l117_117809

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a - b = Real.sqrt 2
axiom h2 : a * b = 4

theorem value_of_a_b_squared : (a + b)^2 = 18 := by
   sorry

end value_of_a_b_squared_l117_117809


namespace sequence_2018_value_l117_117243

theorem sequence_2018_value (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) - a n = (-1 / 2) ^ n) :
  a 2018 = (2 * (1 - (1 / 2) ^ 2018)) / 3 :=
by sorry

end sequence_2018_value_l117_117243


namespace evaluate_expression_l117_117931

def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 7

theorem evaluate_expression : 3 * f 2 - 2 * f (-2) = 55 := by
  sorry

end evaluate_expression_l117_117931


namespace part1_part2_l117_117187

theorem part1 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (6 * Real.sin θ + Real.cos θ) / (3 * Real.sin θ - 2 * Real.cos θ) = 13 / 4 :=
sorry

theorem part2 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
sorry

end part1_part2_l117_117187


namespace find_d_l117_117113

theorem find_d (d : ℚ) (int_part frac_part : ℚ) 
  (h1 : 3 * int_part^2 + 19 * int_part - 28 = 0)
  (h2 : 4 * frac_part^2 - 11 * frac_part + 3 = 0)
  (h3 : frac_part ≥ 0 ∧ frac_part < 1)
  (h4 : d = int_part + frac_part) :
  d = -29 / 4 :=
by
  sorry

end find_d_l117_117113


namespace probability_of_head_l117_117090

def events : Type := {e // e = "H" ∨ e = "T"}

def equallyLikely (e : events) : Prop :=
  e = ⟨"H", Or.inl rfl⟩ ∨ e = ⟨"T", Or.inr rfl⟩

def totalOutcomes := 2

def probOfHead : ℚ := 1 / totalOutcomes

theorem probability_of_head : probOfHead = 1 / 2 :=
by
  sorry

end probability_of_head_l117_117090


namespace sum_of_numbers_l117_117831

theorem sum_of_numbers (x : ℝ) (h : x^2 + (2 * x)^2 + (4 * x)^2 = 4725) : 
  x + 2 * x + 4 * x = 105 := 
sorry

end sum_of_numbers_l117_117831


namespace total_votes_is_correct_l117_117431

-- Definitions and theorem statement
theorem total_votes_is_correct (T : ℝ) 
  (votes_for_A : ℝ) 
  (candidate_A_share : ℝ) 
  (valid_vote_fraction : ℝ) 
  (invalid_vote_fraction : ℝ) 
  (votes_for_A_equals: votes_for_A = 380800) 
  (candidate_A_share_equals: candidate_A_share = 0.80) 
  (valid_vote_fraction_equals: valid_vote_fraction = 0.85) 
  (invalid_vote_fraction_equals: invalid_vote_fraction = 0.15) 
  (valid_vote_computed: votes_for_A = candidate_A_share * valid_vote_fraction * T): 
  T = 560000 := 
by 
  sorry

end total_votes_is_correct_l117_117431


namespace ellipse_eq_line_eq_l117_117507

-- Conditions for part (I)
def cond1 (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a > b
def pt_p_cond (PF1 PF2 : ℝ) : Prop := PF1 = 4 / 3 ∧ PF2 = 14 / 3 ∧ PF1^2 + PF2^2 = 1

-- Theorem for part (I)
theorem ellipse_eq (a b : ℝ) (PF1 PF2 : ℝ) (h₁ : cond1 a b) (h₂ : pt_p_cond PF1 PF2) : 
  (a = 3 ∧ b = 2 ∧ PF1 = 4 / 3 ∧ PF2 = 14 / 3) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Conditions for part (II)
def center_circle (M : ℝ × ℝ) : Prop := M = (-2, 1)
def pts_symmetric (A B M : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

-- Theorem for part (II)
theorem line_eq (A B M : ℝ × ℝ) (k : ℝ) (h₁ : center_circle M) (h₂ : pts_symmetric A B M) :
  k = 8 / 9 → (∀ x y : ℝ, 8 * x - 9 * y + 25 = 0) :=
sorry

end ellipse_eq_line_eq_l117_117507


namespace find_k_l117_117702

theorem find_k (k : ℚ) (h : ∃ k : ℚ, (3 * (4 - k) = 2 * (-5 - 3))): k = -4 / 3 := by
  sorry

end find_k_l117_117702


namespace ball_weights_l117_117164

-- Define the weights of red and white balls we are going to use in our conditions and goal
variables (R W : ℚ)

-- State the conditions as hypotheses
axiom h1 : 7 * R + 5 * W = 43
axiom h2 : 5 * R + 7 * W = 47

-- State the theorem we want to prove, given the conditions
theorem ball_weights :
  4 * R + 8 * W = 49 :=
by
  sorry

end ball_weights_l117_117164


namespace area_of_rectangle_l117_117629

theorem area_of_rectangle (x : ℝ) (hx : 0 < x) :
  let length := 3 * x - 1
  let width := 2 * x + 1 / 2
  let area := length * width
  area = 6 * x^2 - 1 / 2 * x - 1 / 2 :=
by
  sorry

end area_of_rectangle_l117_117629


namespace first_player_wins_l117_117042

def wins (sum_rows sum_cols : ℕ) : Prop := sum_rows > sum_cols

theorem first_player_wins 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (h : a_1 > a_2 ∧ a_2 > a_3 ∧ a_3 > a_4 ∧ a_4 > a_5 ∧ a_5 > a_6 ∧ a_6 > a_7 ∧ a_7 > a_8 ∧ a_8 > a_9) :
  ∃ sum_rows sum_cols, wins sum_rows sum_cols :=
sorry

end first_player_wins_l117_117042


namespace mary_saves_in_five_months_l117_117394

def washing_earnings : ℕ := 20
def walking_earnings : ℕ := 40
def monthly_earnings : ℕ := washing_earnings + walking_earnings
def savings_rate : ℕ := 2
def monthly_savings : ℕ := monthly_earnings / savings_rate
def total_savings_target : ℕ := 150

theorem mary_saves_in_five_months :
  total_savings_target / monthly_savings = 5 :=
by
  sorry

end mary_saves_in_five_months_l117_117394


namespace ammonium_iodide_requirement_l117_117703

theorem ammonium_iodide_requirement :
  ∀ (NH4I KOH NH3 KI H2O : ℕ),
  (NH4I + KOH = NH3 + KI + H2O) → 
  (NH4I = 3) →
  (KOH = 3) →
  (NH3 = 3) →
  (KI = 3) →
  (H2O = 3) →
  NH4I = 3 :=
by
  intros NH4I KOH NH3 KI H2O reaction_balanced NH4I_req KOH_req NH3_prod KI_prod H2O_prod
  exact NH4I_req

end ammonium_iodide_requirement_l117_117703


namespace total_dogs_is_28_l117_117239

def number_of_boxes : ℕ := 7
def dogs_per_box : ℕ := 4
def total_dogs (boxes : ℕ) (dogs_in_each : ℕ) : ℕ := boxes * dogs_in_each

theorem total_dogs_is_28 : total_dogs number_of_boxes dogs_per_box = 28 :=
by
  sorry

end total_dogs_is_28_l117_117239


namespace find_valid_m_l117_117227

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem find_valid_m (m : ℝ) : (∀ x, ∃ y, g m x = y ∧ g m y = x) ↔ (m ∈ Set.Iio (-9 / 4) ∪ Set.Ioi (-9 / 4)) :=
by
  sorry

end find_valid_m_l117_117227


namespace balls_total_correct_l117_117530

-- Definitions based on the problem conditions
def red_balls_initial : ℕ := 16
def blue_balls : ℕ := 2 * red_balls_initial
def red_balls_lost : ℕ := 6
def red_balls_remaining : ℕ := red_balls_initial - red_balls_lost
def total_balls_after : ℕ := 74
def nonblue_red_balls_remaining : ℕ := red_balls_remaining + blue_balls

-- Goal: Find the number of yellow balls
def yellow_balls_bought : ℕ := total_balls_after - nonblue_red_balls_remaining

theorem balls_total_correct :
  yellow_balls_bought = 32 :=
by
  -- Proof would go here
  sorry

end balls_total_correct_l117_117530


namespace system1_solution_correct_system2_solution_correct_l117_117455

theorem system1_solution_correct (x y : ℝ) (h1 : x + y = 5) (h2 : 4 * x - 2 * y = 2) :
    x = 2 ∧ y = 3 :=
  sorry

theorem system2_solution_correct (x y : ℝ) (h1 : 3 * x - 2 * y = 13) (h2 : 4 * x + 3 * y = 6) :
    x = 3 ∧ y = -2 :=
  sorry

end system1_solution_correct_system2_solution_correct_l117_117455


namespace polygon_sides_l117_117983

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∃ (theta theta' : ℝ), theta = (n - 2) * 180 / n ∧ theta' = (n + 7) * 180 / (n + 9) ∧ theta' = theta + 9) : n = 15 :=
sorry

end polygon_sides_l117_117983


namespace squared_remainder_l117_117582

theorem squared_remainder (N : ℤ) (k : ℤ) :
  (N % 9 = 2 ∨ N % 9 = 7) → 
  (N^2 % 9 = 4) :=
by
  sorry

end squared_remainder_l117_117582


namespace union_sets_l117_117457

-- Define the sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem union_sets : A ∪ B = { x | -1 < x ∧ x ≤ 4 } := 
by
   sorry

end union_sets_l117_117457


namespace line_equation_passing_through_P_and_equal_intercepts_l117_117206

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: line passes through point P(1, 3)
def passes_through_P (P : Point) (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq 1 3 = 0

-- Define the condition: equal intercepts on the x-axis and y-axis
def has_equal_intercepts (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ (∀ x y, line_eq x y = 0 ↔ x / a + y / a = 1)

-- Define the specific lines x + y - 4 = 0 and 3x - y = 0
def specific_line1 (x y : ℝ) : ℝ := x + y - 4
def specific_line2 (x y : ℝ) : ℝ := 3 * x - y

-- Define the point P(1, 3)
def P := Point.mk 1 3

theorem line_equation_passing_through_P_and_equal_intercepts :
  (passes_through_P P specific_line1 ∧ has_equal_intercepts specific_line1) ∨
  (passes_through_P P specific_line2 ∧ has_equal_intercepts specific_line2) :=
by
  sorry

end line_equation_passing_through_P_and_equal_intercepts_l117_117206


namespace boys_number_l117_117885

variable (M W B : ℕ)

-- Conditions
axiom h1 : M = W
axiom h2 : W = B
axiom h3 : M * 8 = 120

theorem boys_number :
  B = 15 := by
  sorry

end boys_number_l117_117885


namespace find_value_l117_117018

-- Given points A(a, 1), B(2, b), and C(3, 4).
variables (a b : ℝ)

-- Given condition from the problem
def condition : Prop := (3 * a + 4 = 6 + 4 * b)

-- The target is to find 3a - 4b
def target : ℝ := 3 * a - 4 * b

theorem find_value (h : condition a b) : target a b = 2 := 
by sorry

end find_value_l117_117018


namespace mans_rate_is_19_l117_117705

-- Define the given conditions
def downstream_speed : ℝ := 25
def upstream_speed : ℝ := 13

-- Define the man's rate in still water and state the theorem
theorem mans_rate_is_19 : (downstream_speed + upstream_speed) / 2 = 19 := by
  -- Proof goes here
  sorry

end mans_rate_is_19_l117_117705


namespace direct_proportion_function_l117_117927

-- Define the conditions for the problem
def condition1 (m : ℝ) : Prop := m ^ 2 - 1 = 0
def condition2 (m : ℝ) : Prop := m - 1 ≠ 0

-- The main theorem we need to prove
theorem direct_proportion_function (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = -1 :=
by
  sorry

end direct_proportion_function_l117_117927


namespace correct_order_of_operations_l117_117209

def order_of_operations (e : String) : String :=
  if e = "38 * 50 - 25 / 5" then
    "multiplication, division, subtraction"
  else
    "unknown"

theorem correct_order_of_operations :
  order_of_operations "38 * 50 - 25 / 5" = "multiplication, division, subtraction" :=
by
  sorry

end correct_order_of_operations_l117_117209


namespace new_computer_price_l117_117986

theorem new_computer_price (d : ℕ) (h : 2 * d = 560) : d + 3 * d / 10 = 364 :=
by
  sorry

end new_computer_price_l117_117986


namespace distance_between_first_and_last_is_140_l117_117556

-- Given conditions
def eightFlowers : ℕ := 8
def distanceFirstToFifth : ℕ := 80
def intervalsBetweenFirstAndFifth : ℕ := 4 -- 1 to 5 means 4 intervals
def intervalsBetweenFirstAndLast : ℕ := 7 -- 1 to 8 means 7 intervals
def distanceBetweenConsecutiveFlowers : ℕ := distanceFirstToFifth / intervalsBetweenFirstAndFifth
def totalDistanceFirstToLast : ℕ := distanceBetweenConsecutiveFlowers * intervalsBetweenFirstAndLast

-- Theorem to prove the question equals the correct answer
theorem distance_between_first_and_last_is_140 :
  totalDistanceFirstToLast = 140 := by
  sorry

end distance_between_first_and_last_is_140_l117_117556


namespace kaylee_more_boxes_to_sell_l117_117923

-- Definitions for the conditions
def total_needed_boxes : ℕ := 33
def sold_to_aunt : ℕ := 12
def sold_to_mother : ℕ := 5
def sold_to_neighbor : ℕ := 4

-- Target proof goal
theorem kaylee_more_boxes_to_sell :
  total_needed_boxes - (sold_to_aunt + sold_to_mother + sold_to_neighbor) = 12 :=
sorry

end kaylee_more_boxes_to_sell_l117_117923


namespace problem_statement_l117_117948

-- Define line and plane as types
variable (Line Plane : Type)

-- Define the perpendicularity and parallelism relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLPlane : Line → Plane → Prop)
variable (perpendicularPPlane : Plane → Plane → Prop)

-- Distinctness of lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Conditions given in the problem
axiom distinct_lines : a ≠ b
axiom distinct_planes : α ≠ β

-- Statement to be proven
theorem problem_statement :
  perpendicular a b → 
  perpendicularLPlane a α → 
  perpendicularLPlane b β → 
  perpendicularPPlane α β :=
sorry

end problem_statement_l117_117948


namespace negation_equivalence_l117_117676

-- Define the angles in a triangle as three real numbers
def is_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the proposition
def at_least_one_angle_not_greater_than_60 (a b c : ℝ) : Prop :=
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60

-- Negate the proposition
def all_angles_greater_than_60 (a b c : ℝ) : Prop :=
  a > 60 ∧ b > 60 ∧ c > 60

-- Prove that the negation of the proposition is equivalent
theorem negation_equivalence (a b c : ℝ) (h_triangle : is_triangle a b c) :
  ¬ at_least_one_angle_not_greater_than_60 a b c ↔ all_angles_greater_than_60 a b c :=
by
  sorry

end negation_equivalence_l117_117676


namespace three_primes_sum_odd_l117_117793

theorem three_primes_sum_odd (primes : Finset ℕ) (h_prime : ∀ p ∈ primes, Prime p) :
  primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} →
  (Nat.choose 9 3 / Nat.choose 10 3 : ℚ) = 7 / 10 := by
  -- Let the set of first ten prime numbers.
  -- As per condition, primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  -- Then show that the probability calculation yields 7/10
  sorry

end three_primes_sum_odd_l117_117793


namespace incorrect_statement_l117_117435

theorem incorrect_statement (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬((x = 0 → a^x = 1) ∧
    (x = 1 → a^x = a) ∧
    (x = -1 → a^x = 1/a) ∧
    (x < 0 → 0 < a^x ∧ ∀ ε > 0, ∃ x' < x, a^x' < ε)) :=
sorry

end incorrect_statement_l117_117435


namespace earnings_proof_l117_117161

theorem earnings_proof (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 300) (h3 : C = 100) : A + C = 400 :=
sorry

end earnings_proof_l117_117161


namespace remainder_of_9_pow_333_div_50_l117_117961

theorem remainder_of_9_pow_333_div_50 : (9 ^ 333) % 50 = 29 :=
by
  sorry

end remainder_of_9_pow_333_div_50_l117_117961


namespace problem_statement_l117_117740

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end problem_statement_l117_117740


namespace dubblefud_red_balls_l117_117104

theorem dubblefud_red_balls (R B : ℕ) 
  (h1 : 2 ^ R * 4 ^ B * 5 ^ B = 16000)
  (h2 : B = G) : R = 6 :=
by
  -- Skipping the actual proof
  sorry

end dubblefud_red_balls_l117_117104


namespace fuel_consumption_gallons_l117_117465

theorem fuel_consumption_gallons
  (distance_per_liter : ℝ)
  (speed_mph : ℝ)
  (time_hours : ℝ)
  (mile_to_km : ℝ)
  (gallon_to_liters : ℝ)
  (fuel_consumption : ℝ) :
  distance_per_liter = 56 →
  speed_mph = 91 →
  time_hours = 5.7 →
  mile_to_km = 1.6 →
  gallon_to_liters = 3.8 →
  fuel_consumption = 3.9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end fuel_consumption_gallons_l117_117465


namespace ratio_tough_to_good_sales_l117_117477

-- Define the conditions
def tough_week_sales : ℤ := 800
def total_sales : ℤ := 10400
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the problem in Lean 4:
theorem ratio_tough_to_good_sales : ∃ G : ℤ, (good_weeks * G) + (tough_weeks * tough_week_sales) = total_sales ∧ 
  (tough_week_sales : ℚ) / (G : ℚ) = 1 / 2 :=
sorry

end ratio_tough_to_good_sales_l117_117477


namespace heaviest_lightest_difference_total_excess_weight_total_selling_price_l117_117498

-- Define deviations from standard weight and their counts
def deviations : List (ℚ × ℕ) := [(-3.5, 2), (-2, 4), (-1.5, 2), (0, 1), (1, 3), (2.5, 8)]

-- Define standard weight and price per kg
def standard_weight : ℚ := 18
def price_per_kg : ℚ := 1.8

-- Prove the three statements:
theorem heaviest_lightest_difference :
  (2.5 - (-3.5)) = 6 := by
  sorry

theorem total_excess_weight :
  (2 * -3.5 + 4 * -2 + 2 * -1.5 + 1 * 0 + 3 * 1 + 8 * 2.5) = 5 := by
  sorry

theorem total_selling_price :
  (standard_weight * 20 + 5) * price_per_kg = 657 := by
  sorry

end heaviest_lightest_difference_total_excess_weight_total_selling_price_l117_117498


namespace ten_sided_polygon_diagonals_l117_117279

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem ten_sided_polygon_diagonals :
  number_of_diagonals 10 = 35 :=
by sorry

end ten_sided_polygon_diagonals_l117_117279


namespace curve_is_hyperbola_l117_117493

theorem curve_is_hyperbola (m n x y : ℝ) (h_eq : m * x^2 - m * y^2 = n) (h_mn : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2/a^2 - x^2/a^2 = 1 := 
sorry

end curve_is_hyperbola_l117_117493


namespace problem_solution_l117_117851

theorem problem_solution (n : ℤ) : 
  (1 / (n + 2) + 3 / (n + 2) + 2 * n / (n + 2) = 4) → (n = -2) :=
by
  intro h
  sorry

end problem_solution_l117_117851


namespace travel_time_correct_l117_117781

noncomputable def timeSpentOnRoad : Nat :=
  let startTime := 7  -- 7:00 AM in hours
  let endTime := 20   -- 8:00 PM in hours
  let totalJourneyTime := endTime - startTime
  let stopTimes := [25, 10, 25]  -- minutes
  let totalStopTime := stopTimes.foldl (· + ·) 0
  let stopTimeInHours := totalStopTime / 60
  totalJourneyTime - stopTimeInHours

theorem travel_time_correct : timeSpentOnRoad = 12 :=
by
  sorry

end travel_time_correct_l117_117781


namespace sequence_ratio_l117_117340

theorem sequence_ratio (S T a b : ℕ → ℚ) (h_sum_ratio : ∀ (n : ℕ), S n / T n = (7*n + 2) / (n + 3)) :
  a 7 / b 7 = 93 / 16 :=
by
  sorry

end sequence_ratio_l117_117340


namespace trapezoid_area_calc_l117_117561

noncomputable def isoscelesTrapezoidArea : ℝ :=
  let a := 1
  let b := 9
  let h := 2 * Real.sqrt 3
  0.5 * (a + b) * h

theorem trapezoid_area_calc : isoscelesTrapezoidArea = 20 * Real.sqrt 3 := by
  sorry

end trapezoid_area_calc_l117_117561


namespace range_of_x_in_function_l117_117016

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function_l117_117016


namespace smallest_prime_with_digit_sum_23_l117_117083

-- Definition for the conditions
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The theorem stating the proof problem
theorem smallest_prime_with_digit_sum_23 : ∃ p : ℕ, Prime p ∧ sum_of_digits p = 23 ∧ p = 1993 := 
by {
 sorry
}

end smallest_prime_with_digit_sum_23_l117_117083


namespace sara_walking_distance_l117_117432

noncomputable def circle_area := 616
noncomputable def pi_estimate := (22: ℚ) / 7
noncomputable def extra_distance := 3

theorem sara_walking_distance (r : ℚ) (radius_pos : 0 < r) : 
  pi_estimate * r^2 = circle_area →
  2 * pi_estimate * r + extra_distance = 91 :=
by
  intros h
  sorry

end sara_walking_distance_l117_117432


namespace three_powers_in_two_digit_range_l117_117033

theorem three_powers_in_two_digit_range :
  ∃ n_values : Finset ℕ, (∀ n ∈ n_values, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ n_values.card = 3 :=
by
  sorry

end three_powers_in_two_digit_range_l117_117033


namespace typing_time_in_hours_l117_117085

def words_per_minute := 32
def word_count := 7125
def break_interval := 25
def break_time := 5
def mistake_interval := 100
def correction_time_per_mistake := 1

theorem typing_time_in_hours :
  let typing_time := (word_count + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let total_break_time := breaks * break_time
  let mistakes := (word_count + mistake_interval - 1) / mistake_interval
  let total_correction_time := mistakes * correction_time_per_mistake
  let total_time := typing_time + total_break_time + total_correction_time
  let total_hours := (total_time + 60 - 1) / 60
  total_hours = 6 :=
by
  sorry

end typing_time_in_hours_l117_117085


namespace enchilada_taco_cost_l117_117692

variables (e t : ℝ)

theorem enchilada_taco_cost 
  (h1 : 4 * e + 5 * t = 4.00) 
  (h2 : 5 * e + 3 * t = 3.80) 
  (h3 : 7 * e + 6 * t = 6.10) : 
  4 * e + 7 * t = 4.75 := 
sorry

end enchilada_taco_cost_l117_117692


namespace construct_triangle_from_medians_l117_117773

theorem construct_triangle_from_medians
    (s_a s_b s_c : ℝ)
    (h1 : s_a + s_b > s_c)
    (h2 : s_a + s_c > s_b)
    (h3 : s_b + s_c > s_a) :
    ∃ (a b c : ℝ), 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∃ (median_a median_b median_c : ℝ), 
        median_a = s_a ∧ 
        median_b = s_b ∧ 
        median_c = s_c) :=
sorry

end construct_triangle_from_medians_l117_117773


namespace fraction_sum_equals_l117_117382

theorem fraction_sum_equals : 
    (4 / 2) + (7 / 4) + (11 / 8) + (21 / 16) + (41 / 32) + (81 / 64) - 8 = 63 / 64 :=
by 
    sorry

end fraction_sum_equals_l117_117382


namespace simplify_expression1_simplify_expression2_l117_117044

-- Problem 1
theorem simplify_expression1 (a : ℝ) : 
  (a^2)^3 + 3 * a^4 * a^2 - a^8 / a^2 = 3 * a^6 :=
by sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  (x - 3) * (x + 4) - x * (x + 3) = -2 * x - 12 :=
by sorry

end simplify_expression1_simplify_expression2_l117_117044


namespace correct_operation_l117_117124

variable (a b : ℝ)

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ ((a^2)^3 = a^5) ∧
  ¬ (a^2 * a^3 = a^6) ∧
  ((-a * b)^5 / (-a * b)^3 = a^2 * b^2) :=
by
  sorry

end correct_operation_l117_117124


namespace tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l117_117168

structure Tetrahedron :=
  (faces : Nat := 4)
  (vertices : Nat := 4)
  (valence : Nat := 3)
  (face_shape : String := "triangular")

structure Cube :=
  (faces : Nat := 6)
  (vertices : Nat := 8)
  (valence : Nat := 3)
  (face_shape : String := "square")

structure Octahedron :=
  (faces : Nat := 8)
  (vertices : Nat := 6)
  (valence : Nat := 4)
  (face_shape : String := "triangular")

structure Dodecahedron :=
  (faces : Nat := 12)
  (vertices : Nat := 20)
  (valence : Nat := 3)
  (face_shape : String := "pentagonal")

structure Icosahedron :=
  (faces : Nat := 20)
  (vertices : Nat := 12)
  (valence : Nat := 5)
  (face_shape : String := "triangular")

theorem tetrahedron_is_self_dual:
  Tetrahedron := by
  sorry

theorem cube_is_dual_to_octahedron:
  Cube × Octahedron := by
  sorry

theorem dodecahedron_is_dual_to_icosahedron:
  Dodecahedron × Icosahedron := by
  sorry

end tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l117_117168


namespace yardage_lost_due_to_sacks_l117_117392

theorem yardage_lost_due_to_sacks 
  (throws : ℕ)
  (percent_no_throw : ℝ)
  (half_sack_prob : ℕ)
  (sack_pattern : ℕ → ℕ)
  (correct_answer : ℕ) : 
  throws = 80 →
  percent_no_throw = 0.30 →
  (∀ (n: ℕ), half_sack_prob = n/2) →
  (sack_pattern 1 = 3 ∧ sack_pattern 2 = 5 ∧ ∀ n, n > 2 → sack_pattern n = sack_pattern (n - 1) + 2) →
  correct_answer = 168 :=
by
  sorry

end yardage_lost_due_to_sacks_l117_117392


namespace group_total_cost_l117_117414

noncomputable def total_cost
  (num_people : Nat) 
  (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem group_total_cost (num_people := 15) (cost_per_person := 900) :
  total_cost num_people cost_per_person = 13500 :=
by
  sorry

end group_total_cost_l117_117414


namespace smaller_successive_number_l117_117905

noncomputable def solve_successive_numbers : ℕ :=
  let n := 51
  n

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 2652) : n = solve_successive_numbers :=
  sorry

end smaller_successive_number_l117_117905


namespace find_a_l117_117118

-- Definitions matching the conditions
def seq (a b c d : ℤ) := [a, b, c, d, 0, 1, 1, 2, 3, 5, 8]

-- Conditions provided in the problem
def fib_property (a b c d : ℤ) : Prop :=
    d + 0 = 1 ∧ 
    c + 1 = 0 ∧ 
    b + (-1) = 1 ∧ 
    a + 2 = -1

-- Theorem statement to prove
theorem find_a (a b c d : ℤ) (h : fib_property a b c d) : a = -3 :=
by
  sorry

end find_a_l117_117118


namespace swimming_speed_l117_117966

theorem swimming_speed (s v : ℝ) (h_s : s = 4) (h_time : 1 / (v - s) = 2 * (1 / (v + s))) : v = 12 := 
by
  sorry

end swimming_speed_l117_117966


namespace number_of_packages_sold_l117_117554

noncomputable def supplier_charges (P : ℕ) : ℕ :=
  if P ≤ 10 then 25 * P
  else 250 + 20 * (P - 10)

theorem number_of_packages_sold
  (supplier_received : ℕ)
  (percent_to_X : ℕ)
  (percent_to_Y : ℕ)
  (percent_to_Z : ℕ)
  (per_package_price : ℕ)
  (discount_percent : ℕ)
  (discount_threshold : ℕ)
  (P : ℕ)
  (h_received : supplier_received = 1340)
  (h_to_X : percent_to_X = 15)
  (h_to_Y : percent_to_Y = 15)
  (h_to_Z : percent_to_Z = 70)
  (h_full_price : per_package_price = 25)
  (h_discount : discount_percent = 4 * per_package_price / 5)
  (h_threshold : discount_threshold = 10)
  (h_calculation : supplier_charges P = supplier_received) : P = 65 := 
sorry

end number_of_packages_sold_l117_117554


namespace find_m_from_parallel_vectors_l117_117427

variables (m : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

-- The condition that vectors a and b are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Given that a and b are parallel, prove that m = -3/2
theorem find_m_from_parallel_vectors
  (h : vectors_parallel (1, m) (2, -3)) :
  m = -3 / 2 :=
sorry

end find_m_from_parallel_vectors_l117_117427


namespace prime_add_eq_2001_l117_117068

theorem prime_add_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) : a + b = 2001 :=
sorry

end prime_add_eq_2001_l117_117068


namespace equation_contains_2020_l117_117502

def first_term (n : Nat) : Nat :=
  2 * n^2

theorem equation_contains_2020 :
  ∃ n, first_term n = 2020 :=
by
  use 31
  sorry

end equation_contains_2020_l117_117502


namespace calculate_total_cost_l117_117425

-- Define the cost per workbook
def cost_per_workbook (x : ℝ) : ℝ := x

-- Define the number of workbooks
def number_of_workbooks : ℝ := 400

-- Define the total cost calculation
def total_cost (x : ℝ) : ℝ := number_of_workbooks * cost_per_workbook x

-- State the theorem to prove
theorem calculate_total_cost (x : ℝ) : total_cost x = 400 * x :=
by sorry

end calculate_total_cost_l117_117425


namespace ball_distribution_l117_117997

theorem ball_distribution (n m : Nat) (h_n : n = 6) (h_m : m = 2) : 
  ∃ ways, 
    (ways = 2 ^ n - (1 + n)) ∧ ways = 57 :=
by
  sorry

end ball_distribution_l117_117997


namespace lorelai_jellybeans_correct_l117_117128

-- Define the number of jellybeans Gigi has
def gigi_jellybeans : Nat := 15

-- Define the number of additional jellybeans Rory has compared to Gigi
def rory_additional_jellybeans : Nat := 30

-- Define the number of jellybeans both girls together have
def total_jellybeans : Nat := gigi_jellybeans + (gigi_jellybeans + rory_additional_jellybeans)

-- Define the number of jellybeans Lorelai has eaten
def lorelai_jellybeans : Nat := 3 * total_jellybeans

-- The theorem to prove the number of jellybeans Lorelai has eaten is 180
theorem lorelai_jellybeans_correct : lorelai_jellybeans = 180 := by
  sorry

end lorelai_jellybeans_correct_l117_117128


namespace arithmetic_seq_S11_l117_117178

def Sn (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1)) / 2 * d

theorem arithmetic_seq_S11 (a₁ d : ℤ)
  (h1 : a₁ = -11)
  (h2 : (Sn 10 a₁ d) / 10 - (Sn 8 a₁ d) / 8 = 2) :
  Sn 11 a₁ d = -11 :=
by
  sorry

end arithmetic_seq_S11_l117_117178


namespace log_property_l117_117568

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem log_property (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : f (m * n) = f m + f n :=
by
  sorry

end log_property_l117_117568


namespace trigonometric_expression_value_l117_117489

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l117_117489


namespace sequence_a_n_term_l117_117697

theorem sequence_a_n_term :
  ∃ a : ℕ → ℕ, 
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1) = 2 * a n + 1) ∧
  a 10 = 1023 := by
  sorry

end sequence_a_n_term_l117_117697


namespace days_to_finish_by_b_l117_117225

theorem days_to_finish_by_b (A B C : ℚ) 
  (h1 : A + B + C = 1 / 5) 
  (h2 : A = 1 / 9) 
  (h3 : A + C = 1 / 7) : 
  1 / B = 12.115 :=
by
  sorry

end days_to_finish_by_b_l117_117225


namespace max_n_minus_m_l117_117506

/-- The function defined with given parameters. -/
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem max_n_minus_m (a b : ℝ) (h1 : -a / 2 = 1)
    (h2 : ∀ x, f x a b ≥ 2)
    (h3 : ∃ m n, (∀ x, f x a b ≤ 6 → m ≤ x ∧ x ≤ n) ∧ (n = 3 ∧ m = -1)) : 
    (∀ m n, (m ≤ n) → (n - m ≤ 4)) :=
by sorry

end max_n_minus_m_l117_117506


namespace driving_time_ratio_l117_117707

theorem driving_time_ratio 
  (t : ℝ)
  (h : 30 * t + 60 * (2 * t) = 75) : 
  t / (2 * t) = 1 / 2 := 
by
  sorry

end driving_time_ratio_l117_117707


namespace books_not_sold_l117_117788

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold_l117_117788


namespace percentage_students_qualified_school_A_l117_117968

theorem percentage_students_qualified_school_A 
  (A Q : ℝ)
  (h1 : 1.20 * A = A + 0.20 * A)
  (h2 : 1.50 * Q = Q + 0.50 * Q)
  (h3 : (1.50 * Q / 1.20 * A) * 100 = 87.5) :
  (Q / A) * 100 = 58.33 := sorry

end percentage_students_qualified_school_A_l117_117968


namespace range_of_a_l117_117641

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem range_of_a (a : ℝ) :
  (∀ (b : ℝ), (b ≤ 0) → ∀ (x : ℝ), (x > Real.exp 1 ∧ x ≤ Real.exp 2) → f a b x ≥ x) →
  a ≥ Real.exp 2 / 2 :=
by
  sorry

end range_of_a_l117_117641


namespace shopkeeper_loss_l117_117404

theorem shopkeeper_loss
    (total_stock : ℝ)
    (stock_sold_profit_percent : ℝ)
    (stock_profit_percent : ℝ)
    (stock_sold_loss_percent : ℝ)
    (stock_loss_percent : ℝ) :
    total_stock = 12500 →
    stock_sold_profit_percent = 0.20 →
    stock_profit_percent = 0.10 →
    stock_sold_loss_percent = 0.80 →
    stock_loss_percent = 0.05 →
    ∃ loss_amount, loss_amount = 250 :=
by
  sorry

end shopkeeper_loss_l117_117404


namespace probability_of_red_ball_l117_117871

noncomputable def total_balls : Nat := 4 + 2
noncomputable def red_balls : Nat := 2

theorem probability_of_red_ball :
  (red_balls : ℚ) / (total_balls : ℚ) = 1 / 3 :=
sorry

end probability_of_red_ball_l117_117871


namespace find_a_plus_b_l117_117231

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 2 = a - b / 2) 
  (h2 : 6 = a - b / 3) : 
  a + b = 38 := by
  sorry

end find_a_plus_b_l117_117231


namespace total_handshakes_l117_117666

def total_people := 40
def group_x_people := 25
def group_x_known_others := 5
def group_y_people := 15
def handshakes_between_x_y := group_x_people * group_y_people
def handshakes_within_x := 25 * (25 - 1 - 5) / 2
def handshakes_within_y := (15 * (15 - 1)) / 2

theorem total_handshakes 
    (h1 : total_people = 40)
    (h2 : group_x_people = 25)
    (h3 : group_x_known_others = 5)
    (h4 : group_y_people = 15) :
    handshakes_between_x_y + handshakes_within_x + handshakes_within_y = 717 := 
by
  sorry

end total_handshakes_l117_117666


namespace find_A_l117_117282

def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

theorem find_A (A : ℝ) (h : diamond A 5 = 82) : A = 12 :=
by
  unfold diamond at h
  sorry

end find_A_l117_117282


namespace volume_of_cube_is_correct_l117_117586

-- Define necessary constants and conditions
def cost_in_paise : ℕ := 34398
def rate_per_sq_cm : ℕ := 13
def surface_area : ℕ := cost_in_paise / rate_per_sq_cm
def face_area : ℕ := surface_area / 6
def side_length : ℕ := Nat.sqrt face_area
def volume : ℕ := side_length ^ 3

-- Prove the volume of the cube
theorem volume_of_cube_is_correct : volume = 9261 := by
  -- Using given conditions and basic arithmetic 
  sorry

end volume_of_cube_is_correct_l117_117586


namespace penny_makes_total_revenue_l117_117051

def price_per_slice : ℕ := 7
def slices_per_pie : ℕ := 6
def pies_sold : ℕ := 7

theorem penny_makes_total_revenue :
  (pies_sold * slices_per_pie) * price_per_slice = 294 := by
  sorry

end penny_makes_total_revenue_l117_117051


namespace chord_length_l117_117053

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l117_117053


namespace volleyball_tournament_l117_117665

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end volleyball_tournament_l117_117665


namespace no_real_roots_of_quadratic_l117_117060

theorem no_real_roots_of_quadratic 
  (a b c : ℝ) 
  (h1 : b - a + c > 0) 
  (h2 : b + a - c > 0) 
  (h3 : b - a - c < 0) 
  (h4 : b + a + c > 0) 
  (x : ℝ) : ¬ ∃ x : ℝ, a^2 * x^2 + (b^2 - a^2 - c^2) * x + c^2 = 0 := 
by
  sorry

end no_real_roots_of_quadratic_l117_117060


namespace machine_work_rates_l117_117189

theorem machine_work_rates :
  (∃ x : ℝ, (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2)) = 1 / x ∧ x = 1 / 2) :=
by
  sorry

end machine_work_rates_l117_117189


namespace volume_of_prism_l117_117898

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 60)
                                     (h2 : y * z = 75)
                                     (h3 : x * z = 100) :
  x * y * z = 671 :=
by
  sorry

end volume_of_prism_l117_117898


namespace exists_integer_n_tangent_l117_117875
open Real

noncomputable def degree_to_radian (d : ℝ) : ℝ :=
  d * (π / 180)

theorem exists_integer_n_tangent :
  ∃ (n : ℤ), -90 < (n : ℝ) ∧ (n : ℝ) < 90 ∧ tan (degree_to_radian (n : ℝ)) = tan (degree_to_radian 345) ∧ n = -15 :=
by
  sorry

end exists_integer_n_tangent_l117_117875


namespace quadrilateral_correct_choice_l117_117896

/-- Define the triangle inequality theorem for four line segments.
    A quadrilateral can be formed if for any:
    - The sum of the lengths of any three segments is greater than the length of the fourth segment.
-/
def is_quadrilateral (a b c d : ℕ) : Prop :=
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a)

/-- Determine which set of three line segments can form a quadrilateral with a fourth line segment of length 5.
    We prove that the correct choice is the set (3, 3, 3). --/
theorem quadrilateral_correct_choice :
  is_quadrilateral 3 3 3 5 ∧  ¬ is_quadrilateral 1 1 1 5 ∧  ¬ is_quadrilateral 1 1 8 5 ∧  ¬ is_quadrilateral 1 2 2 5 :=
by
  sorry

end quadrilateral_correct_choice_l117_117896


namespace arithmetic_sequence_sum_square_l117_117130

theorem arithmetic_sequence_sum_square (a d : ℕ) :
  (∀ n : ℕ, ∃ k : ℕ, n * (a + (n-1) * d / 2) = k * k) ↔ (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) := 
by
  sorry

end arithmetic_sequence_sum_square_l117_117130


namespace coins_in_distinct_colors_l117_117115

theorem coins_in_distinct_colors 
  (n : ℕ)  (h1 : 1 < n) (h2 : n < 2010) : (∃ k : ℕ, 2010 = n * k) ↔ 
  ∀ i : ℕ, i < 2010 → (∃ f : ℕ → ℕ, ∀ j : ℕ, j < n → f (j + i) % n = j % n) :=
sorry

end coins_in_distinct_colors_l117_117115


namespace sum_of_exponents_l117_117551

theorem sum_of_exponents (n : ℕ) (h : n = 896) : 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 2^a + 2^b + 2^c = n ∧ a + b + c = 24 :=
by
  sorry

end sum_of_exponents_l117_117551


namespace product_value_l117_117985

theorem product_value : 
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 * (1 / 512) * 1024 = 32 := 
by
  sorry

end product_value_l117_117985


namespace tan_20_add_4sin_20_eq_sqrt3_l117_117630

theorem tan_20_add_4sin_20_eq_sqrt3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end tan_20_add_4sin_20_eq_sqrt3_l117_117630


namespace findYearsForTwiceAge_l117_117642

def fatherSonAges : ℕ := 33

def fatherAge : ℕ := fatherSonAges + 35

def yearsForTwiceAge (x : ℕ) : Prop :=
  fatherAge + x = 2 * (fatherSonAges + x)

theorem findYearsForTwiceAge : ∃ x, yearsForTwiceAge x :=
  ⟨2, sorry⟩

end findYearsForTwiceAge_l117_117642


namespace functional_equation_solution_l117_117354

noncomputable def f : ℕ → ℕ := sorry

theorem functional_equation_solution (f : ℕ → ℕ)
    (h : ∀ n : ℕ, f (f (f n)) + f (f n) + f n = 3 * n) :
    ∀ n : ℕ, f n = n := sorry

end functional_equation_solution_l117_117354


namespace xiao_yu_reading_days_l117_117667

-- Definition of Xiao Yu's reading problem
def number_of_pages_per_day := 15
def total_number_of_days := 24
def additional_pages_per_day := 3
def new_number_of_pages_per_day := number_of_pages_per_day + additional_pages_per_day
def total_pages := number_of_pages_per_day * total_number_of_days
def new_total_number_of_days := total_pages / new_number_of_pages_per_day

-- Theorem statement in Lean 4
theorem xiao_yu_reading_days : new_total_number_of_days = 20 :=
  sorry

end xiao_yu_reading_days_l117_117667


namespace car_distance_in_45_minutes_l117_117936

theorem car_distance_in_45_minutes
  (train_speed : ℝ)
  (car_speed_ratio : ℝ)
  (time_minutes : ℝ)
  (h_train_speed : train_speed = 90)
  (h_car_speed_ratio : car_speed_ratio = 5 / 6)
  (h_time_minutes : time_minutes = 45) :
  ∃ d : ℝ, d = 56.25 ∧ d = (car_speed_ratio * train_speed) * (time_minutes / 60) :=
by
  sorry

end car_distance_in_45_minutes_l117_117936


namespace time_spent_giving_bath_l117_117779

theorem time_spent_giving_bath
  (total_time : ℕ)
  (walk_time : ℕ)
  (bath_time blowdry_time : ℕ)
  (walk_distance walk_speed : ℤ)
  (walk_distance_eq : walk_distance = 3)
  (walk_speed_eq : walk_speed = 6)
  (total_time_eq : total_time = 60)
  (walk_time_eq : walk_time = (walk_distance * 60 / walk_speed))
  (half_blowdry_time : blowdry_time = bath_time / 2)
  (time_eq : bath_time + blowdry_time = total_time - walk_time)
  : bath_time = 20 := by
  sorry

end time_spent_giving_bath_l117_117779


namespace second_discount_percentage_l117_117375

theorem second_discount_percentage (x : ℝ) :
  9356.725146198829 * 0.8 * (1 - x / 100) * 0.95 = 6400 → x = 10 :=
by
  sorry

end second_discount_percentage_l117_117375


namespace find_n_l117_117421

theorem find_n (x y m n : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) 
  (h1 : 100 * y + x = (x + y) * m) (h2 : 100 * x + y = (x + y) * n) : n = 101 - m :=
by
  sorry

end find_n_l117_117421


namespace largest_factor_and_smallest_multiple_of_18_l117_117891

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ x, (x ∈ {d : ℕ | d ∣ 18}) ∧ (∀ y, y ∈ {d : ℕ | d ∣ 18} → y ≤ x) ∧ x = 18)
  ∧ (∃ y, (y ∈ {m : ℕ | 18 ∣ m}) ∧ (∀ z, z ∈ {m : ℕ | 18 ∣ m} → y ≤ z) ∧ y = 18) :=
by
  sorry

end largest_factor_and_smallest_multiple_of_18_l117_117891


namespace range_of_a_l117_117448

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * a * x + a + 2 ≤ 0 → 1 ≤ x ∧ x ≤ 4) ↔ a ∈ Set.Ioo (-1 : ℝ) (18 / 7) ∨ a = 18 / 7 := 
by
  sorry

end range_of_a_l117_117448


namespace coordinates_with_respect_to_origin_l117_117912

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end coordinates_with_respect_to_origin_l117_117912


namespace important_emails_l117_117597

theorem important_emails (total_emails : ℕ) (spam_frac : ℚ) (promotional_frac : ℚ) (spam_email_count : ℕ) (remaining_emails : ℕ) (promotional_email_count : ℕ) (important_email_count : ℕ) :
  total_emails = 800 ∧ spam_frac = 3 / 7 ∧ promotional_frac = 5 / 11 ∧ spam_email_count = 343 ∧ remaining_emails = 457 ∧ promotional_email_count = 208 →
sorry

end important_emails_l117_117597


namespace tangent_line_parabola_l117_117293

theorem tangent_line_parabola (d : ℝ) :
  (∃ (f g : ℝ → ℝ), (∀ x y, y = f x ↔ y = 3 * x + d) ∧ (∀ x y, y = g x ↔ y ^ 2 = 12 * x)
  ∧ (∀ x y, y = f x ∧ y = g x → y = 3 * x + d ∧ y ^ 2 = 12 * x )) →
  d = 1 :=
sorry

end tangent_line_parabola_l117_117293


namespace fractions_with_smallest_difference_l117_117913

theorem fractions_with_smallest_difference 
    (x y : ℤ) 
    (f1 : ℚ := (x : ℚ) / 8) 
    (f2 : ℚ := (y : ℚ) / 13) 
    (h : abs (13 * x - 8 * y) = 1): 
    (f1 ≠ f2) ∧ abs ((x : ℚ) / 8 - (y : ℚ) / 13) = 1 / 104 :=
by
  sorry

end fractions_with_smallest_difference_l117_117913


namespace max_f_value_l117_117541

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin x + 12 * Real.cos x

theorem max_f_value : ∃ x : ℝ, f x = 13 :=
sorry

end max_f_value_l117_117541


namespace rhombus_perimeter_l117_117459

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  (4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))) = 52 := by
  sorry

end rhombus_perimeter_l117_117459


namespace ratio_of_pieces_l117_117838

theorem ratio_of_pieces (total_length shorter_piece longer_piece : ℕ) 
    (h1 : total_length = 6) (h2 : shorter_piece = 2)
    (h3 : longer_piece = total_length - shorter_piece) :
    ((longer_piece : ℚ) / (shorter_piece : ℚ)) = 2 :=
by
    sorry

end ratio_of_pieces_l117_117838


namespace combined_age_l117_117331

-- Conditions as definitions
def AmyAge (j : ℕ) : ℕ :=
  j / 3

def ChrisAge (a : ℕ) : ℕ :=
  2 * a

-- Given condition
def JeremyAge : ℕ := 66

-- Question to prove
theorem combined_age : 
  let j := JeremyAge
  let a := AmyAge j
  let c := ChrisAge a
  a + j + c = 132 :=
by
  sorry

end combined_age_l117_117331


namespace quadratic_inequality_range_l117_117988

theorem quadratic_inequality_range (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + a - 1 < 0) → a ≤ 0 :=
sorry

end quadratic_inequality_range_l117_117988


namespace arithmetic_sequence_8th_term_l117_117928

theorem arithmetic_sequence_8th_term (a d : ℤ) 
  (h1 : a + 3 * d = 23)
  (h2 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by 
  sorry

end arithmetic_sequence_8th_term_l117_117928


namespace infinite_solutions_abs_eq_ax_minus_2_l117_117460

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end infinite_solutions_abs_eq_ax_minus_2_l117_117460


namespace n_is_one_sixth_sum_of_list_l117_117117

-- Define the condition that n is 4 times the average of the other 20 numbers
def satisfies_condition (n : ℝ) (l : List ℝ) : Prop :=
  l.length = 21 ∧
  n ∈ l ∧
  n = 4 * (l.erase n).sum / 20

-- State the main theorem
theorem n_is_one_sixth_sum_of_list {n : ℝ} {l : List ℝ} (h : satisfies_condition n l) :
  n = (1 / 6) * l.sum :=
by
  sorry

end n_is_one_sixth_sum_of_list_l117_117117


namespace simple_interest_rate_l117_117469

theorem simple_interest_rate (P R: ℝ) (T: ℝ) (H: T = 5) (H1: P * (1/6) = P * (R * T / 100)) : R = 10/3 :=
by {
  sorry
}

end simple_interest_rate_l117_117469


namespace max_value_abs_cube_sum_l117_117106

theorem max_value_abs_cube_sum (x : Fin 5 → ℝ) (h : ∀ i, 0 ≤ x i ∧ x i ≤ 1) : 
  (|x 0 - x 1|^3 + |x 1 - x 2|^3 + |x 2 - x 3|^3 + |x 3 - x 4|^3 + |x 4 - x 0|^3) ≤ 4 :=
sorry

end max_value_abs_cube_sum_l117_117106


namespace complement_of_M_in_U_is_correct_l117_117266

def U : Set ℤ := {1, -2, 3, -4, 5, -6}
def M : Set ℤ := {1, -2, 3, -4}
def complement_M_in_U : Set ℤ := {5, -6}

theorem complement_of_M_in_U_is_correct : (U \ M) = complement_M_in_U := by
  sorry

end complement_of_M_in_U_is_correct_l117_117266


namespace value_range_of_quadratic_l117_117081

noncomputable def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_range_of_quadratic :
  ∀ x, -1 ≤ x ∧ x ≤ 2 → (2 : ℝ) ≤ quadratic_function x ∧ quadratic_function x ≤ 6 :=
by
  sorry

end value_range_of_quadratic_l117_117081


namespace coordinates_of_a_l117_117444

theorem coordinates_of_a
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (1, 2))
  (h1 : (a.1)^2 + (a.2)^2 = 5)
  (h2 : ∃ k : ℝ, a = (k, 2 * k))
  : a = (1, 2) ∨ a = (-1, -2) :=
  sorry

end coordinates_of_a_l117_117444


namespace brady_june_hours_l117_117577

variable (x : ℕ) -- Number of hours worked every day in June

def hoursApril : ℕ := 6 * 30 -- Total hours in April
def hoursSeptember : ℕ := 8 * 30 -- Total hours in September
def hoursJune (x : ℕ) : ℕ := x * 30 -- Total hours in June
def totalHours (x : ℕ) : ℕ := hoursApril + hoursJune x + hoursSeptember -- Total hours over three months
def averageHours (x : ℕ) : ℕ := totalHours x / 3 -- Average hours per month

theorem brady_june_hours (h : averageHours x = 190) : x = 5 :=
by
  sorry

end brady_june_hours_l117_117577


namespace cubic_identity_l117_117524

theorem cubic_identity (x : ℝ) (h : x - (1 / x) = 5) : x^3 - (1 / x^3) = 140 := 
  sorry

end cubic_identity_l117_117524


namespace total_amount_l117_117801

def mark_dollars : ℚ := 5 / 8
def carolyn_dollars : ℚ := 2 / 5
def total_dollars : ℚ := mark_dollars + carolyn_dollars

theorem total_amount : total_dollars = 1.025 := by
  sorry

end total_amount_l117_117801


namespace min_value_x1_x2_l117_117889

theorem min_value_x1_x2 (a x_1 x_2 : ℝ) (h_a_pos : 0 < a) (h_sol_set : x_1 + x_2 = 4 * a) (h_prod_set : x_1 * x_2 = 3 * a^2) : 
  x_1 + x_2 + a / (x_1 * x_2) = 4 * a + 1 / (3 * a) :=
sorry

end min_value_x1_x2_l117_117889


namespace max_pies_without_ingredients_l117_117125

theorem max_pies_without_ingredients :
  let total_pies := 48
  let chocolate_pies := total_pies / 3
  let marshmallow_pies := total_pies / 2
  let cayenne_pies := 3 * total_pies / 8
  let soy_nut_pies := total_pies / 8
  total_pies - max chocolate_pies (max marshmallow_pies (max cayenne_pies soy_nut_pies)) = 24 := by
{
  sorry
}

end max_pies_without_ingredients_l117_117125


namespace larger_number_of_two_l117_117955

theorem larger_number_of_two
  (HCF : ℕ)
  (factor1 : ℕ)
  (factor2 : ℕ)
  (cond_HCF : HCF = 23)
  (cond_factor1 : factor1 = 15)
  (cond_factor2 : factor2 = 16) :
  ∃ (A : ℕ), A = 23 * 16 := by
  sorry

end larger_number_of_two_l117_117955


namespace measured_diagonal_in_quadrilateral_l117_117145

-- Defining the conditions (side lengths and diagonals)
def valid_diagonal (side1 side2 side3 side4 diagonal : ℝ) : Prop :=
  side1 + side2 > diagonal ∧ side1 + side3 > diagonal ∧ side1 + side4 > diagonal ∧ 
  side2 + side3 > diagonal ∧ side2 + side4 > diagonal ∧ side3 + side4 > diagonal

theorem measured_diagonal_in_quadrilateral :
  let sides := [1, 2, 2.8, 5]
  let diagonal1 := 7.5
  let diagonal2 := 2.8
  (valid_diagonal 1 2 2.8 5 diagonal2) :=
sorry

end measured_diagonal_in_quadrilateral_l117_117145


namespace angle_CAD_l117_117175

noncomputable def angle_arc (degree: ℝ) (minute: ℝ) : ℝ :=
  degree + minute / 60

theorem angle_CAD :
  angle_arc 117 23 / 2 + angle_arc 42 37 / 2 = 80 :=
by
  sorry

end angle_CAD_l117_117175


namespace find_c_l117_117979

theorem find_c (a b c : ℝ) (h1 : a * 2 = 3 * b / 2) (h2 : a * 2 + 9 = c) (h3 : 4 - 3 * b = -c) : 
  c = 12 :=
by
  sorry

end find_c_l117_117979


namespace paper_pattern_after_unfolding_l117_117387

-- Define the number of layers after folding the square paper four times
def folded_layers (initial_layers : ℕ) : ℕ :=
  initial_layers * 2 ^ 4

-- Define the number of quarter-circles removed based on the layers
def quarter_circles_removed (layers : ℕ) : ℕ :=
  layers

-- Define the number of complete circles from the quarter circles
def complete_circles (quarter_circles : ℕ) : ℕ :=
  quarter_circles / 4

-- The main theorem that we need to prove
theorem paper_pattern_after_unfolding :
  (complete_circles (quarter_circles_removed (folded_layers 1)) = 4) :=
by
  sorry

end paper_pattern_after_unfolding_l117_117387


namespace sin_alpha_beta_gamma_values_l117_117552

open Real

theorem sin_alpha_beta_gamma_values (α β γ : ℝ)
  (h1 : sin α = sin (α + β + γ) + 1)
  (h2 : sin β = 3 * sin (α + β + γ) + 2)
  (h3 : sin γ = 5 * sin (α + β + γ) + 3) :
  sin α * sin β * sin γ = (3/64) ∨ sin α * sin β * sin γ = (1/8) :=
sorry

end sin_alpha_beta_gamma_values_l117_117552


namespace solve_for_x_l117_117201

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x_l117_117201


namespace solve_eq_l117_117348

theorem solve_eq : ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 ↔
  x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2 :=
by 
  intro x
  sorry

end solve_eq_l117_117348


namespace inequality_holds_for_all_x_l117_117614

theorem inequality_holds_for_all_x (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
by
  sorry

end inequality_holds_for_all_x_l117_117614


namespace find_natural_number_l117_117959

theorem find_natural_number :
  ∃ x : ℕ, (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → d2 - d1 = 4) ∧
           (∀ d1 d2 : ℕ, d1 ∣ x → d2 ∣ x → d1 < d2 → x - d2 = 308) ∧
           x = 385 :=
by
  sorry

end find_natural_number_l117_117959


namespace product_evaluation_l117_117852

noncomputable def product_term (n : ℕ) : ℚ :=
  1 - (1 / (n * n))

noncomputable def product_expression : ℚ :=
  10 * 71 * (product_term 2) * (product_term 3) * (product_term 4) * (product_term 5) *
  (product_term 6) * (product_term 7) * (product_term 8) * (product_term 9) * (product_term 10)

theorem product_evaluation : product_expression = 71 := by
  sorry

end product_evaluation_l117_117852


namespace problem1_problem2_l117_117887

-- Define the given angle
def given_angle (α : ℝ) : Prop := α = 2010

-- Define the theorem for the first problem
theorem problem1 (α : ℝ) (k : ℤ) (β : ℝ) (h₁ : given_angle α) 
  (h₂ : 0 ≤ β ∧ β < 360) (h₃ : α = k * 360 + β) : 
  -- Assert that α is in the third quadrant
  (190 ≤ β ∧ β < 270 → true) :=
sorry

-- Define the theorem for the second problem
theorem problem2 (α : ℝ) (θ : ℝ) (h₁ : given_angle α)
  (h₂ : -360 ≤ θ ∧ θ < 720)
  (h₃ : ∃ k : ℤ, θ = α + k * 360) : 
  θ = -150 ∨ θ = 210 ∨ θ = 570 :=
sorry

end problem1_problem2_l117_117887


namespace subway_train_speed_l117_117647

open Nat

-- Define the speed function
def speed (s : ℕ) : ℕ := s^2 + 2*s

-- Define the theorem to be proved
theorem subway_train_speed (t : ℕ) (ht : 0 ≤ t ∧ t ≤ 7) (h_speed : speed 7 - speed t = 28) : t = 5 :=
by
  sorry

end subway_train_speed_l117_117647


namespace tetrahedron_volume_l117_117695

theorem tetrahedron_volume 
  (R S₁ S₂ S₃ S₄ : ℝ) : 
  V = R * (S₁ + S₂ + S₃ + S₄) :=
sorry

end tetrahedron_volume_l117_117695


namespace min_red_hair_students_l117_117415

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l117_117415


namespace probability_same_color_probability_different_color_l117_117290

def count_combinations {α : Type*} (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def count_ways_same_color : ℕ :=
  (count_combinations (Finset.range 3) 2) * 2

noncomputable def count_ways_diff_color : ℕ :=
  (Finset.range 3).card * (Finset.range 3).card

noncomputable def total_ways : ℕ :=
  count_combinations (Finset.range 6) 2

noncomputable def prob_same_color : ℚ :=
  count_ways_same_color / total_ways

noncomputable def prob_diff_color : ℚ :=
  count_ways_diff_color / total_ways

theorem probability_same_color :
  prob_same_color = 2 / 5 := by
  sorry

theorem probability_different_color :
  prob_diff_color = 3 / 5 := by
  sorry

end probability_same_color_probability_different_color_l117_117290


namespace proof_problem_l117_117921

open Real

-- Define the problem statements as Lean hypotheses
def p : Prop := ∀ a : ℝ, exp a ≥ a + 1
def q : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

theorem proof_problem : p ∧ q :=
by
  sorry

end proof_problem_l117_117921


namespace find_some_number_l117_117825

theorem find_some_number (a : ℕ) (h1 : a = 105) (h2 : a^3 = some_number * 35 * 45 * 35) : some_number = 1 := by
  sorry

end find_some_number_l117_117825


namespace find_d_squared_l117_117476

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * Complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : ∀ z : ℂ, Complex.abs (g z c d - z) = 2 * Complex.abs (g z c d)) (h2 : Complex.abs (c + d * Complex.I) = 6) : d^2 = 11305 / 4 := 
sorry

end find_d_squared_l117_117476


namespace simplify_expression_l117_117488

theorem simplify_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_condition : a^3 + b^3 = 3 * (a + b)) : 
  (a / b + b / a + 1 / (a * b) = 4 / (a * b) + 1) :=
by
  sorry

end simplify_expression_l117_117488


namespace least_number_to_make_divisible_l117_117012

theorem least_number_to_make_divisible (k : ℕ) (h : 1202 + k = 1204) : (2 ∣ 1204) := 
by
  sorry

end least_number_to_make_divisible_l117_117012


namespace cos_A_value_l117_117975

theorem cos_A_value (A B C : ℝ) 
  (A_internal : A + B + C = Real.pi) 
  (cos_B : Real.cos B = 1 / 2)
  (sin_C : Real.sin C = 3 / 5) : 
  Real.cos A = (3 * Real.sqrt 3 - 4) / 10 := 
by
  sorry

end cos_A_value_l117_117975


namespace evaluate_expression_l117_117388

theorem evaluate_expression : 2^(Real.log 5 / Real.log 2) + Real.log 25 / Real.log 5 = 7 := by
  sorry

end evaluate_expression_l117_117388


namespace moles_of_water_from_reaction_l117_117976

def moles_of_water_formed (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  nh4cl_moles -- Because 1:1 ratio of reactants producing water

theorem moles_of_water_from_reaction :
  moles_of_water_formed 3 3 = 3 := by
  -- Use the condition of the 1:1 reaction ratio derivable from the problem's setup.
  sorry

end moles_of_water_from_reaction_l117_117976


namespace range_of_a_l117_117450

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end range_of_a_l117_117450


namespace oula_deliveries_count_l117_117330

-- Define the conditions for the problem
def num_deliveries_Oula (O : ℕ) (T : ℕ) : Prop :=
  T = (3 / 4 : ℚ) * O ∧ (100 * O - 100 * T = 2400)

-- Define the theorem we want to prove
theorem oula_deliveries_count : ∃ (O : ℕ), ∃ (T : ℕ), num_deliveries_Oula O T ∧ O = 96 :=
sorry

end oula_deliveries_count_l117_117330


namespace coupon_value_l117_117403

theorem coupon_value (C : ℝ) (original_price : ℝ := 120) (final_price : ℝ := 99) 
(membership_discount : ℝ := 0.1) (reduced_price : ℝ := original_price - C) :
0.9 * reduced_price = final_price → C = 10 :=
by sorry

end coupon_value_l117_117403


namespace max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l117_117660

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem max_value_of_f : ∃ x, (f x) = 1/2 :=
sorry

theorem period_of_f : ∀ x, f (x + π) = f x :=
sorry

theorem not_monotonically_increasing : ¬ ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y :=
sorry

theorem incorrect_zeros : ∃ x y z, (0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ π) ∧ (f x = 0 ∧ f y = 0 ∧ f z = 0) :=
sorry

end max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l117_117660


namespace evaluation_expression_l117_117251

theorem evaluation_expression (a b c d : ℝ) 
  (h1 : a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h2 : b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h3 : c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6)
  (h4 : d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 6) :
  (1/a + 1/b + 1/c + 1/d)^2 = (16 * (11 + 2 * Real.sqrt 30)) / ((11 + 2 * Real.sqrt 30 - 3 * Real.sqrt 6)^2) :=
sorry

end evaluation_expression_l117_117251


namespace initial_bowls_eq_70_l117_117327

def customers : ℕ := 20
def bowls_per_customer : ℕ := 20
def reward_ratio := 10
def reward_bowls := 2
def remaining_bowls : ℕ := 30

theorem initial_bowls_eq_70 :
  let rewards_per_customer := (bowls_per_customer / reward_ratio) * reward_bowls
  let total_rewards := (customers / 2) * rewards_per_customer
  (remaining_bowls + total_rewards) = 70 :=
by
  sorry

end initial_bowls_eq_70_l117_117327


namespace roots_polynomial_pq_sum_l117_117110

theorem roots_polynomial_pq_sum :
  ∀ p q : ℝ, 
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 3) * (x - 4) = x^4 - 10 * x^3 + p * x^2 - q * x + 24) 
  → p + q = 85 :=
by 
  sorry

end roots_polynomial_pq_sum_l117_117110


namespace find_a_l117_117302

theorem find_a (a : ℝ)
  (hl : ∀ x y : ℝ, ax + 2 * y - a - 2 = 0)
  (hm : ∀ x y : ℝ, 2 * x - y = 0)
  (perpendicular : ∀ x y : ℝ, (2 * - (a / 2)) = -1) : 
  a = 1 := sorry

end find_a_l117_117302


namespace green_socks_count_l117_117519

theorem green_socks_count: 
  ∀ (total_socks : ℕ) (white_socks : ℕ) (blue_socks : ℕ) (red_socks : ℕ) (green_socks : ℕ),
  total_socks = 900 →
  white_socks = total_socks / 3 →
  blue_socks = total_socks / 4 →
  red_socks = total_socks / 5 →
  green_socks = total_socks - (white_socks + blue_socks + red_socks) →
  green_socks = 195 :=
by
  intros total_socks white_socks blue_socks red_socks green_socks
  sorry

end green_socks_count_l117_117519


namespace function_equivalence_l117_117528

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = 2020) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (-y) = -g y) ∧ (∀ x : ℝ, f x = g (1 - 2 * x^2) + 1010) :=
sorry

end function_equivalence_l117_117528


namespace Juanita_spends_more_l117_117758

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l117_117758


namespace find_number_added_l117_117982

theorem find_number_added (x : ℕ) : (1250 / 50) + x = 7525 ↔ x = 7500 := by
  sorry

end find_number_added_l117_117982


namespace part1_l117_117768

theorem part1 (a b c t m n : ℝ) (h1 : a > 0) (h2 : m = n) (h3 : t = (3 + (t + 1)) / 2) : t = 4 :=
sorry

end part1_l117_117768


namespace base_conversion_b_l117_117422

-- Define the problem in Lean
theorem base_conversion_b (b : ℕ) : 
  (b^2 + 2 * b - 16 = 0) → b = 4 := 
by
  intro h
  sorry

end base_conversion_b_l117_117422


namespace ratio_mark_days_used_l117_117487

-- Defining the conditions
def num_sick_days : ℕ := 10
def num_vacation_days : ℕ := 10
def total_hours_left : ℕ := 80
def hours_per_workday : ℕ := 8

-- Total days allotted
def total_days_allotted : ℕ :=
  num_sick_days + num_vacation_days

-- Days left for Mark
def days_left : ℕ :=
  total_hours_left / hours_per_workday

-- Days used by Mark
def days_used : ℕ :=
  total_days_allotted - days_left

-- The ratio of days used to total days allotted (expected to be 1:2)
def ratio_used_to_allotted : ℚ :=
  days_used / total_days_allotted

theorem ratio_mark_days_used :
  ratio_used_to_allotted = 1 / 2 :=
sorry

end ratio_mark_days_used_l117_117487


namespace students_doing_hula_hoops_l117_117818

def number_of_students_jumping_rope : ℕ := 7
def number_of_students_doing_hula_hoops : ℕ := 5 * number_of_students_jumping_rope

theorem students_doing_hula_hoops : number_of_students_doing_hula_hoops = 35 :=
by
  sorry

end students_doing_hula_hoops_l117_117818


namespace relationship_p_q_l117_117750

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem relationship_p_q (x a p q : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1)
  (hp : p = |log_a a (1 + x)|) (hq : q = |log_a a (1 - x)|) : p ≤ q :=
sorry

end relationship_p_q_l117_117750


namespace probability_of_rain_l117_117625

theorem probability_of_rain {p : ℝ} (h : p = 0.95) :
  ∃ (q : ℝ), q = (1 - p) ∧ q < p :=
by
  sorry

end probability_of_rain_l117_117625


namespace complex_expression_value_l117_117644

theorem complex_expression_value :
  ((6^2 - 4^2) + 2)^3 / 2 = 5324 :=
by
  sorry

end complex_expression_value_l117_117644


namespace min_value_reciprocal_sum_l117_117407

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  ∀ (c : ℝ), c = (1 / a) + (4 / b) → c ≥ 9 :=
by
  intros c hc
  sorry

end min_value_reciprocal_sum_l117_117407


namespace rectangle_perimeter_l117_117271

theorem rectangle_perimeter {b : ℕ → ℕ} {W H : ℕ}
  (h1 : ∀ i, b i ≠ b (i+1))
  (h2 : b 9 = W / 2)
  (h3 : gcd W H = 1)

  (h4 : b 1 + b 2 = b 3)
  (h5 : b 1 + b 3 = b 4)
  (h6 : b 3 + b 4 = b 5)
  (h7 : b 4 + b 5 = b 6)
  (h8 : b 2 + b 3 + b 5 = b 7)
  (h9 : b 2 + b 7 = b 8)
  (h10 : b 1 + b 4 + b 6 = b 9)
  (h11 : b 6 + b 9 = b 7 + b 8) : 
  2 * (W + H) = 266 :=
  sorry

end rectangle_perimeter_l117_117271


namespace commute_time_x_l117_117792

theorem commute_time_x (d : ℝ) (walk_speed : ℝ) (train_speed : ℝ) (extra_time : ℝ) (diff_time : ℝ) :
  d = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  diff_time = 10 →
  (diff_time : ℝ) * 60 = (d / walk_speed - (d / train_speed + extra_time / 60)) * 60 →
  extra_time = 15.5 :=
by
  sorry

end commute_time_x_l117_117792


namespace x_plus_inv_x_eq_8_then_power_4_l117_117328

theorem x_plus_inv_x_eq_8_then_power_4 (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end x_plus_inv_x_eq_8_then_power_4_l117_117328


namespace angle_C_of_triangle_l117_117211

theorem angle_C_of_triangle (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := 
by
  sorry

end angle_C_of_triangle_l117_117211


namespace rectangle_height_l117_117826

-- Define the given right-angled triangle with its legs and hypotenuse
variables {a b c d : ℝ}

-- Define the conditions: Right-angled triangle with legs a, b and hypotenuse c
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the height of the inscribed rectangle is d
def height_of_rectangle (a b d : ℝ) : Prop :=
  d = a + b

-- The problem statement: Prove that the height of the rectangle is the sum of the heights of the squares
theorem rectangle_height (a b c d : ℝ) (ht : right_angled_triangle a b c) : height_of_rectangle a b d :=
by
  sorry

end rectangle_height_l117_117826


namespace part_a_l117_117109

theorem part_a (a b : ℤ) (x : ℤ) :
  (x % 5 = a) ∧ (x % 6 = b) → x = 6 * a + 25 * b :=
by
  sorry

end part_a_l117_117109


namespace multiplication_correct_l117_117731

theorem multiplication_correct : 121 * 54 = 6534 := by
  sorry

end multiplication_correct_l117_117731


namespace intersection_m_n_l117_117287

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_m_n : M ∩ N = {0, 1, 2} := 
sorry

end intersection_m_n_l117_117287


namespace evaluate_expression_at_two_l117_117484

theorem evaluate_expression_at_two : (2 * (2:ℝ)^2 - 3 * 2 + 4) = 6 := by
  sorry

end evaluate_expression_at_two_l117_117484


namespace peony_total_count_l117_117583

theorem peony_total_count (n : ℕ) (x : ℕ) (total_sample : ℕ) (single_sample : ℕ) (double_sample : ℕ) (thousand_sample : ℕ) (extra_thousand : ℕ)
    (h1 : thousand_sample > single_sample)
    (h2 : thousand_sample - single_sample = extra_thousand)
    (h3 : total_sample = single_sample + double_sample + thousand_sample)
    (h4 : total_sample = 12)
    (h5 : single_sample = 4)
    (h6 : double_sample = 2)
    (h7 : thousand_sample = 6)
    (h8 : extra_thousand = 30) :
    n = 180 :=
by 
  sorry

end peony_total_count_l117_117583


namespace length_of_ae_l117_117023

theorem length_of_ae
  (a b c d e : ℝ)
  (bc : ℝ)
  (cd : ℝ)
  (de : ℝ := 8)
  (ab : ℝ := 5)
  (ac : ℝ := 11)
  (h1 : bc = 2 * cd)
  (h2 : bc = ac - ab)
  : ab + bc + cd + de = 22 := 
by
  sorry

end length_of_ae_l117_117023


namespace factorize_2070_l117_117319

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_unique_factorization (n a b : ℕ) : Prop := a * b = n ∧ is_two_digit a ∧ is_two_digit b

-- The final theorem statement
theorem factorize_2070 : 
  (∃ a b : ℕ, is_unique_factorization 2070 a b) ∧ 
  ∀ a b : ℕ, is_unique_factorization 2070 a b → (a = 30 ∧ b = 69) ∨ (a = 69 ∧ b = 30) :=
by 
  sorry

end factorize_2070_l117_117319


namespace fraction_equality_l117_117333

theorem fraction_equality : 
  (3 ^ 8 + 3 ^ 6) / (3 ^ 8 - 3 ^ 6) = 5 / 4 :=
by
  -- Expression rewrite and manipulation inside parenthesis can be ommited
  sorry

end fraction_equality_l117_117333


namespace mario_haircut_price_l117_117056

theorem mario_haircut_price (P : ℝ) 
  (weekend_multiplier : ℝ := 1.50)
  (sunday_price : ℝ := 27) 
  (weekend_price_eq : sunday_price = P * weekend_multiplier) : 
  P = 18 := 
by
  sorry

end mario_haircut_price_l117_117056


namespace compute_expression_l117_117321

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l117_117321


namespace two_digit_sum_l117_117745

theorem two_digit_sum (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100)
  (hy : 10 ≤ y ∧ y < 100) (h_rev : y = (x % 10) * 10 + x / 10)
  (h_diff_square : x^2 - y^2 = n^2) : x + y + n = 154 :=
sorry

end two_digit_sum_l117_117745


namespace exist_a_b_for_every_n_l117_117965

theorem exist_a_b_for_every_n (n : ℕ) (hn : 0 < n) : 
  ∃ (a b : ℤ), 1 < a ∧ 1 < b ∧ a^2 + 1 = 2 * b^2 ∧ (a - b) % n = 0 := 
sorry

end exist_a_b_for_every_n_l117_117965


namespace ratio_of_weights_l117_117828

noncomputable def tyler_weight (sam_weight : ℝ) : ℝ := sam_weight + 25
noncomputable def ratio_of_peter_to_tyler (peter_weight tyler_weight : ℝ) : ℝ := peter_weight / tyler_weight

theorem ratio_of_weights (sam_weight : ℝ) (peter_weight : ℝ) (h_sam : sam_weight = 105) (h_peter : peter_weight = 65) :
  ratio_of_peter_to_tyler peter_weight (tyler_weight sam_weight) = 0.5 := by
  -- We use the conditions to derive the information
  sorry

end ratio_of_weights_l117_117828


namespace g_of_minus_3_l117_117724

noncomputable def f (x : ℝ) : ℝ := 4 * x - 7
noncomputable def g (y : ℝ) : ℝ := 3 * ((y + 7) / 4) ^ 2 + 4 * ((y + 7) / 4) + 1

theorem g_of_minus_3 : g (-3) = 8 :=
by
  sorry

end g_of_minus_3_l117_117724


namespace john_shots_l117_117086

theorem john_shots :
  let initial_shots := 30
  let initial_percentage := 0.60
  let additional_shots := 10
  let final_percentage := 0.58
  let made_initial := initial_percentage * initial_shots
  let total_shots := initial_shots + additional_shots
  let made_total := final_percentage * total_shots
  let made_additional := made_total - made_initial
  made_additional = 5 :=
by
  sorry

end john_shots_l117_117086


namespace find_number_l117_117881

theorem find_number (x : ℕ) (h : (9 * x) / 3 = 27) : x = 9 :=
by
  sorry

end find_number_l117_117881


namespace problem_1_problem_2_l117_117653

theorem problem_1 (p x : ℝ) (h1 : |p| ≤ 2) (h2 : x^2 + p*x + 1 > 2*x + p) : x < -1 ∨ x > 3 :=
sorry

theorem problem_2 (p x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) (h3 : x^2 + p*x + 1 > 2*x + p) : p > -1 :=
sorry

end problem_1_problem_2_l117_117653


namespace determine_plane_by_trapezoid_legs_l117_117657

-- Defining basic objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Line := (p1 : Point) (p2 : Point)
structure Plane := (l1 : Line) (l2 : Line)

-- Theorem statement for the problem
theorem determine_plane_by_trapezoid_legs (trapezoid_legs : Line) :
  ∃ (pl : Plane), ∀ (l1 l2 : Line), (l1 = trapezoid_legs) ∧ (l2 = trapezoid_legs) → (pl = Plane.mk l1 l2) :=
sorry

end determine_plane_by_trapezoid_legs_l117_117657


namespace range_of_m_l117_117026

variable (m : ℝ)
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 := sorry

end range_of_m_l117_117026


namespace smallest_integer_n_l117_117594

theorem smallest_integer_n (n : ℕ) (h : ∃ k : ℕ, 432 * n = k ^ 2) : n = 3 := 
sorry

end smallest_integer_n_l117_117594


namespace number_is_100_l117_117718

theorem number_is_100 (x : ℝ) (h : 0.60 * (3 / 5) * x = 36) : x = 100 :=
by sorry

end number_is_100_l117_117718


namespace integral_cos8_0_2pi_l117_117850

noncomputable def definite_integral_cos8 (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.cos (x / 4)) ^ 8

theorem integral_cos8_0_2pi :
  definite_integral_cos8 0 (2 * Real.pi) = (35 * Real.pi) / 64 :=
by
  sorry

end integral_cos8_0_2pi_l117_117850


namespace subtraction_example_l117_117776

theorem subtraction_example : 2 - 3 = -1 := 
by {
  -- We need to prove that 2 - 3 = -1
  -- The proof is to be filled here
  sorry
}

end subtraction_example_l117_117776


namespace revision_cost_per_page_is_4_l117_117132

-- Definitions based on conditions
def initial_cost_per_page := 6
def total_pages := 100
def revised_once_pages := 35
def revised_twice_pages := 15
def no_revision_pages := total_pages - revised_once_pages - revised_twice_pages
def total_cost := 860

-- Theorem to be proved
theorem revision_cost_per_page_is_4 : 
  ∃ x : ℝ, 
    ((initial_cost_per_page * total_pages) + 
     (revised_once_pages * x) + 
     (revised_twice_pages * (2 * x)) = total_cost) ∧ x = 4 :=
by
  sorry

end revision_cost_per_page_is_4_l117_117132


namespace find_m_value_l117_117003

noncomputable def pyramid_property (m : ℕ) : Prop :=
  let n1 := 3
  let n2 := 9
  let n3 := 6
  let r2_1 := m + n1
  let r2_2 := n1 + n2
  let r2_3 := n2 + n3
  let r3_1 := r2_1 + r2_2
  let r3_2 := r2_2 + r2_3
  let top := r3_1 + r3_2
  top = 54

theorem find_m_value : ∃ m : ℕ, pyramid_property m ∧ m = 12 := by
  sorry

end find_m_value_l117_117003


namespace sum_of_midpoint_coordinates_l117_117438

theorem sum_of_midpoint_coordinates : 
  let (x1, y1) := (4, 7)
  let (x2, y2) := (10, 19)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 20 := sorry

end sum_of_midpoint_coordinates_l117_117438


namespace rickey_time_l117_117920

/-- Prejean's speed in a race was three-quarters that of Rickey. 
If they both took a total of 70 minutes to run the race, 
the total number of minutes that Rickey took to finish the race is 40. -/
theorem rickey_time (t : ℝ) (h1 : ∀ p : ℝ, p = (3/4) * t) (h2 : t + (3/4) * t = 70) : t = 40 := 
by
  sorry

end rickey_time_l117_117920


namespace max_value_of_reciprocal_sums_of_zeros_l117_117708

noncomputable def quadratic_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + 2 * x - 1

noncomputable def linear_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 1

theorem max_value_of_reciprocal_sums_of_zeros (k : ℝ) (x1 x2 : ℝ)
  (h0 : -1 < k ∧ k < 0)
  (hx1 : x1 ∈ Set.Ioc 0 1 → quadratic_part k x1 = 0)
  (hx2 : x2 ∈ Set.Ioi 1 → linear_part k x2 = 0)
  (hx_distinct : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = 9 / 4 :=
sorry

end max_value_of_reciprocal_sums_of_zeros_l117_117708


namespace arithmetic_sum_2015_l117_117294

-- Definitions based on problem conditions
def a1 : ℤ := -2015
def S (n : ℕ) (d : ℤ) : ℤ := n * a1 + n * (n - 1) / 2 * d
def arithmetic_sequence (n : ℕ) (d : ℤ) : ℤ := a1 + (n - 1) * d

-- Proof problem
theorem arithmetic_sum_2015 (d : ℤ) :
  2 * S 6 d - 3 * S 4 d = 24 →
  S 2015 d = -2015 :=
by
  sorry

end arithmetic_sum_2015_l117_117294


namespace find_x_l117_117362

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l117_117362


namespace quintuplets_babies_l117_117335

theorem quintuplets_babies (a b c d : ℕ) 
  (h1 : d = 2 * c) 
  (h2 : c = 3 * b) 
  (h3 : b = 2 * a) 
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1200) : 
  5 * d = 18000 / 23 :=
by 
  sorry

end quintuplets_babies_l117_117335


namespace mean_of_second_set_l117_117806

theorem mean_of_second_set (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 90) : 
  (128 + 255 + 511 + 1023 + x) / 5 = 423 :=
by
  sorry

end mean_of_second_set_l117_117806


namespace quadratic_inequality_solution_range_l117_117590

theorem quadratic_inequality_solution_range (k : ℝ) :
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0) ↔ (-3 / 2 < k ∧ k < 0) := sorry

end quadratic_inequality_solution_range_l117_117590


namespace part1_part2_l117_117823

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (1 - a) * x + (1 - a)

theorem part1 (x : ℝ) : f x 4 ≥ 7 ↔ x ≥ 5 ∨ x ≤ -2 :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, -1 < x → f x a ≥ 0) ↔ a ≤ 1 :=
sorry

end part1_part2_l117_117823


namespace red_marbles_initial_count_l117_117549

theorem red_marbles_initial_count (r g : ℕ) 
  (h1 : 3 * r = 5 * g)
  (h2 : 4 * (r - 18) = g + 27) :
  r = 29 :=
sorry

end red_marbles_initial_count_l117_117549


namespace eval_expression_solve_inequalities_l117_117539

-- Problem 1: Evaluation of the expression equals sqrt(2)
theorem eval_expression : (1 - 1^2023 + Real.sqrt 9 - (Real.pi - 3)^0 + |Real.sqrt 2 - 1|) = Real.sqrt 2 := 
by sorry

-- Problem 2: Solution set of the inequality system
theorem solve_inequalities (x : ℝ) : 
  ((3 * x + 1) / 2 ≥ (4 * x + 3) / 3 ∧ 2 * x + 7 ≥ 5 * x - 17) ↔ (3 ≤ x ∧ x ≤ 8) :=
by sorry

end eval_expression_solve_inequalities_l117_117539


namespace K_set_I_K_set_III_K_set_IV_K_set_V_l117_117218

-- Definitions for the problem conditions
def K (x y z : ℤ) : ℤ :=
  (x + 2 * y + 3 * z) * (2 * x - y - z) * (y + 2 * z + 3 * x) +
  (y + 2 * z + 3 * x) * (2 * y - z - x) * (z + 2 * x + 3 * y) +
  (z + 2 * x + 3 * y) * (2 * z - x - y) * (x + 2 * y + 3 * z)

-- The equivalent form as a product of terms
def K_equiv (x y z : ℤ) : ℤ :=
  (y + z - 2 * x) * (z + x - 2 * y) * (x + y - 2 * z)

-- Proof statements for each set of numbers
theorem K_set_I : K 1 4 9 = K_equiv 1 4 9 := by
  sorry

theorem K_set_III : K 4 9 1 = K_equiv 4 9 1 := by
  sorry

theorem K_set_IV : K 1 8 11 = K_equiv 1 8 11 := by
  sorry

theorem K_set_V : K 5 8 (-2) = K_equiv 5 8 (-2) := by
  sorry

end K_set_I_K_set_III_K_set_IV_K_set_V_l117_117218


namespace dan_licks_l117_117406

/-- 
Given that Michael takes 63 licks, Sam takes 70 licks, David takes 70 licks, 
Lance takes 39 licks, and the average number of licks for all five people is 60, 
prove that Dan takes 58 licks to get to the center of a lollipop.
-/
theorem dan_licks (D : ℕ) 
  (M : ℕ := 63) 
  (S : ℕ := 70) 
  (Da : ℕ := 70) 
  (L : ℕ := 39)
  (avg : ℕ := 60) :
  ((M + S + Da + L + D) / 5 = avg) → D = 58 :=
by sorry

end dan_licks_l117_117406


namespace smallest_prime_with_digit_sum_23_l117_117733

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l117_117733


namespace inradius_one_third_height_l117_117179

-- The problem explicitly states this triangle's sides form an arithmetic progression.
-- We need to define conditions and then prove the question is equivalent to the answer given those conditions.
theorem inradius_one_third_height (a b c r h_b : ℝ) (h : a ≤ b ∧ b ≤ c) (h_arith : 2 * b = a + c) :
  r = h_b / 3 :=
sorry

end inradius_one_third_height_l117_117179


namespace geometric_to_arithmetic_common_ratio_greater_than_1_9_l117_117116

theorem geometric_to_arithmetic (q : ℝ) (h : q = (1 + Real.sqrt 5) / 2) :
  ∃ (a b c : ℝ), b - a = c - b ∧ a / b = b / c := 
sorry

theorem common_ratio_greater_than_1_9 (q : ℝ) (h_pos : q > 1.9 ∧ q < 2) :
  ∃ (n : ℕ), q^(n+1) - 2 * q^n + 1 = 0 :=
sorry

end geometric_to_arithmetic_common_ratio_greater_than_1_9_l117_117116


namespace value_of_a_sum_l117_117926

theorem value_of_a_sum (a_7 a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^7 = a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 128 := 
by
  sorry

end value_of_a_sum_l117_117926


namespace complete_sets_characterization_l117_117869

-- Definition of a complete set
def complete_set (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, (a + b ∈ A) → (a * b ∈ A)

-- Theorem stating that the complete sets of natural numbers are exactly
-- {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, ℕ.
theorem complete_sets_characterization :
  ∀ (A : Set ℕ), complete_set A ↔ (A = {1} ∨ A = {1, 2} ∨ A = {1, 2, 3} ∨ A = {1, 2, 3, 4} ∨ A = Set.univ) :=
sorry

end complete_sets_characterization_l117_117869


namespace solve_for_x_l117_117883

theorem solve_for_x (x : ℤ) (h_eq : (7 * x - 5) / (x - 2) = 2 / (x - 2)) (h_cond : x ≠ 2) : x = 1 := by
  sorry

end solve_for_x_l117_117883


namespace percentage_regular_cars_l117_117150

theorem percentage_regular_cars (total_cars : ℕ) (truck_percentage : ℚ) (convertibles : ℕ) 
  (h1 : total_cars = 125) (h2 : truck_percentage = 0.08) (h3 : convertibles = 35) : 
  (80 / 125 : ℚ) * 100 = 64 := 
by 
  sorry

end percentage_regular_cars_l117_117150


namespace unique_arrangements_of_MOON_l117_117998

open Nat

theorem unique_arrangements_of_MOON : 
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  factorial n / (factorial numO * factorial numM * factorial numN) = 12 :=
by
  let word := "MOON"
  let n := 4
  let numM := 1
  let numN := 1
  let numO := 2
  sorry

end unique_arrangements_of_MOON_l117_117998


namespace binom_7_4_l117_117640

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l117_117640


namespace stratified_sampling_young_employees_l117_117833

-- Given conditions
def total_young : Nat := 350
def total_middle_aged : Nat := 500
def total_elderly : Nat := 150
def total_employees : Nat := total_young + total_middle_aged + total_elderly
def representatives_to_select : Nat := 20
def sampling_ratio : Rat := representatives_to_select / (total_employees : Rat)

-- Proof goal
theorem stratified_sampling_young_employees :
  (total_young : Rat) * sampling_ratio = 7 := 
by
  sorry

end stratified_sampling_young_employees_l117_117833


namespace pollen_mass_in_scientific_notation_l117_117077

theorem pollen_mass_in_scientific_notation : 
  ∃ c n : ℝ, 0.0000037 = c * 10^n ∧ 1 ≤ c ∧ c < 10 ∧ c = 3.7 ∧ n = -6 :=
sorry

end pollen_mass_in_scientific_notation_l117_117077


namespace valid_number_of_apples_l117_117822

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l117_117822


namespace distance_traveled_on_fifth_day_equals_12_li_l117_117419

theorem distance_traveled_on_fifth_day_equals_12_li:
  ∀ {a_1 : ℝ},
    (a_1 * ((1 - (1 / 2) ^ 6) / (1 - 1 / 2)) = 378) →
    (a_1 * (1 / 2) ^ 4 = 12) :=
by
  intros a_1 h
  sorry

end distance_traveled_on_fifth_day_equals_12_li_l117_117419


namespace fraction_to_decimal_17_625_l117_117007

def fraction_to_decimal (num : ℕ) (den : ℕ) : ℚ := num / den

theorem fraction_to_decimal_17_625 : fraction_to_decimal 17 625 = 272 / 10000 := by
  sorry

end fraction_to_decimal_17_625_l117_117007


namespace notes_count_l117_117397

theorem notes_count (x : ℕ) (num_2_yuan num_5_yuan num_10_yuan total_notes total_amount : ℕ) 
    (h1 : total_amount = 160)
    (h2 : total_notes = 25)
    (h3 : num_5_yuan = x)
    (h4 : num_10_yuan = x)
    (h5 : num_2_yuan = total_notes - 2 * x)
    (h6 : 2 * num_2_yuan + 5 * num_5_yuan + 10 * num_10_yuan = total_amount) :
    num_5_yuan = 10 ∧ num_10_yuan = 10 ∧ num_2_yuan = 5 :=
by
  sorry

end notes_count_l117_117397


namespace weights_problem_l117_117135

theorem weights_problem (n : ℕ) (x : ℝ) (h_avg : ∀ (i : ℕ), i < n → ∃ (w : ℝ), w = x) 
  (h_heaviest : ∃ (w_max : ℝ), w_max = 5 * x) : n > 5 :=
by
  sorry

end weights_problem_l117_117135


namespace remainder_3005_98_l117_117123

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l117_117123


namespace five_pow_sum_of_squares_l117_117868

theorem five_pow_sum_of_squares (n : ℕ) : ∃ a b : ℕ, 5^n = a^2 + b^2 := 
sorry

end five_pow_sum_of_squares_l117_117868


namespace cost_of_three_pencils_and_two_pens_l117_117990

theorem cost_of_three_pencils_and_two_pens
  (p q : ℝ)
  (h₁ : 8 * p + 3 * q = 5.20)
  (h₂ : 2 * p + 5 * q = 4.40) :
  3 * p + 2 * q = 2.5881 :=
by
  sorry

end cost_of_three_pencils_and_two_pens_l117_117990


namespace complete_square_quadratic_t_l117_117147

theorem complete_square_quadratic_t : 
  ∀ x : ℝ, (16 * x^2 - 32 * x - 512 = 0) → (∃ q t : ℝ, (x + q)^2 = t ∧ t = 33) :=
by sorry

end complete_square_quadratic_t_l117_117147


namespace Xiaogang_shooting_probability_l117_117978

theorem Xiaogang_shooting_probability (total_shots : ℕ) (shots_made : ℕ) (h_total : total_shots = 50) (h_made : shots_made = 38) :
  (shots_made : ℝ) / total_shots = 0.76 :=
by
  sorry

end Xiaogang_shooting_probability_l117_117978


namespace similar_rect_tiling_l117_117540

-- Define the dimensions of rectangles A and B
variables {a1 a2 b1 b2 : ℝ}

-- Define the tiling condition
def similar_tiled (a1 a2 b1 b2 : ℝ) : Prop := 
  -- A placeholder for the actual definition of similar tiling
  sorry

-- The main theorem to prove
theorem similar_rect_tiling (h : similar_tiled a1 a2 b1 b2) : similar_tiled b1 b2 a1 a2 :=
sorry

end similar_rect_tiling_l117_117540


namespace election_winning_candidate_votes_l117_117759

theorem election_winning_candidate_votes (V : ℕ) 
  (h1 : V = (4 / 7) * V + 2000 + 4000) : 
  (4 / 7) * V = 8000 :=
by
  sorry

end election_winning_candidate_votes_l117_117759


namespace range_of_m_l117_117126

noncomputable def inequality_solutions (x m : ℝ) := |x + 2| - |x + 3| > m

theorem range_of_m (m : ℝ) : (∃ x : ℝ, inequality_solutions x m) → m < 1 :=
by
  sorry

end range_of_m_l117_117126


namespace math_problem_l117_117716

theorem math_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + x + y = 83) (h4 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 := by 
  sorry

end math_problem_l117_117716


namespace median_of_first_ten_positive_integers_l117_117105

def first_ten_positive_integers := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem median_of_first_ten_positive_integers : 
  ∃ median : ℝ, median = 5.5 := by
  sorry

end median_of_first_ten_positive_integers_l117_117105


namespace subway_ways_l117_117621

theorem subway_ways (total_ways : ℕ) (bus_ways : ℕ) (h1 : total_ways = 7) (h2 : bus_ways = 4) :
  total_ways - bus_ways = 3 :=
by
  sorry

end subway_ways_l117_117621


namespace original_paint_intensity_l117_117534

theorem original_paint_intensity 
  (P : ℝ)
  (H1 : 0 ≤ P ∧ P ≤ 100)
  (H2 : ∀ (unit : ℝ), unit = 100)
  (H3 : ∀ (replaced_fraction : ℝ), replaced_fraction = 1.5)
  (H4 : ∀ (new_intensity : ℝ), new_intensity = 30)
  (H5 : ∀ (solution_intensity : ℝ), solution_intensity = 0.25) :
  P = 15 := 
by
  sorry

end original_paint_intensity_l117_117534


namespace ff1_is_1_l117_117714

noncomputable def f (x : ℝ) := Real.log x - 2 * x + 3

theorem ff1_is_1 : f (f 1) = 1 := by
  sorry

end ff1_is_1_l117_117714


namespace number_of_monkeys_l117_117945

theorem number_of_monkeys (N : ℕ)
  (h1 : N * 1 * 8 = 8)
  (h2 : 3 * 1 * 8 = 3 * 8) :
  N = 8 :=
sorry

end number_of_monkeys_l117_117945


namespace handshakes_at_event_l117_117562

theorem handshakes_at_event 
  (num_couples : ℕ) 
  (num_people : ℕ) 
  (num_handshakes_men : ℕ) 
  (num_handshakes_men_women : ℕ) 
  (total_handshakes : ℕ) 
  (cond1 : num_couples = 15) 
  (cond2 : num_people = 2 * num_couples) 
  (cond3 : num_handshakes_men = (num_couples * (num_couples - 1)) / 2) 
  (cond4 : num_handshakes_men_women = num_couples * (num_couples - 1)) 
  (cond5 : total_handshakes = num_handshakes_men + num_handshakes_men_women) : 
  total_handshakes = 315 := 
by sorry

end handshakes_at_event_l117_117562


namespace isosceles_triangle_area_l117_117723

theorem isosceles_triangle_area (s b : ℝ) (h₁ : s + b = 20) (h₂ : b^2 + 10^2 = s^2) : 
  1/2 * 2 * b * 10 = 75 :=
by sorry

end isosceles_triangle_area_l117_117723


namespace normal_vector_to_line_l117_117212

theorem normal_vector_to_line : 
  ∀ (x y : ℝ), x - 3 * y + 6 = 0 → (1, -3) = (1, -3) :=
by
  intros x y h_line
  sorry

end normal_vector_to_line_l117_117212


namespace sin_13pi_over_6_equals_half_l117_117922

noncomputable def sin_13pi_over_6 : ℝ := Real.sin (13 * Real.pi / 6)

theorem sin_13pi_over_6_equals_half : sin_13pi_over_6 = 1 / 2 := by
  sorry

end sin_13pi_over_6_equals_half_l117_117922


namespace solve_system_of_equations_solve_system_of_inequalities_l117_117207

-- Proof for the system of equations
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 32) 
  (h2 : 2 * x - y = 0) :
  x = 8 ∧ y = 16 :=
by
  sorry

-- Proof for the system of inequalities
theorem solve_system_of_inequalities (x : ℝ)
  (h3 : 3 * x - 1 < 5 - 2 * x)
  (h4 : 5 * x + 1 ≥ 2 * x + 3) :
  (2 / 3 : ℝ) ≤ x ∧ x < (6 / 5 : ℝ) :=
by
  sorry

end solve_system_of_equations_solve_system_of_inequalities_l117_117207


namespace regular_polygon_sides_l117_117837

theorem regular_polygon_sides (n : ℕ) (h : n > 0) (h_exterior_angle : 360 / n = 10) : n = 36 :=
by sorry

end regular_polygon_sides_l117_117837


namespace printers_finish_tasks_l117_117770

theorem printers_finish_tasks :
  ∀ (start_time_1 finish_half_time_1 start_time_2 : ℕ) (half_task_duration full_task_duration second_task_duration : ℕ),
    start_time_1 = 9 * 60 ∧
    finish_half_time_1 = 12 * 60 + 30 ∧
    half_task_duration = finish_half_time_1 - start_time_1 ∧
    full_task_duration = 2 * half_task_duration ∧
    start_time_2 = 13 * 60 ∧
    second_task_duration = 2 * 60 ∧
    start_time_1 + full_task_duration = 4 * 60 ∧
    start_time_2 + second_task_duration = 15 * 60 →
  max (start_time_1 + full_task_duration) (start_time_2 + second_task_duration) = 16 * 60 := 
by
  intros start_time_1 finish_half_time_1 start_time_2 half_task_duration full_task_duration second_task_duration
  intro h
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
  sorry

end printers_finish_tasks_l117_117770


namespace football_club_balance_l117_117256

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end football_club_balance_l117_117256


namespace stable_performance_l117_117205

theorem stable_performance 
  (X_A_mean : ℝ) (X_B_mean : ℝ) (S_A_var : ℝ) (S_B_var : ℝ)
  (h1 : X_A_mean = 82) (h2 : X_B_mean = 82)
  (h3 : S_A_var = 245) (h4 : S_B_var = 190) : S_B_var < S_A_var :=
by {
  sorry
}

end stable_performance_l117_117205


namespace find_b_of_parabola_axis_of_symmetry_l117_117214

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end find_b_of_parabola_axis_of_symmetry_l117_117214


namespace range_of_values_includes_one_integer_l117_117177

theorem range_of_values_includes_one_integer (x : ℝ) (h : -1 < 2 * x + 3 ∧ 2 * x + 3 < 1) :
  ∃! n : ℤ, -7 < (2 * x - 3) ∧ (2 * x - 3) < -5 ∧ n = -6 :=
sorry

end range_of_values_includes_one_integer_l117_117177


namespace find_alpha_beta_sum_l117_117336

theorem find_alpha_beta_sum
  (a : ℝ) (α β φ : ℝ)
  (h1 : 3 * Real.sin α + 4 * Real.cos α = a)
  (h2 : 3 * Real.sin β + 4 * Real.cos β = a)
  (h3 : α ≠ β)
  (h4 : 0 < α ∧ α < 2 * Real.pi)
  (h5 : 0 < β ∧ β < 2 * Real.pi)
  (hφ : φ = Real.arcsin (4/5)) :
  α + β = Real.pi - 2 * φ ∨ α + β = 3 * Real.pi - 2 * φ :=
by
  sorry

end find_alpha_beta_sum_l117_117336


namespace bus_A_speed_l117_117325

-- Define the conditions
variables (v_A v_B : ℝ)
axiom equation1 : v_A - v_B = 15
axiom equation2 : v_A + v_B = 75

-- The main theorem we want to prove
theorem bus_A_speed : v_A = 45 :=
by {
  sorry
}

end bus_A_speed_l117_117325


namespace determine_parabola_equation_l117_117933

-- Define the conditions
def focus_on_line (focus : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, focus = (k - 2, k / 2 - 1)

-- Define the result equations
def is_standard_equation (eq : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, eq x y → x^2 = 4 * y) ∨ (∀ x y : ℝ, eq x y → y^2 = -8 * x)

-- Define the theorem stating that given the condition,
-- the standard equation is one of the two forms
theorem determine_parabola_equation (focus : ℝ × ℝ) (H : focus_on_line focus) :
  ∃ eq : ℝ → ℝ → Prop, is_standard_equation eq :=
sorry

end determine_parabola_equation_l117_117933


namespace equation_of_parabola_passing_through_points_l117_117654

noncomputable def parabola (x : ℝ) (b c : ℝ) : ℝ :=
  x^2 + b * x + c

theorem equation_of_parabola_passing_through_points :
  ∃ (b c : ℝ), 
    (parabola 0 b c = 5) ∧ (parabola 3 b c = 2) ∧
    (∀ x, parabola x b c = x^2 - 4 * x + 5) := 
by
  sorry

end equation_of_parabola_passing_through_points_l117_117654


namespace fib_seventh_term_l117_117300

-- Defining the Fibonacci sequence
def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib n + fib (n + 1)

-- Proving the value of the 7th term given 
-- fib(5) = 5 and fib(6) = 8
theorem fib_seventh_term : fib 7 = 13 :=
by {
    -- Conditions have been used in the definition of Fibonacci sequence
    sorry
}

end fib_seventh_term_l117_117300


namespace find_divisor_l117_117191

theorem find_divisor (D Q R d : ℕ) (h1 : D = 159) (h2 : Q = 9) (h3 : R = 6) (h4 : D = d * Q + R) : d = 17 := by
  sorry

end find_divisor_l117_117191


namespace vector_sum_magnitude_l117_117346

variable (a b : EuclideanSpace ℝ (Fin 3)) -- assuming 3-dimensional Euclidean space for vectors

-- Define the conditions
def mag_a : ℝ := 5
def mag_b : ℝ := 6
def dot_prod_ab : ℝ := -6

-- Prove the required magnitude condition
theorem vector_sum_magnitude (ha : ‖a‖ = mag_a) (hb : ‖b‖ = mag_b) (hab : inner a b = dot_prod_ab) :
  ‖a + b‖ = 7 :=
by
  sorry

end vector_sum_magnitude_l117_117346


namespace ABC_three_digit_number_l117_117946

theorem ABC_three_digit_number : 
    ∃ (A B C : ℕ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
    3 * C % 10 = 8 ∧ 
    3 * B + 1 % 10 = 8 ∧ 
    3 * A + 2 = 8 ∧ 
    100 * A + 10 * B + C = 296 := 
by
  sorry

end ABC_three_digit_number_l117_117946


namespace range_of_a_l117_117879

noncomputable def f (a x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ∧ 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) ↔ a = -1 := 
by
  sorry

end range_of_a_l117_117879


namespace ratio_length_breadth_l117_117203

theorem ratio_length_breadth
  (b : ℝ) (A : ℝ) (h_b : b = 11) (h_A : A = 363) :
  (∃ l : ℝ, A = l * b ∧ l / b = 3) :=
by
  sorry

end ratio_length_breadth_l117_117203


namespace peanuts_in_box_l117_117563

theorem peanuts_in_box (original_peanuts added_peanuts total_peanuts : ℕ) (h1 : original_peanuts = 10) (h2 : added_peanuts = 8) (h3 : total_peanuts = original_peanuts + added_peanuts) : total_peanuts = 18 := 
by {
  sorry
}

end peanuts_in_box_l117_117563


namespace find_constant_c_l117_117352

theorem find_constant_c (c : ℝ) (h : (x + 7) ∣ (c*x^3 + 19*x^2 - 3*c*x + 35)) : c = 3 := by
  sorry

end find_constant_c_l117_117352


namespace probability_of_earning_1900_equals_6_over_125_l117_117728

-- Representation of a slot on the spinner.
inductive Slot
| Bankrupt 
| Dollar1000
| Dollar500
| Dollar4000
| Dollar400 
deriving DecidableEq

-- Condition: There are 5 slots and each has the same probability.
noncomputable def slots := [Slot.Bankrupt, Slot.Dollar1000, Slot.Dollar500, Slot.Dollar4000, Slot.Dollar400]

-- Probability of earning exactly $1900 in three spins.
def probability_of_1900 : ℚ :=
  let target_combination := [Slot.Dollar500, Slot.Dollar400, Slot.Dollar1000]
  let total_ways := 125
  let successful_ways := 6
  (successful_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_earning_1900_equals_6_over_125 :
  probability_of_1900 = 6 / 125 :=
sorry

end probability_of_earning_1900_equals_6_over_125_l117_117728


namespace table_chair_price_l117_117390

theorem table_chair_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : T = 84) : T + C = 96 :=
sorry

end table_chair_price_l117_117390


namespace arithmetic_sequence_geometric_condition_l117_117166

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℝ) (d : ℝ) (h_nonzero : d ≠ 0) 
  (h_a3 : a 3 = 7)
  (h_geo_seq : (a 2 - 1)^2 = (a 1 - 1) * (a 4 - 1)) : 
  a 10 = 21 :=
sorry

end arithmetic_sequence_geometric_condition_l117_117166


namespace determinant_example_l117_117357

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

theorem determinant_example : determinant_2x2 5 (-4) 2 3 = 23 := 
by 
  sorry

end determinant_example_l117_117357


namespace calculate_present_worth_l117_117014

variable (BG : ℝ) (r : ℝ) (t : ℝ)

theorem calculate_present_worth (hBG : BG = 24) (hr : r = 0.10) (ht : t = 2) : 
  ∃ PW : ℝ, PW = 120 := 
by
  sorry

end calculate_present_worth_l117_117014


namespace find_x_l117_117480

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) : 
  let l5 := log_base 5 x
  let l6 := log_base 6 x
  let l7 := log_base 7 x
  let surface_area := 2 * (l5 * l6 + l5 * l7 + l6 * l7)
  let volume := l5 * l6 * l7 
  (surface_area = 2 * volume) → x = 210 :=
by 
  sorry

end find_x_l117_117480


namespace total_toys_l117_117246

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end total_toys_l117_117246


namespace product_of_roots_quadratic_l117_117902

noncomputable def product_of_roots (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  x1 * x2

theorem product_of_roots_quadratic :
  (product_of_roots 1 3 (-5)) = -5 :=
by
  sorry

end product_of_roots_quadratic_l117_117902


namespace location_determined_l117_117649

def determine_location(p : String) : Prop :=
  p = "Longitude 118°E, Latitude 40°N"

axiom row_2_in_cinema : ¬determine_location "Row 2 in a cinema"
axiom daqiao_south_road_nanjing : ¬determine_location "Daqiao South Road in Nanjing"
axiom thirty_degrees_northeast : ¬determine_location "30° northeast"
axiom longitude_latitude : determine_location "Longitude 118°E, Latitude 40°N"

theorem location_determined : determine_location "Longitude 118°E, Latitude 40°N" :=
longitude_latitude

end location_determined_l117_117649


namespace production_difference_l117_117917

variables (p h : ℕ)

def first_day_production := p * h

def second_day_production := (p + 5) * (h - 3)

-- Given condition
axiom p_eq_3h : p = 3 * h

theorem production_difference : first_day_production p h - second_day_production p h = 4 * h + 15 :=
by
  sorry

end production_difference_l117_117917


namespace number_of_valid_arithmetic_sequences_l117_117956

theorem number_of_valid_arithmetic_sequences : 
  ∃ S : Finset (Finset ℕ), 
  S.card = 16 ∧ 
  ∀ s ∈ S, s.card = 3 ∧ 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ s = {a, b, c} ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (b - a = c - b) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) := 
sorry

end number_of_valid_arithmetic_sequences_l117_117956


namespace find_adult_ticket_cost_l117_117304

noncomputable def adult_ticket_cost (A : ℝ) : Prop :=
  let num_adults := 152
  let num_children := num_adults / 2
  let children_ticket_cost := 2.50
  let total_receipts := 1026
  total_receipts = num_adults * A + num_children * children_ticket_cost

theorem find_adult_ticket_cost : adult_ticket_cost 5.50 :=
by
  sorry

end find_adult_ticket_cost_l117_117304


namespace supplement_of_complement_of_30_degrees_l117_117091

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α
def α : ℝ := 30

theorem supplement_of_complement_of_30_degrees : supplement (complement α) = 120 := 
by
  sorry

end supplement_of_complement_of_30_degrees_l117_117091


namespace largest_C_inequality_l117_117492

theorem largest_C_inequality :
  ∃ C : ℝ, C = Real.sqrt (8 / 3) ∧ ∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z) :=
by
  sorry

end largest_C_inequality_l117_117492


namespace wilson_sledding_l117_117938

variable (T : ℕ)

theorem wilson_sledding :
  (4 * T) + 6 = 14 → T = 2 :=
by
  intros h
  sorry

end wilson_sledding_l117_117938


namespace inequality_holds_for_all_m_l117_117932

theorem inequality_holds_for_all_m (m : ℝ) (h1 : ∀ (x : ℝ), x^2 - 8 * x + 20 > 0)
  (h2 : m < -1/2) : ∀ (x : ℝ), (x ^ 2 - 8 * x + 20) / (m * x ^ 2 + 2 * (m + 1) * x + 9 * m + 4) < 0 :=
by
  sorry

end inequality_holds_for_all_m_l117_117932


namespace perpendicular_line_eq_l117_117228

theorem perpendicular_line_eq (x y : ℝ) :
  (∃ (p : ℝ × ℝ), p = (-2, 3) ∧ 
    ∀ y₀ x₀, 3 * x - y = 6 ∧ y₀ = 3 ∧ x₀ = -2 → y = -1 / 3 * x + 7 / 3) :=
sorry

end perpendicular_line_eq_l117_117228


namespace pythagorean_diagonal_l117_117800

variable (m : ℕ) (h_m : m ≥ 3)

theorem pythagorean_diagonal (h : (2 * m)^2 + a^2 = (a + 2)^2) :
  (a + 2) = m^2 + 1 :=
by
  sorry

end pythagorean_diagonal_l117_117800


namespace arithmetic_sequence_sum_l117_117157

theorem arithmetic_sequence_sum :
  (∀ (a : ℕ → ℤ),  a 1 + a 2 = 4 ∧ a 3 + a 4 = 6 → a 8 + a 9 = 10) :=
sorry

end arithmetic_sequence_sum_l117_117157


namespace math_problem_l117_117580

theorem math_problem (x y : ℤ) (a b : ℤ) (h1 : x - 5 = 7 * a) (h2 : y + 7 = 7 * b) (h3 : (x ^ 2 + y ^ 3) % 11 = 0) : 
  ((y - x) / 13) = 13 :=
sorry

end math_problem_l117_117580


namespace function_decreasing_odd_function_m_zero_l117_117511

-- First part: Prove that the function is decreasing
theorem function_decreasing (m : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
    let f := fun x => -2 * x + m
    f x1 > f x2 :=
by
    sorry

-- Second part: Find the value of m when the function is odd
theorem odd_function_m_zero (m : ℝ) :
    (∀ x : ℝ, let f := fun x => -2 * x + m
              f (-x) = -f x) → m = 0 :=
by
    sorry

end function_decreasing_odd_function_m_zero_l117_117511


namespace reduced_price_per_kg_l117_117196

/-- Given that:
1. There is a reduction of 25% in the price of oil.
2. The housewife can buy 5 kgs more for Rs. 700 after the reduction.

Prove that the reduced price per kg of oil is Rs. 35. -/
theorem reduced_price_per_kg (P : ℝ) (R : ℝ) (X : ℝ)
  (h1 : R = 0.75 * P)
  (h2 : 700 = X * P)
  (h3 : 700 = (X + 5) * R)
  : R = 35 := 
sorry

end reduced_price_per_kg_l117_117196


namespace oil_bill_increase_l117_117395

theorem oil_bill_increase :
  ∀ (F x : ℝ), 
    (F / 120 = 5 / 4) → 
    ((F + x) / 120 = 3 / 2) → 
    x = 30 :=
by
  intros F x h1 h2
  -- proof
  sorry

end oil_bill_increase_l117_117395


namespace basic_astrophysics_degrees_l117_117517

-- Define the percentages for various sectors
def microphotonics := 14
def home_electronics := 24
def food_additives := 15
def genetically_modified_microorganisms := 19
def industrial_lubricants := 8

-- The sum of the given percentages
def total_other_percentages := 
    microphotonics + home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants

-- The remaining percentage for basic astrophysics
def basic_astrophysics_percentage := 100 - total_other_percentages

-- Number of degrees in a full circle
def full_circle_degrees := 360

-- Calculate the degrees representing basic astrophysics
def degrees_for_basic_astrophysics := (basic_astrophysics_percentage * full_circle_degrees) / 100

-- Theorem statement
theorem basic_astrophysics_degrees : degrees_for_basic_astrophysics = 72 := 
by
  sorry

end basic_astrophysics_degrees_l117_117517


namespace number_of_color_copies_l117_117741

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end number_of_color_copies_l117_117741


namespace jane_weekly_pages_l117_117426

-- Define the daily reading amounts
def monday_wednesday_morning_pages : ℕ := 5
def monday_wednesday_evening_pages : ℕ := 10
def tuesday_thursday_morning_pages : ℕ := 7
def tuesday_thursday_evening_pages : ℕ := 8
def friday_morning_pages : ℕ := 10
def friday_evening_pages : ℕ := 15
def weekend_morning_pages : ℕ := 12
def weekend_evening_pages : ℕ := 20

-- Define the number of days
def monday_wednesday_days : ℕ := 2
def tuesday_thursday_days : ℕ := 2
def friday_days : ℕ := 1
def weekend_days : ℕ := 2

-- Function to calculate weekly pages
def weekly_pages :=
  (monday_wednesday_days * (monday_wednesday_morning_pages + monday_wednesday_evening_pages)) +
  (tuesday_thursday_days * (tuesday_thursday_morning_pages + tuesday_thursday_evening_pages)) +
  (friday_days * (friday_morning_pages + friday_evening_pages)) +
  (weekend_days * (weekend_morning_pages + weekend_evening_pages))

-- Proof statement
theorem jane_weekly_pages : weekly_pages = 149 := by
  unfold weekly_pages
  norm_num
  sorry

end jane_weekly_pages_l117_117426


namespace pop_spending_original_l117_117307

-- Given conditions
def total_spent := 150
def crackle_spending (P : ℝ) := 3 * P
def snap_spending (P : ℝ) := 2 * crackle_spending P

-- Main statement to prove
theorem pop_spending_original : ∃ P : ℝ, snap_spending P + crackle_spending P + P = total_spent ∧ P = 15 :=
by
  sorry

end pop_spending_original_l117_117307


namespace factor_polynomial_l117_117190

theorem factor_polynomial (x y z : ℂ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) := by
  sorry

end factor_polynomial_l117_117190


namespace two_point_questions_count_l117_117814

theorem two_point_questions_count (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
sorry

end two_point_questions_count_l117_117814


namespace lucy_total_journey_l117_117886

-- Define the length of Lucy's journey
def lucy_journey (x : ℝ) : Prop :=
  (1 / 4) * x + 25 + (1 / 6) * x = x

-- State the theorem
theorem lucy_total_journey : ∃ x : ℝ, lucy_journey x ∧ x = 300 / 7 := by
  sorry

end lucy_total_journey_l117_117886


namespace fibonacci_series_sum_l117_117429

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Theorem to prove that the infinite series sum is 2
theorem fibonacci_series_sum : (∑' n : ℕ, (fib n : ℝ) / (2 ^ n : ℝ)) = 2 :=
sorry

end fibonacci_series_sum_l117_117429


namespace likelihood_of_white_crows_at_birch_unchanged_l117_117312

theorem likelihood_of_white_crows_at_birch_unchanged 
  (a b c d : ℕ) 
  (h1 : a + b = 50) 
  (h2 : c + d = 50) 
  (h3 : b ≥ a) 
  (h4 : d ≥ c - 1) : 
  (bd + ac + a + b : ℝ) / 2550 > (bc + ad : ℝ) / 2550 := by 
  sorry

end likelihood_of_white_crows_at_birch_unchanged_l117_117312


namespace probability_heads_l117_117824

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l117_117824


namespace mn_value_l117_117344

theorem mn_value (m n : ℤ) (h1 : 2 * m = 6) (h2 : m - n = 2) : m * n = 3 := by
  sorry

end mn_value_l117_117344


namespace cost_of_1500_pencils_l117_117899

theorem cost_of_1500_pencils (cost_per_box : ℕ) (pencils_per_box : ℕ) (num_pencils : ℕ) :
  cost_per_box = 30 → pencils_per_box = 100 → num_pencils = 1500 → 
  (num_pencils * (cost_per_box / pencils_per_box) = 450) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end cost_of_1500_pencils_l117_117899


namespace parallel_vectors_implies_scalar_l117_117174

-- Defining the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Stating the condition and required proof
theorem parallel_vectors_implies_scalar (m : ℝ) (h : (vector_a.snd / vector_a.fst) = (vector_b m).snd / (vector_b m).fst) : m = -4 :=
by sorry

end parallel_vectors_implies_scalar_l117_117174


namespace quadratic_with_roots_1_and_2_l117_117503

theorem quadratic_with_roots_1_and_2 : ∃ (a b c : ℝ), (a = 1 ∧ b = 2) ∧ (∀ x : ℝ, x ≠ 1 → x ≠ 2 → a * x^2 + b * x + c = 0) ∧ (a * x^2 + b * x + c = x^2 - 3 * x + 2) :=
by
  sorry

end quadratic_with_roots_1_and_2_l117_117503


namespace geom_sequence_a1_l117_117108

noncomputable def a_n (a1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n-1)

theorem geom_sequence_a1 {a1 q : ℝ} 
  (h1 : 0 < q)
  (h2 : a_n a1 q 4 * a_n a1 q 8 = 2 * (a_n a1 q 5)^2)
  (h3 : a_n a1 q 2 = 1) :
  a1 = (Real.sqrt 2) / 2 :=
sorry

end geom_sequence_a1_l117_117108


namespace reversible_triangle_inequality_l117_117999

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def reversible_triangle (a b c : ℝ) : Prop :=
  (is_triangle a b c) ∧ 
  (is_triangle (1 / a) (1 / b) (1 / c)) ∧
  (a ≤ b) ∧ (b ≤ c)

theorem reversible_triangle_inequality {a b c : ℝ} (h : reversible_triangle a b c) :
  a > (3 - Real.sqrt 5) / 2 * c :=
sorry

end reversible_triangle_inequality_l117_117999


namespace Julia_played_with_11_kids_on_Monday_l117_117635

theorem Julia_played_with_11_kids_on_Monday
  (kids_on_Tuesday : ℕ)
  (kids_on_Monday : ℕ) 
  (h1 : kids_on_Tuesday = 12)
  (h2 : kids_on_Tuesday = kids_on_Monday + 1) : 
  kids_on_Monday = 11 := 
by
  sorry

end Julia_played_with_11_kids_on_Monday_l117_117635


namespace meryll_remaining_questions_l117_117685

variables (total_mc total_ps total_tf : ℕ)
variables (frac_mc frac_ps frac_tf : ℚ)

-- Conditions as Lean definitions:
def written_mc (total_mc : ℕ) (frac_mc : ℚ) := (frac_mc * total_mc).floor
def written_ps (total_ps : ℕ) (frac_ps : ℚ) := (frac_ps * total_ps).floor
def written_tf (total_tf : ℕ) (frac_tf : ℚ) := (frac_tf * total_tf).floor

def remaining_mc (total_mc : ℕ) (frac_mc : ℚ) := total_mc - written_mc total_mc frac_mc
def remaining_ps (total_ps : ℕ) (frac_ps : ℚ) := total_ps - written_ps total_ps frac_ps
def remaining_tf (total_tf : ℕ) (frac_tf : ℚ) := total_tf - written_tf total_tf frac_tf

def total_remaining (total_mc total_ps total_tf : ℕ) (frac_mc frac_ps frac_tf : ℚ) :=
  remaining_mc total_mc frac_mc + remaining_ps total_ps frac_ps + remaining_tf total_tf frac_tf

-- The statement to prove:
theorem meryll_remaining_questions :
  total_remaining 50 30 40 (5/8) (7/12) (2/5) = 56 :=
by
  sorry

end meryll_remaining_questions_l117_117685


namespace count_integer_length_chords_l117_117074

/-- Point P is 9 units from the center of a circle with radius 15. -/
def point_distance_from_center : ℝ := 9

def circle_radius : ℝ := 15

/-- Correct answer to the number of different chords that contain P and have integer lengths. -/
def correct_answer : ℕ := 7

/-- Proving the number of chords containing P with integer lengths given the conditions. -/
theorem count_integer_length_chords : 
  ∀ (r_P : ℝ) (r_circle : ℝ), r_P = point_distance_from_center → r_circle = circle_radius → 
  (∃ n : ℕ, n = correct_answer) :=
by 
  intros r_P r_circle h1 h2
  use 7 
  sorry

end count_integer_length_chords_l117_117074


namespace height_inequality_triangle_l117_117631

theorem height_inequality_triangle (a b c h_a h_b h_c Δ : ℝ) (n : ℝ) 
  (ha : h_a = 2 * Δ / a)
  (hb : h_b = 2 * Δ / b)
  (hc : h_c = 2 * Δ / c)
  (n_pos : n > 0) :
  (a * h_b)^n + (b * h_c)^n + (c * h_a)^n ≥ 3 * 2^n * Δ^n := 
sorry

end height_inequality_triangle_l117_117631


namespace part_a_part_b_l117_117599
open Set

def fantastic (n : ℕ) : Prop :=
  ∃ a b : ℚ, a > 0 ∧ b > 0 ∧ n = a + 1 / a + b + 1 / b

theorem part_a : ∃ᶠ p in at_top, Prime p ∧ ∀ k, ¬ fantastic (k * p) := 
  sorry

theorem part_b : ∃ᶠ p in at_top, Prime p ∧ ∃ k, fantastic (k * p) :=
  sorry

end part_a_part_b_l117_117599


namespace factor_expression_l117_117359

theorem factor_expression (x : ℝ) :
  4 * x * (x - 5) + 7 * (x - 5) + 12 * (x - 5) = (4 * x + 19) * (x - 5) :=
by
  sorry

end factor_expression_l117_117359


namespace binom_18_6_l117_117152

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l117_117152


namespace common_area_of_rectangle_and_circle_eqn_l117_117122

theorem common_area_of_rectangle_and_circle_eqn :
  let rect_length := 8
  let rect_width := 4
  let circle_radius := 3
  let common_area := (3^2 * 2 * Real.pi / 4) - 2 * Real.sqrt 5  
  common_area = (9 * Real.pi / 2) - 2 * Real.sqrt 5 := 
sorry

end common_area_of_rectangle_and_circle_eqn_l117_117122


namespace mul_add_distrib_l117_117035

theorem mul_add_distrib :
  15 * 36 + 15 * 24 = 900 := by
  sorry

end mul_add_distrib_l117_117035


namespace value_of_x3_plus_inv_x3_l117_117153

theorem value_of_x3_plus_inv_x3 (x : ℝ) (h : 728 = x^6 + 1 / x^6) : 
  x^3 + 1 / x^3 = Real.sqrt 730 :=
sorry

end value_of_x3_plus_inv_x3_l117_117153


namespace sally_oscillation_distance_l117_117029

noncomputable def C : ℝ := 5 / 4
noncomputable def D : ℝ := 11 / 4

theorem sally_oscillation_distance :
  abs (C - D) = 3 / 2 :=
by
  sorry

end sally_oscillation_distance_l117_117029


namespace at_most_one_zero_l117_117845

-- Definition of the polynomial f(x)
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^4 - 1994 * x^3 + (1993 + n) * x^2 - 11 * x + n

-- The target theorem statement
theorem at_most_one_zero (n : ℤ) : ∃! x : ℝ, f n x = 0 :=
by
  sorry

end at_most_one_zero_l117_117845


namespace ratio_wy_l117_117180

-- Define the variables and conditions
variables (w x y z : ℚ)
def ratio_wx := w / x = 5 / 4
def ratio_yz := y / z = 7 / 5
def ratio_zx := z / x = 1 / 8

-- Statement to prove
theorem ratio_wy (hwx : ratio_wx w x) (hyz : ratio_yz y z) (hzx : ratio_zx z x) : w / y = 25 / 7 :=
by
  sorry  -- Proof not needed

end ratio_wy_l117_117180


namespace shorter_side_of_rectangular_room_l117_117458

theorem shorter_side_of_rectangular_room 
  (a b : ℕ) 
  (h1 : 2 * a + 2 * b = 52) 
  (h2 : a * b = 168) : 
  min a b = 12 := 
  sorry

end shorter_side_of_rectangular_room_l117_117458


namespace c_investment_l117_117984

theorem c_investment 
  (A_investment B_investment : ℝ)
  (C_share total_profit : ℝ)
  (hA : A_investment = 8000)
  (hB : B_investment = 4000)
  (hC_share : C_share = 36000)
  (h_profit : total_profit = 252000) :
  ∃ (x : ℝ), (x / 4000) / (2 + 1 + x / 4000) = (36000 / 252000) ∧ x = 2000 :=
by
  sorry

end c_investment_l117_117984


namespace ellipse_a_value_l117_117195

theorem ellipse_a_value
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1)
  (e : ℝ)
  (h3 : e = 2 / 3)
  : a = 3 :=
by
  sorry

end ellipse_a_value_l117_117195


namespace curve_is_segment_l117_117463

noncomputable def parametric_curve := {t : ℝ // 0 ≤ t ∧ t ≤ 5}

def x (t : parametric_curve) : ℝ := 3 * t.val ^ 2 + 2
def y (t : parametric_curve) : ℝ := t.val ^ 2 - 1

def line_equation (x y : ℝ) := x - 3 * y - 5 = 0

theorem curve_is_segment :
  ∀ (t : parametric_curve), line_equation (x t) (y t) ∧ 
  2 ≤ x t ∧ x t ≤ 77 :=
by
  sorry

end curve_is_segment_l117_117463


namespace garden_dimensions_l117_117894

theorem garden_dimensions (l w : ℕ) (h1 : 2 * l + 2 * w = 60) (h2 : l * w = 221) : 
    (l = 17 ∧ w = 13) ∨ (l = 13 ∧ w = 17) :=
sorry

end garden_dimensions_l117_117894


namespace part1_part2_l117_117129

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x + a - 1) + abs (x - 2 * a)

-- Part (1) of the proof problem
theorem part1 (a : ℝ) : f 1 a < 3 → - (2 : ℝ)/3 < a ∧ a < 4 / 3 := sorry

-- Part (2) of the proof problem
theorem part2 (a x : ℝ) : a ≥ 1 → f x a ≥ 2 := sorry

end part1_part2_l117_117129


namespace cos_2x_quadratic_l117_117160

theorem cos_2x_quadratic (x : ℝ) (a b c : ℝ)
  (h : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0)
  (h_a : a = 4) (h_b : b = 2) (h_c : c = -1) :
  4 * (Real.cos (2 * x)) ^ 2 + 2 * Real.cos (2 * x) - 1 = 0 := sorry

end cos_2x_quadratic_l117_117160


namespace area_square_EFGH_equiv_144_l117_117571

theorem area_square_EFGH_equiv_144 (a b : ℝ) (h : a = 6) (hb : b = 6)
  (side_length_EFGH : ℝ) (hs : side_length_EFGH = a + 3 + 3) : side_length_EFGH ^ 2 = 144 :=
by
  -- Given conditions
  sorry

end area_square_EFGH_equiv_144_l117_117571


namespace probability_rachel_robert_in_picture_l117_117366

theorem probability_rachel_robert_in_picture :
  let lap_rachel := 120 -- Rachel's lap time in seconds
  let lap_robert := 100 -- Robert's lap time in seconds
  let duration := 900 -- 15 minutes in seconds
  let picture_duration := 60 -- Picture duration in seconds
  let one_third_rachel := lap_rachel / 3 -- One third of Rachel's lap time
  let one_third_robert := lap_robert / 3 -- One third of Robert's lap time
  let rachel_in_window_start := 20 -- Rachel in the window from 20 to 100s
  let rachel_in_window_end := 100
  let robert_in_window_start := 0 -- Robert in the window from 0 to 66.66s
  let robert_in_window_end := 66.66
  let overlap_start := max rachel_in_window_start robert_in_window_start -- The start of overlap
  let overlap_end := min rachel_in_window_end robert_in_window_end -- The end of overlap
  let overlap_duration := overlap_end - overlap_start -- Duration of the overlap
  let probability := overlap_duration / picture_duration -- Probability of both in the picture
  probability = 46.66 / 60 := sorry

end probability_rachel_robert_in_picture_l117_117366


namespace solve_functional_equation_l117_117669

theorem solve_functional_equation
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ d c : ℝ, (∀ x, f x = d * x^2 + c) ∧ (∀ x, g x = d * x^2 + c) :=
sorry

end solve_functional_equation_l117_117669


namespace angle_B_lt_pi_div_two_l117_117176

theorem angle_B_lt_pi_div_two 
  (a b c : ℝ) (B : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : B = π / 2 - B)
  (h5 : 2 / b = 1 / a + 1 / c)
  : B < π / 2 := sorry

end angle_B_lt_pi_div_two_l117_117176


namespace polynomial_binomial_square_l117_117204

theorem polynomial_binomial_square (b : ℝ) : 
  (∃ c : ℝ, (3*X + c)^2 = 9*X^2 - 24*X + b) → b = 16 :=
by
  sorry

end polynomial_binomial_square_l117_117204


namespace zombie_count_today_l117_117467

theorem zombie_count_today (Z : ℕ) (h : Z < 50) : 16 * Z = 48 :=
by
  -- Assume Z, h conditions from a)
  -- Proof will go here, for now replaced with sorry
  sorry

end zombie_count_today_l117_117467


namespace add_to_fraction_eq_l117_117456

theorem add_to_fraction_eq (n : ℕ) : (4 + n) / (7 + n) = 6 / 7 → n = 14 :=
by sorry

end add_to_fraction_eq_l117_117456


namespace store_owner_marked_price_l117_117514

theorem store_owner_marked_price (L M : ℝ) (h1 : M = (56 / 45) * L) : M / L = 124.44 / 100 :=
by
  sorry

end store_owner_marked_price_l117_117514


namespace find_int_k_l117_117182

theorem find_int_k (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^3) :
  K = 11 :=
by
  sorry

end find_int_k_l117_117182


namespace earliest_time_meet_l117_117031

open Nat

def lap_time_anna := 5
def lap_time_bob := 8
def lap_time_carol := 10

def lcm_lap_times : ℕ :=
  Nat.lcm lap_time_anna (Nat.lcm lap_time_bob lap_time_carol)

theorem earliest_time_meet : lcm_lap_times = 40 := by
  sorry

end earliest_time_meet_l117_117031


namespace average_of_other_half_l117_117497

theorem average_of_other_half (avg : ℝ) (sum_half : ℝ) (n : ℕ) (n_half : ℕ)
    (h_avg : avg = 43.1)
    (h_sum_half : sum_half = 158.4)
    (h_n : n = 8)
    (h_n_half : n_half = n / 2) :
    ((n * avg - sum_half) / n_half) = 46.6 :=
by
  -- The proof steps would be given here. We're omitting them as the prompt instructs.
  sorry

end average_of_other_half_l117_117497


namespace gcd_lcm_product_135_l117_117526

theorem gcd_lcm_product_135 (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.gcd a b * Nat.lcm a b = 135 :=
by
  sorry

end gcd_lcm_product_135_l117_117526


namespace molecular_weight_of_aluminum_part_in_Al2_CO3_3_l117_117250

def total_molecular_weight_Al2_CO3_3 : ℝ := 234
def atomic_weight_Al : ℝ := 26.98
def num_atoms_Al_in_Al2_CO3_3 : ℕ := 2

theorem molecular_weight_of_aluminum_part_in_Al2_CO3_3 :
  num_atoms_Al_in_Al2_CO3_3 * atomic_weight_Al = 53.96 :=
by
  sorry

end molecular_weight_of_aluminum_part_in_Al2_CO3_3_l117_117250


namespace prime_factorization_of_expression_l117_117909

theorem prime_factorization_of_expression :
  2 * 3 * 5 * 7 - 1 = 11 * 19 :=
sorry

end prime_factorization_of_expression_l117_117909


namespace students_taking_neither_580_l117_117993

noncomputable def numberOfStudentsTakingNeither (total students_m students_a students_d students_ma students_md students_ad students_mad : ℕ) : ℕ :=
  let total_taking_at_least_one := (students_m + students_a + students_d) 
                                - (students_ma + students_md + students_ad) 
                                + students_mad
  total - total_taking_at_least_one

theorem students_taking_neither_580 :
  let total := 800
  let students_m := 140
  let students_a := 90
  let students_d := 75
  let students_ma := 50
  let students_md := 30
  let students_ad := 25
  let students_mad := 20
  numberOfStudentsTakingNeither total students_m students_a students_d students_ma students_md students_ad students_mad = 580 :=
by
  sorry

end students_taking_neither_580_l117_117993


namespace product_decrease_l117_117856

variable (a b : ℤ)

theorem product_decrease : (a - 3) * (b + 3) - a * b = 900 → a - b = 303 → a * b - (a + 3) * (b - 3) = 918 :=
by
    intros h1 h2
    sorry

end product_decrease_l117_117856


namespace absolute_value_equation_solution_l117_117566

theorem absolute_value_equation_solution (a b c : ℝ) :
  (∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|) ↔
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨ 
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨ 
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by
  sorry

end absolute_value_equation_solution_l117_117566


namespace sonnets_not_read_l117_117092

-- Define the conditions in the original problem
def sonnet_lines := 14
def unheard_lines := 70

-- Define a statement that needs to be proven
-- Prove that the number of sonnets not read is 5
theorem sonnets_not_read : unheard_lines / sonnet_lines = 5 := by
  sorry

end sonnets_not_read_l117_117092


namespace B_work_rate_l117_117473

theorem B_work_rate (A B C : ℕ) (combined_work_rate_A_B_C : ℕ)
  (A_work_days B_work_days C_work_days : ℕ)
  (combined_abc : combined_work_rate_A_B_C = 4)
  (a_work_rate : A_work_days = 6)
  (c_work_rate : C_work_days = 36) :
  B = 18 :=
by
  sorry

end B_work_rate_l117_117473


namespace not_possible_to_obtain_target_triple_l117_117267

def is_target_triple_achievable (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) = (0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y) →
    (b1^2 + b2^2 + b3^2 = 169 → False)

theorem not_possible_to_obtain_target_triple :
  ¬ is_target_triple_achievable 3 4 12 2 8 10 :=
by sorry

end not_possible_to_obtain_target_triple_l117_117267


namespace spending_on_games_l117_117691

-- Definitions converted from conditions
def totalAllowance := 48
def fractionClothes := 1 / 4
def fractionBooks := 1 / 3
def fractionSnacks := 1 / 6
def spentClothes := fractionClothes * totalAllowance
def spentBooks := fractionBooks * totalAllowance
def spentSnacks := fractionSnacks * totalAllowance
def spentGames := totalAllowance - (spentClothes + spentBooks + spentSnacks)

-- The theorem that needs to be proven
theorem spending_on_games : spentGames = 12 :=
by sorry

end spending_on_games_l117_117691


namespace percent_larger_semicircles_l117_117065

theorem percent_larger_semicircles (r1 r2 : ℝ) (d1 d2 : ℝ)
  (hr1 : r1 = d1 / 2) (hr2 : r2 = d2 / 2)
  (hd1 : d1 = 12) (hd2 : d2 = 8) : 
  (2 * (1/2) * Real.pi * r1^2) = (9/4 * (2 * (1/2) * Real.pi * r2^2)) :=
by
  sorry

end percent_larger_semicircles_l117_117065


namespace mean_of_roots_l117_117197

theorem mean_of_roots
  (a b c d k : ℤ)
  (p : ℤ → ℤ)
  (h_poly : ∀ x, p x = (x - a) * (x - b) * (x - c) * (x - d))
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : p k = 4) :
  k = (a + b + c + d) / 4 :=
by
  -- proof goes here
  sorry

end mean_of_roots_l117_117197


namespace basketball_team_initial_players_l117_117804

theorem basketball_team_initial_players
  (n : ℕ)
  (h_average_initial : Real := 190)
  (height_nikolai : Real := 197)
  (height_peter : Real := 181)
  (h_average_new : Real := 188)
  (total_height_initial : Real := h_average_initial * n)
  (total_height_new : Real := total_height_initial - (height_nikolai - height_peter))
  (avg_height_new_calculated : Real := total_height_new / n) :
  n = 8 :=
by
  sorry

end basketball_team_initial_players_l117_117804


namespace selling_price_is_correct_l117_117748

noncomputable def cost_price : ℝ := 192
def profit_percentage : ℝ := 0.25
def profit (cp : ℝ) (pp : ℝ) : ℝ := pp * cp
def selling_price (cp : ℝ) (pft : ℝ) : ℝ := cp + pft

theorem selling_price_is_correct : selling_price cost_price (profit cost_price profit_percentage) = 240 :=
sorry

end selling_price_is_correct_l117_117748


namespace anand_income_l117_117260

theorem anand_income (x y : ℕ)
  (income_A : ℕ := 5 * x)
  (income_B : ℕ := 4 * x)
  (expenditure_A : ℕ := 3 * y)
  (expenditure_B : ℕ := 2 * y)
  (savings_A : ℕ := 800)
  (savings_B : ℕ := 800)
  (hA : income_A - expenditure_A = savings_A)
  (hB : income_B - expenditure_B = savings_B) :
  income_A = 2000 := by
  sorry

end anand_income_l117_117260


namespace ellipse_standard_and_trajectory_l117_117154

theorem ellipse_standard_and_trajectory :
  ∀ a b x y : ℝ, 
  a > b ∧ 0 < b ∧ 
  (b^2 = a^2 - 1) ∧ 
  (9/4 + 6/(8) = 1) →
  (∃ x y : ℝ, (x / 2)^2 / 9 + (y)^2 / 8 = 1) ∧ 
  (x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3) := 
  sorry

end ellipse_standard_and_trajectory_l117_117154


namespace average_of_first_5_subjects_l117_117610

theorem average_of_first_5_subjects (avg_6_subjects : ℚ) (marks_6th_subject : ℚ) (total_subjects : ℕ) (total_marks_6_subjects : ℚ) (total_marks_5_subjects : ℚ) (avg_5_subjects : ℚ) :
  avg_6_subjects = 77 ∧ marks_6th_subject = 92 ∧ total_subjects = 6 ∧ total_marks_6_subjects = avg_6_subjects * total_subjects ∧ total_marks_5_subjects = total_marks_6_subjects - marks_6th_subject ∧ avg_5_subjects = total_marks_5_subjects / 5
  → avg_5_subjects = 74 := by
  sorry

end average_of_first_5_subjects_l117_117610


namespace probability_digits_different_l117_117371

theorem probability_digits_different : 
  let total_numbers := 490
  let same_digits_numbers := 13
  let different_digits_numbers := total_numbers - same_digits_numbers 
  let probability := different_digits_numbers / total_numbers 
  probability = 477 / 490 :=
by
  sorry

end probability_digits_different_l117_117371


namespace no_primes_of_form_2pow5m_plus_2powm_plus_1_l117_117892

theorem no_primes_of_form_2pow5m_plus_2powm_plus_1 {m : ℕ} (hm : m > 0) : ¬ (Prime (2^(5*m) + 2^m + 1)) :=
by
  sorry

end no_primes_of_form_2pow5m_plus_2powm_plus_1_l117_117892


namespace geometric_sequence_formula_l117_117929

theorem geometric_sequence_formula (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 1 + a 1 * q + a 1 * q^2 = 9 / 2)
  (geo : ∀ n, a (n + 1) = a n * q) :
  ∀ n, a n = 3 / 2 * (-2)^(n-1) ∨ a n = 3 / 2 :=
by sorry

end geometric_sequence_formula_l117_117929


namespace expression_is_correct_l117_117440

theorem expression_is_correct (a : ℝ) : 2 * (a + 1) = 2 * a + 1 := 
sorry

end expression_is_correct_l117_117440


namespace ratio_equal_one_of_log_conditions_l117_117247

noncomputable def logBase (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem ratio_equal_one_of_log_conditions
  (p q : ℝ)
  (hp : 0 < p)
  (hq : 0 < q)
  (h : logBase 8 p = logBase 18 q ∧ logBase 18 q = logBase 24 (p + 2 * q)) :
  q / p = 1 :=
by
  sorry

end ratio_equal_one_of_log_conditions_l117_117247


namespace base_s_computation_l117_117119

theorem base_s_computation (s : ℕ) (h : 550 * s + 420 * s = 1100 * s) : s = 7 := by
  sorry

end base_s_computation_l117_117119


namespace total_bird_count_correct_l117_117725

-- Define initial counts
def initial_sparrows : ℕ := 89
def initial_pigeons : ℕ := 68
def initial_finches : ℕ := 74

-- Define additional birds
def additional_sparrows : ℕ := 42
def additional_pigeons : ℕ := 51
def additional_finches : ℕ := 27

-- Define total counts
def initial_total : ℕ := 231
def final_total : ℕ := 312

theorem total_bird_count_correct :
  initial_sparrows + initial_pigeons + initial_finches = initial_total ∧
  (initial_sparrows + additional_sparrows) + 
  (initial_pigeons + additional_pigeons) + 
  (initial_finches + additional_finches) = final_total := by
    sorry

end total_bird_count_correct_l117_117725


namespace soccer_tournament_eq_l117_117262

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end soccer_tournament_eq_l117_117262


namespace area_excircle_gteq_four_times_area_l117_117729

-- Define the area function
def area (A B C : Point) : ℝ := sorry -- Area of triangle ABC (this will be implemented later)

-- Define the centers of the excircles (this needs precise definitions and setup)
def excircle_center (A B C : Point) : Point := sorry -- Centers of the excircles of triangle ABC (implementation would follow)

-- Define the area of the triangle formed by the excircle centers
def excircle_area (A B C : Point) : ℝ :=
  let O1 := excircle_center A B C
  let O2 := excircle_center B C A
  let O3 := excircle_center C A B
  area O1 O2 O3

-- Prove the main statement
theorem area_excircle_gteq_four_times_area (A B C : Point) :
  excircle_area A B C ≥ 4 * area A B C :=
by sorry

end area_excircle_gteq_four_times_area_l117_117729


namespace maximal_value_fraction_l117_117652

noncomputable def maximum_value_ratio (a b c : ℝ) (S : ℝ) : ℝ :=
  if S = c^2 / 4 then 2 * Real.sqrt 2 else 0

theorem maximal_value_fraction (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (area_cond : 1/2 * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c^2 / 4) :
  maximum_value_ratio a b c (c^2/4) = 2 * Real.sqrt 2 :=
sorry

end maximal_value_fraction_l117_117652


namespace algebraic_expression_value_l117_117310

theorem algebraic_expression_value 
  (p q r s : ℝ) 
  (hpq3 : p^2 / q^3 = 4 / 5) 
  (hrs2 : r^3 / s^2 = 7 / 9) : 
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := 
by 
  sorry

end algebraic_expression_value_l117_117310


namespace max_expression_value_l117_117464

theorem max_expression_value {x y : ℝ} (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) :
  x^2 + y^2 ≤ 10 :=
sorry

end max_expression_value_l117_117464


namespace no_real_solution_l117_117221

theorem no_real_solution (x : ℝ) : 
  x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 → 
  ¬ (
    (1 / ((x - 1) * (x - 3))) + (1 / ((x - 3) * (x - 5))) + (1 / ((x - 5) * (x - 7))) = 1 / 4
  ) :=
by sorry

end no_real_solution_l117_117221


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l117_117343

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l117_117343


namespace find_q_minus_p_l117_117865

theorem find_q_minus_p (p q : ℕ) (h1 : 0 < p) (h2 : 0 < q) 
  (h3 : 6 * q < 11 * p) (h4 : 9 * p < 5 * q) (h_min : ∀ r : ℕ, r > 0 → (6:ℚ)/11 < (p:ℚ)/r → (p:ℚ)/r < (5:ℚ)/9 → q ≤ r) :
  q - p = 9 :=
sorry

end find_q_minus_p_l117_117865


namespace more_apples_than_pears_l117_117877

-- Definitions based on conditions
def total_fruits : ℕ := 85
def apples : ℕ := 48

-- Statement to prove
theorem more_apples_than_pears : (apples - (total_fruits - apples)) = 11 := by
  -- proof steps
  sorry

end more_apples_than_pears_l117_117877


namespace sum_of_integers_is_28_l117_117433

theorem sum_of_integers_is_28 (m n p q : ℕ) (hmnpq_diff : m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q)
  (hm_pos : 0 < m) (hn_pos : 0 < n) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_prod : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 :=
by
  sorry

end sum_of_integers_is_28_l117_117433


namespace AK_eq_CK_l117_117737

variable {α : Type*} [LinearOrder α] [LinearOrder ℝ]

variable (A B C L K : ℝ)
variable (triangle : ℝ)
variable (h₁ : AL = LB)
variable (h₂ : AK = CL)

--  Given that in triangle ABC,
--     AL is a bisector such that AL = LB,
--     and AK is on ray AL with AK = CL,
--     prove that AK = CK.
theorem AK_eq_CK (h₁ : AL = LB) (h₂ : AK = CL) : AK = CK := by
  sorry

end AK_eq_CK_l117_117737


namespace quadratic_roots_opposite_signs_l117_117024

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ x * y < 0) ↔ (a < 0) :=
sorry

end quadratic_roots_opposite_signs_l117_117024


namespace gretchen_work_hours_l117_117342

noncomputable def walking_ratio (walking: ℤ) (sitting: ℤ) : Prop :=
  walking * 90 = sitting * 10

theorem gretchen_work_hours (walking_time: ℤ) (h: ℤ) (condition1: walking_ratio 40 (60 * h)) :
  h = 6 :=
by sorry

end gretchen_work_hours_l117_117342


namespace remainder_of_3a_minus_b_divided_by_5_l117_117648

theorem remainder_of_3a_minus_b_divided_by_5 (a b : ℕ) (m n : ℤ) 
(h1 : 3 * a > b) 
(h2 : a = 5 * m + 1) 
(h3 : b = 5 * n + 4) : 
(3 * a - b) % 5 = 4 := 
sorry

end remainder_of_3a_minus_b_divided_by_5_l117_117648


namespace f_odd_f_decreasing_f_max_min_l117_117810

noncomputable def f : ℝ → ℝ := sorry

lemma f_add (x y : ℝ) : f (x + y) = f x + f y := sorry
lemma f_neg1 : f (-1) = 2 := sorry
lemma f_positive_less_than_zero {x : ℝ} (hx : x > 0) : f x < 0 := sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem f_decreasing : ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 := sorry

theorem f_max_min : ∀ (f_max f_min : ℝ),
  f_max = f (-2) ∧ f_min = f 4 ∧
  f (-2) = 4 ∧ f 4 = -8 := sorry

end f_odd_f_decreasing_f_max_min_l117_117810


namespace isosceles_trapezoid_diagonal_length_l117_117761

theorem isosceles_trapezoid_diagonal_length
  (AB CD : ℝ) (AD BC : ℝ) :
  AB = 15 →
  CD = 9 →
  AD = 12 →
  BC = 12 →
  (AC : ℝ) = Real.sqrt 279 :=
by
  intros hAB hCD hAD hBC
  sorry

end isosceles_trapezoid_diagonal_length_l117_117761


namespace circle_transformation_l117_117626

theorem circle_transformation (c : ℝ × ℝ) (v : ℝ × ℝ) (h_center : c = (8, -3)) (h_vector : v = (2, -5)) :
  let reflected := (c.2, c.1)
  let translated := (reflected.1 + v.1, reflected.2 + v.2)
  translated = (-1, 3) :=
by
  sorry

end circle_transformation_l117_117626


namespace car_transport_distance_l117_117097

theorem car_transport_distance
  (d_birdhouse : ℕ) 
  (d_lawnchair : ℕ) 
  (d_car : ℕ)
  (h1 : d_birdhouse = 1200)
  (h2 : d_birdhouse = 3 * d_lawnchair)
  (h3 : d_lawnchair = 2 * d_car) :
  d_car = 200 := 
by
  sorry

end car_transport_distance_l117_117097


namespace find_a_l117_117835

theorem find_a
  (r1 r2 r3 : ℕ)
  (hr1 : r1 > 2) (hr2 : r2 > 2) (hr3 : r3 > 2)
  (a b c : ℤ)
  (hr : (Polynomial.X - Polynomial.C (r1 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r2 : ℤ)) * 
         (Polynomial.X - Polynomial.C (r3 : ℤ)) = 
         Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C c)
  (h : a + b + c + 1 = -2009) :
  a = -58 := sorry

end find_a_l117_117835


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l117_117848

-- Define the ink length of a figure
def ink_length (n : ℕ) : ℕ := 5 * n

-- Part (a): Determine the ink length of Figure 4.
theorem ink_length_figure_4 : ink_length 4 = 20 := by
  sorry

-- Part (b): Determine the difference between the ink length of Figure 9 and the ink length of Figure 8.
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 5 := by
  sorry

-- Part (c): Determine the ink length of Figure 100.
theorem ink_length_figure_100 : ink_length 100 = 500 := by
  sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l117_117848


namespace find_M_at_x_eq_3_l117_117472

noncomputable def M (a b c d x : ℝ) := a * x^5 + b * x^3 + c * x + d

theorem find_M_at_x_eq_3
  (a b c d M : ℝ)
  (h₀ : d = -5)
  (h₁ : 243 * a + 27 * b + 3 * c = -12) :
  M = -17 :=
by
  sorry

end find_M_at_x_eq_3_l117_117472


namespace total_bills_54_l117_117655

/-- A bank teller has some 5-dollar and 20-dollar bills in her cash drawer, 
and the total value of the bills is 780 dollars, with 20 5-dollar bills.
Show that the total number of bills is 54. -/
theorem total_bills_54 (value_total : ℕ) (num_5dollar : ℕ) (num_5dollar_value : ℕ) (num_20dollar : ℕ) :
    value_total = 780 ∧ num_5dollar = 20 ∧ num_5dollar_value = 5 ∧ num_20dollar * 20 + num_5dollar * num_5dollar_value = value_total
    → num_20dollar + num_5dollar = 54 :=
by
  sorry

end total_bills_54_l117_117655


namespace olivia_initial_quarters_l117_117193

theorem olivia_initial_quarters : 
  ∀ (spent_quarters left_quarters initial_quarters : ℕ),
  spent_quarters = 4 → left_quarters = 7 → initial_quarters = spent_quarters + left_quarters → initial_quarters = 11 :=
by
  intros spent_quarters left_quarters initial_quarters h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end olivia_initial_quarters_l117_117193


namespace intersection_M_N_eq_M_l117_117274

-- Definition of M
def M := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Definition of N
def N := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_M_N_eq_M : (M ∩ N) = M :=
  sorry

end intersection_M_N_eq_M_l117_117274


namespace average_death_rate_l117_117275

-- Definitions and given conditions
def birth_rate_per_two_seconds := 6
def net_increase_per_day := 172800

-- Calculate number of seconds in a day as a constant
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the net increase per second
def net_increase_per_second : ℕ := net_increase_per_day / seconds_per_day

-- Define the birth rate per second
def birth_rate_per_second : ℕ := birth_rate_per_two_seconds / 2

-- The final proof statement
theorem average_death_rate : 
  ∃ (death_rate_per_two_seconds : ℕ), 
    death_rate_per_two_seconds = birth_rate_per_two_seconds - 2 * net_increase_per_second := 
by 
  -- We are required to prove this statement
  use (birth_rate_per_second - net_increase_per_second) * 2
  sorry

end average_death_rate_l117_117275


namespace generating_sets_Z2_l117_117585

theorem generating_sets_Z2 (a b : ℤ × ℤ) (h : Submodule.span ℤ ({a, b} : Set (ℤ × ℤ)) = ⊤) :
  let a₁ := a.1
  let a₂ := a.2
  let b₁ := b.1
  let b₂ := b.2
  a₁ * b₂ - a₂ * b₁ = 1 ∨ a₁ * b₂ - a₂ * b₁ = -1 := 
by
  sorry

end generating_sets_Z2_l117_117585


namespace second_offset_l117_117964

theorem second_offset (d : ℝ) (h1 : ℝ) (A : ℝ) (h2 : ℝ) : 
  d = 28 → h1 = 9 → A = 210 → h2 = 6 :=
by
  sorry

end second_offset_l117_117964


namespace total_chairs_agreed_proof_l117_117080

/-
Conditions:
- Carey moved 28 chairs
- Pat moved 29 chairs
- They have 17 chairs left to move
Question:
- How many chairs did they agree to move in total?
Proof Problem:
- Prove that the total number of chairs they agreed to move is equal to 74.
-/

def carey_chairs : ℕ := 28
def pat_chairs : ℕ := 29
def chairs_left : ℕ := 17
def total_chairs_agreed : ℕ := carey_chairs + pat_chairs + chairs_left

theorem total_chairs_agreed_proof : total_chairs_agreed = 74 := 
by
  sorry

end total_chairs_agreed_proof_l117_117080


namespace tom_remaining_balloons_l117_117245

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end tom_remaining_balloons_l117_117245


namespace four_thirds_of_nine_halves_l117_117138

theorem four_thirds_of_nine_halves :
  (4 / 3) * (9 / 2) = 6 := 
sorry

end four_thirds_of_nine_halves_l117_117138


namespace reciprocal_is_correct_l117_117605

-- Define the initial number
def num : ℚ := -1 / 2023

-- Define the expected reciprocal
def reciprocal : ℚ := -2023

-- Theorem stating the reciprocal of the given number is the expected reciprocal
theorem reciprocal_is_correct : 1 / num = reciprocal :=
  by
    -- The actual proof can be filled in here
    sorry

end reciprocal_is_correct_l117_117605


namespace people_got_on_at_third_stop_l117_117853

theorem people_got_on_at_third_stop
  (initial : ℕ)
  (got_off_first : ℕ)
  (got_off_second : ℕ)
  (got_on_second : ℕ)
  (got_off_third : ℕ)
  (people_after_third : ℕ) :
  initial = 50 →
  got_off_first = 15 →
  got_off_second = 8 →
  got_on_second = 2 →
  got_off_third = 4 →
  people_after_third = 28 →
  ∃ got_on_third : ℕ, got_on_third = 3 :=
by
  sorry

end people_got_on_at_third_stop_l117_117853


namespace chessboard_property_exists_l117_117687

theorem chessboard_property_exists (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ i j, x i j = t i - t j := 
sorry

end chessboard_property_exists_l117_117687


namespace abs_inequality_solution_set_l117_117314

theorem abs_inequality_solution_set (x : ℝ) : |x - 1| > 2 ↔ x > 3 ∨ x < -1 :=
by
  sorry

end abs_inequality_solution_set_l117_117314


namespace solve_system_l117_117593

def eq1 (x y : ℝ) : Prop := x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0
def eq2 (x y : ℝ) : Prop := x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0

theorem solve_system :
  ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 4 ∧ y = 1 := by
  sorry

end solve_system_l117_117593


namespace geometric_series_sum_l117_117941

theorem geometric_series_sum :
  let a := 3
  let r := -2
  let n := 10
  let S := a * ((r^n - 1) / (r - 1))
  S = -1023 :=
by 
  -- Sorry allows us to omit the proof details
  sorry

end geometric_series_sum_l117_117941


namespace find_k_l117_117520

theorem find_k 
  (h : ∀ x, 2 * x ^ 2 + 14 * x + k = 0 → x = ((-14 + Real.sqrt 10) / 4) ∨ x = ((-14 - Real.sqrt 10) / 4)) :
  k = 93 / 4 :=
sorry

end find_k_l117_117520


namespace am_gm_inequality_example_l117_117410

theorem am_gm_inequality_example (x1 x2 x3 : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h_sum1 : x1 + x2 + x3 = 1) :
  (x2^2 / x1) + (x3^2 / x2) + (x1^2 / x3) ≥ 1 :=
by
  sorry

end am_gm_inequality_example_l117_117410


namespace complement_of_M_in_U_is_1_4_l117_117535

-- Define U
def U : Set ℕ := {x | x < 5 ∧ x ≠ 0}

-- Define M
def M : Set ℕ := {x | x^2 - 5 * x + 6 = 0}

-- The complement of M in U
def complement_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem complement_of_M_in_U_is_1_4 : complement_U_M = {1, 4} := 
by sorry

end complement_of_M_in_U_is_1_4_l117_117535


namespace allison_craft_items_l117_117273

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l117_117273


namespace calculate_area_of_triangle_l117_117067

theorem calculate_area_of_triangle :
  let p1 := (5, -2)
  let p2 := (5, 8)
  let p3 := (12, 8)
  let area := (1 / 2) * ((p2.2 - p1.2) * (p3.1 - p2.1))
  area = 35 := 
by
  sorry

end calculate_area_of_triangle_l117_117067


namespace number_of_bouquets_l117_117143

theorem number_of_bouquets : ∃ n, n = 9 ∧ ∀ x y : ℕ, 3 * x + 2 * y = 50 → (x < 17) ∧ (x % 2 = 0 → y = (50 - 3 * x) / 2) :=
by
  sorry

end number_of_bouquets_l117_117143


namespace proposition_only_A_l117_117953

def is_proposition (statement : String) : Prop := sorry

def statement_A : String := "Red beans grow in the southern country"
def statement_B : String := "They sprout several branches in spring"
def statement_C : String := "I hope you pick more"
def statement_D : String := "For these beans symbolize longing"

theorem proposition_only_A :
  is_proposition statement_A ∧
  ¬is_proposition statement_B ∧
  ¬is_proposition statement_C ∧
  ¬is_proposition statement_D := 
sorry

end proposition_only_A_l117_117953


namespace smallest_possible_N_l117_117775

theorem smallest_possible_N {p q r s t : ℕ} (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (ht: 0 < t) 
  (sum_eq: p + q + r + s + t = 3015) :
  ∃ N, N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ N = 1508 := 
sorry

end smallest_possible_N_l117_117775


namespace percent_difference_l117_117674

def boys := 100
def girls := 125
def diff := girls - boys
def boys_less_than_girls_percent := (diff : ℚ) / girls  * 100
def girls_more_than_boys_percent := (diff : ℚ) / boys  * 100

theorem percent_difference :
  boys_less_than_girls_percent = 20 ∧ girls_more_than_boys_percent = 25 :=
by
  -- The proof here demonstrates the percentage calculations.
  sorry

end percent_difference_l117_117674


namespace number_of_sheets_l117_117249

theorem number_of_sheets (S E : ℕ) 
  (h1 : S - E = 40)
  (h2 : 5 * E = S) : 
  S = 50 := by 
  sorry

end number_of_sheets_l117_117249


namespace minimum_value_2a_plus_3b_is_25_l117_117039

noncomputable def minimum_value_2a_plus_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / a) + (3 / b) = 1) : ℝ :=
2 * a + 3 * b

theorem minimum_value_2a_plus_3b_is_25
  (a b : ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (2 / a) + (3 / b) = 1) :
  minimum_value_2a_plus_3b a b h₁ h₂ h₃ = 25 :=
sorry

end minimum_value_2a_plus_3b_is_25_l117_117039


namespace girls_joined_school_l117_117491

theorem girls_joined_school
  (initial_girls : ℕ)
  (initial_boys : ℕ)
  (total_pupils_after : ℕ)
  (computed_new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  total_pupils_after = 1346 →
  computed_new_girls = total_pupils_after - (initial_girls + initial_boys) →
  computed_new_girls = 418 :=
by
  intros h_initial_girls h_initial_boys h_total_pupils_after h_computed_new_girls
  sorry

end girls_joined_school_l117_117491


namespace least_distance_between_ticks_l117_117389

theorem least_distance_between_ticks :
  ∃ z : ℝ, ∀ (a b : ℤ), (a / 5 ≠ b / 7) → abs (a / 5 - b / 7) = (1 / 35) := 
sorry

end least_distance_between_ticks_l117_117389


namespace operation_X_value_l117_117904

def operation_X (a b : ℤ) : ℤ := b + 7 * a - a^3 + 2 * b

theorem operation_X_value : operation_X 4 3 = -27 := by
  sorry

end operation_X_value_l117_117904


namespace xy_positive_l117_117004

theorem xy_positive (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 :=
sorry

end xy_positive_l117_117004


namespace relationship_between_p_and_q_l117_117812

variable (x y : ℝ)

def p := x * y ≥ 0
def q := |x + y| = |x| + |y|

theorem relationship_between_p_and_q : (p x y ↔ q x y) :=
sorry

end relationship_between_p_and_q_l117_117812


namespace initial_amount_of_liquid_A_l117_117839

theorem initial_amount_of_liquid_A (A B : ℝ) (initial_ratio : A = 4 * B) (removed_mixture : ℝ) (new_ratio : (A - (4/5) * removed_mixture) = (2 / 3) * ((B - (1/5) * removed_mixture) + removed_mixture)) :
  A = 16 := 
  sorry

end initial_amount_of_liquid_A_l117_117839


namespace determine_abc_l117_117799

-- Definitions
def parabola_equation (a b c : ℝ) (y : ℝ) : ℝ := a * y^2 + b * y + c

def vertex_condition (a b c : ℝ) : Prop :=
  ∀ y, parabola_equation a b c y = a * (y + 6)^2 + 3

def point_condition (a b c : ℝ) : Prop :=
  parabola_equation a b c (-6) = 3 ∧ parabola_equation a b c (-4) = 2

-- Proposition to prove
theorem determine_abc : 
  ∃ a b c : ℝ, vertex_condition a b c ∧ point_condition a b c
  ∧ (a + b + c = -25/4) :=
sorry

end determine_abc_l117_117799


namespace solve_prime_equation_l117_117019

theorem solve_prime_equation (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x^3 + y^3 - 3 * x * y = p - 1 ↔
  (x = 1 ∧ y = 0 ∧ p = 2) ∨
  (x = 0 ∧ y = 1 ∧ p = 2) ∨
  (x = 2 ∧ y = 2 ∧ p = 5) := 
sorry

end solve_prime_equation_l117_117019


namespace probability_two_red_marbles_l117_117495

theorem probability_two_red_marbles
  (red_marbles : ℕ)
  (white_marbles : ℕ)
  (total_marbles : ℕ)
  (prob_first_red : ℚ)
  (prob_second_red_after_first_red : ℚ)
  (combined_probability : ℚ) :
  red_marbles = 5 →
  white_marbles = 7 →
  total_marbles = 12 →
  prob_first_red = 5 / 12 →
  prob_second_red_after_first_red = 4 / 11 →
  combined_probability = 5 / 33 →
  combined_probability = prob_first_red * prob_second_red_after_first_red := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end probability_two_red_marbles_l117_117495


namespace ratio_of_areas_l117_117795

variable (s : ℝ)
def side_length_square := s
def side_length_longer_rect := 1.2 * s
def side_length_shorter_rect := 0.7 * s
def area_square := s^2
def area_rect := (1.2 * s) * (0.7 * s)

theorem ratio_of_areas (h1 : s > 0) :
  area_rect s / area_square s = 21 / 25 :=
by 
  sorry

end ratio_of_areas_l117_117795


namespace exponential_inequality_example_l117_117681

theorem exponential_inequality_example (a b : ℝ) (h : 1.5 > 0 ∧ 1.5 ≠ 1) (h2 : 2.3 < 3.2) : 1.5 ^ 2.3 < 1.5 ^ 3.2 :=
by 
  sorry

end exponential_inequality_example_l117_117681


namespace range_of_m_iff_l117_117627

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (0 < x) → (0 < y) → ((2 / x) + (1 / y) = 1) → (x + 2 * y > m^2 + 2 * m)

theorem range_of_m_iff : (range_of_m m) ↔ (-4 < m ∧ m < 2) :=
  sorry

end range_of_m_iff_l117_117627


namespace functions_are_same_l117_117257

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end functions_are_same_l117_117257


namespace potato_difference_l117_117252

def x := 8 * 13
def k := (67 - 13) / 2
def z := 20 * k
def d := z - x

theorem potato_difference : d = 436 :=
by
  sorry

end potato_difference_l117_117252


namespace jane_received_change_l117_117136

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l117_117136


namespace asymptotes_of_hyperbola_l117_117001

theorem asymptotes_of_hyperbola :
  (∀ x y : ℝ, (x^2 / 16 - y^2 / 25 = 1) →
    (y = (5 / 4) * x ∨ y = -(5 / 4) * x)) :=
by
  intro x y h
  sorry

end asymptotes_of_hyperbola_l117_117001


namespace multiply_powers_same_base_l117_117071

theorem multiply_powers_same_base (a : ℝ) : a^4 * a^2 = a^6 := by
  sorry

end multiply_powers_same_base_l117_117071


namespace hummus_serving_amount_proof_l117_117688

/-- Given conditions: 
    one_can is the number of ounces of chickpeas in one can,
    total_cans is the number of cans Thomas buys,
    total_servings is the number of servings of hummus Thomas needs to make,
    to_produce_one_serving is the amount of chickpeas needed for one serving,
    we prove that to_produce_one_serving = 6.4 given the above conditions. -/
theorem hummus_serving_amount_proof 
  (one_can : ℕ) 
  (total_cans : ℕ) 
  (total_servings : ℕ) 
  (to_produce_one_serving : ℚ) 
  (h_one_can : one_can = 16) 
  (h_total_cans : total_cans = 8)
  (h_total_servings : total_servings = 20) 
  (h_total_ounces : total_cans * one_can = 128) : 
  to_produce_one_serving = 128 / 20 := 
by
  sorry

end hummus_serving_amount_proof_l117_117688


namespace smallest_b_for_composite_l117_117048

theorem smallest_b_for_composite (x : ℤ) : 
  ∃ b : ℕ, b > 0 ∧ Even b ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^2)) ∧ b = 16 := 
by 
  sorry

end smallest_b_for_composite_l117_117048


namespace bird_families_difference_l117_117439

theorem bird_families_difference {initial_families flown_away : ℕ} (h1 : initial_families = 87) (h2 : flown_away = 7) :
  (initial_families - flown_away) - flown_away = 73 := by
sorry

end bird_families_difference_l117_117439


namespace river_width_l117_117957

theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) 
  (h1 : depth = 2) 
  (h2 : flow_rate = 4000 / 60)  -- Flow rate in meters per minute
  (h3 : volume_per_minute = 6000) :
  volume_per_minute / (flow_rate * depth) = 45 :=
by
  sorry

end river_width_l117_117957


namespace consecutive_sum_divisible_by_12_l117_117443

theorem consecutive_sum_divisible_by_12 
  (b : ℤ) 
  (a : ℤ := b - 1) 
  (c : ℤ := b + 1) 
  (d : ℤ := b + 2) :
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k := by
  sorry

end consecutive_sum_divisible_by_12_l117_117443


namespace cos_2alpha_plus_pi_over_3_l117_117992

open Real

theorem cos_2alpha_plus_pi_over_3 
  (alpha : ℝ) 
  (h1 : cos (alpha - π / 12) = 3 / 5) 
  (h2 : 0 < alpha ∧ alpha < π / 2) : 
  cos (2 * alpha + π / 3) = -24 / 25 := 
sorry

end cos_2alpha_plus_pi_over_3_l117_117992


namespace julie_money_left_l117_117268

def cost_of_bike : ℕ := 2345
def initial_savings : ℕ := 1500

def mowing_rate : ℕ := 20
def mowing_jobs : ℕ := 20

def paper_rate : ℚ := 0.40
def paper_jobs : ℕ := 600

def dog_rate : ℕ := 15
def dog_jobs : ℕ := 24

def earnings_from_mowing : ℕ := mowing_rate * mowing_jobs
def earnings_from_papers : ℚ := paper_rate * paper_jobs
def earnings_from_dogs : ℕ := dog_rate * dog_jobs

def total_earnings : ℚ := earnings_from_mowing + earnings_from_papers + earnings_from_dogs
def total_money_available : ℚ := initial_savings + total_earnings

def money_left_after_purchase : ℚ := total_money_available - cost_of_bike

theorem julie_money_left : money_left_after_purchase = 155 := sorry

end julie_money_left_l117_117268


namespace find_k_l117_117281

-- Defining the vectors and the condition for parallelism
def vector_a := (2, 1)
def vector_b (k : ℝ) := (k, 3)

def vector_parallel_condition (k : ℝ) : Prop :=
  let a2b := (2 + 2 * k, 7)
  let a2nb := (4 - k, -1)
  (2 + 2 * k) * (-1) = 7 * (4 - k)

theorem find_k (k : ℝ) (h : vector_parallel_condition k) : k = 6 :=
by
  sorry

end find_k_l117_117281


namespace sum_of_interior_edges_l117_117527

noncomputable def interior_edge_sum (outer_length : ℝ) (wood_width : ℝ) (frame_area : ℝ) : ℝ := 
  let outer_width := (frame_area + 3 * (outer_length - 2 * wood_width) * 4) / outer_length
  let inner_length := outer_length - 2 * wood_width
  let inner_width := outer_width - 2 * wood_width
  2 * inner_length + 2 * inner_width

theorem sum_of_interior_edges :
  interior_edge_sum 7 2 34 = 9 := by
  sorry

end sum_of_interior_edges_l117_117527


namespace ShielaDrawingsPerNeighbor_l117_117479

-- Defining our problem using the given conditions:
def ShielaTotalDrawings : ℕ := 54
def ShielaNeighbors : ℕ := 6

-- Mathematically restating the problem:
theorem ShielaDrawingsPerNeighbor : (ShielaTotalDrawings / ShielaNeighbors) = 9 := by
  sorry

end ShielaDrawingsPerNeighbor_l117_117479


namespace triangle_perimeter_l117_117987

theorem triangle_perimeter (A r p : ℝ) (hA : A = 60) (hr : r = 2.5) (h_eq : A = r * p / 2) : p = 48 := 
by
  sorry

end triangle_perimeter_l117_117987


namespace value_of_y_l117_117188

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 :=
by
  sorry

end value_of_y_l117_117188


namespace non_neg_integers_l117_117950

open Nat

theorem non_neg_integers (n : ℕ) :
  (∃ x y k : ℕ, x.gcd y = 1 ∧ k ≥ 2 ∧ 3^n = x^k + y^k) ↔ (n = 0 ∨ n = 1 ∨ n = 2) := by
  sorry

end non_neg_integers_l117_117950


namespace meaningful_expression_range_l117_117261

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end meaningful_expression_range_l117_117261


namespace cyclic_permutations_sum_41234_l117_117364

theorem cyclic_permutations_sum_41234 :
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  3 * (n1 + n2 + n3 + n4) = 396618 :=
by
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  show 3 * (n1 + n2 + n3 + n4) = 396618
  sorry

end cyclic_permutations_sum_41234_l117_117364


namespace product_increase_by_13_exists_l117_117400

theorem product_increase_by_13_exists :
  ∃ a1 a2 a3 a4 a5 a6 a7 : ℕ,
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * (a1 * a2 * a3 * a4 * a5 * a6 * a7)) :=
by
  sorry

end product_increase_by_13_exists_l117_117400


namespace price_per_strawberry_basket_is_9_l117_117305

-- Define the conditions
def strawberry_plants := 5
def tomato_plants := 7
def strawberries_per_plant := 14
def tomatoes_per_plant := 16
def items_per_basket := 7
def price_per_tomato_basket := 6
def total_revenue := 186

-- Define the total number of strawberries and tomatoes harvested
def total_strawberries := strawberry_plants * strawberries_per_plant
def total_tomatoes := tomato_plants * tomatoes_per_plant

-- Define the number of baskets of strawberries and tomatoes
def strawberry_baskets := total_strawberries / items_per_basket
def tomato_baskets := total_tomatoes / items_per_basket

-- Define the revenue from tomato baskets
def revenue_tomatoes := tomato_baskets * price_per_tomato_basket

-- Define the revenue from strawberry baskets
def revenue_strawberries := total_revenue - revenue_tomatoes

-- Calculate the price per basket of strawberries (which should be $9)
def price_per_strawberry_basket := revenue_strawberries / strawberry_baskets

theorem price_per_strawberry_basket_is_9 : 
  price_per_strawberry_basket = 9 := by
    sorry

end price_per_strawberry_basket_is_9_l117_117305


namespace problem_value_l117_117036

theorem problem_value:
  3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 :=
by sorry

end problem_value_l117_117036


namespace expression_equals_one_l117_117690

theorem expression_equals_one : 
  (Real.sqrt 6 / Real.sqrt 2) + abs (1 - Real.sqrt 3) - Real.sqrt 12 + (1 / 2)⁻¹ = 1 := 
by sorry

end expression_equals_one_l117_117690


namespace geometric_sequence_property_l117_117689

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def S_n (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * (1 - 3^n)) / (1 - 3)

theorem geometric_sequence_property 
  (h₁ : a 1 + a 2 + a 3 = 26)
  (h₂ : S 6 = 728)
  (h₃ : ∀ n, a n = a_n n)
  (h₄ : ∀ n, S n = S_n n) :
  ∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n :=
by sorry

end geometric_sequence_property_l117_117689


namespace monotonicity_of_f_range_of_a_l117_117783

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) - a * x

theorem monotonicity_of_f (a : ℝ) (ha : a ≠ 0) :
  (∀ x < 0, f a x ≥ f a (x + 1)) ∧ (∀ x > 0, f a x ≤ f a (x + 1)) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ Real.sin x - Real.cos x + 2 - a * x) ↔ a ∈ Set.Ici 1 :=
sorry

end monotonicity_of_f_range_of_a_l117_117783


namespace fixed_point_of_line_l117_117269

theorem fixed_point_of_line (a : ℝ) (x y : ℝ)
  (h : ∀ a : ℝ, a * x + y + 1 = 0) :
  x = 0 ∧ y = -1 := 
by
  sorry

end fixed_point_of_line_l117_117269


namespace find_r_l117_117481

theorem find_r (b r : ℝ) (h1 : b / (1 - r) = 18) (h2 : b * r^2 / (1 - r^2) = 6) : r = 1/2 :=
by
  sorry

end find_r_l117_117481


namespace onewaynia_road_closure_l117_117111

variable {V : Type} -- Denoting the type of cities
variable (G : V → V → Prop) -- G represents the directed graph

-- Conditions
variables (outdegree : V → Nat) (indegree : V → Nat)
variables (two_ways : ∀ (u v : V), u ≠ v → ¬(G u v ∧ G v u))
variables (two_out : ∀ v : V, outdegree v = 2)
variables (two_in : ∀ v : V, indegree v = 2)

theorem onewaynia_road_closure:
  ∃ n : Nat, n ≥ 1 ∧ (number_of_closures : Nat) = 2 ^ n :=
by
  sorry

end onewaynia_road_closure_l117_117111


namespace plane_ratio_l117_117393

section

variables (D B T P : ℕ)

-- Given conditions
axiom total_distance : D = 1800
axiom distance_by_bus : B = 720
axiom distance_by_train : T = (2 * B) / 3

-- Prove the ratio of the distance traveled by plane to the whole trip
theorem plane_ratio :
  D = 1800 →
  B = 720 →
  T = (2 * B) / 3 →
  P = D - (T + B) →
  P / D = 1 / 3 := by
  intros h1 h2 h3 h4
  sorry

end

end plane_ratio_l117_117393


namespace common_ratio_of_geometric_sequence_l117_117717

variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b_n (n + 1) = b_n n * r

def arithmetic_to_geometric (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  b_n 0 = a_n 2 ∧ b_n 1 = a_n 3 ∧ b_n 2 = a_n 7

-- Mathematical Proof Problem
theorem common_ratio_of_geometric_sequence :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), d ≠ 0 →
  is_arithmetic_sequence a_n d →
  (∃ (b_n : ℕ → ℝ) (r : ℝ), arithmetic_to_geometric a_n b_n ∧ is_geometric_sequence b_n r) →
  ∃ r, r = 4 :=
sorry

end common_ratio_of_geometric_sequence_l117_117717


namespace diameter_inscribed_circle_l117_117937

noncomputable def diameter_of_circle (r : ℝ) : ℝ :=
2 * r

theorem diameter_inscribed_circle (r : ℝ) (h : 8 * r = π * r ^ 2) : diameter_of_circle r = 16 / π := by
  sorry

end diameter_inscribed_circle_l117_117937


namespace max_area_of_triangle_l117_117320

theorem max_area_of_triangle (a c : ℝ)
    (h1 : a^2 + c^2 = 16 + a * c) : 
    ∃ s : ℝ, s = 4 * Real.sqrt 3 := by
  sorry

end max_area_of_triangle_l117_117320


namespace parity_equivalence_l117_117523

theorem parity_equivalence (p q : ℕ) :
  (Even (p^3 - q^3)) ↔ (Even (p + q)) :=
by
  sorry

end parity_equivalence_l117_117523


namespace area_of_EFCD_l117_117954

noncomputable def area_of_quadrilateral (AB CD altitude: ℝ) :=
  let sum_bases_half := (AB + CD) / 2
  let small_altitude := altitude / 2
  small_altitude * (sum_bases_half + CD) / 2

theorem area_of_EFCD
  (AB CD altitude : ℝ)
  (AB_len : AB = 10)
  (CD_len : CD = 24)
  (altitude_len : altitude = 15)
  : area_of_quadrilateral AB CD altitude = 153.75 :=
by
  rw [AB_len, CD_len, altitude_len]
  simp [area_of_quadrilateral]
  sorry

end area_of_EFCD_l117_117954


namespace baba_yaga_departure_and_speed_l117_117744

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end baba_yaga_departure_and_speed_l117_117744


namespace line_through_point_parallel_to_line_l117_117451

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end line_through_point_parallel_to_line_l117_117451


namespace number_of_triangles_and_squares_l117_117461

theorem number_of_triangles_and_squares (x y : ℕ) (h1 : x + y = 13) (h2 : 3 * x + 4 * y = 47) : 
  x = 5 ∧ y = 8 :=
by
  sorry

end number_of_triangles_and_squares_l117_117461


namespace present_age_of_dan_l117_117082

theorem present_age_of_dan (x : ℕ) : (x + 16 = 4 * (x - 8)) → x = 16 :=
by
  intro h
  sorry

end present_age_of_dan_l117_117082


namespace area_ratio_of_squares_l117_117339

open Real

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 4 * 4 * b) : (a^2) / (b^2) = 16 := 
by
  sorry

end area_ratio_of_squares_l117_117339


namespace number_of_ah_tribe_residents_l117_117895

theorem number_of_ah_tribe_residents 
  (P A U : Nat) 
  (H1 : 16 < P) 
  (H2 : P ≤ 17) 
  (H3 : A + U = P) 
  (H4 : U = 2) : 
  A = 15 := 
by
  sorry

end number_of_ah_tribe_residents_l117_117895


namespace min_students_in_group_l117_117326

theorem min_students_in_group 
  (g1 g2 : ℕ) 
  (n1 n2 e1 e2 f1 f2 : ℕ)
  (H_equal_groups : g1 = g2)
  (H_both_languages_g1 : n1 = 5)
  (H_both_languages_g2 : n2 = 5)
  (H_french_students : f1 * 3 = f2)
  (H_english_students : e1 = 4 * e2)
  (H_total_g1 : g1 = f1 + e1 - n1)
  (H_total_g2 : g2 = f2 + e2 - n2) 
: g1 = 28 :=
sorry

end min_students_in_group_l117_117326


namespace find_interval_l117_117420

theorem find_interval (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 :=
by
  sorry

end find_interval_l117_117420


namespace problem_statement_l117_117813

theorem problem_statement :
  (2 * 3 * 4) * (1/2 + 1/3 + 1/4) = 26 := by
  sorry

end problem_statement_l117_117813


namespace total_games_played_l117_117217

def games_lost : ℕ := 4
def games_won : ℕ := 8

theorem total_games_played : games_lost + games_won = 12 :=
by
  -- Proof is omitted
  sorry

end total_games_played_l117_117217


namespace combustion_CH₄_forming_water_l117_117925

/-
Combustion reaction for Methane: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Given:
  3 moles of Methane
  6 moles of Oxygen
  Balanced equation: CH₄ + 2 O₂ → CO₂ + 2 H₂O
Goal: Prove that 6 moles of Water (H₂O) are formed.
-/

-- Define the necessary definitions for the context
def moles_CH₄ : ℝ := 3
def moles_O₂ : ℝ := 6
def ratio_water_methane : ℝ := 2

theorem combustion_CH₄_forming_water :
  moles_CH₄ * ratio_water_methane = 6 :=
by
  sorry

end combustion_CH₄_forming_water_l117_117925


namespace no_odd_tens_digit_in_square_l117_117112

theorem no_odd_tens_digit_in_square (n : ℕ) (h₁ : n % 2 = 1) (h₂ : n > 0) (h₃ : n < 100) : 
  (n * n / 10) % 10 % 2 = 0 := 
sorry

end no_odd_tens_digit_in_square_l117_117112


namespace carol_rectangle_length_l117_117133

theorem carol_rectangle_length :
  let j_length := 6
  let j_width := 30
  let c_width := 15
  let c_length := j_length * j_width / c_width
  c_length = 12 := by
  sorry

end carol_rectangle_length_l117_117133


namespace toms_age_ratio_l117_117437

variable (T N : ℕ)

def toms_age_condition : Prop :=
  T = 3 * (T - 4 * N) + N

theorem toms_age_ratio (h : toms_age_condition T N) : T / N = 11 / 2 :=
by sorry

end toms_age_ratio_l117_117437


namespace remainder_13_pow_150_mod_11_l117_117513

theorem remainder_13_pow_150_mod_11 : (13^150) % 11 = 1 := 
by 
  sorry

end remainder_13_pow_150_mod_11_l117_117513


namespace spacesMovedBeforeSetback_l117_117000

-- Let's define the conditions as local constants
def totalSpaces : ℕ := 48
def firstTurnMove : ℕ := 8
def thirdTurnMove : ℕ := 6
def remainingSpacesToWin : ℕ := 37
def setback : ℕ := 5

theorem spacesMovedBeforeSetback (x : ℕ) : 
  (firstTurnMove + thirdTurnMove) + x - setback + remainingSpacesToWin = totalSpaces →
  x = 28 := by
  sorry

end spacesMovedBeforeSetback_l117_117000


namespace total_swordfish_catch_l117_117650

-- Definitions
def S_c : ℝ := 5 - 2
def S_m : ℝ := S_c - 1
def S_a : ℝ := 2 * S_m

def W_s : ℕ := 3  -- Number of sunny days
def W_r : ℕ := 2  -- Number of rainy days

-- Sunny and rainy day adjustments
def Shelly_sunny_catch : ℝ := S_c + 0.20 * S_c
def Sam_sunny_catch : ℝ := S_m + 0.20 * S_m
def Sara_sunny_catch : ℝ := S_a + 0.20 * S_a

def Shelly_rainy_catch : ℝ := S_c - 0.10 * S_c
def Sam_rainy_catch : ℝ := S_m - 0.10 * S_m
def Sara_rainy_catch : ℝ := S_a - 0.10 * S_a

-- Total catch calculations
def Shelly_total_catch : ℝ := W_s * Shelly_sunny_catch + W_r * Shelly_rainy_catch
def Sam_total_catch : ℝ := W_s * Sam_sunny_catch + W_r * Sam_rainy_catch
def Sara_total_catch : ℝ := W_s * Sara_sunny_catch + W_r * Sara_rainy_catch

def Total_catch : ℝ := Shelly_total_catch + Sam_total_catch + Sara_total_catch

-- Proof statement
theorem total_swordfish_catch : ⌊Total_catch⌋ = 48 := 
  by sorry

end total_swordfish_catch_l117_117650


namespace ratio_of_perimeters_l117_117882

theorem ratio_of_perimeters (s1 s2 : ℝ) (h : s1^2 / s2^2 = 16 / 81) : 
    s1 / s2 = 4 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_perimeters_l117_117882


namespace average_percentage_15_students_l117_117401

-- Define the average percentage of the 15 students
variable (x : ℝ)

-- Condition 1: Total percentage for the 15 students is 15 * x
def total_15_students : ℝ := 15 * x

-- Condition 2: Total percentage for the 10 students who averaged 88%
def total_10_students : ℝ := 10 * 88

-- Condition 3: Total percentage for all 25 students who averaged 79%
def total_all_students : ℝ := 25 * 79

-- Mathematical problem: Prove that x = 73 given the conditions.
theorem average_percentage_15_students (h : total_15_students x + total_10_students = total_all_students) : x = 73 := 
by
  sorry

end average_percentage_15_students_l117_117401


namespace smallest_enclosing_sphere_radius_l117_117656

-- Define the conditions
def sphere_radius : ℝ := 2

-- Define the sphere center coordinates in each octant
def sphere_centers : List (ℝ × ℝ × ℝ) :=
  [ (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
    (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2) ]

-- Define the theorem statement
theorem smallest_enclosing_sphere_radius :
  (∃ (r : ℝ), r = 2 * Real.sqrt 3 + 2) :=
by
  -- conditions and proof will go here
  sorry

end smallest_enclosing_sphere_radius_l117_117656


namespace factorization_result_l117_117606

theorem factorization_result (a b : ℤ) (h : (16:ℚ) * x^2 - 106 * x - 105 = (8 * x + a) * (2 * x + b)) : a + 2 * b = -23 := by
  sorry

end factorization_result_l117_117606


namespace triangle_area_proof_l117_117368

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1 / 2 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (h1 : b = 3) 
  (h2 : Real.cos B = 1 / 4) 
  (h3 : Real.sin C = 2 * Real.sin A) 
  (h4 : c = 2 * a) 
  (h5 : 9 = 5 * a ^ 2 - 4 * a ^ 2 * Real.cos B): 
  area_of_triangle a b c A B C = 9 * Real.sqrt 15 / 16 :=
by 
  sorry

end triangle_area_proof_l117_117368


namespace polynomial_transformation_l117_117163

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_transformation (x : ℝ) :
  (f (x^2 + 2) = x^4 + 6 * x^2 + 4) →
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  intro h
  sorry

end polynomial_transformation_l117_117163


namespace dan_helmet_craters_l117_117710

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end dan_helmet_craters_l117_117710


namespace dylan_trip_time_l117_117780

def total_time_of_trip (d1 d2 d3 v1 v2 v3 b : ℕ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let time_riding := t1 + t2 + t3
  let time_breaks := b * 25 / 60
  time_riding + time_breaks

theorem dylan_trip_time :
  total_time_of_trip 400 150 700 50 40 60 3 = 24.67 :=
by
  unfold total_time_of_trip
  sorry

end dylan_trip_time_l117_117780


namespace ryan_days_learning_l117_117306

-- Definitions based on conditions
def hours_per_day_chinese : ℕ := 4
def total_hours_chinese : ℕ := 24

-- Theorem stating the number of days Ryan learns
theorem ryan_days_learning : total_hours_chinese / hours_per_day_chinese = 6 := 
by 
  -- Divide the total hours spent on Chinese learning by hours per day
  sorry

end ryan_days_learning_l117_117306


namespace hypotenuse_is_2_sqrt_25_point_2_l117_117555

open Real

noncomputable def hypotenuse_length_of_right_triangle (ma mb : ℝ) (a b c : ℝ) : ℝ :=
  if h1 : ma = 6 ∧ mb = sqrt 27 then
    c
  else
    0

theorem hypotenuse_is_2_sqrt_25_point_2 :
  hypotenuse_length_of_right_triangle 6 (sqrt 27) a b (2 * sqrt 25.2) = 2 * sqrt 25.2 :=
by
  sorry -- proof to be filled

end hypotenuse_is_2_sqrt_25_point_2_l117_117555


namespace jim_less_than_anthony_l117_117236

-- Definitions for the conditions
def scott_shoes : ℕ := 7

def anthony_shoes : ℕ := 3 * scott_shoes

def jim_shoes : ℕ := anthony_shoes - 2

-- Lean statement to prove the problem
theorem jim_less_than_anthony : anthony_shoes - jim_shoes = 2 := by
  sorry

end jim_less_than_anthony_l117_117236


namespace solve_for_x_l117_117296

theorem solve_for_x (x : ℤ) (h : 3 * x + 20 = (1/3 : ℚ) * (7 * x + 60)) : x = 0 :=
sorry

end solve_for_x_l117_117296


namespace min_students_changed_l117_117639

-- Define the initial percentage of "Yes" and "No" at the beginning of the year
def initial_yes_percentage : ℝ := 0.40
def initial_no_percentage : ℝ := 0.60

-- Define the final percentage of "Yes" and "No" at the end of the year
def final_yes_percentage : ℝ := 0.80
def final_no_percentage : ℝ := 0.20

-- Define the minimum possible percentage of students that changed their mind
def min_changed_percentage : ℝ := 0.40

-- Prove that the minimum possible percentage of students that changed their mind is 40%
theorem min_students_changed :
  (final_yes_percentage - initial_yes_percentage = min_changed_percentage) ∧
  (initial_yes_percentage = final_yes_percentage - min_changed_percentage) ∧
  (initial_no_percentage - min_changed_percentage = final_no_percentage) :=
by
  sorry

end min_students_changed_l117_117639


namespace mrs_santiago_more_roses_l117_117564

theorem mrs_santiago_more_roses :
  58 - 24 = 34 :=
by 
  sorry

end mrs_santiago_more_roses_l117_117564


namespace parabola_range_proof_l117_117855

noncomputable def parabola_range (a : ℝ) : Prop := 
  (-2 ≤ a ∧ a < 3) → 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19)

theorem parabola_range_proof (a : ℝ) (h : -2 ≤ a ∧ a < 3) : 
  ∃ b : ℝ, b = a^2 + 2*a + 4 ∧ (3 ≤ b ∧ b < 19) :=
sorry

end parabola_range_proof_l117_117855


namespace bella_truck_stamps_more_l117_117107

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more_l117_117107


namespace equation_equivalence_l117_117827

theorem equation_equivalence (p q : ℝ) (hp₀ : p ≠ 0) (hp₅ : p ≠ 5) (hq₀ : q ≠ 0) (hq₇ : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) → p = 9 * q / (q - 15) :=
by
  sorry

end equation_equivalence_l117_117827


namespace negation_of_exists_l117_117675

theorem negation_of_exists:
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) := sorry

end negation_of_exists_l117_117675


namespace length_of_plot_l117_117137

theorem length_of_plot (W P C r : ℝ) (hW : W = 65) (hP : P = 2.5) (hC : C = 340) (hr : r = 0.4) :
  let L := (C / r - (W + 2 * P) * P) / (W - 2 * P)
  L = 100 :=
by
  sorry

end length_of_plot_l117_117137


namespace find_radius_l117_117819

-- Definition of the conditions
def area_of_sector : ℝ := 10 -- The area of the sector in square centimeters
def arc_length : ℝ := 4     -- The arc length of the sector in centimeters

-- The radius of the circle we want to prove
def radius (r : ℝ) : Prop :=
  (r * 4) / 2 = 10

-- The theorem to be proved
theorem find_radius : ∃ r : ℝ, radius r :=
by
  use 5
  unfold radius
  norm_num

end find_radius_l117_117819


namespace range_of_m_l117_117727
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1))^2

noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

theorem range_of_m {x : ℝ} (m : ℝ) (h1 : 1 / 16 ≤ x) (h2 : x ≤ 1 / 4) 
  (h3 : ∀ (x : ℝ), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)): 
  -1 < m ∧ m < 5 / 4 :=
sorry

end range_of_m_l117_117727


namespace license_plate_count_correct_l117_117009

def rotokas_letters : Finset Char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U'}

def valid_license_plate_count : ℕ :=
  let first_letter_choices := 2 -- Letters A or E
  let last_letter_fixed := 1 -- Fixed as P
  let remaining_letters := rotokas_letters.erase 'V' -- Exclude V
  let second_letter_choices := (remaining_letters.erase 'P').card - 1 -- Exclude P and first letter
  let third_letter_choices := second_letter_choices - 1
  let fourth_letter_choices := third_letter_choices - 1
  2 * 9 * 8 * 7

theorem license_plate_count_correct :
  valid_license_plate_count = 1008 := by
  sorry

end license_plate_count_correct_l117_117009


namespace arthur_hot_dogs_first_day_l117_117061

theorem arthur_hot_dogs_first_day (H D n : ℕ) (h₀ : D = 1)
(h₁ : 3 * H + n = 10)
(h₂ : 2 * H + 3 * D = 7) : n = 4 :=
by sorry

end arthur_hot_dogs_first_day_l117_117061


namespace find_k_value_l117_117286

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l117_117286


namespace prove_inequality_l117_117765

variable (f : ℝ → ℝ)

-- Conditions
axiom condition : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

-- Proof of the desired statement
theorem prove_inequality : ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
by
  sorry

end prove_inequality_l117_117765


namespace total_flowers_purchased_l117_117045

-- Define the conditions
def sets : ℕ := 3
def pieces_per_set : ℕ := 90

-- State the proof problem
theorem total_flowers_purchased : sets * pieces_per_set = 270 :=
by
  sorry

end total_flowers_purchased_l117_117045


namespace average_of_first_20_even_numbers_not_divisible_by_3_or_5_l117_117977

def first_20_valid_even_numbers : List ℕ :=
  [2, 4, 8, 14, 16, 22, 26, 28, 32, 34, 38, 44, 46, 52, 56, 58, 62, 64, 68, 74]

-- Check the sum of these numbers
def sum_first_20_valid_even_numbers : ℕ :=
  first_20_valid_even_numbers.sum

-- Define average calculation
def average_first_20_valid_even_numbers : ℕ :=
  sum_first_20_valid_even_numbers / 20

theorem average_of_first_20_even_numbers_not_divisible_by_3_or_5 :
  average_first_20_valid_even_numbers = 35 :=
by
  sorry

end average_of_first_20_even_numbers_not_divisible_by_3_or_5_l117_117977


namespace balcony_more_than_orchestra_l117_117973

-- Conditions
def total_tickets (O B : ℕ) : Prop := O + B = 340
def total_cost (O B : ℕ) : Prop := 12 * O + 8 * B = 3320

-- The statement we need to prove based on the conditions
theorem balcony_more_than_orchestra (O B : ℕ) (h1 : total_tickets O B) (h2 : total_cost O B) :
  B - O = 40 :=
sorry

end balcony_more_than_orchestra_l117_117973


namespace sin_C_of_arithmetic_sequence_l117_117280

theorem sin_C_of_arithmetic_sequence 
  (A B C : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = Real.pi) 
  (h3 : Real.cos A = 2 / 3) 
  : Real.sin C = (Real.sqrt 5 + 2 * Real.sqrt 3) / 6 :=
sorry

end sin_C_of_arithmetic_sequence_l117_117280


namespace rounding_strategy_correct_l117_117244

-- Definitions of rounding functions
def round_down (n : ℕ) : ℕ := n - 1  -- Assuming n is a large integer, for simplicity
def round_up (n : ℕ) : ℕ := n + 1

-- Definitions for conditions
def cond1 (p q r : ℕ) : ℕ := round_down p / round_down q + round_down r
def cond2 (p q r : ℕ) : ℕ := round_up p / round_down q + round_down r
def cond3 (p q r : ℕ) : ℕ := round_down p / round_up q + round_down r
def cond4 (p q r : ℕ) : ℕ := round_down p / round_down q + round_up r
def cond5 (p q r : ℕ) : ℕ := round_up p / round_up q + round_down r

-- Theorem stating the correct condition
theorem rounding_strategy_correct (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  cond3 p q r < p / q + r :=
sorry

end rounding_strategy_correct_l117_117244


namespace shaded_area_of_squares_l117_117073

theorem shaded_area_of_squares :
  let s_s := 4
  let s_L := 9
  let area_L := s_L * s_L
  let area_s := s_s * s_s
  area_L - area_s = 65 := sorry

end shaded_area_of_squares_l117_117073


namespace integer_solution_system_l117_117288

theorem integer_solution_system (n : ℕ) (H : n ≥ 2) : 
  ∃ (x : ℕ → ℤ), (
    ∀ i : ℕ, x ((i % n) + 1)^2 + x (((i + 1) % n) + 1)^2 + 50 = 16 * x ((i % n) + 1) + 12 * x (((i + 1) % n) + 1)
  ) ↔ n % 3 = 0 :=
by
  sorry

end integer_solution_system_l117_117288


namespace geometric_mean_l117_117259

theorem geometric_mean (a b c : ℝ) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : b^2 = a * c) : b = 1 :=
sorry

end geometric_mean_l117_117259


namespace area_ratio_of_squares_l117_117521

theorem area_ratio_of_squares (R x y : ℝ) (hx : x^2 = (4/5) * R^2) (hy : y = R * Real.sqrt 2) :
  x^2 / y^2 = 2 / 5 :=
by sorry

end area_ratio_of_squares_l117_117521


namespace order_of_numbers_l117_117701

theorem order_of_numbers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by
  sorry

end order_of_numbers_l117_117701


namespace solutions_eq1_solutions_eq2_l117_117755

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l117_117755


namespace joe_used_fraction_paint_in_first_week_l117_117316

variable (x : ℝ) -- Define the fraction x as a real number

-- Given conditions
def given_conditions : Prop := 
  let total_paint := 360
  let paint_first_week := x * total_paint
  let remaining_paint := (1 - x) * total_paint
  let paint_second_week := (1 / 2) * remaining_paint
  paint_first_week + paint_second_week = 225

-- The theorem to prove
theorem joe_used_fraction_paint_in_first_week (h : given_conditions x) : x = 1 / 4 :=
sorry

end joe_used_fraction_paint_in_first_week_l117_117316


namespace solutions_shifted_quadratic_l117_117142

theorem solutions_shifted_quadratic (a h k : ℝ) (x1 x2: ℝ)
  (h1 : a * (-1 - h)^2 + k = 0)
  (h2 : a * (3 - h)^2 + k = 0) :
  a * (0 - (h + 1))^2 + k = 0 ∧ a * (4 - (h + 1))^2 + k = 0 :=
by
  sorry

end solutions_shifted_quadratic_l117_117142


namespace min_value_arithmetic_sequence_l117_117574

theorem min_value_arithmetic_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 1 + a 2014 = 2) :
  (∃ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 ∧ a2 > 0 ∧ a2013 > 0 ∧ ∀ a2 a2013, a 2 = a2 ∧ a 2013 = a2013 → (1/a2 + 1/a2013) ≥ 2) :=
by
  sorry

end min_value_arithmetic_sequence_l117_117574


namespace john_memory_card_cost_l117_117353

-- Define conditions
def pictures_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 3
def pictures_per_card : ℕ := 50
def cost_per_card : ℕ := 60

-- Define total days
def total_days (years : ℕ) (days_per_year : ℕ) : ℕ := years * days_per_year

-- Define total pictures
def total_pictures (pictures_per_day : ℕ) (total_days : ℕ) : ℕ := pictures_per_day * total_days

-- Define required cards
def required_cards (total_pictures : ℕ) (pictures_per_card : ℕ) : ℕ :=
  (total_pictures + pictures_per_card - 1) / pictures_per_card  -- ceiling division

-- Define total cost
def total_cost (required_cards : ℕ) (cost_per_card : ℕ) : ℕ := required_cards * cost_per_card

-- Prove the total cost equals $13,140
theorem john_memory_card_cost : total_cost (required_cards (total_pictures pictures_per_day (total_days years days_per_year)) pictures_per_card) cost_per_card = 13140 :=
by
  sorry

end john_memory_card_cost_l117_117353


namespace scientific_notation_104000000_l117_117072

theorem scientific_notation_104000000 :
  104000000 = 1.04 * 10^8 :=
sorry

end scientific_notation_104000000_l117_117072


namespace day_of_week_after_6_pow_2023_l117_117475

def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem day_of_week_after_6_pow_2023 :
  day_of_week_after_days 4 (6^2023) = 3 :=
by
  sorry

end day_of_week_after_6_pow_2023_l117_117475


namespace second_quadrant_distance_l117_117557

theorem second_quadrant_distance 
    (m : ℝ) 
    (P : ℝ × ℝ)
    (hP1 : P = (m - 3, m + 2))
    (hP2 : (m + 2) > 0)
    (hP3 : (m - 3) < 0)
    (hDist : |(m + 2)| = 4) : P = (-1, 4) := 
by
  have h1 : m + 2 = 4 := sorry
  have h2 : m = 2 := sorry
  have h3 : P = (2 - 3, 2 + 2) := sorry
  have h4 : P = (-1, 4) := sorry
  exact h4

end second_quadrant_distance_l117_117557


namespace moe_mowing_time_l117_117013

noncomputable def effective_swath_width_inches : ℝ := 30 - 6
noncomputable def effective_swath_width_feet : ℝ := (effective_swath_width_inches / 12)
noncomputable def lawn_width : ℝ := 180
noncomputable def lawn_length : ℝ := 120
noncomputable def walking_rate : ℝ := 4500
noncomputable def total_strips : ℝ := lawn_width / effective_swath_width_feet
noncomputable def total_distance : ℝ := total_strips * lawn_length
noncomputable def time_required : ℝ := total_distance / walking_rate

theorem moe_mowing_time :
  time_required = 2.4 := by
  sorry

end moe_mowing_time_l117_117013


namespace Nancy_more_pearl_beads_l117_117445

-- Define the problem conditions
def metal_beads_Nancy : ℕ := 40
def crystal_beads_Rose : ℕ := 20
def stone_beads_Rose : ℕ := crystal_beads_Rose * 2
def total_beads_needed : ℕ := 20 * 8
def total_Rose_beads : ℕ := crystal_beads_Rose + stone_beads_Rose
def pearl_beads_Nancy : ℕ := total_beads_needed - total_Rose_beads

-- State the theorem to prove
theorem Nancy_more_pearl_beads :
  pearl_beads_Nancy = metal_beads_Nancy + 60 :=
by
  -- We leave the proof as an exercise
  sorry

end Nancy_more_pearl_beads_l117_117445


namespace no_representation_of_form_eight_k_plus_3_or_5_l117_117849

theorem no_representation_of_form_eight_k_plus_3_or_5 (k : ℤ) :
  ∀ x y : ℤ, (8 * k + 3 ≠ x^2 - 2 * y^2) ∧ (8 * k + 5 ≠ x^2 - 2 * y^2) :=
by sorry

end no_representation_of_form_eight_k_plus_3_or_5_l117_117849


namespace pencils_in_drawer_l117_117062

theorem pencils_in_drawer (P : ℕ) 
  (h1 : 19 + 16 = 35)
  (h2 : P + 35 = 78) : 
  P = 43 := 
by
  sorry

end pencils_in_drawer_l117_117062
