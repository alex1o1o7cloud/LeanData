import Mathlib

namespace number_of_men_l199_199052

variable (M : ℕ)

-- Define the first condition: M men reaping 80 hectares in 24 days.
def first_work_rate (M : ℕ) : ℚ := (80 : ℚ) / (M * 24)

-- Define the second condition: 36 men reaping 360 hectares in 30 days.
def second_work_rate : ℚ := (360 : ℚ) / (36 * 30)

-- Lean 4 statement: Prove the equivalence given conditions.
theorem number_of_men (h : first_work_rate M = second_work_rate) : M = 45 :=
by
  sorry

end number_of_men_l199_199052


namespace same_terminal_side_l199_199790

theorem same_terminal_side (θ : ℝ) : (∃ k : ℤ, θ = 2 * k * π - π / 6) → θ = 11 * π / 6 :=
sorry

end same_terminal_side_l199_199790


namespace find_k_l199_199944

noncomputable def vec_na (x1 k : ℝ) : ℝ × ℝ := (x1 - k/4, 2 * x1^2)
noncomputable def vec_nb (x2 k : ℝ) : ℝ × ℝ := (x2 - k/4, 2 * x2^2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem find_k (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = k / 2) 
  (h2 : x1 * x2 = -1) 
  (h3 : dot_product (vec_na x1 k) (vec_nb x2 k) = 0) : 
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
by
  sorry

end find_k_l199_199944


namespace div_relation_l199_199752

theorem div_relation (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 3) : c / a = 1 / 2 := 
by 
  sorry

end div_relation_l199_199752


namespace wire_length_after_cuts_l199_199270

-- Given conditions as parameters
def initial_length_cm : ℝ := 23.3
def first_cut_mm : ℝ := 105
def second_cut_cm : ℝ := 4.6

-- Final statement to be proved
theorem wire_length_after_cuts (ell : ℝ) (c1 : ℝ) (c2 : ℝ) : (ell = 23.3) → (c1 = 105) → (c2 = 4.6) → 
  (ell * 10 - c1 - c2 * 10 = 82) := sorry

end wire_length_after_cuts_l199_199270


namespace parabola_distance_l199_199367

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l199_199367


namespace Miss_Stevie_payment_l199_199533

theorem Miss_Stevie_payment:
  let painting_hours := 8
  let painting_rate := 15
  let painting_earnings := painting_hours * painting_rate
  let mowing_hours := 6
  let mowing_rate := 10
  let mowing_earnings := mowing_hours * mowing_rate
  let plumbing_hours := 4
  let plumbing_rate := 18
  let plumbing_earnings := plumbing_hours * plumbing_rate
  let total_earnings := painting_earnings + mowing_earnings + plumbing_earnings
  let discount := 0.10 * total_earnings
  let amount_paid := total_earnings - discount
  amount_paid = 226.80 :=
by
  sorry

end Miss_Stevie_payment_l199_199533


namespace notebooks_left_l199_199041

theorem notebooks_left (bundles : ℕ) (notebooks_per_bundle : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  bundles = 5 ∧ notebooks_per_bundle = 25 ∧ groups = 8 ∧ students_per_group = 13 →
  bundles * notebooks_per_bundle - groups * students_per_group = 21 := 
by sorry

end notebooks_left_l199_199041


namespace number_added_l199_199105

def initial_number : ℕ := 9
def final_resultant : ℕ := 93

theorem number_added : ∃ x : ℕ, 3 * (2 * initial_number + x) = final_resultant ∧ x = 13 := by
  sorry

end number_added_l199_199105


namespace women_in_business_class_l199_199636

theorem women_in_business_class 
  (total_passengers : ℕ) 
  (percent_women : ℝ) 
  (percent_women_in_business : ℝ) 
  (H1 : total_passengers = 300)
  (H2 : percent_women = 0.70)
  (H3 : percent_women_in_business = 0.08) : 
  ∃ (num_women_business_class : ℕ), num_women_business_class = 16 := 
by
  sorry

end women_in_business_class_l199_199636


namespace smallest_n_conditions_l199_199331

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end smallest_n_conditions_l199_199331


namespace triangle_isosceles_of_sin_condition_l199_199569

noncomputable def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

theorem triangle_isosceles_of_sin_condition {A B C : ℝ} (h : 2 * Real.sin A * Real.cos B = Real.sin C) : 
  isosceles_triangle A B C :=
by
  sorry

end triangle_isosceles_of_sin_condition_l199_199569


namespace boys_ages_l199_199384

theorem boys_ages (a b : ℕ) (h1 : a = b) (h2 : a + b + 11 = 29) : a = 9 :=
by
  sorry

end boys_ages_l199_199384


namespace max_matching_pairs_l199_199634

theorem max_matching_pairs (total_pairs : ℕ) (lost_individual : ℕ) (left_pair : ℕ) : 
  total_pairs = 25 ∧ lost_individual = 9 → left_pair = 20 :=
by
  sorry

end max_matching_pairs_l199_199634


namespace conversion_points_worth_two_l199_199442

theorem conversion_points_worth_two
  (touchdowns_per_game : ℕ := 4)
  (points_per_touchdown : ℕ := 6)
  (games_in_season : ℕ := 15)
  (total_touchdowns_scored : ℕ := touchdowns_per_game * games_in_season)
  (total_points_from_touchdowns : ℕ := total_touchdowns_scored * points_per_touchdown)
  (old_record_points : ℕ := 300)
  (points_above_record : ℕ := 72)
  (total_points_scored : ℕ := old_record_points + points_above_record)
  (conversions_scored : ℕ := 6)
  (total_points_from_conversions : ℕ := total_points_scored - total_points_from_touchdowns) :
  total_points_from_conversions / conversions_scored = 2 := by
sorry

end conversion_points_worth_two_l199_199442


namespace quadratic_inequality_l199_199054

theorem quadratic_inequality (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) : ∀ x : ℝ, c * x^2 - b * x + a > c * x - b := 
by
  sorry

end quadratic_inequality_l199_199054


namespace marcy_needs_6_tubs_of_lip_gloss_l199_199717

theorem marcy_needs_6_tubs_of_lip_gloss (people tubes_per_person tubes_per_tub : ℕ) 
  (h1 : people = 36) (h2 : tubes_per_person = 3) (h3 : tubes_per_tub = 2) :
  (people / tubes_per_person) / tubes_per_tub = 6 :=
by
  -- The proof goes here
  sorry

end marcy_needs_6_tubs_of_lip_gloss_l199_199717


namespace ratio_ravi_kiran_l199_199109

-- Definitions for the conditions
def ratio_money_ravi_giri := 6 / 7
def money_ravi := 36
def money_kiran := 105

-- The proof problem
theorem ratio_ravi_kiran : (money_ravi : ℕ) / money_kiran = 12 / 35 := 
by 
  sorry

end ratio_ravi_kiran_l199_199109


namespace correctly_calculated_expression_l199_199665

theorem correctly_calculated_expression (x : ℝ) :
  ¬ (x^3 + x^2 = x^5) ∧ 
  ¬ (x^3 * x^2 = x^6) ∧ 
  (x^3 / x^2 = x) ∧ 
  ¬ ((x^3)^2 = x^9) := by
sorry

end correctly_calculated_expression_l199_199665


namespace solve_for_square_l199_199440

theorem solve_for_square (x : ℤ) (s : ℤ) 
  (h1 : s + x = 80) 
  (h2 : 3 * (s + x) - 2 * x = 164) : 
  s = 42 :=
by 
  -- Include the implementation with sorry
  sorry

end solve_for_square_l199_199440


namespace length_of_equal_sides_l199_199369

-- Definitions based on conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ a = c)

def is_triangle (a b c : ℝ) : Prop :=
(a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def has_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
a + b + c = P

def one_side_length (a : ℝ) : Prop :=
a = 3

-- The proof statement
theorem length_of_equal_sides (a b c : ℝ) :
isosceles_triangle a b c →
is_triangle a b c →
has_perimeter a b c 7 →
one_side_length a ∨ one_side_length b ∨ one_side_length c →
(b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) :=
by
  intros iso tri per side_length
  sorry

end length_of_equal_sides_l199_199369


namespace number_exceeds_part_l199_199647

theorem number_exceeds_part (x : ℝ) (h : x = (5 / 9) * x + 150) : x = 337.5 := sorry

end number_exceeds_part_l199_199647


namespace green_notebook_cost_l199_199830

def total_cost : ℕ := 45
def black_cost : ℕ := 15
def pink_cost : ℕ := 10
def num_green_notebooks : ℕ := 2

theorem green_notebook_cost :
  (total_cost - (black_cost + pink_cost)) / num_green_notebooks = 10 :=
by
  sorry

end green_notebook_cost_l199_199830


namespace initial_men_count_l199_199627

theorem initial_men_count (M : ℕ) :
  let total_food := M * 22
  let food_after_2_days := total_food - 2 * M
  let remaining_food := 20 * M
  let new_total_men := M + 190
  let required_food_for_16_days := new_total_men * 16
  (remaining_food = required_food_for_16_days) → M = 760 :=
by
  intro h
  sorry

end initial_men_count_l199_199627


namespace ratio_of_areas_two_adjacent_triangles_to_one_triangle_l199_199604

-- Definition of a regular hexagon divided into six equal triangles
def is_regular_hexagon_divided_into_six_equal_triangles (s : ℝ) : Prop :=
  s > 0 -- s is the area of one of the six triangles and must be positive

-- Definition of the area of a region formed by two adjacent triangles
def area_of_two_adjacent_triangles (s r : ℝ) : Prop :=
  r = 2 * s

-- The proof problem statement
theorem ratio_of_areas_two_adjacent_triangles_to_one_triangle (s r : ℝ)
  (hs : is_regular_hexagon_divided_into_six_equal_triangles s)
  (hr : area_of_two_adjacent_triangles s r) : 
  r / s = 2 :=
by
  sorry

end ratio_of_areas_two_adjacent_triangles_to_one_triangle_l199_199604


namespace tan_y_eq_tan_x_plus_one_over_cos_x_l199_199862

theorem tan_y_eq_tan_x_plus_one_over_cos_x 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hy : y < π / 2) 
  (h_tan : Real.tan y = Real.tan x + (1 / Real.cos x)) 
  : y - (x / 2) = π / 6 :=
sorry

end tan_y_eq_tan_x_plus_one_over_cos_x_l199_199862


namespace sum_radical_conjugate_l199_199211

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l199_199211


namespace cake_fraction_eaten_l199_199454

theorem cake_fraction_eaten (total_slices kept_slices slices_eaten : ℕ) 
  (h1 : total_slices = 12)
  (h2 : kept_slices = 9)
  (h3 : slices_eaten = total_slices - kept_slices) :
  (slices_eaten : ℚ) / total_slices = 1 / 4 := 
sorry

end cake_fraction_eaten_l199_199454


namespace radius_is_100_div_pi_l199_199519

noncomputable def radius_of_circle (L : ℝ) (θ : ℝ) : ℝ :=
  L * 360 / (θ * 2 * Real.pi)

theorem radius_is_100_div_pi :
  radius_of_circle 25 45 = 100 / Real.pi := 
by
  sorry

end radius_is_100_div_pi_l199_199519


namespace telephone_number_problem_l199_199138

theorem telephone_number_problem :
  ∃ A B C D E F G H I J : ℕ,
    (A > B) ∧ (B > C) ∧ (D > E) ∧ (E > F) ∧ (G > H) ∧ (H > I) ∧ (I > J) ∧
    (D = E + 1) ∧ (E = F + 1) ∧ (D % 2 = 0) ∧ 
    (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2) ∧ (G % 2 = 1) ∧ (H % 2 = 1) ∧ (I % 2 = 1) ∧ (J % 2 = 1) ∧
    (A + B + C = 7) ∧ (B + C + F = 10) ∧ (A = 7) :=
sorry

end telephone_number_problem_l199_199138


namespace triangle_problem_proof_l199_199072

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
variables (h2 : c = Real.sqrt 7)
variables (area : ℝ := 3 * Real.sqrt 3 / 2)

-- Prove angle C = π / 3 and perimeter of triangle
theorem triangle_problem_proof 
(h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
(h2 : c = Real.sqrt 7)
(area_condition : (1 / 2) * a * b * (Real.sin C) = area) :
  (C = Real.pi / 3) ∧ (a + b + c = 5 + Real.sqrt 7) := 
by
  sorry

end triangle_problem_proof_l199_199072


namespace hexagon_longest_side_l199_199864

theorem hexagon_longest_side (x : ℝ) (h₁ : 6 * x = 20) (h₂ : x < 20 - x) : (10 / 3) ≤ x ∧ x < 10 :=
sorry

end hexagon_longest_side_l199_199864


namespace graph_is_pair_of_straight_lines_l199_199019

theorem graph_is_pair_of_straight_lines : ∀ (x y : ℝ), 9 * x^2 - y^2 - 6 * x = 0 → ∃ a b c : ℝ, (y = 3 * x - 2 ∨ y = 2 - 3 * x) :=
by
  intro x y h
  sorry

end graph_is_pair_of_straight_lines_l199_199019


namespace tunnel_length_l199_199485

def train_length : ℝ := 1.5
def exit_time_minutes : ℝ := 4
def speed_mph : ℝ := 45

theorem tunnel_length (d_train : ℝ := train_length)
                      (t_exit : ℝ := exit_time_minutes)
                      (v_mph : ℝ := speed_mph) :
  d_train + ((v_mph / 60) * t_exit - d_train) = 1.5 :=
by
  sorry

end tunnel_length_l199_199485


namespace first_sculpture_weight_is_five_l199_199153

variable (w x y z : ℝ)

def hourly_wage_exterminator := 70
def daily_hours := 20
def price_per_pound := 20
def second_sculpture_weight := 7
def total_income := 1640

def income_exterminator := daily_hours * hourly_wage_exterminator
def income_sculptures := total_income - income_exterminator
def income_second_sculpture := second_sculpture_weight * price_per_pound
def income_first_sculpture := income_sculptures - income_second_sculpture

def weight_first_sculpture := income_first_sculpture / price_per_pound

theorem first_sculpture_weight_is_five :
  weight_first_sculpture = 5 := sorry

end first_sculpture_weight_is_five_l199_199153


namespace binary_representation_of_fourteen_l199_199959

theorem binary_representation_of_fourteen :
  (14 : ℕ) = 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 0 * 2^0 :=
by
  sorry

end binary_representation_of_fourteen_l199_199959


namespace fraction_doubled_l199_199840

variable (x y : ℝ)

theorem fraction_doubled (x y : ℝ) : 
  (x + y) ≠ 0 → (2 * x * 2 * y) / (2 * x + 2 * y) = 2 * (x * y / (x + y)) := 
by
  intro h
  sorry

end fraction_doubled_l199_199840


namespace find_weight_difference_l199_199448

variables (W_A W_B W_C W_D W_E : ℝ)

-- Definitions of the conditions
def average_weight_abc := (W_A + W_B + W_C) / 3 = 84
def average_weight_abcd := (W_A + W_B + W_C + W_D) / 4 = 80
def average_weight_bcde := (W_B + W_C + W_D + W_E) / 4 = 79
def weight_a := W_A = 77

-- The theorem statement
theorem find_weight_difference (h1 : average_weight_abc W_A W_B W_C)
                               (h2 : average_weight_abcd W_A W_B W_C W_D)
                               (h3 : average_weight_bcde W_B W_C W_D W_E)
                               (h4 : weight_a W_A) :
  W_E - W_D = 5 :=
sorry

end find_weight_difference_l199_199448


namespace dingding_minimum_correct_answers_l199_199144

theorem dingding_minimum_correct_answers (x : ℕ) :
  (5 * x - (30 - x) > 100) → x ≥ 22 :=
by
  sorry

end dingding_minimum_correct_answers_l199_199144


namespace shoe_price_monday_final_price_l199_199847

theorem shoe_price_monday_final_price : 
  let thursday_price := 50
  let friday_markup_rate := 0.15
  let monday_discount_rate := 0.12
  let friday_price := thursday_price * (1 + friday_markup_rate)
  let monday_price := friday_price * (1 - monday_discount_rate)
  monday_price = 50.6 := by
  sorry

end shoe_price_monday_final_price_l199_199847


namespace total_people_in_office_even_l199_199618

theorem total_people_in_office_even (M W : ℕ) (h_even : M = W) (h_meeting_women : 6 = 20 / 100 * W) : 
  M + W = 60 :=
by
  sorry

end total_people_in_office_even_l199_199618


namespace polygon_sides_l199_199672

-- Define the conditions
def sum_interior_angles (x : ℕ) : ℝ := 180 * (x - 2)
def sum_given_angles (x : ℕ) : ℝ := 160 + 112 * (x - 1)

-- State the theorem
theorem polygon_sides (x : ℕ) (h : sum_interior_angles x = sum_given_angles x) : x = 6 := by
  sorry

end polygon_sides_l199_199672


namespace custom_op_subtraction_l199_199508

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_subtraction :
  (custom_op 4 2) - (custom_op 2 4) = -8 := by
  sorry

end custom_op_subtraction_l199_199508


namespace number_of_female_students_l199_199129

theorem number_of_female_students (T S f_sample : ℕ) (H_total : T = 1600) (H_sample_size : S = 200) (H_females_in_sample : f_sample = 95) : 
  ∃ F, 95 / 200 = F / 1600 ∧ F = 760 := by 
sorry

end number_of_female_students_l199_199129


namespace combination_add_l199_199767

def combination (n m : ℕ) : ℕ := n.choose m

theorem combination_add {n : ℕ} (h1 : 4 ≤ 9) (h2 : 5 ≤ 9) :
  combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end combination_add_l199_199767


namespace billy_apples_l199_199224

def num_apples_eaten (monday_apples tuesday_apples wednesday_apples thursday_apples friday_apples total_apples : ℕ) : Prop :=
  monday_apples = 2 ∧
  tuesday_apples = 2 * monday_apples ∧
  wednesday_apples = 9 ∧
  friday_apples = monday_apples / 2 ∧
  thursday_apples = 4 * friday_apples ∧
  total_apples = monday_apples + tuesday_apples + wednesday_apples + thursday_apples + friday_apples

theorem billy_apples : num_apples_eaten 2 4 9 4 1 20 := 
by
  unfold num_apples_eaten
  sorry

end billy_apples_l199_199224


namespace monotonic_decreasing_interval_range_of_a_l199_199693

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) * ((a / x) + a + 1)

theorem monotonic_decreasing_interval (a : ℝ) (h : a ≥ -1) :
  (a = -1 → ∀ x, x < -1 → f a x < f a (x + 1)) ∧
  (a ≠ -1 → (∀ x, -1 < a ∧ x < -1 ∨ x > 1 / (a + 1) → f a x < f a (x + 1)) ∧
                (∀ x, -1 < a ∧ -1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 / (a + 1) → f a x < f a (x + 1)))
:= sorry

theorem range_of_a (a : ℝ) (h : a ≥ -1) :
  (∃ x1 x2, x1 > 0 ∧ x2 < 0 ∧ f a x1 < f a x2 → -1 ≤ a ∧ a < 0)
:= sorry

end monotonic_decreasing_interval_range_of_a_l199_199693


namespace rectangle_area_is_200000_l199_199030

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isRectangle (P Q R S : Point) : Prop :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y) = 
  (R.x - S.x) * (R.x - S.x) + (R.y - S.y) * (R.y - S.y) ∧
  (P.x - S.x) * (P.x - S.x) + (P.y - S.y) * (P.y - S.y) = 
  (Q.x - R.x) * (Q.x - R.x) + (Q.y - R.y) * (Q.y - R.y) ∧
  (P.x - Q.x) * (P.x - S.x) + (P.y - Q.y) * (P.y - S.y) = 0

theorem rectangle_area_is_200000:
  ∀ (P Q R S : Point),
  P = ⟨-15, 30⟩ →
  Q = ⟨985, 230⟩ →
  R.x = 985 → 
  S.x = -13 →
  R.y = S.y → 
  isRectangle P Q R S →
  ( ( (Q.x - P.x)^2 + (Q.y - P.y)^2 ).sqrt *
    ( (S.x - P.x)^2 + (S.y - P.y)^2 ).sqrt ) = 200000 :=
by
  intros P Q R S hP hQ hxR hxS hyR hRect
  sorry

end rectangle_area_is_200000_l199_199030


namespace distinct_divisors_sum_factorial_l199_199029

theorem distinct_divisors_sum_factorial (n : ℕ) (h : n ≥ 3) :
  ∃ (d : Fin n → ℕ), (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n!) ∧ (n! = (Finset.univ.sum d)) :=
sorry

end distinct_divisors_sum_factorial_l199_199029


namespace lucas_income_36000_l199_199504

variable (q I : ℝ)

-- Conditions as Lean 4 definitions
def tax_below_30000 : ℝ := 0.01 * q * 30000
def tax_above_30000 (I : ℝ) : ℝ := 0.01 * (q + 3) * (I - 30000)
def total_tax (I : ℝ) : ℝ := tax_below_30000 q + tax_above_30000 q I
def total_tax_condition (I : ℝ) : Prop := total_tax q I = 0.01 * (q + 0.5) * I

theorem lucas_income_36000 (h : total_tax_condition q I) : I = 36000 := by
  sorry

end lucas_income_36000_l199_199504


namespace a_eq_zero_iff_purely_imaginary_l199_199740

open Complex

noncomputable def purely_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem a_eq_zero_iff_purely_imaginary (a b : ℝ) :
  (a = 0) ↔ purely_imaginary (a + b * Complex.I) :=
by
  sorry

end a_eq_zero_iff_purely_imaginary_l199_199740


namespace unique_sum_of_two_primes_l199_199539

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l199_199539


namespace sum_first_13_terms_l199_199312

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (ha : a 2 + a 5 + a 9 + a 12 = 60)

theorem sum_first_13_terms :
  S 13 = 195 := sorry

end sum_first_13_terms_l199_199312


namespace find_value_of_B_l199_199329

theorem find_value_of_B (B : ℚ) (h : 4 * B + 4 = 33) : B = 29 / 4 :=
by
  sorry

end find_value_of_B_l199_199329


namespace tan_phi_eq_sqrt3_l199_199283

theorem tan_phi_eq_sqrt3
  (φ : ℝ)
  (h1 : Real.cos (Real.pi / 2 - φ) = Real.sqrt 3 / 2)
  (h2 : abs φ < Real.pi / 2) :
  Real.tan φ = Real.sqrt 3 :=
sorry

end tan_phi_eq_sqrt3_l199_199283


namespace greatest_possible_radius_of_circle_l199_199139

theorem greatest_possible_radius_of_circle
  (π : Real)
  (r : Real)
  (h : π * r^2 < 100 * π) :
  ∃ (n : ℕ), n = 9 ∧ (r : ℝ) ≤ 10 ∧ (r : ℝ) ≥ 9 :=
by
  sorry

end greatest_possible_radius_of_circle_l199_199139


namespace combined_capacity_l199_199765

theorem combined_capacity (A B : ℝ) : 3 * A + B = A + 2 * A + B :=
by
  sorry

end combined_capacity_l199_199765


namespace systematic_sampling_interval_l199_199656

-- Definitions for the given conditions
def total_students : ℕ := 1203
def sample_size : ℕ := 40

-- Theorem statement to be proven
theorem systematic_sampling_interval (N n : ℕ) (hN : N = total_students) (hn : n = sample_size) : 
  N % n ≠ 0 → ∃ k : ℕ, k = 30 :=
by
  sorry

end systematic_sampling_interval_l199_199656


namespace sum_of_remainders_and_parity_l199_199006

theorem sum_of_remainders_and_parity 
  (n : ℤ) 
  (h₀ : n % 20 = 13) : 
  (n % 4 + n % 5 = 4) ∧ (n % 2 = 1) :=
by
  sorry

end sum_of_remainders_and_parity_l199_199006


namespace intersection_with_y_axis_l199_199822

theorem intersection_with_y_axis :
  ∀ (y : ℝ), (∃ x : ℝ, y = 2 * x + 2 ∧ x = 0) → y = 2 :=
by
  sorry

end intersection_with_y_axis_l199_199822


namespace ellen_smoothie_ingredients_l199_199640

theorem ellen_smoothie_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by
  sorry

end ellen_smoothie_ingredients_l199_199640


namespace unique_solution_of_function_eq_l199_199295

theorem unique_solution_of_function_eq (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) : f = id := 
sorry

end unique_solution_of_function_eq_l199_199295


namespace transformed_parabola_eq_l199_199657

noncomputable def initial_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3
def shift_left (h : ℝ) (c : ℝ): ℝ := h - c
def shift_down (k : ℝ) (d : ℝ): ℝ := k - d

theorem transformed_parabola_eq :
  ∃ (x : ℝ), (initial_parabola (shift_left x 2) - 1 = 2 * (x + 1)^2 + 2) :=
sorry

end transformed_parabola_eq_l199_199657


namespace probability_of_green_ball_l199_199794

def container_X := (5, 7)  -- (red balls, green balls)
def container_Y := (7, 5)  -- (red balls, green balls)
def container_Z := (7, 5)  -- (red balls, green balls)

def total_balls (container : ℕ × ℕ) : ℕ := container.1 + container.2

def probability_green (container : ℕ × ℕ) : ℚ := 
  (container.2 : ℚ) / total_balls container

noncomputable def probability_green_from_random_selection : ℚ :=
  (1 / 3) * probability_green container_X +
  (1 / 3) * probability_green container_Y +
  (1 / 3) * probability_green container_Z

theorem probability_of_green_ball :
  probability_green_from_random_selection = 17 / 36 :=
sorry

end probability_of_green_ball_l199_199794


namespace tsunami_added_sand_l199_199215

noncomputable def dig_rate : ℝ := 8 / 4 -- feet per hour
noncomputable def sand_after_storm : ℝ := 8 / 2 -- feet
noncomputable def time_to_dig_up_treasure : ℝ := 3 -- hours
noncomputable def total_sand_dug_up : ℝ := dig_rate * time_to_dig_up_treasure -- feet

theorem tsunami_added_sand :
  total_sand_dug_up - sand_after_storm = 2 :=
by
  sorry

end tsunami_added_sand_l199_199215


namespace simplify_and_rationalize_denominator_l199_199149

theorem simplify_and_rationalize_denominator :
  ( (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) = 3 * Real.sqrt 420 / 42 ) := 
by {
  sorry
}

end simplify_and_rationalize_denominator_l199_199149


namespace determine_transportation_mode_l199_199174

def distance : ℝ := 60 -- in kilometers
def time : ℝ := 3 -- in hours
def speed_of_walking : ℝ := 5 -- typical speed in km/h
def speed_of_bicycle_riding : ℝ := 15 -- lower bound of bicycle speed in km/h
def speed_of_driving_a_car : ℝ := 20 -- typical minimum speed in km/h

theorem determine_transportation_mode : (distance / time) = speed_of_driving_a_car ∧ speed_of_driving_a_car ≥ speed_of_walking + speed_of_bicycle_riding - speed_of_driving_a_car := sorry

end determine_transportation_mode_l199_199174


namespace distinct_real_roots_range_l199_199932

theorem distinct_real_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ax^2 + 2 * x + 1 = 0 ∧ ay^2 + 2 * y + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
by
  sorry

end distinct_real_roots_range_l199_199932


namespace problem1_problem2_l199_199412

theorem problem1 (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m - |x - 2|) 
  (h2 : ∀ x, f (x + 2) ≥ 0 → -1 ≤ x ∧ x ≤ 1) : 
  m = 1 := 
sorry

theorem problem2 (a b c : ℝ) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 := 
sorry

end problem1_problem2_l199_199412


namespace average_age_of_combined_rooms_l199_199258

theorem average_age_of_combined_rooms
  (num_people_A : ℕ) (avg_age_A : ℕ)
  (num_people_B : ℕ) (avg_age_B : ℕ)
  (num_people_C : ℕ) (avg_age_C : ℕ)
  (hA : num_people_A = 8) (hAA : avg_age_A = 35)
  (hB : num_people_B = 5) (hBB : avg_age_B = 30)
  (hC : num_people_C = 7) (hCC : avg_age_C = 50) :
  ((num_people_A * avg_age_A + num_people_B * avg_age_B + num_people_C * avg_age_C) / 
  (num_people_A + num_people_B + num_people_C) = 39) :=
by
  sorry

end average_age_of_combined_rooms_l199_199258


namespace lines_are_parallel_l199_199679

theorem lines_are_parallel : 
  ∀ (x y : ℝ), (2 * x - y = 7) → (2 * x - y - 1 = 0) → False :=
by
  sorry  -- Proof will be filled in later

end lines_are_parallel_l199_199679


namespace validCardSelections_l199_199316

def numberOfValidSelections : ℕ :=
  let totalCards := 12
  let redCards := 4
  let otherColors := 8 -- 4 yellow + 4 blue
  let totalSelections := Nat.choose totalCards 3
  let nonRedSelections := Nat.choose otherColors 3
  let oneRedSelections := Nat.choose redCards 1 * Nat.choose otherColors 2
  let sameColorSelections := 3 * Nat.choose 4 3 -- 3 colors, 4 cards each, selecting 3
  (nonRedSelections + oneRedSelections)

theorem validCardSelections : numberOfValidSelections = 160 := by
  sorry

end validCardSelections_l199_199316


namespace min_value_of_f_l199_199817

-- Define the problem domain: positive real numbers
variables (a b c x y z : ℝ)
variables (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
variables (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)

-- Define the given equations
variables (h1 : c * y + b * z = a)
variables (h2 : a * z + c * x = b)
variables (h3 : b * x + a * y = c)

-- Define the function f(x, y, z)
noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

-- The theorem statement: under the given conditions the minimum value of f(x, y, z) is 1/2
theorem min_value_of_f :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    c * y + b * z = a →
    a * z + c * x = b →
    b * x + a * y = c →
    f x y z = 1 / 2) :=
sorry

end min_value_of_f_l199_199817


namespace additional_telephone_lines_l199_199713

def telephone_lines_increase : ℕ :=
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  lines_seven_digits - lines_six_digits

theorem additional_telephone_lines : telephone_lines_increase = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l199_199713


namespace k_starts_at_10_l199_199502

variable (V_k V_l : ℝ)
variable (t_k t_l : ℝ)

-- Conditions
axiom k_faster_than_l : V_k = 1.5 * V_l
axiom l_speed : V_l = 50
axiom l_start_time : t_l = 9
axiom meet_time : t_k + 3 = 12
axiom distance_apart : V_l * 3 + V_k * (12 - t_k) = 300

-- Proof goal
theorem k_starts_at_10 : t_k = 10 :=
by
  sorry

end k_starts_at_10_l199_199502


namespace point_M_coordinates_l199_199445

theorem point_M_coordinates :
  (∃ (M : ℝ × ℝ), M.1 < 0 ∧ M.2 > 0 ∧ abs M.2 = 2 ∧ abs M.1 = 1 ∧ M = (-1, 2)) :=
by
  use (-1, 2)
  sorry

end point_M_coordinates_l199_199445


namespace series_fraction_simplify_l199_199086

theorem series_fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by 
  sorry

end series_fraction_simplify_l199_199086


namespace square_side_length_equals_4_l199_199359

theorem square_side_length_equals_4 (s : ℝ) (h : s^2 = 4 * s) : s = 4 :=
sorry

end square_side_length_equals_4_l199_199359


namespace UBA_Capital_bought_8_SUVs_l199_199506

noncomputable def UBA_Capital_SUVs : ℕ := 
  let T := 9  -- Number of Toyotas
  let H := 1  -- Number of Hondas
  let SUV_Toyota := 9 / 10 * T  -- 90% of Toyotas are SUVs
  let SUV_Honda := 1 / 10 * H   -- 10% of Hondas are SUVs
  SUV_Toyota + SUV_Honda  -- Total number of SUVs

theorem UBA_Capital_bought_8_SUVs : UBA_Capital_SUVs = 8 := by
  sorry

end UBA_Capital_bought_8_SUVs_l199_199506


namespace maximize_container_volume_l199_199321

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ ∀ y : ℝ, 0 < y ∧ y < 24 → 
  ( (48 - 2 * x)^2 * x ≥ (48 - 2 * y)^2 * y ) ∧ x = 8 :=
sorry

end maximize_container_volume_l199_199321


namespace required_jogging_speed_l199_199039

-- Definitions based on the conditions
def blocks_to_miles (blocks : ℕ) : ℚ := blocks * (1 / 8 : ℚ)
def time_in_hours (minutes : ℕ) : ℚ := minutes / 60

-- Constants provided by the problem
def beach_distance_in_blocks : ℕ := 16
def ice_cream_melt_time_in_minutes : ℕ := 10

-- The main statement to prove
theorem required_jogging_speed :
  let distance := blocks_to_miles beach_distance_in_blocks
  let time := time_in_hours ice_cream_melt_time_in_minutes
  (distance / time) = 12 := by
  sorry

end required_jogging_speed_l199_199039


namespace twenty_four_x_eq_a_cubed_t_l199_199776

-- Define conditions
variables {x : ℝ} {a t : ℝ}
axiom h1 : 2^x = a
axiom h2 : 3^x = t

-- State the theorem
theorem twenty_four_x_eq_a_cubed_t : 24^x = a^3 * t := 
by sorry

end twenty_four_x_eq_a_cubed_t_l199_199776


namespace domain_f₁_range_f₂_l199_199137

noncomputable def f₁ (x : ℝ) : ℝ := (x - 2)^0 / Real.sqrt (x + 1)
noncomputable def f₂ (x : ℝ) : ℝ := 2 * x - Real.sqrt (x - 1)

theorem domain_f₁ : ∀ x : ℝ, x > -1 ∧ x ≠ 2 → ∃ y : ℝ, y = f₁ x :=
by
  sorry

theorem range_f₂ : ∀ y : ℝ, y ≥ 15 / 8 → ∃ x : ℝ, y = f₂ x :=
by
  sorry

end domain_f₁_range_f₂_l199_199137


namespace edge_length_approx_17_1_l199_199027

-- Define the base dimensions of the rectangular vessel
def length_base : ℝ := 20
def width_base : ℝ := 15

-- Define the rise in water level
def rise_water_level : ℝ := 16.376666666666665

-- Calculate the area of the base
def area_base : ℝ := length_base * width_base

-- Calculate the volume of the cube (which is equal to the volume of water displaced)
def volume_cube : ℝ := area_base * rise_water_level

-- Calculate the edge length of the cube
def edge_length_cube : ℝ := volume_cube^(1/3)

-- Statement: The edge length of the cube is approximately 17.1 cm
theorem edge_length_approx_17_1 : abs (edge_length_cube - 17.1) < 0.1 :=
by sorry

end edge_length_approx_17_1_l199_199027


namespace larger_number_l199_199353

theorem larger_number (x y: ℝ) 
  (h1: x + y = 40)
  (h2: x - y = 6) :
  x = 23 := 
by
  sorry

end larger_number_l199_199353


namespace solve_quadratic_1_solve_quadratic_2_l199_199946

open Real

theorem solve_quadratic_1 :
  (∃ x : ℝ, x^2 - 2 * x - 7 = 0) ∧
  (∀ x : ℝ, x^2 - 2 * x - 7 = 0 → x = 1 + 2 * sqrt 2 ∨ x = 1 - 2 * sqrt 2) :=
sorry

theorem solve_quadratic_2 :
  (∃ x : ℝ, 3 * (x - 2)^2 = x * (x - 2)) ∧
  (∀ x : ℝ, 3 * (x - 2)^2 = x * (x - 2) → x = 2 ∨ x = 3) :=
sorry

end solve_quadratic_1_solve_quadratic_2_l199_199946


namespace student_number_choice_l199_199417

theorem student_number_choice (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 :=
sorry

end student_number_choice_l199_199417


namespace solve_equation_1_solve_equation_2_l199_199670

theorem solve_equation_1 :
  ∀ x : ℝ, 3 * x - 5 = 6 * x - 8 → x = 1 :=
by
  intro x
  intro h
  sorry

theorem solve_equation_2 :
  ∀ x : ℝ, (x + 1) / 2 - (2 * x - 1) / 3 = 1 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_1_solve_equation_2_l199_199670


namespace smallest_AAAB_value_l199_199813

theorem smallest_AAAB_value : ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 111 * A + B = 7 * (10 * A + B) ∧ 111 * A + B = 667 :=
by sorry

end smallest_AAAB_value_l199_199813


namespace boat_speed_in_still_water_l199_199933

theorem boat_speed_in_still_water (B S : ℕ) (h1 : B + S = 13) (h2 : B - S = 5) : B = 9 :=
by
  sorry

end boat_speed_in_still_water_l199_199933


namespace total_area_of_plots_l199_199449

theorem total_area_of_plots (n : ℕ) (side_length : ℕ) (area_one_plot : ℕ) (total_plots : ℕ) (total_area : ℕ)
  (h1 : n = 9)
  (h2 : side_length = 6)
  (h3 : area_one_plot = side_length * side_length)
  (h4 : total_plots = n)
  (h5 : total_area = area_one_plot * total_plots) :
  total_area = 324 := 
by
  sorry

end total_area_of_plots_l199_199449


namespace minimize_theta_l199_199664

theorem minimize_theta (K : ℤ) : ∃ θ : ℝ, -495 = K * 360 + θ ∧ |θ| ≤ 180 ∧ θ = -135 :=
by
  sorry

end minimize_theta_l199_199664


namespace problem_solution_l199_199078

open Function

-- Definitions of the points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-3, 2⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨4, 1⟩
def D : Point := ⟨-2, 4⟩

-- Definitions of vectors
def vec (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Definitions of conditions
def AB := vec A B
def AD := vec A D
def DC := vec D C

-- Definitions of dot product to check orthogonality
def dot (v w : Point) : ℝ := v.x * w.x + v.y * w.y

-- Lean statement to prove the conditions
theorem problem_solution :
  AB ≠ ⟨-4, 2⟩ ∧
  dot AB AD = 0 ∧
  AB.y * DC.x = AB.x * DC.y ∧
  ((AB.y * DC.x = AB.x * DC.y) ∧ (dot AB AD = 0) → 
  (∃ a b : ℝ, a ≠ b ∧ (a = 0 ∨ b = 0) ∧ AB = ⟨a, -a⟩  ∧ DC = ⟨3 * a, -3 * a⟩)) :=
by
  -- Proof omitted
  sorry

end problem_solution_l199_199078


namespace sarah_toads_l199_199402

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end sarah_toads_l199_199402


namespace minutes_before_noon_l199_199593

theorem minutes_before_noon (x : ℕ) (h1 : x = 40)
  (h2 : ∀ (t : ℕ), t = 180 - (x + 40) ∧ t = 3 * x) : x = 35 :=
by {
  sorry
}

end minutes_before_noon_l199_199593


namespace real_solutions_count_l199_199691

theorem real_solutions_count :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, |x + 1| = |x - 3| + |x - 4| → x = 2 ∨ x = 8 :=
by
  sorry

end real_solutions_count_l199_199691


namespace percentage_of_boys_to_girls_l199_199219

theorem percentage_of_boys_to_girls
  (boys : ℕ) (girls : ℕ)
  (h1 : boys = 20)
  (h2 : girls = 26) :
  (boys / girls : ℝ) * 100 = 76.9 := by
  sorry

end percentage_of_boys_to_girls_l199_199219


namespace total_heartbeats_during_race_l199_199164

-- Definitions for conditions
def heart_rate_per_minute : ℕ := 120
def pace_minutes_per_km : ℕ := 4
def race_distance_km : ℕ := 120

-- Lean statement of the proof problem
theorem total_heartbeats_during_race :
  120 * (4 * 120) = 57600 := by
  sorry

end total_heartbeats_during_race_l199_199164


namespace sum_of_first_five_terms_l199_199730

theorem sum_of_first_five_terms : 
  ∀ (S : ℕ → ℕ) (a : ℕ → ℕ), 
    (a 1 = 1) ∧ 
    (∀ n ≥ 2, S n = S (n - 1) + n + 2) → 
    S 5 = 23 :=
by
  sorry

end sum_of_first_five_terms_l199_199730


namespace angle_c_in_triangle_l199_199399

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A/B = 1/3) (h3 : A/C = 1/5) : C = 100 :=
by
  sorry

end angle_c_in_triangle_l199_199399


namespace ratio_of_dad_to_jayson_l199_199698

-- Define the conditions
def JaysonAge : ℕ := 10
def MomAgeWhenBorn : ℕ := 28
def MomCurrentAge (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ) : ℕ := MomAgeWhenBorn + JaysonAge
def DadCurrentAge (MomCurrentAge : ℕ) : ℕ := MomCurrentAge + 2

-- Define the proof problem
theorem ratio_of_dad_to_jayson (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ)
  (h1 : JaysonAge = 10) (h2 : MomAgeWhenBorn = 28) :
  DadCurrentAge (MomCurrentAge JaysonAge MomAgeWhenBorn) / JaysonAge = 4 :=
by 
  sorry

end ratio_of_dad_to_jayson_l199_199698


namespace smallest_positive_integer_l199_199173

theorem smallest_positive_integer (x : ℕ) : 
  (5 * x ≡ 18 [MOD 33]) ∧ (x ≡ 4 [MOD 7]) → x = 10 := 
by 
  sorry

end smallest_positive_integer_l199_199173


namespace relationship_y1_y2_y3_l199_199319

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l199_199319


namespace cricket_average_l199_199538

theorem cricket_average (x : ℕ) (h : 20 * x + 158 = 21 * (x + 6)) : x = 32 :=
by
  sorry

end cricket_average_l199_199538


namespace molecular_weight_4_benzoic_acid_l199_199921

def benzoic_acid_molecular_weight : Float := (7 * 12.01) + (6 * 1.008) + (2 * 16.00)

def molecular_weight_4_moles_benzoic_acid (molecular_weight : Float) : Float := molecular_weight * 4

theorem molecular_weight_4_benzoic_acid :
  molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight = 488.472 :=
by
  unfold molecular_weight_4_moles_benzoic_acid benzoic_acid_molecular_weight
  -- rest of the proof
  sorry

end molecular_weight_4_benzoic_acid_l199_199921


namespace smallest_number_with_property_l199_199431

theorem smallest_number_with_property: 
  ∃ (N : ℕ), N = 25 ∧ (∀ (x : ℕ) (h : N = x + (x / 5)), N ≤ x) := 
  sorry

end smallest_number_with_property_l199_199431


namespace problem_solution_l199_199403

def complex_expression : ℕ := 3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1

theorem problem_solution : complex_expression = 1492 := by
  sorry

end problem_solution_l199_199403


namespace fibonacci_odd_index_not_divisible_by_4k_plus_3_l199_199559

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_odd_index_not_divisible_by_4k_plus_3 (n k : ℕ) (p : ℕ) (h : p = 4 * k + 3) : ¬ (p ∣ fibonacci (2 * n - 1)) :=
by
  sorry

end fibonacci_odd_index_not_divisible_by_4k_plus_3_l199_199559


namespace frequency_distribution_table_understanding_l199_199515

theorem frequency_distribution_table_understanding (size_sample_group : Prop) :
  (∃ (size_proportion : Prop) (corresponding_situation : Prop),
    size_sample_group → size_proportion ∧ corresponding_situation) :=
sorry

end frequency_distribution_table_understanding_l199_199515


namespace sum_of_possible_values_l199_199349

theorem sum_of_possible_values (x : ℝ) (h : x^2 - 4 * x + 4 = 0) : x = 2 :=
sorry

end sum_of_possible_values_l199_199349


namespace smallest_number_to_end_in_four_zeros_l199_199226

theorem smallest_number_to_end_in_four_zeros (x : ℕ) :
  let n1 := 225
  let n2 := 525
  let factor_needed := 16
  (∃ y : ℕ, y = n1 * n2 * x) ∧ (10^4 ∣ n1 * n2 * x) ↔ x = factor_needed :=
by
  sorry

end smallest_number_to_end_in_four_zeros_l199_199226


namespace add_fractions_11_12_7_15_l199_199674

/-- A theorem stating that the sum of 11/12 and 7/15 is 83/60. -/
theorem add_fractions_11_12_7_15 : (11 / 12) + (7 / 15) = (83 / 60) := 
by
  sorry

end add_fractions_11_12_7_15_l199_199674


namespace sum_not_divisible_by_10_iff_l199_199142

theorem sum_not_divisible_by_10_iff (n : ℕ) :
  ¬ (1981^n + 1982^n + 1983^n + 1984^n) % 10 = 0 ↔ n % 4 = 0 :=
sorry

end sum_not_divisible_by_10_iff_l199_199142


namespace max_height_of_ball_l199_199202

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

theorem max_height_of_ball : ∃ t : ℝ, (h t) = 69.5 :=
sorry

end max_height_of_ball_l199_199202


namespace cost_of_fencing_per_meter_l199_199548

theorem cost_of_fencing_per_meter (l b : ℕ) (total_cost : ℕ) (cost_per_meter : ℝ) : 
  (l = 66) → 
  (l = b + 32) → 
  (total_cost = 5300) → 
  (2 * l + 2 * b = 200) → 
  (cost_per_meter = total_cost / 200) → 
  cost_per_meter = 26.5 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof is omitted by design
  sorry

end cost_of_fencing_per_meter_l199_199548


namespace swap_instruments_readings_change_l199_199110

def U0 : ℝ := 45
def R : ℝ := 50
def r : ℝ := 20

theorem swap_instruments_readings_change :
  let I_total := U0 / (R / 2 + r)
  let U1 := I_total * r
  let I1 := I_total / 2
  let I2 := U0 / R
  let I := U0 / (R + r)
  let U2 := I * r
  let ΔI := I2 - I1
  let ΔU := U1 - U2
  ΔI = 0.4 ∧ ΔU = 7.14 :=
by
  sorry

end swap_instruments_readings_change_l199_199110


namespace units_digit_29_pow_8_pow_7_l199_199680

/-- The units digit of 29 raised to an arbitrary power follows a cyclical pattern. 
    For the purposes of this proof, we use that 29^k for even k ends in 1.
    Since 8^7 is even, we prove the units digit of 29^(8^7) is 1. -/
theorem units_digit_29_pow_8_pow_7 : (29^(8^7)) % 10 = 1 :=
by
  have even_power_cycle : ∀ k, k % 2 = 0 → (29^k) % 10 = 1 := sorry
  have eight_power_seven_even : (8^7) % 2 = 0 := by norm_num
  exact even_power_cycle (8^7) eight_power_seven_even

end units_digit_29_pow_8_pow_7_l199_199680


namespace angle_in_third_quadrant_l199_199910

theorem angle_in_third_quadrant (θ : ℤ) (hθ : θ = -510) : 
  (210 % 360 > 180 ∧ 210 % 360 < 270) := 
by
  have h : 210 % 360 = 210 := by norm_num
  sorry

end angle_in_third_quadrant_l199_199910


namespace bag_cost_is_2_l199_199204

-- Define the inputs and conditions
def carrots_per_day := 1
def days_per_year := 365
def carrots_per_bag := 5
def yearly_spending := 146

-- The final goal is to find the cost per bag
def cost_per_bag := yearly_spending / ((carrots_per_day * days_per_year) / carrots_per_bag)

-- Prove that the cost per bag is $2
theorem bag_cost_is_2 : cost_per_bag = 2 := by
  -- Using sorry to complete the proof
  sorry

end bag_cost_is_2_l199_199204


namespace evaluate_expression_l199_199067

theorem evaluate_expression : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := 
by 
  sorry

end evaluate_expression_l199_199067


namespace fred_initial_money_l199_199415

def initial_money (book_count : ℕ) (average_cost : ℕ) (money_left : ℕ) : ℕ :=
  book_count * average_cost + money_left

theorem fred_initial_money :
  initial_money 6 37 14 = 236 :=
by
  sorry

end fred_initial_money_l199_199415


namespace min_value_expr_l199_199743

theorem min_value_expr :
  ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by sorry

end min_value_expr_l199_199743


namespace monotonicity_f_max_value_f_l199_199596

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 1

theorem monotonicity_f :
  (∀ x, 0 < x ∧ x < Real.exp 1 → f x < f (Real.exp 1)) ∧
  (∀ x, x > Real.exp 1 → f x < f (Real.exp 1)) :=
sorry

theorem max_value_f (m : ℝ) (hm : m > 0) :
  (2 * m ≤ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log (2 * m)) / (2 * m) - 1) ∧
  (m ≥ Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = (Real.log m) / m - 1) ∧
  (Real.exp 1 / 2 < m ∧ m < Real.exp 1 → ∃ x ∈ Set.Icc m (2 * m), f x = 1 / Real.exp 1 - 1) :=
sorry

end monotonicity_f_max_value_f_l199_199596


namespace c_value_l199_199770

theorem c_value (c : ℝ) : (∃ a : ℝ, (x : ℝ) → x^2 + 200 * x + c = (x + a)^2) → c = 10000 := 
by
  intro h
  sorry

end c_value_l199_199770


namespace joey_pills_l199_199074

-- Definitions for the initial conditions
def TypeA_initial := 2
def TypeA_increment := 1

def TypeB_initial := 3
def TypeB_increment := 2

def TypeC_initial := 4
def TypeC_increment := 3

def days := 42

-- Function to calculate the sum of an arithmetic series
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

-- The theorem to be proved
theorem joey_pills :
  arithmetic_sum TypeA_initial TypeA_increment days = 945 ∧
  arithmetic_sum TypeB_initial TypeB_increment days = 1848 ∧
  arithmetic_sum TypeC_initial TypeC_increment days = 2751 :=
by sorry

end joey_pills_l199_199074


namespace trig_expression_evaluation_l199_199197

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := 
  sorry

end trig_expression_evaluation_l199_199197


namespace sum_cos_4x_4y_4z_l199_199632

theorem sum_cos_4x_4y_4z (x y z : ℝ)
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) :
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 :=
by
  sorry

end sum_cos_4x_4y_4z_l199_199632


namespace unique_quadruple_exists_l199_199141

theorem unique_quadruple_exists :
  ∃! (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
  a + b + c + d = 2 ∧
  a^2 + b^2 + c^2 + d^2 = 3 ∧
  (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 18 := by
  sorry

end unique_quadruple_exists_l199_199141


namespace total_rent_payment_l199_199524

def weekly_rent : ℕ := 388
def number_of_weeks : ℕ := 1359

theorem total_rent_payment : weekly_rent * number_of_weeks = 526692 := 
  by 
  sorry

end total_rent_payment_l199_199524


namespace gamma_minus_alpha_l199_199606

theorem gamma_minus_alpha (α β γ : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < γ) (h4 : γ < 2 * Real.pi)
    (h5 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
    γ - α = (4 * Real.pi) / 3 :=
sorry

end gamma_minus_alpha_l199_199606


namespace chandler_bike_purchase_weeks_l199_199188

theorem chandler_bike_purchase_weeks (bike_cost birthday_money weekly_earnings total_weeks : ℕ) 
  (h_bike_cost : bike_cost = 600)
  (h_birthday_money : birthday_money = 60 + 40 + 20 + 30)
  (h_weekly_earnings : weekly_earnings = 18)
  (h_total_weeks : total_weeks = 25) :
  birthday_money + weekly_earnings * total_weeks = bike_cost :=
by {
  sorry
}

end chandler_bike_purchase_weeks_l199_199188


namespace Powerjet_pumps_250_gallons_in_30_minutes_l199_199303

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end Powerjet_pumps_250_gallons_in_30_minutes_l199_199303


namespace new_volume_l199_199216

theorem new_volume (l w h : ℝ) 
  (h1: l * w * h = 3000) 
  (h2: l * w + w * h + l * h = 690) 
  (h3: l + w + h = 40) : 
  (l + 2) * (w + 2) * (h + 2) = 4548 := 
  sorry

end new_volume_l199_199216


namespace correct_function_at_x_equals_1_l199_199467

noncomputable def candidate_A (x : ℝ) : ℝ := (x - 1)^3 + 3 * (x - 1)
noncomputable def candidate_B (x : ℝ) : ℝ := 2 * (x - 1)^2
noncomputable def candidate_C (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def candidate_D (x : ℝ) : ℝ := x - 1

theorem correct_function_at_x_equals_1 :
  (deriv candidate_A 1 = 3) ∧ 
  (deriv candidate_B 1 ≠ 3) ∧ 
  (deriv candidate_C 1 ≠ 3) ∧ 
  (deriv candidate_D 1 ≠ 3) := 
by
  sorry

end correct_function_at_x_equals_1_l199_199467


namespace minimum_value_ineq_l199_199092

open Real

theorem minimum_value_ineq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
    (x + 1 / (y * y)) * (x + 1 / (y * y) - 500) + (y + 1 / (x * x)) * (y + 1 / (x * x) - 500) ≥ -125000 :=
by 
  sorry

end minimum_value_ineq_l199_199092


namespace remainder_add_mod_l199_199777

theorem remainder_add_mod (n : ℕ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := 
by sorry

end remainder_add_mod_l199_199777


namespace find_x_l199_199564

def x : ℕ := 70

theorem find_x :
  x + (5 * 12) / (180 / 3) = 71 :=
by
  sorry

end find_x_l199_199564


namespace find_a_l199_199069

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ := 2 * 3^n + a
noncomputable def a_1 (a : ℝ) : ℝ := S_n 1 a
noncomputable def a_2 (a : ℝ) : ℝ := S_n 2 a - S_n 1 a
noncomputable def a_3 (a : ℝ) : ℝ := S_n 3 a - S_n 2 a

theorem find_a (a : ℝ) : a_1 a * a_3 a = (a_2 a)^2 → a = -2 :=
by
  sorry

end find_a_l199_199069


namespace log_ab_is_pi_l199_199764

open Real

noncomputable def log_ab (a b : ℝ) : ℝ :=
(log b) / (log a)

theorem log_ab_is_pi (a b : ℝ)  (ha_pos: 0 < a) (ha_ne_one: a ≠ 1) (hb_pos: 0 < b) 
  (cond1 : log (a ^ 3) = log (b ^ 6)) (cond2 : cos (π * log a) = 1) : log_ab a b = π :=
by
  sorry

end log_ab_is_pi_l199_199764


namespace negation_proof_l199_199274

theorem negation_proof :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
by
  -- proof goes here
  sorry

end negation_proof_l199_199274


namespace axis_of_symmetry_shifted_sine_function_l199_199863

open Real

noncomputable def axisOfSymmetry (k : ℤ) : ℝ := k * π / 2 + π / 6

theorem axis_of_symmetry_shifted_sine_function (x : ℝ) (k : ℤ) :
  ∃ k : ℤ, x = axisOfSymmetry k := by
sorry

end axis_of_symmetry_shifted_sine_function_l199_199863


namespace not_enough_space_in_cube_l199_199370

-- Define the edge length of the cube in kilometers.
def cube_edge_length_km : ℝ := 3

-- Define the global population exceeding threshold.
def global_population : ℝ := 7 * 10^9

-- Define the function to calculate the volume of a cube given its edge length in kilometers.
def cube_volume_km (edge_length: ℝ) : ℝ := edge_length^3

-- Define the conversion from kilometers to meters.
def km_to_m (distance_km: ℝ) : ℝ := distance_km * 1000

-- Define the function to calculate the volume of the cube in cubic meters.
def cube_volume_m (edge_length_km: ℝ) : ℝ := (km_to_m edge_length_km)^3

-- Statement: The entire population and all buildings and structures will not fit inside the cube.
theorem not_enough_space_in_cube :
  cube_volume_m cube_edge_length_km < global_population * (some_constant_value_to_account_for_buildings_and_structures) :=
sorry

end not_enough_space_in_cube_l199_199370


namespace gasoline_price_increase_l199_199536

theorem gasoline_price_increase (high low : ℝ) (high_eq : high = 24) (low_eq : low = 18) : 
  ((high - low) / low) * 100 = 33.33 := 
  sorry

end gasoline_price_increase_l199_199536


namespace geometric_sequence_min_value_l199_199025

theorem geometric_sequence_min_value (r : ℝ) (a1 a2 a3 : ℝ) 
  (h1 : a1 = 1) 
  (h2 : a2 = a1 * r) 
  (h3 : a3 = a2 * r) :
  4 * a2 + 5 * a3 ≥ -(4 / 5) :=
by
  sorry

end geometric_sequence_min_value_l199_199025


namespace velocity_of_point_C_l199_199247

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end velocity_of_point_C_l199_199247


namespace smallest_n_with_units_digit_and_reorder_l199_199145

theorem smallest_n_with_units_digit_and_reorder :
  ∃ n : ℕ, (∃ a : ℕ, n = 10 * a + 6) ∧ (∃ m : ℕ, 6 * 10^m + a = 4 * n) ∧ n = 153846 :=
by
  sorry

end smallest_n_with_units_digit_and_reorder_l199_199145


namespace Woojin_harvested_weight_l199_199816

-- Definitions based on conditions
def younger_brother_harvest : Float := 3.8
def older_sister_harvest : Float := younger_brother_harvest + 8.4
def one_tenth_older_sister : Float := older_sister_harvest / 10
def woojin_extra_g : Float := 3720

-- Convert grams to kilograms
def grams_to_kg (g : Float) : Float := g / 1000

-- Theorem to be proven
theorem Woojin_harvested_weight :
  grams_to_kg (one_tenth_older_sister * 1000 + woojin_extra_g) = 4.94 :=
by
  sorry

end Woojin_harvested_weight_l199_199816


namespace find_c_d_l199_199972

theorem find_c_d (y : ℝ) (c d : ℕ) (hy : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (hform : ∃ (c d : ℕ), y = c + Real.sqrt d) : c + d = 42 :=
sorry

end find_c_d_l199_199972


namespace min_value_of_quadratic_l199_199549

theorem min_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y^2 - 6 * y + 5 ≥ (x - 3)^2 - 4) ∧ (y^2 - 6 * y + 5 = -4) :=
by sorry

end min_value_of_quadratic_l199_199549


namespace min_value_3x_4y_l199_199628

theorem min_value_3x_4y {x y : ℝ} (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
    3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_4y_l199_199628


namespace passengers_on_bus_l199_199112

theorem passengers_on_bus (initial_passengers : ℕ) (got_on : ℕ) (got_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 28 → got_on = 7 → got_off = 9 → final_passengers = initial_passengers + got_on - got_off → final_passengers = 26 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end passengers_on_bus_l199_199112


namespace inequality_minus_x_plus_3_l199_199102

variable (x y : ℝ)

theorem inequality_minus_x_plus_3 (h : x < y) : -x + 3 > -y + 3 :=
by {
  sorry
}

end inequality_minus_x_plus_3_l199_199102


namespace magic_square_sum_l199_199437

theorem magic_square_sum (a b c d e : ℕ) 
    (h1 : a + c + e = 55)
    (h2 : 30 + 10 + a = 55)
    (h3 : 30 + e + 15 = 55)
    (h4 : 10 + 30 + d = 55) :
    d + e = 25 := by
  sorry

end magic_square_sum_l199_199437


namespace sales_fifth_month_l199_199909

theorem sales_fifth_month
  (a1 a2 a3 a4 a6 : ℕ)
  (h1 : a1 = 2435)
  (h2 : a2 = 2920)
  (h3 : a3 = 2855)
  (h4 : a4 = 3230)
  (h6 : a6 = 1000)
  (avg : ℕ)
  (h_avg : avg = 2500) :
  a1 + a2 + a3 + a4 + (15000 - 1000 - (a1 + a2 + a3 + a4)) + a6 = avg * 6 :=
by
  sorry

end sales_fifth_month_l199_199909


namespace acceptable_colorings_correct_l199_199203

def acceptableColorings (n : ℕ) : ℕ :=
  (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2

theorem acceptable_colorings_correct (n : ℕ) :
  acceptableColorings n = (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2 :=
by
  sorry

end acceptable_colorings_correct_l199_199203


namespace initial_items_in_cart_l199_199745

theorem initial_items_in_cart (deleted_items : ℕ) (items_left : ℕ) (initial_items : ℕ) 
  (h1 : deleted_items = 10) (h2 : items_left = 8) : initial_items = 18 :=
by 
  -- Proof goes here
  sorry

end initial_items_in_cart_l199_199745


namespace smallest_possible_value_of_N_l199_199553

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l199_199553


namespace find_k_l199_199368

theorem find_k 
  (e1 : ℝ × ℝ) (h_e1 : e1 = (1, 0))
  (e2 : ℝ × ℝ) (h_e2 : e2 = (0, 1))
  (a : ℝ × ℝ) (h_a : a = (1, -2))
  (b : ℝ × ℝ) (h_b : b = (k, 1))
  (parallel : ∃ m : ℝ, a = (m * b.1, m * b.2)) : 
  k = -1/2 :=
sorry

end find_k_l199_199368


namespace solution_interval_l199_199992

theorem solution_interval (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ (5 / 2 : ℝ) < x ∧ x ≤ (14 / 5 : ℝ) := 
by
  sorry

end solution_interval_l199_199992


namespace infinitely_many_a_not_prime_l199_199254

theorem infinitely_many_a_not_prime (a: ℤ) (n: ℤ) : ∃ (b: ℤ), b ≥ 0 ∧ (∃ (N: ℕ) (a: ℤ), a = 4*(N:ℤ)^4 ∧ ∀ (n: ℤ), ¬Prime (n^4 + a)) :=
by { sorry }

end infinitely_many_a_not_prime_l199_199254


namespace square_area_is_81_l199_199229

def square_perimeter (s : ℕ) : ℕ := 4 * s
def square_area (s : ℕ) : ℕ := s * s

theorem square_area_is_81 (s : ℕ) (h : square_perimeter s = 36) : square_area s = 81 :=
by {
  sorry
}

end square_area_is_81_l199_199229


namespace necessary_not_sufficient_l199_199151

theorem necessary_not_sufficient (a b c d : ℝ) : 
  (a + c > b + d) → (a > b ∧ c > d) :=
sorry

end necessary_not_sufficient_l199_199151


namespace hexagon_vertices_zero_l199_199867

theorem hexagon_vertices_zero (n : ℕ) (a0 a1 a2 a3 a4 a5 : ℕ) 
  (h_sum : a0 + a1 + a2 + a3 + a4 + a5 = n) 
  (h_pos : 0 < n) :
  (n = 2 ∨ n % 2 = 1) → 
  ∃ (b0 b1 b2 b3 b4 b5 : ℕ), b0 = 0 ∧ b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 := sorry

end hexagon_vertices_zero_l199_199867


namespace fixed_point_of_line_l199_199851

theorem fixed_point_of_line (a : ℝ) : 
  (a + 3) * (-2) + (2 * a - 1) * 1 + 7 = 0 := 
by 
  sorry

end fixed_point_of_line_l199_199851


namespace f_increasing_on_positive_l199_199580

noncomputable def f (x : ℝ) : ℝ := - (1 / x) - 1

theorem f_increasing_on_positive (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 > x2) : f x1 > f x2 := by
  sorry

end f_increasing_on_positive_l199_199580


namespace minimum_value_of_expression_l199_199797

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + b^2 + (a + b)^2 + c^2

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  min_value_expression a b c = 9 :=
  sorry

end minimum_value_of_expression_l199_199797


namespace slope_of_line_l199_199043

theorem slope_of_line (x y : ℝ) :
  (∀ (x y : ℝ), (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4)) :=
by
  sorry

end slope_of_line_l199_199043


namespace parallelogram_base_is_36_l199_199441

def parallelogram_base (area height : ℕ) : ℕ :=
  area / height

theorem parallelogram_base_is_36 (h : parallelogram_base 864 24 = 36) : True :=
by
  trivial

end parallelogram_base_is_36_l199_199441


namespace wraps_add_more_l199_199002

/-- Let John's raw squat be 600 pounds. Let sleeves add 30 pounds to his lift. Let wraps add 25% 
to his squat. We aim to prove that wraps add 120 pounds more to John's squat than sleeves. -/
theorem wraps_add_more (raw_squat : ℝ) (sleeves_bonus : ℝ) (wraps_percentage : ℝ) : 
  raw_squat = 600 → sleeves_bonus = 30 → wraps_percentage = 0.25 → 
  (raw_squat * wraps_percentage) - sleeves_bonus = 120 :=
by
  intros h1 h2 h3
  sorry

end wraps_add_more_l199_199002


namespace james_bike_ride_l199_199126

variable {D P : ℝ}

theorem james_bike_ride :
  (∃ D P, 3 * D + (18 + 18 * 0.25) = 55.5 ∧ (18 = D * (1 + P / 100))) → P = 20 := by
  sorry

end james_bike_ride_l199_199126


namespace drinks_left_for_Seungwoo_l199_199969

def coke_taken_liters := 35 + 0.5
def cider_taken_liters := 27 + 0.2
def coke_drank_liters := 1 + 0.75

theorem drinks_left_for_Seungwoo :
  (coke_taken_liters - coke_drank_liters) + cider_taken_liters = 60.95 := by
  sorry

end drinks_left_for_Seungwoo_l199_199969


namespace a_4_eq_15_l199_199991

noncomputable def a : ℕ → ℕ
| 0 => 1
| (n + 1) => 2 * a n + 1

theorem a_4_eq_15 : a 3 = 15 :=
by
  sorry

end a_4_eq_15_l199_199991


namespace students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l199_199324

noncomputable def numStudentsKnowingSecret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem students_on_seventh_day :
  (numStudentsKnowingSecret 7) = 3280 :=
by
  sorry

theorem day_of_week (n : ℕ) : String :=
  if n % 7 = 0 then "Monday" else
  if n % 7 = 1 then "Tuesday" else
  if n % 7 = 2 then "Wednesday" else
  if n % 7 = 3 then "Thursday" else
  if n % 7 = 4 then "Friday" else
  if n % 7 = 5 then "Saturday" else
  "Sunday"

theorem day_when_3280_students_know_secret :
  day_of_week 7 = "Sunday" :=
by
  sorry

end students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l199_199324


namespace major_axis_of_ellipse_l199_199034

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define the length of the major axis
def major_axis_length : ℝ := 8

-- The theorem to prove
theorem major_axis_of_ellipse : 
  (∀ x y : ℝ, ellipse_eq x y) → major_axis_length = 8 :=
by
  sorry

end major_axis_of_ellipse_l199_199034


namespace number_of_boys_l199_199503

def school_problem (x y : ℕ) : Prop :=
  (x + y = 400) ∧ (y = (x / 100) * 400)

theorem number_of_boys (x y : ℕ) (h : school_problem x y) : x = 80 :=
by
  sorry

end number_of_boys_l199_199503


namespace john_hourly_wage_with_bonus_l199_199289

structure JohnJob where
  daily_wage : ℕ
  work_hours : ℕ
  bonus_amount : ℕ
  extra_hours : ℕ

def total_daily_wage (job : JohnJob) : ℕ :=
  job.daily_wage + job.bonus_amount

def total_work_hours (job : JohnJob) : ℕ :=
  job.work_hours + job.extra_hours

def hourly_wage (job : JohnJob) : ℕ :=
  total_daily_wage job / total_work_hours job

noncomputable def johns_job : JohnJob :=
  { daily_wage := 80, work_hours := 8, bonus_amount := 20, extra_hours := 2 }

theorem john_hourly_wage_with_bonus :
  hourly_wage johns_job = 10 :=
by
  sorry

end john_hourly_wage_with_bonus_l199_199289


namespace log_a_interval_l199_199195

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_a_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {a | log_a a 3 - log_a a 1 = 2} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
by
  sorry

end log_a_interval_l199_199195


namespace find_M_plus_N_l199_199021

theorem find_M_plus_N (M N : ℕ)
  (h1 : 4 * 63 = 7 * M)
  (h2 : 4 * N = 7 * 84) :
  M + N = 183 :=
by sorry

end find_M_plus_N_l199_199021


namespace opposite_numbers_expression_l199_199176

theorem opposite_numbers_expression (a b : ℤ) (h : a + b = 0) : 3 * a + 3 * b - 2 = -2 :=
by
  sorry

end opposite_numbers_expression_l199_199176


namespace simplify_expression_l199_199928

theorem simplify_expression (a b : ℤ) (h1 : a = 1) (h2 : b = -4) :
  4 * (a^2 * b + a * b^2) - 3 * (a^2 * b - 1) + 2 * a * b^2 - 6 = 89 := by
  sorry

end simplify_expression_l199_199928


namespace oldest_son_park_visits_l199_199047

theorem oldest_son_park_visits 
    (season_pass_cost : ℕ)
    (cost_per_trip : ℕ)
    (youngest_son_trips : ℕ) 
    (remaining_value : ℕ)
    (oldest_son_trips : ℕ) : 
    season_pass_cost = 100 →
    cost_per_trip = 4 →
    youngest_son_trips = 15 →
    remaining_value = season_pass_cost - youngest_son_trips * cost_per_trip →
    oldest_son_trips = remaining_value / cost_per_trip →
    oldest_son_trips = 10 := 
by sorry

end oldest_son_park_visits_l199_199047


namespace geometric_sequence_general_term_l199_199360

theorem geometric_sequence_general_term (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) 
  (h1 : a 5 = a1 * q^4)
  (h2 : a 10 = a1 * q^9)
  (h3 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h4 : ∀ n, a n = a1 * q^(n - 1))
  (h_inc : q > 1) :
  ∀ n, a n = 2^n :=
by
  sorry

end geometric_sequence_general_term_l199_199360


namespace area_rectangle_relation_l199_199163

theorem area_rectangle_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end area_rectangle_relation_l199_199163


namespace each_friend_paid_l199_199883

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid_l199_199883


namespace birthday_gift_l199_199381

-- Define the conditions
def friends : Nat := 8
def dollars_per_friend : Nat := 15

-- Formulate the statement to prove
theorem birthday_gift : friends * dollars_per_friend = 120 := by
  -- Proof is skipped using 'sorry'
  sorry

end birthday_gift_l199_199381


namespace second_tree_ring_groups_l199_199008

-- Definition of the problem conditions
def group_rings (fat thin : Nat) : Nat := fat + thin

-- Conditions
def FirstTreeRingGroups : Nat := 70
def RingsPerGroup : Nat := group_rings 2 4
def FirstTreeRings : Nat := FirstTreeRingGroups * RingsPerGroup
def AgeDifference : Nat := 180

-- Calculate the total number of rings in the second tree
def SecondTreeRings : Nat := FirstTreeRings - AgeDifference

-- Prove the number of ring groups in the second tree
theorem second_tree_ring_groups : SecondTreeRings / RingsPerGroup = 40 :=
by
  sorry

end second_tree_ring_groups_l199_199008


namespace probability_of_heads_on_999th_toss_l199_199032

theorem probability_of_heads_on_999th_toss (fair_coin : Bool → ℝ) :
  (∀ (i : ℕ), fair_coin true = 1 / 2 ∧ fair_coin false = 1 / 2) →
  fair_coin true = 1 / 2 :=
by
  sorry

end probability_of_heads_on_999th_toss_l199_199032


namespace pinocchio_optimal_success_probability_l199_199755

def success_prob (s : List ℚ) : ℚ :=
  s.foldr (λ x acc => (x * acc) / (1 - (1 - x) * acc)) 1

theorem pinocchio_optimal_success_probability :
  let success_probs := [9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10]
  success_prob success_probs = 0.4315 :=
by 
  sorry

end pinocchio_optimal_success_probability_l199_199755


namespace number_of_ways_to_choose_one_person_l199_199749

-- Definitions for the conditions
def people_using_first_method : ℕ := 3
def people_using_second_method : ℕ := 5

-- Definition of the total number of ways to choose one person
def total_ways_to_choose_one_person : ℕ :=
  people_using_first_method + people_using_second_method

-- Statement of the theorem to be proved
theorem number_of_ways_to_choose_one_person :
  total_ways_to_choose_one_person = 8 :=
by 
  sorry

end number_of_ways_to_choose_one_person_l199_199749


namespace green_balls_count_l199_199113

theorem green_balls_count (b g : ℕ) (h1 : b = 15) (h2 : 5 * g = 3 * b) : g = 9 :=
by
  sorry

end green_balls_count_l199_199113


namespace dropped_student_score_l199_199769

theorem dropped_student_score (total_students : ℕ) (remaining_students : ℕ) (initial_average : ℝ) (new_average : ℝ) (x : ℝ) 
  (h1 : total_students = 16) 
  (h2 : remaining_students = 15) 
  (h3 : initial_average = 62.5) 
  (h4 : new_average = 63.0) 
  (h5 : total_students * initial_average - remaining_students * new_average = x) : 
  x = 55 := 
sorry

end dropped_student_score_l199_199769


namespace min_value_of_function_l199_199903

theorem min_value_of_function (x : ℝ) (h : x > 2) : ∃ y, y = (x^2 - 4*x + 8) / (x - 2) ∧ (∀ z, z = (x^2 - 4*x + 8) / (x - 2) → y ≤ z) :=
sorry

end min_value_of_function_l199_199903


namespace min_diff_two_composite_sum_91_l199_199544

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ p * q = n

-- Minimum positive difference between two composite numbers that sum up to 91
theorem min_diff_two_composite_sum_91 : ∃ a b : ℕ, 
  is_composite a ∧ 
  is_composite b ∧ 
  a + b = 91 ∧ 
  b - a = 1 :=
by
  sorry

end min_diff_two_composite_sum_91_l199_199544


namespace value_of_expression_l199_199805

theorem value_of_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 5 = 6 :=
by
  sorry

end value_of_expression_l199_199805


namespace maximum_value_of_function_l199_199456

noncomputable def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem maximum_value_of_function :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = 25 / 4 :=
by 
  sorry

end maximum_value_of_function_l199_199456


namespace humans_can_live_l199_199146

variable (earth_surface : ℝ)
variable (water_fraction : ℝ := 3 / 5)
variable (inhabitable_land_fraction : ℝ := 2 / 3)

def inhabitable_fraction : ℝ := (1 - water_fraction) * inhabitable_land_fraction

theorem humans_can_live :
  inhabitable_fraction = 4 / 15 :=
by
  sorry

end humans_can_live_l199_199146


namespace mixed_oil_rate_l199_199616

noncomputable def rate_of_mixed_oil
  (volume1 : ℕ) (price1 : ℕ) (volume2 : ℕ) (price2 : ℕ) : ℚ :=
(total_cost : ℚ) / (total_volume : ℚ)
where
  total_cost := volume1 * price1 + volume2 * price2
  total_volume := volume1 + volume2

theorem mixed_oil_rate :
  rate_of_mixed_oil 10 50 5 66 = 55.33 := 
by
  sorry

end mixed_oil_rate_l199_199616


namespace problem_statement_l199_199033

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end problem_statement_l199_199033


namespace k_range_m_range_l199_199252

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem k_range (k : ℝ) : (∃ x : ℝ, g x = (2^x + 1) * f x + k) → k < 1 :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x1 : ℝ, 0 < x1 ∧ x1 < 1 → 
                        ∃ x2 : ℝ, -Real.pi / 4 ≤ x2 ∧ x2 ≤ Real.pi / 6 ∧ f x1 - m * 2^x1 > g x2) 
                       → m ≤ 7 / 6 :=
by
  sorry

end k_range_m_range_l199_199252


namespace biff_break_even_time_l199_199004

noncomputable def total_cost_excluding_wifi : ℝ :=
  11 + 3 + 16 + 8 + 10 + 35 + 0.1 * 35

noncomputable def total_cost_including_wifi_connection : ℝ :=
  total_cost_excluding_wifi + 5

noncomputable def effective_hourly_earning : ℝ := 12 - 1

noncomputable def hours_to_break_even : ℝ :=
  total_cost_including_wifi_connection / effective_hourly_earning

theorem biff_break_even_time : hours_to_break_even ≤ 9 := by
  sorry

end biff_break_even_time_l199_199004


namespace range_of_m_l199_199753

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x < y → -3 < x ∧ y < 3 → f x < f y)
  (h2 : ∀ m : ℝ, f (2 * m) < f (m + 1)) : 
  -3/2 < m ∧ m < 1 :=
  sorry

end range_of_m_l199_199753


namespace gcd_correct_l199_199352

def gcd_87654321_12345678 : ℕ :=
  gcd 87654321 12345678

theorem gcd_correct : gcd_87654321_12345678 = 75 := by 
  sorry

end gcd_correct_l199_199352


namespace josie_remaining_money_l199_199483

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end josie_remaining_money_l199_199483


namespace range_of_expr_l199_199922

theorem range_of_expr (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
by
  sorry

end range_of_expr_l199_199922


namespace tg_sum_equal_l199_199108

variable {a b c : ℝ}
variable {φA φB φC : ℝ}

-- The sides of the triangle are labeled such that a >= b >= c.
axiom sides_ineq : a ≥ b ∧ b ≥ c

-- The angles between the median and the altitude from vertices A, B, and C.
axiom angles_def : true -- This axiom is a placeholder. In actual use, we would define φA, φB, φC properly using the given geometric setup.

theorem tg_sum_equal : Real.tan φA + Real.tan φC = Real.tan φB := 
by 
  sorry

end tg_sum_equal_l199_199108


namespace david_reading_time_l199_199795

def total_time : ℕ := 180
def math_homework : ℕ := 25
def spelling_homework : ℕ := 30
def history_assignment : ℕ := 20
def science_project : ℕ := 15
def piano_practice : ℕ := 30
def study_breaks : ℕ := 2 * 10

def time_other_activities : ℕ := math_homework + spelling_homework + history_assignment + science_project + piano_practice + study_breaks

theorem david_reading_time : total_time - time_other_activities = 40 :=
by
  -- Calculation steps would go here, not provided for the theorem statement.
  sorry

end david_reading_time_l199_199795


namespace notecard_area_new_dimension_l199_199706

theorem notecard_area_new_dimension :
  ∀ (length : ℕ) (width : ℕ) (shortened : ℕ),
    length = 7 →
    width = 5 →
    shortened = 2 →
    (width - shortened) * length = 21 →
    (length - shortened) * (width - shortened + shortened) = 25 :=
by
  intros length width shortened h_length h_width h_shortened h_area
  sorry

end notecard_area_new_dimension_l199_199706


namespace polynomial_coeff_divisible_by_5_l199_199695

theorem polynomial_coeff_divisible_by_5 (a b c d : ℤ) 
  (h : ∀ (x : ℤ), (a * x^3 + b * x^2 + c * x + d) % 5 = 0) : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 := 
by
  sorry

end polynomial_coeff_divisible_by_5_l199_199695


namespace marcus_batches_l199_199645

theorem marcus_batches (B : ℕ) : (5 * B = 35) ∧ (35 - 8 = 27) → B = 7 :=
by {
  sorry
}

end marcus_batches_l199_199645


namespace house_painting_l199_199649

theorem house_painting (n : ℕ) (h1 : n = 1000)
  (occupants : Fin n → Fin n) (perm : ∀ i, occupants i ≠ i) :
  ∃ (coloring : Fin n → Fin 3), ∀ i, coloring i ≠ coloring (occupants i) :=
by
  sorry

end house_painting_l199_199649


namespace equation_of_line_l199_199800

theorem equation_of_line {M : ℝ × ℝ} {a b : ℝ} (hM : M = (4,2)) 
  (hAB : ∃ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ 
    A ≠ B ∧ ∀ x y : ℝ, 
    (x^2 + 4 * y^2 = 36 → (∃ k : ℝ, y - 2 = k * (x - 4) ) )):
  (x + 2 * y - 8 = 0) :=
sorry

end equation_of_line_l199_199800


namespace four_angles_for_shapes_l199_199831

-- Definitions for the shapes
def is_rectangle (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_square (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

def is_parallelogram (fig : Type) : Prop :=
  ∀ a b c d : fig, ∃ angles : ℕ, angles = 4

-- Main proposition
theorem four_angles_for_shapes {fig : Type} :
  (is_rectangle fig) ∧ (is_square fig) ∧ (is_parallelogram fig) →
  ∀ shape : fig, ∃ angles : ℕ, angles = 4 := by
  sorry

end four_angles_for_shapes_l199_199831


namespace production_today_l199_199652

def average_production (P : ℕ) (n : ℕ) := P / n

theorem production_today :
  ∀ (T P n : ℕ), n = 9 → average_production P n = 50 → average_production (P + T) (n + 1) = 54 → T = 90 :=
by
  intros T P n h1 h2 h3
  sorry

end production_today_l199_199652


namespace investment_plan_optimization_l199_199976

-- Define the given conditions.
def max_investment : ℝ := 100000
def max_loss : ℝ := 18000
def max_profit_A_rate : ℝ := 1.0     -- 100%
def max_profit_B_rate : ℝ := 0.5     -- 50%
def max_loss_A_rate : ℝ := 0.3       -- 30%
def max_loss_B_rate : ℝ := 0.1       -- 10%

-- Define the investment amounts.
def invest_A : ℝ := 40000
def invest_B : ℝ := 60000

-- Calculate profit and loss.
def profit : ℝ := (invest_A * max_profit_A_rate) + (invest_B * max_profit_B_rate)
def loss : ℝ := (invest_A * max_loss_A_rate) + (invest_B * max_loss_B_rate)
def total_investment : ℝ := invest_A + invest_B

-- Prove the required statement.
theorem investment_plan_optimization : 
    total_investment ≤ max_investment ∧ loss ≤ max_loss ∧ profit = 70000 :=
by
  simp [total_investment, profit, loss, invest_A, invest_B, 
    max_investment, max_profit_A_rate, max_profit_B_rate, 
    max_loss_A_rate, max_loss_B_rate, max_loss]
  sorry

end investment_plan_optimization_l199_199976


namespace sequence_general_term_l199_199810

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (n / (n + 1 : ℝ)) * a n) : 
  ∀ n, a n = 1 / n := by
  sorry

end sequence_general_term_l199_199810


namespace train_crossing_time_l199_199681

-- Defining basic conditions
def train_length : ℕ := 150
def platform_length : ℕ := 100
def time_to_cross_post : ℕ := 15

-- The time it takes for the train to cross the platform
theorem train_crossing_time :
  (train_length + platform_length) / (train_length / time_to_cross_post) = 25 := 
sorry

end train_crossing_time_l199_199681


namespace f_has_two_zeros_iff_l199_199371

open Real

noncomputable def f (x a : ℝ) : ℝ := (x - 2) * exp x + a * (x - 1)^2

theorem f_has_two_zeros_iff (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 0 < a :=
sorry

end f_has_two_zeros_iff_l199_199371


namespace inequality_of_pos_reals_l199_199956

open Real

theorem inequality_of_pos_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤
  (1 / 4) * (a + b + c) :=
by
  sorry

end inequality_of_pos_reals_l199_199956


namespace unique_n_for_prime_p_l199_199759

theorem unique_n_for_prime_p (p : ℕ) (hp1 : p > 2) (hp2 : Nat.Prime p) :
  ∃! (n : ℕ), (∃ (k : ℕ), n^2 + n * p = k^2) ∧ n = (p - 1) / 2 ^ 2 :=
sorry

end unique_n_for_prime_p_l199_199759


namespace fish_problem_l199_199608

theorem fish_problem : 
  ∀ (B T S : ℕ), 
    B = 10 → 
    T = 3 * B → 
    S = 35 → 
    B + T + S + 2 * S = 145 → 
    S - T = 5 :=
by sorry

end fish_problem_l199_199608


namespace arithmetic_sequence_geometric_condition_l199_199421

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1) (h_d_nonzero : d ≠ 0)
  (h_geom : (1 + d) * (1 + d) = 1 * (1 + 4 * d)) : a 2013 = 4025 := by sorry

end arithmetic_sequence_geometric_condition_l199_199421


namespace meters_to_examine_10000_l199_199472

def projection_for_sample (total_meters_examined : ℕ) (rejection_rate : ℝ) (sample_size : ℕ) :=
  total_meters_examined = sample_size

theorem meters_to_examine_10000 : 
  projection_for_sample 10000 0.015 10000 := by
  sorry

end meters_to_examine_10000_l199_199472


namespace range_of_c_l199_199684

variable {a b c : ℝ} -- Declare the variables

-- Define the conditions
def triangle_condition (a b : ℝ) : Prop :=
|a + b - 4| + (a - b + 2)^2 = 0

-- Define the proof problem
theorem range_of_c {a b c : ℝ} (h : triangle_condition a b) : 2 < c ∧ c < 4 :=
sorry -- Proof to be completed

end range_of_c_l199_199684


namespace sum_of_consecutive_odd_integers_l199_199705

-- Definitions of conditions
def consecutive_odd_integers (a b : ℤ) : Prop :=
  b = a + 2 ∧ (a % 2 = 1) ∧ (b % 2 = 1)

def five_times_smaller_minus_two_condition (a b : ℤ) : Prop :=
  b = 5 * a - 2

-- Theorem statement
theorem sum_of_consecutive_odd_integers (a b : ℤ)
  (h1 : consecutive_odd_integers a b)
  (h2 : five_times_smaller_minus_two_condition a b) : a + b = 4 :=
by
  sorry

end sum_of_consecutive_odd_integers_l199_199705


namespace total_cost_of_pets_l199_199035

theorem total_cost_of_pets 
  (num_puppies num_kittens num_parakeets : ℕ)
  (cost_parakeet cost_puppy cost_kitten : ℕ)
  (h1 : num_puppies = 2)
  (h2 : num_kittens = 2)
  (h3 : num_parakeets = 3)
  (h4 : cost_parakeet = 10)
  (h5 : cost_puppy = 3 * cost_parakeet)
  (h6 : cost_kitten = 2 * cost_parakeet) : 
  num_puppies * cost_puppy + num_kittens * cost_kitten + num_parakeets * cost_parakeet = 130 :=
by
  sorry

end total_cost_of_pets_l199_199035


namespace value_of_expression_l199_199709

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l199_199709


namespace integer_solution_exists_l199_199941

theorem integer_solution_exists : ∃ n : ℤ, (⌊(n^2 : ℚ) / 3⌋ - ⌊(n : ℚ) / 2⌋^2 = 3) ∧ n = 6 := by
  sorry

end integer_solution_exists_l199_199941


namespace keanu_total_spending_l199_199738

-- Definitions based on conditions
def dog_fish : Nat := 40
def cat_fish : Nat := dog_fish / 2
def total_fish : Nat := dog_fish + cat_fish
def cost_per_fish : Nat := 4
def total_cost : Nat := total_fish * cost_per_fish

-- Theorem statement
theorem keanu_total_spending : total_cost = 240 :=
by 
    sorry

end keanu_total_spending_l199_199738


namespace two_digit_number_as_expression_l199_199455

-- Define the conditions of the problem
variables (a : ℕ)

-- Statement to be proved
theorem two_digit_number_as_expression (h : 0 ≤ a ∧ a ≤ 9) : 10 * a + 1 = 10 * a + 1 := by
  sorry

end two_digit_number_as_expression_l199_199455


namespace quadratic_real_root_exists_l199_199299

theorem quadratic_real_root_exists :
  ¬ (∃ x : ℝ, x^2 + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 + x + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 - x + 1 = 0) ∧
  (∃ x : ℝ, x^2 - x - 1 = 0) :=
by
  sorry

end quadratic_real_root_exists_l199_199299


namespace range_of_a_l199_199828

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (1 / 2) * a * x^2 - 2 * x

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x - a * x - 1

theorem range_of_a
  (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f_prime x1 a = 0 ∧ f_prime x2 a = 0) ↔
  0 < a ∧ a < Real.exp (-2) :=
sorry

end range_of_a_l199_199828


namespace solve_for_p_l199_199531

theorem solve_for_p (p : ℕ) : 16^6 = 4^p → p = 12 := by
  sorry

end solve_for_p_l199_199531


namespace john_uses_six_pounds_of_vegetables_l199_199308

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end john_uses_six_pounds_of_vegetables_l199_199308


namespace slope_of_line_l199_199323

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l intersecting the parabola C at points A and B
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Condition based on the intersection and the given relationship 2 * (BF) = FA
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    (y1 = line k x1) ∧ (y2 = line k x2) ∧
    2 * (dist (x2, y2) focus) = dist focus (x1, y1)

-- The main theorem to be proven
theorem slope_of_line (k : ℝ) (A B : ℝ × ℝ) :
  intersection_condition k A B → k = 2 * Real.sqrt 2 :=
sorry

end slope_of_line_l199_199323


namespace matrix_corner_sum_eq_l199_199377

theorem matrix_corner_sum_eq (M : Matrix (Fin 2000) (Fin 2000) ℤ)
  (h : ∀ i j : Fin 1999, M i j + M (i+1) (j+1) = M i (j+1) + M (i+1) j) :
  M 0 0 + M 1999 1999 = M 0 1999 + M 1999 0 :=
sorry

end matrix_corner_sum_eq_l199_199377


namespace motorcycle_price_l199_199393

variable (x : ℝ) -- selling price of each motorcycle
variable (car_cost material_car material_motorcycle : ℝ)

theorem motorcycle_price
  (h1 : car_cost = 100)
  (h2 : material_car = 4 * 50)
  (h3 : material_motorcycle = 250)
  (h4 : 8 * x - material_motorcycle = material_car - car_cost + 50)
  : x = 50 := 
sorry

end motorcycle_price_l199_199393


namespace min_value_y_l199_199827

noncomputable def y (x : ℝ) := (2 - Real.cos x) / Real.sin x

theorem min_value_y (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) : 
  ∃ c ≥ 0, ∀ x, 0 < x ∧ x < Real.pi → y x ≥ c ∧ c = Real.sqrt 3 := 
sorry

end min_value_y_l199_199827


namespace students_interested_in_both_l199_199671

theorem students_interested_in_both (A B C Total : ℕ) (hA : A = 35) (hB : B = 45) (hC : C = 4) (hTotal : Total = 55) :
  A + B - 29 + C = Total :=
by
  -- Assuming the correct answer directly while skipping the proof.
  sorry

end students_interested_in_both_l199_199671


namespace triangle_area_l199_199626

theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 40) (h_inradius : inradius = 2.5) : 
  (inradius * (perimeter / 2)) = 50 :=
by
  -- Lean 4 statement code
  sorry

end triangle_area_l199_199626


namespace paula_paint_cans_l199_199871

variables (rooms_per_can total_rooms_lost initial_rooms final_rooms cans_lost : ℕ)

theorem paula_paint_cans
  (h1 : initial_rooms = 50)
  (h2 : cans_lost = 2)
  (h3 : final_rooms = 42)
  (h4 : total_rooms_lost = initial_rooms - final_rooms)
  (h5 : rooms_per_can = total_rooms_lost / cans_lost) :
  final_rooms / rooms_per_can = 11 :=
by sorry

end paula_paint_cans_l199_199871


namespace min_value_seq_l199_199911

theorem min_value_seq (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 26) (h₂ : ∀ n, a (n + 1) - a n = 2 * n + 1) :
  ∃ m, (m > 0) ∧ (∀ k, k > 0 → (a k / k : ℚ) ≥ 10) ∧ (a m / m : ℚ) = 10 :=
by
  sorry

end min_value_seq_l199_199911


namespace largest_common_term_in_sequences_l199_199389

/-- An arithmetic sequence starts with 3 and has a common difference of 10. A second sequence starts
with 5 and has a common difference of 8. In the range of 1 to 150, the largest number common to 
both sequences is 133. -/
theorem largest_common_term_in_sequences : ∃ (b : ℕ), b < 150 ∧ (∃ (n m : ℤ), b = 3 + 10 * n ∧ b = 5 + 8 * m) ∧ (b = 133) := 
by
  sorry

end largest_common_term_in_sequences_l199_199389


namespace question1_question2_l199_199276

theorem question1 :
  (1:ℝ) * (Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5 := 
by sorry

theorem question2 :
  (4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi)^0 = 1 - 3 * Real.sqrt 3 / 2 :=
by sorry

end question1_question2_l199_199276


namespace box_mass_calculation_l199_199083

variable (h₁ w₁ l₁ : ℝ) (m₁ : ℝ)
variable (h₂ w₂ l₂ density₁ density₂ : ℝ)

theorem box_mass_calculation
  (h₁_eq : h₁ = 3)
  (w₁_eq : w₁ = 4)
  (l₁_eq : l₁ = 6)
  (m₁_eq : m₁ = 72)
  (h₂_eq : h₂ = 1.5 * h₁)
  (w₂_eq : w₂ = 2.5 * w₁)
  (l₂_eq : l₂ = l₁)
  (density₂_eq : density₂ = 2 * density₁)
  (density₁_eq : density₁ = m₁ / (h₁ * w₁ * l₁)) :
  h₂ * w₂ * l₂ * density₂ = 540 := by
  sorry

end box_mass_calculation_l199_199083


namespace age_of_eldest_boy_l199_199915

theorem age_of_eldest_boy (x : ℕ) (h1 : (3*x + 5*x + 7*x) / 3 = 15) :
  7 * x = 21 :=
sorry

end age_of_eldest_boy_l199_199915


namespace find_natural_number_l199_199682

-- Define the problem statement
def satisfies_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ (2 * n^2 - 2) = k * (n^3 - n)

-- The main theorem
theorem find_natural_number (n : ℕ) : satisfies_condition n ↔ n = 2 :=
sorry

end find_natural_number_l199_199682


namespace ratio_jake_to_clementine_l199_199375

-- Definitions based on conditions
def ClementineCookies : Nat := 72
def ToryCookies (J : Nat) : Nat := (J + ClementineCookies) / 2
def TotalCookies (J : Nat) : Nat := ClementineCookies + J + ToryCookies J
def TotalRevenue : Nat := 648
def CookiePrice : Nat := 2
def TotalCookiesSold : Nat := TotalRevenue / CookiePrice

-- The main proof statement
theorem ratio_jake_to_clementine : 
  ∃ J : Nat, TotalCookies J = TotalCookiesSold ∧ J / ClementineCookies = 2 :=
by
  sorry

end ratio_jake_to_clementine_l199_199375


namespace fraction_is_percent_l199_199518

theorem fraction_is_percent (y : ℝ) (hy : y > 0) : (6 * y / 20 + 3 * y / 10) = (60 / 100) * y :=
by
  sorry

end fraction_is_percent_l199_199518


namespace average_student_headcount_l199_199660

theorem average_student_headcount : 
  let headcount_03_04 := 11500
  let headcount_04_05 := 11600
  let headcount_05_06 := 11300
  (headcount_03_04 + headcount_04_05 + headcount_05_06) / 3 = 11467 :=
by
  sorry

end average_student_headcount_l199_199660


namespace sector_area_l199_199158

theorem sector_area (r α : ℝ) (h_r : r = 3) (h_α : α = 2) : (1/2 * r^2 * α) = 9 := by
  sorry

end sector_area_l199_199158


namespace mean_of_all_students_is_79_l199_199140

def mean_score_all_students (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) : ℕ :=
  (36 * s + 75 * s) / ((2/5 * s) + s)

theorem mean_of_all_students_is_79 (F S : ℕ) (f s : ℕ) (hf : f = 2/5 * s) (hF : F = 90) (hS : S = 75) : 
  mean_score_all_students F S f s hf = 79 := by
  sorry

end mean_of_all_students_is_79_l199_199140


namespace mary_fruits_left_l199_199422

-- Conditions as definitions:
def mary_bought_apples : ℕ := 14
def mary_bought_oranges : ℕ := 9
def mary_bought_blueberries : ℕ := 6

def mary_ate_apples : ℕ := 1
def mary_ate_oranges : ℕ := 1
def mary_ate_blueberries : ℕ := 1

-- The problem statement:
theorem mary_fruits_left : 
  (mary_bought_apples - mary_ate_apples) + 
  (mary_bought_oranges - mary_ate_oranges) + 
  (mary_bought_blueberries - mary_ate_blueberries) = 26 := by
  sorry

end mary_fruits_left_l199_199422


namespace triangle_ratio_l199_199583

theorem triangle_ratio (a b c : ℕ) (r s : ℕ) (h1 : a = 9) (h2 : b = 15) (h3 : c = 18) (h4 : r + s = a) (h5 : r < s) : r * 2 = s :=
by
  sorry

end triangle_ratio_l199_199583


namespace andy_coats_l199_199007

theorem andy_coats 
  (initial_minks : ℕ)
  (offspring_4_minks count_4_offspring : ℕ)
  (offspring_6_minks count_6_offspring : ℕ)
  (offspring_8_minks count_8_offspring : ℕ)
  (freed_percentage coat_requirement total_minks offspring_minks freed_minks remaining_minks coats : ℕ) :
  initial_minks = 30 ∧
  offspring_4_minks = 10 ∧ count_4_offspring = 4 ∧
  offspring_6_minks = 15 ∧ count_6_offspring = 6 ∧
  offspring_8_minks = 5 ∧ count_8_offspring = 8 ∧
  freed_percentage = 60 ∧ coat_requirement = 15 ∧
  total_minks = initial_minks + offspring_minks ∧
  offspring_minks = offspring_4_minks * count_4_offspring + offspring_6_minks * count_6_offspring + offspring_8_minks * count_8_offspring ∧
  freed_minks = total_minks * freed_percentage / 100 ∧
  remaining_minks = total_minks - freed_minks ∧
  coats = remaining_minks / coat_requirement →
  coats = 5 :=
sorry

end andy_coats_l199_199007


namespace avg_marks_calculation_l199_199929

theorem avg_marks_calculation (max_score : ℕ)
    (gibi_percent jigi_percent mike_percent lizzy_percent : ℚ)
    (hg : gibi_percent = 0.59) (hj : jigi_percent = 0.55) 
    (hm : mike_percent = 0.99) (hl : lizzy_percent = 0.67)
    (hmax : max_score = 700) :
    ((gibi_percent * max_score + jigi_percent * max_score +
      mike_percent * max_score + lizzy_percent * max_score) / 4 = 490) :=
by
  sorry

end avg_marks_calculation_l199_199929


namespace interest_rate_calculation_l199_199517

theorem interest_rate_calculation (P1 P2 I1 I2 : ℝ) (r1 : ℝ) :
  P2 = 1648 ∧ P1 = 2678 - P2 ∧ I2 = P2 * 0.05 * 3 ∧ I1 = P1 * r1 * 8 ∧ I1 = I2 →
  r1 = 0.03 :=
by sorry

end interest_rate_calculation_l199_199517


namespace transport_load_with_trucks_l199_199451

theorem transport_load_with_trucks
  (total_weight : ℕ)
  (box_max_weight : ℕ)
  (truck_capacity : ℕ)
  (num_trucks : ℕ)
  (H_weight : total_weight = 13500)
  (H_box : box_max_weight = 350)
  (H_truck : truck_capacity = 1500)
  (H_num_trucks : num_trucks = 11) :
  ∃ (boxes : ℕ), boxes * box_max_weight >= total_weight ∧ num_trucks * truck_capacity >= total_weight := 
sorry

end transport_load_with_trucks_l199_199451


namespace total_questions_to_review_is_1750_l199_199079

-- Define the relevant conditions
def num_classes := 5
def students_per_class := 35
def questions_per_exam := 10

-- The total number of questions to be reviewed by Professor Oscar
def total_questions : Nat := num_classes * students_per_class * questions_per_exam

-- The theorem stating the equivalent proof problem
theorem total_questions_to_review_is_1750 : total_questions = 1750 := by
  -- proof steps are skipped here 
  sorry

end total_questions_to_review_is_1750_l199_199079


namespace coins_remainder_l199_199366

theorem coins_remainder (N : ℕ) (h1 : N % 8 = 5) (h2 : N % 7 = 2) (hN_min : ∀ M : ℕ, (M % 8 = 5 ∧ M % 7 = 2) → N ≤ M) : N % 9 = 1 :=
sorry

end coins_remainder_l199_199366


namespace abs_k_eq_sqrt_19_div_4_l199_199379

theorem abs_k_eq_sqrt_19_div_4
  (k : ℝ)
  (h : ∀ x : ℝ, x^2 - 4 * k * x + 1 = 0 → (x = r ∨ x = s))
  (h₁ : r + s = 4 * k)
  (h₂ : r * s = 1)
  (h₃ : r^2 + s^2 = 17) :
  |k| = (Real.sqrt 19) / 4 := by
sorry

end abs_k_eq_sqrt_19_div_4_l199_199379


namespace largest_divisor_of_n_squared_sub_n_squared_l199_199557

theorem largest_divisor_of_n_squared_sub_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n_squared_sub_n_squared_l199_199557


namespace find_angle4_l199_199222

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ)
                    (h1 : angle1 + angle2 = 180)
                    (h2 : angle3 = 2 * angle4)
                    (h3 : angle1 = 50)
                    (h4 : angle3 + angle4 = 130) : 
                    angle4 = 130 / 3 := by 
    sorry

end find_angle4_l199_199222


namespace factorize_quadratic_l199_199912

theorem factorize_quadratic (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 :=
by
  sorry

end factorize_quadratic_l199_199912


namespace frank_oranges_correct_l199_199707

def betty_oranges : ℕ := 12
def sandra_oranges : ℕ := 3 * betty_oranges
def emily_oranges : ℕ := 7 * sandra_oranges
def frank_oranges : ℕ := 5 * emily_oranges

theorem frank_oranges_correct : frank_oranges = 1260 := by
  sorry

end frank_oranges_correct_l199_199707


namespace lcm_gcf_ratio_120_504_l199_199630

theorem lcm_gcf_ratio_120_504 : 
  let a := 120
  let b := 504
  (Int.lcm a b) / (Int.gcd a b) = 105 := by
  sorry

end lcm_gcf_ratio_120_504_l199_199630


namespace fencing_required_l199_199669

theorem fencing_required (L W : ℕ) (area : ℕ) (hL : L = 20) (hA : area = 120) (hW : area = L * W) :
  2 * W + L = 32 :=
by
  -- Steps and proof logic to be provided here
  sorry

end fencing_required_l199_199669


namespace sum_of_absolute_values_l199_199977

variables {a : ℕ → ℤ} {S₁₀ S₁₈ : ℤ} {T₁₈ : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem sum_of_absolute_values 
  (h1 : a 0 > 0) 
  (h2 : a 9 * a 10 < 0) 
  (h3 : sum_n_terms a 9 = 36) 
  (h4 : sum_n_terms a 17 = 12) :
  (sum_n_terms a 9) - (sum_n_terms a 17 - sum_n_terms a 9) = 60 :=
sorry

end sum_of_absolute_values_l199_199977


namespace solve_equation_l199_199696

theorem solve_equation : ∀ (x : ℝ), (2 * x + 5 = 3 * x - 2) → (x = 7) :=
by
  intro x
  intro h
  sorry

end solve_equation_l199_199696


namespace total_seashells_l199_199943

theorem total_seashells 
  (sally_seashells : ℕ)
  (tom_seashells : ℕ)
  (jessica_seashells : ℕ)
  (h1 : sally_seashells = 9)
  (h2 : tom_seashells = 7)
  (h3 : jessica_seashells = 5) : 
  sally_seashells + tom_seashells + jessica_seashells = 21 :=
by
  sorry

end total_seashells_l199_199943


namespace volume_increase_factor_l199_199699

   variable (π : ℝ) (r h : ℝ)

   def original_volume : ℝ := π * r^2 * h

   def new_height : ℝ := 3 * h

   def new_radius : ℝ := 2.5 * r

   def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

   theorem volume_increase_factor :
     new_volume π r h = 18.75 * original_volume π r h := 
   by
     sorry
   
end volume_increase_factor_l199_199699


namespace total_stamps_in_collection_l199_199408

-- Definitions reflecting the problem conditions
def foreign_stamps : ℕ := 90
def old_stamps : ℕ := 60
def both_foreign_and_old_stamps : ℕ := 20
def neither_foreign_nor_old_stamps : ℕ := 70

-- The expected total number of stamps in the collection
def total_stamps : ℕ :=
  (foreign_stamps + old_stamps - both_foreign_and_old_stamps) + neither_foreign_nor_old_stamps

-- Statement to prove the total number of stamps is 200
theorem total_stamps_in_collection : total_stamps = 200 := by
  -- Proof omitted
  sorry

end total_stamps_in_collection_l199_199408


namespace product_of_x_and_y_l199_199288

theorem product_of_x_and_y (x y a b : ℝ)
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := 
  sorry

end product_of_x_and_y_l199_199288


namespace number_of_blocks_needed_to_form_cube_l199_199673

-- Define the dimensions of the rectangular block
def block_length : ℕ := 5
def block_width : ℕ := 4
def block_height : ℕ := 3

-- Define the side length of the cube
def cube_side_length : ℕ := 60

-- The expected number of rectangular blocks needed
def expected_number_of_blocks : ℕ := 3600

-- Statement to prove the number of rectangular blocks needed to form the cube
theorem number_of_blocks_needed_to_form_cube
  (l : ℕ) (w : ℕ) (h : ℕ) (cube_side : ℕ) (expected_count : ℕ)
  (h_l : l = block_length)
  (h_w : w = block_width)
  (h_h : h = block_height)
  (h_cube_side : cube_side = cube_side_length)
  (h_expected : expected_count = expected_number_of_blocks) :
  (cube_side ^ 3) / (l * w * h) = expected_count :=
sorry

end number_of_blocks_needed_to_form_cube_l199_199673


namespace product_of_number_subtracting_7_equals_9_l199_199194

theorem product_of_number_subtracting_7_equals_9 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end product_of_number_subtracting_7_equals_9_l199_199194


namespace g_f_neg5_l199_199156

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end g_f_neg5_l199_199156


namespace original_cube_edge_length_l199_199378

theorem original_cube_edge_length (a : ℕ) (h1 : 6 * (a ^ 3) = 7 * (6 * (a ^ 2))) : a = 7 := 
by 
  sorry

end original_cube_edge_length_l199_199378


namespace joanne_total_weekly_earnings_l199_199057

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end joanne_total_weekly_earnings_l199_199057


namespace sale_price_l199_199465

def original_price : ℝ := 100
def discount_rate : ℝ := 0.80

theorem sale_price (original_price discount_rate : ℝ) : original_price * (1 - discount_rate) = 20 := by
  sorry

end sale_price_l199_199465


namespace max_minutes_sleep_without_missing_happy_moment_l199_199246

def isHappyMoment (h m : ℕ) : Prop :=
  (h = 4 * m ∨ m = 4 * h) ∧ h < 24 ∧ m < 60

def sleepDurationMax : ℕ :=
  239

theorem max_minutes_sleep_without_missing_happy_moment :
  ∀ (sleepDuration : ℕ), sleepDuration ≤ 239 :=
sorry

end max_minutes_sleep_without_missing_happy_moment_l199_199246


namespace oak_grove_libraries_total_books_l199_199273

theorem oak_grove_libraries_total_books :
  let publicLibraryBooks := 1986
  let schoolLibrariesBooks := 5106
  let communityCollegeLibraryBooks := 3294.5
  let medicalLibraryBooks := 1342.25
  let lawLibraryBooks := 2785.75
  publicLibraryBooks + schoolLibrariesBooks + communityCollegeLibraryBooks + medicalLibraryBooks + lawLibraryBooks = 15514.5 :=
by
  sorry

end oak_grove_libraries_total_books_l199_199273


namespace digit_difference_one_l199_199326

variable (d C D : ℕ)

-- Assumptions
variables (h1 : d > 8)
variables (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3)

theorem digit_difference_one (h1 : d > 8) (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3) :
  C - D = 1 :=
by
  sorry

end digit_difference_one_l199_199326


namespace count_distinct_reals_a_with_integer_roots_l199_199633

-- Define the quadratic equation with its roots and conditions
theorem count_distinct_reals_a_with_integer_roots :
  ∃ (a_vals : Finset ℝ), a_vals.card = 6 ∧
    (∀ a ∈ a_vals, ∃ r s : ℤ, 
      (r + s : ℝ) = -a ∧ (r * s : ℝ) = 9 * a) :=
by
  sorry

end count_distinct_reals_a_with_integer_roots_l199_199633


namespace machines_working_time_l199_199550

theorem machines_working_time (y: ℝ) 
  (h1 : y + 8 > 0)  -- condition for time taken by S
  (h2 : y + 2 > 0)  -- condition for time taken by T
  (h3 : 2 * y > 0)  -- condition for time taken by U
  : (1 / (y + 8) + 1 / (y + 2) + 1 / (2 * y) = 1 / y) ↔ (y = 3 / 2) := 
by
  have h4 : y ≠ 0 := by linarith [h1, h2, h3]
  sorry

end machines_working_time_l199_199550


namespace cube_dimension_ratio_l199_199737

theorem cube_dimension_ratio (V1 V2 : ℕ) (h1 : V1 = 27) (h2 : V2 = 216) :
  ∃ r : ℕ, r = 2 ∧ (∃ l1 l2 : ℕ, l1 * l1 * l1 = V1 ∧ l2 * l2 * l2 = V2 ∧ l2 = r * l1) :=
by
  sorry

end cube_dimension_ratio_l199_199737


namespace meiosis_fertilization_correct_l199_199874

theorem meiosis_fertilization_correct :
  (∀ (half_nuclear_sperm half_nuclear_egg mitochondrial_egg : Prop)
     (recognition_basis_clycoproteins : Prop)
     (fusion_basis_nuclei : Prop)
     (meiosis_eukaryotes : Prop)
     (random_fertilization : Prop),
    (half_nuclear_sperm ∧ half_nuclear_egg ∧ mitochondrial_egg ∧ recognition_basis_clycoproteins ∧ fusion_basis_nuclei ∧ meiosis_eukaryotes ∧ random_fertilization) →
    (D : Prop) ) := 
sorry

end meiosis_fertilization_correct_l199_199874


namespace domain_of_tan_arcsin_xsq_l199_199552

noncomputable def domain_f (x : ℝ) : Prop :=
  x ≠ 1 ∧ x ≠ -1 ∧ -1 ≤ x ∧ x ≤ 1

theorem domain_of_tan_arcsin_xsq :
  ∀ x : ℝ, -1 < x ∧ x < 1 ↔ domain_f x := 
sorry

end domain_of_tan_arcsin_xsq_l199_199552


namespace problem_part1_problem_part2_l199_199097

section DecreasingNumber

def is_decreasing_number (a b c d : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧
  10 * a + b - (10 * b + c) = 10 * c + d

theorem problem_part1 (a : ℕ) :
  is_decreasing_number a 3 1 2 → a = 4 :=
by
  intro h
  -- Proof steps
  sorry

theorem problem_part2 (a b c d : ℕ) :
  is_decreasing_number a b c d →
  (100 * a + 10 * b + c + 100 * b + 10 * c + d) % 9 = 0 →
  8165 = max_value :=
by
  intro h1 h2
  -- Proof steps
  sorry

end DecreasingNumber

end problem_part1_problem_part2_l199_199097


namespace determine_integer_n_l199_199358

theorem determine_integer_n (n : ℤ) :
  (n + 15 ≥ 16) ∧ (-5 * n < -10) → n = 3 :=
by
  sorry

end determine_integer_n_l199_199358


namespace total_blue_marbles_correct_l199_199012

def total_blue_marbles (j t e : ℕ) : ℕ :=
  j + t + e

theorem total_blue_marbles_correct :
  total_blue_marbles 44 24 36 = 104 :=
by
  sorry

end total_blue_marbles_correct_l199_199012


namespace problem_statement_l199_199398

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem problem_statement : vec_dot (vec_add a (vec_scalar_mul 2 b)) c = -3 := 
by
  sorry

end problem_statement_l199_199398


namespace star_calculation_l199_199642

-- Define the operation '*' via the given table
def star_table : Matrix (Fin 5) (Fin 5) (Fin 5) :=
  ![
    ![0, 1, 2, 3, 4],
    ![1, 0, 4, 2, 3],
    ![2, 3, 1, 4, 0],
    ![3, 4, 0, 1, 2],
    ![4, 2, 3, 0, 1]
  ]

def star (a b : Fin 5) : Fin 5 := star_table a b

-- Prove (3 * 5) * (2 * 4) = 3
theorem star_calculation : star (star 2 4) (star 4 1) = 2 := by
  sorry

end star_calculation_l199_199642


namespace ratio_of_areas_is_correct_l199_199063

-- Definition of the lengths of the sides of the triangles
def triangle_XYZ_sides := (7, 24, 25)
def triangle_PQR_sides := (9, 40, 41)

-- Definition of the areas of the right triangles
def area_triangle_XYZ := (7 * 24) / 2
def area_triangle_PQR := (9 * 40) / 2

-- The ratio of the areas of the triangles
def ratio_of_areas := area_triangle_XYZ / area_triangle_PQR

-- The expected answer
def expected_ratio := 7 / 15

-- The theorem proving that ratio_of_areas is equal to expected_ratio
theorem ratio_of_areas_is_correct :
  ratio_of_areas = expected_ratio := by
  -- Add the proof here
  sorry

end ratio_of_areas_is_correct_l199_199063


namespace compute_expression_l199_199873

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l199_199873


namespace coefficient_x3y5_l199_199523

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition for the binomial expansion term of (x-y)^7
def expansion_term (r : ℕ) : ℤ := 
  (binom 7 r) * (-1) ^ r

-- The target coefficient for the term x^3 y^5 in (x+y)(x-y)^7
theorem coefficient_x3y5 :
  (expansion_term 5) * 1 + (expansion_term 4) * 1 = 14 :=
by
  -- Proof to be filled in
  sorry

end coefficient_x3y5_l199_199523


namespace plan_b_more_cost_effective_l199_199327

noncomputable def fare (x : ℝ) : ℝ :=
if x < 3 then 5
else if x <= 10 then 1.2 * x + 1.4
else 1.8 * x - 4.6

theorem plan_b_more_cost_effective :
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  plan_a > plan_b :=
by
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  sorry

end plan_b_more_cost_effective_l199_199327


namespace sally_balance_fraction_l199_199658

variable (G : ℝ) (x : ℝ)
-- spending limit on gold card is G
-- spending limit on platinum card is 2G
-- Balance on platinum card is G/2
-- After transfer, 0.5833333333333334 portion of platinum card remains unspent

theorem sally_balance_fraction
  (h1 : (5/12) * 2 * G = G / 2 + x * G) : x = 1 / 3 :=
by
  sorry

end sally_balance_fraction_l199_199658


namespace partial_fraction_identity_l199_199889

theorem partial_fraction_identity
  (P Q R : ℝ)
  (h1 : -2 = P + Q)
  (h2 : 1 = Q + R)
  (h3 : -1 = P + R) :
  (P, Q, R) = (-2, 0, 1) :=
by
  sorry

end partial_fraction_identity_l199_199889


namespace sean_div_julie_l199_199162

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l199_199162


namespace total_canoes_built_l199_199887

-- Defining basic variables and functions for the proof
variable (a : Nat := 5) -- Initial number of canoes in January
variable (r : Nat := 3) -- Common ratio
variable (n : Nat := 6) -- Number of months including January

-- Function to compute sum of the first n terms of a geometric series
def geometric_sum (a r n : Nat) : Nat :=
  a * (r^n - 1) / (r - 1)

-- The proposition we want to prove
theorem total_canoes_built : geometric_sum a r n = 1820 := by
  sorry

end total_canoes_built_l199_199887


namespace find_fraction_l199_199479

theorem find_fraction (f n : ℝ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1 / 5 :=
by
  -- skipping the proof as requested
  sorry

end find_fraction_l199_199479


namespace point_on_line_l199_199328

theorem point_on_line : 
  ∃ t : ℚ, (3 * t + 1 = 0) ∧ ((2 - 4) / (t - 1) = (7 - 4) / (3 - 1)) :=
by
  sorry

end point_on_line_l199_199328


namespace perfect_square_base9_last_digit_l199_199022

-- We define the problem conditions
variable {b d f : ℕ} -- all variables are natural numbers
-- Condition 1: Base 9 representation of a perfect square
variable (n : ℕ) -- n is the perfect square number
variable (sqrt_n : ℕ) -- sqrt_n is the square root of n (so, n = sqrt_n^2)
variable (h1 : n = b * 9^3 + d * 9^2 + 4 * 9 + f)
variable (h2 : b ≠ 0)
-- The question becomes that the possible values of f are 0, 1, or 4
theorem perfect_square_base9_last_digit (h3 : n = sqrt_n^2) (hb : b ≠ 0) : 
  (f = 0) ∨ (f = 1) ∨ (f = 4) :=
by
  sorry

end perfect_square_base9_last_digit_l199_199022


namespace prime_squares_5000_9000_l199_199178

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l199_199178


namespace b_over_a_squared_eq_seven_l199_199807

theorem b_over_a_squared_eq_seven (a b k : ℕ) (ha : a > 1) (hb : b = a * (10^k + 1)) (hdiv : a^2 ∣ b) :
  b / a^2 = 7 :=
sorry

end b_over_a_squared_eq_seven_l199_199807


namespace find_natural_pairs_l199_199134

theorem find_natural_pairs (a b : ℕ) :
  (∃ A, A * A = a ^ 2 + 3 * b) ∧ (∃ B, B * B = b ^ 2 + 3 * a) ↔ 
  (a = 1 ∧ b = 1) ∨ (a = 11 ∧ b = 11) ∨ (a = 16 ∧ b = 11) :=
by
  sorry

end find_natural_pairs_l199_199134


namespace shaded_area_is_28_l199_199989

theorem shaded_area_is_28 (A B : ℕ) (h1 : A = 64) (h2 : B = 28) : B = 28 := by
  sorry

end shaded_area_is_28_l199_199989


namespace sum_of_digits_of_fraction_is_nine_l199_199935

theorem sum_of_digits_of_fraction_is_nine : 
  ∃ (x y : Nat), (4 / 11 : ℚ) = x / 10 + y / 100 + x / 1000 + y / 10000 + (x + y) / 100000 -- and other terms
  ∧ x + y = 9 := 
sorry

end sum_of_digits_of_fraction_is_nine_l199_199935


namespace intersection_with_negative_y_axis_max_value_at_x3_l199_199566

theorem intersection_with_negative_y_axis (m : ℝ) (h : 4 - 2 * m < 0) : m > 2 :=
sorry

theorem max_value_at_x3 (m : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 3 * x + 4 - 2 * m ≤ -4) : m = 8.5 :=
sorry

end intersection_with_negative_y_axis_max_value_at_x3_l199_199566


namespace probability_at_least_one_male_l199_199362

-- Definitions according to the problem conditions
def total_finalists : ℕ := 8
def female_finalists : ℕ := 5
def male_finalists : ℕ := 3
def num_selected : ℕ := 3

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probabilistic statement
theorem probability_at_least_one_male :
  let total_ways := binom total_finalists num_selected
  let ways_all_females := binom female_finalists num_selected
  let ways_at_least_one_male := total_ways - ways_all_females
  (ways_at_least_one_male : ℚ) / total_ways = 23 / 28 :=
by
  sorry

end probability_at_least_one_male_l199_199362


namespace lily_pads_doubling_l199_199236

theorem lily_pads_doubling (patch_half_day: ℕ) (doubling_rate: ℝ)
  (H1: patch_half_day = 49)
  (H2: doubling_rate = 2): (patch_half_day + 1) = 50 :=
by 
  sorry

end lily_pads_doubling_l199_199236


namespace largest_three_digit_sum_l199_199882

open Nat

def isDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def areDistinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem largest_three_digit_sum : 
  ∀ (X Y Z : ℕ), isDigit X → isDigit Y → isDigit Z → areDistinct X Y Z →
  100 ≤  (110 * X + 11 * Y + 2 * Z) → (110 * X + 11 * Y + 2 * Z) ≤ 999 → 
  110 * X + 11 * Y + 2 * Z ≤ 982 :=
by
  intros
  sorry

end largest_three_digit_sum_l199_199882


namespace find_r_over_s_at_0_l199_199820

noncomputable def r (x : ℝ) : ℝ := -3 * (x + 1) * (x - 2)
noncomputable def s (x : ℝ) : ℝ := (x + 1) * (x - 3)

theorem find_r_over_s_at_0 : (r 0) / (s 0) = 2 := by
  sorry

end find_r_over_s_at_0_l199_199820


namespace max_xy_l199_199244

theorem max_xy : 
  ∃ x y : ℕ, 5 * x + 3 * y = 100 ∧ x > 0 ∧ y > 0 ∧ x * y = 165 :=
by
  sorry

end max_xy_l199_199244


namespace P_and_Q_together_l199_199527

theorem P_and_Q_together (W : ℝ) (H : W > 0) :
  (1 / (1 / 4 + 1 / (1 / 3 * (1 / 4)))) = 3 :=
by
  sorry

end P_and_Q_together_l199_199527


namespace smallest_integer_y_l199_199756

theorem smallest_integer_y : ∃ (y : ℤ), (7 + 3 * y < 25) ∧ (∀ z : ℤ, (7 + 3 * z < 25) → y ≤ z) ∧ y = 5 :=
by
  sorry

end smallest_integer_y_l199_199756


namespace each_person_paid_45_l199_199373

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l199_199373


namespace sum_of_coefficients_l199_199923

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) 
  (h1 : quadratic a b c 3 = 0) 
  (h2 : quadratic a b c 7 = 0)
  (h3 : ∃ x0, (∀ x, quadratic a b c x ≥ quadratic a b c x0) ∧ quadratic a b c x0 = 20) :
  a + b + c = -105 :=
by 
  sorry

end sum_of_coefficients_l199_199923


namespace rachel_steps_l199_199392

theorem rachel_steps (x : ℕ) (h1 : x + 325 = 892) : x = 567 :=
sorry

end rachel_steps_l199_199392


namespace correct_average_marks_l199_199571

theorem correct_average_marks
  (n : ℕ) (avg_mks wrong_mk correct_mk correct_avg_mks : ℕ)
  (H1 : n = 10)
  (H2 : avg_mks = 100)
  (H3 : wrong_mk = 50)
  (H4 : correct_mk = 10)
  (H5 : correct_avg_mks = 96) :
  (n * avg_mks - wrong_mk + correct_mk) / n = correct_avg_mks :=
by
  sorry

end correct_average_marks_l199_199571


namespace gym_class_students_correct_l199_199081

noncomputable def check_gym_class_studens :=
  let P1 := 15
  let P2 := 5
  let P3 := 12.5
  let P4 := 9.166666666666666
  let P5 := 8.333333333333334
  P1 = P2 + 10 ∧
  P2 = 2 * P3 - 20 ∧
  P3 = P4 + P5 - 5 ∧
  P4 = (1 / 2) * P5 + 5

theorem gym_class_students_correct : check_gym_class_studens := by
  simp [check_gym_class_studens]
  sorry

end gym_class_students_correct_l199_199081


namespace matrix_problem_l199_199213

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![0, 0], -- Correct values for B can be computed from conditions if needed
  ![0, 0]
]

theorem matrix_problem (A B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : A + B = A * B)
  (h2 : A * B = ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]) :
  B * A = ![
    ![20 / 3, 4 / 3],
    ![-8 / 3, 8 / 3]
  ] :=
sorry

end matrix_problem_l199_199213


namespace solve_for_k_l199_199974

theorem solve_for_k : 
  ∃ (k : ℕ), k > 0 ∧ k * k = 2012 * 2012 + 2010 * 2011 * 2013 * 2014 ∧ k = 4048142 :=
sorry

end solve_for_k_l199_199974


namespace cos_double_angle_l199_199801

theorem cos_double_angle (α β : Real) 
    (h1 : Real.sin α = Real.cos β) 
    (h2 : Real.sin α * Real.cos β - 2 * Real.cos α * Real.sin β = 1 / 2) :
    Real.cos (2 * β) = 2 / 3 :=
by
  sorry

end cos_double_angle_l199_199801


namespace find_value_am2_bm_minus_7_l199_199094

variable {a b m : ℝ}

theorem find_value_am2_bm_minus_7
  (h : a * m^2 + b * m + 5 = 0) : a * m^2 + b * m - 7 = -12 :=
by
  sorry

end find_value_am2_bm_minus_7_l199_199094


namespace problem1_problem2_problem3_problem4_l199_199988

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2023 + (π - 3.14) ^ 0 - ((-1 / 2 : ℚ) ^ (-2 : ℤ)) = -4 := by
  sorry

-- Problem 2
theorem problem2 (x : ℚ) : 
  ((1 / 4 * x^4 + 2 * x^3 - 4 * x^2) / (-(2 * x))^2) = (1 / 16 * x^2 + 1 / 2 * x - 1) := by
  sorry

-- Problem 3
theorem problem3 (x y : ℚ) : 
  (2 * x + y + 1) * (2 * x + y - 1) = 4 * x^2 + 4 * x * y + y^2 - 1 := by
  sorry

-- Problem 4
theorem problem4 (x : ℚ) : 
  (2 * x + 3) * (2 * x - 3) - (2 * x - 1)^2 = 4 * x - 10 := by
  sorry

end problem1_problem2_problem3_problem4_l199_199988


namespace speed_of_train_l199_199076

-- Conditions
def length_of_train : ℝ := 100
def time_to_cross : ℝ := 12

-- Question and answer
theorem speed_of_train : length_of_train / time_to_cross = 8.33 := 
by 
  sorry

end speed_of_train_l199_199076


namespace parabola_focus_distance_l199_199610

open Real

noncomputable def parabola (P : ℝ × ℝ) : Prop := (P.2)^2 = 4 * P.1
def line_eq (P : ℝ × ℝ) : Prop := abs (P.1 + 2) = 6

theorem parabola_focus_distance (P : ℝ × ℝ) 
  (hp : parabola P) 
  (hl : line_eq P) : 
  dist P (1 / 4, 0) = 5 :=
sorry

end parabola_focus_distance_l199_199610


namespace distance_to_lake_l199_199884

theorem distance_to_lake 
  {d : ℝ} 
  (h1 : ¬ (d ≥ 8))
  (h2 : ¬ (d ≤ 7))
  (h3 : ¬ (d ≤ 6)) : 
  (7 < d) ∧ (d < 8) :=
by
  sorry

end distance_to_lake_l199_199884


namespace simplify_sqrt_expression_eq_l199_199839

noncomputable def simplify_sqrt_expression (x : ℝ) : ℝ :=
  let sqrt_45x := Real.sqrt (45 * x)
  let sqrt_20x := Real.sqrt (20 * x)
  let sqrt_30x := Real.sqrt (30 * x)
  sqrt_45x * sqrt_20x * sqrt_30x

theorem simplify_sqrt_expression_eq (x : ℝ) :
  simplify_sqrt_expression x = 30 * x * Real.sqrt 30 := by
  sorry

end simplify_sqrt_expression_eq_l199_199839


namespace correct_calculation_l199_199317

theorem correct_calculation :
    (1 + Real.sqrt 2)^2 = 3 + 2 * Real.sqrt 2 :=
sorry

end correct_calculation_l199_199317


namespace parabola_find_c_l199_199948

theorem parabola_find_c (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 2)
  (h2 : (5 : ℝ)^2 + b * 5 + c = 2) : 
  c = 7 := by
  sorry

end parabola_find_c_l199_199948


namespace smallest_white_erasers_l199_199990

def total_erasers (n : ℕ) (pink : ℕ) (orange : ℕ) (purple : ℕ) (white : ℕ) : Prop :=
  pink = n / 5 ∧ orange = n / 6 ∧ purple = 10 ∧ white = n - (pink + orange + purple)

theorem smallest_white_erasers : ∃ n : ℕ, ∃ pink : ℕ, ∃ orange : ℕ, ∃ purple : ℕ, ∃ white : ℕ,
  total_erasers n pink orange purple white ∧ white = 9 := sorry

end smallest_white_erasers_l199_199990


namespace least_possible_value_of_expression_l199_199077

noncomputable def min_expression_value (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023

theorem least_possible_value_of_expression :
  ∃ x : ℝ, min_expression_value x = 2022 :=
by
  sorry

end least_possible_value_of_expression_l199_199077


namespace smallest_cube_ends_in_584_l199_199635

theorem smallest_cube_ends_in_584 (n : ℕ) : n^3 ≡ 584 [MOD 1000] → n = 34 := by
  sorry

end smallest_cube_ends_in_584_l199_199635


namespace positive_b_3b_sq_l199_199808

variable (a b c : ℝ)

theorem positive_b_3b_sq (h1 : 0 < a ∧ a < 0.5) (h2 : -0.5 < b ∧ b < 0) (h3 : 1 < c ∧ c < 3) : b + 3 * b^2 > 0 :=
sorry

end positive_b_3b_sq_l199_199808


namespace probability_at_least_one_l199_199282

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end probability_at_least_one_l199_199282


namespace solution_l199_199223

noncomputable def polynomial_has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - a * x^2 + a * x - 1 = 0

theorem solution (a : ℝ) : polynomial_has_real_root a :=
sorry

end solution_l199_199223


namespace logarithmic_ratio_l199_199590

theorem logarithmic_ratio (m n : ℝ) (h1 : Real.log 2 = m) (h2 : Real.log 3 = n) :
  (Real.log 12) / (Real.log 15) = (2 * m + n) / (1 - m + n) := 
sorry

end logarithmic_ratio_l199_199590


namespace sum_of_fourth_powers_l199_199734

theorem sum_of_fourth_powers (n : ℤ) (h : (n - 2)^2 + n^2 + (n + 2)^2 = 2450) :
  (n - 2)^4 + n^4 + (n + 2)^4 = 1881632 :=
sorry

end sum_of_fourth_powers_l199_199734


namespace range_of_a_I_minimum_value_of_a_II_l199_199263

open Real

def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a_I (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ 0 ≤ a ∧ a ≤ 2 := sorry

theorem minimum_value_of_a_II :
  ∀ a : ℝ, (∀ x : ℝ, f (x - a) a + f (x + a) a ≥ 1 - 2 * a) ↔ a ≥ (1 / 4) :=
sorry

end range_of_a_I_minimum_value_of_a_II_l199_199263


namespace total_students_l199_199793

-- Define the conditions
def rank_from_right := 17
def rank_from_left := 5

-- The proof statement
theorem total_students : rank_from_right + rank_from_left - 1 = 21 := 
by 
  -- Assuming the conditions represented by the definitions
  -- Without loss of generality the proof would be derived from these, but it is skipped
  sorry

end total_students_l199_199793


namespace smallest_m_n_sum_l199_199300

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end smallest_m_n_sum_l199_199300


namespace polynomial_simplification_l199_199059

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 2)*(5*x^12 + 3*x^11 + 2*x^10 - x^9) = 15*x^13 - x^12 - 7*x^10 + 2*x^9 :=
by {
  sorry
}

end polynomial_simplification_l199_199059


namespace simplify_expression_l199_199869

theorem simplify_expression (w : ℝ) : 3 * w + 6 * w - 9 * w + 12 * w - 15 * w + 21 = -3 * w + 21 :=
by
  sorry

end simplify_expression_l199_199869


namespace first_quarter_days_2016_l199_199424

theorem first_quarter_days_2016 : 
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  (jan_days + feb_days + mar_days) = 91 := 
by
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  have h_leap_year : leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) := by sorry
  have h_feb_days : feb_days = 29 := by sorry
  have h_first_quarter : jan_days + feb_days + mar_days = 31 + 29 + 31 := by sorry
  have h_sum : 31 + 29 + 31 = 91 := by norm_num
  exact h_sum

end first_quarter_days_2016_l199_199424


namespace minimum_restoration_time_l199_199023

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end minimum_restoration_time_l199_199023


namespace average_salary_of_associates_l199_199471

theorem average_salary_of_associates 
  (num_managers : ℕ) (num_associates : ℕ)
  (avg_salary_managers : ℝ) (avg_salary_company : ℝ)
  (H_num_managers : num_managers = 15)
  (H_num_associates : num_associates = 75)
  (H_avg_salary_managers : avg_salary_managers = 90000)
  (H_avg_salary_company : avg_salary_company = 40000) :
  ∃ (A : ℝ), (num_managers * avg_salary_managers + num_associates * A) / (num_managers + num_associates) = avg_salary_company ∧ A = 30000 := by
  sorry

end average_salary_of_associates_l199_199471


namespace person_saves_2000_l199_199435

variable (income expenditure savings : ℕ)
variable (h_ratio : income / expenditure = 7 / 6)
variable (h_income : income = 14000)

theorem person_saves_2000 (h_ratio : income / expenditure = 7 / 6) (h_income : income = 14000) :
  savings = income - (6 * (14000 / 7)) :=
by
  sorry

end person_saves_2000_l199_199435


namespace perimeter_of_region_proof_l199_199075

noncomputable def perimeter_of_region (total_area : ℕ) (num_squares : ℕ) (arrangement : String) : ℕ :=
  if total_area = 512 ∧ num_squares = 8 ∧ arrangement = "vertical rectangle" then 160 else 0

theorem perimeter_of_region_proof :
  perimeter_of_region 512 8 "vertical rectangle" = 160 :=
by
  sorry

end perimeter_of_region_proof_l199_199075


namespace solve_inequality_l199_199615

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

theorem solve_inequality : { x : ℝ | |f x| ≥ 1 / 3 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l199_199615


namespace correct_quotient_divide_8_l199_199136

theorem correct_quotient_divide_8 (N : ℕ) (Q : ℕ) 
  (h1 : N = 7 * 12 + 5) 
  (h2 : N / 8 = Q) : 
  Q = 11 := 
by
  sorry

end correct_quotient_divide_8_l199_199136


namespace total_people_in_school_l199_199189

def number_of_girls := 315
def number_of_boys := 309
def number_of_teachers := 772
def total_number_of_people := number_of_girls + number_of_boys + number_of_teachers

theorem total_people_in_school :
  total_number_of_people = 1396 :=
by sorry

end total_people_in_school_l199_199189


namespace at_least_one_gt_one_l199_199726

variable (a b : ℝ)

theorem at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l199_199726


namespace mirror_side_length_l199_199727

theorem mirror_side_length (width length : ℝ) (area_wall : ℝ) (area_mirror : ℝ) (side_length : ℝ) 
  (h1 : width = 28) 
  (h2 : length = 31.5) 
  (h3 : area_wall = width * length)
  (h4 : area_mirror = area_wall / 2) 
  (h5 : area_mirror = side_length ^ 2) : 
  side_length = 21 := 
by 
  sorry

end mirror_side_length_l199_199727


namespace expected_pairs_socks_l199_199466

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l199_199466


namespace sum_of_numbers_l199_199688

theorem sum_of_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 222) (h2 : a * b + b * c + c * a = 131) : a + b + c = 22 :=
by
  sorry

end sum_of_numbers_l199_199688


namespace cars_more_than_trucks_l199_199089

theorem cars_more_than_trucks (total_vehicles : ℕ) (trucks : ℕ) (h : total_vehicles = 69) (h' : trucks = 21) :
  (total_vehicles - trucks) - trucks = 27 :=
by
  sorry

end cars_more_than_trucks_l199_199089


namespace winning_candidate_votes_l199_199804

theorem winning_candidate_votes  (V W : ℝ) (hW : W = 0.5666666666666664 * V) (hV : V = W + 7636 + 11628) : 
  W = 25216 := 
by 
  sorry

end winning_candidate_votes_l199_199804


namespace equalized_distance_l199_199124

noncomputable def wall_width : ℝ := 320 -- wall width in centimeters
noncomputable def poster_count : ℕ := 6 -- number of posters
noncomputable def poster_width : ℝ := 30 -- width of each poster in centimeters
noncomputable def equal_distance : ℝ := 20 -- equal distance in centimeters to be proven

theorem equalized_distance :
  let total_posters_width := poster_count * poster_width
  let remaining_space := wall_width - total_posters_width
  let number_of_spaces := poster_count + 1
  remaining_space / number_of_spaces = equal_distance :=
by {
  sorry
}

end equalized_distance_l199_199124


namespace pencils_undefined_l199_199621

-- Definitions for the conditions given in the problem
def initial_crayons : Nat := 41
def added_crayons : Nat := 12
def total_crayons : Nat := 53

-- Theorem stating the problem's required proof
theorem pencils_undefined (initial_crayons : Nat) (added_crayons : Nat) (total_crayons : Nat) : Prop :=
  initial_crayons = 41 ∧ added_crayons = 12 ∧ total_crayons = 53 → 
  ∃ (pencils : Nat), true
-- Since the number of pencils is unknown and no direct information is given, we represent it as an existential statement that pencils exist in some quantity, but we cannot determine their exact number based on given information.

end pencils_undefined_l199_199621


namespace marks_in_social_studies_l199_199568

def shekar_marks : ℕ := 82

theorem marks_in_social_studies 
  (marks_math : ℕ := 76)
  (marks_science : ℕ := 65)
  (marks_english : ℕ := 67)
  (marks_biology : ℕ := 55)
  (average_marks : ℕ := 69)
  (num_subjects : ℕ := 5) :
  marks_math + marks_science + marks_english + marks_biology + shekar_marks = average_marks * num_subjects :=
by
  sorry

end marks_in_social_studies_l199_199568


namespace kangaroo_jump_is_8_5_feet_longer_l199_199446

noncomputable def camel_step_length (total_distance : ℝ) (num_steps : ℕ) : ℝ := total_distance / num_steps
noncomputable def kangaroo_jump_length (total_distance : ℝ) (num_jumps : ℕ) : ℝ := total_distance / num_jumps
noncomputable def length_difference (jump_length step_length : ℝ) : ℝ := jump_length - step_length

theorem kangaroo_jump_is_8_5_feet_longer :
  let total_distance := 7920
  let num_gaps := 50
  let camel_steps_per_gap := 56
  let kangaroo_jumps_per_gap := 14
  let num_camel_steps := num_gaps * camel_steps_per_gap
  let num_kangaroo_jumps := num_gaps * kangaroo_jumps_per_gap
  let camel_step := camel_step_length total_distance num_camel_steps
  let kangaroo_jump := kangaroo_jump_length total_distance num_kangaroo_jumps
  length_difference kangaroo_jump camel_step = 8.5 := sorry

end kangaroo_jump_is_8_5_feet_longer_l199_199446


namespace tan_five_pi_over_four_l199_199892

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l199_199892


namespace joanne_first_hour_coins_l199_199570

theorem joanne_first_hour_coins 
  (X : ℕ)
  (H1 : 70 = 35 + 35)
  (H2 : 120 = X + 70 + 35)
  (H3 : 35 = 50 - 15) : 
  X = 15 :=
sorry

end joanne_first_hour_coins_l199_199570


namespace distance_from_P_to_AD_is_correct_l199_199582

noncomputable def P_distance_to_AD : ℝ :=
  let A : ℝ × ℝ := (0, 6)
  let D : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 0)
  let M : ℝ × ℝ := (3, 0)
  let radius1 : ℝ := 5
  let radius2 : ℝ := 6
  let circle1_eq := fun (x y : ℝ) => (x - 3)^2 + y^2 = 25
  let circle2_eq := fun (x y : ℝ) => x^2 + (y - 6)^2 = 36
  let P := (24/5, 18/5)
  let AD := fun x y : ℝ => x = 0
  abs ((P.fst : ℝ) - 0)

theorem distance_from_P_to_AD_is_correct :
  P_distance_to_AD = 24 / 5 := by
  sorry

end distance_from_P_to_AD_is_correct_l199_199582


namespace find_x_l199_199443

theorem find_x :
  ∃ x : ℝ, 12.1212 + x - 9.1103 = 20.011399999999995 ∧ x = 18.000499999999995 :=
sorry

end find_x_l199_199443


namespace product_primes_less_than_20_l199_199496

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l199_199496


namespace sqrt_of_9_eq_3_l199_199651

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_eq_3_l199_199651


namespace mixed_number_solution_l199_199832

noncomputable def mixed_number_problem : Prop :=
  let a := 4 + 2 / 7
  let b := 5 + 1 / 2
  let c := 3 + 1 / 3
  let d := 2 + 1 / 6
  (a * b) - (c + d) = 18 + 1 / 14

theorem mixed_number_solution : mixed_number_problem := by 
  sorry

end mixed_number_solution_l199_199832


namespace find_x_l199_199985

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_x (x : ℝ) (hx : x > 0) :
  distance (1, 3) (x, -4) = 15 → x = 1 + Real.sqrt 176 :=
by
  sorry

end find_x_l199_199985


namespace anya_possible_wins_l199_199516

-- Define the total rounds played
def total_rounds := 25

-- Define Anya's choices
def anya_rock := 12
def anya_scissors := 6
def anya_paper := 7

-- Define Borya's choices
def borya_rock := 13
def borya_scissors := 9
def borya_paper := 3

-- Define the relationships in rock-paper-scissors game
def rock_beats_scissors := true
def scissors_beat_paper := true
def paper_beats_rock := true

-- Define no draws condition
def no_draws := total_rounds = anya_rock + anya_scissors + anya_paper ∧ total_rounds = borya_rock + borya_scissors + borya_paper

-- Proof problem statement
theorem anya_possible_wins : anya_rock + anya_scissors + anya_paper = total_rounds ∧
                             borya_rock + borya_scissors + borya_paper = total_rounds ∧
                             rock_beats_scissors ∧ scissors_beat_paper ∧ paper_beats_rock ∧
                             no_draws →
                             (9 + 3 + 7 = 19) := by
  sorry

end anya_possible_wins_l199_199516


namespace num_men_in_first_group_l199_199930

variable {x m w : ℝ}

theorem num_men_in_first_group (h1 : x * m + 8 * w = 6 * m + 2 * w)
  (h2 : 2 * m + 3 * w = 0.5 * (x * m + 8 * w)) : 
  x = 3 :=
sorry

end num_men_in_first_group_l199_199930


namespace fraction_of_eggs_hatched_l199_199055

variable (x : ℚ)
variable (survived_first_month_fraction : ℚ := 3/4)
variable (survived_first_year_fraction : ℚ := 2/5)
variable (geese_survived : ℕ := 100)
variable (total_eggs : ℕ := 500)

theorem fraction_of_eggs_hatched :
  (x * survived_first_month_fraction * survived_first_year_fraction * total_eggs : ℚ) = geese_survived → x = 2/3 :=
by 
  intro h
  sorry

end fraction_of_eggs_hatched_l199_199055


namespace circle_regions_l199_199500

def regions_divided_by_chords (n : ℕ) : ℕ :=
  (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24

theorem circle_regions (n : ℕ) : 
  regions_divided_by_chords n = (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24 := 
  by 
  sorry

end circle_regions_l199_199500


namespace contrapositive_false_l199_199457

theorem contrapositive_false : ¬ (∀ x : ℝ, x^2 = 1 → x = 1) → ∀ x : ℝ, x^2 = 1 → x ≠ 1 :=
by
  sorry

end contrapositive_false_l199_199457


namespace equilateral_triangle_side_length_l199_199547

theorem equilateral_triangle_side_length (perimeter : ℝ) (h : perimeter = 2) : abs (perimeter / 3 - 0.67) < 0.01 :=
by
  -- The proof will go here.
  sorry

end equilateral_triangle_side_length_l199_199547


namespace actual_average_height_correct_l199_199694

noncomputable def actual_average_height (n : ℕ) (average_height : ℝ) (wrong_height : ℝ) (actual_height : ℝ) : ℝ :=
  let total_height := average_height * n
  let difference := wrong_height - actual_height
  let correct_total_height := total_height - difference
  correct_total_height / n

theorem actual_average_height_correct :
  actual_average_height 35 184 166 106 = 182.29 :=
by
  sorry

end actual_average_height_correct_l199_199694


namespace millimeters_of_78_74_inches_l199_199458

noncomputable def inchesToMillimeters (inches : ℝ) : ℝ :=
  inches * 25.4

theorem millimeters_of_78_74_inches :
  round (inchesToMillimeters 78.74) = 2000 :=
by
  -- This theorem should assert that converting 78.74 inches to millimeters and rounding to the nearest millimeter equals 2000
  sorry

end millimeters_of_78_74_inches_l199_199458


namespace average_age_of_9_students_l199_199984

theorem average_age_of_9_students
  (avg_20_students : ℝ)
  (n_20_students : ℕ)
  (avg_10_students : ℝ)
  (n_10_students : ℕ)
  (age_20th_student : ℝ)
  (total_age_20_students : ℝ := avg_20_students * n_20_students)
  (total_age_10_students : ℝ := avg_10_students * n_10_students)
  (total_age_9_students : ℝ := total_age_20_students - total_age_10_students - age_20th_student)
  (n_9_students : ℕ)
  (expected_avg_9_students : ℝ := total_age_9_students / n_9_students)
  (H1 : avg_20_students = 20)
  (H2 : n_20_students = 20)
  (H3 : avg_10_students = 24)
  (H4 : n_10_students = 10)
  (H5 : age_20th_student = 61)
  (H6 : n_9_students = 9) :
  expected_avg_9_students = 11 :=
sorry

end average_age_of_9_students_l199_199984


namespace function_equivalence_l199_199577

theorem function_equivalence (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 6 * x - 1) : ∀ x : ℝ, f x = 3 * x - 1 :=
by
  sorry

end function_equivalence_l199_199577


namespace thabo_book_ratio_l199_199565

theorem thabo_book_ratio :
  ∃ (P_f P_nf H_nf : ℕ), H_nf = 35 ∧ P_nf = H_nf + 20 ∧ P_f + P_nf + H_nf = 200 ∧ P_f / P_nf = 2 :=
by
  sorry

end thabo_book_ratio_l199_199565


namespace slope_of_line_l199_199484

theorem slope_of_line (x1 x2 y1 y2 : ℝ) (h1 : 1 = (x1 + x2) / 2) (h2 : 1 = (y1 + y2) / 2) 
                      (h3 : (x1^2 / 36) + (y1^2 / 9) = 1) (h4 : (x2^2 / 36) + (y2^2 / 9) = 1) :
  (y2 - y1) / (x2 - x1) = -1 / 4 :=
by
  sorry

end slope_of_line_l199_199484


namespace vector_dot_product_l199_199093

variables (a b : ℝ × ℝ)
variables (ha : a = (1, -1)) (hb : b = (-1, 2))

theorem vector_dot_product : 
  ((2 • a + b) • a) = -1 :=
by
  -- This is where the proof would go
  sorry

end vector_dot_product_l199_199093


namespace pears_thrown_away_on_first_day_l199_199347

theorem pears_thrown_away_on_first_day (x : ℝ) (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.8 * P = P * 0.8)
  (total_thrown_percentage : (x / 100) * 0.2 * P + 0.2 * (1 - x / 100) * 0.2 * P = 0.12 * P ) : 
  x = 50 :=
by
  sorry

end pears_thrown_away_on_first_day_l199_199347


namespace expression_evaluation_l199_199683

noncomputable def x := Real.sqrt 5 + 1
noncomputable def y := Real.sqrt 5 - 1

theorem expression_evaluation : 
  ( ( (5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2) ) / (1 / (x^2 * y - x * y^2)) ) = 12 := 
by 
  -- Provide a proof here
  sorry

end expression_evaluation_l199_199683


namespace probability_two_or_fewer_distinct_digits_l199_199186

def digits : Set ℕ := {1, 2, 3}

def total_3_digit_numbers : ℕ := 27

def distinct_3_digit_numbers : ℕ := 6

def at_most_two_distinct_numbers : ℕ := total_3_digit_numbers - distinct_3_digit_numbers

theorem probability_two_or_fewer_distinct_digits :
  (at_most_two_distinct_numbers : ℚ) / total_3_digit_numbers = 7 / 9 := by
  sorry

end probability_two_or_fewer_distinct_digits_l199_199186


namespace exactly_one_negative_x_or_y_l199_199275

theorem exactly_one_negative_x_or_y
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (x1_ne_zero : x1 ≠ 0) (x2_ne_zero : x2 ≠ 0) (x3_ne_zero : x3 ≠ 0)
  (y1_ne_zero : y1 ≠ 0) (y2_ne_zero : y2 ≠ 0) (y3_ne_zero : y3 ≠ 0)
  (h1 : x1 * x2 * x3 = - y1 * y2 * y3)
  (h2 : x1^2 + x2^2 + x3^2 = y1^2 + y2^2 + y3^2)
  (h3 : x1 + y1 + x2 + y2 ≥ x3 + y3 ∧ x2 + y2 + x3 + y3 ≥ x1 + y1 ∧ x3 + y3 + x1 + y1 ≥ x2 + y2)
  (h4 : (x1 + y1)^2 + (x2 + y2)^2 ≥ (x3 + y3)^2 ∧ (x2 + y2)^2 + (x3 + y3)^2 ≥ (x1 + y1)^2 ∧ (x3 + y3)^2 + (x1 + y1)^2 ≥ (x2 + y2)^2) :
  ∃! (a : ℝ), (a = x1 ∨ a = x2 ∨ a = x3 ∨ a = y1 ∨ a = y2 ∨ a = y3) ∧ a < 0 :=
sorry

end exactly_one_negative_x_or_y_l199_199275


namespace solve_quadratic_l199_199619

theorem solve_quadratic (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
sorry

end solve_quadratic_l199_199619


namespace branches_on_one_stem_l199_199191

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l199_199191


namespace BD_length_l199_199302

theorem BD_length
  (A B C D : Type)
  (dist_AC : ℝ := 10)
  (dist_BC : ℝ := 10)
  (dist_AD : ℝ := 12)
  (dist_CD : ℝ := 5) : (BD : ℝ) = 95 / 12 :=
by
  sorry

end BD_length_l199_199302


namespace nancy_pots_created_on_Wednesday_l199_199986

def nancy_pots_conditions (pots_Monday pots_Tuesday total_pots : ℕ) : Prop :=
  pots_Monday = 12 ∧ pots_Tuesday = 2 * pots_Monday ∧ total_pots = 50

theorem nancy_pots_created_on_Wednesday :
  ∀ pots_Monday pots_Tuesday total_pots,
  nancy_pots_conditions pots_Monday pots_Tuesday total_pots →
  (total_pots - (pots_Monday + pots_Tuesday) = 14) := by
  intros pots_Monday pots_Tuesday total_pots h
  -- proof would go here
  sorry

end nancy_pots_created_on_Wednesday_l199_199986


namespace circle_equation_l199_199780

/-
  Prove that the standard equation for the circle passing through points
  A(-6, 0), B(0, 2), and the origin O(0, 0) is (x+3)^2 + (y-1)^2 = 10.
-/
theorem circle_equation :
  ∃ (x y : ℝ), x = -6 ∨ x = 0 ∨ x = 0 ∧ y = 0 ∨ y = 2 ∨ y = 0 → (∀ P : ℝ × ℝ, P = (-6, 0) ∨ P = (0, 2) ∨ P = (0, 0) → (P.1 + 3)^2 + (P.2 - 1)^2 = 10) := 
sorry

end circle_equation_l199_199780


namespace hexagon_perimeter_l199_199473

theorem hexagon_perimeter (s : ℝ) (h_area : s ^ 2 * (3 * Real.sqrt 3 / 2) = 54 * Real.sqrt 3) :
  6 * s = 36 :=
by
  sorry

end hexagon_perimeter_l199_199473


namespace original_stone_counted_as_99_l199_199391

theorem original_stone_counted_as_99 :
  (99 % 22) = 11 :=
by sorry

end original_stone_counted_as_99_l199_199391


namespace cube_paint_problem_l199_199487

theorem cube_paint_problem : 
  ∀ (n : ℕ),
  n = 6 →
  (∃ k : ℕ, 216 = k^3 ∧ k = n) →
  ∀ (faces inner_faces total_cubelets : ℕ),
  faces = 6 →
  inner_faces = 4 →
  total_cubelets = faces * (inner_faces * inner_faces) →
  total_cubelets = 96 :=
by 
  intros n hn hc faces hfaces inner_faces hinner_faces total_cubelets htotal_cubelets
  sorry

end cube_paint_problem_l199_199487


namespace ratio_ab_bd_l199_199305

-- Definitions based on the given conditions
def ab : ℝ := 4
def bc : ℝ := 8
def cd : ℝ := 5
def bd : ℝ := bc + cd

-- Theorem statement
theorem ratio_ab_bd :
  ((ab / bd) = (4 / 13)) :=
by
  -- Proof goes here
  sorry

end ratio_ab_bd_l199_199305


namespace sum_mod_eleven_l199_199001

variable (x y z : ℕ)

theorem sum_mod_eleven (h1 : (x * y * z) % 11 = 3)
                       (h2 : (7 * z) % 11 = 4)
                       (h3 : (9 * y) % 11 = (5 + y) % 11) :
                       (x + y + z) % 11 = 5 :=
sorry

end sum_mod_eleven_l199_199001


namespace correct_product_of_a_and_b_l199_199843

-- Define reversal function for two-digit numbers
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- State the main problem
theorem correct_product_of_a_and_b (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 0 < b) 
  (h : (reverse_digits a) * b = 284) : a * b = 68 :=
sorry

end correct_product_of_a_and_b_l199_199843


namespace paths_for_content_l199_199689

def grid := [
  [none, none, none, none, none, none, some 'C', none, none, none, none, none, none, none],
  [none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none, none],
  [none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none, none],
  [none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none, none],
  [none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'T', some 'N', some 'O', some 'C', none, none, none],
  [none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none, none],
  [some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C']
]

def spelling_paths : Nat :=
  -- Skipping the actual calculation and providing the given total for now
  127

theorem paths_for_content : spelling_paths = 127 := sorry

end paths_for_content_l199_199689


namespace train_stops_time_l199_199888

theorem train_stops_time 
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 60)
  (h2 : speed_including_stoppages = 40) : 
  ∃ (stoppage_time : ℝ), stoppage_time = 20 := 
by
  sorry

end train_stops_time_l199_199888


namespace average_length_tapes_l199_199296

def lengths (l1 l2 l3 l4 l5 : ℝ) : Prop :=
  l1 = 35 ∧ l2 = 29 ∧ l3 = 35.5 ∧ l4 = 36 ∧ l5 = 30.5

theorem average_length_tapes
  (l1 l2 l3 l4 l5 : ℝ)
  (h : lengths l1 l2 l3 l4 l5) :
  (l1 + l2 + l3 + l4 + l5) / 5 = 33.2 := 
by
  sorry

end average_length_tapes_l199_199296


namespace gcd_lcm_product_l199_199757

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l199_199757


namespace number_of_baskets_l199_199262

-- Define the conditions
def total_peaches : Nat := 10
def red_peaches_per_basket : Nat := 4
def green_peaches_per_basket : Nat := 6
def peaches_per_basket : Nat := red_peaches_per_basket + green_peaches_per_basket

-- The goal is to prove that the number of baskets is 1 given the conditions

theorem number_of_baskets (h1 : total_peaches = 10)
                           (h2 : peaches_per_basket = red_peaches_per_basket + green_peaches_per_basket)
                           (h3 : red_peaches_per_basket = 4)
                           (h4 : green_peaches_per_basket = 6) : 
                           total_peaches / peaches_per_basket = 1 := by
                            sorry

end number_of_baskets_l199_199262


namespace parallel_lines_have_equal_slopes_l199_199562

theorem parallel_lines_have_equal_slopes (a : ℝ) :
  (∃ a : ℝ, (∀ y : ℝ, 2 * a * y - 1 = 0) ∧ (∃ x y : ℝ, (3 * a - 1) * x + y - 1 = 0) 
  → (∃ a : ℝ, (1 / (2 * a)) = - (3 * a - 1))) 
→ a = 1/2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l199_199562


namespace average_speed_of_car_l199_199017

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end average_speed_of_car_l199_199017


namespace pencils_in_drawer_after_operations_l199_199143

def initial_pencils : ℝ := 2
def pencils_added : ℝ := 3.5
def pencils_removed : ℝ := 1.2

theorem pencils_in_drawer_after_operations : ⌊initial_pencils + pencils_added - pencils_removed⌋ = 4 := by
  sorry

end pencils_in_drawer_after_operations_l199_199143


namespace find_two_digit_number_l199_199637

theorem find_two_digit_number
  (X : ℕ)
  (h1 : 57 + (10 * X + 6) = 123)
  (h2 : two_digit_number = 10 * X + 9) :
  two_digit_number = 69 :=
by
  sorry

end find_two_digit_number_l199_199637


namespace hitting_probability_l199_199725

theorem hitting_probability (A_hit B_hit : ℚ) (hA : A_hit = 4/5) (hB : B_hit = 5/6) :
  1 - ((1 - A_hit) * (1 - B_hit)) = 29/30 :=
by 
  sorry

end hitting_probability_l199_199725


namespace largest_integer_not_greater_than_expr_l199_199667

theorem largest_integer_not_greater_than_expr (x : ℝ) (hx : 20 * Real.sin x = 22 * Real.cos x) :
    ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := 
sorry

end largest_integer_not_greater_than_expr_l199_199667


namespace find_n_l199_199603

theorem find_n (n : ℕ) : 5 ^ 29 * 4 ^ 15 = 2 * 10 ^ n → n = 29 :=
by
  sorry

end find_n_l199_199603


namespace order_DABC_l199_199824

-- Definitions of the variables given in the problem
def A : ℕ := 77^7
def B : ℕ := 7^77
def C : ℕ := 7^7^7
def D : ℕ := Nat.factorial 7

-- The theorem stating the required ascending order
theorem order_DABC : D < A ∧ A < B ∧ B < C :=
by sorry

end order_DABC_l199_199824


namespace problem_1_problem_2_l199_199046

noncomputable def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

theorem problem_1 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 :=
by
  sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + 2 * b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
by
  sorry

end problem_1_problem_2_l199_199046


namespace moles_of_NaHCO3_combined_l199_199167

theorem moles_of_NaHCO3_combined (n_HNO3 n_NaHCO3 : ℕ) (mass_H2O : ℝ) : 
  n_HNO3 = 2 ∧ mass_H2O = 36 ∧ n_HNO3 = n_NaHCO3 → n_NaHCO3 = 2 := by
  sorry

end moles_of_NaHCO3_combined_l199_199167


namespace find_other_leg_length_l199_199993

theorem find_other_leg_length (a b c : ℝ) (h1 : a = 15) (h2 : b = 5 * Real.sqrt 3) (h3 : c = 2 * (5 * Real.sqrt 3)) (h4 : a^2 + b^2 = c^2)
  (angle_A : ℝ) (h5 : angle_A = Real.pi / 3) (h6 : angle_A ≠ Real.pi / 2) :
  b = 5 * Real.sqrt 3 :=
by
  sorry

end find_other_leg_length_l199_199993


namespace number_of_cows_on_boat_l199_199953

-- Definitions based on conditions
def number_of_sheep := 20
def number_of_dogs := 14
def sheep_drowned := 3
def cows_drowned := 2 * sheep_drowned  -- Twice as many cows drowned as did sheep.
def dogs_made_it_shore := number_of_dogs  -- All dogs made it to shore.
def total_animals_shore := 35
def total_sheep_shore := number_of_sheep - sheep_drowned
def total_sheep_cows_shore := total_animals_shore - dogs_made_it_shore
def cows_made_it_shore := total_sheep_cows_shore - total_sheep_shore

-- Theorem stating the problem
theorem number_of_cows_on_boat : 
  (cows_made_it_shore + cows_drowned) = 10 := by
  sorry

end number_of_cows_on_boat_l199_199953


namespace min_sum_intercepts_l199_199852

theorem min_sum_intercepts (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (1 : ℝ) * a + (1 : ℝ) * b = a * b) : a + b = 4 :=
by
  sorry

end min_sum_intercepts_l199_199852


namespace sum_remainders_mod_15_l199_199292

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end sum_remainders_mod_15_l199_199292


namespace window_design_ratio_l199_199881

theorem window_design_ratio (AB AD r : ℝ)
  (h1 : AB = 40)
  (h2 : AD / AB = 4 / 3)
  (h3 : r = AB / 2) :
  ((AD - AB) * AB) / (π * r^2 / 2) = 8 / (3 * π) :=
by
  sorry

end window_design_ratio_l199_199881


namespace colorable_graph_l199_199060

variable (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]

/-- Each city has at least one road leading out of it -/
def has_one_road (v : V) : Prop := ∃ w : V, E v w

/-- No city is connected by roads to all other cities -/
def not_connected_to_all (v : V) : Prop := ¬ ∀ w : V, E v w ↔ w ≠ v

/-- A set of cities D is dominating if every city not in D is connected by a road to at least one city in D -/
def is_dominating_set (D : Finset V) : Prop :=
  ∀ v : V, v ∉ D → ∃ d ∈ D, E v d

noncomputable def dominating_set_min_card (k : ℕ) : Prop :=
  ∀ D : Finset V, is_dominating_set V E D → D.card ≥ k

/-- Prove that the graph can be colored using 2001 - k colors such that no two adjacent vertices share the same color -/
theorem colorable_graph (k : ℕ) (hk : dominating_set_min_card V E k) :
    ∃ (colors : V → Fin (2001 - k)), ∀ v w : V, E v w → colors v ≠ colors w := 
by 
  sorry

end colorable_graph_l199_199060


namespace arithmetic_expression_value_l199_199505

theorem arithmetic_expression_value :
  15 * 36 + 15 * 3^3 = 945 :=
by
  sorry

end arithmetic_expression_value_l199_199505


namespace time_shortened_by_opening_both_pipes_l199_199385

theorem time_shortened_by_opening_both_pipes 
  (a b p : ℝ) 
  (hp : a * p > 0) -- To ensure p > 0 and reservoir volume is positive
  (h1 : p = (a * p) / a) -- Given that pipe A alone takes p hours
  : p - (a * p) / (a + b) = (b * p) / (a + b) := 
sorry

end time_shortened_by_opening_both_pipes_l199_199385


namespace segments_to_start_l199_199477

-- Define the problem statement conditions in Lean 4
def concentric_circles : Prop := sorry -- Placeholder, as geometry involving tangents and arcs isn't directly supported

def chord_tangent_small_circle (AB : Prop) : Prop := sorry -- Placeholder, detailing tangency

def angle_ABC_eq_60 (A B C : Prop) : Prop := sorry -- Placeholder, situating angles in terms of Lean formalism

-- Proof statement
theorem segments_to_start (A B C : Prop) :
  concentric_circles →
  chord_tangent_small_circle (A ↔ B) →
  chord_tangent_small_circle (B ↔ C) →
  angle_ABC_eq_60 A B C →
  ∃ n : ℕ, n = 3 :=
sorry

end segments_to_start_l199_199477


namespace dragons_total_games_played_l199_199098

theorem dragons_total_games_played (y x : ℕ)
  (h1 : x = 55 * y / 100)
  (h2 : x + 8 = 60 * (y + 12) / 100) :
  y + 12 = 28 :=
by
  sorry

end dragons_total_games_played_l199_199098


namespace sum_of_fractions_l199_199241

theorem sum_of_fractions :
  (3 / 15) + (6 / 15) + (9 / 15) + (12 / 15) + (15 / 15) + 
  (18 / 15) + (21 / 15) + (24 / 15) + (27 / 15) + (75 / 15) = 14 :=
by
  sorry

end sum_of_fractions_l199_199241


namespace gcd_65536_49152_l199_199886

theorem gcd_65536_49152 : Nat.gcd 65536 49152 = 16384 :=
by
  sorry

end gcd_65536_49152_l199_199886


namespace volume_of_cube_in_pyramid_l199_199419

theorem volume_of_cube_in_pyramid :
  (∃ (s : ℝ), 
    ( ∀ (b h l : ℝ),
      b = 2 ∧ 
      h = 3 ∧ 
      l = 2 * Real.sqrt 2 →
      s = 4 * Real.sqrt 2 - 3 ∧ 
      ((4 * Real.sqrt 2 - 3) ^ 3 = (4 * Real.sqrt 2 - 3) ^ 3))) :=
sorry

end volume_of_cube_in_pyramid_l199_199419


namespace initial_goldfish_correct_l199_199400

-- Define the constants related to the conditions
def weekly_die := 5
def weekly_purchase := 3
def final_goldfish := 4
def weeks := 7

-- Define the initial number of goldfish that we need to prove
def initial_goldfish := 18

-- The proof statement: initial_goldfish - weekly_change * weeks = final_goldfish
theorem initial_goldfish_correct (G : ℕ)
  (h : G - weeks * (weekly_purchase - weekly_die) = final_goldfish) :
  G = initial_goldfish := by
  sorry

end initial_goldfish_correct_l199_199400


namespace total_pencils_l199_199729

def initial_pencils : ℕ := 9
def additional_pencils : ℕ := 56

theorem total_pencils : initial_pencils + additional_pencils = 65 :=
by
  -- proof steps are not required, so we use sorry
  sorry

end total_pencils_l199_199729


namespace savings_by_buying_gallon_l199_199253

def gallon_to_ounces : ℕ := 128
def bottle_volume_ounces : ℕ := 16
def cost_gallon : ℕ := 8
def cost_bottle : ℕ := 3

theorem savings_by_buying_gallon :
  (cost_bottle * (gallon_to_ounces / bottle_volume_ounces)) - cost_gallon = 16 := 
by
  sorry

end savings_by_buying_gallon_l199_199253


namespace minimum_slit_length_l199_199931

theorem minimum_slit_length (circumference : ℝ) (speed_ratio : ℝ) (reliability : ℝ) :
  circumference = 1 → speed_ratio = 2 → (∀ (s : ℝ), (s < 2/3) → (¬ reliable)) → reliability =
    2 / 3 :=
by
  intros hcirc hspeed hrel
  have s := (2 : ℝ) / 3
  sorry

end minimum_slit_length_l199_199931


namespace bounded_sequence_iff_l199_199739

theorem bounded_sequence_iff (x : ℕ → ℝ) (h : ∀ n, x (n + 1) = (n^2 + 1) * x n ^ 2 / (x n ^ 3 + n^2)) :
  (∃ C, ∀ n, x n < C) ↔ (0 < x 0 ∧ x 0 ≤ (Real.sqrt 5 - 1) / 2) ∨ x 0 ≥ 1 := sorry

end bounded_sequence_iff_l199_199739


namespace vector_calc_l199_199639

def vec1 : ℝ × ℝ := (5, -8)
def vec2 : ℝ × ℝ := (2, 6)
def vec3 : ℝ × ℝ := (-1, 4)
def scalar : ℝ := 5

theorem vector_calc :
  (vec1.1 - scalar * vec2.1 + vec3.1, vec1.2 - scalar * vec2.2 + vec3.2) = (-6, -34) :=
sorry

end vector_calc_l199_199639


namespace number_of_grey_birds_l199_199967

variable (G : ℕ)

def grey_birds_condition1 := G + 6
def grey_birds_condition2 := G / 2

theorem number_of_grey_birds
  (H1 : G + 6 + G / 2 = 66) :
  G = 40 :=
by
  sorry

end number_of_grey_birds_l199_199967


namespace marble_theorem_l199_199720

noncomputable def marble_problem (M : ℝ) : Prop :=
  let M_Pedro : ℝ := 0.7 * M
  let M_Ebony : ℝ := 0.85 * M_Pedro
  let M_Jimmy : ℝ := 0.7 * M_Ebony
  (M_Jimmy / M) * 100 = 41.65

theorem marble_theorem (M : ℝ) : marble_problem M := 
by
  sorry

end marble_theorem_l199_199720


namespace total_games_in_season_l199_199088

theorem total_games_in_season :
  let num_teams := 100
  let num_sub_leagues := 5
  let teams_per_league := 20
  let games_per_pair := 6
  let teams_advancing := 4
  let playoff_teams := num_sub_leagues * teams_advancing
  let sub_league_games := (teams_per_league * (teams_per_league - 1) / 2) * games_per_pair
  let total_sub_league_games := sub_league_games * num_sub_leagues
  let playoff_games := (playoff_teams * (playoff_teams - 1)) / 2 
  let total_games := total_sub_league_games + playoff_games
  total_games = 5890 :=
by
  sorry

end total_games_in_season_l199_199088


namespace minimum_perimeter_l199_199281

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end minimum_perimeter_l199_199281


namespace age_difference_36_l199_199591

noncomputable def jack_age (a b : ℕ) : ℕ := 10 * a + b
noncomputable def bill_age (b a : ℕ) : ℕ := 10 * b + a

theorem age_difference_36 (a b : ℕ) (h : 10 * a + b + 3 = 3 * (10 * b + a + 3)) :
  jack_age a b - bill_age b a = 36 :=
by sorry

end age_difference_36_l199_199591


namespace three_inequalities_true_l199_199940

variables {x y a b : ℝ}
-- Declare the conditions as hypotheses
axiom h₁ : 0 < x
axiom h₂ : 0 < y
axiom h₃ : 0 < a
axiom h₄ : 0 < b
axiom hx : x^2 < a^2
axiom hy : y^2 < b^2

theorem three_inequalities_true : 
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  (x^2 / y^2 < a^2 / b^2) :=
sorry

end three_inequalities_true_l199_199940


namespace new_average_weight_is_27_3_l199_199257

-- Define the given conditions as variables/constants in Lean
noncomputable def original_students : ℕ := 29
noncomputable def original_average_weight : ℝ := 28
noncomputable def new_student_weight : ℝ := 7

-- The total weight of the original students
noncomputable def original_total_weight : ℝ := original_students * original_average_weight
-- The new total number of students
noncomputable def new_total_students : ℕ := original_students + 1
-- The new total weight after new student is added
noncomputable def new_total_weight : ℝ := original_total_weight + new_student_weight

-- The theorem to prove that the new average weight is 27.3 kg
theorem new_average_weight_is_27_3 : (new_total_weight / new_total_students) = 27.3 := 
by
  sorry -- The proof will be provided here

end new_average_weight_is_27_3_l199_199257


namespace intersection_M_N_l199_199286

noncomputable def M : Set ℝ := { x | x^2 - x ≤ 0 }
noncomputable def N : Set ℝ := { x | 1 - abs x > 0 }
noncomputable def intersection : Set ℝ := { x | x ≥ 0 ∧ x < 1 }

theorem intersection_M_N : M ∩ N = intersection :=
by
  sorry

end intersection_M_N_l199_199286


namespace sufficient_but_not_necessary_condition_l199_199395

theorem sufficient_but_not_necessary_condition (a1 d : ℝ) : 
  (2 * a1 + 11 * d > 0) → (2 * a1 + 11 * d ≥ 0) :=
by
  intro h
  apply le_of_lt
  exact h

end sufficient_but_not_necessary_condition_l199_199395


namespace calculate_expr_at_3_l199_199157

-- Definition of the expression
def expr (x : ℕ) : ℕ := (x + x * x^(x^2)) * 3

-- The proof statement
theorem calculate_expr_at_3 : expr 3 = 177156 := 
by
  sorry

end calculate_expr_at_3_l199_199157


namespace circles_tangent_l199_199404

/--
Two equal circles each with a radius of 5 are externally tangent to each other and both are internally tangent to a larger circle with a radius of 13. 
Let the points of tangency be A and B. Let AB = m/n where m and n are positive integers and gcd(m, n) = 1. 
We need to prove that m + n = 69.
-/
theorem circles_tangent (r1 r2 r3 : ℝ) (tangent_external : ℝ) (tangent_internal : ℝ) (AB : ℝ) (m n : ℕ) 
  (hmn_coprime : Nat.gcd m n = 1) (hr1 : r1 = 5) (hr2 : r2 = 5) (hr3 : r3 = 13) 
  (ht_external : tangent_external = r1 + r2) (ht_internal : tangent_internal = r3 - r1) 
  (hAB : AB = (130 / 8)): m + n = 69 :=
by
  sorry

end circles_tangent_l199_199404


namespace DF_length_l199_199806

-- Definitions for the given problem.
variable (AB DC EB DE : ℝ)
variable (parallelogram_ABCD : Prop)
variable (DE_altitude_AB : Prop)
variable (DF_altitude_BC : Prop)

-- Conditions
axiom AB_eq_DC : AB = DC
axiom EB_eq_5 : EB = 5
axiom DE_eq_8 : DE = 8

-- The main theorem to prove
theorem DF_length (hAB : AB = 15) (hDC : DC = 15) (hEB : EB = 5) (hDE : DE = 8)
  (hPar : parallelogram_ABCD)
  (hAltAB : DE_altitude_AB)
  (hAltBC : DF_altitude_BC) :
  ∃ DF : ℝ, DF = 8 := 
sorry

end DF_length_l199_199806


namespace price_reduction_daily_profit_l199_199383

theorem price_reduction_daily_profit
    (profit_per_item : ℕ)
    (avg_daily_sales : ℕ)
    (item_increase_per_unit_price_reduction : ℕ)
    (target_daily_profit : ℕ)
    (x : ℕ) :
    profit_per_item = 40 →
    avg_daily_sales = 20 →
    item_increase_per_unit_price_reduction = 2 →
    target_daily_profit = 1200 →

    ((profit_per_item - x) * (avg_daily_sales + item_increase_per_unit_price_reduction * x) = target_daily_profit) →
    x = 20 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_reduction_daily_profit_l199_199383


namespace g_3_2_eq_neg3_l199_199542

noncomputable def f (x y : ℝ) : ℝ := x^3 * y^2 + 4 * x^2 * y - 15 * x

axiom f_symmetric : ∀ x y : ℝ, f x y = f y x
axiom f_2_4_eq_neg2 : f 2 4 = -2

noncomputable def g (x y : ℝ) : ℝ := (x^3 - 3 * x^2 * y + x * y^2) / (x^2 - y^2)

theorem g_3_2_eq_neg3 : g 3 2 = -3 := by
  sorry

end g_3_2_eq_neg3_l199_199542


namespace sufficient_but_not_necessary_not_necessary_l199_199171

theorem sufficient_but_not_necessary (m x y a : ℝ) (h₀ : m > 0) (h₁ : |x - a| < m) (h₂ : |y - a| < m) : |x - y| < 2 * m :=
by
  sorry

theorem not_necessary (m : ℝ) (h₀ : m > 0) : ∃ x y a : ℝ, |x - y| < 2 * m ∧ ¬ (|x - a| < m ∧ |y - a| < m) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l199_199171


namespace solution_l199_199478

axiom f : ℝ → ℝ

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → y ≤ 0 → f x > f y

def main_problem : Prop :=
  even_function f ∧ decreasing_function f ∧ f (-2) = 0 → ∀ x, f x < 0 ↔ x > -2 ∧ x < 2

theorem solution : main_problem :=
by
  sorry

end solution_l199_199478


namespace initial_average_age_l199_199209

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 17) (h2 : n * A + 32 = (n + 1) * 15) : A = 14 := by
  sorry

end initial_average_age_l199_199209


namespace arithmetic_geometric_progression_l199_199916

theorem arithmetic_geometric_progression (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : b = 3 * a) (h4 : 2 * b = a + c) (h5 : b * b = a * c) : c = 9 :=
sorry

end arithmetic_geometric_progression_l199_199916


namespace Elizabeth_More_Revenue_Than_Banks_l199_199646

theorem Elizabeth_More_Revenue_Than_Banks : 
  let banks_investments := 8
  let banks_revenue_per_investment := 500
  let elizabeth_investments := 5
  let elizabeth_revenue_per_investment := 900
  let banks_total_revenue := banks_investments * banks_revenue_per_investment
  let elizabeth_total_revenue := elizabeth_investments * elizabeth_revenue_per_investment
  elizabeth_total_revenue - banks_total_revenue = 500 :=
by
  sorry

end Elizabeth_More_Revenue_Than_Banks_l199_199646


namespace find_cd_l199_199333

noncomputable def g (x : ℝ) (c : ℝ) (d : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) :
  g 2 c d = -9 ∧ g (-1) c d = -19 ↔
  (c = 19/3 ∧ d = -7/3) :=
by
  sorry

end find_cd_l199_199333


namespace arctan_sum_of_roots_eq_pi_div_4_l199_199856

theorem arctan_sum_of_roots_eq_pi_div_4 (x₁ x₂ x₃ : ℝ) 
  (h₁ : Polynomial.eval x₁ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₂ : Polynomial.eval x₂ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h₃ : Polynomial.eval x₃ (Polynomial.C 11 - Polynomial.C 10 * Polynomial.X + Polynomial.X ^ 3) = 0)
  (h_intv : -5 < x₁ ∧ x₁ < 5 ∧ -5 < x₂ ∧ x₂ < 5 ∧ -5 < x₃ ∧ x₃ < 5) :
  Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = Real.pi / 4 :=
sorry

end arctan_sum_of_roots_eq_pi_div_4_l199_199856


namespace geometric_sequence_common_ratio_l199_199546

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
    (h1 : a 1 = -1)
    (h2 : a 2 + a 3 = -2) :
    ∃ q : ℝ, (a 2 = a 1 * q) ∧ (a 3 = a 1 * q^2) ∧ (q = -2 ∨ q = 1) :=
sorry

end geometric_sequence_common_ratio_l199_199546


namespace zoe_has_47_nickels_l199_199746

theorem zoe_has_47_nickels (x : ℕ) 
  (h1 : 5 * x + 10 * x + 50 * x = 3050) : 
  x = 47 := 
sorry

end zoe_has_47_nickels_l199_199746


namespace find_number_l199_199397

theorem find_number (x : ℝ) : 14 * x + 15 * x + 18 * x + 11 = 152 → x = 3 := by
  sorry

end find_number_l199_199397


namespace Annika_hike_time_l199_199513

-- Define the conditions
def hike_rate : ℝ := 12 -- in minutes per kilometer
def initial_distance_east : ℝ := 2.75 -- in kilometers
def total_distance_east : ℝ := 3.041666666666667 -- in kilometers
def total_time_needed : ℝ := 40 -- in minutes

-- The theorem to prove
theorem Annika_hike_time : 
  (initial_distance_east + (total_distance_east - initial_distance_east)) * hike_rate + total_distance_east * hike_rate = total_time_needed := 
by
  sorry

end Annika_hike_time_l199_199513


namespace total_shaded_area_l199_199177

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 :=
by 
  sorry

end total_shaded_area_l199_199177


namespace tracy_candies_l199_199255

variable (x : ℕ) -- number of candies Tracy started with

theorem tracy_candies (h1: x % 4 = 0)
                      (h2 : 46 ≤ x / 2 - 40 ∧ x / 2 - 40 ≤ 50) 
                      (h3 : ∃ k, 2 ≤ k ∧ k ≤ 6 ∧ x / 2 - 40 - k = 4) 
                      (h4 : ∃ n, x = 4 * n) : x = 96 :=
by
  sorry

end tracy_candies_l199_199255


namespace a1_b1_sum_l199_199127

-- Definitions from the conditions:
def strict_inc_seq (s : ℕ → ℕ) : Prop := ∀ n, s n < s (n + 1)

def positive_int_seq (s : ℕ → ℕ) : Prop := ∀ n, s n > 0

def a : ℕ → ℕ := sorry -- Define the sequence 'a' (details skipped).

def b : ℕ → ℕ := sorry -- Define the sequence 'b' (details skipped).

-- Conditions given:
axiom cond_a_inc : strict_inc_seq a

axiom cond_b_inc : strict_inc_seq b

axiom cond_a_pos : positive_int_seq a

axiom cond_b_pos : positive_int_seq b

axiom cond_a10_b10_lt_2017 : a 10 = b 10 ∧ a 10 < 2017

axiom cond_a_rec : ∀ n, a (n + 2) = a (n + 1) + a n

axiom cond_b_rec : ∀ n, b (n + 1) = 2 * b n

-- The theorem to prove:
theorem a1_b1_sum : a 1 + b 1 = 5 :=
sorry

end a1_b1_sum_l199_199127


namespace ratio_problem_l199_199430

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l199_199430


namespace car_dealership_theorem_l199_199169

def car_dealership_problem : Prop :=
  let initial_cars := 100
  let new_shipment := 150
  let initial_silver_percentage := 0.20
  let new_silver_percentage := 0.40
  let initial_silver := initial_silver_percentage * initial_cars
  let new_silver := new_silver_percentage * new_shipment
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_shipment
  let silver_percentage := (total_silver / total_cars) * 100
  silver_percentage = 32

theorem car_dealership_theorem : car_dealership_problem :=
by {
  sorry
}

end car_dealership_theorem_l199_199169


namespace sales_in_fifth_month_l199_199382

-- Define the sales figures and average target
def s1 : ℕ := 6435
def s2 : ℕ := 6927
def s3 : ℕ := 6855
def s4 : ℕ := 7230
def s6 : ℕ := 6191
def s_target : ℕ := 6700
def n_months : ℕ := 6

-- Define the total sales and the required fifth month sale
def total_sales : ℕ := s_target * n_months
def s5 : ℕ := total_sales - (s1 + s2 + s3 + s4 + s6)

-- The main theorem statement we need to prove
theorem sales_in_fifth_month :
  s5 = 6562 :=
sorry

end sales_in_fifth_month_l199_199382


namespace number_of_three_digit_numbers_divisible_by_17_l199_199106

theorem number_of_three_digit_numbers_divisible_by_17 : 
  let k_min := Nat.ceil (100 / 17)
  let k_max := Nat.floor (999 / 17)
  ∃ n, 
    (n = k_max - k_min + 1) ∧ 
    (n = 53) := 
by
    sorry

end number_of_three_digit_numbers_divisible_by_17_l199_199106


namespace sum_of_n_and_k_l199_199818

open Nat

theorem sum_of_n_and_k (n k : ℕ)
  (h1 : 2 = n - 3 * k)
  (h2 : 8 = 2 * n - 5 * k) :
  n + k = 18 :=
sorry

end sum_of_n_and_k_l199_199818


namespace return_trip_time_l199_199754

-- conditions 
variables (d p w : ℝ) (h1 : d = 90 * (p - w)) (h2 : ∀ t : ℝ, t = d / p → d / (p + w) = t - 15)

--  statement
theorem return_trip_time :
  ∃ t : ℝ, t = 30 ∨ t = 45 :=
by
  -- placeholder proof 
  sorry

end return_trip_time_l199_199754


namespace probability_exactly_three_heads_in_seven_tosses_l199_199686

def combinations (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) : ℚ :=
  (combinations n k) / (2^n : ℚ)

theorem probability_exactly_three_heads_in_seven_tosses :
  binomial_probability 7 3 = 35 / 128 := 
by 
  sorry

end probability_exactly_three_heads_in_seven_tosses_l199_199686


namespace arithmetic_sequence_30th_term_l199_199534

theorem arithmetic_sequence_30th_term :
  let a := 3
  let d := 7 - 3
  ∀ n, (n = 30) → (a + (n - 1) * d) = 119 := by
  sorry

end arithmetic_sequence_30th_term_l199_199534


namespace inequality_solution_set_l199_199260

theorem inequality_solution_set :
  ∀ x : ℝ, (1 / (x^2 + 1) > 5 / x + 21 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end inequality_solution_set_l199_199260


namespace intersection_P_Q_l199_199116

open Set

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | abs x < 2}

theorem intersection_P_Q : P ∩ Q = {1} :=
by
  sorry

end intersection_P_Q_l199_199116


namespace sum_of_edges_of_square_l199_199979

theorem sum_of_edges_of_square (u v w x : ℕ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hx : 0 < x) 
(hsum : u * x + u * v + v * w + w * x = 15) : u + v + w + x = 8 :=
by
  sorry

end sum_of_edges_of_square_l199_199979


namespace hexagon_planting_schemes_l199_199118

theorem hexagon_planting_schemes (n m : ℕ) (h : n = 4 ∧ m = 6) : 
  ∃ k, k = 732 := 
by sorry

end hexagon_planting_schemes_l199_199118


namespace probability_15th_roll_last_is_approximately_l199_199845

noncomputable def probability_15th_roll_last : ℝ :=
  (7 / 8) ^ 13 * (1 / 8)

theorem probability_15th_roll_last_is_approximately :
  abs (probability_15th_roll_last - 0.022) < 0.001 :=
by sorry

end probability_15th_roll_last_is_approximately_l199_199845


namespace younger_brother_silver_fraction_l199_199020

def frac_silver (x y : ℕ) : ℚ := (100 - x / 7 ) / y

theorem younger_brother_silver_fraction {x y : ℕ} 
    (cond1 : x / 5 + y / 7 = 100) 
    (cond2 : x / 7 + (100 - x / 7) = 100) : 
    frac_silver x y = 5 / 14 := 
sorry

end younger_brother_silver_fraction_l199_199020


namespace total_visible_surface_area_l199_199766

-- Define the cubes by their volumes
def volumes : List ℝ := [1, 8, 27, 125, 216, 343, 512, 729]

-- Define the arrangement information as specified
def arrangement_conditions : Prop :=
  ∃ (s8 s7 s6 s5 s4 s3 s2 s1 : ℝ),
    s8^3 = 729 ∧ s7^3 = 512 ∧ s6^3 = 343 ∧ s5^3 = 216 ∧
    s4^3 = 125 ∧ s3^3 = 27 ∧ s2^3 = 8 ∧ s1^3 = 1 ∧
    5 * s8^2 + (5 * s7^2 + 4 * s6^2 + 4 * s5^2) + 
    (5 * s4^2 + 4 * s3^2 + 5 * s2^2 + 4 * s1^2) = 1250

-- The proof statement
theorem total_visible_surface_area : arrangement_conditions → 1250 = 1250 := by
  intro _ -- this stands for not proving the condition, taking it as assumption
  exact rfl


end total_visible_surface_area_l199_199766


namespace area_not_covered_by_smaller_squares_l199_199365

-- Define the conditions given in the problem
def side_length_larger_square : ℕ := 10
def side_length_smaller_square : ℕ := 4
def area_of_larger_square : ℕ := side_length_larger_square * side_length_larger_square
def area_of_each_smaller_square : ℕ := side_length_smaller_square * side_length_smaller_square

-- Define the total area of the two smaller squares
def total_area_smaller_squares : ℕ := area_of_each_smaller_square * 2

-- Define the uncovered area
def uncovered_area : ℕ := area_of_larger_square - total_area_smaller_squares

-- State the theorem to prove
theorem area_not_covered_by_smaller_squares :
  uncovered_area = 68 := by
  -- Placeholder for the actual proof
  sorry

end area_not_covered_by_smaller_squares_l199_199365


namespace min_value_l199_199452

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 3 + 2 * Real.sqrt 2 ≤ 2 / a + 1 / b :=
by
  sorry

end min_value_l199_199452


namespace find_a4_l199_199724

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

theorem find_a4 (a_n : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a_n q →
  a_n 1 + a_n 2 = -1 →
  a_n 1 - a_n 3 = -3 →
  a_n 4 = -8 :=
by 
  sorry

end find_a4_l199_199724


namespace seq_identity_l199_199659

-- Define the sequence (a_n)
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 0 ∧ a 2 = 1 ∧ ∀ n, a (n + 3) = a (n + 1) + 1998 * a n

theorem seq_identity (a : ℕ → ℕ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * (a (n - 1))^2 :=
sorry

end seq_identity_l199_199659


namespace part1_part2_l199_199631

noncomputable def f (a x : ℝ) : ℝ := (a / 2) * x * x - (a - 2) * x - 2 * x * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := a * x - a - 2 * Real.log x

theorem part1 (a : ℝ) : (∀ x > 0, f' a x ≥ 0) ↔ a = 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : f' a x1 = 0) (h4 : f' a x2 = 0) (h5 : x1 < x2) : 
  x2 - x1 > 4 / a - 2 :=
sorry

end part1_part2_l199_199631


namespace donny_total_cost_eq_45_l199_199644

-- Definitions for prices of each type of apple
def price_small : ℝ := 1.5
def price_medium : ℝ := 2
def price_big : ℝ := 3

-- Quantities purchased by Donny
def count_small : ℕ := 6
def count_medium : ℕ := 6
def count_big : ℕ := 8

-- Total cost calculation
def total_cost (count_small count_medium count_big : ℕ) : ℝ := 
  (count_small * price_small) + (count_medium * price_medium) + (count_big * price_big)

-- Theorem stating the total cost
theorem donny_total_cost_eq_45 : total_cost count_small count_medium count_big = 45 := by
  sorry

end donny_total_cost_eq_45_l199_199644


namespace mod_pow_difference_l199_199245

theorem mod_pow_difference (a b n : ℕ) (h1 : a ≡ 47 [MOD n]) (h2 : b ≡ 22 [MOD n]) (h3 : n = 8) : (a ^ 2023 - b ^ 2023) % n = 1 :=
by
  sorry

end mod_pow_difference_l199_199245


namespace molecular_weight_n2o_l199_199868

theorem molecular_weight_n2o (w : ℕ) (n : ℕ) (h : w = 352 ∧ n = 8) : (w / n = 44) :=
sorry

end molecular_weight_n2o_l199_199868


namespace subtraction_correctness_l199_199872

theorem subtraction_correctness : 25.705 - 3.289 = 22.416 := 
by
  sorry

end subtraction_correctness_l199_199872


namespace tile_calc_proof_l199_199005

noncomputable def total_tiles (length width : ℕ) : ℕ :=
  let border_tiles_length := (2 * (length - 4)) * 2
  let border_tiles_width := (2 * (width - 4)) * 2
  let total_border_tiles := (border_tiles_length + border_tiles_width) * 2 - 8
  let inner_length := (length - 4)
  let inner_width := (width - 4)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / 4
  total_border_tiles + inner_tiles

theorem tile_calc_proof :
  total_tiles 15 20 = 144 :=
by
  sorry

end tile_calc_proof_l199_199005


namespace problem1_problem2_l199_199221

variable {a b x : ℝ}

theorem problem1 (h₀ : a ≠ b) (h₁ : a ≠ -b) :
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

theorem problem2 (h₀ : x ≠ 2) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  ((x - 2) / (x - 1)) / ((x^2 - 4 * x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end problem1_problem2_l199_199221


namespace count_integers_satisfy_inequality_l199_199711

theorem count_integers_satisfy_inequality : 
  ∃ l : List Int, (∀ n ∈ l, (n - 3) * (n + 5) < 0) ∧ l.length = 7 :=
by
  sorry

end count_integers_satisfy_inequality_l199_199711


namespace arrange_2015_integers_l199_199585

theorem arrange_2015_integers :
  ∃ (f : Fin 2015 → Fin 2015),
    (∀ i, (Nat.gcd ((f i).val + (f (i + 1)).val) 4 = 1 ∨ Nat.gcd ((f i).val + (f (i + 1)).val) 7 = 1)) ∧
    Function.Injective f ∧ 
    (∀ i, 1 ≤ (f i).val ∧ (f i).val ≤ 2015) :=
sorry

end arrange_2015_integers_l199_199585


namespace series_converges_to_one_l199_199000

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n : ℝ)^2 - 2 * (n : ℝ) + 1) / ((n : ℝ)^4 - (n : ℝ)^3 + (n : ℝ)^2 - (n : ℝ) + 1) else 0

theorem series_converges_to_one : series_sum = 1 := 
  sorry

end series_converges_to_one_l199_199000


namespace range_of_z_l199_199350

variable (x y z : ℝ)

theorem range_of_z (hx : x ≥ 0) (hy : y ≥ x) (hxy : 4*x + 3*y ≤ 12) 
(hz : z = (x + 2 * y + 3) / (x + 1)) : 
2 ≤ z ∧ z ≤ 6 :=
sorry

end range_of_z_l199_199350


namespace painter_total_cost_l199_199936

def south_seq (n : Nat) : Nat :=
  4 + 6 * (n - 1)

def north_seq (n : Nat) : Nat :=
  5 + 6 * (n - 1)

noncomputable def digit_cost (n : Nat) : Nat :=
  String.length (toString n)

noncomputable def total_cost : Nat :=
  let south_cost := (List.range 25).map south_seq |>.map digit_cost |>.sum
  let north_cost := (List.range 25).map north_seq |>.map digit_cost |>.sum
  south_cost + north_cost

theorem painter_total_cost : total_cost = 116 := by
  sorry

end painter_total_cost_l199_199936


namespace arithmetic_sequence_sum_l199_199372

theorem arithmetic_sequence_sum :
  let a1 := 1
  let d := 2
  let n := 10
  let an := 19
  let sum := 100
  let general_term := fun (n : ℕ) => a1 + (n - 1) * d
  (general_term n = an) → (n = 10) → (sum = (n * (a1 + an)) / 2) →
  sum = 100 :=
by
  sorry

end arithmetic_sequence_sum_l199_199372


namespace six_digit_number_theorem_l199_199854

-- Define the problem conditions
def six_digit_number_condition (N : ℕ) (x : ℕ) : Prop :=
  N = 200000 + x ∧ N < 1000000 ∧ (10 * x + 2 = 3 * N)

-- Define the value of x
def value_of_x : ℕ := 85714

-- Main theorem to prove
theorem six_digit_number_theorem (N : ℕ) (x : ℕ) (h1 : x = value_of_x) :
  six_digit_number_condition N x → N = 285714 :=
by
  intros h
  sorry

end six_digit_number_theorem_l199_199854


namespace min_M_value_l199_199802

variable {a b c t : ℝ}

theorem min_M_value (h1 : a < b)
                    (h2 : a > 0)
                    (h3 : b^2 - 4 * a * c ≤ 0)
                    (h4 : b = t + a)
                    (h5 : t > 0)
                    (h6 : c ≥ (t + a)^2 / (4 * a)) :
    ∃ M : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c) ≥ 0) → M = 3 := 
  sorry

end min_M_value_l199_199802


namespace complex_square_l199_199011

-- Define z and the condition on i
def z := 5 + (6 * Complex.I)
axiom i_squared : Complex.I ^ 2 = -1

-- State the theorem to prove z^2 = -11 + 60i
theorem complex_square : z ^ 2 = -11 + (60 * Complex.I) := by {
  sorry
}

end complex_square_l199_199011


namespace probability_equivalence_l199_199036

-- Definitions for the conditions:
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3

-- Function to return the probability of selecting a genuine product on the second draw, given first is defective
def probability_genuine_given_defective : ℚ := 
  (defective_products / total_products) * (genuine_products / (total_products - 1))

-- The theorem we need to prove:
theorem probability_equivalence :
  probability_genuine_given_defective = 2 / 3 :=
by
  sorry -- Proof placeholder

end probability_equivalence_l199_199036


namespace problem_part1_problem_part2_l199_199056

-- Statement part (1)
theorem problem_part1 : ( (2 / 3) - (1 / 4) - (1 / 6) ) * 24 = 6 :=
sorry

-- Statement part (2)
theorem problem_part2 : (-2)^3 + (-9 + (-3)^2 * (1 / 3)) = -14 :=
sorry

end problem_part1_problem_part2_l199_199056


namespace kvass_affordability_l199_199294

theorem kvass_affordability (x y : ℚ) (hx : x + y = 1) (hxy : 1.2 * (0.5 * x + y) = 1) : 1.44 * y ≤ 1 :=
by
  -- Placeholder for proof
  sorry

end kvass_affordability_l199_199294


namespace percentage_increase_chef_vs_dishwasher_l199_199322

variables 
  (manager_wage chef_wage dishwasher_wage : ℝ)
  (h_manager_wage : manager_wage = 8.50)
  (h_chef_wage : chef_wage = manager_wage - 3.315)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)

theorem percentage_increase_chef_vs_dishwasher :
  ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 22 :=
by
  sorry

end percentage_increase_chef_vs_dishwasher_l199_199322


namespace line_equations_satisfy_conditions_l199_199775

-- Definitions and conditions:
def intersects_at_distance (k m b : ℝ) : Prop :=
  |(k^2 + 7*k + 12) - (m*k + b)| = 8

def passes_through_point (m b : ℝ) : Prop :=
  7 = 2*m + b

def line_equation_valid (m b : ℝ) : Prop :=
  b ≠ 0

-- Main theorem:
theorem line_equations_satisfy_conditions :
  (line_equation_valid 1 5 ∧ passes_through_point 1 5 ∧ 
  ∃ k, intersects_at_distance k 1 5) ∨
  (line_equation_valid 5 (-3) ∧ passes_through_point 5 (-3) ∧ 
  ∃ k, intersects_at_distance k 5 (-3)) :=
by
  sorry

end line_equations_satisfy_conditions_l199_199775


namespace alcohol_quantity_l199_199742

theorem alcohol_quantity (A W : ℝ) (h1 : A / W = 2 / 5) (h2 : A / (W + 10) = 2 / 7) : A = 10 :=
by
  sorry

end alcohol_quantity_l199_199742


namespace distinct_ordered_pairs_eq_49_l199_199058

theorem distinct_ordered_pairs_eq_49 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 49) (hy : 1 ≤ y ∧ y ≤ 49) (h_eq : x + y = 50) :
  ∃ xs : List (ℕ × ℕ), (∀ p ∈ xs, p.1 + p.2 = 50 ∧ 1 ≤ p.1 ∧ p.1 ≤ 49 ∧ 1 ≤ p.2 ∧ p.2 ≤ 49) ∧ xs.length = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l199_199058


namespace cos120_sin_neg45_equals_l199_199490

noncomputable def cos120_plus_sin_neg45 : ℝ :=
  Real.cos (120 * Real.pi / 180) + Real.sin (-45 * Real.pi / 180)

theorem cos120_sin_neg45_equals : cos120_plus_sin_neg45 = - (1 + Real.sqrt 2) / 2 :=
by
  sorry

end cos120_sin_neg45_equals_l199_199490


namespace sum_of_integers_with_product_2720_l199_199906

theorem sum_of_integers_with_product_2720 (n : ℤ) (h1 : n > 0) (h2 : n * (n + 2) = 2720) : n + (n + 2) = 104 :=
by {
  sorry
}

end sum_of_integers_with_product_2720_l199_199906


namespace intersection_M_N_l199_199951

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | (1/3) ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | (1/3) ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l199_199951


namespace strands_of_duct_tape_used_l199_199893

-- Define the conditions
def hannah_cut_rate : ℕ := 8  -- Hannah's cutting rate
def son_cut_rate : ℕ := 3     -- Son's cutting rate
def minutes : ℕ := 2          -- Time taken to free the younger son

-- Define the total cutting rate
def total_cut_rate : ℕ := hannah_cut_rate + son_cut_rate

-- Define the total number of strands
def total_strands : ℕ := total_cut_rate * minutes

-- State the theorem to prove
theorem strands_of_duct_tape_used : total_strands = 22 :=
by
  sorry

end strands_of_duct_tape_used_l199_199893


namespace k_value_if_root_is_one_l199_199491

theorem k_value_if_root_is_one (k : ℝ) (h : (k - 1) * 1 ^ 2 + 1 - k ^ 2 = 0) : k = 0 := 
by
  sorry

end k_value_if_root_is_one_l199_199491


namespace annual_savings_l199_199476

-- defining the conditions
def current_speed := 10 -- in Mbps
def current_bill := 20 -- in dollars
def bill_30Mbps := 2 * current_bill -- in dollars
def bill_20Mbps := current_bill + 10 -- in dollars
def months_in_year := 12

-- calculating the annual costs
def annual_cost_30Mbps := bill_30Mbps * months_in_year
def annual_cost_20Mbps := bill_20Mbps * months_in_year

-- statement of the problem
theorem annual_savings : (annual_cost_30Mbps - annual_cost_20Mbps) = 120 := by
  sorry -- prove the statement

end annual_savings_l199_199476


namespace fraction_sum_eq_one_l199_199573

variables {a b c x y z : ℝ}

-- Conditions
axiom h1 : 11 * x + b * y + c * z = 0
axiom h2 : a * x + 24 * y + c * z = 0
axiom h3 : a * x + b * y + 41 * z = 0
axiom h4 : a ≠ 11
axiom h5 : x ≠ 0

-- Theorem Statement
theorem fraction_sum_eq_one : 
  a/(a - 11) + b/(b - 24) + c/(c - 41) = 1 :=
by sorry

end fraction_sum_eq_one_l199_199573


namespace profit_percentage_is_40_l199_199227

-- Define the given conditions
def total_cost : ℚ := 44 * 150 + 36 * 125  -- Rs 11100
def total_weight : ℚ := 44 + 36            -- 80 kg
def selling_price_per_kg : ℚ := 194.25     -- Rs 194.25
def total_selling_price : ℚ := total_weight * selling_price_per_kg  -- Rs 15540
def profit : ℚ := total_selling_price - total_cost  -- Rs 4440

-- Define the statement about the profit percentage
def profit_percentage : ℚ := (profit / total_cost) * 100

-- State the theorem
theorem profit_percentage_is_40 :
  profit_percentage = 40 := by
  -- This is where the proof would go
  sorry

end profit_percentage_is_40_l199_199227


namespace expression_value_l199_199460

theorem expression_value (x a b c : ℝ) 
  (ha : a + x^2 = 2006) 
  (hb : b + x^2 = 2007) 
  (hc : c + x^2 = 2008) 
  (h_abc : a * b * c = 3) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1) := 
  sorry

end expression_value_l199_199460


namespace part_a_contradiction_l199_199266

theorem part_a_contradiction :
  ¬ (225 / 25 + 75 = 100 - 16 → 25 * (9 / (1 + 3)) = 84) :=
by
  sorry

end part_a_contradiction_l199_199266


namespace sum_of_differences_of_7_in_657932657_l199_199218

theorem sum_of_differences_of_7_in_657932657 :
  let numeral := 657932657
  let face_value (d : Nat) := d
  let local_value (d : Nat) (pos : Nat) := d * 10 ^ pos
  let indices_of_7 := [6, 0]
  let differences := indices_of_7.map (fun pos => local_value 7 pos - face_value 7)
  differences.sum = 6999993 :=
by
  sorry

end sum_of_differences_of_7_in_657932657_l199_199218


namespace assign_grades_l199_199345

def num_students : ℕ := 15
def options_per_student : ℕ := 4

theorem assign_grades:
  options_per_student ^ num_students = 1073741824 := by
  sorry

end assign_grades_l199_199345


namespace volume_is_six_l199_199927

-- Define the polygons and their properties
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0)
def rectangle (l w : ℝ) := (l > 0 ∧ w > 0)
def equilateral_triangle (s : ℝ) := (s > 0)

-- The given polygons
def A := right_triangle 1 2 (Real.sqrt 5)
def E := right_triangle 1 2 (Real.sqrt 5)
def F := right_triangle 1 2 (Real.sqrt 5)
def B := rectangle 1 2
def C := rectangle 2 3
def D := rectangle 1 3
def G := equilateral_triangle (Real.sqrt 5)

-- The volume of the polyhedron
-- Assume the largest rectangle C forms the base and a reasonable height
def volume_of_polyhedron : ℝ := 6

theorem volume_is_six : 
  (right_triangle 1 2 (Real.sqrt 5)) → 
  (rectangle 1 2) → 
  (rectangle 2 3) → 
  (rectangle 1 3) → 
  (equilateral_triangle (Real.sqrt 5)) → 
  volume_of_polyhedron = 6 := 
by 
  sorry

end volume_is_six_l199_199927


namespace average_weight_of_a_b_c_l199_199271

theorem average_weight_of_a_b_c (A B C : ℕ) 
  (h1 : (A + B) / 2 = 25) 
  (h2 : (B + C) / 2 = 28) 
  (hB : B = 16) : 
  (A + B + C) / 3 = 30 := 
by 
  sorry

end average_weight_of_a_b_c_l199_199271


namespace cos_half_pi_plus_double_alpha_l199_199251

theorem cos_half_pi_plus_double_alpha (α : ℝ) (h : Real.tan α = 1 / 3) : 
  Real.cos (Real.pi / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end cos_half_pi_plus_double_alpha_l199_199251


namespace smallest_n_cond_l199_199207

theorem smallest_n_cond (n : ℕ) (h1 : n >= 100 ∧ n < 1000) (h2 : n ≡ 3 [MOD 9]) (h3 : n ≡ 3 [MOD 4]) : n = 111 := 
sorry

end smallest_n_cond_l199_199207


namespace tan_theta_eq_two_implies_expression_l199_199605

theorem tan_theta_eq_two_implies_expression (θ : ℝ) (h : Real.tan θ = 2) :
    (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
by
  -- Define trig identities and given condition
  have h_sin_cos : Real.sin θ = 2 / Real.sqrt 5 ∧ Real.cos θ = 1 / Real.sqrt 5 :=
    sorry -- This will be derived from the given condition h
  
  -- Main proof
  sorry

end tan_theta_eq_two_implies_expression_l199_199605


namespace condition_nonzero_neither_zero_l199_199792

theorem condition_nonzero_neither_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end condition_nonzero_neither_zero_l199_199792


namespace set_swept_by_all_lines_l199_199716

theorem set_swept_by_all_lines
  (a c x y : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < c)
  (h3 : c < a)
  (h4 : x^2 + y^2 ≤ a^2) : 
  (c^2 - a^2) * x^2 - a^2 * y^2 ≤ (c^2 - a^2) * c^2 :=
sorry

end set_swept_by_all_lines_l199_199716


namespace people_in_third_row_l199_199918

theorem people_in_third_row (row1_ini row2_ini left_row1 left_row2 total_left : ℕ) (h1 : row1_ini = 24) (h2 : row2_ini = 20) (h3 : left_row1 = row1_ini - 3) (h4 : left_row2 = row2_ini - 5) (h_total : total_left = 54) :
  total_left - (left_row1 + left_row2) = 18 := 
by
  sorry

end people_in_third_row_l199_199918


namespace ball_third_bounce_distance_is_correct_l199_199994

noncomputable def total_distance_third_bounce (initial_height : ℝ) (rebound_ratio : ℝ) : ℝ :=
  initial_height + 2 * (initial_height * rebound_ratio) + 2 * (initial_height * rebound_ratio^2)

theorem ball_third_bounce_distance_is_correct : 
  total_distance_third_bounce 80 (2/3) = 257.78 := 
by
  sorry

end ball_third_bounce_distance_is_correct_l199_199994


namespace ceil_minus_floor_eq_one_implies_ceil_minus_y_l199_199914

noncomputable def fractional_part (y : ℝ) : ℝ := y - ⌊y⌋

theorem ceil_minus_floor_eq_one_implies_ceil_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - fractional_part y :=
by
  sorry

end ceil_minus_floor_eq_one_implies_ceil_minus_y_l199_199914


namespace solution_set_of_inequality_l199_199733

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 2*x + 3 > 0 ↔ (-1 < x ∧ x < 3) :=
sorry

end solution_set_of_inequality_l199_199733


namespace relationship_l199_199613

noncomputable def a : ℝ := 3^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 3 / Real.log 2⁻¹
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship (a_def : a = 3^(-1/3 : ℝ)) 
                     (b_def : b = Real.log 3 / Real.log 2⁻¹) 
                     (c_def : c = Real.log 3 / Real.log 2) : 
  b < a ∧ a < c :=
  sorry

end relationship_l199_199613


namespace y_plus_inv_l199_199614

theorem y_plus_inv (y : ℝ) (h : y^3 + 1/y^3 = 110) : y + 1/y = 5 := 
by 
sorry

end y_plus_inv_l199_199614


namespace lefty_jazz_non_basketball_l199_199401

-- Definitions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_loving_members : ℕ := 20
def right_handed_non_jazz_non_basketball : ℕ := 5
def basketball_players : ℕ := 10
def left_handed_jazz_loving_basketball_players : ℕ := 3

-- Problem Statement: Prove the number of lefty jazz lovers who do not play basketball.
theorem lefty_jazz_non_basketball (x : ℕ) :
  (x + left_handed_jazz_loving_basketball_players) + (left_handed_members - x - left_handed_jazz_loving_basketball_players) + 
  (jazz_loving_members - x - left_handed_jazz_loving_basketball_players) + 
  right_handed_non_jazz_non_basketball + left_handed_jazz_loving_basketball_players = 
  total_members → x = 4 :=
by
  sorry

end lefty_jazz_non_basketball_l199_199401


namespace weight_of_a_l199_199952

variables (a b c d e : ℝ)

theorem weight_of_a (h1 : (a + b + c) / 3 = 80)
                    (h2 : (a + b + c + d) / 4 = 82)
                    (h3 : e = d + 3)
                    (h4 : (b + c + d + e) / 4 = 81) :
  a = 95 :=
by
  sorry

end weight_of_a_l199_199952


namespace neither_necessary_nor_sufficient_l199_199579

noncomputable def C1 (m n : ℝ) :=
  (m ^ 2 - 4 * n ≥ 0) ∧ (m > 0) ∧ (n > 0)

noncomputable def C2 (m n : ℝ) :=
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem neither_necessary_nor_sufficient (m n : ℝ) :
  ¬(C1 m n → C2 m n) ∧ ¬(C2 m n → C1 m n) :=
sorry

end neither_necessary_nor_sufficient_l199_199579


namespace intersection_A_B_l199_199980

def A : Set ℝ := { x : ℝ | |x - 1| < 2 }
def B : Set ℝ := { x : ℝ | x^2 - x - 2 > 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l199_199980


namespace question1_question2_l199_199190

noncomputable def minimum_value (x y : ℝ) : ℝ := (1 / x) + (1 / y)

theorem question1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) : 
  minimum_value x y = 2 :=
sorry

theorem question2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
  (x + 1) * (y + 1) ≠ 5 :=
sorry

end question1_question2_l199_199190


namespace distance_between_first_and_last_tree_l199_199901

theorem distance_between_first_and_last_tree (n : ℕ) (d : ℕ) 
  (h₁ : n = 8)
  (h₂ : d = 75)
  : (d / ((4 - 1) : ℕ)) * (n - 1) = 175 := sorry

end distance_between_first_and_last_tree_l199_199901


namespace maximum_value_of_func_l199_199890

noncomputable def func (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximum_value_of_func (x : ℝ) (h : x < 5 / 4) : ∃ y, y = 1 ∧ ∀ z, z = func x → z ≤ y :=
sorry

end maximum_value_of_func_l199_199890


namespace larger_integer_21_l199_199760

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l199_199760


namespace calculate_value_l199_199499

theorem calculate_value :
  let X := (354 * 28) ^ 2
  let Y := (48 * 14) ^ 2
  (X * 9) / (Y * 2) = 2255688 :=
by
  sorry

end calculate_value_l199_199499


namespace approx_change_in_y_l199_199407

-- Definition of the function
def y (x : ℝ) : ℝ := x^3 - 7 * x^2 + 80

-- Derivative of the function, calculated manually
def y_prime (x : ℝ) : ℝ := 3 * x^2 - 14 * x

-- The change in x
def delta_x : ℝ := 0.01

-- The given value of x
def x_initial : ℝ := 5

-- To be proved: the approximate change in y
theorem approx_change_in_y : (y_prime x_initial) * delta_x = 0.05 :=
by
  -- Imported and recognized theorem verifications skipped
  sorry

end approx_change_in_y_l199_199407


namespace probability_no_adjacent_birch_trees_l199_199420

open Nat

theorem probability_no_adjacent_birch_trees : 
    let m := 7
    let n := 990
    m + n = 106 := 
by
  sorry

end probability_no_adjacent_birch_trees_l199_199420


namespace school_club_members_l199_199537

theorem school_club_members :
  ∃ n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 3 ∧
  n % 8 = 5 ∧
  n % 9 = 7 ∧
  n = 269 :=
by
  existsi 269
  sorry

end school_club_members_l199_199537


namespace smallest_solution_l199_199898

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 3) + 1 / (x - 5) + 1 / (x - 6) = 4 / (x - 4))

def valid_x (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 6

theorem smallest_solution (x : ℝ) (h1 : equation x) (h2 : valid_x x) : x = 16 := sorry

end smallest_solution_l199_199898


namespace unique_solution_value_k_l199_199481

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l199_199481


namespace joaozinho_card_mariazinha_card_pedrinho_error_l199_199625

-- Define the card transformation function
def transform_card (number : ℕ) (color_adjustment : ℕ) : ℕ :=
  (number * 2 + 3) * 5 + color_adjustment

-- The proof problems
theorem joaozinho_card : transform_card 3 4 = 49 :=
by
  sorry

theorem mariazinha_card : ∃ number, ∃ color_adjustment, transform_card number color_adjustment = 76 :=
by
  sorry

theorem pedrinho_error : ∀ number color_adjustment, ¬ transform_card number color_adjustment = 61 :=
by
  sorry

end joaozinho_card_mariazinha_card_pedrinho_error_l199_199625


namespace final_price_including_tax_l199_199414

noncomputable def increasedPrice (originalPrice : ℝ) (increasePercentage : ℝ) : ℝ :=
  originalPrice + originalPrice * increasePercentage

noncomputable def discountedPrice (increasedPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  increasedPrice - increasedPrice * discountPercentage

noncomputable def finalPrice (discountedPrice : ℝ) (salesTax : ℝ) : ℝ :=
  discountedPrice + discountedPrice * salesTax

theorem final_price_including_tax :
  let originalPrice := 200
  let increasePercentage := 0.30
  let discountPercentage := 0.30
  let salesTax := 0.07
  let incPrice := increasedPrice originalPrice increasePercentage
  let disPrice := discountedPrice incPrice discountPercentage
  finalPrice disPrice salesTax = 194.74 :=
by
  simp [increasedPrice, discountedPrice, finalPrice]
  sorry

end final_price_including_tax_l199_199414


namespace sum_of_sequences_l199_199287

theorem sum_of_sequences :
  (1 + 11 + 21 + 31 + 41) + (9 + 19 + 29 + 39 + 49) = 250 := 
by 
  sorry

end sum_of_sequences_l199_199287


namespace remainder_when_divided_by_385_l199_199715

theorem remainder_when_divided_by_385 (x : ℤ)
  (h1 : 2 + x ≡ 4 [ZMOD 125])
  (h2 : 3 + x ≡ 9 [ZMOD 343])
  (h3 : 4 + x ≡ 25 [ZMOD 1331]) :
  x ≡ 307 [ZMOD 385] :=
sorry

end remainder_when_divided_by_385_l199_199715


namespace pages_left_l199_199719

variable (a b : ℕ)

theorem pages_left (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

end pages_left_l199_199719


namespace rachel_total_homework_pages_l199_199014

-- Define the conditions
def math_homework_pages : Nat := 10
def additional_reading_pages : Nat := 3

-- Define the proof goal
def total_homework_pages (math_pages reading_extra : Nat) : Nat :=
  math_pages + (math_pages + reading_extra)

-- The final statement with the expected result
theorem rachel_total_homework_pages : total_homework_pages math_homework_pages additional_reading_pages = 23 :=
by
  sorry

end rachel_total_homework_pages_l199_199014


namespace lcm_15_18_l199_199183

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end lcm_15_18_l199_199183


namespace find_n_l199_199687

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.sin (n * Real.pi / 180) = Real.sin (782 * Real.pi / 180)) :
  n = 62 ∨ n = -62 := 
sorry

end find_n_l199_199687


namespace moles_of_H2O_formed_l199_199545

-- Define the initial conditions
def molesNaOH : ℕ := 2
def molesHCl : ℕ := 2

-- Balanced chemical equation behavior definition
def reaction (x y : ℕ) : ℕ := min x y

-- Statement of the problem to prove
theorem moles_of_H2O_formed :
  reaction molesNaOH molesHCl = 2 := by
  sorry

end moles_of_H2O_formed_l199_199545


namespace min_reciprocal_sum_l199_199782

theorem min_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a) + (1 / b) ≥ 2 := by
  sorry

end min_reciprocal_sum_l199_199782


namespace odometer_problem_l199_199410

theorem odometer_problem (a b c : ℕ) (h₀ : a + b + c = 7) (h₁ : 1 ≤ a)
  (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) (h₅ : (c - a) % 20 = 0) : a^2 + b^2 + c^2 = 37 := 
  sorry

end odometer_problem_l199_199410


namespace daily_practice_hours_l199_199600

-- Define the conditions as given in the problem
def total_hours_practiced_this_week : ℕ := 36
def total_days_in_week : ℕ := 7
def days_could_not_practice : ℕ := 1
def actual_days_practiced := total_days_in_week - days_could_not_practice

-- State the theorem including the question and the correct answer, given the conditions
theorem daily_practice_hours :
  total_hours_practiced_this_week / actual_days_practiced = 6 := 
by
  sorry

end daily_practice_hours_l199_199600


namespace scientific_notation_of_604800_l199_199107

theorem scientific_notation_of_604800 : 604800 = 6.048 * 10^5 := 
sorry

end scientific_notation_of_604800_l199_199107


namespace total_oranges_in_box_l199_199291

def initial_oranges_in_box : ℝ := 55.0
def oranges_added_by_susan : ℝ := 35.0

theorem total_oranges_in_box :
  initial_oranges_in_box + oranges_added_by_susan = 90.0 := by
  sorry

end total_oranges_in_box_l199_199291


namespace distance_to_place_is_24_l199_199332

-- Definitions of the problem's conditions
def rowing_speed_still_water := 10    -- kmph
def current_velocity := 2             -- kmph
def round_trip_time := 5              -- hours

-- Effective speeds
def effective_speed_with_current := rowing_speed_still_water + current_velocity
def effective_speed_against_current := rowing_speed_still_water - current_velocity

-- Define the unknown distance D
variable (D : ℕ)

-- Define the times for each leg of the trip
def time_with_current := D / effective_speed_with_current
def time_against_current := D / effective_speed_against_current

-- The final theorem stating the round trip distance
theorem distance_to_place_is_24 :
  time_with_current + time_against_current = round_trip_time → D = 24 :=
by sorry

end distance_to_place_is_24_l199_199332


namespace solve_inequalities_l199_199981

theorem solve_inequalities (x : ℝ) :
    ((x / 2 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x))) ↔ (-6 ≤ x ∧ x < -3 / 2) :=
by
  sorry

end solve_inequalities_l199_199981


namespace range_of_a_l199_199320

theorem range_of_a (h : ¬ ∃ x : ℝ, x < 2023 ∧ x > a) : a ≥ 2023 := 
sorry

end range_of_a_l199_199320


namespace no_real_solutions_l199_199690

theorem no_real_solutions :
  ∀ y : ℝ, ( (-2 * y + 7)^2 + 2 = -2 * |y| ) → false := by
  sorry

end no_real_solutions_l199_199690


namespace gaokun_population_scientific_notation_l199_199357

theorem gaokun_population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (425000 = a * 10^n) ∧ (a = 4.25) ∧ (n = 5) :=
by
  sorry

end gaokun_population_scientific_notation_l199_199357


namespace oshea_bought_basil_seeds_l199_199101

-- Define the number of large and small planters and their capacities.
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30
def seeds_per_small_planter := 4

-- The theorem statement: Oshea bought 200 basil seeds
theorem oshea_bought_basil_seeds :
  large_planters * seeds_per_large_planter + small_planters * seeds_per_small_planter = 200 :=
by sorry

end oshea_bought_basil_seeds_l199_199101


namespace red_window_exchange_l199_199875

-- Defining the total transaction amount for online and offline booths
variables (x y : ℝ)

-- Defining conditions
def offlineMoreThanOnline (y x : ℝ) : Prop := y - 7 * x = 1.8
def averageTransactionDifference (y x : ℝ) : Prop := (y / 71) - (x / 44) = 0.3

-- The proof problem
theorem red_window_exchange (x y : ℝ) :
  offlineMoreThanOnline y x ∧ averageTransactionDifference y x := 
sorry

end red_window_exchange_l199_199875


namespace abs_sum_of_roots_l199_199554

theorem abs_sum_of_roots 
  (a b c m : ℤ) 
  (h1 : a + b + c = 0)
  (h2 : ab + bc + ca = -2023)
  : |a| + |b| + |c| = 102 := 
sorry

end abs_sum_of_roots_l199_199554


namespace sum_radical_conjugates_l199_199663

theorem sum_radical_conjugates : (5 - Real.sqrt 500) + (5 + Real.sqrt 500) = 10 :=
by
  sorry

end sum_radical_conjugates_l199_199663


namespace polynomial_roots_bounds_l199_199982

theorem polynomial_roots_bounds (p : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ (x1^4 + 3*p*x1^3 + x1^2 + 3*p*x1 + 1 = 0) ∧ (x2^4 + 3*p*x2^3 + x2^2 + 3*p*x2 + 1 = 0)) ↔ p ∈ Set.Iio (1 / 4) := by
sorry

end polynomial_roots_bounds_l199_199982


namespace price_reduction_correct_l199_199318

theorem price_reduction_correct (P : ℝ) : 
  let first_reduction := 0.92 * P
  let second_reduction := first_reduction * 0.90
  second_reduction = 0.828 * P := 
by 
  sorry

end price_reduction_correct_l199_199318


namespace bananas_and_cantaloupe_cost_l199_199426

noncomputable def prices (a b c d : ℕ) : Prop :=
  a + b + c + d = 40 ∧
  d = 3 * a ∧
  b = c - 2

theorem bananas_and_cantaloupe_cost (a b c d : ℕ) (h : prices a b c d) : b + c = 20 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  -- Using the given conditions:
  --     a + b + c + d = 40
  --     d = 3 * a
  --     b = c - 2
  -- We find that b + c = 20
  sorry

end bananas_and_cantaloupe_cost_l199_199426


namespace find_a_l199_199700

theorem find_a (a x : ℝ) (h : x = 1) (h_eq : 2 - 3 * (a + x) = 2 * x) : a = -1 := by
  sorry

end find_a_l199_199700


namespace solution_set_of_inequality_l199_199677

theorem solution_set_of_inequality (x : ℝ) : (x - 1) * (2 - x) > 0 ↔ 1 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l199_199677


namespace last_three_digits_of_7_pow_123_l199_199675

theorem last_three_digits_of_7_pow_123 : 7^123 % 1000 = 773 := 
by sorry

end last_three_digits_of_7_pow_123_l199_199675


namespace find_x_from_conditions_l199_199285

theorem find_x_from_conditions 
  (x y : ℕ) 
  (h1 : 1 ≤ x)
  (h2 : x ≤ 100)
  (h3 : 1 ≤ y)
  (h4 : y ≤ 100)
  (h5 : y > x)
  (h6 : (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x) 
  : x = 16 := 
sorry

end find_x_from_conditions_l199_199285


namespace shaded_area_eq_l199_199821

noncomputable def diameter_AB : ℝ := 6
noncomputable def diameter_BC : ℝ := 6
noncomputable def diameter_CD : ℝ := 6
noncomputable def diameter_DE : ℝ := 6
noncomputable def diameter_EF : ℝ := 6
noncomputable def diameter_FG : ℝ := 6
noncomputable def diameter_AG : ℝ := 6 * 6 -- 36

noncomputable def area_small_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

noncomputable def area_large_semicircle (d : ℝ) : ℝ :=
  (1/8) * Real.pi * d^2

theorem shaded_area_eq :
  area_large_semicircle diameter_AG + area_small_semicircle diameter_AB = 166.5 * Real.pi :=
  sorry

end shaded_area_eq_l199_199821


namespace anthony_transactions_more_percentage_l199_199612

def transactions (Mabel Anthony Cal Jade : ℕ) : Prop := 
  Mabel = 90 ∧ 
  Jade = 84 ∧ 
  Jade = Cal + 18 ∧ 
  Cal = (2 * Anthony) / 3 ∧ 
  Anthony = Mabel + (Mabel * 10 / 100)

theorem anthony_transactions_more_percentage (Mabel Anthony Cal Jade : ℕ) 
    (h : transactions Mabel Anthony Cal Jade) : 
  (Anthony = Mabel + (Mabel * 10 / 100)) :=
by 
  sorry

end anthony_transactions_more_percentage_l199_199612


namespace smallest_divisor_subtracted_l199_199068

theorem smallest_divisor_subtracted (a b d : ℕ) (h1: a = 899830) (h2: b = 6) (h3: a - b = 899824) (h4 : 6 < d) 
(h5 : d ∣ (a - b)) : d = 8 :=
by
  sorry

end smallest_divisor_subtracted_l199_199068


namespace hyperbola_eccentricity_sqrt_five_l199_199394

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1 where a > 0 and b > 0,
and its focus lies symmetrically with respect to the asymptote lines and on the hyperbola,
proves that the eccentricity of the hyperbola is sqrt(5). -/
theorem hyperbola_eccentricity_sqrt_five 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
  (c : ℝ) (h_focus : c^2 = 5 * a^2) : 
  (c / a = Real.sqrt 5) := sorry

end hyperbola_eccentricity_sqrt_five_l199_199394


namespace barrels_in_one_ton_l199_199045

-- Definitions (conditions)
def barrel_weight : ℕ := 10 -- in kilograms
def ton_in_kilograms : ℕ := 1000

-- Theorem Statement
theorem barrels_in_one_ton : ton_in_kilograms / barrel_weight = 100 :=
by
  sorry

end barrels_in_one_ton_l199_199045


namespace domain_of_f_log2x_is_0_4_l199_199960

def f : ℝ → ℝ := sorry

-- Given condition: domain of y = f(2x) is (-1, 1)
def dom_f_2x (x : ℝ) : Prop := -1 < 2 * x ∧ 2 * x < 1

-- Conclusion: domain of y = f(log_2 x) is (0, 4)
def dom_f_log2x (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem domain_of_f_log2x_is_0_4 (x : ℝ) :
  (dom_f_2x x) → (dom_f_log2x x) :=
by
  sorry

end domain_of_f_log2x_is_0_4_l199_199960


namespace not_all_perfect_squares_l199_199850

theorem not_all_perfect_squares (d : ℕ) (hd : 0 < d) :
  ¬ (∃ (x y z : ℕ), 2 * d - 1 = x^2 ∧ 5 * d - 1 = y^2 ∧ 13 * d - 1 = z^2) :=
by
  sorry

end not_all_perfect_squares_l199_199850


namespace annie_age_when_anna_three_times_current_age_l199_199798

theorem annie_age_when_anna_three_times_current_age
  (anna_age : ℕ) (annie_age : ℕ)
  (h1 : anna_age = 13)
  (h2 : annie_age = 3 * anna_age) :
  annie_age + 2 * anna_age = 65 :=
by
  sorry

end annie_age_when_anna_three_times_current_age_l199_199798


namespace smallest_n_divisible_11_remainder_1_l199_199272

theorem smallest_n_divisible_11_remainder_1 :
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 1) ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 11 = 0) ∧ 
    (∀ m : ℕ, (m % 2 = 1) ∧ (m % 3 = 1) ∧ (m % 4 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ (m % 11 = 0) → 2521 ≤ m) :=
by
  sorry

end smallest_n_divisible_11_remainder_1_l199_199272


namespace log_domain_l199_199249

theorem log_domain (x : ℝ) : x + 2 > 0 ↔ x ∈ Set.Ioi (-2) :=
by
  sorry

end log_domain_l199_199249


namespace condition_sufficient_but_not_necessary_l199_199206

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (x^3 > 8 → |x| > 2) ∧ (|x| > 2 → ¬ (x^3 ≤ 8 ∨ x^3 ≥ 8)) := by
  sorry

end condition_sufficient_but_not_necessary_l199_199206


namespace tree_height_end_of_third_year_l199_199354

theorem tree_height_end_of_third_year (h : ℝ) : 
    (∃ h0 h3 h6 : ℝ, 
      h3 = h0 * 3^3 ∧ 
      h6 = h3 * 2^3 ∧ 
      h6 = 1458) → h3 = 182.25 :=
by sorry

end tree_height_end_of_third_year_l199_199354


namespace icing_time_is_30_l199_199861

def num_batches : Nat := 4
def baking_time_per_batch : Nat := 20
def total_time : Nat := 200

def baking_time_total : Nat := num_batches * baking_time_per_batch
def icing_time_total : Nat := total_time - baking_time_total
def icing_time_per_batch : Nat := icing_time_total / num_batches

theorem icing_time_is_30 :
  icing_time_per_batch = 30 := by
  sorry

end icing_time_is_30_l199_199861


namespace shorter_side_length_l199_199433

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 50) (h2 : a * b = 126) : b = 9 :=
sorry

end shorter_side_length_l199_199433


namespace quadrilateral_area_inequality_l199_199192

theorem quadrilateral_area_inequality
  (a b c d S : ℝ)
  (hS : 0 ≤ S)
  (h : S = (a + b) / 4 * (c + d) / 4)
  : S ≤ (a + b) / 4 * (c + d) / 4 := by
  sorry

end quadrilateral_area_inequality_l199_199192


namespace evaluate_expression_l199_199125

noncomputable def expr : ℚ := (3 ^ 512 + 7 ^ 513) ^ 2 - (3 ^ 512 - 7 ^ 513) ^ 2
noncomputable def k : ℚ := 28 * 2.1 ^ 512

theorem evaluate_expression : expr = k * 10 ^ 513 :=
by
  sorry

end evaluate_expression_l199_199125


namespace prob1_prob2_prob3_l199_199973

-- Problem (1)
theorem prob1 (a b : ℝ) :
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4) ∧ (2 * (a / 4 - 1) + (b / 3 + 2) = 5) →
  a = 12 ∧ b = -3 :=
by { sorry }

-- Problem (2)
theorem prob2 (m n x y a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (x = 10) ∧ (y = 6) ∧ 
  (5 * a₁ * (m - 3) + 3 * b₁ * (n + 2) = c₁) ∧ (5 * a₂ * (m - 3) + 3 * b₂ * (n + 2) = c₂) →
  (m = 5) ∧ (n = 0) :=
by { sorry }

-- Problem (3)
theorem prob3 (x y z : ℝ) :
  (3 * x - 2 * z + 12 * y = 47) ∧ (2 * x + z + 8 * y = 36) → z = 2 :=
by { sorry }

end prob1_prob2_prob3_l199_199973


namespace volume_inequality_find_min_k_l199_199121

noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

noncomputable def cylinder_volume (R h : ℝ) : ℝ :=
    let r := (R * h) / Real.sqrt (R^2 + h^2)
    Real.pi * r^2 * h

noncomputable def k_value (R h : ℝ) : ℝ := (R^2 + h^2) / (3 * h^2)

theorem volume_inequality (R h : ℝ) (h_pos : R > 0 ∧ h > 0) : 
    cone_volume R h ≠ cylinder_volume R h := by sorry

theorem find_min_k (R h : ℝ) (h_pos : R > 0 ∧ h > 0) (k : ℝ) :
    cone_volume R h = k * cylinder_volume R h → k = (R^2 + h^2) / (3 * h^2) := by sorry

end volume_inequality_find_min_k_l199_199121


namespace olympic_volunteers_selection_l199_199947

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem olympic_volunteers_selection :
  (choose 4 3 * choose 3 1) + (choose 4 2 * choose 3 2) + (choose 4 1 * choose 3 3) = 34 := 
by
  sorry

end olympic_volunteers_selection_l199_199947


namespace ordered_triples_lcm_l199_199521

def lcm_equal (a b n : ℕ) : Prop :=
  a * b / (Nat.gcd a b) = n

theorem ordered_triples_lcm :
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z → 
  lcm_equal x y 48 → lcm_equal x z 900 → lcm_equal y z 180 →
  false :=
by sorry

end ordered_triples_lcm_l199_199521


namespace compute_expression_l199_199662

theorem compute_expression :
  (5 + 7)^2 + 5^2 + 7^2 = 218 :=
by
  sorry

end compute_expression_l199_199662


namespace wrongly_noted_mark_l199_199803

theorem wrongly_noted_mark (x : ℕ) (h_wrong_avg : (30 : ℕ) * 100 = 3000)
    (h_correct_avg : (30 : ℕ) * 98 = 2940) (h_correct_sum : 3000 - x + 10 = 2940) : 
    x = 70 := by
  sorry

end wrongly_noted_mark_l199_199803


namespace pastries_selection_l199_199147

/--
Clara wants to purchase six pastries from an ample supply of five types: muffins, eclairs, croissants, scones, and turnovers. 
Prove that there are 210 possible selections using the stars and bars theorem.
-/
theorem pastries_selection : ∃ (selections : ℕ), selections = (Nat.choose (6 + 5 - 1) (5 - 1)) ∧ selections = 210 := by
  sorry

end pastries_selection_l199_199147


namespace three_digit_numbers_count_correct_l199_199938

def digits : List ℕ := [2, 3, 4, 5, 5, 5, 6, 6]

def three_digit_numbers_count (d : List ℕ) : ℕ := 
  -- To be defined: Full implementation for counting matching three-digit numbers
  sorry

theorem three_digit_numbers_count_correct :
  three_digit_numbers_count digits = 85 :=
sorry

end three_digit_numbers_count_correct_l199_199938


namespace clara_meeting_time_l199_199119

theorem clara_meeting_time (d T : ℝ) :
  (d / 20 = T - 0.5) →
  (d / 12 = T + 0.5) →
  (d / T = 15) :=
by
  intros h1 h2
  sorry

end clara_meeting_time_l199_199119


namespace cost_of_items_l199_199823

theorem cost_of_items {x y z : ℕ} (h1 : x + 3 * y + 2 * z = 98)
                      (h2 : 3 * x + y = 5 * z - 36)
                      (even_x : x % 2 = 0) :
  x = 4 ∧ y = 22 ∧ z = 14 := 
by
  sorry

end cost_of_items_l199_199823


namespace initial_birds_l199_199510

-- Define the initial number of birds (B) and the fact that 13 more birds flew up to the tree
-- Define that the total number of birds after 13 more birds joined is 42
theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
by
  sorry

end initial_birds_l199_199510


namespace part1_solution_part2_solution_l199_199450

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l199_199450


namespace brown_dog_count_l199_199781

theorem brown_dog_count:
  ∀ (T L N : ℕ), T = 45 → L = 36 → N = 8 → (T - N - (T - L - N) = 37) :=
by
  intros T L N hT hL hN
  sorry

end brown_dog_count_l199_199781


namespace find_quotient_l199_199306

-- Define the problem variables and conditions
def larger_number : ℕ := 1620
def smaller_number : ℕ := larger_number - 1365
def remainder : ℕ := 15

-- Define the proof problem
theorem find_quotient :
  larger_number = smaller_number * 6 + remainder :=
sorry

end find_quotient_l199_199306


namespace circle_area_l199_199971

-- Given conditions
variables {BD AC : ℝ} (BD_pos : BD = 6) (AC_pos : AC = 12)
variables {R : ℝ} (R_pos : R = 15 / 2)

-- Prove that the area of the circles is \(\frac{225}{4}\pi\)
theorem circle_area (BD_pos : BD = 6) (AC_pos : AC = 12) (R : ℝ) (R_pos : R = 15 / 2) : 
        ∃ S, S = (225 / 4) * Real.pi := 
by sorry

end circle_area_l199_199971


namespace smallest_n_for_constant_term_l199_199846

theorem smallest_n_for_constant_term :
  ∃ (n : ℕ), (n > 0) ∧ ((∃ (r : ℕ), 2 * n = 5 * r) ∧ (∀ (m : ℕ), m > 0 → (∃ (r' : ℕ), 2 * m = 5 * r') → n ≤ m)) ∧ n = 5 :=
by
  sorry

end smallest_n_for_constant_term_l199_199846


namespace number_of_pairs_l199_199594

theorem number_of_pairs (H : ∀ x y : ℕ , 0 < x → 0 < y → x < y → 2 * x * y / (x + y) = 4 ^ 15) :
  ∃ n : ℕ, n = 29 :=
by
  sorry

end number_of_pairs_l199_199594


namespace mark_gpa_probability_l199_199242

theorem mark_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let GPA_required := 3.5
  let total_subjects := 4
  let total_points_required := GPA_required * total_subjects
  -- Points from guaranteed A's in Mathematics and Science
  let guaranteed_points := 8
  -- Required points from Literature and History
  let points_needed := total_points_required - guaranteed_points
  -- Probabilities for grades in Literature
  let prob_A_Lit := 1 / 3
  let prob_B_Lit := 1 / 3
  let prob_C_Lit := 1 / 3
  -- Probabilities for grades in History
  let prob_A_Hist := 1 / 5
  let prob_B_Hist := 1 / 4
  let prob_C_Hist := 11 / 20
  -- Combinations of grades to achieve the required points
  let prob_two_As := prob_A_Lit * prob_A_Hist
  let prob_A_Lit_B_Hist := prob_A_Lit * prob_B_Hist
  let prob_B_Lit_A_Hist := prob_B_Lit * prob_A_Hist
  let prob_two_Bs := prob_B_Lit * prob_B_Hist
  -- Total probability of achieving at least the required GPA
  let total_probability := prob_two_As + prob_A_Lit_B_Hist + prob_B_Lit_A_Hist + prob_two_Bs
  total_probability = 3 / 10 := sorry

end mark_gpa_probability_l199_199242


namespace determine_pairs_l199_199344

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem determine_pairs (n p : ℕ) (hn_pos : 0 < n) (hp_prime : is_prime p) (hn_le_2p : n ≤ 2 * p) (divisibility : n^p - 1 ∣ (p - 1)^n + 1):
  (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
by
  sorry

end determine_pairs_l199_199344


namespace nine_pow_n_sub_one_l199_199475

theorem nine_pow_n_sub_one (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ (p1 p2 p3 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (9^n - 1) = p1 * p2 * p3 ∧ (p1 = 61 ∨ p2 = 61 ∨ p3 = 61)) : 9^n - 1 = 59048 := 
sorry

end nine_pow_n_sub_one_l199_199475


namespace intersection_complement_l199_199280

def M : Set ℝ := { x | x^2 - x - 6 ≥ 0 }
def N : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def neg_R (A : Set ℝ) : Set ℝ := { x | x ∉ A }

theorem intersection_complement (N : Set ℝ) (M : Set ℝ) :
  N ∩ (neg_R M) = { x | -2 < x ∧ x ≤ 1 } := 
by {
  -- Proof goes here
  sorry
}

end intersection_complement_l199_199280


namespace matt_worked_more_on_wednesday_l199_199087

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end matt_worked_more_on_wednesday_l199_199087


namespace sin_phi_value_l199_199210

theorem sin_phi_value 
  (φ α : ℝ)
  (hφ : φ = 2 * α)
  (hα1 : Real.sin α = (Real.sqrt 5) / 5)
  (hα2 : Real.cos α = 2 * (Real.sqrt 5) / 5) 
  : Real.sin φ = 4 / 5 := 
by 
  sorry

end sin_phi_value_l199_199210


namespace mul_97_103_l199_199310

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l199_199310


namespace equilateral_triangle_perimeter_isosceles_triangle_leg_length_l199_199584

-- Definitions for equilateral triangle problem
def side_length_equilateral : ℕ := 12
def perimeter_equilateral := side_length_equilateral * 3

-- Definitions for isosceles triangle problem
def perimeter_isosceles : ℕ := 72
def base_length_isosceles : ℕ := 28
def leg_length_isosceles := (perimeter_isosceles - base_length_isosceles) / 2

-- Theorem statement
theorem equilateral_triangle_perimeter : perimeter_equilateral = 36 := 
by
  sorry

theorem isosceles_triangle_leg_length : leg_length_isosceles = 22 := 
by
  sorry

end equilateral_triangle_perimeter_isosceles_triangle_leg_length_l199_199584


namespace total_homework_problems_l199_199763

-- Define the conditions as Lean facts
def finished_problems : ℕ := 45
def ratio_finished_to_left := (9, 4)
def problems_left (L : ℕ) := finished_problems * ratio_finished_to_left.2 = L * ratio_finished_to_left.1 

-- State the theorem
theorem total_homework_problems (L : ℕ) (h : problems_left L) : finished_problems + L = 65 :=
sorry

end total_homework_problems_l199_199763


namespace find_phi_l199_199987

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem find_phi (phi : ℝ) (h_shift : ∀ x : ℝ, f (x + phi) = f (-x - phi)) : 
  phi = Real.pi / 8 :=
  sorry

end find_phi_l199_199987


namespace mean_weight_of_soccer_team_l199_199104

-- Define the weights as per the conditions
def weights : List ℕ := [64, 68, 71, 73, 76, 76, 77, 78, 80, 82, 85, 87, 89, 89]

-- Define the total weight
def total_weight : ℕ := 64 + 68 + 71 + 73 + 76 + 76 + 77 + 78 + 80 + 82 + 85 + 87 + 89 + 89

-- Define the number of players
def number_of_players : ℕ := 14

-- Calculate the mean weight
noncomputable def mean_weight : ℚ := total_weight / number_of_players

-- The proof problem statement
theorem mean_weight_of_soccer_team : mean_weight = 75.357 := by
  -- This is where the proof would go.
  sorry

end mean_weight_of_soccer_team_l199_199104


namespace problem_conditions_l199_199917

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (1 + x)
else if -1 < x ∧ x < 0 then x / (1 - x)
else 0

theorem problem_conditions (a b : ℝ) (x : ℝ) :
  (∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = (-a * x - b) / (1 + x)) ∧ 
  (f (1 / 2) = 1 / 3) →
  (a = -1) ∧ (b = 0) ∧
  (∀ x :  ℝ, -1 < x ∧ x < 1 → 
    (if 0 ≤ x ∧ x < 1 then f x = x / (1 + x) else if -1 < x ∧ x < 0 then f x = x / (1 - x) else True)) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2) ∧ 
  (∀ x : ℝ, f (x - 1) + f x > 0 → (1 / 2 < x ∧ x < 1)) :=
by
  sorry

end problem_conditions_l199_199917


namespace mark_brings_in_148_cans_l199_199233

-- Define the given conditions
variable (R : ℕ) (Mark Jaydon Sophie : ℕ)

-- Conditions
def jaydon_cans := 2 * R + 5
def mark_cans := 4 * jaydon_cans
def unit_ratio := mark_cans / 4
def sophie_cans := 2 * unit_ratio

-- Condition: Total cans
def total_cans := mark_cans + jaydon_cans + sophie_cans

-- Condition: Each contributes at least 5 cans
axiom each_contributes_at_least_5 : R ≥ 5

-- Condition: Total cans is an odd number not less than 250
axiom total_odd_not_less_than_250 : ∃ k : ℕ, total_cans = 2 * k + 1 ∧ total_cans ≥ 250

-- Theorem: Prove Mark brings in 148 cans under the conditions
theorem mark_brings_in_148_cans (h : R = 16) : mark_cans = 148 :=
by sorry

end mark_brings_in_148_cans_l199_199233


namespace max_profit_at_grade_5_l199_199815

-- Defining the conditions
def profit_per_item (x : ℕ) : ℕ :=
  4 * (x - 1) + 8

def production_count (x : ℕ) : ℕ := 
  60 - 6 * (x - 1)

def daily_profit (x : ℕ) : ℕ :=
  profit_per_item x * production_count x

-- The grade range
def grade_range (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

-- Prove that the grade that maximizes the profit is 5
theorem max_profit_at_grade_5 : (1 ≤ x ∧ x ≤ 10) → daily_profit x ≤ daily_profit 5 :=
sorry

end max_profit_at_grade_5_l199_199815


namespace solve_for_x_l199_199128

theorem solve_for_x : ∀ x : ℝ, (x - 5) ^ 3 = (1 / 27)⁻¹ → x = 8 := by
  intro x
  intro h
  sorry

end solve_for_x_l199_199128


namespace chocolate_cookies_initial_count_l199_199066

theorem chocolate_cookies_initial_count
  (andy_ate : ℕ) (brother : ℕ) (friends_each : ℕ) (num_friends : ℕ)
  (team_members : ℕ) (first_share : ℕ) (common_diff : ℕ)
  (last_member_share : ℕ) (total_sum_team : ℕ)
  (total_cookies : ℕ) :
  andy_ate = 4 →
  brother = 6 →
  friends_each = 2 →
  num_friends = 3 →
  team_members = 10 →
  first_share = 2 →
  common_diff = 2 →
  last_member_share = first_share + (team_members - 1) * common_diff →
  total_sum_team = team_members / 2 * (first_share + last_member_share) →
  total_cookies = andy_ate + brother + (friends_each * num_friends) + total_sum_team →
  total_cookies = 126 :=
by
  intros ha hb hf hn ht hf1 hc hl hs ht
  sorry

end chocolate_cookies_initial_count_l199_199066


namespace combined_length_of_trains_l199_199148

def length_of_train (speed_kmhr : ℕ) (time_sec : ℕ) : ℚ :=
  (speed_kmhr : ℚ) / 3600 * time_sec

theorem combined_length_of_trains :
  let L1 := length_of_train 300 33
  let L2 := length_of_train 250 44
  let L3 := length_of_train 350 28
  L1 + L2 + L3 = 8.52741 := by
  sorry

end combined_length_of_trains_l199_199148


namespace simplify_expression_l199_199791

theorem simplify_expression (p q r : ℝ) (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) :
  ( ( (p - 7) / (9 - r) ) * ( (q - 8) / (7 - p) ) * ( (r - 9) / (8 - q) ) ) = -1 := 
by 
  sorry

end simplify_expression_l199_199791


namespace range_of_a_l199_199643

noncomputable def in_range (a : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∨ (a ≥ 1)

theorem range_of_a (a : ℝ) (p q : Prop) (h1 : p ↔ (0 < a ∧ a < 1)) (h2 : q ↔ (a ≥ 1 / 2)) (h3 : p ∨ q) (h4 : ¬ (p ∧ q)) :
  in_range a :=
by
  sorry

end range_of_a_l199_199643


namespace uncle_bruce_dough_weight_l199_199050

-- Definitions based on the conditions
variable {TotalChocolate : ℕ} (h1 : TotalChocolate = 13)
variable {ChocolateLeftOver : ℕ} (h2 : ChocolateLeftOver = 4)
variable {ChocolatePercentage : ℝ} (h3 : ChocolatePercentage = 0.2) 
variable {WeightOfDough : ℝ}

-- Target statement expressing the final question and answer
theorem uncle_bruce_dough_weight 
  (h1 : TotalChocolate = 13) 
  (h2 : ChocolateLeftOver = 4) 
  (h3 : ChocolatePercentage = 0.2) : 
  WeightOfDough = 36 := by
  sorry

end uncle_bruce_dough_weight_l199_199050


namespace max_r_value_l199_199343

theorem max_r_value (r : ℕ) (hr : r ≥ 2)
  (m n : Fin r → ℤ)
  (h : ∀ i j : Fin r, i < j → |m i * n j - m j * n i| = 1) :
  r ≤ 3 := 
sorry

end max_r_value_l199_199343


namespace minimum_sum_of_dimensions_l199_199135

-- Define the problem as a Lean 4 statement
theorem minimum_sum_of_dimensions (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 2184) : 
  x + y + z = 36 := 
sorry

end minimum_sum_of_dimensions_l199_199135


namespace probability_at_least_one_die_shows_three_l199_199897

theorem probability_at_least_one_die_shows_three : 
  let outcomes := 36
  let not_three_outcomes := 25
  (outcomes - not_three_outcomes) / outcomes = 11 / 36 := sorry

end probability_at_least_one_die_shows_three_l199_199897


namespace find_monthly_salary_l199_199117

-- Definitions based on the conditions
def initial_saving_rate : ℝ := 0.25
def initial_expense_rate : ℝ := 1 - initial_saving_rate
def expense_increase_rate : ℝ := 1.25
def final_saving : ℝ := 300

-- Theorem: Prove the man's monthly salary
theorem find_monthly_salary (S : ℝ) (h1 : initial_saving_rate = 0.25)
  (h2 : initial_expense_rate = 0.75) (h3 : expense_increase_rate = 1.25)
  (h4 : final_saving = 300) : S = 4800 :=
by
  sorry

end find_monthly_salary_l199_199117


namespace total_dreams_correct_l199_199040

def dreams_per_day : Nat := 4
def days_in_year : Nat := 365
def current_year_dreams : Nat := dreams_per_day * days_in_year
def last_year_dreams : Nat := 2 * current_year_dreams
def total_dreams : Nat := current_year_dreams + last_year_dreams

theorem total_dreams_correct : total_dreams = 4380 :=
by
  -- prime verification needed here
  sorry

end total_dreams_correct_l199_199040


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l199_199187

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l199_199187


namespace steve_halfway_longer_than_danny_l199_199678

theorem steve_halfway_longer_than_danny :
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  (T_s / 2) - (T_d / 2) = 15.5 :=
by
  let T_d : Float := 31
  let T_s : Float := 2 * T_d
  show (T_s / 2) - (T_d / 2) = 15.5
  sorry

end steve_halfway_longer_than_danny_l199_199678


namespace find_k_l199_199735

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l199_199735


namespace quadratic_root_property_l199_199530

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l199_199530


namespace solution_set_of_quadratic_inequality_l199_199703

theorem solution_set_of_quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := 
by
  sorry

end solution_set_of_quadratic_inequality_l199_199703


namespace smallest_n_divisibility_l199_199132

theorem smallest_n_divisibility :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (m > 0) ∧ (72 ∣ m^2) ∧ (1728 ∣ m^3) → (n ≤ m)) ∧
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) :=
by
  sorry

end smallest_n_divisibility_l199_199132


namespace misha_contributes_l199_199870

noncomputable def misha_contribution (k l m : ℕ) : ℕ :=
  if h : k + l + m = 6 ∧ 2 * k ≤ l + m ∧ 2 * l ≤ k + m ∧ 2 * m ≤ k + l ∧ k ≤ 2 ∧ l ≤ 2 ∧ m ≤ 2 then
    2
  else
    0 -- This is a default value; the actual proof will check for exact solution.

theorem misha_contributes (k l m : ℕ) (h1 : k + l + m = 6)
    (h2 : 2 * k ≤ l + m) (h3 : 2 * l ≤ k + m) (h4 : 2 * m ≤ k + l)
    (h5 : k ≤ 2) (h6 : l ≤ 2) (h7 : m ≤ 2) : m = 2 := by
  sorry

end misha_contributes_l199_199870


namespace kim_shirts_left_l199_199064

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l199_199064


namespace find_y_z_l199_199602

theorem find_y_z 
  (y z : ℝ) 
  (h_mean : (8 + 15 + 22 + 5 + y + z) / 6 = 12) 
  (h_diff : y - z = 6) : 
  y = 14 ∧ z = 8 := 
by
  sorry

end find_y_z_l199_199602


namespace original_wire_length_l199_199522

theorem original_wire_length (side_len total_area : ℕ) (h1 : side_len = 2) (h2 : total_area = 92) :
  (total_area / (side_len * side_len)) * (4 * side_len) = 184 := 
by
  sorry

end original_wire_length_l199_199522


namespace paul_score_higher_by_26_l199_199837

variable {R : Type} [LinearOrderedField R]

variables (A1 A2 A3 P1 P2 P3 : R)

-- hypotheses
variable (h1 : A1 = P1 + 10)
variable (h2 : A2 = P2 + 4)
variable (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)

-- goal
theorem paul_score_higher_by_26 : P3 - A3 = 26 := by
  sorry

end paul_score_higher_by_26_l199_199837


namespace sebastian_total_payment_l199_199541

theorem sebastian_total_payment 
  (cost_per_ticket : ℕ) (number_of_tickets : ℕ) (service_fee : ℕ) (total_paid : ℕ)
  (h1 : cost_per_ticket = 44)
  (h2 : number_of_tickets = 3)
  (h3 : service_fee = 18)
  (h4 : total_paid = (number_of_tickets * cost_per_ticket) + service_fee) :
  total_paid = 150 :=
by
  sorry

end sebastian_total_payment_l199_199541


namespace find_y_intercept_l199_199668

theorem find_y_intercept (m : ℝ) (x_intercept: ℝ × ℝ) : (x_intercept.snd = 0) → (x_intercept = (-4, 0)) → m = 3 → (0, m * 4 - m * (-4)) = (0, 12) :=
by
  sorry

end find_y_intercept_l199_199668


namespace number_of_fiction_books_l199_199799

theorem number_of_fiction_books (F NF : ℕ) (h1 : F + NF = 52) (h2 : NF = 7 * F / 6) : F = 24 := 
by
  sorry

end number_of_fiction_books_l199_199799


namespace volume_of_rectangular_prism_l199_199111

-- Define the conditions
def side_of_square : ℕ := 35
def area_of_square : ℕ := 1225
def radius_of_sphere : ℕ := side_of_square
def length_of_prism : ℕ := (2 * radius_of_sphere) / 5
def width_of_prism : ℕ := 10
variable (h : ℕ) -- height of the prism

-- The theorem to prove
theorem volume_of_rectangular_prism :
  area_of_square = side_of_square * side_of_square →
  length_of_prism = (2 * radius_of_sphere) / 5 →
  radius_of_sphere = side_of_square →
  volume_of_prism = (length_of_prism * width_of_prism * h)
  → volume_of_prism = 140 * h :=
by sorry

end volume_of_rectangular_prism_l199_199111


namespace find_c_plus_d_l199_199708

noncomputable def f (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then 2 * c * x + d else 9 - 2 * x

theorem find_c_plus_d (c d : ℝ) (h : ∀ x : ℝ, f c d (f c d x) = x) : c + d = 4.25 :=
by
  sorry

end find_c_plus_d_l199_199708


namespace gcd_1458_1479_l199_199958

def a : ℕ := 1458
def b : ℕ := 1479
def gcd_ab : ℕ := 21

theorem gcd_1458_1479 : Nat.gcd a b = gcd_ab := sorry

end gcd_1458_1479_l199_199958


namespace simplify_and_evaluate_expression_l199_199231

theorem simplify_and_evaluate_expression (a b : ℝ) (h₁ : a = 2 + Real.sqrt 3) (h₂ : b = 2 - Real.sqrt 3) :
  (a^2 - b^2) / a / (a - (2 * a * b - b^2) / a) = 2 * Real.sqrt 3 / 3 :=
by
  -- Proof to be provided
  sorry

end simplify_and_evaluate_expression_l199_199231


namespace trig_identity_l199_199464

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π + α)) / (Real.sin (π / 2 - α)) = -2 :=
by
  sorry

end trig_identity_l199_199464


namespace kyle_age_l199_199265

theorem kyle_age :
  ∃ (kyle shelley julian frederick tyson casey : ℕ),
    shelley = kyle - 3 ∧ 
    shelley = julian + 4 ∧
    julian = frederick - 20 ∧
    frederick = 2 * tyson ∧
    tyson = 2 * casey ∧
    casey = 15 ∧ 
    kyle = 47 :=
by
  sorry

end kyle_age_l199_199265


namespace player_A_wins_even_n_l199_199942

theorem player_A_wins_even_n (n : ℕ) (hn : n > 0) (even_n : Even n) :
  ∃ strategy_A : ℕ → Bool, 
    ∀ (P Q : ℕ), P % 2 = 0 → (Q + P) % 2 = 0 :=
by 
  sorry

end player_A_wins_even_n_l199_199942


namespace sum_floor_ceil_eq_seven_l199_199611

theorem sum_floor_ceil_eq_seven (x : ℝ) 
  (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 := 
sorry

end sum_floor_ceil_eq_seven_l199_199611


namespace polynomial_has_exactly_one_real_root_l199_199459

theorem polynomial_has_exactly_one_real_root :
  ∀ (x : ℝ), (2007 * x^3 + 2006 * x^2 + 2005 * x = 0) → x = 0 :=
by
  sorry

end polynomial_has_exactly_one_real_root_l199_199459


namespace sin_10pi_over_3_l199_199841

theorem sin_10pi_over_3 : Real.sin (10 * Real.pi / 3) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_10pi_over_3_l199_199841


namespace find_number_l199_199653

theorem find_number :
  ∃ x : ℕ, (x / 5 = 80 + x / 6) ∧ x = 2400 := 
by 
  sorry

end find_number_l199_199653


namespace one_fourth_of_8_point_4_is_21_over_10_l199_199865

theorem one_fourth_of_8_point_4_is_21_over_10 : (8.4 / 4 : ℚ) = 21 / 10 := 
by
  sorry

end one_fourth_of_8_point_4_is_21_over_10_l199_199865


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l199_199182

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l199_199182


namespace rectangle_perimeter_gt_16_l199_199919

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_area_gt_perim : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
by
  sorry

end rectangle_perimeter_gt_16_l199_199919


namespace table_sale_price_percentage_l199_199556

theorem table_sale_price_percentage (W : ℝ) : 
  let S := 1.4 * W
  let P := 0.65 * S
  P = 0.91 * W :=
by
  sorry

end table_sale_price_percentage_l199_199556


namespace symmetric_point_x_axis_l199_199532

theorem symmetric_point_x_axis (x y : ℝ) (p : Prod ℝ ℝ) (hx : p = (x, y)) :
  (x, -y) = (1, -2) ↔ (x, y) = (1, 2) :=
by
  sorry

end symmetric_point_x_axis_l199_199532


namespace grouping_schemes_count_l199_199555

/-- Number of possible grouping schemes where each group consists
    of either 2 or 3 students and the total number of students is 25 is 4.-/
theorem grouping_schemes_count : ∃ (x y : ℕ), 2 * x + 3 * y = 25 ∧ 
  (x = 11 ∧ y = 1 ∨ x = 8 ∧ y = 3 ∨ x = 5 ∧ y = 5 ∨ x = 2 ∧ y = 7) :=
sorry

end grouping_schemes_count_l199_199555


namespace expression_1_expression_2_expression_3_expression_4_l199_199736

section problem1

variable {x : ℝ}

theorem expression_1:
  (x^2 - 1 + x)*(x^2 - 1 + 3*x) + x^2  = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 :=
sorry

end problem1

section problem2

variable {x a : ℝ}

theorem expression_2:
  (x - a)^4 + 4*a^4 = (x^2 + a^2)*(x^2 - 4*a*x + 5*a^2) :=
sorry

end problem2

section problem3

variable {a : ℝ}

theorem expression_3:
  (a + 1)^4 + 2*(a + 1)^3 + a*(a + 2) = (a + 1)^4 + 2*(a + 1)^3 + 1 :=
sorry

end problem3

section problem4

variable {p : ℝ}

theorem expression_4:
  (p + 2)^4 + 2*(p^2 - 4)^2 + (p - 2)^4 = 4*p^4 :=
sorry

end problem4

end expression_1_expression_2_expression_3_expression_4_l199_199736


namespace vasya_made_mistake_l199_199588

theorem vasya_made_mistake : 
  ∀ (total_digits : ℕ), 
    total_digits = 301 → 
    ¬∃ (n : ℕ), 
      (n ≤ 9 ∧ total_digits = (n * 1)) ∨ 
      (10 ≤ n ∧ n ≤ 99 ∧ total_digits = (9 * 1) + ((n - 9) * 2)) ∨ 
      (100 ≤ n ∧ total_digits = (9 * 1) + (90 * 2) + ((n - 99) * 3)) := 
by 
  sorry

end vasya_made_mistake_l199_199588


namespace tim_total_spending_l199_199259

def lunch_cost : ℝ := 50.50
def dessert_cost : ℝ := 8.25
def beverage_cost : ℝ := 3.75
def lunch_discount : ℝ := 0.10
def dessert_tax : ℝ := 0.07
def beverage_tax : ℝ := 0.05
def lunch_tip_rate : ℝ := 0.20
def other_items_tip_rate : ℝ := 0.15

def total_spending : ℝ := 
  let lunch_after_discount := lunch_cost * (1 - lunch_discount)
  let dessert_after_tax := dessert_cost * (1 + dessert_tax)
  let beverage_after_tax := beverage_cost * (1 + beverage_tax)
  let tip_on_lunch := lunch_after_discount * lunch_tip_rate
  let combined_other_items := dessert_after_tax + beverage_after_tax
  let tip_on_other_items := combined_other_items * other_items_tip_rate
  lunch_after_discount + dessert_after_tax + beverage_after_tax + tip_on_lunch + tip_on_other_items

theorem tim_total_spending :
  total_spending = 69.23 :=
by
  sorry

end tim_total_spending_l199_199259


namespace Mikaela_initially_planned_walls_l199_199676

/-- 
Mikaela bought 16 containers of paint to cover a certain number of equally-sized walls in her bathroom.
At the last minute, she decided to put tile on one wall and paint flowers on the ceiling with one 
container of paint instead. She had 3 containers of paint left over. 
Prove she initially planned to paint 13 walls.
-/
theorem Mikaela_initially_planned_walls
  (PaintContainers : ℕ)
  (CeilingPaint : ℕ)
  (LeftOverPaint : ℕ)
  (TiledWalls : ℕ) : PaintContainers = 16 → CeilingPaint = 1 → LeftOverPaint = 3 → TiledWalls = 1 → 
    (PaintContainers - CeilingPaint - LeftOverPaint + TiledWalls = 13) :=
by
  -- Given conditions:
  intros h1 h2 h3 h4
  -- Proof goes here.
  sorry

end Mikaela_initially_planned_walls_l199_199676


namespace prob_exactly_M_laws_expected_laws_included_l199_199996

noncomputable def prob_of_exactly_M_laws (K N M : ℕ) (p : ℝ) : ℝ :=
  let q := 1 - (1 - p)^N
  (Nat.choose K M) * q^M * (1 - q)^(K - M)

noncomputable def expected_num_of_laws (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

-- Part (a): Prove that the probability of exactly M laws being included is as follows
theorem prob_exactly_M_laws (K N M : ℕ) (p : ℝ) :
  prob_of_exactly_M_laws K N M p =
    (Nat.choose K M) * (1 - (1 - p)^N)^M * ((1 - (1 - p)^N)^(K - M)) :=
sorry

-- Part (b): Prove that the expected number of laws included is as follows
theorem expected_laws_included (K N : ℕ) (p : ℝ) :
  expected_num_of_laws K N p =
    K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_expected_laws_included_l199_199996


namespace Ivy_cupcakes_l199_199340

theorem Ivy_cupcakes (M : ℕ) (h1 : M + (M + 15) = 55) : M = 20 :=
by
  sorry

end Ivy_cupcakes_l199_199340


namespace part_a_part_b_l199_199920

-- Define n_mid_condition
def n_mid_condition (n : ℕ) : Prop := n % 2 = 1 ∧ n ∣ 2023^n - 1

-- Part a:
theorem part_a : ∃ (k₁ k₂ : ℕ), k₁ = 3 ∧ k₂ = 9 ∧ n_mid_condition k₁ ∧ n_mid_condition k₂ := by
  sorry

-- Part b:
theorem part_b : ∀ k, k ≥ 1 → n_mid_condition (3^k) := by
  sorry

end part_a_part_b_l199_199920


namespace alpha_necessary_not_sufficient_for_beta_l199_199396

def alpha (x : ℝ) : Prop := x^2 = 4
def beta (x : ℝ) : Prop := x = 2

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x : ℝ, beta x → alpha x) ∧ ¬(∀ x : ℝ, alpha x → beta x) :=
by
  sorry

end alpha_necessary_not_sufficient_for_beta_l199_199396


namespace solution_set_f_leq_g_range_of_a_l199_199589

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := x + 2

theorem solution_set_f_leq_g (x : ℝ) : f x 1 ≤ g x ↔ (0 ≤ x ∧ x ≤ 2 / 3) := by
  sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ g x) : 2 ≤ a := by
  sorry

end solution_set_f_leq_g_range_of_a_l199_199589


namespace frames_per_page_l199_199179

theorem frames_per_page (total_frames : ℕ) (pages : ℕ) (frames : ℕ) 
  (h1 : total_frames = 143) 
  (h2 : pages = 13) 
  (h3 : frames = total_frames / pages) : 
  frames = 11 := 
by 
  sorry

end frames_per_page_l199_199179


namespace number_is_seven_l199_199123

-- We will define the problem conditions and assert the answer
theorem number_is_seven (x : ℤ) (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by 
  -- Proof will be filled in here
  sorry

end number_is_seven_l199_199123


namespace sequence_term_formula_l199_199848

theorem sequence_term_formula 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h : ∀ n, S n = n^2 + 3 * n)
  (h₁ : a 1 = 4)
  (h₂ : ∀ n, 1 < n → a n = S n - S (n - 1)) :
  ∀ n, a n = 2 * n + 2 :=
by
  sorry

end sequence_term_formula_l199_199848


namespace cost_of_article_is_308_l199_199772

theorem cost_of_article_is_308 
  (C G : ℝ) 
  (h1 : 348 = C + G)
  (h2 : 350 = C + G + 0.05 * G) : 
  C = 308 :=
by
  sorry

end cost_of_article_is_308_l199_199772


namespace geometric_sequence_a7_a8_l199_199234

-- Define the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 1 + a 2 = 40
axiom h3 : a 3 + a 4 = 60

-- Proof problem: Find a_7 + a_8
theorem geometric_sequence_a7_a8 :
  a 7 + a 8 = 135 :=
by
  sorry

end geometric_sequence_a7_a8_l199_199234


namespace original_profit_percentage_l199_199015

noncomputable def originalCost : ℝ := 80
noncomputable def P := 30
noncomputable def profitPercentage : ℝ := ((100 - originalCost) / originalCost) * 100

theorem original_profit_percentage:
  ∀ (S C : ℝ),
  C = originalCost →
  ( ∀ (newCost : ℝ),
    newCost = 0.8 * C →
    ∀ (newSell : ℝ),
    newSell = S - 16.8 →
    newSell = 1.3 * newCost → P = 30 ) →
  profitPercentage = 25 := sorry

end original_profit_percentage_l199_199015


namespace problem_part1_problem_part2_l199_199427

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def f (x : ℝ) : ℝ :=
  dot_product (Real.cos x, Real.cos x) (Real.sqrt 3 * Real.cos x, Real.sin x)

theorem problem_part1 :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, (x ∈ Set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) → MonotoneOn f (Set.Icc (k * π + π / 12) (k * π + 7 * π / 12))) :=
sorry

theorem problem_part2 (A : ℝ) (a b c : ℝ) (area : ℝ) :
  f (A / 2 - π / 6) = Real.sqrt 3 ∧ 
  c = 2 ∧ 
  area = 2 * Real.sqrt 3 →
  a = 2 * Real.sqrt 3 ∨ a = 2 * Real.sqrt 7 :=
sorry

end problem_part1_problem_part2_l199_199427


namespace math_problem_l199_199031

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end math_problem_l199_199031


namespace sally_picked_peaches_l199_199853

-- Definitions from the conditions
def originalPeaches : ℕ := 13
def totalPeaches : ℕ := 55

-- The proof statement
theorem sally_picked_peaches : totalPeaches - originalPeaches = 42 := by
  sorry

end sally_picked_peaches_l199_199853


namespace probability_area_less_than_circumference_l199_199342

theorem probability_area_less_than_circumference :
  let probability (d : ℕ) := if d = 2 then (1 / 100 : ℚ)
                             else if d = 3 then (1 / 50 : ℚ)
                             else 0
  let sum_prob (d_s : List ℚ) := d_s.foldl (· + ·) 0
  let outcomes : List ℕ := List.range' 2 19 -- dice sum range from 2 to 20
  let valid_outcomes : List ℕ := outcomes.filter (· < 4)
  sum_prob (valid_outcomes.map probability) = (3 / 100 : ℚ) :=
by
  sorry

end probability_area_less_than_circumference_l199_199342


namespace smallest_value_of_x_l199_199091

theorem smallest_value_of_x :
  ∃ x : ℝ, (x / 4 + 2 / (3 * x) = 5 / 6) ∧ (∀ y : ℝ,
    (y / 4 + 2 / (3 * y) = 5 / 6) → x ≤ y) :=
sorry

end smallest_value_of_x_l199_199091


namespace intersection_of_sets_l199_199895

def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }
def B := { x : ℝ | x^2 > 1 }
def C := { x : ℝ | 1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : 
  (A ∩ B) = C := 
by sorry

end intersection_of_sets_l199_199895


namespace arctan_sum_l199_199512

theorem arctan_sum (a b : ℝ) : 
  Real.arctan (a / (a + 2 * b)) + Real.arctan (b / (2 * a + b)) = Real.arctan (1 / 2) :=
by {
  sorry
}

end arctan_sum_l199_199512


namespace sign_of_slope_equals_sign_of_correlation_l199_199747

-- Definitions for conditions
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ t, y t = a + b * x t

def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  r > -1 ∧ r < 1 ∧ ∀ t t', (y t - y t').sign = (x t - x t').sign

def regression_line_slope (b : ℝ) : Prop := True

-- Theorem to prove the sign of b is equal to the sign of r
theorem sign_of_slope_equals_sign_of_correlation (x y : ℝ → ℝ) (r b : ℝ) 
  (h1 : linear_relationship x y) 
  (h2 : correlation_coefficient x y r) 
  (h3 : regression_line_slope b) : 
  b.sign = r.sign := 
sorry

end sign_of_slope_equals_sign_of_correlation_l199_199747


namespace polygon_perimeter_l199_199701

theorem polygon_perimeter :
  let AB := 2
  let BC := 2
  let CD := 2
  let DE := 2
  let EF := 2
  let FG := 3
  let GH := 3
  let HI := 3
  let IJ := 3
  let JA := 4
  AB + BC + CD + DE + EF + FG + GH + HI + IJ + JA = 26 :=
by {
  sorry
}

end polygon_perimeter_l199_199701


namespace product_sum_even_l199_199335

theorem product_sum_even (m n : ℤ) : Even (m * n * (m + n)) := 
sorry

end product_sum_even_l199_199335


namespace quadratic_real_roots_range_l199_199833

theorem quadratic_real_roots_range (m : ℝ) :
  ∃ x : ℝ, (m - 3) * x^2 - 2 * x + 1 = 0 ↔ m ≤ 4 ∧ m ≠ 3 := 
by
  sorry

end quadratic_real_roots_range_l199_199833


namespace inequality_holds_l199_199304

theorem inequality_holds (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
by
  sorry

end inequality_holds_l199_199304


namespace tan_A_plus_C_eq_neg_sqrt3_l199_199995

theorem tan_A_plus_C_eq_neg_sqrt3
  (A B C : Real)
  (hSum : A + B + C = Real.pi)
  (hArithSeq : 2 * B = A + C)
  (hTriangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 := by
  sorry

end tan_A_plus_C_eq_neg_sqrt3_l199_199995


namespace min_value_a_b_c_l199_199406

theorem min_value_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : 9 * a + 4 * b = a * b * c) :
  a + b + c = 10 := sorry

end min_value_a_b_c_l199_199406


namespace square_properties_l199_199095

theorem square_properties (perimeter : ℝ) (h1 : perimeter = 40) :
  ∃ (side length area diagonal : ℝ), side = 10 ∧ length = 10 ∧ area = 100 ∧ diagonal = 10 * Real.sqrt 2 :=
by
  sorry

end square_properties_l199_199095


namespace line_parallel_through_M_line_perpendicular_through_M_l199_199624

-- Define the lines L1 and L2
def L1 (x y: ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def L2 (x y: ℝ) : Prop := x - 3 * y + 8 = 0

-- Define the parallel and perpendicular lines
def parallel_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0
def perpendicular_to_line (x y: ℝ) : Prop := 2 * x + y + 5 = 0

-- Define the intersection points
def M : ℝ × ℝ := (-2, 2)

-- Define the lines that pass through point M and are parallel or perpendicular to the given line
def line_parallel (x y: ℝ) : Prop := 2 * x + y + 2 = 0
def line_perpendicular (x y: ℝ) : Prop := x - 2 * y + 6 = 0

-- The proof statements
theorem line_parallel_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_parallel x y := by
  sorry

theorem line_perpendicular_through_M : ∃ x y : ℝ, L1 x y ∧ L2 x y ∧ x = (-2) ∧ y = 2 -> line_perpendicular x y := by
  sorry

end line_parallel_through_M_line_perpendicular_through_M_l199_199624


namespace quadratic_value_at_point_l199_199493

variable (a b c : ℝ)

-- Given: A quadratic function f(x) = ax^2 + bx + c that passes through the point (3,10)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_value_at_point
  (h : f a b c 3 = 10) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end quadratic_value_at_point_l199_199493


namespace max_slope_no_lattice_points_l199_199572

theorem max_slope_no_lattice_points :
  (∃ b : ℚ, (∀ m : ℚ, 1 / 3 < m ∧ m < b → ∀ x : ℤ, 0 < x ∧ x ≤ 200 → ¬ ∃ y : ℤ, y = m * x + 3) ∧ b = 68 / 203) := 
sorry

end max_slope_no_lattice_points_l199_199572


namespace misty_is_three_times_smaller_l199_199526

-- Define constants representing the favorite numbers of Misty and Glory
def G : ℕ := 450
def total_sum : ℕ := 600

-- Define Misty's favorite number in terms of the total sum and Glory's favorite number
def M : ℕ := total_sum - G

-- The main theorem stating that Misty's favorite number is 3 times smaller than Glory's favorite number
theorem misty_is_three_times_smaller : G / M = 3 := by
  -- Sorry placeholder indicating the need for further proof
  sorry

end misty_is_three_times_smaller_l199_199526


namespace negation_of_universal_is_existential_l199_199692

theorem negation_of_universal_is_existential :
  ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2 * x + 4 > 0) :=
by
  sorry

end negation_of_universal_is_existential_l199_199692


namespace caesars_rental_fee_l199_199235

theorem caesars_rental_fee (C : ℕ) 
  (hc : ∀ (n : ℕ), n = 60 → C + 30 * n = 500 + 35 * n) : 
  C = 800 :=
by
  sorry

end caesars_rental_fee_l199_199235


namespace gcd_polynomials_l199_199161

-- Given condition: a is an even multiple of 1009
def is_even_multiple_of_1009 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * 1009 * k

-- Statement: gcd(2a^2 + 31a + 58, a + 15) = 1
theorem gcd_polynomials (a : ℤ) (ha : is_even_multiple_of_1009 a) :
  gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := 
sorry

end gcd_polynomials_l199_199161


namespace quadratic_root_shift_l199_199238

theorem quadratic_root_shift (d e : ℝ) :
  (∀ r s : ℝ, (r^2 - 2 * r + 0.5 = 0) → (r-3)^2 + (r-3) * (s-3) * d + e = 0) → e = 3.5 := 
by
  intros
  sorry

end quadratic_root_shift_l199_199238


namespace length_more_than_breadth_l199_199702

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l199_199702


namespace product_of_two_integers_l199_199200

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 22) (h2 : x^2 - y^2 = 44) : x * y = 120 :=
by
  sorry

end product_of_two_integers_l199_199200


namespace linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l199_199339

-- Defining the first part of the problem
theorem linear_function_increasing_and_composition_eq_implies_values
  (a b : ℝ)
  (H1 : ∀ x y : ℝ, x < y → a * x + b < a * y + b)
  (H2 : ∀ x : ℝ, a * (a * x + b) + b = 16 * x + 5) :
  a = 4 ∧ b = 1 :=
by
  sorry

-- Defining the second part of the problem
theorem monotonic_gx_implies_m_range (m : ℝ)
  (H3 : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → (x2 + m) * (4 * x2 + 1) > (x1 + m) * (4 * x1 + 1)) :
  -9 / 4 ≤ m :=
by
  sorry

end linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l199_199339


namespace units_digit_G1000_is_3_l199_199758

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 1

theorem units_digit_G1000_is_3 : (G 1000) % 10 = 3 := sorry

end units_digit_G1000_is_3_l199_199758


namespace inequality_bi_l199_199785

variable {α : Type*} [LinearOrderedField α]

-- Sequence of positive real numbers
variable (a : ℕ → α)
-- Conditions for a_i
variable (ha : ∀ i, i > 0 → i * (a i)^2 ≥ (i + 1) * a (i - 1) * a (i + 1))
-- Positive real numbers x and y
variables (x y : α) (hx : x > 0) (hy : y > 0)
-- Definition of b_i
def b (i : ℕ) : α := x * a i + y * a (i - 1)

theorem inequality_bi (i : ℕ) (hi : i ≥ 2) : i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := 
sorry

end inequality_bi_l199_199785


namespace find_a45_l199_199264

theorem find_a45 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 11 ∧ a 1 = 11 ∧ 
    (∀ m n : ℕ, a (m + n) = (1/2) * (a (2 * m) + a (2 * n)) - (m - n)^2) ∧ 
    a 45 = 1991 := by
  sorry

end find_a45_l199_199264


namespace pyarelal_loss_l199_199267

theorem pyarelal_loss (total_loss : ℝ) (P : ℝ) (Ashok_capital : ℝ) (ratio_Ashok_Pyarelal : ℝ) :
  total_loss = 670 →
  Ashok_capital = P / 9 →
  ratio_Ashok_Pyarelal = 1 / 9 →
  Pyarelal_loss = 603 :=
by
  intro total_loss_eq Ashok_capital_eq ratio_eq
  sorry

end pyarelal_loss_l199_199267


namespace quadratic_inequality_l199_199070

theorem quadratic_inequality (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
sorry

end quadratic_inequality_l199_199070


namespace solution_set_of_inequality_l199_199970

theorem solution_set_of_inequality :
  { x : ℝ | (2 * x - 1) / (x + 1) ≤ 1 } = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end solution_set_of_inequality_l199_199970


namespace arithmetic_sequence_S9_l199_199773

noncomputable def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_S9 (a : ℕ → ℕ)
    (h1 : 2 * a 6 = 6 + a 7) :
    Sn a 9 = 54 := 
sorry

end arithmetic_sequence_S9_l199_199773


namespace parabola_point_distance_l199_199857

theorem parabola_point_distance (x y : ℝ) (h : y^2 = 2 * x) (d : ℝ) (focus_x : ℝ) (focus_y : ℝ) :
    focus_x = 1/2 → focus_y = 0 → d = 3 →
    (x + 1/2 = d) → x = 5/2 :=
by
  intros h_focus_x h_focus_y h_d h_dist
  sorry

end parabola_point_distance_l199_199857


namespace randy_fifth_quiz_score_l199_199563

def scores : List ℕ := [90, 98, 92, 94]

def goal_average : ℕ := 94

def total_points (n : ℕ) (avg : ℕ) : ℕ := n * avg

def current_points (l : List ℕ) : ℕ := l.sum

def needed_score (total current : ℕ) : ℕ := total - current

theorem randy_fifth_quiz_score :
  needed_score (total_points 5 goal_average) (current_points scores) = 96 :=
by 
  sorry

end randy_fifth_quiz_score_l199_199563


namespace all_pets_combined_l199_199957

def Teddy_initial_dogs : Nat := 7
def Teddy_initial_cats : Nat := 8
def Teddy_initial_rabbits : Nat := 6

def Teddy_adopted_dogs : Nat := 2
def Teddy_adopted_rabbits : Nat := 4

def Ben_dogs : Nat := 3 * Teddy_initial_dogs
def Ben_cats : Nat := 2 * Teddy_initial_cats

def Dave_dogs : Nat := (Teddy_initial_dogs + Teddy_adopted_dogs) - 4
def Dave_cats : Nat := Teddy_initial_cats + 13
def Dave_rabbits : Nat := 3 * Teddy_initial_rabbits

def Teddy_current_dogs : Nat := Teddy_initial_dogs + Teddy_adopted_dogs
def Teddy_current_cats : Nat := Teddy_initial_cats
def Teddy_current_rabbits : Nat := Teddy_initial_rabbits + Teddy_adopted_rabbits

def Teddy_total : Nat := Teddy_current_dogs + Teddy_current_cats + Teddy_current_rabbits
def Ben_total : Nat := Ben_dogs + Ben_cats
def Dave_total : Nat := Dave_dogs + Dave_cats + Dave_rabbits

def total_pets_combined : Nat := Teddy_total + Ben_total + Dave_total

theorem all_pets_combined : total_pets_combined = 108 :=
by
  sorry

end all_pets_combined_l199_199957


namespace set_of_x_values_l199_199152

theorem set_of_x_values (x : ℝ) : (3 ≤ abs (x + 2) ∧ abs (x + 2) ≤ 6) ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) := by
  sorry

end set_of_x_values_l199_199152


namespace average_difference_l199_199461

theorem average_difference :
  let avg1 := (24 + 35 + 58) / 3
  let avg2 := (19 + 51 + 29) / 3
  avg1 - avg2 = 6 := by
sorry

end average_difference_l199_199461


namespace max_distance_l199_199256

theorem max_distance (x y : ℝ) (u v w : ℝ)
  (h1 : u = Real.sqrt (x^2 + y^2))
  (h2 : v = Real.sqrt ((x - 1)^2 + y^2))
  (h3 : w = Real.sqrt ((x - 1)^2 + (y - 1)^2))
  (h4 : u^2 + v^2 = w^2) :
  ∃ (P : ℝ), P = 2 + Real.sqrt 2 :=
sorry

end max_distance_l199_199256


namespace smallest_solution_to_equation_l199_199950

theorem smallest_solution_to_equation :
  let x := 4 - Real.sqrt 2
  ∃ x, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
       ∀ y, (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → x ≤ y :=
  by
    let x := 4 - Real.sqrt 2
    sorry

end smallest_solution_to_equation_l199_199950


namespace num_lines_satisfying_conditions_l199_199783

-- Define the entities line, angle, and perpendicularity in a geometric framework
variable (Point Line : Type)
variable (P : Point)
variable (a b l : Line)

-- Define geometrical predicates
variable (Perpendicular : Line → Line → Prop)
variable (Passes_Through : Line → Point → Prop)
variable (Forms_Angle : Line → Line → ℝ → Prop)

-- Given conditions
axiom perp_ab : Perpendicular a b
axiom passes_through_P : Passes_Through l P
axiom angle_la_30 : Forms_Angle l a (30 : ℝ)
axiom angle_lb_90 : Forms_Angle l b (90 : ℝ)

-- The statement to prove
theorem num_lines_satisfying_conditions : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
  Passes_Through l1 P ∧ Forms_Angle l1 a (30 : ℝ) ∧ Forms_Angle l1 b (90 : ℝ) ∧
  Passes_Through l2 P ∧ Forms_Angle l2 a (30 : ℝ) ∧ Forms_Angle l2 b (90 : ℝ) ∧
  (∀ l', Passes_Through l' P ∧ Forms_Angle l' a (30 : ℝ) ∧ Forms_Angle l' b (90 : ℝ) → l' = l1 ∨ l' = l2) := sorry

end num_lines_satisfying_conditions_l199_199783


namespace horatio_sonnets_l199_199159

theorem horatio_sonnets (num_lines_per_sonnet : ℕ) (heard_sonnets : ℕ) (unheard_lines : ℕ) (h1 : num_lines_per_sonnet = 16) (h2 : heard_sonnets = 9) (h3 : unheard_lines = 126) :
  ∃ total_sonnets : ℕ, total_sonnets = 16 :=
by
  -- Note: The proof is not required, hence 'sorry' is included to skip it.
  sorry

end horatio_sonnets_l199_199159


namespace solid_could_be_rectangular_prism_or_cylinder_l199_199364

-- Definitions for the conditions
def is_rectangular_prism (solid : Type) : Prop := sorry
def is_cylinder (solid : Type) : Prop := sorry
def front_view_is_rectangle (solid : Type) : Prop := sorry
def side_view_is_rectangle (solid : Type) : Prop := sorry

-- Main statement
theorem solid_could_be_rectangular_prism_or_cylinder
  {solid : Type}
  (h1 : front_view_is_rectangle solid)
  (h2 : side_view_is_rectangle solid) :
  is_rectangular_prism solid ∨ is_cylinder solid :=
sorry

end solid_could_be_rectangular_prism_or_cylinder_l199_199364


namespace garden_perimeter_l199_199470

theorem garden_perimeter
  (width_garden : ℝ) (area_playground : ℝ)
  (length_playground : ℝ) (width_playground : ℝ)
  (area_garden : ℝ) (L : ℝ)
  (h1 : width_garden = 4) 
  (h2 : length_playground = 16)
  (h3 : width_playground = 12)
  (h4 : area_playground = length_playground * width_playground)
  (h5 : area_garden = area_playground)
  (h6 : area_garden = L * width_garden) :
  2 * L + 2 * width_garden = 104 :=
by
  sorry

end garden_perimeter_l199_199470


namespace four_digit_numbers_count_l199_199965

theorem four_digit_numbers_count : 
  (∀ d1 d2 d3 d4 : Fin 4, 
    (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧ 
    d2 ≠ d1 ∧ d2 ≠ 0 ∧ 
    d3 ≠ d1 ∧ d3 ≠ d2 ∧ 
    d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3) →
  3 * 6 = 18 := 
by
  sorry

end four_digit_numbers_count_l199_199965


namespace machine_loan_repaid_in_5_months_l199_199212

theorem machine_loan_repaid_in_5_months :
  ∀ (loan cost selling_price tax_percentage products_per_month profit_per_product months : ℕ),
    loan = 22000 →
    cost = 5 →
    selling_price = 8 →
    tax_percentage = 10 →
    products_per_month = 2000 →
    profit_per_product = (selling_price - cost - (selling_price * tax_percentage / 100)) →
    (products_per_month * months * profit_per_product) ≥ loan →
    months = 5 :=
by
  intros loan cost selling_price tax_percentage products_per_month profit_per_product months
  sorry

end machine_loan_repaid_in_5_months_l199_199212


namespace jesus_squares_l199_199962

theorem jesus_squares (J : ℕ) (linden_squares : ℕ) (pedro_squares : ℕ)
  (h1 : linden_squares = 75)
  (h2 : pedro_squares = 200)
  (h3 : pedro_squares = J + linden_squares + 65) : 
  J = 60 := 
by
  sorry

end jesus_squares_l199_199962


namespace range_of_m_l199_199978

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) m, 1 ≤ f x ∧ f x ≤ 10) ↔ 2 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l199_199978


namespace inclination_angle_of_line_l199_199268

def line_equation (x y : ℝ) : Prop := x * (Real.tan (Real.pi / 3)) + y + 2 = 0

theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : 
  ∃ α : ℝ, α = 2 * Real.pi / 3 ∧ 0 ≤ α ∧ α < Real.pi := by
  sorry

end inclination_angle_of_line_l199_199268


namespace sum_adjacent_odd_l199_199858

/-
  Given 2020 natural numbers written in a circle, prove that the sum of any two adjacent numbers is odd.
-/

noncomputable def numbers_in_circle : Fin 2020 → ℕ := sorry

theorem sum_adjacent_odd (k : Fin 2020) :
  (numbers_in_circle k + numbers_in_circle (k + 1)) % 2 = 1 :=
sorry

end sum_adjacent_odd_l199_199858


namespace office_expense_reduction_l199_199103

theorem office_expense_reduction (x : ℝ) (h : 0 ≤ x) (h' : x ≤ 1) : 
  2500 * (1 - x) ^ 2 = 1600 :=
sorry

end office_expense_reduction_l199_199103


namespace unique_zero_iff_a_eq_half_l199_199474

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (1 - x))

theorem unique_zero_iff_a_eq_half :
  (∃! x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end unique_zero_iff_a_eq_half_l199_199474


namespace katrina_cookies_left_l199_199558

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l199_199558


namespace arithmetic_sequence_l199_199543

theorem arithmetic_sequence (a_n : ℕ → ℕ) (a1 d : ℤ)
  (h1 : 4 * a1 + 6 * d = 0)
  (h2 : a1 + 4 * d = 5) :
  ∀ n : ℕ, a_n n = 2 * n - 5 :=
by
  -- Definitions derived from conditions
  let a_1 := (5 - 4 * d)
  let common_difference := 2
  intro n
  sorry

end arithmetic_sequence_l199_199543


namespace max_trains_ratio_l199_199834

theorem max_trains_ratio (years : ℕ) 
    (birthday_trains : ℕ) 
    (christmas_trains : ℕ) 
    (total_trains : ℕ)
    (parents_multiple : ℕ) 
    (h_years : years = 5)
    (h_birthday_trains : birthday_trains = 1)
    (h_christmas_trains : christmas_trains = 2)
    (h_total_trains : total_trains = 45)
    (h_parents_multiple : parents_multiple = 2) :
  let trains_received_in_years := years * (birthday_trains + 2 * christmas_trains)
  let trains_given_by_parents := total_trains - trains_received_in_years
  let trains_before_gift := total_trains - trains_given_by_parents
  trains_given_by_parents / trains_before_gift = parents_multiple := by
  sorry

end max_trains_ratio_l199_199834


namespace ajay_gain_l199_199964

-- Definitions of the problem conditions as Lean variables/constants.
variables (kg1 kg2 kg_total : ℕ) 
variables (price1 price2 price3 cost1 cost2 total_cost selling_price gain : ℝ)

-- Conditions of the problem.
def conditions : Prop :=
  kg1 = 15 ∧ 
  kg2 = 10 ∧ 
  kg_total = kg1 + kg2 ∧ 
  price1 = 14.5 ∧ 
  price2 = 13 ∧ 
  price3 = 15 ∧ 
  cost1 = kg1 * price1 ∧ 
  cost2 = kg2 * price2 ∧ 
  total_cost = cost1 + cost2 ∧ 
  selling_price = kg_total * price3 ∧ 
  gain = selling_price - total_cost 

-- The theorem for the gain amount proof.
theorem ajay_gain (h : conditions kg1 kg2 kg_total price1 price2 price3 cost1 cost2 total_cost selling_price gain) : 
  gain = 27.50 :=
  sorry

end ajay_gain_l199_199964


namespace GlobalConnect_more_cost_effective_if_x_300_l199_199220

def GlobalConnectCost (x : ℕ) : ℝ := 50 + 0.4 * x
def QuickConnectCost (x : ℕ) : ℝ := 0.6 * x

theorem GlobalConnect_more_cost_effective_if_x_300 : 
  GlobalConnectCost 300 < QuickConnectCost 300 :=
by
  sorry

end GlobalConnect_more_cost_effective_if_x_300_l199_199220


namespace problem1_problem2_l199_199409

-- Define the function f
def f (x b : ℝ) := |2 * x + b|

-- First problem: prove if the solution set of |2x + b| <= 3 is {x | -1 ≤ x ≤ 2}, then b = -1.
theorem problem1 (b : ℝ) : (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2 → |2 * x + b| ≤ 3)) → b = -1 :=
sorry

-- Second problem: given b = -1, prove that for all x ∈ ℝ, |2(x+3)-1| + |2(x+1)-1| ≥ -4.
theorem problem2 : (∀ x : ℝ, f (x + 3) (-1) + f (x + 1) (-1) ≥ -4) :=
sorry

end problem1_problem2_l199_199409


namespace original_faculty_members_l199_199042

theorem original_faculty_members (reduced_faculty : ℕ) (percentage : ℝ) : 
  reduced_faculty = 195 → percentage = 0.80 → 
  (∃ (original_faculty : ℕ), (original_faculty : ℝ) = reduced_faculty / percentage ∧ original_faculty = 244) :=
by
  sorry

end original_faculty_members_l199_199042


namespace man_age_twice_son_age_in_two_years_l199_199330

theorem man_age_twice_son_age_in_two_years :
  ∀ (S M X : ℕ), S = 30 → M = S + 32 → (M + X = 2 * (S + X)) → X = 2 :=
by
  intros S M X hS hM h
  sorry

end man_age_twice_son_age_in_two_years_l199_199330


namespace sum_of_possible_amounts_l199_199789

-- Definitions based on conditions:
def possible_quarters_amounts : Finset ℕ := {5, 30, 55, 80}
def possible_dimes_amounts : Finset ℕ := {15, 20, 30, 35, 40, 50, 60, 70, 80, 90}
def both_possible_amounts : Finset ℕ := possible_quarters_amounts ∩ possible_dimes_amounts

-- Statement of the problem:
theorem sum_of_possible_amounts : (both_possible_amounts.sum id) = 110 :=
by
  sorry

end sum_of_possible_amounts_l199_199789


namespace f_geq_expression_l199_199480

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 1 / a) * x - Real.log x

theorem f_geq_expression (a x : ℝ) (h : a < 0) : f x a ≥ (1 - 2 * a) * (a + 1) := 
  sorry

end f_geq_expression_l199_199480


namespace min_balls_to_draw_l199_199436

theorem min_balls_to_draw (red blue green yellow white black : ℕ) (h_red : red = 35) (h_blue : blue = 25) (h_green : green = 22) (h_yellow : yellow = 18) (h_white : white = 14) (h_black : black = 12) : 
  ∃ n, n = 95 ∧ ∀ (r b g y w bl : ℕ), r ≤ red ∧ b ≤ blue ∧ g ≤ green ∧ y ≤ yellow ∧ w ≤ white ∧ bl ≤ black → (r + b + g + y + w + bl = 95 → r ≥ 18 ∨ b ≥ 18 ∨ g ≥ 18 ∨ y ≥ 18 ∨ w ≥ 18 ∨ bl ≥ 18) :=
by sorry

end min_balls_to_draw_l199_199436


namespace tan_angle_sum_identity_l199_199575

theorem tan_angle_sum_identity
  (θ : ℝ)
  (h1 : θ > π / 2 ∧ θ < π)
  (h2 : Real.cos θ = -3 / 5) :
  Real.tan (θ + π / 4) = -1 / 7 := by
  sorry

end tan_angle_sum_identity_l199_199575


namespace percent_of_a_is_b_l199_199718

theorem percent_of_a_is_b (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : c = 0.25 * b) : b = 1.2 * a :=
by
  -- proof 
  sorry

end percent_of_a_is_b_l199_199718


namespace team_a_builds_per_day_l199_199416

theorem team_a_builds_per_day (x : ℝ) (h1 : (150 / x = 100 / (2 * x - 30))) : x = 22.5 := by
  sorry

end team_a_builds_per_day_l199_199416


namespace total_votes_l199_199666

theorem total_votes (V : ℝ) (h1 : 0.32 * V = 0.32 * V) (h2 : 0.32 * V + 1908 = 0.68 * V) : V = 5300 :=
by
  sorry

end total_votes_l199_199666


namespace line_condition_l199_199778

variable (m n Q : ℝ)

theorem line_condition (h1: m = 8 * n + 5) 
                       (h2: m + Q = 8 * (n + 0.25) + 5) 
                       (h3: p = 0.25) : Q = 2 :=
by
  sorry

end line_condition_l199_199778


namespace product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l199_199301

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l199_199301


namespace bob_needs_50_percent_improvement_l199_199053

def bob_time_in_seconds : ℕ := 640
def sister_time_in_seconds : ℕ := 320
def percentage_improvement_needed (bob_time sister_time : ℕ) : ℚ :=
  ((bob_time - sister_time) / bob_time : ℚ) * 100

theorem bob_needs_50_percent_improvement :
  percentage_improvement_needed bob_time_in_seconds sister_time_in_seconds = 50 := by
  sorry

end bob_needs_50_percent_improvement_l199_199053


namespace StepaMultiplication_l199_199617

theorem StepaMultiplication {a : ℕ} (h1 : Grisha's_answer = (3 / 2) ^ 4 * a)
  (h2 : Grisha's_answer = 81) :
  (∃ (m n : ℕ), m * n = (3 / 2) ^ 3 * a ∧ m < 10 ∧ n < 10) :=
by
  sorry

end StepaMultiplication_l199_199617


namespace tan_add_pi_over_4_sin_over_expression_l199_199439

variable (α : ℝ)

theorem tan_add_pi_over_4 (h : Real.tan α = 2) : 
  Real.tan (α + π / 4) = -3 := 
  sorry

theorem sin_over_expression (h : Real.tan α = 2) : 
  (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 1 := 
  sorry

end tan_add_pi_over_4_sin_over_expression_l199_199439


namespace sequence_sum_l199_199525

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, S n = n^2 * a n) :
  ∀ n : ℕ, S n = 2 * n / (n + 1) := 
by 
  sorry

end sequence_sum_l199_199525


namespace mac_runs_faster_by_120_minutes_l199_199374

theorem mac_runs_faster_by_120_minutes :
  ∀ (D : ℝ), (D / 3 - D / 4 = 2) → 2 * 60 = 120 := by
  -- Definitions matching the conditions
  intro D
  intro h

  -- The proof is not required, hence using sorry
  sorry

end mac_runs_faster_by_120_minutes_l199_199374


namespace inhabitants_number_even_l199_199811

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem inhabitants_number_even
  (K L : ℕ)
  (hK : is_even K)
  (hL : is_even L) :
  ¬ is_even (K + L + 1) :=
by
  sorry

end inhabitants_number_even_l199_199811


namespace average_of_new_sequence_l199_199205

variable (c : ℕ)  -- c is a positive integer
variable (d : ℕ)  -- d is the average of the sequence starting from c 

def average_of_sequence (seq : List ℕ) : ℕ :=
  if h : seq.length ≠ 0 then seq.sum / seq.length else 0

theorem average_of_new_sequence (h : d = average_of_sequence [c, c+1, c+2, c+3, c+4, c+5, c+6]) :
  average_of_sequence [d, d+1, d+2, d+3, d+4, d+5, d+6] = c + 6 := 
sorry

end average_of_new_sequence_l199_199205


namespace fraction_work_completed_by_third_group_l199_199315

def working_speeds (name : String) : ℚ :=
  match name with
  | "A"  => 1
  | "B"  => 2
  | "C"  => 1.5
  | "D"  => 2.5
  | "E"  => 3
  | "F"  => 2
  | "W1" => 1
  | "W2" => 1.5
  | "W3" => 1
  | "W4" => 1
  | "W5" => 0.5
  | "W6" => 1
  | "W7" => 1.5
  | "W8" => 1
  | _    => 0

def work_done_per_hour (workers : List String) : ℚ :=
  workers.map working_speeds |>.sum

def first_group : List String := ["A", "B", "C", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
def second_group : List String := ["A", "B", "C", "D", "E", "F", "W1", "W2"]
def third_group : List String := ["A", "B", "C", "D", "E", "W1", "W2"]

theorem fraction_work_completed_by_third_group :
  (work_done_per_hour third_group) / (work_done_per_hour second_group) = 25 / 29 :=
by
  sorry

end fraction_work_completed_by_third_group_l199_199315


namespace movie_theater_ticket_sales_l199_199170

theorem movie_theater_ticket_sales 
  (A C : ℤ) 
  (h1 : A + C = 900) 
  (h2 : 7 * A + 4 * C = 5100) : 
  A = 500 := 
sorry

end movie_theater_ticket_sales_l199_199170


namespace lawn_mowing_rate_l199_199196

-- Definitions based on conditions
def total_hours_mowed : ℕ := 2 * 7
def money_left_after_expenses (R : ℕ) : ℕ := (14 * R) / 4

-- The problem statement
theorem lawn_mowing_rate (h : money_left_after_expenses R = 49) : R = 14 := 
sorry

end lawn_mowing_rate_l199_199196


namespace total_travel_time_l199_199492

theorem total_travel_time (distance1 distance2 speed time1: ℕ) (h1 : distance1 = 100) (h2 : time1 = 1) (h3 : distance2 = 300) (h4 : speed = distance1 / time1) :
  (time1 + distance2 / speed) = 4 :=
by
  sorry

end total_travel_time_l199_199492


namespace negation_of_proposition_l199_199629

theorem negation_of_proposition :
  (¬ ∃ x_0 : ℝ, x_0^2 + x_0 - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l199_199629


namespace functional_inequality_solution_l199_199774

theorem functional_inequality_solution (f : ℝ → ℝ) (h : ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f (a) + b) * (a - f (b))) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := 
sorry

end functional_inequality_solution_l199_199774


namespace weekly_salary_correct_l199_199567

-- Define the daily salaries for each type of worker
def salary_A : ℝ := 200
def salary_B : ℝ := 250
def salary_C : ℝ := 300
def salary_D : ℝ := 350

-- Define the number of each type of worker
def num_A : ℕ := 3
def num_B : ℕ := 2
def num_C : ℕ := 3
def num_D : ℕ := 1

-- Define the total hours worked per day and the number of working days in a week
def hours_per_day : ℕ := 6
def working_days : ℕ := 7

-- Calculate the total daily salary for the team
def daily_salary_team : ℝ :=
  (num_A * salary_A) + (num_B * salary_B) + (num_C * salary_C) + (num_D * salary_D)

-- Calculate the total weekly salary for the team
def weekly_salary_team : ℝ := daily_salary_team * working_days

-- Problem: Prove that the total weekly salary for the team is Rs. 16,450
theorem weekly_salary_correct : weekly_salary_team = 16450 := by
  sorry

end weekly_salary_correct_l199_199567


namespace pears_worth_l199_199468

variable (apples pears : ℚ)
variable (h : 3/4 * 16 * apples = 6 * pears)

theorem pears_worth (h : 3/4 * 16 * apples = 6 * pears) : 1 / 3 * 9 * apples = 1.5 * pears :=
by
  sorry

end pears_worth_l199_199468


namespace water_volume_correct_l199_199814

def total_initial_solution : ℚ := 0.08 + 0.04 + 0.02
def fraction_water_in_initial : ℚ := 0.04 / total_initial_solution
def desired_total_volume : ℚ := 0.84
def required_water_volume : ℚ := desired_total_volume * fraction_water_in_initial

theorem water_volume_correct : 
  required_water_volume = 0.24 :=
by
  -- The proof is omitted
  sorry

end water_volume_correct_l199_199814


namespace avg_tickets_sold_by_males_100_l199_199290

theorem avg_tickets_sold_by_males_100 
  (female_avg : ℕ := 70) 
  (nonbinary_avg : ℕ := 50) 
  (overall_avg : ℕ := 66) 
  (male_ratio : ℕ := 2) 
  (female_ratio : ℕ := 3) 
  (nonbinary_ratio : ℕ := 5) : 
  ∃ (male_avg : ℕ), male_avg = 100 := 
by 
  sorry

end avg_tickets_sold_by_males_100_l199_199290


namespace tunnel_length_l199_199423

theorem tunnel_length (x : ℕ) (y : ℕ) 
  (h1 : 300 + x = 60 * y) 
  (h2 : x - 300 = 30 * y) : 
  x = 900 := 
by
  sorry

end tunnel_length_l199_199423


namespace minimize_expression_l199_199425

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 :=
sorry

end minimize_expression_l199_199425


namespace regular_pay_correct_l199_199926

noncomputable def regular_pay_per_hour (total_payment : ℝ) (regular_hours : ℕ) (overtime_hours : ℕ) (overtime_rate : ℝ) : ℝ :=
  let R := total_payment / (regular_hours + overtime_rate * overtime_hours)
  R

theorem regular_pay_correct :
  regular_pay_per_hour 198 40 13 2 = 3 :=
by
  sorry

end regular_pay_correct_l199_199926


namespace plane_equation_parametric_l199_199607

theorem plane_equation_parametric 
  (s t : ℝ)
  (v : ℝ × ℝ × ℝ)
  (x y z : ℝ) 
  (A B C D : ℤ)
  (h1 : v = (2 + s + 2 * t, 3 + 2 * s - t, 1 + s + 3 * t))
  (h2 : A = 7)
  (h3 : B = -1)
  (h4 : C = -5)
  (h5 : D = -6)
  (h6 : A > 0)
  (h7 : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) :
  7 * x - y - 5 * z - 6 = 0 := 
sorry

end plane_equation_parametric_l199_199607


namespace value_of_a_l199_199880

theorem value_of_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 20 - 7 * a) : a = 20 / 11 := by
  sorry

end value_of_a_l199_199880


namespace cube_surface_area_l199_199048

theorem cube_surface_area (V : ℝ) (s : ℝ) (A : ℝ) :
  V = 729 ∧ V = s^3 ∧ A = 6 * s^2 → A = 486 := by
  sorry

end cube_surface_area_l199_199048


namespace min_dot_product_value_l199_199096

noncomputable def dot_product_minimum (x : ℝ) : ℝ :=
  8 * x^2 + 4 * x

theorem min_dot_product_value :
  (∀ x, dot_product_minimum x ≥ -1 / 2) ∧ (∃ x, dot_product_minimum x = -1 / 2) :=
by
  sorry

end min_dot_product_value_l199_199096


namespace abs_opposite_of_three_eq_5_l199_199175

theorem abs_opposite_of_three_eq_5 : ∀ (a : ℤ), a = -3 → |a - 2| = 5 := by
  sorry

end abs_opposite_of_three_eq_5_l199_199175


namespace room_width_to_perimeter_ratio_l199_199661

theorem room_width_to_perimeter_ratio (L W : ℕ) (hL : L = 25) (hW : W = 15) :
  let P := 2 * (L + W)
  let ratio := W / P
  ratio = 3 / 16 :=
by
  sorry

end room_width_to_perimeter_ratio_l199_199661


namespace sin_cos_of_tan_is_two_l199_199160

theorem sin_cos_of_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
sorry

end sin_cos_of_tan_is_two_l199_199160


namespace perfect_square_k_value_l199_199489

-- Given condition:
def is_perfect_square (P : ℤ) : Prop := ∃ (z : ℤ), P = z * z

-- Theorem to prove:
theorem perfect_square_k_value (a b k : ℤ) (h : is_perfect_square (4 * a^2 + k * a * b + 9 * b^2)) :
  k = 12 ∨ k = -12 :=
sorry

end perfect_square_k_value_l199_199489


namespace problem_l199_199018

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := ![7, -3]
def result : Fin 2 → ℝ := ![-14, 6]
def expected : Fin 2 → ℝ := ![112, -48]

theorem problem :
    B.vecMul v = result →
    B.vecMul (B.vecMul (B.vecMul (B.vecMul v))) = expected := 
by
  intro h
  sorry

end problem_l199_199018


namespace boxes_in_case_correct_l199_199844

-- Given conditions
def total_boxes : Nat := 2
def blocks_per_box : Nat := 6
def total_blocks : Nat := 12

-- Define the number of boxes in a case as a result of total_blocks divided by blocks_per_box
def boxes_in_case : Nat := total_blocks / blocks_per_box

-- Prove the number of boxes in a case is 2
theorem boxes_in_case_correct : boxes_in_case = 2 := by
  -- Place the actual proof here
  sorry

end boxes_in_case_correct_l199_199844


namespace set_intersection_complement_l199_199961

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {1, 3, 4, 6, 7}

theorem set_intersection_complement :
  A ∩ (U \ B) = {2, 5} := 
by
  sorry

end set_intersection_complement_l199_199961


namespace solve_for_m_l199_199346

theorem solve_for_m (m x : ℤ) (h : 4 * x + 2 * m - 14 = 0) (hx : x = 2) : m = 3 :=
by
  -- Proof steps will go here.
  sorry

end solve_for_m_l199_199346


namespace triangle_height_in_terms_of_s_l199_199809

theorem triangle_height_in_terms_of_s (s h : ℝ)
  (rectangle_area : 2 * s * s = 2 * s^2)
  (base_of_triangle : base = s)
  (areas_equal : (1 / 2) * s * h = 2 * s^2) :
  h = 4 * s :=
by
  sorry

end triangle_height_in_terms_of_s_l199_199809


namespace inequality_proof_l199_199609

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (x^2 + y^2)) + (1 / x^2) + (1 / y^2) ≥ 10 / (x + y)^2 :=
sorry

end inequality_proof_l199_199609


namespace scientific_notation_of_twenty_million_l199_199507

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l199_199507


namespace sufficient_condition_not_monotonic_l199_199597

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - 4 * a * x - Real.log x

def sufficient_not_monotonic (a : ℝ) : Prop :=
  (a > 1 / 6) ∨ (a < -1 / 2)

theorem sufficient_condition_not_monotonic (a : ℝ) :
  sufficient_not_monotonic a → ¬(∀ x y : ℝ, 1 < x ∧ x < 3 ∧ 1 < y ∧ y < 3 ∧ x ≠ y → ((f a x - f a y) / (x - y) ≥ 0 ∨ (f a y - f a x) / (y - x) ≥ 0)) :=
by
  sorry

end sufficient_condition_not_monotonic_l199_199597


namespace arithmetic_sequence_product_l199_199860

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ)
  (h_arith_seq : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b n < b (n + 1))
  (h_condition : b 5 * b 6 = 14) :
  (b 4 * b 7 = -324) ∨ (b 4 * b 7 = -36) :=
sorry

end arithmetic_sequence_product_l199_199860


namespace missing_fraction_is_correct_l199_199243

theorem missing_fraction_is_correct :
  (1 / 3 + 1 / 2 + -5 / 6 + 1 / 5 + -9 / 20 + -9 / 20) = 0.45 - (23 / 20) :=
by
  sorry

end missing_fraction_is_correct_l199_199243


namespace find_a10_l199_199891

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

noncomputable def S_n (a1 d : ℝ) (n : ℕ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem find_a10 (a1 d : ℝ)
  (h1 : a_n a1 d 2 + a_n a1 d 4 = 2)
  (h2 : S_n a1 d 2 + S_n a1 d 4 = 1) :
  a_n a1 d 10 = 8 :=
sorry

end find_a10_l199_199891


namespace man_speed_in_still_water_l199_199028

noncomputable def speedInStillWater 
  (upstreamSpeedWithCurrentAndWind : ℝ)
  (downstreamSpeedWithCurrentAndWind : ℝ)
  (waterCurrentSpeed : ℝ)
  (windSpeedUpstream : ℝ) : ℝ :=
  (upstreamSpeedWithCurrentAndWind + waterCurrentSpeed + windSpeedUpstream + downstreamSpeedWithCurrentAndWind - waterCurrentSpeed + windSpeedUpstream) / 2
  
theorem man_speed_in_still_water :
  speedInStillWater 20 60 5 2.5 = 42.5 :=
  sorry

end man_speed_in_still_water_l199_199028


namespace max_mn_sq_l199_199598

theorem max_mn_sq {m n : ℤ} (h1: 1 ≤ m ∧ m ≤ 2005) (h2: 1 ≤ n ∧ n ≤ 2005) 
(h3: (n^2 + 2*m*n - 2*m^2)^2 = 1): m^2 + n^2 ≤ 702036 :=
sorry

end max_mn_sq_l199_199598


namespace k_range_for_two_zeros_of_f_l199_199261

noncomputable def f (x k : ℝ) : ℝ := x^2 - x * (Real.log x) - k * (x + 2) + 2

theorem k_range_for_two_zeros_of_f :
  ∀ k : ℝ, (∃ x1 x2 : ℝ, (1/2 < x1) ∧ (x1 < x2) ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by
  sorry

end k_range_for_two_zeros_of_f_l199_199261


namespace time_2556_hours_from_now_main_l199_199348

theorem time_2556_hours_from_now (h : ℕ) (mod_res : h % 12 = 0) :
  (3 + h) % 12 = 3 :=
by {
  sorry
}

-- Constants
def current_time : ℕ := 3
def hours_passed : ℕ := 2556
-- Proof input
def modular_result : hours_passed % 12 = 0 := by {
 sorry -- In the real proof, we should show that 2556 is divisible by 12
}

-- Main theorem instance
theorem main : (current_time + hours_passed) % 12 = 3 := 
  time_2556_hours_from_now hours_passed modular_result

end time_2556_hours_from_now_main_l199_199348


namespace expand_expression_l199_199363

theorem expand_expression (x y : ℤ) : (x + 12) * (3 * y + 8) = 3 * x * y + 8 * x + 36 * y + 96 := 
by
  sorry

end expand_expression_l199_199363


namespace geometric_series_sum_l199_199968

theorem geometric_series_sum:
  let a := 1
  let r := 5
  let n := 5
  (1 - r^n) / (1 - r) = 781 :=
by
  let a := 1
  let r := 5
  let n := 5
  sorry

end geometric_series_sum_l199_199968


namespace find_m_n_sum_l199_199601

theorem find_m_n_sum (n m : ℝ) (d : ℝ) 
(h1 : ∀ x y, 2*x + y + n = 0) 
(h2 : ∀ x y, 4*x + m*y - 4 = 0) 
(hd : d = (3/5) * Real.sqrt 5) 
: m + n = -3 ∨ m + n = 3 :=
sorry

end find_m_n_sum_l199_199601


namespace sequence_property_l199_199836

variable (a : ℕ → ℝ)

theorem sequence_property (h : ∀ n : ℕ, 0 < a n) 
  (h_property : ∀ n : ℕ, (a n)^2 ≤ a n - a (n + 1)) :
  ∀ n : ℕ, a n < 1 / n :=
by
  sorry

end sequence_property_l199_199836


namespace extreme_point_properties_l199_199540

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

theorem extreme_point_properties (a x₁ x₂ : ℝ) (h₁ : 0 < a) (h₂ : a < 1 / 4) 
  (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) (h₅ : x₁ < x₂) :
  f x₁ a < 0 ∧ f x₂ a > (-1 / 2) := 
sorry

end extreme_point_properties_l199_199540


namespace minimum_rectangle_area_l199_199771

theorem minimum_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 84) : 
  (l * w) = 41 :=
by sorry

end minimum_rectangle_area_l199_199771


namespace find_positive_n_for_quadratic_l199_199685

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

-- Define the condition: the quadratic equation has exactly one real root if its discriminant is zero
def has_one_real_root (a b c : ℝ) : Prop := discriminant a b c = 0

-- The specific quadratic equation y^2 + 6ny + 9n
def my_quadratic (n : ℝ) : Prop := has_one_real_root 1 (6 * n) (9 * n)

-- The statement to be proven: for the quadratic equation y^2 + 6ny + 9n to have one real root, n must be 1
theorem find_positive_n_for_quadratic : ∃ (n : ℝ), my_quadratic n ∧ n > 0 ∧ n = 1 := 
by
  sorry

end find_positive_n_for_quadratic_l199_199685


namespace correct_exponentiation_rule_l199_199199

theorem correct_exponentiation_rule (x y : ℝ) : ((x^2)^3 = x^6) :=
  by sorry

end correct_exponentiation_rule_l199_199199


namespace copper_alloy_proof_l199_199908

variable (x p : ℝ)

theorem copper_alloy_proof
  (copper_content1 copper_content2 weight1 weight2 total_weight : ℝ)
  (h1 : weight1 = 3)
  (h2 : copper_content1 = 0.4)
  (h3 : weight2 = 7)
  (h4 : copper_content2 = 0.3)
  (h5 : total_weight = 8)
  (h6 : 1 ≤ x ∧ x ≤ 3)
  (h7 : p = 100 * (copper_content1 * x + copper_content2 * (total_weight - x)) / total_weight) :
  31.25 ≤ p ∧ p ≤ 33.75 := 
  sorry

end copper_alloy_proof_l199_199908


namespace train_speed_l199_199741

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l199_199741


namespace find_value_of_2a_minus_b_l199_199062

def A : Set ℝ := {x | x < 1 ∨ x > 5}
def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

theorem find_value_of_2a_minus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = {x | 5 < x ∧ x ≤ 6}) : 2 * a - b = -4 :=
by
  sorry

end find_value_of_2a_minus_b_l199_199062


namespace find_areas_after_shortening_l199_199997

-- Define initial dimensions
def initial_length : ℤ := 5
def initial_width : ℤ := 7
def shortened_by : ℤ := 2

-- Define initial area condition
def initial_area_condition : Prop := 
  initial_length * (initial_width - shortened_by) = 15 ∨ (initial_length - shortened_by) * initial_width = 15

-- Define the resulting areas for shortening each dimension
def area_shortening_length : ℤ := (initial_length - shortened_by) * initial_width
def area_shortening_width : ℤ := initial_length * (initial_width - shortened_by)

-- Statement for proof
theorem find_areas_after_shortening
  (h : initial_area_condition) :
  area_shortening_length = 21 ∧ area_shortening_width = 25 :=
sorry

end find_areas_after_shortening_l199_199997


namespace points_scored_by_others_l199_199648

-- Define the conditions as hypothesis
variables (P_total P_Jessie : ℕ)
  (H1 : P_total = 311)
  (H2 : P_Jessie = 41)
  (H3 : ∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie)

-- Define what we need to prove
theorem points_scored_by_others (P_others : ℕ) :
  P_total = 311 → P_Jessie = 41 → 
  (∀ P_Lisa P_Devin: ℕ, P_Lisa = P_Jessie ∧ P_Devin = P_Jessie) → 
  P_others = 188 :=
by
  sorry

end points_scored_by_others_l199_199648


namespace possible_values_of_P_l199_199498

-- Definition of the conditions
variables (x y : ℕ) (h1 : x < y) (h2 : (x > 0)) (h3 : (y > 0))

-- Definition of P
def P : ℤ := (x^3 - y) / (1 + x * y)

-- Theorem statement
theorem possible_values_of_P : (P = 0) ∨ (P ≥ 2) :=
sorry

end possible_values_of_P_l199_199498


namespace arithmetic_evaluation_l199_199120

theorem arithmetic_evaluation : (64 / 0.08) - 2.5 = 797.5 :=
by
  sorry

end arithmetic_evaluation_l199_199120


namespace total_pawns_left_is_10_l199_199355

noncomputable def total_pawns_left_in_game 
    (initial_pawns : ℕ)
    (sophia_lost : ℕ)
    (chloe_lost : ℕ) : ℕ :=
  initial_pawns - sophia_lost + (initial_pawns - chloe_lost)

theorem total_pawns_left_is_10 :
  total_pawns_left_in_game 8 5 1 = 10 := by
  sorry

end total_pawns_left_is_10_l199_199355


namespace necessary_but_not_sufficient_l199_199622

variable {I : Set ℝ} (f : ℝ → ℝ) (M : ℝ)

theorem necessary_but_not_sufficient :
  (∀ x ∈ I, f x ≤ M) ↔
  (∀ x ∈ I, f x ≤ M ∧ (∃ x ∈ I, f x = M) → M = M ∧ ∃ x ∈ I, f x = M) :=
by
  sorry

end necessary_but_not_sufficient_l199_199622


namespace miguel_socks_probability_l199_199655

theorem miguel_socks_probability :
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  probability = 5 / 21 :=
by
  let total_socks := 10
  let socks_per_color := 2
  let colors := 5
  let draw_socks := 5
  let total_combinations := Nat.choose total_socks draw_socks
  let desired_combinations :=
    (Nat.choose colors 2) * (Nat.choose (colors - 2 + 1) 1) * socks_per_color
  let probability := desired_combinations / total_combinations
  sorry

end miguel_socks_probability_l199_199655


namespace percentage_reduction_is_20_l199_199130

def original_employees : ℝ := 243.75
def reduced_employees : ℝ := 195

theorem percentage_reduction_is_20 :
  (original_employees - reduced_employees) / original_employees * 100 = 20 := 
  sorry

end percentage_reduction_is_20_l199_199130


namespace savings_difference_correct_l199_199065

noncomputable def savings_1989_dick : ℝ := 5000
noncomputable def savings_1989_jane : ℝ := 5000

noncomputable def savings_1990_dick : ℝ := savings_1989_dick + 0.10 * savings_1989_dick
noncomputable def savings_1990_jane : ℝ := savings_1989_jane - 0.05 * savings_1989_jane

noncomputable def savings_1991_dick : ℝ := savings_1990_dick + 0.07 * savings_1990_dick
noncomputable def savings_1991_jane : ℝ := savings_1990_jane + 0.08 * savings_1990_jane

noncomputable def savings_1992_dick : ℝ := savings_1991_dick - 0.12 * savings_1991_dick
noncomputable def savings_1992_jane : ℝ := savings_1991_jane + 0.15 * savings_1991_jane

noncomputable def total_savings_dick : ℝ :=
savings_1989_dick + savings_1990_dick + savings_1991_dick + savings_1992_dick

noncomputable def total_savings_jane : ℝ :=
savings_1989_jane + savings_1990_jane + savings_1991_jane + savings_1992_jane

noncomputable def difference_of_savings : ℝ :=
total_savings_dick - total_savings_jane

theorem savings_difference_correct :
  difference_of_savings = 784.30 :=
by sorry

end savings_difference_correct_l199_199065


namespace catches_difference_is_sixteen_l199_199418

noncomputable def joe_catches : ℕ := 23
noncomputable def derek_catches : ℕ := 2 * joe_catches - 4
noncomputable def tammy_catches : ℕ := 30
noncomputable def one_third_derek : ℕ := derek_catches / 3
noncomputable def difference : ℕ := tammy_catches - one_third_derek

theorem catches_difference_is_sixteen :
  difference = 16 := 
by
  sorry

end catches_difference_is_sixteen_l199_199418


namespace jennas_total_ticket_cost_l199_199388

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end jennas_total_ticket_cost_l199_199388


namespace put_balls_in_boxes_l199_199313

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end put_balls_in_boxes_l199_199313


namespace table_tennis_probability_l199_199003

-- Define the given conditions
def prob_A_wins_set : ℚ := 2 / 3
def prob_B_wins_set : ℚ := 1 / 3
def best_of_five_sets := 5
def needed_wins_for_A := 3
def needed_losses_for_A := 2

-- Define the problem to prove
theorem table_tennis_probability :
  ((prob_A_wins_set ^ 2) * prob_B_wins_set * prob_A_wins_set) = 8 / 27 :=
by
  sorry

end table_tennis_probability_l199_199003


namespace solution_set_of_inequality_l199_199185

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - abs x - 2 < 0) ↔ (-2 < x ∧ x < 2) := 
sorry

end solution_set_of_inequality_l199_199185


namespace arithmetic_sequence_properties_l199_199595

variable {a : ℕ → ℕ}
variable {n : ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_properties 
  (a_3_eq_7 : a 3 = 7)
  (a_5_plus_a_7_eq_26 : a 5 + a 7 = 26) :
  (∃ a1 d, (a 1 = a1) ∧ (∀ n, a n = a1 + (n - 1) * d) ∧ d = 2) ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ S_n, S_n = n^2 + 2 * n) ∧ 
  ∀ T_n n, (∃ b : (ℕ → ℕ) → ℕ → ℕ, b a n = 1 / (a n ^ 2 - 1)) 
  → T_n = n / (4 * (n + 1)) :=
by
  sorry

end arithmetic_sequence_properties_l199_199595


namespace hallway_length_l199_199080

theorem hallway_length (s t d : ℝ) (h1 : 3 * s * t = 12) (h2 : s * t = d - 12) : d = 16 :=
sorry

end hallway_length_l199_199080


namespace greatest_m_value_l199_199849

noncomputable def find_greatest_m : ℝ := sorry

theorem greatest_m_value :
  ∃ m : ℝ, 
    (∀ x, x^2 - m * x + 8 = 0 → x ∈ {x | ∃ y, y^2 = 116}) ∧ 
    m = 2 * Real.sqrt 29 :=
sorry

end greatest_m_value_l199_199849


namespace point_B_coordinates_l199_199751

theorem point_B_coordinates (A B : ℝ) (hA : A = -2) (hDist : |A - B| = 3) : B = -5 ∨ B = 1 :=
by
  sorry

end point_B_coordinates_l199_199751


namespace batsman_average_increases_l199_199307

theorem batsman_average_increases
  (score_17th: ℕ)
  (avg_increase: ℕ)
  (initial_avg: ℕ)
  (final_avg: ℕ)
  (initial_innings: ℕ):
  score_17th = 74 →
  avg_increase = 3 →
  initial_innings = 16 →
  initial_avg = 23 →
  final_avg = initial_avg + avg_increase →
  (final_avg * (initial_innings + 1) = score_17th + (initial_avg * initial_innings)) →
  final_avg = 26 :=
by
  sorry

end batsman_average_increases_l199_199307


namespace intersection_of_A_and_B_l199_199469

variable (A : Set ℕ) (B : Set ℕ)

axiom h1 : A = {1, 2, 3, 4, 5}
axiom h2 : B = {3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} :=
  by sorry

end intersection_of_A_and_B_l199_199469


namespace complex_value_of_z_six_plus_z_inv_six_l199_199180

open Complex

theorem complex_value_of_z_six_plus_z_inv_six (z : ℂ) (h : z + z⁻¹ = 1) : z^6 + (z⁻¹)^6 = 2 := by
  sorry

end complex_value_of_z_six_plus_z_inv_six_l199_199180


namespace remainder_7n_mod_4_l199_199812

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l199_199812


namespace perfect_square_trinomial_l199_199599

theorem perfect_square_trinomial (k : ℤ) : (∃ a : ℤ, (x : ℤ) → x^2 - k * x + 9 = (x - a)^2) → (k = 6 ∨ k = -6) :=
sorry

end perfect_square_trinomial_l199_199599


namespace function_domain_l199_199390

theorem function_domain (x : ℝ) : x ≠ 3 → ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end function_domain_l199_199390


namespace average_rainfall_correct_l199_199829

/-- In July 1861, 366 inches of rain fell in Cherrapunji, India. -/
def total_rainfall : ℤ := 366

/-- July has 31 days. -/
def days_in_july : ℤ := 31

/-- Each day has 24 hours. -/
def hours_per_day : ℤ := 24

/-- The total number of hours in July -/
def total_hours_in_july : ℤ := days_in_july * hours_per_day

/-- The average rainfall in inches per hour during July 1861 in Cherrapunji, India -/
def average_rainfall_per_hour : ℤ := total_rainfall / total_hours_in_july

/-- Proof that the average rainfall in inches per hour is 366 / (31 * 24) -/
theorem average_rainfall_correct : average_rainfall_per_hour = 366 / (31 * 24) :=
by
  /- We skip the proof as it is not required. -/
  sorry

end average_rainfall_correct_l199_199829


namespace shaded_area_size_l199_199168

noncomputable def total_shaded_area : ℝ :=
  let R := 9
  let r := R / 2
  let area_larger_circle := 81 * Real.pi
  let shaded_area_larger_circle := area_larger_circle / 2
  let area_smaller_circle := Real.pi * r^2
  let shaded_area_smaller_circle := area_smaller_circle / 2
  let total_shaded_area := shaded_area_larger_circle + shaded_area_smaller_circle
  total_shaded_area

theorem shaded_area_size:
  total_shaded_area = 50.625 * Real.pi := 
by
  sorry

end shaded_area_size_l199_199168


namespace abc_sum_l199_199904

theorem abc_sum : ∃ a b c : ℤ, 
  (∀ x : ℤ, x^2 + 13 * x + 30 = (x + a) * (x + b)) ∧ 
  (∀ x : ℤ, x^2 + 5 * x - 50 = (x + b) * (x - c)) ∧
  a + b + c = 18 := by
  sorry

end abc_sum_l199_199904


namespace range_of_m_l199_199237

variable (m : ℝ)

/-- Proposition p: For any x in ℝ, x^2 + 1 > m -/
def p := ∀ x : ℝ, x^2 + 1 > m

/-- Proposition q: The linear function f(x) = (2 - m) * x + 1 is an increasing function -/
def q := (2 - m) > 0

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 < m ∧ m < 2 := 
sorry

end range_of_m_l199_199237


namespace remaining_savings_after_purchase_l199_199488

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l199_199488


namespace tennis_tournament_possible_l199_199697

theorem tennis_tournament_possible (p : ℕ) : 
  (∀ i j : ℕ, i ≠ j → ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  i = a ∨ i = b ∨ i = c ∨ i = d ∧ j = a ∨ j = b ∨ j = c ∨ j = d) → 
  ∃ k : ℕ, p = 8 * k + 1 := by
  sorry

end tennis_tournament_possible_l199_199697


namespace geometric_series_common_ratio_l199_199712

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by
  rw [ha, hS] at hS_eq
  -- This statement follows from algebraic manipulation outlined in the solution steps.
  sorry

end geometric_series_common_ratio_l199_199712


namespace Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l199_199520

-- Problem 1: Factorization
theorem Problem1_factorize (a : ℝ) : a^2 - 8 * a + 15 = (a - 3) * (a - 5) :=
  sorry

-- Problem 2: Minimum Perimeter of triangle ABC
theorem Problem2_min_perimeter_triangle (a b c : ℝ) 
  (h : a^2 + b^2 - 14 * a - 8 * b + 65 = 0) (hc : ∃ k : ℤ, 2 * k + 1 = c) : 
  a + b + c ≥ 16 :=
  sorry

-- Problem 3: Maximum Value of the Polynomial
theorem Problem3_max_value_polynomial : 
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, y ≠ -1 → -2 * x^2 - 4 * x + 3 ≥ -2 * y^2 - 4 * y + 3 :=
  sorry

end Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l199_199520


namespace value_of_k_l199_199641

theorem value_of_k :
  ∀ (x k : ℝ), (x + 6) * (x - 5) = x^2 + k * x - 30 → k = 1 :=
by
  intros x k h
  sorry

end value_of_k_l199_199641


namespace complex_number_z_satisfies_l199_199983

theorem complex_number_z_satisfies (z : ℂ) : 
  (z * (1 + I) + (-I) * (1 - I) = 0) → z = -1 := 
by {
  sorry
}

end complex_number_z_satisfies_l199_199983


namespace proposition_false_n5_l199_199826

variable (P : ℕ → Prop)

-- Declaring the conditions as definitions:
def condition1 (k : ℕ) (hk : k > 0) : Prop := P k → P (k + 1)
def condition2 : Prop := ¬ P 6

-- Theorem statement which leverages the conditions to prove the desired result.
theorem proposition_false_n5 (h1: ∀ k (hk : k > 0), condition1 P k hk) (h2: condition2 P) : ¬ P 5 :=
sorry

end proposition_false_n5_l199_199826


namespace all_numbers_equal_l199_199376

theorem all_numbers_equal (x : Fin 101 → ℝ) 
  (h : ∀ i : Fin 100, x i.val^3 + x ⟨(i.val + 1) % 101, sorry⟩ = (x ⟨(i.val + 1) % 101, sorry⟩)^3 + x ⟨(i.val + 2) % 101, sorry⟩) :
  ∀ i j : Fin 101, x i = x j := 
by 
  sorry

end all_numbers_equal_l199_199376


namespace num_boys_is_22_l199_199551

variable (girls boys total_students : ℕ)

-- Conditions
axiom h1 : total_students = 41
axiom h2 : boys = girls + 3
axiom h3 : total_students = girls + boys

-- Goal: Prove that the number of boys is 22
theorem num_boys_is_22 : boys = 22 :=
by
  sorry

end num_boys_is_22_l199_199551


namespace smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l199_199405

theorem smallest_k_repr_19_pow_n_sub_5_pow_m_exists :
  ∃ (k n m : ℕ), k > 0 ∧ n > 0 ∧ m > 0 ∧ k = 19 ^ n - 5 ^ m ∧ k = 14 :=
by
  sorry

end smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l199_199405


namespace cube_mono_l199_199201

theorem cube_mono {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_mono_l199_199201


namespace solve_for_z_l199_199788

theorem solve_for_z (z i : ℂ) (h1 : 1 - i*z + 3*i = -1 + i*z + 3*i) (h2 : i^2 = -1) : z = -i := 
  sorry

end solve_for_z_l199_199788


namespace solve_for_other_diagonal_l199_199879

noncomputable def length_of_other_diagonal
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem solve_for_other_diagonal 
  (h_area : ℝ) (h_d2 : ℝ) (h_condition : h_area = 75 ∧ h_d2 = 15) :
  length_of_other_diagonal h_area h_d2 = 10 :=
by
  -- using h_condition, prove the required theorem
  sorry

end solve_for_other_diagonal_l199_199879


namespace linear_function_correct_max_profit_correct_min_selling_price_correct_l199_199462

-- Definition of the linear function
def linear_function (x : ℝ) : ℝ :=
  -2 * x + 360

-- Definition of monthly profit function
def profit_function (x : ℝ) : ℝ :=
  (-2 * x + 360) * (x - 30)

noncomputable def max_profit_statement : Prop :=
  ∃ x w, x = 105 ∧ w = 11250 ∧ profit_function x = w

noncomputable def min_selling_price (profit : ℝ) : Prop :=
  ∃ x, profit_function x ≥ profit ∧ x ≥ 80

-- The proof statements
theorem linear_function_correct : linear_function 30 = 300 ∧ linear_function 45 = 270 :=
  by
    sorry

theorem max_profit_correct : max_profit_statement :=
  by
    sorry

theorem min_selling_price_correct : min_selling_price 10000 :=
  by
    sorry

end linear_function_correct_max_profit_correct_min_selling_price_correct_l199_199462


namespace any_nat_in_frac_l199_199586

theorem any_nat_in_frac (n : ℕ) : ∃ x y : ℕ, y ≠ 0 ∧ x^2 = y^3 * n := by
  sorry

end any_nat_in_frac_l199_199586


namespace no_sum_of_squares_of_rationals_l199_199574

theorem no_sum_of_squares_of_rationals (p q r s : ℕ) (hq : q ≠ 0) (hs : s ≠ 0)
    (hpq : Nat.gcd p q = 1) (hrs : Nat.gcd r s = 1) :
    (↑p / q : ℚ) ^ 2 + (↑r / s : ℚ) ^ 2 ≠ 168 := by 
    sorry

end no_sum_of_squares_of_rationals_l199_199574


namespace calculate_rent_is_correct_l199_199714

noncomputable def requiredMonthlyRent 
  (purchase_cost : ℝ) 
  (monthly_set_aside_percent : ℝ)
  (annual_property_tax : ℝ)
  (annual_insurance : ℝ)
  (annual_return_percent : ℝ) : ℝ :=
  let annual_return := annual_return_percent * purchase_cost
  let total_yearly_expenses := annual_return + annual_property_tax + annual_insurance
  let monthly_expenses := total_yearly_expenses / 12
  let retention_rate := 1 - monthly_set_aside_percent
  monthly_expenses / retention_rate

theorem calculate_rent_is_correct 
  (purchase_cost : ℝ := 200000)
  (monthly_set_aside_percent : ℝ := 0.2)
  (annual_property_tax : ℝ := 5000)
  (annual_insurance : ℝ := 2400)
  (annual_return_percent : ℝ := 0.08) :
  requiredMonthlyRent purchase_cost monthly_set_aside_percent annual_property_tax annual_insurance annual_return_percent = 2437.50 :=
by
  sorry

end calculate_rent_is_correct_l199_199714


namespace largest_is_A_minus_B_l199_199090

noncomputable def A := 3 * 1005^1006
noncomputable def B := 1005^1006
noncomputable def C := 1004 * 1005^1005
noncomputable def D := 3 * 1005^1005
noncomputable def E := 1005^1005
noncomputable def F := 1005^1004

theorem largest_is_A_minus_B :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B :=
by {
  sorry
}

end largest_is_A_minus_B_l199_199090


namespace solve_for_a_l199_199934

theorem solve_for_a
  (h : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (x^2 - a * x + 2 < 0)) :
  a = 3 :=
sorry

end solve_for_a_l199_199934


namespace correct_regression_equation_l199_199434

variable (x y : ℝ)

-- Assume that y is negatively correlated with x
axiom negative_correlation : x * y ≤ 0

-- The candidate regression equations
def regression_A : ℝ := -2 * x - 100
def regression_B : ℝ := 2 * x - 100
def regression_C : ℝ := -2 * x + 100
def regression_D : ℝ := 2 * x + 100

-- Prove that the correct regression equation reflecting the negative correlation is regression_C
theorem correct_regression_equation : regression_C x = -2 * x + 100 := by
  sorry

end correct_regression_equation_l199_199434


namespace uruguayan_goals_conceded_l199_199877

theorem uruguayan_goals_conceded (x : ℕ) (h : 14 = 9 + x) : x = 5 := by
  sorry

end uruguayan_goals_conceded_l199_199877


namespace six_digit_number_divisible_9_22_l199_199501

theorem six_digit_number_divisible_9_22 (d : ℕ) (h0 : 0 ≤ d) (h1 : d ≤ 9)
  (h2 : 9 ∣ (220140 + d)) (h3 : 22 ∣ (220140 + d)) : 220140 + d = 520146 :=
sorry

end six_digit_number_divisible_9_22_l199_199501


namespace ratio_of_ages_l199_199576

theorem ratio_of_ages
  (Sandy_age : ℕ)
  (Molly_age : ℕ)
  (h1 : Sandy_age = 49)
  (h2 : Molly_age = Sandy_age + 14) : (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- To complete the proof.
  sorry

end ratio_of_ages_l199_199576


namespace parabola_equation_exists_line_m_equation_exists_l199_199428

noncomputable def problem_1 : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ (x y : ℝ), x^2 = 2 * p * y → y = x^2 / (2 * p)) ∧ 
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 = 2 * p * y1 → x2^2 = 2 * p * y2 → 
    (y1 + y2 = 8 - p) ∧ ((y1 + y2) / 2 = 3) → p = 2)

noncomputable def problem_2 : Prop :=
  ∃ (k : ℝ), (k^2 = 1 / 4) ∧ (∀ (x : ℝ), (x^2 - 4 * k * x - 24 = 0) → 
    (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k ∧ x1 * x2 = -24)) ∧
  (∀ (x1 x2 : ℝ), x1^2 = 4 * (k * x1 + 6) ∧ x2^2 = 4 * (k * x2 + 6) → 
    ∀ (x3 x4 : ℝ), (x1 * x2) ^ 2 - 4 * ((x1 + x2) ^ 2 - 2 * x1 * x2) + 16 + 16 * x1 * x2 = 0 → 
    (k = 1 / 2 ∨ k = -1 / 2))

theorem parabola_equation_exists : problem_1 :=
by {
  sorry
}

theorem line_m_equation_exists : problem_2 :=
by {
  sorry
}

end parabola_equation_exists_line_m_equation_exists_l199_199428


namespace determine_k_linear_l199_199710

theorem determine_k_linear (k : ℝ) : |k| = 1 ∧ k + 1 ≠ 0 ↔ k = 1 := by
  sorry

end determine_k_linear_l199_199710


namespace tylenol_interval_l199_199051

/-- Mark takes 2 Tylenol tablets of 500 mg each at certain intervals for 12 hours, and he ends up taking 3 grams of Tylenol in total. Prove that the interval in hours at which he takes the tablets is 2.4 hours. -/
theorem tylenol_interval 
    (total_dose_grams : ℝ)
    (tablet_mg : ℝ)
    (hours : ℝ)
    (tablets_taken_each_time : ℝ) 
    (total_tablets : ℝ) 
    (interval_hours : ℝ) :
    total_dose_grams = 3 → 
    tablet_mg = 500 → 
    hours = 12 → 
    tablets_taken_each_time = 2 → 
    total_tablets = (total_dose_grams * 1000) / tablet_mg → 
    interval_hours = hours / (total_tablets / tablets_taken_each_time - 1) → 
    interval_hours = 2.4 :=
by
  intros
  sorry

end tylenol_interval_l199_199051


namespace exists_base_for_part_a_not_exists_base_for_part_b_l199_199975

theorem exists_base_for_part_a : ∃ b : ℕ, (3 + 4 = b) ∧ (3 * 4 = 1 * b + 5) := 
by
  sorry

theorem not_exists_base_for_part_b : ¬ ∃ b : ℕ, (2 + 3 = b) ∧ (2 * 3 = 1 * b + 1) :=
by
  sorry

end exists_base_for_part_a_not_exists_base_for_part_b_l199_199975


namespace initial_amount_invested_l199_199819

-- Definition of the conditions as Lean definitions
def initial_amount_interest_condition (A r : ℝ) : Prop := 25000 = A * r
def interest_rate_condition (r : ℝ) : Prop := r = 5

-- The main theorem we want to prove
theorem initial_amount_invested (A r : ℝ) (h1 : initial_amount_interest_condition A r) (h2 : interest_rate_condition r) : A = 5000 :=
by {
  sorry
}

end initial_amount_invested_l199_199819


namespace smallest_possible_number_of_apples_l199_199650

theorem smallest_possible_number_of_apples :
  ∃ (M : ℕ), M > 2 ∧ M % 9 = 2 ∧ M % 10 = 2 ∧ M % 11 = 2 ∧ M = 200 :=
by
  sorry

end smallest_possible_number_of_apples_l199_199650


namespace min_value_fraction_l199_199314

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y, y = x - 4 ∧ (x + 11) / Real.sqrt (x - 4) = 2 * Real.sqrt 15 := by
  sorry

end min_value_fraction_l199_199314


namespace b_completion_days_l199_199038

theorem b_completion_days (x : ℝ) :
  (7 * (1 / 24 + 1 / x + 1 / 40) + 4 * (1 / 24 + 1 / x) = 1) → x = 26.25 := 
by 
  sorry

end b_completion_days_l199_199038


namespace find_k_l199_199248

-- Define the conditions
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + k

-- Theorem statement
theorem find_k (k : ℝ) : (∀ x : ℝ, parabola k x = 0 → x = -1) → k = 1 :=
by
  sorry

end find_k_l199_199248


namespace probability_of_equal_numbers_when_throwing_two_fair_dice_l199_199010

theorem probability_of_equal_numbers_when_throwing_two_fair_dice :
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end probability_of_equal_numbers_when_throwing_two_fair_dice_l199_199010


namespace minimum_deposits_needed_l199_199026

noncomputable def annual_salary_expense : ℝ := 100000
noncomputable def annual_fixed_expense : ℝ := 170000
noncomputable def interest_rate_paid : ℝ := 0.0225
noncomputable def interest_rate_earned : ℝ := 0.0405

theorem minimum_deposits_needed :
  ∃ (x : ℝ), 
    (interest_rate_earned * x = annual_salary_expense + annual_fixed_expense + interest_rate_paid * x) →
    x = 1500 :=
by
  sorry

end minimum_deposits_needed_l199_199026


namespace shirts_sold_l199_199073

theorem shirts_sold (pants shorts shirts jackets credit_remaining : ℕ) 
  (price_shirt1 price_shirt2 price_pants : ℕ) 
  (discount tax : ℝ) :
  (pants = 3) →
  (shorts = 5) →
  (jackets = 2) →
  (price_shirt1 = 10) →
  (price_shirt2 = 12) →
  (price_pants = 15) →
  (discount = 0.10) →
  (tax = 0.05) →
  (credit_remaining = 25) →
  (store_credit : ℕ) →
  (store_credit = pants * 5 + shorts * 3 + jackets * 7 + shirts * 4) →
  (total_cost : ℝ) →
  (total_cost = (price_shirt1 + price_shirt2 + price_pants) * (1 - discount) * (1 + tax)) →
  (total_store_credit_used : ℝ) →
  (total_store_credit_used = total_cost - credit_remaining) →
  (initial_credit : ℝ) →
  (initial_credit = total_store_credit_used + (pants * 5 + shorts * 3 + jackets * 7)) →
  shirts = 2 :=
by
  intros
  sorry

end shirts_sold_l199_199073


namespace cylinder_volume_relation_l199_199894

theorem cylinder_volume_relation (r h : ℝ) (π_pos : 0 < π) :
  (∀ B_h B_r A_h A_r : ℝ, B_h = r ∧ B_r = h ∧ A_h = h ∧ A_r = r 
   → 3 * (π * h^2 * r) = π * r^2 * h) → 
  ∃ N : ℝ, (π * (3 * h)^2 * h) = N * π * h^3 ∧ N = 9 :=
by 
  sorry

end cylinder_volume_relation_l199_199894


namespace remainder_of_91_pow_92_mod_100_l199_199896

theorem remainder_of_91_pow_92_mod_100 : (91 ^ 92) % 100 = 81 :=
by
  sorry

end remainder_of_91_pow_92_mod_100_l199_199896


namespace milly_needs_flamingoes_l199_199084

theorem milly_needs_flamingoes
  (flamingo_feathers : ℕ)
  (pluck_percent : ℚ)
  (num_boas : ℕ)
  (feathers_per_boa : ℕ)
  (pluckable_feathers_per_flamingo : ℕ)
  (total_feathers_needed : ℕ)
  (num_flamingoes : ℕ)
  (h1 : flamingo_feathers = 20)
  (h2 : pluck_percent = 0.25)
  (h3 : num_boas = 12)
  (h4 : feathers_per_boa = 200)
  (h5 : pluckable_feathers_per_flamingo = flamingo_feathers * pluck_percent)
  (h6 : total_feathers_needed = num_boas * feathers_per_boa)
  (h7 : num_flamingoes = total_feathers_needed / pluckable_feathers_per_flamingo)
  : num_flamingoes = 480 := 
by
  sorry

end milly_needs_flamingoes_l199_199084


namespace volume_surface_area_ratio_l199_199217

theorem volume_surface_area_ratio
  (V : ℕ := 9)
  (S : ℕ := 34)
  (shape_conditions : ∃ n : ℕ, n = 9 ∧ ∃ m : ℕ, m = 2) :
  V / S = 9 / 34 :=
by
  sorry

end volume_surface_area_ratio_l199_199217


namespace father_ate_oranges_l199_199704

theorem father_ate_oranges (initial_oranges : ℝ) (remaining_oranges : ℝ) (eaten_oranges : ℝ) : 
  initial_oranges = 77.0 → remaining_oranges = 75 → eaten_oranges = initial_oranges - remaining_oranges → eaten_oranges = 2.0 :=
by
  intros h1 h2 h3
  sorry

end father_ate_oranges_l199_199704


namespace total_guests_l199_199723

theorem total_guests (G : ℕ) 
  (hwomen: ∃ n, n = G / 2)
  (hmen: 15 = 15)
  (hchildren: ∃ n, n = G - (G / 2 + 15))
  (men_leaving: ∃ n, n = 1/5 * 15)
  (children_leaving: 4 = 4)
  (people_stayed: 43 = G - ((1/5 * 15) + 4))
  : G = 50 := by
  sorry

end total_guests_l199_199723


namespace Hillary_left_with_amount_l199_199905

theorem Hillary_left_with_amount :
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  remaining_amount = 25 :=
by
  let price_per_craft := 12
  let crafts_sold := 3
  let extra_earnings := 7
  let deposit_amount := 18
  let total_earnings := crafts_sold * price_per_craft + extra_earnings
  let remaining_amount := total_earnings - deposit_amount
  sorry

end Hillary_left_with_amount_l199_199905


namespace rectangle_A_plus_P_ne_162_l199_199269

theorem rectangle_A_plus_P_ne_162 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ) 
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : A + P ≠ 162 :=
by
  sorry

end rectangle_A_plus_P_ne_162_l199_199269


namespace david_more_pushups_than_zachary_l199_199311

theorem david_more_pushups_than_zachary :
  ∀ (zachary_pushups zachary_crunches david_crunches : ℕ),
    zachary_pushups = 34 →
    zachary_crunches = 62 →
    david_crunches = 45 →
    david_crunches + 17 = zachary_crunches →
    david_crunches + 17 - zachary_pushups = 17 :=
by
  intros zachary_pushups zachary_crunches david_crunches
  intros h1 h2 h3 h4
  sorry

end david_more_pushups_than_zachary_l199_199311


namespace division_quotient_example_l199_199963

theorem division_quotient_example :
  ∃ q : ℕ,
    let dividend := 760
    let divisor := 36
    let remainder := 4
    dividend = divisor * q + remainder ∧ q = 21 :=
by
  sorry

end division_quotient_example_l199_199963


namespace students_total_l199_199482

theorem students_total (T : ℝ) (h₁ : 0.675 * T = 594) : T = 880 :=
sorry

end students_total_l199_199482


namespace arithmetic_sequence_15th_term_l199_199955

theorem arithmetic_sequence_15th_term :
  let a1 := 3
  let d := 7
  let n := 15
  a1 + (n - 1) * d = 101 :=
by
  let a1 := 3
  let d := 7
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l199_199955


namespace intersection_is_equilateral_triangle_l199_199114

noncomputable def circle_eq (x y : ℝ) := x^2 + (y - 1)^2 = 1
noncomputable def ellipse_eq (x y : ℝ) := 9*x^2 + (y + 1)^2 = 9

theorem intersection_is_equilateral_triangle :
  ∀ A B C : ℝ × ℝ, circle_eq A.1 A.2 ∧ ellipse_eq A.1 A.2 ∧
                 circle_eq B.1 B.2 ∧ ellipse_eq B.1 B.2 ∧
                 circle_eq C.1 C.2 ∧ ellipse_eq C.1 C.2 → 
                 (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end intersection_is_equilateral_triangle_l199_199114


namespace rectangle_area_divisible_by_12_l199_199453

theorem rectangle_area_divisible_by_12 {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  12 ∣ (a * b) :=
sorry

end rectangle_area_divisible_by_12_l199_199453


namespace fabric_length_l199_199351

-- Define the width and area as given in the problem
def width : ℝ := 3
def area : ℝ := 24

-- Prove that the length is 8 cm
theorem fabric_length : (area / width) = 8 :=
by
  sorry

end fabric_length_l199_199351


namespace divisibility_of_solutions_l199_199387

theorem divisibility_of_solutions (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_form : p = 4 * k + 3)
  (h_eq : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) : 
  p ∣ x₀ ∨ p ∣ y₀ ∨ p ∣ z₀ ∨ p ∣ t₀ :=
sorry

end divisibility_of_solutions_l199_199387


namespace smallest_N_l199_199732

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l199_199732


namespace arithmetic_twelfth_term_l199_199945

theorem arithmetic_twelfth_term 
(a d : ℚ) (n : ℕ) (h_a : a = 1/2) (h_d : d = 1/3) (h_n : n = 12) : 
  a + (n - 1) * d = 25 / 6 := 
by 
  sorry

end arithmetic_twelfth_term_l199_199945


namespace complement_intersection_l199_199044

section SetTheory

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 3}) (hB : B = {2, 3, 4}) : 
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end SetTheory

end complement_intersection_l199_199044


namespace no_all_same_color_l199_199762

def chameleons_initial_counts (c b m : ℕ) : Prop :=
  c = 13 ∧ b = 15 ∧ m = 17

def chameleon_interaction (c b m : ℕ) : Prop :=
  (∃ c' b' m', c' + b' + m' = c + b + m ∧ 
  ((c' = c - 1 ∧ b' = b - 1 ∧ m' = m + 2) ∨
   (c' = c - 1 ∧ b' = b + 2 ∧ m' = m - 1) ∨
   (c' = c + 2 ∧ b' = b - 1 ∧ m' = m - 1)))

theorem no_all_same_color (c b m : ℕ) (h1 : chameleons_initial_counts c b m) : 
  ¬ (∃ x, c = x ∧ b = x ∧ m = x) := 
sorry

end no_all_same_color_l199_199762


namespace simplify_expression_l199_199937

theorem simplify_expression (y : ℝ) : 
  (3 * y) ^ 3 - 2 * y * y ^ 2 + y ^ 4 = 25 * y ^ 3 + y ^ 4 :=
by
  sorry

end simplify_expression_l199_199937


namespace correlation_non_deterministic_relationship_l199_199341

theorem correlation_non_deterministic_relationship
  (independent_var_fixed : Prop)
  (dependent_var_random : Prop)
  (correlation_def : Prop)
  (correlation_randomness : Prop) :
  (correlation_def → non_deterministic) :=
by
  sorry

end correlation_non_deterministic_relationship_l199_199341


namespace selling_price_correct_l199_199337

-- Define the conditions
def boxes := 3
def face_masks_per_box := 20
def cost_price := 15  -- in dollars
def profit := 15      -- in dollars

-- Define the total number of face masks
def total_face_masks := boxes * face_masks_per_box

-- Define the total amount he wants after selling all face masks
def total_amount := cost_price + profit

-- Prove that the selling price per face mask is $0.50
noncomputable def selling_price_per_face_mask : ℚ :=
  total_amount / total_face_masks

theorem selling_price_correct : selling_price_per_face_mask = 0.50 := by
  sorry

end selling_price_correct_l199_199337


namespace number_line_problem_l199_199855

theorem number_line_problem (A B C : ℤ) (hA : A = -1) (hB : B = A - 5 + 6) (hC : abs (C - B) = 5) :
  C = 5 ∨ C = -5 :=
by sorry

end number_line_problem_l199_199855


namespace two_pow_n_minus_one_divisible_by_seven_l199_199731

theorem two_pow_n_minus_one_divisible_by_seven (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := 
sorry

end two_pow_n_minus_one_divisible_by_seven_l199_199731


namespace greatest_possible_value_of_y_l199_199784

theorem greatest_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 :=
sorry

end greatest_possible_value_of_y_l199_199784


namespace solve_for_x_l199_199099

theorem solve_for_x : (3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 2.5 :=
by
  sorry

end solve_for_x_l199_199099


namespace total_oranges_l199_199122

theorem total_oranges (a b c : ℕ) (h1 : a = 80) (h2 : b = 60) (h3 : c = 120) : a + b + c = 260 :=
by
  sorry

end total_oranges_l199_199122


namespace find_a_l199_199297

theorem find_a (a n : ℕ) (h1 : (2 : ℕ) ^ n = 32) (h2 : (a + 1) ^ n = 243) : a = 2 := by
  sorry

end find_a_l199_199297


namespace find_AD_l199_199495

-- Defining points and distances in the context of a triangle
variables {A B C D: Type*}
variables (dist_AB : ℝ) (dist_AC : ℝ) (dist_BC : ℝ) (midpoint_D : Prop)

-- Given conditions
def triangle_conditions : Prop :=
  dist_AB = 26 ∧
  dist_AC = 26 ∧
  dist_BC = 24 ∧
  midpoint_D

-- Problem statement as a Lean theorem
theorem find_AD
  (h : triangle_conditions dist_AB dist_AC dist_BC midpoint_D) :
  ∃ (AD : ℝ), AD = 2 * Real.sqrt 133 :=
sorry

end find_AD_l199_199495


namespace archibald_percentage_wins_l199_199638

def archibald_wins : ℕ := 12
def brother_wins : ℕ := 18
def total_games_played : ℕ := archibald_wins + brother_wins

def percentage_archibald_wins : ℚ := (archibald_wins : ℚ) / (total_games_played : ℚ) * 100

theorem archibald_percentage_wins : percentage_archibald_wins = 40 := by
  sorry

end archibald_percentage_wins_l199_199638


namespace james_carrot_sticks_left_l199_199133

variable (original_carrot_sticks : ℕ)
variable (eaten_before_dinner : ℕ)
variable (eaten_after_dinner : ℕ)
variable (given_away_during_dinner : ℕ)

theorem james_carrot_sticks_left 
  (h1 : original_carrot_sticks = 50)
  (h2 : eaten_before_dinner = 22)
  (h3 : eaten_after_dinner = 15)
  (h4 : given_away_during_dinner = 8) :
  original_carrot_sticks - eaten_before_dinner - eaten_after_dinner - given_away_during_dinner = 5 := 
sorry

end james_carrot_sticks_left_l199_199133


namespace arithmetic_sequence_sum_l199_199721

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (a_1 d : ℤ) 
  (h1: S 3 = (3 * a_1) + (3 * (2 * d) / 2))
  (h2: S 7 = (7 * a_1) + (7 * (6 * d) / 2)) :
  S 5 = (5 * a_1) + (5 * (4 * d) / 2) := by
  sorry

end arithmetic_sequence_sum_l199_199721


namespace f_sum_zero_l199_199581

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero_l199_199581


namespace determinant_of_given_matrix_l199_199497

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix_l199_199497


namespace evaluate_expression_l199_199900

theorem evaluate_expression :
  (4 * 10^2011 - 1) / (4 * (3 * (10^2011 - 1) / 9) + 1) = 3 :=
by
  sorry

end evaluate_expression_l199_199900


namespace bus_stop_time_l199_199779

theorem bus_stop_time 
  (bus_speed_without_stoppages : ℤ)
  (bus_speed_with_stoppages : ℤ)
  (h1 : bus_speed_without_stoppages = 54)
  (h2 : bus_speed_with_stoppages = 36) :
  ∃ t : ℕ, t = 20 :=
by
  sorry

end bus_stop_time_l199_199779


namespace geometric_sequence_a_div_n_sum_first_n_terms_l199_199061

variable {a : ℕ → ℝ} -- sequence a_n
variable {S : ℕ → ℝ} -- sum of first n terms S_n

axiom S_recurrence {n : ℕ} (hn : n > 0) : 
  S (n + 1) = S n + (n + 1) / (3 * n) * a n

axiom a_1 : a 1 = 1

theorem geometric_sequence_a_div_n :
  ∃ (r : ℝ), ∀ {n : ℕ} (hn : n > 0), (a n / n) = r^n := 
sorry

theorem sum_first_n_terms (n : ℕ) :
  S n = (9 / 4) - ((9 / 4) + (3 * n / 2)) * (1 / 3) ^ n :=
sorry

end geometric_sequence_a_div_n_sum_first_n_terms_l199_199061


namespace emily_weight_l199_199835

theorem emily_weight (h_weight : 87 = 78 + e_weight) : e_weight = 9 := by
  sorry

end emily_weight_l199_199835


namespace probability_of_yellow_jelly_bean_l199_199239

theorem probability_of_yellow_jelly_bean (P_red P_orange P_green P_yellow : ℝ)
  (h_red : P_red = 0.1)
  (h_orange : P_orange = 0.4)
  (h_green : P_green = 0.25)
  (h_total : P_red + P_orange + P_green + P_yellow = 1) :
  P_yellow = 0.25 :=
by
  sorry

end probability_of_yellow_jelly_bean_l199_199239


namespace find_length_of_chord_AB_l199_199334

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the coordinates of points A and B
variables (x1 x2 y1 y2 : ℝ)

-- Define the conditions
def conditions : Prop := 
  parabola x1 y1 ∧ parabola x2 y2 ∧ (x1 + x2 = 4 / 3)

-- Define the length of chord AB
def length_of_chord_AB : ℝ := 
  (x1 + 1) + (x2 + 1)

-- Prove the length of chord AB
theorem find_length_of_chord_AB (x1 x2 y1 y2 : ℝ) (h : conditions x1 x2 y1 y2) :
  length_of_chord_AB x1 x2 = 10 / 3 :=
by
  sorry -- Proof is not required

end find_length_of_chord_AB_l199_199334


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l199_199131

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l199_199131


namespace evaluate_expression_l199_199878

theorem evaluate_expression : 4 * (8 - 3) - 6 / 3 = 18 :=
by sorry

end evaluate_expression_l199_199878


namespace alice_speed_proof_l199_199380

-- Problem definitions
def distance : ℕ := 1000
def abel_speed : ℕ := 50
def abel_arrival_time := distance / abel_speed
def alice_delay : ℕ := 1  -- Alice starts 1 hour later
def earlier_arrival_abel : ℕ := 6  -- Abel arrives 6 hours earlier than Alice

noncomputable def alice_speed : ℕ := (distance / (abel_arrival_time + earlier_arrival_abel))

theorem alice_speed_proof : alice_speed = 200 / 3 := by
  sorry -- proof not required as per instructions

end alice_speed_proof_l199_199380


namespace daily_rate_is_three_l199_199750

theorem daily_rate_is_three (r : ℝ) : 
  (∀ (initial bedbugs : ℝ), initial = 30 ∧ 
  (∀ days later_bedbugs, days = 4 ∧ later_bedbugs = 810 →
  later_bedbugs = initial * r ^ days)) → r = 3 :=
by
  intros h
  sorry

end daily_rate_is_three_l199_199750


namespace total_value_of_gold_is_l199_199998

-- Definitions based on the conditions
def legacyBars : ℕ := 5
def aleenaBars : ℕ := legacyBars - 2
def valuePerBar : ℝ := 2200
def totalValue : ℝ := (legacyBars + aleenaBars) * valuePerBar

-- Theorem statement
theorem total_value_of_gold_is :
  totalValue = 17600 := by
  -- We add sorry here to skip the proof
  sorry

end total_value_of_gold_is_l199_199998


namespace train_more_passengers_l199_199230

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end train_more_passengers_l199_199230


namespace positive_root_of_real_root_l199_199885

theorem positive_root_of_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : b^2 - 4*a*c ≥ 0) (h2 : c^2 - 4*b*a ≥ 0) (h3 : a^2 - 4*c*b ≥ 0) : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ (p = c ∧ q = a ∧ r = b) →
  (∃ x : ℝ, x > 0 ∧ p*x^2 + q*x + r = 0) :=
by 
  sorry

end positive_root_of_real_root_l199_199885


namespace find_divisor_l199_199560

-- Define the conditions
def dividend := 689
def quotient := 19
def remainder := 5

-- Define the division formula
def division_formula (divisor : ℕ) : Prop := 
  dividend = (divisor * quotient) + remainder

-- State the theorem to be proved
theorem find_divisor :
  ∃ divisor : ℕ, division_formula divisor ∧ divisor = 36 :=
by
  sorry

end find_divisor_l199_199560


namespace connie_num_markers_l199_199438

def num_red_markers (T : ℝ) := 0.41 * T
def num_total_markers (num_blue_markers : ℝ) (T : ℝ) := num_red_markers T + num_blue_markers

theorem connie_num_markers (T : ℝ) (h1 : num_total_markers 23 T = T) : T = 39 :=
by
sorry

end connie_num_markers_l199_199438


namespace value_of_business_l199_199336

variable (business_value : ℝ) -- We are looking for the value of the business
variable (man_ownership_fraction : ℝ := 2/3) -- The fraction of the business the man owns
variable (sale_fraction : ℝ := 3/4) -- The fraction of the man's shares that were sold
variable (sale_amount : ℝ := 6500) -- The amount for which the fraction of the shares were sold

-- The main theorem we are trying to prove
theorem value_of_business (h1 : man_ownership_fraction = 2/3) (h2 : sale_fraction = 3/4) (h3 : sale_amount = 6500) :
    business_value = 39000 := 
sorry

end value_of_business_l199_199336


namespace roots_quadratic_expression_value_l199_199999

theorem roots_quadratic_expression_value (m n : ℝ) 
  (h1 : m^2 + 2 * m - 2027 = 0)
  (h2 : n^2 + 2 * n - 2027 = 0) :
  (2 * m - m * n + 2 * n) = 2023 :=
by
  sorry

end roots_quadratic_expression_value_l199_199999


namespace hens_not_laying_eggs_l199_199009

def chickens_on_farm := 440
def number_of_roosters := 39
def total_eggs := 1158
def eggs_per_hen := 3

theorem hens_not_laying_eggs :
  (chickens_on_farm - number_of_roosters) - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end hens_not_laying_eggs_l199_199009


namespace average_speed_of_trip_l199_199722

theorem average_speed_of_trip (d1 d2 s1 s2 : ℕ)
  (h1 : d1 = 30) (h2 : d2 = 30)
  (h3 : s1 = 60) (h4 : s2 = 30) :
  (d1 + d2) / (d1 / s1 + d2 / s2) = 40 :=
by sorry

end average_speed_of_trip_l199_199722


namespace find_value_of_xy_plus_yz_plus_xz_l199_199494

variable (x y z : ℝ)

-- Conditions
def cond1 : Prop := x^2 + x * y + y^2 = 108
def cond2 : Prop := y^2 + y * z + z^2 = 64
def cond3 : Prop := z^2 + x * z + x^2 = 172

-- Theorem statement
theorem find_value_of_xy_plus_yz_plus_xz (hx : cond1 x y) (hy : cond2 y z) (hz : cond3 z x) : 
  x * y + y * z + x * z = 96 :=
sorry

end find_value_of_xy_plus_yz_plus_xz_l199_199494


namespace problem_statement_problem_statement_2_l199_199949

noncomputable def A (m : ℝ) : Set ℝ := {x | x > 2^m}
noncomputable def B : Set ℝ := {x | -4 < x - 4 ∧ x - 4 < 4}

theorem problem_statement (m : ℝ) (h1 : m = 2) :
  (A m ∪ B = {x | x > 0}) ∧ (A m ∩ B = {x | 4 < x ∧ x < 8}) :=
by sorry

theorem problem_statement_2 (m : ℝ) (h2 : A m ⊆ {x | x ≤ 0 ∨ 8 ≤ x}) :
  3 ≤ m :=
by sorry

end problem_statement_problem_statement_2_l199_199949


namespace problem1_problem2_l199_199071

noncomputable def triangle_boscos_condition (a b c A B : ℝ) : Prop :=
  b * Real.cos A = (2 * c + a) * Real.cos (Real.pi - B)

noncomputable def triangle_area (a b c : ℝ) (S : ℝ) : Prop :=
  S = (1 / 2) * a * c * Real.sin (2 * Real.pi / 3)

noncomputable def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = b + a + c

theorem problem1 (a b c A : ℝ) (h : triangle_boscos_condition a b c A (2 * Real.pi / 3)) : 
  ∃ B : ℝ, B = 2 * Real.pi / 3 :=
by
  sorry

theorem problem2 (a c : ℝ) (b : ℝ := 4) (area : ℝ := Real.sqrt 3) (P : ℝ) (h : triangle_area a b c area) (h_perim : triangle_perimeter a b c P) :
  ∃ x : ℝ, x = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end problem1_problem2_l199_199071


namespace remaining_distance_proof_l199_199278

/-
In a bicycle course with a total length of 10.5 kilometers (km), if Yoongi goes 1.5 kilometers (km) and then goes another 3730 meters (m), prove that the remaining distance of the course is 5270 meters.
-/

def km_to_m (km : ℝ) : ℝ := km * 1000

def total_course_length_km : ℝ := 10.5
def total_course_length_m : ℝ := km_to_m total_course_length_km

def yoongi_initial_distance_km : ℝ := 1.5
def yoongi_initial_distance_m : ℝ := km_to_m yoongi_initial_distance_km

def yoongi_additional_distance_m : ℝ := 3730

def yoongi_total_distance_m : ℝ := yoongi_initial_distance_m + yoongi_additional_distance_m

def remaining_distance_m (total_course_length_m yoongi_total_distance_m : ℝ) : ℝ :=
  total_course_length_m - yoongi_total_distance_m

theorem remaining_distance_proof : remaining_distance_m total_course_length_m yoongi_total_distance_m = 5270 := 
  sorry

end remaining_distance_proof_l199_199278


namespace sum_of_squares_of_first_10_primes_l199_199907

theorem sum_of_squares_of_first_10_primes :
  ((2^2) + (3^2) + (5^2) + (7^2) + (11^2) + (13^2) + (17^2) + (19^2) + (23^2) + (29^2)) = 2397 :=
by
  sorry

end sum_of_squares_of_first_10_primes_l199_199907


namespace min_value_a_l199_199592

theorem min_value_a (a b c d : ℚ) (h₀ : a > 0)
  (h₁ : ∀ n : ℕ, (a * n^3 + b * n^2 + c * n + d).den = 1) :
  a = 1/6 := by
  -- Proof goes here
  sorry

end min_value_a_l199_199592


namespace double_persons_half_work_l199_199768

theorem double_persons_half_work :
  (∀ (n : ℕ) (d : ℕ), d = 12 → (2 * n) * (d / 2) = n * 3) :=
by
  sorry

end double_persons_half_work_l199_199768


namespace total_hours_before_midterms_l199_199859

-- Define the hours spent on each activity per week
def chess_hours_per_week : ℕ := 2
def drama_hours_per_week : ℕ := 8
def glee_hours_per_week : ℕ := 3

-- Sum up the total hours spent on extracurriculars per week
def total_hours_per_week : ℕ := chess_hours_per_week + drama_hours_per_week + glee_hours_per_week

-- Define semester information
def total_weeks_per_semester : ℕ := 12
def weeks_before_midterms : ℕ := total_weeks_per_semester / 2
def weeks_sick : ℕ := 2
def active_weeks_before_midterms : ℕ := weeks_before_midterms - weeks_sick

-- Define the theorem statement about total hours before midterms
theorem total_hours_before_midterms : total_hours_per_week * active_weeks_before_midterms = 52 := by
  -- We skip the actual proof here
  sorry

end total_hours_before_midterms_l199_199859


namespace number_of_smaller_pipes_l199_199184

theorem number_of_smaller_pipes (D_L D_s : ℝ) (h1 : D_L = 8) (h2 : D_s = 2) (v: ℝ) :
  let A_L := (π * (D_L / 2)^2)
  let A_s := (π * (D_s / 2)^2)
  (A_L / A_s) = 16 :=
by {
  sorry
}

end number_of_smaller_pipes_l199_199184


namespace inequality_solution_l199_199013

theorem inequality_solution (x : ℝ) : 9 - x^2 < 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l199_199013


namespace price_per_glass_on_first_day_eq_half_l199_199535

structure OrangeadeProblem where
  O : ℝ
  W : ℝ
  P1 : ℝ
  P2 : ℝ
  W_eq_O : W = O
  P2_value : P2 = 0.3333333333333333
  revenue_eq : 2 * O * P1 = 3 * O * P2

theorem price_per_glass_on_first_day_eq_half (prob : OrangeadeProblem) : prob.P1 = 0.50 := 
by
  sorry

end price_per_glass_on_first_day_eq_half_l199_199535


namespace a_eq_zero_l199_199587

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ x : ℤ, x^2 = 2^n * a + b) : a = 0 :=
sorry

end a_eq_zero_l199_199587


namespace man_half_father_age_in_years_l199_199925

theorem man_half_father_age_in_years
  (M F Y : ℕ) 
  (h1: M = (2 * F) / 5) 
  (h2: F = 25) 
  (h3: M + Y = (F + Y) / 2) : 
  Y = 5 := by 
  sorry

end man_half_father_age_in_years_l199_199925


namespace inequality_inequality_must_be_true_l199_199796

variables {a b c d : ℝ}

theorem inequality_inequality_must_be_true
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (a / d) < (b / c) :=
sorry

end inequality_inequality_must_be_true_l199_199796


namespace pureAcidInSolution_l199_199509

/-- Define the conditions for the problem -/
def totalVolume : ℝ := 12
def percentageAcid : ℝ := 0.40

/-- State the theorem equivalent to the question:
    calculate the amount of pure acid -/
theorem pureAcidInSolution :
  totalVolume * percentageAcid = 4.8 := by
  sorry

end pureAcidInSolution_l199_199509


namespace semicircle_radius_l199_199429

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (hP : P = 180) : radius_of_semicircle P = 180 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_l199_199429


namespace min_value_of_function_l199_199279

-- Define the function f
def f (x : ℝ) := 3 * x^2 - 6 * x + 9

-- State the theorem about the minimum value of the function.
theorem min_value_of_function : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end min_value_of_function_l199_199279


namespace code_XYZ_to_base_10_l199_199620

def base_6_to_base_10 (x y z : ℕ) : ℕ :=
  x * 6^2 + y * 6^1 + z * 6^0

theorem code_XYZ_to_base_10 :
  ∀ (X Y Z : ℕ), 
    X = 5 ∧ Y = 0 ∧ Z = 4 →
    base_6_to_base_10 X Y Z = 184 :=
by
  intros X Y Z h
  cases' h with hX hYZ
  cases' hYZ with hY hZ
  rw [hX, hY, hZ]
  exact rfl

end code_XYZ_to_base_10_l199_199620


namespace largest_inscribed_triangle_area_l199_199181

theorem largest_inscribed_triangle_area
  (D : Type) 
  (radius : ℝ) 
  (r_eq : radius = 8) 
  (triangle_area : ℝ)
  (max_area : triangle_area = 64) :
  ∃ (base height : ℝ), (base = 2 * radius) ∧ (height = radius) ∧ (triangle_area = (1 / 2) * base * height) := 
by
  sorry

end largest_inscribed_triangle_area_l199_199181


namespace harold_shared_with_five_friends_l199_199514

theorem harold_shared_with_five_friends 
  (total_marbles : ℕ) (kept_marbles : ℕ) (marbles_per_friend : ℕ) (shared : ℕ) (friends : ℕ)
  (H1 : total_marbles = 100)
  (H2 : kept_marbles = 20)
  (H3 : marbles_per_friend = 16)
  (H4 : shared = total_marbles - kept_marbles)
  (H5 : friends = shared / marbles_per_friend) :
  friends = 5 :=
by
  sorry

end harold_shared_with_five_friends_l199_199514


namespace shaded_grid_percentage_l199_199165

theorem shaded_grid_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 64) (h2 : shaded_squares = 48) : 
  ((shaded_squares : ℚ) / (total_squares : ℚ)) * 100 = 75 :=
by
  rw [h1, h2]
  norm_num

end shaded_grid_percentage_l199_199165


namespace candy_bars_per_bag_l199_199842

/-
Define the total number of candy bars and the number of bags
-/
def totalCandyBars : ℕ := 75
def numberOfBags : ℚ := 15.0

/-
Prove that the number of candy bars per bag is 5
-/
theorem candy_bars_per_bag : totalCandyBars / numberOfBags = 5 := by
  sorry

end candy_bars_per_bag_l199_199842


namespace min_value_of_z_l199_199744

theorem min_value_of_z : ∃ x : ℝ, ∀ y : ℝ, 5 * x^2 + 20 * x + 25 ≤ 5 * y^2 + 20 * y + 25 :=
by
  sorry

end min_value_of_z_l199_199744


namespace fraction_of_defective_engines_l199_199825

theorem fraction_of_defective_engines
  (total_batches : ℕ)
  (engines_per_batch : ℕ)
  (non_defective_engines : ℕ)
  (H1 : total_batches = 5)
  (H2 : engines_per_batch = 80)
  (H3 : non_defective_engines = 300)
  : (total_batches * engines_per_batch - non_defective_engines) / (total_batches * engines_per_batch) = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end fraction_of_defective_engines_l199_199825


namespace apples_final_count_l199_199939

theorem apples_final_count :
  let initial_apples := 200
  let shared_apples := 5
  let remaining_after_share := initial_apples - shared_apples
  let sister_takes := remaining_after_share / 2
  let half_rounded_down := 97 -- explicitly rounding down since 195 cannot be split exactly
  let remaining_after_sister := remaining_after_share - half_rounded_down
  let received_gift := 7
  let final_count := remaining_after_sister + received_gift
  final_count = 105 :=
by
  sorry

end apples_final_count_l199_199939


namespace triangle_inequality_l199_199037

theorem triangle_inequality (a b c : ℝ) (h1 : b + c > a) (h2 : c + a > b) (h3 : a + b > c) :
  ab + bc + ca ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (ab + bc + ca) :=
by
  sorry

end triangle_inequality_l199_199037


namespace radius_of_tangent_circle_l199_199561

theorem radius_of_tangent_circle 
    (side_length : ℝ) 
    (tangent_angle : ℝ) 
    (sin_15 : ℝ)
    (circle_radius : ℝ) :
    side_length = 2 * Real.sqrt 3 →
    tangent_angle = 30 →
    sin_15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2) →
    circle_radius = 2 :=
by sorry

end radius_of_tangent_circle_l199_199561


namespace initial_principal_amount_l199_199924

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem initial_principal_amount :
  let P := 4410 / (compound_interest 1 0.07 4 2 * compound_interest 1 0.09 2 2)
  abs (P - 3238.78) < 0.01 :=
by
  sorry

end initial_principal_amount_l199_199924


namespace fourth_root_sq_eq_sixteen_l199_199838

theorem fourth_root_sq_eq_sixteen (x : ℝ) (h : (x^(1/4))^2 = 16) : x = 256 :=
sorry

end fourth_root_sq_eq_sixteen_l199_199838


namespace check_triangle_345_l199_199361

def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem check_triangle_345 : satisfies_triangle_inequality 3 4 5 := by
  sorry

end check_triangle_345_l199_199361


namespace brown_shoes_count_l199_199100

-- Definitions based on given conditions
def total_shoes := 66
def black_shoe_ratio := 2

theorem brown_shoes_count (B : ℕ) (H1 : black_shoe_ratio * B + B = total_shoes) : B = 22 :=
by
  -- Proof here is replaced with sorry for the purpose of this exercise
  sorry

end brown_shoes_count_l199_199100


namespace find_x_l199_199298

theorem find_x (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z)
(h₄ : x^2 / y = 3) (h₅ : y^2 / z = 4) (h₆ : z^2 / x = 5) : 
  x = (6480 : ℝ)^(1/7 : ℝ) :=
by 
  sorry

end find_x_l199_199298


namespace find_x_l199_199155

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l199_199155


namespace find_n_l199_199728

theorem find_n (n : ℕ) 
  (hM : ∀ M, M = n - 7 → 1 ≤ M)
  (hA : ∀ A, A = n - 2 → 1 ≤ A)
  (hT : ∀ M A, M = n - 7 → A = n - 2 → M + A < n) :
  n = 8 :=
by
  sorry

end find_n_l199_199728


namespace smallest_cookies_left_l199_199761

theorem smallest_cookies_left (m : ℤ) (h : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end smallest_cookies_left_l199_199761


namespace solution_set_of_inequality_l199_199413

theorem solution_set_of_inequality (x : ℝ) : |x^2 - 2| < 2 ↔ ((-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)) :=
by sorry

end solution_set_of_inequality_l199_199413


namespace determine_a_range_l199_199913

open Real

theorem determine_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + 2*x + a ≤ 0) → a ≤ 1 :=
sorry

end determine_a_range_l199_199913


namespace dandelions_survive_to_flower_l199_199444

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end dandelions_survive_to_flower_l199_199444


namespace xy_sum_is_2_l199_199578

theorem xy_sum_is_2 (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := 
  sorry

end xy_sum_is_2_l199_199578


namespace trigonometric_identity_l199_199529

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ π / 2)
  (h2 : cos α = 3 / 5) :
  (1 + sqrt 2 * cos (2 * α - π / 4)) / sin (α + π / 2) = 14 / 5 :=
by
  sorry

end trigonometric_identity_l199_199529


namespace total_number_of_boys_in_camp_l199_199250

theorem total_number_of_boys_in_camp (T : ℕ)
  (hA1 : ∃ (boysA : ℕ), boysA = 20 * T / 100)
  (hA2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 30 * boysA / 100 ∧ boysM = 40 * boysA / 100)
  (hB1 : ∃ (boysB : ℕ), boysB = 30 * T / 100)
  (hB2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 25 * boysB / 100 ∧ boysM = 35 * boysB / 100)
  (hC1 : ∃ (boysC : ℕ), boysC = 50 * T / 100)
  (hC2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 15 * boysC / 100 ∧ boysM = 45 * boysC / 100)
  (hA_no_SM : 77 = 70 * boysA / 100)
  (hB_no_SM : 72 = 60 * boysB / 100)
  (hC_no_SM : 98 = 60 * boysC / 100) :
  T = 535 :=
by
  sorry

end total_number_of_boys_in_camp_l199_199250


namespace cabbages_produced_l199_199214

theorem cabbages_produced (x y : ℕ) (h1 : y = x + 1) (h2 : x^2 + 199 = y^2) : y^2 = 10000 :=
by
  sorry

end cabbages_produced_l199_199214


namespace john_toy_store_fraction_l199_199432

theorem john_toy_store_fraction :
  let allowance := 4.80
  let arcade_spent := 3 / 5 * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_store_spent := 1.28
  let toy_store_spent := remaining_after_arcade - candy_store_spent
  (toy_store_spent / remaining_after_arcade) = 1 / 3 := by
    sorry

end john_toy_store_fraction_l199_199432


namespace remainder_of_division_l199_199208

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 - 20 * X^2 + 45 * X + 23
noncomputable def d : Polynomial ℤ := (X - 3)^2

theorem remainder_of_division :
  ∃ q r : Polynomial ℤ, p = q * d + r ∧ degree r < degree d ∧ r = 6 * X + 41 := sorry

end remainder_of_division_l199_199208


namespace leftovers_value_l199_199447

def quarters_in_roll : ℕ := 30
def dimes_in_roll : ℕ := 40
def james_quarters : ℕ := 77
def james_dimes : ℕ := 138
def lindsay_quarters : ℕ := 112
def lindsay_dimes : ℕ := 244
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftovers_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_in_roll
  let leftover_dimes := total_dimes % dimes_in_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 2.45 :=
by
  sorry

end leftovers_value_l199_199447


namespace radius_of_hole_l199_199198

-- Define the dimensions of the rectangular solid
def length1 : ℕ := 3
def length2 : ℕ := 8
def length3 : ℕ := 9

-- Define the radius of the hole
variable (r : ℕ)

-- Condition: The area of the 2 circles removed equals the lateral surface area of the cylinder
axiom area_condition : 2 * Real.pi * r^2 = 2 * Real.pi * r * length1

-- Prove that the radius of the cylindrical hole is 3
theorem radius_of_hole : r = 3 := by
  sorry

end radius_of_hole_l199_199198


namespace value_of_m_l199_199225

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end value_of_m_l199_199225


namespace quadratic_real_roots_a_condition_l199_199240

theorem quadratic_real_roots_a_condition (a : ℝ) (h : ∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) :
  a ≥ 1 ∧ a ≠ 5 :=
by
  sorry

end quadratic_real_roots_a_condition_l199_199240


namespace solve_marble_problem_l199_199232

noncomputable def marble_problem : Prop :=
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 50 ∧ 
  (∀ initial_white initial_black : ℕ, initial_white = 50 ∧ initial_black = 50 → 
  ∃ w b : ℕ, w = 50 + k - initial_black ∧ b = 50 - k ∧ (w, b) = (2, 0))

theorem solve_marble_problem: marble_problem :=
sorry

end solve_marble_problem_l199_199232


namespace downstream_speed_l199_199623

-- Define the speed of the fish in still water
def V_s : ℝ := 45

-- Define the speed of the fish going upstream
def V_u : ℝ := 35

-- Define the speed of the stream
def V_r : ℝ := V_s - V_u

-- Define the speed of the fish going downstream
def V_d : ℝ := V_s + V_r

-- The theorem to be proved
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l199_199623


namespace not_prime_for_any_n_l199_199411

theorem not_prime_for_any_n (k : ℕ) (hk : 1 < k) (n : ℕ) : 
  ¬ Prime (n^4 + 4 * k^4) :=
sorry

end not_prime_for_any_n_l199_199411


namespace math_problems_not_a_set_l199_199654

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end math_problems_not_a_set_l199_199654


namespace consecutive_diff_possible_l199_199787

variable (a b c : ℝ)

def greater_than_2022 :=
  a > 2022 ∨ b > 2022 ∨ c > 2022

def distinct_numbers :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem consecutive_diff_possible :
  greater_than_2022 a b c → distinct_numbers a b c → 
  ∃ (x y z : ℤ), x + 1 = y ∧ y + 1 = z ∧ 
  (a^2 - b^2 = ↑x) ∧ (b^2 - c^2 = ↑y) ∧ (c^2 - a^2 = ↑z) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end consecutive_diff_possible_l199_199787


namespace nearest_whole_number_l199_199325

theorem nearest_whole_number (x : ℝ) (h : x = 7263.4987234) : Int.floor (x + 0.5) = 7263 := by
  sorry

end nearest_whole_number_l199_199325


namespace probability_seven_chairs_probability_n_chairs_l199_199463
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l199_199463


namespace range_of_k_l199_199786

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

def is_ktimes_value_function (f : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  0 < k ∧ a < b ∧ f a = k * a ∧ f b = k * b

theorem range_of_k (k : ℝ) : (∃ a b : ℝ, is_ktimes_value_function f k a b) ↔ 1 < k ∧ k < 1 + 1 / Real.exp 1 := by
  sorry

end range_of_k_l199_199786


namespace man_late_minutes_l199_199866

theorem man_late_minutes (v t t' : ℝ) (hv : v' = 3 / 4 * v) (ht : t = 2) (ht' : t' = 4 / 3 * t) :
  t' * 60 - t * 60 = 40 :=
by
  sorry

end man_late_minutes_l199_199866


namespace ratio_of_discretionary_income_l199_199902

theorem ratio_of_discretionary_income
  (net_monthly_salary : ℝ) 
  (vacation_fund_pct : ℝ) 
  (savings_pct : ℝ) 
  (socializing_pct : ℝ) 
  (gifts_amt : ℝ)
  (D : ℝ) 
  (ratio : ℝ)
  (salary : net_monthly_salary = 3700)
  (vacation_fund : vacation_fund_pct = 0.30)
  (savings : savings_pct = 0.20)
  (socializing : socializing_pct = 0.35)
  (gifts : gifts_amt = 111)
  (discretionary_income : D = gifts_amt / 0.15)
  (net_salary_ratio : ratio = D / net_monthly_salary) :
  ratio = 1 / 5 := sorry

end ratio_of_discretionary_income_l199_199902


namespace total_employees_l199_199166

variable (E : ℕ) -- E is the total number of employees

-- Conditions given in the problem
variable (male_fraction : ℚ := 0.45) -- 45% of the total employees are males
variable (males_below_50 : ℕ := 1170) -- 1170 males are below 50 years old
variable (males_total : ℕ := 2340) -- Total number of male employees

-- Condition derived from the problem (calculation of total males)
lemma male_employees_equiv (h : males_total = 2 * males_below_50) : males_total = 2340 :=
  by sorry

-- Main theorem
theorem total_employees (h : male_fraction * E = males_total) : E = 5200 :=
  by sorry

end total_employees_l199_199166


namespace tom_and_jerry_drank_80_ounces_l199_199748

theorem tom_and_jerry_drank_80_ounces
    (T J : ℝ) 
    (initial_T : T = 40)
    (initial_J : J = 2 * T)
    (T_drank J_drank : ℝ)
    (T_remaining J_remaining : ℝ)
    (T_after_pour J_after_pour : ℝ)
    (T_final J_final : ℝ)
    (H1 : T_drank = (2 / 3) * T)
    (H2 : J_drank = (2 / 3) * J)
    (H3 : T_remaining = T - T_drank)
    (H4 : J_remaining = J - J_drank)
    (H5 : T_after_pour = T_remaining + (1 / 4) * J_remaining)
    (H6 : J_after_pour = J_remaining - (1 / 4) * J_remaining)
    (H7 : T_final = T_after_pour - 5)
    (H8 : J_final = J_after_pour + 5)
    (H9 : T_final = J_final + 4)
    : T_drank + J_drank = 80 :=
by
  sorry

end tom_and_jerry_drank_80_ounces_l199_199748


namespace polygon_vertices_l199_199899

-- Define the number of diagonals from one vertex
def diagonals_from_one_vertex (n : ℕ) := n - 3

-- The main theorem stating the number of vertices is 9 given 6 diagonals from one vertex
theorem polygon_vertices (D : ℕ) (n : ℕ) (h : D = 6) (h_diagonals : diagonals_from_one_vertex n = D) :
  n = 9 := by
  sorry

end polygon_vertices_l199_199899


namespace point_coordinates_l199_199228

namespace CoordinateProof

structure Point where
  x : ℝ
  y : ℝ

def isSecondQuadrant (P : Point) : Prop := P.x < 0 ∧ P.y > 0
def distToXAxis (P : Point) : ℝ := |P.y|
def distToYAxis (P : Point) : ℝ := |P.x|

theorem point_coordinates (P : Point) (h1 : isSecondQuadrant P) (h2 : distToXAxis P = 3) (h3 : distToYAxis P = 7) : P = ⟨-7, 3⟩ :=
by
  sorry

end CoordinateProof

end point_coordinates_l199_199228


namespace points_after_perfect_games_l199_199172

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l199_199172


namespace maggi_initial_packages_l199_199150

theorem maggi_initial_packages (P : ℕ) (h1 : 4 * P - 5 = 12) : P = 4 :=
sorry

end maggi_initial_packages_l199_199150


namespace ammonia_formation_l199_199016

theorem ammonia_formation (Li3N H2O LiOH NH3 : ℕ) (h₁ : Li3N = 1) (h₂ : H2O = 54) (h₃ : Li3N + 3 * H2O = 3 * LiOH + NH3) :
  NH3 = 1 :=
by
  sorry

end ammonia_formation_l199_199016


namespace units_digit_of_expression_l199_199284

theorem units_digit_of_expression :
  (9 * 19 * 1989 - 9 ^ 3) % 10 = 0 :=
by
  sorry

end units_digit_of_expression_l199_199284


namespace intersection_is_interval_l199_199386

-- Let M be the set of numbers where the domain of the function y = log x is defined.
def M : Set ℝ := {x | 0 < x}

-- Let N be the set of numbers where x^2 - 4 > 0.
def N : Set ℝ := {x | x^2 - 4 > 0}

-- The complement of N in the real numbers ℝ.
def complement_N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- We need to prove that the intersection of M and the complement of N is the interval (0, 2].
theorem intersection_is_interval : (M ∩ complement_N) = {x | 0 < x ∧ x ≤ 2} := 
by 
  sorry

end intersection_is_interval_l199_199386


namespace number_of_people_in_group_l199_199085

theorem number_of_people_in_group 
    (N : ℕ)
    (old_person_weight : ℕ) (new_person_weight : ℕ)
    (average_weight_increase : ℕ) :
    old_person_weight = 70 →
    new_person_weight = 94 →
    average_weight_increase = 3 →
    N * average_weight_increase = new_person_weight - old_person_weight →
    N = 8 :=
by
  sorry

end number_of_people_in_group_l199_199085


namespace solve_fractional_equation_l199_199954

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 0) : (x + 1) / x = 2 / 3 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l199_199954


namespace volume_of_parallelepiped_l199_199309

theorem volume_of_parallelepiped 
  (m n Q : ℝ) 
  (ratio_positive : 0 < m ∧ 0 < n)
  (Q_positive : 0 < Q)
  (h_square_area : ∃ a b : ℝ, a / b = m / n ∧ (a^2 + b^2) = Q) :
  ∃ (V : ℝ), V = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) :=
sorry

end volume_of_parallelepiped_l199_199309


namespace park_width_l199_199356

/-- The rectangular park theorem -/
theorem park_width 
  (length : ℕ)
  (lawn_area : ℤ)
  (road_width : ℕ)
  (crossroads : ℕ)
  (W : ℝ) :
  length = 60 →
  lawn_area = 2109 →
  road_width = 3 →
  crossroads = 2 →
  W = (2109 + (2 * 3 * 60) : ℝ) / 60 :=
sorry

end park_width_l199_199356


namespace sufficient_but_not_necessary_l199_199511

theorem sufficient_but_not_necessary (a b : ℝ) : (a > |b|) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > |b|)) := 
sorry

end sufficient_but_not_necessary_l199_199511


namespace ratio_Laura_to_Ken_is_2_to_1_l199_199049

def Don_paint_tiles_per_minute : ℕ := 3

def Ken_paint_tiles_per_minute : ℕ := Don_paint_tiles_per_minute + 2

def multiple : ℕ := sorry -- Needs to be introduced, not directly from the solution steps

def Laura_paint_tiles_per_minute : ℕ := multiple * Ken_paint_tiles_per_minute

def Kim_paint_tiles_per_minute : ℕ := Laura_paint_tiles_per_minute - 3

def total_tiles_in_15_minutes : ℕ := 375

def total_tiles_per_minute : ℕ := total_tiles_in_15_minutes / 15

def total_tiles_equation : Prop :=
  Don_paint_tiles_per_minute + Ken_paint_tiles_per_minute + Laura_paint_tiles_per_minute + Kim_paint_tiles_per_minute = total_tiles_per_minute

theorem ratio_Laura_to_Ken_is_2_to_1 :
  (total_tiles_equation → Laura_paint_tiles_per_minute / Ken_paint_tiles_per_minute = 2) := sorry

end ratio_Laura_to_Ken_is_2_to_1_l199_199049


namespace apple_price_l199_199876

variable (p q : ℝ)

theorem apple_price :
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 :=
by
  intros h1 h2 h3
  have h4 : p = 5 := sorry
  exact h4

end apple_price_l199_199876


namespace max_min_diff_value_l199_199338

noncomputable def max_min_diff_c (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : ℝ :=
  (10 / 3) - (-2)

theorem max_min_diff_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 12) : 
  max_min_diff_c a b c h1 h2 = 16 / 3 := 
by 
  sorry

end max_min_diff_value_l199_199338


namespace solve_diamond_l199_199193

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l199_199193


namespace express_2_175_billion_in_scientific_notation_l199_199115

-- Definition of scientific notation
def scientific_notation (a : ℝ) (n : ℤ) (value : ℝ) : Prop :=
  value = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem stating the problem
theorem express_2_175_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), scientific_notation a n 2.175e9 ∧ a = 2.175 ∧ n = 9 :=
by
  sorry

end express_2_175_billion_in_scientific_notation_l199_199115


namespace longest_side_of_triangle_l199_199082

theorem longest_side_of_triangle (a d : ℕ) (h1 : d = 2) (h2 : a - d > 0) (h3 : a + d > 0)
    (h_angle : ∃ C : ℝ, C = 120) 
    (h_arith_seq : ∃ (b c : ℕ), b = a - d ∧ c = a ∧ b + 2 * d = c + d) : 
    a + d = 7 :=
by
  -- The proof will be provided here
  sorry

end longest_side_of_triangle_l199_199082


namespace contrapositive_equivalence_l199_199966

variable (Person : Type)
variable (Happy Have : Person → Prop)

theorem contrapositive_equivalence :
  (∀ (x : Person), Happy x → Have x) ↔ (∀ (x : Person), ¬Have x → ¬Happy x) :=
by
  sorry

end contrapositive_equivalence_l199_199966


namespace train_speed_l199_199154

theorem train_speed
  (num_carriages : ℕ)
  (length_carriage length_engine : ℕ)
  (bridge_length_km : ℝ)
  (crossing_time_min : ℝ)
  (h1 : num_carriages = 24)
  (h2 : length_carriage = 60)
  (h3 : length_engine = 60)
  (h4 : bridge_length_km = 4.5)
  (h5 : crossing_time_min = 6) :
  (num_carriages * length_carriage + length_engine) / 1000 + bridge_length_km / (crossing_time_min / 60) = 60 :=
by
  sorry

end train_speed_l199_199154


namespace difference_of_squares_401_399_l199_199293

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 :=
by
  sorry

end difference_of_squares_401_399_l199_199293


namespace original_equation_l199_199528

theorem original_equation : 9^2 - 8^2 = 17 := by
  sorry

end original_equation_l199_199528


namespace b_investment_calculation_l199_199024

noncomputable def total_profit : ℝ := 9600
noncomputable def A_investment : ℝ := 2000
noncomputable def A_management_fee : ℝ := 0.10 * total_profit
noncomputable def remaining_profit : ℝ := total_profit - A_management_fee
noncomputable def A_total_received : ℝ := 4416
noncomputable def B_investment : ℝ := 1000

theorem b_investment_calculation (B: ℝ) 
  (h_total_profit: total_profit = 9600)
  (h_A_investment: A_investment = 2000)
  (h_A_management_fee: A_management_fee = 0.10 * total_profit)
  (h_remaining_profit: remaining_profit = total_profit - A_management_fee)
  (h_A_total_received: A_total_received = 4416)
  (h_A_total_formula : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit) :
  B = 1000 :=
by
  have h1 : total_profit = 9600 := h_total_profit
  have h2 : A_investment = 2000 := h_A_investment
  have h3 : A_management_fee = 0.10 * total_profit := h_A_management_fee
  have h4 : remaining_profit = total_profit - A_management_fee := h_remaining_profit
  have h5 : A_total_received = 4416 := h_A_total_received
  have h6 : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit := h_A_total_formula
  
  sorry

end b_investment_calculation_l199_199024


namespace minimum_value_expression_l199_199277

theorem minimum_value_expression 
  (a b c d : ℝ)
  (h1 : (2 * a^2 - Real.log a) / b = 1)
  (h2 : (3 * c - 2) / d = 1) :
  ∃ min_val : ℝ, min_val = (a - c)^2 + (b - d)^2 ∧ min_val = 1 / 10 :=
by {
  sorry
}

end minimum_value_expression_l199_199277


namespace tile_floor_with_polygons_l199_199486

theorem tile_floor_with_polygons (x y z: ℕ) (h1: 3 ≤ x) (h2: 3 ≤ y) (h3: 3 ≤ z) 
  (h_seamless: ((1 - (2 / (x: ℝ))) * 180 + (1 - (2 / (y: ℝ))) * 180 + (1 - (2 / (z: ℝ))) * 180 = 360)) :
  (1 / (x: ℝ) + 1 / (y: ℝ) + 1 / (z: ℝ) = 1 / 2) :=
by
  sorry

end tile_floor_with_polygons_l199_199486
