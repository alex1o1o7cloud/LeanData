import Mathlib

namespace expected_value_of_win_is_162_l37_37158

noncomputable def expected_value_of_win : ℝ :=
  (1/8) * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3)

theorem expected_value_of_win_is_162 : expected_value_of_win = 162 := 
by 
  sorry

end expected_value_of_win_is_162_l37_37158


namespace find_sin_alpha_l37_37876

theorem find_sin_alpha (α : ℝ) (h1 : 0 < α ∧ α < real.pi) (h2 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sin_alpha_l37_37876


namespace equation_squares_l37_37084

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l37_37084


namespace Jill_ball_difference_l37_37263

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l37_37263


namespace smallest_integer_x_l37_37021

theorem smallest_integer_x (x : ℤ) : (x^2 - 11 * x + 24 < 0) → x ≥ 4 ∧ x < 8 :=
by
sorry

end smallest_integer_x_l37_37021


namespace last_student_score_is_61_l37_37284

noncomputable def average_score_19_students := 82
noncomputable def average_score_20_students := 84
noncomputable def total_students := 20
noncomputable def oliver_multiplier := 2

theorem last_student_score_is_61 
  (total_score_19_students : ℝ := total_students - 1 * average_score_19_students)
  (total_score_20_students : ℝ := total_students * average_score_20_students)
  (oliver_score : ℝ := total_score_20_students - total_score_19_students)
  (last_student_score : ℝ := oliver_score / oliver_multiplier) :
  last_student_score = 61 :=
sorry

end last_student_score_is_61_l37_37284


namespace maximize_farmer_profit_l37_37159

theorem maximize_farmer_profit :
  ∃ x y : ℝ, x + y ≤ 2 ∧ 3 * x + y ≤ 5 ∧ x ≥ 0 ∧ y ≥ 0 ∧ x = 1.5 ∧ y = 0.5 ∧ 
  (∀ x' y' : ℝ, x' + y' ≤ 2 ∧ 3 * x' + y' ≤ 5 ∧ x' ≥ 0 ∧ y' ≥ 0 → 14400 * x + 6300 * y ≥ 14400 * x' + 6300 * y') :=
by
  sorry

end maximize_farmer_profit_l37_37159


namespace min_value_of_function_l37_37988

theorem min_value_of_function (x : ℝ) (hx : x > 3) :
  (x + (1 / (x - 3))) ≥ 5 :=
sorry

end min_value_of_function_l37_37988


namespace smallest_x_plus_y_l37_37206

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37206


namespace quadratic_function_l37_37558

theorem quadratic_function :
  ∃ a : ℝ, ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = a * (x - 1) * (x - 5)) ∧ f 3 = 10 ∧ 
  f = fun x => -2.5 * x^2 + 15 * x - 12.5 :=
by
  sorry

end quadratic_function_l37_37558


namespace map_length_represents_75_km_l37_37107
-- First, we broaden the import to bring in all the necessary libraries.

-- Define the conditions given in the problem.
def cm_to_km_ratio (cm : ℕ) (km : ℕ) : ℕ := km / cm

def map_represents (length_cm : ℕ) (length_km : ℕ) : Prop :=
  length_km = length_cm * cm_to_km_ratio 15 45

-- Rewrite the problem statement as a theorem in Lean 4.
theorem map_length_represents_75_km : map_represents 25 75 :=
by
  sorry

end map_length_represents_75_km_l37_37107


namespace max_product_of_xy_on_circle_l37_37834

theorem max_product_of_xy_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  ∃ (x y : ℤ), (x^2 + y^2 = 100) ∧ (∀ x y : ℤ, x^2 + y^2 = 100 → x * y ≤ 48) ∧ x * y = 48 := by
  sorry

end max_product_of_xy_on_circle_l37_37834


namespace number_of_valid_ns_l37_37258

theorem number_of_valid_ns :
  ∃ (n : ℝ), (n = 8 ∨ n = 1/2) ∧ ∀ n₁ n₂, (n₁ = 8 ∨ n₁ = 1/2) ∧ (n₂ = 8 ∨ n₂ = 1/2) → n₁ = n₂ :=
sorry

end number_of_valid_ns_l37_37258


namespace whale_consumption_third_hour_l37_37959

theorem whale_consumption_third_hour (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 450) → ((x + 6) = 90) :=
by
  intro h
  sorry

end whale_consumption_third_hour_l37_37959


namespace find_width_l37_37841

-- Definition of the perimeter of a rectangle
def perimeter (L W : ℝ) : ℝ := 2 * (L + W)

-- The given conditions
def length := 13
def perimeter_value := 50

-- The goal to prove: if the perimeter is 50 and the length is 13, then the width must be 12
theorem find_width :
  ∃ (W : ℝ), perimeter length W = perimeter_value ∧ W = 12 :=
by
  sorry

end find_width_l37_37841


namespace least_number_to_subtract_l37_37367

theorem least_number_to_subtract (n : ℕ) (k : ℕ) (r : ℕ) (h : n = 3674958423) (div : k = 47) (rem : r = 30) :
  (n % k = r) → 3674958423 % 47 = 30 :=
by
  sorry

end least_number_to_subtract_l37_37367


namespace train_length_l37_37323

-- Define the conditions
def equal_length_trains (L : ℝ) : Prop :=
  ∃ (length : ℝ), length = L

def train_speeds : Prop :=
  ∃ v_fast v_slow : ℝ, v_fast = 46 ∧ v_slow = 36

def pass_time (t : ℝ) : Prop :=
  t = 36

-- The proof problem
theorem train_length (L : ℝ) 
  (h_equal_length : equal_length_trains L) 
  (h_speeds : train_speeds)
  (h_time : pass_time 36) : 
  L = 50 :=
sorry

end train_length_l37_37323


namespace required_extra_money_l37_37690

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l37_37690


namespace obtuse_triangle_iff_l37_37236

theorem obtuse_triangle_iff (x : ℝ) :
    (x > 1 ∧ x < 3) ↔ (x + (x + 1) > (x + 2) ∧
                        (x + 1) + (x + 2) > x ∧
                        (x + 2) + x > (x + 1) ∧
                        (x + 2)^2 > x^2 + (x + 1)^2) :=
by
  sorry

end obtuse_triangle_iff_l37_37236


namespace car_collision_frequency_l37_37362

theorem car_collision_frequency
  (x : ℝ)
  (h_collision : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * x)
  (h_big_crash : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * 20)
  (h_total_accidents : 240 / x + 240 / 20 = 36) :
  x = 10 :=
by
  sorry

end car_collision_frequency_l37_37362


namespace mary_books_end_of_year_l37_37102

def total_books_end_of_year (books_start : ℕ) (book_club : ℕ) (lent_to_jane : ℕ) 
 (returned_by_alice : ℕ) (bought_5th_month : ℕ) (bought_yard_sales : ℕ) 
 (birthday_daughter : ℕ) (birthday_mother : ℕ) (received_sister : ℕ)
 (buy_one_get_one : ℕ) (donated_charity : ℕ) (borrowed_neighbor : ℕ)
 (sold_used_store : ℕ) : ℕ :=
  books_start + book_club - lent_to_jane + returned_by_alice + bought_5th_month + bought_yard_sales +
  birthday_daughter + birthday_mother + received_sister + buy_one_get_one - donated_charity - borrowed_neighbor - sold_used_store

theorem mary_books_end_of_year : total_books_end_of_year 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end mary_books_end_of_year_l37_37102


namespace no_right_angle_sequence_l37_37916

theorem no_right_angle_sequence 
  (A B C : Type)
  (angle_A angle_B angle_C : ℝ)
  (angle_A_eq : angle_A = 59)
  (angle_B_eq : angle_B = 61)
  (angle_C_eq : angle_C = 60)
  (midpoint : A → A → A)
  (A0 B0 C0 : A) :
  ¬ ∃ n : ℕ, ∃ An Bn Cn : A, 
    (An = midpoint Bn Cn) ∧ 
    (Bn = midpoint An Cn) ∧ 
    (Cn = midpoint An Bn) ∧ 
    (angle_A = 90 ∨ angle_B = 90 ∨ angle_C = 90) :=
sorry

end no_right_angle_sequence_l37_37916


namespace apples_needed_for_two_weeks_l37_37702

theorem apples_needed_for_two_weeks :
  ∀ (apples_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ),
  apples_per_day = 1 → days_per_week = 7 → weeks = 2 →
  apples_per_day * days_per_week * weeks = 14 :=
by
  intros apples_per_day days_per_week weeks h1 h2 h3
  sorry

end apples_needed_for_two_weeks_l37_37702


namespace triangle_perimeter_l37_37839

theorem triangle_perimeter :
  let a := 15
  let b := 10
  let c := 12
  (a < b + c) ∧ (b < a + c) ∧ (c < a + b) →
  (a + b + c = 37) :=
by
  intros
  sorry

end triangle_perimeter_l37_37839


namespace crayons_per_child_l37_37014

theorem crayons_per_child (children : ℕ) (total_crayons : ℕ) (h1 : children = 18) (h2 : total_crayons = 216) : 
    total_crayons / children = 12 := 
by
  sorry

end crayons_per_child_l37_37014


namespace weight_of_33rd_weight_l37_37749

theorem weight_of_33rd_weight :
  ∃ a : ℕ → ℕ, (∀ k, a k < a (k+1)) ∧
               (∀ k ≤ 29, a k + a (k+3) = a (k+1) + a (k+2)) ∧
               a 2 = 9 ∧
               a 8 = 33 ∧
               a 32 = 257 :=
sorry

end weight_of_33rd_weight_l37_37749


namespace union_of_sets_l37_37327

noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 4, 6}

theorem union_of_sets : A ∪ B = {1, 2, 4, 6} := 
by 
sorry

end union_of_sets_l37_37327


namespace range_of_b_l37_37996

theorem range_of_b (a b : ℝ) (h1 : a ≠ 0) (h2 : a * b^2 > a) (h3 : a > a * b) : b < -1 :=
sorry

end range_of_b_l37_37996


namespace height_of_fifth_tree_l37_37506

theorem height_of_fifth_tree 
  (h₁ : tallest_tree = 108) 
  (h₂ : second_tallest_tree = 54 - 6) 
  (h₃ : third_tallest_tree = second_tallest_tree / 4) 
  (h₄ : fourth_shortest_tree = (second_tallest_tree + third_tallest_tree) - 2) 
  (h₅ : fifth_tree = 0.75 * (tallest_tree + second_tallest_tree + third_tallest_tree + fourth_shortest_tree)) : 
  fifth_tree = 169.5 :=
by
  sorry

end height_of_fifth_tree_l37_37506


namespace find_x_l37_37863

theorem find_x (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 3 * x → x = 3 / 2 := by
  sorry

end find_x_l37_37863


namespace negation_of_prop_l37_37892

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 > x - 1) ↔ ∃ x : ℝ, x^2 ≤ x - 1 :=
sorry

end negation_of_prop_l37_37892


namespace sufficient_but_not_necessary_condition_l37_37756

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l37_37756


namespace cost_to_paint_cube_l37_37122

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_to_paint_cube_l37_37122


namespace hancho_milk_l37_37442

def initial_milk : ℝ := 1
def ye_seul_milk : ℝ := 0.1
def ga_young_milk : ℝ := ye_seul_milk + 0.2
def remaining_milk : ℝ := 0.3

theorem hancho_milk : (initial_milk - (ye_seul_milk + ga_young_milk + remaining_milk)) = 0.3 :=
by
  sorry

end hancho_milk_l37_37442


namespace find_polynomial_l37_37538

theorem find_polynomial
  (M : ℝ → ℝ)
  (h : ∀ x, M x + 5 * x^2 - 4 * x - 3 = -1 * x^2 - 3 * x) :
  ∀ x, M x = -6 * x^2 + x + 3 :=
sorry

end find_polynomial_l37_37538


namespace expression_evaluation_l37_37975

theorem expression_evaluation :
  2 - 3 * (-4) + 5 - (-6) * 7 = 61 :=
sorry

end expression_evaluation_l37_37975


namespace reading_time_difference_l37_37820

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l37_37820


namespace min_num_stamps_is_17_l37_37822

-- Definitions based on problem conditions
def initial_num_stamps : ℕ := 2 + 5 + 3 + 1
def initial_cost : ℝ := 2 * 0.10 + 5 * 0.20 + 3 * 0.50 + 1 * 2
def remaining_cost : ℝ := 10 - initial_cost
def additional_stamps : ℕ := 2 + 2 + 1 + 1
def total_stamps : ℕ := initial_num_stamps + additional_stamps

-- Proof that the minimum number of stamps bought is 17
theorem min_num_stamps_is_17 : total_stamps = 17 := by
  sorry

end min_num_stamps_is_17_l37_37822


namespace alice_burger_spending_l37_37358

theorem alice_burger_spending :
  let daily_burgers := 4
  let burger_cost := 13
  let days_in_june := 30
  let mondays_wednesdays := 8
  let fridays := 4
  let fifth_purchase_coupons := 6
  let discount_10_percent := 0.9
  let discount_50_percent := 0.5
  let full_price := days_in_june * daily_burgers * burger_cost
  let discount_10 := mondays_wednesdays * daily_burgers * burger_cost * discount_10_percent
  let fridays_cost := (daily_burgers - 1) * fridays * burger_cost
  let discount_50 := fifth_purchase_coupons * burger_cost * discount_50_percent
  full_price - discount_10 - fridays_cost - discount_50 + fridays_cost = 1146.6 := by sorry

end alice_burger_spending_l37_37358


namespace sum_last_three_coefficients_l37_37518

theorem sum_last_three_coefficients :
  let expr := (λ (a : ℚ), (1 - 1 / a)^8)
  let coefficients := [1, -8, 28] in
  coefficients.sum = 21 :=
by
  sorry

end sum_last_three_coefficients_l37_37518


namespace sequence_8123_appears_l37_37761

theorem sequence_8123_appears :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) + a (n-4)) % 10) ∧
  (a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4) ∧
  (∃ n, a n = 8 ∧ a (n+1) = 1 ∧ a (n+2) = 2 ∧ a (n+3) = 3) :=
sorry

end sequence_8123_appears_l37_37761


namespace geometric_sequence_first_term_l37_37643

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r^2 = 3)
  (h2 : a * r^4 = 27) : 
  a = - (real.sqrt 9) / 9 :=
by
  sorry

end geometric_sequence_first_term_l37_37643


namespace arrange_digits_l37_37848

theorem arrange_digits (A B C D E F : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E) (h5 : A ≠ F)
  (h6 : B ≠ C) (h7 : B ≠ D) (h8 : B ≠ E) (h9 : B ≠ F)
  (h10 : C ≠ D) (h11 : C ≠ E) (h12 : C ≠ F)
  (h13 : D ≠ E) (h14 : D ≠ F) (h15 : E ≠ F)
  (range_A : 1 ≤ A ∧ A ≤ 6) (range_B : 1 ≤ B ∧ B ≤ 6) (range_C : 1 ≤ C ∧ C ≤ 6)
  (range_D : 1 ≤ D ∧ D ≤ 6) (range_E : 1 ≤ E ∧ E ≤ 6) (range_F : 1 ≤ F ∧ F ≤ 6)
  (sum_line1 : A + D + E = 15) (sum_line2 : A + C + 9 = 15) 
  (sum_line3 : B + D + 9 = 15) (sum_line4 : 7 + C + E = 15) 
  (sum_line5 : 9 + C + A = 15) (sum_line6 : A + 8 + F = 15) 
  (sum_line7 : 7 + D + F = 15) : 
  (A = 4) ∧ (B = 1) ∧ (C = 2) ∧ (D = 5) ∧ (E = 6) ∧ (F = 3) :=
sorry

end arrange_digits_l37_37848


namespace remainder_of_expression_l37_37143

theorem remainder_of_expression (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := 
by 
  sorry

end remainder_of_expression_l37_37143


namespace number_of_friends_l37_37462

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37462


namespace parabola_equation_l37_37567

def is_parabola (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

def has_vertex (h k a b c : ℝ) : Prop :=
  b = -2 * a * h ∧ c = k + a * h^2 

def contains_point (a b c x y : ℝ) : Prop :=
  y = a*x^2 + b*x + c

theorem parabola_equation (a b c : ℝ) :
  has_vertex 3 (-2) a b c ∧ contains_point a b c 5 6 → 
  a = 2 ∧ b = -12 ∧ c = 16 := by
  sorry

end parabola_equation_l37_37567


namespace region_midpoint_area_equilateral_triangle_52_36_l37_37979

noncomputable def equilateral_triangle (A B C: ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

def midpoint_region_area (a b c : ℝ × ℝ) : ℝ := sorry

theorem region_midpoint_area_equilateral_triangle_52_36 (A B C: ℝ × ℝ) (h: equilateral_triangle A B C) :
  let m := (midpoint_region_area A B C)
  100 * m = 52.36 :=
sorry

end region_midpoint_area_equilateral_triangle_52_36_l37_37979


namespace x3_plus_y3_values_l37_37695

noncomputable def x_y_satisfy_eqns (x y : ℝ) : Prop :=
  y^2 - 3 = (x - 3)^3 ∧ x^2 - 3 = (y - 3)^2 ∧ x ≠ y

theorem x3_plus_y3_values (x y : ℝ) (h : x_y_satisfy_eqns x y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 :=
  sorry

end x3_plus_y3_values_l37_37695


namespace range_of_a_l37_37599

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) →
  a ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
by
  sorry

end range_of_a_l37_37599


namespace min_value_of_expression_l37_37280

noncomputable def smallest_value (a b c : ℕ) : ℤ :=
  3 * a - 2 * a * b + a * c

theorem min_value_of_expression : ∃ (a b c : ℕ), 0 < a ∧ a < 7 ∧ 0 < b ∧ b ≤ 3 ∧ 0 < c ∧ c ≤ 4 ∧ smallest_value a b c = -12 := by
  sorry

end min_value_of_expression_l37_37280


namespace dog_catches_rabbit_in_4_minutes_l37_37359

noncomputable def time_to_catch_up (dog_speed rabbit_speed head_start : ℝ) : ℝ :=
  head_start / (dog_speed - rabbit_speed) * 60

theorem dog_catches_rabbit_in_4_minutes :
  time_to_catch_up 24 15 0.6 = 4 :=
by
  unfold time_to_catch_up
  norm_num
  rfl

end dog_catches_rabbit_in_4_minutes_l37_37359


namespace disease_given_positive_l37_37901

-- Definitions and conditions extracted from the problem
def Pr_D : ℚ := 1 / 200
def Pr_Dc : ℚ := 1 - Pr_D
def Pr_T_D : ℚ := 1
def Pr_T_Dc : ℚ := 0.05

-- Derived probabilites from given conditions
def Pr_T : ℚ := Pr_T_D * Pr_D + Pr_T_Dc * Pr_Dc

-- Statement for the probability using Bayes' theorem
theorem disease_given_positive :
  (Pr_T_D * Pr_D) / Pr_T = 20 / 219 :=
sorry

end disease_given_positive_l37_37901


namespace winning_percentage_l37_37249

/-- In an election with two candidates, wherein the winner received 490 votes and won by 280 votes,
we aim to prove that the winner received 70% of the total votes. -/

theorem winning_percentage (votes_winner : ℕ) (votes_margin : ℕ) (total_votes : ℕ)
  (h1 : votes_winner = 490) (h2 : votes_margin = 280)
  (h3 : total_votes = votes_winner + (votes_winner - votes_margin)) :
  (votes_winner * 100 / total_votes) = 70 :=
by
  -- Skipping the proof for now
  sorry

end winning_percentage_l37_37249


namespace jack_buttons_l37_37428

theorem jack_buttons :
  ∀ (shirts_per_kid kids buttons_per_shirt : ℕ),
  shirts_per_kid = 3 →
  kids = 3 →
  buttons_per_shirt = 7 →
  (shirts_per_kid * kids * buttons_per_shirt) = 63 :=
by
  intros shirts_per_kid kids buttons_per_shirt h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 3 * 7 = 9 * 7 : by rw mul_assoc
            ... = 63   : by norm_num

end jack_buttons_l37_37428


namespace arithmetic_series_product_l37_37770

theorem arithmetic_series_product (a b c : ℝ) (h1 : a = b - d) (h2 : c = b + d) (h3 : a * b * c = 125) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) : b ≥ 5 :=
sorry

end arithmetic_series_product_l37_37770


namespace billy_trays_l37_37180

def trays_needed (total_ice_cubes : ℕ) (ice_cubes_per_tray : ℕ) : ℕ :=
  total_ice_cubes / ice_cubes_per_tray

theorem billy_trays (total_ice_cubes ice_cubes_per_tray : ℕ) (h1 : total_ice_cubes = 72) (h2 : ice_cubes_per_tray = 9) :
  trays_needed total_ice_cubes ice_cubes_per_tray = 8 :=
by
  sorry

end billy_trays_l37_37180


namespace average_difference_l37_37120

theorem average_difference : 
  (500 + 1000) / 2 - (100 + 500) / 2 = 450 := 
by
  sorry

end average_difference_l37_37120


namespace even_function_m_value_l37_37742

def f (x m : ℝ) : ℝ := (x - 2) * (x - m)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f x m = f (-x) m) → m = -2 := by
  sorry

end even_function_m_value_l37_37742


namespace no_adjacent_abc_seating_l37_37751

theorem no_adjacent_abc_seating : 
  let total_arrangements := Nat.factorial 8
  let abc_unit_arrangements := Nat.factorial 3
  let reduced_arrangements := Nat.factorial 6
  total_arrangements - reduced_arrangements * abc_unit_arrangements = 36000 :=
by 
  sorry

end no_adjacent_abc_seating_l37_37751


namespace find_f_2015_l37_37042

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2015
  (h1 : ∀ x, f (-x) = -f x) -- f is an odd function
  (h2 : ∀ x, f (x + 2) = -f x) -- f(x+2) = -f(x)
  (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) -- f(x) = 2x^2 for x in (0, 2)
  : f 2015 = -2 :=
sorry

end find_f_2015_l37_37042


namespace polynomial_composite_l37_37117

theorem polynomial_composite (x : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 4 * x^3 + 6 * x^2 + 4 * x + 1 = a * b :=
by
  sorry

end polynomial_composite_l37_37117


namespace number_of_friends_l37_37464

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37464


namespace choose_13_3_equals_286_l37_37231

theorem choose_13_3_equals_286 : (nat.choose 13 3) = 286 :=
by
  sorry

end choose_13_3_equals_286_l37_37231


namespace AY_is_2_sqrt_55_l37_37532

noncomputable def AY_length : ℝ :=
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  2 * Real.sqrt (rA^2 + BD^2)

theorem AY_is_2_sqrt_55 :
  AY_length = 2 * Real.sqrt 55 :=
by
  -- Assuming the given problem's conditions.
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  show AY_length = 2 * Real.sqrt 55
  sorry

end AY_is_2_sqrt_55_l37_37532


namespace smallest_sum_l37_37210

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37210


namespace probability_of_drawing_white_ball_l37_37176

theorem probability_of_drawing_white_ball 
  (total_balls : ℕ) (white_balls : ℕ) 
  (h_total : total_balls = 9) (h_white : white_balls = 4) : 
  (white_balls : ℚ) / total_balls = 4 / 9 := 
by 
  sorry

end probability_of_drawing_white_ball_l37_37176


namespace total_buttons_l37_37426

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l37_37426


namespace symmetric_point_coordinates_l37_37251

theorem symmetric_point_coordinates (M : ℝ × ℝ) (N : ℝ × ℝ) (hM : M = (1, -2)) (h_sym : N = (-M.1, -M.2)) :
  N = (-1, 2) :=
by sorry

end symmetric_point_coordinates_l37_37251


namespace eight_points_in_circle_distance_lt_one_l37_37763

noncomputable theory

open set
open metric

def circle : set (euclidean_space (fin 2)) :=
  {p | ∥p∥ ≤ 1}

theorem eight_points_in_circle_distance_lt_one
  (ps : fin 8 → euclidean_space (fin 2))
  (hps : ∀ i, ps i ∈ circle):
  ∃ (i j : fin 8), i ≠ j ∧ dist (ps i) (ps j) < 1 :=
sorry

end eight_points_in_circle_distance_lt_one_l37_37763


namespace min_value_f_l37_37034

def f (x : ℝ) : ℝ := |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem min_value_f : (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) := 
sorry

end min_value_f_l37_37034


namespace correct_sampling_probability_correct_l37_37515

structure City :=
  (factories_A factories_B factories_C : ℕ)
  (total_factories : ℕ)
  (sampling_ratio : ℚ)

def myCity : City :=
  { factories_A := 9,
    factories_B := 18,
    factories_C := 18,
    total_factories := 45,
    sampling_ratio := 1 / 9 }

def sampled_factories (c : City) : ℕ × ℕ × ℕ :=
  (c.factories_A * c.sampling_ratio.num / c.sampling_ratio.denom,
   c.factories_B * c.sampling_ratio.num / c.sampling_ratio.denom,
   c.factories_C * c.sampling_ratio.num / c.sampling_ratio.denom)

theorem correct_sampling (c : City) :
  sampled_factories c = (1, 2, 2) :=
by
  cases c
  simp [sampled_factories, myCity, sampling_ratio]
  sorry

def pairs_and_probability : ℚ :=
  let pairs := [("A", "B1"), ("A", "B2"), ("A", "C1"), ("A", "C2"), ("B1", "B2"), ("B1", "C1"),
                ("B1", "C2"), ("B2", "C1"), ("B2", "C2"), ("C1", "C2")]
  let pairs_with_C := [("A", "C1"), ("A", "C2"), ("B1", "C1"), ("B1", "C2"), ("B2", "C1"), ("B2", "C2"),
                       ("C1", "C2")]
  pairs_with_C.length / pairs.length

theorem probability_correct :
  pairs_and_probability = 7 / 10 :=
by
  simp [pairs_and_probability]
  sorry

end correct_sampling_probability_correct_l37_37515


namespace endpoint_coordinates_l37_37838

theorem endpoint_coordinates (x y : ℝ) (h : y > 0) :
  let slope_condition := (y - 2) / (x - 2) = 3 / 4
  let distance_condition := (x - 2) ^ 2 + (y - 2) ^ 2 = 64
  slope_condition → distance_condition → 
    (x = 2 + (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 + (4 * Real.sqrt 5475) / 25) + 1 / 2) ∨
    (x = 2 - (4 * Real.sqrt 5475) / 25 ∧ y = (3 / 4) * (2 - (4 * Real.sqrt 5475) / 25) + 1 / 2) :=
by
  intros slope_condition distance_condition
  sorry

end endpoint_coordinates_l37_37838


namespace factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l37_37980

-- Proof Problem 1
theorem factorize_diff_squares_1 (x y : ℝ) :
  4 * x^2 - 9 * y^2 = (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

-- Proof Problem 2
theorem factorize_diff_squares_2 (a b : ℝ) :
  -16 * a^2 + 25 * b^2 = (5 * b + 4 * a) * (5 * b - 4 * a) :=
sorry

-- Proof Problem 3
theorem factorize_common_term (x y : ℝ) :
  x^3 * y - x * y^3 = x * y * (x + y) * (x - y) :=
sorry

end factorize_diff_squares_1_factorize_diff_squares_2_factorize_common_term_l37_37980


namespace cats_to_dogs_ratio_l37_37939

theorem cats_to_dogs_ratio (cats dogs : ℕ) (h1 : 2 * dogs = 3 * cats) (h2 : cats = 14) : dogs = 21 :=
by
  sorry

end cats_to_dogs_ratio_l37_37939


namespace find_divisor_l37_37152

theorem find_divisor (dividend remainder quotient : ℕ) (h1 : dividend = 76) (h2 : remainder = 8) (h3 : quotient = 4) : ∃ d : ℕ, dividend = (d * quotient) + remainder ∧ d = 17 :=
by
  sorry

end find_divisor_l37_37152


namespace part1_unique_zero_part2_inequality_l37_37390

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

theorem part1_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

theorem part2_inequality (n : ℕ) (h : n > 0) : 
  Real.log ((n + 1) / n) < 1 / Real.sqrt (n^2 + n) := by
  sorry

end part1_unique_zero_part2_inequality_l37_37390


namespace number_of_red_balls_l37_37604

theorem number_of_red_balls (x : ℕ) (h₀ : 4 > 0) (h₁ : (x : ℝ) / (x + 4) = 0.6) : x = 6 :=
sorry

end number_of_red_balls_l37_37604


namespace smallest_possible_sum_l37_37199

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37199


namespace congruent_triangles_count_l37_37895

open Set

variables (g l : Line) (A B C : Point)

def number_of_congruent_triangles (g l : Line) (A B C : Point) : ℕ :=
  16

theorem congruent_triangles_count (g l : Line) (A B C : Point) :
  number_of_congruent_triangles g l A B C = 16 :=
sorry

end congruent_triangles_count_l37_37895


namespace range_of_a_l37_37391

def f (x : ℝ) : ℝ := x^3 + x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f (x^2 + a) + f (a * x) > 2) → 0 < a ∧ a < 4 := 
by 
  sorry

end range_of_a_l37_37391


namespace max_possible_value_of_gcd_l37_37278

theorem max_possible_value_of_gcd (n : ℕ) : gcd ((8^n - 1) / 7) ((8^(n+1) - 1) / 7) = 1 := by
  sorry

end max_possible_value_of_gcd_l37_37278


namespace possible_values_of_p1_l37_37537

noncomputable def p (x : ℝ) (n : ℕ) : ℝ := sorry

axiom deg_p (n : ℕ) (h : n ≥ 2) (x : ℝ) : x^n = 1

axiom roots_le_one (r : ℝ) : r ≤ 1

axiom p_at_2 (n : ℕ) (h : n ≥ 2) : p 2 n = 3^n

theorem possible_values_of_p1 (n : ℕ) (h : n ≥ 2) : p 1 n = 0 ∨ p 1 n = (-1)^n * 2^n :=
by
  sorry

end possible_values_of_p1_l37_37537


namespace quadratic_real_roots_l37_37898

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots (k : ℝ) :
  discriminant (k - 1) 4 2 ≥ 0 ↔ k ≤ 3 ∧ k ≠ 1 :=
by
  sorry

end quadratic_real_roots_l37_37898


namespace no_solution_l37_37708

theorem no_solution (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : 
  ¬ (x^2 + y^2 + 41 = 2^n) :=
by sorry

end no_solution_l37_37708


namespace smallest_sum_l37_37211

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37211


namespace range_of_a_l37_37723

theorem range_of_a(p q: Prop)
  (hp: p ↔ (a = 0 ∨ (0 < a ∧ a < 4)))
  (hq: q ↔ (-1 < a ∧ a < 3))
  (hpor: p ∨ q)
  (hpand: ¬(p ∧ q)):
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := by sorry

end range_of_a_l37_37723


namespace satellite_modular_units_24_l37_37967

-- Define basic parameters
variables (U N S : ℕ)
def fraction_upgraded : ℝ := 0.2

-- Define the conditions as Lean premises
axiom non_upgraded_per_unit_eq_sixth_total_upgraded : N = S / 6
axiom fraction_sensors_upgraded : (S : ℝ) = fraction_upgraded * (S + U * N)

-- The main statement to be proved
theorem satellite_modular_units_24 (h1 : N = S / 6) (h2 : (S : ℝ) = fraction_upgraded * (S + U * N)) : U = 24 :=
by
  -- The actual proof steps will be written here.
  sorry

end satellite_modular_units_24_l37_37967


namespace residue_of_minus_963_plus_100_mod_35_l37_37555

-- Defining the problem in Lean 4
theorem residue_of_minus_963_plus_100_mod_35 : 
  ((-963 + 100) % 35) = 12 :=
by
  sorry

end residue_of_minus_963_plus_100_mod_35_l37_37555


namespace minimum_value_of_f_l37_37053

noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem minimum_value_of_f :
  ∃ x ∈ Set.Icc (1/4 : ℝ) (5/4 : ℝ), f x = (4 * Real.sqrt 5 / 5 - 2 * Real.sqrt 10 / 5) :=
sorry

end minimum_value_of_f_l37_37053


namespace problem1_problem2_l37_37393

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + a / (x + 1) + a / 2

theorem problem1 (a : ℝ) (h : a = 9 / 2) :
  (∀ x, 0 < x < 1 / 2 → (f' a x > 0)) ∧ 
  (∀ x, 2 < x → (f' a x > 0)) ∧ 
  (∀ x, 1 / 2 < x < 2 → (f' a x < 0)) :=
sorry

theorem problem2 (a : ℝ) (h : ∀ x, 0 < x → f a x ≤ (a / 2) * (x + 1)) : 
  a = 4 / 3 :=
sorry

-- Helper definition of the derivative f'
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ :=
  (2 * x - 1) * (x - 2) / (2 * x * (x + 1) ^ 2)

end problem1_problem2_l37_37393


namespace find_x_l37_37573

theorem find_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
by
  sorry

end find_x_l37_37573


namespace louis_age_currently_31_l37_37411

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l37_37411


namespace missing_number_evaluation_l37_37017

theorem missing_number_evaluation (x : ℝ) (h : |4 + 9 * x| - 6 = 70) : x = 8 :=
sorry

end missing_number_evaluation_l37_37017


namespace watch_current_price_l37_37335

-- Definitions based on conditions
def original_price : ℝ := 15
def first_reduction_rate : ℝ := 0.25
def second_reduction_rate : ℝ := 0.40

-- The price after the first reduction
def first_reduced_price : ℝ := original_price * (1 - first_reduction_rate)

-- The price after the second reduction
def final_price : ℝ := first_reduced_price * (1 - second_reduction_rate)

-- The theorem that needs to be proved
theorem watch_current_price : final_price = 6.75 :=
by
  -- Proof goes here
  sorry

end watch_current_price_l37_37335


namespace area_excluding_holes_l37_37669

theorem area_excluding_holes (x : ℝ) :
  let A_large : ℝ := (x + 8) * (x + 6)
  let A_hole : ℝ := (2 * x - 4) * (x - 3)
  A_large - 2 * A_hole = -3 * x^2 + 34 * x + 24 := by
  sorry

end area_excluding_holes_l37_37669


namespace solution_l37_37705

noncomputable def problem : Prop :=
  cos (45 * real.pi / 180) * cos (15 * real.pi / 180) + sin (45 * real.pi / 180) * sin (15 * real.pi / 180) = sqrt 3 / 2

theorem solution : problem := by
  sorry

end solution_l37_37705


namespace ab_is_square_l37_37485

theorem ab_is_square (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_main : a + b = b * (a - c)) (h_prime : ∃ p : ℕ, Prime p ∧ c + 1 = p^2) :
  ∃ k : ℕ, a + b = k^2 :=
by
  sorry

end ab_is_square_l37_37485


namespace cost_difference_of_buses_l37_37421

-- Definitions from the conditions
def bus_cost_equations (x y : ℝ) :=
  (x + 2 * y = 260) ∧ (2 * x + y = 280)

-- The statement to prove
theorem cost_difference_of_buses (x y : ℝ) (h : bus_cost_equations x y) :
  x - y = 20 :=
sorry

end cost_difference_of_buses_l37_37421


namespace find_smallest_n_l37_37666

theorem find_smallest_n : ∃ n : ℕ, (n - 4)^3 > (n^3 / 2) ∧ ∀ m : ℕ, m < n → (m - 4)^3 ≤ (m^3 / 2) :=
by
  sorry

end find_smallest_n_l37_37666


namespace pie_chart_degrees_for_cherry_pie_l37_37077

theorem pie_chart_degrees_for_cherry_pie :
  ∀ (total_students chocolate_pie apple_pie blueberry_pie : ℕ)
    (remaining_students cherry_pie_students lemon_pie_students : ℕ),
    total_students = 40 →
    chocolate_pie = 15 →
    apple_pie = 10 →
    blueberry_pie = 7 →
    remaining_students = total_students - chocolate_pie - apple_pie - blueberry_pie →
    cherry_pie_students = remaining_students / 2 →
    lemon_pie_students = remaining_students / 2 →
    (cherry_pie_students : ℝ) / (total_students : ℝ) * 360 = 36 :=
by
  sorry

end pie_chart_degrees_for_cherry_pie_l37_37077


namespace eight_points_in_circle_l37_37764

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end eight_points_in_circle_l37_37764


namespace bucket_weight_l37_37531

variable {p q x y : ℝ}

theorem bucket_weight (h1 : x + (1 / 4) * y = p) (h2 : x + (3 / 4) * y = q) :
  x + y = - (1 / 2) * p + (3 / 2) * q := by
  sorry

end bucket_weight_l37_37531


namespace correlation_identification_l37_37299

noncomputable def relationship (a b : Type) : Prop := 
  ∃ (f : a → b), true

def correlation (a b : Type) : Prop :=
  relationship a b ∧ relationship b a

def deterministic (a b : Type) : Prop :=
  ∀ x y : a, ∃! z : b, true

def age_wealth : Prop := correlation ℕ ℝ
def point_curve_coordinates : Prop := deterministic (ℝ × ℝ) (ℝ × ℝ)
def apple_production_climate : Prop := correlation ℝ ℝ
def tree_diameter_height : Prop := correlation ℝ ℝ

theorem correlation_identification :
  age_wealth ∧ apple_production_climate ∧ tree_diameter_height ∧ ¬point_curve_coordinates := 
by
  -- proof of these properties
  sorry

end correlation_identification_l37_37299


namespace handshake_remainder_l37_37747

noncomputable def handshakes (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_remainder :
  handshakes 12 3 % 1000 = 850 :=
sorry

end handshake_remainder_l37_37747


namespace simplify_expression_l37_37294

variable (x : ℝ)

theorem simplify_expression :
  (2 * x + 25) + (150 * x + 35) + (50 * x + 10) = 202 * x + 70 :=
sorry

end simplify_expression_l37_37294


namespace imaginary_unit_squared_in_set_l37_37885

-- Conditions of the problem
def imaginary_unit (i : ℂ) : Prop := i^2 = -1
def S : Set ℂ := {-1, 0, 1}

-- The statement to prove
theorem imaginary_unit_squared_in_set {i : ℂ} (hi : imaginary_unit i) : i^2 ∈ S := sorry

end imaginary_unit_squared_in_set_l37_37885


namespace binomial_133_133_l37_37186

theorem binomial_133_133 : @Nat.choose 133 133 = 1 := by   
sorry

end binomial_133_133_l37_37186


namespace order_of_logs_l37_37277

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem order_of_logs (a_def : a = Real.log 6 / Real.log 3)
                      (b_def : b = Real.log 10 / Real.log 5)
                      (c_def : c = Real.log 14 / Real.log 7) : a > b ∧ b > c := 
by
  sorry

end order_of_logs_l37_37277


namespace ethanol_percentage_fuel_B_l37_37971

noncomputable def percentage_ethanol_in_fuel_B : ℝ :=
  let tank_capacity := 208
  let ethanol_in_fuelA := 0.12
  let total_ethanol := 30
  let volume_fuelA := 82
  let ethanol_from_fuelA := volume_fuelA * ethanol_in_fuelA
  let ethanol_from_fuelB := total_ethanol - ethanol_from_fuelA
  let volume_fuelB := tank_capacity - volume_fuelA
  (ethanol_from_fuelB / volume_fuelB) * 100

theorem ethanol_percentage_fuel_B :
  percentage_ethanol_in_fuel_B = 16 :=
by
  sorry

end ethanol_percentage_fuel_B_l37_37971


namespace arithmetic_sequence_ratio_l37_37030

theorem arithmetic_sequence_ratio (S T : ℕ → ℕ) (a b : ℕ → ℕ)
  (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) :
  a 8 / b 8 = 6 :=
by
  sorry

end arithmetic_sequence_ratio_l37_37030


namespace cost_of_coffee_B_per_kg_l37_37968

-- Define the cost of coffee A per kilogram
def costA : ℝ := 10

-- Define the amount of coffee A used in the mixture
def amountA : ℝ := 240

-- Define the amount of coffee B used in the mixture
def amountB : ℝ := 240

-- Define the total amount of the mixture
def totalAmount : ℝ := 480

-- Define the selling price of the mixture per kilogram
def sellingPrice : ℝ := 11

-- Define the cost of coffee B per kilogram as a variable B
variable (B : ℝ)

-- Define the total cost of the mixture
def totalCost : ℝ := totalAmount * sellingPrice

-- Define the cost of coffee A used
def costOfA : ℝ := amountA * costA

-- Define the cost of coffee B used as total cost minus the cost of A
def costOfB : ℝ := totalCost - costOfA

-- Calculate the cost of coffee B per kilogram
theorem cost_of_coffee_B_per_kg : B = 12 :=
by
  have h1 : costOfA = 2400 := by sorry
  have h2 : totalCost = 5280 := by sorry
  have h3 : costOfB = 2880 := by sorry
  have h4 : B = costOfB / amountB := by sorry
  have h5 : B = 2880 / 240 := by sorry
  have h6 : B = 12 := by sorry
  exact h6

end cost_of_coffee_B_per_kg_l37_37968


namespace find_m_l37_37887

theorem find_m (x y m : ℝ) (h₁ : x - 2 * y = m) (h₂ : x = 2) (h₃ : y = 1) : m = 0 :=
by 
  -- Proof omitted
  sorry

end find_m_l37_37887


namespace dogs_grouping_l37_37792

theorem dogs_grouping (dogs : Finset α) (fluffy nipper : α) :
  dogs.card = 12 ∧ fluffy ∈ dogs ∧ nipper ∈ dogs →
  ∃ g1 g2 g3 : Finset α,
    (g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3) ∧
    (fluffy ∈ g1) ∧ (nipper ∈ g2) ∧
    (g1 ∪ g2 ∪ g3 = dogs) ∧ (g1 ∩ g2 = ∅) ∧ (g1 ∩ g3 = ∅) ∧ (g2 ∩ g3 = ∅) ∧
    (∃ n : ℕ, n = 4200) :=
by
  sorry

end dogs_grouping_l37_37792


namespace class_total_students_l37_37179

def initial_boys : ℕ := 15
def initial_girls : ℕ := (120 * initial_boys) / 100 -- 1.2 * initial_boys

def final_boys : ℕ := initial_boys
def final_girls : ℕ := 2 * initial_girls

def total_students : ℕ := final_boys + final_girls

theorem class_total_students : total_students = 51 := 
by 
  -- the actual proof will go here
  sorry

end class_total_students_l37_37179


namespace woman_weaves_ten_day_units_l37_37752

theorem woman_weaves_ten_day_units 
  (a₁ d : ℕ)
  (h₁ : 4 * a₁ + 6 * d = 24)
  (h₂ : a₁ + 6 * d = a₁ * (a₁ + d)) :
  a₁ + 9 * d = 21 := 
by
  sorry

end woman_weaves_ten_day_units_l37_37752


namespace wendy_pictures_l37_37516

theorem wendy_pictures (album1_pics rest_albums albums each_album_pics : ℕ)
    (h1 : album1_pics = 44)
    (h2 : rest_albums = 5)
    (h3 : each_album_pics = 7)
    (h4 : albums = rest_albums * each_album_pics)
    (h5 : albums = 5 * 7):
  album1_pics + albums = 79 :=
by
  -- We leave the proof as an exercise
  sorry

end wendy_pictures_l37_37516


namespace smallest_sum_l37_37212

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37212


namespace parabola_vertex_f_l37_37829

theorem parabola_vertex_f (d e f : ℝ) (h_vertex : ∀ y, (d * (y - 3)^2 + 5) = (d * y^2 + e * y + f))
  (h_point : d * (6 - 3)^2 + 5 = 2) : f = 2 :=
by
  sorry

end parabola_vertex_f_l37_37829


namespace main_theorem_l37_37868

noncomputable def problem_statement : Prop :=
  ∀ x : ℂ, (x ≠ -2) →
  ((15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48) ↔
  (x = 12 + 2 * Real.sqrt 38 ∨ x = 12 - 2 * Real.sqrt 38 ∨
  x = -1/2 + Complex.I * Real.sqrt 95 / 2 ∨
  x = -1/2 - Complex.I * Real.sqrt 95 / 2)

-- Provide the main statement without the proof
theorem main_theorem : problem_statement := sorry

end main_theorem_l37_37868


namespace speed_of_man_in_still_water_l37_37333

variables (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 5 = 36 ∧ (v_m - v_s) * 7 = 22 → v_m = 5.17 :=
by 
  sorry

end speed_of_man_in_still_water_l37_37333


namespace prime_ge_7_not_divisible_by_40_l37_37064

theorem prime_ge_7_not_divisible_by_40 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : ¬ (40 ∣ (p^3 - 1)) :=
sorry

end prime_ge_7_not_divisible_by_40_l37_37064


namespace temperature_rise_per_hour_l37_37116

-- Define the conditions
variables (x : ℕ) -- temperature rise per hour

-- Assume the given conditions
axiom power_outage : (3 : ℕ) * x = (6 : ℕ) * 4

-- State the proposition
theorem temperature_rise_per_hour : x = 8 :=
sorry

end temperature_rise_per_hour_l37_37116


namespace add_to_fraction_eq_l37_37316

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l37_37316


namespace groupDivisionWays_l37_37790

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l37_37790


namespace find_C_D_l37_37027

theorem find_C_D : ∃ C D, 
  (∀ x, x ≠ 3 → x ≠ 5 → (6*x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) ∧ 
  C = -15/2 ∧ D = 27/2 := by
  sorry

end find_C_D_l37_37027


namespace louis_age_currently_31_l37_37410

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l37_37410


namespace total_handshakes_l37_37943

theorem total_handshakes :
  let gremlins := 20
  let imps := 20
  let sprites := 10
  let handshakes_gremlins := gremlins * (gremlins - 1) / 2
  let handshakes_gremlins_imps := gremlins * imps
  let handshakes_imps_sprites := imps * sprites
  handshakes_gremlins + handshakes_gremlins_imps + handshakes_imps_sprites = 790 :=
by
  sorry

end total_handshakes_l37_37943


namespace find_interest_rate_l37_37293

theorem find_interest_rate
  (P : ℝ) (A : ℝ) (n t : ℕ) (hP : P = 3000) (hA : A = 3307.5) (hn : n = 2) (ht : t = 1) :
  ∃ r : ℝ, r = 10 :=
by
  sorry

end find_interest_rate_l37_37293


namespace sum_of_coordinates_of_intersection_l37_37004

def h : ℝ → ℝ := -- Define h(x). This would be specific to the function provided; we abstract it here for the proof.
sorry

theorem sum_of_coordinates_of_intersection (a b : ℝ) (h_eq: h a = h (a - 5)) : a + b = 6 :=
by
  -- We need a [step from the problem conditions], hence introducing the given conditions
  have : b = h a := sorry
  have : b = h (a - 5) := sorry
  exact sorry

end sum_of_coordinates_of_intersection_l37_37004


namespace mrs_white_expected_yield_l37_37623

noncomputable def orchard_yield : ℝ :=
  let length_in_feet : ℝ := 10 * 3
  let width_in_feet : ℝ := 30 * 3
  let total_area : ℝ := length_in_feet * width_in_feet
  let half_area : ℝ := total_area / 2
  let tomato_yield : ℝ := half_area * 0.75
  let cucumber_yield : ℝ := half_area * 0.4
  tomato_yield + cucumber_yield

theorem mrs_white_expected_yield :
  orchard_yield = 1552.5 := sorry

end mrs_white_expected_yield_l37_37623


namespace arrangement_non_adjacent_l37_37330

theorem arrangement_non_adjacent :
  let total_arrangements := Nat.factorial 30
  let adjacent_arrangements := 2 * Nat.factorial 29
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements = 28 * Nat.factorial 29 :=
by
  sorry

end arrangement_non_adjacent_l37_37330


namespace sum_interior_numbers_eight_l37_37600

noncomputable def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2 -- This is a general formula derived from the pattern

theorem sum_interior_numbers_eight :
  sum_interior_numbers 8 = 126 :=
by
  -- No proof required, so we use sorry.
  sorry

end sum_interior_numbers_eight_l37_37600


namespace exists_solution_real_l37_37712

theorem exists_solution_real (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3 / 2 :=
by
  sorry

end exists_solution_real_l37_37712


namespace smallest_x_plus_y_l37_37221

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37221


namespace correct_time_fraction_l37_37529

theorem correct_time_fraction : 
  let hours_with_glitch := [5]
  let minutes_with_glitch := [5, 15, 25, 35, 45, 55]
  let total_hours := 12
  let total_minutes_per_hour := 60
  let correct_hours := total_hours - hours_with_glitch.length
  let correct_minutes := total_minutes_per_hour - minutes_with_glitch.length
  (correct_hours * correct_minutes) / (total_hours * total_minutes_per_hour) = 33 / 40 :=
by
  sorry

end correct_time_fraction_l37_37529


namespace arith_seq_sum_nine_l37_37028

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arith_seq := ∀ n : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arith_seq_sum_nine (h_seq : arith_seq a) (h_sum : sum_first_n_terms a S) (h_S9 : S 9 = 18) : 
  a 2 + a 5 + a 8 = 6 :=
  sorry

end arith_seq_sum_nine_l37_37028


namespace total_buttons_l37_37425

-- Define the conditions
def shirts_per_kid : Nat := 3
def number_of_kids : Nat := 3
def buttons_per_shirt : Nat := 7

-- Define the statement to prove
theorem total_buttons : shirts_per_kid * number_of_kids * buttons_per_shirt = 63 := by
  sorry

end total_buttons_l37_37425


namespace q_0_plus_q_5_l37_37766

-- Define the properties of the polynomial q(x)
variable (q : ℝ → ℝ)
variable (monic_q : ∀ x, ∃ a b c d e f, a = 1 ∧ q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
variable (deg_q : ∀ x, degree q = 5)
variable (q_1 : q 1 = 26)
variable (q_2 : q 2 = 52)
variable (q_3 : q 3 = 78)

-- State the theorem to find q(0) + q(5)
theorem q_0_plus_q_5 : q 0 + q 5 = 58 :=
sorry

end q_0_plus_q_5_l37_37766


namespace initial_students_proof_l37_37682

def initial_students (e : ℝ) (transferred : ℝ) (left : ℝ) : ℝ :=
  e + transferred + left

theorem initial_students_proof : initial_students 28 10 4 = 42 :=
  by
    -- This is where the proof would go, but we use 'sorry' to skip it.
    sorry

end initial_students_proof_l37_37682


namespace unsuitable_temperature_for_refrigerator_l37_37799

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l37_37799


namespace height_difference_l37_37675

variable (H_A H_B : ℝ)

-- Conditions
axiom B_is_66_67_percent_more_than_A : H_B = H_A * 1.6667

-- Proof statement
theorem height_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.6667) : 
  (H_B - H_A) / H_B * 100 = 40 := by
sorry

end height_difference_l37_37675


namespace non_black_cows_l37_37782

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l37_37782


namespace ellipse_area_l37_37366

theorem ellipse_area :
  ∃ a b : ℝ, 
    (∀ x y : ℝ, (x^2 - 2 * x + 9 * y^2 + 18 * y + 16 = 0) → 
    (a = 2 ∧ b = (2 / 3) ∧ (π * a * b = 4 * π / 3))) :=
sorry

end ellipse_area_l37_37366


namespace smallest_x_plus_y_l37_37208

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37208


namespace number_of_friends_l37_37450

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37450


namespace student_correct_answers_l37_37526

-- Defining the conditions as variables and equations
def correct_answers (c w : ℕ) : Prop :=
  c + w = 60 ∧ 4 * c - w = 160

-- Stating the problem: proving the number of correct answers is 44
theorem student_correct_answers (c w : ℕ) (h : correct_answers c w) : c = 44 :=
by 
  sorry

end student_correct_answers_l37_37526


namespace arithmetic_sequence_third_term_l37_37772

theorem arithmetic_sequence_third_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  (S 5 = 10) ∧ (S n = n * (a 1 + a n) / 2) ∧ (a 5 = a 1 + 4 * d) ∧ 
  (∀ n, a n = a 1 + (n-1) * d) → (a 3 = 2) :=
by
  intro h
  sorry

end arithmetic_sequence_third_term_l37_37772


namespace smallest_x_plus_y_l37_37220

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37220


namespace smallest_x_plus_y_l37_37209

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37209


namespace no_heptagon_cross_section_l37_37319

-- Define what it means for a plane to intersect a cube and form a shape.
noncomputable def possible_cross_section_shapes (P : Plane) (C : Cube) : Set Polygon :=
  sorry -- Placeholder for the actual definition which involves geometric computations.

-- Prove that a heptagon cannot be one of the possible cross-sectional shapes of a cube.
theorem no_heptagon_cross_section (P : Plane) (C : Cube) : 
  Heptagon ∉ possible_cross_section_shapes P C :=
sorry -- Placeholder for the proof.

end no_heptagon_cross_section_l37_37319


namespace sum_of_solutions_l37_37369

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l37_37369


namespace probability_of_sum_17_is_correct_l37_37001

def probability_sum_17 : ℚ :=
  let favourable_outcomes := 2
  let total_outcomes := 81
  favourable_outcomes / total_outcomes

theorem probability_of_sum_17_is_correct :
  probability_sum_17 = 2 / 81 :=
by
  -- The proof steps are not required for this task
  sorry

end probability_of_sum_17_is_correct_l37_37001


namespace blocks_left_l37_37625

def blocks_initial := 78
def blocks_used := 19

theorem blocks_left : blocks_initial - blocks_used = 59 :=
by
  -- Solution is not required here, so we add a sorry placeholder.
  sorry

end blocks_left_l37_37625


namespace percent_defective_units_shipped_for_sale_l37_37608

theorem percent_defective_units_shipped_for_sale 
  (P : ℝ) -- total number of units produced
  (h_defective : 0.06 * P = d) -- 6 percent of units are defective
  (h_shipped : 0.0024 * P = s) -- 0.24 percent of units are defective units shipped for sale
  : (s / d) * 100 = 4 :=
by
  sorry

end percent_defective_units_shipped_for_sale_l37_37608


namespace little_john_money_left_l37_37150

-- Define the variables with the given conditions
def initAmount : ℚ := 5.10
def spentOnSweets : ℚ := 1.05
def givenToEachFriend : ℚ := 1.00

-- The problem statement
theorem little_john_money_left :
  (initAmount - spentOnSweets - 2 * givenToEachFriend) = 2.05 :=
by
  sorry

end little_john_money_left_l37_37150


namespace total_cartons_used_l37_37119

theorem total_cartons_used (x : ℕ) (y : ℕ) (h1 : y = 24) (h2 : 2 * x + 3 * y = 100) : x + y = 38 :=
sorry

end total_cartons_used_l37_37119


namespace rhombus_side_length_l37_37165

-- Define the conditions including the diagonals and area of the rhombus
def diagonal_ratio (d1 d2 : ℝ) : Prop := d1 = 3 * d2
def area_rhombus (b : ℝ) (K : ℝ) : Prop := K = (1 / 2) * b * (3 * b)

-- Define the side length of the rhombus in terms of K
noncomputable def side_length (K : ℝ) : ℝ := Real.sqrt (5 * K / 3)

-- The main theorem statement
theorem rhombus_side_length (K : ℝ) (b : ℝ) (h1 : diagonal_ratio (3 * b) b) (h2 : area_rhombus b K) : 
  side_length K = Real.sqrt (5 * K / 3) := 
sorry

end rhombus_side_length_l37_37165


namespace vector_calculation_l37_37381

def vector_a : ℝ × ℝ := (1, -1)
def vector_b : ℝ × ℝ := (-1, 2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1 + v1.2 * v2.2)

theorem vector_calculation :
  (dot_product (vector_add (scalar_mult 2 vector_a) vector_b) vector_a) = 1 :=
by
  sorry

end vector_calculation_l37_37381


namespace minimum_other_sales_met_l37_37828

-- Define the sales percentages for pens, pencils, and the condition for other items
def pens_sales : ℝ := 40
def pencils_sales : ℝ := 28
def minimum_other_sales : ℝ := 20

-- Define the total percentage and calculate the required percentage for other items
def total_sales : ℝ := 100
def required_other_sales : ℝ := total_sales - (pens_sales + pencils_sales)

-- The Lean4 statement to prove the percentage of sales for other items
theorem minimum_other_sales_met 
  (pens_sales_eq : pens_sales = 40)
  (pencils_sales_eq : pencils_sales = 28)
  (total_sales_eq : total_sales = 100)
  (minimum_other_sales_eq : minimum_other_sales = 20)
  (required_other_sales_eq : required_other_sales = total_sales - (pens_sales + pencils_sales)) 
  : required_other_sales = 32 ∧ pens_sales + pencils_sales + required_other_sales = 100 := 
by
  sorry

end minimum_other_sales_met_l37_37828


namespace find_alpha_after_five_operations_l37_37162

def returns_to_starting_point_after_operations (α : Real) (n : Nat) : Prop :=
  (n * α) % 360 = 0

theorem find_alpha_after_five_operations (α : Real) 
  (hα1 : 0 < α)
  (hα2 : α < 180)
  (h_return : returns_to_starting_point_after_operations α 5) :
  α = 72 ∨ α = 144 :=
sorry

end find_alpha_after_five_operations_l37_37162


namespace fractional_units_l37_37636

-- Define the mixed number and the smallest composite number
def mixed_number := 3 + 2/7
def smallest_composite := 4

-- To_struct fractional units of 3 2/7
theorem fractional_units (u : ℚ) (n : ℕ) (m : ℕ):
  u = 1/7 ∧ n = 23 ∧ m = 5 :=
by
  have h1 : u = 1 / 7 := sorry
  have h2 : mixed_number = 23 * u := sorry
  have h3 : smallest_composite - mixed_number = 5 * u := sorry
  have h4 : n = 23 := sorry
  have h5 : m = 5 := sorry
  exact ⟨h1, h4, h5⟩

end fractional_units_l37_37636


namespace total_students_in_class_l37_37679

variable (K M Both Total : ℕ)

theorem total_students_in_class
  (hK : K = 38)
  (hM : M = 39)
  (hBoth : Both = 32)
  (hTotal : Total = K + M - Both) :
  Total = 45 := 
by
  rw [hK, hM, hBoth] at hTotal
  exact hTotal

end total_students_in_class_l37_37679


namespace birthday_friends_count_l37_37458

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37458


namespace radical_multiplication_l37_37808

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end radical_multiplication_l37_37808


namespace n_squared_divisible_by_12_l37_37399

theorem n_squared_divisible_by_12 (n : ℕ) : 12 ∣ n^2 * (n^2 - 1) :=
  sorry

end n_squared_divisible_by_12_l37_37399


namespace find_remaining_score_l37_37989

-- Define the problem conditions
def student_scores : List ℕ := [70, 80, 90]
def average_score : ℕ := 70

-- Define the remaining score to prove it equals 40
def remaining_score : ℕ := 40

-- The theorem statement
theorem find_remaining_score (scores : List ℕ) (avg : ℕ) (r : ℕ) 
    (h_scores : scores = [70, 80, 90]) 
    (h_avg : avg = 70) 
    (h_length : scores.length = 3) 
    (h_avg_eq : (scores.sum + r) / (scores.length + 1) = avg) 
    : r = 40 := 
by
  sorry

end find_remaining_score_l37_37989


namespace geometric_arithmetic_sequence_difference_l37_37746

theorem geometric_arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = a 1 * q)
  (ha4 : a 4 = a 1 * q ^ 3)
  (ha5 : a 5 = a 1 * q ^ 4)
  (harith : 2 * (a 4 + 2 * a 5) = 2 * a 2 + (a 4 + 2 * a 5))
  (hS : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 10 - S 4 = 2016 :=
by
  sorry

end geometric_arithmetic_sequence_difference_l37_37746


namespace find_unknown_number_l37_37151

theorem find_unknown_number (x : ℤ) :
  (20 + 40 + 60) / 3 = 5 + (20 + 60 + x) / 3 → x = 25 :=
by
  sorry

end find_unknown_number_l37_37151


namespace no_perfect_squares_in_sequence_l37_37995

theorem no_perfect_squares_in_sequence (x : ℕ → ℤ) (h₀ : x 0 = 1) (h₁ : x 1 = 3)
  (h_rec : ∀ n : ℕ, x (n + 1) = 6 * x n - x (n - 1)) 
  : ∀ n : ℕ, ¬ ∃ k : ℤ, x n = k * k := 
sorry

end no_perfect_squares_in_sequence_l37_37995


namespace number_of_sets_l37_37498

theorem number_of_sets (A : Set ℕ) : ∃ s : Finset (Set ℕ), 
  (∀ x ∈ s, ({1} ⊂ x ∧ x ⊆ {1, 2, 3, 4})) ∧ s.card = 7 :=
sorry

end number_of_sets_l37_37498


namespace johnny_hourly_wage_l37_37614

-- Definitions based on conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Theorem statement
theorem johnny_hourly_wage : total_earnings / hours_worked = 4.75 :=
by
  sorry

end johnny_hourly_wage_l37_37614


namespace q_transformation_l37_37241

theorem q_transformation (w m z : ℝ) (q : ℝ) (h_q : q = 5 * w / (4 * m * z^2)) :
  let w' := 4 * w
  let m' := 2 * m
  let z' := 3 * z
  q = 5 * w / (4 * m * z^2) → (5 * w') / (4 * m' * (z'^2)) = (5 / 18) * q := by
  sorry

end q_transformation_l37_37241


namespace find_k_l37_37129

def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x + y - 2 = 0
def quadrilateral_has_circumscribed_circle (k : ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y → line2 k x y →
  k = -3

theorem find_k (k : ℝ) (x y : ℝ) : 
  (line1 x y) ∧ (line2 k x y) → quadrilateral_has_circumscribed_circle k :=
by 
  sorry

end find_k_l37_37129


namespace kindergarten_solution_l37_37748

def kindergarten_cards (x y z t : ℕ) : Prop :=
  (x + y = 20) ∧ (z + t = 30) ∧ (y + z = 40) → (x + t = 10)

theorem kindergarten_solution : ∃ (x y z t : ℕ), kindergarten_cards x y z t :=
by {
  sorry
}

end kindergarten_solution_l37_37748


namespace employed_males_percentage_l37_37907

theorem employed_males_percentage (p_employed : ℝ) (p_employed_females : ℝ) : 
  (64 / 100) * (1 - 21.875 / 100) * 100 = 49.96 :=
by
  sorry

end employed_males_percentage_l37_37907


namespace mr_yadav_expenses_l37_37104

theorem mr_yadav_expenses (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 0.6 * S > 0) 
  (h3 : (12 * 0.2 * S) = 48456) : 
  0.2 * S = 4038 :=
by
  sorry

end mr_yadav_expenses_l37_37104


namespace find_triples_l37_37565

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def power_of_p (p n : ℕ) : Prop := ∃ (k : ℕ), n = p^k

-- Given the conditions
variable (p x y : ℕ)
variable (h_prime : is_prime p)
variable (h_pos_x : x > 0)
variable (h_pos_y : y > 0)

-- The problem statement
theorem find_triples (h1 : power_of_p p (x^(p-1) + y)) (h2 : power_of_p p (x + y^(p-1))) : 
  (p = 3 ∧ x = 2 ∧ y = 5) ∨
  (p = 3 ∧ x = 5 ∧ y = 2) ∨
  (p = 2 ∧ ∃ (n i : ℕ), n > 0 ∧ i > 0 ∧ x = n ∧ y = 2^i - n ∧ 0 < n ∧ n < 2^i) := 
sorry

end find_triples_l37_37565


namespace min_value_of_exponential_l37_37994

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 1) : 
  2^x + 4^y ≥ 2 * Real.sqrt 2 ∧ 
  (∀ a, (2^x + 4^y = a) → a ≥ 2 * Real.sqrt 2) :=
by
  sorry

end min_value_of_exponential_l37_37994


namespace trig_identity_l37_37352

noncomputable def sin_deg (x : ℝ) := Real.sin (x * Real.pi / 180)
noncomputable def cos_deg (x : ℝ) := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : ℝ) := Real.tan (x * Real.pi / 180)

theorem trig_identity :
  (2 * sin_deg 50 + sin_deg 10 * (1 + Real.sqrt 3 * tan_deg 10) * Real.sqrt 2 * (sin_deg 80)^2) = Real.sqrt 6 :=
by
  sorry

end trig_identity_l37_37352


namespace carla_cream_volume_l37_37184

-- Definitions of the given conditions and problem
def watermelon_puree_volume : ℕ := 500
def servings_count : ℕ := 4
def volume_per_serving : ℕ := 150
def total_smoothies_volume := servings_count * volume_per_serving
def cream_volume := total_smoothies_volume - watermelon_puree_volume

-- Statement of the proposition we want to prove
theorem carla_cream_volume : cream_volume = 100 := by
  sorry

end carla_cream_volume_l37_37184


namespace log_eq_exp_l37_37024

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l37_37024


namespace sum_of_two_even_numbers_is_even_l37_37819

  theorem sum_of_two_even_numbers_is_even (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k) (hb : ∃ m : ℤ, b = 2 * m) : ∃ n : ℤ, a + b = 2 * n := by
    sorry
  
end sum_of_two_even_numbers_is_even_l37_37819


namespace polar_coordinates_full_circle_l37_37957

theorem polar_coordinates_full_circle :
  ∀ (r : ℝ) (θ : ℝ), (r = 3 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) → (r = 3 ∧ ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi ↔ r = 3) :=
by
  intros r θ h
  sorry

end polar_coordinates_full_circle_l37_37957


namespace tangent_normal_lines_l37_37659

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l37_37659


namespace decimal_expansion_of_13_over_625_l37_37373

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l37_37373


namespace peter_age_fraction_l37_37078

theorem peter_age_fraction 
  (harriet_age : ℕ) 
  (mother_age : ℕ) 
  (peter_age_plus_four : ℕ) 
  (harriet_age_plus_four : ℕ) 
  (harriet_age_current : harriet_age = 13)
  (mother_age_current : mother_age = 60)
  (peter_age_condition : peter_age_plus_four = 2 * harriet_age_plus_four)
  (harriet_four_years : harriet_age_plus_four = harriet_age + 4)
  (peter_four_years : ∀ P : ℕ, peter_age_plus_four = P + 4)
: ∃ P : ℕ, P = 30 ∧ P = mother_age / 2 := 
sorry

end peter_age_fraction_l37_37078


namespace friends_at_birthday_l37_37467

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37467


namespace medical_team_compositions_l37_37930

theorem medical_team_compositions : ∃ n, n = 70 ∧ (
  let male_doctors := 5 in
  let female_doctors := 4 in
  let total_doctors := 9 in
  let no_restriction := (total_doctors.choose 3), 
  let all_male := (male_doctors.choose 3), 
  let all_female := (female_doctors.choose 3), 
  no_restriction - all_male - all_female = n
) :=
by
  sorry

end medical_team_compositions_l37_37930


namespace circle_arc_sum_bounds_l37_37514

open Nat

theorem circle_arc_sum_bounds :
  let red_points := 40
  let blue_points := 30
  let green_points := 20
  let total_arcs := 90
  let T := 0 * red_points + 1 * blue_points + 2 * green_points
  let S_min := 6
  let S_max := 140
  (∀ S, (S = 2 * T - A) → (0 ≤ A ∧ A ≤ 134) → (S_min ≤ S ∧ S ≤ S_max))
  → ∃ S_min S_max, S_min = 6 ∧ S_max = 140 :=
by
  intros
  sorry

end circle_arc_sum_bounds_l37_37514


namespace david_reading_time_l37_37009

theorem david_reading_time
  (total_time : ℕ)
  (math_time : ℕ)
  (spelling_time : ℕ)
  (reading_time : ℕ)
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18)
  (h4 : reading_time = total_time - (math_time + spelling_time)) :
  reading_time = 27 := 
by {
  sorry
}

end david_reading_time_l37_37009


namespace paolo_sevilla_birthday_l37_37483

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37483


namespace benny_books_l37_37685

variable (B : ℕ) -- the number of books Benny had initially

theorem benny_books (h : B - 10 + 33 = 47) : B = 24 :=
sorry

end benny_books_l37_37685


namespace find_simple_interest_principal_l37_37502

def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t / 100

theorem find_simple_interest_principal : 
  (simple_interest P 8 3 = 1 / 2 * compound_interest 4000 10 2) → 
  P = 1750 := 
by
  sorry

end find_simple_interest_principal_l37_37502


namespace tan_beta_value_l37_37879

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan (α + β) = -1) : Real.tan β = 3 :=
by
  sorry

end tan_beta_value_l37_37879


namespace exists_natural_pairs_a_exists_natural_pair_b_l37_37019

open Nat

-- Part (a) Statement
theorem exists_natural_pairs_a (x y : ℕ) :
  x^2 - y^2 = 105 → (x, y) = (53, 52) ∨ (x, y) = (19, 16) ∨ (x, y) = (13, 8) ∨ (x, y) = (11, 4) :=
sorry

-- Part (b) Statement
theorem exists_natural_pair_b (x y : ℕ) :
  2*x^2 + 5*x*y - 12*y^2 = 28 → (x, y) = (8, 5) :=
sorry

end exists_natural_pairs_a_exists_natural_pair_b_l37_37019


namespace track_is_600_l37_37849

noncomputable def track_length (x : ℝ) : Prop :=
  ∃ (s_b s_s : ℝ), 
      s_b > 0 ∧ s_s > 0 ∧
      (∀ t, t > 0 → ((s_b * t = 120 ∧ s_s * t = x / 2 - 120) ∨ 
                     (s_s * (t + 180 / s_s) - s_s * t = x / 2 + 60 
                      ∧ s_b * (t + 180 / s_s) - s_b * t = x / 2 - 60)))

theorem track_is_600 : track_length 600 :=
sorry

end track_is_600_l37_37849


namespace smallest_x_plus_y_l37_37204

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37204


namespace more_than_3000_students_l37_37624

-- Define the conditions
def students_know_secret (n : ℕ) : ℕ :=
  3 ^ (n - 1)

-- Define the statement to prove
theorem more_than_3000_students : ∃ n : ℕ, students_know_secret n > 3000 ∧ n = 9 := by
  sorry

end more_than_3000_students_l37_37624


namespace proof_problem_l37_37579

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l37_37579


namespace problem_l37_37047

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1) 
  : f 2019 = -1 := 
sorry

end problem_l37_37047


namespace planar_graph_edge_vertex_inequality_l37_37290

def planar_graph (G : Type _) : Prop := -- Placeholder for planar graph property
  sorry

variables {V E : ℕ}

theorem planar_graph_edge_vertex_inequality (G : Type _) (h : planar_graph G) :
  E ≤ 3 * V - 6 :=
sorry

end planar_graph_edge_vertex_inequality_l37_37290


namespace total_vehicles_correct_l37_37246

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l37_37246


namespace value_of_g_3x_minus_5_l37_37629

variable (R : Type) [Field R]
variable (g : R → R)
variable (x y : R)

-- Given condition: g(x) = -3 for all real numbers x
axiom g_is_constant : ∀ x : R, g x = -3

-- Prove that g(3x - 5) = -3
theorem value_of_g_3x_minus_5 : g (3 * x - 5) = -3 :=
by
  sorry

end value_of_g_3x_minus_5_l37_37629


namespace total_distance_traveled_l37_37336

def distance_from_earth_to_planet_x : ℝ := 0.5
def distance_from_planet_x_to_planet_y : ℝ := 0.1
def distance_from_planet_y_to_earth : ℝ := 0.1

theorem total_distance_traveled : 
  distance_from_earth_to_planet_x + distance_from_planet_x_to_planet_y + distance_from_planet_y_to_earth = 0.7 :=
by
  sorry

end total_distance_traveled_l37_37336


namespace sum_of_digits_in_base_7_l37_37628

theorem sum_of_digits_in_base_7 (A B C : ℕ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (hA7 : A < 7) (hB7 : B < 7) (hC7 : C < 7)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h_eqn : A * 49 + B * 7 + C + (B * 7 + C) = A * 49 + C * 7 + A) : 
  (A + B + C) = 14 := by
  sorry

end sum_of_digits_in_base_7_l37_37628


namespace transaction_mistake_in_cents_l37_37547

theorem transaction_mistake_in_cents
  (x y : ℕ)
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (error_cents : 100 * y + x - (100 * x + y) = 5616) :
  y = x + 56 :=
by {
  sorry
}

end transaction_mistake_in_cents_l37_37547


namespace find_n_l37_37862
-- Import the necessary dependencies.

-- Define the set S.
def S (n : ℕ) := {i | 1 ≤ i ∧ i ≤ 2 * n}

-- Define the property of dividing a set into two subsets with specific properties.
def divides_sum (S S₁ S₂ : Finset ℕ) := (S₁ ∪ S₂ = S ∧ S₁ ∩ S₂ = ∅ ∧ S₁.card = S₂.card ∧ ∑ i in S₁, i ∣ ∑ i in S₂, i)

-- Define the main theorem to be proven.
theorem find_n (n : ℕ) (hpos : 0 < n) : (∃ S₁ S₂ : Finset ℕ, divides_sum (Finset.filter (λ x, x ∈ S n) (Finset.range (2 * n + 1))) S₁ S₂) ↔ n % 6 ≠ 5 := sorry

end find_n_l37_37862


namespace proof_success_probability_l37_37893

noncomputable def success_probability (p : ℝ) : Prop := 
  let xi_pmf := ProbabilityMassFunction.binomial 2 p
  let eta_pmf := ProbabilityMassFunction.binomial 4 p
  xi_pmf.prob ({i | 1 ≤ i}) = 5/9 → eta_pmf.prob ({i | 2 ≤ i}) = 11/27

theorem proof_success_probability :
  ∀ (p : ℝ), success_probability p :=
begin
  intro p,
  sorry -- Proof omitted as per guidelines
end

end proof_success_probability_l37_37893


namespace find_valid_m_l37_37273

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem find_valid_m (m : ℝ) : (∀ x, ∃ y, g m x = y ∧ g m y = x) ↔ (m ∈ Set.Iio (-9 / 4) ∪ Set.Ioi (-9 / 4)) :=
by
  sorry

end find_valid_m_l37_37273


namespace range_of_slope_exists_k_for_collinearity_l37_37900

def line_equation (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x + 3

noncomputable def intersect_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Assume a function that computes the intersection points (x₁, y₁) and (x₂, y₂)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v2 = (c * v1.1, c * v1.2)

theorem range_of_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0) :
  -4/3 < k ∧ k < 0 := 
sorry

theorem exists_k_for_collinearity (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0)
  (h5 : -4/3 < k ∧ k < 0) :
  collinear (2 - x₁ - x₂, -(y₁ + y₂)) (-2, 1) ↔ k = -1/2 :=
sorry


end range_of_slope_exists_k_for_collinearity_l37_37900


namespace five_student_committees_l37_37377

theorem five_student_committees (n k : ℕ) (hn : n = 8) (hk : k = 5) : 
  nat.choose n k = 56 := by
  rw [hn, hk]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end five_student_committees_l37_37377


namespace minimum_value_m_l37_37495

noncomputable def f (x : ℝ) (phi : ℝ) : ℝ :=
  Real.sin (2 * x + phi)

theorem minimum_value_m (phi : ℝ) (m : ℝ) (h1 : |phi| < Real.pi / 2)
  (h2 : ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x (Real.pi / 6) ≤ m) :
  m = -1 / 2 :=
by
  sorry

end minimum_value_m_l37_37495


namespace find_k_l37_37557

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 8
def g (x : ℝ) (k : ℝ) : ℝ := x ^ 2 - k * x + 3

theorem find_k : 
  (f 5 - g 5 k = 12) → k = -53 / 5 :=
by
  intro hyp
  sorry

end find_k_l37_37557


namespace birch_trees_probability_l37_37332

/--
A gardener plants four pine trees, five oak trees, and six birch trees in a row. He plants them in random order, each arrangement being equally likely.
Prove that no two birch trees are next to one another is \(\frac{2}{45}\).
--/
theorem birch_trees_probability: (∃ (m n : ℕ), (m = 2) ∧ (n = 45) ∧ (no_two_birch_trees_adjacent_probability = m / n)) := 
sorry

end birch_trees_probability_l37_37332


namespace batsman_average_after_11th_inning_l37_37530

theorem batsman_average_after_11th_inning (A : ℝ) 
  (h1 : A + 5 = (10 * A + 85) / 11) : A + 5 = 35 :=
by
  sorry

end batsman_average_after_11th_inning_l37_37530


namespace area_of_region_l37_37079

theorem area_of_region :
  ∀ (x y : ℝ), (|2 * x - 2| + |3 * y - 3| ≤ 30) → (area_of_figure = 300) :=
sorry

end area_of_region_l37_37079


namespace rook_reaches_upper_right_in_expected_70_minutes_l37_37694

section RookMoves

noncomputable def E : ℝ := 70

-- Definition of expected number of minutes considering the row and column moves.
-- This is a direct translation from the problem's correct answer.
def rook_expected_minutes_to_upper_right (E_0 E_1 : ℝ) : Prop :=
  E_0 = (70 : ℝ) ∧ E_1 = (70 : ℝ)

theorem rook_reaches_upper_right_in_expected_70_minutes : E = 70 := sorry

end RookMoves

end rook_reaches_upper_right_in_expected_70_minutes_l37_37694


namespace max_positive_integers_l37_37106

theorem max_positive_integers (f : Fin 2018 → ℤ) (h : ∀ i : Fin 2018, f i > f (i - 1) + f (i - 2)) : 
  ∃ n: ℕ, n = 2016 ∧ (∀ i : ℕ, i < 2018 → f i > 0) ∧ (∀ i : ℕ, i < 2 → f i < 0) := 
sorry

end max_positive_integers_l37_37106


namespace longest_diagonal_of_rhombus_l37_37541

theorem longest_diagonal_of_rhombus (d1 d2 : ℝ) (area : ℝ) (ratio : ℝ) (h1 : area = 150) (h2 : d1 / d2 = 4 / 3) :
  max d1 d2 = 20 :=
by 
  let x := sqrt (area * 2 / (d1 * d2))
  have d1_expr : d1 = 4 * x := sorry
  have d2_expr : d2 = 3 * x := sorry
  have x_val : x = 5 := sorry
  have length_longest_diag : max d1 d2 = max (4 * 5) (3 * 5) := sorry
  exact length_longest_diag

end longest_diagonal_of_rhombus_l37_37541


namespace min_value_a_sq_plus_b_sq_l37_37222

theorem min_value_a_sq_plus_b_sq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (a - 1)^3 + (b - 1)^3 ≥ 3 * (2 - a - b)) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x > 0 → y > 0 → (x - 1)^3 + (y - 1)^3 ≥ 3 * (2 - x - y) → x^2 + y^2 ≥ m) :=
by
  sorry

end min_value_a_sq_plus_b_sq_l37_37222


namespace smallest_x_plus_y_l37_37219

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37219


namespace isosceles_triangle_perimeter_l37_37250

-- Define the lengths of the sides of the isosceles triangle
def side1 : ℕ := 12
def side2 : ℕ := 12
def base : ℕ := 17

-- Define the perimeter as the sum of all three sides
def perimeter : ℕ := side1 + side2 + base

-- State the theorem that needs to be proved
theorem isosceles_triangle_perimeter : perimeter = 41 := by
  -- Insert the proof here
  sorry

end isosceles_triangle_perimeter_l37_37250


namespace sufficient_but_not_necessary_condition_l37_37757

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l37_37757


namespace total_legs_among_animals_l37_37860

def legs (chickens sheep grasshoppers spiders : Nat) (legs_chicken legs_sheep legs_grasshopper legs_spider : Nat) : Nat :=
  (chickens * legs_chicken) + (sheep * legs_sheep) + (grasshoppers * legs_grasshopper) + (spiders * legs_spider)

theorem total_legs_among_animals :
  let chickens := 7
  let sheep := 5
  let grasshoppers := 10
  let spiders := 3
  let legs_chicken := 2
  let legs_sheep := 4
  let legs_grasshopper := 6
  let legs_spider := 8
  legs chickens sheep grasshoppers spiders legs_chicken legs_sheep legs_grasshopper legs_spider = 118 :=
by
  sorry

end total_legs_among_animals_l37_37860


namespace P_desert_but_not_Coffee_is_0_15_l37_37300

-- Define the relevant probabilities as constants
def P_desert_and_coffee := 0.60
def P_not_desert := 0.2500000000000001
def P_desert := 1 - P_not_desert
def P_desert_but_not_coffee := P_desert - P_desert_and_coffee

-- The theorem to prove that the probability of ordering dessert but not coffee is 0.15
theorem P_desert_but_not_Coffee_is_0_15 :
  P_desert_but_not_coffee = 0.15 :=
by 
  -- calculation steps can be filled in here eventually
  sorry

end P_desert_but_not_Coffee_is_0_15_l37_37300


namespace Jill_ball_difference_l37_37262

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l37_37262


namespace will_has_123_pieces_of_candy_l37_37955

def initial_candy_pieces (chocolate_boxes mint_boxes caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  chocolate_boxes * pieces_per_chocolate_box + mint_boxes * pieces_per_mint_box + caramel_boxes * pieces_per_caramel_box

def given_away_candy_pieces (given_chocolate_boxes given_mint_boxes given_caramel_boxes : ℕ)
  (pieces_per_chocolate_box pieces_per_mint_box pieces_per_caramel_box : ℕ) : ℕ :=
  given_chocolate_boxes * pieces_per_chocolate_box + given_mint_boxes * pieces_per_mint_box + given_caramel_boxes * pieces_per_caramel_box

def remaining_candy : ℕ :=
  let initial := initial_candy_pieces 7 5 4 12 15 10
  let given_away := given_away_candy_pieces 3 2 1 12 15 10
  initial - given_away

theorem will_has_123_pieces_of_candy : remaining_candy = 123 :=
by
  -- Proof goes here
  sorry

end will_has_123_pieces_of_candy_l37_37955


namespace distribute_gifts_l37_37132

theorem distribute_gifts :
  ∃ n, (4.choose 1 + 4.choose 2 + 4.choose 3 = n) ∧ n = 14 :=
by
  sorry

end distribute_gifts_l37_37132


namespace brother_more_lambs_than_merry_l37_37919

theorem brother_more_lambs_than_merry
  (merry_lambs : ℕ) (total_lambs : ℕ) (more_than_merry : ℕ)
  (h1 : merry_lambs = 10) 
  (h2 : total_lambs = 23)
  (h3 : more_than_merry + merry_lambs + merry_lambs = total_lambs) :
  more_than_merry = 3 :=
by
  sorry

end brother_more_lambs_than_merry_l37_37919


namespace find_x_l37_37026

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l37_37026


namespace remainder_when_divided_by_6_l37_37242

theorem remainder_when_divided_by_6 (n : ℕ) (h : n % 12 = 8) : n % 6 = 2 :=
by sorry

end remainder_when_divided_by_6_l37_37242


namespace simplify_and_evaluate_sqrt_log_product_property_l37_37933

-- Problem I
theorem simplify_and_evaluate_sqrt (a : ℝ) (h : 0 < a) : 
  Real.sqrt (a^(1/4) * Real.sqrt (a * Real.sqrt a)) = Real.sqrt a := 
by
  sorry

-- Problem II
theorem log_product_property : 
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := 
by
  sorry

end simplify_and_evaluate_sqrt_log_product_property_l37_37933


namespace cos_theta_when_f_maximizes_l37_37953

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem cos_theta_when_f_maximizes (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end cos_theta_when_f_maximizes_l37_37953


namespace calculate_exponentiation_l37_37183

theorem calculate_exponentiation : (64^(0.375) * 64^(0.125) = 8) :=
by sorry

end calculate_exponentiation_l37_37183


namespace seashells_count_l37_37308

theorem seashells_count (total_seashells broken_seashells : ℕ) (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) : total_seashells - broken_seashells = 3 := by
  sorry

end seashells_count_l37_37308


namespace friends_at_birthday_l37_37468

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37468


namespace area_tripled_radius_increase_l37_37631

theorem area_tripled_radius_increase (m r : ℝ) (h : (r + m)^2 = 3 * r^2) :
  r = m * (1 + Real.sqrt 3) / 2 :=
sorry

end area_tripled_radius_increase_l37_37631


namespace find_letters_with_dot_but_no_straight_line_l37_37148

-- Define the problem statement and conditions
def DL : ℕ := 16
def L : ℕ := 30
def Total_letters : ℕ := 50

-- Define the function that calculates the number of letters with a dot but no straight line
def letters_with_dot_but_no_straight_line (DL L Total_letters : ℕ) : ℕ := Total_letters - (L + DL)

-- State the theorem to be proved
theorem find_letters_with_dot_but_no_straight_line : letters_with_dot_but_no_straight_line DL L Total_letters = 4 :=
by
  sorry

end find_letters_with_dot_but_no_straight_line_l37_37148


namespace tiling_ratio_l37_37722

theorem tiling_ratio (n a b : ℕ) (ha : a ≠ 0) (H : b = a * 2^(n/2)) :
  b / a = 2^(n/2) :=
  by
  sorry

end tiling_ratio_l37_37722


namespace part1_part2_l37_37170

-- Step 1: Define the problem for a triangle with specific side length conditions and perimeter
theorem part1 (x : ℝ) (h1 : 2 * x + 2 * (2 * x) = 18) : 
  x = 18 / 5 ∧ 2 * x = 36 / 5 :=
by
  sorry

-- Step 2: Verify if an isosceles triangle with a side length of 4 cm can be formed
theorem part2 (a b c : ℝ) (h2 : a = 4 ∨ b = 4 ∨ c = 4) (h3 : a + b + c = 18) : 
  (a = 4 ∧ b = 7 ∧ c = 7 ∨ b = 4 ∧ a = 7 ∧ c = 7 ∨ c = 4 ∧ a = 7 ∧ b = 7) ∨
  (¬(a = 4 ∧ b + c <= a ∨ b = 4 ∧ a + c <= b ∨ c = 4 ∧ a + b <= c)) :=
by
  sorry

end part1_part2_l37_37170


namespace birthday_friends_count_l37_37473

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37473


namespace fried_chicken_total_l37_37096

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

end fried_chicken_total_l37_37096


namespace function_machine_output_15_l37_37255

-- Defining the function machine operation
def function_machine (input : ℕ) : ℕ :=
  let after_multiplication := input * 3 in
  if after_multiplication > 25 then 
    after_multiplication - 7
  else 
    after_multiplication + 10

-- Statement of the problem to be proved
theorem function_machine_output_15 : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_15_l37_37255


namespace percentage_increase_direct_proportionality_l37_37296

variable (x y k q : ℝ)
variable (h1 : x = k * y)
variable (h2 : x' = x * (1 + q / 100))

theorem percentage_increase_direct_proportionality :
  ∃ q_percent : ℝ, y' = y * (1 + q_percent / 100) ∧ q_percent = q := sorry

end percentage_increase_direct_proportionality_l37_37296


namespace arithmetic_sequence_S2008_l37_37252

theorem arithmetic_sequence_S2008 (a1 : ℤ) (S : ℕ → ℤ) (d : ℤ)
  (h1 : a1 = -2008)
  (h2 : ∀ n, S n = n * a1 + n * (n - 1) / 2 * d)
  (h3 : (S 12 / 12) - (S 10 / 10) = 2) :
  S 2008 = -2008 := 
sorry

end arithmetic_sequence_S2008_l37_37252


namespace max_min_f_product_of_roots_f_l37_37049

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log x / Real.log 3 - 3) * (Real.log x / Real.log 3 + 1)

theorem max_min_f
  (x : ℝ) (h : x ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ)) : 
  (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≤ 12)
  ∧ (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≥ 5) :=
sorry

theorem product_of_roots_f
  (m α β : ℝ) (h1 : f α + m = 0) (h2 : f β + m = 0) : 
  (Real.log (α * β) / Real.log 3 = 2) → (α * β = 9) :=
sorry

end max_min_f_product_of_roots_f_l37_37049


namespace value_this_year_l37_37509

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l37_37509


namespace longest_side_of_triangle_l37_37729

theorem longest_side_of_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) 
    (cond : a^2 + b^2 - 6 * a - 10 * b + 34 = 0) 
    (triangle_ineq1 : a + b > c)
    (triangle_ineq2 : a + c > b)
    (triangle_ineq3 : b + c > a)
    (hScalene: a ≠ b ∧ b ≠ c ∧ a ≠ c) : c = 6 ∨ c = 7 := 
by {
  sorry
}

end longest_side_of_triangle_l37_37729


namespace min_days_is_9_l37_37191

theorem min_days_is_9 (n : ℕ) (rain_morning rain_afternoon sunny_morning sunny_afternoon : ℕ)
  (h1 : rain_morning + rain_afternoon = 7)
  (h2 : rain_afternoon ≤ sunny_morning)
  (h3 : sunny_afternoon = 5)
  (h4 : sunny_morning = 6) :
  n ≥ 9 :=
sorry

end min_days_is_9_l37_37191


namespace speed_in_still_water_l37_37161

/--
A man can row upstream at 55 kmph and downstream at 65 kmph.
Prove that his speed in still water is 60 kmph.
-/
theorem speed_in_still_water (upstream_speed downstream_speed : ℕ) (h_upstream : upstream_speed = 55) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 60 := by
  sorry

end speed_in_still_water_l37_37161


namespace solution_set_of_inequality_l37_37889

-- Define conditions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- Lean statement of the proof problem
theorem solution_set_of_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_mono_inc : is_monotonically_increasing_on f {x | x ≤ 0}) :
  { x : ℝ | f (3 - 2 * x) > f (1) } = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l37_37889


namespace find_b_l37_37696

def h (x : ℝ) : ℝ := 5 * x + 7

theorem find_b (b : ℝ) : h b = 0 ↔ b = -7 / 5 := by
  sorry

end find_b_l37_37696


namespace first_term_is_sqrt9_l37_37646

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end first_term_is_sqrt9_l37_37646


namespace smallest_x_plus_y_l37_37205

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37205


namespace parabola_points_relationship_l37_37038

theorem parabola_points_relationship :
  let y_1 := (-2)^2 + 2 * (-2) - 9
  let y_2 := 1^2 + 2 * 1 - 9
  let y_3 := 3^2 + 2 * 3 - 9
  y_3 > y_2 ∧ y_2 > y_1 :=
by
  sorry

end parabola_points_relationship_l37_37038


namespace books_in_either_but_not_both_l37_37677

theorem books_in_either_but_not_both (shared_books alice_books bob_unique_books : ℕ) 
    (h1 : shared_books = 12) 
    (h2 : alice_books = 26)
    (h3 : bob_unique_books = 8) : 
    (alice_books - shared_books) + bob_unique_books = 22 :=
by
  sorry

end books_in_either_but_not_both_l37_37677


namespace exponent_2_prime_factorization_30_exponent_5_prime_factorization_30_l37_37759

open Nat

theorem exponent_2_prime_factorization_30! :
  nat.factorial_prime_pow 30 2 = 26 := by
  sorry

theorem exponent_5_prime_factorization_30! :
  nat.factorial_prime_pow 30 5 = 7 := by
  sorry

end exponent_2_prime_factorization_30_exponent_5_prime_factorization_30_l37_37759


namespace conic_section_is_hyperbola_l37_37857

-- Definitions for the conditions in the problem
def conic_section_equation (x y : ℝ) := (x - 4) ^ 2 = 5 * (y + 2) ^ 2 - 45

-- The theorem that we need to prove
theorem conic_section_is_hyperbola : ∀ x y : ℝ, (conic_section_equation x y) → "H" = "H" :=
by
  intro x y h
  sorry

end conic_section_is_hyperbola_l37_37857


namespace find_sum_of_integers_l37_37374

theorem find_sum_of_integers (x y : ℕ) (h_diff : x - y = 8) (h_prod : x * y = 180) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : x + y = 28 :=
by
  sorry

end find_sum_of_integers_l37_37374


namespace range_of_m_l37_37998

-- Definitions of propositions and their negations
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0
def not_p (x : ℝ) : Prop := x < -2 ∨ x > 10
def not_q (x m : ℝ) : Prop := x < (1 - m) ∨ x > (1 + m) ∧ m > 0

-- Statement that \neg p is a necessary but not sufficient condition for \neg q
def necessary_but_not_sufficient (x m : ℝ) : Prop := 
  (∀ x, not_q x m → not_p x) ∧ ¬(∀ x, not_p x → not_q x m)

-- The main theorem to prove
theorem range_of_m (m : ℝ) : (∀ x, necessary_but_not_sufficient x m) ↔ 9 ≤ m :=
by
  sorry

end range_of_m_l37_37998


namespace sum_remainders_mod_13_l37_37375

theorem sum_remainders_mod_13 :
  ∀ (a b c d e : ℕ),
  a % 13 = 3 →
  b % 13 = 5 →
  c % 13 = 7 →
  d % 13 = 9 →
  e % 13 = 11 →
  (a + b + c + d + e) % 13 = 9 :=
by
  intros a b c d e ha hb hc hd he
  sorry

end sum_remainders_mod_13_l37_37375


namespace not_black_cows_count_l37_37781

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l37_37781


namespace product_as_difference_of_squares_l37_37292

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ( (a + b) / 2 )^2 - ( (a - b) / 2 )^2 :=
by
  sorry

end product_as_difference_of_squares_l37_37292


namespace problem_statement_l37_37769

def g (x : ℝ) : ℝ :=
  x^2 - 5 * x

theorem problem_statement (x : ℝ) :
  (g (g x) = g x) ↔ (x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1) :=
by
  sorry

end problem_statement_l37_37769


namespace not_black_cows_count_l37_37780

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l37_37780


namespace radius_of_circle_proof_l37_37157

noncomputable def radius_of_circle (x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : ℝ :=
  r

theorem radius_of_circle_proof (r x y : ℝ) (h1 : x = Real.pi * r ^ 2) (h2 : y = 2 * Real.pi * r) (h3 : x + y = 100 * Real.pi) : r = 10 :=
by
  sorry

end radius_of_circle_proof_l37_37157


namespace min_re_z4_re_z4_l37_37239

theorem min_re_z4_re_z4 (z : ℂ) (h : z.re ≠ 0) : 
  ∃ t : ℝ, (t = (z.im / z.re)) ∧ ((1 - 6 * (t^2) + (t^4)) = -8) := sorry

end min_re_z4_re_z4_l37_37239


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37810

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37810


namespace total_trucks_l37_37441

-- Define the number of trucks Namjoon has
def trucks_namjoon : ℕ := 3

-- Define the number of trucks Taehyung has
def trucks_taehyung : ℕ := 2

-- Prove that together, Namjoon and Taehyung have 5 trucks
theorem total_trucks : trucks_namjoon + trucks_taehyung = 5 := by 
  sorry

end total_trucks_l37_37441


namespace set_D_is_empty_l37_37340

theorem set_D_is_empty :
  {x : ℝ | x^2 + 2 = 0} = ∅ :=
by {
  sorry
}

end set_D_is_empty_l37_37340


namespace right_pyramid_volume_l37_37334

noncomputable def volume_of_right_pyramid (base_area lateral_face_area total_surface_area : ℝ) : ℝ := 
  let height := (10 : ℝ) / 3
  (1 / 3) * base_area * height

theorem right_pyramid_volume (total_surface_area base_area lateral_face_area : ℝ)
  (h0 : total_surface_area = 300)
  (h1 : base_area + 3 * lateral_face_area = total_surface_area)
  (h2 : lateral_face_area = base_area / 3) 
  : volume_of_right_pyramid base_area lateral_face_area total_surface_area = 500 / 3 := 
by
  sorry

end right_pyramid_volume_l37_37334


namespace ratio_of_girls_to_boys_l37_37243

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) 
  (h1 : total_students = 26) 
  (h2 : girls = boys + 6) 
  (h3 : girls + boys = total_students) : 
  (girls : ℚ) / boys = 8 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l37_37243


namespace jungsoo_number_is_correct_l37_37613

def J := (1 * 4) + (0.1 * 2) + (0.001 * 7)
def Y := 100 * J 
def S := Y + 0.05

theorem jungsoo_number_is_correct : S = 420.75 := by
  sorry

end jungsoo_number_is_correct_l37_37613


namespace square_difference_identity_l37_37704

theorem square_difference_identity (a b : ℕ) : (a - b)^2 = a^2 - 2 * a * b + b^2 :=
  by sorry

lemma evaluate_expression : (101 - 2)^2 = 9801 :=
  by
    have h := square_difference_identity 101 2
    exact h

end square_difference_identity_l37_37704


namespace complex_product_conjugate_l37_37686

theorem complex_product_conjugate : (1 + Complex.I) * (1 - Complex.I) = 2 := 
by 
  -- Lean proof goes here
  sorry

end complex_product_conjugate_l37_37686


namespace simplify_and_evaluate_l37_37295

theorem simplify_and_evaluate (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a - 2) / (a^2 + 2 * a) - (a - 1) / (a^2 + 4 * a + 4)) / ((a - 4) / (a + 2)) = 1 / 3 :=
by sorry

end simplify_and_evaluate_l37_37295


namespace no_real_roots_poly_l37_37110

theorem no_real_roots_poly (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := 
  sorry

end no_real_roots_poly_l37_37110


namespace units_digit_powers_difference_l37_37226

theorem units_digit_powers_difference (p : ℕ) 
  (h1: p > 0) 
  (h2: p % 2 = 0) 
  (h3: (p % 10 + 2) % 10 = 8) : 
  ((p ^ 3) % 10 - (p ^ 2) % 10) % 10 = 0 :=
by
  sorry

end units_digit_powers_difference_l37_37226


namespace solve_for_y_l37_37189

theorem solve_for_y (y : ℚ) : 
  y + 1 / 3 = 3 / 8 - 1 / 4 → y = -5 / 24 := 
by
  sorry

end solve_for_y_l37_37189


namespace decagon_perimeter_l37_37850

-- Define the number of sides in a decagon
def num_sides : ℕ := 10

-- Define the length of each side in the decagon
def side_length : ℕ := 3

-- Define the perimeter of a decagon given the number of sides and the side length
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s

-- State the theorem we want to prove: the perimeter of our given regular decagon
theorem decagon_perimeter : perimeter num_sides side_length = 30 := 
by sorry

end decagon_perimeter_l37_37850


namespace alice_speed_proof_l37_37339

-- Problem definitions
def distance : ℕ := 1000
def abel_speed : ℕ := 50
def abel_arrival_time := distance / abel_speed
def alice_delay : ℕ := 1  -- Alice starts 1 hour later
def earlier_arrival_abel : ℕ := 6  -- Abel arrives 6 hours earlier than Alice

noncomputable def alice_speed : ℕ := (distance / (abel_arrival_time + earlier_arrival_abel))

theorem alice_speed_proof : alice_speed = 200 / 3 := by
  sorry -- proof not required as per instructions

end alice_speed_proof_l37_37339


namespace perfect_square_condition_l37_37086

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l37_37086


namespace polynomial_int_values_l37_37519

variable {R : Type*} [CommRing R] [IsDomain R]

theorem polynomial_int_values (f : Polynomial R) (n : ℕ)
  (h : ∀ k : ℕ, k < n + 1 → (f.eval k).isInt) :
  ∀ x : ℤ, (f.eval x).isInt :=
sorry

end polynomial_int_values_l37_37519


namespace number_of_n_l37_37710

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end number_of_n_l37_37710


namespace find_fraction_l37_37963

noncomputable def fraction_of_third (F N : ℝ) : Prop := F * (1 / 3 * N) = 30

noncomputable def fraction_of_number (G N : ℝ) : Prop := G * N = 75

noncomputable def product_is_90 (F N : ℝ) : Prop := F * N = 90

theorem find_fraction (F G N : ℝ) (h1 : fraction_of_third F N) (h2 : fraction_of_number G N) (h3 : product_is_90 F N) :
  G = 5 / 6 :=
sorry

end find_fraction_l37_37963


namespace percent_value_in_quarters_l37_37823

def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def num_nickels : ℕ := 80
def num_quarters : ℕ := 40

def value_in_nickels : ℕ := num_nickels * nickel_value
def value_in_quarters : ℕ := num_quarters * quarter_value
def total_value : ℕ := value_in_nickels + value_in_quarters

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value = 5 / 7 :=
by
  sorry

end percent_value_in_quarters_l37_37823


namespace problem_statement_l37_37297

variables {R : Type*} [LinearOrderedField R]

theorem problem_statement (a b c : R) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : (b - a) ^ 2 - 4 * (b - c) * (c - a) = 0) : (b - c) / (c - a) = -1 :=
sorry

end problem_statement_l37_37297


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37811

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37811


namespace triangle_area_inequality_l37_37100

variables {a b c S x y z T : ℝ}

-- Definitions based on the given conditions
def side_lengths_of_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def area_of_triangle (a b c S : ℝ) : Prop :=
  16 * S * S = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)

def new_side_lengths (a b c : ℝ) (x y z : ℝ) : Prop :=
  x = a + b / 2 ∧ y = b + c / 2 ∧ z = c + a / 2

def area_condition (S T : ℝ) : Prop :=
  T ≥ 9 / 4 * S

-- Main theorem statement
theorem triangle_area_inequality
  (h_triangle: side_lengths_of_triangle a b c)
  (h_area: area_of_triangle a b c S)
  (h_new_sides: new_side_lengths a b c x y z) :
  ∃ T : ℝ, side_lengths_of_triangle x y z ∧ area_condition S T :=
sorry

end triangle_area_inequality_l37_37100


namespace other_divisor_l37_37866

theorem other_divisor (x : ℕ) (h₁ : 261 % 7 = 2) (h₂ : 261 % x = 2) : x = 259 :=
sorry

end other_divisor_l37_37866


namespace williams_probability_l37_37321

noncomputable def prob_correct (n k : ℕ) : ℝ := 
  (Nat.choose n k) * (1/5)^k * (4/5)^(n-k)

noncomputable def prob_at_least_two_correct (n : ℕ) : ℝ := 
  1 - prob_correct n 0 - prob_correct n 1 

theorem williams_probability :
  prob_at_least_two_correct 6 = 5385 / 15625 := by
  sorry

end williams_probability_l37_37321


namespace smallest_number_is_16_l37_37307

theorem smallest_number_is_16 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c) / 3 = 24 ∧ 
  (b = 25) ∧ (c = b + 6) ∧ min a (min b c) = 16 :=
by
  sorry

end smallest_number_is_16_l37_37307


namespace problem_part1_problem_part2_l37_37881

open Real

variables {a b : EuclideanSpace ℝ (Fin 3)}

def norm_sq (v : EuclideanSpace ℝ (Fin 3)) := ∥v∥^2

noncomputable def angle_between (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.acos (u ⬝ v / (∥u∥ * ∥v∥))

-- Given conditions
def cond_a := ∥a∥ = 4
def cond_b := ∥b∥ = 2
def cond_angle := angle_between a b = π * 2 / 3

-- Proving the two results
theorem problem_part1 (h₁ : cond_a) (h₂ : cond_b) (h₃ : cond_angle) :
  ((a - 2 • b) ⬝ (a + b)) = 12 :=
sorry

theorem problem_part2 (h₁ : cond_a) (h₂ : cond_b) (h₃ : cond_angle) :
  angle_between a (a + b) = π / 6 :=
sorry

end problem_part1_problem_part2_l37_37881


namespace betty_cupcakes_per_hour_l37_37005

theorem betty_cupcakes_per_hour (B : ℕ) (Dora_rate : ℕ) (betty_break_hours : ℕ) (total_hours : ℕ) (cupcake_diff : ℕ) :
  Dora_rate = 8 →
  betty_break_hours = 2 →
  total_hours = 5 →
  cupcake_diff = 10 →
  (total_hours - betty_break_hours) * B = Dora_rate * total_hours - cupcake_diff →
  B = 10 :=
by
  intros hDora_rate hbreak_hours htotal_hours hcupcake_diff hcupcake_eq
  sorry

end betty_cupcakes_per_hour_l37_37005


namespace intersection_l37_37396

noncomputable def M : Set ℝ := { x : ℝ | Real.sqrt (x + 1) ≥ 0 }
noncomputable def N : Set ℝ := { x : ℝ | x^2 + x - 2 < 0 }

theorem intersection (x : ℝ) : x ∈ (M ∩ N) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end intersection_l37_37396


namespace painter_time_remaining_l37_37670

theorem painter_time_remaining (total_rooms : ℕ) (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 12) (h2 : time_per_room = 7) (h3 : rooms_painted = 5) 
  (h4 : remaining_hours = (total_rooms - rooms_painted) * time_per_room) : 
  remaining_hours = 49 :=
by
  sorry

end painter_time_remaining_l37_37670


namespace monotonic_increasing_f_l37_37582

theorem monotonic_increasing_f (f g : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) 
  (hg : ∀ x, g (-x) = g x) (hfg : ∀ x, f x + g x = 3^x) :
  ∀ a b : ℝ, a > b → f a > f b :=
sorry

end monotonic_increasing_f_l37_37582


namespace janice_walk_dog_more_than_homework_l37_37911

theorem janice_walk_dog_more_than_homework 
  (H C T: Nat) 
  (W: Nat) 
  (total_time remaining_time spent_time: Nat) 
  (hw_time room_time trash_time extra_time: Nat)
  (H_eq : H = 30)
  (C_eq : C = H / 2)
  (T_eq : T = H / 6)
  (remaining_time_eq : remaining_time = 35)
  (total_time_eq : total_time = 120)
  (spent_time_eq : spent_time = total_time - remaining_time)
  (task_time_sum_eq : task_time_sum = H + C + T)
  (W_eq : W = spent_time - task_time_sum)
  : W - H = 5 := 
sorry

end janice_walk_dog_more_than_homework_l37_37911


namespace total_number_of_notes_l37_37535

theorem total_number_of_notes 
  (total_money : ℕ)
  (fifty_rupees_notes : ℕ)
  (five_hundred_rupees_notes : ℕ)
  (total_money_eq : total_money = 10350)
  (fifty_rupees_notes_eq : fifty_rupees_notes = 117)
  (money_eq : 50 * fifty_rupees_notes + 500 * five_hundred_rupees_notes = total_money) :
  fifty_rupees_notes + five_hundred_rupees_notes = 126 :=
by sorry

end total_number_of_notes_l37_37535


namespace cos_squared_alpha_plus_pi_over_4_l37_37039

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_l37_37039


namespace team_total_points_l37_37621

theorem team_total_points (three_points_goals: ℕ) (two_points_goals: ℕ) (half_of_total: ℕ) 
  (h1 : three_points_goals = 5) 
  (h2 : two_points_goals = 10) 
  (h3 : half_of_total = (3 * three_points_goals + 2 * two_points_goals) / 2) 
  : 2 * half_of_total = 70 := 
by 
  -- proof to be filled
  sorry

end team_total_points_l37_37621


namespace greatest_number_of_consecutive_integers_sum_36_l37_37949

theorem greatest_number_of_consecutive_integers_sum_36 :
  ∃ (N : ℕ), 
    (∃ a : ℤ, N * a + ((N - 1) * N) / 2 = 36) ∧ 
    (∀ N' : ℕ, (∃ a' : ℤ, N' * a' + ((N' - 1) * N') / 2 = 36) → N' ≤ 72) := by
  sorry

end greatest_number_of_consecutive_integers_sum_36_l37_37949


namespace probability_john_reads_on_both_days_l37_37500

-- Defining the basic events
noncomputable def reads_book_on_monday := 0.8
noncomputable def plays_soccer_on_tuesday := 0.5

-- Defining conditional event probabilities
noncomputable def reads_book_on_tuesday_given_conditions := reads_book_on_monday * plays_soccer_on_tuesday

-- The main theorem to be proved.
theorem probability_john_reads_on_both_days :
  reads_book_on_tuesday_given_conditions = 0.32 := by
  sorry

end probability_john_reads_on_both_days_l37_37500


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37814

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37814


namespace minimum_questions_two_l37_37905

structure Person :=
  (is_liar : Bool)

structure Decagon :=
  (people : Fin 10 → Person)

def minimumQuestionsNaive (d : Decagon) : Nat :=
  match d with 
  -- add the logic here later
  | _ => sorry

theorem minimum_questions_two (d : Decagon) : minimumQuestionsNaive d = 2 :=
  sorry

end minimum_questions_two_l37_37905


namespace number_of_friends_l37_37465

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37465


namespace opposite_of_B_is_I_l37_37921

inductive Face
| A | B | C | D | E | F | G | H | I

open Face

def opposite_face (f : Face) : Face :=
  match f with
  | A => G
  | B => I
  | C => H
  | D => F
  | E => E
  | F => F
  | G => A
  | H => C
  | I => B

theorem opposite_of_B_is_I : opposite_face B = I :=
  by
    sorry

end opposite_of_B_is_I_l37_37921


namespace complete_square_to_d_l37_37361

-- Conditions given in the problem
def quadratic_eq (x : ℝ) : Prop := x^2 + 10 * x + 7 = 0

-- Equivalent Lean 4 statement of the problem
theorem complete_square_to_d (x : ℝ) (c d : ℝ) (h : quadratic_eq x) (hc : c = 5) : (x + c)^2 = d → d = 18 :=
by sorry

end complete_square_to_d_l37_37361


namespace num_five_student_committees_l37_37378

theorem num_five_student_committees (n k : ℕ) (h_n : n = 8) (h_k : k = 5) : choose n k = 56 :=
by
  rw [h_n, h_k]
  -- rest of the proof would go here
  sorry

end num_five_student_committees_l37_37378


namespace evaluate_expression_l37_37193

theorem evaluate_expression : 
  (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 :=
by
  sorry

end evaluate_expression_l37_37193


namespace quadratic_expression_transformation_l37_37075

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l37_37075


namespace samuel_remaining_distance_l37_37784

noncomputable def remaining_distance
  (total_distance : ℕ)
  (segment1_speed : ℕ) (segment1_time : ℕ)
  (segment2_speed : ℕ) (segment2_time : ℕ)
  (segment3_speed : ℕ) (segment3_time : ℕ)
  (segment4_speed : ℕ) (segment4_time : ℕ) : ℕ :=
  total_distance -
  (segment1_speed * segment1_time +
   segment2_speed * segment2_time +
   segment3_speed * segment3_time +
   segment4_speed * segment4_time)

theorem samuel_remaining_distance :
  remaining_distance 1200 60 2 70 3 50 4 80 5 = 270 :=
by
  sorry

end samuel_remaining_distance_l37_37784


namespace find_area_of_triangle_l37_37865

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem find_area_of_triangle :
  let a := 10
  let b := 10
  let c := 12
  triangle_area a b c = 48 := 
by 
  sorry

end find_area_of_triangle_l37_37865


namespace problem_statement_l37_37227

theorem problem_statement (a b : ℝ) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) (h_b_gt_1 : b > 1)
  (h1 : a = 1) (h2 : 1/2 * (b - 1)^2 + 1 = b) : a + b = 4 :=
sorry

end problem_statement_l37_37227


namespace friends_at_birthday_l37_37469

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37469


namespace Jill_ball_difference_l37_37261

theorem Jill_ball_difference (r_packs y_packs balls_per_pack : ℕ)
  (h_r_packs : r_packs = 5) 
  (h_y_packs : y_packs = 4) 
  (h_balls_per_pack : balls_per_pack = 18) :
  (r_packs * balls_per_pack) - (y_packs * balls_per_pack) = 18 :=
by
  sorry

end Jill_ball_difference_l37_37261


namespace balazs_missed_number_l37_37348

theorem balazs_missed_number (n k : ℕ) 
  (h1 : n * (n + 1) / 2 = 3000 + k)
  (h2 : 1 ≤ k)
  (h3 : k < n) : k = 3 := by
  sorry

end balazs_missed_number_l37_37348


namespace pair_cannot_appear_l37_37288

theorem pair_cannot_appear :
  ¬ ∃ (sequence_of_pairs : List (ℤ × ℤ)), 
    (1, 2) ∈ sequence_of_pairs ∧ 
    (2022, 2023) ∈ sequence_of_pairs ∧ 
    ∀ (a b : ℤ) (seq : List (ℤ × ℤ)), 
      (a, b) ∈ seq → 
      ((-a, -b) ∈ seq ∨ (-b, a+b) ∈ seq ∨ 
      ∃ (c d : ℤ), ((a+c, b+d) ∈ seq ∧ (c, d) ∈ seq)) := 
sorry

end pair_cannot_appear_l37_37288


namespace cockatiel_weekly_consumption_is_50_l37_37914

def boxes_bought : ℕ := 3
def boxes_existing : ℕ := 5
def grams_per_box : ℕ := 225
def parrot_weekly_consumption : ℕ := 100
def weeks_supply : ℕ := 12

def total_boxes : ℕ := boxes_bought + boxes_existing
def total_birdseed_grams : ℕ := total_boxes * grams_per_box
def parrot_total_consumption : ℕ := parrot_weekly_consumption * weeks_supply
def cockatiel_total_consumption : ℕ := total_birdseed_grams - parrot_total_consumption
def cockatiel_weekly_consumption : ℕ := cockatiel_total_consumption / weeks_supply

theorem cockatiel_weekly_consumption_is_50 :
  cockatiel_weekly_consumption = 50 := by
  -- Proof goes here
  sorry

end cockatiel_weekly_consumption_is_50_l37_37914


namespace volume_second_cube_l37_37511

open Real

-- Define the ratio of the edges of the cubes
def edge_ratio (a b : ℝ) := a / b = 3 / 1

-- Define the volume of the first cube
def volume_first_cube (a : ℝ) := a^3 = 27

-- Define the edge of the second cube based on the edge of the first cube
def edge_second_cube (a b : ℝ) := a / 3 = b

-- Statement of the problem in Lean 4
theorem volume_second_cube 
  (a b : ℝ) 
  (h_edge_ratio : edge_ratio a b) 
  (h_volume_first : volume_first_cube a) 
  (h_edge_second : edge_second_cube a b) : 
  b^3 = 1 := 
sorry

end volume_second_cube_l37_37511


namespace problem_statement_l37_37777

namespace MathProof

def p : Prop := (2 + 4 = 7)
def q : Prop := ∀ x : ℝ, x = 1 → x^2 ≠ 1

theorem problem_statement : ¬ (p ∧ q) ∧ (p ∨ q) :=
by
  -- To be filled in
  sorry

end MathProof

end problem_statement_l37_37777


namespace binary_digit_one_l37_37719
-- We import the necessary libraries

-- Define the problem and prove the statement as follows
def fractional_part_in_binary (x : ℝ) : ℕ → ℕ := sorry

def sqrt_fractional_binary (k : ℕ) (i : ℕ) : ℕ :=
  fractional_part_in_binary (Real.sqrt ((k : ℝ) * (k + 1))) i

theorem binary_digit_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ i, n + 1 ≤ i ∧ i ≤ 2 * n + 1 ∧ sqrt_fractional_binary k i = 1 :=
sorry

end binary_digit_one_l37_37719


namespace stratified_sampling_sophomores_l37_37680

theorem stratified_sampling_sophomores
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (total_selected : ℕ)
  (H_freshmen : freshmen = 550) (H_sophomores : sophomores = 700) (H_juniors : juniors = 750) (H_total_selected : total_selected = 100) :
  sophomores * total_selected / (freshmen + sophomores + juniors) = 35 :=
by
  sorry

end stratified_sampling_sophomores_l37_37680


namespace cone_rotation_ratio_l37_37166

theorem cone_rotation_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rotation_eq : (20 : ℝ) * (2 * Real.pi * r) = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  let p := 1
  let q := 399
  1 + 399 = 400 := by
{
  sorry
}

end cone_rotation_ratio_l37_37166


namespace smallest_possible_sum_l37_37201

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37201


namespace binom_not_divisible_l37_37706

theorem binom_not_divisible (k : ℤ) : k ≠ 1 → ∃ᶠ n in Filter.atTop, (n + k) ∣ (Nat.choose (2 * n) n) := 
sorry

end binom_not_divisible_l37_37706


namespace average_words_written_l37_37845

def total_words : ℕ := 50000
def total_hours : ℕ := 100
def average_words_per_hour : ℕ := total_words / total_hours

theorem average_words_written :
  average_words_per_hour = 500 := 
by
  sorry

end average_words_written_l37_37845


namespace functional_equation_solution_l37_37981

theorem functional_equation_solution (f : ℚ → ℚ) :
  (∀ x y : ℚ, f (x + f y) = f x * f y) →
  (∀ x : ℚ, f x = 0 ∨ f x = 1) :=
by
  sorry

end functional_equation_solution_l37_37981


namespace jack_buttons_total_l37_37423

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l37_37423


namespace remainder_7_pow_150_mod_4_l37_37310

theorem remainder_7_pow_150_mod_4 : (7 ^ 150) % 4 = 1 :=
by
  sorry

end remainder_7_pow_150_mod_4_l37_37310


namespace ben_remaining_money_l37_37683

/-- Ben's remaining money after business operations /-- 
theorem ben_remaining_money : 
  let initial_money := 2000
  let cheque := 600
  let debtor_payment := 800
  let maintenance_cost := 1200
  initial_money - cheque + debtor_payment - maintenance_cost = 1000 := 
by
  -- Initial money
  let initial_money := 2000
  -- Cheque amount
  let cheque := 600
  -- Debtor payment amount
  let debtor_payment := 800
  -- Maintenance cost
  let maintenance_cost := 1200
  -- Calculation
  have h₁ : initial_money - cheque = 2000 - 600 := by rfl
  let money_after_cheque := 2000 - 600
  have h₂ : money_after_cheque + debtor_payment = 1400 + 800 := by rfl
  let money_after_debtor := 1400 + 800
  have h₃ : money_after_debtor - maintenance_cost = 2200 - 1200 := by rfl
  let remaining_money := 2200 - 1200
  -- Assertion
  show remaining_money = 1000 from sorry

end ben_remaining_money_l37_37683


namespace noemi_starting_money_l37_37922

-- Define the losses and remaining money as constants
constant lost_on_roulette : ℕ := 400
constant lost_on_blackjack : ℕ := 500
constant remaining_money : ℕ := 800

-- Define the initial amount of money as a conjecture to be proven
def initial_money := lost_on_roulette + lost_on_blackjack + remaining_money

-- State the theorem
theorem noemi_starting_money : initial_money = 1700 :=
by
  -- This is where the actual proof would go
  sorry

end noemi_starting_money_l37_37922


namespace cost_of_each_barbell_l37_37432

variables (barbells : ℕ) (money_given money_change : ℝ) (total_cost_per_barbell : ℝ)

-- Given conditions
def conditions := barbells = 3 ∧ money_given = 850 ∧ money_change = 40

-- Theorem statement: Proving the cost of each barbell is $270
theorem cost_of_each_barbell (h : conditions) : total_cost_per_barbell = 270 :=
by
  -- We are using sorry to indicate we are skipping the proof
  sorry

#eval sorry -- Placeholder to verify if the code is syntactically correct

end cost_of_each_barbell_l37_37432


namespace degree_sum_interior_angles_of_star_l37_37342

-- Definitions based on conditions provided.
def extended_polygon_star (n : Nat) (h : n ≥ 6) : Nat := 
  180 * (n - 2)

-- Theorem to prove the degree-sum of the interior angles.
theorem degree_sum_interior_angles_of_star (n : Nat) (h : n ≥ 6) : 
  extended_polygon_star n h = 180 * (n - 2) :=
by
  sorry

end degree_sum_interior_angles_of_star_l37_37342


namespace friends_at_birthday_l37_37471

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37471


namespace find_theta_in_interval_l37_37018

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval_l37_37018


namespace area_of_gray_region_l37_37554

theorem area_of_gray_region (r R : ℝ) (hr : r = 2) (hR : R = 3 * r) : 
  π * R ^ 2 - π * r ^ 2 = 32 * π :=
by
  have hr : r = 2 := hr
  have hR : R = 3 * r := hR
  sorry

end area_of_gray_region_l37_37554


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37816

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37816


namespace solve_for_x_l37_37701

theorem solve_for_x : ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 3.391 := 
by 
  sorry

end solve_for_x_l37_37701


namespace period_and_monotonic_interval_range_of_f_l37_37051

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * Real.cos (2 * x) + Real.sin (x + Real.pi / 4) ^ 2

theorem period_and_monotonic_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∃ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) →
    MonotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2))) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)) :
  f x ∈ Set.Icc 0 (3 / 2) :=
sorry

end period_and_monotonic_interval_range_of_f_l37_37051


namespace edward_work_hours_edward_work_hours_overtime_l37_37654

variable (H : ℕ) -- H represents the number of hours worked
variable (O : ℕ) -- O represents the number of overtime hours

theorem edward_work_hours (H_le_40 : H ≤ 40) (earning_eq_210 : 7 * H = 210) : H = 30 :=
by
  -- Proof to be filled in here
  sorry

theorem edward_work_hours_overtime (H_gt_40 : H > 40) (earning_eq_210 : 7 * 40 + 14 * (H - 40) = 210) : False :=
by
  -- Proof to be filled in here
  sorry

end edward_work_hours_edward_work_hours_overtime_l37_37654


namespace smallest_sum_l37_37215

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37215


namespace factorization_correct_l37_37906

theorem factorization_correct: ∀ (x : ℝ), (x^2 - 9 = (x + 3) * (x - 3)) := 
sorry

end factorization_correct_l37_37906


namespace sum_integer_solutions_l37_37357

theorem sum_integer_solutions (n : ℤ) (h1 : |n^2| < |n - 5|^2) (h2 : |n - 5|^2 < 16) : n = 2 := 
sorry

end sum_integer_solutions_l37_37357


namespace sum_exists_l37_37786

theorem sum_exists 
  (n : ℕ) 
  (hn : n ≥ 5) 
  (k : ℕ) 
  (hk : k > (n + 1) / 2) 
  (a : ℕ → ℕ) 
  (ha1 : ∀ i, 1 ≤ a i) 
  (ha2 : ∀ i, a i < n) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j):
  ∃ i j l, i ≠ j ∧ a i + a j = a l := 
by 
  sorry

end sum_exists_l37_37786


namespace sequence_geometric_and_formula_l37_37884

theorem sequence_geometric_and_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = 2 * a n + 1) :
  (∀ n, a n + 1 = 2 ^ n) ∧ (a n = 2 ^ n - 1) :=
sorry

end sequence_geometric_and_formula_l37_37884


namespace simplify_expression_l37_37788

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 
  3 * (4 - 2 * i) + 2 * i * (3 + i) + 5 * (-1 + i) = 5 + 5 * i :=
by
  sorry

end simplify_expression_l37_37788


namespace longest_diagonal_length_l37_37543

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l37_37543


namespace triangle_inequality_l37_37405

variables {a b c : ℝ}

def sides_of_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality (h : sides_of_triangle a b c) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end triangle_inequality_l37_37405


namespace participants_neither_coffee_nor_tea_l37_37177

-- Define the total number of participants
def total_participants : ℕ := 30

-- Define the number of participants who drank coffee
def coffee_drinkers : ℕ := 15

-- Define the number of participants who drank tea
def tea_drinkers : ℕ := 18

-- Define the number of participants who drank both coffee and tea
def both_drinkers : ℕ := 8

-- The proof statement for the number of participants who drank neither coffee nor tea
theorem participants_neither_coffee_nor_tea :
  total_participants - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
  sorry

end participants_neither_coffee_nor_tea_l37_37177


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l37_37663

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom a_4 : a 4 = 6
axiom a_6 : a 6 = 10
axiom all_positive_b : ∀ n, 0 < b n
axiom b_3 : b 3 = a 3
axiom T_2 : T 2 = 3

-- Required to prove
theorem arithmetic_sequence_general_formula : ∀ n, a n = 2 * n - 2 :=
sorry

theorem geometric_sequence_sum_formula : ∀ n, T n = 2^n - 1 :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l37_37663


namespace hyperbola_eccentricity_l37_37583

def isHyperbolaWithEccentricity (e : ℝ) : Prop :=
  ∃ (a b : ℝ), a = 4 * b ∧ e = (Real.sqrt (a^2 + b^2)) / a

theorem hyperbola_eccentricity : isHyperbolaWithEccentricity (Real.sqrt 17 / 4) :=
sorry

end hyperbola_eccentricity_l37_37583


namespace probability_third_attempt_success_l37_37163

noncomputable def P_xi_eq_3 : ℚ :=
  (4 / 5) * (3 / 4) * (1 / 3)

theorem probability_third_attempt_success :
  P_xi_eq_3 = 1 / 5 := by
  sorry

end probability_third_attempt_success_l37_37163


namespace m_gt_n_l37_37407

variable (m n : ℝ)

-- Definition of points A and B lying on the line y = -2x + 1
def point_A_on_line : Prop := m = -2 * (-1) + 1
def point_B_on_line : Prop := n = -2 * 3 + 1

-- Theorem stating that m > n given the conditions
theorem m_gt_n (hA : point_A_on_line m) (hB : point_B_on_line n) : m > n :=
by
  -- To avoid the proof part, which we skip as per instructions
  sorry

end m_gt_n_l37_37407


namespace smallest_possible_sum_l37_37198

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37198


namespace sum_of_squares_of_roots_of_quadratic_l37_37036

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end sum_of_squares_of_roots_of_quadratic_l37_37036


namespace sqrt_3_between_inequalities_l37_37409

theorem sqrt_3_between_inequalities (n : ℕ) (h1 : 1 + (3 : ℝ) / (n + 1) < Real.sqrt 3) (h2 : Real.sqrt 3 < 1 + (3 : ℝ) / n) : n = 4 := 
sorry

end sqrt_3_between_inequalities_l37_37409


namespace intersection_domains_l37_37586

def domain_f := {x : ℝ | x < 1}
def domain_g := {x : ℝ | x ≠ 0}

theorem intersection_domains :
  {x : ℝ | x < 1} ∩ {x : ℝ | x ≠ 0} = {x : ℝ | x < 1 ∧ x ≠ 0} :=
by 
  sorry

end intersection_domains_l37_37586


namespace travel_time_equation_l37_37550

theorem travel_time_equation
 (d : ℝ) (x t_saved factor : ℝ) 
 (h : d = 202) 
 (h1 : t_saved = 1.8) 
 (h2 : factor = 1.6)
 : (d / x) * factor = d / (x - t_saved) := sorry

end travel_time_equation_l37_37550


namespace altitude_product_difference_eq_zero_l37_37422

variables (A B C P Q H : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited H]
variable {HP HQ BP PC AQ QC AH BH : ℝ}

-- Given conditions
axiom altitude_intersects_at_H : true
axiom HP_val : HP = 3
axiom HQ_val : HQ = 7

-- Statement to prove
theorem altitude_product_difference_eq_zero (h_BP_PC : BP * PC = 3 / (AH + 3))
                                           (h_AQ_QC : AQ * QC = 7 / (BH + 7))
                                           (h_AH_BQ_ratio : AH / BH = 3 / 7) :
  (BP * PC) - (AQ * QC) = 0 :=
by sorry

end altitude_product_difference_eq_zero_l37_37422


namespace total_fish_in_pond_l37_37869

theorem total_fish_in_pond (N : ℕ) (h1 : 80 ≤ N) (h2 : 5 ≤ 150) (h_marked_dist : (5 : ℚ) / 150 = (80 : ℚ) / N) : N = 2400 := by
  sorry

end total_fish_in_pond_l37_37869


namespace y_pow_x_eq_x_pow_y_l37_37238

open Real

noncomputable def x (n : ℕ) : ℝ := (1 + 1 / n) ^ n
noncomputable def y (n : ℕ) : ℝ := (1 + 1 / n) ^ (n + 1)

theorem y_pow_x_eq_x_pow_y (n : ℕ) (hn : 0 < n) : (y n) ^ (x n) = (x n) ^ (y n) :=
by
  sorry

end y_pow_x_eq_x_pow_y_l37_37238


namespace barbell_cost_l37_37431

theorem barbell_cost (num_barbells : ℤ) (total_money_given : ℤ) 
  (change_received : ℤ) (total_cost : ℤ) (each_barbell_cost : ℤ) 
  (h1 : num_barbells = 3) (h2 : total_money_given = 850) 
  (h3 : change_received = 40) (h4 : total_cost = total_money_given - change_received)
  (h5 : each_barbell_cost = total_cost / num_barbells) 
  : each_barbell_cost = 270 :=
by
  rw [h2, h3] at h4
  rw [← h4, h1] at h5
  exact calc 
    each_barbell_cost = (total_money_given - change_received) / num_barbells : h5
                    ... = (850 - 40) / 3 : by rw [h2, h3, h1]
                    ... = 810 / 3 : rfl
                    ... = 270 : rfl

end barbell_cost_l37_37431


namespace faith_earnings_correct_l37_37561

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l37_37561


namespace abs_eq_two_l37_37237

theorem abs_eq_two (m : ℤ) (h : |m| = 2) : m = 2 ∨ m = -2 :=
sorry

end abs_eq_two_l37_37237


namespace measure_α_l37_37168

noncomputable def measure_α_proof (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : ℝ :=
  let α := 120
  α

theorem measure_α (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : measure_α_proof AB BC h1 h2 = 120 :=
  sorry

end measure_α_l37_37168


namespace radius_ratio_of_smaller_to_larger_l37_37533

noncomputable def ratio_of_radii (v_large v_small : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = v_large) (h_small : v_small = 0.25 * v_large) (h_small_sphere : (4/3) * Real.pi * r^3 = v_small) : ℝ :=
  let ratio := r / R
  ratio

theorem radius_ratio_of_smaller_to_larger (v_large : ℝ) (R r : ℝ) (h_large : (4/3) * Real.pi * R^3 = 576 * Real.pi) (h_small_sphere : (4/3) * Real.pi * r^3 = 0.25 * 576 * Real.pi) : r / R = 1 / (2^(2/3)) :=
by
  sorry

end radius_ratio_of_smaller_to_larger_l37_37533


namespace sequence_value_2016_l37_37054

theorem sequence_value_2016 :
  ∀ (a : ℕ → ℤ),
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
    a 2016 = -3 :=
by
  sorry

end sequence_value_2016_l37_37054


namespace smallest_n_l37_37713

theorem smallest_n (n : ℕ) (h1 : n ≥ 1)
  (h2 : ∃ k : ℕ, 2002 * n = k ^ 3)
  (h3 : ∃ m : ℕ, n = 2002 * m ^ 2) :
  n = 2002^5 := sorry

end smallest_n_l37_37713


namespace price_after_discounts_l37_37114

theorem price_after_discounts (full_price : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  full_price = 85 → 
  price_after_first_discount = full_price * (1 - 0.20) → 
  price_after_second_discount = price_after_first_discount * (1 - 0.25) → 
  price_after_second_discount = 51 :=
by
  intro h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  rw h3
  sorry

end price_after_discounts_l37_37114


namespace expression_value_l37_37581

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l37_37581


namespace hyperbola_equation_l37_37888

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                           (h3 : b = 2 * a) (h4 : ((4 : ℝ), 1) ∈ {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1}) :
    {p : ℝ × ℝ | (p.1)^2 / 12 - (p.2)^2 / 3 = 1} = {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1} :=
by
  sorry

end hyperbola_equation_l37_37888


namespace movies_left_to_watch_l37_37512

theorem movies_left_to_watch (total_movies : ℕ) (movies_watched : ℕ) : total_movies = 17 ∧ movies_watched = 7 → (total_movies - movies_watched) = 10 :=
by
  sorry

end movies_left_to_watch_l37_37512


namespace gcd_75_225_l37_37948

theorem gcd_75_225 : Int.gcd 75 225 = 75 :=
by
  sorry

end gcd_75_225_l37_37948


namespace count_non_congruent_triangles_with_perimeter_10_l37_37735

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def unique_triangles_with_perimeter_10 : finset (ℕ × ℕ × ℕ) :=
  ((finset.range 11).product (finset.range 11)).product (finset.range 11)
  |>.filter (λ t, 
    let (a, b, c) := t.fst.fst, t.fst.snd, t.snd in
      a + b + c = 10 ∧ a ≤ b ∧ b ≤ c ∧ is_triangle a b c)

theorem count_non_congruent_triangles_with_perimeter_10 : 
  unique_triangles_with_perimeter_10.card = 3 := 
sorry

end count_non_congruent_triangles_with_perimeter_10_l37_37735


namespace more_red_than_yellow_l37_37269

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l37_37269


namespace find_a_and_solve_inequality_l37_37585

theorem find_a_and_solve_inequality :
  (∀ x : ℝ, |x^2 - 4 * x + a| + |x - 3| ≤ 5 → x ≤ 3) →
  a = 8 :=
by
  sorry

end find_a_and_solve_inequality_l37_37585


namespace number_of_friends_l37_37454

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37454


namespace increasing_function_range_a_l37_37574

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (a - 1) * x + 3 * a - 4 else a^x

theorem increasing_function_range_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 1 < a ∧ a ≤ 5 / 3 :=
sorry

end increasing_function_range_a_l37_37574


namespace number_x_is_divided_by_l37_37144

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by_l37_37144


namespace monotonicity_intervals_inequality_condition_l37_37732

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + 2 * x + 1)

theorem monotonicity_intervals :
  (∀ x ∈ Set.Iio (-3 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ), 0 > (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) := sorry

theorem inequality_condition (a : ℝ) : 
  (∀ x > 0, Real.exp x * (x^2 + 2 * x + 1) > a * x^2 + a * x + 1) ↔ a ≤ 3 := sorry

end monotonicity_intervals_inequality_condition_l37_37732


namespace distance_of_route_l37_37271

-- Define the conditions
def round_trip_time : ℝ := 1 -- in hours
def avg_speed : ℝ := 3 -- in miles per hour
def return_speed : ℝ := 6.000000000000002 -- in miles per hour

-- Problem statement to prove
theorem distance_of_route : 
  ∃ (D : ℝ), 
  2 * D = avg_speed * round_trip_time ∧ 
  D = 1.5 := 
by
  sorry

end distance_of_route_l37_37271


namespace prob_both_selected_l37_37527

-- Define the probabilities of selection
def prob_selection_x : ℚ := 1 / 5
def prob_selection_y : ℚ := 2 / 3

-- Prove that the probability that both x and y are selected is 2 / 15
theorem prob_both_selected : prob_selection_x * prob_selection_y = 2 / 15 := 
by
  sorry

end prob_both_selected_l37_37527


namespace general_formula_constant_c_value_l37_37577

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) - a n = d

-- Given sequence {a_n}
variables {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
-- Conditions
variables (h1 : a 3 * a 4 = 117) (h2 : a 2 + a 5 = 22) (hd_pos : d > 0)
-- Proof that the general formula for the sequence {a_n} is a_n = 4n - 3
theorem general_formula :
  (∀ n, a n = 4 * n - 3) :=
sorry

-- Given new sequence {b_n}
variables (b : ℕ → ℕ → ℝ) {c : ℝ} (hc : c ≠ 0)
-- New condition that bn is an arithmetic sequence
variables (h_b1 : b 1 = S 1 / (1 + c)) (h_b2 : b 2 = S 2 / (2 + c)) (h_b3 : b 3 = S 3 / (3 + c))
-- Proof that c = -1/2 is the correct constant
theorem constant_c_value :
  (c = -1 / 2) :=
sorry

end general_formula_constant_c_value_l37_37577


namespace smallest_population_multiple_of_3_l37_37499

theorem smallest_population_multiple_of_3 : 
  ∃ (a : ℕ), ∃ (b c : ℕ), 
  a^2 + 50 = b^2 + 1 ∧ b^2 + 51 = c^2 ∧ 
  (∃ m : ℕ, a * a = 576 ∧ 576 = 3 * m) :=
by
  sorry

end smallest_population_multiple_of_3_l37_37499


namespace problem1_problem2_l37_37394

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) := 3 * a^2 * Real.log x + b

theorem problem1 (a b x₀ : ℝ) (h : x₀ = a):
  a > 0 →
  (1 / 2) * x₀^2 + 2 * a * x₀ = 3 * a^2 * Real.log x₀ + b →
  x₀ + 2 * a = 3 * a^2 / x₀ →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := sorry

theorem problem2 (a b : ℝ):
  -2 ≤ b ∧ b ≤ 2 →
  ∀ x > 0, x < 4 →
  ∀ x, x - b + 3 * a^2 / x ≥ 0 →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := sorry

end problem1_problem2_l37_37394


namespace fried_busy_frog_l37_37693

open ProbabilityTheory

def initial_position : (ℤ × ℤ) := (0, 0)

def possible_moves : List (ℤ × ℤ) := [(0, 0), (1, 0), (0, 1)]

def p (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = initial_position then 1 else 0

noncomputable def transition (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = (0, 0) then 1/3 * p n (0, 0)
  else if pos = (0, 1) then 1/3 * p n (0, 0) + 1/3 * p n (0, 1)
  else if pos = (1, 0) then 1/3 * p n (0, 0) + 1/3 * p n (1, 0)
  else 0

noncomputable def p_1 (pos : ℤ × ℤ) : ℚ := transition 0 pos

noncomputable def p_2 (pos : ℤ × ℤ) : ℚ := transition 1 pos

noncomputable def p_3 (pos : ℤ × ℤ) : ℚ := transition 2 pos

theorem fried_busy_frog :
  p_3 (0, 0) = 1/27 :=
by
  sorry

end fried_busy_frog_l37_37693


namespace inequality_proof_l37_37993

theorem inequality_proof (a b : ℝ) (x y : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_x : 0 < x) (h_y : 0 < y) : 
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) :=
sorry

end inequality_proof_l37_37993


namespace sin_of_3halfpiplus2theta_l37_37714

theorem sin_of_3halfpiplus2theta (θ : ℝ) (h : Real.tan θ = 1 / 3) : Real.sin (3 * π / 2 + 2 * θ) = -4 / 5 := 
by 
  sorry

end sin_of_3halfpiplus2theta_l37_37714


namespace rain_on_at_least_one_day_l37_37870

noncomputable def rain_prob : ℕ → ℚ
| 0 => 0.30  -- Probability of rain on Saturday
| 1 => 0.70  -- Probability of rain on Sunday if it rains on Saturday
| 2 => 0.60  -- Probability of rain on Sunday if it does not rain on Saturday

theorem rain_on_at_least_one_day :
  let p_sat := rain_prob 0 in
  let p_sun_given_rain_sat := rain_prob 1 in
  let p_sun_given_no_rain_sat := rain_prob 2 in
  let p_no_rain_sat := 1 - p_sat in
  let p_no_rain_sun_given_no_rain_sat := (1 - p_sun_given_no_rain_sat) in
  let p_no_rain_both_days := p_no_rain_sat * p_no_rain_sun_given_no_rain_sat in
  let p_rain_at_least_one_day := 1 - p_no_rain_both_days in
  p_rain_at_least_one_day = 0.72 :=
by
  sorry

end rain_on_at_least_one_day_l37_37870


namespace tangent_and_normal_lines_l37_37662

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l37_37662


namespace problem_solution_l37_37575

theorem problem_solution (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (∀ x y, (2 * y / x) + (8 * x / y) > m^2 + 2 * m) → -4 < m ∧ m < 2 :=
by
  intros h
  sorry

end problem_solution_l37_37575


namespace mass_percentage_of_Cl_in_bleach_l37_37350

-- Definitions based on conditions
def Na_molar_mass : Float := 22.99
def Cl_molar_mass : Float := 35.45
def O_molar_mass : Float := 16.00

def NaClO_molar_mass : Float := Na_molar_mass + Cl_molar_mass + O_molar_mass

def mass_NaClO (mass_na: Float) (mass_cl: Float) (mass_o: Float) : Float :=
  mass_na + mass_cl + mass_o

def mass_of_NaClO : Float := 100.0

def mass_of_Cl_in_NaClO (mass_of_NaClO: Float) : Float :=
  (Cl_molar_mass / NaClO_molar_mass) * mass_of_NaClO

-- Statement to prove
theorem mass_percentage_of_Cl_in_bleach :
  let mass_Cl := mass_of_Cl_in_NaClO mass_of_NaClO
  (mass_Cl / mass_of_NaClO) * 100 = 47.61 :=
by 
  -- Skip the proof
  sorry

end mass_percentage_of_Cl_in_bleach_l37_37350


namespace simplify_expression_l37_37118

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 2))) = (Real.sqrt 3 - 2 * Real.sqrt 5 - 3) :=
by
  sorry

end simplify_expression_l37_37118


namespace trivia_team_members_l37_37548

theorem trivia_team_members (n p s x y : ℕ) (h1 : n = 12) (h2 : p = 64) (h3 : s = 8) (h4 : x = p / s) (h5 : y = n - x) : y = 4 :=
by
  sorry

end trivia_team_members_l37_37548


namespace find_initial_terms_l37_37720

theorem find_initial_terms (a : ℕ → ℕ) (h : ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n))
  (a6 : a 6 = 2288) : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
by
  sorry

end find_initial_terms_l37_37720


namespace solve_fractional_equation_l37_37941

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (2 * x) / (x - 3) = 1 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l37_37941


namespace birthday_friends_count_l37_37477

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37477


namespace supplementary_angle_difference_l37_37497

theorem supplementary_angle_difference (a b : ℝ) (h1 : a + b = 180) (h2 : 5 * b = 3 * a) : abs (a - b) = 45 :=
  sorry

end supplementary_angle_difference_l37_37497


namespace number_of_friends_l37_37452

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37452


namespace count_4_digit_numbers_with_conditions_l37_37000

def num_valid_numbers : Nat :=
  432

-- Statement declaring the proposition to be proved
theorem count_4_digit_numbers_with_conditions :
  (count_valid_numbers == 432) :=
sorry

end count_4_digit_numbers_with_conditions_l37_37000


namespace quadratic_inequality_solution_l37_37031

theorem quadratic_inequality_solution (x : ℝ) :
  (x < -7 ∨ x > 3) → x^2 + 4 * x - 21 > 0 :=
by
  -- The proof will go here
  sorry

end quadratic_inequality_solution_l37_37031


namespace twelve_pow_six_mod_eight_l37_37935

theorem twelve_pow_six_mod_eight : ∃ m : ℕ, 0 ≤ m ∧ m < 8 ∧ 12^6 % 8 = m ∧ m = 0 := by
  sorry

end twelve_pow_six_mod_eight_l37_37935


namespace remainder_when_a_squared_times_b_divided_by_n_l37_37435

theorem remainder_when_a_squared_times_b_divided_by_n (n : ℕ) (a : ℤ) (h1 : a * 3 ≡ 1 [ZMOD n]) : 
  (a^2 * 3) % n = a % n := 
by
  sorry

end remainder_when_a_squared_times_b_divided_by_n_l37_37435


namespace total_blue_marbles_l37_37260

def jason_blue_marbles : Nat := 44
def tom_blue_marbles : Nat := 24

theorem total_blue_marbles : jason_blue_marbles + tom_blue_marbles = 68 := by
  sorry

end total_blue_marbles_l37_37260


namespace quadratic_expression_transformation_l37_37074

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l37_37074


namespace henrietta_paint_gallons_l37_37057

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l37_37057


namespace halved_r_value_of_n_l37_37607

theorem halved_r_value_of_n (r a : ℝ) (n : ℕ) (h₁ : a = (2 * r)^n)
  (h₂ : 0.125 * a = r^n) : n = 3 :=
by
  sorry

end halved_r_value_of_n_l37_37607


namespace problem_1_problem_2_l37_37687

-- Problem 1
theorem problem_1 :
  -((1 / 2) / 3) * (3 - (-3)^2) = 1 :=
by
  sorry

-- Problem 2
theorem problem_2 {x : ℝ} (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 * x) / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) :=
by
  sorry

end problem_1_problem_2_l37_37687


namespace sqrt_x2y_l37_37597

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end sqrt_x2y_l37_37597


namespace simplify_expression_l37_37934

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

theorem simplify_expression : (x⁻¹ - y) ^ 2 = (1 / x ^ 2 - 2 * y / x + y ^ 2) :=
  sorry

end simplify_expression_l37_37934


namespace intersection_M_N_l37_37397

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} :=
sorry

end intersection_M_N_l37_37397


namespace handshakes_meeting_l37_37902

theorem handshakes_meeting (x : ℕ) (h : x * (x - 1) / 2 = 66) : x = 12 := 
by 
  sorry

end handshakes_meeting_l37_37902


namespace quarters_spent_l37_37924

variable (q_initial q_left q_spent : ℕ)

theorem quarters_spent (h1 : q_initial = 11) (h2 : q_left = 7) : q_spent = q_initial - q_left ∧ q_spent = 4 :=
by
  sorry

end quarters_spent_l37_37924


namespace boy_two_girls_work_completion_days_l37_37338

-- Work rates definitions
def man_work_rate := 1 / 6
def woman_work_rate := 1 / 18
def girl_work_rate := 1 / 12
def team_work_rate := 1 / 3

-- Boy's work rate
def boy_work_rate := 1 / 36

-- Combined work rate of boy and two girls
def boy_two_girls_work_rate := boy_work_rate + 2 * girl_work_rate

-- Prove that the number of days it will take for a boy and two girls to complete the work is 36 / 7
theorem boy_two_girls_work_completion_days : (1 / boy_two_girls_work_rate) = 36 / 7 :=
by
  sorry

end boy_two_girls_work_completion_days_l37_37338


namespace total_cost_of_puzzles_l37_37964

-- Definitions for the costs of large and small puzzles
def large_puzzle_cost : ℕ := 15
def small_puzzle_cost : ℕ := 23 - large_puzzle_cost

-- Theorem statement
theorem total_cost_of_puzzles :
  (large_puzzle_cost + 3 * small_puzzle_cost) = 39 :=
by
  -- Placeholder for the proof
  sorry

end total_cost_of_puzzles_l37_37964


namespace jack_buttons_l37_37427

theorem jack_buttons :
  ∀ (shirts_per_kid kids buttons_per_shirt : ℕ),
  shirts_per_kid = 3 →
  kids = 3 →
  buttons_per_shirt = 7 →
  (shirts_per_kid * kids * buttons_per_shirt) = 63 :=
by
  intros shirts_per_kid kids buttons_per_shirt h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 3 * 7 = 9 * 7 : by rw mul_assoc
            ... = 63   : by norm_num

end jack_buttons_l37_37427


namespace cartons_in_case_l37_37665

theorem cartons_in_case (b : ℕ) (hb : b ≥ 1) (h : 2 * c * b * 500 = 1000) : c = 1 :=
by
  -- sorry is used to indicate where the proof would go
  sorry

end cartons_in_case_l37_37665


namespace find_pairs_satisfying_conditions_l37_37707

theorem find_pairs_satisfying_conditions :
  ∀ (m n : ℕ), (0 < m ∧ 0 < n) →
               (∃ k : ℤ, m^2 - 4 * n = k^2) →
               (∃ l : ℤ, n^2 - 4 * m = l^2) →
               (m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5) :=
by
  intros m n hmn h1 h2
  sorry

end find_pairs_satisfying_conditions_l37_37707


namespace age_ratio_in_8_years_l37_37571

-- Define the conditions
variables (s l : ℕ) -- Sam's and Leo's current ages

def condition1 := s - 4 = 2 * (l - 4)
def condition2 := s - 10 = 3 * (l - 10)

-- Define the final problem
theorem age_ratio_in_8_years (h1 : condition1 s l) (h2 : condition2 s l) : 
  ∃ x : ℕ, x = 8 ∧ (s + x) / (l + x) = 3 / 2 :=
sorry

end age_ratio_in_8_years_l37_37571


namespace negation_equiv_l37_37127

-- Define the initial proposition
def initial_proposition (x : ℝ) : Prop :=
  x^2 - x + 1 > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0

-- The statement asserting the negation equivalence
theorem negation_equiv :
  (¬ ∀ x : ℝ, initial_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l37_37127


namespace smallest_x_plus_y_l37_37207

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) 
(h4 : 1 / (x:ℝ) + 1 / (y:ℝ) = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37207


namespace birthday_friends_count_l37_37474

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37474


namespace rearrangement_impossible_l37_37528

-- Definition of an 8x8 chessboard's cell numbering.
def cell_number (i j : ℕ) : ℕ := i + j - 1

-- The initial placement of pieces, represented as a permutation on {1, 2, ..., 8}
def initial_placement (p: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- The rearranged placement of pieces
def rearranged_placement (q: Fin 8 → Fin 8) := True -- simplify for definition purposes

-- Condition for each piece: cell number increases
def cell_increase_condition (p q: Fin 8 → Fin 8) : Prop :=
  ∀ i, cell_number (q i).val (i.val + 1) > cell_number (p i).val (i.val + 1)

-- The main theorem to state it's impossible to rearrange under the given conditions and question
theorem rearrangement_impossible 
  (p q: Fin 8 → Fin 8) 
  (h_initial : initial_placement p) 
  (h_rearranged : rearranged_placement q) 
  (h_increase : cell_increase_condition p q) : False := 
sorry

end rearrangement_impossible_l37_37528


namespace first_divisor_l37_37318

theorem first_divisor (d x : ℕ) (h1 : ∃ k : ℕ, x = k * d + 11) (h2 : ∃ m : ℕ, x = 9 * m + 2) : d = 3 :=
sorry

end first_divisor_l37_37318


namespace students_who_walk_home_l37_37552

theorem students_who_walk_home :
  let car_fraction := (1 : ℚ) / 3
  let bus_fraction := (1 : ℚ) / 5
  let cycle_fraction := (1 : ℚ) / 8
  1 - (car_fraction + bus_fraction + cycle_fraction) = 41 / 120 :=
by {
  let car_fraction := (1 : ℚ) / 3,
  let bus_fraction := (1 : ℚ) / 5,
  let cycle_fraction := (1 : ℚ) / 8,
  let total_fraction := car_fraction + bus_fraction + cycle_fraction,
  have h1 : total_fraction = 79 / 120, { sorry },  -- Calculation step to be filled
  have h2 : 1 = 120 / 120, from by norm_num,
  calc 
    1 - total_fraction 
        = 120 / 120 - 79 / 120 : by rw [h2]
    ... = 41 / 120 : by norm_num
}

end students_who_walk_home_l37_37552


namespace no_solution_for_equation_l37_37688

/-- The given equation expressed using letters as unique digits:
    ∑ (letters as digits) from БАРАНКА + БАРАБАН + КАРАБАС = ПАРАЗИТ
    We aim to prove that there are no valid digit assignments satisfying the equation. -/
theorem no_solution_for_equation :
  ∀ (b a r n k s p i t: ℕ),
  b ≠ a ∧ b ≠ r ∧ b ≠ n ∧ b ≠ k ∧ b ≠ s ∧ b ≠ p ∧ b ≠ i ∧ b ≠ t ∧
  a ≠ r ∧ a ≠ n ∧ a ≠ k ∧ a ≠ s ∧ a ≠ p ∧ a ≠ i ∧ a ≠ t ∧
  r ≠ n ∧ r ≠ k ∧ r ≠ s ∧ r ≠ p ∧ r ≠ i ∧ r ≠ t ∧
  n ≠ k ∧ n ≠ s ∧ n ≠ p ∧ n ≠ i ∧ n ≠ t ∧
  k ≠ s ∧ k ≠ p ∧ k ≠ i ∧ k ≠ t ∧
  s ≠ p ∧ s ≠ i ∧ s ≠ t ∧
  p ≠ i ∧ p ≠ t ∧
  i ≠ t →
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * n + k +
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * b + a + n +
  100000 * k + 10000 * a + 1000 * r + 100 * a + 10 * b + a + s ≠ 
  100000 * p + 10000 * a + 1000 * r + 100 * a + 10 * z + i + t :=
sorry

end no_solution_for_equation_l37_37688


namespace minimize_expression_at_9_l37_37854

noncomputable def minimize_expression (n : ℕ) : ℚ :=
  n / 3 + 27 / n

theorem minimize_expression_at_9 : minimize_expression 9 = 6 := by
  sorry

end minimize_expression_at_9_l37_37854


namespace zeros_at_end_of_product1_value_of_product2_l37_37302

-- Definitions and conditions
def product1 := 360 * 5
def product2 := 250 * 4

-- Statements of the proof problems
theorem zeros_at_end_of_product1 : Nat.digits 10 product1 = [0, 0, 8, 1] := by
  sorry

theorem value_of_product2 : product2 = 1000 := by
  sorry

end zeros_at_end_of_product1_value_of_product2_l37_37302


namespace machine_value_after_two_years_l37_37938

theorem machine_value_after_two_years (initial_value : ℝ) (decrease_rate : ℝ) (years : ℕ) (value_after_two_years : ℝ) :
  initial_value = 8000 ∧ decrease_rate = 0.30 ∧ years = 2 → value_after_two_years = 3200 := by
  intros h
  sorry

end machine_value_after_two_years_l37_37938


namespace part1_part2_l37_37731

noncomputable def f (a x : ℝ) : ℝ := (a / 2) * x * x - (a - 2) * x - 2 * x * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := a * x - a - 2 * Real.log x

theorem part1 (a : ℝ) : (∀ x > 0, f' a x ≥ 0) ↔ a = 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : f' a x1 = 0) (h4 : f' a x2 = 0) (h5 : x1 < x2) : 
  x2 - x1 > 4 / a - 2 :=
sorry

end part1_part2_l37_37731


namespace find_a_and_b_maximize_profit_l37_37676

variable (a b x : ℝ)

-- The given conditions
def condition1 : Prop := 2 * a + b = 120
def condition2 : Prop := 4 * a + 3 * b = 270
def constraint : Prop := 75 ≤ 300 - x

-- The questions translated into a proof problem
theorem find_a_and_b :
  condition1 a b ∧ condition2 a b → a = 45 ∧ b = 30 :=
by
  intros h
  sorry

theorem maximize_profit (a : ℝ) (b : ℝ) (x : ℝ) :
  condition1 a b → condition2 a b → constraint x →
  x = 75 → (300 - x) = 225 → 
  (10 * x + 20 * (300 - x) = 5250) :=
by
  intros h1 h2 hc hx hx1
  sorry

end find_a_and_b_maximize_profit_l37_37676


namespace range_of_a_l37_37356

def S : Set ℝ := {x | (x - 2) ^ 2 > 9 }
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8 }

theorem range_of_a (a : ℝ) : (S ∪ T a) = Set.univ ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l37_37356


namespace compare_log_exp_powers_l37_37033

variable (a b c : ℝ)

theorem compare_log_exp_powers (h1 : a = Real.log 0.3 / Real.log 2)
                               (h2 : b = Real.exp (Real.log 2 * 0.1))
                               (h3 : c = Real.exp (Real.log 0.2 * 1.3)) :
  a < c ∧ c < b :=
by
  sorry

end compare_log_exp_powers_l37_37033


namespace inverse_var_y_l37_37325

theorem inverse_var_y (k : ℝ) (y x : ℝ)
  (h1 : 5 * y = k / x^2)
  (h2 : y = 16) (h3 : x = 1) (h4 : k = 80) :
  y = 1 / 4 :=
by
  sorry

end inverse_var_y_l37_37325


namespace find_angle_B_l37_37081

-- Define the necessary trigonometric identities and dependencies
open Real

-- Declare the conditions under which we are working
theorem find_angle_B : 
  ∀ {a b A B : ℝ}, 
    a = 1 → 
    b = sqrt 3 → 
    A = π / 6 → 
    (B = π / 3 ∨ B = 2 * π / 3) := 
  by 
    intros a b A B ha hb hA
    sorry

end find_angle_B_l37_37081


namespace problem1_problem2_problem3_l37_37197

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^2 + 0.5 * x

theorem problem1 (h : ∀ x : ℝ, f (x + 1) = f x + x + 1) (h0 : f 0 = 0) : 
  ∀ x : ℝ, f x = 0.5 * x^2 + 0.5 * x := by 
  sorry

noncomputable def g (t : ℝ) : ℝ :=
  if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
  else if -1.5 < t ∧ t < -0.5 then -1 / 8
  else 0.5 * t^2 + 0.5 * t

theorem problem2 (h : ∀ t : ℝ, g t = min (f (t)) (f (t + 1))) : 
  ∀ t : ℝ, g t = 
    if t ≤ -1.5 then 0.5 * t^2 + 1.5 * t + 1
    else if -1.5 < t ∧ t < -0.5 then -1 / 8
    else 0.5 * t^2 + 0.5 * t := by 
  sorry

theorem problem3 (m : ℝ) : (∀ t : ℝ, g t + m ≥ 0) → m ≥ 1 / 8 := by 
  sorry

end problem1_problem2_problem3_l37_37197


namespace hyperbola_equation_center_origin_asymptote_l37_37383

theorem hyperbola_equation_center_origin_asymptote
  (center_origin : ∀ x y : ℝ, x = 0 ∧ y = 0)
  (focus_parabola : ∃ x : ℝ, 4 * x^2 = 8 * x)
  (asymptote : ∀ x y : ℝ, x + y = 0):
  ∃ a b : ℝ, a^2 = 2 ∧ b^2 = 2 ∧ (x^2 / 2) - (y^2 / 2) = 1 := 
sorry

end hyperbola_equation_center_origin_asymptote_l37_37383


namespace bobby_candy_l37_37006

theorem bobby_candy (C G : ℕ) (H : C + G = 36) (Hchoc: (2/3 : ℚ) * C = 12) (Hgummy: (3/4 : ℚ) * G = 9) : 
  (1/3 : ℚ) * C + (1/4 : ℚ) * G = 9 :=
by
  sorry

end bobby_candy_l37_37006


namespace remainder_div_x_plus_1_l37_37716

noncomputable def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_div_x_plus_1 : 
  (f (-1) = 4) := 
by
  sorry

end remainder_div_x_plus_1_l37_37716


namespace number_of_mango_trees_l37_37627

-- Define the conditions
variable (M : Nat) -- Number of mango trees
def num_papaya_trees := 2
def papayas_per_tree := 10
def mangos_per_tree := 20
def total_fruits := 80

-- Prove that the number of mango trees M is equal to 3
theorem number_of_mango_trees : 20 + (mangos_per_tree * M) = total_fruits -> M = 3 :=
by
  intro h
  sorry

end number_of_mango_trees_l37_37627


namespace john_streams_hours_per_day_l37_37093

theorem john_streams_hours_per_day :
  (∃ h : ℕ, (7 - 3) * h * 10 = 160) → 
  (∃ h : ℕ, h = 4) :=
sorry

end john_streams_hours_per_day_l37_37093


namespace intersection_points_l37_37488

theorem intersection_points (g : ℝ → ℝ) (hg_inv : Function.Injective g) : 
  ∃ n, n = 3 ∧ ∀ x, g (x^3) = g (x^5) ↔ x = 0 ∨ x = 1 ∨ x = -1 :=
by {
  sorry
}

end intersection_points_l37_37488


namespace geometric_sequence_first_term_l37_37644

theorem geometric_sequence_first_term (a r : ℝ) 
  (h1 : a * r^2 = 3)
  (h2 : a * r^4 = 27) : 
  a = - (real.sqrt 9) / 9 :=
by
  sorry

end geometric_sequence_first_term_l37_37644


namespace yuna_grandfather_age_l37_37522

def age_yuna : ℕ := 8
def age_father : ℕ := age_yuna + 20
def age_grandfather : ℕ := age_father + 25

theorem yuna_grandfather_age : age_grandfather = 53 := by
  sorry

end yuna_grandfather_age_l37_37522


namespace gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l37_37999

theorem gcd_b_squared_plus_11b_plus_28_and_b_plus_6 (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l37_37999


namespace tables_left_l37_37172

theorem tables_left (original_tables number_of_customers_per_table current_customers : ℝ) 
(h1 : original_tables = 44.0)
(h2 : number_of_customers_per_table = 8.0)
(h3 : current_customers = 256) : 
(original_tables - current_customers / number_of_customers_per_table) = 12.0 :=
by
  sorry

end tables_left_l37_37172


namespace max_non_managers_l37_37745

/-- In a department, the ratio of managers to non-managers must always be greater than 7:32.
If the maximum number of non-managers is 36, then prove that the highest number cannot exceed 36,
under the given ratio constraint. -/
theorem max_non_managers (M N : ℕ) (h1 : M > 0)
  (h2 : N ≤ 36) (h3 : (M:ℚ) / (N:ℚ) > 7 / 32) : N = 36 :=
begin
  -- Given that we have the initial ratio condition and the constraints on N,
  -- we aim to prove that N must be equal to the maximum given value.
  suffices : N = 36, from this,
  
  -- Assume the contrary, and proceed by contradiction to establish that N cannot be less than 36.
  by_contradiction h,
  have h_leq := lt_of_le_of_ne (le_of_not_gt h) (ne.symm h),
  -- The ratio condition given:
  -- (M / N) > 7 / 32
  
  -- Inequality setup, translated to real numbers:
  let bound := 252 / 32,
  have M_gt : (M:ℚ) > bound := by 
    calc 
      (M:ℚ) = (M:ℚ)    : by simp 
      ...    > (252:ℚ)/32 : by sorry, -- This follows from the ratio constraint (h3)

  -- Contradiction establishment
  calc 
    252 / 32 ≈ 7.875 : by simp [bound]
    ... < N    : by sorry, -- As per the condition of maximum N
  -- However, nothing contradicts this calculation that allows N to be smaller 

  -- Decompose the above to ensure contradiction 
  suffices h_suff : M ≤ 8, from this,  λ (nq:ne.symm h) this (232 / 32 h2).ne h_suff  
   -- Substitute N=36 favorable bound Value with standards in place ensure maximum falls within 36 range
--Hence 
⟩

end max_non_managers_l37_37745


namespace total_rainfall_l37_37609

theorem total_rainfall (rain_first_hour : ℕ) (rain_second_hour : ℕ) : Prop :=
  rain_first_hour = 5 →
  rain_second_hour = 7 + 2 * rain_first_hour →
  rain_first_hour + rain_second_hour = 22

-- Add sorry to skip the proof.

end total_rainfall_l37_37609


namespace more_red_balls_l37_37265

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l37_37265


namespace probability_of_events_l37_37105

noncomputable def total_types : ℕ := 8

noncomputable def fever_reducing_types : ℕ := 3

noncomputable def cough_suppressing_types : ℕ := 5

noncomputable def total_ways_to_choose_two : ℕ := Nat.choose total_types 2

noncomputable def event_A_ways : ℕ := total_ways_to_choose_two - Nat.choose cough_suppressing_types 2

noncomputable def P_A : ℚ := event_A_ways / total_ways_to_choose_two

noncomputable def event_B_ways : ℕ := fever_reducing_types * cough_suppressing_types

noncomputable def P_B_given_A : ℚ := event_B_ways / event_A_ways

theorem probability_of_events :
  P_A = 9 / 14 ∧ P_B_given_A = 5 / 6 := by
  sorry

end probability_of_events_l37_37105


namespace problem_l37_37878

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * (Real.sqrt 3) * Real.cos x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 
  let dot_product := (a x).fst * (b x).fst + (a x).snd * (b x).snd
  let magnitude_square_b := (b x).fst ^ 2 + (b x).snd ^ 2
  dot_product + magnitude_square_b

theorem problem :
  (∀ x, f x = 5 * Real.sin (2 * x + Real.pi / 6) + 7 / 2) ∧
  (∃ T, T = Real.pi) ∧ 
  (∃ x, f x = 17 / 2) ∧ 
  (∃ x, f x = -3 / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (Real.pi / 6), 0 ≤ x ∧ x ≤ Real.pi / 6) ∧
  (∀ x ∈ Set.Icc (2 * Real.pi / 3) Real.pi, (2 * Real.pi / 3) ≤ x ∧ x ≤ Real.pi)
:= by
  sorry

end problem_l37_37878


namespace rectangle_ratio_l37_37155

open Real

-- Definition of the terms
variables {x y : ℝ}

-- Conditions as per the problem statement
def diagonalSavingsRect (x y : ℝ) := x + y - sqrt (x^2 + y^2) = (2 / 3) * y

-- The ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (h : diagonalSavingsRect x y) : x / y = 8 / 9 :=
by
sorry

end rectangle_ratio_l37_37155


namespace customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l37_37626

theorem customer_B_cost_effectiveness (box_orig_cost box_spec_cost : ℕ) (orig_price spec_price eggs_per_box remaining_eggs : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : remaining_eggs = 20) : 
    ¬ (spec_price * 2 / (eggs_per_box * 2 - remaining_eggs) < orig_price / eggs_per_box) :=
by
  sorry

theorem customer_A_boxes_and_consumption (orig_price spec_price eggs_per_box total_cost_savings : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : total_cost_savings = 90): 
  ∃ (boxes_bought : ℕ) (avg_daily_consumption : ℕ), 
    (spec_price * boxes_bought = orig_price * boxes_bought * 2 - total_cost_savings) ∧ 
    (avg_daily_consumption = eggs_per_box * boxes_bought / 15) :=
by
  sorry

end customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l37_37626


namespace tan_product_identity_l37_37059

theorem tan_product_identity (A B : ℝ) (hA : A = 20) (hB : B = 25) (hSum : A + B = 45) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 := 
  by
  sorry

end tan_product_identity_l37_37059


namespace sum_of_3_consecutive_multiples_of_3_l37_37641

theorem sum_of_3_consecutive_multiples_of_3 (a b c : ℕ) (h₁ : a = b + 3) (h₂ : b = c + 3) (h₃ : a = 42) : a + b + c = 117 :=
by sorry

end sum_of_3_consecutive_multiples_of_3_l37_37641


namespace smallest_possible_sum_l37_37203

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37203


namespace problem_xy_squared_and_product_l37_37404

theorem problem_xy_squared_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 :=
by
  sorry

end problem_xy_squared_and_product_l37_37404


namespace louis_current_age_l37_37415

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l37_37415


namespace original_cost_before_changes_l37_37611

variable (C : ℝ)

theorem original_cost_before_changes (h : 2 * C * 1.20 = 480) : C = 200 :=
by
  -- proof goes here
  sorry

end original_cost_before_changes_l37_37611


namespace parabola_b_value_l37_37883

variable (a b c p : ℝ)
variable (h1 : p ≠ 0)
variable (h2 : ∀ x, y = a*x^2 + b*x + c)
variable (h3 : vertex' y = (p, -p))
variable (h4 : y-intercept' y = (0, p))

theorem parabola_b_value : b = -4 :=
sorry

end parabola_b_value_l37_37883


namespace problem1_problem2_l37_37181

variable (a b : ℝ)

-- Proof problem for Question 1
theorem problem1 : 2 * a * (a^2 - 3 * a - 1) = 2 * a^3 - 6 * a^2 - 2 * a :=
by sorry

-- Proof problem for Question 2
theorem problem2 : (a^2 * b - 2 * a * b^2 + b^3) / b - (a + b)^2 = -4 * a * b :=
by sorry

end problem1_problem2_l37_37181


namespace smallest_sphere_radius_l37_37858

noncomputable def radius_smallest_sphere : ℝ := 2 * Real.sqrt 3 + 2

theorem smallest_sphere_radius (r : ℝ) (h : r = 2) : radius_smallest_sphere = 2 * Real.sqrt 3 + 2 := by
  sorry

end smallest_sphere_radius_l37_37858


namespace values_of_x_l37_37190

theorem values_of_x (x : ℝ) (h1 : x^2 - 3 * x - 10 < 0) (h2 : 1 < x) : 1 < x ∧ x < 5 := 
sorry

end values_of_x_l37_37190


namespace decrease_neg_of_odd_and_decrease_nonneg_l37_37385

-- Define the properties of the function f
variable (f : ℝ → ℝ)

-- f is odd
def odd_function : Prop := ∀ x : ℝ, f (-x) = - f x

-- f is decreasing on [0, +∞)
def decreasing_on_nonneg : Prop := ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2 → f x1 > f x2)

-- Goal: f is decreasing on (-∞, 0)
def decreasing_on_neg : Prop := ∀ x1 x2 : ℝ, (x1 < 0) → (x2 < 0) → (x1 < x2) → f x1 > f x2

-- The theorem to be proved
theorem decrease_neg_of_odd_and_decrease_nonneg 
  (h_odd : odd_function f) (h_decreasing_nonneg : decreasing_on_nonneg f) :
  decreasing_on_neg f :=
sorry

end decrease_neg_of_odd_and_decrease_nonneg_l37_37385


namespace remainder_when_divided_by_multiple_of_10_l37_37317

theorem remainder_when_divided_by_multiple_of_10 (N : ℕ) (hN : ∃ k : ℕ, N = 10 * k) (hrem : (19 ^ 19 + 19) % N = 18) : N = 10 := by
  sorry

end remainder_when_divided_by_multiple_of_10_l37_37317


namespace arithmetic_sequence_sum_l37_37606

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 + a 3 = 2) (h2 : a 2 + a 4 = 6)
  (h_arith : ∀ n, a (n + 1) = a n + d) : a 1 + a 7 = 10 :=
by
  sorry

end arithmetic_sequence_sum_l37_37606


namespace rational_root_of_factors_l37_37279

theorem rational_root_of_factors (p : ℕ) (a : ℚ) (hprime : Nat.Prime p) 
  (f : Polynomial ℚ) (hf : f = Polynomial.X ^ p - Polynomial.C a)
  (hfactors : ∃ g h : Polynomial ℚ, f = g * h ∧ 1 ≤ g.degree ∧ 1 ≤ h.degree) : 
  ∃ r : ℚ, Polynomial.eval r f = 0 :=
sorry

end rational_root_of_factors_l37_37279


namespace calculate_x_n_minus_inverse_x_n_l37_37403

theorem calculate_x_n_minus_inverse_x_n
  (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π) (x : ℝ) (h : x - 1/x = 2 * Real.sin θ) (n : ℕ) (hn : 0 < n) :
  x^n - 1/x^n = 2 * Real.sinh (n * θ) :=
by sorry

end calculate_x_n_minus_inverse_x_n_l37_37403


namespace friends_attended_birthday_l37_37446

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37446


namespace cost_of_adult_ticket_is_8_l37_37175

variables (A : ℕ) (num_people : ℕ := 22) (total_money : ℕ := 50) (num_children : ℕ := 18) (child_ticket_cost : ℕ := 1)

-- Definitions based on the given conditions
def child_tickets_cost := num_children * child_ticket_cost
def num_adults := num_people - num_children
def adult_tickets_cost := total_money - child_tickets_cost
def cost_per_adult_ticket := adult_tickets_cost / num_adults

-- The theorem stating that the cost of an adult ticket is 8 dollars
theorem cost_of_adult_ticket_is_8 : cost_per_adult_ticket = 8 :=
by sorry

end cost_of_adult_ticket_is_8_l37_37175


namespace find_digit_B_l37_37807

theorem find_digit_B (B : ℕ) (h1 : B < 10) : 3 ∣ (5 + 2 + B + 6) → B = 2 :=
by
  sorry

end find_digit_B_l37_37807


namespace fraction_identity_l37_37401

theorem fraction_identity
  (m : ℝ)
  (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 :=
by
  sorry

end fraction_identity_l37_37401


namespace capacity_of_each_type_l37_37173

def total_capacity_barrels : ℕ := 7000

def increased_by_first_type : ℕ := 8000

def decreased_by_second_type : ℕ := 3000

theorem capacity_of_each_type 
  (x y : ℕ) 
  (n k : ℕ)
  (h1 : x + y = total_capacity_barrels)
  (h2 : x * (n + k) / n = increased_by_first_type)
  (h3 : y * (n + k) / k = decreased_by_second_type) :
  x = 6400 ∧ y = 600 := sorry

end capacity_of_each_type_l37_37173


namespace lower_limit_for_x_l37_37743

variable {n : ℝ} {x : ℝ} {y : ℝ}

theorem lower_limit_for_x (h1 : x > n) (h2 : x < 8) (h3 : y > 8) (h4 : y < 13) (h5 : y - x = 7) : x = 2 :=
sorry

end lower_limit_for_x_l37_37743


namespace tribe_leadership_choices_l37_37344

open Nat

theorem tribe_leadership_choices (n m k l : ℕ) (h : n = 15) : 
  (choose 14 2 * choose 12 3 * choose 9 3 * 15 = 27392400) := 
  by sorry

end tribe_leadership_choices_l37_37344


namespace factor_of_polynomial_l37_37108

theorem factor_of_polynomial :
  (x^4 + 4 * x^2 + 16) % (x^2 + 4) = 0 :=
sorry

end factor_of_polynomial_l37_37108


namespace probability_without_favorite_in_6_minutes_correct_l37_37620

noncomputable def probability_without_favorite_in_6_minutes
  (total_songs : ℕ)
  (songs_lengths : Fin total_songs → ℚ)
  (favorite_song_length : ℚ)
  (time_limit : ℚ) : ℚ :=
  have len := total_songs!
  let scenarios := finset.sum (fin_range (total_songs - 1)) (λ i, (total_songs - 1 - i)!)
  (len - scenarios) / len

theorem probability_without_favorite_in_6_minutes_correct :
  probability_without_favorite_in_6_minutes 12 (λ i, 60 + 30 * i) 300 360 = 1813 / 1980 :=
sorry

end probability_without_favorite_in_6_minutes_correct_l37_37620


namespace proof_problem_l37_37043

def h (x : ℝ) : ℝ := x^2 - 3 * x + 7
def k (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : h (k 3) - k (h 3) = 59 := by
  sorry

end proof_problem_l37_37043


namespace value_of_a_l37_37891

theorem value_of_a (a : ℝ) :
  (∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) ↔ (6 ≤ a ∧ a ≤ 9) :=
by
  sorry

end value_of_a_l37_37891


namespace range_of_f_l37_37041

def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f :
  Set.range f = {0, -1} := 
sorry

end range_of_f_l37_37041


namespace remainder_of_expansion_mod_88_l37_37007

theorem remainder_of_expansion_mod_88 :
  ((1 - ∑ k in finset.range 11, (-1)^k * (90^k) * (nat.choose 10 k)) % 88 = 1) :=
by
  sorry

end remainder_of_expansion_mod_88_l37_37007


namespace find_triples_l37_37566

theorem find_triples (a b c : ℝ) : 
  a + b + c = 14 ∧ a^2 + b^2 + c^2 = 84 ∧ a^3 + b^3 + c^3 = 584 ↔ (a = 4 ∧ b = 2 ∧ c = 8) ∨ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 4) :=
by
  sorry

end find_triples_l37_37566


namespace striped_octopus_has_eight_legs_l37_37069

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l37_37069


namespace birthday_friends_count_l37_37459

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37459


namespace longest_diagonal_of_rhombus_l37_37542

variable (d1 d2 : ℝ) (r : ℝ)
variable h_area : 0.5 * d1 * d2 = 150
variable h_ratio : d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus :
  max d1 d2 = 20 :=
by
  sorry

end longest_diagonal_of_rhombus_l37_37542


namespace bus_driver_compensation_l37_37831

theorem bus_driver_compensation : 
  let regular_rate := 16
  let regular_hours := 40
  let total_hours_worked := 57
  let overtime_rate := regular_rate + (0.75 * regular_rate)
  let regular_pay := regular_hours * regular_rate
  let overtime_hours_worked := total_hours_worked - regular_hours
  let overtime_pay := overtime_hours_worked * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 :=
by
  sorry

end bus_driver_compensation_l37_37831


namespace perfect_square_expression_l37_37794

theorem perfect_square_expression : 
    ∀ x : ℝ, (11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02)^2) → (x = 0.4792) :=
by
  intros x h
  -- sorry placeholder for the proof
  sorry

end perfect_square_expression_l37_37794


namespace sticks_form_triangle_l37_37556

theorem sticks_form_triangle:
  (2 + 3 > 4) ∧ (2 + 4 > 3) ∧ (3 + 4 > 2) := by
  sorry

end sticks_form_triangle_l37_37556


namespace true_statement_l37_37925

def statement_i (i : ℕ) (n : ℕ) : Prop := 
  (i = (n - 1))

theorem true_statement :
  ∃! n : ℕ, (n ≤ 100 ∧ ∀ i, (i ≠ n - 1) → statement_i i n = false) ∧ statement_i (n - 1) n = true :=
by
  sorry

end true_statement_l37_37925


namespace trapezoid_diagonals_l37_37291

theorem trapezoid_diagonals {BC AD AB CD AC BD : ℝ} (h b1 b2 : ℝ) 
  (hBC : BC = b1) (hAD : AD = b2) (hAB : AB = h) (hCD : CD = h) 
  (hAC : AC^2 = AB^2 + BC^2) (hBD : BD^2 = CD^2 + AD^2) :
  BD^2 - AC^2 = b2^2 - b1^2 := 
by 
  -- proof is omitted
  sorry

end trapezoid_diagonals_l37_37291


namespace days_for_30_men_to_build_wall_l37_37908

theorem days_for_30_men_to_build_wall 
  (men1 days1 men2 k : ℕ)
  (h1 : men1 = 18)
  (h2 : days1 = 5)
  (h3 : men2 = 30)
  (h_k : men1 * days1 = k)
  : (men2 * 3 = k) := by 
sorry

end days_for_30_men_to_build_wall_l37_37908


namespace intervals_of_increase_and_decrease_of_f_l37_37013

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.log x

theorem intervals_of_increase_and_decrease_of_f :
  (∀ x ∈ Ioi (Real.sqrt 2 / 2), 0 < (deriv f x)) ∧ 
  (∀ x ∈ Ioo 0 (Real.sqrt 2 / 2), (deriv f x) < 0) :=
begin
  -- The proof part is omitted as per the instructions
  sorry
end

end intervals_of_increase_and_decrease_of_f_l37_37013


namespace adult_meals_sold_l37_37681

theorem adult_meals_sold (k a : ℕ) (h1 : 10 * a = 7 * k) (h2 : k = 70) : a = 49 :=
by
  sorry

end adult_meals_sold_l37_37681


namespace fraction_addition_l37_37650

theorem fraction_addition : (1 / 3) + (5 / 12) = 3 / 4 := 
sorry

end fraction_addition_l37_37650


namespace solve_for_x_l37_37487

theorem solve_for_x : ∃ x : ℝ, (x + 36) / 3 = (7 - 2 * x) / 6 ∧ x = -65 / 4 := by
  sorry

end solve_for_x_l37_37487


namespace probability_both_divisible_by_4_when_two_6_sided_dice_tossed_l37_37954

theorem probability_both_divisible_by_4_when_two_6_sided_dice_tossed : 
  let dice := fin 6
  let outcomes := (prod dice dice)
  let favorable := {ab | ab.1 % 4 = 0 ∧ ab.2 % 4 = 0}
  in (favorable.card / outcomes.card) = (1 : ℝ) / 36 :=
by
  -- Definitions
  let dice := fin 6
  let outcomes := (prod dice dice)
  let favorable := {ab | ab.1 % 4 = 0 ∧ ab.2 % 4 = 0}

  -- Calculate the probability
  have h_fav_card : favorable.card = 1 := sorry
  have h_outcomes_card : outcomes.card = 36 := sorry
  have h_prob : (favorable.card : ℝ) / outcomes.card = (1 : ℝ) / 36 := sorry
  exact h_prob

end probability_both_divisible_by_4_when_two_6_sided_dice_tossed_l37_37954


namespace possible_value_is_121_l37_37008

theorem possible_value_is_121
  (x a y z b : ℕ) 
  (hx : x = 1 / 6 * a) 
  (hz : z = 1 / 6 * b) 
  (hy : y = (a + b) % 5) 
  (h_single_digit : ∀ n, n ∈ [x, a, y, z, b] → n < 10 ∧ 0 < n) : 
  100 * x + 10 * y + z = 121 :=
by
  sorry

end possible_value_is_121_l37_37008


namespace circle_numbers_exist_l37_37347

theorem circle_numbers_exist :
  ∃ (a b c d e f : ℚ),
    a = 2 ∧
    b = 3 ∧
    c = 3 / 2 ∧
    d = 1 / 2 ∧
    e = 1 / 3 ∧
    f = 2 / 3 ∧
    a = b * f ∧
    b = a * c ∧
    c = b * d ∧
    d = c * e ∧
    e = d * f ∧
    f = e * a := by
  sorry

end circle_numbers_exist_l37_37347


namespace modular_inverse_sum_correct_l37_37365

theorem modular_inverse_sum_correct :
  (3 * 8 + 9 * 13) % 56 = 29 :=
by
  sorry

end modular_inverse_sum_correct_l37_37365


namespace roberto_outfits_l37_37928

theorem roberto_outfits (trousers shirts jackets : ℕ) (restricted_shirt restricted_jacket : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_restricted_shirt : restricted_shirt = 1) 
  (h_restricted_jacket : restricted_jacket = 1) : 
  ((trousers * shirts * jackets) - (restricted_shirt * restricted_jacket * trousers) = 115) := 
  by 
    sorry

end roberto_outfits_l37_37928


namespace expression_value_l37_37580

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l37_37580


namespace person_A_money_left_l37_37775

-- We define the conditions and question in terms of Lean types.
def initial_money_ratio : ℚ := 7 / 6
def money_spent_A : ℚ := 50
def money_spent_B : ℚ := 60
def final_money_ratio : ℚ := 3 / 2
def x : ℚ := 30

-- The theorem to prove the amount of money left by person A
theorem person_A_money_left 
  (init_ratio : initial_money_ratio = 7 / 6)
  (spend_A : money_spent_A = 50)
  (spend_B : money_spent_B = 60)
  (final_ratio : final_money_ratio = 3 / 2)
  (hx : x = 30) : 3 * x = 90 := by 
  sorry

end person_A_money_left_l37_37775


namespace percentage_25_of_200_l37_37960

def percentage_of (percent : ℝ) (amount : ℝ) : ℝ := percent * amount

theorem percentage_25_of_200 :
  percentage_of 0.25 200 = 50 :=
by sorry

end percentage_25_of_200_l37_37960


namespace number_of_friends_l37_37463

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37463


namespace total_male_students_combined_l37_37637

/-- The number of first-year students is 695, of which 329 are female students. 
If the number of male second-year students is 254, prove that the number of male students in the first-year and second-year combined is 620. -/
theorem total_male_students_combined (first_year_students : ℕ) (female_first_year_students : ℕ) (male_second_year_students : ℕ) :
  first_year_students = 695 →
  female_first_year_students = 329 →
  male_second_year_students = 254 →
  (first_year_students - female_first_year_students + male_second_year_students) = 620 := by
  sorry

end total_male_students_combined_l37_37637


namespace smallest_abundant_number_not_multiple_of_10_l37_37343

-- Definition of proper divisors of a number n
def properDivisors (n : ℕ) : List ℕ := 
  (List.range n).filter (λ d => d > 0 ∧ n % d = 0)

-- Definition of an abundant number
def isAbundant (n : ℕ) : Prop := 
  (properDivisors n).sum > n

-- Definition of not being a multiple of 10
def notMultipleOf10 (n : ℕ) : Prop := 
  n % 10 ≠ 0

-- Statement to prove
theorem smallest_abundant_number_not_multiple_of_10 :
  ∃ n, isAbundant n ∧ notMultipleOf10 n ∧ ∀ m, (isAbundant m ∧ notMultipleOf10 m) → n ≤ m :=
by
  sorry

end smallest_abundant_number_not_multiple_of_10_l37_37343


namespace Jungkook_has_bigger_number_l37_37913

theorem Jungkook_has_bigger_number : (3 + 6) > 4 :=
by {
  sorry
}

end Jungkook_has_bigger_number_l37_37913


namespace cans_in_each_package_of_cat_food_l37_37846

-- Definitions and conditions
def cans_per_package_cat (c : ℕ) := 9 * c
def cans_per_package_dog := 7 * 5
def extra_cans_cat := 55

-- Theorem stating the problem and the answer
theorem cans_in_each_package_of_cat_food (c : ℕ) (h: cans_per_package_cat c = cans_per_package_dog + extra_cans_cat) :
  c = 10 :=
sorry

end cans_in_each_package_of_cat_food_l37_37846


namespace value_this_year_l37_37510

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end value_this_year_l37_37510


namespace reassemble_into_square_conditions_l37_37667

noncomputable def graph_paper_figure : Type := sorry
noncomputable def is_cuttable_into_parts (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def all_parts_are_triangles (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def can_reassemble_to_square (figure : graph_paper_figure) : Prop := sorry

theorem reassemble_into_square_conditions :
  ∀ (figure : graph_paper_figure), 
  (is_cuttable_into_parts figure 4 ∧ can_reassemble_to_square figure) ∧ 
  (is_cuttable_into_parts figure 5 ∧ all_parts_are_triangles figure 5 ∧ can_reassemble_to_square figure) :=
sorry

end reassemble_into_square_conditions_l37_37667


namespace find_number_l37_37873

theorem find_number (x : ℝ) (h : 42 - 3 * x = 12) : x = 10 := 
by 
  sorry

end find_number_l37_37873


namespace probability_win_all_games_l37_37491

variable (p : ℚ) (n : ℕ)

-- Define the conditions
def probability_of_winning := p = 2 / 3
def number_of_games := n = 6
def independent_games := true

-- The theorem we want to prove
theorem probability_win_all_games (h₁ : probability_of_winning p)
                                   (h₂ : number_of_games n)
                                   (h₃ : independent_games) :
  p^n = 64 / 729 :=
sorry

end probability_win_all_games_l37_37491


namespace firstSuperbSunday_is_February_29_2020_l37_37835

/-- Define the concept of "Superb Sunday." A month with a Superb Sunday has five Sundays. -/
def isSuperbSundayMonth (year month : ℕ) : Prop :=
  (Nat.countp (λ d, d.weekday = 6) (finset.range (DateTime.DaysInMonth year month)) = 5)

/-- A company's fiscal year starts on January 13, 2020. -/
def fiscal_start : DateTime := ⟨2020, 1, 13⟩

/-- Calculate the date of the first Superb Sunday after January 13, 2020. -/
noncomputable 
def firstSuperbSundayAfterFiscalStart : DateTime :=
  DateTime.next (λ d, d > fiscal_start ∧ isSuperbSundayMonth d.year d.month) fiscal_start

theorem firstSuperbSunday_is_February_29_2020 :
  firstSuperbSundayAfterFiscalStart = ⟨2020, 2, 29⟩ :=
sorry

end firstSuperbSunday_is_February_29_2020_l37_37835


namespace shelves_filled_l37_37630

theorem shelves_filled (total_teddy_bears teddy_bears_per_shelf : ℕ) (h1 : total_teddy_bears = 98) (h2 : teddy_bears_per_shelf = 7) : 
  total_teddy_bears / teddy_bears_per_shelf = 14 := 
by 
  sorry

end shelves_filled_l37_37630


namespace perfect_square_proof_l37_37090

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l37_37090


namespace loraine_wax_usage_l37_37281

/-
Loraine makes wax sculptures of animals. Large animals take eight sticks of wax, medium animals take five sticks, and small animals take three sticks.
She made twice as many small animals as large animals, and four times as many medium animals as large animals. She used 36 sticks of wax for small animals.
Prove that Loraine used 204 sticks of wax to make all the animals.
-/

theorem loraine_wax_usage :
  ∃ (L M S : ℕ), (S = 2 * L) ∧ (M = 4 * L) ∧ (3 * S = 36) ∧ (8 * L + 5 * M + 3 * S = 204) :=
by {
  sorry
}

end loraine_wax_usage_l37_37281


namespace paolo_sevilla_birthday_l37_37482

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37482


namespace largest_x_value_l37_37698

theorem largest_x_value
  (x : ℝ)
  (h : (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x = 8 * x - 2)
  : x = 5 / 3 :=
sorry

end largest_x_value_l37_37698


namespace perfect_square_condition_l37_37088

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l37_37088


namespace cost_of_painting_cube_l37_37123

-- Definitions for conditions
def cost_per_kg : ℝ := 36.50
def coverage_per_kg : ℝ := 16  -- square feet
def side_length : ℝ := 8       -- feet

-- Derived constants
def area_per_face : ℝ := side_length * side_length
def number_of_faces : ℝ := 6
def total_surface_area : ℝ := number_of_faces * area_per_face
def paint_required : ℝ := total_surface_area / coverage_per_kg
def total_cost : ℝ := paint_required * cost_per_kg

-- Theorem statement
theorem cost_of_painting_cube : total_cost = 876 := by
  sorry

end cost_of_painting_cube_l37_37123


namespace eggs_in_larger_omelette_l37_37003

theorem eggs_in_larger_omelette :
  ∀ (total_eggs : ℕ) (orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette : ℕ),
    total_eggs = 84 →
    orders_3_eggs_first_hour = 5 →
    orders_3_eggs_third_hour = 3 →
    orders_large_eggs_second_hour = 7 →
    orders_large_eggs_last_hour = 8 →
    num_eggs_per_3_omelette = 3 →
    (total_eggs - (orders_3_eggs_first_hour * num_eggs_per_3_omelette + orders_3_eggs_third_hour * num_eggs_per_3_omelette)) / (orders_large_eggs_second_hour + orders_large_eggs_last_hour) = 4 :=
by
  intros total_eggs orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette
  sorry

end eggs_in_larger_omelette_l37_37003


namespace required_extra_money_l37_37689

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l37_37689


namespace quadratic_solution_unique_l37_37489

theorem quadratic_solution_unique (b : ℝ) (hb : b ≠ 0) (hdisc : 30 * 30 - 4 * b * 10 = 0) :
  ∃ x : ℝ, bx ^ 2 + 30 * x + 10 = 0 ∧ x = -2 / 3 :=
by
  sorry

end quadratic_solution_unique_l37_37489


namespace complement_of_A_union_B_in_U_l37_37591

def U : Set ℝ := { x | -5 < x ∧ x < 5 }
def A : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = { x | -5 < x ∧ x ≤ -2 } := by
  sorry

end complement_of_A_union_B_in_U_l37_37591


namespace cost_of_fencing_per_meter_l37_37496

theorem cost_of_fencing_per_meter
  (breadth : ℝ)
  (length : ℝ)
  (cost : ℝ)
  (length_eq : length = breadth + 40)
  (total_cost : cost = 5300)
  (length_given : length = 70) :
  cost / (2 * length + 2 * breadth) = 26.5 :=
by
  sorry

end cost_of_fencing_per_meter_l37_37496


namespace still_need_more_volunteers_l37_37283

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers_l37_37283


namespace non_black_cows_l37_37783

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l37_37783


namespace tan_equality_condition_l37_37718

open Real

theorem tan_equality_condition (α β : ℝ) :
  (α = β) ↔ (tan α = tan β) :=
sorry

end tan_equality_condition_l37_37718


namespace find_scalars_l37_37433

def M : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 7], ![-3, -1]]
def M_squared : Matrix (Fin 2) (Fin 2) ℤ := ![![-17, 7], ![-3, -20]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem find_scalars :
  ∃ p q : ℤ, M_squared = p • M + q • I ∧ (p, q) = (1, -19) := sorry

end find_scalars_l37_37433


namespace equation_squares_l37_37085

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l37_37085


namespace sin_lg_roots_l37_37940

theorem sin_lg_roots (f : ℝ → ℝ) (g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) (h₂ : ∀ x, g x = Real.log x)
  (domain : ∀ x, x > 0 → x < 10) (h₃ : ∀ x, f x ≤ 1 ∧ g x ≤ 1) :
  ∃ x1 x2 x3, (0 < x1 ∧ x1 < 10) ∧ (f x1 = g x1) ∧
               (0 < x2 ∧ x2 < 10) ∧ (f x2 = g x2) ∧
               (0 < x3 ∧ x3 < 10) ∧ (f x3 = g x3) ∧
               x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
by
  sorry

end sin_lg_roots_l37_37940


namespace cost_difference_proof_l37_37259

noncomputable def sailboat_daily_rent : ℕ := 60
noncomputable def ski_boat_hourly_rent : ℕ := 80
noncomputable def sailboat_hourly_fuel_cost : ℕ := 10
noncomputable def ski_boat_hourly_fuel_cost : ℕ := 20
noncomputable def discount : ℕ := 10

noncomputable def rent_time : ℕ := 3
noncomputable def rent_days : ℕ := 2

noncomputable def ken_sailboat_rent_cost :=
  sailboat_daily_rent * rent_days - sailboat_daily_rent * discount / 100

noncomputable def ken_sailboat_fuel_cost :=
  sailboat_hourly_fuel_cost * rent_time * rent_days

noncomputable def ken_total_cost :=
  ken_sailboat_rent_cost + ken_sailboat_fuel_cost

noncomputable def aldrich_ski_boat_rent_cost :=
  ski_boat_hourly_rent * rent_time * rent_days - (ski_boat_hourly_rent * rent_time * discount / 100)

noncomputable def aldrich_ski_boat_fuel_cost :=
  ski_boat_hourly_fuel_cost * rent_time * rent_days

noncomputable def aldrich_total_cost :=
  aldrich_ski_boat_rent_cost + aldrich_ski_boat_fuel_cost

noncomputable def cost_difference :=
  aldrich_total_cost - ken_total_cost

theorem cost_difference_proof : cost_difference = 402 := by
  sorry

end cost_difference_proof_l37_37259


namespace find_x_value_l37_37025

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l37_37025


namespace area_bounded_region_l37_37494

theorem area_bounded_region :
  (∃ (x y : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) →
  ∃ (area : ℝ), area = 1250 :=
by
  sorry

end area_bounded_region_l37_37494


namespace faith_earnings_correct_l37_37560

variable (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) (overtime_hours_per_day : ℝ)
variable (overtime_rate_multiplier : ℝ)

def total_earnings (pay_per_hour : ℝ) (regular_hours_per_day : ℝ) (work_days_per_week : ℝ) 
                   (overtime_hours_per_day : ℝ) (overtime_rate_multiplier : ℝ) : ℝ :=
  let regular_hours := regular_hours_per_day * work_days_per_week
  let overtime_hours := overtime_hours_per_day * work_days_per_week
  let overtime_pay_rate := pay_per_hour * overtime_rate_multiplier
  let regular_earnings := pay_per_hour * regular_hours
  let overtime_earnings := overtime_pay_rate * overtime_hours
  regular_earnings + overtime_earnings

theorem faith_earnings_correct : 
  total_earnings 13.5 8 5 2 1.5 = 742.50 :=
by
  -- This is where the proof would go, but it's omitted as per the instructions
  sorry

end faith_earnings_correct_l37_37560


namespace length_B1C1_l37_37257

variable (AC BC : ℝ) (A1B1 : ℝ) (T : ℝ)

/-- Given a right triangle ABC with legs AC = 3 and BC = 4, and transformations
  of points to A1, B1, and C1 where A1B1 = 1 and angle B1 = 90 degrees,
  prove that the length of B1C1 is 12. -/
theorem length_B1C1 (h1 : AC = 3) (h2 : BC = 4) (h3 : A1B1 = 1) 
  (TABC : T = 6) (right_triangle_ABC : true) (right_triangle_A1B1C1 : true) : 
  B1C1 = 12 := 
sorry

end length_B1C1_l37_37257


namespace five_student_committees_from_eight_l37_37380

theorem five_student_committees_from_eight : nat.choose 8 5 = 56 := by
  sorry

end five_student_committees_from_eight_l37_37380


namespace simplify_fraction_product_l37_37486

theorem simplify_fraction_product :
  4 * (18 / 5) * (35 / -63) * (8 / 14) = - (32 / 7) :=
by sorry

end simplify_fraction_product_l37_37486


namespace total_rubber_bands_l37_37229

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end total_rubber_bands_l37_37229


namespace annual_savings_l37_37773

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

end annual_savings_l37_37773


namespace perfect_square_condition_l37_37087

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l37_37087


namespace lego_count_l37_37909

theorem lego_count 
  (total_legos : ℕ := 500)
  (used_legos : ℕ := total_legos / 2)
  (missing_legos : ℕ := 5) :
  total_legos - used_legos - missing_legos = 245 := 
sorry

end lego_count_l37_37909


namespace correct_statement_is_d_l37_37652

/-- A definition for all the conditions given in the problem --/
def very_small_real_form_set : Prop := false
def smallest_natural_number_is_one : Prop := false
def sets_equal : Prop := false
def empty_set_subset_of_any_set : Prop := true

/-- The main statement to be proven --/
theorem correct_statement_is_d : (very_small_real_form_set = false) ∧ 
                                 (smallest_natural_number_is_one = false) ∧ 
                                 (sets_equal = false) ∧ 
                                 (empty_set_subset_of_any_set = true) :=
by
  sorry

end correct_statement_is_d_l37_37652


namespace min_value_x_y_xy_l37_37588

theorem min_value_x_y_xy (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  x + y + x * y ≥ -9 / 8 :=
sorry

end min_value_x_y_xy_l37_37588


namespace solve_for_q_l37_37065

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by {
  sorry
}

end solve_for_q_l37_37065


namespace find_k_l37_37029

theorem find_k (k : ℕ) (hk : 0 < k) (h : (k + 4) / (k^2 - 1) = 9 / 35) : k = 14 :=
by
  sorry

end find_k_l37_37029


namespace not_suitable_for_storing_l37_37801

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l37_37801


namespace investment_of_c_l37_37525

variable (P_a P_b P_c C_a C_b C_c : ℝ)

theorem investment_of_c (h1 : P_b = 3500) 
                        (h2 : P_a - P_c = 1399.9999999999998) 
                        (h3 : C_a = 8000) 
                        (h4 : C_b = 10000) 
                        (h5 : P_a / C_a = P_b / C_b) 
                        (h6 : P_c / C_c = P_b / C_b) : 
                        C_c = 40000 := 
by 
  sorry

end investment_of_c_l37_37525


namespace circle_tangent_to_line_at_parabola_focus_l37_37493

noncomputable def parabola_focus : (ℝ × ℝ) := (2, 0)

def line_eq (p : ℝ × ℝ) : Prop := p.2 = p.1

def circle_eq (center radius : ℝ) (p : ℝ × ℝ) : Prop := 
  (p.1 - center)^2 + p.2^2 = radius

theorem circle_tangent_to_line_at_parabola_focus : 
  ∀ p : ℝ × ℝ, (circle_eq 2 2 p ↔ (line_eq p ∧ p = parabola_focus)) := by
  sorry

end circle_tangent_to_line_at_parabola_focus_l37_37493


namespace smallest_possible_sum_l37_37202

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37202


namespace seq_a_general_term_seq_b_general_term_inequality_k_l37_37851

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n - 1

def S (n : ℕ) : ℕ := 
match n with
| 0       => 0
| (n + 1) => S n + seq_a (n + 1)

def seq_b (n : ℕ) : ℕ := 3 ^ n

def T (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2

theorem seq_a_general_term (n : ℕ) : seq_a n = if n = 1 then 2 else 2 * n - 1 :=
sorry

theorem seq_b_general_term (n : ℕ) : seq_b n = 3 ^ n :=
sorry

theorem inequality_k (k : ℝ) : (∀ n : ℕ, n > 0 → (T n + 3/2 : ℝ) * k ≥ 3 * n - 6) ↔ k ≥ 2 / 27 :=
sorry

end seq_a_general_term_seq_b_general_term_inequality_k_l37_37851


namespace ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l37_37094

-- Definitions based on the given conditions.
def total_students : ℕ := 25
def percent_girls : ℕ := 60
def percent_boys_like_bb : ℕ := 40
def percent_girls_like_bb : ℕ := 80

-- Results from those conditions.
def num_girls : ℕ := percent_girls * total_students / 100
def num_boys : ℕ := total_students - num_girls
def num_boys_like_bb : ℕ := percent_boys_like_bb * num_boys / 100
def num_boys_dont_like_bb : ℕ := num_boys - num_boys_like_bb
def num_girls_like_bb : ℕ := percent_girls_like_bb * num_girls / 100

-- Proof Problem Statement
theorem ratio_of_girls_who_like_bb_to_boys_dont_like_bb :
  (num_girls_like_bb : ℕ) / num_boys_dont_like_bb = 2 / 1 :=
by
  sorry

end ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l37_37094


namespace robins_total_pieces_of_gum_l37_37112

theorem robins_total_pieces_of_gum :
  let initial_packages := 27
  let pieces_per_initial_package := 18
  let additional_packages := 15
  let pieces_per_additional_package := 12
  let more_packages := 8
  let pieces_per_more_package := 25
  (initial_packages * pieces_per_initial_package) +
  (additional_packages * pieces_per_additional_package) +
  (more_packages * pieces_per_more_package) = 866 :=
by
  sorry

end robins_total_pieces_of_gum_l37_37112


namespace negation_of_P_is_true_l37_37926

theorem negation_of_P_is_true :
  ¬ (∃ x : ℝ, x^2 + 1 < 2 * x) :=
by sorry

end negation_of_P_is_true_l37_37926


namespace intersection_point_of_lines_l37_37615

theorem intersection_point_of_lines (n : ℕ) (x y : ℤ) :
  15 * x + 18 * y = 1005 ∧ y = n * x + 2 → n = 2 :=
by
  sorry

end intersection_point_of_lines_l37_37615


namespace problem_statement_l37_37896

-- Problem statement in Lean 4
theorem problem_statement (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b :=
by 
  sorry

end problem_statement_l37_37896


namespace man_walking_time_l37_37015

theorem man_walking_time (D V_w V_m T : ℝ) (t : ℝ) :
  D = V_w * T →
  D_w = V_m * t →
  D - V_m * t = V_w * (T - t) →
  T - (T - t) = 16 →
  t = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end man_walking_time_l37_37015


namespace area_of_shaded_region_l37_37697

open Real

noncomputable def line1 (x : ℝ) : ℝ := -3/10 * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -1.5 * x + 9

theorem area_of_shaded_region : 
  ∫ x in (2:ℝ)..6, (line2 x - line1 x) = 8 :=
by
  sorry

end area_of_shaded_region_l37_37697


namespace last_ball_probability_l37_37154

theorem last_ball_probability (w b : ℕ) (H : w > 0 ∨ b > 0) :
  (w % 2 = 1 → ∃ p : ℝ, p = 1 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) ∧ 
  (w % 2 = 0 → ∃ p : ℝ, p = 0 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) :=
by sorry

end last_ball_probability_l37_37154


namespace alberto_bjorn_distance_difference_l37_37076

-- Definitions based on given conditions
def alberto_speed : ℕ := 12  -- miles per hour
def bjorn_speed : ℕ := 10    -- miles per hour
def total_time : ℕ := 6      -- hours
def bjorn_rest_time : ℕ := 1 -- hours

def alberto_distance : ℕ := alberto_speed * total_time
def bjorn_distance : ℕ := bjorn_speed * (total_time - bjorn_rest_time)

-- The statement to prove
theorem alberto_bjorn_distance_difference :
  (alberto_distance - bjorn_distance) = 22 :=
by
  sorry

end alberto_bjorn_distance_difference_l37_37076


namespace birthday_friends_count_l37_37457

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37457


namespace tan_difference_identity_l37_37233

theorem tan_difference_identity (a b : ℝ) (h1 : Real.tan a = 2) (h2 : Real.tan b = 3 / 4) :
  Real.tan (a - b) = 1 / 2 :=
sorry

end tan_difference_identity_l37_37233


namespace sqrt_x2y_neg_x_sqrt_y_l37_37598

variables {x y : ℝ} (h : x * y < 0)

theorem sqrt_x2y_neg_x_sqrt_y (h : x * y < 0): real.sqrt (x ^ 2 * y) = -x * real.sqrt y :=
sorry

end sqrt_x2y_neg_x_sqrt_y_l37_37598


namespace find_xy_yz_xz_l37_37324

noncomputable def xy_yz_xz (x y z : ℝ) : ℝ := x * y + y * z + x * z

theorem find_xy_yz_xz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x^2 + x * y + y^2 = 48) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + x * z + x^2 = 64) :
  xy_yz_xz x y z = 32 :=
sorry

end find_xy_yz_xz_l37_37324


namespace average_fuel_consumption_correct_l37_37962

def distance_to_x : ℕ := 150
def distance_to_y : ℕ := 220
def fuel_to_x : ℕ := 20
def fuel_to_y : ℕ := 15

def total_distance : ℕ := distance_to_x + distance_to_y
def total_fuel_used : ℕ := fuel_to_x + fuel_to_y
def avg_fuel_consumption : ℚ := total_fuel_used / total_distance

theorem average_fuel_consumption_correct :
  avg_fuel_consumption = 0.0946 := by
  sorry

end average_fuel_consumption_correct_l37_37962


namespace difference_of_squares_example_l37_37312

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 305) (h2 : b = 295) :
  (a^2 - b^2) / 10 = 600 :=
by
  sorry

end difference_of_squares_example_l37_37312


namespace parabola_vertex_example_l37_37937

-- Definitions based on conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def vertex (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + 3

-- Conditions given in the problem
def condition1 (a b c : ℝ) : Prop := parabola a b c 2 = 5
def condition2 (a : ℝ) : Prop := vertex a 1 = 3

-- Goal statement to be proved
theorem parabola_vertex_example : ∃ (a b c : ℝ), 
  condition1 a b c ∧ condition2 a ∧ a - b + c = 11 :=
by
  sorry

end parabola_vertex_example_l37_37937


namespace fraction_product_l37_37141

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l37_37141


namespace F_equiv_A_l37_37717

-- Define the function F
def F : ℝ → ℝ := sorry

-- Given condition
axiom F_property (x : ℝ) : F ((1 - x) / (1 + x)) = x

-- The theorem that needs to be proved
theorem F_equiv_A (x : ℝ) : F (-2 - x) = -2 - F x := sorry

end F_equiv_A_l37_37717


namespace louis_current_age_l37_37418

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l37_37418


namespace x_is_integer_if_conditions_hold_l37_37400

theorem x_is_integer_if_conditions_hold (x : ℝ)
  (h1 : ∃ (k : ℤ), x^2 - x = k)
  (h2 : ∃ (n : ℕ), n ≥ 3 ∧ ∃ (m : ℤ), x^n - x = m) :
  ∃ (z : ℤ), x = z :=
sorry

end x_is_integer_if_conditions_hold_l37_37400


namespace smallest_sum_l37_37213

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37213


namespace smallest_x_plus_y_l37_37216

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37216


namespace sqrt_7_estimate_l37_37703

theorem sqrt_7_estimate : (2 : Real) < Real.sqrt 7 ∧ Real.sqrt 7 < 3 → (Real.sqrt 7 - 1) / 2 < 1 := 
by
  intro h
  sorry

end sqrt_7_estimate_l37_37703


namespace jack_buttons_total_l37_37424

theorem jack_buttons_total :
  (3 * 3) * 7 = 63 :=
by
  sorry

end jack_buttons_total_l37_37424


namespace friends_attended_birthday_l37_37444

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37444


namespace problem1_problem2_l37_37392

def f (x : ℝ) : ℝ := |x - 3|

theorem problem1 :
  {x : ℝ | f x < 2 + |x + 1|} = {x : ℝ | 0 < x} := sorry

theorem problem2 (m n : ℝ) (h_mn : m > 0) (h_nn : n > 0) (h : (1 / m) + (1 / n) = 2 * m * n) :
  m * f n + n * f (-m) ≥ 6 := sorry

end problem1_problem2_l37_37392


namespace jane_total_investment_in_stocks_l37_37910

-- Definitions
def total_investment := 220000
def bonds_investment := 13750
def stocks_investment := 5 * bonds_investment
def mutual_funds_investment := 2 * stocks_investment

-- Condition: The total amount invested
def total_investment_condition : Prop := 
  bonds_investment + stocks_investment + mutual_funds_investment = total_investment

-- Theorem: Jane's total investment in stocks
theorem jane_total_investment_in_stocks :
  total_investment_condition →
  stocks_investment = 68750 :=
by sorry

end jane_total_investment_in_stocks_l37_37910


namespace sum_of_solutions_l37_37370

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l37_37370


namespace sum_of_squares_of_roots_l37_37035

-- Define the roots of the quadratic equation
def roots (a b c : ℝ) := { x : ℝ | a * x^2 + b * x + c = 0 }

-- The given quadratic equation is x^2 - 3x - 1 = 0
lemma quadratic_roots_property :
  ∀ x ∈ roots 1 (-3) (-1), x^2 - 3 * x - 1 = 0 :=
by {
  intros x hx,
  unfold roots at hx,
  exact hx,
  sorry
}

-- Using Vieta's formulas and properties of quadratic equations
theorem sum_of_squares_of_roots :
  let x1 := Classical.choose (exists (λ x, roots 1 (-3) (-1) x)),
      x2 := Classical.choose (exists ! (λ x, roots 1 (-3) (-1) x)),
  in x1^2 + x2^2 = 11 :=
by {
  let x1 := 3 / 2 + sqrt 13 / 2,
  let x2 := 3 / 2 - sqrt 13 / 2,
  have h1 : x1 + x2 = 3 := by {
    rw [← add_sub_assoc, add_sub_cancel, div_add_div_same],
    norm_num,
    sorry
  },
  have h2 : x1 * x2 = -1 := by {
    -- Similar proof under Classical logic, left as sorry for brevity
    sorry
  },
  calc
    x1^2 + x2^2 = (x1 + x2)^2 - 2 * (x1 * x2) : by norm_num; field_simp
            ... = 3^2 - 2 * (-1) : by rw [h1, h2]
            ... = 9 + 2 : by norm_num
            ... = 11 : by norm_num
  sorry
}

end sum_of_squares_of_roots_l37_37035


namespace find_x_l37_37314

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l37_37314


namespace find_x_l37_37313

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end find_x_l37_37313


namespace savings_fraction_l37_37174

variable (P : ℝ) -- worker's monthly take-home pay, assumed to be a real number
variable (f : ℝ) -- fraction of the take-home pay that she saves each month, assumed to be a real number

-- Condition: 12 times the fraction saved monthly should equal 8 times the amount not saved monthly.
axiom condition : 12 * f * P = 8 * (1 - f) * P

-- Prove: the fraction saved each month is 2/5
theorem savings_fraction : f = 2 / 5 := 
by
  sorry

end savings_fraction_l37_37174


namespace striped_octopus_has_8_legs_l37_37067

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l37_37067


namespace find_n_l37_37767

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end find_n_l37_37767


namespace number_of_roses_l37_37804

theorem number_of_roses 
  (R L T : ℕ)
  (h1 : R + L + T = 100)
  (h2 : R = L + 22)
  (h3 : R = T - 20) : R = 34 := 
sorry

end number_of_roses_l37_37804


namespace colored_ints_square_diff_l37_37678

-- Define a coloring function c as a total function from ℤ to a finite set {0, 1, 2}
def c : ℤ → Fin 3 := sorry

-- Lean 4 statement for the problem
theorem colored_ints_square_diff : 
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ ∃ k : ℤ, a - b = k ^ 2 :=
sorry

end colored_ints_square_diff_l37_37678


namespace compute_value_of_expression_l37_37765

theorem compute_value_of_expression (p q : ℝ) (h₁ : 3 * p ^ 2 - 5 * p - 12 = 0) (h₂ : 3 * q ^ 2 - 5 * q - 12 = 0) :
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 :=
by
  sorry

end compute_value_of_expression_l37_37765


namespace proof_allison_brian_noah_l37_37551

-- Definitions based on the problem conditions

-- Definition for the cubes
def allison_cube := [6, 6, 6, 6, 6, 6]
def brian_cube := [1, 2, 2, 3, 3, 4]
def noah_cube := [3, 3, 3, 3, 5, 5]

-- Helper function to calculate the probability of succeeding conditions
def probability_succeeding (A B C : List ℕ) : ℚ :=
  if (A.all (λ x => x = 6)) ∧ (B.all (λ x => x ≤ 5)) ∧ (C.all (λ x => x ≤ 5)) then 1 else 0

-- Define the proof statement for the given problem
theorem proof_allison_brian_noah :
  probability_succeeding allison_cube brian_cube noah_cube = 1 :=
by
  -- Since all conditions fulfill the requirement, we'll use sorry to skip the proof for now
  sorry

end proof_allison_brian_noah_l37_37551


namespace sum_of_solutions_l37_37371

theorem sum_of_solutions (y : ℝ) (h : y + 16 / y = 12) : y = 4 ∨ y = 8 → 4 + 8 = 12 :=
by sorry

end sum_of_solutions_l37_37371


namespace calculation_A_B_l37_37301

theorem calculation_A_B :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 :=
by
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  sorry

end calculation_A_B_l37_37301


namespace inequality_solution_sum_of_squares_geq_sum_of_products_l37_37328

-- Problem 1
theorem inequality_solution (x : ℝ) : (0 < x ∧ x < 2/3) ↔ (x + 2) / (2 - 3 * x) > 1 :=
by
  sorry

-- Problem 2
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end inequality_solution_sum_of_squares_geq_sum_of_products_l37_37328


namespace min_y_value_l37_37855

theorem min_y_value (x : ℝ) : 
  ∃ y : ℝ, y = 4 * x^2 + 8 * x + 12 ∧ ∀ z, (z = 4 * x^2 + 8 * x + 12) → y ≤ z := sorry

end min_y_value_l37_37855


namespace sin_solution_set_l37_37504

open Real

theorem sin_solution_set (x : ℝ) : 
  (3 * sin x = 1 + cos (2 * x)) ↔ ∃ k : ℤ, x = k * π + (-1) ^ k * (π / 6) :=
by
  sorry

end sin_solution_set_l37_37504


namespace isosceles_triangle_legs_length_l37_37601

theorem isosceles_triangle_legs_length 
  (P : ℝ) (base : ℝ) (leg_length : ℝ) 
  (hp : P = 26) 
  (hb : base = 11) 
  (hP : P = 2 * leg_length + base) : 
  leg_length = 7.5 := 
by 
  sorry

end isosceles_triangle_legs_length_l37_37601


namespace color_block_prob_l37_37972

-- Definitions of the problem's conditions
def colors : List (List String) := [
    ["red", "blue", "yellow", "green"],
    ["red", "blue", "yellow", "white"]
]

-- The events in which at least one box receives 3 blocks of the same color
def event_prob : ℚ := 3 / 64

-- Tuple as a statement to prove in Lean
theorem color_block_prob (m n : ℕ) (h : m + n = 67) : 
  ∃ (m n : ℕ), (m / n : ℚ) = event_prob := 
by
  use 3
  use 64
  simp
  sorry

end color_block_prob_l37_37972


namespace distinct_sequences_count_l37_37595

def letters := ["E", "Q", "U", "A", "L", "S"]

noncomputable def count_sequences : Nat :=
  let remaining_letters := ["E", "Q", "U", "A"] -- 'L' and 'S' are already considered
  3 * (4 * 3) -- as analyzed: (LS__) + (L_S_) + (L__S)

theorem distinct_sequences_count : count_sequences = 36 := 
  by
    unfold count_sequences
    sorry

end distinct_sequences_count_l37_37595


namespace part1_part2_l37_37758

open Real

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * cos α, sqrt 10 * sin α)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 6 = 0

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ + 2 * ρ * sin θ - 12 = 0

theorem part1 (α : ℝ) : ∃ ρ θ : ℝ, curve_polar ρ θ :=
  sorry

theorem part2 : ∃ ρ1 ρ2 : ℝ, curve_polar ρ1 (π / 4) ∧ line_polar ρ2 (π / 4) ∧ abs (ρ1 - ρ2) = sqrt 2 :=
  sorry

end part1_part2_l37_37758


namespace find_k_for_line_l37_37872

theorem find_k_for_line : 
  ∃ k : ℚ, (∀ x y : ℚ, (-1 / 3 - 3 * k * x = 4 * y) ∧ (x = 1 / 3) ∧ (y = -8)) → k = 95 / 3 :=
by
  sorry

end find_k_for_line_l37_37872


namespace units_digit_of_expression_l37_37140

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

def expr : ℕ := 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

theorem units_digit_of_expression : units_digit expr = 6 :=
by
  sorry

end units_digit_of_expression_l37_37140


namespace combined_work_rate_l37_37824

def work_done_in_one_day (A B : ℕ) (work_to_days : ℕ -> ℕ) : ℚ :=
  (work_to_days A + work_to_days B)

theorem combined_work_rate (A : ℕ) (B : ℕ) (work_to_days : ℕ -> ℕ) :
  work_to_days A = 1/18 ∧ work_to_days B = 1/9 → work_done_in_one_day A B (work_to_days) = 1/6 :=
by
  sorry

end combined_work_rate_l37_37824


namespace num_perfect_square_21n_le_500_l37_37711

theorem num_perfect_square_21n_le_500 : 
  {n : ℕ | n ≤ 500 ∧ ∃ k : ℕ, 21 * n = k ^ 2}.to_finset.card = 4 := 
by sorry

end num_perfect_square_21n_le_500_l37_37711


namespace arithmetic_sequence_subtract_l37_37605

theorem arithmetic_sequence_subtract (a : ℕ → ℝ) (d : ℝ) :
  (a 4 + a 6 + a 8 + a 10 + a 12 = 120) →
  (a 9 - (1 / 3) * a 11 = 16) :=
by
  sorry

end arithmetic_sequence_subtract_l37_37605


namespace alex_buys_17p3_pounds_of_corn_l37_37355

noncomputable def pounds_of_corn (c b : ℝ) : Prop :=
    c + b = 30 ∧ 1.05 * c + 0.39 * b = 23.10

theorem alex_buys_17p3_pounds_of_corn :
    ∃ c b, pounds_of_corn c b ∧ c = 17.3 :=
by
    sorry

end alex_buys_17p3_pounds_of_corn_l37_37355


namespace one_third_sugar_amount_l37_37840

-- Define the original amount of sugar as a mixed number
def original_sugar_mixed : ℚ := 6 + 1 / 3

-- Define the fraction representing one-third of the recipe
def one_third : ℚ := 1 / 3

-- Define the expected amount of sugar for one-third of the recipe
def expected_sugar_mixed : ℚ := 2 + 1 / 9

-- The theorem stating the proof problem
theorem one_third_sugar_amount : (one_third * original_sugar_mixed) = expected_sugar_mixed :=
sorry

end one_third_sugar_amount_l37_37840


namespace base_b_square_l37_37011

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, k^2 = b^2 + 4 * b + 4 := 
by 
  sorry

end base_b_square_l37_37011


namespace dave_total_earnings_l37_37559

def hourly_wage (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 7 else
  if day = 2 then 9 else
  if day = 3 then 8 else 
  0

def hours_worked (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 2 else
  if day = 2 then 3 else
  if day = 3 then 5 else 
  0

def unpaid_break (day : ℕ) : ℝ :=
  if day = 0 then 0.5 else
  if day = 1 then 0.25 else
  if day = 2 then 0 else
  if day = 3 then 0.5 else 
  0

def daily_earnings (day : ℕ) : ℝ :=
  (hours_worked day - unpaid_break day) * hourly_wage day

def net_earnings (day : ℕ) : ℝ :=
  daily_earnings day - (daily_earnings day * 0.1)

def total_net_earnings : ℝ :=
  net_earnings 0 + net_earnings 1 + net_earnings 2 + net_earnings 3

theorem dave_total_earnings : total_net_earnings = 97.43 := by
  sorry

end dave_total_earnings_l37_37559


namespace parabola_focus_distance_l37_37587

theorem parabola_focus_distance (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) 
  (h₁ : y₀^2 = 2 * p * 4) 
  (h₂ : dist (4, y₀) (p/2, 0) = 3/2 * p) : 
  p = 4 := 
sorry

end parabola_focus_distance_l37_37587


namespace proof_problem_l37_37578

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l37_37578


namespace rectangles_divided_into_13_squares_l37_37985

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l37_37985


namespace chocolates_vs_gums_l37_37272

theorem chocolates_vs_gums 
    (c g : ℝ) 
    (Kolya_claim : 2 * c > 5 * g) 
    (Sasha_claim : ¬ ( 3 * c > 8 * g )) : 
    7 * c ≤ 19 * g := 
sorry

end chocolates_vs_gums_l37_37272


namespace traveling_distance_l37_37149

/-- Let D be the total distance from the dormitory to the city in kilometers.
Given the following conditions:
1. The student traveled 1/3 of the way by foot.
2. The student traveled 3/5 of the way by bus.
3. The remaining portion of the journey was covered by car, which equals 2 kilometers.
We need to prove that the total distance D is 30 kilometers. -/ 
theorem traveling_distance (D : ℕ) 
  (h1 : (1 / 3 : ℚ) * D + (3 / 5 : ℚ) * D + 2 = D) : D = 30 := 
sorry

end traveling_distance_l37_37149


namespace condition_inequality_l37_37728

theorem condition_inequality (x y : ℝ) :
  (¬ (x ≤ y → |x| ≤ |y|)) ∧ (¬ (|x| ≤ |y| → x ≤ y)) :=
by
  sorry

end condition_inequality_l37_37728


namespace henrietta_paint_needed_l37_37056

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l37_37056


namespace friends_at_birthday_l37_37470

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37470


namespace domain_transformation_l37_37389

variable {α : Type*}
variable {f : α → α}
variable {x y : α}
variable (h₁ : ∀ x, -1 < x ∧ x < 1)

theorem domain_transformation (h₁ : ∀ x, -1 < x ∧ x < 1) : ∀ x, 0 < x ∧ x < 1 →
  ((-1 < (2 * x - 1) ∧ (2 * x - 1) < 1)) :=
by
  intro x
  intro h
  have h₂ : -1 < 2 * x - 1 := sorry
  have h₃ : 2 * x - 1 < 1 := sorry
  exact ⟨h₂, h₃⟩

end domain_transformation_l37_37389


namespace least_number_of_roots_l37_37837

variable {g : ℝ → ℝ}

-- Conditions
axiom g_defined (x : ℝ) : g x = g x
axiom g_symmetry_1 (x : ℝ) : g (3 + x) = g (3 - x)
axiom g_symmetry_2 (x : ℝ) : g (5 + x) = g (5 - x)
axiom g_at_1 : g 1 = 0

-- Root count in the interval
theorem least_number_of_roots : ∃ (n : ℕ), n >= 250 ∧ (∀ m, -1000 ≤ (1 + 8 * m:ℝ) ∧ (1 + 8 * m:ℝ) ≤ 1000 → g (1 + 8 * m) = 0) :=
sorry

end least_number_of_roots_l37_37837


namespace triangle_area_calculation_l37_37920

theorem triangle_area_calculation
  (A : ℕ)
  (BC : ℕ)
  (h : ℕ)
  (nine_parallel_lines : Bool)
  (equal_segments : Bool)
  (largest_area_part : ℕ)
  (largest_part_condition : largest_area_part = 38) :
  9 * (BC / 10) * (h / 10) / 2 = 10 * (BC / 2) * A / 19 :=
sorry

end triangle_area_calculation_l37_37920


namespace birthday_friends_count_l37_37475

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37475


namespace rectangle_divided_into_13_squares_l37_37986

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l37_37986


namespace correct_option_D_l37_37894

variables {a b m : Type}
variables {α β : Type}

axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

variables (a_parallel_b : parallel a b)
variables (a_parallel_alpha : parallel a α)

variables (alpha_perpendicular_beta : perpendicular α β)
variables (a_parallel_alpha : parallel a α)

variables (alpha_parallel_beta : parallel α β)
variables (m_perpendicular_alpha : perpendicular m α)

theorem correct_option_D : parallel α β ∧ perpendicular m α → perpendicular m β := sorry

end correct_option_D_l37_37894


namespace log7_18_l37_37061

theorem log7_18 (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) :=
by
  -- proof to be completed
  sorry

end log7_18_l37_37061


namespace proof_problem_l37_37045

-- Conditions
def in_fourth_quadrant (α : ℝ) : Prop := (α > 3 * Real.pi / 2) ∧ (α < 2 * Real.pi)
def x_coordinate_unit_circle (α : ℝ) : Prop := Real.cos α = 1/3

-- Proof statement
theorem proof_problem (α : ℝ) (h1 : in_fourth_quadrant α) (h2 : x_coordinate_unit_circle α) :
  Real.tan α = -2 * Real.sqrt 2 ∧
  ((Real.sin α)^2 - Real.sqrt 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.cos α)^2) = 6 / 5 :=
by
  sorry

end proof_problem_l37_37045


namespace construct_convex_hexagon_l37_37976

-- Definitions of the sides and their lengths
variables {A B C D E F : Type} -- Points of the hexagon
variables {AB BC CD DE EF FA : ℝ}  -- Lengths of the sides
variables (convex_hexagon : Prop) -- the hexagon is convex

-- Hypotheses of parallel and equal opposite sides
variables (H_AB_DE : AB = DE)
variables (H_BC_EF : BC = EF)
variables (H_CD_AF : CD = AF)

-- Define the construction of the hexagon under the given conditions
theorem construct_convex_hexagon
  (convex_hexagon : Prop)
  (H_AB_DE : AB = DE)
  (H_BC_EF : BC = EF)
  (H_CD_AF : CD = AF) : 
  ∃ (A B C D E F : Type), 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧ convex_hexagon ∧ 
    (AB = FA) ∧ (AF = CD) ∧ (BC = EF) ∧ (AB = DE) := 
sorry -- Proof omitted

end construct_convex_hexagon_l37_37976


namespace louis_age_currently_31_l37_37412

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l37_37412


namespace decimal_expansion_of_13_over_625_l37_37372

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end decimal_expansion_of_13_over_625_l37_37372


namespace Pau_total_fried_chicken_l37_37095

theorem Pau_total_fried_chicken :
  ∀ (kobe_order : ℕ),
  (pau_initial : ℕ) (pau_second : ℕ),
  kobe_order = 5 →
  pau_initial = 2 * kobe_order →
  pau_second = pau_initial →
  pau_initial + pau_second = 20 :=
by
  intros kobe_order pau_initial pau_second
  sorry

end Pau_total_fried_chicken_l37_37095


namespace product_of_solutions_l37_37699

theorem product_of_solutions :
  let a := 2
  let b := 4
  let c := -6
  let discriminant := b^2 - 4*a*c
  ∃ (x₁ x₂ : ℝ), 2*x₁^2 + 4*x₁ - 6 = 0 ∧ 2*x₂^2 + 4*x₂ - 6 = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -3 :=
sorry

end product_of_solutions_l37_37699


namespace carols_rectangle_length_l37_37185

theorem carols_rectangle_length :
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  carol_length = 5 :=
by
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  show carol_length = 5
  sorry

end carols_rectangle_length_l37_37185


namespace tangent_line_at_point_l37_37633

def tangent_line_equation (f : ℝ → ℝ) (slope : ℝ) (p : ℝ × ℝ) :=
  ∃ (a b c : ℝ), a * p.1 + b * p.2 + c = 0 ∧ a = slope ∧ p.2 = f p.1

noncomputable def curve (x : ℝ) : ℝ := x^3 + x + 1

theorem tangent_line_at_point : 
  tangent_line_equation curve 4 (1, 3) :=
sorry

end tangent_line_at_point_l37_37633


namespace mean_of_remaining_four_numbers_l37_37632

theorem mean_of_remaining_four_numbers 
  (a b c d max_num : ℝ) 
  (h1 : max_num = 105) 
  (h2 : (a + b + c + d + max_num) / 5 = 92) : 
  (a + b + c + d) / 4 = 88.75 :=
by
  sorry

end mean_of_remaining_four_numbers_l37_37632


namespace find_x_l37_37309

-- Define the digits used
def digits : List ℕ := [1, 4, 5]

-- Define the sum of all four-digit numbers formed
def sum_of_digits (x : ℕ) : ℕ :=
  24 * (1 + 4 + 5 + x)

-- State the theorem
theorem find_x (x : ℕ) (h : sum_of_digits x = 288) : x = 2 :=
  by
    sorry

end find_x_l37_37309


namespace MiaShots_l37_37282

theorem MiaShots (shots_game1_to_5 : ℕ) (total_shots_game1_to_5 : ℕ) (initial_avg : ℕ → ℕ → Prop)
  (shots_game6 : ℕ) (new_avg_shots : ℕ → ℕ → Prop) (total_shots : ℕ) (new_avg : ℕ): 
  shots_game1_to_5 = 20 →
  total_shots_game1_to_5 = 50 →
  initial_avg shots_game1_to_5 total_shots_game1_to_5 →
  shots_game6 = 15 →
  new_avg_shots 29 65 →
  total_shots = total_shots_game1_to_5 + shots_game6 →
  new_avg = 45 →
  (∃ shots_made_game6 : ℕ, shots_made_game6 = 29 - shots_game1_to_5 ∧ shots_made_game6 = 9) :=
by
  sorry

end MiaShots_l37_37282


namespace minimum_value_l37_37037

def minimum_value_problem (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : Prop :=
  ∃ c : ℝ, c = (1 / (a + 1) + 4 / (b + 1)) ∧ c = 9 / 4

theorem minimum_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : 
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 :=
by 
  -- Proof goes here
  sorry

end minimum_value_l37_37037


namespace largest_n_for_factorable_polynomial_l37_37568

theorem largest_n_for_factorable_polynomial :
  ∃ (n : ℤ), (∀ A B : ℤ, 7 * A * B = 56 → n ≤ 7 * B + A) ∧ n = 393 :=
by {
  sorry
}

end largest_n_for_factorable_polynomial_l37_37568


namespace calculate_expression_l37_37187

theorem calculate_expression : 6^3 - 5 * 7 + 2^4 = 197 := 
by
  -- Generally, we would provide the proof here, but it's not required.
  sorry

end calculate_expression_l37_37187


namespace longest_diagonal_of_rhombus_l37_37539

variables (d1 d2 : ℝ) (x : ℝ)
def rhombus_area := (d1 * d2) / 2
def diagonal_ratio := d1 / d2 = 4 / 3

theorem longest_diagonal_of_rhombus (h : rhombus_area (4 * x) (3 * x) = 150) (r : diagonal_ratio (4 * x) (3 * x)) : d1 = 20 := by
  sorry

end longest_diagonal_of_rhombus_l37_37539


namespace students_later_than_Yoongi_l37_37843

theorem students_later_than_Yoongi (total_students finished_before_Yoongi : ℕ) (h1 : total_students = 20) (h2 : finished_before_Yoongi = 11) :
  total_students - (finished_before_Yoongi + 1) = 8 :=
by {
  -- Proof is omitted as it's not required.
  sorry
}

end students_later_than_Yoongi_l37_37843


namespace striped_octopus_has_8_legs_l37_37068

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l37_37068


namespace total_surface_area_calc_l37_37836

/-- Given a cube with a total volume of 1 cubic foot, cut into four pieces by three parallel cuts:
1) The first cut is 0.4 feet from the top.
2) The second cut is 0.3 feet below the first.
3) The third cut is 0.1 feet below the second.
Prove that the total surface area of the new solid is 6 square feet. -/
theorem total_surface_area_calc :
  ∀ (A B C D : ℝ), 
    A = 0.4 → 
    B = 0.3 → 
    C = 0.1 → 
    D = 1 - (A + B + C) → 
    (6 : ℝ) = 6 := 
by 
  intros A B C D hA hB hC hD 
  sorry

end total_surface_area_calc_l37_37836


namespace total_vehicles_is_120_l37_37248

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l37_37248


namespace invitation_methods_l37_37639

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem invitation_methods (A B : Type) (students : Finset Type) (h : students.card = 10) :
  (∃ s : Finset Type, s.card = 6 ∧ A ∉ s ∧ B ∉ s) ∧ 
  (∃ t : Finset Type, t.card = 6 ∧ (A ∈ t ∨ B ∉ t)) →
  (combination 10 6 - combination 8 4 = 140) :=
by
  sorry

end invitation_methods_l37_37639


namespace smallest_n_l37_37961

-- Define the conditions as properties of integers
def connected (a b : ℕ): Prop := sorry -- Assume we have a definition for connectivity

def condition1 (a b n : ℕ) : Prop :=
  ¬connected a b → Nat.gcd (a^2 + b^2) n = 1

def condition2 (a b n : ℕ) : Prop :=
  connected a b → Nat.gcd (a^2 + b^2) n > 1

theorem smallest_n : ∃ n, n = 65 ∧ ∀ (a b : ℕ), condition1 a b n ∧ condition2 a b n := by
  sorry

end smallest_n_l37_37961


namespace diameter_is_twice_radius_l37_37230

theorem diameter_is_twice_radius {r d : ℝ} (h : d = 2 * r) : d = 2 * r :=
by {
  sorry
}

end diameter_is_twice_radius_l37_37230


namespace additional_width_is_25cm_l37_37951

-- Definitions
def length_of_room_cm := 5000
def width_of_room_cm := 1100
def additional_width_cm := 25
def number_of_tiles := 9000
def side_length_of_tile_cm := 25

-- Statement to prove
theorem additional_width_is_25cm : additional_width_cm = 25 :=
by
  -- The proof is omitted, we assume the proof steps here
  sorry

end additional_width_is_25cm_l37_37951


namespace birthday_friends_count_l37_37455

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37455


namespace a_divides_b_l37_37099

theorem a_divides_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b)
    (h : ∀ n : ℕ, a^n ∣ b^(n+1)) : a ∣ b :=
by
  sorry

end a_divides_b_l37_37099


namespace inscribed_square_area_after_cutting_l37_37874

theorem inscribed_square_area_after_cutting :
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  largest_inscribed_square_area = 9 :=
by
  let original_side := 5
  let cut_side := 1
  let remaining_side := original_side - 2 * cut_side
  let largest_inscribed_square_area := remaining_side ^ 2
  show largest_inscribed_square_area = 9
  sorry

end inscribed_square_area_after_cutting_l37_37874


namespace faith_weekly_earnings_l37_37563

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l37_37563


namespace henrietta_paint_needed_l37_37055

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l37_37055


namespace points_on_same_sphere_l37_37842

-- Define the necessary structures and assumptions
variables {P : Type*} [MetricSpace P]

-- Definitions of spheres and points
structure Sphere (P : Type*) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(positive_radius : 0 < radius)

def symmetric_point (S A1 : P) : P := sorry -- definition to get the symmetric point A2

-- Given conditions
variables (S A B C A1 B1 C1 A2 B2 C2 : P)
variable (omega : Sphere P)
variable (Omega : Sphere P)
variable (M_S_A : P) -- midpoint of SA
variable (M_S_B : P) -- midpoint of SB
variable (M_S_C : P) -- midpoint of SC

-- Assertions of conditions
axiom sphere_through_vertex : omega.center = S
axiom first_intersections : omega.radius = dist S A1 ∧ omega.radius = dist S B1 ∧ omega.radius = dist S C1
axiom omega_Omega_intersection : ∃ (circle_center : P) (plane_parallel_to_ABC : P), true-- some conditions indicating intersection
axiom symmetric_points_A1_A2 : A2 = symmetric_point S A1
axiom symmetric_points_B1_B2 : B2 = symmetric_point S B1
axiom symmetric_points_C1_C2 : C2 = symmetric_point S C1

-- The theorem to prove
theorem points_on_same_sphere : ∃ (sphere : Sphere P), 
  (dist sphere.center A) = sphere.radius ∧ 
  (dist sphere.center B) = sphere.radius ∧ 
  (dist sphere.center C) = sphere.radius ∧ 
  (dist sphere.center A2) = sphere.radius ∧ 
  (dist sphere.center B2) = sphere.radius ∧ 
  (dist sphere.center C2) = sphere.radius := 
sorry

end points_on_same_sphere_l37_37842


namespace louis_current_age_l37_37417

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l37_37417


namespace unique_solution_l37_37010

theorem unique_solution (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  12^x + 13^y - 14^z = 2013^t → (x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1) :=
by
  intros h
  sorry

end unique_solution_l37_37010


namespace greater_number_l37_37304

theorem greater_number (x y : ℕ) (h1 : x + y = 22) (h2 : x - y = 4) : x = 13 := 
by sorry

end greater_number_l37_37304


namespace chord_length_intercepted_by_line_on_curve_l37_37194

-- Define the curve and line from the problem
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Prove the length of the chord intercepted by the line on the curve is 4
theorem chord_length_intercepted_by_line_on_curve : 
  ∀ (x y : ℝ), curve x y → line x y → False := sorry

end chord_length_intercepted_by_line_on_curve_l37_37194


namespace perimeter_ratio_of_divided_square_l37_37253

theorem perimeter_ratio_of_divided_square
  (S_ΔADE : ℝ) (S_EDCB : ℝ)
  (S_ratio : S_ΔADE / S_EDCB = 5 / 19)
  : ∃ (perim_ΔADE perim_EDCB : ℝ),
  perim_ΔADE / perim_EDCB = 15 / 22 :=
by
  -- Let S_ΔADE = 5x and S_EDCB = 19x
  -- x can be calculated based on the given S_ratio = 5/19
  -- Apply geometric properties and simplifications analogous to the described solution.
  sorry

end perimeter_ratio_of_divided_square_l37_37253


namespace rectangle_MQ_l37_37903

theorem rectangle_MQ :
  ∀ (PQ QR PM MQ : ℝ),
    PQ = 4 →
    QR = 10 →
    PM = MQ →
    MQ = 2 * Real.sqrt 10 → 
    0 < MQ
:= by
  intros PQ QR PM MQ h1 h2 h3 h4
  sorry

end rectangle_MQ_l37_37903


namespace find_v_l37_37570

theorem find_v (v : ℝ) (h : (v - v / 3) - ((v - v / 3) / 3) = 4) : v = 9 := 
by 
  sorry

end find_v_l37_37570


namespace max_cities_l37_37672

def city (X : Type) := X

variable (A B C D E : Prop)

-- Conditions as given in the problem
axiom condition1 : A → B
axiom condition2 : D ∨ E
axiom condition3 : B ↔ ¬C
axiom condition4 : C ↔ D
axiom condition5 : E → (A ∧ D)

-- Proof problem: Given the conditions, prove that the maximum set of cities that can be visited is {C, D}
theorem max_cities (h1 : A → B) (h2 : D ∨ E) (h3 : B ↔ ¬C) (h4 : C ↔ D) (h5 : E → (A ∧ D)) : (C ∧ D) ∧ ¬A ∧ ¬B ∧ ¬E :=
by
  -- The core proof would use the constraints to show C and D, and exclude A, B, E
  sorry

end max_cities_l37_37672


namespace friends_attended_birthday_l37_37443

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37443


namespace yanni_money_left_in_cents_l37_37653

-- Define the constants based on the conditions
def initial_amount := 0.85
def mother_amount := 0.40
def found_amount := 0.50
def toy_cost := 1.60

-- Function to calculate the total amount
def total_amount := initial_amount + mother_amount + found_amount

-- Function to calculate the money left
def money_left := total_amount - toy_cost

-- Convert the remaining money from dollars to cents
def money_left_in_cents := money_left * 100

-- The theorem to prove
theorem yanni_money_left_in_cents : money_left_in_cents = 15 := by
  -- placeholder for proof, sorry used to skip the proof
  sorry

end yanni_money_left_in_cents_l37_37653


namespace total_pushups_l37_37523

def Zachary_pushups : ℕ := 44
def David_pushups : ℕ := Zachary_pushups + 58

theorem total_pushups : Zachary_pushups + David_pushups = 146 := by
  sorry

end total_pushups_l37_37523


namespace max_min_f_triangle_area_l37_37594

open Real

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (-2 * sin x, -1)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (-cos x, cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_min_f :
  (∀ x : ℝ, f x ≤ 2) ∧ (∀ x : ℝ, -2 ≤ f x) :=
sorry

theorem triangle_area
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h : A + B + C = π)
  (h_f_A : f A = 1)
  (b c : ℝ)
  (h_bc : b * c = 8) :
  (1 / 2) * b * c * sin A = 2 :=
sorry

end max_min_f_triangle_area_l37_37594


namespace total_vehicles_is_120_l37_37247

def num_trucks : ℕ := 20
def num_tanks : ℕ := 5 * num_trucks
def total_vehicles : ℕ := num_tanks + num_trucks

theorem total_vehicles_is_120 : total_vehicles = 120 :=
by
  sorry

end total_vehicles_is_120_l37_37247


namespace algorithm_characteristics_l37_37320

theorem algorithm_characteristics (finiteness : Prop) (definiteness : Prop) (output_capability : Prop) (unique : Prop) 
  (h1 : finiteness = true) 
  (h2 : definiteness = true) 
  (h3 : output_capability = true) 
  (h4 : unique = false) : 
  incorrect_statement = unique := 
by
  sorry

end algorithm_characteristics_l37_37320


namespace frogs_seen_in_pond_l37_37596

-- Definitions from the problem conditions
def initial_frogs_on_lily_pads : ℕ := 5
def frogs_on_logs : ℕ := 3
def baby_frogs_on_rock : ℕ := 2 * 12  -- Two dozen

-- The statement of the proof
theorem frogs_seen_in_pond : initial_frogs_on_lily_pads + frogs_on_logs + baby_frogs_on_rock = 32 :=
by sorry

end frogs_seen_in_pond_l37_37596


namespace simplify_expression_l37_37932

theorem simplify_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end simplify_expression_l37_37932


namespace students_play_neither_l37_37656

def total_students : ℕ := 35
def play_football : ℕ := 26
def play_tennis : ℕ := 20
def play_both : ℕ := 17

theorem students_play_neither : (total_students - (play_football + play_tennis - play_both)) = 6 := by
  sorry

end students_play_neither_l37_37656


namespace radius_ratio_l37_37534

noncomputable def volume_large : ℝ := 576 * Real.pi
noncomputable def volume_small : ℝ := 0.25 * volume_large

theorem radius_ratio (V_large V_small : ℝ) (h_large : V_large = 576 * Real.pi) (h_small : V_small = 0.25 * V_large) :
  (∃ r_ratio : ℝ, r_ratio = Real.sqrt (Real.sqrt (Real.sqrt (V_small / V_large)))) :=
begin
  rw [h_large, h_small],
  use 1 / Real.sqrt (Real.sqrt 4),
  sorry
end

end radius_ratio_l37_37534


namespace frac_div_l37_37649

theorem frac_div : (3 / 7) / (4 / 5) = 15 / 28 := by
  sorry

end frac_div_l37_37649


namespace log_eq_exponent_eq_l37_37023

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l37_37023


namespace fraction_pizza_covered_by_pepperoni_l37_37192

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni_l37_37192


namespace polynomial_not_product_of_single_var_l37_37927

theorem polynomial_not_product_of_single_var :
  ¬ ∃ (f : Polynomial ℝ) (g : Polynomial ℝ), 
    (∀ (x y : ℝ), (f.eval x) * (g.eval y) = (x^200) * (y^200) + 1) := sorry

end polynomial_not_product_of_single_var_l37_37927


namespace division_remainder_l37_37311

theorem division_remainder : 4053 % 23 = 5 :=
by
  sorry

end division_remainder_l37_37311


namespace cost_of_book_sold_at_loss_l37_37147

theorem cost_of_book_sold_at_loss
  (C1 C2 : ℝ)
  (total_cost : C1 + C2 = 360)
  (selling_price1 : 0.85 * C1 = 1.19 * C2) :
  C1 = 210 :=
sorry

end cost_of_book_sold_at_loss_l37_37147


namespace relationship_between_n_and_m_l37_37438

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def geometric_sequence (a q : ℝ) (m : ℕ) : ℝ :=
  a * q ^ (m - 1)

theorem relationship_between_n_and_m
  (a d q : ℝ) (n m : ℕ)
  (h_d_ne_zero : d ≠ 0)
  (h1 : arithmetic_sequence a d 1 = geometric_sequence a q 1)
  (h2 : arithmetic_sequence a d 3 = geometric_sequence a q 3)
  (h3 : arithmetic_sequence a d 7 = geometric_sequence a q 5)
  (q_pos : 0 < q) (q_sqrt2 : q^2 = 2)
  :
  n = 2 ^ ((m + 1) / 2) - 1 := sorry

end relationship_between_n_and_m_l37_37438


namespace area_of_triangle_l37_37844

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (2, 6)

-- Define the function to calculate the area of the triangle formed by three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The theorem statement that the area of the triangle with given vertices is 14.5
theorem area_of_triangle : triangle_area A B C = 14.5 :=
by 
  -- Skipping the proof part
  sorry

end area_of_triangle_l37_37844


namespace find_selling_price_l37_37501

-- Define the cost price of the article
def cost_price : ℝ := 47

-- Define the profit when the selling price is Rs. 54
def profit : ℝ := 54 - cost_price

-- Assume that the profit is the same as the loss
axiom profit_equals_loss : profit = 7

-- Define the selling price that yields the same loss as the profit
def selling_price_loss : ℝ := cost_price - profit

-- Now state the theorem to prove that the selling price for loss is Rs. 40
theorem find_selling_price : selling_price_loss = 40 :=
sorry

end find_selling_price_l37_37501


namespace find_a_solution_set_a_negative_l37_37395

-- Definitions
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Problem 1: Prove the value of 'a'
theorem find_a (h : ∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ -1/2)) :
  a = -2 :=
sorry

-- Problem 2: Prove the solution sets when a < 0
theorem solution_set_a_negative (h : a < 0) :
  (a = -1 → (∀ x : ℝ, quadratic_inequality a x ↔ x = -1)) ∧
  (a < -1 → (∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ 1/a))) ∧
  (-1 < a ∧ a < 0 → (∀ x : ℝ, quadratic_inequality a x ↔ (1/a ≤ x ∧ x ≤ -1))) :=
sorry

end find_a_solution_set_a_negative_l37_37395


namespace polynomial_roots_l37_37012

theorem polynomial_roots (d e : ℤ) :
  (∀ r, r^2 - 2 * r - 1 = 0 → r^5 - d * r - e = 0) ↔ (d = 29 ∧ e = 12) := by
  sorry

end polynomial_roots_l37_37012


namespace geometric_sequence_a3_l37_37882

theorem geometric_sequence_a3 :
  ∀ (a : ℕ → ℝ), a 1 = 2 → a 5 = 8 → (a 3 = 4 ∨ a 3 = -4) :=
by
  intros a h₁ h₅
  sorry

end geometric_sequence_a3_l37_37882


namespace arithmetic_sequence_term_l37_37760

theorem arithmetic_sequence_term :
  ∀ a : ℕ → ℕ, (a 1 = 1) → (∀ n : ℕ, a (n + 1) - a n = 2) → (a 6 = 11) :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_term_l37_37760


namespace calculate_allocations_l37_37439

variable (new_revenue : ℝ)
variable (ratio_employee_salaries ratio_stock_purchases ratio_rent ratio_marketing_costs : ℕ)

theorem calculate_allocations :
  let total_ratio := ratio_employee_salaries + ratio_stock_purchases + ratio_rent + ratio_marketing_costs
  let part_value := new_revenue / total_ratio
  let employee_salary_alloc := ratio_employee_salaries * part_value
  let rent_alloc := ratio_rent * part_value
  let marketing_costs_alloc := ratio_marketing_costs * part_value
  employee_salary_alloc + rent_alloc + marketing_costs_alloc = 7800 :=
by
  sorry

end calculate_allocations_l37_37439


namespace least_number_remainder_l37_37709

theorem least_number_remainder (n : ℕ) (hn : n = 115) : n % 38 = 1 ∧ n % 3 = 1 := by
  sorry

end least_number_remainder_l37_37709


namespace calc_nabla_example_l37_37741

-- Define the custom operation ∇
def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- State the proof problem
theorem calc_nabla_example : op_nabla (op_nabla 2 3) (op_nabla 4 5) = 49 / 56 := by
  sorry

end calc_nabla_example_l37_37741


namespace count_arrangements_california_l37_37856

-- Defining the counts of letters in "CALIFORNIA"
def word_length : ℕ := 10
def count_A : ℕ := 3
def count_I : ℕ := 2
def count_C : ℕ := 1
def count_L : ℕ := 1
def count_F : ℕ := 1
def count_O : ℕ := 1
def count_R : ℕ := 1
def count_N : ℕ := 1

-- The final proof statement to show the number of unique arrangements
theorem count_arrangements_california : 
  (Nat.factorial word_length) / 
  ((Nat.factorial count_A) * (Nat.factorial count_I)) = 302400 := by
  -- Placeholder for the proof, can be filled in later by providing the actual steps
  sorry

end count_arrangements_california_l37_37856


namespace sum_of_solutions_l37_37368

theorem sum_of_solutions (S : Set ℝ) (h : ∀ y ∈ S, y + 16 / y = 12) :
  ∃ t : ℝ, (∀ y ∈ S, y = 8 ∨ y = 4) ∧ t = 12 := by
  sorry

end sum_of_solutions_l37_37368


namespace C_increases_with_n_l37_37353

variables (n e R r : ℝ)
variables (h_pos_e : e > 0) (h_pos_R : R > 0)
variables (h_pos_r : r > 0) (h_R_nr : R > n * r)
noncomputable def C : ℝ := (e * n) / (R - n * r)

theorem C_increases_with_n (h_pos_e : e > 0) (h_pos_R : R > 0)
(h_pos_r : r > 0) (h_R_nr : R > n * r) (hn1 hn2 : ℝ)
(h_inequality : hn1 < hn2) : 
((e*hn1) / (R - hn1*r)) < ((e*hn2) / (R - hn2*r)) :=
by sorry

end C_increases_with_n_l37_37353


namespace solve_for_x_l37_37232

theorem solve_for_x (x : ℝ) (h : 3 * x + 20 = (1 / 3) * (7 * x + 45)) : x = -7.5 :=
sorry

end solve_for_x_l37_37232


namespace distance_between_stations_l37_37944

theorem distance_between_stations (x y t : ℝ) 
(start_same_hour : t > 0)
(speed_slow_train : ∀ t, x = 16 * t)
(speed_fast_train : ∀ t, y = 21 * t)
(distance_difference : y = x + 60) : 
  x + y = 444 := 
sorry

end distance_between_stations_l37_37944


namespace remainder_is_210_l37_37188

-- Define necessary constants and theorems
def x : ℕ := 2^35
def dividend : ℕ := 2^210 + 210
def divisor : ℕ := 2^105 + 2^63 + 1

theorem remainder_is_210 : (dividend % divisor) = 210 :=
by 
  -- Assume the calculation steps in the preceding solution are correct.
  -- No need to manually re-calculate as we've directly taken from the solution.
  sorry

end remainder_is_210_l37_37188


namespace number_of_friends_l37_37453

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37453


namespace number_of_x_values_l37_37398

theorem number_of_x_values : 
  (∃ x_values : Finset ℕ, (∀ x ∈ x_values, 10 ≤ x ∧ x < 25) ∧ x_values.card = 15) :=
by
  sorry

end number_of_x_values_l37_37398


namespace xiaolong_correct_answers_l37_37244

/-- There are 50 questions in the exam. Correct answers earn 3 points each,
incorrect answers deduct 1 point each, and unanswered questions score 0 points.
Xiaolong scored 120 points. Prove that the maximum number of questions 
Xiaolong answered correctly is 42. -/
theorem xiaolong_correct_answers :
  ∃ (x y : ℕ), 3 * x - y = 120 ∧ x + y = 48 ∧ x ≤ 50 ∧ y ≤ 50 ∧ x = 42 :=
by
  sorry

end xiaolong_correct_answers_l37_37244


namespace rectangle_divided_into_13_squares_l37_37982

-- Define the conditions
variables {a b s : ℝ} (m n : ℕ)

-- Mathematical equivalent proof problem Lean statement
theorem rectangle_divided_into_13_squares (h : a * b = 13 * s^2)
  (hm : a = m * s) (hn : b = n * s) (hmn : m * n = 13) :
  a / b = 13 ∨ b / a = 13 :=
begin
  sorry
end

end rectangle_divided_into_13_squares_l37_37982


namespace only_set_B_is_right_angle_triangle_l37_37520

def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem only_set_B_is_right_angle_triangle :
  is_right_angle_triangle 3 4 5 ∧ ¬is_right_angle_triangle 1 2 2 ∧ ¬is_right_angle_triangle 3 4 9 ∧ ¬is_right_angle_triangle 4 5 7 :=
by
  -- proof steps omitted
  sorry

end only_set_B_is_right_angle_triangle_l37_37520


namespace fraction_product_l37_37142

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l37_37142


namespace find_m_l37_37739

theorem find_m (m : ℝ) (h1 : (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12))) (h2 : m ≠ 0) : m = 12 :=
by
  sorry

end find_m_l37_37739


namespace sufficient_but_not_necessary_l37_37063

theorem sufficient_but_not_necessary (a : ℝ) : a = 1 → |a| = 1 ∧ (|a| = 1 → a = 1 → false) :=
by
  sorry

end sufficient_but_not_necessary_l37_37063


namespace min_distance_equals_sqrt2_over_2_l37_37592

noncomputable def min_distance_from_point_to_line (m n : ℝ) : ℝ :=
  (|m + n + 10|) / Real.sqrt (1^2 + 1^2)

def circle_eq (m n : ℝ) : Prop :=
  (m - 1 / 2)^2 + (n - 1 / 2)^2 = 1 / 2

theorem min_distance_equals_sqrt2_over_2 (m n : ℝ) (h1 : circle_eq m n) :
  min_distance_from_point_to_line m n = 1 / (Real.sqrt 2) :=
sorry

end min_distance_equals_sqrt2_over_2_l37_37592


namespace cube_edge_length_and_volume_l37_37130

variable (edge_length : ℕ)

def cube_edge_total_length (edge_length : ℕ) : ℕ := edge_length * 12
def cube_volume (edge_length : ℕ) : ℕ := edge_length * edge_length * edge_length

theorem cube_edge_length_and_volume (h : cube_edge_total_length edge_length = 96) :
  edge_length = 8 ∧ cube_volume edge_length = 512 :=
by
  sorry

end cube_edge_length_and_volume_l37_37130


namespace paolo_sevilla_birthday_l37_37480

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37480


namespace probability_of_specific_cards_l37_37536

noncomputable def probability_top_heart_second_spade_third_king 
  (deck_size : ℕ) (ranks_per_suit : ℕ) (suits : ℕ) (hearts : ℕ) (spades : ℕ) (kings : ℕ) : ℚ :=
  (hearts * spades * kings) / (deck_size * (deck_size - 1) * (deck_size - 2))

theorem probability_of_specific_cards :
  probability_top_heart_second_spade_third_king 104 26 4 26 26 8 = 169 / 34102 :=
by {
  sorry
}

end probability_of_specific_cards_l37_37536


namespace part_one_part_two_l37_37098

theorem part_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 := sorry

theorem part_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := sorry

end part_one_part_two_l37_37098


namespace child_ticket_cost_l37_37285

theorem child_ticket_cost :
  ∃ x : ℤ, (9 * 11 = 7 * x + 50) ∧ x = 7 :=
by
  sorry

end child_ticket_cost_l37_37285


namespace sin_alpha_eq_sqrt5_over_3_l37_37875

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l37_37875


namespace max_area_of_triangle_l37_37721

theorem max_area_of_triangle (a b c : ℝ) 
  (h1 : ∀ (a b c : ℝ), S = a^2 - (b - c)^2)
  (h2 : b + c = 8) : 
  S ≤ 64 / 17 :=
sorry

end max_area_of_triangle_l37_37721


namespace speech_competition_score_l37_37832

theorem speech_competition_score :
  let speech_content := 90
  let speech_skills := 80
  let speech_effects := 85
  let content_ratio := 4
  let skills_ratio := 2
  let effects_ratio := 4
  (speech_content * content_ratio + speech_skills * skills_ratio + speech_effects * effects_ratio) / (content_ratio + skills_ratio + effects_ratio) = 86 := by
  sorry

end speech_competition_score_l37_37832


namespace simple_interest_principal_l37_37503

theorem simple_interest_principal
  (P_CI : ℝ)
  (r_CI t_CI : ℝ)
  (CI : ℝ)
  (P_SI : ℝ)
  (r_SI t_SI SI : ℝ)
  (h_compound_interest : (CI = P_CI * (1 + r_CI / 100)^t_CI - P_CI))
  (h_simple_interest : SI = (1 / 2) * CI)
  (h_SI_formula : SI = P_SI * r_SI * t_SI / 100) :
  P_SI = 1750 :=
by
  have P_CI := 4000
  have r_CI := 10
  have t_CI := 2
  have r_SI := 8
  have t_SI := 3
  have CI := 840
  have SI := 420
  sorry

end simple_interest_principal_l37_37503


namespace question_statement_l37_37897

-- Definitions based on conditions
def all_cards : List ℕ := [8, 3, 6, 5, 0, 7]
def A : ℕ := 876  -- The largest number from the given cards.
def B : ℕ := 305  -- The smallest number from the given cards with non-zero hundreds place.

-- The proof problem statement
theorem question_statement :
  (A - B) * 6 = 3426 := by
  sorry

end question_statement_l37_37897


namespace part_I_part_II_l37_37228

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a + 2

theorem part_I (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 0) →
  -1 < a ∧ a ≤ 11/5 :=
sorry

noncomputable def g (x a : ℝ) : ℝ := 
  if abs x ≥ 1 then 2 * x^2 - 2 * a * x + a + 1 
  else -2 * a * x + a + 3

theorem part_II (a : ℝ) :
  (∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ x2 < 3 ∧ g x1 a = 0 ∧ g x2 a = 0) →
  1 + Real.sqrt 3 < a ∧ a ≤ 19/5 :=
sorry

end part_I_part_II_l37_37228


namespace rectangle_divided_into_13_squares_l37_37987

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end rectangle_divided_into_13_squares_l37_37987


namespace perfect_square_proof_l37_37091

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l37_37091


namespace worker_wage_before_promotion_l37_37549

variable (W_new : ℝ)
variable (W : ℝ)

theorem worker_wage_before_promotion (h1 : W_new = 45) (h2 : W_new = 1.60 * W) :
  W = 28.125 := by
  sorry

end worker_wage_before_promotion_l37_37549


namespace same_color_probability_l37_37738

def sides := 12
def violet_sides := 3
def orange_sides := 4
def lime_sides := 5

def prob_violet := violet_sides / sides
def prob_orange := orange_sides / sides
def prob_lime := lime_sides / sides

theorem same_color_probability :
  (prob_violet * prob_violet) + (prob_orange * prob_orange) + (prob_lime * prob_lime) = 25 / 72 :=
by
  sorry

end same_color_probability_l37_37738


namespace band_gigs_count_l37_37329

-- Definitions of earnings per role and total earnings
def leadSingerEarnings := 30
def guitaristEarnings := 25
def bassistEarnings := 20
def drummerEarnings := 25
def keyboardistEarnings := 20
def backupSingerEarnings := 15
def totalEarnings := 2055

-- Calculate total per gig earnings
def totalPerGigEarnings :=
  leadSingerEarnings + guitaristEarnings + bassistEarnings + drummerEarnings + keyboardistEarnings + backupSingerEarnings

-- Statement to prove the number of gigs played is 15
theorem band_gigs_count :
  totalEarnings / totalPerGigEarnings = 15 := 
by { sorry }

end band_gigs_count_l37_37329


namespace time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l37_37969

noncomputable def time_to_count_envelopes (num_envelopes : ℕ) : ℕ :=
(num_envelopes / 10) * 10

theorem time_to_count_60_envelopes_is_40 :
  time_to_count_envelopes 60 = 40 := 
sorry

theorem time_to_count_90_envelopes_is_10 :
  time_to_count_envelopes 90 = 10 := 
sorry

end time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l37_37969


namespace tan_pi_div_four_l37_37647

theorem tan_pi_div_four : Real.tan (π / 4) = 1 := by
  sorry

end tan_pi_div_four_l37_37647


namespace mirror_full_body_view_l37_37965

theorem mirror_full_body_view (AB MN : ℝ) (h : AB > 0): 
  (MN = 1/2 * AB) ↔
  ∀ (P : ℝ), (0 < P) → (P < AB) → 
    (P < MN + (AB - P)) ∧ (P > AB - MN + P) := 
by
  sorry

end mirror_full_body_view_l37_37965


namespace rectangular_solid_length_l37_37164

theorem rectangular_solid_length (w h : ℕ) (surface_area : ℕ) (l : ℕ) 
  (hw : w = 4) (hh : h = 1) (hsa : surface_area = 58) 
  (h_surface_area_formula : surface_area = 2 * l * w + 2 * l * h + 2 * w * h) : 
  l = 5 :=
by
  rw [hw, hh, hsa] at h_surface_area_formula
  sorry

end rectangular_solid_length_l37_37164


namespace students_accommodated_l37_37833

theorem students_accommodated 
  (total_students : ℕ)
  (total_workstations : ℕ)
  (workstations_accommodating_x_students : ℕ)
  (x : ℕ)
  (workstations_accommodating_3_students : ℕ)
  (workstation_capacity_10 : ℕ)
  (workstation_capacity_6 : ℕ) :
  total_students = 38 → 
  total_workstations = 16 → 
  workstations_accommodating_x_students = 10 → 
  workstations_accommodating_3_students = 6 → 
  workstation_capacity_10 = 10 * x → 
  workstation_capacity_6 = 6 * 3 → 
  10 * x + 18 = 38 → 
  10 * 2 = 20 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end students_accommodated_l37_37833


namespace geometric_sequence_condition_l37_37753

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < a 1) 
  (h2 : ∀ n, a (n + 1) = a n * q) :
  (a 1 < a 3) ↔ (a 1 < a 3) ∧ (a 3 < a 6) :=
sorry

end geometric_sequence_condition_l37_37753


namespace negation_of_existence_l37_37618

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_l37_37618


namespace M_eq_N_l37_37275

def M : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k + 1) * Real.pi}
def N : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k - 1) * Real.pi}

theorem M_eq_N : M = N := by
  sorry

end M_eq_N_l37_37275


namespace not_suitable_for_storing_l37_37802

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l37_37802


namespace inequality_proof_l37_37778

variable (ha la r R : ℝ)
variable (α β γ : ℝ)

-- Conditions
def condition1 : Prop := ha / la = Real.cos ((β - γ) / 2)
def condition2 : Prop := 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2) = 2 * r / R

-- The theorem to be proved
theorem inequality_proof (h1 : condition1 ha la β γ) (h2 : condition2 α β γ r R) :
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (2 * r / R) :=
sorry

end inequality_proof_l37_37778


namespace relationship_among_a_b_c_l37_37715

theorem relationship_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (1 / 2) ^ (3 / 2))
  (hb : b = Real.log pi)
  (hc : c = Real.logb 0.5 (3 / 2)) :
  c < a ∧ a < b :=
by 
  sorry

end relationship_among_a_b_c_l37_37715


namespace henrietta_paint_gallons_l37_37058

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l37_37058


namespace number_of_polynomials_correct_l37_37436

noncomputable def number_of_polynomials (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 2) + 1 else 0

theorem number_of_polynomials_correct (n : ℕ) (hn : n > 0) :
  number_of_polynomials n = (n / 2) + 1 :=
by {
  unfold number_of_polynomials,
  split_ifs,
  sorry -- Proof is omitted as specified
}

end number_of_polynomials_correct_l37_37436


namespace max_mark_is_600_l37_37169

-- Define the conditions
def forty_percent (M : ℝ) : ℝ := 0.40 * M
def student_score : ℝ := 175
def additional_marks_needed : ℝ := 65

-- The goal is to prove that the maximum mark is 600
theorem max_mark_is_600 (M : ℝ) :
  forty_percent M = student_score + additional_marks_needed → M = 600 := 
by 
  sorry

end max_mark_is_600_l37_37169


namespace birthday_friends_count_l37_37460

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37460


namespace term_2007_in_sequence_is_4_l37_37634

-- Definition of the function to compute the sum of the squares of the digits of a number
def sum_of_squares_of_digits (n : Nat) : Nat := 
  n.digits.sum (λ d => d * d)

-- Definition of the sequence based on the given rules
def sequence : Nat → Nat
| 0 => 2007
| (n + 1) => sum_of_squares_of_digits (sequence n)

-- Theorem stating that the 2007th term in the sequence is 4
theorem term_2007_in_sequence_is_4 : sequence 2007 = 4 :=
  sorry -- Proof skipped

end term_2007_in_sequence_is_4_l37_37634


namespace birthday_friends_count_l37_37456

theorem birthday_friends_count 
  (n : ℕ)
  (h1 : ∃ total_bill, total_bill = 12 * (n + 2))
  (h2 : ∃ total_bill, total_bill = 16 * n) :
  n = 6 := 
by sorry

end birthday_friends_count_l37_37456


namespace sushil_marks_ratio_l37_37490

theorem sushil_marks_ratio
  (E M Science : ℕ)
  (h1 : E + M + Science = 170)
  (h2 : E = M / 4)
  (h3 : Science = 17) :
  E = 31 :=
by
  sorry

end sushil_marks_ratio_l37_37490


namespace min_value_expression_l37_37040

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 2) ^ 2 + (b + 2) ^ 2 = 25 / 2 :=
sorry

end min_value_expression_l37_37040


namespace sufficient_but_not_necessary_not_necessary_l37_37754

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l37_37754


namespace repeating_decimal_fraction_equiv_l37_37648

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.4 + 37 / 990

theorem repeating_decimal_fraction_equiv : repeating_decimal_to_fraction (433 / 990) :=
by
  sorry

end repeating_decimal_fraction_equiv_l37_37648


namespace find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l37_37382

def f : ℝ → ℝ :=
  sorry

noncomputable def f_properties : Prop :=
  (∀ x y : ℝ, x < 0 → f x < 0 → f x + f y = f (x * y) / f (x + y)) ∧ f 1 = 1

theorem find_f2_f_neg1 :
  f_properties →
  f 2 = 1 / 2 ∧ f (-1) = -1 :=
sorry

theorem f_is_odd :
  f_properties →
  ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_monotonic_on_negatives :
  f_properties →
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2 :=
sorry

end find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l37_37382


namespace louis_current_age_l37_37414

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l37_37414


namespace complex_expression_proof_l37_37437

open Complex

theorem complex_expression_proof {x y z : ℂ}
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 18 :=
by
  sorry

end complex_expression_proof_l37_37437


namespace smallest_possible_sum_l37_37200

theorem smallest_possible_sum (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_diff : x ≠ y) (h_eq : 1/x + 1/y = 1/12) : x + y = 49 :=
by
  sorry

end smallest_possible_sum_l37_37200


namespace probability_red_or_white_l37_37524

-- Define the total number of marbles and the counts of blue and red marbles.
def total_marbles : Nat := 60
def blue_marbles : Nat := 5
def red_marbles : Nat := 9

-- Define the remainder to calculate white marbles.
def white_marbles : Nat := total_marbles - (blue_marbles + red_marbles)

-- Lean proof statement to show the probability of selecting a red or white marble.
theorem probability_red_or_white :
  (red_marbles + white_marbles) / total_marbles = 11 / 12 :=
by
  sorry

end probability_red_or_white_l37_37524


namespace find_multiplier_l37_37124

theorem find_multiplier (n x : ℝ) (h1 : n = 1.0) (h2 : 3 * n - 1 = x * n) : x = 2 :=
by
  sorry

end find_multiplier_l37_37124


namespace necessary_and_sufficient_condition_for_geometric_sequence_l37_37505

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {c : ℝ}

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n+1) = r * a_n n

theorem necessary_and_sufficient_condition_for_geometric_sequence :
  (∀ n : ℕ, S_n n = 2^n + c) →
  (∀ n : ℕ, a_n n = S_n n - S_n (n-1)) →
  is_geometric_sequence a_n ↔ c = -1 :=
by
  sorry

end necessary_and_sufficient_condition_for_geometric_sequence_l37_37505


namespace no_positive_integer_n_such_that_14n_plus_19_is_prime_l37_37787

theorem no_positive_integer_n_such_that_14n_plus_19_is_prime :
  ∀ n : Nat, 0 < n → ¬ Nat.Prime (14^n + 19) :=
by
  intro n hn
  sorry

end no_positive_integer_n_such_that_14n_plus_19_is_prime_l37_37787


namespace marbles_count_l37_37977

def num_violet_marbles := 64

def num_red_marbles := 14

def total_marbles (violet : Nat) (red : Nat) : Nat :=
  violet + red

theorem marbles_count :
  total_marbles num_violet_marbles num_red_marbles = 78 := by
  sorry

end marbles_count_l37_37977


namespace math_problem_l37_37044

variables (x y z : ℝ)

theorem math_problem
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( (x^2 / (x + y) >= (3 * x - y) / 4) ) ∧ 
  ( (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) >= (x * y + y * z + z * x) / 2 ) :=
by sorry

end math_problem_l37_37044


namespace principal_amount_l37_37146

theorem principal_amount (P R : ℝ) (h1 : P + (P * R * 2) / 100 = 780) (h2 : P + (P * R * 7) / 100 = 1020) : P = 684 := 
sorry

end principal_amount_l37_37146


namespace solve_quadratic_l37_37240

theorem solve_quadratic (x : ℝ) (h1 : 2 * x ^ 2 = 9 * x - 4) (h2 : x ≠ 4) : 2 * x = 1 :=
by
  -- The proof will go here
  sorry

end solve_quadratic_l37_37240


namespace find_triangle_angles_l37_37082

theorem find_triangle_angles 
  (α β γ : ℝ)
  (a b : ℝ)
  (h1 : γ = 2 * α)
  (h2 : b = 2 * a)
  (h3 : α + β + γ = 180) :
  α = 30 ∧ β = 90 ∧ γ = 60 := 
by 
  sorry

end find_triangle_angles_l37_37082


namespace smallest_d_for_range_of_g_l37_37022

theorem smallest_d_for_range_of_g :
  ∃ d, (∀ x : ℝ, x^2 + 4 * x + d = 3) → d = 7 := by
  sorry

end smallest_d_for_range_of_g_l37_37022


namespace prove_a_21022_le_1_l37_37276

-- Define the sequence a_n
variable (a : ℕ → ℝ)

-- Conditions for the sequence
axiom seq_condition {n : ℕ} (hn : n ≥ 1) :
  (a (n + 1))^2 + a n * a (n + 2) ≤ a n + a (n + 2)

-- Positive real numbers condition
axiom seq_positive {n : ℕ} (hn : n ≥ 1) :
  a n > 0

-- The main theorem to prove
theorem prove_a_21022_le_1 :
  a 21022 ≤ 1 :=
sorry

end prove_a_21022_le_1_l37_37276


namespace quadrilateral_is_kite_l37_37603

open Set

/-- Problem statement: Given the conditions of the original problem,
prove that the quadrilateral AMHN is a kite -/
theorem quadrilateral_is_kite 
    (A B C : Point)
    (acute_triangle : is_acute_triangle A B C)
    (D : Point)
    (AD_bisects_angle_A : is_angle_bisector A D B C)
    (H : Point)
    (altitude_from_A : is_altitude A H B C)
    (M : Point)
    (on_circle_BM : on_circle_center_radius B M (distance B D))
    (N : Point)
    (on_circle_CN : on_circle_center_radius C N (distance C D)) : 
  is_kite A M H N := sorry

end quadrilateral_is_kite_l37_37603


namespace roots_of_unity_l37_37617

noncomputable def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z^n = 1

noncomputable def is_cube_root_of_unity (z : ℂ) : Prop :=
  z^3 = 1

theorem roots_of_unity (x y : ℂ) (hx : is_root_of_unity x) (hy : is_root_of_unity y) (hxy : x ≠ y) :
  is_root_of_unity (x + y) ↔ is_cube_root_of_unity (y / x) :=
sorry

end roots_of_unity_l37_37617


namespace combination_eight_choose_five_l37_37379

theorem combination_eight_choose_five : 
  ∀ (n k : ℕ), n = 8 ∧ k = 5 → Nat.choose n k = 56 :=
by 
  intros n k h
  obtain ⟨hn, hk⟩ := h
  rw [hn, hk]
  exact Nat.choose_eq 8 5
  sorry  -- This signifies that the proof needs to be filled in, but we'll skip it as per instructions.

end combination_eight_choose_five_l37_37379


namespace count_groupings_l37_37791

theorem count_groupings (dogs : Finset ℕ) (Fluffy Nipper : ℕ) (h : Fluffy ≠ Nipper) (h_count : dogs.card = 12) :
  ∃ (g1 g2 g3 : Finset ℕ), 
    g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3 ∧
    Fluffy ∈ g1 ∧ Nipper ∈ g2 ∧
    (∀ x, x ∈ g1 ∨ x ∈ g2 ∨ x ∈ g3) ∧
     ∑ y in dogs, 1 = 12 ∧
     (∏ (a ∈ Finset.choose 10 3), ∏ (b ∈ Finset.choose 7 4), 1) = 4200
:= sorry

end count_groupings_l37_37791


namespace problem_statements_l37_37917

noncomputable def f (x : ℕ) : ℕ := x % 2
noncomputable def g (x : ℕ) : ℕ := x % 3

theorem problem_statements (x : ℕ) : (f (2 * x) = 0) ∧ (f x + f (x + 3) = 1) :=
by
  sorry

end problem_statements_l37_37917


namespace find_m_l37_37224

theorem find_m (m : ℕ) : (11 - m + 1 = 5) → m = 7 :=
by
  sorry

end find_m_l37_37224


namespace even_function_periodic_symmetric_about_2_l37_37584

variables {F : ℝ → ℝ}

theorem even_function_periodic_symmetric_about_2
  (h_even : ∀ x, F x = F (-x))
  (h_symmetric : ∀ x, F (2 - x) = F (2 + x))
  (h_cond : F 2011 + 2 * F 1 = 18) :
  F 2011 = 6 :=
sorry

end even_function_periodic_symmetric_about_2_l37_37584


namespace exists_four_functions_l37_37619

theorem exists_four_functions 
  (f : ℝ → ℝ)
  (h_periodic : ∀ x, f (x + 2 * Real.pi) = f x) :
  ∃ (f1 f2 f3 f4 : ℝ → ℝ), 
    (∀ x, f1 (-x) = f1 x ∧ f1 (x + Real.pi) = f1 x) ∧
    (∀ x, f2 (-x) = f2 x ∧ f2 (x + Real.pi) = f2 x) ∧
    (∀ x, f3 (-x) = f3 x ∧ f3 (x + Real.pi) = f3 x) ∧
    (∀ x, f4 (-x) = f4 x ∧ f4 (x + Real.pi) = f4 x) ∧
    (∀ x, f x = f1 x + f2 x * Real.cos x + f3 x * Real.sin x + f4 x * Real.sin (2 * x)) :=
sorry

end exists_four_functions_l37_37619


namespace female_lion_weight_l37_37918

theorem female_lion_weight (male_weight : ℚ) (weight_difference : ℚ) (female_weight : ℚ) : 
  male_weight = 145/4 → 
  weight_difference = 47/10 → 
  male_weight = female_weight + weight_difference → 
  female_weight = 631/20 :=
by
  intros h₁ h₂ h₃
  sorry

end female_lion_weight_l37_37918


namespace red_pill_cost_l37_37974

theorem red_pill_cost :
  ∃ (r : ℚ) (b : ℚ), (∀ (d : ℕ), d = 21 → 3 * r - 2 = 39) ∧
                      (1 ≤ d → r = b + 1) ∧
                      (21 * (r + 2 * b) = 819) → 
                      r = 41 / 3 :=
by sorry

end red_pill_cost_l37_37974


namespace overall_ratio_men_women_l37_37513

variables (m_w_diff players_total beginners_m beginners_w intermediate_m intermediate_w advanced_m advanced_w : ℕ)

def total_men : ℕ := beginners_m + intermediate_m + advanced_m
def total_women : ℕ := beginners_w + intermediate_w + advanced_w

theorem overall_ratio_men_women 
  (h1 : beginners_m = 2) 
  (h2 : beginners_w = 4)
  (h3 : intermediate_m = 3) 
  (h4 : intermediate_w = 5) 
  (h5 : advanced_m = 1) 
  (h6 : advanced_w = 3) 
  (h7 : m_w_diff = 4)
  (h8 : total_men = 6)
  (h9 : total_women = 12)
  (h10 : players_total = 18) :
  total_men / total_women = 1 / 2 :=
by {
  sorry
}

end overall_ratio_men_women_l37_37513


namespace c_minus_a_value_l37_37071

theorem c_minus_a_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 :=
by 
  sorry

end c_minus_a_value_l37_37071


namespace banker_discount_calculation_l37_37121

-- Define the future value function with given interest rates and periods.
def face_value (PV : ℝ) : ℝ :=
  (PV * (1 + 0.10) ^ 4) * (1 + 0.12) ^ 4

-- Define the true discount as the difference between the future value and the present value.
def true_discount (PV : ℝ) : ℝ :=
  face_value PV - PV

-- Given conditions
def banker_gain : ℝ := 900

-- Define the banker's discount.
def banker_discount (PV : ℝ) : ℝ :=
  banker_gain + true_discount PV

-- The proof statement to prove the relationship.
theorem banker_discount_calculation (PV : ℝ) :
  banker_discount PV = banker_gain + (face_value PV - PV) := by
  sorry

end banker_discount_calculation_l37_37121


namespace longest_diagonal_of_rhombus_l37_37540

theorem longest_diagonal_of_rhombus (A B : ℝ) (h1 : A = 150) (h2 : ∃ x, (A = 1/2 * (4 * x) * (3 * x)) ∧ (x = 5)) : 
  4 * (classical.some h2) = 20 := 
by sorry

end longest_diagonal_of_rhombus_l37_37540


namespace sufficient_but_not_necessary_not_necessary_l37_37755

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l37_37755


namespace more_red_than_yellow_l37_37267

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l37_37267


namespace duty_arrangements_240_l37_37978

/-
We need to define the conditions given:
- 7 days of duty.
- 5 people in the department.
- Each person can be on duty for up to 2 consecutive days.
- Everyone must be on duty at least once.
- The department head is on duty on the first day.

Given these, we need to prove that the total number of valid arrangements is 240.
-/

def numDutyArrangements (days people : ℕ) (maxDaysPerPerson : ℕ) (headOnFirstDay : Bool) : ℕ := sorry

theorem duty_arrangements_240 :
  numDutyArrangements 7 5 2 true = 240 :=
sorry

end duty_arrangements_240_l37_37978


namespace problem_solution_l37_37223

variable (α β : ℝ)

-- Conditions
variable (h1 : 3 * Real.sin α - Real.cos α = 0)
variable (h2 : 7 * Real.sin β + Real.cos β = 0)
variable (h3 : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)

theorem problem_solution : 2 * α - β = - (3 * π / 4) := by
  sorry

end problem_solution_l37_37223


namespace cube_divisibility_l37_37066

theorem cube_divisibility (a : ℤ) (k : ℤ) (h₁ : a > 1) 
(h₂ : (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 4 ∣ a := 
by
  sorry

end cube_divisibility_l37_37066


namespace quadratic_inequality_solution_l37_37733

theorem quadratic_inequality_solution 
  (x : ℝ) (b c : ℝ)
  (h : ∀ x, -x^2 + b*x + c < 0 ↔ x < -3 ∨ x > 2) :
  (6 * x^2 + x - 1 > 0) ↔ (x < -1/2 ∨ x > 1/3) := 
sorry

end quadratic_inequality_solution_l37_37733


namespace yuna_has_biggest_number_l37_37126

-- Define the numbers assigned to each student
def Yoongi_num : ℕ := 7
def Jungkook_num : ℕ := 6
def Yuna_num : ℕ := 9
def Yoojung_num : ℕ := 8

-- State the main theorem that Yuna has the biggest number
theorem yuna_has_biggest_number : 
  (Yuna_num = 9) ∧ (Yuna_num > Yoongi_num) ∧ (Yuna_num > Jungkook_num) ∧ (Yuna_num > Yoojung_num) :=
sorry

end yuna_has_biggest_number_l37_37126


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l37_37813

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l37_37813


namespace wendi_chickens_l37_37806

theorem wendi_chickens : 
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  in total_chickens = 13 :=
by
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  sorry

end wendi_chickens_l37_37806


namespace solution_value_of_a_l37_37590

noncomputable def verify_a (a : ℚ) (A : Set ℚ) : Prop :=
  A = {a - 2, 2 * a^2 + 5 * a, 12} ∧ -3 ∈ A

theorem solution_value_of_a (a : ℚ) (A : Set ℚ) (h : verify_a a A) : a = -3 / 2 := by
  sorry

end solution_value_of_a_l37_37590


namespace distance_between_points_l37_37182

theorem distance_between_points :
  let p1 := (3, -5)
  let p2 := (-4, 4)
  dist p1 p2 = Real.sqrt 130 := by
  sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

end distance_between_points_l37_37182


namespace friends_attended_birthday_l37_37448

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37448


namespace noemi_initial_amount_l37_37923

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end noemi_initial_amount_l37_37923


namespace theo_cookies_eaten_in_9_months_l37_37131

-- Define the basic variable values as per the conditions
def cookiesPerTime : Nat := 25
def timesPerDay : Nat := 5
def daysPerMonth : Nat := 27
def numMonths : Nat := 9

-- Define the total number of cookies Theo can eat in 9 months
def totalCookiesIn9Months : Nat :=
  cookiesPerTime * timesPerDay * daysPerMonth * numMonths

-- The theorem stating the answer
theorem theo_cookies_eaten_in_9_months :
  totalCookiesIn9Months = 30375 := by
  -- Proof will go here
  sorry

end theo_cookies_eaten_in_9_months_l37_37131


namespace most_persuasive_method_l37_37970

-- Survey data and conditions
def male_citizens : ℕ := 4258
def male_believe_doping : ℕ := 2360
def female_citizens : ℕ := 3890
def female_believe_framed : ℕ := 2386

def random_division_by_gender : Prop := true -- Represents the random division into male and female groups

-- Proposition to prove
theorem most_persuasive_method : 
  random_division_by_gender → 
  ∃ method : String, method = "Independence Test" := by
  sorry

end most_persuasive_method_l37_37970


namespace adding_sugar_increases_sweetness_l37_37134

theorem adding_sugar_increases_sweetness 
  (a b m : ℝ) (hb : b > a) (ha : a > 0) (hm : m > 0) : 
  (a / b) < (a + m) / (b + m) := 
by
  sorry

end adding_sugar_increases_sweetness_l37_37134


namespace vehicle_value_this_year_l37_37507

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l37_37507


namespace find_a_l37_37046

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem find_a (a : ℝ) (h : binomial_coefficient 4 2 + 4 * a = 10) : a = 1 :=
by
  sorry

end find_a_l37_37046


namespace relationship_of_sets_l37_37060

def set_A : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 6 + 1}
def set_B : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 3 + 1 / 2}
def set_C : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k : ℝ) / 3 + 1 / 2}

theorem relationship_of_sets : set_C ⊆ set_B ∧ set_B ⊆ set_A := by
  sorry

end relationship_of_sets_l37_37060


namespace friends_at_birthday_l37_37472

theorem friends_at_birthday (n : ℕ) (total_bill : ℕ) :
  total_bill = 12 * (n + 2) ∧ total_bill = 16 * n → n = 6 :=
by
  intro h
  cases h with h1 h2
  have h3 : 12 * (n + 2) = 16 * n := h1
  sorry

end friends_at_birthday_l37_37472


namespace ball_probability_l37_37156

theorem ball_probability:
  let total_balls := 120
  let red_balls := 12
  let purple_balls := 18
  let yellow_balls := 15
  let desired_probability := 33 / 1190
  let probability_red := red_balls / total_balls
  let probability_purple_or_yellow := (purple_balls + yellow_balls) / (total_balls - 1)
  (probability_red * probability_purple_or_yellow = desired_probability) :=
sorry

end ball_probability_l37_37156


namespace y_is_defined_iff_x_not_equal_to_10_l37_37796

def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 10

theorem y_is_defined_iff_x_not_equal_to_10 (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 10)) ↔ range_of_independent_variable x :=
by sorry

end y_is_defined_iff_x_not_equal_to_10_l37_37796


namespace bisection_method_next_interval_l37_37137

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x0 := (a + b) / 2
  (f a * f x0 < 0) ∨ (f x0 * f b < 0) →
  (x0 = 2.5) →
  f 2 * f 2.5 < 0 :=
by
  intros
  sorry

end bisection_method_next_interval_l37_37137


namespace geo_seq_sum_S4_l37_37726

noncomputable def geom_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geo_seq_sum_S4 {a : ℝ} {q : ℝ} (h1 : a * q^2 - a = 15) (h2 : a * q - a = 5) :
  geom_seq_sum a q 4 = 75 :=
by
  sorry

end geo_seq_sum_S4_l37_37726


namespace sally_eggs_l37_37929

def dozen := 12
def total_eggs := 48

theorem sally_eggs : total_eggs / dozen = 4 := by
  -- Normally a proof would follow here, but we will use sorry to skip it
  sorry

end sally_eggs_l37_37929


namespace paolo_sevilla_birthday_l37_37481

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37481


namespace train_length_proof_l37_37546

-- Definitions based on the conditions given in the problem
def speed_km_per_hr := 45 -- speed of the train in km/hr
def time_seconds := 60 -- time taken to pass the platform in seconds
def length_platform_m := 390 -- length of the platform in meters

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Calculate the speed in m/s
def speed_m_per_s : ℕ := km_per_hr_to_m_per_s speed_km_per_hr

-- Calculate the total distance covered by the train while passing the platform
def total_distance_m : ℕ := speed_m_per_s * time_seconds

-- Total distance is the sum of the length of the train and the length of the platform
def length_train_m := total_distance_m - length_platform_m

-- The statement to prove the length of the train
theorem train_length_proof : length_train_m = 360 :=
by
  sorry

end train_length_proof_l37_37546


namespace four_hash_two_equals_forty_l37_37853

def hash_op (a b : ℕ) : ℤ := (a^2 + b^2) * (a - b)

theorem four_hash_two_equals_forty : hash_op 4 2 = 40 := 
by
  sorry

end four_hash_two_equals_forty_l37_37853


namespace find_abc_l37_37564

theorem find_abc (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a, b, c) = (3, 5, 15) ∨ (a, b, c) = (2, 4, 8) :=
by
  sorry

end find_abc_l37_37564


namespace dagger_evaluation_l37_37408

def dagger (a b : ℚ) : ℚ :=
match a, b with
| ⟨m, n, _, _⟩, ⟨p, q, _, _⟩ => (m * p : ℚ) * (q / n : ℚ)

theorem dagger_evaluation : dagger (3/7) (11/4) = 132/7 := by
  sorry

end dagger_evaluation_l37_37408


namespace find_n_of_arithmetic_sequence_l37_37298

theorem find_n_of_arithmetic_sequence :
  ∃ n : ℕ, (∀ (a : ℕ → ℤ), a 2 = 12 ∧ a n = -20 ∧ (∀ m : ℕ, a (m + 1) = a m - 2) → n = 18) :=
by
  sorry

end find_n_of_arithmetic_sequence_l37_37298


namespace min_sum_x_y_condition_l37_37880

theorem min_sum_x_y_condition {x y : ℝ} (h₁ : x > 0) (h₂ : y > 0) (h₃ : 1 / x + 9 / y = 1) : x + y = 16 :=
by
  sorry -- proof skipped

end min_sum_x_y_condition_l37_37880


namespace quadratic_range_extrema_l37_37734

def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem quadratic_range_extrema :
  let y := quadratic
  ∃ x_max x_min,
    (x_min = -2 ∧ y x_min = -2) ∧
    (x_max = -2 ∧ y x_max = 14 ∨ x_max = 5 ∧ y x_max = 7) := 
by
  sorry

end quadratic_range_extrema_l37_37734


namespace solve_for_m_l37_37655

theorem solve_for_m (n : ℝ) (m : ℝ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 1 / 2 := 
sorry

end solve_for_m_l37_37655


namespace parabolic_arch_height_l37_37797

noncomputable def arch_height (a : ℝ) : ℝ :=
  a * (0 : ℝ)^2

theorem parabolic_arch_height :
  ∃ (a : ℝ), (∫ x in (-4 : ℝ)..4, a * x^2) = (160 : ℝ) ∧ arch_height a = 30 :=
by
  sorry

end parabolic_arch_height_l37_37797


namespace total_wait_time_difference_l37_37305

theorem total_wait_time_difference :
  let kids_swings := 6
  let kids_slide := 4 * kids_swings
  let wait_time_swings := [210, 420, 840] -- in seconds
  let total_wait_time_swings := wait_time_swings.sum
  let wait_time_slide := [45, 90, 180] -- in seconds
  let total_wait_time_slide := wait_time_slide.sum
  let total_wait_time_all_kids_swings := kids_swings * total_wait_time_swings
  let total_wait_time_all_kids_slide := kids_slide * total_wait_time_slide
  let difference := total_wait_time_all_kids_swings - total_wait_time_all_kids_slide
  difference = 1260 := sorry

end total_wait_time_difference_l37_37305


namespace kevin_age_l37_37136

theorem kevin_age (x : ℕ) :
  (∃ n : ℕ, x - 2 = n^2) ∧ (∃ m : ℕ, x + 2 = m^3) → x = 6 :=
by
  sorry

end kevin_age_l37_37136


namespace find_value_am2_bm_minus_7_l37_37402

variable {a b m : ℝ}

theorem find_value_am2_bm_minus_7
  (h : a * m^2 + b * m + 5 = 0) : a * m^2 + b * m - 7 = -12 :=
by
  sorry

end find_value_am2_bm_minus_7_l37_37402


namespace parabola_directrix_l37_37936

theorem parabola_directrix (x y : ℝ) (h : x^2 = 12 * y) : y = -3 :=
sorry

end parabola_directrix_l37_37936


namespace smallest_n_inverse_mod_1176_l37_37952

theorem smallest_n_inverse_mod_1176 : ∃ n : ℕ, n > 1 ∧ Nat.Coprime n 1176 ∧ (∀ m : ℕ, m > 1 ∧ Nat.Coprime m 1176 → n ≤ m) ∧ n = 5 := by
  sorry

end smallest_n_inverse_mod_1176_l37_37952


namespace tangent_normal_lines_l37_37660

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l37_37660


namespace second_number_is_90_l37_37825

theorem second_number_is_90 (x y z : ℕ) 
  (h1 : z = 4 * y) 
  (h2 : y = 2 * x) 
  (h3 : (x + y + z) / 3 = 165) : y = 90 := 
by
  sorry

end second_number_is_90_l37_37825


namespace translate_parabola_upwards_l37_37135

theorem translate_parabola_upwards (x y : ℝ) (h : y = x^2) : y + 1 = x^2 + 1 :=
by
  sorry

end translate_parabola_upwards_l37_37135


namespace cylinder_in_cone_l37_37544

noncomputable def cylinder_radius : ℝ :=
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := (10 * 2) / 9  -- based on the derived form of r calculation
  r

theorem cylinder_in_cone :
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := cylinder_radius
  (r = 20 / 9) :=
by
  sorry -- Proof mechanism is skipped as per instructions.

end cylinder_in_cone_l37_37544


namespace more_red_than_yellow_l37_37268

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l37_37268


namespace B_cycling_speed_l37_37673

/--
A walks at 10 kmph. 10 hours after A starts, B cycles after him at a certain speed.
B catches up with A at a distance of 200 km from the start. Prove that B's cycling speed is 20 kmph.
-/
theorem B_cycling_speed (speed_A : ℝ) (time_A_to_start_B : ℝ) 
  (distance_at_catch : ℝ) (B_speed : ℝ)
  (h1 : speed_A = 10) 
  (h2 : time_A_to_start_B = 10)
  (h3 : distance_at_catch = 200)
  (h4 : distance_at_catch = speed_A * time_A_to_start_B + speed_A * (distance_at_catch / speed_B)) :
    B_speed = 20 := by
  sorry

end B_cycling_speed_l37_37673


namespace namjoon_rank_l37_37805

theorem namjoon_rank (total_students : ℕ) (fewer_than_namjoon : ℕ) (rank_of_namjoon : ℕ) 
  (h1 : total_students = 13) (h2 : fewer_than_namjoon = 4) : rank_of_namjoon = 9 :=
sorry

end namjoon_rank_l37_37805


namespace number_of_birdhouses_l37_37351

-- Definitions for the conditions
def cost_per_nail : ℝ := 0.05
def cost_per_plank : ℝ := 3.0
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def total_cost : ℝ := 88.0

-- Total cost calculation per birdhouse
def cost_per_birdhouse := planks_per_birdhouse * cost_per_plank + nails_per_birdhouse * cost_per_nail

-- Proving that the number of birdhouses is 4
theorem number_of_birdhouses : total_cost / cost_per_birdhouse = 4 := by
  sorry

end number_of_birdhouses_l37_37351


namespace rectangles_divided_into_13_squares_l37_37984

theorem rectangles_divided_into_13_squares (m n : ℕ) (h : m * n = 13) : 
  (m = 1 ∧ n = 13) ∨ (m = 13 ∧ n = 1) :=
sorry

end rectangles_divided_into_13_squares_l37_37984


namespace problem_statement_l37_37774

theorem problem_statement (a b : ℝ) :
  a^2 + b^2 - a - b - a * b + 0.25 ≥ 0 ∧ (a^2 + b^2 - a - b - a * b + 0.25 = 0 ↔ ((a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0))) :=
by 
  sorry

end problem_statement_l37_37774


namespace dog_catches_rabbit_in_4_minutes_l37_37360

def dog_speed_mph : ℝ := 24
def rabbit_speed_mph : ℝ := 15
def rabbit_head_start : ℝ := 0.6

theorem dog_catches_rabbit_in_4_minutes : 
  (∃ t : ℝ, t > 0 ∧ 0.4 * t = 0.25 * t + 0.6) → ∃ t : ℝ, t = 4 :=
sorry

end dog_catches_rabbit_in_4_minutes_l37_37360


namespace evaluate_expression_l37_37097

noncomputable def a := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def b := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def c := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def d := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2

theorem evaluate_expression : (1 / a + 1 / b + 1 / c + 1 / d)^2 = 39 / 140 := 
by
  sorry

end evaluate_expression_l37_37097


namespace unique_k_satisfying_eq_l37_37946

theorem unique_k_satisfying_eq (k : ℤ) :
  (∀ a b c : ℝ, (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ k = -1 :=
sorry

end unique_k_satisfying_eq_l37_37946


namespace number_of_friends_l37_37451

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37451


namespace range_of_k_l37_37744

theorem range_of_k (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 + 2*x1 - k = 0) ∧ (x2^2 + 2*x2 - k = 0)) ↔ k > -1 :=
by
  sorry

end range_of_k_l37_37744


namespace min_value_of_expr_l37_37992

theorem min_value_of_expr (a : ℝ) (ha : a > 1) : a + a^2 / (a - 1) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l37_37992


namespace area_error_percent_l37_37657

theorem area_error_percent (L W : ℝ) (L_pos : 0 < L) (W_pos : 0 < W) :
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error_percent := ((A_measured - A) / A) * 100
  error_percent = 0.8 :=
by
  let A := L * W
  let A_measured := (1.05 * L) * (0.96 * W)
  let error := A_measured - A
  let error_percent := (error / A) * 100
  sorry

end area_error_percent_l37_37657


namespace apples_purchased_by_danny_l37_37776

theorem apples_purchased_by_danny (pinky_apples : ℕ) (total_apples : ℕ) (danny_apples : ℕ) :
  pinky_apples = 36 → total_apples = 109 → danny_apples = total_apples - pinky_apples → danny_apples = 73 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_purchased_by_danny_l37_37776


namespace find_c_l37_37306

theorem find_c (c : ℝ) :
  (∀ x y : ℝ, 2*x^2 - 4*c*x*y + (2*c^2 + 1)*y^2 - 2*x - 6*y + 9 ≥ 0) ↔ c = 1/6 :=
by
  sorry

end find_c_l37_37306


namespace perpendicular_line_plane_l37_37388

variables {m : ℝ}

theorem perpendicular_line_plane (h : (4 / 2) = (2 / 1) ∧ (2 / 1) = (m / -1)) : m = -2 :=
by
  sorry

end perpendicular_line_plane_l37_37388


namespace at_least_one_l37_37303

axiom P : Prop  -- person A is an outstanding student
axiom Q : Prop  -- person B is an outstanding student

theorem at_least_one (H : ¬(¬P ∧ ¬Q)) : P ∨ Q :=
sorry

end at_least_one_l37_37303


namespace max_value_sin_sin2x_l37_37886

open Real

/-- Given x is an acute angle, find the maximum value of the function y = sin x * sin (2 * x). -/
theorem max_value_sin_sin2x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
    ∃ max_y : ℝ, ∀ y : ℝ, y = sin x * sin (2 * x) -> y ≤ max_y ∧ max_y = 4 * sqrt 3 / 9 :=
by
  -- To be completed
  sorry

end max_value_sin_sin2x_l37_37886


namespace solve_equation_l37_37640

theorem solve_equation (x : ℝ) (h : (3 * x) / (x + 1) = 9 / (x + 1)) : x = 3 :=
by sorry

end solve_equation_l37_37640


namespace squared_difference_of_roots_l37_37740

theorem squared_difference_of_roots:
  ∀ (Φ φ : ℝ), (∀ x : ℝ, x^2 = 2*x + 1 ↔ (x = Φ ∨ x = φ)) ∧ Φ ≠ φ → (Φ - φ)^2 = 8 :=
by
  intros Φ φ h
  sorry

end squared_difference_of_roots_l37_37740


namespace eval_expression_l37_37326

open Real

theorem eval_expression :
  (0.8^5 - (0.5^6 / 0.8^4) + 0.40 + 0.5^3 - log 0.3 + sin (π / 6)) = 2.51853302734375 :=
  sorry

end eval_expression_l37_37326


namespace number_of_friends_l37_37449

-- Definitions based on conditions
def total_bill_divided_among_all (n : ℕ) : ℕ := 12 * (n + 2)
def total_bill_divided_among_friends (n : ℕ) : ℕ := 16 * n

-- The theorem to prove
theorem number_of_friends (n : ℕ) : total_bill_divided_among_all n = total_bill_divided_among_friends n → n = 6 :=
by
  sorry

end number_of_friends_l37_37449


namespace necessary_but_not_sufficient_condition_l37_37080

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) :
  condition a → ¬ is_geometric_sequence a :=
sorry

end necessary_but_not_sufficient_condition_l37_37080


namespace reading_time_difference_l37_37821

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l37_37821


namespace number_of_friends_l37_37466

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37466


namespace cos_identity_l37_37725

theorem cos_identity (α : ℝ) (h : Real.cos (π / 4 - α) = -1 / 3) :
  Real.cos (3 * π / 4 + α) = 1 / 3 :=
sorry

end cos_identity_l37_37725


namespace paolo_sevilla_birthday_l37_37479

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37479


namespace find_sine_of_alpha_l37_37877

theorem find_sine_of_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * real.cos (2 * α) - 8 * real.cos α = 5) :
  real.sin α = real.sqrt 5 / 3 :=
sorry

end find_sine_of_alpha_l37_37877


namespace peaches_picked_l37_37103

variable (o t : ℕ)
variable (p : ℕ)

theorem peaches_picked : (o = 34) → (t = 86) → (t = o + p) → p = 52 :=
by
  intros ho ht htot
  rw [ho, ht] at htot
  sorry

end peaches_picked_l37_37103


namespace zhang_bing_age_18_l37_37145

theorem zhang_bing_age_18 {x a : ℕ} (h1 : x < 2023) 
  (h2 : a = x - 1953)
  (h3 : a % 9 = 0)
  (h4 : a = (x % 10) + ((x / 10) % 10) + ((x / 100) % 10) + ((x / 1000) % 10)) :
  a = 18 :=
sorry

end zhang_bing_age_18_l37_37145


namespace average_salary_excluding_manager_l37_37492

theorem average_salary_excluding_manager
    (A : ℝ)
    (manager_salary : ℝ)
    (total_employees : ℕ)
    (salary_increase : ℝ)
    (h1 : total_employees = 24)
    (h2 : manager_salary = 4900)
    (h3 : salary_increase = 100)
    (h4 : 24 * A + manager_salary = 25 * (A + salary_increase)) :
    A = 2400 := by
  sorry

end average_salary_excluding_manager_l37_37492


namespace calculate_expression_l37_37817

theorem calculate_expression : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end calculate_expression_l37_37817


namespace difference_between_max_and_min_change_l37_37973

-- Define percentages as fractions for Lean
def initial_yes : ℚ := 60 / 100
def initial_no : ℚ := 40 / 100
def final_yes : ℚ := 80 / 100
def final_no : ℚ := 20 / 100
def new_students : ℚ := 10 / 100

-- Define the minimum and maximum possible values of changes (in percentage as a fraction)
def min_change : ℚ := 10 / 100
def max_change : ℚ := 50 / 100

-- The theorem we need to prove
theorem difference_between_max_and_min_change : (max_change - min_change) = 40 / 100 :=
by
  sorry

end difference_between_max_and_min_change_l37_37973


namespace original_price_of_petrol_l37_37671

theorem original_price_of_petrol (P : ℝ) (h : 0.9 * P * 190 / (0.9 * P) = 190 / P + 5) : P = 4.22 :=
by
  -- The proof goes here
  sorry

end original_price_of_petrol_l37_37671


namespace gcd_168_486_l37_37945

theorem gcd_168_486 : gcd 168 486 = 6 := 
by sorry

end gcd_168_486_l37_37945


namespace Mikail_money_left_after_purchase_l37_37440

def Mikail_age_tomorrow : ℕ := 9  -- Defining Mikail's age tomorrow as 9.

def gift_per_year : ℕ := 5  -- Defining the gift amount per year of age as $5.

def video_game_cost : ℕ := 80  -- Defining the cost of the video game as $80.

def calculate_gift (age : ℕ) : ℕ := age * gift_per_year  -- Function to calculate the gift money he receives based on his age.

-- The statement we need to prove:
theorem Mikail_money_left_after_purchase : 
    calculate_gift Mikail_age_tomorrow < video_game_cost → calculate_gift Mikail_age_tomorrow - video_game_cost = 0 :=
by
  sorry

end Mikail_money_left_after_purchase_l37_37440


namespace measure_angle_BCA_l37_37178

theorem measure_angle_BCA 
  (BCD_angle : ℝ)
  (CBA_angle : ℝ)
  (sum_angles : BCD_angle + CBA_angle + BCA_angle = 190)
  (BCD_right : BCD_angle = 90)
  (CBA_given : CBA_angle = 70) :
  BCA_angle = 30 :=
by
  sorry

end measure_angle_BCA_l37_37178


namespace value_of_five_minus_c_l37_37890

theorem value_of_five_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 7 + d = 10 + c) :
  5 - c = 6 :=
by
  sorry

end value_of_five_minus_c_l37_37890


namespace sum_of_other_endpoint_coords_l37_37287

theorem sum_of_other_endpoint_coords (x y : ℝ) (hx : (6 + x) / 2 = 5) (hy : (2 + y) / 2 = 7) : x + y = 16 := 
  sorry

end sum_of_other_endpoint_coords_l37_37287


namespace one_non_congruent_triangle_with_perimeter_10_l37_37736

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l37_37736


namespace smallest_among_l37_37341

theorem smallest_among {a b c d : ℤ} (h1 : a = -4) (h2 : b = -3) (h3 : c = 0) (h4 : d = 1) :
  a < b ∧ a < c ∧ a < d :=
by
  rw [h1, h2, h3, h4]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_among_l37_37341


namespace divides_power_diff_l37_37274

theorem divides_power_diff (x : ℤ) (y z w : ℕ) (hy : y % 2 = 1) (hz : z % 2 = 1) (hw : w % 2 = 1) : 17 ∣ x^(y^(z^w)) - x^(y^z) := 
by
  sorry

end divides_power_diff_l37_37274


namespace vehicle_value_this_year_l37_37508

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l37_37508


namespace segment_length_segment_fraction_three_segments_fraction_l37_37966

noncomputable def total_length : ℝ := 4
noncomputable def number_of_segments : ℕ := 5

theorem segment_length (L : ℝ) (n : ℕ) (hL : L = total_length) (hn : n = number_of_segments) :
  L / n = (4 / 5 : ℝ) := by
sorry

theorem segment_fraction (n : ℕ) (hn : n = number_of_segments) :
  (1 / n : ℝ) = (1 / 5 : ℝ) := by
sorry

theorem three_segments_fraction (n : ℕ) (hn : n = number_of_segments) :
  (3 / n : ℝ) = (3 / 5 : ℝ) := by
sorry

end segment_length_segment_fraction_three_segments_fraction_l37_37966


namespace smallest_sum_l37_37214

theorem smallest_sum (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 : ℚ)/x + (1 : ℚ)/y = (1 : ℚ)/12) : x + y = 49 :=
sorry

end smallest_sum_l37_37214


namespace non_congruent_triangles_with_perimeter_10_l37_37737

theorem non_congruent_triangles_with_perimeter_10 :
  ∃ (T : Finset (Finset (ℕ × ℕ × ℕ))),
    (∀ (t ∈ T), let (a, b, c) := t in a ≤ b ∧ b ≤ c ∧
                  a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 4 :=
by
  sorry

end non_congruent_triangles_with_perimeter_10_l37_37737


namespace find_hypotenuse_l37_37420

-- Let a, b be the legs of the right triangle, c be the hypotenuse.
-- Let h be the altitude to the hypotenuse and r be the radius of the inscribed circle.
variable (a b c h r : ℝ)

-- Assume conditions of a right-angled triangle
def right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Given the altitude to the hypotenuse
def altitude (h c : ℝ) : Prop :=
  ∃ a b : ℝ, right_angled a b c ∧ h = a * b / c

-- Given the radius of the inscribed circle
def inscribed_radius (r a b c : ℝ) : Prop :=
  r = (a + b - c) / 2

-- The proof problem statement
theorem find_hypotenuse (a b c h r : ℝ) 
  (h_right_angled : right_angled a b c)
  (h_altitude : altitude h c)
  (h_inscribed_radius : inscribed_radius r a b c) : 
  c = 2 * r^2 / (h - 2 * r) :=
  sorry

end find_hypotenuse_l37_37420


namespace factorize_expr_l37_37364

theorem factorize_expr (y : ℝ) : 3 * y ^ 2 - 6 * y + 3 = 3 * (y - 1) ^ 2 :=
by
  sorry

end factorize_expr_l37_37364


namespace chess_team_boys_l37_37331

-- Definitions based on the conditions
def members : ℕ := 30
def attendees : ℕ := 20

-- Variables representing boys (B) and girls (G)
variables (B G : ℕ)

-- Defining the conditions
def condition1 : Prop := B + G = members
def condition2 : Prop := (2 * G) / 3 + B = attendees

-- The problem statement: proving that B = 0
theorem chess_team_boys (h1 : condition1 B G) (h2 : condition2 B G) : B = 0 :=
  sorry

end chess_team_boys_l37_37331


namespace find_a_value_l37_37730

theorem find_a_value : 
  (∀ x, (3 * (x - 2) - 4 * (x - 5 / 4) = 0) ↔ ( ∃ a, ((2 * x - a) / 3 - (x - a) / 2 = x - 1) ∧ a = -11 )) := sorry

end find_a_value_l37_37730


namespace count_ordered_pairs_l37_37867

theorem count_ordered_pairs : 
  ∃ n, n = 719 ∧ 
    (∀ (a b : ℕ), a + b = 1100 → 
      (∀ d ∈ [a, b], 
        ¬(∃ k : ℕ, d = 10 * k ∨ d % 10 = 0 ∨ d / 10 % 10 = 0 ∨ d % 5 = 0))) -> n = 719 :=
by
  sorry

end count_ordered_pairs_l37_37867


namespace max_workers_l37_37658

-- Each worker produces 10 bricks a day and steals as many bricks per day as there are workers at the factory.
def worker_bricks_produced_per_day : ℕ := 10
def worker_bricks_stolen_per_day (n : ℕ) : ℕ := n

-- The factory must have at least 13 more bricks at the end of the day.
def factory_brick_surplus_requirement : ℕ := 13

-- Prove the maximum number of workers that can be hired so that the factory has at least 13 more bricks than at the beginning:
theorem max_workers
  (n : ℕ) -- Let \( n \) be the number of workers at the brick factory.
  (h : worker_bricks_produced_per_day * n - worker_bricks_stolen_per_day n + 13 ≥ factory_brick_surplus_requirement): 
  n = 8 := 
sorry

end max_workers_l37_37658


namespace combined_shoe_size_l37_37429

-- Definitions based on conditions
def Jasmine_size : ℕ := 7
def Alexa_size : ℕ := 2 * Jasmine_size
def Clara_size : ℕ := 3 * Jasmine_size

-- Statement to prove
theorem combined_shoe_size : Jasmine_size + Alexa_size + Clara_size = 42 :=
by
  sorry

end combined_shoe_size_l37_37429


namespace sin_of_acute_angle_l37_37048

theorem sin_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : tan (π - α) + 3 = 0) : 
  sin α = 3 * ( sqrt 10 ) / 10 :=
sorry

end sin_of_acute_angle_l37_37048


namespace solve_for_x_l37_37569

theorem solve_for_x (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 :=
by
  sorry

end solve_for_x_l37_37569


namespace greatest_sum_of_digits_base8_l37_37138

theorem greatest_sum_of_digits_base8 (n : ℕ) (h1 : n > 0) (h2 : n < 1800) : 
  ∃ m, (m < 1800) ∧ (∃ s, s = Nat.digits 8 m ∧ s.sum = 23) :=
sorry

end greatest_sum_of_digits_base8_l37_37138


namespace bamboo_middle_node_capacity_l37_37793

def capacities_form_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem bamboo_middle_node_capacity :
  ∃ (a : ℕ → ℚ) (d : ℚ), 
    capacities_form_arithmetic_sequence a d ∧ 
    (a 1 + a 2 + a 3 = 4) ∧
    (a 6 + a 7 + a 8 + a 9 = 3) ∧
    (a 5 = 67 / 66) :=
  sorry

end bamboo_middle_node_capacity_l37_37793


namespace rectangle_area_k_value_l37_37638

theorem rectangle_area_k_value (d : ℝ) (length width : ℝ) (h1 : 5 * width = 2 * length) (h2 : d^2 = length^2 + width^2) :
  ∃ (k : ℝ), A = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_k_value_l37_37638


namespace greatest_possible_number_of_blue_chips_l37_37942

-- Definitions based on conditions
def total_chips : Nat := 72

-- Definition of the relationship between red and blue chips where p is a prime number
def is_prime (n : Nat) : Prop := Nat.Prime n

def satisfies_conditions (r b p : Nat) : Prop :=
  r + b = total_chips ∧ r = b + p ∧ is_prime p

-- The statement to prove
theorem greatest_possible_number_of_blue_chips (r b p : Nat) 
  (h : satisfies_conditions r b p) : b = 35 := 
sorry

end greatest_possible_number_of_blue_chips_l37_37942


namespace measure_of_one_interior_angle_of_regular_nonagon_is_140_l37_37517

-- Define the number of sides for a nonagon
def number_of_sides_nonagon : ℕ := 9

-- Define the formula for the sum of the interior angles of a regular n-gon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- The sum of the interior angles of a nonagon
def sum_of_interior_angles_nonagon : ℕ := sum_of_interior_angles number_of_sides_nonagon

-- The measure of one interior angle of a regular n-gon
def measure_of_one_interior_angle (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- The measure of one interior angle of a regular nonagon
def measure_of_one_interior_angle_nonagon : ℕ := measure_of_one_interior_angle number_of_sides_nonagon

-- The final theorem statement
theorem measure_of_one_interior_angle_of_regular_nonagon_is_140 : 
  measure_of_one_interior_angle_nonagon = 140 :=
by
  sorry

end measure_of_one_interior_angle_of_regular_nonagon_is_140_l37_37517


namespace more_red_balls_l37_37266

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l37_37266


namespace min_chord_length_intercepted_line_eq_l37_37576

theorem min_chord_length_intercepted_line_eq (m : ℝ)
  (hC : ∀ (x y : ℝ), (x-1)^2 + (y-1)^2 = 16)
  (hL : ∀ (x y : ℝ), (2*m-1)*x + (m-1)*y - 3*m + 1 = 0)
  : ∃ x y : ℝ, x - 2*y - 4 = 0 := sorry

end min_chord_length_intercepted_line_eq_l37_37576


namespace louis_current_age_l37_37416

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end louis_current_age_l37_37416


namespace total_brown_mms_3rd_4th_bags_l37_37111

def brown_mms_in_bags := (9 : ℕ) + (12 : ℕ) + (3 : ℕ)

def total_bags := 5

def average_mms_per_bag := 8

theorem total_brown_mms_3rd_4th_bags (x y : ℕ) 
  (h1 : brown_mms_in_bags + x + y = average_mms_per_bag * total_bags) : 
  x + y = 16 :=
by
  have h2 : brown_mms_in_bags + x + y = 40 := by sorry
  sorry

end total_brown_mms_3rd_4th_bags_l37_37111


namespace midterm_exam_2022_option_probabilities_l37_37256

theorem midterm_exam_2022_option_probabilities :
  let no_option := 4
  let prob_distribution := (1 : ℚ) / 3
  let combs_with_4_correct := 1
  let combs_with_3_correct := 4
  let combs_with_2_correct := 6
  let prob_4_correct := prob_distribution
  let prob_3_correct := prob_distribution / combs_with_3_correct
  let prob_2_correct := prob_distribution / combs_with_2_correct
  
  let prob_B_correct := combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct
  let prob_C_given_event_A := combs_with_3_correct * prob_3_correct / (combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct)
  
  (prob_B_correct > 1 / 2) ∧ (prob_C_given_event_A = 1 / 3) :=
by 
  sorry

end midterm_exam_2022_option_probabilities_l37_37256


namespace smallest_x_plus_y_l37_37218

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37218


namespace probability_xi_l37_37771

open ProbabilityTheory

noncomputable def xi_dist (a : ℝ) : ℕ → ℝ := 
λ k, if k ∈ {1, 2, 3, 4, 5} then a * k else 0

theorem probability_xi :
  ∃ (a : ℝ), (a * (1 + 2 + 3 + 4 + 5) = 1) ∧
  (pmf.of_fn (xi_dist a)).prob (λ x, (1/10 < x) ∧ (x < 1/2)) = 1/5 :=
by
  sorry

end probability_xi_l37_37771


namespace gcd_problem_l37_37387

theorem gcd_problem (x : ℤ) (h : ∃ k, x = 2 * 2027 * k) :
  Int.gcd (3 * x ^ 2 + 47 * x + 101) (x + 23) = 1 :=
sorry

end gcd_problem_l37_37387


namespace birthday_friends_count_l37_37478

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37478


namespace sum_of_arithmetic_sequence_l37_37384

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ):
  (S 4 = S 8 - S 4) →
  (S 4 = S 12 - S 8) →
  (S 4 = S 16 - S 12) →
  S 16 / S 4 = 10 :=
by
  intros h1 h2 h3
  sorry

end sum_of_arithmetic_sequence_l37_37384


namespace Vanya_Journey_Five_times_Anya_Journey_l37_37002

theorem Vanya_Journey_Five_times_Anya_Journey (a_start a_end v_start v_end : ℕ)
  (h1 : a_start = 1) (h2 : a_end = 2) (h3 : v_start = 1) (h4 : v_end = 6) :
  (v_end - v_start) = 5 * (a_end - a_start) :=
  sorry

end Vanya_Journey_Five_times_Anya_Journey_l37_37002


namespace total_vehicles_correct_l37_37245

def num_trucks : ℕ := 20
def num_tanks (num_trucks : ℕ) : ℕ := 5 * num_trucks
def total_vehicles (num_trucks : ℕ) (num_tanks : ℕ) : ℕ := num_trucks + num_tanks

theorem total_vehicles_correct : total_vehicles num_trucks (num_tanks num_trucks) = 120 := by
  sorry

end total_vehicles_correct_l37_37245


namespace cotton_needed_l37_37092

noncomputable def feet_of_cotton_per_teeshirt := 4
noncomputable def number_of_teeshirts := 15

theorem cotton_needed : feet_of_cotton_per_teeshirt * number_of_teeshirts = 60 := 
by 
  sorry

end cotton_needed_l37_37092


namespace eval_expr_at_2_l37_37651

def expr (x : ℝ) : ℝ := (3 * x + 4)^2

theorem eval_expr_at_2 : expr 2 = 100 :=
by sorry

end eval_expr_at_2_l37_37651


namespace symm_diff_complement_l37_37871

variable {U : Type} -- Universal set U
variable (A B : Set U) -- Sets A and B

-- Definition of symmetric difference
def symm_diff (X Y : Set U) : Set U := (X ∪ Y) \ (X ∩ Y)

theorem symm_diff_complement (A B : Set U) :
  (symm_diff A B) = (symm_diff (Aᶜ) (Bᶜ)) :=
sorry

end symm_diff_complement_l37_37871


namespace quadratic_negativity_cond_l37_37864

theorem quadratic_negativity_cond {x m k : ℝ} :
  (∀ x, x^2 - m * x - k + m < 0) ↔ k > m - (m^2 / 4) :=
sorry

end quadratic_negativity_cond_l37_37864


namespace max_value_f_l37_37724

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

theorem max_value_f (x : ℝ) (h : -4 < x ∧ x < 1) : ∃ y, f y = -1 ∧ (∀ z, f z ≤ f y) :=
by 
  sorry

end max_value_f_l37_37724


namespace primes_sum_divisible_by_60_l37_37915

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_sum_divisible_by_60 (p q r s : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (hs : is_prime s) 
  (h_cond1 : 5 < p) 
  (h_cond2 : p < q) 
  (h_cond3 : q < r) 
  (h_cond4 : r < s) 
  (h_cond5 : s < p + 10) : 
  (p + q + r + s) % 60 = 0 :=
sorry

end primes_sum_divisible_by_60_l37_37915


namespace rate_of_discount_l37_37664

theorem rate_of_discount (marked_price selling_price : ℝ) (h1 : marked_price = 200) (h2 : selling_price = 120) : 
  ((marked_price - selling_price) / marked_price) * 100 = 40 :=
by
  sorry

end rate_of_discount_l37_37664


namespace find_divisor_l37_37286

theorem find_divisor (x : ℤ) : 83 = 9 * x + 2 → x = 9 :=
by
  sorry

end find_divisor_l37_37286


namespace additional_money_required_l37_37691

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l37_37691


namespace solve_for_f_sqrt_2_l37_37196

theorem solve_for_f_sqrt_2 (f : ℝ → ℝ) (h : ∀ x, f x = 2 / (2 - x)) : f (Real.sqrt 2) = 2 + Real.sqrt 2 :=
by
  sorry

end solve_for_f_sqrt_2_l37_37196


namespace trig_identity_solution_l37_37195

theorem trig_identity_solution (α : ℝ) (h : Real.tan α = -1 / 2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1 / 3 :=
by
  sorry

end trig_identity_solution_l37_37195


namespace ratio_second_third_l37_37642

theorem ratio_second_third (S T : ℕ) (h_sum : 200 + S + T = 500) (h_third : T = 100) : S / T = 2 := by
  sorry

end ratio_second_third_l37_37642


namespace simplify_logarithmic_expression_l37_37931

theorem simplify_logarithmic_expression :
  (1 / (Real.logb 12 3 + 1) + 1 / (Real.logb 8 2 + 1) + 1 / (Real.logb 18 9 + 1) = 1) :=
sorry

end simplify_logarithmic_expression_l37_37931


namespace roger_candies_left_l37_37113

theorem roger_candies_left (initial_candies : ℕ) (to_stephanie : ℕ) (to_john : ℕ) (to_emily : ℕ) : 
  initial_candies = 350 ∧ to_stephanie = 45 ∧ to_john = 25 ∧ to_emily = 18 → 
  initial_candies - (to_stephanie + to_john + to_emily) = 262 :=
by
  sorry

end roger_candies_left_l37_37113


namespace function_machine_output_l37_37254

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  let step2 := if step1 > 25 then step1 - 7 else step1 + 10
  step2

theorem function_machine_output : function_machine 15 = 38 :=
by
  sorry

end function_machine_output_l37_37254


namespace sum_of_triangles_l37_37789

def triangle (a b c : ℕ) : ℕ :=
  (a * b) + c

theorem sum_of_triangles : 
  triangle 4 2 3 + triangle 5 3 2 = 28 :=
by
  sorry

end sum_of_triangles_l37_37789


namespace louis_current_age_l37_37413

/-- 
  In 6 years, Carla will be 30 years old. 
  The sum of the current ages of Carla and Louis is 55. 
  Prove that Louis is currently 31 years old.
--/
theorem louis_current_age (C L : ℕ) 
  (h1 : C + 6 = 30) 
  (h2 : C + L = 55) 
  : L = 31 := 
sorry

end louis_current_age_l37_37413


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37815

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37815


namespace range_of_a_if_odd_symmetric_points_l37_37376

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a

theorem range_of_a_if_odd_symmetric_points (a : ℝ): 
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ f x₀ a = -f (-x₀) a) → (1 < a) :=
by 
  sorry

end range_of_a_if_odd_symmetric_points_l37_37376


namespace minimum_value_problem1_minimum_value_problem2_l37_37827

theorem minimum_value_problem1 (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y >= 6 := 
sorry

theorem minimum_value_problem2 (x : ℝ) (h : x > 1) : 
  ∃ y, y = (x^2 + 8) / (x - 1) ∧ y >= 8 := 
sorry

end minimum_value_problem1_minimum_value_problem2_l37_37827


namespace students_wearing_other_colors_l37_37904

-- Definitions based on conditions
def total_students := 700
def percentage_blue := 45 / 100
def percentage_red := 23 / 100
def percentage_green := 15 / 100

-- The proof problem statement
theorem students_wearing_other_colors :
  (total_students - total_students * (percentage_blue + percentage_red + percentage_green)) = 119 :=
by
  sorry

end students_wearing_other_colors_l37_37904


namespace total_green_ducks_percentage_l37_37419

def ducks_in_park_A : ℕ := 200
def green_percentage_A : ℕ := 25

def ducks_in_park_B : ℕ := 350
def green_percentage_B : ℕ := 20

def ducks_in_park_C : ℕ := 120
def green_percentage_C : ℕ := 50

def ducks_in_park_D : ℕ := 60
def green_percentage_D : ℕ := 25

def ducks_in_park_E : ℕ := 500
def green_percentage_E : ℕ := 30

theorem total_green_ducks_percentage (green_ducks_A green_ducks_B green_ducks_C green_ducks_D green_ducks_E total_ducks : ℕ)
  (h_A : green_ducks_A = ducks_in_park_A * green_percentage_A / 100)
  (h_B : green_ducks_B = ducks_in_park_B * green_percentage_B / 100)
  (h_C : green_ducks_C = ducks_in_park_C * green_percentage_C / 100)
  (h_D : green_ducks_D = ducks_in_park_D * green_percentage_D / 100)
  (h_E : green_ducks_E = ducks_in_park_E * green_percentage_E / 100)
  (h_total_ducks : total_ducks = ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E) :
  (green_ducks_A + green_ducks_B + green_ducks_C + green_ducks_D + green_ducks_E) * 100 / total_ducks = 2805 / 100 :=
by sorry

end total_green_ducks_percentage_l37_37419


namespace rectangle_divided_into_13_squares_l37_37983

-- Define the conditions
variables {a b s : ℝ} (m n : ℕ)

-- Mathematical equivalent proof problem Lean statement
theorem rectangle_divided_into_13_squares (h : a * b = 13 * s^2)
  (hm : a = m * s) (hn : b = n * s) (hmn : m * n = 13) :
  a / b = 13 ∨ b / a = 13 :=
begin
  sorry
end

end rectangle_divided_into_13_squares_l37_37983


namespace rhombus_area_l37_37779

-- Define the given conditions as parameters
variables (EF GH : ℝ) -- Sides of the rhombus
variables (d1 d2 : ℝ) -- Diagonals of the rhombus

-- Statement of the theorem
theorem rhombus_area
  (rhombus_EFGH : ∀ (EF GH : ℝ), EF = GH)
  (perimeter_EFGH : 4 * EF = 40)
  (diagonal_EG_length : d1 = 16)
  (d1_half : d1 / 2 = 8)
  (side_length : EF = 10)
  (pythagorean_theorem : EF^2 = (d1 / 2)^2 + (d2 / 2)^2)
  (calculate_FI : d2 / 2 = 6)
  (diagonal_FG_length : d2 = 12) :
  (1 / 2) * d1 * d2 = 96 :=
sorry

end rhombus_area_l37_37779


namespace matt_climbing_speed_l37_37912

theorem matt_climbing_speed :
  ∃ (x : ℝ), (12 * 7 = 7 * x + 42) ∧ x = 6 :=
by {
  sorry
}

end matt_climbing_speed_l37_37912


namespace cost_of_each_barbell_l37_37430

theorem cost_of_each_barbell (total_given change_received total_barbells : ℕ)
  (h1 : total_given = 850)
  (h2 : change_received = 40)
  (h3 : total_barbells = 3) :
  (total_given - change_received) / total_barbells = 270 :=
by
  sorry

end cost_of_each_barbell_l37_37430


namespace equation_squares_l37_37083

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l37_37083


namespace remainder_zero_division_l37_37700

theorem remainder_zero_division :
  ∀ x : ℂ, (x^2 - x + 1 = 0) →
    ((x^5 + x^4 - x^3 - x^2 + 1) * (x^3 - 1)) % (x^2 - x + 1) = 0 :=
by sorry

end remainder_zero_division_l37_37700


namespace first_term_is_sqrt9_l37_37645

noncomputable def geometric_first_term (a r : ℝ) : ℝ :=
by
  have h1 : a * r^2 = 3 := by sorry
  have h2 : a * r^4 = 27 := by sorry
  have h3 : (a * r^4) / (a * r^2) = 27 / 3 := by sorry
  have h4 : r^2 = 9 := by sorry
  have h5 : r = 3 ∨ r = -3 := by sorry
  have h6 : (a * 9) = 3 := by sorry
  have h7 : a = 1/3 := by sorry
  exact a

theorem first_term_is_sqrt9 : geometric_first_term 3 9 = 3 :=
by
  sorry

end first_term_is_sqrt9_l37_37645


namespace part1_part2_l37_37997

variable {α : Type*} [LinearOrderedField α]

-- Definitions based on given problem conditions.
def arithmetic_seq(a_n : ℕ → α) := ∃ a1 d, ∀ n, a_n n = a1 + ↑(n - 1) * d

noncomputable def a10_seq := (30 : α)
noncomputable def a20_seq := (50 : α)

-- Theorem statements to prove:
theorem part1 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq) :
  ∀ n, a_n n = 2 * ↑n + 10 := sorry

theorem part2 {a_n : ℕ → α} (h : arithmetic_seq a_n) (h10 : a_n 10 = a10_seq) (h20 : a_n 20 = a20_seq)
  (Sn : α) (hSn : Sn = 242) :
  ∃ n, Sn = (↑n / 2) * (2 * 12 + (↑n - 1) * 2) ∧ n = 11 := sorry

end part1_part2_l37_37997


namespace boat_license_combinations_l37_37545

theorem boat_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let digit_positions := 5
  (letter_choices * (digit_choices ^ digit_positions)) = 300000 :=
  sorry

end boat_license_combinations_l37_37545


namespace quadratic_to_vertex_form_addition_l37_37072

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l37_37072


namespace problem_statement_l37_37616

variable (a b c d : ℝ)

theorem problem_statement :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ (9 / 16) * (a - b) * (b - c) * (c - d) * (d - a) :=
sorry

end problem_statement_l37_37616


namespace second_quadrant_y_value_l37_37899

theorem second_quadrant_y_value :
  ∀ (b : ℝ), (-3, b).2 > 0 → b = 2 :=
by
  sorry

end second_quadrant_y_value_l37_37899


namespace candy_in_each_box_l37_37289

theorem candy_in_each_box (C K : ℕ) (h1 : 6 * C + 4 * K = 90) (h2 : C = K) : C = 9 :=
by
  -- Proof will go here
  sorry

end candy_in_each_box_l37_37289


namespace total_selling_price_l37_37167

theorem total_selling_price (cost1 cost2 cost3 : ℕ) (profit1 profit2 profit3 : ℚ) 
  (h1 : cost1 = 280) (h2 : cost2 = 350) (h3 : cost3 = 500) 
  (h4 : profit1 = 30) (h5 : profit2 = 45) (h6 : profit3 = 25) : 
  (cost1 + (profit1 / 100) * cost1) + (cost2 + (profit2 / 100) * cost2) + (cost3 + (profit3 / 100) * cost3) = 1496.5 := by
  sorry

end total_selling_price_l37_37167


namespace floor_ceil_eq_l37_37386

theorem floor_ceil_eq (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 :=
by
  sorry

end floor_ceil_eq_l37_37386


namespace quadratic_inequality_solution_l37_37798

theorem quadratic_inequality_solution : 
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3 / 2} :=
by
  sorry

end quadratic_inequality_solution_l37_37798


namespace books_left_l37_37109

variable (initialBooks : ℕ) (soldBooks : ℕ) (remainingBooks : ℕ)

-- Conditions
def initial_conditions := initialBooks = 136 ∧ soldBooks = 109

-- Question: Proving the remaining books after the sale
theorem books_left (initial_conditions : initialBooks = 136 ∧ soldBooks = 109) : remainingBooks = 27 :=
by
  cases initial_conditions
  sorry

end books_left_l37_37109


namespace add_to_fraction_eq_l37_37315

theorem add_to_fraction_eq (n : ℤ) (h : (4 + n) / (7 + n) = 3 / 4) : n = 5 :=
by sorry

end add_to_fraction_eq_l37_37315


namespace sum_of_number_and_reverse_l37_37125

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end sum_of_number_and_reverse_l37_37125


namespace sum_of_cubes_condition_l37_37768

theorem sum_of_cubes_condition (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := 
by
  sorry

end sum_of_cubes_condition_l37_37768


namespace sequence_formula_l37_37589

theorem sequence_formula (a : ℕ → ℤ) (h0 : a 0 = 1) (h1 : a 1 = 5)
    (h_rec : ∀ n, n ≥ 2 → a n = (2 * (a (n - 1))^2 - 3 * (a (n - 1)) - 9) / (2 * a (n - 2))) :
  ∀ n, a n = 2^(n + 2) - 3 :=
by
  intros
  sorry

end sequence_formula_l37_37589


namespace product_of_roots_l37_37809

theorem product_of_roots :
  (Real.root 256 4) * (Real.root 8 3) * (Real.sqrt 16) = 32 :=
sorry

end product_of_roots_l37_37809


namespace sin_beta_value_l37_37990

variable {α β : ℝ}
variable (h₁ : 0 < α ∧ α < β ∧ β < π / 2)
variable (h₂ : Real.sin α = 3 / 5)
variable (h₃ : Real.cos (β - α) = 12 / 13)

theorem sin_beta_value : Real.sin β = 56 / 65 :=
by
  sorry

end sin_beta_value_l37_37990


namespace max_value_of_f_l37_37795

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem: the maximum value of f(x) is 4
theorem max_value_of_f : ∃ x : ℝ, f x = 4 := sorry

end max_value_of_f_l37_37795


namespace largest_of_a_b_c_l37_37727

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.sin (Real.pi / 8)

theorem largest_of_a_b_c : b = max (max a b) c :=
by
  have ha : a = 1 / 2 := rfl
  have hb : b = Real.log 3 / Real.log 4 := rfl
  have hc : c = Real.sin (Real.pi / 8) := rfl
  sorry

end largest_of_a_b_c_l37_37727


namespace triangle_area_0_0_0_5_7_12_l37_37947

theorem triangle_area_0_0_0_5_7_12 : 
    let base := 5
    let height := 7
    let area := (1 / 2) * base * height
    area = 17.5 := 
by
    sorry

end triangle_area_0_0_0_5_7_12_l37_37947


namespace sum_of_numbers_l37_37128

-- Definitions that come directly from the conditions
def product_condition (A B : ℕ) : Prop := A * B = 9375
def quotient_condition (A B : ℕ) : Prop := A / B = 15

-- Theorem that proves the sum of A and B is 400, based on the given conditions
theorem sum_of_numbers (A B : ℕ) (h1 : product_condition A B) (h2 : quotient_condition A B) : A + B = 400 :=
sorry

end sum_of_numbers_l37_37128


namespace range_of_m_l37_37052

noncomputable def f (x : ℝ) : ℝ :=
  if x >= -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > m^2 - m) ↔ -1 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l37_37052


namespace simplify_expression_l37_37016

theorem simplify_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 :=
by
  sorry

end simplify_expression_l37_37016


namespace ben_remaining_money_l37_37684

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end ben_remaining_money_l37_37684


namespace value_of_f_12_l37_37668

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end value_of_f_12_l37_37668


namespace unsuitable_temperature_for_refrigerator_l37_37800

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l37_37800


namespace dadAgeWhenXiaoHongIs7_l37_37852

variable {a : ℕ}

-- Condition: Dad's age is given as 'a'
-- Condition: Dad's age is 4 times plus 3 years more than Xiao Hong's age
def xiaoHongAge (a : ℕ) : ℕ := (a - 3) / 4

theorem dadAgeWhenXiaoHongIs7 : xiaoHongAge a = 7 → a = 31 := by
  intro h
  have h1 : a - 3 = 28 := by sorry   -- Algebraic manipulation needed
  have h2 : a = 31 := by sorry       -- Algebraic manipulation needed
  exact h2

end dadAgeWhenXiaoHongIs7_l37_37852


namespace robins_fraction_l37_37602

theorem robins_fraction (B R J : ℕ) (h1 : R + J = B)
  (h2 : 2/3 * (R : ℚ) + 1/3 * (J : ℚ) = 7/15 * (B : ℚ)) :
  (R : ℚ) / B = 2/5 :=
by
  sorry

end robins_fraction_l37_37602


namespace number_of_people_in_each_van_l37_37160

theorem number_of_people_in_each_van (x : ℕ) 
  (h1 : 6 * x + 8 * 18 = 180) : x = 6 :=
by sorry

end number_of_people_in_each_van_l37_37160


namespace range_of_reciprocal_sum_l37_37234

theorem range_of_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : a + b = 1) :
  ∃ c > 4, ∀ x, x = (1 / a + 1 / b) → c < x :=
sorry

end range_of_reciprocal_sum_l37_37234


namespace total_pieces_on_chessboard_l37_37346

-- Given conditions about initial chess pieces and lost pieces.
def initial_pieces_each : Nat := 16
def pieces_lost_arianna : Nat := 3
def pieces_lost_samantha : Nat := 9

-- The remaining pieces for each player.
def remaining_pieces_arianna : Nat := initial_pieces_each - pieces_lost_arianna
def remaining_pieces_samantha : Nat := initial_pieces_each - pieces_lost_samantha

-- The total remaining pieces on the chessboard.
def total_remaining_pieces : Nat := remaining_pieces_arianna + remaining_pieces_samantha

-- The theorem to prove
theorem total_pieces_on_chessboard : total_remaining_pieces = 20 :=
by
  sorry

end total_pieces_on_chessboard_l37_37346


namespace expected_interval_is_correct_l37_37153

-- Define the travel times via northern and southern routes
def travel_time_north : ℝ := 17
def travel_time_south : ℝ := 11

-- Define the average time difference between train arrivals
noncomputable def avg_time_diff : ℝ := 1.25

-- The average time difference for traveling from home to work versus work to home
noncomputable def time_diff_home_to_work : ℝ := 1

-- Define the expected interval between trains
noncomputable def expected_interval_between_trains := 3

-- Proof problem statement
theorem expected_interval_is_correct :
  ∃ (T : ℝ), (T = expected_interval_between_trains)
  → (travel_time_north - travel_time_south + 2 * avg_time_diff = time_diff_home_to_work)
  → (T = 3) := 
by
  use 3 
  intro h1 h2
  sorry

end expected_interval_is_correct_l37_37153


namespace friends_attended_birthday_l37_37447

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37447


namespace number_of_friends_l37_37461

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l37_37461


namespace sequence_2007th_term_is_85_l37_37635

noncomputable def sum_of_square_of_digits (n : ℕ) : ℕ :=
(n.digits 10).map (λ d, d * d).sum

noncomputable def sequence_term : ℕ → ℕ
| 0 := 2007
| (n+1) := sum_of_square_of_digits (sequence_term n)

theorem sequence_2007th_term_is_85 : sequence_term 2007 = 85 := 
sorry

end sequence_2007th_term_is_85_l37_37635


namespace quadratic_to_vertex_form_addition_l37_37073

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l37_37073


namespace lemon_more_valuable_than_banana_l37_37553

variable {L B A V : ℝ}

theorem lemon_more_valuable_than_banana
  (h1 : L + B = 2 * A + 23 * V)
  (h2 : 3 * L = 2 * B + 2 * A + 14 * V) :
  L > B := by
  sorry

end lemon_more_valuable_than_banana_l37_37553


namespace number_of_digits_in_sum_l37_37062

def is_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem number_of_digits_in_sum (C D : ℕ) (hC : is_digit C) (hD : is_digit D) :
  let n1 := 98765
  let n2 := C * 1000 + 433
  let n3 := D * 100 + 22
  let s := n1 + n2 + n3
  100000 ≤ s ∧ s < 1000000 :=
by {
  sorry
}

end number_of_digits_in_sum_l37_37062


namespace birthday_friends_count_l37_37476

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l37_37476


namespace burmese_pythons_required_l37_37406

theorem burmese_pythons_required (single_python_rate : ℕ) (total_alligators : ℕ) (total_weeks : ℕ) (required_pythons : ℕ) :
  single_python_rate = 1 →
  total_alligators = 15 →
  total_weeks = 3 →
  required_pythons = total_alligators / total_weeks →
  required_pythons = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at *
  simp at h4
  sorry

end burmese_pythons_required_l37_37406


namespace faith_weekly_earnings_l37_37562

theorem faith_weekly_earnings :
  let hourly_pay := 13.50
  let regular_hours_per_day := 8
  let workdays_per_week := 5
  let overtime_hours_per_day := 2
  let regular_pay_per_day := hourly_pay * regular_hours_per_day
  let regular_pay_per_week := regular_pay_per_day * workdays_per_week
  let overtime_pay_per_day := hourly_pay * overtime_hours_per_day
  let overtime_pay_per_week := overtime_pay_per_day * workdays_per_week
  let total_weekly_earnings := regular_pay_per_week + overtime_pay_per_week
  total_weekly_earnings = 675 := 
  by
    sorry

end faith_weekly_earnings_l37_37562


namespace sanoop_initial_tshirts_l37_37785

theorem sanoop_initial_tshirts (n : ℕ) (T : ℕ) 
(avg_initial : T = n * 526) 
(avg_remaining : T - 673 = (n - 1) * 505) 
(avg_returned : 673 = 673) : 
n = 8 := 
by 
  sorry

end sanoop_initial_tshirts_l37_37785


namespace necessary_and_sufficient_condition_l37_37235

theorem necessary_and_sufficient_condition (p q : Prop) 
  (hpq : p → q) (hqp : q → p) : 
  (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l37_37235


namespace lisa_speed_correct_l37_37859

def eugene_speed := 5

def carlos_speed := (3 / 4) * eugene_speed

def lisa_speed := (4 / 3) * carlos_speed

theorem lisa_speed_correct : lisa_speed = 5 := by
  sorry

end lisa_speed_correct_l37_37859


namespace maximize_area_minimize_length_l37_37674

-- Problem 1: Prove maximum area of the enclosure
theorem maximize_area (x y : ℝ) (h : x + 2 * y = 36) : 18 * 9 = 162 :=
by
  sorry

-- Problem 2: Prove the minimum length of steel wire mesh
theorem minimize_length (x y : ℝ) (h1 : x * y = 32) : 8 + 2 * 4 = 16 :=
by
  sorry

end maximize_area_minimize_length_l37_37674


namespace smallest_number_of_2_by_3_rectangles_l37_37139

def area_2_by_3_rectangle : Int := 2 * 3

def smallest_square_area_multiple_of_6 : Int :=
  let side_length := 6
  side_length * side_length

def number_of_rectangles_to_cover_square (square_area : Int) (rectangle_area : Int) : Int :=
  square_area / rectangle_area

theorem smallest_number_of_2_by_3_rectangles :
  number_of_rectangles_to_cover_square smallest_square_area_multiple_of_6 area_2_by_3_rectangle = 6 := by
  sorry

end smallest_number_of_2_by_3_rectangles_l37_37139


namespace minimum_time_to_cook_l37_37956

def wash_pot_fill_water : ℕ := 2
def wash_vegetables : ℕ := 3
def prepare_noodles_seasonings : ℕ := 2
def boil_water : ℕ := 7
def cook_noodles_vegetables : ℕ := 3

theorem minimum_time_to_cook : wash_pot_fill_water + boil_water + cook_noodles_vegetables = 12 :=
by
  sorry

end minimum_time_to_cook_l37_37956


namespace price_for_3years_service_l37_37115

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end price_for_3years_service_l37_37115


namespace Frank_worked_days_l37_37572

theorem Frank_worked_days
  (h_per_day : ℕ) (total_hours : ℕ) (d : ℕ) 
  (h_day_def : h_per_day = 8) 
  (total_hours_def : total_hours = 32) 
  (d_def : d = total_hours / h_per_day) : 
  d = 4 :=
by 
  rw [total_hours_def, h_day_def] at d_def
  exact d_def

end Frank_worked_days_l37_37572


namespace minimum_xy_l37_37826

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : xy ≥ 64 :=
sorry

end minimum_xy_l37_37826


namespace hours_learning_english_each_day_l37_37363

theorem hours_learning_english_each_day (total_hours : ℕ) (days : ℕ) (learning_hours_per_day : ℕ) 
  (h1 : total_hours = 12) 
  (h2 : days = 2) 
  (h3 : total_hours = learning_hours_per_day * days) : 
  learning_hours_per_day = 6 := 
by
  sorry

end hours_learning_english_each_day_l37_37363


namespace paolo_sevilla_birthday_l37_37484

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l37_37484


namespace stone_count_150_equals_8_l37_37861

theorem stone_count_150_equals_8 :
  ∃ n, 1 ≤ n ∧ n ≤ 15 ∧ (150 % 28 = 22) ∧ (22 corresponds to stone number 8) :=
by
  -- Conditions for the equivalence of position under the counting pattern
  have h1 : 150 % 28 = 22 := by sorry
  -- Detailed proof and rigorous definition of counting pattern skipped
  sorry

end stone_count_150_equals_8_l37_37861


namespace largest_integer_n_apples_l37_37337

theorem largest_integer_n_apples (t : ℕ) (a : ℕ → ℕ) (h1 : t = 150) 
    (h2 : ∀ i : ℕ, 100 ≤ a i ∧ a i ≤ 130) :
  ∃ n : ℕ, n = 5 ∧ (∀ i j : ℕ, a i = a j → i = j → 5 ≤ i ∧ 5 ≤ j) :=
by
  sorry

end largest_integer_n_apples_l37_37337


namespace max_ab_l37_37225

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_l37_37225


namespace additional_money_required_l37_37692

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l37_37692


namespace liars_are_C_and_D_l37_37345
open Classical 

-- We define inhabitants and their statements
inductive Inhabitant
| A | B | C | D

open Inhabitant

axiom is_liar : Inhabitant → Prop

-- Statements by the inhabitants:
-- A: "At least one of us is a liar."
-- B: "At least two of us are liars."
-- C: "At least three of us are liars."
-- D: "None of us are liars."

def statement_A : Prop := is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D
def statement_B : Prop := (is_liar A ∧ is_liar B) ∨ (is_liar A ∧ is_liar C) ∨ (is_liar A ∧ is_liar D) ∨
                          (is_liar B ∧ is_liar C) ∨ (is_liar B ∧ is_liar D) ∨ (is_liar C ∧ is_liar D)
def statement_C : Prop := (is_liar A ∧ is_liar B ∧ is_liar C) ∨ (is_liar A ∧ is_liar B ∧ is_liar D) ∨
                          (is_liar A ∧ is_liar C ∧ is_liar D) ∨ (is_liar B ∧ is_liar C ∧ is_liar D)
def statement_D : Prop := ¬(is_liar A ∨ is_liar B ∨ is_liar C ∨ is_liar D)

-- Given that there are some liars
axiom some_liars_exist : ∃ x, is_liar x

-- Lean proof statement
theorem liars_are_C_and_D : is_liar C ∧ is_liar D ∧ ¬(is_liar A) ∧ ¬(is_liar B) :=
by
  sorry

end liars_are_C_and_D_l37_37345


namespace perimeter_difference_l37_37133

-- Definitions for the conditions
def num_stakes_sheep : ℕ := 96
def interval_sheep : ℕ := 10
def num_stakes_horse : ℕ := 82
def interval_horse : ℕ := 20

-- Definition for the perimeters
def perimeter_sheep : ℕ := num_stakes_sheep * interval_sheep
def perimeter_horse : ℕ := num_stakes_horse * interval_horse

-- Definition for the target difference
def target_difference : ℕ := 680

-- The theorem stating the proof problem
theorem perimeter_difference : perimeter_horse - perimeter_sheep = target_difference := by
  sorry

end perimeter_difference_l37_37133


namespace tangent_value_prism_QABC_l37_37593

-- Assuming R is the radius of the sphere and considering the given conditions
variables {R x : ℝ} (P Q A B C M H : Type)

-- Given condition: Angle between lateral face and base of prism P-ABC is 45 degrees
def angle_PABC : ℝ := 45
-- Required to prove: tan(angle between lateral face and base of prism Q-ABC) = 4
def tangent_QABC : ℝ := 4

theorem tangent_value_prism_QABC
  (h1 : angle_PABC = 45)
  (h2 : 5 * x - 2 * R = 0) -- Derived condition from the solution
  (h3 : x = 2 * R / 5) -- x, the distance calculation
: tangent_QABC = 4 := by
  sorry

end tangent_value_prism_QABC_l37_37593


namespace each_boy_brought_nine_cups_l37_37803

/--
There are 30 students in Ms. Leech's class. Twice as many girls as boys are in the class.
There are 10 boys in the class and the total number of cups brought by the students 
in the class is 90. Prove that each boy brought 9 cups.
-/
theorem each_boy_brought_nine_cups (students girls boys cups : ℕ) 
  (h1 : students = 30) 
  (h2 : girls = 2 * boys) 
  (h3 : boys = 10) 
  (h4 : cups = 90) 
  : cups / boys = 9 := 
sorry

end each_boy_brought_nine_cups_l37_37803


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37812

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l37_37812


namespace friends_attended_birthday_l37_37445

variable {n : ℕ}

theorem friends_attended_birthday (h1 : ∀ total_bill : ℕ, total_bill = 12 * (n + 2))
(h2 : ∀ total_bill : ℕ, total_bill = 16 * n) : n = 6 :=
by
  sorry

end friends_attended_birthday_l37_37445


namespace a_capital_used_l37_37958

theorem a_capital_used (C P x : ℕ) (h_b_contributes : 3 * C / 4 - C ≥ 0) 
(h_b_receives : 2 * P / 3 - P ≥ 0) 
(h_b_money_used : 10 > 0) 
(h_ratio : 1 / 2 = x / 30) 
: x = 15 :=
sorry

end a_capital_used_l37_37958


namespace speed_of_current_l37_37830

variables (b c : ℝ)

theorem speed_of_current (h1 : b + c = 12) (h2 : b - c = 4) : c = 4 :=
sorry

end speed_of_current_l37_37830


namespace striped_octopus_has_eight_legs_l37_37070

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l37_37070


namespace smallest_x_plus_y_l37_37217

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l37_37217


namespace convert_polar_to_rectangular_l37_37354

theorem convert_polar_to_rectangular :
  ∀ r θ : ℝ, r = 4 ∧ θ = (3 * Real.pi / 4) →
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  x = -2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 :=
by
  intros r θ h
  cases h
  have h1 : Real.cos (3 * Real.pi / 4) = -1 / Real.sqrt 2 := sorry
  have h2 : Real.sin (3 * Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  simp [h1, h2]
  split
  case x =>
    sorry
  case y =>
    sorry

end convert_polar_to_rectangular_l37_37354


namespace find_a_and_b_l37_37434

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_and_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 40) : a + b = 7 :=
by
  sorry

end find_a_and_b_l37_37434


namespace fishAddedIs15_l37_37612

-- Define the number of fish Jason starts with
def initialNumberOfFish : ℕ := 6

-- Define the fish counts on each day
def fishOnDay2 := 2 * initialNumberOfFish
def fishOnDay3 := 2 * fishOnDay2 - (1 / 3 : ℚ) * (2 * fishOnDay2)
def fishOnDay4 := 2 * fishOnDay3
def fishOnDay5 := 2 * fishOnDay4 - (1 / 4 : ℚ) * (2 * fishOnDay4)
def fishOnDay6 := 2 * fishOnDay5
def fishOnDay7 := 2 * fishOnDay6

-- Define the total fish on the seventh day after adding some fish
def totalFishOnDay7 := 207

-- Define the number of fish Jason added on the seventh day
def fishAddedOnDay7 := totalFishOnDay7 - fishOnDay7

-- Prove that the number of fish Jason added on the seventh day is 15
theorem fishAddedIs15 : fishAddedOnDay7 = 15 := sorry

end fishAddedIs15_l37_37612


namespace total_money_together_is_l37_37349

def Sam_has : ℚ := 750.50
def Billy_has (S : ℚ) : ℚ := 4.5 * S - 345.25
def Lila_has (B S : ℚ) : ℚ := 2.25 * (B - S)
def Total_money (S B L : ℚ) : ℚ := S + B + L

theorem total_money_together_is :
  Total_money Sam_has (Billy_has Sam_has) (Lila_has (Billy_has Sam_has) Sam_has) = 8915.88 :=
by sorry

end total_money_together_is_l37_37349


namespace lcm_of_8_12_15_l37_37950

theorem lcm_of_8_12_15 : Nat.lcm 8 (Nat.lcm 12 15) = 120 :=
by
  -- This is where the proof steps would go
  sorry

end lcm_of_8_12_15_l37_37950


namespace cubic_coeff_relationship_l37_37020

theorem cubic_coeff_relationship (a b c d u v w : ℝ) 
  (h_eq : a * (u^3) + b * (u^2) + c * u + d = 0)
  (h_vieta1 : u + v + w = -(b / a)) 
  (h_vieta2 : u * v + u * w + v * w = c / a) 
  (h_vieta3 : u * v * w = -d / a) 
  (h_condition : u + v = u * v) :
  (c + d) * (b + c + d) = a * d :=
by 
  sorry

end cubic_coeff_relationship_l37_37020


namespace annie_initial_money_l37_37847

theorem annie_initial_money
  (hamburger_price : ℕ := 4)
  (milkshake_price : ℕ := 3)
  (num_hamburgers : ℕ := 8)
  (num_milkshakes : ℕ := 6)
  (money_left : ℕ := 70)
  (total_cost_hamburgers : ℕ := num_hamburgers * hamburger_price)
  (total_cost_milkshakes : ℕ := num_milkshakes * milkshake_price)
  (total_cost : ℕ := total_cost_hamburgers + total_cost_milkshakes)
  : num_hamburgers * hamburger_price + num_milkshakes * milkshake_price + money_left = 120 :=
by
  -- proof part skipped
  sorry

end annie_initial_money_l37_37847


namespace suitable_survey_is_D_l37_37521

-- Define the surveys
def survey_A := "Survey on the viewing of the movie 'The Long Way Home' by middle school students in our city"
def survey_B := "Survey on the germination rate of a batch of rose seeds"
def survey_C := "Survey on the water quality of the Jialing River"
def survey_D := "Survey on the health codes of students during the epidemic"

-- Define what it means for a survey to be suitable for a comprehensive census
def suitable_for_census (survey : String) : Prop :=
  survey = survey_D

-- Define the main theorem statement
theorem suitable_survey_is_D : suitable_for_census survey_D :=
by
  -- We assume sorry here to skip the proof
  sorry

end suitable_survey_is_D_l37_37521


namespace tangent_and_normal_lines_l37_37661

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l37_37661


namespace more_red_balls_l37_37264

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end more_red_balls_l37_37264


namespace intersection_of_A_and_B_l37_37101

variable (A : Set ℕ) (B : Set ℕ)

axiom h1 : A = {1, 2, 3, 4, 5}
axiom h2 : B = {3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} :=
  by sorry

end intersection_of_A_and_B_l37_37101


namespace perfect_square_proof_l37_37089

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l37_37089


namespace pump_X_time_l37_37322

-- Definitions for the problem conditions.
variables (W : ℝ) (T_x : ℝ) (R_x R_y : ℝ)

-- Condition 1: Rate of pump X
def pump_X_rate := R_x = (W / 2) / T_x

-- Condition 2: Rate of pump Y
def pump_Y_rate := R_y = W / 18

-- Condition 3: Combined rate when both pumps work together for 3 hours to pump the remaining water
def combined_rate := (R_x + R_y) = (W / 2) / 3

-- The statement to prove
theorem pump_X_time : 
  pump_X_rate W T_x R_x →
  pump_Y_rate W R_y →
  combined_rate W R_x R_y →
  T_x = 9 :=
sorry

end pump_X_time_l37_37322


namespace find_number_l37_37818

noncomputable def number_with_point_one_percent (x : ℝ) : Prop :=
  0.1 * x / 100 = 12.356

theorem find_number :
  ∃ x : ℝ, number_with_point_one_percent x ∧ x = 12356 :=
by
  sorry

end find_number_l37_37818


namespace Matt_buys_10_key_chains_l37_37622

theorem Matt_buys_10_key_chains
  (cost_per_keychain_in_pack_of_10 : ℝ)
  (cost_per_keychain_in_pack_of_4 : ℝ)
  (number_of_keychains : ℝ)
  (savings : ℝ)
  (h1 : cost_per_keychain_in_pack_of_10 = 2)
  (h2 : cost_per_keychain_in_pack_of_4 = 3)
  (h3 : savings = 20)
  (h4 : 3 * number_of_keychains - 2 * number_of_keychains = savings) :
  number_of_keychains = 10 := 
by
  sorry

end Matt_buys_10_key_chains_l37_37622


namespace students_left_is_6_l37_37750

-- Start of the year students
def initial_students : ℕ := 11

-- New students arrived during the year
def new_students : ℕ := 42

-- Students at the end of the year
def final_students : ℕ := 47

-- Definition to calculate the number of students who left
def students_left (initial new final : ℕ) : ℕ := (initial + new) - final

-- Statement to prove
theorem students_left_is_6 : students_left initial_students new_students final_students = 6 :=
by
  -- We skip the proof using sorry
  sorry

end students_left_is_6_l37_37750


namespace joe_avg_speed_l37_37270

noncomputable def total_distance : ℝ :=
  420 + 250 + 120 + 65

noncomputable def total_time : ℝ :=
  (420 / 60) + (250 / 50) + (120 / 40) + (65 / 70)

noncomputable def avg_speed : ℝ :=
  total_distance / total_time

theorem joe_avg_speed : avg_speed = 53.67 := by
  sorry

end joe_avg_speed_l37_37270


namespace alpha_range_in_first_quadrant_l37_37991

open Real

theorem alpha_range_in_first_quadrant (k : ℤ) (α : ℝ) 
  (h1 : cos α ≤ sin α) : 
  (2 * k * π + π / 4) ≤ α ∧ α < (2 * k * π + π / 2) :=
sorry

end alpha_range_in_first_quadrant_l37_37991


namespace JackBuckets_l37_37610

theorem JackBuckets (tank_capacity buckets_per_trip_jill trips_jill time_ratio trip_buckets_jack : ℕ) :
  tank_capacity = 600 → buckets_per_trip_jill = 5 → trips_jill = 30 →
  time_ratio = 3 / 2 → trip_buckets_jack = 2 :=
  sorry

end JackBuckets_l37_37610


namespace train_length_correct_l37_37171

noncomputable def speed_km_per_hour : ℝ := 56
noncomputable def time_seconds : ℝ := 32.142857142857146
noncomputable def bridge_length_m : ℝ := 140
noncomputable def train_length_m : ℝ := 360

noncomputable def speed_m_per_s : ℝ := speed_km_per_hour * (1000 / 3600)
noncomputable def total_distance_m : ℝ := speed_m_per_s * time_seconds

theorem train_length_correct :
  (total_distance_m - bridge_length_m) = train_length_m :=
  by
    sorry

end train_length_correct_l37_37171


namespace intersection_of_M_and_N_l37_37032

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 5}
noncomputable def N : Set ℝ := {x | x * (x - 4) > 0}

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5) } := by
  sorry

end intersection_of_M_and_N_l37_37032


namespace trapezoid_triangle_area_ratio_l37_37762

/-- Given a trapezoid with triangles ABC and ADC such that the ratio of their areas is 4:1 and AB + CD = 150 cm.
Prove that the length of segment AB is 120 cm. --/
theorem trapezoid_triangle_area_ratio
  (h ABC_area ADC_area : ℕ)
  (AB CD : ℕ)
  (h_ratio : ABC_area / ADC_area = 4)
  (area_ABC : ABC_area = AB * h / 2)
  (area_ADC : ADC_area = CD * h / 2)
  (h_sum : AB + CD = 150) :
  AB = 120 := 
sorry

end trapezoid_triangle_area_ratio_l37_37762


namespace part1_part2_l37_37050

noncomputable def f (x b : ℝ) : ℝ := (x + b) * Real.log x

noncomputable def g (x b a : ℝ) : ℝ := Real.exp x * ((f x b) / (x + 2) - 2 * a)

example (b : ℝ) : has_deriv_at (λ x, f x b) (3 : ℝ) 1 := by {
  have h : (∀ b, HasDerivAt (λ x, (x + b) * Real.log x) (Real.log 1 + b / 1 + 1) 1),
  sorry,
  rw Real.log_one at h,
  exact h b,
}

theorem part1 : (∃ b : ℝ, HasDerivAt (λ x, f x b) 3 1) → b = 2 := by {
  intro h,
  cases' h with b hb,
  have : 3 = Real.log 1 + b + 1 := Sorry,
  rw Real.log_one at this,
  linarith,
}

theorem part2 (a : ℝ) : (∀ x : ℝ, x > 0 → g x 2 a ≥ g x 2 a) → a ≤ 1 / 2 := by {
  have h : ∀ x > 0, (Real.log x + 1 / x - 2 * a) * Real.exp x ≥ 0,
  sorry,
  have g_is_increasing : ∀ x > 0, Real.log x + 1 / x ≥ 2 * a,
  sorry,
  have h_min : (∃ a, ∀ x > 0, Real.log x + 1 / x ≥ a) := sorry,
  cases' h_min with m hm,
  have : m = 1,
  sorry,
  have : 2 * a ≤ 1,
  exact (hm 1 (by norm_num)).2,
  linarith,
}

#check part1
#check part2

end part1_part2_l37_37050
