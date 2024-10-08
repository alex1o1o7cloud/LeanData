import Mathlib

namespace min_value_9x_plus_3y_l191_191560

noncomputable def minimum_value_of_expression : ℝ := 6

theorem min_value_9x_plus_3y (x y : ℝ) 
  (h1 : (x - 1) * 4 + 2 * y = 0) 
  (ha : ∃ (a1 a2 : ℝ), (a1, a2) = (x - 1, 2)) 
  (hb : ∃ (b1 b2 : ℝ), (b1, b2) = (4, y)) : 
  9^x + 3^y = minimum_value_of_expression :=
by
  sorry

end min_value_9x_plus_3y_l191_191560


namespace travel_time_l191_191004

theorem travel_time (distance speed : ℕ) (h_distance : distance = 810) (h_speed : speed = 162) :
  distance / speed = 5 :=
by
  sorry

end travel_time_l191_191004


namespace tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l191_191823

def tens_digit_N_pow_20 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  if (N % 5 = 1 ∨ N % 5 = 2 ∨ N % 5 = 3 ∨ N % 5 = 4) then
    (N^20 % 100) / 10  -- tens digit of last two digits
  else
    sorry  -- N should be in form of 5k±1 or 5k±2
else
  sorry  -- N not satisfying conditions

def hundreds_digit_N_pow_200 (N : ℕ) : Nat :=
if (N % 2 = 0 ∧ N % 10 ≠ 0) then
  (N^200 % 1000) / 100  -- hundreds digit of the last three digits
else
  sorry  -- N not satisfying conditions

theorem tens_digit_N_pow_20_is_7 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  tens_digit_N_pow_20 N = 7 := sorry

theorem hundreds_digit_N_pow_200_is_3 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) : 
  hundreds_digit_N_pow_200 N = 3 := sorry

end tens_digit_N_pow_20_is_7_hundreds_digit_N_pow_200_is_3_l191_191823


namespace find_non_negative_integer_pairs_l191_191791

theorem find_non_negative_integer_pairs (m n : ℕ) :
  3 * 2^m + 1 = n^2 ↔ (m = 0 ∧ n = 2) ∨ (m = 3 ∧ n = 5) ∨ (m = 4 ∧ n = 7) := by
  sorry

end find_non_negative_integer_pairs_l191_191791


namespace budget_for_supplies_l191_191143

-- Conditions as definitions
def percentage_transportation := 20
def percentage_research_development := 9
def percentage_utilities := 5
def percentage_equipment := 4
def degrees_salaries := 216
def total_degrees := 360
def total_percentage := 100

-- Mathematical problem: Prove the percentage spent on supplies
theorem budget_for_supplies :
  (total_percentage - (percentage_transportation +
                       percentage_research_development +
                       percentage_utilities +
                       percentage_equipment) - 
   ((degrees_salaries * total_percentage) / total_degrees)) = 2 := by
  sorry

end budget_for_supplies_l191_191143


namespace translation_of_graph_l191_191541

theorem translation_of_graph (f : ℝ → ℝ) (x : ℝ) :
  f x = 2 ^ x →
  f (x - 1) + 2 = 2 ^ (x - 1) + 2 :=
by
  intro
  sorry

end translation_of_graph_l191_191541


namespace perimeter_of_shaded_region_correct_l191_191233

noncomputable def perimeter_of_shaded_region : ℝ :=
  let r := 7
  let perimeter := 2 * r + (3 / 4) * (2 * Real.pi * r)
  perimeter

theorem perimeter_of_shaded_region_correct :
  perimeter_of_shaded_region = 14 + 10.5 * Real.pi :=
by
  sorry

end perimeter_of_shaded_region_correct_l191_191233


namespace necessary_but_not_sufficient_condition_l191_191849

open Real

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 ≤ a ∧ a ≤ 4) → (a^2 - 4 * a < 0) := 
by
  sorry

end necessary_but_not_sufficient_condition_l191_191849


namespace travel_speed_l191_191189

theorem travel_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 195) (h_time : time = 3) : 
  distance / time = 65 :=
by 
  rw [h_distance, h_time]
  norm_num

end travel_speed_l191_191189


namespace major_axis_length_of_ellipse_l191_191318

-- Definition of the conditions
def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 2 = 1
def is_focus (x y m : ℝ) : Prop := line x y ∧ ellipse x y m

theorem major_axis_length_of_ellipse (m : ℝ) (h₀ : m > 0) :
  (∃ (x y : ℝ), is_focus x y m) → 2 * Real.sqrt 6 = 2 * Real.sqrt m :=
sorry

end major_axis_length_of_ellipse_l191_191318


namespace trains_clear_time_l191_191325

noncomputable def time_to_clear (length_train1 length_train2 speed_train1 speed_train2 : ℕ) : ℝ :=
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 1000 / 3600)

theorem trains_clear_time :
  time_to_clear 121 153 80 65 = 6.803 :=
by
  -- This is a placeholder for the proof
  sorry

end trains_clear_time_l191_191325


namespace area_of_square_with_given_diagonal_l191_191895

-- Definition of the conditions
def diagonal := 12
def s := Real
def area (s : Real) := s^2
def diag_relation (d s : Real) := d^2 = 2 * s^2

-- The proof statement
theorem area_of_square_with_given_diagonal :
  ∃ s : Real, diag_relation diagonal s ∧ area s = 72 :=
by
  sorry

end area_of_square_with_given_diagonal_l191_191895


namespace player_A_success_l191_191918

/-- Representation of the problem conditions --/
structure GameState where
  coins : ℕ
  boxes : ℕ
  n_coins : ℕ 
  n_boxes : ℕ 
  arrangement: ℕ → ℕ 
  (h_coins : coins ≥ 2012)
  (h_boxes : boxes = 2012)
  (h_initial_distribution : (∀ b, arrangement b ≥ 1))
  
/-- The main theorem for player A to ensure at least 1 coin in each box --/
theorem player_A_success (s : GameState) : 
  s.coins ≥ 4022 → (∀ b, s.arrangement b ≥ 1) :=
by
  sorry

end player_A_success_l191_191918


namespace p_p_eq_twenty_l191_191956

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
  else 3 * x + 2 * y

theorem p_p_eq_twenty : p (p 2 (-3)) (p (-3) (-4)) = 20 :=
by
  sorry

end p_p_eq_twenty_l191_191956


namespace quadratic_one_real_root_l191_191661

theorem quadratic_one_real_root (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 2 * x + 1 = 0) ↔ ((a = 0) ∨ (a = 1))) :=
sorry

end quadratic_one_real_root_l191_191661


namespace sum_ab_l191_191067

theorem sum_ab (a b : ℕ) (h1 : 1 < b) (h2 : a ^ b < 500) (h3 : ∀ x y : ℕ, (1 < y ∧ x ^ y < 500 ∧ (x + y) % 2 = 0) → a ^ b ≥ x ^ y) (h4 : (a + b) % 2 = 0) : a + b = 24 :=
  sorry

end sum_ab_l191_191067


namespace sin_cos_sum_identity_l191_191470

noncomputable def trigonometric_identity (x y z w : ℝ) := 
  (Real.sin x * Real.cos y + Real.sin z * Real.cos w) = Real.sqrt 2 / 2

theorem sin_cos_sum_identity :
  trigonometric_identity 347 148 77 58 :=
by sorry

end sin_cos_sum_identity_l191_191470


namespace mod_equiv_22_l191_191862

theorem mod_equiv_22 : ∃ m : ℕ, (198 * 864) % 50 = m ∧ 0 ≤ m ∧ m < 50 ∧ m = 22 := by
  sorry

end mod_equiv_22_l191_191862


namespace total_selling_price_of_toys_l191_191430

/-
  Prove that the total selling price (TSP) for 18 toys,
  given that each toy costs Rs. 1100 and the man gains the cost price of 3 toys, is Rs. 23100.
-/
theorem total_selling_price_of_toys :
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  TSP = 23100 :=
by
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  sorry

end total_selling_price_of_toys_l191_191430


namespace compute_expression_l191_191770

theorem compute_expression (x : ℝ) (h : x + (1 / x) = 7) :
  (x - 3)^2 + (49 / (x - 3)^2) = 23 :=
by
  sorry

end compute_expression_l191_191770


namespace measure_of_angle_C_l191_191157

variable (A B C : Real)

theorem measure_of_angle_C (h1 : 4 * Real.sin A + 2 * Real.cos B = 4) 
                           (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) :
                           C = Real.pi / 6 :=
by
  sorry

end measure_of_angle_C_l191_191157


namespace find_y_l191_191924

theorem find_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  -- Proof can go here
  sorry

end find_y_l191_191924


namespace max_mx_plus_ny_l191_191390

theorem max_mx_plus_ny 
  (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 6) 
  (h2 : x^2 + y^2 = 24) : 
  mx + ny ≤ 12 :=
sorry

end max_mx_plus_ny_l191_191390


namespace focus_of_parabola_l191_191719

theorem focus_of_parabola (focus : ℝ × ℝ) : 
  (∃ p : ℝ, y = p * x^2 / 2 → focus = (0, 1 / 2)) :=
by
  sorry

end focus_of_parabola_l191_191719


namespace equal_savings_l191_191916

theorem equal_savings (A B AE BE AS BS : ℕ) 
  (hA : A = 2000)
  (hA_B : 5 * B = 4 * A)
  (hAE_BE : 3 * BE = 2 * AE)
  (hSavings : AS = A - AE ∧ BS = B - BE ∧ AS = BS) :
  AS = 800 ∧ BS = 800 :=
by
  -- Placeholders for definitions and calculations
  sorry

end equal_savings_l191_191916


namespace simplify_polynomial_l191_191426

variable {R : Type*} [CommRing R]

theorem simplify_polynomial (x : R) :
  (12 * x ^ 10 + 9 * x ^ 9 + 5 * x ^ 8) + (2 * x ^ 12 + x ^ 10 + 2 * x ^ 9 + 3 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  2 * x ^ 12 + 13 * x ^ 10 + 11 * x ^ 9 + 8 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
  sorry

end simplify_polynomial_l191_191426


namespace average_speed_l191_191026

-- Define the conditions given in the problem
def distance_first_hour : ℕ := 50 -- distance traveled in the first hour
def distance_second_hour : ℕ := 60 -- distance traveled in the second hour
def total_distance : ℕ := distance_first_hour + distance_second_hour -- total distance traveled

-- Define the total time
def total_time : ℕ := 2 -- total time in hours

-- The problem statement: proving the average speed
theorem average_speed : total_distance / total_time = 55 := by
  unfold total_distance total_time
  sorry

end average_speed_l191_191026


namespace binomial_ratio_l191_191803

theorem binomial_ratio (n : ℕ) (r : ℕ) :
  (Nat.choose n r : ℚ) / (Nat.choose n (r+1) : ℚ) = 1 / 2 →
  (Nat.choose n (r+1) : ℚ) / (Nat.choose n (r+2) : ℚ) = 2 / 3 →
  n = 14 :=
by
  sorry

end binomial_ratio_l191_191803


namespace original_price_of_sarees_l191_191394

theorem original_price_of_sarees (P : ℝ) (h : 0.92 * 0.90 * P = 331.2) : P = 400 :=
by
  sorry

end original_price_of_sarees_l191_191394


namespace estimate_white_balls_l191_191845

-- Statements for conditions
variables (black_balls white_balls : ℕ)
variables (draws : ℕ := 40)
variables (black_draws : ℕ := 10)

-- Define total white draws
def white_draws := draws - black_draws

-- Ratio of black to white draws
def draw_ratio := black_draws / white_draws

-- Given condition on known draws
def black_ball_count := 4
def known_draw_ratio := 1 / 3

-- Lean 4 statement to prove the number of white balls
theorem estimate_white_balls (h : black_ball_count / white_balls = known_draw_ratio) : white_balls = 12 :=
sorry -- Proof omitted

end estimate_white_balls_l191_191845


namespace volume_ratio_of_frustum_l191_191090

theorem volume_ratio_of_frustum
  (h_s h : ℝ)
  (A_s A : ℝ)
  (V_s V : ℝ)
  (ratio_lateral_area : ℝ)
  (ratio_height : ℝ)
  (ratio_base_area : ℝ)
  (H_lateral_area: ratio_lateral_area = 9 / 16)
  (H_height: ratio_height = 3 / 5)
  (H_base_area: ratio_base_area = 9 / 25)
  (H_volume_small: V_s = 1 / 3 * h_s * A_s)
  (H_volume_total: V = 1 / 3 * h * A - 1 / 3 * h_s * A_s) :
  V_s / V = 27 / 98 :=
by
  sorry

end volume_ratio_of_frustum_l191_191090


namespace simplify_and_evaluate_expression_l191_191050

   variable (x : ℝ)

   theorem simplify_and_evaluate_expression (h : x = 2 * Real.sqrt 5 - 1) :
     (1 / (x ^ 2 + 2 * x + 1) * (1 + 3 / (x - 1)) / ((x + 2) / (x ^ 2 - 1))) = Real.sqrt 5 / 10 :=
   sorry
   
end simplify_and_evaluate_expression_l191_191050


namespace total_money_of_james_and_ali_l191_191373

def jamesOwns : ℕ := 145
def jamesAliDifference : ℕ := 40
def aliOwns : ℕ := jamesOwns - jamesAliDifference

theorem total_money_of_james_and_ali :
  jamesOwns + aliOwns = 250 := by
  sorry

end total_money_of_james_and_ali_l191_191373


namespace joan_remaining_oranges_l191_191683

def total_oranges_joan_picked : ℕ := 37
def oranges_sara_sold : ℕ := 10

theorem joan_remaining_oranges : total_oranges_joan_picked - oranges_sara_sold = 27 := by
  sorry

end joan_remaining_oranges_l191_191683


namespace olivia_packs_of_basketball_cards_l191_191225

-- Definitions for the given conditions
def pack_cost : ℕ := 3
def deck_cost : ℕ := 4
def number_of_decks : ℕ := 5
def total_money : ℕ := 50
def change_received : ℕ := 24

-- Statement to be proved
theorem olivia_packs_of_basketball_cards (x : ℕ) (hx : pack_cost * x + deck_cost * number_of_decks = total_money - change_received) : x = 2 :=
by 
  sorry

end olivia_packs_of_basketball_cards_l191_191225


namespace infinitely_many_sum_of_squares_exceptions_l191_191821

-- Define the predicate for a number being expressible as a sum of two squares
def is_sum_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

-- Define the main theorem
theorem infinitely_many_sum_of_squares_exceptions : 
  ∃ f : ℕ → ℕ, (∀ k : ℕ, is_sum_of_squares (f k)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k - 1)) ∧ (∀ k : ℕ, ¬ is_sum_of_squares (f k + 1)) ∧ (∀ k1 k2 : ℕ, k1 ≠ k2 → f k1 ≠ f k2) :=
sorry

end infinitely_many_sum_of_squares_exceptions_l191_191821


namespace lesser_of_two_numbers_l191_191633

theorem lesser_of_two_numbers (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
by
  sorry

end lesser_of_two_numbers_l191_191633


namespace a6_add_b6_geq_ab_a4_add_b4_l191_191023

theorem a6_add_b6_geq_ab_a4_add_b4 (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end a6_add_b6_geq_ab_a4_add_b4_l191_191023


namespace total_stamps_is_38_l191_191015

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l191_191015


namespace total_rectangles_l191_191017

-- Definitions
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4
def exclude_line_pair: ℕ := 1
def total_combinations (n m : ℕ) : ℕ := Nat.choose n m

-- Statement
theorem total_rectangles (h_lines : ℕ) (v_lines : ℕ) 
  (exclude_pair : ℕ) (valid_h_comb : ℕ) (valid_v_comb : ℕ) :
  h_lines = horizontal_lines →
  v_lines = vertical_lines →
  exclude_pair = exclude_line_pair →
  valid_h_comb = total_combinations 5 2 - exclude_pair →
  valid_v_comb = total_combinations 4 2 →
  valid_h_comb * valid_v_comb = 54 :=
by intros; sorry

end total_rectangles_l191_191017


namespace figure_perimeter_l191_191082

theorem figure_perimeter (h_segments v_segments : ℕ) (side_length : ℕ) 
  (h_count : h_segments = 16) (v_count : v_segments = 10) (side_len : side_length = 1) :
  2 * (h_segments + v_segments) * side_length = 26 :=
by
  sorry

end figure_perimeter_l191_191082


namespace grey_eyed_black_haired_students_l191_191841

theorem grey_eyed_black_haired_students (total_students black_haired green_eyed_red_haired grey_eyed : ℕ) 
(h_total : total_students = 60) 
(h_black_haired : black_haired = 35) 
(h_green_eyed_red_haired : green_eyed_red_haired = 20) 
(h_grey_eyed : grey_eyed = 25) : 
grey_eyed - (total_students - black_haired - green_eyed_red_haired) = 20 :=
by
  sorry

end grey_eyed_black_haired_students_l191_191841


namespace find_value_l191_191286

theorem find_value : 3 + 2 * (8 - 3) = 13 := by
  sorry

end find_value_l191_191286


namespace hotel_charge_comparison_l191_191152

theorem hotel_charge_comparison (R G P : ℝ) 
  (h1 : P = R - 0.70 * R)
  (h2 : P = G - 0.10 * G) :
  ((R - G) / G) * 100 = 170 :=
by
  sorry

end hotel_charge_comparison_l191_191152


namespace max_value_of_f_l191_191263

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f 1 = 1 / Real.exp 1 := 
by {
  sorry
}

end max_value_of_f_l191_191263


namespace women_in_room_l191_191961

theorem women_in_room (x q : ℕ) (h1 : 4 * x + 2 = 14) (h2 : q = 2 * (5 * x - 3)) : q = 24 :=
by sorry

end women_in_room_l191_191961


namespace first_term_is_sqrt9_l191_191820

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

end first_term_is_sqrt9_l191_191820


namespace correct_equations_l191_191283

theorem correct_equations (m n : ℕ) (h1 : n = 4 * m - 2) (h2 : n = 2 * m + 58) :
  (4 * m - 2 = 2 * m + 58 ∨ (n + 2) / 4 = (n - 58) / 2) :=
by
  sorry

end correct_equations_l191_191283


namespace final_problem_l191_191965

-- Define the function f
def f (x p q : ℝ) : ℝ := x * abs x + p * x + q

-- Proposition ①: When q=0, f(x) is an odd function
def prop1 (p : ℝ) : Prop :=
  ∀ x : ℝ, f x p 0 = - f (-x) p 0

-- Proposition ②: The graph of y=f(x) is symmetric with respect to the point (0,q)
def prop2 (p q : ℝ) : Prop :=
  ∀ x : ℝ, f x p q = f (-x) p q + 2 * q

-- Proposition ③: When p=0 and q > 0, the equation f(x)=0 has exactly one real root
def prop3 (q : ℝ) : Prop :=
  q > 0 → ∃! x : ℝ, f x 0 q = 0

-- Proposition ④: The equation f(x)=0 has at most two real roots
def prop4 (p q : ℝ) : Prop :=
  ∀ x1 x2 x3 : ℝ, f x1 p q = 0 ∧ f x2 p q = 0 ∧ f x3 p q = 0 → x1 = x2 ∨ x1 = x3 ∨ x2 = x3

-- The final problem to prove that propositions ①, ②, and ③ are true and proposition ④ is false
theorem final_problem (p q : ℝ) :
  prop1 p ∧ prop2 p q ∧ prop3 q ∧ ¬prop4 p q :=
sorry

end final_problem_l191_191965


namespace intersection_always_exists_minimum_chord_length_and_equation_l191_191861

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 4 * x - 8 * y - 11 = 0

noncomputable def line_eq (m x y : ℝ) : Prop :=
  (m - 1) * x + m * y = m + 1

theorem intersection_always_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem minimum_chord_length_and_equation :
  ∃ (k : ℝ) (x y : ℝ), k = sqrt 3 ∧ (3 * x - 2 * y + 7 = 0) ∧
    ∀ m, ∃ (xp yp : ℝ), line_eq m xp yp ∧ ∃ (l1 l2 : ℝ), line_eq m l1 l2 ∧ 
    (circle_eq xp yp ∧ circle_eq l1 l2)  :=
by
  sorry

end intersection_always_exists_minimum_chord_length_and_equation_l191_191861


namespace quadratic_root_proof_l191_191177

noncomputable def root_condition (p q m n : ℝ) :=
  ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x ≠ 0 ∧ (1/x)^2 + m * (1/x) + n = 0

theorem quadratic_root_proof (p q m n : ℝ) (h : root_condition p q m n) :
  (pn - m) * (qm - p) = (qn - 1)^2 :=
sorry

end quadratic_root_proof_l191_191177


namespace three_g_of_x_l191_191285

noncomputable def g (x : ℝ) : ℝ := 3 / (3 + x)

theorem three_g_of_x (x : ℝ) (h : x > 0) : 3 * g x = 27 / (9 + x) :=
by
  sorry

end three_g_of_x_l191_191285


namespace object_speed_conversion_l191_191018

theorem object_speed_conversion 
  (distance : ℝ)
  (velocity : ℝ) 
  (conversion_factor : ℝ) 
  (distance_in_km : ℝ)
  (time_in_seconds : ℝ) 
  (time_in_minutes : ℝ) 
  (speed_in_kmh : ℝ) :
  distance = 200 ∧ 
  velocity = 1/3 ∧ 
  time_in_seconds = distance / velocity ∧ 
  time_in_minutes = time_in_seconds / 60 ∧ 
  conversion_factor = 3600 * 0.001 ∧ 
  speed_in_kmh = velocity * conversion_factor ↔ 
  speed_in_kmh = 0.4 :=
by sorry

end object_speed_conversion_l191_191018


namespace find_y1_l191_191617

theorem find_y1
  (y1 y2 y3 : ℝ)
  (h1 : 0 ≤ y3)
  (h2 : y3 ≤ y2)
  (h3 : y2 ≤ y1)
  (h4 : y1 ≤ 1)
  (h5 : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = 3 / 4 :=
sorry

end find_y1_l191_191617


namespace cube_inequality_of_greater_l191_191144

theorem cube_inequality_of_greater {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l191_191144


namespace range_of_cars_l191_191007

def fuel_vehicle_cost_per_km (x : ℕ) : ℚ := (40 * 9) / x
def new_energy_vehicle_cost_per_km (x : ℕ) : ℚ := (60 * 0.6) / x

theorem range_of_cars : ∃ x : ℕ, fuel_vehicle_cost_per_km x = new_energy_vehicle_cost_per_km x + 0.54 ∧ x = 600 := 
by {
  sorry
}

end range_of_cars_l191_191007


namespace consecutive_differences_equal_l191_191399

-- Define the set and the condition
def S : Set ℕ := {n : ℕ | n > 0}

-- Condition that for any two numbers a and b in S with a > b, at least one of a + b or a - b is also in S
axiom h_condition : ∀ a b : ℕ, a ∈ S → b ∈ S → a > b → (a + b ∈ S ∨ a - b ∈ S)

-- The main theorem that we want to prove
theorem consecutive_differences_equal (a : ℕ) (s : Fin 2003 → ℕ) 
  (hS : ∀ i, s i ∈ S)
  (h_ordered : ∀ i j, i < j → s i < s j) :
  ∃ (d : ℕ), ∀ i, i < 2002 → (s (i + 1)) - (s i) = d :=
sorry

end consecutive_differences_equal_l191_191399


namespace difference_of_squares_divisible_by_9_l191_191155

theorem difference_of_squares_divisible_by_9 (a b : ℤ) : 
  9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) :=
by
  sorry

end difference_of_squares_divisible_by_9_l191_191155


namespace W_555_2_last_three_digits_l191_191921

noncomputable def W : ℕ → ℕ → ℕ
| n, 0     => n ^ n
| n, (k+1) => W (W n k) k

theorem W_555_2_last_three_digits :
  (W 555 2) % 1000 = 875 :=
sorry

end W_555_2_last_three_digits_l191_191921


namespace minimum_other_sales_met_l191_191944

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

end minimum_other_sales_met_l191_191944


namespace participants_l191_191997

variable {A B C D : Prop}

theorem participants (h1 : A → B) (h2 : ¬C → ¬B) (h3 : C → ¬D) :
  (¬A ∧ C ∧ B ∧ ¬D) ∨ ¬B :=
by
  -- The proof is not provided
  sorry

end participants_l191_191997


namespace johns_total_expenditure_l191_191877

-- Conditions
def treats_first_15_days : ℕ := 3 * 15
def treats_next_15_days : ℕ := 4 * 15
def total_treats : ℕ := treats_first_15_days + treats_next_15_days
def cost_per_treat : ℝ := 0.10
def discount_threshold : ℕ := 50
def discount_rate : ℝ := 0.10

-- Intermediate calculations
def total_cost_without_discount : ℝ := total_treats * cost_per_treat
def discounted_cost_per_treat : ℝ := cost_per_treat * (1 - discount_rate)
def total_cost_with_discount : ℝ := total_treats * discounted_cost_per_treat

-- Main theorem statement
theorem johns_total_expenditure : total_cost_with_discount = 9.45 :=
by
  -- Place proof here
  sorry

end johns_total_expenditure_l191_191877


namespace divisible_by_5_last_digit_l191_191064

theorem divisible_by_5_last_digit (B : ℕ) (h : B < 10) : (∃ k : ℕ, 5270 + B = 5 * k) ↔ B = 0 ∨ B = 5 :=
by sorry

end divisible_by_5_last_digit_l191_191064


namespace find_b_plus_k_l191_191392

open Real

noncomputable def semi_major_axis (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  dist p f1 + dist p f2

def c_squared (a : ℝ) (b : ℝ) : ℝ :=
  a ^ 2 - b ^ 2

theorem find_b_plus_k :
  ∀ (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k : ℝ) (a b : ℝ),
  f1 = (-2, 0) →
  f2 = (2, 0) →
  p = (6, 0) →
  (∃ a b, semi_major_axis f1 f2 p = 2 * a ∧ c_squared a b = 4) →
  h = 0 →
  k = 0 →
  b = 4 * sqrt 2 →
  b + k = 4 * sqrt 2 :=
by
  intros f1 f2 p h k a b f1_def f2_def p_def maj_axis_def h_def k_def b_def
  rw [b_def, k_def]
  exact add_zero (4 * sqrt 2)

end find_b_plus_k_l191_191392


namespace distance_between_closest_points_correct_l191_191091

noncomputable def circle_1_center : ℝ × ℝ := (3, 3)
noncomputable def circle_2_center : ℝ × ℝ := (20, 12)
noncomputable def circle_1_radius : ℝ := circle_1_center.2
noncomputable def circle_2_radius : ℝ := circle_2_center.2
noncomputable def distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (12 - 3)^2)
noncomputable def distance_between_closest_points : ℝ := distance_between_centers - (circle_1_radius + circle_2_radius)

theorem distance_between_closest_points_correct :
  distance_between_closest_points = Real.sqrt 370 - 15 :=
sorry

end distance_between_closest_points_correct_l191_191091


namespace number_divided_by_four_l191_191511

variable (x : ℝ)

theorem number_divided_by_four (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by {
  sorry
}

end number_divided_by_four_l191_191511


namespace final_cost_is_35_l191_191293

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end final_cost_is_35_l191_191293


namespace problem1_problem2_l191_191615

-- First Problem
theorem problem1 : 
  Real.cos (Real.pi / 3) + Real.sin (Real.pi / 4) - Real.tan (Real.pi / 4) = (-1 + Real.sqrt 2) / 2 :=
by
  sorry

-- Second Problem
theorem problem2 : 
  6 * (Real.tan (Real.pi / 6))^2 - Real.sqrt 3 * Real.sin (Real.pi / 3) - 2 * Real.cos (Real.pi / 4) = 1 / 2 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l191_191615


namespace a_eq_b_if_conditions_l191_191477

theorem a_eq_b_if_conditions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := 
sorry

end a_eq_b_if_conditions_l191_191477


namespace junghyeon_stickers_l191_191122

def total_stickers : ℕ := 25
def junghyeon_sticker_count (yejin_stickers : ℕ) : ℕ := 2 * yejin_stickers + 1

theorem junghyeon_stickers (yejin_stickers : ℕ) (h : yejin_stickers + junghyeon_sticker_count yejin_stickers = total_stickers) : 
  junghyeon_sticker_count yejin_stickers = 17 :=
  by
  sorry

end junghyeon_stickers_l191_191122


namespace ratio_squirrels_to_raccoons_l191_191110

def animals_total : ℕ := 84
def raccoons : ℕ := 12
def squirrels : ℕ := animals_total - raccoons

theorem ratio_squirrels_to_raccoons : (squirrels : ℚ) / raccoons = 6 :=
by
  sorry

end ratio_squirrels_to_raccoons_l191_191110


namespace maurice_age_l191_191599

theorem maurice_age (M : ℕ) 
  (h₁ : 48 = 4 * (M + 5)) : M = 7 := 
by
  sorry

end maurice_age_l191_191599


namespace algebra_expression_solution_l191_191928

theorem algebra_expression_solution
  (m : ℝ)
  (h : m^2 + m - 1 = 0) :
  m^3 + 2 * m^2 - 2001 = -2000 := by
  sorry

end algebra_expression_solution_l191_191928


namespace tire_usage_is_25714_l191_191439

-- Definitions based on conditions
def car_has_six_tires : Prop := (4 + 2 = 6)
def used_equally_over_miles (total_miles : ℕ) (number_of_tires : ℕ) : Prop := 
  (total_miles * 4) / number_of_tires = 25714

-- Theorem statement based on proof
theorem tire_usage_is_25714 (miles_driven : ℕ) (num_tires : ℕ) 
  (h1 : car_has_six_tires) 
  (h2 : miles_driven = 45000)
  (h3 : num_tires = 7) :
  used_equally_over_miles miles_driven num_tires :=
by
  sorry

end tire_usage_is_25714_l191_191439


namespace harry_james_payment_l191_191993

theorem harry_james_payment (x y H : ℝ) (h1 : H - 12 = 44 / y) (h2 : y > 1) (h3 : H != 12 + 44/3) : H = 23 ∧ y = 4 :=
by
  sorry

end harry_james_payment_l191_191993


namespace roots_of_polynomial_l191_191154

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l191_191154


namespace walmart_knives_eq_three_l191_191134

variable (k : ℕ)

-- Walmart multitool
def walmart_tools : ℕ := 1 + k + 2

-- Target multitool (with twice as many knives as Walmart)
def target_tools : ℕ := 1 + 2 * k + 3 + 1

-- The condition that Target multitool has 5 more tools compared to Walmart
theorem walmart_knives_eq_three (h : target_tools k = walmart_tools k + 5) : k = 3 :=
by
  sorry

end walmart_knives_eq_three_l191_191134


namespace number_of_convex_quadrilaterals_l191_191700

-- Each definition used in Lean 4 statement should directly appear in the conditions problem.

variable {n : ℕ} -- Definition of n in Lean

-- Conditions
def distinct_points_on_circle (n : ℕ) : Prop := n = 10

-- Question and correct answer
theorem number_of_convex_quadrilaterals (h : distinct_points_on_circle n) : 
    (n.choose 4) = 210 := by
  sorry

end number_of_convex_quadrilaterals_l191_191700


namespace midpoint_on_hyperbola_l191_191022

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l191_191022


namespace work_completed_together_in_4_days_l191_191364

/-- A can do the work in 6 days. -/
def A_work_rate : ℚ := 1 / 6

/-- B can do the work in 12 days. -/
def B_work_rate : ℚ := 1 / 12

/-- Combined work rate of A and B working together. -/
def combined_work_rate : ℚ := A_work_rate + B_work_rate

/-- Number of days for A and B to complete the work together. -/
def days_to_complete : ℚ := 1 / combined_work_rate

theorem work_completed_together_in_4_days : days_to_complete = 4 := by
  sorry

end work_completed_together_in_4_days_l191_191364


namespace no_real_solutions_eq_l191_191268

theorem no_real_solutions_eq (x y : ℝ) :
  x^2 + y^2 - 2 * x + 4 * y + 6 ≠ 0 :=
sorry

end no_real_solutions_eq_l191_191268


namespace sophia_estimate_larger_l191_191274

theorem sophia_estimate_larger (x y a b : ℝ) (hx : x > y) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end sophia_estimate_larger_l191_191274


namespace inequality_x_add_inv_x_ge_two_l191_191759

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end inequality_x_add_inv_x_ge_two_l191_191759


namespace mink_ratio_set_free_to_total_l191_191033

-- Given conditions
def coats_needed_per_skin : ℕ := 15
def minks_bought : ℕ := 30
def babies_per_mink : ℕ := 6
def coats_made : ℕ := 7

-- Question as a proof problem
theorem mink_ratio_set_free_to_total :
  let total_minks := minks_bought * (1 + babies_per_mink)
  let minks_used := coats_made * coats_needed_per_skin
  let minks_set_free := total_minks - minks_used
  minks_set_free * 2 = total_minks :=
by
  sorry

end mink_ratio_set_free_to_total_l191_191033


namespace admin_in_sample_l191_191807

-- Define the total number of staff members
def total_staff : ℕ := 200

-- Define the number of administrative personnel
def admin_personnel : ℕ := 24

-- Define the sample size taken
def sample_size : ℕ := 50

-- Goal: Prove the number of administrative personnel in the sample
theorem admin_in_sample : 
  (admin_personnel : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 6 := 
by
  sorry

end admin_in_sample_l191_191807


namespace tree_initial_height_l191_191393

noncomputable def initial_tree_height (H : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ := 
  H + growth_rate * years

theorem tree_initial_height :
  ∀ (H : ℝ), 
  (∀ (years : ℕ), ∃ h : ℝ, h = initial_tree_height H 0.5 years) →
  initial_tree_height H 0.5 6 = initial_tree_height H 0.5 4 * (7 / 6) →
  H = 4 :=
by
  intro H height_increase condition
  sorry

end tree_initial_height_l191_191393


namespace bill_initial_amount_l191_191035

/-- Suppose Ann has $777 and Bill gives Ann $167,
    after which they both have the same amount of money. 
    Prove that Bill initially had $1111. -/
theorem bill_initial_amount (A B : ℕ) (h₁ : A = 777) (h₂ : B - 167 = A + 167) : B = 1111 :=
by
  -- Proof goes here
  sorry

end bill_initial_amount_l191_191035


namespace probability_of_one_of_each_color_l191_191450

-- Definitions based on the conditions
def total_marbles : ℕ := 12
def marbles_of_each_color : ℕ := 3
def number_of_selected_marbles : ℕ := 4

-- Calculation based on problem requirements
def total_ways_to_choose_marbles : ℕ := Nat.choose total_marbles number_of_selected_marbles
def favorable_ways_to_choose : ℕ := marbles_of_each_color ^ number_of_selected_marbles

-- The main theorem to prove the probability
theorem probability_of_one_of_each_color :
  (favorable_ways_to_choose : ℚ) / total_ways_to_choose = 9 / 55 := by
  sorry

end probability_of_one_of_each_color_l191_191450


namespace plane_equation_through_point_parallel_l191_191294

theorem plane_equation_through_point_parallel (A B C D : ℤ) (hx hy hz : ℤ) (x y z : ℤ)
  (h_point : (A, B, C, D) = (-2, 1, -3, 10))
  (h_coordinates : (hx, hy, hz) = (2, -3, 1))
  (h_plane_parallel : ∀ x y z, -2 * x + y - 3 * z = 7 ↔ A * x + B * y + C * z + D = 0)
  (h_form : A > 0):
  ∃ A' B' C' D', A' * (x : ℤ) + B' * (y : ℤ) + C' * (z : ℤ) + D' = 0 :=
by
  sorry

end plane_equation_through_point_parallel_l191_191294


namespace pump_fills_tank_without_leak_l191_191319

variable (T : ℝ)
-- Condition: The effective rate with the leak is equal to the rate it takes for both to fill the tank.
def effective_rate_with_leak (T : ℝ) : Prop :=
  1 / T - 1 / 21 = 1 / 3.5

-- Conclude: the time it takes the pump to fill the tank without the leak
theorem pump_fills_tank_without_leak : effective_rate_with_leak T → T = 3 :=
by
  intro h
  sorry

end pump_fills_tank_without_leak_l191_191319


namespace score_order_l191_191576

-- Definitions that come from the problem conditions
variables (M Q S K : ℝ)
variables (hQK : Q = K) (hMK : M > K) (hSK : S < K)

-- The theorem to prove
theorem score_order (hQK : Q = K) (hMK : M > K) (hSK : S < K) : S < Q ∧ Q < M :=
by {
  sorry
}

end score_order_l191_191576


namespace sale_price_after_discounts_l191_191948

/-- The sale price of the television as a percentage of its original price after successive discounts of 25% followed by 10%. -/
theorem sale_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 350 → discount1 = 0.25 → discount2 = 0.10 →
  (original_price * (1 - discount1) * (1 - discount2) / original_price) * 100 = 67.5 :=
by
  intro h_price h_discount1 h_discount2
  sorry

end sale_price_after_discounts_l191_191948


namespace cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l191_191206

theorem cos_B_arithmetic_sequence (A B C : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180) :
  Real.cos B = 1 / 2 :=
by
  sorry

theorem sin_A_sin_C_geometric_sequence (A B C a b c : ℝ) (h1 : 2 * B = A + C) (h2 : A + B + C = 180)
  (h3 : b^2 = a * c) (h4 : b^2 = a^2 + c^2 - 2 * a * c * Real.cos B) :
  Real.sin A * Real.sin C = 3 / 4 :=
by
  sorry

end cos_B_arithmetic_sequence_sin_A_sin_C_geometric_sequence_l191_191206


namespace find_point_C_coordinates_l191_191805

/-- Given vertices A and B of a triangle, and the centroid G of the triangle, 
prove the coordinates of the third vertex C. 
-/
theorem find_point_C_coordinates : 
  ∀ (x y : ℝ),
  let A := (2, 3)
  let B := (-4, -2)
  let G := (2, -1)
  (2 + -4 + x) / 3 = 2 →
  (3 + -2 + y) / 3 = -1 →
  (x, y) = (8, -4) :=
by
  intro x y A B G h1 h2
  sorry

end find_point_C_coordinates_l191_191805


namespace average_temperature_week_l191_191332

theorem average_temperature_week :
  let d1 := 40
  let d2 := 40
  let d3 := 40
  let d4 := 80
  let d5 := 80
  let remaining_days_total := 140
  d1 + d2 + d3 + d4 + d5 + remaining_days_total = 420 ∧ 420 / 7 = 60 :=
by sorry

end average_temperature_week_l191_191332


namespace initial_percentage_water_l191_191001

theorem initial_percentage_water (P : ℝ) (H1 : 150 * P / 100 + 10 = 40) : P = 20 :=
by
  sorry

end initial_percentage_water_l191_191001


namespace height_of_picture_frame_l191_191984

-- Definitions of lengths and perimeter
def length : ℕ := 10
def perimeter : ℕ := 44

-- Perimeter formula for a rectangle
def rectangle_perimeter (L H : ℕ) : ℕ := 2 * (L + H)

-- Theorem statement: Proving the height is 12 inches based on given conditions
theorem height_of_picture_frame : ∃ H : ℕ, rectangle_perimeter length H = perimeter ∧ H = 12 := by
  sorry

end height_of_picture_frame_l191_191984


namespace digits_property_l191_191323

theorem digits_property (n : ℕ) (h : 100 ≤ n ∧ n < 1000) :
  (∃ (f : ℕ → Prop), ∀ d ∈ [n / 100, (n / 10) % 10, n % 10], f d ∧ (¬ d = 0 ∧ ¬ Nat.Prime d)) ↔ 
  (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ [1, 4, 6, 8, 9]) :=
sorry

end digits_property_l191_191323


namespace max_value_of_a_l191_191468

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 > 0 → x < a) ∧
  (∃ x : ℝ, x < a ∧ ¬(x^2 - 2*x - 3 > 0)) →
  a = -1 :=
by
  sorry

end max_value_of_a_l191_191468


namespace girls_insects_collected_l191_191763

theorem girls_insects_collected (boys_insects groups insects_per_group : ℕ) :
  boys_insects = 200 →
  groups = 4 →
  insects_per_group = 125 →
  (groups * insects_per_group) - boys_insects = 300 :=
by
  intros h1 h2 h3
  -- Prove the statement
  sorry

end girls_insects_collected_l191_191763


namespace jellybean_probability_l191_191327

theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let white_jellybeans := 6
  let total_chosen := 4
  let total_combinations := Nat.choose total_jellybeans total_chosen
  let red_combinations := Nat.choose red_jellybeans 3
  let non_red_combinations := Nat.choose (blue_jellybeans + white_jellybeans) 1
  let successful_outcomes := red_combinations * non_red_combinations
  let probability := (successful_outcomes : ℚ) / total_combinations
  probability = 4 / 91 :=
by 
  sorry

end jellybean_probability_l191_191327


namespace maximum_m_value_l191_191295

variable {a b c : ℝ}

noncomputable def maximum_m : ℝ := 9/8

theorem maximum_m_value 
  (h1 : (a - b)^2 + (b - c)^2 + (c - a)^2 ≥ maximum_m * a^2)
  (h2 : b^2 - 4 * a * c ≥ 0) : 
  maximum_m = 9 / 8 :=
sorry

end maximum_m_value_l191_191295


namespace purely_imaginary_has_specific_a_l191_191926

theorem purely_imaginary_has_specific_a (a : ℝ) :
  (a^2 - 1 + (a - 1 : ℂ) * Complex.I) = (a - 1 : ℂ) * Complex.I → a = -1 := 
by
  sorry

end purely_imaginary_has_specific_a_l191_191926


namespace probability_both_in_photo_correct_l191_191810

noncomputable def probability_both_in_photo (lap_time_Emily : ℕ) (lap_time_John : ℕ) (observation_start : ℕ) (observation_end : ℕ) : ℚ := 
  let GCD := Nat.gcd lap_time_Emily lap_time_John
  let cycle_time := lap_time_Emily * lap_time_John / GCD
  let visible_time := 2 * min (lap_time_Emily / 3) (lap_time_John / 3)
  visible_time / cycle_time

theorem probability_both_in_photo_correct : 
  probability_both_in_photo 100 75 900 1200 = 1 / 6 :=
by
  -- Use previous calculations and observations here to construct the proof.
  -- sorry is used to indicate that proof steps are omitted.
  sorry

end probability_both_in_photo_correct_l191_191810


namespace number_of_circles_l191_191649

theorem number_of_circles (side : ℝ) (enclosed_area : ℝ) (num_circles : ℕ) (radius : ℝ) :
  side = 14 ∧ enclosed_area = 42.06195997410015 ∧ 2 * radius = side ∧ π * radius^2 = 49 * π → num_circles = 4 :=
by
  intros
  sorry

end number_of_circles_l191_191649


namespace number_of_ways_2020_l191_191756

-- We are defining b_i explicitly restricted by the conditions in the problem.
def b (i : ℕ) : ℕ :=
  sorry

-- Given conditions
axiom h_bounds : ∀ i, 0 ≤ b i ∧ b i ≤ 99
axiom h_indices : ∀ (i : ℕ), i < 4

-- Main theorem statement
theorem number_of_ways_2020 (M : ℕ) 
  (h : 2020 = b 3 * 1000 + b 2 * 100 + b 1 * 10 + b 0) 
  (htotal : M = 203) : 
  M = 203 :=
  by 
    sorry

end number_of_ways_2020_l191_191756


namespace fourth_pentagon_has_31_dots_l191_191409

-- Conditions representing the sequence of pentagons
def first_pentagon_dots : ℕ := 1

def second_pentagon_dots : ℕ := first_pentagon_dots + 5

def nth_layer_dots (n : ℕ) : ℕ := 5 * (n - 1)

def nth_pentagon_dots (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + nth_layer_dots (k+1)) first_pentagon_dots

-- Question and proof statement
theorem fourth_pentagon_has_31_dots : nth_pentagon_dots 4 = 31 :=
  sorry

end fourth_pentagon_has_31_dots_l191_191409


namespace find_number_l191_191801

theorem find_number (N : ℕ) :
  let sum := 555 + 445
  let difference := 555 - 445
  let divisor := sum
  let quotient := 2 * difference
  let remainder := 70
  N = divisor * quotient + remainder -> N = 220070 := 
by
  intro h
  sorry

end find_number_l191_191801


namespace sum_geometric_sequence_l191_191659

theorem sum_geometric_sequence {a : ℕ → ℝ} (ha : ∃ q, ∀ n, a n = 3 * q ^ n)
  (h1 : a 1 = 3) (h2 : a 1 + a 2 + a 3 = 9) :
  a 4 + a 5 + a 6 = 9 ∨ a 4 + a 5 + a 6 = -72 :=
sorry

end sum_geometric_sequence_l191_191659


namespace subset_A_B_l191_191084

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2} -- Definition of set A
def B (a : ℝ) := {x : ℝ | x > a} -- Definition of set B

theorem subset_A_B (a : ℝ) : a < 1 → A ⊆ B a :=
by
  sorry

end subset_A_B_l191_191084


namespace rearrange_distinct_sums_mod_4028_l191_191138

theorem rearrange_distinct_sums_mod_4028 
  (x : Fin 2014 → ℤ) (y : Fin 2014 → ℤ) 
  (hx : ∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Fin 2014 → Fin 2014, Function.Bijective σ ∧ 
  ∀ i j : Fin 2014, i ≠ j → ( x i + y (σ i) ) % 4028 ≠ ( x j + y (σ j) ) % 4028 
:= by
  sorry

end rearrange_distinct_sums_mod_4028_l191_191138


namespace total_experiments_non_adjacent_l191_191043

theorem total_experiments_non_adjacent (n_org n_inorg n_add : ℕ) 
  (h_org : n_org = 3) (h_inorg : n_inorg = 2) (h_add : n_add = 2) 
  (no_adjacent : True) : 
  (n_org + n_inorg + n_add).factorial / (n_inorg + n_add).factorial * 
  (n_inorg + n_add + 1).choose n_org = 1440 :=
by
  -- The actual proof will go here.
  sorry

end total_experiments_non_adjacent_l191_191043


namespace unit_vector_opposite_AB_is_l191_191377

open Real

noncomputable def unit_vector_opposite_dir (A B : ℝ × ℝ) : ℝ × ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BA := (-AB.1, -AB.2)
  let mag_BA := sqrt (BA.1^2 + BA.2^2)
  (BA.1 / mag_BA, BA.2 / mag_BA)

theorem unit_vector_opposite_AB_is (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (-2, 6)) :
  unit_vector_opposite_dir A B = (3/5, -4/5) :=
by
  sorry

end unit_vector_opposite_AB_is_l191_191377


namespace solve_for_x_l191_191681

theorem solve_for_x (x : ℝ) (hp : 0 < x) (h : 4 * x^2 = 1024) : x = 16 :=
sorry

end solve_for_x_l191_191681


namespace solveSystem1_solveFractionalEq_l191_191865

-- Definition: system of linear equations
def system1 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ x - 4 * y = 9

-- Theorem: solution to the system of equations
theorem solveSystem1 : ∃ x y : ℝ, system1 x y ∧ x = 5 ∧ y = -1 :=
by
  sorry
  
-- Definition: fractional equation
def fractionalEq (x : ℝ) : Prop :=
  (x + 2) / (x^2 - 2 * x + 1) + 3 / (x - 1) = 0

-- Theorem: solution to the fractional equation
theorem solveFractionalEq : ∃ x : ℝ, fractionalEq x ∧ x = 1 / 4 :=
by
  sorry

end solveSystem1_solveFractionalEq_l191_191865


namespace find_x_parallel_vectors_l191_191938

theorem find_x_parallel_vectors
   (x : ℝ)
   (ha : (x, 2) = (x, 2))
   (hb : (-2, 4) = (-2, 4))
   (hparallel : ∀ (k : ℝ), (x, 2) = (k * -2, k * 4)) :
   x = -1 :=
by
  sorry

end find_x_parallel_vectors_l191_191938


namespace intersect_x_axis_unique_l191_191113

theorem intersect_x_axis_unique (a : ℝ) : (∀ x, (ax^2 + (3 - a) * x + 1) = 0 → x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersect_x_axis_unique_l191_191113


namespace solve_inequality_l191_191847

theorem solve_inequality (x : ℝ) : 
  (-1 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧ 
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 1) ↔ (1 < x) := 
by 
  sorry

end solve_inequality_l191_191847


namespace solve_for_n_l191_191260

theorem solve_for_n (n : ℕ) (h : 2^n * 8^n = 64^(n - 30)) : n = 90 :=
by {
  sorry
}

end solve_for_n_l191_191260


namespace expr_eval_l191_191040

theorem expr_eval : 180 / 6 * 2 + 5 = 65 := by
  sorry

end expr_eval_l191_191040


namespace sin_cos_inequality_l191_191687

open Real

theorem sin_cos_inequality 
  (x : ℝ) (hx : 0 < x ∧ x < π / 2) 
  (m n : ℕ) (hmn : n > m)
  : 2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) :=
sorry

end sin_cos_inequality_l191_191687


namespace probability_neither_defective_l191_191334

noncomputable def n := 9
noncomputable def k := 2
noncomputable def total_pens := 9
noncomputable def defective_pens := 3
noncomputable def non_defective_pens := total_pens - defective_pens

noncomputable def total_combinations := Nat.choose total_pens k
noncomputable def non_defective_combinations := Nat.choose non_defective_pens k

theorem probability_neither_defective :
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 := by
sorry

end probability_neither_defective_l191_191334


namespace problem_statement_l191_191463

theorem problem_statement 
  (p q r x y z a b c : ℝ)
  (h1 : p / x = q / y ∧ q / y = r / z)
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) :=
sorry  -- Proof omitted

end problem_statement_l191_191463


namespace total_problems_l191_191790

theorem total_problems (C : ℕ) (W : ℕ)
  (h1 : C = 20)
  (h2 : 3 * C + 5 * W = 110) : 
  C + W = 30 := by
  sorry

end total_problems_l191_191790


namespace maximum_profit_l191_191420

noncomputable def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def L2 (x : ℝ) : ℝ := 2 * x

theorem maximum_profit :
  (∀ (x1 x2 : ℝ), x1 + x2 = 15 → L1 x1 + L2 x2 ≤ 45.6) := sorry

end maximum_profit_l191_191420


namespace cylinder_volume_ratio_l191_191071

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l191_191071


namespace wind_power_in_scientific_notation_l191_191243

theorem wind_power_in_scientific_notation :
  (56 * 10^6) = (5.6 * 10^7) :=
by
  sorry

end wind_power_in_scientific_notation_l191_191243


namespace kylie_stamps_l191_191163

theorem kylie_stamps (K N : ℕ) (h1 : N = K + 44) (h2 : K + N = 112) : K = 34 :=
by
  sorry

end kylie_stamps_l191_191163


namespace updated_mean_l191_191616

theorem updated_mean (n : ℕ) (observation_mean decrement : ℕ) 
  (h1 : n = 50) (h2 : observation_mean = 200) (h3 : decrement = 15) : 
  ((observation_mean * n - decrement * n) / n = 185) :=
by
  sorry

end updated_mean_l191_191616


namespace kitchen_length_l191_191992

-- Define the conditions
def tile_area : ℕ := 6
def kitchen_width : ℕ := 48
def number_of_tiles : ℕ := 96

-- The total area is the number of tiles times the area of each tile
def total_area : ℕ := number_of_tiles * tile_area

-- Statement to prove the length of the kitchen
theorem kitchen_length : (total_area / kitchen_width) = 12 :=
by
  sorry

end kitchen_length_l191_191992


namespace battery_change_month_battery_change_in_november_l191_191497

theorem battery_change_month :
  (119 % 12) = 11 := by
  sorry

theorem battery_change_in_november (n : Nat) (h1 : n = 18) :
  let month := ((n - 1) * 7) % 12
  month = 11 := by
  sorry

end battery_change_month_battery_change_in_november_l191_191497


namespace domain_of_function_l191_191124

theorem domain_of_function :
  ∀ x : ℝ, (1 / (1 - x) ≥ 0 ∧ 1 - x ≠ 0) ↔ (x < 1) :=
by
  sorry

end domain_of_function_l191_191124


namespace no_valid_coloring_l191_191814

theorem no_valid_coloring (colors : Fin 4 → Prop) (board : Fin 5 → Fin 5 → Fin 4) :
  (∀ i j : Fin 5, ∃ c1 c2 c3 : Fin 4, 
    (c1 ≠ c2) ∧ (c2 ≠ c3) ∧ (c1 ≠ c3) ∧ 
    (board i j = c1 ∨ board i j = c2 ∨ board i j = c3)) → False :=
by
  sorry

end no_valid_coloring_l191_191814


namespace circle_radius_l191_191031

-- Given the equation of a circle, we want to prove its radius
theorem circle_radius : ∀ (x y : ℝ), x^2 + y^2 - 6*y - 16 = 0 → (∃ r, r = 5) :=
  by
    sorry

end circle_radius_l191_191031


namespace inequality_solution_l191_191655

theorem inequality_solution (x : ℝ) (h : 4 ≤ |x + 2| ∧ |x + 2| ≤ 8) :
  (-10 : ℝ) ≤ x ∧ x ≤ -6 ∨ (2 : ℝ) ≤ x ∧ x ≤ 6 :=
sorry

end inequality_solution_l191_191655


namespace olympiad2024_sum_l191_191710

theorem olympiad2024_sum (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_product : A * B * C = 2310) : 
  A + B + C ≤ 390 :=
sorry

end olympiad2024_sum_l191_191710


namespace sum_of_natural_numbers_l191_191578

theorem sum_of_natural_numbers (n : ℕ) (h : n * (n + 1) = 812) : n = 28 := by
  sorry

end sum_of_natural_numbers_l191_191578


namespace time_after_2500_minutes_l191_191069

/-- 
To prove that adding 2500 minutes to midnight on January 1, 2011 results in 
January 2 at 5:40 PM.
-/
theorem time_after_2500_minutes :
  let minutes_in_a_day := 1440 -- 24 hours * 60 minutes
  let minutes_in_an_hour := 60
  let start_time_minutes := 0 -- Midnight January 1, 2011 as zero minutes
  let total_minutes := 2500
  let resulting_minutes := start_time_minutes + total_minutes
  let days_passed := resulting_minutes / minutes_in_a_day
  let remaining_minutes := resulting_minutes % minutes_in_a_day
  let hours := remaining_minutes / minutes_in_an_hour
  let minutes := remaining_minutes % minutes_in_an_hour
  days_passed = 1 ∧ hours = 17 ∧ minutes = 40 :=
by
  -- Proof to be filled in
  sorry

end time_after_2500_minutes_l191_191069


namespace sum_of_coefficients_l191_191991

theorem sum_of_coefficients (a b : ℝ) (h : ∀ x : ℝ, (x > 1 ∧ x < 4) ↔ (ax^2 + bx - 2 > 0)) :
  a + b = 2 :=
by
  sorry

end sum_of_coefficients_l191_191991


namespace expand_expression_l191_191736

theorem expand_expression (x : ℝ) : (2 * x - 3) * (2 * x + 3) * (4 * x ^ 2 + 9) = 4 * x ^ 4 - 81 := by
  sorry

end expand_expression_l191_191736


namespace bus_seat_capacity_l191_191504

theorem bus_seat_capacity (x : ℕ) : 15 * x + (15 - 3) * x + 11 = 92 → x = 3 :=
by
  sorry

end bus_seat_capacity_l191_191504


namespace find_m_of_cos_alpha_l191_191518

theorem find_m_of_cos_alpha (m : ℝ) (h₁ : (2 * Real.sqrt 5) / 5 = m / Real.sqrt (m ^ 2 + 1)) (h₂ : m > 0) : m = 2 :=
sorry

end find_m_of_cos_alpha_l191_191518


namespace number_of_slices_per_pizza_l191_191864

-- Given conditions as definitions in Lean 4
def total_pizzas := 2
def total_slices_per_pizza (S : ℕ) : ℕ := total_pizzas * S
def james_portion : ℚ := 2 / 3
def james_ate_slices (S : ℕ) : ℚ := james_portion * (total_slices_per_pizza S)
def james_ate_exactly := 8

-- The main theorem to prove
theorem number_of_slices_per_pizza (S : ℕ) (h : james_ate_slices S = james_ate_exactly) : S = 6 :=
sorry

end number_of_slices_per_pizza_l191_191864


namespace examination_is_30_hours_l191_191457

noncomputable def examination_time_in_hours : ℝ :=
  let total_questions := 200
  let type_a_problems := 10
  let total_time_on_type_a := 17.142857142857142
  let time_per_type_a := total_time_on_type_a / type_a_problems
  let time_per_type_b := time_per_type_a / 2
  let type_b_problems := total_questions - type_a_problems
  let total_time_on_type_b := time_per_type_b * type_b_problems
  let total_time_in_minutes := total_time_on_type_a * type_a_problems + total_time_on_type_b
  total_time_in_minutes / 60

theorem examination_is_30_hours :
  examination_time_in_hours = 30 := by
  sorry

end examination_is_30_hours_l191_191457


namespace range_of_x_satisfying_inequality_l191_191733

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l191_191733


namespace num_distinct_integers_formed_l191_191403

theorem num_distinct_integers_formed (digits : Multiset ℕ) (h : digits = {2, 2, 3, 3, 3}) : 
  Multiset.card (Multiset.powerset digits).attach = 10 := 
by {
  sorry
}

end num_distinct_integers_formed_l191_191403


namespace Olivia_house_height_l191_191894

variable (h : ℕ)
variable (flagpole_height : ℕ := 35)
variable (flagpole_shadow : ℕ := 30)
variable (house_shadow : ℕ := 70)
variable (bush_height : ℕ := 14)
variable (bush_shadow : ℕ := 12)

theorem Olivia_house_height :
  (house_shadow / flagpole_shadow) * flagpole_height = 81 ∧
  (house_shadow / bush_shadow) * bush_height = 81 :=
by
  sorry

end Olivia_house_height_l191_191894


namespace pigs_total_l191_191507

theorem pigs_total (initial_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joined_pigs = 22) 
  : total_pigs = 86 :=
by
  sorry

end pigs_total_l191_191507


namespace imo_1988_problem_29_l191_191595

variable (d r : ℕ)
variable (h1 : d > 1)
variable (h2 : 1059 % d = r)
variable (h3 : 1417 % d = r)
variable (h4 : 2312 % d = r)

theorem imo_1988_problem_29 :
  d - r = 15 := by sorry

end imo_1988_problem_29_l191_191595


namespace percentage_expression_l191_191139

variable {A B : ℝ} (hA : A > 0) (hB : B > 0)

theorem percentage_expression (h : A = (x / 100) * B) : x = 100 * (A / B) :=
sorry

end percentage_expression_l191_191139


namespace reservoir_percentage_before_storm_l191_191754

variable (total_capacity : ℝ)
variable (water_after_storm : ℝ := 220 + 110)
variable (percentage_after_storm : ℝ := 0.60)
variable (original_contents : ℝ := 220)

theorem reservoir_percentage_before_storm :
  total_capacity = water_after_storm / percentage_after_storm →
  (original_contents / total_capacity) * 100 = 40 :=
by
  sorry

end reservoir_percentage_before_storm_l191_191754


namespace tan_22_5_eq_half_l191_191238

noncomputable def tan_h_LHS (θ : Real) := Real.tan θ / (1 - Real.tan θ ^ 2)

theorem tan_22_5_eq_half :
    tan_h_LHS (Real.pi / 8) = 1 / 2 :=
  sorry

end tan_22_5_eq_half_l191_191238


namespace smallest_possible_odd_b_l191_191856

theorem smallest_possible_odd_b 
    (a b : ℕ) 
    (h1 : a + b = 90) 
    (h2 : Nat.Prime a) 
    (h3 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ b) 
    (h4 : a > b) 
    (h5 : b % 2 = 1) 
    : b = 85 := 
sorry

end smallest_possible_odd_b_l191_191856


namespace point_in_second_quadrant_l191_191654

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l191_191654


namespace playground_area_l191_191476

theorem playground_area :
  ∃ (l w : ℝ), 2 * l + 2 * w = 84 ∧ l = 3 * w ∧ l * w = 330.75 :=
by
  sorry

end playground_area_l191_191476


namespace find_antonym_word_l191_191867

-- Defining the condition that the word means "rarely" or "not often."
def means_rarely_or_not_often (word : String) : Prop :=
  word = "seldom"

-- Theorem statement: There exists a word such that it meets the given condition.
theorem find_antonym_word : 
  ∃ word : String, means_rarely_or_not_often word :=
by
  use "seldom"
  unfold means_rarely_or_not_often
  rfl

end find_antonym_word_l191_191867


namespace principal_amount_l191_191812

noncomputable def exponential (r t : ℝ) :=
  Real.exp (r * t)

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 5673981 ∧ r = 0.1125 ∧ t = 7.5 ∧ P = 2438978.57 →
  P = A / exponential r t := 
by
  intros h
  sorry

end principal_amount_l191_191812


namespace lcm_of_36_and_100_l191_191086

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l191_191086


namespace isosceles_triangle_perimeter_l191_191013

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a^2 - 9 * a + 18 = 0) (h2 : b^2 - 9 * b + 18 = 0) (h3 : a ≠ b) :
  a + 2 * b = 15 :=
by
  -- Proof is omitted.
  sorry

end isosceles_triangle_perimeter_l191_191013


namespace ticket_price_profit_condition_maximize_profit_at_7_point_5_l191_191162

-- Define the ticket price increase and the total profit function
def ticket_price (x : ℝ) := (10 + x) * (500 - 20 * x)

-- Prove that the function equals 6000 at x = 10 and x = 25
theorem ticket_price_profit_condition (x : ℝ) :
  ticket_price x = 6000 ↔ (x = 10 ∨ x = 25) :=
by sorry

-- Prove that m = 7.5 maximizes the profit
def profit (m : ℝ) := -20 * m^2 + 300 * m + 5000

theorem maximize_profit_at_7_point_5 (m : ℝ) :
  m = 7.5 ↔ (∀ m, profit 7.5 ≥ profit m) :=
by sorry

end ticket_price_profit_condition_maximize_profit_at_7_point_5_l191_191162


namespace probability_at_least_6_heads_8_flips_l191_191729

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l191_191729


namespace library_hospital_community_center_bells_ring_together_l191_191234

theorem library_hospital_community_center_bells_ring_together :
  ∀ (library hospital community : ℕ), 
    (library = 18) → (hospital = 24) → (community = 30) → 
    (∀ t, (t = 0) ∨ (∃ n₁ n₂ n₃ : ℕ, 
      t = n₁ * library ∧ t = n₂ * hospital ∧ t = n₃ * community)) → 
    true :=
by
  intros
  sorry

end library_hospital_community_center_bells_ring_together_l191_191234


namespace area_of_rectangle_l191_191281

-- Definitions from problem conditions
variable (AB CD x : ℝ)
variable (h1 : AB = 24)
variable (h2 : CD = 60)
variable (h3 : BC = x)
variable (h4 : BF = 2 * x)
variable (h5 : similar (triangle AEB) (triangle FDC))

-- Goal: Prove the area of rectangle BCFE
theorem area_of_rectangle (h1 : AB = 24) (h2 : CD = 60) (x y : ℝ) 
  (h3 : BC = x) (h4 : BF = 2 * x) (h5 : BC * BF = y) : y = 1440 :=
sorry -- proof will be provided here

end area_of_rectangle_l191_191281


namespace combined_degrees_l191_191289

theorem combined_degrees (S J W : ℕ) (h1 : S = 150) (h2 : J = S - 5) (h3 : W = S - 3) : S + J + W = 442 :=
by
  sorry

end combined_degrees_l191_191289


namespace lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l191_191345

def total_area_of_triangles_and_quadrilateral (A B Q : ℝ) : ℝ :=
  A + B + Q

def lena_triangles_and_quadrilateral_area (A B Q : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_quadrilateral A B Q

def total_area_of_triangles_and_pentagon (C D P : ℝ) : ℝ :=
  C + D + P

def vasya_triangles_and_pentagon_area (C D P : ℝ) : Prop :=
  (24 : ℝ) = total_area_of_triangles_and_pentagon C D P

theorem lena_can_form_rectangles (A B Q : ℝ) (h : lena_triangles_and_quadrilateral_area A B Q) :
  lena_triangles_and_quadrilateral_area A B Q :=
by 
-- We assume the definition holds as given
sorry

theorem vasya_can_form_rectangles (C D P : ℝ) (h : vasya_triangles_and_pentagon_area C D P) :
  vasya_triangles_and_pentagon_area C D P :=
by 
-- We assume the definition holds as given
sorry

theorem lena_and_vasya_can_be_right (A B Q C D P : ℝ)
  (hlena : lena_triangles_and_quadrilateral_area A B Q)
  (hvasya : vasya_triangles_and_pentagon_area C D P) :
  lena_triangles_and_quadrilateral_area A B Q ∧ vasya_triangles_and_pentagon_area C D P :=
by 
-- Combining both assumptions
exact ⟨hlena, hvasya⟩

end lena_can_form_rectangles_vasya_can_form_rectangles_lena_and_vasya_can_be_right_l191_191345


namespace eccentricity_ratio_l191_191192

noncomputable def ellipse_eccentricity (m n : ℝ) : ℝ := (1 - (1 / n) / (1 / m))^(1/2)

theorem eccentricity_ratio (m n : ℝ) (h : ellipse_eccentricity m n = 1 / 2) :
  m / n = 3 / 4 :=
by
  sorry

end eccentricity_ratio_l191_191192


namespace hyperbola_equation_standard_form_l191_191405

noncomputable def point_on_hyperbola_asymptote (A : ℝ × ℝ) (C : ℝ) : Prop :=
  let x := A.1
  let y := A.2
  (4 * y^2 - x^2 = C) ∧
  (y = (1/2) * x ∨ y = -(1/2) * x)

theorem hyperbola_equation_standard_form
  (A : ℝ × ℝ)
  (hA : A = (2 * Real.sqrt 2, 2))
  (asymptote1 asymptote2 : ℝ → ℝ)
  (hasymptote1 : ∀ x, asymptote1 x = (1/2) * x)
  (hasymptote2 : ∀ x, asymptote2 x = -(1/2) * x) :
  (∃ C : ℝ, point_on_hyperbola_asymptote A C) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (4 * (A.2)^2 - (A.1)^2 = 8) ∧ 
    (∀ x y : ℝ, (4 * y^2 - x^2 = 8) ↔ ((y^2) / a - (x^2) / b = 1))) :=
by
  sorry

end hyperbola_equation_standard_form_l191_191405


namespace total_population_l191_191321

-- Definitions based on given conditions
variables (b g t : ℕ)
variables (h1 : b = 4 * g) (h2 : g = 8 * t)

-- Theorem statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * t :=
by
  sorry

end total_population_l191_191321


namespace disease_cases_1975_l191_191709

theorem disease_cases_1975 (cases_1950 cases_2000 : ℕ) (cases_1950_eq : cases_1950 = 500000)
  (cases_2000_eq : cases_2000 = 1000) (linear_decrease : ∀ t : ℕ, 1950 ≤ t ∧ t ≤ 2000 →
  ∃ k : ℕ, cases_1950 - (k * (t - 1950)) = cases_2000) : 
  ∃ cases_1975 : ℕ, cases_1975 = 250500 := 
by
  -- Setting up known values
  let decrease_duration := 2000 - 1950
  let total_decrease := cases_1950 - cases_2000
  let annual_decrease := total_decrease / decrease_duration
  let years_from_1950_to_1975 := 1975 - 1950
  let decline_by_1975 := annual_decrease * years_from_1950_to_1975
  let cases_1975 := cases_1950 - decline_by_1975
  -- Returning the desired value
  use cases_1975
  sorry

end disease_cases_1975_l191_191709


namespace polynomial_solution_l191_191591
-- Import necessary library

-- Define the property to be checked
def polynomial_property (P : Real → Real) : Prop :=
  ∀ a b c : Real, 
    P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

-- The statement that needs to be proven
theorem polynomial_solution (a b : Real) : polynomial_property (λ x => a * x^2 + b * x) := 
by
  sorry

end polynomial_solution_l191_191591


namespace largest_value_in_interval_l191_191100

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x^3, 3*x, x^(1/3), 1/x} : Set ℝ), y ≤ 1/x) :=
sorry

end largest_value_in_interval_l191_191100


namespace sum_arithmetic_series_eq_499500_l191_191804

theorem sum_arithmetic_series_eq_499500 :
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  (n * (a1 + an) / 2) = 499500 := by {
  let a1 := 1
  let an := 999
  let n := 999
  let d := 1
  show (n * (a1 + an) / 2) = 499500
  sorry
}

end sum_arithmetic_series_eq_499500_l191_191804


namespace find_x_l191_191899

theorem find_x (x : ℤ) (h : 7 * x - 18 = 66) : x = 12 :=
  sorry

end find_x_l191_191899


namespace all_numbers_rational_l191_191738

-- Define the mathematical operations for the problem
def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem all_numbers_rational :
    (∃ x1 : ℚ, fourth_root 81 = x1) ∧
    (∃ x2 : ℚ, square_root 0.64 = x2) ∧
    (∃ x3 : ℚ, cube_root 0.001 = x3) ∧
    (∃ x4 : ℚ, (cube_root 8) * (square_root ((0.25)⁻¹)) = x4) :=
  sorry

end all_numbers_rational_l191_191738


namespace determine_k_range_l191_191331

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def g (x : ℝ) : ℝ := (Real.log x) / x
noncomputable def h (x : ℝ) : ℝ := (Real.log x) / (x * x)

theorem determine_k_range :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) → f k x = g x) →
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1) ∧ x2 ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) →
  k ∈ Set.Ico (1 / (Real.exp 1) ^ 2) (1 / (2 * Real.exp 1)) := 
  sorry

end determine_k_range_l191_191331


namespace area_of_OPF_eq_sqrt_2_div_2_l191_191859

noncomputable def area_of_triangle_OPF : ℝ :=
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  if (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) then
    let base := dist O F
    let height := Real.sqrt 2
    (1 / 2) * base * height
  else
    0

theorem area_of_OPF_eq_sqrt_2_div_2 : 
  let O : ℝ × ℝ := (0, 0)
  let F : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (0.5, Real.sqrt 2) -- We assume P is (1/2, sqrt(2))
  let P1 : ℝ × ℝ := (0.5, -Real.sqrt 2) -- We also define the other point P1
  (dist O P = dist P F) ∨ (dist O P1 = dist P1 F) →
  let base := dist O F
  let height := Real.sqrt 2
  area_of_triangle_OPF = Real.sqrt 2 / 2 := 
by 
  sorry

end area_of_OPF_eq_sqrt_2_div_2_l191_191859


namespace function_range_is_interval_l191_191494

theorem function_range_is_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ∧ 
  (256 * x^9 - 576 * x^7 + 432 * x^5 - 120 * x^3 + 9 * x) ≤ 1 := 
by
  sorry

end function_range_is_interval_l191_191494


namespace roof_area_l191_191936

theorem roof_area (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : l - w = 28) : 
  l * w = 3136 / 9 := 
by 
  sorry

end roof_area_l191_191936


namespace sum_of_perimeters_l191_191917

theorem sum_of_perimeters (s : ℝ) : (∀ n : ℕ, n >= 0) → 
  (∑' n : ℕ, (4 * s) / (2 ^ n)) = 8 * s :=
by
  sorry

end sum_of_perimeters_l191_191917


namespace product_of_last_two_digits_l191_191098

theorem product_of_last_two_digits (A B : ℕ) (hn1 : 10 * A + B ≡ 0 [MOD 5]) (hn2 : A + B = 16) : A * B = 30 :=
sorry

end product_of_last_two_digits_l191_191098


namespace part_a_part_b_l191_191878

-- Part (a)
theorem part_a (n : ℕ) (h : n > 0) :
  (2 * n ∣ n * (n + 1) / 2) ↔ ∃ k : ℕ, n = 4 * k - 1 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n > 0) :
  (2 * n + 1 ∣ n * (n + 1) / 2) ↔ (2 * n + 1 ≡ 1 [MOD 4]) ∨ (2 * n + 1 ≡ 3 [MOD 4]) :=
by sorry

end part_a_part_b_l191_191878


namespace DavidCrunchesLessThanZachary_l191_191174

-- Definitions based on conditions
def ZacharyPushUps : ℕ := 44
def ZacharyCrunches : ℕ := 17
def DavidPushUps : ℕ := ZacharyPushUps + 29
def DavidCrunches : ℕ := 4

-- Problem statement we need to prove:
theorem DavidCrunchesLessThanZachary : DavidCrunches = ZacharyCrunches - 13 :=
by
  -- Proof will go here
  sorry

end DavidCrunchesLessThanZachary_l191_191174


namespace only_solution_for_triplet_l191_191544

theorem only_solution_for_triplet (x y z : ℤ) (h : x^2 + y^2 + z^2 - 2 * x * y * z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end only_solution_for_triplet_l191_191544


namespace min_major_axis_length_l191_191460

theorem min_major_axis_length (a b c : ℝ) (h_area : b * c = 1) (h_focal_relation : 2 * a = 2 * Real.sqrt (b^2 + c^2)) :
  2 * a = 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_length_l191_191460


namespace min_translation_phi_l191_191479

theorem min_translation_phi (φ : ℝ) (hφ : φ > 0) : 
  (∃ k : ℤ, φ = (π / 3) - k * π) → φ = π / 3 := 
by 
  sorry

end min_translation_phi_l191_191479


namespace sum_of_squares_eq_three_l191_191258

theorem sum_of_squares_eq_three
  (a b s : ℝ)
  (h₀ : a ≠ b)
  (h₁ : a * s^2 + b * s + b = 0)
  (h₂ : a * (1 / s)^2 + a * (1 / s) + b = 0)
  (h₃ : s * (1 / s) = 1) :
  s^2 + (1 / s)^2 = 3 := 
sorry

end sum_of_squares_eq_three_l191_191258


namespace ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l191_191416

variable (a b : ℝ)
variable (m x : ℝ)
variable (t : ℝ)
variable (A B C : ℝ)

-- Define ideal points
def is_ideal_point (p : ℝ × ℝ) := p.snd = 2 * p.fst

-- Define the conditions for question 1
def distance_from_y_axis (a : ℝ) := abs a = 2

-- Question 1: Prove that M(2, 4) or M(-2, -4)
theorem ideal_point_distance_y_axis (a b : ℝ) (h1 : is_ideal_point (a, b)) (h2 : distance_from_y_axis a) :
  (a = 2 ∧ b = 4) ∨ (a = -2 ∧ b = -4) := sorry

-- Define the linear function
def linear_func (m x : ℝ) : ℝ := 3 * m * x - 1

-- Question 2: Prove or disprove the existence of ideal points in y = 3mx - 1
theorem exists_ideal_point_linear (m x : ℝ) (hx : is_ideal_point (x, linear_func m x)) :
  (m ≠ 2/3 → ∃ x, linear_func m x = 2 * x) ∧ (m = 2/3 → ¬ ∃ x, linear_func m x = 2 * x) := sorry

-- Question 3 conditions
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def quadratic_conditions (a b c : ℝ) : Prop :=
  (quadratic_func a b c 0 = 5 * a + 1) ∧ (quadratic_func a b c (-2) = 5 * a + 1)

-- Question 3: Prove the range of t = a^2 + a + 1 given the quadratic conditions
theorem range_of_t (a b c t : ℝ) (h1 : is_ideal_point (x, quadratic_func a b c x))
  (h2 : quadratic_conditions a b c) (ht : t = a^2 + a + 1) :
    3 / 4 ≤ t ∧ t ≤ 21 / 16 ∧ t ≠ 1 := sorry

end ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l191_191416


namespace simplify_expression_l191_191556

variable (a : ℝ)

theorem simplify_expression :
    5 * a^2 - (a^2 - 2 * (a^2 - 3 * a)) = 6 * a^2 - 6 * a := by
  sorry

end simplify_expression_l191_191556


namespace average_speed_to_SF_l191_191042

theorem average_speed_to_SF (v d : ℝ) (h1 : d ≠ 0) (h2 : v ≠ 0) :
  (2 * d / ((d / v) + (2 * d / v)) = 34) → v = 51 :=
by
  -- proof goes here
  sorry

end average_speed_to_SF_l191_191042


namespace triangle_orthocenter_example_l191_191230

open Real EuclideanGeometry

def point_3d := (ℝ × ℝ × ℝ)

def orthocenter (A B C : point_3d) : point_3d := sorry

theorem triangle_orthocenter_example :
  orthocenter (2, 4, 6) (6, 5, 3) (4, 6, 7) = (4/5, 38/5, 59/5) := sorry

end triangle_orthocenter_example_l191_191230


namespace evaluate_expression_l191_191488

theorem evaluate_expression : (2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9) := by
    sorry

end evaluate_expression_l191_191488


namespace max_abs_sum_l191_191641

-- Define the condition for the ellipse equation
def ellipse_condition (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Prove that the largest possible value of |x| + |y| given the condition is 2√3
theorem max_abs_sum (x y : ℝ) (h : ellipse_condition x y) : |x| + |y| ≤ 2 * Real.sqrt 3 :=
sorry

end max_abs_sum_l191_191641


namespace rate_of_stream_equation_l191_191962

theorem rate_of_stream_equation 
  (v : ℝ) 
  (boat_speed : ℝ) 
  (travel_time : ℝ) 
  (distance : ℝ)
  (h_boat_speed : boat_speed = 16)
  (h_travel_time : travel_time = 5)
  (h_distance : distance = 105)
  (h_equation : distance = (boat_speed + v) * travel_time) : v = 5 :=
by 
  sorry

end rate_of_stream_equation_l191_191962


namespace sum_of_powers_eq_123_l191_191446

section

variables {a b : Real}

-- Conditions provided in the problem
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7

-- Define the theorem to be proved
theorem sum_of_powers_eq_123 : a^10 + b^10 = 123 :=
sorry

end

end sum_of_powers_eq_123_l191_191446


namespace remainder_seven_power_twenty_seven_l191_191407

theorem remainder_seven_power_twenty_seven :
  (7^27) % 1000 = 543 := 
sorry

end remainder_seven_power_twenty_seven_l191_191407


namespace find_A_and_B_l191_191713

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (∀ x : ℝ, x ≠ 10 ∧ x ≠ -3 → 5*x + 2 = A * (x + 3) + B * (x - 10)) ∧ 
    A = 4 ∧ B = 1 :=
  sorry

end find_A_and_B_l191_191713


namespace probability_white_ball_from_first_urn_correct_l191_191632

noncomputable def probability_white_ball_from_first_urn : ℝ :=
  let p_H1 : ℝ := 0.5
  let p_H2 : ℝ := 0.5
  let p_A_given_H1 : ℝ := 0.7
  let p_A_given_H2 : ℝ := 0.6
  let p_A : ℝ := p_H1 * p_A_given_H1 + p_H2 * p_A_given_H2
  p_H1 * p_A_given_H1 / p_A

theorem probability_white_ball_from_first_urn_correct :
  probability_white_ball_from_first_urn = 0.538 :=
sorry

end probability_white_ball_from_first_urn_correct_l191_191632


namespace train_speed_correct_l191_191953

noncomputable def jogger_speed_km_per_hr := 9
noncomputable def jogger_speed_m_per_s := 9 * 1000 / 3600
noncomputable def train_speed_km_per_hr := 45
noncomputable def distance_ahead_m := 270
noncomputable def train_length_m := 120
noncomputable def total_distance_m := distance_ahead_m + train_length_m
noncomputable def time_seconds := 39

theorem train_speed_correct :
  let relative_speed_m_per_s := total_distance_m / time_seconds
  let train_speed_m_per_s := relative_speed_m_per_s + jogger_speed_m_per_s
  let train_speed_km_per_hr_calculated := train_speed_m_per_s * 3600 / 1000
  train_speed_km_per_hr_calculated = train_speed_km_per_hr :=
by
  sorry

end train_speed_correct_l191_191953


namespace base_addition_l191_191787

theorem base_addition (b : ℕ) (h : b > 1) :
  (2 * b^3 + 3 * b^2 + 8 * b + 4) + (3 * b^3 + 4 * b^2 + 1 * b + 7) = 
  1 * b^4 + 0 * b^3 + 2 * b^2 + 0 * b + 1 → b = 10 :=
by
  intro H
  -- skipping the detailed proof steps
  sorry

end base_addition_l191_191787


namespace terminal_side_same_line_37_and_neg143_l191_191824

theorem terminal_side_same_line_37_and_neg143 :
  ∃ k : ℤ, (37 : ℝ) + 180 * k = (-143 : ℝ) :=
by
  -- Proof steps go here
  sorry

end terminal_side_same_line_37_and_neg143_l191_191824


namespace passengers_initial_count_l191_191147

-- Let's define the initial number of passengers
variable (P : ℕ)

-- Given conditions:
def final_passengers (initial additional left : ℕ) : ℕ := initial + additional - left

-- The theorem statement to prove P = 28 given the conditions
theorem passengers_initial_count
  (final_count : ℕ)
  (h1 : final_count = 26)
  (h2 : final_passengers P 7 9 = final_count) 
  : P = 28 :=
by
  sorry

end passengers_initial_count_l191_191147


namespace maximum_distinct_numbers_l191_191855

theorem maximum_distinct_numbers (n : ℕ) (hsum : n = 250) : 
  ∃ k ≤ 21, k = 21 :=
by
  sorry

end maximum_distinct_numbers_l191_191855


namespace blocks_per_friend_l191_191060

theorem blocks_per_friend (total_blocks : ℕ) (friends : ℕ) (h1 : total_blocks = 28) (h2 : friends = 4) :
  total_blocks / friends = 7 :=
by
  sorry

end blocks_per_friend_l191_191060


namespace exists_root_between_roots_l191_191037

theorem exists_root_between_roots 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0) 
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) 
  (hx : x₁ < x₂) :
  ∃ x₃ : ℝ, x₁ < x₃ ∧ x₃ < x₂ ∧ (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by 
  sorry

end exists_root_between_roots_l191_191037


namespace find_vector_result_l191_191712

-- Define the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m: ℝ) : ℝ × ℝ := (-2, m)
def m := -4
def result := 2 • vector_a + 3 • vector_b m

-- State the theorem
theorem find_vector_result : result = (-4, -8) := 
by {
  -- skipping the proof
  sorry
}

end find_vector_result_l191_191712


namespace monotonic_criteria_l191_191634

noncomputable def monotonic_interval (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ 4 → 
  (-2 * x₁^2 + m * x₁ + 1) ≤ (-2 * x₂^2 + m * x₂ + 1)

theorem monotonic_criteria (m : ℝ) : 
  (m ≤ -4 ∨ m ≥ 16) ↔ monotonic_interval m := 
sorry

end monotonic_criteria_l191_191634


namespace decreasing_function_range_l191_191922

theorem decreasing_function_range (f : ℝ → ℝ) (a : ℝ) (h_decreasing : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 → -1 < x2 ∧ x2 < 1 ∧ x1 > x2 → f x1 < f x2)
  (h_ineq: f (1 - a) < f (3 * a - 1)) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l191_191922


namespace frequency_not_equal_probability_l191_191449

theorem frequency_not_equal_probability
  (N : ℕ) -- Total number of trials
  (N1 : ℕ) -- Number of times student A is selected
  (hN : N > 0) -- Ensure the number of trials is positive
  (rand_int_gen : ℕ → ℕ) -- A function generating random integers from 1 to 6
  (h_gen : ∀ n, 1 ≤ rand_int_gen n ∧ rand_int_gen n ≤ 6) -- Generator produces numbers between 1 to 6
: (N1/N : ℚ) ≠ (1/6 : ℚ) := 
sorry

end frequency_not_equal_probability_l191_191449


namespace units_digit_fraction_mod_10_l191_191796

theorem units_digit_fraction_mod_10 : (30 * 32 * 34 * 36 * 38 * 40) % 2000 % 10 = 2 := by
  sorry

end units_digit_fraction_mod_10_l191_191796


namespace total_gulbis_is_correct_l191_191612

-- Definitions based on given conditions
def num_dureums : ℕ := 156
def num_gulbis_in_one_dureum : ℕ := 20

-- Definition of total gulbis calculated
def total_gulbis : ℕ := num_dureums * num_gulbis_in_one_dureum

-- Statement to prove
theorem total_gulbis_is_correct : total_gulbis = 3120 := by
  -- The actual proof would go here
  sorry

end total_gulbis_is_correct_l191_191612


namespace intersection_P_Q_l191_191068

def P := {x : ℝ | 1 < x ∧ x < 3}
def Q := {x : ℝ | 2 < x}

theorem intersection_P_Q :
  P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := sorry

end intersection_P_Q_l191_191068


namespace transylvanian_convinces_l191_191892

theorem transylvanian_convinces (s : Prop) (t : Prop) (h : s ↔ (¬t ∧ ¬s)) : t :=
by
  -- Leverage the existing equivalence to prove the desired result
  sorry

end transylvanian_convinces_l191_191892


namespace right_triangle_ratio_segments_l191_191802

theorem right_triangle_ratio_segments (a b c r s : ℝ) (h : a^2 + b^2 = c^2) (h_drop : r + s = c) (a_to_b_ratio : 2 * b = 5 * a) : r / s = 4 / 25 :=
sorry

end right_triangle_ratio_segments_l191_191802


namespace min_students_participating_l191_191846

def ratio_9th_to_10th (n9 n10 : ℕ) : Prop := n9 * 4 = n10 * 3
def ratio_10th_to_11th (n10 n11 : ℕ) : Prop := n10 * 6 = n11 * 5

theorem min_students_participating (n9 n10 n11 : ℕ) 
    (h1 : ratio_9th_to_10th n9 n10) 
    (h2 : ratio_10th_to_11th n10 n11) : 
    n9 + n10 + n11 = 59 :=
sorry

end min_students_participating_l191_191846


namespace shark_sightings_l191_191389

theorem shark_sightings (x : ℕ) 
  (h1 : 26 = 5 + 3 * x) : x = 7 :=
by
  sorry

end shark_sightings_l191_191389


namespace find_arrays_l191_191079

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays_l191_191079


namespace amount_A_received_l191_191986

-- Define the conditions
def total_amount : ℕ := 600
def ratio_a : ℕ := 1
def ratio_b : ℕ := 2

-- Define the total parts in the ratio
def total_parts : ℕ := ratio_a + ratio_b

-- Define the value of one part
def value_per_part : ℕ := total_amount / total_parts

-- Define the amount A gets
def amount_A_gets : ℕ := ratio_a * value_per_part

-- Lean statement to prove
theorem amount_A_received : amount_A_gets = 200 := by
  sorry

end amount_A_received_l191_191986


namespace no_prime_ratio_circle_l191_191532

theorem no_prime_ratio_circle (A : Fin 2007 → ℕ) :
  ¬ (∀ i : Fin 2007, (∃ p : ℕ, Nat.Prime p ∧ (p = A i / A ((i + 1) % 2007) ∨ p = A ((i + 1) % 2007) / A i))) := by
  sorry

end no_prime_ratio_circle_l191_191532


namespace rectangle_tileable_iff_divisible_l191_191852

def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def tileable_with_0b_tiles (m n b : ℕ) : Prop :=
  ∃ t : ℕ, t * (2 * b) = m * n  -- This comes from the total area divided by the area of one tile

theorem rectangle_tileable_iff_divisible (m n b : ℕ) :
  tileable_with_0b_tiles m n b ↔ divisible_by (2 * b) m ∨ divisible_by (2 * b) n := 
sorry

end rectangle_tileable_iff_divisible_l191_191852


namespace schedule_arrangement_count_l191_191217

-- Given subjects
inductive Subject
| Chinese
| Mathematics
| Politics
| English
| PhysicalEducation
| Art

open Subject

-- Define a function to get the total number of different arrangements
def arrangement_count : Nat := 192

-- The proof statement (problem restated in Lean 4)
theorem schedule_arrangement_count :
  arrangement_count = 192 :=
by
  sorry

end schedule_arrangement_count_l191_191217


namespace repair_time_l191_191020

theorem repair_time {x : ℝ} :
  (∀ (a b : ℝ), a = 3 ∧ b = 6 → (((1 / a) + (1 / b)) * x = 1) → x = 2) :=
by
  intros a b hab h
  rcases hab with ⟨ha, hb⟩
  sorry

end repair_time_l191_191020


namespace range_of_x_for_sqrt_l191_191087

-- Define the condition under which the expression inside the square root is non-negative.
def sqrt_condition (x : ℝ) : Prop :=
  x - 7 ≥ 0

-- Main theorem to prove the range of values for x
theorem range_of_x_for_sqrt (x : ℝ) : sqrt_condition x ↔ x ≥ 7 :=
by
  -- Proof steps go here (omitted as per instructions)
  sorry

end range_of_x_for_sqrt_l191_191087


namespace total_numbers_l191_191272

-- Setting up constants and conditions
variables (n : ℕ)
variables (s1 s2 s3 : ℕ → ℝ)

-- Conditions
axiom avg_all : (s1 n + s2 n + s3 n) / n = 2.5
axiom avg_2_1 : s1 2 / 2 = 1.1
axiom avg_2_2 : s2 2 / 2 = 1.4
axiom avg_2_3 : s3 2 / 2 = 5.0

-- Proposed theorem to prove
theorem total_numbers : n = 6 :=
by
  sorry

end total_numbers_l191_191272


namespace parabola_directrix_l191_191782

theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (O D : ℝ × ℝ) :
  A ≠ B →
  O = (0, 0) →
  D = (1, 2) →
  (∃ k, k = ((2:ℝ) - 0) / ((1:ℝ) - 0) ∧ k = 2) →
  (∃ k, k = - 1 / 2) →
  (∀ x y, y^2 = 2 * p * x) →
  p = 5 / 2 →
  O.1 * A.1 + O.2 * A.2 = 0 →
  O.1 * B.1 + O.2 * B.2 = 0 →
  A.1 * B.1 + A.2 * B.2 = 0 →
  (∃ k, (y - 2) = k * (x - 1) ∧ (A.1 * B.1) = 25 ∧ (A.1 + B.1) = 10 + 8 * p) →
  ∃ dir_eq, dir_eq = -5 / 4 :=
by
  sorry

end parabola_directrix_l191_191782


namespace shaded_region_area_l191_191598

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end shaded_region_area_l191_191598


namespace sum_of_first_two_digits_of_repeating_decimal_l191_191186

theorem sum_of_first_two_digits_of_repeating_decimal (c d : ℕ) (h : (c, d) = (3, 5)) : c + d = 8 :=
by 
  sorry

end sum_of_first_two_digits_of_repeating_decimal_l191_191186


namespace original_decimal_number_l191_191739

theorem original_decimal_number (x : ℝ) (h : 0.375 = (x / 1000) * 10) : x = 37.5 :=
sorry

end original_decimal_number_l191_191739


namespace first_solution_carbonation_l191_191964

-- Definitions of given conditions in the problem
variable (C : ℝ) -- Percentage of carbonated water in the first solution
variable (L : ℝ) -- Percentage of lemonade in the first solution

-- The second solution is 55% carbonated water and 45% lemonade
def second_solution_carbonated : ℝ := 55
def second_solution_lemonade : ℝ := 45

-- The mixture is 65% carbonated water and 40% of the volume is the first solution
def mixture_carbonated : ℝ := 65
def first_solution_contribution : ℝ := 0.40
def second_solution_contribution : ℝ := 0.60

-- The relationship between the solution components
def equation := first_solution_contribution * C + second_solution_contribution * second_solution_carbonated = mixture_carbonated

-- The statement to prove: C = 80
theorem first_solution_carbonation :
  equation C →
  C = 80 :=
sorry

end first_solution_carbonation_l191_191964


namespace find_ab_l191_191998

variable (a b m n : ℝ)

theorem find_ab (h1 : (a + b)^2 = m) (h2 : (a - b)^2 = n) : 
  a * b = (m - n) / 4 :=
by
  sorry

end find_ab_l191_191998


namespace min_value_of_quadratic_l191_191341

theorem min_value_of_quadratic (x : ℝ) : 
  ∃ m : ℝ, (∀ z : ℝ, z = 5 * x ^ 2 + 20 * x + 25 → z ≥ m) ∧ m = 5 :=
by
  sorry

end min_value_of_quadratic_l191_191341


namespace sample_size_l191_191946

theorem sample_size (f r n : ℕ) (freq_def : f = 36) (rate_def : r = 25 / 100) (relation : r = f / n) : n = 144 :=
sorry

end sample_size_l191_191946


namespace number_of_triangles_in_decagon_l191_191471

theorem number_of_triangles_in_decagon :
  let decagon_vertices := 10
  let vertices_needed_for_triangle := 3
  ∀ (no_three_collinear: ∀ (a b c : Fin decagon_vertices), a ≠ b ∧ b ≠ c ∧ c ≠ a -> True), 
  Nat.choose decagon_vertices vertices_needed_for_triangle = 120 := by
  sorry

end number_of_triangles_in_decagon_l191_191471


namespace sandy_initial_books_l191_191422

-- Define the initial conditions as given.
def books_tim : ℕ := 33
def books_lost : ℕ := 24
def books_after_loss : ℕ := 19

-- Define the equation for the total books before Benny's loss and solve for Sandy's books.
def books_total_before_loss : ℕ := books_after_loss + books_lost
def books_sandy_initial : ℕ := books_total_before_loss - books_tim

-- Assert the proof statement:
def proof_sandy_books : Prop :=
  books_sandy_initial = 10

theorem sandy_initial_books : proof_sandy_books := by
  -- Placeholder for the actual proof.
  sorry

end sandy_initial_books_l191_191422


namespace excess_calories_l191_191870

-- Conditions
def calories_from_cheezits (bags: ℕ) (ounces_per_bag: ℕ) (calories_per_ounce: ℕ) : ℕ :=
  bags * ounces_per_bag * calories_per_ounce

def calories_from_chocolate_bars (bars: ℕ) (calories_per_bar: ℕ) : ℕ :=
  bars * calories_per_bar

def calories_from_popcorn (calories: ℕ) : ℕ :=
  calories

def calories_burned_running (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_swimming (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_cycling (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

-- Hypothesis
def total_calories_consumed : ℕ :=
  calories_from_cheezits 3 2 150 + calories_from_chocolate_bars 2 250 + calories_from_popcorn 500

def total_calories_burned : ℕ :=
  calories_burned_running 40 12 + calories_burned_swimming 30 15 + calories_burned_cycling 20 10

-- Theorem
theorem excess_calories : total_calories_consumed - total_calories_burned = 770 := by
  sorry

end excess_calories_l191_191870


namespace abs_a_k_le_fractional_l191_191443

variable (a : ℕ → ℝ) (n : ℕ)

-- Condition 1: a_0 = a_(n+1) = 0
axiom a_0 : a 0 = 0
axiom a_n1 : a (n + 1) = 0

-- Condition 2: |a_{k-1} - 2a_k + a_{k+1}| ≤ 1 for k = 1, 2, ..., n
axiom abs_diff_ineq (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1

-- Theorem statement
theorem abs_a_k_le_fractional (k : ℕ) (h : 0 ≤ k ∧ k ≤ n + 1) : 
  |a k| ≤ k * (n + 1 - k) / 2 := sorry

end abs_a_k_le_fractional_l191_191443


namespace eval_sum_and_subtract_l191_191027

theorem eval_sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by {
  -- The rest of the proof should go here, but we'll use sorry to skip it.
  sorry
}

end eval_sum_and_subtract_l191_191027


namespace no_solutions_l191_191534

theorem no_solutions (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a + b ≠ 0) :
  ¬ (1 / a + 2 / b = 3 / (a + b)) :=
by { sorry }

end no_solutions_l191_191534


namespace fisher_catch_l191_191061

theorem fisher_catch (x y : ℕ) (h1 : x + y = 80)
  (h2 : ∃ a : ℕ, x = 9 * a)
  (h3 : ∃ b : ℕ, y = 11 * b) :
  x = 36 ∧ y = 44 :=
by
  sorry

end fisher_catch_l191_191061


namespace flour_needed_for_two_loaves_l191_191209

-- Define the amount of flour needed for one loaf.
def flour_per_loaf : ℝ := 2.5

-- Define the number of loaves.
def number_of_loaves : ℕ := 2

-- Define the total amount of flour needed for the given number of loaves.
def total_flour_needed : ℝ := flour_per_loaf * number_of_loaves

-- The theorem statement: Prove that the total amount of flour needed is 5 cups.
theorem flour_needed_for_two_loaves : total_flour_needed = 5 := by
  sorry

end flour_needed_for_two_loaves_l191_191209


namespace cos_210_eq_neg_sqrt3_over_2_l191_191120

theorem cos_210_eq_neg_sqrt3_over_2 :
  Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end cos_210_eq_neg_sqrt3_over_2_l191_191120


namespace triangle_inradius_l191_191485

theorem triangle_inradius (p A : ℝ) (h_p : p = 20) (h_A : A = 30) : 
  ∃ r : ℝ, r = 3 ∧ A = r * p / 2 :=
by
  sorry

end triangle_inradius_l191_191485


namespace bananas_left_correct_l191_191194

def initial_bananas : ℕ := 12
def eaten_bananas : ℕ := 1
def bananas_left (initial eaten : ℕ) := initial - eaten

theorem bananas_left_correct : bananas_left initial_bananas eaten_bananas = 11 :=
by
  sorry

end bananas_left_correct_l191_191194


namespace proof_not_necessarily_15_points_l191_191893

-- Define the number of teams
def teams := 14

-- Define a tournament where each team plays every other exactly once
def games := (teams * (teams - 1)) / 2

-- Define a function calculating the total points by summing points for each game
def total_points (wins draws : ℕ) := (3 * wins) + (1 * draws)

-- Define a statement that total points is at least 150
def scores_sum_at_least_150 (wins draws : ℕ) : Prop :=
  total_points wins draws ≥ 150

-- Define a condition that a score could be less than 15
def highest_score_not_necessarily_15 : Prop :=
  ∃ (scores : Finset ℕ), scores.card = teams ∧ ∀ score ∈ scores, score < 15

theorem proof_not_necessarily_15_points :
  ∃ (wins draws : ℕ), wins + draws = games ∧ scores_sum_at_least_150 wins draws ∧ highest_score_not_necessarily_15 :=
by
  sorry

end proof_not_necessarily_15_points_l191_191893


namespace distinct_license_plates_l191_191241

noncomputable def license_plates : ℕ :=
  let digits_possibilities := 10^5
  let letters_possibilities := 26^3
  let positions := 6
  positions * digits_possibilities * letters_possibilities

theorem distinct_license_plates : 
  license_plates = 105456000 := by
  sorry

end distinct_license_plates_l191_191241


namespace volume_ratio_remainder_520_l191_191613

noncomputable def simplex_ratio_mod : Nat :=
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000

theorem volume_ratio_remainder_520 :
  let m := 2 ^ 2015 - 2016
  let n := 2 ^ 2015
  (m + n) % 1000 = 520 :=
by 
  sorry

end volume_ratio_remainder_520_l191_191613


namespace no_nontrivial_integer_solutions_l191_191685

theorem no_nontrivial_integer_solutions (a b c d : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
by
  intro h
  sorry

end no_nontrivial_integer_solutions_l191_191685


namespace area_of_original_triangle_l191_191657

variable (H : ℝ) (H' : ℝ := 0.65 * H) 
variable (A' : ℝ := 14.365)
variable (k : ℝ := 0.65) 
variable (A : ℝ)

theorem area_of_original_triangle (h₁ : H' = k * H) (h₂ : A' = 14.365) (h₃ : k = 0.65) : A = 34 := by
  sorry

end area_of_original_triangle_l191_191657


namespace problem_statement_l191_191454

-- Define A as the number of four-digit odd numbers
def A : ℕ := 4500

-- Define B as the number of four-digit multiples of 3
def B : ℕ := 3000

-- The main theorem stating the sum A + B equals 7500
theorem problem_statement : A + B = 7500 := by
  -- The exact proof is omitted using sorry
  sorry

end problem_statement_l191_191454


namespace pentagon_area_l191_191349

noncomputable def angle_F := 100
noncomputable def angle_G := 100
noncomputable def JF := 3
noncomputable def FG := 3
noncomputable def GH := 3
noncomputable def HI := 5
noncomputable def IJ := 5
noncomputable def area_FGHIJ := 9 * Real.sqrt 3 + Real.sqrt 17.1875

theorem pentagon_area : area_FGHIJ = 9 * Real.sqrt 3 + Real.sqrt 17.1875 :=
by
  sorry

end pentagon_area_l191_191349


namespace factor_expression_l191_191517

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) :=
by
sorry

end factor_expression_l191_191517


namespace line_through_center_eq_line_bisects_chord_eq_l191_191777

section Geometry

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define the point P
def P := (2, 2)

-- Define when line l passes through the center of the circle
def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define when line l bisects chord AB by point P
def line_bisects_chord (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- Prove the equation of line l passing through the center
theorem line_through_center_eq : 
  (∀ (x y : ℝ), line_through_center x y → circleC x y → (x, y) = (1, 0)) →
  2 * (2:ℝ) - 2 - 2 = 0 := sorry

-- Prove the equation of line l bisects chord AB by point P
theorem line_bisects_chord_eq:
  (∀ (x y : ℝ), line_bisects_chord x y → circleC x y → (2, 2) = P) →
  (2 + 2 * 2 - 6 = 0) := sorry

end Geometry

end line_through_center_eq_line_bisects_chord_eq_l191_191777


namespace total_dollars_l191_191883

theorem total_dollars (john emma lucas : ℝ) 
  (h_john : john = 4 / 5) 
  (h_emma : emma = 2 / 5) 
  (h_lucas : lucas = 1 / 2) : 
  john + emma + lucas = 1.7 := by
  sorry

end total_dollars_l191_191883


namespace price_of_large_pizza_l191_191698

variable {price_small_pizza : ℕ}
variable {total_revenue : ℕ}
variable {small_pizzas_sold : ℕ}
variable {large_pizzas_sold : ℕ}
variable {price_large_pizza : ℕ}

theorem price_of_large_pizza
  (h1 : price_small_pizza = 2)
  (h2 : total_revenue = 40)
  (h3 : small_pizzas_sold = 8)
  (h4 : large_pizzas_sold = 3) :
  price_large_pizza = 8 :=
by
  sorry

end price_of_large_pizza_l191_191698


namespace smaller_third_angle_l191_191044

theorem smaller_third_angle (x y : ℕ) (h₁ : x = 64) 
  (h₂ : 2 * x + (x - y) = 180) : y = 12 :=
by
  sorry

end smaller_third_angle_l191_191044


namespace geometric_sequence_b_eq_neg3_l191_191170

theorem geometric_sequence_b_eq_neg3 (a b c : ℝ) : 
  (∃ r : ℝ, -1 = r * a ∧ a = r * b ∧ b = r * c ∧ c = r * (-9)) → b = -3 :=
by
  intro h
  obtain ⟨r, h1, h2, h3, h4⟩ := h
  -- Proof to be filled in later.
  sorry

end geometric_sequence_b_eq_neg3_l191_191170


namespace simplify_fraction_l191_191198

variable (x y : ℝ)

theorem simplify_fraction :
  (2 * x + y) / 4 + (5 * y - 4 * x) / 6 - y / 12 = (-x + 6 * y) / 6 :=
by
  sorry

end simplify_fraction_l191_191198


namespace option_c_opposites_l191_191567

theorem option_c_opposites : -|3| = -3 ∧ 3 = 3 → ( ∃ x y : ℝ, x = -3 ∧ y = 3 ∧ x = -y) :=
by
  sorry

end option_c_opposites_l191_191567


namespace james_total_chore_time_l191_191357

theorem james_total_chore_time
  (V C L : ℝ)
  (hV : V = 3)
  (hC : C = 3 * V)
  (hL : L = C / 2) :
  V + C + L = 16.5 := by
  sorry

end james_total_chore_time_l191_191357


namespace right_triangle_area_l191_191330

theorem right_triangle_area (a b c p : ℝ) (h1 : a = b) (h2 : 3 * p = a + b + c)
  (h3 : c = Real.sqrt (2 * a ^ 2)) :
  (1/2) * a ^ 2 = (9 * p ^ 2 * (3 - 2 * Real.sqrt 2)) / 4 :=
by
  sorry

end right_triangle_area_l191_191330


namespace polynomial_simplified_l191_191387

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end polynomial_simplified_l191_191387


namespace certain_number_is_l191_191301

theorem certain_number_is (x : ℝ) : 
  x * (-4.5) = 2 * (-4.5) - 36 → x = 10 :=
by
  intro h
  -- proof goes here
  sorry

end certain_number_is_l191_191301


namespace prime_condition_l191_191237

theorem prime_condition (p : ℕ) [Fact (Nat.Prime p)] :
  (∀ (a : ℕ), (1 < a ∧ a < p / 2) → (∃ (b : ℕ), (p / 2 < b ∧ b < p) ∧ p ∣ (a * b - 1))) ↔ (p = 5 ∨ p = 7 ∨ p = 13) := by
  sorry

end prime_condition_l191_191237


namespace union_M_N_l191_191435

-- Definitions for the sets M and N
def M : Set ℝ := { x | x^2 = x }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≤ 0 }

-- Proof problem statement
theorem union_M_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_M_N_l191_191435


namespace nh4cl_formed_l191_191221

theorem nh4cl_formed :
  (∀ (nh3 hcl nh4cl : ℝ), nh3 = 1 ∧ hcl = 1 → nh3 + hcl = nh4cl → nh4cl = 1) :=
by
  intros nh3 hcl nh4cl
  sorry

end nh4cl_formed_l191_191221


namespace sum_of_cubes_eq_91_l191_191646

theorem sum_of_cubes_eq_91 (a b : ℤ) (h₁ : a^3 + b^3 = 91) (h₂ : a * b = 12) : a^3 + b^3 = 91 :=
by
  exact h₁

end sum_of_cubes_eq_91_l191_191646


namespace solve_trig_inequality_l191_191548

noncomputable def sin_triple_angle_identity (x : ℝ) : ℝ :=
  3 * (Real.sin x) - 4 * (Real.sin x) ^ 3

theorem solve_trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
  (8 / (3 * Real.sin x - sin_triple_angle_identity x) + 3 * (Real.sin x) ^ 2) ≤ 5 ↔
  x = Real.pi / 2 :=
by
  sorry

end solve_trig_inequality_l191_191548


namespace solve_Cheolsu_weight_l191_191838

def Cheolsu_weight (C M F : ℝ) :=
  (C + M + F) / 3 = M ∧
  C = (2 / 3) * M ∧
  F = 72

theorem solve_Cheolsu_weight {C M F : ℝ} (h : Cheolsu_weight C M F) : C = 36 :=
by
  sorry

end solve_Cheolsu_weight_l191_191838


namespace find_B_coords_l191_191128

-- Define point A and vector a
def A : (ℝ × ℝ) := (1, -3)
def a : (ℝ × ℝ) := (3, 4)

-- Assume B is at coordinates (m, n) and AB = 2a
def B : (ℝ × ℝ) := (7, 5)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Prove point B has the correct coordinates
theorem find_B_coords : AB = (2 * a.1, 2 * a.2) → B = (7, 5) :=
by
  intro h
  sorry

end find_B_coords_l191_191128


namespace sin_minus_cos_eq_minus_1_l191_191930

theorem sin_minus_cos_eq_minus_1 (x : ℝ) 
  (h : Real.sin x ^ 3 - Real.cos x ^ 3 = -1) :
  Real.sin x - Real.cos x = -1 := by
  sorry

end sin_minus_cos_eq_minus_1_l191_191930


namespace determine_z_l191_191290

theorem determine_z (z : ℝ) (h1 : ∃ x : ℤ, 3 * (x : ℝ) ^ 2 + 19 * (x : ℝ) - 84 = 0 ∧ (x : ℝ) = ⌊z⌋) (h2 : 4 * (z - ⌊z⌋) ^ 2 - 14 * (z - ⌊z⌋) + 6 = 0) : 
  z = -11 :=
  sorry

end determine_z_l191_191290


namespace vegan_menu_fraction_suitable_l191_191222

theorem vegan_menu_fraction_suitable (vegan_dishes total_dishes vegan_dishes_with_gluten_or_dairy : ℕ)
  (h1 : vegan_dishes = 9)
  (h2 : vegan_dishes = 3 * total_dishes / 10)
  (h3 : vegan_dishes_with_gluten_or_dairy = 7) :
  (vegan_dishes - vegan_dishes_with_gluten_or_dairy) / total_dishes = 1 / 15 := by
  sorry

end vegan_menu_fraction_suitable_l191_191222


namespace log_sum_exp_log_sub_l191_191980

theorem log_sum : Real.log 2 / Real.log 10 + Real.log 5 / Real.log 10 = 1 := 
by sorry

theorem exp_log_sub : Real.exp (Real.log 3 / Real.log 2 * Real.log 2) - Real.exp (Real.log 8 / 3) = 1 := 
by sorry

end log_sum_exp_log_sub_l191_191980


namespace total_carriages_proof_l191_191183

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l191_191183


namespace length_of_bridge_l191_191210

-- Define the problem conditions
def length_train : ℝ := 110 -- Length of the train in meters
def speed_kmph : ℝ := 60 -- Speed of the train in kmph

-- Convert speed from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Define the time taken to cross the bridge
def time_seconds : ℝ := 16.7986561075114

-- Define the total distance covered
noncomputable def total_distance : ℝ := speed_mps * time_seconds

-- Prove the length of the bridge
theorem length_of_bridge : total_distance - length_train = 170 := 
by
  -- Proof will be here
  sorry

end length_of_bridge_l191_191210


namespace compare_powers_l191_191308

theorem compare_powers:
  (2 ^ 2023) * (7 ^ 2023) < (3 ^ 2023) * (5 ^ 2023) :=
  sorry

end compare_powers_l191_191308


namespace coefficients_sum_l191_191080

theorem coefficients_sum :
  let A := 3
  let B := 14
  let C := 18
  let D := 19
  let E := 30
  A + B + C + D + E = 84 := by
  sorry

end coefficients_sum_l191_191080


namespace baker_bakes_25_hours_per_week_mon_to_fri_l191_191830

-- Define the conditions
def loaves_per_hour_per_oven := 5
def number_of_ovens := 4
def weekend_baking_hours_per_day := 2
def total_weeks := 3
def total_loaves := 1740

-- Calculate the loaves per hour
def loaves_per_hour := loaves_per_hour_per_oven * number_of_ovens

-- Calculate the weekend baking hours in one week
def weekend_baking_hours_per_week := weekend_baking_hours_per_day * 2

-- Calculate the loaves baked on weekends in one week
def loaves_on_weekends_per_week := loaves_per_hour * weekend_baking_hours_per_week

-- Calculate the total loaves baked on weekends in 3 weeks
def loaves_on_weekends_total := loaves_on_weekends_per_week * total_weeks

-- Calculate the loaves baked from Monday to Friday in 3 weeks
def loaves_on_weekdays_total := total_loaves - loaves_on_weekends_total

-- Calculate the total hours baked from Monday to Friday in 3 weeks
def weekday_baking_hours_total := loaves_on_weekdays_total / loaves_per_hour

-- Calculate the number of hours baked from Monday to Friday in one week
def weekday_baking_hours_per_week := weekday_baking_hours_total / total_weeks

-- Proof statement
theorem baker_bakes_25_hours_per_week_mon_to_fri :
  weekday_baking_hours_per_week = 25 :=
by
  sorry

end baker_bakes_25_hours_per_week_mon_to_fri_l191_191830


namespace girls_attending_sports_event_l191_191553

theorem girls_attending_sports_event 
  (total_students attending_sports_event : ℕ) 
  (girls boys : ℕ)
  (h1 : total_students = 1500)
  (h2 : attending_sports_event = 900)
  (h3 : girls + boys = total_students)
  (h4 : (1 / 2) * girls + (3 / 5) * boys = attending_sports_event) :
  (1 / 2) * girls = 500 := 
by
  sorry

end girls_attending_sports_event_l191_191553


namespace find_multiple_l191_191768

-- Given conditions as definitions
def smaller_number := 21
def sum_of_numbers := 84

-- Definition of larger number being a multiple of the smaller number
def is_multiple (k : ℤ) (a b : ℤ) : Prop := b = k * a

-- Given that one number is a multiple of the other and their sum
def problem (L S : ℤ) (k : ℤ) : Prop := 
  is_multiple k S L ∧ S + L = sum_of_numbers

theorem find_multiple (L S : ℤ) (k : ℤ) (h1 : problem L S k) : k = 3 := by
  -- Proof omitted
  sorry

end find_multiple_l191_191768


namespace gcd_lcm_ratio_l191_191410

theorem gcd_lcm_ratio (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 200) (h2 : 2 * k = A) (h3 : 5 * k = B) : Nat.gcd A B = k :=
by
  sorry

end gcd_lcm_ratio_l191_191410


namespace sin_sum_diff_l191_191667

theorem sin_sum_diff (α β : ℝ) 
  (hα : Real.sin α = 1/3) 
  (hβ : Real.sin β = 1/2) : 
  Real.sin (α + β) * Real.sin (α - β) = -5/36 := 
sorry

end sin_sum_diff_l191_191667


namespace num_primes_with_squares_in_range_l191_191011

/-- There are exactly 6 prime numbers whose squares are between 2500 and 5500. -/
theorem num_primes_with_squares_in_range : 
  ∃ primes : Finset ℕ, 
    (∀ p ∈ primes, Prime p) ∧
    (∀ p ∈ primes, 2500 < p^2 ∧ p^2 < 5500) ∧
    primes.card = 6 :=
by
  sorry

end num_primes_with_squares_in_range_l191_191011


namespace solve_trig_eq_l191_191911

noncomputable def rad (d : ℝ) := d * (Real.pi / 180)

theorem solve_trig_eq (z : ℝ) (k : ℤ) :
  (7 * Real.cos (z) ^ 3 - 6 * Real.cos (z) = 3 * Real.cos (3 * z)) ↔
  (z = rad 90 + k * rad 180 ∨
   z = rad 39.2333 + k * rad 180 ∨
   z = rad 140.7667 + k * rad 180) :=
sorry

end solve_trig_eq_l191_191911


namespace angle_not_45_or_135_l191_191261

variable {a b S : ℝ}
variable {C : ℝ} (h : S = (1/2) * a * b * Real.cos C)

theorem angle_not_45_or_135 (h : S = (1/2) * a * b * Real.cos C) : ¬ (C = 45 ∨ C = 135) :=
sorry

end angle_not_45_or_135_l191_191261


namespace number_of_roots_l191_191248

-- Definitions for the conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_in_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → y ≤ a → f x ≤ f y

-- Main theorem to prove
theorem number_of_roots (f : ℝ → ℝ) (a : ℝ) (h1 : 0 < a) 
  (h2 : is_even_function f) (h3 : is_monotonic_in_interval f a) 
  (h4 : f 0 * f a < 0) : ∃ x0 > 0, f x0 = 0 ∧ ∃ x1 < 0, f x1 = 0 :=
sorry

end number_of_roots_l191_191248


namespace max_sequence_length_l191_191679

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ)
  (H1 : ∀ k : ℕ, k + 4 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4)) < 0)
  (H2 : ∀ k : ℕ, k + 8 < n → (a k + a (k+1) + a (k+2) + a (k+3) + a (k+4) + a (k+5) + a (k+6) + a (k+7) + a (k+8)) > 0) : 
  n ≤ 12 :=
sorry

end max_sequence_length_l191_191679


namespace find_d_l191_191223

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3
def h (x : ℝ) (c : ℝ) (d : ℝ) : Prop := f (g x c) c = 15 * x + d

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l191_191223


namespace probability_sunglasses_to_hat_l191_191406

variable (S H : Finset ℕ) -- S: set of people wearing sunglasses, H: set of people wearing hats
variable (num_S : Nat) (num_H : Nat) (num_SH : Nat)
variable (prob_hat_to_sunglasses : ℚ)

-- Conditions
def condition1 : num_S = 80 := sorry
def condition2 : num_H = 50 := sorry
def condition3 : prob_hat_to_sunglasses = 3 / 5 := sorry
def condition4 : num_SH = (3/5) * 50 := sorry

-- Question: Prove that the probability a person wearing sunglasses is also wearing a hat
theorem probability_sunglasses_to_hat :
  (num_SH : ℚ) / num_S = 3 / 8 :=
sorry

end probability_sunglasses_to_hat_l191_191406


namespace number_of_men_in_second_group_l191_191773

variable (n m : ℕ)

theorem number_of_men_in_second_group 
  (h1 : 42 * 18 = n)
  (h2 : n = m * 28) : 
  m = 27 := by
  sorry

end number_of_men_in_second_group_l191_191773


namespace vehicles_with_cd_player_but_no_pw_or_ab_l191_191558

-- Definitions based on conditions from step a)
def P : ℝ := 0.60 -- percentage of vehicles with power windows
def A : ℝ := 0.25 -- percentage of vehicles with anti-lock brakes
def C : ℝ := 0.75 -- percentage of vehicles with a CD player
def PA : ℝ := 0.10 -- percentage of vehicles with both power windows and anti-lock brakes
def AC : ℝ := 0.15 -- percentage of vehicles with both anti-lock brakes and a CD player
def PC : ℝ := 0.22 -- percentage of vehicles with both power windows and a CD player
def PAC : ℝ := 0.00 -- no vehicle has all 3 features

-- The statement we want to prove
theorem vehicles_with_cd_player_but_no_pw_or_ab : C - (PC + AC) = 0.38 := by
  sorry

end vehicles_with_cd_player_but_no_pw_or_ab_l191_191558


namespace inequality_solution_l191_191057

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ (-1 < x)) :=
by sorry

end inequality_solution_l191_191057


namespace find_interest_rate_l191_191960

noncomputable def compoundInterestRate (P A : ℝ) (t : ℕ) : ℝ := 
  ((A / P) ^ (1 / t)) - 1

theorem find_interest_rate :
  ∀ (P A : ℝ) (t : ℕ),
    P = 1200 → 
    A = 1200 + 873.60 →
    t = 3 →
    compoundInterestRate P A t = 0.2 :=
by
  intros P A t hP hA ht
  sorry

end find_interest_rate_l191_191960


namespace red_pairs_l191_191811

theorem red_pairs (total_students green_students red_students total_pairs green_pairs : ℕ) 
  (h1 : total_students = green_students + red_students)
  (h2 : green_students = 67)
  (h3 : red_students = 89)
  (h4 : total_pairs = 78)
  (h5 : green_pairs = 25)
  (h6 : 2 * green_pairs ≤ green_students ∧ 2 * green_pairs ≤ red_students ∧ 2 * green_pairs ≤ 2 * total_pairs) :
  ∃ red_pairs : ℕ, red_pairs = 36 := by
    sorry

end red_pairs_l191_191811


namespace climb_stairs_l191_191794

noncomputable def u (n : ℕ) : ℝ :=
  let Φ := (1 + Real.sqrt 5) / 2
  let φ := (1 - Real.sqrt 5) / 2
  let A := (1 + Real.sqrt 5) / (2 * Real.sqrt 5)
  let B := (Real.sqrt 5 - 1) / (2 * Real.sqrt 5)
  A * (Φ ^ n) + B * (φ ^ n)

theorem climb_stairs (n : ℕ) (hn : n ≥ 1) : u n = A * (Φ ^ n) + B * (φ ^ n) := sorry

end climb_stairs_l191_191794


namespace alice_sold_20_pears_l191_191942

-- Definitions (Conditions)
def canned_more_than_poached (C P : ℝ) : Prop := C = P + 0.2 * P
def poached_less_than_sold (P S : ℝ) : Prop := P = 0.5 * S
def total_pears (S C P : ℝ) : Prop := S + C + P = 42

-- Theorem statement
theorem alice_sold_20_pears (S C P : ℝ) (h1 : canned_more_than_poached C P) (h2 : poached_less_than_sold P S) (h3 : total_pears S C P) : S = 20 :=
by 
  -- This is where the proof would go, but for now, we use sorry to signify it's omitted.
  sorry

end alice_sold_20_pears_l191_191942


namespace prove_problem_statement_l191_191610

noncomputable def problem_statement : Prop :=
  let E := (0, 0)
  let F := (2, 4)
  let G := (6, 2)
  let H := (7, 0)
  let line_through_E x y := y = -2 * x + 14
  let intersection_x := 37 / 8
  let intersection_y := 19 / 4
  let intersection_point := (intersection_x, intersection_y)
  let u := 37
  let v := 8
  let w := 19
  let z := 4
  u + v + w + z = 68

theorem prove_problem_statement : problem_statement :=
  sorry

end prove_problem_statement_l191_191610


namespace least_power_divisible_by_240_l191_191365

theorem least_power_divisible_by_240 (n : ℕ) (a : ℕ) (h_a : a = 60) (h : a^n % 240 = 0) : 
  n = 2 :=
by
  sorry

end least_power_divisible_by_240_l191_191365


namespace sheets_bought_l191_191976

variable (x y : ℕ)

-- Conditions based on the problem statement
def A_condition (x y : ℕ) : Prop := x + 40 = y
def B_condition (x y : ℕ) : Prop := 3 * x + 40 = y

-- Proven that if these conditions are met, then the number of sheets of stationery bought by A and B is 120
theorem sheets_bought (x y : ℕ) (hA : A_condition x y) (hB : B_condition x y) : y = 120 :=
by
  sorry

end sheets_bought_l191_191976


namespace arccos_sin_eq_pi_div_two_sub_1_72_l191_191831

theorem arccos_sin_eq_pi_div_two_sub_1_72 :
  Real.arccos (Real.sin 8) = Real.pi / 2 - 1.72 :=
sorry

end arccos_sin_eq_pi_div_two_sub_1_72_l191_191831


namespace negative_number_unique_l191_191190

theorem negative_number_unique (a b c d : ℚ) (h₁ : a = 1) (h₂ : b = 0) (h₃ : c = 1/2) (h₄ : d = -2) :
  ∃! x : ℚ, x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by 
  sorry

end negative_number_unique_l191_191190


namespace triangle_is_right_angle_l191_191609

theorem triangle_is_right_angle (A B C : ℝ) : 
  (A / B = 2 / 3) ∧ (A / C = 2 / 5) ∧ (A + B + C = 180) →
  (A = 36) ∧ (B = 54) ∧ (C = 90) :=
by 
  intro h
  sorry

end triangle_is_right_angle_l191_191609


namespace veranda_area_l191_191602

/-- The width of the veranda on all sides of the room. -/
def width_of_veranda : ℝ := 2

/-- The length of the room. -/
def length_of_room : ℝ := 21

/-- The width of the room. -/
def width_of_room : ℝ := 12

/-- The area of the veranda given the conditions. -/
theorem veranda_area (length_of_room width_of_room width_of_veranda : ℝ) :
  (length_of_room + 2 * width_of_veranda) * (width_of_room + 2 * width_of_veranda) - length_of_room * width_of_room = 148 :=
by
  sorry

end veranda_area_l191_191602


namespace cos_sin_eq_l191_191299

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l191_191299


namespace thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l191_191897

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_number : triangular_number 30 = 465 := 
by
  sorry

theorem sum_of_thirtieth_and_twentyninth_triangular_numbers : triangular_number 30 + triangular_number 29 = 900 := 
by
  sorry

end thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l191_191897


namespace max_value_ln_x_minus_x_on_interval_l191_191273

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_x_minus_x_on_interval : 
  ∃ x ∈ Set.Ioc 0 (Real.exp 1), ∀ y ∈ Set.Ioc 0 (Real.exp 1), f y ≤ f x ∧ f x = -1 :=
by
  sorry

end max_value_ln_x_minus_x_on_interval_l191_191273


namespace k_l_m_n_values_l191_191973

theorem k_l_m_n_values (k l m n : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 0 < n)
  (hklmn : k + l + m + n = k * m) (hln : k + l + m + n = l * n) :
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end k_l_m_n_values_l191_191973


namespace farmer_shipped_30_boxes_this_week_l191_191310

-- Defining the given conditions
def last_week_boxes : ℕ := 10
def last_week_pomelos : ℕ := 240
def this_week_dozen : ℕ := 60
def pomelos_per_dozen : ℕ := 12

-- Translating conditions into mathematical statements
def pomelos_per_box_last_week : ℕ := last_week_pomelos / last_week_boxes
def this_week_pomelos_total : ℕ := this_week_dozen * pomelos_per_dozen
def boxes_shipped_this_week : ℕ := this_week_pomelos_total / pomelos_per_box_last_week

-- The theorem we prove, that given the conditions, the number of boxes shipped this week is 30.
theorem farmer_shipped_30_boxes_this_week :
  boxes_shipped_this_week = 30 :=
sorry

end farmer_shipped_30_boxes_this_week_l191_191310


namespace cannot_be_value_of_omega_l191_191136

theorem cannot_be_value_of_omega (ω : ℤ) (φ : ℝ) (k n : ℤ) 
  (h1 : 0 < ω) 
  (h2 : |φ| < π / 2)
  (h3 : ω * (π / 12) + φ = k * π + π / 2)
  (h4 : -ω * (π / 6) + φ = n * π) : 
  ∀ m : ℤ, ω ≠ 4 * m := 
sorry

end cannot_be_value_of_omega_l191_191136


namespace f_greater_than_fp_3_2_l191_191228

noncomputable def f (x : ℝ) (a : ℝ) := a * (x - Real.log x) + (2 * x - 1) / (x ^ 2)
noncomputable def f' (x : ℝ) (a : ℝ) := (a * x^3 - a * x^2 + 2 - 2*x) / x^3

theorem f_greater_than_fp_3_2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  f x 1 > f' x 1 + 3 / 2 := sorry

end f_greater_than_fp_3_2_l191_191228


namespace sacred_k_words_n10_k4_l191_191214

/- Definitions for the problem -/
def sacred_k_words_count (n k : ℕ) (hk : k < n / 2) : ℕ :=
  n * Nat.choose (n - k - 1) (k - 1) * (Nat.factorial k / k)

theorem sacred_k_words_n10_k4 : sacred_k_words_count 10 4 (by norm_num : 4 < 10 / 2) = 600 := by
  sorry

end sacred_k_words_n10_k4_l191_191214


namespace simplify_expression_l191_191809

theorem simplify_expression (a : ℤ) : 7 * a - 3 * a = 4 * a :=
by
  sorry

end simplify_expression_l191_191809


namespace additional_cost_per_kg_l191_191586

theorem additional_cost_per_kg (l m : ℝ) 
  (h1 : 168 = 30 * l + 3 * m) 
  (h2 : 186 = 30 * l + 6 * m) 
  (h3 : 20 * l = 100) : 
  m = 6 := 
by
  sorry

end additional_cost_per_kg_l191_191586


namespace sum_of_arithmetic_progression_l191_191837

theorem sum_of_arithmetic_progression 
  (a d : ℚ) 
  (S : ℕ → ℚ)
  (h_sum_15 : S 15 = 150)
  (h_sum_75 : S 75 = 30)
  (h_arith_sum : ∀ n, S n = (n / 2) * (2 * a + (n - 1) * d)) :
  S 90 = -180 :=
by
  sorry

end sum_of_arithmetic_progression_l191_191837


namespace weight_of_b_l191_191165

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 126) (h2 : a + b = 80) (h3 : b + c = 86) : b = 40 :=
sorry

end weight_of_b_l191_191165


namespace smallest_third_term_geometric_l191_191159

theorem smallest_third_term_geometric (d : ℝ) : 
  (∃ d, (7 + d) ^ 2 = 4 * (26 + 2 * d)) → ∃ g3, (g3 = 10 ∨ g3 = 36) ∧ g3 = min (10) (36) :=
by
  sorry

end smallest_third_term_geometric_l191_191159


namespace lines_proportional_l191_191851

variables {x y : ℝ} {p q : ℝ}

theorem lines_proportional (h1 : p * x + 2 * y = 7) (h2 : 3 * x + q * y = 5) :
  p = 21 / 5 := 
sorry

end lines_proportional_l191_191851


namespace identified_rectangle_perimeter_l191_191074

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l191_191074


namespace smaller_of_two_digit_product_l191_191180

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l191_191180


namespace hip_hop_final_percentage_is_39_l191_191573

noncomputable def hip_hop_percentage (total_songs percentage_country: ℝ):
  ℝ :=
  let percentage_non_country := 1 - percentage_country
  let original_ratio_hip_hop := 0.65
  let original_ratio_pop := 0.35
  let total_non_country := original_ratio_hip_hop + original_ratio_pop
  let hip_hop_percentage := original_ratio_hip_hop / total_non_country * percentage_non_country
  hip_hop_percentage

theorem hip_hop_final_percentage_is_39 (total_songs : ℕ) :
  hip_hop_percentage total_songs 0.40 = 0.39 :=
by
  sorry

end hip_hop_final_percentage_is_39_l191_191573


namespace janet_total_miles_run_l191_191252

/-- Janet was practicing for a marathon. She practiced for 9 days, running 8 miles each day.
Prove that Janet ran 72 miles in total. -/
theorem janet_total_miles_run (days_practiced : ℕ) (miles_per_day : ℕ) (total_miles : ℕ) 
  (h1 : days_practiced = 9) (h2 : miles_per_day = 8) : total_miles = 72 := by
  sorry

end janet_total_miles_run_l191_191252


namespace regular_hexagon_perimeter_is_30_l191_191825

-- Define a regular hexagon with each side length 5 cm
def regular_hexagon_side_length : ℝ := 5

-- Define the perimeter of a regular hexagon
def regular_hexagon_perimeter (side_length : ℝ) : ℝ := 6 * side_length

-- State the theorem about the perimeter of a regular hexagon with side length 5 cm
theorem regular_hexagon_perimeter_is_30 : regular_hexagon_perimeter regular_hexagon_side_length = 30 := 
by 
  sorry

end regular_hexagon_perimeter_is_30_l191_191825


namespace tom_gaming_system_value_l191_191255

theorem tom_gaming_system_value
    (V : ℝ) 
    (h1 : 0.80 * V + 80 - 10 = 160 + 30) 
    : V = 150 :=
by
  -- Logical steps for the proof will be added here.
  sorry

end tom_gaming_system_value_l191_191255


namespace arithmetic_sequence_a5_l191_191270

theorem arithmetic_sequence_a5 {a : ℕ → ℕ} 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 2 + a 8 = 12) : 
  a 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l191_191270


namespace find_ABC_l191_191431

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := 
  x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  (∀ x : ℝ, x > 5 → g x 2 (-2) (-24) > 0.5) ∧
  (A = 2) ∧
  (B = -2) ∧
  (C = -24) ∧
  (∀ x, A * x^2 + B * x + C = A * (x + 3) * (x - 4)) → 
  A + B + C = -24 := 
by
  sorry

end find_ABC_l191_191431


namespace sum_of_interior_angles_l191_191587

theorem sum_of_interior_angles (n : ℕ) (h₁ : 180 * (n - 2) = 2340) : 
  180 * ((n - 3) - 2) = 1800 := by
  -- Here, we'll solve the theorem using Lean's capabilities.
  sorry

end sum_of_interior_angles_l191_191587


namespace part_one_part_two_l191_191744

noncomputable def f (x : ℝ) : ℝ := (3 * x) / (x + 1)

-- First part: Prove that f(x) is increasing on [2, 5]
theorem part_one (x₁ x₂ : ℝ) (hx₁ : 2 ≤ x₁) (hx₂ : x₂ ≤ 5) (h : x₁ < x₂) : f x₁ < f x₂ :=
by {
  -- Proof is to be filled in
  sorry
}

-- Second part: Find maximum and minimum of f(x) on [2, 5]
theorem part_two :
  f 2 = 2 ∧ f 5 = 5 / 2 :=
by {
  -- Proof is to be filled in
  sorry
}

end part_one_part_two_l191_191744


namespace exam_student_count_l191_191362

theorem exam_student_count (N T T_5 T_remaining : ℕ)
  (h1 : T = 70 * N)
  (h2 : T_5 = 50 * 5)
  (h3 : T_remaining = 90 * (N - 5))
  (h4 : T = T_5 + T_remaining) :
  N = 10 :=
by
  sorry

end exam_student_count_l191_191362


namespace total_points_scored_l191_191888

def num_members : ℕ := 12
def num_absent : ℕ := 4
def points_per_member : ℕ := 8

theorem total_points_scored : 
  (num_members - num_absent) * points_per_member = 64 := by
  sorry

end total_points_scored_l191_191888


namespace parabola_intersection_diff_l191_191048

theorem parabola_intersection_diff (a b c d : ℝ) 
  (h₁ : ∀ x y, (3 * x^2 - 2 * x + 1 = y) → (c = x ∨ a = x))
  (h₂ : ∀ x y, (-2 * x^2 + 4 * x + 1 = y) → (c = x ∨ a = x))
  (h₃ : c ≥ a) :
  c - a = 6 / 5 :=
by sorry

end parabola_intersection_diff_l191_191048


namespace max_quarters_l191_191860

theorem max_quarters (total_value : ℝ) (n_quarters n_nickels n_dimes : ℕ) 
  (h1 : n_nickels = n_quarters) 
  (h2 : n_dimes = 2 * n_quarters)
  (h3 : 0.25 * n_quarters + 0.05 * n_nickels + 0.10 * n_dimes = total_value)
  (h4 : total_value = 3.80) : 
  n_quarters = 7 := 
by
  sorry

end max_quarters_l191_191860


namespace dimitri_weekly_calories_l191_191784

-- Define the calories for each type of burger
def calories_burger_a : ℕ := 350
def calories_burger_b : ℕ := 450
def calories_burger_c : ℕ := 550

-- Define the daily consumption of each type of burger
def daily_consumption_a : ℕ := 2
def daily_consumption_b : ℕ := 1
def daily_consumption_c : ℕ := 3

-- Define the duration in days
def duration_in_days : ℕ := 7

-- Define the total number of calories Dimitri consumes in a week
noncomputable def total_weekly_calories : ℕ :=
  (daily_consumption_a * calories_burger_a +
   daily_consumption_b * calories_burger_b +
   daily_consumption_c * calories_burger_c) * duration_in_days

theorem dimitri_weekly_calories : total_weekly_calories = 19600 := 
by 
  sorry

end dimitri_weekly_calories_l191_191784


namespace four_painters_small_room_days_l191_191788

-- Define the constants and conditions
def large_room_days : ℕ := 2
def small_room_factor : ℝ := 0.5
def total_painters : ℕ := 5
def painters_available : ℕ := 4

-- Define the total painter-days needed for the small room
def small_room_painter_days : ℝ := total_painters * (small_room_factor * large_room_days)

-- Define the proof problem statement
theorem four_painters_small_room_days : (small_room_painter_days / painters_available) = 5 / 4 :=
by
  -- Placeholder for the proof: we assume the goal is true for now
  sorry

end four_painters_small_room_days_l191_191788


namespace units_digit_27_3_sub_17_3_l191_191188

theorem units_digit_27_3_sub_17_3 : 
  (27 ^ 3 - 17 ^ 3) % 10 = 0 :=
sorry

end units_digit_27_3_sub_17_3_l191_191188


namespace ken_height_l191_191201

theorem ken_height 
  (height_ivan : ℝ) (height_jackie : ℝ) (height_ken : ℝ)
  (h1 : height_ivan = 175) (h2 : height_jackie = 175)
  (h_avg : (height_ivan + height_jackie + height_ken) / 3 = (height_ivan + height_jackie) / 2 * 1.04) :
  height_ken = 196 := 
sorry

end ken_height_l191_191201


namespace smallest_hot_dog_packages_l191_191588

theorem smallest_hot_dog_packages (d : ℕ) (b : ℕ) (hd : d = 10) (hb : b = 15) :
  ∃ n : ℕ, n * d = m * b ∧ n = 3 :=
by
  sorry

end smallest_hot_dog_packages_l191_191588


namespace tens_digit_of_desired_number_is_one_l191_191813

def productOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a * b

def sumOfDigits (n : Nat) : Nat :=
  match n / 10, n % 10 with
  | a, b => a + b

def isDesiredNumber (N : Nat) : Prop :=
  N < 100 ∧ N ≥ 10 ∧ N = (productOfDigits N)^2 + sumOfDigits N

theorem tens_digit_of_desired_number_is_one (N : Nat) (h : isDesiredNumber N) : N / 10 = 1 :=
  sorry

end tens_digit_of_desired_number_is_one_l191_191813


namespace young_member_age_diff_l191_191717

-- Definitions
def A : ℝ := sorry    -- Average age of committee members 4 years ago
def O : ℝ := sorry    -- Age of the old member
def N : ℝ := sorry    -- Age of the new member

-- Hypotheses
axiom avg_same : ∀ (t : ℝ), t = t
axiom replacement : 10 * A + 4 * 10 - 40 = 10 * A

-- Theorem
theorem young_member_age_diff : O - N = 40 := by
  -- proof goes here
  sorry

end young_member_age_diff_l191_191717


namespace original_average_weight_l191_191923

theorem original_average_weight (W : ℝ) (h : (7 * W + 110 + 60) / 9 = 113) : W = 121 :=
by
  sorry

end original_average_weight_l191_191923


namespace is_possible_to_finish_7th_l191_191085

theorem is_possible_to_finish_7th 
  (num_teams : ℕ)
  (wins_ASTC : ℕ)
  (losses_ASTC : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ) 
  (total_points : ℕ)
  (rank_ASTC : ℕ)
  (points_ASTC : ℕ)
  (points_needed_by_top_6 : ℕ → ℕ)
  (points_8th_and_9th : ℕ) :
  num_teams = 9 ∧ wins_ASTC = 5 ∧ losses_ASTC = 3 ∧ points_per_win = 3 ∧ points_per_draw = 1 ∧ 
  total_points = 108 ∧ rank_ASTC = 7 ∧ points_ASTC = 15 ∧ points_needed_by_top_6 7 = 105 ∧ points_8th_and_9th ≤ 3 →
  ∃ (top_7_points : ℕ), 
  top_7_points = 105 ∧ (top_7_points + points_8th_and_9th) = total_points := 
sorry

end is_possible_to_finish_7th_l191_191085


namespace cube_of_composite_as_diff_of_squares_l191_191871

theorem cube_of_composite_as_diff_of_squares (n : ℕ) (h : ∃ a b, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ (A₁ B₁ A₂ B₂ A₃ B₃ : ℕ), 
    n^3 = A₁^2 - B₁^2 ∧ 
    n^3 = A₂^2 - B₂^2 ∧ 
    n^3 = A₃^2 - B₃^2 ∧ 
    (A₁, B₁) ≠ (A₂, B₂) ∧ 
    (A₁, B₁) ≠ (A₃, B₃) ∧ 
    (A₂, B₂) ≠ (A₃, B₃) := sorry

end cube_of_composite_as_diff_of_squares_l191_191871


namespace incorrect_statement_l191_191369

variable (f : ℝ → ℝ)
variable (k : ℝ)
variable (h₁ : f 0 = -1)
variable (h₂ : ∀ x, f' x > k)
variable (h₃ : k > 1)

theorem incorrect_statement :
  ¬ f (1 / (k - 1)) < 1 / (k - 1) :=
sorry

end incorrect_statement_l191_191369


namespace product_of_consecutive_integers_sqrt_50_l191_191482

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l191_191482


namespace count_multiples_of_13_three_digit_l191_191571

-- Definitions based on the conditions in the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

-- Statement of the proof problem
theorem count_multiples_of_13_three_digit :
  ∃ (count : ℕ), count = (76 - 8 + 1) :=
sorry

end count_multiples_of_13_three_digit_l191_191571


namespace no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l191_191046

-- Problem 1: There is no value of m that makes the lines parallel.
theorem no_solution_for_parallel_lines (m : ℝ) :
  ¬ ∃ m, (2 * m^2 + m - 3) / (m^2 - m) = 1 := sorry

-- Problem 2: The values of a that make the lines perpendicular.
theorem values_of_a_for_perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔ (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) := sorry

end no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l191_191046


namespace number_of_numbers_l191_191376

theorem number_of_numbers (N : ℕ) (h_avg : (18 * N + 40) / N = 22) : N = 10 :=
by
  sorry

end number_of_numbers_l191_191376


namespace calculate_expression_l191_191933

theorem calculate_expression :
  3 ^ 3 * 2 ^ 2 * 7 ^ 2 * 11 = 58212 :=
by
  sorry

end calculate_expression_l191_191933


namespace complement_intersection_l191_191145

open Set

variable (U M N : Set ℕ)
variable (H₁ : U = {1, 2, 3, 4, 5, 6})
variable (H₂ : M = {1, 2, 3, 5})
variable (H₃ : N = {1, 3, 4, 6})

theorem complement_intersection :
  (U \ (M ∩ N)) = {2, 4, 5, 6} :=
by
  sorry

end complement_intersection_l191_191145


namespace sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l191_191025

theorem sequence_a_n_general_formula_and_value (a : ℕ → ℕ) 
  (h1 : a 1 = 3) 
  (h10 : a 10 = 21) 
  (h_linear : ∃ (k b : ℕ), ∀ n, a n = k * n + b) :
  (∀ n, a n = 2 * n + 1) ∧ a 2005 = 4011 :=
by 
  sorry

theorem sequence_b_n_general_formula (a b : ℕ → ℕ)
  (h_seq_a : ∀ n, a n = 2 * n + 1) 
  (h_b_formed : ∀ n, b n = a (2 * n)) : 
  ∀ n, b n = 4 * n + 1 :=
by 
  sorry

end sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l191_191025


namespace calculate_difference_square_l191_191702

theorem calculate_difference_square (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 :=
by sorry

end calculate_difference_square_l191_191702


namespace no_real_solution_f_of_f_f_eq_x_l191_191925

-- Defining the quadratic polynomial f(x) = ax^2 + bx + c
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Stating the main theorem
theorem no_real_solution_f_of_f_f_eq_x (a b c : ℝ) (h : (b - 1)^2 - 4 * a * c < 0) :
  ¬ ∃ x : ℝ, f a b c (f a b c x) = x :=
by 
  -- Proof will go here
  sorry

end no_real_solution_f_of_f_f_eq_x_l191_191925


namespace sum_of_positive_factors_of_72_l191_191434

/-- Define the divisor sum function based on the given formula -/
def divisor_sum (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 3
  | 3 => 4
  | 4 => 7
  | 6 => 12
  | 8 => 15
  | 12 => 28
  | 18 => 39
  | 24 => 60
  | 36 => 91
  | 48 => 124
  | 60 => 168
  | 72 => 195
  | _ => 0 -- This is not generally correct, just handles given problem specifically

theorem sum_of_positive_factors_of_72 :
  divisor_sum 72 = 195 :=
sorry

end sum_of_positive_factors_of_72_l191_191434


namespace smallest_number_l191_191745

-- Definitions of conditions for H, P, and S
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5
def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def satisfies_conditions_H (H : ℕ) : Prop :=
  is_cube (H / 2) ∧ is_fifth_power (H / 3) ∧ is_square (H / 5)

def satisfies_conditions_P (P A B C : ℕ) : Prop :=
  P / 2 = A^2 ∧ P / 3 = B^3 ∧ P / 5 = C^5

def satisfies_conditions_S (S D E F : ℕ) : Prop :=
  S / 2 = D^5 ∧ S / 3 = E^2 ∧ S / 5 = F^3

-- Main statement: Prove that P is the smallest number satisfying the conditions
theorem smallest_number (H P S A B C D E F : ℕ)
  (hH : satisfies_conditions_H H)
  (hP : satisfies_conditions_P P A B C)
  (hS : satisfies_conditions_S S D E F) :
  P ≤ H ∧ P ≤ S :=
  sorry

end smallest_number_l191_191745


namespace price_first_variety_is_126_l191_191653

variable (x : ℝ) -- price of the first variety per kg (unknown we need to solve for)
variable (p2 : ℝ := 135) -- price of the second variety per kg
variable (p3 : ℝ := 175.5) -- price of the third variety per kg
variable (mix_ratio : ℝ := 4) -- total weight ratio of the mixture
variable (mix_price : ℝ := 153) -- price of the mixture per kg
variable (w1 w2 w3 : ℝ := 1) -- weights of the first two varieties
variable (w4 : ℝ := 2) -- weight of the third variety

theorem price_first_variety_is_126:
  (w1 * x + w2 * p2 + w4 * p3) / mix_ratio = mix_price → x = 126 := by
  sorry

end price_first_variety_is_126_l191_191653


namespace sum_of_first_15_terms_l191_191239

theorem sum_of_first_15_terms (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 24) : 
  (15 / 2) * (2 * a + 14 * d) = 180 :=
by
  sorry

end sum_of_first_15_terms_l191_191239


namespace Joan_paid_158_l191_191983

theorem Joan_paid_158 (J K : ℝ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end Joan_paid_158_l191_191983


namespace sequence_eq_l191_191220

-- Define the sequence and the conditions
def is_sequence (a : ℕ → ℕ) :=
  (∀ i, a i > 0) ∧ (∀ i j, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j)

-- The theorem we want to prove: for all i, a_i = i
theorem sequence_eq (a : ℕ → ℕ) (h : is_sequence a) : ∀ i, a i = i :=
by
  sorry

end sequence_eq_l191_191220


namespace colorings_without_two_corners_l191_191146

def valid_colorings (n: ℕ) (exclude_cells : Finset (Fin n × Fin n)) : ℕ := sorry

theorem colorings_without_two_corners :
  valid_colorings 5 ∅ = 120 →
  valid_colorings 5 {(0, 0)} = 96 →
  valid_colorings 5 {(0, 0), (4, 4)} = 78 :=
by {
  sorry
}

end colorings_without_two_corners_l191_191146


namespace number_of_ways_to_assign_shifts_l191_191872

def workers : List String := ["A", "B", "C"]

theorem number_of_ways_to_assign_shifts :
  let shifts := ["day", "night"]
  (workers.length * (workers.length - 1)) = 6 := by
  sorry

end number_of_ways_to_assign_shifts_l191_191872


namespace convert_base_7_to_base_10_l191_191636

theorem convert_base_7_to_base_10 (n : ℕ) (h : n = 6 * 7^2 + 5 * 7^1 + 3 * 7^0) : n = 332 := by
  sorry

end convert_base_7_to_base_10_l191_191636


namespace parabola_equation_l191_191382

theorem parabola_equation (x y : ℝ) (hx : x = -2) (hy : y = 3) :
  (y^2 = -(9 / 2) * x) ∨ (x^2 = (4 / 3) * y) :=
by
  sorry

end parabola_equation_l191_191382


namespace find_value_given_conditions_l191_191452

def equation_result (x y k : ℕ) : Prop := x ^ y + y ^ x = k

theorem find_value_given_conditions (y : ℕ) (k : ℕ) : 
  equation_result 2407 y k := 
by 
  sorry

end find_value_given_conditions_l191_191452


namespace value_of_x_l191_191703

theorem value_of_x (x : ℝ) (h₁ : x > 0) (h₂ : x^3 = 19683) : x = 27 :=
sorry

end value_of_x_l191_191703


namespace original_square_perimeter_l191_191193

theorem original_square_perimeter (P : ℝ) (x : ℝ) (h1 : 4 * x * 2 + 4 * x = 56) : P = 32 :=
by
  sorry

end original_square_perimeter_l191_191193


namespace possible_values_for_abc_l191_191257

theorem possible_values_for_abc (a b c : ℝ)
  (h : ∀ x y z : ℤ, (a * x + b * y + c * z) ∣ (b * x + c * y + a * z)) :
  (a, b, c) = (1, 0, 0) ∨ (a, b, c) = (0, 1, 0) ∨ (a, b, c) = (0, 0, 1) ∨
  (a, b, c) = (-1, 0, 0) ∨ (a, b, c) = (0, -1, 0) ∨ (a, b, c) = (0, 0, -1) :=
sorry

end possible_values_for_abc_l191_191257


namespace triangles_area_possibilities_unique_l191_191580

noncomputable def triangle_area_possibilities : ℕ :=
  -- Define lengths of segments on the first line
  let AB := 1
  let BC := 2
  let CD := 3
  -- Sum to get total lengths
  let AC := AB + BC -- 3
  let AD := AB + BC + CD -- 6
  -- Define length of the segment on the second line
  let EF := 2
  -- GH is a segment not parallel to the first two lines
  let GH := 1
  -- The number of unique possible triangle areas
  4

theorem triangles_area_possibilities_unique :
  triangle_area_possibilities = 4 := 
sorry

end triangles_area_possibilities_unique_l191_191580


namespace functional_relationship_remaining_oil_after_4_hours_l191_191751

-- Define the initial conditions and the functional form
def initial_oil : ℝ := 50
def consumption_rate : ℝ := 8
def remaining_oil (t : ℝ) : ℝ := initial_oil - consumption_rate * t

-- Prove the functional relationship and the remaining oil after 4 hours
theorem functional_relationship : ∀ (t : ℝ), remaining_oil t = 50 - 8 * t :=
by intros t
   exact rfl

theorem remaining_oil_after_4_hours : remaining_oil 4 = 18 :=
by simp [remaining_oil]
   norm_num
   sorry

end functional_relationship_remaining_oil_after_4_hours_l191_191751


namespace valid_k_values_l191_191196

theorem valid_k_values
  (k : ℝ)
  (h : k = -7 ∨ k = -5 ∨ k = 1 ∨ k = 4) :
  (∀ x, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) → (k = -7 ∨ k = 1 ∨ k = 4) :=
by sorry

end valid_k_values_l191_191196


namespace horizontal_distance_l191_191106

def curve (x : ℝ) := x^3 - x^2 - x - 6

def P_condition (x : ℝ) := curve x = 10
def Q_condition1 (x : ℝ) := curve x = 2
def Q_condition2 (x : ℝ) := curve x = -2

theorem horizontal_distance (x_P x_Q: ℝ) (hP: P_condition x_P) (hQ1: Q_condition1 x_Q ∨ Q_condition2 x_Q) :
  |x_P - x_Q| = 3 := sorry

end horizontal_distance_l191_191106


namespace discriminant_of_quadratic_is_321_l191_191723

-- Define the quadratic equation coefficients
def a : ℝ := 4
def b : ℝ := -9
def c : ℝ := -15

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The proof statement
theorem discriminant_of_quadratic_is_321 : discriminant a b c = 321 := by
  sorry

end discriminant_of_quadratic_is_321_l191_191723


namespace customer_paid_amount_l191_191974

theorem customer_paid_amount 
  (cost_price : ℝ) 
  (markup_percent : ℝ) 
  (customer_payment : ℝ)
  (h1 : cost_price = 1250) 
  (h2 : markup_percent = 0.60)
  (h3 : customer_payment = cost_price + (markup_percent * cost_price)) :
  customer_payment = 2000 :=
sorry

end customer_paid_amount_l191_191974


namespace PQ_R_exist_l191_191829

theorem PQ_R_exist :
  ∃ P Q R : ℚ, 
    (P = -3/5) ∧ (Q = -1) ∧ (R = 13/5) ∧
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
    (x^2 - 10)/((x - 1)*(x - 4)*(x - 6)) = P/(x - 1) + Q/(x - 4) + R/(x - 6)) :=
by
  sorry

end PQ_R_exist_l191_191829


namespace construction_cost_is_correct_l191_191952

def land_cost (cost_per_sqm : ℕ) (area : ℕ) : ℕ :=
  cost_per_sqm * area

def bricks_cost (cost_per_1000 : ℕ) (quantity : ℕ) : ℕ :=
  (cost_per_1000 * quantity) / 1000

def roof_tiles_cost (cost_per_tile : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_tile * quantity

def cement_bags_cost (cost_per_bag : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_bag * quantity

def wooden_beams_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def steel_bars_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def electrical_wiring_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def plumbing_pipes_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def total_cost : ℕ :=
  land_cost 60 2500 +
  bricks_cost 120 15000 +
  roof_tiles_cost 12 800 +
  cement_bags_cost 8 250 +
  wooden_beams_cost 25 1000 +
  steel_bars_cost 15 500 +
  electrical_wiring_cost 2 2000 +
  plumbing_pipes_cost 4 3000

theorem construction_cost_is_correct : total_cost = 212900 :=
  by
    sorry

end construction_cost_is_correct_l191_191952


namespace greatest_fraction_l191_191570

theorem greatest_fraction 
  (w x y z : ℕ)
  (hw : w > 0)
  (h_ordering : w < x ∧ x < y ∧ y < z) :
  (x + y + z) / (w + x + y) > (w + x + y) / (x + y + z) ∧
  (x + y + z) / (w + x + y) > (w + y + z) / (x + w + z) ∧
  (x + y + z) / (w + x + y) > (x + w + z) / (w + y + z) ∧
  (x + y + z) / (w + x + y) > (y + z + w) / (x + y + z) :=
sorry

end greatest_fraction_l191_191570


namespace time_spent_answering_questions_l191_191024

theorem time_spent_answering_questions (total_questions answered_per_question_minutes unanswered_questions : ℕ) (minutes_per_hour : ℕ) :
  total_questions = 100 → unanswered_questions = 40 → answered_per_question_minutes = 2 → minutes_per_hour = 60 → 
  ((total_questions - unanswered_questions) * answered_per_question_minutes) / minutes_per_hour = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end time_spent_answering_questions_l191_191024


namespace sheila_weekly_earnings_l191_191073

theorem sheila_weekly_earnings:
  (∀(m w f : ℕ), (m = 8) → (w = 8) → (f = 8) → 
   ∀(t th : ℕ), (t = 6) → (th = 6) → 
   ∀(h : ℕ), (h = 6) → 
   (m + w + f + t + th) * h = 216) := by
  sorry

end sheila_weekly_earnings_l191_191073


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l191_191704

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem statement_A : ∀ (x y : ℝ), diamond x y = diamond y x := sorry

theorem statement_B : ∀ (x y : ℝ), 2 * (diamond x y) ≠ diamond (2 * x) (2 * y) := sorry

theorem statement_C : ∀ (x : ℝ), diamond x 0 = x^2 := sorry

theorem statement_D : ∀ (x : ℝ), diamond x x = 0 := sorry

theorem statement_E : ∀ (x y : ℝ), x = y → diamond x y = 0 := sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l191_191704


namespace smallest_number_l191_191637

theorem smallest_number (x : ℕ) (h1 : 2 * x = third) (h2 : 4 * x = second) (h3 : 7 * x = fourth) (h4 : (x + second + third + fourth) / 4 = 77) :
  x = 22 :=
by sorry

end smallest_number_l191_191637


namespace anne_cleaning_time_l191_191415

theorem anne_cleaning_time (B A : ℝ) 
  (h₁ : 4 * (B + A) = 1) 
  (h₂ : 3 * (B + 2 * A) = 1) : 
  1 / A = 12 :=
sorry

end anne_cleaning_time_l191_191415


namespace sum_mod_7_l191_191552

/-- Define the six numbers involved. -/
def a := 102345
def b := 102346
def c := 102347
def d := 102348
def e := 102349
def f := 102350

/-- State the theorem to prove the remainder of their sum when divided by 7. -/
theorem sum_mod_7 : 
  (a + b + c + d + e + f) % 7 = 5 := 
by sorry

end sum_mod_7_l191_191552


namespace minimum_value_of_expression_l191_191009

theorem minimum_value_of_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 := by
  sorry

end minimum_value_of_expression_l191_191009


namespace find_speed_of_stream_l191_191021

def boat_speeds (V_b V_s : ℝ) : Prop :=
  V_b + V_s = 10 ∧ V_b - V_s = 8

theorem find_speed_of_stream (V_b V_s : ℝ) (h : boat_speeds V_b V_s) : V_s = 1 :=
by
  sorry

end find_speed_of_stream_l191_191021


namespace complement_of_A_in_U_l191_191367

noncomputable def U := {x : ℝ | Real.exp x > 1}

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

def A := { x : ℝ | x > 1 }

def compl (U A : Set ℝ) := { x : ℝ | x ∈ U ∧ x ∉ A }

theorem complement_of_A_in_U : compl U A = { x : ℝ | 0 < x ∧ x ≤ 1 } := sorry

end complement_of_A_in_U_l191_191367


namespace gcd_8m_6n_l191_191822

theorem gcd_8m_6n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : Nat.gcd m n = 7) : Nat.gcd (8 * m) (6 * n) = 14 := 
by
  sorry

end gcd_8m_6n_l191_191822


namespace distance_between_consecutive_trees_l191_191171

theorem distance_between_consecutive_trees 
  (yard_length : ℕ) (num_trees : ℕ) (tree_at_each_end : yard_length > 0 ∧ num_trees ≥ 2) 
  (equal_distances : ∀ k, k < num_trees - 1 → (yard_length / (num_trees - 1) : ℝ) = 12) :
  yard_length = 360 → num_trees = 31 → (yard_length / (num_trees - 1) : ℝ) = 12 := 
by
  sorry

end distance_between_consecutive_trees_l191_191171


namespace members_playing_badminton_l191_191957

theorem members_playing_badminton
  (total_members : ℕ := 42)
  (tennis_players : ℕ := 23)
  (neither_players : ℕ := 6)
  (both_players : ℕ := 7) :
  ∃ (badminton_players : ℕ), badminton_players = 20 :=
by
  have union_players := total_members - neither_players
  have badminton_players := union_players - (tennis_players - both_players)
  use badminton_players
  sorry

end members_playing_badminton_l191_191957


namespace adah_practiced_total_hours_l191_191219

theorem adah_practiced_total_hours :
  let minutes_per_day := 86
  let days_practiced := 2
  let minutes_other_days := 278
  let total_minutes := (minutes_per_day * days_practiced) + minutes_other_days
  let total_hours := total_minutes / 60
  total_hours = 7.5 :=
by
  sorry

end adah_practiced_total_hours_l191_191219


namespace find_three_digit_number_l191_191437

theorem find_three_digit_number (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : 0 ≤ c ∧ c ≤ 9)
    (h₄ : (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 33 / 37) :
    100 * a + 10 * b + c = 447 :=
sorry

end find_three_digit_number_l191_191437


namespace geometric_monotonic_condition_l191_191906

-- Definition of a geometrically increasing sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition of a monotonically increasing sequence
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

-- The theorem statement
theorem geometric_monotonic_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ monotonically_increasing a :=
sorry

end geometric_monotonic_condition_l191_191906


namespace division_of_fractions_l191_191799

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l191_191799


namespace top_weight_l191_191275

theorem top_weight (T : ℝ) : 
    (9 * 0.8 + 7 * T = 10.98) → T = 0.54 :=
by 
  intro h
  have H_sum := h
  simp only [mul_add, add_assoc, mul_assoc, mul_comm, add_comm, mul_comm 7] at H_sum
  sorry

end top_weight_l191_191275


namespace total_comics_in_box_l191_191901

theorem total_comics_in_box 
  (pages_per_comic : ℕ)
  (total_pages_found : ℕ)
  (untorn_comics : ℕ)
  (comics_fixed : ℕ := total_pages_found / pages_per_comic)
  (total_comics : ℕ := comics_fixed + untorn_comics)
  (h_pages_per_comic : pages_per_comic = 25)
  (h_total_pages_found : total_pages_found = 150)
  (h_untorn_comics : untorn_comics = 5) :
  total_comics = 11 :=
by
  sorry

end total_comics_in_box_l191_191901


namespace circle_line_bisect_l191_191251

theorem circle_line_bisect (a : ℝ) :
    (∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = 5 → 3 * x + y + a = 0) → a = 1 :=
sorry

end circle_line_bisect_l191_191251


namespace standard_deviations_below_l191_191979

variable (σ : ℝ)
variable (mean : ℝ)
variable (score98 : ℝ)
variable (score58 : ℝ)

-- Conditions translated to Lean definitions
def condition_1 : Prop := score98 = mean + 3 * σ
def condition_2 : Prop := mean = 74
def condition_3 : Prop := σ = 8

-- Target statement: Prove that the score of 58 is 2 standard deviations below the mean
theorem standard_deviations_below : condition_1 σ mean score98 → condition_2 mean → condition_3 σ → score58 = 74 - 2 * σ :=
by
  intro h1 h2 h3
  sorry

end standard_deviations_below_l191_191979


namespace set_difference_M_N_l191_191123

def setM : Set ℝ := { x | -1 < x ∧ x < 1 }
def setN : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem set_difference_M_N :
  setM \ setN = { x | -1 < x ∧ x < 0 } := sorry

end set_difference_M_N_l191_191123


namespace intersection_of_A_and_B_l191_191129

def A : Set ℝ := { x | x < 3 }
def B : Set ℝ := { x | Real.log (x - 1) / Real.log 3 > 0 }

theorem intersection_of_A_and_B :
  (A ∩ B) = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l191_191129


namespace half_angle_in_second_quadrant_l191_191312

def quadrant_of_half_alpha (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : Prop :=
  π / 2 < α / 2 ∧ α / 2 < 3 * π / 4

theorem half_angle_in_second_quadrant (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : quadrant_of_half_alpha α hα1 hα2 hcos :=
sorry

end half_angle_in_second_quadrant_l191_191312


namespace total_right_handed_players_l191_191142

theorem total_right_handed_players
  (total_players throwers mp_players non_throwers L R : ℕ)
  (ratio_L_R : 2 * R = 3 * L)
  (total_eq : total_players = 120)
  (throwers_eq : throwers = 60)
  (mp_eq : mp_players = 20)
  (non_throwers_eq : non_throwers = total_players - throwers - mp_players)
  (non_thrower_sum_eq : L + R = non_throwers) :
  (throwers + mp_players + R = 104) :=
by
  sorry

end total_right_handed_players_l191_191142


namespace factorization_problem1_factorization_problem2_l191_191041

-- Mathematical statements
theorem factorization_problem1 (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 := by
  sorry

theorem factorization_problem2 (a : ℝ) : 18 * a^2 - 50 = 2 * (3 * a + 5) * (3 * a - 5) := by
  sorry

end factorization_problem1_factorization_problem2_l191_191041


namespace find_x_l191_191644

open Real

noncomputable def log_base (b x : ℝ) : ℝ := log x / log b

theorem find_x :
  ∃ x : ℝ, 0 < x ∧
  log_base 5 (x - 1) + log_base (sqrt 5) (x^2 - 1) + log_base (1/5) (x - 1) = 3 ∧
  x = sqrt (5 * sqrt 5 + 1) :=
by
  sorry

end find_x_l191_191644


namespace range_a_l191_191266

open Set Real

-- Define the predicate p: real number x satisfies x^2 - 4ax + 3a^2 < 0, where a < 0
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

-- Define the predicate q: real number x satisfies x^2 - x - 6 ≤ 0, or x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- Define the complement sets
def not_p_set (a : ℝ) : Set ℝ := {x | ¬p x a}
def not_q_set : Set ℝ := {x | ¬q x}

-- Define p as necessary but not sufficient condition for q
def necessary_but_not_sufficient (a : ℝ) : Prop := 
  (not_q_set ⊆ not_p_set a) ∧ ¬(not_p_set a ⊆ not_q_set)

-- The main theorem to prove
theorem range_a : {a : ℝ | necessary_but_not_sufficient a} = {a : ℝ | -4 ≤ a ∧ a < 0 ∨ a ≤ -4} :=
by
  sorry

end range_a_l191_191266


namespace smallest_number_is_61_point_4_l191_191374

theorem smallest_number_is_61_point_4 (x y z t : ℝ)
  (h1 : y = 2 * x)
  (h2 : z = 4 * y)
  (h3 : t = (y + z) / 3)
  (h4 : (x + y + z + t) / 4 = 220) :
  x = 2640 / 43 :=
by sorry

end smallest_number_is_61_point_4_l191_191374


namespace side_length_square_l191_191062

-- Define the length and width of the rectangle
def length_rect := 10 -- cm
def width_rect := 8 -- cm

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Define the perimeter of the square
def perimeter_square (s : ℕ) := 4 * s

-- The theorem to prove
theorem side_length_square : ∃ s : ℕ, perimeter_rect = perimeter_square s ∧ s = 9 :=
by
  sorry

end side_length_square_l191_191062


namespace find_sinα_and_tanα_l191_191224

open Real 

noncomputable def vectors (α : ℝ) := (Real.cos α, 1)

noncomputable def vectors_perpendicular (α : ℝ) := (Real.sin α, -2)

theorem find_sinα_and_tanα (α: ℝ) (hα: π < α ∧ α < 3 * π / 2)
  (h_perp: vectors_perpendicular α = (Real.sin α, -2) ∧ vectors α = (Real.cos α, 1) ∧ (vectors α).1 * (vectors_perpendicular α).1 + (vectors α).2 * (vectors_perpendicular α).2 = 0):
  (Real.sin α = - (2 * Real.sqrt 5) / 5) ∧ 
  (Real.tan (α + π / 4) = -3) := 
sorry 

end find_sinα_and_tanα_l191_191224


namespace white_trees_count_l191_191104

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l191_191104


namespace cost_of_camel_proof_l191_191352

noncomputable def cost_of_camel (C H O E : ℕ) : ℕ :=
  if 10 * C = 24 * H ∧ 16 * H = 4 * O ∧ 6 * O = 4 * E ∧ 10 * E = 120000 then 4800 else 0

theorem cost_of_camel_proof (C H O E : ℕ) 
  (h1 : 10 * C = 24 * H) (h2 : 16 * H = 4 * O) (h3 : 6 * O = 4 * E) (h4 : 10 * E = 120000) :
  cost_of_camel C H O E = 4800 :=
by
  sorry

end cost_of_camel_proof_l191_191352


namespace min_value_x_squared_y_squared_z_squared_l191_191302

theorem min_value_x_squared_y_squared_z_squared
  (x y z : ℝ)
  (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ (18 / 7) :=
sorry

end min_value_x_squared_y_squared_z_squared_l191_191302


namespace students_needed_to_fill_buses_l191_191169

theorem students_needed_to_fill_buses (n : ℕ) (c : ℕ) (h_n : n = 254) (h_c : c = 30) : 
  (c * ((n + c - 1) / c) - n) = 16 :=
by
  sorry

end students_needed_to_fill_buses_l191_191169


namespace ball_draw_probability_red_is_one_ninth_l191_191582

theorem ball_draw_probability_red_is_one_ninth :
  let A_red := 4
  let A_white := 2
  let B_red := 1
  let B_white := 5
  let P_red_A := A_red / (A_red + A_white)
  let P_red_B := B_red / (B_red + B_white)
  P_red_A * P_red_B = 1 / 9 := by
    -- Proof here
    sorry

end ball_draw_probability_red_is_one_ninth_l191_191582


namespace prove_system_of_equations_l191_191537

variables (x y : ℕ)

def system_of_equations (x y : ℕ) : Prop :=
  x = 2*y + 4 ∧ x = 3*y - 9

theorem prove_system_of_equations :
  ∀ (x y : ℕ), system_of_equations x y :=
by sorry

end prove_system_of_equations_l191_191537


namespace contrapositive_equiv_l191_191913

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv_l191_191913


namespace Sam_has_seven_watermelons_l191_191280

-- Declare the initial number of watermelons
def initial_watermelons : Nat := 4

-- Declare the additional number of watermelons Sam grew
def more_watermelons : Nat := 3

-- Prove that the total number of watermelons is 7
theorem Sam_has_seven_watermelons : initial_watermelons + more_watermelons = 7 :=
by
  sorry

end Sam_has_seven_watermelons_l191_191280


namespace combined_height_of_trees_is_correct_l191_191620

noncomputable def original_height_of_trees 
  (h1_current : ℝ) (h1_growth_rate : ℝ)
  (h2_current : ℝ) (h2_growth_rate : ℝ)
  (h3_current : ℝ) (h3_growth_rate : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  let h1 := h1_current / (1 + h1_growth_rate)
  let h2 := h2_current / (1 + h2_growth_rate)
  let h3 := h3_current / (1 + h3_growth_rate)
  (h1 + h2 + h3) / conversion_rate

theorem combined_height_of_trees_is_correct :
  original_height_of_trees 240 0.70 300 0.50 180 0.60 12 = 37.81 :=
by
  sorry

end combined_height_of_trees_is_correct_l191_191620


namespace interval_monotonic_decrease_min_value_g_l191_191445

noncomputable def a (x : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.sin x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := let (a1, a2) := a x; let (b1, b2) := b x; a1 * b1 + a2 * b2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x + m

theorem interval_monotonic_decrease (x : ℝ) (k : ℤ) :
  0 ≤ x ∧ x ≤ Real.pi ∧ (2 * x + Real.pi / 6) ∈ [Real.pi/2 + 2 * (k : ℝ) * Real.pi, 3 * Real.pi/2 + 2 * (k : ℝ) * Real.pi] →
  x ∈ [Real.pi / 6 + (k : ℝ) * Real.pi, 2 * Real.pi / 3 + (k : ℝ) * Real.pi] := sorry

theorem min_value_g (x : ℝ) :
  x ∈ [- Real.pi / 3, Real.pi / 3] →
  ∃ x₀, g x₀ 1 = -1/2 ∧ x₀ = - Real.pi / 3 := sorry

end interval_monotonic_decrease_min_value_g_l191_191445


namespace value_of_f_2_pow_100_l191_191931

def f : ℕ → ℕ :=
sorry

axiom f_base : f 1 = 1
axiom f_recursive : ∀ n : ℕ, f (2 * n) = n * f n

theorem value_of_f_2_pow_100 : f (2^100) = 2^4950 :=
sorry

end value_of_f_2_pow_100_l191_191931


namespace evaluate_expression_l191_191623

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a^a - a*(a-2)^a)^a = 1358954496 :=
by
  rw [h]  -- Substitute a with 4
  sorry

end evaluate_expression_l191_191623


namespace find_consecutive_integers_sum_eq_l191_191287

theorem find_consecutive_integers_sum_eq 
    (M : ℤ) : ∃ n k : ℤ, (0 ≤ k ∧ k ≤ 9) ∧ (M = (9 * n + 45 - k)) := 
sorry

end find_consecutive_integers_sum_eq_l191_191287


namespace pizzas_served_dinner_eq_6_l191_191539

-- Definitions based on the conditions
def pizzas_served_lunch : Nat := 9
def pizzas_served_today : Nat := 15

-- The theorem to prove the number of pizzas served during dinner
theorem pizzas_served_dinner_eq_6 : pizzas_served_today - pizzas_served_lunch = 6 := by
  sorry

end pizzas_served_dinner_eq_6_l191_191539


namespace find_n_l191_191910

theorem find_n (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 9) (h3 : n % 10 = -245 % 10) : n = 5 := 
  sorry

end find_n_l191_191910


namespace carsProducedInEurope_l191_191328

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope_l191_191328


namespace arrange_squares_l191_191397

theorem arrange_squares (n : ℕ) (h : n ≥ 5) :
  ∃ arrangement : Fin n → Fin n × Fin n, 
    (∀ i j : Fin n, i ≠ j → 
      (arrangement i).fst + (arrangement i).snd = (arrangement j).fst + (arrangement j).snd
      ∨ (arrangement i).fst = (arrangement j).fst
      ∨ (arrangement i).snd = (arrangement j).snd) :=
sorry

end arrange_squares_l191_191397


namespace sequence_v_n_l191_191378

theorem sequence_v_n (v : ℕ → ℝ)
  (h_recurr : ∀ n, v (n+2) = 3 * v (n+1) - v n)
  (h_init1 : v 3 = 16)
  (h_init2 : v 6 = 211) : 
  v 5 = 81.125 :=
sorry

end sequence_v_n_l191_191378


namespace shuttle_speed_conversion_l191_191857

-- Define the speed of the space shuttle in kilometers per second
def shuttle_speed_km_per_sec : ℕ := 6

-- Define the number of seconds in an hour
def seconds_per_hour : ℕ := 3600

-- Define the expected speed in kilometers per hour
def expected_speed_km_per_hour : ℕ := 21600

-- Prove that the speed converted to kilometers per hour is equal to the expected speed
theorem shuttle_speed_conversion : shuttle_speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hour :=
by
    sorry

end shuttle_speed_conversion_l191_191857


namespace count_two_digit_even_congruent_to_1_mod_4_l191_191594

theorem count_two_digit_even_congruent_to_1_mod_4 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n % 4 = 1 ∧ 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0) ∧ S.card = 23 := 
sorry

end count_two_digit_even_congruent_to_1_mod_4_l191_191594


namespace valid_set_example_l191_191191

def is_valid_set (S : Set ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, x ≠ y

theorem valid_set_example : is_valid_set { x : ℝ | x > Real.sqrt 2 } :=
sorry

end valid_set_example_l191_191191


namespace readers_both_l191_191783

-- Define the given conditions
def total_readers : ℕ := 250
def readers_S : ℕ := 180
def readers_L : ℕ := 88

-- Define the proof statement
theorem readers_both : (readers_S + readers_L - total_readers = 18) :=
by
  -- Proof is omitted
  sorry

end readers_both_l191_191783


namespace average_fuel_efficiency_round_trip_l191_191985

noncomputable def average_fuel_efficiency (d1 d2 mpg1 mpg2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let fuel_used := (d1 / mpg1) + (d2 / mpg2)
  total_distance / fuel_used

theorem average_fuel_efficiency_round_trip :
  average_fuel_efficiency 180 180 36 24 = 28.8 :=
by 
  sorry

end average_fuel_efficiency_round_trip_l191_191985


namespace vector_CD_l191_191472

-- Define the vector space and the vectors a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b : V)

-- Define the conditions
def is_on_line (D A B : V) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (D = t • A + (1 - t) • B)
def da_eq_2bd (D A B : V) := (A - D) = 2 • (D - B)

-- Define the triangle ABC and the specific vectors CA and CB
variables (CA := C - A) (CB := C - B)
variable (H1 : is_on_line D A B)
variable (H2 : da_eq_2bd D A B)
variable (H3 : CA = a)
variable (H4 : CB = b)

-- Prove the conclusion
theorem vector_CD (H1 : is_on_line D A B) (H2 : da_eq_2bd D A B)
  (H3 : CA = a) (H4 : CB = b) : 
  (C - D) = (1/3 : ℝ) • a + (2/3 : ℝ) • b :=
sorry

end vector_CD_l191_191472


namespace sequence_sixth_term_l191_191626

theorem sequence_sixth_term :
  ∃ (a : ℕ → ℕ),
    a 1 = 3 ∧
    a 5 = 43 ∧
    (∀ n, a (n + 1) = (1/4) * (a n + a (n + 2))) →
    a 6 = 129 :=
sorry

end sequence_sixth_term_l191_191626


namespace f_increasing_f_odd_zero_l191_191666

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- 1. Prove that f(x) is always an increasing function for any real a.
theorem f_increasing (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
by
  sorry

-- 2. Determine the value of a such that f(-x) + f(x) = 0 always holds.
theorem f_odd_zero (a : ℝ) : (∀ x : ℝ, f a (-x) + f a x = 0) → a = 1 :=
by
  sorry

end f_increasing_f_odd_zero_l191_191666


namespace ryan_hours_english_is_6_l191_191253

def hours_chinese : Nat := 2

def hours_english (C : Nat) : Nat := C + 4

theorem ryan_hours_english_is_6 (C : Nat) (hC : C = hours_chinese) : hours_english C = 6 :=
by
  sorry

end ryan_hours_english_is_6_l191_191253


namespace Hiram_age_l191_191850

theorem Hiram_age (H A : ℕ) (h₁ : H + 12 = 2 * A - 4) (h₂ : A = 28) : H = 40 :=
by
  sorry

end Hiram_age_l191_191850


namespace pen_shorter_than_pencil_l191_191419

-- Definitions of the given conditions
def P (R : ℕ) := R + 3
def L : ℕ := 12
def total_length (R : ℕ) := R + P R + L

-- The theorem to be proven
theorem pen_shorter_than_pencil (R : ℕ) (h : total_length R = 29) : L - P R = 2 :=
by
  sorry

end pen_shorter_than_pencil_l191_191419


namespace average_speed_l191_191549

theorem average_speed (speed1 speed2: ℝ) (time1 time2: ℝ) (h1: speed1 = 90) (h2: speed2 = 40) (h3: time1 = 1) (h4: time2 = 1) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 65 := by
  sorry

end average_speed_l191_191549


namespace eliminate_all_evil_with_at_most_one_good_l191_191853

-- Defining the problem setting
structure Wizard :=
  (is_good : Bool)

-- The main theorem
theorem eliminate_all_evil_with_at_most_one_good (wizards : List Wizard) (h_wizard_count : wizards.length = 2015) :
  ∃ (banish_sequence : List Wizard), 
    (∀ w ∈ banish_sequence, w.is_good = false) ∨ (∃ (g : Wizard), g.is_good = true ∧ g ∉ banish_sequence) :=
sorry

end eliminate_all_evil_with_at_most_one_good_l191_191853


namespace solution_l191_191568

noncomputable def problem : Prop :=
  (2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2) ∧
  (1 - 2 * Real.sin (Real.pi / 12) ^ 2 ≠ 1 / 2) ∧
  (Real.cos (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 
   Real.sin (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 2) ∧
  ( (Real.tan (77 * Real.pi / 180) - Real.tan (32 * Real.pi / 180)) /
    (2 * (1 + Real.tan (77 * Real.pi / 180) * Real.tan (32 * Real.pi / 180))) = 1 / 2 )

theorem solution : problem :=
  by 
    sorry

end solution_l191_191568


namespace lattice_points_in_region_l191_191475

theorem lattice_points_in_region : ∃! n : ℕ, n = 14 ∧ ∀ (x y : ℤ), (y = |x| ∨ y = -x^2 + 4) ∧ (-2 ≤ x ∧ x ≤ 1) → 
  (y = -x^2 + 4 ∧ y = |x|) :=
sorry

end lattice_points_in_region_l191_191475


namespace pyramid_surface_area_l191_191455

-- Definitions for the conditions
structure Rectangle where
  length : ℝ
  width : ℝ

structure Pyramid where
  base : Rectangle
  height : ℝ

-- Create instances representing the given conditions
noncomputable def givenRectangle : Rectangle := {
  length := 8,
  width := 6
}

noncomputable def givenPyramid : Pyramid := {
  base := givenRectangle,
  height := 15
}

-- Statement to prove the surface area of the pyramid
theorem pyramid_surface_area
  (rect: Rectangle)
  (length := rect.length)
  (width := rect.width)
  (height: ℝ)
  (hy1: length = 8)
  (hy2: width = 6)
  (hy3: height = 15) :
  let base_area := length * width
  let slant_height := Real.sqrt (height^2 + (length / 2)^2)
  let lateral_area := 2 * ((length * slant_height) / 2 + (width * slant_height) / 2)
  let total_surface_area := base_area + lateral_area 
  total_surface_area = 48 + 7 * Real.sqrt 241 := 
  sorry

end pyramid_surface_area_l191_191455


namespace katie_candy_l191_191016

theorem katie_candy (K : ℕ) (H1 : K + 6 - 9 = 7) : K = 10 :=
by
  sorry

end katie_candy_l191_191016


namespace cone_volume_l191_191000

noncomputable def radius_of_sector : ℝ := 6
noncomputable def arc_length_of_sector : ℝ := (1 / 2) * (2 * Real.pi * radius_of_sector)
noncomputable def radius_of_base : ℝ := arc_length_of_sector / (2 * Real.pi)
noncomputable def slant_height : ℝ := radius_of_sector
noncomputable def height_of_cone : ℝ := Real.sqrt (slant_height^2 - radius_of_base^2)
noncomputable def volume_of_cone : ℝ := (1 / 3) * Real.pi * (radius_of_base^2) * height_of_cone

theorem cone_volume : volume_of_cone = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end cone_volume_l191_191000


namespace find_m_l191_191529

-- Conditions given
def ellipse (x y m : ℝ) : Prop := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- The theorem to prove
theorem find_m (m : ℝ) (h₁ : ellipse 1 1 m) (h₂ : eccentricity 2) : m = 3 ∨ m = 5 :=
  sorry

end find_m_l191_191529


namespace equidistant_P_AP_BP_CP_DP_l191_191827

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2

def A : Point := ⟨10, 0, 0⟩
def B : Point := ⟨0, -6, 0⟩
def C : Point := ⟨0, 0, 8⟩
def D : Point := ⟨0, 0, 0⟩
def P : Point := ⟨5, -3, 4⟩

theorem equidistant_P_AP_BP_CP_DP :
  distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D := 
sorry

end equidistant_P_AP_BP_CP_DP_l191_191827


namespace problem_solution_l191_191501

-- Definitions based on conditions
def p (a b : ℝ) : Prop := a > b → a^2 > b^2
def neg_p (a b : ℝ) : Prop := a > b → a^2 ≤ b^2
def disjunction (p q : Prop) : Prop := p ∨ q
def suff_but_not_nec (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬(x > 1 → x > 2)
def congruent_triangles (T1 T2 : Prop) : Prop := T1 → T2
def neg_congruent_triangles (T1 T2 : Prop) : Prop := ¬(T1 → T2)

-- Mathematical problem as Lean statements
theorem problem_solution :
  ( (∀ a b : ℝ, p a b = (a > b → a^2 > b^2) ∧ neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = true ↔ ¬(T1 → T2)) ) →
  ( (∀ a b : ℝ, neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = false) ) :=
sorry

end problem_solution_l191_191501


namespace ab_value_l191_191072

-- Define sets A and B
def A : Set ℝ := {-1.3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a * x + b = 0}

-- The proof statement: Given A = B, prove ab = 0.104
theorem ab_value (a b : ℝ) (h : A = B a b) : a * b = 0.104 :=
by
  sorry

end ab_value_l191_191072


namespace proposition_false_l191_191884

theorem proposition_false : ¬ ∀ x ∈ ({1, -1, 0} : Set ℤ), 2 * x + 1 > 0 := by
  sorry

end proposition_false_l191_191884


namespace student_weight_loss_l191_191775

variables (S R L : ℕ)

theorem student_weight_loss :
  S = 75 ∧ S + R = 110 ∧ S - L = 2 * R → L = 5 :=
by
  sorry

end student_weight_loss_l191_191775


namespace smallest_value_of_n_l191_191483

theorem smallest_value_of_n :
  ∃ o y m n : ℕ, 10 * o = 16 * y ∧ 16 * y = 18 * m ∧ 18 * m = 18 * n ∧ n = 40 := 
sorry

end smallest_value_of_n_l191_191483


namespace percent_commute_l191_191108

variable (x : ℝ)

theorem percent_commute (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_commute_l191_191108


namespace solve_system_l191_191727

theorem solve_system :
  ∃ x y : ℚ, 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 :=
by {
  sorry
}

end solve_system_l191_191727


namespace snow_first_day_eq_six_l191_191095

variable (snow_first_day snow_second_day snow_fourth_day snow_fifth_day : ℤ)

theorem snow_first_day_eq_six
  (h1 : snow_second_day = snow_first_day + 8)
  (h2 : snow_fourth_day = snow_second_day - 2)
  (h3 : snow_fifth_day = snow_fourth_day + 2 * snow_first_day)
  (h4 : snow_fifth_day = 24) :
  snow_first_day = 6 := by
  sorry

end snow_first_day_eq_six_l191_191095


namespace weight_lifting_requirement_l191_191559

-- Definitions based on conditions
def weight_25 : Int := 25
def weight_10 : Int := 10
def lifts_25 := 16
def total_weight_25 := 2 * weight_25 * lifts_25

def n_lifts_10 (n : Int) := 2 * weight_10 * n

-- Problem statement and theorem to prove
theorem weight_lifting_requirement (n : Int) : n_lifts_10 n = total_weight_25 ↔ n = 40 := by
  sorry

end weight_lifting_requirement_l191_191559


namespace range_f_contained_in_0_1_l191_191473

theorem range_f_contained_in_0_1 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := 
by {
  sorry
}

end range_f_contained_in_0_1_l191_191473


namespace jennifer_score_l191_191547

theorem jennifer_score 
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (unanswered_questions : ℕ)
  (points_per_correct : ℤ)
  (points_deduction_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (h_total : total_questions = 30)
  (h_correct : correct_answers = 15)
  (h_incorrect : incorrect_answers = 10)
  (h_unanswered : unanswered_questions = 5)
  (h_points_correct : points_per_correct = 2)
  (h_deduction_incorrect : points_deduction_incorrect = -1)
  (h_points_unanswered : points_per_unanswered = 0) : 
  ∃ (score : ℤ), score = (correct_answers * points_per_correct 
                          + incorrect_answers * points_deduction_incorrect 
                          + unanswered_questions * points_per_unanswered) 
                        ∧ score = 20 := 
by
  sorry

end jennifer_score_l191_191547


namespace smallest_lambda_inequality_l191_191490

theorem smallest_lambda_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * y * (x^2 + y^2) + y * z * (y^2 + z^2) + z * x * (z^2 + x^2) ≤ (1 / 8) * (x + y + z)^4 :=
sorry

end smallest_lambda_inequality_l191_191490


namespace neg_of_exists_lt_is_forall_ge_l191_191725

theorem neg_of_exists_lt_is_forall_ge :
  (¬ (∃ x : ℝ, x^2 - 2 * x + 1 < 0)) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end neg_of_exists_lt_is_forall_ge_l191_191725


namespace division_identity_l191_191453

theorem division_identity :
  (0.0048 * 3.5) / (0.05 * 0.1 * 0.004) = 840 :=
by
  -- TODO: Provide the proof here
  sorry

end division_identity_l191_191453


namespace solution_set_inequality_l191_191335

theorem solution_set_inequality (x : ℝ) : (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 :=
by
  -- Proof omitted
  sorry

end solution_set_inequality_l191_191335


namespace find_a_from_polynomial_factor_l191_191386

theorem find_a_from_polynomial_factor (a b : ℤ)
  (h: ∀ x : ℝ, x*x - x - 1 = 0 → a*x^5 + b*x^4 + 1 = 0) : a = 3 :=
sorry

end find_a_from_polynomial_factor_l191_191386


namespace x_y_result_l191_191828

noncomputable def x_y_value (x y : ℝ) : ℝ := x + y

theorem x_y_result (x y : ℝ) 
  (h1 : x + Real.cos y = 3009) 
  (h2 : x + 3009 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x_y_value x y = 3009 + Real.pi / 2 :=
by
  sorry

end x_y_result_l191_191828


namespace angle_PTV_60_l191_191757

variables (m n TV TPV PTV : ℝ)

-- We state the conditions
axiom parallel_lines : m = n
axiom angle_TPV : TPV = 150
axiom angle_TVP_perpendicular : TV = 90

-- The goal statement to prove
theorem angle_PTV_60 : PTV = 60 :=
by
  sorry

end angle_PTV_60_l191_191757


namespace count_perfect_cube_or_fourth_power_lt_1000_l191_191130

theorem count_perfect_cube_or_fourth_power_lt_1000 :
  ∃ n, n = 14 ∧ (∀ x, (0 < x ∧ x < 1000 ∧ (∃ k, x = k^3 ∨ x = k^4)) ↔ ∃ i, i < n) :=
by sorry

end count_perfect_cube_or_fourth_power_lt_1000_l191_191130


namespace compare_sine_values_1_compare_sine_values_2_l191_191765

theorem compare_sine_values_1 (h1 : 0 < Real.pi / 10) (h2 : Real.pi / 10 < Real.pi / 8) (h3 : Real.pi / 8 < Real.pi / 2) :
  Real.sin (- Real.pi / 10) > Real.sin (- Real.pi / 8) :=
by
  sorry

theorem compare_sine_values_2 (h1 : 0 < Real.pi / 8) (h2 : Real.pi / 8 < 3 * Real.pi / 8) (h3 : 3 * Real.pi / 8 < Real.pi / 2) :
  Real.sin (7 * Real.pi / 8) < Real.sin (5 * Real.pi / 8) :=
by
  sorry

end compare_sine_values_1_compare_sine_values_2_l191_191765


namespace mike_ride_distance_l191_191815

theorem mike_ride_distance (M : ℕ) 
  (cost_Mike : ℝ) 
  (cost_Annie : ℝ) 
  (annies_miles : ℕ := 26) 
  (annies_toll : ℝ := 5) 
  (mile_cost : ℝ := 0.25) 
  (initial_fee : ℝ := 2.5)
  (hc_Mike : cost_Mike = initial_fee + mile_cost * M)
  (hc_Annie : cost_Annie = initial_fee + annies_toll + mile_cost * annies_miles)
  (heq : cost_Mike = cost_Annie) :
  M = 46 := by 
  sorry

end mike_ride_distance_l191_191815


namespace price_reduction_equation_l191_191070

variable (x : ℝ)

theorem price_reduction_equation (h : 25 * (1 - x) ^ 2 = 16) : 25 * (1 - x) ^ 2 = 16 :=
by
  assumption

end price_reduction_equation_l191_191070


namespace smallest_number_am_median_largest_l191_191520

noncomputable def smallest_number (a b c : ℕ) : ℕ :=
if a ≤ b ∧ a ≤ c then a
else if b ≤ a ∧ b ≤ c then b
else c

theorem smallest_number_am_median_largest (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 28) (h3 : c = b + 6) :
  smallest_number a b c = 28 :=
sorry

end smallest_number_am_median_largest_l191_191520


namespace tangent_line_eq_l191_191750

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

theorem tangent_line_eq {x y : ℝ} (hx : x = 1) (hy : y = 2) (H : circle_eq x y) :
  y = 2 :=
by
  sorry

end tangent_line_eq_l191_191750


namespace difference_two_smallest_integers_l191_191506

/--
There is more than one integer greater than 1 which, when divided by any integer k such that 2 ≤ k ≤ 11, has a remainder of 1.
Prove that the difference between the two smallest such integers is 27720.
-/
theorem difference_two_smallest_integers :
  ∃ n₁ n₂ : ℤ, 
  (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (n₁ % k = 1 ∧ n₂ % k = 1)) ∧ 
  n₁ > 1 ∧ n₂ > 1 ∧ 
  ∀ m : ℤ, (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (m % k =  1)) ∧ m > 1 → m = n₁ ∨ m = n₂ → 
  (n₂ - n₁ = 27720) := 
sorry

end difference_two_smallest_integers_l191_191506


namespace find_a_given_coefficient_l191_191959

theorem find_a_given_coefficient (a : ℝ) :
  (∀ x : ℝ, a ≠ 0 → x ≠ 0 → a^4 * x^4 + 4 * a^3 * x^2 * (1/x) + 6 * a^2 * (1/x)^2 * x^4 + 4 * a * (1/x)^3 * x^6 + (1/x)^4 * x^8 = (ax + 1/x)^4) → (4 * a^3 = 32) → a = 2 :=
by
  intros H1 H2
  sorry

end find_a_given_coefficient_l191_191959


namespace platform_length_l191_191212

theorem platform_length
  (train_length : ℤ)
  (speed_kmph : ℤ)
  (time_sec : ℤ)
  (speed_mps : speed_kmph * 1000 / 3600 = 20)
  (distance_eq : (train_length + 220) = (20 * time_sec))
  (train_length_val : train_length = 180)
  (time_sec_val : time_sec = 20) :
  220 = 220 := by
  sorry

end platform_length_l191_191212


namespace sound_heard_in_4_seconds_l191_191994

/-- Given the distance between a boy and his friend is 1200 meters,
    the speed of the car is 108 km/hr, and the speed of sound is 330 m/s,
    the duration after which the friend hears the whistle is 4 seconds. -/
theorem sound_heard_in_4_seconds :
  let distance := 1200  -- distance in meters
  let speed_of_car_kmh := 108  -- speed of car in km/hr
  let speed_of_sound := 330  -- speed of sound in m/s
  let speed_of_car := speed_of_car_kmh * 1000 / 3600  -- convert km/hr to m/s
  let effective_speed_of_sound := speed_of_sound - speed_of_car
  let time := distance / effective_speed_of_sound
  time = 4 := 
by
  sorry

end sound_heard_in_4_seconds_l191_191994


namespace problem_statement_l191_191348

theorem problem_statement (x y : ℝ) : 
  ((-3 * x * y^2)^3 * (-6 * x^2 * y) / (9 * x^4 * y^5) = 18 * x * y^2) :=
by sorry

end problem_statement_l191_191348


namespace negative_integer_solution_l191_191058

theorem negative_integer_solution (x : ℤ) (h : 3 * x + 13 ≥ 0) : x = -1 :=
by
  sorry

end negative_integer_solution_l191_191058


namespace candy_cost_l191_191608

theorem candy_cost
    (grape_candies : ℕ)
    (cherry_candies : ℕ)
    (apple_candies : ℕ)
    (total_cost : ℝ)
    (total_candies : ℕ)
    (cost_per_candy : ℝ)
    (h1 : grape_candies = 24)
    (h2 : grape_candies = 3 * cherry_candies)
    (h3 : apple_candies = 2 * grape_candies)
    (h4 : total_cost = 200)
    (h5 : total_candies = cherry_candies + grape_candies + apple_candies)
    (h6 : cost_per_candy = total_cost / total_candies) :
    cost_per_candy = 2.50 :=
by
    sorry

end candy_cost_l191_191608


namespace pencils_initial_count_l191_191694

theorem pencils_initial_count (pencils_given : ℕ) (pencils_left : ℕ) (initial_pencils : ℕ) :
  pencils_given = 31 → pencils_left = 111 → initial_pencils = pencils_given + pencils_left → initial_pencils = 142 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end pencils_initial_count_l191_191694


namespace average_first_14_even_numbers_l191_191781

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun x => 2 * (x + 1))

theorem average_first_14_even_numbers :
  let even_nums := first_n_even_numbers 14
  (even_nums.sum / even_nums.length = 15) :=
by
  sorry

end average_first_14_even_numbers_l191_191781


namespace proof_min_k_l191_191981

-- Define the number of teachers
def num_teachers : ℕ := 200

-- Define what it means for a teacher to send a message to another teacher.
-- Represent this as a function where each teacher sends a message to exactly one other teacher.
def sends_message (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∀ i : Fin num_teachers, ∃ j : Fin num_teachers, teachers i = j

-- Define the main proposition: there exists a group of 67 teachers where no one sends a message to anyone else in the group.
def min_k (teachers : Fin num_teachers → Fin num_teachers) : Prop :=
  ∃ (k : ℕ) (reps : Fin k → Fin num_teachers), k ≥ 67 ∧
  ∀ (i j : Fin k), i ≠ j → teachers (reps i) ≠ reps j

theorem proof_min_k : ∀ (teachers : Fin num_teachers → Fin num_teachers),
  sends_message teachers → min_k teachers :=
sorry

end proof_min_k_l191_191981


namespace part1_part2_l191_191945

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l191_191945


namespace tangent_line_at_point_l191_191250

theorem tangent_line_at_point (x y : ℝ) (h : y = x / (x - 2)) (hx : x = 1) (hy : y = -1) : y = -2 * x + 1 :=
sorry

end tangent_line_at_point_l191_191250


namespace solve_for_x_l191_191996

theorem solve_for_x (x y : ℝ) (h1 : 2 * x - 3 * y = 18) (h2 : x + 2 * y = 8) : x = 60 / 7 := sorry

end solve_for_x_l191_191996


namespace stork_count_l191_191972

theorem stork_count (B S : ℕ) (h1 : B = 7) (h2 : B = S + 3) : S = 4 := 
by 
  sorry -- Proof to be filled in


end stork_count_l191_191972


namespace solution_set_transformation_l191_191521

variables (a b c α β : ℝ) (h_root : (α : ℝ) > 0)

open Set

def quadratic_inequality (x : ℝ) : Prop :=
  a * x^2 + b * x + c > 0

def transformed_inequality (x : ℝ) : Prop :=
  c * x^2 + b * x + a < 0

theorem solution_set_transformation :
  (∀ x, quadratic_inequality a b c x ↔ (α < x ∧ x < β)) →
  (∃ α β : ℝ, α > 0 ∧ (∀ x, transformed_inequality c b a x ↔ (x < 1/β ∨ x > 1/α))) :=
by
  sorry

end solution_set_transformation_l191_191521


namespace min_AC_plus_BD_l191_191977

theorem min_AC_plus_BD (k : ℝ) (h : k ≠ 0) :
  (8 + 8 / k^2) + (8 + 2 * k^2) ≥ 24 :=
by
  sorry -- skipping the proof

end min_AC_plus_BD_l191_191977


namespace find_length_y_l191_191826

def length_y (AO OC DO BO BD y : ℝ) : Prop := 
  AO = 3 ∧ OC = 11 ∧ DO = 3 ∧ BO = 6 ∧ BD = 7 ∧ y = 3 * Real.sqrt 91

theorem find_length_y : length_y 3 11 3 6 7 (3 * Real.sqrt 91) :=
by
  sorry

end find_length_y_l191_191826


namespace perimeter_of_cube_face_is_28_l191_191684

-- Define the volume of the cube
def volume_of_cube : ℝ := 343

-- Define the side length of the cube based on the volume
def side_length_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the perimeter of one face of the cube
def perimeter_of_one_face (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: Prove the perimeter of one face of the cube is 28 cm given the volume is 343 cm³
theorem perimeter_of_cube_face_is_28 : 
  perimeter_of_one_face side_length_of_cube = 28 := 
by
  sorry

end perimeter_of_cube_face_is_28_l191_191684


namespace saved_percent_correct_l191_191008

noncomputable def price_kit : ℝ := 144.20
noncomputable def price1 : ℝ := 21.75
noncomputable def price2 : ℝ := 18.60
noncomputable def price3 : ℝ := 23.80
noncomputable def price4 : ℝ := 29.35

noncomputable def total_price_individual : ℝ := 2 * price1 + 2 * price2 + price3 + 2 * price4
noncomputable def amount_saved : ℝ := total_price_individual - price_kit
noncomputable def percent_saved : ℝ := 100 * (amount_saved / total_price_individual)

theorem saved_percent_correct : percent_saved = 11.64 := by
  sorry

end saved_percent_correct_l191_191008


namespace find_k_l191_191502

theorem find_k (k : ℝ) : 2 + (2 + k) / 3 + (2 + 2 * k) / 3^2 + (2 + 3 * k) / 3^3 + 
  ∑' (n : ℕ), (2 + (n + 1) * k) / 3^(n + 1) = 7 ↔ k = 16 / 3 := 
sorry

end find_k_l191_191502


namespace simplify_and_evaluate_l191_191182

theorem simplify_and_evaluate (a b : ℝ) (h_eqn : a^2 + b^2 - 2 * a + 4 * b = -5) :
  (a - 2 * b) * (a^2 + 2 * a * b + 4 * b^2) - a * (a - 5 * b) * (a + 3 * b) = 120 :=
sorry

end simplify_and_evaluate_l191_191182


namespace solve_for_x_l191_191010

theorem solve_for_x (x : ℤ) : 3 * (5 - x) = 9 → x = 2 :=
by {
  sorry
}

end solve_for_x_l191_191010


namespace tagged_fish_in_second_catch_l191_191441

theorem tagged_fish_in_second_catch :
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  (total_tagged / N) * total_caught = 5 :=
by
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  show (total_tagged / N) * total_caught = 5
  sorry

end tagged_fish_in_second_catch_l191_191441


namespace number_of_pupils_l191_191696

theorem number_of_pupils (n : ℕ) 
  (h1 : 83 - 63 = 20) 
  (h2 : (20 : ℝ) / n = 1 / 2) : 
  n = 40 := 
sorry

end number_of_pupils_l191_191696


namespace value_of_m_l191_191726

theorem value_of_m (m : ℤ) (h₁ : |m| = 2) (h₂ : m ≠ 2) : m = -2 :=
by
  sorry

end value_of_m_l191_191726


namespace integer_solutions_to_equation_l191_191971

theorem integer_solutions_to_equation :
  ∃ (x y : ℤ), 2 * x^2 + 8 * y^2 = 17 * x * y - 423 ∧
               ((x = 11 ∧ y = 19) ∨ (x = -11 ∧ y = -19)) :=
by
  sorry

end integer_solutions_to_equation_l191_191971


namespace customer_paid_amount_l191_191279

def cost_price : Real := 7239.13
def percentage_increase : Real := 0.15
def selling_price := (1 + percentage_increase) * cost_price

theorem customer_paid_amount :
  selling_price = 8325.00 :=
by
  sorry

end customer_paid_amount_l191_191279


namespace simplify_and_evaluate_l191_191264

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  (1 / (x - 1) - 2 / (x ^ 2 - 1)) = -1 := by
  sorry

end simplify_and_evaluate_l191_191264


namespace sugar_water_inequality_acute_triangle_inequality_l191_191593

-- Part 1: Proving the inequality \(\frac{a}{b} < \frac{a+m}{b+m}\)
theorem sugar_water_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) :=
by
  sorry

-- Part 2: Proving the inequality in an acute triangle \(\triangle ABC\)
theorem acute_triangle_inequality (A B C : ℝ) (hA : A < B + C) (hB : B < C + A) (hC : C < A + B) : 
  (A / (B + C)) + (B / (C + A)) + (C / (A + B)) < 2 :=
by
  sorry

end sugar_water_inequality_acute_triangle_inequality_l191_191593


namespace g_self_inverse_if_one_l191_191444

variables (f : ℝ → ℝ) (symm_about : ∀ x, f (f x) = x - 1)

def g (b : ℝ) (x : ℝ) : ℝ := f (x + b)

theorem g_self_inverse_if_one (b : ℝ) :
  (∀ x, g f b (g f b x) = x) ↔ b = 1 := 
by
  sorry

end g_self_inverse_if_one_l191_191444


namespace min_area_monochromatic_triangle_l191_191742

-- Definition of the integer lattice in the plane.
def lattice_points : Set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) }

-- The 3-coloring condition
def coloring (c : (ℤ × ℤ) → Fin 3) := ∀ p : (ℤ × ℤ), p ∈ lattice_points → (c p) < 3

-- Definition of the area of a triangle
def triangle_area (A B C : ℤ × ℤ) : ℝ :=
  0.5 * abs (((B.1 - A.1) * (C.2 - A.2)) - ((C.1 - A.1) * (B.2 - A.2)))

-- The statement we need to prove
theorem min_area_monochromatic_triangle :
  ∃ S : ℝ, S = 3 ∧ ∀ (c : (ℤ × ℤ) → Fin 3), coloring c → ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ (c A = c B ∧ c B = c C) ∧ triangle_area A B C = S :=
sorry

end min_area_monochromatic_triangle_l191_191742


namespace ethan_coconut_oil_per_candle_l191_191247

noncomputable def ounces_of_coconut_oil_per_candle (candles: ℕ) (total_weight: ℝ) (beeswax_per_candle: ℝ) : ℝ :=
(total_weight - candles * beeswax_per_candle) / candles

theorem ethan_coconut_oil_per_candle :
  ounces_of_coconut_oil_per_candle 7 63 8 = 1 :=
by
  sorry

end ethan_coconut_oil_per_candle_l191_191247


namespace find_price_of_stock_A_l191_191051

-- Define conditions
def stock_investment_A (price_A : ℝ) : Prop := 
  ∃ (income_A: ℝ), income_A = 0.10 * 100

def stock_investment_B (price_B : ℝ) (investment_B : ℝ) : Prop := 
  price_B = 115.2 ∧ investment_B = 10 / 0.12

-- The main goal statement
theorem find_price_of_stock_A 
  (price_A : ℝ) (investment_B : ℝ) 
  (hA : stock_investment_A price_A) 
  (hB : stock_investment_B price_A investment_B) :
  price_A = 138.24 := 
sorry

end find_price_of_stock_A_l191_191051


namespace fraction_division_l191_191368

theorem fraction_division :
  (3 / 4) / (5 / 6) = 9 / 10 :=
by {
  -- We skip the proof as per the instructions
  sorry
}

end fraction_division_l191_191368


namespace student_distribution_l191_191902

-- Definition to check the number of ways to distribute 7 students into two dormitories A and B
-- with each dormitory having at least 2 students equals 56.
theorem student_distribution (students dorms : Nat) (min_students : Nat) (dist_plans : Nat) :
  students = 7 → dorms = 2 → min_students = 2 → dist_plans = 56 → 
  true := sorry

end student_distribution_l191_191902


namespace no_tetrahedron_with_given_heights_l191_191939

theorem no_tetrahedron_with_given_heights (h1 h2 h3 h4 : ℝ) (V : ℝ) (V_pos : V > 0)
    (S1 : ℝ := 3*V) (S2 : ℝ := (3/2)*V) (S3 : ℝ := V) (S4 : ℝ := V/2) :
    (h1 = 1) → (h2 = 2) → (h3 = 3) → (h4 = 6) → ¬ ∃ (S1 S2 S3 S4 : ℝ), S1 < S2 + S3 + S4 := by
  intros
  sorry

end no_tetrahedron_with_given_heights_l191_191939


namespace range_of_t_l191_191566

theorem range_of_t (x y a t : ℝ) 
  (h1 : x + 3 * y + a = 4) 
  (h2 : x - y - 3 * a = 0) 
  (h3 : -1 ≤ a ∧ a ≤ 1) 
  (h4 : t = x + y) : 
  1 ≤ t ∧ t ≤ 3 := 
sorry

end range_of_t_l191_191566


namespace evaluate_expression_l191_191127

noncomputable def greatest_integer (x : Real) : Int := ⌊x⌋

theorem evaluate_expression (y : Real) (h : y = 7.2) :
  greatest_integer 6.5 * greatest_integer (2 / 3)
  + greatest_integer 2 * y
  + greatest_integer 8.4 - 6.0 = 16.4 := by
  simp [greatest_integer, h]
  sorry

end evaluate_expression_l191_191127


namespace find_asymptote_slope_l191_191614

theorem find_asymptote_slope (x y : ℝ) (h : (y^2) / 9 - (x^2) / 4 = 1) : y = 3 / 2 * x :=
sorry

end find_asymptote_slope_l191_191614


namespace functional_eq_solution_l191_191118

open Real

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c := 
sorry

end functional_eq_solution_l191_191118


namespace line_intersects_parabola_once_l191_191689

theorem line_intersects_parabola_once (k : ℝ) :
  (x = k)
  ∧ (x = -3 * y^2 - 4 * y + 7)
  ∧ (3 * y^2 + 4 * y + (k - 7)) = 0
  ∧ ((4)^2 - 4 * 3 * (k - 7) = 0)
  → k = 25 / 3 := 
by
  sorry

end line_intersects_parabola_once_l191_191689


namespace circle_diameter_length_l191_191728

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l191_191728


namespace total_expenditure_l191_191840

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end total_expenditure_l191_191840


namespace sum_of_squares_of_roots_eq_213_l191_191920

theorem sum_of_squares_of_roots_eq_213
  {a b : ℝ}
  (h1 : a + b = 15)
  (h2 : a * b = 6) :
  a^2 + b^2 = 213 :=
by
  sorry

end sum_of_squares_of_roots_eq_213_l191_191920


namespace distance_between_foci_of_ellipse_l191_191215

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 3
  2 * Real.sqrt (a^2 - b^2) = 8 := by
  let a := 5
  let b := 3
  sorry

end distance_between_foci_of_ellipse_l191_191215


namespace divides_equiv_l191_191065

theorem divides_equiv (m n : ℤ) : 
  (17 ∣ (2 * m + 3 * n)) ↔ (17 ∣ (9 * m + 5 * n)) :=
by
  sorry

end divides_equiv_l191_191065


namespace average_weight_of_whole_class_l191_191572

theorem average_weight_of_whole_class :
  ∀ (n_a n_b : ℕ) (w_avg_a w_avg_b : ℝ),
    n_a = 60 →
    n_b = 70 →
    w_avg_a = 60 →
    w_avg_b = 80 →
    (n_a * w_avg_a + n_b * w_avg_b) / (n_a + n_b) = 70.77 :=
by
  intros n_a n_b w_avg_a w_avg_b h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_weight_of_whole_class_l191_191572


namespace gcd_product_eq_gcd_l191_191133

theorem gcd_product_eq_gcd {a b c : ℤ} (hab : Int.gcd a b = 1) : Int.gcd a (b * c) = Int.gcd a c := 
by 
  sorry

end gcd_product_eq_gcd_l191_191133


namespace trip_to_market_distance_l191_191317

theorem trip_to_market_distance 
  (school_trip_one_way : ℝ) (school_days_per_week : ℕ) 
  (weekly_total_mileage : ℝ) (round_trips_per_day : ℕ) (market_trip_count : ℕ) :
  (school_trip_one_way = 2.5) →
  (school_days_per_week = 4) →
  (round_trips_per_day = 2) →
  (weekly_total_mileage = 44) →
  (market_trip_count = 1) →
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  market_trip_distance = 2 :=
by
  intros h1 h2 h3 h4 h5
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  sorry

end trip_to_market_distance_l191_191317


namespace youngest_brother_age_l191_191333

theorem youngest_brother_age 
  (Rick_age : ℕ)
  (oldest_brother_age : ℕ)
  (middle_brother_age : ℕ)
  (smallest_brother_age : ℕ)
  (youngest_brother_age : ℕ)
  (h1 : Rick_age = 15)
  (h2 : oldest_brother_age = 2 * Rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2) :
  youngest_brother_age = 3 := 
sorry

end youngest_brother_age_l191_191333


namespace right_triangle_sides_l191_191262

theorem right_triangle_sides (r R : ℝ) (a b c : ℝ) 
    (r_eq : r = 8)
    (R_eq : R = 41)
    (right_angle : a^2 + b^2 = c^2)
    (inradius : 2*r = a + b - c)
    (circumradius : 2*R = c) :
    (a = 18 ∧ b = 80 ∧ c = 82) ∨ (a = 80 ∧ b = 18 ∧ c = 82) :=
by
  sorry

end right_triangle_sides_l191_191262


namespace data_instances_in_one_hour_l191_191512

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l191_191512


namespace turnip_total_correct_l191_191958

def turnips_left (melanie benny sarah david m_sold d_sold : ℕ) : ℕ :=
  let melanie_left := melanie - m_sold
  let david_left := david - d_sold
  benny + sarah + melanie_left + david_left

theorem turnip_total_correct :
  turnips_left 139 113 195 87 32 15 = 487 :=
by
  sorry

end turnip_total_correct_l191_191958


namespace result_more_than_half_l191_191313

theorem result_more_than_half (x : ℕ) (h : x = 4) : (2 * x + 5) - (x / 2) = 11 := by
  sorry

end result_more_than_half_l191_191313


namespace solution_set_f_x_minus_2_ge_zero_l191_191562

-- Define the necessary conditions and prove the statement
noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_minus_2_ge_zero (f_even : ∀ x, f x = f (-x))
  (f_mono : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_zero : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x | x ≥ 3 ∨ x ≤ 1} :=
by {
  sorry
}

end solution_set_f_x_minus_2_ge_zero_l191_191562


namespace replace_all_cardio_machines_cost_l191_191524

noncomputable def totalReplacementCost : ℕ :=
  let numGyms := 20
  let bikesPerGym := 10
  let treadmillsPerGym := 5
  let ellipticalsPerGym := 5
  let costPerBike := 700
  let costPerTreadmill := costPerBike * 3 / 2
  let costPerElliptical := costPerTreadmill * 2
  let totalBikes := numGyms * bikesPerGym
  let totalTreadmills := numGyms * treadmillsPerGym
  let totalEllipticals := numGyms * ellipticalsPerGym
  (totalBikes * costPerBike) + (totalTreadmills * costPerTreadmill) + (totalEllipticals * costPerElliptical)

theorem replace_all_cardio_machines_cost :
  totalReplacementCost = 455000 :=
by
  -- All the calculation steps provided as conditions and intermediary results need to be verified here.
  sorry

end replace_all_cardio_machines_cost_l191_191524


namespace john_newspapers_l191_191316

theorem john_newspapers (N : ℕ) (selling_price buying_price total_cost total_revenue : ℝ) 
  (h1 : selling_price = 2)
  (h2 : buying_price = 0.25 * selling_price)
  (h3 : total_cost = N * buying_price)
  (h4 : total_revenue = 0.8 * N * selling_price)
  (h5 : total_revenue - total_cost = 550) :
  N = 500 := 
by 
  -- actual proof here
  sorry

end john_newspapers_l191_191316


namespace vanessa_score_record_l191_191755

theorem vanessa_score_record 
  (team_total_points : ℕ) 
  (other_players_average : ℕ) 
  (num_other_players : ℕ) 
  (total_game_points : team_total_points = 55) 
  (average_points_per_player : other_players_average = 4) 
  (number_of_other_players : num_other_players = 7) 
  : 
  ∃ vanessa_points : ℕ, vanessa_points = 27 :=
by
  sorry

end vanessa_score_record_l191_191755


namespace log_product_l191_191167

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_product (x y : ℝ) (hx : 0 < x) (hy : 1 < y) :
  log_base (y^3) x * log_base (x^4) (y^3) * log_base (y^5) (x^2) * log_base (x^2) (y^5) * log_base (y^3) (x^4) =
  (1/3) * log_base y x :=
by
  sorry

end log_product_l191_191167


namespace shelves_used_l191_191047

def initial_books : ℕ := 86
def books_sold : ℕ := 37
def books_per_shelf : ℕ := 7
def remaining_books : ℕ := initial_books - books_sold
def shelves : ℕ := remaining_books / books_per_shelf

theorem shelves_used : shelves = 7 := by
  -- proof will go here
  sorry

end shelves_used_l191_191047


namespace two_digit_number_tens_place_l191_191178

theorem two_digit_number_tens_place (x y : Nat) (hx1 : 0 ≤ x) (hx2 : x ≤ 9) (hy1 : 0 ≤ y) (hy2 : y ≤ 9)
    (h : (x + y) * 3 = 10 * x + y - 2) : x = 2 := 
sorry

end two_digit_number_tens_place_l191_191178


namespace distance_traveled_l191_191898

theorem distance_traveled (speed1 speed2 hours1 hours2 : ℝ)
  (h1 : speed1 = 45) (h2 : hours1 = 2) (h3 : speed2 = 50) (h4 : hours2 = 3) :
  speed1 * hours1 + speed2 * hours2 = 240 := by
  sorry

end distance_traveled_l191_191898


namespace ada_originally_in_seat2_l191_191766

inductive Seat
| S1 | S2 | S3 | S4 | S5 deriving Inhabited, DecidableEq

def moveRight : Seat → Option Seat
| Seat.S1 => some Seat.S2
| Seat.S2 => some Seat.S3
| Seat.S3 => some Seat.S4
| Seat.S4 => some Seat.S5
| Seat.S5 => none

def moveLeft : Seat → Option Seat
| Seat.S1 => none
| Seat.S2 => some Seat.S1
| Seat.S3 => some Seat.S2
| Seat.S4 => some Seat.S3
| Seat.S5 => some Seat.S4

structure FriendState :=
  (bea ceci dee edie : Seat)
  (ada_left : Bool) -- Ada is away for snacks, identified by her not being in the seat row.

def initial_seating := FriendState.mk Seat.S2 Seat.S3 Seat.S4 Seat.S5 true

def final_seating (init : FriendState) : FriendState :=
  let bea' := match moveRight init.bea with
              | some pos => pos
              | none => init.bea
  let ceci' := init.ceci -- Ceci moves left then back, net zero movement
  let (dee', edie') := match moveRight init.dee, init.dee with
                      | some new_ee, ed => (new_ee, ed) -- Dee and Edie switch and Edie moves right
                      | _, _ => (init.dee, init.edie) -- If moves are invalid
  FriendState.mk bea' ceci' dee' edie' init.ada_left

theorem ada_originally_in_seat2 (init : FriendState) : init = initial_seating → final_seating init ≠ initial_seating → init.bea = Seat.S2 :=
by
  intro h_init h_finalne
  sorry -- Proof steps go here

end ada_originally_in_seat2_l191_191766


namespace quadratic_roots_eccentricities_l191_191244

theorem quadratic_roots_eccentricities :
  (∃ x y : ℝ, 3 * x^2 - 4 * x + 1 = 0 ∧ 3 * y^2 - 4 * y + 1 = 0 ∧ 
              (0 ≤ x ∧ x < 1) ∧ y = 1) :=
by
  -- Proof would go here
  sorry

end quadratic_roots_eccentricities_l191_191244


namespace greatest_p_meets_conditions_l191_191624

-- Define a four-digit number and its reversal being divisible by 63 and another condition of divisibility
def is_divisible_by (n m : ℕ) : Prop :=
  m % n = 0

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ a d => a * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def p := 9507

-- The main theorem we aim to prove.
theorem greatest_p_meets_conditions (p q : ℕ) 
  (h1 : is_four_digit p) 
  (h2 : is_four_digit q) 
  (h3 : reverse_digits p = q) 
  (h4 : is_divisible_by 63 p) 
  (h5 : is_divisible_by 63 q) 
  (h6 : is_divisible_by 9 p) : 
  p = 9507 :=
sorry

end greatest_p_meets_conditions_l191_191624


namespace total_fertilizer_usage_l191_191554

theorem total_fertilizer_usage :
  let daily_A : ℝ := 3 / 12
  let daily_B : ℝ := 4 / 10
  let daily_C : ℝ := 5 / 8
  let final_A : ℝ := daily_A + 6
  let final_B : ℝ := daily_B + 5
  let final_C : ℝ := daily_C + 7
  (final_A + final_B + final_C) = 19.275 := by
  sorry

end total_fertilizer_usage_l191_191554


namespace time_to_cross_tree_l191_191752

def train_length : ℕ := 600
def platform_length : ℕ := 450
def time_to_pass_platform : ℕ := 105

-- Definition of the condition that leads to the speed of the train
def speed_of_train : ℚ := (train_length + platform_length) / time_to_pass_platform

-- Statement to prove the time to cross the tree
theorem time_to_cross_tree :
  (train_length : ℚ) / speed_of_train = 60 :=
by
  sorry

end time_to_cross_tree_l191_191752


namespace sum_of_other_endpoint_coordinates_l191_191932

theorem sum_of_other_endpoint_coordinates {x y : ℝ} :
  let P1 := (1, 2)
  let M := (5, 6)
  let P2 := (x, y)
  (M.1 = (P1.1 + P2.1) / 2 ∧ M.2 = (P1.2 + P2.2) / 2) → (x + y) = 19 :=
by
  intros P1 M P2 h
  sorry

end sum_of_other_endpoint_coordinates_l191_191932


namespace prove_a_star_b_l191_191469

variable (a b : ℤ)
variable (h1 : a + b = 12)
variable (h2 : a * b = 35)

def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem prove_a_star_b : star a b = 12 / 35 :=
by
  sorry

end prove_a_star_b_l191_191469


namespace fifth_equation_l191_191792

theorem fifth_equation :
  (5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81) := 
by sorry

end fifth_equation_l191_191792


namespace problem1_problem2_l191_191699

-- Problem 1
theorem problem1 (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := 
by {
  sorry
}

-- Problem 2
theorem problem2 (a b m n s : ℤ) (h1 : a + b = 0) (h2 : m * n = 1) (h3 : |s| = 3) :
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := 
by {
  sorry
}

end problem1_problem2_l191_191699


namespace find_number_satisfying_condition_l191_191513

-- Define the condition where fifteen percent of x equals 150
def fifteen_percent_eq (x : ℝ) : Prop :=
  (15 / 100) * x = 150

-- Statement to prove the existence of a number x that satisfies the condition, and this x equals 1000
theorem find_number_satisfying_condition : ∃ x : ℝ, fifteen_percent_eq x ∧ x = 1000 :=
by
  -- Proof will be added here
  sorry

end find_number_satisfying_condition_l191_191513


namespace inner_square_area_l191_191408

theorem inner_square_area (side_ABCD : ℝ) (dist_BI : ℝ) (area_IJKL : ℝ) :
  side_ABCD = Real.sqrt 72 →
  dist_BI = 2 →
  area_IJKL = 39 :=
by
  sorry

end inner_square_area_l191_191408


namespace sam_investment_l191_191671

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_investment :
  compound_interest 3000 0.10 4 1 = 3311.44 :=
by
  sorry

end sam_investment_l191_191671


namespace evaluate_expr_l191_191375

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

theorem evaluate_expr : 3 * g 2 + 2 * g (-4) = 169 :=
by
  sorry

end evaluate_expr_l191_191375


namespace new_fig_sides_l191_191817

def hexagon_side := 1
def triangle_side := 1
def hexagon_sides := 6
def triangle_sides := 3
def joined_sides := 2
def total_initial_sides := hexagon_sides + triangle_sides
def lost_sides := joined_sides * 2
def new_shape_sides := total_initial_sides - lost_sides

theorem new_fig_sides : new_shape_sides = 5 := by
  sorry

end new_fig_sides_l191_191817


namespace arithmetic_sum_ratio_l191_191438

variable (a_n : ℕ → ℤ) -- the arithmetic sequence
variable (S : ℕ → ℤ) -- sum of the first n terms of the sequence
variable (d : ℤ) (a₁ : ℤ) -- common difference and first term of the sequence

-- Definition of the sum of the first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given condition
axiom h1 : (S 6) / (S 3) = 3

-- Definition of S_n in terms of the given formula
axiom S_def : ∀ n, S n = arithmetic_sum n

-- The main goal to prove
theorem arithmetic_sum_ratio : S 12 / S 9 = 5 / 3 := by
  sorry

end arithmetic_sum_ratio_l191_191438


namespace necessary_but_not_sufficient_condition_l191_191574

variable (p q : Prop)

theorem necessary_but_not_sufficient_condition (h : ¬p) : p ∨ q ↔ true :=
by
  sorry

end necessary_but_not_sufficient_condition_l191_191574


namespace triangle_side_lengths_l191_191995

theorem triangle_side_lengths (a b c r : ℕ) (h : a / b / c = 25 / 29 / 36) (hinradius : r = 232) :
  (a = 725 ∧ b = 841 ∧ c = 1044) :=
by
  sorry

end triangle_side_lengths_l191_191995


namespace mass_percentage_correct_l191_191522

noncomputable def mass_percentage_C_H_N_O_in_C20H25N3O 
  (m_C : ℚ) (m_H : ℚ) (m_N : ℚ) (m_O : ℚ) 
  (atoms_C : ℚ) (atoms_H : ℚ) (atoms_N : ℚ) (atoms_O : ℚ)
  (total_mass : ℚ)
  (percentage_C : ℚ) (percentage_H : ℚ) (percentage_N : ℚ) (percentage_O : ℚ) :=
  atoms_C = 20 ∧ atoms_H = 25 ∧ atoms_N = 3 ∧ atoms_O = 1 ∧ 
  m_C = 12.01 ∧ m_H = 1.008 ∧ m_N = 14.01 ∧ m_O = 16 ∧ 
  total_mass = (atoms_C * m_C) + (atoms_H * m_H) + (atoms_N * m_N) + (atoms_O * m_O) ∧ 
  percentage_C = (atoms_C * m_C / total_mass) * 100 ∧ 
  percentage_H = (atoms_H * m_H / total_mass) * 100 ∧ 
  percentage_N = (atoms_N * m_N / total_mass) * 100 ∧ 
  percentage_O = (atoms_O * m_O / total_mass) * 100 

theorem mass_percentage_correct : 
  mass_percentage_C_H_N_O_in_C20H25N3O 12.01 1.008 14.01 16 20 25 3 1 323.43 74.27 7.79 12.99 4.95 :=
by {
  sorry
}

end mass_percentage_correct_l191_191522


namespace angle_C_value_sides_a_b_l191_191414

variables (A B C : ℝ) (a b c : ℝ)

-- First part: Proving the value of angle C
theorem angle_C_value
  (h1 : 2*Real.cos (A/2)^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1)
  : C = Real.pi / 3 :=
sorry

-- Second part: Proving the values of a and b given c and the area
theorem sides_a_b
  (c : ℝ)
  (h2 : c = 2)
  (h3 : C = Real.pi / 3)
  (area : ℝ)
  (h4 : area = Real.sqrt 3)
  (h5 : 1/2 * a * b * Real.sin C = Real.sqrt 3)
  : a = 2 ∧ b = 2 :=
sorry

end angle_C_value_sides_a_b_l191_191414


namespace length_of_bridge_l191_191988

noncomputable def speed_kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

def total_distance_covered (speed_mps : ℝ) (time_s : ℕ) : ℝ := speed_mps * time_s

def bridge_length (total_distance : ℝ) (train_length : ℝ) : ℝ := total_distance - train_length

theorem length_of_bridge (train_length : ℝ) (time_s : ℕ) (speed_kmh : ℕ) :
  bridge_length (total_distance_covered (speed_kmh_to_mps speed_kmh) time_s) train_length = 299.9 :=
by
  have speed_mps := speed_kmh_to_mps speed_kmh
  have total_distance := total_distance_covered speed_mps time_s
  have length_of_bridge := bridge_length total_distance train_length
  sorry

end length_of_bridge_l191_191988


namespace quadratic_geometric_sequence_root_l191_191528

theorem quadratic_geometric_sequence_root {a b c : ℝ} (r : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b = a * r) 
  (h3 : c = a * r^2)
  (h4 : a ≥ b) 
  (h5 : b ≥ c) 
  (h6 : c ≥ 0) 
  (h7 : (a * r)^2 - 4 * a * (a * r^2) = 0) : 
  -b / (2 * a) = -1 / 8 := 
sorry

end quadratic_geometric_sequence_root_l191_191528


namespace find_a_l191_191429

theorem find_a (a b c d : ℤ) 
  (h1 : d + 0 = 2)
  (h2 : c + 2 = 2)
  (h3 : b + 0 = 4)
  (h4 : a + 4 = 0) : 
  a = -4 := 
sorry

end find_a_l191_191429


namespace dot_product_property_l191_191561

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

variables (x_P y_P : ℝ) (F1 F2 : ℝ × ℝ)

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0) ∨ F = (-1, 0)

def radius_of_inscribed_circle (r : ℝ) : Prop :=
  r = 1 / 2

theorem dot_product_property (h1 : point_on_ellipse x_P y_P)
  (h2 : is_focus F1) (h3 : is_focus F2) (h4: radius_of_inscribed_circle (1/2)):
  (x_P^2 - 1 + y_P^2) = 9 / 4 :=
sorry

end dot_product_property_l191_191561


namespace seashells_total_l191_191428

theorem seashells_total (joan_seashells jessica_seashells : ℕ)
  (h_joan : joan_seashells = 6)
  (h_jessica : jessica_seashells = 8) :
  joan_seashells + jessica_seashells = 14 :=
by 
  sorry

end seashells_total_l191_191428


namespace min_value_x2_2xy_y2_l191_191674

theorem min_value_x2_2xy_y2 (x y : ℝ) : ∃ (a b : ℝ), (x = a ∧ y = b) → x^2 + 2*x*y + y^2 = 0 :=
by {
  sorry
}

end min_value_x2_2xy_y2_l191_191674


namespace angle_C_max_l191_191278

theorem angle_C_max (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_cond : Real.sin B / Real.sin A = 2 * Real.cos (A + B))
  (h_max_B : B = Real.pi / 3) :
  C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_max_l191_191278


namespace remainder_of_four_m_plus_five_l191_191363

theorem remainder_of_four_m_plus_five (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 :=
by
  -- Proof steps would go here
  sorry

end remainder_of_four_m_plus_five_l191_191363


namespace area_of_triangle_intercepts_l191_191889

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts_l191_191889


namespace percentage_income_diff_l191_191404

variable (A B : ℝ)

-- Condition that B's income is 33.33333333333333% greater than A's income
def income_relation (A B : ℝ) : Prop :=
  B = (4 / 3) * A

-- Proof statement to show that A's income is 25% less than B's income
theorem percentage_income_diff : 
  income_relation A B → 
  ((B - A) / B) * 100 = 25 :=
by
  intros h
  rw [income_relation] at h
  sorry

end percentage_income_diff_l191_191404


namespace machine_working_time_l191_191164

def shirts_per_minute : ℕ := 3
def total_shirts_made : ℕ := 6

theorem machine_working_time : 
  (total_shirts_made / shirts_per_minute) = 2 :=
by
  sorry

end machine_working_time_l191_191164


namespace marcy_minimum_avg_score_l191_191880

variables (s1 s2 s3 : ℝ)
variable (qualified_avg : ℝ := 90)
variable (required_total : ℝ := 5 * qualified_avg)
variable (first_three_total : ℝ := s1 + s2 + s3)
variable (needed_points : ℝ := required_total - first_three_total)
variable (required_avg : ℝ := needed_points / 2)

/-- The admission criteria for a mathematics contest require a contestant to 
    achieve an average score of at least 90% over five rounds to qualify for the final round.
    Marcy scores 87%, 92%, and 85% in the first three rounds. 
    Prove that Marcy must average at least 93% in the next two rounds to qualify for the final. --/
theorem marcy_minimum_avg_score 
    (h1 : s1 = 87) (h2 : s2 = 92) (h3 : s3 = 85)
    : required_avg ≥ 93 :=
sorry

end marcy_minimum_avg_score_l191_191880


namespace geom_seq_a4_l191_191242

theorem geom_seq_a4 (a1 a2 a3 a4 r : ℝ)
  (h1 : a1 + a2 + a3 = 7)
  (h2 : a1 * a2 * a3 = 8)
  (h3 : a1 > 0)
  (h4 : r > 1)
  (h5 : a2 = a1 * r)
  (h6 : a3 = a1 * r^2)
  (h7 : a4 = a1 * r^3) : 
  a4 = 8 :=
sorry

end geom_seq_a4_l191_191242


namespace ratio_of_areas_l191_191303

theorem ratio_of_areas
  (PQ QR RP : ℝ)
  (PQ_pos : 0 < PQ)
  (QR_pos : 0 < QR)
  (RP_pos : 0 < RP)
  (s t u : ℝ)
  (s_pos : 0 < s)
  (t_pos : 0 < t)
  (u_pos : 0 < u)
  (h1 : s + t + u = 3 / 4)
  (h2 : s^2 + t^2 + u^2 = 1 / 2)
  : (1 - (s * (1 - u) + t * (1 - s) + u * (1 - t))) = 7 / 32 := by
  sorry

end ratio_of_areas_l191_191303


namespace rhombus_diagonal_length_l191_191205

theorem rhombus_diagonal_length (d1 d2 : ℝ) (Area : ℝ) 
  (h1 : d1 = 12) (h2 : Area = 60) 
  (h3 : Area = (d1 * d2) / 2) : d2 = 10 := 
by
  sorry

end rhombus_diagonal_length_l191_191205


namespace sequence_length_div_by_four_l191_191631

theorem sequence_length_div_by_four (a : ℕ) (h0 : a = 11664) (H : ∀ n, a = (4 ^ n) * b → b ≠ 0 ∧ n ≤ 3) : 
  ∃ n, n + 1 = 4 :=
by
  sorry

end sequence_length_div_by_four_l191_191631


namespace original_number_is_14_l191_191577

theorem original_number_is_14 (x : ℝ) (h : (2 * x + 2) / 3 = 10) : x = 14 := by
  sorry

end original_number_is_14_l191_191577


namespace find_largest_number_l191_191737

theorem find_largest_number (a b c d e : ℕ)
    (h1 : a + b + c + d = 240)
    (h2 : a + b + c + e = 260)
    (h3 : a + b + d + e = 280)
    (h4 : a + c + d + e = 300)
    (h5 : b + c + d + e = 320)
    (h6 : a + b = 40) :
    max a (max b (max c (max d e))) = 160 := by
  sorry

end find_largest_number_l191_191737


namespace sum_symmetry_l191_191111

def f (x : ℝ) : ℝ :=
  x^2 * (1 - x)^2

theorem sum_symmetry :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 :=
by
  sorry

end sum_symmetry_l191_191111


namespace only_book_A_l191_191388

variable (numA numB numBoth numOnlyB x : ℕ)
variable (h1 : numA = 2 * numB)
variable (h2 : numBoth = 500)
variable (h3 : numBoth = 2 * numOnlyB)
variable (h4 : numB = numOnlyB + numBoth)
variable (h5 : x = numA - numBoth)

theorem only_book_A : 
  x = 1000 := 
by
  sorry

end only_book_A_l191_191388


namespace word_count_in_language_l191_191149

theorem word_count_in_language :
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  num_words = 900 :=
by
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  have : num_words = 900 := sorry
  exact this

end word_count_in_language_l191_191149


namespace tan_neg_3pi_over_4_eq_one_l191_191834

theorem tan_neg_3pi_over_4_eq_one : Real.tan (-3 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_neg_3pi_over_4_eq_one_l191_191834


namespace can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l191_191797

-- Question and conditions
def side_length_of_square (A : ℝ) := A = 400
def area_of_rect (A : ℝ) := A = 300
def ratio_of_rect (length width : ℝ) := 3 * width = 2 * length

-- Prove that Li can cut a rectangle with area 300 from the square with area 400
theorem can_cut_rectangle_with_area_300 
  (a : ℝ) (h1 : side_length_of_square a)
  (length width : ℝ)
  (ha : a ^ 2 = 400) (har : length * width = 300) :
  length ≤ a ∧ width ≤ a :=
by
  sorry

-- Prove that Li cannot cut a rectangle with ratio 3:2 from the square
theorem cannot_cut_rectangle_with_ratio_3_2 (a : ℝ)
  (h1 : side_length_of_square a)
  (length width : ℝ)
  (har : area_of_rect (length * width))
  (hratio : ratio_of_rect length width)
  (ha : a ^ 2 = 400) :
  ¬(length ≤ a ∧ width ≤ a) :=
by
  sorry

end can_cut_rectangle_with_area_300_cannot_cut_rectangle_with_ratio_3_2_l191_191797


namespace population_increase_l191_191306

theorem population_increase (P : ℕ)
  (birth_rate1_per_1000 : ℕ := 25)
  (death_rate1_per_1000 : ℕ := 12)
  (immigration_rate1 : ℕ := 15000)
  (birth_rate2_per_1000 : ℕ := 30)
  (death_rate2_per_1000 : ℕ := 8)
  (immigration_rate2 : ℕ := 30000)
  (pop_increase1_perc : ℤ := 200)
  (pop_increase2_perc : ℤ := 300) :
  (12 * P - P) / P * 100 = 1100 := by
  sorry

end population_increase_l191_191306


namespace sum_f_neg_l191_191271

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_neg {x1 x2 x3 : ℝ}
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end sum_f_neg_l191_191271


namespace base7_to_base10_and_frac_l191_191519

theorem base7_to_base10_and_frac (c d e : ℕ) 
  (h1 : (761 : ℕ) = 7^2 * 7 + 6 * 7^1 + 1 * 7^0)
  (h2 : (10 * 10 * c + 10 * d + e) = 386)
  (h3 : c = 3)
  (h4 : d = 8)
  (h5 : e = 6) :
  (d * e) / 15 = 48 / 15 := 
sorry

end base7_to_base10_and_frac_l191_191519


namespace find_y_when_x_is_8_l191_191413

theorem find_y_when_x_is_8 : 
  ∃ k, (70 * 5 = k ∧ 8 * 25 = k) := 
by
  -- The proof will be filled in here
  sorry

end find_y_when_x_is_8_l191_191413


namespace find_d_for_single_point_l191_191370

/--
  Suppose that the graph of \(3x^2 + y^2 + 6x - 6y + d = 0\) consists of a single point.
  Prove that \(d = 12\).
-/
theorem find_d_for_single_point : 
  ∀ (d : ℝ), (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0) ∧
              (∀ (x1 y1 x2 y2 : ℝ), 
                (3 * x1^2 + y1^2 + 6 * x1 - 6 * y1 + d = 0 ∧ 
                 3 * x2^2 + y2^2 + 6 * x2 - 6 * y2 + d = 0 → 
                 x1 = x2 ∧ y1 = y2)) ↔ d = 12 := 
by 
  sorry

end find_d_for_single_point_l191_191370


namespace example_problem_l191_191358

theorem example_problem : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end example_problem_l191_191358


namespace no_real_root_for_3_in_g_l191_191590

noncomputable def g (x c : ℝ) : ℝ := x^2 + 3 * x + c

theorem no_real_root_for_3_in_g (c : ℝ) :
  (21 - 4 * c) < 0 ↔ c > 21 / 4 := by
sorry

end no_real_root_for_3_in_g_l191_191590


namespace emily_sixth_quiz_score_l191_191611

-- Define the scores Emily has received
def scores : List ℕ := [92, 96, 87, 89, 100]

-- Define the number of quizzes
def num_quizzes : ℕ := 6

-- Define the desired average score
def desired_average : ℕ := 94

-- The theorem to prove the score Emily needs on her sixth quiz to achieve the desired average
theorem emily_sixth_quiz_score : ∃ (x : ℕ), List.sum scores + x = desired_average * num_quizzes := by
  sorry

end emily_sixth_quiz_score_l191_191611


namespace discount_on_pony_jeans_l191_191543

theorem discount_on_pony_jeans 
  (F P : ℕ)
  (h1 : F + P = 25)
  (h2 : 5 * F + 4 * P = 100) : P = 25 :=
by
  sorry

end discount_on_pony_jeans_l191_191543


namespace rectangle_perimeter_l191_191384

theorem rectangle_perimeter :
  ∃ (a b : ℕ), (a ≠ b) ∧ (a * b = 2 * (a + b) - 4) ∧ (2 * (a + b) = 26) :=
by {
  sorry
}

end rectangle_perimeter_l191_191384


namespace net_profit_is_90_l191_191063

theorem net_profit_is_90
    (cost_seeds cost_soil : ℝ)
    (num_plants : ℕ)
    (price_per_plant : ℝ)
    (h0 : cost_seeds = 2)
    (h1 : cost_soil = 8)
    (h2 : num_plants = 20)
    (h3 : price_per_plant = 5) :
    (num_plants * price_per_plant - (cost_seeds + cost_soil)) = 90 := by
  sorry

end net_profit_is_90_l191_191063


namespace largest_negative_integer_is_neg_one_l191_191596

def is_negative_integer (n : Int) : Prop := n < 0

def is_largest_negative_integer (n : Int) : Prop := 
  is_negative_integer n ∧ ∀ m : Int, is_negative_integer m → m ≤ n

theorem largest_negative_integer_is_neg_one : 
  is_largest_negative_integer (-1) := by
  sorry

end largest_negative_integer_is_neg_one_l191_191596


namespace steve_speed_back_home_l191_191216

-- Define a structure to hold the given conditions:
structure Conditions where
  home_to_work_distance : Float := 35 -- km
  v  : Float -- speed on the way to work in km/h
  additional_stop_time : Float := 0.25 -- hours
  total_weekly_time : Float := 30 -- hours

-- Define the main proposition:
theorem steve_speed_back_home (c: Conditions)
  (h1 : 5 * ((c.home_to_work_distance / c.v) + (c.home_to_work_distance / (2 * c.v))) + 3 * c.additional_stop_time = c.total_weekly_time) :
  2 * c.v = 18 := by
  sorry

end steve_speed_back_home_l191_191216


namespace inverse_matrix_correct_l191_191688

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1, 2, 3],
    ![0, -1, 2],
    ![3, 0, 7]
  ]

def A_inv_correct : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-1/2, -1, 1/2],
    ![3/7, -1/7, -1/7],
    ![3/14, 3/7, -1/14]
  ]

theorem inverse_matrix_correct : A⁻¹ = A_inv_correct := by
  sorry

end inverse_matrix_correct_l191_191688


namespace radius_of_circle_l191_191929

theorem radius_of_circle
  (AC BD : ℝ) (h_perpendicular : AC * BD = 0)
  (h_intersect_center : AC / 2 = BD / 2)
  (AB : ℝ) (h_AB : AB = 3)
  (CD : ℝ) (h_CD : CD = 4) :
  (∃ R : ℝ, R = 5 / 2) :=
by
  sorry

end radius_of_circle_l191_191929


namespace pencil_pen_eraser_cost_l191_191669

-- Define the problem conditions and question
theorem pencil_pen_eraser_cost 
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 4.10)
  (h2 : 2 * p + 3 * q = 3.70) :
  p + q + 0.85 = 2.41 :=
sorry

end pencil_pen_eraser_cost_l191_191669


namespace miracle_tree_fruit_count_l191_191153

theorem miracle_tree_fruit_count :
  ∃ (apples oranges pears : ℕ), 
  apples + oranges + pears = 30 ∧
  apples = 6 ∧ oranges = 9 ∧ pears = 15 := by
  sorry

end miracle_tree_fruit_count_l191_191153


namespace limit_does_not_exist_l191_191078

noncomputable def does_not_exist_limit : Prop := 
  ¬ ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    (0 < |x| ∧ 0 < |y| ∧ |x| < δ ∧ |y| < δ) →
    |(x^2 - y^2) / (x^2 + y^2) - l| < ε

theorem limit_does_not_exist :
  does_not_exist_limit :=
sorry

end limit_does_not_exist_l191_191078


namespace exists_k_for_blocks_of_2022_l191_191592

theorem exists_k_for_blocks_of_2022 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ j, 
  k^i / 10^j % 10^4 = 2022)) :=
sorry

end exists_k_for_blocks_of_2022_l191_191592


namespace field_trip_total_l191_191503

-- Define the conditions
def vans := 2
def buses := 3
def people_per_van := 8
def people_per_bus := 20

-- The total number of people
def total_people := (vans * people_per_van) + (buses * people_per_bus)

theorem field_trip_total : total_people = 76 :=
by
  -- skip the proof here
  sorry

end field_trip_total_l191_191503


namespace charlie_acorns_l191_191606

theorem charlie_acorns (x y : ℕ) (hc hs : ℕ)
  (h5 : x = 5 * hc)
  (h7 : y = 7 * hs)
  (total : x + y = 145)
  (holes : hs = hc - 3) :
  x = 70 :=
by
  sorry

end charlie_acorns_l191_191606


namespace middle_number_consecutive_odd_sum_l191_191718

theorem middle_number_consecutive_odd_sum (n : ℤ)
  (h1 : n % 2 = 1) -- n is an odd number
  (h2 : n + (n + 2) + (n + 4) = n + 20) : 
  n + 2 = 9 :=
by
  sorry

end middle_number_consecutive_odd_sum_l191_191718


namespace total_balloons_sam_and_dan_l191_191342

noncomputable def sam_initial_balloons : ℝ := 46.0
noncomputable def balloons_given_to_fred : ℝ := 10.0
noncomputable def dan_balloons : ℝ := 16.0

theorem total_balloons_sam_and_dan :
  (sam_initial_balloons - balloons_given_to_fred) + dan_balloons = 52.0 := 
by 
  sorry

end total_balloons_sam_and_dan_l191_191342


namespace gasoline_price_increase_l191_191311

theorem gasoline_price_increase (highest_price lowest_price : ℝ) (h1 : highest_price = 24) (h2 : lowest_price = 15) : 
  ((highest_price - lowest_price) / lowest_price) * 100 = 60 :=
by
  sorry

end gasoline_price_increase_l191_191311


namespace range_of_k_l191_191987

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Define the condition that the function does not pass through the third quadrant
def does_not_pass_third_quadrant (k : ℝ) : Prop :=
  ∀ x : ℝ, (x < 0 ∧ linear_function k x < 0) → false

-- Theorem statement proving the range of k
theorem range_of_k (k : ℝ) : does_not_pass_third_quadrant k ↔ (0 ≤ k ∧ k < 2) :=
by
  sorry

end range_of_k_l191_191987


namespace remy_used_25_gallons_l191_191101

noncomputable def RomanGallons : ℕ := 8

noncomputable def RemyGallons (R : ℕ) : ℕ := 3 * R + 1

theorem remy_used_25_gallons (R : ℕ) (h1 : RemyGallons R = 1 + 3 * R) (h2 : R + RemyGallons R = 33) : RemyGallons R = 25 := by
  sorry

end remy_used_25_gallons_l191_191101


namespace intersection_of_lines_l191_191836

theorem intersection_of_lines :
  ∃ (x y : ℚ), 3 * y = -2 * x + 6 ∧ 2 * y = -7 * x - 2 ∧ x = -18 / 17 ∧ y = 46 / 17 :=
by
  sorry

end intersection_of_lines_l191_191836


namespace compute_R_at_3_l191_191982

def R (x : ℝ) := 3 * x ^ 4 + x ^ 3 + x ^ 2 + x + 1

theorem compute_R_at_3 : R 3 = 283 := by
  sorry

end compute_R_at_3_l191_191982


namespace minimum_area_l191_191527

-- Define point A
def A : ℝ × ℝ := (-4, 0)

-- Define point B
def B : ℝ × ℝ := (0, 4)

-- Define the circle
def on_circle (C : ℝ × ℝ) : Prop := (C.1 - 2)^2 + C.2^2 = 2

-- Instantiating the proof of the minimum area of △ABC = 8
theorem minimum_area (C : ℝ × ℝ) (h : on_circle C) : 
  ∃ C : ℝ × ℝ, on_circle C ∧ 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 8 := 
sorry

end minimum_area_l191_191527


namespace system_of_equations_solution_l191_191863

theorem system_of_equations_solution :
  ∃ x y : ℝ, (2 * x + y = 6) ∧ (x - y = 3) ∧ (x = 3) ∧ (y = 0) :=
by
  sorry

end system_of_equations_solution_l191_191863


namespace days_to_complete_job_l191_191440

theorem days_to_complete_job (m₁ m₂ d₁ d₂ total_man_days : ℝ)
    (h₁ : m₁ = 30)
    (h₂ : d₁ = 8)
    (h₃ : total_man_days = 240)
    (h₄ : total_man_days = m₁ * d₁)
    (h₅ : m₂ = 40) :
    d₂ = total_man_days / m₂ := by
  sorry

end days_to_complete_job_l191_191440


namespace single_elimination_games_l191_191083

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  ∃ g : ℕ, g = n - 1 ∧ g = 511 := 
by
  use n - 1
  sorry

end single_elimination_games_l191_191083


namespace count_C_sets_l191_191601

-- Definitions of sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 2}

-- The predicate that a set C satisfies B ∪ C = A
def satisfies_condition (C : Set ℕ) : Prop := B ∪ C = A

-- The claim that there are exactly 4 such sets C
theorem count_C_sets : 
  ∃ (C1 C2 C3 C4 : Set ℕ), 
    (satisfies_condition C1 ∧ satisfies_condition C2 ∧ satisfies_condition C3 ∧ satisfies_condition C4) 
    ∧ 
    (∀ C', satisfies_condition C' → C' = C1 ∨ C' = C2 ∨ C' = C3 ∨ C' = C4)
    ∧ 
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C1 ≠ C4 ∧ C2 ≠ C3 ∧ C2 ≠ C4 ∧ C3 ≠ C4) := 
sorry

end count_C_sets_l191_191601


namespace probability_leftmost_blue_off_rightmost_red_on_l191_191304

noncomputable def calculate_probability : ℚ :=
  let total_arrangements := Nat.choose 8 4
  let total_on_choices := Nat.choose 8 4
  let favorable_arrangements := Nat.choose 6 3 * Nat.choose 7 3
  favorable_arrangements / (total_arrangements * total_on_choices)

theorem probability_leftmost_blue_off_rightmost_red_on :
  calculate_probability = 1 / 7 := 
by
  sorry

end probability_leftmost_blue_off_rightmost_red_on_l191_191304


namespace combined_mpg_19_l191_191203

theorem combined_mpg_19 (m: ℕ) (h: m = 100) :
  let ray_car_mpg := 50
  let tom_car_mpg := 25
  let jerry_car_mpg := 10
  let ray_gas_used := m / ray_car_mpg
  let tom_gas_used := m / tom_car_mpg
  let jerry_gas_used := m / jerry_car_mpg
  let total_gas_used := ray_gas_used + tom_gas_used + jerry_gas_used
  let total_miles := 3 * m
  let combined_mpg := total_miles * 25 / (4 * m)
  combined_mpg = 19 := 
by {
  sorry
}

end combined_mpg_19_l191_191203


namespace divisibility_by_37_l191_191052

def sum_of_segments (n : ℕ) : ℕ :=
  let rec split_and_sum (num : ℕ) (acc : ℕ) : ℕ :=
    if num < 1000 then acc + num
    else split_and_sum (num / 1000) (acc + num % 1000)
  split_and_sum n 0

theorem divisibility_by_37 (A : ℕ) : 
  (37 ∣ A) ↔ (37 ∣ sum_of_segments A) :=
sorry

end divisibility_by_37_l191_191052


namespace sum_of_possible_values_l191_191630

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 24) : 
  ∃ x1 x2 : ℝ, (x1 + 3) * (x1 - 4) = 24 ∧ (x2 + 3) * (x2 - 4) = 24 ∧ x1 + x2 = 1 := 
by
  sorry

end sum_of_possible_values_l191_191630


namespace polygon_sides_l191_191761

theorem polygon_sides (h : ∀ (n : ℕ), (180 * (n - 2)) / n = 150) : n = 12 :=
by
  sorry

end polygon_sides_l191_191761


namespace wristband_distribution_l191_191315

open Nat 

theorem wristband_distribution (x y : ℕ) 
  (h1 : 2 * x + 2 * y = 460) 
  (h2 : 2 * x = 3 * y) : x = 138 :=
sorry

end wristband_distribution_l191_191315


namespace smallest_base10_integer_l191_191651

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l191_191651


namespace solve_q_l191_191496

-- Definitions of conditions
variable (p q : ℝ)
variable (k : ℝ) 

-- Initial conditions
axiom h1 : p = 1500
axiom h2 : q = 0.5
axiom h3 : p * q = k
axiom h4 : k = 750

-- Goal
theorem solve_q (hp : p = 3000) : q = 0.250 :=
by
  -- The proof is omitted.
  sorry

end solve_q_l191_191496


namespace point_on_circle_l191_191905

theorem point_on_circle (a b : ℝ) 
  (h1 : (b + 2) * x + a * y + 4 = 0) 
  (h2 : a * x + (2 - b) * y - 3 = 0) 
  (parallel_lines : ∀ x y : ℝ, ∀ C1 C2 : ℝ, 
    (b + 2) * x + a * y + C1 = 0 ∧ a * x + (2 - b) * y + C2 = 0 → 
    - (b + 2) / a = - a / (2 - b)
  ) : a^2 + b^2 = 4 :=
sorry

end point_on_circle_l191_191905


namespace factorize_l191_191904

theorem factorize (m : ℝ) : m^3 - 4 * m = m * (m + 2) * (m - 2) :=
by
  sorry

end factorize_l191_191904


namespace joan_dimes_l191_191132

theorem joan_dimes (initial_dimes spent_dimes remaining_dimes : ℕ) 
    (h1 : initial_dimes = 5) (h2 : spent_dimes = 2) 
    (h3 : remaining_dimes = initial_dimes - spent_dimes) : 
    remaining_dimes = 3 := 
sorry

end joan_dimes_l191_191132


namespace trivia_competition_points_l191_191896

theorem trivia_competition_points 
  (total_members : ℕ := 120) 
  (absent_members : ℕ := 37) 
  (points_per_member : ℕ := 24) : 
  (total_members - absent_members) * points_per_member = 1992 := 
by
  sorry

end trivia_competition_points_l191_191896


namespace tom_candies_left_is_ten_l191_191094

-- Define initial conditions
def initial_candies: ℕ := 2
def friend_gave_candies: ℕ := 7
def bought_candies: ℕ := 10

-- Define total candies before sharing
def total_candies := initial_candies + friend_gave_candies + bought_candies

-- Define the number of candies Tom gives to his sister
def candies_given := total_candies / 2

-- Define the number of candies Tom has left
def candies_left := total_candies - candies_given

-- Prove the final number of candies left
theorem tom_candies_left_is_ten : candies_left = 10 :=
by
  -- The proof is left as an exercise
  sorry

end tom_candies_left_is_ten_l191_191094


namespace sum_of_mixed_numbers_is_between_18_and_19_l191_191779

theorem sum_of_mixed_numbers_is_between_18_and_19 :
  let a := 2 + 3 / 8;
  let b := 4 + 1 / 3;
  let c := 5 + 2 / 21;
  let d := 6 + 1 / 11;
  18 < a + b + c + d ∧ a + b + c + d < 19 :=
by
  sorry

end sum_of_mixed_numbers_is_between_18_and_19_l191_191779


namespace total_tickets_sold_l191_191116

-- Define the parameters and conditions
def VIP_ticket_price : ℝ := 45.00
def general_ticket_price : ℝ := 20.00
def total_revenue : ℝ := 7500.00
def tickets_difference : ℕ := 276

-- Define the total number of tickets sold
def total_number_of_tickets (V G : ℕ) : ℕ := V + G

-- The theorem to be proved
theorem total_tickets_sold (V G : ℕ) 
  (h1 : VIP_ticket_price * V + general_ticket_price * G = total_revenue)
  (h2 : V = G - tickets_difference) : 
  total_number_of_tickets V G = 336 :=
by
  sorry

end total_tickets_sold_l191_191116


namespace tan_periodic_example_l191_191029

theorem tan_periodic_example : Real.tan (13 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_periodic_example_l191_191029


namespace num_males_selected_l191_191789

theorem num_males_selected (total_male total_female total_selected : ℕ)
                           (h_male : total_male = 56)
                           (h_female : total_female = 42)
                           (h_selected : total_selected = 28) :
  (total_male * total_selected) / (total_male + total_female) = 16 := 
by {
  sorry
}

end num_males_selected_l191_191789


namespace value_of_a_is_2_l191_191772

def point_symmetric_x_axis (a b : ℝ) : Prop :=
  (2 * a + b = 1 - 2 * b) ∧ (a - 2 * b = -(-2 * a - b - 1))

theorem value_of_a_is_2 (a b : ℝ) (h : point_symmetric_x_axis a b) : a = 2 :=
by sorry

end value_of_a_is_2_l191_191772


namespace men_entered_room_l191_191053

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l191_191053


namespace brown_house_number_l191_191677

-- Defining the problem conditions
def sum_arithmetic_series (k : ℕ) := k * (k + 1) / 2

theorem brown_house_number (t n : ℕ) (h1 : 20 < t) (h2 : t < 500)
    (h3 : sum_arithmetic_series n = sum_arithmetic_series t / 2) : n = 84 := by
  sorry

end brown_house_number_l191_191677


namespace part_a_l191_191735

theorem part_a (α β : ℝ) (h₁ : α = 1.0000000004) (h₂ : β = 1.00000000002) (h₃ : α > β) :
  2.00000000002 / (β * β + 2.00000000002) > 2.00000000004 / α := 
sorry

end part_a_l191_191735


namespace union_of_A_and_B_l191_191054

open Set

variable {x : ℝ}

-- Define sets A and B based on the given conditions
def A : Set ℝ := { x | 0 < 3 - x ∧ 3 - x ≤ 2 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end union_of_A_and_B_l191_191054


namespace at_least_one_lands_l191_191914

def p : Prop := sorry -- Proposition that Person A lands in the designated area
def q : Prop := sorry -- Proposition that Person B lands in the designated area

theorem at_least_one_lands : p ∨ q := sorry

end at_least_one_lands_l191_191914


namespace simplify_expr_l191_191423

theorem simplify_expr (a b x : ℝ) (h₁ : x = a^3 / b^3) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) := 
by 
  sorry

end simplify_expr_l191_191423


namespace rainfall_difference_l191_191798

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference_l191_191798


namespace megan_removed_albums_l191_191780

theorem megan_removed_albums :
  ∀ (albums_in_cart : ℕ) (songs_per_album : ℕ) (total_songs_bought : ℕ),
    albums_in_cart = 8 →
    songs_per_album = 7 →
    total_songs_bought = 42 →
    albums_in_cart - (total_songs_bought / songs_per_album) = 2 :=
by
  intros albums_in_cart songs_per_album total_songs_bought h1 h2 h3
  sorry

end megan_removed_albums_l191_191780


namespace repeating_decimal_fraction_l191_191545

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end repeating_decimal_fraction_l191_191545


namespace sum_remainder_l191_191254

theorem sum_remainder (a b c : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 14) (h3 : c % 53 = 9) : 
  (a + b + c) % 53 = 3 := 
by 
  sorry

end sum_remainder_l191_191254


namespace compound_interest_correct_l191_191474

variables (SI : ℚ) (R : ℚ) (T : ℕ) (P : ℚ)

def calculate_principal (SI R T : ℚ) : ℚ := SI * 100 / (R * T)

def calculate_compound_interest (P R : ℚ) (T : ℕ) : ℚ :=
  P * ((1 + R / 100)^T - 1)

theorem compound_interest_correct (h1: SI = 52) (h2: R = 5) (h3: T = 2) :
  calculate_compound_interest (calculate_principal SI R T) R T = 53.30 :=
by
  sorry

end compound_interest_correct_l191_191474


namespace min_cubes_needed_l191_191605

def minimum_cubes_for_views (front_view side_view : ℕ) : ℕ :=
  4

theorem min_cubes_needed (front_view_cond side_view_cond : ℕ) :
  front_view_cond = 2 ∧ side_view_cond = 3 → minimum_cubes_for_views front_view_cond side_view_cond = 4 :=
by
  intro h
  cases h
  -- Proving the condition based on provided views
  sorry

end min_cubes_needed_l191_191605


namespace jebb_total_spent_l191_191510

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end jebb_total_spent_l191_191510


namespace alia_markers_count_l191_191385

theorem alia_markers_count :
  ∀ (Alia Austin Steve Bella : ℕ),
  (Alia = 2 * Austin) →
  (Austin = (1 / 3) * Steve) →
  (Steve = 60) →
  (Bella = (3 / 2) * Alia) →
  Alia = 40 :=
by
  intros Alia Austin Steve Bella H1 H2 H3 H4
  sorry

end alia_markers_count_l191_191385


namespace employees_6_or_more_percentage_is_18_l191_191195

-- Defining the employee counts for different year ranges
def count_less_than_1 (y : ℕ) : ℕ := 4 * y
def count_1_to_2 (y : ℕ) : ℕ := 6 * y
def count_2_to_3 (y : ℕ) : ℕ := 7 * y
def count_3_to_4 (y : ℕ) : ℕ := 4 * y
def count_4_to_5 (y : ℕ) : ℕ := 3 * y
def count_5_to_6 (y : ℕ) : ℕ := 3 * y
def count_6_to_7 (y : ℕ) : ℕ := 2 * y
def count_7_to_8 (y : ℕ) : ℕ := 2 * y
def count_8_to_9 (y : ℕ) : ℕ := y
def count_9_to_10 (y : ℕ) : ℕ := y

-- Sum of all employees T
def total_employees (y : ℕ) : ℕ := count_less_than_1 y + count_1_to_2 y + count_2_to_3 y +
                                    count_3_to_4 y + count_4_to_5 y + count_5_to_6 y +
                                    count_6_to_7 y + count_7_to_8 y + count_8_to_9 y +
                                    count_9_to_10 y

-- Employees with 6 years or more E
def employees_6_or_more (y : ℕ) : ℕ := count_6_to_7 y + count_7_to_8 y + count_8_to_9 y + count_9_to_10 y

-- Calculate percentage
def percentage (y : ℕ) : ℚ := (employees_6_or_more y : ℚ) / (total_employees y : ℚ) * 100

-- Proving the final statement
theorem employees_6_or_more_percentage_is_18 (y : ℕ) (hy : y ≠ 0) : percentage y = 18 :=
by
  sorry

end employees_6_or_more_percentage_is_18_l191_191195


namespace max_non_managers_l191_191966

theorem max_non_managers (N : ℕ) : (8 / N : ℚ) > 7 / 32 → N ≤ 36 :=
by sorry

end max_non_managers_l191_191966


namespace division_of_fractions_l191_191459

theorem division_of_fractions : (5 / 6) / (1 + 3 / 9) = 5 / 8 := by
  sorry

end division_of_fractions_l191_191459


namespace fence_cost_l191_191769

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (side_length perimeter cost : ℝ) 
  (h1 : area = 289) 
  (h2 : price_per_foot = 55)
  (h3 : side_length = Real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : cost = perimeter * price_per_foot) :
  cost = 3740 := 
sorry

end fence_cost_l191_191769


namespace XiaoMing_team_award_l191_191950

def points (x : ℕ) : ℕ := 2 * x + (8 - x)

theorem XiaoMing_team_award (x : ℕ) : 2 * x + (8 - x) ≥ 12 := 
by 
  sorry

end XiaoMing_team_award_l191_191950


namespace number_of_8th_graders_l191_191858

variable (x y : ℕ)
variable (y_valid : 0 ≤ y)

theorem number_of_8th_graders (h : x * (x + 3 - 2 * y) = 14) :
  x = 7 :=
by 
  sorry

end number_of_8th_graders_l191_191858


namespace v_function_expression_f_max_value_l191_191181

noncomputable def v (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2
else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2)
else 0

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2 * x
else if 4 < x ∧ x ≤ 20 then - (1/8) * x^2 + (5/2) * x
else 0

theorem v_function_expression :
  ∀ x, 0 < x ∧ x ≤ 20 → 
  v x = (if 0 < x ∧ x ≤ 4 then 2 else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2) else 0) :=
by sorry

theorem f_max_value :
  ∃ x, 0 < x ∧ x ≤ 20 ∧ f x = 12.5 :=
by sorry

end v_function_expression_f_max_value_l191_191181


namespace problem_statement_l191_191708

theorem problem_statement : (-0.125 ^ 2006) * (8 ^ 2005) = -0.125 := by
  sorry

end problem_statement_l191_191708


namespace part1_part2_l191_191665

section
variable (x y : ℝ)

def A : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B : ℝ := y^2 - x * y + 2 * x^2

-- Part (1): Prove that 2A - 3B = y^2 - xy
theorem part1 : 2 * A x y - 3 * B x y = y^2 - x * y := 
sorry

-- Part (2): Given |2x - 3| + (y + 2)^2 = 0, prove that 2A - 3B = 7
theorem part2 (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 :=
sorry

end

end part1_part2_l191_191665


namespace trigonometric_identity_proof_l191_191732

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α)) / ((Real.cos α) ^ 2) = 6 :=
by
  sorry

end trigonometric_identity_proof_l191_191732


namespace fraction_to_decimal_l191_191866

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l191_191866


namespace problem_1_l191_191819

open Set

variable (R : Set ℝ)
variable (A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 })
variable (B : Set ℝ := { x | x^2 + a < 0 })

theorem problem_1 (a : ℝ) : (a = -4 → (A ∩ B = { x : ℝ | 1 / 2 ≤ x ∧ x < 2 } ∧ A ∪ B = { x : ℝ | -2 < x ∧ x ≤ 3 })) ∧
  ((compl A ∩ B = B) → a ≥ -2) := by
  sorry

end problem_1_l191_191819


namespace find_n_l191_191873

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  11 + (n - 1) * 6

-- State the problem
theorem find_n (n : ℕ) : 
  (∀ m : ℕ, m ≥ n → arithmetic_sequence m > 2017) ↔ n = 336 :=
by
  sorry

end find_n_l191_191873


namespace ab_cd_zero_l191_191531

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1)
  (h3 : a * c + b * d = 0) : 
  a * b + c * d = 0 := 
by sorry

end ab_cd_zero_l191_191531


namespace symmetric_y_axis_function_l191_191664

theorem symmetric_y_axis_function (f g : ℝ → ℝ) (h : ∀ (x : ℝ), g x = 3^x + 1) :
  (∀ x, f x = f (-x)) → (∀ x, f x = g (-x)) → (∀ x, f x = 3^(-x) + 1) :=
by
  intros h1 h2
  sorry

end symmetric_y_axis_function_l191_191664


namespace simplify_polynomial_l191_191887

theorem simplify_polynomial (q : ℤ) :
  (4*q^4 - 2*q^3 + 3*q^2 - 7*q + 9) + (5*q^3 - 8*q^2 + 6*q - 1) =
  4*q^4 + 3*q^3 - 5*q^2 - q + 8 :=
sorry

end simplify_polynomial_l191_191887


namespace problem_solution_l191_191970

def f (x : ℤ) : ℤ := 3 * x + 1
def g (x : ℤ) : ℤ := 4 * x - 3

theorem problem_solution :
  (f (g (f 3))) / (g (f (g 3))) = 112 / 109 := by
sorry

end problem_solution_l191_191970


namespace T_value_l191_191465

variable (x : ℝ)

def T : ℝ := (x-2)^4 + 4 * (x-2)^3 + 6 * (x-2)^2 + 4 * (x-2) + 1

theorem T_value : T x = (x-1)^4 := by
  sorry

end T_value_l191_191465


namespace verify_original_prices_l191_191715

noncomputable def original_price_of_sweater : ℝ := 43.11
noncomputable def original_price_of_shirt : ℝ := 35.68
noncomputable def original_price_of_pants : ℝ := 71.36

def price_of_shirt (sweater_price : ℝ) : ℝ := sweater_price - 7.43
def price_of_pants (shirt_price : ℝ) : ℝ := 2 * shirt_price
def discounted_sweater_price (sweater_price : ℝ) : ℝ := 0.85 * sweater_price
def total_cost (shirt_price pants_price discounted_sweater_price : ℝ) : ℝ := shirt_price + pants_price + discounted_sweater_price

theorem verify_original_prices 
  (total_cost_value : ℝ)
  (price_of_shirt_value : ℝ)
  (price_of_pants_value : ℝ)
  (discounted_sweater_price_value : ℝ) :
  total_cost_value = 143.67 ∧ 
  price_of_shirt_value = original_price_of_shirt ∧ 
  price_of_pants_value = original_price_of_pants ∧
  discounted_sweater_price_value = discounted_sweater_price original_price_of_sweater →
  total_cost (price_of_shirt original_price_of_sweater) 
             (price_of_pants (price_of_shirt original_price_of_sweater)) 
             (discounted_sweater_price original_price_of_sweater) = 143.67 :=
by
  intros
  sorry

end verify_original_prices_l191_191715


namespace ratio_of_cost_to_selling_price_l191_191557

-- Define the conditions in Lean
variable (C S : ℝ) -- C is the cost price per pencil, S is the selling price per pencil
variable (h : 90 * C - 40 * S = 90 * S)

-- Define the statement to be proved
theorem ratio_of_cost_to_selling_price (C S : ℝ) (h : 90 * C - 40 * S = 90 * S) : (90 * C) / (90 * S) = 13 :=
by
  sorry

end ratio_of_cost_to_selling_price_l191_191557


namespace percentage_increase_l191_191589

variable {α : Type} [LinearOrderedField α]

theorem percentage_increase (x y : α) (h : x = 0.5 * y) : y = x + x :=
by
  -- The steps of the proof are omitted and 'sorry' is used to skip actual proof.
  sorry

end percentage_increase_l191_191589


namespace brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l191_191879

noncomputable def brocard_vertex_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(a * b * c, c^3, b^3)

theorem brocard_vertex_coordinates_correct (a b c : ℝ) :
  brocard_vertex_trilinear_coordinates a b c = (a * b * c, c^3, b^3) :=
sorry

noncomputable def steiner_point_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(1 / (a * (b^2 - c^2)),
  1 / (b * (c^2 - a^2)),
  1 / (c * (a^2 - b^2)))

theorem steiner_point_coordinates_correct (a b c : ℝ) :
  steiner_point_trilinear_coordinates a b c = 
  (1 / (a * (b^2 - c^2)),
   1 / (b * (c^2 - a^2)),
   1 / (c * (a^2 - b^2))) :=
sorry

end brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l191_191879


namespace alice_pints_wednesday_l191_191778

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end alice_pints_wednesday_l191_191778


namespace polynomial_inequality_l191_191329

-- Define P(x) as a polynomial with non-negative coefficients
def isNonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

-- The main theorem, which states that for any polynomial P with non-negative coefficients,
-- if P(1) * P(1) ≥ 1, then P(x) * P(1/x) ≥ 1 for all positive x.
theorem polynomial_inequality (P : Polynomial ℝ) (hP : isNonNegativePolynomial P) (hP1 : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x : ℝ, 0 < x → P.eval x * P.eval (1 / x) ≥ 1 :=
by {
  sorry
}

end polynomial_inequality_l191_191329


namespace solve_equation_l191_191526

theorem solve_equation (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
    x^2 * (Real.log 27 / Real.log x) * (Real.log x / Real.log 9) = x + 4 → x = 2 :=
by
  sorry

end solve_equation_l191_191526


namespace count_even_numbers_l191_191115

theorem count_even_numbers : 
  ∃ n : ℕ, n = 199 ∧ ∀ m : ℕ, (302 ≤ m ∧ m < 700 ∧ m % 2 = 0) → 
    151 ≤ ((m - 300) / 2) ∧ ((m - 300) / 2) ≤ 349 :=
sorry

end count_even_numbers_l191_191115


namespace obtuse_triangle_side_range_l191_191680

theorem obtuse_triangle_side_range (a : ℝ) (h1 : 0 < a)
  (h2 : a + (a + 1) > a + 2)
  (h3 : (a + 1) + (a + 2) > a)
  (h4 : (a + 2) + a > a + 1)
  (h5 : (a + 2)^2 > a^2 + (a + 1)^2) : 1 < a ∧ a < 3 :=
by
  -- proof omitted
  sorry

end obtuse_triangle_side_range_l191_191680


namespace susan_backward_spaces_l191_191600

variable (spaces_to_win total_spaces : ℕ)
variables (first_turn second_turn_forward second_turn_back third_turn : ℕ)

theorem susan_backward_spaces :
  ∀ (total_spaces first_turn second_turn_forward second_turn_back third_turn win_left : ℕ),
  total_spaces = 48 →
  first_turn = 8 →
  second_turn_forward = 2 →
  third_turn = 6 →
  win_left = 37 →
  first_turn + second_turn_forward + third_turn - second_turn_back + win_left = total_spaces →
  second_turn_back = 6 :=
by
  intros total_spaces first_turn second_turn_forward second_turn_back third_turn win_left
  intros h_total h_first h_second_forward h_third h_win h_eq
  rw [h_total, h_first, h_second_forward, h_third, h_win] at h_eq
  sorry

end susan_backward_spaces_l191_191600


namespace pie_shop_total_earnings_l191_191451

theorem pie_shop_total_earnings :
  let price_per_slice_custard := 3
  let price_per_slice_apple := 4
  let price_per_slice_blueberry := 5
  let slices_per_whole_custard := 10
  let slices_per_whole_apple := 8
  let slices_per_whole_blueberry := 12
  let num_whole_custard_pies := 6
  let num_whole_apple_pies := 4
  let num_whole_blueberry_pies := 5
  let total_earnings :=
    (num_whole_custard_pies * slices_per_whole_custard * price_per_slice_custard) +
    (num_whole_apple_pies * slices_per_whole_apple * price_per_slice_apple) +
    (num_whole_blueberry_pies * slices_per_whole_blueberry * price_per_slice_blueberry)
  total_earnings = 608 := by
  sorry

end pie_shop_total_earnings_l191_191451


namespace machine_tasks_l191_191758

theorem machine_tasks (y : ℕ) 
  (h1 : (1 : ℚ)/(y + 4) + (1 : ℚ)/(y + 3) + (1 : ℚ)/(4 * y) = (1 : ℚ)/y) : y = 1 :=
sorry

end machine_tasks_l191_191758


namespace paving_rate_correct_l191_191229

-- Define the constants
def length (L : ℝ) := L = 5.5
def width (W : ℝ) := W = 4
def cost (C : ℝ) := C = 15400
def area (A : ℝ) := A = 22

-- Given the definitions above, prove the rate per sq. meter
theorem paving_rate_correct (L W C A : ℝ) (hL : length L) (hW : width W) (hC : cost C) (hA : area A) :
  C / A = 700 := 
sorry

end paving_rate_correct_l191_191229


namespace problem1_problem2_problem3_l191_191868

-- Define the function f
noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (b - 2^x) / (2^(x + 1) + a)

-- Problem 1
theorem problem1 (h_odd : ∀ x, f x a b = -f (-x) a b) : a = 2 ∧ b = 1 :=
sorry

-- Problem 2
theorem problem2 : (∀ x, f x 2 1 = -f (-x) 2 1) → ∀ x y, x < y → f x 2 1 > f y 2 1 :=
sorry

-- Problem 3
theorem problem3 (h_pos : ∀ x ≥ 1, f (k * 3^x) 2 1 + f (3^x - 9^x + 2) 2 1 > 0) : k < 4 / 3 :=
sorry

end problem1_problem2_problem3_l191_191868


namespace number_is_40_l191_191160

theorem number_is_40 (N : ℝ) (h : N = (3/8) * N + (1/4) * N + 15) : N = 40 :=
by
  sorry

end number_is_40_l191_191160


namespace mens_wages_l191_191478

theorem mens_wages
  (M : ℝ) (WW : ℝ) (B : ℝ)
  (h1 : 5 * M = WW)
  (h2 : WW = 8 * B)
  (h3 : 5 * M + WW + 8 * B = 60) :
  5 * M = 30 :=
by
  sorry

end mens_wages_l191_191478


namespace find_four_numbers_l191_191284

theorem find_four_numbers
  (a d : ℕ)
  (h_pos : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h_sum : (a - d) + a + (a + d) = 48)
  (b c : ℕ)
  (h_geo : b = a ∧ c = a + d)
  (last : ℕ)
  (h_last_val : last = 25)
  (h_geometric_seq : (a + d) * (a + d) = b * last)
  : (a - d, a, a + d, last) = (12, 16, 20, 25) := 
  sorry

end find_four_numbers_l191_191284


namespace go_stones_perimeter_count_l191_191656

def stones_per_side : ℕ := 6
def sides_of_square : ℕ := 4
def corner_stones : ℕ := 4

theorem go_stones_perimeter_count :
  (stones_per_side * sides_of_square) - corner_stones = 20 := 
by
  sorry

end go_stones_perimeter_count_l191_191656


namespace anita_gave_apples_l191_191731

theorem anita_gave_apples (initial_apples needed_for_pie apples_left_after_pie : ℝ)
  (h_initial : initial_apples = 10.0)
  (h_needed : needed_for_pie = 4.0)
  (h_left : apples_left_after_pie = 11.0) :
  ∃ (anita_apples : ℝ), anita_apples = 5 :=
by
  sorry

end anita_gave_apples_l191_191731


namespace point_distance_from_origin_l191_191077

theorem point_distance_from_origin (x y m : ℝ) (h1 : |y| = 15) (h2 : (x - 2)^2 + (y - 7)^2 = 169) (h3 : x > 2) :
  m = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end point_distance_from_origin_l191_191077


namespace total_winnings_l191_191340

theorem total_winnings (x : ℝ)
  (h1 : x / 4 = first_person_share)
  (h2 : x / 7 = second_person_share)
  (h3 : third_person_share = 17)
  (h4 : first_person_share + second_person_share + third_person_share = x) :
  x = 28 := 
by sorry

end total_winnings_l191_191340


namespace cookies_to_milk_l191_191832

theorem cookies_to_milk (milk_quarts : ℕ) (cookies : ℕ) (cups_in_quart : ℕ) 
  (H : milk_quarts = 3) (C : cookies = 24) (Q : cups_in_quart = 4) : 
  ∃ x : ℕ, x = 3 ∧ ∀ y : ℕ, y = 6 → x = (milk_quarts * cups_in_quart * y) / cookies := 
by {
  sorry
}

end cookies_to_milk_l191_191832


namespace time_to_destination_l191_191808

theorem time_to_destination (speed_ratio : ℕ) (mr_harris_time : ℕ) 
  (distance_multiple : ℕ) (h1 : speed_ratio = 3) 
  (h2 : mr_harris_time = 3) 
  (h3 : distance_multiple = 5) : 
  (mr_harris_time / speed_ratio) * distance_multiple = 5 := by
  sorry

end time_to_destination_l191_191808


namespace area_CDM_l191_191776

noncomputable def AC := 8
noncomputable def BC := 15
noncomputable def AB := 17
noncomputable def M := (AC + BC) / 2
noncomputable def AD := 17
noncomputable def BD := 17

theorem area_CDM (h₁ : AC = 8)
                 (h₂ : BC = 15)
                 (h₃ : AB = 17)
                 (h₄ : AD = 17)
                 (h₅ : BD = 17)
                 : ∃ (m n p : ℕ),
                   m = 121 ∧
                   n = 867 ∧
                   p = 136 ∧
                   m + n + p = 1124 ∧
                   ∃ (area_CDM : ℚ), 
                   area_CDM = (121 * Real.sqrt 867) / 136 :=
by
  sorry

end area_CDM_l191_191776


namespace constant_term_of_product_l191_191339

def P(x: ℝ) : ℝ := x^6 + 2 * x^2 + 3
def Q(x: ℝ) : ℝ := x^4 + x^3 + 4
def R(x: ℝ) : ℝ := 2 * x^2 + 3 * x + 7

theorem constant_term_of_product :
  let C := (P 0) * (Q 0) * (R 0)
  C = 84 :=
by
  let C := (P 0) * (Q 0) * (R 0)
  show C = 84
  sorry

end constant_term_of_product_l191_191339


namespace smallest_number_of_ones_l191_191140

-- Definitions inferred from the problem conditions
def N := (10^100 - 1) / 3
def M_k (k : ℕ) := (10^k - 1) / 9

theorem smallest_number_of_ones (k : ℕ) : M_k k % N = 0 → k = 300 :=
by {
  sorry
}

end smallest_number_of_ones_l191_191140


namespace min_surface_area_of_stacked_solids_l191_191746

theorem min_surface_area_of_stacked_solids :
  ∀ (l w h : ℕ), l = 3 → w = 2 → h = 1 → 
  (2 * (l * w + l * h + w * h) - 2 * l * w = 32) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  sorry

end min_surface_area_of_stacked_solids_l191_191746


namespace sequence_term_500_l191_191034

theorem sequence_term_500 :
  ∃ (a : ℕ → ℤ), 
  a 1 = 1001 ∧
  a 2 = 1005 ∧
  (∀ n, 1 ≤ n → (a n + a (n+1) + a (n+2)) = 2 * n) → 
  a 500 = 1334 := 
sorry

end sequence_term_500_l191_191034


namespace proof_problem_l191_191176

-- Define the problem space
variables (x y : ℝ)

-- Define the conditions
def satisfies_condition (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4)

-- The theorem statement
theorem proof_problem (hx : 0 < x) (hy : 0 < y) (hcond : satisfies_condition x y) :
  x + 2 * y = 1/2 + Real.sqrt 2 :=
sorry

end proof_problem_l191_191176


namespace lowest_possible_price_l191_191740

theorem lowest_possible_price
  (MSRP : ℕ) (max_initial_discount_percent : ℕ) (platinum_discount_percent : ℕ)
  (h1 : MSRP = 35) (h2 : max_initial_discount_percent = 40) (h3 : platinum_discount_percent = 30) :
  let initial_discount := max_initial_discount_percent * MSRP / 100
  let price_after_initial_discount := MSRP - initial_discount
  let platinum_discount := platinum_discount_percent * price_after_initial_discount / 100
  let lowest_price := price_after_initial_discount - platinum_discount
  lowest_price = 147 / 10 :=
by
  sorry

end lowest_possible_price_l191_191740


namespace smallest_a_undefined_inverse_l191_191643

theorem smallest_a_undefined_inverse (a : ℕ) (ha : a = 2) :
  (∀ (a : ℕ), 0 < a → ((Nat.gcd a 40 > 1) ∧ (Nat.gcd a 90 > 1)) ↔ a = 2) :=
by
  sorry

end smallest_a_undefined_inverse_l191_191643


namespace digits_partition_impossible_l191_191581

theorem digits_partition_impossible : 
  ¬ ∃ (A B : Finset ℕ), 
    A.card = 4 ∧ B.card = 4 ∧ A ∪ B = {1, 2, 3, 4, 5, 7, 8, 9} ∧ A ∩ B = ∅ ∧ 
    A.sum id = B.sum id := 
by
  sorry

end digits_partition_impossible_l191_191581


namespace y_intercept_l191_191359

theorem y_intercept (x1 y1 : ℝ) (m : ℝ) (h1 : x1 = -2) (h2 : y1 = 4) (h3 : m = 1 / 2) : 
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1/2 * x + 5) ∧ b = 5 := 
by
  sorry

end y_intercept_l191_191359


namespace find_a_l191_191351

theorem find_a (x y a : ℕ) (h₁ : x = 2) (h₂ : y = 3) (h₃ : a * x + 3 * y = 13) : a = 2 :=
by 
  sorry

end find_a_l191_191351


namespace convert_15_deg_to_rad_l191_191569

theorem convert_15_deg_to_rad (deg_to_rad : ℝ := Real.pi / 180) : 
  15 * deg_to_rad = Real.pi / 12 :=
by sorry

end convert_15_deg_to_rad_l191_191569


namespace slices_eaten_l191_191535

theorem slices_eaten (total_slices : Nat) (slices_left : Nat) (expected_slices_eaten : Nat) :
  total_slices = 32 →
  slices_left = 7 →
  expected_slices_eaten = 25 →
  total_slices - slices_left = expected_slices_eaten :=
by
  intros
  sorry

end slices_eaten_l191_191535


namespace integer_values_abs_lt_5pi_l191_191336

theorem integer_values_abs_lt_5pi : 
  ∃ n : ℕ, n = 31 ∧ ∀ x : ℤ, |(x : ℝ)| < 5 * Real.pi → x ∈ (Finset.Icc (-15) 15) := 
sorry

end integer_values_abs_lt_5pi_l191_191336


namespace algorithm_correct_l191_191705

def algorithm_output (x : Int) : Int :=
  let y := Int.natAbs x
  (2 ^ y) - y

theorem algorithm_correct : 
  algorithm_output (-3) = 5 :=
  by sorry

end algorithm_correct_l191_191705


namespace equation_of_line_l_l191_191218

theorem equation_of_line_l (P : ℝ × ℝ) (hP : P = (1, -1)) (θ₁ θ₂ : ℕ) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = θ₁ * 2) (hθ₂_90 : θ₂ = 90) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = l (P.fst)) := 
sorry

end equation_of_line_l_l191_191218


namespace tan_alpha_l191_191056

variable (α : ℝ)

theorem tan_alpha (h₁ : Real.sin α = -5/13) (h₂ : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l191_191056


namespace sufficient_but_not_necessary_condition_l191_191583

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 2 ∧ y = -1) → (x + y - 1 = 0) ∧ ¬(∀ x y, x + y - 1 = 0 → (x = 2 ∧ y = -1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l191_191583


namespace Tamika_hours_l191_191881

variable (h : ℕ)

theorem Tamika_hours :
  (45 * h = 55 * 5 + 85) → h = 8 :=
by 
  sorry

end Tamika_hours_l191_191881


namespace tom_monthly_fluid_intake_l191_191411

-- Define the daily fluid intake amounts
def daily_soda_intake := 5 * 12
def daily_water_intake := 64
def daily_juice_intake := 3 * 8
def daily_sports_drink_intake := 2 * 16
def additional_weekend_smoothie := 32

-- Define the weekdays and weekend days in a month
def weekdays_in_month := 5 * 4
def weekend_days_in_month := 2 * 4

-- Calculate the total daily intake
def daily_intake := daily_soda_intake + daily_water_intake + daily_juice_intake + daily_sports_drink_intake
def weekend_daily_intake := daily_intake + additional_weekend_smoothie

-- Calculate the total monthly intake
def total_fluid_intake_in_month := (daily_intake * weekdays_in_month) + (weekend_daily_intake * weekend_days_in_month)

-- Statement to prove
theorem tom_monthly_fluid_intake : total_fluid_intake_in_month = 5296 :=
by
  unfold total_fluid_intake_in_month
  unfold daily_intake weekend_daily_intake
  unfold weekdays_in_month weekend_days_in_month
  unfold daily_soda_intake daily_water_intake daily_juice_intake daily_sports_drink_intake additional_weekend_smoothie
  sorry

end tom_monthly_fluid_intake_l191_191411


namespace simplify_polynomials_l191_191148

-- Define the polynomials
def poly1 (q : ℝ) : ℝ := 5 * q^4 + 3 * q^3 - 7 * q + 8
def poly2 (q : ℝ) : ℝ := 6 - 9 * q^3 + 4 * q - 3 * q^4

-- The goal is to prove that the sum of poly1 and poly2 simplifies correctly
theorem simplify_polynomials (q : ℝ) : 
  poly1 q + poly2 q = 2 * q^4 - 6 * q^3 - 3 * q + 14 := 
by 
  sorry

end simplify_polynomials_l191_191148


namespace solution_set_of_inequality_l191_191652

theorem solution_set_of_inequality (x : ℝ) : 
  |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1 :=
sorry

end solution_set_of_inequality_l191_191652


namespace cyclic_quadrilateral_XF_XG_l191_191941

/-- 
Given:
- A cyclic quadrilateral ABCD inscribed in a circle O,
- Side lengths: AB = 4, BC = 3, CD = 7, DA = 9,
- Points X and Y such that DX/BD = 1/3 and BY/BD = 1/4,
- E is the intersection of line AX and the line through Y parallel to BC,
- F is the intersection of line CX and the line through E parallel to AB,
- G is the other intersection of line CX with circle O,
Prove:
- XF * XG = 36.5.
-/
theorem cyclic_quadrilateral_XF_XG (AB BC CD DA DX BD BY : ℝ) 
  (h_AB : AB = 4) (h_BC : BC = 3) (h_CD : CD = 7) (h_DA : DA = 9)
  (h_ratio1 : DX / BD = 1 / 3) (h_ratio2 : BY / BD = 1 / 4)
  (BD := Real.sqrt 73) :
  ∃ (XF XG : ℝ), XF * XG = 36.5 :=
by
  sorry

end cyclic_quadrilateral_XF_XG_l191_191941


namespace find_second_number_l191_191173

def average (nums : List ℕ) : ℕ :=
  nums.sum / nums.length

theorem find_second_number (nums : List ℕ) (a b : ℕ) (avg : ℕ) :
  average [10, 70, 28] = 36 ∧ average (10 :: 70 :: 28 :: []) + 4 = avg ∧ average (a :: b :: nums) = avg ∧ a = 20 ∧ b = 60 → b = 60 :=
by
  sorry

end find_second_number_l191_191173


namespace actual_total_discount_discount_difference_l191_191226

variable {original_price : ℝ}
variable (first_discount second_discount claimed_discount actual_discount : ℝ)

-- Definitions based on the problem conditions
def discount_1 (p : ℝ) : ℝ := (1 - first_discount) * p
def discount_2 (p : ℝ) : ℝ := (1 - second_discount) * discount_1 first_discount p

-- Statements we need to prove
theorem actual_total_discount (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70) :
  actual_discount = 1 - discount_2 first_discount second_discount original_price := 
by 
  sorry

theorem discount_difference (original_price : ℝ)
  (first_discount : ℝ := 0.40) (second_discount : ℝ := 0.30) (claimed_discount : ℝ := 0.70)
  (actual_discount : ℝ := 0.58) :
  claimed_discount - actual_discount = 0.12 := 
by 
  sorry

end actual_total_discount_discount_difference_l191_191226


namespace jenny_spent_180_minutes_on_bus_l191_191391

noncomputable def jennyBusTime : ℕ :=
  let timeAwayFromHome := 9 * 60  -- in minutes
  let classTime := 5 * 45  -- 5 classes each lasting 45 minutes
  let lunchTime := 45  -- in minutes
  let extracurricularTime := 90  -- 1 hour and 30 minutes
  timeAwayFromHome - (classTime + lunchTime + extracurricularTime)

theorem jenny_spent_180_minutes_on_bus : jennyBusTime = 180 :=
  by
  -- We need to prove that the total time Jenny was away from home minus time spent in school activities is 180 minutes.
  sorry  -- Proof to be completed.

end jenny_spent_180_minutes_on_bus_l191_191391


namespace cubic_roots_solution_sum_l191_191618

theorem cubic_roots_solution_sum (u v w : ℝ) (h1 : (u - 2) * (u - 3) * (u - 4) = 1 / 2)
                                     (h2 : (v - 2) * (v - 3) * (v - 4) = 1 / 2)
                                     (h3 : (w - 2) * (w - 3) * (w - 4) = 1 / 2)
                                     (distinct_roots : u ≠ v ∧ v ≠ w ∧ u ≠ w) :
  u^3 + v^3 + w^3 = -42 :=
sorry

end cubic_roots_solution_sum_l191_191618


namespace parallel_lines_slope_l191_191003

theorem parallel_lines_slope (b : ℝ) 
  (h₁ : ∀ x y : ℝ, 3 * y - 3 * b = 9 * x → (b = 3 - 9)) 
  (h₂ : ∀ x y : ℝ, y + 2 = (b + 9) * x → (b = 3 - 9)) : b = -6 :=
by
  sorry

end parallel_lines_slope_l191_191003


namespace chris_remaining_money_l191_191619

variable (video_game_cost : ℝ)
variable (discount_rate : ℝ)
variable (candy_cost : ℝ)
variable (tax_rate : ℝ)
variable (shipping_fee : ℝ)
variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)

noncomputable def remaining_money (video_game_cost discount_rate candy_cost tax_rate shipping_fee hourly_rate hours_worked : ℝ) : ℝ :=
  let discount := discount_rate * video_game_cost
  let discounted_price := video_game_cost - discount
  let total_video_game_cost := discounted_price + shipping_fee
  let video_tax := tax_rate * total_video_game_cost
  let candy_tax := tax_rate * candy_cost
  let total_cost := (total_video_game_cost + video_tax) + (candy_cost + candy_tax)
  let earnings := hourly_rate * hours_worked
  earnings - total_cost

theorem chris_remaining_money : remaining_money 60 0.15 5 0.10 3 8 9 = 7.1 :=
by
  sorry

end chris_remaining_money_l191_191619


namespace fractional_cake_eaten_l191_191762

def total_cake_eaten : ℚ :=
  1 / 3 + 1 / 3 + 1 / 6 + 1 / 12 + 1 / 24 + 1 / 48

theorem fractional_cake_eaten :
  total_cake_eaten = 47 / 48 := by
  sorry

end fractional_cake_eaten_l191_191762


namespace simplify_expression_l191_191579

theorem simplify_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := 
by 
  sorry 

end simplify_expression_l191_191579


namespace expense_of_5_yuan_is_minus_5_yuan_l191_191305

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l191_191305


namespace direct_variation_y_value_l191_191353

theorem direct_variation_y_value (x y : ℝ) (hx1 : x ≤ 10 → y = 3 * x)
  (hx2 : x > 10 → y = 6 * x) : 
  x = 20 → y = 120 := by
  sorry

end direct_variation_y_value_l191_191353


namespace percentage_of_x_l191_191640

variable {x y : ℝ}
variable {P : ℝ}

theorem percentage_of_x (h1 : (P / 100) * x = (20 / 100) * y) (h2 : x / y = 2) : P = 10 := by
  sorry

end percentage_of_x_l191_191640


namespace simplify_expression_l191_191523

theorem simplify_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -(m + n) :=
by sorry

end simplify_expression_l191_191523


namespace correct_scientific_notation_representation_l191_191131

-- Defining the given number of visitors in millions
def visitors_in_millions : Float := 8.0327
-- Converting this number to an integer and expressing in scientific notation
def rounded_scientific_notation (num : Float) : String :=
  if num == 8.0327 then "8.0 × 10^6" else "incorrect"

-- The mathematical proof statement
theorem correct_scientific_notation_representation :
  rounded_scientific_notation visitors_in_millions = "8.0 × 10^6" :=
by
  sorry

end correct_scientific_notation_representation_l191_191131


namespace not_age_of_child_l191_191585

noncomputable def sum_from_1_to_n (n : ℕ) := n * (n + 1) / 2

theorem not_age_of_child (N : ℕ) (S : Finset ℕ) (a b : ℕ) :
  S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11} ∧
  N = 1100 * a + 11 * b ∧
  a ≠ b ∧
  N ≥ 1000 ∧ N < 10000 ∧
  ((S.sum id) = N) ∧
  (∀ age ∈ S, N % age = 0) →
  10 ∉ S := 
by
  sorry

end not_age_of_child_l191_191585


namespace consecutive_page_numbers_sum_l191_191227

theorem consecutive_page_numbers_sum (n : ℕ) (h : n * (n + 1) * (n + 2) = 35280) :
  n + (n + 1) + (n + 2) = 96 := sorry

end consecutive_page_numbers_sum_l191_191227


namespace necessary_but_not_sufficient_l191_191764

-- Variables for the conditions
variables (x y : ℝ)

-- Conditions
def cond1 : Prop := x ≠ 1 ∨ y ≠ 4
def cond2 : Prop := x + y ≠ 5

-- Statement to prove the type of condition
theorem necessary_but_not_sufficient :
  cond2 x y → cond1 x y ∧ ¬(cond1 x y → cond2 x y) :=
sorry

end necessary_but_not_sufficient_l191_191764


namespace residue_of_5_pow_2023_mod_11_l191_191246

theorem residue_of_5_pow_2023_mod_11 : (5 ^ 2023) % 11 = 4 := by
  sorry

end residue_of_5_pow_2023_mod_11_l191_191246


namespace not_possible_in_five_trips_possible_in_six_trips_l191_191300

def truck_capacity := 2000
def rice_sacks := 150
def corn_sacks := 100
def rice_weight_per_sack := 60
def corn_weight_per_sack := 25

def total_rice_weight := rice_sacks * rice_weight_per_sack
def total_corn_weight := corn_sacks * corn_weight_per_sack
def total_weight := total_rice_weight + total_corn_weight

theorem not_possible_in_five_trips : total_weight > 5 * truck_capacity :=
by
  sorry

theorem possible_in_six_trips : total_weight <= 6 * truck_capacity :=
by
  sorry

#print axioms not_possible_in_five_trips
#print axioms possible_in_six_trips

end not_possible_in_five_trips_possible_in_six_trips_l191_191300


namespace price_decrease_percentage_l191_191489

-- Define the conditions
variables {P : ℝ} (original_price increased_price decreased_price : ℝ)
variables (y : ℝ) -- percentage by which increased price is decreased

-- Given conditions
def store_conditions :=
  increased_price = 1.20 * original_price ∧
  decreased_price = increased_price * (1 - y/100) ∧
  decreased_price = 0.75 * original_price

-- The proof problem
theorem price_decrease_percentage 
  (original_price increased_price decreased_price : ℝ)
  (y : ℝ) 
  (h : store_conditions original_price increased_price decreased_price y) :
  y = 37.5 :=
by 
  sorry

end price_decrease_percentage_l191_191489


namespace apples_left_is_ten_l191_191461

noncomputable def appleCost : ℝ := 0.80
noncomputable def orangeCost : ℝ := 0.50
def initialApples : ℕ := 50
def initialOranges : ℕ := 40
def totalEarnings : ℝ := 49
def orangesLeft : ℕ := 6

theorem apples_left_is_ten (A : ℕ) :
  (50 - A) * appleCost + (40 - orangesLeft) * orangeCost = 49 → A = 10 :=
by
  sorry

end apples_left_is_ten_l191_191461


namespace cricket_run_rate_l191_191692

theorem cricket_run_rate
  (run_rate_first_10_overs : ℝ)
  (overs_first_10_overs : ℕ)
  (target_runs : ℕ)
  (remaining_overs : ℕ)
  (run_rate_required : ℝ) :
  run_rate_first_10_overs = 3.2 →
  overs_first_10_overs = 10 →
  target_runs = 242 →
  remaining_overs = 40 →
  run_rate_required = 5.25 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) = 210 →
  (target_runs - (run_rate_first_10_overs * overs_first_10_overs)) / remaining_overs = run_rate_required :=
by
  sorry

end cricket_run_rate_l191_191692


namespace find_max_value_l191_191890

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  2 * x * y * Real.sqrt 3 + 3 * y * z * Real.sqrt 2 + 3 * z * x

theorem find_max_value (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z)
  (h₃ : x^2 + y^2 + z^2 = 1) : 
  maximum_value x y z ≤ Real.sqrt 3 := sorry

end find_max_value_l191_191890


namespace simplify_fraction_l191_191119

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by
  -- proof goes here
  sorry

end simplify_fraction_l191_191119


namespace contrapositive_of_inequality_l191_191711

theorem contrapositive_of_inequality (a b c : ℝ) (h : a > b → a + c > b + c) : a + c ≤ b + c → a ≤ b :=
by
  intro h_le
  apply not_lt.mp
  intro h_gt
  have h2 := h h_gt
  linarith

end contrapositive_of_inequality_l191_191711


namespace triangle_area_l191_191412

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area : 
  ∀ (A B C : (ℝ × ℝ)),
  (A = (3, 3)) →
  (B = (4.5, 7.5)) →
  (C = (7.5, 4.5)) →
  area_of_triangle A B C = 8.625 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  unfold area_of_triangle
  norm_num
  sorry

end triangle_area_l191_191412


namespace pascal_triangle_41st_number_42nd_row_l191_191395

open Nat

theorem pascal_triangle_41st_number_42nd_row :
  Nat.choose 42 40 = 861 := by
  sorry

end pascal_triangle_41st_number_42nd_row_l191_191395


namespace max_product_of_two_positive_numbers_l191_191706

theorem max_product_of_two_positive_numbers (x y s : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = s) : 
  x * y ≤ (s ^ 2) / 4 :=
sorry

end max_product_of_two_positive_numbers_l191_191706


namespace water_pump_rate_l191_191806

theorem water_pump_rate (hourly_rate : ℕ) (minutes : ℕ) (calculated_gallons : ℕ) : 
  hourly_rate = 600 → minutes = 30 → calculated_gallons = (hourly_rate * (minutes / 60)) → 
  calculated_gallons = 300 :=
by 
  sorry

end water_pump_rate_l191_191806


namespace exists_hexagon_in_square_l191_191436

structure Point (α : Type*) :=
(x : α)
(y : α)

def is_in_square (p : Point ℕ) : Prop :=
p.x ≤ 4 ∧ p.y ≤ 4

def area_of_hexagon (vertices : List (Point ℕ)) : ℝ :=
-- placeholder for actual area calculation of a hexagon
sorry

theorem exists_hexagon_in_square : ∃ (p1 p2 : Point ℕ), 
  is_in_square p1 ∧ is_in_square p2 ∧ 
  area_of_hexagon [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 0⟩, ⟨4, 4⟩, p1, p2] = 6 :=
sorry

end exists_hexagon_in_square_l191_191436


namespace percentage_peanut_clusters_is_64_l191_191236

def total_chocolates := 50
def caramels := 3
def nougats := 2 * caramels
def truffles := caramels + 6
def other_chocolates := caramels + nougats + truffles
def peanut_clusters := total_chocolates - other_chocolates
def percentage_peanut_clusters := (peanut_clusters * 100) / total_chocolates

theorem percentage_peanut_clusters_is_64 :
  percentage_peanut_clusters = 64 := by
  sorry

end percentage_peanut_clusters_is_64_l191_191236


namespace raffle_prize_l191_191433

theorem raffle_prize (P : ℝ) :
  (0.80 * P = 80) → (P = 100) :=
by
  intro h1
  sorry

end raffle_prize_l191_191433


namespace GCF_75_135_l191_191848

theorem GCF_75_135 : Nat.gcd 75 135 = 15 :=
by
sorry

end GCF_75_135_l191_191848


namespace sled_dog_race_l191_191103

theorem sled_dog_race (d t : ℕ) (h1 : d + t = 315) (h2 : (1.2 : ℚ) * d + t = (1 / 2 : ℚ) * (2 * d + 3 * t)) :
  d = 225 ∧ t = 90 :=
sorry

end sled_dog_race_l191_191103


namespace triangle_AB_C_min_perimeter_l191_191767

noncomputable def minimum_perimeter (a b c : ℕ) (A B C : ℝ) : ℝ := a + b + c

theorem triangle_AB_C_min_perimeter
  (a b c : ℕ)
  (A B C : ℝ)
  (h1 : A = 2 * B)
  (h2 : C > π / 2)
  (h3 : a^2 = b * (b + c))
  (h4 : ∀ x : ℕ, x > 0 → a ≠ 0)
  (h5 :  a + b > c ∧ a + c > b ∧ b + c > a) :
  minimum_perimeter a b c A B C = 77 := 
sorry

end triangle_AB_C_min_perimeter_l191_191767


namespace petya_must_have_photo_files_on_portable_hard_drives_l191_191343

theorem petya_must_have_photo_files_on_portable_hard_drives 
    (H F P T : ℕ) 
    (h1 : H > F) 
    (h2 : P > T) 
    : ∃ x, x ≠ 0 ∧ x ≤ H :=
by
  sorry

end petya_must_have_photo_files_on_portable_hard_drives_l191_191343


namespace price_of_fruits_l191_191844

theorem price_of_fruits
  (x y : ℝ)
  (h1 : 9 * x + 10 * y = 73.8)
  (h2 : 17 * x + 6 * y = 69.8)
  (hx : x = 2.2)
  (hy : y = 5.4) : 
  9 * 2.2 + 10 * 5.4 = 73.8 ∧ 17 * 2.2 + 6 * 5.4 = 69.8 :=
by
  sorry

end price_of_fruits_l191_191844


namespace largest_number_of_HCF_LCM_l191_191096

theorem largest_number_of_HCF_LCM (HCF : ℕ) (k1 k2 : ℕ) (n1 n2 : ℕ) 
  (hHCF : HCF = 50)
  (hk1 : k1 = 11) 
  (hk2 : k2 = 12) 
  (hn1 : n1 = HCF * k1) 
  (hn2 : n2 = HCF * k2) :
  max n1 n2 = 600 := by
  sorry

end largest_number_of_HCF_LCM_l191_191096


namespace counterexamples_count_l191_191314

def sum_of_digits (n : Nat) : Nat :=
  -- Function to calculate the sum of digits of n
  sorry

def no_zeros (n : Nat) : Prop :=
  -- Function to check that there are no zeros in the digits of n
  sorry

def is_prime (n : Nat) : Prop :=
  -- Function to check if a number is prime
  sorry

theorem counterexamples_count : 
  ∃ (M : List Nat), 
  (∀ m ∈ M, sum_of_digits m = 5 ∧ no_zeros m) ∧ 
  (∀ m ∈ M, ¬ is_prime m) ∧
  M.length = 9 := 
sorry

end counterexamples_count_l191_191314


namespace population_in_2050_l191_191785

def population : ℕ → ℕ := sorry

theorem population_in_2050 : population 2050 = 2700 :=
by
  -- sorry statement to skip the proof
  sorry

end population_in_2050_l191_191785


namespace composite_divisible_by_six_l191_191903

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l191_191903


namespace negation_of_existential_l191_191530

theorem negation_of_existential : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 ≤ 0) := 
by 
  sorry

end negation_of_existential_l191_191530


namespace measure_of_angle_C_sin_A_plus_sin_B_l191_191125

-- Problem 1
theorem measure_of_angle_C (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) : C = Real.pi / 3 := 
sorry

-- Problem 2
theorem sin_A_plus_sin_B (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) (h3 : c = 2 * Real.sqrt 3) : Real.sin A + Real.sin B = 3 / 2 := 
sorry

end measure_of_angle_C_sin_A_plus_sin_B_l191_191125


namespace nine_digit_palindrome_count_l191_191337

-- Defining the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

-- Defining the proposition of the number of 9-digit palindromes
def num_9_digit_palindromes (digs : Multiset ℕ) : ℕ := 36

-- The proof statement
theorem nine_digit_palindrome_count : num_9_digit_palindromes digits = 36 := 
sorry

end nine_digit_palindrome_count_l191_191337


namespace total_marbles_l191_191691

-- Definitions to state the problem
variables {r b g : ℕ}
axiom ratio_condition : r / b = 2 / 4 ∧ r / g = 2 / 6
axiom blue_marbles : b = 30

-- Theorem statement
theorem total_marbles : r + b + g = 90 :=
by sorry

end total_marbles_l191_191691


namespace earnings_from_cauliflower_correct_l191_191487

-- Define the earnings from each vegetable
def earnings_from_broccoli : ℕ := 57
def earnings_from_carrots : ℕ := 2 * earnings_from_broccoli
def earnings_from_spinach : ℕ := (earnings_from_carrots / 2) + 16
def total_earnings : ℕ := 380

-- Define the total earnings from vegetables other than cauliflower
def earnings_from_others : ℕ := earnings_from_broccoli + earnings_from_carrots + earnings_from_spinach

-- Define the earnings from cauliflower
def earnings_from_cauliflower : ℕ := total_earnings - earnings_from_others

-- Theorem to prove the earnings from cauliflower
theorem earnings_from_cauliflower_correct : earnings_from_cauliflower = 136 :=
by
  sorry

end earnings_from_cauliflower_correct_l191_191487


namespace max_value_of_a_l191_191990

theorem max_value_of_a :
  ∀ (m : ℚ) (x : ℤ),
    (0 < x ∧ x ≤ 50) →
    (1 / 2 < m ∧ m < 25 / 49) →
    (∀ k : ℤ, m * x + 3 ≠ k) →
  m < 25 / 49 :=
sorry

end max_value_of_a_l191_191990


namespace boxes_per_class_l191_191886

variable (boxes : ℕ) (classes : ℕ)

theorem boxes_per_class (h1 : boxes = 3) (h2 : classes = 4) : 
  (boxes : ℚ) / (classes : ℚ) = 3 / 4 :=
by
  rw [h1, h2]
  norm_num

end boxes_per_class_l191_191886


namespace combined_experience_is_correct_l191_191036

-- Define the conditions as given in the problem
def james_experience : ℕ := 40
def partner_less_years : ℕ := 10
def partner_experience : ℕ := james_experience - partner_less_years

-- The combined experience of James and his partner
def combined_experience : ℕ := james_experience + partner_experience

-- Lean statement to prove the combined experience is 70 years
theorem combined_experience_is_correct : combined_experience = 70 := by sorry

end combined_experience_is_correct_l191_191036


namespace shoe_price_on_monday_l191_191259

theorem shoe_price_on_monday
  (price_on_thursday : ℝ)
  (price_increase : ℝ)
  (discount : ℝ)
  (price_on_friday : ℝ := price_on_thursday * (1 + price_increase))
  (price_on_monday : ℝ := price_on_friday * (1 - discount))
  (price_on_thursday_eq : price_on_thursday = 50)
  (price_increase_eq : price_increase = 0.2)
  (discount_eq : discount = 0.15) :
  price_on_monday = 51 :=
by
  sorry

end shoe_price_on_monday_l191_191259


namespace value_ne_one_l191_191533

theorem value_ne_one (a b: ℝ) (h : a * b ≠ 0) : (|a| / a) + (|b| / b) ≠ 1 := 
by 
  sorry

end value_ne_one_l191_191533


namespace ratio_of_radii_of_cylinders_l191_191540

theorem ratio_of_radii_of_cylinders
  (r_V r_B h_V h_B : ℝ)
  (h1 : h_V = 1/2 * h_B)
  (h2 : π * r_B^2 * h_B / 2  = 4)
  (h3 : π * r_V^2 * h_V = 16) :
  r_V / r_B = 2 := 
by 
  sorry

end ratio_of_radii_of_cylinders_l191_191540


namespace opposite_of_one_fourth_l191_191156

/-- The opposite of the fraction 1/4 is -1/4 --/
theorem opposite_of_one_fourth : - (1 / 4) = -1 / 4 :=
by
  sorry

end opposite_of_one_fourth_l191_191156


namespace B_participated_Huangmei_Opera_l191_191720

-- Definitions using given conditions
def participated_A (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∨ 
  (c "Huangmei Flower Picking" ∧ ¬ c "Yue Family Boxing")

def participated_B (c : String → Prop) : Prop :=
  (c "Huangmei Opera" ∧ ¬ c "Huangmei Flower Picking") ∨
  (c "Yue Family Boxing" ∧ ¬ c "Huangmei Flower Picking")

def participated_C (c : String → Prop) : Prop :=
  c "Huangmei Opera" ∧ c "Huangmei Flower Picking" ∧ c "Yue Family Boxing" ->
  (c "Huangmei Opera" ∨ c "Huangmei Flower Picking" ∨ c "Yue Family Boxing")

-- Proving the special class that B participated in
theorem B_participated_Huangmei_Opera :
  ∃ c : String → Prop, participated_A c ∧ participated_B c ∧ participated_C c → c "Huangmei Opera" :=
by
  -- proof steps would go here
  sorry

end B_participated_Huangmei_Opera_l191_191720


namespace geometric_series_sum_eq_l191_191686

theorem geometric_series_sum_eq (a r : ℝ) 
  (h_sum : (∑' n:ℕ, a * r^n) = 20) 
  (h_odd_sum : (∑' n:ℕ, a * r^(2 * n + 1)) = 8) : 
  r = 2 / 3 := 
sorry

end geometric_series_sum_eq_l191_191686


namespace find_constant_l191_191240

-- Define the conditions
def is_axles (x : ℕ) : Prop := x = 5
def toll_for_truck (t : ℝ) : Prop := t = 4

-- Define the formula for the toll
def toll_formula (t : ℝ) (constant : ℝ) (x : ℕ) : Prop :=
  t = 2.50 + constant * (x - 2)

-- Proof problem statement
theorem find_constant : ∃ (constant : ℝ), 
  ∀ x : ℕ, is_axles x → toll_for_truck 4 →
  toll_formula 4 constant x → constant = 0.50 :=
sorry

end find_constant_l191_191240


namespace baseball_games_in_season_l191_191265

def games_per_month : ℕ := 7
def months_in_season : ℕ := 2
def total_games_in_season : ℕ := games_per_month * months_in_season

theorem baseball_games_in_season : total_games_in_season = 14 := by
  sorry

end baseball_games_in_season_l191_191265


namespace isosceles_triangle_height_l191_191603

theorem isosceles_triangle_height (l w h : ℝ) 
  (h1 : l * w = (1 / 2) * w * h) : h = 2 * l :=
by
  sorry

end isosceles_triangle_height_l191_191603


namespace round_robin_total_points_l191_191105

theorem round_robin_total_points :
  let points_per_match := 2
  let total_matches := 3
  (total_matches * points_per_match) = 6 :=
by
  sorry

end round_robin_total_points_l191_191105


namespace greatest_whole_number_solution_l191_191175

theorem greatest_whole_number_solution (x : ℤ) (h : 6 * x - 5 < 7 - 3 * x) : x ≤ 1 :=
sorry

end greatest_whole_number_solution_l191_191175


namespace equal_elements_l191_191297

theorem equal_elements {n : ℕ} (a : ℕ → ℝ) (h₁ : n ≥ 2) (h₂ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≠ -1) 
  (h₃ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1)) 
  (hn1 : a (n + 1) = a 1) (hn2 : a (n + 2) = a 2) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = a 1 := by
  sorry

end equal_elements_l191_191297


namespace sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l191_191030

variable {α : ℝ} (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)

theorem sin_and_tan (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3/5 ∧ Real.tan α = 3/4 :=
sorry

theorem sin_add_pi_over_4_and_tan_2alpha (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)
  (h_sin : Real.sin α = -3/5) (h_tan : Real.tan α = 3/4) :
  Real.sin (α + π/4) = -7 * Real.sqrt 2 / 10 ∧ Real.tan (2 * α) = 24/7 :=
sorry

end sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l191_191030


namespace number_of_welders_left_l191_191355

-- Definitions for the given problem
def total_welders : ℕ := 36
def initial_days : ℝ := 1
def remaining_days : ℝ := 3.0000000000000004
def total_days : ℝ := 3

-- Condition equations
variable (r : ℝ) -- rate at which each welder works
variable (W : ℝ) -- total work

-- Equation representing initial total work
def initial_work : W = total_welders * r * total_days := by sorry

-- Welders who left for another project
variable (X : ℕ) -- number of welders who left

-- Equation representing remaining work
def remaining_work : (total_welders - X) * r * remaining_days = W - (total_welders * r * initial_days) := by sorry

-- Theorem to prove
theorem number_of_welders_left :
  (total_welders * total_days : ℝ) = W →
  (total_welders - X) * remaining_days = W - (total_welders * r * initial_days) →
  X = 12 :=
sorry

end number_of_welders_left_l191_191355


namespace find_a_l191_191876

noncomputable def log_a (a: ℝ) (x: ℝ) : ℝ := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : log_a a 2 - log_a a 4 = 2) :
  a = Real.sqrt 2 / 2 :=
sorry

end find_a_l191_191876


namespace abc_geq_expression_l191_191743

variable (a b c : ℝ) -- Define variables a, b, c as real numbers
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) -- Define conditions of a, b, c being positive

theorem abc_geq_expression : 
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
by 
  sorry -- Proof goes here

end abc_geq_expression_l191_191743


namespace percentage_two_sections_cleared_l191_191432

noncomputable def total_candidates : ℕ := 1200
def pct_cleared_all_sections : ℝ := 0.05
def pct_cleared_none_sections : ℝ := 0.05
def pct_cleared_one_section : ℝ := 0.25
def pct_cleared_four_sections : ℝ := 0.20
def cleared_three_sections : ℕ := 300

theorem percentage_two_sections_cleared :
  (total_candidates - total_candidates * (pct_cleared_all_sections + pct_cleared_none_sections + pct_cleared_one_section + pct_cleared_four_sections) - cleared_three_sections) / total_candidates * 100 = 20 := by
  sorry

end percentage_two_sections_cleared_l191_191432


namespace factorization_option_D_l191_191402

-- Define variables
variables (x y : ℝ)

-- Define the expressions
def left_side_D := -4 * x^2 + 12 * x * y - 9 * y^2
def right_side_D := -(2 * x - 3 * y)^2

-- Theorem statement
theorem factorization_option_D : left_side_D x y = right_side_D x y :=
sorry

end factorization_option_D_l191_191402


namespace total_farm_tax_collected_l191_191793

noncomputable def totalFarmTax (taxPaid: ℝ) (percentage: ℝ) : ℝ := taxPaid / (percentage / 100)

theorem total_farm_tax_collected (taxPaid : ℝ) (percentage : ℝ) (h_taxPaid : taxPaid = 480) (h_percentage : percentage = 16.666666666666668) :
  totalFarmTax taxPaid percentage = 2880 :=
by
  rw [h_taxPaid, h_percentage]
  simp [totalFarmTax]
  norm_num
  sorry

end total_farm_tax_collected_l191_191793


namespace minimum_value_l191_191546

variable {a b : ℝ}

noncomputable def given_conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + 2 * b = 2

theorem minimum_value :
  given_conditions a b →
  ∃ x, x = (1 + 4 * a + 3 * b) / (a * b) ∧ x ≥ 25 / 2 :=
by
  sorry

end minimum_value_l191_191546


namespace mark_total_eggs_in_a_week_l191_191099

-- Define the given conditions
def first_store_eggs_per_day := 5 * 12 -- 5 dozen eggs per day
def second_store_eggs_per_day := 30
def third_store_eggs_per_odd_day := 25 * 12 -- 25 dozen eggs per odd day
def third_store_eggs_per_even_day := 15 * 12 -- 15 dozen eggs per even day
def days_per_week := 7
def odd_days_per_week := 4
def even_days_per_week := 3

-- Lean theorem statement to prove the total eggs supplied in a week
theorem mark_total_eggs_in_a_week : 
    first_store_eggs_per_day * days_per_week + 
    second_store_eggs_per_day * days_per_week + 
    third_store_eggs_per_odd_day * odd_days_per_week + 
    third_store_eggs_per_even_day * even_days_per_week =
    2370 := 
    sorry  -- Placeholder for the actual proof

end mark_total_eggs_in_a_week_l191_191099


namespace no_valid_angles_l191_191298

open Real

theorem no_valid_angles (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 2 * π)
    (h3 : ∀ k : ℤ, θ ≠ k * (π / 2))
    (h4 : cos θ * tan θ = sin θ ^ 3) : false :=
by
  -- The proof goes here
  sorry

end no_valid_angles_l191_191298


namespace blackboard_final_number_lower_bound_l191_191907

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def L (c : ℝ) : ℝ := 1 + Real.log c / Real.log phi

theorem blackboard_final_number_lower_bound (c : ℝ) (n : ℕ) (h_pos_c : c > 1) (h_pos_n : n > 0) :
  ∃ x, x ≥ ((c^(n / (L c)) - 1) / (c^(1 / (L c)) - 1))^(L c) :=
sorry

end blackboard_final_number_lower_bound_l191_191907


namespace gcd_182_98_l191_191481

theorem gcd_182_98 : Nat.gcd 182 98 = 14 :=
by
  -- Provide the proof here, but as per instructions, we'll use sorry to skip it.
  sorry

end gcd_182_98_l191_191481


namespace phi_range_l191_191019

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem phi_range (φ : ℝ) : 
  (|φ| ≤ Real.pi / 2) ∧ 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x φ > 2) →
  (Real.pi / 12 ≤ φ ∧ φ ≤ Real.pi / 6) :=
by
  sorry

end phi_range_l191_191019


namespace sector_area_l191_191668

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360) * Real.pi * r^2 = (35 * Real.pi) / 3 :=
by
  -- Using the provided conditions to simplify the expression
  rw [h_r, h_θ]
  -- Simplify and solve the expression
  sorry

end sector_area_l191_191668


namespace concert_revenue_l191_191354

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end concert_revenue_l191_191354


namespace cubic_inequality_l191_191417

theorem cubic_inequality (a b : ℝ) : (a > b) ↔ (a^3 > b^3) := sorry

end cubic_inequality_l191_191417


namespace solve_problem_l191_191565

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (0.65 * x > 26) ∧ (0.40 * y < -3) ∧ ((x - y)^2 ≥ 100) 
  → (x > 40) ∧ (y < -7.5)

theorem solve_problem (x y : ℝ) (h : proof_problem x y) : (x > 40) ∧ (y < -7.5) := 
sorry

end solve_problem_l191_191565


namespace ram_krish_together_time_l191_191508

theorem ram_krish_together_time : 
  let t_R := 36
  let t_K := t_R / 2
  let task_per_day_R := 1 / t_R
  let task_per_day_K := 1 / t_K
  let task_per_day_together := task_per_day_R + task_per_day_K
  let T := 1 / task_per_day_together
  T = 12 := 
by
  sorry

end ram_krish_together_time_l191_191508


namespace ratio_pea_patch_to_radish_patch_l191_191627

-- Definitions
def sixth_of_pea_patch : ℝ := 5
def whole_radish_patch : ℝ := 15

-- Theorem to prove
theorem ratio_pea_patch_to_radish_patch :
  (6 * sixth_of_pea_patch) / whole_radish_patch = 2 :=
by 
  -- skip the actual proof since it's not required
  sorry

end ratio_pea_patch_to_radish_patch_l191_191627


namespace triangle_medians_inequality_l191_191245

-- Define the parameters
variables {a b c t_a t_b t_c D : ℝ}

-- Assume the sides and medians of the triangle and the diameter of the circumcircle
axiom sides_of_triangle (a b c : ℝ) : Prop
axiom medians_of_triangle (t_a t_b t_c : ℝ) : Prop
axiom diameter_of_circumcircle (D : ℝ) : Prop

-- The theorem to prove
theorem triangle_medians_inequality
  (h_sides : sides_of_triangle a b c)
  (h_medians : medians_of_triangle t_a t_b t_c)
  (h_diameter : diameter_of_circumcircle D)
  : (a^2 + b^2) / t_c + (b^2 + c^2) / t_a + (c^2 + a^2) / t_b ≤ 6 * D :=
sorry -- proof omitted

end triangle_medians_inequality_l191_191245


namespace books_per_author_l191_191014

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end books_per_author_l191_191014


namespace sin_sum_bound_l191_191816

theorem sin_sum_bound (x : ℝ) : 
  |(Real.sin x) + (Real.sin (Real.sqrt 2 * x))| < 2 - 1 / (100 * (x^2 + 1)) :=
by sorry

end sin_sum_bound_l191_191816


namespace range_of_k_l191_191466

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > k → (3 / (x + 1) < 1)) ↔ k ≥ 2 := sorry

end range_of_k_l191_191466


namespace folding_cranes_together_l191_191366

theorem folding_cranes_together (rateA rateB combined_time : ℝ)
  (hA : rateA = 1 / 30)
  (hB : rateB = 1 / 45)
  (combined_rate : ℝ := rateA + rateB)
  (h_combined_rate : combined_rate = 1 / combined_time):
  combined_time = 18 :=
by
  sorry

end folding_cranes_together_l191_191366


namespace sum_of_reciprocals_of_squares_l191_191380

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 41) :
  (1 / (a^2) + 1 / (b^2)) = 1682 / 1681 := sorry

end sum_of_reciprocals_of_squares_l191_191380


namespace shortest_remaining_side_length_l191_191150

noncomputable def triangle_has_right_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem shortest_remaining_side_length {a b : ℝ} (ha : a = 5) (hb : b = 12) (h_right_angle : ∃ c, triangle_has_right_angle a b c) :
  ∃ c, c = 5 :=
by 
  sorry

end shortest_remaining_side_length_l191_191150


namespace whitney_money_leftover_l191_191551

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end whitney_money_leftover_l191_191551


namespace EventB_is_random_l191_191943

-- Define the events A, B, C, and D as propositions
def EventA : Prop := ∀ (x : ℕ), true -- A coin thrown will fall due to gravity (certain event)
def EventB : Prop := ∃ (n : ℕ), n > 0 -- Hitting the target with a score of 10 points (random event)
def EventC : Prop := ∀ (x : ℕ), true -- The sun rises from the east (certain event)
def EventD : Prop := ∀ (x : ℕ), false -- Horse runs at 70 meters per second (impossible event)

-- Prove that EventB is random, we can use a custom predicate for random events
def is_random_event (e : Prop) : Prop := (∃ (n : ℕ), n > 1) ∧ ¬ ∀ (x : ℕ), e

-- Main statement
theorem EventB_is_random :
  is_random_event EventB :=
by sorry -- The proof will be written here

end EventB_is_random_l191_191943


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l191_191491

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l191_191491


namespace min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l191_191695

noncomputable def line_equation (A B C x y : ℝ) : Prop := A * x + B * y + C = 0

noncomputable def point_on_line (x y A B C : ℝ) : Prop := line_equation A B C x y

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  (|C2 - C1|) / (Real.sqrt (A^2 + B^2))

theorem min_distance_between_parallel_lines :
  ∀ (A B C1 C2 x y : ℝ),
  point_on_line x y A B C1 ∧ point_on_line x y A B C2 →
  distance_between_parallel_lines A B C1 C2 = 3 :=
by
  intros A B C1 C2 x y h
  sorry

theorem distance_when_line_parallel_to_x_axis :
  ∀ (x1 x2 y k A B C1 C2 : ℝ),
  k = 3 →
  point_on_line x1 k A B C1 →
  point_on_line x2 k A B C2 →
  |x2 - x1| = 5 :=
by
  intros x1 x2 y k A B C1 C2 hk h1 h2
  sorry

end min_distance_between_parallel_lines_distance_when_line_parallel_to_x_axis_l191_191695


namespace perimeter_of_rectangular_field_l191_191126

theorem perimeter_of_rectangular_field (L B : ℝ) 
    (h1 : B = 0.60 * L) 
    (h2 : L * B = 37500) : 
    2 * L + 2 * B = 800 :=
by 
  -- proof goes here
  sorry

end perimeter_of_rectangular_field_l191_191126


namespace find_y_l191_191379

theorem find_y 
  (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 18) 
  (h2 : x + 2 * y = 10) : 
  y = 1.5 := 
by 
  sorry

end find_y_l191_191379


namespace blue_balls_needed_l191_191292

-- Conditions
variables (R Y B W : ℝ)
axiom h1 : 2 * R = 5 * B
axiom h2 : 3 * Y = 7 * B
axiom h3 : 9 * B = 6 * W

-- Proof Problem
theorem blue_balls_needed : (3 * R + 4 * Y + 3 * W) = (64 / 3) * B := by
  sorry

end blue_balls_needed_l191_191292


namespace female_democrats_l191_191207

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats_l191_191207


namespace lindy_distance_l191_191166

theorem lindy_distance
  (d : ℝ) (v_j : ℝ) (v_c : ℝ) (v_l : ℝ) (t : ℝ)
  (h1 : d = 270)
  (h2 : v_j = 4)
  (h3 : v_c = 5)
  (h4 : v_l = 8)
  (h_time : t = d / (v_j + v_c)) :
  v_l * t = 240 := by
  sorry

end lindy_distance_l191_191166


namespace greatest_possible_value_of_squares_l191_191515

theorem greatest_possible_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 15)
  (h2 : ab + c + d = 78)
  (h3 : ad + bc = 160)
  (h4 : cd = 96) :
  a^2 + b^2 + c^2 + d^2 ≤ 717 ∧ ∃ a b c d, a + b = 15 ∧ ab + c + d = 78 ∧ ad + bc = 160 ∧ cd = 96 ∧ a^2 + b^2 + c^2 + d^2 = 717 :=
sorry

end greatest_possible_value_of_squares_l191_191515


namespace part1_part2_part3_l191_191372

-- Part (1): Proving \( p \implies m > \frac{3}{2} \)
theorem part1 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0) → (m > 3 / 2) :=
by
  sorry

-- Part (2): Proving \( q \implies (m < -1 \text{ or } m > 2) \)
theorem part2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → (m < -1 ∨ m > 2) :=
by
  sorry

-- Part (3): Proving \( (p ∨ q) \implies ((-\infty, -1) ∪ (\frac{3}{2}, +\infty)) \)
theorem part3 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0 ∨ ∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → ((m < -1) ∨ (3 / 2 < m)) :=
by
  sorry

end part1_part2_part3_l191_191372


namespace regular_pentagons_similar_l191_191854

-- Define a regular pentagon
structure RegularPentagon :=
  (side_length : ℝ)
  (internal_angle : ℝ)
  (angle_eq : internal_angle = 108)
  (side_positive : side_length > 0)

-- The theorem stating that two regular pentagons are always similar
theorem regular_pentagons_similar (P Q : RegularPentagon) : 
  ∀ P Q : RegularPentagon, P.internal_angle = Q.internal_angle ∧ P.side_length * Q.side_length ≠ 0 := 
sorry

end regular_pentagons_similar_l191_191854


namespace box_2008_count_l191_191660

noncomputable def box_count (a : ℕ → ℕ) : Prop :=
  a 1 = 7 ∧ a 4 = 8 ∧ ∀ n : ℕ, 1 ≤ n ∧ n + 3 ≤ 2008 → a n + a (n + 1) + a (n + 2) + a (n + 3) = 30

theorem box_2008_count (a : ℕ → ℕ) (h : box_count a) : a 2008 = 8 :=
by
  sorry

end box_2008_count_l191_191660


namespace train_length_is_correct_l191_191200

noncomputable def length_of_train (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := relative_speed_kmh * (5/18)
  let length := relative_speed_ms * time_s
  length

theorem train_length_is_correct (h1 : 84 = 84) (h2 : 6 = 6) (h3 : 4.399648028157747 = 4.399648028157747) :
  length_of_train 84 6 4.399648028157747 = 110.991201 := by
  dsimp [length_of_train]
  norm_num
  sorry

end train_length_is_correct_l191_191200


namespace cartesian_coordinates_problem_l191_191642

theorem cartesian_coordinates_problem
  (x1 y1 x2 y2 : ℕ)
  (h1 : x1 < y1)
  (h2 : x2 > y2)
  (h3 : x2 * y2 = x1 * y1 + 67)
  (h4 : 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2)
  : Nat.digits 10 (x1 * 1000 + y1 * 100 + x2 * 10 + y2) = [1, 9, 8, 5] :=
by
  sorry

end cartesian_coordinates_problem_l191_191642


namespace find_sin_2alpha_l191_191291

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 4) Real.pi) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (Real.pi / 4 - α)) : 
  Real.sin (2 * α) = -1 / 9 :=
sorry

end find_sin_2alpha_l191_191291


namespace systematic_sampling_method_l191_191869

def num_rows : ℕ := 50
def num_seats_per_row : ℕ := 30

def is_systematic_sampling (select_interval : ℕ) : Prop :=
  ∀ n, select_interval = n * num_seats_per_row + 8

theorem systematic_sampling_method :
  is_systematic_sampling 30 :=
by
  sorry

end systematic_sampling_method_l191_191869


namespace intersection_of_A_and_B_l191_191012

def A := {x : ℝ | x^2 - 5 * x + 6 > 0}
def B := {x : ℝ | x / (x - 1) < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end intersection_of_A_and_B_l191_191012


namespace area_of_backyard_eq_400_l191_191112

-- Define the conditions
def length_condition (l : ℕ) : Prop := 25 * l = 1000
def perimeter_condition (l w : ℕ) : Prop := 20 * (l + w) = 1000

-- State the theorem
theorem area_of_backyard_eq_400 (l w : ℕ) (h_length : length_condition l) (h_perimeter : perimeter_condition l w) : l * w = 400 :=
  sorry

end area_of_backyard_eq_400_l191_191112


namespace original_team_members_l191_191309

theorem original_team_members (m p total_points : ℕ) (h_m : m = 3) (h_p : p = 2) (h_total : total_points = 12) :
  (total_points / p) + m = 9 := by
  sorry

end original_team_members_l191_191309


namespace coeff_x3y5_in_expansion_l191_191882

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l191_191882


namespace items_priced_at_9_yuan_l191_191625

theorem items_priced_at_9_yuan (equal_number_items : ℕ)
  (total_cost : ℕ)
  (price_8_yuan : ℕ)
  (price_9_yuan : ℕ)
  (price_8_yuan_count : ℕ)
  (price_9_yuan_count : ℕ) :
  equal_number_items * 2 = price_8_yuan_count + price_9_yuan_count ∧
  (price_8_yuan_count * price_8_yuan + price_9_yuan_count * price_9_yuan = total_cost) ∧
  (price_8_yuan = 8) ∧
  (price_9_yuan = 9) ∧
  (total_cost = 172) →
  price_9_yuan_count = 12 :=
by
  sorry

end items_priced_at_9_yuan_l191_191625


namespace polynomial_divisibility_l191_191232

theorem polynomial_divisibility (
  p q r s : ℝ
) :
  (x^5 + 5 * x^4 + 10 * p * x^3 + 10 * q * x^2 + 5 * r * x + s) % (x^4 + 4 * x^3 + 6 * x^2 + 4 * x + 1) = 0 ->
  (p + q + r) * s = -2 :=
by {
  sorry
}

end polynomial_divisibility_l191_191232


namespace negation_exists_gt_one_l191_191107

theorem negation_exists_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
sorry

end negation_exists_gt_one_l191_191107


namespace range_of_f_l191_191141

/-- Define the piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else Real.cos x

/-- Prove that the range of f(x) is [-1, ∞) -/
theorem range_of_f : Set.range f = Set.Ici (-1) :=
by sorry

end range_of_f_l191_191141


namespace ice_cream_ratio_l191_191951

theorem ice_cream_ratio :
  ∃ (B C : ℕ), 
    C = 1 ∧
    (∃ (W D : ℕ), 
      D = 2 ∧
      W = B + 1 ∧
      B + W + C + D = 10 ∧
      B / C = 3
    ) := sorry

end ice_cream_ratio_l191_191951


namespace arrangeable_sequence_l191_191204

theorem arrangeable_sequence (n : Fin 2017 → ℤ) :
  (∀ i : Fin 2017, ∃ (perm : Fin 5 → Fin 5),
    let a := n ((i + perm 0) % 2017)
    let b := n ((i + perm 1) % 2017)
    let c := n ((i + perm 2) % 2017)
    let d := n ((i + perm 3) % 2017)
    let e := n ((i + perm 4) % 2017)
    a - b + c - d + e = 29) →
  (∀ i : Fin 2017, n i = 29) :=
by
  sorry

end arrangeable_sequence_l191_191204


namespace percent_calculation_l191_191604

-- Given conditions
def part : ℝ := 120.5
def whole : ℝ := 80.75

-- Theorem statement
theorem percent_calculation : (part / whole) * 100 = 149.26 := 
sorry

end percent_calculation_l191_191604


namespace total_water_capacity_of_coolers_l191_191722

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l191_191722


namespace set_intersection_l191_191492
noncomputable def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2 }
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 1 }

theorem set_intersection (x : ℝ) : x ∈ A ∩ B ↔ x ∈ A := sorry

end set_intersection_l191_191492


namespace unique_reversible_six_digit_number_exists_l191_191622

theorem unique_reversible_six_digit_number_exists :
  ∃! (N : ℤ), 100000 ≤ N ∧ N < 1000000 ∧
  ∃ (f e d c b a : ℤ), 
  N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ 
  9 * N = 100000 * f + 10000 * e + 1000 * d + 100 * c + 10 * b + a := 
sorry

end unique_reversible_six_digit_number_exists_l191_191622


namespace floor_e_is_two_l191_191940

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l191_191940


namespace min_ab_minus_cd_l191_191833

theorem min_ab_minus_cd (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9) (h5 : a^2 + b^2 + c^2 + d^2 = 21) : ab - cd ≥ 2 := sorry

end min_ab_minus_cd_l191_191833


namespace g_of_square_sub_one_l191_191875

variable {R : Type*} [LinearOrderedField R]

def g (x : R) : R := 3

theorem g_of_square_sub_one (x : R) : g ((x - 1)^2) = 3 := 
by sorry

end g_of_square_sub_one_l191_191875


namespace min_needed_framing_l191_191288

-- Define the original dimensions of the picture
def original_width_inch : ℕ := 5
def original_height_inch : ℕ := 7

-- Define the factor by which the dimensions are doubled
def doubling_factor : ℕ := 2

-- Define the width of the border
def border_width_inch : ℕ := 3

-- Define the function to calculate the new dimensions after doubling
def new_width_inch : ℕ := original_width_inch * doubling_factor
def new_height_inch : ℕ := original_height_inch * doubling_factor

-- Define the function to calculate dimensions including the border
def total_width_inch : ℕ := new_width_inch + 2 * border_width_inch
def total_height_inch : ℕ := new_height_inch + 2 * border_width_inch

-- Define the function to calculate the perimeter of the picture with border
def perimeter_inch : ℕ := 2 * (total_width_inch + total_height_inch)

-- Conversision from inches to feet (1 foot = 12 inches)
def inch_to_foot_conversion_factor : ℕ := 12

-- Define the function to calculate the minimum linear feet of framing needed
noncomputable def min_linear_feet_of_framing : ℕ := (perimeter_inch + inch_to_foot_conversion_factor - 1) / inch_to_foot_conversion_factor

-- The main theorem statement
theorem min_needed_framing : min_linear_feet_of_framing = 6 := by
  -- Proof construction is omitted as per the instructions
  sorry

end min_needed_framing_l191_191288


namespace largest_five_digit_number_divisible_by_5_l191_191088

theorem largest_five_digit_number_divisible_by_5 : 
  ∃ n, (n % 5 = 0) ∧ (99990 ≤ n) ∧ (n ≤ 99995) ∧ (∀ m, (m % 5 = 0) → (99990 ≤ m) → (m ≤ 99995) → m ≤ n) :=
by
  -- The proof is omitted as per the instructions
  sorry

end largest_five_digit_number_divisible_by_5_l191_191088


namespace find_angle_D_l191_191749

theorem find_angle_D
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_C = 2 * angle_D)
  (h3 : angle_A = 100)
  (h4 : angle_B + angle_C + angle_D = 180) :
  angle_D = 100 / 3 :=
by
  sorry

end find_angle_D_l191_191749


namespace denominator_of_speed_l191_191628

theorem denominator_of_speed (h : 0.8 = 8 / d * 3600 / 1000) : d = 36 := 
by
  sorry

end denominator_of_speed_l191_191628


namespace inscribed_rectangle_area_correct_l191_191949

noncomputable def area_of_inscribed_rectangle : Prop := 
  let AD : ℝ := 15 / (12 / (1 / 3) + 3)
  let AB : ℝ := 1 / 3 * AD
  AD * AB = 25 / 12

theorem inscribed_rectangle_area_correct :
  area_of_inscribed_rectangle
  := by
  let hf : ℝ := 12
  let eg : ℝ := 15
  let ad : ℝ := 15 / (hf / (1 / 3) + 3)
  let ab : ℝ := 1 / 3 * ad
  have area : ad * ab = 25 / 12 := by sorry
  exact area

end inscribed_rectangle_area_correct_l191_191949


namespace prob_chair_theorem_l191_191714

def numAvailableChairs : ℕ := 10 - 1

def totalWaysToChooseTwoChairs : ℕ := Nat.choose numAvailableChairs 2

def adjacentPairs : ℕ :=
  let pairs := [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8), (8, 9)]
  pairs.length

def probNextToEachOther : ℚ := adjacentPairs / totalWaysToChooseTwoChairs

def probNotNextToEachOther : ℚ := 1 - probNextToEachOther

theorem prob_chair_theorem : probNotNextToEachOther = 5/6 :=
by
  sorry

end prob_chair_theorem_l191_191714


namespace find_a_prove_inequality_l191_191282

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + 2 * x + a * Real.log x

theorem find_a (a : ℝ) (h : (2 * Real.exp 1 + 2 + a) * (-1 / 2) = -1) : a = -2 * Real.exp 1 :=
by
  sorry

theorem prove_inequality (a : ℝ) (h1 : a = -2 * Real.exp 1) :
    ∀ x : ℝ, x > 0 → f x a > x^2 + 2 :=
by
  sorry

end find_a_prove_inequality_l191_191282


namespace completing_square_solution_l191_191542

theorem completing_square_solution (x : ℝ) :
  2 * x^2 + 4 * x - 3 = 0 →
  (x + 1)^2 = 5 / 2 :=
by
  sorry

end completing_square_solution_l191_191542


namespace min_occupied_seats_l191_191947

theorem min_occupied_seats (n : ℕ) (h_n : n = 150) : 
  ∃ k : ℕ, k = 37 ∧ ∀ (occupied : Finset ℕ), 
    occupied.card < k → ∃ i : ℕ, i ∉ occupied ∧ ∀ j : ℕ, j ∈ occupied → j + 1 ≠ i ∧ j - 1 ≠ i :=
by
  sorry

end min_occupied_seats_l191_191947


namespace problem_l191_191967

def a : ℝ := (-2)^2002
def b : ℝ := (-2)^2003

theorem problem : a + b = -2^2002 := by
  sorry

end problem_l191_191967


namespace value_of_x_l191_191059

theorem value_of_x (x : ℝ) (m : ℕ) (h1 : m = 31) :
  ((x ^ m) / (5 ^ m)) * ((x ^ 16) / (4 ^ 16)) = 1 / (2 * 10 ^ 31) → x = 1 := by
  sorry

end value_of_x_l191_191059


namespace gcd_1343_816_l191_191514

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l191_191514


namespace sin_C_and_area_of_triangle_l191_191748

open Real

noncomputable section

theorem sin_C_and_area_of_triangle 
  (A B C : ℝ)
  (cos_A : Real := sqrt 3 / 3)
  (a b c : ℝ := (3 * sqrt 2)) 
  (cosA : cos A = sqrt 3 / 3)
  -- angles in radians, use radians for the angles when proving
  (side_c : c = sqrt 3)
  (side_a : a = 3 * sqrt 2) :
  (sin C = 1 / 3) ∧ (1 / 2 * a * b * sin C = 5 * sqrt 6 / 3) :=
by
  sorry

end sin_C_and_area_of_triangle_l191_191748


namespace initial_wage_illiterate_l191_191456

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l191_191456


namespace circles_intersect_l191_191208

variable (r1 r2 d : ℝ)
variable (h1 : r1 = 4)
variable (h2 : r2 = 5)
variable (h3 : d = 7)

theorem circles_intersect : 1 < d ∧ d < r1 + r2 :=
by sorry

end circles_intersect_l191_191208


namespace production_units_l191_191989

-- Define the production function U
def U (women hours days : ℕ) : ℕ := women * hours * days

-- State the theorem
theorem production_units (x z : ℕ) (hx : ¬ x = 0) :
  U z z z = (z^3 / x) :=
  sorry

end production_units_l191_191989


namespace necessary_and_sufficient_condition_l191_191795

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 0) ↔ (a + 1 / a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l191_191795


namespace sphere_volume_l191_191563

theorem sphere_volume (π : ℝ) (r : ℝ):
  4 * π * r^2 = 144 * π →
  (4 / 3) * π * r^3 = 288 * π :=
by
  sorry

end sphere_volume_l191_191563


namespace system_solutions_l191_191322

noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

theorem system_solutions (x y z : ℝ) :
  (f x = y ∧ f y = z ∧ f z = x) ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_l191_191322


namespace least_non_lucky_multiple_of_12_l191_191347

/- Defines what it means for a number to be a lucky integer -/
def isLucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

/- Proves the least positive multiple of 12 that is not a lucky integer is 96 -/
theorem least_non_lucky_multiple_of_12 : ∃ n, n % 12 = 0 ∧ ¬isLucky n ∧ ∀ m, m % 12 = 0 ∧ ¬isLucky m → n ≤ m :=
  by
  sorry

end least_non_lucky_multiple_of_12_l191_191347


namespace largest_integer_dividing_consecutive_product_l191_191935

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l191_191935


namespace symmetric_line_l191_191741

theorem symmetric_line (x y : ℝ) : (2 * x + y - 4 = 0) → (2 * x - y + 4 = 0) :=
by
  sorry

end symmetric_line_l191_191741


namespace part1_part2_l191_191092

open Set Real

def M (x : ℝ) : Prop := x^2 - 3 * x - 18 ≤ 0
def N (x : ℝ) (a : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 2 * a + 1

theorem part1 (a : ℝ) (h : a = 3) : (Icc (-2 : ℝ) 6 = {x | M x ∧ N x a}) ∧ (compl {x | N x a} = Iic (-2) ∪ Ioi 7) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, M x ∧ N x a ↔ N x a) → a ≤ 5 / 2 :=
by
  sorry

end part1_part2_l191_191092


namespace last_two_digits_of_7_pow_2016_l191_191381

theorem last_two_digits_of_7_pow_2016 : (7^2016 : ℕ) % 100 = 1 := 
by {
  sorry
}

end last_two_digits_of_7_pow_2016_l191_191381


namespace carrots_total_l191_191493

theorem carrots_total (sandy_carrots : Nat) (sam_carrots : Nat) (h1 : sandy_carrots = 6) (h2 : sam_carrots = 3) :
  sandy_carrots + sam_carrots = 9 :=
by
  sorry

end carrots_total_l191_191493


namespace area_of_triangle_le_one_fourth_l191_191760

open Real

noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_le_one_fourth (t : ℝ) (x y : ℝ) (h_t : 0 < t ∧ t < 1) (h_x : 0 ≤ x ∧ x ≤ 1)
  (h_y : y = t * (2 * x - t)) :
  area_triangle t (t^2) 1 0 x y ≤ 1 / 4 :=
by
  sorry

end area_of_triangle_le_one_fourth_l191_191760


namespace mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l191_191211

noncomputable def Mork_base_income (M : ℝ) : ℝ := M
noncomputable def Mindy_base_income (M : ℝ) : ℝ := 4 * M
noncomputable def Mork_total_income (M : ℝ) : ℝ := 1.5 * M
noncomputable def Mindy_total_income (M : ℝ) : ℝ := 6 * M

noncomputable def Mork_total_tax (M : ℝ) : ℝ :=
  0.4 * M + 0.5 * 0.5 * M
noncomputable def Mindy_total_tax (M : ℝ) : ℝ :=
  0.3 * 4 * M + 0.35 * 2 * M

noncomputable def Mork_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M) / (Mork_total_income M)

noncomputable def Mindy_effective_tax_rate (M : ℝ) : ℝ :=
  (Mindy_total_tax M) / (Mindy_total_income M)

noncomputable def combined_effective_tax_rate (M : ℝ) : ℝ :=
  (Mork_total_tax M + Mindy_total_tax M) / (Mork_total_income M + Mindy_total_income M)

theorem mork_effective_tax_rate_theorem (M : ℝ) : Mork_effective_tax_rate M = 43.33 / 100 := sorry
theorem mindy_effective_tax_rate_theorem (M : ℝ) : Mindy_effective_tax_rate M = 31.67 / 100 := sorry
theorem combined_effective_tax_rate_theorem (M : ℝ) : combined_effective_tax_rate M = 34 / 100 := sorry

end mork_effective_tax_rate_theorem_mindy_effective_tax_rate_theorem_combined_effective_tax_rate_theorem_l191_191211


namespace sum_areas_frequency_distribution_histogram_l191_191673

theorem sum_areas_frequency_distribution_histogram :
  ∀ (rectangles : List ℝ), (∀ r ∈ rectangles, 0 ≤ r ∧ r ≤ 1) → rectangles.sum = 1 := 
  by
    intro rectangles h
    sorry

end sum_areas_frequency_distribution_histogram_l191_191673


namespace original_cost_l191_191912

theorem original_cost (C : ℝ) (h : 670 = C + 0.35 * C) : C = 496.30 :=
by
  -- The proof is omitted
  sorry

end original_cost_l191_191912


namespace prove_proposition_l191_191396

-- Define the propositions p and q
def p : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def q : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Define the main theorem to prove
theorem prove_proposition : (¬ p) ∨ q :=
by { sorry }

end prove_proposition_l191_191396


namespace ben_and_sue_answer_l191_191197

theorem ben_and_sue_answer :
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  z = 84
:= by
  let x := 8
  let y := 3 * (x + 2)
  let z := 3 * (y - 2)
  show z = 84
  sorry

end ben_and_sue_answer_l191_191197


namespace maxwell_walking_speed_l191_191842

variable (distance : ℕ) (brad_speed : ℕ) (maxwell_time : ℕ) (brad_time : ℕ) (maxwell_speed : ℕ)

-- Given conditions
def conditions := distance = 54 ∧ brad_speed = 6 ∧ maxwell_time = 6 ∧ brad_time = 5

-- Problem statement
theorem maxwell_walking_speed (h : conditions distance brad_speed maxwell_time brad_time) : maxwell_speed = 4 := sorry

end maxwell_walking_speed_l191_191842


namespace cube_sphere_volume_relation_l191_191102

theorem cube_sphere_volume_relation (n : ℕ) (h : 2 < n)
  (h_volume : n^3 - (n^3 * pi / 6) = (n^3 * pi / 3)) : n = 8 :=
sorry

end cube_sphere_volume_relation_l191_191102


namespace solve_equation_solve_inequality_system_l191_191968

theorem solve_equation (x : ℝ) : x^2 - 2 * x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 :=
by
  sorry

theorem solve_inequality_system (x : ℝ) : (4 * (x - 1) < x + 2) ∧ ((x + 7) / 3 > x) ↔ x < 2 :=
by
  sorry

end solve_equation_solve_inequality_system_l191_191968


namespace positive_integer_solutions_l191_191584

theorem positive_integer_solutions
  (m n k : ℕ)
  (hm : 0 < m) (hn : 0 < n) (hk : 0 < k) :
  3 * m + 4 * n = 5 * k ↔ (m = 1 ∧ n = 2 ∧ k = 2) := 
by
  sorry

end positive_integer_solutions_l191_191584


namespace regular_polygon_sides_l191_191137

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l191_191137


namespace angelas_insects_l191_191629

variable (DeanInsects : ℕ) (JacobInsects : ℕ) (AngelaInsects : ℕ)

theorem angelas_insects
  (h1 : DeanInsects = 30)
  (h2 : JacobInsects = 5 * DeanInsects)
  (h3 : AngelaInsects = JacobInsects / 2):
  AngelaInsects = 75 := 
by
  sorry

end angelas_insects_l191_191629


namespace average_expenditure_Feb_to_July_l191_191277

theorem average_expenditure_Feb_to_July (avg_Jan_to_Jun : ℝ) (spend_Jan : ℝ) (spend_July : ℝ) 
    (total_Jan_to_Jun : avg_Jan_to_Jun = 4200) (spend_Jan_eq : spend_Jan = 1200) (spend_July_eq : spend_July = 1500) :
    (4200 * 6 - 1200 + 1500) / 6 = 4250 :=
by
  sorry

end average_expenditure_Feb_to_July_l191_191277


namespace expand_and_simplify_fraction_l191_191963

theorem expand_and_simplify_fraction (x : ℝ) (hx : x ≠ 0) : 
  (3 / 7) * ((7 / (x^2)) + 15 * (x^3) - 4 * x) = (3 / (x^2)) + (45 * (x^3) / 7) - (12 * x / 7) :=
by
  sorry

end expand_and_simplify_fraction_l191_191963


namespace int_cubed_bound_l191_191028

theorem int_cubed_bound (a : ℤ) (h : 0 < a^3 ∧ a^3 < 9) : a = 1 ∨ a = 2 :=
sorry

end int_cubed_bound_l191_191028


namespace hyperbola_center_l191_191213

theorem hyperbola_center :
  ∀ (x y : ℝ), 
  (4 * x + 8)^2 / 36 - (3 * y - 6)^2 / 25 = 1 → (x, y) = (-2, 2) :=
by
  intros x y h
  sorry

end hyperbola_center_l191_191213


namespace total_selection_methods_l191_191269

def num_courses_group_A := 3
def num_courses_group_B := 4
def total_courses_selected := 3

theorem total_selection_methods 
  (at_least_one_from_each : num_courses_group_A > 0 ∧ num_courses_group_B > 0)
  (total_courses : total_courses_selected = 3) :
  ∃ N, N = 30 :=
sorry

end total_selection_methods_l191_191269


namespace winter_sales_l191_191066

theorem winter_sales (spring_sales summer_sales fall_sales : ℕ) (fall_sales_pct : ℝ) (total_sales winter_sales : ℕ) :
  spring_sales = 6 →
  summer_sales = 7 →
  fall_sales = 5 →
  fall_sales_pct = 0.20 →
  fall_sales = ⌊fall_sales_pct * total_sales⌋ →
  total_sales = spring_sales + summer_sales + fall_sales + winter_sales →
  winter_sales = 7 :=
by
  sorry

end winter_sales_l191_191066


namespace length_of_AB_l191_191371

-- Defining the parabola and the condition on x1 and x2
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def condition (x1 x2 : ℝ) : Prop := x1 + x2 = 9

-- The main statement to prove |AB| = 11
theorem length_of_AB (x1 x2 y1 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (hx : condition x1 x2) :
  abs (x1 - x2) + abs (y1 - y2) = 11 :=
sorry

end length_of_AB_l191_191371


namespace smallest_yummy_integer_l191_191874

theorem smallest_yummy_integer :
  ∃ (n A : ℤ), 4046 = n * (2 * A + n - 1) ∧ A ≥ 0 ∧ (∀ m, 4046 = m * (2 * A + m - 1) ∧ m ≥ 0 → A ≤ 1011) :=
sorry

end smallest_yummy_integer_l191_191874


namespace solve_fractional_equation_l191_191885

theorem solve_fractional_equation (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (3 / (x^2 - x) + 1 = x / (x - 1)) → x = 3 :=
by
  sorry -- Placeholder for the actual proof

end solve_fractional_equation_l191_191885


namespace distance_to_angle_bisector_l191_191401

theorem distance_to_angle_bisector 
  (P : ℝ × ℝ) 
  (h_hyperbola : P.1^2 - P.2^2 = 9) 
  (h_distance_to_line_neg_x : abs (P.1 + P.2) = 2016 * Real.sqrt 2) : 
  abs (P.1 - P.2) / Real.sqrt 2 = 448 :=
sorry

end distance_to_angle_bisector_l191_191401


namespace xanthia_hot_dogs_l191_191658

theorem xanthia_hot_dogs (a b : ℕ) (h₁ : a = 5) (h₂ : b = 7) :
  ∃ n m : ℕ, n * a = m * b ∧ n = 7 := by 
sorry

end xanthia_hot_dogs_l191_191658


namespace shaded_area_fraction_l191_191701

theorem shaded_area_fraction (total_grid_squares : ℕ) (number_1_squares : ℕ) (number_9_squares : ℕ) (number_8_squares : ℕ) (partial_squares_1 : ℕ) (partial_squares_2 : ℕ) (partial_squares_3 : ℕ) :
  total_grid_squares = 18 * 8 →
  number_1_squares = 8 →
  number_9_squares = 15 →
  number_8_squares = 16 →
  partial_squares_1 = 6 →
  partial_squares_2 = 6 →
  partial_squares_3 = 8 →
  (2 * (number_1_squares + number_9_squares + number_9_squares + number_8_squares) + (partial_squares_1 + partial_squares_2 + partial_squares_3)) = 2 * (74 : ℕ) →
  (74 / 144 : ℚ) = 37 / 72 :=
by
  intros _ _ _ _ _ _ _ _
  sorry

end shaded_area_fraction_l191_191701


namespace candy_lollipops_l191_191734

theorem candy_lollipops (κ c l : ℤ) 
  (h1 : κ = l + c - 8)
  (h2 : c = l + κ - 14) :
  l = 11 :=
by
  sorry

end candy_lollipops_l191_191734


namespace relationship_a_b_l191_191891

theorem relationship_a_b (a b : ℝ) :
  (∃ (P : ℝ × ℝ), P ∈ {Q : ℝ × ℝ | Q.snd = -3 * Q.fst + b} ∧
                   ∃ (R : ℝ × ℝ), R ∈ {S : ℝ × ℝ | S.snd = -a * S.fst + 3} ∧
                   R = (-P.snd, -P.fst)) →
  a = 1 / 3 ∧ b = -9 :=
by
  intro h
  sorry

end relationship_a_b_l191_191891


namespace min_n_probability_l191_191721

-- Define the number of members in teams
def num_members (n : ℕ) : ℕ := n

-- Define the total number of handshakes
def total_handshakes (n : ℕ) : ℕ := n * n

-- Define the number of ways to choose 2 handshakes from total handshakes
def choose_two_handshakes (n : ℕ) : ℕ := (total_handshakes n).choose 2

-- Define the number of ways to choose event A (involves exactly 3 different members)
def event_a_count (n : ℕ) : ℕ := 2 * n.choose 1 * (n - 1).choose 1

-- Define the probability of event A
def probability_event_a (n : ℕ) : ℚ := (event_a_count n : ℚ) / (choose_two_handshakes n : ℚ)

-- The minimum value of n such that the probability of event A is less than 1/10
theorem min_n_probability :
  ∃ n : ℕ, (probability_event_a n < (1 : ℚ) / 10) ∧ n ≥ 20 :=
by {
  sorry
}

end min_n_probability_l191_191721


namespace coin_flip_sequences_l191_191747

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l191_191747


namespace perpendicular_lines_slope_condition_l191_191117

theorem perpendicular_lines_slope_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x - 1 ↔ x + 2 * y + 3 = 0) → k = 2 :=
by
  sorry

end perpendicular_lines_slope_condition_l191_191117


namespace find_position_of_2017_l191_191909

theorem find_position_of_2017 :
  ∃ (row col : ℕ), row = 45 ∧ col = 81 ∧ 2017 = (row - 1)^2 + col :=
by
  sorry

end find_position_of_2017_l191_191909


namespace initial_pigs_l191_191635

theorem initial_pigs (x : ℕ) (h1 : x + 22 = 86) : x = 64 :=
by
  sorry

end initial_pigs_l191_191635


namespace box_volume_increase_l191_191707

-- Conditions
def volume (l w h : ℝ) : ℝ := l * w * h
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def sum_of_edges (l w h : ℝ) : ℝ := 4 * (l + w + h)

-- The main theorem we want to state
theorem box_volume_increase
  (l w h : ℝ)
  (h_volume : volume l w h = 5000)
  (h_surface_area : surface_area l w h = 1800)
  (h_sum_of_edges : sum_of_edges l w h = 210) :
  volume (l + 2) (w + 2) (h + 2) = 7018 := 
by sorry

end box_volume_increase_l191_191707


namespace sin_cos_term_side_l191_191495

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end sin_cos_term_side_l191_191495


namespace probability_other_side_red_given_seen_red_l191_191121

-- Definition of conditions
def total_cards := 9
def black_black_cards := 5
def black_red_cards := 2
def red_red_cards := 2
def red_sides := (2 * red_red_cards) + black_red_cards -- Total number of red sides
def favorable_red_red_sides := 2 * red_red_cards      -- Number of red sides on fully red cards

-- The required probability
def probability_other_side_red_given_red : ℚ := sorry

-- The main statement to prove
theorem probability_other_side_red_given_seen_red :
  probability_other_side_red_given_red = 2/3 :=
sorry

end probability_other_side_red_given_seen_red_l191_191121


namespace hvac_cost_per_vent_l191_191621

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l191_191621


namespace arithmetic_sequence_terms_l191_191724

theorem arithmetic_sequence_terms (a d n : ℤ) (last_term : ℤ)
  (h_a : a = 5)
  (h_d : d = 3)
  (h_last_term : last_term = 149)
  (h_n_eq : last_term = a + (n - 1) * d) :
  n = 49 :=
by sorry

end arithmetic_sequence_terms_l191_191724


namespace annual_rent_per_square_foot_l191_191267

theorem annual_rent_per_square_foot
  (length width : ℕ) (monthly_rent : ℕ) (h_length : length = 10)
  (h_width : width = 8) (h_monthly_rent : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := 
by 
  -- We assume the theorem is true.
  sorry

end annual_rent_per_square_foot_l191_191267


namespace range_of_a_l191_191550

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 :=
sorry

end range_of_a_l191_191550


namespace find_g_3_16_l191_191361

theorem find_g_3_16 (g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x → x ≤ 1 → g x = g x) 
(h2 : g 0 = 0) 
(h3 : ∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) 
(h4 : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x) 
(h5 : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) : 
  g (3 / 16) = 8 / 27 :=
sorry

end find_g_3_16_l191_191361


namespace value_of_v3_at_neg4_l191_191999

def poly (x : ℤ) : ℤ := (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem value_of_v3_at_neg4 : poly (-4) = -49 := 
by
  sorry

end value_of_v3_at_neg4_l191_191999


namespace probability_of_selecting_meiqi_l191_191771

def four_red_bases : List String := ["Meiqi", "Wangcunkou", "Zhulong", "Xiaoshun"]

theorem probability_of_selecting_meiqi :
  (1 / 4 : ℝ) = 1 / (four_red_bases.length : ℝ) :=
  by sorry

end probability_of_selecting_meiqi_l191_191771


namespace sum_of_legs_le_sqrt2_hypotenuse_l191_191202

theorem sum_of_legs_le_sqrt2_hypotenuse
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2) :
  a + b ≤ Real.sqrt 2 * c :=
sorry

end sum_of_legs_le_sqrt2_hypotenuse_l191_191202


namespace total_votes_l191_191499

theorem total_votes (V : ℝ) 
  (h1 : 0.5 / 100 * V = 0.005 * V) 
  (h2 : 50.5 / 100 * V = 0.505 * V) 
  (h3 : 0.505 * V - 0.005 * V = 3000) : 
  V = 6000 := 
by
  sorry

end total_votes_l191_191499


namespace population_percentage_l191_191638

-- Definitions based on the given conditions
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Conditions from the problem statement
def part_population : ℕ := 23040
def total_population : ℕ := 25600

-- The theorem stating that the percentage is 90
theorem population_percentage : percentage part_population total_population = 90 :=
  by
    -- Proof steps would go here, we only need to state the theorem
    sorry

end population_percentage_l191_191638


namespace tan_neg_1140_eq_neg_sqrt3_l191_191005

theorem tan_neg_1140_eq_neg_sqrt3 
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_periodicity : ∀ θ : ℝ, ∀ n : ℤ, Real.tan (θ + n * 180) = Real.tan θ)
  (tan_60 : Real.tan 60 = Real.sqrt 3) :
  Real.tan (-1140) = -Real.sqrt 3 := 
sorry

end tan_neg_1140_eq_neg_sqrt3_l191_191005


namespace regular_17gon_symmetries_l191_191350

theorem regular_17gon_symmetries : 
  let L := 17
  let R := 360 / 17
  L + R = 17 + 360 / 17 :=
by
  sorry

end regular_17gon_symmetries_l191_191350


namespace rectangle_area_integer_length_width_l191_191400

theorem rectangle_area_integer_length_width (l w : ℕ) (h1 : w = l / 2) (h2 : 2 * l + 2 * w = 200) :
  l * w = 2178 :=
by
  sorry

end rectangle_area_integer_length_width_l191_191400


namespace pastries_made_initially_l191_191697

theorem pastries_made_initially 
  (sold : ℕ) (remaining : ℕ) (initial : ℕ) 
  (h1 : sold = 103) (h2 : remaining = 45) : 
  initial = 148 :=
by
  have h := h1
  have r := h2
  sorry

end pastries_made_initially_l191_191697


namespace imaginary_unit_squared_in_set_l191_191424

-- Conditions of the problem
def imaginary_unit (i : ℂ) : Prop := i^2 = -1
def S : Set ℂ := {-1, 0, 1}

-- The statement to prove
theorem imaginary_unit_squared_in_set {i : ℂ} (hi : imaginary_unit i) : i^2 ∈ S := sorry

end imaginary_unit_squared_in_set_l191_191424


namespace cos_b4_b6_l191_191480

theorem cos_b4_b6 (a b : ℕ → ℝ) (d : ℝ) 
  (ha_geom : ∀ n, a (n + 1) / a n = a 1)
  (hb_arith : ∀ n, b (n + 1) = b n + d)
  (ha_prod : a 1 * a 5 * a 9 = -8)
  (hb_sum : b 2 + b 5 + b 8 = 6 * Real.pi) : 
  Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1 / 2 :=
sorry

end cos_b4_b6_l191_191480


namespace yz_sub_zx_sub_xy_l191_191645

theorem yz_sub_zx_sub_xy (x y z : ℝ) (h1 : x - y - z = 19) (h2 : x^2 + y^2 + z^2 ≠ 19) :
  yz - zx - xy = 171 := by
  sorry

end yz_sub_zx_sub_xy_l191_191645


namespace volume_of_rectangular_prism_l191_191296

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l191_191296


namespace point_B_coordinates_l191_191398

variable (A : ℝ × ℝ)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

theorem point_B_coordinates : 
  (move_left (move_up (-3, -5) 4) 3) = (-6, -1) :=
by
  sorry

end point_B_coordinates_l191_191398


namespace tan_difference_l191_191786

theorem tan_difference (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h₁ : Real.sin α = 3 / 5) (h₂ : Real.cos β = 12 / 13) : 
    Real.tan (α - β) = 16 / 63 := 
by
  sorry

end tan_difference_l191_191786


namespace solve_custom_eq_l191_191249

namespace CustomProof

def custom_mul (a b : ℕ) : ℕ := a * b + a + b

theorem solve_custom_eq (x : ℕ) (h : custom_mul 3 x = 31) : x = 7 := 
by
  sorry

end CustomProof

end solve_custom_eq_l191_191249


namespace remainder_8_pow_900_mod_29_l191_191839

theorem remainder_8_pow_900_mod_29 : 8^900 % 29 = 7 :=
by sorry

end remainder_8_pow_900_mod_29_l191_191839


namespace percentage_j_of_k_theorem_l191_191458

noncomputable def percentage_j_of_k 
  (j k l m : ℝ) (x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : Prop :=
  x = 500

theorem percentage_j_of_k_theorem 
  (j k l m : ℝ) (x : ℝ)
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100)) : percentage_j_of_k j k l m x h1 h2 h3 h4 :=
by 
  sorry

end percentage_j_of_k_theorem_l191_191458


namespace find_y_given_conditions_l191_191650

def is_value_y (x y : ℕ) : Prop :=
  (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200

theorem find_y_given_conditions : ∃ y : ℕ, ∀ x : ℕ, (100 + 200 + 300 + x) / 4 = 250 ∧ (300 + 150 + 100 + x + y) / 5 = 200 → y = 50 :=
by
  sorry

end find_y_given_conditions_l191_191650


namespace no_real_roots_range_a_l191_191538

theorem no_real_roots_range_a (a : ℝ) : (¬∃ x : ℝ, 2 * x^2 + (a - 5) * x + 2 = 0) → 1 < a ∧ a < 9 :=
by
  sorry

end no_real_roots_range_a_l191_191538


namespace ratio_of_ages_in_two_years_l191_191231

theorem ratio_of_ages_in_two_years
  (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : S = 20)
  (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l191_191231


namespace quadratic_equation_with_given_roots_l191_191002

theorem quadratic_equation_with_given_roots :
  (∃ (x : ℝ), (x - 3) * (x + 4) = 0 ↔ x = 3 ∨ x = -4) :=
by
  sorry

end quadratic_equation_with_given_roots_l191_191002


namespace max_term_of_sequence_l191_191338

def a (n : ℕ) : ℚ := (n : ℚ) / (n^2 + 156)

theorem max_term_of_sequence : ∃ n, (n = 12 ∨ n = 13) ∧ (∀ m, a m ≤ a n) := by 
  sorry

end max_term_of_sequence_l191_191338


namespace right_triangle_hypotenuse_45_deg_4_inradius_l191_191185

theorem right_triangle_hypotenuse_45_deg_4_inradius : 
  ∀ (R : ℝ) (hypotenuse_length : ℝ), R = 4 ∧ 
  (∀ (A B C : ℝ), A = 45 ∧ B = 45 ∧ C = 90) →
  hypotenuse_length = 8 :=
by
  sorry

end right_triangle_hypotenuse_45_deg_4_inradius_l191_191185


namespace factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l191_191525

-- Problem 1
theorem factorize_polynomial (x y : ℝ) : 
  x^2 - y^2 + 2*x - 2*y = (x - y)*(x + y + 2) := 
sorry

-- Problem 2
theorem triangle_equilateral (a b c : ℝ) (h : a^2 + c^2 - 2*b*(a - b + c) = 0) : 
  a = b ∧ b = c :=
sorry

-- Problem 3
theorem prove_2p_eq_m_plus_n (m n p : ℝ) (h : 1/4*(m - n)^2 = (p - n)*(m - p)) : 
  2*p = m + n :=
sorry

end factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l191_191525


namespace seungjun_clay_cost_l191_191536

theorem seungjun_clay_cost (price_per_gram : ℝ) (qty1 qty2 : ℝ) 
  (h1 : price_per_gram = 17.25) 
  (h2 : qty1 = 1000) 
  (h3 : qty2 = 10) :
  (qty1 * price_per_gram + qty2 * price_per_gram) = 17422.5 :=
by
  sorry

end seungjun_clay_cost_l191_191536


namespace value_of_expression_l191_191670

theorem value_of_expression (a b : ℝ) (h1 : a^2 + 2012 * a + 1 = 0) (h2 : b^2 + 2012 * b + 1 = 0) :
  (2 + 2013 * a + a^2) * (2 + 2013 * b + b^2) = -2010 := 
  sorry

end value_of_expression_l191_191670


namespace sum_of_legs_is_43_l191_191915

theorem sum_of_legs_is_43 (x : ℕ) (h1 : x * x + (x + 1) * (x + 1) = 31 * 31) :
  x + (x + 1) = 43 :=
sorry

end sum_of_legs_is_43_l191_191915


namespace sum_m_b_eq_neg_five_halves_l191_191969

theorem sum_m_b_eq_neg_five_halves : 
  let x1 := 1 / 2
  let y1 := -1
  let x2 := -1 / 2
  let y2 := 2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = -5 / 2 :=
by 
  sorry

end sum_m_b_eq_neg_five_halves_l191_191969


namespace quadratic_solution_l191_191360

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l191_191360


namespace angle_bisector_theorem_l191_191843

noncomputable def angle_bisector_length (a b : ℝ) (C : ℝ) (CX : ℝ) : Prop :=
  C = 120 ∧
  CX = (a * b) / (a + b)

theorem angle_bisector_theorem (a b : ℝ) (C : ℝ) (CX : ℝ) :
  angle_bisector_length a b C CX :=
by
  sorry

end angle_bisector_theorem_l191_191843


namespace fraction_fliers_afternoon_l191_191135

theorem fraction_fliers_afternoon :
  ∀ (initial_fliers remaining_fliers next_day_fliers : ℕ),
    initial_fliers = 2500 →
    next_day_fliers = 1500 →
    remaining_fliers = initial_fliers - initial_fliers / 5 →
    (remaining_fliers - next_day_fliers) / remaining_fliers = 1 / 4 :=
by
  intros initial_fliers remaining_fliers next_day_fliers
  sorry

end fraction_fliers_afternoon_l191_191135


namespace circle_equation_with_diameter_endpoints_l191_191093

theorem circle_equation_with_diameter_endpoints (A B : ℝ × ℝ) (x y : ℝ) :
  A = (1, 4) → B = (3, -2) → (x-2)^2 + (y-1)^2 = 10 :=
by
  sorry

end circle_equation_with_diameter_endpoints_l191_191093


namespace sum_invested_l191_191442

theorem sum_invested (P R: ℝ) (h1: SI₁ = P * R * 20 / 100) (h2: SI₂ = P * (R + 10) * 20 / 100) (h3: SI₂ = SI₁ + 3000) : P = 1500 :=
by
  sorry

end sum_invested_l191_191442


namespace train_crosses_bridge_in_30_seconds_l191_191647

noncomputable def train_length : ℝ := 100
noncomputable def bridge_length : ℝ := 200
noncomputable def train_speed_kmph : ℝ := 36

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def total_distance : ℝ := train_length + bridge_length

noncomputable def crossing_time : ℝ := total_distance / train_speed_mps

theorem train_crosses_bridge_in_30_seconds :
  crossing_time = 30 := 
by
  sorry

end train_crosses_bridge_in_30_seconds_l191_191647


namespace variation_of_x_l191_191564

theorem variation_of_x (k j z : ℝ) : ∃ m : ℝ, ∀ x y : ℝ, (x = k * y^2) ∧ (y = j * z^(1 / 3)) → (x = m * z^(2 / 3)) :=
sorry

end variation_of_x_l191_191564


namespace problem_l191_191505

def f (x : ℝ) : ℝ := (x^4 + 2*x^3 + 4*x - 5) ^ 2004 + 2004

theorem problem (x : ℝ) (h : x = Real.sqrt 3 - 1) : f x = 2005 :=
by
  sorry

end problem_l191_191505


namespace triangle_is_isosceles_l191_191672

theorem triangle_is_isosceles 
  (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_sin_identity : Real.sin A = 2 * Real.sin C * Real.cos B) : 
  (B = C) :=
sorry

end triangle_is_isosceles_l191_191672


namespace solve_determinant_l191_191081

-- Definitions based on the conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c

-- The problem translated to Lean 4:
theorem solve_determinant (x : ℤ) 
  (h : determinant (x + 1) x (2 * x - 6) (2 * (x - 1)) = 10) :
  x = 2 :=
sorry -- Proof is skipped

end solve_determinant_l191_191081


namespace seq_a_2012_value_l191_191486

theorem seq_a_2012_value :
  ∀ (a : ℕ → ℕ),
  (a 1 = 0) →
  (∀ n : ℕ, a (n + 1) = a n + 2 * n) →
  a 2012 = 2011 * 2012 :=
by
  intros a h₁ h₂
  sorry

end seq_a_2012_value_l191_191486


namespace area_diff_circle_square_l191_191607

theorem area_diff_circle_square (s r : ℝ) (A_square A_circle : ℝ) (d : ℝ) (pi : ℝ) 
  (h1 : d = 8) -- diagonal of the square
  (h2 : d = 2 * r) -- diameter of the circle is 8, so radius is 4
  (h3 : s^2 + s^2 = d^2) -- Pythagorean Theorem for the square
  (h4 : A_square = s^2) -- area of the square
  (h5 : A_circle = pi * r^2) -- area of the circle
  (h6 : pi = 3.14159) -- approximation for π
  : abs (A_circle - A_square) - 18.3 < 0.1 := sorry

end area_diff_circle_square_l191_191607


namespace find_number_l191_191978

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
sorry

end find_number_l191_191978


namespace ratio_of_height_to_radius_min_surface_area_l191_191199

theorem ratio_of_height_to_radius_min_surface_area 
  (r h : ℝ)
  (V : ℝ := 500)
  (volume_cond : π * r^2 * h = V)
  (surface_area : ℝ := 2 * π * r^2 + 2 * π * r * h) : 
  h / r = 2 :=
by
  sorry

end ratio_of_height_to_radius_min_surface_area_l191_191199


namespace how_many_toys_l191_191662

theorem how_many_toys (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ)
  (h1 : initial_savings = 21)
  (h2 : allowance = 15)
  (h3 : toy_cost = 6) :
  (initial_savings + allowance) / toy_cost = 6 :=
by
  sorry

end how_many_toys_l191_191662


namespace carB_distance_traveled_l191_191256

-- Define the initial conditions
def initial_separation : ℝ := 150
def distance_carA_main_road : ℝ := 25
def distance_between_cars : ℝ := 38

-- Define the question as a theorem where we need to show the distance Car B traveled
theorem carB_distance_traveled (initial_separation distance_carA_main_road distance_between_cars : ℝ) :
  initial_separation - (distance_carA_main_road + distance_between_cars) = 87 :=
  sorry

end carB_distance_traveled_l191_191256


namespace neg_P_l191_191498

/-
Proposition: There exists a natural number n such that 2^n > 1000.
-/
def P : Prop := ∃ n : ℕ, 2^n > 1000

/-
Theorem: The negation of the above proposition P is:
For all natural numbers n, 2^n ≤ 1000.
-/
theorem neg_P : ¬ P ↔ ∀ n : ℕ, 2^n ≤ 1000 :=
by
  sorry

end neg_P_l191_191498


namespace max_k_value_l191_191076

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value : ∃ k : ℤ, (∀ x > 2, k * (x - 2) < f x) ∧ k = 4 :=
by
  sorry

end max_k_value_l191_191076


namespace regular_polygon_sides_l191_191937

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 160) : n = 18 :=
by
  sorry

end regular_polygon_sides_l191_191937


namespace ball_probability_l191_191500

theorem ball_probability :
  ∀ (total_balls red_balls white_balls : ℕ),
  total_balls = 10 → red_balls = 6 → white_balls = 4 →
  -- Given conditions: Total balls, red balls, and white balls.
  -- First ball drawn is red
  ∀ (first_ball_red : true),
  -- Prove that the probability of the second ball being red is 5/9.
  (red_balls - 1) / (total_balls - 1) = 5/9 :=
by
  intros total_balls red_balls white_balls h_total h_red h_white first_ball_red
  sorry

end ball_probability_l191_191500


namespace cricket_team_rh_players_l191_191172

theorem cricket_team_rh_players (total_players throwers non_throwers lh_non_throwers rh_non_throwers rh_players : ℕ)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : non_throwers = total_players - throwers)
    (h4 : lh_non_throwers = non_throwers / 3)
    (h5 : rh_non_throwers = non_throwers - lh_non_throwers)
    (h6 : rh_players = throwers + rh_non_throwers) :
  rh_players = 51 := by
  sorry

end cricket_team_rh_players_l191_191172


namespace missy_yells_total_l191_191383

variable {O S M : ℕ}
variable (yells_at_obedient : ℕ)

-- Conditions:
def yells_stubborn (yells_at_obedient : ℕ) : ℕ := 4 * yells_at_obedient
def yells_mischievous (yells_at_obedient : ℕ) : ℕ := 2 * yells_at_obedient

-- Prove the total yells equal to 84 when yells_at_obedient = 12
theorem missy_yells_total (h : yells_at_obedient = 12) :
  yells_at_obedient + yells_stubborn yells_at_obedient + yells_mischievous yells_at_obedient = 84 :=
by
  sorry

end missy_yells_total_l191_191383


namespace students_all_three_classes_l191_191184

variables (H M E HM HE ME HME : ℕ)

-- Conditions from the problem
def student_distribution : Prop :=
  H = 12 ∧
  M = 17 ∧
  E = 36 ∧
  HM + HE + ME = 3 ∧
  86 = H + M + E - (HM + HE + ME) + HME

-- Prove the number of students registered for all three classes
theorem students_all_three_classes (h : student_distribution H M E HM HE ME HME) : HME = 24 :=
  by sorry

end students_all_three_classes_l191_191184


namespace range_of_a_l191_191114

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l191_191114


namespace sine_triangle_inequality_l191_191693

theorem sine_triangle_inequality 
  {a b c : ℝ} (h_triangle : a + b + c ≤ 2 * Real.pi) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) 
  (ha_lt_pi : a < Real.pi) (hb_lt_pi : b < Real.pi) (hc_lt_pi : c < Real.pi) :
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sine_triangle_inequality_l191_191693


namespace intersection_A_complement_B_l191_191467

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}
def comR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem intersection_A_complement_B : A ∩ (comR B) = {x | -1 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_A_complement_B_l191_191467


namespace pipe_A_fill_time_l191_191690

theorem pipe_A_fill_time :
  (∃ x : ℕ, (1 / (x : ℝ) + 1 / 60 - 1 / 72 = 1 / 40) ∧ x = 45) :=
sorry

end pipe_A_fill_time_l191_191690


namespace fraction_eval_l191_191151

theorem fraction_eval :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 :=
by
  sorry

end fraction_eval_l191_191151


namespace peggy_stamps_l191_191038

-- Defining the number of stamps Peggy, Ernie, and Bert have
variables (P : ℕ) (E : ℕ) (B : ℕ)

-- Given conditions
def bert_has_four_times_ernie (B : ℕ) (E : ℕ) : Prop := B = 4 * E
def ernie_has_three_times_peggy (E : ℕ) (P : ℕ) : Prop := E = 3 * P
def peggy_needs_stamps (P : ℕ) (B : ℕ) : Prop := B = P + 825

-- Question to Answer / Theorem Statement
theorem peggy_stamps (P : ℕ) (E : ℕ) (B : ℕ)
  (h1 : bert_has_four_times_ernie B E)
  (h2 : ernie_has_three_times_peggy E P)
  (h3 : peggy_needs_stamps P B) :
  P = 75 :=
sorry

end peggy_stamps_l191_191038


namespace min_value_inequality_l191_191555

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end min_value_inequality_l191_191555


namespace exists_numbering_for_nonagon_no_numbering_for_decagon_l191_191675

-- Definitions for the problem setup
variable (n : ℕ) 
variable (A : Fin n → Point)
variable (O : Point)

-- Definition for the numbering function
variable (f : Fin (2 * n) → ℕ)

-- First statement for n = 9
theorem exists_numbering_for_nonagon :
  ∃ (f : Fin 18 → ℕ), (∀ i : Fin 9, f (i : Fin 9) + f (i + 9) + f ((i + 1) % 9) = 15) :=
sorry

-- Second statement for n = 10
theorem no_numbering_for_decagon :
  ¬ ∃ (f : Fin 20 → ℕ), (∀ i : Fin 10, f (i : Fin 10) + f (i + 10) + f ((i + 1) % 10) = 16) :=
sorry

end exists_numbering_for_nonagon_no_numbering_for_decagon_l191_191675


namespace landscape_length_l191_191161

theorem landscape_length (b : ℝ) 
  (h1 : ∀ (l : ℝ), l = 8 * b) 
  (A : ℝ)
  (h2 : A = 8 * b^2)
  (Playground_area : ℝ)
  (h3 : Playground_area = 1200)
  (h4 : Playground_area = (1 / 6) * A) :
  ∃ (l : ℝ), l = 240 :=
by 
  sorry

end landscape_length_l191_191161


namespace total_surfers_is_60_l191_191975

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l191_191975


namespace find_initial_cards_l191_191927

theorem find_initial_cards (B : ℕ) :
  let Tim_initial := 20
  let Sarah_initial := 15
  let Tim_after_give_to_Sarah := Tim_initial - 5
  let Sarah_after_give_to_Sarah := Sarah_initial + 5
  let Tim_after_receive_from_Sarah := Tim_after_give_to_Sarah + 2
  let Sarah_after_receive_from_Sarah := Sarah_after_give_to_Sarah - 2
  let Tim_after_exchange_with_Ben := Tim_after_receive_from_Sarah - 3
  let Ben_after_exchange := B + 13
  let Ben_after_all_transactions := 3 * Tim_after_exchange_with_Ben
  Ben_after_exchange = Ben_after_all_transactions -> B = 29 := by
  sorry

end find_initial_cards_l191_191927


namespace mul_mental_math_l191_191356

theorem mul_mental_math :
  96 * 104 = 9984 := by
  sorry

end mul_mental_math_l191_191356


namespace calvin_overall_score_l191_191109

theorem calvin_overall_score :
  let test1_pct := 0.6
  let test1_total := 15
  let test2_pct := 0.85
  let test2_total := 20
  let test3_pct := 0.75
  let test3_total := 40
  let total_problems := 75

  let correct_test1 := test1_pct * test1_total
  let correct_test2 := test2_pct * test2_total
  let correct_test3 := test3_pct * test3_total
  let total_correct := correct_test1 + correct_test2 + correct_test3

  let overall_percentage := (total_correct / total_problems) * 100
  overall_percentage.round = 75 :=
sorry

end calvin_overall_score_l191_191109


namespace coffee_shop_sold_lattes_l191_191425

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l191_191425


namespace min_value_expression_l191_191276

theorem min_value_expression (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 4 * a + b = 1) :
  (1 / a) + (4 / b) = 16 := sorry

end min_value_expression_l191_191276


namespace total_number_of_tiles_l191_191954

theorem total_number_of_tiles {s : ℕ} 
  (h1 : ∃ s : ℕ, (s^2 - 4*s + 896 = 0))
  (h2 : 225 = 2*s - 1 + s^2 / 4 - s / 2) :
  s^2 = 1024 := by
  sorry

end total_number_of_tiles_l191_191954


namespace count_paths_COMPUTER_l191_191055

theorem count_paths_COMPUTER : 
  let possible_paths (n : ℕ) := 2 ^ n 
  possible_paths 7 + possible_paths 7 + 1 = 257 :=
by sorry

end count_paths_COMPUTER_l191_191055


namespace largest_angle_of_convex_hexagon_l191_191235

theorem largest_angle_of_convex_hexagon (a d : ℕ) (h_seq : ∀ i, a + i * d < 180 ∧ a + i * d > 0)
  (h_sum : 6 * a + 15 * d = 720)
  (h_seq_arithmetic : ∀ (i j : ℕ), (a + i * d) < (a + j * d) ↔ i < j) :
  ∃ m : ℕ, (m = a + 5 * d ∧ m = 175) :=
by
  sorry

end largest_angle_of_convex_hexagon_l191_191235


namespace milk_left_in_storage_l191_191158

-- Define initial and rate conditions
def initialMilk : ℕ := 30000
def pumpedRate : ℕ := 2880
def pumpedHours : ℕ := 4
def addedRate : ℕ := 1500
def addedHours : ℕ := 7

-- The proof problem: Prove the final amount in storage tank == 28980 gallons
theorem milk_left_in_storage : 
  initialMilk - (pumpedRate * pumpedHours) + (addedRate * addedHours) = 28980 := 
sorry

end milk_left_in_storage_l191_191158


namespace problem_solution_l191_191900

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x < -6 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 ))
  (h2 : a < b)
  : a + 2 * b + 3 * c = 74 := 
sorry

end problem_solution_l191_191900


namespace quadratic_sequence_l191_191678

theorem quadratic_sequence (a x₁ b x₂ c : ℝ)
  (h₁ : a + b = 2 * x₁)
  (h₂ : x₁ + x₂ = 2 * b)
  (h₃ : a + c = 2 * b)
  (h₄ : x₁ + x₂ = -6 / a)
  (h₅ : x₁ * x₂ = c / a) :
  b = -2 * a ∧ c = -5 * a :=
by
  sorry

end quadratic_sequence_l191_191678


namespace find_m_l191_191464

theorem find_m
  (θ : Real)
  (m : Real)
  (h_sin_cos_roots : ∀ x : Real, 4 * x^2 + 2 * m * x + m = 0 → x = Real.sin θ ∨ x = Real.cos θ)
  (h_real_roots : ∃ x : Real, 4 * x^2 + 2 * m * x + m = 0) :
  m = 1 - Real.sqrt 5 :=
sorry

end find_m_l191_191464


namespace car_robot_collections_l191_191648

variable (t m b s j : ℕ)

axiom tom_has_15 : t = 15
axiom michael_robots : m = 3 * t - 5
axiom bob_robots : b = 8 * (t + m)
axiom sarah_robots : s = b / 2 - 7
axiom jane_robots : j = (s - t) / 3

theorem car_robot_collections :
  t = 15 ∧
  m = 40 ∧
  b = 440 ∧
  s = 213 ∧
  j = 66 :=
  by
    sorry

end car_robot_collections_l191_191648


namespace intersection_with_y_axis_l191_191676

theorem intersection_with_y_axis (y : ℝ) : 
  (∃ y, (0, y) ∈ {(x, 2 * x + 4) | x : ℝ}) ↔ y = 4 :=
by 
  sorry

end intersection_with_y_axis_l191_191676


namespace shaded_area_ratio_l191_191484

-- Definitions based on conditions
def large_square_area : ℕ := 16
def shaded_components : ℕ := 4
def component_fraction : ℚ := 1 / 2
def shaded_square_area : ℚ := shaded_components * component_fraction
def large_square_area_q : ℚ := large_square_area

-- Goal statement
theorem shaded_area_ratio : (shaded_square_area / large_square_area_q) = (1 / 8) :=
by sorry

end shaded_area_ratio_l191_191484


namespace divisibility_by_n5_plus_1_l191_191344

theorem divisibility_by_n5_plus_1 (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n^5 + 1 ∣ (n^4 - 1) * (n^3 - n^2 + n - 1)^k + (n + 1) * n^(4 * k - 1) :=
sorry

end divisibility_by_n5_plus_1_l191_191344


namespace fraction_cows_sold_is_one_fourth_l191_191597

def num_cows : ℕ := 184
def num_dogs (C : ℕ) : ℕ := C / 2
def remaining_animals : ℕ := 161
def fraction_dogs_sold : ℚ := 3 / 4
def fraction_cows_sold (C remaining_cows : ℕ) : ℚ := (C - remaining_cows) / C

theorem fraction_cows_sold_is_one_fourth :
  ∀ (C remaining_dogs remaining_cows: ℕ),
    C = 184 →
    remaining_animals = 161 →
    remaining_dogs = (1 - fraction_dogs_sold) * num_dogs C →
    remaining_cows = remaining_animals - remaining_dogs →
    fraction_cows_sold C remaining_cows = 1 / 4 :=
by sorry

end fraction_cows_sold_is_one_fourth_l191_191597


namespace prime_cannot_be_sum_of_three_squares_l191_191089

theorem prime_cannot_be_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
by
  sorry

end prime_cannot_be_sum_of_three_squares_l191_191089


namespace fruit_basket_combinations_l191_191447

namespace FruitBasket

def apples := 3
def oranges := 8
def min_apples := 1
def min_oranges := 1

theorem fruit_basket_combinations : 
  (apples + 1 - min_apples) * (oranges + 1 - min_oranges) = 36 := by
  sorry

end FruitBasket

end fruit_basket_combinations_l191_191447


namespace max_ab_bc_cd_l191_191045

-- Definitions of nonnegative numbers and their sum condition
variables (a b c d : ℕ) 
variables (h_sum : a + b + c + d = 120)

-- The goal to prove
theorem max_ab_bc_cd : ab + bc + cd <= 3600 :=
sorry

end max_ab_bc_cd_l191_191045


namespace race_distance_l191_191818

theorem race_distance {d a b c : ℝ} 
    (h1 : d / a = (d - 25) / b)
    (h2 : d / b = (d - 15) / c)
    (h3 : d / a = (d - 35) / c) :
  d = 75 :=
by
  sorry

end race_distance_l191_191818


namespace smallest_possible_w_l191_191448

theorem smallest_possible_w 
  (h1 : 936 = 2^3 * 3 * 13)
  (h2 : 2^5 = 32)
  (h3 : 3^3 = 27)
  (h4 : 14^2 = 196) :
  ∃ w : ℕ, (w > 0) ∧ (936 * w) % 32 = 0 ∧ (936 * w) % 27 = 0 ∧ (936 * w) % 196 = 0 ∧ w = 1764 :=
sorry

end smallest_possible_w_l191_191448


namespace find_coefficients_l191_191320

theorem find_coefficients (a b : ℚ) (h_a_nonzero : a ≠ 0)
  (h_prod : (3 * b - 2 * a = 0) ∧ (-2 * b + 3 = 0)) : 
  a = 9 / 4 ∧ b = 3 / 2 :=
by
  sorry

end find_coefficients_l191_191320


namespace greatest_multiple_of_4_l191_191049

/-- 
Given x is a positive multiple of 4 and x^3 < 2000, 
prove that x is at most 12 and 
x = 12 is the greatest value that satisfies these conditions. 
-/
theorem greatest_multiple_of_4 (x : ℕ) (hx1 : x % 4 = 0) (hx2 : x^3 < 2000) : x ≤ 12 ∧ x = 12 :=
by
  sorry

end greatest_multiple_of_4_l191_191049


namespace absolute_value_inequality_l191_191187

theorem absolute_value_inequality (x : ℝ) : ¬ (|x - 3| + |x + 4| < 6) :=
sorry

end absolute_value_inequality_l191_191187


namespace percent_difference_z_w_l191_191716

theorem percent_difference_z_w (w x y z : ℝ)
  (h1 : w = 0.60 * x)
  (h2 : x = 0.60 * y)
  (h3 : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
sorry

end percent_difference_z_w_l191_191716


namespace quadratic_to_square_l191_191516

theorem quadratic_to_square (x h k : ℝ) : 
  (x * x - 4 * x + 3 = 0) →
  ((x + h) * (x + h) = k) →
  k = 1 :=
by
  sorry

end quadratic_to_square_l191_191516


namespace inequality_proof_l191_191324

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end inequality_proof_l191_191324


namespace sum_of_p_q_r_s_t_l191_191418

theorem sum_of_p_q_r_s_t (p q r s t : ℤ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120) : 
  p + q + r + s + t = 32 := 
sorry

end sum_of_p_q_r_s_t_l191_191418


namespace series_sum_eq_half_l191_191462

theorem series_sum_eq_half :
  ∑' (n : ℕ), 2^n / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_sum_eq_half_l191_191462


namespace avocados_per_serving_l191_191032

-- Definitions for the conditions
def original_avocados : ℕ := 5
def additional_avocados : ℕ := 4
def total_avocados : ℕ := original_avocados + additional_avocados
def servings : ℕ := 3

-- Theorem stating the result
theorem avocados_per_serving : (total_avocados / servings) = 3 :=
by
  sorry

end avocados_per_serving_l191_191032


namespace chipped_marbles_is_22_l191_191307

def bags : List ℕ := [20, 22, 25, 30, 32, 34, 36]

-- Jane and George take some bags and one bag with chipped marbles is left.
theorem chipped_marbles_is_22
  (h1 : ∃ (jane_bags george_bags : List ℕ) (remaining_bag : ℕ),
    (jane_bags ++ george_bags ++ [remaining_bag] = bags ∧
     jane_bags.length = 3 ∧
     (george_bags.length = 2 ∨ george_bags.length = 3) ∧
     3 * remaining_bag = List.sum jane_bags + List.sum george_bags)) :
  ∃ (c : ℕ), c = 22 := 
sorry

end chipped_marbles_is_22_l191_191307


namespace reciprocal_inequalities_l191_191326

theorem reciprocal_inequalities (a b c : ℝ)
  (h1 : -1 < a ∧ a < -2/3)
  (h2 : -1/3 < b ∧ b < 0)
  (h3 : 1 < c) :
  1/c < 1/(b - a) ∧ 1/(b - a) < 1/(a * b) :=
by
  sorry

end reciprocal_inequalities_l191_191326


namespace number_of_terms_l191_191421

noncomputable def Sn (n : ℕ) : ℝ := sorry

def an_arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem number_of_terms {a : ℕ → ℝ}
  (h_arith : an_arithmetic_seq a)
  (cond1 : a 1 + a 2 + a 3 + a 4 = 1)
  (cond2 : a 5 + a 6 + a 7 + a 8 = 2)
  (cond3 : Sn = 15) :
  ∃ n, n = 16 :=
sorry

end number_of_terms_l191_191421


namespace negative_two_squared_l191_191753

theorem negative_two_squared :
  (-2 : ℤ)^2 = 4 := 
sorry

end negative_two_squared_l191_191753


namespace michael_made_small_balls_l191_191006

def num_small_balls (total_bands : ℕ) (bands_per_small : ℕ) (bands_per_large : ℕ) (num_large : ℕ) : ℕ :=
  (total_bands - num_large * bands_per_large) / bands_per_small

theorem michael_made_small_balls :
  num_small_balls 5000 50 300 13 = 22 :=
by
  sorry

end michael_made_small_balls_l191_191006


namespace find_number_of_cups_l191_191955

theorem find_number_of_cups (a C B : ℝ) (h1 : a * C + 2 * B = 12.75) (h2 : 2 * C + 5 * B = 14.00) (h3 : B = 1.5) : a = 3 :=
by
  sorry

end find_number_of_cups_l191_191955


namespace express_in_scientific_notation_l191_191509

theorem express_in_scientific_notation : (0.0000028 = 2.8 * 10^(-6)) :=
sorry

end express_in_scientific_notation_l191_191509


namespace total_distance_is_10_miles_l191_191179

noncomputable def total_distance_back_to_town : ℕ :=
  let distance1 := 3
  let distance2 := 3
  let distance3 := 4
  distance1 + distance2 + distance3

theorem total_distance_is_10_miles :
  total_distance_back_to_town = 10 :=
by
  sorry

end total_distance_is_10_miles_l191_191179


namespace correct_calculation_l191_191427

theorem correct_calculation : -Real.sqrt ((-5)^2) = -5 := 
by 
  sorry

end correct_calculation_l191_191427


namespace proof_problem_l191_191346

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / x^3

theorem proof_problem {x0 m n : ℝ} (hx0_pos : 0 < x0)
  (H : f'' x0 = 0) (hm : 0 < m) (hmx0 : m < x0) (hn : x0 < n) :
  f'' m < 0 ∧ f'' n > 0 := sorry

end proof_problem_l191_191346


namespace value_at_x12_l191_191919

def quadratic_function (d e f x : ℝ) : ℝ :=
  d * x^2 + e * x + f

def axis_of_symmetry (d e f : ℝ) : ℝ := 10.5

def point_on_graph (d e f : ℝ) : Prop :=
  quadratic_function d e f 3 = -5

theorem value_at_x12 (d e f : ℝ)
  (Hsymm : axis_of_symmetry d e f = 10.5)
  (Hpoint : point_on_graph d e f) :
  quadratic_function d e f 12 = -5 :=
sorry

end value_at_x12_l191_191919


namespace sun_salutations_per_year_l191_191774

theorem sun_salutations_per_year :
  let poses_per_day := 5
  let days_per_week := 5
  let weeks_per_year := 52
  poses_per_day * days_per_week * weeks_per_year = 1300 :=
by
  sorry

end sun_salutations_per_year_l191_191774


namespace cubed_expression_l191_191800

theorem cubed_expression (a : ℝ) (h : (a + 1/a)^2 = 4) : a^3 + 1/a^3 = 2 ∨ a^3 + 1/a^3 = -2 :=
sorry

end cubed_expression_l191_191800


namespace problem1_solution_problem2_solution_l191_191908

-- Problem 1: System of Equations
theorem problem1_solution (x y : ℝ) (h_eq1 : x - y = 2) (h_eq2 : 2 * x + y = 7) : x = 3 ∧ y = 1 :=
by {
  sorry -- Proof to be filled in
}

-- Problem 2: Fractional Equation
theorem problem2_solution (y : ℝ) (h_eq : 3 / (1 - y) = y / (y - 1) - 5) : y = 2 :=
by {
  sorry -- Proof to be filled in
}

end problem1_solution_problem2_solution_l191_191908


namespace field_length_to_width_ratio_l191_191097
-- Import the math library

-- Define the problem conditions and proof goal statement
theorem field_length_to_width_ratio (w : ℝ) (l : ℝ) (area_pond : ℝ) (area_field : ℝ) 
    (h_length : l = 16) (h_area_pond : area_pond = 64) 
    (h_area_relation : area_pond = (1/2) * area_field)
    (h_field_area : area_field = l * w) : l / w = 2 :=
by 
  -- Leaving the proof as an exercise
  sorry

end field_length_to_width_ratio_l191_191097


namespace count_and_largest_special_numbers_l191_191039

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000

theorem count_and_largest_special_numbers :
  ∃ (nums : List ℕ), 
    (∀ n ∈ nums, ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ 
      55 * x * y = n ∧ is_four_digit_number (n * 5))
    ∧ nums.length = 3
    ∧ nums.maximum = some 4785 :=
sorry

end count_and_largest_special_numbers_l191_191039


namespace more_pencils_than_pens_l191_191663

theorem more_pencils_than_pens : 
  ∀ (P L : ℕ), L = 30 → (P / L: ℚ) = 5 / 6 → ((L - P) = 5) := by
  intros P L hL hRatio
  sorry

end more_pencils_than_pens_l191_191663


namespace inequality_2_pow_n_gt_n_sq_for_n_5_l191_191730

theorem inequality_2_pow_n_gt_n_sq_for_n_5 : 2^5 > 5^2 := 
by {
    sorry -- Placeholder for the proof
}

end inequality_2_pow_n_gt_n_sq_for_n_5_l191_191730


namespace moscow_probability_higher_l191_191575

def total_combinations : ℕ := 64 * 63

def invalid_combinations_ural : ℕ := 8 * 7 + 8 * 7

def valid_combinations_moscow : ℕ := total_combinations

def valid_combinations_ural : ℕ := total_combinations - invalid_combinations_ural

def probability_moscow : ℚ := valid_combinations_moscow / total_combinations

def probability_ural : ℚ := valid_combinations_ural / total_combinations

theorem moscow_probability_higher :
  probability_moscow > probability_ural :=
by
  unfold probability_moscow probability_ural
  unfold valid_combinations_moscow valid_combinations_ural invalid_combinations_ural total_combinations
  sorry

end moscow_probability_higher_l191_191575


namespace triangles_in_extended_figure_l191_191682

theorem triangles_in_extended_figure : 
  ∀ (row1_tri : ℕ) (row2_tri : ℕ) (row3_tri : ℕ) (row4_tri : ℕ) 
  (row1_2_med_tri : ℕ) (row2_3_med_tri : ℕ) (row3_4_med_tri : ℕ) 
  (large_tri : ℕ), 
  row1_tri = 6 →
  row2_tri = 5 →
  row3_tri = 4 →
  row4_tri = 3 →
  row1_2_med_tri = 5 →
  row2_3_med_tri = 2 →
  row3_4_med_tri = 1 →
  large_tri = 1 →
  row1_tri + row2_tri + row3_tri + row4_tri
  + row1_2_med_tri + row2_3_med_tri + row3_4_med_tri
  + large_tri = 27 :=
by
  intro row1_tri row2_tri row3_tri row4_tri
  intro row1_2_med_tri row2_3_med_tri row3_4_med_tri
  intro large_tri
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end triangles_in_extended_figure_l191_191682


namespace max_a_plus_b_l191_191934

/-- Given real numbers a and b such that 5a + 3b <= 11 and 3a + 6b <= 12,
    the largest possible value of a + b is 23/9. -/
theorem max_a_plus_b (a b : ℝ) (h1 : 5 * a + 3 * b ≤ 11) (h2 : 3 * a + 6 * b ≤ 12) :
  a + b ≤ 23 / 9 :=
sorry

end max_a_plus_b_l191_191934


namespace minimize_y_l191_191835

def y (x a b : ℝ) : ℝ := (x-a)^2 * (x-b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, y x a b = 0 := by
  use a
  sorry

end minimize_y_l191_191835


namespace relationship_a_b_l191_191168

noncomputable def e : ℝ := Real.exp 1

theorem relationship_a_b
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : e^a + 2 * a = e^b + 3 * b) :
  a > b :=
sorry

end relationship_a_b_l191_191168


namespace ratio_of_apple_to_orange_cost_l191_191639

-- Define the costs of fruits based on the given conditions.
def cost_per_kg_oranges : ℝ := 12
def cost_per_kg_apples : ℝ := 2

-- The theorem to prove.
theorem ratio_of_apple_to_orange_cost : cost_per_kg_apples / cost_per_kg_oranges = 1 / 6 :=
by
  sorry

end ratio_of_apple_to_orange_cost_l191_191639


namespace g_extreme_points_product_inequality_l191_191075

noncomputable def f (a x : ℝ) : ℝ := (-x^2 + a * x - a) / Real.exp x

noncomputable def f' (a x : ℝ) : ℝ := (x^2 - (a + 2) * x + 2 * a) / Real.exp x

noncomputable def g (a x : ℝ) : ℝ := (f a x + f' a x) / (x - 1)

theorem g_extreme_points_product_inequality {a x1 x2 : ℝ} 
  (h_cond1 : a > 2)
  (h_cond2 : x1 + x2 = (a + 2) / 2)
  (h_cond3 : x1 * x2 = 1)
  (h_cond4 : x1 ≠ 1 ∧ x2 ≠ 1)
  (h_x1 : x1 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_x2 : x2 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  g a x1 * g a x2 < 4 / Real.exp 2 :=
sorry

end g_extreme_points_product_inequality_l191_191075
