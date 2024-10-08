import Mathlib

namespace marla_drive_time_l158_158850

theorem marla_drive_time (x : ℕ) (h_total : x + 70 + x = 110) : x = 20 :=
sorry

end marla_drive_time_l158_158850


namespace arithmetic_sequence_fifth_term_l158_158568

theorem arithmetic_sequence_fifth_term:
  ∀ (a₁ aₙ : ℕ) (n : ℕ),
    n = 20 → a₁ = 2 → aₙ = 59 →
    ∃ d a₅, d = (59 - 2) / (20 - 1) ∧ a₅ = 2 + (5 - 1) * d ∧ a₅ = 14 :=
by
  sorry

end arithmetic_sequence_fifth_term_l158_158568


namespace benny_spent_amount_l158_158139

-- Definitions based on given conditions
def initial_amount : ℕ := 79
def amount_left : ℕ := 32

-- Proof problem statement
theorem benny_spent_amount :
  initial_amount - amount_left = 47 :=
sorry

end benny_spent_amount_l158_158139


namespace Jerry_age_l158_158569

theorem Jerry_age (M J : ℕ) (h1 : M = 2 * J - 6) (h2 : M = 22) : J = 14 :=
by
  sorry

end Jerry_age_l158_158569


namespace problem_inequality_l158_158644

theorem problem_inequality 
  (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) : 
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
by
  sorry

end problem_inequality_l158_158644


namespace trapezium_area_example_l158_158242

noncomputable def trapezium_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

theorem trapezium_area_example :
  trapezium_area 20 18 16 = 304 :=
by
  -- The proof steps would go here, but we're skipping them.
  sorry

end trapezium_area_example_l158_158242


namespace junior_average_score_l158_158477

def total_students : ℕ := 20
def proportion_juniors : ℝ := 0.2
def proportion_seniors : ℝ := 0.8
def average_class_score : ℝ := 78
def average_senior_score : ℝ := 75

theorem junior_average_score :
  let num_juniors := total_students * proportion_juniors
  let num_seniors := total_students * proportion_seniors
  let total_score := total_students * average_class_score
  let total_senior_score := num_seniors * average_senior_score
  let total_junior_score := total_score - total_senior_score
  total_junior_score / num_juniors = 90 := 
by
  sorry

end junior_average_score_l158_158477


namespace equal_money_distribution_l158_158094

theorem equal_money_distribution (y : ℝ) : 
  ∃ z : ℝ, z = 0.1 * (1.25 * y) ∧ (1.25 * y) - z = y + z - y :=
by
  sorry

end equal_money_distribution_l158_158094


namespace no_more_than_four_intersection_points_l158_158236

noncomputable def conic1 (a b c d e f : ℝ) (x y : ℝ) : Prop := 
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

noncomputable def conic2_param (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

theorem no_more_than_four_intersection_points (a b c d e f : ℝ)
  (P Q A : ℝ → ℝ) :
  (∃ t1 t2 t3 t4 t5,
    conic1 a b c d e f (P t1 / A t1) (Q t1 / A t1) ∧
    conic1 a b c d e f (P t2 / A t2) (Q t2 / A t2) ∧
    conic1 a b c d e f (P t3 / A t3) (Q t3 / A t3) ∧
    conic1 a b c d e f (P t4 / A t4) (Q t4 / A t4) ∧
    conic1 a b c d e f (P t5 / A t5) (Q t5 / A t5)) → false :=
sorry

end no_more_than_four_intersection_points_l158_158236


namespace rowing_time_from_A_to_B_and_back_l158_158961

-- Define the problem parameters and conditions
def rowing_speed_still_water : ℝ := 5
def distance_AB : ℝ := 12
def stream_speed : ℝ := 1

-- Define the problem to prove
theorem rowing_time_from_A_to_B_and_back :
  let downstream_speed := rowing_speed_still_water + stream_speed
  let upstream_speed := rowing_speed_still_water - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := distance_AB / upstream_speed
  let total_time := time_downstream + time_upstream
  total_time = 5 :=
by
  sorry

end rowing_time_from_A_to_B_and_back_l158_158961


namespace unique_flavors_l158_158638

noncomputable def distinctFlavors : Nat :=
  let redCandies := 5
  let greenCandies := 4
  let blueCandies := 2
  (90 - 15 - 18 - 30 + 3 + 5 + 6) / 3  -- Adjustments and consideration for equivalent ratios.
  
theorem unique_flavors :
  distinctFlavors = 11 :=
  by
    sorry

end unique_flavors_l158_158638


namespace percentage_of_students_wearing_red_shirts_l158_158046

/-- In a school of 700 students:
    - 45% of students wear blue shirts.
    - 15% of students wear green shirts.
    - 119 students wear shirts of other colors.
    We are proving that the percentage of students wearing red shirts is 23%. --/
theorem percentage_of_students_wearing_red_shirts:
  let total_students := 700
  let blue_shirt_percentage := 45 / 100
  let green_shirt_percentage := 15 / 100
  let other_colors_students := 119
  let students_with_blue_shirts := blue_shirt_percentage * total_students
  let students_with_green_shirts := green_shirt_percentage * total_students
  let students_with_other_colors := other_colors_students
  let students_with_blue_green_or_red_shirts := total_students - students_with_other_colors
  let students_with_red_shirts := students_with_blue_green_or_red_shirts - students_with_blue_shirts - students_with_green_shirts
  (students_with_red_shirts / total_students) * 100 = 23 := by
  sorry

end percentage_of_students_wearing_red_shirts_l158_158046


namespace smallest_whole_number_l158_158298

theorem smallest_whole_number :
  ∃ x : ℕ, x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 ∧ x = 23 :=
sorry

end smallest_whole_number_l158_158298


namespace inequality_solution_set_l158_158503

theorem inequality_solution_set (x : ℝ) :
  (1 / |x - 1| > 3 / 2) ↔ (1 / 3 < x ∧ x < 5 / 3 ∧ x ≠ 1) :=
by
  sorry

end inequality_solution_set_l158_158503


namespace books_from_library_l158_158403

def initial_books : ℝ := 54.5
def additional_books_1 : ℝ := 23.7
def returned_books_1 : ℝ := 12.3
def additional_books_2 : ℝ := 15.6
def returned_books_2 : ℝ := 9.1
def additional_books_3 : ℝ := 7.2

def total_books : ℝ :=
  initial_books + additional_books_1 - returned_books_1 + additional_books_2 - returned_books_2 + additional_books_3

theorem books_from_library : total_books = 79.6 := by
  sorry

end books_from_library_l158_158403


namespace ages_proof_l158_158463

def hans_now : ℕ := 8

def sum_ages (annika_now emil_now frida_now : ℕ) :=
  hans_now + annika_now + emil_now + frida_now = 58

def annika_age_in_4_years (annika_now : ℕ) : ℕ :=
  3 * (hans_now + 4)

def emil_age_in_4_years (emil_now : ℕ) : ℕ :=
  2 * (hans_now + 4)

def frida_age_in_4_years (frida_now : ℕ) :=
  2 * 12

def annika_frida_age_difference (annika_now frida_now : ℕ) : Prop :=
  annika_now = frida_now + 5

theorem ages_proof :
  ∃ (annika_now emil_now frida_now : ℕ),
    sum_ages annika_now emil_now frida_now ∧
    annika_age_in_4_years annika_now = 36 ∧
    emil_age_in_4_years emil_now = 24 ∧
    frida_age_in_4_years frida_now = 24 ∧
    annika_frida_age_difference annika_now frida_now :=
by
  sorry

end ages_proof_l158_158463


namespace solve_for_x_l158_158567

theorem solve_for_x :
  ∀ x : ℝ, 4 * x + 9 * x = 360 - 9 * (x - 4) → x = 18 :=
by
  intros x h
  sorry

end solve_for_x_l158_158567


namespace solid_views_same_shape_and_size_l158_158756

theorem solid_views_same_shape_and_size (solid : Type) (sphere triangular_pyramid cube cylinder : solid)
  (views_same_shape_and_size : solid → Bool) : 
  views_same_shape_and_size cylinder = false :=
sorry

end solid_views_same_shape_and_size_l158_158756


namespace part_I_part_II_l158_158806

-- Condition definitions:
def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

-- Part I: Prove m = 1
theorem part_I (m : ℝ) : (∀ x : ℝ, f (x + 2) m ≥ 0) ↔ m = 1 :=
by
  sorry

-- Part II: Prove a + 2b + 3c ≥ 9
theorem part_II (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : a + 2 * b + 3 * c ≥ 9 :=
by
  sorry

end part_I_part_II_l158_158806


namespace consecutive_integers_l158_158921

theorem consecutive_integers (a b c : ℝ)
  (h1 : ∃ k : ℤ, a + b = k ∧ b + c = k + 1 ∧ c + a = k + 2)
  (h2 : ∃ k : ℤ, b + c = 2 * k + 1) :
  ∃ n : ℤ, a = n + 2 ∧ b = n + 1 ∧ c = n := 
sorry

end consecutive_integers_l158_158921


namespace intersection_points_count_l158_158470

-- Definition of the two equations as conditions
def eq1 (x y : ℝ) : Prop := y = 3 * x^2
def eq2 (x y : ℝ) : Prop := y^2 - 6 * y + 8 = x^2

-- The theorem stating that the number of intersection points of the two graphs is exactly 4
theorem intersection_points_count : 
  ∃ (points : Finset (ℝ × ℝ)), (∀ p : ℝ × ℝ, p ∈ points ↔ eq1 p.1 p.2 ∧ eq2 p.1 p.2) ∧ points.card = 4 :=
by
  sorry

end intersection_points_count_l158_158470


namespace sum_of_digits_l158_158443

theorem sum_of_digits (a b c d : ℕ) (h_diff : ∀ x y : ℕ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) (h1 : a + c = 10) (h2 : b + c = 8) (h3 : a + d = 11) : 
  a + b + c + d = 18 :=
by
  sorry

end sum_of_digits_l158_158443


namespace correct_choice_is_C_l158_158648

-- Define the proposition C.
def prop_C : Prop := ∃ x : ℝ, |x - 1| < 0

-- The problem statement in Lean 4.
theorem correct_choice_is_C : ¬ prop_C :=
by
  sorry

end correct_choice_is_C_l158_158648


namespace line_through_origin_and_intersection_eq_x_y_l158_158301

theorem line_through_origin_and_intersection_eq_x_y :
  ∀ (x y : ℝ), (x - 2 * y + 2 = 0) ∧ (2 * x - y - 2 = 0) →
  ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ (y = m * x + b) :=
by
  sorry

end line_through_origin_and_intersection_eq_x_y_l158_158301


namespace sale_price_for_50_percent_profit_l158_158867

theorem sale_price_for_50_percent_profit
  (C L: ℝ)
  (h1: 892 - C = C - L)
  (h2: 1005 = 1.5 * C) :
  1.5 * C = 1005 :=
by
  sorry

end sale_price_for_50_percent_profit_l158_158867


namespace problem_statement_l158_158058

-- Definitions of A and B based on the given conditions
def A : ℤ := -5 * -3
def B : ℤ := 2 - 2

-- The theorem stating that A + B = 15
theorem problem_statement : A + B = 15 := 
by 
  sorry

end problem_statement_l158_158058


namespace solve_problem_l158_158680

theorem solve_problem (nabla odot : ℕ) 
  (h1 : 0 < nabla) 
  (h2 : nabla < 20) 
  (h3 : 0 < odot) 
  (h4 : odot < 20) 
  (h5 : nabla ≠ odot) 
  (h6 : nabla * nabla * nabla = nabla) : 
  nabla * nabla = 64 :=
by
  sorry

end solve_problem_l158_158680


namespace micheal_item_count_l158_158929

theorem micheal_item_count : ∃ a b c : ℕ, a + b + c = 50 ∧ 60 * a + 500 * b + 400 * c = 10000 ∧ a = 30 :=
  by
    sorry

end micheal_item_count_l158_158929


namespace shortest_side_of_triangle_with_medians_l158_158088

noncomputable def side_lengths_of_triangle_with_medians (a b c m_a m_b m_c : ℝ) : Prop :=
  m_a = 3 ∧ m_b = 4 ∧ m_c = 5 →
  a^2 = 2*b^2 + 2*c^2 - 36 ∧
  b^2 = 2*a^2 + 2*c^2 - 64 ∧
  c^2 = 2*a^2 + 2*b^2 - 100

theorem shortest_side_of_triangle_with_medians :
  ∀ (a b c : ℝ), side_lengths_of_triangle_with_medians a b c 3 4 5 → 
  min a (min b c) = c :=
sorry

end shortest_side_of_triangle_with_medians_l158_158088


namespace harry_started_with_79_l158_158561

-- Definitions using the conditions
def harry_initial_apples (x : ℕ) : Prop :=
  (x + 5 = 84)

-- Theorem statement proving the initial number of apples Harry started with
theorem harry_started_with_79 : ∃ x : ℕ, harry_initial_apples x ∧ x = 79 :=
by
  sorry

end harry_started_with_79_l158_158561


namespace find_number_l158_158893

theorem find_number (n : ℕ) : (n / 2) + 5 = 15 → n = 20 :=
by
  intro h
  sorry

end find_number_l158_158893


namespace value_of_x_l158_158238

def is_whole_number (n : ℝ) : Prop := ∃ (k : ℤ), n = k

theorem value_of_x (n : ℝ) (x : ℝ) :
  n = 1728 →
  is_whole_number (Real.log n / Real.log x + Real.log n / Real.log 12) →
  x = 12 :=
by
  intro h₁ h₂
  sorry

end value_of_x_l158_158238


namespace sum_of_consecutive_integers_exists_l158_158028

theorem sum_of_consecutive_integers_exists : 
  ∃ k : ℕ, 150 * k + 11325 = 5827604250 :=
by
  sorry

end sum_of_consecutive_integers_exists_l158_158028


namespace sqrt_of_six_l158_158173

theorem sqrt_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end sqrt_of_six_l158_158173


namespace second_less_than_first_third_less_than_first_l158_158576

variable (X : ℝ)

def first_number : ℝ := 0.70 * X
def second_number : ℝ := 0.63 * X
def third_number : ℝ := 0.59 * X

theorem second_less_than_first : 
  ((first_number X - second_number X) / first_number X * 100) = 10 :=
by
  sorry

theorem third_less_than_first : 
  ((third_number X - first_number X) / first_number X * 100) = -15.71 :=
by
  sorry

end second_less_than_first_third_less_than_first_l158_158576


namespace total_legs_in_room_l158_158389

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l158_158389


namespace intersection_of_A_and_B_l158_158837

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l158_158837


namespace reaction_completion_l158_158223

-- Definitions from conditions
def NaOH_moles : ℕ := 2
def H2O_moles : ℕ := 2

-- Given the balanced equation
-- 2 NaOH + H2SO4 → Na2SO4 + 2 H2O

theorem reaction_completion (H2SO4_moles : ℕ) :
  (2 * (NaOH_moles / 2)) = H2O_moles → H2SO4_moles = 1 :=
by 
  -- Skip proof
  sorry

end reaction_completion_l158_158223


namespace max_height_piston_l158_158013

theorem max_height_piston (M a P c_v g R: ℝ) (h : ℝ) 
  (h_pos : 0 < h) (M_pos : 0 < M) (a_pos : 0 < a) (P_pos : 0 < P)
  (c_v_pos : 0 < c_v) (g_pos : 0 < g) (R_pos : 0 < R) :
  h = (2 * P ^ 2) / (M ^ 2 * g * a ^ 2 * (1 + c_v / R) ^ 2) := sorry

end max_height_piston_l158_158013


namespace hyperbola_asymptotes_l158_158729

theorem hyperbola_asymptotes (p : ℝ) (h : (p / 2, 0) ∈ {q : ℝ × ℝ | q.1 ^ 2 / 8 - q.2 ^ 2 / p = 1}) :
  (y = x) ∨ (y = -x) :=
by
  sorry

end hyperbola_asymptotes_l158_158729


namespace minimum_people_who_like_both_l158_158412

open Nat

theorem minimum_people_who_like_both (total : ℕ) (mozart : ℕ) (bach : ℕ)
  (h_total: total = 100) (h_mozart: mozart = 87) (h_bach: bach = 70) :
  ∃ x, x = mozart + bach - total ∧ x ≥ 57 :=
by
  sorry

end minimum_people_who_like_both_l158_158412


namespace emily_cleaning_time_l158_158464

noncomputable def total_time : ℝ := 8 -- total time in hours
noncomputable def lilly_fiona_time : ℝ := 1/4 * total_time -- Lilly and Fiona's combined time in hours
noncomputable def jack_time : ℝ := 1/3 * total_time -- Jack's time in hours
noncomputable def emily_time : ℝ := total_time - lilly_fiona_time - jack_time -- Emily's time in hours
noncomputable def emily_time_minutes : ℝ := emily_time * 60 -- Emily's time in minutes

theorem emily_cleaning_time :
  emily_time_minutes = 200 := by
  sorry

end emily_cleaning_time_l158_158464


namespace evaluate_f_at_7_l158_158435

theorem evaluate_f_at_7 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4) :
  f 7 = -3 :=
by
  sorry

end evaluate_f_at_7_l158_158435


namespace range_of_a_l158_158353

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, a ≤ x ∧ (x : ℝ) < 2 → x = -1 ∨ x = 0 ∨ x = 1) ↔ (-2 < a ∧ a ≤ -1) :=
by
  sorry

end range_of_a_l158_158353


namespace bottles_from_Shop_C_l158_158111

theorem bottles_from_Shop_C (TotalBottles ShopA ShopB ShopC : ℕ) 
  (h1 : TotalBottles = 550) 
  (h2 : ShopA = 150) 
  (h3 : ShopB = 180) 
  (h4 : TotalBottles = ShopA + ShopB + ShopC) : 
  ShopC = 220 := 
by
  sorry

end bottles_from_Shop_C_l158_158111


namespace verify_compound_interest_rate_l158_158219

noncomputable def compound_interest_rate
  (P A : ℝ) (t n : ℕ) : ℝ :=
  let r := (A / P) ^ (1 / (n * t)) - 1
  n * r

theorem verify_compound_interest_rate :
  let P := 5000
  let A := 6800
  let t := 4
  let n := 1
  compound_interest_rate P A t n = 8.02 / 100 :=
by
  sorry

end verify_compound_interest_rate_l158_158219


namespace real_part_of_z_l158_158917

theorem real_part_of_z (z : ℂ) (h : ∃ (r : ℝ), z^2 + z = r) : z.re = -1 / 2 :=
by
  sorry

end real_part_of_z_l158_158917


namespace ratio_of_squares_l158_158407

noncomputable def right_triangle : Type := sorry -- Placeholder for the right triangle type

variables (a b c : ℕ)

-- Given lengths of the triangle sides
def triangle_sides (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13 ∧ a^2 + b^2 = c^2

-- Define x and y based on the conditions in the problem
def side_length_square_x (x : ℝ) : Prop :=
  0 < x ∧ x < 5 ∧ x < 12

def side_length_square_y (y : ℝ) : Prop :=
  0 < y ∧ y < 13

-- The main theorem to prove
theorem ratio_of_squares (x y : ℝ) :
  ∀ a b c, triangle_sides a b c →
  side_length_square_x x →
  side_length_square_y y →
  x / y = 1 :=
sorry

end ratio_of_squares_l158_158407


namespace white_chocolate_bars_sold_l158_158256

theorem white_chocolate_bars_sold (W D : ℕ) (h1 : D = 15) (h2 : W / D = 4 / 3) : W = 20 :=
by
  -- This is where the proof would go.
  sorry

end white_chocolate_bars_sold_l158_158256


namespace max_square_area_in_rhombus_l158_158449

noncomputable def side_length_triangle := 10
noncomputable def height_triangle := Real.sqrt (side_length_triangle^2 - (side_length_triangle / 2)^2)
noncomputable def diag_long := 2 * height_triangle
noncomputable def diag_short := side_length_triangle
noncomputable def side_square := diag_short / Real.sqrt 2
noncomputable def area_square := side_square^2

theorem max_square_area_in_rhombus :
  area_square = 50 := by sorry

end max_square_area_in_rhombus_l158_158449


namespace closest_point_l158_158043

theorem closest_point 
  (x y z : ℝ) 
  (h_plane : 3 * x - 4 * y + 5 * z = 30)
  (A : ℝ × ℝ × ℝ := (1, 2, 3)) 
  (P : ℝ × ℝ × ℝ := (x, y, z)) :
  P = (11 / 5, 2 / 5, 5) := 
sorry

end closest_point_l158_158043


namespace work_completion_days_l158_158461

theorem work_completion_days (A B C : ℕ) 
  (hA : A = 4) (hB : B = 8) (hC : C = 8) : 
  2 = 1 / (1 / A + 1 / B + 1 / C) :=
by
  -- skip the proof for now
  sorry

end work_completion_days_l158_158461


namespace force_magnitudes_ratio_l158_158657

theorem force_magnitudes_ratio (a d : ℝ) (h1 : (a + 2 * d)^2 = a^2 + (a + d)^2) :
  ∃ k : ℝ, k > 0 ∧ (a + d) = a * (4 / 3) ∧ (a + 2 * d) = a * (5 / 3) :=
by
  sorry

end force_magnitudes_ratio_l158_158657


namespace negation_of_universal_proposition_l158_158530

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
by sorry

end negation_of_universal_proposition_l158_158530


namespace find_a2_l158_158077

open Classical

variable {a_n : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

theorem find_a2 (h1 : geometric_sequence a_n q)
                (h2 : a_n 7 = 1 / 4)
                (h3 : a_n 3 * a_n 5 = 4 * (a_n 4 - 1)) :
  a_n 2 = 8 :=
sorry

end find_a2_l158_158077


namespace garden_area_l158_158733

theorem garden_area (w l A : ℕ) (h1 : w = 12) (h2 : l = 3 * w) (h3 : A = l * w) : A = 432 := by
  sorry

end garden_area_l158_158733


namespace factorization1_factorization2_l158_158089

theorem factorization1 (x y : ℝ) : 4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3 * x + 3 * y)^2 :=
by
  sorry

theorem factorization2 (x : ℝ) (a : ℝ) : 2 * a * (x^2 + 1)^2 - 8 * a * x^2 = 2 * a * (x - 1)^2 * (x + 1)^2 :=
by
  sorry

end factorization1_factorization2_l158_158089


namespace least_possible_sections_l158_158017

theorem least_possible_sections (A C N : ℕ) (h1 : 7 * A = 11 * C) (h2 : N = A + C) : N = 18 :=
sorry

end least_possible_sections_l158_158017


namespace quadratic_roots_identity_l158_158175

noncomputable def sum_of_roots (a b : ℝ) : Prop := a + b = -10
noncomputable def product_of_roots (a b : ℝ) : Prop := a * b = 5

theorem quadratic_roots_identity (a b : ℝ)
  (h₁ : sum_of_roots a b)
  (h₂ : product_of_roots a b) :
  (a / b + b / a) = 18 :=
by sorry

end quadratic_roots_identity_l158_158175


namespace find_x_l158_158057

theorem find_x (x : ℝ) (h : 3550 - (x / 20.04) = 3500) : x = 1002 :=
by
  sorry

end find_x_l158_158057


namespace curve_symmetric_about_y_eq_x_l158_158276

def curve_eq (x y : ℝ) : Prop := x * y * (x + y) = 1

theorem curve_symmetric_about_y_eq_x :
  ∀ (x y : ℝ), curve_eq x y ↔ curve_eq y x :=
by sorry

end curve_symmetric_about_y_eq_x_l158_158276


namespace rational_sqrt_condition_l158_158537

variable (r q n : ℚ)

theorem rational_sqrt_condition
  (h : (1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q))) : 
  ∃ x : ℚ, x^2 = (n - 3) / (n + 1) :=
sorry

end rational_sqrt_condition_l158_158537


namespace range_of_m_increasing_function_l158_158817

theorem range_of_m_increasing_function :
  (2 : ℝ) ≤ m ∧ m ≤ 4 ↔ ∀ x : ℝ, (1 / 3 : ℝ) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2 ≤ 
                                 ((1 / 3 : ℝ) * (x + 1) ^ 3 - (4 * m - 1) * (x + 1) ^ 2 + (15 * m ^ 2 - 2 * m - 7) * (x + 1) + 2) :=
by
  sorry

end range_of_m_increasing_function_l158_158817


namespace larger_investment_value_l158_158336

-- Definitions of the conditions given in the problem
def investment_value_1 : ℝ := 500
def yearly_return_rate_1 : ℝ := 0.07
def yearly_return_rate_2 : ℝ := 0.27
def combined_return_rate : ℝ := 0.22

-- Stating the proof problem
theorem larger_investment_value :
  ∃ X : ℝ, X = 1500 ∧ 
    yearly_return_rate_1 * investment_value_1 + yearly_return_rate_2 * X = combined_return_rate * (investment_value_1 + X) :=
by {
  sorry -- Proof is omitted as per instructions
}

end larger_investment_value_l158_158336


namespace find_p_l158_158039

theorem find_p (p : ℝ) :
  (∀ x : ℝ, x^2 + p * x + p - 1 = 0) →
  ((exists x1 x2 : ℝ, x^2 + p * x + p - 1 = 0 ∧ x1^2 + x1^3 = - (x2^2 + x2^3) ) → (p = 1 ∨ p = 2)) :=
by
  intro h
  sorry

end find_p_l158_158039


namespace point_D_sum_is_ten_l158_158118

noncomputable def D_coordinates_sum_eq_ten : Prop :=
  ∃ (D : ℝ × ℝ), (5, 5) = ( (7 + D.1) / 2, (3 + D.2) / 2 ) ∧ (D.1 + D.2 = 10)

theorem point_D_sum_is_ten : D_coordinates_sum_eq_ten :=
  sorry

end point_D_sum_is_ten_l158_158118


namespace find_sum_l158_158617

noncomputable def principal_sum (P R : ℝ) := 
  let I := (P * R * 10) / 100
  let new_I := (P * (R + 5) * 10) / 100
  I + 600 = new_I

theorem find_sum (P R : ℝ) (h : principal_sum P R) : P = 1200 := 
  sorry

end find_sum_l158_158617


namespace container_volume_ratio_l158_158280

theorem container_volume_ratio (C D : ℝ) (hC: C > 0) (hD: D > 0)
  (h: (3/4) * C = (5/8) * D) : (C / D) = (5 / 6) :=
by
  sorry

end container_volume_ratio_l158_158280


namespace unique_not_in_range_l158_158515

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 23 = 23) (h₆ : f a b c d 101 = 101) (h₇ : ∀ x ≠ -d / c, f a b c d (f a b c d x) = x) :
  (a / c) = 62 := 
 sorry

end unique_not_in_range_l158_158515


namespace weight_of_second_new_player_l158_158346

theorem weight_of_second_new_player
  (number_of_original_players : ℕ)
  (average_weight_of_original_players : ℝ)
  (weight_of_first_new_player : ℝ)
  (new_average_weight : ℝ)
  (total_number_of_players : ℕ)
  (total_weight_of_9_players : ℝ)
  (combined_weight_of_original_and_first_new : ℝ)
  (weight_of_second_new_player : ℝ)
  (h1 : number_of_original_players = 7)
  (h2 : average_weight_of_original_players = 103)
  (h3 : weight_of_first_new_player = 110)
  (h4 : new_average_weight = 99)
  (h5 : total_number_of_players = 9)
  (h6 : total_weight_of_9_players = total_number_of_players * new_average_weight)
  (h7 : combined_weight_of_original_and_first_new = number_of_original_players * average_weight_of_original_players + weight_of_first_new_player)
  (h8 : total_weight_of_9_players - combined_weight_of_original_and_first_new = weight_of_second_new_player) :
  weight_of_second_new_player = 60 :=
by
  sorry

end weight_of_second_new_player_l158_158346


namespace train_meeting_distance_l158_158649

theorem train_meeting_distance :
  let distance := 150
  let time_x := 4
  let time_y := 3.5
  let speed_x := distance / time_x
  let speed_y := distance / time_y
  let relative_speed := speed_x + speed_y
  let time_to_meet := distance / relative_speed
  let distance_x_at_meeting := time_to_meet * speed_x
  distance_x_at_meeting = 70 := by
sorry

end train_meeting_distance_l158_158649


namespace lindsey_final_money_l158_158835

-- Define the savings in each month
def save_sep := 50
def save_oct := 37
def save_nov := 11

-- Total savings over the three months
def total_savings := save_sep + save_oct + save_nov

-- Condition for Mom's contribution
def mom_contribution := if total_savings > 75 then 25 else 0

-- Total savings including mom's contribution
def total_with_mom := total_savings + mom_contribution

-- Amount spent on the video game
def spent := 87

-- Final amount left
def final_amount := total_with_mom - spent

-- Proof statement
theorem lindsey_final_money : final_amount = 36 := by
  sorry

end lindsey_final_money_l158_158835


namespace set_d_pythagorean_triple_l158_158716

theorem set_d_pythagorean_triple : (9^2 + 40^2 = 41^2) :=
by sorry

end set_d_pythagorean_triple_l158_158716


namespace number_half_reduction_l158_158014

/-- Define the conditions -/
def percentage_more (percent : Float) (amount : Float) : Float := amount + (percent / 100) * amount

theorem number_half_reduction (x : Float) : percentage_more 30 75 = 97.5 → (x / 2) = 97.5 → x = 195 := by
  intros h1 h2
  sorry

end number_half_reduction_l158_158014


namespace total_price_of_property_l158_158132

theorem total_price_of_property (price_per_sq_ft: ℝ) (house_size barn_size: ℝ) (house_price barn_price total_price: ℝ) :
  price_per_sq_ft = 98 ∧ house_size = 2400 ∧ barn_size = 1000 → 
  house_price = price_per_sq_ft * house_size ∧
  barn_price = price_per_sq_ft * barn_size ∧
  total_price = house_price + barn_price →
  total_price = 333200 :=
by
  sorry

end total_price_of_property_l158_158132


namespace julie_age_end_of_period_is_15_l158_158404

-- Define necessary constants and variables
def hours_per_day : ℝ := 3
def pay_rate_per_hour_per_year : ℝ := 0.75
def total_days_worked : ℝ := 60
def total_earnings : ℝ := 810

-- Define Julie's age at the end of the four-month period
def julies_age_end_of_period (age: ℝ) : Prop :=
  hours_per_day * pay_rate_per_hour_per_year * age * total_days_worked = total_earnings

-- The final Lean 4 statement that needs proof
theorem julie_age_end_of_period_is_15 : ∃ age : ℝ, julies_age_end_of_period age ∧ age = 15 :=
by {
  sorry
}

end julie_age_end_of_period_is_15_l158_158404


namespace initial_cherry_sweets_30_l158_158076

/-!
# Problem Statement
A packet of candy sweets has some cherry-flavored sweets (C), 40 strawberry-flavored sweets, 
and 50 pineapple-flavored sweets. Aaron eats half of each type of sweet and then gives away 
5 cherry-flavored sweets to his friend. There are still 55 sweets in the packet of candy.
Prove that the initial number of cherry-flavored sweets was 30.
-/

noncomputable def initial_cherry_sweets (C : ℕ) : Prop :=
  let remaining_cherry_sweets := C / 2 - 5
  let remaining_strawberry_sweets := 40 / 2
  let remaining_pineapple_sweets := 50 / 2
  remaining_cherry_sweets + remaining_strawberry_sweets + remaining_pineapple_sweets = 55

theorem initial_cherry_sweets_30 : initial_cherry_sweets 30 :=
  sorry

end initial_cherry_sweets_30_l158_158076


namespace f_f_3_eq_651_over_260_l158_158347

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (2 + x⁻¹))

/-- Prove that f(f(3)) = 651/260 -/
theorem f_f_3_eq_651_over_260 : f (f (3)) = 651 / 260 := 
sorry

end f_f_3_eq_651_over_260_l158_158347


namespace units_digit_calculation_l158_158682

-- Define a function to compute the units digit of a number in base 10
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_calculation :
  units_digit (8 * 18 * 1988 - 8^3) = 0 := by
  sorry

end units_digit_calculation_l158_158682


namespace walking_time_proof_l158_158769

-- Define the conditions from the problem
def bus_ride : ℕ := 75
def train_ride : ℕ := 360
def total_trip_time : ℕ := 480

-- Define the walking time as variable
variable (W : ℕ)

-- State the theorem as a Lean statement
theorem walking_time_proof :
  bus_ride + W + 2 * W + train_ride = total_trip_time → W = 15 :=
by
  intros h
  sorry

end walking_time_proof_l158_158769


namespace gcd_372_684_l158_158775

theorem gcd_372_684 : Nat.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l158_158775


namespace paul_collected_total_cans_l158_158597

theorem paul_collected_total_cans :
  let saturday_bags := 10
  let sunday_bags := 5
  let saturday_cans_per_bag := 12
  let sunday_cans_per_bag := 15
  let saturday_total_cans := saturday_bags * saturday_cans_per_bag
  let sunday_total_cans := sunday_bags * sunday_cans_per_bag
  let total_cans := saturday_total_cans + sunday_total_cans
  total_cans = 195 := 
by
  sorry

end paul_collected_total_cans_l158_158597


namespace total_seats_at_round_table_l158_158712

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l158_158712


namespace x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l158_158320

theorem x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one {x : ℝ} :
  (x > 1 → |x| > 1) ∧ (¬(|x| > 1 → x > 1)) :=
by
  sorry

end x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l158_158320


namespace top_layer_lamps_l158_158771

theorem top_layer_lamps (a : ℕ) :
  (a + 2 * a + 4 * a + 8 * a + 16 * a + 32 * a + 64 * a = 381) → a = 3 := 
by
  intro h
  sorry

end top_layer_lamps_l158_158771


namespace quadratic_z_and_u_l158_158997

variables (a b c α β γ : ℝ)
variable (d : ℝ)
variable (δ : ℝ)
variables (x₁ x₂ y₁ y₂ z₁ z₂ u₁ u₂ : ℝ)

-- Given conditions
variable (h_nonzero : a * α ≠ 0)
variable (h_discriminant1 : b^2 - 4 * a * c ≥ 0)
variable (h_discriminant2 : β^2 - 4 * α * γ ≥ 0)
variable (hx_roots_order : x₁ ≤ x₂)
variable (hy_roots_order : y₁ ≤ y₂)
variable (h_eq_discriminant1 : b^2 - 4 * a * c = d^2)
variable (h_eq_discriminant2 : β^2 - 4 * α * γ = δ^2)

-- Translate into mathematical constraints for the roots
variable (hx1 : x₁ = (-b - d) / (2 * a))
variable (hx2 : x₂ = (-b + d) / (2 * a))
variable (hy1 : y₁ = (-β - δ) / (2 * α))
variable (hy2 : y₂ = (-β + δ) / (2 * α))

-- Variables for polynomial equations roots
axiom h_z1 : z₁ = x₁ + y₁
axiom h_z2 : z₂ = x₂ + y₂
axiom h_u1 : u₁ = x₁ + y₂
axiom h_u2 : u₂ = x₂ + y₁

theorem quadratic_z_and_u :
  (2 * a * α) * z₂ * z₂ + 2 * (a * β + α * b) * z₁ + (2 * a * γ + 2 * α * c + b * β - d * δ) = 0 ∧
  (2 * a * α) * u₂ * u₂ + 2 * (a * β + α * b) * u₁ + (2 * a * γ + 2 * α * c + b * β + d * δ) = 0 := sorry

end quadratic_z_and_u_l158_158997


namespace trapezoid_CD_length_l158_158217

/-- In trapezoid ABCD with AD parallel to BC and diagonals intersecting:
  - BD = 2
  - ∠DBC = 36°
  - ∠BDA = 72°
  - The ratio BC : AD = 5 : 3

We are to show that the length of CD is 4/3. --/
theorem trapezoid_CD_length
  {A B C D : Type}
  (BD : ℝ) (DBC : ℝ) (BDA : ℝ) (BC_over_AD : ℝ)
  (AD_parallel_BC : Prop) (diagonals_intersect : Prop)
  (hBD : BD = 2) 
  (hDBC : DBC = 36) 
  (hBDA : BDA = 72)
  (hBC_over_AD : BC_over_AD = 5 / 3) 
  :  CD = 4 / 3 :=
by
  sorry

end trapezoid_CD_length_l158_158217


namespace find_x_plus_y_l158_158496

theorem find_x_plus_y (x y : ℝ) (h1 : |x| + x + y = 16) (h2 : x + |y| - y = 18) : x + y = 6 := 
sorry

end find_x_plus_y_l158_158496


namespace identity_true_for_any_abc_l158_158884

theorem identity_true_for_any_abc : 
  ∀ (a b c : ℝ), (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c :=
by
  sorry

end identity_true_for_any_abc_l158_158884


namespace payment_per_minor_character_l158_158274

noncomputable def M : ℝ := 285000 / 19 

theorem payment_per_minor_character
    (num_main_characters : ℕ := 5)
    (num_minor_characters : ℕ := 4)
    (total_payment : ℝ := 285000)
    (payment_ratio : ℝ := 3)
    (eq1 : 5 * 3 * M + 4 * M = total_payment) :
    M = 15000 :=
by
  sorry

end payment_per_minor_character_l158_158274


namespace find_expression_l158_158908

theorem find_expression (a b : ℝ) (h₁ : a - b = 5) (h₂ : a * b = 2) :
  a^2 - a * b + b^2 = 27 := 
by
  sorry

end find_expression_l158_158908


namespace store_A_total_cost_store_B_total_cost_cost_effective_store_l158_158332

open Real

def total_cost_store_A (x : ℝ) : ℝ :=
  110 * x + 210 * (100 - x)

def total_cost_store_B (x : ℝ) : ℝ :=
  120 * x + 202 * (100 - x)

theorem store_A_total_cost (x : ℝ) :
  total_cost_store_A x = -100 * x + 21000 :=
by
  sorry

theorem store_B_total_cost (x : ℝ) :
  total_cost_store_B x = -82 * x + 20200 :=
by
  sorry

theorem cost_effective_store (x : ℝ) (h : x = 60) :
  total_cost_store_A x < total_cost_store_B x :=
by
  rw [h]
  sorry

end store_A_total_cost_store_B_total_cost_cost_effective_store_l158_158332


namespace function_satisfies_equation_l158_158462

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x - 1)

theorem function_satisfies_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  f ((x - 1) / x) + f (1 / (1 - x)) = 2 - 2 * x := by
  sorry

end function_satisfies_equation_l158_158462


namespace evaluate_f_at_5_l158_158115

def f (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 38*x^2 - 35*x - 40

theorem evaluate_f_at_5 : f 5 = 110 :=
by
  sorry

end evaluate_f_at_5_l158_158115


namespace average_speed_l158_158504

theorem average_speed (v1 v2 : ℝ) (h1 : v1 = 110) (h2 : v2 = 88) : 
  (2 * v1 * v2) / (v1 + v2) = 97.78 := 
by sorry

end average_speed_l158_158504


namespace james_goals_product_l158_158400

theorem james_goals_product :
  ∃ (g7 g8 : ℕ), g7 < 7 ∧ g8 < 7 ∧ 
  (22 + g7) % 7 = 0 ∧ (22 + g7 + g8) % 8 = 0 ∧ 
  g7 * g8 = 24 :=
by
  sorry

end james_goals_product_l158_158400


namespace frisbee_price_l158_158505

theorem frisbee_price 
  (total_frisbees : ℕ)
  (frisbees_at_3 : ℕ)
  (price_x_frisbees : ℕ)
  (total_revenue : ℕ) 
  (min_frisbees_at_x : ℕ)
  (price_at_3 : ℕ) 
  (n_min_at_x : ℕ)
  (h1 : total_frisbees = 60)
  (h2 : price_at_3 = 3)
  (h3 : total_revenue = 200)
  (h4 : n_min_at_x = 20)
  (h5 : min_frisbees_at_x >= n_min_at_x)
  : price_x_frisbees = 4 :=
by
  sorry

end frisbee_price_l158_158505


namespace range_of_a_l158_158271

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x - 3)

def monotonic_increasing (a : ℝ) : Prop :=
  ∀ x > 1, 2 * x - a > 0

def positive_argument (a : ℝ) : Prop :=
  ∀ x > 1, x^2 - a * x - 3 > 0

theorem range_of_a :
  {a : ℝ | monotonic_increasing a ∧ positive_argument a} = {a : ℝ | a ≤ -2} :=
sorry

end range_of_a_l158_158271


namespace last_four_digits_of_5_pow_2011_l158_158651

theorem last_four_digits_of_5_pow_2011 :
  (5^2011) % 10000 = 8125 := 
by
  -- Using modular arithmetic and periodicity properties of powers of 5.
  sorry

end last_four_digits_of_5_pow_2011_l158_158651


namespace four_consecutive_integers_plus_one_is_square_l158_158452

theorem four_consecutive_integers_plus_one_is_square (n : ℤ) : 
  (n - 1) * n * (n + 1) * (n + 2) + 1 = (n ^ 2 + n - 1) ^ 2 := 
by 
  sorry

end four_consecutive_integers_plus_one_is_square_l158_158452


namespace count_integer_values_l158_158060

theorem count_integer_values (x : ℤ) (h1 : 4 < Real.sqrt (3 * x + 1)) (h2 : Real.sqrt (3 * x + 1) < 5) : 
  (5 < x ∧ x < 8 ∧ ∃ (N : ℕ), N = 2) :=
by sorry

end count_integer_values_l158_158060


namespace solve_system_of_equations_l158_158036

theorem solve_system_of_equations :
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℤ), 
    x1 + x2 + x3 = 6 ∧
    x2 + x3 + x4 = 9 ∧
    x3 + x4 + x5 = 3 ∧
    x4 + x5 + x6 = -3 ∧
    x5 + x6 + x7 = -9 ∧
    x6 + x7 + x8 = -6 ∧
    x7 + x8 + x1 = -2 ∧
    x8 + x1 + x2 = 2 ∧
    (x1, x2, x3, x4, x5, x6, x7, x8) = (1, 2, 3, 4, -4, -3, -2, -1) :=
by
  -- solution will be here
  sorry

end solve_system_of_equations_l158_158036


namespace plant_supplier_money_left_correct_l158_158269

noncomputable def plant_supplier_total_earnings : ℕ :=
  35 * 52 + 30 * 32 + 20 * 77 + 25 * 22 + 40 * 15

noncomputable def plant_supplier_total_expenses : ℕ :=
  3 * 65 + 2 * 45 + 280 + 150 + 100 + 125 + 225 + 550

noncomputable def plant_supplier_money_left : ℕ :=
  plant_supplier_total_earnings - plant_supplier_total_expenses

theorem plant_supplier_money_left_correct :
  plant_supplier_money_left = 3755 :=
by
  sorry

end plant_supplier_money_left_correct_l158_158269


namespace sum_of_reciprocals_l158_158912

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : 
  1/x + 1/y = 3/8 := by
  sorry

end sum_of_reciprocals_l158_158912


namespace f_of_pi_over_6_l158_158292

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem f_of_pi_over_6 (ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : -Real.pi / 2 ≤ ϕ) (h₂ : ϕ < Real.pi / 2) 
  (transformed : ∀ x, f ω ϕ (x/2 - Real.pi/6) = Real.sin x) :
  f ω ϕ (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end f_of_pi_over_6_l158_158292


namespace find_equation_of_line_l158_158879

-- Define the conditions
def line_passes_through_A (m b : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (1, 1) ∧ A.2 = -A.1 + b

def intercepts_equal (m b : ℝ) : Prop :=
  b = m

-- The goal to prove the equations of the line
theorem find_equation_of_line :
  ∃ (m b : ℝ), line_passes_through_A m b (1, 1) ∧ intercepts_equal m b ↔ 
  (∃ m b : ℝ, (m = -1 ∧ b = 2) ∨ (m = 1 ∧ b = 0)) :=
sorry

end find_equation_of_line_l158_158879


namespace angle_A_area_triangle_l158_158044

-- The first problem: Proving angle A
theorem angle_A (a b c : ℝ) (A C : ℝ) 
  (h1 : (2 * b - c) * Real.cos A = a * Real.cos C) : 
  A = Real.pi / 3 :=
by sorry

-- The second problem: Finding the area of triangle ABC
theorem area_triangle (a b c : ℝ) (A : ℝ)
  (h1 : a = 3)
  (h2 : b = 2 * c)
  (h3 : A = Real.pi / 3) :
  0.5 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2 :=
by sorry

end angle_A_area_triangle_l158_158044


namespace shaniqua_income_per_haircut_l158_158378

theorem shaniqua_income_per_haircut (H : ℝ) :
  (8 * H + 5 * 25 = 221) → (H = 12) :=
by
  intro h
  sorry

end shaniqua_income_per_haircut_l158_158378


namespace sum_of_coefficients_l158_158497

theorem sum_of_coefficients (a : Fin 7 → ℕ) (x : ℕ) : 
  (1 - x) ^ 6 = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 → 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 := 
by
  intro h
  by_cases hx : x = 1
  · rw [hx] at h
    sorry
  · sorry

end sum_of_coefficients_l158_158497


namespace mr_green_garden_yield_l158_158924

noncomputable def garden_yield (steps_length steps_width step_length yield_per_sqft : ℝ) : ℝ :=
  let length_ft := steps_length * step_length
  let width_ft := steps_width * step_length
  let area := length_ft * width_ft
  area * yield_per_sqft

theorem mr_green_garden_yield :
  garden_yield 18 25 2.5 0.5 = 1406.25 :=
by
  sorry

end mr_green_garden_yield_l158_158924


namespace mode_is_necessary_characteristic_of_dataset_l158_158512

-- Define a dataset as a finite set of elements from any type.
variable {α : Type*} [Fintype α]

-- Define a mode for a dataset as an element that occurs most frequently.
def mode (dataset : Multiset α) : α :=
sorry  -- Mode definition and computation are omitted for this high-level example.

-- Define the theorem that mode is a necessary characteristic of a dataset.
theorem mode_is_necessary_characteristic_of_dataset (dataset : Multiset α) : 
  exists mode_elm : α, mode_elm = mode dataset :=
sorry

end mode_is_necessary_characteristic_of_dataset_l158_158512


namespace find_c_work_rate_l158_158679

variables (A B C : ℚ)   -- Using rational numbers for the work rates

theorem find_c_work_rate (h1 : A + B = 1/3) (h2 : B + C = 1/4) (h3 : C + A = 1/6) : 
  C = 1/24 := 
sorry 

end find_c_work_rate_l158_158679


namespace total_onions_grown_l158_158748

-- Given conditions
def onions_grown_by_Nancy : ℕ := 2
def onions_grown_by_Dan : ℕ := 9
def onions_grown_by_Mike : ℕ := 4
def days_worked : ℕ := 6

-- Statement we need to prove
theorem total_onions_grown : onions_grown_by_Nancy + onions_grown_by_Dan + onions_grown_by_Mike = 15 :=
by sorry

end total_onions_grown_l158_158748


namespace probability_of_ge_four_is_one_eighth_l158_158394

noncomputable def probability_ge_four : ℝ :=
sorry

theorem probability_of_ge_four_is_one_eighth :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 2) ∧ (0 ≤ y ∧ y ≤ 2) →
  (probability_ge_four = 1 / 8) :=
sorry

end probability_of_ge_four_is_one_eighth_l158_158394


namespace probability_equal_white_black_probability_white_ge_black_l158_158181

/-- Part (a) -/
theorem probability_equal_white_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (2 * m) / (n + m)) := 
  sorry

/-- Part (b) -/
theorem probability_white_ge_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (n - m + 1) / (n + 1)) := 
  sorry

end probability_equal_white_black_probability_white_ge_black_l158_158181


namespace solve_for_h_l158_158734

-- Define the given polynomials
def p1 (x : ℝ) : ℝ := 2*x^5 + 4*x^3 - 3*x^2 + x + 7
def p2 (x : ℝ) : ℝ := -x^3 + 2*x^2 - 5*x + 4

-- Define h(x) as the unknown polynomial to solve for
def h (x : ℝ) : ℝ := -2*x^5 - x^3 + 5*x^2 - 6*x - 3

-- The theorem to prove
theorem solve_for_h : 
  (∀ (x : ℝ), p1 x + h x = p2 x) → (∀ (x : ℝ), h x = -2*x^5 - x^3 + 5*x^2 - 6*x - 3) :=
by
  intro h_cond
  sorry

end solve_for_h_l158_158734


namespace correct_operation_l158_158851

theorem correct_operation (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 :=
sorry

end correct_operation_l158_158851


namespace problem1_problem2_l158_158423

theorem problem1 (x : ℝ) : 2 * (x - 1) ^ 2 = 18 ↔ x = 4 ∨ x = -2 := by
  sorry

theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 := by
  sorry

end problem1_problem2_l158_158423


namespace profit_at_15_is_correct_l158_158904

noncomputable def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

theorem profit_at_15_is_correct :
  profit 15 = 1250 := by
  sorry

end profit_at_15_is_correct_l158_158904


namespace possible_combinations_l158_158272

noncomputable def dark_chocolate_price : ℝ := 5
noncomputable def milk_chocolate_price : ℝ := 4.50
noncomputable def white_chocolate_price : ℝ := 6
noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def leonardo_money : ℝ := 4 + 0.59

noncomputable def total_money := leonardo_money

noncomputable def dark_chocolate_with_tax := dark_chocolate_price * (1 + sales_tax_rate)
noncomputable def milk_chocolate_with_tax := milk_chocolate_price * (1 + sales_tax_rate)
noncomputable def white_chocolate_with_tax := white_chocolate_price * (1 + sales_tax_rate)

theorem possible_combinations :
  total_money = 4.59 ∧ (total_money >= 0 ∧ total_money < dark_chocolate_with_tax ∧ total_money < white_chocolate_with_tax ∧
  total_money ≥ milk_chocolate_with_tax ∧ milk_chocolate_with_tax = 4.82) :=
by
  sorry

end possible_combinations_l158_158272


namespace find_a_b_c_sum_l158_158333

-- Define the necessary conditions and constants
def radius : ℝ := 10  -- tower radius in feet
def rope_length : ℝ := 30  -- length of the rope in feet
def unicorn_height : ℝ := 6  -- height of the unicorn from ground in feet
def rope_end_distance : ℝ := 6  -- distance from the unicorn to the nearest point on the tower

def a : ℕ := 30
def b : ℕ := 900
def c : ℕ := 10  -- assuming c is not necessarily prime for the purpose of this exercise

-- The theorem we want to prove
theorem find_a_b_c_sum : a + b + c = 940 :=
by
  sorry

end find_a_b_c_sum_l158_158333


namespace difference_of_squares_l158_158913

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
sorry

end difference_of_squares_l158_158913


namespace prove_f_f_x_eq_4_prove_f_f_x_eq_5_l158_158222

variable (f : ℝ → ℝ)

-- Conditions
axiom f_of_4 : f (-2) = 4 ∧ f 2 = 4 ∧ f 6 = 4
axiom f_of_5 : f (-4) = 5 ∧ f 4 = 5

-- Intermediate Values
axiom f_inv_of_4 : f 0 = -2 ∧ f (-1) = 2 ∧ f 3 = 6
axiom f_inv_of_5 : f 2 = 4

theorem prove_f_f_x_eq_4 :
  {x : ℝ | f (f x) = 4} = {0, -1, 3} :=
by
  sorry

theorem prove_f_f_x_eq_5 :
  {x : ℝ | f (f x) = 5} = {2} :=
by
  sorry

end prove_f_f_x_eq_4_prove_f_f_x_eq_5_l158_158222


namespace train_length_is_225_m_l158_158560

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_s : ℝ := 9

noncomputable def speed_ms : ℝ := speed_kmph / 3.6
noncomputable def distance_m (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length_is_225_m :
  distance_m speed_ms time_s = 225 := by
  sorry

end train_length_is_225_m_l158_158560


namespace range_of_b_l158_158201

-- Define the conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16
def line_eq (x y b : ℝ) : Prop := y = x + b
def distance_point_line_eq (x y b d : ℝ) : Prop := 
  d = abs (b) / (Real.sqrt 2)
def at_least_three_points_on_circle_at_distance_one (b : ℝ) : Prop := 
  ∃ p1 p2 p3 : ℝ × ℝ, circle_eq p1.1 p1.2 ∧ circle_eq p2.1 p2.2 ∧ circle_eq p3.1 p3.2 ∧ 
  distance_point_line_eq p1.1 p1.2 b 1 ∧ distance_point_line_eq p2.1 p2.2 b 1 ∧ distance_point_line_eq p3.1 p3.2 b 1

-- The theorem statement to prove
theorem range_of_b (b : ℝ) (h : at_least_three_points_on_circle_at_distance_one b) : 
  -3 * Real.sqrt 2 ≤ b ∧ b ≤ 3 * Real.sqrt 2 := 
sorry

end range_of_b_l158_158201


namespace probability_of_closer_to_D_in_triangle_DEF_l158_158381

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem probability_of_closer_to_D_in_triangle_DEF :
  let D := (0, 0)
  let E := (0, 6)
  let F := (8, 0)
  let M := ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  let N := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area_DEF := triangle_area D E F
  let area_DMN := triangle_area D M N
  area_DMN / area_DEF = 1 / 4 := by
    sorry

end probability_of_closer_to_D_in_triangle_DEF_l158_158381


namespace marias_profit_l158_158199

theorem marias_profit 
  (initial_loaves : ℕ)
  (morning_price : ℝ)
  (afternoon_discount : ℝ)
  (late_afternoon_price : ℝ)
  (cost_per_loaf : ℝ)
  (loaves_sold_morning : ℕ)
  (loaves_sold_afternoon : ℕ)
  (loaves_remaining : ℕ)
  (revenue_morning : ℝ)
  (revenue_afternoon : ℝ)
  (revenue_late_afternoon : ℝ)
  (total_revenue : ℝ)
  (total_cost : ℝ)
  (profit : ℝ) :
  initial_loaves = 60 →
  morning_price = 3.0 →
  afternoon_discount = 0.75 →
  late_afternoon_price = 1.50 →
  cost_per_loaf = 1.0 →
  loaves_sold_morning = initial_loaves / 3 →
  loaves_sold_afternoon = (initial_loaves - loaves_sold_morning) / 2 →
  loaves_remaining = initial_loaves - loaves_sold_morning - loaves_sold_afternoon →
  revenue_morning = loaves_sold_morning * morning_price →
  revenue_afternoon = loaves_sold_afternoon * (afternoon_discount * morning_price) →
  revenue_late_afternoon = loaves_remaining * late_afternoon_price →
  total_revenue = revenue_morning + revenue_afternoon + revenue_late_afternoon →
  total_cost = initial_loaves * cost_per_loaf →
  profit = total_revenue - total_cost →
  profit = 75 := sorry

end marias_profit_l158_158199


namespace multiple_of_a_age_l158_158894

theorem multiple_of_a_age (A B M : ℝ) (h1 : A = B + 5) (h2 : A + B = 13) (h3 : M * (A + 7) = 4 * (B + 7)) : M = 2.75 :=
sorry

end multiple_of_a_age_l158_158894


namespace conclusion1_conclusion2_l158_158048

theorem conclusion1 (x y a b : ℝ) (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b :=
sorry

theorem conclusion2 (x a : ℝ) (h1 : (x-1)*(x^2 + a*x + 1) - x^2 = x^3 - (a-1)*x^2 - (1-a)*x - 1) : a = 1 :=
sorry

end conclusion1_conclusion2_l158_158048


namespace pairs_of_integers_l158_158006

-- The main theorem to prove:
theorem pairs_of_integers (x y : ℤ) :
  y ^ 2 = x ^ 3 + 16 ↔ (x = 0 ∧ (y = 4 ∨ y = -4)) :=
by sorry

end pairs_of_integers_l158_158006


namespace sum_of_coordinates_l158_158789

-- Definitions based on conditions
variable (f k : ℝ → ℝ)
variable (h₁ : f 4 = 8)
variable (h₂ : ∀ x, k x = (f x) ^ 3)

-- Statement of the theorem
theorem sum_of_coordinates : 4 + k 4 = 516 := by
  -- Proof would go here
  sorry

end sum_of_coordinates_l158_158789


namespace bulbs_on_perfect_squares_l158_158499

def is_on (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k * k

theorem bulbs_on_perfect_squares (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  (∀ i : ℕ, 1 ≤ i → i ≤ 100 → ∃ j : ℕ, i = j * j ↔ is_on i) := sorry

end bulbs_on_perfect_squares_l158_158499


namespace rowing_speed_still_water_l158_158708

theorem rowing_speed_still_water (v r : ℕ) (h1 : r = 18) (h2 : 1 / (v - r) = 3 * (1 / (v + r))) : v = 36 :=
by sorry

end rowing_speed_still_water_l158_158708


namespace order_of_numbers_l158_158939

def base16_to_dec (s : String) : ℕ := sorry
def base6_to_dec (s : String) : ℕ := sorry
def base4_to_dec (s : String) : ℕ := sorry
def base2_to_dec (s : String) : ℕ := sorry

theorem order_of_numbers:
  let a := base16_to_dec "3E"
  let b := base6_to_dec "210"
  let c := base4_to_dec "1000"
  let d := base2_to_dec "111011"
  a = 62 ∧ b = 78 ∧ c = 64 ∧ d = 59 →
  b > c ∧ c > a ∧ a > d :=
by
  intros
  sorry

end order_of_numbers_l158_158939


namespace train_length_l158_158611

theorem train_length (bridge_length time_seconds speed_kmh : ℝ) (S : speed_kmh = 64) (T : time_seconds = 45) (B : bridge_length = 300) : 
  ∃ (train_length : ℝ), train_length = 500 := 
by
  -- Add your proof here 
  sorry

end train_length_l158_158611


namespace range_of_m_l158_158854

noncomputable def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
noncomputable def q (x m : ℝ) : Prop := (x^2 - (2*m + 1)*x + m^2 + m) ≤ 0

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, p x → q x m) ∧ ¬ (∀ x : ℝ, p x → q x m) ↔ m ∈ Set.Ico (-2 : ℝ) (-1) ∪ Set.Ioc 0 (1 : ℝ) :=
by
  sorry

end range_of_m_l158_158854


namespace initial_number_of_cards_l158_158239

theorem initial_number_of_cards (x : ℕ) (h : x + 76 = 79) : x = 3 :=
by
  sorry

end initial_number_of_cards_l158_158239


namespace problem_solution_l158_158522

theorem problem_solution (m n : ℤ) (h : m + 1 = (n - 2) / 3) : 3 * m - n = -5 :=
by
  sorry

end problem_solution_l158_158522


namespace all_rationals_on_number_line_l158_158467

theorem all_rationals_on_number_line :
  ∀ q : ℚ, ∃ p : ℝ, p = ↑q :=
by
  sorry

end all_rationals_on_number_line_l158_158467


namespace min_trials_to_ensure_pass_l158_158642

theorem min_trials_to_ensure_pass (p : ℝ) (n : ℕ) (h₁ : p = 3 / 4) (h₂ : n ≥ 1): 
  (1 - (1 - p) ^ n) > 0.99 → n ≥ 4 :=
by sorry

end min_trials_to_ensure_pass_l158_158642


namespace A_inter_B_eq_A_l158_158075

def A := {x : ℝ | 0 < x ∧ x ≤ 2}
def B := {x : ℝ | x ≤ 3}

theorem A_inter_B_eq_A : A ∩ B = A := 
by 
  sorry 

end A_inter_B_eq_A_l158_158075


namespace fuel_needed_to_empty_l158_158007

theorem fuel_needed_to_empty (x : ℝ) 
  (h1 : (3/4) * x - (1/3) * x = 15) :
  (1/3) * x = 12 :=
by 
-- Proving the result
sorry

end fuel_needed_to_empty_l158_158007


namespace prove_by_contradiction_l158_158460

-- Statement: To prove "a > b" by contradiction, assuming the negation "a ≤ b".
theorem prove_by_contradiction (a b : ℝ) (h : a ≤ b) : false := sorry

end prove_by_contradiction_l158_158460


namespace no_positive_integer_n_has_perfect_square_form_l158_158744

theorem no_positive_integer_n_has_perfect_square_form (n : ℕ) (h : 0 < n) : 
  ¬ ∃ k : ℕ, n^4 + 2 * n^3 + 2 * n^2 + 2 * n + 1 = k^2 := 
sorry

end no_positive_integer_n_has_perfect_square_form_l158_158744


namespace linear_function_quadrants_passing_through_l158_158154

theorem linear_function_quadrants_passing_through :
  ∀ (x : ℝ) (y : ℝ), (y = 2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end linear_function_quadrants_passing_through_l158_158154


namespace product_of_sequence_is_256_l158_158703

-- Definitions for conditions
def seq : List ℚ := [1 / 4, 16 / 1, 1 / 64, 256 / 1, 1 / 1024, 4096 / 1, 1 / 16384, 65536 / 1]

-- The main theorem
theorem product_of_sequence_is_256 : (seq.prod = 256) :=
by
  sorry

end product_of_sequence_is_256_l158_158703


namespace total_initial_candles_l158_158126

-- Define the conditions
def used_candles : ℕ := 32
def leftover_candles : ℕ := 12

-- State the theorem
theorem total_initial_candles : used_candles + leftover_candles = 44 := by
  sorry

end total_initial_candles_l158_158126


namespace sin_600_eq_neg_sqrt_3_div_2_l158_158263

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l158_158263


namespace same_gender_probability_l158_158570

-- Define the total number of teachers in School A and their gender distribution.
def schoolA_teachers : Nat := 3
def schoolA_males : Nat := 2
def schoolA_females : Nat := 1

-- Define the total number of teachers in School B and their gender distribution.
def schoolB_teachers : Nat := 3
def schoolB_males : Nat := 1
def schoolB_females : Nat := 2

-- Calculate the probability of selecting two teachers of the same gender.
theorem same_gender_probability :
  (schoolA_males * schoolB_males + schoolA_females * schoolB_females) / (schoolA_teachers * schoolB_teachers) = 4 / 9 :=
by
  sorry

end same_gender_probability_l158_158570


namespace contrapositive_of_quadratic_l158_158456

theorem contrapositive_of_quadratic (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by
  sorry

end contrapositive_of_quadratic_l158_158456


namespace cost_price_of_ball_l158_158944

variable (C : ℝ)

theorem cost_price_of_ball (h : 15 * C - 720 = 5 * C) : C = 72 :=
by
  sorry

end cost_price_of_ball_l158_158944


namespace shoe_length_increase_l158_158218

theorem shoe_length_increase
  (L : ℝ)
  (x : ℝ)
  (h1 : L + 9*x = L * 1.2)
  (h2 : L + 7*x = 10.4) :
  x = 0.2 :=
by
  sorry

end shoe_length_increase_l158_158218


namespace number_of_customers_l158_158510

theorem number_of_customers (total_sandwiches : ℕ) (office_orders : ℕ) (customers_half : ℕ) (num_offices : ℕ) (num_sandwiches_per_office : ℕ) 
  (sandwiches_per_customer : ℕ) (group_sandwiches : ℕ) (total_customers : ℕ) :
  total_sandwiches = 54 →
  num_offices = 3 →
  num_sandwiches_per_office = 10 →
  group_sandwiches = total_sandwiches - num_offices * num_sandwiches_per_office →
  customers_half * sandwiches_per_customer = group_sandwiches →
  sandwiches_per_customer = 4 →
  customers_half = total_customers / 2 →
  total_customers = 12 :=
by
  intros
  sorry

end number_of_customers_l158_158510


namespace systematic_sampling_l158_158760

-- Definitions for the class of 50 students numbered from 1 to 50, sampling interval, and starting number.
def students : Set ℕ := {n | n ∈ Finset.range 50 ∧ n ≥ 1}
def sampling_interval : ℕ := 10
def start : ℕ := 6

-- The main theorem stating that the selected students' numbers are as given.
theorem systematic_sampling : ∃ (selected : List ℕ), selected = [6, 16, 26, 36, 46] ∧ 
  ∀ x ∈ selected, x ∈ students := 
  sorry

end systematic_sampling_l158_158760


namespace boys_at_reunion_l158_158324

theorem boys_at_reunion (n : ℕ) (H : n * (n - 1) / 2 = 45) : n = 10 :=
by sorry

end boys_at_reunion_l158_158324


namespace sandy_age_correct_l158_158420

def is_age_ratio (S M : ℕ) : Prop := S * 9 = M * 7
def is_age_difference (S M : ℕ) : Prop := M = S + 12

theorem sandy_age_correct (S M : ℕ) (h1 : is_age_ratio S M) (h2 : is_age_difference S M) : S = 42 := by
  sorry

end sandy_age_correct_l158_158420


namespace max_operations_l158_158954

def arithmetic_mean (a b : ℕ) := (a + b) / 2

theorem max_operations (b : ℕ) (hb : b < 2002) (heven : (2002 + b) % 2 = 0) :
  ∃ n, n = 10 ∧ (2002 - b) / 2^n = 1 :=
by
  sorry

end max_operations_l158_158954


namespace sec_150_eq_neg_two_sqrt_three_over_three_l158_158384

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l158_158384


namespace day_of_100th_day_of_2005_l158_158918

-- Define the days of the week
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Define a function to add days to a given weekday
def add_days (d: Weekday) (n: ℕ) : Weekday :=
  match d with
  | Sunday => [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday].get? (n % 7) |>.getD Sunday
  | Monday => [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday].get? (n % 7) |>.getD Monday
  | Tuesday => [Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Monday].get? (n % 7) |>.getD Tuesday
  | Wednesday => [Wednesday, Thursday, Friday, Saturday, Sunday, Monday, Tuesday].get? (n % 7) |>.getD Wednesday
  | Thursday => [Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday].get? (n % 7) |>.getD Thursday
  | Friday => [Friday, Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday].get? (n % 7) |>.getD Friday
  | Saturday => [Saturday, Sunday, Monday, Tuesday, Wednesday, Thursday, Friday].get? (n % 7) |>.getD Saturday

-- State the theorem
theorem day_of_100th_day_of_2005 :
  add_days Tuesday 55 = Monday :=
by sorry

end day_of_100th_day_of_2005_l158_158918


namespace income_expenditure_ratio_l158_158907

variable (I S E : ℕ)
variable (hI : I = 16000)
variable (hS : S = 3200)
variable (hExp : S = I - E)

theorem income_expenditure_ratio (I S E : ℕ) (hI : I = 16000) (hS : S = 3200) (hExp : S = I - E) : I / Nat.gcd I E = 5 ∧ E / Nat.gcd I E = 4 := by
  sorry

end income_expenditure_ratio_l158_158907


namespace central_angle_of_sector_l158_158809

theorem central_angle_of_sector (r A : ℝ) (h₁ : r = 4) (h₂ : A = 4) :
  (1 / 2) * r^2 * (1 / 4) = A :=
by
  sorry

end central_angle_of_sector_l158_158809


namespace tan_beta_minus_2alpha_l158_158475

theorem tan_beta_minus_2alpha
  (α β : ℝ)
  (h1 : Real.tan α = 1/2)
  (h2 : Real.tan (α - β) = -1/3) :
  Real.tan (β - 2 * α) = -1/7 := 
sorry

end tan_beta_minus_2alpha_l158_158475


namespace Indians_drink_tea_is_zero_l158_158808

-- Definitions based on given conditions and questions
variable (total_people : Nat)
variable (total_drink_tea : Nat)
variable (total_drink_coffee : Nat)
variable (answer_do_you_drink_coffee : Nat)
variable (answer_are_you_a_turk : Nat)
variable (answer_is_it_raining : Nat)
variable (Indians_drink_tea : Nat)
variable (Indians_drink_coffee : Nat)
variable (Turks_drink_coffee : Nat)
variable (Turks_drink_tea : Nat)

-- The given facts and conditions
axiom hx1 : total_people = 55
axiom hx2 : answer_do_you_drink_coffee = 44
axiom hx3 : answer_are_you_a_turk = 33
axiom hx4 : answer_is_it_raining = 22
axiom hx5 : Indians_drink_tea + Indians_drink_coffee + Turks_drink_coffee + Turks_drink_tea = total_people
axiom hx6 : Indians_drink_coffee + Turks_drink_coffee = answer_do_you_drink_coffee
axiom hx7 : Indians_drink_coffee + Turks_drink_tea = answer_are_you_a_turk
axiom hx8 : Indians_drink_tea + Turks_drink_coffee = answer_is_it_raining

-- Prove that the number of Indians drinking tea is 0
theorem Indians_drink_tea_is_zero : Indians_drink_tea = 0 :=
by {
    sorry
}

end Indians_drink_tea_is_zero_l158_158808


namespace time_to_cross_bridge_l158_158593

noncomputable def train_crossing_time
  (length_train : ℕ)
  (length_bridge : ℕ)
  (speed_kmph : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  (length_train + length_bridge) / (speed_kmph * conversion_factor)

theorem time_to_cross_bridge :
  train_crossing_time 135 240 45 (5 / 18) = 30 := by
  sorry

end time_to_cross_bridge_l158_158593


namespace ship_length_in_steps_l158_158920

theorem ship_length_in_steps (E S L : ℝ) (H1 : L + 300 * S = 300 * E) (H2 : L - 60 * S = 60 * E) :
  L = 100 * E :=
by sorry

end ship_length_in_steps_l158_158920


namespace probability_red_side_first_on_third_roll_l158_158140

noncomputable def red_side_probability_first_on_third_roll : ℚ :=
  let p_non_red := 7 / 10
  let p_red := 3 / 10
  (p_non_red * p_non_red * p_red)

theorem probability_red_side_first_on_third_roll :
  red_side_probability_first_on_third_roll = 147 / 1000 := 
sorry

end probability_red_side_first_on_third_roll_l158_158140


namespace value_of_1_minus_a_l158_158382

theorem value_of_1_minus_a (a : ℤ) (h : a = -(-6)) : 1 - a = -5 := 
by 
  sorry

end value_of_1_minus_a_l158_158382


namespace tangent_line_eqn_at_one_l158_158797

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_eqn_at_one :
  let k := (Real.exp 1)
  let p := (1, Real.exp 1)
  ∃ m b : ℝ, (m = k) ∧ (b = p.2 - m * p.1) ∧ (∀ x, f x = y → y = m * x + b) :=
sorry

end tangent_line_eqn_at_one_l158_158797


namespace train_length_l158_158249

theorem train_length (S L : ℝ)
  (h1 : L = S * 11)
  (h2 : L + 120 = S * 22) : 
  L = 120 := 
by
  -- proof goes here
  sorry

end train_length_l158_158249


namespace rectangle_problem_l158_158772

theorem rectangle_problem (x : ℝ) (h1 : 4 * x = l) (h2 : x + 7 = w) (h3 : l * w = 2 * (2 * l + 2 * w)) : x = 1 := 
by {
  sorry
}

end rectangle_problem_l158_158772


namespace cards_per_box_l158_158087

-- Define the conditions
def total_cards : ℕ := 75
def cards_not_in_box : ℕ := 5
def boxes_given_away : ℕ := 2
def boxes_left : ℕ := 5

-- Calculating the total number of boxes initially
def initial_boxes : ℕ := boxes_given_away + boxes_left

-- Define the number of cards in each box
def num_cards_per_box (number_of_cards : ℕ) (number_of_boxes : ℕ) : ℕ :=
  (number_of_cards - cards_not_in_box) / number_of_boxes

-- The proof problem statement
theorem cards_per_box :
  num_cards_per_box total_cards initial_boxes = 10 :=
by
  -- Proof is omitted with sorry
  sorry

end cards_per_box_l158_158087


namespace fraction_of_jenny_bounce_distance_l158_158777

-- Definitions for the problem conditions
def jenny_initial_distance := 18
def jenny_bounce_fraction (f : ℚ) : ℚ := 18 * f
def jenny_total_distance (f : ℚ) : ℚ := jenny_initial_distance + jenny_bounce_fraction f

def mark_initial_distance := 15
def mark_bounce_distance := 2 * mark_initial_distance
def mark_total_distance : ℚ := mark_initial_distance + mark_bounce_distance

def distance_difference := 21

-- The theorem to prove
theorem fraction_of_jenny_bounce_distance (f : ℚ) :
  mark_total_distance = jenny_total_distance f + distance_difference →
  f = 1 / 3 :=
by
  sorry

end fraction_of_jenny_bounce_distance_l158_158777


namespace relationship_between_A_and_p_l158_158293

variable {x y p : ℝ}

theorem relationship_between_A_and_p (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : x ≠ y * 2) (h4 : x ≠ p * y)
  (A : ℝ) (hA : A = (x^2 - 3 * y^2) / (3 * x^2 + y^2))
  (hEq : (p * x * y) / (x^2 - (2 + p) * x * y + 2 * p * y^2) - y / (x - 2 * y) = 1 / 2) :
  A = (9 * p^2 - 3) / (27 * p^2 + 1) := 
sorry

end relationship_between_A_and_p_l158_158293


namespace diff_baseball_soccer_l158_158841

variable (totalBalls soccerBalls basketballs tennisBalls baseballs volleyballs : ℕ)

axiom h1 : totalBalls = 145
axiom h2 : soccerBalls = 20
axiom h3 : basketballs = soccerBalls + 5
axiom h4 : tennisBalls = 2 * soccerBalls
axiom h5 : baseballs > soccerBalls
axiom h6 : volleyballs = 30

theorem diff_baseball_soccer : baseballs - soccerBalls = 10 :=
  by {
    sorry
  }

end diff_baseball_soccer_l158_158841


namespace largest_int_less_than_100_rem_5_by_7_l158_158416

theorem largest_int_less_than_100_rem_5_by_7 :
  ∃ k : ℤ, (7 * k + 5 = 96) ∧ ∀ n : ℤ, (7 * n + 5 < 100) → (n ≤ k) :=
sorry

end largest_int_less_than_100_rem_5_by_7_l158_158416


namespace part_a_l158_158368

theorem part_a (c : ℤ) : (∃ x : ℤ, x + (x / 2) = c) ↔ (c % 3 ≠ 2) :=
sorry

end part_a_l158_158368


namespace proposition_C_is_true_l158_158545

theorem proposition_C_is_true :
  (∀ θ : ℝ, 90 < θ ∧ θ < 180 → θ > 90) :=
by
  sorry

end proposition_C_is_true_l158_158545


namespace fill_cistern_time_l158_158380

theorem fill_cistern_time (A B C : ℕ) (hA : A = 10) (hB : B = 12) (hC : C = 50) :
    1 / (1 / A + 1 / B - 1 / C) = 300 / 49 :=
by
  sorry

end fill_cistern_time_l158_158380


namespace simplify_fractional_exponents_l158_158033

theorem simplify_fractional_exponents :
  (5 ^ (1/6) * 5 ^ (1/2)) / 5 ^ (1/3) = 5 ^ (1/6) :=
by
  sorry

end simplify_fractional_exponents_l158_158033


namespace sand_needed_l158_158588

def area_rectangular_patch : ℕ := 6 * 7
def area_square_patch : ℕ := 5 * 5
def sand_per_square_inch : ℕ := 3

theorem sand_needed : area_rectangular_patch + area_square_patch * sand_per_square_inch = 201 := sorry

end sand_needed_l158_158588


namespace number_of_students_l158_158779

def total_students (a b : ℕ) : ℕ :=
  a + b

variables (a b : ℕ)

theorem number_of_students (h : 48 * a + 45 * b = 972) : total_students a b = 21 :=
by
  sorry

end number_of_students_l158_158779


namespace percentage_of_a_added_to_get_x_l158_158755

variable (a b x m : ℝ) (P : ℝ) (k : ℝ)
variable (h1 : a / b = 4 / 5)
variable (h2 : x = a * (1 + P / 100))
variable (h3 : m = b * 0.2)
variable (h4 : m / x = 0.14285714285714285)

theorem percentage_of_a_added_to_get_x :
  P = 75 :=
by
  sorry

end percentage_of_a_added_to_get_x_l158_158755


namespace compute_expression_l158_158647

theorem compute_expression : (-9 * 3 - (-7 * -4) + (-11 * -6) = 11) := by
  sorry

end compute_expression_l158_158647


namespace slope_angle_y_eq_neg1_l158_158300

theorem slope_angle_y_eq_neg1 : (∃ line : ℝ → ℝ, ∀ y: ℝ, line y = -1 → ∃ θ : ℝ, θ = 0) :=
by
  -- Sorry is used to skip the proof.
  sorry

end slope_angle_y_eq_neg1_l158_158300


namespace smallest_num_rectangles_l158_158784

theorem smallest_num_rectangles (a b : ℕ) (h_a : a = 3) (h_b : b = 4) : 
  ∃ n : ℕ, n = 12 ∧ ∀ s : ℕ, (s = lcm a b) → s^2 / (a * b) = 12 :=
by 
  sorry

end smallest_num_rectangles_l158_158784


namespace find_a_l158_158720

theorem find_a :
  ∃ a : ℝ, (2 * x - (a * Real.exp x + x) + 1 = 0) = (a = 1) :=
by
  sorry

end find_a_l158_158720


namespace sum_of_interior_edges_l158_158430

-- Conditions
def width_of_frame_piece : ℝ := 1.5
def one_interior_edge : ℝ := 4.5
def total_frame_area : ℝ := 27

-- Statement of the problem as a theorem in Lean
theorem sum_of_interior_edges : 
  (∃ y : ℝ, (width_of_frame_piece * 2 + one_interior_edge) * (width_of_frame_piece * 2 + y) 
    - one_interior_edge * y = total_frame_area) →
  (4 * (one_interior_edge + y) = 12) :=
sorry

end sum_of_interior_edges_l158_158430


namespace additional_hours_equal_five_l158_158231

-- The total hovering time constraint over two days
def total_time : ℕ := 24

-- Hovering times for each zone on the first day
def day1_mountain_time : ℕ := 3
def day1_central_time : ℕ := 4
def day1_eastern_time : ℕ := 2

-- Additional hours on the second day (variables M, C, E)
variables (M C E : ℕ)

-- The main proof statement
theorem additional_hours_equal_five 
  (h : day1_mountain_time + M + day1_central_time + C + day1_eastern_time + E = total_time) :
  M = 5 ∧ C = 5 ∧ E = 5 :=
by
  sorry

end additional_hours_equal_five_l158_158231


namespace inequality_solution_set_l158_158134

theorem inequality_solution_set :
  {x : ℝ | (x^2 - 4) / (x^2 - 9) > 0} = {x : ℝ | x < -3 ∨ x > 3} :=
sorry

end inequality_solution_set_l158_158134


namespace percent_brandA_in_mix_l158_158774

theorem percent_brandA_in_mix (x : Real) :
  (0.60 * x + 0.35 * (100 - x) = 50) → x = 60 :=
by
  intro h
  sorry

end percent_brandA_in_mix_l158_158774


namespace student_weight_l158_158136

theorem student_weight (S R : ℕ) (h1 : S - 5 = 2 * R) (h2 : S + R = 116) : S = 79 :=
sorry

end student_weight_l158_158136


namespace minimum_value_of_F_l158_158174

theorem minimum_value_of_F (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_odd_g : ∀ x, g (-x) = -g x) (h_max_F : ∃ x > 0, a * f x + b * g x + 3 = 10) 
  : ∃ x < 0, a * f x + b * g x + 3 = -4 := 
sorry

end minimum_value_of_F_l158_158174


namespace M_eq_N_l158_158224

-- Define the sets M and N
def M : Set ℤ := {u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r}

-- Prove that M equals N
theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l158_158224


namespace prove_k_eq_one_l158_158343

theorem prove_k_eq_one 
  (n m k : ℕ) 
  (h_positive : 0 < n)  -- implies n, and hence n-1, n+1, are all positive
  (h_eq : (n-1) * n * (n+1) = m^k): 
  k = 1 := 
sorry

end prove_k_eq_one_l158_158343


namespace train_passes_in_two_minutes_l158_158652

noncomputable def time_to_pass_through_tunnel : ℕ := 
  let train_length := 100 -- Length of the train in meters
  let train_speed := 72 * 1000 / 60 -- Speed of the train in m/min (converted)
  let tunnel_length := 2300 -- Length of the tunnel in meters (converted from 2.3 km to meters)
  let total_distance := train_length + tunnel_length -- Total distance to travel
  total_distance / train_speed -- Time in minutes (total distance divided by speed)

theorem train_passes_in_two_minutes : time_to_pass_through_tunnel = 2 := 
  by
  -- proof would go here, but for this statement, we use 'sorry'
  sorry

end train_passes_in_two_minutes_l158_158652


namespace hyperbola_equation_l158_158948

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end hyperbola_equation_l158_158948


namespace equation_of_parallel_line_l158_158337

theorem equation_of_parallel_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (x y : ℝ) (m : ℝ) (H_1 : P = (1, 2)) (H_2 : ∀ x y m, l x y ↔ (2 * x + y + m = 0) )
  (H_3 : l x y) : 
  l 2 (y - 4) := 
  sorry

end equation_of_parallel_line_l158_158337


namespace line_intersects_circle_l158_158936

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (kx - y - k +1 = 0) ∧ (x^2 + y^2 = 4) :=
sorry

end line_intersects_circle_l158_158936


namespace find_PF2_l158_158488

open Real

noncomputable def hyperbola_equation (x y : ℝ) := (x^2 / 16) - (y^2 / 20) = 1

noncomputable def distance (P F : ℝ × ℝ) : ℝ := 
  let (px, py) := P
  let (fx, fy) := F
  sqrt ((px - fx)^2 + (py - fy)^2)

theorem find_PF2
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (on_hyperbola : hyperbola_equation P.1 P.2)
  (foci_F1_F2 : F1 = (-6, 0) ∧ F2 = (6, 0))
  (distance_PF1 : distance P F1 = 9) : 
  distance P F2 = 17 := 
by
  sorry

end find_PF2_l158_158488


namespace no_real_solution_l158_158594

theorem no_real_solution :
    ∀ x : ℝ, (5 * x^2 - 3 * x + 2) / (x + 2) ≠ 2 * x - 3 :=
by
  intro x
  sorry

end no_real_solution_l158_158594


namespace y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l158_158586

def y : ℕ := 42 + 98 + 210 + 333 + 175 + 28

theorem y_not_multiple_of_7 : ¬ (7 ∣ y) := sorry
theorem y_not_multiple_of_14 : ¬ (14 ∣ y) := sorry
theorem y_not_multiple_of_21 : ¬ (21 ∣ y) := sorry
theorem y_not_multiple_of_28 : ¬ (28 ∣ y) := sorry

end y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l158_158586


namespace pyramid_x_value_l158_158794

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value_l158_158794


namespace warehouse_problem_l158_158713

/-- 
Problem Statement:
A certain unit decides to invest 3200 yuan to build a warehouse (in the shape of a rectangular prism) with a constant height.
The back wall will be built reusing the old wall at no cost, the front will be made of iron grilles at a cost of 40 yuan per meter in length,
and the two side walls will be built with bricks at a cost of 45 yuan per meter in length.
The top will have a cost of 20 yuan per square meter.
Let the length of the iron grilles be x meters and the length of one brick wall be y meters.
Find:
1. Write down the relationship between x and y.
2. Determine the maximum allowable value of the warehouse area S. In order to maximize S without exceeding the budget, how long should the front iron grille be designed
-/

theorem warehouse_problem (x y : ℝ) :
    (40 * x + 90 * y + 20 * x * y = 3200 ∧ 0 < x ∧ x < 80) →
    (y = (320 - 4 * x) / (9 + 2 * x) ∧ x = 15 ∧ y = 20 / 3 ∧ x * y = 100) :=
by
  sorry

end warehouse_problem_l158_158713


namespace greater_expected_area_vasya_l158_158999

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end greater_expected_area_vasya_l158_158999


namespace inverse_variation_solution_l158_158000

theorem inverse_variation_solution :
  ∀ (x y k : ℝ),
    (x * y^3 = k) →
    (∃ k, x = 8 ∧ y = 1 ∧ k = 8) →
    (y = 2 → x = 1) :=
by
  intros x y k h1 h2 hy2
  sorry

end inverse_variation_solution_l158_158000


namespace Cameron_task_completion_l158_158458

theorem Cameron_task_completion (C : ℝ) (h1 : ∃ x, x = 9 / C) (h2 : ∃ y, y = 1 / 2) (total_work : ∃ z, z = 1):
  9 - 9 / C + 1/2 = 1 -> C = 18 := by
  sorry

end Cameron_task_completion_l158_158458


namespace speed_of_second_part_of_trip_l158_158780

-- Given conditions
def total_distance : Real := 50
def first_part_distance : Real := 25
def first_part_speed : Real := 66
def average_speed : Real := 44.00000000000001

-- The statement we want to prove
theorem speed_of_second_part_of_trip :
  ∃ second_part_speed : Real, second_part_speed = 33 :=
by
  sorry

end speed_of_second_part_of_trip_l158_158780


namespace garden_ratio_l158_158093

theorem garden_ratio 
  (P : ℕ) (L : ℕ) (W : ℕ) 
  (h1 : P = 900) 
  (h2 : L = 300) 
  (h3 : P = 2 * (L + W)) : 
  L / W = 2 :=
by 
  sorry

end garden_ratio_l158_158093


namespace infinite_integer_triples_solution_l158_158311

theorem infinite_integer_triples_solution (a b c : ℤ) : 
  ∃ (a b c : ℤ), ∀ n : ℤ, a^2 + b^2 = c^2 + 3 :=
sorry

end infinite_integer_triples_solution_l158_158311


namespace calculate_total_money_made_l158_158157

def original_price : ℕ := 51
def discount : ℕ := 8
def num_tshirts_sold : ℕ := 130
def discounted_price : ℕ := original_price - discount
def total_money_made : ℕ := discounted_price * num_tshirts_sold

theorem calculate_total_money_made :
  total_money_made = 5590 := 
sorry

end calculate_total_money_made_l158_158157


namespace exists_multiple_with_odd_digit_sum_l158_158125

theorem exists_multiple_with_odd_digit_sum (M : Nat) :
  ∃ N : Nat, N % M = 0 ∧ (Nat.digits 10 N).sum % 2 = 1 :=
by
  sorry

end exists_multiple_with_odd_digit_sum_l158_158125


namespace arithmetic_sequence_k_value_l158_158667

theorem arithmetic_sequence_k_value 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_first_term : a 1 = 0) 
  (h_nonzero_diff : d ≠ 0) 
  (h_sum : ∃ k, a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) : 
  ∃ k, k = 22 := 
by 
  sorry

end arithmetic_sequence_k_value_l158_158667


namespace table_covered_with_three_layers_l158_158669

theorem table_covered_with_three_layers (A T table_area two_layers : ℕ)
    (hA : A = 204)
    (htable : table_area = 175)
    (hcover : 140 = 80 * table_area / 100)
    (htwo_layers : two_layers = 24) :
    3 * T + 2 * two_layers + (140 - two_layers - T) = 204 → T = 20 := by
  sorry

end table_covered_with_three_layers_l158_158669


namespace num_integers_div_10_or_12_l158_158783

-- Define the problem in Lean
theorem num_integers_div_10_or_12 (N : ℕ) : (1 ≤ N ∧ N ≤ 2007) ∧ (N % 10 = 0 ∨ N % 12 = 0) ↔ ∃ k, k = 334 := by
  sorry

end num_integers_div_10_or_12_l158_158783


namespace team_arrangements_l158_158282

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem team_arrangements :
  let num_players := 10
  let team_blocks := 4
  let cubs_players := 3
  let red_sox_players := 3
  let yankees_players := 2
  let dodgers_players := 2
  (factorial team_blocks) * (factorial cubs_players) * (factorial red_sox_players) * (factorial yankees_players) * (factorial dodgers_players) = 3456 := 
by
  -- Proof steps will be inserted here
  sorry

end team_arrangements_l158_158282


namespace no_such_n_exists_l158_158438

theorem no_such_n_exists : ∀ n : ℕ, n > 1 → ∀ (p1 p2 : ℕ), 
  (Nat.Prime p1) → (Nat.Prime p2) → n = p1^2 → n + 60 = p2^2 → False :=
by
  intro n hn p1 p2 hp1 hp2 h1 h2
  sorry

end no_such_n_exists_l158_158438


namespace fred_found_43_seashells_l158_158991

-- Define the conditions
def tom_seashells : ℕ := 15
def additional_seashells : ℕ := 28

-- Define Fred's total seashells based on the conditions
def fred_seashells : ℕ := tom_seashells + additional_seashells

-- The theorem to prove that Fred found 43 seashells
theorem fred_found_43_seashells : fred_seashells = 43 :=
by
  -- Proof goes here
  sorry

end fred_found_43_seashells_l158_158991


namespace value_of_smaller_denom_l158_158945

-- We are setting up the conditions given in the problem.
variables (x : ℕ) -- The value of the smaller denomination bill.

-- Condition 1: She has 4 bills of denomination x.
def value_smaller_denomination : ℕ := 4 * x

-- Condition 2: She has 8 bills of $10 denomination.
def value_ten_bills : ℕ := 8 * 10

-- Condition 3: The total value of the bills is $100.
def total_value : ℕ := 100

-- Prove that x = 5 using the given conditions.
theorem value_of_smaller_denom : value_smaller_denomination x + value_ten_bills = total_value → x = 5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end value_of_smaller_denom_l158_158945


namespace perimeter_of_square_l158_158165

theorem perimeter_of_square (s : ℝ) (area : s^2 = 468) : 4 * s = 24 * Real.sqrt 13 := 
by
  sorry

end perimeter_of_square_l158_158165


namespace reflection_correct_l158_158508

/-- Definition of reflection across the line y = -x -/
def reflection_across_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- Given points C and D, and their images C' and D' respectively, under reflection,
    prove the transformation is correct. -/
theorem reflection_correct :
  (reflection_across_y_eq_neg_x (-3, 2) = (3, -2)) ∧ (reflection_across_y_eq_neg_x (-2, 5) = (2, -5)) :=
  by
    sorry

end reflection_correct_l158_158508


namespace constant_in_quadratic_eq_l158_158080

theorem constant_in_quadratic_eq (C : ℝ) (x₁ x₂ : ℝ) 
  (h1 : 2 * x₁ * x₁ + 5 * x₁ - C = 0) 
  (h2 : 2 * x₂ * x₂ + 5 * x₂ - C = 0) 
  (h3 : x₁ - x₂ = 5.5) : C = 12 := 
sorry

end constant_in_quadratic_eq_l158_158080


namespace valentine_day_spending_l158_158690

structure DogTreatsConfig where
  heart_biscuits_count_A : Nat
  puppy_boots_count_A : Nat
  small_toy_count_A : Nat
  heart_biscuits_count_B : Nat
  puppy_boots_count_B : Nat
  large_toy_count_B : Nat
  heart_biscuit_price : Nat
  puppy_boots_price : Nat
  small_toy_price : Nat
  large_toy_price : Nat
  heart_biscuits_discount : Float
  large_toy_discount : Float

def treats_config : DogTreatsConfig :=
  { heart_biscuits_count_A := 5
    puppy_boots_count_A := 1
    small_toy_count_A := 1
    heart_biscuits_count_B := 7
    puppy_boots_count_B := 2
    large_toy_count_B := 1
    heart_biscuit_price := 2
    puppy_boots_price := 15
    small_toy_price := 10
    large_toy_price := 20
    heart_biscuits_discount := 0.20
    large_toy_discount := 0.15 }

def total_discounted_amount_spent (cfg : DogTreatsConfig) : Float :=
  let heart_biscuits_total_cost := (cfg.heart_biscuits_count_A + cfg.heart_biscuits_count_B) * cfg.heart_biscuit_price
  let puppy_boots_total_cost := (cfg.puppy_boots_count_A * cfg.puppy_boots_price) + (cfg.puppy_boots_count_B * cfg.puppy_boots_price)
  let small_toy_total_cost := cfg.small_toy_count_A * cfg.small_toy_price
  let large_toy_total_cost := cfg.large_toy_count_B * cfg.large_toy_price
  let total_cost_without_discount := Float.ofNat (heart_biscuits_total_cost + puppy_boots_total_cost + small_toy_total_cost + large_toy_total_cost)
  let heart_biscuits_discount_amount := cfg.heart_biscuits_discount * Float.ofNat heart_biscuits_total_cost
  let large_toy_discount_amount := cfg.large_toy_discount * Float.ofNat large_toy_total_cost
  let total_discount_amount := heart_biscuits_discount_amount + large_toy_discount_amount
  total_cost_without_discount - total_discount_amount

theorem valentine_day_spending : total_discounted_amount_spent treats_config = 91.20 := by
  sorry

end valentine_day_spending_l158_158690


namespace three_squares_not_divisible_by_three_l158_158946

theorem three_squares_not_divisible_by_three 
  (N : ℕ) (a b c : ℤ) 
  (h₁ : N = 9 * (a^2 + b^2 + c^2)) :
  ∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) := 
sorry

end three_squares_not_divisible_by_three_l158_158946


namespace x_intercept_of_parabola_l158_158869

theorem x_intercept_of_parabola (a b c : ℝ)
    (h_vertex : ∀ x, (a * (x - 5)^2 + 9 = y) → (x, y) = (5, 9))
    (h_intercept : ∀ x, (a * x^2 + b * x + c = 0) → x = 0 ∨ y = 0) :
    ∃ x0 : ℝ, x0 = 10 :=
by
  sorry

end x_intercept_of_parabola_l158_158869


namespace inequality_proof_l158_158431

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

theorem inequality_proof :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) >= 3 / 2 :=
by
  sorry

end inequality_proof_l158_158431


namespace area_PST_correct_l158_158614

noncomputable def area_of_triangle_PST : ℚ :=
  let P : ℚ × ℚ := (0, 0)
  let Q : ℚ × ℚ := (4, 0)
  let R : ℚ × ℚ := (0, 4)
  let S : ℚ × ℚ := (0, 2)
  let T : ℚ × ℚ := (8 / 3, 4 / 3)
  1 / 2 * (|P.1 * (S.2 - T.2) + S.1 * (T.2 - P.2) + T.1 * (P.2 - S.2)|)

theorem area_PST_correct : area_of_triangle_PST = 8 / 3 := sorry

end area_PST_correct_l158_158614


namespace possible_values_of_m_l158_158986

theorem possible_values_of_m
  (m : ℕ)
  (h1 : ∃ (m' : ℕ), m = m' ∧ 0 < m)            -- m is a positive integer
  (h2 : 2 * (m - 1) + 3 * (m + 2) > 4 * (m - 5))    -- AB + AC > BC
  (h3 : 2 * (m - 1) + 4 * (m + 5) > 3 * (m + 2))    -- AB + BC > AC
  (h4 : 3 * (m + 2) + 4 * (m + 5) > 2 * (m - 1))    -- AC + BC > AB
  (h5 : 3 * (m + 2) > 2 * (m - 1))                  -- AC > AB
  (h6 : 4 * (m + 5) > 3 * (m + 2))                  -- BC > AC
  : m ≥ 7 := 
sorry

end possible_values_of_m_l158_158986


namespace jerrys_age_l158_158478

theorem jerrys_age (M J : ℕ) (h1 : M = 3 * J - 4) (h2 : M = 14) : J = 6 :=
by 
  sorry

end jerrys_age_l158_158478


namespace chess_team_boys_l158_158778

variable {B G : ℕ}

theorem chess_team_boys
    (h1 : B + G = 30)
    (h2 : 1/3 * G + B = 18) :
    B = 12 :=
by
  sorry

end chess_team_boys_l158_158778


namespace true_propositions_l158_158200

-- Definitions for the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (a b : ℝ) : Prop := a^2 > b^2 → |a| > |b|
def proposition3 (a b c : ℝ) : Prop := (a > b ↔ a + c > b + c)

-- Theorem to state the true propositions
theorem true_propositions (a b c : ℝ) :
  -- Proposition 3 is true
  (proposition3 a b c) →
  -- Assert that the serial number of the true propositions is 3
  {3} = { i | (i = 1 ∧ proposition1 a b) ∨ (i = 2 ∧ proposition2 a b) ∨ (i = 3 ∧ proposition3 a b c)} :=
by
  sorry

end true_propositions_l158_158200


namespace axis_of_symmetry_l158_158781

-- Given conditions
variables {b c : ℝ}
axiom eq_roots : ∃ (x1 x2 : ℝ), (x1 = -1 ∧ x2 = 2) ∧ (x1 + x2 = -b) ∧ (x1 * x2 = c)

-- Question translation to Lean statement
theorem axis_of_symmetry : 
  ∀ b c, 
  (∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ x1 + x2 = -b ∧ x1 * x2 = c) 
  → -b / 2 = 1 / 2 := 
by 
  sorry

end axis_of_symmetry_l158_158781


namespace loaned_out_books_l158_158833

def initial_books : ℕ := 75
def added_books : ℕ := 10 + 15 + 6
def removed_books : ℕ := 3 + 2 + 4
def end_books : ℕ := 90
def return_percentage : ℝ := 0.80

theorem loaned_out_books (L : ℕ) :
  (end_books - initial_books = added_books - removed_books - ⌊(1 - return_percentage) * L⌋) →
  (L = 35) :=
sorry

end loaned_out_books_l158_158833


namespace max_of_expression_l158_158683

theorem max_of_expression (a b c : ℝ) (hbc : b > c) (hca : c > a) (ha : a > 0) (hb : b > 0) (hc : c > 0) (ha_nonzero : a ≠ 0) :
  ∃ (max_val : ℝ), max_val = 44 ∧ (∀ x, x = (2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 → x ≤ max_val) := 
sorry

end max_of_expression_l158_158683


namespace ihsan_children_l158_158317

theorem ihsan_children :
  ∃ n : ℕ, (n + n^2 + n^3 + n^4 = 2800) ∧ (n = 7) :=
sorry

end ihsan_children_l158_158317


namespace boundary_length_is_25_point_7_l158_158455

-- Define the side length derived from the given area.
noncomputable def sideLength (area : ℝ) : ℝ :=
  Real.sqrt area

-- Define the length of each segment when the square's side is divided into four equal parts.
noncomputable def segmentLength (side : ℝ) : ℝ :=
  side / 4

-- Define the total boundary length, which includes the circumference of the quarter-circle arcs and the straight segments.
noncomputable def totalBoundaryLength (area : ℝ) : ℝ :=
  let side := sideLength area
  let segment := segmentLength side
  let arcsLength := 2 * Real.pi * segment  -- the full circle's circumference
  let straightLength := 4 * segment
  arcsLength + straightLength

-- State the theorem that the total boundary length is approximately 25.7 units.
theorem boundary_length_is_25_point_7 :
  totalBoundaryLength 100 = 5 * Real.pi + 10 :=
by sorry

end boundary_length_is_25_point_7_l158_158455


namespace lemonade_water_l158_158776

theorem lemonade_water (L S W : ℝ) (h1 : S = 1.5 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 18 :=
by
  sorry

end lemonade_water_l158_158776


namespace inequality_holds_for_interval_l158_158327

theorem inequality_holds_for_interval (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 5 → x^2 - 2 * (a - 2) * x + a < 0) → a ≥ 5 :=
by
  intros h
  sorry

end inequality_holds_for_interval_l158_158327


namespace article_word_limit_l158_158254

theorem article_word_limit 
  (total_pages : ℕ) (large_font_pages : ℕ) (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) (remaining_pages : ℕ) (total_words : ℕ)
  (h1 : total_pages = 21) 
  (h2 : large_font_pages = 4) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) 
  (h5 : remaining_pages = total_pages - large_font_pages) 
  (h6 : total_words = large_font_pages * words_per_large_page + remaining_pages * words_per_small_page) :
  total_words = 48000 := 
by
  sorry

end article_word_limit_l158_158254


namespace total_weight_of_nuts_l158_158344

theorem total_weight_of_nuts:
  let almonds := 0.14
  let pecans := 0.38
  let walnuts := 0.22
  let cashews := 0.47
  let pistachios := 0.29
  almonds + pecans + walnuts + cashews + pistachios = 1.50 :=
by
  sorry

end total_weight_of_nuts_l158_158344


namespace number_of_employees_l158_158547

def fixed_time_coffee : ℕ := 5
def time_per_status_update : ℕ := 2
def time_per_payroll_update : ℕ := 3
def total_morning_routine : ℕ := 50

def time_per_employee : ℕ := time_per_status_update + time_per_payroll_update
def time_spent_on_employees : ℕ := total_morning_routine - fixed_time_coffee

theorem number_of_employees : (time_spent_on_employees / time_per_employee) = 9 := by
  sorry

end number_of_employees_l158_158547


namespace last_three_digits_of_8_pow_108_l158_158718

theorem last_three_digits_of_8_pow_108 :
  (8^108 % 1000) = 38 := 
sorry

end last_three_digits_of_8_pow_108_l158_158718


namespace sufficient_not_necessary_l158_158971

theorem sufficient_not_necessary (x y : ℝ) : (x > |y|) → (x > y ∧ ¬ (x > y → x > |y|)) :=
by
  sorry

end sufficient_not_necessary_l158_158971


namespace part1_l158_158433

theorem part1 (x : ℝ) (hx : x > 0) : 
  (1 / (2 * Real.sqrt (x + 1))) < (Real.sqrt (x + 1) - Real.sqrt x) ∧ (Real.sqrt (x + 1) - Real.sqrt x) < (1 / (2 * Real.sqrt x)) := 
sorry

end part1_l158_158433


namespace fraction_is_five_sixths_l158_158658

-- Define the conditions as given in the problem
def number : ℝ := -72.0
def target_value : ℝ := -60

-- The statement we aim to prove
theorem fraction_is_five_sixths (f : ℝ) (h : f * number = target_value) : f = 5/6 :=
  sorry

end fraction_is_five_sixths_l158_158658


namespace xyz_solution_l158_158645

theorem xyz_solution (x y z : ℂ) (h1 : x * y + 5 * y = -20) 
                                 (h2 : y * z + 5 * z = -20) 
                                 (h3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := 
sorry

end xyz_solution_l158_158645


namespace tan_sum_product_l158_158116

theorem tan_sum_product (tan : ℝ → ℝ) : 
  (1 + tan 23) * (1 + tan 22) = 2 + tan 23 * tan 22 := by sorry

end tan_sum_product_l158_158116


namespace inverse_proportion_quadrant_l158_158220

theorem inverse_proportion_quadrant (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, (0 < x → y = k / x → y < 0) ∧ (x < 0 → y = k / x → 0 < y) :=
by
  sorry

end inverse_proportion_quadrant_l158_158220


namespace C_is_necessary_but_not_sufficient_for_A_l158_158635

-- Define C, B, A to be logical propositions
variables (A B C : Prop)

-- The conditions given
axiom h1 : A → B
axiom h2 : ¬ (B → A)
axiom h3 : B ↔ C

-- The conclusion: Prove that C is a necessary but not sufficient condition for A
theorem C_is_necessary_but_not_sufficient_for_A : (A → C) ∧ ¬ (C → A) :=
by
  sorry

end C_is_necessary_but_not_sufficient_for_A_l158_158635


namespace cyclic_inequality_l158_158198

theorem cyclic_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^3 * b^3 * (a * b - a * c - b * c + c^2) +
   b^3 * c^3 * (b * c - b * a - c * a + a^2) +
   c^3 * a^3 * (c * a - c * b - a * b + b^2)) ≥ 0 :=
sorry

end cyclic_inequality_l158_158198


namespace geometric_sequence_problem_l158_158410

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 3 * a 7 = 8)
  (h2 : a 4 + a 6 = 6)
  (h_geom : ∀ n, a n = a 1 * q ^ (n - 1)):
  a 2 + a 8 = 9 :=
sorry

end geometric_sequence_problem_l158_158410


namespace find_value_of_s_l158_158454

variable {r s : ℝ}

theorem find_value_of_s (hr : r > 1) (hs : s > 1) (h1 : 1/r + 1/s = 1) (h2 : r * s = 9) :
  s = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_value_of_s_l158_158454


namespace relationship_of_a_b_l158_158440

theorem relationship_of_a_b
  (a b : Real)
  (h1 : a < 0)
  (h2 : b > 0)
  (h3 : a + b < 0) : 
  -a > b ∧ b > -b ∧ -b > a := 
by
  sorry

end relationship_of_a_b_l158_158440


namespace problem1_problem2_problem3_l158_158291

-- Problem 1 Statement
theorem problem1 : (π - 3.14)^0 + (1 / 2)^(-1) + (-1)^(2023) = 2 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 2 Statement
theorem problem2 (b : ℝ) : (-b)^2 * b + 6 * b^4 / (2 * b) + (-2 * b)^3 = -4 * b^3 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 3 Statement
theorem problem3 (x : ℝ) : (x - 1)^2 - x * (x + 2) = -4 * x + 1 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

end problem1_problem2_problem3_l158_158291


namespace number_of_females_l158_158674

theorem number_of_females 
  (total_students : ℕ) 
  (sampled_students : ℕ) 
  (sampled_female_less_than_male : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sampled_students = 200)
  (h_diff : sampled_female_less_than_male = 20) : 
  ∃ F M : ℕ, F + M = total_students ∧ (F / M : ℝ) = 9 / 11 ∧ F = 720 :=
by
  sorry

end number_of_females_l158_158674


namespace number_of_salads_bought_l158_158573

variable (hot_dogs_cost : ℝ := 5 * 1.50)
variable (initial_money : ℝ := 2 * 10)
variable (change_given_back : ℝ := 5)
variable (total_spent : ℝ := initial_money - change_given_back)
variable (salad_cost : ℝ := 2.50)

theorem number_of_salads_bought : (total_spent - hot_dogs_cost) / salad_cost = 3 := 
by 
  sorry

end number_of_salads_bought_l158_158573


namespace max_type_A_stationery_l158_158671

-- Define the variables and constraints
variables (x y : ℕ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * x + 2 * (x - 2) + y = 66
def condition2 : Prop := 3 * x ≤ 33

-- The statement to prove
theorem max_type_A_stationery : condition1 x y ∧ condition2 x → x ≤ 11 :=
by sorry

end max_type_A_stationery_l158_158671


namespace find_next_score_l158_158678

def scores := [95, 85, 75, 65, 90]
def current_avg := (95 + 85 + 75 + 65 + 90) / 5
def target_avg := current_avg + 4

theorem find_next_score (s : ℕ) (h : (95 + 85 + 75 + 65 + 90 + s) / 6 = target_avg) : s = 106 :=
by
  -- Proof steps here
  sorry

end find_next_score_l158_158678


namespace percentage_markup_l158_158428

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 3000 for a computer table, and the cost price of the computer table was Rs. 2500.
Prove that the percentage markup on the cost price is 20%.
-/
theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 3000) (h₂ : cost_price = 2500) :
  ((selling_price - cost_price) / cost_price) * 100 = 20 :=
by
  -- proof omitted
  sorry

end percentage_markup_l158_158428


namespace directrix_of_parabola_l158_158124

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = -3 * x^2 + 6 * x - 5 → y = -35 / 18 := by
  sorry

end directrix_of_parabola_l158_158124


namespace side_length_of_S2_l158_158886

variable (r s : ℝ)

theorem side_length_of_S2 (h1 : 2 * r + s = 2100) (h2 : 2 * r + 3 * s = 3400) : s = 650 := by
  sorry

end side_length_of_S2_l158_158886


namespace right_triangle_exists_l158_158753

theorem right_triangle_exists :
  (3^2 + 4^2 = 5^2) ∧ ¬(2^2 + 3^2 = 4^2) ∧ ¬(4^2 + 6^2 = 7^2) ∧ ¬(5^2 + 11^2 = 12^2) :=
by
  sorry

end right_triangle_exists_l158_158753


namespace harmonic_mean_ordered_pairs_l158_158984

theorem harmonic_mean_ordered_pairs :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b : ℕ), 
    0 < a ∧ 0 < b ∧ a < b ∧ (2 * a * b = 2 ^ 24 * (a + b)) → n = 23 :=
by sorry

end harmonic_mean_ordered_pairs_l158_158984


namespace small_supermarkets_sample_count_l158_158761

def large := 300
def medium := 600
def small := 2100
def sample_size := 100
def total := large + medium + small

theorem small_supermarkets_sample_count :
  small * (sample_size / total) = 70 := by
  sorry

end small_supermarkets_sample_count_l158_158761


namespace geometric_sequence_Sn_geometric_sequence_Sn_l158_158572

noncomputable def Sn (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1/3 then (27/2) - (1/2) * 3^(n - 3)
  else if q = 3 then (3^n - 1) / 2
  else 0

theorem geometric_sequence_Sn (a1 : ℝ) (n : ℕ) (h1 : a1 * (1/3) = 3)
  (h2 : a1 + a1 * (1/3)^2 = 10) : 
  Sn a1 (1/3) n = (27/2) - (1/2) * 3^(n - 3) :=
by
  sorry

theorem geometric_sequence_Sn' (a1 : ℝ) (n : ℕ) (h1 : a1 * 3 = 3) 
  (h2 : a1 + a1 * 3^2 = 10) : 
  Sn a1 3 n = (3^n - 1) / 2 :=
by
  sorry

end geometric_sequence_Sn_geometric_sequence_Sn_l158_158572


namespace Iesha_num_books_about_school_l158_158612

theorem Iesha_num_books_about_school (total_books sports_books : ℕ) (h1 : total_books = 58) (h2 : sports_books = 39) : total_books - sports_books = 19 :=
by
  sorry

end Iesha_num_books_about_school_l158_158612


namespace find_n_l158_158244

theorem find_n (x n : ℝ) (h1 : ((x / n) * 5) + 10 - 12 = 48) (h2 : x = 40) : n = 4 :=
sorry

end find_n_l158_158244


namespace relationship_of_y_values_l158_158655

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l158_158655


namespace caltech_equilateral_triangles_l158_158299

theorem caltech_equilateral_triangles (n : ℕ) (h : n = 900) :
  let total_triangles := (n * (n - 1) / 2) * 2
  let overcounted_triangles := n / 3
  total_triangles - overcounted_triangles = 808800 :=
by
  sorry

end caltech_equilateral_triangles_l158_158299


namespace mike_weekly_avg_time_l158_158852

theorem mike_weekly_avg_time :
  let mon_wed_fri_tv := 4 -- hours per day on Mon, Wed, Fri
  let tue_thu_tv := 3 -- hours per day on Tue, Thu
  let weekend_tv := 5 -- hours per day on weekends
  let num_mon_wed_fri := 3 -- days
  let num_tue_thu := 2 -- days
  let num_weekend := 2 -- days
  let num_days_week := 7 -- days
  let num_video_game_days := 3 -- days
  let weeks := 4 -- weeks
  let mon_wed_fri_total := mon_wed_fri_tv * num_mon_wed_fri
  let tue_thu_total := tue_thu_tv * num_tue_thu
  let weekend_total := weekend_tv * num_weekend
  let weekly_tv_time := mon_wed_fri_total + tue_thu_total + weekend_total
  let daily_avg_tv_time := weekly_tv_time / num_days_week
  let daily_video_game_time := daily_avg_tv_time / 2
  let weekly_video_game_time := daily_video_game_time * num_video_game_days
  let total_tv_time_4_weeks := weekly_tv_time * weeks
  let total_video_game_time_4_weeks := weekly_video_game_time * weeks
  let total_time_4_weeks := total_tv_time_4_weeks + total_video_game_time_4_weeks
  let weekly_avg_time := total_time_4_weeks / weeks
  weekly_avg_time = 34 := sorry

end mike_weekly_avg_time_l158_158852


namespace find_boxes_l158_158926

variable (John Jules Joseph Stan : ℕ)

-- Conditions
axiom h1 : John = 30
axiom h2 : John = 6 * Jules / 5 -- Equivalent to John having 20% more boxes than Jules
axiom h3 : Jules = Joseph + 5
axiom h4 : Joseph = Stan / 5 -- Equivalent to Joseph having 80% fewer boxes than Stan

-- Theorem to prove
theorem find_boxes (h1 : John = 30) (h2 : John = 6 * Jules / 5) (h3 : Jules = Joseph + 5) (h4 : Joseph = Stan / 5) : Stan = 100 :=
sorry

end find_boxes_l158_158926


namespace matrix_equation_l158_158595

open Matrix

-- Define matrix B
def B : Matrix (Fin 2) (Fin 2) (ℤ) :=
  ![![1, -2], 
    ![-3, 5]]

-- The proof problem statement in Lean 4
theorem matrix_equation (r s : ℤ) (I : Matrix (Fin 2) (Fin 2) (ℤ))  [DecidableEq (ℤ)] [Fintype (Fin 2)] : 
  I = 1 ∧ B ^ 6 = r • B + s • I ↔ r = 2999 ∧ s = 2520 := by {
    sorry
}

end matrix_equation_l158_158595


namespace total_time_for_journey_l158_158295

theorem total_time_for_journey (x : ℝ) : 
  let time_first_part := x / 50
  let time_second_part := 3 * x / 80
  time_first_part + time_second_part = 23 * x / 400 :=
by 
  sorry

end total_time_for_journey_l158_158295


namespace num_common_points_l158_158133

noncomputable def curve (x : ℝ) : ℝ := 3 * x ^ 4 - 2 * x ^ 3 - 9 * x ^ 2 + 4

noncomputable def tangent_line (x : ℝ) : ℝ :=
  -12 * (x - 1) - 4

theorem num_common_points :
  ∃ (x1 x2 x3 : ℝ), curve x1 = tangent_line x1 ∧
                    curve x2 = tangent_line x2 ∧
                    curve x3 = tangent_line x3 ∧
                    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
sorry

end num_common_points_l158_158133


namespace average_pages_per_book_deshaun_l158_158109

-- Definitions related to the conditions
def summer_days : ℕ := 80
def deshaun_books : ℕ := 60
def person_closest_percentage : ℚ := 0.75
def second_person_daily_pages : ℕ := 180

-- Derived definitions
def second_person_total_pages : ℕ := second_person_daily_pages * summer_days
def deshaun_total_pages : ℚ := second_person_total_pages / person_closest_percentage

-- The final proof statement
theorem average_pages_per_book_deshaun : 
  deshaun_total_pages / deshaun_books = 320 := 
by
  -- We would provide the proof here
  sorry

end average_pages_per_book_deshaun_l158_158109


namespace Jack_heavier_than_Sam_l158_158719

def total_weight := 96 -- total weight of Jack and Sam in pounds
def jack_weight := 52 -- Jack's weight in pounds

def sam_weight := total_weight - jack_weight

theorem Jack_heavier_than_Sam : jack_weight - sam_weight = 8 := by
  -- Here we would provide a proof, but we leave it as sorry for now.
  sorry

end Jack_heavier_than_Sam_l158_158719


namespace white_square_area_l158_158098

theorem white_square_area
  (edge_length : ℝ)
  (total_green_area : ℝ)
  (faces : ℕ)
  (green_per_face : ℝ)
  (total_surface_area : ℝ)
  (white_area_per_face : ℝ) :
  edge_length = 12 ∧ total_green_area = 432 ∧ faces = 6 ∧ total_surface_area = 864 ∧ green_per_face = total_green_area / faces ∧ white_area_per_face = total_surface_area / faces - green_per_face → white_area_per_face = 72 :=
by
  sorry

end white_square_area_l158_158098


namespace problem1_problem2_problem3_problem4_l158_158361

def R : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

theorem problem1 : A ∩ B = {x | 3 ≤ x ∧ x < 5} := sorry

theorem problem2 : A ∪ B = {x | 1 < x ∧ x ≤ 6} := sorry

theorem problem3 : (Set.compl A) ∩ B = {x | 5 ≤ x ∧ x ≤ 6} :=
sorry

theorem problem4 : Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 5} := sorry

end problem1_problem2_problem3_problem4_l158_158361


namespace area_of_cos_integral_l158_158609

theorem area_of_cos_integral : 
  (∫ x in (0:ℝ)..(3 * Real.pi / 2), |Real.cos x|) = 3 :=
by
  sorry

end area_of_cos_integral_l158_158609


namespace monthly_salary_l158_158754

variables (S : ℝ) (savings : ℝ) (new_expenses : ℝ)

theorem monthly_salary (h1 : savings = 0.20 * S)
                      (h2 : new_expenses = 0.96 * S)
                      (h3 : S = 200 + new_expenses) :
                      S = 5000 :=
by
  sorry

end monthly_salary_l158_158754


namespace volume_triangular_pyramid_correctness_l158_158155

noncomputable def volume_of_regular_triangular_pyramid 
  (a α l : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α

theorem volume_triangular_pyramid_correctness (a α l : ℝ) : volume_of_regular_triangular_pyramid a α l =
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α := 
sorry

end volume_triangular_pyramid_correctness_l158_158155


namespace least_number_of_cars_per_work_day_l158_158018

-- Define the conditions as constants in Lean
def paul_work_hours_per_day := 8
def jack_work_hours_per_day := 8
def paul_cars_per_hour := 2
def jack_cars_per_hour := 3

-- Define the total number of cars Paul and Jack can change in a workday
def total_cars_per_day := (paul_cars_per_hour + jack_cars_per_hour) * paul_work_hours_per_day

-- State the theorem to be proved
theorem least_number_of_cars_per_work_day : total_cars_per_day = 40 := by
  -- Proof goes here
  sorry

end least_number_of_cars_per_work_day_l158_158018


namespace find_fifth_month_sale_l158_158279

theorem find_fifth_month_sale (s1 s2 s3 s4 s6 A : ℝ) (h1 : s1 = 800) (h2 : s2 = 900) (h3 : s3 = 1000) (h4 : s4 = 700) (h5 : s6 = 900) (h6 : A = 850) :
  ∃ s5 : ℝ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = A ∧ s5 = 800 :=
by
  sorry

end find_fifth_month_sale_l158_158279


namespace max_cables_used_eq_375_l158_158665

-- Conditions for the problem
def total_employees : Nat := 40
def brand_A_computers : Nat := 25
def brand_B_computers : Nat := 15

-- The main theorem we want to prove
theorem max_cables_used_eq_375 
  (h_employees : total_employees = 40)
  (h_brand_A_computers : brand_A_computers = 25)
  (h_brand_B_computers : brand_B_computers = 15)
  (cables_connectivity : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), Prop)
  (no_initial_connections : ∀ (a : Fin brand_A_computers) (b : Fin brand_B_computers), ¬ cables_connectivity a b)
  (each_brand_B_connected : ∀ (b : Fin brand_B_computers), ∃ (a : Fin brand_A_computers), cables_connectivity a b)
  : ∃ (n : Nat), n = 375 := 
sorry

end max_cables_used_eq_375_l158_158665


namespace total_score_is_correct_l158_158369

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end total_score_is_correct_l158_158369


namespace part1_part2_l158_158928

noncomputable def f (x a : ℝ) := |x - a|

theorem part1 (a m : ℝ) :
  (∀ x, f x a ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
by
  sorry

theorem part2 (t x : ℝ) (h_t : 0 ≤ t ∧ t < 2) :
  f x 2 + t ≥ f (x + 2) 2 ↔ x ≤ (t + 2) / 2 :=
by
  sorry

end part1_part2_l158_158928


namespace least_four_digit_divisible_by_15_25_40_75_is_1200_l158_158998

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

def divisible_by_40 (n : ℕ) : Prop :=
  n % 40 = 0

def divisible_by_75 (n : ℕ) : Prop :=
  n % 75 = 0

theorem least_four_digit_divisible_by_15_25_40_75_is_1200 :
  ∃ n : ℕ, is_four_digit n ∧ divisible_by_15 n ∧ divisible_by_25 n ∧ divisible_by_40 n ∧ divisible_by_75 n ∧
  (∀ m : ℕ, is_four_digit m ∧ divisible_by_15 m ∧ divisible_by_25 m ∧ divisible_by_40 m ∧ divisible_by_75 m → n ≤ m) ∧
  n = 1200 := 
sorry

end least_four_digit_divisible_by_15_25_40_75_is_1200_l158_158998


namespace striped_shorts_difference_l158_158540

variable (students : ℕ)
variable (striped_shirts checkered_shirts shorts : ℕ)

-- Conditions
variable (Hstudents : students = 81)
variable (Hstriped : striped_shirts = 2 * checkered_shirts)
variable (Hcheckered : checkered_shirts = students / 3)
variable (Hshorts : shorts = checkered_shirts + 19)

-- Goal
theorem striped_shorts_difference :
  striped_shirts - shorts = 8 :=
sorry

end striped_shorts_difference_l158_158540


namespace virus_infection_l158_158397

theorem virus_infection (x : ℕ) (h : 1 + x + x^2 = 121) : x = 10 := 
sorry

end virus_infection_l158_158397


namespace simplify_expression_l158_158566

theorem simplify_expression :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 :=
by
  sorry

end simplify_expression_l158_158566


namespace TotalLaddersClimbedInCentimeters_l158_158399

def keaton_ladder_height := 50  -- height of Keaton's ladder in meters
def keaton_climbs := 30  -- number of times Keaton climbs the ladder

def reece_ladder_height := keaton_ladder_height - 6  -- height of Reece's ladder in meters
def reece_climbs := 25  -- number of times Reece climbs the ladder

def total_meters_climbed := (keaton_ladder_height * keaton_climbs) + (reece_ladder_height * reece_climbs)

def total_cm_climbed := total_meters_climbed * 100

theorem TotalLaddersClimbedInCentimeters :
  total_cm_climbed = 260000 :=
by
  sorry

end TotalLaddersClimbedInCentimeters_l158_158399


namespace max_books_l158_158191

theorem max_books (price_per_book available_money : ℕ) (h1 : price_per_book = 15) (h2 : available_money = 200) :
  ∃ n : ℕ, n = 13 ∧ n ≤ available_money / price_per_book :=
by {
  sorry
}

end max_books_l158_158191


namespace earn_2800_probability_l158_158099

def total_outcomes : ℕ := 7 ^ 4

def favorable_outcomes : ℕ :=
  (1 * 3 * 2 * 1) * 4 -- For each combination: \$1000, \$600, \$600, \$600; \$1000, \$1000, \$400, \$400; \$800, \$800, \$600, \$600; \$800, \$800, \$800, \$400

noncomputable def probability_of_earning_2800 : ℚ := favorable_outcomes / total_outcomes

theorem earn_2800_probability : probability_of_earning_2800 = 96 / 2401 := by
  sorry

end earn_2800_probability_l158_158099


namespace part1_part2_l158_158826

variables (a b : ℝ)

theorem part1 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) : ab ≥ 16 :=
sorry

theorem part2 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) :
  ∃ (a b : ℝ), a = 7 ∧ b = 5 / 2 ∧ a + 4 * b = 17 :=
sorry

end part1_part2_l158_158826


namespace rooks_same_distance_l158_158956

theorem rooks_same_distance (rooks : Fin 8 → (ℕ × ℕ)) 
    (h_non_attacking : ∀ i j, i ≠ j → Prod.fst (rooks i) ≠ Prod.fst (rooks j) ∧ Prod.snd (rooks i) ≠ Prod.snd (rooks j)) 
    : ∃ i j k l, i ≠ j ∧ k ≠ l ∧ (Prod.fst (rooks i) - Prod.fst (rooks k))^2 + (Prod.snd (rooks i) - Prod.snd (rooks k))^2 = (Prod.fst (rooks j) - Prod.fst (rooks l))^2 + (Prod.snd (rooks j) - Prod.snd (rooks l))^2 :=
by 
  -- Proof goes here
  sorry

end rooks_same_distance_l158_158956


namespace multiplication_factor_l158_158106

theorem multiplication_factor 
  (avg1 : ℕ → ℕ → ℕ)
  (avg2 : ℕ → ℕ → ℕ)
  (sum1 : ℕ)
  (num1 : ℕ)
  (num2 : ℕ)
  (sum2 : ℕ)
  (factor : ℚ) :
  avg1 sum1 num1 = 7 →
  avg2 sum2 num2 = 84 →
  sum1 = 10 * 7 →
  sum2 = 10 * 84 →
  factor = sum2 / sum1 →
  factor = 12 :=
by
  sorry

end multiplication_factor_l158_158106


namespace matrix_eq_sum_35_l158_158619

theorem matrix_eq_sum_35 (a b c d : ℤ) (h1 : 2 * a = 14 * a - 15 * b)
  (h2 : 2 * b = 9 * a - 10 * b)
  (h3 : 3 * c = 14 * c - 15 * d)
  (h4 : 3 * d = 9 * c - 10 * d) :
  a + b + c + d = 35 :=
sorry

end matrix_eq_sum_35_l158_158619


namespace domain_of_function_l158_158358

noncomputable def domain_f (x : ℝ) : Prop :=
  -x^2 + 2 * x + 3 > 0 ∧ 1 - x > 0 ∧ x ≠ 0

theorem domain_of_function :
  {x : ℝ | domain_f x} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_function_l158_158358


namespace one_fifth_of_ten_x_plus_three_l158_158054

theorem one_fifth_of_ten_x_plus_three (x : ℝ) : 
  (1 / 5) * (10 * x + 3) = 2 * x + 3 / 5 := 
  sorry

end one_fifth_of_ten_x_plus_three_l158_158054


namespace tetrahedron_sphere_relations_l158_158379

theorem tetrahedron_sphere_relations 
  (ρ ρ1 ρ2 ρ3 ρ4 m1 m2 m3 m4 : ℝ)
  (hρ_pos : ρ > 0)
  (hρ1_pos : ρ1 > 0)
  (hρ2_pos : ρ2 > 0)
  (hρ3_pos : ρ3 > 0)
  (hρ4_pos : ρ4 > 0)
  (hm1_pos : m1 > 0)
  (hm2_pos : m2 > 0)
  (hm3_pos : m3 > 0)
  (hm4_pos : m4 > 0) : 
  (2 / ρ = 1 / ρ1 + 1 / ρ2 + 1 / ρ3 + 1 / ρ4) ∧
  (1 / ρ = 1 / m1 + 1 / m2 + 1 / m3 + 1 / m4) ∧
  ( 1 / ρ1 = -1 / m1 + 1 / m2 + 1 / m3 + 1 / m4 ) := sorry

end tetrahedron_sphere_relations_l158_158379


namespace bn_is_arithmetic_seq_an_general_term_l158_158601

def seq_an (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ ∀ n, (a (n + 1) - 1) * (a n - 1) = 3 * (a n - a (n + 1))

def seq_bn (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / (a n - 1)

theorem bn_is_arithmetic_seq (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, b (n + 1) - b n = 1 / 3 :=
sorry

theorem an_general_term (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : seq_an a) (h2 : seq_bn a b) : 
∀ n, a n = (n + 5) / (n + 2) :=
sorry

end bn_is_arithmetic_seq_an_general_term_l158_158601


namespace combination_5_3_eq_10_l158_158930

-- Define the combination function according to its formula
noncomputable def combination (n k : ℕ) : ℕ :=
  (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem stating the required result
theorem combination_5_3_eq_10 : combination 5 3 = 10 := by
  sorry

end combination_5_3_eq_10_l158_158930


namespace trig_eq_solution_l158_158143

open Real

theorem trig_eq_solution (x : ℝ) :
    (∃ k : ℤ, x = -arccos ((sqrt 13 - 1) / 4) + 2 * k * π) ∨ 
    (∃ k : ℤ, x = -arccos ((1 - sqrt 13) / 4) + 2 * k * π) ↔ 
    (cos 5 * x - cos 7 * x) / (sin 4 * x + sin 2 * x) = 2 * abs (sin 2 * x) := by
  sorry

end trig_eq_solution_l158_158143


namespace sum_of_reciprocals_of_roots_eq_17_div_8_l158_158129

theorem sum_of_reciprocals_of_roots_eq_17_div_8 :
  ∀ p q : ℝ, (p + q = 17) → (p * q = 8) → (1 / p + 1 / q = 17 / 8) :=
by
  intros p q h1 h2
  sorry

end sum_of_reciprocals_of_roots_eq_17_div_8_l158_158129


namespace tan_alpha_minus_pi_over_4_l158_158829

theorem tan_alpha_minus_pi_over_4 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (β + π/4) = 3) 
  : Real.tan (α - π/4) = -1 / 7 :=
by
  sorry

end tan_alpha_minus_pi_over_4_l158_158829


namespace common_difference_is_3_l158_158955

theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ) (h1 : a 2 = 4) (h2 : 1 + a 3 = 5 + d)
  (h3 : a 6 = 4 + 4 * d) (h4 : 4 + a 10 = 8 + 8 * d) :
  (5 + d) * (8 + 8 * d) = (4 + 4 * d) ^ 2 → d = 3 := 
by
  intros hg
  sorry

end common_difference_is_3_l158_158955


namespace base3_last_two_digits_l158_158750

open Nat

theorem base3_last_two_digits (a b c : ℕ) (h1 : a = 2005) (h2 : b = 2003) (h3 : c = 2004) :
  (2005 ^ (2003 ^ 2004 + 3) % 81) = 11 :=
by
  sorry

end base3_last_two_digits_l158_158750


namespace greatest_multiple_of_3_lt_1000_l158_158739

theorem greatest_multiple_of_3_lt_1000 :
  ∃ (x : ℕ), (x % 3 = 0) ∧ (x > 0) ∧ (x^3 < 1000) ∧ ∀ (y : ℕ), (y % 3 = 0) ∧ (y > 0) ∧ (y^3 < 1000) → y ≤ x := 
sorry

end greatest_multiple_of_3_lt_1000_l158_158739


namespace time_for_A_and_C_l158_158208

variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := A + B = 1 / 8
def condition2 : Prop := B + C = 1 / 12
def condition3 : Prop := A + B + C = 1 / 6

theorem time_for_A_and_C (h1 : condition1 A B)
                        (h2 : condition2 B C)
                        (h3 : condition3 A B C) :
  1 / (A + C) = 8 :=
sorry

end time_for_A_and_C_l158_158208


namespace hypotenuse_of_45_45_90_triangle_l158_158348

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end hypotenuse_of_45_45_90_triangle_l158_158348


namespace ratio_of_areas_of_squares_l158_158993

theorem ratio_of_areas_of_squares (side_C side_D : ℕ) 
  (hC : side_C = 48) (hD : side_D = 60) : 
  (side_C^2 : ℚ)/(side_D^2 : ℚ) = 16/25 :=
by
  -- sorry, proof omitted
  sorry

end ratio_of_areas_of_squares_l158_158993


namespace pencils_count_l158_158023

theorem pencils_count (P L : ℕ) (h₁ : 6 * P = 5 * L) (h₂ : L = P + 4) : L = 24 :=
by sorry

end pencils_count_l158_158023


namespace wickets_before_last_match_l158_158880

theorem wickets_before_last_match (R W : ℕ) 
  (initial_average : ℝ) (runs_last_match wickets_last_match : ℕ) (average_decrease : ℝ)
  (h_initial_avg : initial_average = 12.4)
  (h_last_match_runs : runs_last_match = 26)
  (h_last_match_wickets : wickets_last_match = 5)
  (h_avg_decrease : average_decrease = 0.4)
  (h_initial_runs_eq : R = initial_average * W)
  (h_new_average : (R + runs_last_match) / (W + wickets_last_match) = initial_average - average_decrease) :
  W = 85 :=
by
  sorry

end wickets_before_last_match_l158_158880


namespace monotonicity_and_range_l158_158078

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 3 * Real.log x + 1

theorem monotonicity_and_range (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < 1/a → f a x < f a (1/a)) ∧ 
  (∀ x : ℝ, x > 1/a → f a x > f a (1/a)) ∧ 
  (∀ x : ℝ, f a x ≠ 0 → a > 1/Real.exp 1) :=
by
  sorry

end monotonicity_and_range_l158_158078


namespace eval_expression_l158_158120

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem eval_expression : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end eval_expression_l158_158120


namespace four_distinct_real_roots_l158_158103

theorem four_distinct_real_roots (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 4 * |x| + 5 - m) ∧ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) ↔ (1 < m ∧ m < 5) :=
by
  sorry

end four_distinct_real_roots_l158_158103


namespace balloons_left_l158_158554

def total_balloons (r w g c: Nat) : Nat := r + w + g + c

def num_friends : Nat := 10

theorem balloons_left (r w g c : Nat) (total := total_balloons r w g c) (h_r : r = 24) (h_w : w = 38) (h_g : g = 68) (h_c : c = 75) :
  total % num_friends = 5 := by
  sorry

end balloons_left_l158_158554


namespace dozen_Pokemon_cards_per_friend_l158_158766

theorem dozen_Pokemon_cards_per_friend
  (total_cards : ℕ) (num_friends : ℕ) (cards_per_dozen : ℕ)
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : cards_per_dozen = 12) :
  (total_cards / num_friends) / cards_per_dozen = 9 := 
sorry

end dozen_Pokemon_cards_per_friend_l158_158766


namespace bamboo_pole_is_10_l158_158724

noncomputable def bamboo_pole_length (x : ℕ) : Prop :=
  (x - 4)^2 + (x - 2)^2 = x^2

theorem bamboo_pole_is_10 : bamboo_pole_length 10 :=
by
  -- The proof is not provided
  sorry

end bamboo_pole_is_10_l158_158724


namespace cannot_achieve_141_cents_l158_158663
-- Importing the required library

-- Definitions corresponding to types of coins and their values
def penny := 1
def nickel := 5
def dime := 10
def half_dollar := 50

-- The main statement to prove
theorem cannot_achieve_141_cents :
  ¬∃ (x y z : ℕ), x + y + z = 3 ∧ 
    x * penny + y * nickel + z * dime + (3 - x - y - z) * half_dollar = 141 := 
by
  -- Currently leaving the proof as a sorry
  sorry

end cannot_achieve_141_cents_l158_158663


namespace B_N_Q_collinear_l158_158528

/-- Define point positions -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

/-- Define the curve C -/
def on_curve_C (P : Point) : Prop :=
  P.x^2 + P.y^2 - 6 * P.x + 1 = 0

/-- Define reflection of point A across the x-axis -/
def reflection_across_x (A : Point) : Point :=
  ⟨A.x, -A.y⟩

/-- Define the condition that line l passes through M and intersects curve C at two distinct points A and B -/
def line_l_condition (A B: Point) (k : ℝ) (hk : k ≠ 0) : Prop :=
  A.y = k * (A.x + 1) ∧ B.y = k * (B.x + 1) ∧ on_curve_C A ∧ on_curve_C B

/-- Main theorem to prove collinearity of B, N, Q -/
theorem B_N_Q_collinear (A B : Point) (k : ℝ) (hk : k ≠ 0)
  (hA : on_curve_C A) (hB : on_curve_C B)
  (h_l : line_l_condition A B k hk) :
  let Q := reflection_across_x A
  (B.x - N.x) * (Q.y - N.y) = (B.y - N.y) * (Q.x - N.x) :=
sorry

end B_N_Q_collinear_l158_158528


namespace cost_of_fencing_l158_158388

/-- Define given conditions: -/
def sides_ratio (length width : ℕ) : Prop := length = 3 * width / 2

def park_area : ℕ := 3750

def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

/-- Prove that the cost of fencing the park is 150 rupees: -/
theorem cost_of_fencing 
  (length width : ℕ) 
  (h : sides_ratio length width) 
  (h_area : length * width = park_area) 
  (cost_per_meter_paise : ℕ := 60) : 
  (length + width) * 2 * (paise_to_rupees cost_per_meter_paise) = 150 :=
by sorry

end cost_of_fencing_l158_158388


namespace train_speed_conversion_l158_158616

/-- Define a function to convert kmph to m/s --/
def kmph_to_ms (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

/-- Theorem stating that 72 kmph is equivalent to 20 m/s --/
theorem train_speed_conversion : kmph_to_ms 72 = 20 :=
by
  sorry

end train_speed_conversion_l158_158616


namespace equivalent_integer_l158_158419

theorem equivalent_integer (a b n : ℤ) (h1 : a ≡ 33 [ZMOD 60]) (h2 : b ≡ 85 [ZMOD 60]) (hn : 200 ≤ n ∧ n ≤ 251) : 
  a - b ≡ 248 [ZMOD 60] :=
sorry

end equivalent_integer_l158_158419


namespace derivative_of_f_at_pi_over_2_l158_158861

noncomputable def f (x : Real) := 5 * Real.sin x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = 0 :=
by
  -- The proof is omitted
  sorry

end derivative_of_f_at_pi_over_2_l158_158861


namespace monotonicity_F_range_k_l158_158395

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - Real.log (1 - x)
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x + a * x
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := f x - k * (x^3 - 3 * x)

theorem monotonicity_F (a : ℝ) (ha : a ≠ 0) :
(∀ x : ℝ, (-1 < x ∧ x < 1) → 
    (if (-2 ≤ a ∧ a < 0) ∨ (a > 0) then 0 ≤ (a - a * x^2 + 2) / (1 - x^2)
     else if a < -2 then 
        ((-1 < x ∧ x < -Real.sqrt ((a + 2) / a)) ∨ (Real.sqrt ((a + 2) / a) < x ∧ x < 1)) → 0 ≤ (a - a * x^2 + 2) / (1 - x^2) ∧ 
        (-Real.sqrt ((a + 2) / a) < x ∧ x < Real.sqrt ((a + 2) / a)) → 0 > (a - a * x^2 + 2) / (1 - x^2)
    else false)) :=
sorry

theorem range_k (k : ℝ) (hk : ∀ x : ℝ, (0 < x ∧ x < 1) → f x > k * (x^3 - 3 * x)) :
k ≥ -2 / 3 :=
sorry

end monotonicity_F_range_k_l158_158395


namespace arithmetic_sequence_third_term_l158_158805

theorem arithmetic_sequence_third_term 
    (a d : ℝ) 
    (h1 : a = 2)
    (h2 : (a + d) + (a + 3 * d) = 10) : 
    a + 2 * d = 5 := 
by
  sorry

end arithmetic_sequence_third_term_l158_158805


namespace find_a_l158_158607

theorem find_a (x a : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  A = (7, 1) ∧ B = (1, 4) ∧ C = (x, a * x) ∧ 
  (x - 7, a * x - 1) = (2 * (1 - x), 2 * (4 - a * x)) → 
  a = 1 :=
sorry

end find_a_l158_158607


namespace factorize_x_cube_minus_4x_l158_158824

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l158_158824


namespace koala_fiber_consumption_l158_158618

theorem koala_fiber_consumption
  (absorbed_fiber : ℝ) (total_fiber : ℝ) 
  (h1 : absorbed_fiber = 0.40 * total_fiber)
  (h2 : absorbed_fiber = 12) :
  total_fiber = 30 := 
by
  sorry

end koala_fiber_consumption_l158_158618


namespace find_q_l158_158153

noncomputable def q_value (m q : ℕ) : Prop := 
  ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (q * 10 ^ 31)

theorem find_q (m : ℕ) (q : ℕ) (h1 : m = 31) (h2 : q_value m q) : q = 2 :=
by
  sorry

end find_q_l158_158153


namespace number_of_tacos_you_ordered_l158_158699

variable {E : ℝ} -- E represents the cost of one enchilada in dollars

-- Conditions
axiom h1 : ∃ t : ℕ, 0.9 * (t : ℝ) + 3 * E = 7.80
axiom h2 : 0.9 * 3 + 5 * E = 12.70

theorem number_of_tacos_you_ordered (E : ℝ) : ∃ t : ℕ, t = 2 := by
  sorry

end number_of_tacos_you_ordered_l158_158699


namespace equation1_solution_equation2_no_solution_l158_158059

theorem equation1_solution (x: ℝ) (h: x ≠ -1/2 ∧ x ≠ 1):
  (1 / (x - 1) = 5 / (2 * x + 1)) ↔ (x = 2) :=
sorry

theorem equation2_no_solution (x: ℝ) (h: x ≠ 1 ∧ x ≠ -1):
  ¬ ( (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 ) :=
sorry

end equation1_solution_equation2_no_solution_l158_158059


namespace find_a_l158_158069

noncomputable def calculation (a : ℝ) (x : ℝ) (y : ℝ) (b : ℝ) (c : ℝ) : Prop :=
  (x * y) / (a * b * c) = 840

theorem find_a : calculation 50 0.0048 3.5 0.1 0.004 :=
by
  sorry

end find_a_l158_158069


namespace area_OBEC_is_19_5_l158_158008

-- Definitions for the points and lines from the conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨5, 0⟩
def B : Point := ⟨0, 15⟩
def C : Point := ⟨6, 0⟩
def E : Point := ⟨3, 6⟩

-- Function to calculate the area of a triangle given its vertices
def triangle_area (P1 P2 P3 : Point) : ℝ :=
  0.5 * |(P1.x * P2.y + P2.x * P3.y + P3.x * P1.y) - (P1.y * P2.x + P2.y * P3.x + P3.y * P1.x)|

-- Definitions of the vertices of the quadrilateral
def O : Point := ⟨0, 0⟩

-- Calculating the area of triangles OCE and OBE
def OCE_area : ℝ := triangle_area O C E
def OBE_area : ℝ := triangle_area O B E

-- Total area of quadrilateral OBEC
def OBEC_area : ℝ := OCE_area + OBE_area

-- Proof statement: The area of quadrilateral OBEC is 19.5
theorem area_OBEC_is_19_5 : OBEC_area = 19.5 := sorry

end area_OBEC_is_19_5_l158_158008


namespace find_number_l158_158608

theorem find_number (x : ℝ) (h : 3034 - x / 200.4 = 3029) : x = 1002 :=
sorry

end find_number_l158_158608


namespace ab_eq_neg_two_l158_158354

theorem ab_eq_neg_two (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a * b^a = -2 :=
by
  sorry

end ab_eq_neg_two_l158_158354


namespace larger_number_is_37_point_435_l158_158909

theorem larger_number_is_37_point_435 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 96) (h3 : x > y) : x = 37.435 :=
by
  sorry

end larger_number_is_37_point_435_l158_158909


namespace dan_total_purchase_cost_l158_158042

noncomputable def snake_toy_cost : ℝ := 11.76
noncomputable def cage_cost : ℝ := 14.54
noncomputable def heat_lamp_cost : ℝ := 6.25
noncomputable def cage_discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.08
noncomputable def found_dollar : ℝ := 1.00

noncomputable def total_cost : ℝ :=
  let cage_discount := cage_discount_rate * cage_cost
  let discounted_cage := cage_cost - cage_discount
  let subtotal_before_tax := snake_toy_cost + discounted_cage + heat_lamp_cost
  let sales_tax := sales_tax_rate * subtotal_before_tax
  let total_after_tax := subtotal_before_tax + sales_tax
  total_after_tax - found_dollar

theorem dan_total_purchase_cost : total_cost = 32.58 :=
  by 
    -- Placeholder for the proof
    sorry

end dan_total_purchase_cost_l158_158042


namespace PQRS_product_eq_one_l158_158643

noncomputable def P := Real.sqrt 2011 + Real.sqrt 2012
noncomputable def Q := -Real.sqrt 2011 - Real.sqrt 2012
noncomputable def R := Real.sqrt 2011 - Real.sqrt 2012
noncomputable def S := Real.sqrt 2012 - Real.sqrt 2011

theorem PQRS_product_eq_one : P * Q * R * S = 1 := by
  sorry

end PQRS_product_eq_one_l158_158643


namespace number_of_dogs_l158_158516

variable (D C : ℕ)
variable (x : ℚ)

-- Conditions
def ratio_dogs_to_cats := D = (x * (C: ℚ) / 7)
def new_ratio_dogs_to_cats := D = (15 / 11) * (C + 8)

theorem number_of_dogs (h1 : ratio_dogs_to_cats D C x) (h2 : new_ratio_dogs_to_cats D C) : D = 77 := 
by sorry

end number_of_dogs_l158_158516


namespace find_angle_B_l158_158721

noncomputable def angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) : ℝ :=
  B

theorem find_angle_B (a b c : ℝ) (A B C : ℝ) (h : b * Real.cos A - c * Real.cos B = (c - a) * Real.cos B) (h_sum : A + B + C = Real.pi) :
  B = Real.pi / 3 :=
sorry

end find_angle_B_l158_158721


namespace remainder_when_divided_by_6_eq_5_l158_158251

theorem remainder_when_divided_by_6_eq_5 (k : ℕ) (hk1 : k % 5 = 2) (hk2 : k < 41) (hk3 : k % 7 = 3) : k % 6 = 5 :=
sorry

end remainder_when_divided_by_6_eq_5_l158_158251


namespace find_z_l158_158086

theorem find_z (a z : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * 25 * 45 * z) : z = 49 :=
sorry

end find_z_l158_158086


namespace floor_e_eq_two_l158_158137

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l158_158137


namespace count_true_statements_l158_158800

theorem count_true_statements (a b c d : ℝ) : 
  (∃ (H1 : a ≠ b) (H2 : c ≠ d), a + c = b + d) →
  ((a ≠ b) ∧ (c ≠ d) → a + c ≠ b + d) = false ∧ 
  ((a + c ≠ b + d) → (a ≠ b) ∧ (c ≠ d)) = false ∧ 
  (∃ (H3 : a = b) (H4 : c = d), a + c ≠ b + d) = false ∧ 
  ((a + c = b + d) → (a = b) ∨ (c = d)) = false → 
  number_of_true_statements = 0 := 
by
  sorry

end count_true_statements_l158_158800


namespace closest_perfect_square_multiple_of_4_l158_158587

theorem closest_perfect_square_multiple_of_4 (n : ℕ) (h1 : ∃ k : ℕ, k^2 = n) (h2 : n % 4 = 0) : n = 324 := by
  -- Define 350 as the target
  let target := 350

  -- Conditions
  have cond1 : ∃ k : ℕ, k^2 = n := h1
  
  have cond2 : n % 4 = 0 := h2

  -- Check possible values meeting conditions
  by_cases h : n = 324
  { exact h }
  
  -- Exclude non-multiples of 4 and perfect squares further away from 350
  sorry

end closest_perfect_square_multiple_of_4_l158_158587


namespace average_weight_l158_158602

theorem average_weight :
  ∀ (A B C : ℝ),
    (A + B = 84) → 
    (B + C = 86) → 
    (B = 35) → 
    (A + B + C) / 3 = 45 :=
by
  intros A B C hab hbc hb
  -- proof omitted
  sorry

end average_weight_l158_158602


namespace evaluate_expression_l158_158141

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end evaluate_expression_l158_158141


namespace lateral_surface_area_truncated_cone_l158_158693

theorem lateral_surface_area_truncated_cone :
  let r := 1
  let R := 4
  let h := 4
  let l := Real.sqrt ((R - r)^2 + h^2)
  let S := Real.pi * (r + R) * l
  S = 25 * Real.pi :=
by
  sorry

end lateral_surface_area_truncated_cone_l158_158693


namespace total_amount_received_is_1465_l158_158931

-- defining the conditions
def principal_1 : ℝ := 4000
def principal_2 : ℝ := 8200
def rate_1 : ℝ := 0.11
def rate_2 : ℝ := rate_1 + 0.015

-- defining the interest from each account
def interest_1 := principal_1 * rate_1
def interest_2 := principal_2 * rate_2

-- stating the total amount received
def total_received := interest_1 + interest_2

-- proving the total amount received
theorem total_amount_received_is_1465 : total_received = 1465 := by
  -- proof goes here
  sorry

end total_amount_received_is_1465_l158_158931


namespace arithmetic_problem_l158_158005

theorem arithmetic_problem : 987 + 113 - 1000 = 100 :=
by
  sorry

end arithmetic_problem_l158_158005


namespace find_n_l158_158401

/-- Given: 
1. The second term in the expansion of (x + a)^n is binom n 1 * x^(n-1) * a = 210.
2. The third term in the expansion of (x + a)^n is binom n 2 * x^(n-2) * a^2 = 840.
3. The fourth term in the expansion of (x + a)^n is binom n 3 * x^(n-3) * a^3 = 2520.
We are to prove that n = 10. -/
theorem find_n (x a : ℕ) (n : ℕ)
  (h1 : Nat.choose n 1 * x^(n-1) * a = 210)
  (h2 : Nat.choose n 2 * x^(n-2) * a^2 = 840)
  (h3 : Nat.choose n 3 * x^(n-3) * a^3 = 2520) : 
  n = 10 := by sorry

end find_n_l158_158401


namespace diameter_of_tripled_volume_sphere_l158_158845

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l158_158845


namespace red_balls_count_l158_158203

theorem red_balls_count (w r : ℕ) (h1 : w = 16) (h2 : 4 * r = 3 * w) : r = 12 :=
by
  sorry

end red_balls_count_l158_158203


namespace parabola_intersection_min_y1_y2_sqr_l158_158989

theorem parabola_intersection_min_y1_y2_sqr :
  ∀ (x1 x2 y1 y2 : ℝ)
    (h1 : y1 ^ 2 = 4 * x1)
    (h2 : y2 ^ 2 = 4 * x2)
    (h3 : (∃ k : ℝ, x1 = 4 ∧ y1 = k * (4 - 4)) ∨ x1 = 4 ∧ y1 ≠ x2),
    ∃ m : ℝ, (y1^2 + y2^2) = m ∧ m = 32 := 
sorry

end parabola_intersection_min_y1_y2_sqr_l158_158989


namespace fixed_point_exists_l158_158262

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y : ℝ, (x = 2 ∧ y = -2 ∧ (ax - 5 = y)) :=
by
  sorry

end fixed_point_exists_l158_158262


namespace polygon_sides_eq_14_l158_158889

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem polygon_sides_eq_14 (n : ℕ) (h : n + num_diagonals n = 77) : n = 14 :=
by
  sorry

end polygon_sides_eq_14_l158_158889


namespace sravan_distance_l158_158363

theorem sravan_distance {D : ℝ} :
  (D / 90 + D / 60 = 15) ↔ (D = 540) :=
by sorry

end sravan_distance_l158_158363


namespace equation_line_through_intersections_l158_158747

theorem equation_line_through_intersections (A1 B1 A2 B2 : ℝ)
  (h1 : 2 * A1 + 3 * B1 = 1)
  (h2 : 2 * A2 + 3 * B2 = 1) :
  ∃ (a b c : ℝ), a = 2 ∧ b = 3 ∧ c = -1 ∧ (a * x + b * y + c = 0) := 
sorry

end equation_line_through_intersections_l158_158747


namespace total_cats_and_kittens_received_l158_158260

theorem total_cats_and_kittens_received 
  (adult_cats : ℕ) 
  (perc_female : ℕ) 
  (frac_litters : ℚ) 
  (kittens_per_litter : ℕ)
  (rescued_cats : ℕ) 
  (total_received : ℕ)
  (h1 : adult_cats = 120)
  (h2 : perc_female = 60)
  (h3 : frac_litters = 2/3)
  (h4 : kittens_per_litter = 3)
  (h5 : rescued_cats = 30)
  (h6 : total_received = 294) :
  adult_cats + rescued_cats + (frac_litters * (perc_female * adult_cats / 100) * kittens_per_litter) = total_received := 
sorry

end total_cats_and_kittens_received_l158_158260


namespace contractor_earnings_l158_158273

def total_days : ℕ := 30
def work_rate : ℝ := 25
def fine_rate : ℝ := 7.5
def absent_days : ℕ := 8
def worked_days : ℕ := total_days - absent_days
def total_earned : ℝ := worked_days * work_rate
def total_fine : ℝ := absent_days * fine_rate
def total_received : ℝ := total_earned - total_fine

theorem contractor_earnings : total_received = 490 :=
by
  sorry

end contractor_earnings_l158_158273


namespace unique_a_for_intersection_l158_158398

def A (a : ℝ) : Set ℝ := {-4, 2 * a - 1, a^2}
def B (a : ℝ) : Set ℝ := {a - 5, 1 - a, 9}

theorem unique_a_for_intersection (a : ℝ) :
  (9 ∈ A a ∩ B a ∧ ∀ x, x ∈ A a ∩ B a → x = 9) ↔ a = -3 := by
  sorry

end unique_a_for_intersection_l158_158398


namespace solve_linear_system_l158_158032

theorem solve_linear_system (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : 3 * x + 2 * y = 10) : x + y = 3 := 
by
  sorry

end solve_linear_system_l158_158032


namespace solve_system_of_equations_l158_158184

theorem solve_system_of_equations : 
  ∃ (x y : ℤ), 2 * x + 5 * y = 8 ∧ 3 * x - 5 * y = -13 ∧ x = -1 ∧ y = 2 :=
by
  sorry

end solve_system_of_equations_l158_158184


namespace sum_of_intercepts_l158_158479

theorem sum_of_intercepts (x₀ y₀ : ℕ) (hx₀ : 4 * x₀ ≡ 2 [MOD 25]) (hy₀ : 5 * y₀ ≡ 23 [MOD 25]) 
  (hx_cond : x₀ < 25) (hy_cond : y₀ < 25) : x₀ + y₀ = 28 :=
  sorry

end sum_of_intercepts_l158_158479


namespace range_of_a_l158_158417

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) (f_decreasing : ∀ x y : ℝ, x ≤ y → f x a ≥ f y a) : 
  1/2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l158_158417


namespace part1_solution_set_part2_range_of_a_l158_158726

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end part1_solution_set_part2_range_of_a_l158_158726


namespace weight_gain_difference_l158_158767

theorem weight_gain_difference :
  let orlando_gain := 5
  let jose_gain := 2 * orlando_gain + 2
  let total_gain := 20
  let fernando_gain := total_gain - (orlando_gain + jose_gain)
  let half_jose_gain := jose_gain / 2
  half_jose_gain - fernando_gain = 3 :=
by
  sorry

end weight_gain_difference_l158_158767


namespace probability_one_left_one_right_l158_158038

/-- Define the conditions: 12 left-handed gloves, 10 right-handed gloves. -/
def num_left_handed_gloves : ℕ := 12

def num_right_handed_gloves : ℕ := 10

/-- Total number of gloves is 22. -/
def total_gloves : ℕ := num_left_handed_gloves + num_right_handed_gloves

/-- Total number of ways to pick any two gloves from 22 gloves. -/
def total_pick_two_ways : ℕ := (total_gloves * (total_gloves - 1)) / 2

/-- Number of favorable outcomes picking one left-handed and one right-handed glove. -/
def favorable_outcomes : ℕ := num_left_handed_gloves * num_right_handed_gloves

/-- Define the probability as favorable outcomes divided by total outcomes. 
 It should yield 40/77. -/
theorem probability_one_left_one_right : 
  (favorable_outcomes : ℚ) / total_pick_two_ways = 40 / 77 :=
by
  -- Skip the proof.
  sorry

end probability_one_left_one_right_l158_158038


namespace jelly_price_l158_158686

theorem jelly_price (d1 h1 d2 h2 : ℝ) (P1 : ℝ)
    (hd1 : d1 = 2) (hh1 : h1 = 5) (hd2 : d2 = 4) (hh2 : h2 = 8) (P1_cond : P1 = 0.75) :
    ∃ P2 : ℝ, P2 = 2.40 :=
by
  sorry

end jelly_price_l158_158686


namespace entry_exit_options_l158_158591

theorem entry_exit_options :
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  (total_gates * total_gates = 49) :=
by {
  let south_gates := 4
  let north_gates := 3
  let total_gates := south_gates + north_gates
  show total_gates * total_gates = 49
  sorry
}

end entry_exit_options_l158_158591


namespace ratio_long_side_brush_width_l158_158890

theorem ratio_long_side_brush_width 
  (l : ℝ) (w : ℝ) (d : ℝ) (total_area : ℝ) (painted_area : ℝ) (b : ℝ) 
  (h1 : l = 9)
  (h2 : w = 4)
  (h3 : total_area = l * w)
  (h4 : total_area / 3 = painted_area)
  (h5 : d = Real.sqrt (l^2 + w^2))
  (h6 : d * b = painted_area) :
  l / b = (3 * Real.sqrt 97) / 4 :=
by
  sorry

end ratio_long_side_brush_width_l158_158890


namespace pages_written_in_a_year_l158_158026

def pages_per_friend_per_letter : ℕ := 3
def friends : ℕ := 2
def letters_per_week : ℕ := 2
def weeks_per_year : ℕ := 52

theorem pages_written_in_a_year : 
  (pages_per_friend_per_letter * friends * letters_per_week * weeks_per_year) = 624 :=
by
  sorry

end pages_written_in_a_year_l158_158026


namespace oldest_bride_age_l158_158047

theorem oldest_bride_age (B G : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) :
  B = 102 :=
by
  sorry

end oldest_bride_age_l158_158047


namespace tetrahedron_volume_from_cube_l158_158444

theorem tetrahedron_volume_from_cube {s : ℝ} (h : s = 8) :
  let cube_volume := s^3
  let smaller_tetrahedron_volume := (1/3) * (1/2) * s * s * s
  let total_smaller_tetrahedron_volume := 4 * smaller_tetrahedron_volume
  let tetrahedron_volume := cube_volume - total_smaller_tetrahedron_volume
  tetrahedron_volume = 170.6666 :=
by
  sorry

end tetrahedron_volume_from_cube_l158_158444


namespace fifth_friend_paid_l158_158859

theorem fifth_friend_paid (a b c d e : ℝ)
  (h1 : a = (1/3) * (b + c + d + e))
  (h2 : b = (1/4) * (a + c + d + e))
  (h3 : c = (1/5) * (a + b + d + e))
  (h4 : a + b + c + d + e = 120) :
  e = 40 :=
sorry

end fifth_friend_paid_l158_158859


namespace ram_first_year_balance_l158_158684

-- Given conditions
def initial_deposit : ℝ := 1000
def interest_first_year : ℝ := 100

-- Calculate end of the first year balance
def balance_first_year := initial_deposit + interest_first_year

-- Prove that balance_first_year is $1100
theorem ram_first_year_balance :
  balance_first_year = 1100 :=
by 
  sorry

end ram_first_year_balance_l158_158684


namespace min_increase_air_quality_days_l158_158063

theorem min_increase_air_quality_days {days_in_year : ℕ} (last_year_ratio next_year_ratio : ℝ) (good_air_days : ℕ) :
  days_in_year = 365 → last_year_ratio = 0.6 → next_year_ratio > 0.7 →
  (good_air_days / days_in_year < last_year_ratio → ∀ n: ℕ, good_air_days + n ≥ 37) :=
by
  intros hdays_in_year hlast_year_ratio hnext_year_ratio h_good_air_days
  sorry

end min_increase_air_quality_days_l158_158063


namespace angle_A_in_triangle_l158_158692

noncomputable def is_angle_A (a b : ℝ) (B A: ℝ) : Prop :=
  a = 2 * Real.sqrt 3 ∧ b = 2 * Real.sqrt 2 ∧ B = Real.pi / 4 ∧
  (A = Real.pi / 3 ∨ A = 2 * Real.pi / 3)

theorem angle_A_in_triangle (a b A B : ℝ) (h : is_angle_A a b B A) : A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end angle_A_in_triangle_l158_158692


namespace matrix_exponentiation_l158_158229

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2],
    ![2, -1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-4, 6],
    ![-6, 5]]

theorem matrix_exponentiation :
  A^4 = B :=
by
  sorry

end matrix_exponentiation_l158_158229


namespace can_form_triangle_l158_158020

theorem can_form_triangle : Prop :=
  ∃ (a b c : ℝ), 
    (a = 8 ∧ b = 6 ∧ c = 4) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a)

#check can_form_triangle

end can_form_triangle_l158_158020


namespace minimum_value_ineq_l158_158646

theorem minimum_value_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
    (1 / (a + 2 * b)) + (1 / (b + 2 * c)) + (1 / (c + 2 * a)) ≥ 3 := 
by
  sorry

end minimum_value_ineq_l158_158646


namespace simplify_and_evaluate_l158_158304

-- Define the condition as a predicate
def condition (a b : ℝ) : Prop := (a + 1/2)^2 + |b - 2| = 0

-- The simplified expression
def simplified_expression (a b : ℝ) : ℝ := 12 * a^2 * b - 6 * a * b^2

-- Statement: Given the condition, prove that the simplified expression evaluates to 18
theorem simplify_and_evaluate : ∀ (a b : ℝ), condition a b → simplified_expression a b = 18 :=
by
  intros a b hc
  sorry  -- Proof omitted

end simplify_and_evaluate_l158_158304


namespace value_of_x0_l158_158439

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x
noncomputable def f_deriv (x : ℝ) : ℝ := ((x - 1) * Real.exp x) / (x * x)

theorem value_of_x0 (x0 : ℝ) (h : f_deriv x0 = -f x0) : x0 = 1 / 2 := by
  sorry

end value_of_x0_l158_158439


namespace find_f_of_7_6_l158_158105

-- Definitions from conditions
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x k : ℤ, f (x + T * (k : ℝ)) = f x

def f_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x = x

-- The periodic function f with period 4
def f : ℝ → ℝ := sorry

-- Hypothesis
axiom f_periodic : periodic_function f 4
axiom f_on_interval : f_in_interval f

-- Theorem to prove
theorem find_f_of_7_6 : f 7.6 = 3.6 :=
by
  sorry

end find_f_of_7_6_l158_158105


namespace find_coordinates_of_P_l158_158482

theorem find_coordinates_of_P (P : ℝ × ℝ) (hx : abs P.2 = 5) (hy : abs P.1 = 3) (hq : P.1 < 0 ∧ P.2 > 0) : 
  P = (-3, 5) := 
  sorry

end find_coordinates_of_P_l158_158482


namespace remaining_pie_portion_l158_158189

theorem remaining_pie_portion (Carlos_takes: ℝ) (fraction_Maria: ℝ) :
  Carlos_takes = 0.60 →
  fraction_Maria = 0.25 →
  (1 - Carlos_takes) * (1 - fraction_Maria) = 0.30 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end remaining_pie_portion_l158_158189


namespace cost_of_trip_per_student_l158_158375

def raised_fund : ℕ := 50
def contribution_per_student : ℕ := 5
def num_students : ℕ := 20
def remaining_fund : ℕ := 10

theorem cost_of_trip_per_student :
  ((raised_fund - remaining_fund) / num_students) = 2 := by
  sorry

end cost_of_trip_per_student_l158_158375


namespace minValue_l158_158735

theorem minValue (x y z : ℝ) (h : 1/x + 2/y + 3/z = 1) : x + y/2 + z/3 ≥ 9 :=
by
  sorry

end minValue_l158_158735


namespace smallest_distance_l158_158122

open Complex

variable (z w : ℂ)

def a : ℂ := -2 - 4 * I
def b : ℂ := 5 + 6 * I

-- Conditions
def cond1 : Prop := abs (z + 2 + 4 * I) = 2
def cond2 : Prop := abs (w - 5 - 6 * I) = 4

-- Problem
theorem smallest_distance (h1 : cond1 z) (h2 : cond2 w) : abs (z - w) = Real.sqrt 149 - 6 :=
sorry

end smallest_distance_l158_158122


namespace train_speed_on_time_l158_158746

theorem train_speed_on_time :
  ∃ (v : ℝ), 
  (∀ (d : ℝ) (t : ℝ),
    d = 133.33 ∧ 
    80 * (t + 1/3) = d ∧ 
    v * t = d) → 
  v = 100 :=
by
  sorry

end train_speed_on_time_l158_158746


namespace seq_properties_l158_158004

-- Conditions for the sequence a_n
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n * a n + 1

-- The statements to prove given the sequence definition
theorem seq_properties (a : ℕ → ℝ) (h : seq a) :
  (∀ n, a (n + 1) ≥ 2 * a n) ∧
  (∀ n, a (n + 1) / a n ≥ a n) ∧
  (∀ n, a n ≥ n * n - 2 * n + 2) :=
by
  sorry

end seq_properties_l158_158004


namespace cevians_concurrent_circumscribable_l158_158334

-- Define the problem
variables {A B C D X Y Z : Type}

-- Define concurrent cevians
def cevian_concurrent (A B C X Y Z D : Type) : Prop := true

-- Define circumscribable quadrilaterals
def circumscribable (A B C D : Type) : Prop := true

-- The theorem statement
theorem cevians_concurrent_circumscribable (h_conc: cevian_concurrent A B C X Y Z D) 
(h1: circumscribable D Y A Z) (h2: circumscribable D Z B X) : circumscribable D X C Y :=
sorry

end cevians_concurrent_circumscribable_l158_158334


namespace initial_total_balls_l158_158967

theorem initial_total_balls (B T : Nat) (h1 : B = 9) (h2 : ∀ (n : Nat), (T - 5) * 1/5 = 4) :
  T = 25 := sorry

end initial_total_balls_l158_158967


namespace graph_properties_l158_158092

theorem graph_properties (x : ℝ) :
  (∃ p : ℝ × ℝ, p = (1, -7) ∧ y = -7 * x) ∧
  (x ≠ 0 → y * x < 0) ∧
  (x > 0 → y < 0) :=
by
  sorry

end graph_properties_l158_158092


namespace nitin_borrowed_amount_l158_158359

theorem nitin_borrowed_amount (P : ℝ) (interest_paid : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) 
  (h_rates1 : rate1 = 0.06) (h_rates2 : rate2 = 0.09) 
  (h_rates3 : rate3 = 0.13) (h_time1 : time1 = 3) 
  (h_time2 : time2 = 5) (h_time3 : time3 = 3)
  (h_interest : interest_paid = 8160) :
  P * (rate1 * time1 + rate2 * time2 + rate3 * time3) = interest_paid → 
  P = 8000 := 
by 
  sorry

end nitin_borrowed_amount_l158_158359


namespace volunteer_comprehensive_score_is_92_l158_158812

noncomputable def written_score : ℝ := 90
noncomputable def trial_lecture_score : ℝ := 94
noncomputable def interview_score : ℝ := 90

noncomputable def written_weight : ℝ := 0.3
noncomputable def trial_lecture_weight : ℝ := 0.5
noncomputable def interview_weight : ℝ := 0.2

noncomputable def comprehensive_score : ℝ :=
  written_score * written_weight +
  trial_lecture_score * trial_lecture_weight +
  interview_score * interview_weight

theorem volunteer_comprehensive_score_is_92 :
  comprehensive_score = 92 := by
  sorry

end volunteer_comprehensive_score_is_92_l158_158812


namespace total_gain_percentage_combined_l158_158119

theorem total_gain_percentage_combined :
  let CP1 := 20
  let CP2 := 35
  let CP3 := 50
  let SP1 := 25
  let SP2 := 44
  let SP3 := 65
  let totalCP := CP1 + CP2 + CP3
  let totalSP := SP1 + SP2 + SP3
  let totalGain := totalSP - totalCP
  let gainPercentage := (totalGain / totalCP) * 100
  gainPercentage = 27.62 :=
by sorry

end total_gain_percentage_combined_l158_158119


namespace area_under_arccos_cos_l158_158446

noncomputable def func (x : ℝ) : ℝ := Real.arccos (Real.cos x)

theorem area_under_arccos_cos :
  ∫ x in (0:ℝ)..3 * Real.pi, func x = 3 * Real.pi ^ 2 / 2 :=
by
  sorry

end area_under_arccos_cos_l158_158446


namespace a_n_bound_l158_158681

theorem a_n_bound (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ m n : ℕ, 0 < m ∧ 0 < n → (m + n) * a (m + n) ≤ a m + a n) →
  1 / a 200 > 4 * 10^7 := 
sorry

end a_n_bound_l158_158681


namespace total_rattlesnakes_l158_158938

-- Definitions based on the problem's conditions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def other_snakes : ℕ := total_snakes - (pythons + boa_constrictors)

-- Statement to be proved
theorem total_rattlesnakes : other_snakes = 40 := 
by 
  -- Skipping the proof
  sorry

end total_rattlesnakes_l158_158938


namespace original_six_digit_number_is_105262_l158_158195

def is_valid_number (N : ℕ) : Prop :=
  ∃ A : ℕ, A < 100000 ∧ (N = 10 * A + 2) ∧ (200000 + A = 2 * N + 2)

theorem original_six_digit_number_is_105262 :
  ∃ N : ℕ, is_valid_number N ∧ N = 105262 :=
by
  sorry

end original_six_digit_number_is_105262_l158_158195


namespace base_500_in_base_has_six_digits_l158_158585

theorem base_500_in_base_has_six_digits (b : ℕ) : b^5 ≤ 500 ∧ 500 < b^6 ↔ b = 3 := 
by
  sorry

end base_500_in_base_has_six_digits_l158_158585


namespace graph_translation_l158_158342

variable (f : ℝ → ℝ)

theorem graph_translation (h : f 1 = 3) : f (-1) + 1 = 4 :=
sorry

end graph_translation_l158_158342


namespace markup_rate_l158_158590

theorem markup_rate (S : ℝ) (C : ℝ) (hS : S = 8) (h1 : 0.20 * S = 0.10 * S + (S - C)) :
  ((S - C) / C) * 100 = 42.857 :=
by
  -- Assume given conditions and reasoning to conclude the proof
  sorry

end markup_rate_l158_158590


namespace amount_of_CaO_required_l158_158164

theorem amount_of_CaO_required (n_H2O : ℝ) (n_CaOH2 : ℝ) (n_CaO : ℝ) 
  (h1 : n_H2O = 2) (h2 : n_CaOH2 = 2) :
  n_CaO = 2 :=
by
  sorry

end amount_of_CaO_required_l158_158164


namespace Kato_finishes_first_l158_158629

-- Define constants and variables from the problem conditions
def Kato_total_pages : ℕ := 10
def Kato_lines_per_page : ℕ := 20
def Gizi_lines_per_page : ℕ := 30
def conversion_ratio : ℚ := 3 / 4
def initial_pages_written_by_Kato : ℕ := 4
def initial_additional_lines_by_Kato : ℚ := 2.5
def Kato_to_Gizi_writing_ratio : ℚ := 3 / 4

-- Calculate total lines in Kato's manuscript
def Kato_total_lines : ℕ := Kato_total_pages * Kato_lines_per_page

-- Convert Kato's lines to Gizi's format
def Kato_lines_in_Gizi_format : ℚ := Kato_total_lines * conversion_ratio

-- Calculate total pages Gizi needs to type
def Gizi_total_pages : ℚ := Kato_lines_in_Gizi_format / Gizi_lines_per_page

-- Calculate initial lines by Kato before Gizi starts typing
def initial_lines_by_Kato : ℚ := initial_pages_written_by_Kato * Kato_lines_per_page + initial_additional_lines_by_Kato

-- Lines Kato writes for every page Gizi types including setup time consideration
def additional_lines_by_Kato_per_Gizi_page : ℚ := Gizi_lines_per_page * Kato_to_Gizi_writing_ratio + initial_additional_lines_by_Kato / Gizi_total_pages

-- Calculate total lines Kato writes while Gizi finishes 5 pages
def final_lines_by_Kato : ℚ := additional_lines_by_Kato_per_Gizi_page * Gizi_total_pages

-- Remaining lines after initial setup for Kato
def remaining_lines_by_Kato_after_initial : ℚ := Kato_total_lines - initial_lines_by_Kato

-- Final proof statement
theorem Kato_finishes_first : final_lines_by_Kato ≥ remaining_lines_by_Kato_after_initial :=
by sorry

end Kato_finishes_first_l158_158629


namespace calc_4_op_3_l158_158549

def specific_op (m n : ℕ) : ℕ := n^2 - m

theorem calc_4_op_3 :
  specific_op 4 3 = 5 :=
by
  sorry

end calc_4_op_3_l158_158549


namespace probability_red_then_white_l158_158100

-- Define the total number of balls and the probabilities
def total_balls : ℕ := 9
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probabilities
def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

-- Define the combined probability of drawing a red and then a white ball 
theorem probability_red_then_white : (prob_red * prob_white) = 2/27 :=
by
  sorry

end probability_red_then_white_l158_158100


namespace multiples_of_10_between_11_and_103_l158_158827

def countMultiplesOf10 (lower_bound upper_bound : Nat) : Nat :=
  Nat.div (upper_bound - lower_bound) 10 + 1

theorem multiples_of_10_between_11_and_103 : 
  countMultiplesOf10 11 103 = 9 :=
by
  sorry

end multiples_of_10_between_11_and_103_l158_158827


namespace min_sum_of_bases_l158_158810

theorem min_sum_of_bases (a b : ℕ) (h : 3 * a + 5 = 4 * b + 2) : a + b = 13 :=
sorry

end min_sum_of_bases_l158_158810


namespace mod_equiv_inverse_sum_l158_158583

theorem mod_equiv_inverse_sum :
  (3^15 + 3^14 + 3^13 + 3^12) % 17 = 5 :=
by sorry

end mod_equiv_inverse_sum_l158_158583


namespace glee_club_female_members_l158_158321

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l158_158321


namespace symmetric_difference_card_l158_158919

variable (x y : Finset ℤ)
variable (h1 : x.card = 16)
variable (h2 : y.card = 18)
variable (h3 : (x ∩ y).card = 6)

theorem symmetric_difference_card :
  (x \ y ∪ y \ x).card = 22 := by sorry

end symmetric_difference_card_l158_158919


namespace average_speed_of_train_l158_158061

theorem average_speed_of_train (distance time : ℝ) (h1 : distance = 80) (h2 : time = 8) :
  distance / time = 10 :=
by
  sorry

end average_speed_of_train_l158_158061


namespace henry_total_cost_l158_158738

def henry_initial_figures : ℕ := 3
def henry_total_needed_figures : ℕ := 15
def cost_per_figure : ℕ := 12

theorem henry_total_cost :
  (henry_total_needed_figures - henry_initial_figures) * cost_per_figure = 144 :=
by
  sorry

end henry_total_cost_l158_158738


namespace average_speed_for_trip_l158_158763

theorem average_speed_for_trip (t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (total_time : ℝ) 
  (h₁ : t₁ = 6) 
  (h₂ : v₁ = 30) 
  (h₃ : t₂ = 2) 
  (h₄ : v₂ = 46) 
  (h₅ : total_time = t₁ + t₂) 
  (h₆ : total_time = 8) :
  ((v₁ * t₁ + v₂ * t₂) / total_time) = 34 := 
  by 
    sorry

end average_speed_for_trip_l158_158763


namespace james_older_brother_is_16_l158_158235

variables (John James James_older_brother : ℕ)

-- Given conditions
def current_age_john : ℕ := 39
def three_years_ago_john (caj : ℕ) : ℕ := caj - 3
def twice_as_old_condition (ja : ℕ) (james_age_in_6_years : ℕ) : Prop :=
  ja = 2 * james_age_in_6_years
def james_age_in_6_years (jc : ℕ) : ℕ := jc + 6
def james_older_brother_age (jc : ℕ) : ℕ := jc + 4

-- Theorem to be proved
theorem james_older_brother_is_16
  (H1 : current_age_john = John)
  (H2 : three_years_ago_john current_age_john = 36)
  (H3 : twice_as_old_condition 36 (james_age_in_6_years James))
  (H4 : james_older_brother_age James = James_older_brother) :
  James_older_brother = 16 := sorry

end james_older_brother_is_16_l158_158235


namespace find_solutions_l158_158996

theorem find_solutions :
  {x : ℝ | 1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0} = {1, -9, 3, -3} :=
by
  sorry

end find_solutions_l158_158996


namespace draw_13_cards_no_straight_flush_l158_158906

theorem draw_13_cards_no_straight_flush :
  let deck_size := 52
  let suit_count := 4
  let rank_count := 13
  let non_straight_flush_draws (n : ℕ) := 3^n - 3
  n = rank_count →
  ∀ (draw : ℕ), draw = non_straight_flush_draws n :=
by
-- Proof would be here
sorry

end draw_13_cards_no_straight_flush_l158_158906


namespace find_b_l158_158539

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x ^ 3 + b * x - 3

theorem find_b (b : ℝ) (h : g b (g b 1) = 1) : b = 1 / 2 :=
by
  sorry

end find_b_l158_158539


namespace pipe_fill_rate_l158_158484

variable (R_A R_B : ℝ)

theorem pipe_fill_rate :
  R_A = 1 / 32 →
  R_A + R_B = 1 / 6.4 →
  R_B / R_A = 4 :=
by
  intros hRA hSum
  have hRA_pos : R_A ≠ 0 := by linarith
  sorry

end pipe_fill_rate_l158_158484


namespace cylinder_volume_l158_158696

theorem cylinder_volume (r h : ℝ) (hr : r = 1) (hh : h = 1) : (π * r^2 * h) = π :=
by
  sorry

end cylinder_volume_l158_158696


namespace find_other_integer_l158_158584

theorem find_other_integer (x y : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 7 * x + y = 68) : y = 5 :=
by
  sorry

end find_other_integer_l158_158584


namespace triangle_area_problem_l158_158624

theorem triangle_area_problem (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_area : (∃ t : ℝ, t > 0 ∧ (2 * c * t + 3 * d * (12 / (2 * c)) = 12) ∧ (∃ s : ℝ, s > 0 ∧ 2 * c * (12 / (3 * d)) + 3 * d * s = 12)) ∧ 
    ((1 / 2) * (12 / (2 * c)) * (12 / (3 * d)) = 12)) : c * d = 1 := 
by 
  sorry

end triangle_area_problem_l158_158624


namespace milk_water_ratio_l158_158613

theorem milk_water_ratio (total_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) (added_water : ℕ)
  (h₁ : total_volume = 45) (h₂ : initial_milk_ratio = 4) (h₃ : initial_water_ratio = 1) (h₄ : added_water = 9) :
  (36 : ℕ) / (18 : ℕ) = 2 :=
by sorry

end milk_water_ratio_l158_158613


namespace show_watching_days_l158_158518

def numberOfEpisodes := 20
def lengthOfEachEpisode := 30
def dailyWatchingTime := 2

theorem show_watching_days:
  (numberOfEpisodes * lengthOfEachEpisode) / 60 / dailyWatchingTime = 5 := 
by
  sorry

end show_watching_days_l158_158518


namespace Julio_current_age_l158_158240

theorem Julio_current_age (J : ℕ) (James_current_age : ℕ) (h1 : James_current_age = 11)
    (h2 : J + 14 = 2 * (James_current_age + 14)) : 
    J = 36 := 
by 
  sorry

end Julio_current_age_l158_158240


namespace complement_A_is_closed_interval_l158_158257

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A with the given condition
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U
def complement_A : Set ℝ := Set.compl A

theorem complement_A_is_closed_interval :
  complement_A = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry  -- Proof to be inserted

end complement_A_is_closed_interval_l158_158257


namespace chicken_nugget_ratio_l158_158326

theorem chicken_nugget_ratio (k d a t : ℕ) (h1 : a = 20) (h2 : t = 100) (h3 : k + d + a = t) : (k + d) / a = 4 :=
by
  sorry

end chicken_nugget_ratio_l158_158326


namespace new_train_distance_l158_158773

-- Given conditions
def distance_older_train : ℝ := 200
def percent_more : ℝ := 0.20

-- Conclusion to prove
theorem new_train_distance : (distance_older_train * (1 + percent_more)) = 240 := by
  -- Placeholder to indicate that we are skipping the actual proof steps
  sorry

end new_train_distance_l158_158773


namespace total_team_players_l158_158345

-- Conditions
def team_percent_boys : ℚ := 0.6
def team_percent_girls := 1 - team_percent_boys
def junior_girls_count : ℕ := 10
def total_girls := junior_girls_count * 2
def girl_percentage_as_decimal := team_percent_girls

-- Problem
theorem total_team_players : (total_girls : ℚ) / girl_percentage_as_decimal = 50 := 
by 
    sorry

end total_team_players_l158_158345


namespace min_S_min_S_values_range_of_c_l158_158628

-- Part 1
theorem min_S (a b c : ℝ) (h : a + b + c = 1) : 
  2 * a^2 + 3 * b^2 + c^2 ≥ (6 / 11) :=
sorry

-- Part 1, finding exact values of a, b, c where minimum is reached
theorem min_S_values (a b c : ℝ) (h : a + b + c = 1) :
  2 * a^2 + 3 * b^2 + c^2 = (6 / 11) ↔ a = (3 / 11) ∧ b = (2 / 11) ∧ c = (6 / 11) :=
sorry
  
-- Part 2
theorem range_of_c (a b c : ℝ) (h1 : 2 * a^2 + 3 * b^2 + c^2 = 1) : 
  (1 / 11) ≤ c ∧ c ≤ 1 :=
sorry

end min_S_min_S_values_range_of_c_l158_158628


namespace triangle_area_l158_158340

theorem triangle_area {a b : ℝ} (h : a ≠ 0) :
  (∃ x y : ℝ, 3 * x + a * y = 12) → b = 24 / a ↔ (∃ x y : ℝ, x = 4 ∧ y = 12 / a ∧ b = (1/2) * 4 * (12 / a)) :=
by
  sorry

end triangle_area_l158_158340


namespace baseball_card_value_decrease_l158_158341

theorem baseball_card_value_decrease (initial_value : ℝ) :
  (1 - 0.70 * 0.90) * 100 = 37 := 
by sorry

end baseball_card_value_decrease_l158_158341


namespace race_outcomes_l158_158387

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"]

theorem race_outcomes (h : ¬ "Fiona" ∈ ["Abe", "Bobby", "Charles", "Devin", "Edwin"]) : 
  (participants.length - 1) * (participants.length - 2) * (participants.length - 3) = 60 :=
by
  sorry

end race_outcomes_l158_158387


namespace find_first_term_and_ratio_l158_158212

variable (b1 q : ℝ)

-- Conditions
def infinite_geometric_series (q : ℝ) : Prop := |q| < 1

def sum_odd_even_difference (b1 q : ℝ) : Prop := 
  b1 / (1 - q^2) = 2 + (b1 * q) / (1 - q^2)

def sum_square_odd_even_difference (b1 q : ℝ) : Prop :=
  b1^2 / (1 - q^4) - (b1^2 * q^2) / (1 - q^4) = 36 / 5

-- Proof problem
theorem find_first_term_and_ratio (b1 q : ℝ) 
  (h1 : infinite_geometric_series q) 
  (h2 : sum_odd_even_difference b1 q)
  (h3 : sum_square_odd_even_difference b1 q) : 
  b1 = 3 ∧ q = 1 / 2 := by
  sorry

end find_first_term_and_ratio_l158_158212


namespace find_f_2009_l158_158811

-- Defining the function f and specifying the conditions
variable (f : ℝ → ℝ)
axiom h1 : f 3 = -Real.sqrt 3
axiom h2 : ∀ x : ℝ, f (x + 2) * (1 - f x) = 1 + f x

-- Proving the desired statement
theorem find_f_2009 : f 2009 = 2 + Real.sqrt 3 :=
sorry

end find_f_2009_l158_158811


namespace ratio_of_side_lengths_of_frustum_l158_158673

theorem ratio_of_side_lengths_of_frustum (L1 L2 H : ℚ) (V_prism V_frustum : ℚ)
  (h1 : V_prism = L1^2 * H)
  (h2 : V_frustum = (1/3) * (L1^2 * (H * (L1 / (L1 - L2))) - L2^2 * (H * (L2 / (L1 - L2)))))
  (h3 : V_frustum = (2/3) * V_prism) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_side_lengths_of_frustum_l158_158673


namespace volume_pyramid_correct_l158_158631

noncomputable def volume_of_regular_triangular_pyramid 
  (R : ℝ) (β : ℝ) (a : ℝ) : ℝ :=
  (a^3 * (Real.tan β)) / 24

theorem volume_pyramid_correct 
  (R : ℝ) (β : ℝ) (a : ℝ) : 
  volume_of_regular_triangular_pyramid R β a = (a^3 * (Real.tan β)) / 24 :=
sorry

end volume_pyramid_correct_l158_158631


namespace smallest_b_for_quadratic_factorization_l158_158581

theorem smallest_b_for_quadratic_factorization : ∃ (b : ℕ), 
  (∀ r s : ℤ, (r * s = 4032) ∧ (r + s = b) → b ≥ 127) ∧ 
  (∃ r s : ℤ, (r * s = 4032) ∧ (r + s = b) ∧ (b = 127))
:= sorry

end smallest_b_for_quadratic_factorization_l158_158581


namespace randy_blocks_l158_158702

theorem randy_blocks (total_blocks house_blocks diff_blocks tower_blocks : ℕ) 
  (h_total : total_blocks = 90)
  (h_house : house_blocks = 89)
  (h_diff : house_blocks = tower_blocks + diff_blocks)
  (h_diff_value : diff_blocks = 26) :
  tower_blocks = 63 :=
by
  -- sorry is placed here to skip the proof.
  sorry

end randy_blocks_l158_158702


namespace tax_deduction_cents_l158_158940

def bob_hourly_wage : ℝ := 25
def tax_rate : ℝ := 0.025

theorem tax_deduction_cents :
  (bob_hourly_wage * 100 * tax_rate) = 62.5 :=
by
  -- This is the statement that needs to be proven.
  sorry

end tax_deduction_cents_l158_158940


namespace regular_seminar_fee_l158_158891

-- Define the main problem statement
theorem regular_seminar_fee 
  (F : ℝ) 
  (discount_per_teacher : ℝ) 
  (number_of_teachers : ℕ)
  (food_allowance_per_teacher : ℝ)
  (total_spent : ℝ) :
  discount_per_teacher = 0.95 * F →
  number_of_teachers = 10 →
  food_allowance_per_teacher = 10 →
  total_spent = 1525 →
  (number_of_teachers * discount_per_teacher + number_of_teachers * food_allowance_per_teacher = total_spent) →
  F = 150 := 
  by sorry

end regular_seminar_fee_l158_158891


namespace exponent_comparison_l158_158973

theorem exponent_comparison : 1.7 ^ 0.3 > 0.9 ^ 11 := 
by sorry

end exponent_comparison_l158_158973


namespace yvonnes_probability_l158_158138

open Classical

variables (P_X P_Y P_Z : ℝ)

theorem yvonnes_probability
  (h1 : P_X = 1/5)
  (h2 : P_Z = 5/8)
  (h3 : P_X * P_Y * (1 - P_Z) = 0.0375) :
  P_Y = 0.5 :=
by
  sorry

end yvonnes_probability_l158_158138


namespace solve_for_2a_2d_l158_158494

noncomputable def f (a b c d x : ℝ) : ℝ :=
  (2 * a * x + b) / (c * x + 2 * d)

theorem solve_for_2a_2d (a b c d : ℝ) (habcd_ne_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h : ∀ x, f a b c d (f a b c d x) = x) : 2 * a + 2 * d = 0 :=
sorry

end solve_for_2a_2d_l158_158494


namespace product_of_cubes_l158_158030

theorem product_of_cubes :
  ( (2^3 - 1) / (2^3 + 1) * (3^3 - 1) / (3^3 + 1) * (4^3 - 1) / (4^3 + 1) * 
    (5^3 - 1) / (5^3 + 1) * (6^3 - 1) / (6^3 + 1) * (7^3 - 1) / (7^3 + 1) 
  ) = 57 / 72 := 
by
  sorry

end product_of_cubes_l158_158030


namespace gabrielle_total_crates_l158_158405

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l158_158405


namespace downstream_speed_l158_158486

noncomputable def V_b : ℝ := 7
noncomputable def V_up : ℝ := 4
noncomputable def V_s : ℝ := V_b - V_up

theorem downstream_speed :
  V_b + V_s = 10 := sorry

end downstream_speed_l158_158486


namespace find_g2_l158_158351

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 4 * g x - 3 * g (1 / x) = x^2

theorem find_g2 : g 2 = 67 / 28 :=
by {
  sorry
}

end find_g2_l158_158351


namespace orchard_apples_relation_l158_158228

/-- 
A certain orchard has 10 apple trees, and on average each tree can produce 200 apples. 
Based on experience, for each additional tree planted, the average number of apples produced per tree decreases by 5. 
We are to show that if the orchard has planted x additional apple trees and the total number of apples is y, then the relationship between y and x is:
y = (10 + x) * (200 - 5x)
-/
theorem orchard_apples_relation (x : ℕ) (y : ℕ) 
    (initial_trees : ℕ := 10)
    (initial_apples : ℕ := 200)
    (decrease_per_tree : ℕ := 5)
    (total_trees := initial_trees + x)
    (average_apples := initial_apples - decrease_per_tree * x)
    (total_apples := total_trees * average_apples) :
    y = total_trees * average_apples := 
  by 
    sorry

end orchard_apples_relation_l158_158228


namespace distance_between_x_intercepts_l158_158453

theorem distance_between_x_intercepts :
  ∀ (x1 x2 : ℝ),
  (∀ x, x1 = 8 → x2 = 20 → 20 = 4 * (x - 8)) → 
  (∀ x, x1 = 8 → x2 = 20 → 20 = 7 * (x - 8)) → 
  abs ((3 : ℝ) - (36 / 7)) = (15 / 7) :=
by
  intros x1 x2 h1 h2
  sorry

end distance_between_x_intercepts_l158_158453


namespace initial_number_of_girls_l158_158740

theorem initial_number_of_girls (b g : ℤ) 
  (h1 : b = 3 * (g - 20)) 
  (h2 : 3 * (b - 30) = g - 20) : 
  g = 31 :=
by
  sorry

end initial_number_of_girls_l158_158740


namespace fifth_inequality_nth_inequality_solve_given_inequality_l158_158571

theorem fifth_inequality :
  ∀ x, 1 < x ∧ x < 2 → (x + 2 / x < 3) →
  ∀ x, 3 < x ∧ x < 4 → (x + 12 / x < 7) →
  ∀ x, 5 < x ∧ x < 6 → (x + 30 / x < 11) →
  (x + 90 / x < 19) := by
  sorry

theorem nth_inequality (n : ℕ) :
  ∀ x, (2 * n - 1 < x ∧ x < 2 * n) →
  (x + 2 * n * (2 * n - 1) / x < 4 * n - 1) := by
  sorry

theorem solve_given_inequality (a : ℕ) (x : ℝ) (h_a_pos: 0 < a) :
  x + 12 * a / (x + 1) < 4 * a + 2 →
  (2 < x ∧ x < 4 * a - 1) := by
  sorry

end fifth_inequality_nth_inequality_solve_given_inequality_l158_158571


namespace evaluate_expression_l158_158836

theorem evaluate_expression : (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 :=
by
  sorry

end evaluate_expression_l158_158836


namespace isosceles_triangle_perimeter_l158_158559

theorem isosceles_triangle_perimeter (m x₁ x₂ : ℝ) (h₁ : 1^2 + m * 1 + 5 = 0) 
  (hx : x₁^2 + m * x₁ + 5 = 0 ∧ x₂^2 + m * x₂ + 5 = 0)
  (isosceles : (x₁ = x₂ ∨ x₁ = 1 ∨ x₂ = 1)) : 
  ∃ (P : ℝ), P = 11 :=
by 
  -- Here, you'd prove that under these conditions, the perimeter must be 11.
  sorry

end isosceles_triangle_perimeter_l158_158559


namespace time_for_2km_l158_158328

def distance_over_time (t : ℕ) : ℝ := 
  sorry -- Function representing the distance walked over time

theorem time_for_2km : ∃ t : ℕ, distance_over_time t = 2 ∧ t = 105 :=
by
  sorry

end time_for_2km_l158_158328


namespace machines_make_2550_copies_l158_158226

def total_copies (rate1 rate2 : ℕ) (time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

theorem machines_make_2550_copies :
  total_copies 30 55 30 = 2550 :=
by
  unfold total_copies
  decide

end machines_make_2550_copies_l158_158226


namespace exists_perfect_square_subtraction_l158_158205

theorem exists_perfect_square_subtraction {k : ℕ} (hk : k > 0) : 
  ∃ (n : ℕ), n > 0 ∧ ∃ m : ℕ, n * 2^k - 7 = m^2 := 
  sorry

end exists_perfect_square_subtraction_l158_158205


namespace right_triangle_median_l158_158524

variable (A B C M N : Type) [LinearOrder B] [LinearOrder C] [LinearOrder A] [LinearOrder M] [LinearOrder N]
variable (AC BC AM BN AB : ℝ)
variable (right_triangle : AC * AC + BC * BC = AB * AB)
variable (median_A : AC * AC + (1 / 4) * BC * BC = 81)
variable (median_B : BC * BC + (1 / 4) * AC * AC = 99)

theorem right_triangle_median :
  ∀ (AC BC AB : ℝ),
  (AC * AC + BC * BC = 144) → (AC * AC + BC * BC = AB * AB) → AB = 12 :=
by
  intros
  sorry

end right_triangle_median_l158_158524


namespace even_decreasing_function_l158_158899

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_decreasing_function :
  is_even f →
  is_decreasing_on_nonneg f →
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end even_decreasing_function_l158_158899


namespace gcd_a2_14a_49_a_7_l158_158898

theorem gcd_a2_14a_49_a_7 (a : ℤ) (k : ℤ) (h : a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := 
by
  sorry

end gcd_a2_14a_49_a_7_l158_158898


namespace riding_is_four_times_walking_l158_158393

variable (D : ℝ) -- Total distance of the route
variable (v_r v_w : ℝ) -- Riding speed and walking speed
variable (t_r t_w : ℝ) -- Time spent riding and walking

-- Conditions given in the problem
axiom distance_riding : (2/3) * D = v_r * t_r
axiom distance_walking : (1/3) * D = v_w * t_w
axiom time_relation : t_w = 2 * t_r

-- Desired statement to prove
theorem riding_is_four_times_walking : v_r = 4 * v_w := by
  sorry

end riding_is_four_times_walking_l158_158393


namespace number_of_connections_l158_158053

theorem number_of_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 :=
by
  sorry

end number_of_connections_l158_158053


namespace last_three_digits_of_7_pow_120_l158_158302

theorem last_three_digits_of_7_pow_120 :
  7^120 % 1000 = 681 :=
by
  sorry

end last_three_digits_of_7_pow_120_l158_158302


namespace distinct_solutions_subtract_eight_l158_158958

noncomputable def f (x : ℝ) : ℝ := (6 * x - 18) / (x^2 + 2 * x - 15)
noncomputable def equation := ∀ x, f x = x + 3

noncomputable def r_solutions (r s : ℝ) := (r > s) ∧ (f r = r + 3) ∧ (f s = s + 3)

theorem distinct_solutions_subtract_eight
  (r s : ℝ) (h : r_solutions r s) : r - s = 8 :=
sorry

end distinct_solutions_subtract_eight_l158_158958


namespace sum_of_squares_of_roots_eq_21_l158_158910

theorem sum_of_squares_of_roots_eq_21 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + x2^2 = 21 ∧ x1 + x2 = -a ∧ x1 * x2 = 2*a) ↔ a = -3 :=
by
  sorry

end sum_of_squares_of_roots_eq_21_l158_158910


namespace rectangle_perimeters_l158_158339

theorem rectangle_perimeters (length width : ℕ) (h1 : length = 7) (h2 : width = 5) :
  (∃ (L1 L2 : ℕ), L1 = 4 * width ∧ L2 = length ∧ 2 * (L1 + L2) = 54) ∧
  (∃ (L3 L4 : ℕ), L3 = 4 * length ∧ L4 = width ∧ 2 * (L3 + L4) = 66) ∧
  (∃ (L5 L6 : ℕ), L5 = 2 * length ∧ L6 = 2 * width ∧ 2 * (L5 + L6) = 48) :=
by
  sorry

end rectangle_perimeters_l158_158339


namespace shortest_side_15_l158_158741

theorem shortest_side_15 (b c : ℕ) (h : ℕ) (hb : b < c)
  (h_perimeter : 24 + b + c = 66)
  (h_area_int : ∃ A : ℕ, A*A = 33 * 9 * (33 - b) * (b - 9))
  (h_altitude_int : ∃ A : ℕ, 24 * h = 2 * A) : b = 15 :=
sorry

end shortest_side_15_l158_158741


namespace percentage_of_loss_l158_158383

-- Define the conditions as given in the problem
def original_selling_price : ℝ := 720
def gain_selling_price : ℝ := 880
def gain_percentage : ℝ := 0.10

-- Define the main theorem
theorem percentage_of_loss : ∀ (CP : ℝ),
  (1.10 * CP = gain_selling_price) → 
  ((CP - original_selling_price) / CP * 100 = 10) :=
by
  intro CP
  intro h
  have h1 : CP = gain_selling_price / 1.10 := by sorry
  have h2 : (CP - original_selling_price) = 80 := by sorry -- Intermediate step to show loss
  have h3 : ((80 / CP) * 100 = 10) := by sorry -- Calculation of percentage of loss
  sorry

end percentage_of_loss_l158_158383


namespace tickets_savings_percentage_l158_158500

theorem tickets_savings_percentage (P S : ℚ) (h : 8 * S = 5 * P) :
  (12 * P - 12 * S) / (12 * P) * 100 = 37.5 :=
by 
  sorry

end tickets_savings_percentage_l158_158500


namespace incorrect_transformation_l158_158676

theorem incorrect_transformation :
  ¬ ∀ (a b c : ℝ), ac = bc → a = b :=
by
  sorry

end incorrect_transformation_l158_158676


namespace suitable_comprehensive_survey_l158_158562

def investigate_service_life_of_lamps : Prop := 
  -- This would typically involve checking a subset rather than every lamp
  sorry

def investigate_water_quality : Prop := 
  -- This would typically involve sampling rather than checking every point
  sorry

def investigate_sports_activities : Prop := 
  -- This would typically involve sampling rather than collecting data on every student
  sorry

def test_components_of_rocket : Prop := 
  -- Given the critical importance and manageable number of components, this requires comprehensive examination
  sorry

def most_suitable_for_comprehensive_survey : Prop :=
  test_components_of_rocket ∧ ¬investigate_service_life_of_lamps ∧ 
  ¬investigate_water_quality ∧ ¬investigate_sports_activities

theorem suitable_comprehensive_survey : most_suitable_for_comprehensive_survey :=
  sorry

end suitable_comprehensive_survey_l158_158562


namespace find_number_l158_158795

theorem find_number
  (x : ℝ)
  (h : 0.90 * x = 0.50 * 1080) :
  x = 600 :=
by
  sorry

end find_number_l158_158795


namespace face_opposite_of_E_l158_158758

-- Definitions of faces and their relationships
inductive Face : Type
| A | B | C | D | E | F | x

open Face

-- Adjacency relationship
def is_adjacent_to (f1 f2 : Face) : Prop :=
(f1 = x ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = x ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D)) ∨
(f1 = E ∧ (f2 = A ∨ f2 = B ∨ f2 = C ∨ f2 = D)) ∨
(f2 = E ∧ (f1 = A ∨ f1 = B ∨ f1 = C ∨ f1 = D))

-- Non-adjacency relationship
def is_opposite (f1 f2 : Face) : Prop :=
∀ f : Face, is_adjacent_to f1 f → ¬ is_adjacent_to f2 f

-- Theorem to prove that F is opposite of E
theorem face_opposite_of_E : is_opposite E F :=
sorry

end face_opposite_of_E_l158_158758


namespace roots_condition_l158_158288

theorem roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 3 ∧ x2 < 3 ∧ x1^2 - m * x1 + 2 * m = 0 ∧ x2^2 - m * x2 + 2 * m = 0) ↔ m > 9 :=
by sorry

end roots_condition_l158_158288


namespace smallest_of_three_consecutive_l158_158501

theorem smallest_of_three_consecutive (x : ℤ) (h : x + (x + 1) + (x + 2) = 90) : x = 29 :=
by
  sorry

end smallest_of_three_consecutive_l158_158501


namespace at_most_one_negative_l158_158881

theorem at_most_one_negative (a b c : ℝ) (h1 : a + b + c ≥ 0) (h2 : abc ≤ 0) : 
  (a < 0 ∧ b >= 0 ∧ c >= 0) ∨ (a >= 0 ∧ b < 0 ∧ c >= 0) ∨ (a >= 0 ∧ b >= 0 ∧ c < 0) ∨ 
  (a >= 0 ∧ b >= 0 ∧ c >= 0) :=
sorry

end at_most_one_negative_l158_158881


namespace boxes_needed_l158_158070

theorem boxes_needed (total_muffins available_boxes muffins_per_box : ℕ) (h1 : total_muffins = 95) 
  (h2 : available_boxes = 10) (h3 : muffins_per_box = 5) : 
  ((total_muffins - (available_boxes * muffins_per_box)) / muffins_per_box) = 9 := 
by
  -- the proof will be constructed here
  sorry

end boxes_needed_l158_158070


namespace largest_integral_solution_l158_158857

noncomputable def largest_integral_value : ℤ :=
  let a : ℚ := 1 / 4
  let b : ℚ := 7 / 11 
  let lower_bound : ℚ := 7 * a
  let upper_bound : ℚ := 7 * b
  let x := 3  -- The largest integral value within the bounds
  x

-- A theorem to prove that x = 3 satisfies the inequality conditions and is the largest integer.
theorem largest_integral_solution (x : ℤ) (h₁ : 1 / 4 < x / 7) (h₂ : x / 7 < 7 / 11) : x = 3 := by
  sorry

end largest_integral_solution_l158_158857


namespace work_problem_l158_158598

/--
Given:
1. A and B together can finish the work in 16 days.
2. B alone can finish the work in 48 days.
To Prove:
A alone can finish the work in 24 days.
-/
theorem work_problem (a b : ℕ)
  (h1 : a + b = 16)
  (h2 : b = 48) :
  a = 24 := 
sorry

end work_problem_l158_158598


namespace julia_monday_kids_l158_158225

theorem julia_monday_kids (x : ℕ) (h1 : x + 14 = 16) : x = 2 := 
by
  sorry

end julia_monday_kids_l158_158225


namespace finite_perfect_squares_l158_158265

noncomputable def finite_squares (a b : ℕ) : Prop :=
  ∃ (f : Finset ℕ), ∀ n, n ∈ f ↔ 
    ∃ (x y : ℕ), a * n ^ 2 + b = x ^ 2 ∧ a * (n + 1) ^ 2 + b = y ^ 2

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  finite_squares a b :=
sorry

end finite_perfect_squares_l158_158265


namespace math_problem_l158_158476

-- Conditions
def ellipse_eq (a b : ℝ) : Prop := ∀ x y : ℝ, x^2 / (a^2) + y^2 / (b^2) = 1
def eccentricity (a c : ℝ) : Prop := c / a = (Real.sqrt 2) / 2
def major_axis_length (a : ℝ) : Prop := 2 * a = 6 * Real.sqrt 2

-- Equations and properties to be proven
def ellipse_equation : Prop := ∃ a b : ℝ, a = 3 * Real.sqrt 2 ∧ b = 3 ∧ ellipse_eq a b
def length_AB (θ : ℝ) : Prop := ∃ AB : ℝ, AB = (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2)
def min_AB_CD : Prop := ∃ θ : ℝ, (Real.sin (2 * θ) = 1) ∧ (6 * Real.sqrt 2) / (1 + (Real.sin θ)^2) + (6 * Real.sqrt 2) / (1 + (Real.cos θ)^2) = 8 * Real.sqrt 2

-- The complete proof problem
theorem math_problem : ellipse_equation ∧
                       (∀ θ : ℝ, length_AB θ) ∧
                       min_AB_CD := by
  sorry

end math_problem_l158_158476


namespace ants_in_field_l158_158338

-- Defining constants
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 4
def inches_per_foot : ℕ := 12

-- Converting dimensions from feet to inches
def width_inches : ℕ := width_feet * inches_per_foot
def length_inches : ℕ := length_feet * inches_per_foot

-- Calculating the area of the field in square inches
def field_area_square_inches : ℕ := width_inches * length_inches

-- Calculating the total number of ants
def total_ants : ℕ := ants_per_square_inch * field_area_square_inches

-- Theorem statement
theorem ants_in_field : total_ants = 172800000 :=
by
  -- Proof is skipped
  sorry

end ants_in_field_l158_158338


namespace abs_diff_of_numbers_l158_158221

theorem abs_diff_of_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 320) : |x - y| = 4 :=
by
  sorry

end abs_diff_of_numbers_l158_158221


namespace fraction_white_surface_area_l158_158553

-- Definitions of the given conditions
def cube_side_length : ℕ := 4
def small_cubes : ℕ := 64
def black_cubes : ℕ := 34
def white_cubes : ℕ := 30
def total_surface_area : ℕ := 6 * cube_side_length^2
def black_faces_exposed : ℕ := 32 
def white_faces_exposed : ℕ := total_surface_area - black_faces_exposed

-- The proof statement
theorem fraction_white_surface_area (cube_side_length_eq : cube_side_length = 4)
                                    (small_cubes_eq : small_cubes = 64)
                                    (black_cubes_eq : black_cubes = 34)
                                    (white_cubes_eq : white_cubes = 30)
                                    (black_faces_eq : black_faces_exposed = 32)
                                    (total_surface_area_eq : total_surface_area = 96)
                                    (white_faces_eq : white_faces_exposed = 64) : 
                                    (white_faces_exposed : ℚ) / (total_surface_area : ℚ) = 2 / 3 :=
by
  sorry

end fraction_white_surface_area_l158_158553


namespace four_digit_number_l158_158485

def digit_constraint (A B C D : ℕ) : Prop :=
  A = B / 3 ∧ C = A + B ∧ D = 3 * B

theorem four_digit_number 
  (A B C D : ℕ) 
  (h₁ : A = B / 3) 
  (h₂ : C = A + B) 
  (h₃ : D = 3 * B)
  (hA_digit : A < 10) 
  (hB_digit : B < 10)
  (hC_digit : C < 10)
  (hD_digit : D < 10) :
  1000 * A + 100 * B + 10 * C + D = 1349 := 
sorry

end four_digit_number_l158_158485


namespace average_ABC_l158_158436

/-- Given three numbers A, B, and C such that 1503C - 3006A = 6012 and 1503B + 4509A = 7509,
their average is 3  -/
theorem average_ABC (A B C : ℚ) 
  (h1 : 1503 * C - 3006 * A = 6012) 
  (h2 : 1503 * B + 4509 * A = 7509) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_l158_158436


namespace cos_double_beta_eq_24_over_25_l158_158622

theorem cos_double_beta_eq_24_over_25
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.cos (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (π / 2) π) :
  Real.cos (2 * β) = 24 / 25 := sorry

end cos_double_beta_eq_24_over_25_l158_158622


namespace sin_alpha_beta_eq_l158_158414

theorem sin_alpha_beta_eq 
  (α β : ℝ) 
  (h1 : π / 4 < α) (h2 : α < 3 * π / 4)
  (h3 : 0 < β) (h4 : β < π / 4)
  (h5: Real.sin (α + π / 4) = 3 / 5)
  (h6: Real.cos (π / 4 + β) = 5 / 13) :
  Real.sin (α + β) = 56 / 65 :=
sorry

end sin_alpha_beta_eq_l158_158414


namespace total_bulbs_needed_l158_158846

-- Definitions according to the conditions.
variables (T S M L XL : ℕ)

-- Conditions
variables (cond1 : L = 2 * M)
variables (cond2 : S = 5 * M / 4)  -- since 1.25M = 5/4M
variables (cond3 : XL = S - T)
variables (cond4 : 4 * T = 3 * M) -- equivalent to T / M = 3 / 4
variables (cond5 : 2 * S + 3 * M = 4 * L + 5 * XL)
variables (cond6 : XL = 14)

-- Prove total bulbs needed
theorem total_bulbs_needed :
  T + 2 * S + 3 * M + 4 * L + 5 * XL = 469 :=
sorry

end total_bulbs_needed_l158_158846


namespace olympic_medals_l158_158335

theorem olympic_medals :
  ∃ (a b c : ℕ),
    (a + b + c = 100) ∧
    (3 * a - 153 = 0) ∧
    (c - b = 7) ∧
    (a = 51) ∧
    (a - 13 = 38) ∧
    (c = 28) :=
by
  sorry

end olympic_medals_l158_158335


namespace number_of_dots_in_120_circles_l158_158627

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_l158_158627


namespace seashells_left_sam_seashells_now_l158_158656

-- Problem conditions
def initial_seashells : ℕ := 35
def seashells_given : ℕ := 18

-- Proof problem statement
theorem seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

-- The required statement
theorem sam_seashells_now : seashells_left initial_seashells seashells_given = 17 := by 
  sorry

end seashells_left_sam_seashells_now_l158_158656


namespace determine_digits_l158_158768

theorem determine_digits (h t u : ℕ) (hu: h > u) (h_subtr: t = h - 5) (unit_result: u = 3) : (h = 9 ∧ t = 4 ∧ u = 3) := by
  sorry

end determine_digits_l158_158768


namespace minimum_packs_needed_l158_158820

theorem minimum_packs_needed (n : ℕ) :
  (∃ x y z : ℕ, 30 * x + 18 * y + 9 * z = 120 ∧ x + y + z = n ∧ x ≥ 2 ∧ z' = if x ≥ 2 then z + 1 else z) → n = 4 := 
by
  sorry

end minimum_packs_needed_l158_158820


namespace harmon_high_voting_l158_158135

theorem harmon_high_voting
  (U : Finset ℝ) -- Universe of students
  (A B : Finset ℝ) -- Sets of students favoring proposals
  (hU : U.card = 215)
  (hA : A.card = 170)
  (hB : B.card = 142)
  (hAcBc : (U \ (A ∪ B)).card = 38) :
  (A ∩ B).card = 135 :=
by {
  sorry
}

end harmon_high_voting_l158_158135


namespace eight_step_paths_board_l158_158159

theorem eight_step_paths_board (P Q : ℕ) (hP : P = 0) (hQ : Q = 7) : 
  ∃ (paths : ℕ), paths = 70 :=
by
  sorry

end eight_step_paths_board_l158_158159


namespace hotel_bill_amount_l158_158723

-- Definition of the variables used in the conditions
def each_paid : ℝ := 124.11
def friends : ℕ := 9

-- The Lean 4 theorem statement
theorem hotel_bill_amount :
  friends * each_paid = 1116.99 := sorry

end hotel_bill_amount_l158_158723


namespace solution_set_of_inequalities_l158_158589

-- Define the conditions of the inequality system
def inequality1 (x : ℝ) : Prop := x - 2 ≥ -5
def inequality2 (x : ℝ) : Prop := 3 * x < x + 2

-- The statement to prove the solution set of the inequalities
theorem solution_set_of_inequalities :
  { x : ℝ | inequality1 x ∧ inequality2 x } = { x : ℝ | -3 ≤ x ∧ x < 1 } :=
  sorry

end solution_set_of_inequalities_l158_158589


namespace remainder_when_divided_by_multiple_of_10_l158_158896

theorem remainder_when_divided_by_multiple_of_10 (N : ℕ) (hN : ∃ k : ℕ, N = 10 * k) (hrem : (19 ^ 19 + 19) % N = 18) : N = 10 := by
  sorry

end remainder_when_divided_by_multiple_of_10_l158_158896


namespace probability_heads_mod_coin_l158_158792

theorem probability_heads_mod_coin (p : ℝ) (h : 20 * p ^ 3 * (1 - p) ^ 3 = 1 / 20) : p = (1 - Real.sqrt 0.6816) / 2 :=
by
  sorry

end probability_heads_mod_coin_l158_158792


namespace correct_multiplication_result_l158_158445

theorem correct_multiplication_result (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 :=
  by
  sorry

end correct_multiplication_result_l158_158445


namespace num_k_values_lcm_l158_158888

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l158_158888


namespace number_of_semesters_l158_158952

-- Define the given conditions
def units_per_semester : ℕ := 20
def cost_per_unit : ℕ := 50
def total_cost : ℕ := 2000

-- Define the cost per semester using the conditions
def cost_per_semester := units_per_semester * cost_per_unit

-- Prove the number of semesters is 2 given the conditions
theorem number_of_semesters : total_cost / cost_per_semester = 2 := by
  -- Add a placeholder "sorry" to skip the actual proof
  sorry

end number_of_semesters_l158_158952


namespace mary_needs_more_apples_l158_158187

theorem mary_needs_more_apples :
  let pies := 15
  let apples_per_pie := 10
  let harvested_apples := 40
  let total_apples_needed := pies * apples_per_pie
  let more_apples_needed := total_apples_needed - harvested_apples
  more_apples_needed = 110 :=
by
  sorry

end mary_needs_more_apples_l158_158187


namespace equilateral_triangle_perimeter_l158_158731

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l158_158731


namespace find_integer_x_l158_158206

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  -1 < x ∧ x < 7 ∧
  0 < x ∧ x < 4 ∧
  x + 1 < 5 → 
  x = 3 :=
by
  sorry

end find_integer_x_l158_158206


namespace find_other_root_l158_158418

theorem find_other_root (a b : ℝ) (h₁ : 3^2 + 3 * a - 2 * a = 0) (h₂ : ∀ x, x^2 + a * x - 2 * a = 0 → (x = 3 ∨ x = b)) :
  b = 6 := 
sorry

end find_other_root_l158_158418


namespace number_divided_by_3_equals_subtract_3_l158_158556

theorem number_divided_by_3_equals_subtract_3 (x : ℝ) (h : x / 3 = x - 3) : x = 4.5 :=
sorry

end number_divided_by_3_equals_subtract_3_l158_158556


namespace gcd_2703_1113_l158_158668

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := 
by 
  sorry

end gcd_2703_1113_l158_158668


namespace worker_times_l158_158145

-- Define the problem
theorem worker_times (x y : ℝ) (h1 : (1 / x + 1 / y = 1 / 8)) (h2 : x = y - 12) :
    x = 24 ∧ y = 12 :=
by
  sorry

end worker_times_l158_158145


namespace gcd_poly_l158_158582

theorem gcd_poly {b : ℕ} (h : 1116 ∣ b) : Nat.gcd (b^2 + 11 * b + 36) (b + 6) = 6 :=
by
  sorry

end gcd_poly_l158_158582


namespace star_is_addition_l158_158142

variable {α : Type} [AddCommGroup α]

-- Define the binary operation star
variable (star : α → α → α)

-- Define the condition given in the problem
axiom star_condition : ∀ (a b c : α), star (star a b) c = a + b + c

-- Prove that star is the same as usual addition
theorem star_is_addition : ∀ (a b : α), star a b = a + b :=
  sorry

end star_is_addition_l158_158142


namespace range_of_m_value_of_m_l158_158787

variables (m p x : ℝ)

-- Conditions: The quadratic equation x^2 - 2x + m - 1 = 0 must have two real roots.
def discriminant (m : ℝ) := (-2)^2 - 4 * 1 * (m - 1)

-- Part 1: Finding the range of values for m
theorem range_of_m (h : discriminant m ≥ 0) : m ≤ 2 := 
by sorry

-- Additional Condition: p is a real root of the equation x^2 - 2x + m - 1 = 0
def is_root (p m : ℝ) := p^2 - 2 * p + m - 1 = 0

-- Another condition: (p^2 - 2p + 3)(m + 4) = 7
def satisfies_condition (p m : ℝ) := (p^2 - 2 * p + 3) * (m + 4) = 7

-- Part 2: Finding the value of m given p is a real root and satisfies (p^2 - 2p + 3)(m + 4) = 7
theorem value_of_m (h1 : is_root p m) (h2 : satisfies_condition p m) : m = -3 := 
by sorry

end range_of_m_value_of_m_l158_158787


namespace min_value_max_value_l158_158214

theorem min_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 0) := sorry

theorem max_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 44) := sorry

end min_value_max_value_l158_158214


namespace find_alpha_l158_158860

noncomputable def parametric_eq_line (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_eq_curve (x y : Real) : Prop :=
  y^2 = 4 * x

def intersection_condition (α t₁ t₂ : Real) : Prop :=
  Real.sin α ≠ 0 ∧ 
  (1 + t₁ * Real.cos α, t₁ * Real.sin α) = (1 + t₂ * Real.cos α, t₂ * Real.sin α) ∧ 
  Real.sqrt ((t₁ + t₂)^2 - 4 * (-4 / (Real.sin α)^2)) = 8

theorem find_alpha (α : Real) (t₁ t₂ : Real) 
  (h1: 0 < α) (h2: α < π) (h3: intersection_condition α t₁ t₂) : 
  α = π/4 ∨ α = 3*π/4 :=
by 
  sorry

end find_alpha_l158_158860


namespace inequality_solution_l158_158527

theorem inequality_solution (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17 / 5) := 
sorry

end inequality_solution_l158_158527


namespace fraction_operation_l158_158983

theorem fraction_operation : (3 / 5 - 1 / 10 + 2 / 15 = 19 / 30) :=
by
  sorry

end fraction_operation_l158_158983


namespace average_percentage_decrease_l158_158575

theorem average_percentage_decrease : 
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ ((2000 * (1 - x)^2 = 1280) ↔ (x = 0.18)) :=
by
  sorry

end average_percentage_decrease_l158_158575


namespace problem1_problem2_l158_158565

theorem problem1 (x : ℚ) (h : x ≠ -4) : (3 - x) / (x + 4) = 1 / 2 → x = 2 / 3 :=
by
  sorry

theorem problem2 (x : ℚ) (h : x ≠ 1) : x / (x - 1) - 2 * x / (3 * (x - 1)) = 1 → x = 3 / 2 :=
by
  sorry

end problem1_problem2_l158_158565


namespace sqrt_mul_power_expr_l158_158432

theorem sqrt_mul_power_expr : ( (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 ) = (Real.sqrt 3 + Real.sqrt 2) := 
  sorry

end sqrt_mul_power_expr_l158_158432


namespace solution_l158_158759

noncomputable def polynomial (x m : ℝ) := 3 * x^2 - 5 * x + m

theorem solution (m : ℝ) : (∃ a : ℝ, a = 2 ∧ polynomial a m = 0) -> m = -2 := by
  sorry

end solution_l158_158759


namespace max_value_of_expression_l158_158297

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 :=
sorry

end max_value_of_expression_l158_158297


namespace apples_remaining_in_each_basket_l158_158974

-- Definition of conditions
def total_apples : ℕ := 128
def number_of_baskets : ℕ := 8
def apples_taken_per_basket : ℕ := 7

-- Definition of the problem
theorem apples_remaining_in_each_basket :
  (total_apples / number_of_baskets) - apples_taken_per_basket = 9 := 
by 
  sorry

end apples_remaining_in_each_basket_l158_158974


namespace other_endpoint_sum_l158_158825

def endpoint_sum (A B M : (ℝ × ℝ)) : ℝ := 
  let (Ax, Ay) := A
  let (Mx, My) := M
  let (Bx, By) := B
  Bx + By

theorem other_endpoint_sum (A M : (ℝ × ℝ)) (hA : A = (6, 1)) (hM : M = (5, 7)) :
  ∃ B : (ℝ × ℝ), endpoint_sum A B M = 17 :=
by
  use (4, 13)
  rw [endpoint_sum, hA, hM]
  simp
  sorry

end other_endpoint_sum_l158_158825


namespace blankets_warmth_increase_l158_158509

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end blankets_warmth_increase_l158_158509


namespace rongrong_bike_speed_l158_158130

theorem rongrong_bike_speed :
  ∃ (x : ℝ), (15 / x - 15 / (4 * x) = 45 / 60) → x = 15 :=
by
  sorry

end rongrong_bike_speed_l158_158130


namespace problem_solution_l158_158960

noncomputable def question (x y z : ℝ) : Prop := 
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) → 
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) )

theorem problem_solution (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) →
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ) :=
sorry

end problem_solution_l158_158960


namespace XiaoYing_minimum_water_usage_l158_158083

-- Definitions based on the problem's conditions
def first_charge_rate : ℝ := 2.8
def excess_charge_rate : ℝ := 3
def initial_threshold : ℝ := 5
def minimum_bill : ℝ := 29

-- Main statement for the proof based on the derived inequality
theorem XiaoYing_minimum_water_usage (x : ℝ) (h1 : 2.8 * initial_threshold + 3 * (x - initial_threshold) ≥ 29) : x ≥ 10 := by
  sorry

end XiaoYing_minimum_water_usage_l158_158083


namespace find_line_equation_l158_158390

theorem find_line_equation :
  ∃ (a b c : ℝ), (a * -5 + b * -1 = c) ∧ (a * 1 + b * 1 = c + 2) ∧ (b ≠ 0) ∧ (a * 2 + b = 0) → (∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = -5) :=
by
  sorry

end find_line_equation_l158_158390


namespace train_crossing_time_l158_158507

theorem train_crossing_time
  (train_length : ℕ)           -- length of the train in meters
  (train_speed_kmh : ℕ)        -- speed of the train in kilometers per hour
  (conversion_factor : ℕ)      -- conversion factor from km/hr to m/s
  (train_speed_ms : ℕ)         -- speed of the train in meters per second
  (time_to_cross : ℚ)          -- time to cross in seconds
  (h1 : train_length = 60)
  (h2 : train_speed_kmh = 144)
  (h3 : conversion_factor = 1000 / 3600)
  (h4 : train_speed_ms = train_speed_kmh * conversion_factor)
  (h5 : time_to_cross = train_length / train_speed_ms) :
  time_to_cross = 1.5 :=
by sorry

end train_crossing_time_l158_158507


namespace find_a_l158_158659

theorem find_a
  (x1 x2 a : ℝ)
  (h1 : x1^2 + 4 * x1 - 3 = 0)
  (h2 : x2^2 + 4 * x2 - 3 = 0)
  (h3 : 2 * x1 * (x2^2 + 3 * x2 - 3) + a = 2) :
  a = -4 :=
sorry

end find_a_l158_158659


namespace total_cost_of_toys_l158_158688

def cost_of_toy_cars : ℝ := 14.88
def cost_of_toy_trucks : ℝ := 5.86

theorem total_cost_of_toys :
  cost_of_toy_cars + cost_of_toy_trucks = 20.74 :=
by
  sorry

end total_cost_of_toys_l158_158688


namespace min_distance_parabola_l158_158472

open Real

theorem min_distance_parabola {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) : ∃ m : ℝ, m = 2 * sqrt 3 ∧ ∀ Q : ℝ × ℝ, Q = (4, 0) → dist P Q ≥ m :=
by sorry

end min_distance_parabola_l158_158472


namespace anne_more_drawings_l158_158538

/-- Anne's markers problem setup. -/
structure MarkerProblem :=
  (markers : ℕ)
  (drawings_per_marker : ℚ)
  (drawings_made : ℕ)

-- Given conditions
def anne_conditions : MarkerProblem :=
  { markers := 12, drawings_per_marker := 1.5, drawings_made := 8 }

-- Equivalent proof problem statement in Lean
theorem anne_more_drawings(conditions : MarkerProblem) : 
  conditions.markers * conditions.drawings_per_marker - conditions.drawings_made = 10 :=
by
  -- The proof of this theorem is omitted
  sorry

end anne_more_drawings_l158_158538


namespace incorrect_average_initially_calculated_l158_158149

theorem incorrect_average_initially_calculated :
  ∀ (S' S : ℕ) (n : ℕ) (incorrect_correct_difference : ℕ),
  n = 10 →
  incorrect_correct_difference = 30 →
  S = 200 →
  S' = S - incorrect_correct_difference →
  (S' / n) = 17 :=
by
  intros S' S n incorrect_correct_difference h_n h_diff h_S h_S' 
  sorry

end incorrect_average_initially_calculated_l158_158149


namespace original_price_l158_158856

theorem original_price (P : ℕ) (h : (1 / 8) * P = 8) : P = 64 :=
sorry

end original_price_l158_158856


namespace arctan_sum_eq_pi_div_two_l158_158450

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l158_158450


namespace total_sandwiches_prepared_l158_158365

def num_people := 219.0
def sandwiches_per_person := 3.0

theorem total_sandwiches_prepared : num_people * sandwiches_per_person = 657.0 :=
by
  sorry

end total_sandwiches_prepared_l158_158365


namespace cole_round_trip_time_l158_158424

theorem cole_round_trip_time :
  ∀ (speed_to_work speed_return : ℝ) (time_to_work_minutes : ℝ),
  speed_to_work = 75 ∧ speed_return = 105 ∧ time_to_work_minutes = 210 →
  (time_to_work_minutes / 60 + (speed_to_work * (time_to_work_minutes / 60)) / speed_return) = 6 := 
by
  sorry

end cole_round_trip_time_l158_158424


namespace sum_product_le_four_l158_158957

theorem sum_product_le_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := 
sorry

end sum_product_le_four_l158_158957


namespace quadrant_conditions_l158_158055

-- Formalizing function and conditions in Lean specifics
variable {a b : ℝ}

theorem quadrant_conditions 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 0 < a ∧ a < 1)
  (h4 : ∀ x < 0, a^x + b - 1 > 0)
  (h5 : ∀ x > 0, a^x + b - 1 > 0) :
  0 < b ∧ b < 1 := 
sorry

end quadrant_conditions_l158_158055


namespace find_k_value_l158_158178

-- Define the lines l1 and l2 with given conditions
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the condition for the quadrilateral to be circumscribed by a circle
def is_circumscribed (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ line2 k x y ∧ 0 < x ∧ 0 < y

theorem find_k_value (k : ℝ) : is_circumscribed k → k = 3 := 
sorry

end find_k_value_l158_158178


namespace integral_solution_l158_158162

noncomputable def definite_integral : ℝ :=
  ∫ x in (-2 : ℝ)..(0 : ℝ), (x + 2)^2 * (Real.cos (3 * x))

theorem integral_solution :
  definite_integral = (12 - 2 * Real.sin 6) / 27 :=
sorry

end integral_solution_l158_158162


namespace sum_of_digits_of_77_is_14_l158_158307

-- Define the conditions given in the problem
def triangular_array_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define what it means to be the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- The actual Lean theorem statement
theorem sum_of_digits_of_77_is_14 (N : ℕ) (h : triangular_array_sum N = 3003) : sum_of_digits N = 14 :=
by
  sorry  -- Proof to be completed here

end sum_of_digits_of_77_is_14_l158_158307


namespace squirrel_travel_distance_l158_158016

theorem squirrel_travel_distance
  (height: ℝ)
  (circumference: ℝ)
  (vertical_rise: ℝ)
  (num_circuits: ℝ):
  height = 25 →
  circumference = 3 →
  vertical_rise = 5 →
  num_circuits = height / vertical_rise →
  (num_circuits * circumference) ^ 2 + height ^ 2 = 850 :=
by
  sorry

end squirrel_travel_distance_l158_158016


namespace gcd_14568_78452_l158_158751

theorem gcd_14568_78452 : Nat.gcd 14568 78452 = 4 :=
sorry

end gcd_14568_78452_l158_158751


namespace math_problem_l158_158085

theorem math_problem (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1)^2 = 0) : (x + 2 * y)^3 = 125 / 8 := 
sorry

end math_problem_l158_158085


namespace solve_for_diamond_l158_158243

-- Define what it means for a digit to represent a base-9 number and base-10 number
noncomputable def fromBase (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc d => acc * b + d) 0

-- The theorem we want to prove
theorem solve_for_diamond (diamond : ℕ) (h_digit : diamond < 10) :
  fromBase 9 [diamond, 3] = fromBase 10 [diamond, 2] → diamond = 1 :=
by 
  sorry

end solve_for_diamond_l158_158243


namespace Spot_dog_reachable_area_l158_158447

noncomputable def Spot_reachable_area (side_length tether_length : ℝ) : ℝ := 
  -- Note here we compute using the areas described in the problem
  6 * Real.pi * (tether_length^2) / 3 - Real.pi * (side_length^2)

theorem Spot_dog_reachable_area (side_length tether_length : ℝ)
  (H1 : side_length = 2) (H2 : tether_length = 3) :
    Spot_reachable_area side_length tether_length = (22 * Real.pi) / 3 := by
  sorry

end Spot_dog_reachable_area_l158_158447


namespace quadratic_roots_equal_l158_158081

theorem quadratic_roots_equal {k : ℝ} (h : (2 * k) ^ 2 - 4 * 1 * (k^2 + k + 3) = 0) : k^2 + k + 3 = 9 :=
by
  sorry

end quadratic_roots_equal_l158_158081


namespace statement_B_not_true_l158_158653

def op_star (x y : ℝ) := x^2 - 2*x*y + y^2

theorem statement_B_not_true (x y : ℝ) : 3 * (op_star x y) ≠ op_star (3 * x) (3 * y) :=
by
  have h1 : 3 * (op_star x y) = 3 * (x^2 - 2 * x * y + y^2) := rfl
  have h2 : op_star (3 * x) (3 * y) = (3 * x)^2 - 2 * (3 * x) * (3 * y) + (3 * y)^2 := rfl
  sorry

end statement_B_not_true_l158_158653


namespace edward_made_in_summer_l158_158592

def edward_made_in_spring := 2
def cost_of_supplies := 5
def money_left_over := 24

theorem edward_made_in_summer : edward_made_in_spring + x - cost_of_supplies = money_left_over → x = 27 :=
by
  intros h
  sorry

end edward_made_in_summer_l158_158592


namespace unique_solution_quadratic_l158_158171

theorem unique_solution_quadratic (q : ℚ) :
  (∃ x : ℚ, q ≠ 0 ∧ q * x^2 - 16 * x + 9 = 0) ∧ (∀ y z : ℚ, (q * y^2 - 16 * y + 9 = 0 ∧ q * z^2 - 16 * z + 9 = 0) → y = z) → q = 64 / 9 :=
by
  sorry

end unique_solution_quadratic_l158_158171


namespace gretchen_flavors_l158_158391

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end gretchen_flavors_l158_158391


namespace twenty_yuan_banknotes_count_l158_158372

theorem twenty_yuan_banknotes_count (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
                                    (total_banknotes : x + y + z = 24)
                                    (total_amount : 10 * x + 20 * y + 50 * z = 1000) :
                                    y = 4 := 
sorry

end twenty_yuan_banknotes_count_l158_158372


namespace expression_pos_intervals_l158_158700

theorem expression_pos_intervals :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ∨ (x > 3) ↔ (x + 1) * (x - 1) * (x - 3) > 0 := by
  sorry

end expression_pos_intervals_l158_158700


namespace factor_polynomial_l158_158469

theorem factor_polynomial : 
  (x : ℝ) → (x^2 - 6 * x + 9 - 49 * x^4) = (-7 * x^2 + x - 3) * (7 * x^2 + x - 3) :=
by
  sorry

end factor_polynomial_l158_158469


namespace largest_tangential_quadrilaterals_l158_158737

-- Definitions and conditions
def convex_ngon {n : ℕ} (h : n ≥ 5) : Type := sorry -- Placeholder for defining a convex n-gon with ≥ 5 sides
def tangential_quadrilateral {n : ℕ} (h : n ≥ 5) (k : ℕ) : Prop := 
  -- Placeholder for the property that exactly k quadrilaterals out of all possible ones 
  -- in a convex n-gon have an inscribed circle
  sorry

theorem largest_tangential_quadrilaterals {n : ℕ} (h : n ≥ 5) : 
  ∃ k : ℕ, tangential_quadrilateral h k ∧ k = n / 2 :=
sorry

end largest_tangential_quadrilaterals_l158_158737


namespace pentagon_area_l158_158634

open Real

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 950 square units -/
theorem pentagon_area (a b c d e : ℝ) (h₁ : a = 18) (h₂ : b = 25) (h₃ : c = 30) (h₄ : d = 28) (h₅ : e = 25) : 
  ∃ (area : ℝ), area = 950 :=
by {
  sorry
}

end pentagon_area_l158_158634


namespace vector_subtraction_l158_158019

def a : ℝ × ℝ := (3, 5)
def b : ℝ × ℝ := (-2, 1)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)

theorem vector_subtraction : (a.1 - two_b.1, a.2 - two_b.2) = (7, 3) := by
  sorry

end vector_subtraction_l158_158019


namespace minimum_passed_l158_158544

def total_participants : Nat := 100
def num_questions : Nat := 10
def correct_answers : List Nat := [93, 90, 86, 91, 80, 83, 72, 75, 78, 59]
def passing_criteria : Nat := 6

theorem minimum_passed (total_participants : ℕ) (num_questions : ℕ) (correct_answers : List ℕ) (passing_criteria : ℕ) :
  100 = total_participants → 10 = num_questions → correct_answers = [93, 90, 86, 91, 80, 83, 72, 75, 78, 59] →
  passing_criteria = 6 → 
  ∃ p : ℕ, p = 62 := 
by
  sorry

end minimum_passed_l158_158544


namespace find_triangle_l158_158819

theorem find_triangle : ∀ (triangle : ℕ), (∀ (d : ℕ), 0 ≤ d ∧ d ≤ 9) → (5 * 3 + triangle = 12 * triangle + 4) → triangle = 1 :=
by
  sorry

end find_triangle_l158_158819


namespace min_sum_abs_l158_158520

theorem min_sum_abs (x : ℝ) : ∃ m, m = 4 ∧ ∀ x : ℝ, |x + 2| + |x - 2| + |x - 1| ≥ m := 
sorry

end min_sum_abs_l158_158520


namespace intersection_point_is_neg3_l158_158526

def f (x : ℝ) : ℝ := x^3 + 6 * x^2 + 9 * x + 15

theorem intersection_point_is_neg3 :
  ∃ a b : ℝ, (f a = b) ∧ (f b = a) ∧ (a, b) = (-3, -3) := sorry

end intersection_point_is_neg3_l158_158526


namespace gcd_condition_implies_equality_l158_158002

theorem gcd_condition_implies_equality (a b : ℤ) (h : ∀ n : ℤ, n ≥ 1 → Int.gcd (a + n) (b + n) > 1) : a = b :=
sorry

end gcd_condition_implies_equality_l158_158002


namespace trapezium_shorter_side_l158_158529

theorem trapezium_shorter_side (a b h : ℝ) (H1 : a = 10) (H2 : b = 18) (H3 : h = 10.00001) : a = 10 :=
by
  sorry

end trapezium_shorter_side_l158_158529


namespace find_avg_speed_l158_158268

variables (v t : ℝ)

noncomputable def avg_speed_cond := 
  (v + Real.sqrt 15) * (t - Real.pi / 4) = v * t

theorem find_avg_speed (h : avg_speed_cond v t) : v = Real.sqrt 15 :=
by
  sorry

end find_avg_speed_l158_158268


namespace find_number_l158_158185

-- Definitions from the conditions
def condition1 (x : ℝ) := 16 * x = 3408
def condition2 (x : ℝ) := 1.6 * x = 340.8

-- The statement to prove
theorem find_number (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x = 213 :=
by
  sorry

end find_number_l158_158185


namespace negate_proposition_l158_158408

theorem negate_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + x - 2 > 0) := 
sorry

end negate_proposition_l158_158408


namespace p_distance_300_l158_158495

-- Assume q's speed is v meters per second, and the race ends in a tie
variables (v : ℝ) (t : ℝ)
variable (d : ℝ)

-- Conditions
def q_speed : ℝ := v
def p_speed : ℝ := 1.25 * v
def q_distance : ℝ := d
def p_distance : ℝ := d + 60

-- Time equations
def q_time_eq : Prop := d = v * t
def p_time_eq : Prop := d + 60 = (1.25 * v) * t

-- Given the conditions, prove that p ran 300 meters in the race
theorem p_distance_300
  (v_pos : v > 0) 
  (t_pos : t > 0)
  (q_time : q_time_eq v d t)
  (p_time : p_time_eq v d t) :
  p_distance d = 300 :=
by
  sorry

end p_distance_300_l158_158495


namespace xy_inequality_l158_158670

theorem xy_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  x^2 * y^2 * (x^2 + y^2) ≤ 2 := 
sorry

end xy_inequality_l158_158670


namespace factorial_mod_13_l158_158552

open Nat

theorem factorial_mod_13 :
  let n := 10
  let p := 13
  n! % p = 6 := by
sorry

end factorial_mod_13_l158_158552


namespace find_values_l158_158034

theorem find_values (a b c : ℕ) 
    (h1 : a + b + c = 1024) 
    (h2 : c = b - 88) 
    (h3 : a = b + c) : 
    a = 712 ∧ b = 400 ∧ c = 312 :=
by {
    sorry
}

end find_values_l158_158034


namespace wedge_top_half_volume_l158_158876

theorem wedge_top_half_volume (r : ℝ) (C : ℝ) (V : ℝ) : 
  (C = 18 * π) ∧ (C = 2 * π * r) ∧ (V = (4/3) * π * r^3) ∧ 
  (V / 3 / 2) = 162 * π :=
  sorry

end wedge_top_half_volume_l158_158876


namespace find_a_value_l158_158728

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a_value :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a 0 + f a 1 = a) → a = 1 / 2 :=
by
  sorry

end find_a_value_l158_158728


namespace no_integral_value_2001_l158_158215

noncomputable def P (x : ℤ) : ℤ := sorry -- Polynomial definition needs to be filled in

theorem no_integral_value_2001 (a0 a1 a2 a3 a4 : ℤ) (x1 x2 x3 x4 : ℤ) :
  (P x1 = 2020) ∧ (P x2 = 2020) ∧ (P x3 = 2020) ∧ (P x4 = 2020) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  ¬ ∃ x : ℤ, P x = 2001 :=
sorry

end no_integral_value_2001_l158_158215


namespace proposition_does_not_hold_6_l158_158207

-- Define P as a proposition over positive integers
variable (P : ℕ → Prop)

-- Assumptions
variables (h1 : ∀ k : ℕ, P k → P (k + 1))  
variable (h2 : ¬ P 7)

-- Statement of the Problem
theorem proposition_does_not_hold_6 : ¬ P 6 :=
sorry

end proposition_does_not_hold_6_l158_158207


namespace intersection_S_T_eq_U_l158_158517

def S : Set ℝ := {x | abs x < 5}
def T : Set ℝ := {x | (x + 7) * (x - 3) < 0}
def U : Set ℝ := {x | -5 < x ∧ x < 3}

theorem intersection_S_T_eq_U : (S ∩ T) = U := 
by 
  sorry

end intersection_S_T_eq_U_l158_158517


namespace solution_l158_158253

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), (x > 0 ∧ y > 0) ∧ (6 * x^2 + 18 * x * y = 2 * x^3 + 3 * x^2 * y^2) ∧ x = (3 + Real.sqrt 153) / 4

theorem solution : problem_statement :=
by
  sorry

end solution_l158_158253


namespace price_of_20_percent_stock_l158_158074

theorem price_of_20_percent_stock (annual_income : ℝ) (investment : ℝ) (dividend_rate : ℝ) (price_of_stock : ℝ) :
  annual_income = 1000 →
  investment = 6800 →
  dividend_rate = 20 →
  price_of_stock = 136 :=
by
  intros h_income h_investment h_dividend_rate
  sorry

end price_of_20_percent_stock_l158_158074


namespace chlorine_discount_l158_158915

theorem chlorine_discount
  (cost_chlorine : ℕ)
  (cost_soap : ℕ)
  (num_chlorine : ℕ)
  (num_soap : ℕ)
  (discount_soap : ℤ)
  (total_savings : ℤ)
  (price_chlorine : ℤ)
  (price_soap_after_discount : ℤ)
  (total_price_before_discount : ℤ)
  (total_price_after_discount : ℤ)
  (goal_discount : ℤ) :
  cost_chlorine = 10 →
  cost_soap = 16 →
  num_chlorine = 3 →
  num_soap = 5 →
  discount_soap = 25 →
  total_savings = 26 →
  price_soap_after_discount = (1 - (discount_soap / 100)) * 16 →
  total_price_before_discount = (num_chlorine * cost_chlorine) + (num_soap * cost_soap) →
  total_price_after_discount = (num_chlorine * ((100 - goal_discount) / 100) * cost_chlorine) + (num_soap * 12) →
  total_price_before_discount - total_price_after_discount = total_savings →
  goal_discount = 20 :=
by
  intros
  sorry

end chlorine_discount_l158_158915


namespace sum_of_polynomials_l158_158533

-- Define the given polynomials f, g, and h
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- Prove that the sum of f(x), g(x), and h(x) is a specific polynomial
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := 
by {
  -- Proof is omitted
  sorry
}

end sum_of_polynomials_l158_158533


namespace num_dress_designs_l158_158258

-- Define the number of fabric colors and patterns
def fabric_colors : ℕ := 4
def patterns : ℕ := 5

-- Define the number of possible dress designs
def total_dress_designs : ℕ := fabric_colors * patterns

-- State the theorem that needs to be proved
theorem num_dress_designs : total_dress_designs = 20 := by
  sorry

end num_dress_designs_l158_158258


namespace y_intercept_of_line_l158_158966

theorem y_intercept_of_line (m x y b : ℝ) (h_slope : m = 4) (h_point : (x, y) = (199, 800)) (h_line : y = m * x + b) :
    b = 4 :=
by
  sorry

end y_intercept_of_line_l158_158966


namespace truth_values_l158_158082

-- Define the region D as a set
def D (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 ≤ 4

-- Define propositions p and q
def p : Prop := ∀ x y, D x y → 2 * x + y ≤ 8
def q : Prop := ∃ x y, D x y ∧ 2 * x + y ≤ -1

-- State the propositions to be proven
def prop1 : Prop := p ∨ q
def prop2 : Prop := ¬p ∨ q
def prop3 : Prop := p ∧ ¬q
def prop4 : Prop := ¬p ∧ ¬q

-- State the main theorem asserting the truth values of the propositions
theorem truth_values : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end truth_values_l158_158082


namespace normal_level_shortage_l158_158352

theorem normal_level_shortage
  (T : ℝ) (Normal_level : ℝ)
  (h1 : 0.75 * T = 30)
  (h2 : 30 = 2 * Normal_level) :
  T - Normal_level = 25 := 
by
  sorry

end normal_level_shortage_l158_158352


namespace right_triangles_product_hypotenuses_square_l158_158150

/-- 
Given two right triangles T₁ and T₂ with areas 2 and 8 respectively. 
The hypotenuse of T₁ is congruent to one leg of T₂.
The shorter leg of T₁ is congruent to the hypotenuse of T₂.
Prove that the square of the product of the lengths of their hypotenuses is 4624.
-/
theorem right_triangles_product_hypotenuses_square :
  ∃ x y z u : ℝ, 
    (1 / 2) * x * y = 2 ∧
    (1 / 2) * y * u = 8 ∧
    x^2 + y^2 = z^2 ∧
    y^2 + (16 / y)^2 = z^2 ∧ 
    (z^2)^2 = 4624 := 
sorry

end right_triangles_product_hypotenuses_square_l158_158150


namespace bread_count_at_end_of_day_l158_158167

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_count_at_end_of_day_l158_158167


namespace middle_pile_cards_l158_158489

theorem middle_pile_cards (x : Nat) (h : x ≥ 2) : 
    let left := x - 2
    let middle := x + 2
    let right := x
    let middle_after_step3 := middle + 1
    let final_middle := middle_after_step3 - left
    final_middle = 5 := 
by
  sorry

end middle_pile_cards_l158_158489


namespace inequality_abc_l158_158727

theorem inequality_abc (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a^2 + b^2 = 1/2) :
  (1 / (1 - a) + 1 / (1 - b) >= 4)
  ∧ ((1 / (1 - a) + 1 / (1 - b) = 4) ↔ (a = 1/2 ∧ b = 1/2)) :=
by
  sorry

end inequality_abc_l158_158727


namespace rectangle_inscribed_circle_hypotenuse_l158_158843

open Real

theorem rectangle_inscribed_circle_hypotenuse
  (AB BC : ℝ)
  (h_AB : AB = 20)
  (h_BC : BC = 10)
  (r : ℝ)
  (h_r : r = 10 / 3) :
  sqrt ((AB - 2 * r) ^ 2 + BC ^ 2) = 50 / 3 :=
by {
  sorry
}

end rectangle_inscribed_circle_hypotenuse_l158_158843


namespace cube_surface_area_calc_l158_158152

-- Edge length of the cube
def edge_length : ℝ := 7

-- Definition of the surface area formula for a cube
def surface_area (a : ℝ) : ℝ := 6 * (a ^ 2)

-- The main theorem stating the surface area of the cube with given edge length
theorem cube_surface_area_calc : surface_area edge_length = 294 :=
by
  sorry

end cube_surface_area_calc_l158_158152


namespace larger_number_is_38_l158_158687

theorem larger_number_is_38 (x y : ℕ) (h1 : x + y = 64) (h2 : y = x + 12) : y = 38 :=
by
  sorry

end larger_number_is_38_l158_158687


namespace equilateral_triangle_area_with_inscribed_circle_l158_158603

theorem equilateral_triangle_area_with_inscribed_circle
  (r : ℝ) (area_circle : ℝ) (area_triangle : ℝ) 
  (h_inscribed_circle_area : area_circle = 9 * Real.pi)
  (h_radius : r = 3) :
  area_triangle = 27 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_with_inscribed_circle_l158_158603


namespace shaded_trapezium_area_l158_158362

theorem shaded_trapezium_area :
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  area = 55 / 4 :=
by
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  show area = 55 / 4
  sorry

end shaded_trapezium_area_l158_158362


namespace ninth_grade_class_notification_l158_158278

theorem ninth_grade_class_notification (n : ℕ) (h1 : 1 + n + n * n = 43) : n = 6 :=
by
  sorry

end ninth_grade_class_notification_l158_158278


namespace sin_cos_double_angle_identity_l158_158064

theorem sin_cos_double_angle_identity (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioc (π/2) π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_cos_double_angle_identity_l158_158064


namespace regions_formed_l158_158192

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l158_158192


namespace remainder_when_divided_by_13_l158_158329

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) : (N = 39 * k + 17) → (N % 13 = 4) := by
  sorry

end remainder_when_divided_by_13_l158_158329


namespace exponent_division_l158_158373

-- We need to reformulate the given condition into Lean definitions
def twenty_seven_is_three_cubed : Prop := 27 = 3^3

-- Using the condition to state the problem
theorem exponent_division (h : twenty_seven_is_three_cubed) : 
  3^15 / 27^3 = 729 :=
by
  sorry

end exponent_division_l158_158373


namespace prob_male_given_obese_correct_l158_158725

-- Definitions based on conditions
def ratio_male_female : ℚ := 3 / 2
def prob_obese_male : ℚ := 1 / 5
def prob_obese_female : ℚ := 1 / 10

-- Definition of events
def total_employees : ℚ := ratio_male_female + 1

-- Probability calculations
def prob_male : ℚ := ratio_male_female / total_employees
def prob_female : ℚ := 1 / total_employees

def prob_obese_and_male : ℚ := prob_male * prob_obese_male
def prob_obese_and_female : ℚ := prob_female * prob_obese_female

def prob_obese : ℚ := prob_obese_and_male + prob_obese_and_female

def prob_male_given_obese : ℚ := prob_obese_and_male / prob_obese

-- Theorem statement
theorem prob_male_given_obese_correct : prob_male_given_obese = 3 / 4 := sorry

end prob_male_given_obese_correct_l158_158725


namespace fort_blocks_count_l158_158471

noncomputable def volume_of_blocks (l w h : ℕ) (wall_thickness floor_thickness top_layer_volume : ℕ) : ℕ :=
  let interior_length := l - 2 * wall_thickness
  let interior_width := w - 2 * wall_thickness
  let interior_height := h - floor_thickness
  let volume_original := l * w * h
  let volume_interior := interior_length * interior_width * interior_height
  volume_original - volume_interior + top_layer_volume

theorem fort_blocks_count : volume_of_blocks 15 12 7 2 1 180 = 912 :=
by
  sorry

end fort_blocks_count_l158_158471


namespace solve_congruences_l158_158114

theorem solve_congruences :
  ∃ x : ℤ, 
  x ≡ 3 [ZMOD 7] ∧ 
  x^2 ≡ 44 [ZMOD 49] ∧ 
  x^3 ≡ 111 [ZMOD 343] ∧ 
  x ≡ 17 [ZMOD 343] :=
sorry

end solve_congruences_l158_158114


namespace problem_statement_l158_158730

-- Define the conditions and the goal
theorem problem_statement {x y : ℝ} 
  (h1 : (x + y)^2 = 36)
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := 
by
  sorry

end problem_statement_l158_158730


namespace cost_of_pen_is_51_l158_158355

-- Definitions of variables and conditions
variables {p q : ℕ}
variables (h1 : 6 * p + 2 * q = 348)
variables (h2 : 3 * p + 4 * q = 234)

-- Goal: Prove the cost of a pen (p) is 51 cents
theorem cost_of_pen_is_51 : p = 51 :=
by
  -- placeholder for the proof
  sorry

end cost_of_pen_is_51_l158_158355


namespace find_n_for_positive_root_l158_158605

theorem find_n_for_positive_root :
  ∃ x : ℝ, x > 0 ∧ (∃ n : ℝ, (n / (x - 1) + 2 / (1 - x) = 1)) ↔ n = 2 :=
by
  sorry

end find_n_for_positive_root_l158_158605


namespace local_minimum_condition_l158_158245

-- Define the function f(x)
def f (x b : ℝ) : ℝ := x ^ 3 - 3 * b * x + 3 * b

-- Define the first derivative of f(x)
def f_prime (x b : ℝ) : ℝ := 3 * x ^ 2 - 3 * b

-- Define the second derivative of f(x)
def f_double_prime (x b : ℝ) : ℝ := 6 * x

-- Theorem stating that f(x) has a local minimum if and only if b > 0
theorem local_minimum_condition (b : ℝ) (x : ℝ) (h : f_prime x b = 0) : f_double_prime x b > 0 ↔ b > 0 :=
by sorry

end local_minimum_condition_l158_158245


namespace part_I_part_II_l158_158969

noncomputable def curve_M (theta : ℝ) : ℝ := 4 * Real.cos theta

noncomputable def line_l (t m alpha : ℝ) : ℝ × ℝ :=
  let x := m + t * Real.cos alpha
  let y := t * Real.sin alpha
  (x, y)

theorem part_I (varphi : ℝ) :
  let OB := curve_M (varphi + π / 4)
  let OC := curve_M (varphi - π / 4)
  let OA := curve_M varphi
  OB + OC = Real.sqrt 2 * OA := by
  sorry

theorem part_II (m alpha : ℝ) :
  let varphi := π / 12
  let B := (1, Real.sqrt 3)
  let C := (3, -Real.sqrt 3)
  exists t1 t2, line_l t1 m alpha = B ∧ line_l t2 m alpha = C :=
  have hα : alpha = 2 * π / 3 := by sorry
  have hm : m = 2 := by sorry
  sorry

end part_I_part_II_l158_158969


namespace tricia_age_l158_158872

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l158_158872


namespace function_inverse_overlap_form_l158_158294

theorem function_inverse_overlap_form (a b c d : ℝ) (h : ¬(a = 0 ∧ c = 0)) : 
  (∀ x, (c * x + d) * (dx - b) = (a * x + b) * (-c * x + a)) → 
  (∃ f : ℝ → ℝ, (∀ x, f x = x ∨ f x = (a * x + b) / (c * x - a))) :=
by 
  sorry

end function_inverse_overlap_form_l158_158294


namespace bus_speed_including_stoppages_l158_158532

theorem bus_speed_including_stoppages 
  (speed_excl_stoppages : ℚ) 
  (ten_minutes_per_hour : ℚ) 
  (bus_stops_for_10_minutes : ten_minutes_per_hour = 10/60) 
  (speed_is_54_kmph : speed_excl_stoppages = 54) : 
  (speed_excl_stoppages * (1 - ten_minutes_per_hour)) = 45 := 
by 
  sorry

end bus_speed_including_stoppages_l158_158532


namespace case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l158_158151

-- Conditions for Case (a)
def corner_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is a corner cell
  sorry

theorem case_a_second_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  corner_cell board starting_cell → 
  player = 2 :=
by
  sorry
  
-- Conditions for Case (b)
def initial_setup_according_to_figure (board : Type) (starting_cell : board) : Prop :=
  -- definition to determine if a cell setup matches the figure
  sorry

theorem case_b_first_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  initial_setup_according_to_figure board starting_cell → 
  player = 1 :=
by
  sorry

-- Conditions for Case (c)
def black_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is black
  sorry

theorem case_c_winner_based_on_cell_color (board : Type) (starting_cell : board) (player : ℕ) :
  (black_cell board starting_cell → player = 1) ∧ (¬ black_cell board starting_cell → player = 2) :=
by
  sorry
  
-- Conditions for Case (d)
def same_starting_cell_two_games (board : Type) (starting_cell : board) : Prop :=
  -- definition for same starting cell but different outcomes in games
  sorry

theorem case_d_examples (board : Type) (starting_cell : board) (player1 player2 : ℕ) :
  (same_starting_cell_two_games board starting_cell → (player1 = 1 ∧ player2 = 2)) ∨ 
  (same_starting_cell_two_games board starting_cell → (player1 = 2 ∧ player2 = 1)) :=
by
  sorry

end case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l158_158151


namespace maryann_work_time_l158_158261

variables (C A R : ℕ)

theorem maryann_work_time
  (h1 : A = 2 * C)
  (h2 : R = 6 * C)
  (h3 : C + A + R = 1440) :
  C = 160 ∧ A = 320 ∧ R = 960 :=
by
  sorry

end maryann_work_time_l158_158261


namespace dress_designs_count_l158_158318

theorem dress_designs_count :
  let colors := 5
  let patterns := 4
  let sizes := 3
  colors * patterns * sizes = 60 :=
by
  let colors := 5
  let patterns := 4
  let sizes := 3
  have h : colors * patterns * sizes = 60 := by norm_num
  exact h

end dress_designs_count_l158_158318


namespace prime_divides_product_of_divisors_l158_158883

theorem prime_divides_product_of_divisors (p : ℕ) (n : ℕ) (a : Fin n → ℕ) 
(Hp : Nat.Prime p) (Hdiv : p ∣ (Finset.univ.prod a)) : 
∃ i : Fin n, p ∣ a i :=
sorry

end prime_divides_product_of_divisors_l158_158883


namespace larry_stickers_l158_158911

theorem larry_stickers (initial_stickers : ℕ) (lost_stickers : ℕ) (final_stickers : ℕ) 
  (initial_eq_93 : initial_stickers = 93) 
  (lost_eq_6 : lost_stickers = 6) 
  (final_eq : final_stickers = initial_stickers - lost_stickers) : 
  final_stickers = 87 := 
  by 
  -- proof goes here
  sorry

end larry_stickers_l158_158911


namespace inequality_relations_l158_158804

noncomputable def a : ℝ := Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 125 ^ (1 / 6)
noncomputable def c : ℝ := Real.log 7 / Real.log (1 / 6)

theorem inequality_relations :
  c < a ∧ a < b := 
by 
  sorry

end inequality_relations_l158_158804


namespace transformations_map_figure_l158_158830

noncomputable def count_transformations : ℕ := sorry

theorem transformations_map_figure :
  count_transformations = 3 :=
sorry

end transformations_map_figure_l158_158830


namespace field_area_l158_158834

theorem field_area (L W : ℝ) (hL : L = 20) (h_fencing : 2 * W + L = 59) :
  L * W = 390 :=
by {
  -- We will skip the proof
  sorry
}

end field_area_l158_158834


namespace sin_double_angle_l158_158107

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α such that its terminal side passes through point P
noncomputable def α : ℝ := sorry -- The exact definition of α is not needed for this statement

-- Define r as the distance from the origin to the point P
noncomputable def r : ℝ := Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

-- Define sin(α) and cos(α)
noncomputable def sin_α : ℝ := P.2 / r
noncomputable def cos_α : ℝ := P.1 / r

-- The proof statement
theorem sin_double_angle : 2 * sin_α * cos_α = -4 / 5 := by
  sorry

end sin_double_angle_l158_158107


namespace gcf_60_90_l158_158828

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l158_158828


namespace find_other_number_l158_158364

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 192) (h_hcf : Nat.gcd A B = 16) (h_A : A = 48) : B = 64 :=
by
  sorry

end find_other_number_l158_158364


namespace radicals_like_simplest_forms_l158_158866

theorem radicals_like_simplest_forms (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a = b + 2) :
  a = 3 ∧ b = 1 :=
by
  sorry

end radicals_like_simplest_forms_l158_158866


namespace ratio_of_other_triangle_l158_158110

noncomputable def ratioAreaOtherTriangle (m : ℝ) : ℝ := 1 / (4 * m)

theorem ratio_of_other_triangle (m : ℝ) (h : m > 0) : ratioAreaOtherTriangle m = 1 / (4 * m) :=
by
  -- Proof will be provided here
  sorry

end ratio_of_other_triangle_l158_158110


namespace log_three_twenty_seven_sqrt_three_l158_158675

noncomputable def twenty_seven : ℝ := 27
noncomputable def sqrt_three : ℝ := Real.sqrt 3

theorem log_three_twenty_seven_sqrt_three :
  Real.logb 3 (twenty_seven * sqrt_three) = 7 / 2 :=
by
  sorry -- Proof omitted

end log_three_twenty_seven_sqrt_three_l158_158675


namespace negation_of_exactly_one_is_even_l158_158227

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_is_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
   (¬ is_even a ∧ ¬ is_even b ∧ is_even c))

def at_least_two_even (a b c : ℕ) : Prop :=
  ((is_even a ∧ is_even b) ∨ (is_even b ∧ is_even c) ∨ (is_even a ∧ is_even c))

def all_are_odd (a b c : ℕ) : Prop := ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c 

theorem negation_of_exactly_one_is_even (a b c : ℕ) :
  ¬ exactly_one_is_even a b c ↔ at_least_two_even a b c ∨ all_are_odd a b c := by
  sorry

end negation_of_exactly_one_is_even_l158_158227


namespace travel_same_direction_time_l158_158090

variable (A B : Type) [MetricSpace A] (downstream_speed upstream_speed : ℝ)
  (H_A_downstream_speed : downstream_speed = 8)
  (H_A_upstream_speed : upstream_speed = 4)
  (H_B_downstream_speed : downstream_speed = 8)
  (H_B_upstream_speed : upstream_speed = 4)
  (H_equal_travel_time : (∃ x : ℝ, x * downstream_speed + (3 - x) * upstream_speed = 3)
                      ∧ (∃ x : ℝ, x * upstream_speed + (3 - x) * downstream_speed = 3))

theorem travel_same_direction_time (A_α_downstream B_β_upstream A_α_upstream B_β_downstream : ℝ)
  (H_travel_time : (∃ x : ℝ, x = 1) ∧ (A_α_upstream = 3 - A_α_downstream) ∧ (B_β_downstream = 3 - B_β_upstream)) :
  A_α_downstream = 1 → A_α_upstream = 3 - 1 → B_β_downstream = 1 → B_β_upstream = 3 - 1 → ∃ t, t = 1 :=
by
  sorry

end travel_same_direction_time_l158_158090


namespace casey_nail_decorating_time_l158_158179

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l158_158179


namespace vector_subtraction_l158_158319

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Statement we want to prove: 2a - b = (-1, 6)
theorem vector_subtraction : 2 • a - b = (-1, 6) := by
  sorry

end vector_subtraction_l158_158319


namespace circle_passing_through_pole_l158_158710

noncomputable def equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * Real.cos θ

theorem circle_passing_through_pole :
  equation_of_circle 2 θ := 
sorry

end circle_passing_through_pole_l158_158710


namespace unique_two_digit_perfect_square_divisible_by_5_l158_158052

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

-- The statement to prove: there is exactly 1 two-digit perfect square that is divisible by 5
theorem unique_two_digit_perfect_square_divisible_by_5 :
  ∃! n : ℕ, is_perfect_square n ∧ two_digit n ∧ divisible_by_5 n :=
sorry

end unique_two_digit_perfect_square_divisible_by_5_l158_158052


namespace function_domain_length_correct_l158_158264

noncomputable def function_domain_length : ℕ :=
  let p : ℕ := 240 
  let q : ℕ := 1
  p + q

theorem function_domain_length_correct : function_domain_length = 241 := by
  sorry

end function_domain_length_correct_l158_158264


namespace sahil_machine_purchase_price_l158_158525

theorem sahil_machine_purchase_price
  (repair_cost : ℕ)
  (transportation_cost : ℕ)
  (selling_price : ℕ)
  (profit_percent : ℤ)
  (purchase_price : ℕ)
  (total_cost : ℕ)
  (profit_ratio : ℚ)
  (h1 : repair_cost = 5000)
  (h2 : transportation_cost = 1000)
  (h3 : selling_price = 30000)
  (h4 : profit_percent = 50)
  (h5 : total_cost = purchase_price + repair_cost + transportation_cost)
  (h6 : profit_ratio = profit_percent / 100)
  (h7 : selling_price = (1 + profit_ratio) * total_cost) :
  purchase_price = 14000 :=
by
  sorry

end sahil_machine_purchase_price_l158_158525


namespace octagon_area_equals_eight_one_plus_sqrt_two_l158_158875

theorem octagon_area_equals_eight_one_plus_sqrt_two
  (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a ^ 2 = 16) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2) :=
by
  sorry

end octagon_area_equals_eight_one_plus_sqrt_two_l158_158875


namespace hyperbola_asymptote_eq_l158_158916

-- Define the given hyperbola equation and its asymptote
def hyperbola_eq (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) - (y^2 / 4) = 1

def asymptote_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, y = (1/2) * x

-- State the main theorem
theorem hyperbola_asymptote_eq :
  (∃ a : ℝ, hyperbola_eq a ∧ asymptote_eq a) →
  (∃ x y : ℝ, (x^2 / 16) - (y^2 / 4) = 1) := 
by
  sorry

end hyperbola_asymptote_eq_l158_158916


namespace problem_statement_l158_158722

theorem problem_statement (
  a b c d x y z t : ℝ
) (habcd : 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1) 
  (hxyz : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 1 ≤ t)
  (h_sum : a + b + c + d + x + y + z + t = 8) :
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := 
sorry

end problem_statement_l158_158722


namespace cosine_product_l158_158935

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l158_158935


namespace solve_for_a_l158_158640

open Complex

noncomputable def question (a : ℝ) : Prop :=
  ∃ z : ℂ, z = (a + I) / (1 - I) ∧ z.im ≠ 0 ∧ z.re = 0

theorem solve_for_a (a : ℝ) (h : question a) : a = 1 :=
sorry

end solve_for_a_l158_158640


namespace sum_of_three_numbers_eq_16_l158_158519

variable {a b c : ℝ}

theorem sum_of_three_numbers_eq_16
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_eq_16_l158_158519


namespace modulus_of_complex_z_l158_158853

open Complex

theorem modulus_of_complex_z (z : ℂ) (h : z * (2 - 3 * I) = 6 + 4 * I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 :=
by
  sorry

end modulus_of_complex_z_l158_158853


namespace sqrt_of_9_eq_3_l158_158551

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := by
  sorry

end sqrt_of_9_eq_3_l158_158551


namespace find_number_l158_158765

theorem find_number (x : ℤ) (h : 16 * x = 32) : x = 2 :=
sorry

end find_number_l158_158765


namespace sally_spent_total_l158_158402

section SallySpending

def peaches : ℝ := 12.32
def cherries : ℝ := 11.54
def total_spent : ℝ := peaches + cherries

theorem sally_spent_total :
  total_spent = 23.86 := by
  sorry

end SallySpending

end sally_spent_total_l158_158402


namespace eggs_in_basket_l158_158934

theorem eggs_in_basket (x : ℕ) (h₁ : 600 / x + 1 = 600 / (x - 20)) : x = 120 :=
sorry

end eggs_in_basket_l158_158934


namespace simplify_and_evaluate_l158_158694

-- Definitions and conditions 
def x := ℝ
def given_condition (x: ℝ) : Prop := x + 2 = Real.sqrt 2

-- The problem statement translated into Lean 4
theorem simplify_and_evaluate (x: ℝ) (h: given_condition x) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3 * x)) = Real.sqrt 2 - 1 :=
sorry

end simplify_and_evaluate_l158_158694


namespace length_of_faster_train_l158_158357

/-- 
Let the faster train have a speed of 144 km per hour, the slower train a speed of 
72 km per hour, and the time taken for the faster train to cross a man in the 
slower train be 19 seconds. Then the length of the faster train is 380 meters.
-/
theorem length_of_faster_train 
  (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_to_cross : ℝ)
  (h_speed_faster_train : speed_faster_train = 144) 
  (h_speed_slower_train : speed_slower_train = 72) 
  (h_time_to_cross : time_to_cross = 19) :
  (speed_faster_train - speed_slower_train) * (5 / 18) * time_to_cross = 380 :=
by
  sorry

end length_of_faster_train_l158_158357


namespace Hezekiah_age_l158_158531

variable (H : ℕ)
variable (R : ℕ) -- Ryanne's age

-- Defining the conditions
def condition1 : Prop := R = H + 7
def condition2 : Prop := H + R = 15

-- The main theorem we want to prove
theorem Hezekiah_age : condition1 H R → condition2 H R → H = 4 :=
by  -- proof will be here
  sorry

end Hezekiah_age_l158_158531


namespace joe_total_time_to_school_l158_158246

theorem joe_total_time_to_school:
  ∀ (d r_w: ℝ), (1 / 3) * d = r_w * 9 →
                  4 * r_w * (2 * (r_w * 9) / (3 * (4 * r_w))) = (2 / 3) * d →
                  (1 / 3) * d / r_w + (2 / 3) * d / (4 * r_w) = 13.5 :=
by
  intros d r_w h1 h2
  sorry

end joe_total_time_to_school_l158_158246


namespace circle_equation_l158_158409

-- Defining the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - x = 0

-- Defining the point
def point : ℝ × ℝ := (1, -1)

-- Proving the equation of the new circle that passes through the intersection points 
-- of the given circles and the given point
theorem circle_equation (x y : ℝ) :
  (circle1 x y ∧ circle2 x y ∧ x = 1 ∧ y = -1) → 9 * x^2 + 9 * y^2 - 14 * x + 4 * y = 0 :=
sorry

end circle_equation_l158_158409


namespace carbonated_water_solution_l158_158982

variable (V V_1 V_2 : ℝ)
variable (C2 : ℝ)

def carbonated_water_percent (V V1 V2 C2 : ℝ) : Prop :=
  0.8 * V1 + C2 * V2 = 0.6 * V

theorem carbonated_water_solution :
  ∀ (V : ℝ),
  (V1 = 0.1999999999999997 * V) →
  (V2 = 0.8000000000000003 * V) →
  carbonated_water_percent V V1 V2 C2 →
  C2 = 0.55 :=
by
  intros V V1_eq V2_eq carbonated_eq
  sorry

end carbonated_water_solution_l158_158982


namespace min_sum_of_dimensions_l158_158457

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a * b * c = 1645) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  a + b + c ≥ 129 :=
sorry

end min_sum_of_dimensions_l158_158457


namespace intersection_of_M_and_N_l158_158979

-- Define the sets M and N with the given conditions
def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem that the intersection of M and N is as described
theorem intersection_of_M_and_N : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 1} :=
by
  -- the proof will go here
  sorry

end intersection_of_M_and_N_l158_158979


namespace expected_value_of_win_is_correct_l158_158097

noncomputable def expected_value_of_win : ℝ :=
  (1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) +
  (1 / 8) * (8 - 4) + (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) +
  (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8)

theorem expected_value_of_win_is_correct :
  expected_value_of_win = 3.5 :=
by
  sorry

end expected_value_of_win_is_correct_l158_158097


namespace percent_females_employed_l158_158176

noncomputable def employed_percent (population: ℕ) : ℚ := 0.60
noncomputable def employed_males_percent (population: ℕ) : ℚ := 0.48

theorem percent_females_employed (population: ℕ) : ((employed_percent population) - (employed_males_percent population)) / (employed_percent population) = 0.20 :=
by
  sorry

end percent_females_employed_l158_158176


namespace jello_cost_l158_158745

def cost_to_fill_tub_with_jello (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : ℕ :=
  water_volume_cubic_feet * gallons_per_cubic_foot * pounds_per_gallon * tablespoons_per_pound * cost_per_tablespoon

theorem jello_cost (water_volume_cubic_feet : ℕ) (gallons_per_cubic_foot : ℕ) 
    (pounds_per_gallon : ℕ) (tablespoons_per_pound : ℕ) (cost_per_tablespoon : ℕ) : 
    water_volume_cubic_feet = 6 ∧ gallons_per_cubic_foot = 7 ∧ pounds_per_gallon = 8 ∧ 
    tablespoons_per_pound = 1 ∧ cost_per_tablespoon = 1 →
    cost_to_fill_tub_with_jello water_volume_cubic_feet gallons_per_cubic_foot pounds_per_gallon tablespoons_per_pound cost_per_tablespoon = 270 :=
  by 
    sorry

end jello_cost_l158_158745


namespace solve_equation_naturals_l158_158163

theorem solve_equation_naturals :
  ∀ (X Y Z : ℕ), X^Y + Y^Z = X * Y * Z ↔ 
    (X = 1 ∧ Y = 1 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 4) := 
by
  sorry

end solve_equation_naturals_l158_158163


namespace sum_of_possible_values_of_x_l158_158029

-- Define the concept of an isosceles triangle with specific angles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

-- Define the angle sum property of a triangle
def angle_sum_property (a b c : ℝ) : Prop := 
  a + b + c = 180

-- State the problem using the given conditions and the required proof
theorem sum_of_possible_values_of_x :
  ∀ (x : ℝ), 
    is_isosceles_triangle 70 70 x ∨
    is_isosceles_triangle 70 x x ∨
    is_isosceles_triangle x 70 70 →
    angle_sum_property 70 70 x →
    angle_sum_property 70 x x →
    angle_sum_property x 70 70 →
    (70 + 55 + 40) = 165 :=
  by
    sorry

end sum_of_possible_values_of_x_l158_158029


namespace dad_real_age_l158_158801

theorem dad_real_age (x : ℝ) (h : (5/7) * x = 35) : x = 49 :=
by
  sorry

end dad_real_age_l158_158801


namespace find_interest_rate_l158_158620

-- Define the given conditions
variables (P A t n CI : ℝ) (r : ℝ)

-- Suppose given conditions
variables (hP : P = 1200)
variables (hCI : CI = 240)
variables (hA : A = P + CI)
variables (ht : t = 1)
variables (hn : n = 1)

-- Define the statement to prove 
theorem find_interest_rate : (A = P * (1 + r / n)^(n * t)) → (r = 0.2) :=
by
  sorry

end find_interest_rate_l158_158620


namespace arithmetic_sequence_common_difference_l158_158270

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Statement of the problem
theorem arithmetic_sequence_common_difference
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h_arith : ∀ n, a (n+1) = a n + d) :
  d = 5 := by
  sorry

end arithmetic_sequence_common_difference_l158_158270


namespace six_star_three_l158_158252

def binary_op (x y : ℕ) : ℕ := 4 * x + 5 * y - x * y

theorem six_star_three : binary_op 6 3 = 21 := by
  sorry

end six_star_three_l158_158252


namespace angle_between_north_and_south_southeast_l158_158791

-- Given a circular floor pattern with 12 equally spaced rays
def num_rays : ℕ := 12
def total_degrees : ℕ := 360

-- Proving each central angle measure
def central_angle_measure : ℕ := total_degrees / num_rays

-- Define rays of interest
def segments_between_rays : ℕ := 5

-- Prove the angle between the rays pointing due North and South-Southeast
theorem angle_between_north_and_south_southeast :
  (segments_between_rays * central_angle_measure) = 150 := by
  sorry

end angle_between_north_and_south_southeast_l158_158791


namespace lena_nicole_candy_difference_l158_158466

variables (L K N : ℕ)

theorem lena_nicole_candy_difference
  (hL : L = 16)
  (hLK : L + 5 = 3 * K)
  (hKN : K = N - 4) :
  L - N = 5 :=
sorry

end lena_nicole_candy_difference_l158_158466


namespace total_height_of_buildings_l158_158350

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end total_height_of_buildings_l158_158350


namespace seventy_five_percent_of_number_l158_158483

variable (N : ℝ)

theorem seventy_five_percent_of_number :
  (1 / 8) * (3 / 5) * (4 / 7) * (5 / 11) * N - (1 / 9) * (2 / 3) * (3 / 4) * (5 / 8) * N = 30 →
  0.75 * N = -1476 :=
by
  sorry

end seventy_five_percent_of_number_l158_158483


namespace seats_not_occupied_l158_158541

theorem seats_not_occupied (seats_per_row : ℕ) (rows : ℕ) (fraction_allowed : ℚ) (total_seats : ℕ) (allowed_seats_per_row : ℕ) (allowed_total : ℕ) (unoccupied_seats : ℕ) :
  seats_per_row = 8 →
  rows = 12 →
  fraction_allowed = 3 / 4 →
  total_seats = seats_per_row * rows →
  allowed_seats_per_row = seats_per_row * fraction_allowed →
  allowed_total = allowed_seats_per_row * rows →
  unoccupied_seats = total_seats - allowed_total →
  unoccupied_seats = 24 :=
by sorry

end seats_not_occupied_l158_158541


namespace sum_of_powers_of_i_l158_158370

noncomputable def i : Complex := Complex.I

theorem sum_of_powers_of_i :
  (Finset.range 2011).sum (λ n => i^(n+1)) = -1 := by
  sorry

end sum_of_powers_of_i_l158_158370


namespace hotel_floors_l158_158474

/-- Given:
  - Each floor has 10 identical rooms.
  - The last floor is unavailable for guests.
  - Hans could be checked into 90 different rooms.
  - There are no other guests.
 - Prove that the total number of floors in the hotel is 10.
--/
theorem hotel_floors :
  (∃ n : ℕ, n ≥ 1 ∧ 10 * (n - 1) = 90) → n = 10 :=
by 
  sorry

end hotel_floors_l158_158474


namespace Julio_fish_catch_rate_l158_158309

theorem Julio_fish_catch_rate (F : ℕ) : 
  (9 * F) - 15 = 48 → F = 7 :=
by
  intro h1
  --- proof
  sorry

end Julio_fish_catch_rate_l158_158309


namespace at_least_one_not_less_than_two_l158_158312

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ 2 ≤ x :=
by
  sorry

end at_least_one_not_less_than_two_l158_158312


namespace scallops_per_pound_l158_158849

theorem scallops_per_pound
  (cost_per_pound : ℝ)
  (scallops_per_person : ℕ)
  (number_of_people : ℕ)
  (total_cost : ℝ)
  (total_scallops : ℕ)
  (total_pounds : ℝ)
  (scallops_per_pound : ℕ)
  (h1 : cost_per_pound = 24)
  (h2 : scallops_per_person = 2)
  (h3 : number_of_people = 8)
  (h4 : total_cost = 48)
  (h5 : total_scallops = scallops_per_person * number_of_people)
  (h6 : total_pounds = total_cost / cost_per_pound)
  (h7 : scallops_per_pound = total_scallops / total_pounds) : 
  scallops_per_pound = 8 :=
sorry

end scallops_per_pound_l158_158849


namespace find_f_4_l158_158247

-- Lean code to encapsulate the conditions and the goal
theorem find_f_4 (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x * f y = y * f x)
  (h2 : f 12 = 24) : 
  f 4 = 8 :=
sorry

end find_f_4_l158_158247


namespace diagonals_in_decagon_l158_158172

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_decagon : number_of_diagonals 10 = 35 := by
  sorry

end diagonals_in_decagon_l158_158172


namespace perimeter_of_structure_l158_158480

noncomputable def structure_area : ℝ := 576
noncomputable def num_squares : ℕ := 9
noncomputable def square_area : ℝ := structure_area / num_squares
noncomputable def side_length : ℝ := Real.sqrt square_area
noncomputable def perimeter (side_length : ℝ) : ℝ := 8 * side_length

theorem perimeter_of_structure : perimeter side_length = 64 := by
  -- proof will follow here
  sorry

end perimeter_of_structure_l158_158480


namespace evaluate_f_at_minus_2_l158_158513

def f (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_f_at_minus_2 : f (-2) = 7 / 3 := by
  -- Proof is omitted
  sorry

end evaluate_f_at_minus_2_l158_158513


namespace power_mod_l158_158197

theorem power_mod (x n m : ℕ) : (x^n) % m = x % m := by 
  sorry

example : 5^2023 % 150 = 5 % 150 :=
by exact power_mod 5 2023 150

end power_mod_l158_158197


namespace min_sqrt_eq_sum_sqrt_implies_param_l158_158813

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem min_sqrt_eq_sum_sqrt_implies_param (a b c : ℝ) (r s t : ℝ)
    (h1 : 0 < a ∧ a ≤ 1)
    (h2 : 0 < b ∧ b ≤ 1)
    (h3 : 0 < c ∧ c ≤ 1)
    (h4 : min (sqrt ((a * b + 1) / (a * b * c))) (min (sqrt ((b * c + 1) / (a * b * c))) (sqrt ((a * c + 1) / (a * b * c)))) 
          = (sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c))) :
    ∃ r, a = 1 / (1 + r^2) ∧ b = 1 / (1 + (1 / r^2)) ∧ c = (r + 1 / r)^2 / (1 + (r + 1 / r)^2) :=
sorry

end min_sqrt_eq_sum_sqrt_implies_param_l158_158813


namespace cube_volume_from_surface_area_l158_158871

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l158_158871


namespace distinct_integer_roots_iff_l158_158877

theorem distinct_integer_roots_iff (a : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ a = -2 ∨ a = 18 :=
by
  sorry

end distinct_integer_roots_iff_l158_158877


namespace range_of_a_l158_158121

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic a x > 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l158_158121


namespace find_a2018_l158_158523

-- Given Conditions
def initial_condition (a : ℕ → ℤ) : Prop :=
  a 1 = -1

def absolute_difference (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → abs (a n - a (n-1)) = 2^(n-1)

def subseq_decreasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n-1) > a (2*(n+1)-1)

def subseq_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n) < a (2*(n+1))

-- Theorem to Prove
theorem find_a2018 (a : ℕ → ℤ)
  (h1 : initial_condition a)
  (h2 : absolute_difference a)
  (h3 : subseq_decreasing a)
  (h4 : subseq_increasing a) :
  a 2018 = (2^2018 - 1) / 3 :=
sorry

end find_a2018_l158_158523


namespace simplify_fraction_l158_158313

theorem simplify_fraction : (270 / 18) * (7 / 140) * (9 / 4) = 27 / 16 :=
by sorry

end simplify_fraction_l158_158313


namespace doughnut_problem_l158_158183

theorem doughnut_problem :
  ∀ (total_doughnuts first_two_box_doughnuts boxes : ℕ),
  total_doughnuts = 72 →
  first_two_box_doughnuts = 12 →
  boxes = 4 →
  (total_doughnuts - 2 * first_two_box_doughnuts) / boxes = 12 :=
by
  intros total_doughnuts first_two_box_doughnuts boxes ht12 hb12 b4
  sorry

end doughnut_problem_l158_158183


namespace math_problem_l158_158041

-- Definitions of the conditions
variable (x y : ℝ)
axiom h1 : x + y = 5
axiom h2 : x * y = 3

-- Prove the desired equality
theorem math_problem : x + (x^4 / y^3) + (y^4 / x^3) + y = 1021 := 
by 
sorry

end math_problem_l158_158041


namespace find_speed_in_second_hour_l158_158009

-- Define the given conditions as hypotheses
def speed_in_first_hour : ℝ := 50
def average_speed : ℝ := 55
def total_time : ℝ := 2

-- Define a function that represents the speed in the second hour
def speed_second_hour (s2 : ℝ) := 
  (speed_in_first_hour + s2) / total_time = average_speed

-- The statement to prove: the speed in the second hour is 60 km/h
theorem find_speed_in_second_hour : speed_second_hour 60 :=
by sorry

end find_speed_in_second_hour_l158_158009


namespace sum_of_possible_areas_of_square_in_xy_plane_l158_158330

theorem sum_of_possible_areas_of_square_in_xy_plane (x1 x2 x3 : ℝ) (A : ℝ)
    (h1 : x1 = 2 ∨ x1 = 0 ∨ x1 = 18)
    (h2 : x2 = 2 ∨ x2 = 0 ∨ x2 = 18)
    (h3 : x3 = 2 ∨ x3 = 0 ∨ x3 = 18) :
  A = 1168 := sorry

end sum_of_possible_areas_of_square_in_xy_plane_l158_158330


namespace line_tangent_to_circle_l158_158577

theorem line_tangent_to_circle (l : ℝ → ℝ) (P : ℝ × ℝ) 
  (hP1 : P = (0, 1)) (hP2 : ∀ x y : ℝ, x^2 + y^2 = 1 -> l x = y)
  (hTangent : ∀ x y : ℝ, l x = y ↔ x^2 + y^2 = 1 ∧ y = 1):
  l x = 1 := by
  sorry

end line_tangent_to_circle_l158_158577


namespace ccamathbonanza_2016_2_1_l158_158574

-- Definitions of the speeds of the runners
def bhairav_speed := 28 -- in miles per hour
def daniel_speed := 15 -- in miles per hour
def tristan_speed := 10 -- in miles per hour

-- Distance of the race
def race_distance := 15 -- in miles

-- Time conversion from hours to minutes
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

-- Time taken by each runner to complete the race (in hours)
def time_bhairav := race_distance / bhairav_speed
def time_daniel := race_distance / daniel_speed
def time_tristan := race_distance / tristan_speed

-- Time taken by each runner to complete the race (in minutes)
def time_bhairav_minutes := hours_to_minutes time_bhairav
def time_daniel_minutes := hours_to_minutes time_daniel
def time_tristan_minutes := hours_to_minutes time_tristan

-- Time differences between consecutive runners' finishes (in minutes)
def time_diff_bhairav_daniel := time_daniel_minutes - time_bhairav_minutes
def time_diff_daniel_tristan := time_tristan_minutes - time_daniel_minutes

-- Greatest length of time between consecutive runners' finishes
def greatest_time_diff := max time_diff_bhairav_daniel time_diff_daniel_tristan

-- The theorem we need to prove
theorem ccamathbonanza_2016_2_1 : greatest_time_diff = 30 := by
  sorry

end ccamathbonanza_2016_2_1_l158_158574


namespace sum_of_a_and_b_l158_158959

theorem sum_of_a_and_b (a b : ℝ) (h1 : abs a = 5) (h2 : b = -2) (h3 : a * b > 0) : a + b = -7 := by
  sorry

end sum_of_a_and_b_l158_158959


namespace largest_integral_x_l158_158441

theorem largest_integral_x (x : ℤ) : 
  (1 / 4 : ℝ) < (x / 7) ∧ (x / 7) < (7 / 11 : ℝ) → x ≤ 4 := 
  sorry

end largest_integral_x_l158_158441


namespace units_digit_of_7_pow_3_l158_158937

theorem units_digit_of_7_pow_3 : (7 ^ 3) % 10 = 3 :=
by
  sorry

end units_digit_of_7_pow_3_l158_158937


namespace right_triangle_largest_side_l158_158128

theorem right_triangle_largest_side (b d : ℕ) (h_triangle : (b - d)^2 + b^2 = (b + d)^2)
  (h_arith_seq : (b - d) < b ∧ b < (b + d))
  (h_perimeter : (b - d) + b + (b + d) = 840) :
  (b + d = 350) :=
by sorry

end right_triangle_largest_side_l158_158128


namespace koschei_never_equal_l158_158978

-- Define the problem setup 
def coins_at_vertices (n1 n2 n3 n4 n5 n6 : ℕ) : Prop := 
  ∃ k : ℕ, n1 = k ∧ n2 = k ∧ n3 = k ∧ n4 = k ∧ n5 = k ∧ n6 = k

-- Define the operation condition
def operation_condition (n1 n2 n3 n4 n5 n6 : ℕ) : Prop :=
  ∃ x : ℕ, (n1 - x = x ∧ n2 + 6 * x = x) ∨ (n2 - x = x ∧ n3 + 6 * x = x) ∨ 
  (n3 - x = x ∧ n4 + 6 * x = x) ∨ (n4 - x = x ∧ n5 + 6 * x = x) ∨ 
  (n5 - x = x ∧ n6 + 6 * x = x) ∨ (n6 - x = x ∧ n1 + 6 * x = x)

-- The main theorem 
theorem koschei_never_equal (n1 n2 n3 n4 n5 n6 : ℕ) : 
  (∃ x : ℕ, coins_at_vertices n1 n2 n3 n4 n5 n6) → False :=
by
  sorry

end koschei_never_equal_l158_158978


namespace sum_of_digits_l158_158001

theorem sum_of_digits (x y z w : ℕ) 
  (hxz : z + x = 10) 
  (hyz : y + z = 9) 
  (hxw : x + w = 9) 
  (hx_ne_hy : x ≠ y)
  (hx_ne_hz : x ≠ z)
  (hx_ne_hw : x ≠ w)
  (hy_ne_hz : y ≠ z)
  (hy_ne_hw : y ≠ w)
  (hz_ne_hw : z ≠ w) :
  x + y + z + w = 19 := by
  sorry

end sum_of_digits_l158_158001


namespace fuel_used_l158_158654

theorem fuel_used (x : ℝ) (h1 : x + 0.8 * x = 27) : x = 15 :=
sorry

end fuel_used_l158_158654


namespace scientific_notation_of_number_l158_158465

theorem scientific_notation_of_number :
  1214000 = 1.214 * 10^6 :=
by
  sorry

end scientific_notation_of_number_l158_158465


namespace percentage_equivalence_l158_158196

theorem percentage_equivalence (A B C P : ℝ)
  (hA : A = 0.80 * 600)
  (hB : B = 480)
  (hC : C = 960)
  (hP : P = (B / C) * 100) :
  A = P * 10 :=  -- Since P is the percentage, we use it to relate A to C
sorry

end percentage_equivalence_l158_158196


namespace largest_sum_product_l158_158563

theorem largest_sum_product (p q : ℕ) (h1 : p * q = 100) (h2 : 0 < p) (h3 : 0 < q) : p + q ≤ 101 :=
sorry

end largest_sum_product_l158_158563


namespace yellow_candy_percentage_l158_158874

variable (b : ℝ) (y : ℝ) (r : ℝ)

-- Conditions from the problem
-- 14% more yellow candies than blue candies
axiom yellow_candies : y = 1.14 * b
-- 14% fewer red candies than blue candies
axiom red_candies : r = 0.86 * b
-- Total number of candies equals 1 (or 100%)
axiom total_candies : r + b + y = 1

-- Question to prove: The percentage of yellow candies in the jar is 38%
theorem yellow_candy_percentage  : y = 0.38 := by
  sorry

end yellow_candy_percentage_l158_158874


namespace min_count_to_ensure_multiple_of_5_l158_158360

theorem min_count_to_ensure_multiple_of_5 (n : ℕ) (S : Finset ℕ) (hS : S = Finset.range 31) :
  25 ≤ S.card ∧ (∀ (T : Finset ℕ), T ⊆ S → T.card = 24 → ↑(∃ x ∈ T, x % 5 = 0)) :=
by sorry

end min_count_to_ensure_multiple_of_5_l158_158360


namespace sextuple_angle_terminal_side_on_xaxis_l158_158548

-- Define angle and conditions
variable (α : ℝ)
variable (isPositiveAngle : 0 < α ∧ α < 360)
variable (sextupleAngleOnXAxis : ∃ k : ℕ, 6 * α = k * 360)

-- Prove the possible values of the angle
theorem sextuple_angle_terminal_side_on_xaxis :
  α = 60 ∨ α = 120 ∨ α = 180 ∨ α = 240 ∨ α = 300 :=
  sorry

end sextuple_angle_terminal_side_on_xaxis_l158_158548


namespace max_value_of_a_plus_b_l158_158764

theorem max_value_of_a_plus_b (a b : ℕ) 
  (h : 5 * a + 19 * b = 213) : a + b ≤ 37 :=
  sorry

end max_value_of_a_plus_b_l158_158764


namespace find_c_l158_158050

theorem find_c {A B C : ℝ} (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) 
(h3 : a * Real.sin A + b * Real.sin B - c * Real.sin C = (6 * Real.sqrt 7 / 7) * a * Real.sin B * Real.sin C) :
  c = 2 :=
sorry

end find_c_l158_158050


namespace find_angle_C_find_area_of_triangle_l158_158994

variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides of the triangle

-- Proof 1: Prove \(C = \frac{\pi}{3}\) given \(a \cos B \cos C + b \cos A \cos C = \frac{c}{2}\).

theorem find_angle_C 
  (h : a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2) : C = π / 3 :=
sorry

-- Proof 2: Prove the area of triangle \(ABC = \frac{3\sqrt{3}}{2}\) given \(c = \sqrt{7}\), \(a + b = 5\), and \(C = \frac{\pi}{3}\).

theorem find_area_of_triangle 
  (h1 : c = Real.sqrt 7) (h2 : a + b = 5) (h3 : C = π / 3) : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_of_triangle_l158_158994


namespace find_g50_l158_158623

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g50 (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x * y) = y * g x)
  (h1 : g 1 = 10) : g 50 = 50 * 10 :=
by
  -- The proof sketch here; the detailed proof is omitted
  sorry

end find_g50_l158_158623


namespace find_polynomial_l158_158166

-- Define the polynomial conditions
structure CubicPolynomial :=
  (P : ℝ → ℝ)
  (P0 : ℝ)
  (P1 : ℝ)
  (P2 : ℝ)
  (P3 : ℝ)
  (cubic_eq : ∀ x, P x = P0 + P1 * x + P2 * x^2 + P3 * x^3)

theorem find_polynomial (P : CubicPolynomial) (h_neg1 : P.P (-1) = 2) (h0 : P.P 0 = 3) (h1 : P.P 1 = 1) (h2 : P.P 2 = 15) :
  ∀ x, P.P x = 3 + x - 2 * x^2 - x^3 :=
sorry

end find_polynomial_l158_158166


namespace ellipse_equation_with_m_l158_158281

theorem ellipse_equation_with_m (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m ∈ Set.Ioi 5 := 
sorry

end ellipse_equation_with_m_l158_158281


namespace annual_interest_rate_l158_158714

theorem annual_interest_rate (r : ℝ) :
  (6000 * r + 4000 * 0.09 = 840) → r = 0.08 :=
by sorry

end annual_interest_rate_l158_158714


namespace length_of_train_l158_158493

theorem length_of_train
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_mps : ℝ := speed_kmph * (1000 / 3600))
  (total_distance : ℝ := train_speed_mps * crossing_time)
  (train_length : ℝ := total_distance - platform_length)
  (h_speed : speed_kmph = 72)
  (h_platform : platform_length = 260)
  (h_time : crossing_time = 26)
  : train_length = 260 := by
  sorry

end length_of_train_l158_158493


namespace sqrt_square_multiply_l158_158814

theorem sqrt_square_multiply (a : ℝ) (h : a = 49284) :
  (Real.sqrt a)^2 * 3 = 147852 :=
by
  sorry

end sqrt_square_multiply_l158_158814


namespace square_neg_2x_squared_l158_158209

theorem square_neg_2x_squared (x : ℝ) : (-2 * x ^ 2) ^ 2 = 4 * x ^ 4 :=
by
  sorry

end square_neg_2x_squared_l158_158209


namespace printer_z_time_l158_158255

theorem printer_z_time (t_z : ℝ)
  (hx : (∀ (p : ℝ), p = 16))
  (hy : (∀ (q : ℝ), q = 12))
  (ratio : (16 / (1 /  ((1 / 12) + (1 / t_z)))) = 10 / 3) :
  t_z = 8 := by
  sorry

end printer_z_time_l158_158255


namespace solve_quadratics_l158_158067

theorem solve_quadratics :
  ∃ x y : ℝ, (9 * x^2 - 36 * x - 81 = 0) ∧ (y^2 + 6 * y + 9 = 0) ∧ (x + y = -1 + Real.sqrt 13 ∨ x + y = -1 - Real.sqrt 13) := 
by 
  sorry

end solve_quadratics_l158_158067


namespace value_of_expression_l158_158027

variables {x y z w : ℝ}

theorem value_of_expression (h1 : 4 * x * z + y * w = 4) (h2 : x * w + y * z = 8) :
  (2 * x + y) * (2 * z + w) = 20 :=
by
  sorry

end value_of_expression_l158_158027


namespace total_days_on_jury_duty_l158_158473

-- Definitions based on conditions
def jurySelectionDays := 2
def juryDeliberationDays := 6
def deliberationHoursPerDay := 16
def trialDurationMultiplier := 4

-- Calculate the total number of hours spent in deliberation
def totalDeliberationHours := juryDeliberationDays * 24

-- Calculate the number of days spent in deliberation based on hours per day
def deliberationDays := totalDeliberationHours / deliberationHoursPerDay

-- Calculate the trial days based on trial duration multiplier
def trialDays := jurySelectionDays * trialDurationMultiplier

-- Calculate the total days on jury duty
def totalJuryDutyDays := jurySelectionDays + trialDays + deliberationDays

theorem total_days_on_jury_duty : totalJuryDutyDays = 19 := by
  sorry

end total_days_on_jury_duty_l158_158473


namespace ruby_total_classes_l158_158873

noncomputable def average_price_per_class (pack_cost : ℝ) (pack_classes : ℕ) : ℝ :=
  pack_cost / pack_classes

noncomputable def additional_class_price (average_price : ℝ) : ℝ :=
  average_price + (1/3 * average_price)

noncomputable def total_classes_taken (total_payment : ℝ) (pack_cost : ℝ) (pack_classes : ℕ) : ℕ :=
  let avg_price := average_price_per_class pack_cost pack_classes
  let additional_price := additional_class_price avg_price
  let additional_classes := (total_payment - pack_cost) / additional_price
  pack_classes + Nat.floor additional_classes -- We use Nat.floor to convert from real to natural number of classes

theorem ruby_total_classes 
  (pack_cost : ℝ) 
  (pack_classes : ℕ) 
  (total_payment : ℝ) 
  (h_pack_cost : pack_cost = 75) 
  (h_pack_classes : pack_classes = 10) 
  (h_total_payment : total_payment = 105) :
  total_classes_taken total_payment pack_cost pack_classes = 13 :=
by
  -- The proof would go here
  sorry

end ruby_total_classes_l158_158873


namespace angle_equiv_330_neg390_l158_158316

theorem angle_equiv_330_neg390 : ∃ k : ℤ, 330 = -390 + 360 * k :=
by
  sorry

end angle_equiv_330_neg390_l158_158316


namespace circle_formed_by_PO_equals_3_l158_158705

variable (P : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ)
variable (h_O_fixed : True)
variable (h_PO_constant : dist P O = 3)

theorem circle_formed_by_PO_equals_3 : 
  {P | ∃ (x y : ℝ), dist (x, y) O = 3} = {P | (dist P O = r) ∧ (r = 3)} :=
by
  sorry

end circle_formed_by_PO_equals_3_l158_158705


namespace exterior_angle_decreases_l158_158821

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) (n' : ℕ) (hn' : n' ≥ n) :
  (360 : ℝ) / n' < (360 : ℝ) / n := by sorry

end exterior_angle_decreases_l158_158821


namespace candy_cost_l158_158816

theorem candy_cost (candy_cost_in_cents : ℕ) (pieces : ℕ) (dollar_in_cents : ℕ)
  (h1 : candy_cost_in_cents = 2) (h2 : pieces = 500) (h3 : dollar_in_cents = 100) :
  (pieces * candy_cost_in_cents) / dollar_in_cents = 10 :=
by
  sorry

end candy_cost_l158_158816


namespace sum_reciprocal_inequality_l158_158303

theorem sum_reciprocal_inequality (p q a b c d e : ℝ) (hp : 0 < p) (ha : p ≤ a) (hb : p ≤ b) (hc : p ≤ c) (hd : p ≤ d) (he : p ≤ e) (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) ≤ 25 + 6 * ((Real.sqrt (q / p) - Real.sqrt (p / q)) ^ 2) :=
by sorry

end sum_reciprocal_inequality_l158_158303


namespace original_number_is_10_l158_158011

theorem original_number_is_10 (x : ℤ) (h : 2 * x + 3 = 23) : x = 10 :=
sorry

end original_number_is_10_l158_158011


namespace domain_of_f_l158_158356

noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + (1 / (x - 2))

theorem domain_of_f : { x : ℝ | x ≥ 1 ∧ x ≠ 2 } = { x : ℝ | ∃ (y : ℝ), f x = y } :=
sorry

end domain_of_f_l158_158356


namespace area_in_terms_of_diagonal_l158_158091

variables (l w d : ℝ)

-- Given conditions
def length_to_width_ratio := l / w = 5 / 2
def diagonal_relation := d^2 = l^2 + w^2

-- Proving the area is kd^2 with k = 10 / 29
theorem area_in_terms_of_diagonal 
    (ratio : length_to_width_ratio l w)
    (diag_rel : diagonal_relation l w d) :
  ∃ k, k = 10 / 29 ∧ (l * w = k * d^2) :=
sorry

end area_in_terms_of_diagonal_l158_158091


namespace polynomial_division_l158_158161

-- Define the polynomials P and D
noncomputable def P : Polynomial ℤ := 5 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 7 * Polynomial.X ^ 2 - 9 * Polynomial.X + 12
noncomputable def D : Polynomial ℤ := Polynomial.X - 3
noncomputable def Q : Polynomial ℤ := 5 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2 + 43 * Polynomial.X + 120
def R : ℤ := 372

-- State the theorem
theorem polynomial_division :
  P = D * Q + Polynomial.C R := 
sorry

end polynomial_division_l158_158161


namespace average_of_five_digits_l158_158630

theorem average_of_five_digits 
  (S : ℝ)
  (S3 : ℝ)
  (h_avg8 : S / 8 = 20)
  (h_avg3 : S3 / 3 = 33.333333333333336) :
  (S - S3) / 5 = 12 := 
by
  sorry

end average_of_five_digits_l158_158630


namespace subcommittee_formation_l158_158144

/-- A Senate committee consists of 10 Republicans and 7 Democrats.
    The number of ways to form a subcommittee with 4 Republicans and 3 Democrats is 7350. -/
theorem subcommittee_formation :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end subcommittee_formation_l158_158144


namespace max_sum_x_y_under_condition_l158_158025

-- Define the conditions
variables (x y : ℝ)

-- State the problem and what needs to be proven
theorem max_sum_x_y_under_condition : 
  (3 * (x^2 + y^2) = x - y) → (x + y) ≤ (1 / Real.sqrt 2) :=
by
  sorry

end max_sum_x_y_under_condition_l158_158025


namespace cabbage_price_l158_158661

theorem cabbage_price
  (earnings_wednesday : ℕ)
  (earnings_friday : ℕ)
  (earnings_today : ℕ)
  (total_weight : ℕ)
  (h1 : earnings_wednesday = 30)
  (h2 : earnings_friday = 24)
  (h3 : earnings_today = 42)
  (h4 : total_weight = 48) :
  (earnings_wednesday + earnings_friday + earnings_today) / total_weight = 2 := by
  sorry

end cabbage_price_l158_158661


namespace derivative_f_l158_158897

noncomputable def f (x : ℝ) := x * Real.cos x - Real.sin x

theorem derivative_f :
  ∀ x : ℝ, deriv f x = -x * Real.sin x :=
by
  sorry

end derivative_f_l158_158897


namespace new_person_weight_l158_158502

-- Define the total number of persons and their average weight increase
def num_persons : ℕ := 9
def avg_increase : ℝ := 1.5

-- Define the weight of the person being replaced
def weight_of_replaced_person : ℝ := 65

-- Define the total increase in weight
def total_increase_in_weight : ℝ := num_persons * avg_increase

-- Define the weight of the new person
def weight_of_new_person : ℝ := weight_of_replaced_person + total_increase_in_weight

-- Theorem to prove the weight of the new person is 78.5 kg
theorem new_person_weight : weight_of_new_person = 78.5 := by
  -- proof is omitted
  sorry

end new_person_weight_l158_158502


namespace sum_possible_values_l158_158840

def abs_eq_2023 (a : ℤ) : Prop := abs a = 2023
def abs_eq_2022 (b : ℤ) : Prop := abs b = 2022
def greater_than (a b : ℤ) : Prop := a > b

theorem sum_possible_values (a b : ℤ) (h1 : abs_eq_2023 a) (h2 : abs_eq_2022 b) (h3 : greater_than a b) :
  a + b = 1 ∨ a + b = 4045 := 
sorry

end sum_possible_values_l158_158840


namespace power_function_passing_through_point_l158_158367

theorem power_function_passing_through_point :
  ∃ (α : ℝ), (2:ℝ)^α = 4 := by
  sorry

end power_function_passing_through_point_l158_158367


namespace range_of_varphi_l158_158024

noncomputable def f (x : ℝ) (ω : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ) + 1

theorem range_of_varphi (ω ϕ : ℝ) (h_ω_pos : ω > 0) (h_ϕ_bound : |ϕ| ≤ (Real.pi) / 2)
  (h_intersection : (∀ x, f x ω ϕ = -1 → (∃ k : ℤ, x = (k * Real.pi) / ω)))
  (h_f_gt_1 : (∀ x, -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ω ϕ > 1)) :
  ω = 2 → (Real.pi / 6 ≤ ϕ) ∧ (ϕ ≤ Real.pi / 3) :=
by
  sorry

end range_of_varphi_l158_158024


namespace sin_pi_minus_alpha_l158_158021

theorem sin_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 Real.pi) (h2 : Real.cos α = 4 / 5) :
  Real.sin (Real.pi - α) = 3 / 5 := 
sorry

end sin_pi_minus_alpha_l158_158021


namespace ratio_of_areas_l158_158514

-- Defining the variables for sides of rectangles
variables {a b c d : ℝ}

-- Given conditions
axiom h1 : a / c = 4 / 5
axiom h2 : b / d = 4 / 5

-- Statement to prove the ratio of areas
theorem ratio_of_areas (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) : (a * b) / (c * d) = 16 / 25 :=
sorry

end ratio_of_areas_l158_158514


namespace quadratic_pairs_square_diff_exists_l158_158807

open Nat Polynomial

theorem quadratic_pairs_square_diff_exists (P : Polynomial ℤ) (u v w a b n : ℤ) (n_pos : 0 < n)
    (hp : ∃ (u v w : ℤ), P = C u * X ^ 2 + C v * X + C w)
    (h_ab : P.eval a - P.eval b = n^2) : ∃ k > 10^6, ∃ m : ℕ, ∃ c d : ℤ, (c - d = a - b + 2 * k) ∧ 
    (P.eval c - P.eval d = n^2 * m ^ 2) :=
by
  sorry

end quadratic_pairs_square_diff_exists_l158_158807


namespace fifth_inequality_l158_158022

theorem fifth_inequality :
  1 + (1 / 2^2) + (1 / 3^2) + (1 / 4^2) + (1 / 5^2) + (1 / 6^2) < 11 / 6 :=
sorry

end fifth_inequality_l158_158022


namespace percent_unionized_men_is_70_l158_158882

open Real

def total_employees : ℝ := 100
def percent_men : ℝ := 0.5
def percent_unionized : ℝ := 0.6
def percent_women_nonunion : ℝ := 0.8
def percent_men_nonunion : ℝ := 0.2

def num_men := total_employees * percent_men
def num_unionized := total_employees * percent_unionized
def num_nonunion := total_employees - num_unionized
def num_men_nonunion := num_nonunion * percent_men_nonunion
def num_men_unionized := num_men - num_men_nonunion

theorem percent_unionized_men_is_70 :
  (num_men_unionized / num_unionized) * 100 = 70 := by
  sorry

end percent_unionized_men_is_70_l158_158882


namespace find_principal_l158_158742

theorem find_principal
  (SI : ℝ)
  (R : ℝ)
  (T : ℝ)
  (h_SI : SI = 4025.25)
  (h_R : R = 0.09)
  (h_T : T = 5) : 
  (SI / (R * T / 100)) = 8950 :=
by
  rw [h_SI, h_R, h_T]
  sorry

end find_principal_l158_158742


namespace fixed_point_of_function_l158_158211

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : (2, 3) ∈ { (x, y) | y = 2 + a^(x-2) } :=
sorry

end fixed_point_of_function_l158_158211


namespace prize_distribution_l158_158822

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prize_distribution :
  let total_ways := 
    (binomial_coefficient 7 3) * 5 * (Nat.factorial 4) + 
    (binomial_coefficient 7 2 * binomial_coefficient 5 2 / 2) * 
    (binomial_coefficient 5 2) * (Nat.factorial 3)
  total_ways = 10500 :=
by 
  sorry

end prize_distribution_l158_158822


namespace shaded_percentage_seven_by_seven_grid_l158_158844

theorem shaded_percentage_seven_by_seven_grid :
  let total_squares := 49
  let shaded_squares := 7
  let shaded_fraction := shaded_squares / total_squares
  let shaded_percentage := shaded_fraction * 100
  shaded_percentage = 14.29 := by
  sorry

end shaded_percentage_seven_by_seven_grid_l158_158844


namespace nancy_threw_out_2_carrots_l158_158903

theorem nancy_threw_out_2_carrots :
  ∀ (x : ℕ), 12 - x + 21 = 31 → x = 2 :=
by
  sorry

end nancy_threw_out_2_carrots_l158_158903


namespace part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l158_158862

-- Definitions of the sets and conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part 1
theorem part1_union (a : ℝ) (ha : a = 1) : 
  A ∪ B a = { x | -4 < x ∧ x ≤ 3 } :=
sorry

theorem part1_intersection_complement (a : ℝ) (ha : a = 1) : 
  A ∩ (U \ B a) = { x | -4 < x ∧ x < 0 } :=
sorry

-- Part 2
theorem part2_necessary_sufficient_condition (a : ℝ) : 
  (∀ x, x ∈ B a ↔ x ∈ A) ↔ (-3 < a ∧ a < -1) :=
sorry

end part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l158_158862


namespace correct_option_D_l158_158838

theorem correct_option_D : -2 = -|-2| := 
by 
  sorry

end correct_option_D_l158_158838


namespace value_of_a_l158_158793

noncomputable def M : Set ℝ := {x | x^2 = 2}
noncomputable def N (a : ℝ) : Set ℝ := {x | a*x = 1}

theorem value_of_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2 :=
by
  intro h
  sorry

end value_of_a_l158_158793


namespace math_problem_l158_158677

theorem math_problem (a b c d x y : ℝ) (h1 : a = -b) (h2 : c * d = 1) 
  (h3 : (x + 3)^2 + |y - 2| = 0) : 2 * (a + b) - 2 * (c * d)^4 + (x + y)^2022 = -1 :=
by
  sorry

end math_problem_l158_158677


namespace sample_size_eq_100_l158_158943

variables (frequency : ℕ) (frequency_rate : ℚ)

theorem sample_size_eq_100 (h1 : frequency = 50) (h2 : frequency_rate = 0.5) :
  frequency / frequency_rate = 100 :=
by
  sorry

end sample_size_eq_100_l158_158943


namespace kenya_more_peanuts_l158_158711

-- Define the number of peanuts Jose has
def Jose_peanuts : ℕ := 85

-- Define the number of peanuts Kenya has
def Kenya_peanuts : ℕ := 133

-- The proof problem: Prove that Kenya has 48 more peanuts than Jose
theorem kenya_more_peanuts : Kenya_peanuts - Jose_peanuts = 48 :=
by
  -- The proof will go here
  sorry

end kenya_more_peanuts_l158_158711


namespace knight_will_be_freed_l158_158148

/-- Define a structure to hold the state of the piles -/
structure PileState where
  pile1_magical : ℕ
  pile1_non_magical : ℕ
  pile2_magical : ℕ
  pile2_non_magical : ℕ
deriving Repr

-- Function to move one coin from pile1 to pile2
def move_coin (state : PileState) : PileState :=
  if state.pile1_magical > 0 then
    { state with
      pile1_magical := state.pile1_magical - 1,
      pile2_magical := state.pile2_magical + 1 }
  else if state.pile1_non_magical > 0 then
    { state with
      pile1_non_magical := state.pile1_non_magical - 1,
      pile2_non_magical := state.pile2_non_magical + 1 }
  else
    state -- If no coins to move, the state remains unchanged

-- The initial state of the piles
def initial_state : PileState :=
  { pile1_magical := 0, pile1_non_magical := 49, pile2_magical := 50, pile2_non_magical := 1 }

-- Check if the knight can be freed (both piles have the same number of magical or non-magical coins)
def knight_free (state : PileState) : Prop :=
  state.pile1_magical = state.pile2_magical ∨ state.pile1_non_magical = state.pile2_non_magical

noncomputable def knight_can_be_freed_by_25th_day : Prop :=
  exists n : ℕ, n ≤ 25 ∧ knight_free (Nat.iterate move_coin n initial_state)

theorem knight_will_be_freed : knight_can_be_freed_by_25th_day :=
  sorry

end knight_will_be_freed_l158_158148


namespace product_of_xy_l158_158437

theorem product_of_xy : 
  ∃ (x y : ℝ), 3 * x + 4 * y = 60 ∧ 6 * x - 4 * y = 12 ∧ x * y = 72 :=
by
  sorry

end product_of_xy_l158_158437


namespace taylor_one_basket_in_three_tries_l158_158786

theorem taylor_one_basket_in_three_tries (P_no_make : ℚ) (h : P_no_make = 1/3) : 
  (∃ P_make : ℚ, P_make = 1 - P_no_make ∧ P_make * P_no_make * P_no_make * 3 = 2/9) := 
by
  sorry

end taylor_one_basket_in_three_tries_l158_158786


namespace solutions_to_equation_l158_158234

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 10*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 12*x - 8)) = 0

theorem solutions_to_equation :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2) :=
sorry

end solutions_to_equation_l158_158234


namespace tan_alpha_eq_2_l158_158012

theorem tan_alpha_eq_2 (α : ℝ) (h : Real.tan α = 2) : (Real.cos α + 3 * Real.sin α) / (3 * Real.cos α - Real.sin α) = 7 := by
  sorry

end tan_alpha_eq_2_l158_158012


namespace one_prime_p_10_14_l158_158964

theorem one_prime_p_10_14 :
  ∃! (p : ℕ), Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end one_prime_p_10_14_l158_158964


namespace book_discount_l158_158104

theorem book_discount (a b : ℕ) (x y : ℕ) (h1 : x = 10 * a + b) (h2 : y = 10 * b + a) (h3 : (3 / 8) * x = y) :
  x - y = 45 := 
sorry

end book_discount_l158_158104


namespace cruzs_marbles_l158_158990

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end cruzs_marbles_l158_158990


namespace find_starting_number_l158_158901

-- Define that there are 15 even integers between a starting number and 40
def even_integers_range (n : ℕ) : Prop :=
  ∃ k : ℕ, (1 ≤ k) ∧ (k = 15) ∧ (n + 2*(k-1) = 40)

-- Proof statement
theorem find_starting_number : ∃ n : ℕ, even_integers_range n ∧ n = 12 :=
by
  sorry

end find_starting_number_l158_158901


namespace speeds_and_time_l158_158546

theorem speeds_and_time (x s : ℕ) (t : ℝ)
  (h1 : ∀ {t : ℝ}, t = 2 → x * t > s * t + 24)
  (h2 : ∀ {t : ℝ}, t = 0.5 → x * t = 8) :
  x = 16 ∧ s = 4 ∧ t = 8 :=
by {
  sorry
}

end speeds_and_time_l158_158546


namespace money_left_l158_158386

-- Conditions
def initial_savings : ℤ := 6000
def spent_on_flight : ℤ := 1200
def spent_on_hotel : ℤ := 800
def spent_on_food : ℤ := 3000

-- Total spent
def total_spent : ℤ := spent_on_flight + spent_on_hotel + spent_on_food

-- Prove that the money left is $1,000
theorem money_left (h1 : initial_savings = 6000)
                   (h2 : spent_on_flight = 1200)
                   (h3 : spent_on_hotel = 800)
                   (h4 : spent_on_food = 3000) :
                   initial_savings - total_spent = 1000 :=
by
  -- Insert proof steps here
  sorry

end money_left_l158_158386


namespace braiding_time_l158_158241

variables (n_dancers : ℕ) (b_braids_per_dancer : ℕ) (t_seconds_per_braid : ℕ)

theorem braiding_time : n_dancers = 8 → b_braids_per_dancer = 5 → t_seconds_per_braid = 30 → 
  (n_dancers * b_braids_per_dancer * t_seconds_per_braid) / 60 = 20 :=
by
  intros
  sorry

end braiding_time_l158_158241


namespace jack_round_trip_speed_l158_158535

noncomputable def jack_average_speed (d1 d2 : ℕ) (t1 t2 : ℕ) : ℕ :=
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let total_time_hours := total_time / 60
  total_distance / total_time_hours

theorem jack_round_trip_speed : jack_average_speed 3 3 45 15 = 6 := by
  -- Import necessary library
  sorry

end jack_round_trip_speed_l158_158535


namespace quadratic_radical_same_type_l158_158415

theorem quadratic_radical_same_type (a : ℝ) (h : (∃ (t : ℝ), t ^ 2 = 3 * a - 4) ∧ (∃ (t : ℝ), t ^ 2 = 8)) : a = 2 :=
by
  -- Extract the properties of the radicals
  sorry

end quadratic_radical_same_type_l158_158415


namespace closest_point_on_line_to_target_l158_158492

noncomputable def parametricPoint (s : ℝ) : ℝ × ℝ × ℝ :=
  (6 + 3 * s, 2 - 9 * s, 0 + 6 * s)

noncomputable def closestPoint : ℝ × ℝ × ℝ :=
  (249/42, 95/42, -1/7)

theorem closest_point_on_line_to_target :
  ∃ s : ℝ, parametricPoint s = closestPoint :=
by
  sorry

end closest_point_on_line_to_target_l158_158492


namespace factorize_expression_l158_158942

theorem factorize_expression (x : ℝ) : -2 * x^2 + 2 * x - (1 / 2) = -2 * (x - (1 / 2))^2 :=
by
  sorry

end factorize_expression_l158_158942


namespace complex_point_in_fourth_quadrant_l158_158555

theorem complex_point_in_fourth_quadrant (z : ℂ) (h : z = 1 / (1 + I)) :
  z.re > 0 ∧ z.im < 0 :=
by
  -- Here we would provide the proof, but it is omitted as per the instructions.
  sorry

end complex_point_in_fourth_quadrant_l158_158555


namespace problem1_problem2_problem3_l158_158037

-- Proof Problem 1: $A$ and $B$ are not standing together
theorem problem1 : 
  ∃ (n : ℕ), n = 480 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "A" ∨ students 1 ≠ "B" :=
sorry

-- Proof Problem 2: $C$ and $D$ must stand together
theorem problem2 : 
  ∃ (n : ℕ), n = 240 ∧ 
  ∀ (students : Fin 6 → String),
    (students 0 = "C" ∧ students 1 = "D") ∨ 
    (students 1 = "C" ∧ students 2 = "D") :=
sorry

-- Proof Problem 3: $E$ is not at the beginning and $F$ is not at the end
theorem problem3 : 
  ∃ (n : ℕ), n = 504 ∧ 
  ∀ (students : Fin 6 → String),
    students 0 ≠ "E" ∧ students 5 ≠ "F" :=
sorry

end problem1_problem2_problem3_l158_158037


namespace minimum_negative_factors_l158_158056

theorem minimum_negative_factors (a b c d : ℝ) (h1 : a * b * c * d < 0) (h2 : a + b = 0) (h3 : c * d > 0) : 
    (∃ x ∈ [a, b, c, d], x < 0) :=
by
  sorry

end minimum_negative_factors_l158_158056


namespace total_distance_covered_l158_158636

theorem total_distance_covered (up_speed down_speed up_time down_time : ℕ) (H1 : up_speed = 30) (H2 : down_speed = 50) (H3 : up_time = 5) (H4 : down_time = 5) :
  (up_speed * up_time + down_speed * down_time) = 400 := 
by
  sorry

end total_distance_covered_l158_158636


namespace poly_div_l158_158892

theorem poly_div (A B : ℂ) :
  (∀ x : ℂ, x^3 + x^2 + 1 = 0 → x^202 + A * x + B = 0) → A + B = 0 :=
by
  intros h
  sorry

end poly_div_l158_158892


namespace factory_toys_production_each_day_l158_158932

theorem factory_toys_production_each_day 
  (weekly_production : ℕ)
  (days_worked_per_week : ℕ)
  (h1 : weekly_production = 4560)
  (h2 : days_worked_per_week = 4) : 
  (weekly_production / days_worked_per_week) = 1140 :=
  sorry

end factory_toys_production_each_day_l158_158932


namespace min_abs_E_value_l158_158995

theorem min_abs_E_value (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 10) : |E| = 9 :=
sorry

end min_abs_E_value_l158_158995


namespace harriet_speed_l158_158953

/-- Harriet drove back from B-town to A-ville at a constant speed of 145 km/hr.
    The entire trip took 5 hours, and it took Harriet 2.9 hours to drive from A-ville to B-town.
    Prove that Harriet's speed while driving from A-ville to B-town was 105 km/hr. -/
theorem harriet_speed (v_return : ℝ) (T_total : ℝ) (t_AB : ℝ) (v_AB : ℝ) :
  v_return = 145 →
  T_total = 5 →
  t_AB = 2.9 →
  v_AB = 105 :=
by
  intros
  sorry

end harriet_speed_l158_158953


namespace value_of_a_l158_158660

theorem value_of_a (a : ℝ) (x y : ℝ) : 
  (x + a^2 * y + 6 = 0 ∧ (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ a = -1 :=
by
  sorry

end value_of_a_l158_158660


namespace sum_of_heights_less_than_perimeter_l158_158491

theorem sum_of_heights_less_than_perimeter
  (a b c h1 h2 h3 : ℝ) 
  (H1 : h1 ≤ b) 
  (H2 : h2 ≤ c) 
  (H3 : h3 ≤ a) 
  (H4 : h1 < b ∨ h2 < c ∨ h3 < a) : 
  h1 + h2 + h3 < a + b + c :=
by {
  sorry
}

end sum_of_heights_less_than_perimeter_l158_158491


namespace find_number_l158_158331

theorem find_number (x : ℝ) : 
  220050 = (555 + x) * (2 * (x - 555)) + 50 ↔ x = 425.875 ∨ x = -980.875 := 
by 
  sorry

end find_number_l158_158331


namespace intersection_A_B_l158_158413

def A : Set Real := { y | ∃ x : Real, y = Real.cos x }
def B : Set Real := { x | x^2 < 9 }

theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } :=
by
  sorry

end intersection_A_B_l158_158413


namespace find_max_value_l158_158858

theorem find_max_value (f : ℝ → ℝ) (h₀ : f 0 = -5) (h₁ : ∀ x, deriv f x = 4 * x^3 - 4 * x) :
  ∃ x, f x = -5 ∧ (∀ y, f y ≤ f x) ∧ x = 0 :=
sorry

end find_max_value_l158_158858


namespace tangents_of_convex_quad_l158_158855

theorem tangents_of_convex_quad (
  α β γ δ : ℝ
) (m : ℝ) (h₀ : α + β + γ + δ = 2 * Real.pi) (h₁ : 0 < α ∧ α < Real.pi) (h₂ : 0 < β ∧ β < Real.pi) 
  (h₃ : 0 < γ ∧ γ < Real.pi) (h₄ : 0 < δ ∧ δ < Real.pi) (t1 : Real.tan α = m) :
  ¬ (Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) :=
sorry

end tangents_of_convex_quad_l158_158855


namespace largest_multiple_of_7_smaller_than_neg_55_l158_158878

theorem largest_multiple_of_7_smaller_than_neg_55 : ∃ m : ℤ, m % 7 = 0 ∧ m < -55 ∧ ∀ n : ℤ, n % 7 = 0 → n < -55 → n ≤ m :=
sorry

end largest_multiple_of_7_smaller_than_neg_55_l158_158878


namespace exponentiation_calculation_l158_158900

theorem exponentiation_calculation : 3000 * (3000 ^ 3000) ^ 2 = 3000 ^ 6001 := by
  sorry

end exponentiation_calculation_l158_158900


namespace problem_equivalent_l158_158371

theorem problem_equivalent (a c : ℕ) (h : (3 * 100 + a * 10 + 7) + 214 = 5 * 100 + c * 10 + 1) (h5c1_div3 : (5 + c + 1) % 3 = 0) : a + c = 4 :=
sorry

end problem_equivalent_l158_158371


namespace sam_friend_points_l158_158815

theorem sam_friend_points (sam_points total_points : ℕ) (h1 : sam_points = 75) (h2 : total_points = 87) :
  total_points - sam_points = 12 :=
by sorry

end sam_friend_points_l158_158815


namespace students_on_bus_l158_158045

theorem students_on_bus
    (initial_students : ℝ) (first_get_on : ℝ) (first_get_off : ℝ)
    (second_get_on : ℝ) (second_get_off : ℝ)
    (third_get_on : ℝ) (third_get_off : ℝ) :
  initial_students = 21 →
  first_get_on = 7.5 → first_get_off = 2 → 
  second_get_on = 1.2 → second_get_off = 5.3 →
  third_get_on = 11 → third_get_off = 4.8 →
  (initial_students + (first_get_on - first_get_off) +
   (second_get_on - second_get_off) +
   (third_get_on - third_get_off)) = 28.6 := by
  intros
  sorry

end students_on_bus_l158_158045


namespace shorter_piece_length_l158_158902

theorem shorter_piece_length :
  ∃ (x : ℝ), x + 2 * x = 69 ∧ x = 23 :=
by
  sorry

end shorter_piece_length_l158_158902


namespace archie_needs_sod_l158_158790

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end archie_needs_sod_l158_158790


namespace find_solutions_l158_158685

noncomputable def solution_exists (x y z p : ℝ) : Prop :=
  (x^2 - 1 = p * (y + z)) ∧
  (y^2 - 1 = p * (z + x)) ∧
  (z^2 - 1 = p * (x + y))

theorem find_solutions (x y z p : ℝ) :
  solution_exists x y z p ↔
  (x = (p + Real.sqrt (p^2 + 1)) ∧ y = (p + Real.sqrt (p^2 + 1)) ∧ z = (p + Real.sqrt (p^2 + 1)) ∨
   x = (p - Real.sqrt (p^2 + 1)) ∧ y = (p - Real.sqrt (p^2 + 1)) ∧ z = (p - Real.sqrt (p^2 + 1))) ∨
  (x = (Real.sqrt (1 - p^2)) ∧ y = (Real.sqrt (1 - p^2)) ∧ z = (-p - Real.sqrt (1 - p^2)) ∨
   x = (-Real.sqrt (1 - p^2)) ∧ y = (-Real.sqrt (1 - p^2)) ∧ z = (-p + Real.sqrt (1 - p^2))) :=
by
  -- Proof starts here
  sorry

end find_solutions_l158_158685


namespace find_a5_l158_158451

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (a1 : ℝ)

-- Geometric sequence definition
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
∀ (n : ℕ), a (n + 1) = a1 * q^n

-- Given conditions
def condition1 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 1 + a 3 = 10

def condition2 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
a 2 + a 4 = -30

-- Theorem to prove
theorem find_a5 (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h1 : geometric_sequence a a1 q)
  (h2 : condition1 a a1 q)
  (h3 : condition2 a a1 q) :
  a 5 = 81 := by
  sorry

end find_a5_l158_158451


namespace mixed_candy_price_l158_158736

noncomputable def price_per_pound (a b c : ℕ) (pa pb pc : ℝ) : ℝ :=
  (a * pa + b * pb + c * pc) / (a + b + c)

theorem mixed_candy_price :
  let a := 30
  let b := 15
  let c := 20
  let pa := 10.0
  let pb := 12.0
  let pc := 15.0
  price_per_pound a b c pa pb pc * 0.9 = 10.8 := by
  sorry

end mixed_candy_price_l158_158736


namespace sum_sequence_up_to_2015_l158_158481

def sequence_val (n : ℕ) : ℕ :=
  if n % 288 = 0 then 7 
  else if n % 224 = 0 then 9
  else if n % 63 = 0 then 32
  else 0

theorem sum_sequence_up_to_2015 : 
  (Finset.range 2016).sum sequence_val = 1106 :=
by
  sorry

end sum_sequence_up_to_2015_l158_158481


namespace smallest_number_is_16_l158_158147

theorem smallest_number_is_16 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c) / 3 = 24 ∧ 
  (b = 25) ∧ (c = b + 6) ∧ min a (min b c) = 16 :=
by
  sorry

end smallest_number_is_16_l158_158147


namespace smallest_q_exists_l158_158839

noncomputable def p_q_r_are_consecutive_terms (p q r : ℝ) : Prop :=
∃ d : ℝ, p = q - d ∧ r = q + d

theorem smallest_q_exists
  (p q r : ℝ)
  (h1 : p_q_r_are_consecutive_terms p q r)
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : r > 0)
  (h5 : p * q * r = 216) :
  q = 6 :=
sorry

end smallest_q_exists_l158_158839


namespace value_of_x_plus_y_l158_158193

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := 3

theorem value_of_x_plus_y
  (hx : 1 / x = 2)
  (hy : 1 / x + 3 / y = 3) :
  x + y = 7 / 2 :=
  sorry

end value_of_x_plus_y_l158_158193


namespace third_part_of_division_l158_158406

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division_l158_158406


namespace mark_min_correct_problems_l158_158785

noncomputable def mark_score (x : ℕ) : ℤ :=
  8 * x - 21

theorem mark_min_correct_problems (x : ℕ) :
  (4 * 2) + mark_score x ≥ 120 ↔ x ≥ 17 :=
by
  sorry

end mark_min_correct_problems_l158_158785


namespace sample_size_proportion_l158_158506

theorem sample_size_proportion (n : ℕ) (ratio_A B C : ℕ) (A_sample : ℕ) (ratio_A_val : ratio_A = 5) (ratio_B_val : ratio_B = 2) (ratio_C_val : ratio_C = 3) (A_sample_val : A_sample = 15) (total_ratio : ratio_A + ratio_B + ratio_C = 10) : 
  15 / n = 5 / 10 → n = 30 :=
sorry

end sample_size_proportion_l158_158506


namespace people_with_fewer_than_seven_cards_l158_158633

theorem people_with_fewer_than_seven_cards (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ)
  (h1 : total_cards = 52) (h2 : num_people = 8) (h3 : total_cards = num_people * cards_per_person + extra_cards) (h4 : extra_cards < num_people) :
  ∃ fewer_than_seven : ℕ, num_people - extra_cards = fewer_than_seven :=
by
  have remainder := (52 % 8)
  have cards_per_person := (52 / 8)
  have number_fewer_than_seven := num_people - remainder
  existsi number_fewer_than_seven
  sorry

end people_with_fewer_than_seven_cards_l158_158633


namespace find_real_m_of_purely_imaginary_z_l158_158448

theorem find_real_m_of_purely_imaginary_z (m : ℝ) 
  (h1 : m^2 - 8 * m + 15 = 0) 
  (h2 : m^2 - 9 * m + 18 ≠ 0) : 
  m = 5 := 
by 
  sorry

end find_real_m_of_purely_imaginary_z_l158_158448


namespace simplify_expression_l158_158429

noncomputable def algebraic_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) : ℚ :=
(1 - 3 / (a + 2)) / ((a^2 - 2 * a + 1) / (a^2 - 4))

theorem simplify_expression (a : ℚ) (h1 : a ≠ -2) (h2 : a ≠ 2) (h3 : a ≠ 1) :
  algebraic_expression a h1 h2 h3 = (a - 2) / (a - 1) :=
by
  sorry

end simplify_expression_l158_158429


namespace ratio_proof_l158_158701

theorem ratio_proof (a b x : ℝ) (h : a > b) (h_b_pos : b > 0)
  (h_x : x = 0.5 * Real.sqrt (a / b) + 0.5 * Real.sqrt (b / a)) :
  2 * b * Real.sqrt (x^2 - 1) / (x - Real.sqrt (x^2 - 1)) = a - b := 
sorry

end ratio_proof_l158_158701


namespace problem_1_problem_2_problem_3_problem_4_problem_5_l158_158123

theorem problem_1 : 286 = 200 + 80 + 6 := sorry
theorem problem_2 : 7560 = 7000 + 500 + 60 := sorry
theorem problem_3 : 2048 = 2000 + 40 + 8 := sorry
theorem problem_4 : 8009 = 8000 + 9 := sorry
theorem problem_5 : 3070 = 3000 + 70 := sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_l158_158123


namespace hall_of_mirrors_l158_158987

theorem hall_of_mirrors (h : ℝ) 
    (condition1 : 2 * (30 * h) + (20 * h) = 960) :
  h = 12 :=
by
  sorry

end hall_of_mirrors_l158_158987


namespace minimum_reciprocal_sum_l158_158925

theorem minimum_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  (∃ z : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 2 * y = 1 → z ≤ (1 / x + 2 / y)) ∧ z = 35 / 6) :=
  sorry

end minimum_reciprocal_sum_l158_158925


namespace box_width_is_target_width_l158_158863

-- Defining the conditions
def cube_volume : ℝ := 27
def box_length : ℝ := 8
def box_height : ℝ := 12
def max_cubes : ℕ := 24

-- Defining the target width we want to prove
def target_width : ℝ := 6.75

-- The proof statement
theorem box_width_is_target_width :
  ∃ w : ℝ,
  (∀ v : ℝ, (v = max_cubes * cube_volume) →
   ∀ l : ℝ, (l = box_length) →
   ∀ h : ℝ, (h = box_height) →
   v = l * w * h) →
   w = target_width :=
by
  sorry

end box_width_is_target_width_l158_158863


namespace nail_insertion_l158_158625

theorem nail_insertion (k : ℝ) (h1 : 0 < k) (h2 : k < 1) : 
  (4/7) + (4/7) * k + (4/7) * k^2 = 1 :=
by sorry

end nail_insertion_l158_158625


namespace critical_temperature_of_water_l158_158914

/--
Given the following conditions:
1. The temperature at which solid, liquid, and gaseous water coexist is the triple point.
2. The temperature at which water vapor condenses is the condensation point.
3. The maximum temperature at which liquid water can exist.
4. The minimum temperature at which water vapor can exist.

Prove that the critical temperature of water is the maximum temperature at which liquid water can exist.
-/
theorem critical_temperature_of_water :
    ∀ (triple_point condensation_point maximum_liquid_temp minimum_vapor_temp critical_temp : ℝ), 
    (critical_temp = maximum_liquid_temp) ↔
    ((critical_temp ≠ triple_point) ∧ (critical_temp ≠ condensation_point) ∧ (critical_temp ≠ minimum_vapor_temp)) := 
  sorry

end critical_temperature_of_water_l158_158914


namespace completing_the_square_l158_158980

theorem completing_the_square (x : ℝ) :
  4 * x^2 - 2 * x - 1 = 0 → (x - 1/4)^2 = 5/16 := 
by
  sorry

end completing_the_square_l158_158980


namespace right_triangle_area_l158_158831

theorem right_triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a * a + b * b = c * c ∧ (1/2 : ℝ) * a * b = 6 := 
sorry

end right_triangle_area_l158_158831


namespace least_possible_faces_two_dice_l158_158498

noncomputable def least_possible_sum_of_faces (a b : ℕ) : ℕ :=
(a + b)

theorem least_possible_faces_two_dice (a b : ℕ) (h1 : 8 ≤ a) (h2 : 8 ≤ b)
  (h3 : ∃ k, 9 * k = 2 * (11 * k)) 
  (h4 : ∃ m, 9 * m = a * b) : 
  least_possible_sum_of_faces a b = 22 :=
sorry

end least_possible_faces_two_dice_l158_158498


namespace regular_octagon_interior_angle_l158_158557

theorem regular_octagon_interior_angle:
  let n := 8
  let sum_interior_angles (n: ℕ) := 180 * (n - 2)
  let interior_angle (n: ℕ) := sum_interior_angles n / n
  interior_angle n = 135 :=
by
  sorry

end regular_octagon_interior_angle_l158_158557


namespace regular_price_one_bag_l158_158542

theorem regular_price_one_bag (p : ℕ) (h : 3 * p + 5 = 305) : p = 100 :=
by
  sorry

end regular_price_one_bag_l158_158542


namespace remaining_pages_l158_158847

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l158_158847


namespace number_of_adult_males_l158_158697

def population := 480
def ratio_children := 1
def ratio_adult_males := 2
def ratio_adult_females := 2
def total_ratio_parts := ratio_children + ratio_adult_males + ratio_adult_females

theorem number_of_adult_males : 
  (population / total_ratio_parts) * ratio_adult_males = 192 :=
by
  sorry

end number_of_adult_males_l158_158697


namespace fifth_pile_magazines_l158_158941

theorem fifth_pile_magazines :
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  fifth_pile = 13 :=
by
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  show fifth_pile = 13
  sorry

end fifth_pile_magazines_l158_158941


namespace notebooks_multiple_of_3_l158_158275

theorem notebooks_multiple_of_3 (N : ℕ) (h1 : ∃ k : ℕ, N = 3 * k) :
  ∃ k : ℕ, N = 3 * k :=
by
  sorry

end notebooks_multiple_of_3_l158_158275


namespace quadratic_roots_condition_l158_158422

theorem quadratic_roots_condition (m : ℝ) :
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1^2 - 3 * x1 + 2 * m = 0 ∧ x2^2 - 3 * x2 + 2 * m = 0) →
  m < 9 / 8 :=
by
  sorry

end quadratic_roots_condition_l158_158422


namespace find_a_l158_158015

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
    (h3 : 13 ∣ 53^2016 + a) : a = 12 := 
by 
  -- proof would be written here
  sorry

end find_a_l158_158015


namespace larger_number_is_8_l158_158762

-- Define the conditions
def is_twice (x y : ℕ) : Prop := x = 2 * y
def product_is_40 (x y : ℕ) : Prop := x * y = 40
def sum_is_14 (x y : ℕ) : Prop := x + y = 14

-- The proof statement
theorem larger_number_is_8 (x y : ℕ) (h1 : is_twice x y) (h2 : product_is_40 x y) (h3 : sum_is_14 x y) : x = 8 :=
  sorry

end larger_number_is_8_l158_158762


namespace systematic_sampling_remove_l158_158421

theorem systematic_sampling_remove (total_people : ℕ) (sample_size : ℕ) (remove_count : ℕ): 
  total_people = 162 → sample_size = 16 → remove_count = 2 → 
  (total_people - 1) % sample_size = sample_size - 1 :=
by
  sorry

end systematic_sampling_remove_l158_158421


namespace value_of_f2008_plus_f2009_l158_158180

variable {f : ℤ → ℤ}

-- Conditions
axiom h1 : ∀ x : ℤ, f (-(x) + 2) = -f (x + 2)
axiom h2 : ∀ x : ℤ, f (6 - x) = f x
axiom h3 : f 3 = 2

-- The theorem to prove
theorem value_of_f2008_plus_f2009 : f 2008 + f 2009 = -2 :=
  sorry

end value_of_f2008_plus_f2009_l158_158180


namespace fraction_of_power_l158_158102

theorem fraction_of_power (m : ℕ) (h : m = 16^1500) : m / 8 = 2^5997 := by
  sorry

end fraction_of_power_l158_158102


namespace smallest_divisor_28_l158_158868

theorem smallest_divisor_28 : ∃ (d : ℕ), d > 0 ∧ d ∣ 28 ∧ ∀ (d' : ℕ), d' > 0 ∧ d' ∣ 28 → d ≤ d' := by
  sorry

end smallest_divisor_28_l158_158868


namespace henry_has_more_games_l158_158286

-- Define the conditions and initial states
def initial_games_henry : ℕ := 33
def given_games_neil : ℕ := 5
def initial_games_neil : ℕ := 2

-- Define the number of games Henry and Neil have now
def games_henry_now : ℕ := initial_games_henry - given_games_neil
def games_neil_now : ℕ := initial_games_neil + given_games_neil

-- State the theorem to be proven
theorem henry_has_more_games : games_henry_now / games_neil_now = 4 :=
by
  sorry

end henry_has_more_games_l158_158286


namespace cube_volume_l158_158213

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V, V = 125 := 
by
  sorry

end cube_volume_l158_158213


namespace number_of_circles_is_3_l158_158864

-- Define the radius and diameter of the circles
def radius := 4
def diameter := 2 * radius

-- Given the total horizontal length
def total_horizontal_length := 24

-- Number of circles calculated as per the given conditions
def number_of_circles := total_horizontal_length / diameter

-- The proof statement to verify
theorem number_of_circles_is_3 : number_of_circles = 3 := by
  sorry

end number_of_circles_is_3_l158_158864


namespace smallest_w_l158_158788

theorem smallest_w (w : ℕ) (h1 : 2^4 ∣ 1452 * w) (h2 : 3^3 ∣ 1452 * w) (h3 : 13^3 ∣ 1452 * w) : w = 79132 :=
by
  sorry

end smallest_w_l158_158788


namespace seashells_solution_l158_158975

def seashells_problem (T : ℕ) : Prop :=
  T + 13 = 50 → T = 37

theorem seashells_solution : seashells_problem 37 :=
by
  intro h
  sorry

end seashells_solution_l158_158975


namespace star_k_l158_158071

def star (x y : ℤ) : ℤ := x^2 - 2 * y + 1

theorem star_k (k : ℤ) : star k (star k k) = -k^2 + 4 * k - 1 :=
by 
  sorry

end star_k_l158_158071


namespace exists_common_point_l158_158374

-- Definitions: Rectangle and the problem conditions
structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(h_valid : x_min ≤ x_max ∧ y_min ≤ y_max)

def rectangles_intersect (R1 R2 : Rectangle) : Prop :=
¬(R1.x_max < R2.x_min ∨ R2.x_max < R1.x_min ∨ R1.y_max < R2.y_min ∨ R2.y_max < R1.y_min)

def all_rectangles_intersect (rects : List Rectangle) : Prop :=
∀ (R1 R2 : Rectangle), R1 ∈ rects → R2 ∈ rects → rectangles_intersect R1 R2

-- Theorem: Existence of a common point
theorem exists_common_point (rects : List Rectangle) (h_intersect : all_rectangles_intersect rects) : 
  ∃ (T : ℝ × ℝ), ∀ (R : Rectangle), R ∈ rects → 
    R.x_min ≤ T.1 ∧ T.1 ≤ R.x_max ∧ 
    R.y_min ≤ T.2 ∧ T.2 ≤ R.y_max := 
sorry

end exists_common_point_l158_158374


namespace log_relationship_l158_158927

theorem log_relationship (a b : ℝ) (x : ℝ) (h₁ : 6 * (Real.log (x) / Real.log (a)) ^ 2 + 5 * (Real.log (x) / Real.log (b)) ^ 2 = 12 * (Real.log (x) ^ 2) / (Real.log (a) * Real.log (b))) :
  a = b^(5/3) ∨ a = b^(3/5) := by
  sorry

end log_relationship_l158_158927


namespace johns_raise_percentage_increase_l158_158988

def initial_earnings : ℚ := 65
def new_earnings : ℚ := 70
def percentage_increase (initial new : ℚ) : ℚ := ((new - initial) / initial) * 100

theorem johns_raise_percentage_increase : percentage_increase initial_earnings new_earnings = 7.692307692 :=
by
  sorry

end johns_raise_percentage_increase_l158_158988


namespace tetrahedron_volume_formula_l158_158905

-- Definitions used directly in the conditions
variable (a b d : ℝ) (φ : ℝ)

-- Tetrahedron volume formula theorem statement
theorem tetrahedron_volume_formula 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b) 
  (hd_pos : 0 < d) 
  (hφ_pos : 0 < φ) 
  (hφ_le_pi : φ ≤ Real.pi) :
  (∀ V : ℝ, V = 1 / 6 * a * b * d * Real.sin φ) :=
sorry

end tetrahedron_volume_formula_l158_158905


namespace kimberly_initial_skittles_l158_158704

theorem kimberly_initial_skittles (total new initial : ℕ) (h1 : total = 12) (h2 : new = 7) (h3 : total = initial + new) : initial = 5 :=
by {
  -- Using the given conditions to form the proof
  sorry
}

end kimberly_initial_skittles_l158_158704


namespace total_lines_to_write_l158_158641

theorem total_lines_to_write (lines_per_page pages_needed : ℕ) (h1 : lines_per_page = 30) (h2 : pages_needed = 5) : lines_per_page * pages_needed = 150 :=
by {
  sorry
}

end total_lines_to_write_l158_158641


namespace min_sum_log_geq_four_l158_158706

theorem min_sum_log_geq_four (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (hlog : Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4) : 
  m + n ≥ 18 :=
sorry

end min_sum_log_geq_four_l158_158706


namespace lion_king_box_office_earnings_l158_158558

-- Definitions and conditions
def cost_lion_king : ℕ := 10  -- Lion King cost 10 million
def cost_star_wars : ℕ := 25  -- Star Wars cost 25 million
def earnings_star_wars : ℕ := 405  -- Star Wars earned 405 million

-- Calculate profit of Star Wars
def profit_star_wars : ℕ := earnings_star_wars - cost_star_wars

-- Define the profit of The Lion King, given it's half of Star Wars' profit
def profit_lion_king : ℕ := profit_star_wars / 2

-- Calculate the earnings of The Lion King
def earnings_lion_king : ℕ := cost_lion_king + profit_lion_king

-- Theorem to prove
theorem lion_king_box_office_earnings : earnings_lion_king = 200 :=
by
  sorry

end lion_king_box_office_earnings_l158_158558


namespace ellipse_equation_point_M_exists_l158_158112

-- Condition: Point (1, sqrt(2)/2) lies on the ellipse
def point_lies_on_ellipse (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (a_gt_b : a > b) : Prop :=
  (1, Real.sqrt 2 / 2).fst^2 / a^2 + (1, Real.sqrt 2 / 2).snd^2 / b^2 = 1

-- Condition: Eccentricity of the ellipse is sqrt(2)/2
def eccentricity_condition (a b : ℝ) (c : ℝ) : Prop :=
  c / a = Real.sqrt 2 / 2 ∧ a^2 = b^2 + c^2

-- Question (I): Equation of ellipse should be (x^2 / 2 + y^2 = 1)
theorem ellipse_equation (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
    (a_gt_b : a > b) (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : a = Real.sqrt 2 ∧ b = 1 := 
sorry

-- Question (II): There exists M such that MA · MB is constant
theorem point_M_exists (a b c x0 : ℝ)
    (a_pos : 0 < a) (b_pos : 0 < b) (a_gt_b : a > b) 
    (a_val : a = Real.sqrt 2) (b_val : b = 1) 
    (h : point_lies_on_ellipse a b a_pos b_pos a_gt_b)
    (h_ecc : eccentricity_condition a b c) : 
    ∃ (M : ℝ × ℝ), M.fst = 5 / 4 ∧ M.snd = 0 ∧ -7 / 16 = -7 / 16 := 
sorry

end ellipse_equation_point_M_exists_l158_158112


namespace sum_first_n_terms_arithmetic_sequence_eq_l158_158156

open Nat

noncomputable def sum_arithmetic_sequence (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if h: n = 0 then 0 else n * a₁ + (n * (n - 1) * d) / 2

theorem sum_first_n_terms_arithmetic_sequence_eq 
  (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) 
  (h₀ : d ≠ 0)
  (h₁ : a₁ = 4)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₆ = a₁ + 5 * d)
  (h₄ : a₃^2 = a₁ * a₆) :
  sum_arithmetic_sequence a₁ a₃ a₆ d n = (n^2 + 7 * n) / 2 := 
by
  sorry

end sum_first_n_terms_arithmetic_sequence_eq_l158_158156


namespace sneaker_final_price_l158_158310

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l158_158310


namespace complement_of_A_in_U_l158_158564

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}
def complement_set (U A : Set ℕ) : Set ℕ := U \ A

theorem complement_of_A_in_U :
  complement_set U A = {2, 3, 5} :=
by
  apply Set.ext
  intro x
  simp [complement_set, U, A]
  sorry

end complement_of_A_in_U_l158_158564


namespace cost_of_machines_max_type_A_machines_l158_158977

-- Defining the cost equations for type A and type B machines
theorem cost_of_machines (x y : ℝ) (h1 : 3 * x + 2 * y = 31) (h2 : x - y = 2) : x = 7 ∧ y = 5 :=
sorry

-- Defining the budget constraint and computing the maximum number of type A machines purchasable
theorem max_type_A_machines (m : ℕ) (h : 7 * m + 5 * (6 - m) ≤ 34) : m ≤ 2 :=
sorry

end cost_of_machines_max_type_A_machines_l158_158977


namespace path_length_l158_158003

theorem path_length (scale_ratio : ℕ) (map_path_length : ℝ) 
  (h1 : scale_ratio = 500)
  (h2 : map_path_length = 3.5) : 
  (map_path_length * scale_ratio = 1750) :=
sorry

end path_length_l158_158003


namespace pet_food_total_weight_l158_158963

theorem pet_food_total_weight:
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3 -- pounds
  let dog_food_bags := 4 
  let weight_per_dog_food_bag := 5 -- pounds
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2 -- pounds
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  total_weight_ounces = 624 :=
by
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3
  let dog_food_bags := 4
  let weight_per_dog_food_bag := 5
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  show total_weight_ounces = 624
  sorry

end pet_food_total_weight_l158_158963


namespace isosceles_triangle_perimeter_l158_158210

variable (a b c : ℝ)
variable (h1 : a = 4 ∨ a = 8)
variable (h2 : b = 4 ∨ b = 8)
variable (h3 : a = b ∨ c = 8)

theorem isosceles_triangle_perimeter (h : a + b + c = 20) : a = b ∨ b = 8 ∧ (a = 8 ∧ c = 4 ∨ b = c) := 
  by
  sorry

end isosceles_triangle_perimeter_l158_158210


namespace least_value_expression_l158_158306

open Real

theorem least_value_expression (x : ℝ) : 
  let expr := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * cos (2 * x)
  ∃ a : ℝ, expr = a ∧ ∀ b : ℝ, b < a → False :=
sorry

end least_value_expression_l158_158306


namespace unique_solution_abs_eq_l158_158376

theorem unique_solution_abs_eq (x : ℝ) : (|x - 9| = |x + 3| + 2) ↔ x = 2 :=
by
  sorry

end unique_solution_abs_eq_l158_158376


namespace find_cookies_on_second_plate_l158_158600

theorem find_cookies_on_second_plate (a : ℕ → ℕ) :
  (a 1 = 5) ∧ (a 3 = 10) ∧ (a 4 = 14) ∧ (a 5 = 19) ∧ (a 6 = 25) ∧
  (∀ n, a (n + 2) - a (n + 1) = if (n + 1) % 2 = 0 then 5 else 4) →
  a 2 = 5 :=
by
  sorry

end find_cookies_on_second_plate_l158_158600


namespace smith_trip_times_same_l158_158799

theorem smith_trip_times_same (v : ℝ) (hv : v > 0) : 
  let t1 := 80 / v 
  let t2 := 160 / (2 * v) 
  t1 = t2 :=
by
  sorry

end smith_trip_times_same_l158_158799


namespace gardener_cabbages_l158_158186

theorem gardener_cabbages (area_this_year : ℕ) (side_length_this_year : ℕ) (side_length_last_year : ℕ) (area_last_year : ℕ) (additional_cabbages : ℕ) :
  area_this_year = 9801 →
  side_length_this_year = 99 →
  side_length_last_year = side_length_this_year - 1 →
  area_last_year = side_length_last_year * side_length_last_year →
  additional_cabbages = area_this_year - area_last_year →
  additional_cabbages = 197 :=
by
  sorry

end gardener_cabbages_l158_158186


namespace range_of_k_l158_158981

theorem range_of_k (k : ℝ) :
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l158_158981


namespace trig_inequality_l158_158621

open Real

theorem trig_inequality (x : ℝ) (n m : ℕ) (hx : 0 < x ∧ x < π / 2) (hnm : n > m) : 
  2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) := 
sorry

end trig_inequality_l158_158621


namespace strap_pieces_l158_158933

/-
  Given the conditions:
  1. The sum of the lengths of the two straps is 64 cm.
  2. The longer strap is 48 cm longer than the shorter strap.
  
  Prove that the number of pieces of strap that equal the length of the shorter strap 
  that can be cut from the longer strap is 7.
-/

theorem strap_pieces (S L : ℕ) (h1 : S + L = 64) (h2 : L = S + 48) :
  L / S = 7 :=
by
  sorry

end strap_pieces_l158_158933


namespace rectangular_solid_surface_area_l158_158072

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hvol : a * b * c = 455) : 
  let surface_area := 2 * (a * b + b * c + c * a)
  surface_area = 382 := by
-- proof
sorry

end rectangular_solid_surface_area_l158_158072


namespace sum_of_squares_99_in_distinct_ways_l158_158596

theorem sum_of_squares_99_in_distinct_ways : 
  ∃ a b c d e f g h i j k l : ℕ, 
    (a^2 + b^2 + c^2 + d^2 = 99) ∧ (e^2 + f^2 + g^2 + h^2 = 99) ∧ (i^2 + j^2 + k^2 + l^2 = 99) ∧ 
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) ∧ 
    (a ≠ i ∨ b ≠ j ∨ c ≠ k ∨ d ≠ l) ∧ 
    (i ≠ e ∨ j ≠ f ∨ k ≠ g ∨ l ≠ h) 
    :=
sorry

end sum_of_squares_99_in_distinct_ways_l158_158596


namespace area_difference_equal_28_5_l158_158870

noncomputable def square_side_length (d: ℝ) : ℝ := d / Real.sqrt 2
noncomputable def square_area (d: ℝ) : ℝ := (square_side_length d) ^ 2
noncomputable def circle_radius (D: ℝ) : ℝ := D / 2
noncomputable def circle_area (D: ℝ) : ℝ := Real.pi * (circle_radius D) ^ 2
noncomputable def area_difference (d D : ℝ) : ℝ := |circle_area D - square_area d|

theorem area_difference_equal_28_5 :
  ∀ (d D : ℝ), d = 10 → D = 10 → area_difference d D = 28.5 :=
by
  intros d D hd hD
  rw [hd, hD]
  -- Remaining steps involve computing the known areas and their differences
  sorry

end area_difference_equal_28_5_l158_158870


namespace cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l158_158216

/-- A 4x4 chessboard is entirely white except for one square which is black.
The allowed operations are flipping the colors of all squares in a column or in a row.
Prove that it is impossible to have all the squares the same color regardless of the position of the black square. -/
theorem cannot_all_white_without_diagonals :
  ∀ (i j : Fin 4), False :=
by sorry

/-- If diagonal flips are also allowed, prove that 
it is impossible to have all squares the same color if the black square is at certain positions. -/
theorem cannot_all_white_with_diagonals :
  ∀ (i j : Fin 4), (i, j) ≠ (0, 1) ∧ (i, j) ≠ (0, 2) ∧
                   (i, j) ≠ (1, 0) ∧ (i, j) ≠ (1, 3) ∧
                   (i, j) ≠ (2, 0) ∧ (i, j) ≠ (2, 3) ∧
                   (i, j) ≠ (3, 1) ∧ (i, j) ≠ (3, 2) → False :=
by sorry

end cannot_all_white_without_diagonals_cannot_all_white_with_diagonals_l158_158216


namespace polygon_interior_sum_sum_of_exterior_angles_l158_158068

theorem polygon_interior_sum (n : ℕ) (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

theorem sum_of_exterior_angles (n : ℕ) : 360 = 360 :=
by
  sorry

end polygon_interior_sum_sum_of_exterior_angles_l158_158068


namespace min_radius_circle_condition_l158_158170

theorem min_radius_circle_condition (r : ℝ) (a b : ℝ) 
    (h_circle : (a - (r + 1))^2 + b^2 = r^2)
    (h_condition : b^2 ≥ 4 * a) :
    r ≥ 4 := 
sorry

end min_radius_circle_condition_l158_158170


namespace adults_tickets_sold_eq_1200_l158_158842

variable (A : ℕ)
variable (S : ℕ := 300) -- Number of student tickets
variable (P_adult : ℕ := 12) -- Price per adult ticket
variable (P_student : ℕ := 6) -- Price per student ticket
variable (total_tickets : ℕ := 1500) -- Total tickets sold
variable (total_amount : ℕ := 16200) -- Total amount collected

theorem adults_tickets_sold_eq_1200
  (h1 : S = 300)
  (h2 : A + S = total_tickets)
  (h3 : P_adult * A + P_student * S = total_amount) :
  A = 1200 := by
  sorry

end adults_tickets_sold_eq_1200_l158_158842


namespace problem1_l158_158066

theorem problem1 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2 * α) + Real.cos α ^ 2 = 3 / 2 := 
sorry

end problem1_l158_158066


namespace find_initial_oranges_l158_158250

variable (O : ℕ)
variable (reserved_fraction : ℚ := 1 / 4)
variable (sold_fraction : ℚ := 3 / 7)
variable (rotten_oranges : ℕ := 4)
variable (good_oranges_today : ℕ := 32)

-- Define the total oranges before finding the rotten oranges
def oranges_before_rotten := good_oranges_today + rotten_oranges

-- Define the remaining fraction of oranges after reserving for friends and selling some
def remaining_fraction := (1 - reserved_fraction) * (1 - sold_fraction)

-- State the theorem to be proven
theorem find_initial_oranges (h : remaining_fraction * O = oranges_before_rotten) : O = 84 :=
sorry

end find_initial_oranges_l158_158250


namespace find_z_when_y_is_6_l158_158096

variable {y z : ℚ}

/-- Condition: y^4 varies inversely with √[4]{z}. -/
def inverse_variation (k : ℚ) (y z : ℚ) : Prop :=
  y^4 * z^(1/4) = k

/-- Given constant k based on y = 3 and z = 16. -/
def k_value : ℚ := 162

theorem find_z_when_y_is_6
  (h_inv : inverse_variation k_value 3 16)
  (h_y : y = 6) :
  z = 1 / 4096 := 
sorry

end find_z_when_y_is_6_l158_158096


namespace determine_b_l158_158411

theorem determine_b (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x) ∧ (∀ x y : ℝ, y - 2 = (b + 9) * x) → 
  b = -6 :=
by
  sorry

end determine_b_l158_158411


namespace ages_correct_l158_158177

def ages : List ℕ := [5, 8, 13, 15]
def Tanya : ℕ := 13
def Yura : ℕ := 8
def Sveta : ℕ := 5
def Lena : ℕ := 15

theorem ages_correct (h1 : Tanya ∈ ages) 
                     (h2: Yura ∈ ages)
                     (h3: Sveta ∈ ages)
                     (h4: Lena ∈ ages)
                     (h5: Tanya ≠ Yura)
                     (h6: Tanya ≠ Sveta)
                     (h7: Tanya ≠ Lena)
                     (h8: Yura ≠ Sveta)
                     (h9: Yura ≠ Lena)
                     (h10: Sveta ≠ Lena)
                     (h11: Sveta = 5)
                     (h12: Tanya > Yura)
                     (h13: (Tanya + Sveta) % 3 = 0) :
                     Tanya = 13 ∧ Yura = 8 ∧ Sveta = 5 ∧ Lena = 15 := by
  sorry

end ages_correct_l158_158177


namespace solve_system_l158_158950

def x : ℚ := 2.7 / 13
def y : ℚ := 1.0769

theorem solve_system :
  (∃ (x' y' : ℚ), 4 * x' - 3 * y' = -2.4 ∧ 5 * x' + 6 * y' = 7.5) ↔
  (x = 2.7 / 13 ∧ y = 1.0769) :=
by
  sorry

end solve_system_l158_158950


namespace large_block_volume_l158_158803

theorem large_block_volume (W D L : ℝ) (h1 : W * D * L = 3) : 
  (2 * W) * (2 * D) * (3 * L) = 36 := 
by 
  sorry

end large_block_volume_l158_158803


namespace correct_average_l158_158065

theorem correct_average (n : ℕ) (average incorrect correct : ℕ) (h1 : n = 10) (h2 : average = 15) 
(h3 : incorrect = 26) (h4 : correct = 36) :
  (n * average - incorrect + correct) / n = 16 :=
  sorry

end correct_average_l158_158065


namespace cheaper_to_buy_more_books_l158_158040

def C (n : ℕ) : ℕ :=
  if n < 1 then 0
  else if n ≤ 20 then 15 * n
  else if n ≤ 40 then 14 * n - 5
  else 13 * n

noncomputable def apply_discount (n : ℕ) (cost : ℕ) : ℕ :=
  cost - 10 * (n / 10)

theorem cheaper_to_buy_more_books : 
  ∃ (n_vals : Finset ℕ), n_vals.card = 5 ∧ ∀ n ∈ n_vals, apply_discount (n + 1) (C (n + 1)) < apply_discount n (C n) :=
sorry

end cheaper_to_buy_more_books_l158_158040


namespace ratio_of_arithmetic_sequence_sums_l158_158757

theorem ratio_of_arithmetic_sequence_sums :
  let a1 := 2
  let d1 := 3
  let l1 := 41
  let n1 := (l1 - a1) / d1 + 1
  let sum1 := n1 / 2 * (a1 + l1)

  let a2 := 4
  let d2 := 4
  let l2 := 60
  let n2 := (l2 - a2) / d2 + 1
  let sum2 := n2 / 2 * (a2 + l2)
  sum1 / sum2 = 301 / 480 :=
by
  sorry

end ratio_of_arithmetic_sequence_sums_l158_158757


namespace distinct_sums_l158_158610

theorem distinct_sums (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 :=
by
  sorry

end distinct_sums_l158_158610


namespace no_two_distinct_real_roots_l158_158798

-- Definitions of the conditions and question in Lean 4
theorem no_two_distinct_real_roots (a : ℝ) (h : a ≥ 1) : ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*x1 + a = 0) ∧ (x2^2 - 2*x2 + a = 0) :=
sorry

end no_two_distinct_real_roots_l158_158798


namespace find_vertex_l158_158296

noncomputable def parabola_vertex (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

theorem find_vertex :
  ∃ (x y : ℝ), parabola_vertex x y ∧ x = -14/3 ∧ y = -2 :=
by
  sorry

end find_vertex_l158_158296


namespace compute_expression_l158_158732

theorem compute_expression :
  2 * 2^5 - 8^58 / 8^56 = 0 := by
  sorry

end compute_expression_l158_158732


namespace chord_ratio_l158_158426

variable (XQ WQ YQ ZQ : ℝ)

theorem chord_ratio (h1 : XQ = 5) (h2 : WQ = 7) (h3 : XQ * YQ = WQ * ZQ) : YQ / ZQ = 7 / 5 :=
by
  sorry

end chord_ratio_l158_158426


namespace addition_neg3_plus_2_multiplication_neg3_times_2_l158_158079

theorem addition_neg3_plus_2 : -3 + 2 = -1 :=
  by
    sorry

theorem multiplication_neg3_times_2 : (-3) * 2 = -6 :=
  by
    sorry

end addition_neg3_plus_2_multiplication_neg3_times_2_l158_158079


namespace base9_number_perfect_square_l158_158578

theorem base9_number_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : 0 ≤ d ∧ d ≤ 8) (n : ℕ) 
  (h3 : n = 729 * a + 81 * b + 45 + d) (h4 : ∃ k : ℕ, k * k = n) : d = 0 := 
sorry

end base9_number_perfect_square_l158_158578


namespace tunnel_length_l158_158117

theorem tunnel_length (L L_1 L_2 v v_new t t_new : ℝ) (H1: L_1 = 6) (H2: L_2 = 12) 
  (H3: v_new = 0.8 * v) (H4: t = (L + L_1) / v) (H5: t_new = 1.5 * t)
  (H6: t_new = (L + L_2) / v_new) : 
  L = 24 :=
by
  sorry

end tunnel_length_l158_158117


namespace largest_angle_in_triangle_l158_158010

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 3 * b + 3 * c = a ^ 2) (h2 : a + 3 * b - 3 * c = -4) 
  (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : a + b > c) (h7 : a + c > b) (h8 : b + c > a) : 
  ∃ C : ℝ, C = 120 ∧ (by exact sorry) := sorry

end largest_angle_in_triangle_l158_158010


namespace smallest_positive_angle_l158_158823

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ (6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 / 2) ∧ y = 22.5 :=
by
  sorry

end smallest_positive_angle_l158_158823


namespace log_xy_eq_5_over_11_l158_158672

-- Definitions of the conditions
axiom log_xy4_eq_one {x y : ℝ} : Real.log (x * y^4) = 1
axiom log_x3y_eq_one {x y : ℝ} : Real.log (x^3 * y) = 1

-- The statement to be proven
theorem log_xy_eq_5_over_11 {x y : ℝ} (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 :=
by
  sorry

end log_xy_eq_5_over_11_l158_158672


namespace sum_of_prime_factors_172944_l158_158182

theorem sum_of_prime_factors_172944 : 
  (∃ (a b c : ℕ), 2^a * 3^b * 1201^c = 172944 ∧ a = 4 ∧ b = 2 ∧ c = 1) → 2 + 3 + 1201 = 1206 := 
by 
  intros h 
  exact sorry

end sum_of_prime_factors_172944_l158_158182


namespace james_hours_per_day_l158_158427

theorem james_hours_per_day (h : ℕ) (rental_rate : ℕ) (days_per_week : ℕ) (weekly_income : ℕ)
  (H1 : rental_rate = 20)
  (H2 : days_per_week = 4)
  (H3 : weekly_income = 640)
  (H4 : rental_rate * days_per_week * h = weekly_income) :
  h = 8 :=
sorry

end james_hours_per_day_l158_158427


namespace probability_correct_l158_158266

/-
  Problem statement:
  Consider a modified city map where a student walks from intersection A to intersection B, passing through C and D.
  The student always walks east or south and at each intersection, decides the direction to go with a probability of 1/2.
  The map requires 4 eastward and 3 southward moves to reach B from A. C is 2 east, 1 south move from A. D is 3 east, 2 south moves from A.
  Prove that the probability the student goes through both C and D is 12/35.
-/

noncomputable def probability_passing_C_and_D : ℚ :=
  let total_paths_A_to_B := Nat.choose 7 4
  let paths_A_to_C := Nat.choose 3 2
  let paths_C_to_D := Nat.choose 2 1
  let paths_D_to_B := Nat.choose 2 1
  (paths_A_to_C * paths_C_to_D * paths_D_to_B) / total_paths_A_to_B

theorem probability_correct :
  probability_passing_C_and_D = 12 / 35 :=
by
  sorry

end probability_correct_l158_158266


namespace largest_root_in_range_l158_158204

-- Define the conditions for the equation parameters
variables (a0 a1 a2 : ℝ)
-- Define the conditions for the absolute value constraints
variables (h0 : |a0| < 2) (h1 : |a1| < 2) (h2 : |a2| < 2)

-- Define the equation
def cubic_equation (x : ℝ) : ℝ := x^3 + a2 * x^2 + a1 * x + a0

-- Define the property we want to prove about the largest positive root r
theorem largest_root_in_range :
  ∃ r > 0, (∃ x, cubic_equation a0 a1 a2 x = 0 ∧ r = x) ∧ (5 / 2 < r ∧ r < 3) :=
by sorry

end largest_root_in_range_l158_158204


namespace find_income_l158_158972

-- Define the conditions
def income_and_expenditure (income expenditure : ℕ) : Prop :=
  5 * expenditure = 3 * income

def savings (income expenditure : ℕ) (saving : ℕ) : Prop :=
  income - expenditure = saving

-- State the theorem
theorem find_income (expenditure : ℕ) (saving : ℕ) (h1 : income_and_expenditure 5 3) (h2 : savings (5 * expenditure) (3 * expenditure) saving) :
  5 * expenditure = 10000 :=
by
  -- Use the provided hint or conditions
  sorry

end find_income_l158_158972


namespace final_balance_is_60_million_l158_158285

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l158_158285


namespace total_rainfall_january_l158_158511

theorem total_rainfall_january (R1 R2 T : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 21) : T = 35 :=
by 
  let R1 := 14
  let R2 := 21
  let T := R1 + R2
  sorry

end total_rainfall_january_l158_158511


namespace problem_min_a2_area_l158_158976

noncomputable def area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

noncomputable def min_a2_area (a b c : ℝ) (A B C : ℝ): ℝ := 
  let S := area a b c A B C
  a^2 / S

theorem problem_min_a2_area :
  ∀ (a b c A B C : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    A + B + C = Real.pi →
    a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C →
    b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
    min_a2_area a b c A B C ≥ 2 * Real.sqrt 2 :=
by
  sorry

end problem_min_a2_area_l158_158976


namespace Alex_runs_faster_l158_158308

def Rick_speed : ℚ := 5
def Jen_speed : ℚ := (3 / 4) * Rick_speed
def Mark_speed : ℚ := (4 / 3) * Jen_speed
def Alex_speed : ℚ := (5 / 6) * Mark_speed

theorem Alex_runs_faster : Alex_speed = 25 / 6 :=
by
  -- Proof is skipped
  sorry

end Alex_runs_faster_l158_158308


namespace july_percentage_is_correct_l158_158237

def total_scientists : ℕ := 120
def july_scientists : ℕ := 16
def july_percentage : ℚ := (july_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem july_percentage_is_correct : july_percentage = 13.33 := 
by 
  -- Provides the proof directly as a statement
  sorry

end july_percentage_is_correct_l158_158237


namespace problem_1_problem_2_problem_3_l158_158168

theorem problem_1 : 
  ∀ x : ℝ, x^2 - 2 * x + 5 = (x - 1)^2 + 4 := 
sorry

theorem problem_2 (n : ℝ) (h : ∀ x : ℝ, x^2 + 2 * n * x + 3 = (x + 5)^2 - 25 + 3) : 
  n = -5 := 
sorry

theorem problem_3 (a : ℝ) (h : ∀ x : ℝ, (x^2 + 6 * x + 9) * (x^2 - 4 * x + 4) = ((x + a)^2 + b)^2) : 
  a = -1/2 := 
sorry

end problem_1_problem_2_problem_3_l158_158168


namespace least_number_of_homeowners_l158_158770

theorem least_number_of_homeowners (total_members : ℕ) 
(num_men : ℕ) (num_women : ℕ) 
(homeowners_men : ℕ) (homeowners_women : ℕ) 
(h_total : total_members = 5000)
(h_men_women : num_men + num_women = total_members) 
(h_percentage_men : homeowners_men = 15 * num_men / 100)
(h_percentage_women : homeowners_women = 25 * num_women / 100):
  homeowners_men + homeowners_women = 4 :=
sorry

end least_number_of_homeowners_l158_158770


namespace sin_A_value_l158_158194

theorem sin_A_value
  (f : ℝ → ℝ)
  (cos_B : ℝ)
  (f_C_div_2 : ℝ)
  (C_acute : Prop) :
  (∀ x, f x = Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2) →
  cos_B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (0 < C ∧ C < Real.pi / 2) →
  Real.sin (Real.arcsin (Real.sqrt 3 / 2) + Real.arcsin (2 * Real.sqrt 2 / 3)) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by
  intros
  sorry

end sin_A_value_l158_158194


namespace maria_correct_answers_l158_158284

theorem maria_correct_answers (x : ℕ) (n c d s : ℕ) (h1 : n = 30) (h2 : c = 20) (h3 : d = 5) (h4 : s = 325)
  (h5 : n = x + (n - x)) : 20 * x - 5 * (30 - x) = 325 → x = 19 :=
by 
  intros h_eq
  sorry

end maria_correct_answers_l158_158284


namespace sin_double_angle_given_cos_identity_l158_158277

theorem sin_double_angle_given_cos_identity (α : ℝ) 
  (h : Real.cos (α + π / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 :=
by
  sorry

end sin_double_angle_given_cos_identity_l158_158277


namespace regular_hexagon_area_l158_158802

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l158_158802


namespace moles_of_Cl2_required_l158_158232

theorem moles_of_Cl2_required (n_C2H6 n_HCl : ℕ) (balance : n_C2H6 = 3) (HCl_needed : n_HCl = 6) :
  ∃ n_Cl2 : ℕ, n_Cl2 = 9 :=
by
  sorry

end moles_of_Cl2_required_l158_158232


namespace cos_double_angle_identity_l158_158579

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_double_angle_identity_l158_158579


namespace find_angle_B_l158_158366

noncomputable def angle_B (A B C a b c : ℝ): Prop := 
  a * Real.cos B - b * Real.cos A = b ∧ 
  C = Real.pi / 5

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B A B C a b c) : 
  B = 4 * Real.pi / 15 :=
by
  sorry

end find_angle_B_l158_158366


namespace total_tennis_balls_used_l158_158664

theorem total_tennis_balls_used 
  (round1_games : Nat := 8) 
  (round2_games : Nat := 4) 
  (round3_games : Nat := 2) 
  (finals_games : Nat := 1)
  (cans_per_game : Nat := 5) 
  (balls_per_can : Nat := 3) : 

  3 * (5 * (8 + 4 + 2 + 1)) = 225 := 
by
  sorry

end total_tennis_balls_used_l158_158664


namespace parallel_lines_m_l158_158188

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 1 = 0 → 6 ≠ 0) ∧ 
  (∀ x y : ℝ, m * x + 6 * y - 5 = 0 → 6 ≠ 0) → 
  m = 4 :=
by
  intro h
  sorry

end parallel_lines_m_l158_158188


namespace sara_schavenger_hunt_l158_158698

theorem sara_schavenger_hunt :
  let monday := 1 -- Sara rearranges the books herself
  let tuesday := 2 -- Sara can choose from Liam or Mia
  let wednesday := 4 -- There are 4 classmates
  let thursday := 3 -- There are 3 new volunteers
  let friday := 1 -- Sara and Zoe do it together
  monday * tuesday * wednesday * thursday * friday = 24 :=
by
  sorry

end sara_schavenger_hunt_l158_158698


namespace four_digit_numbers_with_property_l158_158127

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l158_158127


namespace rectangle_color_invariance_l158_158062

/-- A theorem stating that in any 3x7 rectangle with some cells colored black at random, there necessarily exist four cells of the same color, whose centers are the vertices of a rectangle with sides parallel to the sides of the original rectangle. -/
theorem rectangle_color_invariance :
  ∀ (color : Fin 3 × Fin 7 → Bool), 
  ∃ i1 i2 j1 j2 : Fin 3, i1 < i2 ∧ j1 < j2 ∧ 
  color ⟨i1, j1⟩ = color ⟨i1, j2⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j1⟩ ∧ 
  color ⟨i1, j1⟩ = color ⟨i2, j2⟩ :=
by
  -- The proof is omitted
  sorry

end rectangle_color_invariance_l158_158062


namespace bank_robbery_car_l158_158796

def car_statement (make color : String) : Prop :=
  (make = "Buick" ∨ color = "blue") ∧
  (make = "Chrysler" ∨ color = "black") ∧
  (make = "Ford" ∨ color ≠ "blue")

theorem bank_robbery_car : ∃ make color : String, car_statement make color ∧ make = "Buick" ∧ color = "black" :=
by
  sorry

end bank_robbery_car_l158_158796


namespace lawrence_worked_hours_l158_158965

-- Let h_M, h_T, h_F be the hours worked on Monday, Tuesday, and Friday respectively
-- Let h_W be the hours worked on Wednesday (h_W = 5.5)
-- Let h_R be the hours worked on Thursday (h_R = 5.5)
-- Let total hours worked in 5 days be 25
-- Prove that h_M + h_T + h_F = 14

theorem lawrence_worked_hours :
  ∀ (h_M h_T h_F : ℝ), h_W = 5.5 → h_R = 5.5 → (5 * 5 = 25) → 
  h_M + h_T + h_F + h_W + h_R = 25 → h_M + h_T + h_F = 14 :=
by
  intros h_M h_T h_F h_W h_R h_total h_sum
  sorry

end lawrence_worked_hours_l158_158965


namespace exists_f_with_f3_eq_9_forall_f_f3_le_9_l158_158951

-- Define the real-valued function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_real : ∀ x : ℝ, true)  -- f is real-valued and defined for all real numbers
variable (f_mul : ∀ x y : ℝ, f (x * y) = f x * f y)  -- f(xy) = f(x)f(y)
variable (f_add : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))  -- f(x+y) ≤ 2(f(x) + f(y))
variable (f_2 : f 2 = 4)  -- f(2) = 4

-- Part a
theorem exists_f_with_f3_eq_9 : ∃ f : ℝ → ℝ, (∀ x : ℝ, true) ∧ 
                              (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
                              (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) ∧ 
                              (f 2 = 4) ∧ 
                              (f 3 = 9) := 
sorry

-- Part b
theorem forall_f_f3_le_9 : ∀ f : ℝ → ℝ, 
                        (∀ x : ℝ, true) → 
                        (∀ x y : ℝ, f (x * y) = f x * f y) → 
                        (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) → 
                        (f 2 = 4) → 
                        (f 3 ≤ 9) := 
sorry

end exists_f_with_f3_eq_9_forall_f_f3_le_9_l158_158951


namespace amelia_drove_distance_on_Monday_l158_158715

theorem amelia_drove_distance_on_Monday 
  (total_distance : ℕ) (tuesday_distance : ℕ) (remaining_distance : ℕ)
  (total_distance_eq : total_distance = 8205) 
  (tuesday_distance_eq : tuesday_distance = 582) 
  (remaining_distance_eq : remaining_distance = 6716) :
  ∃ x : ℕ, x + tuesday_distance + remaining_distance = total_distance ∧ x = 907 :=
by
  sorry

end amelia_drove_distance_on_Monday_l158_158715


namespace max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l158_158392

-- Definitions and conditions related to the given problem
def unit_circle (r : ℝ) : Prop := r = 1

-- Maximum number of non-intersecting circles of radius 1 tangent to a unit circle.
theorem max_non_intersecting_circles_tangent (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 6 := sorry

-- Maximum number of circles of radius 1 intersecting a given unit circle without intersecting centers.
theorem max_intersecting_circles_without_center_containment (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 12 := sorry

-- Maximum number of circles of radius 1 intersecting a unit circle K without containing the center of K or any other circle's center.
theorem max_intersecting_circles_without_center_containment_2 (r : ℝ) (K : ℝ)
  (h_r : unit_circle r) (h_K : unit_circle K) :
  ∃ n, n = 18 := sorry

end max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l158_158392


namespace find_n_l158_158101

theorem find_n :
  ∀ (n : ℕ),
    2^200 * 2^203 + 2^163 * 2^241 + 2^126 * 2^277 = 32^n →
    n = 81 :=
by
  intros n h
  sorry

end find_n_l158_158101


namespace train_passing_time_l158_158970

noncomputable def length_of_train : ℝ := 450
noncomputable def speed_kmh : ℝ := 80
noncomputable def length_of_station : ℝ := 300
noncomputable def speed_m_per_s : ℝ := speed_kmh * 1000 / 3600 -- Convert km/hour to m/second
noncomputable def total_distance : ℝ := length_of_train + length_of_station
noncomputable def passing_time : ℝ := total_distance / speed_m_per_s

theorem train_passing_time : abs (passing_time - 33.75) < 0.01 :=
by
  sorry

end train_passing_time_l158_158970


namespace total_growing_space_l158_158049

noncomputable def garden_area : ℕ :=
  let area_3x3 := 3 * 3
  let total_area_3x3 := 2 * area_3x3
  let area_4x3 := 4 * 3
  let total_area_4x3 := 2 * area_4x3
  total_area_3x3 + total_area_4x3

theorem total_growing_space : garden_area = 42 :=
by
  sorry

end total_growing_space_l158_158049


namespace find_second_number_l158_158169

-- Define the two numbers A and B
variables (A B : ℝ)

-- Define the conditions
def condition1 := 0.20 * A = 0.30 * B + 80
def condition2 := A = 580

-- Define the goal
theorem find_second_number (h1 : condition1 A B) (h2 : condition2 A) : B = 120 :=
by sorry

end find_second_number_l158_158169


namespace volume_of_stone_l158_158442

def width := 16
def length := 14
def full_height := 9
def initial_water_height := 4
def final_water_height := 9

def volume_before := length * width * initial_water_height
def volume_after := length * width * final_water_height

def volume_stone := volume_after - volume_before

theorem volume_of_stone : volume_stone = 1120 := by
  unfold volume_stone
  unfold volume_after volume_before
  unfold final_water_height initial_water_height width length
  sorry

end volume_of_stone_l158_158442


namespace parameterized_line_l158_158709

noncomputable def g (t : ℝ) : ℝ := 9 * t + 10

theorem parameterized_line (t : ℝ) :
  let x := g t
  let y := 18 * t - 10
  y = 2 * x - 30 :=
by
  sorry

end parameterized_line_l158_158709


namespace sin_is_odd_l158_158885

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem sin_is_odd : is_odd_function sin :=
by
  sorry

end sin_is_odd_l158_158885


namespace exterior_angle_of_regular_octagon_l158_158073

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon_l158_158073


namespace tank_saltwater_solution_l158_158314

theorem tank_saltwater_solution (x : ℝ) :
  let water1 := 0.75 * x
  let water1_evaporated := (1/3) * water1
  let water2 := water1 - water1_evaporated
  let salt2 := 0.25 * x
  let water3 := water2 + 12
  let salt3 := salt2 + 24
  let step2_eq := (salt3 / (water3 + 24)) = 0.4
  let water4 := water3 - (1/4) * water3
  let salt4 := salt3
  let water5 := water4 + 15
  let salt5 := salt4 + 30
  let step4_eq := (salt5 / (water5 + 30)) = 0.5
  step2_eq ∧ step4_eq → x = 192 :=
by
  sorry

end tank_saltwater_solution_l158_158314


namespace incorrect_expression_l158_158743

variable {x y : ℚ}

theorem incorrect_expression (h : x / y = 5 / 3) : (x - 2 * y) / y ≠ 1 / 3 := by
  have h1 : x / y = 5 / 3 := h
  have h2 : (x - 2 * y) / y = (x / y) - (2 * y) / y := by sorry
  have h3 : (x - 2 * y) / y = (5 / 3) - 2 := by sorry
  have h4 : (x - 2 * y) / y = (5 / 3) - (6 / 3) := by sorry
  have h5 : (x - 2 * y) / y = -1 / 3 := by sorry
  exact sorry

end incorrect_expression_l158_158743


namespace smallest_nonneg_integer_divisible_by_4_l158_158283

theorem smallest_nonneg_integer_divisible_by_4 :
  ∃ n : ℕ, (7 * (n - 3)^5 - n^2 + 16 * n - 30) % 4 = 0 ∧ ∀ m : ℕ, m < n -> (7 * (m - 3)^5 - m^2 + 16 * m - 30) % 4 ≠ 0 :=
by
  use 1
  sorry

end smallest_nonneg_integer_divisible_by_4_l158_158283


namespace minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l158_158267

theorem minute_hand_angle_is_pi_six (radius : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : fast_min = 5) :
  (fast_min / 60 * 2 * Real.pi = Real.pi / 6) :=
by sorry

theorem minute_hand_arc_length_is_2pi_third (radius : ℝ) (angle : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : angle = Real.pi / 6) (h3 : fast_min = 5) :
  (radius * angle = 2 * Real.pi / 3) :=
by sorry

end minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l158_158267


namespace triangle_neg3_4_l158_158599

def triangle (a b : ℚ) : ℚ := -a + b

theorem triangle_neg3_4 : triangle (-3) 4 = 7 := 
by 
  sorry

end triangle_neg3_4_l158_158599


namespace departure_of_30_tons_of_grain_l158_158534

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l158_158534


namespace right_triangle_area_l158_158160

theorem right_triangle_area (hypotenuse : ℝ)
  (angle_deg : ℝ)
  (h_hyp : hypotenuse = 10 * Real.sqrt 2)
  (h_angle : angle_deg = 45) : 
  (1 / 2) * (hypotenuse / Real.sqrt 2)^2 = 50 := 
by 
  sorry

end right_triangle_area_l158_158160


namespace distance_between_lines_l158_158487

-- Define lines l1 and l2
def line_l1 (x y : ℝ) := x + y + 1 = 0
def line_l2 (x y : ℝ) := 2 * x + 2 * y + 3 = 0

-- Proof statement for the distance between parallel lines
theorem distance_between_lines :
  let a := 1
  let b := 1
  let c1 := 1
  let c2 := 3 / 2
  let distance := |c2 - c1| / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 2 / 4 :=
by
  sorry

end distance_between_lines_l158_158487


namespace simplify_expression_l158_158305

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b - 4) - 2 * b^2 = 9 * b^3 + 4 * b^2 - 12 * b :=
by sorry

end simplify_expression_l158_158305


namespace minimum_value_l158_158095

noncomputable def min_value (a b c d : ℝ) : ℝ :=
(a - c) ^ 2 + (b - d) ^ 2

theorem minimum_value (a b c d : ℝ) (hab : a * b = 3) (hcd : c + 3 * d = 0) :
  min_value a b c d ≥ (18 / 5) :=
by
  sorry

end minimum_value_l158_158095


namespace tan_double_angle_l158_158832

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end tan_double_angle_l158_158832


namespace largest_multiple_of_15_less_than_neg_150_l158_158606

theorem largest_multiple_of_15_less_than_neg_150 : ∃ m : ℤ, m % 15 = 0 ∧ m < -150 ∧ (∀ n : ℤ, n % 15 = 0 ∧ n < -150 → n ≤ m) ∧ m = -165 := sorry

end largest_multiple_of_15_less_than_neg_150_l158_158606


namespace line_passes_through_center_l158_158349

theorem line_passes_through_center (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end line_passes_through_center_l158_158349


namespace negation_of_proposition_l158_158632

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l158_158632


namespace sheets_in_stack_l158_158707

theorem sheets_in_stack (thickness_per_500_sheets : ℝ) (stack_height : ℝ) (total_sheets : ℕ) :
  thickness_per_500_sheets = 4 → stack_height = 10 → total_sheets = 1250 :=
by
  intros h1 h2
  -- We will provide the mathematical proof steps here.
  sorry

end sheets_in_stack_l158_158707


namespace store_discount_l158_158818

theorem store_discount (P : ℝ) :
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  P2 = 0.774 * P :=
by
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  sorry

end store_discount_l158_158818


namespace percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l158_158323

theorem percentage_increase_first_job :
  let old_salary := 65
  let new_salary := 70
  (new_salary - old_salary) / old_salary * 100 = 7.69 := by
  sorry

theorem percentage_increase_second_job :
  let old_salary := 120
  let new_salary := 138
  (new_salary - old_salary) / old_salary * 100 = 15 := by
  sorry

theorem percentage_increase_third_job :
  let old_salary := 200
  let new_salary := 220
  (new_salary - old_salary) / old_salary * 100 = 10 := by
  sorry

end percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l158_158323


namespace farm_produce_weeks_l158_158158

def eggs_needed_per_week (saly_eggs ben_eggs ked_eggs : ℕ) : ℕ :=
  saly_eggs + ben_eggs + ked_eggs

def number_of_weeks (total_eggs : ℕ) (weekly_eggs : ℕ) : ℕ :=
  total_eggs / weekly_eggs

theorem farm_produce_weeks :
  let saly_eggs := 10
  let ben_eggs := 14
  let ked_eggs := 14 / 2
  let total_eggs := 124
  let weekly_eggs := eggs_needed_per_week saly_eggs ben_eggs ked_eggs
  number_of_weeks total_eggs weekly_eggs = 4 :=
by
  sorry 

end farm_produce_weeks_l158_158158


namespace eddie_weekly_earnings_l158_158259

theorem eddie_weekly_earnings :
  let mon_hours := 2.5
  let tue_hours := 7 / 6
  let wed_hours := 7 / 4
  let sat_hours := 3 / 4
  let weekday_rate := 4
  let saturday_rate := 6
  let mon_earnings := mon_hours * weekday_rate
  let tue_earnings := tue_hours * weekday_rate
  let wed_earnings := wed_hours * weekday_rate
  let sat_earnings := sat_hours * saturday_rate
  let total_earnings := mon_earnings + tue_earnings + wed_earnings + sat_earnings
  total_earnings = 26.17 := by
  simp only
  norm_num
  sorry

end eddie_weekly_earnings_l158_158259


namespace find_B_l158_158131

theorem find_B (B: ℕ) (h1: 5457062 % 2 = 0 ∧ 200 * B % 4 = 0) (h2: 5457062 % 5 = 0 ∧ B % 5 = 0) (h3: 5450062 % 8 = 0 ∧ 100 * B % 8 = 0) : B = 0 :=
sorry

end find_B_l158_158131


namespace length_of_train_l158_158031

theorem length_of_train (speed_kmh : ℝ) (time_min : ℝ) (tunnel_length_m : ℝ) (train_length_m : ℝ) :
  speed_kmh = 78 → time_min = 1 → tunnel_length_m = 500 → train_length_m = 800.2 :=
by
  sorry

end length_of_train_l158_158031


namespace tanya_bought_11_pears_l158_158922

variable (P : ℕ)

-- Define the given conditions about the number of different fruits Tanya bought
def apples : ℕ := 4
def pineapples : ℕ := 2
def basket_of_plums : ℕ := 1

-- Define the total number of fruits initially and the remaining fruits
def initial_fruit_total : ℕ := 18
def remaining_fruit_total : ℕ := 9
def half_fell_out_of_bag : ℕ := remaining_fruit_total * 2

-- The main theorem to prove
theorem tanya_bought_11_pears (h : P + apples + pineapples + basket_of_plums = initial_fruit_total) : P = 11 := by
  -- providing a placeholder for the proof
  sorry

end tanya_bought_11_pears_l158_158922


namespace triangle_inequality_l158_158639

variables {a b c h : ℝ}
variable {n : ℕ}

theorem triangle_inequality
  (h_triangle : a^2 + b^2 = c^2)
  (h_height : a * b = c * h)
  (h_cond : a + b < c + h)
  (h_pos_n : n > 0) :
  a^n + b^n < c^n + h^n :=
sorry

end triangle_inequality_l158_158639


namespace polygon_edges_l158_158051

theorem polygon_edges (n : ℕ) (h1 : (n - 2) * 180 = 4 * 360 + 180) : n = 11 :=
by {
  sorry
}

end polygon_edges_l158_158051


namespace factors_of_2520_l158_158315

theorem factors_of_2520 : (∃ (factors : Finset ℕ), factors.card = 48 ∧ ∀ d, d ∈ factors ↔ d > 0 ∧ 2520 % d = 0) :=
sorry

end factors_of_2520_l158_158315


namespace percent_singles_l158_158289

theorem percent_singles (total_hits home_runs triples doubles : ℕ) 
  (h_total: total_hits = 50) 
  (h_hr: home_runs = 3) 
  (h_tr: triples = 2) 
  (h_double: doubles = 8) : 
  100 * (total_hits - (home_runs + triples + doubles)) / total_hits = 74 := 
by
  -- proofs
  sorry

end percent_singles_l158_158289


namespace smallest_rel_prime_120_l158_158782

theorem smallest_rel_prime_120 : ∃ (x : ℕ), x > 1 ∧ Nat.gcd x 120 = 1 ∧ ∀ y, y > 1 ∧ Nat.gcd y 120 = 1 → x ≤ y :=
by
  use 7
  sorry

end smallest_rel_prime_120_l158_158782


namespace minimum_value_expr_l158_158666

theorem minimum_value_expr (x y : ℝ) : 
  (xy - 2)^2 + (x^2 + y^2)^2 ≥ 4 :=
sorry

end minimum_value_expr_l158_158666


namespace lemonade_percentage_correct_l158_158580
noncomputable def lemonade_percentage (first_lemonade first_carbon second_carbon mixture_carbon first_portion : ℝ) : ℝ :=
  100 - second_carbon

theorem lemonade_percentage_correct :
  let first_lemonade := 20
  let first_carbon := 80
  let second_carbon := 55
  let mixture_carbon := 60
  let first_portion := 19.99999999999997
  lemonade_percentage first_lemonade first_carbon second_carbon mixture_carbon first_portion = 45 :=
by
  -- Proof to be completed.
  sorry

end lemonade_percentage_correct_l158_158580


namespace a_5_eq_14_l158_158521

def S (n : ℕ) : ℚ := (3 / 2) * n ^ 2 + (1 / 2) * n

def a (n : ℕ) : ℚ := S n - S (n - 1)

theorem a_5_eq_14 : a 5 = 14 := by {
  -- Proof steps go here
  sorry
}

end a_5_eq_14_l158_158521


namespace min_max_product_l158_158248

noncomputable def min_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the minimum value of 3x^2 + 4xy + 3y^2
  sorry

noncomputable def max_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the maximum value of 3x^2 + 4xy + 3y^2
  sorry

theorem min_max_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) :
  min_value x y h * max_value x y h = 7 / 16 :=
sorry

end min_max_product_l158_158248


namespace ball_height_25_l158_158543

theorem ball_height_25 (t : ℝ) (h : ℝ) 
  (h_eq : h = 45 - 7 * t - 6 * t^2) : 
  h = 25 ↔ t = 4 / 3 := 
by 
  sorry

end ball_height_25_l158_158543


namespace thread_length_l158_158992

theorem thread_length (initial_length : ℝ) (fraction : ℝ) (additional_length : ℝ) (total_length : ℝ) 
  (h1 : initial_length = 12) 
  (h2 : fraction = 3 / 4) 
  (h3 : additional_length = initial_length * fraction)
  (h4 : total_length = initial_length + additional_length) : 
  total_length = 21 := 
by
  -- proof steps would go here
  sorry

end thread_length_l158_158992


namespace find_k_l158_158604

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 3)

def vec_add_2b (k : ℝ) : ℝ × ℝ := (2 + 2 * k, 7)
def vec_sub_b (k : ℝ) : ℝ × ℝ := (4 - k, -1)

def vectors_not_parallel (k : ℝ) : Prop :=
  (vec_add_2b k).fst * (vec_sub_b k).snd ≠ (vec_add_2b k).snd * (vec_sub_b k).fst

theorem find_k (k : ℝ) (h : vectors_not_parallel k) : k ≠ 6 :=
by
  sorry

end find_k_l158_158604


namespace total_pages_in_book_l158_158113

-- Define the conditions
def pagesDay1To5 : Nat := 5 * 25
def pagesDay6To9 : Nat := 4 * 40
def pagesLastDay : Nat := 30

-- Total calculation
def totalPages (p1 p2 pLast : Nat) : Nat := p1 + p2 + pLast

-- The proof problem statement
theorem total_pages_in_book :
  totalPages pagesDay1To5 pagesDay6To9 pagesLastDay = 315 :=
  by
    sorry

end total_pages_in_book_l158_158113


namespace opera_house_earnings_l158_158949

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l158_158949


namespace sum_of_digits_next_l158_158190

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem sum_of_digits_next (n : ℕ) (h : sum_of_digits n = 1399) : 
  sum_of_digits (n + 1) = 1402 :=
sorry

end sum_of_digits_next_l158_158190


namespace simplify_expression_l158_158434

-- Definitions for conditions and parameters
variables {x y : ℝ}

-- The problem statement and proof
theorem simplify_expression : 12 * x^5 * y / (6 * x * y) = 2 * x^4 :=
by sorry

end simplify_expression_l158_158434


namespace largest_n_fact_product_of_four_consecutive_integers_l158_158947

theorem largest_n_fact_product_of_four_consecutive_integers :
  ∀ (n : ℕ), (∃ x : ℕ, n.factorial = x * (x + 1) * (x + 2) * (x + 3)) → n ≤ 6 :=
by
  sorry

end largest_n_fact_product_of_four_consecutive_integers_l158_158947


namespace find_modulus_z_l158_158084

open Complex

noncomputable def z_w_condition1 (z w : ℂ) : Prop := abs (3 * z - w) = 17
noncomputable def z_w_condition2 (z w : ℂ) : Prop := abs (z + 3 * w) = 4
noncomputable def z_w_condition3 (z w : ℂ) : Prop := abs (z + w) = 6

theorem find_modulus_z (z w : ℂ) (h1 : z_w_condition1 z w) (h2 : z_w_condition2 z w) (h3 : z_w_condition3 z w) :
  abs z = 5 :=
by
  sorry

end find_modulus_z_l158_158084


namespace ratio_of_metals_l158_158962

theorem ratio_of_metals (G C S : ℝ) (h1 : 11 * G + 5 * C + 7 * S = 9 * (G + C + S)) : 
  G / C = 1 / 2 ∧ G / S = 1 :=
by
  sorry

end ratio_of_metals_l158_158962


namespace passes_through_1_1_l158_158325

theorem passes_through_1_1 (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^ (x - 1))} :=
by
  -- proof not required
  sorry

end passes_through_1_1_l158_158325


namespace intersection_eq_l158_158691

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l158_158691


namespace smallest_base_l158_158290

theorem smallest_base (b : ℕ) (n : ℕ) : (n = 512) → (b^3 ≤ n ∧ n < b^4) → ((n / b^3) % b + 1) % 2 = 0 → b = 6 := sorry

end smallest_base_l158_158290


namespace series_sum_eq_half_l158_158468

theorem series_sum_eq_half : ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
by
  sorry

end series_sum_eq_half_l158_158468


namespace weight_of_replaced_student_l158_158865

variable (W : ℝ) -- total weight of the original 10 students
variable (new_student_weight : ℝ := 60) -- weight of the new student
variable (weight_decrease_per_student : ℝ := 6) -- average weight decrease per student

theorem weight_of_replaced_student (replaced_student_weight : ℝ) :
  (W - replaced_student_weight + new_student_weight = W - 10 * weight_decrease_per_student) →
  replaced_student_weight = 120 := by
  sorry

end weight_of_replaced_student_l158_158865


namespace sum_of_squares_ineq_l158_158146

theorem sum_of_squares_ineq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_sq : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 :=
sorry

end sum_of_squares_ineq_l158_158146


namespace B_investment_time_l158_158749

theorem B_investment_time (x : ℝ) (m : ℝ) :
  let A_share := x * 12
  let B_share := 2 * x * (12 - m)
  let C_share := 3 * x * 4
  let total_gain := 18600
  let A_gain := 6200
  let ratio := A_gain / total_gain
  ratio = 1 / 3 →
  A_share = 1 / 3 * (A_share + B_share + C_share) →
  m = 6 := by
sorry

end B_investment_time_l158_158749


namespace range_of_a_l158_158536

theorem range_of_a (a : ℝ) : (-1/Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1/Real.exp 1) :=
  sorry

end range_of_a_l158_158536


namespace extremum_at_one_eq_a_one_l158_158396

theorem extremum_at_one_eq_a_one 
  (a : ℝ) 
  (h : ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * a * x^2 - 3) ∧ f' 1 = 0) : 
  a = 1 :=
sorry

end extremum_at_one_eq_a_one_l158_158396


namespace train_crossing_time_l158_158615

-- Definitions based on conditions from the problem
def length_of_train_and_platform := 900 -- in meters
def speed_km_per_hr := 108 -- in km/hr
def distance := 2 * length_of_train_and_platform -- distance to be covered
def speed_m_per_s := (speed_km_per_hr * 1000) / 3600 -- converted speed

-- Theorem stating the time to cross the platform is 60 seconds
theorem train_crossing_time : distance / speed_m_per_s = 60 := by
  sorry

end train_crossing_time_l158_158615


namespace smallest_positive_integer_l158_158202

def is_prime_gt_60 (n : ℕ) : Prop :=
  n > 60 ∧ Prime n

def smallest_integer_condition (k : ℕ) : Prop :=
  ¬ Prime k ∧ ¬ (∃ m : ℕ, m * m = k) ∧ 
  ∀ p : ℕ, Prime p → p ∣ k → p > 60

theorem smallest_positive_integer : ∃ k : ℕ, k = 4087 ∧ smallest_integer_condition k := by
  sorry

end smallest_positive_integer_l158_158202


namespace no_solution_frac_eq_l158_158230

theorem no_solution_frac_eq (k : ℝ) : (∀ x : ℝ, ¬(1 / (x + 1) = 3 * k / x)) ↔ (k = 0 ∨ k = 1 / 3) :=
by
  sorry

end no_solution_frac_eq_l158_158230


namespace depth_of_river_bank_l158_158287

theorem depth_of_river_bank (top_width bottom_width area depth : ℝ) 
  (h₁ : top_width = 12)
  (h₂ : bottom_width = 8)
  (h₃ : area = 500)
  (h₄ : area = (1 / 2) * (top_width + bottom_width) * depth) :
  depth = 50 :=
sorry

end depth_of_river_bank_l158_158287


namespace total_bronze_needed_l158_158626

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l158_158626


namespace smallest_solution_l158_158322

theorem smallest_solution (x : ℝ) (h : x^4 - 16 * x^2 + 63 = 0) :
  x = -3 :=
sorry

end smallest_solution_l158_158322


namespace find_ratio_l158_158459

open Nat

def sequence_def (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 →
    (a ((n + 2)) / a ((n + 1))) - (a ((n + 1)) / a n) = d

def geometric_difference_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3 ∧ sequence_def a 2

theorem find_ratio (a : ℕ → ℕ) (h : geometric_difference_sequence a) :
  a 12 / a 10 = 399 := sorry

end find_ratio_l158_158459


namespace find_a_l158_158717

theorem find_a (a : ℝ) (h₁ : a ≠ 0) (h₂ : ∀ x y : ℝ, x^2 + a*y^2 + a^2 = 0) (h₃ : 4 = 4) :
  a = (1 - Real.sqrt 17) / 2 := sorry

end find_a_l158_158717


namespace solve_for_x_l158_158895

theorem solve_for_x :
  ∃ x : ℤ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27)) = 113 ∧ x = 40 := 
by
  sorry

end solve_for_x_l158_158895


namespace determine_house_height_l158_158035

-- Definitions for the conditions
def house_shadow : ℚ := 75
def tree_height : ℚ := 15
def tree_shadow : ℚ := 20

-- Desired Height of Lily's house
def house_height : ℚ := 56

-- Theorem stating the height of the house
theorem determine_house_height :
  (house_shadow / tree_shadow = house_height / tree_height) -> house_height = 56 :=
  by
  unfold house_shadow tree_height tree_shadow house_height
  sorry

end determine_house_height_l158_158035


namespace max_value_of_XYZ_XY_YZ_ZX_l158_158968

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end max_value_of_XYZ_XY_YZ_ZX_l158_158968


namespace find_natural_numbers_l158_158425

theorem find_natural_numbers (x y z : ℕ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_ordered : x < y ∧ y < z)
  (h_reciprocal_sum_nat : ∃ a : ℕ, 1/x + 1/y + 1/z = a) : (x, y, z) = (2, 3, 6) := 
sorry

end find_natural_numbers_l158_158425


namespace apples_handed_out_l158_158923

theorem apples_handed_out 
  (initial_apples : ℕ)
  (pies_made : ℕ)
  (apples_per_pie : ℕ)
  (H : initial_apples = 50)
  (H1 : pies_made = 9)
  (H2 : apples_per_pie = 5) :
  initial_apples - (pies_made * apples_per_pie) = 5 := 
by
  sorry

end apples_handed_out_l158_158923


namespace three_friends_at_least_50_mushrooms_l158_158887

theorem three_friends_at_least_50_mushrooms (a : Fin 7 → ℕ) (h_sum : (Finset.univ.sum a) = 100) (h_different : Function.Injective a) :
  ∃ i j k : Fin 7, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ (a i + a j + a k) ≥ 50 :=
by
  sorry

end three_friends_at_least_50_mushrooms_l158_158887


namespace triangle_type_and_area_l158_158233

theorem triangle_type_and_area (x : ℝ) (hpos : 0 < x) (h : 3 * x + 4 * x + 5 * x = 36) :
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  a^2 + b^2 = c^2 ∧ (1 / 2) * a * b = 54 :=
by {
  sorry
}

end triangle_type_and_area_l158_158233


namespace p_sq_plus_q_sq_l158_158689

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := 
by 
  sorry

end p_sq_plus_q_sq_l158_158689


namespace bryan_total_books_magazines_l158_158385

-- Conditions as definitions
def novels : ℕ := 90
def comics : ℕ := 160
def rooms : ℕ := 12
def x := (3 / 4 : ℚ) * novels
def y := (6 / 5 : ℚ) * comics
def z := (1 / 2 : ℚ) * rooms

-- Calculations based on conditions
def books_per_shelf := 27 * x
def magazines_per_shelf := 80 * y
def total_shelves := 23 * z
def total_books := books_per_shelf * total_shelves
def total_magazines := magazines_per_shelf * total_shelves
def grand_total := total_books + total_magazines

-- Theorem to prove
theorem bryan_total_books_magazines :
  grand_total = 2371275 := by
  sorry

end bryan_total_books_magazines_l158_158385


namespace biology_marks_l158_158848

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end biology_marks_l158_158848


namespace initial_men_colouring_l158_158752

theorem initial_men_colouring (M : ℕ) : 
  (∀ m : ℕ, ∀ d : ℕ, ∀ l : ℕ, m * d = 48 * 2 → 8 * 0.75 = 6 → M = 4) :=
by
  sorry

end initial_men_colouring_l158_158752


namespace unique_solution_of_system_l158_158490

theorem unique_solution_of_system (n k m : ℕ) (hnk : n + k = Nat.gcd n k ^ 2) (hkm : k + m = Nat.gcd k m ^ 2) (hmn : m + n = Nat.gcd m n ^ 2) : 
  n = 2 ∧ k = 2 ∧ m = 2 :=
by
  sorry

end unique_solution_of_system_l158_158490


namespace M_subsetneq_P_l158_158637

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subsetneq_P : M ⊂ P :=
by sorry

end M_subsetneq_P_l158_158637


namespace find_a_l158_158695

noncomputable def f (x : ℝ) : ℝ := 5^(abs x)

noncomputable def g (a x : ℝ) : ℝ := a*x^2 - x

theorem find_a (a : ℝ) (h : f (g a 1) = 1) : a = 1 := 
by
  sorry

end find_a_l158_158695


namespace determine_digit_l158_158662

theorem determine_digit (Θ : ℕ) (hΘ : Θ > 0 ∧ Θ < 10) (h : 630 / Θ = 40 + 3 * Θ) : Θ = 9 :=
sorry

end determine_digit_l158_158662


namespace additional_carpet_needed_l158_158985

-- Define the given conditions as part of the hypothesis:
def carpetArea : ℕ := 18
def roomLength : ℕ := 4
def roomWidth : ℕ := 20

-- The theorem we want to prove:
theorem additional_carpet_needed : (roomLength * roomWidth - carpetArea) = 62 := by
  sorry

end additional_carpet_needed_l158_158985


namespace chair_and_desk_prices_l158_158108

theorem chair_and_desk_prices (c d : ℕ) 
  (h1 : c + d = 115)
  (h2 : d - c = 45) :
  c = 35 ∧ d = 80 := 
by
  sorry

end chair_and_desk_prices_l158_158108


namespace a_minus_b_ge_one_l158_158377

def a : ℕ := 19^91
def b : ℕ := (999991)^19

theorem a_minus_b_ge_one : a - b ≥ 1 :=
by
  sorry

end a_minus_b_ge_one_l158_158377


namespace correct_equation_l158_158650

theorem correct_equation (x Planned : ℝ) (h1 : 6 * x = Planned + 7) (h2 : 5 * x = Planned - 13) :
  6 * x - 7 = 5 * x + 13 :=
by
  sorry

end correct_equation_l158_158650


namespace sum_of_distances_l158_158550

theorem sum_of_distances (a b : ℤ) (k : ℕ) 
  (h1 : |k - a| + |(k + 1) - a| + |(k + 2) - a| + |(k + 3) - a| + |(k + 4) - a| + |(k + 5) - a| + |(k + 6) - a| = 609)
  (h2 : |k - b| + |(k + 1) - b| + |(k + 2) - b| + |(k + 3) - b| + |(k + 4) - b| + |(k + 5) - b| + |(k + 6) - b| = 721)
  (h3 : a + b = 192) :
  a = 1 ∨ a = 104 ∨ a = 191 := 
sorry

end sum_of_distances_l158_158550
