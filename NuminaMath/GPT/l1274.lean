import Mathlib

namespace price_change_l1274_127444

theorem price_change (P : ℝ) : 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  P4 = P * 0.9216 := 
by 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  show P4 = P * 0.9216
  sorry

end price_change_l1274_127444


namespace find_roots_of_polynomial_l1274_127419

noncomputable def polynomial := Polynomial ℝ

theorem find_roots_of_polynomial :
  (∃ (x : ℝ), x^3 + 3 * x^2 - 6 * x - 8 = 0) ↔ (x = -1 ∨ x = 2 ∨ x = -4) :=
sorry

end find_roots_of_polynomial_l1274_127419


namespace my_age_now_l1274_127446

theorem my_age_now (Y S : ℕ) (h1 : Y - 9 = 5 * (S - 9)) (h2 : Y = 3 * S) : Y = 54 := by
  sorry

end my_age_now_l1274_127446


namespace smallest_b_exists_l1274_127435

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l1274_127435


namespace compute_expression_l1274_127402

noncomputable def roots_exist (P : Polynomial ℝ) (α β γ : ℝ) : Prop :=
  P = Polynomial.C (-13) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X))

theorem compute_expression (α β γ : ℝ) (h : roots_exist (Polynomial.X^3 - 7 * Polynomial.X^2 + 11 * Polynomial.X - 13) α β γ) :
  (α ≠ 0) → (β ≠ 0) → (γ ≠ 0) → (α^2 * β^2 + β^2 * γ^2 + γ^2 * α^2 = -61) :=
  sorry

end compute_expression_l1274_127402


namespace no_pos_int_squares_l1274_127429

open Nat

theorem no_pos_int_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬(∃ k m : ℕ, k ^ 2 = a ^ 2 + b ∧ m ^ 2 = b ^ 2 + a) :=
sorry

end no_pos_int_squares_l1274_127429


namespace digit_five_occurrences_l1274_127483

variable (fives_ones fives_tens fives_hundreds : ℕ)

def count_fives := fives_ones + fives_tens + fives_hundreds

theorem digit_five_occurrences :
  ( ∀ (fives_ones fives_tens fives_hundreds : ℕ), 
    fives_ones = 100 ∧ fives_tens = 100 ∧ fives_hundreds = 100 → 
    count_fives fives_ones fives_tens fives_hundreds = 300 ) :=
by
  sorry

end digit_five_occurrences_l1274_127483


namespace value_of_g_at_x_minus_5_l1274_127438

-- Definition of the function g
def g (x : ℝ) : ℝ := -3

-- The theorem we need to prove
theorem value_of_g_at_x_minus_5 (x : ℝ) : g (x - 5) = -3 := by
  sorry

end value_of_g_at_x_minus_5_l1274_127438


namespace weight_difference_l1274_127432

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h_avg_ABC : (W_A + W_B + W_C) / 3 = 80)
  (h_WA : W_A = 95)
  (h_avg_ABCD : (W_A + W_B + W_C + W_D) / 4 = 82)
  (h_avg_BCDE : (W_B + W_C + W_D + W_E) / 4 = 81) :
  W_E - W_D = 3 :=
by
  sorry

end weight_difference_l1274_127432


namespace min_buses_needed_l1274_127469

theorem min_buses_needed (total_students : ℕ) (bus45_capacity : ℕ) (bus40_capacity : ℕ) : 
  total_students = 530 ∧ bus45_capacity = 45 ∧ bus40_capacity = 40 → 
  ∃ (n : ℕ), n = 12 :=
by 
  intro h
  obtain ⟨htotal, hbus45, hbus40⟩ := h
  -- Proof would go here...
  sorry

end min_buses_needed_l1274_127469


namespace Tim_age_l1274_127482

theorem Tim_age : ∃ (T : ℕ), (T = (3 * T + 2 - 12)) ∧ (T = 5) :=
by
  existsi 5
  sorry

end Tim_age_l1274_127482


namespace sport_formulation_water_content_l1274_127417

theorem sport_formulation_water_content :
  ∀ (f_s c_s w_s : ℕ) (f_p c_p w_p : ℕ),
    f_s / c_s = 1 / 12 →
    f_s / w_s = 1 / 30 →
    f_p / c_p = 1 / 4 →
    f_p / w_p = 1 / 60 →
    c_p = 4 →
    w_p = 60 := by
  sorry

end sport_formulation_water_content_l1274_127417


namespace width_of_foil_covered_prism_l1274_127472

noncomputable def foil_covered_prism_width : ℕ :=
  let (l, w, h) := (4, 8, 4)
  let inner_width := 2 * l
  let increased_width := w + 2
  increased_width

theorem width_of_foil_covered_prism : foil_covered_prism_width = 10 := 
by
  let l := 4
  let w := 2 * l
  let h := w / 2
  have volume : l * w * h = 128 := by
    sorry
  have width_foil_covered := w + 2
  have : foil_covered_prism_width = width_foil_covered := by
    sorry
  sorry

end width_of_foil_covered_prism_l1274_127472


namespace xiao_li_first_three_l1274_127471

def q1_proba_correct (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem xiao_li_first_three (p1 p2 p3 : ℚ) (h1 : p1 = 3/4) (h2 : p2 = 1/2) (h3 : p3 = 5/6) :
  q1_proba_correct p1 p2 p3 = 11 / 24 := by
  rw [h1, h2, h3]
  sorry

end xiao_li_first_three_l1274_127471


namespace computation_problem_points_l1274_127447

def num_problems : ℕ := 30
def points_per_word_problem : ℕ := 5
def total_points : ℕ := 110
def num_computation_problems : ℕ := 20

def points_per_computation_problem : ℕ := 3

theorem computation_problem_points :
  ∃ x : ℕ, (num_computation_problems * x + (num_problems - num_computation_problems) * points_per_word_problem = total_points) ∧ x = points_per_computation_problem :=
by
  use points_per_computation_problem
  simp
  sorry

end computation_problem_points_l1274_127447


namespace max_m_divides_f_l1274_127496

noncomputable def f (n : ℕ) : ℤ :=
  (2 * n + 7) * 3^n + 9

theorem max_m_divides_f (m n : ℕ) (h1 : n > 0) (h2 : ∀ n : ℕ, n > 0 → m ∣ ((2 * n + 7) * 3^n + 9)) : m = 36 :=
sorry

end max_m_divides_f_l1274_127496


namespace joan_total_seashells_l1274_127478

def seashells_given_to_Sam : ℕ := 43
def seashells_left_with_Joan : ℕ := 27
def total_seashells_found := seashells_given_to_Sam + seashells_left_with_Joan

theorem joan_total_seashells : total_seashells_found = 70 := by
  -- proof goes here, but for now we will use sorry
  sorry

end joan_total_seashells_l1274_127478


namespace number_of_girls_is_eleven_l1274_127465

-- Conditions transformation
def boys_wear_red_hats : Prop := true
def girls_wear_yellow_hats : Prop := true
def teachers_wear_blue_hats : Prop := true
def cannot_see_own_hat : Prop := true
def little_qiang_sees_hats (x k : ℕ) : Prop := (x + 2) = (x + 2)
def little_hua_sees_hats (x k : ℕ) : Prop := x = 2 * k
def teacher_sees_hats (x k : ℕ) : Prop := k + 2 = (x + 2) + k - 11

-- Proof Statement
theorem number_of_girls_is_eleven (x k : ℕ) (h1 : boys_wear_red_hats)
  (h2 : girls_wear_yellow_hats) (h3 : teachers_wear_blue_hats)
  (h4 : cannot_see_own_hat) (hq : little_qiang_sees_hats x k)
  (hh : little_hua_sees_hats x k) (ht : teacher_sees_hats x k) : x = 11 :=
sorry

end number_of_girls_is_eleven_l1274_127465


namespace triangle_ineq_l1274_127443

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 2 * (a^2 + b^2) > c^2 := 
by 
  sorry

end triangle_ineq_l1274_127443


namespace inscribed_circle_radius_integer_l1274_127424

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end inscribed_circle_radius_integer_l1274_127424


namespace illegal_simplification_works_for_specific_values_l1274_127416

-- Definitions for the variables
def a : ℕ := 43
def b : ℕ := 17
def c : ℕ := 26

-- Define the sum of cubes
def sum_of_cubes (x y : ℕ) : ℕ := x ^ 3 + y ^ 3

-- Define the illegal simplification fraction
def illegal_simplification_fraction_correct (a b c : ℕ) : Prop :=
  (a^3 + b^3) / (a^3 + c^3) = (a + b) / (a + c)

-- The theorem to prove
theorem illegal_simplification_works_for_specific_values :
  illegal_simplification_fraction_correct a b c :=
by
  -- Proof will reside here
  sorry

end illegal_simplification_works_for_specific_values_l1274_127416


namespace marching_band_total_weight_l1274_127428

def weight_trumpets := 5
def weight_clarinets := 5
def weight_trombones := 10
def weight_tubas := 20
def weight_drums := 15

def count_trumpets := 6
def count_clarinets := 9
def count_trombones := 8
def count_tubas := 3
def count_drums := 2

theorem marching_band_total_weight :
  (count_trumpets * weight_trumpets) + (count_clarinets * weight_clarinets) + (count_trombones * weight_trombones) + 
  (count_tubas * weight_tubas) + (count_drums * weight_drums) = 245 :=
by
  sorry

end marching_band_total_weight_l1274_127428


namespace cube_faces_sum_l1274_127456

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : a = 12) (h2 : b = 13) (h3 : c = 14)
  (h4 : d = 15) (h5 : e = 16) (h6 : f = 17)
  (h_pairs : a + f = b + e ∧ b + e = c + d) :
  a + b + c + d + e + f = 87 := by
  sorry

end cube_faces_sum_l1274_127456


namespace arithmetic_sequence_sum_l1274_127457

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (d : α)

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = d

def sum_condition (a : ℕ → α) : Prop :=
  a 2 + a 5 + a 8 = 39

-- The goal statement to prove
theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_sum : sum_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
  sorry

end arithmetic_sequence_sum_l1274_127457


namespace cauchy_functional_eq_l1274_127415

theorem cauchy_functional_eq
  (f : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end cauchy_functional_eq_l1274_127415


namespace uncle_bob_can_park_l1274_127485

-- Define the conditions
def total_spaces : Nat := 18
def cars : Nat := 15
def rv_spaces : Nat := 3

-- Define a function to calculate the probability (without implementation)
noncomputable def probability_RV_can_park (total_spaces cars rv_spaces : Nat) : Rat :=
  if h : rv_spaces <= total_spaces - cars then
    -- The probability calculation logic would go here
    16 / 51
  else
    0

-- The theorem stating the desired result
theorem uncle_bob_can_park : probability_RV_can_park total_spaces cars rv_spaces = 16 / 51 :=
  sorry

end uncle_bob_can_park_l1274_127485


namespace neg_universal_to_existential_l1274_127462

theorem neg_universal_to_existential :
  (¬ (∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by 
  sorry

end neg_universal_to_existential_l1274_127462


namespace probability_white_marble_l1274_127498

theorem probability_white_marble :
  ∀ (p_blue p_green p_white : ℝ),
    p_blue = 0.25 →
    p_green = 0.4 →
    p_blue + p_green + p_white = 1 →
    p_white = 0.35 :=
by
  intros p_blue p_green p_white h_blue h_green h_total
  sorry

end probability_white_marble_l1274_127498


namespace jovana_initial_shells_l1274_127466

theorem jovana_initial_shells (x : ℕ) (h₁ : x + 12 = 17) : x = 5 :=
by
  -- Proof omitted
  sorry

end jovana_initial_shells_l1274_127466


namespace Miss_Adamson_paper_usage_l1274_127433

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end Miss_Adamson_paper_usage_l1274_127433


namespace moles_of_CaCO3_formed_l1274_127412

theorem moles_of_CaCO3_formed (m n : ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : ∀ m n : ℕ, (m = n) → (m = 3) → (n = 3) → moles_of_CaCO3 = m) : 
  moles_of_CaCO3 = 3 := by
  sorry

end moles_of_CaCO3_formed_l1274_127412


namespace point_location_l1274_127408

variables {A B C m n : ℝ}

theorem point_location (h1 : A > 0) (h2 : B < 0) (h3 : A * m + B * n + C < 0) : 
  -- Statement: the point P(m, n) is on the upper right side of the line Ax + By + C = 0
  true :=
sorry

end point_location_l1274_127408


namespace profit_at_15_percent_off_l1274_127448

theorem profit_at_15_percent_off 
    (cost_price marked_price : ℝ) 
    (cost_price_eq : cost_price = 2000)
    (marked_price_eq : marked_price = (200 + cost_price) / 0.8) :
    (0.85 * marked_price - cost_price) = 337.5 := by
  sorry

end profit_at_15_percent_off_l1274_127448


namespace largest_angle_in_pentagon_l1274_127413

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (hA : A = 70) 
  (hB : B = 120) 
  (hCD : C = D) 
  (hE : E = 3 * C - 30) 
  (sum_angles : A + B + C + D + E = 540) :
  E = 198 := 
by 
  sorry

end largest_angle_in_pentagon_l1274_127413


namespace sum_original_numbers_is_five_l1274_127453

noncomputable def sum_original_numbers (a b c d : ℤ) : ℤ :=
  a + b + c + d

theorem sum_original_numbers_is_five (a b c d : ℤ) (hab : 10 * a + b = overline_ab) 
  (h : 100 * (10 * a + b) + 10 * c + 7 * d = 2024) : sum_original_numbers a b c d = 5 :=
sorry

end sum_original_numbers_is_five_l1274_127453


namespace irrational_of_sqrt_3_l1274_127451

noncomputable def is_irritational (x : ℝ) : Prop :=
  ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)

theorem irrational_of_sqrt_3 :
  is_irritational 0 = false ∧
  is_irritational 3.14 = false ∧
  is_irritational (-1) = false ∧
  is_irritational (Real.sqrt 3) = true := 
by
  -- Proof omitted
  sorry

end irrational_of_sqrt_3_l1274_127451


namespace find_d_values_l1274_127420

theorem find_d_values (u v : ℝ) (c d : ℝ)
  (hpu : u^3 + c * u + d = 0)
  (hpv : v^3 + c * v + d = 0)
  (hqu : (u + 2)^3 + c * (u + 2) + d - 120 = 0)
  (hqv : (v - 5)^3 + c * (v - 5) + d - 120 = 0) :
  d = 396 ∨ d = 8 :=
by
  -- placeholder for the actual proof
  sorry

end find_d_values_l1274_127420


namespace parking_lot_wheels_l1274_127439

-- definitions for the conditions
def num_cars : ℕ := 10
def num_bikes : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- statement of the theorem
theorem parking_lot_wheels : (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 44 := by
  sorry

end parking_lot_wheels_l1274_127439


namespace fraction_meaningful_iff_l1274_127479

theorem fraction_meaningful_iff (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 :=
by 
  sorry

end fraction_meaningful_iff_l1274_127479


namespace surface_area_of_cylinder_with_square_cross_section_l1274_127495

theorem surface_area_of_cylinder_with_square_cross_section
  (side_length : ℝ) (h1 : side_length = 2) : 
  (2 * Real.pi * 2 + 2 * Real.pi * 1^2) = 6 * Real.pi :=
by
  rw [←h1]
  sorry

end surface_area_of_cylinder_with_square_cross_section_l1274_127495


namespace parabola_translation_l1274_127468

theorem parabola_translation :
  ∀ (x : ℝ),
  (∃ x' y' : ℝ, x' = x - 1 ∧ y' = 2 * x' ^ 2 - 3 ∧ y = y' + 3) →
  (y = 2 * x ^ 2) :=
by
  sorry

end parabola_translation_l1274_127468


namespace work_completion_time_l1274_127487

variable (p q : Type)

def efficient (p q : Type) : Prop :=
  ∃ (Wp Wq : ℝ), Wp = 1.5 * Wq ∧ Wp = 1 / 25

def work_done_together (p q : Type) := 1/15

theorem work_completion_time {p q : Type} (h1 : efficient p q) :
  ∃ d : ℝ, d = 15 :=
  sorry

end work_completion_time_l1274_127487


namespace f_2015_value_l1274_127407

noncomputable def f : ℝ → ℝ := sorry -- Define f with appropriate conditions

theorem f_2015_value :
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 2015 = -2 :=
by
  sorry -- Proof to be provided

end f_2015_value_l1274_127407


namespace John_l1274_127442

theorem John's_earnings_on_Saturday :
  ∃ S : ℝ, (S + S / 2 + 20 = 47) ∧ (S = 18) := by
    sorry

end John_l1274_127442


namespace solution_exists_l1274_127491

def age_problem (S F Y : ℕ) : Prop :=
  S = 12 ∧ S = F / 3 ∧ S - Y = (F - Y) / 5 ∧ Y = 6

theorem solution_exists : ∃ (Y : ℕ), ∃ (S F : ℕ), age_problem S F Y :=
by sorry

end solution_exists_l1274_127491


namespace katya_female_classmates_l1274_127474

theorem katya_female_classmates (g b : ℕ) (h1 : b = 2 * g) (h2 : b = g + 7) :
  g - 1 = 6 :=
by
  sorry

end katya_female_classmates_l1274_127474


namespace range_of_m_satisfying_obtuse_triangle_l1274_127490

theorem range_of_m_satisfying_obtuse_triangle (m : ℝ) 
(h_triangle: m > 0 
  → m + (m + 1) > (m + 2) 
  ∧ m + (m + 2) > (m + 1) 
  ∧ (m + 1) + (m + 2) > m
  ∧ (m + 2) ^ 2 > m ^ 2 + (m + 1) ^ 2) : 1 < m ∧ m < 1.5 :=
by
  sorry

end range_of_m_satisfying_obtuse_triangle_l1274_127490


namespace roots_real_and_equal_l1274_127454

theorem roots_real_and_equal (a b c : ℝ) (h_eq : a = 1) (h_b : b = -4 * Real.sqrt 2) (h_c : c = 8) :
  ∃ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) :=
by
  have h_a : a = 1 := h_eq;
  have h_b : b = -4 * Real.sqrt 2 := h_b;
  have h_c : c = 8 := h_c;
  sorry

end roots_real_and_equal_l1274_127454


namespace shortest_chord_line_l1274_127404

theorem shortest_chord_line (x y : ℝ) (P : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) (h₁ : C x y) (hx : P = (1, 1)) (hC : ∀ x y, C x y ↔ x^2 + y^2 = 4) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ a * x + b * y + c = 0 :=
by
  sorry

end shortest_chord_line_l1274_127404


namespace cricket_team_members_l1274_127406

theorem cricket_team_members (avg_whole_team: ℕ) (captain_age: ℕ) (wicket_keeper_age: ℕ) 
(remaining_avg_age: ℕ) (n: ℕ):
avg_whole_team = 23 →
captain_age = 25 →
wicket_keeper_age = 30 →
remaining_avg_age = 22 →
(n * avg_whole_team - captain_age - wicket_keeper_age = (n - 2) * remaining_avg_age) →
n = 11 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cricket_team_members_l1274_127406


namespace factorize_expression_l1274_127430

theorem factorize_expression (a : ℝ) : 
  (2 * a + 1) * a - 4 * a - 2 = (2 * a + 1) * (a - 2) :=
by 
  -- proof is skipped with sorry
  sorry

end factorize_expression_l1274_127430


namespace radius_squared_of_intersection_circle_l1274_127480

def parabola1 (x y : ℝ) := y = (x - 2) ^ 2
def parabola2 (x y : ℝ) := x + 6 = (y - 5) ^ 2

theorem radius_squared_of_intersection_circle
    (x y : ℝ)
    (h₁ : parabola1 x y)
    (h₂ : parabola2 x y) :
    ∃ r, r ^ 2 = 83 / 4 :=
sorry

end radius_squared_of_intersection_circle_l1274_127480


namespace thabo_hardcover_books_l1274_127411

theorem thabo_hardcover_books:
  ∃ (H P F : ℕ), H + P + F = 280 ∧ P = H + 20 ∧ F = 2 * P ∧ H = 55 := by
  sorry

end thabo_hardcover_books_l1274_127411


namespace solve_y_equation_l1274_127460

theorem solve_y_equation :
  ∃ y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 :=
by
  sorry

end solve_y_equation_l1274_127460


namespace find_x_l1274_127461

theorem find_x (x y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 :=
by
  sorry

end find_x_l1274_127461


namespace unique_combination_of_segments_l1274_127421

theorem unique_combination_of_segments :
  ∃! (x y : ℤ), 7 * x + 12 * y = 100 := sorry

end unique_combination_of_segments_l1274_127421


namespace inequality_holds_for_all_x_l1274_127410

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3 / 5 < a ∧ a ≤ 1) :=
by
  sorry

end inequality_holds_for_all_x_l1274_127410


namespace problem1_problem2_l1274_127422

-- Define Set A
def SetA : Set ℝ := { y | ∃ x, (2 ≤ x ∧ x ≤ 3) ∧ y = -2^x }

-- Define Set B parameterized by a
def SetB (a : ℝ) : Set ℝ := { x | x^2 + 3 * x - a^2 - 3 * a > 0 }

-- Problem 1: Prove that when a = 4, A ∩ B = {-8 < x < -7}
theorem problem1 : A ∩ SetB 4 = { x | -8 < x ∧ x < -7 } :=
sorry

-- Problem 2: Prove the range of a for which "x ∈ A" is a sufficient but not necessary condition for "x ∈ B"
theorem problem2 : ∀ a : ℝ, (∀ x, x ∈ SetA → x ∈ SetB a) → -4 < a ∧ a < 1 :=
sorry

end problem1_problem2_l1274_127422


namespace find_width_l1274_127470

-- Definitions and Conditions
def length : ℝ := 6
def depth : ℝ := 2
def total_surface_area : ℝ := 104

-- Statement to prove the width
theorem find_width (width : ℝ) (h : 12 * width + 4 * width + 24 = total_surface_area) : width = 5 := 
by { 
  -- lean 4 statement only, proof omitted
  sorry 
}

end find_width_l1274_127470


namespace problem_divisible_by_1946_l1274_127484

def F (n : ℕ) : ℤ := 1492 ^ n - 1770 ^ n - 1863 ^ n + 2141 ^ n

theorem problem_divisible_by_1946 
  (n : ℕ) 
  (hn : n ≤ 1945) : 
  1946 ∣ F n :=
sorry

end problem_divisible_by_1946_l1274_127484


namespace complement_intersection_l1274_127492

open Finset

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 3, 4}
def B : Finset ℕ := {3, 5}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by sorry

end complement_intersection_l1274_127492


namespace cos_alpha_minus_7pi_over_2_l1274_127437

-- Given conditions
variable (α : Real) (h : Real.sin α = 3/5)

-- Statement to prove
theorem cos_alpha_minus_7pi_over_2 : Real.cos (α - 7 * Real.pi / 2) = -3/5 :=
by
  sorry

end cos_alpha_minus_7pi_over_2_l1274_127437


namespace min_days_to_plant_trees_l1274_127449

theorem min_days_to_plant_trees (n : ℕ) (h : 2 ≤ n) :
  (2 ^ (n + 1) - 2 ≥ 1000) ↔ (n ≥ 9) :=
by sorry

end min_days_to_plant_trees_l1274_127449


namespace sin_gt_cos_interval_l1274_127418

theorem sin_gt_cos_interval (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x > Real.cos x) : 
  Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4) :=
by
  sorry

end sin_gt_cos_interval_l1274_127418


namespace pete_travel_time_l1274_127405

-- Definitions for the given conditions
def map_distance := 5.0          -- in inches
def scale := 0.05555555555555555 -- in inches per mile
def speed := 60.0                -- in miles per hour
def real_distance := map_distance / scale

-- The theorem to state the proof problem
theorem pete_travel_time : 
  real_distance = 90 → -- Based on condition deduced from earlier
  real_distance / speed = 1.5 := 
by 
  intro h1
  rw[h1]
  norm_num
  sorry

end pete_travel_time_l1274_127405


namespace sum_of_numbers_l1274_127400

theorem sum_of_numbers (x : ℝ) 
  (h_ratio : ∃ x, (2 * x) / x = 2 ∧ (3 * x) / x = 3)
  (h_squares : x^2 + (2 * x)^2 + (3 * x)^2 = 2744) :
  x + 2 * x + 3 * x = 84 :=
by
  sorry

end sum_of_numbers_l1274_127400


namespace paul_earns_from_license_plates_l1274_127409

theorem paul_earns_from_license_plates
  (plates_from_40_states : ℕ)
  (total_50_states : ℕ)
  (reward_per_percentage_point : ℕ)
  (h1 : plates_from_40_states = 40)
  (h2 : total_50_states = 50)
  (h3 : reward_per_percentage_point = 2) :
  (40 / 50) * 100 * 2 = 160 := 
sorry

end paul_earns_from_license_plates_l1274_127409


namespace calculate_amount_left_l1274_127426

def base_income : ℝ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℝ := 500
def utilities : ℝ := 100
def food : ℝ := 300
def miscellaneous_percentage : ℝ := 0.10
def savings_percentage : ℝ := 0.07
def investment_percentage : ℝ := 0.05
def medical_expense : ℝ := 250
def tax_percentage : ℝ := 0.15

def total_income (base_income : ℝ) (bonus_percentage : ℝ) : ℝ :=
  base_income + (bonus_percentage * base_income)

def taxes (base_income : ℝ) (tax_percentage : ℝ) : ℝ :=
  tax_percentage * base_income

def total_fixed_expenses (rent : ℝ) (utilities : ℝ) (food : ℝ) : ℝ :=
  rent + utilities + food

def public_transport_expense (total_income : ℝ) (public_transport_percentage : ℝ) : ℝ :=
  public_transport_percentage * total_income

def miscellaneous_expense (total_income : ℝ) (miscellaneous_percentage : ℝ) : ℝ :=
  miscellaneous_percentage * total_income

def variable_expenses (public_transport_expense : ℝ) (miscellaneous_expense : ℝ) : ℝ :=
  public_transport_expense + miscellaneous_expense

def savings (total_income : ℝ) (savings_percentage : ℝ) : ℝ :=
  savings_percentage * total_income

def investment (total_income : ℝ) (investment_percentage : ℝ) : ℝ :=
  investment_percentage * total_income

def total_savings_investments (savings : ℝ) (investment : ℝ) : ℝ :=
  savings + investment

def total_expenses_contributions 
  (fixed_expenses : ℝ) 
  (variable_expenses : ℝ) 
  (medical_expense : ℝ) 
  (total_savings_investments : ℝ) : ℝ :=
  fixed_expenses + variable_expenses + medical_expense + total_savings_investments

def amount_left (income_after_taxes : ℝ) (total_expenses_contributions : ℝ) : ℝ :=
  income_after_taxes - total_expenses_contributions

theorem calculate_amount_left 
  (base_income : ℝ)
  (bonus_percentage : ℝ)
  (public_transport_percentage : ℝ)
  (rent : ℝ)
  (utilities : ℝ)
  (food : ℝ)
  (miscellaneous_percentage : ℝ)
  (savings_percentage : ℝ)
  (investment_percentage : ℝ)
  (medical_expense : ℝ)
  (tax_percentage : ℝ)
  (total_income : ℝ := total_income base_income bonus_percentage)
  (taxes : ℝ := taxes base_income tax_percentage)
  (income_after_taxes : ℝ := total_income - taxes)
  (fixed_expenses : ℝ := total_fixed_expenses rent utilities food)
  (public_transport_expense : ℝ := public_transport_expense total_income public_transport_percentage)
  (miscellaneous_expense : ℝ := miscellaneous_expense total_income miscellaneous_percentage)
  (variable_expenses : ℝ := variable_expenses public_transport_expense miscellaneous_expense)
  (savings : ℝ := savings total_income savings_percentage)
  (investment : ℝ := investment total_income investment_percentage)
  (total_savings_investments : ℝ := total_savings_investments savings investment)
  (total_expenses_contributions : ℝ := total_expenses_contributions fixed_expenses variable_expenses medical_expense total_savings_investments)
  : amount_left income_after_taxes total_expenses_contributions = 229 := 
sorry

end calculate_amount_left_l1274_127426


namespace alice_outfits_l1274_127423

theorem alice_outfits :
  let trousers := 5
  let shirts := 8
  let jackets := 4
  let shoes := 2
  trousers * shirts * jackets * shoes = 320 :=
by
  sorry

end alice_outfits_l1274_127423


namespace total_pupils_correct_l1274_127489

def number_of_girls : ℕ := 868
def difference_girls_boys : ℕ := 281
def number_of_boys : ℕ := number_of_girls - difference_girls_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

theorem total_pupils_correct : total_pupils = 1455 := by
  sorry

end total_pupils_correct_l1274_127489


namespace total_worth_of_stock_l1274_127441

theorem total_worth_of_stock (W : ℝ) 
    (h1 : 0.2 * W * 0.1 = 0.02 * W)
    (h2 : 0.6 * (0.8 * W) * 0.05 = 0.024 * W)
    (h3 : 0.2 * (0.8 * W) = 0.16 * W)
    (h4 : (0.024 * W) - (0.02 * W) = 400) 
    : W = 100000 := 
sorry

end total_worth_of_stock_l1274_127441


namespace smallest_total_cells_marked_l1274_127455

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end smallest_total_cells_marked_l1274_127455


namespace fraction_ordering_l1274_127434

theorem fraction_ordering:
  (6 / 22) < (5 / 17) ∧ (5 / 17) < (8 / 24) :=
by
  sorry

end fraction_ordering_l1274_127434


namespace part1_part2_l1274_127488

open Set

variable {U : Type} [TopologicalSpace U]

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def set_B (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem part1 (k : ℝ) (hk : k = 1) :
  A ∩ (univ \ set_B k) = {x | 1 < x ∧ x < 3} :=
by
  sorry

theorem part2 (k : ℝ) (h : set_A ∩ set_B k ≠ ∅) :
  k ≥ -1 :=
by
  sorry

end part1_part2_l1274_127488


namespace solve_equation_l1274_127445

theorem solve_equation (x y : ℤ) (h : 3 * (y - 2) = 5 * (x - 1)) :
  (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
sorry

end solve_equation_l1274_127445


namespace sum_largest_smallest_prime_factors_1155_l1274_127464

theorem sum_largest_smallest_prime_factors_1155 : 
  ∃ smallest largest : ℕ, 
  smallest ∣ 1155 ∧ largest ∣ 1155 ∧ 
  Prime smallest ∧ Prime largest ∧ 
  smallest <= largest ∧ 
  (∀ p : ℕ, p ∣ 1155 → Prime p → (smallest ≤ p ∧ p ≤ largest)) ∧ 
  (smallest + largest = 14) := 
by {
  sorry
}

end sum_largest_smallest_prime_factors_1155_l1274_127464


namespace harvest_season_weeks_l1274_127452

-- Definitions based on given conditions
def weekly_earnings : ℕ := 491
def weekly_rent : ℕ := 216
def total_savings : ℕ := 324775

-- Definition to calculate net earnings per week
def net_earnings_per_week (earnings rent : ℕ) : ℕ :=
  earnings - rent

-- Definition to calculate number of weeks
def number_of_weeks (savings net_earnings : ℕ) : ℕ :=
  savings / net_earnings

theorem harvest_season_weeks :
  number_of_weeks total_savings (net_earnings_per_week weekly_earnings weekly_rent) = 1181 :=
by
  sorry

end harvest_season_weeks_l1274_127452


namespace thor_jumps_to_exceed_29000_l1274_127477

theorem thor_jumps_to_exceed_29000 :
  ∃ (n : ℕ), (3 ^ n) > 29000 ∧ n = 10 := sorry

end thor_jumps_to_exceed_29000_l1274_127477


namespace cake_pieces_in_pan_l1274_127497

theorem cake_pieces_in_pan :
  (24 * 30) / (3 * 2) = 120 := by
  sorry

end cake_pieces_in_pan_l1274_127497


namespace net_rate_of_pay_l1274_127440

/-- The net rate of pay in dollars per hour for a truck driver after deducting gasoline expenses. -/
theorem net_rate_of_pay
  (hrs : ℕ) (speed : ℕ) (miles_per_gallon : ℕ) (pay_per_mile : ℚ) (cost_per_gallon : ℚ) 
  (H1 : hrs = 3)
  (H2 : speed = 50)
  (H3 : miles_per_gallon = 25)
  (H4 : pay_per_mile = 0.6)
  (H5 : cost_per_gallon = 2.50) :
  pay_per_mile * (hrs * speed) - cost_per_gallon * ((hrs * speed) / miles_per_gallon) = 25 * hrs :=
by sorry

end net_rate_of_pay_l1274_127440


namespace gumballs_multiple_purchased_l1274_127499

-- Definitions
def joanna_initial : ℕ := 40
def jacques_initial : ℕ := 60
def final_each : ℕ := 250

-- Proof statement
theorem gumballs_multiple_purchased (m : ℕ) :
  (joanna_initial + joanna_initial * m) + (jacques_initial + jacques_initial * m) = 2 * final_each →
  m = 4 :=
by 
  sorry

end gumballs_multiple_purchased_l1274_127499


namespace find_b_l1274_127494

theorem find_b (a b : ℕ) (h1 : (a + b) % 10 = 5) (h2 : (a + b) % 7 = 4) : b = 2 := 
sorry

end find_b_l1274_127494


namespace inequality_pos_xy_l1274_127473

theorem inequality_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := 
by {
    sorry
}

end inequality_pos_xy_l1274_127473


namespace angles_relation_l1274_127427

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation_l1274_127427


namespace find_g_at_1_l1274_127436

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem find_g_at_1 : 
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → 
  g 1 = 7 := 
by
  intro h
  -- Proof goes here
  sorry

end find_g_at_1_l1274_127436


namespace paint_grid_condition_l1274_127425

variables {a b c d e A B C D E : ℕ}

def is_valid (n : ℕ) : Prop := n = 2 ∨ n = 3

theorem paint_grid_condition 
  (ha : is_valid a) (hb : is_valid b) (hc : is_valid c) 
  (hd : is_valid d) (he : is_valid e) (hA : is_valid A) 
  (hB : is_valid B) (hC : is_valid C) (hD : is_valid D) 
  (hE : is_valid E) :
  a + b + c + d + e = A + B + C + D + E :=
sorry

end paint_grid_condition_l1274_127425


namespace geom_seq_solution_l1274_127467

theorem geom_seq_solution (a b x y : ℝ) 
  (h1 : x * (1 + y + y^2) = a) 
  (h2 : x^2 * (1 + y^2 + y^4) = b) :
  x = 1 / (4 * a) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨ 
  x = 1 / (4 * a) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∧
  y = 1 / (2 * (a^2 - b)) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨
  y = 1 / (2 * (a^2 - b)) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) := 
  sorry

end geom_seq_solution_l1274_127467


namespace smallest_common_multiple_five_digit_l1274_127431

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def smallest_five_digit_multiple_of_3_and_5 (x : ℕ) : Prop :=
  is_multiple x 3 ∧ is_multiple x 5 ∧ 10000 ≤ x ∧ x ≤ 99999 ∧ (∀ y, (10000 ≤ y ∧ y ≤ 99999 ∧ is_multiple y 3 ∧ is_multiple y 5) → x ≤ y)

theorem smallest_common_multiple_five_digit : smallest_five_digit_multiple_of_3_and_5 10005 :=
sorry

end smallest_common_multiple_five_digit_l1274_127431


namespace relationship_between_a_b_c_l1274_127475

noncomputable def a : ℝ := 1 / 3
noncomputable def b : ℝ := Real.sin (1 / 3)
noncomputable def c : ℝ := 1 / Real.pi

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l1274_127475


namespace Felicity_used_23_gallons_l1274_127481

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l1274_127481


namespace david_spent_difference_l1274_127458

-- Define the initial amount, remaining amount, amount spent and the correct answer
def initial_amount : Real := 1800
def remaining_amount : Real := 500
def spent_amount : Real := initial_amount - remaining_amount
def correct_difference : Real := spent_amount - remaining_amount

-- Prove that the difference between the amount spent and the remaining amount is $800
theorem david_spent_difference : correct_difference = 800 := by
  sorry

end david_spent_difference_l1274_127458


namespace find_y_l1274_127450

/-- Given (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10) and x = 12, prove that y = 10 -/
theorem find_y (x y : ℕ) (h : (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10)) (hx : x = 12) : y = 10 :=
by
  sorry

end find_y_l1274_127450


namespace probability_one_instrument_l1274_127493

-- Definitions based on conditions
def total_people : Nat := 800
def play_at_least_one : Nat := total_people / 5
def play_two_or_more : Nat := 32
def play_exactly_one : Nat := play_at_least_one - play_two_or_more

-- Target statement to prove the equivalence
theorem probability_one_instrument: (play_exactly_one : ℝ) / (total_people : ℝ) = 0.16 := by
  sorry

end probability_one_instrument_l1274_127493


namespace trapezoid_area_l1274_127403

-- Definitions of the problem's conditions
def a : ℕ := 4
def b : ℕ := 8
def h : ℕ := 3

-- Lean statement to prove the area of the trapezoid is 18 square centimeters
theorem trapezoid_area : (a + b) * h / 2 = 18 := by
  sorry

end trapezoid_area_l1274_127403


namespace sequence_mod_100_repeats_l1274_127414

theorem sequence_mod_100_repeats (a0 : ℕ) : ∃ k l, k ≠ l ∧ (∃ seq : ℕ → ℕ, seq 0 = a0 ∧ (∀ n, seq (n + 1) = seq n + 54 ∨ seq (n + 1) = seq n + 77) ∧ (seq k % 100 = seq l % 100)) :=
by 
  sorry

end sequence_mod_100_repeats_l1274_127414


namespace meeting_point_l1274_127463

theorem meeting_point :
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  (Paul_start.1 + Lisa_start.1) / 2 = -2 ∧ (Paul_start.2 + Lisa_start.2) / 2 = 3 :=
by
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  have x_coord : (Paul_start.1 + Lisa_start.1) / 2 = -2 := sorry
  have y_coord : (Paul_start.2 + Lisa_start.2) / 2 = 3 := sorry
  exact ⟨x_coord, y_coord⟩

end meeting_point_l1274_127463


namespace calculation_simplifies_l1274_127486

theorem calculation_simplifies :
  120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end calculation_simplifies_l1274_127486


namespace smallest_k_for_a_l1274_127401

theorem smallest_k_for_a (a n : ℕ) (h : 10 ^ 2013 ≤ a^n ∧ a^n < 10 ^ 2014) : ∀ k : ℕ, k < 46 → ∃ n : ℕ, (10 ^ (k - 1)) ≤ a ∧ a < 10 ^ k :=
by sorry

end smallest_k_for_a_l1274_127401


namespace gcd_180_450_l1274_127459

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l1274_127459


namespace expansion_sum_l1274_127476

theorem expansion_sum (A B C : ℤ) (h1 : A = (2 - 1)^10) (h2 : B = (2 + 0)^10) (h3 : C = -5120) : 
A + B + C = -4095 :=
by 
  sorry

end expansion_sum_l1274_127476
