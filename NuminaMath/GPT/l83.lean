import Mathlib

namespace g_three_eighths_l83_83848

variable (g : ℝ → ℝ)

-- Conditions
axiom g_zero : g 0 = 0
axiom monotonic : ∀ {x y : ℝ}, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y
axiom symmetry : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x
axiom scaling : ∀ {x : ℝ}, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3

-- The theorem statement we need to prove
theorem g_three_eighths : g (3 / 8) = 2 / 9 :=
sorry

end g_three_eighths_l83_83848


namespace intersection_A_B_l83_83572

def A : Set ℝ := {x | 2*x - 1 ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1/2} := 
by 
  sorry

end intersection_A_B_l83_83572


namespace circle_n_gon_area_ineq_l83_83827

variable {n : ℕ} {S S1 S2 : ℝ}

theorem circle_n_gon_area_ineq (h1 : S1 > 0) (h2 : S > 0) (h3 : S2 > 0) : 
  S * S = S1 * S2 := 
sorry

end circle_n_gon_area_ineq_l83_83827


namespace total_animals_peppersprayed_l83_83780

-- Define the conditions
def number_of_raccoons : ℕ := 12
def squirrels_vs_raccoons : ℕ := 6
def number_of_squirrels (raccoons : ℕ) (factor : ℕ) : ℕ := raccoons * factor

-- Define the proof statement
theorem total_animals_peppersprayed : 
  number_of_squirrels number_of_raccoons squirrels_vs_raccoons + number_of_raccoons = 84 :=
by
  -- The proof would go here
  sorry

end total_animals_peppersprayed_l83_83780


namespace abs_abc_eq_one_l83_83246

variable (a b c : ℝ)

-- Conditions
axiom distinct_nonzero : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)
axiom condition : a^2 + 1/(b^2) = b^2 + 1/(c^2) ∧ b^2 + 1/(c^2) = c^2 + 1/(a^2)

theorem abs_abc_eq_one : |a * b * c| = 1 :=
by
  sorry

end abs_abc_eq_one_l83_83246


namespace set_equality_l83_83031

-- Define the universe U
def U := ℝ

-- Define the set M
def M := {x : ℝ | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N := {x : ℝ | x > 1}

-- Define the set we want to prove is equal to the intersection of M and N
def target_set := {x : ℝ | 1 < x ∧ x ≤ 2}

theorem set_equality : target_set = M ∩ N := 
by sorry

end set_equality_l83_83031


namespace volume_of_cube_l83_83525

theorem volume_of_cube (SA : ℝ) (h : SA = 486) : ∃ V : ℝ, V = 729 :=
by
  sorry

end volume_of_cube_l83_83525


namespace prove_product_reduced_difference_l83_83027

-- We are given two numbers x and y such that:
variable (x y : ℚ)
-- 1. The sum of the numbers is 6
axiom sum_eq_six : x + y = 6
-- 2. The quotient of the larger number by the smaller number is 6
axiom quotient_eq_six : x / y = 6

-- We need to prove that the product of these two numbers reduced by their difference is 6/49
theorem prove_product_reduced_difference (x y : ℚ) 
  (sum_eq_six : x + y = 6) (quotient_eq_six : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 := 
by
  sorry

end prove_product_reduced_difference_l83_83027


namespace volume_of_box_l83_83472

-- Defining the initial parameters of the problem
def length_sheet := 48
def width_sheet := 36
def side_length_cut_square := 3

-- Define the transformed dimensions after squares are cut off
def length_box := length_sheet - 2 * side_length_cut_square
def width_box := width_sheet - 2 * side_length_cut_square
def height_box := side_length_cut_square

-- The target volume of the box
def target_volume := 3780

-- Prove that the volume of the box is equal to the target volume
theorem volume_of_box : length_box * width_box * height_box = target_volume := by
  -- Calculate the expected volume
  -- Expected volume = 42 m * 30 m * 3 m
  -- Which equals 3780 m³
  sorry

end volume_of_box_l83_83472


namespace gold_coins_percent_l83_83631

variable (total_objects beads papers coins silver_gold total_gold : ℝ)
variable (h1 : total_objects = 100)
variable (h2 : beads = 15)
variable (h3 : papers = 10)
variable (h4 : silver_gold = 30)
variable (h5 : total_gold = 52.5)

theorem gold_coins_percent : (total_objects - beads - papers) * (100 - silver_gold) / 100 = total_gold :=
by 
  -- Insert proof here
  sorry

end gold_coins_percent_l83_83631


namespace remaining_digits_product_l83_83365

theorem remaining_digits_product (a b c : ℕ)
  (h1 : (a + b) % 10 = c % 10)
  (h2 : (b + c) % 10 = a % 10)
  (h3 : (c + a) % 10 = b % 10) :
  ((a * b * c) % 1000 = 0 ∨
   (a * b * c) % 1000 = 250 ∨
   (a * b * c) % 1000 = 500 ∨
   (a * b * c) % 1000 = 750) :=
sorry

end remaining_digits_product_l83_83365


namespace johns_actual_marks_l83_83491

def actual_marks (T : ℝ) (x : ℝ) (incorrect : ℝ) (students : ℕ) (avg_increase : ℝ) :=
  (incorrect = 82) ∧ (students = 80) ∧ (avg_increase = 1/2) ∧
  ((T + incorrect) / students = (T + x) / students + avg_increase)

theorem johns_actual_marks (T : ℝ) :
  ∃ x : ℝ, actual_marks T x 82 80 (1/2) ∧ x = 42 :=
by
  sorry

end johns_actual_marks_l83_83491


namespace arithmetic_geometric_seq_proof_l83_83348

theorem arithmetic_geometric_seq_proof
  (a1 a2 b1 b2 b3 : ℝ)
  (h1 : a1 - a2 = -1)
  (h2 : 1 * (b2 * b2) = 4)
  (h3 : b2 > 0) :
  (a1 - a2) / b2 = -1 / 2 :=
by
  sorry

end arithmetic_geometric_seq_proof_l83_83348


namespace find_other_integer_l83_83859

theorem find_other_integer (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : y = 14 ∨ x = 14 :=
  sorry

end find_other_integer_l83_83859


namespace multiple_of_cans_of_corn_l83_83517

theorem multiple_of_cans_of_corn (peas corn : ℕ) (h1 : peas = 35) (h2 : corn = 10) (h3 : peas = 10 * x + 15) : x = 2 := 
by
  sorry

end multiple_of_cans_of_corn_l83_83517


namespace point_P_lies_on_x_axis_l83_83189

noncomputable def point_on_x_axis (x : ℝ) : Prop :=
  (0 = (0 : ℝ)) -- This is a placeholder definition stating explicitly that point lies on the x-axis

theorem point_P_lies_on_x_axis (x : ℝ) : point_on_x_axis x :=
by
  sorry

end point_P_lies_on_x_axis_l83_83189


namespace find_m_l83_83188

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

theorem find_m (h : (1 + 3, m - 2) = (4, m - 2) ∧ (4 * 3 + (m - 2) * (-2) = 0)) : m = 8 := by
  sorry

end find_m_l83_83188


namespace Alan_total_cost_is_84_l83_83953

theorem Alan_total_cost_is_84 :
  let D := 2 * 12
  let A := 12
  let cost_other := 2 * D + A
  let M := 0.4 * cost_other
  2 * D + A + M = 84 := by
    sorry

end Alan_total_cost_is_84_l83_83953


namespace point_coordinates_l83_83437

def point : Type := ℝ × ℝ

def x_coordinate (P : point) : ℝ := P.1

def y_coordinate (P : point) : ℝ := P.2

theorem point_coordinates (P : point) (h1 : x_coordinate P = -3) (h2 : abs (y_coordinate P) = 5) :
  P = (-3, 5) ∨ P = (-3, -5) :=
by
  sorry

end point_coordinates_l83_83437


namespace comic_books_l83_83785

variables (x y : ℤ)

def condition1 (x y : ℤ) : Prop := y + 7 = 5 * (x - 7)
def condition2 (x y : ℤ) : Prop := y - 9 = 3 * (x + 9)

theorem comic_books (x y : ℤ) (h₁ : condition1 x y) (h₂ : condition2 x y) : x = 39 ∧ y = 153 :=
by
  sorry

end comic_books_l83_83785


namespace outer_circle_radius_l83_83668

theorem outer_circle_radius (r R : ℝ) (hr : r = 4)
  (radius_increase : ∀ R, R' = 1.5 * R)
  (radius_decrease : ∀ r, r' = 0.75 * r)
  (area_increase : ∀ (A1 A2 : ℝ), A2 = 3.6 * A1)
  (initial_area : ∀ A1, A1 = π * R^2 - π * r^2)
  (new_area : ∀ A2 R' r', A2 = π * R'^2 - π * r'^2) :
  R = 6 := sorry

end outer_circle_radius_l83_83668


namespace students_end_year_10_l83_83734

def students_at_end_of_year (initial_students : ℕ) (left_students : ℕ) (increase_percent : ℕ) : ℕ :=
  let remaining_students := initial_students - left_students
  let increased_students := (remaining_students * increase_percent) / 100
  remaining_students + increased_students

theorem students_end_year_10 : 
  students_at_end_of_year 10 4 70 = 10 := by 
  sorry

end students_end_year_10_l83_83734


namespace Wolfgang_marble_count_l83_83725

theorem Wolfgang_marble_count
  (W L M : ℝ)
  (hL : L = 5/4 * W)
  (hM : M = 2/3 * (W + L))
  (hTotal : W + L + M = 60) :
  W = 16 :=
by {
  sorry
}

end Wolfgang_marble_count_l83_83725


namespace gcd_of_72_90_120_l83_83119

theorem gcd_of_72_90_120 : Nat.gcd (Nat.gcd 72 90) 120 = 6 := 
by 
  have h1 : 72 = 2^3 * 3^2 := by norm_num
  have h2 : 90 = 2 * 3^2 * 5 := by norm_num
  have h3 : 120 = 2^3 * 3 * 5 := by norm_num
  sorry

end gcd_of_72_90_120_l83_83119


namespace slope_to_y_intercept_ratio_l83_83608

theorem slope_to_y_intercept_ratio (m b : ℝ) (c : ℝ) (h1 : m = c * b) (h2 : 2 * m + b = 0) : c = -1 / 2 :=
by sorry

end slope_to_y_intercept_ratio_l83_83608


namespace triangle_shortest_side_l83_83655

theorem triangle_shortest_side (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) 
    (r : ℝ) (h3 : r = 5) 
    (h4 : a = 4) (h5 : b = 10)
    (circumcircle_tangent_property : 2 * (4 + 10) * r = 30) :
  min a (min b c) = 30 :=
by 
  sorry

end triangle_shortest_side_l83_83655


namespace point_on_hyperbola_l83_83420

theorem point_on_hyperbola (p r : ℝ) (h1 : p > 0) (h2 : r > 0)
  (h_el : ∀ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1)
  (h_par : ∀ (x y : ℝ), y^2 = 2 * p * x)
  (h_circum : ∀ (a b c : ℝ), a = 2 * r - 2 * p) :
  r^2 - p^2 = 1 := sorry

end point_on_hyperbola_l83_83420


namespace xiaoming_interview_pass_probability_l83_83256

theorem xiaoming_interview_pass_probability :
  let p_correct := 0.7
  let p_fail_per_attempt := 1 - p_correct
  let p_fail_all_attempts := p_fail_per_attempt ^ 3
  let p_pass_interview := 1 - p_fail_all_attempts
  p_pass_interview = 0.973 := by
    let p_correct := 0.7
    let p_fail_per_attempt := 1 - p_correct
    let p_fail_all_attempts := p_fail_per_attempt ^ 3
    let p_pass_interview := 1 - p_fail_all_attempts
    sorry

end xiaoming_interview_pass_probability_l83_83256


namespace smallest_k_correct_l83_83204

noncomputable def smallest_k (n m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) : ℕ :=
    6

theorem smallest_k_correct (n : ℕ) (m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) :
  64 ^ smallest_k n m hn hm + 32 ^ m > 4 ^ (16 + n) :=
sorry

end smallest_k_correct_l83_83204


namespace cost_of_one_book_l83_83085

theorem cost_of_one_book (s b c : ℕ) (h1 : s > 18) (h2 : b > 1) (h3 : c > b) (h4 : s * b * c = 3203) (h5 : s ≤ 36) : c = 11 :=
by
  sorry

end cost_of_one_book_l83_83085


namespace geometric_sequence_a7_l83_83097

theorem geometric_sequence_a7
  (a : ℕ → ℤ)
  (is_geom_seq : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h1 : a 1 = -16)
  (h4 : a 4 = 8) :
  a 7 = -4 := 
sorry

end geometric_sequence_a7_l83_83097


namespace smaller_angle_at_7_30_is_45_degrees_l83_83626

noncomputable def calculateAngle (hour minute : Nat) : Real :=
  let minuteAngle := (minute * 6 : Real)
  let hourAngle := (hour % 12 * 30 : Real) + (minute / 60 * 30 : Real)
  let diff := abs (hourAngle - minuteAngle)
  if diff > 180 then 360 - diff else diff

theorem smaller_angle_at_7_30_is_45_degrees :
  calculateAngle 7 30 = 45 := 
sorry

end smaller_angle_at_7_30_is_45_degrees_l83_83626


namespace gain_percent_l83_83257

-- Let C be the cost price of one chocolate
-- Let S be the selling price of one chocolate
-- Given: 35 * C = 21 * S
-- Prove: The gain percent is 66.67%

theorem gain_percent (C S : ℝ) (h : 35 * C = 21 * S) : (S - C) / C * 100 = 200 / 3 :=
by sorry

end gain_percent_l83_83257


namespace absolute_value_inequality_solution_l83_83029

theorem absolute_value_inequality_solution (x : ℝ) : |2*x - 1| < 3 ↔ -1 < x ∧ x < 2 := 
sorry

end absolute_value_inequality_solution_l83_83029


namespace train_length_is_120_l83_83111

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let total_distance := speed_ms * time_s
  total_distance - bridge_length_m

theorem train_length_is_120 :
  length_of_train 70 13.884603517432893 150 = 120 :=
by
  sorry

end train_length_is_120_l83_83111


namespace parallel_lines_value_of_a_l83_83356

theorem parallel_lines_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, ax + (a+2)*y + 2 = 0 → x + a*y + 1 = 0 → ∀ m n : ℝ, ax + (a + 2)*n + 2 = 0 → x + a*n + 1 = 0) →
  a = -1 := 
sorry

end parallel_lines_value_of_a_l83_83356


namespace moles_of_Na2SO4_formed_l83_83267

/-- 
Given the following conditions:
1. 1 mole of H2SO4 reacts with 2 moles of NaOH.
2. In the presence of 0.5 moles of HCl and 0.5 moles of KOH.
3. At a temperature of 25°C and a pressure of 1 atm.
Prove that the moles of Na2SO4 formed is 1 mole.
-/

theorem moles_of_Na2SO4_formed
  (H2SO4 : ℝ) -- moles of H2SO4
  (NaOH : ℝ) -- moles of NaOH
  (HCl : ℝ) -- moles of HCl
  (KOH : ℝ) -- moles of KOH
  (T : ℝ) -- temperature in °C
  (P : ℝ) -- pressure in atm
  : H2SO4 = 1 ∧ NaOH = 2 ∧ HCl = 0.5 ∧ KOH = 0.5 ∧ T = 25 ∧ P = 1 → 
  ∃ Na2SO4 : ℝ, Na2SO4 = 1 :=
by
  sorry

end moles_of_Na2SO4_formed_l83_83267


namespace max_value_of_k_l83_83786

theorem max_value_of_k (m : ℝ) (h₁ : 0 < m) (h₂ : m < 1/2) : 
  (1 / m + 2 / (1 - 2 * m)) ≥ 8 :=
sorry

end max_value_of_k_l83_83786


namespace Brad_pumpkin_weight_l83_83537

theorem Brad_pumpkin_weight (B : ℝ)
  (h1 : ∃ J : ℝ, J = B / 2)
  (h2 : ∃ Be : ℝ, Be = 4 * (B / 2))
  (h3 : ∃ Be J : ℝ, Be - J = 81) : B = 54 := by
  obtain ⟨J, hJ⟩ := h1
  obtain ⟨Be, hBe⟩ := h2
  obtain ⟨_, hBeJ⟩ := h3
  sorry

end Brad_pumpkin_weight_l83_83537


namespace cubic_common_roots_l83_83360

theorem cubic_common_roots:
  ∃ (c d : ℝ), 
  (∀ r s : ℝ,
    r ≠ s ∧ 
    (r ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + c * x^2 + 16 * x + 9 = 0}) ∧ 
    (r ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0}) ∧
    (s ∈ {x : ℝ | x^3 + d * x^2 + 20 * x + 12 = 0})) → 
  c = 8 ∧ d = 9 := 
by
  sorry

end cubic_common_roots_l83_83360


namespace only_zero_and_one_square_equal_themselves_l83_83395

theorem only_zero_and_one_square_equal_themselves (x: ℝ) : (x^2 = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end only_zero_and_one_square_equal_themselves_l83_83395


namespace problems_per_page_l83_83036

theorem problems_per_page (total_problems finished_problems remaining_pages : Nat) (h1 : total_problems = 101) 
  (h2 : finished_problems = 47) (h3 : remaining_pages = 6) :
  (total_problems - finished_problems) / remaining_pages = 9 :=
by
  sorry

end problems_per_page_l83_83036


namespace quadratic_bound_l83_83897

theorem quadratic_bound (a b c : ℝ) :
  (∀ (u : ℝ), |u| ≤ 10 / 11 → ∃ (v : ℝ), |u - v| ≤ 1 / 11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2 := by
  sorry

end quadratic_bound_l83_83897


namespace monotonic_function_a_range_l83_83229

theorem monotonic_function_a_range :
  ∀ (f : ℝ → ℝ) (a : ℝ), 
  (f x = x^2 + (2 * a + 1) * x + 1) →
  (∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (f x ≤ f y ∨ f x ≥ f y)) ↔ 
  (a ∈ Set.Ici (-3/2) ∪ Set.Iic (-5/2)) := 
sorry

end monotonic_function_a_range_l83_83229


namespace geometric_sequence_first_term_l83_83326

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end geometric_sequence_first_term_l83_83326


namespace complement_U_M_l83_83374

theorem complement_U_M :
  let U := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  let M := {x : ℤ | ∃ k : ℤ, x = 4 * k}
  {x | x ∈ U ∧ x ∉ M} = {x : ℤ | ∃ k : ℤ, x = 4 * k - 2} :=
by
  sorry

end complement_U_M_l83_83374


namespace range_of_real_number_l83_83155

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B (a : ℝ) : Set ℝ := {-1, -3, a}
def complement_A : Set ℝ := {x | x ≥ 0}

theorem range_of_real_number (a : ℝ) (h : (complement_A ∩ (B a)) ≠ ∅) : a ≥ 0 :=
sorry

end range_of_real_number_l83_83155


namespace probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l83_83064

noncomputable def total_members := 42
noncomputable def boys := 28
noncomputable def girls := 14
noncomputable def selected := 6

theorem probability_athlete_A_selected :
  (selected : ℚ) / total_members = 1 / 7 :=
by sorry

theorem number_of_males_selected :
  (selected * (boys : ℚ)) / total_members = 4 :=
by sorry

theorem number_of_females_selected :
  (selected * (girls : ℚ)) / total_members = 2 :=
by sorry

end probability_athlete_A_selected_number_of_males_selected_number_of_females_selected_l83_83064


namespace money_left_is_41_l83_83005

-- Define the amounts saved by Tanner in each month
def savings_september : ℕ := 17
def savings_october : ℕ := 48
def savings_november : ℕ := 25

-- Define the amount spent by Tanner on the video game
def spent_video_game : ℕ := 49

-- Total savings after the three months
def total_savings : ℕ := savings_september + savings_october + savings_november

-- Calculate the money left after spending on the video game
def money_left : ℕ := total_savings - spent_video_game

-- The theorem we need to prove
theorem money_left_is_41 : money_left = 41 := by
  sorry

end money_left_is_41_l83_83005


namespace no_real_solutions_l83_83665

theorem no_real_solutions :
  ∀ z : ℝ, ¬ ((-6 * z + 27) ^ 2 + 4 = -2 * |z|) :=
by
  sorry

end no_real_solutions_l83_83665


namespace find_sum_l83_83099

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Geometric sequence condition
def geometric_seq (a : ℕ → α) (r : α) := ∀ n : ℕ, a (n + 1) = a n * r

theorem find_sum (r : α)
  (h1 : geometric_seq a r)
  (h2 : a 4 + a 7 = 2)
  (h3 : a 5 * a 6 = -8) :
  a 1 + a 10 = -7 := 
sorry

end find_sum_l83_83099


namespace angle_BAC_measure_l83_83127

variable (A B C X Y : Type)
variables (angle_ABC angle_BAC : ℝ)
variables (len_AX len_XY len_YB len_BC : ℝ)

theorem angle_BAC_measure 
  (h1 : AX = XY) 
  (h2 : XY = YB) 
  (h3 : XY = 2 * AX) 
  (h4 : angle_ABC = 150) :
  angle_BAC = 26.25 :=
by
  -- The proof would be required here.
  -- Following the statement as per instructions.
  sorry

end angle_BAC_measure_l83_83127


namespace caleb_spent_more_on_ice_cream_l83_83129

theorem caleb_spent_more_on_ice_cream :
  ∀ (number_of_ic_cartons number_of_fy_cartons : ℕ)
    (cost_per_ic_carton cost_per_fy_carton : ℝ)
    (discount_rate sales_tax_rate : ℝ),
    number_of_ic_cartons = 10 →
    number_of_fy_cartons = 4 →
    cost_per_ic_carton = 4 →
    cost_per_fy_carton = 1 →
    discount_rate = 0.15 →
    sales_tax_rate = 0.05 →
    (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
     (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
      number_of_fy_cartons * cost_per_fy_carton) * sales_tax_rate) -
    (number_of_fy_cartons * cost_per_fy_carton) = 30 :=
by
  intros number_of_ic_cartons number_of_fy_cartons cost_per_ic_carton cost_per_fy_carton discount_rate sales_tax_rate
  sorry

end caleb_spent_more_on_ice_cream_l83_83129


namespace total_gum_correct_l83_83911

def num_cousins : ℕ := 4  -- Number of cousins
def gum_per_cousin : ℕ := 5  -- Pieces of gum per cousin

def total_gum : ℕ := num_cousins * gum_per_cousin  -- Total pieces of gum Kim needs

theorem total_gum_correct : total_gum = 20 :=
by sorry

end total_gum_correct_l83_83911


namespace ratio_of_area_l83_83977

noncomputable def area_ratio (l w r : ℝ) : ℝ :=
  if h1 : 2 * l + 2 * w = 2 * Real.pi * r 
  ∧ l = 2 * w then 
    (l * w) / (Real.pi * r ^ 2) 
  else 
    0

theorem ratio_of_area (l w r : ℝ) 
  (h1 : 2 * l + 2 * w = 2 * Real.pi * r) 
  (h2 : l = 2 * w) :
  area_ratio l w r = 2 * Real.pi / 9 :=
by
  unfold area_ratio
  simp [h1, h2]
  sorry

end ratio_of_area_l83_83977


namespace frac_subtraction_simplified_l83_83382

-- Definitions of the fractions involved.
def frac1 : ℚ := 8 / 19
def frac2 : ℚ := 5 / 57

-- The primary goal is to prove the equality.
theorem frac_subtraction_simplified : frac1 - frac2 = 1 / 3 :=
by {
  -- Proof of the statement.
  sorry
}

end frac_subtraction_simplified_l83_83382


namespace solve_system_for_x_l83_83825

theorem solve_system_for_x :
  ∃ x y : ℝ, (2 * x + y = 4) ∧ (x + 2 * y = 5) ∧ (x = 1) :=
by
  sorry

end solve_system_for_x_l83_83825


namespace inclination_angle_of_line_l83_83663

-- Lean definition for the line equation and inclination angle problem
theorem inclination_angle_of_line : 
  ∃ θ : ℝ, (θ ∈ Set.Ico 0 Real.pi) ∧ (∀ x y: ℝ, x + y - 1 = 0 → Real.tan θ = -1) ∧ θ = 3 * Real.pi / 4 :=
sorry

end inclination_angle_of_line_l83_83663


namespace find_b_value_l83_83300

theorem find_b_value (a b : ℤ) (h₁ : a + 2 * b = 32) (h₂ : |a| > 2) (h₃ : a = 4) : b = 14 :=
by
  -- proof goes here
  sorry

end find_b_value_l83_83300


namespace calculate_f_50_l83_83417

noncomputable def f (x : ℝ) : ℝ := sorry

theorem calculate_f_50 (f : ℝ → ℝ) (h_fun : ∀ x y : ℝ, f (x * y) = y * f x) (h_f2 : f 2 = 10) :
  f 50 = 250 :=
by
  sorry

end calculate_f_50_l83_83417


namespace permutations_BANANA_l83_83541

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l83_83541


namespace opposite_of_neg_11_l83_83041

-- Define the opposite (negative) of a number
def opposite (a : ℤ) : ℤ := -a

-- Prove that the opposite of -11 is 11
theorem opposite_of_neg_11 : opposite (-11) = 11 := 
by
  -- Proof not required, so using sorry as placeholder
  sorry

end opposite_of_neg_11_l83_83041


namespace problem1_problem2_l83_83556

-- Definition for Problem 1
def cube_root_8 : ℝ := 2
def abs_neg5 : ℝ := 5
def pow_neg1_2023 : ℝ := -1

theorem problem1 : cube_root_8 + abs_neg5 + pow_neg1_2023 = 6 := by
  sorry

-- Definitions for Problem 2
structure Point where
  x : ℝ
  y : ℝ

def point1 : Point := { x := 0, y := 1 }
def point2 : Point := { x := 2, y := 5 }

def linear_function (k b x : ℝ) : ℝ := k * x + b

def is_linear_function (p1 p2 : Point) (k b : ℝ) : Prop :=
  p1.y = linear_function k b p1.x ∧ p2.y = linear_function k b p2.x

theorem problem2 : ∃ k b : ℝ, is_linear_function point1 point2 k b ∧ (linear_function k b x = 2 * x + 1) := by
  sorry

end problem1_problem2_l83_83556


namespace ratatouille_cost_per_quart_l83_83828

theorem ratatouille_cost_per_quart:
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let zucchini_pounds := 4
  let zucchini_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let total_quarts := 4
  let eggplants_cost := eggplants_pounds * eggplants_cost_per_pound
  let zucchini_cost := zucchini_pounds * zucchini_cost_per_pound
  let tomatoes_cost := tomatoes_pounds * tomatoes_cost_per_pound
  let onions_cost := onions_pounds * onions_cost_per_pound
  let basil_cost := basil_pounds * (basil_cost_per_half_pound / 0.5)
  let total_cost := eggplants_cost + zucchini_cost + tomatoes_cost + onions_cost + basil_cost
  let cost_per_quart := total_cost / total_quarts
  cost_per_quart = 10.00 :=
  by
    sorry

end ratatouille_cost_per_quart_l83_83828


namespace contribution_proof_l83_83901

theorem contribution_proof (total : ℕ) (a_months b_months : ℕ) (a_total b_total a_received b_received : ℕ) :
  total = 3400 →
  a_months = 12 →
  b_months = 16 →
  a_received = 2070 →
  b_received = 1920 →
  (∃ (a_contributed b_contributed : ℕ), a_contributed = 1800 ∧ b_contributed = 1600) :=
by
  sorry

end contribution_proof_l83_83901


namespace quadratic_completing_square_l83_83974

theorem quadratic_completing_square :
  ∃ (a b c : ℚ), a = 12 ∧ b = 6 ∧ c = 1296 ∧ 12 + 6 + 1296 = 1314 ∧
  (12 * (x + b)^2 + c = 12 * x^2 + 144 * x + 1728) :=
by
  sorry

end quadratic_completing_square_l83_83974


namespace percentage_increase_of_cube_surface_area_l83_83833

-- Basic setup definitions and conditions
variable (a : ℝ)

-- Step 1: Initial surface area
def initial_surface_area : ℝ := 6 * a^2

-- Step 2: New edge length after 50% growth
def new_edge_length : ℝ := 1.5 * a

-- Step 3: New surface area after edge growth
def new_surface_area : ℝ := 6 * (new_edge_length a)^2

-- Step 4: Surface area after scaling by 1.5
def scaled_surface_area : ℝ := new_surface_area a * (1.5)^2

-- Prove the percentage increase
theorem percentage_increase_of_cube_surface_area :
  (scaled_surface_area a - initial_surface_area a) / initial_surface_area a * 100 = 406.25 := by
  sorry

end percentage_increase_of_cube_surface_area_l83_83833


namespace frictional_force_is_12N_l83_83801

-- Given conditions
variables (m1 m2 a μ : ℝ)
-- Constants
def g : ℝ := 9.8

-- Frictional force on the tank
def F_friction : ℝ := μ * m1 * g

-- Proof statement
theorem frictional_force_is_12N (m1_value : m1 = 3) (m2_value : m2 = 15) (a_value : a = 4) (μ_value : μ = 0.6) :
  m1 * a = 12 :=
by
  sorry

end frictional_force_is_12N_l83_83801


namespace hourly_wage_increase_is_10_percent_l83_83931

theorem hourly_wage_increase_is_10_percent :
  ∀ (H W : ℝ), 
    ∀ (H' : ℝ), H' = H * (1 - 0.09090909090909092) →
    (H * W = H' * W') →
    (W' = (100 * W) / 90) := by
  sorry

end hourly_wage_increase_is_10_percent_l83_83931


namespace possible_values_f_l83_83766

noncomputable def f (x y z : ℝ) : ℝ := (y / (y + x)) + (z / (z + y)) + (x / (x + z))

theorem possible_values_f (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x^2 + y^3 = z^4) : 
  1 < f x y z ∧ f x y z < 2 :=
sorry

end possible_values_f_l83_83766


namespace horizontal_asymptote_of_rational_function_l83_83316

theorem horizontal_asymptote_of_rational_function :
  ∀ (x : ℝ), (y = (7 * x^2 - 5) / (4 * x^2 + 6 * x + 3)) → (∃ b : ℝ, b = 7 / 4) :=
by
  intro x y
  sorry

end horizontal_asymptote_of_rational_function_l83_83316


namespace nina_money_l83_83056

theorem nina_money (W M : ℕ) (h1 : 6 * W = M) (h2 : 8 * (W - 2) = M) : M = 48 :=
by
  sorry

end nina_money_l83_83056


namespace find_x_l83_83469

theorem find_x (x y : ℤ) (h1 : y = 3) (h2 : x + 3 * y = 10) : x = 1 :=
by
  sorry

end find_x_l83_83469


namespace alpha_proportional_l83_83836

theorem alpha_proportional (alpha beta gamma : ℝ) (h1 : ∀ β γ, (β = 15 ∧ γ = 3) → α = 5)
    (h2 : beta = 30) (h3 : gamma = 6) : alpha = 2.5 :=
sorry

end alpha_proportional_l83_83836


namespace factorize_x_squared_sub_xy_l83_83819

theorem factorize_x_squared_sub_xy (x y : ℝ) : x^2 - x * y = x * (x - y) :=
sorry

end factorize_x_squared_sub_xy_l83_83819


namespace determine_n_l83_83588

theorem determine_n (x a : ℝ) (n : ℕ)
  (h1 : (n.choose 3) * x^(n-3) * a^3 = 120)
  (h2 : (n.choose 4) * x^(n-4) * a^4 = 360)
  (h3 : (n.choose 5) * x^(n-5) * a^5 = 720) :
  n = 12 :=
sorry

end determine_n_l83_83588


namespace geometric_to_arithmetic_sequence_l83_83396

theorem geometric_to_arithmetic_sequence {a : ℕ → ℝ} (q : ℝ) 
    (h_gt0 : 0 < q) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : ∀ n, a (n + 1) = a n * q)
    (h_arith_seq : 2 * (1 / 2 * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := 
by
  sorry

end geometric_to_arithmetic_sequence_l83_83396


namespace simplify_fraction_result_l83_83646

theorem simplify_fraction_result : (130 / 16900) * 65 = 1 / 2 :=
by sorry

end simplify_fraction_result_l83_83646


namespace minimize_expression_l83_83416

theorem minimize_expression (x : ℝ) (h : 0 < x) : 
  x = 9 ↔ (∀ y : ℝ, 0 < y → x + 81 / x ≤ y + 81 / y) :=
sorry

end minimize_expression_l83_83416


namespace shaded_area_proof_l83_83841

noncomputable def shaded_area (side_length : ℝ) (radius_factor : ℝ) : ℝ :=
  let square_area := side_length * side_length
  let radius := radius_factor * side_length
  let circle_area := Real.pi * (radius * radius)
  square_area - circle_area

theorem shaded_area_proof : shaded_area 8 0.6 = 64 - 23.04 * Real.pi :=
by sorry

end shaded_area_proof_l83_83841


namespace largest_interior_angle_l83_83724

theorem largest_interior_angle (x : ℝ) (h₀ : 50 + 55 + x = 180) : 
  max 50 (max 55 x) = 75 := by
  sorry

end largest_interior_angle_l83_83724


namespace units_digit_F500_is_7_l83_83098

def F (n : ℕ) : ℕ := 2 ^ (2 ^ (2 * n)) + 1

theorem units_digit_F500_is_7 : (F 500) % 10 = 7 := 
  sorry

end units_digit_F500_is_7_l83_83098


namespace sonia_and_joss_time_spent_moving_l83_83507

def total_time_spent_moving (fill_time_per_trip drive_time_per_trip trips : ℕ) :=
  (fill_time_per_trip + drive_time_per_trip) * trips

def total_time_in_hours (total_time_in_minutes : ℕ) : ℚ :=
  total_time_in_minutes / 60

theorem sonia_and_joss_time_spent_moving :
  total_time_in_hours (total_time_spent_moving 15 30 6) = 4.5 :=
by
  sorry

end sonia_and_joss_time_spent_moving_l83_83507


namespace number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l83_83337

def cars_sold_each_day_first_three_days : ℕ := 5
def days_first_period : ℕ := 3
def quota : ℕ := 50
def cars_remaining_after_next_four_days : ℕ := 23
def days_next_period : ℕ := 4

theorem number_of_cars_sold_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period) - cars_remaining_after_next_four_days = 12 :=
by
  sorry

theorem cars_sold_each_day_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period - cars_remaining_after_next_four_days) / days_next_period = 3 :=
by
  sorry

end number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l83_83337


namespace range_of_a_l83_83092

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ - 3 / 5 < a ∧ a ≤ 1 := sorry

end range_of_a_l83_83092


namespace possible_area_l83_83768

theorem possible_area (A : ℝ) (B : ℝ) (L : ℝ × ℝ) (H₁ : L.1 = 13) (H₂ : L.2 = 14) (area_needed : ℝ) (H₃ : area_needed = 200) : 
∃ x y : ℝ, x = 13 ∧ y = 16 ∧ x * y ≥ area_needed :=
by
  sorry

end possible_area_l83_83768


namespace find_number_l83_83443

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 15) : x = 7.5 :=
sorry

end find_number_l83_83443


namespace ratio_of_speeds_l83_83402

theorem ratio_of_speeds (L V : ℝ) (R : ℝ) (h1 : L > 0) (h2 : V > 0) (h3 : R ≠ 0)
  (h4 : (1.48 * L) / (R * V) = (1.40 * L) / V) : R = 37 / 35 :=
by
  -- Proof would be inserted here
  sorry

end ratio_of_speeds_l83_83402


namespace smallest_white_marbles_l83_83566

/-
Let n be the total number of Peter's marbles.
Half of the marbles are orange.
One fifth of the marbles are purple.
Peter has 8 silver marbles.
-/
def total_marbles (n : ℕ) : ℕ :=
  n

def orange_marbles (n : ℕ) : ℕ :=
  n / 2

def purple_marbles (n : ℕ) : ℕ :=
  n / 5

def silver_marbles : ℕ :=
  8

def white_marbles (n : ℕ) : ℕ :=
  n - (orange_marbles n + purple_marbles n + silver_marbles)

-- Prove that the smallest number of white marbles Peter could have is 1.
theorem smallest_white_marbles : ∃ n : ℕ, n % 10 = 0 ∧ white_marbles n = 1 :=
sorry

end smallest_white_marbles_l83_83566


namespace horner_multiplications_additions_l83_83784

def f (x : ℝ) : ℝ := 6 * x^6 + 5

def x : ℝ := 2

theorem horner_multiplications_additions :
  (6 : ℕ) = 6 ∧ (6 : ℕ) = 6 := 
by 
  sorry

end horner_multiplications_additions_l83_83784


namespace sector_area_l83_83088

theorem sector_area (θ r arc_length : ℝ) (h_arc_length : arc_length = r * θ) (h_values : θ = 2 ∧ arc_length = 2) :
  1 / 2 * r^2 * θ = 1 := by
  sorry

end sector_area_l83_83088


namespace tommy_saw_100_wheels_l83_83073

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l83_83073


namespace collective_land_area_l83_83521

theorem collective_land_area 
  (C W : ℕ) 
  (h1 : 42 * C + 35 * W = 165200)
  (h2 : W = 3400)
  : C + W = 4500 :=
sorry

end collective_land_area_l83_83521


namespace y_value_l83_83927

theorem y_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_eq1 : (1 / x) + (1 / y) = 3 / 2) (h_eq2 : x * y = 9) : y = 6 :=
sorry

end y_value_l83_83927


namespace number_of_students_l83_83410

theorem number_of_students (n : ℕ) (h1 : n < 60) (h2 : n % 6 = 4) (h3 : n % 8 = 5) : n = 46 := by
  sorry

end number_of_students_l83_83410


namespace combined_exceeds_limit_l83_83721

-- Let Zone A, Zone B, and Zone C be zones on a road.
-- Let pA be the percentage of motorists exceeding the speed limit in Zone A.
-- Let pB be the percentage of motorists exceeding the speed limit in Zone B.
-- Let pC be the percentage of motorists exceeding the speed limit in Zone C.
-- Each zone has an equal amount of motorists.

def pA : ℝ := 15
def pB : ℝ := 20
def pC : ℝ := 10

/-
Prove that the combined percentage of motorists who exceed the speed limit
across all three zones is 15%.
-/
theorem combined_exceeds_limit :
  (pA + pB + pC) / 3 = 15 := 
by sorry

end combined_exceeds_limit_l83_83721


namespace min_value_of_quadratic_l83_83880

noncomputable def quadratic_min_value (x : ℕ) : ℝ :=
  3 * (x : ℝ)^2 - 12 * x + 800

theorem min_value_of_quadratic : (∀ x : ℕ, quadratic_min_value x ≥ 788) ∧ (quadratic_min_value 2 = 788) :=
by
  sorry

end min_value_of_quadratic_l83_83880


namespace number_of_4_letter_words_with_B_l83_83872

-- Define the set of letters.
inductive Alphabet
| A | B | C | D | E

-- The number of 4-letter words with repetition allowed and must include 'B' at least once.
noncomputable def words_with_at_least_one_B : ℕ :=
  let total := 5 ^ 4 -- Total number of 4-letter words.
  let without_B := 4 ^ 4 -- Total number of 4-letter words without 'B'.
  total - without_B

-- The main theorem statement.
theorem number_of_4_letter_words_with_B : words_with_at_least_one_B = 369 :=
  by sorry

end number_of_4_letter_words_with_B_l83_83872


namespace age_of_youngest_l83_83083

theorem age_of_youngest
  (y : ℕ)
  (h1 : 4 * 25 = y + (y + 2) + (y + 7) + (y + 11)) : y = 20 :=
by
  sorry

end age_of_youngest_l83_83083


namespace quadratic_solution1_quadratic_solution2_l83_83890

theorem quadratic_solution1 (x : ℝ) :
  (x^2 + 4 * x - 4 = 0) ↔ (x = -2 + 2 * Real.sqrt 2 ∨ x = -2 - 2 * Real.sqrt 2) :=
by sorry

theorem quadratic_solution2 (x : ℝ) :
  ((x - 1)^2 = 2 * (x - 1)) ↔ (x = 1 ∨ x = 3) :=
by sorry

end quadratic_solution1_quadratic_solution2_l83_83890


namespace locus_of_point_R_l83_83218

theorem locus_of_point_R :
  ∀ (P Q O F R : ℝ × ℝ)
    (hP_on_parabola : ∃ x1 y1, P = (x1, y1) ∧ y1^2 = 2 * x1)
    (h_directrix : Q.1 = -1 / 2)
    (hQ : ∃ x1 y1, Q = (x1, y1) ∧ P = (x1, y1))
    (hO : O = (0, 0))
    (hF : F = (1 / 2, 0))
    (h_intersection : ∃ x y, 
      R = (x, y) ∧
      ∃ x1 y1,
      P = (x1, y1) ∧ 
      y1^2 = 2 * x1 ∧
      ∃ (m_OP : ℝ), 
        m_OP = y1 / x1 ∧ 
        y = m_OP * x ∧
      ∃ (m_FQ : ℝ), 
        m_FQ = -y1 ∧
        y = m_FQ * x + y1 * (1 + 3 / 2)),
  R.2^2 = -2 * R.1^2 + R.1 :=
by sorry

end locus_of_point_R_l83_83218


namespace determine_x_l83_83575

variable (a b c d x : ℝ)
variable (h1 : (a^2 + x)/(b^2 + x) = c/d)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : d ≠ c) -- added condition from solution step

theorem determine_x : x = (a^2 * d - b^2 * c) / (c - d) := by
  sorry

end determine_x_l83_83575


namespace election_winner_votes_l83_83990

theorem election_winner_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 360) :
  0.62 * V = 930 :=
by {
  sorry
}

end election_winner_votes_l83_83990


namespace floor_tiling_l83_83210

theorem floor_tiling (n : ℕ) (x : ℕ) (h1 : 6 * x = n^2) : 6 ∣ n := sorry

end floor_tiling_l83_83210


namespace fixed_monthly_fee_l83_83550

def FebruaryBill (x y : ℝ) : Prop := x + y = 18.72
def MarchBill (x y : ℝ) : Prop := x + 3 * y = 28.08

theorem fixed_monthly_fee (x y : ℝ) (h1 : FebruaryBill x y) (h2 : MarchBill x y) : x = 14.04 :=
by 
  sorry

end fixed_monthly_fee_l83_83550


namespace vectors_orthogonal_x_value_l83_83380

theorem vectors_orthogonal_x_value :
  (∀ x : ℝ, (3 * x + 4 * (-7) = 0) → (x = 28 / 3)) := 
by 
  sorry

end vectors_orthogonal_x_value_l83_83380


namespace solve_quadratic_l83_83998

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l83_83998


namespace no_preimage_iff_lt_one_l83_83105

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem no_preimage_iff_lt_one (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k < 1 := 
by
  sorry

end no_preimage_iff_lt_one_l83_83105


namespace abc_inequality_l83_83138

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3 / 4 :=
by
  sorry

end abc_inequality_l83_83138


namespace total_population_of_cities_l83_83126

theorem total_population_of_cities (n : ℕ) (avg_pop : ℕ) (pn : (n = 20)) (avg_factor: (avg_pop = (4500 + 5000) / 2)) : 
  n * avg_pop = 95000 := 
by 
  sorry

end total_population_of_cities_l83_83126


namespace arithmetic_sequence_problem_l83_83571

theorem arithmetic_sequence_problem (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 6 = 36)
  (h2 : S n = 324)
  (h3 : S (n - 6) = 144) :
  n = 18 := by
  sorry

end arithmetic_sequence_problem_l83_83571


namespace ratio_of_ages_in_two_years_l83_83135

theorem ratio_of_ages_in_two_years (S M : ℕ) 
  (h1 : M = S + 37) 
  (h2 : S = 35) : 
  (M + 2) / (S + 2) = 2 := 
by 
  -- We skip the proof steps as instructed
  sorry

end ratio_of_ages_in_two_years_l83_83135


namespace modulus_of_product_l83_83477

namespace ComplexModule

open Complex

-- Definition of the complex numbers z1 and z2
def z1 : ℂ := 1 + I
def z2 : ℂ := 2 - I

-- Definition of their product z1z2
def z1z2 : ℂ := z1 * z2

-- Statement we need to prove (the modulus of z1z2 is √10)
theorem modulus_of_product : abs z1z2 = Real.sqrt 10 := by
  sorry

end ComplexModule

end modulus_of_product_l83_83477


namespace min_games_to_predict_l83_83548

theorem min_games_to_predict (W B : ℕ) (total_games : ℕ) (n : ℕ) : 
  W = 15 → B = 20 → total_games = W * B → n = 280 → 
  (∃ x, x ∈ {i | ∃ j, i < W ∧ j < B}) :=
by
  intros hW hB htotal hn
  sorry

end min_games_to_predict_l83_83548


namespace suitcase_lock_settings_l83_83744

-- Define the number of settings for each dial choice considering the conditions
noncomputable def first_digit_choices : ℕ := 9
noncomputable def second_digit_choices : ℕ := 9
noncomputable def third_digit_choices : ℕ := 8
noncomputable def fourth_digit_choices : ℕ := 7

-- Theorem to prove the total number of different settings
theorem suitcase_lock_settings : first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices = 4536 :=
by sorry

end suitcase_lock_settings_l83_83744


namespace toothpicks_150th_stage_l83_83731

-- Define the arithmetic sequence parameters
def first_term : ℕ := 4
def common_difference : ℕ := 4

-- Define the term number we are interested in
def stage_number : ℕ := 150

-- The total number of toothpicks in the nth stage of an arithmetic sequence
def num_toothpicks (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

-- Theorem stating the number of toothpicks in the 150th stage
theorem toothpicks_150th_stage : num_toothpicks first_term common_difference stage_number = 600 :=
by
  sorry

end toothpicks_150th_stage_l83_83731


namespace number_of_dissimilar_terms_l83_83722

theorem number_of_dissimilar_terms :
  let n := 7;
  let k := 4;
  let number_of_terms := Nat.choose (n + k - 1) (k - 1);
  number_of_terms = 120 :=
by
  sorry

end number_of_dissimilar_terms_l83_83722


namespace find_range_of_a_l83_83201

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem find_range_of_a : {a : ℝ | set_B a ⊆ set_A} = {a : ℝ | a < -1} ∪ {1} :=
by
  sorry

end find_range_of_a_l83_83201


namespace factorize_cubic_l83_83704

theorem factorize_cubic : ∀ x : ℝ, x^3 - 4 * x = x * (x + 2) * (x - 2) :=
by
  sorry

end factorize_cubic_l83_83704


namespace solve_for_t_l83_83110

variable (A P0 r t : ℝ)

theorem solve_for_t (h : A = P0 * Real.exp (r * t)) : t = (Real.log (A / P0)) / r :=
  by
  sorry

end solve_for_t_l83_83110


namespace number_line_distance_l83_83296

theorem number_line_distance (x : ℝ) : (abs (-3 - x) = 2) ↔ (x = -5 ∨ x = -1) :=
by
  sorry

end number_line_distance_l83_83296


namespace condition_sufficient_but_not_necessary_l83_83718

theorem condition_sufficient_but_not_necessary (a : ℝ) : (a > 9 → (1 / a < 1 / 9)) ∧ ¬(1 / a < 1 / 9 → a > 9) :=
by 
  sorry

end condition_sufficient_but_not_necessary_l83_83718


namespace tina_earned_more_l83_83810

def candy_bar_problem_statement : Prop :=
  let type_a_price := 2
  let type_b_price := 3
  let marvin_type_a_sold := 20
  let marvin_type_b_sold := 15
  let tina_type_a_sold := 70
  let tina_type_b_sold := 35
  let marvin_discount_per_5_type_a := 1
  let tina_discount_per_10_type_b := 2
  let tina_returns_type_b := 2
  let marvin_total_earnings := 
    (marvin_type_a_sold * type_a_price) + 
    (marvin_type_b_sold * type_b_price) -
    (marvin_type_a_sold / 5 * marvin_discount_per_5_type_a)
  let tina_total_earnings := 
    (tina_type_a_sold * type_a_price) + 
    (tina_type_b_sold * type_b_price) -
    (tina_type_b_sold / 10 * tina_discount_per_10_type_b) -
    (tina_returns_type_b * type_b_price)
  let difference := tina_total_earnings - marvin_total_earnings
  difference = 152

theorem tina_earned_more :
  candy_bar_problem_statement :=
by
  sorry

end tina_earned_more_l83_83810


namespace richmond_tigers_revenue_l83_83508

theorem richmond_tigers_revenue
  (total_tickets : ℕ)
  (first_half_tickets : ℕ)
  (catA_first_half : ℕ)
  (catB_first_half : ℕ)
  (catC_first_half : ℕ)
  (priceA : ℕ)
  (priceB : ℕ)
  (priceC : ℕ)
  (catA_second_half : ℕ)
  (catB_second_half : ℕ)
  (catC_second_half : ℕ)
  (total_revenue_second_half : ℕ)
  (h_total_tickets : total_tickets = 9570)
  (h_first_half_tickets : first_half_tickets = 3867)
  (h_catA_first_half : catA_first_half = 1350)
  (h_catB_first_half : catB_first_half = 1150)
  (h_catC_first_half : catC_first_half = 1367)
  (h_priceA : priceA = 50)
  (h_priceB : priceB = 40)
  (h_priceC : priceC = 30)
  (h_catA_second_half : catA_second_half = 1350)
  (h_catB_second_half : catB_second_half = 1150)
  (h_catC_second_half : catC_second_half = 1367)
  (h_total_revenue_second_half : total_revenue_second_half = 154510)
  :
  catA_second_half * priceA + catB_second_half * priceB + catC_second_half * priceC = total_revenue_second_half :=
by
  sorry

end richmond_tigers_revenue_l83_83508


namespace valid_propositions_l83_83719

theorem valid_propositions :
  (∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧ (∃ n : ℝ, ∀ m : ℝ, m * n = m) :=
by
  sorry

end valid_propositions_l83_83719


namespace min_value_a_l83_83274

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1 / 3 :=
by
  sorry

end min_value_a_l83_83274


namespace tetrahedron_in_cube_l83_83760

theorem tetrahedron_in_cube (a x : ℝ) (h : a = 6) :
  (∃ x, x = 6 * Real.sqrt 2) :=
sorry

end tetrahedron_in_cube_l83_83760


namespace not_always_true_inequality_l83_83353

variable {x y z : ℝ} {k : ℤ}

theorem not_always_true_inequality :
  x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → ¬ ( ∀ z, (x / (z^k) > y / (z^k)) ) :=
by
  intro hx hy hxy hz hk
  sorry

end not_always_true_inequality_l83_83353


namespace find_smallest_a_l83_83449
open Real

noncomputable def a_min := 2 / 9

theorem find_smallest_a (a b c : ℝ)
  (h1 : (1/4, -9/8) = (1/4, a * (1/4) * (1/4) - 9/8))
  (h2 : ∃ n : ℤ, a + b + c = n)
  (h3 : a > 0)
  (h4 : b = - a / 2)
  (h5 : c = a / 16 - 9 / 8): 
  a = a_min :=
by {
  -- Lean code equivalent to the provided mathematical proof will be placed here.
  sorry
}

end find_smallest_a_l83_83449


namespace stock_market_value_l83_83803

def face_value : ℝ := 100
def dividend_rate : ℝ := 0.05
def yield_rate : ℝ := 0.10

theorem stock_market_value :
  (dividend_rate * face_value / yield_rate = 50) :=
by
  sorry

end stock_market_value_l83_83803


namespace number_of_houses_in_block_l83_83235

theorem number_of_houses_in_block (pieces_per_house pieces_per_block : ℕ) (h1 : pieces_per_house = 32) (h2 : pieces_per_block = 640) :
  pieces_per_block / pieces_per_house = 20 :=
by
  sorry

end number_of_houses_in_block_l83_83235


namespace equal_sharing_l83_83554

theorem equal_sharing (total_cards friends : ℕ) (h1 : total_cards = 455) (h2 : friends = 5) : total_cards / friends = 91 := by
  sorry

end equal_sharing_l83_83554


namespace fraction_add_eq_l83_83362

theorem fraction_add_eq (x y : ℝ) (hx : y / x = 3 / 7) : (x + y) / x = 10 / 7 :=
by
  sorry

end fraction_add_eq_l83_83362


namespace part_1_part_2_part_3_l83_83039

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (2 * Real.exp x) / (Real.exp x + 1) + k

theorem part_1 (k : ℝ) :
  (∀ x, f x k = -f (-x) k) → k = -1 :=
sorry

theorem part_2 (m : ℝ) :
  (∀ x > 0, (2 * Real.exp x - 1) / (Real.exp x + 1) ≤ m * (Real.exp x - 1) / (Real.exp x + 1)) → 2 ≤ m :=
sorry

noncomputable def g (x : ℝ) : ℝ := (f x (-1) + 1) / (1 - f x (-1))

theorem part_3 (n : ℝ) :
  (∀ a b c : ℝ, 0 < a ∧ a ≤ n → 0 < b ∧ b ≤ n → 0 < c ∧ c ≤ n → (a + b > c ∧ b + c > a ∧ c + a > b) →
   (g a + g b > g c ∧ g b + g c > g a ∧ g c + g a > g b)) → n = 2 * Real.log 2 :=
sorry

end part_1_part_2_part_3_l83_83039


namespace solve_firm_problem_l83_83531

def firm_problem : Prop :=
  ∃ (P A : ℕ), 
    (P / A = 2 / 63) ∧ 
    (P / (A + 50) = 1 / 34) ∧ 
    (P = 20)

theorem solve_firm_problem : firm_problem :=
  sorry

end solve_firm_problem_l83_83531


namespace fewest_coach_handshakes_l83_83536

noncomputable def binom (n : ℕ) := n * (n - 1) / 2

theorem fewest_coach_handshakes : 
  ∃ (k1 k2 k3 : ℕ), binom 43 + k1 + k2 + k3 = 903 ∧ k1 + k2 + k3 = 0 := 
by
  use 0, 0, 0
  sorry

end fewest_coach_handshakes_l83_83536


namespace sandwiches_provided_now_l83_83442

-- Define the initial number of sandwich kinds
def initialSandwichKinds : ℕ := 23

-- Define the number of sold out sandwich kinds
def soldOutSandwichKinds : ℕ := 14

-- Define the proof that the actual number of sandwich kinds provided now
theorem sandwiches_provided_now : initialSandwichKinds - soldOutSandwichKinds = 9 :=
by
  -- The proof goes here
  sorry

end sandwiches_provided_now_l83_83442


namespace math_problem_l83_83141

theorem math_problem : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := 
by
  sorry

end math_problem_l83_83141


namespace divisors_of_2700_l83_83453

def prime_factors_2700 : ℕ := 2^2 * 3^3 * 5^2

def number_of_positive_divisors (n : ℕ) (a b c : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1)

theorem divisors_of_2700 : number_of_positive_divisors 2700 2 3 2 = 36 := by
  sorry

end divisors_of_2700_l83_83453


namespace total_first_year_students_400_l83_83065

theorem total_first_year_students_400 (N : ℕ) (A B C : ℕ) 
  (h1 : A = 80) 
  (h2 : B = 100) 
  (h3 : C = 20) 
  (h4 : A * B = C * N) : 
  N = 400 :=
sorry

end total_first_year_students_400_l83_83065


namespace average_glasses_is_15_l83_83038

variable (S L : ℕ)

-- Conditions:
def box1 := 12 -- One box contains 12 glasses
def box2 := 16 -- Another box contains 16 glasses
def total_glasses := 480 -- Total number of glasses
def diff_L_S := 16 -- There are 16 more larger boxes

-- Equations derived from conditions:
def eq1 : Prop := (12 * S + 16 * L = total_glasses)
def eq2 : Prop := (L = S + diff_L_S)

-- We need to prove that the average number of glasses per box is 15:
def avg_glasses_per_box := total_glasses / (S + L)

-- The statement we need to prove:
theorem average_glasses_is_15 :
  (12 * S + 16 * L = total_glasses) ∧ (L = S + diff_L_S) → avg_glasses_per_box = 15 :=
by
  sorry

end average_glasses_is_15_l83_83038


namespace mustard_bottles_total_l83_83266

theorem mustard_bottles_total (b1 b2 b3 : ℝ) (h1 : b1 = 0.25) (h2 : b2 = 0.25) (h3 : b3 = 0.38) :
  b1 + b2 + b3 = 0.88 :=
by
  sorry

end mustard_bottles_total_l83_83266


namespace tom_is_15_years_younger_l83_83613

/-- 
Alice is now 30 years old.
Ten years ago, Alice was 4 times as old as Tom was then.
Prove that Tom is 15 years younger than Alice.
-/
theorem tom_is_15_years_younger (A T : ℕ) (h1 : A = 30) (h2 : A - 10 = 4 * (T - 10)) : A - T = 15 :=
by
  sorry

end tom_is_15_years_younger_l83_83613


namespace range_of_a_l83_83307

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), ((a^2 - 1) * x^2 + (a + 1) * x + 1) > 0) → 1 ≤ a ∧ a ≤ 5 / 3 := 
by
  sorry

end range_of_a_l83_83307


namespace minimal_range_of_sample_l83_83202

theorem minimal_range_of_sample (x1 x2 x3 x4 x5 : ℝ) 
  (mean_condition : (x1 + x2 + x3 + x4 + x5) / 5 = 6) 
  (median_condition : x3 = 10) 
  (sample_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) : 
  (x5 - x1) = 10 :=
sorry

end minimal_range_of_sample_l83_83202


namespace percentage_B_D_l83_83454

variables (A B C D : ℝ)

-- Conditions as hypotheses
theorem percentage_B_D
  (h1 : B = 1.71 * A)
  (h2 : C = 1.80 * A)
  (h3 : D = 1.90 * B)
  (h4 : B = 1.62 * C)
  (h5 : A = 0.65 * D)
  (h6 : C = 0.55 * D) : 
  B = 1.1115 * D :=
sorry

end percentage_B_D_l83_83454


namespace algebraic_expression_value_l83_83162

theorem algebraic_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x/(x + y))^2005 + (y/(x + y))^2005 = -1 :=
by
  sorry

end algebraic_expression_value_l83_83162


namespace problem1_problem2_l83_83265

-- Define that a quadratic is a root-multiplying equation if one root is twice the other
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 ≠ 0 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)

-- Problem 1: Prove that x^2 - 3x + 2 = 0 is a root-multiplying equation
theorem problem1 : is_root_multiplying 1 (-3) 2 :=
  sorry

-- Problem 2: Given ax^2 + bx - 6 = 0 is a root-multiplying equation with one root being 2, determine a and b
theorem problem2 (a b : ℝ) : is_root_multiplying a b (-6) → (∃ x1 x2 : ℝ, x1 = 2 ∧ x1 ≠ 0 ∧ a * x1^2 + b * x1 - 6 = 0 ∧ a * x2^2 + b * x2 - 6 = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)) →
( (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
  sorry

end problem1_problem2_l83_83265


namespace prove_union_sets_l83_83208

universe u

variable {α : Type u}
variable {M N : Set ℕ}
variable (a b : ℕ)

theorem prove_union_sets (h1 : M = {3, 4^a}) (h2 : N = {a, b}) (h3 : M ∩ N = {1}) : M ∪ N = {0, 1, 3} := sorry

end prove_union_sets_l83_83208


namespace center_of_circle_l83_83370

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (3, 8)) (h2 : (x2, y2) = (11, -4)) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 2) := by
  sorry

end center_of_circle_l83_83370


namespace geom_sequence_sum_l83_83560

theorem geom_sequence_sum (n : ℕ) (a : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 4 ^ n + a) : 
  a = -1 := 
by
  sorry

end geom_sequence_sum_l83_83560


namespace determinant_expression_l83_83585

theorem determinant_expression (a b c p q : ℝ) 
  (h_root : ∀ x : ℝ, (x = a ∨ x = b ∨ x = c) → (Polynomial.eval x (Polynomial.X ^ 3 - 3 * Polynomial.C p * Polynomial.X + 2 * Polynomial.C q) = 0)) :
  Matrix.det ![![2 + a, 1, 1], ![1, 2 + b, 1], ![1, 1, 2 + c]] = -3 * p - 2 * q + 4 :=
by {
  sorry
}

end determinant_expression_l83_83585


namespace impossible_even_n_m_if_n3_plus_m3_is_odd_l83_83984

theorem impossible_even_n_m_if_n3_plus_m3_is_odd
  (n m : ℤ) (h : (n^3 + m^3) % 2 = 1) : ¬((n % 2 = 0) ∧ (m % 2 = 0)) := by
  sorry

end impossible_even_n_m_if_n3_plus_m3_is_odd_l83_83984


namespace mn_equals_neg16_l83_83919

theorem mn_equals_neg16 (m n : ℤ) (h1 : m = -2) (h2 : |n| = 8) (h3 : m + n > 0) : m * n = -16 := by
  sorry

end mn_equals_neg16_l83_83919


namespace find_b_if_continuous_l83_83185

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

theorem find_b_if_continuous (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 23 / 2 :=
by
  sorry

end find_b_if_continuous_l83_83185


namespace profit_percentage_l83_83639

theorem profit_percentage (SP : ℝ) (h1 : SP > 0) (h2 : CP = 0.99 * SP) : (SP - CP) / CP * 100 = 1.01 :=
by
  sorry

end profit_percentage_l83_83639


namespace tile_size_l83_83459

theorem tile_size (length width : ℕ) (total_tiles : ℕ) 
  (h_length : length = 48) 
  (h_width : width = 72) 
  (h_total_tiles : total_tiles = 96) : 
  ((length * width) / total_tiles) = 36 := 
by
  sorry

end tile_size_l83_83459


namespace percentage_of_girls_taking_lunch_l83_83187

theorem percentage_of_girls_taking_lunch 
  (total_students : ℕ)
  (boys_ratio girls_ratio : ℕ)
  (boys_to_girls_ratio : boys_ratio + girls_ratio = 10)
  (boys : ℕ)
  (girls : ℕ)
  (boys_calc : boys = (boys_ratio * total_students) / 10)
  (girls_calc : girls = (girls_ratio * total_students) / 10)
  (boys_lunch_percentage : ℕ)
  (boys_lunch : ℕ)
  (boys_lunch_calc : boys_lunch = (boys_lunch_percentage * boys) / 100)
  (total_lunch_percentage : ℕ)
  (total_lunch : ℕ)
  (total_lunch_calc : total_lunch = (total_lunch_percentage * total_students) / 100)
  (girls_lunch : ℕ)
  (girls_lunch_calc : girls_lunch = total_lunch - boys_lunch) :
  ((girls_lunch * 100) / girls) = 40 :=
by 
  -- The proof can be filled in here
  sorry

end percentage_of_girls_taking_lunch_l83_83187


namespace total_worth_is_correct_l83_83910

-- Define the conditions
def rows : ℕ := 4
def gold_bars_per_row : ℕ := 20
def worth_per_gold_bar : ℕ := 20000

-- Define the calculated values
def total_gold_bars : ℕ := rows * gold_bars_per_row
def total_worth_of_gold_bars : ℕ := total_gold_bars * worth_per_gold_bar

-- Theorem statement to prove the correct total worth
theorem total_worth_is_correct : total_worth_of_gold_bars = 1600000 := by
  sorry

end total_worth_is_correct_l83_83910


namespace mask_usage_duration_l83_83542

-- Define given conditions
def TotalMasks : ℕ := 75
def FamilyMembers : ℕ := 7
def MaskChangeInterval : ℕ := 2

-- Define the goal statement, which is to prove that the family will take 21 days to use all masks
theorem mask_usage_duration 
  (M : ℕ := 75)  -- total masks
  (N : ℕ := 7)   -- family members
  (d : ℕ := 2)   -- mask change interval
  : (M / N) * d + 1 = 21 :=
sorry

end mask_usage_duration_l83_83542


namespace find_supplementary_angle_l83_83737

def A := 45
def supplementary_angle (A S : ℕ) := A + S = 180
def complementary_angle (A C : ℕ) := A + C = 90
def thrice_complementary (S C : ℕ) := S = 3 * C

theorem find_supplementary_angle : 
  ∀ (A S C : ℕ), 
    A = 45 → 
    supplementary_angle A S →
    complementary_angle A C →
    thrice_complementary S C → 
    S = 135 :=
by
  intros A S C hA hSupp hComp hThrice
  have h1 : A = 45 := by assumption
  have h2 : A + S = 180 := by assumption
  have h3 : A + C = 90 := by assumption
  have h4 : S = 3 * C := by assumption
  sorry

end find_supplementary_angle_l83_83737


namespace range_of_a_l83_83344

-- Definitions related to the conditions in the problem
def polynomial (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x ^ 5 - 4 * a * x ^ 3 + 2 * b ^ 2 * x ^ 2 + 1

def v_2 (x : ℝ) (a : ℝ) : ℝ := (3 * x + 0) * x - 4 * a

def v_3 (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (((3 * x + 0) * x - 4 * a) * x + 2 * b ^ 2)

-- The main statement to prove
theorem range_of_a (x a b : ℝ) (h1 : x = 2) (h2 : ∀ b : ℝ, (v_2 x a) < (v_3 x a b)) : a < 3 :=
by
  sorry

end range_of_a_l83_83344


namespace mark_candy_bars_consumption_l83_83387

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end mark_candy_bars_consumption_l83_83387


namespace number_of_scenarios_l83_83195

theorem number_of_scenarios :
  ∃ (count : ℕ), count = 42244 ∧
  (∃ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
    x1 % 7 = 0 ∧ x2 % 7 = 0 ∧ x3 % 7 = 0 ∧ x4 % 7 = 0 ∧
    x5 % 13 = 0 ∧ x6 % 13 = 0 ∧ x7 % 13 = 0 ∧
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) :=
sorry

end number_of_scenarios_l83_83195


namespace quadratic_m_condition_l83_83958

theorem quadratic_m_condition (m : ℝ) (h_eq : (m - 2) * x ^ (m ^ 2 - 2) - m * x + 1 = 0) (h_pow : m ^ 2 - 2 = 2) :
  m = -2 :=
by sorry

end quadratic_m_condition_l83_83958


namespace acme_horseshoes_production_l83_83928

theorem acme_horseshoes_production
  (profit : ℝ)
  (initial_outlay : ℝ)
  (cost_per_set : ℝ)
  (selling_price : ℝ)
  (number_of_sets : ℕ) :
  profit = selling_price * number_of_sets - (initial_outlay + cost_per_set * number_of_sets) →
  profit = 15337.5 →
  initial_outlay = 12450 →
  cost_per_set = 20.75 →
  selling_price = 50 →
  number_of_sets = 950 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end acme_horseshoes_production_l83_83928


namespace ball_hits_ground_at_t_l83_83217

noncomputable def ball_height (t : ℝ) : ℝ := -6 * t^2 - 10 * t + 56

theorem ball_hits_ground_at_t :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 7 / 3 := by
  sorry

end ball_hits_ground_at_t_l83_83217


namespace square_area_from_diagonal_l83_83342

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal_l83_83342


namespace other_religion_students_l83_83191

theorem other_religion_students (total_students : ℕ) 
  (muslims_percent hindus_percent sikhs_percent christians_percent buddhists_percent : ℝ) 
  (h1 : total_students = 1200) 
  (h2 : muslims_percent = 0.35) 
  (h3 : hindus_percent = 0.25) 
  (h4 : sikhs_percent = 0.15) 
  (h5 : christians_percent = 0.10) 
  (h6 : buddhists_percent = 0.05) : 
  ∃ other_religion_students : ℕ, other_religion_students = 120 :=
by
  sorry

end other_religion_students_l83_83191


namespace tiffany_uploaded_7_pics_from_her_phone_l83_83782

theorem tiffany_uploaded_7_pics_from_her_phone
  (camera_pics : ℕ)
  (albums : ℕ)
  (pics_per_album : ℕ)
  (total_pics : ℕ)
  (h_camera_pics : camera_pics = 13)
  (h_albums : albums = 5)
  (h_pics_per_album : pics_per_album = 4)
  (h_total_pics : total_pics = albums * pics_per_album) :
  total_pics - camera_pics = 7 := by
  sorry

end tiffany_uploaded_7_pics_from_her_phone_l83_83782


namespace candy_distribution_l83_83248

-- Define the required parameters and conditions.
def num_distinct_candies : ℕ := 9
def num_bags : ℕ := 3

-- The result that we need to prove
theorem candy_distribution :
  (3 ^ num_distinct_candies) - 3 * (2 ^ (num_distinct_candies - 1) - 2) = 18921 := by
  sorry

end candy_distribution_l83_83248


namespace weight_of_original_piece_of_marble_l83_83259

theorem weight_of_original_piece_of_marble (W : ℝ) 
  (h1 : W > 0)
  (h2 : (0.75 * 0.56 * W) = 105) : 
  W = 250 :=
by
  sorry

end weight_of_original_piece_of_marble_l83_83259


namespace twentieth_triangular_number_l83_83570

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentieth_triangular_number : triangular_number 20 = 210 :=
by
  sorry

end twentieth_triangular_number_l83_83570


namespace sum_faces_of_pentahedron_l83_83128

def pentahedron := {f : ℕ // f = 5}

theorem sum_faces_of_pentahedron (p : pentahedron) : p.val = 5 := 
by
  sorry

end sum_faces_of_pentahedron_l83_83128


namespace probability_red_next_ball_l83_83777

-- Definitions of initial conditions
def initial_red_balls : ℕ := 50
def initial_blue_balls : ℕ := 50
def initial_yellow_balls : ℕ := 30
def total_pulled_balls : ℕ := 65

-- Condition that Calvin pulled out 5 more red balls than blue balls
def red_balls_pulled (blue_balls_pulled : ℕ) : ℕ := blue_balls_pulled + 5

-- Compute the remaining balls
def remaining_balls (blue_balls_pulled : ℕ) : Prop :=
  let remaining_red_balls := initial_red_balls - red_balls_pulled blue_balls_pulled
  let remaining_blue_balls := initial_blue_balls - blue_balls_pulled
  let remaining_yellow_balls := initial_yellow_balls - (total_pulled_balls - red_balls_pulled blue_balls_pulled - blue_balls_pulled)
  (remaining_red_balls + remaining_blue_balls + remaining_yellow_balls) = 15

-- Main theorem to be proven
theorem probability_red_next_ball (blue_balls_pulled : ℕ) (h : remaining_balls blue_balls_pulled) :
  (initial_red_balls - red_balls_pulled blue_balls_pulled) / 15 = 9 / 26 :=
sorry

end probability_red_next_ball_l83_83777


namespace students_in_fifth_and_sixth_classes_l83_83612

theorem students_in_fifth_and_sixth_classes :
  let c1 := 20
  let c2 := 25
  let c3 := 25
  let c4 := c1 / 2
  let total_students := 136
  let total_first_four_classes := c1 + c2 + c3 + c4
  let c5_and_c6 := total_students - total_first_four_classes
  c5_and_c6 = 56 :=
by
  sorry

end students_in_fifth_and_sixth_classes_l83_83612


namespace problem_statement_l83_83581

def scientific_notation (n: ℝ) (mantissa: ℝ) (exponent: ℤ) : Prop :=
  n = mantissa * 10 ^ exponent

theorem problem_statement : scientific_notation 320000 3.2 5 :=
by {
  sorry
}

end problem_statement_l83_83581


namespace determine_1000g_weight_l83_83319

-- Define the weights
def weights : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- Define the weight sets
def Group1 : List ℕ := [weights.get! 0, weights.get! 1]
def Group2 : List ℕ := [weights.get! 2, weights.get! 3]
def Group3 : List ℕ := [weights.get! 4]

-- Definition to choose the lighter group or determine equality
def lighterGroup (g1 g2 : List ℕ) : List ℕ :=
  if g1.sum = g2.sum then Group3 else if g1.sum < g2.sum then g1 else g2

-- Determine the 1000 g weight functionally
def identify1000gWeightUsing3Weighings : ℕ :=
  let firstWeighing := lighterGroup Group1 Group2
  if firstWeighing = Group3 then Group3.get! 0 else
  let remainingWeights := firstWeighing
  if remainingWeights.get! 0 = remainingWeights.get! 1 then Group3.get! 0
  else if remainingWeights.get! 0 < remainingWeights.get! 1 then remainingWeights.get! 0 else remainingWeights.get! 1

theorem determine_1000g_weight : identify1000gWeightUsing3Weighings = 1000 :=
sorry

end determine_1000g_weight_l83_83319


namespace Yella_last_week_usage_l83_83498

/-- 
Yella's computer usage last week was some hours. If she plans to use the computer 8 hours a day for this week, 
her computer usage for this week is 35 hours less. Given these conditions, prove that Yella's computer usage 
last week was 91 hours.
-/
theorem Yella_last_week_usage (daily_usage : ℕ) (days_in_week : ℕ) (difference : ℕ)
  (h1: daily_usage = 8)
  (h2: days_in_week = 7)
  (h3: difference = 35) :
  daily_usage * days_in_week + difference = 91 := 
by
  sorry

end Yella_last_week_usage_l83_83498


namespace perpendicular_vectors_l83_83667

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n : ℝ × ℝ := (-3, 2)

-- Define the conditions to be checked
def km_plus_n (k : ℝ) : ℝ × ℝ := (k * m.1 + n.1, k * m.2 + n.2)
def m_minus_3n : ℝ × ℝ := (m.1 - 3 * n.1, m.2 - 3 * n.2)

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for k = 19, the two vectors are perpendicular
theorem perpendicular_vectors (k : ℝ) (h : k = 19) : dot_product (km_plus_n k) (m_minus_3n) = 0 := by
  rw [h]
  simp [km_plus_n, m_minus_3n, dot_product]
  sorry

end perpendicular_vectors_l83_83667


namespace remainder_of_3_pow_19_mod_10_l83_83922

theorem remainder_of_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_mod_10_l83_83922


namespace roger_current_money_l83_83501

noncomputable def roger_initial_money : ℕ := 16
noncomputable def roger_birthday_money : ℕ := 28
noncomputable def roger_game_spending : ℕ := 25

theorem roger_current_money : roger_initial_money + roger_birthday_money - roger_game_spending = 19 := by
  sorry

end roger_current_money_l83_83501


namespace unique_solution_eq_condition_l83_83999

theorem unique_solution_eq_condition (p q : ℝ) :
  (∃! x : ℝ, (2 * x - 2 * p + q) / (2 * x - 2 * p - q) = (2 * q + p + x) / (2 * q - p - x)) ↔ (p = 3 * q / 4 ∧ q ≠ 0) :=
  sorry

end unique_solution_eq_condition_l83_83999


namespace polynomial_exists_l83_83653

open Polynomial

noncomputable def exists_polynomial_2013 : Prop :=
  ∃ (f : Polynomial ℤ), (∀ (n : ℕ), n ≤ f.natDegree → (coeff f n = 1 ∨ coeff f n = -1))
                         ∧ ((X - 1) ^ 2013 ∣ f)

theorem polynomial_exists : exists_polynomial_2013 :=
  sorry

end polynomial_exists_l83_83653


namespace hotel_towels_l83_83561

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l83_83561


namespace range_of_a_l83_83522

variable (a : ℝ)
def p : Prop := a > 1/4
def q : Prop := a ≤ -1 ∨ a ≥ 1

theorem range_of_a :
  ((p a ∧ ¬ (q a)) ∨ (q a ∧ ¬ (p a))) ↔ (a > 1/4 ∧ a < 1) ∨ (a ≤ -1) :=
by
  sorry

end range_of_a_l83_83522


namespace sequence_general_formula_l83_83519

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n - 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^(n - 1) + 2 :=
sorry

end sequence_general_formula_l83_83519


namespace value_of_expression_l83_83438

theorem value_of_expression (a b : ℝ) (h : a - b = 1) : a^2 - b^2 - 2 * b = 1 := 
by
  sorry

end value_of_expression_l83_83438


namespace AM_GM_Inequality_l83_83045

theorem AM_GM_Inequality 
  (a b c : ℝ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end AM_GM_Inequality_l83_83045


namespace cone_prism_volume_ratio_l83_83643

/--
Given:
- The base of the prism is a rectangle with side lengths 2r and 3r.
- The height of the prism is h.
- The base of the cone is a circle with radius r and height h.

Prove:
- The ratio of the volume of the cone to the volume of the prism is (π / 18).
-/
theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (1 / 3 * Real.pi * r^2 * h) / (6 * r^2 * h) = Real.pi / 18 := by
  sorry

end cone_prism_volume_ratio_l83_83643


namespace probability_all_co_captains_l83_83524

-- Define the number of students in each team
def students_team1 : ℕ := 4
def students_team2 : ℕ := 6
def students_team3 : ℕ := 7
def students_team4 : ℕ := 9

-- Define the probability of selecting each team
def prob_selecting_team : ℚ := 1 / 4

-- Define the probability of selecting three co-captains from each team
def prob_team1 : ℚ := 1 / Nat.choose students_team1 3
def prob_team2 : ℚ := 1 / Nat.choose students_team2 3
def prob_team3 : ℚ := 1 / Nat.choose students_team3 3
def prob_team4 : ℚ := 1 / Nat.choose students_team4 3

-- Define the total probability
def total_prob : ℚ :=
  prob_selecting_team * (prob_team1 + prob_team2 + prob_team3 + prob_team4)

theorem probability_all_co_captains :
  total_prob = 59 / 1680 := by
  sorry

end probability_all_co_captains_l83_83524


namespace tangent_segments_area_l83_83518

theorem tangent_segments_area (r : ℝ) (l : ℝ) (area : ℝ) :
  r = 4 ∧ l = 6 → area = 9 * Real.pi :=
by
  sorry

end tangent_segments_area_l83_83518


namespace boat_navigation_under_arch_l83_83338

theorem boat_navigation_under_arch (h_arch : ℝ) (w_arch: ℝ) (boat_width: ℝ) (boat_height: ℝ) (boat_above_water: ℝ) :
  (h_arch = 5) → 
  (w_arch = 8) → 
  (boat_width = 4) → 
  (boat_height = 2) → 
  (boat_above_water = 0.75) →
  (h_arch - 2 = 3) :=
by
  intros h_arch_eq w_arch_eq boat_w_eq boat_h_eq boat_above_water_eq
  sorry

end boat_navigation_under_arch_l83_83338


namespace atomic_number_l83_83151

theorem atomic_number (mass_number : ℕ) (neutrons : ℕ) (protons : ℕ) :
  mass_number = 288 →
  neutrons = 169 →
  (protons = mass_number - neutrons) →
  protons = 119 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end atomic_number_l83_83151


namespace obtuse_triangle_side_range_l83_83888

theorem obtuse_triangle_side_range {a : ℝ} (h1 : a > 3) (h2 : (a - 3)^2 < 36) : 3 < a ∧ a < 9 := 
by
  sorry

end obtuse_triangle_side_range_l83_83888


namespace crayons_lost_or_given_away_l83_83049

theorem crayons_lost_or_given_away (P E L : ℕ) (h1 : P = 479) (h2 : E = 134) (h3 : L = P - E) : L = 345 :=
by
  rw [h1, h2] at h3
  exact h3

end crayons_lost_or_given_away_l83_83049


namespace probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l83_83220

def probability_of_excellence_A : ℚ := 2/5
def probability_of_excellence_B1 : ℚ := 1/4
def probability_of_excellence_B2 : ℚ := 2/5
def probability_of_excellence_B3 (n : ℚ) : ℚ := n

def one_excellence_A : ℚ := 3 * (2/5) * (3/5)^2
def one_excellence_B (n : ℚ) : ℚ := 
    (probability_of_excellence_B1 * (3/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (2/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (3/5) * n)

theorem probability_one_excellence_A : one_excellence_A = 54/125 := sorry

theorem probability_one_excellence_B (n : ℚ) (hn : n = 1/3) : one_excellence_B n = 9/20 := sorry

def expected_excellence_A : ℚ := 3 * (2/5)
def expected_excellence_B (n : ℚ) : ℚ := (13/20) + n

theorem range_n_for_A (n : ℚ) (hn1 : 0 < n) (hn2 : n < 11/20): 
    expected_excellence_A > expected_excellence_B n := sorry

end probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l83_83220


namespace latest_time_temp_decreasing_l83_83481

theorem latest_time_temp_decreasing (t : ℝ) 
  (h1 : -t^2 + 12 * t + 55 = 82) 
  (h2 : ∀ t0 : ℝ, -2 * t0 + 12 < 0 → t > t0) : 
  t = 6 + (3 * Real.sqrt 28 / 2) :=
sorry

end latest_time_temp_decreasing_l83_83481


namespace fraction_received_A_correct_l83_83580

def fraction_of_students_received_A := 0.7
def fraction_of_students_received_B := 0.2
def fraction_of_students_received_A_or_B := 0.9

theorem fraction_received_A_correct :
  fraction_of_students_received_A_or_B - fraction_of_students_received_B = fraction_of_students_received_A :=
by
  sorry

end fraction_received_A_correct_l83_83580


namespace part1_part2_l83_83769

theorem part1 (x y : ℝ) (h1 : y = x + 30) (h2 : 2 * x + 3 * y = 340) : x = 50 ∧ y = 80 :=
by {
  -- Later, we can place the steps to prove x = 50 and y = 80 here.
  sorry
}

theorem part2 (m : ℕ) (h3 : 0 ≤ m ∧ m ≤ 50)
               (h4 : 54 * (50 - m) + 72 * m = 3060) : m = 20 :=
by {
  -- Later, we can place the steps to prove m = 20 here.
  sorry
}

end part1_part2_l83_83769


namespace find_square_sum_of_xy_l83_83378

theorem find_square_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y + x + y = 83) (h2 : x^2 * y + x * y^2 = 1056) : x^2 + y^2 = 458 :=
sorry

end find_square_sum_of_xy_l83_83378


namespace files_remaining_correct_l83_83289

-- Definitions for the original number of files
def music_files_original : ℕ := 4
def video_files_original : ℕ := 21
def document_files_original : ℕ := 12
def photo_files_original : ℕ := 30
def app_files_original : ℕ := 7

-- Definitions for the number of deleted files
def video_files_deleted : ℕ := 15
def document_files_deleted : ℕ := 10
def photo_files_deleted : ℕ := 18
def app_files_deleted : ℕ := 3

-- Definitions for the remaining number of files
def music_files_remaining : ℕ := music_files_original
def video_files_remaining : ℕ := video_files_original - video_files_deleted
def document_files_remaining : ℕ := document_files_original - document_files_deleted
def photo_files_remaining : ℕ := photo_files_original - photo_files_deleted
def app_files_remaining : ℕ := app_files_original - app_files_deleted

-- The proof problem statement
theorem files_remaining_correct : 
  music_files_remaining + video_files_remaining + document_files_remaining + photo_files_remaining + app_files_remaining = 28 :=
by
  rw [music_files_remaining, video_files_remaining, document_files_remaining, photo_files_remaining, app_files_remaining]
  exact rfl


end files_remaining_correct_l83_83289


namespace triangle_arithmetic_geometric_equilateral_l83_83773

theorem triangle_arithmetic_geometric_equilateral :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ (∃ d, β = α + d ∧ γ = α + 2 * d) ∧ (∃ r, β = α * r ∧ γ = α * r^2) →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry

end triangle_arithmetic_geometric_equilateral_l83_83773


namespace find_m_n_pairs_l83_83147

theorem find_m_n_pairs (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∀ᶠ a in Filter.atTop, (a^m + a - 1) % (a^n + a^2 - 1) = 0) → m = n + 2 :=
by
  sorry

end find_m_n_pairs_l83_83147


namespace phi_varphi_difference_squared_l83_83516

theorem phi_varphi_difference_squared :
  ∀ (Φ φ : ℝ), (Φ ≠ φ) → (Φ^2 - 2*Φ - 1 = 0) → (φ^2 - 2*φ - 1 = 0) →
  (Φ - φ)^2 = 8 :=
by
  intros Φ φ distinct hΦ hφ
  sorry

end phi_varphi_difference_squared_l83_83516


namespace age_difference_problem_l83_83234

theorem age_difference_problem 
    (minimum_age : ℕ := 25)
    (current_age_Jane : ℕ := 28)
    (years_ahead : ℕ := 6)
    (Dara_age_in_6_years : ℕ := (current_age_Jane + years_ahead) / 2):
    minimum_age - (Dara_age_in_6_years - years_ahead) = 14 :=
by
  -- all definition parts: minimum_age, current_age_Jane, years_ahead,
  -- Dara_age_in_6_years are present
  sorry

end age_difference_problem_l83_83234


namespace triangle_type_l83_83539

-- Let's define what it means for a triangle to be acute, obtuse, and right in terms of angle
def is_acute_triangle (a b c : ℝ) : Prop := (a < 90) ∧ (b < 90) ∧ (c < 90)
def is_obtuse_triangle (a b c : ℝ) : Prop := (a > 90) ∨ (b > 90) ∨ (c > 90)
def is_right_triangle (a b c : ℝ) : Prop := (a = 90) ∨ (b = 90) ∨ (c = 90)

-- The problem statement
theorem triangle_type (A B C : ℝ) (h : A = 100) : is_obtuse_triangle A B C :=
by {
  -- Sorry is used to indicate a placeholder for the proof
  sorry
}

end triangle_type_l83_83539


namespace paco_initial_salty_cookies_l83_83293

variable (S : ℕ)
variable (sweet_cookies : ℕ := 40)
variable (salty_cookies_eaten1 : ℕ := 28)
variable (sweet_cookies_eaten : ℕ := 15)
variable (extra_salty_cookies_eaten : ℕ := 13)

theorem paco_initial_salty_cookies 
  (h1 : salty_cookies_eaten1 = 28)
  (h2 : sweet_cookies_eaten = 15)
  (h3 : extra_salty_cookies_eaten = 13)
  (h4 : sweet_cookies = 40)
  : (S = (salty_cookies_eaten1 + (extra_salty_cookies_eaten + sweet_cookies_eaten))) :=
by
  -- starting with the equation S = number of salty cookies Paco
  -- initially had, which should be equal to the total salty 
  -- cookies he ate.
  sorry

end paco_initial_salty_cookies_l83_83293


namespace product_of_rational_solutions_eq_twelve_l83_83534

theorem product_of_rational_solutions_eq_twelve :
  ∃ c1 c2 : ℕ, (c1 > 0) ∧ (c2 > 0) ∧ 
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c1 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c1 = d^2) ∧
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c2 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c2 = d^2) ∧
               c1 * c2 = 12 := sorry

end product_of_rational_solutions_eq_twelve_l83_83534


namespace phone_purchase_initial_max_profit_additional_purchase_l83_83227

-- Definitions for phone purchase prices and selling prices
def purchase_price_A : ℕ := 3000
def selling_price_A : ℕ := 3400
def purchase_price_B : ℕ := 3500
def selling_price_B : ℕ := 4000

-- Definitions for total expenditure and profit
def total_spent : ℕ := 32000
def total_profit : ℕ := 4400

-- Definitions for initial number of units purchased
def initial_units_A : ℕ := 6
def initial_units_B : ℕ := 4

-- Definitions for the additional purchase constraints and profit calculation
def max_additional_units : ℕ := 30
def additional_units_A : ℕ := 10
def additional_units_B : ℕ := max_additional_units - additional_units_A 
def max_profit : ℕ := 14000

theorem phone_purchase_initial:
  3000 * initial_units_A + 3500 * initial_units_B = total_spent ∧
  (selling_price_A - purchase_price_A) * initial_units_A + (selling_price_B - purchase_price_B) * initial_units_B = total_profit := by
  sorry 

theorem max_profit_additional_purchase:
  additional_units_A + additional_units_B = max_additional_units ∧
  additional_units_B ≤ 2 * additional_units_A ∧
  (selling_price_A - purchase_price_A) * additional_units_A + (selling_price_B - purchase_price_B) * additional_units_B = max_profit := by
  sorry

end phone_purchase_initial_max_profit_additional_purchase_l83_83227


namespace total_cost_mulch_l83_83004

-- Define the conditions
def tons_to_pounds (tons : ℕ) : ℕ := tons * 2000

def price_per_pound : ℝ := 2.5

-- Define the statement to prove
theorem total_cost_mulch (mulch_in_tons : ℕ) (h₁ : mulch_in_tons = 3) : 
  tons_to_pounds mulch_in_tons * price_per_pound = 15000 :=
by
  -- The proof would normally go here.
  sorry

end total_cost_mulch_l83_83004


namespace intersection_sum_x_coordinates_mod_17_l83_83452

theorem intersection_sum_x_coordinates_mod_17 :
  ∃ x : ℤ, (∃ y₁ y₂ : ℤ, (y₁ ≡ 7 * x + 3 [ZMOD 17]) ∧ (y₂ ≡ 13 * x + 4 [ZMOD 17]))
       ∧ x ≡ 14 [ZMOD 17]  :=
by
  sorry

end intersection_sum_x_coordinates_mod_17_l83_83452


namespace box_length_is_10_l83_83916

theorem box_length_is_10
  (width height vol_cube num_cubes : ℕ)
  (h₀ : width = 13)
  (h₁ : height = 5)
  (h₂ : vol_cube = 5)
  (h₃ : num_cubes = 130) :
  (num_cubes * vol_cube) / (width * height) = 10 :=
by
  -- Proof steps will be filled here.
  sorry

end box_length_is_10_l83_83916


namespace molecular_weight_H2O_correct_l83_83109

-- Define the atomic weights of hydrogen and oxygen, and the molecular weight of H2O
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight calculation of H2O
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O

-- Theorem to state the molecular weight of H2O is approximately 18.016 g/mol
theorem molecular_weight_H2O_correct : molecular_weight_H2O = 18.016 :=
by
  -- Putting the value and calculation
  sorry

end molecular_weight_H2O_correct_l83_83109


namespace probe_distance_before_refuel_l83_83180

def total_distance : ℕ := 5555555555555
def distance_from_refuel : ℕ := 3333333333333
def distance_before_refuel : ℕ := 2222222222222

theorem probe_distance_before_refuel :
  total_distance - distance_from_refuel = distance_before_refuel := by
  sorry

end probe_distance_before_refuel_l83_83180


namespace perpendicular_lines_b_l83_83642

theorem perpendicular_lines_b (b : ℝ) : 
  (∃ (k m: ℝ), k = 3 ∧ 2 * m + b * k = 14 ∧ (k * m = -1)) ↔ b = 2 / 3 :=
sorry

end perpendicular_lines_b_l83_83642


namespace find_a_and_c_range_of_m_l83_83830

theorem find_a_and_c (a c : ℝ) 
  (h : ∀ x, 1 < x ∧ x < 3 ↔ ax^2 + x + c > 0) 
  : a = -1/4 ∧ c = -3/4 := 
sorry

theorem range_of_m (m : ℝ) 
  (h : ∀ x, (-1/4)*x^2 + 2*x - 3 > 0 → x + m > 0) 
  : m ≥ -2 :=
sorry

end find_a_and_c_range_of_m_l83_83830


namespace gcd_g_y_l83_83676

noncomputable def g (y : ℕ) : ℕ := (3 * y + 5) * (6 * y + 7) * (10 * y + 3) * (5 * y + 11) * (y + 7)

theorem gcd_g_y (y : ℕ) (h : ∃ k : ℕ, y = 18090 * k) : Nat.gcd (g y) y = 8085 := 
sorry

end gcd_g_y_l83_83676


namespace missing_number_is_correct_l83_83937

theorem missing_number_is_correct (mean : ℝ) (observed_numbers : List ℝ) (total_obs : ℕ) (x : ℝ) :
  mean = 14.2 →
  observed_numbers = [8, 13, 21, 7, 23] →
  total_obs = 6 →
  (mean * total_obs = x + observed_numbers.sum) →
  x = 13.2 :=
by
  intros h_mean h_obs h_total h_sum
  sorry

end missing_number_is_correct_l83_83937


namespace work_alone_days_l83_83018

theorem work_alone_days (d : ℝ) (p q : ℝ) (h1 : q = 10) (h2 : 2 * (1/d + 1/q) = 0.3) : d = 20 :=
by
  sorry

end work_alone_days_l83_83018


namespace die_top_face_after_path_l83_83868

def opposite_face (n : ℕ) : ℕ :=
  7 - n

def roll_die (start : ℕ) (sequence : List String) : ℕ :=
  sequence.foldl
    (λ top movement =>
      match movement with
      | "left" => opposite_face (7 - top) -- simplified assumption for movements
      | "forward" => opposite_face (top - 1)
      | "right" => opposite_face (7 - top + 1)
      | "back" => opposite_face (top + 1)
      | _ => top) start

theorem die_top_face_after_path : roll_die 3 ["left", "forward", "right", "back", "forward", "back"] = 4 :=
  by
  sorry

end die_top_face_after_path_l83_83868


namespace Tonya_initial_stamps_l83_83313

theorem Tonya_initial_stamps :
  ∀ (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (jimmy_matchbooks : ℕ) (tonya_remaining_stamps : ℕ),
  stamps_per_match = 12 →
  matches_per_matchbook = 24 →
  jimmy_matchbooks = 5 →
  tonya_remaining_stamps = 3 →
  tonya_remaining_stamps + (jimmy_matchbooks * matches_per_matchbook) / stamps_per_match = 13 := 
by
  intros stamps_per_match matches_per_matchbook jimmy_matchbooks tonya_remaining_stamps
  sorry

end Tonya_initial_stamps_l83_83313


namespace sum_of_squares_arithmetic_geometric_l83_83711

theorem sum_of_squares_arithmetic_geometric (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 225) : x^2 + y^2 = 1150 :=
by
  sorry

end sum_of_squares_arithmetic_geometric_l83_83711


namespace original_perimeter_of_rectangle_l83_83707

theorem original_perimeter_of_rectangle
  (a b : ℝ)
  (h : (a + 3) * (b + 3) - a * b = 90) :
  2 * (a + b) = 54 :=
sorry

end original_perimeter_of_rectangle_l83_83707


namespace solution_set_of_xf_gt_0_l83_83007

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_ineq : ∀ x : ℝ, x > 0 → f x < x * (deriv f x)
axiom f_at_one : f 1 = 0

theorem solution_set_of_xf_gt_0 : {x : ℝ | x * f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_xf_gt_0_l83_83007


namespace morgan_hula_hooping_time_l83_83263

-- Definitions based on conditions
def nancy_can_hula_hoop : ℕ := 10
def casey_can_hula_hoop : ℕ := nancy_can_hula_hoop - 3
def morgan_can_hula_hoop : ℕ := 3 * casey_can_hula_hoop

-- Theorem statement to show the solution is correct
theorem morgan_hula_hooping_time : morgan_can_hula_hoop = 21 :=
by
  sorry

end morgan_hula_hooping_time_l83_83263


namespace chocolates_bought_at_cost_price_l83_83578

variables (C S : ℝ) (n : ℕ)

-- Given conditions
def cost_eq_selling_50 := n * C = 50 * S
def gain_percent := (S - C) / C = 0.30

-- Question to prove
theorem chocolates_bought_at_cost_price (h1 : cost_eq_selling_50 C S n) (h2 : gain_percent C S) : n = 65 :=
sorry

end chocolates_bought_at_cost_price_l83_83578


namespace karen_drive_l83_83682

theorem karen_drive (a b c x : ℕ) (h1 : a ≥ 1) (h2 : a + b + c ≤ 9) (h3 : 33 * (c - a) = 25 * x) :
  a^2 + b^2 + c^2 = 75 :=
sorry

end karen_drive_l83_83682


namespace d_value_l83_83899

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l83_83899


namespace ratio_second_to_third_l83_83455

-- Define the three numbers A, B, C, and their conditions.
variables (A B C : ℕ)

-- Conditions derived from the problem statement.
def sum_condition : Prop := A + B + C = 98
def ratio_condition : Prop := 3 * A = 2 * B
def second_number_value : Prop := B = 30

-- The main theorem stating the problem to prove.
theorem ratio_second_to_third (h1 : sum_condition A B C) (h2 : ratio_condition A B) (h3 : second_number_value B) :
  B = 30 ∧ A = 20 ∧ C = 48 → B / C = 5 / 8 :=
by
  sorry

end ratio_second_to_third_l83_83455


namespace point_in_fourth_quadrant_l83_83405

variable (a : ℝ)

theorem point_in_fourth_quadrant (h : a < -1) : 
    let x := a^2 - 2*a - 1
    let y := (a + 1) / abs (a + 1)
    (x > 0) ∧ (y < 0) := 
by
  let x := a^2 - 2*a - 1
  let y := (a + 1) / abs (a + 1)
  sorry

end point_in_fourth_quadrant_l83_83405


namespace exists_m_in_range_l83_83252

theorem exists_m_in_range :
  ∃ m : ℝ, 0 ≤ m ∧ m < 1 ∧ ∀ x : ℕ, (x > m ∧ x < 2) ↔ (x = 1) :=
by
  sorry

end exists_m_in_range_l83_83252


namespace fewer_cans_collected_today_than_yesterday_l83_83963

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l83_83963


namespace students_not_taking_french_or_spanish_l83_83664

theorem students_not_taking_french_or_spanish 
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (both_languages_students : ℕ) 
  (h_total_students : total_students = 28)
  (h_french_students : french_students = 5)
  (h_spanish_students : spanish_students = 10)
  (h_both_languages_students : both_languages_students = 4) :
  total_students - (french_students + spanish_students - both_languages_students) = 17 := 
by {
  -- Correct answer can be verified with the given conditions
  -- The proof itself is omitted (as instructed)
  sorry
}

end students_not_taking_french_or_spanish_l83_83664


namespace roller_coaster_costs_7_tickets_l83_83433

-- Define the number of tickets for the Ferris wheel, log ride, and the initial and additional tickets Zach needs.
def ferris_wheel_tickets : ℕ := 2
def log_ride_tickets : ℕ := 1
def initial_tickets : ℕ := 1
def additional_tickets : ℕ := 9

-- Define the total number of tickets Zach needs.
def total_tickets : ℕ := initial_tickets + additional_tickets

-- Define the number of tickets needed for the Ferris wheel and log ride together.
def combined_tickets_needed : ℕ := ferris_wheel_tickets + log_ride_tickets

-- Define the number of tickets the roller coaster costs.
def roller_coaster_tickets : ℕ := total_tickets - combined_tickets_needed

-- The theorem stating what we need to prove.
theorem roller_coaster_costs_7_tickets :
  roller_coaster_tickets = 7 :=
by sorry

end roller_coaster_costs_7_tickets_l83_83433


namespace production_analysis_l83_83331

def daily_change (day: ℕ) : ℤ :=
  match day with
  | 0 => 40    -- Monday
  | 1 => -30   -- Tuesday
  | 2 => 90    -- Wednesday
  | 3 => -50   -- Thursday
  | 4 => -20   -- Friday
  | 5 => -10   -- Saturday
  | 6 => 20    -- Sunday
  | _ => 0     -- Invalid day, just in case

def planned_daily_production : ℤ := 500

def actual_production (day: ℕ) : ℤ :=
  planned_daily_production + (List.sum (List.map daily_change (List.range (day + 1))))

def total_production : ℤ :=
  List.sum (List.map actual_production (List.range 7))

theorem production_analysis :
  ∃ largest_increase_day smallest_increase_day : ℕ,
    largest_increase_day = 2 ∧  -- Wednesday
    smallest_increase_day = 1 ∧  -- Tuesday
    total_production = 3790 ∧
    total_production > 7 * planned_daily_production := by
  sorry

end production_analysis_l83_83331


namespace caffeine_over_l83_83273

section caffeine_problem

-- Definitions of the given conditions
def cups_of_coffee : Nat := 3
def cans_of_soda : Nat := 1
def cups_of_tea : Nat := 2

def caffeine_per_cup_coffee : Nat := 80
def caffeine_per_can_soda : Nat := 40
def caffeine_per_cup_tea : Nat := 50

def caffeine_goal : Nat := 200

-- Calculate the total caffeine consumption
def caffeine_from_coffee : Nat := cups_of_coffee * caffeine_per_cup_coffee
def caffeine_from_soda : Nat := cans_of_soda * caffeine_per_can_soda
def caffeine_from_tea : Nat := cups_of_tea * caffeine_per_cup_tea

def total_caffeine : Nat := caffeine_from_coffee + caffeine_from_soda + caffeine_from_tea

-- Calculate the caffeine amount over the goal
def caffeine_over_goal : Nat := total_caffeine - caffeine_goal

-- Theorem statement
theorem caffeine_over {total_caffeine caffeine_goal : Nat} (h : total_caffeine = 380) (g : caffeine_goal = 200) :
  caffeine_over_goal = 180 := by
  -- The proof goes here.
  sorry

end caffeine_problem

end caffeine_over_l83_83273


namespace area_of_trapezoid_EFGH_l83_83002

-- Define the vertices of the trapezoid
structure Point where
  x : ℤ
  y : ℤ

def E : Point := ⟨-2, -3⟩
def F : Point := ⟨-2, 2⟩
def G : Point := ⟨4, 5⟩
def H : Point := ⟨4, 0⟩

-- Define the formula for the area of a trapezoid
def trapezoid_area (b1 b2 height : ℤ) : ℤ :=
  (b1 + b2) * height / 2

-- The proof statement
theorem area_of_trapezoid_EFGH : trapezoid_area (F.y - E.y) (G.y - H.y) (G.x - E.x) = 30 := by
  sorry -- proof not required

end area_of_trapezoid_EFGH_l83_83002


namespace rabbit_turtle_travel_distance_l83_83866

-- Define the initial conditions and their values
def rabbit_velocity : ℕ := 40 -- meters per minute when jumping
def rabbit_jump_time : ℕ := 3 -- minutes of jumping
def rabbit_rest_time : ℕ := 2 -- minutes of resting
def rabbit_start_time : ℕ := 9 * 60 -- 9:00 AM in minutes from midnight

def turtle_velocity : ℕ := 10 -- meters per minute
def turtle_start_time : ℕ := 6 * 60 + 40 -- 6:40 AM in minutes from midnight
def lead_time : ℕ := 15 -- turtle leads the rabbit by 15 seconds at the end

-- Define the final distance the turtle traveled by the time rabbit arrives
def distance_traveled_by_turtle (total_time : ℕ) : ℕ :=
  total_time * turtle_velocity

-- Define time intervals for periodic calculations (in minutes)
def time_interval : ℕ := 5

-- Define the total distance rabbit covers in one periodic interval
def rabbit_distance_in_interval : ℕ :=
  rabbit_velocity * rabbit_jump_time

-- Calculate total time taken by the rabbit to close the gap before starting actual run
def initial_time_to_close_gap (gap : ℕ) : ℕ := 
  gap * time_interval / rabbit_distance_in_interval

-- Define the total time the rabbit travels
def total_travel_time : ℕ :=
  initial_time_to_close_gap ((rabbit_start_time - turtle_start_time) * turtle_velocity) + 97

-- Define the total distance condition to be proved as 2370 meters
theorem rabbit_turtle_travel_distance :
  distance_traveled_by_turtle (total_travel_time + lead_time) = 2370 :=
  by sorry

end rabbit_turtle_travel_distance_l83_83866


namespace p1a_p1b_l83_83206

theorem p1a (m : ℕ) (hm : m > 1) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3 := by
  sorry  -- Proof is omitted

theorem p1b : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 ∧ x = 4 ∧ y = 63 := by
  sorry  -- Proof is omitted

end p1a_p1b_l83_83206


namespace angle_bisector_ratio_l83_83569

theorem angle_bisector_ratio (A B C Q : Type) (AC CB AQ QB : ℝ) (k : ℝ) 
  (hAC : AC = 4 * k) (hCB : CB = 5 * k) (angle_bisector_theorem : AQ / QB = AC / CB) :
  AQ / QB = 4 / 5 := 
by sorry

end angle_bisector_ratio_l83_83569


namespace suyeong_ran_distance_l83_83434

theorem suyeong_ran_distance 
  (circumference : ℝ) 
  (laps : ℕ) 
  (h_circumference : circumference = 242.7)
  (h_laps : laps = 5) : 
  (circumference * laps = 1213.5) := 
  by sorry

end suyeong_ran_distance_l83_83434


namespace no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l83_83163

theorem no_perfect_squares_in_ap (n x : ℤ) : ¬(3 * n + 2 = x^2) :=
sorry

theorem infinitely_many_perfect_cubes_in_ap : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^3 :=
sorry

theorem no_terms_of_form_x_pow_2m (n x : ℤ) (m : ℕ) : 3 * n + 2 ≠ x^(2 * m) :=
sorry

theorem infinitely_many_terms_of_form_x_pow_2m_plus_1 (m : ℕ) : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^(2 * m + 1) :=
sorry

end no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l83_83163


namespace s_point_condition_l83_83829

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def g_prime (a : ℝ) (x : ℝ) : ℝ := 1 / x

theorem s_point_condition (a : ℝ) (x₀ : ℝ) (h_f_g : f a x₀ = g a x₀) (h_f'g' : f_prime a x₀ = g_prime a x₀) :
  a = 2 / Real.exp 1 :=
by
  sorry

end s_point_condition_l83_83829


namespace ratio_of_candy_bar_to_caramel_l83_83772

noncomputable def price_of_caramel : ℝ := 3
noncomputable def price_of_candy_bar (k : ℝ) : ℝ := k * price_of_caramel
noncomputable def price_of_cotton_candy (C : ℝ) : ℝ := 2 * C 

theorem ratio_of_candy_bar_to_caramel (k : ℝ) (C CC : ℝ) :
  C = price_of_candy_bar k →
  CC = price_of_cotton_candy C →
  6 * C + 3 * price_of_caramel + CC = 57 →
  C / price_of_caramel = 2 :=
by
  sorry

end ratio_of_candy_bar_to_caramel_l83_83772


namespace joe_two_kinds_of_fruit_l83_83242

-- Definitions based on the conditions
def meals := ["breakfast", "lunch", "snack", "dinner"] -- 4 meals
def fruits := ["apple", "orange", "banana"] -- 3 kinds of fruits

-- Probability that Joe consumes the same fruit for all meals
noncomputable def prob_same_fruit := (1 / 3) ^ 4

-- Probability that Joe eats at least two different kinds of fruits
noncomputable def prob_at_least_two_kinds := 1 - 3 * prob_same_fruit

theorem joe_two_kinds_of_fruit :
  prob_at_least_two_kinds = 26 / 27 :=
by
  -- Proof omitted for this theorem
  sorry

end joe_two_kinds_of_fruit_l83_83242


namespace quadratic_has_real_root_l83_83791

theorem quadratic_has_real_root (b : ℝ) : 
  (b^2 - 100 ≥ 0) ↔ (b ≤ -10 ∨ b ≥ 10) :=
by
  sorry

end quadratic_has_real_root_l83_83791


namespace workers_days_not_worked_l83_83284

theorem workers_days_not_worked (W N : ℕ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 :=
sorry

end workers_days_not_worked_l83_83284


namespace cricket_run_rate_l83_83762

theorem cricket_run_rate (initial_run_rate : ℝ) (initial_overs : ℕ) (target : ℕ) (remaining_overs : ℕ) 
    (run_rate_in_remaining_overs : ℝ)
    (h1 : initial_run_rate = 3.2)
    (h2 : initial_overs = 10)
    (h3 : target = 272)
    (h4 : remaining_overs = 40) :
    run_rate_in_remaining_overs = 6 :=
  sorry

end cricket_run_rate_l83_83762


namespace students_more_than_pets_l83_83467

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end students_more_than_pets_l83_83467


namespace cubic_inches_in_two_cubic_feet_l83_83863

theorem cubic_inches_in_two_cubic_feet :
  (12 ^ 3) * 2 = 3456 := by
  sorry

end cubic_inches_in_two_cubic_feet_l83_83863


namespace greatest_int_radius_lt_75pi_l83_83749

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l83_83749


namespace length_of_BD_is_six_l83_83713

-- Definitions of the conditions
def AB : ℕ := 6
def BC : ℕ := 11
def CD : ℕ := 6
def DA : ℕ := 8
def BD : ℕ := 6 -- adding correct answer into definition

-- The statement we want to prove
theorem length_of_BD_is_six (hAB : AB = 6) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 8) (hBD_int : BD = 6) : 
  BD = 6 :=
by
  -- Proof placeholder
  sorry

end length_of_BD_is_six_l83_83713


namespace snakes_in_pond_l83_83621

theorem snakes_in_pond (S : ℕ) (alligators : ℕ := 10) (total_eyes : ℕ := 56) (alligator_eyes : ℕ := 2) (snake_eyes : ℕ := 2) :
  (alligators * alligator_eyes) + (S * snake_eyes) = total_eyes → S = 18 :=
by
  intro h
  sorry

end snakes_in_pond_l83_83621


namespace division_remainder_false_l83_83014

theorem division_remainder_false :
  ¬(1700 / 500 = 17 / 5 ∧ (1700 % 500 = 3 ∧ 17 % 5 = 2)) := by
  sorry

end division_remainder_false_l83_83014


namespace arithmetic_sequence_problem_l83_83400

variable {a₁ d : ℝ} (S : ℕ → ℝ)

axiom Sum_of_terms (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_problem
  (h : S 10 = 4 * S 5) :
  (a₁ / d) = 1 / 2 :=
by
  -- definitional expansion and algebraic simplification would proceed here
  sorry

end arithmetic_sequence_problem_l83_83400


namespace intersect_sphere_circle_l83_83495

-- Define the given sphere equation
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the equation of a circle in the plane x = x0 parallel to the yz-plane
def circle_in_plane (x0 y0 z0 r : ℝ) (y z : ℝ) : Prop :=
  (y - y0)^2 + (z - z0)^2 = r^2

-- Define the intersecting circle from the sphere equation in the x = c plane
def intersecting_circle (h k l c R : ℝ) (y z : ℝ) : Prop :=
  (y - k)^2 + (z - l)^2 = R^2 - (h - c)^2

-- The main proof statement
theorem intersect_sphere_circle (h k l R c x0 y0 z0 r: ℝ) :
  ∀ y z, intersecting_circle h k l c R y z ↔ circle_in_plane x0 y0 z0 r y z :=
sorry

end intersect_sphere_circle_l83_83495


namespace difference_of_fractions_l83_83015

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h1 : a = 700) (h2 : b = 7) : a - b = 693 :=
by
  rw [h1, h2]
  norm_num

end difference_of_fractions_l83_83015


namespace new_person_weight_l83_83078

noncomputable def weight_of_new_person (W : ℝ) : ℝ :=
  W + 61 - 25

theorem new_person_weight {W : ℝ} : 
  ((W + 61 - 25) / 12 = W / 12 + 3) → 
  weight_of_new_person W = 61 :=
by
  intro h
  sorry

end new_person_weight_l83_83078


namespace first_sphere_weight_l83_83894

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * (r ^ 2)

noncomputable def weight (r1 r2 : ℝ) (W2 : ℝ) : ℝ :=
  let A1 := surface_area r1
  let A2 := surface_area r2
  (W2 * A1) / A2

theorem first_sphere_weight :
  let r1 := 0.15
  let r2 := 0.3
  let W2 := 32
  weight r1 r2 W2 = 8 := 
by
  sorry

end first_sphere_weight_l83_83894


namespace inequality_solution_l83_83297

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

noncomputable def lhs (x : ℝ) := 
  log_b 5 250 + ((4 - (log_b 5 2) ^ 2) / (2 + log_b 5 2))

noncomputable def rhs (x : ℝ) := 
  125 ^ (log_b 5 x) ^ 2 - 24 * x ^ (log_b 5 x)

theorem inequality_solution (x : ℝ) : 
  (lhs x <= rhs x) ↔ (0 < x ∧ x ≤ 1/5) ∨ (5 ≤ x) := 
sorry

end inequality_solution_l83_83297


namespace february_saving_l83_83876

-- Definitions for the conditions
variable {F D : ℝ}

-- Condition 1: Saving in January
def january_saving : ℝ := 2

-- Condition 2: Saving in March
def march_saving : ℝ := 8

-- Condition 3: Total savings after 6 months
def total_savings : ℝ := 126

-- Condition 4: Savings increase by a fixed amount D each month
def fixed_increase : ℝ := D

-- Condition 5: Difference between savings in March and January
def difference_jan_mar : ℝ := 8 - 2

-- The main theorem to prove: Robi saved 50 in February
theorem february_saving : F = 50 :=
by
  -- The required proof is omitted
  sorry

end february_saving_l83_83876


namespace certain_number_unique_l83_83345

theorem certain_number_unique (x : ℝ) (hx1 : 213 * x = 3408) (hx2 : 21.3 * x = 340.8) : x = 16 :=
by
  sorry

end certain_number_unique_l83_83345


namespace complex_solution_l83_83181

theorem complex_solution (x : ℂ) (h : x^2 + 1 = 0) : x = Complex.I ∨ x = -Complex.I :=
by sorry

end complex_solution_l83_83181


namespace andrew_total_donation_l83_83824

/-
Problem statement:
Andrew started donating 7k to an organization on his 11th birthday. Yesterday, Andrew turned 29.
Verify that the total amount Andrew has donated is 126k.
-/

theorem andrew_total_donation 
  (annual_donation : ℕ := 7000) 
  (start_age : ℕ := 11) 
  (current_age : ℕ := 29) 
  (years_donating : ℕ := current_age - start_age) 
  (total_donated : ℕ := annual_donation * years_donating) :
  total_donated = 126000 := 
by 
  sorry

end andrew_total_donation_l83_83824


namespace bottom_rightmost_rectangle_is_E_l83_83795

-- Definitions of the given conditions
structure Rectangle where
  w : ℕ
  y : ℕ

def A : Rectangle := { w := 5, y := 8 }
def B : Rectangle := { w := 2, y := 4 }
def C : Rectangle := { w := 4, y := 6 }
def D : Rectangle := { w := 8, y := 5 }
def E : Rectangle := { w := 10, y := 9 }

-- The theorem we need to prove
theorem bottom_rightmost_rectangle_is_E :
    (E.w = 10) ∧ (E.y = 9) :=
by
  -- Proof would go here
  sorry

end bottom_rightmost_rectangle_is_E_l83_83795


namespace cloves_of_garlic_needed_l83_83170

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l83_83170


namespace remaining_work_hours_l83_83388

theorem remaining_work_hours (initial_hours_per_week initial_weeks total_earnings first_weeks first_week_hours : ℝ) 
  (hourly_wage remaining_weeks remaining_earnings total_hours_required : ℝ) : 
  15 = initial_hours_per_week →
  15 = initial_weeks →
  4500 = total_earnings →
  3 = first_weeks →
  5 = first_week_hours →
  hourly_wage = total_earnings / (initial_hours_per_week * initial_weeks) →
  remaining_earnings = total_earnings - (first_week_hours * hourly_wage * first_weeks) →
  remaining_weeks = initial_weeks - first_weeks →
  total_hours_required = remaining_earnings / (hourly_wage * remaining_weeks) →
  total_hours_required = 17.5 :=
by
  intros
  sorry

end remaining_work_hours_l83_83388


namespace trajectory_of_circle_center_l83_83775

theorem trajectory_of_circle_center :
  ∀ (M : ℝ × ℝ), (∃ r : ℝ, (M.1 + r = 1 ∧ M.1 - r = -1) ∧ (M.1 - 1)^2 + (M.2 - 0)^2 = r^2) → M.2^2 = 4 * M.1 :=
by
  intros M h
  sorry

end trajectory_of_circle_center_l83_83775


namespace evaluate_expression_l83_83003

theorem evaluate_expression (m n : ℝ) (h : m - n = 2) :
  (2 * m^2 - 4 * m * n + 2 * n^2 - 1) = 7 := by
  sorry

end evaluate_expression_l83_83003


namespace passengers_in_7_buses_l83_83512

theorem passengers_in_7_buses (passengers_total buses_total_given buses_required : ℕ) 
    (h1 : passengers_total = 456) 
    (h2 : buses_total_given = 12) 
    (h3 : buses_required = 7) :
    (passengers_total / buses_total_given) * buses_required = 266 := 
sorry

end passengers_in_7_buses_l83_83512


namespace true_q_if_not_p_and_p_or_q_l83_83076

variables {p q : Prop}

theorem true_q_if_not_p_and_p_or_q (h1 : ¬p) (h2 : p ∨ q) : q :=
by 
  sorry

end true_q_if_not_p_and_p_or_q_l83_83076


namespace ordered_concrete_weight_l83_83846

def weight_of_materials : ℝ := 0.83
def weight_of_bricks : ℝ := 0.17
def weight_of_stone : ℝ := 0.5

theorem ordered_concrete_weight :
  weight_of_materials - (weight_of_bricks + weight_of_stone) = 0.16 := by
  sorry

end ordered_concrete_weight_l83_83846


namespace sum_of_coordinates_l83_83965

theorem sum_of_coordinates {g h : ℝ → ℝ} 
  (h₁ : g 4 = 5)
  (h₂ : ∀ x, h x = (g x)^2) :
  4 + h 4 = 29 := by
  sorry

end sum_of_coordinates_l83_83965


namespace value_of_polynomial_l83_83861

theorem value_of_polynomial (a b : ℝ) (h : a^2 - 2 * b - 1 = 0) : -2 * a^2 + 4 * b + 2025 = 2023 :=
by
  sorry

end value_of_polynomial_l83_83861


namespace tank_capacity_is_48_l83_83895

-- Define the conditions
def num_4_liter_bucket_used : ℕ := 12
def num_3_liter_bucket_used : ℕ := num_4_liter_bucket_used + 4

-- Define the capacities of the buckets and the tank
def bucket_4_liters_capacity : ℕ := 4 * num_4_liter_bucket_used
def bucket_3_liters_capacity : ℕ := 3 * num_3_liter_bucket_used

-- Tank capacity
def tank_capacity : ℕ := 48

-- Statement to prove
theorem tank_capacity_is_48 : 
    bucket_4_liters_capacity = tank_capacity ∧
    bucket_3_liters_capacity = tank_capacity := by
  sorry

end tank_capacity_is_48_l83_83895


namespace math_problem_l83_83406

theorem math_problem :
  ( (1 / 3 * 9) ^ 2 * (1 / 27 * 81) ^ 2 * (1 / 243 * 729) ^ 2) = 729 := by
  sorry

end math_problem_l83_83406


namespace initial_welders_count_l83_83951

theorem initial_welders_count (W : ℕ) (h1: (1 + 16 * (W - 9) / W = 8)) : W = 16 :=
by {
  sorry
}

end initial_welders_count_l83_83951


namespace shobha_current_age_l83_83799

theorem shobha_current_age (S B : ℕ) (h1 : S / B = 4 / 3) (h2 : S + 6 = 26) : B = 15 :=
by
  -- Here we would begin the proof
  sorry

end shobha_current_age_l83_83799


namespace primes_less_than_200_with_ones_digit_3_l83_83988

theorem primes_less_than_200_with_ones_digit_3 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, Prime n ∧ n < 200 ∧ n % 10 = 3) ∧ S.card = 12 := 
by
  sorry

end primes_less_than_200_with_ones_digit_3_l83_83988


namespace sampling_interval_l83_83115

theorem sampling_interval (total_students sample_size k : ℕ) (h1 : total_students = 1200) (h2 : sample_size = 40) (h3 : k = total_students / sample_size) : k = 30 :=
by
  sorry

end sampling_interval_l83_83115


namespace polynomial_remainder_l83_83118
-- Importing the broader library needed

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 3

-- The statement of the theorem
theorem polynomial_remainder :
  p 2 = 43 :=
sorry

end polynomial_remainder_l83_83118


namespace binomial_term_is_constant_range_of_a_over_b_l83_83577

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end binomial_term_is_constant_range_of_a_over_b_l83_83577


namespace car_average_speed_l83_83046

noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 24
  let t3 := (D / 3) / 30
  let total_time := t1 + t2 + t3
  D / total_time

theorem car_average_speed :
  average_speed D = 34.2857 := by
  sorry

end car_average_speed_l83_83046


namespace area_of_quadrilateral_EFGH_l83_83733

noncomputable def trapezium_ABCD_midpoints_area : ℝ :=
  let A := (0, 0)
  let B := (2, 0)
  let C := (4, 3)
  let D := (0, 3)
  let E := ((B.1 + C.1)/2, (B.2 + C.2)/2) -- midpoint of BC
  let F := ((C.1 + D.1)/2, (C.2 + D.2)/2) -- midpoint of CD
  let G := ((A.1 + D.1)/2, (A.2 + D.2)/2) -- midpoint of AD
  let H := ((G.1 + E.1)/2, (G.2 + E.2)/2) -- midpoint of GE
  let area := (E.1 * F.2 + F.1 * G.2 + G.1 * H.2 + H.1 * E.2 - F.1 * E.2 - G.1 * F.2 - H.1 * G.2 - E.1 * H.2) / 2
  abs area

theorem area_of_quadrilateral_EFGH : trapezium_ABCD_midpoints_area = 0.75 := by
  sorry

end area_of_quadrilateral_EFGH_l83_83733


namespace total_apples_collected_l83_83087

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end total_apples_collected_l83_83087


namespace problem_1_problem_2_l83_83075

noncomputable def problem_1_solution : Set ℝ := {6, -2}
noncomputable def problem_2_solution : Set ℝ := {2 + Real.sqrt 7, 2 - Real.sqrt 7}

theorem problem_1 :
  {x : ℝ | x^2 - 4 * x - 12 = 0} = problem_1_solution :=
by
  sorry

theorem problem_2 :
  {x : ℝ | x^2 - 4 * x - 3 = 0} = problem_2_solution :=
by
  sorry

end problem_1_problem_2_l83_83075


namespace friends_boat_crossing_impossible_l83_83797

theorem friends_boat_crossing_impossible : 
  ∀ (friends : Finset ℕ) (boat_capacity : ℕ), friends.card = 5 → boat_capacity ≥ 5 → 
  ¬ (∀ group : Finset ℕ, group ⊆ friends → group ≠ ∅ → group.card ≤ boat_capacity → 
  ∃ crossing : ℕ, (crossing = group.card ∧ group ⊆ friends)) :=
by
  intro friends boat_capacity friends_card boat_capacity_cond goal
  sorry

end friends_boat_crossing_impossible_l83_83797


namespace perfect_square_trinomial_l83_83079

theorem perfect_square_trinomial (a k : ℝ) : (∃ b : ℝ, (a^2 + 2*k*a + 9 = (a + b)^2)) ↔ (k = 3 ∨ k = -3) := 
by
  sorry

end perfect_square_trinomial_l83_83079


namespace rectangle_area_l83_83255

-- Definitions based on the conditions
def radius := 6
def diameter := 2 * radius
def width := diameter
def length := 3 * width

-- Statement of the theorem
theorem rectangle_area : (width * length = 432) := by
  sorry

end rectangle_area_l83_83255


namespace correct_barometric_pressure_l83_83219

noncomputable def true_barometric_pressure (p1 p2 v1 v2 T1 T2 observed_pressure_final observed_pressure_initial : ℝ) : ℝ :=
  let combined_gas_law : ℝ := (p1 * v1 * T2) / (v2 * T1)
  observed_pressure_final + combined_gas_law

theorem correct_barometric_pressure :
  true_barometric_pressure 58 56 143 155 288 303 692 704 = 748 :=
by
  sorry

end correct_barometric_pressure_l83_83219


namespace smallest_prime_perimeter_l83_83693

-- Define a function that checks if a number is an odd prime
def is_odd_prime (n : ℕ) : Prop :=
  n > 2 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)) ∧ (n % 2 = 1)

-- Define a function that checks if three numbers are consecutive odd primes
def consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  b = a + 2 ∧ c = b + 2

-- Define a function that checks if three numbers form a scalene triangle and satisfy the triangle inequality
def scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem to prove
theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), consecutive_odd_primes a b c ∧ scalene_triangle a b c ∧ (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l83_83693


namespace avg_expenditure_Feb_to_July_l83_83245

noncomputable def avg_expenditure_Jan_to_Jun : ℝ := 4200
noncomputable def expenditure_January : ℝ := 1200
noncomputable def expenditure_July : ℝ := 1500
noncomputable def total_months_Jan_to_Jun : ℝ := 6
noncomputable def total_months_Feb_to_July : ℝ := 6

theorem avg_expenditure_Feb_to_July :
  (avg_expenditure_Jan_to_Jun * total_months_Jan_to_Jun - expenditure_January + expenditure_July) / total_months_Feb_to_July = 4250 :=
by sorry

end avg_expenditure_Feb_to_July_l83_83245


namespace tip_is_24_l83_83194

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end tip_is_24_l83_83194


namespace maximum_profit_l83_83058

noncomputable def sales_volume (x : ℝ) : ℝ := -10 * x + 1000
noncomputable def profit (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

theorem maximum_profit : ∀ x : ℝ, 44 ≤ x ∧ x ≤ 46 → profit x ≤ 8640 :=
by
  intro x hx
  sorry

end maximum_profit_l83_83058


namespace kilometers_driven_equal_l83_83559

theorem kilometers_driven_equal (x : ℝ) :
  (20 + 0.25 * x = 24 + 0.16 * x) → x = 44 := by
  sorry

end kilometers_driven_equal_l83_83559


namespace imo1987_q6_l83_83851

theorem imo1987_q6 (m n : ℤ) (h : n = m + 2) :
  ⌊(n : ℝ) * Real.sqrt 2⌋ = 2 + ⌊(m : ℝ) * Real.sqrt 2⌋ := 
by
  sorry -- We skip the detailed proof steps here.

end imo1987_q6_l83_83851


namespace sufficient_but_not_necessary_condition_l83_83883

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l83_83883


namespace div_eq_frac_l83_83385

theorem div_eq_frac : 250 / (5 + 12 * 3^2) = 250 / 113 :=
by
  sorry

end div_eq_frac_l83_83385


namespace digit_five_occurrences_l83_83957

/-- 
  Define that a 24-hour digital clock display shows times containing at least one 
  occurrence of the digit '5' a total of 450 times in a 24-hour period.
--/
def contains_digit_five (n : Nat) : Prop := 
  n / 10 = 5 ∨ n % 10 = 5

def count_times_with_digit_five : Nat :=
  let hours_with_five := 2 * 60  -- 05:00-05:59 and 15:00-15:59, each hour has 60 minutes
  let remaining_hours := 22 * 15 -- 22 hours, each hour has 15 minutes
  hours_with_five + remaining_hours

theorem digit_five_occurrences : count_times_with_digit_five = 450 := by
  sorry

end digit_five_occurrences_l83_83957


namespace problem_solution_l83_83614

theorem problem_solution 
  (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) :
  4 * x^4 + 17 * x^2 * y + 4 * y^2 < (m / 4) * (x^4 + 2 * x^2 * y + y^2) ↔ 25 < m :=
sorry

end problem_solution_l83_83614


namespace convex_quad_no_triangle_l83_83553

/-- Given four angles of a convex quadrilateral, it is not always possible to choose any 
three of these angles so that they represent the lengths of the sides of some triangle. -/
theorem convex_quad_no_triangle (α β γ δ : ℝ) 
  (h_sum : α + β + γ + δ = 360) :
  ¬(∀ a b c : ℝ, a + b + c = 360 → (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end convex_quad_no_triangle_l83_83553


namespace solve_for_x_l83_83783

theorem solve_for_x {x : ℤ} (h : x - 2 * x + 3 * x - 4 * x = 120) : x = -60 :=
sorry

end solve_for_x_l83_83783


namespace circle_center_l83_83506

theorem circle_center (a b : ℝ)
  (passes_through_point : (a - 0)^2 + (b - 9)^2 = r^2)
  (is_tangent : (a - 3)^2 + (b - 9)^2 = r^2 ∧ b = 6 * (a - 3) + 9 ∧ (b - 9) / (a - 3) = -1/6) :
  a = 3/2 ∧ b = 37/4 := 
by 
  sorry

end circle_center_l83_83506


namespace find_a2_an_le_2an_next_sum_bounds_l83_83205

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

-- Given conditions
axiom seq_condition (n : ℕ) (h_pos : a n > 0) : 
  a n ^ 2 + a n = 3 * (a (n + 1)) ^ 2 + 2 * a (n + 1)
axiom a1_condition : a 1 = 1

-- Question 1: Prove the value of a2
theorem find_a2 : a 2 = (Real.sqrt 7 - 1) / 3 :=
  sorry

-- Question 2: Prove a_n ≤ 2 * a_{n+1} for any n ∈ N*
theorem an_le_2an_next (n : ℕ) (h_n : n > 0) : a n ≤ 2 * a (n + 1) :=
  sorry

-- Question 3: Prove 2 - 1 / 2^(n - 1) ≤ S_n < 3 for any n ∈ N*
theorem sum_bounds (n : ℕ) (h_n : n > 0) : 
  2 - 1 / 2 ^ (n - 1) ≤ S n ∧ S n < 3 :=
  sorry

end find_a2_an_le_2an_next_sum_bounds_l83_83205


namespace not_possible_select_seven_distinct_weights_no_equal_subsets_l83_83736

theorem not_possible_select_seven_distinct_weights_no_equal_subsets :
  ∀ (s : Finset ℕ), s ⊆ Finset.range 27 → s.card = 7 → ∃ (a b : Finset ℕ), a ≠ b ∧ a ⊆ s ∧ b ⊆ s ∧ a.sum id = b.sum id :=
by
  intro s hs hcard
  sorry

end not_possible_select_seven_distinct_weights_no_equal_subsets_l83_83736


namespace polynomial_degree_one_condition_l83_83463

theorem polynomial_degree_one_condition (P : ℝ → ℝ) (c : ℝ) :
  (∀ a b : ℝ, a < b → (P = fun x => x + c) ∨ (P = fun x => -x + c)) ∧
  (∀ a b : ℝ, a < b →
    (max (P a) (P b) - min (P a) (P b) = b - a)) :=
sorry

end polynomial_degree_one_condition_l83_83463


namespace exists_function_f_l83_83280

theorem exists_function_f :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n * n :=
by
  sorry

end exists_function_f_l83_83280


namespace job_completion_days_l83_83714

variable (m r h d : ℕ)

theorem job_completion_days :
  (m + 2 * r) * (h + 1) * (m * h * d / ((m + 2 * r) * (h + 1))) = m * h * d :=
by
  sorry

end job_completion_days_l83_83714


namespace min_value_f_l83_83016

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x + 81 / x^4

theorem min_value_f : ∃ x > 0, f x = 21 ∧ ∀ y > 0, f y ≥ 21 := by
  sorry

end min_value_f_l83_83016


namespace remainder_g10_div_g_l83_83821

-- Conditions/Definitions
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1
def g10 (x : ℝ) : ℝ := (g (x^10))

-- Theorem/Question
theorem remainder_g10_div_g : (g10 x) % (g x) = 6 :=
by
  sorry

end remainder_g10_div_g_l83_83821


namespace three_digit_numbers_with_2_without_4_l83_83908

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l83_83908


namespace an_values_and_formula_is_geometric_sequence_l83_83558

-- Definitions based on the conditions
def Sn (n : ℕ) : ℝ := sorry  -- S_n to be defined in the context or problem details
def a (n : ℕ) : ℝ := 2 - Sn n

-- Prove the specific values and general formula given the condition a_n = 2 - S_n
theorem an_values_and_formula (Sn : ℕ → ℝ) :
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ a 4 = 1 / 8 ∧ (∀ n, a n = (1 / 2)^(n-1)) :=
sorry

-- Prove the sequence is geometric
theorem is_geometric_sequence (Sn : ℕ → ℝ) :
  (∀ n, a n = (1 / 2)^(n-1)) → ∀ n, a (n + 1) / a n = 1 / 2 :=
sorry

end an_values_and_formula_is_geometric_sequence_l83_83558


namespace pie_shop_revenue_l83_83640

def costPerSlice : Int := 5
def slicesPerPie : Int := 4
def piesSold : Int := 9

theorem pie_shop_revenue : (costPerSlice * slicesPerPie * piesSold) = 180 := 
by
  sorry

end pie_shop_revenue_l83_83640


namespace D_working_alone_completion_time_l83_83279

variable (A_rate D_rate : ℝ)
variable (A_job_hours D_job_hours : ℝ)

-- Conditions
def A_can_complete_in_15_hours : Prop := (A_job_hours = 15)
def A_and_D_together_complete_in_10_hours : Prop := (1/A_rate + 1/D_rate = 10)

-- Proof statement
theorem D_working_alone_completion_time
  (hA : A_job_hours = 15)
  (hAD : 1/A_rate + 1/D_rate = 10) :
  D_job_hours = 30 := sorry

end D_working_alone_completion_time_l83_83279


namespace Natalia_Tuesday_distance_l83_83471

theorem Natalia_Tuesday_distance :
  ∃ T : ℕ, (40 + T + T / 2 + (40 + T / 2) = 180) ∧ T = 33 :=
by
  existsi 33
  -- proof can be filled here
  sorry

end Natalia_Tuesday_distance_l83_83471


namespace a_2008_lt_5_l83_83389

theorem a_2008_lt_5 :
  ∃ a b : ℕ → ℝ, 
    a 1 = 1 ∧ 
    b 1 = 2 ∧ 
    (∀ n, a (n + 1) = (1 + a n + a n * b n) / (b n)) ∧ 
    (∀ n, b (n + 1) = (1 + b n + a n * b n) / (a n)) ∧ 
    a 2008 < 5 := 
sorry

end a_2008_lt_5_l83_83389


namespace farey_sequence_problem_l83_83633

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l83_83633


namespace ball_more_expensive_l83_83727

theorem ball_more_expensive (B L : ℝ) (h1 : 2 * B + 3 * L = 1300) (h2 : 3 * B + 2 * L = 1200) : 
  L - B = 100 := 
sorry

end ball_more_expensive_l83_83727


namespace fifth_term_arithmetic_sequence_l83_83945

variable (x y : ℝ)

def a1 := x + 2 * y^2
def a2 := x - 2 * y^2
def a3 := x + 3 * y
def a4 := x - 4 * y
def d := a2 - a1

theorem fifth_term_arithmetic_sequence : y = -1/2 → 
  x - 10 * y^2 - 4 * y^2 = x - 7/2 := by
  sorry

end fifth_term_arithmetic_sequence_l83_83945


namespace burger_cost_l83_83933

theorem burger_cost (b s : ℕ) (h1 : 3 * b + 2 * s = 385) (h2 : 2 * b + 3 * s = 360) : b = 87 :=
sorry

end burger_cost_l83_83933


namespace jelly_sold_l83_83490

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l83_83490


namespace HA_appears_at_least_once_l83_83630

-- Define the set of letters to be arranged
def letters : List Char := ['A', 'A', 'A', 'H', 'H']

-- Define a function to count the number of ways to arrange letters such that "HA" appears at least once
def countHA(A : List Char) : Nat := sorry

-- The proof problem to establish that there are 9 such arrangements
theorem HA_appears_at_least_once : countHA letters = 9 :=
sorry

end HA_appears_at_least_once_l83_83630


namespace geometric_seq_ratio_l83_83478

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l83_83478


namespace mass_percentage_Al_in_AlBr3_l83_83715

theorem mass_percentage_Al_in_AlBr3 
  (molar_mass_Al : Real := 26.98) 
  (molar_mass_Br : Real := 79.90) 
  (molar_mass_AlBr3 : Real := molar_mass_Al + 3 * molar_mass_Br)
  : (molar_mass_Al / molar_mass_AlBr3) * 100 = 10.11 := 
by 
  -- Here we would provide the proof; skipping with sorry
  sorry

end mass_percentage_Al_in_AlBr3_l83_83715


namespace second_number_l83_83696

theorem second_number (x : ℝ) (h : 3 + x + 333 + 33.3 = 399.6) : x = 30.3 :=
sorry

end second_number_l83_83696


namespace find_numbers_l83_83398

theorem find_numbers (a b c : ℕ) (h₁ : 10 ≤ b ∧ b < 100) (h₂ : 10 ≤ c ∧ c < 100)
    (h₃ : 10^4 * a + 100 * b + c = (a + b + c)^3) : (a = 9 ∧ b = 11 ∧ c = 25) :=
by
  sorry

end find_numbers_l83_83398


namespace three_digit_numbers_left_l83_83857

def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def isABAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ n = 100 * A + 10 * B + A

def isAABOrBAAForm (n : ℕ) : Prop :=
  ∃ A B : ℕ, A ≠ 0 ∧ A ≠ B ∧ (n = 100 * A + 10 * A + B ∨ n = 100 * B + 10 * A + A)

def totalThreeDigitNumbers : ℕ := 900

def countABA : ℕ := 81

def countAABAndBAA : ℕ := 153

theorem three_digit_numbers_left : 
  (totalThreeDigitNumbers - countABA - countAABAndBAA) = 666 := 
by
   sorry

end three_digit_numbers_left_l83_83857


namespace max_value_ahn_operation_l83_83182

theorem max_value_ahn_operation :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (300 - n)^2 - 10 = 39990 :=
by
  sorry

end max_value_ahn_operation_l83_83182


namespace A_visits_all_seats_iff_even_l83_83606

def move_distance_unique (n : ℕ) : Prop := 
  ∀ k l : ℕ, (1 ≤ k ∧ k < n) → (1 ≤ l ∧ l < n) → k ≠ l → (k ≠ l % n)

def visits_all_seats (n : ℕ) : Prop := 
  ∃ A : ℕ → ℕ, 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → (0 ≤ A k ∧ A k < n)) ∧ 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → ∃ (m : ℕ), m ≠ n ∧ A k ≠ (A m % n))

theorem A_visits_all_seats_iff_even (n : ℕ) :
  (move_distance_unique n ∧ visits_all_seats n) ↔ (n % 2 = 0) := 
sorry

end A_visits_all_seats_iff_even_l83_83606


namespace pressure_on_trapezoidal_dam_l83_83254

noncomputable def water_pressure_on_trapezoidal_dam (ρ g h a b : ℝ) : ℝ :=
  ρ * g * (h^2) * (2 * a + b) / 6

theorem pressure_on_trapezoidal_dam
  (ρ g h a b : ℝ) : water_pressure_on_trapezoidal_dam ρ g h a b = ρ * g * (h^2) * (2 * a + b) / 6 := by
  sorry

end pressure_on_trapezoidal_dam_l83_83254


namespace sum_of_squares_of_extremes_l83_83860

theorem sum_of_squares_of_extremes
  (a b c : ℕ)
  (h1 : 2*b = 3*a)
  (h2 : 3*b = 4*c)
  (h3 : b = 9) :
  a^2 + c^2 = 180 :=
sorry

end sum_of_squares_of_extremes_l83_83860


namespace rectangle_shaded_area_equal_l83_83855

theorem rectangle_shaded_area_equal {x : ℝ} :
  let total_area := 72
  let shaded_area := 24 + 6*x
  let non_shaded_area := total_area / 2
  shaded_area = non_shaded_area → x = 2 := 
by 
  intros h
  sorry

end rectangle_shaded_area_equal_l83_83855


namespace emily_furniture_assembly_time_l83_83211

-- Definitions based on conditions
def chairs := 4
def tables := 2
def time_per_piece := 8

-- Proof statement
theorem emily_furniture_assembly_time : (chairs + tables) * time_per_piece = 48 :=
by
  sorry

end emily_furniture_assembly_time_l83_83211


namespace algebra_minimum_value_l83_83800

theorem algebra_minimum_value :
  ∀ x y : ℝ, ∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 + 6*x - 2*y + 12 ≥ m) ∧ m = 2 :=
by
  sorry

end algebra_minimum_value_l83_83800


namespace probability_max_min_difference_is_five_l83_83427

theorem probability_max_min_difference_is_five : 
  let total_outcomes := 6 ^ 4
  let outcomes_without_1 := 5 ^ 4
  let outcomes_without_6 := 5 ^ 4
  let outcomes_without_1_and_6 := 4 ^ 4
  total_outcomes - 2 * outcomes_without_1 + outcomes_without_1_and_6 = 302 →
  (302 : ℚ) / total_outcomes = 151 / 648 :=
by
  intros
  sorry

end probability_max_min_difference_is_five_l83_83427


namespace recycling_weight_l83_83906

theorem recycling_weight :
  let marcus_milk_bottles := 25
  let john_milk_bottles := 20
  let sophia_milk_bottles := 15
  let marcus_cans := 30
  let john_cans := 25
  let sophia_cans := 35
  let milk_bottle_weight := 0.5
  let can_weight := 0.025

  let total_milk_bottles_weight := (marcus_milk_bottles + john_milk_bottles + sophia_milk_bottles) * milk_bottle_weight
  let total_cans_weight := (marcus_cans + john_cans + sophia_cans) * can_weight
  let combined_weight := total_milk_bottles_weight + total_cans_weight

  combined_weight = 32.25 :=
by
  sorry

end recycling_weight_l83_83906


namespace problem_one_problem_two_problem_three_l83_83270

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

noncomputable def M (x : ℝ) : ℝ :=
  if f x >= g x then g x else f x

noncomputable def condition_one : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → -6 ≤ h x ∧ h x ≤ 2

noncomputable def condition_two : Prop :=
  ∃ x, (M x = 1 ∧ 0 < x ∧ x ≤ 2) ∧ (∀ y, 0 < y ∧ y < x → M y < 1)

noncomputable def condition_three : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → f (x^2) * f (Real.sqrt x) ≥ g x * -3

theorem problem_one : condition_one := sorry
theorem problem_two : condition_two := sorry
theorem problem_three : condition_three := sorry

end problem_one_problem_two_problem_three_l83_83270


namespace sum_of_1984_consecutive_integers_not_square_l83_83601

theorem sum_of_1984_consecutive_integers_not_square :
  ∀ n : ℕ, ¬ ∃ k : ℕ, 992 * (2 * n + 1985) = k * k := by
  sorry

end sum_of_1984_consecutive_integers_not_square_l83_83601


namespace cylinder_new_volume_l83_83660

-- Definitions based on conditions
def original_volume_r_h (π R H : ℝ) : ℝ := π * R^2 * H

def new_volume (π R H : ℝ) : ℝ := π * (3 * R)^2 * (2 * H)

theorem cylinder_new_volume (π R H : ℝ) (h_original_volume : original_volume_r_h π R H = 15) :
  new_volume π R H = 270 :=
by sorry

end cylinder_new_volume_l83_83660


namespace monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l83_83034

variables {f : ℝ → ℝ}

-- Definition that f is monotonically increasing
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

-- Definition of the derivative being non-negative everywhere
def non_negative_derivative (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ (deriv f) x

theorem monotonically_increasing_implies_non_negative_derivative (f : ℝ → ℝ) :
  monotonically_increasing f → non_negative_derivative f :=
sorry

theorem non_negative_derivative_not_implies_monotonically_increasing (f : ℝ → ℝ) :
  non_negative_derivative f → ¬ monotonically_increasing f :=
sorry

end monotonically_increasing_implies_non_negative_derivative_non_negative_derivative_not_implies_monotonically_increasing_l83_83034


namespace transformed_sum_l83_83912

theorem transformed_sum (n : ℕ) (y : Fin n → ℝ) (s : ℝ) (h : s = (Finset.univ.sum (fun i => y i))) :
  Finset.univ.sum (fun i => 3 * (y i) + 30) = 3 * s + 30 * n :=
by 
  sorry

end transformed_sum_l83_83912


namespace problem_part1_problem_part2_l83_83954

theorem problem_part1 (α : ℝ) (h : Real.tan α = -2) :
    (3 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α - Real.sin α) = -4 / 7 := 
    sorry

theorem problem_part2 (α : ℝ) (h : Real.tan α = -2) :
    3 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = -5 := 
    sorry

end problem_part1_problem_part2_l83_83954


namespace black_friday_sales_l83_83735

variable (n : ℕ) (initial_sales increment : ℕ)

def yearly_sales (sales: ℕ) (inc: ℕ) (years: ℕ) : ℕ :=
  sales + years * inc

theorem black_friday_sales (h1 : initial_sales = 327) (h2 : increment = 50) :
  yearly_sales initial_sales increment 3 = 477 := by
  sorry

end black_friday_sales_l83_83735


namespace unique_integer_sequence_l83_83903

theorem unique_integer_sequence :
  ∃ a : ℕ → ℤ, a 1 = 1 ∧ a 2 > 1 ∧ ∀ n ≥ 1, (a (n + 1))^3 + 1 = a n * a (n + 2) :=
sorry

end unique_integer_sequence_l83_83903


namespace total_snakes_in_park_l83_83124

theorem total_snakes_in_park :
  ∀ (pythons boa_constrictors rattlesnakes total_snakes : ℕ),
    boa_constrictors = 40 →
    pythons = 3 * boa_constrictors →
    rattlesnakes = 40 →
    total_snakes = boa_constrictors + pythons + rattlesnakes →
    total_snakes = 200 :=
by
  intros pythons boa_constrictors rattlesnakes total_snakes h1 h2 h3 h4
  rw [h1, h3] at h4
  rw [h2] at h4
  sorry

end total_snakes_in_park_l83_83124


namespace second_order_arithmetic_progression_a100_l83_83681

theorem second_order_arithmetic_progression_a100 :
  ∀ (a : ℕ → ℕ), 
    a 1 = 2 → 
    a 2 = 3 → 
    a 3 = 5 → 
    (∀ n, a (n + 1) - a n = n) → 
    a 100 = 4952 :=
by
  intros a h1 h2 h3 hdiff
  sorry

end second_order_arithmetic_progression_a100_l83_83681


namespace ratio_problem_l83_83358

theorem ratio_problem
  (w x y z : ℝ)
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 2 / 3)
  (h3 : w / z = 3 / 5) :
  (x + y) / z = 27 / 10 :=
by
  sorry

end ratio_problem_l83_83358


namespace find_f_value_l83_83317

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^5 - b * x^3 + c * x - 3

theorem find_f_value (a b c : ℝ) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by
  sorry

end find_f_value_l83_83317


namespace angle_terminal_side_equiv_l83_83602

def angle_equiv_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_equiv : angle_equiv_terminal_side (-Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end angle_terminal_side_equiv_l83_83602


namespace range_of_x_coordinate_l83_83440

def is_on_line (A : ℝ × ℝ) : Prop := A.1 + A.2 = 6

def is_on_circle (C : ℝ × ℝ) : Prop := (C.1 - 1)^2 + (C.2 - 1)^2 = 4

def angle_BAC_is_60_degrees (A B C : ℝ × ℝ) : Prop :=
  -- This definition is simplified as an explanation. Angle computation in Lean might be more intricate.
  sorry 

theorem range_of_x_coordinate (A : ℝ × ℝ) (B C : ℝ × ℝ)
  (hA_on_line : is_on_line A)
  (hB_on_circle : is_on_circle B)
  (hC_on_circle : is_on_circle C)
  (h_angle_BAC : angle_BAC_is_60_degrees A B C) :
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
sorry

end range_of_x_coordinate_l83_83440


namespace combined_tennis_percentage_l83_83375

variable (totalStudentsNorth totalStudentsSouth : ℕ)
variable (percentTennisNorth percentTennisSouth : ℕ)

def studentsPreferringTennisNorth : ℕ := totalStudentsNorth * percentTennisNorth / 100
def studentsPreferringTennisSouth : ℕ := totalStudentsSouth * percentTennisSouth / 100

def totalStudentsBothSchools : ℕ := totalStudentsNorth + totalStudentsSouth
def studentsPreferringTennisBothSchools : ℕ := studentsPreferringTennisNorth totalStudentsNorth percentTennisNorth
                                            + studentsPreferringTennisSouth totalStudentsSouth percentTennisSouth

def combinedPercentTennis : ℕ := studentsPreferringTennisBothSchools totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth
                                 * 100 / totalStudentsBothSchools totalStudentsNorth totalStudentsSouth

theorem combined_tennis_percentage :
  (totalStudentsNorth = 1800) →
  (totalStudentsSouth = 2700) →
  (percentTennisNorth = 25) →
  (percentTennisSouth = 35) →
  combinedPercentTennis totalStudentsNorth totalStudentsSouth percentTennisNorth percentTennisSouth = 31 :=
by
  intros
  sorry

end combined_tennis_percentage_l83_83375


namespace somu_present_age_l83_83117

theorem somu_present_age (S F : ℕ) (h1 : S = (1 / 3) * F)
    (h2 : S - 5 = (1 / 5) * (F - 5)) : S = 10 := by
  sorry

end somu_present_age_l83_83117


namespace waiter_tip_amount_l83_83094

theorem waiter_tip_amount (n n_no_tip E : ℕ) (h_n : n = 10) (h_no_tip : n_no_tip = 5) (h_E : E = 15) :
  (E / (n - n_no_tip) = 3) :=
by
  -- Proof goes here (we are only writing the statement with sorry)
  sorry

end waiter_tip_amount_l83_83094


namespace intersection_nonempty_implies_range_l83_83371

namespace ProofProblem

def M (x y : ℝ) : Prop := x + y + 1 ≥ Real.sqrt (2 * (x^2 + y^2))
def N (a x y : ℝ) : Prop := |x - a| + |y - 1| ≤ 1

theorem intersection_nonempty_implies_range (a : ℝ) :
  (∃ x y : ℝ, M x y ∧ N a x y) → (1 - Real.sqrt 6 ≤ a ∧ a ≤ 3 + Real.sqrt 10) :=
by
  sorry

end ProofProblem

end intersection_nonempty_implies_range_l83_83371


namespace solve_for_b_l83_83889

theorem solve_for_b (b : ℝ) : 
  (∀ x y, 3 * y - 2 * x + 6 = 0 ↔ y = (2 / 3) * x - 2) → 
  (∀ x y, 4 * y + b * x + 3 = 0 ↔ y = -(b / 4) * x - 3 / 4) → 
  (∀ m1 m2, (m1 = (2 / 3)) → (m2 = -(b / 4)) → m1 * m2 = -1) → 
  b = 6 :=
sorry

end solve_for_b_l83_83889


namespace dishwasher_spending_l83_83082

theorem dishwasher_spending (E : ℝ) (h1 : E > 0) 
    (rent : ℝ := 0.40 * E)
    (left_over : ℝ := 0.28 * E)
    (spent : ℝ := 0.72 * E)
    (dishwasher : ℝ := spent - rent)
    (difference : ℝ := rent - dishwasher) :
    ((difference / rent) * 100) = 20 := 
by
  sorry

end dishwasher_spending_l83_83082


namespace probability_of_A_given_B_l83_83090

-- Definitions of events
def tourist_attractions : List String := ["Pengyuan", "Jiuding Mountain", "Garden Expo Park", "Yunlong Lake", "Pan'an Lake"]

-- Probabilities for each scenario
noncomputable def P_AB : ℝ := 8 / 25
noncomputable def P_B : ℝ := 20 / 25
noncomputable def P_A_given_B : ℝ := 2 / 5

-- Proof statement
theorem probability_of_A_given_B : (P_AB / P_B) = P_A_given_B :=
by
  sorry

end probability_of_A_given_B_l83_83090


namespace gcf_75_100_l83_83258

theorem gcf_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcf_75_100_l83_83258


namespace gcd_1113_1897_l83_83286

theorem gcd_1113_1897 : Int.gcd 1113 1897 = 7 := by
  sorry

end gcd_1113_1897_l83_83286


namespace gcd_of_78_and_36_l83_83334

theorem gcd_of_78_and_36 : Int.gcd 78 36 = 6 := by
  sorry

end gcd_of_78_and_36_l83_83334


namespace find_Y_length_l83_83979

theorem find_Y_length (Y : ℝ) : 
  (3 + 2 + 3 + 4 + Y = 7 + 4 + 2) → Y = 1 :=
by
  intro h
  sorry

end find_Y_length_l83_83979


namespace factorize_expression_l83_83121

theorem factorize_expression (a b : ℝ) : 
  a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 := by sorry

end factorize_expression_l83_83121


namespace negation_of_universal_statement_l83_83332

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^4 - x^3 + x^2 + 5 ≤ 0) ↔ (∃ x : ℝ, x^4 - x^3 + x^2 + 5 > 0) :=
by sorry

end negation_of_universal_statement_l83_83332


namespace steve_commute_l83_83761

theorem steve_commute :
  ∃ (D : ℝ), 
    (∃ (V : ℝ), 2 * V = 5 ∧ (D / V + D / (2 * V) = 6)) ∧ D = 10 :=
by
  sorry

end steve_commute_l83_83761


namespace Riku_stickers_more_times_l83_83310

theorem Riku_stickers_more_times (Kristoff_stickers Riku_stickers : ℕ) 
  (h1 : Kristoff_stickers = 85) (h2 : Riku_stickers = 2210) : 
  Riku_stickers / Kristoff_stickers = 26 := 
by
  sorry

end Riku_stickers_more_times_l83_83310


namespace interest_rate_per_annum_is_four_l83_83174

-- Definitions
def P : ℕ := 300
def t : ℕ := 8
def I : ℤ := P - 204

-- Interest formula
def simple_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℤ := P * r * t / 100

-- Statement to prove
theorem interest_rate_per_annum_is_four :
  ∃ r : ℕ, I = simple_interest P r t ∧ r = 4 :=
by sorry

end interest_rate_per_annum_is_four_l83_83174


namespace min_radius_of_circumcircle_l83_83419

theorem min_radius_of_circumcircle {a b : ℝ} (ha : a = 3) (hb : b = 4) : 
∃ R : ℝ, R = 2.5 ∧ (∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ a^2 + b^2 = c^2 ∧ 2 * R = c) :=
by 
  sorry

end min_radius_of_circumcircle_l83_83419


namespace commutative_matrices_implies_fraction_l83_83107

-- Definitions
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 3], ![4, 5]]
def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

-- Theorem Statement
theorem commutative_matrices_implies_fraction (a b c d : ℝ) 
    (h1 : A * B a b c d = B a b c d * A) 
    (h2 : 4 * b ≠ c) : 
    (a - d) / (c - 4 * b) = 3 / 8 :=
by
  sorry

end commutative_matrices_implies_fraction_l83_83107


namespace journey_speed_l83_83808

theorem journey_speed 
  (total_time : ℝ)
  (total_distance : ℝ)
  (second_half_speed : ℝ)
  (first_half_speed : ℝ) :
  total_time = 30 ∧ total_distance = 400 ∧ second_half_speed = 10 ∧
  2 * (total_distance / 2 / second_half_speed) + total_distance / 2 / first_half_speed = total_time →
  first_half_speed = 20 :=
by
  intros hyp
  sorry

end journey_speed_l83_83808


namespace piglet_straws_l83_83086

theorem piglet_straws (total_straws : ℕ) (straws_adult_pigs_ratio : ℚ) (straws_piglets_ratio : ℚ) (number_piglets : ℕ) :
  total_straws = 300 →
  straws_adult_pigs_ratio = 3/5 →
  straws_piglets_ratio = 1/3 →
  number_piglets = 20 →
  (total_straws * straws_piglets_ratio) / number_piglets = 5 := 
by
  intros
  sorry

end piglet_straws_l83_83086


namespace binom_coeffs_not_coprime_l83_83854

open Nat

theorem binom_coeffs_not_coprime (n k m : ℕ) (h1 : 0 < k) (h2 : k < m) (h3 : m < n) : 
  Nat.gcd (Nat.choose n k) (Nat.choose n m) > 1 := 
sorry

end binom_coeffs_not_coprime_l83_83854


namespace sufficient_condition_not_necessary_condition_l83_83814

variable {a b : ℝ} 

theorem sufficient_condition (h : a < b ∧ b < 0) : a ^ 2 > b ^ 2 :=
sorry

theorem not_necessary_condition : ¬ (∀ {a b : ℝ}, a ^ 2 > b ^ 2 → a < b ∧ b < 0) :=
sorry

end sufficient_condition_not_necessary_condition_l83_83814


namespace television_price_l83_83700

theorem television_price (SP : ℝ) (RP : ℕ) (discount : ℝ) (h1 : discount = 0.20) (h2 : SP = RP - discount * RP) (h3 : SP = 480) : RP = 600 :=
by
  sorry

end television_price_l83_83700


namespace simplify_expression_l83_83701

theorem simplify_expression : (1 / (1 + Real.sqrt 3) * 1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 :=
by
  sorry

end simplify_expression_l83_83701


namespace expression_value_l83_83527

theorem expression_value (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x ^ 4 - 6 * x ^ 3 - 2 * x ^ 2 + 18 * x + 23) / (x ^ 2 - 8 * x + 15) = 5 :=
by
  sorry

end expression_value_l83_83527


namespace imaginary_part_of_z_l83_83354

open Complex -- open complex number functions

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) * (2 - I) = 5 * I) :
  z.im = 2 :=
sorry

end imaginary_part_of_z_l83_83354


namespace tom_needs_more_blue_tickets_l83_83241

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l83_83241


namespace four_digit_numbers_property_l83_83231

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l83_83231


namespace b_over_c_equals_1_l83_83823

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end b_over_c_equals_1_l83_83823


namespace base_case_inequality_induction_inequality_l83_83708

theorem base_case_inequality : 2^5 > 5^2 + 1 := by
  -- Proof not required
  sorry

theorem induction_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  -- Proof not required
  sorry

end base_case_inequality_induction_inequality_l83_83708


namespace find_m_value_l83_83030

noncomputable def m_value (x : ℤ) (m : ℝ) : Prop :=
  3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1 ∧
  (∃ x, x ≥ 12 ∧ (1 / 2 : ℝ) * x - m = 5)

theorem find_m_value : ∃ m : ℝ, ∀ x : ℤ, m_value x m → m = 1 :=
by
  sorry

end find_m_value_l83_83030


namespace ice_cream_ratio_l83_83485

-- Definitions based on the conditions
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := oli_scoops + 4

-- Statement to prove the ratio
theorem ice_cream_ratio :
  victoria_scoops / oli_scoops = 2 :=
by
  -- The exact proof strategy here is omitted with 'sorry'
  sorry

end ice_cream_ratio_l83_83485


namespace Jake_weight_correct_l83_83465

def Mildred_weight : ℕ := 59
def Carol_weight : ℕ := Mildred_weight + 9
def Jake_weight : ℕ := 2 * Carol_weight

theorem Jake_weight_correct : Jake_weight = 136 := by
  sorry

end Jake_weight_correct_l83_83465


namespace find_first_discount_percentage_l83_83112

def first_discount_percentage 
  (price_initial : ℝ) 
  (price_final : ℝ) 
  (discount_x : ℝ) 
  : Prop := 
  price_initial * (1 - discount_x / 100) * 0.9 * 0.95 = price_final

theorem find_first_discount_percentage :
  first_discount_percentage 9941.52 6800 20.02 :=
by
  sorry

end find_first_discount_percentage_l83_83112


namespace rice_yield_l83_83627

theorem rice_yield (X : ℝ) (h1 : 0 ≤ X ∧ X ≤ 40) :
    0.75 * 400 * X + 0.25 * 800 * X + 500 * (40 - X) = 20000 := by
  sorry

end rice_yield_l83_83627


namespace nitin_ranks_from_last_l83_83546

def total_students : ℕ := 75

def math_rank_start : ℕ := 24
def english_rank_start : ℕ := 18

def rank_from_last (total : ℕ) (rank_start : ℕ) : ℕ :=
  total - rank_start + 1

theorem nitin_ranks_from_last :
  rank_from_last total_students math_rank_start = 52 ∧
  rank_from_last total_students english_rank_start = 58 :=
by
  sorry

end nitin_ranks_from_last_l83_83546


namespace jane_nail_polish_drying_time_l83_83486

theorem jane_nail_polish_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let index_finger_1 := 8
  let index_finger_2 := 10
  let middle_finger := 12
  let ring_finger := 11
  let pinky_finger := 14
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + index_finger_1 + index_finger_2 + middle_finger + ring_finger + pinky_finger + top_coat = 86 :=
by sorry

end jane_nail_polish_drying_time_l83_83486


namespace grid_problem_l83_83809

theorem grid_problem 
    (A B : ℕ)
    (H1 : 1 ≠ A)
    (H2 : 1 ≠ B)
    (H3 : 2 ≠ A)
    (H4 : 2 ≠ B)
    (H5 : 3 ≠ A)
    (H6 : 3 ≠ B)
    (H7 : A = 2)
    (H8 : B = 1)
    :
    A * B = 2 :=
by
  sorry

end grid_problem_l83_83809


namespace interior_diagonal_length_l83_83813

theorem interior_diagonal_length (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26)
  (h2 : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2) = 23 :=
by
  sorry

end interior_diagonal_length_l83_83813


namespace x_squared_plus_y_squared_geq_five_l83_83381

theorem x_squared_plus_y_squared_geq_five (x y : ℝ) (h : abs (x - 2 * y) = 5) : x^2 + y^2 ≥ 5 := 
sorry

end x_squared_plus_y_squared_geq_five_l83_83381


namespace a3_plus_a4_l83_83592

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 3^(n + 1)

theorem a3_plus_a4 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : sum_of_sequence S a) :
  a 3 + a 4 = 216 :=
sorry

end a3_plus_a4_l83_83592


namespace manager_decision_correct_l83_83838

theorem manager_decision_correct (x : ℝ) (profit : ℝ) 
  (h_condition1 : ∀ (x : ℝ), profit = (2 * x + 20) * (40 - x)) 
  (h_condition2 : 0 ≤ x ∧ x ≤ 40)
  (h_price_reduction : x = 15) :
  profit = 1250 :=
by
  sorry

end manager_decision_correct_l83_83838


namespace range_of_k_l83_83771

noncomputable def equation (k x : ℝ) : ℝ := 4^x - k * 2^x + k + 3

theorem range_of_k {x : ℝ} (h : ∀ k, equation k x = 0 → ∃! x : ℝ, equation k x = 0) :
  ∃ k : ℝ, (k = 6 ∨ k < -3)∧ (∀ y, equation k y ≠ 0 → (y ≠ x)) :=
sorry

end range_of_k_l83_83771


namespace total_food_items_in_one_day_l83_83091

-- Define the food consumption for each individual
def JorgeCroissants := 7
def JorgeCakes := 18
def JorgePizzas := 30

def GiulianaCroissants := 5
def GiulianaCakes := 14
def GiulianaPizzas := 25

def MatteoCroissants := 6
def MatteoCakes := 16
def MatteoPizzas := 28

-- Define the total number of each food type consumed
def totalCroissants := JorgeCroissants + GiulianaCroissants + MatteoCroissants
def totalCakes := JorgeCakes + GiulianaCakes + MatteoCakes
def totalPizzas := JorgePizzas + GiulianaPizzas + MatteoPizzas

-- The theorem statement
theorem total_food_items_in_one_day : 
  totalCroissants + totalCakes + totalPizzas = 149 :=
by
  -- Proof is omitted
  sorry

end total_food_items_in_one_day_l83_83091


namespace loss_percentage_is_13_l83_83842

def cost_price : ℕ := 1500
def selling_price : ℕ := 1305
def loss : ℕ := cost_price - selling_price
def loss_percentage : ℚ := (loss : ℚ) / cost_price * 100

theorem loss_percentage_is_13 :
  loss_percentage = 13 := 
by
  sorry

end loss_percentage_is_13_l83_83842


namespace car_avg_mpg_B_to_C_is_11_11_l83_83484

noncomputable def avg_mpg_B_to_C (D : ℝ) : ℝ :=
  let avg_mpg_total := 42.857142857142854
  let x := (100 : ℝ) / 9
  let total_distance := (3 / 2) * D
  let total_gallons := (D / 40) + (D / (2 * x))
  (total_distance / total_gallons)

/-- Prove the car's average miles per gallon from town B to town C is 100/9 mpg. -/
theorem car_avg_mpg_B_to_C_is_11_11 (D : ℝ) (h1 : D > 0):
  avg_mpg_B_to_C D = 100 / 9 :=
by
  sorry

end car_avg_mpg_B_to_C_is_11_11_l83_83484


namespace max_rectangle_perimeter_l83_83562

theorem max_rectangle_perimeter (n : ℕ) (a b : ℕ) (ha : a * b = 180) (hb: ∀ (a b : ℕ),  6 ∣ (a * b) → a * b = 180): 
  2 * (a + b) ≤ 184 :=
sorry

end max_rectangle_perimeter_l83_83562


namespace norm_squared_sum_l83_83224

variables (p q : ℝ × ℝ)
def n : ℝ × ℝ := (4, -2)
variables (h_midpoint : n = ((p.1 + q.1) / 2, (p.2 + q.2) / 2))
variables (h_dot_product : p.1 * q.1 + p.2 * q.2 = 12)

theorem norm_squared_sum : (p.1 ^ 2 + p.2 ^ 2) + (q.1 ^ 2 + q.2 ^ 2) = 56 :=
by
  sorry

end norm_squared_sum_l83_83224


namespace painted_cube_l83_83686

theorem painted_cube (n : ℕ) (h : 3 / 4 * (6 * n ^ 3) = 4 * n ^ 2) : n = 2 := sorry

end painted_cube_l83_83686


namespace percentage_increase_is_50_l83_83019

-- Defining the conditions
def new_wage : ℝ := 51
def original_wage : ℝ := 34
def increase : ℝ := new_wage - original_wage

-- Proving the required percentage increase is 50%
theorem percentage_increase_is_50 :
  (increase / original_wage) * 100 = 50 := by
  sorry

end percentage_increase_is_50_l83_83019


namespace computer_price_problem_l83_83102

theorem computer_price_problem (x : ℝ) (h : x + 0.30 * x = 351) : x + 351 = 621 :=
by
  sorry

end computer_price_problem_l83_83102


namespace find_years_l83_83404

def sum_interest_years (P R : ℝ) (T : ℝ) : Prop :=
  (P * (R + 5) / 100 * T = P * R / 100 * T + 300) ∧ P = 600

theorem find_years {R : ℝ} {T : ℝ} (h1 : sum_interest_years 600 R T) : T = 10 :=
by
  -- proof omitted
  sorry

end find_years_l83_83404


namespace solve_for_y_l83_83565

noncomputable def roots := [(-126 + Real.sqrt 13540) / 8, (-126 - Real.sqrt 13540) / 8]

theorem solve_for_y (y : ℝ) :
  (8*y^2 + 176*y + 2) / (3*y + 74) = 4*y + 2 →
  y = roots[0] ∨ y = roots[1] :=
by
  intros
  sorry

end solve_for_y_l83_83565


namespace number_of_vans_needed_l83_83183

theorem number_of_vans_needed (capacity_per_van : ℕ) (students : ℕ) (adults : ℕ)
  (h_capacity : capacity_per_van = 9)
  (h_students : students = 40)
  (h_adults : adults = 14) :
  (students + adults + capacity_per_van - 1) / capacity_per_van = 6 := by
  sorry

end number_of_vans_needed_l83_83183


namespace Thelma_cuts_each_tomato_into_8_slices_l83_83321

-- Conditions given in the problem
def slices_per_meal := 20
def family_size := 8
def tomatoes_needed := 20

-- The quantity we want to prove
def slices_per_tomato := 8

-- Statement to be proven: Thelma cuts each green tomato into the correct number of slices
theorem Thelma_cuts_each_tomato_into_8_slices :
  (slices_per_meal * family_size) = (tomatoes_needed * slices_per_tomato) :=
by 
  sorry

end Thelma_cuts_each_tomato_into_8_slices_l83_83321


namespace find_a6_l83_83169

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem find_a6 :
  ∀ (a b : ℕ → ℕ),
    a 1 = 3 →
    b 1 = 2 →
    b 3 = 6 →
    is_arithmetic_sequence b →
    (∀ n, b n = a (n + 1) - a n) →
    a 6 = 33 :=
by
  intros a b h_a1 h_b1 h_b3 h_arith h_diff
  sorry

end find_a6_l83_83169


namespace angle_sum_around_point_l83_83158

theorem angle_sum_around_point {x : ℝ} (h : 2 * x + 210 = 360) : x = 75 :=
by
  sorry

end angle_sum_around_point_l83_83158


namespace find_number_l83_83238

theorem find_number (x : ℤ) (h : 5 + x * 5 = 15) : x = 2 :=
by
  sorry

end find_number_l83_83238


namespace simplify_expression_l83_83500

variable (z : ℝ)

theorem simplify_expression: (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z :=
by sorry

end simplify_expression_l83_83500


namespace selling_price_l83_83690

def initial_cost : ℕ := 600
def food_cost_per_day : ℕ := 20
def number_of_days : ℕ := 40
def vaccination_and_deworming_cost : ℕ := 500
def profit : ℕ := 600

theorem selling_price (S : ℕ) :
  S = initial_cost + (food_cost_per_day * number_of_days) + vaccination_and_deworming_cost + profit :=
by
  sorry

end selling_price_l83_83690


namespace factorial_mod_prime_l83_83483
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l83_83483


namespace profit_percentage_l83_83909

theorem profit_percentage (C S : ℝ) (h1 : C > 0) (h2 : S > 0)
  (h3 : S - 1.25 * C = 0.7023809523809523 * S) :
  ((S - C) / C) * 100 = 320 := by
sorry

end profit_percentage_l83_83909


namespace tangent_line_value_of_a_l83_83952

theorem tangent_line_value_of_a (a : ℝ) :
  (∃ (m : ℝ), (2 * m - 1 = a * m + Real.log m) ∧ (a + 1 / m = 2)) → a = 1 :=
by 
sorry

end tangent_line_value_of_a_l83_83952


namespace find_m_from_intersection_l83_83499

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

-- Prove the relationship given the conditions
theorem find_m_from_intersection (m : ℕ) (h : A ∩ B m = {2, 3}) : m = 3 := 
by 
  sorry

end find_m_from_intersection_l83_83499


namespace emily_51_49_calculations_l83_83754

theorem emily_51_49_calculations :
  (51^2 = 50^2 + 101) ∧ (49^2 = 50^2 - 99) :=
by
  sorry

end emily_51_49_calculations_l83_83754


namespace chord_to_diameter_ratio_l83_83657

open Real

theorem chord_to_diameter_ratio
  (r R : ℝ) (h1 : r = R / 2)
  (a : ℝ)
  (h2 : r^2 = a^2 * 3 / 2) :
  3 * a / (2 * R) = 3 * sqrt 6 / 8 :=
by
  sorry

end chord_to_diameter_ratio_l83_83657


namespace directrix_of_parabola_l83_83122

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = (x^2 - 4 * x + 3) / 8 → y = -9 / 8 :=
by
  sorry

end directrix_of_parabola_l83_83122


namespace oil_remaining_in_tank_l83_83540

/- Definitions for the problem conditions -/
def tankCapacity : Nat := 32
def totalOilPurchased : Nat := 728

/- Theorem statement -/
theorem oil_remaining_in_tank : totalOilPurchased % tankCapacity = 24 := by
  sorry

end oil_remaining_in_tank_l83_83540


namespace teal_more_blue_l83_83390

theorem teal_more_blue (total : ℕ) (green : ℕ) (both_green_blue : ℕ) (neither_green_blue : ℕ)
  (h1 : total = 150) (h2 : green = 90) (h3 : both_green_blue = 40) (h4 : neither_green_blue = 25) :
  ∃ (blue : ℕ), blue = 75 :=
by
  sorry

end teal_more_blue_l83_83390


namespace total_red_cards_l83_83962

theorem total_red_cards (num_standard_decks : ℕ) (num_special_decks : ℕ)
  (red_standard_deck : ℕ) (additional_red_special_deck : ℕ)
  (total_decks : ℕ) (h1 : num_standard_decks = 5)
  (h2 : num_special_decks = 10)
  (h3 : red_standard_deck = 26)
  (h4 : additional_red_special_deck = 4)
  (h5 : total_decks = num_standard_decks + num_special_decks) :
  num_standard_decks * red_standard_deck +
  num_special_decks * (red_standard_deck + additional_red_special_deck) = 430 := by
  -- Proof is omitted.
  sorry

end total_red_cards_l83_83962


namespace correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l83_83330

-- Definition 1: Variance and regression coefficients and correlation coefficient calculation
noncomputable def correlation_coefficient : ℝ := 4.7 * (Real.sqrt (2 / 50))

-- Theorem 1: Correlation coefficient computation
theorem correlation_coefficient_value :
  correlation_coefficient = 0.94 :=
sorry

-- Definition 2: Chi-square calculation for independence test
noncomputable def chi_square : ℝ :=
  (100 * ((30 * 35 - 20 * 15)^2 : ℝ)) / (50 * 50 * 45 * 55)

-- Theorem 2: Chi-square test result
theorem relation_between_gender_and_electric_car :
  chi_square > 6.635 :=
sorry

-- Definition 3: Probability distribution and expectation calculation
def probability_distribution : Finset ℚ :=
{(21/55), (28/55), (6/55)}

noncomputable def expectation_X : ℚ :=
(0 * (21/55) + 1 * (28/55) + 2 * (6/55))

-- Theorem 3: Expectation of X calculation
theorem expectation_X_value :
  expectation_X = 8/11 :=
sorry

end correlation_coefficient_value_relation_between_gender_and_electric_car_expectation_X_value_l83_83330


namespace present_worth_approx_l83_83752

noncomputable def amount_after_years (P : ℝ) : ℝ :=
  let A1 := P * (1 + 5 / 100)                      -- Amount after the first year.
  let A2 := A1 * (1 + 5 / 100)^2                   -- Amount after the second year.
  let A3 := A2 * (1 + 3 / 100)^4                   -- Amount after the third year.
  A3

noncomputable def banker's_gain (P : ℝ) : ℝ :=
  amount_after_years P - P

theorem present_worth_approx :
  ∃ P : ℝ, abs (P - 114.94) < 1 ∧ banker's_gain P = 36 :=
sorry

end present_worth_approx_l83_83752


namespace intersection_points_of_line_l83_83599

theorem intersection_points_of_line (x y : ℝ) :
  ((y = 2 * x - 1) → (y = 0 → x = 0.5)) ∧
  ((y = 2 * x - 1) → (x = 0 → y = -1)) :=
by sorry

end intersection_points_of_line_l83_83599


namespace simplify_tan_cot_fraction_l83_83812

theorem simplify_tan_cot_fraction :
  let tan45 := 1
  let cot45 := 1
  (tan45^3 + cot45^3) / (tan45 + cot45) = 1 := by
    sorry

end simplify_tan_cot_fraction_l83_83812


namespace circumradius_of_triangle_l83_83000

theorem circumradius_of_triangle (a b c : ℕ) (h₁ : a = 8) (h₂ : b = 6) (h₃ : c = 10) 
  (h₄ : a^2 + b^2 = c^2) : 
  (c : ℝ) / 2 = 5 := 
by {
  -- proof goes here
  sorry
}

end circumradius_of_triangle_l83_83000


namespace set_D_is_empty_l83_83341

theorem set_D_is_empty :
  {x : ℝ | x > 6 ∧ x < 1} = ∅ :=
by
  sorry

end set_D_is_empty_l83_83341


namespace number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l83_83759

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime 
  (n : ℕ) (h : n ≥ 2) : (∃ (a b : ℕ), a ≠ b ∧ is_prime (a^3 + 2) ∧ is_prime (b^3 + 2)) :=
by
  sorry

end number_of_possible_n_ge_2_satisfying_n_cubed_plus_two_is_prime_l83_83759


namespace lilyPadsFullCoverage_l83_83697

def lilyPadDoubling (t: ℕ) : ℕ :=
  t + 1

theorem lilyPadsFullCoverage (t: ℕ) (h: t = 47) : lilyPadDoubling t = 48 :=
by
  rw [h]
  unfold lilyPadDoubling
  rfl

end lilyPadsFullCoverage_l83_83697


namespace Janice_time_left_l83_83589

-- Define the conditions as variables and parameters
def homework_time := 30
def cleaning_time := homework_time / 2
def dog_walking_time := homework_time + 5
def trash_time := homework_time / 6
def total_time_before_movie := 2 * 60

-- Calculation of total time required for all tasks
def total_time_required_for_tasks : Nat :=
  homework_time + cleaning_time + dog_walking_time + trash_time

-- Time left before the movie starts after completing all tasks
def time_left_before_movie : Nat :=
  total_time_before_movie - total_time_required_for_tasks

-- The final statement to prove
theorem Janice_time_left : time_left_before_movie = 35 :=
  by
    -- This will execute automatically to verify the theorem
    sorry

end Janice_time_left_l83_83589


namespace find_a_l83_83654

-- Define the curve y = x^2 + x
def curve (x : ℝ) : ℝ := x^2 + x

-- Line equation ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, line a x y → y = x^2 + x) ∧
  (deriv curve 1 = 2 * 1 + 1) →
  (2 * 1 + 1 = -1 / a) →
  a = -1 / 3 :=
by
  sorry

end find_a_l83_83654


namespace length_QF_l83_83939

noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def directrix (x y : ℝ) : Prop := x = 1 -- Directrix of the given parabola

def point_on_directrix (P : ℝ × ℝ) : Prop := directrix P.1 P.2

def point_on_parabola (Q : ℝ × ℝ) : Prop := parabola Q.1 Q.2

def point_on_line_PF (P F Q : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m ≠ 0 ∧ (Q.2 = m * (Q.1 - F.1) + F.2) ∧ point_on_parabola Q

def vector_equality (F P Q : ℝ × ℝ) : Prop :=
  (4 * (Q.1 - F.1), 4 * (Q.2 - F.2)) = (P.1 - F.1, P.2 - F.2)

theorem length_QF 
  (P Q : ℝ × ℝ)
  (hPd : point_on_directrix P)
  (hPQ : point_on_line_PF P focus Q)
  (hVec : vector_equality focus P Q) : 
  dist Q focus = 3 :=
by
  sorry

end length_QF_l83_83939


namespace wicket_keeper_older_than_captain_l83_83758

variables (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ)

def x_older_than_captain (captain_age team_avg_age num_players remaining_avg_age : ℕ) : ℕ :=
  team_avg_age * num_players - remaining_avg_age * (num_players - 2) - 2 * captain_age

theorem wicket_keeper_older_than_captain 
  (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ) 
  (h1 : captain_age = 25) (h2 : team_avg_age = 23) (h3 : num_players = 11) (h4 : remaining_avg_age = 22) :
  x_older_than_captain captain_age team_avg_age num_players remaining_avg_age = 5 :=
by sorry

end wicket_keeper_older_than_captain_l83_83758


namespace peter_total_distance_is_six_l83_83447

def total_distance_covered (d : ℝ) :=
  let first_part_time := (2/3) * d / 4
  let second_part_time := (1/3) * d / 5
  (first_part_time + second_part_time) = 1.4

theorem peter_total_distance_is_six :
  ∃ d : ℝ, total_distance_covered d ∧ d = 6 := 
by
  -- Proof can be filled here
  sorry

end peter_total_distance_is_six_l83_83447


namespace sqrt_x2y_l83_83662

theorem sqrt_x2y (x y : ℝ) (h : x * y < 0) : Real.sqrt (x^2 * y) = -x * Real.sqrt y :=
sorry

end sqrt_x2y_l83_83662


namespace base6_addition_problem_l83_83444

theorem base6_addition_problem (X Y : ℕ) (h1 : Y + 3 = X) (h2 : X + 2 = 7) : X + Y = 7 := 
by
  sorry

end base6_addition_problem_l83_83444


namespace binary_add_sub_l83_83320

theorem binary_add_sub : 
  (1101 + 111 - 101 + 1001 - 11 : ℕ) = (10101 : ℕ) := by
  sorry

end binary_add_sub_l83_83320


namespace pipe_C_draining_rate_l83_83043

noncomputable def pipe_rate := 25

def tank_capacity := 2000
def pipe_A_rate := 200
def pipe_B_rate := 50
def pipe_C_duration_per_cycle := 2
def pipe_A_duration := 1
def pipe_B_duration := 2
def cycle_duration := pipe_A_duration + pipe_B_duration + pipe_C_duration_per_cycle
def total_time := 40
def number_of_cycles := total_time / cycle_duration
def water_filled_per_cycle := (pipe_A_rate * pipe_A_duration) + (pipe_B_rate * pipe_B_duration)
def total_water_filled := number_of_cycles * water_filled_per_cycle
def excess_water := total_water_filled - tank_capacity 
def pipe_C_rate := excess_water / (pipe_C_duration_per_cycle * number_of_cycles)

theorem pipe_C_draining_rate :
  pipe_C_rate = pipe_rate := by
  sorry

end pipe_C_draining_rate_l83_83043


namespace rotate_circle_sectors_l83_83276

theorem rotate_circle_sectors (n : ℕ) (h : n > 0) :
  (∀ i, i < n → ∃ θ : ℝ, θ < (π / (n^2 - n + 1))) →
  ∃ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧
  (∀ i : ℕ, i < n → (θ * i) % (2 * π) > (π / (n^2 - n + 1))) :=
sorry

end rotate_circle_sectors_l83_83276


namespace smallest_square_l83_83435

theorem smallest_square 
  (a b : ℕ) 
  (h1 : 15 * a + 16 * b = m ^ 2) 
  (h2 : 16 * a - 15 * b = n ^ 2)
  (hm : m > 0) 
  (hn : n > 0) : 
  min (15 * a + 16 * b) (16 * a - 15 * b) = 481 ^ 2 := 
sorry

end smallest_square_l83_83435


namespace intersection_complement_l83_83062

open Set

variable (U : Type) [TopologicalSpace U]

def A : Set ℝ := { x | x ≥ 0 }

def B : Set ℝ := { y | y ≤ 0 }

theorem intersection_complement (U : Type) [TopologicalSpace U] : 
  A ∩ (compl B) = { x | x > 0 } :=
by
  sorry

end intersection_complement_l83_83062


namespace range_of_expression_l83_83774

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 4) :
  1 ≤ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ∧ 4 * (x - 1/2)^2 + (y - 1)^2 + 4 * x * y ≤ 22 + 4 * Real.sqrt 5 :=
sorry

end range_of_expression_l83_83774


namespace boat_travel_distance_l83_83656

variable (v c d : ℝ) (c_eq_1 : c = 1)

theorem boat_travel_distance : 
  (∀ (v : ℝ), d = (v + c) * 4 → d = (v - c) * 6) → d = 24 := 
by
  intro H
  sorry

end boat_travel_distance_l83_83656


namespace product_of_roots_l83_83116

variable {k m x1 x2 : ℝ}

theorem product_of_roots (h1 : 4 * x1 ^ 2 - k * x1 - m = 0) (h2 : 4 * x2 ^ 2 - k * x2 - m = 0) (h3 : x1 ≠ x2) :
  x1 * x2 = -m / 4 :=
sorry

end product_of_roots_l83_83116


namespace min_value_x2_y2_z2_l83_83831

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  4 ≤ x^2 + y^2 + z^2 :=
sorry

end min_value_x2_y2_z2_l83_83831


namespace min_value_expression_l83_83399

theorem min_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+2)*(x+3)*(x+4)*(x+5) + 3033) ∧ y ≥ 3032 ∧ 
  (∀ z : ℝ, (z = (x+2)*(x+3)*(x+4)*(x+5) + 3033) → z ≥ 3032) := 
sorry

end min_value_expression_l83_83399


namespace rectangle_perimeter_l83_83460

theorem rectangle_perimeter (a b : ℚ) (ha : ¬ a.den = 1) (hb : ¬ b.den = 1) (hab : a ≠ b) (h : (a - 2) * (b - 2) = -7) : 2 * (a + b) = 20 :=
by
  sorry

end rectangle_perimeter_l83_83460


namespace evaluate_expression_at_zero_l83_83363

theorem evaluate_expression_at_zero :
  ∀ x : ℝ, (x ≠ -1) ∧ (x ≠ 3) →
  ( (3 * x^2 - 2 * x + 1) / ((x + 1) * (x - 3)) - (5 + 2 * x) / ((x + 1) * (x - 3)) ) = 2 :=
by
  sorry

end evaluate_expression_at_zero_l83_83363


namespace simplify_and_evaluate_expression_l83_83222

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2))

theorem simplify_and_evaluate_expression (x : ℝ) (hx : |x| = 2) (h_ne : x ≠ -2) :
  given_expression x = 3 :=
by
  sorry

end simplify_and_evaluate_expression_l83_83222


namespace max_min_values_l83_83100

noncomputable def max_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    max (x + y + z + w) 3
  else
    0

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    min (x + y + z + w) (-2 + 5 / 2 * Real.sqrt 2)
  else
    0

theorem max_min_values (x y z w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_nonneg_w : 0 ≤ w)
  (h_eqn : x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2) :
  (x + y + z + w ≤ 3) ∧ (x + y + z + w ≥ -2 + 5 / 2 * Real.sqrt 2) :=
by
  sorry

end max_min_values_l83_83100


namespace ab_greater_than_1_l83_83125

noncomputable def log10_abs (x : ℝ) : ℝ :=
  abs (Real.logb 10 x)

theorem ab_greater_than_1
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
  (hf : log10_abs a < log10_abs b) : a * b > 1 := by
  sorry

end ab_greater_than_1_l83_83125


namespace panda_bamboo_consumption_l83_83108

theorem panda_bamboo_consumption (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
  sorry

end panda_bamboo_consumption_l83_83108


namespace union_P_Q_l83_83133

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | x^2 - 2*x < 0}

theorem union_P_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end union_P_Q_l83_83133


namespace sin_240_deg_l83_83200

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l83_83200


namespace correct_option_B_l83_83392

theorem correct_option_B (a b : ℝ) : (-a^2 * b^3)^2 = a^4 * b^6 := 
  sorry

end correct_option_B_l83_83392


namespace count_implications_l83_83441

theorem count_implications (p q r : Prop) :
  ((p ∧ q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (¬p ∧ ¬q ∧ ¬r → ((q → p) → ¬r)) ∧ 
   (p ∧ ¬q ∧ r → ¬ ((q → p) → ¬r)) ∧ 
   (¬p ∧ q ∧ ¬r → ((q → p) → ¬r))) →
   (3 = 3) := sorry

end count_implications_l83_83441


namespace part_a_part_b_l83_83884

-- the conditions
variables (r R x : ℝ) (h_rltR : r < R)
variables (h_x : x = (R - r) / 2)
variables (h1 : 0 < x)
variables (h12_circles : ∀ i : ℕ, i ∈ Finset.range 12 → ∃ c_i : ℝ × ℝ, True)  -- Informal way to note 12 circles of radius x are placed

-- prove each part
theorem part_a (r R : ℝ) (h_rltR : r < R) : x = (R - r) / 2 :=
sorry

theorem part_b (r R : ℝ) (h_rltR : r < R) (h_x : x = (R - r) / 2) :
  (R / r) = (4 + Real.sqrt 6 - Real.sqrt 2) / (4 - Real.sqrt 6 + Real.sqrt 2) :=
sorry

end part_a_part_b_l83_83884


namespace speed_in_still_water_l83_83739

def upstream_speed : ℝ := 20
def downstream_speed : ℝ := 28

theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 24 := by
  sorry

end speed_in_still_water_l83_83739


namespace work_rate_proof_l83_83137

theorem work_rate_proof (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : C = 1 / 60) : 
  1 / (A + B + C) = 12 :=
by
  sorry

end work_rate_proof_l83_83137


namespace jason_initial_money_l83_83946

theorem jason_initial_money (M : ℝ) 
  (h1 : M - (M / 4 + 10 + (2 / 5 * (3 / 4 * M - 10) + 8)) = 130) : 
  M = 320 :=
by
  sorry

end jason_initial_money_l83_83946


namespace product_of_reals_condition_l83_83896

theorem product_of_reals_condition (x : ℝ) (h : x + 1/x = 3 * x) : 
  ∃ x1 x2 : ℝ, x1 + 1/x1 = 3 * x1 ∧ x2 + 1/x2 = 3 * x2 ∧ x1 * x2 = -1/2 := 
sorry

end product_of_reals_condition_l83_83896


namespace product_b1_b13_l83_83921

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) := ∀ n m k : ℕ, m > 0 → k > 0 → a (n + m) - a n = a (n + k) - a (n + k - m)

-- Conditions for the geometric sequence
def is_geometric_seq (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

-- Given conditions
def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (a 3 - (a 7 ^ 2) / 2 + a 11 = 0) ∧ (b 7 = a 7)

theorem product_b1_b13 
  (ha : is_arithmetic_seq a)
  (hb : is_geometric_seq b)
  (h : conditions a b) :
  b 1 * b 13 = 16 :=
sorry

end product_b1_b13_l83_83921


namespace annie_ride_miles_l83_83938

noncomputable def annie_ride_distance : ℕ := 14

theorem annie_ride_miles
  (mike_base_rate : ℝ := 2.5)
  (mike_per_mile_rate : ℝ := 0.25)
  (mike_miles : ℕ := 34)
  (annie_base_rate : ℝ := 2.5)
  (annie_bridge_toll : ℝ := 5.0)
  (annie_per_mile_rate : ℝ := 0.25)
  (annie_miles : ℕ := annie_ride_distance)
  (mike_cost : ℝ := mike_base_rate + mike_per_mile_rate * mike_miles)
  (annie_cost : ℝ := annie_base_rate + annie_bridge_toll + annie_per_mile_rate * annie_miles) :
  mike_cost = annie_cost → annie_miles = 14 := 
by
  sorry

end annie_ride_miles_l83_83938


namespace largest_angle_l83_83832

theorem largest_angle (y : ℝ) (h : 40 + 70 + y = 180) : y = 70 :=
by
  sorry

end largest_angle_l83_83832


namespace base7_addition_l83_83458

theorem base7_addition : (21 : ℕ) + 254 = 505 :=
by sorry

end base7_addition_l83_83458


namespace union_set_equiv_l83_83325

namespace ProofProblem

-- Define the sets A and B
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x^2 - x - 2 > 0 }

-- Define the union of A and B
def unionAB : Set ℝ := A ∪ B

-- State the proof problem
theorem union_set_equiv : unionAB = (Set.Iio (-1)) ∪ (Set.Ioi 1) := by
  sorry

end ProofProblem

end union_set_equiv_l83_83325


namespace future_age_relation_l83_83240

-- Conditions
def son_present_age : ℕ := 8
def father_present_age : ℕ := 4 * son_present_age

-- Theorem statement
theorem future_age_relation : ∃ x : ℕ, 32 + x = 3 * (8 + x) ↔ x = 4 :=
by {
  sorry
}

end future_age_relation_l83_83240


namespace equipment_unit_prices_purchasing_scenarios_l83_83430

theorem equipment_unit_prices
  (x : ℝ)
  (price_A_eq_price_B_minus_10 : ∀ y, ∃ z, z = y + 10)
  (eq_purchases_equal_cost_A : ∀ n : ℕ, 300 / x = n)
  (eq_purchases_equal_cost_B : ∀ n : ℕ, 360 / (x + 10) = n) :
  x = 50 ∧ (x + 10) = 60 :=
by
  sorry

theorem purchasing_scenarios
  (m n : ℕ)
  (price_A : ℝ := 50)
  (price_B : ℝ := 60)
  (budget : ℝ := 1000)
  (purchase_eq_budget : 50 * m + 60 * n = 1000)
  (pos_integers : m > 0 ∧ n > 0) :
  (m = 14 ∧ n = 5) ∨ (m = 8 ∧ n = 10) ∨ (m = 2 ∧ n = 15) :=
by
  sorry

end equipment_unit_prices_purchasing_scenarios_l83_83430


namespace sample_size_correct_l83_83022

-- Definitions derived from conditions in a)
def total_employees : ℕ := 120
def male_employees : ℕ := 90
def sampled_male_employees : ℕ := 18

-- Theorem stating the mathematically equivalent proof problem
theorem sample_size_correct : 
  ∃ (sample_size : ℕ), sample_size = (total_employees * (sampled_male_employees / male_employees)) :=
sorry

end sample_size_correct_l83_83022


namespace find_t_l83_83239

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l83_83239


namespace smallest_natural_number_l83_83994

theorem smallest_natural_number (n : ℕ) (h : 2006 ^ 1003 < n ^ 2006) : n ≥ 45 := 
by {
    sorry
}

end smallest_natural_number_l83_83994


namespace complement_union_eq_l83_83966

open Set

-- Define the universe and sets P and Q
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 3, 5}
def Q : Set ℕ := {1, 2, 4}

-- State the theorem
theorem complement_union_eq :
  ((U \ P) ∪ Q) = {1, 2, 4, 6} := by
  sorry

end complement_union_eq_l83_83966


namespace quadratic_root_range_l83_83930

theorem quadratic_root_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, (x₁ > 0) ∧ (x₂ < 0) ∧ (x₁^2 + 2 * (a - 1) * x₁ + 2 * a + 6 = 0) ∧ (x₂^2 + 2 * (a - 1) * x₂ + 2 * a + 6 = 0)) → a < -3 :=
by
  sorry

end quadratic_root_range_l83_83930


namespace complementary_angle_of_60_l83_83146

theorem complementary_angle_of_60 (a : ℝ) : 
  (∀ (a b : ℝ), a + b = 180 → a = 60 → b = 120) := 
by
  sorry

end complementary_angle_of_60_l83_83146


namespace triangle_area_l83_83418

theorem triangle_area :
  let a := 4
  let c := 5
  let b := Real.sqrt (c^2 - a^2)
  (1 / 2) * a * b = 6 :=
by sorry

end triangle_area_l83_83418


namespace polynomial_simplification_l83_83197

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end polynomial_simplification_l83_83197


namespace find_f1_plus_g1_l83_83834

-- Definition of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Definition of g being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

-- Statement of the proof problem
theorem find_f1_plus_g1 
  (f g : ℝ → ℝ) 
  (hf : is_even_function f) 
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 2 :=
sorry

end find_f1_plus_g1_l83_83834


namespace real_solution_of_equation_l83_83535

theorem real_solution_of_equation :
  ∀ x : ℝ, (x ≠ 5) → (x ≠ 3) →
  ((x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3)) 
  / ((x - 5) * (x - 3) * (x - 5)) = 1 ↔ x = 1 :=
by sorry

end real_solution_of_equation_l83_83535


namespace greatest_possible_bxa_l83_83515

-- Define the property of the number being divisible by 35
def div_by_35 (n : ℕ) : Prop :=
  n % 35 = 0

-- Define the main proof problem
theorem greatest_possible_bxa :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ div_by_35 (10 * a + b) ∧ (∀ (a' b' : ℕ), a' < 10 → b' < 10 → div_by_35 (10 * a' + b') → b * a ≥ b' * a') :=
sorry

end greatest_possible_bxa_l83_83515


namespace students_passing_in_sixth_year_l83_83638

def numStudentsPassed (year : ℕ) : ℕ :=
 if year = 1 then 200 else 
 if year = 2 then 300 else 
 if year = 3 then 390 else 
 if year = 4 then 565 else 
 if year = 5 then 643 else 
 if year = 6 then 780 else 0

theorem students_passing_in_sixth_year : numStudentsPassed 6 = 780 := by
  sorry

end students_passing_in_sixth_year_l83_83638


namespace least_common_multiple_1260_980_l83_83439

def LCM (a b : ℕ) : ℕ :=
  a * b / Nat.gcd a b

theorem least_common_multiple_1260_980 : LCM 1260 980 = 8820 := by
  sorry

end least_common_multiple_1260_980_l83_83439


namespace sec_240_eq_neg2_l83_83619

noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

theorem sec_240_eq_neg2 : sec 240 = -2 := by
  -- Proof goes here
  sorry

end sec_240_eq_neg2_l83_83619


namespace cotton_equals_iron_l83_83961

theorem cotton_equals_iron (cotton_weight : ℝ) (iron_weight : ℝ)
  (h_cotton : cotton_weight = 1)
  (h_iron : iron_weight = 4) :
  (4 / 5) * cotton_weight = (1 / 5) * iron_weight :=
by
  rw [h_cotton, h_iron]
  simp
  sorry

end cotton_equals_iron_l83_83961


namespace jason_messages_l83_83161

theorem jason_messages :
  ∃ M : ℕ, (M + M / 2 + 150) / 5 = 96 ∧ M = 220 := by
  sorry

end jason_messages_l83_83161


namespace range_of_k_l83_83048

-- Given conditions
variables {k : ℝ} (h : ∃ (x y : ℝ), x^2 + k * y^2 = 2)

-- Theorem statement
theorem range_of_k : 0 < k ∧ k < 1 :=
by
  sorry

end range_of_k_l83_83048


namespace find_number_l83_83949

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 8) : x = 4 :=
by
  sorry

end find_number_l83_83949


namespace MaireadRan40Miles_l83_83095

def MaireadRanMiles (R : ℝ) (W : ℝ) (J : ℝ) : Prop :=
  W = (3 / 5) * R ∧ J = 3 * R ∧ R + W + J = 184

theorem MaireadRan40Miles : ∃ R W J, MaireadRanMiles R W J ∧ R = 40 :=
by sorry

end MaireadRan40Miles_l83_83095


namespace drive_photos_storage_l83_83451

theorem drive_photos_storage (photo_size: ℝ) (num_photos_with_videos: ℕ) (photo_storage_with_videos: ℝ) (video_size: ℝ) (num_videos_with_photos: ℕ) : 
  num_photos_with_videos * photo_size + num_videos_with_photos * video_size = 3000 → 
  (3000 / photo_size) = 2000 :=
by
  sorry

end drive_photos_storage_l83_83451


namespace sum_of_interior_angles_of_regular_polygon_l83_83871

theorem sum_of_interior_angles_of_regular_polygon :
  (∀ (n : ℕ), (n ≠ 0) ∧ ((360 / 45 = n) → (180 * (n - 2) = 1080))) := by sorry

end sum_of_interior_angles_of_regular_polygon_l83_83871


namespace square_area_l83_83658

theorem square_area (y : ℝ) (x : ℝ → ℝ) : 
    (∀ x, y = x ^ 2 + 4 * x + 3) → (y = 7) → 
    ∃ area : ℝ, area = 32 := 
by
  intro h₁ h₂ 
  -- Proof steps would go here
  sorry

end square_area_l83_83658


namespace mutually_exclusive_event_l83_83652

def shooting_twice : Type := 
  { hit_first : Bool // hit_first = true ∨ hit_first = false }

def hitting_at_least_once (shoots : shooting_twice) : Prop :=
  shoots.1 ∨ (¬shoots.1 ∧ true)

def missing_both_times (shoots : shooting_twice) : Prop :=
  ¬shoots.1 ∧ (¬true ∨ true)

def mutually_exclusive (A : Prop) (B : Prop) : Prop :=
  A ∨ B → ¬ (A ∧ B)

theorem mutually_exclusive_event :
  ∀ shoots : shooting_twice, 
  mutually_exclusive (hitting_at_least_once shoots) (missing_both_times shoots) :=
by
  intro shoots
  unfold mutually_exclusive
  sorry

end mutually_exclusive_event_l83_83652


namespace find_pairs_l83_83322

def isDivisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def satisfiesConditions (a b : ℕ) : Prop :=
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  isPrime (a + 6 * b + 2)) ∨
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  ¬ isPrime (a + 6 * b + 2))

theorem find_pairs (a b : ℕ) :
  (a = 5 ∧ b = 1) ∨ 
  (a = 17 ∧ b = 7) → 
  satisfiesConditions a b :=
by
  -- Proof to be completed
  sorry

end find_pairs_l83_83322


namespace quadratic_expression_value_l83_83253

theorem quadratic_expression_value
  (x : ℝ)
  (h : x^2 + x - 2 = 0)
: x^3 + 2*x^2 - x + 2021 = 2023 :=
sorry

end quadratic_expression_value_l83_83253


namespace candies_initial_count_l83_83160

theorem candies_initial_count (x : ℕ) (h : (x - 29) / 13 = 15) : x = 224 :=
sorry

end candies_initial_count_l83_83160


namespace quadratic_function_opens_downwards_l83_83311

theorem quadratic_function_opens_downwards (m : ℝ) (h₁ : m - 1 < 0) (h₂ : m^2 + 1 = 2) : m = -1 :=
by {
  -- Proof would go here.
  sorry
}

end quadratic_function_opens_downwards_l83_83311


namespace units_digit_of_product_l83_83488

def is_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem units_digit_of_product : 
  is_units_digit (6 * 8 * 9 * 10 * 12) 0 := 
by
  sorry

end units_digit_of_product_l83_83488


namespace other_acute_angle_in_right_triangle_l83_83179

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end other_acute_angle_in_right_triangle_l83_83179


namespace A_eq_D_l83_83026

def A := {θ : ℝ | 0 < θ ∧ θ < 90}
def D := {θ : ℝ | 0 < θ ∧ θ < 90}

theorem A_eq_D : A = D :=
by
  sorry

end A_eq_D_l83_83026


namespace count_possible_integer_values_l83_83496

theorem count_possible_integer_values :
  ∃ n : ℕ, (∀ x : ℤ, (25 < x ∧ x < 55) ↔ (26 ≤ x ∧ x ≤ 54)) ∧ n = 29 := by
  sorry

end count_possible_integer_values_l83_83496


namespace arith_seq_a1_eq_15_l83_83687

variable {a : ℕ → ℤ} (a_seq : ∀ n, a n = a 1 + (n-1) * d)
variable {a_4 : ℤ} (h4 : a 4 = 9)
variable {a_8 : ℤ} (h8 : a 8 = -a 9)

theorem arith_seq_a1_eq_15 (a_seq : ∀ n, a n = a 1 + (n-1) * d) (h4 : a 4 = 9) (h8 : a 8 = -a 9) : a 1 = 15 :=
by
  -- Proof should go here
  sorry

end arith_seq_a1_eq_15_l83_83687


namespace hyperbola_aux_lines_l83_83591

theorem hyperbola_aux_lines (a : ℝ) (h_a_positive : a > 0)
  (h_hyperbola_eqn : ∀ x y, (x^2 / a^2) - (y^2 / 16) = 1)
  (h_asymptotes : ∀ x y, y = 4/3 * x ∨ y = -4/3 * x) : 
  ∀ x, (x = 9/5 ∨ x = -9/5) := sorry

end hyperbola_aux_lines_l83_83591


namespace remaining_payment_l83_83552

theorem remaining_payment (part_payment total_cost : ℝ) (percent_payment : ℝ) 
  (h1 : part_payment = 650) 
  (h2 : percent_payment = 15 / 100) 
  (h3 : part_payment = percent_payment * total_cost) : 
  total_cost - part_payment = 3683.33 := 
by 
  sorry

end remaining_payment_l83_83552


namespace discount_profit_percentage_l83_83318

theorem discount_profit_percentage (CP : ℝ) (P_no_discount : ℝ) (D : ℝ) (profit_with_discount : ℝ) (SP_no_discount : ℝ) (SP_discount : ℝ) :
  P_no_discount = 50 ∧ D = 4 ∧ SP_no_discount = CP + 0.5 * CP ∧ SP_discount = SP_no_discount - (D / 100) * SP_no_discount ∧ profit_with_discount = SP_discount - CP →
  (profit_with_discount / CP) * 100 = 44 :=
by sorry

end discount_profit_percentage_l83_83318


namespace portion_apples_weight_fraction_l83_83729

-- Given conditions
def total_apples : ℕ := 28
def total_weight_kg : ℕ := 3
def number_of_portions : ℕ := 7

-- Proof statement
theorem portion_apples_weight_fraction :
  (1 / number_of_portions = 1 / 7) ∧ (3 / number_of_portions = 3 / 7) :=
by
  -- Proof goes here
  sorry

end portion_apples_weight_fraction_l83_83729


namespace sin_cos_product_neg_l83_83285

theorem sin_cos_product_neg (α : ℝ) (h : Real.tan α < 0) : Real.sin α * Real.cos α < 0 :=
sorry

end sin_cos_product_neg_l83_83285


namespace total_points_l83_83878

def points_earned (goblins orcs dragons : ℕ): ℕ :=
  goblins * 3 + orcs * 5 + dragons * 10

theorem total_points :
  points_earned 10 7 1 = 75 :=
by
  sorry

end total_points_l83_83878


namespace f_analytical_expression_g_value_l83_83035

noncomputable def f (ω x : ℝ) : ℝ := (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi / 2)

noncomputable def g (ω x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem f_analytical_expression (x : ℝ) (hω : ω = 2 ∧ ω > 0) : 
  f 2 x = Real.sin (2 * x - Real.pi / 3) :=
sorry

theorem g_value (α : ℝ) (hω : ω = 2 ∧ ω > 0) (h : g 2 (α / 2) = 4/5) : 
  g 2 (-α) = -7/25 :=
sorry

end f_analytical_expression_g_value_l83_83035


namespace find_distance_between_posters_and_wall_l83_83770

-- Definitions for given conditions
def poster_width : ℝ := 29.05
def num_posters : ℕ := 8
def wall_width : ℝ := 394.4

-- The proof statement: find the distance 'd' between posters and ends
theorem find_distance_between_posters_and_wall :
  ∃ d : ℝ, (wall_width - num_posters * poster_width) / (num_posters + 1) = d ∧ d = 18 := 
by {
  -- The proof would involve showing that this specific d meets the constraints.
  sorry
}

end find_distance_between_posters_and_wall_l83_83770


namespace smallest_n_in_range_l83_83059

theorem smallest_n_in_range (n : ℤ) (h1 : 4 ≤ n ∧ n ≤ 12) (h2 : n ≡ 2 [ZMOD 9]) : n = 11 :=
sorry

end smallest_n_in_range_l83_83059


namespace real_roots_of_quad_eq_l83_83077

theorem real_roots_of_quad_eq (p q a : ℝ) (h : p^2 - 4 * q > 0) : 
  (2 * a - p)^2 + 3 * (p^2 - 4 * q) > 0 := 
by
  sorry

end real_roots_of_quad_eq_l83_83077


namespace fraction_is_integer_l83_83166

theorem fraction_is_integer (b t : ℤ) (hb : b ≠ 1) :
  ∃ (k : ℤ), (t^5 - 5 * b + 4) = k * (b^2 - 2 * b + 1) :=
by 
  sorry

end fraction_is_integer_l83_83166


namespace condition_necessary_but_not_sufficient_l83_83101

variable (a b : ℝ)

theorem condition_necessary_but_not_sufficient (h : a ≠ 1 ∨ b ≠ 2) : (a + b ≠ 3) ∧ ¬(a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  --Proof will go here
  sorry

end condition_necessary_but_not_sufficient_l83_83101


namespace linear_function_solution_l83_83349

theorem linear_function_solution (f : ℝ → ℝ) (h1 : ∀ x, f (f x) = 16 * x - 15) :
  (∀ x, f x = 4 * x - 3) ∨ (∀ x, f x = -4 * x + 5) :=
sorry

end linear_function_solution_l83_83349


namespace cakes_served_yesterday_l83_83967

theorem cakes_served_yesterday (lunch_cakes dinner_cakes total_cakes served_yesterday : ℕ)
  (h1 : lunch_cakes = 5)
  (h2 : dinner_cakes = 6)
  (h3 : total_cakes = 14)
  (h4 : total_cakes = lunch_cakes + dinner_cakes + served_yesterday) :
  served_yesterday = 3 := 
by 
  sorry

end cakes_served_yesterday_l83_83967


namespace smallest_x_inequality_l83_83237

theorem smallest_x_inequality : ∃ x : ℝ, (x^2 - 8 * x + 15 ≤ 0) ∧ (∀ y : ℝ, (y^2 - 8 * y + 15 ≤ 0) → (3 ≤ y)) ∧ x = 3 := 
sorry

end smallest_x_inequality_l83_83237


namespace rate_of_interest_l83_83753

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

theorem rate_of_interest :
  ∃ R : ℝ, simple_interest 8925 R 5 = 4016.25 ∧ R = 9 := 
by
  use 9
  simp [simple_interest]
  norm_num
  sorry

end rate_of_interest_l83_83753


namespace hypotenuse_length_l83_83622

theorem hypotenuse_length (x y : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = 1350 * Real.pi) 
  (h2 : V2 = 2430 * Real.pi) 
  (h3 : (1/3) * Real.pi * y^2 * x = V1) 
  (h4 : (1/3) * Real.pi * x^2 * y = V2) 
  : Real.sqrt (x^2 + y^2) = Real.sqrt 954 :=
sorry

end hypotenuse_length_l83_83622


namespace inequality_positive_l83_83717

theorem inequality_positive (x : ℝ) : (1 / 3) * x - x > 0 ↔ (-2 / 3) * x > 0 := 
  sorry

end inequality_positive_l83_83717


namespace min_segment_length_l83_83688

theorem min_segment_length 
  (angle : ℝ) (P : ℝ × ℝ)
  (dist_x : ℝ) (dist_y : ℝ) 
  (hx : P.1 ≤ dist_x ∧ P.2 = dist_y)
  (hy : P.2 ≤ dist_y ∧ P.1 = dist_x)
  (right_angle : angle = 90) 
  : ∃ (d : ℝ), d = 10 :=
by
  sorry

end min_segment_length_l83_83688


namespace how_many_green_towels_l83_83199

-- Define the conditions
def initial_white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34
def towels_left_after_giving : ℕ := 22

-- Define the statement to prove
theorem how_many_green_towels (G : ℕ) (initial_white : ℕ) (given : ℕ) (left_after : ℕ) :
  initial_white = initial_white_towels →
  given = towels_given_to_mother →
  left_after = towels_left_after_giving →
  (G + initial_white) - given = left_after →
  G = 35 :=
by
  intros
  sorry

end how_many_green_towels_l83_83199


namespace houses_with_dogs_l83_83359

theorem houses_with_dogs (C B Total : ℕ) (hC : C = 30) (hB : B = 10) (hTotal : Total = 60) :
  ∃ D, D = 40 :=
by
  -- The overall proof would go here
  sorry

end houses_with_dogs_l83_83359


namespace non_negative_integer_solutions_of_inequality_system_l83_83466

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l83_83466


namespace solution_set_of_inequality_l83_83579

theorem solution_set_of_inequality (x : ℝ) : (1 / x ≤ x) ↔ (-1 ≤ x ∧ x < 0) ∨ (x ≥ 1) := sorry

end solution_set_of_inequality_l83_83579


namespace hamburger_count_l83_83290

-- Define the number of condiments and their possible combinations
def condiment_combinations : ℕ := 2 ^ 10

-- Define the number of choices for meat patties
def meat_patties_choices : ℕ := 4

-- Define the total count of different hamburgers
def total_hamburgers : ℕ := condiment_combinations * meat_patties_choices

-- The theorem statement proving the total number of different hamburgers
theorem hamburger_count : total_hamburgers = 4096 := by
  sorry

end hamburger_count_l83_83290


namespace circumscribed_circle_radius_l83_83071

noncomputable def radius_of_circumscribed_circle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
  let R := a / (2 * Real.sin A)
  R

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (hb : b = 4) (hc : c = 2) (hA : A = Real.pi / 3) :
  radius_of_circumscribed_circle b c A = 2 := by
  sorry

end circumscribed_circle_radius_l83_83071


namespace pages_read_on_Sunday_l83_83020

def total_pages : ℕ := 93
def pages_read_on_Saturday : ℕ := 30
def pages_remaining_after_Sunday : ℕ := 43

theorem pages_read_on_Sunday : total_pages - pages_read_on_Saturday - pages_remaining_after_Sunday = 20 := by
  sorry

end pages_read_on_Sunday_l83_83020


namespace sum_of_powers_of_two_l83_83422

theorem sum_of_powers_of_two (n : ℕ) (h : 1 ≤ n ∧ n ≤ 511) : 
  ∃ (S : Finset ℕ), S ⊆ ({2^8, 2^7, 2^6, 2^5, 2^4, 2^3, 2^2, 2^1, 2^0} : Finset ℕ) ∧ 
  S.sum id = n :=
by
  sorry

end sum_of_powers_of_two_l83_83422


namespace change_in_expression_l83_83530

theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let original_expr := x^2 - 5 * x + 2
  let new_x := x + b
  let new_expr := (new_x)^2 - 5 * (new_x) + 2
  new_expr - original_expr = 2 * b * x + b^2 - 5 * b :=
by
  sorry

end change_in_expression_l83_83530


namespace number_of_groups_of_bananas_l83_83335

theorem number_of_groups_of_bananas (total_bananas : ℕ) (bananas_per_group : ℕ) (H_total_bananas : total_bananas = 290) (H_bananas_per_group : bananas_per_group = 145) :
    (total_bananas / bananas_per_group) = 2 :=
by {
  sorry
}

end number_of_groups_of_bananas_l83_83335


namespace value_of_m_l83_83551

-- Defining the quadratic equation condition
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * x + m^2 - 4

-- Defining the condition where the constant term in the quadratic equation is 0
def constant_term_zero (m : ℝ) : Prop := m^2 - 4 = 0

-- Stating the proof problem: given the conditions, prove that m = -2
theorem value_of_m (m : ℝ) (h1 : constant_term_zero m) (h2 : m ≠ 2) : m = -2 :=
by {
  sorry -- Proof to be developed
}

end value_of_m_l83_83551


namespace find_Y_value_l83_83260

-- Define the conditions
def P : ℕ := 4020 / 4
def Q : ℕ := P * 2
def Y : ℤ := P - Q

-- State the theorem
theorem find_Y_value : Y = -1005 := by
  -- Proof goes here
  sorry

end find_Y_value_l83_83260


namespace find_a_l83_83514

theorem find_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^3 + 3 * x^2 + 2)
  (hf' : ∀ x, f' x = 3 * a * x^2 + 6 * x) 
  (h : f' (-1) = 4) : 
  a = (10 : ℝ) / 3 := 
sorry

end find_a_l83_83514


namespace Q_transform_l83_83968

def rotate_180_clockwise (p q : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  let (qx, qy) := q
  (2 * px - qx, 2 * py - qy)

def reflect_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (px, py) := p
  (py, px)

def Q := (8, -11) -- from the reverse transformations

theorem Q_transform (c d : ℝ) :
  (reflect_y_equals_x (rotate_180_clockwise (2, -3) (c, d)) = (5, -4)) → (d - c = -19) :=
by sorry

end Q_transform_l83_83968


namespace fewest_toothpicks_proof_l83_83964

noncomputable def fewest_toothpicks_to_remove (total_toothpicks : ℕ) (additional_row_and_column : ℕ) (triangles : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) (max_destroyed_per_toothpick : ℕ) (horizontal_toothpicks : ℕ) : ℕ :=
  horizontal_toothpicks

theorem fewest_toothpicks_proof 
  (total_toothpicks : ℕ := 40) 
  (additional_row_and_column : ℕ := 1) 
  (triangles : ℕ := 35) 
  (upward_triangles : ℕ := 15) 
  (downward_triangles : ℕ := 10)
  (max_destroyed_per_toothpick : ℕ := 1)
  (horizontal_toothpicks : ℕ := 15) :
  fewest_toothpicks_to_remove total_toothpicks additional_row_and_column triangles upward_triangles downward_triangles max_destroyed_per_toothpick horizontal_toothpicks = 15 := 
by 
  sorry

end fewest_toothpicks_proof_l83_83964


namespace rate_per_kg_for_apples_l83_83190

theorem rate_per_kg_for_apples (A : ℝ) :
  (8 * A + 9 * 45 = 965) → (A = 70) :=
by
  sorry

end rate_per_kg_for_apples_l83_83190


namespace sum_of_squared_residuals_l83_83103

theorem sum_of_squared_residuals (S : ℝ) (r : ℝ) (hS : S = 100) (hr : r = 0.818) : 
    S * (1 - r^2) = 33.0876 :=
by
  rw [hS, hr]
  sorry

end sum_of_squared_residuals_l83_83103


namespace total_amount_shared_l83_83140

theorem total_amount_shared (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) 
  (portion_a : ℕ) (portion_b : ℕ) (portion_c : ℕ)
  (h_ratio : ratio_a = 3 ∧ ratio_b = 4 ∧ ratio_c = 9)
  (h_portion_a : portion_a = 30)
  (h_portion_b : portion_b = 2 * portion_a + 10)
  (h_portion_c : portion_c = (ratio_c / ratio_a) * portion_a) :
  portion_a + portion_b + portion_c = 190 :=
by sorry

end total_amount_shared_l83_83140


namespace breakfast_plate_contains_2_eggs_l83_83705

-- Define the conditions
def breakfast_plate := Nat
def num_customers := 14
def num_bacon_strips := 56

-- Define the bacon strips per plate
def bacon_strips_per_plate (num_bacon_strips num_customers : Nat) : Nat :=
  num_bacon_strips / num_customers

-- Define the number of eggs per plate given twice as many bacon strips as eggs
def eggs_per_plate (bacon_strips_per_plate : Nat) : Nat :=
  bacon_strips_per_plate / 2

-- The main theorem we need to prove
theorem breakfast_plate_contains_2_eggs :
  eggs_per_plate (bacon_strips_per_plate 56 14) = 2 :=
by
  sorry

end breakfast_plate_contains_2_eggs_l83_83705


namespace vertex_y_coord_of_h_l83_83671

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 1
def h (x : ℝ) : ℝ := f x - g x

theorem vertex_y_coord_of_h : h (-1 / 10) = 79 / 20 := by
  sorry

end vertex_y_coord_of_h_l83_83671


namespace solve_problem_for_m_n_l83_83867

theorem solve_problem_for_m_n (m n : ℕ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m * (n + m) = n * (n - m)) :
  ((∃ h : ℕ, m = (2 * h + 1) * h ∧ n = (2 * h + 1) * (h + 1)) ∨ 
   (∃ h : ℕ, h > 0 ∧ m = 2 * h * (4 * h^2 - 1) ∧ n = 2 * h * (4 * h^2 + 1))) := 
sorry

end solve_problem_for_m_n_l83_83867


namespace obrien_hats_after_loss_l83_83013

noncomputable def hats_simpson : ℕ := 15

noncomputable def initial_hats_obrien : ℕ := 2 * hats_simpson + 5

theorem obrien_hats_after_loss : initial_hats_obrien - 1 = 34 :=
by
  sorry

end obrien_hats_after_loss_l83_83013


namespace folded_quadrilateral_has_perpendicular_diagonals_l83_83089

-- Define a quadrilateral and its properties
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

structure Point :=
(x y : ℝ)

-- Define the diagonals within a quadrilateral
def diagonal1 (q : Quadrilateral) : ℝ × ℝ := (q.A.1 - q.C.1, q.A.2 - q.C.2)
def diagonal2 (q : Quadrilateral) : ℝ × ℝ := (q.B.1 - q.D.1, q.B.2 - q.D.2)

-- Define dot product to check perpendicularity
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition when folding quadrilateral vertices to a common point ensures no gaps or overlaps
def folding_condition (q : Quadrilateral) (P : Point) : Prop :=
sorry -- Detailed folding condition logic here if needed

-- The statement we need to prove
theorem folded_quadrilateral_has_perpendicular_diagonals (q : Quadrilateral) (P : Point)
    (h_folding : folding_condition q P)
    : dot_product (diagonal1 q) (diagonal2 q) = 0 :=
sorry

end folded_quadrilateral_has_perpendicular_diagonals_l83_83089


namespace find_functions_l83_83249

open Function

theorem find_functions (f g : ℚ → ℚ) :
  (∀ x y : ℚ, f (g x - g y) = f (g x) - y) →
  (∀ x y : ℚ, g (f x - f y) = g (f x) - y) →
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
by
  sorry

end find_functions_l83_83249


namespace part_1_part_2_l83_83196

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end part_1_part_2_l83_83196


namespace yi_catches_jia_on_DA_l83_83555

def square_side_length : ℝ := 90
def jia_speed : ℝ := 65
def yi_speed : ℝ := 72
def jia_start : ℝ := 0
def yi_start : ℝ := 90

theorem yi_catches_jia_on_DA :
  let square_perimeter := 4 * square_side_length
  let initial_gap := 3 * square_side_length
  let relative_speed := yi_speed - jia_speed
  let time_to_catch := initial_gap / relative_speed
  let distance_travelled_by_yi := yi_speed * time_to_catch
  let number_of_laps := distance_travelled_by_yi / square_perimeter
  let additional_distance := distance_travelled_by_yi % square_perimeter
  additional_distance = 0 →
  square_side_length * (number_of_laps % 4) = 0 ∨ number_of_laps % 4 = 3 :=
by
  -- We only provide the statement, the proof is omitted.
  sorry

end yi_catches_jia_on_DA_l83_83555


namespace chocolate_cost_is_correct_l83_83069

def total_spent : ℕ := 13
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := total_spent - candy_bar_cost

theorem chocolate_cost_is_correct : chocolate_cost = 6 :=
by
  sorry

end chocolate_cost_is_correct_l83_83069


namespace possible_values_of_n_l83_83942

theorem possible_values_of_n (n : ℕ) (h_pos : 0 < n) (h_prime_n : Nat.Prime n) (h_prime_double_sub1 : Nat.Prime (2 * n - 1)) (h_prime_quad_sub1 : Nat.Prime (4 * n - 1)) :
  n = 2 ∨ n = 3 :=
by
  sorry

end possible_values_of_n_l83_83942


namespace rectangle_area_l83_83074

theorem rectangle_area (w l: ℝ) (h1: l = 2 * w) (h2: 2 * l + 2 * w = 4) : l * w = 8 / 9 := by
  sorry

end rectangle_area_l83_83074


namespace arrange_books_correct_l83_83691

def math_books : Nat := 4
def history_books : Nat := 4

def arrangements (m h : Nat) : Nat := sorry

theorem arrange_books_correct :
  arrangements math_books history_books = 576 := sorry

end arrange_books_correct_l83_83691


namespace angle_C_in_triangle_l83_83684

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 110) (ht : A + B + C = 180) : C = 70 :=
by
  -- proof steps go here
  sorry

end angle_C_in_triangle_l83_83684


namespace cube_skew_lines_l83_83870

theorem cube_skew_lines (cube : Prop) (diagonal : Prop) (edges : Prop) :
  ( ∃ n : ℕ, n = 6 ) :=
by
  sorry

end cube_skew_lines_l83_83870


namespace probability_of_U_l83_83590

def pinyin : List Char := ['S', 'H', 'U', 'X', 'U', 'E']
def total_letters : Nat := 6
def u_count : Nat := 2

theorem probability_of_U :
  ((u_count : ℚ) / (total_letters : ℚ)) = (1 / 3) :=
by
  sorry

end probability_of_U_l83_83590


namespace smallest_part_proportional_division_l83_83012

theorem smallest_part_proportional_division (a b c d total : ℕ) (h : a + b + c + d = total) (sum_equals_360 : 360 = total * 15):
  min (4 * 15) (min (5 * 15) (min (7 * 15) (8 * 15))) = 60 :=
by
  -- Defining the proportions and overall total
  let a := 5
  let b := 7
  let c := 4
  let d := 8
  let total_parts := a + b + c + d

  -- Given that the division is proportional
  let part_value := 360 / total_parts

  -- Assert that the smallest part is equal to the smallest proportion times the value of one part
  let smallest_part := c * part_value
  trivial

end smallest_part_proportional_division_l83_83012


namespace martin_class_number_l83_83595

theorem martin_class_number (b : ℕ) (h1 : 100 < b) (h2 : b < 200) 
  (h3 : b % 3 = 2) (h4 : b % 4 = 1) (h5 : b % 5 = 1) : 
  b = 101 ∨ b = 161 := 
by
  sorry

end martin_class_number_l83_83595


namespace sum_digits_3times_l83_83233

-- Define the sum of digits function
noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the 2006-th power of 2
noncomputable def power_2006 := 2 ^ 2006

-- State the theorem
theorem sum_digits_3times (n : ℕ) (h : n = power_2006) : 
  digit_sum (digit_sum (digit_sum n)) = 4 := by
  -- Add the proof steps here
  sorry

end sum_digits_3times_l83_83233


namespace geometric_seq_sum_identity_l83_83494

noncomputable def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

theorem geometric_seq_sum_identity (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (hgeom : is_geometric_seq a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end geometric_seq_sum_identity_l83_83494


namespace atLeastOneNotLessThanTwo_l83_83470

open Real

theorem atLeastOneNotLessThanTwo (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → False := 
by
  sorry

end atLeastOneNotLessThanTwo_l83_83470


namespace union_of_complements_l83_83461

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x | x^2 + 4 = 5 * x}
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem union_of_complements :
  complement_U A ∪ complement_U B = {0, 2, 3, 4, 5} := by
sorry

end union_of_complements_l83_83461


namespace find_x_l83_83093

open Real

noncomputable def satisfies_equation (x : ℝ) : Prop :=
  log (x - 1) / log 3 + log (x^2 - 1) / log (sqrt 3) + log (x - 1) / log (1 / 3) = 3

theorem find_x : ∃ x : ℝ, 1 < x ∧ satisfies_equation x ∧ x = sqrt (1 + 3 * sqrt 3) := by
  sorry

end find_x_l83_83093


namespace prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l83_83587

theorem prime_of_form_4k_plus_1_as_sum_of_two_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 4 * k + 1) :
  ∃ a b : ℤ, p = a^2 + b^2 :=
sorry

theorem prime_of_form_8k_plus_3_as_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 8 * k + 3) :
  ∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
sorry

end prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l83_83587


namespace container_volume_ratio_l83_83024

theorem container_volume_ratio (V1 V2 : ℚ)
  (h1 : (3 / 5) * V1 = (2 / 3) * V2) :
  V1 / V2 = 10 / 9 :=
by sorry

end container_volume_ratio_l83_83024


namespace quadrilateral_diagonal_length_l83_83811

theorem quadrilateral_diagonal_length (d : ℝ) 
  (h_offsets : true) 
  (area_quadrilateral : 195 = ((1 / 2) * d * 9) + ((1 / 2) * d * 6)) : 
  d = 26 :=
by 
  sorry

end quadrilateral_diagonal_length_l83_83811


namespace find_n_l83_83544

theorem find_n (n : ℕ) (h : (1 + n) / (2 ^ n) = 3 / 16) : n = 5 :=
by sorry

end find_n_l83_83544


namespace product_remainder_mod_7_l83_83677

theorem product_remainder_mod_7 (a b c : ℕ) 
  (h1 : a % 7 = 2) 
  (h2 : b % 7 = 3) 
  (h3 : c % 7 = 5) : 
  (a * b * c) % 7 = 2 := 
by 
  sorry

end product_remainder_mod_7_l83_83677


namespace compression_strength_value_l83_83625

def compression_strength (T H : ℕ) : ℚ :=
  (15 * T^5) / (H^3)

theorem compression_strength_value : 
  compression_strength 3 6 = 55 / 13 := by
  sorry

end compression_strength_value_l83_83625


namespace lines_parallel_iff_a_eq_neg2_l83_83699

def line₁_eq (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def line₂_eq (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y - 1 = 0

theorem lines_parallel_iff_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, line₁_eq a x y → line₂_eq a x y) ↔ a = -2 :=
by sorry

end lines_parallel_iff_a_eq_neg2_l83_83699


namespace largest_three_digit_product_l83_83850

theorem largest_three_digit_product : 
    ∃ (n : ℕ), 
    (n = 336) ∧ 
    (n > 99 ∧ n < 1000) ∧ 
    (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = x * y * (5 * x + 2 * y) ∧ 
        ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ k * m = (5 * x + 2 * y)) :=
by
  sorry

end largest_three_digit_product_l83_83850


namespace fraction_meaningfulness_l83_83574

def fraction_is_meaningful (x : ℝ) : Prop :=
  x ≠ 3 / 2

theorem fraction_meaningfulness (x : ℝ) : 
  (2 * x - 3) ≠ 0 ↔ fraction_is_meaningful x :=
by
  sorry

end fraction_meaningfulness_l83_83574


namespace optimal_selling_price_minimize_loss_l83_83446

theorem optimal_selling_price_minimize_loss 
  (C : ℝ) (h1 : 17 * C = 720 + 5 * C) 
  (h2 : ∀ x : ℝ, x * (1 - 0.1) = 720 * 0.9)
  (h3 : ∀ y : ℝ, y * (1 + 0.05) = 648 * 1.05)
  (selling_price : ℝ)
  (optimal_selling_price : selling_price = 60) :
  selling_price = C :=
by 
  sorry

end optimal_selling_price_minimize_loss_l83_83446


namespace more_time_in_swamp_l83_83139

theorem more_time_in_swamp (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : 2 * a + 4 * b + 6 * c = 15) : a > c :=
by {
  sorry
}

end more_time_in_swamp_l83_83139


namespace mixture_ratio_l83_83413

variables (p q : ℝ)

theorem mixture_ratio 
  (h1 : (5/8) * p + (1/4) * q = 0.5)
  (h2 : (3/8) * p + (3/4) * q = 0.5) : 
  p / q = 1 := 
by 
  sorry

end mixture_ratio_l83_83413


namespace trucks_sold_l83_83243

-- Definitions for conditions
def cars_and_trucks_total (T C : Nat) : Prop :=
  T + C = 69

def cars_more_than_trucks (T C : Nat) : Prop :=
  C = T + 27

-- Theorem statement
theorem trucks_sold (T C : Nat) (h1 : cars_and_trucks_total T C) (h2 : cars_more_than_trucks T C) : T = 21 :=
by
  -- This will be replaced by the proof
  sorry

end trucks_sold_l83_83243


namespace log_exp_sum_l83_83732

theorem log_exp_sum :
  2^(Real.log 3 / Real.log 2) + Real.log (Real.sqrt 5) / Real.log 10 + Real.log (Real.sqrt 20) / Real.log 10 = 4 :=
by
  sorry

end log_exp_sum_l83_83732


namespace avg_weight_class_l83_83305

-- Definitions based on the conditions
def students_section_A : Nat := 36
def students_section_B : Nat := 24
def avg_weight_section_A : ℝ := 30.0
def avg_weight_section_B : ℝ := 30.0

-- The statement we want to prove
theorem avg_weight_class :
  (avg_weight_section_A * students_section_A + avg_weight_section_B * students_section_B) / (students_section_A + students_section_B) = 30.0 := 
by
  sorry

end avg_weight_class_l83_83305


namespace area_of_right_angled_isosceles_triangle_l83_83212

-- Definitions
variables {x y : ℝ}
def is_right_angled_isosceles (x y : ℝ) : Prop := y^2 = 2 * x^2
def sum_of_square_areas (x y : ℝ) : Prop := x^2 + x^2 + y^2 = 72

-- Theorem
theorem area_of_right_angled_isosceles_triangle (x y : ℝ) 
  (h1 : is_right_angled_isosceles x y) 
  (h2 : sum_of_square_areas x y) : 
  1/2 * x^2 = 9 :=
sorry

end area_of_right_angled_isosceles_triangle_l83_83212


namespace number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l83_83011

-- Definitions for the sets A and B
def A : Set Int := {x | x^2 - 3 * x - 10 <= 0}
def B (m : Int) : Set Int := {x | m - 1 <= x ∧ x <= 2 * m + 1}

-- Proof for the number of non-empty proper subsets of A
theorem number_of_non_empty_proper_subsets_of_A (x : Int) (h : x ∈ A) : 2^(8 : Nat) - 2 = 254 := by
  sorry

-- Proof for the range of m such that A ⊇ B
theorem range_of_m_for_A_superset_B (m : Int) : (∀ x, x ∈ B m → x ∈ A) ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by
  sorry

end number_of_non_empty_proper_subsets_of_A_range_of_m_for_A_superset_B_l83_83011


namespace min_value_of_m_squared_plus_n_squared_l83_83429

theorem min_value_of_m_squared_plus_n_squared (m n : ℝ) 
  (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) : m^2 + n^2 = 2 :=
sorry

end min_value_of_m_squared_plus_n_squared_l83_83429


namespace ball_reaches_height_l83_83156

theorem ball_reaches_height (h₀ : ℝ) (ratio : ℝ) (target_height : ℝ) (bounces : ℕ) 
  (initial_height : h₀ = 16) 
  (bounce_ratio : ratio = 1/3) 
  (target : target_height = 2) 
  (bounce_count : bounces = 7) :
  h₀ * (ratio ^ bounces) < target_height := 
sorry

end ball_reaches_height_l83_83156


namespace length_of_platform_l83_83683

variable (Vtrain : Real := 55)
variable (str_len : Real := 360)
variable (cross_time : Real := 57.59539236861051)
variable (conversion_factor : Real := 5/18)

theorem length_of_platform :
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  ∃ L : Real, str_len + L = distance_covered → L = 520 :=
by
  let Vtrain_mps := Vtrain * conversion_factor
  let distance_covered := Vtrain_mps * cross_time
  exists (distance_covered - str_len)
  intro h
  have h1 : distance_covered - str_len = 520 := sorry
  exact h1


end length_of_platform_l83_83683


namespace freddy_travel_time_l83_83292

theorem freddy_travel_time (dist_A_B : ℝ) (time_Eddy : ℝ) (dist_A_C : ℝ) (speed_ratio : ℝ) (travel_time_Freddy : ℝ) :
  dist_A_B = 540 ∧ time_Eddy = 3 ∧ dist_A_C = 300 ∧ speed_ratio = 2.4 →
  travel_time_Freddy = dist_A_C / (dist_A_B / time_Eddy / speed_ratio) :=
  sorry

end freddy_travel_time_l83_83292


namespace greatest_two_digit_multiple_of_17_is_85_l83_83970

theorem greatest_two_digit_multiple_of_17_is_85 :
  ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ (n % 17 = 0) ∧ (∀ m : ℕ, (10 ≤ m ∧ m < 100 ∧ m % 17 = 0) → m ≤ n) ∧ n = 85 :=
sorry

end greatest_two_digit_multiple_of_17_is_85_l83_83970


namespace find_y_l83_83001

-- Definitions of the given conditions
def angle_ABC_is_straight_line := true  -- This is to ensure the angle is a straight line.
def angle_ABD_is_exterior_of_triangle_BCD := true -- This is to ensure ABD is an exterior angle.
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Theorem to prove y = 36 given the conditions
theorem find_y (A B C D : Type) (y : ℝ) 
    (h1 : angle_ABC_is_straight_line)
    (h2 : angle_ABD_is_exterior_of_triangle_BCD)
    (h3 : angle_ABD = 118)
    (h4 : angle_BCD = 82) : 
            y = 36 :=
  by
  sorry

end find_y_l83_83001


namespace total_perimeter_l83_83431

/-- 
A rectangular plot where the long sides are three times the length of the short sides. 
One short side is 80 feet. Prove the total perimeter is 640 feet.
-/
theorem total_perimeter (s : ℕ) (h : s = 80) : 8 * s = 640 :=
  by sorry

end total_perimeter_l83_83431


namespace distance_last_day_l83_83852

theorem distance_last_day
  (total_distance : ℕ)
  (days : ℕ)
  (initial_distance : ℕ)
  (common_ratio : ℚ)
  (sum_geometric : initial_distance * (1 - common_ratio^days) / (1 - common_ratio) = total_distance) :
  total_distance = 378 → days = 6 → common_ratio = 1/2 → 
  initial_distance = 192 → initial_distance * common_ratio^(days - 1) = 6 := 
by
  intros h1 h2 h3 h4
  sorry

end distance_last_day_l83_83852


namespace triangle_angle_and_side_ratio_l83_83464

theorem triangle_angle_and_side_ratio
  (A B C : Real)
  (a b c : Real)
  (h1 : a / Real.sin A = b / Real.sin B)
  (h2 : b / Real.sin B = c / Real.sin C)
  (h3 : (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C)) :
  C = Real.pi / 3 ∧ (1 < (a + b) / c ∧ (a + b) / c < 2) :=
by
  sorry


end triangle_angle_and_side_ratio_l83_83464


namespace jackie_sleeping_hours_l83_83312

def hours_in_a_day : ℕ := 24
def work_hours : ℕ := 8
def exercise_hours : ℕ := 3
def free_time_hours : ℕ := 5
def accounted_hours : ℕ := work_hours + exercise_hours + free_time_hours

theorem jackie_sleeping_hours :
  hours_in_a_day - accounted_hours = 8 := by
  sorry

end jackie_sleeping_hours_l83_83312


namespace trivia_team_members_l83_83350

theorem trivia_team_members (n p s x y : ℕ) (h1 : n = 12) (h2 : p = 64) (h3 : s = 8) (h4 : x = p / s) (h5 : y = n - x) : y = 4 :=
by
  sorry

end trivia_team_members_l83_83350


namespace sales_volume_expression_reduction_for_desired_profit_l83_83299

-- Initial conditions definitions.
def initial_purchase_price : ℝ := 3
def initial_selling_price : ℝ := 5
def initial_sales_volume : ℝ := 100
def sales_increase_per_0_1_yuan : ℝ := 20
def desired_profit : ℝ := 300
def minimum_sales_volume : ℝ := 220

-- Question (1): Sales Volume Expression
theorem sales_volume_expression (x : ℝ) : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) = 100 + 200 * x :=
by sorry

-- Question (2): Determine Reduction for Desired Profit and Minimum Sales Volume
theorem reduction_for_desired_profit (x : ℝ) 
  (hx : (initial_selling_price - initial_purchase_price - x) * (initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x)) = desired_profit)
  (hy : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) >= minimum_sales_volume) :
  x = 1 :=
by sorry

end sales_volume_expression_reduction_for_desired_profit_l83_83299


namespace cross_product_u_v_l83_83198

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (3, -4, 7)
def v : ℝ × ℝ × ℝ := (2, 5, -3)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- State the theorem to be proved
theorem cross_product_u_v : cross_product u v = (-23, 23, 23) :=
  sorry

end cross_product_u_v_l83_83198


namespace rita_daily_minimum_payment_l83_83143

theorem rita_daily_minimum_payment (total_cost down_payment balance daily_payment : ℝ) 
    (h1 : total_cost = 120)
    (h2 : down_payment = total_cost / 2)
    (h3 : balance = total_cost - down_payment)
    (h4 : daily_payment = balance / 10) : daily_payment = 6 :=
by
  sorry

end rita_daily_minimum_payment_l83_83143


namespace fuel_ethanol_problem_l83_83976

theorem fuel_ethanol_problem (x : ℝ) (h : 0.12 * x + 0.16 * (200 - x) = 28) : x = 100 := 
by
  sorry

end fuel_ethanol_problem_l83_83976


namespace number_of_n_l83_83584

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end number_of_n_l83_83584


namespace integral_solutions_count_l83_83875

theorem integral_solutions_count (m : ℕ) (h : m > 0) :
  ∃ S : Finset (ℕ × ℕ), S.card = m ∧ 
  ∀ (p : ℕ × ℕ), p ∈ S → (p.1^2 + p.2^2 + 2 * p.1 * p.2 - m * p.1 - m * p.2 - m - 1 = 0) := 
sorry

end integral_solutions_count_l83_83875


namespace range_of_3a_minus_b_l83_83920

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 4) (h2 : -1 ≤ a - b ∧ a - b ≤ 2) :
  -1 ≤ (3 * a - b) ∧ (3 * a - b) ≤ 8 :=
sorry

end range_of_3a_minus_b_l83_83920


namespace coefficient_x2_in_expansion_l83_83995

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove the coefficient of the x^2 term in (x + 1)^42 is 861
theorem coefficient_x2_in_expansion :
  (binomial 42 2) = 861 := by
  sorry

end coefficient_x2_in_expansion_l83_83995


namespace marshmallow_per_smore_l83_83177

theorem marshmallow_per_smore (graham_crackers : ℕ) (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) 
                               (graham_crackers_per_smore : ℕ) :
  graham_crackers = 48 ∧ initial_marshmallows = 6 ∧ additional_marshmallows = 18 ∧ graham_crackers_per_smore = 2 →
  (initial_marshmallows + additional_marshmallows) / (graham_crackers / graham_crackers_per_smore) = 1 :=
by
  intro h
  sorry

end marshmallow_per_smore_l83_83177


namespace interior_angle_of_regular_hexagon_l83_83302

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l83_83302


namespace find_roots_l83_83272

theorem find_roots (x : ℝ) (h : 21 / (x^2 - 9) - 3 / (x - 3) = 1) : x = -3 ∨ x = 7 :=
by {
  sorry
}

end find_roots_l83_83272


namespace differentiable_function_inequality_l83_83545

theorem differentiable_function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * (f 1) :=
sorry

end differentiable_function_inequality_l83_83545


namespace cake_eating_classmates_l83_83468

theorem cake_eating_classmates (n : ℕ) :
  (Alex_ate : ℚ := 1/11) → (Alena_ate : ℚ := 1/14) → 
  (12 ≤ n ∧ n ≤ 13) :=
by
  sorry

end cake_eating_classmates_l83_83468


namespace lowest_test_score_dropped_l83_83892

theorem lowest_test_score_dropped (A B C D : ℕ) 
  (h_avg_four : A + B + C + D = 140) 
  (h_avg_three : A + B + C = 120) : 
  D = 20 := 
by
  sorry

end lowest_test_score_dropped_l83_83892


namespace four_digit_non_convertible_to_1992_multiple_l83_83742

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_multiple_of_1992 (n : ℕ) : Prop :=
  n % 1992 = 0

def reachable (n m : ℕ) (k : ℕ) : Prop :=
  ∃ x y z : ℕ, 
    x ≠ m ∧ y ≠ m ∧ z ≠ m ∧
    (n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3)) % 1992 = 0 ∧
    n + x * 10^(k-1) + y * 10^(k-2) + z * 10^(k-3) < 10000

theorem four_digit_non_convertible_to_1992_multiple :
  ∃ n : ℕ, is_four_digit n ∧ (∀ m : ℕ, is_four_digit m ∧ is_multiple_of_1992 m → ¬ reachable n m 3) :=
sorry

end four_digit_non_convertible_to_1992_multiple_l83_83742


namespace minimum_value_condition_l83_83672

theorem minimum_value_condition (a b : ℝ) (h : 16 * a ^ 2 + 2 * a + 8 * a * b + b ^ 2 - 1 = 0) : 
  ∃ m : ℝ, m = 3 * a + b ∧ m ≥ -1 :=
sorry

end minimum_value_condition_l83_83672


namespace total_parking_spaces_l83_83598

-- Definitions of conditions
def caravan_space : ℕ := 2
def number_of_caravans : ℕ := 3
def spaces_left : ℕ := 24

-- Proof statement
theorem total_parking_spaces :
  (number_of_caravans * caravan_space + spaces_left) = 30 :=
by
  sorry

end total_parking_spaces_l83_83598


namespace tetrahedron_sum_l83_83301

theorem tetrahedron_sum :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  sorry

end tetrahedron_sum_l83_83301


namespace possible_values_of_b_l83_83033

theorem possible_values_of_b 
        (b : ℤ)
        (h : ∃ x : ℤ, (x ^ 3 + 2 * x ^ 2 + b * x + 8 = 0)) :
        b = -81 ∨ b = -26 ∨ b = -12 ∨ b = -6 ∨ b = 4 ∨ b = 9 ∨ b = 47 :=
  sorry

end possible_values_of_b_l83_83033


namespace bhanu_income_l83_83839

theorem bhanu_income (I P : ℝ) (h1 : (P / 100) * I = 300) (h2 : (20 / 100) * (I - 300) = 140) : P = 30 := by
  sorry

end bhanu_income_l83_83839


namespace kiyiv_first_problem_kiyiv_second_problem_l83_83567

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that x^3 + y^3 + 4xy ≥ x^2 + y^2 + x + y + 2. -/
theorem kiyiv_first_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  x^3 + y^3 + 4 * x * y ≥ x^2 + y^2 + x + y + 2 :=
sorry

/-- Let x and y be positive real numbers such that xy ≥ 1.
Prove that 2(x^3 + y^3 + xy + x + y) ≥ 5(x^2 + y^2). -/
theorem kiyiv_second_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 ≤ x * y) :
  2 * (x^3 + y^3 + x * y + x + y) ≥ 5 * (x^2 + y^2) :=
sorry

end kiyiv_first_problem_kiyiv_second_problem_l83_83567


namespace necessary_but_not_sufficient_condition_l83_83304

theorem necessary_but_not_sufficient_condition (p : ℝ) : 
  p < 2 → (¬(p^2 - 4 < 0) → ∃ q, q < p ∧ q^2 - 4 < 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l83_83304


namespace add_neg3_and_2_mul_neg3_and_2_l83_83067

theorem add_neg3_and_2 : -3 + 2 = -1 := 
by
  sorry

theorem mul_neg3_and_2 : (-3) * 2 = -6 := 
by
  sorry

end add_neg3_and_2_mul_neg3_and_2_l83_83067


namespace area_comparison_l83_83436

noncomputable def area_difference_decagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 10))
  let r := s / (2 * Real.tan (Real.pi / 10))
  Real.pi * (R^2 - r^2)

noncomputable def area_difference_nonagon (s : ℝ) : ℝ := 
  let R := s / (2 * Real.sin (Real.pi / 9))
  let r := s / (2 * Real.tan (Real.pi / 9))
  Real.pi * (R^2 - r^2)

theorem area_comparison :
  (area_difference_decagon 3 > area_difference_nonagon 3) :=
sorry

end area_comparison_l83_83436


namespace complete_square_transformation_l83_83010

theorem complete_square_transformation : 
  ∀ (x : ℝ), (x^2 - 8 * x + 9 = 0) → ((x - 4)^2 = 7) :=
by
  intros x h
  sorry

end complete_square_transformation_l83_83010


namespace probability_of_drawing_K_is_2_over_27_l83_83176

-- Define the total number of cards in a standard deck of 54 cards
def total_cards : ℕ := 54

-- Define the number of "K" cards in the standard deck
def num_K_cards : ℕ := 4

-- Define the probability function for drawing a "K"
def probability_drawing_K (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- Prove that the probability of drawing a "K" is 2/27
theorem probability_of_drawing_K_is_2_over_27 :
  probability_drawing_K total_cards num_K_cards = 2 / 27 :=
by
  sorry

end probability_of_drawing_K_is_2_over_27_l83_83176


namespace Ed_lost_marble_count_l83_83474

variable (D : ℕ) -- Number of marbles Doug has

noncomputable def Ed_initial := D + 19 -- Ed initially had D + 19 marbles
noncomputable def Ed_now := D + 8 -- Ed now has D + 8 marbles
noncomputable def Ed_lost := Ed_initial D - Ed_now D -- Ed lost Ed_initial - Ed_now marbles

theorem Ed_lost_marble_count : Ed_lost D = 11 := by 
  sorry

end Ed_lost_marble_count_l83_83474


namespace proportion_solution_l83_83993

-- Define the given proportion condition as a hypothesis
variable (x : ℝ)

-- The definition is derived directly from the given problem
def proportion_condition : Prop := x / 5 = 1.2 / 8

-- State the theorem using the given proportion condition to prove x = 0.75
theorem proportion_solution (h : proportion_condition x) : x = 0.75 :=
  by
    sorry

end proportion_solution_l83_83993


namespace true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l83_83394

theorem true_if_a_gt_1_and_b_gt_1_then_ab_gt_1 (a b : ℝ) (ha : a > 1) (hb : b > 1) : ab > 1 :=
sorry

end true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l83_83394


namespace derivative_of_f_eq_f_deriv_l83_83547

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x - (Real.sin a) ^ x

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x * Real.log (Real.cos a) - (Real.sin a) ^ x * Real.log (Real.sin a)

theorem derivative_of_f_eq_f_deriv (a : ℝ) (h : 0 < a ∧ a < Real.pi / 2) :
  (deriv (f a)) = f_deriv a :=
by
  sorry

end derivative_of_f_eq_f_deriv_l83_83547


namespace tobias_downloads_l83_83492

theorem tobias_downloads : 
  ∀ (m : ℕ), (∀ (price_per_app total_spent : ℝ), 
  price_per_app = 2.00 + 2.00 * 0.10 ∧ 
  total_spent = 52.80 → 
  m = total_spent / price_per_app) → 
  m = 24 := 
  sorry

end tobias_downloads_l83_83492


namespace hockey_players_count_l83_83372

theorem hockey_players_count (cricket_players : ℕ) (football_players : ℕ) (softball_players : ℕ) (total_players : ℕ) 
(h_cricket : cricket_players = 16) 
(h_football : football_players = 18) 
(h_softball : softball_players = 13) 
(h_total : total_players = 59) : 
  total_players - (cricket_players + football_players + softball_players) = 12 := 
by sorry

end hockey_players_count_l83_83372


namespace problem_integer_pairs_l83_83329

theorem problem_integer_pairs (a b q r : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + r = 1977) :
    (a, b) = (50, 7) ∨ (a, b) = (50, 37) ∨ (a, b) = (7, 50) ∨ (a, b) = (37, 50) :=
sorry

end problem_integer_pairs_l83_83329


namespace calc_f_g_h_2_l83_83148

def f (x : ℕ) : ℕ := x + 5
def g (x : ℕ) : ℕ := x^2 - 8
def h (x : ℕ) : ℕ := 2 * x + 1

theorem calc_f_g_h_2 : f (g (h 2)) = 22 := by
  sorry

end calc_f_g_h_2_l83_83148


namespace are_naptime_l83_83347

def flight_duration := 11 * 60 + 20  -- in minutes

def time_spent_reading := 2 * 60      -- in minutes
def time_spent_watching_movies := 4 * 60  -- in minutes
def time_spent_eating_dinner := 30    -- in minutes
def time_spent_listening_to_radio := 40   -- in minutes
def time_spent_playing_games := 1 * 60 + 10   -- in minutes

def total_time_spent_on_activities := 
  time_spent_reading + 
  time_spent_watching_movies + 
  time_spent_eating_dinner + 
  time_spent_listening_to_radio + 
  time_spent_playing_games

def remaining_time := (flight_duration - total_time_spent_on_activities) / 60  -- in hours

theorem are_naptime : remaining_time = 3 := by
  sorry

end are_naptime_l83_83347


namespace algebra_expression_eq_l83_83582

theorem algebra_expression_eq (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 := by
  sorry

end algebra_expression_eq_l83_83582


namespace find_ratio_MH_NH_OH_l83_83051

-- Defining the main problem variables.
variable {A B C O H M N : Type} -- A, B, C are points, O is circumcenter, H is orthocenter, M and N are points on other segments
variables (angleA : ℝ) (AB AC : ℝ)
variables (angleBOC angleBHC : ℝ)
variables (BM CN MH NH OH : ℝ)

-- Conditions: Given constraints from the problem.
axiom angle_A_eq_60 : angleA = 60 -- ∠A = 60°
axiom AB_greater_AC : AB > AC -- AB > AC
axiom circumcenter_property : angleBOC = 120 -- ∠BOC = 120°
axiom orthocenter_property : angleBHC = 120 -- ∠BHC = 120°
axiom BM_eq_CN : BM = CN -- BM = CN

-- Statement of the mathematical proof we need to show.
theorem find_ratio_MH_NH_OH : (MH + NH) / OH = Real.sqrt 3 :=
by
  sorry

end find_ratio_MH_NH_OH_l83_83051


namespace find_remainder_division_l83_83576

/--
Given:
1. A dividend of 100.
2. A quotient of 9.
3. A divisor of 11.

Prove: The remainder \( r \) when dividing 100 by 11 is 1.
-/
theorem find_remainder_division :
  ∀ (q d r : Nat), q = 9 → d = 11 → 100 = (d * q + r) → r = 1 :=
by
  intros q d r hq hd hdiv
  -- Proof steps would go here
  sorry

end find_remainder_division_l83_83576


namespace maximum_notebooks_maria_can_buy_l83_83586

def price_single : ℕ := 1
def price_pack_4 : ℕ := 3
def price_pack_7 : ℕ := 5
def total_budget : ℕ := 10

def max_notebooks (budget : ℕ) : ℕ :=
  if budget < price_single then 0
  else if budget < price_pack_4 then budget / price_single
  else if budget < price_pack_7 then max (budget / price_single) (4 * (budget / price_pack_4))
  else max (budget / price_single) (7 * (budget / price_pack_7))

theorem maximum_notebooks_maria_can_buy :
  max_notebooks total_budget = 14 := by
  sorry

end maximum_notebooks_maria_can_buy_l83_83586


namespace two_trains_clearing_time_l83_83008

noncomputable def length_train1 : ℝ := 100  -- Length of Train 1 in meters
noncomputable def length_train2 : ℝ := 160  -- Length of Train 2 in meters
noncomputable def speed_train1 : ℝ := 42 * 1000 / 3600  -- Speed of Train 1 in m/s
noncomputable def speed_train2 : ℝ := 30 * 1000 / 3600  -- Speed of Train 2 in m/s
noncomputable def total_distance : ℝ := length_train1 + length_train2  -- Total distance to be covered
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2  -- Relative speed

theorem two_trains_clearing_time : total_distance / relative_speed = 13 := by
  sorry

end two_trains_clearing_time_l83_83008


namespace train_pass_time_l83_83873

theorem train_pass_time
  (v : ℝ) (l_tunnel l_train : ℝ) (h_v : v = 75) (h_l_tunnel : l_tunnel = 3.5) (h_l_train : l_train = 0.25) :
  (l_tunnel + l_train) / v * 60 = 3 :=
by 
  -- Placeholder for the proof
  sorry

end train_pass_time_l83_83873


namespace reciprocal_of_neg_one_sixth_is_neg_six_l83_83843

theorem reciprocal_of_neg_one_sixth_is_neg_six : 1 / (- (1 / 6)) = -6 :=
by sorry

end reciprocal_of_neg_one_sixth_is_neg_six_l83_83843


namespace modulus_of_z_l83_83969

noncomputable def z : ℂ := sorry
def condition (z : ℂ) : Prop := z * (1 - Complex.I) = 2 * Complex.I

theorem modulus_of_z (hz : condition z) : Complex.abs z = Real.sqrt 2 := sorry

end modulus_of_z_l83_83969


namespace total_legs_of_passengers_l83_83529

theorem total_legs_of_passengers :
  ∀ (total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs : ℕ),
  total_heads = 15 →
  cats = 7 →
  cat_legs = 4 →
  human_heads = (total_heads - cats) →
  normal_human_legs = 2 →
  one_legged_captain_legs = 1 →
  ((cats * cat_legs) + ((human_heads - 1) * normal_human_legs) + one_legged_captain_legs) = 43 :=
by
  intros total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs h1 h2 h3 h4 h5 h6
  sorry

end total_legs_of_passengers_l83_83529


namespace darnel_jog_laps_l83_83746

theorem darnel_jog_laps (x : ℝ) (h1 : 0.88 = x + 0.13) : x = 0.75 := by
  sorry

end darnel_jog_laps_l83_83746


namespace horizontal_asymptote_of_rational_function_l83_83379

theorem horizontal_asymptote_of_rational_function :
  (∃ y, y = (10 * x ^ 4 + 3 * x ^ 3 + 7 * x ^ 2 + 6 * x + 4) / (2 * x ^ 4 + 5 * x ^ 3 + 4 * x ^ 2 + 2 * x + 1) → y = 5) := sorry

end horizontal_asymptote_of_rational_function_l83_83379


namespace avg_of_arithmetic_series_is_25_l83_83673

noncomputable def arithmetic_series_avg : ℝ :=
  let a₁ := 15
  let d := 1 / 4
  let aₙ := 35
  let n := (aₙ - a₁) / d + 1
  let S := n * (a₁ + aₙ) / 2
  S / n

theorem avg_of_arithmetic_series_is_25 : arithmetic_series_avg = 25 := 
by
  -- Sorry, proof omitted due to instruction.
  sorry

end avg_of_arithmetic_series_is_25_l83_83673


namespace train_length_l83_83412

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l83_83412


namespace fraction_of_succeeding_number_l83_83936

theorem fraction_of_succeeding_number (N : ℝ) (hN : N = 24.000000000000004) :
  ∃ f : ℝ, (1 / 4) * N > f * (N + 1) + 1 ∧ f = 0.2 :=
by
  sorry

end fraction_of_succeeding_number_l83_83936


namespace tax_collected_from_village_l83_83629

-- Definitions according to the conditions in the problem
def MrWillamTax : ℝ := 500
def MrWillamPercentage : ℝ := 0.21701388888888893

-- The theorem to prove the total tax collected
theorem tax_collected_from_village : ∃ (total_collected : ℝ), MrWillamPercentage * total_collected = MrWillamTax ∧ total_collected = 2303.7037037037035 :=
sorry

end tax_collected_from_village_l83_83629


namespace arithmetic_sequence_problem_l83_83981

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : ℝ)
  (d : ℝ)
  (h1 : d = 2)
  (h2 : ∀ n : ℕ, a n = a1 + (n - 1) * d)
  (h3 :  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2)
  (h4 : S 6 = 3 * S 3) :
  a 9 = 20 :=
by sorry

end arithmetic_sequence_problem_l83_83981


namespace plane_equation_l83_83262

theorem plane_equation (p q r : ℝ × ℝ × ℝ)
  (h₁ : p = (2, -1, 3))
  (h₂ : q = (0, -1, 5))
  (h₃ : r = (-1, -3, 4)) :
  ∃ A B C D : ℤ, A = 1 ∧ B = 2 ∧ C = -1 ∧ D = 3 ∧
               A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
               ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔
                             (x, y, z) = p ∨ (x, y, z) = q ∨ (x, y, z) = r :=
by
  sorry

end plane_equation_l83_83262


namespace books_left_over_l83_83411

theorem books_left_over (n_boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) :
  n_boxes = 1575 → books_per_box = 45 → new_box_capacity = 46 →
  (n_boxes * books_per_box) % new_box_capacity = 15 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  -- Actual proof steps would go here
  sorry

end books_left_over_l83_83411


namespace simplify_expression_l83_83856

theorem simplify_expression (a b : ℝ) :
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 :=
by
  sorry

end simplify_expression_l83_83856


namespace fraction_subtraction_l83_83207

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 :=
by
  sorry

end fraction_subtraction_l83_83207


namespace price_25_bag_l83_83167

noncomputable def price_per_bag_25 : ℝ := 28.97

def price_per_bag_5 : ℝ := 13.85
def price_per_bag_10 : ℝ := 20.42

def total_cost (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ) : ℝ :=
  n5 * p5 + n10 * p10 + n25 * p25

theorem price_25_bag :
  ∃ (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ),
    p5 = price_per_bag_5 ∧
    p10 = price_per_bag_10 ∧
    p25 = price_per_bag_25 ∧
    65 ≤ 5 * n5 + 10 * n10 + 25 * n25 ∧
    5 * n5 + 10 * n10 + 25 * n25 ≤ 80 ∧
    total_cost p5 p10 p25 n5 n10 n25 = 98.77 :=
by
  sorry

end price_25_bag_l83_83167


namespace max_rocket_height_l83_83543

-- Define the quadratic function representing the rocket's height
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

-- State the maximum height problem
theorem max_rocket_height : ∃ t : ℝ, rocket_height t = 175 ∧ ∀ t' : ℝ, rocket_height t' ≤ 175 :=
by
  use 2.5
  sorry -- The proof will show that the maximum height is 175 meters at time t = 2.5 seconds

end max_rocket_height_l83_83543


namespace half_product_unique_l83_83132

theorem half_product_unique (x : ℕ) (n k : ℕ) 
  (hn : x = n * (n + 1) / 2) (hk : x = k * (k + 1) / 2) : 
  n = k := 
sorry

end half_product_unique_l83_83132


namespace find_a_plus_b_l83_83991

-- Define the constants and conditions
variables (a b c : ℤ)
variables (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13)
variables (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31)

-- State the theorem
theorem find_a_plus_b (a b c : ℤ) (a_cond : 0 ≤ a ∧ a < 5) (b_cond : 0 ≤ b ∧ b < 13) (frac_decomp : (1 : ℚ) / 2015 = (a : ℚ) / 5 + (b : ℚ) / 13 + (c : ℚ) / 31) :
  a + b = 14 := 
sorry

end find_a_plus_b_l83_83991


namespace sqrt_of_0_01_l83_83283

theorem sqrt_of_0_01 : Real.sqrt 0.01 = 0.1 :=
by
  sorry

end sqrt_of_0_01_l83_83283


namespace smallest_whole_number_l83_83397

theorem smallest_whole_number (m : ℕ) :
  m % 2 = 1 ∧
  m % 3 = 1 ∧
  m % 4 = 1 ∧
  m % 5 = 1 ∧
  m % 6 = 1 ∧
  m % 8 = 1 ∧
  m % 11 = 0 → 
  m = 1801 :=
by
  intros h
  sorry

end smallest_whole_number_l83_83397


namespace part1_part2_l83_83050

-- Define the absolute value function
def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

-- Given conditions
def condition1 : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 ↔ f x a ≤ 6

def condition2 (a : ℝ) : Prop :=
  ∃ t m : ℝ, f (t / 2) a ≤ m - f (-t) a

-- Statements to prove
theorem part1 : ∃ a : ℝ, condition1 ∧ a = 1 := by
  sorry

theorem part2 : ∀ {a : ℝ}, a = 1 → ∃ m : ℝ, m ≥ 3.5 ∧ condition2 a := by
  sorry

end part1_part2_l83_83050


namespace find_coefficients_l83_83747

variable (P Q x : ℝ)

theorem find_coefficients :
  (∀ x, x^2 - 8 * x - 20 = (x - 10) * (x + 2))
  → (∀ x, 6 * x - 4 = P * (x + 2) + Q * (x - 10))
  → P = 14 / 3 ∧ Q = 4 / 3 :=
by
  intros h1 h2
  sorry

end find_coefficients_l83_83747


namespace min_value_of_reciprocal_sum_l83_83145

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) :
  ∃ z, (z = 3 + 2 * Real.sqrt 2) ∧ (∀ z', (z' = 1 / x + 1 / y) → z ≤ z') :=
sorry

end min_value_of_reciprocal_sum_l83_83145


namespace house_A_cost_l83_83634

theorem house_A_cost (base_salary earnings commission_rate total_houses cost_A cost_B cost_C : ℝ)
  (H_base_salary : base_salary = 3000)
  (H_earnings : earnings = 8000)
  (H_commission_rate : commission_rate = 0.02)
  (H_cost_B : cost_B = 3 * cost_A)
  (H_cost_C : cost_C = 2 * cost_A - 110000)
  (H_total_commission : earnings - base_salary = 5000)
  (H_total_cost : 5000 / commission_rate = 250000)
  (H_total_houses : base_salary + commission_rate * (cost_A + cost_B + cost_C) = earnings) :
  cost_A = 60000 := sorry

end house_A_cost_l83_83634


namespace symmetry_condition_l83_83712

theorem symmetry_condition (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∀ x y : ℝ, y = x ↔ x = (ax + b) / (cx - d)) ∧ 
  (∀ x y : ℝ, y = -x ↔ x = (-ax + b) / (-cx - d)) → 
  d + b = 0 :=
by sorry

end symmetry_condition_l83_83712


namespace cafeteria_extra_apples_l83_83805

-- Define the conditions from the problem
def red_apples : ℕ := 33
def green_apples : ℕ := 23
def students : ℕ := 21

-- Define the total apples and apples given out based on the conditions
def total_apples : ℕ := red_apples + green_apples
def apples_given : ℕ := students

-- Define the extra apples as the difference between total apples and apples given out
def extra_apples : ℕ := total_apples - apples_given

-- The theorem to prove that the number of extra apples is 35
theorem cafeteria_extra_apples : extra_apples = 35 :=
by
  -- The structure of the proof would go here, but is omitted
  sorry

end cafeteria_extra_apples_l83_83805


namespace parents_without_fulltime_jobs_l83_83409

theorem parents_without_fulltime_jobs (total : ℕ) (mothers fathers full_time_mothers full_time_fathers : ℕ) 
(h1 : mothers = 2 * fathers / 3)
(h2 : full_time_mothers = 9 * mothers / 10)
(h3 : full_time_fathers = 3 * fathers / 4)
(h4 : mothers + fathers = total) :
(100 * (total - (full_time_mothers + full_time_fathers))) / total = 19 :=
by
  sorry

end parents_without_fulltime_jobs_l83_83409


namespace largest_possible_a_l83_83294

theorem largest_possible_a (a b c d : ℕ) (ha : a < 2 * b) (hb : b < 3 * c) (hc : c < 4 * d) (hd : d < 100) : 
  a ≤ 2367 :=
sorry

end largest_possible_a_l83_83294


namespace paint_house_18_women_4_days_l83_83971

theorem paint_house_18_women_4_days :
  (∀ (m1 m2 : ℕ) (d1 d2 : ℕ), m1 * d1 = m2 * d2) →
  (12 * 6 = 72) →
  (72 = 18 * d) →
  d = 4.0 :=
by
  sorry

end paint_house_18_women_4_days_l83_83971


namespace min_value_of_function_l83_83421

open Real

theorem min_value_of_function (x : ℝ) (h : x > 2) : (∃ a : ℝ, (∀ y : ℝ, y = (4 / (x - 2) + x) → y ≥ a) ∧ a = 6) :=
sorry

end min_value_of_function_l83_83421


namespace compute_z_pow_8_l83_83882

noncomputable def z : ℂ := (1 - Real.sqrt 3 * Complex.I) / 2

theorem compute_z_pow_8 : z ^ 8 = -(1 + Real.sqrt 3 * Complex.I) / 2 :=
by
  sorry

end compute_z_pow_8_l83_83882


namespace calculate_expression_l83_83669

theorem calculate_expression :
  107 * 107 + 93 * 93 = 20098 := by
  sorry

end calculate_expression_l83_83669


namespace polygon_sides_l83_83862

theorem polygon_sides :
  ∃ (n : ℕ), (n * (n - 3) / 2) = n + 33 ∧ n = 11 :=
by
  sorry

end polygon_sides_l83_83862


namespace max_z_under_D_le_1_l83_83647

noncomputable def f (x a b : ℝ) : ℝ := x - a * x^2 + b
noncomputable def f0 (x b0 : ℝ) : ℝ := x^2 + b0
noncomputable def g (x a b b0 : ℝ) : ℝ := f x a b - f0 x b0

theorem max_z_under_D_le_1 
  (a b b0 : ℝ) (D : ℝ)
  (h_a : a = 0) 
  (h_b0 : b0 = 0) 
  (h_D : D ≤ 1)
  (h_maxD : ∀ x : ℝ, - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 → g (Real.sin x) a b b0 ≤ D) :
  ∃ z : ℝ, z = b - a^2 / 4 ∧ z = 1 :=
by
  sorry

end max_z_under_D_le_1_l83_83647


namespace coolers_total_capacity_l83_83482

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l83_83482


namespace chimes_1000_on_march_7_l83_83493

theorem chimes_1000_on_march_7 : 
  ∀ (initial_time : Nat) (start_date : Nat) (chimes_before_noon : Nat) 
  (chimes_per_day : Nat) (target_chime : Nat) (final_date : Nat),
  initial_time = 10 * 60 + 15 ∧
  start_date = 26 ∧
  chimes_before_noon = 25 ∧
  chimes_per_day = 103 ∧
  target_chime = 1000 ∧
  final_date = start_date + (target_chime - chimes_before_noon) / chimes_per_day ∧
  (target_chime - chimes_before_noon) % chimes_per_day ≤ chimes_per_day
  → final_date = 7 := 
by
  intros
  sorry

end chimes_1000_on_march_7_l83_83493


namespace tan_identity_given_condition_l83_83175

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end tan_identity_given_condition_l83_83175


namespace number_of_lines_l83_83594

-- Define the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the condition that a line intersects a parabola at only one point
def line_intersects_parabola_at_one_point (m b x y : ℝ) : Prop :=
  y - (m * x + b) = 0 ∧ parabola x y

-- The proof problem: Prove there are 3 such lines
theorem number_of_lines : ∃ (n : ℕ), n = 3 ∧ (
  ∃ (m b : ℝ), line_intersects_parabola_at_one_point m b 0 1) :=
sorry

end number_of_lines_l83_83594


namespace angle_triple_supplementary_l83_83887

theorem angle_triple_supplementary (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
  sorry

end angle_triple_supplementary_l83_83887


namespace Ryan_reads_more_l83_83632

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end Ryan_reads_more_l83_83632


namespace sole_mart_meals_l83_83528

theorem sole_mart_meals (c_c_meals : ℕ) (meals_given_away : ℕ) (meals_left : ℕ)
  (h1 : c_c_meals = 113) (h2 : meals_givenAway = 85) (h3 : meals_left = 78)  :
  ∃ m : ℕ, m + c_c_meals = meals_givenAway + meals_left ∧ m = 50 := 
by
  sorry

end sole_mart_meals_l83_83528


namespace no_girl_can_avoid_losing_bet_l83_83306

theorem no_girl_can_avoid_losing_bet
  (G1 G2 G3 : Prop)
  (h1 : G1 ↔ ¬G2)
  (h2 : G2 ↔ ¬G3)
  (h3 : G3 ↔ ¬G1)
  : G1 ∧ G2 ∧ G3 → False := by
  sorry

end no_girl_can_avoid_losing_bet_l83_83306


namespace smallest_number_of_students_l83_83401

-- Define the conditions as given in the problem
def eight_to_six_ratio : ℕ × ℕ := (5, 3) -- ratio of 8th-graders to 6th-graders
def eight_to_nine_ratio : ℕ × ℕ := (7, 4) -- ratio of 8th-graders to 9th-graders

theorem smallest_number_of_students (a b c : ℕ)
  (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : a = 7 * c) : a + b + c = 76 := 
sorry

end smallest_number_of_students_l83_83401


namespace integers_multiples_of_d_l83_83457

theorem integers_multiples_of_d (d m n : ℕ) 
  (h1 : 2 ≤ m) 
  (h2 : 1 ≤ n) 
  (gcd_m_n : Nat.gcd m n = d) 
  (gcd_m_4n1 : Nat.gcd m (4 * n + 1) = 1) : 
  m % d = 0 :=
sorry

end integers_multiples_of_d_l83_83457


namespace root_relation_l83_83649

theorem root_relation (a b x y : ℝ)
  (h1 : x + y = a)
  (h2 : (1 / x) + (1 / y) = 1 / b)
  (h3 : x = 3 * y)
  (h4 : y = a / 4) :
  b = 3 * a / 16 :=
by
  sorry

end root_relation_l83_83649


namespace find_k_l83_83144

-- Definitions
def a (n : ℕ) : ℤ := 1 + (n - 1) * 2
def S (n : ℕ) : ℤ := n / 2 * (2 * 1 + (n - 1) * 2)

-- Main theorem statement
theorem find_k (k : ℕ) (h : S (k + 2) - S k = 24) : k = 5 :=
by sorry

end find_k_l83_83144


namespace sin_squared_value_l83_83956

theorem sin_squared_value (x : ℝ) (h : Real.tan x = 1 / 2) : 
  Real.sin (π / 4 + x) ^ 2 = 9 / 10 :=
by
  -- Proof part, skipped.
  sorry

end sin_squared_value_l83_83956


namespace part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l83_83996

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x + Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 - a / (x^2) + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f' x a - x

-- Problem (I)
theorem part_I (a : ℝ) : f' 1 a = 0 → a = 2 := sorry

-- Problem (II)
theorem part_II (a : ℝ) : (∀ x, 1 < x ∧ x < 2 → f' x a ≥ 0) → a ≤ 2 := sorry

-- Problem (III)
theorem part_III_no_zeros (a : ℝ) : a > 1 → ∀ x, g x a ≠ 0 := sorry
theorem part_III_one_zero (a : ℝ) : (a = 1 ∨ a ≤ 0) → ∃! x, g x a = 0 := sorry
theorem part_III_two_zeros (a : ℝ) : 0 < a ∧ a < 1 → ∃ x1 x2, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0 := sorry

end part_I_part_II_part_III_no_zeros_part_III_one_zero_part_III_two_zeros_l83_83996


namespace points_on_line_initial_l83_83066

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l83_83066


namespace Z_4_1_eq_27_l83_83650

def Z (a b : ℕ) : ℕ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem Z_4_1_eq_27 : Z 4 1 = 27 := by
  sorry

end Z_4_1_eq_27_l83_83650


namespace mean_height_of_players_l83_83694

def heights_50s : List ℕ := [57, 59]
def heights_60s : List ℕ := [62, 64, 64, 65, 65, 68, 69]
def heights_70s : List ℕ := [70, 71, 73, 75, 75, 77, 78]

def all_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s

def mean_height (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / (l.length : ℚ)

theorem mean_height_of_players :
  mean_height all_heights = 68.25 :=
by
  sorry

end mean_height_of_players_l83_83694


namespace decompose_five_eighths_l83_83905

theorem decompose_five_eighths : 
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (5 : ℚ) / 8 = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) := 
by
  sorry

end decompose_five_eighths_l83_83905


namespace trip_correct_graph_l83_83557

-- Define a structure representing the trip
structure Trip :=
  (initial_city_traffic_duration : ℕ)
  (highway_duration_to_mall : ℕ)
  (shopping_duration : ℕ)
  (highway_duration_from_mall : ℕ)
  (return_city_traffic_duration : ℕ)

-- Define the conditions about the trip
def conditions (t : Trip) : Prop :=
  t.shopping_duration = 1 ∧ -- Shopping for one hour
  t.initial_city_traffic_duration < t.highway_duration_to_mall ∧ -- Travel more rapidly on the highway
  t.return_city_traffic_duration < t.highway_duration_from_mall -- Return more rapidly on the highway

-- Define the graph representation of the trip
inductive Graph
| A | B | C | D | E

-- Define the property that graph B correctly represents the trip
def correct_graph (t : Trip) (g : Graph) : Prop :=
  g = Graph.B

-- The theorem stating that given the conditions, the correct graph is B
theorem trip_correct_graph (t : Trip) (h : conditions t) : correct_graph t Graph.B :=
by
  sorry

end trip_correct_graph_l83_83557


namespace only_solution_2_pow_eq_y_sq_plus_y_plus_1_l83_83997

theorem only_solution_2_pow_eq_y_sq_plus_y_plus_1 {x y : ℕ} (h1 : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := 
by {
  sorry -- proof goes here
}

end only_solution_2_pow_eq_y_sq_plus_y_plus_1_l83_83997


namespace health_risk_probability_l83_83057

theorem health_risk_probability :
  let p := 26
  let q := 57
  p + q = 83 :=
by {
  sorry
}

end health_risk_probability_l83_83057


namespace total_sum_is_2696_l83_83432

def numbers := (100, 4900)

def harmonic_mean (a b : ℕ) : ℕ :=
  2 * a * b / (a + b)

def arithmetic_mean (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem total_sum_is_2696 : 
  harmonic_mean numbers.1 numbers.2 + arithmetic_mean numbers.1 numbers.2 = 2696 :=
by
  sorry

end total_sum_is_2696_l83_83432


namespace diamond_eq_l83_83564

noncomputable def diamond_op (a b : ℝ) (k : ℝ) : ℝ := sorry

theorem diamond_eq (x : ℝ) :
  let k := 2
  let a := 2023
  let b := 7
  let c := x
  (diamond_op a (diamond_op b c k) k = 150) ∧ 
  (∀ a b c, diamond_op a (diamond_op b c k) k = k * (diamond_op a b k) * c) ∧
  (∀ a, diamond_op a a k = k) →
  x = 150 / 2023 :=
sorry

end diamond_eq_l83_83564


namespace sum_geometric_sequence_first_10_terms_l83_83391

theorem sum_geometric_sequence_first_10_terms :
  let a₁ : ℚ := 12
  let r : ℚ := 1 / 3
  let S₁₀ : ℚ := 12 * (1 - (1 / 3)^10) / (1 - 1 / 3)
  S₁₀ = 1062864 / 59049 := by
  sorry

end sum_geometric_sequence_first_10_terms_l83_83391


namespace total_pools_l83_83275

def patsPools (numAStores numPStores poolsA ratio : ℕ) : ℕ :=
  numAStores * poolsA + numPStores * (ratio * poolsA)

theorem total_pools : 
  patsPools 6 4 200 3 = 3600 := 
by 
  sorry

end total_pools_l83_83275


namespace degree_of_divisor_polynomial_l83_83917

theorem degree_of_divisor_polynomial (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15)
  (hq : q.degree = 9)
  (hr : r.degree = 4)
  (hfdqr : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_polynomial_l83_83917


namespace cookie_division_l83_83340

theorem cookie_division (C : ℝ) (blue_fraction : ℝ := 1/4) (green_fraction_of_remaining : ℝ := 5/9)
  (remaining_fraction : ℝ := 3/4) (green_fraction : ℝ := 5/12) :
  blue_fraction + green_fraction = 2/3 := by
  sorry

end cookie_division_l83_83340


namespace john_payment_correct_l83_83278

noncomputable def camera_value : ℝ := 5000
noncomputable def base_rental_fee_per_week : ℝ := 0.10 * camera_value
noncomputable def high_demand_fee_per_week : ℝ := base_rental_fee_per_week + 0.03 * camera_value
noncomputable def low_demand_fee_per_week : ℝ := base_rental_fee_per_week - 0.02 * camera_value
noncomputable def total_rental_fee : ℝ :=
  high_demand_fee_per_week + low_demand_fee_per_week + high_demand_fee_per_week + low_demand_fee_per_week
noncomputable def insurance_fee : ℝ := 0.05 * camera_value
noncomputable def pre_tax_total_cost : ℝ := total_rental_fee + insurance_fee
noncomputable def tax : ℝ := 0.08 * pre_tax_total_cost
noncomputable def total_cost : ℝ := pre_tax_total_cost + tax

noncomputable def mike_contribution : ℝ := 0.20 * total_cost
noncomputable def sarah_contribution : ℝ := min (0.30 * total_cost) 1000
noncomputable def alex_contribution : ℝ := min (0.10 * total_cost) 700
noncomputable def total_friends_contributions : ℝ := mike_contribution + sarah_contribution + alex_contribution

noncomputable def john_final_payment : ℝ := total_cost - total_friends_contributions

theorem john_payment_correct : john_final_payment = 1015.20 :=
by
  sorry

end john_payment_correct_l83_83278


namespace max_sum_abc_divisible_by_13_l83_83623

theorem max_sum_abc_divisible_by_13 :
  ∃ (A B C : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ 13 ∣ (2000 + 100 * A + 10 * B + C) ∧ (A + B + C = 26) :=
by
  sorry

end max_sum_abc_divisible_by_13_l83_83623


namespace ladder_base_length_l83_83054

theorem ladder_base_length {a b c : ℕ} (h1 : c = 13) (h2 : b = 12) (h3 : a^2 + b^2 = c^2) :
  a = 5 := 
by 
  sorry

end ladder_base_length_l83_83054


namespace final_score_proof_l83_83042

def final_score (bullseye_points : ℕ) (miss_points : ℕ) (half_bullseye_points : ℕ) : ℕ :=
  bullseye_points + miss_points + half_bullseye_points

theorem final_score_proof : final_score 50 0 25 = 75 :=
by
  -- Considering the given conditions
  -- bullseye_points = 50
  -- miss_points = 0
  -- half_bullseye_points = half of bullseye_points = 25
  -- Summing them up: 50 + 0 + 25 = 75
  sorry

end final_score_proof_l83_83042


namespace find_missing_ratio_l83_83793

def compounded_ratio (x y : ℚ) : ℚ := (x / y) * (6 / 11) * (11 / 2)

theorem find_missing_ratio (x y : ℚ) (h : compounded_ratio x y = 2) :
  x / y = 2 / 3 :=
sorry

end find_missing_ratio_l83_83793


namespace problem_statement_l83_83333

open Real Polynomial

theorem problem_statement (a1 a2 a3 d1 d2 d3 : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
                 (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end problem_statement_l83_83333


namespace candies_eaten_l83_83568

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l83_83568


namespace average_visitors_on_other_days_l83_83755

theorem average_visitors_on_other_days 
  (avg_sunday : ℕ) (avg_month : ℕ) 
  (days_in_month : ℕ) (sundays : ℕ) (other_days : ℕ) 
  (visitors_on_other_days : ℕ) :
  avg_sunday = 510 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  (sundays * avg_sunday + other_days * visitors_on_other_days = avg_month * days_in_month) →
  visitors_on_other_days = 240 :=
by
  intros hs hm hd hsunded hotherdays heq
  sorry

end average_visitors_on_other_days_l83_83755


namespace ambiguous_dates_in_year_l83_83628

def is_ambiguous_date (m d : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 12 ∧ m ≠ d

theorem ambiguous_dates_in_year :
  ∃ n : ℕ, n = 132 ∧ (∀ m d : ℕ, is_ambiguous_date m d → n = 132) :=
sorry

end ambiguous_dates_in_year_l83_83628


namespace intersection_A_B_l83_83674

-- Definition of set A
def A (x : ℝ) : Prop := -1 < x ∧ x < 2

-- Definition of set B
def B (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 0

-- Theorem stating the intersection of sets A and B
theorem intersection_A_B (x : ℝ) : (A x ∧ B x) ↔ (-1 < x ∧ x ≤ 0) :=
by sorry

end intersection_A_B_l83_83674


namespace shirts_per_kid_l83_83756

-- Define given conditions
def n_buttons : Nat := 63
def buttons_per_shirt : Nat := 7
def n_kids : Nat := 3

-- The proof goal
theorem shirts_per_kid : (n_buttons / buttons_per_shirt) / n_kids = 3 := by
  sorry

end shirts_per_kid_l83_83756


namespace son_l83_83980

variable (S M : ℕ)

theorem son's_age
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2))
  : S = 22 :=
sorry

end son_l83_83980


namespace lana_picked_37_roses_l83_83114

def total_flowers_picked (used : ℕ) (extra : ℕ) := used + extra

def picked_roses (total : ℕ) (tulips : ℕ) := total - tulips

theorem lana_picked_37_roses :
    ∀ (tulips used extra : ℕ), tulips = 36 → used = 70 → extra = 3 → 
    picked_roses (total_flowers_picked used extra) tulips = 37 :=
by
  intros tulips used extra htulips husd hextra
  sorry

end lana_picked_37_roses_l83_83114


namespace smaller_cube_surface_area_l83_83858

theorem smaller_cube_surface_area (edge_length : ℝ) (h : edge_length = 12) :
  let sphere_diameter := edge_length
  let smaller_cube_side := sphere_diameter / Real.sqrt 3
  let surface_area := 6 * smaller_cube_side ^ 2
  surface_area = 288 := by
  sorry

end smaller_cube_surface_area_l83_83858


namespace positive_integer_solution_exists_l83_83120

theorem positive_integer_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h_eq : x^2 = y^2 + 7 * y + 6) : (x, y) = (6, 3) := 
sorry

end positive_integer_solution_exists_l83_83120


namespace find_r4_l83_83047

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4_l83_83047


namespace verify_sum_l83_83247

-- Definitions and conditions
def C : ℕ := 1
def D : ℕ := 2
def E : ℕ := 5

-- Base-6 addition representation
def is_valid_base_6_addition (a b c d : ℕ) : Prop :=
  (a + b) % 6 = c ∧ (a + b) / 6 = d

-- Given the addition problem:
def addition_problem : Prop :=
  is_valid_base_6_addition 2 5 C 0 ∧
  is_valid_base_6_addition 4 C E 0 ∧
  is_valid_base_6_addition D 2 4 0

-- Goal to prove
theorem verify_sum : addition_problem → C + D + E = 6 :=
by
  sorry

end verify_sum_l83_83247


namespace slope_of_line_l83_83480

noncomputable def slope_range : Set ℝ :=
  {α | (5 * Real.pi / 6) ≤ α ∧ α < Real.pi}

theorem slope_of_line (x a : ℝ) :
  let k := -1 / (a^2 + Real.sqrt 3)
  ∃ α ∈ slope_range, k = Real.tan α :=
sorry

end slope_of_line_l83_83480


namespace alpha_minus_beta_l83_83287

-- Providing the conditions
variable (α β : ℝ)
variable (hα1 : 0 < α ∧ α < Real.pi / 2)
variable (hβ1 : 0 < β ∧ β < Real.pi / 2)
variable (hα2 : Real.tan α = 4 / 3)
variable (hβ2 : Real.tan β = 1 / 7)

-- The goal is to show that α - β = π / 4 given the conditions
theorem alpha_minus_beta :
  α - β = Real.pi / 4 := by
  sorry

end alpha_minus_beta_l83_83287


namespace bowling_ball_weight_l83_83489

theorem bowling_ball_weight :
  ∃ (b : ℝ) (c : ℝ),
    8 * b = 5 * c ∧
    4 * c = 100 ∧
    b = 15.625 :=
by 
  sorry

end bowling_ball_weight_l83_83489


namespace infinite_solutions_of_linear_system_l83_83150

theorem infinite_solutions_of_linear_system :
  ∀ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) → ∃ (k : ℝ), x = (3 * k + 5) / 2 :=
by
  sorry

end infinite_solutions_of_linear_system_l83_83150


namespace mn_value_l83_83837

theorem mn_value (m n : ℝ) 
  (h1 : m^2 + 1 = 4)
  (h2 : 2 * m + n = 0) :
  m * n = -6 := 
sorry

end mn_value_l83_83837


namespace max_positive_integer_difference_l83_83533

theorem max_positive_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) : ∃ d : ℕ, d = 6 :=
by
  sorry

end max_positive_integer_difference_l83_83533


namespace each_child_gets_twelve_cupcakes_l83_83154

def total_cupcakes := 96
def children := 8
def cupcakes_per_child : ℕ := total_cupcakes / children

theorem each_child_gets_twelve_cupcakes :
  cupcakes_per_child = 12 :=
by
  sorry

end each_child_gets_twelve_cupcakes_l83_83154


namespace range_of_a_l83_83757

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 3 * x + 2 = 0) → ∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end range_of_a_l83_83757


namespace max_single_player_salary_l83_83373

theorem max_single_player_salary (n : ℕ) (m : ℕ) (T : ℕ) (n_pos : n = 18) (m_pos : m = 20000) (T_pos : T = 800000) :
  ∃ x : ℕ, (∀ y : ℕ, y ≤ x → y ≤ 460000) ∧ (17 * m + x ≤ T) :=
by
  sorry

end max_single_player_salary_l83_83373


namespace arithmetic_sequence_geometric_subsequence_l83_83232

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℤ) (a1 a3 a4 : ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 2)
  (h3 : a1 = a 1)
  (h4 : a3 = a 3)
  (h5 : a4 = a 4)
  (h6 : a3^2 = a1 * a4) :
  a 6 = 2 := 
by
  sorry

end arithmetic_sequence_geometric_subsequence_l83_83232


namespace gcd_max_value_l83_83877

noncomputable def max_gcd (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 1

theorem gcd_max_value :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → gcd (13 * m + 4) (7 * m + 2) ≤ max_gcd m) ∧
              (∀ m : ℕ, m > 0 → max_gcd m ≤ 2) :=
by {
  sorry
}

end gcd_max_value_l83_83877


namespace rectangle_area_l83_83835

-- Declare the given conditions
def circle_radius : ℝ := 5
def rectangle_width : ℝ := 2 * circle_radius
def length_to_width_ratio : ℝ := 2

-- Given that the length to width ratio is 2:1, calculate the length
def rectangle_length : ℝ := length_to_width_ratio * rectangle_width

-- Define the statement we need to prove
theorem rectangle_area :
  rectangle_length * rectangle_width = 200 :=
by
  sorry

end rectangle_area_l83_83835


namespace min_value_of_frac_sum_l83_83157

theorem min_value_of_frac_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 2) :
  (1 / a + 2 / b) = 9 / 2 :=
sorry

end min_value_of_frac_sum_l83_83157


namespace cost_price_of_table_l83_83215

theorem cost_price_of_table (C S : ℝ) (h1 : S = 1.25 * C) (h2 : S = 4800) : C = 3840 := 
by 
  sorry

end cost_price_of_table_l83_83215


namespace necessary_but_not_sufficient_condition_for_x_equals_0_l83_83164

theorem necessary_but_not_sufficient_condition_for_x_equals_0 (x : ℝ) :
  ((2 * x - 1) * x = 0 → x = 0 ∨ x = 1 / 2) ∧ (x = 0 → (2 * x - 1) * x = 0) ∧ ¬((2 * x - 1) * x = 0 → x = 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_x_equals_0_l83_83164


namespace find_c_l83_83635

theorem find_c
  (m b d c : ℝ)
  (h : m = b * d * c / (d + c)) :
  c = m * d / (b * d - m) :=
sorry

end find_c_l83_83635


namespace mass_of_fourth_metal_l83_83641

theorem mass_of_fourth_metal 
  (m1 m2 m3 m4 : ℝ)
  (total_mass : m1 + m2 + m3 + m4 = 20)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = 3/4 * m3)
  (h3 : m3 = 5/6 * m4) :
  m4 = 20 * (48 / 163) :=
sorry

end mass_of_fourth_metal_l83_83641


namespace ticket_cost_before_rally_l83_83426

-- We define the variables and constants given in the problem
def total_attendance : ℕ := 750
def tickets_before_rally : ℕ := 475
def tickets_at_door : ℕ := total_attendance - tickets_before_rally
def cost_at_door : ℝ := 2.75
def total_receipts : ℝ := 1706.25

-- Problem statement: Prove that the cost of each ticket bought before the rally (x) is 2 dollars.
theorem ticket_cost_before_rally (x : ℝ) 
  (h₁ : tickets_before_rally * x + tickets_at_door * cost_at_door = total_receipts) :
  x = 2 :=
by
  sorry

end ticket_cost_before_rally_l83_83426


namespace ellipse_with_foci_on_x_axis_l83_83423

theorem ellipse_with_foci_on_x_axis {a : ℝ} (h1 : a - 5 > 0) (h2 : 2 > 0) (h3 : a - 5 > 2) :
  a > 7 :=
by
  sorry

end ellipse_with_foci_on_x_axis_l83_83423


namespace triangle_angle_ABC_l83_83790

theorem triangle_angle_ABC
  (ABD CBD ABC : ℝ) 
  (h1 : ABD = 70)
  (h2 : ABD + CBD + ABC = 200)
  (h3 : CBD = 60) : ABC = 70 := 
sorry

end triangle_angle_ABC_l83_83790


namespace intersection_M_N_l83_83473

def M : Set ℤ := { -2, -1, 0, 1, 2 }
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N :
  M ∩ N = { -2 } :=
by
  sorry

end intersection_M_N_l83_83473


namespace correct_order_of_numbers_l83_83695

theorem correct_order_of_numbers :
  let a := (4 / 5 : ℝ)
  let b := (81 / 100 : ℝ)
  let c := 0.801
  (a ≤ c ∧ c ≤ b) :=
by
  sorry

end correct_order_of_numbers_l83_83695


namespace find_y_coordinate_l83_83826

-- Define points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-2, 2)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the property that a point P satisfies PA + PD = PB + PC = 10
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PD := Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  PA + PD = 10 ∧ PB + PC = 10

-- Lean statement to prove the y-coordinate of P that satisfies the condition
theorem find_y_coordinate :
  ∃ (P : ℝ × ℝ), satisfies_condition P ∧ ∃ (a b c d : ℕ), a = 0 ∧ b = 1 ∧ c = 21 ∧ d = 3 ∧ P.2 = (14 + Real.sqrt 21) / 3 ∧ a + b + c + d = 25 :=
by
  sorry

end find_y_coordinate_l83_83826


namespace beth_sold_coins_l83_83748

theorem beth_sold_coins :
  let initial_coins := 125
  let gift_coins := 35
  let total_coins := initial_coins + gift_coins
  let sold_coins := total_coins / 2
  sold_coins = 80 :=
by
  sorry

end beth_sold_coins_l83_83748


namespace exponent_product_to_sixth_power_l83_83366

theorem exponent_product_to_sixth_power :
  ∃ n : ℤ, 3^(12) * 3^(18) = n^6 ∧ n = 243 :=
by
  use 243
  sorry

end exponent_product_to_sixth_power_l83_83366


namespace comparison_of_a_b_c_l83_83610

noncomputable def a : ℝ := 2018 ^ (1 / 2018)
noncomputable def b : ℝ := Real.logb 2017 (Real.sqrt 2018)
noncomputable def c : ℝ := Real.logb 2018 (Real.sqrt 2017)

theorem comparison_of_a_b_c :
  a > b ∧ b > c :=
by
  -- Definitions
  have def_a : a = 2018 ^ (1 / 2018) := rfl
  have def_b : b = Real.logb 2017 (Real.sqrt 2018) := rfl
  have def_c : c = Real.logb 2018 (Real.sqrt 2017) := rfl

  -- Sorry is added to skip the proof
  sorry

end comparison_of_a_b_c_l83_83610


namespace profit_function_equation_maximum_profit_l83_83336

noncomputable def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
noncomputable def sales_revenue (x : ℝ) : ℝ := 18*x
noncomputable def production_profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_function_equation (x : ℝ) : production_profit x = -x^3 + 24*x^2 - 45*x - 10 :=
  by
    unfold production_profit sales_revenue production_cost
    sorry

theorem maximum_profit : (production_profit 15 = 1340) ∧ ∀ x, production_profit 15 ≥ production_profit x :=
  by
    sorry

end profit_function_equation_maximum_profit_l83_83336


namespace sallys_woodworking_llc_reimbursement_l83_83617

/-
Conditions:
1. Remy paid $20,700 for 150 pieces of furniture.
2. The cost of a piece of furniture is $134.
-/
def reimbursement_amount (pieces_paid : ℕ) (total_paid : ℕ) (price_per_piece : ℕ) : ℕ :=
  total_paid - (pieces_paid * price_per_piece)

theorem sallys_woodworking_llc_reimbursement :
  reimbursement_amount 150 20700 134 = 600 :=
by 
  sorry

end sallys_woodworking_llc_reimbursement_l83_83617


namespace cos_diff_of_symmetric_sines_l83_83510

theorem cos_diff_of_symmetric_sines (a β : Real) (h1 : Real.sin a = 1 / 3) 
  (h2 : Real.sin β = 1 / 3) (h3 : Real.cos a = -Real.cos β) : 
  Real.cos (a - β) = -7 / 9 := by
  sorry

end cos_diff_of_symmetric_sines_l83_83510


namespace reading_rate_l83_83448

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l83_83448


namespace sum_of_arithmetic_sequence_l83_83216

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) (S5 : S 5 = 30) (S10 : S 10 = 110) : S 15 = 240 :=
by
  sorry

end sum_of_arithmetic_sequence_l83_83216


namespace left_building_percentage_l83_83703

theorem left_building_percentage (L R : ℝ)
  (middle_building_height : ℝ := 100)
  (total_height : ℝ := 340)
  (condition1 : L + middle_building_height + R = total_height)
  (condition2 : R = L + middle_building_height - 20) :
  (L / middle_building_height) * 100 = 80 := by
  sorry

end left_building_percentage_l83_83703


namespace distance_from_dorm_to_city_l83_83230

theorem distance_from_dorm_to_city (D : ℚ) (h1 : (1/3) * D = (1/3) * D) (h2 : (3/5) * D = (3/5) * D) (h3 : D - ((1 / 3) * D + (3 / 5) * D) = 2) :
  D = 30 := 
by sorry

end distance_from_dorm_to_city_l83_83230


namespace smallest_gcd_six_l83_83021

theorem smallest_gcd_six (x : ℕ) (hx1 : 70 ≤ x) (hx2 : x ≤ 90) (hx3 : Nat.gcd 24 x = 6) : x = 78 :=
by
  sorry

end smallest_gcd_six_l83_83021


namespace sum_of_center_coords_l83_83303

theorem sum_of_center_coords (x y : ℝ) :
  (∃ k : ℝ, (x + 2)^2 + (y + 3)^2 = k ∧ (x^2 + y^2 = -4 * x - 6 * y + 5)) -> x + y = -5 :=
by
sorry

end sum_of_center_coords_l83_83303


namespace height_of_smaller_cone_l83_83702

theorem height_of_smaller_cone (h_frustum : ℝ) (area_lower_base area_upper_base : ℝ) 
  (h_frustum_eq : h_frustum = 18) 
  (area_lower_base_eq : area_lower_base = 144 * Real.pi) 
  (area_upper_base_eq : area_upper_base = 16 * Real.pi) : 
  ∃ (x : ℝ), x = 9 :=
by
  -- Definitions and assumptions go here
  sorry

end height_of_smaller_cone_l83_83702


namespace find_x_l83_83624

noncomputable def satisfy_equation (x : ℝ) : Prop :=
  8 / (Real.sqrt (x - 10) - 10) +
  2 / (Real.sqrt (x - 10) - 5) +
  10 / (Real.sqrt (x - 10) + 5) +
  16 / (Real.sqrt (x - 10) + 10) = 0

theorem find_x : ∃ x : ℝ, satisfy_equation x ∧ x = 60 := sorry

end find_x_l83_83624


namespace min_abs_sum_of_products_l83_83153

noncomputable def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

theorem min_abs_sum_of_products (w : Fin 4 → ℝ) (h_roots : ∀ i, g (w i) = 0)
  : ∃ a b c d : Fin 4, a ≠ b ∧ c ≠ d ∧ (∀ i j, i ≠ j → a ≠ i ∧ b ≠ i ∧ c ≠ i ∧ d ≠ i → a ≠ j ∧ b ≠ j ∧ c ≠ j ∧ d ≠ j) ∧
    |w a * w b + w c * w d| = 6 :=
sorry

end min_abs_sum_of_products_l83_83153


namespace candy_bar_cost_correct_l83_83142

def quarters : ℕ := 4
def dimes : ℕ := 3
def nickel : ℕ := 1
def change_received : ℕ := 4

def total_paid : ℕ :=
  (quarters * 25) + (dimes * 10) + (nickel * 5)

def candy_bar_cost : ℕ :=
  total_paid - change_received

theorem candy_bar_cost_correct : candy_bar_cost = 131 := by
  sorry

end candy_bar_cost_correct_l83_83142


namespace least_pounds_of_sugar_l83_83915

theorem least_pounds_of_sugar :
  ∃ s : ℝ, (∀ f : ℝ, (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s = 4) :=
by {
    use 4,
    sorry
}

end least_pounds_of_sugar_l83_83915


namespace base8_1724_to_base10_l83_83080

/-- Define the base conversion function from base-eight to base-ten -/
def base8_to_base10 (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- Base-eight representation conditions for the number 1724 -/
def base8_1724_digits := (1, 7, 2, 4)

/-- Prove the base-ten equivalent of the base-eight number 1724 is 980 -/
theorem base8_1724_to_base10 : base8_to_base10 1 7 2 4 = 980 :=
  by
    -- skipping the proof; just state that it is a theorem to be proved.
    sorry

end base8_1724_to_base10_l83_83080


namespace precision_tens_place_l83_83352

-- Given
def given_number : ℝ := 4.028 * (10 ^ 5)

-- Prove that the precision of the given_number is to the tens place.
theorem precision_tens_place : true := by
  -- Proof goes here
  sorry

end precision_tens_place_l83_83352


namespace time_difference_l83_83377

-- Definitions
def time_chinese : ℕ := 5
def time_english : ℕ := 7

-- Statement to prove
theorem time_difference : time_english - time_chinese = 2 := by
  -- Proof goes here
  sorry

end time_difference_l83_83377


namespace chantel_bracelets_final_count_l83_83869

-- Definitions for conditions
def bracelets_made_days (days : ℕ) (bracelets_per_day : ℕ) : ℕ :=
  days * bracelets_per_day

def initial_bracelets (days1 : ℕ) (bracelets_per_day1 : ℕ) : ℕ :=
  bracelets_made_days days1 bracelets_per_day1

def after_giving_away1 (initial_count : ℕ) (given_away1 : ℕ) : ℕ :=
  initial_count - given_away1

def additional_bracelets (days2 : ℕ) (bracelets_per_day2 : ℕ) : ℕ :=
  bracelets_made_days days2 bracelets_per_day2

def final_count (remaining_after_giving1 : ℕ) (additional_made : ℕ) (given_away2 : ℕ) : ℕ :=
  remaining_after_giving1 + additional_made - given_away2

-- Main theorem statement
theorem chantel_bracelets_final_count :
  ∀ (days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 : ℕ),
  days1 = 5 →
  bracelets_per_day1 = 2 →
  given_away1 = 3 →
  days2 = 4 →
  bracelets_per_day2 = 3 →
  given_away2 = 6 →
  final_count (after_giving_away1 (initial_bracelets days1 bracelets_per_day1) given_away1)
              (additional_bracelets days2 bracelets_per_day2)
              given_away2 = 13 :=
by
  intros days1 days2 bracelets_per_day1 bracelets_per_day2 given_away1 given_away2 hdays1 hbracelets_per_day1 hgiven_away1 hdays2 hbracelets_per_day2 hgiven_away2
  -- Proof is not required, so we use sorry
  sorry

end chantel_bracelets_final_count_l83_83869


namespace number_of_valid_pairs_l83_83386

theorem number_of_valid_pairs (m n : ℕ) (h1 : n > m) (h2 : 3 * (m - 4) * (n - 4) = m * n) : 
  (m, n) = (7, 18) ∨ (m, n) = (8, 12) ∨ (m, n) = (9, 10) ∨ (m-6) * (n-6) = 12 := sorry

end number_of_valid_pairs_l83_83386


namespace sarah_average_speed_l83_83604

theorem sarah_average_speed :
  ∀ (total_distance race_time : ℕ) 
    (sadie_speed sadie_time ariana_speed ariana_time : ℕ)
    (distance_sarah speed_sarah time_sarah : ℚ),
  sadie_speed = 3 → 
  sadie_time = 2 → 
  ariana_speed = 6 → 
  ariana_time = 1 / 2 → 
  race_time = 9 / 2 → 
  total_distance = 17 →
  distance_sarah = total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time) →
  time_sarah = race_time - (sadie_time + ariana_time) →
  speed_sarah = distance_sarah / time_sarah →
  speed_sarah = 4 :=
by
  intros total_distance race_time sadie_speed sadie_time ariana_speed ariana_time distance_sarah speed_sarah time_sarah
  intros sadie_speed_eq sadie_time_eq ariana_speed_eq ariana_time_eq race_time_eq total_distance_eq distance_sarah_eq time_sarah_eq speed_sarah_eq
  sorry

end sarah_average_speed_l83_83604


namespace complement_union_l83_83738

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4}

-- Define the set S
def S : Set ℕ := {1, 3}

-- Define the set T
def T : Set ℕ := {4}

-- Define the complement of S in I
def complement_I_S : Set ℕ := I \ S

-- State the theorem to be proved
theorem complement_union : (complement_I_S ∪ T) = {2, 4} := by
  sorry

end complement_union_l83_83738


namespace remainder_of_7_pow_145_mod_12_l83_83972

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l83_83972


namespace range_of_z_l83_83068

theorem range_of_z (α β : ℝ) (z : ℝ) (h1 : -2 < α) (h2 : α ≤ 3) (h3 : 2 < β) (h4 : β ≤ 4) (h5 : z = 2 * α - (1 / 2) * β) :
  -6 < z ∧ z < 5 :=
by
  sorry

end range_of_z_l83_83068


namespace f_at_3_l83_83840

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 5

theorem f_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 11 :=
by
  sorry

end f_at_3_l83_83840


namespace proposition_P_l83_83779

theorem proposition_P (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) := 
by 
  sorry

end proposition_P_l83_83779


namespace log_27_3_l83_83985

noncomputable def log_base (a b : ℝ) : ℝ := Real.log a / Real.log b

theorem log_27_3 :
  log_base 3 27 = 1 / 3 := by
  sorry

end log_27_3_l83_83985


namespace diameter_large_circle_correct_l83_83898

noncomputable def diameter_of_large_circle : ℝ :=
  2 * (Real.sqrt 17 + 4)

theorem diameter_large_circle_correct :
  ∃ (d : ℝ), (∀ (r : ℝ), r = Real.sqrt 17 + 4 → d = 2 * r) ∧ d = diameter_of_large_circle := by
    sorry

end diameter_large_circle_correct_l83_83898


namespace smallest_three_digit_multiple_of_17_l83_83743

theorem smallest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m :=
by
  sorry

end smallest_three_digit_multiple_of_17_l83_83743


namespace eventually_periodic_sequence_l83_83934

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h1 : ∀ n m : ℕ, 0 < n → 0 < m → a (n + 2 * m) ∣ (a n + a (n + m)))
  : ∃ N d : ℕ, 0 < N ∧ 0 < d ∧ ∀ n > N, a n = a (n + d) :=
sorry

end eventually_periodic_sequence_l83_83934


namespace value_of_x_l83_83271

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end value_of_x_l83_83271


namespace sequence_a_n_a_99_value_l83_83277

theorem sequence_a_n_a_99_value :
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ (∀ n, 2 * (a (n + 1)) - 2 * (a n) = 1) ∧ a 99 = 52 :=
by {
  sorry
}

end sequence_a_n_a_99_value_l83_83277


namespace x0_in_M_implies_x0_in_N_l83_83462

def M : Set ℝ := {x | ∃ (k : ℤ), x = k + 1 / 2}
def N : Set ℝ := {x | ∃ (k : ℤ), x = k / 2 + 1}

theorem x0_in_M_implies_x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := 
sorry

end x0_in_M_implies_x0_in_N_l83_83462


namespace PQ_parallel_to_AB_3_times_l83_83369

-- Definitions for the problem
structure Rectangle :=
  (A B C D : Type)
  (AB AD : ℝ)
  (P Q : ℝ → ℝ)
  (P_speed Q_speed : ℝ)
  (time : ℝ)

noncomputable def rectangle_properties (R : Rectangle) : Prop :=
  R.AB = 4 ∧
  R.AD = 12 ∧
  ∀ t, 0 ≤ t → t ≤ 12 → R.P t = t ∧  -- P moves from A to D at 1 cm/s
  R.Q_speed = 3 ∧                     -- Q moves at 3 cm/s
  ∀ t, R.Q t = R.Q_speed * t ∧             -- Q moves from C to B and back
  ∃ s1 s2 s3, R.P s1 = 4 ∧ R.P s2 = 8 ∧ R.P s3 = 12 ∧
  (R.Q s1 = 3 ∨ R.Q s1 = 1) ∧
  (R.Q s2 = 6 ∨ R.Q s2 = 2) ∧
  (R.Q s3 = 9 ∨ R.Q s3 = 0)

theorem PQ_parallel_to_AB_3_times : 
  ∀ (R : Rectangle), rectangle_properties R → 
  ∃ (times : ℕ), times = 3 :=
by
  sorry

end PQ_parallel_to_AB_3_times_l83_83369


namespace area_in_sq_yds_l83_83060

-- Definitions based on conditions
def side_length_ft : ℕ := 9
def sq_ft_per_sq_yd : ℕ := 9

-- Statement to prove
theorem area_in_sq_yds : (side_length_ft * side_length_ft) / sq_ft_per_sq_yd = 9 :=
by
  sorry

end area_in_sq_yds_l83_83060


namespace Humphrey_birds_l83_83308

-- Definitions for the given conditions:
def Marcus_birds : ℕ := 7
def Darrel_birds : ℕ := 9
def average_birds : ℕ := 9
def number_of_people : ℕ := 3

-- Proof statement
theorem Humphrey_birds : ∀ x : ℕ, (average_birds * number_of_people = Marcus_birds + Darrel_birds + x) → x = 11 :=
by
  intro x h
  sorry

end Humphrey_birds_l83_83308


namespace option_d_correct_factorization_l83_83940

theorem option_d_correct_factorization (x : ℝ) : 
  -8 * x ^ 2 + 8 * x - 2 = -2 * (2 * x - 1) ^ 2 :=
by 
  sorry

end option_d_correct_factorization_l83_83940


namespace cyclists_original_number_l83_83745

theorem cyclists_original_number (x : ℕ) (h : x > 2) : 
  (80 / (x - 2 : ℕ) = 80 / x + 2) → x = 10 :=
by
  sorry

end cyclists_original_number_l83_83745


namespace larger_exceeds_smaller_by_16_l83_83037

-- Define the smaller number S and the larger number L in terms of the ratio 7:11
def S : ℕ := 28
def L : ℕ := (11 * S) / 7

-- State the theorem that the larger number exceeds the smaller number by 16
theorem larger_exceeds_smaller_by_16 : L - S = 16 :=
by
  -- Proof steps will go here
  sorry

end larger_exceeds_smaller_by_16_l83_83037


namespace fraction_division_correct_l83_83136

theorem fraction_division_correct :
  (5/6 : ℚ) / (7/9) / (11/13) = 195/154 := 
by {
  sorry
}

end fraction_division_correct_l83_83136


namespace weight_replacement_proof_l83_83900

noncomputable def weight_of_replaced_person (increase_in_average_weight new_person_weight : ℝ) : ℝ :=
  new_person_weight - (5 * increase_in_average_weight)

theorem weight_replacement_proof (h1 : ∀ w : ℝ, increase_in_average_weight = 5.5) (h2 : new_person_weight = 95.5) :
  weight_of_replaced_person 5.5 95.5 = 68 := by
  sorry

end weight_replacement_proof_l83_83900


namespace exists_n_sum_three_digit_identical_digit_l83_83228

theorem exists_n_sum_three_digit_identical_digit:
  ∃ (n : ℕ), (∃ (k : ℕ), (k ≥ 1 ∧ k ≤ 9) ∧ (n*(n+1)/2 = 111*k)) ∧ n = 36 :=
by
  -- Placeholder for the proof
  sorry

end exists_n_sum_three_digit_identical_digit_l83_83228


namespace least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l83_83955

theorem least_prime_factor_of_5_to_the_3_minus_5_to_the_2 : 
  Nat.minFac (5^3 - 5^2) = 2 := by
  sorry

end least_prime_factor_of_5_to_the_3_minus_5_to_the_2_l83_83955


namespace find_2a_2b_2c_2d_l83_83943

open Int

theorem find_2a_2b_2c_2d (a b c d : ℤ) 
  (h1 : a - b + c = 7) 
  (h2 : b - c + d = 8) 
  (h3 : c - d + a = 4) 
  (h4 : d - a + b = 1) : 
  2*a + 2*b + 2*c + 2*d = 20 := 
sorry

end find_2a_2b_2c_2d_l83_83943


namespace ratio_of_pieces_l83_83052

theorem ratio_of_pieces (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 ∧ shorter_piece = 20 → shorter_piece / (total_length - shorter_piece) = 1 / 2 :=
by
  sorry

end ratio_of_pieces_l83_83052


namespace find_factor_l83_83893

theorem find_factor (f : ℝ) : (120 * f - 138 = 102) → f = 2 :=
by
  sorry

end find_factor_l83_83893


namespace luke_fish_fillets_l83_83935

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l83_83935


namespace project_completion_time_l83_83620

theorem project_completion_time 
    (w₁ w₂ : ℕ) 
    (d₁ d₂ : ℕ) 
    (fraction₁ fraction₂ : ℝ)
    (h_work_fraction : fraction₁ = 1/2)
    (h_work_time : d₁ = 6)
    (h_first_workforce : w₁ = 90)
    (h_second_workforce : w₂ = 60)
    (h_fraction_done_by_first_team : w₁ * d₁ * (1 / 1080) = fraction₁)
    (h_fraction_done_by_second_team : w₂ * d₂ * (1 / 1080) = fraction₂)
    (h_total_fraction : fraction₂ = 1 - fraction₁) :
    d₂ = 9 :=
by 
  sorry

end project_completion_time_l83_83620


namespace power_exponent_multiplication_l83_83428

variable (a : ℝ)

theorem power_exponent_multiplication : (a^3)^2 = a^6 := sorry

end power_exponent_multiplication_l83_83428


namespace quadratic_residue_one_mod_p_l83_83918

theorem quadratic_residue_one_mod_p (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ) :
  (a^2 % p = 1 % p) ↔ (a % p = 1 % p ∨ a % p = (p-1) % p) :=
sorry

end quadratic_residue_one_mod_p_l83_83918


namespace joan_gave_27_apples_l83_83776

theorem joan_gave_27_apples (total_apples : ℕ) (current_apples : ℕ)
  (h1 : total_apples = 43) 
  (h2 : current_apples = 16) : 
  total_apples - current_apples = 27 := 
by
  sorry

end joan_gave_27_apples_l83_83776


namespace problem_equiv_proof_l83_83802

noncomputable def simplify_and_evaluate (a : ℝ) :=
  ((a + 1) / (a + 2) + 1 / (a - 2)) / (2 / (a^2 - 4))

theorem problem_equiv_proof :
  simplify_and_evaluate (Real.sqrt 2) = 1 := 
  sorry

end problem_equiv_proof_l83_83802


namespace smallest_integer_l83_83532

-- Given positive integer M such that
def satisfies_conditions (M : ℕ) : Prop :=
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  M % 8 = 7 ∧
  M % 9 = 8 ∧
  M % 10 = 9 ∧
  M % 11 = 10 ∧
  M % 13 = 12

-- The main theorem to prove
theorem smallest_integer (M : ℕ) (h : satisfies_conditions M) : M = 360359 :=
  sorry

end smallest_integer_l83_83532


namespace common_ratio_of_geometric_seq_l83_83670

variable {α : Type} [LinearOrderedField α] 
variables (a d : α) (h₁ : d ≠ 0) (h₂ : (a + 2 * d) / (a + d) = (a + 5 * d) / (a + 2 * d))

theorem common_ratio_of_geometric_seq : (a + 2 * d) / (a + d) = 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l83_83670


namespace possible_values_of_a_l83_83644

theorem possible_values_of_a :
  (∀ x, (x^2 - 3 * x + 2 = 0) → (ax - 2 = 0)) → (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  intro h
  sorry

end possible_values_of_a_l83_83644


namespace dog_food_bags_needed_l83_83959

theorem dog_food_bags_needed
  (cup_weight: ℝ)
  (dogs: ℕ)
  (cups_per_day: ℕ)
  (days_in_month: ℕ)
  (bag_weight: ℝ)
  (hcw: cup_weight = 1/4)
  (hd: dogs = 2)
  (hcd: cups_per_day = 6 * 2)
  (hdm: days_in_month = 30)
  (hbw: bag_weight = 20) :
  (dogs * cups_per_day * days_in_month * cup_weight) / bag_weight = 9 :=
by
  sorry

end dog_food_bags_needed_l83_83959


namespace total_books_l83_83637

-- Defining the conditions
def darla_books := 6
def katie_books := darla_books / 2
def combined_books := darla_books + katie_books
def gary_books := 5 * combined_books

-- Statement to prove
theorem total_books : darla_books + katie_books + gary_books = 54 := by
  sorry

end total_books_l83_83637


namespace intersection_A_B_l83_83343

-- Define the sets A and B based on given conditions
def A : Set ℝ := { x | x^2 ≤ 1 }
def B : Set ℝ := { x | (x - 2) / x ≤ 0 }

-- State the proof problem
theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end intersection_A_B_l83_83343


namespace constant_value_AP_AQ_l83_83648

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end constant_value_AP_AQ_l83_83648


namespace example_theorem_l83_83298

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end example_theorem_l83_83298


namespace valid_a_value_l83_83268

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end valid_a_value_l83_83268


namespace ravi_jump_height_l83_83505

theorem ravi_jump_height (j1 j2 j3 : ℕ) (average : ℕ) (ravi_jump_height : ℕ) (h : j1 = 23 ∧ j2 = 27 ∧ j3 = 28) 
  (ha : average = (j1 + j2 + j3) / 3) (hr : ravi_jump_height = 3 * average / 2) : ravi_jump_height = 39 :=
by
  sorry

end ravi_jump_height_l83_83505


namespace nuts_mixture_weight_l83_83710

variable (m n : ℕ)
variable (weight_almonds per_part total_weight : ℝ)

theorem nuts_mixture_weight (h1 : m = 5) (h2 : n = 2) (h3 : weight_almonds = 250) 
  (h4 : per_part = weight_almonds / m) (h5 : total_weight = per_part * (m + n)) : 
  total_weight = 350 := by
  sorry

end nuts_mixture_weight_l83_83710


namespace range_of_m_l83_83904

theorem range_of_m (m : ℝ) (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 + x2 = -(m + 2) ∧ x1 * x2 = m + 5) : -5 < m ∧ m < -2 := 
sorry

end range_of_m_l83_83904


namespace find_x_l83_83583

theorem find_x (x : ℝ) (h1 : ⌈x⌉ * x = 210) (h2 : ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) (h3 : 0 < x) : x = 14 :=
sorry

end find_x_l83_83583


namespace sandy_correct_value_t_l83_83947

theorem sandy_correct_value_t (p q r s : ℕ) (t : ℕ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8)
  (expr1 : p + q - r + s - t = p + (q - (r + (s - t)))) :
  t = 8 := 
by
  sorry

end sandy_correct_value_t_l83_83947


namespace max_cosine_value_l83_83907

theorem max_cosine_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 1 ≥ Real.cos a :=
sorry

end max_cosine_value_l83_83907


namespace pool_depth_multiple_l83_83513

theorem pool_depth_multiple
  (johns_pool : ℕ)
  (sarahs_pool : ℕ)
  (h1 : johns_pool = 15)
  (h2 : sarahs_pool = 5)
  (h3 : johns_pool = x * sarahs_pool + 5) :
  x = 2 := by
  sorry

end pool_depth_multiple_l83_83513


namespace binary_difference_l83_83178

theorem binary_difference (n : ℕ) (b_2 : List ℕ) (x y : ℕ) (h1 : n = 157)
  (h2 : b_2 = [1, 0, 0, 1, 1, 1, 0, 1])
  (hx : x = b_2.count 0)
  (hy : y = b_2.count 1) : y - x = 2 := by
  sorry

end binary_difference_l83_83178


namespace scarlet_savings_l83_83973

theorem scarlet_savings :
  ∀ (initial_savings cost_of_earrings cost_of_necklace amount_left : ℕ),
    initial_savings = 80 →
    cost_of_earrings = 23 →
    cost_of_necklace = 48 →
    amount_left = initial_savings - (cost_of_earrings + cost_of_necklace) →
    amount_left = 9 :=
by
  intros initial_savings cost_of_earrings cost_of_necklace amount_left h_is h_earrings h_necklace h_left
  rw [h_is, h_earrings, h_necklace] at h_left
  exact h_left

end scarlet_savings_l83_83973


namespace solve_equation_l83_83327

theorem solve_equation (x : ℝ) : 
  (1 / (x^2 + 13*x - 16) + 1 / (x^2 + 4*x - 16) + 1 / (x^2 - 15*x - 16) = 0) ↔ 
    (x = 1 ∨ x = -16 ∨ x = 4 ∨ x = -4) :=
by
  sorry

end solve_equation_l83_83327


namespace parabola_directrix_eq_l83_83563

theorem parabola_directrix_eq (x : ℝ) : 
  (∀ y : ℝ, y = 3 * x^2 - 6 * x + 2 → True) →
  y = -13/12 := 
  sorry

end parabola_directrix_eq_l83_83563


namespace license_plate_count_correct_l83_83407

-- Define the number of choices for digits and letters
def num_digit_choices : ℕ := 10^3
def num_letter_block_choices : ℕ := 26^3
def num_position_choices : ℕ := 4

-- Compute the total number of distinct license plates
def total_license_plates : ℕ := num_position_choices * num_digit_choices * num_letter_block_choices

-- The proof statement
theorem license_plate_count_correct : total_license_plates = 70304000 := by
  -- This proof is left as an exercise
  sorry

end license_plate_count_correct_l83_83407


namespace part1_part2_l83_83315

def P (x : ℝ) : Prop := |x - 1| > 2
def S (x : ℝ) (a : ℝ) : Prop := x^2 - (a + 1) * x + a > 0

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, S x a ↔ x < 1 ∨ x > 2 :=
by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 1) : ∀ x, (P x → S x a) → (-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l83_83315


namespace log_domain_is_pos_real_l83_83314

noncomputable def domain_log : Set ℝ := {x | x > 0}
noncomputable def domain_reciprocal : Set ℝ := {x | x ≠ 0}
noncomputable def domain_sqrt : Set ℝ := {x | x ≥ 0}
noncomputable def domain_exp : Set ℝ := {x | true}

theorem log_domain_is_pos_real :
  (domain_log = {x : ℝ | 0 < x}) ∧ 
  (domain_reciprocal = {x : ℝ | x ≠ 0}) ∧ 
  (domain_sqrt = {x : ℝ | 0 ≤ x}) ∧ 
  (domain_exp = {x : ℝ | true}) →
  domain_log = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end log_domain_is_pos_real_l83_83314


namespace maximum_obtuse_vectors_l83_83661

-- Definition: A vector in 3D space
structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition: Dot product of two vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition: Two vectors form an obtuse angle if their dot product is negative
def obtuse_angle (v1 v2 : Vector3D) : Prop :=
  dot_product v1 v2 < 0

-- Main statement incorporating the conditions and the conclusion
theorem maximum_obtuse_vectors :
  ∀ (v1 v2 v3 v4 : Vector3D),
  (obtuse_angle v1 v2) →
  (obtuse_angle v1 v3) →
  (obtuse_angle v1 v4) →
  (obtuse_angle v2 v3) →
  (obtuse_angle v2 v4) →
  (obtuse_angle v3 v4) →
  -- Conclusion: At most 4 vectors can be pairwise obtuse
  ∃ (v5 : Vector3D),
  ¬ (obtuse_angle v1 v5 ∧ obtuse_angle v2 v5 ∧ obtuse_angle v3 v5 ∧ obtuse_angle v4 v5) :=
sorry

end maximum_obtuse_vectors_l83_83661


namespace sin_squared_alpha_plus_pi_over_4_l83_83666

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + Real.pi / 4) ^ 2 = 5 / 6 := 
sorry

end sin_squared_alpha_plus_pi_over_4_l83_83666


namespace roots_sum_product_l83_83645

theorem roots_sum_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : ∀ x : ℝ, x^2 - p*x - 2*q = 0) :
  (p + q = p) ∧ (p * q = -2*q) :=
by
  sorry

end roots_sum_product_l83_83645


namespace minimum_group_members_round_table_l83_83053

theorem minimum_group_members_round_table (n : ℕ) (h1 : ∀ (a : ℕ),  a < n) : 5 ≤ n :=
by
  sorry

end minimum_group_members_round_table_l83_83053


namespace Sandy_phone_bill_expense_l83_83403
noncomputable def Sandy_age_now : ℕ := 34
noncomputable def Kim_age_now : ℕ := 10
noncomputable def Sandy_phone_bill : ℕ := 10 * Sandy_age_now

theorem Sandy_phone_bill_expense :
  (Sandy_age_now - 2 = 36 - 2) ∧ (Kim_age_now + 2 = 12) ∧ (36 = 3 * 12) ∧ (Sandy_phone_bill = 340) := by
sorry

end Sandy_phone_bill_expense_l83_83403


namespace probability_sum_sixteen_l83_83096

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end probability_sum_sixteen_l83_83096


namespace g_at_zero_l83_83716

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4)

theorem g_at_zero : g 0 = -Real.sqrt 2 :=
by
  -- proof to be completed
  sorry

end g_at_zero_l83_83716


namespace tangent_line_curve_l83_83611

theorem tangent_line_curve (a b : ℝ)
  (h1 : ∀ (x : ℝ), (x - (x^2 + a*x + b) + 1 = 0) ↔ (a = 1 ∧ b = 1))
  (h2 : ∀ (y : ℝ), (0, y) ∈ { p : ℝ × ℝ | p.2 = 0 ^ 2 + a * 0 + b }) :
  a = 1 ∧ b = 1 :=
by
  sorry

end tangent_line_curve_l83_83611


namespace tan_double_angle_identity_l83_83503

theorem tan_double_angle_identity (theta : ℝ) (h1 : 0 < theta ∧ theta < Real.pi / 2)
  (h2 : Real.sin theta - Real.cos theta = Real.sqrt 5 / 5) :
  Real.tan (2 * theta) = -(4 / 3) := 
by
  sorry

end tan_double_angle_identity_l83_83503


namespace solve_mod_equation_l83_83063

theorem solve_mod_equation (x : ℤ) (h : 10 * x + 3 ≡ 7 [ZMOD 18]) : x ≡ 4 [ZMOD 9] :=
sorry

end solve_mod_equation_l83_83063


namespace value_of_g_13_l83_83864

def g (n : ℕ) : ℕ := n^2 + 2 * n + 23

theorem value_of_g_13 : g 13 = 218 :=
by 
  sorry

end value_of_g_13_l83_83864


namespace collinear_k_perpendicular_k_l83_83450

def vector := ℝ × ℝ

def a : vector := (1, 3)
def b : vector := (3, -4)

def collinear (u v : vector) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : vector) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def k_vector_a_minus_b (k : ℝ) (a b : vector) : vector :=
  (k * a.1 - b.1, k * a.2 - b.2)

def a_plus_b (a b : vector) : vector :=
  (a.1 + b.1, a.2 + b.2)

theorem collinear_k (k : ℝ) : collinear (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = -1 :=
sorry

theorem perpendicular_k (k : ℝ) : perpendicular (k_vector_a_minus_b k a b) (a_plus_b a b) ↔ k = 16 :=
sorry

end collinear_k_perpendicular_k_l83_83450


namespace LCM_of_36_and_220_l83_83975

theorem LCM_of_36_and_220:
  let A := 36
  let B := 220
  let productAB := A * B
  let HCF := 4
  let LCM := (A * B) / HCF
  LCM = 1980 := 
by
  sorry

end LCM_of_36_and_220_l83_83975


namespace Mel_weight_is_70_l83_83806

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end Mel_weight_is_70_l83_83806


namespace temperature_difference_l83_83593

-- Definitions based on the conditions
def refrigeration_compartment_temperature : ℤ := 5
def freezer_compartment_temperature : ℤ := -2

-- Mathematically equivalent proof problem statement
theorem temperature_difference : refrigeration_compartment_temperature - freezer_compartment_temperature = 7 := by
  sorry

end temperature_difference_l83_83593


namespace missed_the_bus_by_5_minutes_l83_83171

theorem missed_the_bus_by_5_minutes 
    (usual_time : ℝ)
    (new_time : ℝ)
    (h1 : usual_time = 20)
    (h2 : new_time = usual_time * (5 / 4)) : 
    new_time - usual_time = 5 := 
by
  sorry

end missed_the_bus_by_5_minutes_l83_83171


namespace lorenzo_cans_l83_83032

theorem lorenzo_cans (c : ℕ) (tacks_per_can : ℕ) (total_tacks : ℕ) (boards_tested : ℕ) (remaining_tacks : ℕ) :
  boards_tested = 120 →
  remaining_tacks = 30 →
  total_tacks = 450 →
  tacks_per_can = (boards_tested + remaining_tacks) →
  c * tacks_per_can = total_tacks →
  c = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lorenzo_cans_l83_83032


namespace probability_student_less_than_25_l83_83511

-- Defining the problem conditions
def total_students : ℕ := 100
def percent_male : ℕ := 40
def percent_female : ℕ := 100 - percent_male
def percent_male_25_or_older : ℕ := 40
def percent_female_25_or_older : ℕ := 30

-- Calculation based on the conditions
def num_male_students := (percent_male * total_students) / 100
def num_female_students := (percent_female * total_students) / 100
def num_male_25_or_older := (percent_male_25_or_older * num_male_students) / 100
def num_female_25_or_older := (percent_female_25_or_older * num_female_students) / 100

def num_25_or_older := num_male_25_or_older + num_female_25_or_older
def num_less_than_25 := total_students - num_25_or_older
def probability_less_than_25 := (num_less_than_25: ℚ) / total_students

-- Define the theorem
theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.66 := by
  sorry

end probability_student_less_than_25_l83_83511


namespace y2_minus_x2_l83_83796

theorem y2_minus_x2 (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h1 : 56 ≤ x + y) (h2 : x + y ≤ 59) (h3 : 9 < 10 * x) (h4 : 10 * x < 91 * y) : y^2 - x^2 = 177 :=
by
  sorry

end y2_minus_x2_l83_83796


namespace tangent_line_at_1_l83_83603

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x - 3

theorem tangent_line_at_1 :
  let y := f 1
  let k := f' 1
  y = -3 ∧ k = -2 →
  ∀ (x y : ℝ), y = k * (x - 1) + f 1 ↔ 2 * x + y + 1 = 0 :=
by
  sorry

end tangent_line_at_1_l83_83603


namespace union_of_A_and_B_l83_83526

open Set

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end union_of_A_and_B_l83_83526


namespace trig_identity_l83_83425

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity_l83_83425


namespace no_prime_divisible_by_77_l83_83929

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_divisible (a b : ℕ) : Prop := b ∣ a

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → is_divisible p 77 → false :=
by
  sorry

end no_prime_divisible_by_77_l83_83929


namespace incenter_ineq_l83_83504

open Real

-- Definitions of the incenter and angle bisector intersection points
def incenter (A B C : Point) : Point := sorry
def angle_bisector_intersect (A B C I : Point) (angle_vertex : Point) : Point := sorry
def AI (A I : Point) : ℝ := sorry
def AA' (A A' : Point) : ℝ := sorry
def BI (B I : Point) : ℝ := sorry
def BB' (B B' : Point) : ℝ := sorry
def CI (C I : Point) : ℝ := sorry
def CC' (C C' : Point) : ℝ := sorry

-- Statement of the problem
theorem incenter_ineq 
    (A B C I A' B' C' : Point)
    (h1 : I = incenter A B C)
    (h2 : A' = angle_bisector_intersect A B C I A)
    (h3 : B' = angle_bisector_intersect A B C I B)
    (h4 : C' = angle_bisector_intersect A B C I C) :
    (1/4 : ℝ) < (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ∧ 
    (AI A I * BI B I * CI C I) / (AA' A A' * BB' B B' * CC' C C') ≤ (8/27 : ℝ) :=
sorry

end incenter_ineq_l83_83504


namespace dan_minimum_speed_to_beat_cara_l83_83844

theorem dan_minimum_speed_to_beat_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 120 →
  cara_speed = 30 →
  dan_delay = 1 →
  ∃ (dan_speed : ℕ), dan_speed > 40 :=
by
  sorry

end dan_minimum_speed_to_beat_cara_l83_83844


namespace yi_jianlian_shots_l83_83264

theorem yi_jianlian_shots (x y : ℕ) 
  (h1 : x + y = 16 - 3) 
  (h2 : 2 * x + y = 28 - 3 * 3) : 
  x = 6 ∧ y = 7 := 
by 
  sorry

end yi_jianlian_shots_l83_83264


namespace change_combinations_50_cents_l83_83475

-- Define the conditions for creating 50 cents using standard coins
def ways_to_make_change (pennies nickels dimes : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes

theorem change_combinations_50_cents : 
  ∃ num_ways, 
    num_ways = 28 ∧
    ∀ (pennies nickels dimes : ℕ), 
      pennies + 5 * nickels + 10 * dimes = 50 → 
      -- Exclude using only a single half-dollar
      ¬(num_ways = if (pennies = 0 ∧ nickels = 0 ∧ dimes = 0) then 1 else 28) := 
sorry

end change_combinations_50_cents_l83_83475


namespace angles_on_axes_correct_l83_83978

-- Definitions for angles whose terminal sides lie on x-axis and y-axis.
def angles_on_x_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def angles_on_y_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

-- Combined definition for angles on the coordinate axes using Lean notation
def angles_on_axes (α : ℝ) : Prop := ∃ n : ℤ, α = n * (Real.pi / 2)

-- Theorem stating that angles on the coordinate axes are of the form nπ/2.
theorem angles_on_axes_correct : ∀ α : ℝ, (angles_on_x_axis α ∨ angles_on_y_axis α) ↔ angles_on_axes α := 
sorry -- Proof is omitted.

end angles_on_axes_correct_l83_83978


namespace Q_is_234_l83_83244

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {z | ∃ x y : ℕ, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_is_234 : Q = {2, 3, 4} :=
by
  sorry

end Q_is_234_l83_83244


namespace joggers_difference_l83_83223

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l83_83223


namespace determine_N_l83_83709

variable (U M N : Set ℕ)

theorem determine_N (h1 : U = {1, 2, 3, 4, 5})
  (h2 : U = M ∪ N)
  (h3 : M ∩ (U \ N) = {2, 4}) :
  N = {1, 3, 5} :=
by
  sorry

end determine_N_l83_83709


namespace sum_of_five_consecutive_even_integers_l83_83751

theorem sum_of_five_consecutive_even_integers (a : ℤ) (h : a + (a + 4) = 150) :
  a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385 :=
by
  sorry

end sum_of_five_consecutive_even_integers_l83_83751


namespace find_m_l83_83203

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem find_m (m : ℝ) (h : f m = 3) : m = 3 := 
by
  sorry

end find_m_l83_83203


namespace northbound_vehicle_count_l83_83123

theorem northbound_vehicle_count :
  ∀ (southbound_speed northbound_speed : ℝ) (vehicles_passed : ℕ) 
  (time_minutes : ℝ) (section_length : ℝ), 
  southbound_speed = 70 → northbound_speed = 50 → vehicles_passed = 30 → time_minutes = 10
  → section_length = 150
  → (vehicles_passed / ((southbound_speed + northbound_speed) * (time_minutes / 60))) * section_length = 270 :=
by sorry

end northbound_vehicle_count_l83_83123


namespace Nina_can_buy_8_widgets_at_reduced_cost_l83_83221

def money_Nina_has : ℕ := 48
def widgets_she_can_buy_initially : ℕ := 6
def reduction_per_widget : ℕ := 2

theorem Nina_can_buy_8_widgets_at_reduced_cost :
  let initial_cost_per_widget := money_Nina_has / widgets_she_can_buy_initially
  let reduced_cost_per_widget := initial_cost_per_widget - reduction_per_widget
  money_Nina_has / reduced_cost_per_widget = 8 :=
by
  sorry

end Nina_can_buy_8_widgets_at_reduced_cost_l83_83221


namespace solve_problem_l83_83415

def question : ℝ := -7.8
def answer : ℕ := 22

theorem solve_problem : 2 * (⌊|question|⌋) + (|⌊question⌋|) = answer := by
  sorry

end solve_problem_l83_83415


namespace roger_final_money_l83_83902

variable (initial_money : ℕ)
variable (spent_money : ℕ)
variable (received_money : ℕ)

theorem roger_final_money (h1 : initial_money = 45) (h2 : spent_money = 20) (h3 : received_money = 46) :
  (initial_money - spent_money + received_money) = 71 :=
by
  sorry

end roger_final_money_l83_83902


namespace age_ratio_l83_83618

theorem age_ratio (A B : ℕ) 
  (h1 : A = 39) 
  (h2 : B = 16) 
  (h3 : (A - 5) + (B - 5) = 45) 
  (h4 : A + 5 = 44) : A / B = 39 / 16 := 
by 
  sorry

end age_ratio_l83_83618


namespace reach_any_position_l83_83368

/-- We define a configuration of marbles in terms of a finite list of natural numbers, which corresponds to the number of marbles in each hole. A configuration transitions to another by moving marbles from one hole to subsequent holes in a circular manner. -/
def configuration (n : ℕ) := List ℕ 

/-- Define the operation of distributing marbles from one hole to subsequent holes. -/
def redistribute (l : configuration n) (i : ℕ) : configuration n :=
  sorry -- The exact redistribution function would need to be implemented based on the conditions.

theorem reach_any_position (n : ℕ) (m : ℕ) (init_config final_config : configuration n)
  (h_num_marbles : init_config.sum = m)
  (h_final_marbles : final_config.sum = m) :
  ∃ steps, final_config = (steps : List ℕ).foldl redistribute init_config :=
sorry

end reach_any_position_l83_83368


namespace bee_flight_time_l83_83680

theorem bee_flight_time (t : ℝ) : 
  let speed_daisy_to_rose := 2.6
  let speed_rose_to_poppy := speed_daisy_to_rose + 3
  let distance_daisy_to_rose := speed_daisy_to_rose * 10
  let distance_rose_to_poppy := distance_daisy_to_rose - 8
  distance_rose_to_poppy = speed_rose_to_poppy * t
  ∧ abs (t - 3) < 1 := 
sorry

end bee_flight_time_l83_83680


namespace g_g_g_g_3_eq_101_l83_83849

def g (m : ℕ) : ℕ :=
  if m < 5 then m^2 + 1 else 2 * m + 3

theorem g_g_g_g_3_eq_101 : g (g (g (g 3))) = 101 :=
  by {
    -- the proof goes here
    sorry
  }

end g_g_g_g_3_eq_101_l83_83849


namespace infinite_subsets_exists_divisor_l83_83914

-- Definition of the set M
def M : Set ℕ := { n | ∃ a b : ℕ, n = 2^a * 3^b }

-- Infinite family of subsets of M
variable (A : ℕ → Set ℕ)
variables (inf_family : ∀ i, A i ⊆ M)

-- Theorem statement
theorem infinite_subsets_exists_divisor :
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x := by
  sorry

end infinite_subsets_exists_divisor_l83_83914


namespace days_before_reinforcement_l83_83357

theorem days_before_reinforcement
    (garrison_1 : ℕ)
    (initial_days : ℕ)
    (reinforcement : ℕ)
    (additional_days : ℕ)
    (total_men_after_reinforcement : ℕ)
    (man_days_initial : ℕ)
    (man_days_after : ℕ)
    (x : ℕ) :
    garrison_1 * (initial_days - x) = total_men_after_reinforcement * additional_days →
    garrison_1 = 2000 →
    initial_days = 54 →
    reinforcement = 1600 →
    additional_days = 20 →
    total_men_after_reinforcement = garrison_1 + reinforcement →
    man_days_initial = garrison_1 * initial_days →
    man_days_after = total_men_after_reinforcement * additional_days →
    x = 18 :=
by
  intros h_eq g_1 i_days r_f a_days total_men m_days_i m_days_a
  sorry

end days_before_reinforcement_l83_83357


namespace range_of_alpha_minus_beta_l83_83549

theorem range_of_alpha_minus_beta (α β : ℝ) (h1 : -180 < α) (h2 : α < β) (h3 : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l83_83549


namespace bonnets_per_orphanage_l83_83675

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end bonnets_per_orphanage_l83_83675


namespace can_construct_prism_with_fewer_than_20_shapes_l83_83309

/-
  We have 5 congruent unit cubes glued together to form complex shapes.
  4 of these cubes form a 4-unit high prism, and the fifth is attached to one of the inner cubes with a full face.
  Prove that we can construct a solid rectangular prism using fewer than 20 of these shapes.
-/

theorem can_construct_prism_with_fewer_than_20_shapes :
  ∃ (n : ℕ), n < 20 ∧ (∃ (length width height : ℕ), length * width * height = 5 * n) :=
sorry

end can_construct_prism_with_fewer_than_20_shapes_l83_83309


namespace interval_solution_l83_83497

-- Let the polynomial be defined
def polynomial (x : ℝ) : ℝ := x^3 - 12 * x^2 + 30 * x

-- Prove the inequality for the specified intervals
theorem interval_solution :
  { x : ℝ | polynomial x > 0 } = { x : ℝ | (0 < x ∧ x < 5) ∨ x > 6 } :=
by
  sorry

end interval_solution_l83_83497


namespace hal_battery_change_25th_time_l83_83351

theorem hal_battery_change_25th_time (months_in_year : ℕ) 
    (battery_interval : ℕ) 
    (first_change_month : ℕ) 
    (change_count : ℕ) : 
    (battery_interval * (change_count-1)) % months_in_year + first_change_month % months_in_year = first_change_month % months_in_year :=
by
    have h1 : months_in_year = 12 := by sorry
    have h2 : battery_interval = 5 := by sorry
    have h3 : first_change_month = 5 := by sorry -- May is represented by 5 (0 = January, 1 = February, ..., 4 = April, 5 = May, ...)
    have h4 : change_count = 25 := by sorry
    sorry

end hal_battery_change_25th_time_l83_83351


namespace tom_speed_from_A_to_B_l83_83885

theorem tom_speed_from_A_to_B (D S : ℝ) (h1 : 2 * D = S * (3 * D / 36 - D / 20))
  (h2 : S * (3 * D / 36 - D / 20) = 3 * D / 36 ∨ 3 * D / 36 = S * (3 * D / 36 - D / 20))
  (h3 : D > 0) : S = 60 :=
by { sorry }

end tom_speed_from_A_to_B_l83_83885


namespace disjoint_subsets_less_elements_l83_83950

open Nat

theorem disjoint_subsets_less_elements (m : ℕ) (A B : Finset ℕ) (hA : A ⊆ Finset.range (m + 1))
  (hB : B ⊆ Finset.range (m + 1)) (h_disjoint : Disjoint A B)
  (h_sum : A.sum id = B.sum id) : ↑(A.card) < m / Real.sqrt 2 ∧ ↑(B.card) < m / Real.sqrt 2 := 
sorry

end disjoint_subsets_less_elements_l83_83950


namespace negation_proposition_l83_83172

theorem negation_proposition (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
sorry

end negation_proposition_l83_83172


namespace min_value_function_l83_83017

theorem min_value_function (x : ℝ) (h : x > 0) : 
  ∃ y, y = (x^2 + x + 25) / x ∧ y ≥ 11 :=
sorry

end min_value_function_l83_83017


namespace book_pages_count_l83_83502

theorem book_pages_count :
  (∀ n : ℕ, n = 4 → 42 * n = 168) ∧
  (∀ n : ℕ, n = 2 → 50 * n = 100) ∧
  (∀ p1 p2 : ℕ, p1 = 168 ∧ p2 = 100 → p1 + p2 = 268) ∧
  (∀ p : ℕ, p = 268 → p + 30 = 298) →
  298 = 298 := by
  sorry

end book_pages_count_l83_83502


namespace committee_probability_l83_83226

def num_boys : ℕ := 10
def num_girls : ℕ := 15
def num_total : ℕ := 25
def committee_size : ℕ := 5

def num_ways_total : ℕ := Nat.choose num_total committee_size
def num_ways_boys_only : ℕ := Nat.choose num_boys committee_size
def num_ways_girls_only : ℕ := Nat.choose num_girls committee_size

def probability_boys_or_girls_only : ℚ :=
  (num_ways_boys_only + num_ways_girls_only) / num_ways_total

def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - probability_boys_or_girls_only

theorem committee_probability :
  probability_at_least_one_boy_and_one_girl = 475 / 506 :=
sorry

end committee_probability_l83_83226


namespace quadratic_inequality_k_range_l83_83815

variable (k : ℝ)

theorem quadratic_inequality_k_range (h : ∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) :
  -1 < k ∧ k < 0 := by
sorry

end quadratic_inequality_k_range_l83_83815


namespace trader_sold_bags_l83_83765

-- Define the conditions as constants
def initial_bags : ℕ := 55
def restocked_bags : ℕ := 132
def current_bags : ℕ := 164

-- Define a function to calculate the number of bags sold
def bags_sold (initial restocked current : ℕ) : ℕ :=
  initial + restocked - current

-- Statement of the proof problem
theorem trader_sold_bags : bags_sold initial_bags restocked_bags current_bags = 23 :=
by
  -- Proof is omitted
  sorry

end trader_sold_bags_l83_83765


namespace ratio_of_numbers_l83_83600

theorem ratio_of_numbers (a b : ℕ) (ha : a = 45) (hb : b = 60) (lcm_ab : Nat.lcm a b = 180) : (a : ℚ) / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l83_83600


namespace angle_I_measure_l83_83282

theorem angle_I_measure {x y : ℝ} 
  (h1 : x = y - 50) 
  (h2 : 3 * x + 2 * y = 540)
  : y = 138 := 
by 
  sorry

end angle_I_measure_l83_83282


namespace vector_perpendicular_iff_l83_83376

theorem vector_perpendicular_iff (k : ℝ) :
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  a.1 * c.1 + ab.2 * c.2 = 0 → k = -3 :=
by
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  intro h
  sorry

end vector_perpendicular_iff_l83_83376


namespace odot_property_l83_83261

def odot (x y : ℤ) := 2 * x + y

theorem odot_property (a b : ℤ) (h : odot a (-6 * b) = 4) : odot (a - 5 * b) (a + b) = 6 :=
by
  sorry

end odot_property_l83_83261


namespace interesting_quadruples_count_l83_83236

/-- Definition of interesting ordered quadruples (a, b, c, d) where 1 ≤ a < b < c < d ≤ 15 and a + b > c + d --/
def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + b > c + d

/-- The number of interesting ordered quadruples (a, b, c, d) is 455 --/
theorem interesting_quadruples_count : 
  (∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s.card = 455 ∧ ∀ (a b c d : ℕ), 
    ((a, b, c, d) ∈ s ↔ is_interesting_quadruple a b c d)) :=
sorry

end interesting_quadruples_count_l83_83236


namespace probability_of_sum_being_6_l83_83383

noncomputable def prob_sum_6 : ℚ :=
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_of_sum_being_6 :
  prob_sum_6 = 5 / 36 :=
by
  sorry

end probability_of_sum_being_6_l83_83383


namespace seven_people_arrangement_l83_83250

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def perm (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem seven_people_arrangement : 
  (perm 5 5) * (perm 6 2) = 3600 := by
sorry

end seven_people_arrangement_l83_83250


namespace lesser_of_two_numbers_l83_83084

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x * y = 1050) : min x y = 30 :=
sorry

end lesser_of_two_numbers_l83_83084


namespace six_digit_number_all_equal_l83_83740

open Nat

theorem six_digit_number_all_equal (n : ℕ) (h : n = 21) : 12 * n^2 + 12 * n + 11 = 5555 :=
by
  rw [h]  -- Substitute n = 21
  sorry  -- Omit the actual proof steps

end six_digit_number_all_equal_l83_83740


namespace express_y_in_terms_of_x_l83_83323

theorem express_y_in_terms_of_x (x y : ℝ) (h : 4 * x - y = 7) : y = 4 * x - 7 :=
sorry

end express_y_in_terms_of_x_l83_83323


namespace total_turns_to_fill_drum_l83_83706

variable (Q : ℝ) -- Capacity of bucket Q
variable (turnsP : ℝ) (P_capacity : ℝ) (R_capacity : ℝ) (drum_capacity : ℝ)

-- Condition: It takes 60 turns for bucket P to fill the empty drum
def bucketP_fills_drum_in_60_turns : Prop := turnsP = 60 ∧ P_capacity = 3 * Q ∧ drum_capacity = 60 * P_capacity

-- Condition: Bucket P has thrice the capacity as bucket Q
def bucketP_capacity : Prop := P_capacity = 3 * Q

-- Condition: Bucket R has half the capacity as bucket Q
def bucketR_capacity : Prop := R_capacity = Q / 2

-- Computation: Using all three buckets together, find the combined capacity filled in one turn
def combined_capacity_per_turn : ℝ := P_capacity + Q + R_capacity

-- Main Theorem: It takes 40 turns to fill the drum using all three buckets together
theorem total_turns_to_fill_drum
  (h1 : bucketP_fills_drum_in_60_turns Q turnsP P_capacity drum_capacity)
  (h2 : bucketP_capacity Q P_capacity)
  (h3 : bucketR_capacity Q R_capacity) :
  drum_capacity / combined_capacity_per_turn Q P_capacity (Q / 2) = 40 :=
by
  sorry

end total_turns_to_fill_drum_l83_83706


namespace sequence_general_formula_l83_83044

theorem sequence_general_formula
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (3 / 2) * (a n) - 3) :
  ∀ n, a n = 3 * (2 : ℝ) ^ n :=
by sorry

end sequence_general_formula_l83_83044


namespace largest_xy_l83_83081

-- Define the problem conditions
def conditions (x y : ℕ) : Prop := 27 * x + 35 * y ≤ 945 ∧ x > 0 ∧ y > 0

-- Define the largest value of xy
def largest_xy_value : ℕ := 234

-- Prove that the largest possible value of xy given conditions is 234
theorem largest_xy (x y : ℕ) (h : conditions x y) : x * y ≤ largest_xy_value :=
sorry

end largest_xy_l83_83081


namespace two_digit_numbers_sum_reversed_l83_83173

theorem two_digit_numbers_sum_reversed (a b : ℕ) (h₁ : 0 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : a + b = 12) :
  ∃ n : ℕ, n = 7 := 
sorry

end two_digit_numbers_sum_reversed_l83_83173


namespace sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l83_83193

theorem sum_two_consecutive : ∃ x : ℕ, 75 = x + (x + 1) := by
  sorry

theorem sum_three_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) := by
  sorry

theorem sum_five_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) := by
  sorry

theorem sum_six_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) := by
  sorry

end sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l83_83193


namespace find_multiplier_l83_83989

theorem find_multiplier (x n : ℤ) (h : 2 * n + 20 = x * n - 4) (hn : n = 4) : x = 8 :=
by
  sorry

end find_multiplier_l83_83989


namespace sin_2alpha_val_l83_83817

-- Define the conditions and the problem in Lean 4
theorem sin_2alpha_val (α : ℝ) (h1 : π < α ∨ α < 3 * π / 2)
  (h2 : 2 * (Real.tan α) ^ 2 - 7 * Real.tan α + 3 = 0) :
  (π < α ∧ α < 5 * π / 4 → Real.sin (2 * α) = 4 / 5) ∧ 
  (5 * π / 4 < α ∧ α < 3 * π / 2 → Real.sin (2 * α) = 3 / 5) := 
sorry

end sin_2alpha_val_l83_83817


namespace max_sum_pyramid_l83_83636

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid_l83_83636


namespace grain_to_rice_system_l83_83040

variable (x y : ℕ)

/-- Conversion rate of grain to rice is 3/5. -/
def conversion_rate : ℚ := 3 / 5

/-- Total bucket capacity is 10 dou. -/
def total_capacity : ℕ := 10

/-- Rice obtained after threshing is 7 dou. -/
def rice_obtained : ℕ := 7

/-- The system of equations representing the problem. -/
theorem grain_to_rice_system :
  (x + y = total_capacity) ∧ (conversion_rate * x + y = rice_obtained) := 
sorry

end grain_to_rice_system_l83_83040


namespace train_length_l83_83615

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end train_length_l83_83615


namespace grid_problem_l83_83605

theorem grid_problem 
  (A B : ℕ) 
  (grid : (Fin 3) → (Fin 3) → ℕ)
  (h1 : ∀ i, grid 0 i ≠ grid 1 i)
  (h2 : ∀ i, grid 0 i ≠ grid 2 i)
  (h3 : ∀ i, grid 1 i ≠ grid 2 i)
  (h4 : ∀ i, (∃! x, grid x i = 1))
  (h5 : ∀ i, (∃! x, grid x i = 2))
  (h6 : ∀ i, (∃! x, grid x i = 3))
  (h7 : grid 1 2 = A)
  (h8 : grid 2 2 = B) : 
  A + B + 4 = 8 :=
by sorry

end grid_problem_l83_83605


namespace division_result_l83_83741

-- Define the arithmetic expression
def arithmetic_expression : ℕ := (20 + 15 * 3) - 10

-- Define the main problem
def problem : Prop := 250 / arithmetic_expression = 250 / 55

-- The theorem statement that needs to be proved
theorem division_result : problem := by
    sorry

end division_result_l83_83741


namespace prob_five_coins_heads_or_one_tail_l83_83028

theorem prob_five_coins_heads_or_one_tail : 
  (∃ (H T : ℚ), H = 1/32 ∧ T = 31/32 ∧ H + T = 1) ↔ 1 = 1 :=
by sorry

end prob_five_coins_heads_or_one_tail_l83_83028


namespace problem_solution_l83_83944

theorem problem_solution : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end problem_solution_l83_83944


namespace jerry_initial_action_figures_l83_83324

theorem jerry_initial_action_figures 
(A : ℕ) 
(h1 : ∀ A, A + 7 = 9 + 3)
: A = 5 :=
by
  sorry

end jerry_initial_action_figures_l83_83324


namespace smallest_positive_m_integral_solutions_l83_83983

theorem smallest_positive_m_integral_solutions (m : ℕ) :
  (∃ (x y : ℤ), 10 * x * x - m * x + 660 = 0 ∧ 10 * y * y - m * y + 660 = 0 ∧ x ≠ y)
  → m = 170 := sorry

end smallest_positive_m_integral_solutions_l83_83983


namespace cost_of_Roger_cookie_l83_83818

theorem cost_of_Roger_cookie
  (art_cookie_length : ℕ := 4)
  (art_cookie_width : ℕ := 3)
  (art_cookie_count : ℕ := 10)
  (roger_cookie_side : ℕ := 3)
  (art_cookie_price : ℕ := 50)
  (same_dough_used : ℕ := art_cookie_count * art_cookie_length * art_cookie_width)
  (roger_cookie_area : ℕ := roger_cookie_side * roger_cookie_side)
  (roger_cookie_count : ℕ := same_dough_used / roger_cookie_area) :
  (500 / roger_cookie_count) = 38 := by
  sorry

end cost_of_Roger_cookie_l83_83818


namespace total_cases_after_three_days_l83_83789

def initial_cases : ℕ := 2000
def increase_rate : ℝ := 0.20
def recovery_rate : ℝ := 0.02

def day_cases (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_cases
  | n + 1 => 
      let prev_cases := day_cases n
      let new_cases := increase_rate * prev_cases
      let recovered := recovery_rate * prev_cases
      prev_cases + new_cases - recovered

theorem total_cases_after_three_days : day_cases 3 = 3286 := by sorry

end total_cases_after_three_days_l83_83789


namespace num_ways_to_assign_grades_l83_83006

-- Define the number of students
def num_students : ℕ := 12

-- Define the number of grades available to each student
def num_grades : ℕ := 4

-- The theorem stating that the total number of ways to assign grades is 4^12
theorem num_ways_to_assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end num_ways_to_assign_grades_l83_83006


namespace number_is_composite_l83_83295

theorem number_is_composite : ∃ k l : ℕ, k * l = 53 * 83 * 109 + 40 * 66 * 96 ∧ k > 1 ∧ l > 1 :=
by
  have h1 : 53 + 96 = 149 := by norm_num
  have h2 : 83 + 66 = 149 := by norm_num
  have h3 : 109 + 40 = 149 := by norm_num
  sorry

end number_is_composite_l83_83295


namespace cauchy_solution_l83_83913

theorem cauchy_solution (f : ℚ → ℚ) (h : ∀ x y : ℚ, f ((x + y) / 2) = (f x) / 2 + (f y) / 2) : 
  ∃ c : ℚ, ∀ x : ℚ, f x = c * x := 
sorry

end cauchy_solution_l83_83913


namespace common_ratio_of_geometric_sequence_l83_83982

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 1 + 4 * d = (a 0 + 16 * d) * (a 0 + 4 * d) / a 0 ) :
  (a 1 + 4 * d) / a 0 = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l83_83982


namespace chips_reach_end_l83_83778

theorem chips_reach_end (n k : ℕ) (h : n > k * 2^k) : True := sorry

end chips_reach_end_l83_83778


namespace minimum_value_of_f_l83_83424

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x) + 2 * Real.sin x)

theorem minimum_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -3 := 
  sorry

end minimum_value_of_f_l83_83424


namespace largest_divisor_of_m_l83_83879

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : ∃ k : ℕ, m^3 = 847 * k) : ∃ d : ℕ, d = 77 ∧ ∀ x : ℕ, x > d → ¬ (x ∣ m) :=
sorry

end largest_divisor_of_m_l83_83879


namespace wholesale_price_of_milk_l83_83408

theorem wholesale_price_of_milk (W : ℝ) 
  (h1 : ∀ p : ℝ, p = 1.25 * W) 
  (h2 : ∀ q : ℝ, q = 0.95 * (1.25 * W)) 
  (h3 : q = 4.75) :
  W = 4 :=
by
  sorry

end wholesale_price_of_milk_l83_83408


namespace minimal_guests_l83_83794

-- Problem statement: For 120 chairs arranged in a circle,
-- determine the smallest number of guests (N) needed 
-- so that any additional guest must sit next to an already seated guest.

theorem minimal_guests (N : ℕ) : 
  (∀ (chairs : ℕ), chairs = 120 → 
    ∃ (N : ℕ), N = 20 ∧ 
      (∀ (new_guest : ℕ), new_guest + chairs = 120 → 
        new_guest ≤ N + 1 ∧ new_guest ≤ N - 1)) :=
by
  sorry

end minimal_guests_l83_83794


namespace radius_circumcircle_l83_83186

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end radius_circumcircle_l83_83186


namespace total_growing_space_is_correct_l83_83445

def garden_bed_area (length : ℕ) (width : ℕ) (count : ℕ) : ℕ :=
  length * width * count

def total_growing_space : ℕ :=
  garden_bed_area 5 4 3 +
  garden_bed_area 6 3 4 +
  garden_bed_area 7 5 2 +
  garden_bed_area 8 4 1

theorem total_growing_space_is_correct :
  total_growing_space = 234 := by
  sorry

end total_growing_space_is_correct_l83_83445


namespace quadratic_real_roots_range_l83_83159

theorem quadratic_real_roots_range (m : ℝ) : (∃ x y : ℝ, x ≠ y ∧ mx^2 + 2*x + 1 = 0 ∧ yx^2 + 2*y + 1 = 0) → m ≤ 1 ∧ m ≠ 0 :=
by 
sorry

end quadratic_real_roots_range_l83_83159


namespace roger_gave_candies_l83_83822

theorem roger_gave_candies :
  ∀ (original_candies : ℕ) (remaining_candies : ℕ) (given_candies : ℕ),
  original_candies = 95 → remaining_candies = 92 → given_candies = original_candies - remaining_candies → given_candies = 3 :=
by
  intros
  sorry

end roger_gave_candies_l83_83822


namespace find_numbers_l83_83355

theorem find_numbers (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
                     (hxy_mul : 2000 ≤ x * y ∧ x * y < 3000) (hxy_add : 100 ≤ x + y ∧ x + y < 1000)
                     (h_digit_relation : x * y = 2000 + x + y) : 
                     (x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30) :=
by
  -- The proof will go here
  sorry

end find_numbers_l83_83355


namespace largest_common_multiple_3_5_l83_83573

theorem largest_common_multiple_3_5 (n : ℕ) :
  (n < 10000) ∧ (n ≥ 1000) ∧ (n % 3 = 0) ∧ (n % 5 = 0) → n ≤ 9990 :=
sorry

end largest_common_multiple_3_5_l83_83573


namespace ellipse_foci_coordinates_l83_83788

theorem ellipse_foci_coordinates :
  ∀ x y : ℝ,
  25 * x^2 + 16 * y^2 = 1 →
  (x, y) = (0, 3/20) ∨ (x, y) = (0, -3/20) :=
by
  intro x y h
  sorry

end ellipse_foci_coordinates_l83_83788


namespace fraction_identity_l83_83538

theorem fraction_identity (m n r t : ℚ) (h1 : m / n = 5 / 3) (h2 : r / t = 8 / 15) : 
  (4 * m * r - 2 * n * t) / (5 * n * t - 9 * m * r) = -14 / 27 :=
by 
  sorry

end fraction_identity_l83_83538


namespace each_persons_share_l83_83456

def total_bill : ℝ := 211.00
def number_of_people : ℕ := 5
def tip_rate : ℝ := 0.15

theorem each_persons_share :
  (total_bill * (1 + tip_rate)) / number_of_people = 48.53 := 
by sorry

end each_persons_share_l83_83456


namespace sector_area_is_80pi_l83_83607

noncomputable def sectorArea (θ r : ℝ) : ℝ := 
  1 / 2 * θ * r^2

theorem sector_area_is_80pi :
  sectorArea (2 * Real.pi / 5) 20 = 80 * Real.pi :=
by
  sorry

end sector_area_is_80pi_l83_83607


namespace solution_a_eq_2_solution_a_in_real_l83_83764

-- Define the polynomial inequality for the given conditions
def inequality (x : ℝ) (a : ℝ) : Prop := 12 * x ^ 2 - a * x > a ^ 2

-- Proof statement for when a = 2
theorem solution_a_eq_2 :
  ∀ x : ℝ, inequality x 2 ↔ (x < - (1 : ℝ) / 2) ∨ (x > (2 : ℝ) / 3) :=
sorry

-- Proof statement for when a is in ℝ
theorem solution_a_in_real (a : ℝ) :
  ∀ x : ℝ, inequality x a ↔
    if h : 0 < a then (x < - a / 4) ∨ (x > a / 3)
    else if h : a = 0 then (x ≠ 0)
    else (x < a / 3) ∨ (x > - a / 4) :=
sorry

end solution_a_eq_2_solution_a_in_real_l83_83764


namespace eval_expression_l83_83328

theorem eval_expression : (4^2 - 2^3) = 8 := by
  sorry

end eval_expression_l83_83328


namespace inequality_C_l83_83609

variable (a b : ℝ)
variable (h : a > b)
variable (h' : b > 0)

theorem inequality_C : a + b > 2 * b := by
  sorry

end inequality_C_l83_83609


namespace smallest_n_l83_83723

theorem smallest_n (n : ℕ) :
  (1 / 4 : ℚ) + (n / 8 : ℚ) > 1 ↔ n ≥ 7 := by
  sorry

end smallest_n_l83_83723


namespace cost_of_fencing_per_meter_l83_83596

theorem cost_of_fencing_per_meter (length breadth : ℕ) (total_cost : ℚ) 
    (h_length : length = 61) 
    (h_rule : length = breadth + 22) 
    (h_total_cost : total_cost = 5300) :
    total_cost / (2 * length + 2 * breadth) = 26.5 := 
by 
  sorry

end cost_of_fencing_per_meter_l83_83596


namespace seq_ratio_l83_83597

noncomputable def arith_seq (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem seq_ratio (a d : ℝ) (h₁ : d ≠ 0) (h₂ : (arith_seq a d 2)^2 = (arith_seq a d 0) * (arith_seq a d 8)) :
  (arith_seq a d 0 + arith_seq a d 2 + arith_seq a d 4) / (arith_seq a d 1 + arith_seq a d 3 + arith_seq a d 5) = 3 / 4 :=
by
  sorry

end seq_ratio_l83_83597


namespace function_value_l83_83992

theorem function_value (f : ℝ → ℝ) (h : ∀ x, x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  sorry

end function_value_l83_83992


namespace total_milks_taken_l83_83346

def total_milks (chocolateMilk strawberryMilk regularMilk : Nat) : Nat :=
  chocolateMilk + strawberryMilk + regularMilk

theorem total_milks_taken :
  total_milks 2 15 3 = 20 :=
by
  sorry

end total_milks_taken_l83_83346


namespace eval_x_plus_one_eq_4_l83_83152

theorem eval_x_plus_one_eq_4 (x : ℕ) (h : x = 3) : x + 1 = 4 :=
by
  sorry

end eval_x_plus_one_eq_4_l83_83152


namespace simplify_expression_eq_69_l83_83820

theorem simplify_expression_eq_69 : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end simplify_expression_eq_69_l83_83820


namespace mars_bars_count_l83_83891

theorem mars_bars_count (total_candy_bars snickers butterfingers : ℕ) (h_total : total_candy_bars = 12) (h_snickers : snickers = 3) (h_butterfingers : butterfingers = 7) :
  total_candy_bars - (snickers + butterfingers) = 2 :=
by sorry

end mars_bars_count_l83_83891


namespace graphs_differ_l83_83130

theorem graphs_differ (x : ℝ) :
  (∀ (y : ℝ), y = x + 3 ↔ y ≠ (x^2 - 1) / (x - 1) ∧
              y ≠ (x^2 - 1) / (x - 1) ∧
              ∀ (y : ℝ), y = (x^2 - 1) / (x - 1) ↔ ∀ (z : ℝ), y ≠ x + 3 ∧ y ≠ x + 1) := sorry

end graphs_differ_l83_83130


namespace third_angle_of_triangle_l83_83251

theorem third_angle_of_triangle (a b : ℝ) (ha : a = 50) (hb : b = 60) : 
  ∃ (c : ℝ), a + b + c = 180 ∧ c = 70 :=
by
  sorry

end third_angle_of_triangle_l83_83251


namespace three_digit_multiples_of_7_l83_83225

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end three_digit_multiples_of_7_l83_83225


namespace planks_from_friends_l83_83816

theorem planks_from_friends :
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  planks_from_friends = 20 :=
by
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  rfl

end planks_from_friends_l83_83816


namespace balloon_highest_elevation_l83_83367

theorem balloon_highest_elevation 
  (lift_rate : ℕ)
  (descend_rate : ℕ)
  (pull_time1 : ℕ)
  (release_time : ℕ)
  (pull_time2 : ℕ) :
  lift_rate = 50 →
  descend_rate = 10 →
  pull_time1 = 15 →
  release_time = 10 →
  pull_time2 = 15 →
  (lift_rate * pull_time1 - descend_rate * release_time + lift_rate * pull_time2) = 1400 :=
by
  sorry

end balloon_highest_elevation_l83_83367


namespace reduced_price_is_55_l83_83168

variables (P R : ℝ) (X : ℕ)

-- Conditions
def condition1 : R = 0.75 * P := sorry
def condition2 : P * X = 1100 := sorry
def condition3 : 0.75 * P * (X + 5) = 1100 := sorry

-- Theorem
theorem reduced_price_is_55 (P R : ℝ) (X : ℕ) (h1 : R = 0.75 * P) (h2 : P * X = 1100) (h3 : 0.75 * P * (X + 5) = 1100) :
  R = 55 :=
sorry

end reduced_price_is_55_l83_83168


namespace question1_question2_l83_83845

def setA : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def setB (a : ℝ) : Set ℝ := {x : ℝ | abs (x - a) ≤ 1 }

def complementA : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 3}

theorem question1 : A = setA := sorry

theorem question2 (a : ℝ) : setB a ∩ complementA = setB a → a ∈ Set.union (Set.Iic (-2)) (Set.Ioi 4) := sorry

end question1_question2_l83_83845


namespace households_subscribing_to_F_l83_83763

theorem households_subscribing_to_F
  (x y : ℕ)
  (hx : x ≥ 1)
  (h_subscriptions : 1 + 4 + 2 + 2 + 2 + y = 2 + 2 + 4 + 3 + 5 + x)
  : y = 6 :=
sorry

end households_subscribing_to_F_l83_83763


namespace ratio_of_areas_is_five_l83_83804

-- Define a convex quadrilateral ABCD
structure Quadrilateral (α : Type) :=
  (A B C D : α)
  (convex : True)  -- We assume convexity

-- Define the additional points B1, C1, D1, A1
structure ExtendedPoints (α : Type) (q : Quadrilateral α) :=
  (B1 C1 D1 A1 : α)
  (BB1_eq_AB : True) -- we assume the conditions BB1 = AB
  (CC1_eq_BC : True) -- CC1 = BC
  (DD1_eq_CD : True) -- DD1 = CD
  (AA1_eq_DA : True) -- AA1 = DA

-- Define the areas of the quadrilaterals
noncomputable def area {α : Type} [MetricSpace α] (A B C D : α) : ℝ := sorry
noncomputable def ratio_of_areas {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) : ℝ :=
  (area p.A1 p.B1 p.C1 p.D1) / (area q.A q.B q.C q.D)

theorem ratio_of_areas_is_five {α : Type} [MetricSpace α] (q : Quadrilateral α) (p : ExtendedPoints α q) :
  ratio_of_areas q p = 5 := sorry

end ratio_of_areas_is_five_l83_83804


namespace range_of_n_l83_83807

theorem range_of_n (n : ℝ) (x : ℝ) (h1 : 180 - n > 0) (h2 : ∀ x, 180 - n != x ∧ 180 - n != x + 24 → 180 - n + x + x + 24 = 180 → 44 ≤ x ∧ x ≤ 52 → 112 ≤ n ∧ n ≤ 128)
  (h3 : ∀ n, 180 - n = max (180 - n) (180 - n) - 24 ∧ min (180 - n) (180 - n) = n - 24 → 104 ≤ n ∧ n ≤ 112)
  (h4 : ∀ n, 180 - n = min (180 - n) (180 - n) ∧ max (180 - n) (180 - n) = 180 - n + 24 → 128 ≤ n ∧ n ≤ 136) :
  104 ≤ n ∧ n ≤ 136 :=
by sorry

end range_of_n_l83_83807


namespace tan_ratio_l83_83055

theorem tan_ratio (p q : Real) (hpq1 : Real.sin (p + q) = 0.6) (hpq2 : Real.sin (p - q) = 0.3) : 
  Real.tan p / Real.tan q = 3 := 
by
  sorry

end tan_ratio_l83_83055


namespace amy_balloons_l83_83798

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 232) (h2 : james_balloons = amy_balloons + 131) :
  amy_balloons = 101 :=
by
  sorry

end amy_balloons_l83_83798


namespace pair_exists_l83_83520

theorem pair_exists (x : Fin 670 → ℝ) (h_distinct : Function.Injective x) (h_bounds : ∀ i, 0 < x i ∧ x i < 1) :
  ∃ (i j : Fin 670), 0 < x i * x j * (x j - x i) ∧ x i * x j * (x j - x i) < 1 / 2007 := 
by
  sorry

end pair_exists_l83_83520


namespace cyclist_speed_ratio_l83_83728

-- conditions: 
variables (T₁ T₂ o₁ o₂ : ℝ)
axiom h1 : o₁ + T₁ = o₂ + T₂
axiom h2 : T₁ = 2 * o₂
axiom h3 : T₂ = 4 * o₁

-- Proof statement to show that the second cyclist rides 1.5 times faster:
theorem cyclist_speed_ratio : T₁ / T₂ = 1.5 :=
by
  sorry

end cyclist_speed_ratio_l83_83728


namespace selena_trip_length_l83_83131

variable (y : ℚ)

def selena_trip (y : ℚ) : Prop :=
  y / 4 + 16 + y / 6 = y

theorem selena_trip_length : selena_trip y → y = 192 / 7 :=
by
  sorry

end selena_trip_length_l83_83131


namespace inverse_proportional_x_y_l83_83523

theorem inverse_proportional_x_y (x y k : ℝ) (h_inverse : x * y = k) (h_given : 40 * 5 = k) : x = 20 :=
by 
  sorry

end inverse_proportional_x_y_l83_83523


namespace total_sacks_needed_l83_83184

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l83_83184


namespace train_crossing_time_l83_83487

/-- A train 400 m long traveling at a speed of 36 km/h crosses an electric pole in 40 seconds. -/
theorem train_crossing_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) 
  (h1 : length = 400)
  (h2 : speed_kmph = 36)
  (h3 : speed_mps = speed_kmph * 1000 / 3600)
  (h4 : time = length / speed_mps) :
  time = 40 :=
by {
  sorry
}

end train_crossing_time_l83_83487


namespace area_of_largest_square_l83_83692

theorem area_of_largest_square (a b c : ℕ) (h_triangle : c^2 = a^2 + b^2) (h_sum_areas : a^2 + b^2 + c^2 = 450) :
  c^2 = 225 :=
by
  sorry

end area_of_largest_square_l83_83692


namespace profit_per_meal_A_and_B_l83_83926

theorem profit_per_meal_A_and_B (x y : ℝ) 
  (h1 : x + 2 * y = 35) 
  (h2 : 2 * x + 3 * y = 60) : 
  x = 15 ∧ y = 10 :=
sorry

end profit_per_meal_A_and_B_l83_83926


namespace number_of_participants_l83_83874

theorem number_of_participants (n : ℕ) (hn : n = 862) 
    (h_lower : 575 ≤ n * 2 / 3) 
    (h_upper : n * 7 / 9 ≤ 670) : 
    ∃ p, (575 ≤ p) ∧ (p ≤ 670) ∧ (p % 11 = 0) ∧ ((p - 575) / 11 + 1 = 8) :=
by
  sorry

end number_of_participants_l83_83874


namespace reciprocal_of_neg2019_l83_83288

theorem reciprocal_of_neg2019 : (1 / -2019) = - (1 / 2019) := 
by
  sorry

end reciprocal_of_neg2019_l83_83288


namespace balance_after_transactions_l83_83792

variable (x : ℝ)

def monday_spent : ℝ := 0.525 * x
def tuesday_spent (remaining : ℝ) : ℝ := 0.106875 * remaining
def wednesday_spent (remaining : ℝ) : ℝ := 0.131297917 * remaining
def thursday_spent (remaining : ℝ) : ℝ := 0.040260605 * remaining

def final_balance (x : ℝ) : ℝ :=
  let after_monday := x - monday_spent x
  let after_tuesday := after_monday - tuesday_spent after_monday
  let after_wednesday := after_tuesday - wednesday_spent after_tuesday
  after_wednesday - thursday_spent after_wednesday

theorem balance_after_transactions (x : ℝ) :
  final_balance x = 0.196566478 * x :=
by
  sorry

end balance_after_transactions_l83_83792


namespace jo_thinking_greatest_integer_l83_83948

theorem jo_thinking_greatest_integer :
  ∃ n : ℕ, n < 150 ∧ 
           (∃ k : ℤ, n = 9 * k - 2) ∧ 
           (∃ m : ℤ, n = 11 * m - 4) ∧ 
           (∀ N : ℕ, (N < 150 ∧ 
                      (∃ K : ℤ, N = 9 * K - 2) ∧ 
                      (∃ M : ℤ, N = 11 * M - 4)) → N ≤ n) 
:= by
  sorry

end jo_thinking_greatest_integer_l83_83948


namespace proportion_of_second_prize_winners_l83_83932

-- conditions
variables (A B C : ℝ) -- A, B, and C represent the proportions of first, second, and third prize winners respectively.
variables (h1 : A + B = 3 / 4)
variables (h2 : B + C = 2 / 3)

-- statement
theorem proportion_of_second_prize_winners : B = 5 / 12 :=
by
  sorry

end proportion_of_second_prize_winners_l83_83932


namespace teairras_pants_count_l83_83616

-- Definitions according to the given conditions
def total_shirts := 5
def plaid_shirts := 3
def purple_pants := 5
def neither_plaid_nor_purple := 21

-- The theorem we need to prove
theorem teairras_pants_count :
  ∃ (pants : ℕ), pants = (neither_plaid_nor_purple - (total_shirts - plaid_shirts)) + purple_pants ∧ pants = 24 :=
by
  sorry

end teairras_pants_count_l83_83616


namespace triangle_sides_length_a_triangle_perimeter_l83_83291

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end triangle_sides_length_a_triangle_perimeter_l83_83291


namespace inequality_holds_iff_m_range_l83_83339

theorem inequality_holds_iff_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 - 2 * m * x - 3 < 0) ↔ (-3 < m ∧ m ≤ 0) :=
by
  sorry

end inequality_holds_iff_m_range_l83_83339


namespace youngest_person_age_l83_83364

theorem youngest_person_age (total_age_now : ℕ) (total_age_when_born : ℕ) (Y : ℕ) (h1 : total_age_now = 210) (h2 : total_age_when_born = 162) : Y = 48 :=
by
  sorry

end youngest_person_age_l83_83364


namespace minValue_equality_l83_83072

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (a + 3 * b) * (b + 3 * c) * (a * c + 3)

theorem minValue_equality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  minValue a b c = 48 :=
sorry

end minValue_equality_l83_83072


namespace countSumPairs_correct_l83_83987

def countSumPairs (n : ℕ) : ℕ :=
  n / 2

theorem countSumPairs_correct (n : ℕ) : countSumPairs n = n / 2 := by
  sorry

end countSumPairs_correct_l83_83987


namespace area_of_triangle_formed_by_intercepts_l83_83214

theorem area_of_triangle_formed_by_intercepts :
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := [-3, 4]
  let y_intercept := 48
  let base := 7
  let height := 48
  let area := (1 / 2 : ℝ) * base * height
  area = 168 :=
by
  sorry

end area_of_triangle_formed_by_intercepts_l83_83214


namespace all_girls_probability_l83_83750

-- Definition of the problem conditions
def probability_of_girl : ℚ := 1 / 2
def events_independent (P1 P2 P3 : ℚ) : Prop := P1 * P2 = P1 ∧ P2 * P3 = P2

-- The statement to prove
theorem all_girls_probability :
  events_independent probability_of_girl probability_of_girl probability_of_girl →
  (probability_of_girl * probability_of_girl * probability_of_girl) = 1 / 8 := 
by
  intros h
  sorry

end all_girls_probability_l83_83750


namespace min_points_necessary_l83_83104

noncomputable def min_points_on_circle (circumference : ℝ) (dist1 dist2 : ℝ) : ℕ :=
  1304

theorem min_points_necessary :
  ∀ (circumference : ℝ) (dist1 dist2 : ℝ),
  circumference = 1956 →
  dist1 = 1 →
  dist2 = 2 →
  (min_points_on_circle circumference dist1 dist2) = 1304 :=
sorry

end min_points_necessary_l83_83104


namespace f_g_2_eq_256_l83_83865

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem f_g_2_eq_256 : f (g 2) = 256 := by
  sorry

end f_g_2_eq_256_l83_83865


namespace math_problem_l83_83960
noncomputable def sum_of_terms (a b c d : ℕ) : ℕ := a + b + c + d

theorem math_problem
  (x y : ℝ)
  (h₁ : x + y = 5)
  (h₂ : 5 * x * y = 7) :
  ∃ a b c d : ℕ, 
  x = (a + b * Real.sqrt c) / d ∧
  a = 25 ∧ b = 1 ∧ c = 485 ∧ d = 10 ∧ sum_of_terms a b c d = 521 := by
sorry

end math_problem_l83_83960


namespace graduation_ceremony_l83_83209

theorem graduation_ceremony (teachers administrators graduates chairs : ℕ) 
  (h1 : teachers = 20) 
  (h2 : administrators = teachers / 2) 
  (h3 : graduates = 50) 
  (h4 : chairs = 180) :
  (chairs - (teachers + administrators + graduates)) / graduates = 2 :=
by 
  sorry

end graduation_ceremony_l83_83209


namespace cost_of_child_ticket_l83_83414

theorem cost_of_child_ticket
  (total_seats : ℕ)
  (adult_ticket_price : ℕ)
  (num_children : ℕ)
  (total_revenue : ℕ)
  (H1 : total_seats = 250)
  (H2 : adult_ticket_price = 6)
  (H3 : num_children = 188)
  (H4 : total_revenue = 1124) :
  let num_adults := total_seats - num_children
  let revenue_from_adults := num_adults * adult_ticket_price
  let cost_of_child_ticket := (total_revenue - revenue_from_adults) / num_children
  cost_of_child_ticket = 4 :=
by
  sorry

end cost_of_child_ticket_l83_83414


namespace range_of_a_l83_83651

noncomputable def A : Set ℝ := {x | x^2 ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : a ≥ 1 := 
by
  sorry

end range_of_a_l83_83651


namespace digit_number_is_203_l83_83269

theorem digit_number_is_203 {A B C : ℕ} (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) :
  100 * A + 10 * B + C = 203 :=
by
  sorry

end digit_number_is_203_l83_83269


namespace monotonic_intervals_range_of_c_l83_83361

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := c * Real.log x + (1 / 2) * x ^ 2 + b * x

lemma extreme_point_condition {b c : ℝ} (h1 : c ≠ 0) (h2 : f 1 b c = 0) : b + c + 1 = 0 :=
sorry

theorem monotonic_intervals (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : c > 1) :
  (∀ x, 0 < x ∧ x < 1 → f 1 b c < f x b c) ∧ 
  (∀ x, 1 < x ∧ x < c → f 1 b c > f x b c) ∧ 
  (∀ x, x > c → f 1 b c < f x b c) :=
sorry

theorem range_of_c (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : (f 1 b c < 0)) :
  -1 / 2 < c ∧ c < 0 :=
sorry

end monotonic_intervals_range_of_c_l83_83361


namespace customer_total_payment_l83_83106

structure PaymentData where
  rate : ℕ
  discount1 : ℕ
  lateFee1 : ℕ
  discount2 : ℕ
  lateFee2 : ℕ
  discount3 : ℕ
  lateFee3 : ℕ
  discount4 : ℕ
  lateFee4 : ℕ
  onTime1 : Bool
  onTime2 : Bool
  onTime3 : Bool
  onTime4 : Bool

noncomputable def monthlyPayment (rate discount late_fee : ℕ) (onTime : Bool) : ℕ :=
  if onTime then rate - (rate * discount / 100) else rate + (rate * late_fee / 100)

theorem customer_total_payment (data : PaymentData) : 
  monthlyPayment data.rate data.discount1 data.lateFee1 data.onTime1 +
  monthlyPayment data.rate data.discount2 data.lateFee2 data.onTime2 +
  monthlyPayment data.rate data.discount3 data.lateFee3 data.onTime3 +
  monthlyPayment data.rate data.discount4 data.lateFee4 data.onTime4 = 195 := by
  sorry

end customer_total_payment_l83_83106


namespace middle_number_l83_83509

theorem middle_number {a b c : ℚ} 
  (h1 : a + b = 15) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) 
  (h4 : c = 2 * a) : 
  b = 25 / 3 := 
by 
  sorry

end middle_number_l83_83509


namespace jellybean_ratio_l83_83847

theorem jellybean_ratio (jellybeans_large: ℕ) (large_glasses: ℕ) (small_glasses: ℕ) (total_jellybeans: ℕ) (jellybeans_per_large: ℕ) (jellybeans_per_small: ℕ)
  (h1 : jellybeans_large = 50)
  (h2 : large_glasses = 5)
  (h3 : small_glasses = 3)
  (h4 : total_jellybeans = 325)
  (h5 : jellybeans_per_large = jellybeans_large * large_glasses)
  (h6 : jellybeans_per_small * small_glasses = total_jellybeans - jellybeans_per_large)
  : jellybeans_per_small = 25 ∧ jellybeans_per_small / jellybeans_large = 1 / 2 :=
by
  sorry

end jellybean_ratio_l83_83847


namespace gambler_final_amount_l83_83941

theorem gambler_final_amount :
  let initial_money := 100
  let win_multiplier := (3/2 : ℚ)
  let loss_multiplier := (1/2 : ℚ)
  let final_multiplier := (win_multiplier * loss_multiplier)^4
  let final_amount := initial_money * final_multiplier
  final_amount = (8100 / 256) :=
by
  sorry

end gambler_final_amount_l83_83941


namespace multiply_scientific_notation_l83_83923

theorem multiply_scientific_notation (a b : ℝ) (e1 e2 : ℤ) 
  (h1 : a = 2) (h2 : b = 8) (h3 : e1 = 3) (h4 : e2 = 3) :
  (a * 10^e1) * (b * 10^e2) = 1.6 * 10^7 :=
by
  simp [h1, h2, h3, h4]
  sorry

end multiply_scientific_notation_l83_83923


namespace find_first_odd_number_l83_83070

theorem find_first_odd_number (x : ℤ)
  (h : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) : x = 7 :=
by
  sorry

end find_first_odd_number_l83_83070


namespace units_digit_of_factorial_sum_l83_83061

theorem units_digit_of_factorial_sum : 
  (1 + 2 + 6 + 4) % 10 = 3 := sorry

end units_digit_of_factorial_sum_l83_83061


namespace polar_equation_parabola_l83_83281

/-- Given a polar equation 4 * ρ * (sin(θ / 2))^2 = 5, prove that it represents a parabola in Cartesian coordinates. -/
theorem polar_equation_parabola (ρ θ : ℝ) (h : 4 * ρ * (Real.sin (θ / 2))^ 2 = 5) : 
  ∃ (a : ℝ), a ≠ 0 ∧ (∃ b c : ℝ, ∀ x y : ℝ, (y^2 = a * (x + b)) ∨ (x = c ∨ y = 0)) := 
sorry

end polar_equation_parabola_l83_83281


namespace num_double_yolk_eggs_l83_83213

noncomputable def double_yolk_eggs (total_eggs total_yolks : ℕ) (double_yolk_contrib : ℕ) : ℕ :=
(total_yolks - total_eggs + double_yolk_contrib) / double_yolk_contrib

theorem num_double_yolk_eggs (total_eggs total_yolks double_yolk_contrib expected : ℕ)
    (h1 : total_eggs = 12)
    (h2 : total_yolks = 17)
    (h3 : double_yolk_contrib = 2)
    (h4 : expected = 5) :
  double_yolk_eggs total_eggs total_yolks double_yolk_contrib = expected :=
by
  rw [h1, h2, h3, h4]
  dsimp [double_yolk_eggs]
  norm_num
  sorry

end num_double_yolk_eggs_l83_83213


namespace nominal_rate_of_interest_l83_83134

noncomputable def nominal_rate (EAR : ℝ) (n : ℕ) : ℝ :=
  2 * (Real.sqrt (1 + EAR) - 1)

theorem nominal_rate_of_interest :
  nominal_rate 0.1025 2 = 0.100476 :=
by sorry

end nominal_rate_of_interest_l83_83134


namespace exists_good_number_in_interval_l83_83924

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≤ 5

theorem exists_good_number_in_interval (x : ℕ) (hx : x ≠ 0) :
  ∃ g : ℕ, is_good_number g ∧ x ≤ g ∧ g < ((9 * x) / 5) + 1 := 
sorry

end exists_good_number_in_interval_l83_83924


namespace distance_from_pole_to_line_l83_83479

-- Definitions based on the problem condition
def polar_equation_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3

-- The statement of the proof problem
theorem distance_from_pole_to_line (ρ θ : ℝ) (h : polar_equation_line ρ θ) :
  ρ = Real.sqrt 6 / 2 := sorry

end distance_from_pole_to_line_l83_83479


namespace gcd_apb_ab_eq1_gcd_aplusb_aminsb_l83_83689

theorem gcd_apb_ab_eq1 (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a * b) = 1 ∧ Int.gcd (a - b) (a * b) = 1 := by
  sorry

theorem gcd_aplusb_aminsb (a b : ℤ) (h : Int.gcd a b = 1) : 
  Int.gcd (a + b) (a - b) = 1 ∨ Int.gcd (a + b) (a - b) = 2 := by
  sorry

end gcd_apb_ab_eq1_gcd_aplusb_aminsb_l83_83689


namespace four_digit_sum_divisible_l83_83678

theorem four_digit_sum_divisible (A B C D : ℕ) :
  (10 * A + B + 10 * C + D = 94) ∧ (1000 * A + 100 * B + 10 * C + D % 94 = 0) →
  false :=
by
  sorry

end four_digit_sum_divisible_l83_83678


namespace lillian_candies_l83_83113

theorem lillian_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_candies : ℕ) :
  initial_candies = 88 → additional_candies = 5 → total_candies = initial_candies + additional_candies → total_candies = 93 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end lillian_candies_l83_83113


namespace jessica_minimal_withdrawal_l83_83149

theorem jessica_minimal_withdrawal 
  (initial_withdrawal : ℝ)
  (initial_fraction : ℝ)
  (minimum_balance : ℝ)
  (deposit_fraction : ℝ)
  (after_withdrawal_balance : ℝ)
  (deposit_amount : ℝ)
  (current_balance : ℝ) :
  initial_withdrawal = 400 →
  initial_fraction = 2/5 →
  minimum_balance = 300 →
  deposit_fraction = 1/4 →
  after_withdrawal_balance = 1000 - initial_withdrawal →
  deposit_amount = deposit_fraction * after_withdrawal_balance →
  current_balance = after_withdrawal_balance + deposit_amount →
  current_balance - minimum_balance ≥ 0 →
  0 = 0 :=
by
  sorry

end jessica_minimal_withdrawal_l83_83149


namespace find_divisor_l83_83023

theorem find_divisor (n : ℕ) (h_n : n = 36) : 
  ∃ D : ℕ, ((n + 10) * 2 / D) - 2 = 44 → D = 2 :=
by
  use 2
  intros h
  sorry

end find_divisor_l83_83023


namespace monotonic_decreasing_interval_l83_83659

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end monotonic_decreasing_interval_l83_83659


namespace distribute_candies_l83_83720

-- Definition of the problem conditions
def candies : ℕ := 10

-- The theorem stating the proof problem
theorem distribute_candies : (2 ^ (candies - 1)) = 512 := 
by
  sorry

end distribute_candies_l83_83720


namespace find_n_l83_83726

theorem find_n (n : ℕ) : 
  (1/5 : ℝ)^35 * (1/4 : ℝ)^n = (1 : ℝ) / (2 * 10^35) → n = 18 :=
by
  intro h
  sorry

end find_n_l83_83726


namespace lego_set_cost_l83_83698

-- Define the cost per doll and number of dolls
def costPerDoll : ℝ := 15
def numberOfDolls : ℝ := 4

-- Define the total amount spent on the younger sister's dolls
def totalAmountOnDolls : ℝ := numberOfDolls * costPerDoll

-- Define the number of lego sets
def numberOfLegoSets : ℝ := 3

-- Define the total amount spent on lego sets (needs to be equal to totalAmountOnDolls)
def totalAmountOnLegoSets : ℝ := 60

-- Define the cost per lego set that we need to prove
def costPerLegoSet : ℝ := 20

-- Theorem to prove that the cost per lego set is $20
theorem lego_set_cost (h : totalAmountOnLegoSets = totalAmountOnDolls) : 
  totalAmountOnLegoSets / numberOfLegoSets = costPerLegoSet := by
  sorry

end lego_set_cost_l83_83698


namespace greatest_drop_in_june_l83_83393

def monthly_changes := [("January", 1.50), ("February", -2.25), ("March", 0.75), ("April", -3.00), ("May", 1.00), ("June", -4.00)]

theorem greatest_drop_in_june : ∀ months : List (String × Float), (months = monthly_changes) → 
  (∃ month : String, 
    month = "June" ∧ 
    ∀ m p, m ≠ "June" → (m, p) ∈ months → p ≥ -4.00) :=
by
  sorry

end greatest_drop_in_june_l83_83393


namespace no_positive_integer_n_satisfies_l83_83781

theorem no_positive_integer_n_satisfies :
  ¬∃ (n : ℕ), (1000 ≤ n / 5 ∧ n / 5 ≤ 9999) ∧ (1000 ≤ 5 * n ∧ 5 * n ≤ 9999) := by
  sorry

end no_positive_integer_n_satisfies_l83_83781


namespace determinant_expression_l83_83767

theorem determinant_expression (a b c d p q r : ℝ)
  (h1: (∃ x: ℝ, x^4 + p*x^2 + q*x + r = 0) → (x = a ∨ x = b ∨ x = c ∨ x = d))
  (h2: a*b + a*c + a*d + b*c + b*d + c*d = p)
  (h3: a*b*c + a*b*d + a*c*d + b*c*d = q)
  (h4: a*b*c*d = -r):
  (Matrix.det ![![1 + a, 1, 1, 1], ![1, 1 + b, 1, 1], ![1, 1, 1 + c, 1], ![1, 1, 1, 1 + d]]) 
  = r + q + p := 
sorry

end determinant_expression_l83_83767


namespace Jonas_needs_to_buy_35_pairs_of_socks_l83_83384

theorem Jonas_needs_to_buy_35_pairs_of_socks
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)
  (double_items : ℕ)
  (needed_items : ℕ)
  (pairs_of_socks_needed : ℕ) :
  socks = 20 →
  shoes = 5 →
  pants = 10 →
  tshirts = 10 →
  double_items = 2 * (2 * socks + 2 * shoes + pants + tshirts) →
  needed_items = double_items - (2 * socks + 2 * shoes + pants + tshirts) →
  pairs_of_socks_needed = needed_items / 2 →
  pairs_of_socks_needed = 35 :=
by sorry

end Jonas_needs_to_buy_35_pairs_of_socks_l83_83384


namespace solve_system_of_inequalities_l83_83881

theorem solve_system_of_inequalities {x : ℝ} :
  (|x^2 + 5 * x| < 6) ∧ (|x + 1| ≤ 1) ↔ (0 ≤ x ∧ x < 2) ∨ (4 < x ∧ x ≤ 6) :=
by
  sorry

end solve_system_of_inequalities_l83_83881


namespace profit_percentage_l83_83886

-- Definitions and conditions
variable (SP : ℝ) (CP : ℝ)
variable (h : CP = 0.98 * SP)

-- Lean statement to prove the profit percentage is 2.04%
theorem profit_percentage (h : CP = 0.98 * SP) : (SP - CP) / CP * 100 = 2.04 := 
sorry

end profit_percentage_l83_83886


namespace fraction_inequality_l83_83730

theorem fraction_inequality (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) :
  (a / d) < (b / c) :=
sorry

end fraction_inequality_l83_83730


namespace diplomats_neither_french_nor_russian_l83_83009

variable (total_diplomats : ℕ)
variable (speak_french : ℕ)
variable (not_speak_russian : ℕ)
variable (speak_both : ℕ)

theorem diplomats_neither_french_nor_russian {total_diplomats speak_french not_speak_russian speak_both : ℕ} 
  (h1 : total_diplomats = 100)
  (h2 : speak_french = 22)
  (h3 : not_speak_russian = 32)
  (h4 : speak_both = 10) :
  ((total_diplomats - (speak_french + (total_diplomats - not_speak_russian) - speak_both)) * 100) / total_diplomats = 20 := 
by
  sorry

end diplomats_neither_french_nor_russian_l83_83009


namespace mean_score_40_l83_83925

theorem mean_score_40 (mean : ℝ) (std_dev : ℝ) (h_std_dev : std_dev = 10)
  (h_within_2_std_dev : ∀ (score : ℝ), score ≥ mean - 2 * std_dev)
  (h_lowest_score : ∀ (score : ℝ), score = 20 → score = mean - 20) :
  mean = 40 :=
by
  -- Placeholder for the proof
  sorry

end mean_score_40_l83_83925


namespace problem_inequality_l83_83853

theorem problem_inequality (a : ℝ) (h_pos : 0 < a) : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a :=
by 
  sorry

end problem_inequality_l83_83853


namespace age_difference_l83_83165

variable (A B C D : ℕ)

theorem age_difference (h₁ : A + B > B + C) (h₂ : C = A - 15) : (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l83_83165


namespace proposition_5_l83_83986

/-! 
  Proposition 5: If there are four points A, B, C, D in a plane, 
  then the vector addition relation: \overrightarrow{AC} + \overrightarrow{BD} = \overrightarrow{BC} + \overrightarrow{AD} must hold.
--/

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (AC BD BC AD : A)

-- Theorem Statement in Lean 4
theorem proposition_5 (AC BD BC AD : A)
  : AC + BD = BC + AD := by
  -- Proof by congruence and equality, will add actual steps here
  sorry

end proposition_5_l83_83986


namespace perimeter_hypotenuse_ratios_l83_83025

variable {x y : Real}
variable (h_pos_x : x > 0) (h_pos_y : y > 0)

theorem perimeter_hypotenuse_ratios
    (h_sides : (3 * x + 3 * y = (3 * x + 3 * y)) ∨ 
               (4 * x = (4 * x)) ∨
               (4 * y = (4 * y)))
    : 
    (∃ p : Real, p = 7 * (x + y) / (3 * (x + y)) ∨
                 p = 32 * y / (100 / 7 * y) ∨
                 p = 224 / 25 * y / 4 * y ∨ 
                 p = 7 / 3 ∨ 
                 p = 56 / 25) := by sorry

end perimeter_hypotenuse_ratios_l83_83025


namespace range_of_ab_c2_l83_83679

theorem range_of_ab_c2
  (a b c : ℝ)
  (h₁: -3 < b)
  (h₂: b < a)
  (h₃: a < -1)
  (h₄: -2 < c)
  (h₅: c < -1) :
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := 
by 
  sorry

end range_of_ab_c2_l83_83679


namespace smallest_flash_drives_l83_83476

theorem smallest_flash_drives (total_files : ℕ) (flash_drive_space: ℝ)
  (files_size : ℕ → ℝ)
  (h1 : total_files = 40)
  (h2 : flash_drive_space = 2.0)
  (h3 : ∀ n, (n < 4 → files_size n = 1.2) ∧ 
              (4 ≤ n ∧ n < 20 → files_size n = 0.9) ∧ 
              (20 ≤ n → files_size n = 0.6)) :
  ∃ min_flash_drives, min_flash_drives = 20 :=
sorry

end smallest_flash_drives_l83_83476


namespace sum_of_three_positives_eq_2002_l83_83192

theorem sum_of_three_positives_eq_2002 : 
  ∃ (n : ℕ), n = 334000 ∧ (∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ (A B C : ℕ), f A B C ↔ (0 < A ∧ A ≤ B ∧ B ≤ C ∧ A + B + C = 2002))) := by
  sorry

end sum_of_three_positives_eq_2002_l83_83192


namespace no_positive_integers_satisfy_l83_83787

theorem no_positive_integers_satisfy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 + 1 ≠ (x + 2)^5 + (y - 3)^5 :=
sorry

end no_positive_integers_satisfy_l83_83787


namespace rectangle_perimeter_l83_83685

theorem rectangle_perimeter {w l : ℝ} 
  (h_area : l * w = 450)
  (h_length : l = 2 * w) :
  2 * (l + w) = 90 :=
by sorry

end rectangle_perimeter_l83_83685
