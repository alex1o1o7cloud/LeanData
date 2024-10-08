import Mathlib

namespace james_profit_l167_167184

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l167_167184


namespace necessary_and_sufficient_condition_l167_167233

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (a^2 + 4 * a - 5 > 0) ↔ (|a + 2| > 3) := sorry

end necessary_and_sufficient_condition_l167_167233


namespace prob_odd_sum_l167_167710

-- Given conditions on the spinners
def spinner_P := [1, 2, 3]
def spinner_Q := [2, 4, 6]
def spinner_R := [1, 3, 5]

-- Probability of spinner P landing on an even number is 1/3
def prob_even_P : ℚ := 1 / 3

-- Probability of odd sum from spinners P, Q, and R
theorem prob_odd_sum : 
  (prob_even_P = 1 / 3) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end prob_odd_sum_l167_167710


namespace value_of_a_l167_167158

variable (a : ℝ)

theorem value_of_a (h1 : (0.5 / 100) * a = 0.80) : a = 160 := by
  sorry

end value_of_a_l167_167158


namespace shortest_side_of_triangle_l167_167202

theorem shortest_side_of_triangle 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_inequal : a^2 + b^2 > 5 * c^2) :
  c < a ∧ c < b := 
by 
  sorry

end shortest_side_of_triangle_l167_167202


namespace tourist_tax_l167_167406

theorem tourist_tax (total_value : ℝ) (non_taxable_amount : ℝ) (tax_rate : ℝ) 
  (h1 : total_value = 1720) (h2 : non_taxable_amount = 600) (h3 : tax_rate = 0.08) : 
  ((total_value - non_taxable_amount) * tax_rate = 89.60) :=
by 
  sorry

end tourist_tax_l167_167406


namespace min_distance_sum_coordinates_l167_167987

theorem min_distance_sum_coordinates (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, -1)) :
  ∃ P : ℝ × ℝ, P = (0, 3) ∧ ∀ Q : ℝ × ℝ, Q.1 = 0 → |A.1 - Q.1| + |A.2 - Q.2| + |B.1 - Q.1| + |B.2 - Q.2| ≥ |A.1 - (0 : ℝ)| + |A.2 - (3 : ℝ)| + |B.1 - (0 : ℝ)| + |B.2 - (3 : ℝ)| := 
sorry

end min_distance_sum_coordinates_l167_167987


namespace solve_quadratic_equation_l167_167037

theorem solve_quadratic_equation (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 :=
by
  sorry

end solve_quadratic_equation_l167_167037


namespace parabola_standard_equation_l167_167339

theorem parabola_standard_equation (directrix : ℝ) (h_directrix : directrix = 1) : 
  ∃ (a : ℝ), y^2 = a * x ∧ a = -4 :=
by
  sorry

end parabola_standard_equation_l167_167339


namespace sum_sequence_l167_167854

noncomputable def sum_first_n_minus_1_terms (n : ℕ) : ℕ :=
  (2^n - n - 1)

theorem sum_sequence (n : ℕ) : 
  sum_first_n_minus_1_terms n = (2^n - n - 1) :=
by
  sorry 

end sum_sequence_l167_167854


namespace problem_x_value_l167_167427

theorem problem_x_value (x : ℝ) (h : 0.25 * x = 0.15 * 1500 - 15) : x = 840 :=
by
  sorry

end problem_x_value_l167_167427


namespace geometric_series_sum_l167_167413

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 6
  S = a * (r ^ n - 1) / (r - 1) → S = 728 :=
by
  intros a r n h
  sorry

end geometric_series_sum_l167_167413


namespace min_value_of_S_l167_167329

variable (x : ℝ)
def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S : ∀ x : ℝ, S x ≥ 112.5 :=
by
  sorry

end min_value_of_S_l167_167329


namespace determinant_transformation_l167_167794

theorem determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
    p * (5 * r + 2 * s) - r * (5 * p + 2 * q) = -6 := by
  sorry

end determinant_transformation_l167_167794


namespace determine_a_range_f_l167_167800

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - (2 / (2 ^ x + 1))

theorem determine_a (a : ℝ) : (∀ x : ℝ, f a (-x) = -f a x) -> a = 1 :=
by
  sorry

theorem range_f (x : ℝ) : (∀ x : ℝ, f 1 (-x) = -f 1 x) -> -1 < f 1 x ∧ f 1 x < 1 :=
by
  sorry

end determine_a_range_f_l167_167800


namespace find_unknown_rate_l167_167145

variable (x : ℕ)

theorem find_unknown_rate
    (c3 : ℕ := 3 * 100)
    (c5 : ℕ := 5 * 150)
    (n : ℕ := 10)
    (avg_price : ℕ := 160) 
    (h : c3 + c5 + 2 * x = avg_price * n) :
    x = 275 := 
by
  -- Proof goes here.
  sorry

end find_unknown_rate_l167_167145


namespace can_form_all_numbers_l167_167735

noncomputable def domino_tiles : List (ℕ × ℕ) := [(1, 3), (6, 6), (6, 2), (3, 2)]

def form_any_number (n : ℕ) : Prop :=
  ∃ (comb : List (ℕ × ℕ)), comb ⊆ domino_tiles ∧ (comb.bind (λ p => [p.1, p.2])).sum = n

theorem can_form_all_numbers : ∀ n, 1 ≤ n → n ≤ 23 → form_any_number n :=
by sorry

end can_form_all_numbers_l167_167735


namespace number_of_students_in_first_class_l167_167833

theorem number_of_students_in_first_class 
  (x : ℕ) -- number of students in the first class
  (avg_first_class : ℝ := 50) 
  (num_second_class : ℕ := 50)
  (avg_second_class : ℝ := 60)
  (avg_all_students : ℝ := 56.25)
  (total_avg_eqn : (avg_first_class * x + avg_second_class * num_second_class) / (x + num_second_class) = avg_all_students) : 
  x = 30 :=
by sorry

end number_of_students_in_first_class_l167_167833


namespace sphere_radius_is_five_l167_167737

theorem sphere_radius_is_five
    (π : ℝ)
    (r r_cylinder h : ℝ)
    (A_sphere A_cylinder : ℝ)
    (h1 : A_sphere = 4 * π * r ^ 2)
    (h2 : A_cylinder = 2 * π * r_cylinder * h)
    (h3 : h = 10)
    (h4 : r_cylinder = 5)
    (h5 : A_sphere = A_cylinder) :
    r = 5 :=
by
  sorry

end sphere_radius_is_five_l167_167737


namespace provider_choices_count_l167_167471

theorem provider_choices_count :
  let num_providers := 25
  let num_s_providers := 6
  let remaining_providers_after_laura := num_providers - 1
  let remaining_providers_after_brother := remaining_providers_after_laura - 1

  (num_providers * num_s_providers * remaining_providers_after_laura * remaining_providers_after_brother) = 75900 :=
by
  sorry

end provider_choices_count_l167_167471


namespace new_rectangle_perimeters_l167_167971

theorem new_rectangle_perimeters {l w : ℕ} (h_l : l = 4) (h_w : w = 2) :
  (∃ P, P = 2 * (8 + 2) ∨ P = 2 * (4 + 4)) ∧ (P = 20 ∨ P = 16) :=
by
  sorry

end new_rectangle_perimeters_l167_167971


namespace absentees_in_morning_session_is_three_l167_167666

theorem absentees_in_morning_session_is_three
  (registered_morning : ℕ)
  (registered_afternoon : ℕ)
  (absent_afternoon : ℕ)
  (total_students : ℕ)
  (total_registered : ℕ)
  (attended_afternoon : ℕ)
  (attended_morning : ℕ)
  (absent_morning : ℕ) :
  registered_morning = 25 →
  registered_afternoon = 24 →
  absent_afternoon = 4 →
  total_students = 42 →
  total_registered = registered_morning + registered_afternoon →
  attended_afternoon = registered_afternoon - absent_afternoon →
  attended_morning = total_students - attended_afternoon →
  absent_morning = registered_morning - attended_morning →
  absent_morning = 3 :=
by
  intros
  sorry

end absentees_in_morning_session_is_three_l167_167666


namespace line_equation_mb_l167_167111

theorem line_equation_mb (b m : ℤ) (h_b : b = -2) (h_m : m = 5) : m * b = -10 :=
by
  rw [h_b, h_m]
  norm_num

end line_equation_mb_l167_167111


namespace number_of_students_taking_math_l167_167395

variable (totalPlayers physicsOnly physicsAndMath mathOnly : ℕ)
variable (h1 : totalPlayers = 15) (h2 : physicsOnly = 9) (h3 : physicsAndMath = 3)

theorem number_of_students_taking_math : mathOnly = 9 :=
by {
  sorry
}

end number_of_students_taking_math_l167_167395


namespace selling_price_eq_120_l167_167268

-- Definitions based on the conditions
def cost_price : ℝ := 96
def profit_percentage : ℝ := 0.25

-- The proof statement
theorem selling_price_eq_120 (cost_price : ℝ) (profit_percentage : ℝ) : cost_price = 96 → profit_percentage = 0.25 → (cost_price + cost_price * profit_percentage) = 120 :=
by
  intros hcost hprofit
  rw [hcost, hprofit]
  sorry

end selling_price_eq_120_l167_167268


namespace third_consecutive_even_l167_167375

theorem third_consecutive_even {a b c d : ℕ} (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h_sum : a + b + c + d = 52) : c = 14 :=
by
  sorry

end third_consecutive_even_l167_167375


namespace systematic_sampling_first_group_l167_167518

theorem systematic_sampling_first_group (S : ℕ) (n : ℕ) (students_per_group : ℕ) (group_number : ℕ)
(h1 : n = 160)
(h2 : students_per_group = 8)
(h3 : group_number = 16)
(h4 : S + (group_number - 1) * students_per_group = 126)
: S = 6 := by
  sorry

end systematic_sampling_first_group_l167_167518


namespace part_a_part_b_l167_167256

-- Part (a): Prove that for N = a^2 + 2, the equation has positive integral solutions for infinitely many a.
theorem part_a (N : ℕ) (a : ℕ) (x y z t : ℕ) (hx : x = a * (a^2 + 2)) (hy : y = a) (hz : z = 1) (ht : t = 1) :
  (∃ (N : ℕ), ∀ (a : ℕ), ∃ (x y z t : ℕ),
    x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N ∧
    x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) :=
sorry

-- Part (b): Prove that for N = 4^k(8m + 7), the equation has no positive integral solutions.
theorem part_b (N : ℕ) (k m : ℕ) (x y z t : ℕ) (hN : N = 4^k * (8 * m + 7)) :
  ¬ (x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ x^2 + y^2 + z^2 + t^2 = N * x * y * z * t + N) :=
sorry

end part_a_part_b_l167_167256


namespace car_speed_conversion_l167_167309

noncomputable def miles_to_yards : ℕ :=
  1760

theorem car_speed_conversion (speed_mph : ℕ) (time_sec : ℝ) (distance_yards : ℕ) :
  speed_mph = 90 →
  time_sec = 0.5 →
  distance_yards = 22 →
  (1 : ℕ) * miles_to_yards = 1760 := by
  intros h1 h2 h3
  sorry

end car_speed_conversion_l167_167309


namespace factor_expression_l167_167697

theorem factor_expression (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2) ^ 2 :=
by sorry

end factor_expression_l167_167697


namespace find_a1_l167_167378

variable {a_n : ℕ → ℤ}
variable (common_difference : ℤ) (a1 : ℤ)

-- Define that a_n is an arithmetic sequence with common difference of 2
def is_arithmetic_seq (a_n : ℕ → ℤ) (common_difference : ℤ) : Prop :=
  ∀ n, a_n (n + 1) - a_n n = common_difference

-- State the condition that a1, a2, a4 form a geometric sequence
def forms_geometric_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ a1 a2 a4, a2 * a2 = a1 * a4 ∧ a_n 1 = a1 ∧ a_n 2 = a2 ∧ a_n 4 = a4

-- Define the problem statement
theorem find_a1 (h_arith : is_arithmetic_seq a_n 2) (h_geom : forms_geometric_seq a_n) :
  a_n 1 = 2 :=
by
  sorry

end find_a1_l167_167378


namespace puzzle_pieces_l167_167572

theorem puzzle_pieces
  (total_puzzles : ℕ)
  (pieces_per_10_min : ℕ)
  (total_minutes : ℕ)
  (h1 : total_puzzles = 2)
  (h2 : pieces_per_10_min = 100)
  (h3 : total_minutes = 400) :
  ((total_minutes / 10) * pieces_per_10_min) / total_puzzles = 2000 :=
by
  sorry

end puzzle_pieces_l167_167572


namespace sheila_will_attend_picnic_l167_167561

def P_Rain : ℝ := 0.3
def P_Cloudy : ℝ := 0.4
def P_Sunny : ℝ := 0.3

def P_Attend_if_Rain : ℝ := 0.25
def P_Attend_if_Cloudy : ℝ := 0.5
def P_Attend_if_Sunny : ℝ := 0.75

def P_Attend : ℝ :=
  P_Rain * P_Attend_if_Rain +
  P_Cloudy * P_Attend_if_Cloudy +
  P_Sunny * P_Attend_if_Sunny

theorem sheila_will_attend_picnic : P_Attend = 0.5 := by
  sorry

end sheila_will_attend_picnic_l167_167561


namespace max_sum_first_n_terms_formula_sum_terms_abs_l167_167967

theorem max_sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  ∃ (n : ℕ), n = 15 ∧ S 15 = 225 := by
  sorry

theorem formula_sum_terms_abs (a : ℕ → ℤ) (S T : ℕ → ℤ) :
  a 1 = 29 ∧ S 10 = S 20 →
  (∀ n, n ≤ 15 → T n = 30 * n - n * n) ∧
  (∀ n, n ≥ 16 → T n = n * n - 30 * n + 450) := by
  sorry

end max_sum_first_n_terms_formula_sum_terms_abs_l167_167967


namespace A_investment_l167_167213

variable (x : ℕ)
variable (A_share : ℕ := 3780)
variable (Total_profit : ℕ := 12600)
variable (B_invest : ℕ := 4200)
variable (C_invest : ℕ := 10500)

theorem A_investment :
  (A_share : ℝ) / (Total_profit : ℝ) = (x : ℝ) / (x + B_invest + C_invest) →
  x = 6300 :=
by
  sorry

end A_investment_l167_167213


namespace min_value_of_a_l167_167031

variables (a b c d : ℕ)

-- Conditions
def conditions : Prop :=
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2004 ∧
  a^2 - b^2 + c^2 - d^2 = 2004

-- Theorem: minimum value of a
theorem min_value_of_a (h : conditions a b c d) : a = 503 :=
sorry

end min_value_of_a_l167_167031


namespace find_a_range_of_a_l167_167336

noncomputable def f (x a : ℝ) := x + a * Real.log x

-- Proof problem 1: Prove that a = 2 given f' (1) = 3 for f (x) = x + a log x
theorem find_a (a : ℝ) : 
  (1 + a = 3) → (a = 2) := sorry

-- Proof problem 2: Prove that the range of a such that f(x) ≥ a always holds is [-e^2, 0]
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ a) → (-Real.exp 2 ≤ a ∧ a ≤ 0) := sorry

end find_a_range_of_a_l167_167336


namespace average_headcount_11600_l167_167788

theorem average_headcount_11600 : 
  let h02_03 := 11700
  let h03_04 := 11500
  let h04_05 := 11600
  (h02_03 + h03_04 + h04_05) / 3 = 11600 := 
by
  sorry

end average_headcount_11600_l167_167788


namespace stock_return_to_original_l167_167139

theorem stock_return_to_original (x : ℝ) : 
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  1.56 * (1 - p/100) = 1 :=
by
  intro x
  let price_2006 := x
  let price_end_2006 := 1.30 * price_2006
  let price_end_2007 := 1.20 * price_end_2006
  let p := (0.56 * 100 / 1.56)
  show 1.56 * (1 - p / 100) = 1
  sorry

end stock_return_to_original_l167_167139


namespace line_of_intersection_canonical_form_l167_167459

def canonical_form_of_line (A B : ℝ) (x y z : ℝ) :=
  (x / A) = (y / B) ∧ (y / B) = (z)

theorem line_of_intersection_canonical_form :
  ∀ (x y z : ℝ),
  x + y - 2*z - 2 = 0 →
  x - y + z + 2 = 0 →
  canonical_form_of_line (-1) (-3) x (y - 2) (-2) :=
by
  intros x y z h_eq1 h_eq2
  sorry

end line_of_intersection_canonical_form_l167_167459


namespace physics_teacher_min_count_l167_167004

theorem physics_teacher_min_count 
  (maths_teachers : ℕ) 
  (chemistry_teachers : ℕ) 
  (max_subjects_per_teacher : ℕ) 
  (min_total_teachers : ℕ) 
  (physics_teachers : ℕ)
  (h1 : maths_teachers = 7)
  (h2 : chemistry_teachers = 5)
  (h3 : max_subjects_per_teacher = 3)
  (h4 : min_total_teachers = 6) 
  (h5 : 7 + physics_teachers + 5 ≤ 6 * 3) :
  0 < physics_teachers :=
  by 
  sorry

end physics_teacher_min_count_l167_167004


namespace calculate_lives_lost_l167_167928

-- Define the initial number of lives
def initial_lives : ℕ := 98

-- Define the remaining number of lives
def remaining_lives : ℕ := 73

-- Define the number of lives lost
def lives_lost : ℕ := initial_lives - remaining_lives

-- Prove that Kaleb lost 25 lives
theorem calculate_lives_lost : lives_lost = 25 := 
by {
  -- The proof would go here, but we'll skip it
  sorry
}

end calculate_lives_lost_l167_167928


namespace second_discount_percentage_l167_167255

-- Define the original price as P
variables {P : ℝ} (hP : P > 0)

-- Define the price increase by 34%
def price_after_increase (P : ℝ) := 1.34 * P

-- Define the first discount of 10%
def price_after_first_discount (P : ℝ) := 0.90 * (price_after_increase P)

-- Define the second discount percentage as D (in decimal form)
variables {D : ℝ}

-- Define the price after the second discount
def price_after_second_discount (P D : ℝ) := (1 - D) * (price_after_first_discount P)

-- Define the overall percentage gain of 2.51%
def final_price (P : ℝ) := 1.0251 * P

-- The main theorem to prove
theorem second_discount_percentage (hP : P > 0) (hD : 0 ≤ D ∧ D ≤ 1) :
  price_after_second_discount P D = final_price P ↔ D = 0.1495 :=
by
  sorry

end second_discount_percentage_l167_167255


namespace dart_prob_center_square_l167_167933

noncomputable def hexagon_prob (s : ℝ) : ℝ :=
  let square_area := s^2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  square_area / hexagon_area

theorem dart_prob_center_square (s : ℝ) : hexagon_prob s = 2 * Real.sqrt 3 / 9 :=
by
  -- Proof omitted
  sorry

end dart_prob_center_square_l167_167933


namespace pencils_difference_l167_167410

theorem pencils_difference
  (pencils_in_backpack : ℕ := 2)
  (pencils_at_home : ℕ := 15) :
  pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end pencils_difference_l167_167410


namespace men_l167_167024

-- Given conditions
variable (W M : ℕ)
variable (B : ℕ) [DecidableEq ℕ] -- number of boys
variable (total_earnings : ℕ)

def earnings : ℕ := 5 * M + W * M + 8 * W

-- Total earnings of men, women, and boys is Rs. 150.
def conditions : Prop := 
  5 * M = W * M ∧ 
  W * M = 8 * W ∧ 
  earnings = total_earnings

-- Prove men's wages (total wages for 5 men) is Rs. 50.
theorem men's_wages (hm : total_earnings = 150) (hb : W = 8) : 
  5 * M = 50 :=
by
  sorry

end men_l167_167024


namespace average_abc_l167_167435

theorem average_abc (A B C : ℚ) 
  (h1 : 2002 * C - 3003 * A = 6006) 
  (h2 : 2002 * B + 4004 * A = 8008) 
  (h3 : B - C = A + 1) :
  (A + B + C) / 3 = 7 / 3 := 
sorry

end average_abc_l167_167435


namespace real_solutions_count_l167_167678

theorem real_solutions_count :
  (∃ x : ℝ, |x - 2| - 4 = 1 / |x - 3|) ∧
  (∃ y : ℝ, |y - 2| - 4 = 1 / |y - 3| ∧ x ≠ y) :=
sorry

end real_solutions_count_l167_167678


namespace problem_l167_167576

-- Define the functions f and g with their properties
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Express the given conditions in Lean
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom g_def : ∀ x : ℝ, g x = f (x - 1)
axiom f_at_2 : f 2 = 2

-- What we need to prove
theorem problem : f 2014 = 2 := 
by sorry

end problem_l167_167576


namespace chess_team_selection_l167_167463

theorem chess_team_selection
  (players : Finset ℕ) (twin1 twin2 : ℕ)
  (H1 : players.card = 10)
  (H2 : twin1 ∈ players)
  (H3 : twin2 ∈ players) :
  ∃ n : ℕ, n = 182 ∧ 
  (∃ team : Finset ℕ, team.card = 4 ∧
    (twin1 ∉ team ∨ twin2 ∉ team)) ∧
  n = (players.card.choose 4 - 
      ((players.erase twin1).erase twin2).card.choose 2) := sorry

end chess_team_selection_l167_167463


namespace standard_heat_of_formation_Fe2O3_l167_167407

def Q_form_Al2O3 := 1675.5 -- kJ/mol

def Q1 := 854.2 -- kJ

-- Definition of the standard heat of formation of Fe2O3
def Q_form_Fe2O3 := Q_form_Al2O3 - Q1

-- The proof goal
theorem standard_heat_of_formation_Fe2O3 : Q_form_Fe2O3 = 821.3 := by
  sorry

end standard_heat_of_formation_Fe2O3_l167_167407


namespace ellipse_eccentricity_range_of_ratio_l167_167118

-- The setup conditions
variables {a b c : ℝ}
variables (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (h1 : a^2 - b^2 = c^2)
variables (M : ℝ) (m : ℝ)
variables (hM : M = a + c) (hm : m = a - c) (hMm : M * m = 3 / 4 * a^2)

-- Proof statement for the eccentricity of the ellipse
theorem ellipse_eccentricity : c / a = 1 / 2 := by
  sorry

-- The setup for the second part
variables {S1 S2 : ℝ}
variables (ellipse_eq : ∀ x y : ℝ, (x^2 / (4 * c^2) + y^2 / (3 * c^2) = 1) → x + y = 0)
variables (range_S : S1 / S2 > 9)

-- Proof statement for the range of the given ratio
theorem range_of_ratio : 0 < (2 * S1 * S2) / (S1^2 + S2^2) ∧ (2 * S1 * S2) / (S1^2 + S2^2) < 9 / 41 := by
  sorry

end ellipse_eccentricity_range_of_ratio_l167_167118


namespace combined_points_correct_l167_167069

-- Definitions for the points scored by each player
def points_Lemuel := 7 * 2 + 5 * 3 + 4
def points_Marcus := 4 * 2 + 6 * 3 + 7
def points_Kevin := 9 * 2 + 4 * 3 + 5
def points_Olivia := 6 * 2 + 3 * 3 + 6

-- Definition for the combined points scored by both teams
def combined_points := points_Lemuel + points_Marcus + points_Kevin + points_Olivia

-- Theorem statement to prove combined points equals 128
theorem combined_points_correct : combined_points = 128 :=
by
  -- Lean proof goes here
  sorry

end combined_points_correct_l167_167069


namespace cars_parked_l167_167565

def front_parking_spaces : ℕ := 52
def back_parking_spaces : ℕ := 38
def filled_back_spaces : ℕ := back_parking_spaces / 2
def available_spaces : ℕ := 32
def total_parking_spaces : ℕ := front_parking_spaces + back_parking_spaces
def filled_spaces : ℕ := total_parking_spaces - available_spaces

theorem cars_parked : 
  filled_spaces = 58 := by
  sorry

end cars_parked_l167_167565


namespace cheaper_fluid_cost_is_20_l167_167884

variable (x : ℕ) -- Denote the cost per drum of the cheaper fluid as x

-- Given conditions:
variable (total_drums : ℕ) (cheaper_drums : ℕ) (expensive_cost : ℕ) (total_cost : ℕ)
variable (remaining_drums : ℕ) (total_expensive_cost : ℕ)

axiom total_drums_eq : total_drums = 7
axiom cheaper_drums_eq : cheaper_drums = 5
axiom expensive_cost_eq : expensive_cost = 30
axiom total_cost_eq : total_cost = 160
axiom remaining_drums_eq : remaining_drums = total_drums - cheaper_drums
axiom total_expensive_cost_eq : total_expensive_cost = remaining_drums * expensive_cost

-- The equation for the total cost:
axiom total_cost_eq2 : total_cost = cheaper_drums * x + total_expensive_cost

-- The goal: Prove that the cheaper fluid cost per drum is $20
theorem cheaper_fluid_cost_is_20 : x = 20 :=
by
  sorry

end cheaper_fluid_cost_is_20_l167_167884


namespace triangle_perimeter_l167_167121

theorem triangle_perimeter (L R B : ℕ) (hL : L = 12) (hR : R = L + 2) (hB : B = 24) : L + R + B = 50 :=
by
  -- proof steps go here
  sorry

end triangle_perimeter_l167_167121


namespace stickers_per_student_l167_167968

theorem stickers_per_student (G S B N: ℕ) (hG: G = 50) (hS: S = 2 * G) (hB: B = S - 20) (hN: N = 5) : 
  (G + S + B) / N = 46 := by
  sorry

end stickers_per_student_l167_167968


namespace correct_average_is_15_l167_167198

theorem correct_average_is_15 (n incorrect_avg correct_num wrong_num : ℕ) 
  (h1 : n = 10) (h2 : incorrect_avg = 14) (h3 : correct_num = 36) (h4 : wrong_num = 26) : 
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 15 := 
by 
  sorry

end correct_average_is_15_l167_167198


namespace find_K_l167_167465

theorem find_K (surface_area_cube : ℝ) (volume_sphere : ℝ) (r : ℝ) (K : ℝ) 
  (cube_side_length : ℝ) (surface_area_sphere_eq : surface_area_cube = 4 * Real.pi * (r ^ 2))
  (volume_sphere_eq : volume_sphere = (4 / 3) * Real.pi * (r ^ 3)) 
  (surface_area_cube_eq : surface_area_cube = 6 * (cube_side_length ^ 2)) 
  (volume_sphere_form : volume_sphere = (K * Real.sqrt 6) / Real.sqrt Real.pi) :
  K = 8 :=
by
  sorry

end find_K_l167_167465


namespace Fabian_total_cost_correct_l167_167394

noncomputable def total_spent_by_Fabian (mouse_cost : ℝ) : ℝ :=
  let keyboard_cost := 2 * mouse_cost
  let headphones_cost := mouse_cost + 15
  let usb_hub_cost := 36 - mouse_cost
  let webcam_cost := keyboard_cost / 2
  let total_cost := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost + webcam_cost
  let discounted_total := total_cost * 0.90
  let final_total := discounted_total * 1.05
  final_total

theorem Fabian_total_cost_correct :
  total_spent_by_Fabian 20 = 123.80 :=
by
  sorry

end Fabian_total_cost_correct_l167_167394


namespace rin_craters_difference_l167_167022

theorem rin_craters_difference (d da r : ℕ) (h1 : d = 35) (h2 : da = d - 10) (h3 : r = 75) :
  r - (d + da) = 15 :=
by
  sorry

end rin_craters_difference_l167_167022


namespace remainder_of_876539_div_7_l167_167520

theorem remainder_of_876539_div_7 : 876539 % 7 = 6 :=
by
  sorry

end remainder_of_876539_div_7_l167_167520


namespace product_of_radii_l167_167720

theorem product_of_radii (x y r₁ r₂ : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hr₁ : (x - r₁)^2 + (y - r₁)^2 = r₁^2)
  (hr₂ : (x - r₂)^2 + (y - r₂)^2 = r₂^2)
  (hroots : r₁ + r₂ = 2 * (x + y)) : r₁ * r₂ = x^2 + y^2 := by
  sorry

end product_of_radii_l167_167720


namespace equation_of_l_symmetric_point_l167_167591

/-- Define points O, A, B in the coordinate plane --/
def O := (0, 0)
def A := (2, 0)
def B := (3, 2)

/-- Define midpoint of OA --/
def midpoint_OA := ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

/-- Line l passes through midpoint_OA and B. Prove line l has equation y = x - 1 --/
theorem equation_of_l :
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

/-- Prove the symmetric point of A with respect to line l is (1, 1) --/
theorem symmetric_point :
  ∃ (a b : ℝ), (a, b) = (1, 1) ∧
                (b * (2 - 1)) / (a - 2) = -1 ∧
                b / 2 = (2 + a - 1) / 2 - 1 :=
sorry

end equation_of_l_symmetric_point_l167_167591


namespace rounding_estimate_lt_exact_l167_167081

variable (a b c a' b' c' : ℕ)

theorem rounding_estimate_lt_exact (ha : a' ≤ a) (hb : b' ≥ b) (hc : c' ≤ c) (hb_pos : b > 0) (hb'_pos : b' > 0) :
  (a':ℚ) / (b':ℚ) + (c':ℚ) < (a:ℚ) / (b:ℚ) + (c:ℚ) :=
sorry

end rounding_estimate_lt_exact_l167_167081


namespace tempo_insured_fraction_l167_167099

theorem tempo_insured_fraction (premium : ℝ) (rate : ℝ) (original_value : ℝ) (h1 : premium = 300) (h2 : rate = 0.03) (h3 : original_value = 14000) : 
  premium / rate / original_value = 5 / 7 :=
by 
  sorry

end tempo_insured_fraction_l167_167099


namespace salary_reduction_l167_167215

noncomputable def percentageIncrease : ℝ := 16.27906976744186 / 100

theorem salary_reduction (S R : ℝ) (P : ℝ) (h1 : R = S * (1 - P / 100)) (h2 : S = R * (1 + percentageIncrease)) : P = 14 :=
by
  sorry

end salary_reduction_l167_167215


namespace unique_10_digit_number_property_l167_167762

def ten_digit_number (N : ℕ) : Prop :=
  10^9 ≤ N ∧ N < 10^10

def first_digits_coincide (N : ℕ) : Prop :=
  ∀ M : ℕ, N^2 < 10^M → N^2 / 10^(M - 10) = N

theorem unique_10_digit_number_property :
  ∀ (N : ℕ), ten_digit_number N ∧ first_digits_coincide N → N = 1000000000 := 
by
  intros N hN
  sorry

end unique_10_digit_number_property_l167_167762


namespace opposite_of_2023_is_neg_2023_l167_167526

theorem opposite_of_2023_is_neg_2023 (x : ℝ) (h : x = 2023) : -x = -2023 :=
by
  /- proof begins here, but we are skipping it with sorry -/
  sorry

end opposite_of_2023_is_neg_2023_l167_167526


namespace arithmetic_mean_is_correct_l167_167585

-- Define the numbers
def num1 : ℕ := 18
def num2 : ℕ := 27
def num3 : ℕ := 45

-- Define the number of terms
def n : ℕ := 3

-- Define the sum of the numbers
def total_sum : ℕ := num1 + num2 + num3

-- Define the arithmetic mean
def arithmetic_mean : ℕ := total_sum / n

-- Theorem stating that the arithmetic mean of the numbers is 30
theorem arithmetic_mean_is_correct : arithmetic_mean = 30 := by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l167_167585


namespace problem_abcd_eq_14400_l167_167839

theorem problem_abcd_eq_14400
 (a b c d : ℝ)
 (h1 : a^2 + b^2 + c^2 + d^2 = 762)
 (h2 : a * b + c * d = 260)
 (h3 : a * c + b * d = 365)
 (h4 : a * d + b * c = 244) :
 a * b * c * d = 14400 := 
sorry

end problem_abcd_eq_14400_l167_167839


namespace BurjKhalifaHeight_l167_167799

def SearsTowerHeight : ℕ := 527
def AdditionalHeight : ℕ := 303

theorem BurjKhalifaHeight : (SearsTowerHeight + AdditionalHeight) = 830 :=
by
  sorry

end BurjKhalifaHeight_l167_167799


namespace random_event_is_crane_among_chickens_l167_167064

-- Definitions of the idioms as events
def coveringTheSkyWithOneHand : Prop := false
def fumingFromAllSevenOrifices : Prop := false
def stridingLikeAMeteor : Prop := false
def standingOutLikeACraneAmongChickens : Prop := ¬false

-- The theorem stating that Standing out like a crane among chickens is a random event
theorem random_event_is_crane_among_chickens :
  ¬coveringTheSkyWithOneHand ∧ ¬fumingFromAllSevenOrifices ∧ ¬stridingLikeAMeteor → standingOutLikeACraneAmongChickens :=
by 
  sorry

end random_event_is_crane_among_chickens_l167_167064


namespace largest_number_eq_l167_167686

theorem largest_number_eq (x y z : ℚ) (h1 : x + y + z = 82) (h2 : z - y = 10) (h3 : y - x = 4) :
  z = 106 / 3 :=
sorry

end largest_number_eq_l167_167686


namespace can_combine_fig1_can_combine_fig2_l167_167283

-- Given areas for rectangle partitions
variables (S1 S2 S3 S4 : ℝ)
-- Condition: total area of black rectangles equals total area of white rectangles
variable (h1 : S1 + S2 = S3 + S4)

-- Proof problem for Figure 1
theorem can_combine_fig1 : ∃ A : ℝ, S1 + S2 = A ∧ S3 + S4 = A := by
  sorry

-- Proof problem for Figure 2
theorem can_combine_fig2 : ∃ B : ℝ, S1 + S2 = B ∧ S3 + S4 = B := by
  sorry

end can_combine_fig1_can_combine_fig2_l167_167283


namespace cost_price_per_meter_l167_167212

theorem cost_price_per_meter (number_of_meters : ℕ) (selling_price : ℝ) (profit_per_meter : ℝ) (total_cost_price : ℝ) (cost_per_meter : ℝ) :
  number_of_meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 15 →
  total_cost_price = selling_price - (profit_per_meter * number_of_meters) →
  cost_per_meter = total_cost_price / number_of_meters →
  cost_per_meter = 90 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end cost_price_per_meter_l167_167212


namespace derivative_of_f_l167_167353

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 * x - 2 :=
by
  intro x
  -- proof skipped
  sorry

end derivative_of_f_l167_167353


namespace arthur_initial_amount_l167_167901

def initial_amount (X : ℝ) : Prop :=
  (1/5) * X = 40

theorem arthur_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by
  sorry

end arthur_initial_amount_l167_167901


namespace tennis_tournament_l167_167059

theorem tennis_tournament (n : ℕ) (w m : ℕ) 
  (total_matches : ℕ)
  (women_wins men_wins : ℕ) :
  n + 2 * n = 3 * n →
  total_matches = (3 * n * (3 * n - 1)) / 2 →
  women_wins + men_wins = total_matches →
  women_wins / men_wins = 7 / 5 →
  n = 3 :=
by sorry

end tennis_tournament_l167_167059


namespace unique_prime_sum_diff_l167_167260

theorem unique_prime_sum_diff :
  ∀ p : ℕ, Prime p ∧ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ (p = p1 + 2) ∧ (p = p3 - 2)) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l167_167260


namespace five_coins_all_heads_or_tails_l167_167791

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end five_coins_all_heads_or_tails_l167_167791


namespace max_value_of_perfect_sequence_l167_167316

def isPerfectSequence (c : ℕ → ℕ) : Prop := ∀ n m : ℕ, 1 ≤ m ∧ m ≤ (Finset.range (n + 1)).sum (fun k => c k) → 
  ∃ (a : ℕ → ℕ), m = (Finset.range (n + 1)).sum (fun k => c k / a k)

theorem max_value_of_perfect_sequence (n : ℕ) : 
  ∃ c : ℕ → ℕ, isPerfectSequence c ∧
    (∀ i, i ≤ n → c i ≤ if i = 1 then 2 else 4 * 3^(i - 2)) ∧
    c n = if n = 1 then 2 else 4 * 3^(n - 2) :=
by
  sorry

end max_value_of_perfect_sequence_l167_167316


namespace circumference_difference_l167_167180

theorem circumference_difference (r : ℝ) (width : ℝ) (hp : width = 10.504226244065093) : 
  2 * Real.pi * (r + width) - 2 * Real.pi * r = 66.00691339889247 := by
  sorry

end circumference_difference_l167_167180


namespace rectangle_perimeter_of_triangle_area_l167_167813

theorem rectangle_perimeter_of_triangle_area
  (h_right : ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 12 ∧ c = 15)
  (rect_length : ℕ) 
  (rect_area_eq_triangle_area : ∃ (area : ℕ), area = 1/2 * 9 * 12 ∧ area = rect_length * rect_width ) 
  : ∃ (perimeter : ℕ), perimeter = 2 * (6 + rect_width) ∧ perimeter = 30 :=
sorry

end rectangle_perimeter_of_triangle_area_l167_167813


namespace expression_value_l167_167176

-- The problem statement definition
def expression := 2 + 3 * 4 - 5 / 5 + 7

-- Theorem statement asserting the final result
theorem expression_value : expression = 20 := 
by sorry

end expression_value_l167_167176


namespace ratio_AB_CD_lengths_AB_CD_l167_167003

theorem ratio_AB_CD 
  (AM MD BN NC : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  : (AM / MD) / (BN / NC) = 5 / 6 :=
by
  sorry

theorem lengths_AB_CD
  (AM MD BN NC AB CD : ℝ)
  (h_AM : AM = 25 / 7)
  (h_MD : MD = 10 / 7)
  (h_BN : BN = 3 / 2)
  (h_NC : NC = 9 / 2)
  (AB_div_CD : (AM / MD) / (BN / NC) = 5 / 6)
  (h_touch : true)  -- A placeholder condition indicating circles touch each other
  : AB = 5 ∧ CD = 6 :=
by
  sorry

end ratio_AB_CD_lengths_AB_CD_l167_167003


namespace sum_of_five_primes_is_145_l167_167560

-- Condition: common difference is 12
def common_difference : ℕ := 12

-- Five prime numbers forming an arithmetic sequence with the given common difference
def a1 : ℕ := 5
def a2 : ℕ := a1 + common_difference
def a3 : ℕ := a2 + common_difference
def a4 : ℕ := a3 + common_difference
def a5 : ℕ := a4 + common_difference

-- The sum of the arithmetic sequence
def sum_of_primes : ℕ := a1 + a2 + a3 + a4 + a5

-- Prove that the sum of these five prime numbers is 145
theorem sum_of_five_primes_is_145 : sum_of_primes = 145 :=
by
  -- Proof goes here
  sorry

end sum_of_five_primes_is_145_l167_167560


namespace number_of_girls_l167_167609

variable (g b : ℕ) -- Number of girls (g) and boys (b) in the class
variable (h_ratio : g / b = 4 / 3) -- The ratio condition
variable (h_total : g + b = 63) -- The total number of students condition

theorem number_of_girls (g b : ℕ) (h_ratio : g / b = 4 / 3) (h_total : g + b = 63) :
    g = 36 :=
sorry

end number_of_girls_l167_167609


namespace total_ticket_sales_cost_l167_167535

theorem total_ticket_sales_cost
  (num_orchestra num_balcony : ℕ)
  (price_orchestra price_balcony : ℕ)
  (total_tickets total_revenue : ℕ)
  (h1 : num_orchestra + num_balcony = 370)
  (h2 : num_balcony = num_orchestra + 190)
  (h3 : price_orchestra = 12)
  (h4 : price_balcony = 8)
  (h5 : total_tickets = 370)
  : total_revenue = 3320 := by
  sorry

end total_ticket_sales_cost_l167_167535


namespace product_of_two_numbers_l167_167116

theorem product_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 :=
sorry

end product_of_two_numbers_l167_167116


namespace find_shortage_l167_167014

def total_capacity (T : ℝ) : Prop :=
  0.70 * T = 14

def normal_level (normal : ℝ) : Prop :=
  normal = 14 / 2

def capacity_shortage (T : ℝ) (normal : ℝ) : Prop :=
  T - normal = 13

theorem find_shortage (T : ℝ) (normal : ℝ) : 
  total_capacity T →
  normal_level normal →
  capacity_shortage T normal :=
by
  sorry

end find_shortage_l167_167014


namespace digit_in_thousandths_place_l167_167009

theorem digit_in_thousandths_place : (3 / 16 : ℚ) = 0.1875 :=
by sorry

end digit_in_thousandths_place_l167_167009


namespace gym_class_total_students_l167_167043

theorem gym_class_total_students (group1_members group2_members : ℕ) 
  (h1 : group1_members = 34) (h2 : group2_members = 37) :
  group1_members + group2_members = 71 :=
by
  sorry

end gym_class_total_students_l167_167043


namespace remainder_ab_cd_l167_167276

theorem remainder_ab_cd (n : ℕ) (hn: n > 0) (a b c d : ℤ) 
  (hac : a * c ≡ 1 [ZMOD n]) (hbd : b * d ≡ 1 [ZMOD n]) : 
  (a * b + c * d) % n = 2 :=
by
  sorry

end remainder_ab_cd_l167_167276


namespace buy_beams_l167_167376

theorem buy_beams (C T x : ℕ) (hC : C = 6210) (hT : T = 3) (hx: x > 0):
  T * (x - 1) = C / x :=
by
  rw [hC, hT]
  sorry

end buy_beams_l167_167376


namespace cos_5_theta_l167_167299

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l167_167299


namespace train_length_l167_167393

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

end train_length_l167_167393


namespace six_digit_numbers_with_zero_count_l167_167712

def count_six_digit_numbers_with_at_least_one_zero : ℕ :=
  let total_numbers := 9 * 10^5
  let numbers_without_zero := 9^6
  total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero_count :
  count_six_digit_numbers_with_at_least_one_zero = 368559 := 
  by 
    sorry

end six_digit_numbers_with_zero_count_l167_167712


namespace depth_of_first_hole_l167_167028

theorem depth_of_first_hole :
  (45 * 8 * (80 * 6 * 40) / (45 * 8) : ℝ) = 53.33 := by
  -- This is where you would provide the proof, but it will be skipped with 'sorry'
  sorry

end depth_of_first_hole_l167_167028


namespace area_inside_S_outside_R_l167_167831

theorem area_inside_S_outside_R (area_R area_S : ℝ) (h1: area_R = 1 + 3 * Real.sqrt 3) (h2: area_S = 6 * Real.sqrt 3) :
  area_S - area_R = 1 :=
by {
   sorry
}

end area_inside_S_outside_R_l167_167831


namespace p_sq_plus_q_sq_l167_167719

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 :=
by
  sorry

end p_sq_plus_q_sq_l167_167719


namespace triangle_base_l167_167167

theorem triangle_base (A h b : ℝ) (hA : A = 15) (hh : h = 6) (hbase : A = 0.5 * b * h) : b = 5 := by
  sorry

end triangle_base_l167_167167


namespace betty_needs_more_flies_l167_167110

-- Definitions for the number of flies consumed by the frog each day
def fliesMonday : ℕ := 3
def fliesTuesday : ℕ := 2
def fliesWednesday : ℕ := 4
def fliesThursday : ℕ := 5
def fliesFriday : ℕ := 1
def fliesSaturday : ℕ := 2
def fliesSunday : ℕ := 3

-- Definition for the total number of flies eaten by the frog in a week
def totalFliesEaten : ℕ :=
  fliesMonday + fliesTuesday + fliesWednesday + fliesThursday + fliesFriday + fliesSaturday + fliesSunday

-- Definitions for the number of flies caught by Betty
def fliesMorning : ℕ := 5
def fliesAfternoon : ℕ := 6
def fliesEscaped : ℕ := 1

-- Definition for the total number of flies caught by Betty considering the escape
def totalFliesCaught : ℕ := fliesMorning + fliesAfternoon - fliesEscaped

-- Lean 4 statement to prove the number of additional flies Betty needs to catch
theorem betty_needs_more_flies : 
  totalFliesEaten - totalFliesCaught = 10 := 
by
  sorry

end betty_needs_more_flies_l167_167110


namespace arithmetic_progression_y_value_l167_167996

theorem arithmetic_progression_y_value (x y : ℚ) 
  (h1 : x = 2)
  (h2 : 2 * y - x = (y + x + 3) - (2 * y - x))
  (h3 : (3 * y + x) - (y + x + 3) = (y + x + 3) - (2 * y - x)) : 
  y = 10 / 3 :=
by
  sorry

end arithmetic_progression_y_value_l167_167996


namespace Democrats_in_House_l167_167860

-- Let D be the number of Democrats.
-- Let R be the number of Republicans.
-- Given conditions.

def Democrats (D R : ℕ) : Prop := 
  D + R = 434 ∧ R = D + 30

theorem Democrats_in_House : ∃ D, ∃ R, Democrats D R ∧ D = 202 :=
by
  -- skip the proof
  sorry

end Democrats_in_House_l167_167860


namespace y_intercept_of_line_l167_167683

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  -- The proof steps will go here.
  sorry

end y_intercept_of_line_l167_167683


namespace find_side_length_of_cut_out_square_l167_167294

noncomputable def cardboard_box (x : ℝ) : Prop :=
  let length_initial := 80
  let width_initial := 60
  let area_base := 1500
  let length_final := length_initial - 2 * x
  let width_final := width_initial - 2 * x
  length_final * width_final = area_base

theorem find_side_length_of_cut_out_square : ∃ x : ℝ, cardboard_box x ∧ 0 ≤ x ∧ (80 - 2 * x) > 0 ∧ (60 - 2 * x) > 0 ∧ x = 15 :=
by
  sorry

end find_side_length_of_cut_out_square_l167_167294


namespace simplify_expression_l167_167972

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (3 * x^2 * x^3) = 29 * x^5 := 
  sorry

end simplify_expression_l167_167972


namespace determine_F_value_l167_167669

theorem determine_F_value (D E F : ℕ) (h1 : (9 + 6 + D + 1 + E + 8 + 2) % 3 = 0) (h2 : (5 + 4 + E + D + 2 + 1 + F) % 3 = 0) : 
  F = 2 := 
by
  sorry

end determine_F_value_l167_167669


namespace max_value_char_l167_167728

theorem max_value_char (m x a b : ℕ) (h_sum : 28 * m + x + a + 2 * b = 368)
  (h1 : x ≤ 23) (h2 : x > a) (h3 : a > b) (h4 : b ≥ 0) :
  m + x ≤ 35 := 
sorry

end max_value_char_l167_167728


namespace slice_of_bread_area_l167_167049

theorem slice_of_bread_area (total_area : ℝ) (number_of_parts : ℕ) (h1 : total_area = 59.6) (h2 : number_of_parts = 4) : 
  total_area / number_of_parts = 14.9 :=
by
  rw [h1, h2]
  norm_num


end slice_of_bread_area_l167_167049


namespace simple_interest_calculation_l167_167608

-- Defining the given values
def principal : ℕ := 1500
def rate : ℕ := 7
def time : ℕ := rate -- time is the same as the rate of interest

-- Define the simple interest calculation
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Proof statement
theorem simple_interest_calculation : simple_interest principal rate time = 735 := by
  sorry

end simple_interest_calculation_l167_167608


namespace subcommittee_has_teacher_l167_167231

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k

def teacher_subcommittee_count : ℕ := total_combinations 12 5 - total_combinations 7 5

theorem subcommittee_has_teacher : teacher_subcommittee_count = 771 := 
by
  sorry

end subcommittee_has_teacher_l167_167231


namespace parallel_vectors_l167_167512

theorem parallel_vectors (m : ℝ) : (m = 1) ↔ (∃ k : ℝ, (m, 1) = k • (1, m)) := sorry

end parallel_vectors_l167_167512


namespace retirement_hire_year_l167_167371

theorem retirement_hire_year (A : ℕ) (R : ℕ) (Y : ℕ) (W : ℕ) 
  (h1 : A + W = 70) 
  (h2 : A = 32) 
  (h3 : R = 2008) 
  (h4 : W = R - Y) : Y = 1970 :=
by
  sorry

end retirement_hire_year_l167_167371


namespace negation_propositional_logic_l167_167840

theorem negation_propositional_logic :
  ¬ (∀ x : ℝ, x^2 + x + 1 < 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≥ 0 :=
by sorry

end negation_propositional_logic_l167_167840


namespace xiao_dong_actual_jump_distance_l167_167780

-- Conditions are defined here
def standard_jump_distance : ℝ := 4.00
def xiao_dong_recorded_result : ℝ := -0.32

-- Here we structure our problem
theorem xiao_dong_actual_jump_distance :
  standard_jump_distance + xiao_dong_recorded_result = 3.68 :=
by
  sorry

end xiao_dong_actual_jump_distance_l167_167780


namespace leak_emptying_time_l167_167708

theorem leak_emptying_time (A_rate L_rate : ℚ) 
  (hA : A_rate = 1 / 4)
  (hCombined : A_rate - L_rate = 1 / 8) :
  1 / L_rate = 8 := 
by
  sorry

end leak_emptying_time_l167_167708


namespace proj_onto_w_equals_correct_l167_167645

open Real

noncomputable def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let scalar_mul c (a : ℝ × ℝ) := (c * a.1, c * a.2)
  let w_dot_w := dot w w
  if w_dot_w = 0 then (0, 0) else scalar_mul (dot v w / w_dot_w) w

theorem proj_onto_w_equals_correct (v w : ℝ × ℝ)
  (hv : v = (2, 3))
  (hw : w = (-4, 1)) :
  proj w v = (20 / 17, -5 / 17) :=
by
  -- The proof would go here. We add sorry to skip it.
  sorry

end proj_onto_w_equals_correct_l167_167645


namespace tennis_player_games_l167_167722

theorem tennis_player_games (b : ℕ → ℕ) (h1 : ∀ k, b k ≥ k) (h2 : ∀ k, b k ≤ 12 * (k / 7)) :
  ∃ i j : ℕ, i < j ∧ b j - b i = 20 :=
by
  sorry

end tennis_player_games_l167_167722


namespace fraction_subtraction_proof_l167_167141

theorem fraction_subtraction_proof : 
  (21 / 12) - (18 / 15) = 11 / 20 := 
by 
  sorry

end fraction_subtraction_proof_l167_167141


namespace simplify_trig_expression_l167_167317

open Real

theorem simplify_trig_expression (A : ℝ) (h1 : cos A ≠ 0) (h2 : sin A ≠ 0) :
  (1 - (cos A) / (sin A) + 1 / (sin A)) * (1 + (sin A) / (cos A) - 1 / (cos A)) = -2 * (cos (2 * A) / sin (2 * A)) :=
by
  sorry

end simplify_trig_expression_l167_167317


namespace cost_per_book_l167_167932

-- Definitions and conditions
def number_of_books : ℕ := 8
def amount_tommy_has : ℕ := 13
def amount_tommy_needs_to_save : ℕ := 27

-- Total money Tommy needs to buy the books
def total_amount_needed : ℕ := amount_tommy_has + amount_tommy_needs_to_save

-- Proven statement
theorem cost_per_book : (total_amount_needed / number_of_books) = 5 := by
  -- Skip proof
  sorry

end cost_per_book_l167_167932


namespace find_x_l167_167165

/-- Given real numbers x and y,
    under the condition that (y^3 + 2y - 1)/(y^3 + 2y - 3) = x/(x - 1),
    we want to prove that x = (y^3 + 2y - 1)/2 -/
theorem find_x (x y : ℝ) (h1 : y^3 + 2*y - 3 ≠ 0) (h2 : y^3 + 2*y - 1 ≠ 0)
  (h : x / (x - 1) = (y^3 + 2*y - 1) / (y^3 + 2*y - 3)) :
  x = (y^3 + 2*y - 1) / 2 :=
by sorry

end find_x_l167_167165


namespace left_person_truthful_right_person_lies_l167_167143

theorem left_person_truthful_right_person_lies
  (L R M : Prop)
  (L_truthful_or_false : L ∨ ¬L)
  (R_truthful_or_false : R ∨ ¬R)
  (M_always_answers : M = (L → M) ∨ (¬L → M))
  (left_statement : L → (M = (L → M)))
  (right_statement : R → (M = (¬L → M))) :
  (L ∧ ¬R) ∨ (¬L ∧ R) :=
by
  sorry

end left_person_truthful_right_person_lies_l167_167143


namespace geometric_sequence_common_ratio_l167_167730

/--
  Given a geometric sequence with the first three terms:
  a₁ = 27,
  a₂ = 54,
  a₃ = 108,
  prove that the common ratio is r = 2.
-/
theorem geometric_sequence_common_ratio :
  let a₁ := 27
  let a₂ := 54
  let a₃ := 108
  ∃ r : ℕ, (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ r = 2 := by
  sorry

end geometric_sequence_common_ratio_l167_167730


namespace directrix_of_parabola_l167_167201

theorem directrix_of_parabola : 
  let y := 3 * x^2 - 6 * x + 1
  y = -25 / 12 :=
sorry

end directrix_of_parabola_l167_167201


namespace calculate_retail_price_l167_167597

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end calculate_retail_price_l167_167597


namespace number_of_factors_l167_167082

theorem number_of_factors (b n : ℕ) (hb1 : b = 6) (hn1 : n = 15) (hb2 : b > 0) (hb3 : b ≤ 15) (hn2 : n > 0) (hn3 : n ≤ 15) :
  let factors := (15 + 1) * (15 + 1)
  factors = 256 :=
by
  sorry

end number_of_factors_l167_167082


namespace correct_divisor_l167_167079

theorem correct_divisor :
  ∀ (D : ℕ), (D = 12 * 63) → (D = x * 36) → (x = 21) := 
by 
  intros D h1 h2
  sorry

end correct_divisor_l167_167079


namespace solve_equation_l167_167493

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^3 + (x - 6)^3 = 54 ∧ x = 7 := by
sorry

end solve_equation_l167_167493


namespace sum_of_x_y_l167_167768

theorem sum_of_x_y :
  ∀ (x y : ℚ), (1 / x + 1 / y = 4) → (1 / x - 1 / y = -8) → x + y = -1 / 3 := 
by
  intros x y h1 h2
  sorry

end sum_of_x_y_l167_167768


namespace team_total_score_is_correct_l167_167204

-- Define the total number of team members
def total_members : ℕ := 30

-- Define the number of members who didn't show up
def members_absent : ℕ := 8

-- Define the score per member
def score_per_member : ℕ := 4

-- Define the points deducted per incorrect answer
def points_per_incorrect_answer : ℕ := 2

-- Define the total number of incorrect answers
def total_incorrect_answers : ℕ := 6

-- Define the bonus multiplier
def bonus_multiplier : ℝ := 1.5

-- Define the total score calculation
def total_score_calculation (total_members : ℕ) (members_absent : ℕ) (score_per_member : ℕ)
  (points_per_incorrect_answer : ℕ) (total_incorrect_answers : ℕ) (bonus_multiplier : ℝ) : ℝ :=
  let members_present := total_members - members_absent
  let initial_score := members_present * score_per_member
  let total_deductions := total_incorrect_answers * points_per_incorrect_answer
  let final_score := initial_score - total_deductions
  final_score * bonus_multiplier

-- Prove that the total score is 114 points
theorem team_total_score_is_correct : total_score_calculation total_members members_absent score_per_member
  points_per_incorrect_answer total_incorrect_answers bonus_multiplier = 114 :=
by
  sorry

end team_total_score_is_correct_l167_167204


namespace common_rational_root_l167_167263

-- Definitions for the given conditions
def polynomial1 (a b c : ℤ) (x : ℚ) := 50 * x^4 + a * x^3 + b * x^2 + c * x + 16 = 0
def polynomial2 (d e f g : ℤ) (x : ℚ) := 16 * x^5 + d * x^4 + e * x^3 + f * x^2 + g * x + 50 = 0

-- The proof problem statement: Given the conditions, proving that -1/2 is a common rational root
theorem common_rational_root (a b c d e f g : ℤ) (k : ℚ) 
  (h1 : polynomial1 a b c k)
  (h2 : polynomial2 d e f g k) 
  (h3 : ∃ m n : ℤ, k = -((m : ℚ) / n) ∧ Int.gcd m n = 1) :
  k = -1/2 :=
sorry

end common_rational_root_l167_167263


namespace existence_of_unique_root_l167_167464

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 5

theorem existence_of_unique_root :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  f 0 = -4 ∧
  f 2 = Real.exp 2 - 1 →
  ∃! c, f c = 0 :=
by
  sorry

end existence_of_unique_root_l167_167464


namespace integer_triplets_satisfy_eq_l167_167662

theorem integer_triplets_satisfy_eq {x y z : ℤ} : 
  x^2 + y^2 + z^2 - x * y - y * z - z * x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) := 
by
  sorry

end integer_triplets_satisfy_eq_l167_167662


namespace age_ratio_l167_167713

theorem age_ratio (R D : ℕ) (h1 : R + 2 = 26) (h2 : D = 18) : R / D = 4 / 3 :=
sorry

end age_ratio_l167_167713


namespace hyperbola_asymptote_l167_167672

theorem hyperbola_asymptote (y x : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) → (y = x * 3 / 4 ∨ y = -x * 3 / 4) :=
sorry

end hyperbola_asymptote_l167_167672


namespace find_tan_theta_l167_167947

theorem find_tan_theta
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioc 0 (Real.pi / 4))
  (h2 : Real.sin θ + Real.cos θ = 17 / 13) :
  Real.tan θ = 5 / 12 :=
sorry

end find_tan_theta_l167_167947


namespace find_value_of_expression_l167_167374

theorem find_value_of_expression (a b c d : ℤ) (h₁ : a = -1) (h₂ : b + c = 0) (h₃ : abs d = 2) :
  4 * a + (b + c) - abs (3 * d) = -10 := by
  sorry

end find_value_of_expression_l167_167374


namespace primes_in_sequence_are_12_l167_167575

-- Definition of Q
def Q : Nat := (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47)

-- Set of m values
def ms : List Nat := List.range' 3 101

-- Function to check if Q + m is prime
def is_prime_minus_Q (m : Nat) : Bool := Nat.Prime (Q + m)

-- Counting primes in the sequence
def count_primes_in_sequence : Nat := (ms.filter (λ m => is_prime_minus_Q m = true)).length

theorem primes_in_sequence_are_12 :
  count_primes_in_sequence = 12 := by 
  sorry

end primes_in_sequence_are_12_l167_167575


namespace problem1_l167_167369

theorem problem1 (α : Real) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 := 
sorry

end problem1_l167_167369


namespace exactly_2_std_devs_less_than_mean_l167_167086

noncomputable def mean : ℝ := 14.5
noncomputable def std_dev : ℝ := 1.5
noncomputable def value : ℝ := mean - 2 * std_dev

theorem exactly_2_std_devs_less_than_mean : value = 11.5 := by
  sorry

end exactly_2_std_devs_less_than_mean_l167_167086


namespace division_of_decimals_l167_167848

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals_l167_167848


namespace tegwen_family_total_children_l167_167061

variable (Tegwen : Type)

-- Variables representing the number of girls and boys
variable (g b : ℕ)

-- Conditions from the problem
variable (h1 : b = g - 1)
variable (h2 : g = (3/2:ℚ) * (b - 1))

-- Proposition that the total number of children is 11
theorem tegwen_family_total_children : g + b = 11 := by
  sorry

end tegwen_family_total_children_l167_167061


namespace question1_question2_l167_167564

section

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition 1: A = {x | -1 ≤ x < 3}
def setA : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Condition 2: B = {x | 2x - 4 ≥ x - 2}
def setB : Set ℝ := {x | x ≥ 2}

-- Condition 3: C = {x | x ≥ a - 1}
def setC (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Question 1: Prove A ∩ B = {x | 2 ≤ x < 3}
theorem question1 : A = setA → B = setB → A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by intros hA hB; rw [hA, hB]; sorry

-- Question 2: If B ∪ C = C, prove a ∈ (-∞, 3]
theorem question2 : B = setB → C = setC a → (B ∪ C = C) → a ≤ 3 :=
by intros hB hC hBUC; rw [hB, hC] at hBUC; sorry

end

end question1_question2_l167_167564


namespace adult_ticket_cost_l167_167797

variable (A : ℝ)

theorem adult_ticket_cost :
  (20 * 6) + (12 * A) = 216 → A = 8 :=
by
  intro h
  sorry

end adult_ticket_cost_l167_167797


namespace line_passes_vertex_parabola_l167_167826

theorem line_passes_vertex_parabola :
  ∃ (b₁ b₂ : ℚ), (b₁ ≠ b₂) ∧ (∀ b, (b = b₁ ∨ b = b₂) → 
    (∃ x y, y = x + b ∧ y = x^2 + 4 * b^2 ∧ x = 0 ∧ y = 4 * b^2)) :=
by 
  sorry

end line_passes_vertex_parabola_l167_167826


namespace equivalent_statements_l167_167187

variable (P Q : Prop)

theorem equivalent_statements : 
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by 
  sorry

end equivalent_statements_l167_167187


namespace nonagon_side_length_l167_167739

theorem nonagon_side_length (perimeter : ℝ) (n : ℕ) (h_reg_nonagon : n = 9) (h_perimeter : perimeter = 171) :
  perimeter / n = 19 := by
  sorry

end nonagon_side_length_l167_167739


namespace max_expr_value_l167_167558

noncomputable def expr (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_expr_value : 
  ∃ (a b c d : ℝ),
    a ∈ Set.Icc (-5 : ℝ) 5 ∧
    b ∈ Set.Icc (-5 : ℝ) 5 ∧
    c ∈ Set.Icc (-5 : ℝ) 5 ∧
    d ∈ Set.Icc (-5 : ℝ) 5 ∧
    expr a b c d = 110 :=
by
  -- Proof omitted
  sorry

end max_expr_value_l167_167558


namespace susan_books_l167_167902

theorem susan_books (S : ℕ) (h1 : S + 4 * S = 3000) : S = 600 :=
by 
  sorry

end susan_books_l167_167902


namespace average_of_data_set_is_five_l167_167072

def data_set : List ℕ := [2, 5, 5, 6, 7]

def sum_of_data_set : ℕ := data_set.sum
def count_of_data_set : ℕ := data_set.length

theorem average_of_data_set_is_five :
  (sum_of_data_set / count_of_data_set) = 5 :=
by
  sorry

end average_of_data_set_is_five_l167_167072


namespace factor_difference_of_squares_l167_167092

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l167_167092


namespace difference_of_numbers_l167_167439

theorem difference_of_numbers 
  (L S : ℤ) (hL : L = 1636) (hdiv : L = 6 * S + 10) : 
  L - S = 1365 :=
sorry

end difference_of_numbers_l167_167439


namespace rhombus_diagonals_not_equal_l167_167516

-- Define what a rhombus is
structure Rhombus where
  sides_equal : ∀ a b : ℝ, a = b  -- all sides are equal
  symmetrical : Prop -- it is a symmetrical figure
  centrally_symmetrical : Prop -- it is a centrally symmetrical figure

-- Theorem to state that the diagonals of a rhombus are not necessarily equal
theorem rhombus_diagonals_not_equal (r : Rhombus) : ¬(∀ a b : ℝ, a = b) := by
  sorry

end rhombus_diagonals_not_equal_l167_167516


namespace find_v2_poly_l167_167051

theorem find_v2_poly (x : ℤ) (v0 v1 v2 : ℤ) 
  (h1 : x = -4)
  (h2 : v0 = 1) 
  (h3 : v1 = v0 * x)
  (h4 : v2 = v1 * x + 6) :
  v2 = 22 :=
by
  -- To be filled with proof (example problem requirement specifies proof is not needed)
  sorry

end find_v2_poly_l167_167051


namespace jessica_balloons_l167_167286

-- Defining the number of blue balloons Joan, Sally, and the total number.
def balloons_joan : ℕ := 9
def balloons_sally : ℕ := 5
def balloons_total : ℕ := 16

-- The statement to prove that Jessica has 2 blue balloons
theorem jessica_balloons : balloons_total - (balloons_joan + balloons_sally) = 2 :=
by
  -- Using the given information and arithmetic, we can show the main statement
  sorry

end jessica_balloons_l167_167286


namespace magazine_page_height_l167_167210

theorem magazine_page_height
  (charge_per_sq_inch : ℝ := 8)
  (half_page_cost : ℝ := 432)
  (page_width : ℝ := 12) : 
  ∃ h : ℝ, (1/2) * h * page_width * charge_per_sq_inch = half_page_cost :=
by sorry

end magazine_page_height_l167_167210


namespace sample_std_dev_range_same_l167_167837

noncomputable def sample_std_dev (data : List ℝ) : ℝ := sorry
noncomputable def sample_range (data : List ℝ) : ℝ := sorry

theorem sample_std_dev_range_same (n : ℕ) (c : ℝ) (Hc : c ≠ 0) (x : Fin n → ℝ) :
  sample_std_dev (List.ofFn (λ i => x i)) = sample_std_dev (List.ofFn (λ i => x i + c)) ∧
  sample_range (List.ofFn (λ i => x i)) = sample_range (List.ofFn (λ i => x i + c)) :=
by
  sorry

end sample_std_dev_range_same_l167_167837


namespace initial_men_garrison_l167_167350

-- Conditions:
-- A garrison has provisions for 31 days.
-- At the end of 16 days, a reinforcement of 300 men arrives.
-- The provisions last only for 5 days more after the reinforcement arrives.

theorem initial_men_garrison (M : ℕ) (P : ℕ) (d1 d2 : ℕ) (r : ℕ) (remaining1 remaining2 : ℕ) :
  P = M * d1 →
  remaining1 = P - M * d2 →
  remaining2 = r * (d1 - d2) →
  remaining1 = remaining2 →
  r = M + 300 →
  d1 = 31 →
  d2 = 16 →
  M = 150 :=
by 
  sorry

end initial_men_garrison_l167_167350


namespace pure_water_to_achieve_desired_concentration_l167_167786

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l167_167786


namespace find_k_l167_167529

theorem find_k : 
  ∀ (k : ℤ), 2^4 - 6 = 3^3 + k ↔ k = -17 :=
by sorry

end find_k_l167_167529


namespace geometric_sequence_common_ratio_l167_167433

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 2 = a 1 * q)
    (h2 : a 5 = a 1 * q ^ 4)
    (h3 : a 2 = 8)
    (h4 : a 5 = 64) :
    q = 2 := 
sorry

end geometric_sequence_common_ratio_l167_167433


namespace binomial_510_510_l167_167065

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end binomial_510_510_l167_167065


namespace simplify_trig_l167_167420

theorem simplify_trig (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) / 
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) = 
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by 
  sorry

end simplify_trig_l167_167420


namespace evaluate_expression_l167_167522

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l167_167522


namespace eval_expression_l167_167409

theorem eval_expression : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := sorry

end eval_expression_l167_167409


namespace books_read_in_8_hours_l167_167975

def reading_speed := 100 -- pages per hour
def book_pages := 400 -- pages per book
def hours_available := 8 -- hours

theorem books_read_in_8_hours :
  (hours_available * reading_speed) / book_pages = 2 :=
by
  sorry

end books_read_in_8_hours_l167_167975


namespace solve_pairs_l167_167242

theorem solve_pairs (m n : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (m, n) = (6, 3) ∨ (m, n) = (9, 3) ∨ (m, n) = (9, 5) ∨ (m, n) = (54, 5) :=
by
  sorry

end solve_pairs_l167_167242


namespace minimum_value_of_expression_l167_167052

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2 / 5) ^ (1 / 5) := by
sorry

end minimum_value_of_expression_l167_167052


namespace original_acid_percentage_l167_167915

variables (a w : ℝ)

-- Conditions from the problem
def cond1 : Prop := a / (a + w + 2) = 0.18
def cond2 : Prop := (a + 2) / (a + w + 4) = 0.36

-- The Lean statement to prove
theorem original_acid_percentage (hc1 : cond1 a w) (hc2 : cond2 a w) : (a / (a + w)) * 100 = 19 :=
sorry

end original_acid_percentage_l167_167915


namespace remainder_5_pow_2048_mod_17_l167_167974

theorem remainder_5_pow_2048_mod_17 : (5 ^ 2048) % 17 = 0 :=
by
  sorry

end remainder_5_pow_2048_mod_17_l167_167974


namespace delores_initial_money_l167_167513

def computer_price : ℕ := 400
def printer_price : ℕ := 40
def headphones_price : ℕ := 60
def discount_percentage : ℕ := 10
def left_money : ℕ := 10

theorem delores_initial_money :
  ∃ initial_money : ℕ,
    initial_money = printer_price + headphones_price + (computer_price - (discount_percentage * computer_price / 100)) + left_money :=
  sorry

end delores_initial_money_l167_167513


namespace find_y_l167_167823

theorem find_y (x y : ℝ) (h : x = 180) (h1 : 0.25 * x = 0.10 * y - 5) : y = 500 :=
by sorry

end find_y_l167_167823


namespace brother_age_in_5_years_l167_167908

theorem brother_age_in_5_years
  (nick_age : ℕ)
  (sister_age : ℕ)
  (brother_age : ℕ)
  (h_nick : nick_age = 13)
  (h_sister : sister_age = nick_age + 6)
  (h_brother : brother_age = (nick_age + sister_age) / 2) :
  brother_age + 5 = 21 := 
by 
  sorry

end brother_age_in_5_years_l167_167908


namespace sequence_formula_l167_167076

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 2) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n ^ 2 - n * a n + 1) :
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

end sequence_formula_l167_167076


namespace max_dot_product_between_ellipses_l167_167896

noncomputable def ellipse1 (x y : ℝ) : Prop := (x^2 / 25 + y^2 / 9 = 1)
noncomputable def ellipse2 (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 9 = 1)

theorem max_dot_product_between_ellipses :
  ∀ (M N : ℝ × ℝ),
    ellipse1 M.1 M.2 →
    ellipse2 N.1 N.2 →
    ∃ θ φ : ℝ,
      M = (5 * Real.cos θ, 3 * Real.sin θ) ∧
      N = (3 * Real.cos φ, 3 * Real.sin φ) ∧
      (15 * Real.cos θ * Real.cos φ + 9 * Real.sin θ * Real.sin φ ≤ 15) :=
by
  sorry

end max_dot_product_between_ellipses_l167_167896


namespace solve_for_x_l167_167910

theorem solve_for_x (x : ℝ) (h : 4 * x - 5 = 3) : x = 2 :=
by sorry

end solve_for_x_l167_167910


namespace unique_prime_value_l167_167625

theorem unique_prime_value :
  ∃! n : ℕ, n > 0 ∧ Nat.Prime (n^3 - 7 * n^2 + 17 * n - 11) :=
by {
  sorry
}

end unique_prime_value_l167_167625


namespace new_determinant_l167_167357

-- Given the condition that the determinant of the original matrix is 12
def original_determinant (x y z w : ℝ) : Prop :=
  x * w - y * z = 12

-- Proof that the determinant of the new matrix equals the expected result
theorem new_determinant (x y z w : ℝ) (h : original_determinant x y z w) :
  (2 * x + z) * w - (2 * y - w) * z = 24 + z * w + w * z := by
  sorry

end new_determinant_l167_167357


namespace find_value_l167_167909

variable (a b : ℝ)

def quadratic_equation_roots : Prop :=
  a^2 - 4 * a - 1 = 0 ∧ b^2 - 4 * b - 1 = 0

def sum_of_roots : Prop :=
  a + b = 4

def product_of_roots : Prop :=
  a * b = -1

theorem find_value (ha : quadratic_equation_roots a b) (hs : sum_of_roots a b) (hp : product_of_roots a b) :
  2 * a^2 + 3 / b + 5 * b = 22 :=
sorry

end find_value_l167_167909


namespace tan_alpha_values_l167_167142

theorem tan_alpha_values (α : ℝ) (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := 
by sorry

end tan_alpha_values_l167_167142


namespace sequence_formula_l167_167130

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) - 2 * a n + 3 = 0) :
  ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end sequence_formula_l167_167130


namespace find_number_of_boys_l167_167514

noncomputable def number_of_boys (B G : ℕ) : Prop :=
  (B : ℚ) / (G : ℚ) = 7.5 / 15.4 ∧ G = B + 174

theorem find_number_of_boys : ∃ B G : ℕ, number_of_boys B G ∧ B = 165 := 
by 
  sorry

end find_number_of_boys_l167_167514


namespace solve_for_y_l167_167226

theorem solve_for_y (y : ℝ) : y^2 - 6 * y + 5 = 0 ↔ y = 1 ∨ y = 5 :=
by
  sorry

end solve_for_y_l167_167226


namespace average_length_correct_l167_167613

-- Given lengths of the two pieces
def length1 : ℕ := 2
def length2 : ℕ := 6

-- Define the average length
def average_length (l1 l2 : ℕ) : ℕ := (l1 + l2) / 2

-- State the theorem to prove
theorem average_length_correct : average_length length1 length2 = 4 := 
by 
  sorry

end average_length_correct_l167_167613


namespace problem_1_problem_2_l167_167097

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 2 * y + 5 = 0
def point_M : ℝ × ℝ := (1, 2)
def point_P : ℝ × ℝ := (3, 1)

def line_l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

theorem problem_1 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) → 
  (line_l1 point_P.1 point_P.2) ∧ (line_l1 point_M.1 point_M.2) :=
by 
  sorry

theorem problem_2 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) →
  (∀ (x y : ℝ), line_l2 x y ↔ line3 x y) :=
by
  sorry

end problem_1_problem_2_l167_167097


namespace polynomial_simplification_l167_167102

theorem polynomial_simplification (p : ℝ) :
  (4 * p^4 + 2 * p^3 - 7 * p + 3) + (5 * p^3 - 8 * p^2 + 3 * p + 2) = 
  4 * p^4 + 7 * p^3 - 8 * p^2 - 4 * p + 5 :=
by
  sorry

end polynomial_simplification_l167_167102


namespace cards_drawn_to_product_even_l167_167721

theorem cards_drawn_to_product_even :
  ∃ n, (∀ (cards_drawn : Finset ℕ), 
    (cards_drawn ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) ∧
    (cards_drawn.card = n) → 
    ¬ (∀ c ∈ cards_drawn, c % 2 = 1)
  ) ∧ n = 8 :=
by
  sorry

end cards_drawn_to_product_even_l167_167721


namespace triangular_weight_is_60_l167_167368

/-- Suppose there are weights: 5 identical round, 2 identical triangular, and 1 rectangular weight of 90 grams.
    The conditions are: 
    1. One round weight and one triangular weight balance three round weights.
    2. Four round weights and one triangular weight balance one triangular weight, one round weight, and one rectangular weight.
    Prove that the weight of the triangular weight is 60 grams. -/
theorem triangular_weight_is_60 
  (R T : ℕ)  -- We declare weights of round and triangular weights as natural numbers
  (h1 : R + T = 3 * R)  -- The first balance condition
  (h2 : 4 * R + T = T + R + 90)  -- The second balance condition
  : T = 60 := 
by
  sorry  -- Proof omitted

end triangular_weight_is_60_l167_167368


namespace tiffany_max_points_l167_167002

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l167_167002


namespace total_oranges_proof_l167_167531

def jeremyMonday : ℕ := 100
def jeremyTuesdayPlusBrother : ℕ := 3 * jeremyMonday
def jeremyWednesdayPlusBrotherPlusCousin : ℕ := 2 * jeremyTuesdayPlusBrother
def jeremyThursday : ℕ := (70 * jeremyMonday) / 100
def cousinWednesday : ℕ := jeremyTuesdayPlusBrother - (20 * jeremyTuesdayPlusBrother) / 100
def cousinThursday : ℕ := cousinWednesday + (30 * cousinWednesday) / 100

def total_oranges : ℕ :=
  jeremyMonday + jeremyTuesdayPlusBrother + jeremyWednesdayPlusBrotherPlusCousin + (jeremyThursday + (jeremyWednesdayPlusBrotherPlusCousin - cousinWednesday) + cousinThursday)

theorem total_oranges_proof : total_oranges = 1642 :=
by
  sorry

end total_oranges_proof_l167_167531


namespace perfect_square_iff_divisibility_l167_167846

theorem perfect_square_iff_divisibility (A : ℕ) :
  (∃ d : ℕ, A = d^2) ↔ ∀ n : ℕ, n > 0 → ∃ j : ℕ, 1 ≤ j ∧ j ≤ n ∧ n ∣ (A + j)^2 - A :=
sorry

end perfect_square_iff_divisibility_l167_167846


namespace points_after_perfect_games_l167_167626

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l167_167626


namespace range_of_a_for_f_increasing_l167_167189

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a_for_f_increasing :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_for_f_increasing_l167_167189


namespace find_x_eq_neg15_l167_167264

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l167_167264


namespace simplify_and_evaluate_l167_167776

/-- 
Given the expression (1 + 1 / (x - 2)) ÷ ((x ^ 2 - 2 * x + 1) / (x - 2)), 
prove that it evaluates to -1 when x = 0.
-/
theorem simplify_and_evaluate (x : ℝ) (h : x = 0) :
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x - 2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l167_167776


namespace range_of_x_when_y_lt_0_l167_167832

variable (a b c n m : ℝ)

-- The definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions given in the problem
axiom value_at_neg1 : quadratic_function a b c (-1) = 4
axiom value_at_0 : quadratic_function a b c 0 = 0
axiom value_at_1 : quadratic_function a b c 1 = n
axiom value_at_2 : quadratic_function a b c 2 = m
axiom value_at_3 : quadratic_function a b c 3 = 4

-- Proof statement
theorem range_of_x_when_y_lt_0 : ∀ (x : ℝ), quadratic_function a b c x < 0 ↔ 0 < x ∧ x < 2 :=
sorry

end range_of_x_when_y_lt_0_l167_167832


namespace fewer_VIP_tickets_sold_l167_167039

variable (V G : ℕ)

-- Definitions: total number of tickets sold and the total revenue from tickets sold
def total_tickets : Prop := V + G = 320
def total_revenue : Prop := 45 * V + 20 * G = 7500

-- Definition of the number of fewer VIP tickets than general admission tickets
def fewer_VIP_tickets : Prop := G - V = 232

-- The theorem to be proven
theorem fewer_VIP_tickets_sold (h1 : total_tickets V G) (h2 : total_revenue V G) : fewer_VIP_tickets V G :=
sorry

end fewer_VIP_tickets_sold_l167_167039


namespace point_on_inverse_proportion_function_l167_167870

theorem point_on_inverse_proportion_function :
  ∀ (x y k : ℝ), k ≠ 0 ∧ y = k / x ∧ (2, -3) = (2, -(3 : ℝ)) → (x, y) = (-2, 3) → (y = -6 / x) :=
sorry

end point_on_inverse_proportion_function_l167_167870


namespace cows_sold_l167_167468

/-- 
A man initially had 39 cows, 25 of them died last year, he sold some remaining cows, this year,
the number of cows increased by 24, he bought 43 more cows, his friend gave him 8 cows.
Now, he has 83 cows. How many cows did he sell last year?
-/
theorem cows_sold (S : ℕ) : (39 - 25 - S + 24 + 43 + 8 = 83) → S = 6 :=
by
  intro h
  sorry

end cows_sold_l167_167468


namespace min_value_of_quadratic_l167_167119

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 9

theorem min_value_of_quadratic : ∃ (x : ℝ), f x = 6 :=
by sorry

end min_value_of_quadratic_l167_167119


namespace toby_photos_l167_167238

variable (p0 d c e x : ℕ)
def photos_remaining : ℕ := p0 - d + c + x - e

theorem toby_photos (h1 : p0 = 63) (h2 : d = 7) (h3 : c = 15) (h4 : e = 3) : photos_remaining p0 d c e x = 68 + x :=
by
  rw [h1, h2, h3, h4]
  sorry

end toby_photos_l167_167238


namespace basketball_points_total_l167_167615

variable (Tobee_points Jay_points Sean_points Remy_points Alex_points : ℕ)

def conditions := 
  Tobee_points = 4 ∧
  Jay_points = 2 * Tobee_points + 6 ∧
  Sean_points = Jay_points / 2 ∧
  Remy_points = Tobee_points + Jay_points - 3 ∧
  Alex_points = Sean_points + Remy_points + 4

theorem basketball_points_total 
  (h : conditions Tobee_points Jay_points Sean_points Remy_points Alex_points) :
  Tobee_points + Jay_points + Sean_points + Remy_points + Alex_points = 66 :=
by sorry

end basketball_points_total_l167_167615


namespace factorization_and_evaluation_l167_167172

noncomputable def polynomial_q1 (x : ℝ) : ℝ := x
noncomputable def polynomial_q2 (x : ℝ) : ℝ := x^2 - 2
noncomputable def polynomial_q3 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def polynomial_q4 (x : ℝ) : ℝ := x^2 + 1

theorem factorization_and_evaluation :
  polynomial_q1 3 + polynomial_q2 3 + polynomial_q3 3 + polynomial_q4 3 = 33 := by
  sorry

end factorization_and_evaluation_l167_167172


namespace f_of_x_l167_167604

theorem f_of_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x-1) = 3*x - 1) : ∀ x : ℤ, f x = 3*x + 2 :=
by
  sorry

end f_of_x_l167_167604


namespace min_value_expression_l167_167132

theorem min_value_expression (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) : (x * y + x^2) ≥ 4 :=
sorry

end min_value_expression_l167_167132


namespace totalCostOfCombinedSubscriptions_l167_167649

-- Define the given conditions
def packageACostPerMonth : ℝ := 10
def packageAMonths : ℝ := 6
def packageADiscount : ℝ := 0.10

def packageBCostPerMonth : ℝ := 12
def packageBMonths : ℝ := 9
def packageBDiscount : ℝ := 0.15

-- Define the total cost after discounts
def packageACostAfterDiscount : ℝ := packageACostPerMonth * packageAMonths * (1 - packageADiscount)
def packageBCostAfterDiscount : ℝ := packageBCostPerMonth * packageBMonths * (1 - packageBDiscount)

-- Statement to be proved
theorem totalCostOfCombinedSubscriptions :
  packageACostAfterDiscount + packageBCostAfterDiscount = 145.80 := by
  sorry

end totalCostOfCombinedSubscriptions_l167_167649


namespace find_p_q_l167_167533

theorem find_p_q (p q : ℤ) 
    (h1 : (3:ℤ)^5 - 2 * (3:ℤ)^4 + 3 * (3:ℤ)^3 - p * (3:ℤ)^2 + q * (3:ℤ) - 12 = 0)
    (h2 : (-1:ℤ)^5 - 2 * (-1:ℤ)^4 + 3 * (-1:ℤ)^3 - p * (-1:ℤ)^2 + q * (-1:ℤ) - 12 = 0) : 
    (p, q) = (-8, -10) :=
by
  sorry

end find_p_q_l167_167533


namespace eccentricity_of_ellipse_l167_167487

noncomputable def e (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a - c) * (a + c) = (2 * c)^2) : e a b c = (Real.sqrt 5) / 5 := 
by
  sorry

end eccentricity_of_ellipse_l167_167487


namespace tens_digit_of_seven_times_cubed_is_one_l167_167411

-- Variables and definitions
variables (p : ℕ) (h1 : p < 10)

-- Main theorem statement
theorem tens_digit_of_seven_times_cubed_is_one (hp : p < 10) :
  let N := 11 * p
  let m := 7
  let result := m * N^3
  (result / 10) % 10 = 1 := 
sorry

end tens_digit_of_seven_times_cubed_is_one_l167_167411


namespace find_h_l167_167782

noncomputable def h (x : ℝ) : ℝ := -x^4 - 2 * x^3 + 4 * x^2 + 9 * x - 5

def f (x : ℝ) : ℝ := x^4 + 2 * x^3 - x^2 - 4 * x + 1

def p (x : ℝ) : ℝ := 3 * x^2 + 5 * x - 4

theorem find_h (x : ℝ) : (f x) + (h x) = p x :=
by sorry

end find_h_l167_167782


namespace marius_scored_3_more_than_darius_l167_167648

theorem marius_scored_3_more_than_darius 
  (D M T : ℕ) 
  (h1 : D = 10) 
  (h2 : T = D + 5) 
  (h3 : M + D + T = 38) : 
  M = D + 3 := 
by
  sorry

end marius_scored_3_more_than_darius_l167_167648


namespace system_solution_l167_167183

theorem system_solution :
  ∃ x y : ℝ, (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧ 
            (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
            x = -3 / 4 ∧ y = 1 / 2 :=
by
  sorry

end system_solution_l167_167183


namespace double_increase_divide_l167_167211

theorem double_increase_divide (x : ℤ) (h : (2 * x + 7) / 5 = 17) : x = 39 := by
  sorry

end double_increase_divide_l167_167211


namespace xyz_sum_eq_48_l167_167346

theorem xyz_sum_eq_48 (x y z : ℕ) (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
  sorry

end xyz_sum_eq_48_l167_167346


namespace marbles_problem_l167_167880

theorem marbles_problem :
  let red_marbles := 20
  let green_marbles := 3 * red_marbles
  let yellow_marbles := 0.20 * green_marbles
  let total_marbles := green_marbles + 3 * green_marbles
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 := by
  sorry

end marbles_problem_l167_167880


namespace line_through_point_equal_distance_l167_167247

noncomputable def line_equation (x0 y0 a b c x1 y1 : ℝ) : Prop :=
  (a * x0 + b * y0 + c = 0) ∧ (a * x1 + b * y1 + c = 0)

theorem line_through_point_equal_distance (A B : ℝ × ℝ) (P : ℝ × ℝ) :
  ∃ (a b c : ℝ), 
    line_equation P.1 P.2 a b c A.1 A.2 ∧ 
    line_equation P.1 P.2 a b c B.1 B.2 ∧
    (a = 2) ∧ (b = 3) ∧ (c = -18) ∨
    (a = 2) ∧ (b = -1) ∧ (c = -2)
:=
sorry

end line_through_point_equal_distance_l167_167247


namespace chef_bought_almonds_l167_167382

theorem chef_bought_almonds (total_nuts pecans : ℝ)
  (h1 : total_nuts = 0.52) (h2 : pecans = 0.38) :
  total_nuts - pecans = 0.14 :=
by
  sorry

end chef_bought_almonds_l167_167382


namespace binary_representation_253_l167_167718

-- Define the decimal number
def decimal := 253

-- Define the number of zeros (x) and ones (y) in the binary representation of 253
def num_zeros := 1
def num_ones := 7

-- Prove that 2y - x = 13 given these conditions
theorem binary_representation_253 : (2 * num_ones - num_zeros) = 13 :=
by
  sorry

end binary_representation_253_l167_167718


namespace find_integer_l167_167698

theorem find_integer (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 150)
  (h2 : n % 7 = 0)
  (h3 : n % 9 = 3)
  (h4 : n % 6 = 3) : 
  n = 63 := by 
  sorry

end find_integer_l167_167698


namespace sum_of_variables_is_38_l167_167331

theorem sum_of_variables_is_38
  (x y z w : ℤ)
  (h₁ : x - y + z = 10)
  (h₂ : y - z + w = 15)
  (h₃ : z - w + x = 9)
  (h₄ : w - x + y = 4) :
  x + y + z + w = 38 := by
  sorry

end sum_of_variables_is_38_l167_167331


namespace total_gallons_l167_167163

def gallons_used (A F : ℕ) := F = 4 * A - 5

theorem total_gallons
  (A F : ℕ)
  (h1 : gallons_used A F)
  (h2 : F = 23) :
  A + F = 30 :=
by
  sorry

end total_gallons_l167_167163


namespace gcd_of_90_and_405_l167_167934

def gcd_90_405 : ℕ := Nat.gcd 90 405

theorem gcd_of_90_and_405 : gcd_90_405 = 45 :=
by
  -- proof goes here
  sorry

end gcd_of_90_and_405_l167_167934


namespace correlation_statements_l167_167330

variables {x y : ℝ}
variables (r : ℝ) (h1 : r > 0) (h2 : r = 1) (h3 : r = -1)

theorem correlation_statements :
  (r > 0 → (∀ x y, x > 0 → y > 0)) ∧
  (r = 1 ∨ r = -1 → (∀ x y, ∃ m b : ℝ, y = m * x + b)) :=
sorry

end correlation_statements_l167_167330


namespace valid_three_digit_numbers_no_seven_nine_l167_167461

noncomputable def count_valid_three_digit_numbers : Nat := 
  let hundredsChoices := 7
  let tensAndUnitsChoices := 8
  hundredsChoices * tensAndUnitsChoices * tensAndUnitsChoices

theorem valid_three_digit_numbers_no_seven_nine : 
  count_valid_three_digit_numbers = 448 := by
  sorry

end valid_three_digit_numbers_no_seven_nine_l167_167461


namespace cost_of_large_fries_l167_167021

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end cost_of_large_fries_l167_167021


namespace checkerboard_black_squares_l167_167774

theorem checkerboard_black_squares (n : ℕ) (hn : n = 33) :
  let black_squares : ℕ := (n * n + 1) / 2
  black_squares = 545 :=
by
  sorry

end checkerboard_black_squares_l167_167774


namespace compute_radii_sum_l167_167424

def points_on_circle (A B C D : ℝ × ℝ) (r : ℝ) : Prop :=
  let dist := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist A B) * (dist C D) = (dist A C) * (dist B D)

theorem compute_radii_sum :
  ∃ (r1 r2 : ℝ), points_on_circle (0,0) (-1,-1) (5,2) (6,2) r1
               ∧ points_on_circle (0,0) (-1,-1) (34,14) (35,14) r2
               ∧ r1 > 0
               ∧ r2 > 0
               ∧ r1 < r2
               ∧ r1^2 + r2^2 = 1381 :=
by {
  sorry -- proof not required
}

end compute_radii_sum_l167_167424


namespace range_of_a_l167_167659

theorem range_of_a (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 3 → x^2 - a * x - 3 ≤ 0) ↔ (2 ≤ a) := by
  sorry

end range_of_a_l167_167659


namespace g_does_not_pass_through_fourth_quadrant_l167_167131

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) - 2
noncomputable def g (x : ℝ) : ℝ := 1 + (1 / x)

theorem g_does_not_pass_through_fourth_quadrant (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 
    ¬(∃ x, x > 0 ∧ g x < 0) :=
by
    sorry

end g_does_not_pass_through_fourth_quadrant_l167_167131


namespace triangle_equilateral_l167_167773

variables {A B C : ℝ} -- angles of the triangle
variables {a b c : ℝ} -- sides opposite to the angles

-- Given conditions
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a * Real.cos C = c * Real.cos A ∧ (b * b = a * c)

-- The proof goal
theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c → a = b ∧ b = c :=
sorry

end triangle_equilateral_l167_167773


namespace part1_solution_set_of_inequality_part2_range_of_m_l167_167032

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x - 3|

theorem part1_solution_set_of_inequality :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/2} :=
by
  sorry

theorem part2_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x > 6 * m ^ 2 - 4 * m) ↔ -1/3 < m ∧ m < 1 :=
by
  sorry

end part1_solution_set_of_inequality_part2_range_of_m_l167_167032


namespace home_run_difference_l167_167738

def hank_aaron_home_runs : ℕ := 755
def dave_winfield_home_runs : ℕ := 465

theorem home_run_difference :
  2 * dave_winfield_home_runs - hank_aaron_home_runs = 175 := by
  sorry

end home_run_difference_l167_167738


namespace intersection_A_B_l167_167485

-- Definitions of sets A and B
def A := { x : ℝ | x ≥ -1 }
def B := { y : ℝ | y < 1 }

-- Statement to prove the intersection of A and B
theorem intersection_A_B : A ∩ B = { x : ℝ | -1 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l167_167485


namespace min_value_of_expr_l167_167688

noncomputable def min_expr (a b c : ℝ) := (2 * a / b) + (3 * b / c) + (4 * c / a)

theorem min_value_of_expr (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) 
    (habc : a * b * c = 1) : 
  min_expr a b c ≥ 9 := 
sorry

end min_value_of_expr_l167_167688


namespace find_other_number_l167_167354

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 2310) (h_hcf : Nat.gcd A B = 61) (h_a : A = 210) : B = 671 :=
by
  sorry

end find_other_number_l167_167354


namespace tom_seashells_l167_167751

theorem tom_seashells (days : ℕ) (seashells_per_day : ℕ) (h1 : days = 5) (h2 : seashells_per_day = 7) : 
  seashells_per_day * days = 35 := 
by
  sorry

end tom_seashells_l167_167751


namespace fraction_of_salary_spent_on_house_rent_l167_167530

theorem fraction_of_salary_spent_on_house_rent
    (S : ℕ) (H : ℚ)
    (cond1 : S = 180000)
    (cond2 : S / 5 + H * S + 3 * S / 5 + 18000 = S) :
    H = 1 / 10 := by
  sorry

end fraction_of_salary_spent_on_house_rent_l167_167530


namespace clients_using_radio_l167_167873

theorem clients_using_radio (total_clients T R M TR TM RM TRM : ℕ)
  (h1 : total_clients = 180)
  (h2 : T = 115)
  (h3 : M = 130)
  (h4 : TR = 75)
  (h5 : TM = 85)
  (h6 : RM = 95)
  (h7 : TRM = 80) : R = 30 :=
by
  -- Using Inclusion-Exclusion Principle
  have h : total_clients = T + R + M - TR - TM - RM + TRM :=
    sorry  -- Proof of Inclusion-Exclusion principle for these sets
  rw [h1, h2, h3, h4, h5, h6, h7] at h
  -- Solve for R
  sorry

end clients_using_radio_l167_167873


namespace highest_possible_average_l167_167367

theorem highest_possible_average (average_score : ℕ) (total_tests : ℕ) (lowest_score : ℕ) 
  (total_marks : ℕ := total_tests * average_score)
  (new_total_tests : ℕ := total_tests - 1)
  (resulting_average : ℚ := (total_marks - lowest_score) / new_total_tests) :
  average_score = 68 ∧ total_tests = 9 ∧ lowest_score = 0 → resulting_average = 76.5 := sorry

end highest_possible_average_l167_167367


namespace order_of_a_b_c_l167_167853

noncomputable def a : ℝ := (Real.log (Real.sqrt 2)) / 2
noncomputable def b : ℝ := Real.log 3 / 6
noncomputable def c : ℝ := 1 / (2 * Real.exp 1)

theorem order_of_a_b_c : c > b ∧ b > a := by
  sorry

end order_of_a_b_c_l167_167853


namespace equilateral_triangle_area_l167_167169

theorem equilateral_triangle_area (h : ℝ) (h_eq : h = 2 * Real.sqrt 3) : 
  (Real.sqrt 3 / 4) * (2 * h / (Real.sqrt 3))^2 = 4 * Real.sqrt 3 := 
by
  rw [h_eq]
  sorry

end equilateral_triangle_area_l167_167169


namespace min_value_a_4b_l167_167876

theorem min_value_a_4b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 / (a - 1) + 1 / (b - 1) = 1) : a + 4 * b = 14 := 
sorry

end min_value_a_4b_l167_167876


namespace businessmen_drink_neither_l167_167515

theorem businessmen_drink_neither : 
  ∀ (total coffee tea both : ℕ), 
    total = 30 → 
    coffee = 15 → 
    tea = 13 → 
    both = 8 → 
    total - (coffee - both + tea - both + both) = 10 := 
by 
  intros total coffee tea both h_total h_coffee h_tea h_both
  sorry

end businessmen_drink_neither_l167_167515


namespace ratio_x_to_w_as_percentage_l167_167916

theorem ratio_x_to_w_as_percentage (x y z w : ℝ) 
    (h1 : x = 1.20 * y) 
    (h2 : y = 0.30 * z) 
    (h3 : z = 1.35 * w) : 
    (x / w) * 100 = 48.6 := 
by sorry

end ratio_x_to_w_as_percentage_l167_167916


namespace locus_of_midpoint_of_chord_l167_167772

theorem locus_of_midpoint_of_chord 
  (A B C : ℝ) (h_arith_seq : A - 2 * B + C = 0) 
  (h_passing_through : ∀ t : ℝ,  t*A + -2*B + C = 0) :
  ∀ (x y : ℝ), 
    (Ax + By + C = 0) → 
    (h_on_parabola : y = -2 * x ^ 2) 
    → y + 1 = -(2 * x - 1) ^ 2 :=
sorry

end locus_of_midpoint_of_chord_l167_167772


namespace quadratic_equal_real_roots_l167_167077

theorem quadratic_equal_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) ↔ m = 1/4 :=
by sorry

end quadratic_equal_real_roots_l167_167077


namespace common_difference_l167_167504

theorem common_difference (a : ℕ → ℝ) (d : ℝ) (h_seq : ∀ n, a n = 1 + (n - 1) * d) 
  (h_geom : (a 3) ^ 2 = (a 1) * (a 13)) (h_ne_zero: d ≠ 0) : d = 2 :=
by
  sorry

end common_difference_l167_167504


namespace combined_height_of_rockets_l167_167412

noncomputable def height_of_rocket (a t : ℝ) : ℝ := (1/2) * a * t^2

theorem combined_height_of_rockets
  (h_A_ft : ℝ)
  (fuel_type_B_coeff : ℝ)
  (g : ℝ)
  (ft_to_m : ℝ)
  (h_combined : ℝ) :
  h_A_ft = 850 →
  fuel_type_B_coeff = 1.7 →
  g = 9.81 →
  ft_to_m = 0.3048 →
  h_combined = 348.96 :=
by sorry

end combined_height_of_rockets_l167_167412


namespace div_of_floats_l167_167838

theorem div_of_floats : (0.2 : ℝ) / (0.005 : ℝ) = 40 := 
by
  sorry

end div_of_floats_l167_167838


namespace consecutive_odd_integers_sum_l167_167495

theorem consecutive_odd_integers_sum (a b c : ℤ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (h3 : c % 2 = 1) (h4 : a < b) (h5 : b < c) (h6 : c = -47) : a + b + c = -141 := 
sorry

end consecutive_odd_integers_sum_l167_167495


namespace total_annual_interest_l167_167311

theorem total_annual_interest 
    (principal1 principal2 : ℝ)
    (rate1 rate2 : ℝ)
    (time : ℝ)
    (h1 : principal1 = 26000)
    (h2 : rate1 = 0.08)
    (h3 : principal2 = 24000)
    (h4 : rate2 = 0.085)
    (h5 : time = 1) :
    principal1 * rate1 * time + principal2 * rate2 * time = 4120 := 
sorry

end total_annual_interest_l167_167311


namespace problem_a_problem_b_l167_167684

noncomputable def gini_coefficient_separate_operations : ℝ := 
  let population_north := 24
  let population_south := population_north / 4
  let income_per_north_inhabitant := (6000 * 18) / population_north
  let income_per_south_inhabitant := (6000 * 12) / population_south
  let total_population := population_north + population_south
  let total_income := 6000 * (18 + 12)
  let share_pop_north := population_north / total_population
  let share_income_north := (income_per_north_inhabitant * population_north) / total_income
  share_pop_north - share_income_north

theorem problem_a : gini_coefficient_separate_operations = 0.2 := 
  by sorry

noncomputable def change_in_gini_coefficient_after_collaboration : ℝ :=
  let previous_income_north := 6000 * 18
  let compensation := previous_income_north + 1983
  let total_combined_income := 6000 * 30.5
  let remaining_income_south := total_combined_income - compensation
  let population := 24 + 6
  let income_per_capita_north := compensation / 24
  let income_per_capita_south := remaining_income_south / 6
  let new_gini_coefficient := 
    let share_pop_north := 24 / population
    let share_income_north := compensation / total_combined_income
    share_pop_north - share_income_north
  (0.2 - new_gini_coefficient)

theorem problem_b : change_in_gini_coefficient_after_collaboration = 0.001 := 
  by sorry

end problem_a_problem_b_l167_167684


namespace find_foreign_language_score_l167_167385

variable (c m f : ℝ)

theorem find_foreign_language_score
  (h1 : (c + m + f) / 3 = 94)
  (h2 : (c + m) / 2 = 92) :
  f = 98 := by
  sorry

end find_foreign_language_score_l167_167385


namespace find_D_l167_167207

-- Definitions
variable (A B C D E F : ℕ)

-- Conditions
axiom sum_AB : A + B = 16
axiom sum_BC : B + C = 12
axiom sum_EF : E + F = 8
axiom total_sum : A + B + C + D + E + F = 18

-- Theorem statement
theorem find_D : D = 6 :=
by
  sorry

end find_D_l167_167207


namespace cosine_angle_is_zero_l167_167223

-- Define the structure of an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  angle_60_deg : Prop

-- Define the structure of a parallelogram built from 6 equilateral triangles
structure Parallelogram where
  composed_of_6_equilateral_triangles : Prop
  folds_into_hexahedral_shape : Prop

-- Define the angle and its cosine computation between two specific directions in the folded hexahedral shape
def cosine_of_angle_between_AB_and_CD (parallelogram : Parallelogram) : ℝ := sorry

-- The condition that needs to be proved
axiom parallelogram_conditions : Parallelogram
axiom cosine_angle_proof : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0

-- Final proof statement
theorem cosine_angle_is_zero : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0 :=
cosine_angle_proof

end cosine_angle_is_zero_l167_167223


namespace opposite_of_negative_five_l167_167770

theorem opposite_of_negative_five : (-(-5) = 5) :=
by
  sorry

end opposite_of_negative_five_l167_167770


namespace jet_flight_distance_l167_167805

-- Setting up the hypotheses and the statement
theorem jet_flight_distance (v d : ℕ) (h1 : d = 4 * (v + 50)) (h2 : d = 5 * (v - 50)) : d = 2000 :=
sorry

end jet_flight_distance_l167_167805


namespace group_selection_l167_167389

theorem group_selection (m f : ℕ) (h1 : m + f = 8) (h2 : (m * (m - 1) / 2) * f = 30) : f = 3 :=
sorry

end group_selection_l167_167389


namespace value_of_difference_power_l167_167581

theorem value_of_difference_power (a b : ℝ) (h₁ : a^3 - 6 * a^2 + 15 * a = 9) 
                                  (h₂ : b^3 - 3 * b^2 + 6 * b = -1) 
                                  : (a - b)^2014 = 1 := 
by sorry

end value_of_difference_power_l167_167581


namespace infinity_gcd_binom_l167_167817

theorem infinity_gcd_binom {k l : ℕ} : ∃ᶠ m in at_top, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end infinity_gcd_binom_l167_167817


namespace find_num_of_boys_l167_167018

-- Define the constants for number of girls and total number of kids
def num_of_girls : ℕ := 3
def total_kids : ℕ := 9

-- The theorem stating the number of boys based on the given conditions
theorem find_num_of_boys (g t : ℕ) (h1 : g = num_of_girls) (h2 : t = total_kids) :
  t - g = 6 :=
by
  sorry

end find_num_of_boys_l167_167018


namespace system_solution_l167_167162

theorem system_solution (m n : ℚ) (x y : ℚ) 
  (h₁ : 2 * x + m * y = 5) 
  (h₂ : n * x - 3 * y = 2) 
  (h₃ : x = 3)
  (h₄ : y = 1) : 
  m / n = -3 / 5 :=
by sorry

end system_solution_l167_167162


namespace smallest_n_for_pencil_purchase_l167_167643

theorem smallest_n_for_pencil_purchase (a b c d n : ℕ)
  (h1 : 6 * a + 10 * b = n)
  (h2 : 6 * c + 10 * d = n + 2)
  (h3 : 7 * a + 12 * b > 7 * c + 12 * d)
  (h4 : 3 * (c - a) + 5 * (d - b) = 1)
  (h5 : d - b > 0) :
  n = 100 :=
by
  sorry

end smallest_n_for_pencil_purchase_l167_167643


namespace base_5_conversion_correct_l167_167034

def base_5_to_base_10 : ℕ := 2 * 5^2 + 4 * 5^1 + 2 * 5^0

theorem base_5_conversion_correct : base_5_to_base_10 = 72 :=
by {
  -- Proof (not required in the problem statement)
  sorry
}

end base_5_conversion_correct_l167_167034


namespace smallest_positive_integer_divisible_l167_167550

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l167_167550


namespace no_real_roots_iff_k_gt_1_div_4_l167_167882

theorem no_real_roots_iff_k_gt_1_div_4 (k : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - x + k = 0)) ↔ k > 1 / 4 :=
by
  sorry

end no_real_roots_iff_k_gt_1_div_4_l167_167882


namespace total_height_correct_l167_167114

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l167_167114


namespace james_points_l167_167732

theorem james_points (x : ℕ) :
  13 * 3 + 20 * x = 79 → x = 2 :=
by
  sorry

end james_points_l167_167732


namespace cubic_inequality_solution_l167_167419

theorem cubic_inequality_solution (x : ℝ) : x^3 - 12 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (9 < x) :=
by sorry

end cubic_inequality_solution_l167_167419


namespace find_f_2008_l167_167011

noncomputable def f : ℝ → ℝ := sorry

axiom f_zero : f 0 = 2008

axiom f_inequality1 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x
axiom f_inequality2 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x

theorem find_f_2008 : f 2008 = 2^2008 + 2007 :=
sorry

end find_f_2008_l167_167011


namespace expression_equivalence_l167_167693

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by
  sorry

end expression_equivalence_l167_167693


namespace book_pages_l167_167936

theorem book_pages (P : ℝ) (h1 : P / 2 + 0.15 * (P / 2) + 210 = P) : P = 600 := 
sorry

end book_pages_l167_167936


namespace solve_equation1_solve_equation2_l167_167537

-- Define the equations and the problem.
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = ((3 - x) / (x - 2))

-- Proof problem for the first equation: Prove that x = -4 is the solution.
theorem solve_equation1 : ∀ x : ℝ, equation1 x → x = -4 :=
by {
  sorry
}

-- Proof problem for the second equation: Prove that there are no solutions.
theorem solve_equation2 : ∀ x : ℝ, ¬equation2 x :=
by {
  sorry
}

end solve_equation1_solve_equation2_l167_167537


namespace product_is_in_A_l167_167748

def is_sum_of_squares (z : Int) : Prop :=
  ∃ t s : Int, z = t^2 + s^2

variable {x y : Int}

theorem product_is_in_A (hx : is_sum_of_squares x) (hy : is_sum_of_squares y) :
  is_sum_of_squares (x * y) :=
sorry

end product_is_in_A_l167_167748


namespace total_ants_found_l167_167288

-- Definitions for the number of ants each child finds
def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2

-- Statement that needs to be proven
theorem total_ants_found : abe_ants + beth_ants + cece_ants + duke_ants = 20 :=
by sorry

end total_ants_found_l167_167288


namespace each_tree_takes_one_square_foot_l167_167306

theorem each_tree_takes_one_square_foot (total_length : ℝ) (num_trees : ℕ) (gap_length : ℝ)
    (total_length_eq : total_length = 166) (num_trees_eq : num_trees = 16) (gap_length_eq : gap_length = 10) :
    (total_length - (((num_trees - 1) : ℝ) * gap_length)) / (num_trees : ℝ) = 1 :=
by
  rw [total_length_eq, num_trees_eq, gap_length_eq]
  sorry

end each_tree_takes_one_square_foot_l167_167306


namespace max_min_values_of_function_l167_167408

theorem max_min_values_of_function :
  (∀ x, 0 ≤ 2 * Real.sin x + 2 ∧ 2 * Real.sin x + 2 ≤ 4) ↔ (∃ x, 2 * Real.sin x + 2 = 0) ∧ (∃ y, 2 * Real.sin y + 2 = 4) :=
by
  sorry

end max_min_values_of_function_l167_167408


namespace positive_integer_solutions_l167_167289

theorem positive_integer_solutions (x : ℕ) (h : 2 * x + 9 ≥ 3 * (x + 2)) : x = 1 ∨ x = 2 ∨ x = 3 :=
by
  sorry

end positive_integer_solutions_l167_167289


namespace find_y_is_90_l167_167911

-- Definitions for given conditions
def angle_ABC : ℝ := 120
def angle_ABD : ℝ := 180 - angle_ABC
def angle_BDA : ℝ := 30

-- The theorem to prove y = 90 degrees
theorem find_y_is_90 :
  ∃ y : ℝ, angle_ABD = 60 ∧ angle_BDA = 30 ∧ (30 + 60 + y = 180) → y = 90 :=
by
  sorry

end find_y_is_90_l167_167911


namespace find_sales_discount_l167_167208

noncomputable def salesDiscountPercentage (P N : ℝ) (D : ℝ): Prop :=
  let originalGrossIncome := P * N
  let newPrice := P * (1 - D / 100)
  let newNumberOfItems := N * 1.20
  let newGrossIncome := newPrice * newNumberOfItems
  newGrossIncome = originalGrossIncome * 1.08

theorem find_sales_discount (P N : ℝ) (hP : P > 0) (hN : N > 0) (h: ∃ D, salesDiscountPercentage P N D) :
  ∃ D, D = 10 :=
sorry

end find_sales_discount_l167_167208


namespace no_nonzero_real_solutions_l167_167647

theorem no_nonzero_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ¬ (2 / x + 3 / y = 1 / (x + y)) :=
by sorry

end no_nonzero_real_solutions_l167_167647


namespace division_of_decimals_l167_167988

theorem division_of_decimals : 0.36 / 0.004 = 90 := by
  sorry

end division_of_decimals_l167_167988


namespace total_weekly_water_consumption_l167_167587

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end total_weekly_water_consumption_l167_167587


namespace max_distance_unit_circle_l167_167742

open Complex

theorem max_distance_unit_circle : 
  ∀ (z : ℂ), abs z = 1 → ∃ M : ℝ, M = abs (z - (1 : ℂ) - I) ∧ ∀ w : ℂ, abs w = 1 → abs (w - 1 - I) ≤ M :=
by
  sorry

end max_distance_unit_circle_l167_167742


namespace sum_zero_quotient_l167_167893

   theorem sum_zero_quotient (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + y + z = 0) :
     (xy + yz + zx) / (x^2 + y^2 + z^2) = -1 / 2 :=
   by
     sorry
   
end sum_zero_quotient_l167_167893


namespace throwers_count_l167_167798

variable (totalPlayers : ℕ) (rightHandedPlayers : ℕ) (nonThrowerLeftHandedFraction nonThrowerRightHandedFraction : ℚ)

theorem throwers_count
  (h1 : totalPlayers = 70)
  (h2 : rightHandedPlayers = 64)
  (h3 : nonThrowerLeftHandedFraction = 1 / 3)
  (h4 : nonThrowerRightHandedFraction = 2 / 3)
  (h5 : nonThrowerLeftHandedFraction + nonThrowerRightHandedFraction = 1) : 
  ∃ T : ℕ, T = 52 := by
  sorry

end throwers_count_l167_167798


namespace find_n_in_arithmetic_sequence_l167_167175

noncomputable def arithmetic_sequence (a1 d n : ℕ) := a1 + (n - 1) * d

theorem find_n_in_arithmetic_sequence (a1 d an : ℕ) (h1 : a1 = 1) (h2 : d = 5) (h3 : an = 2016) :
  ∃ n : ℕ, an = arithmetic_sequence a1 d n :=
  by
  sorry

end find_n_in_arithmetic_sequence_l167_167175


namespace functional_eq_one_l167_167358

theorem functional_eq_one (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
    (h2 : ∀ x > 0, ∀ y > 0, f x * f (y * f x) = f (x + y)) :
    ∀ x, 0 < x → f x = 1 := 
by
  sorry

end functional_eq_one_l167_167358


namespace sum_of_possible_values_l167_167191

theorem sum_of_possible_values (x y : ℝ) 
  (h : x * y - 2 * x / y ^ 3 - 2 * y / x ^ 3 = 4) : 
  (x - 2) * (y - 2) = 1 := 
sorry

end sum_of_possible_values_l167_167191


namespace number_of_integers_covered_l167_167912

-- Define the number line and the length condition
def unit_length_cm (p : ℝ) := p = 1
def length_AB_cm (length : ℝ) := length = 2009

-- Statement of the proof problem in Lean
theorem number_of_integers_covered (ab_length : ℝ) (unit_length : ℝ) 
    (h1 : unit_length_cm unit_length) (h2 : length_AB_cm ab_length) :
    ∃ n : ℕ, n = 2009 ∨ n = 2010 :=
by
  sorry

end number_of_integers_covered_l167_167912


namespace total_chocolate_bars_in_large_box_l167_167741

-- Define the given conditions
def small_boxes : ℕ := 16
def chocolate_bars_per_box : ℕ := 25

-- State the proof problem
theorem total_chocolate_bars_in_large_box :
  small_boxes * chocolate_bars_per_box = 400 :=
by
  -- The proof is omitted
  sorry

end total_chocolate_bars_in_large_box_l167_167741


namespace can_cut_rectangle_l167_167397

def original_rectangle_width := 100
def original_rectangle_height := 70
def total_area := original_rectangle_width * original_rectangle_height

def area1 := 1000
def area2 := 2000
def area3 := 4000

theorem can_cut_rectangle : 
  (area1 + area2 + area3 = total_area) ∧ 
  (area1 * 2 = area2) ∧ 
  (area1 * 4 = area3) ∧ 
  (area1 > 0) ∧ (area2 > 0) ∧ (area3 > 0) ∧
  (∃ (w1 h1 w2 h2 w3 h3 : ℕ), 
    w1 * h1 = area1 ∧ w2 * h2 = area2 ∧ w3 * h3 = area3 ∧
    ((w1 + w2 ≤ original_rectangle_width ∧ max h1 h2 + h3 ≤ original_rectangle_height) ∨
     (h1 + h2 ≤ original_rectangle_height ∧ max w1 w2 + w3 ≤ original_rectangle_width)))
:=
  sorry

end can_cut_rectangle_l167_167397


namespace ratio_third_to_first_second_l167_167923

-- Define the times spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_total : ℕ := 90
def time_third_step : ℕ := time_total - (time_first_step + time_second_step)

-- Define the combined time for the first two steps
def time_combined_first_second : ℕ := time_first_step + time_second_step

-- The goal is to prove that the ratio of the time spent on the third step to the combined time spent on the first and second steps is 1:1
theorem ratio_third_to_first_second : time_third_step = time_combined_first_second :=
by
  -- Proof goes here
  sorry

end ratio_third_to_first_second_l167_167923


namespace find_ravish_marks_l167_167090

-- Define the data according to the conditions.
def max_marks : ℕ := 200
def passing_percentage : ℕ := 40
def failed_by : ℕ := 40

-- The main theorem we need to prove.
theorem find_ravish_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) 
  (passing_marks := (max_marks * passing_percentage) / 100)
  (ravish_marks := passing_marks - failed_by) 
  : ravish_marks = 40 := by sorry

end find_ravish_marks_l167_167090


namespace number_of_black_and_white_films_l167_167240

theorem number_of_black_and_white_films (B x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_fraction : (6 * y : ℚ) / ((y / (x : ℚ))/100 * (B : ℚ) + 6 * y) = 20 / 21) :
  B = 30 * x :=
sorry

end number_of_black_and_white_films_l167_167240


namespace right_isosceles_hypotenuse_angle_l167_167245

theorem right_isosceles_hypotenuse_angle (α β : ℝ) (γ : ℝ)
  (h1 : α = 45) (h2 : β = 45) (h3 : γ = 90)
  (triangle_isosceles : α = β)
  (triangle_right : γ = 90) :
  γ = 90 :=
by
  sorry

end right_isosceles_hypotenuse_angle_l167_167245


namespace log_expression_value_l167_167497

theorem log_expression_value (lg : ℕ → ℤ) :
  (lg 4 = 2 * lg 2) →
  (lg 20 = lg 4 + lg 5) →
  lg 4 + lg 5 * lg 20 + (lg 5)^2 = 2 :=
by
  intros h1 h2
  sorry

end log_expression_value_l167_167497


namespace find_polynomial_parameters_and_minimum_value_l167_167881

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_polynomial_parameters_and_minimum_value 
  (a b c : ℝ)
  (h1 : f (-1) a b c = 7)
  (h2 : 3 * (-1)^2 + 2 * a * (-1) + b = 0)
  (h3 : 3 * 3^2 + 2 * a * 3 + b = 0)
  (h4 : a = -3)
  (h5 : b = -9)
  (h6 : c = 2) :
  f 3 (-3) (-9) 2 = -25 :=
by
  sorry

end find_polynomial_parameters_and_minimum_value_l167_167881


namespace age_of_B_l167_167999

theorem age_of_B (a b c d : ℕ) 
  (h1: a + b + c + d = 112)
  (h2: a + c = 58)
  (h3: 2 * b + 3 * d = 135)
  (h4: b + d = 54) :
  b = 27 :=
by
  sorry

end age_of_B_l167_167999


namespace subset_contains_square_l167_167451

theorem subset_contains_square {A : Finset ℕ} (hA₁ : A ⊆ Finset.range 101) (hA₂ : A.card = 50) (hA₃ : ∀ x ∈ A, ∀ y ∈ A, x + y ≠ 100) : 
  ∃ x ∈ A, ∃ k : ℕ, x = k^2 := 
sorry

end subset_contains_square_l167_167451


namespace seqAN_81_eq_640_l167_167620

-- Definitions and hypotheses
def seqAN (n : ℕ) : ℝ := sorry   -- A sequence a_n to be defined properly.

def sumSN (n : ℕ) : ℝ := sorry  -- The sum of the first n terms of a_n.

axiom condition_positivity : ∀ n : ℕ, 0 < seqAN n
axiom condition_a1 : seqAN 1 = 1
axiom condition_sum (n : ℕ) (h : 2 ≤ n) : 
  sumSN n * Real.sqrt (sumSN (n-1)) - sumSN (n-1) * Real.sqrt (sumSN n) = 
  2 * Real.sqrt (sumSN n * sumSN (n-1))

-- Proof problem: 
theorem seqAN_81_eq_640 : seqAN 81 = 640 := by sorry

end seqAN_81_eq_640_l167_167620


namespace curve_self_intersection_l167_167273

def curve_crosses_itself_at_point (x y : ℝ) : Prop :=
∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ (t₁^2 - 4 = x) ∧ (t₁^3 - 6 * t₁ + 7 = y) ∧ (t₂^2 - 4 = x) ∧ (t₂^3 - 6 * t₂ + 7 = y)

theorem curve_self_intersection : curve_crosses_itself_at_point 2 7 :=
sorry

end curve_self_intersection_l167_167273


namespace range_of_a_l167_167675

variable (a : ℝ)

def proposition_p : Prop :=
  ∃ x₀ : ℝ, x₀^2 - a * x₀ + a = 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 < x → x + 1 / (x - 1) ≥ a

theorem range_of_a (h : ¬proposition_p a ∧ proposition_q a) : 0 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l167_167675


namespace trig_expression_l167_167596

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end trig_expression_l167_167596


namespace simplify_and_evaluate_l167_167343

theorem simplify_and_evaluate (x : ℝ) (h : x^2 + 4 * x - 4 = 0) :
  3 * (x - 2) ^ 2 - 6 * (x + 1) * (x - 1) = 6 :=
by
  sorry

end simplify_and_evaluate_l167_167343


namespace find_k_l167_167071

def vec2 := ℝ × ℝ

-- Definitions
def i : vec2 := (1, 0)
def j : vec2 := (0, 1)
def a : vec2 := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : vec2 := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Dot product definition for 2D vectors
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by
  sorry

end find_k_l167_167071


namespace megan_popsicles_l167_167752

variable (t_rate : ℕ) (t_hours : ℕ)

def popsicles_eaten (rate: ℕ) (hours: ℕ) : ℕ :=
  60 * hours / rate

theorem megan_popsicles : popsicles_eaten 20 5 = 15 := by
  sorry

end megan_popsicles_l167_167752


namespace overall_average_marks_is_57_l167_167084

-- Define the number of students and average mark per class
def students_class_A := 26
def avg_marks_class_A := 40

def students_class_B := 50
def avg_marks_class_B := 60

def students_class_C := 35
def avg_marks_class_C := 55

def students_class_D := 45
def avg_marks_class_D := 65

-- Define the total marks per class
def total_marks_class_A := students_class_A * avg_marks_class_A
def total_marks_class_B := students_class_B * avg_marks_class_B
def total_marks_class_C := students_class_C * avg_marks_class_C
def total_marks_class_D := students_class_D * avg_marks_class_D

-- Define the grand total of marks
def grand_total_marks := total_marks_class_A + total_marks_class_B + total_marks_class_C + total_marks_class_D

-- Define the total number of students
def total_students := students_class_A + students_class_B + students_class_C + students_class_D

-- Define the overall average marks
def overall_avg_marks := grand_total_marks / total_students

-- The target theorem we want to prove
theorem overall_average_marks_is_57 : overall_avg_marks = 57 := by
  sorry

end overall_average_marks_is_57_l167_167084


namespace books_on_desk_none_useful_l167_167284

theorem books_on_desk_none_useful :
  ∃ (answer : String), answer = "none" ∧ 
  (answer = "nothing" ∨ answer = "no one" ∨ answer = "neither" ∨ answer = "none")
  → answer = "none"
:= by
  sorry

end books_on_desk_none_useful_l167_167284


namespace total_toys_correct_l167_167135

-- Define the conditions
def JaxonToys : ℕ := 15
def GabrielToys : ℕ := 2 * JaxonToys
def JerryToys : ℕ := GabrielToys + 8

-- Define the total number of toys
def TotalToys : ℕ := JaxonToys + GabrielToys + JerryToys

-- State the theorem
theorem total_toys_correct : TotalToys = 83 :=
by
  -- Skipping the proof as per instruction
  sorry

end total_toys_correct_l167_167135


namespace least_number_to_subtract_l167_167159

theorem least_number_to_subtract (n : ℕ) (h : n = 42739) : 
    ∃ k, k = 4 ∧ (n - k) % 15 = 0 := by
  sorry

end least_number_to_subtract_l167_167159


namespace leak_empties_in_24_hours_l167_167640

noncomputable def tap_rate := 1 / 6
noncomputable def combined_rate := 1 / 8
noncomputable def leak_rate := tap_rate - combined_rate
noncomputable def time_to_empty := 1 / leak_rate

theorem leak_empties_in_24_hours :
  time_to_empty = 24 := by
  sorry

end leak_empties_in_24_hours_l167_167640


namespace negation_of_implication_l167_167107

theorem negation_of_implication (x : ℝ) :
  ¬ (x ≠ 3 ∧ x ≠ 2 → x^2 - 5 * x + 6 ≠ 0) ↔ (x = 3 ∨ x = 2 → x^2 - 5 * x + 6 = 0) := 
by {
  sorry
}

end negation_of_implication_l167_167107


namespace solve_system_of_equations_l167_167416

theorem solve_system_of_equations :
  ∃ (x y : ℝ),
    (5 * x^2 - 14 * x * y + 10 * y^2 = 17) ∧ (4 * x^2 - 10 * x * y + 6 * y^2 = 8) ∧
    ((x = -1 ∧ y = -2) ∨ (x = 11 ∧ y = 7) ∨ (x = -11 ∧ y = -7) ∨ (x = 1 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l167_167416


namespace jack_pays_back_total_l167_167115

noncomputable def principal : ℝ := 1200
noncomputable def rate : ℝ := 0.10
noncomputable def interest : ℝ := principal * rate
noncomputable def total : ℝ := principal + interest

theorem jack_pays_back_total : total = 1320 := by
  sorry

end jack_pays_back_total_l167_167115


namespace sequence_length_l167_167085

theorem sequence_length :
  ∃ n : ℕ, ∀ (a_1 : ℤ) (d : ℤ) (a_n : ℤ), a_1 = -6 → d = 4 → a_n = 50 → a_n = a_1 + (n - 1) * d ∧ n = 15 :=
by
  sorry

end sequence_length_l167_167085


namespace trig_identity_l167_167755

open Real

theorem trig_identity :
  (1 - 1 / cos (23 * π / 180)) *
  (1 + 1 / sin (67 * π / 180)) *
  (1 - 1 / sin (23 * π / 180)) * 
  (1 + 1 / cos (67 * π / 180)) = 1 :=
by
  sorry

end trig_identity_l167_167755


namespace group_B_population_calculation_l167_167341

variable {total_population : ℕ}
variable {sample_size : ℕ}
variable {sample_A : ℕ}
variable {total_B : ℕ}

theorem group_B_population_calculation 
  (h_total : total_population = 200)
  (h_sample_size : sample_size = 40)
  (h_sample_A : sample_A = 16)
  (h_sample_B : sample_size - sample_A = 24) :
  total_B = 120 :=
sorry

end group_B_population_calculation_l167_167341


namespace max_value_a4_a6_l167_167521

theorem max_value_a4_a6 (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6) :
  ∃ m, ∀ (a : ℕ → ℝ) (d : ℝ) (h1 : d ≥ 0) (h2 : ∀ n, a n > 0) (h3 : a 3 + 2 * a 6 = 6), a 4 * a 6 ≤ m :=
sorry

end max_value_a4_a6_l167_167521


namespace problem_l167_167771

def op (x y : ℝ) : ℝ := x^2 - y

theorem problem (h : ℝ) : op h (op h h) = h :=
by
  sorry

end problem_l167_167771


namespace proof_problem_l167_167955

variable (x y : ℕ) -- define x and y as natural numbers

-- Define the problem-specific variables m and n
variable (m n : ℕ)

-- Assume the conditions given in the problem
axiom H1 : 2 = m
axiom H2 : n = 3

-- The goal is to prove that -m^n equals -8 given the conditions H1 and H2
theorem proof_problem : - (m^n : ℤ) = -8 :=
by
  sorry

end proof_problem_l167_167955


namespace simplify_expression_l167_167217

theorem simplify_expression (m n : ℝ) (h : m^2 + 3 * m * n = 5) : 
  5 * m^2 - 3 * m * n - (-9 * m * n + 3 * m^2) = 10 :=
by 
  sorry

end simplify_expression_l167_167217


namespace infinitely_many_a_not_sum_of_seven_sixth_powers_l167_167603

theorem infinitely_many_a_not_sum_of_seven_sixth_powers :
  ∃ᶠ (a: ℕ) in at_top, (∀ (a_i : ℕ) (h0 : a_i > 0), a ≠ a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 ∧ a % 9 = 8) :=
sorry

end infinitely_many_a_not_sum_of_seven_sixth_powers_l167_167603


namespace sin_half_alpha_plus_beta_eq_sqrt2_div_2_l167_167444

open Real

theorem sin_half_alpha_plus_beta_eq_sqrt2_div_2
  (α β : ℝ)
  (hα : α ∈ Set.Icc (π / 2) (3 * π / 2))
  (hβ : β ∈ Set.Icc (-π / 2) 0)
  (h1 : (α - π / 2)^3 - sin α - 2 = 0)
  (h2 : 8 * β^3 + 2 * (cos β)^2 + 1 = 0) :
  sin (α / 2 + β) = sqrt 2 / 2 := 
sorry

end sin_half_alpha_plus_beta_eq_sqrt2_div_2_l167_167444


namespace maria_average_speed_l167_167920

theorem maria_average_speed:
  let distance1 := 180
  let time1 := 4.5
  let distance2 := 270
  let time2 := 5.25
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time = 46.15 := by
  -- Sorry to skip the proof
  sorry

end maria_average_speed_l167_167920


namespace find_fraction_value_l167_167442

theorem find_fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : (m * n) / (m - n) = -1/6 :=
sorry

end find_fraction_value_l167_167442


namespace stratified_sampling_l167_167177

-- Definition of conditions as hypothesis
def total_employees : ℕ := 100
def under_35 : ℕ := 45
def between_35_49 : ℕ := 25
def over_50 : ℕ := total_employees - under_35 - between_35_49
def sample_size : ℕ := 20
def sampling_ratio : ℚ := sample_size / total_employees

-- The target number of people from each group
def under_35_sample : ℚ := sampling_ratio * under_35
def between_35_49_sample : ℚ := sampling_ratio * between_35_49
def over_50_sample : ℚ := sampling_ratio * over_50

-- Problem statement
theorem stratified_sampling : 
  under_35_sample = 9 ∧ 
  between_35_49_sample = 5 ∧ 
  over_50_sample = 6 :=
  by
  sorry

end stratified_sampling_l167_167177


namespace peaches_eaten_correct_l167_167254

-- Given conditions
def total_peaches : ℕ := 18
def initial_ripe_peaches : ℕ := 4
def peaches_ripen_per_day : ℕ := 2
def days_passed : ℕ := 5
def ripe_unripe_difference : ℕ := 7

-- Definitions derived from conditions
def ripe_peaches_after_days := initial_ripe_peaches + peaches_ripen_per_day * days_passed
def unripe_peaches_initial := total_peaches - initial_ripe_peaches
def unripe_peaches_after_days := unripe_peaches_initial - peaches_ripen_per_day * days_passed
def actual_ripe_peaches_needed := unripe_peaches_after_days + ripe_unripe_difference
def peaches_eaten := ripe_peaches_after_days - actual_ripe_peaches_needed

-- Prove that the number of peaches eaten is equal to 3
theorem peaches_eaten_correct : peaches_eaten = 3 := by
  sorry

end peaches_eaten_correct_l167_167254


namespace at_least_one_divisible_by_5_l167_167571

theorem at_least_one_divisible_by_5 (k m n : ℕ) (hk : ¬ (5 ∣ k)) (hm : ¬ (5 ∣ m)) (hn : ¬ (5 ∣ n)) : 
  (5 ∣ (k^2 - m^2)) ∨ (5 ∣ (m^2 - n^2)) ∨ (5 ∣ (n^2 - k^2)) :=
by {
    sorry
}

end at_least_one_divisible_by_5_l167_167571


namespace rectangle_area_y_value_l167_167426

theorem rectangle_area_y_value :
  ∀ (y : ℝ), 
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  (y > 1) → 
  (abs (R.1 - P.1) * abs (Q.2 - P.2) = 36) → 
  y = 13 :=
by
  intros y P Q R S hy harea
  let P := (1, 1)
  let Q := (1, 4)
  let R := (y, 4)
  let S := (y, 1)
  sorry

end rectangle_area_y_value_l167_167426


namespace vote_percentage_for_candidate_A_l167_167324

noncomputable def percent_democrats : ℝ := 0.60
noncomputable def percent_republicans : ℝ := 0.40
noncomputable def percent_voting_a_democrats : ℝ := 0.70
noncomputable def percent_voting_a_republicans : ℝ := 0.20

theorem vote_percentage_for_candidate_A :
    (percent_democrats * percent_voting_a_democrats + percent_republicans * percent_voting_a_republicans) * 100 = 50 := by
  sorry

end vote_percentage_for_candidate_A_l167_167324


namespace certain_number_l167_167333

theorem certain_number (n w : ℕ) (h1 : w = 132)
  (h2 : ∃ m1 m2 m3, 32 = 2^5 * 3^3 * 11^2 * m1 * m2 * m3)
  (h3 : n * w = 132 * 2^3 * 3^2 * 11)
  (h4 : m1 = 1) (h5 : m2 = 1) (h6 : m3 = 1): 
  n = 792 :=
by sorry

end certain_number_l167_167333


namespace square_side_to_diagonal_ratio_l167_167777

theorem square_side_to_diagonal_ratio (s : ℝ) : 
  s / (s * Real.sqrt 2) = Real.sqrt 2 / 2 :=
by
  sorry

end square_side_to_diagonal_ratio_l167_167777


namespace vector_arithmetic_l167_167425

-- Define the vectors
def v1 : ℝ × ℝ := (3, -5)
def v2 : ℝ × ℝ := (2, -6)
def v3 : ℝ × ℝ := (-1, 4)

-- Define scalar multiplications
def scalar_mult1 : ℝ × ℝ := (12, -20)  -- 4 * v1
def scalar_mult2 : ℝ × ℝ := (6, -18)   -- 3 * v2

-- Define intermediate vector operations
def intermediate_vector1 : ℝ × ℝ := (6, -2)  -- (12, -20) - (6, -18)

-- Final operation
def final_vector : ℝ × ℝ := (5, 2)  -- (6, -2) + (-1, 4)

-- Prove the main statement
theorem vector_arithmetic : 
  (4 : ℝ) • v1 - (3 : ℝ) • v2 + v3 = final_vector := by
  sorry  -- proof placeholder

end vector_arithmetic_l167_167425


namespace find_m_l167_167356

theorem find_m (m : ℕ) : 5 ^ m = 5 * 25 ^ 2 * 125 ^ 3 ↔ m = 14 := by
  sorry

end find_m_l167_167356


namespace skillful_hands_award_prob_cannot_enter_finals_after_training_l167_167775

noncomputable def combinatorial_probability : ℚ :=
  let P1 := (4 * 3) / (10 * 10)    -- P1: 1 specified, 2 creative
  let P2 := (6 * 3) / (10 * 10)    -- P2: 2 specified, 1 creative
  let P3 := (6 * 3) / (10 * 10)    -- P3: 2 specified, 2 creative
  P1 + P2 + P3

theorem skillful_hands_award_prob : combinatorial_probability = 33 / 50 := 
  sorry

def after_training_probability := 3 / 4
theorem cannot_enter_finals_after_training : after_training_probability * 5 < 4 := 
  sorry

end skillful_hands_award_prob_cannot_enter_finals_after_training_l167_167775


namespace eggs_used_to_bake_cake_l167_167552

theorem eggs_used_to_bake_cake
    (initial_eggs : ℕ)
    (omelet_eggs : ℕ)
    (aunt_eggs : ℕ)
    (meal_eggs : ℕ)
    (num_meals : ℕ)
    (remaining_eggs_after_omelet : initial_eggs - omelet_eggs = 22)
    (eggs_given_to_aunt : 2 * aunt_eggs = initial_eggs - omelet_eggs)
    (remaining_eggs_after_aunt : initial_eggs - omelet_eggs - aunt_eggs = 11)
    (total_eggs_for_meals : meal_eggs * num_meals = 9)
    (remaining_eggs_after_meals : initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2) :
  initial_eggs - omelet_eggs - aunt_eggs - meal_eggs * num_meals = 2 :=
sorry

end eggs_used_to_bake_cake_l167_167552


namespace dinitrogen_monoxide_molecular_weight_l167_167783

def atomic_weight_N : Real := 14.01
def atomic_weight_O : Real := 16.00

def chemical_formula_N2O_weight : Real :=
  (2 * atomic_weight_N) + (1 * atomic_weight_O)

theorem dinitrogen_monoxide_molecular_weight :
  chemical_formula_N2O_weight = 44.02 :=
by
  sorry

end dinitrogen_monoxide_molecular_weight_l167_167783


namespace smallest_Y_l167_167173

theorem smallest_Y (U : ℕ) (Y : ℕ) (hU : U = 15 * Y) 
  (digits_U : ∀ d ∈ Nat.digits 10 U, d = 0 ∨ d = 1) 
  (div_15 : U % 15 = 0) : Y = 74 :=
sorry

end smallest_Y_l167_167173


namespace ratio_xy_half_l167_167898

noncomputable def common_ratio_k (x y z : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) : ℝ := sorry

theorem ratio_xy_half (x y z k : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  ∃ k, (x + 4) = 2 * k ∧ (y + 9) = k * (z - 3) ∧ (x + 5) = k * (z - 5) → (x / y) = 1 / 2 :=
sorry

end ratio_xy_half_l167_167898


namespace max_2b_div_a_l167_167196

theorem max_2b_div_a (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : 
  ∃ max_val, max_val = (2 * b) / a ∧ max_val = (32 / 3) :=
by
  sorry

end max_2b_div_a_l167_167196


namespace mimi_spending_adidas_l167_167304

theorem mimi_spending_adidas
  (total_spending : ℤ)
  (nike_to_adidas_ratio : ℤ)
  (adidas_to_skechers_ratio : ℤ)
  (clothes_spending : ℤ)
  (eq1 : total_spending = 8000)
  (eq2 : nike_to_adidas_ratio = 3)
  (eq3 : adidas_to_skechers_ratio = 5)
  (eq4 : clothes_spending = 2600) :
  ∃ A : ℤ, A + nike_to_adidas_ratio * A + adidas_to_skechers_ratio * A + clothes_spending = total_spending ∧ A = 600 := by
  sorry

end mimi_spending_adidas_l167_167304


namespace solve_congruence_l167_167703

theorem solve_congruence : ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ 11 * n % 43 = 7 :=
by
  sorry

end solve_congruence_l167_167703


namespace real_cube_inequality_l167_167252

theorem real_cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end real_cube_inequality_l167_167252


namespace pascal_triangle_ratios_l167_167789
open Nat

theorem pascal_triangle_ratios :
  ∃ n r : ℕ, 
  (choose n r) * 4 = (choose n (r + 1)) * 3 ∧ 
  (choose n (r + 1)) * 3 = (choose n (r + 2)) * 4 ∧ 
  n = 34 :=
by
  sorry

end pascal_triangle_ratios_l167_167789


namespace woman_works_finish_days_l167_167790

theorem woman_works_finish_days (M W : ℝ) 
  (hm_work : ∀ n : ℝ, n * M = 1 / 100)
  (hw_work : ∀ men women : ℝ, (10 * M + 15 * women) * 6 = 1) :
  W = 1 / 225 :=
by
  have man_work := hm_work 1
  have woman_work := hw_work 10 W
  sorry

end woman_works_finish_days_l167_167790


namespace prism_cubes_paint_condition_l167_167690

theorem prism_cubes_paint_condition
  (m n r : ℕ)
  (h1 : m ≤ n)
  (h2 : n ≤ r)
  (h3 : (m - 2) * (n - 2) * (r - 2)
        - 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2)) 
        + 4 * (m - 2 + n - 2 + r - 2)
        = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := 
sorry

end prism_cubes_paint_condition_l167_167690


namespace triangle_side_length_x_l167_167447

theorem triangle_side_length_x (x : ℤ) (hpos : x > 0) (hineq1 : 7 < x^2) (hineq2 : x^2 < 17) :
    x = 3 ∨ x = 4 :=
by {
  apply sorry
}

end triangle_side_length_x_l167_167447


namespace initial_amount_correct_l167_167070

noncomputable def initial_amount (A R T : ℝ) : ℝ :=
  A / (1 + (R * T) / 100)

theorem initial_amount_correct :
  initial_amount 2000 3.571428571428571 4 = 1750 :=
by
  sorry

end initial_amount_correct_l167_167070


namespace unique_surjective_f_l167_167203

-- Define the problem conditions
variable (f : ℕ → ℕ)

-- Define that f is surjective
axiom surjective_f : Function.Surjective f

-- Define condition that for every m, n and prime p
axiom condition_f : ∀ m n : ℕ, ∀ p : ℕ, Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)

-- The theorem we need to prove: the only surjective function f satisfying the condition is the identity function
theorem unique_surjective_f : ∀ x : ℕ, f x = x :=
by
  sorry

end unique_surjective_f_l167_167203


namespace percentage_died_by_bombardment_l167_167248

def initial_population : ℕ := 4675
def remaining_population : ℕ := 3553
def left_percentage : ℕ := 20

theorem percentage_died_by_bombardment (x : ℕ) (h : initial_population * (100 - x) / 100 * 8 / 10 = remaining_population) : 
  x = 5 :=
by
  sorry

end percentage_died_by_bombardment_l167_167248


namespace range_a_satisfies_l167_167285

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end range_a_satisfies_l167_167285


namespace price_per_litre_of_second_oil_l167_167390

-- Define the conditions given in the problem
def oil1_volume : ℝ := 10 -- 10 litres of first oil
def oil1_rate : ℝ := 50 -- Rs. 50 per litre

def oil2_volume : ℝ := 5 -- 5 litres of the second oil
def total_mixed_volume : ℝ := oil1_volume + oil2_volume -- Total volume of mixed oil

def mixed_rate : ℝ := 55.33 -- Rs. 55.33 per litre for the mixed oil

-- Define the target value to prove: price per litre of the second oil
def price_of_second_oil : ℝ := 65.99

-- Prove the statement
theorem price_per_litre_of_second_oil : 
  (oil1_volume * oil1_rate + oil2_volume * price_of_second_oil) = total_mixed_volume * mixed_rate :=
by 
  sorry -- actual proof to be provided

end price_per_litre_of_second_oil_l167_167390


namespace max_pairs_correct_l167_167990

def max_pairs (n : ℕ) : ℕ :=
  if h : n > 1 then (n * n) / 4 else 0

theorem max_pairs_correct (n : ℕ) (h : n ≥ 2) :
  (max_pairs n = (n * n) / 4) :=
by sorry

end max_pairs_correct_l167_167990


namespace inversely_proportional_ratios_l167_167804

theorem inversely_proportional_ratios (x y x₁ x₂ y₁ y₂ : ℝ) (hx_inv : ∀ x y, x * y = 1)
  (hx_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end inversely_proportional_ratios_l167_167804


namespace problem_statement_l167_167931

theorem problem_statement (n : ℕ) : 2 ^ n ∣ (1 + ⌊(3 + Real.sqrt 5) ^ n⌋) :=
by
  sorry

end problem_statement_l167_167931


namespace compare_cube_roots_l167_167642

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem compare_cube_roots : 2 + cube_root 7 < cube_root 60 :=
sorry

end compare_cube_roots_l167_167642


namespace function_behavior_l167_167544

noncomputable def f (x : ℝ) : ℝ := abs (2^x - 2)

theorem function_behavior :
  (∀ x y : ℝ, x < y ∧ y ≤ 1 → f x ≥ f y) ∧ (∀ x y : ℝ, x < y ∧ x ≥ 1 → f x ≤ f y) :=
by
  sorry

end function_behavior_l167_167544


namespace find_k_l167_167914

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k, -2)
def b : ℝ × ℝ := (2, 2)

-- Condition 1: a + b is not the zero vector
def non_zero_sum (k : ℝ) := (a k).1 + b.1 ≠ 0 ∨ (a k).2 + b.2 ≠ 0

-- Condition 2: a is perpendicular to a + b
def perpendicular (k : ℝ) := (a k).1 * ((a k).1 + b.1) + (a k).2 * ((a k).2 + b.2) = 0

-- The theorem to prove
theorem find_k (k : ℝ) (cond1 : non_zero_sum k) (cond2 : perpendicular k) : k = 0 := 
sorry

end find_k_l167_167914


namespace martha_makes_40_cookies_martha_needs_7_5_cups_l167_167835

theorem martha_makes_40_cookies :
  (24 / 3) * 5 = 40 :=
by
  sorry

theorem martha_needs_7_5_cups :
  60 / (24 / 3) = 7.5 :=
by
  sorry

end martha_makes_40_cookies_martha_needs_7_5_cups_l167_167835


namespace value_of_g_800_l167_167000

noncomputable def g : ℝ → ℝ :=
sorry

theorem value_of_g_800 (g_eq : ∀ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), g (x * y) = g x / (y^2))
  (g_at_1000 : g 1000 = 4) : g 800 = 625 / 2 :=
sorry

end value_of_g_800_l167_167000


namespace circumscribed_circle_radius_l167_167676

noncomputable def radius_of_circumcircle (a b c : ℚ) (h_a : a = 15/2) (h_b : b = 10) (h_c : c = 25/2) : ℚ :=
if h_triangle : a^2 + b^2 = c^2 then (c / 2) else 0

theorem circumscribed_circle_radius :
  radius_of_circumcircle (15/2 : ℚ) 10 (25/2 : ℚ) (by norm_num) (by norm_num) (by norm_num) = 25 / 4 := 
by
  sorry

end circumscribed_circle_radius_l167_167676


namespace simplify_fraction_l167_167517

theorem simplify_fraction (a b gcd : ℕ) (h1 : a = 72) (h2 : b = 108) (h3 : gcd = Nat.gcd a b) : (a / gcd) / (b / gcd) = 2 / 3 :=
by
  -- the proof is omitted here
  sorry

end simplify_fraction_l167_167517


namespace original_numbers_l167_167593

theorem original_numbers (a b c d : ℝ) (h1 : a + b + c + d = 45)
    (h2 : ∃ x : ℝ, a + 2 = x ∧ b - 2 = x ∧ 2 * c = x ∧ d / 2 = x) : 
    a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 :=
by
  sorry

end original_numbers_l167_167593


namespace range_of_values_for_a_l167_167828

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_values_for_a_l167_167828


namespace white_balls_count_l167_167379

theorem white_balls_count (a : ℕ) (h : 3 / (3 + a) = 3 / 7) : a = 4 :=
by sorry

end white_balls_count_l167_167379


namespace reporters_covering_local_politics_l167_167507

theorem reporters_covering_local_politics (R : ℕ) (P Q A B : ℕ)
  (h1 : P = 70)
  (h2 : Q = 100 - P)
  (h3 : A = 40)
  (h4 : B = 100 - A) :
  B % 30 = 18 :=
by
  sorry

end reporters_covering_local_politics_l167_167507


namespace cole_drive_time_l167_167885

theorem cole_drive_time (d : ℝ) (h1 : d / 75 + d / 105 = 1) : (d / 75) * 60 = 35 :=
by
  -- Using the given equation: d / 75 + d / 105 = 1
  -- We solve it step by step and finally show that the time it took to drive to work is 35 minutes.
  sorry

end cole_drive_time_l167_167885


namespace arithmetic_problem_l167_167047

theorem arithmetic_problem : 
  (888.88 - 555.55 + 111.11) * 2 = 888.88 := 
sorry

end arithmetic_problem_l167_167047


namespace quadratic_inequality_solution_l167_167363

theorem quadratic_inequality_solution (x m : ℝ) :
  (x^2 + (2*m + 1)*x + m^2 + m > 0) ↔ (x > -m ∨ x < -m - 1) :=
by
  sorry

end quadratic_inequality_solution_l167_167363


namespace problem1_problem2_l167_167287

theorem problem1 : -20 - (-8) + (-4) = -16 := by
  sorry

theorem problem2 : -1^3 * (-2)^2 / (4 / 3 : ℚ) + |5 - 8| = 0 := by
  sorry

end problem1_problem2_l167_167287


namespace line_transformation_l167_167300

theorem line_transformation (a b : ℝ)
  (h1 : ∀ x y : ℝ, a * x + y - 7 = 0)
  (A : Matrix (Fin 2) (Fin 2) ℝ) (hA : A = ![![3, 0], ![-1, b]])
  (h2 : ∀ x' y' : ℝ, 9 * x' + y' - 91 = 0) :
  (a = 2) ∧ (b = 13) :=
by
  sorry

end line_transformation_l167_167300


namespace pencil_cost_is_4_l167_167348

variables (pencils pens : ℕ) (pen_cost total_cost : ℕ)

def total_pencils := 15 * 80
def total_pens := (2 * total_pencils) + 300
def total_pen_cost := total_pens * pen_cost
def total_pencil_cost := total_cost - total_pen_cost
def pencil_cost := total_pencil_cost / total_pencils

theorem pencil_cost_is_4
  (pen_cost_eq_5 : pen_cost = 5)
  (total_cost_eq_18300 : total_cost = 18300)
  : pencil_cost = 4 :=
by
  sorry

end pencil_cost_is_4_l167_167348


namespace circle_equation_exists_l167_167592

theorem circle_equation_exists :
  ∃ (x_c y_c r : ℝ), 
  x_c > 0 ∧ y_c > 0 ∧ 0 < r ∧ r < 5 ∧ (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = r^2) :=
sorry

end circle_equation_exists_l167_167592


namespace max_d_6_digit_multiple_33_l167_167993

theorem max_d_6_digit_multiple_33 (x d e : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 9) 
  (hd : 0 ≤ d ∧ d ≤ 9) 
  (he : 0 ≤ e ∧ e ≤ 9)
  (h1 : (x * 100000 + 50000 + d * 1000 + 300 + 30 + e) ≥ 100000) 
  (h2 : (x + d + e + 11) % 3 = 0)
  (h3 : ((x + d - e - 5 + 11) % 11 = 0)) :
  d = 9 := 
sorry

end max_d_6_digit_multiple_33_l167_167993


namespace bananas_to_oranges_l167_167423

theorem bananas_to_oranges :
  (3 / 4) * 16 * (1 / 1 : ℝ) = 10 * (1 / 1 : ℝ) → 
  (3 / 5) * 15 * (1 / 1 : ℝ) = 7.5 * (1 / 1 : ℝ) := 
by
  intros h
  sorry

end bananas_to_oranges_l167_167423


namespace convert_degrees_to_radians_l167_167869

theorem convert_degrees_to_radians (deg : ℝ) (deg_eq : deg = -300) : 
  deg * (π / 180) = - (5 * π) / 3 := 
by
  rw [deg_eq]
  sorry

end convert_degrees_to_radians_l167_167869


namespace smaller_odd_number_l167_167619

theorem smaller_odd_number (n : ℤ) (h : n + (n + 2) = 48) : n = 23 :=
by
  sorry

end smaller_odd_number_l167_167619


namespace Joyce_final_apples_l167_167342

def initial_apples : ℝ := 350.5
def apples_given_to_larry : ℝ := 218.7
def percentage_given_to_neighbors : ℝ := 0.375
def final_apples : ℝ := 82.375

theorem Joyce_final_apples :
  (initial_apples - apples_given_to_larry - percentage_given_to_neighbors * (initial_apples - apples_given_to_larry)) = final_apples :=
by
  sorry

end Joyce_final_apples_l167_167342


namespace sufficient_but_not_necessary_condition_l167_167761

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, |2*x - 1| ≤ x → x^2 + x - 2 ≤ 0) ∧ 
  ¬(∀ x : ℝ, x^2 + x - 2 ≤ 0 → |2 * x - 1| ≤ x) := sorry

end sufficient_but_not_necessary_condition_l167_167761


namespace pair_d_same_function_l167_167978

theorem pair_d_same_function : ∀ x : ℝ, x = (x ^ 5) ^ (1 / 5) := 
by
  intro x
  sorry

end pair_d_same_function_l167_167978


namespace find_y_l167_167519

theorem find_y (x y : ℕ) (h1 : x^2 = y + 3) (h2 : x = 6) : y = 33 := 
by
  sorry

end find_y_l167_167519


namespace toothpick_grid_l167_167818

theorem toothpick_grid (l w : ℕ) (h_l : l = 45) (h_w : w = 25) :
  let effective_vertical_lines := l + 1 - (l + 1) / 5
  let effective_horizontal_lines := w + 1 - (w + 1) / 5
  let vertical_toothpicks := effective_vertical_lines * w
  let horizontal_toothpicks := effective_horizontal_lines * l
  let total_toothpicks := vertical_toothpicks + horizontal_toothpicks
  total_toothpicks = 1722 :=
by {
  sorry
}

end toothpick_grid_l167_167818


namespace find_question_mark_l167_167074

noncomputable def c1 : ℝ := (5568 / 87)^(1/3)
noncomputable def c2 : ℝ := (72 * 2)^(1/2)
noncomputable def sum_c1_c2 : ℝ := c1 + c2

theorem find_question_mark : sum_c1_c2 = 16 → 256 = 16^2 :=
by
  sorry

end find_question_mark_l167_167074


namespace product_of_solutions_eq_zero_l167_167935

theorem product_of_solutions_eq_zero : 
  (∀ x : ℝ, (x + 3) / (2 * x + 3) = (4 * x + 4) / (7 * x + 4)) → 
  ∃ (x1 x2 : ℝ), (x1 = 0 ∨ x1 = 5) ∧ (x2 = 0 ∨ x2 = 5) ∧ x1 * x2 = 0 :=
by
  sorry

end product_of_solutions_eq_zero_l167_167935


namespace isosceles_triangle_congruent_side_length_l167_167194

theorem isosceles_triangle_congruent_side_length (BC : ℝ) (BM : ℝ) :
  BC = 4 * Real.sqrt 2 → BM = 5 → ∃ (AB : ℝ), AB = Real.sqrt 34 :=
by
  -- sorry is used here to indicate proof is not provided, but the statement is expected to build successfully.
  sorry

end isosceles_triangle_congruent_side_length_l167_167194


namespace ratio_of_spending_is_one_to_two_l167_167588

-- Definitions
def initial_amount : ℕ := 24
def doris_spent : ℕ := 6
def final_amount : ℕ := 15

-- Amount remaining after Doris spent
def remaining_after_doris : ℕ := initial_amount - doris_spent

-- Amount Martha spent
def martha_spent : ℕ := remaining_after_doris - final_amount

-- Ratio of the amounts spent
def ratio_martha_doris : ℕ × ℕ := (martha_spent, doris_spent)

-- Theorem to prove
theorem ratio_of_spending_is_one_to_two : ratio_martha_doris = (1, 2) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_spending_is_one_to_two_l167_167588


namespace years_taught_third_grade_l167_167856

def total_years : ℕ := 26
def years_taught_second_grade : ℕ := 8

theorem years_taught_third_grade :
  total_years - years_taught_second_grade = 18 :=
by {
  -- Subtract the years taught second grade from the total years
  -- Exact the result
  sorry
}

end years_taught_third_grade_l167_167856


namespace evaluate_expression_l167_167470

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end evaluate_expression_l167_167470


namespace train_length_l167_167481

def speed_kmph := 72   -- Speed in kilometers per hour
def time_sec := 14     -- Time in seconds

/-- Function to convert speed from km/hr to m/s -/
def convert_speed (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

/-- Function to calculate distance given speed and time -/
def calculate_distance (speed : ℕ) (time : ℕ) : ℕ :=
  speed * time

theorem train_length :
  calculate_distance (convert_speed speed_kmph) time_sec = 280 :=
by
  sorry

end train_length_l167_167481


namespace identify_quadratic_l167_167048

def is_quadratic (eq : String) : Prop :=
  eq = "x^2 - 2x + 1 = 0"

theorem identify_quadratic :
  is_quadratic "x^2 - 2x + 1 = 0" :=
by
  sorry

end identify_quadratic_l167_167048


namespace triangle_classification_l167_167313

theorem triangle_classification (a b c : ℕ) (h : a + b + c = 12) :
((
  (a = b ∨ b = c ∨ a = c)  -- Isosceles
  ∨ (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2)  -- Right-angled
  ∨ (a = b ∧ b = c)  -- Equilateral
)) :=
sorry

end triangle_classification_l167_167313


namespace middle_elementary_students_l167_167322

theorem middle_elementary_students (S S_PS S_MS S_MR : ℕ) 
  (h1 : S = 12000)
  (h2 : S_PS = (15 * S) / 16)
  (h3 : S_MS = S - S_PS)
  (h4 : S_MR + S_MS = (S_PS) / 2) : 
  S_MR = 4875 :=
by
  sorry

end middle_elementary_students_l167_167322


namespace dacid_weighted_average_l167_167849

theorem dacid_weighted_average :
  let english := 96
  let mathematics := 95
  let physics := 82
  let chemistry := 87
  let biology := 92
  let weight_english := 0.20
  let weight_mathematics := 0.25
  let weight_physics := 0.15
  let weight_chemistry := 0.25
  let weight_biology := 0.15
  (english * weight_english) + (mathematics * weight_mathematics) +
  (physics * weight_physics) + (chemistry * weight_chemistry) +
  (biology * weight_biology) = 90.8 :=
by
  sorry

end dacid_weighted_average_l167_167849


namespace total_amount_l167_167557

theorem total_amount (A B C T : ℝ)
  (h1 : A = 1 / 4 * (B + C))
  (h2 : B = 3 / 5 * (A + C))
  (h3 : A = 20) :
  T = A + B + C → T = 100 := by
  sorry

end total_amount_l167_167557


namespace solution_for_a_if_fa_eq_a_l167_167080

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

theorem solution_for_a_if_fa_eq_a (a : ℝ) (h : f a = a) : a = -1 :=
sorry

end solution_for_a_if_fa_eq_a_l167_167080


namespace g_two_eq_one_l167_167509

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0

theorem g_two_eq_one : g 2 = 1 := by
  sorry

end g_two_eq_one_l167_167509


namespace no_solution_value_of_m_l167_167868

theorem no_solution_value_of_m (m : ℤ) : ¬ ∃ x : ℤ, x ≠ 3 ∧ (x - 5) * (x - 3) = (m * (x - 3) + 2 * (x - 3) * (x - 3)) → m = -2 :=
by
  sorry

end no_solution_value_of_m_l167_167868


namespace arithmetic_sequence_geometric_sequence_l167_167206

-- Problem 1
theorem arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) (Sₙ : ℝ) 
  (h₁ : a₁ = 3 / 2) (h₂ : d = -1 / 2) (h₃ : Sₙ = -15) :
  n = 12 ∧ (a₁ + (n - 1) * d) = -4 := 
sorry

-- Problem 2
theorem geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) (aₙ Sₙ : ℝ) 
  (h₁ : q = 2) (h₂ : aₙ = 96) (h₃ : Sₙ = 189) :
  a₁ = 3 ∧ n = 6 := 
sorry

end arithmetic_sequence_geometric_sequence_l167_167206


namespace eval_expression_l167_167166

theorem eval_expression :
  -((18 / 3 * 8) - 80 + (4 ^ 2 * 2)) = 0 :=
by
  sorry

end eval_expression_l167_167166


namespace pq_iff_cond_l167_167991

def p (a : ℝ) := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem pq_iff_cond (a : ℝ) : (p a ∧ q a) ↔ (a ≤ -2 ∨ a = 1) := 
by
  sorry

end pq_iff_cond_l167_167991


namespace divisible_by_10_l167_167230

theorem divisible_by_10 : (11 * 21 * 31 * 41 * 51 - 1) % 10 = 0 := by
  sorry

end divisible_by_10_l167_167230


namespace net_percentage_error_l167_167462

noncomputable section
def calculate_percentage_error (true_side excess_error deficit_error : ℝ) : ℝ :=
  let measured_side1 := true_side * (1 + excess_error / 100)
  let measured_side2 := measured_side1 * (1 - deficit_error / 100)
  let true_area := true_side ^ 2
  let calculated_area := measured_side2 * true_side
  let percentage_error := ((true_area - calculated_area) / true_area) * 100
  percentage_error

theorem net_percentage_error 
  (S : ℝ) (h1 : S > 0) : calculate_percentage_error S 3 (-4) = 1.12 := by
  sorry

end net_percentage_error_l167_167462


namespace age_difference_l167_167803

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l167_167803


namespace systematic_sampling_questionnaire_B_count_l167_167633

theorem systematic_sampling_questionnaire_B_count (n : ℕ) (N : ℕ) (first_random : ℕ) (range_A_start range_A_end range_B_start range_B_end : ℕ) 
  (h1 : n = 32) (h2 : N = 960) (h3 : first_random = 9) (h4 : range_A_start = 1) (h5 : range_A_end = 460) 
  (h6 : range_B_start = 461) (h7 : range_B_end = 761) :
  ∃ count : ℕ, count = 10 := by
  sorry

end systematic_sampling_questionnaire_B_count_l167_167633


namespace julia_age_after_10_years_l167_167308

-- Define the conditions
def Justin_age : Nat := 26
def Jessica_older_by : Nat := 6
def James_older_by : Nat := 7
def Julia_younger_by : Nat := 8
def years_after : Nat := 10

-- Define the ages now
def Jessica_age_now : Nat := Justin_age + Jessica_older_by
def James_age_now : Nat := Jessica_age_now + James_older_by
def Julia_age_now : Nat := Justin_age - Julia_younger_by

-- Prove that Julia's age after 10 years is 28
theorem julia_age_after_10_years : Julia_age_now + years_after = 28 := by
  sorry

end julia_age_after_10_years_l167_167308


namespace symmetric_axis_of_g_l167_167905

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 6))

theorem symmetric_axis_of_g :
  ∃ k : ℤ, (∃ x : ℝ, g x = 2 * Real.sin (k * Real.pi + (Real.pi / 2)) ∧ x = (k * Real.pi) / 2 + (Real.pi / 3)) :=
sorry

end symmetric_axis_of_g_l167_167905


namespace min_large_trucks_needed_l167_167298

-- Define the parameters for the problem
def total_fruit : ℕ := 134
def load_large_truck : ℕ := 15
def load_small_truck : ℕ := 7

-- Define the main theorem to be proved
theorem min_large_trucks_needed :
  ∃ (n : ℕ), n = 8 ∧ (total_fruit = n * load_large_truck + 2 * load_small_truck) :=
by sorry

end min_large_trucks_needed_l167_167298


namespace wanda_walks_days_per_week_l167_167651

theorem wanda_walks_days_per_week 
  (daily_distance : ℝ) (weekly_distance : ℝ) (weeks : ℕ) (total_distance : ℝ) 
  (h_daily_walk: daily_distance = 2) 
  (h_total_walk: total_distance = 40) 
  (h_weeks: weeks = 4) : 
  ∃ d : ℕ, (d * daily_distance * weeks = total_distance) ∧ (d = 5) := 
by 
  sorry

end wanda_walks_days_per_week_l167_167651


namespace maximum_elevation_l167_167220

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 200 * t - 17 * t^2 - 3 * t^3

-- State that the maximum elevation is 368.1 feet
theorem maximum_elevation :
  ∃ t : ℝ, t > 0 ∧ (∀ t' : ℝ, t' ≠ t → elevation t ≤ elevation t') ∧ elevation t = 368.1 :=
by
  sorry

end maximum_elevation_l167_167220


namespace sum_of_digits_square_1111111_l167_167340

noncomputable def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_square_1111111 :
  sum_of_digits (1111111 * 1111111) = 49 :=
sorry

end sum_of_digits_square_1111111_l167_167340


namespace floor_div_eq_floor_div_l167_167820

theorem floor_div_eq_floor_div
  (a : ℝ) (n : ℤ) (ha_pos : 0 < a) :
  (⌊⌊a⌋ / n⌋ : ℤ) = ⌊a / n⌋ := 
sorry

end floor_div_eq_floor_div_l167_167820


namespace lemon_pie_degrees_l167_167422

noncomputable def num_students := 45
noncomputable def chocolate_pie_students := 15
noncomputable def apple_pie_students := 9
noncomputable def blueberry_pie_students := 9
noncomputable def other_pie_students := num_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
noncomputable def each_remaining_pie_students := other_pie_students / 3
noncomputable def fraction_lemon_pie := each_remaining_pie_students / num_students
noncomputable def degrees_lemon_pie := fraction_lemon_pie * 360

theorem lemon_pie_degrees : degrees_lemon_pie = 32 :=
sorry

end lemon_pie_degrees_l167_167422


namespace sin_double_angle_identity_l167_167431

open Real

theorem sin_double_angle_identity (α : ℝ) (h : sin (α - π / 4) = 3 / 5) : sin (2 * α) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l167_167431


namespace carpet_area_l167_167168

theorem carpet_area (length_ft : ℕ) (width_ft : ℕ) (ft_per_yd : ℕ) (A_y : ℕ) 
  (h_length : length_ft = 15) (h_width : width_ft = 12) (h_ft_per_yd : ft_per_yd = 9) :
  A_y = (length_ft * width_ft) / ft_per_yd := 
by sorry

#check carpet_area

end carpet_area_l167_167168


namespace divisibility_condition_l167_167614

theorem divisibility_condition (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1) ∣ ((a + 1)^n) ↔ (a = 1 ∧ 1 ≤ m ∧ 1 ≤ n) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n) := 
by 
  sorry

end divisibility_condition_l167_167614


namespace imaginary_part_of_complex_l167_167327

open Complex

theorem imaginary_part_of_complex (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z = (3 - 2 * i^3) / (1 + i)) : z.im = -1 / 2 :=
by {
  -- Proof would go here
  sorry
}

end imaginary_part_of_complex_l167_167327


namespace q_minus_r_max_value_l167_167033

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), q > 99 ∧ q < 1000 ∧ r > 99 ∧ r < 1000 ∧ 
    q = 100 * (q / 100) + 10 * ((q / 10) % 10) + (q % 10) ∧ 
    r = 100 * (q % 10) + 10 * ((q / 10) % 10) + (q / 100) ∧ 
    q - r = 297 :=
by sorry

end q_minus_r_max_value_l167_167033


namespace mr_ray_customers_without_fish_l167_167630

def mr_ray_num_customers_without_fish
  (total_customers : ℕ)
  (total_tuna_weight : ℕ)
  (specific_customers_30lb : ℕ)
  (specific_weight_30lb : ℕ)
  (specific_customers_20lb : ℕ)
  (specific_weight_20lb : ℕ)
  (weight_per_customer : ℕ)
  (remaining_tuna_weight : ℕ)
  (num_customers_served_with_remaining_tuna : ℕ)
  (total_satisfied_customers : ℕ) : ℕ :=
  total_customers - total_satisfied_customers

theorem mr_ray_customers_without_fish :
  mr_ray_num_customers_without_fish 100 2000 10 30 15 20 25 1400 56 81 = 19 :=
by 
  sorry

end mr_ray_customers_without_fish_l167_167630


namespace find_p_q_coprime_sum_l167_167787

theorem find_p_q_coprime_sum (x y n m: ℕ) (h_sum: x + y = 30)
  (h_prob: ((n/x) * (n-1)/(x-1) * (n-2)/(x-2)) * ((m/y) * (m-1)/(y-1) * (m-2)/(y-2)) = 18/25)
  : ∃ p q : ℕ, p.gcd q = 1 ∧ p + q = 1006 :=
by
  sorry

end find_p_q_coprime_sum_l167_167787


namespace probability_of_opposite_middle_vertex_l167_167727

noncomputable def ant_moves_to_opposite_middle_vertex_prob : ℚ := 1 / 2

-- Specification of the problem conditions
structure Octahedron :=
  (middle_vertices : Finset ℕ) -- Assume some identification of middle vertices
  (adjacent_vertices : ℕ → Finset ℕ) -- Function mapping a vertex to its adjacent vertices
  (is_middle_vertex : ℕ → Prop) -- Predicate to check if a vertex is a middle vertex
  (is_top_or_bottom_vertex : ℕ → Prop) -- Predicate to check if a vertex is a top or bottom vertex
  (start_vertex : ℕ)

variables (O : Octahedron)

-- Main theorem statement
theorem probability_of_opposite_middle_vertex :
  ∃ A B : ℕ, A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B) →
  (∀ (A B : ℕ), (A ∈ O.adjacent_vertices O.start_vertex ∧ B ∈ O.adjacent_vertices A ∧ B ≠ O.start_vertex ∧ (∃ x ∈ O.middle_vertices, x = B)) →
    ant_moves_to_opposite_middle_vertex_prob = 1 / 2) := sorry

end probability_of_opposite_middle_vertex_l167_167727


namespace proof_problem_l167_167852

noncomputable def problem : Prop :=
  ∃ (m n l : Type) (α β : Type) 
    (is_line : ∀ x, x = m ∨ x = n ∨ x = l)
    (is_plane : ∀ x, x = α ∨ x = β)
    (perpendicular : ∀ (l α : Type), Prop)
    (parallel : ∀ (l α : Type), Prop)
    (belongs_to : ∀ (l α : Type), Prop),
    (parallel l α → ∃ l', parallel l' α ∧ parallel l l') ∧
    (perpendicular m α ∧ perpendicular m β → parallel α β)

theorem proof_problem : problem :=
sorry

end proof_problem_l167_167852


namespace factor_expression_l167_167745

theorem factor_expression (z : ℂ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) :=
sorry

end factor_expression_l167_167745


namespace sum_of_sequence_l167_167040

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, a n = (-1 : ℤ)^(n+1) * (2*n - 1)) →
  (S 0 = 0) →
  (∀ n, S (n+1) = S n + a (n+1)) →
  (∀ n, S (n+1) = (-1 : ℤ)^(n+1) * (n+1)) :=
by
  intros h_a h_S0 h_S
  sorry

end sum_of_sequence_l167_167040


namespace rectangle_area_l167_167883

theorem rectangle_area :
  ∀ (width length : ℝ), (length = 3 * width) → (width = 5) → (length * width = 75) :=
by
  intros width length h1 h2
  rw [h2, h1]
  sorry

end rectangle_area_l167_167883


namespace oak_trees_problem_l167_167590

theorem oak_trees_problem (c t n : ℕ) 
  (h1 : c = 9) 
  (h2 : t = 11) 
  (h3 : t = c + n) 
  : n = 2 := 
by 
  sorry

end oak_trees_problem_l167_167590


namespace roots_difference_squared_l167_167366

-- Defining the solutions to the quadratic equation
def quadratic_equation_roots (a b : ℚ) : Prop :=
  (2 * a^2 - 7 * a + 6 = 0) ∧ (2 * b^2 - 7 * b + 6 = 0)

-- The main theorem we aim to prove
theorem roots_difference_squared (a b : ℚ) (h : quadratic_equation_roots a b) :
    (a - b)^2 = 1 / 4 := 
  sorry

end roots_difference_squared_l167_167366


namespace major_axis_length_l167_167038

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (hr : r = 2) 
  (h_minor : minor_axis = 2 * r)
  (h_major : major_axis = 1.25 * minor_axis) :
  major_axis = 5 :=
by
  sorry

end major_axis_length_l167_167038


namespace find_number_of_dimes_l167_167415

def total_value (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies * 1 + nickels * 5 + dimes * 10 + quarters * 25 + half_dollars * 50

def number_of_coins (pennies nickels dimes quarters half_dollars : Nat) : Nat :=
  pennies + nickels + dimes + quarters + half_dollars

theorem find_number_of_dimes
  (pennies nickels dimes quarters half_dollars : Nat)
  (h_value : total_value pennies nickels dimes quarters half_dollars = 163)
  (h_coins : number_of_coins pennies nickels dimes quarters half_dollars = 13)
  (h_penny : 1 ≤ pennies)
  (h_nickel : 1 ≤ nickels)
  (h_dime : 1 ≤ dimes)
  (h_quarter : 1 ≤ quarters)
  (h_half_dollar : 1 ≤ half_dollars) :
  dimes = 3 :=
sorry

end find_number_of_dimes_l167_167415


namespace pow_five_2010_mod_seven_l167_167347

theorem pow_five_2010_mod_seven :
  (5 ^ 2010) % 7 = 1 :=
by
  have h : (5 ^ 6) % 7 = 1 := sorry
  sorry

end pow_five_2010_mod_seven_l167_167347


namespace length_GH_l167_167058

theorem length_GH (AB BC : ℝ) (hAB : AB = 10) (hBC : BC = 5) (DG DH GH : ℝ)
  (hDG : DG = DH) (hArea_DGH : 1 / 2 * DG * DH = 1 / 5 * (AB * BC)) :
  GH = 2 * Real.sqrt 10 :=
by
  sorry

end length_GH_l167_167058


namespace probability_sum_eight_l167_167524

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 5

theorem probability_sum_eight :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_eight_l167_167524


namespace cos_identity_l167_167501

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x, (Real.sqrt 3) / 2)
  let b := (Real.sin (x - Real.pi / 3), 1)
  a.1 * b.1 + a.2 * b.2

theorem cos_identity (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (5 * Real.pi / 12) (2 * Real.pi / 3))
  (hf : f x0 = 4 / 5) :
  Real.cos (2 * x0 - Real.pi / 12) = -7 * Real.sqrt 2 / 10 :=
sorry

end cos_identity_l167_167501


namespace StockPriceAdjustment_l167_167998

theorem StockPriceAdjustment (P₀ P₁ P₂ P₃ P₄ : ℝ) (january_increase february_decrease march_increase : ℝ) :
  P₀ = 150 →
  january_increase = 0.10 →
  february_decrease = 0.15 →
  march_increase = 0.30 →
  P₁ = P₀ * (1 + january_increase) →
  P₂ = P₁ * (1 - february_decrease) →
  P₃ = P₂ * (1 + march_increase) →
  142.5 <= P₃ * (1 - 0.17) ∧ P₃ * (1 - 0.17) <= 157.5 :=
by
  intros hP₀ hJanuaryIncrease hFebruaryDecrease hMarchIncrease hP₁ hP₂ hP₃
  sorry

end StockPriceAdjustment_l167_167998


namespace employees_in_room_l167_167036

-- Define variables
variables (E : ℝ) (M : ℝ) (L : ℝ)

-- Given conditions
def condition1 : Prop := M = 0.99 * E
def condition2 : Prop := (M - L) / E = 0.98
def condition3 : Prop := L = 99.99999999999991

-- Prove statement
theorem employees_in_room (h1 : condition1 E M) (h2 : condition2 E M L) (h3 : condition3 L) : E = 10000 :=
by
  sorry

end employees_in_room_l167_167036


namespace original_price_of_car_l167_167781

-- Let P be the original price of the car
variable (P : ℝ)

-- Condition: The car's value is reduced by 30%
-- Condition: The car's current value is $2800, which means 70% of the original price
def car_current_value_reduced (P : ℝ) : Prop :=
  0.70 * P = 2800

-- Theorem: Prove that the original price of the car is $4000
theorem original_price_of_car (P : ℝ) (h : car_current_value_reduced P) : P = 4000 := by
  sorry

end original_price_of_car_l167_167781


namespace total_ticket_sales_is_48_l167_167401

noncomputable def ticket_sales (total_revenue : ℕ) (price_per_ticket : ℕ) (discount_1 : ℕ) (discount_2 : ℕ) : ℕ :=
  let number_first_batch := 10
  let number_second_batch := 20
  let revenue_first_batch := number_first_batch * (price_per_ticket - (price_per_ticket * discount_1 / 100))
  let revenue_second_batch := number_second_batch * (price_per_ticket - (price_per_ticket * discount_2 / 100))
  let revenue_full_price := total_revenue - (revenue_first_batch + revenue_second_batch)
  let number_full_price_tickets := revenue_full_price / price_per_ticket
  number_first_batch + number_second_batch + number_full_price_tickets

theorem total_ticket_sales_is_48 : ticket_sales 820 20 40 15 = 48 :=
by
  sorry

end total_ticket_sales_is_48_l167_167401


namespace age_difference_is_18_l167_167160

variable (A B C : ℤ)
variable (h1 : A + B > B + C)
variable (h2 : C = A - 18)

theorem age_difference_is_18 : (A + B) - (B + C) = 18 :=
by
  sorry

end age_difference_is_18_l167_167160


namespace pages_left_to_read_l167_167875

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end pages_left_to_read_l167_167875


namespace sphere_surface_area_l167_167806

theorem sphere_surface_area
  (a : ℝ)
  (expansion : (1 - 2 * 1 : ℝ)^6 = a)
  (a_value : a = 1) :
  4 * Real.pi * ((Real.sqrt (2^2 + 3^2 + a^2) / 2)^2) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l167_167806


namespace line_not_in_fourth_quadrant_l167_167824

-- Let the line be defined as y = 3x + 2
def line_eq (x : ℝ) : ℝ := 3 * x + 2

-- The Fourth quadrant is defined by x > 0 and y < 0
def in_fourth_quadrant (x : ℝ) (y : ℝ) : Prop := x > 0 ∧ y < 0

-- Prove that the line does not intersect the Fourth quadrant
theorem line_not_in_fourth_quadrant : ¬ (∃ x : ℝ, in_fourth_quadrant x (line_eq x)) :=
by
  -- Proof goes here (abstracted)
  sorry

end line_not_in_fourth_quadrant_l167_167824


namespace sufficient_condition_for_inequality_l167_167746

theorem sufficient_condition_for_inequality (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 :=
by
  sorry

end sufficient_condition_for_inequality_l167_167746


namespace find_a_extreme_value_at_2_l167_167802

noncomputable def f (x : ℝ) (a : ℝ) := (2 / 3) * x^3 + a * x^2

theorem find_a_extreme_value_at_2 (a : ℝ) :
  (∀ x : ℝ, x ≠ 2 -> 0 = 2 * x^2 + 2 * a * x) ->
  (2 * 2^2 + 2 * a * 2 = 0) ->
  a = -2 :=
by {
  sorry
}

end find_a_extreme_value_at_2_l167_167802


namespace book_original_selling_price_l167_167639

theorem book_original_selling_price (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.1 * CP)
  (h3 : SP2 = 990) : 
  SP1 = 810 :=
by
  sorry

end book_original_selling_price_l167_167639


namespace disjoint_subsets_exist_l167_167922

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) 
  (A : Fin (n + 1) → Set (Fin n)) (hA : ∀ i : Fin (n + 1), A i ≠ ∅) :
  ∃ (I J : Finset (Fin (n + 1))), I ≠ ∅ ∧ J ≠ ∅ ∧ Disjoint I J ∧ 
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) :=
sorry

end disjoint_subsets_exist_l167_167922


namespace range_of_f_l167_167054

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, y = f x ∧ (y ≥ -3 / 2 ∧ y ≤ 3) :=
by {
  sorry
}

end range_of_f_l167_167054


namespace min_value_xy_l167_167989

theorem min_value_xy (x y : ℝ) (h : x * y = 1) : x^2 + 4 * y^2 ≥ 4 := by
  sorry

end min_value_xy_l167_167989


namespace find_m_collinear_l167_167188

theorem find_m_collinear (m : ℝ) 
    (a : ℝ × ℝ := (m + 3, 2)) 
    (b : ℝ × ℝ := (m, 1)) 
    (collinear : a.1 * 1 - 2 * b.1 = 0) : 
    m = 3 :=
by {
    sorry
}

end find_m_collinear_l167_167188


namespace positive_difference_is_9107_03_l167_167925

noncomputable def Cedric_balance : ℝ :=
  15000 * (1 + 0.06) ^ 20

noncomputable def Daniel_balance : ℝ :=
  15000 * (1 + 20 * 0.08)

noncomputable def Elaine_balance : ℝ :=
  15000 * (1 + 0.055 / 2) ^ 40

-- Positive difference between highest and lowest balances.
noncomputable def positive_difference : ℝ :=
  let highest := max Cedric_balance (max Daniel_balance Elaine_balance)
  let lowest := min Cedric_balance (min Daniel_balance Elaine_balance)
  highest - lowest

theorem positive_difference_is_9107_03 :
  positive_difference = 9107.03 := by
  sorry

end positive_difference_is_9107_03_l167_167925


namespace least_number_remainder_l167_167112

open Nat

theorem least_number_remainder (n : ℕ) :
  (n ≡ 4 [MOD 5]) →
  (n ≡ 4 [MOD 6]) →
  (n ≡ 4 [MOD 9]) →
  (n ≡ 4 [MOD 12]) →
  n = 184 :=
by
  intros h1 h2 h3 h4
  sorry

end least_number_remainder_l167_167112


namespace coin_stack_height_l167_167913

def alpha_thickness : ℝ := 1.25
def beta_thickness : ℝ := 2.00
def gamma_thickness : ℝ := 0.90
def delta_thickness : ℝ := 1.60
def stack_height : ℝ := 18.00

theorem coin_stack_height :
  (∃ n : ℕ, stack_height = n * beta_thickness) ∨ (∃ n : ℕ, stack_height = n * gamma_thickness) :=
sorry

end coin_stack_height_l167_167913


namespace parallel_slope_l167_167538

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l167_167538


namespace part_one_part_two_l167_167134

-- Definitions based on the conditions
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 - a}

-- Prove intersection A ∩ B = (0, 1)
theorem part_one : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

-- Prove range of a when A ∪ C = A
theorem part_two (a : ℝ) (h : A ∪ C a = A) : 1 < a := by
  sorry

end part_one_part_two_l167_167134


namespace total_adults_wearing_hats_l167_167894

theorem total_adults_wearing_hats (total_adults : ℕ) (men_percentage : ℝ) (men_hats_percentage : ℝ) 
  (women_hats_percentage : ℝ) (total_men_wearing_hats : ℕ) (total_women_wearing_hats : ℕ) : 
  (total_adults = 1200) ∧ (men_percentage = 0.60) ∧ (men_hats_percentage = 0.15) 
  ∧ (women_hats_percentage = 0.10)
     → total_men_wearing_hats + total_women_wearing_hats = 156 :=
by
  -- Definitions
  let total_men := total_adults * men_percentage
  let total_women := total_adults - total_men
  let men_wearing_hats := total_men * men_hats_percentage
  let women_wearing_hats := total_women * women_hats_percentage
  sorry

end total_adults_wearing_hats_l167_167894


namespace smallest_positive_integer_solution_l167_167402

theorem smallest_positive_integer_solution (x : ℤ) 
  (hx : |5 * x - 8| = 47) : x = 11 :=
by
  sorry

end smallest_positive_integer_solution_l167_167402


namespace radius_large_circle_l167_167498

/-- Let R be the radius of the large circle. Assume three circles of radius 2 are externally 
tangent to each other. Two of these circles are internally tangent to the larger circle, 
and the third circle is tangent to the larger circle both internally and externally. 
Prove that the radius of the large circle is 4 + 2 * sqrt 3. -/
theorem radius_large_circle (R : ℝ)
  (h1 : ∃ (C1 C2 C3 : ℝ × ℝ), 
    dist C1 C2 = 4 ∧ dist C2 C3 = 4 ∧ dist C3 C1 = 4 ∧ 
    (∃ (O : ℝ × ℝ), 
      (dist O C1 = R - 2) ∧ 
      (dist O C2 = R - 2) ∧ 
      (dist O C3 = R + 2) ∧ 
      (dist C1 C2 = 4) ∧ (dist C2 C3 = 4) ∧ (dist C3 C1 = 4))):
  R = 4 + 2 * Real.sqrt 3 := 
sorry

end radius_large_circle_l167_167498


namespace total_pages_in_book_l167_167482

def pagesReadMonday := 23
def pagesReadTuesday := 38
def pagesReadWednesday := 61
def pagesReadThursday := 12
def pagesReadFriday := 2 * pagesReadThursday

def totalPagesRead := pagesReadMonday + pagesReadTuesday + pagesReadWednesday + pagesReadThursday + pagesReadFriday

theorem total_pages_in_book :
  totalPagesRead = 158 :=
by
  sorry

end total_pages_in_book_l167_167482


namespace expected_value_of_die_l167_167345

noncomputable def expected_value : ℚ :=
  (1/14) * 1 + (1/14) * 2 + (1/14) * 3 + (1/14) * 4 + (1/14) * 5 + (1/14) * 6 + (1/14) * 7 + (3/8) * 8

theorem expected_value_of_die : expected_value = 5 :=
by
  sorry

end expected_value_of_die_l167_167345


namespace sugar_for_cake_l167_167784

-- Definitions of given values
def sugar_for_frosting : ℝ := 0.6
def total_sugar_required : ℝ := 0.8

-- Proof statement
theorem sugar_for_cake : (total_sugar_required - sugar_for_frosting) = 0.2 :=
by
  sorry

end sugar_for_cake_l167_167784


namespace full_time_employees_l167_167055

theorem full_time_employees (total_employees part_time_employees number_full_time_employees : ℕ)
  (h1 : total_employees = 65134)
  (h2 : part_time_employees = 2041)
  (h3 : number_full_time_employees = total_employees - part_time_employees)
  : number_full_time_employees = 63093 :=
by {
  sorry
}

end full_time_employees_l167_167055


namespace inequality_solution_l167_167243

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 4) > 2 / x + 12 / 5 ↔ x < 0 :=
by
  sorry

end inequality_solution_l167_167243


namespace triangle_inradius_exradii_relation_l167_167612

theorem triangle_inradius_exradii_relation
  (a b c : ℝ) (S : ℝ) (r r_a r_b r_c : ℝ)
  (h_inradius : S = (1/2) * r * (a + b + c))
  (h_exradii_a : r_a = 2 * S / (b + c - a))
  (h_exradii_b : r_b = 2 * S / (c + a - b))
  (h_exradii_c : r_c = 2 * S / (a + b - c))
  (h_area : S = (1/2) * (a * r_a + b * r_b + c * r_c - a * r - b * r - c * r)) :
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  by sorry

end triangle_inradius_exradii_relation_l167_167612


namespace original_cone_volume_l167_167822

theorem original_cone_volume
  (H R h r : ℝ)
  (Vcylinder : ℝ) (Vfrustum : ℝ)
  (cylinder_volume : Vcylinder = π * r^2 * h)
  (frustum_volume : Vfrustum = (1 / 3) * π * (R^2 + R * r + r^2) * (H - h))
  (Vcylinder_value : Vcylinder = 9)
  (Vfrustum_value : Vfrustum = 63) :
  (1 / 3) * π * R^2 * H = 64 :=
by
  sorry

end original_cone_volume_l167_167822


namespace proof_by_contradiction_example_l167_167961

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l167_167961


namespace inequality_x4_y4_l167_167103

theorem inequality_x4_y4 (x y : ℝ) : x^4 + y^4 + 8 ≥ 8 * x * y := 
by {
  sorry
}

end inequality_x4_y4_l167_167103


namespace original_population_before_changes_l167_167458

open Nat

def halved_population (p: ℕ) (years: ℕ) : ℕ := p / (2^years)

theorem original_population_before_changes (P_init P_final : ℕ)
    (new_people : ℕ) (people_moved_out : ℕ) :
    new_people = 100 →
    people_moved_out = 400 →
    ∀ years, (years = 4 → halved_population P_final years = 60) →
    ∃ P_before_change, P_before_change = 780 ∧
    P_init = P_before_change + new_people - people_moved_out ∧
    halved_population P_init years = P_final := 
by
  intros
  sorry

end original_population_before_changes_l167_167458


namespace xiao_ming_returns_and_distance_is_correct_l167_167687

theorem xiao_ming_returns_and_distance_is_correct :
  ∀ (walk_distance : ℝ) (turn_angle : ℝ), 
  walk_distance = 5 ∧ turn_angle = 20 → 
  (∃ n : ℕ, (360 % turn_angle = 0) ∧ n = 360 / turn_angle ∧ walk_distance * n = 90) :=
by
  sorry

end xiao_ming_returns_and_distance_is_correct_l167_167687


namespace min_value_4x_3y_l167_167750

theorem min_value_4x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y = 5 * x * y) : 
  4 * x + 3 * y ≥ 5 :=
sorry

end min_value_4x_3y_l167_167750


namespace value_of_b_minus_a_l167_167699

open Real

def condition (a b : ℝ) : Prop := 
  abs a = 3 ∧ abs b = 2 ∧ a + b > 0

theorem value_of_b_minus_a (a b : ℝ) (h : condition a b) :
  b - a = -1 ∨ b - a = -5 :=
  sorry

end value_of_b_minus_a_l167_167699


namespace algebra_expression_value_l167_167578

theorem algebra_expression_value (a b : ℝ)
  (h1 : |a + 2| = 0)
  (h2 : (b - 5 / 2) ^ 2 = 0) : (2 * a + 3 * b) * (2 * b - 3 * a) = 26 := by
sorry

end algebra_expression_value_l167_167578


namespace unique_solution_of_system_l167_167705

theorem unique_solution_of_system (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : x * (x + y + z) = 26) (h2 : y * (x + y + z) = 27) (h3 : z * (x + y + z) = 28) :
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 :=
by
  sorry

end unique_solution_of_system_l167_167705


namespace exists_same_color_rectangle_l167_167418

open Finset

-- Define the grid size
def gridSize : ℕ := 12

-- Define the type of colors
inductive Color
| red
| white
| blue

-- Define a point in the grid
structure Point :=
(x : ℕ)
(y : ℕ)
(hx : x ≥ 1 ∧ x ≤ gridSize)
(hy : y ≥ 1 ∧ y ≤ gridSize)

-- Assume a coloring function
def color (p : Point) : Color := sorry

-- The theorem statement
theorem exists_same_color_rectangle :
  ∃ (p1 p2 p3 p4 : Point),
    p1.x = p2.x ∧ p3.x = p4.x ∧
    p1.y = p3.y ∧ p2.y = p4.y ∧
    color p1 = color p2 ∧
    color p1 = color p3 ∧
    color p1 = color p4 :=
sorry

end exists_same_color_rectangle_l167_167418


namespace exp_decreasing_function_range_l167_167634

theorem exp_decreasing_function_range (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1) (h_f : a^(x+1) ≥ 1) : x ≤ -1 :=
sorry

end exp_decreasing_function_range_l167_167634


namespace remainder_of_1234567_div_257_l167_167474

theorem remainder_of_1234567_div_257 : 1234567 % 257 = 123 := by
  sorry

end remainder_of_1234567_div_257_l167_167474


namespace fraction_value_l167_167305

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l167_167305


namespace value_of_t_l167_167888

theorem value_of_t (t : ℝ) (x y : ℝ) (h1 : x = 1 - 2 * t) (h2 : y = 2 * t - 2) (h3 : x = y) : t = 3 / 4 := 
by
  sorry

end value_of_t_l167_167888


namespace set_intersection_l167_167623

def S : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def T : Set ℝ := {x | x > 1}
def result : Set ℝ := {x | x ≥ 3 ∨ (1 < x ∧ x ≤ 2)}

theorem set_intersection (x : ℝ) : x ∈ (S ∩ T) ↔ x ∈ result := by
  sorry

end set_intersection_l167_167623


namespace simplify_exponents_l167_167265

theorem simplify_exponents : (10^0.5) * (10^0.3) * (10^0.2) * (10^0.1) * (10^0.9) = 100 := 
by 
  sorry

end simplify_exponents_l167_167265


namespace most_likely_outcome_l167_167536

-- Defining the conditions
def equally_likely (n : ℕ) (k : ℕ) := (Nat.choose n k) * (1 / 2)^n

-- Defining the problem statement
theorem most_likely_outcome :
  (equally_likely 5 3 = 5 / 16 ∧ equally_likely 5 2 = 5 / 16) :=
sorry

end most_likely_outcome_l167_167536


namespace area_of_square_plot_l167_167314

-- Defining the given conditions and question in Lean 4
theorem area_of_square_plot 
  (cost_per_foot : ℕ := 58)
  (total_cost : ℕ := 2784) :
  ∃ (s : ℕ), (4 * s * cost_per_foot = total_cost) ∧ (s * s = 144) :=
by
  sorry

end area_of_square_plot_l167_167314


namespace common_difference_ne_3_l167_167938

theorem common_difference_ne_3 
  (d : ℕ) (hd_pos : d > 0) 
  (exists_n : ∃ n : ℕ, 81 = 1 + (n - 1) * d) : 
  d ≠ 3 :=
by sorry

end common_difference_ne_3_l167_167938


namespace cost_of_one_shirt_l167_167297

theorem cost_of_one_shirt
  (J S : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 76) :
  S = 18 :=
by
  sorry

end cost_of_one_shirt_l167_167297


namespace inequality_solution_l167_167680

theorem inequality_solution (x : ℝ) :
  ( (x^2 + 3*x + 3) > 0 ) → ( ((x^2 + 3*x + 3)^(5*x^3 - 3*x^2)) ≤ ((x^2 + 3*x + 3)^(3*x^3 + 5*x)) )
  ↔ ( x ∈ (Set.Iic (-2) ∪ ({-1} : Set ℝ) ∪ Set.Icc 0 (5/2)) ) :=
by
  sorry

end inequality_solution_l167_167680


namespace no_such_function_exists_l167_167137

open Set

theorem no_such_function_exists
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → y > x → f y > (y - x) * f x ^ 2) :
  False :=
sorry

end no_such_function_exists_l167_167137


namespace twice_x_minus_three_lt_zero_l167_167769

theorem twice_x_minus_three_lt_zero (x : ℝ) : (2 * x - 3 < 0) ↔ (2 * x < 3) :=
by
  sorry

end twice_x_minus_three_lt_zero_l167_167769


namespace extreme_value_at_3_increasing_on_interval_l167_167053

def f (a : ℝ) (x : ℝ) : ℝ := 2*x^3 - 3*(a+1)*x^2 + 6*a*x + 8

theorem extreme_value_at_3 (a : ℝ) : (∃ x, x = 3 ∧ 6*x^2 - 6*(a+1)*x + 6*a = 0) → a = 3 :=
by
  sorry

theorem increasing_on_interval (a : ℝ) : (∀ x, x < 0 → 6*(x-a)*(x-1) > 0) → 0 ≤ a :=
by
  sorry

end extreme_value_at_3_increasing_on_interval_l167_167053


namespace initial_black_pieces_is_118_l167_167400

open Nat

-- Define the initial conditions and variables
variables (b w n : ℕ)

-- Hypotheses based on the conditions
axiom h1 : b = 2 * w
axiom h2 : w - 2 * n = 1
axiom h3 : b - 3 * n = 31

-- Goal to prove the initial number of black pieces were 118
theorem initial_black_pieces_is_118 : b = 118 :=
by 
  -- We only state the theorem, proof will be added as sorry
  sorry

end initial_black_pieces_is_118_l167_167400


namespace classroom_gpa_l167_167113

theorem classroom_gpa (n : ℕ) (x : ℝ)
  (h1 : n > 0)
  (h2 : (1/3 : ℝ) * n * 45 + (2/3 : ℝ) * n * x = n * 55) : x = 60 :=
by
  sorry

end classroom_gpa_l167_167113


namespace ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l167_167109

theorem ten_times_ten_thousand : 10 * 10000 = 100000 :=
by sorry

theorem ten_times_one_million : 10 * 1000000 = 10000000 :=
by sorry

theorem ten_times_ten_million : 10 * 10000000 = 100000000 :=
by sorry

theorem tens_of_thousands_in_hundred_million : 100000000 / 10000 = 10000 :=
by sorry

end ten_times_ten_thousand_ten_times_one_million_ten_times_ten_million_tens_of_thousands_in_hundred_million_l167_167109


namespace cost_per_lb_of_mixture_l167_167586

def millet_weight : ℝ := 100
def millet_cost_per_lb : ℝ := 0.60
def sunflower_weight : ℝ := 25
def sunflower_cost_per_lb : ℝ := 1.10

theorem cost_per_lb_of_mixture :
  let millet_weight := 100
  let millet_cost_per_lb := 0.60
  let sunflower_weight := 25
  let sunflower_cost_per_lb := 1.10
  let millet_total_cost := millet_weight * millet_cost_per_lb
  let sunflower_total_cost := sunflower_weight * sunflower_cost_per_lb
  let total_cost := millet_total_cost + sunflower_total_cost
  let total_weight := millet_weight + sunflower_weight
  (total_cost / total_weight) = 0.70 :=
by
  sorry

end cost_per_lb_of_mixture_l167_167586


namespace harry_worked_34_hours_l167_167437

noncomputable def Harry_hours_worked (x : ℝ) : ℝ := 34

theorem harry_worked_34_hours (x : ℝ)
  (H : ℝ) (James_hours : ℝ) (Harry_pay James_pay: ℝ) 
  (h1 : Harry_pay = 18 * x + 1.5 * x * (H - 18)) 
  (h2 : James_pay = 40 * x + 2 * x * (James_hours - 40)) 
  (h3 : James_hours = 41) 
  (h4 : Harry_pay = James_pay) : 
  H = Harry_hours_worked x :=
by
  sorry

end harry_worked_34_hours_l167_167437


namespace sum_of_six_terms_arithmetic_sequence_l167_167827

theorem sum_of_six_terms_arithmetic_sequence (S : ℕ → ℕ)
    (h1 : S 2 = 2)
    (h2 : S 4 = 10) :
    S 6 = 42 :=
by
  sorry

end sum_of_six_terms_arithmetic_sequence_l167_167827


namespace train_speed_l167_167622

theorem train_speed
  (train_length : ℕ)
  (man_speed_kmph : ℕ)
  (time_to_pass : ℕ)
  (speed_of_train : ℝ) :
  train_length = 180 →
  man_speed_kmph = 8 →
  time_to_pass = 4 →
  speed_of_train = 154 := 
by
  sorry

end train_speed_l167_167622


namespace solution_l167_167448

noncomputable def inequality_prove (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5)

noncomputable def equality_condition (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : Prop :=
  (1 / (x + 3) + 1 / (y + 3) = 2 / 5) ↔ (x = 2 ∧ y = 2)

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) : 
  inequality_prove x y h1 h2 h3 ∧ equality_condition x y h1 h2 h3 := by
  sorry

end solution_l167_167448


namespace measure_of_angle_D_l167_167892

def angle_A := 95 -- Defined in step b)
def angle_B := angle_A
def angle_C := angle_A
def angle_D := angle_A + 50
def angle_E := angle_D
def angle_F := angle_D

theorem measure_of_angle_D (x : ℕ) (y : ℕ) :
  (angle_A = x) ∧ (angle_D = y) ∧ (y = x + 50) ∧ (3 * x + 3 * y = 720) → y = 145 :=
by
  intros
  sorry

end measure_of_angle_D_l167_167892


namespace find_ABC_sum_l167_167361

-- Conditions
def poly (A B C : ℤ) (x : ℤ) := x^3 + A * x^2 + B * x + C
def roots_condition (A B C : ℤ) := poly A B C (-1) = 0 ∧ poly A B C 3 = 0 ∧ poly A B C 4 = 0

-- Proof goal
theorem find_ABC_sum (A B C : ℤ) (h : roots_condition A B C) : A + B + C = 11 :=
sorry

end find_ABC_sum_l167_167361


namespace Aiyanna_has_more_cookies_l167_167164

theorem Aiyanna_has_more_cookies (Alyssa_cookies : ℕ) (Aiyanna_cookies : ℕ) (hAlyssa : Alyssa_cookies = 129) (hAiyanna : Aiyanna_cookies = 140) : Aiyanna_cookies - Alyssa_cookies = 11 := 
by sorry

end Aiyanna_has_more_cookies_l167_167164


namespace total_balloons_correct_l167_167489

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l167_167489


namespace gcd_pow_diff_l167_167995

theorem gcd_pow_diff :
  gcd (2 ^ 2100 - 1) (2 ^ 2091 - 1) = 511 := 
sorry

end gcd_pow_diff_l167_167995


namespace tan_cos_identity_15deg_l167_167492

theorem tan_cos_identity_15deg :
  (1 - (Real.tan (Real.pi / 12))^2) * (Real.cos (Real.pi / 12))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end tan_cos_identity_15deg_l167_167492


namespace option_c_equals_9_l167_167452

theorem option_c_equals_9 : (3 * 3 - 3 + 3) = 9 :=
by
  sorry

end option_c_equals_9_l167_167452


namespace total_length_segments_in_figure2_l167_167707

-- Define the original dimensions of the figure
def vertical_side : ℕ := 10
def bottom_horizontal_side : ℕ := 3
def middle_horizontal_side : ℕ := 4
def topmost_horizontal_side : ℕ := 2

-- Define the lengths that are removed to form Figure 2
def removed_sides_length : ℕ :=
  bottom_horizontal_side + topmost_horizontal_side + vertical_side

-- Define the remaining lengths in Figure 2
def remaining_vertical_side : ℕ := vertical_side
def remaining_horizontal_side : ℕ := middle_horizontal_side

-- Total length of segments in Figure 2
def total_length_figure2 : ℕ :=
  remaining_vertical_side + remaining_horizontal_side

-- Conjecture that this total length is 14 units
theorem total_length_segments_in_figure2 : total_length_figure2 = 14 := by
  -- Proof goes here
  sorry

end total_length_segments_in_figure2_l167_167707


namespace payment_difference_correct_l167_167126

noncomputable def prove_payment_difference (x : ℕ) (h₀ : x > 0) : Prop :=
  180 / x - 180 / (x + 2) = 3

theorem payment_difference_correct (x : ℕ) (h₀ : x > 0) : prove_payment_difference x h₀ :=
  by
    sorry

end payment_difference_correct_l167_167126


namespace cone_volume_l167_167733

theorem cone_volume (l : ℝ) (circumference : ℝ) (radius : ℝ) (height : ℝ) (volume : ℝ) 
  (h1 : l = 8) 
  (h2 : circumference = 6 * Real.pi) 
  (h3 : radius = circumference / (2 * Real.pi))
  (h4 : height = Real.sqrt (l^2 - radius^2)) 
  (h5 : volume = (1 / 3) * Real.pi * radius^2 * height) :
  volume = 3 * Real.sqrt 55 * Real.pi := 
  by 
    sorry

end cone_volume_l167_167733


namespace break_even_shirts_needed_l167_167380

-- Define the conditions
def initialInvestment : ℕ := 1500
def costPerShirt : ℕ := 3
def sellingPricePerShirt : ℕ := 20

-- Define the profit per T-shirt and the number of T-shirts to break even
def profitPerShirt (sellingPrice costPrice : ℕ) : ℕ := sellingPrice - costPrice

def shirtsToBreakEven (investment profit : ℕ) : ℕ :=
  (investment + profit - 1) / profit -- ceil division

-- The theorem to prove
theorem break_even_shirts_needed :
  shirtsToBreakEven initialInvestment (profitPerShirt sellingPricePerShirt costPerShirt) = 89 :=
by
  -- Calculation
  sorry

end break_even_shirts_needed_l167_167380


namespace integer_values_m_l167_167016

theorem integer_values_m (m x y : ℤ) (h1 : x - 2 * y = m) (h2 : 2 * x + 3 * y = 2 * m - 3)
    (h3 : 3 * x + y ≥ 0) (h4 : x + 5 * y < 0) : m = 1 ∨ m = 2 :=
by
  sorry

end integer_values_m_l167_167016


namespace arrangement_count_l167_167559

noncomputable def count_arrangements (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  sorry -- The implementation of this function is out of scope for this task

theorem arrangement_count :
  count_arrangements ({1, 2, 3, 4} : Finset ℕ) ({1, 2, 3} : Finset ℕ) = 18 :=
sorry

end arrangement_count_l167_167559


namespace triangles_pentagons_difference_l167_167816

theorem triangles_pentagons_difference :
  ∃ x y : ℕ, 
  (x + y = 50) ∧ (3 * x + 5 * y = 170) ∧ (x - y = 30) :=
sorry

end triangles_pentagons_difference_l167_167816


namespace each_child_play_time_l167_167726

-- Define the conditions
def number_of_children : ℕ := 6
def pair_play_time : ℕ := 120
def pairs_playing_at_a_time : ℕ := 2

-- Define main theorem
theorem each_child_play_time : 
  (pairs_playing_at_a_time * pair_play_time) / number_of_children = 40 :=
sorry

end each_child_play_time_l167_167726


namespace seven_digit_palindromes_count_l167_167477

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end seven_digit_palindromes_count_l167_167477


namespace min_value_x1_x2_frac1_x1x2_l167_167266

theorem min_value_x1_x2_frac1_x1x2 (a x1 x2 : ℝ) (ha : a > 2) (h_sum : x1 + x2 = a) (h_prod : x1 * x2 = a - 2) :
  x1 + x2 + 1 / (x1 * x2) ≥ 4 :=
sorry

end min_value_x1_x2_frac1_x1x2_l167_167266


namespace boat_length_in_steps_l167_167062

theorem boat_length_in_steps (L E S : ℝ) 
  (h1 : 250 * E = L + 250 * S) 
  (h2 : 50 * E = L - 50 * S) :
  L = 83 * E :=
by sorry

end boat_length_in_steps_l167_167062


namespace inequality_solution_l167_167478

theorem inequality_solution :
  {x : ℝ | (x - 1) * (x - 4) * (x - 5)^2 / ((x - 3) * (x^2 - 9)) > 0} = { x : ℝ | -3 < x ∧ x < 3 } :=
sorry

end inequality_solution_l167_167478


namespace proof_of_value_of_6y_plus_3_l167_167553

theorem proof_of_value_of_6y_plus_3 (y : ℤ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 :=
by
  sorry

end proof_of_value_of_6y_plus_3_l167_167553


namespace equation_of_motion_l167_167917

section MotionLaw

variable (t s : ℝ)
variable (v : ℝ → ℝ)
variable (C : ℝ)

-- Velocity function
def velocity (t : ℝ) : ℝ := 6 * t^2 + 1

-- Displacement function (indefinite integral of velocity)
def displacement (t : ℝ) (C : ℝ) : ℝ := 2 * t^3 + t + C

-- Given condition: displacement at t = 3 is 60
axiom displacement_at_3 : displacement 3 C = 60

-- Prove that the equation of motion is s = 2t^3 + t + 3
theorem equation_of_motion :
  ∃ C, displacement t C = 2 * t^3 + t + 3 :=
by
  use 3
  sorry

end MotionLaw

end equation_of_motion_l167_167917


namespace remainder_8927_div_11_l167_167605

theorem remainder_8927_div_11 : 8927 % 11 = 8 :=
by
  sorry

end remainder_8927_div_11_l167_167605


namespace brad_money_l167_167891

noncomputable def money_problem : Prop :=
  ∃ (B J D : ℝ), 
    J = 2 * B ∧
    J = (3/4) * D ∧
    B + J + D = 68 ∧
    B = 12

theorem brad_money : money_problem :=
by {
  -- Insert proof steps here if necessary
  sorry
}

end brad_money_l167_167891


namespace area_expression_l167_167801

noncomputable def overlapping_area (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) : ℝ :=
if h : m ≤ 2 * Real.sqrt 2 then
  6 - Real.sqrt 2 * m
else
  (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8

theorem area_expression (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  let y := overlapping_area m h1 h2
  (if h : m ≤ 2 * Real.sqrt 2 then y = 6 - Real.sqrt 2 * m
   else y = (1 / 4) * m^2 - 2 * Real.sqrt 2 * m + 8) := 
sorry

end area_expression_l167_167801


namespace least_three_digit_product_of_digits_is_8_l167_167511

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l167_167511


namespace heavy_equipment_pay_l167_167819

theorem heavy_equipment_pay
  (total_workers : ℕ)
  (total_payroll : ℕ)
  (laborers : ℕ)
  (laborer_pay : ℕ)
  (heavy_operator_pay : ℕ)
  (h1 : total_workers = 35)
  (h2 : total_payroll = 3950)
  (h3 : laborers = 19)
  (h4 : laborer_pay = 90)
  (h5 : (total_workers - laborers) * heavy_operator_pay + laborers * laborer_pay = total_payroll) :
  heavy_operator_pay = 140 :=
by
  sorry

end heavy_equipment_pay_l167_167819


namespace amanda_family_painting_theorem_l167_167616

theorem amanda_family_painting_theorem
  (rooms_with_4_walls : ℕ)
  (walls_per_room_with_4_walls : ℕ)
  (rooms_with_5_walls : ℕ)
  (walls_per_room_with_5_walls : ℕ)
  (walls_per_person : ℕ)
  (total_rooms : ℕ)
  (h1 : rooms_with_4_walls = 5)
  (h2 : walls_per_room_with_4_walls = 4)
  (h3 : rooms_with_5_walls = 4)
  (h4 : walls_per_room_with_5_walls = 5)
  (h5 : walls_per_person = 8)
  (h6 : total_rooms = 9)
  : rooms_with_4_walls * walls_per_room_with_4_walls +
    rooms_with_5_walls * walls_per_room_with_5_walls =
    5 * walls_per_person :=
by
  sorry

end amanda_family_painting_theorem_l167_167616


namespace smallest_w_l167_167709

theorem smallest_w (w : ℕ) (h1 : 1916 = 2^2 * 479) (h2 : w > 0) : w = 74145392000 ↔ 
  (∀ p e, (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11) → (∃ k, (1916 * w = p^e * k ∧ e ≥ if p = 2 then 6 else 3))) :=
sorry

end smallest_w_l167_167709


namespace find_f_x_l167_167246

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x - 1) = x^2 - x) : ∀ x : ℝ, f x = (1/4) * (x^2 - 1) := 
sorry

end find_f_x_l167_167246


namespace cube_surface_area_l167_167785

-- Define the volume condition
def volume (s : ℕ) : ℕ := s^3

-- Define the surface area function
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- State the theorem to be proven
theorem cube_surface_area (s : ℕ) (h : volume s = 729) : surface_area s = 486 :=
by
  sorry

end cube_surface_area_l167_167785


namespace library_book_configurations_l167_167429

def number_of_valid_configurations (total_books : ℕ) (min_in_library : ℕ) (min_checked_out : ℕ) : ℕ :=
  (total_books - (min_in_library + min_checked_out + 1)) + 1

theorem library_book_configurations : number_of_valid_configurations 8 2 2 = 5 :=
by
  -- Here we would write the Lean proof, but since we are only interested in the statement:
  sorry

end library_book_configurations_l167_167429


namespace sum_of_coeffs_in_expansion_l167_167548

theorem sum_of_coeffs_in_expansion (n : ℕ) : 
  (1 - 2 : ℤ)^n = (-1 : ℤ)^n :=
by
  sorry

end sum_of_coeffs_in_expansion_l167_167548


namespace line_through_points_l167_167767

-- Define the conditions and the required proof statement
theorem line_through_points (x1 y1 z1 x2 y2 z2 x y z m n p : ℝ) :
  (∃ m n p, (x-x1) / m = (y-y1) / n ∧ (y-y1) / n = (z-z1) / p) → 
  (x-x1) / (x2 - x1) = (y-y1) / (y2 - y1) ∧ 
  (y-y1) / (y2 - y1) = (z-z1) / (z2 - z1) :=
sorry

end line_through_points_l167_167767


namespace find_certain_number_l167_167046

theorem find_certain_number (x : ℝ) (h : ((7 * (x + 5)) / 5) - 5 = 33) : x = 22 :=
by
  sorry

end find_certain_number_l167_167046


namespace find_g_one_l167_167704

variable {α : Type} [AddGroup α]

def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = - f x

def is_even (g : α → α) : Prop :=
∀ x, g (-x) = g x

theorem find_g_one
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end find_g_one_l167_167704


namespace graph_symmetry_l167_167864

theorem graph_symmetry (f : ℝ → ℝ) : 
  ∀ x : ℝ, f (x - 1) = f (-(x - 1)) ↔ x = 1 :=
by 
  sorry

end graph_symmetry_l167_167864


namespace arithmetic_geometric_l167_167370

theorem arithmetic_geometric (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, a 1 * r = a 3 ∧ a 3 * r = a 4) :
  a 2 = -6 :=
by sorry

end arithmetic_geometric_l167_167370


namespace smallest_lcm_l167_167249

theorem smallest_lcm (k l : ℕ) (hk : k ≥ 1000) (hl : l ≥ 1000) (huk : k < 10000) (hul : l < 10000) (hk_pos : 0 < k) (hl_pos : 0 < l) (h_gcd: Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
by
  sorry

end smallest_lcm_l167_167249


namespace area_ratio_of_triangles_l167_167403

theorem area_ratio_of_triangles (AC AD : ℝ) (h : ℝ) (hAC : AC = 1) (hAD : AD = 4) :
  (AC * h / 2) / ((AD - AC) * h / 2) = 1 / 3 :=
by
  sorry

end area_ratio_of_triangles_l167_167403


namespace ratio_josh_to_selena_l167_167337

def total_distance : ℕ := 36
def selena_distance : ℕ := 24

def josh_distance (td sd : ℕ) : ℕ := td - sd

theorem ratio_josh_to_selena : (josh_distance total_distance selena_distance) / selena_distance = 1 / 2 :=
by
  sorry

end ratio_josh_to_selena_l167_167337


namespace solve_for_x_l167_167008

variable (x : ℝ)

theorem solve_for_x (h : (4 * x + 2) / (5 * x - 5) = 3 / 4) : x = -23 := 
by
  sorry

end solve_for_x_l167_167008


namespace birds_flew_away_l167_167222

-- Define the initial and remaining birds
def original_birds : ℕ := 12
def remaining_birds : ℕ := 4

-- Define the number of birds that flew away
noncomputable def flew_away_birds : ℕ := original_birds - remaining_birds

-- State the theorem that the number of birds that flew away is 8
theorem birds_flew_away : flew_away_birds = 8 := by
  -- Lean expects a proof here. For now, we use sorry to indicate the proof is skipped.
  sorry

end birds_flew_away_l167_167222


namespace total_bees_l167_167127

theorem total_bees 
    (B : ℕ) 
    (h1 : (1/5 : ℚ) * B + (1/3 : ℚ) * B + (2/5 : ℚ) * B + 1 = B) : 
    B = 15 := sorry

end total_bees_l167_167127


namespace solve_inequality_l167_167315

theorem solve_inequality (x : ℝ) (h : x < 4) : (x - 2) / (x - 4) ≥ 3 := sorry

end solve_inequality_l167_167315


namespace units_digit_34_pow_30_l167_167398

theorem units_digit_34_pow_30 :
  (34 ^ 30) % 10 = 6 :=
by
  sorry

end units_digit_34_pow_30_l167_167398


namespace probability_one_hits_l167_167013

theorem probability_one_hits 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 1 / 2) (hB : p_B = 1 / 3):
  p_A * (1 - p_B) + (1 - p_A) * p_B = 1 / 2 := by
  sorry

end probability_one_hits_l167_167013


namespace first_batch_students_l167_167594

theorem first_batch_students 
  (x : ℕ) 
  (avg1 avg2 avg3 overall_avg : ℝ) 
  (n2 n3 : ℕ) 
  (h_avg1 : avg1 = 45) 
  (h_avg2 : avg2 = 55) 
  (h_avg3 : avg3 = 65) 
  (h_n2 : n2 = 50) 
  (h_n3 : n3 = 60) 
  (h_overall_avg : overall_avg = 56.333333333333336) 
  (h_eq : overall_avg = (45 * x + 55 * 50 + 65 * 60) / (x + 50 + 60)) 
  : x = 40 :=
sorry

end first_batch_students_l167_167594


namespace sweatshirt_sales_l167_167405

variables (S H : ℝ)

theorem sweatshirt_sales (h1 : 13 * S + 9 * H = 370) (h2 : 9 * S + 2 * H = 180) :
  12 * S + 6 * H = 300 :=
sorry

end sweatshirt_sales_l167_167405


namespace min_value_x_y_l167_167449

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : 
  x + y ≥ 20 :=
sorry

end min_value_x_y_l167_167449


namespace addition_subtraction_result_l167_167731

theorem addition_subtraction_result :
  27474 + 3699 + 1985 - 2047 = 31111 :=
by {
  sorry
}

end addition_subtraction_result_l167_167731


namespace volume_tetrahedron_OABC_correct_l167_167551

noncomputable def volume_tetrahedron_OABC : ℝ :=
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  (1 / 6) * a * b * c

theorem volume_tetrahedron_OABC_correct :
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  let volume := (1 / 6) * a * b * c
  volume = 8 * Real.sqrt 99 / 3 :=
by
  sorry

end volume_tetrahedron_OABC_correct_l167_167551


namespace find_a_l167_167270

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 3 else 4 / x

theorem find_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 2 :=
sorry

end find_a_l167_167270


namespace weight_of_sugar_is_16_l167_167325

def weight_of_sugar_bag (weight_of_sugar weight_of_salt remaining_weight weight_removed : ℕ) : Prop :=
  weight_of_sugar + weight_of_salt - weight_removed = remaining_weight

theorem weight_of_sugar_is_16 :
  ∃ (S : ℕ), weight_of_sugar_bag S 30 42 4 ∧ S = 16 :=
by
  sorry

end weight_of_sugar_is_16_l167_167325


namespace total_material_weight_l167_167812

def gravel_weight : ℝ := 5.91
def sand_weight : ℝ := 8.11

theorem total_material_weight : gravel_weight + sand_weight = 14.02 := by
  sorry

end total_material_weight_l167_167812


namespace complex_division_l167_167595

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (2 / (1 + i)) = (1 - i) :=
by
  sorry

end complex_division_l167_167595


namespace six_people_with_A_not_on_ends_l167_167319

-- Define the conditions and the problem statement
def standing_arrangements (n : ℕ) (A : Type) :=
  {l : List A // l.length = n}

theorem six_people_with_A_not_on_ends : 
  (arr : standing_arrangements 6 ℕ) → 
  (∀ a ∈ arr.val, a ≠ 0 ∧ a ≠ 5) → 
  ∃! (total_arrangements : ℕ), total_arrangements = 480 :=
  by
    sorry

end six_people_with_A_not_on_ends_l167_167319


namespace Q_equals_10_04_l167_167696
-- Import Mathlib for mathematical operations and equivalence checking

-- Define the given conditions
def a := 6
def b := 3
def c := 2

-- Define the expression to be evaluated
def Q : ℚ := (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2)

-- Prove that the expression equals 10.04
theorem Q_equals_10_04 : Q = 10.04 := by
  -- Proof goes here
  sorry

end Q_equals_10_04_l167_167696


namespace solution_set_l167_167862

noncomputable def f : ℝ → ℝ := sorry
axiom f'_lt_one_third (x : ℝ) : deriv f x < 1 / 3
axiom f_at_two : f 2 = 1

theorem solution_set : {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.logb 2 x) > (Real.logb 2 x + 1) / 3} :=
by
  sorry

end solution_set_l167_167862


namespace domain_of_sqrt_fraction_l167_167156

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (x - 2 ≥ 0 ∧ 5 - x > 0) ↔ (2 ≤ x ∧ x < 5) :=
by
  sorry

end domain_of_sqrt_fraction_l167_167156


namespace proof_problem_l167_167326

-- Given conditions for propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Combined proposition p and q
def p_and_q (a : ℝ) := p a ∧ q a

-- Statement of the proof problem: Prove that p_and_q a → a ≤ -1
theorem proof_problem (a : ℝ) : p_and_q a → (a ≤ -1) :=
by
  sorry

end proof_problem_l167_167326


namespace find_p_l167_167258

theorem find_p (m n p : ℝ) 
  (h1 : m = 3 * n + 5) 
  (h2 : m + 2 = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  sorry

end find_p_l167_167258


namespace speed_limit_correct_l167_167582

def speed_limit_statement (v : ℝ) : Prop :=
  v ≤ 70

theorem speed_limit_correct (v : ℝ) (h : v ≤ 70) : speed_limit_statement v :=
by
  exact h

#print axioms speed_limit_correct

end speed_limit_correct_l167_167582


namespace trig_identity_l167_167200

theorem trig_identity {α : ℝ} (h : Real.tan α = 2) : 
  (Real.sin (π + α) - Real.cos (π - α)) / 
  (Real.sin (π / 2 + α) - Real.cos (3 * π / 2 - α)) 
  = -1 / 3 := 
by 
  sorry

end trig_identity_l167_167200


namespace andrew_vacation_days_l167_167349

-- Andrew's working days and vacation accrual rate
def days_worked : ℕ := 300
def vacation_rate : Nat := 10
def vacation_days_earned : ℕ := days_worked / vacation_rate

-- Days off in March and September
def days_off_march : ℕ := 5
def days_off_september : ℕ := 2 * days_off_march
def total_days_off : ℕ := days_off_march + days_off_september

-- Remaining vacation days calculation
def remaining_vacation_days : ℕ := vacation_days_earned - total_days_off

-- Problem statement to prove
theorem andrew_vacation_days : remaining_vacation_days = 15 :=
by
  -- Substitute the known values and perform the calculation
  unfold remaining_vacation_days vacation_days_earned total_days_off vacation_rate days_off_march days_off_september days_worked
  norm_num
  sorry

end andrew_vacation_days_l167_167349


namespace infinite_series_sum_l167_167275

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * (n + 1) * (n + 1) + 2 * (n + 1) + 1) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))) 
  = 7 / 6 := 
by
  sorry

end infinite_series_sum_l167_167275


namespace volume_of_region_l167_167641

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

theorem volume_of_region (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) :
  volume_of_sphere r_large - volume_of_sphere r_small = 372 * Real.pi := by
  rw [h_small, h_large]
  sorry

end volume_of_region_l167_167641


namespace directrix_of_parabola_l167_167610

theorem directrix_of_parabola (x y : ℝ) : (x ^ 2 = y) → (4 * y + 1 = 0) :=
sorry

end directrix_of_parabola_l167_167610


namespace solve_n_l167_167387

open Nat

def condition (n : ℕ) : Prop := 2^(n + 1) * 2^3 = 2^10

theorem solve_n (n : ℕ) (hn_pos : 0 < n) (h_cond : condition n) : n = 6 :=
by
  sorry

end solve_n_l167_167387


namespace arrange_polynomial_descending_l167_167075

variable (a b : ℤ)

def polynomial := -a + 3 * a^5 * b^3 + 5 * a^3 * b^5 - 9 + 4 * a^2 * b^2 

def rearranged_polynomial := 3 * a^5 * b^3 + 5 * a^3 * b^5 + 4 * a^2 * b^2 - a - 9

theorem arrange_polynomial_descending :
  polynomial a b = rearranged_polynomial a b :=
sorry

end arrange_polynomial_descending_l167_167075


namespace count_even_numbers_is_320_l167_167985

noncomputable def count_even_numbers_with_distinct_digits : Nat := 
  let unit_choices := 5  -- Choices for the unit digit (0, 2, 4, 6, 8)
  let hundreds_choices := 8  -- Choices for the hundreds digit (1 to 9, excluding the unit digit)
  let tens_choices := 8  -- Choices for the tens digit (0 to 9, excluding the hundreds and unit digit)
  unit_choices * hundreds_choices * tens_choices

theorem count_even_numbers_is_320 : count_even_numbers_with_distinct_digits = 320 := by
  sorry

end count_even_numbers_is_320_l167_167985


namespace measure_of_angle_C_l167_167120

variable (a b c : ℝ) (S : ℝ)

-- Conditions
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_equation : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)

-- The problem
theorem measure_of_angle_C (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.arctan (Real.sqrt 3) ∧ C = Real.pi / 3 :=
by
  sorry

end measure_of_angle_C_l167_167120


namespace inverse_proposition_false_l167_167186

-- Definitions for the conditions
def congruent (A B C D E F: ℝ) : Prop := 
  A = D ∧ B = E ∧ C = F

def angles_equal (α β γ δ ε ζ: ℝ) : Prop := 
  α = δ ∧ β = ε ∧ γ = ζ

def original_proposition (A B C D E F α β γ : ℝ) : Prop :=
  congruent A B C D E F → angles_equal α β γ A B C

-- The inverse proposition
def inverse_proposition (α β γ δ ε ζ A B C D E F : ℝ) : Prop :=
  angles_equal α β γ δ ε ζ → congruent A B C D E F

-- The main theorem: the inverse proposition is false
theorem inverse_proposition_false (α β γ δ ε ζ A B C D E F : ℝ) :
  ¬(inverse_proposition α β γ δ ε ζ A B C D E F) := by
  sorry

end inverse_proposition_false_l167_167186


namespace trick_deck_cost_l167_167125

theorem trick_deck_cost :
  (∃ x : ℝ, 4 * x + 4 * x = 72) → ∃ x : ℝ, x = 9 := sorry

end trick_deck_cost_l167_167125


namespace calculate_gain_percentage_l167_167954

theorem calculate_gain_percentage (CP SP : ℝ) (h1 : 0.9 * CP = 450) (h2 : SP = 550) : 
  (SP - CP) / CP * 100 = 10 :=
by
  sorry

end calculate_gain_percentage_l167_167954


namespace average_salary_l167_167388

theorem average_salary (R S T : ℝ) 
  (h1 : (R + S) / 2 = 4000) 
  (h2 : T = 7000) : 
  (R + S + T) / 3 = 5000 :=
by
  sorry

end average_salary_l167_167388


namespace sum_of_quarter_circle_arcs_l167_167763

-- Define the main variables and problem statement.
variable (D : ℝ) -- Diameter of the original circle.
variable (n : ℕ) (hn : 0 < n) -- Number of parts (positive integer).

-- Define a theorem stating that the sum of quarter-circle arcs is greater than D, but less than (pi D / 2) as n tends to infinity.
theorem sum_of_quarter_circle_arcs (hn : 0 < n) :
  D < (π * D) / 4 ∧ (π * D) / 4 < (π * D) / 2 :=
by
  sorry -- Proof of the theorem goes here.

end sum_of_quarter_circle_arcs_l167_167763


namespace p_6_is_126_l167_167897

noncomputable def p (x : ℝ) : ℝ := sorry

axiom h1 : p 1 = 1
axiom h2 : p 2 = 2
axiom h3 : p 3 = 3
axiom h4 : p 4 = 4
axiom h5 : p 5 = 5

theorem p_6_is_126 : p 6 = 126 := sorry

end p_6_is_126_l167_167897


namespace collinear_points_b_value_l167_167858

theorem collinear_points_b_value :
  ∃ b : ℝ, (3 - (-2)) * (11 - b) = (8 - 3) * (1 - b) → b = -9 :=
by
  sorry

end collinear_points_b_value_l167_167858


namespace simplify_expression_eq_sqrt3_l167_167681

theorem simplify_expression_eq_sqrt3
  (a : ℝ)
  (h : a = Real.sqrt 3 + 1) :
  ( (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) ) = Real.sqrt 3 := sorry

end simplify_expression_eq_sqrt3_l167_167681


namespace isabella_initial_hair_length_l167_167890

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end isabella_initial_hair_length_l167_167890


namespace sum_of_real_values_l167_167583

theorem sum_of_real_values (x : ℝ) (h : |3 * x + 1| = 3 * |x - 3|) : x = 4 / 3 := sorry

end sum_of_real_values_l167_167583


namespace solve_equation_l167_167842

theorem solve_equation :
  ∀ x : ℝ, (3 * x^2 / (x - 2) - (3 * x + 4) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) →
    (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6) :=
by
  intro x h
  -- the proof would go here
  sorry

end solve_equation_l167_167842


namespace division_of_squares_l167_167466

theorem division_of_squares {a b : ℕ} (h1 : a < 1000) (h2 : b > 0) (h3 : b^10 ∣ a^21) : b ∣ a^2 := 
sorry

end division_of_squares_l167_167466


namespace smallest_k_sum_sequence_l167_167108

theorem smallest_k_sum_sequence (n : ℕ) (h : 100 = (n + 1) * (2 * 9 + n) / 2) : k = 9 := 
sorry

end smallest_k_sum_sequence_l167_167108


namespace range_of_m_l167_167677

def M := {y : ℝ | ∃ (x : ℝ), y = (1/2)^x}
def N (m : ℝ) := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = ((1/(m-1) + 1) * (x - 1) + (|m| - 1) * (x - 2))}

theorem range_of_m (m : ℝ) : (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l167_167677


namespace seeds_per_can_l167_167778

theorem seeds_per_can (total_seeds : ℝ) (number_of_cans : ℝ) (h1 : total_seeds = 54.0) (h2 : number_of_cans = 9.0) : (total_seeds / number_of_cans = 6.0) :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end seeds_per_can_l167_167778


namespace andrey_gifts_l167_167029

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end andrey_gifts_l167_167029


namespace solveEquation_l167_167475

noncomputable def findNonZeroSolution (z : ℝ) : Prop :=
  (5 * z) ^ 10 = (20 * z) ^ 5 ∧ z ≠ 0

theorem solveEquation : ∃ z : ℝ, findNonZeroSolution z ∧ z = 4 / 5 := by
  exists 4 / 5
  simp [findNonZeroSolution]
  sorry

end solveEquation_l167_167475


namespace one_fourth_of_7point2_is_9div5_l167_167088

theorem one_fourth_of_7point2_is_9div5 : (7.2 / 4 : ℚ) = 9 / 5 := 
by sorry

end one_fourth_of_7point2_is_9div5_l167_167088


namespace find_missing_number_l167_167992

theorem find_missing_number (n x : ℕ) (h : n * (n + 1) / 2 - x = 2012) : x = 4 := by
  sorry

end find_missing_number_l167_167992


namespace reduced_bucket_fraction_l167_167836

theorem reduced_bucket_fraction (C : ℝ) (F : ℝ) (h : 25 * F * C = 10 * C) : F = 2 / 5 :=
by sorry

end reduced_bucket_fraction_l167_167836


namespace root_integer_l167_167825

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

def is_root (x_0 : ℝ) : Prop := f x_0 = 0

theorem root_integer (x_0 : ℝ) (h : is_root x_0) : Int.floor x_0 = 2 := by
  sorry

end root_integer_l167_167825


namespace range_of_m_l167_167956

theorem range_of_m (m : ℝ) :
  (1 - 2 * m > 0) ∧ (m + 1 > 0) → -1 < m ∧ m < 1/2 :=
by
  sorry

end range_of_m_l167_167956


namespace sin_105_mul_sin_15_eq_one_fourth_l167_167957

noncomputable def sin_105_deg := Real.sin (105 * Real.pi / 180)
noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)

theorem sin_105_mul_sin_15_eq_one_fourth :
  sin_105_deg * sin_15_deg = 1 / 4 :=
by
  sorry

end sin_105_mul_sin_15_eq_one_fourth_l167_167957


namespace arithmetic_progression_rth_term_l167_167964

open Nat

theorem arithmetic_progression_rth_term (n r : ℕ) (Sn : ℕ → ℕ) 
  (h : ∀ n, Sn n = 5 * n + 4 * n^2) : Sn r - Sn (r - 1) = 8 * r + 1 :=
by
  sorry

end arithmetic_progression_rth_term_l167_167964


namespace Jeff_total_laps_l167_167484

theorem Jeff_total_laps (laps_saturday : ℕ) (laps_sunday_morning : ℕ) (laps_remaining : ℕ)
  (h1 : laps_saturday = 27) (h2 : laps_sunday_morning = 15) (h3 : laps_remaining = 56) :
  (laps_saturday + laps_sunday_morning + laps_remaining) = 98 := 
by
  sorry

end Jeff_total_laps_l167_167484


namespace intersection_count_l167_167454

def M (x y : ℝ) : Prop := y^2 = x - 1
def N (x y m : ℝ) : Prop := y = 2 * x - 2 * m^2 + m - 2

theorem intersection_count (m x y : ℝ) :
  (M x y ∧ N x y m) → (∃ n : ℕ, n = 1 ∨ n = 2) :=
sorry

end intersection_count_l167_167454


namespace minimum_glue_drops_to_prevent_37_gram_subset_l167_167944

def stones : List ℕ := List.range' 1 36  -- List of stones with masses from 1 to 36 grams

def glue_drop_combination_invalid (stones : List ℕ) : Prop :=
  ¬ (∃ (subset : List ℕ), subset.sum = 37 ∧ (∀ s ∈ subset, s ∈ stones))

def min_glue_drops (stones : List ℕ) : ℕ := 
  9 -- as per the solution

theorem minimum_glue_drops_to_prevent_37_gram_subset :
  ∀ (s : List ℕ), s = stones → glue_drop_combination_invalid s → min_glue_drops s = 9 :=
by intros; sorry

end minimum_glue_drops_to_prevent_37_gram_subset_l167_167944


namespace total_truck_loads_needed_l167_167607

noncomputable def truck_loads_of_material : ℝ :=
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5 -- log is the natural logarithm in Lean
  sand + dirt + cement + gravel

theorem total_truck_loads_needed : truck_loads_of_material = 1.8401374808985008 := by
  sorry

end total_truck_loads_needed_l167_167607


namespace susan_age_is_11_l167_167736

theorem susan_age_is_11 (S A : ℕ) 
  (h1 : A = S + 5) 
  (h2 : A + S = 27) : 
  S = 11 := 
by 
  sorry

end susan_age_is_11_l167_167736


namespace factorial_not_multiple_of_57_l167_167093

theorem factorial_not_multiple_of_57 (n : ℕ) (h : ¬ (57 ∣ n!)) : n < 19 := 
sorry

end factorial_not_multiple_of_57_l167_167093


namespace bundles_burned_in_afternoon_l167_167192

theorem bundles_burned_in_afternoon 
  (morning_burn : ℕ)
  (start_bundles : ℕ)
  (end_bundles : ℕ)
  (h_morning_burn : morning_burn = 4)
  (h_start : start_bundles = 10)
  (h_end : end_bundles = 3)
  : (start_bundles - morning_burn - end_bundles) = 3 := 
by 
  sorry

end bundles_burned_in_afternoon_l167_167192


namespace find_geometric_progression_l167_167929

theorem find_geometric_progression (a b c : ℚ)
  (h1 : a * c = b * b)
  (h2 : a + c = 2 * (b + 8))
  (h3 : a * (c + 64) = (b + 8) * (b + 8)) :
  (a = 4/9 ∧ b = -20/9 ∧ c = 100/9) ∨ (a = 4 ∧ b = 12 ∧ c = 36) :=
sorry

end find_geometric_progression_l167_167929


namespace simplify_expression_l167_167779

theorem simplify_expression (a b c : ℝ) (h : a + b + c = 3) : 
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) →
  (1 / (b^2 + c^2 - 3 * a^2) + 1 / (a^2 + c^2 - 3 * b^2) + 1 / (a^2 + b^2 - 3 * c^2) = -3) :=
by
  intros
  sorry

end simplify_expression_l167_167779


namespace problem_statement_l167_167292

theorem problem_statement (a b c : ℝ) (h1 : a ∈ Set.Ioi 0) (h2 : b ∈ Set.Ioi 0) (h3 : c ∈ Set.Ioi 0) (h4 : a^2 + b^2 + c^2 = 3) : 
  1 / (2 - a) + 1 / (2 - b) + 1 / (2 - c) ≥ 3 := 
sorry

end problem_statement_l167_167292


namespace Sara_has_8_balloons_l167_167278

theorem Sara_has_8_balloons (Tom_balloons Sara_balloons total_balloons : ℕ)
  (htom : Tom_balloons = 9)
  (htotal : Tom_balloons + Sara_balloons = 17) :
  Sara_balloons = 8 :=
by
  sorry

end Sara_has_8_balloons_l167_167278


namespace right_triangle_area_l167_167271

theorem right_triangle_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  (1 / 2 : ℝ) * a * b = 24 := by
  sorry

end right_triangle_area_l167_167271


namespace square_of_binomial_l167_167250

-- Define a condition that the given term is the square of a binomial.
theorem square_of_binomial (a b: ℝ) : (a + b) * (a + b) = (a + b) ^ 2 :=
by {
  -- The proof is omitted.
  sorry
}

end square_of_binomial_l167_167250


namespace combines_like_terms_l167_167655

theorem combines_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := 
by sorry

end combines_like_terms_l167_167655


namespace distance_to_nearest_river_l167_167665

theorem distance_to_nearest_river (d : ℝ) (h₁ : ¬ (d ≤ 12)) (h₂ : ¬ (d ≥ 15)) (h₃ : ¬ (d ≥ 10)) :
  12 < d ∧ d < 15 :=
by 
  sorry

end distance_to_nearest_river_l167_167665


namespace carrie_total_spend_l167_167949

def cost_per_tshirt : ℝ := 9.15
def number_of_tshirts : ℝ := 22

theorem carrie_total_spend : (cost_per_tshirt * number_of_tshirts) = 201.30 := by 
  sorry

end carrie_total_spend_l167_167949


namespace container_capacity_l167_167977

-- Definitions based on the conditions
def tablespoons_per_cup := 3
def ounces_per_cup := 8
def tablespoons_added := 15

-- Problem statement
theorem container_capacity : 
  (tablespoons_added / tablespoons_per_cup) * ounces_per_cup = 40 :=
  sorry

end container_capacity_l167_167977


namespace Newville_Academy_fraction_l167_167430

theorem Newville_Academy_fraction :
  let total_students := 100
  let enjoy_sports := 0.7 * total_students
  let not_enjoy_sports := 0.3 * total_students
  let say_enjoy_right := 0.75 * enjoy_sports
  let say_not_enjoy_wrong := 0.25 * enjoy_sports
  let say_not_enjoy_right := 0.85 * not_enjoy_sports
  let say_enjoy_wrong := 0.15 * not_enjoy_sports
  let say_not_enjoy_total := say_not_enjoy_wrong + say_not_enjoy_right
  let say_not_enjoy_but_enjoy := say_not_enjoy_wrong
  (say_not_enjoy_but_enjoy / say_not_enjoy_total) = (7 / 17) := by
  sorry

end Newville_Academy_fraction_l167_167430


namespace find_x_l167_167532

theorem find_x (x : ℝ) (hx1 : x > 0) 
  (h1 : 0.20 * x + 14 = (1 / 3) * ((3 / 4) * x + 21)) : x = 140 :=
sorry

end find_x_l167_167532


namespace table_tennis_total_rounds_l167_167450

-- Mathematical equivalent proof problem in Lean 4 statement
theorem table_tennis_total_rounds
  (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) (total_rounds : ℕ)
  (hA : A_played = 5) (hB : B_played = 4) (hC : C_referee = 2) :
  total_rounds = 7 :=
by
  -- Proof omitted
  sorry

end table_tennis_total_rounds_l167_167450


namespace chairs_left_to_move_l167_167148

theorem chairs_left_to_move (total_chairs : ℕ) (carey_chairs : ℕ) (pat_chairs : ℕ) (h1 : total_chairs = 74)
  (h2 : carey_chairs = 28) (h3 : pat_chairs = 29) : total_chairs - carey_chairs - pat_chairs = 17 :=
by 
  sorry

end chairs_left_to_move_l167_167148


namespace range_of_a_if_q_sufficient_but_not_necessary_for_p_l167_167580

variable {x a : ℝ}

def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

theorem range_of_a_if_q_sufficient_but_not_necessary_for_p :
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a) → a ∈ Set.Ici 1 := 
sorry

end range_of_a_if_q_sufficient_but_not_necessary_for_p_l167_167580


namespace geometric_sequence_sixth_term_l167_167296

theorem geometric_sequence_sixth_term (a b : ℚ) (h : a = 3 ∧ b = -1/2) : 
  (a * (b / a) ^ 5) = -1/2592 :=
by
  sorry

end geometric_sequence_sixth_term_l167_167296


namespace base_conversion_l167_167523

theorem base_conversion (C D : ℕ) (h₁ : 0 ≤ C ∧ C < 8) (h₂ : 0 ≤ D ∧ D < 5) (h₃ : 7 * C = 4 * D) :
  8 * C + D = 0 := by
  sorry

end base_conversion_l167_167523


namespace operation_5_7_eq_35_l167_167503

noncomputable def operation (x y : ℝ) : ℝ := sorry

axiom condition1 :
  ∀ (x y : ℝ), (x * y > 0) → (operation (x * y) y = x * (operation y y))

axiom condition2 :
  ∀ (x : ℝ), (x > 0) → (operation (operation x 1) x = operation x 1)

axiom condition3 :
  (operation 1 1 = 2)

theorem operation_5_7_eq_35 : operation 5 7 = 35 :=
by
  sorry

end operation_5_7_eq_35_l167_167503


namespace smaller_circle_y_coordinate_l167_167950

theorem smaller_circle_y_coordinate 
  (center : ℝ × ℝ) 
  (P : ℝ × ℝ)
  (S : ℝ × ℝ) 
  (QR : ℝ)
  (r_large : ℝ):
    center = (0, 0) → P = (5, 12) → QR = 2 → S.1 = 0 → S.2 = k → r_large = 13 → k = 11 := 
by
  intros h_center hP hQR hSx hSy hr_large
  sorry

end smaller_circle_y_coordinate_l167_167950


namespace single_jalapeno_strips_l167_167138

-- Definitions based on conditions
def strips_per_sandwich : ℕ := 4
def minutes_per_sandwich : ℕ := 5
def hours_per_day : ℕ := 8
def total_jalapeno_peppers_used : ℕ := 48
def minutes_per_hour : ℕ := 60

-- Calculate intermediate steps
def total_minutes : ℕ := hours_per_day * minutes_per_hour
def total_sandwiches_served : ℕ := total_minutes / minutes_per_sandwich
def total_strips_needed : ℕ := total_sandwiches_served * strips_per_sandwich

theorem single_jalapeno_strips :
  total_strips_needed / total_jalapeno_peppers_used = 8 := 
by
  sorry

end single_jalapeno_strips_l167_167138


namespace geometric_series_sum_l167_167872

theorem geometric_series_sum :
  let a := (1/2 : ℚ)
  let r := (-1/3 : ℚ)
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 547 / 1458 :=
by
  sorry

end geometric_series_sum_l167_167872


namespace determine_x_squared_plus_y_squared_l167_167984

theorem determine_x_squared_plus_y_squared (x y : ℝ) 
(h : (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6) : x^2 + y^2 = 4 :=
sorry

end determine_x_squared_plus_y_squared_l167_167984


namespace estimate_sqrt_diff_l167_167966

-- Defining approximate values for square roots
def approx_sqrt_90 : ℝ := 9.5
def approx_sqrt_88 : ℝ := 9.4

-- Main statement
theorem estimate_sqrt_diff : |(approx_sqrt_90 - approx_sqrt_88) - 0.10| < 0.01 := by
  sorry

end estimate_sqrt_diff_l167_167966


namespace line_does_not_pass_first_quadrant_l167_167083

open Real

theorem line_does_not_pass_first_quadrant (a b : ℝ) (h₁ : a > 0) (h₂ : b < 0) : 
  ¬∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (ax + y - b = 0) :=
sorry

end line_does_not_pass_first_quadrant_l167_167083


namespace number_of_ways_l167_167241

-- Define the conditions
def num_people : ℕ := 3
def num_sports : ℕ := 4

-- Prove the total number of different ways
theorem number_of_ways : num_sports ^ num_people = 64 := by
  sorry

end number_of_ways_l167_167241


namespace wire_length_before_cut_l167_167847

-- Defining the conditions
def wire_cut (L S : ℕ) : Prop :=
  S = 20 ∧ S = (2 / 5 : ℚ) * L

-- The statement we need to prove
theorem wire_length_before_cut (L S : ℕ) (h : wire_cut L S) : (L + S) = 70 := 
by 
  sorry

end wire_length_before_cut_l167_167847


namespace emma_bank_account_balance_l167_167436

theorem emma_bank_account_balance
  (initial_balance : ℕ)
  (daily_spend : ℕ)
  (days_in_week : ℕ)
  (unit_bill : ℕ) :
  initial_balance = 100 → daily_spend = 8 → days_in_week = 7 → unit_bill = 5 →
  (initial_balance - daily_spend * days_in_week) % unit_bill = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end emma_bank_account_balance_l167_167436


namespace train_length_approx_l167_167026

noncomputable def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_seconds

theorem train_length_approx (speed_km_hr time_seconds : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_seconds = 4) :
  length_of_train speed_km_hr time_seconds = 133.32 :=
by
  sorry

end train_length_approx_l167_167026


namespace total_cost_eq_1400_l167_167716

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l167_167716


namespace probability_A_or_B_complement_l167_167899

-- Define the sample space for rolling a die
def sample_space : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define Event A: the outcome is an even number not greater than 4
def event_A : Finset ℕ := {2, 4}

-- Define Event B: the outcome is less than 6
def event_B : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the complement of Event B
def event_B_complement : Finset ℕ := {6}

-- Mutually exclusive property of events A and B_complement
axiom mutually_exclusive (A B_complement: Finset ℕ) : A ∩ B_complement = ∅

-- Define the probability function
def probability (events: Finset ℕ) : ℚ := (events.card : ℚ) / (sample_space.card : ℚ)

-- Theorem stating the probability of event (A + B_complement)
theorem probability_A_or_B_complement : probability (event_A ∪ event_B_complement) = 1 / 2 :=
by 
  sorry

end probability_A_or_B_complement_l167_167899


namespace ratio_addition_l167_167133

theorem ratio_addition (x : ℤ) (h : 4 + x = 3 * (15 + x) / 4): x = 29 :=
by
  sorry

end ratio_addition_l167_167133


namespace percentage_vanilla_orders_l167_167124

theorem percentage_vanilla_orders 
  (V C : ℕ) 
  (h1 : V = 2 * C) 
  (h2 : V + C = 220) 
  (h3 : C = 22) : 
  (V * 100) / 220 = 20 := 
by 
  sorry

end percentage_vanilla_orders_l167_167124


namespace proof_problem_l167_167005

noncomputable def arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  n * (a 1) + ((n * (n - 1)) / 2) * (a 2 - a 1)

theorem proof_problem
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (d : ℕ)
  (h_d_gt_zero : d > 0)
  (h_a1 : a 1 = 1)
  (h_S : ∀ n, S n = arithmetic_sequence_sum n a)
  (h_S2_S3 : S 2 * S 3 = 36)
  (h_arith_seq : ∀ n, a (n + 1) = a 1 + n * d)
  (m k : ℕ)
  (h_mk_pos : m > 0 ∧ k > 0)
  (sum_condition : (k + 1) * (a m + a (m + k)) / 2 = 65) :
  d = 2 ∧ (∀ n, S n = n * n) ∧ m = 5 ∧ k = 4 :=
by 
  sorry

end proof_problem_l167_167005


namespace radius_of_tangent_circle_l167_167921

theorem radius_of_tangent_circle (side_length : ℝ) (num_semicircles : ℕ)
  (r_s : ℝ) (r : ℝ)
  (h1 : side_length = 4)
  (h2 : num_semicircles = 16)
  (h3 : r_s = side_length / 4 / 2)
  (h4 : r = (9 : ℝ) / (2 * Real.sqrt 5)) :
  r = (9 * Real.sqrt 5) / 10 :=
by
  rw [h4]
  sorry

end radius_of_tangent_circle_l167_167921


namespace jills_uncles_medicine_last_time_l167_167660

theorem jills_uncles_medicine_last_time :
  let pills := 90
  let third_of_pill_days := 3
  let days_per_full_pill := 9
  let days_per_month := 30
  let total_days := pills * days_per_full_pill
  let total_months := total_days / days_per_month
  total_months = 27 :=
by {
  sorry
}

end jills_uncles_medicine_last_time_l167_167660


namespace smallest_sum_of_consecutive_integers_is_square_l167_167584

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l167_167584


namespace y_exceeds_x_by_35_percent_l167_167045

theorem y_exceeds_x_by_35_percent {x y : ℝ} (h : x = 0.65 * y) : ((y - x) / x) * 100 = 35 :=
by
  sorry

end y_exceeds_x_by_35_percent_l167_167045


namespace sum_of_repeating_decimals_l167_167510

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals_l167_167510


namespace count_perfect_squares_mul_36_l167_167631

theorem count_perfect_squares_mul_36 (n : ℕ) (h1 : n < 10^7) (h2 : ∃k, n = k^2) (h3 : 36 ∣ n) :
  ∃ m : ℕ, m = 263 :=
by
  sorry

end count_perfect_squares_mul_36_l167_167631


namespace dimes_count_l167_167502

-- Definitions of types of coins and their values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def halfDollar := 50

-- Condition statements as assumptions
variables (num_pennies num_nickels num_dimes num_quarters num_halfDollars : ℕ)

-- Sum of all coins and their values (in cents)
def total_value := num_pennies * penny + num_nickels * nickel + num_dimes * dime + num_quarters * quarter + num_halfDollars * halfDollar

-- Total number of coins
def total_coins := num_pennies + num_nickels + num_dimes + num_quarters + num_halfDollars

-- Proving the number of dimes is 5 given the conditions.
theorem dimes_count : 
  total_value = 163 ∧ 
  total_coins = 12 ∧ 
  num_pennies ≥ 1 ∧ 
  num_nickels ≥ 1 ∧ 
  num_dimes ≥ 1 ∧ 
  num_quarters ≥ 1 ∧ 
  num_halfDollars ≥ 1 → 
  num_dimes = 5 :=
by
  sorry

end dimes_count_l167_167502


namespace cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l167_167734

-- Definitions related to the problem statements
def good_tetrahedron (V S : ℝ) := V = S

def good_parallelepiped (V' S1 S2 S3 : ℝ) := V' = 2 * (S1 + S2 + S3)

-- Theorem statement
theorem cannot_inscribe_good_tetrahedron_in_good_parallelepiped
  (V V' S : ℝ) (S1 S2 S3 : ℝ) (h1 h2 h3 : ℝ)
  (HT : good_tetrahedron V S)
  (HP : good_parallelepiped V' S1 S2 S3)
  (Hheights : S1 ≥ S2 ∧ S2 ≥ S3) :
  ¬ (V = S ∧ V' = 2 * (S1 + S2 + S3) ∧ h1 > 6 * S1 ∧ h2 > 6 * S2 ∧ h3 > 6 * S3) := 
sorry

end cannot_inscribe_good_tetrahedron_in_good_parallelepiped_l167_167734


namespace scientific_notation_280000_l167_167101

theorem scientific_notation_280000 : 
  ∃ n: ℝ, n * 10^5 = 280000 ∧ n = 2.8 :=
by
-- our focus is on the statement outline, thus we use sorry to skip the proof part
  sorry

end scientific_notation_280000_l167_167101


namespace find_abc_sol_l167_167100

theorem find_abc_sol (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (1 / ↑a + 1 / ↑b + 1 / ↑c = 1) →
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end find_abc_sol_l167_167100


namespace high_quality_chip_prob_l167_167500

variable (chipsA chipsB chipsC : ℕ)
variable (qualityA qualityB qualityC : ℝ)
variable (totalChips : ℕ)

noncomputable def probability_of_high_quality_chip (chipsA chipsB chipsC : ℕ) (qualityA qualityB qualityC : ℝ) (totalChips : ℕ) : ℝ :=
  (chipsA / totalChips) * qualityA + (chipsB / totalChips) * qualityB + (chipsC / totalChips) * qualityC

theorem high_quality_chip_prob :
  let chipsA := 5
  let chipsB := 10
  let chipsC := 10
  let qualityA := 0.8
  let qualityB := 0.8
  let qualityC := 0.7
  let totalChips := 25
  probability_of_high_quality_chip chipsA chipsB chipsC qualityA qualityB qualityC totalChips = 0.76 :=
by
  sorry

end high_quality_chip_prob_l167_167500


namespace problem_l167_167365

theorem problem (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 3 :=
by
  sorry

end problem_l167_167365


namespace dany_farm_bushels_l167_167170

theorem dany_farm_bushels :
  let cows := 5
  let cows_bushels_per_day := 3
  let sheep := 4
  let sheep_bushels_per_day := 2
  let chickens := 8
  let chickens_bushels_per_day := 1
  let pigs := 6
  let pigs_bushels_per_day := 4
  let horses := 2
  let horses_bushels_per_day := 5
  cows * cows_bushels_per_day +
  sheep * sheep_bushels_per_day +
  chickens * chickens_bushels_per_day +
  pigs * pigs_bushels_per_day +
  horses * horses_bushels_per_day = 65 := by
  sorry

end dany_farm_bushels_l167_167170


namespace walking_representation_l167_167695

-- Definitions based on conditions
def represents_walking_eastward (m : ℤ) : Prop := m > 0

-- The theorem to prove based on the problem statement
theorem walking_representation :
  represents_walking_eastward 5 →
  ¬ represents_walking_eastward (-10) ∧ abs (-10) = 10 :=
by
  sorry

end walking_representation_l167_167695


namespace Pam_read_more_than_Harrison_l167_167421

theorem Pam_read_more_than_Harrison :
  ∀ (assigned : ℕ) (Harrison : ℕ) (Pam : ℕ) (Sam : ℕ),
    assigned = 25 →
    Harrison = assigned + 10 →
    Sam = 2 * Pam →
    Sam = 100 →
    Pam - Harrison = 15 :=
by
  intros assigned Harrison Pam Sam h1 h2 h3 h4
  sorry

end Pam_read_more_than_Harrison_l167_167421


namespace exists_N_binary_representation_l167_167404

theorem exists_N_binary_representation (n p : ℕ) (h_composite : ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0) (h_proper_divisor : p > 0 ∧ p < n ∧ n % p = 0) :
  ∃ N : ℕ, ((1 + 2^p + 2^(n-p)) * N) % 2^n = 1 % 2^n :=
by
  sorry

end exists_N_binary_representation_l167_167404


namespace tan_of_perpendicular_vectors_l167_167259

theorem tan_of_perpendicular_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (ha : ℝ × ℝ := (Real.cos θ, 2)) (hb : ℝ × ℝ := (-1, Real.sin θ))
  (h_perpendicular : ha.1 * hb.1 + ha.2 * hb.2 = 0) :
  Real.tan θ = 1 / 2 := 
sorry

end tan_of_perpendicular_vectors_l167_167259


namespace a_n_is_perfect_square_l167_167277

def seqs (a b : ℕ → ℤ) : Prop :=
  a 0 = 1 ∧ b 0 = 0 ∧ ∀ n, a (n + 1) = 7 * a n + 6 * b n - 3 ∧ b (n + 1) = 8 * a n + 7 * b n - 4

theorem a_n_is_perfect_square (a b : ℕ → ℤ) (h : seqs a b) :
  ∀ n, ∃ k : ℤ, a n = k^2 :=
by
  sorry

end a_n_is_perfect_square_l167_167277


namespace no_corner_cut_possible_l167_167506

-- Define the cube and the triangle sides
def cube_edge_length : ℝ := 15
def triangle_side1 : ℝ := 5
def triangle_side2 : ℝ := 6
def triangle_side3 : ℝ := 8

-- Main statement: Prove that it's not possible to cut off a corner of the cube to form the given triangle
theorem no_corner_cut_possible :
  ¬ (∃ (a b c : ℝ),
    a^2 + b^2 = triangle_side1^2 ∧
    b^2 + c^2 = triangle_side2^2 ∧
    c^2 + a^2 = triangle_side3^2 ∧
    a^2 + b^2 + c^2 = 62.5) :=
sorry

end no_corner_cut_possible_l167_167506


namespace minimum_value_l167_167190

theorem minimum_value (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 2) :
  (1 / a) + (1 / b) ≥ 2 :=
by {
  sorry
}

end minimum_value_l167_167190


namespace stockholm_to_uppsala_distance_l167_167939

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l167_167939


namespace congruence_from_overlap_l167_167942

-- Definitions used in the conditions
def figure := Type
def equal_area (f1 f2 : figure) : Prop := sorry
def equal_perimeter (f1 f2 : figure) : Prop := sorry
def equilateral_triangle (f : figure) : Prop := sorry
def can_completely_overlap (f1 f2 : figure) : Prop := sorry

-- Theorem that should be proven
theorem congruence_from_overlap (f1 f2 : figure) (h: can_completely_overlap f1 f2) : f1 = f2 := sorry

end congruence_from_overlap_l167_167942


namespace inequality_proof_l167_167237

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
    (x^4 / (y * (1 - y^2))) + (y^4 / (z * (1 - z^2))) + (z^4 / (x * (1 - x^2))) ≥ 1 / 8 :=
sorry

end inequality_proof_l167_167237


namespace f_odd_and_minimum_period_pi_l167_167889

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem f_odd_and_minimum_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end f_odd_and_minimum_period_pi_l167_167889


namespace slope_range_l167_167534

noncomputable def directed_distance (a b c x0 y0 : ℝ) : ℝ :=
  (a * x0 + b * y0 + c) / (Real.sqrt (a^2 + b^2))

theorem slope_range {A B P : ℝ × ℝ} (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : P = (3, 0))
                   {C : ℝ × ℝ} (hC : ∃ θ : ℝ, C = (9 * Real.cos θ, 18 + 9 * Real.sin θ))
                   {a b c : ℝ} (h_line : c = -3 * a)
                   (h_sum_distances : directed_distance a b c (-1) 0 +
                                      directed_distance a b c 1 0 +
                                      directed_distance a b c (9 * Real.cos θ) (18 + 9 * Real.sin θ) = 0) :
  -3 ≤ - (a / b) ∧ - (a / b) ≤ -1 := sorry

end slope_range_l167_167534


namespace complement_union_correct_l167_167147

noncomputable def U : Set ℕ := {2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {x | x^2 - 6*x + 8 = 0}
noncomputable def B : Set ℕ := {2, 5, 6}

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 5, 6} := 
by
  sorry

end complement_union_correct_l167_167147


namespace park_area_l167_167679

theorem park_area (w : ℝ) (h1 : 2 * (w + 3 * w) = 72) : w * (3 * w) = 243 :=
by
  sorry

end park_area_l167_167679


namespace point_N_in_second_quadrant_l167_167700

theorem point_N_in_second_quadrant (a b : ℝ) (h1 : 1 + a < 0) (h2 : 2 * b - 1 < 0) :
    (a - 1 < 0) ∧ (1 - 2 * b > 0) :=
by
  -- Insert proof here
  sorry

end point_N_in_second_quadrant_l167_167700


namespace solve_m_correct_l167_167155

noncomputable def solve_for_m (Q t h : ℝ) : ℝ :=
  if h >= 0 ∧ Q > 0 ∧ t > 0 then
    (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h))
  else
    0 -- Define default output for invalid inputs

theorem solve_m_correct (Q t h : ℝ) (m : ℝ) :
  Q = t / (1 + Real.sqrt h)^m → m = (Real.log (t / Q)) / (Real.log (1 + Real.sqrt h)) :=
by
  intros h1
  rw [h1]
  sorry

end solve_m_correct_l167_167155


namespace percentage_error_in_area_l167_167540

theorem percentage_error_in_area (s : ℝ) (h_s_pos: s > 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 2.01 :=
by
  sorry

end percentage_error_in_area_l167_167540


namespace prime_diff_of_cubes_sum_of_square_and_three_times_square_l167_167568

theorem prime_diff_of_cubes_sum_of_square_and_three_times_square 
  (p : ℕ) (a b : ℕ) (h_prime : Nat.Prime p) (h_diff : p = a^3 - b^3) :
  ∃ c d : ℤ, p = c^2 + 3 * d^2 := 
  sorry

end prime_diff_of_cubes_sum_of_square_and_three_times_square_l167_167568


namespace largest_value_of_d_l167_167161

noncomputable def maximum_possible_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : ℝ :=
  (5 + Real.sqrt 123) / 2

theorem largest_value_of_d 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 17) : 
  d ≤ maximum_possible_value_of_d a b c d h1 h2 :=
sorry

end largest_value_of_d_l167_167161


namespace fractional_inequality_solution_l167_167338

theorem fractional_inequality_solution :
  {x : ℝ | (2 * x - 1) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 1 / 2} := 
by
  sorry

end fractional_inequality_solution_l167_167338


namespace shaded_triangle_ratio_is_correct_l167_167483

noncomputable def ratio_of_shaded_triangle_to_large_square (total_area : ℝ) 
  (midpoint_area_ratio : ℝ := 1 / 24) : ℝ :=
  midpoint_area_ratio * total_area

theorem shaded_triangle_ratio_is_correct 
  (shaded_area total_area : ℝ)
  (n : ℕ)
  (h1 : n = 36)
  (grid_area : ℝ)
  (condition1 : grid_area = total_area / n)
  (condition2 : shaded_area = grid_area / 2 * 3)
  : shaded_area / total_area = 1 / 24 :=
by
  sorry

end shaded_triangle_ratio_is_correct_l167_167483


namespace count_consecutive_integers_l167_167924

theorem count_consecutive_integers : 
  ∃ n : ℕ, (∀ x : ℕ, (1 < x ∧ x < 111) → (x - 1) + x + (x + 1) < 333) ∧ n = 109 := 
  by
    sorry

end count_consecutive_integers_l167_167924


namespace max_value_of_k_l167_167555

theorem max_value_of_k (n : ℕ) (k : ℕ) (h : 3^11 = k * (2 * n + k + 1) / 2) : k = 486 :=
sorry

end max_value_of_k_l167_167555


namespace train_speed_l167_167952

theorem train_speed (v : ℝ) (h1 : 60 * 6.5 + v * 6.5 = 910) : v = 80 := 
sorry

end train_speed_l167_167952


namespace medication_price_reduction_l167_167017

variable (a : ℝ)

theorem medication_price_reduction (h : 0.60 * x = a) : x = 5/3 * a := by
  sorry

end medication_price_reduction_l167_167017


namespace ratio_x_y_l167_167759

-- Definitions based on conditions
variables (a b c x y : ℝ) 

-- Conditions
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2)
def a_b_ratio (a b : ℝ) := (a / b = 2 / 5)
def segments_ratio (a b c x y : ℝ) := (x = a^2 / c) ∧ (y = b^2 / c)
def perpendicular_division (x y a b : ℝ) := ((a^2 / x) = c) ∧ ((b^2 / y) = c)

-- The proof statement we need
theorem ratio_x_y : 
  ∀ (a b c x y : ℝ),
    right_triangle a b c → 
    a_b_ratio a b → 
    segments_ratio a b c x y → 
    (x / y = 4 / 25) :=
by sorry

end ratio_x_y_l167_167759


namespace stack_crates_height_l167_167104

theorem stack_crates_height :
  ∀ a b c : ℕ, (3 * a + 4 * b + 5 * c = 50) ∧ (a + b + c = 12) → false :=
by
  sorry

end stack_crates_height_l167_167104


namespace johns_total_working_hours_l167_167098

theorem johns_total_working_hours (d h t : Nat) (h_d : d = 5) (h_h : h = 8) : t = d * h := by
  rewrite [h_d, h_h]
  sorry

end johns_total_working_hours_l167_167098


namespace quadratic_factoring_even_a_l167_167479

theorem quadratic_factoring_even_a (a : ℤ) :
  (∃ (m p n q : ℤ), 21 * x^2 + a * x + 21 = (m * x + n) * (p * x + q) ∧ m * p = 21 ∧ n * q = 21 ∧ (∃ (k : ℤ), a = 2 * k)) :=
sorry

end quadratic_factoring_even_a_l167_167479


namespace price_for_70_cans_is_correct_l167_167859

def regular_price_per_can : ℝ := 0.55
def discount_percentage : ℝ := 0.25
def purchase_quantity : ℕ := 70

def discount_per_can : ℝ := discount_percentage * regular_price_per_can
def discounted_price_per_can : ℝ := regular_price_per_can - discount_per_can

def price_for_72_cans : ℝ := 72 * discounted_price_per_can
def price_for_2_cans : ℝ := 2 * discounted_price_per_can

def final_price_for_70_cans : ℝ := price_for_72_cans - price_for_2_cans

theorem price_for_70_cans_is_correct
    (regular_price_per_can : ℝ := 0.55)
    (discount_percentage : ℝ := 0.25)
    (purchase_quantity : ℕ := 70)
    (disc_per_can : ℝ := discount_percentage * regular_price_per_can)
    (disc_price_per_can : ℝ := regular_price_per_can - disc_per_can)
    (price_72_cans : ℝ := 72 * disc_price_per_can)
    (price_2_cans : ℝ := 2 * disc_price_per_can):
    final_price_for_70_cans = 28.875 :=
by
  sorry

end price_for_70_cans_is_correct_l167_167859


namespace number_of_people_l167_167399

open Nat

theorem number_of_people (n : ℕ) (h : n^2 = 100) : n = 10 := by
  sorry

end number_of_people_l167_167399


namespace amount_of_flour_per_large_tart_l167_167321

-- Statement without proof
theorem amount_of_flour_per_large_tart 
  (num_small_tarts : ℕ) (flour_per_small_tart : ℚ) 
  (num_large_tarts : ℕ) (total_flour : ℚ) 
  (h1 : num_small_tarts = 50) 
  (h2 : flour_per_small_tart = 1/8) 
  (h3 : num_large_tarts = 25) 
  (h4 : total_flour = num_small_tarts * flour_per_small_tart) : 
  total_flour = num_large_tarts * (1/4) := 
sorry

end amount_of_flour_per_large_tart_l167_167321


namespace smaller_consecutive_number_divisibility_l167_167796

theorem smaller_consecutive_number_divisibility :
  ∃ (m : ℕ), (m < m + 1) ∧ (1 ≤ m ∧ m ≤ 200) ∧ (1 ≤ m + 1 ∧ m + 1 ≤ 200) ∧
              (∀ n, (1 ≤ n ∧ n ≤ 200 ∧ n ≠ m ∧ n ≠ m + 1) → ∃ k, chosen_num = k * n) ∧
              (128 = m) :=
sorry

end smaller_consecutive_number_divisibility_l167_167796


namespace molecular_weight_of_1_mole_l167_167653

theorem molecular_weight_of_1_mole (m : ℝ) (w : ℝ) (h : 7 * m = 420) : m = 60 :=
by
  sorry

end molecular_weight_of_1_mole_l167_167653


namespace collinear_dot_probability_computation_l167_167117

def collinear_dot_probability : ℚ := 12 / Nat.choose 25 5

theorem collinear_dot_probability_computation :
  collinear_dot_probability = 12 / 53130 :=
by
  -- This is where the proof steps would be if provided.
  sorry

end collinear_dot_probability_computation_l167_167117


namespace star_compound_l167_167019

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_compound : star (star 3 11) 6 = 2.375 := by
  sorry

end star_compound_l167_167019


namespace k_interval_l167_167025

noncomputable def f (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem k_interval (k : ℝ) :
  (∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x k = 0) ↔ (2 < k ∧ k < 3) :=
by
  sorry

end k_interval_l167_167025


namespace smallest_n_l167_167078

theorem smallest_n (n : ℕ) (h1: n ≥ 100) (h2: n ≤ 999) 
  (h3: (n + 5) % 8 = 0) (h4: (n - 8) % 5 = 0) : 
  n = 123 :=
sorry

end smallest_n_l167_167078


namespace goals_even_more_likely_l167_167941

theorem goals_even_more_likely (p_1 : ℝ) (q_1 : ℝ) (h1 : p_1 + q_1 = 1) :
  let p := p_1^2 + q_1^2 
  let q := 2 * p_1 * q_1
  p ≥ q := by
    sorry

end goals_even_more_likely_l167_167941


namespace binom_mult_l167_167636

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l167_167636


namespace express_in_scientific_notation_l167_167754

-- Definition for expressing number in scientific notation
def scientific_notation (n : ℝ) (a : ℝ) (b : ℕ) : Prop :=
  n = a * 10 ^ b

-- Condition of the problem
def condition : ℝ := 1300000

-- Stating the theorem to be proved
theorem express_in_scientific_notation : scientific_notation condition 1.3 6 :=
by
  -- Placeholder for the proof
  sorry

end express_in_scientific_notation_l167_167754


namespace vector_parallel_solution_l167_167122

-- Define the vectors and the condition
def a (m : ℝ) := (2 * m + 1, 3)
def b (m : ℝ) := (2, m)

-- The proof problem statement
theorem vector_parallel_solution (m : ℝ) :
  (2 * m + 1) * m = 3 * 2 ↔ m = 3 / 2 ∨ m = -2 :=
by
  sorry

end vector_parallel_solution_l167_167122


namespace money_left_after_purchases_is_correct_l167_167871

noncomputable def initial_amount : ℝ := 12.50
noncomputable def cost_pencil : ℝ := 1.25
noncomputable def cost_notebook : ℝ := 3.45
noncomputable def cost_pens : ℝ := 4.80

noncomputable def total_cost : ℝ := cost_pencil + cost_notebook + cost_pens
noncomputable def money_left : ℝ := initial_amount - total_cost

theorem money_left_after_purchases_is_correct : money_left = 3.00 :=
by
  -- proof goes here, skipping with sorry for now
  sorry

end money_left_after_purchases_is_correct_l167_167871


namespace area_S4_is_3_125_l167_167193

theorem area_S4_is_3_125 (S_1 : Type) (area_S1 : ℝ) 
  (hS1 : area_S1 = 25)
  (bisect_and_construct : ∀ (S : Type) (area : ℝ),
    ∃ S' : Type, ∃ area' : ℝ, area' = area / 2) :
  ∃ S_4 : Type, ∃ area_S4 : ℝ, area_S4 = 3.125 :=
by
  sorry

end area_S4_is_3_125_l167_167193


namespace intersection_proof_l167_167886

-- Definitions based on conditions
def circle1 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 10) ^ 2 = 50
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 + 2 * (x - y) - 18 = 0

-- Correct answer tuple
def intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 3), (-3, 5))

-- The goal statement to prove
theorem intersection_proof :
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by
  sorry

end intersection_proof_l167_167886


namespace mileage_per_gallon_l167_167566

-- Definitions for the conditions
def total_distance_to_grandma (d : ℕ) : Prop := d = 100
def gallons_to_grandma (g : ℕ) : Prop := g = 5

-- The statement to be proved
theorem mileage_per_gallon :
  ∀ (d g m : ℕ), total_distance_to_grandma d → gallons_to_grandma g → m = d / g → m = 20 :=
sorry

end mileage_per_gallon_l167_167566


namespace least_possible_number_l167_167600

theorem least_possible_number (k : ℕ) (n : ℕ) (r : ℕ) (h1 : k = 34 * n + r) 
  (h2 : k / 5 = r + 8) (h3 : r < 34) : k = 68 :=
by
  -- Proof to be filled
  sorry

end least_possible_number_l167_167600


namespace sequence_is_arithmetic_max_value_a_n_b_n_l167_167760

open Real

theorem sequence_is_arithmetic (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (Sn : ℕ → ℝ) 
  (h_Sn : ∀ n, Sn n = (a n ^ 2 + a n) / 2) :
    ∀ n, a n = n := sorry 

theorem max_value_a_n_b_n (a b : ℕ → ℝ)
  (h_b : ∀ n, b n = - n + 5)
  (h_a : ∀ n, a n = n) :
    ∀ n, n ≥ 2 → n ≤ 3 → 
    ∃ k, a k * b k = 25 / 4 := by 
      sorry

end sequence_is_arithmetic_max_value_a_n_b_n_l167_167760


namespace min_sum_of_factors_l167_167579

theorem min_sum_of_factors (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 3432) :
  a + b + c ≥ 56 :=
sorry

end min_sum_of_factors_l167_167579


namespace part1_solution_set_part2_range_a_l167_167549

noncomputable def f (x a : ℝ) := 5 - abs (x + a) - abs (x - 2)

-- Part 1
theorem part1_solution_set (x : ℝ) (a : ℝ) (h : a = 1) :
  (f x a ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 3) := sorry

-- Part 2
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a ≤ 1) ↔ (a ≤ -6 ∨ a ≥ 2) := sorry

end part1_solution_set_part2_range_a_l167_167549


namespace smallest_solution_abs_eq_20_l167_167638

theorem smallest_solution_abs_eq_20 : ∃ x : ℝ, x = -7 ∧ |4 * x + 8| = 20 ∧ (∀ y : ℝ, |4 * y + 8| = 20 → x ≤ y) :=
by
  sorry

end smallest_solution_abs_eq_20_l167_167638


namespace domain_of_f_x_plus_2_l167_167979

theorem domain_of_f_x_plus_2 (f : ℝ → ℝ) (dom_f_x_minus_1 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ x-1 ∧ x-1 ≤ 1) :
  ∀ y, 0 ≤ y ∧ y ≤ 1 ↔ -2 ≤ y-2 ∧ y-2 ≤ -1 :=
by
  sorry

end domain_of_f_x_plus_2_l167_167979


namespace train_speed_is_45_kmph_l167_167809

noncomputable def speed_of_train_kmph (train_length bridge_length total_time : ℕ) : ℕ :=
  let total_distance := train_length + bridge_length
  let speed_mps := total_distance / total_time
  let speed_kmph := speed_mps * 36 / 10
  speed_kmph

theorem train_speed_is_45_kmph :
  speed_of_train_kmph 150 225 30 = 45 :=
  sorry

end train_speed_is_45_kmph_l167_167809


namespace johns_daily_calorie_intake_l167_167073

variable (breakfast lunch dinner shake : ℕ)
variable (num_shakes meals_per_day : ℕ)
variable (lunch_inc : ℕ)
variable (dinner_mult : ℕ)

-- Define the conditions from the problem
def john_calories_per_day 
  (breakfast := 500)
  (lunch := breakfast + lunch_inc)
  (dinner := lunch * dinner_mult)
  (shake := 300)
  (num_shakes := 3)
  (lunch_inc := breakfast / 4)
  (dinner_mult := 2)
  : ℕ :=
  breakfast + lunch + dinner + (shake * num_shakes)

theorem johns_daily_calorie_intake : john_calories_per_day = 3275 := by
  sorry

end johns_daily_calorie_intake_l167_167073


namespace number_of_boundaries_l167_167216

def total_runs : ℕ := 120
def sixes : ℕ := 4
def runs_per_six : ℕ := 6
def percentage_runs_by_running : ℚ := 0.60
def runs_per_boundary : ℕ := 4

theorem number_of_boundaries :
  let runs_by_running := (percentage_runs_by_running * total_runs : ℚ)
  let runs_by_sixes := (sixes * runs_per_six)
  let runs_by_boundaries := (total_runs - runs_by_running - runs_by_sixes : ℚ)
  (runs_by_boundaries / runs_per_boundary) = 6 := by
  sorry

end number_of_boundaries_l167_167216


namespace students_number_l167_167577

theorem students_number (x a o : ℕ)
  (h1 : o = 3 * a + 3)
  (h2 : a = 2 * x + 6)
  (h3 : o = 7 * x - 5) :
  x = 26 :=
by sorry

end students_number_l167_167577


namespace max_visible_cubes_from_point_l167_167851

theorem max_visible_cubes_from_point (n : ℕ) (h : n = 12) :
  let total_cubes := n^3
  let face_cube_count := n * n
  let edge_count := n
  let visible_face_count := 3 * face_cube_count
  let double_counted_edges := 3 * (edge_count - 1)
  let corner_cube_count := 1
  visible_face_count - double_counted_edges + corner_cube_count = 400 := by
  sorry

end max_visible_cubes_from_point_l167_167851


namespace eliot_account_balance_l167_167042

-- Definitions for the conditions
variables {A E : ℝ}

--- Conditions rephrased into Lean:
-- 1. Al has more money than Eliot.
def al_more_than_eliot (A E : ℝ) : Prop := A > E

-- 2. The difference between their two accounts is 1/12 of the sum of their two accounts.
def difference_condition (A E : ℝ) : Prop := A - E = (1 / 12) * (A + E)

-- 3. If Al's account were to increase by 10% and Eliot's account were to increase by 15%, 
--     then Al would have exactly $22 more than Eliot in his account.
def percentage_increase_condition (A E : ℝ) : Prop := 1.10 * A = 1.15 * E + 22

-- Prove the total statement
theorem eliot_account_balance : 
  ∀ (A E : ℝ), al_more_than_eliot A E → difference_condition A E → percentage_increase_condition A E → E = 146.67 :=
by
  intros A E h1 h2 h3
  sorry

end eliot_account_balance_l167_167042


namespace exp_inequality_l167_167096

theorem exp_inequality (n : ℕ) (h : 0 < n) : 2 ≤ (1 + 1 / (n : ℝ)) ^ n ∧ (1 + 1 / (n : ℝ)) ^ n < 3 :=
sorry

end exp_inequality_l167_167096


namespace fraction_checked_by_worker_y_l167_167228

variable (P : ℝ) -- Total number of products
variable (f_X f_Y : ℝ) -- Fraction of products checked by worker X and Y
variable (dx : ℝ) -- Defective rate for worker X
variable (dy : ℝ) -- Defective rate for worker Y
variable (dt : ℝ) -- Total defective rate

-- Conditions
axiom f_sum : f_X + f_Y = 1
axiom dx_val : dx = 0.005
axiom dy_val : dy = 0.008
axiom dt_val : dt = 0.0065

-- Proof
theorem fraction_checked_by_worker_y : f_Y = 1 / 2 :=
by
  sorry

end fraction_checked_by_worker_y_l167_167228


namespace average_percentage_of_first_20_percent_l167_167793

theorem average_percentage_of_first_20_percent (X : ℝ) 
  (h1 : 0.20 * X + 0.50 * 60 + 0.30 * 40 = 58) : 
  X = 80 :=
sorry

end average_percentage_of_first_20_percent_l167_167793


namespace pentagon_area_correct_l167_167556

-- Define the side lengths of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 22

-- Define the specific angle between the sides of lengths 30 and 28
def angle := 110 -- degrees

-- Define the heights used for the trapezoids and triangle calculations
def height_trapezoid1 := 10
def height_trapezoid2 := 15
def height_triangle := 8

-- Function to calculate the area of a trapezoid
def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

-- Function to calculate the area of a triangle
def triangle_area (base height : ℕ) : ℕ :=
  base * height / 2

-- Calculation of individual areas
def area_trapezoid1 := trapezoid_area side1 side2 height_trapezoid1
def area_trapezoid2 := trapezoid_area side3 side4 height_trapezoid2
def area_triangle := triangle_area side5 height_triangle

-- Total area calculation
def total_area := area_trapezoid1 + area_trapezoid2 + area_triangle

-- Expected total area
def expected_area := 738

-- Lean statement to assert the total area equals the expected value
theorem pentagon_area_correct :
  total_area = expected_area :=
by sorry

end pentagon_area_correct_l167_167556


namespace area_to_be_painted_correct_l167_167239

-- Define the dimensions and areas involved
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def painting_height : ℕ := 2
def painting_length : ℕ := 2

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def painting_area : ℕ := painting_height * painting_length
def area_not_painted : ℕ := window_area + painting_area
def area_to_be_painted : ℕ := wall_area - area_not_painted

-- Theorem: The area to be painted is 131 square feet
theorem area_to_be_painted_correct : area_to_be_painted = 131 := by
  sorry

end area_to_be_painted_correct_l167_167239


namespace adam_spent_on_ferris_wheel_l167_167850

theorem adam_spent_on_ferris_wheel (t_initial t_left t_price : ℕ) (h1 : t_initial = 13)
  (h2 : t_left = 4) (h3 : t_price = 9) : t_initial - t_left = 9 ∧ (t_initial - t_left) * t_price = 81 := 
by
  sorry

end adam_spent_on_ferris_wheel_l167_167850


namespace n_leq_84_l167_167128

theorem n_leq_84 (n : ℕ) (hn : 0 < n) (h: (1 / 2 + 1 / 3 + 1 / 7 + 1 / ↑n : ℚ).den ≤ 1): n ≤ 84 :=
sorry

end n_leq_84_l167_167128


namespace stewart_farm_horse_food_l167_167747

def sheep_to_horse_ratio := 3 / 7
def horses_needed (sheep : ℕ) := (sheep * 7) / 3 
def daily_food_per_horse := 230
def sheep_count := 24
def total_horses := horses_needed sheep_count
def total_daily_horse_food := total_horses * daily_food_per_horse

theorem stewart_farm_horse_food : total_daily_horse_food = 12880 := by
  have num_horses : horses_needed 24 = 56 := by
    unfold horses_needed
    sorry -- Omitted for brevity, this would be solved

  have food_needed : 56 * 230 = 12880 := by
    sorry -- Omitted for brevity, this would be solved

  exact food_needed

end stewart_farm_horse_food_l167_167747


namespace irreducible_positive_fraction_unique_l167_167480

theorem irreducible_positive_fraction_unique
  (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_gcd : Nat.gcd a b = 1)
  (h_eq : (a + 12) * b = 3 * a * (b + 12)) :
  a = 2 ∧ b = 9 :=
by
  sorry

end irreducible_positive_fraction_unique_l167_167480


namespace find_number_l167_167199

theorem find_number (x : ℝ) (h : x = 0.16 * x + 21) : x = 25 :=
by
  sorry

end find_number_l167_167199


namespace identical_digit_square_l167_167355

theorem identical_digit_square {b x y : ℕ} (hb : b ≥ 2) (hx : x < b) (hy : y < b) (hx_pos : x ≠ 0) (hy_pos : y ≠ 0) :
  (x * b + x)^2 = y * b^3 + y * b^2 + y * b + y ↔ b = 7 :=
by
  sorry

end identical_digit_square_l167_167355


namespace work_completion_days_l167_167694

theorem work_completion_days (D : ℕ) (W : ℕ) :
  (D : ℕ) = 6 :=
by 
  -- define constants and given conditions
  let original_men := 10
  let additional_men := 10
  let early_days := 3

  -- define the premise
  -- work done with original men in original days
  have work_done_original : W = (original_men * D) := sorry
  -- work done with additional men in reduced days
  have work_done_with_additional : W = ((original_men + additional_men) * (D - early_days)) := sorry

  -- prove the equality from the condition
  have eq : original_men * D = (original_men + additional_men) * (D - early_days) := sorry

  -- simplify to solve for D
  have solution : D = 6 := sorry

  exact solution

end work_completion_days_l167_167694


namespace solve_equation_l167_167528

theorem solve_equation (x : ℤ) (h1 : x ≠ 2) : x - 8 / (x - 2) = 5 - 8 / (x - 2) → x = 5 := by
  sorry

end solve_equation_l167_167528


namespace tape_length_division_l167_167701

theorem tape_length_division (n_pieces : ℕ) (length_piece overlap : ℝ) (n_parts : ℕ) 
  (h_pieces : n_pieces = 5) (h_length : length_piece = 2.7) (h_overlap : overlap = 0.3) 
  (h_parts : n_parts = 6) : 
  ((n_pieces * length_piece) - ((n_pieces - 1) * overlap)) / n_parts = 2.05 :=
  by
    sorry

end tape_length_division_l167_167701


namespace sin_beta_value_l167_167171

theorem sin_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h1 : Real.cos α = 4 / 5) (h2 : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_value_l167_167171


namespace Sam_has_walked_25_miles_l167_167446

variables (d : ℕ) (v_fred v_sam : ℕ)

def Fred_and_Sam_meet (d : ℕ) (v_fred v_sam : ℕ) := 
  d / (v_fred + v_sam) * v_sam

theorem Sam_has_walked_25_miles :
  Fred_and_Sam_meet 50 5 5 = 25 :=
by
  sorry

end Sam_has_walked_25_miles_l167_167446


namespace simplify_and_evaluate_l167_167843

theorem simplify_and_evaluate (a : ℚ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5 / 3 :=
by
  sorry

end simplify_and_evaluate_l167_167843


namespace fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l167_167682

section FoldingNumberLine

-- Part (1)
def coincides_point_3_if_minus2_2_fold (x : ℝ) : Prop :=
  x = -3

theorem fold_minus2_2_3_coincides_neg3 :
  coincides_point_3_if_minus2_2_fold 3 :=
by
  sorry

-- Part (2) ①
def coincides_point_7_if_minus1_3_fold (x : ℝ) : Prop :=
  x = -5

theorem fold_minus1_3_7_coincides_neg5 :
  coincides_point_7_if_minus1_3_fold 7 :=
by
  sorry

-- Part (2) ②
def B_position_after_folding (m : ℝ) (h : m > 0) (A B : ℝ) : Prop :=
  B = 1 + m / 2

theorem fold_distanceA_to_B_coincide (m : ℝ) (h : m > 0) (A B : ℝ) :
  B_position_after_folding m h A B :=
by
  sorry

end FoldingNumberLine

end fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l167_167682


namespace equation1_solution_equation2_solution_l167_167091

variable (x : ℝ)

theorem equation1_solution :
  ((2 * x - 5) / 6 - (3 * x + 1) / 2 = 1) → (x = -2) :=
by
  sorry

theorem equation2_solution :
  (3 * x - 7 * (x - 1) = 3 - 2 * (x + 3)) → (x = 5) :=
by
  sorry

end equation1_solution_equation2_solution_l167_167091


namespace quadratic_properties_l167_167624

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  -- 1. The parabola opens downwards.
  (∀ x : ℝ, quadratic_function x < quadratic_function (x + 1) → false) ∧
  -- 2. The axis of symmetry is x = 1.
  (∀ x : ℝ, ∃ y : ℝ, quadratic_function x = quadratic_function y → x = y ∨ x + y = 2) ∧
  -- 3. The vertex coordinates are (1, 5).
  (quadratic_function 1 = 5) ∧
  -- 4. y decreases for x > 1.
  (∀ x : ℝ, x > 1 → quadratic_function x < quadratic_function (x - 1)) :=
by
  sorry

end quadratic_properties_l167_167624


namespace find_pairs_l167_167396

-- Define predicative statements for the conditions
def is_integer (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = n

def condition1 (m n : ℕ) : Prop := 
  (n^2 + 1) % (2 * m) = 0

def condition2 (m n : ℕ) : Prop := 
  is_integer (Real.sqrt (2^(n-1) + m + 4))

-- The goal is to find the pairs of positive integers
theorem find_pairs (m n : ℕ) (h1: condition1 m n) (h2: condition2 m n) : 
  (m = 61 ∧ n = 11) :=
sorry

end find_pairs_l167_167396


namespace mistaken_multiplier_is_34_l167_167467

-- Define the main conditions
def correct_number : ℕ := 135
def correct_multiplier : ℕ := 43
def difference : ℕ := 1215

-- Define what we need to prove
theorem mistaken_multiplier_is_34 :
  (correct_number * correct_multiplier - correct_number * x = difference) →
  x = 34 :=
by
  sorry

end mistaken_multiplier_is_34_l167_167467


namespace distinct_placements_of_two_pieces_l167_167841

-- Definitions of the conditions
def grid_size : ℕ := 3
def cell_count : ℕ := grid_size * grid_size
def pieces_count : ℕ := 2

-- The theorem statement
theorem distinct_placements_of_two_pieces : 
  (number_of_distinct_placements : ℕ) = 10 := by
  -- Proof goes here with calculations and accounting for symmetry
  sorry

end distinct_placements_of_two_pieces_l167_167841


namespace min_workers_to_profit_l167_167332

/-- Definitions of constants used in the problem. --/
def daily_maintenance_cost : ℕ := 500
def wage_per_hour : ℕ := 20
def widgets_per_hour_per_worker : ℕ := 5
def sell_price_per_widget : ℕ := 350 / 100 -- since the input is 3.50
def workday_hours : ℕ := 8

/-- Profit condition: the revenue should be greater than the cost. 
    The problem specifies that the number of workers must be at least 26 to make a profit. --/

theorem min_workers_to_profit (n : ℕ) :
  (widgets_per_hour_per_worker * workday_hours * sell_price_per_widget * n > daily_maintenance_cost + (workday_hours * wage_per_hour * n)) → n ≥ 26 :=
sorry


end min_workers_to_profit_l167_167332


namespace polynomial_has_real_root_l167_167391

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, x^3 + b * x^2 - 4 * x + b = 0 := 
sorry

end polynomial_has_real_root_l167_167391


namespace probability_correct_l167_167293

variable (new_balls old_balls total_balls : ℕ)

-- Define initial conditions
def initial_conditions (new_balls old_balls : ℕ) : Prop :=
  new_balls = 4 ∧ old_balls = 2

-- Define total number of balls in the box
def total_balls_condition (new_balls old_balls total_balls : ℕ) : Prop :=
  total_balls = new_balls + old_balls ∧ total_balls = 6

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of picking one new ball and one old ball
def probability_one_new_one_old (new_balls old_balls total_balls : ℕ) : ℚ :=
  (combination new_balls 1 * combination old_balls 1) / (combination total_balls 2)

-- The theorem to prove the probability
theorem probability_correct (new_balls old_balls total_balls : ℕ)
  (h_initial : initial_conditions new_balls old_balls)
  (h_total : total_balls_condition new_balls old_balls total_balls) :
  probability_one_new_one_old new_balls old_balls total_balls = 8 / 15 := by
  sorry

end probability_correct_l167_167293


namespace relationship_among_a_b_c_l167_167740

variable (x y : ℝ)
variable (hx_pos : x > 0) (hy_pos : y > 0) (hxy_ne : x ≠ y)

noncomputable def a := (x + y) / 2
noncomputable def b := Real.sqrt (x * y)
noncomputable def c := 2 / ((1 / x) + (1 / y))

theorem relationship_among_a_b_c :
    a > b ∧ b > c := by
    sorry

end relationship_among_a_b_c_l167_167740


namespace appropriate_term_for_assessment_l167_167044

-- Definitions
def price : Type := String
def value : Type := String
def cost : Type := String
def expense : Type := String

-- Context for assessment of the project
def assessment_context : Type := Π (word : String), word ∈ ["price", "value", "cost", "expense"] → Prop

-- Main Lean statement
theorem appropriate_term_for_assessment (word : String) (h : word ∈ ["price", "value", "cost", "expense"]) :
  word = "value" :=
sorry

end appropriate_term_for_assessment_l167_167044


namespace natasha_dimes_l167_167214

theorem natasha_dimes (n : ℕ) :
  100 < n ∧ n < 200 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2 ↔ n = 182 := by
sorry

end natasha_dimes_l167_167214


namespace maximum_modest_number_l167_167473

-- Definitions and Conditions
def is_modest (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
  5 * a = b + c + d ∧
  d % 2 = 0

def G (a b c d : ℕ) : ℕ :=
  (1000 * a + 100 * b + 10 * c + d - (1000 * c + 100 * d + 10 * a + b)) / 99

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_divisible_by_3 (abc : ℕ) : Prop :=
  abc % 3 = 0

-- Theorem statement
theorem maximum_modest_number :
  ∃ a b c d : ℕ, is_modest a b c d ∧ is_divisible_by_11 (G a b c d) ∧ is_divisible_by_3 (100 * a + 10 * b + c) ∧ 
  (1000 * a + 100 * b + 10 * c + d) = 3816 := 
sorry

end maximum_modest_number_l167_167473


namespace original_time_to_cover_distance_l167_167691

theorem original_time_to_cover_distance (S : ℝ) (T : ℝ) (D : ℝ) :
  (0.8 * S) * (T + 10 / 60) = S * T → T = 2 / 3 :=
  by sorry

end original_time_to_cover_distance_l167_167691


namespace cube_inequality_l167_167303

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l167_167303


namespace odd_function_f_l167_167001

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

theorem odd_function_f :
  odd_function f :=
sorry

end odd_function_f_l167_167001


namespace scramble_language_words_count_l167_167943

theorem scramble_language_words_count :
  let total_words (n : ℕ) := 25 ^ n
  let words_without_B (n : ℕ) := 24 ^ n
  let words_with_B (n : ℕ) := total_words n - words_without_B n
  words_with_B 1 + words_with_B 2 + words_with_B 3 + words_with_B 4 + words_with_B 5 = 1863701 :=
by
  sorry

end scramble_language_words_count_l167_167943


namespace range_of_a_l167_167281

theorem range_of_a (a : ℝ) :  (5 - a > 0) ∧ (a - 2 > 0) ∧ (a - 2 ≠ 1) → (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) :=
by
  intro h
  sorry

end range_of_a_l167_167281


namespace minimum_employment_age_l167_167067

/-- This structure represents the conditions of the problem -/
structure EmploymentConditions where
  jane_current_age : ℕ  -- Jane's current age
  years_until_dara_half_age : ℕ  -- Years until Dara is half Jane's age
  years_until_dara_min_age : ℕ  -- Years until Dara reaches minimum employment age

/-- The proof problem statement -/
theorem minimum_employment_age (conds : EmploymentConditions)
  (h_jane : conds.jane_current_age = 28)
  (h_half_age : conds.years_until_dara_half_age = 6)
  (h_min_age : conds.years_until_dara_min_age = 14) :
  let jane_in_six := conds.jane_current_age + conds.years_until_dara_half_age
  let dara_in_six := jane_in_six / 2
  let dara_now := dara_in_six - conds.years_until_dara_half_age
  let M := dara_now + conds.years_until_dara_min_age
  M = 25 :=
by
  sorry

end minimum_employment_age_l167_167067


namespace add_base6_l167_167178

def base6_to_base10 (n : Nat) : Nat :=
  let rec aux (n : Nat) (exp : Nat) : Nat :=
    match n with
    | 0     => 0
    | n + 1 => aux n (exp + 1) + (n % 6) * (6 ^ exp)
  aux n 0

def base10_to_base6 (n : Nat) : Nat :=
  let rec aux (n : Nat) : Nat :=
    if n = 0 then 0
    else
      let q := n / 6
      let r := n % 6
      r + 10 * aux q
  aux n

theorem add_base6 (a b : Nat) (h1 : base6_to_base10 a = 5) (h2 : base6_to_base10 b = 13) : base10_to_base6 (base6_to_base10 a + base6_to_base10 b) = 30 :=
by
  sorry

end add_base6_l167_167178


namespace find_d_l167_167945

theorem find_d {x d : ℤ} (h : (x + (x + 2) + (x + 4) + (x + 7) + (x + d)) / 5 = (x + 4) + 6) : d = 37 :=
sorry

end find_d_l167_167945


namespace factorize_expression_l167_167657

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l167_167657


namespace solution_exists_unique_l167_167629

theorem solution_exists_unique (x y : ℝ) : (x + y = 2 ∧ x - y = 0) ↔ (x = 1 ∧ y = 1) := 
by
  sorry

end solution_exists_unique_l167_167629


namespace find_y_given_conditions_l167_167723

theorem find_y_given_conditions (k : ℝ) (h1 : ∀ (x y : ℝ), xy = k) (h2 : ∀ (x y : ℝ), x + y = 30) (h3 : ∀ (x y : ℝ), x - y = 10) :
    ∀ x y, x = 8 → y = 25 :=
by
  sorry

end find_y_given_conditions_l167_167723


namespace sampling_method_D_is_the_correct_answer_l167_167428

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer_l167_167428


namespace abs_inequality_no_solution_l167_167547

theorem abs_inequality_no_solution (a : ℝ) : (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) ↔ a ≤ 8 :=
by sorry

end abs_inequality_no_solution_l167_167547


namespace wooden_toys_count_l167_167635

theorem wooden_toys_count :
  ∃ T : ℤ, 
    10 * 40 + 20 * T - (10 * 36 + 17 * T) = 64 ∧ T = 8 :=
by
  use 8
  sorry

end wooden_toys_count_l167_167635


namespace f_minimum_at_l167_167689

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem f_minimum_at : ∀ x : ℝ, x = -Real.log 2 → (∀ y : ℝ, f y ≥ f x) :=
by
  sorry

end f_minimum_at_l167_167689


namespace second_number_is_34_l167_167360

theorem second_number_is_34 (x y z : ℝ) (h1 : x + y + z = 120) 
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 34 :=
by 
  sorry

end second_number_is_34_l167_167360


namespace student_correct_answers_l167_167267

variable (C I : ℕ)

theorem student_correct_answers :
  C + I = 100 ∧ C - 2 * I = 76 → C = 92 :=
by
  intros h
  sorry

end student_correct_answers_l167_167267


namespace P_roots_implies_Q_square_roots_l167_167749

noncomputable def P (x : ℝ) : ℝ := x^3 - 2 * x + 1

noncomputable def Q (x : ℝ) : ℝ := x^3 - 4 * x^2 + 4 * x - 1

theorem P_roots_implies_Q_square_roots (r : ℝ) (h : P r = 0) : Q (r^2) = 0 := sorry

end P_roots_implies_Q_square_roots_l167_167749


namespace positive_number_sum_square_eq_210_l167_167644

theorem positive_number_sum_square_eq_210 (x : ℕ) (h1 : x^2 + x = 210) (h2 : 0 < x) (h3 : x < 15) : x = 14 :=
by
  sorry

end positive_number_sum_square_eq_210_l167_167644


namespace coordinates_at_5PM_l167_167455

noncomputable def particle_coords_at_5PM : ℝ × ℝ :=
  let t1 : ℝ := 7  -- 7 AM
  let t2 : ℝ := 9  -- 9 AM
  let t3 : ℝ := 17  -- 5 PM in 24-hour format
  let coord1 : ℝ × ℝ := (1, 2)
  let coord2 : ℝ × ℝ := (3, -2)
  let dx : ℝ := (coord2.1 - coord1.1) / (t2 - t1)
  let dy : ℝ := (coord2.2 - coord1.2) / (t2 - t1)
  (coord2.1 + dx * (t3 - t2), coord2.2 + dy * (t3 - t2))

theorem coordinates_at_5PM
  (t1 t2 t3 : ℝ)
  (coord1 coord2 : ℝ × ℝ)
  (h_t1 : t1 = 7)
  (h_t2 : t2 = 9)
  (h_t3 : t3 = 17)
  (h_coord1 : coord1 = (1, 2))
  (h_coord2 : coord2 = (3, -2))
  (h_dx : (coord2.1 - coord1.1) / (t2 - t1) = 1)
  (h_dy : (coord2.2 - coord1.2) / (t2 - t1) = -2)
  : particle_coords_at_5PM = (11, -18) :=
by
  sorry

end coordinates_at_5PM_l167_167455


namespace largest_A_l167_167830

theorem largest_A (A B C : ℕ) (h1 : A = 7 * B + C) (h2 : B = C) : A ≤ 48 :=
  sorry

end largest_A_l167_167830


namespace find_y_l167_167599

def binary_op (a b c d : Int) : Int × Int := (a + d, b - c)

theorem find_y : ∃ y : Int, (binary_op 3 y 2 0) = (3, 4) ↔ y = 6 := by
  sorry

end find_y_l167_167599


namespace smallest_x_abs_eq_9_l167_167386

theorem smallest_x_abs_eq_9 : ∃ x : ℝ, |x - 4| = 9 ∧ ∀ y : ℝ, |y - 4| = 9 → x ≤ y :=
by
  -- Prove there exists an x such that |x - 4| = 9 and for all y satisfying |y - 4| = 9, x is the minimum.
  sorry

end smallest_x_abs_eq_9_l167_167386


namespace manuscript_pages_l167_167185

theorem manuscript_pages (P : ℝ)
  (h1 : 10 * (0.05 * P) + 10 * 5 = 250) : P = 400 :=
sorry

end manuscript_pages_l167_167185


namespace roots_quadratic_expression_l167_167757

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + 2 * m - 5 = 0) (h2 : n^2 + 2 * n - 5 = 0) 
  (sum_roots : m + n = -2) (product_roots : m * n = -5) : m^2 + m * n + 3 * m + n = -2 :=
sorry

end roots_quadratic_expression_l167_167757


namespace pow_mul_eq_add_l167_167432

theorem pow_mul_eq_add (a : ℝ) : a^3 * a^4 = a^7 :=
by
  -- This is where the proof would go.
  sorry

end pow_mul_eq_add_l167_167432


namespace find_a_l167_167041

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : (6 * a * (-1) + 6) = 4) : 
  a = 10 / 3 :=
by {
  sorry
}

end find_a_l167_167041


namespace dan_picked_l167_167274

-- Definitions:
def benny_picked : Nat := 2
def total_picked : Nat := 11

-- Problem statement:
theorem dan_picked (b : Nat) (t : Nat) (d : Nat) (h1 : b = benny_picked) (h2 : t = total_picked) (h3 : t = b + d) : d = 9 := by
  sorry

end dan_picked_l167_167274


namespace relay_team_permutations_l167_167251

-- Definitions of conditions
def runners := ["Tony", "Leah", "Nina"]
def fixed_positions := ["Maria runs the third lap", "Jordan runs the fifth lap"]

-- Proof statement
theorem relay_team_permutations : 
  ∃ permutations, permutations = 6 := by
sorry

end relay_team_permutations_l167_167251


namespace simplify_and_evaluate_expression_l167_167887

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.pi^0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expression_l167_167887


namespace largest_sum_of_distinct_factors_of_1764_l167_167302

theorem largest_sum_of_distinct_factors_of_1764 :
  ∃ (A B C : ℕ), A * B * C = 1764 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A + B + C = 33 :=
by
  sorry

end largest_sum_of_distinct_factors_of_1764_l167_167302


namespace tamia_total_slices_and_pieces_l167_167670

-- Define the conditions
def num_bell_peppers : ℕ := 5
def slices_per_pepper : ℕ := 20
def num_large_slices : ℕ := num_bell_peppers * slices_per_pepper
def num_half_slices : ℕ := num_large_slices / 2
def small_pieces_per_slice : ℕ := 3
def num_small_pieces : ℕ := num_half_slices * small_pieces_per_slice
def num_uncut_slices : ℕ := num_half_slices

-- Define the total number of pieces and slices
def total_pieces_and_slices : ℕ := num_uncut_slices + num_small_pieces

-- State the theorem and provide a placeholder for the proof
theorem tamia_total_slices_and_pieces : total_pieces_and_slices = 200 :=
by
  sorry

end tamia_total_slices_and_pieces_l167_167670


namespace quadratic_func_condition_l167_167906

noncomputable def f (x b c : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_func_condition (b c : ℝ) (h : f (-3) b c = f 1 b c) :
  f 1 b c > c ∧ c > f (-1) b c :=
by
  sorry

end quadratic_func_condition_l167_167906


namespace sufficient_but_not_necessary_l167_167301

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1 / a^2) : a^2 > 1 / a ∧ ¬ ∀ a, a^2 > 1 / a → a > 1 / a^2 :=
by
  sorry

end sufficient_but_not_necessary_l167_167301


namespace proof_x_square_ab_a_square_l167_167527

variable {x b a : ℝ}

/-- Given that x < b < a < 0 where x, b, and a are real numbers, we need to prove x^2 > ab > a^2. -/
theorem proof_x_square_ab_a_square (hx : x < b) (hb : b < a) (ha : a < 0) :
  x^2 > ab ∧ ab > a^2 := 
by
  sorry

end proof_x_square_ab_a_square_l167_167527


namespace range_of_t_l167_167060

noncomputable def a_n (t : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then ((1 / 3) - t) * (n:ℝ) + 2 else t ^ (n - 7)

theorem range_of_t (t : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → a_n t n > a_n t (n + 1)) →
  (1/2 < t ∧ t < 1) :=
by
  intros h
  -- The proof would go here.
  sorry

end range_of_t_l167_167060


namespace expression_evaluation_l167_167209

def e1 : ℤ := 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8)

theorem expression_evaluation : e1 = -50 :=
by
  sorry

end expression_evaluation_l167_167209


namespace probability_at_least_one_l167_167224

variable (p_A p_B : ℚ) (hA : p_A = 1 / 4) (hB : p_B = 2 / 5)

theorem probability_at_least_one (h : p_A * (1 - p_B) + (1 - p_A) * p_B + p_A * p_B = 11 / 20) : 
  (1 - (1 - p_A) * (1 - p_B) = 11 / 20) :=
by
  rw [hA, hB,←h]
  sorry

end probability_at_least_one_l167_167224


namespace election_debate_conditions_l167_167986

theorem election_debate_conditions (n : ℕ) (h_n : n ≥ 3) :
  ¬ ∃ (p : ℕ), n = 2 * (2 ^ p - 2) + 1 :=
sorry

end election_debate_conditions_l167_167986


namespace washing_machine_capacity_l167_167235

-- Define the conditions:
def shirts : ℕ := 39
def sweaters : ℕ := 33
def loads : ℕ := 9
def total_clothes : ℕ := shirts + sweaters -- which is 72

-- Define the statement to be proved:
theorem washing_machine_capacity : ∃ x : ℕ, loads * x = total_clothes ∧ x = 8 :=
by
  -- proof to be completed
  sorry

end washing_machine_capacity_l167_167235


namespace points_on_circle_l167_167602

theorem points_on_circle (n : ℕ) (h1 : ∃ (k : ℕ), k = (35 - 7) ∧ n = 2 * k) : n = 56 :=
sorry

end points_on_circle_l167_167602


namespace find_value_added_l167_167545

open Classical

variable (n : ℕ) (avg_initial avg_final : ℝ)

-- Initial conditions
axiom avg_then_sum (n : ℕ) (avg : ℝ) : n * avg = 600

axiom avg_after_addition (n : ℕ) (avg : ℝ) : n * avg = 825

theorem find_value_added (n : ℕ) (avg_initial avg_final : ℝ) (h1 : n * avg_initial = 600) (h2 : n * avg_final = 825) :
  avg_final - avg_initial = 15 := by
  -- Proof goes here
  sorry

end find_value_added_l167_167545


namespace exists_representation_of_77_using_fewer_sevens_l167_167195

-- Definition of the problem
def represent_77 (expr : String) : Prop :=
  ∀ n : ℕ, expr = "77" ∨ 
             expr = "(77 - 7) + 7" ∨ 
             expr = "(10 * 7) + 7" ∨ 
             expr = "(70 + 7)" ∨ 
             expr = "(7 * 11)" ∨ 
             expr = "7 + 7 * 7 + (7 / 7)"

-- The proof statement
theorem exists_representation_of_77_using_fewer_sevens : ∃ expr : String, represent_77 expr ∧ String.length expr < 3 := 
sorry

end exists_representation_of_77_using_fewer_sevens_l167_167195


namespace domain_of_f_f_is_monotonically_increasing_l167_167654

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 8) + 3

theorem domain_of_f :
  ∀ x, (x ≠ 5 * π / 16 + k * π / 2) := sorry

theorem f_is_monotonically_increasing :
  ∀ x, (π / 16 < x ∧ x < 3 * π / 16 → f x < f (x + ε)) := sorry

end domain_of_f_f_is_monotonically_increasing_l167_167654


namespace stephen_total_distance_l167_167525

def speed_first_segment := 16 -- miles per hour
def time_first_segment := 10 / 60 -- hours

def speed_second_segment := 12 -- miles per hour
def headwind := 2 -- miles per hour
def actual_speed_second_segment := speed_second_segment - headwind
def time_second_segment := 20 / 60 -- hours

def speed_third_segment := 20 -- miles per hour
def tailwind := 4 -- miles per hour
def actual_speed_third_segment := speed_third_segment + tailwind
def time_third_segment := 15 / 60 -- hours

def distance_first_segment := speed_first_segment * time_first_segment
def distance_second_segment := actual_speed_second_segment * time_second_segment
def distance_third_segment := actual_speed_third_segment * time_third_segment

theorem stephen_total_distance : distance_first_segment + distance_second_segment + distance_third_segment = 12 := by
  sorry

end stephen_total_distance_l167_167525


namespace find_f_log_log_3_value_l167_167351

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.logb 3 (Real.sqrt (x*x + 1) - x) + 1

theorem find_f_log_log_3_value
  (a b : ℝ)
  (h1 : f a b (Real.log 10 / Real.log 3) = 5) :
  f a b (-Real.log 10 / Real.log 3) = -3 :=
  sorry

end find_f_log_log_3_value_l167_167351


namespace percent_decaffeinated_second_batch_l167_167383

theorem percent_decaffeinated_second_batch :
  ∀ (initial_stock : ℝ) (initial_percent : ℝ) (additional_stock : ℝ) (total_percent : ℝ) (second_batch_percent : ℝ),
  initial_stock = 400 →
  initial_percent = 0.20 →
  additional_stock = 100 →
  total_percent = 0.26 →
  (initial_percent * initial_stock + second_batch_percent * additional_stock = total_percent * (initial_stock + additional_stock)) →
  second_batch_percent = 0.50 :=
by
  intros initial_stock initial_percent additional_stock total_percent second_batch_percent
  intros h1 h2 h3 h4 h5
  sorry

end percent_decaffeinated_second_batch_l167_167383


namespace slope_angle_of_y_eq_0_l167_167456

theorem slope_angle_of_y_eq_0  :
  ∀ (α : ℝ), (∀ (y x : ℝ), y = 0) → α = 0 :=
by
  intros α h
  sorry

end slope_angle_of_y_eq_0_l167_167456


namespace red_marbles_count_l167_167229

theorem red_marbles_count :
  ∀ (total marbles white yellow green red : ℕ),
    total = 50 →
    white = total / 2 →
    yellow = 12 →
    green = yellow / 2 →
    red = total - (white + yellow + green) →
    red = 7 :=
by
  intros total marbles white yellow green red Htotal Hwhite Hyellow Hgreen Hred
  sorry

end red_marbles_count_l167_167229


namespace least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l167_167865

theorem least_addition_for_divisibility (n : ℕ) : (1100 + n) % 53 = 0 ↔ n = 9 := by
  sorry

theorem least_subtraction_for_divisibility (n : ℕ) : (1100 - n) % 71 = 0 ↔ n = 0 := by
  sorry

theorem least_addition_for_common_divisibility (X : ℕ) : (1100 + X) % (Nat.lcm 19 43) = 0 ∧ X = 534 := by
  sorry

end least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l167_167865


namespace sum_equals_one_l167_167948

noncomputable def sum_proof (x y z : ℝ) (h : x * y * z = 1) : ℝ :=
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x))

theorem sum_equals_one (x y z : ℝ) (h : x * y * z = 1) : 
  sum_proof x y z h = 1 := sorry

end sum_equals_one_l167_167948


namespace problem_l167_167574

def f (x : ℚ) : ℚ :=
  x⁻¹ - (x⁻¹ / (1 - x⁻¹))

theorem problem : f (f (-3)) = 6 / 5 :=
by
  sorry

end problem_l167_167574


namespace inequality_solution_l167_167795

theorem inequality_solution (x : ℝ) 
  (hx1 : x ≠ 1) 
  (hx2 : x ≠ 2) 
  (hx3 : x ≠ 3) 
  (hx4 : x ≠ 4) :
  (1 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ (x ∈ Set.Ioo (-7 : ℝ) 1 ∪ Set.Ioo 3 4) := 
sorry

end inequality_solution_l167_167795


namespace two_y_minus_three_x_l167_167257

variable (x y : ℝ)

noncomputable def x_val : ℝ := 1.2 * 98
noncomputable def y_val : ℝ := 0.9 * (x_val + 35)

theorem two_y_minus_three_x : 2 * y_val - 3 * x_val = -78.12 := by
  sorry

end two_y_minus_three_x_l167_167257


namespace multiplication_is_247_l167_167392

theorem multiplication_is_247 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 247) : 
a = 13 ∧ b = 19 :=
by sorry

end multiplication_is_247_l167_167392


namespace matt_total_points_l167_167919

variable (n2_successful_shots : Nat) (n3_successful_shots : Nat)

def total_points (n2 : Nat) (n3 : Nat) : Nat :=
  2 * n2 + 3 * n3

theorem matt_total_points :
  total_points 4 2 = 14 :=
by
  sorry

end matt_total_points_l167_167919


namespace distance_focus_directrix_parabola_l167_167744

theorem distance_focus_directrix_parabola (p : ℝ) (h : y^2 = 20 * x) : 
  2 * p = 10 :=
by
  -- h represents the given condition y^2 = 20x.
  sorry

end distance_focus_directrix_parabola_l167_167744


namespace dad_contribution_is_correct_l167_167262

noncomputable def carl_savings_weekly : ℕ := 25
noncomputable def savings_duration_weeks : ℕ := 6
noncomputable def coat_cost : ℕ := 170

-- Total savings after 6 weeks
noncomputable def total_savings : ℕ := carl_savings_weekly * savings_duration_weeks

-- Amount used to pay bills in the seventh week
noncomputable def bills_payment : ℕ := total_savings / 3

-- Money left after paying bills
noncomputable def remaining_savings : ℕ := total_savings - bills_payment

-- Amount needed from Dad
noncomputable def dad_contribution : ℕ := coat_cost - remaining_savings

theorem dad_contribution_is_correct : dad_contribution = 70 := by
  sorry

end dad_contribution_is_correct_l167_167262


namespace figure_area_l167_167372

-- Given conditions
def right_angles (α β γ δ: ℕ): Prop :=
  α = 90 ∧ β = 90 ∧ γ = 90 ∧ δ = 90

def segment_lengths (a b c d e f g: ℕ): Prop :=
  a = 15 ∧ b = 8 ∧ c = 7 ∧ d = 3 ∧ e = 4 ∧ f = 2 ∧ g = 5

-- Define the problem
theorem figure_area :
  ∀ (α β γ δ a b c d e f g: ℕ),
    right_angles α β γ δ →
    segment_lengths a b c d e f g →
    a * b - (g * 1 + (d * f)) = 109 :=
by
  sorry

end figure_area_l167_167372


namespace set_diff_M_N_l167_167766

def set_diff {α : Type} (A B : Set α) : Set α := {x | x ∈ A ∧ x ∉ B}

def M : Set ℝ := {x | |x + 1| ≤ 2}

def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α| }

theorem set_diff_M_N :
  set_diff M N = {x | -3 ≤ x ∧ x < 0} :=
by
  sorry

end set_diff_M_N_l167_167766


namespace problem_part1_problem_part2_l167_167758

theorem problem_part1 (k m : ℝ) :
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ k ≠ 3)) →
  k = -3 :=
sorry

theorem problem_part2 (k m : ℝ) :
  ((∃ x1 x2 : ℝ, 
     ((|k|-3) * x1^2 - (k-3) * x1 + 2*m + 1 = 0) ∧
     (3 * x2 - 2 = 4 - 5 * x2 + 2 * x2) ∧
     x1 = -x2) →
  (∀ x : ℝ, (|k|-3) * x^2 - (k-3) * x + 2*m + 1 = 0 → (|k|-3 = 0 ∧ x = -1)) →
  (k = -3 ∧ m = 5/2)) :=
sorry

end problem_part1_problem_part2_l167_167758


namespace yogurt_calories_per_ounce_l167_167963

variable (calories_strawberries_per_unit : ℕ)
variable (calories_yogurt_total : ℕ)
variable (calories_total : ℕ)
variable (strawberries_count : ℕ)
variable (yogurt_ounces_count : ℕ)

theorem yogurt_calories_per_ounce (h1: strawberries_count = 12)
                                   (h2: yogurt_ounces_count = 6)
                                   (h3: calories_strawberries_per_unit = 4)
                                   (h4: calories_total = 150)
                                   (h5: calories_yogurt_total = calories_total - strawberries_count * calories_strawberries_per_unit):
                                   calories_yogurt_total / yogurt_ounces_count = 17 :=
by
  -- We conjecture that this is correct based on given conditions.
  sorry

end yogurt_calories_per_ounce_l167_167963


namespace proof_problem_l167_167808

theorem proof_problem (p q r : ℝ) 
  (h1 : p + q = 20)
  (h2 : p * q = 144) 
  (h3 : q + r = 52) 
  (h4 : 4 * (r + p) = r * p) : 
  r - p = 32 := 
sorry

end proof_problem_l167_167808


namespace new_ratio_alcohol_water_l167_167743

theorem new_ratio_alcohol_water (alcohol water: ℕ) (initial_ratio: alcohol * 3 = water * 4) 
  (extra_water: ℕ) (extra_water_added: extra_water = 4) (alcohol_given: alcohol = 20):
  20 * 19 = alcohol * (water + extra_water) :=
by
  sorry

end new_ratio_alcohol_water_l167_167743


namespace range_of_a_l167_167567

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_a_l167_167567


namespace laura_total_miles_per_week_l167_167829

def round_trip_school : ℕ := 20
def round_trip_supermarket : ℕ := 40
def round_trip_gym : ℕ := 10
def round_trip_friends_house : ℕ := 24

def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2
def gym_trips_per_week : ℕ := 3
def friends_house_trips_per_week : ℕ := 1

def total_miles_driven_per_week :=
  round_trip_school * school_trips_per_week +
  round_trip_supermarket * supermarket_trips_per_week +
  round_trip_gym * gym_trips_per_week +
  round_trip_friends_house * friends_house_trips_per_week

theorem laura_total_miles_per_week : total_miles_driven_per_week = 234 :=
by
  sorry

end laura_total_miles_per_week_l167_167829


namespace false_proposition_l167_167219

theorem false_proposition :
  ¬ (∀ x : ℕ, (x > 0) → (x - 2)^2 > 0) :=
by
  sorry

end false_proposition_l167_167219


namespace find_a_l167_167792

theorem find_a (A B : Real) (b a : Real) (hA : A = 45) (hB : B = 60) (hb : b = Real.sqrt 3) : 
  a = Real.sqrt 2 :=
sorry

end find_a_l167_167792


namespace fractional_eq_has_root_l167_167753

theorem fractional_eq_has_root (x : ℝ) (m : ℝ) (h : x ≠ 4) :
    (3 / (x - 4) + (x + m) / (4 - x) = 1) → m = -1 :=
by
    intros h_eq
    sorry

end fractional_eq_has_root_l167_167753


namespace min_distance_l167_167318

variables {P Q : ℝ × ℝ}

def line (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2) ^ 2 + (Q.2 - 2) ^ 2 = 4

theorem min_distance (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : line P) (hQ : circle Q) :
  ∃ d : ℝ, d = dist P Q ∧ d = 9 / 5 := sorry

end min_distance_l167_167318


namespace arithmetic_sequence_third_term_l167_167663

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end arithmetic_sequence_third_term_l167_167663


namespace curve_not_parabola_l167_167598

theorem curve_not_parabola (k : ℝ) : ¬ ∃ (x y : ℝ), (x^2 + k * y^2 = 1) ↔ (k = -y / x) :=
by
  sorry

end curve_not_parabola_l167_167598


namespace square_area_divided_into_equal_rectangles_l167_167611

theorem square_area_divided_into_equal_rectangles (w : ℝ) (a : ℝ) (h : 5 = w) :
  (∃ s : ℝ, s * s = a ∧ s * s / 5 = a / 5) ↔ a = 400 :=
by
  sorry

end square_area_divided_into_equal_rectangles_l167_167611


namespace moles_of_water_formed_l167_167490

-- Definitions (conditions)
def reaction : String := "NaOH + HCl → NaCl + H2O"

def initial_moles_NaOH : ℕ := 1
def initial_moles_HCl : ℕ := 1
def mole_ratio_NaOH_HCl : ℕ := 1
def mole_ratio_NaOH_H2O : ℕ := 1

-- The proof problem
theorem moles_of_water_formed :
  initial_moles_NaOH = mole_ratio_NaOH_HCl →
  initial_moles_HCl = mole_ratio_NaOH_HCl →
  mole_ratio_NaOH_H2O * initial_moles_NaOH = 1 :=
by
  intros h1 h2
  sorry

end moles_of_water_formed_l167_167490


namespace rectangle_area_l167_167867

theorem rectangle_area (l w : ℕ) (h_diagonal : l^2 + w^2 = 17^2) (h_perimeter : l + w = 23) : l * w = 120 :=
by
  sorry

end rectangle_area_l167_167867


namespace find_positive_number_l167_167997

theorem find_positive_number 
  (x : ℝ) (h_pos : x > 0) 
  (h_eq : (2 / 3) * x = (16 / 216) * (1 / x)) : 
  x = 1 / 3 :=
by
  -- This is indicating that we're skipping the actual proof steps
  sorry

end find_positive_number_l167_167997


namespace net_profit_positive_max_average_net_profit_l167_167123

def initial_investment : ℕ := 720000
def first_year_expense : ℕ := 120000
def annual_expense_increase : ℕ := 40000
def annual_sales : ℕ := 500000

def net_profit (n : ℕ) : ℕ := annual_sales - (first_year_expense + (n-1) * annual_expense_increase)
def average_net_profit (y n : ℕ) : ℕ := y / n

theorem net_profit_positive (n : ℕ) : net_profit n > 0 :=
sorry -- prove when net profit is positive

theorem max_average_net_profit (n : ℕ) : 
∀ m, average_net_profit (net_profit m) m ≤ average_net_profit (net_profit n) n :=
sorry -- prove when the average net profit is maximized

end net_profit_positive_max_average_net_profit_l167_167123


namespace choir_members_max_l167_167234

-- Define the conditions and the proof for the equivalent problem.
theorem choir_members_max (c s y : ℕ) (h1 : c < 120) (h2 : s * y + 3 = c) (h3 : (s - 1) * (y + 2) = c) : c = 120 := by
  sorry

end choir_members_max_l167_167234


namespace average_marks_l167_167685

/-- Shekar scored 76, 65, 82, 67, and 85 marks in Mathematics, Science, Social Studies, English, and Biology respectively.
    We aim to prove that his average marks are 75. -/

def marks : List ℕ := [76, 65, 82, 67, 85]

theorem average_marks : (marks.sum / marks.length) = 75 := by
  sorry

end average_marks_l167_167685


namespace quadratic_equation_root_condition_l167_167282

theorem quadratic_equation_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, (a - 1) * x1^2 - 4 * x1 - 1 = 0 ∧ (a - 1) * x2^2 - 4 * x2 - 1 = 0) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end quadratic_equation_root_condition_l167_167282


namespace crude_oil_mixture_l167_167225

theorem crude_oil_mixture (x y : ℝ) 
  (h1 : x + y = 50)
  (h2 : 0.25 * x + 0.75 * y = 0.55 * 50) : 
  y = 30 :=
by
  sorry

end crude_oil_mixture_l167_167225


namespace largest_is_D_l167_167958

-- Definitions based on conditions
def A : ℕ := 27
def B : ℕ := A + 7
def C : ℕ := B - 9
def D : ℕ := 2 * C

-- Theorem stating D is the largest
theorem largest_is_D : D = max (max A B) (max C D) :=
by
  -- Inserting sorry because the proof is not required.
  sorry

end largest_is_D_l167_167958


namespace average_earning_week_l167_167543

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ) 
  (h1 : (D1 + D2 + D3 + D4) / 4 = 18)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 13) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 22.86 := 
by 
  sorry

end average_earning_week_l167_167543


namespace total_sales_l167_167646

-- Define sales of Robyn and Lucy
def Robyn_sales : Nat := 47
def Lucy_sales : Nat := 29

-- Prove total sales
theorem total_sales : Robyn_sales + Lucy_sales = 76 :=
by
  sorry

end total_sales_l167_167646


namespace distinct_valid_c_values_l167_167959

theorem distinct_valid_c_values : 
  let is_solution (c : ℤ) (x : ℚ) := (5 * ⌊x⌋₊ + 3 * ⌈x⌉₊ = c) 
  ∃ s : Finset ℤ, (∀ c ∈ s, (∃ x : ℚ, is_solution c x)) ∧ s.card = 500 :=
by sorry

end distinct_valid_c_values_l167_167959


namespace right_triangle_altitude_l167_167377

theorem right_triangle_altitude {DE DF EF altitude : ℝ} (h_right_triangle : DE^2 = DF^2 + EF^2)
  (h_DE : DE = 15) (h_DF : DF = 9) (h_EF : EF = 12) (h_area : (DF * EF) / 2 = 54) :
  altitude = 7.2 := 
  sorry

end right_triangle_altitude_l167_167377


namespace sample_size_stratified_sampling_l167_167136

theorem sample_size_stratified_sampling :
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  sample_size = 20 :=
by
  -- Definitions:
  let N_business := 120
  let N_management := 24
  let N_logistics := 16
  let N_total := N_business + N_management + N_logistics
  let n_management_chosen := 3
  let sampling_fraction := n_management_chosen / N_management
  let sample_size := N_total * sampling_fraction
  
  -- Proof:
  sorry

end sample_size_stratified_sampling_l167_167136


namespace find_range_of_r_l167_167261

noncomputable def range_of_r : Set ℝ :=
  {r : ℝ | 3 * Real.sqrt 5 - 3 * Real.sqrt 2 ≤ r ∧ r ≤ 3 * Real.sqrt 5 + 3 * Real.sqrt 2}

theorem find_range_of_r 
  (O : ℝ × ℝ) (A : ℝ × ℝ) (r : ℝ) (h : r > 0)
  (hA : A = (0, 3))
  (C : Set (ℝ × ℝ)) (hC : C = {M : ℝ × ℝ | (M.1 - 3)^2 + (M.2 - 3)^2 = r^2})
  (M : ℝ × ℝ) (hM : M ∈ C)
  (h_cond : (M.1 - 0)^2 + (M.2 - 3)^2 = 2 * ((M.1 - 0)^2 + (M.2 - 0)^2)) :
  r ∈ range_of_r :=
sorry

end find_range_of_r_l167_167261


namespace yellow_yellow_pairs_count_l167_167290

def num_blue_students : ℕ := 75
def num_yellow_students : ℕ := 105
def total_pairs : ℕ := 90
def blue_blue_pairs : ℕ := 30

theorem yellow_yellow_pairs_count :
  -- number of pairs where both students are wearing yellow shirts is 45.
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 :=
by
  sorry

end yellow_yellow_pairs_count_l167_167290


namespace relationship_among_abc_l167_167627

theorem relationship_among_abc 
  (f : ℝ → ℝ)
  (h_symm : ∀ x, f (x) = f (-x))
  (h_def : ∀ x, 0 < x → f x = |Real.log x / Real.log 2|)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = f (1 / 3))
  (hb : b = f (-4))
  (hc : c = f 2) :
  c < a ∧ a < b :=
by
  sorry

end relationship_among_abc_l167_167627


namespace total_girls_is_68_l167_167874

-- Define the initial conditions
def track_length : ℕ := 100
def student_spacing : ℕ := 2
def girls_per_cycle : ℕ := 2
def cycle_length : ℕ := 3

-- Calculate the number of students on one side
def students_on_one_side : ℕ := track_length / student_spacing + 1

-- Number of cycles of three students
def num_cycles : ℕ := students_on_one_side / cycle_length

-- Number of girls on one side
def girls_on_one_side : ℕ := num_cycles * girls_per_cycle

-- Total number of girls on both sides
def total_girls : ℕ := girls_on_one_side * 2

theorem total_girls_is_68 : total_girls = 68 := by
  -- proof will be provided here
  sorry

end total_girls_is_68_l167_167874


namespace jen_age_when_son_born_l167_167866

theorem jen_age_when_son_born (S : ℕ) (Jen_present_age : ℕ) 
  (h1 : S = 16) (h2 : Jen_present_age = 3 * S - 7) : 
  Jen_present_age - S = 25 :=
by {
  sorry -- Proof would be here, but it is not required as per the instructions.
}

end jen_age_when_son_born_l167_167866


namespace choose_with_at_least_one_girl_l167_167344

theorem choose_with_at_least_one_girl :
  let boys := 4
  let girls := 2
  let total_students := boys + girls
  let ways_choose_4 := Nat.choose total_students 4
  let ways_all_boys := Nat.choose boys 4
  ways_choose_4 - ways_all_boys = 14 := by
  sorry

end choose_with_at_least_one_girl_l167_167344


namespace max_articles_produced_l167_167023

variables (a b c d p q r s z : ℝ)
variables (h1 : d = (a^2 * b * c) / z)
variables (h2 : p * q * r ≤ s)

theorem max_articles_produced : 
  p * q * r * (a / z) = s * (a / z) :=
by
  sorry

end max_articles_produced_l167_167023


namespace cos_2alpha_minus_pi_over_6_l167_167926

theorem cos_2alpha_minus_pi_over_6 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hSin : Real.sin (α + π / 6) = 3 / 5) :
  Real.cos (2 * α - π / 6) = 24 / 25 :=
sorry

end cos_2alpha_minus_pi_over_6_l167_167926


namespace total_students_l167_167637

theorem total_students (T : ℕ) (h1 : (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T) : 
  T = 400 :=
sorry

end total_students_l167_167637


namespace periodic_function_l167_167457

variable {α : Type*} [AddGroup α] {f : α → α} {a b : α}

def symmetric_around (c : α) (f : α → α) : Prop := ∀ x, f (c - x) = f (c + x)

theorem periodic_function (h1 : symmetric_around a f) (h2 : symmetric_around b f) (h_ab : a ≠ b) : ∃ T, (∀ x, f (x + T) = f x) := 
sorry

end periodic_function_l167_167457


namespace geometric_sequence_sum_l167_167671

-- We state the main problem in Lean as a theorem.
theorem geometric_sequence_sum (S : ℕ → ℕ) (S_4_eq : S 4 = 8) (S_8_eq : S 8 = 24) : S 12 = 88 :=
  sorry

end geometric_sequence_sum_l167_167671


namespace overtime_rate_is_correct_l167_167962

/-
Define the parameters:
ordinary_rate: Rate per hour for ordinary time in dollars
total_hours: Total hours worked in a week
overtime_hours: Overtime hours worked in a week
total_earnings: Total earnings for the week in dollars
-/

def ordinary_rate : ℝ := 0.60
def total_hours : ℝ := 50
def overtime_hours : ℝ := 8
def total_earnings : ℝ := 32.40

noncomputable def overtime_rate : ℝ :=
(total_earnings - ordinary_rate * (total_hours - overtime_hours)) / overtime_hours

theorem overtime_rate_is_correct :
  overtime_rate = 0.90 :=
by
  sorry

end overtime_rate_is_correct_l167_167962


namespace hexagon_circle_radius_l167_167930

noncomputable def hexagon_radius (sides : List ℝ) (probability : ℝ) : ℝ :=
  let total_angle := 360.0
  let visible_angle := probability * total_angle
  let side_length_average := (sides.sum / sides.length : ℝ)
  let theta := (visible_angle / 6 : ℝ) -- assuming θ approximately splits equally among 6 gaps
  side_length_average / Real.sin (theta / 2 * Real.pi / 180.0)

theorem hexagon_circle_radius :
  hexagon_radius [3, 2, 4, 3, 2, 4] (1 / 3) = 17.28 :=
by
  sorry

end hexagon_circle_radius_l167_167930


namespace John_has_22_quarters_l167_167815

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l167_167815


namespace find_units_digit_l167_167310

def units_digit (n : ℕ) : ℕ := n % 10

theorem find_units_digit :
  units_digit (3 * 19 * 1933 - 3^4) = 0 :=
by
  sorry

end find_units_digit_l167_167310


namespace orthocenter_circumradii_equal_l167_167994

-- Define a triangle with its orthocenter and circumradius
variables {A B C H : Point} (R r : ℝ)

-- Assume H is the orthocenter of triangle ABC
def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  sorry -- This should state the definition or properties of an orthocenter

-- Assume the circumradius of triangle ABC is R 
def is_circumradius_ABC (A B C : Point) (R : ℝ) : Prop :=
  sorry -- This should capture the circumradius property

-- Assume circumradius of triangle BHC is r
def is_circumradius_BHC (B H C : Point) (r : ℝ) : Prop :=
  sorry -- This should capture the circumradius property
  
-- Prove that if H is the orthocenter of triangle ABC, the circumradius of ABC is R 
-- and the circumradius of BHC is r, then R = r
theorem orthocenter_circumradii_equal (h_orthocenter : is_orthocenter H A B C) 
  (h_circumradius_ABC : is_circumradius_ABC A B C R)
  (h_circumradius_BHC : is_circumradius_BHC B H C r) : R = r :=
  sorry

end orthocenter_circumradii_equal_l167_167994


namespace total_robots_correct_l167_167221

def number_of_shapes : ℕ := 3
def number_of_colors : ℕ := 4
def total_types_of_robots : ℕ := number_of_shapes * number_of_colors

theorem total_robots_correct : total_types_of_robots = 12 := by
  sorry

end total_robots_correct_l167_167221


namespace green_papayas_left_l167_167335

/-- Define the initial number of green papayas on the tree -/
def initial_green_papayas : ℕ := 14

/-- Define the number of papayas that turned yellow on Friday -/
def friday_yellow_papayas : ℕ := 2

/-- Define the number of papayas that turned yellow on Sunday -/
def sunday_yellow_papayas : ℕ := 2 * friday_yellow_papayas

/-- The remaining number of green papayas after Friday and Sunday -/
def remaining_green_papayas : ℕ := initial_green_papayas - friday_yellow_papayas - sunday_yellow_papayas

theorem green_papayas_left : remaining_green_papayas = 8 := by
  sorry

end green_papayas_left_l167_167335


namespace my_inequality_l167_167711

open Real

variable {a b c : ℝ}

theorem my_inequality 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a * b + b * c + c * a = 1) :
  sqrt (a ^ 3 + a) + sqrt (b ^ 3 + b) + sqrt (c ^ 3 + c) ≥ 2 * sqrt (a + b + c) := 
  sorry

end my_inequality_l167_167711


namespace student_allowance_l167_167320

theorem student_allowance (A : ℝ) (h1 : A * (2/5) = A - (A * (3/5)))
  (h2 : (A - (A * (2/5))) * (1/3) = ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) * (1/3))
  (h3 : ((A - (A * (2/5))) - ((A - (A * (2/5))) * (1/3))) = 1.20) :
  A = 3.00 :=
by
  sorry

end student_allowance_l167_167320


namespace trig_identity_l167_167307

theorem trig_identity :
  let s60 := Real.sin (60 * Real.pi / 180)
  let c1 := Real.cos (1 * Real.pi / 180)
  let c20 := Real.cos (20 * Real.pi / 180)
  let s10 := Real.sin (10 * Real.pi / 180)
  s60 * c1 * c20 - s10 = Real.sqrt 3 / 2 - s10 :=
by
  sorry

end trig_identity_l167_167307


namespace rectangle_perimeter_of_right_triangle_l167_167508

-- Define the conditions for the triangle and the rectangle
def rightTriangleArea (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℕ :=
  (1 / 2) * a * b

def rectanglePerimeter (width area : ℕ) : ℕ :=
  2 * ((area / width) + width)

theorem rectangle_perimeter_of_right_triangle :
  ∀ (a b c width : ℕ) (h_a : a = 5) (h_b : b = 12) (h_c : c = 13)
    (h_pyth : a^2 + b^2 = c^2) (h_width : width = 5)
    (h_area_eq : rightTriangleArea a b c h_pyth = width * (rightTriangleArea a b c h_pyth / width)),
  rectanglePerimeter width (rightTriangleArea a b c h_pyth) = 22 :=
by
  intros
  sorry

end rectangle_perimeter_of_right_triangle_l167_167508


namespace cows_in_herd_l167_167965

theorem cows_in_herd (n : ℕ) (h1 : n / 3 + n / 6 + n / 7 < n) (h2 : 15 = n * 5 / 14) : n = 42 :=
sorry

end cows_in_herd_l167_167965


namespace angle_division_l167_167499

theorem angle_division (α : ℝ) (n : ℕ) (θ : ℝ) (h : α = 78) (hn : n = 26) (ht : θ = 3) :
  α / n = θ :=
by
  sorry

end angle_division_l167_167499


namespace move_line_down_l167_167667

theorem move_line_down (x : ℝ) : (y = -x + 1) → (y = -x - 2) := by
  sorry

end move_line_down_l167_167667


namespace fish_fishermen_problem_l167_167027

theorem fish_fishermen_problem (h: ℕ) (r: ℕ) (w_h: ℕ) (w_r: ℕ) (claimed_weight: ℕ) (total_real_weight: ℕ) 
  (total_fishermen: ℕ) :
  -- conditions
  (claimed_weight = 60) →
  (total_real_weight = 120) →
  (total_fishermen = 10) →
  (w_h = 30) →
  (w_r < 60 / 7) →
  (h + r = total_fishermen) →
  (2 * w_h * h + r * claimed_weight = claimed_weight * total_fishermen) →
  -- prove the number of regular fishermen
  (r = 7 ∨ r = 8) :=
sorry

end fish_fishermen_problem_l167_167027


namespace min_value_of_ratio_l167_167714

theorem min_value_of_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (4 / x + 1 / y) ≥ 6 + 4 * Real.sqrt 2 :=
sorry

end min_value_of_ratio_l167_167714


namespace percentage_increase_l167_167291

theorem percentage_increase (L : ℕ) (h : L + 60 = 240) : 
  ((60:ℝ) / (L:ℝ)) * 100 = 33.33 := 
by
  sorry

end percentage_increase_l167_167291


namespace no_solution_when_k_eq_7_l167_167453

theorem no_solution_when_k_eq_7 
  (x : ℝ) (h₁ : x ≠ 4) (h₂ : x ≠ 8) : 
  (∀ k : ℝ, (x - 3) / (x - 4) = (x - k) / (x - 8) → False) ↔ k = 7 :=
by
  sorry

end no_solution_when_k_eq_7_l167_167453


namespace barrel_contents_lost_l167_167182

theorem barrel_contents_lost (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 220) 
  (h2 : remaining_amount = 198) : 
  (initial_amount - remaining_amount) / initial_amount * 100 = 10 :=
by
  rw [h1, h2]
  sorry

end barrel_contents_lost_l167_167182


namespace solve_x4_minus_16_eq_0_l167_167900

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end solve_x4_minus_16_eq_0_l167_167900


namespace alicia_total_payment_l167_167232

def daily_rent_cost : ℕ := 30
def miles_cost_per_mile : ℝ := 0.25
def rental_days : ℕ := 5
def driven_miles : ℕ := 500

def total_cost (daily_rent_cost : ℕ) (rental_days : ℕ)
               (miles_cost_per_mile : ℝ) (driven_miles : ℕ) : ℝ :=
  (daily_rent_cost * rental_days) + (miles_cost_per_mile * driven_miles)

theorem alicia_total_payment :
  total_cost daily_rent_cost rental_days miles_cost_per_mile driven_miles = 275 := by
  sorry

end alicia_total_payment_l167_167232


namespace departure_sequences_count_l167_167562

noncomputable def total_departure_sequences (trains: Finset ℕ) (A B : ℕ) 
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6) 
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : ℕ := 6 * 6 * 6

-- The main theorem statement: given the conditions, prove the total number of different sequences is 216
theorem departure_sequences_count (trains: Finset ℕ) (A B : ℕ)
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6)
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : total_departure_sequences trains A B h hAB = 216 := 
by 
  sorry

end departure_sequences_count_l167_167562


namespace jesse_initial_blocks_l167_167494

def total_blocks_initial (blocks_cityscape blocks_farmhouse blocks_zoo blocks_first_area blocks_second_area blocks_third_area blocks_left : ℕ) : ℕ :=
  blocks_cityscape + blocks_farmhouse + blocks_zoo + blocks_first_area + blocks_second_area + blocks_third_area + blocks_left

theorem jesse_initial_blocks :
  total_blocks_initial 80 123 95 57 43 62 84 = 544 :=
sorry

end jesse_initial_blocks_l167_167494


namespace alice_catch_up_time_l167_167095

def alice_speed : ℝ := 45
def tom_speed : ℝ := 15
def initial_distance : ℝ := 4
def minutes_per_hour : ℝ := 60

theorem alice_catch_up_time :
  (initial_distance / (alice_speed - tom_speed)) * minutes_per_hour = 8 :=
by
  sorry

end alice_catch_up_time_l167_167095


namespace probability_two_heads_one_tail_in_three_tosses_l167_167946

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses_l167_167946


namespace negation_of_proposition_l167_167106

theorem negation_of_proposition (a b : ℝ) (h : a > b → a^2 > b^2) : a ≤ b → a^2 ≤ b^2 :=
by
  sorry

end negation_of_proposition_l167_167106


namespace pages_with_same_units_digit_count_l167_167606

def same_units_digit (x : ℕ) (y : ℕ) : Prop :=
  x % 10 = y % 10

theorem pages_with_same_units_digit_count :
  ∃! (n : ℕ), n = 12 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ 61) → same_units_digit x (62 - x) → 
  (x % 10 = 2 ∨ x % 10 = 7) :=
by
  sorry

end pages_with_same_units_digit_count_l167_167606


namespace rectangle_area_change_l167_167656

theorem rectangle_area_change 
  (L B : ℝ) 
  (A : ℝ := L * B) 
  (L' : ℝ := 1.30 * L) 
  (B' : ℝ := 0.75 * B) 
  (A' : ℝ := L' * B') : 
  A' / A = 0.975 := 
by sorry

end rectangle_area_change_l167_167656


namespace combined_weight_proof_l167_167460

-- Definitions of atomic weights
def weight_C : ℝ := 12.01
def weight_H : ℝ := 1.01
def weight_O : ℝ := 16.00
def weight_S : ℝ := 32.07

-- Definitions of molar masses of compounds
def molar_mass_C6H8O7 : ℝ := (6 * weight_C) + (8 * weight_H) + (7 * weight_O)
def molar_mass_H2SO4 : ℝ := (2 * weight_H) + weight_S + (4 * weight_O)

-- Definitions of number of moles
def moles_C6H8O7 : ℝ := 8
def moles_H2SO4 : ℝ := 4

-- Combined weight
def combined_weight : ℝ := (moles_C6H8O7 * molar_mass_C6H8O7) + (moles_H2SO4 * molar_mass_H2SO4)

theorem combined_weight_proof : combined_weight = 1929.48 :=
by
  -- calculations as explained in the problem
  let wC6H8O7 := moles_C6H8O7 * molar_mass_C6H8O7
  let wH2SO4 := moles_H2SO4 * molar_mass_H2SO4
  have h1 : wC6H8O7 = 8 * 192.14 := by sorry
  have h2 : wH2SO4 = 4 * 98.09 := by sorry
  have h3 : combined_weight = wC6H8O7 + wH2SO4 := by simp [combined_weight, wC6H8O7, wH2SO4]
  rw [h3, h1, h2]
  simp
  sorry -- finish the proof as necessary

end combined_weight_proof_l167_167460


namespace fraction_simplification_l167_167491

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_simplification_l167_167491


namespace cara_between_friends_l167_167863

theorem cara_between_friends (n : ℕ) (h : n = 6) : ∃ k : ℕ, k = 15 :=
by {
  sorry
}

end cara_between_friends_l167_167863


namespace find_some_number_l167_167445

-- Definitions based on the given condition
def some_number : ℝ := sorry
def equation := some_number * 3.6 / (0.04 * 0.1 * 0.007) = 990.0000000000001

-- An assertion/proof that given the equation, some_number equals 7.7
theorem find_some_number (h : equation) : some_number = 7.7 :=
sorry

end find_some_number_l167_167445


namespace fg_equals_seven_l167_167272

def g (x : ℤ) : ℤ := x * x
def f (x : ℤ) : ℤ := 2 * x - 1

theorem fg_equals_seven : f (g 2) = 7 := by
  sorry

end fg_equals_seven_l167_167272


namespace total_balloons_are_48_l167_167618

theorem total_balloons_are_48 
  (brooke_initial : ℕ) (brooke_add : ℕ) (tracy_initial : ℕ) (tracy_add : ℕ)
  (brooke_half_given : ℕ) (tracy_third_popped : ℕ) : 
  brooke_initial = 20 →
  brooke_add = 15 →
  tracy_initial = 10 →
  tracy_add = 35 →
  brooke_half_given = (brooke_initial + brooke_add) / 2 →
  tracy_third_popped = (tracy_initial + tracy_add) / 3 →
  (brooke_initial + brooke_add - brooke_half_given) + (tracy_initial + tracy_add - tracy_third_popped) = 48 := 
by
  intros
  sorry

end total_balloons_are_48_l167_167618


namespace complex_div_eq_half_sub_half_i_l167_167469

theorem complex_div_eq_half_sub_half_i (i : ℂ) (hi : i^2 = -1) : 
  (i^3 / (1 - i)) = (1 / 2) - (1 / 2) * i :=
by
  sorry

end complex_div_eq_half_sub_half_i_l167_167469


namespace number_of_ways_to_choose_positions_l167_167030

-- Definition of the problem conditions
def number_of_people : ℕ := 8

-- Statement of the proof problem
theorem number_of_ways_to_choose_positions : 
  (number_of_people) * (number_of_people - 1) * (number_of_people - 2) = 336 := by
  -- skipping the proof itself
  sorry

end number_of_ways_to_choose_positions_l167_167030


namespace speed_in_first_hour_l167_167879

variable (x : ℕ)
-- Conditions: 
-- The speed of the car in the second hour:
def speed_in_second_hour : ℕ := 30
-- The average speed of the car:
def average_speed : ℕ := 60
-- The total time traveled:
def total_time : ℕ := 2

-- Proof problem: Prove that the speed of the car in the first hour is 90 km/h.
theorem speed_in_first_hour : x + speed_in_second_hour = average_speed * total_time → x = 90 := 
by 
  intro h
  sorry

end speed_in_first_hour_l167_167879


namespace range_of_m_l167_167129

noncomputable def f (x : ℝ) : ℝ :=
  if x >= -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > m^2 - m) ↔ -1 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l167_167129


namespace total_hours_worked_l167_167007

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l167_167007


namespace project_completion_by_B_l167_167181

-- Definitions of the given conditions
def person_A_work_rate := 1 / 10
def person_B_work_rate := 1 / 15
def days_A_worked := 3

-- Definition of the mathematical proof problem
theorem project_completion_by_B {x : ℝ} : person_A_work_rate * days_A_worked + person_B_work_rate * x = 1 :=
by
  sorry

end project_completion_by_B_l167_167181


namespace minimum_area_rectangle_l167_167244

noncomputable def minimum_rectangle_area (a : ℝ) : ℝ :=
  if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
  else if a < 1 / 2 then 1 - 2 * a
  else 0

theorem minimum_area_rectangle (a : ℝ) :
  minimum_rectangle_area a =
    if a ≤ 0 then (1 - a) * Real.sqrt (1 - a)
    else if a < 1 / 2 then 1 - 2 * a
    else 0 :=
by
  sorry

end minimum_area_rectangle_l167_167244


namespace number_of_girls_l167_167505

variable (b g d : ℕ)

-- Conditions
axiom boys_count : b = 1145
axiom difference : d = 510
axiom boys_equals_girls_plus_difference : b = g + d

-- Theorem to prove
theorem number_of_girls : g = 635 := by
  sorry

end number_of_girls_l167_167505


namespace a_and_b_finish_work_in_72_days_l167_167087

noncomputable def work_rate_A_B {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : ℝ :=
  A + B

theorem a_and_b_finish_work_in_72_days {A B C : ℝ} 
  (h1 : B + C = 1 / 24) 
  (h2 : A + C = 1 / 36) 
  (h3 : A + B + C = 1 / 16.000000000000004) : 
  work_rate_A_B h1 h2 h3 = 1 / 72 :=
sorry

end a_and_b_finish_work_in_72_days_l167_167087


namespace min_value_x_y_l167_167295

theorem min_value_x_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) : x + y ≥ 2 :=
sorry

end min_value_x_y_l167_167295


namespace distance_from_center_to_line_l167_167706

-- Define the circle and its center
def is_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def center : (ℝ × ℝ) := (1, 0)

-- Define the line equation y = tan(30°) * x
def is_line (x y : ℝ) : Prop := y = (1 / Real.sqrt 3) * x

-- Function to compute the distance from a point to a line
noncomputable def distance_point_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  (abs (A * p.1 + B * p.2 + C)) / Real.sqrt (A^2 + B^2)

-- The main theorem to be proven:
theorem distance_from_center_to_line : 
  distance_point_to_line center (1 / Real.sqrt 3) (-1) 0 = 1 / 2 :=
  sorry

end distance_from_center_to_line_l167_167706


namespace zoo_ticket_sales_l167_167554

-- Define the number of total people, number of adults, and ticket prices
def total_people : ℕ := 254
def num_adults : ℕ := 51
def adult_ticket_price : ℕ := 28
def kid_ticket_price : ℕ := 12

-- Define the number of kids as the difference between total people and number of adults
def num_kids : ℕ := total_people - num_adults

-- Define the revenue from adult tickets and kid tickets
def revenue_adult_tickets : ℕ := num_adults * adult_ticket_price
def revenue_kid_tickets : ℕ := num_kids * kid_ticket_price

-- Define the total revenue
def total_revenue : ℕ := revenue_adult_tickets + revenue_kid_tickets

-- Theorem to prove the total revenue equals 3864
theorem zoo_ticket_sales : total_revenue = 3864 :=
  by {
    -- sorry allows us to skip the proof
    sorry
  }

end zoo_ticket_sales_l167_167554


namespace average_apples_per_hour_l167_167664

theorem average_apples_per_hour (A H : ℝ) (hA : A = 12) (hH : H = 5) : A / H = 2.4 := by
  -- sorry skips the proof
  sorry

end average_apples_per_hour_l167_167664


namespace find_a_l167_167364

def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem find_a (a : ℝ) (h : {x | x^2 - 3 * x + 2 = 0} ∩ {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0} = {2}) :
  a = -3 ∨ a = -1 :=
by
  sorry

end find_a_l167_167364


namespace abs_val_of_minus_two_and_half_l167_167717

-- Definition of the absolute value function for real numbers
def abs_val (x : ℚ) : ℚ := if x < 0 then -x else x

-- Prove that the absolute value of -2.5 (which is -5/2) is equal to 2.5 (which is 5/2)
theorem abs_val_of_minus_two_and_half : abs_val (-5/2) = 5/2 := by
  sorry

end abs_val_of_minus_two_and_half_l167_167717


namespace sleeping_bag_selling_price_l167_167951

def wholesale_cost : ℝ := 24.56
def gross_profit_percentage : ℝ := 0.14

def gross_profit (x : ℝ) : ℝ := gross_profit_percentage * x

def selling_price (x y : ℝ) : ℝ := x + y

theorem sleeping_bag_selling_price :
  selling_price wholesale_cost (gross_profit wholesale_cost) = 28 := by
  sorry

end sleeping_bag_selling_price_l167_167951


namespace train_speed_is_40_kmh_l167_167227

noncomputable def speed_of_train (train_length_m : ℝ) 
                                   (man_speed_kmh : ℝ) 
                                   (pass_time_s : ℝ) : ℝ :=
  let train_length_km := train_length_m / 1000
  let pass_time_h := pass_time_s / 3600
  let relative_speed_kmh := train_length_km / pass_time_h
  relative_speed_kmh - man_speed_kmh
  
theorem train_speed_is_40_kmh :
  speed_of_train 110 4 9 = 40 := 
by
  sorry

end train_speed_is_40_kmh_l167_167227


namespace linear_inequality_m_eq_one_l167_167725

theorem linear_inequality_m_eq_one
  (m : ℤ)
  (h1 : |m| = 1)
  (h2 : m + 1 ≠ 0) :
  m = 1 :=
sorry

end linear_inequality_m_eq_one_l167_167725


namespace min_value_expression_l167_167878

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + z = 3) (h2 : z = (x + y) / 2) : 
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) = 3 / 2 :=
by sorry

end min_value_expression_l167_167878


namespace find_range_of_m_l167_167205

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 7
def B (m x : ℝ) : Prop := m + 1 < x ∧ x < 2 * m - 1

theorem find_range_of_m (m : ℝ) : 
  (∀ x, B m x → A x) ∧ (∃ x, B m x) → 2 < m ∧ m ≤ 4 :=
by
  sorry

end find_range_of_m_l167_167205


namespace metal_beams_per_panel_l167_167632

theorem metal_beams_per_panel (panels sheets_per_panel rods_per_sheet rods_needed beams_per_panel rods_per_beam : ℕ)
    (h1 : panels = 10)
    (h2 : sheets_per_panel = 3)
    (h3 : rods_per_sheet = 10)
    (h4 : rods_needed = 380)
    (h5 : rods_per_beam = 4)
    (h6 : beams_per_panel = 2) :
    (panels * sheets_per_panel * rods_per_sheet + panels * beams_per_panel * rods_per_beam = rods_needed) :=
by
  sorry

end metal_beams_per_panel_l167_167632


namespace total_flowers_is_288_l167_167006

-- Definitions from the Conditions in a)
def arwen_tulips : ℕ := 20
def arwen_roses : ℕ := 18
def elrond_tulips : ℕ := 2 * arwen_tulips
def elrond_roses : ℕ := 3 * arwen_roses
def galadriel_tulips : ℕ := 3 * elrond_tulips
def galadriel_roses : ℕ := 2 * arwen_roses

-- Total number of tulips
def total_tulips : ℕ := arwen_tulips + elrond_tulips + galadriel_tulips

-- Total number of roses
def total_roses : ℕ := arwen_roses + elrond_roses + galadriel_roses

-- Total number of flowers
def total_flowers : ℕ := total_tulips + total_roses

theorem total_flowers_is_288 : total_flowers = 288 :=
by
  -- Placeholder for proof
  sorry

end total_flowers_is_288_l167_167006


namespace tan_periodic_n_solution_l167_167417

open Real

theorem tan_periodic_n_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ tan (n * (π / 180)) = tan (1540 * (π / 180)) ∧ n = 40 :=
by
  sorry

end tan_periodic_n_solution_l167_167417


namespace average_minutes_proof_l167_167674

noncomputable def average_minutes_heard (total_minutes : ℕ) (total_attendees : ℕ) (full_listened_fraction : ℚ) (none_listened_fraction : ℚ) (half_remainder_fraction : ℚ) : ℚ := 
  let full_listeners := full_listened_fraction * total_attendees
  let none_listeners := none_listened_fraction * total_attendees
  let remaining_listeners := total_attendees - full_listeners - none_listeners
  let half_listeners := half_remainder_fraction * remaining_listeners
  let quarter_listeners := remaining_listeners - half_listeners
  let total_heard := (full_listeners * total_minutes) + (none_listeners * 0) + (half_listeners * (total_minutes / 2)) + (quarter_listeners * (total_minutes / 4))
  total_heard / total_attendees

theorem average_minutes_proof : 
  average_minutes_heard 120 100 (30/100) (15/100) (40/100) = 59.1 := 
by
  sorry

end average_minutes_proof_l167_167674


namespace train_speed_approx_72_km_hr_l167_167861

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 14.098872090232781
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := total_distance / crossing_time
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def speed_km_hr : ℝ := speed_m_s * conversion_factor

theorem train_speed_approx_72_km_hr : abs (speed_km_hr - 72) < 0.01 :=
sorry

end train_speed_approx_72_km_hr_l167_167861


namespace probability_not_siblings_l167_167151

-- Define the number of people and the sibling condition
def number_of_people : ℕ := 6
def siblings_count (x : Fin number_of_people) : ℕ := 2

-- Define the probability that two individuals randomly selected are not siblings
theorem probability_not_siblings (P Q : Fin number_of_people) (h : P ≠ Q) :
  let K := number_of_people - 1
  let non_siblings := K - siblings_count P
  (non_siblings / K : ℚ) = 3 / 5 :=
by
  intros
  sorry

end probability_not_siblings_l167_167151


namespace budget_spent_on_utilities_l167_167384

noncomputable def budget_is_correct : Prop :=
  let total_budget := 100
  let salaries := 60
  let r_and_d := 9
  let equipment := 4
  let supplies := 2
  let degrees_in_circle := 360
  let transportation_degrees := 72
  let transportation_percentage := (transportation_degrees * total_budget) / degrees_in_circle
  let known_percentages := salaries + r_and_d + equipment + supplies + transportation_percentage
  let utilities_percentage := total_budget - known_percentages
  utilities_percentage = 5

theorem budget_spent_on_utilities : budget_is_correct :=
  sorry

end budget_spent_on_utilities_l167_167384


namespace point_quadrant_l167_167105

theorem point_quadrant (x y : ℝ) (h : (x + y)^2 = x^2 + y^2 - 2) : 
  ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  sorry

end point_quadrant_l167_167105


namespace total_team_cost_l167_167834

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l167_167834


namespace quadratic_root_relationship_l167_167362

theorem quadratic_root_relationship (a b c : ℂ) (alpha beta : ℂ) (h1 : a ≠ 0) (h2 : alpha + beta = -b / a) (h3 : alpha * beta = c / a) (h4 : beta = 3 * alpha) : 3 * b ^ 2 = 16 * a * c := by
  sorry

end quadratic_root_relationship_l167_167362


namespace product_eq_1280_l167_167056

axiom eq1 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48
axiom eq2 (a b c d : ℝ) : 4 * d + 2 * c = 2 * b
axiom eq3 (a b c d : ℝ) : 4 * b + 2 * c = 2 * a
axiom eq4 (a b c d : ℝ) : c - 2 = d
axiom eq5 (a b c d : ℝ) : d + b = 10

theorem product_eq_1280 (a b c d : ℝ) : 2 * a + 4 * b + 6 * c + 8 * d = 48 → 4 * d + 2 * c = 2 * b → 4 * b + 2 * c = 2 * a → c - 2 = d → d + b = 10 → a * b * c * d = 1280 :=
by 
  intro h1 h2 h3 h4 h5
  -- we put the proof here
  sorry

end product_eq_1280_l167_167056


namespace distance_from_P_to_y_axis_l167_167918

theorem distance_from_P_to_y_axis (P : ℝ × ℝ) :
  (P.2 ^ 2 = -12 * P.1) → (dist P (-3, 0) = 9) → abs P.1 = 6 :=
by
  sorry

end distance_from_P_to_y_axis_l167_167918


namespace complex_inverse_identity_l167_167035

theorem complex_inverse_identity : ∀ (i : ℂ), i^2 = -1 → (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by
  -- Let's introduce the variables and the condition.
  intro i h

  -- Sorry is used to signify the proof is omitted.
  sorry

end complex_inverse_identity_l167_167035


namespace range_of_a_l167_167359

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 1 then x + a / x + 1 else -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f a x ≤ f a y) : -1 ≤ a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l167_167359


namespace basketball_player_possible_scores_l167_167937

-- Define the conditions
def isValidBasketCount (n : Nat) : Prop := n = 7
def isValidBasketValue (v : Nat) : Prop := v = 1 ∨ v = 2 ∨ v = 3

-- Define the theorem statement
theorem basketball_player_possible_scores :
  ∃ (s : Finset ℕ), s = {n | ∃ n1 n2 n3 : Nat, 
                                n1 + n2 + n3 = 7 ∧ 
                                n = 1 * n1 + 2 * n2 + 3 * n3 ∧ 
                                n1 + n2 + n3 = 7 ∧ 
                                n >= 7 ∧ n <= 21} ∧
                                s.card = 15 :=
by
  sorry

end basketball_player_possible_scores_l167_167937


namespace candies_left_after_carlos_ate_l167_167280

def num_red_candies : ℕ := 50
def num_yellow_candies : ℕ := 3 * num_red_candies - 35
def num_blue_candies : ℕ := (2 * num_yellow_candies) / 3
def num_green_candies : ℕ := 20
def num_purple_candies : ℕ := num_green_candies / 2
def num_silver_candies : ℕ := 10
def num_candies_eaten_by_carlos : ℕ := num_yellow_candies + num_green_candies / 2

def total_candies : ℕ := num_red_candies + num_yellow_candies + num_blue_candies + num_green_candies + num_purple_candies + num_silver_candies
def candies_remaining : ℕ := total_candies - num_candies_eaten_by_carlos

theorem candies_left_after_carlos_ate : candies_remaining = 156 := by
  sorry

end candies_left_after_carlos_ate_l167_167280


namespace find_ab_l167_167542

-- Define the "¤" operation
def op (x y : ℝ) := (x + y)^2 - (x - y)^2

-- The Lean 4 theorem statement
theorem find_ab (a b : ℝ) (h : op a b = 24) : a * b = 6 := 
by
  -- We leave the proof as an exercise
  sorry

end find_ab_l167_167542


namespace largest_possible_s_l167_167983

theorem largest_possible_s :
  ∃ s r : ℕ, (r ≥ s) ∧ (s ≥ 5) ∧ (122 * r - 120 * s = r * s) ∧ (s = 121) :=
by sorry

end largest_possible_s_l167_167983


namespace line_through_A_parallel_line_through_B_perpendicular_l167_167563

-- 1. Prove the equation of the line passing through point A(2, 1) and parallel to the line 2x + y - 10 = 0 is 2x + y - 5 = 0.
theorem line_through_A_parallel :
  ∃ (l : ℝ → ℝ), (∀ x, 2 * x + l x - 5 = 0) ∧ (l 2 = 1) ∧ (∃ k, ∀ x, l x = -2 * (x - 2) + k) :=
sorry

-- 2. Prove the equation of the line passing through point B(3, 2) and perpendicular to the line 4x + 5y - 8 = 0 is 5x - 4y - 7 = 0.
theorem line_through_B_perpendicular :
  ∃ (m : ℝ) (l : ℝ → ℝ), (∀ x, 5 * x - 4 * l x - 7 = 0) ∧ (l 3 = 2) ∧ (m = -7) :=
sorry

end line_through_A_parallel_line_through_B_perpendicular_l167_167563


namespace Heather_total_distance_walked_l167_167089

theorem Heather_total_distance_walked :
  let d1 := 0.645
  let d2 := 1.235
  let d3 := 0.875
  let d4 := 1.537
  let d5 := 0.932
  (d1 + d2 + d3 + d4 + d5) = 5.224 := 
by
  sorry -- Proof goes here

end Heather_total_distance_walked_l167_167089


namespace weight_shifted_count_l167_167150

def is_weight_shifted (a b x y : ℕ) : Prop :=
  a + b = 2 * (x + y) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9

theorem weight_shifted_count : 
  ∃ count : ℕ, count = 225 ∧ 
  (∀ (a b x y : ℕ), is_weight_shifted a b x y → count = 225) := 
sorry

end weight_shifted_count_l167_167150


namespace distance_between_chords_l167_167488

theorem distance_between_chords (R AB CD : ℝ) (hR : R = 15) (hAB : AB = 18) (hCD : CD = 24) : 
  ∃ d : ℝ, d = 21 :=
by 
  sorry

end distance_between_chords_l167_167488


namespace average_income_PQ_l167_167068

/-
Conditions:
1. The average monthly income of Q and R is Rs. 5250.
2. The average monthly income of P and R is Rs. 6200.
3. The monthly income of P is Rs. 3000.
-/

def avg_income_QR := 5250
def avg_income_PR := 6200
def income_P := 3000

theorem average_income_PQ :
  ∃ (Q R : ℕ), ((Q + R) / 2 = avg_income_QR) ∧ ((income_P + R) / 2 = avg_income_PR) ∧ 
               (∀ (p q : ℕ), p = income_P → q = (Q + income_P) / 2 → q = 2050) :=
by
  sorry

end average_income_PQ_l167_167068


namespace celia_receives_correct_amount_of_aranha_l167_167844

def borboleta_to_tubarao (b : Int) : Int := 3 * b
def tubarao_to_periquito (t : Int) : Int := 2 * t
def periquito_to_aranha (p : Int) : Int := 3 * p
def macaco_to_aranha (m : Int) : Int := 4 * m
def cobra_to_periquito (c : Int) : Int := 3 * c

def celia_stickers_to_aranha (borboleta tubarao cobra periquito macaco : Int) : Int :=
  let borboleta_to_aranha := periquito_to_aranha (tubarao_to_periquito (borboleta_to_tubarao borboleta))
  let tubarao_to_aranha := periquito_to_aranha (tubarao_to_periquito tubarao)
  let cobra_to_aranha := periquito_to_aranha (cobra_to_periquito cobra)
  let periquito_to_aranha := periquito_to_aranha periquito
  let macaco_to_aranha := macaco_to_aranha macaco
  borboleta_to_aranha + tubarao_to_aranha + cobra_to_aranha + periquito_to_aranha + macaco_to_aranha

theorem celia_receives_correct_amount_of_aranha : 
  celia_stickers_to_aranha 4 5 3 6 6 = 171 := 
by
  simp only [celia_stickers_to_aranha, borboleta_to_tubarao, tubarao_to_periquito, periquito_to_aranha, cobra_to_periquito, macaco_to_aranha]
  -- Here we need to perform the arithmetic steps to verify the sum
  sorry -- This is the placeholder for the actual proof

end celia_receives_correct_amount_of_aranha_l167_167844


namespace find_a2016_l167_167352

-- Define the sequence according to the conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

-- State the main theorem we want to prove
theorem find_a2016 :
  ∃ a : ℕ → ℤ, seq a ∧ a 2016 = -4 :=
by
  sorry

end find_a2016_l167_167352


namespace Juanita_weekday_spending_l167_167601

/- Defining the variables and conditions in the problem -/

def Grant_spending : ℝ := 200
def Sunday_spending : ℝ := 2
def extra_spending : ℝ := 60

-- We need to prove that Juanita spends $0.50 per day from Monday through Saturday on newspapers.

theorem Juanita_weekday_spending :
  (∃ x : ℝ, 6 * 52 * x + 52 * 2 = Grant_spending + extra_spending) -> (∃ x : ℝ, x = 0.5) := by {
  sorry
}

end Juanita_weekday_spending_l167_167601


namespace exists_polyhedron_with_given_vertices_and_edges_l167_167877

theorem exists_polyhedron_with_given_vertices_and_edges :
  ∃ (V : Finset (String)) (E : Finset (Finset (String))),
    V = { "A", "B", "C", "D", "E", "F", "G", "H" } ∧
    E = { { "A", "B" }, { "A", "C" }, { "A", "H" }, { "B", "C" },
          { "B", "D" }, { "C", "D" }, { "D", "E" }, { "E", "F" },
          { "E", "G" }, { "F", "G" }, { "F", "H" }, { "G", "H" } } ∧
    (V.card : ℤ) - (E.card : ℤ) + 6 = 2 :=
by
  sorry

end exists_polyhedron_with_given_vertices_and_edges_l167_167877


namespace percentage_difference_l167_167715

theorem percentage_difference :
  let a := 0.80 * 40
  let b := (4 / 5) * 15
  a - b = 20 := by
sorry

end percentage_difference_l167_167715


namespace expenses_neg_of_income_pos_l167_167807

theorem expenses_neg_of_income_pos :
  ∀ (income expense : Int), income = 5 → expense = -income → expense = -5 :=
by
  intros income expense h_income h_expense
  rw [h_income] at h_expense
  exact h_expense

end expenses_neg_of_income_pos_l167_167807


namespace problem_a_l167_167381

def continuous (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib
def monotonic (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib

theorem problem_a :
  ¬ (∀ (f : ℝ → ℝ), continuous f ∧ (∀ y, ∃ x, f x = y) → monotonic f) := sorry

end problem_a_l167_167381


namespace smallest_integer_of_inequality_l167_167334

theorem smallest_integer_of_inequality :
  ∃ x : ℤ, (8 - 7 * x ≥ 4 * x - 3) ∧ (∀ y : ℤ, (8 - 7 * y ≥ 4 * y - 3) → y ≥ x) ∧ x = 1 :=
sorry

end smallest_integer_of_inequality_l167_167334


namespace repeating_decimal_mul_l167_167904

theorem repeating_decimal_mul (x : ℝ) (hx : x = 0.3333333333333333) :
  x * 12 = 4 :=
sorry

end repeating_decimal_mul_l167_167904


namespace savings_on_discounted_milk_l167_167063

theorem savings_on_discounted_milk :
  let num_gallons := 8
  let price_per_gallon := 3.20
  let discount_rate := 0.25
  let discount_per_gallon := price_per_gallon * discount_rate
  let discounted_price_per_gallon := price_per_gallon - discount_per_gallon
  let total_cost_without_discount := num_gallons * price_per_gallon
  let total_cost_with_discount := num_gallons * discounted_price_per_gallon
  let savings := total_cost_without_discount - total_cost_with_discount
  savings = 6.40 :=
by
  sorry

end savings_on_discounted_milk_l167_167063


namespace range_of_a_l167_167960

open Real

/-- Proposition p: x^2 + 2*a*x + 4 > 0 for all x in ℝ -/
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: the exponential function (3 - 2*a)^x is increasing -/
def q (a : ℝ) : Prop :=
  3 - 2*a > 1

/-- Given p ∧ q, prove that -2 < a < 1 -/
theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a < 1 :=
sorry

end range_of_a_l167_167960


namespace h_inverse_correct_l167_167845

noncomputable def f (x : ℝ) := 4 * x + 7
noncomputable def g (x : ℝ) := 3 * x - 2
noncomputable def h (x : ℝ) := f (g x)
noncomputable def h_inv (y : ℝ) := (y + 1) / 12

theorem h_inverse_correct : ∀ x : ℝ, h_inv (h x) = x :=
by
  intro x
  sorry

end h_inverse_correct_l167_167845


namespace coffee_per_cup_for_weak_l167_167724

-- Defining the conditions
def weak_coffee_cups : ℕ := 12
def strong_coffee_cups : ℕ := 12
def total_coffee_tbsp : ℕ := 36
def weak_increase_factor : ℕ := 1
def strong_increase_factor : ℕ := 2

-- The theorem stating the problem
theorem coffee_per_cup_for_weak :
  ∃ W : ℝ, (weak_coffee_cups * W + strong_coffee_cups * (strong_increase_factor * W) = total_coffee_tbsp) ∧ (W = 1) :=
  sorry

end coffee_per_cup_for_weak_l167_167724


namespace set_intersection_complement_l167_167969

def U : Set ℝ := Set.univ
def A : Set ℝ := { y | ∃ x, x > 0 ∧ y = 4 / x }
def B : Set ℝ := { y | ∃ x, x < 1 ∧ y = 2^x }
def comp_B : Set ℝ := { y | y ≤ 0 } ∪ { y | y ≥ 2 }
def intersection : Set ℝ := { y | y ≥ 2 }

theorem set_intersection_complement :
  A ∩ comp_B = intersection :=
by
  sorry

end set_intersection_complement_l167_167969


namespace remaining_paint_fraction_l167_167811

def initial_paint : ℚ := 1

def paint_day_1 : ℚ := initial_paint - (1/2) * initial_paint
def paint_day_2 : ℚ := paint_day_1 - (1/4) * paint_day_1
def paint_day_3 : ℚ := paint_day_2 - (1/3) * paint_day_2

theorem remaining_paint_fraction : paint_day_3 = 1/4 :=
by
  sorry

end remaining_paint_fraction_l167_167811


namespace initial_maintenance_time_l167_167855

theorem initial_maintenance_time (x : ℝ) 
  (h1 : (1 + (1 / 3)) * x = 60) : 
  x = 45 :=
by
  sorry

end initial_maintenance_time_l167_167855


namespace fourth_group_trees_l167_167569

theorem fourth_group_trees (x : ℕ) :
  5 * 13 = 12 + 15 + 12 + x + 11 → x = 15 :=
by
  sorry

end fourth_group_trees_l167_167569


namespace total_number_of_parts_l167_167443

-- Identify all conditions in the problem: sample size and probability
def sample_size : ℕ := 30
def probability : ℝ := 0.25

-- Statement of the proof problem: The total number of parts N is 120 given the conditions
theorem total_number_of_parts (N : ℕ) (h : (sample_size : ℝ) / N = probability) : N = 120 :=
sorry

end total_number_of_parts_l167_167443


namespace satellite_modular_units_l167_167617

variable (U N S T : ℕ)

def condition1 : Prop := N = (1/8 : ℝ) * S
def condition2 : Prop := T = 4 * S
def condition3 : Prop := U * N = 3 * S

theorem satellite_modular_units
  (h1 : condition1 N S)
  (h2 : condition2 T S)
  (h3 : condition3 U N S) :
  U = 24 :=
sorry

end satellite_modular_units_l167_167617


namespace solve_for_k_l167_167434

theorem solve_for_k : 
  ∃ k : ℤ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := 
by
  use -4
  sorry

end solve_for_k_l167_167434


namespace length_of_X_l167_167814

theorem length_of_X
  {X : ℝ}
  (h1 : 2 + 2 + X = 4 + X)
  (h2 : 3 + 4 + 1 = 8)
  (h3 : ∃ y : ℝ, y * (4 + X) = 29) : 
  X = 4 := sorry

end length_of_X_l167_167814


namespace total_buttons_needed_l167_167729

def shirts_sewn_on_monday := 4
def shirts_sewn_on_tuesday := 3
def shirts_sewn_on_wednesday := 2
def buttons_per_shirt := 5

theorem total_buttons_needed : 
  (shirts_sewn_on_monday + shirts_sewn_on_tuesday + shirts_sewn_on_wednesday) * buttons_per_shirt = 45 :=
by 
  sorry

end total_buttons_needed_l167_167729


namespace henry_change_l167_167066

theorem henry_change (n : ℕ) (p m : ℝ) (h_n : n = 4) (h_p : p = 0.75) (h_m : m = 10) : 
  m - (n * p) = 7 := 
by 
  sorry

end henry_change_l167_167066


namespace solve_system_of_equations_l167_167144

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : x + y = 55) 
  (h2 : x - y = 15) 
  (h3 : x > y) : 
  x = 35 ∧ y = 20 := 
sorry

end solve_system_of_equations_l167_167144


namespace range_of_m_l167_167328

-- Define the constants used in the problem
def a : ℝ := 0.8
def b : ℝ := 1.2

-- Define the logarithmic inequality problem
theorem range_of_m (m : ℝ) : (a^(b^m) < b^(a^m)) → m < 0 := sorry

end range_of_m_l167_167328


namespace qs_length_l167_167668

theorem qs_length
  (PQR : Triangle)
  (PQ QR PR : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 8)
  (h3 : PR = 9)
  (bugs_meet_half_perimeter : PQ + QR + PR = 24)
  (bugs_meet_distance : PQ + qs = 12) :
  qs = 5 :=
by
  sorry

end qs_length_l167_167668


namespace sixth_power_of_sqrt_l167_167895

variable (x : ℝ)
axiom h1 : x = Real.sqrt (2 + Real.sqrt 2)

theorem sixth_power_of_sqrt : x^6 = 16 + 10 * Real.sqrt 2 :=
by {
    sorry
}

end sixth_power_of_sqrt_l167_167895


namespace factorize_expression_l167_167628

variables (a b x : ℝ)

theorem factorize_expression :
    5 * a * (x^2 - 1) - 5 * b * (x^2 - 1) = 5 * (x + 1) * (x - 1) * (a - b) := 
by
  sorry

end factorize_expression_l167_167628


namespace final_number_correct_l167_167414

noncomputable def initial_number : ℝ := 1256
noncomputable def first_increase_rate : ℝ := 3.25
noncomputable def second_increase_rate : ℝ := 1.47

theorem final_number_correct :
  initial_number * first_increase_rate * second_increase_rate = 6000.54 := 
by
  sorry

end final_number_correct_l167_167414


namespace min_val_z_is_7_l167_167621

noncomputable def min_val_z (x y : ℝ) (h : x + 3 * y = 2) : ℝ := 3^x + 27^y + 1

theorem min_val_z_is_7  : ∃ x y : ℝ, x + 3 * y = 2 ∧ min_val_z x y (by sorry) = 7 := sorry

end min_val_z_is_7_l167_167621


namespace equalize_nuts_l167_167015

open Nat

noncomputable def transfer (p1 p2 p3 : ℕ) : Prop :=
  ∃ (m1 m2 m3 : ℕ), 
    m1 ≤ p1 ∧ m1 ≤ p2 ∧ 
    m2 ≤ (p2 + m1) ∧ m2 ≤ p3 ∧ 
    m3 ≤ (p3 + m2) ∧ m3 ≤ (p1 - m1) ∧
    (p1 - m1 + m3 = 16) ∧ 
    (p2 + m1 - m2 = 16) ∧ 
    (p3 + m2 - m3 = 16)

theorem equalize_nuts : transfer 22 14 12 := 
  sorry

end equalize_nuts_l167_167015


namespace ratio_of_people_on_buses_l167_167472

theorem ratio_of_people_on_buses (P_2 P_3 P_4 : ℕ) 
  (h1 : P_1 = 12) 
  (h2 : P_3 = P_2 - 6) 
  (h3 : P_4 = P_1 + 9) 
  (h4 : P_1 + P_2 + P_3 + P_4 = 75) : 
  P_2 / P_1 = 2 := 
by
  sorry

end ratio_of_people_on_buses_l167_167472


namespace students_not_receiving_A_l167_167973

theorem students_not_receiving_A (total_students : ℕ) (students_A_physics : ℕ) (students_A_chemistry : ℕ) (students_A_both : ℕ) (h_total : total_students = 40) (h_A_physics : students_A_physics = 10) (h_A_chemistry : students_A_chemistry = 18) (h_A_both : students_A_both = 6) : (total_students - ((students_A_physics + students_A_chemistry) - students_A_both)) = 18 := 
by
  sorry

end students_not_receiving_A_l167_167973


namespace new_area_shortening_other_side_l167_167821

-- Define the dimensions of the original card
def original_length : ℕ := 5
def original_width : ℕ := 7

-- Define the shortened length and the resulting area after shortening one side by 2 inches
def shortened_length_1 := original_length - 2
def new_area_1 : ℕ := shortened_length_1 * original_width
def condition_1 : Prop := new_area_1 = 21

-- Prove that shortening the width by 2 inches results in an area of 25 square inches
theorem new_area_shortening_other_side : condition_1 → (original_length * (original_width - 2) = 25) :=
by
  intro h
  sorry

end new_area_shortening_other_side_l167_167821


namespace min_value_sqrt_sum_l167_167157

open Real

theorem min_value_sqrt_sum (x : ℝ) : 
    ∃ c : ℝ, (∀ x : ℝ, c ≤ sqrt (x^2 - 4 * x + 13) + sqrt (x^2 - 10 * x + 26)) ∧ 
             (sqrt ((17/4)^2 - 4 * (17/4) + 13) + sqrt ((17/4)^2 - 10 * (17/4) + 26) = 5 ∧ c = 5) := 
by
  sorry

end min_value_sqrt_sum_l167_167157


namespace sum_of_brothers_ages_l167_167312

theorem sum_of_brothers_ages (Bill Eric: ℕ) 
  (h1: 4 = Bill - Eric) 
  (h2: Bill = 16) : 
  Bill + Eric = 28 := 
by 
  sorry

end sum_of_brothers_ages_l167_167312


namespace math_equivalence_proof_problem_l167_167094

-- Define the initial radii in L0
def r1 := 50^2
def r2 := 53^2

-- Define the formula for constructing a new circle in subsequent layers
def next_radius (r1 r2 : ℕ) : ℕ :=
  (r1 * r2) / ((Nat.sqrt r1 + Nat.sqrt r2)^2)

-- Compute the sum of reciprocals of the square roots of the radii 
-- of all circles up to and including layer L6
def sum_of_reciprocals_of_square_roots_up_to_L6 : ℚ :=
  let initial_sum := (1 / (50 : ℚ)) + (1 / (53 : ℚ))
  (127 * initial_sum) / (50 * 53)

theorem math_equivalence_proof_problem : 
  sum_of_reciprocals_of_square_roots_up_to_L6 = 13021 / 2650 := 
sorry

end math_equivalence_proof_problem_l167_167094


namespace initial_investment_l167_167692

theorem initial_investment (P : ℝ) 
  (h1: ∀ (r : ℝ) (n : ℕ), r = 0.20 ∧ n = 3 → P * (1 + r)^n = P * 1.728)
  (h2: ∀ (A : ℝ), A = P * 1.728 → 3 * A = 5.184 * P)
  (h3: ∀ (P_new : ℝ) (r_new : ℝ), P_new = 5.184 * P ∧ r_new = 0.15 → P_new * (1 + r_new) = 5.9616 * P)
  (h4: 5.9616 * P = 59616)
  : P = 10000 :=
sorry

end initial_investment_l167_167692


namespace minimum_purchase_price_mod6_l167_167756

theorem minimum_purchase_price_mod6 
  (coin_values : List ℕ)
  (h1 : (1 : ℕ) ∈ coin_values)
  (h15 : (15 : ℕ) ∈ coin_values)
  (h50 : (50 : ℕ) ∈ coin_values)
  (A C : ℕ)
  (k : ℕ)
  (hA : A ≡ k [MOD 7])
  (hC : C ≡ k + 1 [MOD 7])
  (hP : ∃ P, P = A - C) : 
  ∃ P, P ≡ 6 [MOD 7] ∧ P > 0 :=
by
  sorry

end minimum_purchase_price_mod6_l167_167756


namespace range_of_a_l167_167907

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Lean statement asserting the requirement
theorem range_of_a (a : ℝ) (h : A ⊆ B a ∧ A ≠ B a) : 2 < a := by
  sorry

end range_of_a_l167_167907


namespace arithmetic_sum_problem_l167_167970

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The problem statement -/
theorem arithmetic_sum_problem
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms S a)
  (h_S10 : S 10 = 4) :
  a 3 + a 8 = 4 / 5 := 
sorry

end arithmetic_sum_problem_l167_167970


namespace inverse_proportion_quadrants_l167_167152

theorem inverse_proportion_quadrants (m : ℝ) : (∀ (x : ℝ), x ≠ 0 → y = (m - 2) / x → (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0)) ↔ m > 2 :=
by
  sorry

end inverse_proportion_quadrants_l167_167152


namespace maximize_profit_l167_167441

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

theorem maximize_profit :
  let max_price := 14
  let max_profit := 360
  (∀ x > 10, profit x ≤ profit max_price) ∧ profit max_price = max_profit :=
by
  let max_price := 14
  let max_profit := 360
  sorry

end maximize_profit_l167_167441


namespace maximum_value_at_vertex_l167_167953

-- Defining the parabola as a function
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Defining the vertex condition
def vertex_condition (a b c : ℝ) := ∀ x : ℝ, parabola a b c x = a * x^2 + b * x + c

-- Defining the condition that the parabola opens downward
def opens_downward (a : ℝ) := a < 0

-- Defining the vertex coordinates condition
def vertex_coordinates (a b c : ℝ) := 
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ parabola a b c x₀ = y₀

-- The main theorem statement
theorem maximum_value_at_vertex (a b c : ℝ) (h1 : opens_downward a) (h2 : vertex_coordinates a b c) : ∃ y₀, y₀ = -3 ∧ ∀ x : ℝ, parabola a b c x ≤ y₀ :=
by
  sorry

end maximum_value_at_vertex_l167_167953


namespace layers_removed_l167_167810

theorem layers_removed (n : ℕ) (original_volume remaining_volume side_length : ℕ) :
  original_volume = side_length^3 →
  remaining_volume = (side_length - 2 * n)^3 →
  original_volume = 1000 →
  remaining_volume = 512 →
  side_length = 10 →
  n = 1 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end layers_removed_l167_167810


namespace abc_product_l167_167020

/-- Given a b c + a b + b c + a c + a + b + c = 164 -/
theorem abc_product :
  ∃ (a b c : ℕ), a * b * c + a * b + b * c + a * c + a + b + c = 164 ∧ a * b * c = 80 :=
by
  sorry

end abc_product_l167_167020


namespace rectangle_ratio_l167_167197

theorem rectangle_ratio (w : ℝ) (h : ℝ)
  (hw : h = 10)   -- Length is 10
  (hp : 2 * w + 2 * h = 30) :  -- Perimeter is 30
  w / h = 1 / 2 :=             -- Ratio of width to length is 1/2
by
  -- Pending proof
  sorry

end rectangle_ratio_l167_167197


namespace smallest_n_modulo_l167_167650

theorem smallest_n_modulo :
  ∃ (n : ℕ), 0 < n ∧ 1031 * n % 30 = 1067 * n % 30 ∧ ∀ (m : ℕ), 0 < m ∧ 1031 * m % 30 = 1067 * m % 30 → n ≤ m :=
by
  sorry

end smallest_n_modulo_l167_167650


namespace smallest_y_for_perfect_cube_l167_167546

-- Define the given conditions
def x : ℕ := 5 * 24 * 36

-- State the theorem to prove
theorem smallest_y_for_perfect_cube (y : ℕ) (h : y = 50) : 
  ∃ y, (x * y) % (y * y * y) = 0 :=
by
  sorry

end smallest_y_for_perfect_cube_l167_167546


namespace frank_ryan_problem_ratio_l167_167146

theorem frank_ryan_problem_ratio 
  (bill_problems : ℕ)
  (h1 : bill_problems = 20)
  (ryan_problems : ℕ)
  (h2 : ryan_problems = 2 * bill_problems)
  (frank_problems_per_type : ℕ)
  (h3 : frank_problems_per_type = 30)
  (types : ℕ)
  (h4 : types = 4) : 
  frank_problems_per_type * types / ryan_problems = 3 := by
  sorry

end frank_ryan_problem_ratio_l167_167146


namespace segment_ratio_ae_ad_l167_167982

/-- Given points B, C, and E lie on line segment AD, and the following conditions:
  1. The length of segment AB is twice the length of segment BD.
  2. The length of segment AC is 5 times the length of segment CD.
  3. The length of segment BE is one-third the length of segment EC.
Prove that the fraction of the length of segment AD that segment AE represents is 17/24. -/
theorem segment_ratio_ae_ad (AB BD AC CD BE EC AD AE : ℝ)
    (h1 : AB = 2 * BD)
    (h2 : AC = 5 * CD)
    (h3 : BE = (1/3) * EC)
    (h4 : AD = 6 * CD)
    (h5 : AE = 4.25 * CD) :
    AE / AD = 17 / 24 := 
  by 
  sorry

end segment_ratio_ae_ad_l167_167982


namespace arnold_protein_intake_l167_167057

theorem arnold_protein_intake :
  (∀ p q s : ℕ,  p = 18 / 2 ∧ q = 21 ∧ s = 56 → (p + q + s = 86)) := by
  sorry

end arnold_protein_intake_l167_167057


namespace complement_correct_l167_167903

universe u

-- We define sets A and B
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

-- Define the complement of B with respect to A
def complement (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- The theorem we need to prove
theorem complement_correct : complement A B = {2, 4} := 
  sorry

end complement_correct_l167_167903


namespace find_prime_and_int_solutions_l167_167323

-- Define the conditions
def is_solution (p x : ℕ) : Prop :=
  x^(p-1) ∣ (p-1)^x + 1

-- Define the statement to be proven
theorem find_prime_and_int_solutions :
  ∀ p x : ℕ, Prime p → (1 ≤ x ∧ x ≤ 2 * p) →
  (is_solution p x ↔ 
    (p = 2 ∧ (x = 1 ∨ x = 2)) ∨ 
    (p = 3 ∧ (x = 1 ∨ x = 3)) ∨
    (x = 1))
:=
by sorry

end find_prime_and_int_solutions_l167_167323


namespace cost_of_song_book_l167_167661

-- Define the costs as constants
def cost_trumpet : ℝ := 149.16
def cost_music_tool : ℝ := 9.98
def total_spent : ℝ := 163.28

-- Define the statement to prove
theorem cost_of_song_book : total_spent - (cost_trumpet + cost_music_tool) = 4.14 := 
by
  sorry

end cost_of_song_book_l167_167661


namespace tens_digit_of_9_to_2023_l167_167980

theorem tens_digit_of_9_to_2023 :
  (9^2023 % 100) / 10 % 10 = 8 :=
sorry

end tens_digit_of_9_to_2023_l167_167980


namespace larger_integer_value_l167_167253

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l167_167253


namespace complement_S_union_T_eq_l167_167279

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3 * x - 4 ≤ 0}
noncomputable def complement_S := {x : ℝ | x ≤ -2}

theorem complement_S_union_T_eq : (complement_S ∪ T) = {x : ℝ | x ≤ 1} := by 
  sorry

end complement_S_union_T_eq_l167_167279


namespace solve_eq_l167_167857

theorem solve_eq (a b : ℕ) : a * a = b * (b + 7) ↔ (a, b) = (0, 0) ∨ (a, b) = (12, 9) :=
by
  sorry

end solve_eq_l167_167857


namespace boys_to_admit_or_expel_l167_167541

-- Definitions from the conditions
def total_students : ℕ := 500

def girls_percent (x : ℕ) : ℕ := (x * total_students) / 100

-- Definition of the calculation under the new policy
def required_boys : ℕ := (total_students * 3) / 5

-- Main statement we need to prove
theorem boys_to_admit_or_expel (x : ℕ) (htotal : x + girls_percent x = total_students) :
  required_boys - x = 217 := by
  sorry

end boys_to_admit_or_expel_l167_167541


namespace units_digit_of_sum_sequence_is_8_l167_167573

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit_sum_sequence : ℕ :=
  let term (n : ℕ) := (factorial n + n * n) % 10
  (term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7 + term 8 + term 9) % 10

theorem units_digit_of_sum_sequence_is_8 :
  units_digit_sum_sequence = 8 :=
sorry

end units_digit_of_sum_sequence_is_8_l167_167573


namespace gcd_of_360_and_150_l167_167154

theorem gcd_of_360_and_150 : Nat.gcd 360 150 = 30 := 
by
  sorry

end gcd_of_360_and_150_l167_167154


namespace log_problem_l167_167764

open Real

noncomputable def lg (x : ℝ) := log x / log 10

theorem log_problem :
  lg 2 ^ 2 + lg 2 * lg 5 + lg 5 = 1 :=
by
  sorry

end log_problem_l167_167764


namespace sqrt_of_sum_of_powers_l167_167702

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l167_167702


namespace inequality_ab_l167_167539

theorem inequality_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := 
sorry

end inequality_ab_l167_167539


namespace basic_computer_price_l167_167765

variables (C P : ℕ)

theorem basic_computer_price (h1 : C + P = 2500)
                            (h2 : C + 500 + P = 6 * P) : C = 2000 :=
by
  sorry

end basic_computer_price_l167_167765


namespace minimize_G_l167_167149

noncomputable def F (p q : ℝ) : ℝ :=
  2 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def G (p : ℝ) : ℝ :=
  max (F p 0) (F p 1)

theorem minimize_G :
  ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 0.75 → G p = G 0 → p = 0 :=
by
  intro p hp hG
  -- The proof goes here
  sorry

end minimize_G_l167_167149


namespace isosceles_right_triangle_example_l167_167486

theorem isosceles_right_triangle_example :
  (5 = 5) ∧ (5^2 + 5^2 = (5 * Real.sqrt 2)^2) :=
by {
  sorry
}

end isosceles_right_triangle_example_l167_167486


namespace mean_equality_l167_167673

theorem mean_equality (y : ℝ) (h : (6 + 9 + 18) / 3 = (12 + y) / 2) : y = 10 :=
by sorry

end mean_equality_l167_167673


namespace son_age_is_15_l167_167269

theorem son_age_is_15 (S F : ℕ) (h1 : 2 * S + F = 70) (h2 : 2 * F + S = 95) (h3 : F = 40) :
  S = 15 :=
by {
  sorry
}

end son_age_is_15_l167_167269


namespace max_possible_N_l167_167589

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l167_167589


namespace solve_fraction_l167_167496

theorem solve_fraction (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : (x - 2) * (x + 1) ≠ 0) : x = 1 := 
sorry

end solve_fraction_l167_167496


namespace solution1_solution2_l167_167174

noncomputable def problem1 : ℝ :=
  40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12

theorem solution1 : problem1 = 43 := by
  sorry

noncomputable def problem2 : ℝ :=
  (-1 : ℝ) ^ 2021 + |(-9 : ℝ)| * (2 / 3) + (-3) / (1 / 5)

theorem solution2 : problem2 = -10 := by
  sorry

end solution1_solution2_l167_167174


namespace ConfuciusBirthYear_l167_167940

-- Definitions based on the conditions provided
def birthYearAD (year : Int) : Int := year

def birthYearBC (year : Int) : Int := -year

theorem ConfuciusBirthYear :
  birthYearBC 551 = -551 :=
by
  sorry

end ConfuciusBirthYear_l167_167940


namespace zeros_of_f_l167_167153

def f (x : ℝ) : ℝ := (x^2 - 3 * x) * (x + 4)

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = 0 ∨ x = 3 ∨ x = -4 := by
  sorry

end zeros_of_f_l167_167153


namespace pounds_of_sugar_l167_167140

theorem pounds_of_sugar (x p : ℝ) (h1 : x * p = 216) (h2 : (x + 3) * (p - 1) = 216) : x = 24 :=
sorry

end pounds_of_sugar_l167_167140


namespace no_triangles_if_all_horizontal_removed_l167_167570

/-- 
Given a figure that consists of 40 identical toothpicks, making up a symmetric figure with 
additional rows on the top and bottom. We need to prove that removing all 40 horizontal toothpicks 
ensures there are no remaining triangles in the figure.
-/
theorem no_triangles_if_all_horizontal_removed
  (initial_toothpicks : ℕ)
  (horizontal_toothpicks_in_figure : ℕ) 
  (rows : ℕ)
  (top_row : ℕ)
  (second_row : ℕ)
  (third_row : ℕ)
  (fourth_row : ℕ)
  (bottom_row : ℕ)
  (additional_rows : ℕ)
  (triangles_for_upward : ℕ)
  (triangles_for_downward : ℕ):
  initial_toothpicks = 40 →
  horizontal_toothpicks_in_figure = top_row + second_row + third_row + fourth_row + bottom_row →
  rows = 5 →
  top_row = 5 →
  second_row = 10 →
  third_row = 10 →
  fourth_row = 10 →
  bottom_row = 5 →
  additional_rows = 2 →
  triangles_for_upward = 15 →
  triangles_for_downward = 10 →
  horizontal_toothpicks_in_figure = 40 → 
  ∀ toothpicks_removed, toothpicks_removed = 40 →
  no_triangles_remain :=
by
  intros
  sorry

end no_triangles_if_all_horizontal_removed_l167_167570


namespace total_cost_38_pencils_56_pens_l167_167981

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l167_167981


namespace arrange_numbers_l167_167658

noncomputable def a := (10^100)^10
noncomputable def b := 10^(10^10)
noncomputable def c := Nat.factorial 1000000
noncomputable def d := (Nat.factorial 100)^10

theorem arrange_numbers :
  a < d ∧ d < c ∧ c < b := 
sorry

end arrange_numbers_l167_167658


namespace solve_eq1_solve_system_l167_167438

theorem solve_eq1 : ∃ x y : ℝ, (3 / x) + (2 / y) = 4 :=
by
  use 1
  use 2
  sorry

theorem solve_system :
  ∃ x y : ℝ,
    (3 / x + 2 / y = 4) ∧ (5 / x - 6 / y = 2) ∧ (x = 1) ∧ (y = 2) :=
by
  use 1
  use 2
  sorry

end solve_eq1_solve_system_l167_167438


namespace evaluate_expression_l167_167976

theorem evaluate_expression : 4^1 + 3^2 - 2^3 + 1^4 = 6 := by
  -- We will skip the proof steps here using sorry
  sorry

end evaluate_expression_l167_167976


namespace integer_solutions_exist_l167_167476

theorem integer_solutions_exist (R₀ : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℤ), (x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃) ∧ (R₀ < x₁) ∧ (R₀ < x₂) ∧ (R₀ < x₃) := 
sorry

end integer_solutions_exist_l167_167476


namespace maximize_profit_l167_167373

noncomputable def annual_profit : ℝ → ℝ
| x => if x < 80 then - (1/3) * x^2 + 40 * x - 250 
       else 1200 - (x + 10000 / x)

theorem maximize_profit : ∃ x : ℝ, x = 100 ∧ annual_profit x = 1000 :=
by
  sorry

end maximize_profit_l167_167373


namespace max_a_value_l167_167050

theorem max_a_value (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 3| ≥ 2 * a) : a ≤ 1 :=
sorry

end max_a_value_l167_167050


namespace temperature_difference_l167_167218

theorem temperature_difference (H L : ℝ) (hH : H = 8) (hL : L = -2) :
  H - L = 10 :=
by
  rw [hH, hL]
  norm_num

end temperature_difference_l167_167218


namespace exponent_division_l167_167652

theorem exponent_division (h1 : 27 = 3^3) : 3^18 / 27^3 = 19683 := by
  sorry

end exponent_division_l167_167652


namespace rahul_matches_played_l167_167440

theorem rahul_matches_played
  (current_avg runs_today new_avg : ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 69)
  (h3 : new_avg = 54)
  : ∃ m : ℕ, ((51 * m + 69) / (m + 1) = 54) ∧ (m = 5) :=
by
  sorry

end rahul_matches_played_l167_167440


namespace triangle_area_inequality_l167_167927

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

end triangle_area_inequality_l167_167927


namespace set_intersection_l167_167012

open Set

/-- Given sets M and N as defined below, we wish to prove that their complements and intersections work as expected. -/
theorem set_intersection (R : Set ℝ)
  (M : Set ℝ := {x | x > 1})
  (N : Set ℝ := {x | abs x ≤ 2})
  (R_universal : R = univ) :
  ((compl M) ∩ N) = Icc (-2 : ℝ) (1 : ℝ) := by
  sorry

end set_intersection_l167_167012


namespace total_amount_divided_l167_167179

variables (T x : ℝ)
variables (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
variables (h₂ : T - x = 1100)

theorem total_amount_divided (T x : ℝ) 
  (h₁ : 0.06 * x + 0.05 * (T - x) = 85) 
  (h₂ : T - x = 1100) : 
  T = 1600 := 
sorry

end total_amount_divided_l167_167179


namespace canoes_more_than_kayaks_l167_167010

theorem canoes_more_than_kayaks (C K : ℕ)
  (h1 : 14 * C + 15 * K = 288)
  (h2 : C = 3 * K / 2) :
  C - K = 4 :=
sorry

end canoes_more_than_kayaks_l167_167010


namespace probability_same_color_l167_167236

theorem probability_same_color (pairs : ℕ) (total_shoes : ℕ) (select_shoes : ℕ)
  (h_pairs : pairs = 6) 
  (h_total_shoes : total_shoes = 12) 
  (h_select_shoes : select_shoes = 2) : 
  (Nat.choose total_shoes select_shoes > 0) → 
  (Nat.div (pairs * (Nat.choose 2 2)) (Nat.choose total_shoes select_shoes) = 1/11) :=
by
  sorry

end probability_same_color_l167_167236
