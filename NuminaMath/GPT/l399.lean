import Mathlib

namespace max_diff_real_roots_l399_39964

-- Definitions of the quadratic equations
def eq1 (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq2 (a b c x : ℝ) : Prop := b * x^2 + c * x + a = 0
def eq3 (a b c x : ℝ) : Prop := c * x^2 + a * x + b = 0

-- The proof statement
theorem max_diff_real_roots (a b c : ℝ) (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  ∃ x y : ℝ, eq1 a b c x ∧ eq1 a b c y ∧ eq2 a b c x ∧ eq2 a b c y ∧ eq3 a b c x ∧ eq3 a b c y ∧ 
  abs (x - y) = 0 := sorry

end max_diff_real_roots_l399_39964


namespace avg_cost_apple_tv_200_l399_39919

noncomputable def average_cost_apple_tv (iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost: ℝ) : ℝ :=
  (overall_avg_cost * (iphones_sold + ipads_sold + apple_tvs_sold) - (iphones_sold * iphone_cost + ipads_sold * ipad_cost)) / apple_tvs_sold

theorem avg_cost_apple_tv_200 :
  let iphones_sold := 100
  let ipads_sold := 20
  let apple_tvs_sold := 80
  let iphone_cost := 1000
  let ipad_cost := 900
  let overall_avg_cost := 670
  average_cost_apple_tv iphones_sold ipads_sold apple_tvs_sold iphone_cost ipad_cost overall_avg_cost = 200 :=
by
  sorry

end avg_cost_apple_tv_200_l399_39919


namespace largest_n_satisfies_conditions_l399_39952

theorem largest_n_satisfies_conditions :
  ∃ (n m a : ℤ), n = 313 ∧ n^2 = (m + 1)^3 - m^3 ∧ ∃ (k : ℤ), 2 * n + 103 = k^2 :=
by
  sorry

end largest_n_satisfies_conditions_l399_39952


namespace exists_four_scientists_l399_39972

theorem exists_four_scientists {n : ℕ} (h1 : n = 50)
  (knows : Fin n → Finset (Fin n))
  (h2 : ∀ x, (knows x).card ≥ 25) :
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  a ≠ c ∧ b ≠ d ∧
  a ∈ knows b ∧ b ∈ knows c ∧ c ∈ knows d ∧ d ∈ knows a :=
by
  sorry

end exists_four_scientists_l399_39972


namespace total_length_of_scale_l399_39949

theorem total_length_of_scale 
  (n : ℕ) (len_per_part : ℕ) 
  (h_n : n = 5) 
  (h_len_per_part : len_per_part = 25) :
  n * len_per_part = 125 :=
by
  sorry

end total_length_of_scale_l399_39949


namespace geometric_sequence_sum_l399_39921

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
    (h : ∀ n : ℕ, a (n + 1) = a n * r) 
    (h1 : a 1 + a 2 = 40) 
    (h2 : a 3 + a 4 = 60) : 
    a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l399_39921


namespace problem_statement_l399_39930

noncomputable def square : ℝ := sorry -- We define a placeholder
noncomputable def pentagon : ℝ := sorry -- We define a placeholder

axiom eq1 : 2 * square + 4 * pentagon = 25
axiom eq2 : 3 * square + 3 * pentagon = 22

theorem problem_statement : 4 * pentagon = 20.67 := 
by
  sorry

end problem_statement_l399_39930


namespace find_rth_term_l399_39993

theorem find_rth_term (n r : ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 4 * n + 5 * n^2) :
  r > 0 → (S r) - (S (r - 1)) = 10 * r - 1 :=
by
  intro h
  have hr_pos := h
  sorry

end find_rth_term_l399_39993


namespace natural_number_triplets_l399_39912

theorem natural_number_triplets (x y z : ℕ) : 
  3^x + 4^y = 5^z → 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by 
  sorry

end natural_number_triplets_l399_39912


namespace find_length_QR_l399_39996

-- Conditions
variables {D E F Q R : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace Q] [MetricSpace R]
variables {DE EF DF QR : ℝ} (tangent : Q = E ∧ R = D)
variables (t₁ : de = 5) (t₂ : ef = 12) (t₃ : df = 13)

-- Problem: Prove that QR = 5 given the conditions.
theorem find_length_QR : QR = 5 :=
sorry

end find_length_QR_l399_39996


namespace probability_after_first_new_draw_is_five_ninths_l399_39980

-- Defining the conditions in Lean
def total_balls : ℕ := 10
def new_balls : ℕ := 6
def old_balls : ℕ := 4

def balls_remaining_after_first_draw : ℕ := total_balls - 1
def new_balls_after_first_draw : ℕ := new_balls - 1

-- Using the classic probability definition
def probability_of_drawing_second_new_ball := (new_balls_after_first_draw : ℚ) / (balls_remaining_after_first_draw : ℚ)

-- Stating the theorem to be proved
theorem probability_after_first_new_draw_is_five_ninths :
  probability_of_drawing_second_new_ball = 5/9 := sorry

end probability_after_first_new_draw_is_five_ninths_l399_39980


namespace value_of_g_at_13_l399_39963

-- Define the function g
def g (n : ℕ) : ℕ := n^2 + n + 23

-- The theorem to prove
theorem value_of_g_at_13 : g 13 = 205 := by
  -- Rewrite using the definition of g
  unfold g
  -- Perform the arithmetic
  sorry

end value_of_g_at_13_l399_39963


namespace sum_of_squares_not_divisible_by_4_or_8_l399_39935

theorem sum_of_squares_not_divisible_by_4_or_8 (n : ℤ) (h : n % 2 = 1) :
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  ¬(4 ∣ sum_squares ∨ 8 ∣ sum_squares) :=
by
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  sorry

end sum_of_squares_not_divisible_by_4_or_8_l399_39935


namespace same_number_assigned_to_each_point_l399_39936

namespace EqualNumberAssignment

def is_arithmetic_mean (f : ℤ × ℤ → ℕ) (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

theorem same_number_assigned_to_each_point (f : ℤ × ℤ → ℕ) :
  (∀ p : ℤ × ℤ, is_arithmetic_mean f p) → ∃ m : ℕ, ∀ p : ℤ × ℤ, f p = m :=
by
  intros h
  sorry

end EqualNumberAssignment

end same_number_assigned_to_each_point_l399_39936


namespace problem_l399_39999

theorem problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1 / 2) :
  (1 - x) / (1 + x) * (1 - y) / (1 + y) * (1 - z) / (1 + z) ≥ 1 / 3 :=
by
  sorry

end problem_l399_39999


namespace total_number_of_boys_in_all_class_sections_is_380_l399_39925

theorem total_number_of_boys_in_all_class_sections_is_380 :
  let students_section1 := 160
  let students_section2 := 200
  let students_section3 := 240
  let girls_section1 := students_section1 / 4
  let boys_section1 := students_section1 - girls_section1
  let boys_section2 := (3 / 5) * students_section2
  let total_parts := 7 + 5
  let boys_section3 := (7 / total_parts) * students_section3
  boys_section1 + boys_section2 + boys_section3 = 380 :=
sorry

end total_number_of_boys_in_all_class_sections_is_380_l399_39925


namespace thomas_weekly_wage_l399_39989

theorem thomas_weekly_wage (monthly_wage : ℕ) (weeks_in_month : ℕ) (weekly_wage : ℕ) 
    (h1 : monthly_wage = 19500) (h2 : weeks_in_month = 4) :
    weekly_wage = 4875 :=
by
  have h3 : weekly_wage = monthly_wage / weeks_in_month := sorry
  rw [h1, h2] at h3
  exact h3

end thomas_weekly_wage_l399_39989


namespace intersection_M_N_l399_39901

noncomputable def M := {x : ℝ | x > 1}
noncomputable def N := {x : ℝ | x < 2}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l399_39901


namespace minimum_value_of_reciprocal_squares_l399_39953

theorem minimum_value_of_reciprocal_squares
  (a b : ℝ)
  (h : a ≠ 0 ∧ b ≠ 0)
  (h_eq : (a^2) + 4 * (b^2) = 9)
  : (1/(a^2) + 1/(b^2)) = 1 :=
sorry

end minimum_value_of_reciprocal_squares_l399_39953


namespace vampire_pints_per_person_l399_39932

-- Definitions based on conditions
def gallons_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7
def people_per_day : ℕ := 4

-- The statement to be proven
theorem vampire_pints_per_person :
  (gallons_per_week * pints_per_gallon) / (days_per_week * people_per_day) = 2 :=
by
  sorry

end vampire_pints_per_person_l399_39932


namespace distinct_real_roots_absolute_sum_l399_39928

theorem distinct_real_roots_absolute_sum {r1 r2 p : ℝ} (h_root1 : r1 ^ 2 + p * r1 + 7 = 0) 
(h_root2 : r2 ^ 2 + p * r2 + 7 = 0) (h_distinct : r1 ≠ r2) : 
|r1 + r2| > 2 * Real.sqrt 7 := 
sorry

end distinct_real_roots_absolute_sum_l399_39928


namespace dot_product_a_b_l399_39920

open Real

noncomputable def cos_deg (x : ℝ) := cos (x * π / 180)
noncomputable def sin_deg (x : ℝ) := sin (x * π / 180)

theorem dot_product_a_b :
  let a_magnitude := 2 * cos_deg 15
  let b_magnitude := 4 * sin_deg 15
  let angle_ab := 30
  a_magnitude * b_magnitude * cos_deg angle_ab = sqrt 3 :=
by
  -- proof omitted
  sorry

end dot_product_a_b_l399_39920


namespace sqrt_product_simplifies_l399_39973

theorem sqrt_product_simplifies (p : ℝ) : 
  Real.sqrt (12 * p) * Real.sqrt (20 * p) * Real.sqrt (15 * p^2) = 60 * p^2 := 
by
  sorry

end sqrt_product_simplifies_l399_39973


namespace circle_properties_l399_39903

theorem circle_properties (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 4 * y + 5 * m = 0) →
  (m < 1 ∨ m > 4) ∧
  (m = -2 → ∃ d : ℝ, d = 2 * Real.sqrt (18 - 5)) :=
by
  sorry

end circle_properties_l399_39903


namespace second_candidate_marks_l399_39924

variable (T : ℝ) (pass_mark : ℝ := 160)

-- Conditions
def condition1 : Prop := 0.20 * T + 40 = pass_mark
def condition2 : Prop := 0.30 * T - pass_mark > 0 

-- The statement we want to prove
theorem second_candidate_marks (h1 : condition1 T) (h2 : condition2 T) : 
  (0.30 * T - pass_mark = 20) :=
by 
  -- Skipping proof steps as per the guidelines
  sorry

end second_candidate_marks_l399_39924


namespace natural_numbers_satisfying_conditions_l399_39945

variable (a b : ℕ)

theorem natural_numbers_satisfying_conditions :
  (90 < a + b ∧ a + b < 100) ∧ (0.9 < (a : ℝ) / b ∧ (a : ℝ) / b < 0.91) ↔ (a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52) := by
  sorry

end natural_numbers_satisfying_conditions_l399_39945


namespace radius_any_positive_real_l399_39948

theorem radius_any_positive_real (r : ℝ) (h₁ : r > 0) 
    (h₂ : r * (2 * Real.pi * r) = 2 * Real.pi * r^2) : True :=
by
  sorry

end radius_any_positive_real_l399_39948


namespace problem_solution_l399_39983

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 :=
by
  sorry

end problem_solution_l399_39983


namespace simplify_expression_l399_39954

variable (x y : ℝ)

theorem simplify_expression : 3 * x^2 * y * (2 / (9 * x^3 * y)) = 2 / (3 * x) :=
by sorry

end simplify_expression_l399_39954


namespace compare_magnitudes_l399_39902

theorem compare_magnitudes : -0.5 > -0.75 :=
by
  have h1 : |(-0.5: ℝ)| = 0.5 := by norm_num
  have h2 : |(-0.75: ℝ)| = 0.75 := by norm_num
  have h3 : (0.5: ℝ) < 0.75 := by norm_num
  sorry

end compare_magnitudes_l399_39902


namespace ants_meet_at_QS_l399_39981

theorem ants_meet_at_QS (P Q R S : Type)
  (dist_PQ : Nat)
  (dist_QR : Nat)
  (dist_PR : Nat)
  (ants_meet : 2 * (dist_PQ + (5 : Nat)) = dist_PQ + dist_QR + dist_PR)
  (perimeter : dist_PQ + dist_QR + dist_PR = 24)
  (distance_each_ant_crawls : (dist_PQ + 5) = 12) :
  5 = 5 :=
by
  sorry

end ants_meet_at_QS_l399_39981


namespace conor_total_vegetables_weekly_l399_39978

def conor_vegetables_daily (e c p o z : ℕ) : ℕ :=
  e + c + p + o + z

def conor_vegetables_weekly (vegetables_daily days_worked : ℕ) : ℕ :=
  vegetables_daily * days_worked

theorem conor_total_vegetables_weekly :
  conor_vegetables_weekly (conor_vegetables_daily 12 9 8 15 7) 6 = 306 := by
  sorry

end conor_total_vegetables_weekly_l399_39978


namespace remainder_of_2_pow_87_plus_3_mod_7_l399_39908

theorem remainder_of_2_pow_87_plus_3_mod_7 : (2^87 + 3) % 7 = 4 := by
  sorry

end remainder_of_2_pow_87_plus_3_mod_7_l399_39908


namespace probability_calculations_l399_39938

-- Define the number of students
def total_students : ℕ := 2006

-- Number of students eliminated in the first step
def eliminated_students : ℕ := 6

-- Number of students remaining after elimination
def remaining_students : ℕ := total_students - eliminated_students

-- Number of students to be selected in the second step
def selected_students : ℕ := 50

-- Calculate the probability of a specific student being eliminated
def elimination_probability := (6 : ℚ) / total_students

-- Calculate the probability of a specific student being selected from the remaining students
def selection_probability := (50 : ℚ) / remaining_students

-- The theorem to prove our equivalent proof problem
theorem probability_calculations :
  elimination_probability = (3 : ℚ) / 1003 ∧
  selection_probability = (25 : ℚ) / 1003 :=
by
  sorry

end probability_calculations_l399_39938


namespace triangle_inequality_l399_39914

noncomputable def f (K : ℝ) (x : ℝ) : ℝ :=
  (x^4 + K * x^2 + 1) / (x^4 + x^2 + 1)

theorem triangle_inequality (K : ℝ) (a b c : ℝ) :
  (-1 / 2) < K ∧ K < 4 → ∃ (A B C : ℝ), A = f K a ∧ B = f K b ∧ C = f K c ∧ A + B > C ∧ A + C > B ∧ B + C > A :=
by
  sorry

end triangle_inequality_l399_39914


namespace train_speed_proof_l399_39937

theorem train_speed_proof :
  (∀ (speed : ℝ), 
    let train_length := 120
    let cross_time := 16
    let total_distance := 240
    let relative_speed := total_distance / cross_time
    let individual_speed := relative_speed / 2
    let speed_kmh := individual_speed * 3.6
    (speed_kmh = 27) → speed = 27
  ) :=
by
  sorry

end train_speed_proof_l399_39937


namespace hemisphere_surface_area_l399_39926

theorem hemisphere_surface_area (base_area : ℝ) (r : ℝ) (total_surface_area : ℝ) 
(h1: base_area = 64 * Real.pi) 
(h2: r^2 = 64)
(h3: total_surface_area = base_area + 2 * Real.pi * r^2) : 
total_surface_area = 192 * Real.pi := 
sorry

end hemisphere_surface_area_l399_39926


namespace marty_combination_count_l399_39987

theorem marty_combination_count (num_colors : ℕ) (num_methods : ℕ) 
  (h1 : num_colors = 5) (h2 : num_methods = 4) : 
  num_colors * num_methods = 20 := by
  sorry

end marty_combination_count_l399_39987


namespace radar_placement_coverage_l399_39917

noncomputable def max_distance_radars (r : ℝ) (n : ℕ) : ℝ :=
  r / Real.sin (Real.pi / n)

noncomputable def coverage_ring_area (r : ℝ) (width : ℝ) (n : ℕ) : ℝ :=
  (1440 * Real.pi) / Real.tan (Real.pi / n)

theorem radar_placement_coverage :
  let r := 41
  let width := 18
  let n := 7
  max_distance_radars r n = 40 / Real.sin (Real.pi / 7) ∧
  coverage_ring_area r width n = (1440 * Real.pi) / Real.tan (Real.pi / 7) :=
by
  sorry

end radar_placement_coverage_l399_39917


namespace system1_solution_system2_solution_l399_39974

theorem system1_solution (x y : ℝ) (h₁ : y = 2 * x) (h₂ : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 := 
by sorry

theorem system2_solution (x y : ℝ) (h₁ : x - 3 * y = -2) (h₂ : 2 * x + 3 * y = 3) : x = (1 / 3) ∧ y = (7 / 9) := 
by sorry

end system1_solution_system2_solution_l399_39974


namespace sector_max_area_l399_39984

noncomputable def max_sector_area (R c : ℝ) : ℝ := 
  if h : R = c / 4 then c^2 / 16 else 0 -- This is just a skeleton, actual proof requires conditions
-- State the theorem that relates conditions to the maximum area.
theorem sector_max_area (R c α : ℝ) 
  (hc : c = 2 * R + R * α) : 
  (∃ R, R = c / 4) → max_sector_area R c = c^2 / 16 :=
by 
  sorry

end sector_max_area_l399_39984


namespace Ruth_sandwiches_l399_39995

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l399_39995


namespace retail_price_machine_l399_39961

theorem retail_price_machine (P : ℝ) :
  let wholesale_price := 99
  let discount_rate := 0.10
  let profit_rate := 0.20
  let selling_price := wholesale_price + (profit_rate * wholesale_price)
  0.90 * P = selling_price → P = 132 :=

by
  intro wholesale_price discount_rate profit_rate selling_price h
  sorry -- Proof will be handled here

end retail_price_machine_l399_39961


namespace min_value_l399_39986

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y = 1) : 
  ∃ m, m = (1 / x + 1 / y) ∧ m = 9 :=
by
  sorry

end min_value_l399_39986


namespace Granger_payment_correct_l399_39913

noncomputable def Granger_total_payment : ℝ :=
  let spam_per_can := 3.0
  let peanut_butter_per_jar := 5.0
  let bread_per_loaf := 2.0
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_dis := 0.1
  let peanut_butter_tax := 0.05
  let spam_cost := spam_quantity * spam_per_can
  let peanut_butter_cost := peanut_butter_quantity * peanut_butter_per_jar
  let bread_cost := bread_quantity * bread_per_loaf
  let spam_discount := spam_dis * spam_cost
  let peanut_butter_tax_amount := peanut_butter_tax * peanut_butter_cost
  let spam_final_cost := spam_cost - spam_discount
  let peanut_butter_final_cost := peanut_butter_cost + peanut_butter_tax_amount
  let total := spam_final_cost + peanut_butter_final_cost + bread_cost
  total

theorem Granger_payment_correct :
  Granger_total_payment = 56.15 :=
by
  sorry

end Granger_payment_correct_l399_39913


namespace max_value_fraction_l399_39916

theorem max_value_fraction {a b c : ℝ} (h1 : c = Real.sqrt (a^2 + b^2)) 
  (h2 : a > 0) (h3 : b > 0) (A : ℝ) (hA : A = 1 / 2 * a * b) :
  ∃ x : ℝ, x = (a + b + A) / c ∧ x ≤ (5 / 4) * Real.sqrt 2 :=
by
  sorry

end max_value_fraction_l399_39916


namespace hyperbola_eccentricity_l399_39988

theorem hyperbola_eccentricity (a b : ℝ) (h_asymptote : a = 3 * b) : 
    (a^2 + b^2) / a^2 = 10 / 9 := 
by
    sorry

end hyperbola_eccentricity_l399_39988


namespace sandwiches_per_person_l399_39915

open Nat

theorem sandwiches_per_person (total_sandwiches : ℕ) (total_people : ℕ) (h1 : total_sandwiches = 657) (h2 : total_people = 219) : 
(total_sandwiches / total_people) = 3 :=
by
  -- a proof would go here
  sorry

end sandwiches_per_person_l399_39915


namespace range_of_m_l399_39907

noncomputable def unique_zero_point (m : ℝ) : Prop :=
  ∀ x : ℝ, m * (1/4)^x - (1/2)^x + 1 = 0 → ∀ x' : ℝ, m * (1/4)^x' - (1/2)^x' + 1 = 0 → x = x'

theorem range_of_m (m : ℝ) : unique_zero_point m → (m ≤ 0 ∨ m = 1/4) :=
sorry

end range_of_m_l399_39907


namespace find_p_and_q_l399_39969

theorem find_p_and_q (p q : ℝ)
    (M : Set ℝ := {x | x^2 + p * x - 2 = 0})
    (N : Set ℝ := {x | x^2 - 2 * x + q = 0})
    (h : M ∪ N = {-1, 0, 2}) :
    p = -1 ∧ q = 0 :=
sorry

end find_p_and_q_l399_39969


namespace cos_neg_300_eq_half_l399_39975

theorem cos_neg_300_eq_half : Real.cos (-300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_neg_300_eq_half_l399_39975


namespace bottles_left_l399_39942

variable (initial_bottles : ℕ) (jason_bottles : ℕ) (harry_bottles : ℕ)

theorem bottles_left (h1 : initial_bottles = 35) (h2 : jason_bottles = 5) (h3 : harry_bottles = 6) :
    initial_bottles - (jason_bottles + harry_bottles) = 24 := by
  sorry

end bottles_left_l399_39942


namespace root_interval_l399_39927

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 2 * x - 1

theorem root_interval : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
by
  have h_decreasing : ∀ x y : ℝ, x < y → f x < f y :=
    sorry -- Proof that f is increasing on (-1, +∞)
  have h_f0 : f 0 = -1 := by
    sorry -- Calculation that f(0) = -1
  have h_f1 : f 1 = Real.log 2 + 1 := by
    sorry -- Calculation that f(1) = ln(2) + 1
  have h_exist_root : ∃ m : ℝ, f m = 0 ∧ 0 < m ∧ m < 1 :=
    by
      sorry -- Existence of a root in (0,1)
  exact h_exist_root

end root_interval_l399_39927


namespace regular_decagon_interior_angle_degree_measure_l399_39911

theorem regular_decagon_interior_angle_degree_measure :
  ∀ (n : ℕ), n = 10 → (2 * 180 / n : ℝ) = 144 :=
by
  sorry

end regular_decagon_interior_angle_degree_measure_l399_39911


namespace cos_150_eq_negative_cos_30_l399_39982

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l399_39982


namespace charges_needed_to_vacuum_house_l399_39966

-- Conditions definitions
def battery_last_minutes : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def number_of_bedrooms : ℕ := 3
def number_of_kitchens : ℕ := 1
def number_of_living_rooms : ℕ := 1

-- Question (proof problem statement)
theorem charges_needed_to_vacuum_house :
  ((number_of_bedrooms + number_of_kitchens + number_of_living_rooms) * vacuum_time_per_room) / battery_last_minutes = 2 :=
by
  sorry

end charges_needed_to_vacuum_house_l399_39966


namespace non_zero_real_solution_of_equation_l399_39994

noncomputable def equation_solution : Prop :=
  ∀ (x : ℝ), x ≠ 0 ∧ (7 * x) ^ 14 = (14 * x) ^ 7 → x = 2 / 7

theorem non_zero_real_solution_of_equation : equation_solution := sorry

end non_zero_real_solution_of_equation_l399_39994


namespace find_m_geq_9_l399_39906

-- Define the real numbers
variables {x m : ℝ}

-- Define the conditions
def p (x : ℝ) := x ≤ 2
def q (x m : ℝ) := x^2 - 2*x + 1 - m^2 ≤ 0

-- Main theorem statement based on the given problem
theorem find_m_geq_9 (m : ℝ) (hm : m > 0) :
  (¬ p x → ¬ q x m) → (p x → q x m) → m ≥ 9 :=
  sorry

end find_m_geq_9_l399_39906


namespace abc_le_one_eighth_l399_39939

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l399_39939


namespace candy_distribution_l399_39968

theorem candy_distribution (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 0) (h3 : 99 % n = 0) : n = 11 :=
sorry

end candy_distribution_l399_39968


namespace num_girls_went_to_spa_l399_39929

-- Define the condition that each girl has 20 nails
def nails_per_girl : ℕ := 20

-- Define the total number of nails polished
def total_nails_polished : ℕ := 40

-- Define the number of girls
def number_of_girls : ℕ := total_nails_polished / nails_per_girl

-- The theorem we want to prove
theorem num_girls_went_to_spa : number_of_girls = 2 :=
by
  unfold number_of_girls
  unfold total_nails_polished
  unfold nails_per_girl
  sorry

end num_girls_went_to_spa_l399_39929


namespace days_per_week_l399_39990

def threeChildren := 3
def schoolYearWeeks := 25
def totalJuiceBoxes := 375

theorem days_per_week (d : ℕ) :
  (threeChildren * d * schoolYearWeeks = totalJuiceBoxes) → d = 5 :=
by
  sorry

end days_per_week_l399_39990


namespace product_of_divisors_sum_l399_39941

theorem product_of_divisors_sum :
  ∃ (a b c : ℕ), (a ∣ 11^3) ∧ (b ∣ 11^3) ∧ (c ∣ 11^3) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a * b * c = 11^3) ∧ (a + b + c = 133) :=
sorry

end product_of_divisors_sum_l399_39941


namespace arithmetic_mean_l399_39962

variable (x b : ℝ)

theorem arithmetic_mean (hx : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
  sorry

end arithmetic_mean_l399_39962


namespace find_costs_l399_39934

theorem find_costs (a b : ℝ) (h1 : a - b = 3) (h2 : 3 * b - 2 * a = 3) : a = 12 ∧ b = 9 :=
sorry

end find_costs_l399_39934


namespace eq_solutions_count_l399_39976

theorem eq_solutions_count : 
  ∃! (n : ℕ), n = 126 ∧ (∀ x y : ℕ, 2*x + 3*y = 768 → x > 0 ∧ y > 0 → ∃ t : ℤ, x = 384 + 3*t ∧ y = -2*t ∧ -127 ≤ t ∧ t <= -1) := sorry

end eq_solutions_count_l399_39976


namespace expression_evaluation_l399_39955

theorem expression_evaluation :
  (4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2) ^ 0 = 1) :=
by
  -- Step by step calculations skipped
  sorry

end expression_evaluation_l399_39955


namespace part1_part2_l399_39998

variable (x α β : ℝ)

noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.sqrt 3 * (Real.cos x) ^ 2 + 2 * (Real.sin x) * (Real.cos x) - Real.sqrt 3

theorem part1 (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  -Real.sqrt 3 ≤ f x ∧ f x ≤ 2 := 
sorry

theorem part2 (hα : 0 < α ∧ α < Real.pi / 2) (hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : f (α / 2 - Real.pi / 6) = 8 / 5) 
(h2 : Real.cos (α + β) = -12 / 13) : 
  Real.sin β = 63 / 65 := 
sorry

end part1_part2_l399_39998


namespace add_fractions_l399_39967

theorem add_fractions : (7 / 12) + (3 / 8) = 23 / 24 := by
  sorry

end add_fractions_l399_39967


namespace total_cakes_needed_l399_39918

theorem total_cakes_needed (C : ℕ) (h : C / 4 - C / 12 = 10) : C = 60 := by
  sorry

end total_cakes_needed_l399_39918


namespace gcd_91_49_l399_39979

theorem gcd_91_49 : Nat.gcd 91 49 = 7 :=
by
  -- Using the Euclidean algorithm
  -- 91 = 49 * 1 + 42
  -- 49 = 42 * 1 + 7
  -- 42 = 7 * 6 + 0
  sorry

end gcd_91_49_l399_39979


namespace delta_zeta_finish_time_l399_39943

noncomputable def delta_epsilon_zeta_proof_problem (D E Z : ℝ) (k : ℝ) : Prop :=
  (1 / D + 1 / E + 1 / Z = 1 / (D - 4)) ∧
  (1 / D + 1 / E + 1 / Z = 1 / (E - 3.5)) ∧
  (1 / E + 1 / Z = 2 / E) → 
  k = 2

-- Now we prepare the theorem statement
theorem delta_zeta_finish_time (D E Z k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z = 1 / (D - 4))
                                (h2 : 1 / D + 1 / E + 1 / Z = 1 / (E - 3.5))
                                (h3 : 1 / E + 1 / Z = 2 / E) 
                                (h4 : E = 6) :
  k = 2 := 
sorry

end delta_zeta_finish_time_l399_39943


namespace sum_of_last_two_digits_l399_39960

-- Definitions based on given conditions
def six_power_twenty_five := 6^25
def fourteen_power_twenty_five := 14^25
def expression := six_power_twenty_five + fourteen_power_twenty_five
def modulo := 100

-- The statement we need to prove
theorem sum_of_last_two_digits : expression % modulo = 0 := by
  sorry

end sum_of_last_two_digits_l399_39960


namespace factor_expression_l399_39992

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) :=
sorry

end factor_expression_l399_39992


namespace original_length_of_ribbon_l399_39957

theorem original_length_of_ribbon (n : ℕ) (cm_per_piece : ℝ) (remaining_meters : ℝ) 
  (pieces_cm_to_m : cm_per_piece / 100 = 0.15) (remaining_ribbon : remaining_meters = 36) 
  (pieces_cut : n = 100) : n * (cm_per_piece / 100) + remaining_meters = 51 := 
by 
  sorry

end original_length_of_ribbon_l399_39957


namespace age_of_beckett_l399_39977

variables (B O S J : ℕ)

theorem age_of_beckett
  (h1 : B = O - 3)
  (h2 : S = O - 2)
  (h3 : J = 2 * S + 5)
  (h4 : B + O + S + J = 71) :
  B = 12 :=
by
  sorry

end age_of_beckett_l399_39977


namespace total_movies_purchased_l399_39931

theorem total_movies_purchased (x : ℕ) (h1 : 17 * x > 0) (h2 : 4 * x > 0) (h3 : 4 * x - 4 > 0) :
  (17 * x) / (4 * x - 4) = 9 / 2 → 17 * x + 4 * x = 378 :=
by 
  intro hab
  sorry

end total_movies_purchased_l399_39931


namespace number_of_pairs_l399_39950

theorem number_of_pairs :
  (∃ (pairs : Finset (ℕ × ℕ)), (∀ (pair : ℕ × ℕ), pair ∈ pairs → 1 ≤ pair.1 ∧ pair.1 ≤ 30 ∧ 3 ≤ pair.2 ∧ pair.2 ≤ 30 ∧ (pair.1 % pair.2 = 0) ∧ (pair.1 % (pair.2 - 2) = 0)) ∧ pairs.card = 22) := by
  sorry

end number_of_pairs_l399_39950


namespace find_x_such_that_custom_op_neg3_eq_one_l399_39970

def custom_op (x y : Int) : Int := x * y - 2 * (x + y)

theorem find_x_such_that_custom_op_neg3_eq_one :
  ∃ x : Int, custom_op x (-3) = 1 ∧ x = 1 :=
by
  use 1
  sorry

end find_x_such_that_custom_op_neg3_eq_one_l399_39970


namespace smallest_n_for_Qn_l399_39991

theorem smallest_n_for_Qn (n : ℕ) : 
  (∃ n : ℕ, 1 / (n * (2 * n + 1)) < 1 / 2023 ∧ ∀ m < n, 1 / (m * (2 * m + 1)) ≥ 1 / 2023) ↔ n = 32 := by
sorry

end smallest_n_for_Qn_l399_39991


namespace total_lunch_bill_l399_39923

theorem total_lunch_bill (cost_hotdog cost_salad : ℝ) (h1 : cost_hotdog = 5.36) (h2 : cost_salad = 5.10) : 
  cost_hotdog + cost_salad = 10.46 := 
by 
  sorry

end total_lunch_bill_l399_39923


namespace center_of_circle_is_2_1_l399_39944

-- Definition of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y - 5 = 0

-- Theorem stating the center of the circle
theorem center_of_circle_is_2_1 (x y : ℝ) (h : circle_eq x y) : (x, y) = (2, 1) := sorry

end center_of_circle_is_2_1_l399_39944


namespace andy_max_cookies_l399_39958

-- Definitions for the problem conditions
def total_cookies := 36
def bella_eats (andy_cookies : ℕ) := 2 * andy_cookies
def charlie_eats (andy_cookies : ℕ) := andy_cookies
def consumed_cookies (andy_cookies : ℕ) := andy_cookies + bella_eats andy_cookies + charlie_eats andy_cookies

-- The statement to prove
theorem andy_max_cookies : ∃ (a : ℕ), consumed_cookies a = total_cookies ∧ a = 9 :=
by
  sorry

end andy_max_cookies_l399_39958


namespace no_solutions_iff_a_positive_and_discriminant_non_positive_l399_39985

theorem no_solutions_iff_a_positive_and_discriminant_non_positive (a b c : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, ¬ (a * x^2 + b * x + c < 0)) ↔ (a > 0 ∧ (b^2 - 4 * a * c) ≤ 0) :=
  sorry

end no_solutions_iff_a_positive_and_discriminant_non_positive_l399_39985


namespace soft_drink_company_bottle_count_l399_39959

theorem soft_drink_company_bottle_count
  (B : ℕ)
  (initial_small_bottles : ℕ := 6000)
  (percent_sold_small : ℝ := 0.12)
  (percent_sold_big : ℝ := 0.14)
  (bottles_remaining_total : ℕ := 18180) :
  (initial_small_bottles * (1 - percent_sold_small) + B * (1 - percent_sold_big) = bottles_remaining_total) → B = 15000 :=
by
  sorry

end soft_drink_company_bottle_count_l399_39959


namespace inequality_x_y_z_l399_39965

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) + y * (1 - z) + z * (1 - x) < 1 := 
by sorry

end inequality_x_y_z_l399_39965


namespace infinite_geometric_series_sum_l399_39956

-- First term of the geometric series
def a : ℚ := 5/3

-- Common ratio of the geometric series
def r : ℚ := -1/4

-- The sum of the infinite geometric series
def S : ℚ := a / (1 - r)

-- Prove that the sum of the series is equal to 4/3
theorem infinite_geometric_series_sum : S = 4/3 := by
  sorry

end infinite_geometric_series_sum_l399_39956


namespace stockholm_to_uppsala_distance_l399_39971

-- Definition of conditions
def map_distance : ℝ := 45 -- in cm
def scale1 : ℝ := 10 -- first scale 1 cm : 10 km
def scale2 : ℝ := 5 -- second scale 1 cm : 5 km
def boundary : ℝ := 15 -- first 15 cm at scale 2

-- Calculation of the two parts
def part1_distance (boundary : ℝ) (scale2 : ℝ) := boundary * scale2
def remaining_distance (map_distance boundary : ℝ) := map_distance - boundary
def part2_distance (remaining_distance : ℝ) (scale1 : ℝ) := remaining_distance * scale1

-- Total distance
def total_distance (part1 part2: ℝ) := part1 + part2

theorem stockholm_to_uppsala_distance : 
  total_distance (part1_distance boundary scale2) 
                 (part2_distance (remaining_distance map_distance boundary) scale1) 
  = 375 := 
by
  -- Proof to be provided
  sorry

end stockholm_to_uppsala_distance_l399_39971


namespace cost_of_one_dozen_pens_l399_39904

variable (x : ℝ)

-- Conditions 1 and 2 as assumptions
def pen_cost := 5 * x
def pencil_cost := x

axiom cost_equation  : 3 * pen_cost + 5 * pencil_cost = 200
axiom cost_ratio     : pen_cost / pencil_cost = 5 / 1 -- ratio is given

-- Question and target statement
theorem cost_of_one_dozen_pens : 12 * pen_cost = 600 :=
by
  sorry

end cost_of_one_dozen_pens_l399_39904


namespace abs_neg_2035_l399_39947

theorem abs_neg_2035 : abs (-2035) = 2035 := 
by {
  sorry
}

end abs_neg_2035_l399_39947


namespace difference_between_c_and_a_l399_39909

variable (a b c : ℝ)

theorem difference_between_c_and_a (h1 : (a + b) / 2 = 30) (h2 : c - a = 60) : c - a = 60 :=
by
  exact h2

end difference_between_c_and_a_l399_39909


namespace value_of_3x_plus_5y_l399_39933

variable (x y : ℚ)

theorem value_of_3x_plus_5y
  (h1 : x + 4 * y = 5) 
  (h2 : 5 * x + 6 * y = 7) : 3 * x + 5 * y = 6 := 
sorry

end value_of_3x_plus_5y_l399_39933


namespace roots_opposite_k_eq_2_l399_39905

theorem roots_opposite_k_eq_2 (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 + x2 = 0 ∧ x1 * x2 = -1 ∧ x1 ≠ x2 ∧ x1*x1 + (k-2)*x1 - 1 = 0 ∧ x2*x2 + (k-2)*x2 - 1 = 0) → k = 2 :=
by
  sorry

end roots_opposite_k_eq_2_l399_39905


namespace notification_probability_l399_39900

theorem notification_probability
  (num_students : ℕ)
  (num_notified_Li : ℕ)
  (num_notified_Zhang : ℕ)
  (prob_Li : ℚ)
  (prob_Zhang : ℚ)
  (h1 : num_students = 10)
  (h2 : num_notified_Li = 4)
  (h3 : num_notified_Zhang = 4)
  (h4 : prob_Li = (4 : ℚ) / 10)
  (h5 : prob_Zhang = (4 : ℚ) / 10) :
  prob_Li + prob_Zhang - prob_Li * prob_Zhang = (16 : ℚ) / 25 := 
by 
  sorry

end notification_probability_l399_39900


namespace even_number_of_divisors_less_than_100_l399_39997

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l399_39997


namespace sarah_apples_calc_l399_39910

variable (brother_apples : ℕ)
variable (sarah_apples : ℕ)
variable (multiplier : ℕ)

theorem sarah_apples_calc
  (h1 : brother_apples = 9)
  (h2 : multiplier = 5)
  (h3 : sarah_apples = multiplier * brother_apples) : sarah_apples = 45 := by
  sorry

end sarah_apples_calc_l399_39910


namespace unique_solution_integer_equation_l399_39940

theorem unique_solution_integer_equation : 
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end unique_solution_integer_equation_l399_39940


namespace triangle_side_relationship_l399_39951

theorem triangle_side_relationship
  (a b c : ℝ)
  (habc : a < b + c)
  (ha_pos : a > 0) :
  a^2 < a * b + a * c :=
by
  sorry

end triangle_side_relationship_l399_39951


namespace max_y_diff_eq_0_l399_39922

-- Definitions for the given conditions
def eq1 (x : ℝ) : ℝ := 4 - 2 * x + x^2
def eq2 (x : ℝ) : ℝ := 2 + 2 * x + x^2

-- Statement of the proof problem
theorem max_y_diff_eq_0 : 
  (∀ x y, eq1 x = y ∧ eq2 x = y → y = (13 / 4)) →
  ∀ (x1 x2 : ℝ), (∃ y1 y2, eq1 x1 = y1 ∧ eq2 x1 = y1 ∧ eq1 x2 = y2 ∧ eq2 x2 = y2) → 
  (x1 = x2) → (y1 = y2) →
  0 = 0 := 
by
  sorry

end max_y_diff_eq_0_l399_39922


namespace houses_with_animals_l399_39946

theorem houses_with_animals (n A B C x y : ℕ) (h1 : n = 2017) (h2 : A = 1820) (h3 : B = 1651) (h4 : C = 1182) 
    (hx : x = 1182) (hy : y = 619) : x - y = 563 := 
by {
  sorry
}

end houses_with_animals_l399_39946
