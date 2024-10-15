import Mathlib

namespace NUMINAMATH_GPT_randy_feeds_per_day_l1707_170797

theorem randy_feeds_per_day
  (pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ)
  (h1 : pigs = 2) (h2 : total_feed_per_week = 140) (h3 : days_per_week = 7) :
  total_feed_per_week / pigs / days_per_week = 10 :=
by
  sorry

end NUMINAMATH_GPT_randy_feeds_per_day_l1707_170797


namespace NUMINAMATH_GPT_simplified_expression_correct_l1707_170712

noncomputable def simplified_expression : ℝ := 0.3 * 0.8 + 0.1 * 0.5

theorem simplified_expression_correct : simplified_expression = 0.29 := by 
  sorry

end NUMINAMATH_GPT_simplified_expression_correct_l1707_170712


namespace NUMINAMATH_GPT_b_divisible_by_8_l1707_170789

variable (b : ℕ) (n : ℕ)
variable (hb_even : b % 2 = 0) (hb_pos : b > 0) (hn_gt1 : n > 1)
variable (h_square : ∃ k : ℕ, k^2 = (b^n - 1) / (b - 1))

theorem b_divisible_by_8 : b % 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_b_divisible_by_8_l1707_170789


namespace NUMINAMATH_GPT_count_valid_n_l1707_170739

theorem count_valid_n :
  ∃ (count : ℕ), count = 9 ∧ 
  (∀ (n : ℕ), 0 < n ∧ n ≤ 2000 ∧ ∃ (k : ℕ), 21 * n = k * k ↔ count = 9) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_n_l1707_170739


namespace NUMINAMATH_GPT_Mrs_Hilt_remaining_money_l1707_170723

theorem Mrs_Hilt_remaining_money :
  let initial_amount : ℝ := 3.75
  let pencil_cost : ℝ := 1.15
  let eraser_cost : ℝ := 0.85
  let notebook_cost : ℝ := 2.25
  initial_amount - (pencil_cost + eraser_cost + notebook_cost) = -0.50 :=
by
  sorry

end NUMINAMATH_GPT_Mrs_Hilt_remaining_money_l1707_170723


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l1707_170713

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A = 45) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : 
  max A (max B C) = 75 :=
by
  -- Since no proof is needed, we mark it as sorry
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l1707_170713


namespace NUMINAMATH_GPT_g_at_three_l1707_170754

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_nonzero_at_zero : g 0 ≠ 0
axiom g_at_one : g 1 = 2

theorem g_at_three : g 3 = 8 := sorry

end NUMINAMATH_GPT_g_at_three_l1707_170754


namespace NUMINAMATH_GPT_ratio_of_speeds_l1707_170735

-- Conditions
def total_distance_Eddy : ℕ := 200 + 240 + 300
def total_distance_Freddy : ℕ := 180 + 420
def total_time_Eddy : ℕ := 5
def total_time_Freddy : ℕ := 6

-- Average speeds
def avg_speed_Eddy (d t : ℕ) : ℚ := d / t
def avg_speed_Freddy (d t : ℕ) : ℚ := d / t

-- Ratio of average speeds
def ratio_speeds (s1 s2 : ℚ) : ℚ := s1 / s2

theorem ratio_of_speeds : 
  ratio_speeds (avg_speed_Eddy total_distance_Eddy total_time_Eddy) 
               (avg_speed_Freddy total_distance_Freddy total_time_Freddy) 
  = 37 / 25 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_of_speeds_l1707_170735


namespace NUMINAMATH_GPT_team_a_faster_than_team_t_l1707_170753

-- Definitions for the conditions
def course_length : ℕ := 300
def team_t_speed : ℕ := 20
def team_t_time : ℕ := course_length / team_t_speed
def team_a_time : ℕ := team_t_time - 3
def team_a_speed : ℕ := course_length / team_a_time

-- Theorem to prove
theorem team_a_faster_than_team_t :
  team_a_speed - team_t_speed = 5 :=
by
  -- Define the necessary elements based on conditions
  let course_length := 300
  let team_t_speed := 20
  let team_t_time := course_length / team_t_speed -- 15 hours
  let team_a_time := team_t_time - 3 -- 12 hours
  let team_a_speed := course_length / team_a_time -- 25 mph
  
  -- Prove the statement
  have h : team_a_speed - team_t_speed = 5 := by sorry
  exact h

end NUMINAMATH_GPT_team_a_faster_than_team_t_l1707_170753


namespace NUMINAMATH_GPT_find_number_l1707_170779

variable (N : ℝ)

theorem find_number (h : (5 / 6) * N = (5 / 16) * N + 50) : N = 96 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1707_170779


namespace NUMINAMATH_GPT_dormitory_to_city_distance_l1707_170720

theorem dormitory_to_city_distance
  (D : ℝ)
  (h1 : (1/5) * D + (2/3) * D + 14 = D) :
  D = 105 :=
by
  sorry

end NUMINAMATH_GPT_dormitory_to_city_distance_l1707_170720


namespace NUMINAMATH_GPT_coffee_mix_price_l1707_170710

theorem coffee_mix_price (
  weight1 price1 weight2 price2 total_weight : ℝ)
  (h1 : weight1 = 9)
  (h2 : price1 = 2.15)
  (h3 : weight2 = 9)
  (h4 : price2 = 2.45)
  (h5 : total_weight = 18)
  :
  (weight1 * price1 + weight2 * price2) / total_weight = 2.30 :=
by
  sorry

end NUMINAMATH_GPT_coffee_mix_price_l1707_170710


namespace NUMINAMATH_GPT_insurance_coverage_is_80_percent_l1707_170793

-- Definitions and conditions
def MRI_cost : ℕ := 1200
def doctor_hourly_fee : ℕ := 300
def doctor_examination_time : ℕ := 30  -- in minutes
def seen_fee : ℕ := 150
def amount_paid_by_tim : ℕ := 300

-- The total cost calculation
def total_cost : ℕ := MRI_cost + (doctor_hourly_fee * doctor_examination_time / 60) + seen_fee

-- The amount covered by insurance
def amount_covered_by_insurance : ℕ := total_cost - amount_paid_by_tim

-- The percentage of coverage by insurance
def insurance_coverage_percentage : ℕ := (amount_covered_by_insurance * 100) / total_cost

theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := by
  sorry

end NUMINAMATH_GPT_insurance_coverage_is_80_percent_l1707_170793


namespace NUMINAMATH_GPT_find_a_range_l1707_170748

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * a * x^2 + 2 * x + 1
def f' (x a : ℝ) : ℝ := x^2 - a * x + 2

theorem find_a_range (a : ℝ) :
  (0 < x1) ∧ (x1 < 1) ∧ (1 < x2) ∧ (x2 < 3) ∧
  (f' 0 a > 0) ∧ (f' 1 a < 0) ∧ (f' 3 a > 0) →
  3 < a ∧ a < 11 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_range_l1707_170748


namespace NUMINAMATH_GPT_problem1_l1707_170709

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1707_170709


namespace NUMINAMATH_GPT_boys_without_pencils_l1707_170782

variable (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)

theorem boys_without_pencils
  (h1 : total_boys = 18)
  (h2 : students_with_pencils = 25)
  (h3 : girls_with_pencils = 15)
  (h4 : total_students = 30) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by
  sorry

end NUMINAMATH_GPT_boys_without_pencils_l1707_170782


namespace NUMINAMATH_GPT_equal_partitions_l1707_170728

def weights : List ℕ := List.range (81 + 1) |>.map (λ n => n * n)

theorem equal_partitions (h : weights.sum = 178605) :
  ∃ P1 P2 P3 : List ℕ, P1.sum = 59535 ∧ P2.sum = 59535 ∧ P3.sum = 59535 ∧ P1 ++ P2 ++ P3 = weights := sorry

end NUMINAMATH_GPT_equal_partitions_l1707_170728


namespace NUMINAMATH_GPT_number_of_years_borrowed_l1707_170760

theorem number_of_years_borrowed (n : ℕ)
  (H1 : ∃ (p : ℕ), 5000 = p ∧ 4 = 4 ∧ n * 200 = 150)
  (H2 : ∃ (q : ℕ), 5000 = q ∧ 7 = 7 ∧ n * 350 = 150)
  : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_years_borrowed_l1707_170760


namespace NUMINAMATH_GPT_happy_children_count_l1707_170750

theorem happy_children_count (total_children sad_children neither_children total_boys total_girls happy_boys sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : sad_children = 10)
  (h3 : neither_children = 20)
  (h4 : total_boys = 18)
  (h5 : total_girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4) :
  ∃ happy_children, happy_children = 30 :=
  sorry

end NUMINAMATH_GPT_happy_children_count_l1707_170750


namespace NUMINAMATH_GPT_no_permutation_exists_l1707_170747

open Function Set

theorem no_permutation_exists (f : ℕ → ℕ) (h : ∀ n m : ℕ, f n = f m ↔ n = m) :
  ¬ ∃ n : ℕ, (Finset.range n).image f = Finset.range n :=
by
  sorry

end NUMINAMATH_GPT_no_permutation_exists_l1707_170747


namespace NUMINAMATH_GPT_problem_a_l1707_170708

variable {S : Type*}
variables (a b : S)
variables [Inhabited S] -- Ensures S has at least one element
variables (op : S → S → S) -- Defines the binary operation

-- Condition: binary operation a * (b * a) = b holds for all a, b in S
axiom binary_condition : ∀ a b : S, op a (op b a) = b

-- Theorem to prove: (a * b) * a ≠ a
theorem problem_a : (op (op a b) a) ≠ a :=
sorry

end NUMINAMATH_GPT_problem_a_l1707_170708


namespace NUMINAMATH_GPT_probability_of_yellow_ball_is_correct_l1707_170798

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end NUMINAMATH_GPT_probability_of_yellow_ball_is_correct_l1707_170798


namespace NUMINAMATH_GPT_solve_for_n_l1707_170778

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 34) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1707_170778


namespace NUMINAMATH_GPT_arithmetic_seq_a7_l1707_170724

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (d : ℕ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 8)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 7 = 6 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_l1707_170724


namespace NUMINAMATH_GPT_age_of_youngest_boy_l1707_170799

theorem age_of_youngest_boy (average_age : ℕ) (age_proportion : ℕ → ℕ) 
  (h1 : average_age = 120) 
  (h2 : ∀ x, age_proportion x = 2 * x ∨ age_proportion x = 6 * x ∨ age_proportion x = 8 * x)
  (total_age : ℕ) 
  (h3 : total_age = 3 * average_age) :
  ∃ x, age_proportion x = 2 * x ∧ 2 * x * (3 * average_age / total_age) = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_age_of_youngest_boy_l1707_170799


namespace NUMINAMATH_GPT_triangle_sides_external_tangent_l1707_170704

theorem triangle_sides_external_tangent (R r : ℝ) (h : R > r) :
  ∃ (AB BC AC : ℝ),
    AB = 2 * Real.sqrt (R * r) ∧
    AC = 2 * r * Real.sqrt (R / (R + r)) ∧
    BC = 2 * R * Real.sqrt (r / (R + r)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_sides_external_tangent_l1707_170704


namespace NUMINAMATH_GPT_complement_intersection_l1707_170763

open Set

-- Definitions based on conditions given
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- The mathematical proof problem
theorem complement_intersection :
  (U \ A) ∩ B = {1, 3, 7} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1707_170763


namespace NUMINAMATH_GPT_wayne_needs_30_more_blocks_l1707_170773

def initial_blocks : ℕ := 9
def additional_blocks : ℕ := 6
def total_blocks : ℕ := initial_blocks + additional_blocks
def triple_total : ℕ := 3 * total_blocks

theorem wayne_needs_30_more_blocks :
  triple_total - total_blocks = 30 := by
  sorry

end NUMINAMATH_GPT_wayne_needs_30_more_blocks_l1707_170773


namespace NUMINAMATH_GPT_average_salary_feb_mar_apr_may_l1707_170729

theorem average_salary_feb_mar_apr_may 
  (average_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_months_1 : ℤ)
  (total_months_2 : ℤ)
  (total_sum_jan_apr : average_jan_feb_mar_apr * (total_months_1:ℝ) = 32000)
  (january_salary: salary_jan = 4700)
  (may_salary: salary_may = 6500)
  (total_months_1_eq: total_months_1 = 4)
  (total_months_2_eq: total_months_2 = 4):
  average_jan_feb_mar_apr * (total_months_1:ℝ) - salary_jan + salary_may/total_months_2 = 8450 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_feb_mar_apr_may_l1707_170729


namespace NUMINAMATH_GPT_emani_money_l1707_170703

def emani_has_30_more (E H : ℝ) : Prop := E = H + 30
def equal_share (E H : ℝ) : Prop := (E + H) / 2 = 135

theorem emani_money (E H : ℝ) (h1: emani_has_30_more E H) (h2: equal_share E H) : E = 150 :=
by
  sorry

end NUMINAMATH_GPT_emani_money_l1707_170703


namespace NUMINAMATH_GPT_carnival_earnings_l1707_170769

theorem carnival_earnings (days : ℕ) (total_earnings : ℕ) (h1 : days = 22) (h2 : total_earnings = 3168) : 
  (total_earnings / days) = 144 := 
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_carnival_earnings_l1707_170769


namespace NUMINAMATH_GPT_inequality_ab_ab2_a_l1707_170702

theorem inequality_ab_ab2_a (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end NUMINAMATH_GPT_inequality_ab_ab2_a_l1707_170702


namespace NUMINAMATH_GPT_secret_sharing_problem_l1707_170733

theorem secret_sharing_problem : 
  ∃ n : ℕ, (3280 = (3^(n + 1) - 1) / 2) ∧ (n = 7) :=
by
  use 7
  sorry

end NUMINAMATH_GPT_secret_sharing_problem_l1707_170733


namespace NUMINAMATH_GPT_cos_alpha_half_l1707_170726

theorem cos_alpha_half (α : ℝ) (h : Real.cos (Real.pi + α) = -1/2) : Real.cos α = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_half_l1707_170726


namespace NUMINAMATH_GPT_blue_balls_in_box_l1707_170749

theorem blue_balls_in_box (total_balls : ℕ) (p_two_blue : ℚ) (b : ℕ) 
  (h1 : total_balls = 12) (h2 : p_two_blue = 1/22) 
  (h3 : (↑b / 12) * (↑(b-1) / 11) = p_two_blue) : b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_blue_balls_in_box_l1707_170749


namespace NUMINAMATH_GPT_Jake_weight_loss_l1707_170766

variables (J K x : ℕ)

theorem Jake_weight_loss : 
  J = 198 ∧ J + K = 293 ∧ J - x = 2 * K → x = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_Jake_weight_loss_l1707_170766


namespace NUMINAMATH_GPT_total_bananas_in_collection_l1707_170764

theorem total_bananas_in_collection (groups_of_bananas : ℕ) (bananas_per_group : ℕ) 
    (h1 : groups_of_bananas = 7) (h2 : bananas_per_group = 29) :
    groups_of_bananas * bananas_per_group = 203 := by
  sorry

end NUMINAMATH_GPT_total_bananas_in_collection_l1707_170764


namespace NUMINAMATH_GPT_number_of_negative_x_l1707_170792

theorem number_of_negative_x (n : ℤ) (hn : 1 ≤ n ∧ n * n < 200) : 
  ∃ m ≥ 1, m = 14 := sorry

end NUMINAMATH_GPT_number_of_negative_x_l1707_170792


namespace NUMINAMATH_GPT_smallest_multiple_l1707_170732

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_l1707_170732


namespace NUMINAMATH_GPT_student_total_marks_l1707_170741

variables {M P C : ℕ}

theorem student_total_marks
  (h1 : C = P + 20)
  (h2 : (M + C) / 2 = 35) :
  M + P = 50 :=
sorry

end NUMINAMATH_GPT_student_total_marks_l1707_170741


namespace NUMINAMATH_GPT_min_value_frac_sum_l1707_170743

theorem min_value_frac_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 4) : 
  (4 / a^2 + 1 / b^2) ≥ 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_sum_l1707_170743


namespace NUMINAMATH_GPT_range_of_m_perimeter_of_isosceles_triangle_l1707_170751

-- Define the variables for the lengths of the sides and the range of m
variables (AB BC AC : ℝ) (m : ℝ)

-- Conditions given in the problem
def triangle_conditions (AB BC : ℝ) (AC : ℝ) (m : ℝ) : Prop :=
  AB = 17 ∧ BC = 8 ∧ AC = 2 * m - 1

-- Proof that the range for m is between 5 and 13
theorem range_of_m (AB BC : ℝ) (m : ℝ) (h : triangle_conditions AB BC (2 * m - 1) m) : 
  5 < m ∧ m < 13 :=
by
  sorry

-- Proof that the perimeter is 42 when triangle is isosceles with given conditions
theorem perimeter_of_isosceles_triangle (AB BC AC : ℝ) (h : triangle_conditions AB BC AC 0) : 
  (AB = AC ∨ BC = AC) → (2 * AB + BC = 42) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_perimeter_of_isosceles_triangle_l1707_170751


namespace NUMINAMATH_GPT_rectangle_perimeter_greater_than_16_l1707_170788

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_greater_than_16_l1707_170788


namespace NUMINAMATH_GPT_arlene_average_pace_l1707_170745

theorem arlene_average_pace :
  ∃ pace : ℝ, pace = 24 / (6 - 0.75) ∧ pace = 4.57 := 
by
  sorry

end NUMINAMATH_GPT_arlene_average_pace_l1707_170745


namespace NUMINAMATH_GPT_geometric_sequence_sum_inequality_l1707_170796

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geom : ∀ k, a (k + 1) = a k * q)
  (h_pos : ∀ k ≤ 7, a k > 0)
  (h_q_ne_one : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_inequality_l1707_170796


namespace NUMINAMATH_GPT_minimize_area_of_quadrilateral_l1707_170759

noncomputable def minimize_quad_area (AB BC CD DA A1 B1 C1 D1 : ℝ) (k : ℝ) : Prop :=
  -- Conditions
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ k > 0 ∧
  A1 = k * AB ∧ B1 = k * BC ∧ C1 = k * CD ∧ D1 = k * DA →
  -- Conclusion
  k = 1 / 2

-- Statement without proof
theorem minimize_area_of_quadrilateral (AB BC CD DA : ℝ) : ∃ k : ℝ, minimize_quad_area AB BC CD DA (k * AB) (k * BC) (k * CD) (k * DA) k :=
sorry

end NUMINAMATH_GPT_minimize_area_of_quadrilateral_l1707_170759


namespace NUMINAMATH_GPT_stadium_length_in_feet_l1707_170780

theorem stadium_length_in_feet (length_in_yards : ℕ) (conversion_factor : ℕ) (h1 : length_in_yards = 62) (h2 : conversion_factor = 3) : length_in_yards * conversion_factor = 186 :=
by
  sorry

end NUMINAMATH_GPT_stadium_length_in_feet_l1707_170780


namespace NUMINAMATH_GPT_equivalent_discount_l1707_170756

variable (P d1 d2 d : ℝ)

-- Given conditions:
def original_price : ℝ := 50
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.10
def equivalent_single_discount_rate : ℝ := 0.325

-- Final conclusion:
theorem equivalent_discount :
  let final_price_after_first_discount := (original_price * (1 - first_discount_rate))
  let final_price_after_second_discount := (final_price_after_first_discount * (1 - second_discount_rate))
  final_price_after_second_discount = (original_price * (1 - equivalent_single_discount_rate)) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_discount_l1707_170756


namespace NUMINAMATH_GPT_number_of_people_in_team_l1707_170714

def total_distance : ℕ := 150
def distance_per_member : ℕ := 30

theorem number_of_people_in_team :
  (total_distance / distance_per_member) = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_people_in_team_l1707_170714


namespace NUMINAMATH_GPT_distinct_units_digits_of_perfect_cube_l1707_170762

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end NUMINAMATH_GPT_distinct_units_digits_of_perfect_cube_l1707_170762


namespace NUMINAMATH_GPT_gcd_is_12_l1707_170775

noncomputable def gcd_problem (b : ℤ) : Prop :=
  b % 2027 = 0 → Int.gcd (b^2 + 7*b + 18) (b + 6) = 12

-- Now, let's state the theorem
theorem gcd_is_12 (b : ℤ) : gcd_problem b :=
  sorry

end NUMINAMATH_GPT_gcd_is_12_l1707_170775


namespace NUMINAMATH_GPT_holly_pills_per_week_l1707_170705

theorem holly_pills_per_week 
  (insulin_pills_per_day : ℕ)
  (blood_pressure_pills_per_day : ℕ)
  (anticonvulsants_per_day : ℕ)
  (H1 : insulin_pills_per_day = 2)
  (H2 : blood_pressure_pills_per_day = 3)
  (H3 : anticonvulsants_per_day = 2 * blood_pressure_pills_per_day) :
  (insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsants_per_day) * 7 = 77 := 
by
  sorry

end NUMINAMATH_GPT_holly_pills_per_week_l1707_170705


namespace NUMINAMATH_GPT_percentage_of_men_l1707_170731

theorem percentage_of_men (M : ℝ) 
  (h1 : 0 < M ∧ M < 1) 
  (h2 : 0.2 * M + 0.4 * (1 - M) = 0.3) : M = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_l1707_170731


namespace NUMINAMATH_GPT_difference_is_correct_l1707_170752

-- Define the given constants and conditions
def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def monthly_payment : ℕ := 65
def number_of_monthly_payments : ℕ := 24

-- Define the derived quantities based on the given conditions
def total_monthly_payments : ℕ := monthly_payment * number_of_monthly_payments
def total_amount_paid : ℕ := down_payment + total_monthly_payments
def difference : ℕ := total_amount_paid - purchase_price

-- The statement to be proven
theorem difference_is_correct : difference = 260 := by
  sorry

end NUMINAMATH_GPT_difference_is_correct_l1707_170752


namespace NUMINAMATH_GPT_composite_polynomial_l1707_170721

-- Definition that checks whether a number is composite
def is_composite (a : ℕ) : Prop := ∃ (b c : ℕ), b > 1 ∧ c > 1 ∧ a = b * c

-- Problem translated into a Lean 4 statement
theorem composite_polynomial (n : ℕ) (h : n ≥ 2) :
  is_composite (n ^ (5 * n - 1) + n ^ (5 * n - 2) + n ^ (5 * n - 3) + n + 1) :=
sorry

end NUMINAMATH_GPT_composite_polynomial_l1707_170721


namespace NUMINAMATH_GPT_James_total_tabs_l1707_170777

theorem James_total_tabs (browsers windows tabs additional_tabs : ℕ) 
  (h_browsers : browsers = 4)
  (h_windows : windows = 5)
  (h_tabs : tabs = 12)
  (h_additional_tabs : additional_tabs = 3) : 
  browsers * (windows * (tabs + additional_tabs)) = 300 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_James_total_tabs_l1707_170777


namespace NUMINAMATH_GPT_range_of_x_l1707_170738

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (h₁ : abs (a + b) + abs (a - b) ≥ abs a * f x) :
  0 ≤ x ∧ x ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_x_l1707_170738


namespace NUMINAMATH_GPT_mutually_exclusive_events_l1707_170770

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end NUMINAMATH_GPT_mutually_exclusive_events_l1707_170770


namespace NUMINAMATH_GPT_sum_of_two_digit_factors_l1707_170715

theorem sum_of_two_digit_factors (a b : ℕ) (h : a * b = 5681) (h1 : 10 ≤ a) (h2 : a < 100) (h3 : 10 ≤ b) (h4 : b < 100) : a + b = 154 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_digit_factors_l1707_170715


namespace NUMINAMATH_GPT_contrapositive_proof_l1707_170758

theorem contrapositive_proof (a : ℝ) (h : a ≤ 2 → a^2 ≤ 4) : a > 2 → a^2 > 4 :=
by
  intros ha
  sorry

end NUMINAMATH_GPT_contrapositive_proof_l1707_170758


namespace NUMINAMATH_GPT_graph_passes_through_point_l1707_170706

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 1) + 3

theorem graph_passes_through_point (a : ℝ) : f a 1 = 4 := by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1707_170706


namespace NUMINAMATH_GPT_new_babysitter_rate_l1707_170725

theorem new_babysitter_rate (x : ℝ) :
  (6 * 16) - 18 = 6 * x + 3 * 2 → x = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_new_babysitter_rate_l1707_170725


namespace NUMINAMATH_GPT_solvable_system_of_inequalities_l1707_170700

theorem solvable_system_of_inequalities (n : ℕ) : 
  (∃ x : ℝ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k < x ^ k ∧ x ^ k < k + 1)) ∧ (1 < x ∧ x < 2)) ↔ (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end NUMINAMATH_GPT_solvable_system_of_inequalities_l1707_170700


namespace NUMINAMATH_GPT_product_of_values_l1707_170755

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end NUMINAMATH_GPT_product_of_values_l1707_170755


namespace NUMINAMATH_GPT_housewife_oil_expense_l1707_170757

theorem housewife_oil_expense:
  ∃ M P R: ℝ, (R = 30) ∧ (0.8 * P = R) ∧ ((M / R) - (M / P) = 10) ∧ (M = 1500) :=
by
  sorry

end NUMINAMATH_GPT_housewife_oil_expense_l1707_170757


namespace NUMINAMATH_GPT_bivalid_positions_count_l1707_170718

/-- 
A position of the hands of a (12-hour, analog) clock is called valid if it occurs in the course of a day.
A position of the hands is called bivalid if it is valid and, in addition, the position formed by interchanging the hour and minute hands is valid.
-/
def is_valid (h m : ℕ) : Prop := 
  0 ≤ h ∧ h < 360 ∧ 
  0 ≤ m ∧ m < 360

def satisfies_conditions (h m : Int) (a b : Int) : Prop :=
  m = 12 * h - 360 * a ∧ h = 12 * m - 360 * b

def is_bivalid (h m : ℕ) : Prop := 
  ∃ (a b : Int), satisfies_conditions (h : Int) (m : Int) a b ∧ satisfies_conditions (m : Int) (h : Int) b a

theorem bivalid_positions_count : 
  ∃ (n : ℕ), n = 143 ∧ 
  ∀ (h m : ℕ), is_bivalid h m → n = 143 :=
sorry

end NUMINAMATH_GPT_bivalid_positions_count_l1707_170718


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l1707_170711

-- Given conditions
def water_speed : ℝ := 4
def swim_time_against_current : ℝ := 2
def swim_distance_against_current : ℝ := 8

-- What we are trying to prove
theorem swimming_speed_in_still_water (v : ℝ) 
    (h1 : swim_distance_against_current = 8) 
    (h2 : swim_time_against_current = 2)
    (h3 : water_speed = 4) :
    v - water_speed = swim_distance_against_current / swim_time_against_current → v = 8 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l1707_170711


namespace NUMINAMATH_GPT_intersection_range_l1707_170791

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem intersection_range :
  {m : ℝ | ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m} = Set.Ioo (-3 : ℝ) 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_range_l1707_170791


namespace NUMINAMATH_GPT_quadratic_root_condition_l1707_170772

theorem quadratic_root_condition (a : ℝ) :
  (4 * Real.sqrt 2) = 3 * Real.sqrt (3 - 2 * a) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1707_170772


namespace NUMINAMATH_GPT_class_C_payment_l1707_170783

-- Definitions based on conditions
variables (x y z : ℤ) (total_C : ℤ)

-- Given conditions
def condition_A : Prop := 3 * x + 7 * y + z = 14
def condition_B : Prop := 4 * x + 10 * y + z = 16
def condition_C : Prop := 3 * (x + y + z) = total_C

-- The theorem to prove
theorem class_C_payment (hA : condition_A x y z) (hB : condition_B x y z) : total_C = 30 :=
sorry

end NUMINAMATH_GPT_class_C_payment_l1707_170783


namespace NUMINAMATH_GPT_rows_colored_red_l1707_170765

theorem rows_colored_red (total_rows total_squares_per_row blue_rows green_squares red_squares_per_row red_rows : ℕ)
  (h_total_squares : total_rows * total_squares_per_row = 150)
  (h_blue_squares : blue_rows * total_squares_per_row = 60)
  (h_green_squares : green_squares = 66)
  (h_red_squares : 150 - 60 - 66 = 24)
  (h_red_rows : 24 / red_squares_per_row = 4) :
  red_rows = 4 := 
by sorry

end NUMINAMATH_GPT_rows_colored_red_l1707_170765


namespace NUMINAMATH_GPT_train_speed_l1707_170781

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) (total_distance : ℝ) 
    (speed_mps : ℝ) (speed_kmph : ℝ) 
    (h1 : train_length = 360) 
    (h2 : bridge_length = 140) 
    (h3 : time = 34.61538461538461) 
    (h4 : total_distance = train_length + bridge_length) 
    (h5 : speed_mps = total_distance / time) 
    (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 52 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_l1707_170781


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_locus_of_midpoint_M_l1707_170716

-- Define the conditions of the ellipse
def isEllipse (a b c : ℝ) : Prop :=
  a = 2 ∧ c = Real.sqrt 3 ∧ b = Real.sqrt (a^2 - c^2)

-- Define the equation of the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the locus of the midpoint M
def locus_midpoint (x y : ℝ) : Prop :=
  x^2 / 4 + 4 * y^2 = 1

theorem standard_equation_of_ellipse :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, ellipse_equation x y) :=
sorry

theorem locus_of_midpoint_M :
  ∃ a b c : ℝ, isEllipse a b c ∧ (∀ x y : ℝ, locus_midpoint x y) :=
sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_locus_of_midpoint_M_l1707_170716


namespace NUMINAMATH_GPT_fibonacci_arith_sequence_a_eq_665_l1707_170722

theorem fibonacci_arith_sequence_a_eq_665 (F : ℕ → ℕ) (a b c : ℕ) :
  (F 1 = 1) →
  (F 2 = 1) →
  (∀ n, n ≥ 3 → F n = F (n - 1) + F (n - 2)) →
  (a + b + c = 2000) →
  (F a < F b ∧ F b < F c ∧ F b - F a = F c - F b) →
  a = 665 :=
by
  sorry

end NUMINAMATH_GPT_fibonacci_arith_sequence_a_eq_665_l1707_170722


namespace NUMINAMATH_GPT_symmetrical_point_l1707_170727

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetrical_point (M : Point) (hM : M = {x := 3, y := -4}) : reflect_x_axis M = {x := 3, y := 4} :=
  by
  sorry

end NUMINAMATH_GPT_symmetrical_point_l1707_170727


namespace NUMINAMATH_GPT_remainder_when_2013_divided_by_85_l1707_170795

theorem remainder_when_2013_divided_by_85 : 2013 % 85 = 58 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_2013_divided_by_85_l1707_170795


namespace NUMINAMATH_GPT_range_of_m_l1707_170736

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : x1 > x2) (h2 : y1 > y2) (h3 : y1 = (m-2)*x1) (h4 : y2 = (m-2)*x2) : m > 2 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1707_170736


namespace NUMINAMATH_GPT_rate_of_second_batch_of_wheat_l1707_170744

theorem rate_of_second_batch_of_wheat (total_cost1 cost_per_kg1 weight1 weight2 total_weight total_cost selling_price_per_kg profit_rate cost_per_kg2 : ℝ)
  (H1 : total_cost1 = cost_per_kg1 * weight1)
  (H2 : total_weight = weight1 + weight2)
  (H3 : total_cost = total_cost1 + cost_per_kg2 * weight2)
  (H4 : selling_price_per_kg = (1 + profit_rate) * total_cost / total_weight)
  (H5 : profit_rate = 0.30)
  (H6 : cost_per_kg1 = 11.50)
  (H7 : weight1 = 30)
  (H8 : weight2 = 20)
  (H9 : selling_price_per_kg = 16.38) :
  cost_per_kg2 = 14.25 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_second_batch_of_wheat_l1707_170744


namespace NUMINAMATH_GPT_tank_capacity_l1707_170768

variable (C : ℕ) (t : ℕ)
variable (hC_nonzero : C > 0)
variable (ht_nonzero : t > 0)
variable (h_rate_pipe_A : t = C / 5)
variable (h_rate_pipe_B : t = C / 8)
variable (h_rate_inlet : t = 4 * 60)
variable (h_combined_time : t = 5 + 3)

theorem tank_capacity (C : ℕ) (h1 : C / 5 + C / 8 - 4 * 60 = 8) : C = 1200 := 
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1707_170768


namespace NUMINAMATH_GPT_ratio_of_sphere_radii_l1707_170776

noncomputable def ratio_of_radius (V_large : ℝ) (percentage : ℝ) : ℝ :=
  let V_small := (percentage / 100) * V_large
  let ratio := (V_small / V_large) ^ (1/3)
  ratio

theorem ratio_of_sphere_radii : 
  ratio_of_radius (450 * Real.pi) 27.04 = 0.646 := 
  by
  sorry

end NUMINAMATH_GPT_ratio_of_sphere_radii_l1707_170776


namespace NUMINAMATH_GPT_find_smaller_number_l1707_170717

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l1707_170717


namespace NUMINAMATH_GPT_algebraic_expression_value_l1707_170774

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -1) : 6 + 2 * x - 4 * y = 4 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1707_170774


namespace NUMINAMATH_GPT_gasoline_price_increase_percent_l1707_170701

theorem gasoline_price_increase_percent {P Q : ℝ}
  (h₁ : P > 0)
  (h₂: Q > 0)
  (x : ℝ)
  (condition : P * Q * 1.08 = P * (1 + x/100) * Q * 0.90) :
  x = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_gasoline_price_increase_percent_l1707_170701


namespace NUMINAMATH_GPT_initial_ratio_is_four_five_l1707_170794

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_ratio_is_four_five_l1707_170794


namespace NUMINAMATH_GPT_segment_parametrization_pqrs_l1707_170746

theorem segment_parametrization_pqrs :
  ∃ (p q r s : ℤ), 
    q = 1 ∧ 
    s = -3 ∧ 
    p + q = 6 ∧ 
    r + s = 4 ∧ 
    p^2 + q^2 + r^2 + s^2 = 84 :=
by
  use 5, 1, 7, -3
  sorry

end NUMINAMATH_GPT_segment_parametrization_pqrs_l1707_170746


namespace NUMINAMATH_GPT_cost_price_of_each_clock_l1707_170719

theorem cost_price_of_each_clock
  (C : ℝ)
  (h1 : 40 * C * 1.1 + 50 * C * 1.2 - 90 * C * 1.15 = 40) :
  C = 80 :=
sorry

end NUMINAMATH_GPT_cost_price_of_each_clock_l1707_170719


namespace NUMINAMATH_GPT_math_problem_l1707_170767

theorem math_problem 
  (a b : ℂ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) : 
  a^(2*n + 1) + b^(2*n + 1) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l1707_170767


namespace NUMINAMATH_GPT_find_g_l1707_170761

theorem find_g (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x+1) = 3 - 2 * x) (h2 : ∀ x : ℝ, f (g x) = 6 * x - 3) : 
  ∀ x : ℝ, g x = 4 - 3 * x := 
by
  sorry

end NUMINAMATH_GPT_find_g_l1707_170761


namespace NUMINAMATH_GPT_polar_coordinates_standard_representation_l1707_170784

theorem polar_coordinates_standard_representation :
  ∀ (r θ : ℝ), (r, θ) = (-4, 5 * Real.pi / 6) → (∃ (r' θ' : ℝ), r' > 0 ∧ (r', θ') = (4, 11 * Real.pi / 6))
:= by
  sorry

end NUMINAMATH_GPT_polar_coordinates_standard_representation_l1707_170784


namespace NUMINAMATH_GPT_range_g_a_values_l1707_170707

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem range_g : ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 :=
sorry

theorem a_values (a : ℝ) : (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) :=
sorry

end NUMINAMATH_GPT_range_g_a_values_l1707_170707


namespace NUMINAMATH_GPT_goods_train_length_l1707_170740

-- Conditions
def train1_speed := 60 -- kmph
def train2_speed := 52 -- kmph
def passing_time := 9 -- seconds

-- Conversion factor from kmph to meters per second
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (train1_speed + train2_speed)

-- Final theorem statement
theorem goods_train_length :
  relative_speed_mps * passing_time = 280 :=
sorry

end NUMINAMATH_GPT_goods_train_length_l1707_170740


namespace NUMINAMATH_GPT_sum_a5_a8_eq_six_l1707_170730

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∀ {m n : ℕ}, a (m + 1) / a m = a (n + 1) / a n

theorem sum_a5_a8_eq_six (h_seq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36) :
  a 5 + a 8 = 6 := 
sorry

end NUMINAMATH_GPT_sum_a5_a8_eq_six_l1707_170730


namespace NUMINAMATH_GPT_negation_of_proposition_l1707_170737

theorem negation_of_proposition :
  ¬(∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ m : ℤ, n = 2 * m)) ↔ ∃ n : ℤ, (∃ k : ℤ, n = 2 * k) ∧ ¬(∃ m : ℤ, n = 2 * m) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1707_170737


namespace NUMINAMATH_GPT_yoki_cans_collected_l1707_170785

theorem yoki_cans_collected (total_cans LaDonna_cans Prikya_cans Avi_cans : ℕ) (half_Avi_cans Yoki_cans : ℕ) 
    (h1 : total_cans = 85) 
    (h2 : LaDonna_cans = 25) 
    (h3 : Prikya_cans = 2 * LaDonna_cans - 3) 
    (h4 : Avi_cans = 8) 
    (h5 : half_Avi_cans = Avi_cans / 2) 
    (h6 : total_cans = LaDonna_cans + Prikya_cans + half_Avi_cans + Yoki_cans) :
    Yoki_cans = 9 := sorry

end NUMINAMATH_GPT_yoki_cans_collected_l1707_170785


namespace NUMINAMATH_GPT_sum_of_xyz_l1707_170786

theorem sum_of_xyz (x y z : ℝ) (h : (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0) : x + y + z = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_xyz_l1707_170786


namespace NUMINAMATH_GPT_find_x_value_l1707_170787

-- Define vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the condition that a + b is parallel to 2a - b
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1 - b.1) = k * (a.1 + b.1) ∧ (2 * a.2 - b.2) = k * (a.2 + b.2)

-- Problem statement: Prove that x = -4
theorem find_x_value : ∀ (x : ℝ),
  parallel_vectors vector_a (vector_b x) → x = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1707_170787


namespace NUMINAMATH_GPT_stratified_sampling_l1707_170734

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) 
  (h1 : total_teachers = 150)
  (h2 : senior_teachers = 45)
  (h3 : intermediate_teachers = 90)
  (h4 : junior_teachers = 15)
  (h5 : sample_size = 30)
  (h6 : x + y + z = sample_size)
  (h7 : x * 10 = sample_size / 5)
  (h8 : y * 10 = (2 * sample_size) / 5)
  (h9 : z * 10 = sample_size / 15) :
  (x, y, z) = (9, 18, 3) := sorry

end NUMINAMATH_GPT_stratified_sampling_l1707_170734


namespace NUMINAMATH_GPT_percent_of_a_is_b_l1707_170771

variable (a b c : ℝ)
variable (h1 : c = 0.20 * a) (h2 : c = 0.10 * b)

theorem percent_of_a_is_b : b = 2 * a :=
by sorry

end NUMINAMATH_GPT_percent_of_a_is_b_l1707_170771


namespace NUMINAMATH_GPT_sum_of_squared_projections_l1707_170742

theorem sum_of_squared_projections (a l m n : ℝ) (l_proj m_proj n_proj : ℝ)
  (h : l_proj = a * Real.cos θ)
  (h1 : m_proj = a * Real.cos (Real.pi / 3 - θ))
  (h2 : n_proj = a * Real.cos (Real.pi / 3 + θ)) :
  l_proj ^ 2 + m_proj ^ 2 + n_proj ^ 2 = 3 / 2 * a ^ 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_squared_projections_l1707_170742


namespace NUMINAMATH_GPT_compare_points_on_line_l1707_170790

theorem compare_points_on_line (m n : ℝ) 
  (hA : ∃ (x : ℝ), x = -3 ∧ m = -2 * x + 1) 
  (hB : ∃ (x : ℝ), x = 2 ∧ n = -2 * x + 1) : 
  m > n :=
by sorry

end NUMINAMATH_GPT_compare_points_on_line_l1707_170790
