import Mathlib

namespace NUMINAMATH_GPT_eq_is_quadratic_iff_m_zero_l1027_102715

theorem eq_is_quadratic_iff_m_zero (m : ℝ) : (|m| + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 := by
  sorry

end NUMINAMATH_GPT_eq_is_quadratic_iff_m_zero_l1027_102715


namespace NUMINAMATH_GPT_original_number_of_people_is_fifteen_l1027_102707

/-!
The average age of all the people who gathered at a family celebration was equal to the number of attendees. 
Aunt Beta, who was 29 years old, soon excused herself and left. 
Even after Aunt Beta left, the average age of all the remaining attendees was still equal to their number.
Prove that the original number of people at the celebration is 15.
-/

theorem original_number_of_people_is_fifteen
  (n : ℕ)
  (s : ℕ)
  (h1 : s = n^2)
  (h2 : s - 29 = (n - 1)^2):
  n = 15 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_people_is_fifteen_l1027_102707


namespace NUMINAMATH_GPT_average_income_correct_l1027_102756

-- Define the incomes for each day
def income_day_1 : ℕ := 300
def income_day_2 : ℕ := 150
def income_day_3 : ℕ := 750
def income_day_4 : ℕ := 400
def income_day_5 : ℕ := 500

-- Define the number of days
def number_of_days : ℕ := 5

-- Define the total income
def total_income : ℕ := income_day_1 + income_day_2 + income_day_3 + income_day_4 + income_day_5

-- Define the average income
def average_income : ℕ := total_income / number_of_days

-- State that the average income is 420
theorem average_income_correct :
  average_income = 420 := by
  sorry

end NUMINAMATH_GPT_average_income_correct_l1027_102756


namespace NUMINAMATH_GPT_smallest_y_value_l1027_102731

theorem smallest_y_value : ∃ y : ℝ, 2 * y ^ 2 + 7 * y + 3 = 5 ∧ (∀ y' : ℝ, 2 * y' ^ 2 + 7 * y' + 3 = 5 → y ≤ y') := sorry

end NUMINAMATH_GPT_smallest_y_value_l1027_102731


namespace NUMINAMATH_GPT_platinum_earrings_percentage_l1027_102718

theorem platinum_earrings_percentage
  (rings_percentage ornaments_percentage : ℝ)
  (rings_percentage_eq : rings_percentage = 0.30)
  (earrings_percentage_eq : ornaments_percentage - rings_percentage = 0.70)
  (platinum_earrings_percentage : ℝ)
  (platinum_earrings_percentage_eq : platinum_earrings_percentage = 0.70) :
  ornaments_percentage * platinum_earrings_percentage = 0.49 :=
by 
  have earrings_percentage := 0.70
  have ornaments_percentage := 0.70
  sorry

end NUMINAMATH_GPT_platinum_earrings_percentage_l1027_102718


namespace NUMINAMATH_GPT_range_of_m_l1027_102777

-- Given definitions and conditions
def sequence_a (n : ℕ) : ℕ := if n = 1 then 2 else n * 2^n

def vec_a : ℕ × ℤ := (2, -1)

def vec_b (n : ℕ) : ℕ × ℤ := (sequence_a n + 2^n, sequence_a (n + 1))

def orthogonal (v1 v2 : ℕ × ℤ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Translate the proof problem
theorem range_of_m (n : ℕ) (m : ℝ) (h1 : orthogonal vec_a (vec_b n))
  (h2 : ∀ n : ℕ, n > 0 → (sequence_a n) / (n * (n + 1)^2) > (m^2 - 3 * m) / 9) :
  -1 < m ∧ m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1027_102777


namespace NUMINAMATH_GPT_horner_eval_f_at_5_eval_f_at_5_l1027_102763

def f (x: ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem horner_eval_f_at_5 :
  f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
  sorry

theorem eval_f_at_5 : f 5 = 2015 := by 
  have h : f 5 = ((((5 - 2) * 5 + 1) * 5 + 1) * 5 - 1) * 5 - 5 := by
    apply horner_eval_f_at_5
  rw [h]
  norm_num

end NUMINAMATH_GPT_horner_eval_f_at_5_eval_f_at_5_l1027_102763


namespace NUMINAMATH_GPT_find_value_of_2_times_x_minus_y_squared_minus_3_l1027_102798

-- Define the conditions as noncomputable variables
variables (x y : ℝ)

-- State the main theorem
theorem find_value_of_2_times_x_minus_y_squared_minus_3 :
  (x^2 - x*y = 12) →
  (y^2 - y*x = 15) →
  2 * (x - y)^2 - 3 = 51 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_value_of_2_times_x_minus_y_squared_minus_3_l1027_102798


namespace NUMINAMATH_GPT_average_income_l1027_102751

theorem average_income :
  let income_day1 := 300
  let income_day2 := 150
  let income_day3 := 750
  let income_day4 := 200
  let income_day5 := 600
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / 5 = 400 := by
  sorry

end NUMINAMATH_GPT_average_income_l1027_102751


namespace NUMINAMATH_GPT_Lyka_saves_for_8_weeks_l1027_102775

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end NUMINAMATH_GPT_Lyka_saves_for_8_weeks_l1027_102775


namespace NUMINAMATH_GPT_ball_more_than_bat_l1027_102720

theorem ball_more_than_bat :
  ∃ x y : ℕ, (2 * x + 3 * y = 1300) ∧ (3 * x + 2 * y = 1200) ∧ (y - x = 100) :=
by
  sorry

end NUMINAMATH_GPT_ball_more_than_bat_l1027_102720


namespace NUMINAMATH_GPT_other_train_speed_l1027_102761

noncomputable def speed_of_other_train (l1 l2 v1 : ℕ) (t : ℝ) : ℝ := 
  let relative_speed := (l1 + l2) / 1000 / (t / 3600)
  relative_speed - v1

theorem other_train_speed :
  speed_of_other_train 210 260 40 16.918646508279338 = 60 := 
by
  sorry

end NUMINAMATH_GPT_other_train_speed_l1027_102761


namespace NUMINAMATH_GPT_tan_20_plus_4_sin_20_eq_sqrt_3_l1027_102757

theorem tan_20_plus_4_sin_20_eq_sqrt_3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_20_plus_4_sin_20_eq_sqrt_3_l1027_102757


namespace NUMINAMATH_GPT_rotor_permutations_l1027_102706

-- Define the factorial function for convenience
def fact : Nat → Nat
| 0     => 1
| (n + 1) => (n + 1) * fact n

-- The main statement to prove
theorem rotor_permutations : (fact 5) / ((fact 2) * (fact 2)) = 30 := by
  sorry

end NUMINAMATH_GPT_rotor_permutations_l1027_102706


namespace NUMINAMATH_GPT_parallel_lines_condition_l1027_102790

-- We define the conditions as Lean definitions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0
def parallel_condition (a : ℝ) : Prop := (a ≠ 0) ∧ (a ≠ 1) ∧ (a ≠ -1) ∧ (a * (a^2 - 1) ≠ 6)

-- Mathematically equivalent Lean 4 statement
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y → line2 a x y → (line1 a x y ↔ line2 a x y)) ↔ (a = -1) :=
by 
  -- The full proof would be written here
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1027_102790


namespace NUMINAMATH_GPT_train_speed_in_kmh_l1027_102758

def length_of_train : ℝ := 156
def length_of_bridge : ℝ := 219.03
def time_to_cross_bridge : ℝ := 30
def speed_of_train_kmh : ℝ := 45.0036

theorem train_speed_in_kmh :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = speed_of_train_kmh :=
by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l1027_102758


namespace NUMINAMATH_GPT_geometric_progression_identity_l1027_102795

theorem geometric_progression_identity 
  (a b c d : ℝ) 
  (h1 : c^2 = b * d) 
  (h2 : b^2 = a * c) 
  (h3 : a * d = b * c) : 
  (a - c)^2 + (b - c)^2 + (b - d)^2 = (a - d)^2 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_progression_identity_l1027_102795


namespace NUMINAMATH_GPT_find_value_of_a_l1027_102735

theorem find_value_of_a (x a : ℝ) (h : 2 * x - a + 5 = 0) (h_x : x = -2) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1027_102735


namespace NUMINAMATH_GPT_savings_duration_before_investment_l1027_102750

---- Definitions based on conditions ----
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def stock_price_per_share : ℕ := 50
def shares_bought : ℕ := 25

---- Derived conditions based on definitions ----
def total_spent_on_stocks := shares_bought * stock_price_per_share
def total_savings_before_investment := 2 * total_spent_on_stocks
def monthly_savings_wife := weekly_savings_wife * 4
def total_monthly_savings := monthly_savings_wife + monthly_savings_husband

---- The theorem statement ----
theorem savings_duration_before_investment :
  total_savings_before_investment / total_monthly_savings = 4 :=
sorry

end NUMINAMATH_GPT_savings_duration_before_investment_l1027_102750


namespace NUMINAMATH_GPT_shortest_distance_from_vertex_to_path_l1027_102727

theorem shortest_distance_from_vertex_to_path
  (r l : ℝ)
  (hr : r = 1)
  (hl : l = 3) :
  ∃ d : ℝ, d = 1.5 :=
by
  -- Given a cone with a base radius of 1 cm and a slant height of 3 cm
  -- We need to prove the shortest distance from the vertex to the path P back to P is 1.5 cm
  sorry

end NUMINAMATH_GPT_shortest_distance_from_vertex_to_path_l1027_102727


namespace NUMINAMATH_GPT_transformed_quadratic_l1027_102759

theorem transformed_quadratic (a b c n x : ℝ) (h : a * x^2 + b * x + c = 0) :
  a * x^2 + n * b * x + n^2 * c = 0 :=
sorry

end NUMINAMATH_GPT_transformed_quadratic_l1027_102759


namespace NUMINAMATH_GPT_product_of_slopes_constant_l1027_102770

noncomputable def ellipse (x y : ℝ) := x^2 / 8 + y^2 / 4 = 1

theorem product_of_slopes_constant (a b : ℝ) (h_a_gt_b : a > b) (h_a_b_pos : 0 < a ∧ 0 < b)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 2) / 2) (P : ℝ × ℝ) (h_point_on_ellipse : (P.1, P.2) = (2, Real.sqrt 2)) :
  (∃ C : ℝ → ℝ → Prop, C = ellipse) ∧ (∃ k : ℝ, -k * 1/2 = -1 / 2) := sorry

end NUMINAMATH_GPT_product_of_slopes_constant_l1027_102770


namespace NUMINAMATH_GPT_WidgetsPerHour_l1027_102782

theorem WidgetsPerHour 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (widgets_per_week : ℕ) 
  (H1 : hours_per_day = 8)
  (H2 : days_per_week = 5)
  (H3 : widgets_per_week = 800) : 
  widgets_per_week / (hours_per_day * days_per_week) = 20 := 
sorry

end NUMINAMATH_GPT_WidgetsPerHour_l1027_102782


namespace NUMINAMATH_GPT_intersection_A_B_l1027_102765

/-- Definition of set A -/
def A : Set ℕ := {1, 2, 3, 4}

/-- Definition of set B -/
def B : Set ℕ := {x | x > 2}

/-- The theorem to prove the intersection of sets A and B -/
theorem intersection_A_B : A ∩ B = {3, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1027_102765


namespace NUMINAMATH_GPT_total_amount_l1027_102797

theorem total_amount (x y z : ℝ) 
  (hx : y = 0.45 * x) 
  (hz : z = 0.50 * x) 
  (hy_share : y = 63) : 
  x + y + z = 273 :=
by 
  sorry

end NUMINAMATH_GPT_total_amount_l1027_102797


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l1027_102728

-- Problem (1)
theorem inequality_one {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : 2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

-- Problem (2)
theorem inequality_two {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : (a ^ 2 / b + b ^ 2 / c + c ^ 2 / a) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l1027_102728


namespace NUMINAMATH_GPT_valid_inequalities_l1027_102709

theorem valid_inequalities (a b c : ℝ) (h : 0 < c) 
  (h1 : b > c - b)
  (h2 : c > a)
  (h3 : c > b - a) :
  a < c / 2 ∧ b < a + c / 2 :=
by
  sorry

end NUMINAMATH_GPT_valid_inequalities_l1027_102709


namespace NUMINAMATH_GPT_remainder_when_divided_by_198_l1027_102799

-- Define the conditions as Hypotheses
variables (x : ℤ)

-- Hypotheses stating the given conditions
def cond1 : Prop := 2 + x ≡ 9 [ZMOD 8]
def cond2 : Prop := 3 + x ≡ 4 [ZMOD 27]
def cond3 : Prop := 11 + x ≡ 49 [ZMOD 1331]

-- Final statement to prove
theorem remainder_when_divided_by_198 (h1 : cond1 x) (h2 : cond2 x) (h3 : cond3 x) : x ≡ 1 [ZMOD 198] := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_198_l1027_102799


namespace NUMINAMATH_GPT_point_not_in_first_quadrant_l1027_102729

theorem point_not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) : ¬ (m > 0 ∧ n > 0) :=
sorry

end NUMINAMATH_GPT_point_not_in_first_quadrant_l1027_102729


namespace NUMINAMATH_GPT_theorem_1_valid_theorem_6_valid_l1027_102787

theorem theorem_1_valid (a b : ℤ) (h1 : a % 7 = 0) (h2 : b % 7 = 0) : (a + b) % 7 = 0 :=
by sorry

theorem theorem_6_valid (a b : ℤ) (h : (a + b) % 7 ≠ 0) : a % 7 ≠ 0 ∨ b % 7 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_theorem_1_valid_theorem_6_valid_l1027_102787


namespace NUMINAMATH_GPT_nolan_monthly_savings_l1027_102736

theorem nolan_monthly_savings (m k : ℕ) (H : 12 * m = 36 * k) : m = 3 * k := 
by sorry

end NUMINAMATH_GPT_nolan_monthly_savings_l1027_102736


namespace NUMINAMATH_GPT_simplest_square_root_among_choices_l1027_102730

variable {x : ℝ}

def is_simplest_square_root (n : ℝ) : Prop :=
  ∀ m, (m^2 = n) → (m = n)

theorem simplest_square_root_among_choices :
  is_simplest_square_root 7 ∧ ∀ n, n = 24 ∨ n = 1/3 ∨ n = 0.2 → ¬ is_simplest_square_root n :=
by
  sorry

end NUMINAMATH_GPT_simplest_square_root_among_choices_l1027_102730


namespace NUMINAMATH_GPT_marbles_remaining_correct_l1027_102705

-- Define the number of marbles Chris has
def marbles_chris : ℕ := 12

-- Define the number of marbles Ryan has
def marbles_ryan : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := marbles_chris + marbles_ryan

-- Define the number of marbles each person takes away from the pile
def marbles_taken_each : ℕ := total_marbles / 4

-- Define the total number of marbles taken away
def total_marbles_taken : ℕ := 2 * marbles_taken_each

-- Define the number of marbles remaining in the pile
def marbles_remaining : ℕ := total_marbles - total_marbles_taken

theorem marbles_remaining_correct : marbles_remaining = 20 := by
  sorry

end NUMINAMATH_GPT_marbles_remaining_correct_l1027_102705


namespace NUMINAMATH_GPT_complex_point_quadrant_l1027_102719

def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_point_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  inFourthQuadrant z :=
by
  sorry

end NUMINAMATH_GPT_complex_point_quadrant_l1027_102719


namespace NUMINAMATH_GPT_polynomial_remainder_l1027_102773

theorem polynomial_remainder (x : ℂ) :
  (x ^ 2030 + 1) % (x ^ 6 - x ^ 4 + x ^ 2 - 1) = x ^ 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1027_102773


namespace NUMINAMATH_GPT_stratified_sampling_sophomores_l1027_102732

theorem stratified_sampling_sophomores
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (total_selected : ℕ)
  (H_freshmen : freshmen = 550) (H_sophomores : sophomores = 700) (H_juniors : juniors = 750) (H_total_selected : total_selected = 100) :
  sophomores * total_selected / (freshmen + sophomores + juniors) = 35 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_sophomores_l1027_102732


namespace NUMINAMATH_GPT_smallest_integer_CC6_DD8_l1027_102702

def is_valid_digit_in_base (n : ℕ) (b : ℕ) : Prop :=
  n < b

theorem smallest_integer_CC6_DD8 : 
  ∃ C D : ℕ, is_valid_digit_in_base C 6 ∧ is_valid_digit_in_base D 8 ∧ 7 * C = 9 * D ∧ 7 * C = 63 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_CC6_DD8_l1027_102702


namespace NUMINAMATH_GPT_Q_finishes_in_6_hours_l1027_102726

def Q_time_to_finish_job (T_Q : ℝ) : Prop :=
  let P_rate := 1 / 3
  let Q_rate := 1 / T_Q
  let work_together_2hr := 2 * (P_rate + Q_rate)
  let P_alone_work_40min := (2 / 3) * P_rate
  work_together_2hr + P_alone_work_40min = 1

theorem Q_finishes_in_6_hours : Q_time_to_finish_job 6 :=
  sorry -- Proof skipped

end NUMINAMATH_GPT_Q_finishes_in_6_hours_l1027_102726


namespace NUMINAMATH_GPT_evaluate_expression_correct_l1027_102708

def evaluate_expression : ℚ :=
  let a := 17
  let b := 19
  let c := 23
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b + 1/c) + b * (1/c + 1/a) + c * (1/a + 1/b)
  numerator / denominator

theorem evaluate_expression_correct : evaluate_expression = 59 := 
by {
  -- proof skipped
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_correct_l1027_102708


namespace NUMINAMATH_GPT_cube_red_faces_one_third_l1027_102771

theorem cube_red_faces_one_third (n : ℕ) (h : 6 * n^3 ≠ 0) : 
  (2 * n^2) / (6 * n^3) = 1 / 3 → n = 1 :=
by sorry

end NUMINAMATH_GPT_cube_red_faces_one_third_l1027_102771


namespace NUMINAMATH_GPT_coordinates_of_P_tangent_line_equation_l1027_102760

-- Define point P and center of the circle
def point_P : ℝ × ℝ := (-2, 1)
def center_C : ℝ × ℝ := (-1, 0)

-- Define the circle equation (x + 1)^2 + y^2 = 2
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the tangent line at point P
def tangent_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Prove the coordinates of point P are (-2, 1) given the conditions
theorem coordinates_of_P (n : ℝ) (h1 : n > 0) (h2 : circle_equation (-2) n) :
  point_P = (-2, 1) :=
by
  -- Proof steps would go here
  sorry

-- Prove the equation of the tangent line to the circle C passing through point P is x - y + 3 = 0
theorem tangent_line_equation :
  tangent_line (-2) 1 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_coordinates_of_P_tangent_line_equation_l1027_102760


namespace NUMINAMATH_GPT_pq_false_implies_m_range_l1027_102738

def p : Prop := ∀ x : ℝ, abs x + x ≥ 0

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem pq_false_implies_m_range (m : ℝ) :
  (¬ (p ∧ q m)) → -2 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_pq_false_implies_m_range_l1027_102738


namespace NUMINAMATH_GPT_october_birth_percentage_l1027_102701

theorem october_birth_percentage 
  (jan feb mar apr may jun jul aug sep oct nov dec total : ℕ) 
  (h_total : total = 100)
  (h_jan : jan = 2) (h_feb : feb = 4) (h_mar : mar = 8) (h_apr : apr = 5) 
  (h_may : may = 4) (h_jun : jun = 9) (h_jul : jul = 7) (h_aug : aug = 12) 
  (h_sep : sep = 8) (h_oct : oct = 6) (h_nov : nov = 5) (h_dec : dec = 4) : 
  (oct : ℕ) * 100 / total = 6 := 
by
  sorry

end NUMINAMATH_GPT_october_birth_percentage_l1027_102701


namespace NUMINAMATH_GPT_trajectory_of_center_of_P_l1027_102739

-- Define circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the conditions for the moving circle P
def externally_tangent (x y r : ℝ) : Prop := (x + 1)^2 + y^2 = (1 + r)^2
def internally_tangent (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = (5 - r)^2

-- The statement we need to prove
theorem trajectory_of_center_of_P : ∃ (x y : ℝ), 
  (externally_tangent x y r) ∧ (internally_tangent x y r) →
  (x^2 / 9 + y^2 / 8 = 1) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_trajectory_of_center_of_P_l1027_102739


namespace NUMINAMATH_GPT_simplify_exponent_product_l1027_102779

theorem simplify_exponent_product :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 :=
by
  sorry

end NUMINAMATH_GPT_simplify_exponent_product_l1027_102779


namespace NUMINAMATH_GPT_sum_of_three_integers_l1027_102741

theorem sum_of_three_integers (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) (h_product : a * b * c = 5^3) : a + b + c = 31 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_l1027_102741


namespace NUMINAMATH_GPT_day_50_of_year_N_minus_1_l1027_102749

-- Definitions for the problem conditions
def day_of_week (n : ℕ) : ℕ := n % 7

-- Given that the 250th day of year N is a Friday
axiom day_250_of_year_N_is_friday : day_of_week 250 = 5

-- Given that the 150th day of year N+1 is a Friday
axiom day_150_of_year_N_plus_1_is_friday : day_of_week 150 = 5

-- Calculate the day of the week for the 50th day of year N-1
theorem day_50_of_year_N_minus_1 :
  day_of_week 50 = 4 :=
  sorry

end NUMINAMATH_GPT_day_50_of_year_N_minus_1_l1027_102749


namespace NUMINAMATH_GPT_pete_miles_walked_l1027_102772

noncomputable def steps_from_first_pedometer (flips1 : ℕ) (final_reading1 : ℕ) : ℕ :=
  flips1 * 100000 + final_reading1 

noncomputable def steps_from_second_pedometer (flips2 : ℕ) (final_reading2 : ℕ) : ℕ :=
  flips2 * 400000 + final_reading2 * 4

noncomputable def total_steps (flips1 flips2 final_reading1 final_reading2 : ℕ) : ℕ :=
  steps_from_first_pedometer flips1 final_reading1 + steps_from_second_pedometer flips2 final_reading2

noncomputable def miles_walked (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

theorem pete_miles_walked
  (flips1 flips2 final_reading1 final_reading2 steps_per_mile : ℕ)
  (h_flips1 : flips1 = 50)
  (h_final_reading1 : final_reading1 = 25000)
  (h_flips2 : flips2 = 15)
  (h_final_reading2 : final_reading2 = 30000)
  (h_steps_per_mile : steps_per_mile = 1500) :
  miles_walked (total_steps flips1 flips2 final_reading1 final_reading2) steps_per_mile = 7430 :=
by sorry

end NUMINAMATH_GPT_pete_miles_walked_l1027_102772


namespace NUMINAMATH_GPT_sally_investment_l1027_102764

theorem sally_investment (m : ℝ) (hmf : 0 ≤ m) 
  (total_investment : m + 7 * m = 200000) : 
  7 * m = 175000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sally_investment_l1027_102764


namespace NUMINAMATH_GPT_gabby_l1027_102752

-- Define variables and conditions
variables (watermelons peaches plums total_fruit : ℕ)
variables (h_watermelons : watermelons = 1)
variables (h_peaches : peaches = watermelons + 12)
variables (h_plums : plums = 3 * peaches)
variables (h_total_fruit : total_fruit = watermelons + peaches + plums)

-- The theorem we aim to prove
theorem gabby's_fruit_count (h_watermelons : watermelons = 1)
                           (h_peaches : peaches = watermelons + 12)
                           (h_plums : plums = 3 * peaches)
                           (h_total_fruit : total_fruit = watermelons + peaches + plums) :
  total_fruit = 53 := by
sorry

end NUMINAMATH_GPT_gabby_l1027_102752


namespace NUMINAMATH_GPT_quadratic_root_sum_product_l1027_102788

theorem quadratic_root_sum_product (m n : ℝ)
  (h1 : m + n = 4)
  (h2 : m * n = -1) :
  m + n - m * n = 5 :=
sorry

end NUMINAMATH_GPT_quadratic_root_sum_product_l1027_102788


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1027_102745

theorem geometric_sequence_sum (a : ℕ → ℝ) (S₄ : ℝ) (S₈ : ℝ) (r : ℝ) 
    (h1 : r = 2) 
    (h2 : S₄ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3)
    (h3 : S₄ = 1) 
    (h4 : S₈ = a 0 + a 0 * r + a 0 * r^2 + a 0 * r^3 + a 0 * r^4 + a 0 * r^5 + a 0 * r^6 + a 0 * r^7) :
    S₈ = 17 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1027_102745


namespace NUMINAMATH_GPT_number_that_multiplies_b_l1027_102780

theorem number_that_multiplies_b (a b x : ℝ) (h0 : 4 * a = x * b) (h1 : a * b ≠ 0) (h2 : (a / 5) / (b / 4) = 1) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_that_multiplies_b_l1027_102780


namespace NUMINAMATH_GPT_find_C_l1027_102734

noncomputable def h (C D : ℝ) (x : ℝ) : ℝ := 2 * C * x - 3 * D ^ 2
def k (D : ℝ) (x : ℝ) := D * x

theorem find_C (C D : ℝ) (h_eq : h C D (k D 2) = 0) (hD : D ≠ 0) : C = 3 * D / 4 :=
by
  unfold h k at h_eq
  sorry

end NUMINAMATH_GPT_find_C_l1027_102734


namespace NUMINAMATH_GPT_value_of_a_l1027_102742

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1027_102742


namespace NUMINAMATH_GPT_distance_to_airport_l1027_102733

theorem distance_to_airport
  (t : ℝ)
  (d : ℝ)
  (h1 : 45 * (t + 1) + 20 = d)
  (h2 : d - 65 = 65 * (t - 1))
  : d = 390 := by
  sorry

end NUMINAMATH_GPT_distance_to_airport_l1027_102733


namespace NUMINAMATH_GPT_tan_A_mul_tan_B_lt_one_l1027_102792

theorem tan_A_mul_tan_B_lt_one (A B C : ℝ) (hC: C > 90) (hABC : A + B + C = 180) :
    Real.tan A * Real.tan B < 1 :=
sorry

end NUMINAMATH_GPT_tan_A_mul_tan_B_lt_one_l1027_102792


namespace NUMINAMATH_GPT_candy_eating_l1027_102746

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end NUMINAMATH_GPT_candy_eating_l1027_102746


namespace NUMINAMATH_GPT_tan_identity_l1027_102786

theorem tan_identity
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 7)
  (h2 : Real.tan (β - Real.pi / 4) = -1 / 3)
  : Real.tan (α + Real.pi / 4) = 8 / 9 := by
  sorry

end NUMINAMATH_GPT_tan_identity_l1027_102786


namespace NUMINAMATH_GPT_least_positive_int_to_next_multiple_l1027_102776

theorem least_positive_int_to_next_multiple (x : ℕ) (n : ℕ) (h : x = 365 ∧ n > 0) 
  (hm : (x + n) % 5 = 0) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_int_to_next_multiple_l1027_102776


namespace NUMINAMATH_GPT_seventeen_number_selection_l1027_102794

theorem seventeen_number_selection : ∃ (n : ℕ), (∀ s : Finset ℕ, (s ⊆ Finset.range 17) → (Finset.card s = n) → ∃ x y : ℕ, (x ∈ s) ∧ (y ∈ s) ∧ (x ≠ y) ∧ (x = 3 * y ∨ y = 3 * x)) ∧ (n = 13) :=
by
  sorry

end NUMINAMATH_GPT_seventeen_number_selection_l1027_102794


namespace NUMINAMATH_GPT_range_of_x_l1027_102743

-- Define the function h(a).
def h (a : ℝ) : ℝ := a^2 + 2 * a + 3

-- Define the main theorem
theorem range_of_x (a : ℝ) (x : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) : 
  x^2 + 4 * x - 2 ≤ h a → -5 ≤ x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_x_l1027_102743


namespace NUMINAMATH_GPT_abs_difference_of_opposite_signs_l1027_102754

theorem abs_difference_of_opposite_signs (a b : ℝ) (ha : |a| = 4) (hb : |b| = 2) (hdiff : a * b < 0) : |a - b| = 6 := 
sorry

end NUMINAMATH_GPT_abs_difference_of_opposite_signs_l1027_102754


namespace NUMINAMATH_GPT_min_cubes_l1027_102711

-- Define the conditions as properties
structure FigureViews :=
  (front_view : ℕ)
  (side_view : ℕ)
  (top_view : ℕ)
  (adjacency_requirement : Bool)

-- Define the given views
def given_views : FigureViews := {
  front_view := 3,  -- as described: 2 cubes at bottom + 1 on top
  side_view := 3,   -- same as front view
  top_view := 3,    -- L-shape consists of 3 cubes
  adjacency_requirement := true
}

-- The theorem to state that the minimum number of cubes is 3
theorem min_cubes (views : FigureViews) : views.front_view = 3 ∧ views.side_view = 3 ∧ views.top_view = 3 ∧ views.adjacency_requirement = true → ∃ n, n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_cubes_l1027_102711


namespace NUMINAMATH_GPT_minimum_value_of_function_l1027_102766

theorem minimum_value_of_function :
  ∃ (y : ℝ), y > 0 ∧
  (∀ z : ℝ, z > 0 → y^2 + 10 * y + 100 / y^3 ≤ z^2 + 10 * z + 100 / z^3) ∧ 
  y^2 + 10 * y + 100 / y^3 = 50^(2/3) + 10 * 50^(1/3) + 2 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1027_102766


namespace NUMINAMATH_GPT_hikers_rate_l1027_102767

noncomputable def rate_up (rate_down := 15) : ℝ := 5

theorem hikers_rate :
  let R := rate_up
  let distance_down := rate_down
  let time := 2
  let rate_down := 1.5 * R
  distance_down = rate_down * time → R = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hikers_rate_l1027_102767


namespace NUMINAMATH_GPT_quadratic_polynomial_value_bound_l1027_102747

theorem quadratic_polynomial_value_bound (a b : ℝ) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, |(x^2 + a * x + b)| ≥ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_value_bound_l1027_102747


namespace NUMINAMATH_GPT_river_length_l1027_102785

theorem river_length (x : ℝ) (h1 : 3 * x + x = 80) : x = 20 :=
sorry

end NUMINAMATH_GPT_river_length_l1027_102785


namespace NUMINAMATH_GPT_expand_product_correct_l1027_102703

noncomputable def expand_product (x : ℝ) : ℝ :=
  3 * (x + 4) * (x + 5)

theorem expand_product_correct (x : ℝ) :
  expand_product x = 3 * x^2 + 27 * x + 60 :=
by
  unfold expand_product
  sorry

end NUMINAMATH_GPT_expand_product_correct_l1027_102703


namespace NUMINAMATH_GPT_sum_of_terms_l1027_102714

theorem sum_of_terms (a d : ℕ) (h1 : a + d < a + 2 * d)
  (h2 : (a + d) * (a + 20) = (a + 2 * d) ^ 2)
  (h3 : a + 20 - a = 20) :
  a + (a + d) + (a + 2 * d) + (a + 20) = 46 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_terms_l1027_102714


namespace NUMINAMATH_GPT_tan_x_plus_pi_over_4_l1027_102712

theorem tan_x_plus_pi_over_4 (x : ℝ) (hx : Real.tan x = 2) : Real.tan (x + Real.pi / 4) = -3 :=
by
  sorry

end NUMINAMATH_GPT_tan_x_plus_pi_over_4_l1027_102712


namespace NUMINAMATH_GPT_roots_product_of_polynomials_l1027_102769

theorem roots_product_of_polynomials :
  ∃ (b c : ℤ), (∀ r : ℂ, r ^ 2 - 2 * r - 1 = 0 → r ^ 5 - b * r - c = 0) ∧ b * c = 348 :=
by 
  sorry

end NUMINAMATH_GPT_roots_product_of_polynomials_l1027_102769


namespace NUMINAMATH_GPT_hardcover_books_count_l1027_102748

theorem hardcover_books_count (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 30 * h + 15 * p = 270) : h = 6 :=
by
  sorry

end NUMINAMATH_GPT_hardcover_books_count_l1027_102748


namespace NUMINAMATH_GPT_find_length_of_segment_l1027_102740

noncomputable def radius : ℝ := 4
noncomputable def volume_cylinder (L : ℝ) : ℝ := 16 * Real.pi * L
noncomputable def volume_hemispheres : ℝ := 2 * (128 / 3) * Real.pi
noncomputable def total_volume (L : ℝ) : ℝ := volume_cylinder L + volume_hemispheres

theorem find_length_of_segment (L : ℝ) (h : total_volume L = 544 * Real.pi) : 
  L = 86 / 3 :=
by sorry

end NUMINAMATH_GPT_find_length_of_segment_l1027_102740


namespace NUMINAMATH_GPT_determine_pairs_l1027_102774

theorem determine_pairs (p q : ℕ) (h : (p + 1)^(p - 1) + (p - 1)^(p + 1) = q^q) : (p = 1 ∧ q = 1) ∨ (p = 2 ∧ q = 2) :=
by
  sorry

end NUMINAMATH_GPT_determine_pairs_l1027_102774


namespace NUMINAMATH_GPT_fixed_point_quadratic_l1027_102768

theorem fixed_point_quadratic : 
  (∀ m : ℝ, 3 * a ^ 2 - m * a + 2 * m + 1 = b) → (a = 2 ∧ b = 13) := 
by sorry

end NUMINAMATH_GPT_fixed_point_quadratic_l1027_102768


namespace NUMINAMATH_GPT_thrushes_left_l1027_102783

theorem thrushes_left {init_thrushes : ℕ} (additional_thrushes : ℕ) (killed_ratio : ℚ) (killed : ℕ) (remaining : ℕ) :
  init_thrushes = 20 →
  additional_thrushes = 4 * 2 →
  killed_ratio = 1 / 7 →
  killed = killed_ratio * (init_thrushes + additional_thrushes) →
  remaining = init_thrushes + additional_thrushes - killed →
  remaining = 24 :=
by sorry

end NUMINAMATH_GPT_thrushes_left_l1027_102783


namespace NUMINAMATH_GPT_find_b_l1027_102781

theorem find_b (b : ℝ) : (∃ x : ℝ, (x^3 - 3*x^2 = -3*x + b ∧ (3*x^2 - 6*x = -3))) → b = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_b_l1027_102781


namespace NUMINAMATH_GPT_John_break_time_l1027_102755

-- Define the constants
def John_dancing_hours : ℕ := 8

-- Define the condition for James's dancing time 
def James_dancing_time (B : ℕ) : ℕ := 
  let total_time := John_dancing_hours + B
  total_time + total_time / 3

-- State the problem as a theorem
theorem John_break_time (B : ℕ) : John_dancing_hours + James_dancing_time B = 20 → B = 1 := 
  by sorry

end NUMINAMATH_GPT_John_break_time_l1027_102755


namespace NUMINAMATH_GPT_partA_partB_partC_partD_l1027_102724

variable (α β : ℝ)
variable (hα : 0 < α) (hα1 : α < 1)
variable (hβ : 0 < β) (hβ1 : β < 1)

theorem partA : 
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 := by
  sorry

theorem partB :
  β * (1 - β)^2 = β * (1 - β)^2 := by
  sorry

theorem partC :
  β * (1 - β)^2 + (1 - β)^3 = β * (1 - β)^2 + (1 - β)^3 := by
  sorry

theorem partD (hα0 : α < 0.5) :
  (1 - α) * (α - α^2) < (1 - α) := by
  sorry

end NUMINAMATH_GPT_partA_partB_partC_partD_l1027_102724


namespace NUMINAMATH_GPT_men_entered_l1027_102753

theorem men_entered (M W x : ℕ) 
  (h1 : 5 * M = 4 * W)
  (h2 : M + x = 14)
  (h3 : 2 * (W - 3) = 24) : 
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_men_entered_l1027_102753


namespace NUMINAMATH_GPT_shaded_rectangle_area_l1027_102721

def area_polygon : ℝ := 2016
def sides_polygon : ℝ := 18
def segments_persh : ℝ := 4

theorem shaded_rectangle_area :
  (area_polygon / sides_polygon) * segments_persh = 448 := 
sorry

end NUMINAMATH_GPT_shaded_rectangle_area_l1027_102721


namespace NUMINAMATH_GPT_gcf_of_48_180_120_l1027_102778

theorem gcf_of_48_180_120 : Nat.gcd (Nat.gcd 48 180) 120 = 12 := by
  sorry

end NUMINAMATH_GPT_gcf_of_48_180_120_l1027_102778


namespace NUMINAMATH_GPT_rectangle_area_l1027_102717

-- Definitions:
variables (l w : ℝ)

-- Conditions:
def condition1 : Prop := l = 4 * w
def condition2 : Prop := 2 * l + 2 * w = 200

-- Theorem statement:
theorem rectangle_area (h1 : condition1 l w) (h2 : condition2 l w) : l * w = 1600 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1027_102717


namespace NUMINAMATH_GPT_topaz_sapphire_value_equal_l1027_102722

/-
  Problem statement: Given the following conditions:
  1. One sapphire and two topazes are three times more valuable than an emerald: S + 2T = 3E
  2. Seven sapphires and one topaz are eight times more valuable than an emerald: 7S + T = 8E
  
  Prove that the value of one topaz is equal to the value of one sapphire (T = S).
-/

theorem topaz_sapphire_value_equal
  (S T E : ℝ) 
  (h1 : S + 2 * T = 3 * E) 
  (h2 : 7 * S + T = 8 * E) :
  T = S := 
  sorry

end NUMINAMATH_GPT_topaz_sapphire_value_equal_l1027_102722


namespace NUMINAMATH_GPT_stephen_hawking_philosophical_implications_l1027_102784

/-- Stephen Hawking's statements -/
def stephen_hawking_statement_1 := "The universe was not created by God"
def stephen_hawking_statement_2 := "Modern science can explain the origin of the universe"

/-- Definitions implied by Hawking's statements -/
def unity_of_world_lies_in_materiality := "The unity of the world lies in its materiality"
def thought_and_existence_identical := "Thought and existence are identical"

/-- Combined implication of Stephen Hawking's statements -/
def correct_philosophical_implications := [unity_of_world_lies_in_materiality, thought_and_existence_identical]

/-- Theorem: The correct philosophical implications of Stephen Hawking's statements are ① and ②. -/
theorem stephen_hawking_philosophical_implications :
  (stephen_hawking_statement_1 = "The universe was not created by God") →
  (stephen_hawking_statement_2 = "Modern science can explain the origin of the universe") →
  correct_philosophical_implications = ["The unity of the world lies in its materiality", "Thought and existence are identical"] :=
by
  sorry

end NUMINAMATH_GPT_stephen_hawking_philosophical_implications_l1027_102784


namespace NUMINAMATH_GPT_scientific_notation_448000_l1027_102723

theorem scientific_notation_448000 :
  ∃ a n, (448000 : ℝ) = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 4.48 ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_448000_l1027_102723


namespace NUMINAMATH_GPT_sum_of_ages_l1027_102791

variable (S T : ℕ)

theorem sum_of_ages (h1 : S = T + 7) (h2 : S + 10 = 3 * (T - 3)) : S + T = 33 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1027_102791


namespace NUMINAMATH_GPT_fraction_subtraction_l1027_102762

theorem fraction_subtraction (h : ((8 : ℚ) / 21 - (10 / 63) = (2 / 9))) : 
  8 / 21 - 10 / 63 = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1027_102762


namespace NUMINAMATH_GPT_tanvi_rank_among_girls_correct_l1027_102710

def Vikas_rank : ℕ := 9
def Tanvi_rank : ℕ := 17
def girls_between : ℕ := 2
def Tanvi_rank_among_girls : ℕ := 8

theorem tanvi_rank_among_girls_correct (Vikas_rank Tanvi_rank girls_between Tanvi_rank_among_girls : ℕ) 
  (h1 : Vikas_rank = 9) 
  (h2 : Tanvi_rank = 17) 
  (h3 : girls_between = 2)
  (h4 : Tanvi_rank_among_girls = 8): 
  Tanvi_rank_among_girls = 8 := by
  sorry

end NUMINAMATH_GPT_tanvi_rank_among_girls_correct_l1027_102710


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1027_102744

theorem distance_between_parallel_lines (a d : ℝ) :
  (∀ x y : ℝ, 2 * x - y + 3 = 0 ∧ a * x - y + 4 = 0 → (2 = a ∧ d = |(3 - 4)| / Real.sqrt (2 ^ 2 + (-1) ^ 2))) → 
  (a = 2 ∧ d = Real.sqrt 5 / 5) :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1027_102744


namespace NUMINAMATH_GPT_most_stable_city_l1027_102796

def variance_STD : ℝ := 12.5
def variance_A : ℝ := 18.3
def variance_B : ℝ := 17.4
def variance_C : ℝ := 20.1

theorem most_stable_city : variance_STD < variance_A ∧ variance_STD < variance_B ∧ variance_STD < variance_C :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_most_stable_city_l1027_102796


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1027_102704

theorem arithmetic_sequence_sum :
  ∃ x y z d : ℝ, 
  d = (31 - 4) / 5 ∧ 
  x = 4 + d ∧ 
  y = x + d ∧ 
  z = 16 + d ∧ 
  (x + y + z) = 45.6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1027_102704


namespace NUMINAMATH_GPT_line_through_point_and_area_l1027_102700

theorem line_through_point_and_area (a b : ℝ) (x y : ℝ) 
  (hx : x = -2) (hy : y = 2) 
  (h_area : 1/2 * |a * b| = 1): 
  (2 * x + y + 2 = 0 ∨ x + 2 * y - 2 = 0) :=
  sorry

end NUMINAMATH_GPT_line_through_point_and_area_l1027_102700


namespace NUMINAMATH_GPT_digitalEarth_correct_l1027_102713

-- Define the possible descriptions of "Digital Earth"
inductive DigitalEarthDescription
| optionA : DigitalEarthDescription
| optionB : DigitalEarthDescription
| optionC : DigitalEarthDescription
| optionD : DigitalEarthDescription

-- Define the correct description according to the solution
def correctDescription : DigitalEarthDescription := DigitalEarthDescription.optionB

-- Define the theorem to prove the equivalence
theorem digitalEarth_correct :
  correctDescription = DigitalEarthDescription.optionB :=
sorry

end NUMINAMATH_GPT_digitalEarth_correct_l1027_102713


namespace NUMINAMATH_GPT_wooden_block_length_is_correct_l1027_102789

noncomputable def length_of_block : ℝ :=
  let initial_length := 31
  let reduction := 30 / 100
  initial_length - reduction

theorem wooden_block_length_is_correct :
  length_of_block = 30.7 :=
by
  sorry

end NUMINAMATH_GPT_wooden_block_length_is_correct_l1027_102789


namespace NUMINAMATH_GPT_find_n_l1027_102716

noncomputable def binom (n k : ℕ) := Nat.choose n k

theorem find_n 
  (n : ℕ)
  (h1 : (binom (n-6) 7) / binom n 7 = (6 * binom (n-7) 6) / binom n 7)
  : n = 48 := by
  sorry

end NUMINAMATH_GPT_find_n_l1027_102716


namespace NUMINAMATH_GPT_number_of_1989_periodic_points_l1027_102793

noncomputable def f (z : ℂ) (m : ℕ) : ℂ := z ^ m

noncomputable def is_periodic_point (z : ℂ) (f : ℂ → ℂ) (n : ℕ) : Prop :=
f^[n] z = z ∧ ∀ k : ℕ, k < n → (f^[k] z) ≠ z

noncomputable def count_periodic_points (m n : ℕ) : ℕ :=
m^n - m^(n / 3) - m^(n / 13) - m^(n / 17) + m^(n / 39) + m^(n / 51) + m^(n / 117) - m^(n / 153)

theorem number_of_1989_periodic_points (m : ℕ) (hm : 1 < m) :
  count_periodic_points m 1989 = m^1989 - m^663 - m^153 - m^117 + m^51 + m^39 + m^9 - m^3 :=
sorry

end NUMINAMATH_GPT_number_of_1989_periodic_points_l1027_102793


namespace NUMINAMATH_GPT_average_wage_per_day_l1027_102725

variable (numMaleWorkers : ℕ) (wageMale : ℕ) (numFemaleWorkers : ℕ) (wageFemale : ℕ) (numChildWorkers : ℕ) (wageChild : ℕ)

theorem average_wage_per_day :
  numMaleWorkers = 20 →
  wageMale = 35 →
  numFemaleWorkers = 15 →
  wageFemale = 20 →
  numChildWorkers = 5 →
  wageChild = 8 →
  (20 * 35 + 15 * 20 + 5 * 8) / (20 + 15 + 5) = 26 :=
by
  intros
  -- Proof would follow here
  sorry

end NUMINAMATH_GPT_average_wage_per_day_l1027_102725


namespace NUMINAMATH_GPT_min_a4_in_arithmetic_sequence_l1027_102737

noncomputable def arithmetic_sequence_min_a4 (a1 d : ℝ) 
(S4 : ℝ := 4 * a1 + 6 * d)
(S5 : ℝ := 5 * a1 + 10 * d)
(a4 : ℝ := a1 + 3 * d) : Prop :=
  S4 ≤ 4 ∧ S5 ≥ 15 → a4 = 7

theorem min_a4_in_arithmetic_sequence (a1 d : ℝ) (h1 : 4 * a1 + 6 * d ≤ 4) 
(h2 : 5 * a1 + 10 * d ≥ 15) : 
arithmetic_sequence_min_a4 a1 d := 
by {
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_min_a4_in_arithmetic_sequence_l1027_102737
