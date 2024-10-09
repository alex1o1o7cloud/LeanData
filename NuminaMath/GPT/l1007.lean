import Mathlib

namespace min_value_of_A2_minus_B2_nonneg_l1007_100757

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)

theorem min_value_of_A2_minus_B2_nonneg (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z) ^ 2 - (B x y z) ^ 2 ≥ 36 :=
by
  sorry

end min_value_of_A2_minus_B2_nonneg_l1007_100757


namespace lily_read_total_books_l1007_100737

-- Definitions
def books_weekdays_last_month : ℕ := 4
def books_weekends_last_month : ℕ := 4

def books_weekdays_this_month : ℕ := 2 * books_weekdays_last_month
def books_weekends_this_month : ℕ := 3 * books_weekends_last_month

def total_books_last_month : ℕ := books_weekdays_last_month + books_weekends_last_month
def total_books_this_month : ℕ := books_weekdays_this_month + books_weekends_this_month
def total_books_two_months : ℕ := total_books_last_month + total_books_this_month

-- Proof problem statement
theorem lily_read_total_books : total_books_two_months = 28 :=
by
  sorry

end lily_read_total_books_l1007_100737


namespace find_m_l1007_100785

-- Mathematical conditions definitions
def line1 (x y : ℝ) (m : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0

-- Given the lines are parallel
def lines_parallel (l1 l2 : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m → l2 x y m → (3 / (m + 2)) = (m / (-(m - 2)))

-- The proof problem statement
theorem find_m (m : ℝ) : 
  lines_parallel (line1) (line2) m → (m = -6 ∨ m = 1) :=
by
  sorry

end find_m_l1007_100785


namespace field_ratio_l1007_100765

theorem field_ratio (w : ℝ) (h : ℝ) (pond_len : ℝ) (field_len : ℝ) 
  (h1 : pond_len = 8) 
  (h2 : field_len = 112) 
  (h3 : w > 0) 
  (h4 : field_len = w * h) 
  (h5 : pond_len * pond_len = (1 / 98) * (w * h * h)) : 
  field_len / h = 2 := 
by 
  sorry

end field_ratio_l1007_100765


namespace decrease_in_sales_percentage_l1007_100740

theorem decrease_in_sales_percentage (P Q : Real) :
  let P' := 1.40 * P
  let R := P * Q
  let R' := 1.12 * R
  ∃ (D : Real), Q' = Q * (1 - D / 100) ∧ R' = P' * Q' → D = 20 :=
by
  sorry

end decrease_in_sales_percentage_l1007_100740


namespace percentage_of_b_l1007_100704

variable (a b c p : ℝ)

-- Conditions
def condition1 : Prop := 0.02 * a = 8
def condition2 : Prop := c = b / a
def condition3 : Prop := p * b = 2

-- Theorem statement
theorem percentage_of_b (h1 : condition1 a)
                        (h2 : condition2 b a c)
                        (h3 : condition3 p b) :
  p = 0.005 := sorry

end percentage_of_b_l1007_100704


namespace divisible_by_17_l1007_100726

theorem divisible_by_17 (k : ℕ) : 17 ∣ (2^(2*k+3) + 3^(k+2) * 7^k) :=
  sorry

end divisible_by_17_l1007_100726


namespace trigonometric_identity_l1007_100708

theorem trigonometric_identity (α : ℝ)
 (h : Real.sin (α / 2) - 2 * Real.cos (α / 2) = 1) :
  (1 + Real.sin α + Real.cos α) / (1 + Real.sin α - Real.cos α) = 3 / 4 := 
sorry

end trigonometric_identity_l1007_100708


namespace percent_workday_in_meetings_l1007_100795

def first_meeting_duration : ℕ := 30
def second_meeting_duration : ℕ := 3 * first_meeting_duration
def third_meeting_duration : ℕ := 2 * second_meeting_duration
def total_meeting_time : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration
def workday_duration : ℕ := 10 * 60

theorem percent_workday_in_meetings : (total_meeting_time : ℚ) / workday_duration * 100 = 50 := by
  sorry

end percent_workday_in_meetings_l1007_100795


namespace isosceles_triangle_base_l1007_100788

theorem isosceles_triangle_base (b : ℝ) (h1 : 7 + 7 + b = 20) : b = 6 :=
by {
    sorry
}

end isosceles_triangle_base_l1007_100788


namespace clerical_percentage_after_reduction_l1007_100798

-- Define the initial conditions
def total_employees : ℕ := 3600
def clerical_fraction : ℚ := 1/4
def reduction_fraction : ℚ := 1/4

-- Define the intermediate calculations
def initial_clerical_employees : ℚ := clerical_fraction * total_employees
def clerical_reduction : ℚ := reduction_fraction * initial_clerical_employees
def new_clerical_employees : ℚ := initial_clerical_employees - clerical_reduction
def total_employees_after_reduction : ℚ := total_employees - clerical_reduction

-- State the theorem
theorem clerical_percentage_after_reduction :
  (new_clerical_employees / total_employees_after_reduction) * 100 = 20 :=
sorry

end clerical_percentage_after_reduction_l1007_100798


namespace sunset_time_l1007_100780

def length_of_daylight_in_minutes := 11 * 60 + 12
def sunrise_time_in_minutes := 6 * 60 + 45
def sunset_time_in_minutes := sunrise_time_in_minutes + length_of_daylight_in_minutes
def sunset_time_hour := sunset_time_in_minutes / 60
def sunset_time_minute := sunset_time_in_minutes % 60
def sunset_time_12hr_format := if sunset_time_hour >= 12 
    then (sunset_time_hour - 12, sunset_time_minute)
    else (sunset_time_hour, sunset_time_minute)

theorem sunset_time : sunset_time_12hr_format = (5, 57) :=
by
  sorry

end sunset_time_l1007_100780


namespace ac_lt_bc_if_c_lt_zero_l1007_100738

variables {a b c : ℝ}
theorem ac_lt_bc_if_c_lt_zero (h : a > b) (h1 : b > c) (h2 : c < 0) : a * c < b * c :=
sorry

end ac_lt_bc_if_c_lt_zero_l1007_100738


namespace find_a_sequence_formula_l1007_100702

variable (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ)
variable (h_f_def : ∀ x ≠ -a, f x = (a * x) / (a + x))
variable (h_f_2 : f 2 = 1) (h_seq_def : ∀ n : ℕ, a_seq (n+1) = f (a_seq n)) (h_a1 : a_seq 1 = 1)

theorem find_a : a = 2 :=
  sorry

theorem sequence_formula : ∀ n : ℕ, a_seq n = 2 / (n + 1) :=
  sorry

end find_a_sequence_formula_l1007_100702


namespace max_two_integers_abs_leq_50_l1007_100742

theorem max_two_integers_abs_leq_50
  (a b c : ℤ) (h_a : a > 100) :
  ∀ {x1 x2 x3 : ℤ}, (abs (a * x1^2 + b * x1 + c) ≤ 50) →
                    (abs (a * x2^2 + b * x2 + c) ≤ 50) →
                    (abs (a * x3^2 + b * x3 + c) ≤ 50) →
                    false :=
sorry

end max_two_integers_abs_leq_50_l1007_100742


namespace find_equation_of_line_l1007_100703

-- Define the given conditions
def center_of_circle : ℝ × ℝ := (0, 3)
def perpendicular_line_slope : ℝ := -1
def perpendicular_line_equation (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the proof problem
theorem find_equation_of_line (x y : ℝ) (l_passes_center : (x, y) = center_of_circle)
 (l_is_perpendicular : ∀ x y, perpendicular_line_equation x y ↔ (x-y+3=0)) : x - y + 3 = 0 :=
sorry

end find_equation_of_line_l1007_100703


namespace quotient_when_divided_by_44_l1007_100751

theorem quotient_when_divided_by_44 (N Q P : ℕ) (h1 : N = 44 * Q) (h2 : N = 35 * P + 3) : Q = 12 :=
by {
  -- Proof
  sorry
}

end quotient_when_divided_by_44_l1007_100751


namespace vertex_angle_double_angle_triangle_l1007_100730

theorem vertex_angle_double_angle_triangle 
  {α β : ℝ} (h1 : α + β + β = 180) (h2 : α = 2 * β ∨ β = 2 * α) :
  α = 36 ∨ α = 90 :=
by
  sorry

end vertex_angle_double_angle_triangle_l1007_100730


namespace linear_transform_determined_by_points_l1007_100710

theorem linear_transform_determined_by_points
  (z1 z2 w1 w2 : ℂ)
  (h1 : z1 ≠ z2)
  (h2 : w1 ≠ w2)
  : ∃ (a b : ℂ), ∀ (z : ℂ), a = (w2 - w1) / (z2 - z1) ∧ b = (w1 * z2 - w2 * z1) / (z2 - z1) ∧ (a * z1 + b = w1) ∧ (a * z2 + b = w2) := 
sorry

end linear_transform_determined_by_points_l1007_100710


namespace remainder_when_200_divided_by_k_l1007_100772

theorem remainder_when_200_divided_by_k 
  (k : ℕ) (k_pos : 0 < k)
  (h : 120 % k^2 = 12) :
  200 % k = 2 :=
sorry

end remainder_when_200_divided_by_k_l1007_100772


namespace amusement_park_admission_fees_l1007_100779

theorem amusement_park_admission_fees
  (num_children : ℕ) (num_adults : ℕ)
  (fee_child : ℝ) (fee_adult : ℝ)
  (total_people : ℕ) (expected_total_fees : ℝ) :
  num_children = 180 →
  fee_child = 1.5 →
  fee_adult = 4.0 →
  total_people = 315 →
  expected_total_fees = 810 →
  num_children + num_adults = total_people →
  (num_children : ℝ) * fee_child + (num_adults : ℝ) * fee_adult = expected_total_fees := by
  intros h1 h2 h3 h4 h5 h6
  sorry

end amusement_park_admission_fees_l1007_100779


namespace part1_part2_l1007_100706

theorem part1 : ∃ x : ℝ, 3 * x = 4.5 ∧ x = 4.5 - 3 :=
by {
  -- Skipping the proof for now
  sorry
}

theorem part2 (m : ℝ) (h : ∃ x : ℝ, 5 * x - m = 1 ∧ x = 1 - m - 5) : m = 21 / 4 :=
by {
  -- Skipping the proof for now
  sorry
}

end part1_part2_l1007_100706


namespace percentage_problem_l1007_100778

theorem percentage_problem (x : ℝ) (h : 0.30 * 0.15 * x = 18) : 0.15 * 0.30 * x = 18 :=
by
  sorry

end percentage_problem_l1007_100778


namespace math_problem_l1007_100728

open Real

theorem math_problem (x : ℝ) (p q : ℕ)
  (h1 : (1 + sin x) * (1 + cos x) = 9 / 4)
  (h2 : (1 - sin x) * (1 - cos x) = p - sqrt q)
  (hp_pos : p > 0) (hq_pos : q > 0) : p + q = 1 := sorry

end math_problem_l1007_100728


namespace line_eq_circle_eq_l1007_100789

section
  variable (A B : ℝ × ℝ)
  variable (A_eq : A = (4, 6))
  variable (B_eq : B = (-2, 4))

  theorem line_eq : ∃ (a b c : ℝ), (a, b, c) = (1, -3, 14) ∧ ∀ x y, (y - 6) = ((4 - 6) / (-2 - 4)) * (x - 4) → a * x + b * y + c = 0 :=
  sorry

  theorem circle_eq : ∃ (h k r : ℝ), (h, k, r) = (1, 5, 10) ∧ ∀ x y, (x - 1)^2 + (y - 5)^2 = 10 :=
  sorry
end

end line_eq_circle_eq_l1007_100789


namespace odd_function_f1_eq_4_l1007_100786

theorem odd_function_f1_eq_4 (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x < 0 → f x = x^2 + a * x)
  (h3 : f 2 = 6) : 
  f 1 = 4 :=
by sorry

end odd_function_f1_eq_4_l1007_100786


namespace correct_equations_l1007_100720

theorem correct_equations (x y : ℝ) :
  (9 * x - y = 4) → (y - 8 * x = 3) → (9 * x - y = 4 ∧ y - 8 * x = 3) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end correct_equations_l1007_100720


namespace sum_of_roots_proof_l1007_100784

noncomputable def sum_of_roots (x1 x2 x3 : ℝ) : ℝ :=
  let eq1 := (11 - x1)^3 + (13 - x1)^3 = (24 - 2 * x1)^3
  let eq2 := (11 - x2)^3 + (13 - x2)^3 = (24 - 2 * x2)^3
  let eq3 := (11 - x3)^3 + (13 - x3)^3 = (24 - 2 * x3)^3
  x1 + x2 + x3

theorem sum_of_roots_proof : sum_of_roots 11 12 13 = 36 :=
  sorry

end sum_of_roots_proof_l1007_100784


namespace original_prices_l1007_100700

theorem original_prices 
  (S P J : ℝ)
  (hS : 0.80 * S = 780)
  (hP : 0.70 * P = 2100)
  (hJ : 0.90 * J = 2700) :
  S = 975 ∧ P = 3000 ∧ J = 3000 :=
by
  sorry

end original_prices_l1007_100700


namespace tan_C_l1007_100796

theorem tan_C (A B C : ℝ) (hABC : A + B + C = π) (tan_A : Real.tan A = 1 / 2) 
  (cos_B : Real.cos B = 3 * Real.sqrt 10 / 10) : Real.tan C = -1 :=
by
  sorry

end tan_C_l1007_100796


namespace prism_cutout_l1007_100787

noncomputable def original_volume : ℕ := 15 * 5 * 4 -- Volume of the original prism
noncomputable def cutout_width : ℕ := 5

variables {x y : ℕ}

theorem prism_cutout:
  -- Given conditions
  (15 > 0) ∧ (5 > 0) ∧ (4 > 0) ∧ (x > 0) ∧ (y > 0) ∧ 
  -- The volume condition
  (original_volume - y * cutout_width * x = 120) →
  -- Prove that x + y = 15
  (x + y = 15) :=
sorry

end prism_cutout_l1007_100787


namespace find_pairs_l1007_100760

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_l1007_100760


namespace correct_quotient_and_remainder_l1007_100741

theorem correct_quotient_and_remainder:
  let incorrect_divisor := 47
  let incorrect_quotient := 5
  let incorrect_remainder := 8
  let incorrect_dividend := incorrect_divisor * incorrect_quotient + incorrect_remainder
  let correct_dividend := 243
  let correct_divisor := 74
  (correct_dividend / correct_divisor = 3 ∧ correct_dividend % correct_divisor = 21) :=
by sorry

end correct_quotient_and_remainder_l1007_100741


namespace farm_cows_l1007_100763

theorem farm_cows (x y : ℕ) (h : 4 * x + 2 * y = 20 + 3 * (x + y)) : x = 20 + y :=
sorry

end farm_cows_l1007_100763


namespace intersection_complement_l1007_100761

def set_M : Set ℝ := {x : ℝ | x^2 - x = 0}

def set_N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = 2 * n + 1}

theorem intersection_complement (h : UniversalSet = Set.univ) :
  set_M ∩ (UniversalSet \ set_N) = {0} := 
sorry

end intersection_complement_l1007_100761


namespace polynomial_roots_arithmetic_progression_not_all_real_l1007_100732

theorem polynomial_roots_arithmetic_progression_not_all_real :
  ∀ (a : ℝ), (∃ r d : ℂ, r - d ≠ r ∧ r ≠ r + d ∧ r - d + r + (r + d) = 9 ∧ (r - d) * r + (r - d) * (r + d) + r * (r + d) = 33 ∧ d ≠ 0) →
  a = -45 :=
by
  sorry

end polynomial_roots_arithmetic_progression_not_all_real_l1007_100732


namespace machine_value_after_two_years_l1007_100721

def machine_value (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - rate)^years

theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 :=
by
  sorry

end machine_value_after_two_years_l1007_100721


namespace sum_first_11_terms_l1007_100707

variable (a : ℕ → ℤ) -- The arithmetic sequence
variable (d : ℤ) -- Common difference
variable (S : ℕ → ℤ) -- Sum of the arithmetic sequence

-- The properties of the arithmetic sequence and sum
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_arith_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 5 + a 8 = a 2 + 12

-- To prove
theorem sum_first_11_terms : S 11 = 66 := by
  sorry

end sum_first_11_terms_l1007_100707


namespace number_of_true_propositions_is_zero_l1007_100770

theorem number_of_true_propositions_is_zero :
  (∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0) →
  (¬ ∃ x : ℚ, x^2 = 2) →
  (¬ ∃ x : ℝ, x^2 + 1 = 0) →
  (∀ x : ℝ, 4 * x^2 ≤ 2 * x - 1 + 3 * x^2) →
  true :=  -- representing that the number of true propositions is 0
by
  intros h1 h2 h3 h4
  sorry

end number_of_true_propositions_is_zero_l1007_100770


namespace frank_spent_on_mower_blades_l1007_100724

def money_made := 19
def money_spent_on_games := 4 * 2
def money_left := money_made - money_spent_on_games

theorem frank_spent_on_mower_blades : money_left = 11 :=
by
  -- we are providing the proof steps here in comments, but in the actual code, it's just sorry
  -- calc money_left
  --    = money_made - money_spent_on_games : by refl
  --    = 19 - 8 : by norm_num
  --    = 11 : by norm_num
  sorry

end frank_spent_on_mower_blades_l1007_100724


namespace cubic_sum_of_roots_l1007_100768

theorem cubic_sum_of_roots (a b c : ℝ) 
  (h1 : a + b + c = -1)
  (h2 : a * b + b * c + c * a = -333)
  (h3 : a * b * c = 1001) :
  a^3 + b^3 + c^3 = 2003 :=
sorry

end cubic_sum_of_roots_l1007_100768


namespace simple_and_compound_interest_difference_l1007_100774

theorem simple_and_compound_interest_difference (r : ℝ) :
  let P := 3600
  let t := 2
  let SI := P * r * t / 100
  let CI := P * (1 + r / 100)^t - P
  CI - SI = 225 → r = 25 := by
  intros
  sorry

end simple_and_compound_interest_difference_l1007_100774


namespace value_of_b_l1007_100714

theorem value_of_b (b : ℝ) : 
  (∀ x : ℝ, -x ^ 2 + b * x + 7 < 0 ↔ x < -2 ∨ x > 3) → b = 1 :=
by
  sorry

end value_of_b_l1007_100714


namespace translation_invariant_line_l1007_100792

theorem translation_invariant_line (k : ℝ) :
  (∀ x : ℝ, k * (x - 2) + 5 = k * x + 2) → k = 3 / 2 :=
by
  sorry

end translation_invariant_line_l1007_100792


namespace smallest_m_inequality_l1007_100747

theorem smallest_m_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1) : 27 * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1 :=
sorry

end smallest_m_inequality_l1007_100747


namespace geometric_sequence_ratio_l1007_100782

variable {a : ℕ → ℝ} -- Define the geometric sequence {a_n}

-- Conditions: The sequence is geometric with positive terms
variable (q : ℝ) (hq : q > 0) (hgeo : ∀ n, a (n + 1) = q * a n)

-- Additional condition: a2, 1/2 a3, and a1 form an arithmetic sequence
variable (hseq : a 1 - (1 / 2) * a 2 = (1 / 2) * a 2 - a 0)

theorem geometric_sequence_ratio :
  (a 3 + a 4) / (a 2 + a 3) = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_sequence_ratio_l1007_100782


namespace value_of_a_l1007_100748

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l1007_100748


namespace soccer_team_points_l1007_100746

theorem soccer_team_points 
  (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_draws : draws = total_games - (wins + losses))
  (h_points_per_win : points_per_win = 3)
  (h_points_per_draw : points_per_draw = 1)
  (h_points_per_loss : points_per_loss = 0) :
  (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) = 46 :=
by
  -- the actual proof steps will be inserted here
  sorry

end soccer_team_points_l1007_100746


namespace sequence_term_l1007_100722

theorem sequence_term (a : ℕ → ℕ) 
  (h1 : a 1 = 2009) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n + 1) 
  : a 1000 = 2342 := 
by 
  sorry

end sequence_term_l1007_100722


namespace regression_lines_have_common_point_l1007_100705

theorem regression_lines_have_common_point
  (n m : ℕ)
  (h₁ : n = 10)
  (h₂ : m = 15)
  (s t : ℝ)
  (data_A data_B : Fin n → Fin n → ℝ)
  (avg_x_A avg_x_B : ℝ)
  (avg_y_A avg_y_B : ℝ)
  (regression_line_A regression_line_B : ℝ → ℝ)
  (h₃ : avg_x_A = s)
  (h₄ : avg_x_B = s)
  (h₅ : avg_y_A = t)
  (h₆ : avg_y_B = t)
  (h₇ : ∀ x, regression_line_A x = a*x + b)
  (h₈ : ∀ x, regression_line_B x = c*x + d)
  : regression_line_A s = t ∧ regression_line_B s = t :=
by
  sorry

end regression_lines_have_common_point_l1007_100705


namespace arcsin_sqrt_one_half_l1007_100752

theorem arcsin_sqrt_one_half : Real.arcsin (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  -- TODO: provide proof
  sorry

end arcsin_sqrt_one_half_l1007_100752


namespace bars_sold_this_week_l1007_100755

-- Definitions based on conditions
def total_bars : Nat := 18
def bars_sold_last_week : Nat := 5
def bars_needed_to_sell : Nat := 6

-- Statement of the proof problem
theorem bars_sold_this_week : (total_bars - (bars_needed_to_sell + bars_sold_last_week)) = 2 := by
  -- proof goes here
  sorry

end bars_sold_this_week_l1007_100755


namespace find_a_l1007_100713

theorem find_a (a : ℝ) : 
  let term_coeff (r : ℕ) := (Nat.choose 10 r : ℝ)
  let coeff_x6 := term_coeff 3 - (a * term_coeff 2)
  coeff_x6 = 30 → a = 2 :=
by
  intro h
  sorry

end find_a_l1007_100713


namespace charles_draws_yesterday_after_work_l1007_100750

theorem charles_draws_yesterday_after_work :
  ∀ (initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work : ℕ),
    initial_papers = 20 →
    today_drawn = 6 →
    yesterday_drawn_before_work = 6 →
    current_papers_left = 2 →
    (initial_papers - (today_drawn + yesterday_drawn_before_work) - yesterday_drawn_after_work = current_papers_left) →
    yesterday_drawn_after_work = 6 :=
by
  intros initial_papers today_drawn yesterday_drawn_before_work current_papers_left yesterday_drawn_after_work
  intro h1 h2 h3 h4 h5
  sorry

end charles_draws_yesterday_after_work_l1007_100750


namespace product_of_two_integers_l1007_100781

def gcd_lcm_prod (x y : ℕ) :=
  Nat.gcd x y = 8 ∧ Nat.lcm x y = 48

theorem product_of_two_integers (x y : ℕ) (h : gcd_lcm_prod x y) : x * y = 384 :=
by
  sorry

end product_of_two_integers_l1007_100781


namespace tammy_haircuts_l1007_100754

theorem tammy_haircuts (total_haircuts free_haircuts haircuts_to_next_free : ℕ) 
(h1 : free_haircuts = 5) 
(h2 : haircuts_to_next_free = 5) 
(h3 : total_haircuts = 79) : 
(haircuts_to_next_free = 5) :=
by {
  sorry
}

end tammy_haircuts_l1007_100754


namespace evaluate_expression_l1007_100719

theorem evaluate_expression :
  (15 - 14 + 13 - 12 + 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1) / (1 - 2 + 3 - 4 + 5 - 6 + 7) = 2 :=
by
  sorry

end evaluate_expression_l1007_100719


namespace total_students_in_class_l1007_100716

theorem total_students_in_class 
  (hockey_players : ℕ)
  (basketball_players : ℕ)
  (neither_players : ℕ)
  (both_players : ℕ)
  (hockey_players_eq : hockey_players = 15)
  (basketball_players_eq : basketball_players = 16)
  (neither_players_eq : neither_players = 4)
  (both_players_eq : both_players = 10) :
  hockey_players + basketball_players - both_players + neither_players = 25 := 
by 
  sorry

end total_students_in_class_l1007_100716


namespace monotonic_decreasing_intervals_l1007_100776

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem monotonic_decreasing_intervals : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) ∧
  (∀ x : ℝ, (1 < x ∧ x < Real.exp 1) → ∃ ε > 0, ∀ y : ℝ, x ≤ y ∧ y ≤ x + ε → f y < f x) :=
by
  sorry

end monotonic_decreasing_intervals_l1007_100776


namespace fraction_difference_l1007_100717

variable (x y : ℝ)

theorem fraction_difference (h : x / y = 2) : (x - y) / y = 1 :=
  sorry

end fraction_difference_l1007_100717


namespace reduction_percentage_toy_l1007_100756

-- Definition of key parameters
def paintings_bought : ℕ := 10
def cost_per_painting : ℕ := 40
def toys_bought : ℕ := 8
def cost_per_toy : ℕ := 20
def total_cost : ℕ := (paintings_bought * cost_per_painting) + (toys_bought * cost_per_toy) -- $560
def painting_selling_price_per_unit : ℕ := cost_per_painting - (cost_per_painting * 10 / 100) -- $36
def total_loss : ℕ := 64

-- Define percentage reduction in the selling price of a wooden toy
variable {x : ℕ} -- Define x as a percentage value to be solved

-- Theorems to prove
theorem reduction_percentage_toy (x) : 
  (paintings_bought * painting_selling_price_per_unit) 
  + (toys_bought * (cost_per_toy - (cost_per_toy * x / 100))) 
  = (total_cost - total_loss) 
  → x = 15 := 
by
  sorry

end reduction_percentage_toy_l1007_100756


namespace tiling_2002_gon_with_rhombuses_l1007_100731

theorem tiling_2002_gon_with_rhombuses : ∀ n : ℕ, n = 1001 → (n * (n - 1) / 2) = 500500 :=
by sorry

end tiling_2002_gon_with_rhombuses_l1007_100731


namespace modulus_difference_l1007_100711

def z1 : Complex := 1 + 2 * Complex.I
def z2 : Complex := 2 + Complex.I

theorem modulus_difference :
  Complex.abs (z2 - z1) = Real.sqrt 2 := by sorry

end modulus_difference_l1007_100711


namespace two_R_theta_bounds_l1007_100744

variables {R : ℝ} (θ : ℝ)
variables (h_pos : 0 < R) (h_triangle : (R + 1 + (R + 1/2)) > 2 *R)

-- Define that θ is the angle between sides R and R + 1/2
-- Here we assume θ is defined via the cosine rule for simplicity

noncomputable def angle_between_sides (R : ℝ) := 
  Real.arccos ((R^2 + (R + 1/2)^2 - 1^2) / (2 * R * (R + 1/2)))

-- State the theorem
theorem two_R_theta_bounds (h : θ = angle_between_sides R) : 
  1 < 2 * R * θ ∧ 2 * R * θ < π :=
by
  sorry

end two_R_theta_bounds_l1007_100744


namespace cakes_baked_yesterday_l1007_100734

noncomputable def BakedToday : ℕ := 5
noncomputable def SoldDinner : ℕ := 6
noncomputable def Left : ℕ := 2

theorem cakes_baked_yesterday (CakesBakedYesterday : ℕ) : 
  BakedToday + CakesBakedYesterday - SoldDinner = Left → CakesBakedYesterday = 3 := 
by 
  intro h 
  sorry

end cakes_baked_yesterday_l1007_100734


namespace average_salary_of_all_workers_is_correct_l1007_100773

noncomputable def average_salary_all_workers (n_total n_tech : ℕ) (avg_salary_tech avg_salary_others : ℝ) : ℝ :=
  let n_others := n_total - n_tech
  let total_salary_tech := n_tech * avg_salary_tech
  let total_salary_others := n_others * avg_salary_others
  let total_salary := total_salary_tech + total_salary_others
  total_salary / n_total

theorem average_salary_of_all_workers_is_correct :
  average_salary_all_workers 21 7 12000 6000 = 8000 :=
by
  unfold average_salary_all_workers
  sorry

end average_salary_of_all_workers_is_correct_l1007_100773


namespace number_of_days_to_catch_fish_l1007_100766

variable (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ)

theorem number_of_days_to_catch_fish (h1 : fish_per_day = 2) 
                                    (h2 : fillets_per_fish = 2) 
                                    (h3 : total_fillets = 120) : 
                                    (total_fillets / fillets_per_fish) / fish_per_day = 30 :=
by sorry

end number_of_days_to_catch_fish_l1007_100766


namespace range_of_a_l1007_100718

theorem range_of_a (a : ℝ) :
  (∀ x: ℝ, |x - a| < 4 → -x^2 + 5 * x - 6 > 0) → (-1 ≤ a ∧ a ≤ 6) :=
by
  intro h
  sorry

end range_of_a_l1007_100718


namespace monotonicity_of_f_range_of_a_l1007_100777

noncomputable def f (a x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x, f a x ≥ f a (2 * a) → x ≤ 0 ∨ x ≤ 2 * a)) ∧
  (a = 0 → ∀ x y, x ≤ y → f a x ≥ f a y) ∧
  (a > 0 → (∀ x, f a x ≤ f a 0 → x ≤ 0) ∧
           (∀ x, 0 < x ∧ x < 2 * a → f a x ≥ f a 2 * a) ∧
           (∀ x, 2 * a < x → f a x ≤ f a (2 * a))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, a ≥ 1 / 2 →
  ∃ x1 : ℝ, x1 > 0 ∧ ∃ x2 : ℝ, f a x1 ≥ g a x2 :=
sorry

end monotonicity_of_f_range_of_a_l1007_100777


namespace correct_calculation_l1007_100701

variable (a : ℝ)

theorem correct_calculation :
  a^6 / (1/2 * a^2) = 2 * a^4 :=
by
  sorry

end correct_calculation_l1007_100701


namespace polynomial_roots_l1007_100797

theorem polynomial_roots (p q BD DC : ℝ) (h_sum : BD + DC = p) (h_prod : BD * DC = q^2) :
    Polynomial.roots (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C p * Polynomial.X + Polynomial.C (q^2)) = {BD, DC} :=
sorry

end polynomial_roots_l1007_100797


namespace second_number_mod_12_l1007_100725

theorem second_number_mod_12 (x : ℕ) (h : (1274 * x * 1277 * 1285) % 12 = 6) : x % 12 = 1 := 
by 
  sorry

end second_number_mod_12_l1007_100725


namespace product_of_marbles_l1007_100745

theorem product_of_marbles (R B : ℕ) (h1 : R - B = 12) (h2 : R + B = 52) : R * B = 640 := by
  sorry

end product_of_marbles_l1007_100745


namespace x_in_terms_of_y_y_in_terms_of_x_l1007_100715

-- Define the main equation
variable (x y : ℝ)

-- First part: Expressing x in terms of y given the condition
theorem x_in_terms_of_y (h : x + 3 * y = 3) : x = 3 - 3 * y :=
by
  sorry

-- Second part: Expressing y in terms of x given the condition
theorem y_in_terms_of_x (h : x + 3 * y = 3) : y = (3 - x) / 3 :=
by
  sorry

end x_in_terms_of_y_y_in_terms_of_x_l1007_100715


namespace arithmetic_sequence_properties_l1007_100793

theorem arithmetic_sequence_properties
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * ((a_n 0 + a_n (n-1)) / 2))
  (h2 : S 6 < S 7)
  (h3 : S 7 > S 8) :
  (a_n 8 - a_n 7 < 0) ∧ (S 9 < S 6) ∧ (∀ m, S m ≤ S 7) :=
by
  sorry

end arithmetic_sequence_properties_l1007_100793


namespace solution_set_of_quadratic_inequality_l1007_100759

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} :=
sorry

end solution_set_of_quadratic_inequality_l1007_100759


namespace minimum_sum_distances_square_l1007_100743

noncomputable def minimum_sum_of_distances
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : ℝ :=
(1 + Real.sqrt 2) * d

theorem minimum_sum_distances_square
    (A B : ℝ × ℝ)
    (d : ℝ)
    (h_dist: dist A B = d)
    : minimum_sum_of_distances A B d h_dist = (1 + Real.sqrt 2) * d := by
sorry

end minimum_sum_distances_square_l1007_100743


namespace total_cans_from_256_l1007_100712

-- Define the recursive function to compute the number of new cans produced.
def total_new_cans (n : ℕ) : ℕ :=
  if n < 4 then 0
  else
    let rec_cans := total_new_cans (n / 4)
    (n / 4) + rec_cans

-- Theorem stating the total number of new cans that can be made from 256 initial cans.
theorem total_cans_from_256 : total_new_cans 256 = 85 := by
  sorry

end total_cans_from_256_l1007_100712


namespace paperclips_exceed_200_at_friday_l1007_100791

def paperclips_on_day (n : ℕ) : ℕ :=
  3 * 4^n

theorem paperclips_exceed_200_at_friday : 
  ∃ n : ℕ, n = 4 ∧ paperclips_on_day n > 200 :=
by
  sorry

end paperclips_exceed_200_at_friday_l1007_100791


namespace volume_pyramid_PABCD_is_384_l1007_100762

noncomputable def volume_of_pyramid : ℝ :=
  let AB := 12
  let BC := 6
  let PA := Real.sqrt (20^2 - 12^2)
  let base_area := AB * BC
  (1 / 3) * base_area * PA

theorem volume_pyramid_PABCD_is_384 :
  volume_of_pyramid = 384 := 
by
  sorry

end volume_pyramid_PABCD_is_384_l1007_100762


namespace loss_percentage_l1007_100723

-- Definitions related to the problem
def CPA : Type := ℝ
def SPAB (CPA: ℝ) : ℝ := 1.30 * CPA
def SPBC (CPA: ℝ) : ℝ := 1.040000000000000036 * CPA

-- Theorem to prove the loss percentage when B sold the bicycle to C 
theorem loss_percentage (CPA : ℝ) (L : ℝ) (h1 : SPAB CPA * (1 - L) = SPBC CPA) : 
  L = 0.20 :=
by
  sorry

end loss_percentage_l1007_100723


namespace largest_c_in_range_l1007_100799

theorem largest_c_in_range (c : ℝ) (h : ∃ x : ℝ,  2 * x ^ 2 - 4 * x + c = 5) : c ≤ 7 :=
by sorry

end largest_c_in_range_l1007_100799


namespace trajectory_equation_necessary_not_sufficient_l1007_100775

theorem trajectory_equation_necessary_not_sufficient :
  ∀ (x y : ℝ), (|x| = |y|) → (y = |x|) ↔ (necessary_not_sufficient) :=
by
  sorry

end trajectory_equation_necessary_not_sufficient_l1007_100775


namespace tony_remaining_money_l1007_100735

theorem tony_remaining_money :
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by
  let initial_amount := 20
  let ticket_cost := 8
  let hotdog_cost := 3
  show initial_amount - ticket_cost - hotdog_cost = 9
  sorry

end tony_remaining_money_l1007_100735


namespace total_marbles_l1007_100709

/-- A craftsman makes 35 jars. This is exactly 2.5 times the number of clay pots he made.
If each jar has 5 marbles and each clay pot has four times as many marbles as the jars plus an additional 3 marbles, 
prove that the total number of marbles is 497. -/
theorem total_marbles (number_of_jars : ℕ) (number_of_clay_pots : ℕ) (marbles_in_jar : ℕ) (marbles_in_clay_pot : ℕ) :
  number_of_jars = 35 →
  (number_of_jars : ℝ) = 2.5 * number_of_clay_pots →
  marbles_in_jar = 5 →
  marbles_in_clay_pot = 4 * marbles_in_jar + 3 →
  (number_of_jars * marbles_in_jar + number_of_clay_pots * marbles_in_clay_pot) = 497 :=
by 
  sorry

end total_marbles_l1007_100709


namespace pencil_and_pen_cost_l1007_100758

theorem pencil_and_pen_cost
  (p q : ℝ)
  (h1 : 3 * p + 2 * q = 3.75)
  (h2 : 2 * p + 3 * q = 4.05) :
  p + q = 1.56 :=
by
  sorry

end pencil_and_pen_cost_l1007_100758


namespace last_four_digits_of_5_pow_15000_l1007_100727

theorem last_four_digits_of_5_pow_15000 (h : 5^500 ≡ 1 [MOD 2000]) : 
  5^15000 ≡ 1 [MOD 2000] :=
sorry

end last_four_digits_of_5_pow_15000_l1007_100727


namespace min_odd_solution_l1007_100749

theorem min_odd_solution (a m1 m2 n1 n2 : ℕ)
  (h1: a = m1^2 + n1^2)
  (h2: a^2 = m2^2 + n2^2)
  (h3: m1 - n1 = m2 - n2)
  (h4: a > 5)
  (h5: a % 2 = 1) :
  a = 261 :=
sorry

end min_odd_solution_l1007_100749


namespace functional_relationship_selling_price_l1007_100733

open Real

-- Definitions used from conditions
def cost_price : ℝ := 20
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 80

-- Functional relationship between daily sales profit W and selling price x
def daily_sales_profit (x : ℝ) : ℝ :=
  (x - cost_price) * daily_sales_quantity x

-- Part (1): Prove the functional relationship
theorem functional_relationship (x : ℝ) :
  daily_sales_profit x = -2 * x^2 + 120 * x - 1600 :=
by {
  sorry
}

-- Part (2): Prove the selling price should be $25 to achieve $150 profit with condition x ≤ 30
theorem selling_price (x : ℝ) :
  daily_sales_profit x = 150 ∧ x ≤ 30 → x = 25 :=
by {
  sorry
}

end functional_relationship_selling_price_l1007_100733


namespace find_largest_value_l1007_100753

theorem find_largest_value
  (h1: 0 < Real.sin 2) (h2: Real.sin 2 < 1)
  (h3: Real.log 2 / Real.log (1 / 3) < 0)
  (h4: Real.log (1 / 3) / Real.log (1 / 2) > 1) :
  Real.log (1 / 3) / Real.log (1 / 2) > Real.sin 2 ∧ 
  Real.log (1 / 3) / Real.log (1 / 2) > Real.log 2 / Real.log (1 / 3) := by
  sorry

end find_largest_value_l1007_100753


namespace total_pencils_l1007_100729

def pencils_per_person : Nat := 15
def number_of_people : Nat := 5

theorem total_pencils : pencils_per_person * number_of_people = 75 := by
  sorry

end total_pencils_l1007_100729


namespace maximal_x2009_l1007_100783

theorem maximal_x2009 (x : ℕ → ℝ) 
    (h_seq : ∀ n, x n - 2 * x (n + 1) + x (n + 2) ≤ 0)
    (h_x0 : x 0 = 1)
    (h_x20 : x 20 = 9)
    (h_x200 : x 200 = 6) :
    x 2009 ≤ 6 :=
sorry

end maximal_x2009_l1007_100783


namespace total_pairs_sold_l1007_100790

theorem total_pairs_sold (H S : ℕ) 
    (soft_lens_cost hard_lens_cost : ℕ)
    (total_sales : ℕ)
    (h1 : soft_lens_cost = 150)
    (h2 : hard_lens_cost = 85)
    (h3 : S = H + 5)
    (h4 : soft_lens_cost * S + hard_lens_cost * H = total_sales)
    (h5 : total_sales = 1455) :
    H + S = 11 := 
  sorry

end total_pairs_sold_l1007_100790


namespace polynomial_evaluation_l1007_100736

theorem polynomial_evaluation (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 + 2005 = 2006 :=
sorry

end polynomial_evaluation_l1007_100736


namespace two_digit_numbers_condition_l1007_100771

theorem two_digit_numbers_condition :
  ∃ (x y : ℕ), x > y ∧ x < 100 ∧ y < 100 ∧ x - y = 56 ∧ (x^2 % 100) = (y^2 % 100) ∧
  ((x = 78 ∧ y = 22) ∨ (x = 22 ∧ y = 78)) :=
by sorry

end two_digit_numbers_condition_l1007_100771


namespace rectangle_section_properties_l1007_100764

structure Tetrahedron where
  edge_length : ℝ

structure RectangleSection where
  perimeter : ℝ
  area : ℝ

def regular_tetrahedron : Tetrahedron :=
  { edge_length := 1 }

theorem rectangle_section_properties :
  ∀ (rect : RectangleSection), 
  (∃ tetra : Tetrahedron, tetra = regular_tetrahedron) →
  (rect.perimeter = 2) ∧ (0 ≤ rect.area) ∧ (rect.area ≤ 1/4) :=
by
  -- Provide the hypothesis of the existence of such a tetrahedron and rectangular section
  sorry

end rectangle_section_properties_l1007_100764


namespace original_laborers_count_l1007_100739

theorem original_laborers_count (L : ℕ) (h1 : (L - 7) * 10 = L * 6) : L = 18 :=
sorry

end original_laborers_count_l1007_100739


namespace impossible_cube_configuration_l1007_100794

theorem impossible_cube_configuration :
  ∀ (cube: ℕ → ℕ) (n : ℕ), 
    (∀ n, 1 ≤ n ∧ n ≤ 27 → ∃ k, 1 ≤ k ∧ k ≤ 27 ∧ cube k = n) →
    (∀ n, 1 ≤ n ∧ n ≤ 27 → (cube 27 = 27 ∧ ∀ m, 1 ≤ m ∧ m ≤ 26 → cube m = 27 - m)) → 
    false :=
by
  intros cube n hcube htarget
  -- any detailed proof steps would go here, skipping with sorry
  sorry

end impossible_cube_configuration_l1007_100794


namespace min_m_n_sum_divisible_by_27_l1007_100769

theorem min_m_n_sum_divisible_by_27 (m n : ℕ) (h : 180 * m * (n - 2) % 27 = 0) : m + n = 6 :=
sorry

end min_m_n_sum_divisible_by_27_l1007_100769


namespace weight_of_dog_l1007_100767

theorem weight_of_dog (k r d : ℕ) (h1 : k + r + d = 30) (h2 : k + r = 2 * d) (h3 : k + d = r) : d = 10 :=
by
  sorry

end weight_of_dog_l1007_100767
