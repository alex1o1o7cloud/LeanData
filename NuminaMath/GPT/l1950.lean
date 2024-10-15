import Mathlib

namespace NUMINAMATH_GPT_Isabella_redeem_day_l1950_195030

def is_coupon_day_closed_sunday (start_day : ℕ) (num_coupons : ℕ) (cycle_days : ℕ) : Prop :=
  ∃ n, n < num_coupons ∧ (start_day + n * cycle_days) % 7 = 0

theorem Isabella_redeem_day: 
  ∀ (day : ℕ), day ≡ 1 [MOD 7]
  → ¬ is_coupon_day_closed_sunday day 6 11 :=
by
  intro day h_mod
  simp [is_coupon_day_closed_sunday]
  sorry

end NUMINAMATH_GPT_Isabella_redeem_day_l1950_195030


namespace NUMINAMATH_GPT_find_B_squared_l1950_195052

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 31 + 85 / x

theorem find_B_squared :
  let x1 := (Real.sqrt 31 + Real.sqrt 371) / 2
  let x2 := (Real.sqrt 31 - Real.sqrt 371) / 2
  let B := |x1| + |x2|
  B^2 = 371 :=
by
  sorry

end NUMINAMATH_GPT_find_B_squared_l1950_195052


namespace NUMINAMATH_GPT_pen_sales_average_l1950_195037

theorem pen_sales_average :
  ∃ d : ℕ, (48 = (96 + 44 * d) / (d + 1)) → d = 12 :=
by
  sorry

end NUMINAMATH_GPT_pen_sales_average_l1950_195037


namespace NUMINAMATH_GPT_emails_left_in_inbox_l1950_195034

-- Define the initial conditions and operations
def initial_emails : ℕ := 600

def move_half_to_trash (emails : ℕ) : ℕ := emails / 2
def move_40_percent_to_work (emails : ℕ) : ℕ := emails - (emails * 40 / 100)
def move_25_percent_to_personal (emails : ℕ) : ℕ := emails - (emails * 25 / 100)
def move_10_percent_to_miscellaneous (emails : ℕ) : ℕ := emails - (emails * 10 / 100)
def filter_30_percent_to_subfolders (emails : ℕ) : ℕ := emails - (emails * 30 / 100)
def archive_20_percent (emails : ℕ) : ℕ := emails - (emails * 20 / 100)

-- Statement we need to prove
theorem emails_left_in_inbox : 
  archive_20_percent
    (filter_30_percent_to_subfolders
      (move_10_percent_to_miscellaneous
        (move_25_percent_to_personal
          (move_40_percent_to_work
            (move_half_to_trash initial_emails))))) = 69 := 
by sorry

end NUMINAMATH_GPT_emails_left_in_inbox_l1950_195034


namespace NUMINAMATH_GPT_original_selling_price_l1950_195080

variable (P : ℝ)

def SP1 := 1.10 * P
def P_new := 0.90 * P
def SP2 := 1.17 * P
def price_diff := SP2 - SP1

theorem original_selling_price : price_diff = 49 → SP1 = 770 :=
by
  sorry

end NUMINAMATH_GPT_original_selling_price_l1950_195080


namespace NUMINAMATH_GPT_segment_length_R_R_l1950_195078

theorem segment_length_R_R' :
  let R := (-4, 1)
  let R' := (-4, -1)
  let distance : ℝ := Real.sqrt ((R'.1 - R.1)^2 + (R'.2 - R.2)^2)
  distance = 2 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_R_R_l1950_195078


namespace NUMINAMATH_GPT_smallest_possible_value_l1950_195067

theorem smallest_possible_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (h : (1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 15) : x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_possible_value_l1950_195067


namespace NUMINAMATH_GPT_positive_difference_sum_of_squares_l1950_195013

-- Given definitions
def sum_of_squares_even (n : ℕ) : ℕ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6

def sum_of_squares_odd (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- The explicit values for this problem
def sum_of_squares_first_25_even := sum_of_squares_even 25
def sum_of_squares_first_20_odd := sum_of_squares_odd 20

-- The required proof statement
theorem positive_difference_sum_of_squares : 
  (sum_of_squares_first_25_even - sum_of_squares_first_20_odd) = 19230 := by
  sorry

end NUMINAMATH_GPT_positive_difference_sum_of_squares_l1950_195013


namespace NUMINAMATH_GPT_sequence_inequality_l1950_195066

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₇ : a 7 = 0) :
  ∃ k : ℕ, k ≤ 5 ∧ a k + a (k + 2) ≤ a (k + 1) * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_sequence_inequality_l1950_195066


namespace NUMINAMATH_GPT_koala_fiber_intake_l1950_195093

theorem koala_fiber_intake (x : ℝ) (h : 0.30 * x = 12) : x = 40 := 
sorry

end NUMINAMATH_GPT_koala_fiber_intake_l1950_195093


namespace NUMINAMATH_GPT_inequality_abc_l1950_195011

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_abc_l1950_195011


namespace NUMINAMATH_GPT_sequence_general_term_l1950_195079

theorem sequence_general_term (a : ℕ → ℚ) (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n+1) = (n * a n + 2 * (n+1)^2) / (n+2)) :
  ∀ n : ℕ, a n = (1 / 2 : ℚ) * n * (n + 1) := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1950_195079


namespace NUMINAMATH_GPT_total_distance_correct_l1950_195008

def day1_distance : ℕ := (5 * 4) + (3 * 2) + (4 * 3)
def day2_distance : ℕ := (6 * 3) + (2 * 1) + (6 * 3) + (3 * 4)
def day3_distance : ℕ := (4 * 2) + (2 * 1) + (7 * 3) + (5 * 2)

def total_distance : ℕ := day1_distance + day2_distance + day3_distance

theorem total_distance_correct :
  total_distance = 129 := by
  sorry

end NUMINAMATH_GPT_total_distance_correct_l1950_195008


namespace NUMINAMATH_GPT_area_of_union_of_rectangle_and_circle_l1950_195057

theorem area_of_union_of_rectangle_and_circle :
  let width := 8
  let length := 12
  let radius := 12
  let A_rectangle := length * width
  let A_circle := Real.pi * radius ^ 2
  let A_overlap := (1 / 4) * A_circle
  A_rectangle + A_circle - A_overlap = 96 + 108 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_union_of_rectangle_and_circle_l1950_195057


namespace NUMINAMATH_GPT_total_pencils_correct_l1950_195070

def pencils_in_drawer : ℕ := 43
def pencils_on_desk_originally : ℕ := 19
def pencils_added_by_dan : ℕ := 16
def total_pencils : ℕ := pencils_in_drawer + pencils_on_desk_originally + pencils_added_by_dan

theorem total_pencils_correct : total_pencils = 78 := by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l1950_195070


namespace NUMINAMATH_GPT_triple_f_of_3_l1950_195041

def f (x : ℤ) : ℤ := -3 * x + 5

theorem triple_f_of_3 : f (f (f 3)) = -46 := by
  sorry

end NUMINAMATH_GPT_triple_f_of_3_l1950_195041


namespace NUMINAMATH_GPT_sum_of_first_three_terms_is_zero_l1950_195045

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_three_terms_is_zero_l1950_195045


namespace NUMINAMATH_GPT_Lesha_received_11_gifts_l1950_195085

theorem Lesha_received_11_gifts (x : ℕ) 
    (h1 : x < 100) 
    (h2 : x % 2 = 0) 
    (h3 : x % 5 = 0) 
    (h4 : x % 7 = 0) :
    x - (x / 2 + x / 5 + x / 7) = 11 :=
by {
    sorry
}

end NUMINAMATH_GPT_Lesha_received_11_gifts_l1950_195085


namespace NUMINAMATH_GPT_union_of_A_and_B_l1950_195010

def A : Set ℤ := {-1, 0, 2}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1950_195010


namespace NUMINAMATH_GPT_conic_section_is_ellipse_l1950_195084

theorem conic_section_is_ellipse :
  (∃ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0) ∧
  ∀ (x y : ℝ), 3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0 →
    ((x - 2)^2 / (20 / 3) + (y - 2)^2 / 20 = 1) :=
sorry

end NUMINAMATH_GPT_conic_section_is_ellipse_l1950_195084


namespace NUMINAMATH_GPT_jessica_initial_money_l1950_195042

def amount_spent : ℝ := 10.22
def amount_left : ℝ := 1.51
def initial_amount : ℝ := 11.73

theorem jessica_initial_money :
  amount_spent + amount_left = initial_amount := 
  by
    sorry

end NUMINAMATH_GPT_jessica_initial_money_l1950_195042


namespace NUMINAMATH_GPT_john_recreation_percent_l1950_195014

theorem john_recreation_percent (W : ℝ) (P : ℝ) (H1 : 0 ≤ P ∧ P ≤ 1) (H2 : 0 ≤ W) (H3 : 0.15 * W = 0.50 * (P * W)) :
  P = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_john_recreation_percent_l1950_195014


namespace NUMINAMATH_GPT_factorization_correct_l1950_195029

-- Defining the expressions
def expr1 (x : ℝ) : ℝ := 4 * x^2 + 4 * x
def expr2 (x : ℝ) : ℝ := 4 * x * (x + 1)

-- Theorem statement: Prove that expr1 and expr2 are equivalent
theorem factorization_correct (x : ℝ) : expr1 x = expr2 x :=
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l1950_195029


namespace NUMINAMATH_GPT_calculate_sum_of_triangles_l1950_195005

def operation_triangle (a b c : Int) : Int :=
  a * b - c 

theorem calculate_sum_of_triangles :
  operation_triangle 3 4 5 + operation_triangle 1 2 4 + operation_triangle 2 5 6 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_sum_of_triangles_l1950_195005


namespace NUMINAMATH_GPT_gcd_1729_1768_l1950_195063

theorem gcd_1729_1768 : Int.gcd 1729 1768 = 13 := by
  sorry

end NUMINAMATH_GPT_gcd_1729_1768_l1950_195063


namespace NUMINAMATH_GPT_lengthDE_is_correct_l1950_195096

noncomputable def triangleBase : ℝ := 12

noncomputable def triangleArea (h : ℝ) : ℝ := (1 / 2) * triangleBase * h

noncomputable def projectedArea (h : ℝ) : ℝ := 0.16 * triangleArea h

noncomputable def lengthDE (h : ℝ) : ℝ := 0.4 * triangleBase

theorem lengthDE_is_correct (h : ℝ) :
  lengthDE h = 4.8 :=
by
  simp [lengthDE, triangleBase, triangleArea, projectedArea]
  sorry

end NUMINAMATH_GPT_lengthDE_is_correct_l1950_195096


namespace NUMINAMATH_GPT_dollar_function_twice_l1950_195095

noncomputable def f (N : ℝ) : ℝ := 0.4 * N + 2

theorem dollar_function_twice (N : ℝ) (h : N = 30) : (f ∘ f) N = 5 := 
by
  sorry

end NUMINAMATH_GPT_dollar_function_twice_l1950_195095


namespace NUMINAMATH_GPT_not_suitable_for_storing_l1950_195090

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

end NUMINAMATH_GPT_not_suitable_for_storing_l1950_195090


namespace NUMINAMATH_GPT_greater_number_l1950_195002

theorem greater_number (a b : ℕ) (h1 : a + b = 36) (h2 : a - b = 8) : a = 22 :=
by
  sorry

end NUMINAMATH_GPT_greater_number_l1950_195002


namespace NUMINAMATH_GPT_find_n_l1950_195065

-- Definitions for conditions given in the problem
def a₂ (a : ℕ → ℕ) : Prop := a 2 = 3
def consecutive_sum (S : ℕ → ℕ) (n : ℕ) : Prop := ∀ n > 3, S n - S (n - 3) = 51
def total_sum (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 100

-- The main proof problem
theorem find_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h₁ : a₂ a) (h₂ : consecutive_sum S n) (h₃ : total_sum S n) : n = 10 :=
sorry

end NUMINAMATH_GPT_find_n_l1950_195065


namespace NUMINAMATH_GPT_fraction_never_simplifiable_l1950_195040

theorem fraction_never_simplifiable (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end NUMINAMATH_GPT_fraction_never_simplifiable_l1950_195040


namespace NUMINAMATH_GPT_amn_div_l1950_195000

theorem amn_div (a m n : ℕ) (a_pos : a > 1) (h : a > 1 ∧ (a^m + 1) ∣ (a^n + 1)) : m ∣ n :=
by sorry

end NUMINAMATH_GPT_amn_div_l1950_195000


namespace NUMINAMATH_GPT_pq_problem_l1950_195074

theorem pq_problem
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x - 7) * (2 * x + 11) = x^2 - 19 * x +  60)
  (h2 : p * q = 7 * (-9))
  (h3 : 7 + (-9) = -16):
  (p - 2) * (q - 2) = -55 :=
by
  sorry

end NUMINAMATH_GPT_pq_problem_l1950_195074


namespace NUMINAMATH_GPT_largest_difference_l1950_195054

def A : ℕ := 3 * 2005^2006
def B : ℕ := 2005^2006
def C : ℕ := 2004 * 2005^2005
def D : ℕ := 3 * 2005^2005
def E : ℕ := 2005^2005
def F : ℕ := 2005^2004

theorem largest_difference : (A - B > B - C) ∧ (A - B > C - D) ∧ (A - B > D - E) ∧ (A - B > E - F) :=
by
  sorry  -- Proof is omitted as per instructions.

end NUMINAMATH_GPT_largest_difference_l1950_195054


namespace NUMINAMATH_GPT_parabola_y_values_order_l1950_195025

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_parabola_y_values_order_l1950_195025


namespace NUMINAMATH_GPT_find_A_l1950_195004

def clubsuit (A B : ℤ) : ℤ := 4 * A + 2 * B + 6

theorem find_A : ∃ A : ℤ, clubsuit A 6 = 70 → A = 13 := 
by
  sorry

end NUMINAMATH_GPT_find_A_l1950_195004


namespace NUMINAMATH_GPT_distance_PF_l1950_195076

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the point P on the parabola with x-coordinate 4
def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

-- Prove the distance |PF| for given conditions
theorem distance_PF
  (hP : ∃ y : ℝ, parabola 4 y)
  (hF : focus = (2, 0)) :
  ∃ y : ℝ, y^2 = 8 * 4 ∧ abs (4 - 2) + abs y = 6 := 
by
  sorry

end NUMINAMATH_GPT_distance_PF_l1950_195076


namespace NUMINAMATH_GPT_inf_solutions_integers_l1950_195009

theorem inf_solutions_integers (x y z : ℕ) : ∃ (n : ℕ), ∀ n > 0, (x = 2^(32 + 72 * n)) ∧ (y = 2^(28 + 63 * n)) ∧ (z = 2^(25 + 56 * n)) → x^7 + y^8 = z^9 :=
by {
  sorry
}

end NUMINAMATH_GPT_inf_solutions_integers_l1950_195009


namespace NUMINAMATH_GPT_ceil_of_fractional_square_l1950_195024

theorem ceil_of_fractional_square :
  (Int.ceil ((- (7/4) + 1/4) ^ 2) = 3) :=
by
  sorry

end NUMINAMATH_GPT_ceil_of_fractional_square_l1950_195024


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1950_195075

theorem solution_set_of_inequality (x : ℝ) : (x^2 + 4*x - 5 < 0) ↔ (-5 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1950_195075


namespace NUMINAMATH_GPT_perpendicular_line_slope_l1950_195061

theorem perpendicular_line_slope (a : ℝ) :
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  k_MN * (-a / 2) = -1 → a = 1 :=
by
  intros M N k_MN H
  let M := (0, -1)
  let N := (2, 3)
  let k_MN := (N.2 - M.2) / (N.1 - M.1)
  sorry

end NUMINAMATH_GPT_perpendicular_line_slope_l1950_195061


namespace NUMINAMATH_GPT_time_to_drain_tank_l1950_195048

theorem time_to_drain_tank (P L: ℝ) (hP : P = 1/3) (h_combined : P - L = 2/7) : 1 / L = 21 :=
by
  -- Proof omitted. Use the conditions given to show that 1 / L = 21.
  sorry

end NUMINAMATH_GPT_time_to_drain_tank_l1950_195048


namespace NUMINAMATH_GPT_sticks_problem_solution_l1950_195046

theorem sticks_problem_solution :
  ∃ n : ℕ, n > 0 ∧ 1012 = 2 * n * (n + 1) ∧ 1012 > 1000 ∧ 
           1012 % 3 = 1 ∧ 1012 % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sticks_problem_solution_l1950_195046


namespace NUMINAMATH_GPT_total_goals_is_15_l1950_195031

-- Define the conditions as variables
def KickersFirstPeriodGoals : ℕ := 2
def KickersSecondPeriodGoals : ℕ := 2 * KickersFirstPeriodGoals
def SpidersFirstPeriodGoals : ℕ := KickersFirstPeriodGoals / 2
def SpidersSecondPeriodGoals : ℕ := 2 * KickersSecondPeriodGoals

-- Define total goals by each team
def TotalKickersGoals : ℕ := KickersFirstPeriodGoals + KickersSecondPeriodGoals
def TotalSpidersGoals : ℕ := SpidersFirstPeriodGoals + SpidersSecondPeriodGoals

-- Define total goals by both teams
def TotalGoals : ℕ := TotalKickersGoals + TotalSpidersGoals

-- Prove the statement
theorem total_goals_is_15 : TotalGoals = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_goals_is_15_l1950_195031


namespace NUMINAMATH_GPT_correct_substitution_l1950_195097

theorem correct_substitution (x y : ℤ) (h1 : x = 3 * y - 1) (h2 : x - 2 * y = 4) :
  3 * y - 1 - 2 * y = 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_substitution_l1950_195097


namespace NUMINAMATH_GPT_billy_unknown_lap_time_l1950_195064

theorem billy_unknown_lap_time :
  ∀ (time_first_5_laps time_next_3_laps time_last_lap time_margaret total_time_billy : ℝ) (lap_time_unknown : ℝ),
    time_first_5_laps = 2 ∧
    time_next_3_laps = 4 ∧
    time_last_lap = 2.5 ∧
    time_margaret = 10 ∧
    total_time_billy = time_margaret - 0.5 →
    (time_first_5_laps + time_next_3_laps + time_last_lap + lap_time_unknown = total_time_billy) →
    lap_time_unknown = 1 :=
by
  sorry

end NUMINAMATH_GPT_billy_unknown_lap_time_l1950_195064


namespace NUMINAMATH_GPT_problem1_problem2_l1950_195094

-- Problem (1)
theorem problem1 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : ∀ n, a n = a (n + 1) + 3) : a 10 = -23 :=
by {
  sorry
}

-- Problem (2)
theorem problem2 (a : ℕ → ℚ) (h1 : a 6 = (1 / 4)) (h2 : ∃ d : ℚ, ∀ n, 1 / a n = 1 / a 1 + (n - 1) * d) : 
  ∀ n, a n = (4 / (3 * n - 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l1950_195094


namespace NUMINAMATH_GPT_gcd_2024_2048_l1950_195060

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end NUMINAMATH_GPT_gcd_2024_2048_l1950_195060


namespace NUMINAMATH_GPT_platform_length_l1950_195071

theorem platform_length (train_length : ℕ) (tree_cross_time : ℕ) (platform_cross_time : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 1200)
  (h_tree_cross_time : tree_cross_time = 120)
  (h_platform_cross_time : platform_cross_time = 160)
  (h_speed_calculation : (train_length / tree_cross_time = 10))
  : (train_length + platform_length) / 10 = platform_cross_time → platform_length = 400 :=
sorry

end NUMINAMATH_GPT_platform_length_l1950_195071


namespace NUMINAMATH_GPT_number_of_matches_in_first_set_l1950_195073

theorem number_of_matches_in_first_set
  (avg_next_13_matches : ℕ := 15)
  (total_matches : ℕ := 35)
  (avg_all_matches : ℚ := 23.17142857142857)
  (x : ℕ := total_matches - 13) :
  x = 22 := by
  sorry

end NUMINAMATH_GPT_number_of_matches_in_first_set_l1950_195073


namespace NUMINAMATH_GPT_explicit_expression_l1950_195043

variable {α : Type*} [LinearOrder α] {f : α → α}

/-- Given that the function satisfies a specific condition, prove the function's explicit expression. -/
theorem explicit_expression (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) : 
  ∀ x, f x = 3 * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_explicit_expression_l1950_195043


namespace NUMINAMATH_GPT_tan_15_degree_identity_l1950_195003

theorem tan_15_degree_identity : (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_tan_15_degree_identity_l1950_195003


namespace NUMINAMATH_GPT_total_waiting_time_l1950_195023

def t1 : ℕ := 20
def t2 : ℕ := 4 * t1 + 14
def T : ℕ := t1 + t2

theorem total_waiting_time : T = 114 :=
by {
  -- Preliminary calculations and justification would go here
  sorry
}

end NUMINAMATH_GPT_total_waiting_time_l1950_195023


namespace NUMINAMATH_GPT_units_digit_of_expression_l1950_195056

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def expression := (20 * 21 * 22 * 23 * 24 * 25) / 1000

theorem units_digit_of_expression : units_digit (expression) = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l1950_195056


namespace NUMINAMATH_GPT_triangle_equilateral_l1950_195062

variable {a b c : ℝ}

theorem triangle_equilateral (h : a^2 + 2 * b^2 = 2 * b * (a + c) - c^2) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_GPT_triangle_equilateral_l1950_195062


namespace NUMINAMATH_GPT_largest_band_members_l1950_195032

theorem largest_band_members 
  (r x m : ℕ) 
  (h1 : (r * x + 3 = m)) 
  (h2 : ((r - 3) * (x + 1) = m))
  (h3 : m < 100) : 
  m = 75 :=
sorry

end NUMINAMATH_GPT_largest_band_members_l1950_195032


namespace NUMINAMATH_GPT_train_B_departure_time_l1950_195021

def distance : ℕ := 65
def speed_A : ℕ := 20
def speed_B : ℕ := 25
def departure_A := 7
def meeting_time := 9

theorem train_B_departure_time : ∀ (d : ℕ) (vA : ℕ) (vB : ℕ) (tA : ℕ) (m : ℕ), 
  d = 65 → vA = 20 → vB = 25 → tA = 7 → m = 9 → ((9 - (m - tA + (d - (2 * vA)) / vB)) = 1) → 
  8 = ((9 - (meeting_time - departure_A + (distance - (2 * speed_A)) / speed_B))) := 
  by {
    sorry
  }

end NUMINAMATH_GPT_train_B_departure_time_l1950_195021


namespace NUMINAMATH_GPT_smallest_integer_sum_to_2020_l1950_195053

theorem smallest_integer_sum_to_2020 :
  ∃ B : ℤ, (∃ (n : ℤ), (B * (B + 1) / 2) + ((n * (n + 1)) / 2) = 2020) ∧ (∀ C : ℤ, (∃ (m : ℤ), (C * (C + 1) / 2) + ((m * (m + 1)) / 2) = 2020) → B ≤ C) ∧ B = -2019 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_sum_to_2020_l1950_195053


namespace NUMINAMATH_GPT_range_of_a_l1950_195069

variables {x a : ℝ}

def p (x : ℝ) : Prop := (x - 5) / (x - 3) ≥ 2
def q (x a : ℝ) : Prop := x ^ 2 - a * x ≤ x - a

theorem range_of_a (h : ¬(∃ x, p x) → ¬(∃ x, q x a)) :
  1 ≤ a ∧ a < 3 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1950_195069


namespace NUMINAMATH_GPT_total_accidents_l1950_195068

theorem total_accidents :
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  accidentsA + accidentsB + accidentsC = 3730 :=
by
  let accidentsA := (75 / 100) * 2500
  let accidentsB := (50 / 80) * 1600
  let accidentsC := (90 / 200) * 1900
  sorry

end NUMINAMATH_GPT_total_accidents_l1950_195068


namespace NUMINAMATH_GPT_isosceles_triangle_angles_l1950_195012

theorem isosceles_triangle_angles (y : ℝ) (h : y > 0) :
  let P := y
  let R := 5 * y
  let Q := R
  P + Q + R = 180 → Q = 81.82 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angles_l1950_195012


namespace NUMINAMATH_GPT_find_value_of_a_l1950_195017

theorem find_value_of_a (a : ℝ) (h : a^3 = 21 * 25 * 315 * 7) : a = 105 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1950_195017


namespace NUMINAMATH_GPT_trig_proof_l1950_195022

theorem trig_proof (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end NUMINAMATH_GPT_trig_proof_l1950_195022


namespace NUMINAMATH_GPT_eval_infinite_series_eq_4_l1950_195001

open BigOperators

noncomputable def infinite_series_sum : ℝ :=
  ∑' k, (k^2) / (3^k)

theorem eval_infinite_series_eq_4 : infinite_series_sum = 4 := 
  sorry

end NUMINAMATH_GPT_eval_infinite_series_eq_4_l1950_195001


namespace NUMINAMATH_GPT_total_time_for_12000_dolls_l1950_195028

noncomputable def total_combined_machine_operation_time (num_dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) (time_per_doll time_per_accessory : ℕ) : ℕ :=
  let total_accessories_per_doll := shoes_per_doll + bags_per_doll + cosmetics_per_doll + hats_per_doll
  let total_accessories := num_dolls * total_accessories_per_doll
  let time_for_dolls := num_dolls * time_per_doll
  let time_for_accessories := total_accessories * time_per_accessory
  time_for_dolls + time_for_accessories

theorem total_time_for_12000_dolls (h1 : ∀ (x : ℕ), x = 12000) (h2 : ∀ (x : ℕ), x = 2) (h3 : ∀ (x : ℕ), x = 3) (h4 : ∀ (x : ℕ), x = 1) (h5 : ∀ (x : ℕ), x = 5) (h6 : ∀ (x : ℕ), x = 45) (h7 : ∀ (x : ℕ), x = 10) :
  total_combined_machine_operation_time 12000 2 3 1 5 45 10 = 1860000 := by 
  sorry

end NUMINAMATH_GPT_total_time_for_12000_dolls_l1950_195028


namespace NUMINAMATH_GPT_area_of_PQRS_l1950_195081

noncomputable def length_square_EFGH := 6
noncomputable def height_equilateral_triangle := 3 * Real.sqrt 3
noncomputable def diagonal_PQRS := length_square_EFGH + 2 * height_equilateral_triangle
noncomputable def area_PQRS := (1 / 2) * (diagonal_PQRS * diagonal_PQRS)

theorem area_of_PQRS :
  (area_PQRS = 72 + 36 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_area_of_PQRS_l1950_195081


namespace NUMINAMATH_GPT_cyclist_speed_l1950_195051

theorem cyclist_speed:
  ∀ (c : ℝ), 
  ∀ (hiker_speed : ℝ), 
  (hiker_speed = 4) → 
  (4 * (5 / 60) + 4 * (25 / 60) = c * (5 / 60)) → 
  c = 24 := 
by
  intros c hiker_speed hiker_speed_def distance_eq
  sorry

end NUMINAMATH_GPT_cyclist_speed_l1950_195051


namespace NUMINAMATH_GPT_flagpole_height_l1950_195088

theorem flagpole_height (x : ℝ) (h1 : (x + 2)^2 = x^2 + 6^2) : x = 8 := 
by 
  sorry

end NUMINAMATH_GPT_flagpole_height_l1950_195088


namespace NUMINAMATH_GPT_sum_of_ages_five_years_ago_l1950_195027

-- Definitions from the conditions
variables (A B : ℕ) -- Angela's current age and Beth's current age

-- Conditions
def angela_is_four_times_as_old_as_beth := A = 4 * B
def angela_will_be_44_in_five_years := A + 5 = 44

-- Theorem statement to prove the sum of their ages five years ago
theorem sum_of_ages_five_years_ago (h1 : angela_is_four_times_as_old_as_beth A B) (h2 : angela_will_be_44_in_five_years A) : 
  (A - 5) + (B - 5) = 39 :=
by sorry

end NUMINAMATH_GPT_sum_of_ages_five_years_ago_l1950_195027


namespace NUMINAMATH_GPT_inscribed_circle_radii_rel_l1950_195058

theorem inscribed_circle_radii_rel {a b c r r1 r2 : ℝ} :
  (a^2 + b^2 = c^2) ∧
  (r1 = (a / c) * r) ∧
  (r2 = (b / c) * r) →
  r^2 = r1^2 + r2^2 :=
by 
  sorry

end NUMINAMATH_GPT_inscribed_circle_radii_rel_l1950_195058


namespace NUMINAMATH_GPT_run_to_cafe_time_l1950_195018

theorem run_to_cafe_time (h_speed_const : ∀ t1 t2 d1 d2 : ℝ, (t1 / d1) = (t2 / d2))
  (h_store_time : 24 = 3 * (24 / 3))
  (h_cafe_halfway : ∀ d : ℝ, d = 1.5) :
  ∃ t : ℝ, t = 12 :=
by
  sorry

end NUMINAMATH_GPT_run_to_cafe_time_l1950_195018


namespace NUMINAMATH_GPT_prob_four_vertical_faces_same_color_l1950_195033

noncomputable def painted_cube_probability : ℚ :=
  let total_arrangements := 3^6
  let suitable_arrangements := 3 + 18 + 6
  suitable_arrangements / total_arrangements

theorem prob_four_vertical_faces_same_color : 
  painted_cube_probability = 1 / 27 := by
  sorry

end NUMINAMATH_GPT_prob_four_vertical_faces_same_color_l1950_195033


namespace NUMINAMATH_GPT_harmony_numbers_with_first_digit_2_count_l1950_195006

def is_harmony_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  (1000 ≤ n ∧ n < 10000) ∧ (a + b + c + d = 6)

noncomputable def count_harmony_numbers_with_first_digit_2 : ℕ :=
  Nat.card { n : ℕ // is_harmony_number n ∧ n / 1000 = 2 }

theorem harmony_numbers_with_first_digit_2_count :
  count_harmony_numbers_with_first_digit_2 = 15 :=
sorry

end NUMINAMATH_GPT_harmony_numbers_with_first_digit_2_count_l1950_195006


namespace NUMINAMATH_GPT_valid_license_plates_count_l1950_195091

/--
The problem is to prove that the total number of valid license plates under the given format is equal to 45,697,600.
The given conditions are:
1. A valid license plate in Xanadu consists of three letters followed by two digits, and then one more letter at the end.
2. There are 26 choices of letters for each letter spot.
3. There are 10 choices of digits for each digit spot.

We need to conclude that the number of possible license plates is:
26^4 * 10^2 = 45,697,600.
-/

def num_valid_license_plates : Nat :=
  let letter_choices := 26
  let digit_choices := 10
  let total_choices := letter_choices ^ 3 * digit_choices ^ 2 * letter_choices
  total_choices

theorem valid_license_plates_count : num_valid_license_plates = 45697600 := by
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l1950_195091


namespace NUMINAMATH_GPT_max_cos_x_l1950_195039

theorem max_cos_x (x y : ℝ) (h : Real.cos (x - y) = Real.cos x - Real.cos y) : 
  ∃ M, (∀ x, Real.cos x <= M) ∧ M = 1 := 
sorry

end NUMINAMATH_GPT_max_cos_x_l1950_195039


namespace NUMINAMATH_GPT_compute_expression_in_terms_of_k_l1950_195089

-- Define the main theorem to be proven, with all conditions directly translated to Lean statements.
theorem compute_expression_in_terms_of_k
  (x y : ℝ)
  (h : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
    (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = ((k - 2)^2 * (k + 2)^2) / (4 * k * (k^2 + 4)) :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_in_terms_of_k_l1950_195089


namespace NUMINAMATH_GPT_car_distance_kilometers_l1950_195082

theorem car_distance_kilometers (d_amar : ℝ) (d_car : ℝ) (ratio : ℝ) (total_d_amar : ℝ) :
  d_amar = 24 ->
  d_car = 60 ->
  ratio = 2 / 5 ->
  total_d_amar = 880 ->
  (d_car / d_amar) = 5 / 2 ->
  (total_d_amar * 5 / 2) / 1000 = 2.2 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_car_distance_kilometers_l1950_195082


namespace NUMINAMATH_GPT_original_number_increased_by_45_percent_is_870_l1950_195099

theorem original_number_increased_by_45_percent_is_870 (x : ℝ) (h : x * 1.45 = 870) : x = 870 / 1.45 :=
by sorry

end NUMINAMATH_GPT_original_number_increased_by_45_percent_is_870_l1950_195099


namespace NUMINAMATH_GPT_find_pairs_s_t_l1950_195098

theorem find_pairs_s_t (n : ℤ) (hn : n > 1) : 
  ∃ s t : ℤ, (
    (∀ x : ℝ, x ^ n + s * x = 2007 ∧ x ^ n + t * x = 2008 → 
     (s, t) = (2006, 2007) ∨ (s, t) = (-2008, -2009) ∨ (s, t) = (-2006, -2007))
  ) :=
sorry

end NUMINAMATH_GPT_find_pairs_s_t_l1950_195098


namespace NUMINAMATH_GPT_arithmetic_seq_a3_a9_zero_l1950_195038

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_11_zero (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 0

theorem arithmetic_seq_a3_a9_zero (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_11_zero a) :
  a 3 + a 9 = 0 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_a3_a9_zero_l1950_195038


namespace NUMINAMATH_GPT_polynomial_rewrite_l1950_195087

theorem polynomial_rewrite :
  ∃ (a b c d e f : ℤ), 
  (2401 * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f)) ∧
  (a + b + c + d + e + f = 274) :=
sorry

end NUMINAMATH_GPT_polynomial_rewrite_l1950_195087


namespace NUMINAMATH_GPT_total_amount_spent_l1950_195019

theorem total_amount_spent (T : ℝ) (h1 : 5000 + 200 + 0.30 * T = T) : 
  T = 7428.57 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l1950_195019


namespace NUMINAMATH_GPT_number_of_factors_in_224_l1950_195072

def smallest_is_half_largest (n1 n2 : ℕ) : Prop :=
  n1 * 2 = n2

theorem number_of_factors_in_224 :
  ∃ n1 n2 n3 : ℕ, n1 * n2 * n3 = 224 ∧ smallest_is_half_largest (min n1 (min n2 n3)) (max n1 (max n2 n3)) ∧
    (if h : n1 < n2 ∧ n1 < n3 then
      if h2 : n2 < n3 then 
        smallest_is_half_largest n1 n3 
        else 
        smallest_is_half_largest n1 n2 
    else if h : n2 < n1 ∧ n2 < n3 then 
      if h2 : n1 < n3 then 
        smallest_is_half_largest n2 n3 
        else 
        smallest_is_half_largest n2 n1 
    else 
      if h2 : n1 < n2 then 
        smallest_is_half_largest n3 n2 
        else 
        smallest_is_half_largest n3 n1) = true ∧ 
    (if h : n1 < n2 ∧ n1 < n3 then
       if h2 : n2 < n3 then 
         n1 * n2 * n3 
         else 
         n1 * n3 * n2 
     else if h : n2 < n1 ∧ n2 < n3 then 
       if h2 : n1 < n3 then 
         n2 * n1 * n3
         else 
         n2 * n3 * n1 
     else 
       if h2 : n1 < n2 then 
         n3 * n1 * n2 
         else 
         n3 * n2 * n1) = 224 := sorry

end NUMINAMATH_GPT_number_of_factors_in_224_l1950_195072


namespace NUMINAMATH_GPT_slips_numbers_exist_l1950_195083

theorem slips_numbers_exist (x y z : ℕ) (h₁ : x + y + z = 20) (h₂ : 5 * x + 3 * y = 46) : 
  (x = 4) ∧ (y = 10) ∧ (z = 6) :=
by {
  -- Technically, the actual proving steps should go here, but skipped due to 'sorry'
  sorry
}

end NUMINAMATH_GPT_slips_numbers_exist_l1950_195083


namespace NUMINAMATH_GPT_clients_number_l1950_195015

theorem clients_number (C : ℕ) (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ)
  (h1 : total_cars = 12)
  (h2 : cars_per_client = 4)
  (h3 : selections_per_car = 3)
  (h4 : C * cars_per_client = total_cars * selections_per_car) : C = 9 :=
by sorry

end NUMINAMATH_GPT_clients_number_l1950_195015


namespace NUMINAMATH_GPT_no_ordered_triples_l1950_195050

theorem no_ordered_triples (x y z : ℕ)
  (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  x * y * z + 2 * (x * y + y * z + z * x) ≠ 2 * (2 * (x * y + y * z + z * x)) + 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_no_ordered_triples_l1950_195050


namespace NUMINAMATH_GPT_percentage_of_number_l1950_195049

/-- 
  Given a certain percentage \( P \) of 600 is 90.
  If 30% of 50% of a number 4000 is 90,
  Then P equals to 15%.
-/
theorem percentage_of_number (P : ℝ) (h1 : (0.30 : ℝ) * (0.50 : ℝ) * 4000 = 600) (h2 : P * 600 = 90) :
  P = 0.15 :=
  sorry

end NUMINAMATH_GPT_percentage_of_number_l1950_195049


namespace NUMINAMATH_GPT_totalPoundsOfFoodConsumed_l1950_195016

def maxConsumptionPerGuest : ℝ := 2.5
def minNumberOfGuests : ℕ := 165

theorem totalPoundsOfFoodConsumed : 
    maxConsumptionPerGuest * (minNumberOfGuests : ℝ) = 412.5 := by
  sorry

end NUMINAMATH_GPT_totalPoundsOfFoodConsumed_l1950_195016


namespace NUMINAMATH_GPT_simplify_polynomial_l1950_195020

def p (x : ℝ) : ℝ := 3 * x^5 - x^4 + 2 * x^3 + 5 * x^2 - 3 * x + 7
def q (x : ℝ) : ℝ := -x^5 + 4 * x^4 + x^3 - 6 * x^2 + 5 * x - 4
def r (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^3 - x^2 - x + 2

theorem simplify_polynomial (x : ℝ) :
  (p x) + (q x) - (r x) = 6 * x^4 - x^3 + 3 * x + 1 :=
by sorry

end NUMINAMATH_GPT_simplify_polynomial_l1950_195020


namespace NUMINAMATH_GPT_find_mangoes_l1950_195047

def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 1165
def cost_per_kg_of_mangoes : ℕ := 55

theorem find_mangoes (m : ℕ) : cost_of_grapes + m * cost_per_kg_of_mangoes = total_amount_paid → m = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_mangoes_l1950_195047


namespace NUMINAMATH_GPT_difference_of_distances_l1950_195026

-- Definition of John's walking distance to school
def John_distance : ℝ := 0.7

-- Definition of Nina's walking distance to school
def Nina_distance : ℝ := 0.4

-- Assertion that the difference in walking distance is 0.3 miles
theorem difference_of_distances : (John_distance - Nina_distance) = 0.3 := 
by 
  sorry

end NUMINAMATH_GPT_difference_of_distances_l1950_195026


namespace NUMINAMATH_GPT_perimeter_of_monster_is_correct_l1950_195007

/-
  The problem is to prove that the perimeter of a shaded sector of a circle
  with radius 2 cm and a central angle of 120 degrees (where the mouth is a chord)
  is equal to (8 * π / 3 + 2 * sqrt 3) cm.
-/

noncomputable def perimeter_of_monster (r : ℝ) (theta_deg : ℝ) : ℝ :=
  let theta_rad := theta_deg * Real.pi / 180
  let chord_length := 2 * r * Real.sin (theta_rad / 2)
  let arc_length := (2 * (2 * Real.pi) * (240 / 360))
  arc_length + chord_length

theorem perimeter_of_monster_is_correct : perimeter_of_monster 2 120 = (8 * Real.pi / 3 + 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_monster_is_correct_l1950_195007


namespace NUMINAMATH_GPT_regular_polygon_sides_l1950_195059

theorem regular_polygon_sides 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : B = 3 * A)
  (h₃ : C = 6 * A) :
  ∃ (n : ℕ), n = 5 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1950_195059


namespace NUMINAMATH_GPT_planting_cost_l1950_195092

-- Define the costs of the individual items
def cost_of_flowers : ℝ := 9
def cost_of_clay_pot : ℝ := cost_of_flowers + 20
def cost_of_soil : ℝ := cost_of_flowers - 2
def cost_of_fertilizer : ℝ := cost_of_flowers + (0.5 * cost_of_flowers)
def cost_of_tools : ℝ := cost_of_clay_pot - (0.25 * cost_of_clay_pot)

-- Define the total cost
def total_cost : ℝ :=
  cost_of_flowers + cost_of_clay_pot + cost_of_soil + cost_of_fertilizer + cost_of_tools

-- The statement to prove
theorem planting_cost : total_cost = 80.25 :=
by
  sorry

end NUMINAMATH_GPT_planting_cost_l1950_195092


namespace NUMINAMATH_GPT_rocky_miles_total_l1950_195044

-- Defining the conditions
def m1 : ℕ := 4
def m2 : ℕ := 2 * m1
def m3 : ℕ := 3 * m2

-- The statement to be proven
theorem rocky_miles_total : m1 + m2 + m3 = 36 := by
  sorry

end NUMINAMATH_GPT_rocky_miles_total_l1950_195044


namespace NUMINAMATH_GPT_proof_problem_l1950_195036

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 2 * x + 1

theorem proof_problem : f (g 3) - g (f 3) = -5 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1950_195036


namespace NUMINAMATH_GPT_max_d_77733e_divisible_by_33_l1950_195086

open Int

theorem max_d_77733e_divisible_by_33 : ∃ d e : ℕ, 
  (7 * 100000 + d * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e) % 33 = 0 ∧ 
  (d ≤ 9) ∧ (e ≤ 9) ∧ 
  (∀ d' e', ((7 * 100000 + d' * 10000 + 7 * 1000 + 3 * 100 + 3 * 10 + e') % 33 = 0 ∧ d' ≤ 9 ∧ e' ≤ 9 → d' ≤ d)) 
  := ⟨6, 0, by sorry⟩

end NUMINAMATH_GPT_max_d_77733e_divisible_by_33_l1950_195086


namespace NUMINAMATH_GPT_germination_percentage_in_second_plot_l1950_195077

theorem germination_percentage_in_second_plot
     (seeds_first_plot : ℕ := 300)
     (seeds_second_plot : ℕ := 200)
     (germination_first_plot : ℕ := 75)
     (total_seeds : ℕ := 500)
     (germination_total : ℕ := 155)
     (x : ℕ := 40) :
  (x : ℕ) = (80 / 2) := by
  -- Provided conditions, skipping the proof part with sorry
  have h1 : 75 = 0.25 * 300 := sorry
  have h2 : 500 = 300 + 200 := sorry
  have h3 : 155 = 0.31 * 500 := sorry
  have h4 : 80 = 155 - 75 := sorry
  have h5 : x = (80 / 2) := sorry
  exact h5

end NUMINAMATH_GPT_germination_percentage_in_second_plot_l1950_195077


namespace NUMINAMATH_GPT_printer_time_l1950_195055

theorem printer_time (Tx : ℝ) 
  (h1 : ∀ (Ty Tz : ℝ), Ty = 10 → Tz = 20 → 1 / Ty + 1 / Tz = 3 / 20) 
  (h2 : ∀ (T_combined : ℝ), T_combined = 20 / 3 → Tx / T_combined = 2.4) :
  Tx = 16 := 
by 
  sorry

end NUMINAMATH_GPT_printer_time_l1950_195055


namespace NUMINAMATH_GPT_selection_methods_count_l1950_195035

theorem selection_methods_count
  (multiple_choice_questions : ℕ)
  (fill_in_the_blank_questions : ℕ)
  (h1 : multiple_choice_questions = 9)
  (h2 : fill_in_the_blank_questions = 3) :
  multiple_choice_questions + fill_in_the_blank_questions = 12 := by
  sorry

end NUMINAMATH_GPT_selection_methods_count_l1950_195035
