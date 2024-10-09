import Mathlib

namespace fraction_of_people_under_21_correct_l1815_181520

variable (P : ℕ) (frac_over_65 : ℚ) (num_under_21 : ℕ) (frac_under_21 : ℚ)

def total_people_in_range (P : ℕ) : Prop := 50 < P ∧ P < 100

def fraction_of_people_over_65 (frac_over_65 : ℚ) : Prop := frac_over_65 = 5/12

def number_of_people_under_21 (num_under_21 : ℕ) : Prop := num_under_21 = 36

def fraction_of_people_under_21 (frac_under_21 : ℚ) : Prop := frac_under_21 = 3/7

theorem fraction_of_people_under_21_correct :
  ∀ (P : ℕ),
  total_people_in_range P →
  fraction_of_people_over_65 (5 / 12) →
  number_of_people_under_21 36 →
  P = 84 →
  fraction_of_people_under_21 (36 / P) :=
by
  intros P h_range h_over_65 h_under_21 h_P
  sorry

end fraction_of_people_under_21_correct_l1815_181520


namespace range_of_first_term_in_geometric_sequence_l1815_181578

theorem range_of_first_term_in_geometric_sequence (q a₁ : ℝ)
  (h_q : |q| < 1)
  (h_sum : a₁ / (1 - q) = q) :
  -2 < a₁ ∧ a₁ ≤ 0.25 ∧ a₁ ≠ 0 :=
by
  sorry

end range_of_first_term_in_geometric_sequence_l1815_181578


namespace number_of_elements_in_union_l1815_181516

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem number_of_elements_in_union : ncard (A ∪ B) = 4 :=
by
  sorry

end number_of_elements_in_union_l1815_181516


namespace min_bdf_proof_exists_l1815_181536

noncomputable def minBDF (a b c d e f : ℕ) (A : ℕ) :=
  (A = 3 * a ∧ A = 4 * c ∧ A = 5 * e) →
  (a / b * c / d * e / f = A) →
  b * d * f = 60

theorem min_bdf_proof_exists :
  ∃ (a b c d e f A : ℕ), minBDF a b c d e f A :=
by
  sorry

end min_bdf_proof_exists_l1815_181536


namespace final_combined_price_correct_l1815_181575

theorem final_combined_price_correct :
  let i_p := 1000
  let d_1 := 0.10
  let d_2 := 0.20
  let t_1 := 0.08
  let t_2 := 0.06
  let s_p := 30
  let c_p := 50
  let t_a := 0.05
  let price_after_first_month := i_p * (1 - d_1) * (1 + t_1)
  let price_after_second_month := price_after_first_month * (1 - d_2) * (1 + t_2)
  let screen_protector_final := s_p * (1 + t_a)
  let case_final := c_p * (1 + t_a)
  price_after_second_month + screen_protector_final + case_final = 908.256 := by
  sorry  -- Proof not required

end final_combined_price_correct_l1815_181575


namespace john_average_increase_l1815_181551

theorem john_average_increase :
  let initial_scores := [92, 85, 91]
  let fourth_score := 95
  let initial_avg := (initial_scores.sum / initial_scores.length : ℚ)
  let new_avg := ((initial_scores.sum + fourth_score) / (initial_scores.length + 1) : ℚ)
  new_avg - initial_avg = 1.42 := 
by 
  sorry

end john_average_increase_l1815_181551


namespace parabola_directrix_l1815_181528

theorem parabola_directrix (x y : ℝ) (h : x^2 + 12 * y = 0) : y = 3 :=
sorry

end parabola_directrix_l1815_181528


namespace remainder_difference_l1815_181526

theorem remainder_difference :
  ∃ (d r: ℤ), (1 < d) ∧ (1250 % d = r) ∧ (1890 % d = r) ∧ (2500 % d = r) ∧ (d - r = 10) :=
sorry

end remainder_difference_l1815_181526


namespace probability_green_marbles_correct_l1815_181503

noncomputable def probability_of_two_green_marbles : ℚ :=
  let total_marbles := 12
  let green_marbles := 7
  let prob_first_green := green_marbles / total_marbles
  let prob_second_green := (green_marbles - 1) / (total_marbles - 1)
  prob_first_green * prob_second_green

theorem probability_green_marbles_correct :
  probability_of_two_green_marbles = 7 / 22 := by
    sorry

end probability_green_marbles_correct_l1815_181503


namespace grazing_months_for_b_l1815_181534

/-
  We define the problem conditions and prove that b put his oxen for grazing for 5 months.
-/

theorem grazing_months_for_b (x : ℕ) :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let c_oxen := 15
  let c_months := 3
  let total_rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * x
  let c_ox_months := c_oxen * c_months
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  (c_share : ℚ) / total_rent = (c_ox_months : ℚ) / total_ox_months →
  x = 5 :=
by
  sorry

end grazing_months_for_b_l1815_181534


namespace largest_possible_product_l1815_181590

theorem largest_possible_product : 
  ∃ S1 S2 : Finset ℕ, 
  (S1 ∪ S2 = {1, 3, 4, 6, 7, 8, 9} ∧ S1 ∩ S2 = ∅ ∧ S1.prod id = S2.prod id) ∧ 
  (S1.prod id = 504 ∧ S2.prod id = 504) :=
by
  sorry

end largest_possible_product_l1815_181590


namespace arrangement_of_students_l1815_181530

theorem arrangement_of_students :
  let total_students := 5
  let total_communities := 2
  (2 ^ total_students - 2) = 30 :=
by
  let total_students := 5
  let total_communities := 2
  sorry

end arrangement_of_students_l1815_181530


namespace second_grade_survey_count_l1815_181573

theorem second_grade_survey_count :
  ∀ (total_students first_ratio second_ratio third_ratio total_surveyed : ℕ),
  total_students = 1500 →
  first_ratio = 4 →
  second_ratio = 5 →
  third_ratio = 6 →
  total_surveyed = 150 →
  second_ratio * total_surveyed / (first_ratio + second_ratio + third_ratio) = 50 :=
by 
  intros total_students first_ratio second_ratio third_ratio total_surveyed
  sorry

end second_grade_survey_count_l1815_181573


namespace total_savings_over_12_weeks_l1815_181591

-- Define the weekly savings and durations for each period
def weekly_savings_period_1 : ℕ := 5
def duration_period_1 : ℕ := 4

def weekly_savings_period_2 : ℕ := 10
def duration_period_2 : ℕ := 4

def weekly_savings_period_3 : ℕ := 20
def duration_period_3 : ℕ := 4

-- Define the total savings calculation for each period
def total_savings_period_1 : ℕ := weekly_savings_period_1 * duration_period_1
def total_savings_period_2 : ℕ := weekly_savings_period_2 * duration_period_2
def total_savings_period_3 : ℕ := weekly_savings_period_3 * duration_period_3

-- Prove that the total savings over 12 weeks equals $140.00
theorem total_savings_over_12_weeks : total_savings_period_1 + total_savings_period_2 + total_savings_period_3 = 140 := 
by 
  sorry

end total_savings_over_12_weeks_l1815_181591


namespace difference_of_squares_l1815_181599

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x + y = 15
def condition2 : Prop := x - y = 10

-- Goal to prove
theorem difference_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 - y^2 = 150 := 
by sorry

end difference_of_squares_l1815_181599


namespace value_of_Y_l1815_181510

/- Define the conditions given in the problem -/
def first_row_arithmetic_seq (a1 d1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d1
def fourth_row_arithmetic_seq (a4 d4 : ℕ) (n : ℕ) : ℕ := a4 + (n - 1) * d4

/- Constants given by the problem -/
def a1 : ℕ := 3
def fourth_term_first_row : ℕ := 27
def a4 : ℕ := 6
def fourth_term_fourth_row : ℕ := 66

/- Calculating common differences for first and fourth rows -/
def d1 : ℕ := (fourth_term_first_row - a1) / 3
def d4 : ℕ := (fourth_term_fourth_row - a4) / 3

/- Note that we are given that Y is at position (2, 2)
   Express Y in definition forms -/
def Y_row := first_row_arithmetic_seq (a1 + d1) d4 2
def Y_column := fourth_row_arithmetic_seq (a4 + d4) d1 2

/- Problem statement in Lean 4 -/
theorem value_of_Y : Y_row = 35 ∧ Y_column = 35 := by
  sorry

end value_of_Y_l1815_181510


namespace number_of_sequences_of_length_100_l1815_181539

def sequence_count (n : ℕ) : ℕ :=
  3^n - 2^n

theorem number_of_sequences_of_length_100 :
  sequence_count 100 = 3^100 - 2^100 :=
by
  sorry

end number_of_sequences_of_length_100_l1815_181539


namespace apples_eaten_l1815_181505

-- Define the number of apples eaten by Anna on Tuesday
def apples_eaten_on_Tuesday : ℝ := 4

theorem apples_eaten (A : ℝ) (h1 : A = apples_eaten_on_Tuesday) 
                      (h2 : 2 * A = 2 * apples_eaten_on_Tuesday) 
                      (h3 : A / 2 = apples_eaten_on_Tuesday / 2) 
                      (h4 : A + (2 * A) + (A / 2) = 14) : 
  A = 4 :=
by {
  sorry
}

end apples_eaten_l1815_181505


namespace remainder_mul_three_division_l1815_181584

theorem remainder_mul_three_division
    (N : ℤ) (k : ℤ)
    (h1 : N = 1927 * k + 131) :
    ((3 * N) % 43) = 6 :=
by
  sorry

end remainder_mul_three_division_l1815_181584


namespace average_speed_of_journey_is_24_l1815_181518

noncomputable def average_speed (D : ℝ) (speed_to_office speed_to_home : ℝ) : ℝ :=
  let time_to_office := D / speed_to_office
  let time_to_home := D / speed_to_home
  let total_distance := 2 * D
  let total_time := time_to_office + time_to_home
  total_distance / total_time

theorem average_speed_of_journey_is_24 (D : ℝ) : average_speed D 20 30 = 24 := by
  -- nonconstructive proof to fulfill theorem definition
  sorry

end average_speed_of_journey_is_24_l1815_181518


namespace combined_age_in_ten_years_l1815_181567

theorem combined_age_in_ten_years (B A: ℕ) (hA : A = 20) (h1: A + 10 = 2 * (B + 10)): 
  (A + 10) + (B + 10) = 45 := 
by
  sorry

end combined_age_in_ten_years_l1815_181567


namespace part1_solution_part2_solution_l1815_181592

-- Define the inequality
def inequality (m x : ℝ) : Prop := (m - 1) * x ^ 2 + (m - 1) * x + 2 > 0

-- Part (1): Prove the solution set for m = 0 is (-2, 1)
theorem part1_solution :
  (∀ x : ℝ, inequality 0 x → (-2 : ℝ) < x ∧ x < 1) := 
by
  sorry

-- Part (2): Prove the range of values for m such that the solution set is R
theorem part2_solution (m : ℝ) :
  (∀ x : ℝ, inequality m x) ↔ (1 ≤ m ∧ m < 9) := 
by
  sorry

end part1_solution_part2_solution_l1815_181592


namespace sequence_difference_l1815_181525

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sequence_difference (hS : ∀ n, S n = n^2 - 5 * n)
                            (hna : ∀ n, a n = S n - S (n - 1))
                            (hpq : p - q = 4) :
                            a p - a q = 8 := by
    sorry

end sequence_difference_l1815_181525


namespace percent_of_dollar_is_37_l1815_181580

variable (coins_value_in_cents : ℕ)
variable (percent_of_one_dollar : ℕ)

def value_of_pennies : ℕ := 2 * 1
def value_of_nickels : ℕ := 3 * 5
def value_of_dimes : ℕ := 2 * 10

def total_coin_value : ℕ := value_of_pennies + value_of_nickels + value_of_dimes

theorem percent_of_dollar_is_37
  (h1 : total_coin_value = coins_value_in_cents)
  (h2 : percent_of_one_dollar = (coins_value_in_cents * 100) / 100) : 
  percent_of_one_dollar = 37 := 
by
  sorry

end percent_of_dollar_is_37_l1815_181580


namespace smallest_n_l1815_181521

theorem smallest_n :
  ∃ n : ℕ, n = 10 ∧ (n * (n + 1) > 100 ∧ ∀ m : ℕ, m < n → m * (m + 1) ≤ 100) := by
  sorry

end smallest_n_l1815_181521


namespace irrational_pi_l1815_181512

theorem irrational_pi :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (π = a / b)) :=
sorry

end irrational_pi_l1815_181512


namespace hyperbola_eccentricity_range_l1815_181538

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (1 < (Real.sqrt (a^2 + b^2)) / a) ∧ ((Real.sqrt (a^2 + b^2)) / a < (2 * Real.sqrt 3) / 3) :=
sorry

end hyperbola_eccentricity_range_l1815_181538


namespace overlap_area_of_sectors_l1815_181597

/--
Given two sectors of a circle with radius 10, with centers at points P and R respectively, 
one having a central angle of 45 degrees and the other having a central angle of 90 degrees, 
prove that the area of the shaded region where they overlap is 12.5π.
-/
theorem overlap_area_of_sectors 
  (r : ℝ) (θ₁ θ₂ : ℝ) (A₁ A₂ : ℝ)
  (h₀ : r = 10)
  (h₁ : θ₁ = 45)
  (h₂ : θ₂ = 90)
  (hA₁ : A₁ = (θ₁ / 360) * π * r ^ 2)
  (hA₂ : A₂ = (θ₂ / 360) * π * r ^ 2)
  : A₁ = 12.5 * π := 
sorry

end overlap_area_of_sectors_l1815_181597


namespace solve_equation_l1815_181515

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end solve_equation_l1815_181515


namespace ball_bounce_height_l1815_181587

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (hₖ : ℕ → ℝ) :
  h₀ = 500 ∧ r = 0.6 ∧ (∀ k, hₖ k = h₀ * r^k) → 
  ∃ k, hₖ k < 3 ∧ k ≥ 22 := 
by
  sorry

end ball_bounce_height_l1815_181587


namespace ellipse_fixed_point_l1815_181502

theorem ellipse_fixed_point (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (c : ℝ) (h3 : c = 1) 
    (h4 : a = 2) (h5 : b = Real.sqrt 3) :
    (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
        ∃ M : ℝ × ℝ, (M.1 = 4) ∧ 
        ∃ Q : ℝ × ℝ, (Q.1= (P.1) ∧ Q.2 = - (P.2)) ∧ 
            ∃ fixed_point : ℝ × ℝ, (fixed_point.1 = 5 / 2) ∧ (fixed_point.2 = 0) ∧ 
            ∃ k, (Q.2 - M.2) = k * (Q.1 - M.1) ∧ 
            ∃ l, fixed_point.2 = l * (fixed_point.1 - M.1)) :=
sorry

end ellipse_fixed_point_l1815_181502


namespace no_integer_solution_l1815_181576

theorem no_integer_solution :
  ∀ (x y : ℤ), ¬(x^4 + x + y^2 = 3 * y - 1) :=
by
  intros x y
  sorry

end no_integer_solution_l1815_181576


namespace max_len_sequence_x_l1815_181514

theorem max_len_sequence_x :
  ∃ x : ℕ, 3088 < x ∧ x < 3091 :=
sorry

end max_len_sequence_x_l1815_181514


namespace inequality_holds_l1815_181533

-- Define parameters for the problem
variables (p q x y z : ℝ) (n : ℕ)

-- Define the conditions on x, y, and z
def condition1 : Prop := y = x^n + p*x + q
def condition2 : Prop := z = y^n + p*y + q
def condition3 : Prop := x = z^n + p*z + q

-- Define the statement of the inequality
theorem inequality_holds (h1 : condition1 p q x y n) (h2 : condition2 p q y z n) (h3 : condition3 p q x z n):
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y :=
sorry

end inequality_holds_l1815_181533


namespace factorization_correct_l1815_181527

theorem factorization_correct: 
  (a : ℝ) → a^2 - 9 = (a - 3) * (a + 3) :=
by
  intro a
  sorry

end factorization_correct_l1815_181527


namespace number_subtracted_l1815_181544

theorem number_subtracted (x y : ℤ) (h1 : x = 127) (h2 : 2 * x - y = 102) : y = 152 :=
by
  sorry

end number_subtracted_l1815_181544


namespace statement_T_true_for_given_values_l1815_181532

/-- Statement T: If the sum of the digits of a whole number m is divisible by 9, 
    then m is divisible by 9.
    The given values to check are 45, 54, 81, 63, and none of these. --/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem statement_T_true_for_given_values :
  ∀ (m : ℕ), (m = 45 ∨ m = 54 ∨ m = 81 ∨ m = 63) →
    (is_divisible_by_9 (sum_of_digits m) → is_divisible_by_9 m) :=
by
  intros m H
  cases H
  case inl H1 => sorry
  case inr H2 =>
    cases H2
    case inl H1 => sorry
    case inr H2 =>
      cases H2
      case inl H1 => sorry
      case inr H2 => sorry

end statement_T_true_for_given_values_l1815_181532


namespace divisor_is_18_l1815_181570

def dividend : ℕ := 165
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem divisor_is_18 (divisor : ℕ) : dividend = quotient * divisor + remainder → divisor = 18 :=
by sorry

end divisor_is_18_l1815_181570


namespace percentage_of_60_l1815_181519

theorem percentage_of_60 (x : ℝ) : 
  (0.2 * 40) + (x / 100) * 60 = 23 → x = 25 :=
by
  sorry

end percentage_of_60_l1815_181519


namespace abs_neg_one_tenth_l1815_181550

theorem abs_neg_one_tenth : |(-1 : ℚ) / 10| = 1 / 10 :=
by
  sorry

end abs_neg_one_tenth_l1815_181550


namespace two_discounts_l1815_181598

theorem two_discounts (p : ℝ) : (0.9 * 0.9 * p) = 0.81 * p :=
by
  sorry

end two_discounts_l1815_181598


namespace polynomial_identity_l1815_181500

theorem polynomial_identity
  (z1 z2 : ℂ)
  (h1 : z1 + z2 = -6)
  (h2 : z1 * z2 = 11)
  : (1 + z1^2 * z2) * (1 + z1 * z2^2) = 1266 := 
by 
  sorry

end polynomial_identity_l1815_181500


namespace exists_integers_a_b_for_m_l1815_181524

theorem exists_integers_a_b_for_m (m : ℕ) (h : 0 < m) :
  ∃ a b : ℤ, |a| ≤ m ∧ |b| ≤ m ∧ 0 < a + b * Real.sqrt 2 ∧ a + b * Real.sqrt 2 ≤ (1 + Real.sqrt 2) / (m + 2) :=
by
  sorry

end exists_integers_a_b_for_m_l1815_181524


namespace boat_avg_speed_ratio_l1815_181557

/--
A boat moves at a speed of 20 mph in still water. When traveling in a river with a current of 3 mph, it travels 24 miles downstream and then returns upstream to the starting point. Prove that the ratio of the average speed for the entire round trip to the boat's speed in still water is 97765 / 100000.
-/
theorem boat_avg_speed_ratio :
  let boat_speed := 20 -- mph in still water
  let current_speed := 3 -- mph river current
  let distance := 24 -- miles downstream and upstream
  let downstream_speed := boat_speed + current_speed
  let upstream_speed := boat_speed - current_speed
  let time_downstream := distance / downstream_speed
  let time_upstream := distance / upstream_speed
  let total_time := time_downstream + time_upstream
  let total_distance := distance * 2
  let average_speed := total_distance / total_time
  (average_speed / boat_speed) = 97765 / 100000 :=
by
  sorry

end boat_avg_speed_ratio_l1815_181557


namespace find_number_l1815_181595

theorem find_number (p q N : ℝ) (h1 : N / p = 8) (h2 : N / q = 18) (h3 : p - q = 0.20833333333333334) : N = 3 :=
sorry

end find_number_l1815_181595


namespace intersection_A_B_l1815_181517

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l1815_181517


namespace right_handed_players_total_l1815_181547

def total_players : ℕ := 64
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers
def total_right_handed : ℕ := throwers + right_handed_non_throwers

theorem right_handed_players_total : total_right_handed = 55 := by
  sorry

end right_handed_players_total_l1815_181547


namespace solution_l1815_181577

noncomputable def problem (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (∀ x : ℝ, x^2 - 14 * p * x - 15 * q = 0 → x = r ∨ x = s) ∧
  (∀ x : ℝ, x^2 - 14 * r * x - 15 * s = 0 → x = p ∨ x = q)

theorem solution (p q r s : ℝ) (h : problem p q r s) : p + q + r + s = 3150 :=
sorry

end solution_l1815_181577


namespace ratio_of_numbers_l1815_181555

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l1815_181555


namespace minimum_x_plus_2y_exists_l1815_181585

theorem minimum_x_plus_2y_exists (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) :
  ∃ z : ℝ, z = x + 2 * y ∧ z = -2 * Real.sqrt 2 - 1 :=
sorry

end minimum_x_plus_2y_exists_l1815_181585


namespace gum_total_l1815_181508

theorem gum_total (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) : 
  58 + x + y = 58 + x + y :=
by sorry

end gum_total_l1815_181508


namespace elise_saving_correct_l1815_181511

-- Definitions based on the conditions
def initial_money : ℤ := 8
def spent_comic_book : ℤ := 2
def spent_puzzle : ℤ := 18
def final_money : ℤ := 1

-- The theorem to prove the amount saved
theorem elise_saving_correct (x : ℤ) : 
  initial_money + x - spent_comic_book - spent_puzzle = final_money → x = 13 :=
by
  sorry

end elise_saving_correct_l1815_181511


namespace conditions_for_inequality_l1815_181569

theorem conditions_for_inequality (a b : ℝ) :
  (∀ x : ℝ, abs ((x^2 + a * x + b) / (x^2 + 2 * x + 2)) < 1) → 
  (a = 2 ∧ 0 < b ∧ b < 2) :=
sorry

end conditions_for_inequality_l1815_181569


namespace math_problem_proof_l1815_181560

noncomputable def problem_statement (a b c : ℝ) : Prop :=
 (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧ (a + b + c = 0) ∧ (a^4 + b^4 + c^4 = a^6 + b^6 + c^6) → 
 (a^2 + b^2 + c^2 = 3 / 2)

theorem math_problem_proof : ∀ (a b c : ℝ), problem_statement a b c :=
by
  intros
  sorry

end math_problem_proof_l1815_181560


namespace b_completes_work_alone_l1815_181594

theorem b_completes_work_alone (A_twice_B : ∀ (B : ℕ), A = 2 * B)
  (together : ℕ := 7) : ∃ (B : ℕ), 21 = 3 * together :=
by
  sorry

end b_completes_work_alone_l1815_181594


namespace real_numbers_division_l1815_181566

def is_non_neg (x : ℝ) : Prop := x ≥ 0

theorem real_numbers_division :
  ∀ x : ℝ, x < 0 ∨ is_non_neg x :=
by
  intro x
  by_cases h : x < 0
  · left
    exact h
  · right
    push_neg at h
    exact h

end real_numbers_division_l1815_181566


namespace arithmetic_sequence_a2_l1815_181504

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a1_a3 : a 1 + a 3 = 2) : a 2 = 1 :=
sorry

end arithmetic_sequence_a2_l1815_181504


namespace range_of_a_l1815_181596

theorem range_of_a (a : ℝ) (h₁ : ∀ x : ℝ, x > 0 → x + 4 / x ≥ a) (h₂ : ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) :
  a ≤ -1 ∨ (2 ≤ a ∧ a ≤ 4) :=
sorry

end range_of_a_l1815_181596


namespace evaluate_expression_l1815_181562

theorem evaluate_expression : (7 - 3) ^ 2 + (7 ^ 2 - 3 ^ 2) = 56 := by
  sorry

end evaluate_expression_l1815_181562


namespace sophia_pages_difference_l1815_181537

theorem sophia_pages_difference (total_pages : ℕ) (f_fraction : ℚ) (l_fraction : ℚ) 
  (finished_pages : ℕ) (left_pages : ℕ) :
  f_fraction = 2/3 ∧ 
  l_fraction = 1/3 ∧
  total_pages = 270 ∧
  finished_pages = f_fraction * total_pages ∧
  left_pages = l_fraction * total_pages
  →
  finished_pages - left_pages = 90 :=
by
  intro h
  sorry

end sophia_pages_difference_l1815_181537


namespace Tom_runs_60_miles_per_week_l1815_181529

theorem Tom_runs_60_miles_per_week
  (days_per_week : ℕ := 5)
  (hours_per_day : ℝ := 1.5)
  (speed_mph : ℝ := 8) :
  (days_per_week * hours_per_day * speed_mph = 60) := by
  sorry

end Tom_runs_60_miles_per_week_l1815_181529


namespace largest_fraction_l1815_181546

theorem largest_fraction (d x : ℕ) 
  (h1: (2 * x / d) + (3 * x / d) + (4 * x / d) = 10 / 11)
  (h2: d = 11 * x) : (4 / 11 : ℚ) = (4 * x / d : ℚ) :=
by
  sorry

end largest_fraction_l1815_181546


namespace average_girls_score_l1815_181579

open Function

variable (C c D d : ℕ)
variable (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)

-- Conditions
def CedarBoys := avgCedarBoys = 85
def CedarGirls := avgCedarGirls = 80
def CedarCombined := avgCedarCombined = 83
def DeltaBoys := avgDeltaBoys = 76
def DeltaGirls := avgDeltaGirls = 95
def DeltaCombined := avgDeltaCombined = 87
def CombinedBoys := avgCombinedBoys = 73

-- Correct answer
def CombinedGirls (avgCombinedGirls : ℤ) := avgCombinedGirls = 86

-- Final statement
theorem average_girls_score (C c D d : ℕ)
    (avgCedarBoys avgCedarGirls avgCedarCombined avgDeltaBoys avgDeltaGirls avgDeltaCombined avgCombinedBoys : ℤ)
    (H1 : CedarBoys avgCedarBoys)
    (H2 : CedarGirls avgCedarGirls)
    (H3 : CedarCombined avgCedarCombined)
    (H4 : DeltaBoys avgDeltaBoys)
    (H5 : DeltaGirls avgDeltaGirls)
    (H6 : DeltaCombined avgDeltaCombined)
    (H7 : CombinedBoys avgCombinedBoys) :
    ∃ avgCombinedGirls, CombinedGirls avgCombinedGirls :=
sorry

end average_girls_score_l1815_181579


namespace greatest_possible_integer_radius_l1815_181582

theorem greatest_possible_integer_radius :
  ∃ r : ℤ, (50 < (r : ℝ)^2) ∧ ((r : ℝ)^2 < 75) ∧ 
  (∀ s : ℤ, (50 < (s : ℝ)^2) ∧ ((s : ℝ)^2 < 75) → s ≤ r) :=
sorry

end greatest_possible_integer_radius_l1815_181582


namespace simplify_and_evaluate_l1815_181509

theorem simplify_and_evaluate (a b : ℝ) (h1 : a = -1) (h2 : b = 1) :
  (4/5 * a * b - (2 * a * b^2 - 4 * (-1/5 * a * b + 3 * a^2 * b)) + 2 * a * b^2) = 12 :=
by
  have ha : a = -1 := h1
  have hb : b = 1 := h2
  sorry

end simplify_and_evaluate_l1815_181509


namespace hall_width_l1815_181574

theorem hall_width (w : ℝ) (length height cost_per_m2 total_expenditure : ℝ)
  (h_length : length = 20)
  (h_height : height = 5)
  (h_cost : cost_per_m2 = 50)
  (h_expenditure : total_expenditure = 47500)
  (h_area : total_expenditure = cost_per_m2 * (2 * (length * w) + 2 * (length * height) + 2 * (w * height))) :
  w = 15 := 
sorry

end hall_width_l1815_181574


namespace correct_statement_about_algorithms_l1815_181513

-- Definitions based on conditions
def is_algorithm (A B C D : Prop) : Prop :=
  ¬A ∧ B ∧ ¬C ∧ ¬D

-- Ensure the correct statement using the conditions specified
theorem correct_statement_about_algorithms (A B C D : Prop) (h : is_algorithm A B C D) : B :=
by
  obtain ⟨hnA, hB, hnC, hnD⟩ := h
  exact hB

end correct_statement_about_algorithms_l1815_181513


namespace michael_passes_donovan_l1815_181564

theorem michael_passes_donovan
  (track_length : ℕ)
  (donovan_lap_time : ℕ)
  (michael_lap_time : ℕ)
  (start_time : ℕ)
  (L : ℕ)
  (h1 : track_length = 500)
  (h2 : donovan_lap_time = 45)
  (h3 : michael_lap_time = 40)
  (h4 : start_time = 0)
  : L = 9 :=
by
  sorry

end michael_passes_donovan_l1815_181564


namespace percentage_increase_is_20_percent_l1815_181554

noncomputable def originalSalary : ℝ := 575 / 1.15
noncomputable def increasedSalary : ℝ := 600
noncomputable def percentageIncreaseTo600 : ℝ := (increasedSalary - originalSalary) / originalSalary * 100

theorem percentage_increase_is_20_percent :
  percentageIncreaseTo600 = 20 := 
by
  sorry -- The proof will go here

end percentage_increase_is_20_percent_l1815_181554


namespace infinite_series_sum_l1815_181553

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * n - 1) / 3 ^ (n + 1)) = 2 :=
by
  sorry

end infinite_series_sum_l1815_181553


namespace general_term_sequence_l1815_181583

variable {a : ℕ → ℝ}
variable {n : ℕ}

def sequence_condition (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ (∀ n : ℕ, n ≥ 1 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a n) * (a (n + 1)) = 0)

theorem general_term_sequence (a : ℕ → ℝ) (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 / n := by
  sorry

end general_term_sequence_l1815_181583


namespace cost_of_building_fence_square_plot_l1815_181506

-- Definition of conditions
def area_of_square_plot : ℕ := 289
def price_per_foot : ℕ := 60

-- Resulting theorem statement
theorem cost_of_building_fence_square_plot : 
  let side_length := Int.sqrt area_of_square_plot
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 4080 := 
by
  -- Placeholder for the actual proof
  sorry

end cost_of_building_fence_square_plot_l1815_181506


namespace smallest_three_digit_number_l1815_181559

theorem smallest_three_digit_number (digits : Finset ℕ) (h_digits : digits = {0, 3, 5, 6}) : 
  ∃ n, n = 305 ∧ ∀ m, (m ∈ digits) → (m ≠ 0) → (m < 305) → false :=
by
  sorry

end smallest_three_digit_number_l1815_181559


namespace max_net_income_meeting_point_l1815_181543

theorem max_net_income_meeting_point :
  let A := (9 : ℝ)
  let B := (6 : ℝ)
  let cost_per_mile := 1
  let payment_per_mile := 2
  ∃ x : ℝ, 
  let AP := Real.sqrt ((x - 9)^2 + 12^2)
  let PB := Real.sqrt ((x - 6)^2 + 3^2)
  let net_income := payment_per_mile * PB - (AP + PB)
  x = -12.5 := 
sorry

end max_net_income_meeting_point_l1815_181543


namespace line_tangent_to_parabola_l1815_181535

theorem line_tangent_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → 16 - 16 * c = 0) → c = 1 :=
by
  intros h
  sorry

end line_tangent_to_parabola_l1815_181535


namespace a_minus_b_greater_than_one_l1815_181593

open Real

theorem a_minus_b_greater_than_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b)
  (f_has_three_roots : ∃ (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r1 ≠ r3 ∧ r2 ≠ r3 ∧ (Polynomial.aeval r1 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r2 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0 ∧ (Polynomial.aeval r3 (Polynomial.C (-1) + Polynomial.X * (Polynomial.C (2*b)) + Polynomial.X^2 * (Polynomial.C a) + Polynomial.X^3)) = 0)
  (g_no_real_roots : ∀ (x : ℝ), (2*x^2 + 2*b*x + a) ≠ 0) :
  a - b > 1 := by
  sorry

end a_minus_b_greater_than_one_l1815_181593


namespace original_price_of_sarees_l1815_181548

theorem original_price_of_sarees
  (P : ℝ)
  (h_sale_price : 0.80 * P * 0.85 = 306) :
  P = 450 :=
sorry

end original_price_of_sarees_l1815_181548


namespace range_of_a_l1815_181586

def proposition_p (a : ℝ) : Prop :=
  (a + 6) * (a - 7) < 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4 * x + a < 0

def neg_q (a : ℝ) : Prop :=
  a ≥ 4

theorem range_of_a (a : ℝ) :
  (proposition_p a ∨ neg_q a) ↔ a ∈ Set.Ioo (-6 : ℝ) (7 : ℝ) ∪ Set.Ici (4 : ℝ) :=
sorry

end range_of_a_l1815_181586


namespace polygon_sides_from_diagonals_l1815_181581

theorem polygon_sides_from_diagonals (n : ℕ) (h : ↑((n * (n - 3)) / 2) = 14) : n = 7 :=
by
  sorry

end polygon_sides_from_diagonals_l1815_181581


namespace rational_x_of_rational_x3_and_x2_add_x_l1815_181501

variable {x : ℝ}

theorem rational_x_of_rational_x3_and_x2_add_x (hx3 : ∃ a : ℚ, x^3 = a)
  (hx2_add_x : ∃ b : ℚ, x^2 + x = b) : ∃ r : ℚ, x = r :=
sorry

end rational_x_of_rational_x3_and_x2_add_x_l1815_181501


namespace max_correct_answers_l1815_181507

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 25) (h2 : 6 * c - 3 * w = 60) : c ≤ 15 :=
by {
  sorry
}

end max_correct_answers_l1815_181507


namespace initial_cargo_l1815_181558

theorem initial_cargo (initial_cargo additional_cargo total_cargo : ℕ) 
  (h1 : additional_cargo = 8723) 
  (h2 : total_cargo = 14696) 
  (h3 : initial_cargo + additional_cargo = total_cargo) : 
  initial_cargo = 5973 := 
by 
  -- Start with the assumptions and directly obtain the calculation as required
  sorry

end initial_cargo_l1815_181558


namespace sum_of_two_longest_altitudes_l1815_181542

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l1815_181542


namespace sum_intercepts_of_line_l1815_181552

theorem sum_intercepts_of_line (x y : ℝ) (h_eq : y - 6 = -2 * (x - 3)) :
  (∃ x_int : ℝ, (0 - 6 = -2 * (x_int - 3)) ∧ x_int = 6) ∧
  (∃ y_int : ℝ, (y_int - 6 = -2 * (0 - 3)) ∧ y_int = 12) →
  6 + 12 = 18 :=
by sorry

end sum_intercepts_of_line_l1815_181552


namespace x_minus_y_values_l1815_181549

theorem x_minus_y_values (x y : ℝ) 
  (h1 : y = Real.sqrt (x^2 - 9) - Real.sqrt (9 - x^2) + 4) : x - y = -1 ∨ x - y = -7 := 
  sorry

end x_minus_y_values_l1815_181549


namespace part1_l1815_181556

variable {a b : ℝ}
variable {A B C : ℝ}
variable {S : ℝ}

-- Given Conditions
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (b * Real.cos C - c * Real.cos B = 2 * a) ∧ (c = a)

-- To prove
theorem part1 (h : is_triangle A B C a b a) : B = 2 * Real.pi / 3 := sorry

end part1_l1815_181556


namespace construction_company_sand_weight_l1815_181561

theorem construction_company_sand_weight :
  ∀ (total_weight gravel_weight : ℝ), total_weight = 14.02 → gravel_weight = 5.91 → 
  total_weight - gravel_weight = 8.11 :=
by 
  intros total_weight gravel_weight h_total h_gravel 
  sorry

end construction_company_sand_weight_l1815_181561


namespace evaluate_g_ggg_neg1_l1815_181531

def g (y : ℤ) : ℤ := y^3 - 3*y + 1

theorem evaluate_g_ggg_neg1 : g (g (g (-1))) = 6803 := 
by
  sorry

end evaluate_g_ggg_neg1_l1815_181531


namespace number_of_solutions_l1815_181563

open Real

-- Define main condition
def condition (θ : ℝ) : Prop := sin θ * tan θ = 2 * (cos θ)^2

-- Define the interval and exclusions
def valid_theta (θ : ℝ) : Prop := 
  0 ≤ θ ∧ θ ≤ 2 * π ∧ ¬ ( ∃ k : ℤ, (θ = k * (π/2)) )

-- Define the set of thetas that satisfy both the condition and the valid interval
def valid_solutions (θ : ℝ) : Prop := valid_theta θ ∧ condition θ

-- Formal statement of the problem
theorem number_of_solutions : 
  ∃ (s : Finset ℝ), (∀ θ ∈ s, valid_solutions θ) ∧ (s.card = 4) := by
  sorry

end number_of_solutions_l1815_181563


namespace circle_equation_l1815_181572

theorem circle_equation 
  (x y : ℝ)
  (center : ℝ × ℝ)
  (tangent_point : ℝ × ℝ)
  (line1 : ℝ × ℝ → Prop)
  (line2 : ℝ × ℝ → Prop)
  (hx : line1 center)
  (hy : line2 tangent_point)
  (tangent_point_val : tangent_point = (2, -1))
  (line1_def : ∀ (p : ℝ × ℝ), line1 p ↔ 2 * p.1 + p.2 = 0)
  (line2_def : ∀ (p : ℝ × ℝ), line2 p ↔ p.1 + p.2 - 1 = 0) :
  (∃ (x0 y0 r : ℝ), center = (x0, y0) ∧ r > 0 ∧ (x - x0)^2 + (y - y0)^2 = r^2 ∧ 
                        (x - x0)^2 + (y - y0)^2 = (x - 1)^2 + (y + 2)^2 ∧ 
                        (x - 1)^2 + (y + 2)^2 = 2) :=
by {
  sorry
}

end circle_equation_l1815_181572


namespace cos_seven_pi_over_six_l1815_181565

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l1815_181565


namespace speed_of_first_car_l1815_181545

variable (V1 V2 V3 : ℝ) -- Define the speeds of the three cars
variable (t x : ℝ) -- Time interval and distance from A to B

-- Conditions of the problem
axiom condition_1 : x / V1 = (x / V2) + t
axiom condition_2 : x / V2 = (x / V3) + t
axiom condition_3 : 120 / V1  = (120 / V2) + 1
axiom condition_4 : 40 / V1 = 80 / V3

-- Proof statement
theorem speed_of_first_car : V1 = 30 := by
  sorry

end speed_of_first_car_l1815_181545


namespace cube_weight_doubled_side_length_l1815_181588

-- Theorem: Prove that the weight of a new cube with sides twice as long as the original cube is 40 pounds, given the conditions.
theorem cube_weight_doubled_side_length (s : ℝ) (h₁ : s > 0) (h₂ : (s^3 : ℝ) > 0) (w : ℝ) (h₃ : w = 5) : 
  8 * w = 40 :=
by
  sorry

end cube_weight_doubled_side_length_l1815_181588


namespace geometric_sequence_condition_l1815_181540

theorem geometric_sequence_condition (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → (a * d = b * c) ∧ 
  ¬ (∀ a b c d : ℝ, a * d = b * c → ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) := 
by
  sorry

end geometric_sequence_condition_l1815_181540


namespace value_of_m_l1815_181571

theorem value_of_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∃ (k : ℝ), (2 * m - 1) * x ^ (m ^ 2) = k * x ^ n) → m = 1 :=
by
  sorry

end value_of_m_l1815_181571


namespace avg_of_first_5_numbers_equal_99_l1815_181523

def avg_of_first_5 (S1 : ℕ) : ℕ := S1 / 5

theorem avg_of_first_5_numbers_equal_99
  (avg_9 : ℕ := 104) (avg_last_5 : ℕ := 100) (fifth_num : ℕ := 59)
  (sum_9 := 9 * avg_9) (sum_last_5 := 5 * avg_last_5) :
  avg_of_first_5 (sum_9 - sum_last_5 + fifth_num) = 99 :=
by
  sorry

end avg_of_first_5_numbers_equal_99_l1815_181523


namespace cheryl_initial_skitttles_l1815_181589

-- Given conditions
def cheryl_ends_with (ends_with : ℕ) : Prop := ends_with = 97
def kathryn_gives (gives : ℕ) : Prop := gives = 89

-- To prove: cheryl_starts_with + kathryn_gives = cheryl_ends_with
theorem cheryl_initial_skitttles (cheryl_starts_with : ℕ) :
  (∃ ends_with gives, cheryl_ends_with ends_with ∧ kathryn_gives gives ∧ 
  cheryl_starts_with + gives = ends_with) →
  cheryl_starts_with = 8 :=
by
  sorry

end cheryl_initial_skitttles_l1815_181589


namespace angle_B_is_arcsin_l1815_181522

-- Define the triangle and its conditions
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    a = 8 ∧ b = Real.sqrt 3 ∧ 
    (2 * Real.cos (A - B) / 2 ^ 2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)

-- Prove that the measure of ∠B is arcsin(√3 / 10)
theorem angle_B_is_arcsin (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
sorry

end angle_B_is_arcsin_l1815_181522


namespace total_oranges_and_weight_l1815_181541

theorem total_oranges_and_weight 
  (oranges_per_child : ℕ) (num_children : ℕ) (average_weight_per_orange : ℝ)
  (h1 : oranges_per_child = 3)
  (h2 : num_children = 4)
  (h3 : average_weight_per_orange = 0.3) :
  oranges_per_child * num_children = 12 ∧ (oranges_per_child * num_children : ℝ) * average_weight_per_orange = 3.6 :=
by
  sorry

end total_oranges_and_weight_l1815_181541


namespace tan_half_A_mul_tan_half_C_eq_third_l1815_181568

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem tan_half_A_mul_tan_half_C_eq_third (h : a + c = 2 * b) :
  (Real.tan (A / 2)) * (Real.tan (C / 2)) = 1 / 3 :=
sorry

end tan_half_A_mul_tan_half_C_eq_third_l1815_181568
