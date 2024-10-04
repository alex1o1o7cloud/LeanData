import Mathlib

namespace x_intercept_of_line_l258_258939

theorem x_intercept_of_line (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 10) (hy1 : y1 = 3) (hx2 : x2 = -8) (hy2 : y2 = -6) :
  ∃ x0 : ℝ, (∀ y : ℝ, y = 0 → (∃ m : ℝ, y = m * (x0 - x1) + y1)) ∧ x0 = 4 :=
by
  sorry

end x_intercept_of_line_l258_258939


namespace pq_eq_real_nums_l258_258901

theorem pq_eq_real_nums (p q r x y z : ℝ) 
  (h1 : x / p + q / y = 1) 
  (h2 : y / q + r / z = 1) : 
  p * q * r + x * y * z = 0 := 
by 
  sorry

end pq_eq_real_nums_l258_258901


namespace Elberta_has_21_dollars_l258_258583

theorem Elberta_has_21_dollars
  (Granny_Smith : ℕ)
  (Anjou : ℕ)
  (Elberta : ℕ)
  (h1 : Granny_Smith = 72)
  (h2 : Anjou = Granny_Smith / 4)
  (h3 : Elberta = Anjou + 3) :
  Elberta = 21 := 
  by {
    sorry
  }

end Elberta_has_21_dollars_l258_258583


namespace smallest_points_to_guarantee_victory_l258_258905

noncomputable def pointsForWinning : ℕ := 5
noncomputable def pointsForSecond : ℕ := 3
noncomputable def pointsForThird : ℕ := 1

theorem smallest_points_to_guarantee_victory :
  ∀ (student_points : ℕ),
  (exists (x y z : ℕ), (x = pointsForWinning ∨ x = pointsForSecond ∨ x = pointsForThird) ∧
                         (y = pointsForWinning ∨ y = pointsForSecond ∨ y = pointsForThird) ∧
                         (z = pointsForWinning ∨ z = pointsForSecond ∨ z = pointsForThird) ∧
                         student_points = x + y + z) →
  (∃ (victory_points : ℕ), victory_points = 13) →
  (∀ other_points : ℕ, other_points < victory_points) :=
sorry

end smallest_points_to_guarantee_victory_l258_258905


namespace normal_price_of_article_l258_258927

theorem normal_price_of_article 
  (final_price : ℝ) 
  (d1 d2 d3 : ℝ) 
  (P : ℝ) 
  (h_final_price : final_price = 36) 
  (h_d1 : d1 = 0.15) 
  (h_d2 : d2 = 0.25) 
  (h_d3 : d3 = 0.20) 
  (h_eq : final_price = P * (1 - d1) * (1 - d2) * (1 - d3)) : 
  P = 70.59 := sorry

end normal_price_of_article_l258_258927


namespace quadratic_expression_negative_for_all_x_l258_258832

theorem quadratic_expression_negative_for_all_x (k : ℝ) :
  (∀ x : ℝ, (5-k) * x^2 - 2 * (1-k) * x + 2 - 2 * k < 0) ↔ k > 9 :=
sorry

end quadratic_expression_negative_for_all_x_l258_258832


namespace ways_to_write_10003_as_sum_of_two_primes_l258_258418

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l258_258418


namespace invalid_inverse_statement_l258_258514

/- Define the statements and their inverses -/

/-- Statement A: Vertical angles are equal. -/
def statement_A : Prop := ∀ {α β : ℝ}, α ≠ β → α = β

/-- Inverse of Statement A: If two angles are equal, then they are vertical angles. -/
def inverse_A : Prop := ∀ {α β : ℝ}, α = β → α ≠ β

/-- Statement B: If |a| = |b|, then a = b. -/
def statement_B (a b : ℝ) : Prop := abs a = abs b → a = b

/-- Inverse of Statement B: If a = b, then |a| = |b|. -/
def inverse_B (a b : ℝ) : Prop := a = b → abs a = abs b

/-- Statement C: If two lines are parallel, then the alternate interior angles are equal. -/
def statement_C (l1 l2 : Prop) : Prop := l1 → l2

/-- Inverse of Statement C: If the alternate interior angles are equal, then the two lines are parallel. -/
def inverse_C (l1 l2 : Prop) : Prop := l2 → l1

/-- Statement D: If a^2 = b^2, then a = b. -/
def statement_D (a b : ℝ) : Prop := a^2 = b^2 → a = b

/-- Inverse of Statement D: If a = b, then a^2 = b^2. -/
def inverse_D (a b : ℝ) : Prop := a = b → a^2 = b^2

/-- The statement that does not have a valid inverse among A, B, C, and D is statement A. -/
theorem invalid_inverse_statement : ¬inverse_A :=
by
sorry

end invalid_inverse_statement_l258_258514


namespace sufficient_and_necessary_condition_l258_258615

variable {a : ℕ → ℝ}
variable {a1 a2 : ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) : Prop :=
  (∀ n, a n = a1 * q ^ n)

noncomputable def increasing (a : ℕ → ℝ) : Prop :=
  (∀ n, a n < a (n + 1))

theorem sufficient_and_necessary_condition
  (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ)
  (h_geom : geometric_sequence a a1 q)
  (h_a1_pos : a1 > 0)
  (h_a1_lt_a2 : a1 < a1 * q) :
  increasing a ↔ a1 < a1 * q := 
sorry

end sufficient_and_necessary_condition_l258_258615


namespace teapot_volume_proof_l258_258782

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem teapot_volume_proof (a d : ℝ)
  (h1 : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 + arithmetic_sequence a d 3 = 0.5)
  (h2 : arithmetic_sequence a d 7 + arithmetic_sequence a d 8 + arithmetic_sequence a d 9 = 2.5) :
  arithmetic_sequence a d 5 = 0.5 :=
by {
  sorry
}

end teapot_volume_proof_l258_258782


namespace distance_home_gym_l258_258049

theorem distance_home_gym 
  (v_WangLei v_ElderSister : ℕ)  -- speeds in meters per minute
  (d_meeting : ℕ)                -- distance in meters from the gym to the meeting point
  (t_gym : ℕ)                    -- time in minutes for the older sister to the gym
  (speed_diff : v_ElderSister = v_WangLei + 20)  -- speed difference
  (t_gym_reached : d_meeting / 2 = (25 * (v_WangLei + 20)) - d_meeting): 
  v_WangLei * t_gym = 1500 :=
by
  sorry

end distance_home_gym_l258_258049


namespace people_per_column_in_second_scenario_l258_258722

def total_people (num_people_per_column_1 : ℕ) (num_columns_1 : ℕ) : ℕ :=
  num_people_per_column_1 * num_columns_1

def people_per_column_second_scenario (P: ℕ) (num_columns_2 : ℕ) : ℕ :=
  P / num_columns_2

theorem people_per_column_in_second_scenario
  (num_people_per_column_1 : ℕ)
  (num_columns_1 : ℕ)
  (num_columns_2 : ℕ)
  (P : ℕ)
  (h1 : total_people num_people_per_column_1 num_columns_1 = P) :
  people_per_column_second_scenario P num_columns_2 = 48 :=
by
  -- the proof would go here
  sorry

end people_per_column_in_second_scenario_l258_258722


namespace smallest_prime_with_digit_sum_23_l258_258090

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258090


namespace length_of_goods_train_l258_258362

theorem length_of_goods_train
  (speed_man_train : ℕ) (speed_goods_train : ℕ) (passing_time : ℕ)
  (h1 : speed_man_train = 40)
  (h2 : speed_goods_train = 72)
  (h3 : passing_time = 9) :
  (112 * 1000 / 3600) * passing_time = 280 := 
by
  sorry

end length_of_goods_train_l258_258362


namespace find_f_x_l258_258578

theorem find_f_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x - 1) : 
  ∀ x : ℤ, f x = 2 * x - 3 :=
sorry

end find_f_x_l258_258578


namespace inequality_x_alpha_y_beta_l258_258899

theorem inequality_x_alpha_y_beta (x y α β : ℝ) (hx : 0 < x) (hy : 0 < y) 
(hα : 0 < α) (hβ : 0 < β) (hαβ : α + β = 1) : x^α * y^β ≤ α * x + β * y := 
sorry

end inequality_x_alpha_y_beta_l258_258899


namespace rice_mixture_ratio_l258_258000

theorem rice_mixture_ratio (x y z : ℕ) (h : 16 * x + 24 * y + 30 * z = 18 * (x + y + z)) : 
  x = 9 * y + 18 * z :=
by
  sorry

end rice_mixture_ratio_l258_258000


namespace sum_of_numbers_l258_258512

theorem sum_of_numbers :
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := 
  by
  sorry

end sum_of_numbers_l258_258512


namespace angelfish_goldfish_difference_l258_258674

-- Given statements
variables {A G : ℕ}
def goldfish := 8
def total_fish := 44

-- Conditions
axiom twice_as_many_guppies : G = 2 * A
axiom total_fish_condition : A + G + goldfish = total_fish

-- Theorem
theorem angelfish_goldfish_difference : A - goldfish = 4 :=
by
  sorry

end angelfish_goldfish_difference_l258_258674


namespace range_of_a_min_value_a_plus_4_over_a_sq_l258_258273

noncomputable def f (x : ℝ) : ℝ :=
  |x - 10| + |x - 20|

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < 10 * a + 10) ↔ 0 < a :=
sorry

theorem min_value_a_plus_4_over_a_sq (a : ℝ) (h : 0 < a) :
  ∃ y : ℝ, a + 4 / a ^ 2 = y ∧ y = 3 :=
sorry

end range_of_a_min_value_a_plus_4_over_a_sq_l258_258273


namespace find_x_eq_neg15_l258_258259

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l258_258259


namespace balance_proof_l258_258689

variables (a b c : ℝ)

theorem balance_proof (h1 : 4 * a + 2 * b = 12 * c) (h2 : 2 * a = b + 3 * c) : 3 * b = 4.5 * c :=
sorry

end balance_proof_l258_258689


namespace rainfall_on_tuesday_is_correct_l258_258105

-- Define the total days in a week
def days_in_week : ℕ := 7

-- Define the average rainfall for the whole week
def avg_rainfall : ℝ := 3.0

-- Define the total rainfall for the week
def total_rainfall : ℝ := avg_rainfall * days_in_week

-- Define a proposition that states rainfall on Tuesday equals 10.5 cm
def rainfall_on_tuesday (T : ℝ) : Prop :=
  T = 10.5

-- Prove that the rainfall on Tuesday is 10.5 cm given the conditions
theorem rainfall_on_tuesday_is_correct : rainfall_on_tuesday (total_rainfall / 2) :=
by
  sorry

end rainfall_on_tuesday_is_correct_l258_258105


namespace value_of_y_l258_258867

theorem value_of_y (x y : ℤ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := 
by
  sorry

end value_of_y_l258_258867


namespace expression_for_A_plus_2B_A_plus_2B_independent_of_b_l258_258978

theorem expression_for_A_plus_2B (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * b - 1
  let B := -a^2 - a * b + 1
  A + 2 * B = a * b - 2 * b + 1 :=
by
  sorry

theorem A_plus_2B_independent_of_b (a : ℝ) :
  (∀ b : ℝ, let A := 2 * a^2 + 3 * a * b - 2 * b - 1
            let B := -a^2 - a * b + 1
            A + 2 * B = a * b - 2 * b + 1) →
  a = 2 :=
by
  sorry

end expression_for_A_plus_2B_A_plus_2B_independent_of_b_l258_258978


namespace train_pass_time_approx_l258_258821

noncomputable def time_to_pass_platform
  (L_t L_p : ℝ)
  (V_t : ℝ) : ℝ :=
  (L_t + L_p) / (V_t * (1000 / 3600))

theorem train_pass_time_approx
  (L_t L_p V_t : ℝ)
  (hL_t : L_t = 720)
  (hL_p : L_p = 360)
  (hV_t : V_t = 75) :
  abs (time_to_pass_platform L_t L_p V_t - 51.85) < 0.01 := 
by
  rw [hL_t, hL_p, hV_t]
  sorry

end train_pass_time_approx_l258_258821


namespace no_solution_condition_l258_258165

theorem no_solution_condition (n : ℝ) : ¬(∃ x y z : ℝ, n^2 * x + y = 1 ∧ n * y + z = 1 ∧ x + n^2 * z = 1) ↔ n = -1 := 
by {
    sorry
}

end no_solution_condition_l258_258165


namespace koala_fiber_consumption_l258_258731

theorem koala_fiber_consumption (x : ℝ) (h : 0.40 * x = 8) : x = 20 :=
sorry

end koala_fiber_consumption_l258_258731


namespace depth_of_right_frustum_l258_258364

-- Definitions
def volume_cm3 := 190000 -- Volume in cubic centimeters (190 liters)
def top_edge := 60 -- Length of the top edge in centimeters
def bottom_edge := 40 -- Length of the bottom edge in centimeters
def expected_depth := 75 -- Expected depth in centimeters

-- The following is the statement of the proof
theorem depth_of_right_frustum 
  (V : ℝ) (A1 A2 : ℝ) (h : ℝ)
  (hV : V = 190 * 1000)
  (hA1 : A1 = top_edge * top_edge)
  (hA2 : A2 = bottom_edge * bottom_edge)
  (h_avg : 2 * A1 / (top_edge + bottom_edge) = 2 * A2 / (top_edge + bottom_edge))
  : h = expected_depth := 
sorry

end depth_of_right_frustum_l258_258364


namespace determine_value_of_a_l258_258146

theorem determine_value_of_a (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 
  1 ≤ (1 / 2 * x^2 - x + 3 / 2) ∧ (1 / 2 * x^2 - x + 3 / 2) ≤ a) →
  a = 3 :=
by
  sorry

end determine_value_of_a_l258_258146


namespace no_base_for_final_digit_one_l258_258562

theorem no_base_for_final_digit_one (b : ℕ) (h : 3 ≤ b ∧ b ≤ 10) : ¬ (842 % b = 1) :=
by
  cases h with 
  | intro hb1 hb2 => sorry

end no_base_for_final_digit_one_l258_258562


namespace original_number_value_l258_258651

theorem original_number_value (x : ℝ) (h : 0 < x) (h_eq : 10^4 * x = 4 / x) : x = 0.02 :=
sorry

end original_number_value_l258_258651


namespace polynomial_inequality_l258_258914

open Polynomial

noncomputable def polynomial_f (n : ℕ) (a : Fin n → ℝ) : ℝ[X] :=
  (X ^ n) + (∑ i in Finset.range n, (coeff n i a) * (X ^ (n - 1 - i)))
  where coeff (n : ℕ) (i : ℕ) (a : Fin n → ℝ) : ℝ := ite (i < n) (a ⟨i, nat.lt_succ_self i⟩) 1

theorem polynomial_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 ≤ a i) (hroots : (polynomial_f n a).roots.card = n) :
    eval 2 (polynomial_f n a) ≥ 3^n :=
begin
  sorry
end

end polynomial_inequality_l258_258914


namespace fran_speed_l258_258610

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l258_258610


namespace minimum_value_fraction_1_x_plus_1_y_l258_258358

theorem minimum_value_fraction_1_x_plus_1_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) :
  1 / x + 1 / y = 1 :=
sorry

end minimum_value_fraction_1_x_plus_1_y_l258_258358


namespace abs_neg_2023_l258_258486

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l258_258486


namespace num_ordered_pairs_l258_258381

theorem num_ordered_pairs (N : ℕ) :
  (N = 20) ↔ ∃ (a b : ℕ), 
  (a < b) ∧ (100 ≤ a ∧ a ≤ 1000)
  ∧ (100 ≤ b ∧ b ≤ 1000)
  ∧ (gcd a b * lcm a b = 495 * gcd a b)
  := 
sorry

end num_ordered_pairs_l258_258381


namespace parabolas_intersect_at_single_point_l258_258472

theorem parabolas_intersect_at_single_point (p q : ℝ) (h : -2 * p + q = 2023) :
  ∃ (x0 y0 : ℝ), (∀ p q : ℝ, y0 = x0^2 + p * x0 + q → -2 * p + q = 2023) ∧ x0 = -2 ∧ y0 = 2027 :=
by
  -- Proof to be filled in
  sorry

end parabolas_intersect_at_single_point_l258_258472


namespace sum_smallest_largest_eq_2z_l258_258200

theorem sum_smallest_largest_eq_2z (m b z : ℤ) (h1 : m > 0) (h2 : z = (b + (b + 2 * (m - 1))) / 2) :
  b + (b + 2 * (m - 1)) = 2 * z :=
sorry

end sum_smallest_largest_eq_2z_l258_258200


namespace sqrt_eight_plus_n_eq_nine_l258_258407

theorem sqrt_eight_plus_n_eq_nine (n : ℕ) (h : sqrt (8 + n) = 9) : n = 73 := by
  sorry

end sqrt_eight_plus_n_eq_nine_l258_258407


namespace remainder_div_power10_l258_258474

theorem remainder_div_power10 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, (10^n - 1) % 37 = k^2 := by
  sorry

end remainder_div_power10_l258_258474


namespace polygon_has_area_144_l258_258294

noncomputable def polygonArea (n_sides : ℕ) (perimeter : ℕ) (n_squares : ℕ) : ℕ :=
  let s := perimeter / n_sides
  let square_area := s * s
  square_area * n_squares

theorem polygon_has_area_144 :
  polygonArea 32 64 36 = 144 :=
by
  sorry

end polygon_has_area_144_l258_258294


namespace factorial_floor_value_l258_258129

noncomputable def compute_expression : ℝ :=
  (Nat.factorial 2010 + Nat.factorial 2006) / (Nat.factorial 2009 + Nat.factorial 2008)

theorem factorial_floor_value : 
  ⌊compute_expression⌋ = 2009 := by
  sorry

end factorial_floor_value_l258_258129


namespace largest_square_tile_for_board_l258_258473

theorem largest_square_tile_for_board (length width gcd_val : ℕ) (h1 : length = 16) (h2 : width = 24) 
  (h3 : gcd_val = Int.gcd length width) : gcd_val = 8 := by
  sorry

end largest_square_tile_for_board_l258_258473


namespace nth_wise_number_1990_l258_258812

/--
A natural number that can be expressed as the difference of squares 
of two other natural numbers is called a "wise number".
-/
def is_wise_number (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 - y^2 = n

/--
The 1990th "wise number" is 2659.
-/
theorem nth_wise_number_1990 : ∃ n : ℕ, is_wise_number n ∧ n = 2659 :=
  sorry

end nth_wise_number_1990_l258_258812


namespace polar_bear_daily_food_l258_258679

-- Definitions based on the conditions
def bucketOfTroutDaily : ℝ := 0.2
def bucketOfSalmonDaily : ℝ := 0.4

-- The proof statement
theorem polar_bear_daily_food : bucketOfTroutDaily + bucketOfSalmonDaily = 0.6 := by
  sorry

end polar_bear_daily_food_l258_258679


namespace max_n_value_l258_258001

noncomputable def max_n_avoid_repetition : ℕ :=
sorry

theorem max_n_value : max_n_avoid_repetition = 155 :=
by
  -- Assume factorial reciprocals range from 80 to 99
  -- We show no n-digit segments are repeated in such range while n <= 155
  sorry

end max_n_value_l258_258001


namespace range_of_m_l258_258589

open Real

theorem range_of_m (a m y1 y2 : ℝ) (h_a_pos : a > 0)
  (hA : y1 = a * (m - 1)^2 + 4 * a * (m - 1) + 3)
  (hB : y2 = a * m^2 + 4 * a * m + 3)
  (h_y1_lt_y2 : y1 < y2) : 
  m > -3 / 2 := 
sorry

end range_of_m_l258_258589


namespace evaluate_at_minus_two_l258_258569

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem evaluate_at_minus_two : f (-2) = -1 := 
by 
  unfold f 
  sorry

end evaluate_at_minus_two_l258_258569


namespace length_of_second_platform_l258_258665

theorem length_of_second_platform (train_length first_platform_length : ℕ) (time_to_cross_first_platform time_to_cross_second_platform : ℕ) 
  (H1 : train_length = 110) (H2 : first_platform_length = 160) (H3 : time_to_cross_first_platform = 15) 
  (H4 : time_to_cross_second_platform = 20) : ∃ second_platform_length, second_platform_length = 250 := 
by
  sorry

end length_of_second_platform_l258_258665


namespace prob_abs_le_1_96_l258_258338

open ProbabilityTheory MeasureTheory

noncomputable def stdNormalDist : ProbabilityMeasure ℝ :=
  ProbMeasure.stdNormal

theorem prob_abs_le_1_96 :
  ∀ (ξ : ℝ →ₘ[stdNormalDist] ℝ), -- Random variable ξ follows a standard normal distribution
  (prob (ξ ≤ -1.96) stdNormalDist = 0.025) →
  (prob (|ξ| < 1.96) stdNormalDist = 0.950) :=
sorry

end prob_abs_le_1_96_l258_258338


namespace find_side_b_in_triangle_l258_258169

theorem find_side_b_in_triangle 
  (A B : ℝ) (a : ℝ)
  (h_cosA : Real.cos A = -1/2)
  (h_B : B = Real.pi / 4)
  (h_a : a = 3) :
  ∃ b, b = Real.sqrt 6 :=
by
  sorry

end find_side_b_in_triangle_l258_258169


namespace standard_equation_of_circle_l258_258343

-- Definitions based on problem conditions
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := 2

-- Lean statement of the problem
theorem standard_equation_of_circle :
  ∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = radius ^ 2 ↔ (x + 1)^2 + (y - 2)^2 = 4 :=
by sorry

end standard_equation_of_circle_l258_258343


namespace range_of_a_l258_258577

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) : (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l258_258577


namespace remaining_pie_l258_258533

theorem remaining_pie (carlos_take: ℝ) (sophia_share : ℝ) (final_remaining : ℝ) :
  carlos_take = 0.6 ∧ sophia_share = (1 - carlos_take) / 4 ∧ final_remaining = (1 - carlos_take) - sophia_share →
  final_remaining = 0.3 :=
by
  intros h
  sorry

end remaining_pie_l258_258533


namespace six_times_expression_l258_258711

theorem six_times_expression {x y Q : ℝ} (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q :=
by
  sorry

end six_times_expression_l258_258711


namespace movie_store_additional_movie_needed_l258_258523

theorem movie_store_additional_movie_needed (movies shelves : ℕ) (h_movies : movies = 999) (h_shelves : shelves = 5) : 
  (shelves - (movies % shelves)) % shelves = 1 :=
by
  sorry

end movie_store_additional_movie_needed_l258_258523


namespace apples_handed_out_l258_258030

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

end apples_handed_out_l258_258030


namespace tickets_needed_l258_258729

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l258_258729


namespace sin_cos_y_range_l258_258710

theorem sin_cos_y_range (x y : ℝ) (hx : 0 < x) (hπx : x < π / 2) (hy : 0 < y) (hπy : y < π / 2)
    (h : sin x = x * cos y) : x / 2 < y ∧ y < x :=
by
  sorry

end sin_cos_y_range_l258_258710


namespace part1_a2_part1_a3_part2_general_formula_l258_258269

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| n + 1 => (n + 1) * n / 2

noncomputable def S (n : ℕ) : ℚ := (n + 2) * a n / 3

theorem part1_a2 : a 2 = 3 := sorry

theorem part1_a3 : a 3 = 6 := sorry

theorem part2_general_formula (n : ℕ) (h : n > 0) : a n = n * (n + 1) / 2 := sorry

end part1_a2_part1_a3_part2_general_formula_l258_258269


namespace rectangle_width_l258_258590

theorem rectangle_width (w : ℝ) 
  (h1 : ∃ w : ℝ, w > 0 ∧ (2 * w + 2 * (w - 2)) = 16) 
  (h2 : ∀ w, w > 0 → 2 * w + 2 * (w - 2) = 16 → w = 5) : 
  w = 5 := 
sorry

end rectangle_width_l258_258590


namespace expression_divisible_by_9_for_any_int_l258_258757

theorem expression_divisible_by_9_for_any_int (a b : ℤ) : 9 ∣ ((3 * a + 2)^2 - (3 * b + 2)^2) := 
by 
  sorry

end expression_divisible_by_9_for_any_int_l258_258757


namespace probability_two_randomly_chosen_diagonals_intersect_l258_258595

def convex_hexagon_diagonals := 9
def main_diagonals := 3
def secondary_diagonals := 6
def intersections_from_main_diagonals := 3 * 4
def intersections_from_secondary_diagonals := 6 * 3
def total_unique_intersections := (intersections_from_main_diagonals + intersections_from_secondary_diagonals) / 2
def total_diagonal_pairs := convex_hexagon_diagonals * (convex_hexagon_diagonals - 1) / 2

theorem probability_two_randomly_chosen_diagonals_intersect :
  (total_unique_intersections / total_diagonal_pairs : ℚ) = 5 / 12 := sorry

end probability_two_randomly_chosen_diagonals_intersect_l258_258595


namespace smallest_prime_with_digit_sum_23_l258_258058

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l258_258058


namespace volume_of_pyramid_l258_258235

theorem volume_of_pyramid 
  (QR RS : ℝ) (PT : ℝ) 
  (hQR_pos : 0 < QR) (hRS_pos : 0 < RS) (hPT_pos : 0 < PT)
  (perp1 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * QR) * (x * y) = 0)
  (perp2 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * RS) * (x * y) = 0) :
  QR = 10 -> RS = 5 -> PT = 9 -> 
  (1/3) * QR * RS * PT = 150 :=
by
  sorry

end volume_of_pyramid_l258_258235


namespace find_a_l258_258863

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l258_258863


namespace seq_property_l258_258010

theorem seq_property (m : ℤ) (h1 : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - m * a n)
  (r s : ℕ)
  (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| :=
by
  sorry

end seq_property_l258_258010


namespace MrBrown_more_sons_or_daughters_probability_l258_258186

noncomputable def probability_more_sons_or_daughters : ℚ :=
  let total_outcomes := 2^8
  let balanced_cases := Nat.choose 8 4
  let favourable_cases := total_outcomes - balanced_cases
  favourable_cases / total_outcomes

theorem MrBrown_more_sons_or_daughters_probability :
  probability_more_sons_or_daughters = 93 / 128 := 
  sorry

end MrBrown_more_sons_or_daughters_probability_l258_258186


namespace solution_l258_258982

theorem solution (x y : ℝ) (h₁ : x + 3 * y = -1) (h₂ : x - 3 * y = 5) : x^2 - 9 * y^2 = -5 := 
by
  sorry

end solution_l258_258982


namespace sum_of_corners_10x10_l258_258319

theorem sum_of_corners_10x10 : 
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  (top_left + top_right + bottom_left + bottom_right) = 202 :=
by
  let top_left := 1
  let top_right := 10
  let bottom_left := 91
  let bottom_right := 100
  show top_left + top_right + bottom_left + bottom_right = 202
  sorry

end sum_of_corners_10x10_l258_258319


namespace calculate_expression_solve_quadratic_l258_258803

-- Problem 1
theorem calculate_expression (x : ℝ) (hx : x > 0) :
  (2 / 3) * Real.sqrt (9 * x) + 6 * Real.sqrt (x / 4) - x * Real.sqrt (1 / x) = 4 * Real.sqrt x :=
sorry

-- Problem 2
theorem solve_quadratic (x : ℝ) (h : x^2 - 4 * x + 1 = 0) :
  x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
sorry

end calculate_expression_solve_quadratic_l258_258803


namespace smallest_prime_with_digit_sum_23_l258_258069

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l258_258069


namespace find_a_for_even_l258_258696

def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

theorem find_a_for_even (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = f a x) ↔ a = 1 :=
by
  -- proof steps here
  sorry

end find_a_for_even_l258_258696


namespace min_possible_value_l258_258587

theorem min_possible_value (a b : ℤ) (h : a > b) :
  (∃ x : ℚ, x = (2 * a + 3 * b) / (a - 2 * b) ∧ (x + 1 / x = (2 : ℚ))) :=
sorry

end min_possible_value_l258_258587


namespace person_b_lap_time_l258_258750

noncomputable def lap_time_b (a_lap_time : ℕ) (meet_time : ℕ) : ℕ :=
  let combined_speed := 1 / meet_time
  let a_speed := 1 / a_lap_time
  let b_speed := combined_speed - a_speed
  1 / b_speed

theorem person_b_lap_time 
  (a_lap_time : ℕ) 
  (meet_time : ℕ) 
  (h1 : a_lap_time = 80) 
  (h2 : meet_time = 30) : 
  lap_time_b a_lap_time meet_time = 48 := 
by 
  rw [lap_time_b, h1, h2]
  -- Provided steps to solve the proof, skipped here only for statement
  sorry

end person_b_lap_time_l258_258750


namespace vectors_parallel_l258_258401

def are_parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

theorem vectors_parallel :
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  are_parallel a b :=
by
  let a := (1, 2, -2)
  let b := (-2, -4, 4)
  -- Proof omitted
  sorry

end vectors_parallel_l258_258401


namespace polynomial_modulus_l258_258183

-- Define complex modulus function
def cmod (z : ℂ) : ℝ := complex.abs z

-- Define polynomial P(z)
def P (a b c d z : ℂ) : ℂ := a * z^3 + b * z^2 + c * z + d

theorem polynomial_modulus (a b c d : ℂ) (h1 : cmod a = 1) (h2 : cmod b = 1) (h3 : cmod c = 1) (h4 : cmod d = 1) :
  ∃ (z : ℂ), cmod z = 1 ∧ cmod (P a b c d z) ≥ real.sqrt 6 :=
sorry

end polynomial_modulus_l258_258183


namespace maggi_ate_5_cupcakes_l258_258462

theorem maggi_ate_5_cupcakes
  (packages : ℕ)
  (cupcakes_per_package : ℕ)
  (left_cupcakes : ℕ)
  (total_cupcakes : ℕ := packages * cupcakes_per_package)
  (eaten_cupcakes : ℕ := total_cupcakes - left_cupcakes)
  (h1 : packages = 3)
  (h2 : cupcakes_per_package = 4)
  (h3 : left_cupcakes = 7) :
  eaten_cupcakes = 5 :=
by
  sorry

end maggi_ate_5_cupcakes_l258_258462


namespace intersection_is_correct_l258_258573

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_is_correct : M ∩ N = {1, 2} := 
by {
  sorry
}

end intersection_is_correct_l258_258573


namespace number_of_positions_forming_cube_with_missing_face_l258_258777

-- Define the polygon formed by 6 congruent squares in a cross shape
inductive Square
| center : Square
| top : Square
| bottom : Square
| left : Square
| right : Square

-- Define the indices for the additional square positions
inductive Position
| pos1 : Position
| pos2 : Position
| pos3 : Position
| pos4 : Position
| pos5 : Position
| pos6 : Position
| pos7 : Position
| pos8 : Position
| pos9 : Position
| pos10 : Position
| pos11 : Position

-- Define a function that takes a position and returns whether the polygon can form the missing-face cube
def can_form_cube_missing_face : Position → Bool
  | Position.pos1   => true
  | Position.pos2   => true
  | Position.pos3   => true
  | Position.pos4   => true
  | Position.pos5   => false
  | Position.pos6   => false
  | Position.pos7   => false
  | Position.pos8   => false
  | Position.pos9   => true
  | Position.pos10  => true
  | Position.pos11  => true

-- Count valid positions for forming the cube with one face missing
def count_valid_positions : Nat :=
  List.length (List.filter can_form_cube_missing_face 
    [Position.pos1, Position.pos2, Position.pos3, Position.pos4, Position.pos5, Position.pos6, Position.pos7, Position.pos8, Position.pos9, Position.pos10, Position.pos11])

-- Prove that the number of valid positions is 7
theorem number_of_positions_forming_cube_with_missing_face : count_valid_positions = 7 :=
  by
    -- Implementation of the proof
    sorry

end number_of_positions_forming_cube_with_missing_face_l258_258777


namespace moms_took_chocolates_l258_258013

theorem moms_took_chocolates (N : ℕ) (A : ℕ) (M : ℕ) : 
  N = 10 → 
  A = 3 * N →
  A - M = N + 15 →
  M = 5 :=
by
  intros h1 h2 h3
  sorry

end moms_took_chocolates_l258_258013


namespace burrs_count_l258_258367

variable (B T : ℕ)

theorem burrs_count 
  (h1 : T = 6 * B) 
  (h2 : B + T = 84) : 
  B = 12 := 
by
  sorry

end burrs_count_l258_258367


namespace total_estate_value_l258_258463

theorem total_estate_value :
  ∃ (E : ℝ), ∀ (x : ℝ),
    (5 * x + 4 * x = (2 / 3) * E) ∧
    (E = 13.5 * x) ∧
    (wife_share = 3 * 4 * x) ∧
    (gardener_share = 600) ∧
    (nephew_share = 1000) →
    E = 2880 := 
by 
  -- Declarations
  let E : ℝ := sorry
  let x : ℝ := sorry
  
  -- Set up conditions
  -- Daughter and son share
  have c1 : 5 * x + 4 * x = (2 / 3) * E := sorry
  
  -- E expressed through x
  have c2 : E = 13.5 * x := sorry
  
  -- Wife's share
  have c3 : wife_share = 3 * (4 * x) := sorry
  
  -- Gardener's share and Nephew's share
  have c4 : gardener_share = 600 := sorry
  have c5 : nephew_share = 1000 := sorry
  
  -- Equate expressions and solve
  have eq1 : E = 21 * x + 1600 := sorry
  have eq2 : E = 2880 := sorry
  use E
  intro x
  -- Prove the equalities under the given conditions
  sorry

end total_estate_value_l258_258463


namespace square_area_l258_258256

theorem square_area (side : ℕ) (h : side = 19) : side * side = 361 := by
  sorry

end square_area_l258_258256


namespace rearrange_rooks_possible_l258_258469

theorem rearrange_rooks_possible (board : Fin 8 × Fin 8 → Prop) (rooks : Fin 8 → Fin 8 × Fin 8) (painted : Fin 8 × Fin 8 → Prop) :
  (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) → -- no two rooks are in the same row or column
  (∃ (unpainted_count : ℕ), (unpainted_count = 64 - 27)) → -- 27 squares are painted red
  (∃ new_rooks : Fin 8 → Fin 8 × Fin 8,
    (∀ i : Fin 8, ¬painted (new_rooks i)) ∧ -- all rooks are on unpainted squares
    (∀ i j : Fin 8, i ≠ j → (new_rooks i).1 ≠ (new_rooks j).1 ∧ (new_rooks i).2 ≠ (new_rooks j).2) ∧ -- no two rooks are in the same row or column
    (∃ i : Fin 8, rooks i ≠ new_rooks i)) -- at least one rook has moved
:=
sorry

end rearrange_rooks_possible_l258_258469


namespace probability_odd_divisor_of_15_factorial_l258_258037

-- Define the factorial function
def fact : ℕ → ℕ
  | 0 => 1
  | (n+1) => (n+1) * fact n

-- Probability function for choosing an odd divisor
noncomputable def probability_odd_divisor (n : ℕ) : ℚ :=
  let prime_factors := [(2, 11), (3, 6), (5, 3), (7, 2), (11, 1), (13, 1)]
  let total_factors := prime_factors.foldr (λ p acc => (p.2 + 1) * acc) 1
  let odd_factors := ((prime_factors.filter (λ p => p.1 ≠ 2)).foldr (λ p acc => (p.2 + 1) * acc) 1)
  (odd_factors : ℚ) / (total_factors : ℚ)

-- Statement to prove the probability of an odd divisor
theorem probability_odd_divisor_of_15_factorial :
  probability_odd_divisor 15 = 1 / 12 :=
by
  -- Proof goes here, which is omitted as per the instructions
  sorry

end probability_odd_divisor_of_15_factorial_l258_258037


namespace exp_thirteen_pi_over_two_eq_i_l258_258131

theorem exp_thirteen_pi_over_two_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I := 
by
  sorry

end exp_thirteen_pi_over_two_eq_i_l258_258131


namespace num_real_solutions_system_l258_258135

theorem num_real_solutions_system :
  ∃! (num_solutions : ℕ), 
  num_solutions = 5 ∧
  ∃ x y z w : ℝ, 
    (x = z + w + x * z) ∧ 
    (y = w + x + y * w) ∧ 
    (z = x + y + z * x) ∧ 
    (w = y + z + w * z) :=
sorry

end num_real_solutions_system_l258_258135


namespace find_num_boys_l258_258941

-- Definitions for conditions
def num_children : ℕ := 13
def num_girls (num_boys : ℕ) : ℕ := num_children - num_boys

-- We will assume we have a predicate representing the truthfulness of statements.
-- boys tell the truth to boys and lie to girls
-- girls tell the truth to girls and lie to boys

theorem find_num_boys (boys_truth_to_boys : Prop) 
                      (boys_lie_to_girls : Prop) 
                      (girls_truth_to_girls : Prop) 
                      (girls_lie_to_boys : Prop)
                      (alternating_statements : Prop) : 
  ∃ (num_boys : ℕ), num_boys = 7 := 
  sorry

end find_num_boys_l258_258941


namespace remainder_3_pow_405_mod_13_l258_258792

theorem remainder_3_pow_405_mod_13 : (3^405) % 13 = 1 :=
by
  sorry

end remainder_3_pow_405_mod_13_l258_258792


namespace smallest_mu_ineq_l258_258846

theorem smallest_mu_ineq (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    a^2 + b^2 + c^2 + d^2 + 2 * a * d ≥ 2 * (a * b + b * c + c * d) := by {
    sorry
}

end smallest_mu_ineq_l258_258846


namespace part1_part2_l258_258985

variable (x y : ℤ) (A B : ℤ)

def A_def : ℤ := 3 * x^2 - 5 * x * y - 2 * y^2
def B_def : ℤ := x^2 - 3 * y

theorem part1 : A_def x y - 2 * B_def x y = x^2 - 5 * x * y - 2 * y^2 + 6 * y := by
  sorry

theorem part2 : A_def 2 (-1) - 2 * B_def 2 (-1) = 6 := by
  sorry

end part1_part2_l258_258985


namespace find_dividend_l258_258356

theorem find_dividend (D Q R dividend : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) (h4 : dividend = D * Q + R) :
  dividend = 5336 :=
by
  -- We will complete the proof using the provided conditions
  sorry

end find_dividend_l258_258356


namespace non_degenerate_ellipse_condition_l258_258828

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ x y : ℝ, 9 * x^2 + y^2 - 18 * x - 2 * y = k) ↔ k > -10 :=
sorry

end non_degenerate_ellipse_condition_l258_258828


namespace employees_6_or_more_percentage_is_18_l258_258923

-- Defining the employee counts for different year ranges
def count_less_than_1 (y : ℕ) : ℕ := 4 * y
def count_1_to_2 (y : ℕ) : ℕ := 6 * y
def count_2_to_3 (y : ℕ) : ℕ := 7 * y
def count_3_to_4 (y : ℕ) : ℕ := 4 * y
def count_4_to_5 (y : ℕ) : ℕ := 3 * y
def count_5_to_6 (y : ℕ) : ℕ := 3 * y
def count_6_to_7 (y : ℕ) : ℕ := 2 * y
def count_7_to_8 (y : ℕ) : ℕ := 2 * y
def count_8_to_9 (y : ℕ) : ℕ := y
def count_9_to_10 (y : ℕ) : ℕ := y

-- Sum of all employees T
def total_employees (y : ℕ) : ℕ := count_less_than_1 y + count_1_to_2 y + count_2_to_3 y +
                                    count_3_to_4 y + count_4_to_5 y + count_5_to_6 y +
                                    count_6_to_7 y + count_7_to_8 y + count_8_to_9 y +
                                    count_9_to_10 y

-- Employees with 6 years or more E
def employees_6_or_more (y : ℕ) : ℕ := count_6_to_7 y + count_7_to_8 y + count_8_to_9 y + count_9_to_10 y

-- Calculate percentage
def percentage (y : ℕ) : ℚ := (employees_6_or_more y : ℚ) / (total_employees y : ℚ) * 100

-- Proving the final statement
theorem employees_6_or_more_percentage_is_18 (y : ℕ) (hy : y ≠ 0) : percentage y = 18 :=
by
  sorry

end employees_6_or_more_percentage_is_18_l258_258923


namespace money_left_after_purchase_l258_258824

def initial_toonies : Nat := 4
def value_per_toonie : Nat := 2
def total_coins : Nat := 10
def value_per_loonie : Nat := 1
def frappuccino_cost : Nat := 3

def toonies_value : Nat := initial_toonies * value_per_toonie
def loonies : Nat := total_coins - initial_toonies
def loonies_value : Nat := loonies * value_per_loonie
def initial_total : Nat := toonies_value + loonies_value
def remaining_money : Nat := initial_total - frappuccino_cost

theorem money_left_after_purchase : remaining_money = 11 := by
  sorry

end money_left_after_purchase_l258_258824


namespace parabola_transformation_zeros_sum_l258_258775

theorem parabola_transformation_zeros_sum :
  let y := fun x => (x - 3)^2 + 4
  let y_rotated := fun x => -(x - 3)^2 + 4
  let y_shifted_right := fun x => -(x - 7)^2 + 4
  let y_final := fun x => -(x - 7)^2 + 7
  ∃ a b, y_final a = 0 ∧ y_final b = 0 ∧ (a + b) = 14 :=
by
  sorry

end parabola_transformation_zeros_sum_l258_258775


namespace find_x_l258_258516

theorem find_x (x : ℤ) (h : 3 * x = (26 - x) + 10) : x = 9 :=
by
  -- proof steps would be provided here
  sorry

end find_x_l258_258516


namespace intersection_points_l258_258479

theorem intersection_points (x y : ℝ) (h1 : x^2 - 4 * y^2 = 4) (h2 : x = 3 * y) : 
  (x, y) = (3, 1) ∨ (x, y) = (-3, -1) :=
sorry

end intersection_points_l258_258479


namespace total_points_correct_l258_258625

structure PaperRecycling where
  white_paper_points : ℚ
  colored_paper_points : ℚ

def paige_paper : PaperRecycling := {
  white_paper_points := (12 / 6) * 2,
  colored_paper_points := (18 / 8) * 3
}

def alex_paper : PaperRecycling := {
  white_paper_points := (⟨26, 6, by norm_num⟩.num / ⟨26, 6, by norm_num⟩.den) * 2,
  colored_paper_points := (⟨10, 8, by norm_num⟩.num / ⟨10, 8, by norm_num⟩.den) * 3
}

def jordan_paper : PaperRecycling := {
  white_paper_points := (30 / 6) * 2,
  colored_paper_points := 0
}

def total_points : ℚ :=
  paige_paper.white_paper_points + paige_paper.colored_paper_points +
  alex_paper.white_paper_points + alex_paper.colored_paper_points +
  jordan_paper.white_paper_points + jordan_paper.colored_paper_points

theorem total_points_correct : total_points = 31 := by
  sorry

end total_points_correct_l258_258625


namespace running_speed_equiv_l258_258660

variable (R : ℝ)
variable (walking_speed : ℝ) (total_distance : ℝ) (total_time: ℝ) (distance_walked : ℝ) (distance_ran : ℝ)

theorem running_speed_equiv :
  walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4 →
  1 + (4 / R) = 1.5 →
  R = 8 :=
by
  intros H1 H2
  -- H1: Condition set (walking_speed = 4 ∧ total_distance = 8 ∧ total_time = 1.5 ∧ distance_walked = 4 ∧ distance_ran = 4)
  -- H2: Equation (1 + (4 / R) = 1.5)
  sorry

end running_speed_equiv_l258_258660


namespace xy_leq_half_x_squared_plus_y_squared_l258_258906

theorem xy_leq_half_x_squared_plus_y_squared (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := 
by 
  sorry

end xy_leq_half_x_squared_plus_y_squared_l258_258906


namespace radius_of_spherical_circle_correct_l258_258640

noncomputable def radius_of_spherical_circle (rho theta phi : ℝ) : ℝ :=
  if rho = 1 ∧ phi = Real.pi / 4 then Real.sqrt 2 / 2 else 0

theorem radius_of_spherical_circle_correct :
  ∀ (theta : ℝ), radius_of_spherical_circle 1 theta (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end radius_of_spherical_circle_correct_l258_258640


namespace field_trip_people_per_bus_l258_258810

def number_of_people_on_each_bus (vans buses people_per_van total_people : ℕ) : ℕ :=
  (total_people - (vans * people_per_van)) / buses

theorem field_trip_people_per_bus :
  let vans := 9
  let buses := 10
  let people_per_van := 8
  let total_people := 342
  number_of_people_on_each_bus vans buses people_per_van total_people = 27 :=
by
  sorry

end field_trip_people_per_bus_l258_258810


namespace solve_for_x_l258_258580

-- Define the conditions as mathematical statements in Lean
def conditions (x y : ℝ) : Prop :=
  (2 * x - 3 * y = 10) ∧ (y = -x)

-- State the theorem that needs to be proven
theorem solve_for_x : ∃ x : ℝ, ∃ y : ℝ, conditions x y ∧ x = 2 :=
by 
  -- Provide a sketch of the proof to show that the statement is well-formed
  sorry

end solve_for_x_l258_258580


namespace probability_of_event_A_eq_five_fourteenth_l258_258594

noncomputable theory

open_locale classical 
open_locale big_operators 

def probability_event_exactly_two_students_from_same_school (n m : ℕ) : ℚ :=
  let total_ways := nat.choose 10 4 in
  let ways_event_A := nat.choose 5 1 * nat.choose 2 2 * nat.choose 8 2 in
  ways_event_A / total_ways

theorem probability_of_event_A_eq_five_fourteenth :
  probability_event_exactly_two_students_from_same_school 4 10 = 5/14 :=
by sorry

end probability_of_event_A_eq_five_fourteenth_l258_258594


namespace local_minimum_condition_l258_258166

-- Define the function f(x)
def f (x b : ℝ) : ℝ := x ^ 3 - 3 * b * x + 3 * b

-- Define the first derivative of f(x)
def f_prime (x b : ℝ) : ℝ := 3 * x ^ 2 - 3 * b

-- Define the second derivative of f(x)
def f_double_prime (x b : ℝ) : ℝ := 6 * x

-- Theorem stating that f(x) has a local minimum if and only if b > 0
theorem local_minimum_condition (b : ℝ) (x : ℝ) (h : f_prime x b = 0) : f_double_prime x b > 0 ↔ b > 0 :=
by sorry

end local_minimum_condition_l258_258166


namespace inequality_direction_change_l258_258242

theorem inequality_direction_change :
  ∃ (a b c : ℝ), (a < b) ∧ (c < 0) ∧ (a * c > b * c) :=
by
  sorry

end inequality_direction_change_l258_258242


namespace smallest_prime_with_digit_sum_23_l258_258093

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l258_258093


namespace part_a_l258_258550

noncomputable def S : ℕ → ℕ
| 0       := 1
| (n + 1) := sorry  -- To be defined according to the given recurrence relation.

axiom S_recurrence : ∀ n, S n.succ = S n + n * S (n - 1)

theorem part_a (n : ℕ) : S (n + 1) = S n + n * S (n - 1) :=
  S_recurrence n

end part_a_l258_258550


namespace find_theta_l258_258137

theorem find_theta (θ : ℝ) :
  (0 : ℝ) ≤ θ ∧ θ ≤ 2 * Real.pi →
  (∀ x, (0 : ℝ) ≤ x ∧ x ≤ 2 →
    x^2 * Real.cos θ - 2 * x * (1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intros hθ hx
  sorry

end find_theta_l258_258137


namespace smallest_prime_with_digit_sum_23_l258_258064

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l258_258064


namespace find_BA_prime_l258_258811

theorem find_BA_prime (BA BC A_prime C_1 : ℝ) 
  (h1 : BA = 3)
  (h2 : BC = 2)
  (h3 : A_prime < BA)
  (h4 : A_prime * C_1 = 3) : A_prime = 3 / 2 := 
by 
  sorry

end find_BA_prime_l258_258811


namespace simplify_expression_l258_258478

theorem simplify_expression (x : ℝ) : (5 * x + 2 * x + 7 * x) = 14 * x :=
by
  sorry

end simplify_expression_l258_258478


namespace color_of_face_opposite_blue_l258_258206

/-- Assume we have a cube with each face painted in distinct colors. -/
structure Cube where
  top : String
  front : String
  right_side : String
  back : String
  left_side : String
  bottom : String

/-- Given three views of a colored cube, determine the color of the face opposite the blue face. -/
theorem color_of_face_opposite_blue (c : Cube)
  (h_top : c.top = "R")
  (h_right : c.right_side = "G")
  (h_view1 : c.front = "W")
  (h_view2 : c.front = "O")
  (h_view3 : c.front = "Y") :
  c.back = "Y" :=
sorry

end color_of_face_opposite_blue_l258_258206


namespace range_of_a_l258_258276

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2 * Real.log x
noncomputable def h (x : ℝ) : ℝ := x^2 - 2 * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x y, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (1 / Real.exp 1) ≤ y ∧ y ≤ Real.exp 1 ∧ f x a = g x ∧ f y a = g y → x ≠ y) →
  1 < a ∧ a ≤ (1 / Real.exp 2) + 2 :=
sorry

end range_of_a_l258_258276


namespace math_problem_l258_258351

theorem math_problem (a b : ℕ) (h₁ : a = 6) (h₂ : b = 6) : 
  (a^3 + b^3) / (a^2 - a * b + b^2) = 12 :=
by
  sorry

end math_problem_l258_258351


namespace factorize_def_l258_258215

def factorize_polynomial (p q r : Polynomial ℝ) : Prop :=
  p = q * r

theorem factorize_def (p q r : Polynomial ℝ) :
  factorize_polynomial p q r → p = q * r :=
  sorry

end factorize_def_l258_258215


namespace calc_f_g_h_2_l258_258008

def f (x : ℕ) : ℕ := x + 5
def g (x : ℕ) : ℕ := x^2 - 8
def h (x : ℕ) : ℕ := 2 * x + 1

theorem calc_f_g_h_2 : f (g (h 2)) = 22 := by
  sorry

end calc_f_g_h_2_l258_258008


namespace number_of_elements_l258_258613

noncomputable def set_mean (S : Set ℝ) : ℝ := sorry

theorem number_of_elements (S : Set ℝ) (M : ℝ)
  (h1 : set_mean (S ∪ {15}) = M + 2)
  (h2 : set_mean (S ∪ {15, 1}) = M + 1) :
  ∃ k : ℕ, (M * k + 15 = (M + 2) * (k + 1)) ∧ (M * k + 16 = (M + 1) * (k + 2)) ∧ k = 4 := sorry

end number_of_elements_l258_258613


namespace unique_sum_of_two_primes_l258_258447

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l258_258447


namespace find_triplets_l258_258461

theorem find_triplets (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a ∣ b + c + 1) (h5 : b ∣ c + a + 1) (h6 : c ∣ a + b + 1) :
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 2) ∨ (a, b, c) = (3, 4, 4) ∨ 
  (a, b, c) = (1, 1, 3) ∨ (a, b, c) = (2, 2, 5) :=
sorry

end find_triplets_l258_258461


namespace students_not_enrolled_in_bio_l258_258220

theorem students_not_enrolled_in_bio (total_students : ℕ) (p : ℕ) (p_half : p = (total_students / 2)) (total_students_eq : total_students = 880) : 
  total_students - p = 440 :=
by sorry

end students_not_enrolled_in_bio_l258_258220


namespace man_l258_258949

theorem man's_speed_downstream (v : ℝ) (speed_of_stream : ℝ) (speed_upstream : ℝ) : 
  speed_upstream = v - speed_of_stream ∧ speed_of_stream = 1.5 ∧ speed_upstream = 8 → v + speed_of_stream = 11 :=
by
  sorry

end man_l258_258949


namespace width_of_box_l258_258228

theorem width_of_box 
(length depth num_cubes : ℕ)
(h_length : length = 49)
(h_depth : depth = 14)
(h_num_cubes : num_cubes = 84)
: ∃ width : ℕ, width = 42 := 
sorry

end width_of_box_l258_258228


namespace smallest_prime_with_digit_sum_23_l258_258060

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l258_258060


namespace ratio_of_jumps_l258_258902

theorem ratio_of_jumps (run_ric: ℕ) (jump_ric: ℕ) (run_mar: ℕ) (extra_dist: ℕ)
    (h1 : run_ric = 20)
    (h2 : jump_ric = 4)
    (h3 : run_mar = 18)
    (h4 : extra_dist = 1) :
    (run_mar + extra_dist - run_ric - jump_ric) / jump_ric = 7 / 4 :=
by
  sorry

end ratio_of_jumps_l258_258902


namespace coordinates_of_point_with_respect_to_origin_l258_258031

theorem coordinates_of_point_with_respect_to_origin (P : ℝ × ℝ) (h : P = (-2, 4)) : P = (-2, 4) := 
by 
  exact h

end coordinates_of_point_with_respect_to_origin_l258_258031


namespace smallest_prime_with_digit_sum_23_l258_258063

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l258_258063


namespace lily_remaining_milk_l258_258315

def initial_milk : ℚ := (11 / 2)
def given_away : ℚ := (17 / 4)
def remaining_milk : ℚ := initial_milk - given_away

theorem lily_remaining_milk : remaining_milk = 5 / 4 :=
by
  -- Here, we would provide the proof steps, but we can use sorry to skip it.
  exact sorry

end lily_remaining_milk_l258_258315


namespace total_seats_l258_258119

theorem total_seats (s : ℕ) 
  (h1 : 30 + (0.20 * s : ℝ) + (0.60 * s : ℝ) = s) : s = 150 :=
  sorry

end total_seats_l258_258119


namespace longest_tape_length_l258_258106

theorem longest_tape_length {a b c : ℕ} (h1 : a = 100) (h2 : b = 225) (h3 : c = 780) : 
  Int.gcd (Int.gcd a b) c = 5 := by
  sorry

end longest_tape_length_l258_258106


namespace train_speed_l258_258115

noncomputable def train_speed_kmph (L_t L_b : ℝ) (T : ℝ) : ℝ :=
  (L_t + L_b) / T * 3.6

theorem train_speed (L_t L_b : ℝ) (T : ℝ) :
  L_t = 110 ∧ L_b = 190 ∧ T = 17.998560115190784 → train_speed_kmph L_t L_b T = 60 :=
by
  intro h
  sorry

end train_speed_l258_258115


namespace percent_of_y_l258_258592

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 * y) / 20 + (3 * y) / 10) = (35/100) * y :=
by
  sorry

end percent_of_y_l258_258592


namespace integer_solutions_no_solutions_2891_l258_258016

-- Define the main problem statement
-- Prove that if the equation x^3 - 3xy^2 + y^3 = n has a solution in integers x, y, then it has at least three such solutions.
theorem integer_solutions (n : ℕ) (x y : ℤ) (h : x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x₁ y₁ x₂ y₂ : ℤ, x₁ ≠ x ∧ y₁ ≠ y ∧ x₂ ≠ x ∧ y₂ ≠ y ∧ 
  x₁^3 - 3 * x₁ * y₁^2 + y₁^3 = n ∧ 
  x₂^3 - 3 * x₂ * y₂^2 + y₂^3 = n := sorry

-- Prove that if n = 2891 then no such integer solutions exist.
theorem no_solutions_2891 (x y : ℤ) : ¬ (x^3 - 3 * x * y^2 + y^3 = 2891) := sorry

end integer_solutions_no_solutions_2891_l258_258016


namespace fraction_eq_l258_258866

def f(x : ℤ) : ℤ := 3 * x + 2
def g(x : ℤ) : ℤ := 2 * x - 3

theorem fraction_eq : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by 
  sorry

end fraction_eq_l258_258866


namespace arithmetic_sequence_sum_l258_258154

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : a 8 + a 10 = 2) : 
  (17 * (a 1 + a 17) / 2) = 17 := by
sorry

end arithmetic_sequence_sum_l258_258154


namespace jordan_rectangle_width_l258_258219

theorem jordan_rectangle_width (length_carol width_carol length_jordan width_jordan : ℝ)
  (h1: length_carol = 15) (h2: width_carol = 20) (h3: length_jordan = 6)
  (area_equal: length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 50 :=
by
  sorry

end jordan_rectangle_width_l258_258219


namespace geometric_sequence_problem_l258_258293

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Geometric sequence definition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
def condition_1 : Prop := a 5 * a 8 = 6
def condition_2 : Prop := a 3 + a 10 = 5

-- Concluded value of q^7
def q_seven (q : ℝ) (a : ℕ → ℝ) : Prop := 
  q^7 = a 20 / a 13

theorem geometric_sequence_problem
  (h1 : is_geometric_sequence a q)
  (h2 : condition_1 a)
  (h3 : condition_2 a) :
  q_seven q a = (q = 3/2) ∨ (q = 2/3) :=
sorry

end geometric_sequence_problem_l258_258293


namespace diane_total_loss_l258_258836

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end diane_total_loss_l258_258836


namespace sum_of_midpoint_coords_l258_258490

theorem sum_of_midpoint_coords (x1 y1 x2 y2 : ℝ) (hx1 : x1 = 3) (hy1 : y1 = 5) (hx2 : x2 = 11) (hy2 : y2 = 21) :
  ((x1 + x2) / 2 + (y1 + y2) / 2) = 20 :=
by
  sorry

end sum_of_midpoint_coords_l258_258490


namespace incorrect_statement_B_l258_258411

def two_times_root_equation (a b c x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x1 = 2 * x2 ∨ x2 = 2 * x1)

theorem incorrect_statement_B (m n : ℝ) (h : (x - 2) * (m * x + n) = 0) :
  ¬(two_times_root_equation 1 (-m+n) (-mn) 2 (-n / m) -> m + n = 0) :=
sorry

end incorrect_statement_B_l258_258411


namespace bailey_dog_treats_l258_258529

-- Definitions based on conditions
def total_charges_per_card : Nat := 5
def number_of_cards : Nat := 4
def chew_toys : Nat := 2
def rawhide_bones : Nat := 10

-- Total number of items bought
def total_items : Nat := total_charges_per_card * number_of_cards

-- Definition of the number of dog treats
def dog_treats : Nat := total_items - (chew_toys + rawhide_bones)

-- Theorem to prove the number of dog treats
theorem bailey_dog_treats : dog_treats = 8 := by
  -- Proof is skipped with sorry
  sorry

end bailey_dog_treats_l258_258529


namespace digit_positions_in_8008_l258_258038

theorem digit_positions_in_8008 :
  (8008 % 10 = 8) ∧ (8008 / 1000 % 10 = 8) :=
by
  sorry

end digit_positions_in_8008_l258_258038


namespace number_of_pairs_l258_258758

theorem number_of_pairs (f m : ℕ) (n : ℕ) :
  n = 6 →
  (f + m ≤ n) →
  ∃! pairs : ℕ, pairs = 2 :=
by
  intro h1 h2
  sorry

end number_of_pairs_l258_258758


namespace shadow_length_false_if_approaching_lamp_at_night_l258_258825

theorem shadow_length_false_if_approaching_lamp_at_night
  (night : Prop)
  (approaches_lamp : Prop)
  (shadow_longer : Prop) :
  night → approaches_lamp → ¬shadow_longer :=
by
  -- assume it is night and person is approaching lamp
  intros h_night h_approaches
  -- proof is omitted
  sorry

end shadow_length_false_if_approaching_lamp_at_night_l258_258825


namespace change_calculation_l258_258525

/-!
# Problem
Adam has $5 to buy an airplane that costs $4.28. How much change will he get after buying the airplane?

# Conditions
Adam has $5.
The airplane costs $4.28.

# Statement
Prove that the change Adam will get is $0.72.
-/

theorem change_calculation : 
  let amount := 5.00
  let cost := 4.28
  let change := 0.72
  amount - cost = change :=
by 
  sorry

end change_calculation_l258_258525


namespace number_of_prime_pairs_for_10003_l258_258429

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l258_258429


namespace range_of_quadratic_function_l258_258638

noncomputable def quadratic_function_range : Set ℝ :=
  { y : ℝ | ∃ x : ℝ, y = x^2 - 6 * x + 7 }

theorem range_of_quadratic_function :
  quadratic_function_range = { y : ℝ | y ≥ -2 } :=
by
  -- Insert proof here
  sorry

end range_of_quadratic_function_l258_258638


namespace distinct_intersection_points_l258_258250

theorem distinct_intersection_points : 
  ∃! (x y : ℝ), (x + 2*y = 6 ∧ x - 3*y = 2) ∨ (x + 2*y = 6 ∧ 4*x + y = 14) :=
by
  -- proof would be here
  sorry

end distinct_intersection_points_l258_258250


namespace sum_first_six_terms_geometric_seq_l258_258288

theorem sum_first_six_terms_geometric_seq (a r : ℝ)
  (h1 : a + a * r = 12)
  (h2 : a + a * r + a * r^2 + a * r^3 = 36) :
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 84 :=
sorry

end sum_first_six_terms_geometric_seq_l258_258288


namespace min_kinder_surprises_l258_258678

theorem min_kinder_surprises (gnomes : Finset ℕ) (hs: gnomes.card = 12) :
  ∃ k, k ≤ 166 ∧ ∀ kinder_surprises : Finset (Finset ℕ), kinder_surprises.card = k → 
  (∀ s ∈ kinder_surprises, s.card = 3 ∧ s ⊆ gnomes ∧ (∀ t ∈ kinder_surprises, s ≠ t → s ≠ t)) → 
  ∀ g ∈ gnomes, ∃ s ∈ kinder_surprises, g ∈ s :=
sorry

end min_kinder_surprises_l258_258678


namespace max_value_g_l258_258611

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * (100 - x)) + Real.sqrt (x * (4 - x))

theorem max_value_g : ∃ (x₁ N : ℝ), (0 ≤ x₁ ∧ x₁ ≤ 4) ∧ (N = 16) ∧ (x₁ = 2) ∧ (∀ x, 0 ≤ x ∧ x ≤ 4 → g x ≤ N) :=
by
  sorry

end max_value_g_l258_258611


namespace lowest_possible_score_l258_258673

def total_points_first_four_tests : ℕ := 82 + 90 + 78 + 85
def required_total_points_for_seven_tests : ℕ := 80 * 7
def points_needed_for_last_three_tests : ℕ :=
  required_total_points_for_seven_tests - total_points_first_four_tests

theorem lowest_possible_score 
  (max_points_per_test : ℕ)
  (points_first_four_tests : ℕ := total_points_first_four_tests)
  (required_points : ℕ := required_total_points_for_seven_tests)
  (total_points_needed_last_three : ℕ := points_needed_for_last_three_tests) :
  ∃ (lowest_score : ℕ), 
    max_points_per_test = 100 ∧
    points_first_four_tests = 335 ∧
    required_points = 560 ∧
    total_points_needed_last_three = 225 ∧
    lowest_score = 25 :=
by
  sorry

end lowest_possible_score_l258_258673


namespace smallest_prime_with_digit_sum_23_l258_258074

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l258_258074


namespace correct_propositions_l258_258324

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_propositions :
  ¬ ∀ (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 → ∃ k : ℤ, x1 - x2 = k * Real.pi ∧
  (∀ (x : ℝ), f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (f (- (Real.pi / 6)) = 0) ∧
  ¬ ∀ (x : ℝ), f x = f (-x - Real.pi / 6) :=
sorry

end correct_propositions_l258_258324


namespace unique_solution_for_log_problem_l258_258995

noncomputable def log_problem (x : ℝ) :=
  let a := Real.log (x / 2 - 1) / Real.log (x - 11 / 4).sqrt
  let b := 2 * Real.log (x - 11 / 4) / Real.log (x / 2 - 1 / 4)
  let c := Real.log (x / 2 - 1 / 4) / (2 * Real.log (x / 2 - 1))
  a * b * c = 2 ∧ (a = b ∧ c = a + 1)

theorem unique_solution_for_log_problem :
  ∃! x, log_problem x = true := sorry

end unique_solution_for_log_problem_l258_258995


namespace smallest_prime_with_digit_sum_23_l258_258096

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l258_258096


namespace expression_increase_l258_258449

variable {x y : ℝ}

theorem expression_increase (hx : x > 0) (hy : y > 0) :
  let original_expr := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expr := 3 * new_x ^ 2 * new_y
  (new_expr / original_expr) = 3.456 :=
by
-- original_expr is 3 * x^2 * y
-- new_x = 1.2 * x
-- new_y = 2.4 * y
-- new_expr = 3 * (1.2 * x)^2 * (2.4 * y)
-- (new_expr / original_expr) = (10.368 * x^2 * y) / (3 * x^2 * y)
-- (new_expr / original_expr) = 10.368 / 3
-- (new_expr / original_expr) = 3.456
sorry

end expression_increase_l258_258449


namespace students_going_to_tournament_l258_258464

theorem students_going_to_tournament :
  ∀ (total_students : ℕ) (one_third : ℚ) (half : ℚ),
    total_students = 24 →
    one_third * total_students = 8 →
    half * 8 = 4 →
    4 = 4 :=
by
  intros total_students one_third half h1 h2 h3
  exact h3.symm

end students_going_to_tournament_l258_258464


namespace period_2_students_l258_258762

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l258_258762


namespace product_of_two_numbers_l258_258799
noncomputable def find_product (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : ℝ :=
x * y

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : find_product x y h1 h2 = 200 :=
sorry

end product_of_two_numbers_l258_258799


namespace smallest_prime_with_digit_sum_23_l258_258071

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l258_258071


namespace smallest_prime_digit_sum_23_l258_258078

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l258_258078


namespace connected_paper_area_l258_258025

def side_length := 30 -- side of each square paper in cm
def overlap_length := 7 -- overlap length in cm
def num_pieces := 6 -- number of paper pieces

def effective_length (side_length overlap_length : ℕ) := side_length - overlap_length
def total_connected_length (num_pieces : ℕ) (side_length overlap_length : ℕ) :=
  side_length + (num_pieces - 1) * (effective_length side_length overlap_length)

def width := side_length -- width of the connected paper is the side of each square piece of paper

def area (length width : ℕ) := length * width

theorem connected_paper_area : area (total_connected_length num_pieces side_length overlap_length) width = 4350 :=
by
  sorry

end connected_paper_area_l258_258025


namespace population_change_over_3_years_l258_258210

-- Define the initial conditions
def annual_growth_rate := 0.09
def migration_rate_year1 := -0.01
def migration_rate_year2 := -0.015
def migration_rate_year3 := -0.02
def natural_disaster_rate := -0.03

-- Lemma stating the overall percentage increase in population over three years
theorem population_change_over_3_years :
  (1 + annual_growth_rate) * (1 + migration_rate_year1) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year2) * 
  (1 + annual_growth_rate) * (1 + migration_rate_year3) * 
  (1 + natural_disaster_rate) = 1.195795 := 
sorry

end population_change_over_3_years_l258_258210


namespace deer_distribution_l258_258760

theorem deer_distribution :
  ∃ a : ℕ → ℚ,
    (a 1 + a 2 + a 3 + a 4 + a 5 = 5) ∧
    (a 4 = 2 / 3) ∧ 
    (a 3 = 1) ∧ 
    (a 1 = 5 / 3) :=
by
  sorry

end deer_distribution_l258_258760


namespace find_sum_of_a_and_d_l258_258390

theorem find_sum_of_a_and_d 
  {a b c d : ℝ} 
  (h1 : ab + ac + bd + cd = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 :=
sorry

end find_sum_of_a_and_d_l258_258390


namespace range_of_a_l258_258460

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) : (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 := by
  intro h
  sorry

end range_of_a_l258_258460


namespace average_age_of_John_Mary_Tonya_is_35_l258_258301

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l258_258301


namespace roots_equal_implies_a_eq_3_l258_258877

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l258_258877


namespace q_negative_one_is_minus_one_l258_258369

-- Define the function q and the point on the graph
def q (x : ℝ) : ℝ := sorry

-- The condition: point (-1, -1) lies on the graph of q
axiom point_on_graph : q (-1) = -1

-- The theorem to prove that q(-1) = -1
theorem q_negative_one_is_minus_one : q (-1) = -1 :=
by exact point_on_graph

end q_negative_one_is_minus_one_l258_258369


namespace average_marks_class_l258_258172

theorem average_marks_class (total_students : ℕ)
  (students_98 : ℕ) (score_98 : ℕ)
  (students_0 : ℕ) (score_0 : ℕ)
  (remaining_avg : ℝ)
  (h1 : total_students = 40)
  (h2 : students_98 = 6)
  (h3 : score_98 = 98)
  (h4 : students_0 = 9)
  (h5 : score_0 = 0)
  (h6 : remaining_avg = 57) :
  ( (( students_98 * score_98) + (students_0 * score_0) + ((total_students - students_98 - students_0) * remaining_avg)) / total_students ) = 50.325 :=
by 
  -- This is where the proof steps would go
  sorry

end average_marks_class_l258_258172


namespace volume_of_larger_prism_is_correct_l258_258548

noncomputable def volume_of_larger_solid : ℝ :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (2, 2, 0)
  let D := (0, 2, 0)
  let E := (0, 0, 2)
  let F := (2, 0, 2)
  let G := (2, 2, 2)
  let H := (0, 2, 2)
  let P := (1, 1, 1)
  let Q := (1, 0, 1)
  
  -- Assume the plane equation here divides the cube into equal halves
  -- Calculate the volume of one half of the cube
  let volume := 2 -- This represents the volume of the larger solid

  volume

theorem volume_of_larger_prism_is_correct :
  volume_of_larger_solid = 2 :=
sorry

end volume_of_larger_prism_is_correct_l258_258548


namespace shortest_distance_between_circles_l258_258218

def circle_eq1 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y - 15 = 0
def circle_eq2 (x y : ℝ) : Prop := x^2 + 10*x + y^2 + 12*y + 21 = 0

theorem shortest_distance_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ), circle_eq1 x1 y1 → circle_eq2 x2 y2 → 
  (abs ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) - (15^(1/2) + 82^(1/2))) =
  2 * 41^(1/2) - 97^(1/2) :=
by sorry

end shortest_distance_between_circles_l258_258218


namespace tan_alpha_expression_l258_258586

theorem tan_alpha_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 :=
by
  sorry

end tan_alpha_expression_l258_258586


namespace average_age_of_John_Mary_Tonya_is_35_l258_258299

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l258_258299


namespace original_pencils_count_l258_258784

theorem original_pencils_count (total_pencils : ℕ) (added_pencils : ℕ) (original_pencils : ℕ) : total_pencils = original_pencils + added_pencils → original_pencils = 2 :=
by
  sorry

end original_pencils_count_l258_258784


namespace initial_average_is_16_l258_258767

def average_of_six_observations (A : ℝ) : Prop :=
  ∃ s : ℝ, s = 6 * A

def new_observation (A : ℝ) (new_obs : ℝ := 9) : Prop :=
  ∃ t : ℝ, t = 7 * (A - 1)

theorem initial_average_is_16 (A : ℝ) (new_obs : ℝ := 9) :
  (average_of_six_observations A) → (new_observation A new_obs) → A = 16 :=
by
  intro h1 h2
  sorry

end initial_average_is_16_l258_258767


namespace ken_climbing_pace_l258_258192

noncomputable def sari_pace : ℝ := 350 -- Sari's pace in meters per hour, derived from 700 meters in 2 hours.

def ken_pace : ℝ := 500 -- We will need to prove this.

theorem ken_climbing_pace :
  let start_time_sari := 5
  let start_time_ken := 7
  let end_time_ken := 12
  let time_ken_climbs := end_time_ken - start_time_ken
  let sari_initial_headstart := 700 -- meters
  let sari_behind_ken := 50 -- meters
  let sari_total_climb := sari_pace * time_ken_climbs
  let total_distance_ken := sari_total_climb + sari_initial_headstart + sari_behind_ken
  ken_pace = total_distance_ken / time_ken_climbs :=
by
  sorry

end ken_climbing_pace_l258_258192


namespace four_c_plus_d_l258_258331

theorem four_c_plus_d (c d : ℝ) (h1 : 2 * c = -6) (h2 : c^2 - d = 1) : 4 * c + d = -4 :=
by
  sorry

end four_c_plus_d_l258_258331


namespace problem_remainder_1000th_in_S_mod_1000_l258_258457

def S : Nat → Prop :=
  λ n => Nat.bitcount n = 8

def N_1000th := Nat.find (Nat.gt_wf (λ n, S n))

theorem problem_remainder_1000th_in_S_mod_1000 :
  let N := N_1000th 1000
  N % 1000 = 32 :=
by
  sorry

end problem_remainder_1000th_in_S_mod_1000_l258_258457


namespace find_angle_A_find_area_triangle_l258_258719

-- Definitions for the triangle and the angles
def triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi

-- Given conditions
variables (a b c A B C : ℝ)
variables (hTriangle : triangle A B C)
variables (hEq : 2 * b * Real.cos A - Real.sqrt 3 * c * Real.cos A = Real.sqrt 3 * a * Real.cos C)
variables (hAngleB : B = Real.pi / 6)
variables (hMedianAM : Real.sqrt 7 = Real.sqrt (b^2 + (b / 2)^2 - 2 * b * (b / 2) * Real.cos (2 * Real.pi / 3)))

-- Proof statements
theorem find_angle_A : A = Real.pi / 6 :=
sorry

theorem find_area_triangle : (1/2) * b^2 * Real.sin C = Real.sqrt 3 :=
sorry

end find_angle_A_find_area_triangle_l258_258719


namespace find_m_l258_258735

-- Let m be a real number such that m > 1 and
-- \sum_{n=1}^{\infty} \frac{3n+2}{m^n} = 2.
theorem find_m (m : ℝ) (h1 : m > 1) 
(h2 : ∑' n : ℕ, (3 * (n + 1) + 2) / m^(n + 1) = 2) : 
  m = 3 :=
sorry

end find_m_l258_258735


namespace fraction_operations_l258_258677

theorem fraction_operations :
  let a := 1 / 3
  let b := 1 / 4
  let c := 1 / 2
  (a + b = 7 / 12) ∧ ((7 / 12) / c = 7 / 6) := by
{
  sorry
}

end fraction_operations_l258_258677


namespace gcd_p4_minus_1_eq_240_l258_258017

theorem gcd_p4_minus_1_eq_240 (p : ℕ) (hp : Prime p) (h_gt_5 : p > 5) :
  gcd (p^4 - 1) 240 = 240 :=
by sorry

end gcd_p4_minus_1_eq_240_l258_258017


namespace quadratic_no_real_roots_l258_258848

theorem quadratic_no_real_roots 
  (k : ℝ) 
  (h : ¬ ∃ (x : ℝ), 2 * x^2 + x - k = 0) : 
  k < -1/8 :=
by {
  -- Proof will go here.
  sorry
}

end quadratic_no_real_roots_l258_258848


namespace fran_speed_l258_258602

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l258_258602


namespace remainder_when_dividing_25197631_by_17_l258_258588

theorem remainder_when_dividing_25197631_by_17 :
  25197631 % 17 = 10 :=
by
  sorry

end remainder_when_dividing_25197631_by_17_l258_258588


namespace brandon_skittles_final_l258_258245
-- Conditions
def brandon_initial_skittles := 96
def brandon_lost_skittles := 9

-- Theorem stating the question and answer
theorem brandon_skittles_final : brandon_initial_skittles - brandon_lost_skittles = 87 := 
by
  -- Proof steps go here
  sorry

end brandon_skittles_final_l258_258245


namespace kirin_calculations_l258_258403

theorem kirin_calculations (calculations_per_second : ℝ) (seconds : ℝ) (h1 : calculations_per_second = 10^10) (h2 : seconds = 2022) : 
    calculations_per_second * seconds = 2.022 * 10^13 := 
by
  sorry

end kirin_calculations_l258_258403


namespace diana_took_six_candies_l258_258212

-- Define the initial number of candies in the box
def initial_candies : ℕ := 88

-- Define the number of candies left in the box after Diana took some
def remaining_candies : ℕ := 82

-- Define the number of candies taken by Diana
def candies_taken : ℕ := initial_candies - remaining_candies

-- The theorem we need to prove
theorem diana_took_six_candies : candies_taken = 6 := by
  sorry

end diana_took_six_candies_l258_258212


namespace tangent_line_at_point_l258_258205

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = Real.exp x - 2 * x) (h_point : (0, 1) = (x, y)) :
  x + y - 1 = 0 := 
by 
  sorry

end tangent_line_at_point_l258_258205


namespace value_of_p_l258_258886

theorem value_of_p (m n p : ℝ) (h₁ : m = 8 * n + 5) (h₂ : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by {
  sorry
}

end value_of_p_l258_258886


namespace incorrect_statements_A_D_l258_258994

variables {m x y : ℝ}

def line1 (m x y : ℝ) : Prop := (m + 2) * x + y + 1 = 0
def line2 (m x y : ℝ) : Prop := 3 * x + m * y + 4 * m - 3 = 0

theorem incorrect_statements_A_D (m : ℝ) : 
  (¬ ∃ (x y : ℝ), line1 (-3) x y ∧ line2 (-3) x y) ∧ 
  (¬ ∃ (d : ℝ), ∀ (x0 y0 : ℝ), x0 = 0 ∧ y0 = 0 → d = |(m + 2) * x0 + y0 + 1| / sqrt ((m + 2)^2 + 1^2) ∧ d = sqrt 17) :=
sorry

end incorrect_statements_A_D_l258_258994


namespace hexagon_interior_angle_Q_l258_258707

theorem hexagon_interior_angle_Q 
  (A B C D E F : ℕ)
  (hA : A = 135) (hB : B = 150) (hC : C = 120) (hD : D = 130) (hE : E = 100)
  (hex_angle_sum : A + B + C + D + E + F = 720) :
  F = 85 :=
by
  rw [hA, hB, hC, hD, hE] at hex_angle_sum
  sorry

end hexagon_interior_angle_Q_l258_258707


namespace gcd_1020_multiple_38962_l258_258145

-- Define that x is a multiple of 38962
def multiple_of (x n : ℤ) : Prop := ∃ k : ℤ, x = k * n

-- The main theorem statement
theorem gcd_1020_multiple_38962 (x : ℤ) (h : multiple_of x 38962) : Int.gcd 1020 x = 6 := 
sorry

end gcd_1020_multiple_38962_l258_258145


namespace dandelions_initial_l258_258213

theorem dandelions_initial (y w : ℕ) (h1 : y + w = 35) (h2 : y - 2 = 2 * (w - 6)) : y = 20 ∧ w = 15 :=
by
  sorry

end dandelions_initial_l258_258213


namespace arlo_books_l258_258368

theorem arlo_books (total_items : ℕ) (books_ratio : ℕ) (pens_ratio : ℕ) (notebooks_ratio : ℕ) 
  (ratio_sum : ℕ) (items_per_part : ℕ) (parts_for_books : ℕ) (total_parts : ℕ) :
  total_items = 600 →
  books_ratio = 7 →
  pens_ratio = 3 →
  notebooks_ratio = 2 →
  total_parts = books_ratio + pens_ratio + notebooks_ratio →
  items_per_part = total_items / total_parts →
  parts_for_books = books_ratio →
  parts_for_books * items_per_part = 350 := by
  intros
  sorry

end arlo_books_l258_258368


namespace find_Y_exists_l258_258903

variable {X : Finset ℕ} -- Consider a finite set X of natural numbers for generality
variable (S : Finset (Finset ℕ)) -- Set of all subsets of X with even number of elements
variable (f : Finset ℕ → ℝ) -- Real-valued function on subsets of X

-- Conditions
variable (hS : ∀ s ∈ S, s.card % 2 = 0) -- All elements in S have even number of elements
variable (h1 : ∃ A ∈ S, f A > 1990) -- f(A) > 1990 for some A ∈ S
variable (h2 : ∀ ⦃B C⦄, B ∈ S → C ∈ S → (Disjoint B C) → (f (B ∪ C) = f B + f C - 1990)) -- f respects the functional equation for disjoint subsets

theorem find_Y_exists :
  ∃ Y ⊆ X, (∀ D ∈ S, D ⊆ Y → f D > 1990) ∧ (∀ D ∈ S, D ⊆ (X \ Y) → f D ≤ 1990) :=
by
  sorry

end find_Y_exists_l258_258903


namespace triangle_AX_length_l258_258601

noncomputable def length_AX (AB AC BC : ℝ) (h1 : AB = 60) (h2 : AC = 34) (h3 : BC = 52) : ℝ :=
  1020 / 43

theorem triangle_AX_length 
  (AB AC BC AX : ℝ)
  (h1 : AB = 60)
  (h2 : AC = 34)
  (h3 : BC = 52)
  (h4 : AX + (AB - AX) = AB)
  (h5 : AX / (AB - AX) = AC / BC) :
  AX = 1020 / 43 := 
sorry

end triangle_AX_length_l258_258601


namespace find_product_of_roots_l258_258891

namespace ProductRoots

variables {k m : ℝ} {x1 x2 : ℝ}

theorem find_product_of_roots (h1 : x1 ≠ x2) 
    (hx1 : 5 * x1 ^ 2 - k * x1 = m) 
    (hx2 : 5 * x2 ^ 2 - k * x2 = m) : x1 * x2 = -m / 5 :=
sorry

end ProductRoots

end find_product_of_roots_l258_258891


namespace ten_faucets_fill_time_l258_258138

theorem ten_faucets_fill_time (rate : ℕ → ℕ → ℝ) (gallons : ℕ) (minutes : ℝ) :
  rate 5 9 = 150 / 5 ∧
  rate 10 135 = 75 / 30 * rate 10 9 / 0.9 * 60 →
  9 * 60 / 30 * 75 / 10 * 60 = 135 :=
sorry

end ten_faucets_fill_time_l258_258138


namespace number_of_people_study_only_cooking_l258_258721

def total_yoga : Nat := 25
def total_cooking : Nat := 18
def total_weaving : Nat := 10
def cooking_and_yoga : Nat := 5
def all_three : Nat := 4
def cooking_and_weaving : Nat := 5

theorem number_of_people_study_only_cooking :
  (total_cooking - (cooking_and_yoga + cooking_and_weaving - all_three)) = 12 :=
by
  sorry

end number_of_people_study_only_cooking_l258_258721


namespace polynomial_degree_one_condition_l258_258842

theorem polynomial_degree_one_condition (P : ℝ → ℝ) (c : ℝ) :
  (∀ a b : ℝ, a < b → (P = fun x => x + c) ∨ (P = fun x => -x + c)) ∧
  (∀ a b : ℝ, a < b →
    (max (P a) (P b) - min (P a) (P b) = b - a)) :=
sorry

end polynomial_degree_one_condition_l258_258842


namespace grasshopper_twenty_five_jumps_l258_258522

noncomputable def sum_natural (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem grasshopper_twenty_five_jumps :
  let total_distance := sum_natural 25
  total_distance % 2 = 1 -> 0 % 2 = 0 -> total_distance ≠ 0 :=
by
  intros total_distance_odd zero_even
  sorry

end grasshopper_twenty_five_jumps_l258_258522


namespace g_100_l258_258208

noncomputable def g (x : ℝ) : ℝ := sorry

lemma g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x * g y - y * g x = g (X^2 / y) := sorry

theorem g_100 : g 100 = 0 :=
begin
  sorry
end

end g_100_l258_258208


namespace period_2_students_l258_258761

theorem period_2_students (x : ℕ) (h1 : 2 * x - 5 = 11) : x = 8 :=
by {
  sorry
}

end period_2_students_l258_258761


namespace num_adult_tickets_l258_258785

theorem num_adult_tickets (adult_ticket_cost child_ticket_cost total_tickets_sold total_receipts : ℕ) 
  (h1 : adult_ticket_cost = 12) 
  (h2 : child_ticket_cost = 4) 
  (h3 : total_tickets_sold = 130) 
  (h4 : total_receipts = 840) :
  ∃ A C : ℕ, A + C = total_tickets_sold ∧ adult_ticket_cost * A + child_ticket_cost * C = total_receipts ∧ A = 40 :=
by {
  sorry
}

end num_adult_tickets_l258_258785


namespace sum_of_reciprocals_eq_six_l258_258040

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + y = 6 * x * y) (h2 : y = 2 * x) :
  (1 / x) + (1 / y) = 6 := by
  sorry

end sum_of_reciprocals_eq_six_l258_258040


namespace tangential_tetrahedron_triangle_impossibility_l258_258270

theorem tangential_tetrahedron_triangle_impossibility (a b c d : ℝ) 
  (h : ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x > 0) :
  ¬ (∀ (x y z : ℝ) , (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    (y = a ∨ y = b ∨ y = c ∨ y = d) →
    (z = a ∨ z = b ∨ z = c ∨ z = d) → 
    x ≠ y → y ≠ z → z ≠ x → x + y > z ∧ x + z > y ∧ y + z > x) :=
sorry

end tangential_tetrahedron_triangle_impossibility_l258_258270


namespace abs_neg_2023_l258_258482

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l258_258482


namespace find_constant_l258_258880

-- Define the variables: t, x, y, and the constant
variable (t x y constant : ℝ)

-- Conditions
def x_def : x = constant - 2 * t :=
  by sorry

def y_def : y = 2 * t - 2 :=
  by sorry

def x_eq_y_at_t : t = 0.75 → x = y :=
  by sorry

-- Proposition: Prove that the constant in the equation for x is 1
theorem find_constant (ht : t = 0.75) (hx : x = constant - 2 * t) (hy : y = 2 * t - 2) (he : x = y) :
  constant = 1 :=
  by sorry

end find_constant_l258_258880


namespace julia_cookies_l258_258935

theorem julia_cookies (N : ℕ) 
  (h1 : N % 6 = 5) 
  (h2 : N % 8 = 7) 
  (h3 : N < 100) : 
  N = 17 ∨ N = 41 ∨ N = 65 ∨ N = 89 → 17 + 41 + 65 + 89 = 212 :=
sorry

end julia_cookies_l258_258935


namespace people_sharing_cookies_l258_258103

theorem people_sharing_cookies (total_cookies : ℕ) (cookies_per_person : ℕ) (people : ℕ) 
  (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) (h3 : total_cookies = cookies_per_person * people) : 
  people = 6 :=
by
  sorry

end people_sharing_cookies_l258_258103


namespace combination_seven_four_l258_258539

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l258_258539


namespace num_articles_cost_price_l258_258874

theorem num_articles_cost_price (N C S : ℝ) (h1 : N * C = 50 * S) (h2 : (S - C) / C * 100 = 10) : N = 55 := 
sorry

end num_articles_cost_price_l258_258874


namespace A_wins_if_perfect_square_or_prime_l258_258572

theorem A_wins_if_perfect_square_or_prime (n : ℕ) (h_pos : 0 < n) : 
  (∃ A_wins : Bool, A_wins = true ↔ (∃ k : ℕ, n = k^2) ∨ (∃ p : ℕ, Nat.Prime p ∧ n = p)) :=
by
  sorry

end A_wins_if_perfect_square_or_prime_l258_258572


namespace pictures_vertically_l258_258742

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end pictures_vertically_l258_258742


namespace find_first_train_length_l258_258507

namespace TrainProblem

-- Define conditions
def speed_first_train_kmph := 42
def speed_second_train_kmph := 48
def length_second_train_m := 163
def time_clear_s := 12
def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph

-- Convert kmph to m/s
def kmph_to_mps(kmph : ℕ) : ℕ := kmph * 5 / 18
def relative_speed_mps := kmph_to_mps relative_speed_kmph

-- Calculate total distance covered by the trains in meters
def total_distance_m := relative_speed_mps * time_clear_s

-- Define the length of the first train to be proved
def length_first_train_m := 137

-- Theorem statement
theorem find_first_train_length :
  total_distance_m = length_first_train_m + length_second_train_m :=
sorry

end TrainProblem

end find_first_train_length_l258_258507


namespace pizza_slice_division_l258_258505

theorem pizza_slice_division : 
  ∀ (num_coworkers num_pizzas slices_per_pizza : ℕ),
  num_coworkers = 12 →
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  (num_pizzas * slices_per_pizza) / num_coworkers = 2 := 
by
  intros num_coworkers num_pizzas slices_per_pizza h_coworkers h_pizzas h_slices
  rw [h_coworkers, h_pizzas, h_slices]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slice_division_l258_258505


namespace expression_bounds_l258_258185

theorem expression_bounds (a b c d : ℝ) (h0a : 0 ≤ a) (h1a : a ≤ 1) (h0b : 0 ≤ b) (h1b : b ≤ 1)
  (h0c : 0 ≤ c) (h1c : c ≤ 1) (h0d : 0 ≤ d) (h1d : d ≤ 1) :
  2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ∧
    Real.sqrt (a^4 + (1 - b^2)^2) +
    Real.sqrt (b^4 + (c^2 - b^2)^2) +
    Real.sqrt (c^4 + (d^2 - c^2)^2) +
    Real.sqrt (d^4 + (a^2 - d^2)^2) ≤ 4 :=
by sorry

end expression_bounds_l258_258185


namespace same_root_a_eq_3_l258_258878

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l258_258878


namespace roots_of_varying_signs_l258_258253

theorem roots_of_varying_signs :
  (∃ x : ℝ, (4 * x^2 - 8 = 40 ∧ x != 0) ∧
           (∃ y : ℝ, (3 * y - 2)^2 = (y + 2)^2 ∧ y != 0) ∧
           (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z1 = 0 ∨ z2 = 0) ∧ x^3 - 8 * x^2 + 13 * x + 10 = 0)) :=
sorry

end roots_of_varying_signs_l258_258253


namespace number_of_prime_pairs_for_10003_l258_258427

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l258_258427


namespace abs_neg_2023_l258_258485

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l258_258485


namespace allowance_calculation_l258_258162

theorem allowance_calculation (A : ℝ)
  (h1 : (3 / 5) * A + (1 / 3) * (2 / 5) * A + 0.40 = A)
  : A = 1.50 :=
sorry

end allowance_calculation_l258_258162


namespace same_root_a_eq_3_l258_258879

theorem same_root_a_eq_3 {x a : ℝ} (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
by
  sorry

end same_root_a_eq_3_l258_258879


namespace problem_part_I_problem_part_II_l258_258990

-- Problem Part I
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_part_I (x : ℝ) :
    (f (x + 3/2) ≥ 0) ↔ (-2 ≤ x ∧ x ≤ 2) :=
  sorry

-- Problem Part II
theorem problem_part_II (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
    (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
    3*p + 2*q + r ≥ 9/4 :=
  sorry

end problem_part_I_problem_part_II_l258_258990


namespace same_color_probability_correct_l258_258226

noncomputable def prob_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green / total) * (green / total) +
  (red / total) * (red / total) +
  (blue / total) * (blue / total)

theorem same_color_probability_correct :
  prob_same_color 5 7 3 = 83 / 225 :=
by
  sorry

end same_color_probability_correct_l258_258226


namespace smallest_prime_with_digit_sum_23_proof_l258_258068

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l258_258068


namespace right_triangle_legs_sum_l258_258911

-- Definitions
def sum_of_legs (a b : ℕ) : ℕ := a + b

-- Main theorem statement
theorem right_triangle_legs_sum (x : ℕ) (h : x^2 + (x + 1)^2 = 53^2) :
  sum_of_legs x (x + 1) = 75 :=
sorry

end right_triangle_legs_sum_l258_258911


namespace egg_distribution_l258_258780

-- Definitions of the conditions
def total_eggs := 10.0
def large_eggs := 6.0
def small_eggs := 4.0

def box_A_capacity := 5.0
def box_B_capacity := 4.0
def box_C_capacity := 6.0

def at_least_one_small_egg (box_A_small box_B_small box_C_small : Float) := 
  box_A_small >= 1.0 ∧ box_B_small >= 1.0 ∧ box_C_small >= 1.0

-- Problem statement
theorem egg_distribution : 
  ∃ (box_A_small box_A_large box_B_small box_B_large box_C_small box_C_large : Float),
  box_A_small + box_A_large <= box_A_capacity ∧
  box_B_small + box_B_large <= box_B_capacity ∧
  box_C_small + box_C_large <= box_C_capacity ∧
  box_A_small + box_B_small + box_C_small = small_eggs ∧
  box_A_large + box_B_large + box_C_large = large_eggs ∧
  at_least_one_small_egg box_A_small box_B_small box_C_small :=
sorry

end egg_distribution_l258_258780


namespace tree_age_when_23_feet_l258_258645

theorem tree_age_when_23_feet (initial_age initial_height growth_rate final_height : ℕ) 
(h_initial_age : initial_age = 1)
(h_initial_height : initial_height = 5) 
(h_growth_rate : growth_rate = 3) 
(h_final_height : final_height = 23) : 
initial_age + (final_height - initial_height) / growth_rate = 7 := 
by sorry

end tree_age_when_23_feet_l258_258645


namespace no_prime_sum_10003_l258_258434

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l258_258434


namespace find_fx_l258_258334

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

theorem find_fx (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x : ℝ, x < 0 → f x = x * (x - 1)) :
  ∀ x : ℝ, x > 0 → f x = x * (x + 1) :=
by
  sorry

end find_fx_l258_258334


namespace find_k_value_l258_258854

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l258_258854


namespace measure_of_angle_l258_258344

theorem measure_of_angle (x : ℝ) 
  (h₁ : 180 - x = 3 * x - 10) : x = 47.5 :=
by 
  sorry

end measure_of_angle_l258_258344


namespace total_money_l258_258950

-- Define the problem statement
theorem total_money (n : ℕ) (hn : 3 * n = 75) : (n * 1 + n * 5 + n * 10) = 400 :=
by sorry

end total_money_l258_258950


namespace hyperbola_eccentricity_l258_258491

-- Let's define the variables and conditions first
variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variable (h_asymptote : b = a)

-- We need to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l258_258491


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l258_258054

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 12 = 0 ∧ (n.digits 10).sum = 24 ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 12 = 0 ∧ (m.digits 10).sum = 24 → m ≤ n) ∧ n = 888 :=
by {
  sorry -- Proof to be filled in
}

#eval largest_three_digit_multiple_of_12_with_digit_sum_24 -- Should output: ⊤ (True)

end largest_three_digit_multiple_of_12_with_digit_sum_24_l258_258054


namespace books_leftover_l258_258366

-- Definitions of the conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def books_per_shelf : ℕ := 20
def books_bought : ℕ := 26

-- The theorem stating the proof problem
theorem books_leftover : (initial_books + books_bought) - (shelves * books_per_shelf) = 2 := by
  sorry

end books_leftover_l258_258366


namespace ratio_of_dogs_to_cats_l258_258246

-- Definition of conditions
def total_animals : Nat := 21
def cats_to_spay : Nat := 7
def dogs_to_spay : Nat := total_animals - cats_to_spay

-- Ratio of dogs to cats
def dogs_to_cats_ratio : Nat := dogs_to_spay / cats_to_spay

-- Statement to prove
theorem ratio_of_dogs_to_cats : dogs_to_cats_ratio = 2 :=
by
  -- Proof goes here
  sorry

end ratio_of_dogs_to_cats_l258_258246


namespace eq_sets_M_N_l258_258341

def setM : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def setN : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem eq_sets_M_N : setM = setN := by
  sorry

end eq_sets_M_N_l258_258341


namespace smallest_prime_with_digit_sum_23_l258_258091

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258091


namespace sufficient_but_not_necessary_condition_l258_258222

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ (|x| > 1 → (x > 1 ∨ x < -1)) ∧ ¬(|x| > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l258_258222


namespace average_score_l258_258231

variable (T : ℝ) -- Total number of students
variable (M : ℝ) -- Number of male students
variable (F : ℝ) -- Number of female students

variable (avgM : ℝ) -- Average score for male students
variable (avgF : ℝ) -- Average score for female students

-- Conditions
def M_condition : Prop := M = 0.4 * T
def F_condition : Prop := F = 0.6 * T
def avgM_condition : Prop := avgM = 75
def avgF_condition : Prop := avgF = 80

theorem average_score (h1 : M_condition T M) (h2 : F_condition T F) 
    (h3 : avgM_condition avgM) (h4 : avgF_condition avgF) :
    (75 * M + 80 * F) / T = 78 := by
  sorry

end average_score_l258_258231


namespace lcm_5_6_8_18_l258_258926

/-- The least common multiple of the numbers 5, 6, 8, and 18 is 360. -/
theorem lcm_5_6_8_18 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 18) = 360 := by
  sorry

end lcm_5_6_8_18_l258_258926


namespace prime_sum_10003_l258_258422

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l258_258422


namespace clock_in_2023_hours_l258_258631

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l258_258631


namespace solve_box_dimensions_l258_258816

theorem solve_box_dimensions (m n r : ℕ) (h1 : m ≤ n) (h2 : n ≤ r) (h3 : m ≥ 1) (h4 : n ≥ 1) (h5 : r ≥ 1) :
  let k₀ := (m - 2) * (n - 2) * (r - 2)
  let k₁ := 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2))
  let k₂ := 4 * ((m - 2) + (n - 2) + (r - 2))
  (k₀ + k₂ - k₁ = 1985) ↔ ((m = 5 ∧ n = 7 ∧ r = 663) ∨ 
                            (m = 5 ∧ n = 5 ∧ r = 1981) ∨
                            (m = 3 ∧ n = 3 ∧ r = 1981) ∨
                            (m = 1 ∧ n = 7 ∧ r = 399) ∨
                            (m = 1 ∧ n = 3 ∧ r = 1987)) :=
sorry

end solve_box_dimensions_l258_258816


namespace smallest_prime_with_digit_sum_23_l258_258084

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l258_258084


namespace coin_flip_probability_l258_258050

def total_outcomes := (2:ℕ)^12
def favorable_outcomes := Nat.choose 12 9

theorem coin_flip_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 55 / 1024 := 
by
  sorry

end coin_flip_probability_l258_258050


namespace determine_digits_l258_258826

def product_eq_digits (A B C D x : ℕ) : Prop :=
  x * (x + 1) = 1000 * A + 100 * B + 10 * C + D

def product_minus_3_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 3) * (x - 2) = 1000 * C + 100 * A + 10 * B + D

def product_minus_30_eq_digits (A B C D x : ℕ) : Prop :=
  (x - 30) * (x - 29) = 1000 * B + 100 * C + 10 * A + D

theorem determine_digits :
  ∃ (A B C D x : ℕ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  product_eq_digits A B C D x ∧
  product_minus_3_eq_digits A B C D x ∧
  product_minus_30_eq_digits A B C D x ∧
  A = 8 ∧ B = 3 ∧ C = 7 ∧ D = 2 :=
by
  sorry

end determine_digits_l258_258826


namespace bird_count_l258_258041

def initial_birds : ℕ := 12
def new_birds : ℕ := 8
def total_birds : ℕ := initial_birds + new_birds

theorem bird_count : total_birds = 20 := by
  sorry

end bird_count_l258_258041


namespace smallest_c_for_f_inverse_l258_258893

noncomputable def f (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_c_for_f_inverse :
  ∃ c : ℝ, (∀ x₁ x₂ : ℝ, x₁ ≥ c → x₂ ≥ c → f x₁ = f x₂ → x₁ = x₂) ∧ (∀ d : ℝ, d < c → ∃ x₁ x₂ : ℝ, x₁ ≥ d ∧ x₂ ≥ d ∧ f x₁ = f x₂ ∧ x₁ ≠ x₂) ∧ c = 3 :=
by
  sorry

end smallest_c_for_f_inverse_l258_258893


namespace log_base_problem_l258_258713

noncomputable def log_of_base (base value : ℝ) : ℝ := Real.log value / Real.log base

theorem log_base_problem (x : ℝ) (h : log_of_base 16 (x - 3) = 1 / 4) : 1 / log_of_base (x - 3) 2 = 1 := 
by
  sorry

end log_base_problem_l258_258713


namespace triangle_pyramid_angle_l258_258209

theorem triangle_pyramid_angle (φ : ℝ) (vertex_angle : ∀ (A B C : ℝ), (A + B + C = φ)) :
  ∃ θ : ℝ, θ = φ :=
by
  sorry

end triangle_pyramid_angle_l258_258209


namespace smallest_sum_of_squares_value_l258_258223

noncomputable def collinear_points_min_value (A B C D E P : ℝ): Prop :=
  let AB := 3
  let BC := 2
  let CD := 5
  let DE := 4
  let pos_A := 0
  let pos_B := pos_A + AB
  let pos_C := pos_B + BC
  let pos_D := pos_C + CD
  let pos_E := pos_D + DE
  let P := P
  let AP := (P - pos_A)
  let BP := (P - pos_B)
  let CP := (P - pos_C)
  let DP := (P - pos_D)
  let EP := (P - pos_E)
  let sum_squares := AP^2 + BP^2 + CP^2 + DP^2 + EP^2
  (sum_squares = 85.2)

theorem smallest_sum_of_squares_value : ∃ (A B C D E P : ℝ), collinear_points_min_value A B C D E P :=
sorry

end smallest_sum_of_squares_value_l258_258223


namespace cars_in_first_section_l258_258749

noncomputable def first_section_rows : ℕ := 15
noncomputable def first_section_cars_per_row : ℕ := 10
noncomputable def total_cars_first_section : ℕ := first_section_rows * first_section_cars_per_row

theorem cars_in_first_section : total_cars_first_section = 150 :=
by
  sorry

end cars_in_first_section_l258_258749


namespace smallest_prime_with_digit_sum_23_proof_l258_258065

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l258_258065


namespace max_number_of_liars_l258_258593

open Finset

noncomputable def max_liars_in_castle : Nat :=
  let n := 4
  let rooms := Fin n × Fin n
  let neighbors (i j : rooms) : Prop :=
    (i.1 = j.1 ∧ (i.2 = j.2 + 1 ∨ i.2 + 1 = j.2)) ∨
    (i.2 = j.2 ∧ (i.1 = j.1 + 1 ∨ i.1 + 1 = j.1))
  sorry

theorem max_number_of_liars (liar knight : Fin 16 -> Prop)
  (liars_knights_split : ∀ x, liar x ∨ knight x)
  (liar_truth : ∀ (i : Fin 16),
    liar i → (∑ j in (filter (neighbors − [i])) id).card = 0)
  (knight_truth : ∀ (i : Fin 16),
    knight i → (∑ j in (filter (neighbors − [i])) id).card ≥ 1) :
  max_liars_in_castle = 8 :=
sorry

end max_number_of_liars_l258_258593


namespace percent_increase_between_maintenance_checks_l258_258798

theorem percent_increase_between_maintenance_checks (original_time new_time : ℕ) (h_orig : original_time = 50) (h_new : new_time = 60) :
  ((new_time - original_time : ℚ) / original_time) * 100 = 20 := by
  sorry

end percent_increase_between_maintenance_checks_l258_258798


namespace tangent_line_to_parabola_l258_258035

noncomputable def parabola (x : ℝ) : ℝ := 4 * x^2

def derivative_parabola (x : ℝ) : ℝ := 8 * x

def tangent_line_eq (x y : ℝ) : Prop := 8 * x - y - 4 = 0

theorem tangent_line_to_parabola (x : ℝ) (hx : x = 1) (hy : parabola x = 4) :
    tangent_line_eq 1 4 :=
by 
  -- Sorry to skip the detailed proof, but it should follow the steps outlined in the solution.
  sorry

end tangent_line_to_parabola_l258_258035


namespace period2_students_is_8_l258_258763

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l258_258763


namespace pictures_vertical_l258_258744

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l258_258744


namespace geom_series_sum_l258_258248

def a : ℚ := 1 / 3
def r : ℚ := 2 / 3
def n : ℕ := 9

def S_n (a r : ℚ) (n : ℕ) := a * (1 - r^n) / (1 - r)

theorem geom_series_sum :
  S_n a r n = 19171 / 19683 := by
    sorry

end geom_series_sum_l258_258248


namespace specific_certain_event_l258_258955

theorem specific_certain_event :
  ∀ (A B C D : Prop), 
    (¬ A) →
    (¬ B) →
    (¬ C) →
    D →
    D :=
by
  intros A B C D hA hB hC hD
  exact hD

end specific_certain_event_l258_258955


namespace correct_option_C_l258_258391

noncomputable theory

def parabola (x : ℝ) : ℝ :=
  (x - 1)^2 - 2

theorem correct_option_C (a b c d : ℝ)
  (ha : a < 0) 
  (hb : b > 0)
  (hc : c = a ∨ c = b ∨ (a < c ∧ c < b))
  (hd : d < 1)
  (hA : parabola a = 2)
  (hB : parabola b = 6)
  (hC : parabola c = d) :
  a < c ∧ c < b := 
sorry

end correct_option_C_l258_258391


namespace smallest_c_over_a_plus_b_l258_258953

theorem smallest_c_over_a_plus_b (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ d : ℝ, d = (c / (a + b)) ∧ d = (Real.sqrt 2 / 2) :=
by
  sorry

end smallest_c_over_a_plus_b_l258_258953


namespace factor_expression_l258_258254

variable (x y : ℝ)

theorem factor_expression : 3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := 
by 
  sorry

end factor_expression_l258_258254


namespace option_d_correct_l258_258796

theorem option_d_correct (x : ℝ) : (-3 * x + 2) * (-3 * x - 2) = 9 * x^2 - 4 := 
  sorry

end option_d_correct_l258_258796


namespace solve_equation_l258_258641

theorem solve_equation : ∃ x : ℝ, 2 * x + 1 = 0 ∧ x = -1 / 2 := by
  sorry

end solve_equation_l258_258641


namespace closest_perfect_square_l258_258934

theorem closest_perfect_square (n : ℕ) (h : n = 315) : ∃ k : ℕ, k^2 = 324 ∧ ∀ m : ℕ, m^2 ≠ 315 ∨ abs (n - m^2) > abs (n - k^2) :=
by
  use 18
  sorry

end closest_perfect_square_l258_258934


namespace ratio_equiv_solve_x_l258_258287

theorem ratio_equiv_solve_x (x : ℕ) (h : 3 / 12 = 3 / x) : x = 12 :=
sorry

end ratio_equiv_solve_x_l258_258287


namespace sum_of_primes_10003_l258_258439

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l258_258439


namespace smallest_prime_with_digit_sum_23_l258_258062

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l258_258062


namespace bathroom_area_is_50_square_feet_l258_258657

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet_l258_258657


namespace not_divisible_by_4_l258_258801

theorem not_divisible_by_4 (n : Int) : ¬ (1 + n + n^2 + n^3 + n^4) % 4 = 0 := by
  sorry

end not_divisible_by_4_l258_258801


namespace gcf_of_lcm_9_15_and_10_21_is_5_l258_258509

theorem gcf_of_lcm_9_15_and_10_21_is_5
  (h9 : 9 = 3 ^ 2)
  (h15 : 15 = 3 * 5)
  (h10 : 10 = 2 * 5)
  (h21 : 21 = 3 * 7) :
  Nat.gcd (Nat.lcm 9 15) (Nat.lcm 10 21) = 5 := by
  sorry

end gcf_of_lcm_9_15_and_10_21_is_5_l258_258509


namespace floor_sqrt_equality_l258_258191

theorem floor_sqrt_equality (n : ℕ) : 
  (Int.floor (Real.sqrt (4 * n + 1))) = (Int.floor (Real.sqrt (4 * n + 3))) := 
by 
  sorry

end floor_sqrt_equality_l258_258191


namespace smallest_prime_with_digit_sum_23_proof_l258_258067

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l258_258067


namespace large_A_exists_l258_258975

noncomputable def F_n (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem large_A_exists : ∃ n1 n2 n3 n4 n5 n6 : ℕ,
  ∀ a : ℕ, a ≤ 53590 → 
  F_n n6 (F_n n5 (F_n n4 (F_n n3 (F_n n2 (F_n n1 a))))) = 1 :=
by
  sorry

end large_A_exists_l258_258975


namespace range_of_m_l258_258637

noncomputable def quadratic_function : Type := ℝ → ℝ

variable (f : quadratic_function)

axiom quadratic : ∃ a b : ℝ, ∀ x : ℝ, f x = a * (x-2)^2 + b
axiom symmetry : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom condition1 : f 0 = 3
axiom condition2 : f 2 = 1
axiom max_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), f x ≤ 3
axiom min_value : ∀ m : ℝ, ∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x

theorem range_of_m : ∀ m : ℝ, (∀ (x : ℝ) (hx : 0 ≤ x ∧ x ≤ m), 1 ≤ f x ∧ f x ≤ 3) → 2 ≤ m ∧ m ≤ 4 :=
by
  intro m
  intro h
  sorry

end range_of_m_l258_258637


namespace find_height_of_cuboid_l258_258378

-- Define the cuboid structure and its surface area formula
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

def surface_area (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Given conditions
def given_cuboid : Cuboid := { length := 12, width := 14, height := 7 }
def given_surface_area : ℝ := 700

-- The theorem to prove
theorem find_height_of_cuboid :
  surface_area given_cuboid = given_surface_area :=
by
  sorry

end find_height_of_cuboid_l258_258378


namespace price_of_two_identical_filters_l258_258805

def price_of_individual_filters (x : ℝ) : Prop :=
  let total_individual := 2 * 14.05 + 19.50 + 2 * x
  total_individual = 87.50 / 0.92

theorem price_of_two_identical_filters
  (h1 : price_of_individual_filters 23.76) :
  23.76 * 2 + 28.10 + 19.50 = 87.50 / 0.92 :=
by sorry

end price_of_two_identical_filters_l258_258805


namespace smallest_int_cond_l258_258793

theorem smallest_int_cond (b : ℕ) :
  (b % 9 = 5) ∧ (b % 11 = 7) → b = 95 :=
by
  intro h
  sorry

end smallest_int_cond_l258_258793


namespace solveForN_l258_258406

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end solveForN_l258_258406


namespace smallest_prime_with_digit_sum_23_l258_258075

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l258_258075


namespace hyperbola_foci_coordinates_l258_258769

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), x^2 - (y^2 / 3) = 1 → (∃ c : ℝ, c = 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_coordinates_l258_258769


namespace pictures_vertical_l258_258743

theorem pictures_vertical (V H X : ℕ) (h1 : V + H + X = 30) (h2 : H = 15) (h3 : X = 5) : V = 10 := 
by 
  sorry

end pictures_vertical_l258_258743


namespace simplify_expression_l258_258327

theorem simplify_expression (r : ℝ) (h1 : r^2 ≠ 0) (h2 : r^4 > 16) :
  ( ( ( (r^2 + 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 + 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ)
    - ( (r^2 - 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 - 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ) ) ^ 2 )
  / ( r^2 - (r^4 - 16) ^ (1 / 2 : ℝ) )
  = 2 * r ^ (-(2 / 3 : ℝ)) := by
  sorry

end simplify_expression_l258_258327


namespace Gake_needs_fewer_boards_than_Tom_l258_258644

noncomputable def Tom_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (3 * width_char + 2 * 6) / width_board

noncomputable def Gake_boards_needed : ℕ :=
  let width_board : ℕ := 5
  let width_char : ℕ := 9
  (4 * width_char + 3 * 1) / width_board

theorem Gake_needs_fewer_boards_than_Tom :
  Gake_boards_needed < Tom_boards_needed :=
by
  -- Here you will put the actual proof steps
  sorry

end Gake_needs_fewer_boards_than_Tom_l258_258644


namespace find_difference_l258_258004

theorem find_difference (a b : ℕ) (h1 : Nat.coprime a b) (h2 : a > b) (h3 : (a^3 - b^3) / (a - b)^3 = 50 / 3) :
  a - b = 3 :=
sorry

end find_difference_l258_258004


namespace combined_experience_is_correct_l258_258298

-- Define the conditions as given in the problem
def james_experience : ℕ := 40
def partner_less_years : ℕ := 10
def partner_experience : ℕ := james_experience - partner_less_years

-- The combined experience of James and his partner
def combined_experience : ℕ := james_experience + partner_experience

-- Lean statement to prove the combined experience is 70 years
theorem combined_experience_is_correct : combined_experience = 70 := by sorry

end combined_experience_is_correct_l258_258298


namespace labor_cost_per_hour_l258_258308

theorem labor_cost_per_hour (total_repair_cost part_cost labor_hours : ℕ)
    (h1 : total_repair_cost = 2400)
    (h2 : part_cost = 1200)
    (h3 : labor_hours = 16) :
    (total_repair_cost - part_cost) / labor_hours = 75 := by
  sorry

end labor_cost_per_hour_l258_258308


namespace y_share_per_rupee_l258_258114

theorem y_share_per_rupee (a p : ℝ) (h1 : a * p = 18)
                            (h2 : p + a * p + 0.30 * p = 70) :
    a = 0.45 :=
by 
  sorry

end y_share_per_rupee_l258_258114


namespace exists_function_l258_258561

theorem exists_function {n : ℕ} (hn : n ≥ 3) (S : Finset ℤ) (hS : S.card = n) :
  ∃ f : Fin (n) → S, 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i j k : Fin n, i < j ∧ j < k → 2 * (f j : ℤ) ≠ (f i : ℤ) + (f k : ℤ)) :=
by
  sorry

end exists_function_l258_258561


namespace range_f_l258_258734

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 

theorem range_f : Set.Icc (0 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_f_l258_258734


namespace smallest_sum_B_c_l258_258712

theorem smallest_sum_B_c : 
  ∃ (B : ℕ) (c : ℕ), (0 ≤ B ∧ B ≤ 4) ∧ (c ≥ 6) ∧ 31 * B = 4 * (c + 1) ∧ B + c = 8 := 
sorry

end smallest_sum_B_c_l258_258712


namespace find_m_l258_258395

theorem find_m (x m : ℤ) (h : x = -1 ∧ x - 2 * m = 9) : m = -5 :=
sorry

end find_m_l258_258395


namespace dot_product_eq_half_l258_258279

noncomputable def vector_dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2
  
theorem dot_product_eq_half :
  vector_dot_product (Real.cos (25 * Real.pi / 180), Real.sin (25 * Real.pi / 180))
                     (Real.cos (85 * Real.pi / 180), Real.cos (5 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end dot_product_eq_half_l258_258279


namespace split_numbers_cubic_l258_258560

theorem split_numbers_cubic (m : ℕ) (hm : 1 < m) (assumption : m^2 - m + 1 = 73) : m = 9 :=
sorry

end split_numbers_cubic_l258_258560


namespace no_such_base_exists_l258_258140

theorem no_such_base_exists : ¬ ∃ b : ℕ, (b^3 ≤ 630 ∧ 630 < b^4) ∧ (630 % b) % 2 = 1 := by
  sorry

end no_such_base_exists_l258_258140


namespace find_a_l258_258397

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 - x) - Real.log (1 + x) + a

theorem find_a 
  (M : ℝ) (N : ℝ) (a : ℝ)
  (h1 : M = f a (-1/2))
  (h2 : N = f a (1/2))
  (h3 : M + N = 1) :
  a = 1 / 2 := 
sorry

end find_a_l258_258397


namespace minimum_value_l258_258380

open Real

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2) + y^2 / (x - 2)) ≥ 12 :=
sorry

end minimum_value_l258_258380


namespace tanner_savings_in_november_l258_258759

theorem tanner_savings_in_november(savings_sep : ℕ) (savings_oct : ℕ) 
(spending : ℕ) (leftover : ℕ) (N : ℕ) :
savings_sep = 17 →
savings_oct = 48 →
spending = 49 →
leftover = 41 →
((savings_sep + savings_oct + N - spending) = leftover) →
N = 25 :=
by
  intros h_sep h_oct h_spending h_leftover h_equation
  sorry

end tanner_savings_in_november_l258_258759


namespace range_of_a_l258_258405

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, a*x^2 - 2*a*x + 3 ≤ 0) ↔ (0 ≤ a ∧ a < 3) := 
sorry

end range_of_a_l258_258405


namespace rem_neg_one_third_quarter_l258_258840

noncomputable def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_neg_one_third_quarter :
  rem (-1/3) (1/4) = 1/6 :=
by
  sorry

end rem_neg_one_third_quarter_l258_258840


namespace points_concyclic_l258_258648

variables {O₁ O₂ O₃ O₄ A B C D : Point}

/-- Centers of the coins -/
axiom centers_coins (O₁ O₂ O₃ O₄ : Point) : true

/-- Collinearity of the points with respective centers -/
axiom collinear_O₁_A_O₂ : collinear {O₁, A, O₂}
axiom collinear_O₂_B_O₃ : collinear {O₂, B, O₃}
axiom collinear_O₃_C_O₄ : collinear {O₃, C, O₄}
axiom collinear_O₄_D_O₁ : collinear {O₄, D, O₁}

/-- The goal is to show that points A, B, C, D are concyclic -/
theorem points_concyclic 
  (h₁ : centers_coins O₁ O₂ O₃ O₄)
  (h₂ : collinear_O₁_A_O₂)
  (h₃ : collinear_O₂_B_O₃)
  (h₄ : collinear_O₃_C_O₄)
  (h₅ : collinear_O₄_D_O₁) : 
  cyclic_quad {A, B, C, D} :=
  sorry

end points_concyclic_l258_258648


namespace range_of_a_l258_258143

noncomputable def f (x a : ℝ) := Real.log x + a / x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 0, x * (2 * Real.log a - Real.log x) ≤ a) : 
  0 < a ∧ a ≤ 1 / Real.exp 1 :=
by
  sorry

end range_of_a_l258_258143


namespace benny_leftover_money_l258_258530

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money_l258_258530


namespace remainder_approximately_14_l258_258015

def dividend : ℝ := 14698
def quotient : ℝ := 89
def divisor : ℝ := 164.98876404494382
def remainder : ℝ := dividend - (quotient * divisor)

theorem remainder_approximately_14 : abs (remainder - 14) < 1e-10 := 
by
-- using abs since the problem is numerical/approximate
sorry

end remainder_approximately_14_l258_258015


namespace luke_played_rounds_l258_258745

theorem luke_played_rounds (total_points : ℕ) (points_per_round : ℕ) (result : ℕ)
  (h1 : total_points = 154)
  (h2 : points_per_round = 11)
  (h3 : result = total_points / points_per_round) :
  result = 14 :=
by
  rw [h1, h2] at h3
  exact h3

end luke_played_rounds_l258_258745


namespace intersection_nonempty_condition_l258_258993

theorem intersection_nonempty_condition (m n : ℝ) :
  (∃ x : ℝ, (m - 1 < x ∧ x < m + 1) ∧ (3 - n < x ∧ x < 4 - n)) ↔ (2 < m + n ∧ m + n < 5) := 
by
  sorry

end intersection_nonempty_condition_l258_258993


namespace cylinder_surface_area_l258_258112

theorem cylinder_surface_area (side : ℝ) (h : ℝ) (r : ℝ) : 
  side = 2 ∧ h = side ∧ r = side → 
  (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 16 * Real.pi := 
by
  intro h
  sorry

end cylinder_surface_area_l258_258112


namespace smallest_prime_with_digit_sum_23_l258_258095

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l258_258095


namespace eq_one_solution_in_interval_l258_258286

theorem eq_one_solution_in_interval (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ (2 * a * x^2 - x - 1 = 0) ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 1 ∧ y ≠ x → (2 * a * y^2 - y - 1 ≠ 0))) → (1 < a) :=
by
  sorry

end eq_one_solution_in_interval_l258_258286


namespace temperature_at_midnight_is_minus4_l258_258751

-- Definitions of initial temperature and changes
def initial_temperature : ℤ := -2
def temperature_rise_noon : ℤ := 6
def temperature_drop_midnight : ℤ := 8

-- Temperature at midnight
def temperature_midnight : ℤ :=
  initial_temperature + temperature_rise_noon - temperature_drop_midnight

theorem temperature_at_midnight_is_minus4 :
  temperature_midnight = -4 := by
  sorry

end temperature_at_midnight_is_minus4_l258_258751


namespace solve_system_eqns_l258_258328

theorem solve_system_eqns 
  {a b c : ℝ} (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
  {x y z : ℝ} 
  (h4 : a^3 + a^2 * x + a * y + z = 0)
  (h5 : b^3 + b^2 * x + b * y + z = 0)
  (h6 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + bc + ca ∧ z = -abc :=
by {
  sorry
}

end solve_system_eqns_l258_258328


namespace interval_of_a_l258_258272

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  f a n.succ  -- since ℕ in Lean includes 0, use n.succ to start from 1

-- The main theorem to prove
theorem interval_of_a (a : ℝ) : (∀ n : ℕ, n ≠ 0 → a_n a n < a_n a (n + 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end interval_of_a_l258_258272


namespace solve_inequality_l258_258329

open Real

theorem solve_inequality (a : ℝ) :
  ((a < 0 ∨ a > 1) → (∀ x, a < x ∧ x < a^2 ↔ (x - a) * (x - a^2) < 0)) ∧
  ((0 < a ∧ a < 1) → (∀ x, a^2 < x ∧ x < a ↔ (x - a) * (x - a^2) < 0)) ∧
  ((a = 0 ∨ a = 1) → (∀ x, ¬((x - a) * (x - a^2) < 0))) :=
by
  sorry

end solve_inequality_l258_258329


namespace trapezoid_angle_l258_258451

theorem trapezoid_angle
  (EFGH : Type)
  (EF GH : EFGH)
  (parallel : ∃ (EF GH : EFGH), EF ∥ GH)
  (angle_E_eq : ∃ (E H : ℝ), E = 3 * H)
  (angle_G_eq : ∃ (G F : ℝ), G = 2 * F)
  (angle_sum : ∃ (F G : ℝ), F + G = 180) :
  ∃ (F : ℝ), F = 60 :=
by
  sorry

end trapezoid_angle_l258_258451


namespace find_C_l258_258365

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 330) : C = 30 := 
sorry

end find_C_l258_258365


namespace cookie_weight_l258_258706

theorem cookie_weight :
  ∀ (pounds_per_box cookies_per_box ounces_per_pound : ℝ),
    pounds_per_box = 40 →
    cookies_per_box = 320 →
    ounces_per_pound = 16 →
    (pounds_per_box * ounces_per_pound) / cookies_per_box = 2 := 
by 
  intros pounds_per_box cookies_per_box ounces_per_pound hpounds hcookies hounces
  rw [hpounds, hcookies, hounces]
  norm_num

end cookie_weight_l258_258706


namespace concentric_circles_circumference_difference_and_area_l258_258829

theorem concentric_circles_circumference_difference_and_area {r_inner r_outer : ℝ} (h1 : r_inner = 25) (h2 : r_outer = r_inner + 15) :
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi ∧ Real.pi * r_outer^2 - Real.pi * r_inner^2 = 975 * Real.pi :=
by
  sorry

end concentric_circles_circumference_difference_and_area_l258_258829


namespace cistern_length_is_four_l258_258520

noncomputable def length_of_cistern (width depth total_area : ℝ) : ℝ :=
  let L := ((total_area - (2 * width * depth)) / (2 * (width + depth)))
  L

theorem cistern_length_is_four
  (width depth total_area : ℝ)
  (h_width : width = 2)
  (h_depth : depth = 1.25)
  (h_total_area : total_area = 23) :
  length_of_cistern width depth total_area = 4 :=
by 
  sorry

end cistern_length_is_four_l258_258520


namespace min_cut_length_no_triangle_l258_258666

theorem min_cut_length_no_triangle (a b c x : ℝ) 
  (h_y : a = 7) 
  (h_z : b = 24) 
  (h_w : c = 25) 
  (h1 : a - x > 0)
  (h2 : b - x > 0)
  (h3 : c - x > 0)
  (h4 : (a - x) + (b - x) ≤ (c - x)) :
  x = 6 :=
by
  sorry

end min_cut_length_no_triangle_l258_258666


namespace roots_opposite_signs_l258_258111

theorem roots_opposite_signs (a b c: ℝ) 
  (h1 : (b^2 - a * c) > 0)
  (h2 : (b^4 - a^2 * c^2) < 0) :
  a * c < 0 :=
sorry

end roots_opposite_signs_l258_258111


namespace prime_sum_l258_258284

theorem prime_sum (m n : ℕ) (hm : Prime m) (hn : Prime n) (h : 5 * m + 7 * n = 129) :
  m + n = 19 ∨ m + n = 25 := by
  sorry

end prime_sum_l258_258284


namespace period2_students_is_8_l258_258764

-- Definitions according to conditions
def period1_students : Nat := 11
def relationship (x : Nat) := 2 * x - 5

-- Lean 4 statement
theorem period2_students_is_8 (x: Nat) (h: relationship x = period1_students) : x = 8 := 
by 
  -- Placeholder for the proof
  sorry

end period2_students_is_8_l258_258764


namespace impossible_to_use_up_components_l258_258346

theorem impossible_to_use_up_components 
  (p q r x y z : ℕ) 
  (condition1 : 2 * x + 2 * z = 2 * p + 2 * r + 2)
  (condition2 : 2 * x + y = 2 * p + q + 1)
  (condition3 : y + z = q + r) : 
  False :=
by sorry

end impossible_to_use_up_components_l258_258346


namespace smallest_prime_with_digit_sum_23_l258_258092

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258092


namespace simplify_expression_l258_258126

variable (x : ℝ)

theorem simplify_expression : (x + 2)^2 - (x + 1) * (x + 3) = 1 := 
by 
  sorry

end simplify_expression_l258_258126


namespace unit_cubes_with_paint_l258_258549

/-- Conditions:
1. Cubes with each side one inch long are glued together to form a larger cube.
2. The larger cube's face is painted with red color and the entire assembly is taken apart.
3. 23 small cubes are found with no paints on them.
-/
theorem unit_cubes_with_paint (n : ℕ) (h1 : n^3 - (n - 2)^3 = 23) (h2 : n = 4) :
    n^3 - 23 = 41 :=
by
  sorry

end unit_cubes_with_paint_l258_258549


namespace no_prime_sum_10003_l258_258444

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l258_258444


namespace years_in_future_l258_258671

theorem years_in_future (Shekhar Shobha : ℕ) (h1 : Shekhar / Shobha = 4 / 3) (h2 : Shobha = 15) (h3 : Shekhar + t = 26)
  : t = 6 :=
by
  sorry

end years_in_future_l258_258671


namespace cost_of_white_washing_l258_258116

-- Definitions for room dimensions, doors, windows, and cost per square foot
def length : ℕ := 25
def width : ℕ := 15
def height1 : ℕ := 12
def height2 : ℕ := 8
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def cost_per_sq_ft : ℕ := 10
def ceiling_decoration_area : ℕ := 10

-- Definitions for the areas calculation
def area_walls_height1 : ℕ := 2 * (length * height1)
def area_walls_height2 : ℕ := 2 * (width * height2)
def total_wall_area : ℕ := area_walls_height1 + area_walls_height2

def area_one_door : ℕ := door_height * door_width
def total_doors_area : ℕ := 2 * area_one_door

def area_one_window : ℕ := window_height * window_width
def total_windows_area : ℕ := 3 * area_one_window

def adjusted_wall_area : ℕ := total_wall_area - total_doors_area - total_windows_area - ceiling_decoration_area

def total_cost : ℕ := adjusted_wall_area * cost_per_sq_ft

-- The theorem we want to prove
theorem cost_of_white_washing : total_cost = 7580 := by
  sorry

end cost_of_white_washing_l258_258116


namespace find_real_pairs_l258_258376

theorem find_real_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end find_real_pairs_l258_258376


namespace parameter_a_solution_exists_l258_258971

theorem parameter_a_solution_exists (a : ℝ) : 
  (a < -2 / 3 ∨ a > 0) → ∃ b x y : ℝ, 
  x = 6 / a - abs (y - a) ∧ x^2 + y^2 + b^2 + 63 = 2 * (b * y - 8 * x) :=
by
  intro h
  sorry

end parameter_a_solution_exists_l258_258971


namespace inhabitants_number_even_l258_258189

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem inhabitants_number_even
  (K L : ℕ)
  (hK : is_even K)
  (hL : is_even L) :
  ¬ is_even (K + L + 1) :=
by
  sorry

end inhabitants_number_even_l258_258189


namespace max_value_neg_a_inv_l258_258865

theorem max_value_neg_a_inv (a : ℝ) (h : a < 0) : a + (1 / a) ≤ -2 := 
by
  sorry

end max_value_neg_a_inv_l258_258865


namespace book_pages_total_l258_258102

-- Define the conditions
def pagesPerNight : ℝ := 120.0
def nights : ℝ := 10.0

-- State the theorem to prove
theorem book_pages_total : pagesPerNight * nights = 1200.0 := by
  sorry

end book_pages_total_l258_258102


namespace time_after_2023_hours_l258_258632

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l258_258632


namespace square_free_odd_integers_count_l258_258834

def is_square_free (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → false

lemma odd_integer_in_range (n : ℕ) : Prop :=
  n > 1 ∧ n < 200 ∧ odd n

theorem square_free_odd_integers_count :
  (∃ count : ℕ, count = 81 ∧
   count = (λ S, S.card) {n : ℕ | odd_integer_in_range n ∧ is_square_free n} ) :=
begin
  sorry
end

end square_free_odd_integers_count_l258_258834


namespace minimal_height_exists_l258_258547

noncomputable def height_min_material (x : ℝ) : ℝ := 4 / (x^2)

theorem minimal_height_exists
  (x h : ℝ)
  (volume_cond : x^2 * h = 4)
  (surface_area_cond : h = height_min_material x) :
  h = 1 := by
  sorry

end minimal_height_exists_l258_258547


namespace parametric_equations_hyperbola_l258_258772

variable {θ t : ℝ}
variable (n : ℤ)

theorem parametric_equations_hyperbola (hθ : θ ≠ (n / 2) * π)
  (hx : ∀ t, x t = 1 / 2 * (Real.exp t + Real.exp (-t)) * Real.cos θ)
  (hy : ∀ t, y t = 1 / 2 * (Real.exp t - Real.exp (-t)) * Real.sin θ) :
  (∀ t, (x t)^2 / (Real.cos θ)^2 - (y t)^2 / (Real.sin θ)^2 = 1) := sorry

end parametric_equations_hyperbola_l258_258772


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l258_258869

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l258_258869


namespace part_1_part_2_l258_258702

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

-- (Part 1): Prove the value of a
theorem part_1 (a : ℝ) (P : ℝ × ℝ) (hP : P = (a, -4)) :
  (∃ t : ℝ, ∃ t₂ : ℝ, t ≠ t₂ ∧ P.2 = (2 * t^3 - 3 * t^2 + 1) + (6 * t^2 - 6 * t) * (a - t)) →
  a = -1 ∨ a = 7 / 2 :=
sorry

-- (Part 2): Prove the range of k
noncomputable def g (x k : ℝ) : ℝ := k * x + 1 - Real.log x

noncomputable def h (x k : ℝ) : ℝ := min (f x) (g x k)

theorem part_2 (k : ℝ) :
  (∀ x > 0, h x k = 0 → (x = 1 ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 k = 0 ∧ h x2 k = 0)) →
  0 < k ∧ k < 1 / Real.exp 2 :=
sorry

end part_1_part_2_l258_258702


namespace combined_weight_l258_258051

theorem combined_weight (x y z : ℕ) (h1 : x + z = 78) (h2 : x + y = 69) (h3 : y + z = 137) : x + y + z = 142 :=
by
  -- Intermediate steps or any additional lemmas could go here
sorry

end combined_weight_l258_258051


namespace proof_problem_l258_258737

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < (π / 2))
variable (htan : tan α = (1 + sin β) / cos β)

theorem proof_problem : 2 * α - β = π / 2 :=
by
  sorry

end proof_problem_l258_258737


namespace santana_brothers_birthday_l258_258326

theorem santana_brothers_birthday (b : ℕ) (oct : ℕ) (nov : ℕ) (dec : ℕ) (c_presents_diff : ℕ) :
  b = 7 → oct = 1 → nov = 1 → dec = 2 → c_presents_diff = 8 → (∃ M : ℕ, M = 3) :=
by
  sorry

end santana_brothers_birthday_l258_258326


namespace prime_sum_10003_l258_258421

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l258_258421


namespace value_of_f_at_minus_point_two_l258_258652

noncomputable def f (x : ℝ) : ℝ := 1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem value_of_f_at_minus_point_two : f (-0.2) = 0.81873 :=
by {
  sorry
}

end value_of_f_at_minus_point_two_l258_258652


namespace decompose_fraction1_decompose_fraction2_l258_258132

-- Define the first problem as a theorem
theorem decompose_fraction1 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1)) = (1 / (x - 1)) - (1 / (x + 1)) :=
sorry  -- Proof required

-- Define the second problem as a theorem
theorem decompose_fraction2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 * x / (x^2 - 1)) = (1 / (x - 1)) + (1 / (x + 1)) :=
sorry  -- Proof required

end decompose_fraction1_decompose_fraction2_l258_258132


namespace groups_division_count_l258_258197

open Finset

def count_ways_to_divide_dogs (dogs : Finset ℕ) : ℕ :=
  let rocky := 1 -- Assume 1 is Rocky
  let nipper := 2 -- Assume 2 is Nipper
  let scruffy := 3 -- Assume 3 is Scruffy
  let remaining_dogs := dogs \ {rocky, nipper, scruffy}
  let ways_3_dog_group := (remaining_dogs.card.choose 2)
  let ways_4_dog_group := ((remaining_dogs \ (remaining_dogs.choose 2)).card).choose 3
  ways_3_dog_group * ways_4_dog_group

theorem groups_division_count :
  count_ways_to_divide_dogs (range 12) = 1260 :=
by
  dsimp [count_ways_to_divide_dogs]
  sorry

end groups_division_count_l258_258197


namespace find_a_for_even_function_l258_258698

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) →
  a = 1 :=
by
  intro h
  sorry

end find_a_for_even_function_l258_258698


namespace solve_for_x_l258_258262

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l258_258262


namespace incorrect_judgment_l258_258144

theorem incorrect_judgment : (∀ x : ℝ, x^2 - 1 ≥ -1) ∧ (4 + 2 ≠ 7) :=
by 
  sorry

end incorrect_judgment_l258_258144


namespace gallon_of_water_weighs_eight_pounds_l258_258297

theorem gallon_of_water_weighs_eight_pounds
  (pounds_per_tablespoon : ℝ := 1.5)
  (cubic_feet_per_gallon : ℝ := 7.5)
  (cost_per_tablespoon : ℝ := 0.50)
  (total_cost : ℝ := 270)
  (bathtub_capacity_cubic_feet : ℝ := 6)
  : (6 * 7.5) * pounds_per_tablespoon = 270 / cost_per_tablespoon / 1.5 :=
by
  sorry

end gallon_of_water_weighs_eight_pounds_l258_258297


namespace probability_of_snow_on_most_3_days_l258_258496

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def cumulative_binomial_probability (n : ℕ) (kmax : ℕ) (p : ℝ) : ℝ :=
  finset.sum (finset.range (kmax + 1)) (λ k => binomial_probability n k p)

theorem probability_of_snow_on_most_3_days :
  let p := 1/5 in
  let n := 31 in
  abs (cumulative_binomial_probability n 3 p - 0.257) < 0.001 :=
by
  let p := 1/5
  let n := 31
  -- Here we define the cumulative probability up to 3 days
  let approx := cumulative_binomial_probability n 3 p
  -- We assert that the calculated value should be approximately 0.257
  have h : abs (approx - 0.257) < 0.001
  sorry

end probability_of_snow_on_most_3_days_l258_258496


namespace ways_to_write_10003_as_sum_of_two_primes_l258_258420

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l258_258420


namespace stu_books_count_l258_258023

theorem stu_books_count (S : ℕ) (h1 : S + 4 * S = 45) : S = 9 := 
by
  sorry

end stu_books_count_l258_258023


namespace range_of_m_range_of_a_l258_258156

-- Define the function \( f(x) \)
def f (a : ℝ) (x : ℝ) : ℝ := (a - 0.5) * x^2 + Real.log x

-- Define the interval for part (1)
def interval1 := Icc 1 Real.exp 1

-- Statement for part (1):
theorem range_of_m (m : ℝ) : 
  (∃ x0 ∈ interval1, f 1 x0 ≤ m) ↔ m ∈ Icc 0.5 ⊤ := sorry

-- Define the condition for part (2)
def below_line (a : ℝ) : Prop :=
  ∀ x > 1, f a x < 2 * a * x

-- Statement for part (2):
theorem range_of_a : 
  (∀ x, 1 < x → f a x < 2 * a * x) ↔ (a ∈ Icc (-0.5) 0.5) := sorry

end range_of_m_range_of_a_l258_258156


namespace remainder_1425_1427_1429_mod_12_l258_258654

theorem remainder_1425_1427_1429_mod_12 : 
  (1425 * 1427 * 1429) % 12 = 3 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l258_258654


namespace simplify_product_l258_258475

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l258_258475


namespace binomial_constant_term_l258_258629

theorem binomial_constant_term : 
  (∃ c : ℕ, ∀ x : ℝ, (x + (1 / (3 * x)))^8 = c * (x ^ (4 * 2 - 8) / 3)) → 
  ∃ c : ℕ, c = 28 :=
sorry

end binomial_constant_term_l258_258629


namespace sqrt_meaningful_range_l258_258282

-- Define the condition
def sqrt_condition (x : ℝ) : Prop := 1 - 3 * x ≥ 0

-- State the theorem
theorem sqrt_meaningful_range (x : ℝ) (h : sqrt_condition x) : x ≤ 1 / 3 :=
sorry

end sqrt_meaningful_range_l258_258282


namespace solve_system_of_equations_l258_258555

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations_l258_258555


namespace find_n_150_l258_258966

def special_sum (k n : ℕ) : ℕ := (n * (2 * k + n - 1)) / 2

theorem find_n_150 : ∃ n : ℕ, special_sum 3 n = 150 ∧ n = 15 :=
by
  sorry

end find_n_150_l258_258966


namespace find_a_l258_258864

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l258_258864


namespace watch_A_accurate_l258_258043

variable (T : ℕ) -- Standard time, represented as natural numbers for simplicity
variable (A B : ℕ) -- Watches A and B, also represented as natural numbers
variable (h1 : A = B + 2) -- Watch A is 2 minutes faster than Watch B
variable (h2 : B = T - 2) -- Watch B is 2 minutes slower than the standard time

theorem watch_A_accurate : A = T :=
by
  -- The proof would go here
  sorry

end watch_A_accurate_l258_258043


namespace find_k_l258_258617

-- Define the conditions
variables (a b : Real) (x y : Real)

-- The problem's conditions
def tan_x : Prop := Real.tan x = a / b
def tan_2x : Prop := Real.tan (x + x) = b / (a + b)
def y_eq_x : Prop := y = x

-- The goal to prove
theorem find_k (ha : tan_x a b x) (hb : tan_2x a b x) (hy : y_eq_x x y) :
  ∃ k, x = Real.arctan k ∧ k = 1 / (a + 2) :=
sorry

end find_k_l258_258617


namespace max_consecutive_semi_primes_l258_258895

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semi_primes : ∃ (N : ℕ), N = 5 ∧
  ∀ (a b : ℕ), (a > 25) ∧ (b = a + 4) → 
  (∀ n, a ≤ n ∧ n ≤ b → is_semi_prime n) ↔ N = 5 := sorry

end max_consecutive_semi_primes_l258_258895


namespace average_age_of_John_Mary_Tonya_is_35_l258_258300

-- Define the ages of the individuals
variable (John Mary Tonya : ℕ)

-- Conditions given in the problem
def John_is_twice_as_old_as_Mary : Prop := John = 2 * Mary
def John_is_half_as_old_as_Tonya : Prop := John = Tonya / 2
def Tonya_is_60 : Prop := Tonya = 60

-- The average age calculation
def average_age (a b c : ℕ) : ℕ := (a + b + c) / 3

-- The statement we need to prove
theorem average_age_of_John_Mary_Tonya_is_35 :
  John_is_twice_as_old_as_Mary John Mary →
  John_is_half_as_old_as_Tonya John Tonya →
  Tonya_is_60 Tonya →
  average_age John Mary Tonya = 35 :=
by
  sorry

end average_age_of_John_Mary_Tonya_is_35_l258_258300


namespace ratio_of_speeds_l258_258348

variable (x y n : ℝ)

-- Conditions
def condition1 : Prop := 3 * (x - y) = n
def condition2 : Prop := 2 * (x + y) = n

-- Problem Statement
theorem ratio_of_speeds (h1 : condition1 x y n) (h2 : condition2 x y n) : x = 5 * y :=
by
  sorry

end ratio_of_speeds_l258_258348


namespace sum_of_solutions_l258_258642

theorem sum_of_solutions :
  (∃ S : Finset ℝ, (∀ x ∈ S, x^2 - 8*x + 21 = abs (x - 5) + 4) ∧ S.sum id = 18) :=
by
  sorry

end sum_of_solutions_l258_258642


namespace hidden_message_is_correct_l258_258178

def russian_alphabet_mapping : Char → Nat
| 'А' => 1
| 'Б' => 2
| 'В' => 3
| 'Г' => 4
| 'Д' => 5
| 'Е' => 6
| 'Ё' => 7
| 'Ж' => 8
| 'З' => 9
| 'И' => 10
| 'Й' => 11
| 'К' => 12
| 'Л' => 13
| 'М' => 14
| 'Н' => 15
| 'О' => 16
| 'П' => 17
| 'Р' => 18
| 'С' => 19
| 'Т' => 20
| 'У' => 21
| 'Ф' => 22
| 'Х' => 23
| 'Ц' => 24
| 'Ч' => 25
| 'Ш' => 26
| 'Щ' => 27
| 'Ъ' => 28
| 'Ы' => 29
| 'Ь' => 30
| 'Э' => 31
| 'Ю' => 32
| 'Я' => 33
| _ => 0

def prime_p : ℕ := 7 -- Assume some prime number p

def grid_position (p : ℕ) (k : ℕ) := p * k

theorem hidden_message_is_correct :
  ∃ m : String, m = "ПАРОЛЬ МЕДВЕЖАТА" :=
by
  let message := "ПАРОЛЬ МЕДВЕЖАТА"
  have h1 : russian_alphabet_mapping 'П' = 17 := by sorry
  have h2 : russian_alphabet_mapping 'А' = 1 := by sorry
  have h3 : russian_alphabet_mapping 'Р' = 18 := by sorry
  have h4 : russian_alphabet_mapping 'О' = 16 := by sorry
  have h5 : russian_alphabet_mapping 'Л' = 13 := by sorry
  have h6 : russian_alphabet_mapping 'Ь' = 29 := by sorry
  have h7 : russian_alphabet_mapping 'М' = 14 := by sorry
  have h8 : russian_alphabet_mapping 'Е' = 5 := by sorry
  have h9 : russian_alphabet_mapping 'Д' = 10 := by sorry
  have h10 : russian_alphabet_mapping 'В' = 3 := by sorry
  have h11 : russian_alphabet_mapping 'Ж' = 8 := by sorry
  have h12 : russian_alphabet_mapping 'Т' = 20 := by sorry
  have g1 : grid_position prime_p 17 = 119 := by sorry
  have g2 : grid_position prime_p 1 = 7 := by sorry
  have g3 : grid_position prime_p 18 = 126 := by sorry
  have g4 : grid_position prime_p 16 = 112 := by sorry
  have g5 : grid_position prime_p 13 = 91 := by sorry
  have g6 : grid_position prime_p 29 = 203 := by sorry
  have g7 : grid_position prime_p 14 = 98 := by sorry
  have g8 : grid_position prime_p 5 = 35 := by sorry
  have g9 : grid_position prime_p 10 = 70 := by sorry
  have g10 : grid_position prime_p 3 = 21 := by sorry
  have g11 : grid_position prime_p 8 = 56 := by sorry
  have g12 : grid_position prime_p 20 = 140 := by sorry
  existsi message
  rfl

end hidden_message_is_correct_l258_258178


namespace machine_shirts_per_minute_l258_258527

def shirts_made_yesterday : ℕ := 13
def shirts_made_today : ℕ := 3
def minutes_worked : ℕ := 2
def total_shirts_made : ℕ := shirts_made_yesterday + shirts_made_today
def shirts_per_minute : ℕ := total_shirts_made / minutes_worked

theorem machine_shirts_per_minute :
  shirts_per_minute = 8 := by
  sorry

end machine_shirts_per_minute_l258_258527


namespace cafeteria_pies_l258_258204

theorem cafeteria_pies (total_apples handed_out_apples apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_apples = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_apples) / apples_per_pie = 5 :=
by {
  sorry
}

end cafeteria_pies_l258_258204


namespace rental_cost_l258_258180

theorem rental_cost (total_cost gallons gas_price mile_cost miles : ℝ)
    (H1 : gallons = 8)
    (H2 : gas_price = 3.50)
    (H3 : mile_cost = 0.50)
    (H4 : miles = 320)
    (H5 : total_cost = 338) :
    total_cost - (gallons * gas_price + miles * mile_cost) = 150 := by
  sorry

end rental_cost_l258_258180


namespace cube_root_sum_lt_sqrt_sum_l258_258142

theorem cube_root_sum_lt_sqrt_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
    sorry

end cube_root_sum_lt_sqrt_sum_l258_258142


namespace range_of_x_l258_258275

theorem range_of_x (x : ℝ) : x + 2 ≥ 0 ∧ x - 3 ≠ 0 → x ≥ -2 ∧ x ≠ 3 :=
by
  sorry

end range_of_x_l258_258275


namespace no_such_integers_exist_l258_258753

theorem no_such_integers_exist (x y z : ℤ) (hx : x ≠ 0) :
  ¬ (2 * x ^ 4 + 2 * x ^ 2 * y ^ 2 + y ^ 4 = z ^ 2) :=
by
  sorry

end no_such_integers_exist_l258_258753


namespace find_value_perpendicular_distances_l258_258894

variable {R a b c D E F : ℝ}
variable {ABC : Triangle}

-- Assume the distances from point P on the circumcircle of triangle ABC
-- to the sides BC, CA, and AB respectively.
axiom D_def : D = R * a / (2 * R)
axiom E_def : E = R * b / (2 * R)
axiom F_def : F = R * c / (2 * R)

theorem find_value_perpendicular_distances
    (a b c R : ℝ) (D E F : ℝ) 
    (hD : D = R * a / (2 * R)) 
    (hE : E = R * b / (2 * R)) 
    (hF : F = R * c / (2 * R)) : 
    a^2 * D^2 + b^2 * E^2 + c^2 * F^2 = (a^4 + b^4 + c^4) / (4 * R^2) :=
by
  sorry

end find_value_perpendicular_distances_l258_258894


namespace painted_cube_problem_l258_258823

theorem painted_cube_problem (n : ℕ) (h1 : n > 2)
  (h2 : 6 * (n - 2)^2 = (n - 2)^3) : n = 8 :=
by {
  sorry
}

end painted_cube_problem_l258_258823


namespace existence_of_special_numbers_l258_258969

theorem existence_of_special_numbers :
  ∃ (N : Finset ℕ), N.card = 1998 ∧ 
  ∀ (a b : ℕ), a ∈ N → b ∈ N → a ≠ b → a * b ∣ (a - b)^2 :=
sorry

end existence_of_special_numbers_l258_258969


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258872

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258872


namespace smallest_x_value_l258_258931

theorem smallest_x_value : ∃ x : ℝ, (x = 0) ∧ (∀ y : ℝ, (left_side y x) = 0 → x ≤ y)
where
  left_side y x : ℝ := (y^2 + y - 20)

limit smaller

end smallest_x_value_l258_258931


namespace vectors_orthogonal_x_value_l258_258374

theorem vectors_orthogonal_x_value :
  (∀ x : ℝ, (3 * x + 4 * (-7) = 0) → (x = 28 / 3)) := 
by 
  sorry

end vectors_orthogonal_x_value_l258_258374


namespace sin_theta_value_l258_258387

theorem sin_theta_value (f : ℝ → ℝ)
  (hx : ∀ x, f x = 3 * Real.sin x - 8 * Real.cos (x / 2) ^ 2)
  (h_cond : ∀ x, f x ≤ f θ) : Real.sin θ = 3 / 5 := 
sorry

end sin_theta_value_l258_258387


namespace no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l258_258039

theorem no_real_roots_eq_xsq_abs_x_plus_1_eq_0 :
  ¬ ∃ x : ℝ, x^2 + abs x + 1 = 0 :=
by
  sorry

end no_real_roots_eq_xsq_abs_x_plus_1_eq_0_l258_258039


namespace arccos_zero_eq_pi_div_two_l258_258676

theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l258_258676


namespace equivalence_of_statements_l258_258195

variable (S M : Prop)

theorem equivalence_of_statements : 
  (S → M) ↔ ((¬M → ¬S) ∧ (¬S ∨ M)) :=
by
  sorry

end equivalence_of_statements_l258_258195


namespace find_numbers_l258_258647

theorem find_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (geom_mean_cond : Real.sqrt (a * b) = Real.sqrt 5)
  (harm_mean_cond : 2 / ((1 / a) + (1 / b)) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by
  sorry

end find_numbers_l258_258647


namespace fran_speed_l258_258609

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l258_258609


namespace find_f_zero_l258_258694

variable (f : ℝ → ℝ)

def odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = -g (-x + 1)

def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 1) = g (-x - 1)

theorem find_f_zero
  (H1 : odd_function f)
  (H2 : even_function f)
  (H3 : f 4 = 6) :
  f 0 = -6 := by
  sorry

end find_f_zero_l258_258694


namespace equation_of_perpendicular_line_through_point_l258_258333

theorem equation_of_perpendicular_line_through_point :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), (a = 3) ∧ (b = 1) ∧ (x - 2 * y - 3 = 0 → y = (-(1/2)) * x + 3/2) ∧ (2 * a + b - 7 = 0) := sorry

end equation_of_perpendicular_line_through_point_l258_258333


namespace train_crossing_time_l258_258225

theorem train_crossing_time (length_of_train : ℝ) (speed_kmh : ℝ) :
  length_of_train = 180 →
  speed_kmh = 72 →
  (180 / (72 * (1000 / 3600))) = 9 :=
by 
  intros h1 h2
  sorry

end train_crossing_time_l258_258225


namespace find_f_7_l258_258007

noncomputable def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

theorem find_f_7 : f 7 = 17 :=
  by
  -- The proof steps will go here
  sorry

end find_f_7_l258_258007


namespace quadratic_inequality_solution_l258_258277

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : 1 + 2 = b / a)
  (h3 : 1 * 2 = c / a) :
  ∀ x : ℝ, cx^2 + bx + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1 / 2 :=
by
  sorry

end quadratic_inequality_solution_l258_258277


namespace symm_property_interval_condition_main_theorem_l258_258859

-- Define the function f with the given conditions
def f (x : ℝ) : ℝ := sorry -- Given function f, to be defined from the conditions

-- Define the conditions
theorem symm_property (f : ℝ → ℝ) : (∀ x, f(x+1) = f(-x+1)) := sorry
theorem interval_condition (f : ℝ → ℝ) : (∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) := sorry

-- Main theorem to be proved
theorem main_theorem (f : ℝ → ℝ) 
  (symm : ∀ x, f(x+1) = f(-x+1))
  (interval : ∀ x, (0 < x ∧ x < 1) → f(x) = Real.exp(-x)) : 
  f(Real.log 3) = 3 * Real.exp(-2) := sorry

end symm_property_interval_condition_main_theorem_l258_258859


namespace find_angle_F_l258_258450

variable (EF GH : ℝ) -- Lengths of sides EF and GH
variable (angle_E angle_F angle_G angle_H : ℝ) -- Angles at vertices E, F, G, and H

-- Conditions given in the problem
axiom EF_parallel_GH : EF ∥ GH
axiom angle_E_eq_3_angle_H : angle_E = 3 * angle_H
axiom angle_G_eq_2_angle_F : angle_G = 2 * angle_F

-- Target statement to prove
theorem find_angle_F : angle_F = 60 := by
  -- Conditions setup:
  have angle_F_plus_angle_G := 180 - angle_G ; sorry
  -- Solve for angle_F
  have angle_F_eq_60 := 180 / 3; sorry
  sorry

end find_angle_F_l258_258450


namespace boats_meeting_distance_l258_258216

theorem boats_meeting_distance (X : ℝ) 
  (H1 : ∃ (X : ℝ), (1200 - X) + 900 = X + 1200 + 300) 
  (H2 : X + 1200 + 300 = 2100 + X): 
  X = 300 :=
by
  sorry

end boats_meeting_distance_l258_258216


namespace intersecting_lines_a_value_l258_258776

theorem intersecting_lines_a_value :
  ∀ t a b : ℝ, (b = 12) ∧ (b = 2 * a + t) ∧ (t = 4) → a = 4 :=
by
  intros t a b h
  obtain ⟨hb1, hb2, ht⟩ := h
  sorry

end intersecting_lines_a_value_l258_258776


namespace smallest_value_of_x_l258_258930

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l258_258930


namespace exists_three_distinct_div_l258_258309

theorem exists_three_distinct_div (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ m : ℕ, ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ abc ∣ (x * y * z) ∧ m ≤ x ∧ x < m + 2*c ∧ m ≤ y ∧ y < m + 2*c ∧ m ≤ z ∧ z < m + 2*c :=
by
  sorry

end exists_three_distinct_div_l258_258309


namespace range_of_m_l258_258394

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (x^2 : ℝ) / (2 - m) + (y^2 : ℝ) / (m - 1) = 1 → 2 - m < 0 ∧ m - 1 > 0) →
  (∀ Δ : ℝ, Δ = 16 * (m - 2) ^ 2 - 16 → Δ < 0 → 1 < m ∧ m < 3) →
  (∀ (p q : Prop), p ∨ q ∧ ¬ q → p ∧ ¬ q) →
  m ≥ 3 :=
by
  intros h1 h2 h3
  sorry

end range_of_m_l258_258394


namespace find_x_value_l258_258336

theorem find_x_value (a b x : ℤ) (h : a * b = (a - 1) * (b - 1)) (h2 : x * 9 = 160) :
  x = 21 :=
sorry

end find_x_value_l258_258336


namespace tangent_line_eq_area_independent_of_a_l258_258571

open Real

section TangentLineAndArea

def curve (x : ℝ) := x^2 - 1

def tangentCurvey (x : ℝ) := x^2

noncomputable def tangentLine (a : ℝ) (ha : a > 0) : (ℝ → ℝ) :=
  if a > 1 then λ x => (2*(a + 1)) * x - (a+1)^2
  else λ x => (2*(a - 1)) * x - (a-1)^2

theorem tangent_line_eq (a : ℝ) (ha : a > 0) :
  ∃ (line : ℝ → ℝ), (line = tangentLine a ha) :=
sorry

theorem area_independent_of_a (a : ℝ) (ha : a > 0) :
  (∫ x in (a - 1)..a, (tangentCurvey x - tangentLine a ha x)) +
  (∫ x in a..(a + 1), (tangentCurvey x - tangentLine a ha x)) = (2 / 3 : Real) :=
sorry

end TangentLineAndArea

end tangent_line_eq_area_independent_of_a_l258_258571


namespace seashells_count_l258_258620

theorem seashells_count (mary_seashells : ℕ) (keith_seashells : ℕ) (cracked_seashells : ℕ) 
  (h_mary : mary_seashells = 2) (h_keith : keith_seashells = 5) (h_cracked : cracked_seashells = 9) :
  (mary_seashells + keith_seashells = 7) ∧ (cracked_seashells > mary_seashells + keith_seashells) → false := 
by {
  sorry
}

end seashells_count_l258_258620


namespace no_solution_exists_l258_258314

theorem no_solution_exists : 
  ¬ ∃ (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0), 
    45 * x = (35 / 100) * 900 ∧
    y^2 + x = 100 ∧
    z = x^3 * y - (2 * x + 1) / (y + 4) :=
by
  sorry

end no_solution_exists_l258_258314


namespace total_canoes_built_l258_258960

-- Defining basic variables and functions for the proof
variable (a : Nat := 5) -- Initial number of canoes in January
variable (r : Nat := 3) -- Common ratio
variable (n : Nat := 6) -- Number of months including January

-- Function to compute sum of the first n terms of a geometric series
def geometric_sum (a r n : Nat) : Nat :=
  a * (r^n - 1) / (r - 1)

-- The proposition we want to prove
theorem total_canoes_built : geometric_sum a r n = 1820 := by
  sorry

end total_canoes_built_l258_258960


namespace find_y_in_terms_of_x_l258_258389

theorem find_y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * (y - 1) + 3) : 
  y = (1 / 4) * x - (1 / 4) := 
by
  sorry

end find_y_in_terms_of_x_l258_258389


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258871

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258871


namespace sin_double_angle_l258_258164

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_double_angle_l258_258164


namespace first_term_exceeding_1000_l258_258396

variable (a₁ : Int := 2)
variable (d : Int := 3)

def arithmetic_sequence (n : Int) : Int :=
  a₁ + (n - 1) * d

theorem first_term_exceeding_1000 :
  ∃ n : Int, n = 334 ∧ arithmetic_sequence n > 1000 := by
  sorry

end first_term_exceeding_1000_l258_258396


namespace circumscribed_sphere_radius_l258_258497

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt (6 + Real.sqrt 20)) / 8

theorem circumscribed_sphere_radius (a : ℝ) :
  radius_of_circumscribed_sphere a = a * (Real.sqrt (6 + Real.sqrt 20)) / 8 :=
by
  sorry

end circumscribed_sphere_radius_l258_258497


namespace point_M_coordinates_l258_258470

/- Define the conditions -/

def isInFourthQuadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distanceToXAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.2 = d

def distanceToYAxis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  abs M.1 = d

/- Write the Lean theorem statement -/

theorem point_M_coordinates :
  ∀ (M : ℝ × ℝ), isInFourthQuadrant M ∧ distanceToXAxis M 3 ∧ distanceToYAxis M 4 → M = (4, -3) :=
by
  intro M
  sorry

end point_M_coordinates_l258_258470


namespace time_to_complete_project_alone_l258_258229

variable (A B : Type)
variable (day x : ℕ)
variable (work_rate_A : ℚ)
variable (work_rate_B : ℚ)

theorem time_to_complete_project_alone (hA : work_rate_A = 1 / x)
                                      (hB : work_rate_B = 1 / 30)
                                      (hAB_joint : ∀ (d : ℕ), d = 21 → 6 * (work_rate_A + work_rate_B) + 15 * work_rate_B = 1)
                                      (hx : ∀ (day : ℕ), day = 21 - 15 → true) :
  x = 20 := 
begin
  -- proof will go here
  sorry
end

end time_to_complete_project_alone_l258_258229


namespace movie_theater_charge_l258_258559

theorem movie_theater_charge 
    (charge_adult : ℝ) 
    (children : ℕ) 
    (adults : ℕ) 
    (total_receipts : ℝ) 
    (charge_child : ℝ) 
    (condition1 : charge_adult = 6.75) 
    (condition2 : children = adults + 20) 
    (condition3 : total_receipts = 405) 
    (condition4 : children = 48) 
    : charge_child = 4.5 :=
sorry

end movie_theater_charge_l258_258559


namespace binomial_7_4_eq_35_l258_258544

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l258_258544


namespace roots_equal_implies_a_eq_3_l258_258876

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l258_258876


namespace male_to_female_cat_weight_ratio_l258_258120

variable (w_f w_m w_t : ℕ)

def female_cat_weight : Prop := w_f = 2
def total_weight : Prop := w_t = 6
def male_cat_heavier : Prop := w_m > w_f

theorem male_to_female_cat_weight_ratio
  (h_female_cat_weight : female_cat_weight w_f)
  (h_total_weight : total_weight w_t)
  (h_male_cat_heavier : male_cat_heavier w_m w_f) :
  w_m = 4 ∧ w_t = w_f + w_m ∧ (w_m / w_f) = 2 :=
by
  sorry

end male_to_female_cat_weight_ratio_l258_258120


namespace number_of_ways_sum_of_primes_l258_258436

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l258_258436


namespace time_to_paint_remaining_rooms_l258_258655

-- Definitions for the conditions
def total_rooms : ℕ := 11
def time_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Statement of the problem
theorem time_to_paint_remaining_rooms : 
  total_rooms - painted_rooms = 9 →
  (total_rooms - painted_rooms) * time_per_room = 63 := 
by 
  intros h1
  sorry

end time_to_paint_remaining_rooms_l258_258655


namespace total_words_in_poem_l258_258325

theorem total_words_in_poem (s l w : ℕ) (h1 : s = 35) (h2 : l = 15) (h3 : w = 12) : 
  s * l * w = 6300 := 
by 
  -- the proof will be inserted here
  sorry

end total_words_in_poem_l258_258325


namespace sum_of_smallest_and_largest_l258_258199

theorem sum_of_smallest_and_largest (z : ℤ) (b m : ℤ) (h : even m) 
  (H_mean : z = (b + (b + 2 * (m - 1))) / 2) : 
  2 * z = b + b + 2 * (m - 1) :=
by 
  sorry

end sum_of_smallest_and_largest_l258_258199


namespace gcd_max_value_l258_258919

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l258_258919


namespace gcf_of_lcm_eq_15_l258_258511

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_of_lcm_eq_15 : gcf (lcm 9 15) (lcm 10 21) = 15 := by
  sorry

end gcf_of_lcm_eq_15_l258_258511


namespace set_intersection_complement_l258_258161

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {1, 3, 4, 6, 7}

theorem set_intersection_complement :
  A ∩ (U \ B) = {2, 5} := 
by
  sorry

end set_intersection_complement_l258_258161


namespace tickets_needed_l258_258728

def tickets_per_roller_coaster : ℕ := 5
def tickets_per_giant_slide : ℕ := 3
def roller_coaster_rides : ℕ := 7
def giant_slide_rides : ℕ := 4

theorem tickets_needed : tickets_per_roller_coaster * roller_coaster_rides + tickets_per_giant_slide * giant_slide_rides = 47 := 
by
  sorry

end tickets_needed_l258_258728


namespace pizza_consumption_order_l258_258382

theorem pizza_consumption_order :
  let e := 1/6
  let s := 1/4
  let n := 1/3
  let o := 1/8
  let j := 1 - e - s - n - o
  (n > s) ∧ (s > e) ∧ (e = j) ∧ (j > o) :=
by
  sorry

end pizza_consumption_order_l258_258382


namespace probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l258_258350

noncomputable def diameter := 19 -- mm
noncomputable def side_length := 50 -- mm, side length of each square
noncomputable def total_area := side_length^2 -- 2500 mm^2 for each square
noncomputable def coin_radius := diameter / 2 -- 9.5 mm

theorem probability_completely_inside_square : 
  (side_length - 2 * coin_radius)^2 / total_area = 961 / 2500 :=
by sorry

theorem probability_partial_one_edge :
  4 * ((side_length - 2 * coin_radius) * coin_radius) / total_area = 1178 / 2500 :=
by sorry

theorem probability_partial_two_edges_not_vertex :
  (4 * ((diameter)^2 - (coin_radius^2 * Real.pi / 4))) / total_area = (4 * 290.12) / 2500 :=
by sorry

theorem probability_vertex :
  4 * (coin_radius^2 * Real.pi / 4) / total_area = 4 * 70.88 / 2500 :=
by sorry

end probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l258_258350


namespace ceiling_floor_expression_l258_258125

theorem ceiling_floor_expression :
  (Int.ceil ((12:ℚ) / 5 * ((-19:ℚ) / 4 - 3)) - Int.floor (((12:ℚ) / 5) * Int.floor ((-19:ℚ) / 4)) = -6) :=
by 
  sorry

end ceiling_floor_expression_l258_258125


namespace XY_sum_l258_258291

theorem XY_sum (A B C D X Y : ℕ) 
  (h1 : A + B + C + D = 22) 
  (h2 : X = A + B) 
  (h3 : Y = C + D) 
  : X + Y = 4 := 
  sorry

end XY_sum_l258_258291


namespace average_age_l258_258306

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l258_258306


namespace correct_option_l258_258393

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end correct_option_l258_258393


namespace positive_number_l258_258788

theorem positive_number (x : ℝ) (h1 : 0 < x) (h2 : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := sorry

end positive_number_l258_258788


namespace polynomial_root_theorem_l258_258471

theorem polynomial_root_theorem
  (α β γ δ p q : ℝ)
  (h₁ : α + β = -p)
  (h₂ : α * β = 1)
  (h₃ : γ + δ = -q)
  (h₄ : γ * δ = 1) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
by
  sorry

end polynomial_root_theorem_l258_258471


namespace weekly_caloric_allowance_l258_258528

-- Define the given conditions
def average_daily_allowance : ℕ := 2000
def daily_reduction_goal : ℕ := 500
def intense_workout_extra_calories : ℕ := 300
def moderate_exercise_extra_calories : ℕ := 200
def days_intense_workout : ℕ := 2
def days_moderate_exercise : ℕ := 3
def days_rest : ℕ := 2

-- Lean statement to prove the total weekly caloric intake
theorem weekly_caloric_allowance :
  (days_intense_workout * (average_daily_allowance - daily_reduction_goal + intense_workout_extra_calories)) +
  (days_moderate_exercise * (average_daily_allowance - daily_reduction_goal + moderate_exercise_extra_calories)) +
  (days_rest * (average_daily_allowance - daily_reduction_goal)) = 11700 := by
  sorry

end weekly_caloric_allowance_l258_258528


namespace simplify_expression_l258_258963

variables (x y z : ℝ)

theorem simplify_expression (h₁ : x ≠ 2) (h₂ : y ≠ 3) (h₃ : z ≠ 4) : 
  ((x - 2) / (4 - z)) * ((y - 3) / (2 - x)) * ((z - 4) / (3 - y)) = -1 :=
by sorry

end simplify_expression_l258_258963


namespace number_of_prime_pairs_for_10003_l258_258428

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem number_of_prime_pairs_for_10003 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ 10003 = p + q :=
by {
  use [2, 10001],
  repeat { sorry }
}

end number_of_prime_pairs_for_10003_l258_258428


namespace savings_account_amount_l258_258241

-- Definitions and conditions from the problem
def checking_account_yen : ℕ := 6359
def total_yen : ℕ := 9844

-- Question we aim to prove - the amount in the savings account
def savings_account_yen : ℕ := total_yen - checking_account_yen

-- Lean statement to prove the equality
theorem savings_account_amount : savings_account_yen = 3485 :=
by
  sorry

end savings_account_amount_l258_258241


namespace sum_of_squares_of_non_zero_digits_from_10_to_99_l258_258733

-- Definition of the sum of squares of digits from 1 to 9
def P : ℕ := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2)

-- Definition of the sum of squares of the non-zero digits of the integers from 10 to 99
def T : ℕ := 20 * P

-- Theorem stating that T equals 5700
theorem sum_of_squares_of_non_zero_digits_from_10_to_99 : T = 5700 :=
by
  sorry

end sum_of_squares_of_non_zero_digits_from_10_to_99_l258_258733


namespace total_soccer_balls_l258_258917

theorem total_soccer_balls (boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ) 
  (h1 : boxes = 10) (h2 : packages_per_box = 8) (h3 : balls_per_package = 13) : 
  (boxes * packages_per_box * balls_per_package = 1040) :=
by 
  sorry

end total_soccer_balls_l258_258917


namespace smallest_prime_with_digit_sum_23_l258_258061

/-- Defines the digit sum of a natural number. -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Specifies that a number is prime and its digits sum to a given value. -/
def is_prime_with_digit_sum (n : ℕ) (sum : ℕ) : Prop :=
  Nat.Prime n ∧ digit_sum n = sum

/-- Defines the smallest prime whose digits sum to 23. -/
theorem smallest_prime_with_digit_sum_23 :
  ∃ n : ℕ, is_prime_with_digit_sum n 23 ∧ ∀ m : ℕ, is_prime_with_digit_sum m 23 → n ≤ m :=
  ⟨599, by
    split
      -- Proof that 599 is prime
      · exact Nat.prime_def_lt'.mpr (by norm_num1; apply_instance)
      -- Proof that the digit sum of 599 is 23
      · exact congr_arg (List.sum) (by simp)
    sorry
  ⟩

end smallest_prime_with_digit_sum_23_l258_258061


namespace factorize_expression_l258_258255

theorem factorize_expression (a x : ℝ) : a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
  sorry

end factorize_expression_l258_258255


namespace ali_less_nada_l258_258118

variable (Ali Nada John : ℕ)

theorem ali_less_nada
  (h_total : Ali + Nada + John = 67)
  (h_john_nada : John = 4 * Nada)
  (h_john : John = 48) :
  Nada - Ali = 5 :=
by
  sorry

end ali_less_nada_l258_258118


namespace prob_A_B_path_l258_258459

open ProbabilityTheory

structure Point (α : Type*) :=
(coord : α)

noncomputable def edge_prob (u v : Point ℝ) : ℝ := 1 / 2

noncomputable def prob_A_B_connected (A B C D : Point ℝ) (indep : ∀ u v w x: Point ℝ, u ≠ v → w ≠ x → Prob.indep_event (u, v) (w, x)) : ℝ :=
  3 / 4

theorem prob_A_B_path (A B C D : Point ℝ) (h : ∀ u v w x: Point ℝ, u ≠ v → w ≠ x → Prob.indep_event (u, v) (w, x)) (ncoplanar : ¬ (A.coord, B.coord, C.coord, D.coord).coplanar):
  prob_A_B_connected A B C D h = 3 / 4 :=
sorry

end prob_A_B_path_l258_258459


namespace find_factors_of_224_l258_258045

theorem find_factors_of_224 : ∃ (a b c : ℕ), a * b * c = 224 ∧ c = 2 * a ∧ a ≠ b ∧ b ≠ c :=
by
  -- Prove that the factors meeting the criteria exist
  sorry

end find_factors_of_224_l258_258045


namespace Fran_speed_l258_258607

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l258_258607


namespace exponentiation_rule_l258_258101

theorem exponentiation_rule (a : ℝ) : (a^4) * (a^4) = a^8 :=
by 
  sorry

end exponentiation_rule_l258_258101


namespace altitude_inequality_not_universally_true_l258_258616

noncomputable def altitudes (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a m_b m_c : ℝ, m_a ≤ m_b ∧ m_b ≤ m_c 

noncomputable def seg_to_orthocenter (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a_star m_b_star m_c_star : ℝ, True

theorem altitude_inequality (a b c m_a m_b m_c : ℝ) 
  (h₀ : a ≥ b) (h₁ : b ≥ c) (h₂ : m_a ≤ m_b) (h₃ : m_b ≤ m_c) :
  (a + m_a ≥ b + m_b) ∧ (b + m_b ≥ c + m_c) :=
by
  sorry

theorem not_universally_true (a b c m_a_star m_b_star m_c_star : ℝ)
  (h₀ : a ≥ b) (h₁ : b ≥ c) :
  ¬(a + m_a_star ≥ b + m_b_star ∧ b + m_b_star ≥ c + m_c_star) :=
by
  sorry

end altitude_inequality_not_universally_true_l258_258616


namespace calc_m_l258_258961

theorem calc_m (m : ℤ) (h : (64 : ℝ)^(1 / 3) = 2^m) : m = 2 :=
sorry

end calc_m_l258_258961


namespace operation_difference_l258_258830

def operation (x y : ℕ) : ℕ := x * y - 3 * x + y

theorem operation_difference : operation 5 9 - operation 9 5 = 16 :=
by
  sorry

end operation_difference_l258_258830


namespace total_payment_for_combined_shopping_trip_l258_258237

noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then amount * 0.9
  else 500 * 0.9 + (amount - 500) * 0.7

theorem total_payment_for_combined_shopping_trip :
  discount (168 + 423 / 0.9) = 546.6 :=
by
  sorry

end total_payment_for_combined_shopping_trip_l258_258237


namespace number_of_girls_l258_258174

open Rat

theorem number_of_girls 
  (G B : ℕ) 
  (h1 : G / B = 5 / 8)
  (h2 : G + B = 300) 
  : G = 116 := 
by
  sorry

end number_of_girls_l258_258174


namespace line_intersects_circle_l258_258944

-- Definitions
def radius : ℝ := 5
def distance_to_center : ℝ := 3

-- Theorem statement
theorem line_intersects_circle (r : ℝ) (d : ℝ) (h_r : r = radius) (h_d : d = distance_to_center) : d < r :=
by
  rw [h_r, h_d]
  exact sorry

end line_intersects_circle_l258_258944


namespace highest_place_value_734_48_l258_258791

theorem highest_place_value_734_48 : 
  (∃ k, 10^4 = k ∧ k * 10^4 ≤ 734 * 48 ∧ 734 * 48 < (k + 1) * 10^4) := 
sorry

end highest_place_value_734_48_l258_258791


namespace measure_angle_ACB_l258_258884

-- Definitions of angles and a given triangle
variable (α β γ : ℝ)
variable (angleABD angle75 : ℝ)
variable (triangleABC : Prop)

-- Conditions from the problem
def angle_supplementary : Prop := angleABD + α = 180
def sum_angles_triangle : Prop := α + β + γ = 180
def known_angle : Prop := β = 75
def angleABD_value : Prop := angleABD = 150

-- The theorem to prove
theorem measure_angle_ACB : 
  angle_supplementary angleABD α ∧
  sum_angles_triangle α β γ ∧
  known_angle β ∧
  angleABD_value angleABD
  → γ = 75 := by
  sorry


end measure_angle_ACB_l258_258884


namespace angle_C_is_108_l258_258323

theorem angle_C_is_108
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : B < C)
  (h3 : C < D)
  (h4 : D < E)
  (h5 : B - A = C - B)
  (h6 : C - B = D - C)
  (h7 : D - C = E - D)
  (angle_sum : A + B + C + D + E = 540) :
  C = 108 := 
sorry

end angle_C_is_108_l258_258323


namespace martin_probability_360_feet_l258_258747

noncomputable def probability_walking_distance_within_360_feet : ℚ :=
  let total_gates := 15
  let distance_between_gates := 90
  let max_distance := 360
  let total_possible_changes := total_gates * (total_gates - 1)

  let feasible_choices_per_gate :=
    [4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4].sum

  (feasible_choices_per_gate : ℚ) / total_possible_changes

theorem martin_probability_360_feet : probability_walking_distance_within_360_feet = 59 / 105 :=
by
  sorry

end martin_probability_360_feet_l258_258747


namespace solve_equation_l258_258193

theorem solve_equation : ∀ x : ℝ, (3 * (x - 2) + 1 = x - (2 * x - 1)) → x = 3 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l258_258193


namespace largest_number_among_options_l258_258526

def option_a : ℝ := -abs (-4)
def option_b : ℝ := 0
def option_c : ℝ := 1
def option_d : ℝ := -( -3)

theorem largest_number_among_options : 
  max (max option_a (max option_b option_c)) option_d = option_d := by
  sorry

end largest_number_among_options_l258_258526


namespace james_writing_time_l258_258455

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l258_258455


namespace probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258873

noncomputable theory
open ProbabilityTheory

-- Definitions based on given conditions
def events := {1, 2, 3, 4, 5, 6}
def eventA := {1, 2, 3}
def eventB := {1, 2, 4}

def numTrials : ℕ := 10
def numA : ℕ := 6
def pA : ℚ := 1 / 2
def pB_given_A : ℚ := 2 / 3
def pB_given_Ac : ℚ := 1 / 3

-- Theorem for probability that B does not occur given A occurred 6 times.
theorem probability_B_does_not_occur_given_A_6_occur :
  -- The probability of B not occurring given A occurred exactly 6 times.
  -- Should be approximately 2.71 * 10^(-4)
  true := sorry

-- Theorem for the expected number of times B occurs.
theorem expected_value_B_occurances : 
  -- The expected value of the number of occurrences of event B given the conditions.
  -- Should be 16 / 3
  true := sorry

end probability_B_does_not_occur_given_A_6_occur_expected_value_B_occurances_l258_258873


namespace find_k_l258_258855

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l258_258855


namespace find_angle_F_l258_258452

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l258_258452


namespace largest_possible_value_of_sum_of_products_l258_258498

open Finset

noncomputable def largest_sum_of_products : ℕ :=
  let s : Finset ℕ := {1, 2, 3, 4}
  let products := s.powerset.filter(λ t, t.card = 4).image(λ t,
    let ⟨a, b, c, d⟩ := ⟨t.to_list.nth 0, t.to_list.nth 1, t.to_list.nth 2, t.to_list.nth 3⟩ in
    option.get_or_else (a.feval nat 0 * b.feval nat 0 + b.feval nat 0 * c.feval nat 0 
    + c.feval nat 0 * d.feval nat 0 + d.feval nat 0 * a.feval nat 0) 0)
  products.max'

theorem largest_possible_value_of_sum_of_products (a b c d : ℕ) (h₁ : a ∈ {1, 2, 3, 4})
  (h₂ : b ∈ {1, 2, 3, 4}) (h₃ : c ∈ {1, 2, 3, 4}) (h₄ : d ∈ {1, 2, 3, 4})
  (h₅ : a ≠ b) (h₆ : b ≠ c) (h₇ : c ≠ d) (h₈ : d ≠ a) (h₉ : a ≠ c) (h₁₀ : b ≠ d) :
  ab + bc + cd + da = 25 :=
sorry

end largest_possible_value_of_sum_of_products_l258_258498


namespace least_possible_sum_l258_258913

theorem least_possible_sum {c d : ℕ} (hc : c ≥ 2) (hd : d ≥ 2) (h : 3 * c + 6 = 6 * d + 3) : c + d = 5 :=
by
  sorry

end least_possible_sum_l258_258913


namespace George_spending_l258_258841

theorem George_spending (B m s : ℝ) (h1 : m = 0.25 * (B - s)) (h2 : s = 0.05 * (B - m)) : 
  (m + s) / B = 1 := 
by
  sorry

end George_spending_l258_258841


namespace option_b_correct_l258_258283

theorem option_b_correct (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3: a ≠ 1) (h4: b ≠ 1) (h5 : 0 < m) (h6 : m < 1) :
  m^a < m^b :=
sorry

end option_b_correct_l258_258283


namespace area_of_square_l258_258662

-- Define the problem setting and the conditions
def square (side_length : ℝ) : Prop :=
  ∃ (width height : ℝ), width * height = side_length^2
    ∧ width = 5
    ∧ side_length / height = 5 / height

-- State the theorem to be proven
theorem area_of_square (side_length : ℝ) (width height : ℝ) (h1 : width = 5) (h2: side_length = 5 + 2 * height): 
  square side_length → side_length^2 = 400 :=
by
  intro h
  sorry

end area_of_square_l258_258662


namespace total_pages_to_read_l258_258020

theorem total_pages_to_read 
  (total_books : ℕ)
  (pages_per_book : ℕ)
  (books_read_first_month : ℕ)
  (books_remaining_second_month : ℕ) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  books_remaining_second_month = (total_books - books_read_first_month) / 2 →
  ((total_books * pages_per_book) - ((books_read_first_month + books_remaining_second_month) * pages_per_book) = 1000) :=
by
  sorry

end total_pages_to_read_l258_258020


namespace intersection_correct_l258_258107

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := { y | ∃ x ∈ A, y = 2 * x - 1 }

def intersection : Set ℕ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_correct : intersection = {1, 3} := by
  sorry

end intersection_correct_l258_258107


namespace find_xy_l258_258682

theorem find_xy (x y : ℕ) (hx : x ≥ 1) (hy : y ≥ 1) : 
  2^x - 5 = 11^y ↔ (x = 4 ∧ y = 1) :=
by sorry

end find_xy_l258_258682


namespace cubic_ineq_solution_l258_258383

theorem cubic_ineq_solution (x : ℝ) :
  (4 < x ∧ x < 4 + 2 * Real.sqrt 3) ∨ (x > 4 + 2 * Real.sqrt 3) → (x^3 - 12 * x^2 + 44 * x - 16 > 0) :=
by
  sorry

end cubic_ineq_solution_l258_258383


namespace value_of_k_l258_258373

theorem value_of_k {k : ℝ} :
  (∀ x : ℝ, (x^2 + k * x + 24 > 0) ↔ (x < -6 ∨ x > 4)) →
  k = 2 :=
by
  sorry

end value_of_k_l258_258373


namespace family_members_l258_258720

theorem family_members (cost_purify : ℝ) (water_per_person : ℝ) (total_cost : ℝ) 
  (h1 : cost_purify = 1) (h2 : water_per_person = 1 / 2) (h3 : total_cost = 3) : 
  total_cost / (cost_purify * water_per_person) = 6 :=
by
  sorry

end family_members_l258_258720


namespace range_of_dot_product_l258_258579

theorem range_of_dot_product
  (a b : ℝ)
  (h: ∃ (A B : ℝ × ℝ), (A ≠ B) ∧ ∃ m n : ℝ, A = (m, n) ∧ B = (-m, -n) ∧ m^2 + (n^2 / 9) = 1)
  : ∃ r : Set ℝ, r = (Set.Icc 41 49) :=
  sorry

end range_of_dot_product_l258_258579


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l258_258052

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l258_258052


namespace power_function_odd_f_m_plus_1_l258_258656

noncomputable def f (x : ℝ) (m : ℝ) := x^(2 + m)

theorem power_function_odd_f_m_plus_1 (m : ℝ) (h_odd : ∀ x : ℝ, f (-x) m = -f x m)
  (h_domain : -1 ≤ m) : f (m + 1) m = 1 := by
  sorry

end power_function_odd_f_m_plus_1_l258_258656


namespace find_n_of_geometric_sum_l258_258918

-- Define the first term and common ratio of the sequence
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3

-- Define the sum of the first n terms of the geometric sequence
def S_n (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

-- Mathematical statement to be proved
theorem find_n_of_geometric_sum (h : S_n 5 = 80 / 243) : ∃ n, S_n n = 80 / 243 ↔ n = 5 :=
by
  sorry

end find_n_of_geometric_sum_l258_258918


namespace value_of_a_l258_258591

theorem value_of_a :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - -1)^2 + (y - 1)^2 = 4) := sorry

end value_of_a_l258_258591


namespace number_of_ways_sum_of_primes_l258_258438

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l258_258438


namespace rita_canoe_distance_l258_258756

theorem rita_canoe_distance 
  (up_speed : ℕ) (down_speed : ℕ)
  (wind_up_decrease : ℕ) (wind_down_increase : ℕ)
  (total_time : ℕ) 
  (effective_up_speed : ℕ := up_speed - wind_up_decrease)
  (effective_down_speed : ℕ := down_speed + wind_down_increase)
  (T_up : ℚ := D / effective_up_speed)
  (T_down : ℚ := D / effective_down_speed) :
  (T_up + T_down = total_time) ->
  (D = 7) := 
by
  sorry

-- Parameters as defined in the problem
def up_speed : ℕ := 3
def down_speed : ℕ := 9
def wind_up_decrease : ℕ := 2
def wind_down_increase : ℕ := 4
def total_time : ℕ := 8

end rita_canoe_distance_l258_258756


namespace find_x_l258_258794

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x_l258_258794


namespace smallest_w_l258_258515

theorem smallest_w (w : ℕ) (h1 : Nat.gcd 1452 w = 1) (h2 : 2 ∣ w ∧ 3 ∣ w ∧ 13 ∣ w) :
  (∃ (w : ℕ), 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w ∧ w > 0) ∧
  ∀ (w' : ℕ), (2^4 ∣ 1452 * w' ∧ 3^3 ∣ 1452 * w' ∧ 13^3 ∣ 1452 * w' ∧ w' > 0) → w ≤ w' :=
  sorry

end smallest_w_l258_258515


namespace log_equivalence_l258_258576

theorem log_equivalence :
  (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 :=
by
  sorry

end log_equivalence_l258_258576


namespace furniture_store_revenue_increase_l258_258653

noncomputable def percentage_increase_in_gross (P R : ℕ) : ℚ :=
  ((0.80 * P) * (1.70 * R) - (P * R)) / (P * R) * 100

theorem furniture_store_revenue_increase (P R : ℕ) :
  percentage_increase_in_gross P R = 36 := 
by
  -- We include the conditions directly in the proof.
  -- Follow theorem from the given solution.
  sorry

end furniture_store_revenue_increase_l258_258653


namespace time_after_2023_hours_l258_258633

theorem time_after_2023_hours (current_time : ℕ) (hours_later : ℕ) (modulus : ℕ) : 
    (current_time = 3) → 
    (hours_later = 2023) → 
    (modulus = 12) → 
    ((current_time + (hours_later % modulus)) % modulus = 2) :=
by 
    intros h1 h2 h3
    rw [h1, h2, h3]
    sorry

end time_after_2023_hours_l258_258633


namespace angle_B_in_progression_l258_258887

theorem angle_B_in_progression (A B C a b c : ℝ) (h1: A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) 
(h2: B - A = C - B) (h3: b^2 - a^2 = a * c) (h4: A + B + C = Real.pi) : 
B = 2 * Real.pi / 7 := sorry

end angle_B_in_progression_l258_258887


namespace number_of_raised_beds_l258_258240

def length_feed := 8
def width_feet := 4
def height_feet := 1
def cubic_feet_per_bag := 4
def total_bags_needed := 16

theorem number_of_raised_beds :
  ∀ (length_feed width_feet height_feet : ℕ) (cubic_feet_per_bag total_bags_needed : ℕ),
    (length_feed * width_feet * height_feet) / cubic_feet_per_bag = 8 →
    total_bags_needed / (8 : ℕ) = 2 :=
by sorry

end number_of_raised_beds_l258_258240


namespace norm_of_5v_l258_258979

noncomputable def norm_scale (v : ℝ × ℝ) (c : ℝ) : ℝ := c * (Real.sqrt (v.1^2 + v.2^2))

theorem norm_of_5v (v : ℝ × ℝ) (h : Real.sqrt (v.1^2 + v.2^2) = 6) : norm_scale v 5 = 30 := by
  sorry

end norm_of_5v_l258_258979


namespace smallest_prime_with_digit_sum_23_l258_258076

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l258_258076


namespace constant_two_l258_258618

theorem constant_two (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) (c : ℕ) (n : ℕ) (h_n : n = c * p) (h_even_divisors : ∀ d : ℕ, d ∣ n → (d % 2 = 0) → d = 2) : c = 2 := by
  sorry

end constant_two_l258_258618


namespace company_annual_income_l258_258658

variable {p a : ℝ}

theorem company_annual_income (h : 280 * p + (a - 280) * (p + 2) = a * (p + 0.25)) : a = 320 := 
sorry

end company_annual_income_l258_258658


namespace symmetric_periodic_l258_258693

theorem symmetric_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a ≠ b)
  (h2 : ∀ x : ℝ, f (a - x) = f (a + x))
  (h3 : ∀ x : ℝ, f (b - x) = f (b + x)) :
  ∀ x : ℝ, f x = f (x + 2 * (b - a)) :=
by
  sorry

end symmetric_periodic_l258_258693


namespace length_of_greater_segment_l258_258357

theorem length_of_greater_segment (x : ℤ) (h1 : (x + 2)^2 - x^2 = 32) : x + 2 = 9 := by
  sorry

end length_of_greater_segment_l258_258357


namespace GCF_LCM_calculation_l258_258510

theorem GCF_LCM_calculation : 
  GCD (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end GCF_LCM_calculation_l258_258510


namespace smallest_prime_with_digit_sum_23_l258_258081

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l258_258081


namespace no_prime_sum_10003_l258_258425

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l258_258425


namespace unique_sum_of_two_primes_l258_258445

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l258_258445


namespace pizza_slice_division_l258_258506

theorem pizza_slice_division : 
  ∀ (num_coworkers num_pizzas slices_per_pizza : ℕ),
  num_coworkers = 12 →
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  (num_pizzas * slices_per_pizza) / num_coworkers = 2 := 
by
  intros num_coworkers num_pizzas slices_per_pizza h_coworkers h_pizzas h_slices
  rw [h_coworkers, h_pizzas, h_slices]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

end pizza_slice_division_l258_258506


namespace lines_intersect_l258_258948

def line1 (t : ℝ) : ℝ × ℝ :=
  (1 - 2 * t, 2 + 4 * t)

def line2 (u : ℝ) : ℝ × ℝ :=
  (3 + u, 5 + 3 * u)

theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (1.2, 1.6) :=
by
  sorry

end lines_intersect_l258_258948


namespace makarala_meetings_percentage_l258_258746

def work_day_to_minutes (hours: ℕ) : ℕ :=
  60 * hours

def total_meeting_time (first: ℕ) (second: ℕ) : ℕ :=
  let third := first + second
  first + second + third

def percentage_of_day_spent (meeting_time: ℕ) (work_day_time: ℕ) : ℚ :=
  (meeting_time : ℚ) / (work_day_time : ℚ) * 100

theorem makarala_meetings_percentage
  (work_hours: ℕ)
  (first_meeting: ℕ)
  (second_meeting: ℕ)
  : percentage_of_day_spent (total_meeting_time first_meeting second_meeting) (work_day_to_minutes work_hours) = 37.5 :=
by
  sorry

end makarala_meetings_percentage_l258_258746


namespace amoeba_population_after_5_days_l258_258196

theorem amoeba_population_after_5_days 
  (initial : ℕ)
  (split_factor : ℕ)
  (days : ℕ)
  (h_initial : initial = 2)
  (h_split : split_factor = 3)
  (h_days : days = 5) :
  (initial * split_factor ^ days) = 486 :=
by sorry

end amoeba_population_after_5_days_l258_258196


namespace find_f_2011_l258_258152

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry  -- Placeholder, since f is only defined in (0, 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_2011 : f 2011 = -2 :=
by
  -- Use properties of f to reduce and eventually find f(2011)
  sorry

end find_f_2011_l258_258152


namespace amphibians_count_l258_258404

-- Define the conditions
def frogs : Nat := 7
def salamanders : Nat := 4
def tadpoles : Nat := 30
def newt : Nat := 1

-- Define the total number of amphibians observed by Hunter
def total_amphibians : Nat := frogs + salamanders + tadpoles + newt

-- State the theorem
theorem amphibians_count : total_amphibians = 42 := 
by 
  -- proof goes here
  sorry

end amphibians_count_l258_258404


namespace infinite_solutions_abs_eq_ax_minus_2_l258_258565

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end infinite_solutions_abs_eq_ax_minus_2_l258_258565


namespace sophia_estimate_larger_l258_258600

theorem sophia_estimate_larger (x y a b : ℝ) (hx : x > y) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end sophia_estimate_larger_l258_258600


namespace green_pill_cost_l258_258243

variable (x : ℝ) -- cost of a green pill in dollars
variable (y : ℝ) -- cost of a pink pill in dollars
variable (total_cost : ℝ) -- total cost for 21 days

theorem green_pill_cost
  (h1 : x = y + 2) -- a green pill costs $2 more than a pink pill
  (h2 : total_cost = 819) -- total cost for 21 days is $819
  (h3 : ∀ n, n = 21 ∧ total_cost / n = (x + y)) :
  x = 20.5 :=
by
  sorry

end green_pill_cost_l258_258243


namespace smallest_prime_with_digit_sum_23_l258_258089

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Nat.Prime p ∧ sum_digits p = 23 ∧ ∀ q : ℕ, Nat.Prime q ∧ sum_digits q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258089


namespace a_3_value_l258_258399

def arithmetic_seq (a: ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_value :
  ∃ a : ℕ → ℤ, a 1 = 19 ∧ arithmetic_seq a ∧ a 3 = 13 :=
by
  sorry

end a_3_value_l258_258399


namespace geometric_sequence_sum_l258_258699

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n : ℕ, a (n + 1) = a n * q)
  (h₁ : a 3 = 4) (h₂ : a 2 + a 4 = -10) (h₃ : |q| > 1) : 
  (a 0 + a 1 + a 2 + a 3 = -5) := 
by 
  sorry

end geometric_sequence_sum_l258_258699


namespace increasing_sequence_range_l258_258398

theorem increasing_sequence_range (a : ℝ) (f : ℝ → ℝ) (a_n : ℕ+ → ℝ) :
  (∀ n : ℕ+, a_n n = f n) →
  (∀ n m : ℕ+, n < m → a_n n < a_n m) →
  (∀ x : ℝ, f x = if  x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6) ) →
  2 < a ∧ a < 3 :=
by
  sorry

end increasing_sequence_range_l258_258398


namespace find_radius_l258_258908

-- Definition of the conditions
def area_of_sector : ℝ := 10 -- The area of the sector in square centimeters
def arc_length : ℝ := 4     -- The arc length of the sector in centimeters

-- The radius of the circle we want to prove
def radius (r : ℝ) : Prop :=
  (r * 4) / 2 = 10

-- The theorem to be proved
theorem find_radius : ∃ r : ℝ, radius r :=
by
  use 5
  unfold radius
  norm_num

end find_radius_l258_258908


namespace train_length_l258_258822

theorem train_length (time : ℝ) (speed_in_kmph : ℝ) (speed_in_mps : ℝ) (length_of_train : ℝ) :
  (time = 6) →
  (speed_in_kmph = 96) →
  (speed_in_mps = speed_in_kmph * (5 / 18)) →
  length_of_train = speed_in_mps * time →
  length_of_train = 480 := by
  sorry

end train_length_l258_258822


namespace range_of_a_l258_258717

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ Set.Icc (-1 : ℝ) 1, |4^x_0 - a * 2^x_0 + 1| ≤ 2^(x_0 + 1)) →
  0 ≤ a ∧ a ≤ (9/2) :=
by
  sorry

end range_of_a_l258_258717


namespace closest_perfect_square_to_315_l258_258933

theorem closest_perfect_square_to_315 : ∃ n : ℤ, n^2 = 324 ∧
  (∀ m : ℤ, m ≠ n → (abs (315 - m^2) > abs (315 - n^2))) := 
sorry

end closest_perfect_square_to_315_l258_258933


namespace convex_polygon_from_non_overlapping_rectangles_is_rectangle_l258_258321

def isConvexPolygon (P : Set Point) : Prop := sorry
def canBeFormedByNonOverlappingRectangles (P : Set Point) (rects: List (Set Point)) : Prop := sorry
def isRectangle (P : Set Point) : Prop := sorry

theorem convex_polygon_from_non_overlapping_rectangles_is_rectangle
  (P : Set Point)
  (rects : List (Set Point))
  (h_convex : isConvexPolygon P)
  (h_form : canBeFormedByNonOverlappingRectangles P rects) :
  isRectangle P :=
sorry

end convex_polygon_from_non_overlapping_rectangles_is_rectangle_l258_258321


namespace real_solution_for_any_y_l258_258136

theorem real_solution_for_any_y (x : ℝ) :
  (∀ y z : ℝ, x^2 + y^2 + z^2 + 2 * x * y * z = 1 → ∃ z : ℝ,  x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) :=
by sorry

end real_solution_for_any_y_l258_258136


namespace slices_per_person_l258_258503

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l258_258503


namespace odd_square_mod_eight_l258_258018

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l258_258018


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l258_258868

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l258_258868


namespace length_of_each_part_l258_258104

-- Conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def parts_count : ℕ := 4

-- Question
theorem length_of_each_part : total_length_in_inches / parts_count = 20 :=
by
  -- leave the proof as a sorry
  sorry

end length_of_each_part_l258_258104


namespace probability_of_drawing_red_ball_l258_258598

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l258_258598


namespace even_and_monotonically_increasing_f3_l258_258957

noncomputable def f1 (x : ℝ) : ℝ := x^3
noncomputable def f2 (x : ℝ) : ℝ := -x^2 + 1
noncomputable def f3 (x : ℝ) : ℝ := abs x + 1
noncomputable def f4 (x : ℝ) : ℝ := 2^(-abs x)

theorem even_and_monotonically_increasing_f3 :
  (∀ x, f3 x = f3 (-x)) ∧ (∀ x > 0, ∀ y > x, f3 y > f3 x) := 
sorry

end even_and_monotonically_increasing_f3_l258_258957


namespace coefficient_of_monomial_l258_258332

theorem coefficient_of_monomial : 
  ∀ (m n : ℝ), -((2 * Real.pi) / 3) * m * (n ^ 5) = -((2 * Real.pi) / 3) * m * (n ^ 5) :=
by
  sorry

end coefficient_of_monomial_l258_258332


namespace incorrect_statement_for_proportional_function_l258_258847

theorem incorrect_statement_for_proportional_function (x y : ℝ) : y = -5 * x →
  ¬ (∀ x, (x > 0 → y > 0) ∧ (x < 0 → y < 0)) :=
by
  sorry

end incorrect_statement_for_proportional_function_l258_258847


namespace correct_option_C_l258_258392

-- Define points A, B and C given their coordinates and conditions
structure Point (α : Type _) :=
(x : α)
(y : α)

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

variables (a b c d : ℝ)
variable hA : Point ℝ := ⟨a, 2⟩
variable hB : Point ℝ := ⟨b, 6⟩
variable hC : Point ℝ := ⟨c, d⟩
variables (ha_ON_parabola : hA.y = parabola hA.x)
          (hb_ON_parabola : hB.y = parabola hB.x)
          (hc_ON_parabola : hC.y = parabola hC.x)
          (hd_lt_one : d < 1)

theorem correct_option_C (ha_lt_0 : a < 0) (hb_gt_0 : b > 0) : a < c ∧ c < b :=
by
-- Proof will be done here, currently left as sorry just to state the theorem.
sorry

end correct_option_C_l258_258392


namespace sqrt_expression_eq_36_l258_258371

theorem sqrt_expression_eq_36 : (Real.sqrt ((3^2 + 3^3)^2)) = 36 := 
by
  sorry

end sqrt_expression_eq_36_l258_258371


namespace arithmetic_sequence_value_l258_258177

variable (a : ℕ → ℝ)
variable (a₁ d a₇ a₅ : ℝ)
variable (h_seq : ∀ n, a n = a₁ + (n - 1) * d)
variable (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120)

theorem arithmetic_sequence_value :
  a 7 - 1/3 * a 5 = 16 :=
sorry

end arithmetic_sequence_value_l258_258177


namespace smallest_prime_digit_sum_23_l258_258080

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l258_258080


namespace domain_of_function_l258_258372

noncomputable def function_domain := {x : ℝ | x * (3 - x) ≥ 0 ∧ x - 1 ≥ 0 }

theorem domain_of_function: function_domain = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end domain_of_function_l258_258372


namespace average_donation_is_integer_l258_258110

variable (num_classes : ℕ) (students_per_class : ℕ) (num_teachers : ℕ) (total_donation : ℕ)

def valid_students (n : ℕ) : Prop := 30 < n ∧ n ≤ 45

theorem average_donation_is_integer (h_classes : num_classes = 14)
                                    (h_teachers : num_teachers = 35)
                                    (h_donation : total_donation = 1995)
                                    (h_students_per_class : valid_students students_per_class)
                                    (h_total_people : ∃ n, 
                                      n = num_teachers + num_classes * students_per_class ∧ 30 < students_per_class ∧ students_per_class ≤ 45) :
  total_donation % (num_teachers + num_classes * students_per_class) = 0 ∧ 
  total_donation / (num_teachers + num_classes * students_per_class) = 3 := 
sorry

end average_donation_is_integer_l258_258110


namespace find_value_of_x_l258_258726

theorem find_value_of_x (a b c d e f x : ℕ) (h1 : a ≠ 1 ∧ a ≠ 6 ∧ b ≠ 1 ∧ b ≠ 6 ∧ c ≠ 1 ∧ c ≠ 6 ∧ d ≠ 1 ∧ d ≠ 6 ∧ e ≠ 1 ∧ e ≠ 6 ∧ f ≠ 1 ∧ f ≠ 6 ∧ x ≠ 1 ∧ x ≠ 6)
  (h2 : a + x + d = 18)
  (h3 : b + x + f = 18)
  (h4 : c + x + 6 = 18)
  (h5 : a + b + c + d + e + f + x + 6 + 1 = 45) :
  x = 7 :=
sorry

end find_value_of_x_l258_258726


namespace scientific_notation_of_12000000000_l258_258480

theorem scientific_notation_of_12000000000 :
  12000000000 = 1.2 * 10^10 :=
by sorry

end scientific_notation_of_12000000000_l258_258480


namespace distance_A_B_l258_258705

variable (x : ℚ)

def pointA := x
def pointB := 1
def pointC := -1

theorem distance_A_B : |pointA x - pointB| = |x - 1| := by
  sorry

end distance_A_B_l258_258705


namespace odd_squarefree_integers_1_to_199_l258_258833

noncomputable def count_squarefree_odd_integers (n : ℕ) :=
  n - List.sum [
    n / 18,   -- for 3^2 = 9
    n / 50,   -- for 5^2 = 25
    n / 98,   -- for 7^2 = 49
    n / 162,  -- for 9^2 = 81
    n / 242,  -- for 11^2 = 121
    n / 338   -- for 13^2 = 169
  ]

theorem odd_squarefree_integers_1_to_199 : count_squarefree_odd_integers 198 = 79 := 
by
  sorry

end odd_squarefree_integers_1_to_199_l258_258833


namespace num_rectangular_arrays_with_36_chairs_l258_258815

theorem num_rectangular_arrays_with_36_chairs :
  ∃ n : ℕ, (∀ r c : ℕ, r * c = 36 ∧ r ≥ 2 ∧ c ≥ 2 ↔ n = 7) :=
sorry

end num_rectangular_arrays_with_36_chairs_l258_258815


namespace sum_of_two_primes_unique_l258_258430

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l258_258430


namespace average_age_is_35_l258_258303

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l258_258303


namespace road_trip_cost_l258_258384

theorem road_trip_cost 
  (x : ℝ)
  (initial_cost_per_person: ℝ) 
  (redistributed_cost_per_person: ℝ)
  (cost_difference: ℝ) :
  initial_cost_per_person = x / 4 →
  redistributed_cost_per_person = x / 7 →
  cost_difference = 8 →
  initial_cost_per_person - redistributed_cost_per_person = cost_difference →
  x = 74.67 :=
by
  intro h1 h2 h3 h4
  -- starting the proof
  rw [h1, h2] at h4
  sorry

end road_trip_cost_l258_258384


namespace students_going_to_tournament_l258_258465

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end students_going_to_tournament_l258_258465


namespace urea_moles_produced_l258_258684

-- Define the reaction
def chemical_reaction (CO2 NH3 Urea Water : ℕ) :=
  CO2 = 1 ∧ NH3 = 2 ∧ Urea = 1 ∧ Water = 1

-- Given initial moles of reactants
def initial_moles (CO2 NH3 : ℕ) :=
  CO2 = 1 ∧ NH3 = 2

-- The main theorem to prove
theorem urea_moles_produced (CO2 NH3 Urea Water : ℕ) :
  initial_moles CO2 NH3 → chemical_reaction CO2 NH3 Urea Water → Urea = 1 :=
by
  intro H1 H2
  rcases H1 with ⟨HCO2, HNH3⟩
  rcases H2 with ⟨HCO2', HNH3', HUrea, _⟩
  sorry

end urea_moles_produced_l258_258684


namespace brady_june_hours_l258_258672

variable (x : ℕ) -- Number of hours worked every day in June

def hoursApril : ℕ := 6 * 30 -- Total hours in April
def hoursSeptember : ℕ := 8 * 30 -- Total hours in September
def hoursJune (x : ℕ) : ℕ := x * 30 -- Total hours in June
def totalHours (x : ℕ) : ℕ := hoursApril + hoursJune x + hoursSeptember -- Total hours over three months
def averageHours (x : ℕ) : ℕ := totalHours x / 3 -- Average hours per month

theorem brady_june_hours (h : averageHours x = 190) : x = 5 :=
by
  sorry

end brady_june_hours_l258_258672


namespace average_speed_monday_to_wednesday_l258_258203

theorem average_speed_monday_to_wednesday :
  ∃ x : ℝ, (∀ (total_hours total_distance thursday_friday_distance : ℝ),
    total_hours = 2 * 5 ∧
    thursday_friday_distance = 9 * 2 * 2 ∧
    total_distance = 108 ∧
    total_distance - thursday_friday_distance = x * (2 * 3))
    → x = 12 :=
sorry

end average_speed_monday_to_wednesday_l258_258203


namespace circular_arc_sum_l258_258967

theorem circular_arc_sum (n : ℕ) (h₁ : n > 0) :
  ∀ s : ℕ, (1 ≤ s ∧ s ≤ (n * (n + 1)) / 2) →
  ∃ arc_sum : ℕ, arc_sum = s := 
by
  sorry

end circular_arc_sum_l258_258967


namespace crackers_given_to_friends_l258_258622

theorem crackers_given_to_friends (crackers_per_friend : ℕ) (number_of_friends : ℕ) (h1 : crackers_per_friend = 6) (h2 : number_of_friends = 6) : (crackers_per_friend * number_of_friends) = 36 :=
by
  sorry

end crackers_given_to_friends_l258_258622


namespace log_ordering_l258_258266

theorem log_ordering {x a b c : ℝ} (h1 : 1 < x) (h2 : x < 10) (ha : a = Real.log x^2) (hb : b = Real.log (Real.log x)) (hc : c = (Real.log x)^2) :
  a > c ∧ c > b :=
by
  sorry

end log_ordering_l258_258266


namespace intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l258_258160

-- Define the solution sets A and B given conditions
def solution_set_A (a : ℝ) : Set ℝ :=
  { x | |x - 1| ≤ a }

def solution_set_B : Set ℝ :=
  { x | (x - 2) * (x + 2) > 0 }

theorem intersection_A_B_when_a_eq_2 :
  solution_set_A 2 ∩ solution_set_B = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ (a : ℝ), solution_set_A a ∩ solution_set_B = ∅ → 0 < a ∧ a ≤ 1 :=
by
  sorry

end intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l258_258160


namespace max_median_soda_cans_l258_258198

theorem max_median_soda_cans (total_customers total_cans : ℕ) 
    (h_customers : total_customers = 120)
    (h_cans : total_cans = 300) 
    (h_min_cans_per_customer : ∀ (n : ℕ), n < total_customers → 2 ≤ n) :
    ∃ (median : ℝ), median = 3.5 := 
sorry

end max_median_soda_cans_l258_258198


namespace floor_length_l258_258036

theorem floor_length (b l : ℝ)
  (h1 : l = 3 * b)
  (h2 : 3 * b^2 = 484 / 3) :
  l = 22 := 
sorry

end floor_length_l258_258036


namespace proof_problem_l258_258709

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a + b = 2 * c) ∧ (a * b = -5 * d) ∧ (c + d = 2 * a) ∧ (c * d = -5 * b)

theorem proof_problem (a b c d : ℝ) (h : problem_statement a b c d) : a + b + c + d = 30 :=
by
  sorry

end proof_problem_l258_258709


namespace penelope_saving_days_l258_258752

theorem penelope_saving_days :
  ∀ (daily_savings total_saved : ℕ),
  daily_savings = 24 ∧ total_saved = 8760 →
    total_saved / daily_savings = 365 :=
by
  rintro _ _ ⟨rfl, rfl⟩
  sorry

end penelope_saving_days_l258_258752


namespace find_y_l258_258448

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

end find_y_l258_258448


namespace binomial_7_4_equals_35_l258_258540

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l258_258540


namespace equation_of_circle_l258_258566

def center : ℝ × ℝ := (3, -2)
def radius : ℝ := 5

theorem equation_of_circle (x y : ℝ) :
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 :=
by
  simp [center, radius]
  sorry

end equation_of_circle_l258_258566


namespace trigonometric_expression_eval_l258_258972

theorem trigonometric_expression_eval :
  2 * (Real.cos (5 * Real.pi / 16))^6 +
  2 * (Real.sin (11 * Real.pi / 16))^6 +
  (3 * Real.sqrt 2 / 8) = 5 / 4 :=
by
  sorry

end trigonometric_expression_eval_l258_258972


namespace balance_the_scale_l258_258318

theorem balance_the_scale (w1 : ℝ) (w2 : ℝ) (book_weight : ℝ) (h1 : w1 = 0.5) (h2 : w2 = 0.3) :
  book_weight = w1 + 2 * w2 :=
by
  sorry

end balance_the_scale_l258_258318


namespace find_a_for_even_function_l258_258697

open Function

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- State the problem in Lean.
theorem find_a_for_even_function :
  (∀ x, f a x = f a (-x)) → a = 1 :=
sorry

end find_a_for_even_function_l258_258697


namespace strictly_monotone_function_l258_258612

open Function

-- Define the problem
theorem strictly_monotone_function (f : ℝ → ℝ) (F : ℝ → ℝ → ℝ)
  (hf_cont : Continuous f) (hf_nonconst : ¬ (∃ c, ∀ x, f x = c))
  (hf_eq : ∀ x y : ℝ, f (x + y) = F (f x) (f y)) :
  StrictMono f :=
sorry

end strictly_monotone_function_l258_258612


namespace james_selling_price_l258_258454

variable (P : ℝ)  -- Selling price per candy bar

theorem james_selling_price 
  (boxes_sold : ℕ)
  (candy_bars_per_box : ℕ) 
  (cost_price_per_candy_bar : ℝ)
  (total_profit : ℝ)
  (H1 : candy_bars_per_box = 10)
  (H2 : boxes_sold = 5)
  (H3 : cost_price_per_candy_bar = 1)
  (H4 : total_profit = 25)
  (profit_eq : boxes_sold * candy_bars_per_box * (P - cost_price_per_candy_bar) = total_profit)
  : P = 1.5 :=
by 
  sorry

end james_selling_price_l258_258454


namespace common_ratio_of_increasing_geometric_sequence_l258_258151

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_inc : ∀ n, a n < a (n + 1))
  (h_a2 : a 2 = 2)
  (h_a4_a3 : a 4 - a 3 = 4) : 
  q = 2 :=
by
  -- sorry - placeholder for proof
  sorry

end common_ratio_of_increasing_geometric_sequence_l258_258151


namespace find_x_eq_neg15_l258_258261

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l258_258261


namespace zero_in_P_two_not_in_P_l258_258524

variables (P : Set Int)

-- Conditions
def condition_1 := ∃ x ∈ P, x > 0 ∧ ∃ y ∈ P, y < 0
def condition_2 := ∃ x ∈ P, x % 2 = 0 ∧ ∃ y ∈ P, y % 2 ≠ 0 
def condition_3 := 1 ∉ P
def condition_4 := ∀ x y, x ∈ P → y ∈ P → x + y ∈ P

-- Proving 0 ∈ P
theorem zero_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 0 ∈ P := 
sorry

-- Proving 2 ∉ P
theorem two_not_in_P (h1 : condition_1 P) (h2 : condition_2 P) (h3 : condition_3 P) (h4 : condition_4 P) : 2 ∉ P := 
sorry

end zero_in_P_two_not_in_P_l258_258524


namespace complement_of_log_set_l258_258414

-- Define the set A based on the logarithmic inequality condition
def A : Set ℝ := { x : ℝ | Real.log x / Real.log (1 / 2) ≥ 2 }

-- Define the complement of A in the real numbers
noncomputable def complement_A : Set ℝ := { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 }

-- The goal is to prove the equivalence
theorem complement_of_log_set :
  complement_A = { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 } :=
by
  sorry

end complement_of_log_set_l258_258414


namespace prime_iff_sum_four_distinct_products_l258_258003

variable (n : ℕ) (a b c d : ℕ)

theorem prime_iff_sum_four_distinct_products (h : n ≥ 5) :
  (Prime n ↔ ∀ (a b c d : ℕ), n = a + b + c + d → a > 0 → b > 0 → c > 0 → d > 0 → ab ≠ cd) :=
sorry

end prime_iff_sum_four_distinct_products_l258_258003


namespace least_z_minus_x_l258_258221

theorem least_z_minus_x (x y z : ℤ) (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) : z - x = 7 :=
sorry

end least_z_minus_x_l258_258221


namespace michelle_travel_distance_l258_258958

-- Define the conditions
def initial_fee : ℝ := 2
def charge_per_mile : ℝ := 2.5
def total_paid : ℝ := 12

-- Define the theorem to prove the distance Michelle traveled
theorem michelle_travel_distance : (total_paid - initial_fee) / charge_per_mile = 4 := by
  sorry

end michelle_travel_distance_l258_258958


namespace james_writing_time_l258_258456

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end james_writing_time_l258_258456


namespace diana_shops_for_newborns_l258_258252

theorem diana_shops_for_newborns (total_children : ℕ) (num_toddlers : ℕ) (teenager_ratio : ℕ) (num_teens : ℕ) (num_newborns : ℕ)
    (h1 : total_children = 40) (h2 : num_toddlers = 6) (h3 : teenager_ratio = 5) (h4 : num_teens = teenager_ratio * num_toddlers) 
    (h5 : num_newborns = total_children - num_teens - num_toddlers) : 
    num_newborns = 4 := sorry

end diana_shops_for_newborns_l258_258252


namespace max_value_of_quadratic_l258_258585

theorem max_value_of_quadratic (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ y, y = x * (1 - 2 * x) ∧ y ≤ 1 / 8 ∧ (y = 1 / 8 ↔ x = 1 / 4) :=
by sorry

end max_value_of_quadratic_l258_258585


namespace bottle_ratio_l258_258467

theorem bottle_ratio (C1 C2 : ℝ)  
  (h1 : (C1 / 2) + (C2 / 4) = (C1 + C2) / 3) :
  C2 = 2 * C1 :=
sorry

end bottle_ratio_l258_258467


namespace find_n_l258_258274

theorem find_n (n : ℝ) (h1 : ∀ x y : ℝ, (n + 1) * x^(n^2 - 5) = y) 
               (h2 : ∀ x > 0, (n + 1) * x^(n^2 - 5) > 0) :
               n = 2 :=
by
  sorry

end find_n_l258_258274


namespace evaluate_fraction_l258_258546

theorem evaluate_fraction : 
  ( (20 - 19) + (18 - 17) + (16 - 15) + (14 - 13) + (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1) ) 
  / 
  ( (1 - 2) + (3 - 4) + (5 - 6) + (7 - 8) + (9 - 10) + (11 - 12) + 13 ) 
  = (10 / 7) := 
by
  -- proof skipped
  sorry

end evaluate_fraction_l258_258546


namespace no_sol_x_y_pos_int_eq_2015_l258_258683

theorem no_sol_x_y_pos_int_eq_2015 (x y : ℕ) (hx : x > 0) (hy : y > 0) : ¬ (x^2 - y! = 2015) :=
sorry

end no_sol_x_y_pos_int_eq_2015_l258_258683


namespace angle_in_triangle_PQR_l258_258179

theorem angle_in_triangle_PQR
  (Q P R : ℝ)
  (h1 : P = 2 * Q)
  (h2 : R = 5 * Q)
  (h3 : Q + P + R = 180) : 
  P = 45 := 
by sorry

end angle_in_triangle_PQR_l258_258179


namespace line_through_center_and_perpendicular_l258_258257

def center_of_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 1

def perpendicular_to_line (slope : ℝ) : Prop :=
  slope = 1

theorem line_through_center_and_perpendicular (x y : ℝ) :
  center_of_circle x y →
  perpendicular_to_line 1 →
  (x - y + 1 = 0) :=
by
  intros h_center h_perpendicular
  sorry

end line_through_center_and_perpendicular_l258_258257


namespace negation_of_p_negation_of_q_l258_258458

def p (x : ℝ) : Prop := x > 0 → x^2 - 5 * x ≥ -25 / 4

def even (n : ℕ) : Prop := ∃ k, n = 2 * k

def q : Prop := ∃ n, even n ∧ ∃ m, n = 3 * m

theorem negation_of_p : ¬(∀ x : ℝ, x > 0 → x^2 - 5 * x ≥ - 25 / 4) → ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x < - 25 / 4 := 
by sorry

theorem negation_of_q : ¬ (∃ n : ℕ, even n ∧ ∃ m : ℕ, n = 3 * m) → ∀ n : ℕ, even n → ¬ (∃ m : ℕ, n = 3 * m) := 
by sorry

end negation_of_p_negation_of_q_l258_258458


namespace problem_solution_l258_258688

theorem problem_solution (m n : ℕ) (h1 : m + 7 < n + 3) 
  (h2 : (m + (m+3) + (m+7) + (n+3) + (n+6) + 2 * n) / 6 = n + 3) 
  (h3 : (m + 7 + n + 3) / 2 = n + 3) : m + n = 12 := 
  sorry

end problem_solution_l258_258688


namespace expression_C_eq_seventeen_l258_258354

theorem expression_C_eq_seventeen : (3 + 4 * 5 - 6) = 17 := 
by 
  sorry

end expression_C_eq_seventeen_l258_258354


namespace smallest_prime_with_digit_sum_23_l258_258087

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258087


namespace smallest_prime_digit_sum_23_l258_258077

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l258_258077


namespace living_room_area_l258_258804

theorem living_room_area (L W : ℝ) (percent_covered : ℝ) (expected_area : ℝ) 
  (hL : L = 6.5) (hW : W = 12) (hpercent : percent_covered = 0.85) 
  (hexpected_area : expected_area = 91.76) : 
  (L * W / percent_covered = expected_area) :=
by
  sorry  -- The proof is omitted.

end living_room_area_l258_258804


namespace cost_per_kg_mixture_l258_258117

variables (C1 C2 R Cm : ℝ)

-- Statement of the proof problem
theorem cost_per_kg_mixture :
  C1 = 6 → C2 = 8.75 → R = 5 / 6 → Cm = C1 * R + C2 * (1 - R) → Cm = 6.458333333333333 :=
by intros hC1 hC2 hR hCm; sorry

end cost_per_kg_mixture_l258_258117


namespace height_to_width_ratio_l258_258493

theorem height_to_width_ratio (w h l : ℝ) (V : ℝ) (x : ℝ) :
  (h = x * w) →
  (l = 7 * h) →
  (V = l * w * h) →
  (V = 129024) →
  (w = 8) →
  (x = 6) :=
by
  intros h_eq_xw l_eq_7h V_eq_lwh V_val w_val
  -- Proof omitted
  sorry

end height_to_width_ratio_l258_258493


namespace cone_sector_central_angle_l258_258147

noncomputable def base_radius := 1
noncomputable def slant_height := 2
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) := circumference r
noncomputable def central_angle (l : ℝ) (s : ℝ) := l / s

theorem cone_sector_central_angle : central_angle (arc_length base_radius) slant_height = Real.pi := 
by 
  -- Here we acknowledge that the proof would go, but it is left out as per instructions.
  sorry

end cone_sector_central_angle_l258_258147


namespace population_after_panic_l258_258667

noncomputable def original_population : ℕ := 7200
def first_event_loss (population : ℕ) : ℕ := population * 10 / 100
def after_first_event (population : ℕ) : ℕ := population - first_event_loss population
def second_event_loss (population : ℕ) : ℕ := population * 25 / 100
def after_second_event (population : ℕ) : ℕ := population - second_event_loss population

theorem population_after_panic : after_second_event (after_first_event original_population) = 4860 := sorry

end population_after_panic_l258_258667


namespace math_problem_l258_258313

theorem math_problem
  (n : ℕ) (d : ℕ)
  (h1 : d ≤ 9)
  (h2 : 3 * n^2 + 2 * n + d = 263)
  (h3 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) :
  n + d = 11 := 
sorry

end math_problem_l258_258313


namespace percentage_increase_l258_258718

theorem percentage_increase (x : ℝ) (y : ℝ) (h1 : x = 114.4) (h2 : y = 88) : 
  ((x - y) / y) * 100 = 30 := 
by 
  sorry

end percentage_increase_l258_258718


namespace question_1_question_2_l258_258148

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem question_1 :
  f 1 * f 2 * f 3 = 36 * 108 * 360 := by
  sorry

theorem question_2 :
  ∃ m ≥ 2, ∀ n : ℕ, n > 0 → f n % m = 0 ∧ m = 36 := by
  sorry

end question_1_question_2_l258_258148


namespace number_div_0_04_eq_100_9_l258_258659

theorem number_div_0_04_eq_100_9 :
  ∃ number : ℝ, (number / 0.04 = 100.9) ∧ (number = 4.036) :=
sorry

end number_div_0_04_eq_100_9_l258_258659


namespace stephanie_bills_l258_258627

theorem stephanie_bills :
  let electricity_bill := 120
  let electricity_paid := 0.80 * electricity_bill
  let gas_bill := 80
  let gas_paid := (3 / 4) * gas_bill
  let additional_gas_payment := 10
  let water_bill := 60
  let water_paid := 0.65 * water_bill
  let internet_bill := 50
  let internet_paid := 6 * 5
  let internet_remaining_before_discount := internet_bill - internet_paid
  let internet_discount := 0.10 * internet_remaining_before_discount
  let phone_bill := 45
  let phone_paid := 0.20 * phone_bill
  let remaining_electricity := electricity_bill - electricity_paid
  let remaining_gas := gas_bill - (gas_paid + additional_gas_payment)
  let remaining_water := water_bill - water_paid
  let remaining_internet := internet_remaining_before_discount - internet_discount
  let remaining_phone := phone_bill - phone_paid
  (remaining_electricity + remaining_gas + remaining_water + remaining_internet + remaining_phone) = 109 :=
by
  sorry

end stephanie_bills_l258_258627


namespace binom_7_4_eq_35_l258_258537

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l258_258537


namespace remaining_candy_l258_258531

def initial_candy : ℕ := 36
def ate_candy1 : ℕ := 17
def ate_candy2 : ℕ := 15
def total_ate_candy : ℕ := ate_candy1 + ate_candy2

theorem remaining_candy : initial_candy - total_ate_candy = 4 := by
  sorry

end remaining_candy_l258_258531


namespace probability_red_ball_l258_258599

theorem probability_red_ball (red_balls black_balls : ℕ) (h_red : red_balls = 7) (h_black : black_balls = 3) :
  (red_balls.to_rat / (red_balls + black_balls).to_rat) = 7 / 10 :=
by
  sorry

end probability_red_ball_l258_258599


namespace unique_handshakes_l258_258121

theorem unique_handshakes :
  let twins_sets := 12
  let triplets_sets := 3
  let twins := twins_sets * 2
  let triplets := triplets_sets * 3
  let twin_shakes_twins := twins * (twins - 2)
  let triplet_shakes_triplets := triplets * (triplets - 3)
  let twin_shakes_triplets := twins * (triplets / 3)
  (twin_shakes_twins + triplet_shakes_triplets + twin_shakes_triplets) / 2 = 327 := by
  sorry

end unique_handshakes_l258_258121


namespace sum_of_ages_53_l258_258370

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53_l258_258370


namespace range_of_c_l258_258570

noncomputable def is_monotonically_decreasing (c: ℝ) : Prop := ∀ x1 x2: ℝ, x1 < x2 → c^x2 ≤ c^x1

def inequality_holds (c: ℝ) : Prop := ∀ x: ℝ, x^2 + x + (1/2)*c > 0

theorem range_of_c (c: ℝ) (h1: c > 0) :
  ((is_monotonically_decreasing c ∨ inequality_holds c) ∧ ¬(is_monotonically_decreasing c ∧ inequality_holds c)) 
  → (0 < c ∧ c ≤ 1/2 ∨ c ≥ 1) := 
sorry

end range_of_c_l258_258570


namespace distinctPaintedCubeConfigCount_l258_258361

-- Define a painted cube with given face colors
structure PaintedCube where
  blue_face : ℤ
  yellow_faces : Finset ℤ
  red_faces : Finset ℤ
  -- Ensure logical conditions about faces
  face_count : blue_face ∉ yellow_faces ∧ blue_face ∉ red_faces ∧
               yellow_faces ∩ red_faces = ∅ ∧ yellow_faces.card = 2 ∧
               red_faces.card = 3

-- There are no orientation-invariant rotations that change the configuration
def equivPaintedCube (c1 c2 : PaintedCube) : Prop :=
  ∃ (r: ℤ), 
    -- rotate c1 by r to get c2
    true -- placeholder for rotation logic

-- The set of all possible distinct painted cubes under rotation constraints is defined
def possibleConfigurations : Finset PaintedCube :=
  sorry  -- construct this set considering rotations

-- The main proposition
theorem distinctPaintedCubeConfigCount : (possibleConfigurations.card = 4) :=
  sorry

end distinctPaintedCubeConfigCount_l258_258361


namespace largest_r_l258_258182

theorem largest_r (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p*q + p*r + q*r = 8) : 
  r ≤ 2 + Real.sqrt (20/3) := 
sorry

end largest_r_l258_258182


namespace smallest_prime_with_digit_sum_23_l258_258088

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258088


namespace seungjun_clay_cost_l258_258915

theorem seungjun_clay_cost (price_per_gram : ℝ) (qty1 qty2 : ℝ) 
  (h1 : price_per_gram = 17.25) 
  (h2 : qty1 = 1000) 
  (h3 : qty2 = 10) :
  (qty1 * price_per_gram + qty2 * price_per_gram) = 17422.5 :=
by
  sorry

end seungjun_clay_cost_l258_258915


namespace meeting_time_when_speeds_doubled_l258_258048

noncomputable def meeting_time (x y z : ℝ) : ℝ :=
  2 * 91

theorem meeting_time_when_speeds_doubled
  (x y z : ℝ)
  (h1 : 2 * z * (x + y) = (2 * z - 56) * (2 * x + y))
  (h2 : 2 * z * (x + y) = (2 * z - 65) * (x + 2 * y))
  : meeting_time x y z = 182 := 
sorry

end meeting_time_when_speeds_doubled_l258_258048


namespace hot_drink_sales_l258_258663

theorem hot_drink_sales (x y : ℝ) (h : y = -2.35 * x + 147.7) (hx : x = 2) : y = 143 := 
by sorry

end hot_drink_sales_l258_258663


namespace diana_shopping_for_newborns_l258_258251

-- Define the number of children, toddlers, and the toddler-to-teenager ratio
def total_children : ℕ := 40
def toddlers : ℕ := 6
def toddler_to_teenager_ratio : ℕ := 5

-- The result we need to prove
def number_of_newborns : ℕ := 
  total_children - (toddlers + toddler_to_teenager_ratio * toddlers) = 4

-- Define the proof statement
theorem diana_shopping_for_newborns : 
  total_children = 40 ∧ toddlers = 6 ∧ toddler_to_teenager_ratio = 5 → 
  number_of_newborns := 4 :=
by sorry

end diana_shopping_for_newborns_l258_258251


namespace find_d_l258_258410

-- Definitions of the conditions
variables (r s t u d : ℤ)

-- Assume r, s, t, and u are positive integers
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom t_pos : t > 0
axiom u_pos : u > 0

-- Given conditions
axiom h1 : r ^ 5 = s ^ 4
axiom h2 : t ^ 3 = u ^ 2
axiom h3 : t - r = 19
axiom h4 : d = u - s

-- Proof statement
theorem find_d : d = 757 :=
by sorry

end find_d_l258_258410


namespace investment_principal_l258_258345

theorem investment_principal (A r : ℝ) (n t : ℕ) (P : ℝ) : 
  r = 0.07 → n = 4 → t = 5 → A = 60000 → 
  A = P * (1 + r / n)^(n * t) →
  P = 42409 :=
by
  sorry

end investment_principal_l258_258345


namespace value_of_f_at_2_l258_258932

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_of_f_at_2 : f 2 = 3 := sorry

end value_of_f_at_2_l258_258932


namespace contradiction_proof_l258_258789

theorem contradiction_proof (a b : ℕ) (h : a + b ≥ 3) : (a ≥ 2) ∨ (b ≥ 2) :=
sorry

end contradiction_proof_l258_258789


namespace find_integer_x_l258_258844

theorem find_integer_x : ∃ x : ℤ, x^5 - 3 * x^2 = 216 ∧ x = 3 :=
by {
  sorry
}

end find_integer_x_l258_258844


namespace sufficient_but_not_necessary_condition_l258_258857

-- Define a sequence of positive terms
def is_positive_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∀ i, 0 < seq i

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∃ q > 0, q ≠ 1 ∧ ∀ i j, i < j → seq j = (q ^ (j - i : ℤ)) * seq i

-- State the theorem
theorem sufficient_but_not_necessary_condition (seq : Fin 8 → ℝ) (h_pos : is_positive_sequence seq) :
  ¬is_geometric_sequence seq → seq 0 + seq 7 < seq 3 + seq 4 ∧ 
  (seq 0 + seq 7 < seq 3 + seq 4 → ¬is_geometric_sequence seq) ∧
  (¬is_geometric_sequence seq → ¬(seq 0 + seq 7 < seq 3 + seq 4) -> ¬ is_geometric_sequence seq) :=
sorry

end sufficient_but_not_necessary_condition_l258_258857


namespace negative_root_no_positive_l258_258700

theorem negative_root_no_positive (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = ax + 1) ∧ (¬ ∃ x : ℝ, x > 0 ∧ |x| = ax + 1) → a > -1 :=
by
  sorry

end negative_root_no_positive_l258_258700


namespace variables_and_unknowns_l258_258176

theorem variables_and_unknowns (f_1 f_2: ℝ → ℝ → ℝ) (f: ℝ → ℝ → ℝ) :
  (∀ x y, f_1 x y = 0 ∧ f_2 x y = 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (∀ x y, f x y = 0 → (∃ a b, x = a ∧ y = b)) :=
by sorry

end variables_and_unknowns_l258_258176


namespace square_perimeter_l258_258765

theorem square_perimeter (a : ℝ) (side : ℝ) (perimeter : ℝ) (h1 : a = 144) (h2 : side = Real.sqrt a) (h3 : perimeter = 4 * side) : perimeter = 48 := by
  sorry

end square_perimeter_l258_258765


namespace integral_equals_result_l258_258681

noncomputable def integral_value : ℝ :=
  ∫ x in 1.0..2.0, (x^2 + 1) / x

theorem integral_equals_result :
  integral_value = (3 / 2) + Real.log 2 := 
by
  sorry

end integral_equals_result_l258_258681


namespace infinite_solutions_eq_one_l258_258564

theorem infinite_solutions_eq_one (a : ℝ) :
  (∃ᶠ x in filter.at_top, abs (x - 2) = a * x - 2) →
  a = 1 :=
by
  sorry

end infinite_solutions_eq_one_l258_258564


namespace trigonometric_relationship_l258_258141

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < π)
variable (h : Real.tan α = Real.cos β / (1 - Real.sin β))

theorem trigonometric_relationship : 
    2 * α - β = π / 2 :=
sorry

end trigonometric_relationship_l258_258141


namespace car_mpg_l258_258230

open Nat

theorem car_mpg (x : ℕ) (h1 : ∀ (m : ℕ), m = 4 * (3 * x) -> x = 27) 
                (h2 : ∀ (d1 d2 : ℕ), d2 = (4 * d1) / 3 - d1 -> d2 = 126) 
                (h3 : ∀ g : ℕ, g = 14)
                : x = 27 := 
by
  sorry

end car_mpg_l258_258230


namespace ratio_a_b_equals_sqrt2_l258_258170

variable (A B C a b c : ℝ) -- Define the variables representing the angles and sides.

-- Assuming the sides a, b, c are positive and a triangle is formed (non-degenerate)
axiom triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c

-- Assuming the sum of the angles in a triangle equals 180 degrees (π radians)
axiom sum_angles_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : b * Real.cos C + c * Real.cos B = Real.sqrt 2 * b

-- Problem statement to be proven
theorem ratio_a_b_equals_sqrt2 : (a / b) = Real.sqrt 2 :=
by
  -- Assume the problem statement is correct
  sorry

end ratio_a_b_equals_sqrt2_l258_258170


namespace actors_duration_l258_258723

-- Definition of conditions
def actors_at_a_time := 5
def total_actors := 20
def total_minutes := 60

-- Main statement to prove
theorem actors_duration : total_minutes / (total_actors / actors_at_a_time) = 15 := 
by
  sorry

end actors_duration_l258_258723


namespace f_recurrence_l258_258011

noncomputable def f (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem f_recurrence (n : ℕ) : f (n + 1) - f (n - 1) = (3 * Real.sqrt 7 / 14) * f n := 
  sorry

end f_recurrence_l258_258011


namespace find_hourly_wage_l258_258817

noncomputable def hourly_wage_inexperienced (x : ℝ) : Prop :=
  let sailors_total := 17
  let inexperienced_sailors := 5
  let experienced_sailors := sailors_total - inexperienced_sailors
  let wage_experienced := (6 / 5) * x
  let total_hours_month := 240
  let total_monthly_earnings_experienced := 34560
  (experienced_sailors * wage_experienced * total_hours_month) = total_monthly_earnings_experienced

theorem find_hourly_wage (x : ℝ) : hourly_wage_inexperienced x → x = 10 :=
by
  sorry

end find_hourly_wage_l258_258817


namespace fill_time_first_and_fourth_taps_l258_258499

noncomputable def pool_filling_time (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) : ℝ :=
  m / (x + u)

theorem fill_time_first_and_fourth_taps (m x y z u : ℝ) (h₁ : 2 * (x + y) = m) (h₂ : 3 * (y + z) = m) (h₃ : 4 * (z + u) = m) :
  pool_filling_time m x y z u h₁ h₂ h₃ = 12 / 5 :=
sorry

end fill_time_first_and_fourth_taps_l258_258499


namespace equivalent_proof_problem_l258_258163

noncomputable def perimeter_inner_polygon (pentagon_perimeter : ℕ) : ℕ :=
  let side_length := pentagon_perimeter / 5
  let inner_polygon_sides := 10
  inner_polygon_sides * side_length

theorem equivalent_proof_problem :
  perimeter_inner_polygon 65 = 130 :=
by
  sorry

end equivalent_proof_problem_l258_258163


namespace min_value_xy_l258_258409

theorem min_value_xy (x y : ℝ) (h : x * y = 1) : x^2 + 4 * y^2 ≥ 4 := by
  sorry

end min_value_xy_l258_258409


namespace largest_circle_at_A_l258_258771

/--
Given a pentagon with side lengths AB = 16 cm, BC = 14 cm, CD = 17 cm, DE = 13 cm, and EA = 14 cm,
and given five circles with centers A, B, C, D, and E such that each pair of circles with centers at
the ends of a side of the pentagon touch on that side, the circle with center A
has the largest radius.
-/
theorem largest_circle_at_A
  (rA rB rC rD rE : ℝ) 
  (hAB : rA + rB = 16)
  (hBC : rB + rC = 14)
  (hCD : rC + rD = 17)
  (hDE : rD + rE = 13)
  (hEA : rE + rA = 14) :
  rA ≥ rB ∧ rA ≥ rC ∧ rA ≥ rD ∧ rA ≥ rE := 
sorry

end largest_circle_at_A_l258_258771


namespace min_abs_phi_l258_258646

open Real

theorem min_abs_phi {k : ℤ} :
  ∃ (φ : ℝ), ∀ (k : ℤ), φ = - (5 * π) / 6 + k * π ∧ |φ| = π / 6 := sorry

end min_abs_phi_l258_258646


namespace total_candy_bars_correct_l258_258012

-- Define the number of each type of candy bar.
def snickers : Nat := 3
def marsBars : Nat := 2
def butterfingers : Nat := 7

-- Define the total number of candy bars.
def totalCandyBars : Nat := snickers + marsBars + butterfingers

-- Formulate the theorem about the total number of candy bars.
theorem total_candy_bars_correct : totalCandyBars = 12 :=
sorry

end total_candy_bars_correct_l258_258012


namespace sqrt3_f_pi6_lt_f_pi3_l258_258831

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_derivative_tan_lt (x : ℝ) (h : 0 < x ∧ x < π / 2) : f x < (deriv f x) * tan x

theorem sqrt3_f_pi6_lt_f_pi3 :
  sqrt 3 * f (π / 6) < f (π / 3) :=
by
  sorry

end sqrt3_f_pi6_lt_f_pi3_l258_258831


namespace total_shoes_l258_258998

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l258_258998


namespace circle_diameter_l258_258808

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l258_258808


namespace local_min_f_at_2_implies_a_eq_2_l258_258875

theorem local_min_f_at_2_implies_a_eq_2 (a : ℝ) : 
  (∃ f : ℝ → ℝ, 
     (∀ x : ℝ, f x = x * (x - a)^2) ∧ 
     (∀ f' : ℝ → ℝ, 
       (∀ x : ℝ, f' x = 3 * x^2 - 4 * a * x + a^2) ∧ 
       f' 2 = 0 ∧ 
       (∀ f'' : ℝ → ℝ, 
         (∀ x : ℝ, f'' x = 6 * x - 4 * a) ∧ 
         f'' 2 > 0
       )
     )
  ) → a = 2 :=
sorry

end local_min_f_at_2_implies_a_eq_2_l258_258875


namespace fraction_ordering_l258_258650

theorem fraction_ordering :
  (8 : ℚ) / 31 < (11 : ℚ) / 33 ∧
  (11 : ℚ) / 33 < (12 : ℚ) / 29 ∧
  (8 : ℚ) / 31 < (12 : ℚ) / 29 := 
by  
  sorry

end fraction_ordering_l258_258650


namespace compute_expression_l258_258675

theorem compute_expression : 20 * (150 / 3 + 36 / 4 + 4 / 25 + 2) = 1223 + 1/5 :=
by
  sorry

end compute_expression_l258_258675


namespace remainder_2015_div_28_l258_258928

theorem remainder_2015_div_28 : 2015 % 28 = 17 :=
by
  sorry

end remainder_2015_div_28_l258_258928


namespace abs_neg_2023_l258_258481

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l258_258481


namespace total_shoes_l258_258997

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l258_258997


namespace problem1_problem2_l258_258155

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then (1/2)^x - 2 else (x - 2) * (|x| - 1)

theorem problem1 : f (f (-2)) = 0 := by 
  sorry

theorem problem2 (x : ℝ) (h : f x ≥ 2) : x ≥ 3 ∨ x = 0 := by
  sorry

end problem1_problem2_l258_258155


namespace greatest_k_inequality_l258_258556

theorem greatest_k_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    ( ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → 
    (a / b + b / c + c / a - 3) ≥ k * (a / (b + c) + b / (c + a) + c / (a + b) - 3 / 2) ) ↔ k = 1 := 
sorry

end greatest_k_inequality_l258_258556


namespace weight_of_replaced_person_l258_258202

theorem weight_of_replaced_person
  (avg_increase : ∀ W : ℝ, W + 8 * 2.5 = W - X + 80)
  (new_person_weight : 80 = 80):
  X = 60 := by
  sorry

end weight_of_replaced_person_l258_258202


namespace arithmetic_sequence_a3a6_l258_258006

theorem arithmetic_sequence_a3a6 (a : ℕ → ℤ)
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_inc : ∀ n, a n < a (n + 1))
  (h_eq : a 3 * a 4 = 45): 
  a 2 * a 5 = 13 := 
sorry

end arithmetic_sequence_a3a6_l258_258006


namespace anna_money_left_eur_l258_258244

noncomputable def total_cost_usd : ℝ := 4 * 1.50 + 7 * 2.25 + 3 * 0.75 + 3.00 * 0.80
def sales_tax_rate : ℝ := 0.075
def exchange_rate : ℝ := 0.85
def initial_amount_usd : ℝ := 50

noncomputable def total_cost_with_tax_usd : ℝ := total_cost_usd * (1 + sales_tax_rate)
noncomputable def total_cost_eur : ℝ := total_cost_with_tax_usd * exchange_rate
noncomputable def initial_amount_eur : ℝ := initial_amount_usd * exchange_rate

noncomputable def money_left_eur : ℝ := initial_amount_eur - total_cost_eur

theorem anna_money_left_eur : abs (money_left_eur - 18.38) < 0.01 := by
  -- Add proof steps here
  sorry

end anna_money_left_eur_l258_258244


namespace anika_age_l258_258668

/-- Given:
 1. Anika is 10 years younger than Clara.
 2. Clara is 5 years older than Ben.
 3. Ben is 20 years old.
 Prove:
 Anika's age is 15 years.
 -/
theorem anika_age (Clara Anika Ben : ℕ) 
  (h1 : Anika = Clara - 10) 
  (h2 : Clara = Ben + 5) 
  (h3 : Ben = 20) : Anika = 15 := 
by
  sorry

end anika_age_l258_258668


namespace Fran_speed_l258_258605

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l258_258605


namespace lives_lost_l258_258797

-- Conditions given in the problem
def initial_lives : ℕ := 83
def current_lives : ℕ := 70

-- Prove the number of lives lost
theorem lives_lost : initial_lives - current_lives = 13 :=
by
  sorry

end lives_lost_l258_258797


namespace smallest_prime_with_digit_sum_23_l258_258070

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l258_258070


namespace convert_to_rectangular_form_l258_258130

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end convert_to_rectangular_form_l258_258130


namespace ball_distributions_l258_258839

theorem ball_distributions (p q : ℚ) (h1 : p = (Nat.choose 5 1 * Nat.choose 4 1 * Nat.choose 20 2 * Nat.choose 18 6 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20)
                            (h2 : q = (Nat.choose 20 4 * Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4) / Nat.choose 20 20) :
  p / q = 10 :=
by
  sorry

end ball_distributions_l258_258839


namespace total_weight_of_rhinos_l258_258624

def white_rhino_weight : ℕ := 5100
def black_rhino_weight : ℕ := 2000

theorem total_weight_of_rhinos :
  7 * white_rhino_weight + 8 * black_rhino_weight = 51700 :=
by
  sorry

end total_weight_of_rhinos_l258_258624


namespace triangle_inequality_range_l258_258852

theorem triangle_inequality_range {a b c : ℝ} (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  1 ≤ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) ∧ (a^2 + b^2 + c^2) / (a * b + b * c + c * a) < 2 := 
by 
  sorry

end triangle_inequality_range_l258_258852


namespace range_of_m_l258_258965

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + Real.log x + 3)) :
  ∃ m, m ∈ Set.Icc (1 / (2 * Real.exp 1)) (1 + Real.log 3 / 6) :=
sorry

end range_of_m_l258_258965


namespace find_a_l258_258984

noncomputable def tangent_to_circle_and_parallel (a : ℝ) : Prop := 
  let P := (2, 2)
  let circle_center := (1, 0)
  let on_circle := (P.1 - 1)^2 + P.2^2 = 5
  let perpendicular_slope := (P.2 - circle_center.2) / (P.1 - circle_center.1) * (1 / a) = -1
  on_circle ∧ perpendicular_slope

theorem find_a (a : ℝ) : tangent_to_circle_and_parallel a ↔ a = -2 :=
by
  sorry

end find_a_l258_258984


namespace cos_C_eq_3_5_l258_258417

theorem cos_C_eq_3_5 (A B C : ℝ) (hABC : A^2 + B^2 = C^2) (hRight : B ^ 2 + C ^ 2 = A ^ 2) (hTan : B / C = 4 / 3) : B / A = 3 / 5 :=
by
  sorry

end cos_C_eq_3_5_l258_258417


namespace max_tickets_l258_258026

/-- Given the cost of each ticket and the total amount of money available, 
    prove that the maximum number of tickets that can be purchased is 8. -/
theorem max_tickets (ticket_cost : ℝ) (total_amount : ℝ) (h1 : ticket_cost = 18.75) (h2 : total_amount = 150) :
  (∃ n : ℕ, ticket_cost * n ≤ total_amount ∧ ∀ m : ℕ, ticket_cost * m ≤ total_amount → m ≤ n) ∧
  ∃ n : ℤ, (n : ℤ) = 8 :=
by
  sorry

end max_tickets_l258_258026


namespace initial_catfish_count_l258_258190

theorem initial_catfish_count (goldfish : ℕ) (remaining_fish : ℕ) (disappeared_fish : ℕ) (catfish : ℕ) :
  goldfish = 7 → 
  remaining_fish = 15 → 
  disappeared_fish = 4 → 
  catfish + goldfish = 19 →
  catfish = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_catfish_count_l258_258190


namespace delta_discount_percentage_l258_258128

theorem delta_discount_percentage (original_delta : ℝ) (original_united : ℝ)
  (united_discount_percent : ℝ) (savings : ℝ) (delta_discounted : ℝ) : 
  original_delta - delta_discounted = 0.2 * original_delta := by
  -- Given conditions
  let discounted_united := original_united * (1 - united_discount_percent / 100)
  have : delta_discounted = discounted_united - savings := sorry
  let delta_discount_amount := original_delta - delta_discounted
  have : delta_discount_amount = 0.2 * original_delta := sorry
  exact this

end delta_discount_percentage_l258_258128


namespace new_person_weight_l258_258940

theorem new_person_weight (avg_increase : ℝ) (num_people : ℕ) (weight_replaced : ℝ) (new_weight : ℝ) : 
    num_people = 8 → avg_increase = 1.5 → weight_replaced = 65 → 
    new_weight = weight_replaced + num_people * avg_increase → 
    new_weight = 77 :=
by
  intros h1 h2 h3 h4
  sorry

end new_person_weight_l258_258940


namespace solve_for_a_l258_258980

theorem solve_for_a {f : ℝ → ℝ} (h1 : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h2 : f a = 7) : a = 7 :=
sorry

end solve_for_a_l258_258980


namespace sqrt_eq_9_implies_n_eq_73_l258_258408

theorem sqrt_eq_9_implies_n_eq_73 (n : ℕ) : sqrt (8 + n) = 9 → n = 73 := by
  sorry

end sqrt_eq_9_implies_n_eq_73_l258_258408


namespace sin_theta_l258_258388

def f (x : ℝ) : ℝ := 3 * Real.sin x - 8 * (Real.cos (x / 2))^2

theorem sin_theta:
  (∀ x, f x ≤ f θ) → Real.sin θ = 3 / 5 :=
by
  sorry

end sin_theta_l258_258388


namespace find_side_length_of_left_square_l258_258773

theorem find_side_length_of_left_square (x : ℕ) 
  (h1 : x + (x + 17) + (x + 11) = 52) : 
  x = 8 :=
by
  -- The proof will go here
  sorry

end find_side_length_of_left_square_l258_258773


namespace solve_for_x_l258_258263

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l258_258263


namespace total_number_of_tiles_l258_258661

theorem total_number_of_tiles {s : ℕ} 
  (h1 : ∃ s : ℕ, (s^2 - 4*s + 896 = 0))
  (h2 : 225 = 2*s - 1 + s^2 / 4 - s / 2) :
  s^2 = 1024 := by
  sorry

end total_number_of_tiles_l258_258661


namespace quadratic_distinct_real_roots_l258_258581

theorem quadratic_distinct_real_roots (a : ℝ) (h : a ≠ 1) : 
(a < 2) → 
(∃ x y : ℝ, x ≠ y ∧ (a-1)*x^2 - 2*x + 1 = 0 ∧ (a-1)*y^2 - 2*y + 1 = 0) :=
sorry

end quadratic_distinct_real_roots_l258_258581


namespace total_people_transport_l258_258838

-- Define the conditions
def boatA_trips_day1 := 7
def boatB_trips_day1 := 5
def boatA_capacity := 20
def boatB_capacity := 15
def boatA_trips_day2 := 5
def boatB_trips_day2 := 6

-- Define the theorem statement
theorem total_people_transport :
  (boatA_trips_day1 * boatA_capacity + boatB_trips_day1 * boatB_capacity) +
  (boatA_trips_day2 * boatA_capacity + boatB_trips_day2 * boatB_capacity)
  = 405 := 
  by
  sorry

end total_people_transport_l258_258838


namespace pictures_vertically_l258_258741

def total_pictures := 30
def haphazard_pictures := 5
def horizontal_pictures := total_pictures / 2

theorem pictures_vertically : total_pictures - (horizontal_pictures + haphazard_pictures) = 10 := by
  sorry

end pictures_vertically_l258_258741


namespace prime_sum_10003_l258_258423

def is_prime (n : ℕ) : Prop := sorry -- Assume we have internal support for prime checking

def count_prime_sums (n : ℕ) : ℕ :=
  if is_prime (n - 2) then 1 else 0

theorem prime_sum_10003 :
  count_prime_sums 10003 = 1 :=
by
  sorry

end prime_sum_10003_l258_258423


namespace number_of_cannoneers_l258_258508

-- Define the variables for cannoneers, women, and men respectively
variables (C W M : ℕ)

-- Define the conditions as assumptions
def conditions : Prop :=
  W = 2 * C ∧
  M = 2 * W ∧
  M + W = 378

-- Prove that the number of cannoneers is 63
theorem number_of_cannoneers (h : conditions C W M) : C = 63 :=
by sorry

end number_of_cannoneers_l258_258508


namespace additional_stars_needed_l258_258942

-- Defining the number of stars required per bottle
def stars_per_bottle : Nat := 85

-- Defining the number of bottles Luke needs to fill
def bottles_to_fill : Nat := 4

-- Defining the number of stars Luke has already made
def stars_made : Nat := 33

-- Calculating the number of stars Luke still needs to make
theorem additional_stars_needed : (stars_per_bottle * bottles_to_fill - stars_made) = 307 := by
  sorry  -- Proof to be provided

end additional_stars_needed_l258_258942


namespace number_of_middle_managers_selected_l258_258232

-- Definitions based on conditions
def total_employees := 1000
def senior_managers := 50
def middle_managers := 150
def general_staff := 800
def survey_size := 200

-- Proposition to state the question and correct answer formally
theorem number_of_middle_managers_selected:
  200 * (150 / 1000) = 30 :=
by
  sorry

end number_of_middle_managers_selected_l258_258232


namespace find_a_l258_258862

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l258_258862


namespace fran_speed_l258_258608

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l258_258608


namespace tangent_line_at_point_l258_258034

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 4 * x + 2

def point : ℝ × ℝ := (1, -3)

def tangent_line (x y : ℝ) : Prop := 5 * x + y - 2 = 0

theorem tangent_line_at_point : tangent_line 1 (-3) :=
  sorry

end tangent_line_at_point_l258_258034


namespace geometric_series_sum_l258_258827

-- Define the terms of the series
def a : ℚ := 1 / 5
def r : ℚ := -1 / 3
def n : ℕ := 6

-- Define the expected sum
def expected_sum : ℚ := 182 / 1215

-- Prove that the sum of the geometric series equals the expected sum
theorem geometric_series_sum : 
  (a * (1 - r^n)) / (1 - r) = expected_sum := 
by
  sorry

end geometric_series_sum_l258_258827


namespace g_18_66_l258_258910

def g (x y : ℕ) : ℕ := sorry

axiom g_prop1 : ∀ x, g x x = x
axiom g_prop2 : ∀ x y, g x y = g y x
axiom g_prop3 : ∀ x y, (x + 2 * y) * g x y = y * g x (x + 2 * y)

theorem g_18_66 : g 18 66 = 198 :=
by
  sorry

end g_18_66_l258_258910


namespace tub_emptying_time_l258_258413

variables (x C D T : ℝ) (hx : x > 0) (hC : C > 0) (hD : D > 0)

theorem tub_emptying_time (h1 : 4 * (D - x) = (5 / 7) * C) :
  T = 8 / (5 + (28 * x) / C) :=
by sorry

end tub_emptying_time_l258_258413


namespace sum_abc_geq_half_l258_258704

theorem sum_abc_geq_half (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
(h_abs_sum : |a - b| + |b - c| + |c - a| = 1) : 
a + b + c ≥ 0.5 := 
sorry

end sum_abc_geq_half_l258_258704


namespace probability_le_neg2_from_normal_distribution_and_interval_probability_l258_258310

noncomputable def normal_distribution_p_le_neg2 (ξ : ℝ) (δ : ℝ) : Prop :=
∀ (ξ ∈ ℝ), (ξ ≈ μ) → 0.4 → P(ξ <= -2) = 0.1

theorem probability_le_neg2_from_normal_distribution_and_interval_probability
  (ξ : ℝ) (μ : ℝ := 0) (δ : ℝ) (hξ : ξ ≈ μ ⟹ δ ^ 2) 
  (h2 : P(-2 ≤ ξ ⟹ 0) = 0.4) : P(ξ ≤ -2) = 0.1 :=
sorry

end probability_le_neg2_from_normal_distribution_and_interval_probability_l258_258310


namespace certain_number_l258_258519

theorem certain_number (x : ℝ) (h : 0.65 * 40 = (4/5) * x + 6) : x = 25 :=
sorry

end certain_number_l258_258519


namespace not_always_true_inequality_l258_258582

variable {x y z : ℝ} {k : ℤ}

theorem not_always_true_inequality :
  x > 0 → y > 0 → x > y → z ≠ 0 → k ≠ 0 → ¬ ( ∀ z, (x / (z^k) > y / (z^k)) ) :=
by
  intro hx hy hxy hz hk
  sorry

end not_always_true_inequality_l258_258582


namespace simplify_expression_l258_258907

variable (y : ℝ)

theorem simplify_expression :
  4 * y^3 + 8 * y + 6 - (3 - 4 * y^3 - 8 * y) = 8 * y^3 + 16 * y + 3 :=
by
  sorry

end simplify_expression_l258_258907


namespace find_coefficients_l258_258415

theorem find_coefficients (k b : ℝ) :
    (∀ x y : ℝ, (y = k * x) → ((x-2)^2 + y^2 = 1) → (2*x + y + b = 0)) →
    ((k = 1/2) ∧ (b = -4)) :=
by
  sorry

end find_coefficients_l258_258415


namespace b_has_infinite_solutions_l258_258968

noncomputable def b_value_satisfies_infinite_solutions : Prop :=
  ∃ b : ℚ, (∀ x : ℚ, 4 * (3 * x - b) = 3 * (4 * x + 7)) → b = -21 / 4

theorem b_has_infinite_solutions : b_value_satisfies_infinite_solutions :=
  sorry

end b_has_infinite_solutions_l258_258968


namespace mark_pond_depth_l258_258619

def depth_of_Peter_pond := 5

def depth_of_Mark_pond := 3 * depth_of_Peter_pond + 4

theorem mark_pond_depth : depth_of_Mark_pond = 19 := by
  sorry

end mark_pond_depth_l258_258619


namespace calculate_spadesuit_l258_258976

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem calculate_spadesuit : spadesuit 3 (spadesuit 5 6) = -112 := by
  sorry

end calculate_spadesuit_l258_258976


namespace total_team_points_l258_258290

theorem total_team_points :
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  (A + B + C + D + E + F + G + H = 22) :=
by
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  sorry

end total_team_points_l258_258290


namespace total_shoes_l258_258999

variable (a b c d : Nat)

theorem total_shoes (h1 : a = 7) (h2 : b = a + 2) (h3 : c = 0) (h4 : d = 2 * (a + b + c)) :
  a + b + c + d = 48 :=
sorry

end total_shoes_l258_258999


namespace total_apples_l258_258466

def green_apples : ℕ := 2
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

theorem total_apples : green_apples + red_apples + yellow_apples = 19 :=
by
  -- Placeholder for the proof
  sorry

end total_apples_l258_258466


namespace smallest_prime_with_digit_sum_23_l258_258072

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ (n : ℕ), is_prime n ∧ sum_of_digits n = 23 ∧ ∀ m : ℕ, is_prime m ∧ sum_of_digits m = 23 → n ≤ m :=
begin
  existsi 599,
  split,
  { -- Prove that 599 is prime
    unfold is_prime,
    split,
    { exact dec_trivial, }, -- 599 > 1
    { intros m h,
      fin_cases m,
      { sorry, }, -- Here, we will use divisibility tests on primes less than 25
    },
  },
  split,
  { -- Prove that the sum of digits of 599 is 23
    unfold sum_of_digits,
    exact dec_trivial, -- 5 + 9 + 9 = 23
  },
  { -- Prove that 599 is the smallest such number
    intros m h_prime h_sum,
    sorry, -- Here, we will show that any other number with these properties is not smaller
  }
end

end smallest_prime_with_digit_sum_23_l258_258072


namespace solution_set_non_empty_iff_l258_258716

theorem solution_set_non_empty_iff (a : ℝ) : (∃ x : ℝ, |x - 1| + |x + 2| < a) ↔ (a > 3) := 
sorry

end solution_set_non_empty_iff_l258_258716


namespace binomial_probability_l258_258740

namespace binomial_proof

open ProbabilityTheory

def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- Given a random variable ξ that follows a binomial distribution B(6, 1/2),
  prove that the probability that ξ equals 3 is 5/16. -/
theorem binomial_probability : 
  let ξ : ℝ → ℝ := λ ω, ite (binom 6 3 = ω) 1 0
  in (binom 6 3) * (0.5^3) * (0.5^(6-3)) = 5 / 16 :=
by
  sorry

end binomial_proof

end binomial_probability_l258_258740


namespace pizza_slices_with_both_toppings_l258_258813

theorem pizza_slices_with_both_toppings (total_slices ham_slices pineapple_slices slices_with_both : ℕ)
  (h_total: total_slices = 15)
  (h_ham: ham_slices = 8)
  (h_pineapple: pineapple_slices = 12)
  (h_slices_with_both: slices_with_both + (ham_slices - slices_with_both) + (pineapple_slices - slices_with_both) = total_slices)
  : slices_with_both = 5 :=
by
  -- the proof would go here, but we use sorry to skip it
  sorry

end pizza_slices_with_both_toppings_l258_258813


namespace f_monotonic_decreasing_interval_l258_258495

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 2*x)

theorem f_monotonic_decreasing_interval : 
  ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 ≤ x2 → f x2 ≤ f x1 := 
sorry

end f_monotonic_decreasing_interval_l258_258495


namespace smallest_prime_with_digit_sum_23_l258_258086

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258086


namespace find_k_value_l258_258853

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l258_258853


namespace selling_price_percentage_l258_258639

-- Definitions for conditions
def ratio_cara_janet_jerry (c j je : ℕ) : Prop := 4 * (c + j + je) = 4 * c + 5 * j + 6 * je
def total_money (c j je total : ℕ) : Prop := c + j + je = total
def combined_loss (c j loss : ℕ) : Prop := c + j - loss = 36

-- The theorem statement to be proven
theorem selling_price_percentage (c j je total loss : ℕ) (h1 : ratio_cara_janet_jerry c j je) (h2 : total_money c j je total) (h3 : combined_loss c j loss)
    (h4 : total = 75) (h5 : loss = 9) : (36 * 100 / (c + j) = 80) := by
  sorry

end selling_price_percentage_l258_258639


namespace probability_same_color_opposite_foot_l258_258024

def total_shoes := 28

def black_pairs := 7
def brown_pairs := 4
def gray_pairs := 2
def red_pair := 1

def total_pairs := black_pairs + brown_pairs + gray_pairs + red_pair

theorem probability_same_color_opposite_foot : 
  (7 + 4 + 2 + 1) * 2 = total_shoes →
  (14 / 28 * (7 / 27) + 8 / 28 * (4 / 27) + 4 / 28 * (2 / 27) + 2 / 28 * (1 / 27)) = (20 / 63) :=
by
  sorry

end probability_same_color_opposite_foot_l258_258024


namespace average_age_l258_258307

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l258_258307


namespace smallest_prime_with_digit_sum_23_l258_258094

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem smallest_prime_with_digit_sum_23 : ∃ p, digit_sum p = 23 ∧ is_prime p ∧ p = 599 :=
sorry

end smallest_prime_with_digit_sum_23_l258_258094


namespace value_of_f_ln3_l258_258858

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end value_of_f_ln3_l258_258858


namespace number_of_ways_sum_of_primes_l258_258437

def is_prime (n : ℕ) : Prop := nat.prime n

theorem number_of_ways_sum_of_primes {a b : ℕ} (h₁ : a + b = 10003) (h₂ : is_prime a) (h₃ : is_prime b) : 
  finset.card {p : ℕ × ℕ | p.1 + p.2 = 10003 ∧ is_prime p.1 ∧ is_prime p.2} = 1 :=
sorry

end number_of_ways_sum_of_primes_l258_258437


namespace regression_line_is_y_eq_x_plus_1_l258_258724

theorem regression_line_is_y_eq_x_plus_1 :
  let points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by
  sorry 

end regression_line_is_y_eq_x_plus_1_l258_258724


namespace smallest_prime_with_digit_sum_23_proof_l258_258066

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def smallest_prime_with_digit_sum_23 : ℕ :=
  if h : 599.prime ∧ sum_of_digits 599 = 23 then 599 else 0

theorem smallest_prime_with_digit_sum_23_proof :
  ∃ n : ℕ, 599.prime ∧ sum_of_digits 599 = 23 ∧
  ∀ m : ℕ, m.prime ∧ sum_of_digits m = 23 → m ≥ 599 :=
by
  sorry

end smallest_prime_with_digit_sum_23_proof_l258_258066


namespace ratio_of_vanilla_chips_l258_258649

-- Definitions from the conditions
variable (V_c S_c V_v S_v : ℕ)
variable (H1 : V_c = S_c + 5)
variable (H2 : S_c = 25)
variable (H3 : V_v = 20)
variable (H4 : V_c + S_c + V_v + S_v = 90)

-- The statement we want to prove
theorem ratio_of_vanilla_chips : S_v / V_v = 3 / 4 := by
  sorry

end ratio_of_vanilla_chips_l258_258649


namespace f_even_l258_258774

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem f_even : ∀ x : ℝ, f x = f (-x) :=
by
  intro x
  sorry

end f_even_l258_258774


namespace problem_solution_l258_258981

open Real

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (∃ (C₁ : ℝ), (2 : ℝ)^x + (4 : ℝ)^y = C₁ ∧ C₁ = 2 * sqrt 2) ∧
  (∃ (C₂ : ℝ), 1 / x + 2 / y = C₂ ∧ C₂ = 9) ∧
  (∃ (C₃ : ℝ), x^2 + 4 * y^2 = C₃ ∧ C₃ = 1 / 2) :=
by
  sorry

end problem_solution_l258_258981


namespace circle_diameter_l258_258807

theorem circle_diameter (r d : ℝ) (h₀ : ∀ (r : ℝ), ∃ (d : ℝ), d = 2 * r) (h₁ : π * r^2 = 9 * π) :
  d = 6 :=
by
  rcases h₀ r with ⟨d, hd⟩
  sorry

end circle_diameter_l258_258807


namespace greater_of_T_N_l258_258188

/-- Define an 8x8 board and the number of valid domino placements. -/
def N : ℕ := 12988816

/-- A combinatorial number T representing the number of ways to place 24 dominoes on an 8x8 board. -/
axiom T : ℕ 

/-- We need to prove that T is greater than -N, where N is defined as 12988816. -/
theorem greater_of_T_N : T > - (N : ℤ) := sorry

end greater_of_T_N_l258_258188


namespace areas_equal_l258_258890

open EuclideanGeometry Metric

namespace Example

variables {A B C P H_A H_B H_C : Point ℝ}
variables {triangleABC : Triangle ℝ} (hABC : triangleABC = Triangle.mk A B C)

def isOrtho (P₁ P₂ P₃ P₄ : Point ℝ) : Prop :=
∠ P₁ P₂ P₃ = 90 ∨ ∠ P₁ P₂ P₄ = 90 ∨ ∠ P₁ P₄ P₃ = 90

theorem areas_equal (hP : InteriorPoint P (triangleABC))
                    (hH_A : is_orthocenter H_A P B C)
                    (hH_B : is_orthocenter H_B P A C)
                    (hH_C : is_orthocenter H_C P A B) :
  area (Triangle.mk H_A H_B H_C) = area triangleABC :=
sorry

end Example

end areas_equal_l258_258890


namespace problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l258_258989

noncomputable def f (x : ℝ) (k : ℝ) := (Real.log x - k - 1) * x

-- Problem 1: Interval of monotonicity and extremum.
theorem problem1_monotonic_and_extremum (k : ℝ):
  (k ≤ 0 → ∀ x, 1 < x → f x k = (Real.log x - k - 1) * x) ∧
  (k > 0 → (∀ x, 1 < x ∧ x < Real.exp k → f x k = (Real.log x - k - 1) * x) ∧
           (∀ x, Real.exp k < x → f x k = (Real.log x - k - 1) * x) ∧
           f (Real.exp k) k = -Real.exp k) := sorry

-- Problem 2: Range of k.
theorem problem2_range_of_k (k : ℝ):
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f x k < 4 * Real.log x) ↔
  k > 1 - (8 / Real.exp 2) := sorry

-- Problem 3: Inequality involving product of x1 and x2.
theorem problem3_inequality (x1 x2 : ℝ) (k : ℝ):
  x1 ≠ x2 ∧ f x1 k = f x2 k → x1 * x2 < Real.exp (2 * k) := sorry

end problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l258_258989


namespace percentage_neither_language_l258_258187

def total_diplomats : ℕ := 150
def french_speaking : ℕ := 17
def russian_speaking : ℕ := total_diplomats - 32
def both_languages : ℕ := 10 * total_diplomats / 100

theorem percentage_neither_language :
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  neither_language * 100 / total_diplomats = 20 :=
by
  let at_least_one_language := french_speaking + russian_speaking - both_languages
  let neither_language := total_diplomats - at_least_one_language
  sorry

end percentage_neither_language_l258_258187


namespace smallest_prime_with_digit_sum_23_l258_258073

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end smallest_prime_with_digit_sum_23_l258_258073


namespace gcd_max_value_l258_258920

theorem gcd_max_value (x y : ℤ) (h_posx : x > 0) (h_posy : y > 0) (h_sum : x + y = 780) :
  gcd x y ≤ 390 ∧ ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 780 ∧ gcd x' y' = 390 := by
  sorry

end gcd_max_value_l258_258920


namespace fran_speed_l258_258604

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l258_258604


namespace inscribed_circle_radius_third_of_circle_l258_258904

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ := 
  R * (Real.sqrt 3 - 1) / 2

theorem inscribed_circle_radius_third_of_circle (R : ℝ) (hR : R = 5) :
  inscribed_circle_radius R = 5 * (Real.sqrt 3 - 1) / 2 := by
  sorry

end inscribed_circle_radius_third_of_circle_l258_258904


namespace numbers_lcm_sum_l258_258552

theorem numbers_lcm_sum :
  ∃ A : List ℕ, A.length = 100 ∧
    (A.count 1 = 89 ∧ A.count 2 = 8 ∧ [4, 5, 6] ⊆ A) ∧
    A.sum = A.foldr lcm 1 :=
by
  sorry

end numbers_lcm_sum_l258_258552


namespace average_age_is_35_l258_258304

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l258_258304


namespace difference_between_largest_and_smallest_quarters_l258_258002

noncomputable def coin_collection : Prop :=
  ∃ (n d q : ℕ), 
    (n + d + q = 150) ∧ 
    (5 * n + 10 * d + 25 * q = 2000) ∧ 
    (forall (q1 q2 : ℕ), (n + d + q1 = 150) ∧ (5 * n + 10 * d + 25 * q1 = 2000) → 
     (n + d + q2 = 150) ∧ (5 * n + 10 * d + 25 * q2 = 2000) → 
     (q1 = q2))

theorem difference_between_largest_and_smallest_quarters : coin_collection :=
  sorry

end difference_between_largest_and_smallest_quarters_l258_258002


namespace sum_of_two_primes_unique_l258_258432

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l258_258432


namespace num_odd_five_digit_numbers_l258_258977

theorem num_odd_five_digit_numbers : 
  (∃ (digits : finset ℕ), 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, (n >= 1 ∧ n <= 5)) →
  (∃ n : ℕ, n = 72) :=
by
  sorry

end num_odd_five_digit_numbers_l258_258977


namespace ray_total_grocery_bill_l258_258755

noncomputable def meat_cost : ℝ := 5
noncomputable def crackers_cost : ℝ := 3.50
noncomputable def veg_cost_per_bag : ℝ := 2
noncomputable def veg_bags : ℕ := 4
noncomputable def cheese_cost : ℝ := 3.50
noncomputable def discount_rate : ℝ := 0.10

noncomputable def total_grocery_bill : ℝ :=
  let veg_total := veg_cost_per_bag * (veg_bags:ℝ)
  let total_before_discount := meat_cost + crackers_cost + veg_total + cheese_cost
  let discount := discount_rate * total_before_discount
  total_before_discount - discount

theorem ray_total_grocery_bill : total_grocery_bill = 18 :=
  by
  sorry

end ray_total_grocery_bill_l258_258755


namespace find_N_l258_258715

theorem find_N (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) :
  (x + y) / 3 = 1.222222222222222 := 
by
  -- We state the conditions.
  -- Lean will check whether these assumptions are consistent 
  sorry

end find_N_l258_258715


namespace function_increment_l258_258992

theorem function_increment (f : ℝ → ℝ) 
  (h : ∀ x, f x = 2 / x) : f 1.5 - f 2 = 1 / 3 := 
by {
  sorry
}

end function_increment_l258_258992


namespace Ria_original_savings_l258_258099

variables {R F : ℕ}

def initial_ratio (R F : ℕ) : Prop :=
  R * 3 = F * 5

def withdrawn_amount (R : ℕ) : ℕ :=
  R - 160

def new_ratio (R' F : ℕ) : Prop :=
  R' * 5 = F * 3

theorem Ria_original_savings (initial_ratio: initial_ratio R F)
  (new_ratio: new_ratio (withdrawn_amount R) F) : 
  R = 250 :=
by
  sorry

end Ria_original_savings_l258_258099


namespace find_num_3_year_olds_l258_258042

noncomputable def num_4_year_olds := 20
noncomputable def num_5_year_olds := 15
noncomputable def num_6_year_olds := 22
noncomputable def average_class_size := 35
noncomputable def num_students_class1 (num_3_year_olds : ℕ) := num_3_year_olds + num_4_year_olds
noncomputable def num_students_class2 := num_5_year_olds + num_6_year_olds
noncomputable def total_students (num_3_year_olds : ℕ) := num_students_class1 num_3_year_olds + num_students_class2

theorem find_num_3_year_olds (num_3_year_olds : ℕ) : 
  (total_students num_3_year_olds) / 2 = average_class_size → num_3_year_olds = 13 :=
by
  sorry

end find_num_3_year_olds_l258_258042


namespace new_team_average_weight_is_113_l258_258800

-- Defining the given constants and conditions
def original_players := 7
def original_average_weight := 121 
def weight_new_player1 := 110 
def weight_new_player2 := 60 

-- Definition to calculate the new average weight
def new_average_weight : ℕ :=
  let original_total_weight := original_players * original_average_weight
  let new_total_weight := original_total_weight + weight_new_player1 + weight_new_player2
  let new_total_players := original_players + 2
  new_total_weight / new_total_players

-- Statement to prove
theorem new_team_average_weight_is_113 : new_average_weight = 113 :=
sorry

end new_team_average_weight_is_113_l258_258800


namespace markup_rate_l258_258959

theorem markup_rate (S : ℝ) (C : ℝ) (hS : S = 8) (h1 : 0.20 * S = 0.10 * S + (S - C)) :
  ((S - C) / C) * 100 = 42.857 :=
by
  -- Assume given conditions and reasoning to conclude the proof
  sorry

end markup_rate_l258_258959


namespace numerical_puzzle_unique_solution_l258_258843

theorem numerical_puzzle_unique_solution :
  ∃ (A X Y P : ℕ), 
    A ≠ X ∧ A ≠ Y ∧ A ≠ P ∧ X ≠ Y ∧ X ≠ P ∧ Y ≠ P ∧
    (A * 10 + X) + (Y * 10 + X) = Y * 100 + P * 10 + A ∧
    A = 8 ∧ X = 9 ∧ Y = 1 ∧ P = 0 :=
sorry

end numerical_puzzle_unique_solution_l258_258843


namespace percentage_of_women_picnic_l258_258171

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic_l258_258171


namespace car_speed_constant_l258_258806

theorem car_speed_constant (v : ℝ) (hv : v ≠ 0)
  (condition_1 : (1 / 36) * 3600 = 100) 
  (condition_2 : (1 / v) * 3600 = 120) :
  v = 30 := by
  sorry

end car_speed_constant_l258_258806


namespace solve_for_wood_length_l258_258802

theorem solve_for_wood_length (y x : ℝ) (h1 : y - x = 4.5) (h2 : x - (1/2) * y = 1) :
  ∃! (x y : ℝ), (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  -- The content of the proof is omitted
  sorry

end solve_for_wood_length_l258_258802


namespace brad_running_speed_l258_258316

variable (dist_between_homes : ℕ)
variable (maxwell_speed : ℕ)
variable (time_maxwell_walks : ℕ)
variable (maxwell_start_time : ℕ)
variable (brad_start_time : ℕ)

#check dist_between_homes = 94
#check maxwell_speed = 4
#check time_maxwell_walks = 10
#check brad_start_time = maxwell_start_time + 1

theorem brad_running_speed (dist_between_homes : ℕ) (maxwell_speed : ℕ) (time_maxwell_walks : ℕ) (maxwell_start_time : ℕ) (brad_start_time : ℕ) :
  dist_between_homes = 94 →
  maxwell_speed = 4 →
  time_maxwell_walks = 10 →
  brad_start_time = maxwell_start_time + 1 →
  (dist_between_homes - maxwell_speed * time_maxwell_walks) / (time_maxwell_walks - (brad_start_time - maxwell_start_time)) = 6 :=
by
  intros
  sorry

end brad_running_speed_l258_258316


namespace positive_number_property_l258_258951

theorem positive_number_property (x : ℝ) (h_pos : 0 < x) (h_eq : (x^2) / 100 = 9) : x = 30 :=
sorry

end positive_number_property_l258_258951


namespace jennifer_score_l258_258173

theorem jennifer_score 
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (unanswered_questions : ℕ)
  (points_per_correct : ℤ)
  (points_deduction_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (h_total : total_questions = 30)
  (h_correct : correct_answers = 15)
  (h_incorrect : incorrect_answers = 10)
  (h_unanswered : unanswered_questions = 5)
  (h_points_correct : points_per_correct = 2)
  (h_deduction_incorrect : points_deduction_incorrect = -1)
  (h_points_unanswered : points_per_unanswered = 0) : 
  ∃ (score : ℤ), score = (correct_answers * points_per_correct 
                          + incorrect_answers * points_deduction_incorrect 
                          + unanswered_questions * points_per_unanswered) 
                        ∧ score = 20 := 
by
  sorry

end jennifer_score_l258_258173


namespace book_sale_total_amount_l258_258360

noncomputable def total_amount_received (total_books price_per_book : ℕ → ℝ) : ℝ :=
  price_per_book 80

theorem book_sale_total_amount (B : ℕ)
  (h1 : (1/3 : ℚ) * B = 40)
  (h2 : ∀ (n : ℕ), price_per_book n = 3.50) :
  total_amount_received B price_per_book = 280 := 
by
  sorry

end book_sale_total_amount_l258_258360


namespace pick_three_different_cards_in_order_l258_258818

theorem pick_three_different_cards_in_order :
  (52 * 51 * 50) = 132600 :=
by
  sorry

end pick_three_different_cards_in_order_l258_258818


namespace cost_of_one_bag_of_onions_l258_258636

theorem cost_of_one_bag_of_onions (price_per_onion : ℕ) (total_onions : ℕ) (num_bags : ℕ) (h_price : price_per_onion = 200) (h_onions : total_onions = 180) (h_bags : num_bags = 6) :
  (total_onions / num_bags) * price_per_onion = 6000 := 
  by
  sorry

end cost_of_one_bag_of_onions_l258_258636


namespace no_prime_sum_10003_l258_258426

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l258_258426


namespace g_eq_g_inv_solution_l258_258133

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem g_eq_g_inv_solution (x : ℝ) : g x = g_inv x ↔ x = 5 / 3 :=
by
  sorry

end g_eq_g_inv_solution_l258_258133


namespace calculate_value_l258_258532

theorem calculate_value :
  12 * ( (1 / 3 : ℝ) + (1 / 4) + (1 / 6) )⁻¹ = 16 :=
sorry

end calculate_value_l258_258532


namespace total_weight_proof_l258_258962

-- Define molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Define moles of elements in each compound
def moles_C4H10 : ℕ := 8
def moles_C3H8 : ℕ := 5
def moles_CH4 : ℕ := 3

-- Define the molar masses of each compound
def molar_mass_C4H10 : ℝ := 4 * molar_mass_C + 10 * molar_mass_H
def molar_mass_C3H8 : ℝ := 3 * molar_mass_C + 8 * molar_mass_H
def molar_mass_CH4 : ℝ := 1 * molar_mass_C + 4 * molar_mass_H

-- Define the total weight
def total_weight : ℝ :=
  moles_C4H10 * molar_mass_C4H10 +
  moles_C3H8 * molar_mass_C3H8 +
  moles_CH4 * molar_mass_CH4

theorem total_weight_proof :
  total_weight = 733.556 := by
  sorry

end total_weight_proof_l258_258962


namespace partial_fraction_product_is_correct_l258_258337

-- Given conditions
def fraction_decomposition (x A B C : ℝ) :=
  ( (x^2 + 5 * x - 14) / (x^3 - 3 * x^2 - x + 3) = A / (x - 1) + B / (x - 3) + C / (x + 1) )

-- Statement we want to prove
theorem partial_fraction_product_is_correct (A B C : ℝ) (h : ∀ x : ℝ, fraction_decomposition x A B C) :
  A * B * C = -25 / 2 :=
sorry

end partial_fraction_product_is_correct_l258_258337


namespace number_of_possible_values_b_l258_258330

theorem number_of_possible_values_b : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ b : ℕ, b ≥ 2 → (b^3 ≤ 256) ∧ (256 < b^4) ↔ (b = 5 ∨ b = 6)) :=
by {
  sorry
}

end number_of_possible_values_b_l258_258330


namespace polynomial_coeff_sum_l258_258194

variable (d : ℤ)
variable (h : d ≠ 0)

theorem polynomial_coeff_sum : 
  (∃ a b c e : ℤ, (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e ∧ a + b + c + e = 42) :=
by
  sorry

end polynomial_coeff_sum_l258_258194


namespace octahedron_plane_intersection_l258_258952

theorem octahedron_plane_intersection 
  (s : ℝ) 
  (a b c : ℕ) 
  (ha : Nat.Coprime a c) 
  (hb : ∀ p : ℕ, Prime p → p^2 ∣ b → False) 
  (hs : s = 2) 
  (hangle : ∀ θ, θ = 45 ∧ θ = 45) 
  (harea : ∃ A, A = (s^2 * Real.sqrt 3) / 2 ∧ A = a * Real.sqrt b / c): 
  a + b + c = 11 := 
by 
  sorry

end octahedron_plane_intersection_l258_258952


namespace max_square_test_plots_l258_258234

theorem max_square_test_plots (length width fence : ℕ)
  (h_length : length = 36)
  (h_width : width = 66)
  (h_fence : fence = 2200) :
  ∃ (n : ℕ), n * (11 / 6) * n = 264 ∧
      (36 * n + (11 * n - 6) * 66) ≤ 2200 := sorry

end max_square_test_plots_l258_258234


namespace unique_f_l258_258736

open Rat

def pos_rat := {q : ℚ // q > 0}

noncomputable def f : pos_rat → pos_rat := sorry

axiom f_eq (x y : pos_rat) : f(x) = f(x + y) + f(x + x^2 * f(y))

theorem unique_f (f : pos_rat → pos_rat) 
(h : ∀ x y : pos_rat, f(x) = f(x + y) + f(x + x^2 * f(y))) : 
(f = λ x, ⟨1 / x.1, by simp ⟩) := sorry

end unique_f_l258_258736


namespace min_empty_squares_eq_nine_l258_258964

-- Definition of the problem conditions
def chessboard_size : ℕ := 9
def total_squares : ℕ := chessboard_size * chessboard_size
def number_of_white_squares : ℕ := 4 * chessboard_size
def number_of_black_squares : ℕ := 5 * chessboard_size
def minimum_number_of_empty_squares : ℕ := number_of_black_squares - number_of_white_squares

-- Theorem to prove minimum number of empty squares
theorem min_empty_squares_eq_nine :
  minimum_number_of_empty_squares = 9 :=
by
  -- Placeholder for the proof
  sorry

end min_empty_squares_eq_nine_l258_258964


namespace fraction_of_ponies_with_horseshoes_l258_258363

theorem fraction_of_ponies_with_horseshoes 
  (P H : ℕ) 
  (h1 : H = P + 4) 
  (h2 : H + P ≥ 164) 
  (x : ℚ)
  (h3 : ∃ (n : ℕ), n = (5 / 8) * (x * P)) :
  x = 1 / 10 := by
  sorry

end fraction_of_ponies_with_horseshoes_l258_258363


namespace slices_per_person_l258_258501

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l258_258501


namespace average_age_is_35_l258_258302

variable (Tonya_age : ℕ)
variable (John_age : ℕ)
variable (Mary_age : ℕ)

noncomputable def average_age (Tonya_age John_age Mary_age : ℕ) : ℕ :=
  (Tonya_age + John_age + Mary_age) / 3

theorem average_age_is_35 (h1 : Tonya_age = 60) 
                          (h2 : John_age = Tonya_age / 2)
                          (h3 : John_age = 2 * Mary_age) : 
                          average_age Tonya_age John_age Mary_age = 35 :=
by 
  sorry

end average_age_is_35_l258_258302


namespace proof_min_value_a3_and_a2b2_l258_258738

noncomputable def min_value_a3_and_a2b2 (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  (a1 > 0) ∧ (a2 > 0) ∧ (a3 > 0) ∧ (b1 > 0) ∧ (b2 > 0) ∧ (b3 > 0) ∧
  (a2 = a1 + b1) ∧ (a3 = a1 + 2 * b1) ∧ (b2 = b1 * a1) ∧ 
  (b3 = b1 * a1^2) ∧ (a3 = b3) ∧ 
  (a3 = 3 * Real.sqrt 6 / 2) ∧
  (a2 * b2 = 15 * Real.sqrt 6 / 8) 

theorem proof_min_value_a3_and_a2b2 : ∃ (a1 a2 a3 b1 b2 b3 : ℝ), min_value_a3_and_a2b2 a1 a2 a3 b1 b2 b3 :=
by
  use 2*Real.sqrt 6/3, 5*Real.sqrt 6/4, 3*Real.sqrt 6/2, Real.sqrt 6/4, 3/2, 3*Real.sqrt 6/2
  sorry

end proof_min_value_a3_and_a2b2_l258_258738


namespace binom_7_4_l258_258542

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l258_258542


namespace positive_intervals_of_product_l258_258835

theorem positive_intervals_of_product (x : ℝ) : 
  ((x + 2) * (x - 3) > 0) ↔ (x < -2 ∨ x > 3) := 
sorry

end positive_intervals_of_product_l258_258835


namespace chameleons_impossible_all_white_l258_258014

/--
On Easter Island, there are initial counts of blue (12), white (25), and red (8) chameleons.
When two chameleons of different colors meet, they both change to the third color.
Prove that it is impossible for all chameleons to become white.
--/
theorem chameleons_impossible_all_white :
  let n1 := 12 -- Blue chameleons
  let n2 := 25 -- White chameleons
  let n3 := 8  -- Red chameleons
  (∀ (n1 n2 n3 : ℕ), (n1 + n2 + n3 = 45) → 
   ∀ (k : ℕ), ∃ m1 m2 m3 : ℕ, (m1 - m2) % 3 = (n1 - n2) % 3 ∧ (m1 - m3) % 3 = (n1 - n3) % 3 ∧ 
   (m2 - m3) % 3 = (n2 - n3) % 3) → False := sorry

end chameleons_impossible_all_white_l258_258014


namespace sum_of_two_primes_unique_l258_258431

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l258_258431


namespace coin_toss_sequence_probability_l258_258597

/-- The probability of getting 4 heads followed by 3 tails and then finishing with 3 heads 
in a sequence of 10 coin tosses is 1/1024. -/
theorem coin_toss_sequence_probability : 
  ( (1 / 2 : ℝ) ^ 4) * ( (1 / 2 : ℝ) ^ 3) * ((1 / 2 : ℝ) ^ 3) = 1 / 1024 := 
by sorry

end coin_toss_sequence_probability_l258_258597


namespace min_value_y_l258_258851

theorem min_value_y (x : ℝ) (h : x > 0) : ∃ y, y = x + 4 / x^2 ∧ (∀ z, z = x + 4 / x^2 → y ≤ z) := 
sorry

end min_value_y_l258_258851


namespace problem1_problem2_l258_258127

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 :=
by sorry

theorem problem2 : (-1)^7 * 2 + (-3)^2 / 9 = -1 :=
by sorry

end problem1_problem2_l258_258127


namespace train_length_300_l258_258954

/-- 
Proving the length of the train given the conditions on crossing times and length of the platform.
-/
theorem train_length_300 (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 200 = V * 30) : 
  L = 300 := 
by
  sorry

end train_length_300_l258_258954


namespace determine_function_l258_258134

theorem determine_function (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (x + 1)) :=
by
  sorry

end determine_function_l258_258134


namespace geometric_sequence_a5_l258_258885

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n, a n + a (n + 1) = 3 * (1 / 2) ^ n)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 1 / 2) :
  a 5 = 1 / 16 :=
sorry

end geometric_sequence_a5_l258_258885


namespace average_of_remaining_numbers_l258_258766

theorem average_of_remaining_numbers 
  (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50) = 20)
  (h_disc : 45 ∈ numbers ∧ 55 ∈ numbers) 
  (h_count_45_55 : numbers.count 45 = 1 ∧ numbers.count 55 = 1) :
  (numbers.sum - 45 - 55) / (50 - 2) = 18.75 :=
by
  sorry

end average_of_remaining_numbers_l258_258766


namespace purchase_price_eq_360_l258_258916

theorem purchase_price_eq_360 (P : ℝ) (M : ℝ) (H1 : M = 30) (H2 : M = 0.05 * P + 12) : P = 360 :=
by
  sorry

end purchase_price_eq_360_l258_258916


namespace fraction_subtraction_proof_l258_258352

theorem fraction_subtraction_proof : 
  (21 / 12) - (18 / 15) = 11 / 20 := 
by 
  sorry

end fraction_subtraction_proof_l258_258352


namespace simplify_product_l258_258476

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l258_258476


namespace modular_inverse_addition_l258_258554

theorem modular_inverse_addition :
  (3 * 9 + 9 * 37) % 63 = 45 :=
by
  sorry

end modular_inverse_addition_l258_258554


namespace find_original_number_l258_258947

theorem find_original_number (x : ℕ) :
  (43 * x - 34 * x = 1251) → x = 139 :=
by
  sorry

end find_original_number_l258_258947


namespace solve_for_x_l258_258264

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l258_258264


namespace inverse_variation_l258_258900

variable (a b : ℝ)

theorem inverse_variation (h_ab : a * b = 400) :
  (b = 0.25 ∧ a = 1600) ∨ (b = 1.0 ∧ a = 400) :=
  sorry

end inverse_variation_l258_258900


namespace tan_angle_sum_identity_l258_258153

theorem tan_angle_sum_identity
  (θ : ℝ)
  (h1 : θ > π / 2 ∧ θ < π)
  (h2 : Real.cos θ = -3 / 5) :
  Real.tan (θ + π / 4) = -1 / 7 := by
  sorry

end tan_angle_sum_identity_l258_258153


namespace no_positive_integer_n_satisfies_conditions_l258_258974

theorem no_positive_integer_n_satisfies_conditions :
  ¬ ∃ (n : ℕ), (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ (100 ≤ 4 * n ∧ 4 * n ≤ 999) :=
by
  sorry

end no_positive_integer_n_satisfies_conditions_l258_258974


namespace simplify_product_l258_258477

theorem simplify_product : (18 : ℚ) * (8 / 12) * (1 / 6) = 2 := by
  sorry

end simplify_product_l258_258477


namespace smallest_prime_with_digit_sum_23_l258_258057

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l258_258057


namespace max_female_students_min_people_in_group_l258_258238

-- Problem 1: Given z = 4, the maximum number of female students is 6
theorem max_female_students (x y : ℕ) (h1 : x > y) (h2 : y > 4) (h3 : x < 8) : y <= 6 :=
sorry

-- Problem 2: The minimum number of people in the group is 12
theorem min_people_in_group (x y z : ℕ) (h1 : x > y) (h2 : y > z) (h3 : 2 * z > x) : 12 <= x + y + z :=
sorry

end max_female_students_min_people_in_group_l258_258238


namespace completing_the_square_step_l258_258100

theorem completing_the_square_step (x : ℝ) : 
  x^2 + 4 * x + 2 = 0 → x^2 + 4 * x = -2 :=
by
  intro h
  sorry

end completing_the_square_step_l258_258100


namespace kelly_snacks_l258_258730

theorem kelly_snacks (peanuts raisins : ℝ) (h_peanuts : peanuts = 0.1) (h_raisins : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end kelly_snacks_l258_258730


namespace geometric_sequence_S28_l258_258558

noncomputable def geom_sequence_sum (S : ℕ → ℝ) (a : ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, S (n * (n + 1) / 2) = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S28 {S : ℕ → ℝ} (a r : ℝ)
  (h1 : geom_sequence_sum S a r)
  (h2 : S 14 = 3)
  (h3 : 3 * S 7 = 3) :
  S 28 = 15 :=
by
  sorry

end geometric_sequence_S28_l258_258558


namespace binom_7_4_eq_35_l258_258534

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l258_258534


namespace necessary_but_not_sufficient_condition_l258_258956

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (a > b → a + 1 > b) ∧ (∃ a b : ℝ, a + 1 > b ∧ ¬ a > b) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l258_258956


namespace part_one_part_two_l258_258680

def f (x : ℝ) : ℝ := abs (3 * x + 2)

theorem part_one (x : ℝ) : f x < 4 - abs (x - 1) ↔ x ∈ Set.Ioo (-5 / 4) (1 / 2) :=
sorry

noncomputable def g (x a : ℝ) : ℝ :=
if x < -2/3 then 2 * x + 2 + a
else if x ≤ a then -4 * x - 2 + a
else -2 * x - 2 - a

theorem part_two (m n a : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) :
  (∀ (x : ℝ), abs (x - a) - f x ≤ 1 / m + 1 / n) ↔ (0 < a ∧ a ≤ 10 / 3) :=
sorry

end part_one_part_two_l258_258680


namespace product_of_last_two_digits_l258_258353

theorem product_of_last_two_digits (A B : ℕ) (h₁ : A + B = 17) (h₂ : 4 ∣ (10 * A + B)) :
  A * B = 72 := sorry

end product_of_last_two_digits_l258_258353


namespace ball_falls_in_middle_pocket_l258_258596

theorem ball_falls_in_middle_pocket (p q : ℕ) (hp : p % 2 = 1) (hq : q % 2 = 1) : 
  ∃ k : ℕ, (k * p) % (2 * q) = 0 :=
by
  sorry

end ball_falls_in_middle_pocket_l258_258596


namespace bullets_shot_per_person_l258_258781

-- Definitions based on conditions
def num_people : ℕ := 5
def initial_bullets_per_person : ℕ := 25
def total_remaining_bullets : ℕ := 25

-- Statement to prove
theorem bullets_shot_per_person (x : ℕ) :
  (initial_bullets_per_person * num_people - num_people * x) = total_remaining_bullets → x = 20 :=
by
  sorry

end bullets_shot_per_person_l258_258781


namespace deepak_age_l258_258340

theorem deepak_age : ∀ (R D : ℕ), (R / D = 4 / 3) ∧ (R + 6 = 18) → D = 9 :=
by
  sorry

end deepak_age_l258_258340


namespace abs_neg_2023_l258_258483

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l258_258483


namespace total_students_experimental_primary_school_l258_258375

theorem total_students_experimental_primary_school : 
  ∃ (n : ℕ), 
  n = (21 + 11) * 28 ∧ 
  n = 896 := 
by {
  -- Since the proof is not required, we use "sorry"
  sorry
}

end total_students_experimental_primary_school_l258_258375


namespace whose_number_is_larger_l258_258349

theorem whose_number_is_larger
    (vasya_prod : ℕ := 4^12)
    (petya_prod : ℕ := 2^25) :
    petya_prod > vasya_prod :=
    by
    sorry

end whose_number_is_larger_l258_258349


namespace probability_B_not_occur_given_A_occurs_expected_value_X_l258_258870

namespace DieProblem

def event_A := {1, 2, 3}
def event_B := {1, 2, 4}

def num_trials := 10
def num_occurrences_A := 6

theorem probability_B_not_occur_given_A_occurs :
  (∑ i in Finset.range (num_trials.choose num_occurrences_A), 
    (1/6)^num_occurrences_A * (1/3)^(num_trials - num_occurrences_A)) / 
  (num_trials.choose num_occurrences_A * (1/2)^(num_trials)) = 2.71 * 10^(-4) :=
sorry

theorem expected_value_X : 
  (6 * (2/3)) + (4 * (1/3)) = 16 / 3 :=
sorry

end DieProblem

end probability_B_not_occur_given_A_occurs_expected_value_X_l258_258870


namespace propositions_false_l258_258614

structure Plane :=
(is_plane : Prop)

structure Line :=
(in_plane : Plane → Prop)

def is_parallel (p1 p2 : Plane) : Prop := sorry
def is_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular (l1 l2 : Line) : Prop := sorry

variable (α β : Plane)
variable (l m : Line)

axiom α_neq_β : α ≠ β
axiom l_in_α : l.in_plane α
axiom m_in_β : m.in_plane β

theorem propositions_false :
  ¬(is_parallel α β → line_parallel l m) ∧ 
  ¬(line_perpendicular l m → is_perpendicular α β) := 
sorry

end propositions_false_l258_258614


namespace average_sales_is_167_5_l258_258247

def sales_january : ℝ := 150
def sales_february : ℝ := 90
def sales_march : ℝ := 1.5 * sales_february
def sales_april : ℝ := 180
def sales_may : ℝ := 210
def sales_june : ℝ := 240
def total_sales : ℝ := sales_january + sales_february + sales_march + sales_april + sales_may + sales_june
def number_of_months : ℝ := 6

theorem average_sales_is_167_5 :
  total_sales / number_of_months = 167.5 :=
sorry

end average_sales_is_167_5_l258_258247


namespace download_time_ratio_l258_258621

-- Define the conditions of the problem
def mac_download_time : ℕ := 10
def audio_glitches : ℕ := 2 * 4
def video_glitches : ℕ := 6
def time_with_glitches : ℕ := audio_glitches + video_glitches
def time_without_glitches : ℕ := 2 * time_with_glitches
def total_time : ℕ := 82

-- Define the Windows download time as a variable
def windows_download_time : ℕ := total_time - (mac_download_time + time_with_glitches + time_without_glitches)

-- Prove the required ratio
theorem download_time_ratio : 
  (windows_download_time / mac_download_time = 3) :=
by
  -- Perform a straightforward calculation as defined in the conditions and solution steps
  sorry

end download_time_ratio_l258_258621


namespace clock_in_2023_hours_l258_258630

theorem clock_in_2023_hours (current_time : ℕ) (h_current_time : current_time = 3) : 
  (current_time + 2023) % 12 = 10 := 
by {
  -- context: non-computational (time kept in modulo world and not real increments)
  sorry
}

end clock_in_2023_hours_l258_258630


namespace cubic_polynomial_root_l258_258488

theorem cubic_polynomial_root (a b c : ℕ) (h : 27 * x^3 - 9 * x^2 - 9 * x - 3 = 0) : 
  (a + b + c = 11) :=
sorry

end cubic_polynomial_root_l258_258488


namespace find_x4_l258_258814

open Real

theorem find_x4 (x : ℝ) (h₁ : 0 < x) (h₂ : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end find_x4_l258_258814


namespace derivative_f_l258_258770

noncomputable def f (x : ℝ) := x * Real.cos x - Real.sin x

theorem derivative_f :
  ∀ x : ℝ, deriv f x = -x * Real.sin x :=
by
  sorry

end derivative_f_l258_258770


namespace maximum_value_expression_l258_258386

theorem maximum_value_expression (a b : ℝ) (h : a^2 + b^2 = 9) : 
  ∃ x, x = 5 ∧ ∀ y, y = ab - b + a → y ≤ x :=
by
  sorry

end maximum_value_expression_l258_258386


namespace binom_7_4_eq_35_l258_258536

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l258_258536


namespace unique_polynomial_solution_l258_258970

def polynomial_homogeneous_of_degree_n (P : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (t x y : ℝ), P (t * x) (t * y) = t^n * P x y

def polynomial_symmetric_condition (P : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0

def polynomial_value_at_point (P : ℝ → ℝ → ℝ) : Prop :=
  P 1 0 = 1

theorem unique_polynomial_solution (P : ℝ → ℝ → ℝ) (n : ℕ) :
  polynomial_homogeneous_of_degree_n P n →
  polynomial_symmetric_condition P →
  polynomial_value_at_point P →
  ∀ x y : ℝ, P x y = (x + y)^n * (x - 2 * y) := 
by
  intros h_deg h_symm h_value x y
  sorry

end unique_polynomial_solution_l258_258970


namespace binom_7_4_l258_258543

theorem binom_7_4 : Nat.choose 7 4 = 35 := 
by
  sorry

end binom_7_4_l258_258543


namespace cube_ratio_sum_l258_258912

theorem cube_ratio_sum (a b : ℝ) (h1 : |a| ≠ |b|) (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 :=
by
  sorry

end cube_ratio_sum_l258_258912


namespace diane_total_loss_l258_258837

def initial_amount : ℤ := 100
def amount_won : ℤ := 65
def amount_owed : ℤ := 50

theorem diane_total_loss : initial_amount + amount_won - (initial_amount + amount_won) + amount_owed = 215 := by
  calc
    initial_amount + amount_won - (initial_amount + amount_won) + amount_owed
      = amount_owed : by rw [add_sub_cancel'_right]
      = 50 : rfl
      = 215 : by sorry

end diane_total_loss_l258_258837


namespace nesbitts_inequality_l258_258727

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end nesbitts_inequality_l258_258727


namespace smallest_prime_digit_sum_23_l258_258079

open Nat

def digit_sum (n : ℕ) : ℕ :=
  (n.toString.toList.map (λ c, c.toNat - '0'.toNat)).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)

theorem smallest_prime_digit_sum_23 : ∃ p, is_prime p ∧ digit_sum p = 23 ∧ ∀ q, is_prime q ∧ digit_sum q = 23 → p <= q :=
by
  use 193
  sorry

end smallest_prime_digit_sum_23_l258_258079


namespace exists_n_consecutive_non_prime_or_prime_power_l258_258320

theorem exists_n_consecutive_non_prime_or_prime_power (n : ℕ) (h : n > 0) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ (Nat.Prime (seq i)) ∧ ¬ (∃ p k : ℕ, p.Prime ∧ k > 1 ∧ seq i = p ^ k)) :=
by
  sorry

end exists_n_consecutive_non_prime_or_prime_power_l258_258320


namespace arc_length_of_sector_l258_258029

theorem arc_length_of_sector 
  (R : ℝ) (θ : ℝ) (hR : R = Real.pi) (hθ : θ = 2 * Real.pi / 3) : 
  (R * θ = 2 * Real.pi^2 / 3) := 
by
  rw [hR, hθ]
  sorry

end arc_length_of_sector_l258_258029


namespace average_age_l258_258305

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l258_258305


namespace Fran_speed_l258_258606

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l258_258606


namespace find_a_l258_258267

noncomputable def a_value_given_conditions : ℝ :=
  let A := 30 * Real.pi / 180
  let C := 105 * Real.pi / 180
  let B := 180 * Real.pi / 180 - A - C
  let b := 8
  let a := (b * Real.sin A) / Real.sin B
  a

theorem find_a :
  a_value_given_conditions = 4 * Real.sqrt 2 :=
by
  -- We assume that the value computation as specified is correct
  -- hence this is just stating the problem.
  sorry

end find_a_l258_258267


namespace solve_system_of_equations_l258_258021

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ (x = 2) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l258_258021


namespace smallest_prime_with_digit_sum_23_l258_258059

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl Nat.add 0

-- Statement of the proof
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Prime p ∧ digit_sum p = 23 ∧ (∀ q, Prime q → digit_sum q = 23 → p ≤ q) :=
sorry

end smallest_prime_with_digit_sum_23_l258_258059


namespace simple_interest_calculation_l258_258819

theorem simple_interest_calculation (P R T : ℝ) (H₁ : P = 8925) (H₂ : R = 9) (H₃ : T = 5) : 
  P * R * T / 100 = 4016.25 :=
by
  sorry

end simple_interest_calculation_l258_258819


namespace acute_triangle_cannot_divide_into_two_obtuse_l258_258295

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem acute_triangle_cannot_divide_into_two_obtuse (A B C A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_divide : A + B + C = 180 ∧ A1 + B1 + C1 = 180 ∧ A2 + B2 + C2 = 180)
  (h_sum : A1 + A2 = A ∧ B1 + B2 = B ∧ C1 + C2 = C) :
  ¬ (is_obtuse_triangle A1 B1 C1 ∧ is_obtuse_triangle A2 B2 C2) :=
sorry

end acute_triangle_cannot_divide_into_two_obtuse_l258_258295


namespace sufficient_not_necessary_condition_l258_258009

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (hx : x = 2)
    (ha : a = (x, 1)) (hb : b = (4, x)) : 
    (∃ k : ℝ, a = (k * b.1, k * b.2)) ∧ (¬ (∀ k : ℝ, a = (k * b.1, k * b.2))) :=
by 
  sorry

end sufficient_not_necessary_condition_l258_258009


namespace per_capita_income_ratio_l258_258027

theorem per_capita_income_ratio
  (PL_10 PZ_10 PL_now PZ_now : ℝ)
  (h1 : PZ_10 = 0.4 * PL_10)
  (h2 : PZ_now = 0.8 * PL_now)
  (h3 : PL_now = 3 * PL_10) :
  PZ_now / PZ_10 = 6 := by
  -- Proof to be filled
  sorry

end per_capita_income_ratio_l258_258027


namespace min_weighings_to_identify_fake_l258_258924

def piles := 1000000
def coins_per_pile := 1996
def weight_real_coin := 10
def weight_fake_coin := 9
def expected_total_weight : Nat :=
  (piles * (piles + 1) / 2) * weight_real_coin

theorem min_weighings_to_identify_fake :
  (∃ k : ℕ, k < piles ∧ 
  ∀ (W : ℕ), W = expected_total_weight - k → k = expected_total_weight - W) →
  true := 
by
  sorry

end min_weighings_to_identify_fake_l258_258924


namespace unique_zero_in_interval_l258_258991

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x ^ 2

theorem unique_zero_in_interval
  (a : ℝ) (ha : a > 0)
  (x₀ : ℝ) (hx₀ : f a x₀ = 0)
  (h_interval : -1 < x₀ ∧ x₀ < 0) :
  Real.exp (-2) < x₀ + 1 ∧ x₀ + 1 < Real.exp (-1) :=
sorry

end unique_zero_in_interval_l258_258991


namespace combined_population_correct_l258_258778

theorem combined_population_correct (W PP LH N : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : LH = 2 * W + 600)
  (hN : N = 3 * (PP - W)) :
  PP + LH + N = 24900 :=
by
  sorry

end combined_population_correct_l258_258778


namespace unique_solution_for_all_y_l258_258685

theorem unique_solution_for_all_y (x : ℝ) (h : ∀ y : ℝ, 8 * x * y - 12 * y + 2 * x - 3 = 0) : x = 3 / 2 :=
sorry

end unique_solution_for_all_y_l258_258685


namespace miles_ridden_further_l258_258643

theorem miles_ridden_further (distance_ridden distance_walked : ℝ) (h1 : distance_ridden = 3.83) (h2 : distance_walked = 0.17) :
  distance_ridden - distance_walked = 3.66 := 
by sorry

end miles_ridden_further_l258_258643


namespace bacteria_colony_exceeds_500_l258_258175

theorem bacteria_colony_exceeds_500 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → 4 * 3^m ≤ 500) ∧ 4 * 3^n > 500 :=
sorry

end bacteria_colony_exceeds_500_l258_258175


namespace max_brownie_pieces_l258_258882

theorem max_brownie_pieces (base height piece_width piece_height : ℕ) 
    (h_base : base = 30) (h_height : height = 24)
    (h_piece_width : piece_width = 3) (h_piece_height : piece_height = 4) :
  (base / piece_width) * (height / piece_height) = 60 :=
by sorry

end max_brownie_pieces_l258_258882


namespace total_people_waiting_l258_258122

theorem total_people_waiting 
  (initial_first_line : ℕ := 7)
  (left_first_line : ℕ := 4)
  (joined_first_line : ℕ := 8)
  (initial_second_line : ℕ := 12)
  (left_second_line : ℕ := 3)
  (joined_second_line : ℕ := 10)
  (initial_third_line : ℕ := 15)
  (left_third_line : ℕ := 5)
  (joined_third_line : ℕ := 7) :
  (initial_first_line - left_first_line + joined_first_line) +
  (initial_second_line - left_second_line + joined_second_line) +
  (initial_third_line - left_third_line + joined_third_line) = 47 :=
by
  sorry

end total_people_waiting_l258_258122


namespace sum_of_averages_is_six_l258_258335

variable (a b c d e : ℕ)

def average_teacher : ℚ :=
  (5 * a + 4 * b + 3 * c + 2 * d + e) / (a + b + c + d + e)

def average_kati : ℚ :=
  (5 * e + 4 * d + 3 * c + 2 * b + a) / (a + b + c + d + e)

theorem sum_of_averages_is_six (a b c d e : ℕ) : 
    average_teacher a b c d e + average_kati a b c d e = 6 := by
  sorry

end sum_of_averages_is_six_l258_258335


namespace cos_value_third_quadrant_l258_258691

theorem cos_value_third_quadrant (x : Real) (h1 : Real.sin x = -1 / 3) (h2 : π < x ∧ x < 3 * π / 2) : 
  Real.cos x = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_value_third_quadrant_l258_258691


namespace no_prime_sum_10003_l258_258443

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l258_258443


namespace find_a_l258_258861

theorem find_a (a b c : ℕ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end find_a_l258_258861


namespace minimum_value_of_PA_PF_l258_258987

noncomputable def ellipse_min_distance : ℝ :=
  let F := (1, 0)
  let A := (1, 1)
  let a : ℝ := 3
  let F1 := (-1, 0)
  let d_A_F1 : ℝ := Real.sqrt ((-1 - 1)^2 + (0 - 1)^2)
  6 - d_A_F1

theorem minimum_value_of_PA_PF :
  ellipse_min_distance = 6 - Real.sqrt 5 :=
by
  sorry

end minimum_value_of_PA_PF_l258_258987


namespace sum_of_extremes_of_even_sequence_l258_258201

theorem sum_of_extremes_of_even_sequence (m : ℕ) (h : Even m) (z : ℤ)
  (hs : ∀ b : ℤ, z = (m * b + (2 * (1 to m-1).sum id) / m)) :
  ∃ b : ℤ, (2 * b + 2 * (m - 1)) = 2 * z :=
by
  sorry

end sum_of_extremes_of_even_sequence_l258_258201


namespace slices_per_person_l258_258504

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l258_258504


namespace probability_two_rainy_days_l258_258634

theorem probability_two_rainy_days : 
  let numbers := ["907", "966", "191", "925", "271", "932", "812", "458", "569", "683",
                  "431", "257", "393", "027", "556", "488", "730", "113", "537", "989"];
  let rain_condition := ['1', '2', '3', '4'];
  let is_two_rainy_days := λ s : String, s.to_list.filter (λ x, x ∈ rain_condition).length = 2;
  let valid_groups := numbers.filter is_two_rainy_days;
  valid_groups.length = 5 →
  (5 : ℚ) / (20 : ℚ) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end probability_two_rainy_days_l258_258634


namespace supplementary_angles_ratio_l258_258786

theorem supplementary_angles_ratio (A B : ℝ) (h1 : A + B = 180) (h2 : A / B = 5 / 4) : B = 80 :=
by
   sorry

end supplementary_angles_ratio_l258_258786


namespace binomial_7_4_eq_35_l258_258545

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l258_258545


namespace people_in_room_l258_258281

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l258_258281


namespace point_on_xOz_plane_l258_258551

def point : ℝ × ℝ × ℝ := (1, 0, 4)

theorem point_on_xOz_plane : point.snd = 0 :=
by 
  -- Additional definitions and conditions might be necessary,
  -- but they should come directly from the problem statement:
  -- * Define conditions for being on the xOz plane.
  -- For the purpose of this example, we skip the proof.
  sorry

end point_on_xOz_plane_l258_258551


namespace car_travel_speed_l258_258109

theorem car_travel_speed (v : ℝ) : 
  (1 / 60) * 3600 + 5 = (1 / v) * 3600 → v = 65 := 
by
  intros h
  sorry

end car_travel_speed_l258_258109


namespace no_prime_sum_10003_l258_258442

theorem no_prime_sum_10003 :
  ¬ ∃ (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ p₁ + p₂ = 10003 :=
begin
  sorry
end

end no_prime_sum_10003_l258_258442


namespace ways_to_write_10003_as_sum_of_two_primes_l258_258419

theorem ways_to_write_10003_as_sum_of_two_primes : 
  (how_many_ways (n : ℕ) (is_prime n) (exists p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n)) 10003 = 0 :=
by
  sorry

end ways_to_write_10003_as_sum_of_two_primes_l258_258419


namespace ellipse_slope_product_l258_258022

variables {a b x1 y1 x2 y2 : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) ∧ (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2))

theorem ellipse_slope_product : 
  (a > b) → (b > 0) → (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) → 
  (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2) → 
  ( (y1 + y2)/(x1 + x2) ) * ( (y1 - y2)/(x1 - x2) ) = - (b^2 / a^2) :=
by
  intros ha hb hxy1 hxy2
  sorry

end ellipse_slope_product_l258_258022


namespace solution_l258_258943

-- Define the linear equations and their solutions
def system_of_equations (x y : ℕ) :=
  3 * x + y = 500 ∧ x + 2 * y = 250

-- Define the budget constraint
def budget_constraint (m : ℕ) :=
  150 * m + 50 * (25 - m) ≤ 2700

-- Define the purchasing plans and costs
def purchasing_plans (m n : ℕ) :=
  (m = 12 ∧ n = 13 ∧ 150 * m + 50 * n = 2450) ∨ 
  (m = 13 ∧ n = 12 ∧ 150 * m + 50 * n = 2550) ∨ 
  (m = 14 ∧ n = 11 ∧ 150 * m + 50 * n = 2650)

-- Define the Lean statement
theorem solution :
  (∃ x y, system_of_equations x y ∧ x = 150 ∧ y = 50) ∧
  (∃ m, budget_constraint m ∧ m ≤ 14) ∧
  (∃ m n, 12 ≤ m ∧ m ≤ 14 ∧ m + n = 25 ∧ purchasing_plans m n ∧ 150 * m + 50 * n = 2450) :=
sorry

end solution_l258_258943


namespace find_x_eq_neg15_l258_258260

theorem find_x_eq_neg15 :
  ∃ x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ↔ (x = -15) :=
by
  sorry

end find_x_eq_neg15_l258_258260


namespace consecutive_odd_natural_numbers_sum_l258_258779

theorem consecutive_odd_natural_numbers_sum (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : b = a + 6) 
  (h4 : c = a + 12) 
  (h5 : c = 27) 
  (h6 : a % 2 = 1) 
  (h7 : b % 2 = 1) 
  (h8 : c % 2 = 1) 
  (h9 : a % 3 = 0) 
  (h10 : b % 3 = 0) 
  (h11 : c % 3 = 0) 
  : a + b + c = 63 :=
by
  sorry

end consecutive_odd_natural_numbers_sum_l258_258779


namespace new_ratio_is_one_half_l258_258217

theorem new_ratio_is_one_half (x : ℕ) (y : ℕ) (h1 : y = 4 * x) (h2 : y = 48) :
  (x + 12) / y = 1 / 2 :=
by
  sorry

end new_ratio_is_one_half_l258_258217


namespace last_three_digits_7_pow_105_l258_258379

theorem last_three_digits_7_pow_105 : (7^105) % 1000 = 783 :=
  sorry

end last_three_digits_7_pow_105_l258_258379


namespace product_abc_l258_258317

theorem product_abc 
  (a b c : ℝ)
  (h1 : a + b + c = 1) 
  (h2 : 3 * (4 * a + 2 * b + c) = 15) 
  (h3 : 5 * (9 * a + 3 * b + c) = 65) :
  a * b * c = -4 :=
by
  sorry

end product_abc_l258_258317


namespace children_got_off_l258_258518

theorem children_got_off {x : ℕ} 
  (initial_children : ℕ := 22)
  (children_got_on : ℕ := 40)
  (children_left : ℕ := 2)
  (equation : initial_children + children_got_on - x = children_left) :
  x = 60 :=
sorry

end children_got_off_l258_258518


namespace tank_capacity_l258_258355

theorem tank_capacity (C : ℝ) :
  (C / 10 - 960 = C / 18) → C = 21600 := by
  intro h
  sorry

end tank_capacity_l258_258355


namespace part1_part2_l258_258692

variable (α : Real)
-- Condition
axiom tan_neg_alpha : Real.tan (-α) = -2

-- Question 1
theorem part1 : ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α)) = 3 := 
by
  sorry

-- Question 2
theorem part2 : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end part1_part2_l258_258692


namespace matrix_addition_correct_l258_258258

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 4, -2], ![5, -3, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![ -3,  2, -4], ![ 1, -6,  3], ![-2,  4,  0]]

def expectedSum : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-1,  1, -1], ![ 1, -2,  1], ![ 3,  1,  1]]

theorem matrix_addition_correct :
  A + B = expectedSum := by
  sorry

end matrix_addition_correct_l258_258258


namespace find_angle_B_l258_258416

theorem find_angle_B
  (a : ℝ) (c : ℝ) (A B C : ℝ)
  (h1 : a = 5 * Real.sqrt 2)
  (h2 : c = 10)
  (h3 : A = π / 6) -- 30 degrees in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  : B = 7 * π / 12 ∨ B = π / 12 := -- 105 degrees or 15 degrees in radians
sorry

end find_angle_B_l258_258416


namespace find_angle_F_l258_258453

-- Define the given conditions and the goal
variable (EF GH : ℝ) (angleE angleF angleG angleH : ℝ)
variable (h1 : EF ∥ GH) (h2 : angleE = 3 * angleH) (h3 : angleG = 2 * angleF) 

theorem find_angle_F (h_sum : angleF + angleG = 180) : angleF = 60 :=
by sorry

end find_angle_F_l258_258453


namespace smallest_prime_with_digit_sum_23_l258_258082

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l258_258082


namespace sum_of_primes_10003_l258_258441

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l258_258441


namespace residue_11_pow_2016_mod_19_l258_258056

theorem residue_11_pow_2016_mod_19 : (11^2016) % 19 = 17 := 
sorry

end residue_11_pow_2016_mod_19_l258_258056


namespace at_least_one_greater_than_zero_l258_258311

noncomputable def a (x : ℝ) : ℝ := x^2 - 2 * x + (Real.pi / 2)
noncomputable def b (y : ℝ) : ℝ := y^2 - 2 * y + (Real.pi / 2)
noncomputable def c (z : ℝ) : ℝ := z^2 - 2 * z + (Real.pi / 2)

theorem at_least_one_greater_than_zero (x y z : ℝ) : (a x > 0) ∨ (b y > 0) ∨ (c z > 0) :=
by sorry

end at_least_one_greater_than_zero_l258_258311


namespace contrapositive_proof_l258_258487

theorem contrapositive_proof (a b : ℕ) : (a = 1 ∧ b = 2) → (a + b = 3) :=
by {
  sorry
}

end contrapositive_proof_l258_258487


namespace minimize_y_l258_258249

noncomputable def y (x a b k : ℝ) : ℝ :=
  (x - a) ^ 2 + (x - b) ^ 2 + k * x

theorem minimize_y (a b k : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b k ≤ y x' a b k) ∧ x = (a + b - k / 2) / 2 :=
by
  have x := (a + b - k / 2) / 2
  use x
  sorry

end minimize_y_l258_258249


namespace warehouse_box_storage_l258_258124

theorem warehouse_box_storage (S : ℝ) (h1 : (3 - 1/4) * S = 55000) : (1/4) * S = 5000 :=
by
  sorry

end warehouse_box_storage_l258_258124


namespace smallest_prime_with_digit_sum_23_l258_258085

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end smallest_prime_with_digit_sum_23_l258_258085


namespace sin_double_angle_l258_258574

open Real

theorem sin_double_angle (α : ℝ) (h : tan α = -3/5) : sin (2 * α) = -15/17 :=
by
  -- We are skipping the proof here
  sorry

end sin_double_angle_l258_258574


namespace annies_initial_amount_l258_258669

theorem annies_initial_amount :
  let hamburger_cost := 4
  let cheeseburger_cost := 5
  let french_fries_cost := 3
  let milkshake_cost := 5
  let smoothie_cost := 6
  let people_count := 8
  let burger_discount := 1
  let milkshake_discount := 2
  let smoothie_discount_buy2_get1free := 6
  let sales_tax := 0.08
  let tip_rate := 0.15
  let max_single_person_cost := cheeseburger_cost + french_fries_cost + smoothie_cost
  let total_cost := people_count * max_single_person_cost
  let total_burger_discount := people_count * burger_discount
  let total_milkshake_discount := 4 * milkshake_discount
  let total_smoothie_discount := smoothie_discount_buy2_get1free
  let total_discount := total_burger_discount + total_milkshake_discount + total_smoothie_discount
  let discounted_cost := total_cost - total_discount
  let tax_amount := discounted_cost * sales_tax
  let subtotal_with_tax := discounted_cost + tax_amount
  let original_total_cost := people_count * max_single_person_cost
  let tip_amount := original_total_cost * tip_rate
  let final_amount := subtotal_with_tax + tip_amount
  let annie_has_left := 30
  let annies_initial_money := final_amount + annie_has_left
  annies_initial_money = 144 :=
by
  sorry

end annies_initial_amount_l258_258669


namespace baseball_card_decrease_l258_258227

theorem baseball_card_decrease (V : ℝ) (hV : V > 0) (x : ℝ) :
  (1 - x / 100) * (1 - 0.30) = 1 - 0.44 -> x = 20 :=
by {
  -- proof omitted 
  sorry
}

end baseball_card_decrease_l258_258227


namespace total_typing_cost_l258_258339

def typingCost (totalPages revisedOncePages revisedTwicePages : ℕ) (firstTimeCost revisionCost : ℕ) : ℕ := 
  let initialCost := totalPages * firstTimeCost
  let firstRevisionCost := revisedOncePages * revisionCost
  let secondRevisionCost := revisedTwicePages * (revisionCost * 2)
  initialCost + firstRevisionCost + secondRevisionCost

theorem total_typing_cost : typingCost 200 80 20 5 3 = 1360 := 
  by 
    rfl

end total_typing_cost_l258_258339


namespace Norm_photo_count_l258_258108

variables (L M N : ℕ)

-- Conditions from the problem
def cond1 : Prop := L = N - 60
def cond2 : Prop := N = 2 * L + 10

-- Given the conditions, prove N = 110
theorem Norm_photo_count (h1 : cond1 L N) (h2 : cond2 L N) : N = 110 :=
by
  sorry

end Norm_photo_count_l258_258108


namespace alice_minimum_speed_l258_258489

noncomputable def minimum_speed_to_exceed (d t_bob t_alice : ℝ) (v_bob : ℝ) : ℝ :=
  d / t_alice

theorem alice_minimum_speed (d : ℝ) (v_bob : ℝ) (t_lag : ℝ) (v_alice : ℝ) :
  d = 30 → v_bob = 40 → t_lag = 0.5 → v_alice = d / (d / v_bob - t_lag) → v_alice > 60 :=
by
  intros hd hv hb ht
  rw [hd, hv, hb] at ht
  simp at ht
  sorry

end alice_minimum_speed_l258_258489


namespace binom_7_4_eq_35_l258_258535

theorem binom_7_4_eq_35 : nat.choose 7 4 = 35 := by
  sorry

end binom_7_4_eq_35_l258_258535


namespace slope_of_intersection_points_l258_258563

theorem slope_of_intersection_points {s x y : ℝ} 
  (h1 : 2 * x - 3 * y = 6 * s - 5) 
  (h2 : 3 * x + y = 9 * s + 4) : 
  ∃ m : ℝ, m = 3 ∧ (∀ s : ℝ, (∃ x y : ℝ, 2 * x - 3 * y = 6 * s - 5 ∧ 3 * x + y = 9 * s + 4) → y = m * x + (23/11)) := 
by
  sorry

end slope_of_intersection_points_l258_258563


namespace day_after_75_days_l258_258896

theorem day_after_75_days (day_of_week : ℕ → String) (h : day_of_week 0 = "Tuesday") :
  day_of_week 75 = "Sunday" :=
sorry

end day_after_75_days_l258_258896


namespace share_of_y_is_63_l258_258113

theorem share_of_y_is_63 (x y z : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : x + y + z = 273) : y = 63 :=
by
  -- The proof will go here
  sorry

end share_of_y_is_63_l258_258113


namespace total_grocery_bill_l258_258754

theorem total_grocery_bill
    (hamburger_meat_cost : ℝ := 5.00)
    (crackers_cost : ℝ := 3.50)
    (frozen_vegetables_bags : ℝ := 4)
    (frozen_vegetables_cost_per_bag : ℝ := 2.00)
    (cheese_cost : ℝ := 3.50)
    (discount_rate : ℝ := 0.10) :
    let total_cost_before_discount := hamburger_meat_cost + crackers_cost + (frozen_vegetables_bags * frozen_vegetables_cost_per_bag) + cheese_cost
    let discount := total_cost_before_discount * discount_rate
    let total_cost_after_discount := total_cost_before_discount - discount
in
total_cost_after_discount = 18.00 :=
by
   -- total_cost_before_discount = 5.00 + 3.50 + (4 * 2.00) + 3.50 = 20.00
   -- discount = 20.00 * 0.10 = 2.00
   -- total_cost_after_discount = 20.00 - 2.00 = 18.00
   sorry

end total_grocery_bill_l258_258754


namespace div_sqrt_81_by_3_is_3_l258_258098

-- Definitions based on conditions
def sqrt_81 := Nat.sqrt 81
def number_3 := 3

-- Problem statement
theorem div_sqrt_81_by_3_is_3 : sqrt_81 / number_3 = 3 := by
  sorry

end div_sqrt_81_by_3_is_3_l258_258098


namespace inequality_holds_l258_258322

theorem inequality_holds (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  Real.sin (2 * x)^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
sorry

end inequality_holds_l258_258322


namespace polynomial_factors_l258_258468

theorem polynomial_factors (x : ℝ) : 
  (x^4 - 4*x^2 + 4) = (x^2 - 2*x + 2) * (x^2 + 2*x + 2) :=
by
  sorry

end polynomial_factors_l258_258468


namespace dice_prime_product_probability_l258_258214

theorem dice_prime_product_probability :
  let outcomes := finset.pi finset.univ (λ _, finset.range 6.succ),
      prime_products := {x ∈ outcomes | nat.prime (x.1 * x.2 * x.3)},
      favorable_outcomes := finset.card prime_products,
      total_outcomes := finset.card outcomes in
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 24) :=
by sorry

end dice_prime_product_probability_l258_258214


namespace distance_centers_triangle_l258_258239

noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let circumradius := (a * b * c) / (4 * K)
  let hypotenuse := by
    by_cases hc : a * a + b * b = c * c
    exact c
    by_cases hb : a * a + c * c = b * b
    exact b
    by_cases ha : b * b + c * c = a * a
    exact a
    exact 0
  let oc := hypotenuse / 2
  Real.sqrt (oc * oc + r * r)

theorem distance_centers_triangle :
  distance_between_centers 7 24 25 = Real.sqrt 165.25 := sorry

end distance_centers_triangle_l258_258239


namespace find_a2_b2_l258_258567

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_a2_b2 (a b : ℝ) (h1 : (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit) : a^2 + b^2 = 5 :=
  sorry

end find_a2_b2_l258_258567


namespace pq_implications_l258_258168

theorem pq_implications (p q : Prop) (hpq_or : p ∨ q) (hpq_and : p ∧ q) : p ∧ q :=
by
  sorry

end pq_implications_l258_258168


namespace simplify_expression_l258_258626

-- Define the initial expression
def expr (q : ℚ) := (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q)

-- Define the simplified expression
def simplified_expr (q : ℚ) := 5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5

-- The theorem stating that the two expressions are equal
theorem simplify_expression (q : ℚ) : expr q = simplified_expr q :=
by
  sorry

end simplify_expression_l258_258626


namespace probability_product_multiple_of_4_l258_258265

/--
Geoff and Trevor each roll a fair eight-sided die (numbered 1 through 8).
Prove that the probability that the product of the numbers they roll is a multiple of 4 is 15/16.
-/
theorem probability_product_multiple_of_4 :
  let outcomes := { (d1, d2) | d1 ∈ Finset.range 1 9 ∧ d2 ∈ Finset.range 1 9 },
      multiples_of_4 := { (d1, d2) | d1 * d2 % 4 = 0 },
      favorable := Finset.card multiples_of_4
  let total := Finset.card outcomes
  in (favorable.toRat / total.toRat) = 15/16 :=
by
  sorry

end probability_product_multiple_of_4_l258_258265


namespace find_x_l258_258714

theorem find_x (x : ℝ) : (0.75 / x = 10 / 8) → (x = 0.6) := by
  sorry

end find_x_l258_258714


namespace supplement_of_angle_l258_258285

theorem supplement_of_angle (θ : ℝ) 
  (h_complement: θ = 90 - 30) : 180 - θ = 120 :=
by
  sorry

end supplement_of_angle_l258_258285


namespace interest_rate_for_first_part_l258_258664

def sum_amount : ℝ := 2704
def part2 : ℝ := 1664
def part1 : ℝ := sum_amount - part2
def rate2 : ℝ := 0.05
def years2 : ℝ := 3
def interest2 : ℝ := part2 * rate2 * years2
def years1 : ℝ := 8

theorem interest_rate_for_first_part (r1 : ℝ) :
  part1 * r1 * years1 = interest2 → r1 = 0.03 :=
by
  sorry

end interest_rate_for_first_part_l258_258664


namespace inequality_and_equality_hold_l258_258892

theorem inequality_and_equality_hold (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ a = b) :=
sorry

end inequality_and_equality_hold_l258_258892


namespace abs_neg_2023_l258_258484

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 :=
by
  sorry

end abs_neg_2023_l258_258484


namespace probability_sum_3_or_7_or_10_l258_258945

-- Definitions of the faces of each die
def die_1_faces : List ℕ := [1, 2, 2, 5, 5, 6]
def die_2_faces : List ℕ := [1, 2, 4, 4, 5, 6]

-- Probability of a sum being 3 (valid_pairs: (1, 2))
def probability_sum_3 : ℚ :=
  (1 / 6) * (1 / 6)

-- Probability of a sum being 7 (valid pairs: (1, 6), (2, 5))
def probability_sum_7 : ℚ :=
  ((1 / 6) * (1 / 6)) + ((1 / 3) * (1 / 6))

-- Probability of a sum being 10 (valid pairs: (5, 5))
def probability_sum_10 : ℚ :=
  (1 / 3) * (1 / 6)

-- Total probability for sums being 3, 7, or 10
def total_probability : ℚ :=
  probability_sum_3 + probability_sum_7 + probability_sum_10

-- The proof statement
theorem probability_sum_3_or_7_or_10 : total_probability = 1 / 6 :=
  sorry

end probability_sum_3_or_7_or_10_l258_258945


namespace binomial_7_4_equals_35_l258_258541

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Problem statement: Prove that binomial(7, 4) = 35
theorem binomial_7_4_equals_35 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_equals_35_l258_258541


namespace algebraic_expression_eq_five_l258_258850

theorem algebraic_expression_eq_five (a b : ℝ)
  (h₁ : a^2 - a = 1)
  (h₂ : b^2 - b = 1) :
  3 * a^2 + 2 * b^2 - 3 * a - 2 * b = 5 :=
by
  sorry

end algebraic_expression_eq_five_l258_258850


namespace average_paper_tape_length_l258_258236

-- Define the lengths of the paper tapes as given in the conditions
def red_tape_length : ℝ := 20
def purple_tape_length : ℝ := 16

-- State the proof problem
theorem average_paper_tape_length : 
  (red_tape_length + purple_tape_length) / 2 = 18 := 
by
  sorry

end average_paper_tape_length_l258_258236


namespace Q_proper_subset_P_l258_258400

open Set

def P : Set ℝ := { x | x ≥ 1 }
def Q : Set ℝ := { 2, 3 }

theorem Q_proper_subset_P : Q ⊂ P :=
by
  sorry

end Q_proper_subset_P_l258_258400


namespace largest_three_digit_multiple_of_12_and_sum_of_digits_24_l258_258053

def sum_of_digits (n : ℕ) : ℕ :=
  ((n / 100) + ((n / 10) % 10) + (n % 10))

def is_multiple_of_12 (n : ℕ) : Prop :=
  n % 12 = 0

def largest_three_digit_multiple_of_12_with_digits_sum_24 : ℕ :=
  996

theorem largest_three_digit_multiple_of_12_and_sum_of_digits_24 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ sum_of_digits n = 24 ∧ is_multiple_of_12 n ∧ n = largest_three_digit_multiple_of_12_with_digits_sum_24 :=
by 
  sorry

end largest_three_digit_multiple_of_12_and_sum_of_digits_24_l258_258053


namespace largest_possible_cupcakes_without_any_ingredients_is_zero_l258_258181

-- Definitions of properties of the cupcakes
def total_cupcakes : ℕ := 60
def blueberries (n : ℕ) : Prop := n = total_cupcakes / 3
def sprinkles (n : ℕ) : Prop := n = total_cupcakes / 4
def frosting (n : ℕ) : Prop := n = total_cupcakes / 2
def pecans (n : ℕ) : Prop := n = total_cupcakes / 5

-- Theorem statement
theorem largest_possible_cupcakes_without_any_ingredients_is_zero :
  ∃ n, blueberries n ∧ sprinkles n ∧ frosting n ∧ pecans n → n = 0 := 
sorry

end largest_possible_cupcakes_without_any_ingredients_is_zero_l258_258181


namespace min_a_b_l258_258820

theorem min_a_b (a b : ℕ) (h1 : 43 * a + 17 * b = 731) (h2 : a ≤ 17) (h3 : b ≤ 43) : a + b = 17 :=
by
  sorry

end min_a_b_l258_258820


namespace no_prime_sum_10003_l258_258424

theorem no_prime_sum_10003 : 
  ∀ p q : Nat, Nat.Prime p → Nat.Prime q → p + q = 10003 → False :=
by sorry

end no_prime_sum_10003_l258_258424


namespace range_of_a_l258_258739

open Set

variable {a x : ℝ}

def A (a : ℝ) : Set ℝ := {x | abs (x - a) < 1}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem range_of_a (h : A a ∩ B = ∅) : a ≤ 0 ∨ a ≥ 6 := 
by 
  sorry

end range_of_a_l258_258739


namespace combination_seven_four_l258_258538

theorem combination_seven_four : nat.choose 7 4 = 35 :=
by
  sorry

end combination_seven_four_l258_258538


namespace equivalent_region_l258_258402

def satisfies_conditions (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 2 ∧ -1 ≤ x / (x + y) ∧ x / (x + y) ≤ 1

def region (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≥ -2*x ∧ x^2 + y^2 ≤ 2

theorem equivalent_region (x y : ℝ) :
  satisfies_conditions x y = region x y := 
sorry

end equivalent_region_l258_258402


namespace max_marked_points_l258_258623

theorem max_marked_points (segments : ℕ) (ratio : ℚ) (h_segments : segments = 10) (h_ratio : ratio = 3 / 4) : 
  ∃ n, n ≤ (segments * 2 / 2) ∧ n = 10 :=
by
  sorry

end max_marked_points_l258_258623


namespace tangent_line_eq_l258_258492

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 - x + 1) (h_point : (x, y) = (0, 1)) : x + y - 1 = 0 := 
sorry

end tangent_line_eq_l258_258492


namespace sqrt_meaningful_l258_258795

theorem sqrt_meaningful (x : ℝ) : x + 1 >= 0 ↔ (∃ y : ℝ, y * y = x + 1) := by
  sorry

end sqrt_meaningful_l258_258795


namespace gcd_largest_value_l258_258922

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l258_258922


namespace not_sum_of_squares_or_cubes_in_ap_l258_258377

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a + b * b = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a * a + b * b * b = n

def arithmetic_progression (a d k : ℕ) : ℕ :=
  a + d * k

theorem not_sum_of_squares_or_cubes_in_ap :
  ∀ k : ℕ, ¬ is_sum_of_two_squares (arithmetic_progression 31 36 k) ∧
           ¬ is_sum_of_two_cubes (arithmetic_progression 31 36 k) := by
  sorry

end not_sum_of_squares_or_cubes_in_ap_l258_258377


namespace find_xyz_l258_258575

def divisible_by (n k : ℕ) : Prop := k % n = 0

def is_7_digit_number (a b c d e f g : ℕ) : ℕ := 
  10^6 * a + 10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + g

theorem find_xyz
  (x y z : ℕ)
  (h : divisible_by 792 (is_7_digit_number 1 4 x y 7 8 z))
  : (100 * x + 10 * y + z) = 644 :=
by
  sorry

end find_xyz_l258_258575


namespace rationalize_denominator_l258_258019

theorem rationalize_denominator :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) := by
  sorry

end rationalize_denominator_l258_258019


namespace total_stops_is_seven_l258_258233

-- Definitions of conditions
def initial_stops : ℕ := 3
def additional_stops : ℕ := 4

-- Statement to be proved
theorem total_stops_is_seven : initial_stops + additional_stops = 7 :=
by {
  -- this is a placeholder for the proof
  sorry
}

end total_stops_is_seven_l258_258233


namespace fran_speed_l258_258603

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l258_258603


namespace tangent_line_at_1_tangent_line_through_2_3_l258_258157

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2

-- Problem 1: Prove that the tangent line at point (1, 1) is y = 3x - 2
theorem tangent_line_at_1 (x y : ℝ) (h : y = f 1 + f' 1 * (x - 1)) : y = 3 * x - 2 := 
sorry

-- Problem 2: Prove that the tangent line passing through (2/3, 0) is either y = 0 or y = 3x - 2
theorem tangent_line_through_2_3 (x y x0 : ℝ) 
  (hx0 : y = f x0 + f' x0 * (x - x0))
  (hp : 0 = f' x0 * (2/3 - x0)) :
  y = 0 ∨ y = 3 * x - 2 := 
sorry

end tangent_line_at_1_tangent_line_through_2_3_l258_258157


namespace value_of_expr_l258_258005

noncomputable def verify_inequality (x a b c : ℝ) : Prop :=
  (x - a) * (x - b) / (x - c) ≥ 0

theorem value_of_expr (a b c : ℝ) :
  (∀ x : ℝ, verify_inequality x a b c ↔ (x < -6 ∨ abs (x - 30) ≤ 2)) →
  a < b →
  a = 28 →
  b = 32 →
  c = -6 →
  a + 2 * b + 3 * c = 74 := by
  sorry

end value_of_expr_l258_258005


namespace no_such_m_for_equivalence_existence_of_m_for_implication_l258_258568

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem no_such_m_for_equivalence :
  ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S x m :=
sorry

theorem existence_of_m_for_implication :
  ∃ m : ℝ, (∀ x : ℝ, S x m → P x) ∧ m ≤ 3 :=
sorry

end no_such_m_for_equivalence_existence_of_m_for_implication_l258_258568


namespace part1_monotonicity_part2_intersection_l258_258158

noncomputable def f (a x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

theorem part1_monotonicity (a : ℝ) : 
  ∃ interval : Set ℝ, 
    (∀ x ∈ interval, ∃ interval' : Set ℝ, 
      (∀ x' ∈ interval', f a x' ≤ f a x) ∧ 
      (∀ x' ∈ Set.univ \ interval', f a x' > f a x)) :=
sorry

theorem part2_intersection (a b x_1 x_2 : ℝ) (h1 : a > 0) (h2 : b ≠ 0)
  (h3 : f a x_1 = -b * Real.exp 1) (h4 : f a x_2 = -b * Real.exp 1)
  (h5 : x_1 ≠ x_2) : 
  - (1 / Real.exp 1) < a * b ∧ a * b < 0 ∧ a * (x_1 + x_2) < -2 :=
sorry

end part1_monotonicity_part2_intersection_l258_258158


namespace ratio_part_to_whole_l258_258898

/-- One part of one third of two fifth of a number is 17, and 40% of that number is 204. 
Prove that the ratio of the part to the whole number is 1:30. -/
theorem ratio_part_to_whole 
  (N : ℝ)
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 17) 
  (h2 : 0.40 * N = 204) : 
  17 / N = 1 / 30 :=
  sorry

end ratio_part_to_whole_l258_258898


namespace joans_remaining_kittens_l258_258889

theorem joans_remaining_kittens (initial_kittens given_away : ℕ) (h1 : initial_kittens = 15) (h2 : given_away = 7) : initial_kittens - given_away = 8 := sorry

end joans_remaining_kittens_l258_258889


namespace largest_n_base_conditions_l258_258925

theorem largest_n_base_conditions :
  ∃ n: ℕ, n < 10000 ∧ 
  (∃ a: ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b: ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c: ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1)) ∧
  n = 4369 :=
sorry

end largest_n_base_conditions_l258_258925


namespace find_initial_lion_population_l258_258783

-- Define the conditions as integers
def lion_cubs_per_month : ℕ := 5
def lions_die_per_month : ℕ := 1
def total_lions_after_one_year : ℕ := 148

-- Define a formula for calculating the initial number of lions
def initial_number_of_lions (net_increase : ℕ) (final_count : ℕ) (months : ℕ) : ℕ :=
  final_count - (net_increase * months)

-- Main theorem statement
theorem find_initial_lion_population : initial_number_of_lions (lion_cubs_per_month - lions_die_per_month) total_lions_after_one_year 12 = 100 :=
  sorry

end find_initial_lion_population_l258_258783


namespace probability_relationship_l258_258047

def total_outcomes : ℕ := 36

def P1 : ℚ := 1 / total_outcomes
def P2 : ℚ := 2 / total_outcomes
def P3 : ℚ := 3 / total_outcomes

theorem probability_relationship :
  P1 < P2 ∧ P2 < P3 :=
by
  sorry

end probability_relationship_l258_258047


namespace sum_of_digits_inequality_l258_258149

-- Assume that S(x) represents the sum of the digits of x in its decimal representation.
axiom sum_of_digits (x : ℕ) : ℕ

-- Given condition: for any natural numbers a and b, the sum of digits function satisfies the inequality
axiom sum_of_digits_add (a b : ℕ) : sum_of_digits (a + b) ≤ sum_of_digits a + sum_of_digits b

-- Theorem statement we want to prove
theorem sum_of_digits_inequality (k : ℕ) : sum_of_digits k ≤ 8 * sum_of_digits (8 * k) := 
  sorry

end sum_of_digits_inequality_l258_258149


namespace sum_of_dice_less_than_10_probability_l258_258946

/-
  Given:
  - A fair die with faces labeled 1, 2, 3, 4, 5, 6.
  - The die is rolled twice.

  Prove that the probability that the sum of the face values is less than 10 is 5/6.
-/

noncomputable def probability_sum_less_than_10 : ℚ :=
  let total_outcomes := 36
  let favorable_outcomes := 30
  favorable_outcomes / total_outcomes

theorem sum_of_dice_less_than_10_probability :
  probability_sum_less_than_10 = 5 / 6 :=
by
  sorry

end sum_of_dice_less_than_10_probability_l258_258946


namespace tangent_line_equation_l258_258983

noncomputable def circle_eq1 (x y : ℝ) := x^2 + (y - 2)^2 - 4
noncomputable def circle_eq2 (x y : ℝ) := (x - 3)^2 + (y + 2)^2 - 21
noncomputable def line_eq (x y : ℝ) := 3*x - 4*y - 4

theorem tangent_line_equation :
  ∀ (x y : ℝ), (circle_eq1 x y = 0 ∧ circle_eq2 x y = 0) ↔ line_eq x y = 0 :=
sorry

end tangent_line_equation_l258_258983


namespace remove_one_piece_l258_258271

theorem remove_one_piece (pieces : Finset (Fin 8 × Fin 8)) (h_card : pieces.card = 15)
  (h_row : ∀ r : Fin 8, ∃ c, (r, c) ∈ pieces)
  (h_col : ∀ c : Fin 8, ∃ r, (r, c) ∈ pieces) :
  ∃ pieces' : Finset (Fin 8 × Fin 8), pieces'.card = 14 ∧ 
  (∀ r : Fin 8, ∃ c, (r, c) ∈ pieces') ∧ 
  (∀ c : Fin 8, ∃ r, (r, c) ∈ pieces') :=
sorry

end remove_one_piece_l258_258271


namespace smallest_prime_with_digit_sum_23_l258_258083

/-- The sum of the digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

/-- 1997 is the smallest prime number whose digits sum to 23 -/
theorem smallest_prime_with_digit_sum_23 : 
  ∃ p : ℕ, p.prime ∧ digit_sum p = 23 ∧ ∀ q : ℕ, q.prime ∧ digit_sum q = 23 → q ≥ p :=
sorry

end smallest_prime_with_digit_sum_23_l258_258083


namespace people_in_room_l258_258280

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end people_in_room_l258_258280


namespace count_integer_values_l258_258687

-- Statement of the problem in Lean 4
theorem count_integer_values (x : ℤ) : 
  (7 * x^2 + 23 * x + 20 ≤ 30) → 
  ∃ (n : ℕ), n = 6 :=
sorry

end count_integer_values_l258_258687


namespace angle_B_is_30_degrees_l258_258881

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Assuming the conditions given in the problem
variables (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) 
          (h2 : a > b)

-- The proof to establish the measure of angle B as 30 degrees
theorem angle_B_is_30_degrees (h1 : a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 0.5 * b) (h2 : a > b) : B = Real.pi / 6 :=
sorry

end angle_B_is_30_degrees_l258_258881


namespace largest_possible_value_l258_258986

-- Definitions for the conditions
def lower_x_bound := -4
def upper_x_bound := -2
def lower_y_bound := 2
def upper_y_bound := 4

-- The proposition to prove
theorem largest_possible_value (x y : ℝ) 
    (h1 : lower_x_bound ≤ x) (h2 : x ≤ upper_x_bound)
    (h3 : lower_y_bound ≤ y) (h4 : y ≤ upper_y_bound) :
    ∃ v, v = (x + y) / x ∧ ∀ (w : ℝ), w = (x + y) / x → w ≤ 1/2 :=
by
  sorry

end largest_possible_value_l258_258986


namespace comparison_of_a_b_c_l258_258268

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

end comparison_of_a_b_c_l258_258268


namespace product_is_solution_quotient_is_solution_l258_258312

-- Definitions and conditions from the problem statement
variable (a b c d : ℤ)

-- The conditions
axiom h1 : a^2 - 5 * b^2 = 1
axiom h2 : c^2 - 5 * d^2 = 1

-- Lean 4 statement for the first part: the product
theorem product_is_solution :
  ∃ (m n : ℤ), ((m + n * (5:ℚ)) = (a + b * (5:ℚ)) * (c + d * (5:ℚ))) ∧ (m^2 - 5 * n^2 = 1) :=
sorry

-- Lean 4 statement for the second part: the quotient
theorem quotient_is_solution :
  ∃ (p q : ℤ), ((p + q * (5:ℚ)) = (a + b * (5:ℚ)) / (c + d * (5:ℚ))) ∧ (p^2 - 5 * q^2 = 1) :=
sorry

end product_is_solution_quotient_is_solution_l258_258312


namespace value_of_f_2_plus_g_3_l258_258184

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 - 1

theorem value_of_f_2_plus_g_3 : f (2 + g 3) = 26 :=
by
  sorry

end value_of_f_2_plus_g_3_l258_258184


namespace ball_hits_ground_l258_258032

theorem ball_hits_ground :
  ∃ (t : ℝ), (t = 2) ∧ (-4.9 * t^2 + 5.7 * t + 7 = 0) :=
sorry

end ball_hits_ground_l258_258032


namespace intersection_M_N_l258_258278

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := U \ complement_U_N

theorem intersection_M_N : M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
by
  sorry

end intersection_M_N_l258_258278


namespace largest_D_l258_258748

theorem largest_D (D : ℝ) : (∀ x y : ℝ, x^2 + 2 * y^2 + 3 ≥ D * (3 * x + 4 * y)) → D ≤ Real.sqrt (12 / 17) :=
by
  sorry

end largest_D_l258_258748


namespace no_prime_sum_10003_l258_258433

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l258_258433


namespace find_equations_of_lines_l258_258845

-- Define the given constants and conditions
def point_P := (2, 2)
def line_l1 (x y : ℝ) := 3 * x - 2 * y + 1 = 0
def line_l2 (x y : ℝ) := x + 3 * y + 4 = 0
def intersection_point := (-1, -1)
def slope_perpendicular_line := 3

-- The theorem that we need to prove
theorem find_equations_of_lines :
  (∀ k, k = 0 → line_l1 2 2 → (x = y ∨ x + y = 4)) ∧
  (line_l1 (-1) (-1) ∧ line_l2 (-1) (-1) →
   (3 * x - y + 2 = 0))
:=
sorry

end find_equations_of_lines_l258_258845


namespace slices_per_person_l258_258502

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end slices_per_person_l258_258502


namespace y_x_cubed_monotonic_increasing_l258_258635

theorem y_x_cubed_monotonic_increasing : 
  ∀ x1 x2 : ℝ, (x1 ≤ x2) → (x1^3 ≤ x2^3) :=
by
  intros x1 x2 h
  sorry

end y_x_cubed_monotonic_increasing_l258_258635


namespace sum_of_primes_10003_l258_258440

theorem sum_of_primes_10003 : ∃! (p₁ p₂ : ℕ), prime p₁ ∧ prime p₂ ∧ 10003 = p₁ + p₂ :=
sorry

end sum_of_primes_10003_l258_258440


namespace probability_of_h_and_l_l258_258897

-- Define the possible levels of service
inductive BusServiceLevel
| H -- high/good
| M -- medium/average
| L -- low/poor

open BusServiceLevel

-- Define the sequences of bus arrivals
def sequences := [
  [L, M, H],
  [L, H, M],
  [M, L, H],
  [M, H, L],
  [H, L, M],
  [H, M, L]
]

-- Define Mr. Zhang's bus-taking strategy
def takeBus (seq : List BusServiceLevel) : BusServiceLevel :=
  match seq with
  | [first, second, third] =>
    if second = H ∨ (first ≠ H ∧ second = M) then
      second
    else
      third
  | _ => L -- This case won't occur due to our predefined sequences

-- Calculate probabilities
theorem probability_of_h_and_l :
  let counts := (sequences.count (λ seq => takeBus seq = H), sequences.count (λ seq => takeBus seq = L))
  counts.1 = 3 ∧ counts.2 = 1 ∧ sequences.length = 6 →
  (counts.1.toRat / sequences.length.toRat = 1/2) ∧ (counts.2.toRat / sequences.length.toRat = 1/6) :=
by sorry

end probability_of_h_and_l_l258_258897


namespace range_of_a_l258_258703

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 / exp 1 ≤ x ∧ x ≤ exp 1 → a - x^2 = - (2 * log x)) →
  1 ≤ a ∧ a ≤ exp 1^2 - 2 :=
by
  sorry

end range_of_a_l258_258703


namespace triangle_perimeter_l258_258494

-- Definitions of the geometric problem conditions
def inscribed_circle_tangent (A B C P : Type) : Prop := sorry
def radius_of_inscribed_circle (r : ℕ) : Prop := r = 24
def segment_lengths (AP PB : ℕ) : Prop := AP = 25 ∧ PB = 29

-- Main theorem to prove the perimeter of the triangle ABC
theorem triangle_perimeter (A B C P : Type) (r AP PB : ℕ)
  (H1 : inscribed_circle_tangent A B C P)
  (H2 : radius_of_inscribed_circle r)
  (H3 : segment_lengths AP PB) :
  2 * (54 + 208.72) = 525.44 :=
  sorry

end triangle_perimeter_l258_258494


namespace g_value_at_100_l258_258207

-- Given function g and its property
theorem g_value_at_100 (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y →
  x * g y - y * g x = g (x^2 / y)) : g 100 = 0 :=
sorry

end g_value_at_100_l258_258207


namespace derivative_at_one_l258_258909

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one_l258_258909


namespace deepak_age_l258_258938

variable (A D : ℕ)

theorem deepak_age (h1 : A / D = 2 / 3) (h2 : A + 5 = 25) : D = 30 :=
sorry

end deepak_age_l258_258938


namespace line_through_point_l258_258224

-- Definitions for conditions
def point : (ℝ × ℝ) := (1, 2)

-- Function to check if a line equation holds for the given form 
def is_line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main Lean theorem statement
theorem line_through_point (a b c : ℝ) :
  (∃ a b c, (is_line_eq a b c 1 2) ∧ 
           ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 2 ∧ b = -1 ∧ c = 0))) :=
sorry

end line_through_point_l258_258224


namespace pentagon_perpendicular_sums_l258_258517

noncomputable def FO := 2
noncomputable def FQ := 2
noncomputable def FR := 2

theorem pentagon_perpendicular_sums :
  FO + FQ + FR = 6 :=
by
  sorry

end pentagon_perpendicular_sums_l258_258517


namespace range_of_m_for_false_proposition_l258_258159

theorem range_of_m_for_false_proposition :
  (∀ x ∈ (Set.Icc 0 (Real.pi / 4)), Real.tan x < m) → False ↔ m ≤ 1 :=
by
  sorry

end range_of_m_for_false_proposition_l258_258159


namespace prove_smallest_x_is_zero_l258_258929

noncomputable def smallest_x : ℝ :=
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in if h : f(x)^2 + f(x) = 20 then x else 0

theorem prove_smallest_x_is_zero :
  let f := λ (x : ℝ), (5 * x - 20) / (4 * x - 5)
  in ∃ x : ℝ, f(x)^2 + f(x) = 20 ∧ (∀ y : ℝ, f(y)^2 + f(y) = 20 → x ≤ y) :=
begin
  sorry, -- proof goes here
end

end prove_smallest_x_is_zero_l258_258929


namespace directrix_of_parabola_l258_258033

theorem directrix_of_parabola (x y : ℝ) : (y^2 = 8*x) → (x = -2) :=
by
  sorry

end directrix_of_parabola_l258_258033


namespace arithmetic_sequence_problem_l258_258292

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) 
  (a1 : a 1 = 1 / 3) 
  (a2_a5 : a 2 + a 5 = 4) 
  (an : ∃ n, a n = 33) :
  ∃ n, a n = 33 ∧ n = 50 := 
by 
  sorry

end arithmetic_sequence_problem_l258_258292


namespace mixture_contains_pecans_l258_258708

theorem mixture_contains_pecans 
  (price_per_cashew_per_pound : ℝ)
  (cashews_weight : ℝ)
  (price_per_mixture_per_pound : ℝ)
  (price_of_cashews : ℝ)
  (mixture_weight : ℝ)
  (pecans_weight : ℝ)
  (price_per_pecan_per_pound : ℝ)
  (pecans_price : ℝ)
  (total_cost_of_mixture : ℝ)
  
  (h1 : price_per_cashew_per_pound = 3.50) 
  (h2 : cashews_weight = 2)
  (h3 : price_per_mixture_per_pound = 4.34) 
  (h4 : pecans_weight = 1.33333333333)
  (h5 : price_per_pecan_per_pound = 5.60)
  
  (h6 : price_of_cashews = cashews_weight * price_per_cashew_per_pound)
  (h7 : mixture_weight = cashews_weight + pecans_weight)
  (h8 : pecans_price = pecans_weight * price_per_pecan_per_pound)
  (h9 : total_cost_of_mixture = price_of_cashews + pecans_price)

  (h10 : price_per_mixture_per_pound = total_cost_of_mixture / mixture_weight)
  
  : pecans_weight = 1.33333333333 :=
sorry

end mixture_contains_pecans_l258_258708


namespace unattainable_y_l258_258973

theorem unattainable_y (x : ℚ) (hx : x ≠ -4 / 3) : 
    ∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3 :=
sorry

end unattainable_y_l258_258973


namespace range_of_g_l258_258732

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x)^4 + (arcsin x)^4

theorem range_of_g : 
  ∀ x ∈ Icc (-1 : ℝ) 1, 
  g x ∈ set.Icc (π^4 / 16) (17 * π^4 / 16) :=
by 
  sorry

end range_of_g_l258_258732


namespace gcd_largest_value_l258_258921

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l258_258921


namespace count_two_digit_primes_with_given_conditions_l258_258139

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens + units = 9

def tens_greater_than_units (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens > units

theorem count_two_digit_primes_with_given_conditions :
  ∃ count : ℕ, count = 0 ∧ ∀ n, is_two_digit_prime n ∧ sum_of_digits_is_nine n ∧ tens_greater_than_units n → false :=
by
  -- proof goes here
  sorry

end count_two_digit_primes_with_given_conditions_l258_258139


namespace triangle_side_length_c_l258_258883

theorem triangle_side_length_c (a b : ℝ) (α β γ : ℝ) (h_angle_sum : α + β + γ = 180) (h_angle_eq : 3 * α + 2 * β = 180) (h_a : a = 2) (h_b : b = 3) : 
∃ c : ℝ, c = 4 :=
by
  sorry

end triangle_side_length_c_l258_258883


namespace fractions_sum_simplified_l258_258553

noncomputable def frac12over15 : ℚ := 12 / 15
noncomputable def frac7over9 : ℚ := 7 / 9
noncomputable def frac1and1over6 : ℚ := 1 + 1 / 6

theorem fractions_sum_simplified :
  frac12over15 + frac7over9 + frac1and1over6 = 247 / 90 :=
by
  -- This step will be left as a proof to complete.
  sorry

end fractions_sum_simplified_l258_258553


namespace polynomial_remainder_division_l258_258557

theorem polynomial_remainder_division :
  ∀ (x : ℝ), (3 * x^7 + 2 * x^5 - 5 * x^3 + x^2 - 9) % (x^2 + 2 * x + 1) = 14 * x - 16 :=
by
  sorry

end polynomial_remainder_division_l258_258557


namespace attendees_not_from_companies_l258_258521

theorem attendees_not_from_companies :
  let A := 30 
  let B := 2 * A
  let C := A + 10
  let D := C - 5
  let T := 185 
  T - (A + B + C + D) = 20 :=
by
  sorry

end attendees_not_from_companies_l258_258521


namespace average_carnations_l258_258347

theorem average_carnations (c1 c2 c3 n : ℕ) (h1 : c1 = 9) (h2 : c2 = 14) (h3 : c3 = 13) (h4 : n = 3) :
  (c1 + c2 + c3) / n = 12 :=
by
  sorry

end average_carnations_l258_258347


namespace speed_conversion_l258_258359

theorem speed_conversion (speed_kmph : ℝ) (h : speed_kmph = 18) : speed_kmph * (1000 / 3600) = 5 := by
  sorry

end speed_conversion_l258_258359


namespace union_A_B_eq_intersection_A_B_complement_eq_l258_258860

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x ≤ 0}
def B_complement : Set ℝ := {x | x < 0 ∨ x > 4}

theorem union_A_B_eq : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} := by
  sorry

theorem intersection_A_B_complement_eq : A ∩ B_complement = {x | -1 ≤ x ∧ x < 0} := by
  sorry

end union_A_B_eq_intersection_A_B_complement_eq_l258_258860


namespace fraction_subtraction_l258_258097

theorem fraction_subtraction (a b : ℕ) (h₁ : a = 18) (h₂ : b = 14) :
  (↑a / ↑b - ↑b / ↑a) = (32 / 63) := by
  sorry

end fraction_subtraction_l258_258097


namespace polygon_with_45_deg_exterior_angle_is_eight_gon_l258_258412

theorem polygon_with_45_deg_exterior_angle_is_eight_gon
  (each_exterior_angle : ℝ) (h1 : each_exterior_angle = 45) 
  (sum_exterior_angles : ℝ) (h2 : sum_exterior_angles = 360) :
  ∃ (n : ℕ), n = 8 :=
by
  sorry

end polygon_with_45_deg_exterior_angle_is_eight_gon_l258_258412


namespace number_of_round_trips_each_bird_made_l258_258787

theorem number_of_round_trips_each_bird_made
  (distance_to_materials : ℕ)
  (total_distance_covered : ℕ)
  (distance_one_round_trip : ℕ)
  (total_number_of_trips : ℕ)
  (individual_bird_trips : ℕ) :
  distance_to_materials = 200 →
  total_distance_covered = 8000 →
  distance_one_round_trip = 2 * distance_to_materials →
  total_number_of_trips = total_distance_covered / distance_one_round_trip →
  individual_bird_trips = total_number_of_trips / 2 →
  individual_bird_trips = 10 :=
by
  intros
  sorry

end number_of_round_trips_each_bird_made_l258_258787


namespace build_time_40_workers_l258_258296

theorem build_time_40_workers (r : ℝ) : 
  (60 * r) * 5 = 1 → (40 * r) * t = 1 → t = 7.5 :=
by
  intros h1 h2
  sorry

end build_time_40_workers_l258_258296


namespace find_relationship_l258_258385

noncomputable def log_equation (c d : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → 6 * (Real.log (x) / Real.log (c))^2 + 5 * (Real.log (x) / Real.log (d))^2 = 12 * (Real.log (x))^2 / (Real.log (c) * Real.log (d))

theorem find_relationship (c d : ℝ) :
  log_equation c d → 
    (d = c ^ (5 / (6 + Real.sqrt 6)) ∨ d = c ^ (5 / (6 - Real.sqrt 6))) :=
by
  sorry

end find_relationship_l258_258385


namespace foxes_hunt_duration_l258_258289

variable (initial_weasels : ℕ) (initial_rabbits : ℕ) (remaining_rodents : ℕ)
variable (foxes : ℕ) (weasels_per_week : ℕ) (rabbits_per_week : ℕ)

def total_rodents_per_week (weasels_per_week rabbits_per_week foxes : ℕ) : ℕ :=
  foxes * (weasels_per_week + rabbits_per_week)

def initial_rodents (initial_weasels initial_rabbits : ℕ) : ℕ :=
  initial_weasels + initial_rabbits

def total_rodents_caught (initial_rodents remaining_rodents : ℕ) : ℕ :=
  initial_rodents - remaining_rodents

def weeks_hunted (total_rodents_caught total_rodents_per_week : ℕ) : ℕ :=
  total_rodents_caught / total_rodents_per_week

theorem foxes_hunt_duration
  (initial_weasels := 100) (initial_rabbits := 50) (remaining_rodents := 96)
  (foxes := 3) (weasels_per_week := 4) (rabbits_per_week := 2) :
  weeks_hunted (total_rodents_caught (initial_rodents initial_weasels initial_rabbits) remaining_rodents) 
                 (total_rodents_per_week weasels_per_week rabbits_per_week foxes) = 3 :=
by
  sorry

end foxes_hunt_duration_l258_258289


namespace relationship_between_m_and_n_l258_258167

theorem relationship_between_m_and_n
  (b m n : ℝ)
  (h₁ : m = 2 * (-1 / 2) + b)
  (h₂ : n = 2 * 2 + b) :
  m < n :=
by
  sorry

end relationship_between_m_and_n_l258_258167


namespace distance_center_to_point_l258_258790

theorem distance_center_to_point : 
  let center := (2, 3)
  let point  := (5, -2)
  let distance := Real.sqrt ((5 - 2)^2 + (-2 - 3)^2)
  distance = Real.sqrt 34 := by
  sorry

end distance_center_to_point_l258_258790


namespace find_k_l258_258856

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end find_k_l258_258856


namespace unique_sum_of_two_primes_l258_258446

theorem unique_sum_of_two_primes (p1 p2 : ℕ) (hp1_prime : Prime p1) (hp2_prime : Prime p2) (hp1_even : p1 = 2) (sum_eq : p1 + p2 = 10003) : 
  p1 = 2 ∧ p2 = 10001 ∧ (∀ p1' p2', Prime p1' → Prime p2' → p1' + p2' = 10003 → (p1' = 2 ∧ p2' = 10001) ∨ (p1' = 10001 ∧ p2' = 2)) :=
by
  sorry

end unique_sum_of_two_primes_l258_258446


namespace no_prime_sum_10003_l258_258435

theorem no_prime_sum_10003 : ¬∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 10003 :=
by
  -- Lean proof skipped, as per the instructions.
  exact sorry

end no_prime_sum_10003_l258_258435


namespace prob_of_yellow_second_l258_258123

-- Defining the probabilities based on the given conditions
def prob_white_from_X : ℚ := 5 / 8
def prob_black_from_X : ℚ := 3 / 8
def prob_yellow_from_Y : ℚ := 8 / 10
def prob_yellow_from_Z : ℚ := 3 / 7

-- Combining probabilities
def combined_prob_white_Y : ℚ := prob_white_from_X * prob_yellow_from_Y
def combined_prob_black_Z : ℚ := prob_black_from_X * prob_yellow_from_Z

-- Total probability of drawing a yellow marble in the second draw
def total_prob_yellow_second : ℚ := combined_prob_white_Y + combined_prob_black_Z

-- Proof statement
theorem prob_of_yellow_second :
  total_prob_yellow_second = 37 / 56 := 
sorry

end prob_of_yellow_second_l258_258123


namespace billion_in_scientific_notation_l258_258028

theorem billion_in_scientific_notation :
  (10^9 = 1 * 10^9) :=
by
  sorry

end billion_in_scientific_notation_l258_258028


namespace expression_of_24ab_in_P_and_Q_l258_258628

theorem expression_of_24ab_in_P_and_Q (a b : ℕ) (P Q : ℝ)
  (hP : P = 2^a) (hQ : Q = 5^b) : 24^(a*b) = P^(3*b) * 3^(a*b) := 
  by
  sorry

end expression_of_24ab_in_P_and_Q_l258_258628


namespace total_footprints_l258_258513

def pogo_footprints_per_meter : ℕ := 4
def grimzi_footprints_per_6_meters : ℕ := 3
def distance_traveled_meters : ℕ := 6000

theorem total_footprints : (pogo_footprints_per_meter * distance_traveled_meters) + (grimzi_footprints_per_6_meters * (distance_traveled_meters / 6)) = 27000 :=
by
  sorry

end total_footprints_l258_258513


namespace Ariella_total_amount_l258_258670

-- We define the conditions
def Daniella_initial (daniella_amount : ℝ) := daniella_amount = 400
def Ariella_initial (daniella_amount : ℝ) (ariella_amount : ℝ) := ariella_amount = daniella_amount + 200
def simple_interest_rate : ℝ := 0.10
def investment_period : ℕ := 2

-- We state the goal to prove
theorem Ariella_total_amount (daniella_amount ariella_amount : ℝ) :
  Daniella_initial daniella_amount →
  Ariella_initial daniella_amount ariella_amount →
  ariella_amount + ariella_amount * simple_interest_rate * (investment_period : ℝ) = 720 :=
by
  sorry

end Ariella_total_amount_l258_258670


namespace find_angle_l258_258768

-- Define the conditions
variables (x : ℝ)

-- Conditions given in the problem
def angle_complement_condition (x : ℝ) := (10 : ℝ) + 3 * x
def complementary_condition (x : ℝ) := x + angle_complement_condition x = 90

-- Prove that the angle x equals to 20 degrees
theorem find_angle : (complementary_condition x) → x = 20 := 
by
  -- Placeholder for the proof
  sorry

end find_angle_l258_258768


namespace problem_statement_l258_258725

noncomputable def find_pq_sum (XZ YZ : ℕ) (XY_perimeter_ratio : ℕ × ℕ) : ℕ :=
  let XY := Real.sqrt (XZ^2 + YZ^2)
  let ZD := Real.sqrt (XZ * YZ)
  let O_radius := 0.5 * ZD
  let tangent_length := Real.sqrt ((XY / 2)^2 - O_radius^2)
  let perimeter := XY + 2 * tangent_length
  let (p, q) := XY_perimeter_ratio
  p + q

theorem problem_statement :
  find_pq_sum 8 15 (30, 17) = 47 :=
by sorry

end problem_statement_l258_258725


namespace total_letters_received_l258_258584

theorem total_letters_received 
  (Brother_received Greta_received Mother_received : ℕ) 
  (h1 : Greta_received = Brother_received + 10)
  (h2 : Brother_received = 40)
  (h3 : Mother_received = 2 * (Greta_received + Brother_received)) :
  Brother_received + Greta_received + Mother_received = 270 := 
sorry

end total_letters_received_l258_258584


namespace transformed_sin_graph_l258_258500

noncomputable def transform_sin_graph : ℝ → ℝ :=
  λ x, sin (1/2 * x - π/10)

theorem transformed_sin_graph (x : ℝ) :
  let f := λ x, sin x
  let g := λ x, f (x - π/10)
  let h := λ x, g (2 * x)
  transform_sin_graph x = h x :=
by
  simp [transform_sin_graph, h, g, f]
  sorry

end transformed_sin_graph_l258_258500


namespace largest_three_digit_multiple_of_12_with_digit_sum_24_l258_258055

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 12 = 0 ∧ (n.digits 10).sum = 24 ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 12 = 0 ∧ (m.digits 10).sum = 24 → m ≤ n) ∧ n = 888 :=
by {
  sorry -- Proof to be filled in
}

#eval largest_three_digit_multiple_of_12_with_digit_sum_24 -- Should output: ⊤ (True)

end largest_three_digit_multiple_of_12_with_digit_sum_24_l258_258055


namespace complex_equation_solution_l258_258690

theorem complex_equation_solution (x y : ℝ)
  (h : (x / (1 - (-ⅈ)) + y / (1 - 2 * (-ⅈ)) = 5 / (1 - 3 * (-ⅈ)))) :
  x + y = 4 :=
sorry

end complex_equation_solution_l258_258690


namespace part1_part2_l258_258701

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  k - |x - 3|

theorem part1 (k : ℝ) (h : ∀ x, f (x + 3) k ≥ 0 ↔ x ∈ [-1, 1]) : k = 1 :=
sorry

variable (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c)

theorem part2 (h : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1) : 
  (1 / 9) * a + (2 / 9) * b + (3 / 9) * c ≥ 1 :=
sorry

end part1_part2_l258_258701


namespace compound_interest_calculation_l258_258342

theorem compound_interest_calculation :
  let P_SI := 1750.0000000000018
  let r_SI := 0.08
  let t_SI := 3
  let r_CI := 0.10
  let t_CI := 2
  let SI := P_SI * r_SI * t_SI
  let CI (P_CI : ℝ) := P_CI * ((1 + r_CI) ^ t_CI - 1)
  (SI = 420.0000000000004) →
  (SI = (1 / 2) * CI P_CI) →
  P_CI = 4000.000000000004 :=
by
  intros P_SI r_SI t_SI r_CI t_CI SI CI h1 h2
  sorry

end compound_interest_calculation_l258_258342


namespace minimal_area_circle_equation_circle_equation_center_on_line_l258_258809

-- Question (1): Prove the equation of the circle with minimal area
theorem minimal_area_circle_equation :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  C = (0, -4) ∧ r = Real.sqrt 5 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → P.1 ^ 2 + (P.2 + 4) ^ 2 = 5) :=
sorry

-- Question (2): Prove the equation of a circle with the center on a specific line
theorem circle_equation_center_on_line :
  (∃ (C : ℝ × ℝ) (r : ℝ), (r > 0) ∧ 
  (C.1 - 2 * C.2 - 3 = 0) ∧
  C = (-1, -2) ∧ r = Real.sqrt 10 ∧ 
  ∀ (P : ℝ × ℝ), (P = (2, -3) ∨ P = (-2, -5)) → (P.1 + 1) ^ 2 + (P.2 + 2) ^ 2 = 10) :=
sorry

end minimal_area_circle_equation_circle_equation_center_on_line_l258_258809


namespace production_today_l258_258849

theorem production_today (n x: ℕ) (avg_past: ℕ) 
  (h1: avg_past = 50) 
  (h2: n = 1) 
  (h3: (avg_past * n + x) / (n + 1) = 55): 
  x = 60 := 
by 
  sorry

end production_today_l258_258849


namespace mother_age_4times_daughter_l258_258046

-- Conditions
def Y := 12
def M := 42

-- Proof statement: Prove that 2 years ago, the mother's age was 4 times Yujeong's age.
theorem mother_age_4times_daughter (X : ℕ) (hY : Y = 12) (hM : M = 42) : (42 - X) = 4 * (12 - X) :=
by
  intros
  sorry

end mother_age_4times_daughter_l258_258046


namespace length_of_intervals_l258_258150

theorem length_of_intervals (n : ℕ) (hn : 0 < n) :
  ∃ A : set ℝ, (∀ x ∈ A, 0 < x ∧ x < 1) ∧ (∀ (p q : ℕ), q ≤ n^2 → (x ∈ A → |x - p/q| > 1/n^3)) ∧
  (∀ I ∈ A, I.is_interval) ∧ measurable_set A ∧ measure A ≤ 100 / n := sorry

end length_of_intervals_l258_258150


namespace even_function_iff_a_eq_1_l258_258695

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- Lean 4 statement for the given math proof problem
theorem even_function_iff_a_eq_1 :
  (∀ x : ℝ, f a x = f a (-x)) ↔ a = 1 :=
sorry

end even_function_iff_a_eq_1_l258_258695


namespace combinatorial_identity_l258_258686

open BigOperators

theorem combinatorial_identity (n : ℕ) :
  ∑ k in Finset.range (n + 1), n.choose k * 2^k * ((n - k) / 2).choose (n - k) = (2 * n + 1).choose n :=
by
  sorry

end combinatorial_identity_l258_258686


namespace projection_of_b_onto_a_l258_258996
-- Import the entire library for necessary functions and definitions.

-- Define the problem in Lean 4, using relevant conditions and statement.
theorem projection_of_b_onto_a (m : ℝ) (h : (1 : ℝ) * 3 + (Real.sqrt 3) * m = 6) : m = Real.sqrt 3 :=
by
  sorry

end projection_of_b_onto_a_l258_258996


namespace max_side_length_of_squares_l258_258044

variable (l w : ℕ)
variable (h_l : l = 54)
variable (h_w : w = 24)

theorem max_side_length_of_squares : gcd l w = 6 :=
by
  rw [h_l, h_w]
  sorry

end max_side_length_of_squares_l258_258044


namespace value_of_c_plus_d_l258_258988

theorem value_of_c_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : a + d = 2) : c + d = 3 :=
by
  sorry

end value_of_c_plus_d_l258_258988


namespace difference_of_two_numbers_l258_258211

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l258_258211


namespace remainder_proof_l258_258936

-- Definitions and conditions
variables {x y u v : ℕ}
variables (hx : x = u * y + v)

-- Problem statement in Lean 4
theorem remainder_proof (hx : x = u * y + v) : ((x + 3 * u * y + y) % y) = v :=
sorry

end remainder_proof_l258_258936


namespace football_team_progress_l258_258937

theorem football_team_progress (loss gain : ℤ) (h_loss : loss = -5) (h_gain : gain = 8) :
  (loss + gain = 3) :=
by
  sorry

end football_team_progress_l258_258937


namespace james_pay_for_two_semesters_l258_258888

theorem james_pay_for_two_semesters
  (units_per_semester : ℕ) (cost_per_unit : ℕ) (num_semesters : ℕ)
  (h_units : units_per_semester = 20) (h_cost : cost_per_unit = 50) (h_semesters : num_semesters = 2) :
  units_per_semester * cost_per_unit * num_semesters = 2000 := 
by 
  rw [h_units, h_cost, h_semesters]
  norm_num

end james_pay_for_two_semesters_l258_258888
