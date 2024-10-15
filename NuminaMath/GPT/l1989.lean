import Mathlib

namespace NUMINAMATH_GPT_union_complement_real_domain_l1989_198972

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -2 < x}

theorem union_complement_real_domain :
  M ∪ (Set.univ \ N) = {x : ℝ | x < 2} :=
by
  sorry

end NUMINAMATH_GPT_union_complement_real_domain_l1989_198972


namespace NUMINAMATH_GPT_find_max_value_l1989_198951

theorem find_max_value (f : ℝ → ℝ) (h₀ : f 0 = -5) (h₁ : ∀ x, deriv f x = 4 * x^3 - 4 * x) :
  ∃ x, f x = -5 ∧ (∀ y, f y ≤ f x) ∧ x = 0 :=
sorry

end NUMINAMATH_GPT_find_max_value_l1989_198951


namespace NUMINAMATH_GPT_mike_weekly_avg_time_l1989_198953

theorem mike_weekly_avg_time :
  let mon_wed_fri_tv := 4 -- hours per day on Mon, Wed, Fri
  let tue_thu_tv := 3 -- hours per day on Tue, Thu
  let weekend_tv := 5 -- hours per day on weekends
  let num_mon_wed_fri := 3 -- days
  let num_tue_thu := 2 -- days
  let num_weekend := 2 -- days
  let num_days_week := 7 -- days
  let num_video_game_days := 3 -- days
  let weeks := 4 -- weeks
  let mon_wed_fri_total := mon_wed_fri_tv * num_mon_wed_fri
  let tue_thu_total := tue_thu_tv * num_tue_thu
  let weekend_total := weekend_tv * num_weekend
  let weekly_tv_time := mon_wed_fri_total + tue_thu_total + weekend_total
  let daily_avg_tv_time := weekly_tv_time / num_days_week
  let daily_video_game_time := daily_avg_tv_time / 2
  let weekly_video_game_time := daily_video_game_time * num_video_game_days
  let total_tv_time_4_weeks := weekly_tv_time * weeks
  let total_video_game_time_4_weeks := weekly_video_game_time * weeks
  let total_time_4_weeks := total_tv_time_4_weeks + total_video_game_time_4_weeks
  let weekly_avg_time := total_time_4_weeks / weeks
  weekly_avg_time = 34 := sorry

end NUMINAMATH_GPT_mike_weekly_avg_time_l1989_198953


namespace NUMINAMATH_GPT_lindsey_final_money_l1989_198927

-- Define the savings in each month
def save_sep := 50
def save_oct := 37
def save_nov := 11

-- Total savings over the three months
def total_savings := save_sep + save_oct + save_nov

-- Condition for Mom's contribution
def mom_contribution := if total_savings > 75 then 25 else 0

-- Total savings including mom's contribution
def total_with_mom := total_savings + mom_contribution

-- Amount spent on the video game
def spent := 87

-- Final amount left
def final_amount := total_with_mom - spent

-- Proof statement
theorem lindsey_final_money : final_amount = 36 := by
  sorry

end NUMINAMATH_GPT_lindsey_final_money_l1989_198927


namespace NUMINAMATH_GPT_opposite_sign_pairs_l1989_198980

theorem opposite_sign_pairs :
  ¬ ((- 2 ^ 3 < 0) ∧ (- (2 ^ 3) > 0)) ∧
  ¬ (|-4| < 0 ∧ -(-4) > 0) ∧
  ((- 3 ^ 4 < 0 ∧ (-(3 ^ 4)) = 81)) ∧
  ¬ (10 ^ 2 < 0 ∧ 2 ^ 10 > 0) :=
by
  sorry

end NUMINAMATH_GPT_opposite_sign_pairs_l1989_198980


namespace NUMINAMATH_GPT_adults_tickets_sold_eq_1200_l1989_198949

variable (A : ℕ)
variable (S : ℕ := 300) -- Number of student tickets
variable (P_adult : ℕ := 12) -- Price per adult ticket
variable (P_student : ℕ := 6) -- Price per student ticket
variable (total_tickets : ℕ := 1500) -- Total tickets sold
variable (total_amount : ℕ := 16200) -- Total amount collected

theorem adults_tickets_sold_eq_1200
  (h1 : S = 300)
  (h2 : A + S = total_tickets)
  (h3 : P_adult * A + P_student * S = total_amount) :
  A = 1200 := by
  sorry

end NUMINAMATH_GPT_adults_tickets_sold_eq_1200_l1989_198949


namespace NUMINAMATH_GPT_inequality_ineq_l1989_198968

variable (x y z : Real)

theorem inequality_ineq {x y z : Real} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 3) :
  (1 / (x^5 - x^2 + 3)) + (1 / (y^5 - y^2 + 3)) + (1 / (z^5 - z^2 + 3)) ≤ 1 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_ineq_l1989_198968


namespace NUMINAMATH_GPT_percent_of_x_is_z_l1989_198983

theorem percent_of_x_is_z (x y z : ℝ) (h1 : 0.45 * z = 1.2 * y) (h2 : y = 0.75 * x) : z = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_z_l1989_198983


namespace NUMINAMATH_GPT_octagon_area_equals_eight_one_plus_sqrt_two_l1989_198909

theorem octagon_area_equals_eight_one_plus_sqrt_two
  (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a ^ 2 = 16) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_octagon_area_equals_eight_one_plus_sqrt_two_l1989_198909


namespace NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1989_198994

theorem cubic_roots_sum_of_cubes (r s t a b c : ℚ) 
  (h1 : r + s + t = a) 
  (h2 : r * s + r * t + s * t = b)
  (h3 : r * s * t = c) 
  (h_poly : ∀ x : ℚ, x^3 - a*x^2 + b*x - c = 0 ↔ (x = r ∨ x = s ∨ x = t)) :
  r^3 + s^3 + t^3 = a^3 - 3 * a * b + 3 * c :=
sorry

end NUMINAMATH_GPT_cubic_roots_sum_of_cubes_l1989_198994


namespace NUMINAMATH_GPT_number_of_circles_is_3_l1989_198911

-- Define the radius and diameter of the circles
def radius := 4
def diameter := 2 * radius

-- Given the total horizontal length
def total_horizontal_length := 24

-- Number of circles calculated as per the given conditions
def number_of_circles := total_horizontal_length / diameter

-- The proof statement to verify
theorem number_of_circles_is_3 : number_of_circles = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_circles_is_3_l1989_198911


namespace NUMINAMATH_GPT_sum_of_coefficients_of_parabolas_kite_formed_l1989_198962

theorem sum_of_coefficients_of_parabolas_kite_formed (a b : ℝ) 
  (h1 : ∃ (x : ℝ), y = ax^2 - 4)
  (h2 : ∃ (y : ℝ), y = 6 - bx^2)
  (h3 : (a > 0) ∧ (b > 0) ∧ (ax^2 - 4 = 0) ∧ (6 - bx^2 = 0))
  (h4 : kite_area = 18) :
  a + b = 125/36 := 
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_parabolas_kite_formed_l1989_198962


namespace NUMINAMATH_GPT_sqrt_square_multiply_l1989_198926

theorem sqrt_square_multiply (a : ℝ) (h : a = 49284) :
  (Real.sqrt a)^2 * 3 = 147852 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_square_multiply_l1989_198926


namespace NUMINAMATH_GPT_find_equation_of_line_l1989_198952

-- Define the conditions
def line_passes_through_A (m b : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (1, 1) ∧ A.2 = -A.1 + b

def intercepts_equal (m b : ℝ) : Prop :=
  b = m

-- The goal to prove the equations of the line
theorem find_equation_of_line :
  ∃ (m b : ℝ), line_passes_through_A m b (1, 1) ∧ intercepts_equal m b ↔ 
  (∃ m b : ℝ, (m = -1 ∧ b = 2) ∨ (m = 1 ∧ b = 0)) :=
sorry

end NUMINAMATH_GPT_find_equation_of_line_l1989_198952


namespace NUMINAMATH_GPT_geometric_sequence_arithmetic_sequence_l1989_198984

def seq₃ := 7
def rec_rel (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + a 2 - 2

-- Problem Part 1: Prove that {a_n+1} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℕ) (h_rec_rel : rec_rel a) :
  ∃ r, ∀ n, n ≥ 1 → (a n + 1) = r * (a (n - 1) + 1) :=
sorry

-- Problem Part 2: Given a general formula, prove n, a_n, and S_n form an arithmetic sequence
def general_formula (a : ℕ → ℕ) := ∀ n, a n = 2^n - 1
def sum_formula (S : ℕ → ℕ) := ∀ n, S n = 2^(n+1) - n - 2

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_general : general_formula a) (h_sum : sum_formula S) :
  ∀ n, n + S n = 2 * a n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_arithmetic_sequence_l1989_198984


namespace NUMINAMATH_GPT_x_intercept_of_parabola_l1989_198948

theorem x_intercept_of_parabola (a b c : ℝ)
    (h_vertex : ∀ x, (a * (x - 5)^2 + 9 = y) → (x, y) = (5, 9))
    (h_intercept : ∀ x, (a * x^2 + b * x + c = 0) → x = 0 ∨ y = 0) :
    ∃ x0 : ℝ, x0 = 10 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_parabola_l1989_198948


namespace NUMINAMATH_GPT_derivative_of_f_at_pi_over_2_l1989_198934

noncomputable def f (x : Real) := 5 * Real.sin x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = 0 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_derivative_of_f_at_pi_over_2_l1989_198934


namespace NUMINAMATH_GPT_shaded_percentage_seven_by_seven_grid_l1989_198917

theorem shaded_percentage_seven_by_seven_grid :
  let total_squares := 49
  let shaded_squares := 7
  let shaded_fraction := shaded_squares / total_squares
  let shaded_percentage := shaded_fraction * 100
  shaded_percentage = 14.29 := by
  sorry

end NUMINAMATH_GPT_shaded_percentage_seven_by_seven_grid_l1989_198917


namespace NUMINAMATH_GPT_sum_possible_values_l1989_198915

def abs_eq_2023 (a : ℤ) : Prop := abs a = 2023
def abs_eq_2022 (b : ℤ) : Prop := abs b = 2022
def greater_than (a b : ℤ) : Prop := a > b

theorem sum_possible_values (a b : ℤ) (h1 : abs_eq_2023 a) (h2 : abs_eq_2022 b) (h3 : greater_than a b) :
  a + b = 1 ∨ a + b = 4045 := 
sorry

end NUMINAMATH_GPT_sum_possible_values_l1989_198915


namespace NUMINAMATH_GPT_rectangle_inscribed_circle_hypotenuse_l1989_198950

open Real

theorem rectangle_inscribed_circle_hypotenuse
  (AB BC : ℝ)
  (h_AB : AB = 20)
  (h_BC : BC = 10)
  (r : ℝ)
  (h_r : r = 10 / 3) :
  sqrt ((AB - 2 * r) ^ 2 + BC ^ 2) = 50 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_inscribed_circle_hypotenuse_l1989_198950


namespace NUMINAMATH_GPT_quadratic_roots_real_equal_l1989_198965

theorem quadratic_roots_real_equal (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ a = 3 ∧ b = 2 - m ∧ c = 6 ∧
    (b^2 - 4 * a * c = 0)) ↔ (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_equal_l1989_198965


namespace NUMINAMATH_GPT_largest_multiple_of_7_smaller_than_neg_55_l1989_198906

theorem largest_multiple_of_7_smaller_than_neg_55 : ∃ m : ℤ, m % 7 = 0 ∧ m < -55 ∧ ∀ n : ℤ, n % 7 = 0 → n < -55 → n ≤ m :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_7_smaller_than_neg_55_l1989_198906


namespace NUMINAMATH_GPT_remaining_pages_l1989_198943

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end NUMINAMATH_GPT_remaining_pages_l1989_198943


namespace NUMINAMATH_GPT_time_to_cross_tree_l1989_198966

variable (length_train : ℕ) (time_platform : ℕ) (length_platform : ℕ)

theorem time_to_cross_tree (h1 : length_train = 1200) (h2 : time_platform = 190) (h3 : length_platform = 700) :
  let distance_platform := length_train + length_platform
  let speed_train := distance_platform / time_platform
  let time_to_cross_tree := length_train / speed_train
  time_to_cross_tree = 120 :=
by
  -- Using the conditions to prove the goal
  sorry

end NUMINAMATH_GPT_time_to_cross_tree_l1989_198966


namespace NUMINAMATH_GPT_marla_drive_time_l1989_198945

theorem marla_drive_time (x : ℕ) (h_total : x + 70 + x = 110) : x = 20 :=
sorry

end NUMINAMATH_GPT_marla_drive_time_l1989_198945


namespace NUMINAMATH_GPT_transformations_map_figure_l1989_198936

noncomputable def count_transformations : ℕ := sorry

theorem transformations_map_figure :
  count_transformations = 3 :=
sorry

end NUMINAMATH_GPT_transformations_map_figure_l1989_198936


namespace NUMINAMATH_GPT_exterior_angle_decreases_l1989_198933

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) (n' : ℕ) (hn' : n' ≥ n) :
  (360 : ℝ) / n' < (360 : ℝ) / n := by sorry

end NUMINAMATH_GPT_exterior_angle_decreases_l1989_198933


namespace NUMINAMATH_GPT_sam_friend_points_l1989_198919

theorem sam_friend_points (sam_points total_points : ℕ) (h1 : sam_points = 75) (h2 : total_points = 87) :
  total_points - sam_points = 12 :=
by sorry

end NUMINAMATH_GPT_sam_friend_points_l1989_198919


namespace NUMINAMATH_GPT_min_sqrt_eq_sum_sqrt_implies_param_l1989_198925

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem min_sqrt_eq_sum_sqrt_implies_param (a b c : ℝ) (r s t : ℝ)
    (h1 : 0 < a ∧ a ≤ 1)
    (h2 : 0 < b ∧ b ≤ 1)
    (h3 : 0 < c ∧ c ≤ 1)
    (h4 : min (sqrt ((a * b + 1) / (a * b * c))) (min (sqrt ((b * c + 1) / (a * b * c))) (sqrt ((a * c + 1) / (a * b * c)))) 
          = (sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c))) :
    ∃ r, a = 1 / (1 + r^2) ∧ b = 1 / (1 + (1 / r^2)) ∧ c = (r + 1 / r)^2 / (1 + (r + 1 / r)^2) :=
sorry

end NUMINAMATH_GPT_min_sqrt_eq_sum_sqrt_implies_param_l1989_198925


namespace NUMINAMATH_GPT_smaller_cube_size_l1989_198998

theorem smaller_cube_size
  (original_cube_side : ℕ)
  (number_of_smaller_cubes : ℕ)
  (painted_cubes : ℕ)
  (unpainted_cubes : ℕ) :
  original_cube_side = 3 → 
  number_of_smaller_cubes = 27 → 
  painted_cubes = 26 → 
  unpainted_cubes = 1 →
  (∃ (side : ℕ), side = original_cube_side / 3 ∧ side = 1) :=
by
  intros h1 h2 h3 h4
  use 1
  have h : 1 = original_cube_side / 3 := sorry
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_smaller_cube_size_l1989_198998


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1989_198978

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : x^2 + y^2 = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1989_198978


namespace NUMINAMATH_GPT_worker_efficiency_l1989_198996

theorem worker_efficiency (W_p W_q : ℚ) 
  (h1 : W_p = 1 / 24) 
  (h2 : W_p + W_q = 1 / 14) :
  (W_p - W_q) / W_q * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_worker_efficiency_l1989_198996


namespace NUMINAMATH_GPT_relationship_between_length_and_width_l1989_198991

theorem relationship_between_length_and_width 
  (x y : ℝ) (h : 2 * (x + y) = 20) : y = 10 - x := 
by
  sorry

end NUMINAMATH_GPT_relationship_between_length_and_width_l1989_198991


namespace NUMINAMATH_GPT_volunteer_comprehensive_score_is_92_l1989_198904

noncomputable def written_score : ℝ := 90
noncomputable def trial_lecture_score : ℝ := 94
noncomputable def interview_score : ℝ := 90

noncomputable def written_weight : ℝ := 0.3
noncomputable def trial_lecture_weight : ℝ := 0.5
noncomputable def interview_weight : ℝ := 0.2

noncomputable def comprehensive_score : ℝ :=
  written_score * written_weight +
  trial_lecture_score * trial_lecture_weight +
  interview_score * interview_weight

theorem volunteer_comprehensive_score_is_92 :
  comprehensive_score = 92 := by
  sorry

end NUMINAMATH_GPT_volunteer_comprehensive_score_is_92_l1989_198904


namespace NUMINAMATH_GPT_multiples_of_10_between_11_and_103_l1989_198928

def countMultiplesOf10 (lower_bound upper_bound : Nat) : Nat :=
  Nat.div (upper_bound - lower_bound) 10 + 1

theorem multiples_of_10_between_11_and_103 : 
  countMultiplesOf10 11 103 = 9 :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_10_between_11_and_103_l1989_198928


namespace NUMINAMATH_GPT_trig_identity_l1989_198973

-- Define the given condition
def tan_half (α : ℝ) : Prop := Real.tan (α / 2) = 2

-- The main statement we need to prove
theorem trig_identity (α : ℝ) (h : tan_half α) : (1 + Real.cos α) / (Real.sin α) = 1 / 2 :=
  by
  sorry

end NUMINAMATH_GPT_trig_identity_l1989_198973


namespace NUMINAMATH_GPT_radicals_like_simplest_forms_l1989_198912

theorem radicals_like_simplest_forms (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a = b + 2) :
  a = 3 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_radicals_like_simplest_forms_l1989_198912


namespace NUMINAMATH_GPT_factorize_x_cube_minus_4x_l1989_198902

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end NUMINAMATH_GPT_factorize_x_cube_minus_4x_l1989_198902


namespace NUMINAMATH_GPT_find_alpha_l1989_198924

noncomputable def parametric_eq_line (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_eq_curve (x y : Real) : Prop :=
  y^2 = 4 * x

def intersection_condition (α t₁ t₂ : Real) : Prop :=
  Real.sin α ≠ 0 ∧ 
  (1 + t₁ * Real.cos α, t₁ * Real.sin α) = (1 + t₂ * Real.cos α, t₂ * Real.sin α) ∧ 
  Real.sqrt ((t₁ + t₂)^2 - 4 * (-4 / (Real.sin α)^2)) = 8

theorem find_alpha (α : Real) (t₁ t₂ : Real) 
  (h1: 0 < α) (h2: α < π) (h3: intersection_condition α t₁ t₂) : 
  α = π/4 ∨ α = 3*π/4 :=
by 
  sorry

end NUMINAMATH_GPT_find_alpha_l1989_198924


namespace NUMINAMATH_GPT_fifth_friend_paid_l1989_198938

theorem fifth_friend_paid (a b c d e : ℝ)
  (h1 : a = (1/3) * (b + c + d + e))
  (h2 : b = (1/4) * (a + c + d + e))
  (h3 : c = (1/5) * (a + b + d + e))
  (h4 : a + b + c + d + e = 120) :
  e = 40 :=
sorry

end NUMINAMATH_GPT_fifth_friend_paid_l1989_198938


namespace NUMINAMATH_GPT_friend_selling_price_l1989_198976

-- Definitions and conditions
def original_cost_price : ℝ := 51724.14

def loss_percentage : ℝ := 0.13
def gain_percentage : ℝ := 0.20

def selling_price_man (CP : ℝ) : ℝ := (1 - loss_percentage) * CP
def selling_price_friend (SP1 : ℝ) : ℝ := (1 + gain_percentage) * SP1

-- Prove that the friend's selling price is 54,000 given the conditions
theorem friend_selling_price :
  selling_price_friend (selling_price_man original_cost_price) = 54000 :=
by
  sorry

end NUMINAMATH_GPT_friend_selling_price_l1989_198976


namespace NUMINAMATH_GPT_store_discount_l1989_198907

theorem store_discount (P : ℝ) :
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  P2 = 0.774 * P :=
by
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  sorry

end NUMINAMATH_GPT_store_discount_l1989_198907


namespace NUMINAMATH_GPT_selection_problem_l1989_198977

def group_size : ℕ := 10
def selected_group_size : ℕ := 3
def total_ways_without_C := Nat.choose 9 3
def ways_without_A_B_C := Nat.choose 7 3
def correct_answer := total_ways_without_C - ways_without_A_B_C

theorem selection_problem:
  (∃ (A B C : ℕ), total_ways_without_C - ways_without_A_B_C = 49) :=
by
  sorry

end NUMINAMATH_GPT_selection_problem_l1989_198977


namespace NUMINAMATH_GPT_correct_operation_l1989_198920

theorem correct_operation (a b : ℝ) : (a^3 * b)^2 = a^6 * b^2 :=
sorry

end NUMINAMATH_GPT_correct_operation_l1989_198920


namespace NUMINAMATH_GPT_find_siblings_l1989_198956

-- Define the characteristics of each child
structure Child where
  name : String
  eyeColor : String
  hairColor : String
  age : Nat

-- List of children
def Olivia : Child := { name := "Olivia", eyeColor := "Green", hairColor := "Red", age := 12 }
def Henry  : Child := { name := "Henry", eyeColor := "Gray", hairColor := "Brown", age := 12 }
def Lucas  : Child := { name := "Lucas", eyeColor := "Green", hairColor := "Red", age := 10 }
def Emma   : Child := { name := "Emma", eyeColor := "Green", hairColor := "Brown", age := 12 }
def Mia    : Child := { name := "Mia", eyeColor := "Gray", hairColor := "Red", age := 10 }
def Noah   : Child := { name := "Noah", eyeColor := "Gray", hairColor := "Brown", age := 12 }

-- Define a family as a set of children who share at least one characteristic
def isFamily (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.eyeColor = c3.eyeColor ∨ c2.eyeColor = c3.eyeColor) ∨
  (c1.hairColor = c2.hairColor ∨ c1.hairColor = c3.hairColor ∨ c2.hairColor = c3.hairColor) ∨
  (c1.age = c2.age ∨ c1.age = c3.age ∨ c2.age = c3.age)

-- The main theorem
theorem find_siblings : isFamily Olivia Lucas Emma :=
by
  sorry

end NUMINAMATH_GPT_find_siblings_l1989_198956


namespace NUMINAMATH_GPT_smallest_q_exists_l1989_198914

noncomputable def p_q_r_are_consecutive_terms (p q r : ℝ) : Prop :=
∃ d : ℝ, p = q - d ∧ r = q + d

theorem smallest_q_exists
  (p q r : ℝ)
  (h1 : p_q_r_are_consecutive_terms p q r)
  (h2 : p > 0) 
  (h3 : q > 0) 
  (h4 : r > 0)
  (h5 : p * q * r = 216) :
  q = 6 :=
sorry

end NUMINAMATH_GPT_smallest_q_exists_l1989_198914


namespace NUMINAMATH_GPT_correct_option_D_l1989_198913

theorem correct_option_D : -2 = -|-2| := 
by 
  sorry

end NUMINAMATH_GPT_correct_option_D_l1989_198913


namespace NUMINAMATH_GPT_flowerbed_width_l1989_198986

theorem flowerbed_width (w : ℝ) (h₁ : 22 = 2 * (2 * w - 1) + 2 * w) : w = 4 :=
sorry

end NUMINAMATH_GPT_flowerbed_width_l1989_198986


namespace NUMINAMATH_GPT_Rachel_made_total_amount_l1989_198955

def cost_per_bar : ℝ := 3.25
def total_bars_sold : ℕ := 25 - 7
def total_amount_made : ℝ := total_bars_sold * cost_per_bar

theorem Rachel_made_total_amount :
  total_amount_made = 58.50 :=
by
  sorry

end NUMINAMATH_GPT_Rachel_made_total_amount_l1989_198955


namespace NUMINAMATH_GPT_minimum_packs_needed_l1989_198932

theorem minimum_packs_needed (n : ℕ) :
  (∃ x y z : ℕ, 30 * x + 18 * y + 9 * z = 120 ∧ x + y + z = n ∧ x ≥ 2 ∧ z' = if x ≥ 2 then z + 1 else z) → n = 4 := 
by
  sorry

end NUMINAMATH_GPT_minimum_packs_needed_l1989_198932


namespace NUMINAMATH_GPT_f_in_neg_interval_l1989_198993

variables (f : ℝ → ℝ)

-- Conditions
def is_even := ∀ x, f x = f (-x)
def symmetry := ∀ x, f (2 + x) = f (2 - x)
def in_interval := ∀ x, 0 < x ∧ x < 2 → f x = 1 / x

-- Target statement
theorem f_in_neg_interval
  (h_even : is_even f)
  (h_symm : symmetry f)
  (h_interval : in_interval f)
  (x : ℝ)
  (hx : -4 < x ∧ x < -2) :
  f x = 1 / (x + 4) :=
sorry

end NUMINAMATH_GPT_f_in_neg_interval_l1989_198993


namespace NUMINAMATH_GPT_solve_for_a_l1989_198959

theorem solve_for_a (a x : ℝ) (h1 : 3 * x - 5 = x + a) (h2 : x = 2) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1989_198959


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l1989_198900

theorem cube_volume_from_surface_area (A : ℕ) (h1 : A = 864) : 
  ∃ V : ℕ, V = 1728 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l1989_198900


namespace NUMINAMATH_GPT_six_digit_mod7_l1989_198988

theorem six_digit_mod7 (a b c d e f : ℕ) (N : ℕ) (h : N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) (h_div7 : N % 7 = 0) :
    (10^5 * f + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_six_digit_mod7_l1989_198988


namespace NUMINAMATH_GPT_hexagon_area_of_circle_l1989_198964

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_area_of_circle_l1989_198964


namespace NUMINAMATH_GPT_positive_integers_satisfy_condition_l1989_198979

theorem positive_integers_satisfy_condition :
  ∃! n : ℕ, (n > 0 ∧ 30 - 6 * n > 18) :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_satisfy_condition_l1989_198979


namespace NUMINAMATH_GPT_rhombus_area_l1989_198958

-- Define the parameters given in the problem
namespace MathProof

def perimeter (EFGH : ℝ) : ℝ := 80
def diagonal_EG (EFGH : ℝ) : ℝ := 30

-- Considering the rhombus EFGH with the given perimeter and diagonal
theorem rhombus_area : 
  ∃ (area : ℝ), area = 150 * Real.sqrt 7 ∧ 
  (perimeter EFGH = 80) ∧ 
  (diagonal_EG EFGH = 30) :=
  sorry
end MathProof

end NUMINAMATH_GPT_rhombus_area_l1989_198958


namespace NUMINAMATH_GPT_line_passes_through_vertex_count_l1989_198981

theorem line_passes_through_vertex_count :
  (∃ a : ℝ, ∀ (x : ℝ), x = 0 → (x + a = a^2)) ↔ (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_vertex_count_l1989_198981


namespace NUMINAMATH_GPT_simplify_tan_pi_over_24_add_tan_7pi_over_24_l1989_198975

theorem simplify_tan_pi_over_24_add_tan_7pi_over_24 :
  let a := Real.tan (Real.pi / 24)
  let b := Real.tan (7 * Real.pi / 24)
  a + b = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  -- conditions and definitions:
  let tan_eq_sin_div_cos := ∀ x, Real.tan x = Real.sin x / Real.cos x
  let sin_add := ∀ a b, Real.sin (a + b) = Real.sin a * Real.cos b + Real.cos a * Real.sin b
  let cos_mul := ∀ a b, Real.cos a * Real.cos b = 1 / 2 * (Real.cos (a + b) + Real.cos (a - b))
  let sin_pi_over_3 := Real.sin (Real.pi / 3) = Real.sqrt 3 / 2
  let cos_pi_over_3 := Real.cos (Real.pi / 3) = 1 / 2
  let cos_pi_over_4 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
  have cond1 := tan_eq_sin_div_cos
  have cond2 := sin_add
  have cond3 := cos_mul
  have cond4 := sin_pi_over_3
  have cond5 := cos_pi_over_3
  have cond6 := cos_pi_over_4
  sorry

end NUMINAMATH_GPT_simplify_tan_pi_over_24_add_tan_7pi_over_24_l1989_198975


namespace NUMINAMATH_GPT_right_triangle_area_l1989_198937

theorem right_triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a * a + b * b = c * c ∧ (1/2 : ℝ) * a * b = 6 := 
sorry

end NUMINAMATH_GPT_right_triangle_area_l1989_198937


namespace NUMINAMATH_GPT_lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l1989_198987

noncomputable def lucky_point (m n : ℝ) : Prop := 2 * m = 4 + n ∧ ∃ (x y : ℝ), (x = m - 1) ∧ (y = (n + 2) / 2)

theorem lucky_point_m2 :
  lucky_point 2 0 := sorry

theorem is_lucky_point_A33 :
  lucky_point 4 4 := sorry

theorem point_M_quadrant (a : ℝ) :
  lucky_point (a + 1) (2 * (2 * a - 1) - 2) → (a = 1) := sorry

end NUMINAMATH_GPT_lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l1989_198987


namespace NUMINAMATH_GPT_f_zero_f_odd_f_inequality_solution_l1989_198995

open Real

-- Given definitions
variables {f : ℝ → ℝ}
variable (h_inc : ∀ x y, x < y → f x < f y)
variable (h_eq : ∀ x y, y * f x - x * f y = x * y * (x^2 - y^2))

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := 
sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := 
sorry

-- Prove the range of x satisfying the given inequality
theorem f_inequality_solution : {x : ℝ | f (x^2 + 1) + f (3 * x - 5) < 0} = {x : ℝ | -4 < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_f_zero_f_odd_f_inequality_solution_l1989_198995


namespace NUMINAMATH_GPT_diff_baseball_soccer_l1989_198918

variable (totalBalls soccerBalls basketballs tennisBalls baseballs volleyballs : ℕ)

axiom h1 : totalBalls = 145
axiom h2 : soccerBalls = 20
axiom h3 : basketballs = soccerBalls + 5
axiom h4 : tennisBalls = 2 * soccerBalls
axiom h5 : baseballs > soccerBalls
axiom h6 : volleyballs = 30

theorem diff_baseball_soccer : baseballs - soccerBalls = 10 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_diff_baseball_soccer_l1989_198918


namespace NUMINAMATH_GPT_range_of_m_increasing_function_l1989_198922

theorem range_of_m_increasing_function :
  (2 : ℝ) ≤ m ∧ m ≤ 4 ↔ ∀ x : ℝ, (1 / 3 : ℝ) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2 ≤ 
                                 ((1 / 3 : ℝ) * (x + 1) ^ 3 - (4 * m - 1) * (x + 1) ^ 2 + (15 * m ^ 2 - 2 * m - 7) * (x + 1) + 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_increasing_function_l1989_198922


namespace NUMINAMATH_GPT_biology_marks_l1989_198944

theorem biology_marks 
  (e m p c : ℤ) 
  (avg : ℚ) 
  (marks_biology : ℤ)
  (h1 : e = 70) 
  (h2 : m = 63) 
  (h3 : p = 80)
  (h4 : c = 63)
  (h5 : avg = 68.2) 
  (h6 : avg * 5 = (e + m + p + c + marks_biology)) : 
  marks_biology = 65 :=
sorry

end NUMINAMATH_GPT_biology_marks_l1989_198944


namespace NUMINAMATH_GPT_yellow_candy_percentage_l1989_198908

variable (b : ℝ) (y : ℝ) (r : ℝ)

-- Conditions from the problem
-- 14% more yellow candies than blue candies
axiom yellow_candies : y = 1.14 * b
-- 14% fewer red candies than blue candies
axiom red_candies : r = 0.86 * b
-- Total number of candies equals 1 (or 100%)
axiom total_candies : r + b + y = 1

-- Question to prove: The percentage of yellow candies in the jar is 38%
theorem yellow_candy_percentage  : y = 0.38 := by
  sorry

end NUMINAMATH_GPT_yellow_candy_percentage_l1989_198908


namespace NUMINAMATH_GPT_Louis_ate_whole_boxes_l1989_198961

def package_size := 6
def total_lemon_heads := 54

def whole_boxes : ℕ := total_lemon_heads / package_size

theorem Louis_ate_whole_boxes :
  whole_boxes = 9 :=
by
  sorry

end NUMINAMATH_GPT_Louis_ate_whole_boxes_l1989_198961


namespace NUMINAMATH_GPT_scallops_per_pound_l1989_198901

theorem scallops_per_pound
  (cost_per_pound : ℝ)
  (scallops_per_person : ℕ)
  (number_of_people : ℕ)
  (total_cost : ℝ)
  (total_scallops : ℕ)
  (total_pounds : ℝ)
  (scallops_per_pound : ℕ)
  (h1 : cost_per_pound = 24)
  (h2 : scallops_per_person = 2)
  (h3 : number_of_people = 8)
  (h4 : total_cost = 48)
  (h5 : total_scallops = scallops_per_person * number_of_people)
  (h6 : total_pounds = total_cost / cost_per_pound)
  (h7 : scallops_per_pound = total_scallops / total_pounds) : 
  scallops_per_pound = 8 :=
sorry

end NUMINAMATH_GPT_scallops_per_pound_l1989_198901


namespace NUMINAMATH_GPT_diameter_of_tripled_volume_sphere_l1989_198941

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_tripled_volume_sphere_l1989_198941


namespace NUMINAMATH_GPT_ruby_total_classes_l1989_198947

noncomputable def average_price_per_class (pack_cost : ℝ) (pack_classes : ℕ) : ℝ :=
  pack_cost / pack_classes

noncomputable def additional_class_price (average_price : ℝ) : ℝ :=
  average_price + (1/3 * average_price)

noncomputable def total_classes_taken (total_payment : ℝ) (pack_cost : ℝ) (pack_classes : ℕ) : ℕ :=
  let avg_price := average_price_per_class pack_cost pack_classes
  let additional_price := additional_class_price avg_price
  let additional_classes := (total_payment - pack_cost) / additional_price
  pack_classes + Nat.floor additional_classes -- We use Nat.floor to convert from real to natural number of classes

theorem ruby_total_classes 
  (pack_cost : ℝ) 
  (pack_classes : ℕ) 
  (total_payment : ℝ) 
  (h_pack_cost : pack_cost = 75) 
  (h_pack_classes : pack_classes = 10) 
  (h_total_payment : total_payment = 105) :
  total_classes_taken total_payment pack_cost pack_classes = 13 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ruby_total_classes_l1989_198947


namespace NUMINAMATH_GPT_gcf_60_90_l1989_198905

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_GPT_gcf_60_90_l1989_198905


namespace NUMINAMATH_GPT_box_width_is_target_width_l1989_198910

-- Defining the conditions
def cube_volume : ℝ := 27
def box_length : ℝ := 8
def box_height : ℝ := 12
def max_cubes : ℕ := 24

-- Defining the target width we want to prove
def target_width : ℝ := 6.75

-- The proof statement
theorem box_width_is_target_width :
  ∃ w : ℝ,
  (∀ v : ℝ, (v = max_cubes * cube_volume) →
   ∀ l : ℝ, (l = box_length) →
   ∀ h : ℝ, (h = box_height) →
   v = l * w * h) →
   w = target_width :=
by
  sorry

end NUMINAMATH_GPT_box_width_is_target_width_l1989_198910


namespace NUMINAMATH_GPT_directrix_of_given_parabola_l1989_198967

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end NUMINAMATH_GPT_directrix_of_given_parabola_l1989_198967


namespace NUMINAMATH_GPT_candy_cost_l1989_198921

theorem candy_cost (candy_cost_in_cents : ℕ) (pieces : ℕ) (dollar_in_cents : ℕ)
  (h1 : candy_cost_in_cents = 2) (h2 : pieces = 500) (h3 : dollar_in_cents = 100) :
  (pieces * candy_cost_in_cents) / dollar_in_cents = 10 :=
by
  sorry

end NUMINAMATH_GPT_candy_cost_l1989_198921


namespace NUMINAMATH_GPT_percentage_of_apples_sold_l1989_198969

variables (A P : ℝ) 

theorem percentage_of_apples_sold :
  (A = 700) →
  (A * (1 - P / 100) = 420) →
  (P = 40) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percentage_of_apples_sold_l1989_198969


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1989_198997

theorem part1_solution (x : ℝ) (h1 : (2 * x) / (x - 2) + 3 / (2 - x) = 1) : x = 1 := by
  sorry

theorem part2_solution (x : ℝ) 
  (h1 : 2 * x - 1 ≥ 3 * (x - 1)) 
  (h2 : (5 - x) / 2 < x + 3) : -1 / 3 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1989_198997


namespace NUMINAMATH_GPT_inequality_equivalence_l1989_198992

theorem inequality_equivalence (a : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
sorry

end NUMINAMATH_GPT_inequality_equivalence_l1989_198992


namespace NUMINAMATH_GPT_total_stamps_l1989_198974

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_stamps_l1989_198974


namespace NUMINAMATH_GPT_score_calculation_l1989_198985

theorem score_calculation (N : ℕ) (C : ℕ) (hN: 1 ≤ N ∧ N ≤ 20) (hC: 1 ≤ C) : 
  ∃ (score: ℕ), score = Nat.floor (N / C) :=
by sorry

end NUMINAMATH_GPT_score_calculation_l1989_198985


namespace NUMINAMATH_GPT_field_area_l1989_198940

theorem field_area (L W : ℝ) (hL : L = 20) (h_fencing : 2 * W + L = 59) :
  L * W = 390 :=
by {
  -- We will skip the proof
  sorry
}

end NUMINAMATH_GPT_field_area_l1989_198940


namespace NUMINAMATH_GPT_find_four_digit_number_l1989_198982

variable {N : ℕ} {a x y : ℕ}

theorem find_four_digit_number :
  (∃ a x y : ℕ, y < 10 ∧ 10 + a = x * y ∧ x = 9 + a ∧ N = 1000 + a + 10 * b + 100 * b ∧
  (N = 1014 ∨ N = 1035 ∨ N = 1512)) :=
by
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l1989_198982


namespace NUMINAMATH_GPT_range_of_m_l1989_198929

noncomputable def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
noncomputable def q (x m : ℝ) : Prop := (x^2 - (2*m + 1)*x + m^2 + m) ≤ 0

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, p x → q x m) ∧ ¬ (∀ x : ℝ, p x → q x m) ↔ m ∈ Set.Ico (-2 : ℝ) (-1) ∪ Set.Ioc 0 (1 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1989_198929


namespace NUMINAMATH_GPT_gcd_f_50_51_l1989_198999

def f (x : ℤ) : ℤ :=
  x ^ 2 - 2 * x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end NUMINAMATH_GPT_gcd_f_50_51_l1989_198999


namespace NUMINAMATH_GPT_tricia_age_l1989_198946

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end NUMINAMATH_GPT_tricia_age_l1989_198946


namespace NUMINAMATH_GPT_tangents_of_convex_quad_l1989_198939

theorem tangents_of_convex_quad (
  α β γ δ : ℝ
) (m : ℝ) (h₀ : α + β + γ + δ = 2 * Real.pi) (h₁ : 0 < α ∧ α < Real.pi) (h₂ : 0 < β ∧ β < Real.pi) 
  (h₃ : 0 < γ ∧ γ < Real.pi) (h₄ : 0 < δ ∧ δ < Real.pi) (t1 : Real.tan α = m) :
  ¬ (Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) :=
sorry

end NUMINAMATH_GPT_tangents_of_convex_quad_l1989_198939


namespace NUMINAMATH_GPT_value_of_a_plus_d_l1989_198957

variable (a b c d : ℝ)

theorem value_of_a_plus_d (h1 : a + b = 4) (h2 : b + c = 7) (h3 : c + d = 5) : a + d = 4 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_d_l1989_198957


namespace NUMINAMATH_GPT_slices_per_pizza_l1989_198989

theorem slices_per_pizza (num_pizzas num_slices : ℕ) (h1 : num_pizzas = 17) (h2 : num_slices = 68) :
  (num_slices / num_pizzas) = 4 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_pizza_l1989_198989


namespace NUMINAMATH_GPT_find_triangle_l1989_198923

theorem find_triangle : ∀ (triangle : ℕ), (∀ (d : ℕ), 0 ≤ d ∧ d ≤ 9) → (5 * 3 + triangle = 12 * triangle + 4) → triangle = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_l1989_198923


namespace NUMINAMATH_GPT_wedge_top_half_volume_l1989_198930

theorem wedge_top_half_volume (r : ℝ) (C : ℝ) (V : ℝ) : 
  (C = 18 * π) ∧ (C = 2 * π * r) ∧ (V = (4/3) * π * r^3) ∧ 
  (V / 3 / 2) = 162 * π :=
  sorry

end NUMINAMATH_GPT_wedge_top_half_volume_l1989_198930


namespace NUMINAMATH_GPT_total_bulbs_needed_l1989_198916

-- Definitions according to the conditions.
variables (T S M L XL : ℕ)

-- Conditions
variables (cond1 : L = 2 * M)
variables (cond2 : S = 5 * M / 4)  -- since 1.25M = 5/4M
variables (cond3 : XL = S - T)
variables (cond4 : 4 * T = 3 * M) -- equivalent to T / M = 3 / 4
variables (cond5 : 2 * S + 3 * M = 4 * L + 5 * XL)
variables (cond6 : XL = 14)

-- Prove total bulbs needed
theorem total_bulbs_needed :
  T + 2 * S + 3 * M + 4 * L + 5 * XL = 469 :=
sorry

end NUMINAMATH_GPT_total_bulbs_needed_l1989_198916


namespace NUMINAMATH_GPT_outdoor_section_width_l1989_198971

theorem outdoor_section_width (Length Area Width : ℝ) (h1 : Length = 6) (h2 : Area = 24) : Width = 4 :=
by
  -- We'll use "?" to represent the parts that need to be inferred by the proof assistant. 
  sorry

end NUMINAMATH_GPT_outdoor_section_width_l1989_198971


namespace NUMINAMATH_GPT_part1_part2_l1989_198942

variables (a b : ℝ)

theorem part1 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) : ab ≥ 16 :=
sorry

theorem part2 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) :
  ∃ (a b : ℝ), a = 7 ∧ b = 5 / 2 ∧ a + 4 * b = 17 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1989_198942


namespace NUMINAMATH_GPT_zoo_revenue_is_61_l1989_198970

def children_monday := 7
def children_tuesday := 4
def adults_monday := 5
def adults_tuesday := 2
def child_ticket_cost := 3
def adult_ticket_cost := 4

def total_children := children_monday + children_tuesday
def total_adults := adults_monday + adults_tuesday

def total_children_cost := total_children * child_ticket_cost
def total_adult_cost := total_adults * adult_ticket_cost

def total_zoo_revenue := total_children_cost + total_adult_cost

theorem zoo_revenue_is_61 : total_zoo_revenue = 61 := by
  sorry

end NUMINAMATH_GPT_zoo_revenue_is_61_l1989_198970


namespace NUMINAMATH_GPT_arthur_bakes_muffins_l1989_198990

-- Definitions of the conditions
def james_muffins : ℚ := 9.58333333299999
def multiplier : ℚ := 12.0

-- Statement of the problem
theorem arthur_bakes_muffins : 
  abs (multiplier * james_muffins - 115) < 1 :=
by
  sorry

end NUMINAMATH_GPT_arthur_bakes_muffins_l1989_198990


namespace NUMINAMATH_GPT_mod_abc_eq_zero_l1989_198960

open Nat

theorem mod_abc_eq_zero
    (a b c : ℕ)
    (h1 : (a + 2 * b + 3 * c) % 9 = 1)
    (h2 : (2 * a + 3 * b + c) % 9 = 2)
    (h3 : (3 * a + b + 2 * c) % 9 = 3) :
    (a * b * c) % 9 = 0 := by
  sorry

end NUMINAMATH_GPT_mod_abc_eq_zero_l1989_198960


namespace NUMINAMATH_GPT_weight_of_replaced_student_l1989_198931

variable (W : ℝ) -- total weight of the original 10 students
variable (new_student_weight : ℝ := 60) -- weight of the new student
variable (weight_decrease_per_student : ℝ := 6) -- average weight decrease per student

theorem weight_of_replaced_student (replaced_student_weight : ℝ) :
  (W - replaced_student_weight + new_student_weight = W - 10 * weight_decrease_per_student) →
  replaced_student_weight = 120 := by
  sorry

end NUMINAMATH_GPT_weight_of_replaced_student_l1989_198931


namespace NUMINAMATH_GPT_battery_current_l1989_198963

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end NUMINAMATH_GPT_battery_current_l1989_198963


namespace NUMINAMATH_GPT_modulus_of_complex_z_l1989_198954

open Complex

theorem modulus_of_complex_z (z : ℂ) (h : z * (2 - 3 * I) = 6 + 4 * I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 :=
by
  sorry

end NUMINAMATH_GPT_modulus_of_complex_z_l1989_198954


namespace NUMINAMATH_GPT_other_endpoint_sum_l1989_198903

def endpoint_sum (A B M : (ℝ × ℝ)) : ℝ := 
  let (Ax, Ay) := A
  let (Mx, My) := M
  let (Bx, By) := B
  Bx + By

theorem other_endpoint_sum (A M : (ℝ × ℝ)) (hA : A = (6, 1)) (hM : M = (5, 7)) :
  ∃ B : (ℝ × ℝ), endpoint_sum A B M = 17 :=
by
  use (4, 13)
  rw [endpoint_sum, hA, hM]
  simp
  sorry

end NUMINAMATH_GPT_other_endpoint_sum_l1989_198903


namespace NUMINAMATH_GPT_part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l1989_198935

-- Definitions of the sets and conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part 1
theorem part1_union (a : ℝ) (ha : a = 1) : 
  A ∪ B a = { x | -4 < x ∧ x ≤ 3 } :=
sorry

theorem part1_intersection_complement (a : ℝ) (ha : a = 1) : 
  A ∩ (U \ B a) = { x | -4 < x ∧ x < 0 } :=
sorry

-- Part 2
theorem part2_necessary_sufficient_condition (a : ℝ) : 
  (∀ x, x ∈ B a ↔ x ∈ A) ↔ (-3 < a ∧ a < -1) :=
sorry

end NUMINAMATH_GPT_part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l1989_198935
