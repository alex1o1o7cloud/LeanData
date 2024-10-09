import Mathlib

namespace factor_difference_of_squares_example_l649_64923

theorem factor_difference_of_squares_example :
    (m : ℝ) → (m ^ 2 - 4 = (m + 2) * (m - 2)) :=
by
    intro m
    sorry

end factor_difference_of_squares_example_l649_64923


namespace subset_S_A_inter_B_nonempty_l649_64988

open Finset

-- Definitions of sets A and B
def A : Finset ℕ := {1, 2, 3, 4, 5, 6}
def B : Finset ℕ := {4, 5, 6, 7, 8}

-- Definition of the subset S and its condition
def S : Finset ℕ := {5, 6}

-- The statement to be proved
theorem subset_S_A_inter_B_nonempty : S ⊆ A ∧ S ∩ B ≠ ∅ :=
by {
  sorry -- proof to be provided
}

end subset_S_A_inter_B_nonempty_l649_64988


namespace non_degenerate_ellipse_l649_64997

theorem non_degenerate_ellipse (k : ℝ) : (∃ (x y : ℝ), x^2 + 4*y^2 - 10*x + 56*y = k) ↔ k > -221 :=
sorry

end non_degenerate_ellipse_l649_64997


namespace at_least_two_pass_written_test_expectation_number_of_admission_advantage_l649_64926

noncomputable def probability_of_passing_written_test_A : ℝ := 0.4
noncomputable def probability_of_passing_written_test_B : ℝ := 0.8
noncomputable def probability_of_passing_written_test_C : ℝ := 0.5

noncomputable def probability_of_passing_interview_A : ℝ := 0.8
noncomputable def probability_of_passing_interview_B : ℝ := 0.4
noncomputable def probability_of_passing_interview_C : ℝ := 0.64

theorem at_least_two_pass_written_test :
  (probability_of_passing_written_test_A * probability_of_passing_written_test_B * (1 - probability_of_passing_written_test_C) +
  probability_of_passing_written_test_A * (1 - probability_of_passing_written_test_B) * probability_of_passing_written_test_C +
  (1 - probability_of_passing_written_test_A) * probability_of_passing_written_test_B * probability_of_passing_written_test_C +
  probability_of_passing_written_test_A * probability_of_passing_written_test_B * probability_of_passing_written_test_C = 0.6) :=
sorry

theorem expectation_number_of_admission_advantage :
  (3 * (probability_of_passing_written_test_A * probability_of_passing_interview_A) +
  3 * (probability_of_passing_written_test_B * probability_of_passing_interview_B) +
  3 * (probability_of_passing_written_test_C * probability_of_passing_interview_C) = 0.96) :=
sorry

end at_least_two_pass_written_test_expectation_number_of_admission_advantage_l649_64926


namespace original_price_of_cycle_l649_64906

theorem original_price_of_cycle (SP : ℝ) (P : ℝ) (loss_percent : ℝ) 
  (h_loss : loss_percent = 18) 
  (h_SP : SP = 1148) 
  (h_eq : SP = (1 - loss_percent / 100) * P) : 
  P = 1400 := 
by 
  sorry

end original_price_of_cycle_l649_64906


namespace equal_sets_implies_value_of_m_l649_64917

theorem equal_sets_implies_value_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {3, m}) (hB : B = {3 * m, 3}) (hAB : A = B) : m = 0 :=
by
  -- Proof goes here
  sorry

end equal_sets_implies_value_of_m_l649_64917


namespace negation_P_eq_Q_l649_64985

-- Define the proposition P: For any x ∈ ℝ, x^2 - 2x - 3 ≤ 0
def P : Prop := ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0

-- Define its negation which is the proposition Q
def Q : Prop := ∃ x : ℝ, x^2 - 2*x - 3 > 0

-- Prove that the negation of P is equivalent to Q
theorem negation_P_eq_Q : ¬P = Q :=
  by
  sorry

end negation_P_eq_Q_l649_64985


namespace trigonometric_identity_l649_64930

theorem trigonometric_identity 
  (α m : ℝ) 
  (h : Real.tan (α / 2) = m) :
  (1 - 2 * (Real.sin (α / 2))^2) / (1 + Real.sin α) = (1 - m) / (1 + m) :=
by
  sorry

end trigonometric_identity_l649_64930


namespace calculate_n_l649_64938

theorem calculate_n (n : ℕ) : 3^n = 3 * 9^5 * 81^3 -> n = 23 :=
by
  -- Proof omitted
  sorry

end calculate_n_l649_64938


namespace centers_distance_ABC_l649_64984

-- Define triangle ABC with the given properties
structure RightTriangle (ABC : Type) :=
(angle_A : ℝ)
(angle_C : ℝ)
(shorter_leg : ℝ)

-- Given: angle A is 30 degrees, angle C is 90 degrees, and shorter leg AC is 1
def triangle_ABC : RightTriangle ℝ := {
  angle_A := 30,
  angle_C := 90,
  shorter_leg := 1
}

-- Define the distance between the centers of the inscribed circles of triangles ACD and BCD
noncomputable def distance_between_centers (ABC : RightTriangle ℝ): ℝ :=
  sorry  -- placeholder for the actual proof

-- Example problem statement
theorem centers_distance_ABC (ABC : RightTriangle ℝ) (h_ABC : ABC = triangle_ABC) :
  distance_between_centers ABC = (Real.sqrt 3 - 1) / Real.sqrt 2 :=
sorry

end centers_distance_ABC_l649_64984


namespace jim_age_is_55_l649_64981

-- Definitions of the conditions
def jim_age (t : ℕ) : ℕ := 3 * t + 10

def sum_ages (j t : ℕ) : Prop := j + t = 70

-- Statement of the proof problem
theorem jim_age_is_55 : ∃ t : ℕ, jim_age t = 55 ∧ sum_ages (jim_age t) t :=
by
  sorry

end jim_age_is_55_l649_64981


namespace boat_speed_in_still_water_l649_64977

theorem boat_speed_in_still_water (v s : ℝ) (h1 : v + s = 15) (h2 : v - s = 7) : v = 11 := 
by
  sorry

end boat_speed_in_still_water_l649_64977


namespace calculate_altitude_l649_64905

-- Define the conditions
def Speed_up : ℕ := 18
def Speed_down : ℕ := 24
def Avg_speed : ℝ := 20.571428571428573

-- Define what we want to prove
theorem calculate_altitude : 
  2 * Speed_up * Speed_down / (Speed_up + Speed_down) = Avg_speed →
  (864 : ℝ) / 2 = 432 :=
by
  sorry

end calculate_altitude_l649_64905


namespace isosceles_trapezoid_perimeter_l649_64957

/-- In an isosceles trapezoid ABCD with bases AB = 10 units and CD = 18 units, 
and height from AB to CD is 4 units, the perimeter of ABCD is 28 + 8 * sqrt(2) units. -/
theorem isosceles_trapezoid_perimeter :
  ∃ (A B C D : Type) (AB CD AD BC h : ℝ), 
      AB = 10 ∧ 
      CD = 18 ∧ 
      AD = BC ∧ 
      h = 4 →
      ∀ (P : ℝ), P = AB + BC + CD + DA → 
      P = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end isosceles_trapezoid_perimeter_l649_64957


namespace find_a_plus_b_l649_64992

noncomputable def f (a b x : ℝ) : ℝ := (a * x^3) / 3 - b * x^2 + a^2 * x - 1 / 3
noncomputable def f_prime (a b x : ℝ) : ℝ := a * x^2 - 2 * b * x + a^2

theorem find_a_plus_b 
  (a b : ℝ)
  (h_deriv : f_prime a b 1 = 0)
  (h_extreme : f a b 1 = 0) :
  a + b = -7 / 9 := 
sorry

end find_a_plus_b_l649_64992


namespace local_min_c_value_l649_64904

-- Definition of the function f(x) with its local minimum condition
def f (x c : ℝ) := x * (x - c)^2

-- Theorem stating that for the given function f(x) to have a local minimum at x = 1, the value of c must be 1
theorem local_min_c_value (c : ℝ) (h : ∀ ε > 0, f 1 ε < f c ε) : c = 1 := sorry

end local_min_c_value_l649_64904


namespace cost_price_percentage_l649_64962

theorem cost_price_percentage (CP SP : ℝ) (h1 : SP = 4 * CP) : (CP / SP) * 100 = 25 :=
by
  sorry

end cost_price_percentage_l649_64962


namespace ducks_cows_legs_l649_64909

theorem ducks_cows_legs (D C : ℕ) (L H X : ℤ)
  (hC : C = 13)
  (hL : L = 2 * D + 4 * C)
  (hH : H = D + C)
  (hCond : L = 3 * H + X) : X = 13 := by
  sorry

end ducks_cows_legs_l649_64909


namespace project_completion_l649_64976

theorem project_completion (x : ℕ) :
  (21 - x) * (1 / 12 : ℚ) + x * (1 / 30 : ℚ) = 1 → x = 15 :=
by
  sorry

end project_completion_l649_64976


namespace green_ball_probability_l649_64951

-- Defining the number of red and green balls in each container
def containerI_red : ℕ := 10
def containerI_green : ℕ := 5

def containerII_red : ℕ := 3
def containerII_green : ℕ := 6

def containerIII_red : ℕ := 3
def containerIII_green : ℕ := 6

-- Probability of selecting any container
def prob_container : ℚ := 1 / 3

-- Defining the probabilities of drawing a green ball from each container
def prob_green_I : ℚ := containerI_green / (containerI_red + containerI_green)
def prob_green_II : ℚ := containerII_green / (containerII_red + containerII_green)
def prob_green_III : ℚ := containerIII_green / (containerIII_red + containerIII_green)

-- Law of total probability
def prob_green_total : ℚ :=
  prob_container * prob_green_I +
  prob_container * prob_green_II +
  prob_container * prob_green_III

-- The mathematical statement to be proven
theorem green_ball_probability :
  prob_green_total = 5 / 9 := by
  sorry

end green_ball_probability_l649_64951


namespace triangle_interior_angle_contradiction_l649_64978

theorem triangle_interior_angle_contradiction :
  (∀ (A B C : ℝ), A + B + C = 180 ∧ A > 60 ∧ B > 60 ∧ C > 60 → false) :=
by
  sorry

end triangle_interior_angle_contradiction_l649_64978


namespace probability_of_reaching_3_1_without_2_0_in_8_steps_l649_64901

theorem probability_of_reaching_3_1_without_2_0_in_8_steps :
  let n_total := 1680
  let invalid := 30
  let total := n_total - invalid
  let q := total / 4^8
  let gcd := Nat.gcd total 65536
  let m := total / gcd
  let n := 65536 / gcd
  (m + n = 11197) :=
by
  sorry

end probability_of_reaching_3_1_without_2_0_in_8_steps_l649_64901


namespace rod_length_of_weight_l649_64960

theorem rod_length_of_weight (w10 : ℝ) (wL : ℝ) (L : ℝ) (h1 : w10 = 23.4) (h2 : wL = 14.04) : L = 6 :=
by
  sorry

end rod_length_of_weight_l649_64960


namespace solve_for_x_l649_64983

theorem solve_for_x (x : ℝ) (h : 3034 - 1002 / x = 2984) : x = 20.04 :=
by
  sorry

end solve_for_x_l649_64983


namespace instantaneous_velocity_at_2_l649_64931

def s (t : ℝ) : ℝ := 3 * t^2 + t

theorem instantaneous_velocity_at_2 : (deriv s 2) = 13 :=
by
  sorry

end instantaneous_velocity_at_2_l649_64931


namespace john_initial_investment_in_alpha_bank_is_correct_l649_64940

-- Definition of the problem conditions
def initial_investment : ℝ := 2000
def alpha_rate : ℝ := 0.04
def beta_rate : ℝ := 0.06
def final_amount : ℝ := 2398.32
def years : ℕ := 3

-- Alpha Bank growth factor after 3 years
def alpha_growth_factor : ℝ := (1 + alpha_rate) ^ years

-- Beta Bank growth factor after 3 years
def beta_growth_factor : ℝ := (1 + beta_rate) ^ years

-- The main theorem
theorem john_initial_investment_in_alpha_bank_is_correct (x : ℝ) 
  (hx : x * alpha_growth_factor + (initial_investment - x) * beta_growth_factor = final_amount) : 
  x = 246.22 :=
sorry

end john_initial_investment_in_alpha_bank_is_correct_l649_64940


namespace remainder_of_sum_l649_64971

theorem remainder_of_sum :
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 :=
by
  sorry

end remainder_of_sum_l649_64971


namespace intersection_P_Q_l649_64908

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

theorem intersection_P_Q : P ∩ Q = {y | y ≤ 2} :=
sorry

end intersection_P_Q_l649_64908


namespace TV_cost_exact_l649_64942

theorem TV_cost_exact (savings : ℝ) (fraction_furniture : ℝ) (fraction_tv : ℝ) (original_savings : ℝ) (tv_cost : ℝ) :
  savings = 880 →
  fraction_furniture = 3 / 4 →
  fraction_tv = 1 - fraction_furniture →
  tv_cost = fraction_tv * savings →
  tv_cost = 220 :=
by
  sorry

end TV_cost_exact_l649_64942


namespace locus_of_feet_of_perpendiculars_from_focus_l649_64990

def parabola_locus (p : ℝ) : Prop :=
  ∀ x y : ℝ, (y^2 = (p / 2) * x)

theorem locus_of_feet_of_perpendiculars_from_focus (p : ℝ) :
    parabola_locus p :=
by
  sorry

end locus_of_feet_of_perpendiculars_from_focus_l649_64990


namespace speed_of_stream_l649_64993

variable (v : ℝ)

theorem speed_of_stream (h : (64 / (24 + v)) = (32 / (24 - v))) : v = 8 := 
by
  sorry

end speed_of_stream_l649_64993


namespace intersection_A_B_l649_64967

-- Define the sets A and the function f
def A : Set ℤ := {-2, 0, 2}
def f (x : ℤ) : ℤ := |x|

-- Define the set B as the image of A under the function f
def B : Set ℤ := {b | ∃ a ∈ A, f a = b}

-- State the property that every element in B has a pre-image in A
axiom B_has_preimage : ∀ b ∈ B, ∃ a ∈ A, f a = b

-- The theorem we want to prove
theorem intersection_A_B : A ∩ B = {0, 2} :=
by sorry

end intersection_A_B_l649_64967


namespace range_of_a_l649_64979

-- Problem statement and conditions definition
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def Q (a : ℝ) : Prop := (5 - 2 * a) > 1

-- Proof problem statement
theorem range_of_a (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
sorry

end range_of_a_l649_64979


namespace sum_of_a_b_vert_asymptotes_l649_64937

theorem sum_of_a_b_vert_asymptotes (a b : ℝ) 
  (h1 : ∀ x : ℝ, x = -1 → x^2 + a * x + b = 0) 
  (h2 : ∀ x : ℝ, x = 3 → x^2 + a * x + b = 0) : 
  a + b = -5 :=
sorry

end sum_of_a_b_vert_asymptotes_l649_64937


namespace simplify_fraction_l649_64925

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 3 + 1) + 3 / (Real.sqrt 5 - 2))) = 2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11) :=
by
  sorry

end simplify_fraction_l649_64925


namespace number_is_square_l649_64944

theorem number_is_square (x y : ℕ) : (∃ n : ℕ, (1100 * x + 11 * y = n^2)) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_is_square_l649_64944


namespace squirrel_journey_time_l649_64936

theorem squirrel_journey_time : 
  (let distance := 2
  let speed_to_tree := 3
  let speed_return := 2
  let time_to_tree := distance / speed_to_tree
  let time_return := distance / speed_return
  let total_time := (time_to_tree + time_return) * 60
  total_time = 100) :=
by
  sorry

end squirrel_journey_time_l649_64936


namespace find_range_of_m_l649_64964

theorem find_range_of_m:
  (∀ x: ℝ, ¬ ∃ x: ℝ, x^2 + (m - 3) * x + 1 = 0) →
  (∀ y: ℝ, ¬ ∀ y: ℝ, x^2 + y^2 / (m - 1) = 1) → 
  1 < m ∧ m ≤ 2 :=
by
  sorry

end find_range_of_m_l649_64964


namespace original_price_double_value_l649_64915

theorem original_price_double_value :
  ∃ (P : ℝ), P + 0.30 * P = 351 ∧ 2 * P = 540 :=
by
  sorry

end original_price_double_value_l649_64915


namespace find_vector_c_l649_64914

def angle_equal_coordinates (c : ℝ × ℝ) : Prop :=
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (1, -Real.sqrt 3)
  let cos_angle_ab (u v : ℝ × ℝ) : ℝ :=
    (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1^2 + u.2^2) * Real.sqrt (v.1^2 + v.2^2))
  cos_angle_ab c a = cos_angle_ab c b

theorem find_vector_c :
  angle_equal_coordinates (Real.sqrt 3, -1) :=
sorry

end find_vector_c_l649_64914


namespace min_value_of_expression_l649_64953

theorem min_value_of_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * (a + c) = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l649_64953


namespace min_value_of_sum_of_squares_l649_64970

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x + 2 * y + z = 1) : 
    x^2 + y^2 + z^2 ≥ (1 / 6) := 
  sorry

noncomputable def min_val_xy2z (x y z : ℝ) (h : x + 2 * y + z = 1) : ℝ :=
  if h_sq : x^2 + y^2 + z^2 = 1 / 6 then (x^2 + y^2 + z^2) else if x = 1 / 6 ∧ z = 1 / 6 ∧ y = 1 / 3 then 1 / 6 else (1 / 6)

example (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + y^2 + z^2 = min_val_xy2z x y z h :=
  sorry

end min_value_of_sum_of_squares_l649_64970


namespace log_expression_zero_l649_64932

theorem log_expression_zero (log : Real → Real) (exp : Real → Real) (log_mul : ∀ a b, log (a * b) = log a + log b) :
  log 2 ^ 2 + log 2 * log 50 - log 4 = 0 :=
by
  sorry

end log_expression_zero_l649_64932


namespace arithmetic_sequence_problem_l649_64921

theorem arithmetic_sequence_problem (a₁ d S₁₀ : ℝ) (h1 : d < 0) (h2 : (a₁ + d) * (a₁ + 3 * d) = 12) 
  (h3 : (a₁ + d) + (a₁ + 3 * d) = 8) (h4 : S₁₀ = 10 * a₁ + 10 * (10 - 1) / 2 * d) : 
  true := sorry

end arithmetic_sequence_problem_l649_64921


namespace tangerine_and_orange_percentage_l649_64986

-- Given conditions
def initial_apples := 9
def initial_oranges := 5
def initial_tangerines := 17
def initial_grapes := 12
def initial_kiwis := 7

def removed_oranges := 2
def removed_tangerines := 10
def removed_grapes := 4
def removed_kiwis := 3

def added_oranges := 3
def added_tangerines := 6

-- Computed values based on the initial conditions and changes
def remaining_apples := initial_apples
def remaining_oranges := initial_oranges - removed_oranges + added_oranges
def remaining_tangerines := initial_tangerines - removed_tangerines + added_tangerines
def remaining_grapes := initial_grapes - removed_grapes
def remaining_kiwis := initial_kiwis - removed_kiwis

def total_remaining_fruits := remaining_apples + remaining_oranges + remaining_tangerines + remaining_grapes + remaining_kiwis
def total_citrus_fruits := remaining_oranges + remaining_tangerines

-- Statement to prove
def citrus_percentage := (total_citrus_fruits : ℚ) / total_remaining_fruits * 100

theorem tangerine_and_orange_percentage : citrus_percentage = 47.5 := by
  sorry

end tangerine_and_orange_percentage_l649_64986


namespace loss_percentage_is_ten_l649_64994

variable (CP SP SP_new : ℝ)  -- introduce the cost price, selling price, and new selling price as variables

theorem loss_percentage_is_ten
  (h1 : CP = 2000)
  (h2 : SP_new = CP + 80)
  (h3 : SP_new = SP + 280)
  (h4 : SP = CP - (L / 100 * CP)) : L = 10 :=
by
  -- proof goes here
  sorry

end loss_percentage_is_ten_l649_64994


namespace distance_with_father_l649_64947

variable (total_distance driven_with_mother driven_with_father: ℝ)

theorem distance_with_father :
  total_distance = 0.67 ∧ driven_with_mother = 0.17 → driven_with_father = 0.50 := 
by
  sorry

end distance_with_father_l649_64947


namespace marching_band_members_l649_64975

theorem marching_band_members :
  ∃ (n : ℕ), 100 < n ∧ n < 200 ∧
             n % 4 = 1 ∧
             n % 5 = 2 ∧
             n % 7 = 3 :=
  by sorry

end marching_band_members_l649_64975


namespace river_depth_mid_July_l649_64968

theorem river_depth_mid_July :
  let d_May := 5
  let d_June := d_May + 10
  let d_July := 3 * d_June
  d_July = 45 :=
by
  sorry

end river_depth_mid_July_l649_64968


namespace range_of_a_l649_64911

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 : ℝ, (x1 + x2 = -2 * a) ∧ (x1 * x2 = 1) ∧ (x1 < 0) ∧ (x2 < 0)) ↔ (a ≥ 1) :=
by
  sorry

end range_of_a_l649_64911


namespace problem_proof_l649_64989

-- Assume definitions for lines and planes, and their relationships like parallel and perpendicular exist.

variables (m n : Line) (α β : Plane)

-- Define conditions
def line_is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_is_parallel_to_line (l1 l2 : Line) : Prop := sorry
def planes_are_perpendicular (p1 p2 : Plane) : Prop := sorry

-- Problem statement
theorem problem_proof :
  (line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n α) → 
  (line_is_parallel_to_line m n) ∧
  ((line_is_perpendicular_to_plane m α) ∧ (line_is_perpendicular_to_plane n β) ∧ (line_is_perpendicular_to_plane m n) → 
  (planes_are_perpendicular α β)) := 
sorry

end problem_proof_l649_64989


namespace consecutive_even_numbers_sum_is_3_l649_64965

-- Definitions from the conditions provided
def consecutive_even_numbers := [80, 82, 84]
def sum_of_numbers := 246

-- The problem is to prove that there are 3 consecutive even numbers summing up to 246
theorem consecutive_even_numbers_sum_is_3 :
  (consecutive_even_numbers.sum = sum_of_numbers) → consecutive_even_numbers.length = 3 :=
by
  sorry

end consecutive_even_numbers_sum_is_3_l649_64965


namespace example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l649_64980

-- Define what it means to be a three-digit number using only two distinct digits
def two_digit_natural (d1 d2 : ℕ) (n : ℕ) : Prop :=
  (∀ (d : ℕ), d ∈ n.digits 10 → d = d1 ∨ d = d2) ∧ 100 ≤ n ∧ n < 1000

-- State the main theorem
theorem example_of_four_three_digit_numbers_sum_2012_two_digits_exists :
  ∃ a b c d : ℕ, 
    two_digit_natural 3 5 a ∧
    two_digit_natural 3 5 b ∧
    two_digit_natural 3 5 c ∧
    two_digit_natural 3 5 d ∧
    a + b + c + d = 2012 :=
by
  sorry

end example_of_four_three_digit_numbers_sum_2012_two_digits_exists_l649_64980


namespace total_candies_in_third_set_l649_64928

theorem total_candies_in_third_set :
  ∀ (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ),
  L1 + L2 + L3 = S1 + S2 + S3 → 
  L1 + L2 + L3 = M1 + M2 + M3 → 
  S1 = M1 → 
  L1 = S1 + 7 → 
  L2 = S2 → 
  M2 = L2 - 15 → 
  L3 = 0 → 
  S3 = 7 → 
  M3 = 22 → 
  L3 + S3 + M3 = 29 :=
by
  intros L1 L2 L3 S1 S2 S3 M1 M2 M3 h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end total_candies_in_third_set_l649_64928


namespace total_marbles_l649_64919

theorem total_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := by
  sorry

end total_marbles_l649_64919


namespace remainder_when_divided_by_x_minus_4_l649_64999

noncomputable def f (x : ℝ) : ℝ := x^4 - 9 * x^3 + 21 * x^2 + x - 18

theorem remainder_when_divided_by_x_minus_4 : f 4 = 2 :=
by
  sorry

end remainder_when_divided_by_x_minus_4_l649_64999


namespace find_p_l649_64950

theorem find_p (m n p : ℝ)
  (h1 : m = 5 * n + 5)
  (h2 : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by
  sorry

end find_p_l649_64950


namespace problem_statement_l649_64954

theorem problem_statement (g : ℝ → ℝ) :
  (∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - x + 2) →
  (∃ m t : ℝ, m = 1 ∧ t = 3 ∧ m * t = 3) :=
sorry

end problem_statement_l649_64954


namespace banana_group_size_l649_64952

theorem banana_group_size (bananas groups : ℕ) (h1 : bananas = 407) (h2 : groups = 11) : bananas / groups = 37 :=
by sorry

end banana_group_size_l649_64952


namespace talias_fathers_age_l649_64934

-- Definitions based on the conditions
variable (T M F : ℕ)

-- The conditions
axiom h1 : T + 7 = 20
axiom h2 : M = 3 * T
axiom h3 : F + 3 = M

-- Goal: Prove that Talia's father (F) is currently 36 years old
theorem talias_fathers_age : F = 36 :=
by
  sorry

end talias_fathers_age_l649_64934


namespace number_of_bookshelves_l649_64933

theorem number_of_bookshelves (books_in_each: ℕ) (total_books: ℕ) (h_books_in_each: books_in_each = 56) (h_total_books: total_books = 504) : total_books / books_in_each = 9 :=
by
  sorry

end number_of_bookshelves_l649_64933


namespace largest_k_rooks_l649_64900

noncomputable def rooks_max_k (board_size : ℕ) : ℕ := 
  if board_size = 10 then 16 else 0

theorem largest_k_rooks {k : ℕ} (h : 0 ≤ k ∧ k ≤ 100) :
  k ≤ rooks_max_k 10 := 
sorry

end largest_k_rooks_l649_64900


namespace total_revenue_correct_l649_64969

noncomputable def total_ticket_revenue : ℕ :=
  let revenue_2pm := 180 * 6 + 20 * 5 + 60 * 4 + 20 * 3 + 20 * 5
  let revenue_5pm := 95 * 8 + 30 * 7 + 110 * 5 + 15 * 6
  let revenue_8pm := 122 * 10 + 74 * 7 + 29 * 8
  revenue_2pm + revenue_5pm + revenue_8pm

theorem total_revenue_correct : total_ticket_revenue = 5160 := by
  sorry

end total_revenue_correct_l649_64969


namespace percentage_of_alcohol_in_mixture_A_l649_64995

theorem percentage_of_alcohol_in_mixture_A (x : ℝ) :
  (10 * x / 100 + 5 * 50 / 100 = 15 * 30 / 100) → x = 20 :=
by
  intro h
  sorry

end percentage_of_alcohol_in_mixture_A_l649_64995


namespace min_english_score_l649_64974

theorem min_english_score (A B : ℕ) (h_avg_AB : (A + B) / 2 = 90) : 
  ∀ E : ℕ, ((A + B + E) / 3 ≥ 92) ↔ E ≥ 96 := by
  sorry

end min_english_score_l649_64974


namespace line_BC_l649_64912

noncomputable def Point := (ℝ × ℝ)
def A : Point := (-1, -4)
def l₁ := { p : Point | p.2 + 1 = 0 }
def l₂ := { p : Point | p.1 + p.2 + 1 = 0 }
def A' : Point := (-1, 2)
def A'' : Point := (3, 0)

theorem line_BC :
  ∃ (c₁ c₂ c₃ : ℝ), c₁ ≠ 0 ∨ c₂ ≠ 0 ∧
  ∀ (p : Point), (c₁ * p.1 + c₂ * p.2 + c₃ = 0) ↔ p ∈ { x | x = A ∨ x = A'' } :=
by sorry

end line_BC_l649_64912


namespace find_angle_measure_l649_64910

theorem find_angle_measure (x : ℝ) (hx : 90 - x + 40 = (1 / 2) * (180 - x)) : x = 80 :=
by
  sorry

end find_angle_measure_l649_64910


namespace range_of_a_l649_64949

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- Prove that if f(x) is decreasing on ℝ, then a must be less than or equal to -3
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 + 6 * x - 1) < 0 ) → a ≤ -3 :=
sorry

end range_of_a_l649_64949


namespace initial_avg_production_is_50_l649_64973

-- Define the initial conditions and parameters
variables (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55)

-- State that the initial total production over n days
def initial_total_production (A : ℝ) (n : ℕ) : ℝ := A * n

-- State the total production after today's production is added
def post_total_production (A : ℝ) (n : ℕ) (today_prod : ℝ) : ℝ := initial_total_production A n + today_prod

-- State the new average production calculation
def new_avg_production (n : ℕ) (new_avg : ℝ) : ℝ := new_avg * (n + 1)

-- State the main claim: Prove that the initial average daily production was 50 units per day
theorem initial_avg_production_is_50 (A : ℝ) (n : ℕ := 10) (today_prod : ℝ := 105) (new_avg : ℝ := 55) 
  (h : post_total_production A n today_prod = new_avg_production n new_avg) : 
  A = 50 := 
by {
  -- Preliminary setups (we don't need detailed proof steps here)
  sorry
}

end initial_avg_production_is_50_l649_64973


namespace integer_square_root_35_consecutive_l649_64961

theorem integer_square_root_35_consecutive : 
  ∃ n : ℕ, ∀ k : ℕ, n^2 ≤ k ∧ k < (n+1)^2 ∧ ((n + 1)^2 - n^2 = 35) ∧ (n = 17) := by 
  sorry

end integer_square_root_35_consecutive_l649_64961


namespace final_speed_is_zero_l649_64907

-- Define physical constants and conversion
def initial_speed_kmh : ℝ := 189
def initial_speed_ms : ℝ := initial_speed_kmh * 0.277778
def deceleration : ℝ := -0.5
def distance : ℝ := 4000

-- The goal is to prove the final speed is 0 m/s
theorem final_speed_is_zero (v_i : ℝ) (a : ℝ) (d : ℝ) (v_f : ℝ) 
  (hv_i : v_i = initial_speed_ms) 
  (ha : a = deceleration) 
  (hd : d = distance) 
  (h : v_f^2 = v_i^2 + 2 * a * d) : 
  v_f = 0 := 
by 
  sorry 

end final_speed_is_zero_l649_64907


namespace Eva_numbers_l649_64966

theorem Eva_numbers : ∃ (a b : ℕ), a + b = 43 ∧ a - b = 15 ∧ a = 29 ∧ b = 14 :=
by
  sorry

end Eva_numbers_l649_64966


namespace blackjack_payout_ratio_l649_64958

theorem blackjack_payout_ratio (total_payout original_bet : ℝ) (h1 : total_payout = 60) (h2 : original_bet = 40):
  total_payout - original_bet = (1 / 2) * original_bet :=
by
  sorry

end blackjack_payout_ratio_l649_64958


namespace savings_by_paying_cash_l649_64929

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end savings_by_paying_cash_l649_64929


namespace height_percentage_difference_l649_64959

theorem height_percentage_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.5384615384615385) :
  (H_B - H_A) / H_B * 100 = 35 := 
sorry

end height_percentage_difference_l649_64959


namespace find_a_l649_64916

theorem find_a (a : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f (g x) = x)
  (h2 : f x = (Real.log (x + 1) / Real.log 2) + a)
  (h3 : g 4 = 1) :
  a = 3 :=
sorry

end find_a_l649_64916


namespace ship_B_has_highest_rt_no_cars_l649_64935

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l649_64935


namespace laundry_loads_l649_64946

theorem laundry_loads (usual_price : ℝ) (sale_price : ℝ) (cost_per_load : ℝ) (total_loads_2_bottles : ℝ) :
  usual_price = 25 ∧ sale_price = 20 ∧ cost_per_load = 0.25 ∧ total_loads_2_bottles = (2 * sale_price) / cost_per_load →
  (total_loads_2_bottles / 2) = 80 :=
by
  sorry

end laundry_loads_l649_64946


namespace smallest_positive_period_tan_l649_64998

noncomputable def max_value (a b x : ℝ) := b + a * Real.sin x = -1
noncomputable def min_value (a b x : ℝ) := b - a * Real.sin x = -5
noncomputable def a_negative (a : ℝ) := a < 0

theorem smallest_positive_period_tan :
  ∃ (a b : ℝ), (max_value a b 0) ∧ (min_value a b 0) ∧ (a_negative a) →
  (1 / |3 * a + b|) * Real.pi = Real.pi / 9 :=
by
  sorry

end smallest_positive_period_tan_l649_64998


namespace find_a_value_l649_64913

theorem find_a_value (a : ℝ) (f : ℝ → ℝ)
  (h_def : ∀ x, f x = (Real.exp (x - a) - 1) * Real.log (x + 2 * a - 1))
  (h_ge_0 : ∀ x, x > 1 - 2 * a → f x ≥ 0) : a = 2 / 3 :=
by
  -- Omitted proof
  sorry

end find_a_value_l649_64913


namespace simplify_polynomial_l649_64939

variable (x : ℝ)

theorem simplify_polynomial :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 6*x^3 =
  6*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end simplify_polynomial_l649_64939


namespace three_digit_numbers_divisible_by_17_l649_64920

theorem three_digit_numbers_divisible_by_17 : ∃ (n : ℕ), n = 53 ∧ ∀ k, 100 <= 17 * k ∧ 17 * k <= 999 ↔ (6 <= k ∧ k <= 58) :=
by
  sorry

end three_digit_numbers_divisible_by_17_l649_64920


namespace binomial_multiplication_subtract_240_l649_64987

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_multiplication_subtract_240 :
  binom 10 3 * binom 8 3 - 240 = 6480 :=
by
  sorry

end binomial_multiplication_subtract_240_l649_64987


namespace exists_n_for_binomial_congruence_l649_64922

theorem exists_n_for_binomial_congruence 
  (p : ℕ) (a k : ℕ) (prime_p : Nat.Prime p) 
  (positive_a : a > 0) (positive_k : k > 0)
  (h1 : p^a < k) (h2 : k < 2 * p^a) : 
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k) % p^a = n % p^a ∧ n % p^a = k % p^a :=
by
  sorry

end exists_n_for_binomial_congruence_l649_64922


namespace bulb_probability_gt4000_l649_64927

-- Definitions given in conditions
def P_X : ℝ := 0.60
def P_Y : ℝ := 0.40
def P_gt4000_X : ℝ := 0.59
def P_gt4000_Y : ℝ := 0.65

-- The proof statement
theorem bulb_probability_gt4000 : 
  (P_X * P_gt4000_X + P_Y * P_gt4000_Y) = 0.614 :=
  by
  sorry

end bulb_probability_gt4000_l649_64927


namespace base_8_to_base_10_4652_l649_64956

def convert_base_8_to_base_10 (n : ℕ) : ℕ :=
  (4 * 8^3) + (6 * 8^2) + (5 * 8^1) + (2 * 8^0)

theorem base_8_to_base_10_4652 :
  convert_base_8_to_base_10 4652 = 2474 :=
by
  -- Skipping the proof steps
  sorry

end base_8_to_base_10_4652_l649_64956


namespace age_twice_of_father_l649_64972

theorem age_twice_of_father (S M Y : ℕ) (h₁ : S = 22) (h₂ : M = S + 24) (h₃ : M + Y = 2 * (S + Y)) : Y = 2 := by
  sorry

end age_twice_of_father_l649_64972


namespace number_of_terms_in_expansion_l649_64924

theorem number_of_terms_in_expansion :
  (∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 c1 c2 c3 : ℕ), (a1 + a2 + a3 + a4 + a5) * (b1 + b2 + b3 + b4) * (c1 + c2 + c3) = 60) :=
by
  sorry

end number_of_terms_in_expansion_l649_64924


namespace sum_of_decimals_as_fraction_l649_64903

theorem sum_of_decimals_as_fraction :
  (0.2 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) + (0.000008 : ℝ) + (0.0000009 : ℝ) = 
  (2340087 / 10000000 : ℝ) :=
sorry

end sum_of_decimals_as_fraction_l649_64903


namespace function_C_is_odd_and_decreasing_l649_64948

-- Conditions
def f (x : ℝ) : ℝ := -x^3 - x

-- Odd function condition
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Strictly decreasing condition
def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

-- The theorem we want to prove
theorem function_C_is_odd_and_decreasing : 
  is_odd f ∧ is_strictly_decreasing f :=
by
  sorry

end function_C_is_odd_and_decreasing_l649_64948


namespace negation_of_universal_proposition_l649_64902

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - 3 * x + 2 > 0)) ↔ (∃ x : ℝ, x^2 - 3 * x + 2 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l649_64902


namespace soda_difference_l649_64996

theorem soda_difference :
  let Julio_orange_bottles := 4
  let Julio_grape_bottles := 7
  let Mateo_orange_bottles := 1
  let Mateo_grape_bottles := 3
  let liters_per_bottle := 2
  let Julio_total_liters := Julio_orange_bottles * liters_per_bottle + Julio_grape_bottles * liters_per_bottle
  let Mateo_total_liters := Mateo_orange_bottles * liters_per_bottle + Mateo_grape_bottles * liters_per_bottle
  Julio_total_liters - Mateo_total_liters = 14 := by
    sorry

end soda_difference_l649_64996


namespace bob_total_earnings_l649_64991

def hourly_rate_regular := 5
def hourly_rate_overtime := 6
def regular_hours_per_week := 40

def hours_worked_week1 := 44
def hours_worked_week2 := 48

def earnings_week1 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week1 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def earnings_week2 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week2 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def total_earnings : ℕ := earnings_week1 + earnings_week2

theorem bob_total_earnings : total_earnings = 472 := by
  sorry

end bob_total_earnings_l649_64991


namespace integer_multiplication_for_ones_l649_64963

theorem integer_multiplication_for_ones :
  ∃ x : ℤ, (10^9 - 1) * x = (10^81 - 1) / 9 :=
by
  sorry

end integer_multiplication_for_ones_l649_64963


namespace tourists_escape_l649_64941

theorem tourists_escape (T : ℕ) (hT : T = 10) (hats : Fin T → Bool) (could_see : ∀ (i : Fin T), Fin (i) → Bool) :
  ∃ strategy : (Fin T → Bool), (∀ (i : Fin T), (strategy i = hats i) ∨ (strategy i ≠ hats i)) →
  (∀ (i : Fin T), (∀ (j : Fin T), i < j → strategy i = hats i) → ∃ count : ℕ, count ≥ 9 ∧ ∀ (i : Fin T), count ≥ i → strategy i = hats i) := sorry

end tourists_escape_l649_64941


namespace average_price_per_person_excluding_gratuity_l649_64918

def total_cost_with_gratuity : ℝ := 207.00
def gratuity_rate : ℝ := 0.15
def number_of_people : ℕ := 15

theorem average_price_per_person_excluding_gratuity :
  (total_cost_with_gratuity / (1 + gratuity_rate) / number_of_people) = 12.00 :=
by
  sorry

end average_price_per_person_excluding_gratuity_l649_64918


namespace range_of_m_for_second_quadrant_l649_64955

theorem range_of_m_for_second_quadrant (m : ℝ) :
  (P : ℝ × ℝ) → P = (1 + m, 3) → P.fst < 0 → m < -1 :=
by
  intro P hP hQ
  sorry

end range_of_m_for_second_quadrant_l649_64955


namespace marilyn_bottle_caps_start_l649_64982

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l649_64982


namespace profit_percentage_calc_l649_64943

noncomputable def sale_price_incl_tax : ℝ := 616
noncomputable def sales_tax_rate : ℝ := 0.10
noncomputable def cost_price : ℝ := 531.03
noncomputable def expected_profit_percentage : ℝ := 5.45

theorem profit_percentage_calc :
  let sale_price_before_tax := sale_price_incl_tax / (1 + sales_tax_rate)
  let profit := sale_price_before_tax - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = expected_profit_percentage :=
by
  sorry

end profit_percentage_calc_l649_64943


namespace chili_problem_l649_64945

def cans_of_chili (x y z : ℕ) : Prop := x + 2 * y + z = 6

def percentage_more_tomatoes_than_beans (x y z : ℕ) : ℕ :=
  100 * (z - 2 * y) / (2 * y)

theorem chili_problem (x y z : ℕ) (h1 : cans_of_chili x y z) (h2 : x = 1) (h3 : y = 1) : 
  percentage_more_tomatoes_than_beans x y z = 50 :=
by
  sorry

end chili_problem_l649_64945
