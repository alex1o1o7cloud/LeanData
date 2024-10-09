import Mathlib

namespace radius_of_regular_polygon_l339_33946

theorem radius_of_regular_polygon :
  ∃ (p : ℝ), 
        (∀ n : ℕ, 3 ≤ n → (n : ℝ) = 6) ∧ 
        (∀ s : ℝ, s = 2 → s = 2) → 
        (∀ i : ℝ, i = 720 → i = 720) →
        (∀ e : ℝ, e = 360 → e = 360) →
        p = 2 :=
by
  sorry

end radius_of_regular_polygon_l339_33946


namespace problem_solution_l339_33924

theorem problem_solution (x m : ℝ) (h1 : x ≠ 0) (h2 : x / (x^2 - m*x + 1) = 1) :
  x^3 / (x^6 - m^3 * x^3 + 1) = 1 / (3 * m^2 - 2) :=
by
  sorry

end problem_solution_l339_33924


namespace valid_votes_other_candidate_l339_33955

theorem valid_votes_other_candidate (total_votes : ℕ) (invalid_percentage : ℕ) (candidate1_percentage : ℕ) (valid_votes_other_candidate : ℕ) : 
  total_votes = 7500 → 
  invalid_percentage = 20 → 
  candidate1_percentage = 55 → 
  valid_votes_other_candidate = 2700 :=
by
  sorry

end valid_votes_other_candidate_l339_33955


namespace not_all_inequalities_hold_l339_33925

theorem not_all_inequalities_hold (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(hlt_a : a < 1) (hlt_b : b < 1) (hlt_c : c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
by
  sorry

end not_all_inequalities_hold_l339_33925


namespace disjoint_sets_condition_l339_33950

theorem disjoint_sets_condition (A B : Set ℕ) (h_disjoint: Disjoint A B) (h_union: A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a > n ∧ b > n ∧ a ≠ b ∧ 
             ((a ∈ A ∧ b ∈ A ∧ a + b ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ a + b ∈ B)) := 
by
  sorry

end disjoint_sets_condition_l339_33950


namespace Liam_savings_after_trip_and_bills_l339_33986

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l339_33986


namespace fertilizer_needed_per_acre_l339_33940

-- Definitions for the conditions
def horse_daily_fertilizer : ℕ := 5 -- Each horse produces 5 gallons of fertilizer per day.
def horses : ℕ := 80 -- Janet has 80 horses.
def days : ℕ := 25 -- It takes 25 days until all her fields are fertilized.
def total_acres : ℕ := 20 -- Janet's farmland is 20 acres.

-- Calculated intermediate values
def total_fertilizer : ℕ := horse_daily_fertilizer * horses * days -- Total fertilizer produced
def fertilizer_per_acre : ℕ := total_fertilizer / total_acres -- Fertilizer needed per acre

-- Theorem to prove
theorem fertilizer_needed_per_acre : fertilizer_per_acre = 500 := by
  sorry

end fertilizer_needed_per_acre_l339_33940


namespace tank_plastering_cost_proof_l339_33965

/-- 
Given a tank with the following dimensions:
length = 35 meters,
width = 18 meters,
depth = 10 meters.
The cost of plastering per square meter is ₹135.
Prove that the total cost of plastering the walls and bottom of the tank is ₹228,150.
-/
theorem tank_plastering_cost_proof (length width depth cost_per_sq_meter : ℕ)
  (h_length : length = 35)
  (h_width : width = 18)
  (h_depth : depth = 10)
  (h_cost_per_sq_meter : cost_per_sq_meter = 135) : 
  (2 * (length * depth) + 2 * (width * depth) + length * width) * cost_per_sq_meter = 228150 := 
by 
  -- The proof is not required as per the problem statement
  sorry

end tank_plastering_cost_proof_l339_33965


namespace f_one_zero_range_of_a_l339_33991

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x
def increasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y
def f_neg_one_zero : Prop := f (-1) = 0
def f_a_minus_half_neg : Prop := f (a - 1/2) < 0

-- Questions
theorem f_one_zero (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) : f 1 = 0 := 
sorry

theorem range_of_a (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) (h4 : f_a_minus_half_neg f a) :
  1/2 < a ∧ a < 3/2 ∨ a < -1/2 :=
sorry

end f_one_zero_range_of_a_l339_33991


namespace range_of_a_l339_33999

theorem range_of_a {a : ℝ} (h : (a^2) / 4 + 1 / 2 < 1) : -Real.sqrt 2 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_l339_33999


namespace parallel_lines_d_l339_33910

theorem parallel_lines_d (d : ℝ) : (∀ x : ℝ, -3 * x + 5 = (-6 * d) * x + 10) → d = 1 / 2 :=
by sorry

end parallel_lines_d_l339_33910


namespace no_solution_for_x6_eq_2y2_plus_2_l339_33936

theorem no_solution_for_x6_eq_2y2_plus_2 :
  ¬ ∃ (x y : ℤ), x^6 = 2 * y^2 + 2 :=
sorry

end no_solution_for_x6_eq_2y2_plus_2_l339_33936


namespace base7_to_base10_conversion_l339_33917

theorem base7_to_base10_conversion (n : ℤ) (h : n = 2 * 7^2 + 4 * 7^1 + 6 * 7^0) : n = 132 := by
  sorry

end base7_to_base10_conversion_l339_33917


namespace cubic_expression_value_l339_33998

theorem cubic_expression_value (a b c : ℝ) 
  (h1 : a + b + c = 13) 
  (h2 : ab + ac + bc = 32) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 949 := 
by
  sorry

end cubic_expression_value_l339_33998


namespace trajectory_of_moving_circle_l339_33971

-- Define the two given circles C1 and C2
def C1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 + 2)^2 + p.2^2 = 1}
def C2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 81}

-- Define a moving circle P with center P_center and radius r
structure Circle (α : Type) := 
(center : α × α) 
(radius : ℝ)

def isExternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (P.radius + 1)^2

def isInternallyTangentTo (P : Circle ℝ) (C : Set (ℝ × ℝ)) :=
  ∃ k ∈ C, (P.center.1 - k.1)^2 + (P.center.2 - k.2)^2 = (9 - P.radius)^2

-- Formulate the problem statement
theorem trajectory_of_moving_circle :
  ∀ P : Circle ℝ, 
  isExternallyTangentTo P C1 → 
  isInternallyTangentTo P C2 → 
  (P.center.1^2 / 25 + P.center.2^2 / 21 = 1) := 
sorry

end trajectory_of_moving_circle_l339_33971


namespace fraction_of_juniors_equals_seniors_l339_33928

theorem fraction_of_juniors_equals_seniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J * 7 = 4 * (J + S)) : J / S = 4 / 3 :=
sorry

end fraction_of_juniors_equals_seniors_l339_33928


namespace angle_between_line_and_plane_l339_33972

-- Define the conditions
def angle_direct_vector_normal_vector (direction_vector_angle : ℝ) := direction_vector_angle = 120

-- Define the goal to prove
theorem angle_between_line_and_plane (direction_vector_angle : ℝ) :
  angle_direct_vector_normal_vector direction_vector_angle → direction_vector_angle = 120 → 90 - (180 - direction_vector_angle) = 30 :=
by
  intros h_angle_eq angle_120
  sorry

end angle_between_line_and_plane_l339_33972


namespace man_speed_approx_l339_33908

noncomputable def speed_of_man : ℝ :=
  let L := 700    -- Length of the train in meters
  let u := 63 / 3.6  -- Speed of the train in meters per second (converted)
  let t := 41.9966402687785 -- Time taken to cross the man in seconds
  let v := (u * t - L) / t  -- Speed of the man
  v

-- The main theorem to prove that the speed of the man is approximately 0.834 m/s.
theorem man_speed_approx : abs (speed_of_man - 0.834) < 1e-3 :=
by
  -- Simplification and exact calculations will be handled by the Lean prover or could be manually done.
  sorry

end man_speed_approx_l339_33908


namespace xy_sum_l339_33902

-- Define the problem conditions
variable (x y : ℚ)
variable (h1 : 1 / x + 1 / y = 4)
variable (h2 : 1 / x - 1 / y = -8)

-- Define the theorem to prove
theorem xy_sum : x + y = -1 / 3 := by
  sorry

end xy_sum_l339_33902


namespace masha_more_cakes_l339_33907

theorem masha_more_cakes (S : ℝ) (m n : ℝ) (H1 : S > 0) (H2 : m > 0) (H3 : n > 0) 
  (H4 : 2 * S * (m + n) ≤ S * m + (1/3) * S * n) :
  m > n := 
by 
  sorry

end masha_more_cakes_l339_33907


namespace dishes_combinations_is_correct_l339_33956

-- Define the number of dishes
def num_dishes : ℕ := 15

-- Define the number of appetizers
def num_appetizers : ℕ := 5

-- Compute the total number of combinations
def combinations_of_dishes : ℕ :=
  num_dishes * num_dishes * num_appetizers

-- The theorem that states the total number of combinations is 1125
theorem dishes_combinations_is_correct :
  combinations_of_dishes = 1125 := by
  sorry

end dishes_combinations_is_correct_l339_33956


namespace sin_tan_condition_l339_33993

theorem sin_tan_condition (x : ℝ) (h : Real.sin x = (Real.sqrt 2) / 2) : ¬((∀ x, Real.sin x = (Real.sqrt 2) / 2 → Real.tan x = 1) ∧ (∀ x, Real.tan x = 1 → Real.sin x = (Real.sqrt 2) / 2)) :=
sorry

end sin_tan_condition_l339_33993


namespace remainder_when_c_divided_by_b_eq_2_l339_33922

theorem remainder_when_c_divided_by_b_eq_2 
(a b c : ℕ) 
(hb : b = 3 * a + 3) 
(hc : c = 9 * a + 11) : 
  c % b = 2 := 
sorry

end remainder_when_c_divided_by_b_eq_2_l339_33922


namespace Adam_spent_21_dollars_l339_33905

-- Define the conditions as given in the problem
def initial_money : ℕ := 91
def spent_money (x : ℕ) : Prop := (initial_money - x) * 3 = 10 * x

-- The theorem we want to prove: Adam spent 21 dollars on new books
theorem Adam_spent_21_dollars : spent_money 21 :=
by sorry

end Adam_spent_21_dollars_l339_33905


namespace a8_value_l339_33916

variable {an : ℕ → ℕ}

def S (n : ℕ) : ℕ := n ^ 2

theorem a8_value : an 8 = S 8 - S 7 := by
  sorry

end a8_value_l339_33916


namespace simplify_expr_at_sqrt6_l339_33942

noncomputable def simplifyExpression (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2)))

theorem simplify_expr_at_sqrt6 : simplifyExpression (Real.sqrt 6) = - (Real.sqrt 6) / 2 :=
by
  sorry

end simplify_expr_at_sqrt6_l339_33942


namespace shorter_leg_of_right_triangle_l339_33915

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : a^2 + b^2 = 65^2) (ha : a ≤ b) : a = 25 :=
by sorry

end shorter_leg_of_right_triangle_l339_33915


namespace correct_mean_of_values_l339_33988

variable (n : ℕ) (mu_incorrect : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (mu_correct : ℝ)

theorem correct_mean_of_values
  (h1 : n = 30)
  (h2 : mu_incorrect = 150)
  (h3 : incorrect_value = 135)
  (h4 : correct_value = 165)
  : mu_correct = 151 :=
by
  let S_incorrect := mu_incorrect * n
  let S_correct := S_incorrect - incorrect_value + correct_value
  let mu_correct := S_correct / n
  sorry

end correct_mean_of_values_l339_33988


namespace map_distance_correct_l339_33923

noncomputable def distance_on_map : ℝ :=
  let speed := 60  -- miles per hour
  let time := 6.5  -- hours
  let scale := 0.01282051282051282 -- inches per mile
  let actual_distance := speed * time -- in miles
  actual_distance * scale -- convert to inches

theorem map_distance_correct :
  distance_on_map = 5 :=
by 
  sorry

end map_distance_correct_l339_33923


namespace find_numbers_l339_33901

theorem find_numbers :
  ∃ a b : ℕ, a + b = 60 ∧ Nat.gcd a b + Nat.lcm a b = 84 :=
by
  sorry

end find_numbers_l339_33901


namespace circle_center_and_radius_sum_l339_33961

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end circle_center_and_radius_sum_l339_33961


namespace rectangle_breadth_l339_33943

theorem rectangle_breadth (l b : ℕ) (hl : l = 15) (h : l * b = 15 * b) (h2 : l - b = 10) : b = 5 := 
sorry

end rectangle_breadth_l339_33943


namespace sum_of_legs_of_right_triangle_l339_33941

theorem sum_of_legs_of_right_triangle (y : ℤ) (hyodd : y % 2 = 1) (hyp : y ^ 2 + (y + 2) ^ 2 = 17 ^ 2) :
  y + (y + 2) = 24 :=
sorry

end sum_of_legs_of_right_triangle_l339_33941


namespace probability_all_white_is_correct_l339_33914

-- Define the total number of balls
def total_balls : ℕ := 25

-- Define the number of white balls
def white_balls : ℕ := 10

-- Define the number of black balls
def black_balls : ℕ := 15

-- Define the number of balls drawn
def balls_drawn : ℕ := 4

-- Define combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to choose 4 balls from 25
def total_ways : ℕ := C total_balls balls_drawn

-- Ways to choose 4 white balls from 10 white balls
def white_ways : ℕ := C white_balls balls_drawn

-- Probability that all 4 drawn balls are white
def prob_all_white : ℚ := white_ways / total_ways

theorem probability_all_white_is_correct :
  prob_all_white = (3 : ℚ) / 181 := by
  -- Proof statements go here
  sorry

end probability_all_white_is_correct_l339_33914


namespace M_diff_N_l339_33968

def A : Set ℝ := sorry
def B : Set ℝ := sorry

def M := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def N := {y : ℝ | ∃ x : ℝ, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

-- Definition of set subtraction
def set_diff (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∉ B}

-- Given problem statement
theorem M_diff_N : set_diff M N = {x : ℝ | -3 ≤ x ∧ x < 0} := 
by
  sorry

end M_diff_N_l339_33968


namespace candies_left_l339_33970

-- Defining the given conditions
def initial_candies : Nat := 30
def eaten_candies : Nat := 23

-- Define the target statement to prove
theorem candies_left : initial_candies - eaten_candies = 7 := by
  sorry

end candies_left_l339_33970


namespace expected_volunteers_by_2022_l339_33973

noncomputable def initial_volunteers : ℕ := 1200
noncomputable def increase_2021 : ℚ := 0.15
noncomputable def increase_2022 : ℚ := 0.30

theorem expected_volunteers_by_2022 :
  (initial_volunteers * (1 + increase_2021) * (1 + increase_2022)) = 1794 := 
by
  sorry

end expected_volunteers_by_2022_l339_33973


namespace inequality_problem_l339_33982

theorem inequality_problem (x : ℝ) (h_denom : 2 * x^2 + 2 * x + 1 ≠ 0) : 
  -4 ≤ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ∧ (x^2 - 2*x - 3)/(2*x^2 + 2*x + 1) ≤ 1 :=
sorry

end inequality_problem_l339_33982


namespace domain_of_f_l339_33995

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (abs (x + 1) - 5)

theorem domain_of_f :
  {x : ℝ | x - 3 ≥ 0 ∧ abs (x + 1) - 5 ≠ 0} = {x : ℝ | (3 ≤ x ∧ x < 4) ∨ (4 < x)} :=
by
  sorry

end domain_of_f_l339_33995


namespace problem_solution_l339_33906

theorem problem_solution (a b : ℝ) (h1 : a^3 - 15 * a^2 + 25 * a - 75 = 0) (h2 : 8 * b^3 - 60 * b^2 - 310 * b + 2675 = 0) :
  a + b = 15 / 2 :=
sorry

end problem_solution_l339_33906


namespace number_of_questionnaires_from_unit_D_l339_33934

theorem number_of_questionnaires_from_unit_D 
  (a d : ℕ) 
  (total : ℕ) 
  (samples : ℕ → ℕ) 
  (h_seq : samples 0 = a ∧ samples 1 = a + d ∧ samples 2 = a + 2 * d ∧ samples 3 = a + 3 * d)
  (h_total : samples 0 + samples 1 + samples 2 + samples 3 = total)
  (h_stratified : ∀ (i : ℕ), i < 4 → samples i * 100 / total = 20 → i = 1) 
  : samples 3 = 40 := sorry

end number_of_questionnaires_from_unit_D_l339_33934


namespace roots_cubic_roots_sum_of_squares_l339_33967

variables {R : Type*} [CommRing R] {p q r s t : R}

theorem roots_cubic_roots_sum_of_squares (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) :
  r^2 + s^2 + t^2 = p^2 - 2 * q :=
sorry

end roots_cubic_roots_sum_of_squares_l339_33967


namespace tomatoes_grew_in_absence_l339_33996

def initial_tomatoes : ℕ := 36
def multiplier : ℕ := 100
def total_tomatoes_after_vacation : ℕ := initial_tomatoes * multiplier

theorem tomatoes_grew_in_absence : 
  total_tomatoes_after_vacation - initial_tomatoes = 3564 :=
by
  -- skipped proof with 'sorry'
  sorry

end tomatoes_grew_in_absence_l339_33996


namespace handshakes_at_gathering_l339_33930

noncomputable def total_handshakes : Nat :=
  let twins := 16
  let triplets := 15
  let handshakes_among_twins := twins * 14 / 2
  let handshakes_among_triplets := 0
  let cross_handshakes := twins * triplets
  handshakes_among_twins + handshakes_among_triplets + cross_handshakes

theorem handshakes_at_gathering : total_handshakes = 352 := 
by
  -- By substituting the values, we can solve and show that the total handshakes equal to 352.
  sorry

end handshakes_at_gathering_l339_33930


namespace time_wandered_l339_33966

-- Definitions and Hypotheses
def distance : ℝ := 4
def speed : ℝ := 2

-- Proof statement
theorem time_wandered : distance / speed = 2 := by
  sorry

end time_wandered_l339_33966


namespace num_ways_4x4_proof_l339_33985

-- Define a function that represents the number of ways to cut a 2x2 square
noncomputable def num_ways_2x2_cut : ℕ := 4

-- Define a function that represents the number of ways to cut a 3x3 square
noncomputable def num_ways_3x3_cut (ways_2x2 : ℕ) : ℕ :=
  ways_2x2 * 4

-- Define a function that represents the number of ways to cut a 4x4 square
noncomputable def num_ways_4x4_cut (ways_3x3 : ℕ) : ℕ :=
  ways_3x3 * 4

-- Prove the final number of ways to cut the 4x4 square into 3 L-shaped pieces and 1 small square
theorem num_ways_4x4_proof : num_ways_4x4_cut (num_ways_3x3_cut num_ways_2x2_cut) = 64 := by
  sorry

end num_ways_4x4_proof_l339_33985


namespace trips_per_student_l339_33974

theorem trips_per_student
  (num_students : ℕ := 5)
  (chairs_per_trip : ℕ := 5)
  (total_chairs : ℕ := 250)
  (T : ℕ) :
  num_students * chairs_per_trip * T = total_chairs → T = 10 :=
by
  intro h
  sorry

end trips_per_student_l339_33974


namespace simplify_fraction_l339_33935

theorem simplify_fraction (x y : ℕ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end simplify_fraction_l339_33935


namespace arithmetic_sequence_sum_l339_33984

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 + a 6 = 18) :
  S 10 = 90 :=
sorry

end arithmetic_sequence_sum_l339_33984


namespace cattle_train_left_6_hours_before_l339_33920

theorem cattle_train_left_6_hours_before 
  (Vc : ℕ) (Vd : ℕ) (T : ℕ) 
  (h1 : Vc = 56)
  (h2 : Vd = Vc - 33)
  (h3 : 12 * Vd + 12 * Vc + T * Vc = 1284) : 
  T = 6 := 
by
  sorry

end cattle_train_left_6_hours_before_l339_33920


namespace ratio_of_speeds_l339_33938

variables (v_A v_B v_C : ℝ)

-- Conditions definitions
def condition1 : Prop := v_A - v_B = 5
def condition2 : Prop := v_A + v_C = 15

-- Theorem statement (the mathematically equivalent proof problem)
theorem ratio_of_speeds (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_C) : (v_A / v_B) = 3 :=
sorry

end ratio_of_speeds_l339_33938


namespace initial_speed_is_7_l339_33957

-- Definitions based on conditions
def distance_travelled (S : ℝ) (T : ℝ) : ℝ := S * T

-- Constants from problem
def time_initial : ℝ := 6
def time_final : ℝ := 3
def speed_final : ℝ := 14

-- Theorem statement
theorem initial_speed_is_7 : ∃ S : ℝ, distance_travelled S time_initial = distance_travelled speed_final time_final ∧ S = 7 := by
  sorry

end initial_speed_is_7_l339_33957


namespace batsman_average_after_11th_inning_l339_33963

theorem batsman_average_after_11th_inning (x : ℝ) (h : 10 * x + 110 = 11 * (x + 5)) : 
    (10 * x + 110) / 11 = 60 := by
  sorry

end batsman_average_after_11th_inning_l339_33963


namespace pushups_difference_l339_33978

theorem pushups_difference :
  let David_pushups := 44
  let Zachary_pushups := 35
  David_pushups - Zachary_pushups = 9 :=
by
  -- Here we define the push-ups counts
  let David_pushups := 44
  let Zachary_pushups := 35
  -- We need to show that David did 9 more push-ups than Zachary.
  show David_pushups - Zachary_pushups = 9
  sorry

end pushups_difference_l339_33978


namespace set_membership_l339_33909

theorem set_membership :
  {m : ℤ | ∃ k : ℤ, 10 = k * (m + 1)} = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by sorry

end set_membership_l339_33909


namespace count_primes_5p2p1_minus_1_perfect_square_l339_33900

-- Define the predicate for a prime number
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Predicate for perfect square
def is_perfect_square (n : ℕ) : Prop := 
  ∃ m : ℕ, m * m = n

-- The main theorem statement
theorem count_primes_5p2p1_minus_1_perfect_square :
  (∀ p : ℕ, is_prime p → is_perfect_square (5 * p * (2^(p + 1) - 1))) → ∃! p : ℕ, is_prime p ∧ is_perfect_square (5 * p * (2^(p + 1) - 1)) :=
sorry

end count_primes_5p2p1_minus_1_perfect_square_l339_33900


namespace simplify_expression_simplify_and_evaluate_evaluate_expression_l339_33981

theorem simplify_expression (a b : ℝ) : 8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := 
by sorry

theorem simplify_and_evaluate (x y : ℝ) (h : x + y = 1/2) : 
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := 
by sorry

theorem evaluate_expression (x y : ℝ) (h : x^2 - 2 * y = 4) : -3 * x^2 + 6 * y + 2 = -10 := 
by sorry

end simplify_expression_simplify_and_evaluate_evaluate_expression_l339_33981


namespace cricket_player_innings_l339_33997

theorem cricket_player_innings (n : ℕ) (h1 : 35 * n = 35 * n) (h2 : 35 * n + 79 = 39 * (n + 1)) : n = 10 := by
  sorry

end cricket_player_innings_l339_33997


namespace heavy_cream_cost_l339_33952

theorem heavy_cream_cost
  (cost_strawberries : ℕ)
  (cost_raspberries : ℕ)
  (total_cost : ℕ)
  (cost_heavy_cream : ℕ) :
  (cost_strawberries = 3 * 2) →
  (cost_raspberries = 5 * 2) →
  (total_cost = 20) →
  (cost_heavy_cream = total_cost - (cost_strawberries + cost_raspberries)) →
  cost_heavy_cream = 4 :=
by
  sorry

end heavy_cream_cost_l339_33952


namespace rectangular_field_perimeter_l339_33945

theorem rectangular_field_perimeter (A L : ℝ) (h1 : A = 300) (h2 : L = 15) : 
  let W := A / L 
  let P := 2 * (L + W)
  P = 70 := by
  sorry

end rectangular_field_perimeter_l339_33945


namespace problem1_problem2_l339_33969

-- Problem 1: (-3xy)² * 4x² = 36x⁴y²
theorem problem1 (x y : ℝ) : ((-3 * x * y) ^ 2) * (4 * x ^ 2) = 36 * x ^ 4 * y ^ 2 := by
  sorry

-- Problem 2: (x + 2)(2x - 3) = 2x² + x - 6
theorem problem2 (x : ℝ) : (x + 2) * (2 * x - 3) = 2 * x ^ 2 + x - 6 := by
  sorry

end problem1_problem2_l339_33969


namespace min_books_borrowed_l339_33926

theorem min_books_borrowed 
    (h1 : 12 * 1 = 12) 
    (h2 : 10 * 2 = 20) 
    (h3 : 2 = 2) 
    (h4 : 32 = 32) 
    (h5 : (32 * 2 = 64))
    (h6 : ∀ x, x ≤ 11) :
    ∃ (x : ℕ), (8 * x = 32) ∧ x ≤ 11 := 
  sorry

end min_books_borrowed_l339_33926


namespace basketball_lineup_count_l339_33953

theorem basketball_lineup_count :
  (∃ (players : Finset ℕ), players.card = 15) → 
  ∃ centers power_forwards small_forwards shooting_guards point_guards sixth_men : ℕ,
  ∃ b : Fin (15) → Fin (15),
  15 * 14 * 13 * 12 * 11 * 10 = 360360 
:= by sorry

end basketball_lineup_count_l339_33953


namespace minimum_a_l339_33911

theorem minimum_a (x : ℝ) (h : ∀ x ≥ 0, x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a) : 
    a ≥ -1 := by
  sorry

end minimum_a_l339_33911


namespace area_of_triangle_DEF_l339_33919

theorem area_of_triangle_DEF :
  let s := 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let radius := s
  let distance_between_centers := 2 * radius
  let side_of_triangle_DEF := distance_between_centers
  let triangle_area := (Real.sqrt 3 / 4) * side_of_triangle_DEF^2
  triangle_area = 4 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_DEF_l339_33919


namespace degree_not_determined_from_characteristic_l339_33989

def characteristic (P : Polynomial ℝ) : Set ℝ := sorry -- define this characteristic function

noncomputable def P₁ : Polynomial ℝ := Polynomial.X -- polynomial x
noncomputable def P₂ : Polynomial ℝ := Polynomial.X ^ 3 -- polynomial x^3

theorem degree_not_determined_from_characteristic (A : Polynomial ℝ → Set ℝ)
  (h₁ : A P₁ = A P₂) : 
  ¬∀ P : Polynomial ℝ, ∃ n : ℕ, P.degree = n → A P = A P -> P.degree = n :=
sorry

end degree_not_determined_from_characteristic_l339_33989


namespace chris_sick_weeks_l339_33948

theorem chris_sick_weeks :
  ∀ (h1 : ∀ w : ℕ, w = 4 → 2 * w = 8),
    ∀ (h2 : ∀ h w : ℕ, h = 20 → ∀ m : ℕ, 2 * (w * m) = 160),
    ∀ (h3 : ∀ h : ℕ, h = 180 → 180 - 160 = 20),
    ∀ (h4 : ∀ h w : ℕ, h = 20 → w = 20 → 20 / 20 = 1),
    180 - 160 = (20 / 20) * 20 :=
by
  intros
  sorry

end chris_sick_weeks_l339_33948


namespace midpoint_integer_of_five_points_l339_33937

theorem midpoint_integer_of_five_points 
  (P : Fin 5 → ℤ × ℤ) 
  (distinct : Function.Injective P) :
  ∃ i j : Fin 5, i ≠ j ∧ (P i).1 + (P j).1 % 2 = 0 ∧ (P i).2 + (P j).2 % 2 = 0 :=
by
  sorry

end midpoint_integer_of_five_points_l339_33937


namespace intersection_points_of_segments_l339_33979

noncomputable def num_intersection_points (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) : ℕ :=
  3000

theorem intersection_points_of_segments (A B C : Point) (P : Fin 60 → Point) (Q : Fin 50 → Point) :
  num_intersection_points A B C P Q = 3000 :=
  by sorry

end intersection_points_of_segments_l339_33979


namespace side_lengths_are_10_and_50_l339_33913

-- Define variables used in the problem
variables {s t : ℕ}

-- Define the conditions
def condition1 (s t : ℕ) : Prop := 4 * s = 20 * t
def condition2 (s t : ℕ) : Prop := s + t = 60

-- Prove that given the conditions, the side lengths of the squares are 10 and 50
theorem side_lengths_are_10_and_50 (s t : ℕ) (h1 : condition1 s t) (h2 : condition2 s t) : (s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50) :=
by sorry

end side_lengths_are_10_and_50_l339_33913


namespace smallest_n_for_multiple_of_7_l339_33949

theorem smallest_n_for_multiple_of_7 (x y : ℤ) (h1 : x % 7 = -1 % 7) (h2 : y % 7 = 2 % 7) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 ∧ n = 4 :=
sorry

end smallest_n_for_multiple_of_7_l339_33949


namespace dot_product_is_one_l339_33990

def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (-1, 2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

theorem dot_product_is_one : dot_product vec_a vec_b = 1 :=
by sorry

end dot_product_is_one_l339_33990


namespace train_speed_l339_33932

theorem train_speed 
  (length : ℝ)
  (time : ℝ)
  (relative_speed : ℝ)
  (conversion_factor : ℝ)
  (h_length : length = 120)
  (h_time : time = 4)
  (h_relative_speed : relative_speed = 60)
  (h_conversion_factor : conversion_factor = 3.6) :
  (relative_speed / 2) * conversion_factor = 108 :=
by
  sorry

end train_speed_l339_33932


namespace work_days_by_a_l339_33944

-- Given
def work_days_by_b : ℕ := 10  -- B can do the work alone in 10 days
def combined_work_days : ℕ := 5  -- A and B together can do the work in 5 days

-- Question: In how many days can A do the work alone?
def days_for_a_work_alone : ℕ := 10  -- The correct answer from the solution

-- Proof statement
theorem work_days_by_a (x : ℕ) : 
  ((1 : ℝ) / (x : ℝ) + (1 : ℝ) / (work_days_by_b : ℝ) = (1 : ℝ) / (combined_work_days : ℝ)) → 
  x = days_for_a_work_alone :=
by 
  sorry

end work_days_by_a_l339_33944


namespace origami_papers_per_cousin_l339_33939

/-- Haley has 48 origami papers and 6 cousins. Each cousin should receive the same number of papers. -/
theorem origami_papers_per_cousin : ∀ (total_papers : ℕ) (number_of_cousins : ℕ),
  total_papers = 48 → number_of_cousins = 6 → total_papers / number_of_cousins = 8 :=
by
  intros total_papers number_of_cousins
  sorry

end origami_papers_per_cousin_l339_33939


namespace impossibility_of_transition_l339_33947

theorem impossibility_of_transition 
  {a b c : ℤ}
  (h1 : a = 2)
  (h2 : b = 2)
  (h3 : c = 2) :
  ¬(∃ x y z : ℤ, x = 19 ∧ y = 1997 ∧ z = 1999 ∧
    (∃ n : ℕ, ∀ i < n, ∃ a' b' c' : ℤ, 
      if i = 0 then a' = 2 ∧ b' = 2 ∧ c' = 2 
      else (a', b', c') = 
        if i % 3 = 0 then (b + c - 1, b, c)
        else if i % 3 = 1 then (a, a + c - 1, c)
        else (a, b, a + b - 1) 
  )) :=
sorry

end impossibility_of_transition_l339_33947


namespace kim_initial_classes_l339_33931

-- Necessary definitions for the problem
def hours_per_class := 2
def total_hours_after_dropping := 6
def classes_after_dropping := total_hours_after_dropping / hours_per_class
def initial_classes := classes_after_dropping + 1

theorem kim_initial_classes : initial_classes = 4 :=
by
  -- Proof will be derived here
  sorry

end kim_initial_classes_l339_33931


namespace cartesian_equation_of_line_l339_33983

theorem cartesian_equation_of_line (t x y : ℝ)
  (h1 : x = 1 + t / 2)
  (h2 : y = 2 + (Real.sqrt 3 / 2) * t) :
  Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0 :=
sorry

end cartesian_equation_of_line_l339_33983


namespace solve_inequality_l339_33977

open Set Real

def condition1 (x : ℝ) : Prop := 6 * x + 2 < (x + 2) ^ 2
def condition2 (x : ℝ) : Prop := (x + 2) ^ 2 < 8 * x + 4

theorem solve_inequality (x : ℝ) : condition1 x ∧ condition2 x ↔ x ∈ Ioo (2 + Real.sqrt 2) 4 := by
  sorry

end solve_inequality_l339_33977


namespace common_chord_line_l339_33960

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line_l339_33960


namespace avg_of_sequence_is_x_l339_33975

noncomputable def sum_naturals (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem avg_of_sequence_is_x (x : ℝ) :
  let n := 100
  let sum := sum_naturals n
  (sum + x) / (n + 1) = 50 * x → 
  x = 5050 / 5049 :=
by
  intro n sum h
  exact sorry

end avg_of_sequence_is_x_l339_33975


namespace divides_2pow18_minus_1_l339_33954

theorem divides_2pow18_minus_1 (n : ℕ) : 20 ≤ n ∧ n < 30 ∧ (n ∣ 2^18 - 1) ↔ (n = 19 ∨ n = 27) := by
  sorry

end divides_2pow18_minus_1_l339_33954


namespace find_intersection_sums_l339_33976

noncomputable def cubic_expression (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 2
noncomputable def linear_expression (x : ℝ) : ℝ := -x / 2 + 1

theorem find_intersection_sums :
  (∃ x1 x2 x3 y1 y2 y3,
    cubic_expression x1 = linear_expression x1 ∧
    cubic_expression x2 = linear_expression x2 ∧
    cubic_expression x3 = linear_expression x3 ∧
    (x1 + x2 + x3 = 4) ∧ (y1 + y2 + y3 = 1)) :=
sorry

end find_intersection_sums_l339_33976


namespace molecular_weight_CaO_is_56_l339_33904

def atomic_weight_Ca : ℕ := 40
def atomic_weight_O : ℕ := 16
def molecular_weight_CaO : ℕ := atomic_weight_Ca + atomic_weight_O

theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = 56 := by
  sorry

end molecular_weight_CaO_is_56_l339_33904


namespace dennis_floor_l339_33918

theorem dennis_floor :
  ∃ d c b f e: ℕ, 
  (d = c + 2) ∧ 
  (c = b + 1) ∧ 
  (c = f / 4) ∧ 
  (f = 16) ∧ 
  (e = d / 2) ∧ 
  (d = 6) :=
by
  sorry

end dennis_floor_l339_33918


namespace correct_option_A_l339_33992

theorem correct_option_A : 
  (∀ a : ℝ, a^3 * a^4 = a^7) ∧ 
  ¬ (∀ a : ℝ, a^6 / a^2 = a^3) ∧ 
  ¬ (∀ a : ℝ, a^4 - a^2 = a^2) ∧ 
  ¬ (∀ a b : ℝ, (a - b)^2 = a^2 - b^2) :=
by
  /- omitted proofs -/
  sorry

end correct_option_A_l339_33992


namespace robot_handling_capacity_l339_33921

variables (x : ℝ) (A B : ℝ)

def robot_speed_condition1 : Prop :=
  A = B + 30

def robot_speed_condition2 : Prop :=
  1000 / A = 800 / B

theorem robot_handling_capacity
  (h1 : robot_speed_condition1 A B)
  (h2 : robot_speed_condition2 A B) :
  B = 120 ∧ A = 150 :=
by
  sorry

end robot_handling_capacity_l339_33921


namespace simplify_fraction_l339_33964

theorem simplify_fraction : (3 ^ 100 + 3 ^ 98) / (3 ^ 100 - 3 ^ 98) = 5 / 4 := 
by sorry

end simplify_fraction_l339_33964


namespace pirate_prob_l339_33980

def probability_treasure_no_traps := 1 / 3
def probability_traps_no_treasure := 1 / 6
def probability_neither := 1 / 2

theorem pirate_prob : (70 : ℝ) * ((1 / 3)^4 * (1 / 2)^4) = 35 / 648 := by
  sorry

end pirate_prob_l339_33980


namespace fib_inequality_l339_33951

def Fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => Fib n + Fib (n + 1)

theorem fib_inequality {n : ℕ} (h : 2 ≤ n) : Fib (n + 5) > 10 * Fib n :=
  sorry

end fib_inequality_l339_33951


namespace min_value_of_expression_l339_33912

theorem min_value_of_expression (x y : ℝ) : 
  ∃ x y, 2 * x^2 + 3 * y^2 - 8 * x + 12 * y + 40 = 20 := 
sorry

end min_value_of_expression_l339_33912


namespace exists_natural_multiple_of_2015_with_digit_sum_2015_l339_33933

-- Definition of sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Proposition that we need to prove
theorem exists_natural_multiple_of_2015_with_digit_sum_2015 :
  ∃ n : ℕ, (2015 ∣ n) ∧ sum_of_digits n = 2015 :=
sorry

end exists_natural_multiple_of_2015_with_digit_sum_2015_l339_33933


namespace geometric_sequence_from_second_term_l339_33962

open Nat

-- Define the sequence S_n
def S (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- to handle the 0th term which is typically not used here
  | 1 => 1
  | 2 => 2
  | n + 3 => 3 * S (n + 2) - 2 * S (n + 1) -- given recurrence relation

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 
  match n with
  | 0 => 0 -- Define a_0 as 0 since it's not used in the problem
  | 1 => 1 -- a1
  | n + 2 => S (n + 2) - S (n + 1) -- a_n = S_n - S_(n-1)

theorem geometric_sequence_from_second_term :
  ∀ n ≥ 2, a (n + 1) = 2 * a n := by
  -- Proof step not provided
  sorry

end geometric_sequence_from_second_term_l339_33962


namespace greatest_integer_jean_thinks_of_l339_33958

theorem greatest_integer_jean_thinks_of :
  ∃ n : ℕ, n < 150 ∧ (∃ a : ℤ, n + 2 = 9 * a) ∧ (∃ b : ℤ, n + 3 = 11 * b) ∧ n = 142 :=
by
  sorry

end greatest_integer_jean_thinks_of_l339_33958


namespace find_num_terms_in_AP_l339_33903

-- Define the necessary conditions and prove the final result
theorem find_num_terms_in_AP
  (a d : ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h_last_term_difference : (n - 1 : ℝ) * d = 7.5)
  (h_sum_odd_terms : n * (a + (n - 2 : ℝ) / 2 * d) = 60)
  (h_sum_even_terms : n * (a + ((n - 1 : ℝ) / 2) * d + d) = 90) :
  n = 12 := 
sorry

end find_num_terms_in_AP_l339_33903


namespace total_spent_correct_l339_33959

def cost_gifts : ℝ := 561.00
def cost_giftwrapping : ℝ := 139.00
def total_spent : ℝ := cost_gifts + cost_giftwrapping

theorem total_spent_correct : total_spent = 700.00 := by
  sorry

end total_spent_correct_l339_33959


namespace displacement_representation_l339_33987

def represents_north (d : ℝ) : Prop := d > 0

theorem displacement_representation (d : ℝ) (h : represents_north 80) : represents_north d ↔ d > 0 :=
by trivial

example (h : represents_north 80) : 
  ∀ d, d = -50 → ¬ represents_north d ∧ abs d = 50 → ∃ s, s = "south" :=
sorry

end displacement_representation_l339_33987


namespace percentage_trucks_returned_l339_33927

theorem percentage_trucks_returned (total_trucks rented_trucks returned_trucks : ℕ)
  (h1 : total_trucks = 24)
  (h2 : rented_trucks = total_trucks)
  (h3 : returned_trucks ≥ 12)
  (h4 : returned_trucks ≤ total_trucks) :
  (returned_trucks / rented_trucks) * 100 = 50 :=
by sorry

end percentage_trucks_returned_l339_33927


namespace convert_deg_to_rad_l339_33994

theorem convert_deg_to_rad (deg : ℝ) (h : deg = -630) : deg * (Real.pi / 180) = -7 * Real.pi / 2 :=
by
  rw [h]
  simp
  sorry

end convert_deg_to_rad_l339_33994


namespace value_of_2_Z_6_l339_33929

def Z (a b : ℝ) : ℝ := b + 10 * a - a^2

theorem value_of_2_Z_6 : Z 2 6 = 22 :=
by
  sorry

end value_of_2_Z_6_l339_33929
