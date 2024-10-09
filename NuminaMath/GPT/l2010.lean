import Mathlib

namespace chord_length_l2010_201016

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l2010_201016


namespace calculate_difference_l2010_201081

theorem calculate_difference : (-3) - (-5) = 2 := by
  sorry

end calculate_difference_l2010_201081


namespace initial_apples_count_l2010_201092

variable (initial_apples : ℕ)
variable (used_apples : ℕ := 2)
variable (bought_apples : ℕ := 23)
variable (final_apples : ℕ := 38)

theorem initial_apples_count :
  initial_apples - used_apples + bought_apples = final_apples ↔ initial_apples = 17 := by
  sorry

end initial_apples_count_l2010_201092


namespace katie_earnings_l2010_201058

theorem katie_earnings 
  (bead_necklaces : ℕ)
  (gem_stone_necklaces : ℕ)
  (bead_cost : ℕ)
  (gem_stone_cost : ℕ)
  (h1 : bead_necklaces = 4)
  (h2 : gem_stone_necklaces = 3)
  (h3 : bead_cost = 5)
  (h4 : gem_stone_cost = 8) :
  (bead_necklaces * bead_cost + gem_stone_necklaces * gem_stone_cost = 44) :=
by
  sorry

end katie_earnings_l2010_201058


namespace adelaide_ducks_l2010_201011

variable (A E K : ℕ)

theorem adelaide_ducks (h1 : A = 2 * E) (h2 : E = K - 45) (h3 : (A + E + K) / 3 = 35) :
  A = 30 := by
  sorry

end adelaide_ducks_l2010_201011


namespace jack_piggy_bank_after_8_weeks_l2010_201059

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l2010_201059


namespace bryce_raisins_l2010_201009

theorem bryce_raisins (x : ℕ) (h1 : x = 2 * (x - 8)) : x = 16 :=
by
  sorry

end bryce_raisins_l2010_201009


namespace general_form_of_equation_l2010_201095

theorem general_form_of_equation : 
  ∀ x : ℝ, (x - 1) * (x - 2) = 4 → x^2 - 3 * x - 2 = 0 := by
  sorry

end general_form_of_equation_l2010_201095


namespace validate_equation_l2010_201066

variable (x : ℝ)

def price_of_notebook : ℝ := x - 2
def price_of_pen : ℝ := x

def total_cost (x : ℝ) : ℝ := 5 * price_of_notebook x + 3 * price_of_pen x

theorem validate_equation (x : ℝ) : total_cost x = 14 :=
by
  unfold total_cost
  unfold price_of_notebook
  unfold price_of_pen
  sorry

end validate_equation_l2010_201066


namespace range_of_m_l2010_201079

open Real

noncomputable def f (x : ℝ) : ℝ := 1 + sin (2 * x)
noncomputable def g (x m : ℝ) : ℝ := 2 * (cos x)^2 + m

theorem range_of_m (x₀ : ℝ) (m : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ π / 2) (h₁ : f x₀ ≥ g x₀ m) : m ≤ sqrt 2 :=
by
  sorry

end range_of_m_l2010_201079


namespace closest_approx_of_q_l2010_201097

theorem closest_approx_of_q :
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  abs (q - 9.24) < 0.005 := 
by 
  let result : ℝ := 69.28 * 0.004
  let q : ℝ := result / 0.03
  sorry

end closest_approx_of_q_l2010_201097


namespace find_value_of_expression_l2010_201051

theorem find_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 + 4 ≤ ab + 3 * b + 2 * c) :
  200 * a + 9 * b + c = 219 :=
sorry

end find_value_of_expression_l2010_201051


namespace g_1987_l2010_201085

def g (x : ℕ) : ℚ := sorry

axiom g_defined_for_all (x : ℕ) : true

axiom g1 : g 1 = 1

axiom g_rec (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b) + 1

theorem g_1987 : g 1987 = 2 := sorry

end g_1987_l2010_201085


namespace tangents_intersect_on_line_l2010_201034

theorem tangents_intersect_on_line (a : ℝ) (x y : ℝ) (hx : 8 * a = 1) (hx_line : x - y = 5) (hx_point : x = 3) (hy_point : y = -2) : 
  x - y = 5 :=
by
  sorry -- Proof to be completed

end tangents_intersect_on_line_l2010_201034


namespace oranges_per_box_l2010_201029

theorem oranges_per_box (h_oranges : 56 = 56) (h_boxes : 8 = 8) : 56 / 8 = 7 :=
by
  -- Placeholder for the proof
  sorry

end oranges_per_box_l2010_201029


namespace tetrahedron_volume_l2010_201038

theorem tetrahedron_volume (S R V : ℝ) (h : V = (1/3) * S * R) : 
  V = (1/3) * S * R := 
by 
  sorry

end tetrahedron_volume_l2010_201038


namespace trajectory_of_point_l2010_201007

theorem trajectory_of_point (x y : ℝ) (P A : ℝ × ℝ × ℝ) (hP : P = (x, y, 0)) (hA : A = (0, 0, 4)) (hPA : dist P A = 5) : 
  x^2 + y^2 = 9 :=
by sorry

end trajectory_of_point_l2010_201007


namespace probability_interval_l2010_201064

variable (P_A P_B q : ℚ)

axiom prob_A : P_A = 5/6
axiom prob_B : P_B = 3/4
axiom prob_A_and_B : q = P_A + P_B - 1

theorem probability_interval :
  7/12 ≤ q ∧ q ≤ 3/4 :=
by
  sorry

end probability_interval_l2010_201064


namespace four_digit_numbers_with_3_or_7_l2010_201089

theorem four_digit_numbers_with_3_or_7 : 
  let total_four_digit_numbers := 9000
  let numbers_without_3_or_7 := 3584
  total_four_digit_numbers - numbers_without_3_or_7 = 5416 :=
by
  trivial

end four_digit_numbers_with_3_or_7_l2010_201089


namespace positive_integer_solutions_count_l2010_201020

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l2010_201020


namespace flat_fee_first_night_l2010_201076

theorem flat_fee_first_night :
  ∃ f n : ℚ, (f + 3 * n = 195) ∧ (f + 6 * n = 350) ∧ (f = 40) :=
by
  -- Skipping the detailed proof:
  sorry

end flat_fee_first_night_l2010_201076


namespace inequalities_hold_l2010_201024

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end inequalities_hold_l2010_201024


namespace arithmetic_sequence_a3_value_l2010_201098

theorem arithmetic_sequence_a3_value 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + 2) 
  (h2 : (a 1 + 2)^2 = a 1 * (a 1 + 8)) : 
  a 2 = 5 := 
by 
  sorry

end arithmetic_sequence_a3_value_l2010_201098


namespace length_of_legs_of_cut_off_triangles_l2010_201042

theorem length_of_legs_of_cut_off_triangles
    (side_length : ℝ) 
    (reduction_percentage : ℝ) 
    (area_reduced : side_length * side_length * reduction_percentage = 0.32 * (side_length * side_length) ) :
    ∃ (x : ℝ), 4 * (1/2 * x^2) = 0.32 * (side_length * side_length) ∧ x = 2.4 := 
by {
  sorry
}

end length_of_legs_of_cut_off_triangles_l2010_201042


namespace PASCAL_paths_correct_l2010_201082

def number_of_paths_PASCAL : Nat :=
  12

theorem PASCAL_paths_correct :
  number_of_paths_PASCAL = 12 :=
by
  sorry

end PASCAL_paths_correct_l2010_201082


namespace vasya_days_l2010_201010

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l2010_201010


namespace common_chord_l2010_201041

theorem common_chord (circle1 circle2 : ℝ × ℝ → Prop)
  (h1 : ∀ x y, circle1 (x, y) ↔ x^2 + y^2 + 2 * x = 0)
  (h2 : ∀ x y, circle2 (x, y) ↔ x^2 + y^2 - 4 * y = 0) :
  ∀ x y, circle1 (x, y) ∧ circle2 (x, y) ↔ x + 2 * y = 0 := 
by
  sorry

end common_chord_l2010_201041


namespace base8_base6_eq_l2010_201060

-- Defining the base representations
def base8 (A C : ℕ) := 8 * A + C
def base6 (C A : ℕ) := 6 * C + A

-- The main theorem stating that the integer is 47 in base 10 given the conditions
theorem base8_base6_eq (A C : ℕ) (hAC: base8 A C = base6 C A) (hA: A = 5) (hC: C = 7) : 
  8 * A + C = 47 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end base8_base6_eq_l2010_201060


namespace largest_n_divisibility_condition_l2010_201049

def S1 (n : ℕ) : ℕ := (n * (n + 1)) / 2
def S2 (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem largest_n_divisibility_condition : ∀ (n : ℕ), (n = 1) → (S2 n) % (S1 n) = 0 :=
by
  intros n hn
  rw [hn]
  sorry

end largest_n_divisibility_condition_l2010_201049


namespace sequence_length_l2010_201077

theorem sequence_length :
  ∀ (a d n : ℤ), a = -6 → d = 4 → (a + (n - 1) * d = 50) → n = 15 :=
by
  intros a d n ha hd h_seq
  sorry

end sequence_length_l2010_201077


namespace second_student_marks_l2010_201019

theorem second_student_marks (x y : ℝ) 
  (h1 : x = y + 9) 
  (h2 : x = 0.56 * (x + y)) : 
  y = 33 := 
sorry

end second_student_marks_l2010_201019


namespace geometric_sequence_value_a6_l2010_201071

theorem geometric_sequence_value_a6
    (q a1 : ℝ) (a : ℕ → ℝ)
    (h1 : ∀ n, a n = a1 * q ^ (n - 1))
    (h2 : a 2 = 1)
    (h3 : a 8 = a 6 + 2 * a 4)
    (h4 : q > 0)
    (h5 : ∀ n, a n > 0) : 
    a 6 = 4 :=
by
  sorry

end geometric_sequence_value_a6_l2010_201071


namespace calculate_expression_l2010_201070

theorem calculate_expression :
  8^8 + 8^8 + 8^8 + 8^8 + 8^5 = 4 * 8^8 + 8^5 := 
by sorry

end calculate_expression_l2010_201070


namespace closest_point_on_plane_l2010_201031

theorem closest_point_on_plane 
  (x y z : ℝ) 
  (h : 4 * x - 3 * y + 2 * z = 40) 
  (h_closest : ∀ (px py pz : ℝ), (4 * px - 3 * py + 2 * pz = 40) → dist (px, py, pz) (3, 1, 4) ≥ dist (x, y, z) (3, 1, 4)) :
  (x, y, z) = (139/19, -58/19, 86/19) :=
sorry

end closest_point_on_plane_l2010_201031


namespace intersection_M_N_l2010_201090

open Set

noncomputable def M : Set ℕ := {x | x < 6}
noncomputable def N : Set ℕ := {x | x^2 - 11 * x + 18 < 0}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end intersection_M_N_l2010_201090


namespace total_time_from_first_station_to_workplace_l2010_201028

-- Pick-up time is defined as a constant for clarity in minutes from midnight (6 AM)
def pickup_time_in_minutes : ℕ := 6 * 60

-- Travel time to first station in minutes
def travel_time_to_station_in_minutes : ℕ := 40

-- Arrival time at work (9 AM) in minutes from midnight
def arrival_time_at_work_in_minutes : ℕ := 9 * 60

-- Definition to calculate arrival time at the first station
def arrival_time_at_first_station_in_minutes : ℕ := pickup_time_in_minutes + travel_time_to_station_in_minutes

-- Theorem to prove the total time taken from the first station to the workplace
theorem total_time_from_first_station_to_workplace :
  arrival_time_at_work_in_minutes - arrival_time_at_first_station_in_minutes = 140 :=
by
  -- Placeholder for the actual proof
  sorry

end total_time_from_first_station_to_workplace_l2010_201028


namespace find_x_l2010_201040

-- Define the vectors and collinearity condition
def vector_a : ℝ × ℝ := (3, 6)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 8)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2)

-- Define the proof problem
theorem find_x (x : ℝ) (h : collinear vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l2010_201040


namespace tangents_to_discriminant_parabola_l2010_201030

variable (a : ℝ) (p q : ℝ)

theorem tangents_to_discriminant_parabola :
  (a^2 + a * p + q = 0) ↔ (p^2 - 4 * q = 0) :=
sorry

end tangents_to_discriminant_parabola_l2010_201030


namespace no_perfect_square_in_form_l2010_201062

noncomputable def is_special_form (x : ℕ) : Prop := 99990000 ≤ x ∧ x ≤ 99999999

theorem no_perfect_square_in_form :
  ¬∃ (x : ℕ), is_special_form x ∧ ∃ (n : ℕ), x = n ^ 2 := 
by 
  sorry

end no_perfect_square_in_form_l2010_201062


namespace calculate_expression_l2010_201073

theorem calculate_expression : 
  (12 * 0.5 * 3 * 0.0625 - 1.5) = -3 / 8 := 
by 
  sorry 

end calculate_expression_l2010_201073


namespace arithmetic_geometric_ratio_l2010_201002

theorem arithmetic_geometric_ratio
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n+1) = a n + d)
  (h_geometric : (a 3)^2 = a 1 * a 9)
  (h_nonzero_d : d ≠ 0) :
  a 11 / a 5 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l2010_201002


namespace evaluate_expression_l2010_201025

variable (a b : ℤ)

-- Define the original expression
def orig_expr (a b : ℤ) : ℤ :=
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (b^2 * a - 2 * a^2 * b + 1)

-- Specify the values for a and b
def a_val : ℤ := -1
def b_val : ℤ := 1

-- Prove that the expression evaluates to 10 when a = -1 and b = 1
theorem evaluate_expression : orig_expr a_val b_val = 10 := 
  by sorry

end evaluate_expression_l2010_201025


namespace catch_up_distance_l2010_201047

/-- 
  Assume that A walks at 10 km/h, starts at time 0, and B starts cycling at 20 km/h, 
  6 hours after A starts. Prove that B catches up with A 120 km from the start.
-/
theorem catch_up_distance (speed_A speed_B : ℕ) (initial_delay : ℕ) (distance : ℕ) : 
  initial_delay = 6 →
  speed_A = 10 →
  speed_B = 20 →
  distance = 120 →
  distance = speed_B * (initial_delay * speed_A / (speed_B - speed_A)) :=
by sorry

end catch_up_distance_l2010_201047


namespace circle_covers_three_points_l2010_201036

open Real

theorem circle_covers_three_points 
  (points : Finset (ℝ × ℝ))
  (h_points : points.card = 111)
  (triangle_side : ℝ)
  (h_side : triangle_side = 15) :
  ∃ (circle_center : ℝ × ℝ), ∃ (circle_radius : ℝ), circle_radius = sqrt 3 / 2 ∧ 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              dist circle_center p1 ≤ circle_radius ∧ 
              dist circle_center p2 ≤ circle_radius ∧ 
              dist circle_center p3 ≤ circle_radius :=
by
  sorry

end circle_covers_three_points_l2010_201036


namespace greatest_two_digit_with_product_12_l2010_201074

theorem greatest_two_digit_with_product_12 : 
  ∃ x y : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ x * y = 12 ∧ 
  ((10 * x + y = 62) ∨ (10 * y + x = 62)) := 
by 
  sorry

end greatest_two_digit_with_product_12_l2010_201074


namespace circle_problems_satisfy_conditions_l2010_201050

noncomputable def circle1_center_x := 11
noncomputable def circle1_center_y := 8
noncomputable def circle1_radius_squared := 87

noncomputable def circle2_center_x := 14
noncomputable def circle2_center_y := -3
noncomputable def circle2_radius_squared := 168

theorem circle_problems_satisfy_conditions :
  (∀ x y, (x-11)^2 + (y-8)^2 = 87 ∨ (x-14)^2 + (y+3)^2 = 168) := sorry

end circle_problems_satisfy_conditions_l2010_201050


namespace arccos_sin_3_l2010_201055

theorem arccos_sin_3 : Real.arccos (Real.sin 3) = (Real.pi / 2) + 3 := 
by
  sorry

end arccos_sin_3_l2010_201055


namespace solve_equation_l2010_201087

theorem solve_equation : ∃ x : ℚ, 3 * (x - 2) = x - (2 * x - 1) ∧ x = 7/4 := by
  sorry

end solve_equation_l2010_201087


namespace sqrt10_solution_l2010_201006

theorem sqrt10_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : (1/a) + (1/b) = 2) :
  m = Real.sqrt 10 :=
sorry

end sqrt10_solution_l2010_201006


namespace rhombus_area_outside_circle_l2010_201088

theorem rhombus_area_outside_circle (d : ℝ) (r : ℝ) (h_d : d = 10) (h_r : r = 3) : 
  (d * d / 2 - 9 * Real.pi) > 9 :=
by
  sorry

end rhombus_area_outside_circle_l2010_201088


namespace evaluate_six_applications_problem_solution_l2010_201001

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

theorem evaluate_six_applications (θ : ℚ) : 
  r (r (r (r (r (r θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

theorem problem_solution : r (r (r (r (r (r 30))))) = 158 / 253 :=
by
  have h : r (r (r (r (r (r 30))))) = (8 + 5 * 30) / (13 + 8 * 30) := by
    exact evaluate_six_applications 30
  rw [h]
  norm_num

end evaluate_six_applications_problem_solution_l2010_201001


namespace number_of_ways_to_choose_committee_l2010_201084

-- Definitions of the conditions
def eligible_members : ℕ := 30
def new_members : ℕ := 3
def committee_size : ℕ := 5
def eligible_pool : ℕ := eligible_members - new_members

-- Problem statement to prove
theorem number_of_ways_to_choose_committee : (Nat.choose eligible_pool committee_size) = 80730 := by
  -- This space is reserved for the proof which is not required per instructions.
  sorry

end number_of_ways_to_choose_committee_l2010_201084


namespace max_planes_15_points_l2010_201014

-- Define the total number of points
def total_points : ℕ := 15

-- Define the number of collinear points
def collinear_points : ℕ := 5

-- Compute the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of planes formed by any 3 out of 15 points
def total_planes : ℕ := binom total_points 3

-- Number of degenerate planes formed by the collinear points
def degenerate_planes : ℕ := binom collinear_points 3

-- Maximum number of unique planes
def max_unique_planes : ℕ := total_planes - degenerate_planes

-- Lean theorem statement
theorem max_planes_15_points : max_unique_planes = 445 :=
by
  sorry

end max_planes_15_points_l2010_201014


namespace complete_square_sum_l2010_201069

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end complete_square_sum_l2010_201069


namespace smallest_four_digit_multiple_of_17_l2010_201075

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end smallest_four_digit_multiple_of_17_l2010_201075


namespace sequence_26th_term_l2010_201004

theorem sequence_26th_term (a d : ℕ) (n : ℕ) (h_a : a = 4) (h_d : d = 3) (h_n : n = 26) :
  a + (n - 1) * d = 79 :=
by
  sorry

end sequence_26th_term_l2010_201004


namespace smallest_fraction_numerator_l2010_201096

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (5 * b < 7 * a) ∧ 
    ∀ (a' b' : ℕ), (10 ≤ a' ∧ a' < 100) ∧ (10 ≤ b' ∧ b' < 100) ∧ (5 * b' < 7 * a') →
    (a * b' ≤ a' * b) → a = 68 :=
sorry

end smallest_fraction_numerator_l2010_201096


namespace CindyHomework_l2010_201046

theorem CindyHomework (x : ℤ) (h : (x - 7) * 4 = 48) : (4 * x - 7) = 69 := by
  sorry

end CindyHomework_l2010_201046


namespace geometric_series_first_term_l2010_201015

theorem geometric_series_first_term (a : ℕ) (r : ℚ) (S : ℕ) (h_r : r = 1 / 4) (h_S : S = 40) (h_sum : S = a / (1 - r)) : a = 30 := sorry

end geometric_series_first_term_l2010_201015


namespace candy_cost_l2010_201003

theorem candy_cost (C : ℝ) 
  (h1 : 20 + 40 = 60) 
  (h2 : 5 * 40 + 20 * C = 60 * 6) : 
  C = 8 :=
by
  sorry

end candy_cost_l2010_201003


namespace gcd_150_m_l2010_201052

theorem gcd_150_m (m : ℕ)
  (h : ∃ d : ℕ, d ∣ 150 ∧ d ∣ m ∧ (∀ x, x ∣ 150 → x ∣ m → x = 1 ∨ x = 5 ∨ x = 25)) :
  gcd 150 m = 25 :=
sorry

end gcd_150_m_l2010_201052


namespace flight_duration_NY_to_CT_l2010_201044

theorem flight_duration_NY_to_CT :
  let departure_London_to_NY : Nat := 6 -- time in ET on Monday
  let arrival_NY_later_hours : Nat := 18 -- hours after departure
  let arrival_NY : Nat := (departure_London_to_NY + arrival_NY_later_hours) % 24 -- time in ET on Tuesday
  let arrival_CapeTown : Nat := 10 -- time in ET on Tuesday
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24 -- duration calculation
  duration_flight_NY_to_CT = 10 :=
by
  let departure_London_to_NY := 6
  let arrival_NY_later_hours := 18
  let arrival_NY := (departure_London_to_NY + arrival_NY_later_hours) % 24
  let arrival_CapeTown := 10
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24
  show duration_flight_NY_to_CT = 10
  sorry

end flight_duration_NY_to_CT_l2010_201044


namespace correct_subtraction_result_l2010_201022

theorem correct_subtraction_result (n : ℕ) (h : 40 / n = 5) : 20 - n = 12 := by
sorry

end correct_subtraction_result_l2010_201022


namespace modulo_7_example_l2010_201026

def sum := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem modulo_7_example : (sum % 7) = 5 :=
by
  sorry

end modulo_7_example_l2010_201026


namespace points_per_enemy_l2010_201067

theorem points_per_enemy (kills: ℕ) (bonus_threshold: ℕ) (bonus_multiplier: ℝ) (total_score_with_bonus: ℕ) (P: ℝ) 
(hk: kills = 150) (hbt: bonus_threshold = 100) (hbm: bonus_multiplier = 1.5) (hts: total_score_with_bonus = 2250)
(hP: 150 * P * bonus_multiplier = total_score_with_bonus) : 
P = 10 := sorry

end points_per_enemy_l2010_201067


namespace max_value_q_l2010_201035

open Nat

theorem max_value_q (X Y Z : ℕ) (h : 2 * X + 3 * Y + Z = 18) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 24 :=
sorry

end max_value_q_l2010_201035


namespace geom_seq_sum_4n_l2010_201012

-- Assume we have a geometric sequence with positive terms and common ratio q
variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)

-- The sum of the first n terms of the geometric sequence is S_n
noncomputable def S_n : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Given conditions
axiom h1 : S_n a q n = 2
axiom h2 : S_n a q (3 * n) = 14

-- We need to prove that S_{4n} = 30
theorem geom_seq_sum_4n : S_n a q (4 * n) = 30 :=
by
  sorry

end geom_seq_sum_4n_l2010_201012


namespace prove_triangular_cake_volume_surface_area_sum_l2010_201086

def triangular_cake_volume_surface_area_sum_proof : Prop :=
  let length : ℝ := 3
  let width : ℝ := 2
  let height : ℝ := 2
  let base_area : ℝ := (1 / 2) * length * width
  let volume : ℝ := base_area * height
  let top_area : ℝ := base_area
  let side_area : ℝ := (1 / 2) * width * height
  let icing_area : ℝ := top_area + 3 * side_area
  volume + icing_area = 15

theorem prove_triangular_cake_volume_surface_area_sum : triangular_cake_volume_surface_area_sum_proof := by
  sorry

end prove_triangular_cake_volume_surface_area_sum_l2010_201086


namespace arith_prog_a1_a10_geom_prog_a1_a10_l2010_201043

-- First we define our sequence and conditions for the arithmetic progression case
def is_arith_prog (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + d * (n - 1)

-- Arithmetic progression case
theorem arith_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_ap : is_arith_prog a) :
  a 1 * a 10 = -728 := 
  sorry

-- Then we define our sequence and conditions for the geometric progression case
def is_geom_prog (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a n = a 1 * q ^ (n - 1)

-- Geometric progression case
theorem geom_prog_a1_a10 (a : ℕ → ℝ)
  (h1 : a 4 + a 7 = 2)
  (h2 : a 5 * a 6 = -8)
  (h_gp : is_geom_prog a) :
  a 1 + a 10 = -7 := 
  sorry

end arith_prog_a1_a10_geom_prog_a1_a10_l2010_201043


namespace disjoint_subsets_same_sum_l2010_201027

-- Define the main theorem
theorem disjoint_subsets_same_sum (S : Finset ℕ) (hS_len : S.card = 10) (hS_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by {
  sorry
}

end disjoint_subsets_same_sum_l2010_201027


namespace helen_gas_needed_l2010_201061

-- Defining constants for the problem
def largeLawnGasPerUsage (n : ℕ) : ℕ := (n / 3) * 2
def smallLawnGasPerUsage (n : ℕ) : ℕ := (n / 2) * 1

def monthsSpringFall : ℕ := 4
def monthsSummer : ℕ := 4

def largeLawnCutsSpringFall : ℕ := 1
def largeLawnCutsSummer : ℕ := 3

def smallLawnCutsSpringFall : ℕ := 2
def smallLawnCutsSummer : ℕ := 2

-- Number of times Helen cuts large lawn in March-April and September-October
def largeLawnSpringFallCuts : ℕ := monthsSpringFall * largeLawnCutsSpringFall

-- Number of times Helen cuts large lawn in May-August
def largeLawnSummerCuts : ℕ := monthsSummer * largeLawnCutsSummer

-- Total cuts for large lawn
def totalLargeLawnCuts : ℕ := largeLawnSpringFallCuts + largeLawnSummerCuts

-- Number of times Helen cuts small lawn in March-April and September-October
def smallLawnSpringFallCuts : ℕ := monthsSpringFall * smallLawnCutsSpringFall

-- Number of times Helen cuts small lawn in May-August
def smallLawnSummerCuts : ℕ := monthsSummer * smallLawnCutsSummer

-- Total cuts for small lawn
def totalSmallLawnCuts : ℕ := smallLawnSpringFallCuts + smallLawnSummerCuts

-- Total gas needed for both lawns
def totalGasNeeded : ℕ :=
  largeLawnGasPerUsage totalLargeLawnCuts + smallLawnGasPerUsage totalSmallLawnCuts

-- The statement to prove
theorem helen_gas_needed : totalGasNeeded = 18 := sorry

end helen_gas_needed_l2010_201061


namespace junk_mail_per_block_l2010_201039

theorem junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) (total_mail : ℕ) :
  houses_per_block = 20 → mail_per_house = 32 → total_mail = 640 := by
  intros hpb_price mph_correct
  sorry

end junk_mail_per_block_l2010_201039


namespace actual_plot_area_in_acres_l2010_201021

-- Define the conditions
def base1_cm := 18
def base2_cm := 12
def height_cm := 8
def scale_cm_to_miles := 5
def sq_mile_to_acres := 640

-- Prove the question which is to find the actual plot area in acres
theorem actual_plot_area_in_acres : 
  (1/2 * (base1_cm + base2_cm) * height_cm * (scale_cm_to_miles ^ 2) * sq_mile_to_acres) = 1920000 :=
by
  sorry

end actual_plot_area_in_acres_l2010_201021


namespace same_number_of_acquaintances_l2010_201013

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) (acquaintances : Fin n → Fin n) :
  ∃ i j : Fin n, i ≠ j ∧ acquaintances i = acquaintances j :=
by
  -- Insert proof here
  sorry

end same_number_of_acquaintances_l2010_201013


namespace tangent_line_at_P_l2010_201033

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def f_prime (x : ℝ) : ℝ := 3 * x^2 - 3
def P : ℝ × ℝ := (2, -6)

theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ (x : ℝ), f_prime x = m) ∧ (∀ (x : ℝ), f x - f 2 = m * (x - 2) + b) ∧ (2 : ℝ) = 2 → b = 0 ∧ m = -3 :=
by
  sorry

end tangent_line_at_P_l2010_201033


namespace minimal_fencing_l2010_201017

theorem minimal_fencing (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l ≥ 400) : 
  2 * (w + l) = 60 * Real.sqrt 2 :=
by
  sorry

end minimal_fencing_l2010_201017


namespace abs_inequality_l2010_201078

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end abs_inequality_l2010_201078


namespace oranges_per_glass_l2010_201083

theorem oranges_per_glass (total_oranges glasses_of_juice oranges_per_glass : ℕ)
    (h_oranges : total_oranges = 12)
    (h_glasses : glasses_of_juice = 6) : 
    total_oranges / glasses_of_juice = oranges_per_glass :=
by 
    sorry

end oranges_per_glass_l2010_201083


namespace max_initial_segment_length_l2010_201054

theorem max_initial_segment_length (sequence1 : ℕ → ℕ) (sequence2 : ℕ → ℕ)
  (period1 : ℕ) (period2 : ℕ)
  (h1 : ∀ n, sequence1 (n + period1) = sequence1 n)
  (h2 : ∀ n, sequence2 (n + period2) = sequence2 n)
  (p1 : period1 = 7) (p2 : period2 = 13) :
  ∃ max_length : ℕ, max_length = 18 :=
sorry

end max_initial_segment_length_l2010_201054


namespace victoria_should_return_22_l2010_201018

theorem victoria_should_return_22 :
  let initial_money := 50
  let pizza_cost_per_box := 12
  let pizzas_bought := 2
  let juice_cost_per_pack := 2
  let juices_bought := 2
  let total_spent := (pizza_cost_per_box * pizzas_bought) + (juice_cost_per_pack * juices_bought)
  let money_returned := initial_money - total_spent
  money_returned = 22 :=
by
  sorry

end victoria_should_return_22_l2010_201018


namespace solve_system_of_inequalities_l2010_201005

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l2010_201005


namespace smallest_n_power_2013_ends_001_l2010_201057

theorem smallest_n_power_2013_ends_001 :
  ∃ n : ℕ, n > 0 ∧ 2013^n % 1000 = 1 ∧ ∀ m : ℕ, m > 0 ∧ 2013^m % 1000 = 1 → n ≤ m := 
sorry

end smallest_n_power_2013_ends_001_l2010_201057


namespace find_y_value_l2010_201099

theorem find_y_value (y : ℕ) : (1/8 * 2^36 = 2^33) ∧ (8^y = 2^(3 * y)) → y = 11 :=
by
  intros h
  -- additional elaboration to verify each step using Lean, skipped for simplicity
  sorry

end find_y_value_l2010_201099


namespace sum_of_values_l2010_201072

theorem sum_of_values :
  1 + 0.01 + 0.0001 = 1.0101 :=
by sorry

end sum_of_values_l2010_201072


namespace highway_total_vehicles_l2010_201000

theorem highway_total_vehicles (num_trucks : ℕ) (num_cars : ℕ) (total_vehicles : ℕ)
  (h1 : num_trucks = 100)
  (h2 : num_cars = 2 * num_trucks)
  (h3 : total_vehicles = num_cars + num_trucks) :
  total_vehicles = 300 :=
by
  sorry

end highway_total_vehicles_l2010_201000


namespace remainder_of_3_pow_244_mod_5_l2010_201032

theorem remainder_of_3_pow_244_mod_5 : 3^244 % 5 = 1 := by
  sorry

end remainder_of_3_pow_244_mod_5_l2010_201032


namespace boudin_hormel_ratio_l2010_201045

noncomputable def ratio_boudin_hormel : Prop :=
  let foster_chickens := 45
  let american_bottles := 2 * foster_chickens
  let hormel_chickens := 3 * foster_chickens
  let del_monte_bottles := american_bottles - 30
  let total_items := 375
  ∃ (boudin_chickens : ℕ), 
    foster_chickens + american_bottles + hormel_chickens + boudin_chickens + del_monte_bottles = total_items ∧
    boudin_chickens / hormel_chickens = 1 / 3

theorem boudin_hormel_ratio : ratio_boudin_hormel :=
sorry

end boudin_hormel_ratio_l2010_201045


namespace find_a10_l2010_201068

variable {a : ℕ → ℝ} (d a1 : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

theorem find_a10 (h1 : a 7 + a 9 = 10) 
                (h2 : sum_of_arithmetic_sequence a S)
                (h3 : S 11 = 11) : a 10 = 9 :=
sorry

end find_a10_l2010_201068


namespace battery_charging_budget_l2010_201037

def cost_per_charge : ℝ := 3.5
def charges : ℕ := 4
def leftover : ℝ := 6
def budget : ℝ := 20

theorem battery_charging_budget :
  (charges : ℝ) * cost_per_charge + leftover = budget :=
by
  sorry

end battery_charging_budget_l2010_201037


namespace solve_equation_l2010_201093

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) :
  ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1 )
  ↔ x = 5 / 4 ∨ x = -2 :=
by sorry

end solve_equation_l2010_201093


namespace multiply_equality_l2010_201094

variable (a b c d e : ℝ)

theorem multiply_equality
  (h1 : a = 2994)
  (h2 : b = 14.5)
  (h3 : c = 173)
  (h4 : d = 29.94)
  (h5 : e = 1.45)
  (h6 : a * b = c) : d * e = 1.73 :=
sorry

end multiply_equality_l2010_201094


namespace total_feet_in_garden_l2010_201056

def dogs : ℕ := 6
def ducks : ℕ := 2
def cats : ℕ := 4
def birds : ℕ := 7
def insects : ℕ := 10

def feet_per_dog : ℕ := 4
def feet_per_duck : ℕ := 2
def feet_per_cat : ℕ := 4
def feet_per_bird : ℕ := 2
def feet_per_insect : ℕ := 6

theorem total_feet_in_garden :
  dogs * feet_per_dog + 
  ducks * feet_per_duck + 
  cats * feet_per_cat + 
  birds * feet_per_bird + 
  insects * feet_per_insect = 118 := by
  sorry

end total_feet_in_garden_l2010_201056


namespace symmetrical_character_l2010_201091

def is_symmetrical (char : String) : Prop := 
  sorry  -- Here the definition for symmetry will be elaborated

theorem symmetrical_character : 
  let A : String := "坡"
  let B : String := "上"
  let C : String := "草"
  let D : String := "原"
  is_symmetrical C := 
  sorry

end symmetrical_character_l2010_201091


namespace arccos_one_half_l2010_201008

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l2010_201008


namespace haley_total_trees_l2010_201063

-- Define the number of dead trees and remaining trees
def dead_trees : ℕ := 5
def remaining_trees : ℕ := 12

-- Prove the total number of trees Haley originally grew
theorem haley_total_trees :
  (dead_trees + remaining_trees) = 17 :=
by
  -- Providing the proof using sorry as placeholder
  sorry

end haley_total_trees_l2010_201063


namespace maria_punch_l2010_201080

variable (L S W : ℕ)

theorem maria_punch (h1 : S = 3 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 36 :=
by
  sorry

end maria_punch_l2010_201080


namespace KimFridayToMondayRatio_l2010_201023

variable (MondaySweaters : ℕ) (TuesdaySweaters : ℕ) (WednesdaySweaters : ℕ) (ThursdaySweaters : ℕ) (FridaySweaters : ℕ)

def KimSweaterKnittingConditions (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ) : Prop :=
  MondaySweaters = 8 ∧
  TuesdaySweaters = MondaySweaters + 2 ∧
  WednesdaySweaters = TuesdaySweaters - 4 ∧
  ThursdaySweaters = TuesdaySweaters - 4 ∧
  MondaySweaters + TuesdaySweaters + WednesdaySweaters + ThursdaySweaters + FridaySweaters = 34

theorem KimFridayToMondayRatio 
  (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ)
  (h : KimSweaterKnittingConditions MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters) :
  FridaySweaters / MondaySweaters = 1/2 :=
  sorry

end KimFridayToMondayRatio_l2010_201023


namespace cloth_sale_total_amount_l2010_201053

theorem cloth_sale_total_amount :
  let CP := 70 -- Cost Price per metre in Rs.
  let Loss := 10 -- Loss per metre in Rs.
  let SP := CP - Loss -- Selling Price per metre in Rs.
  let total_metres := 600 -- Total metres sold
  let total_amount := SP * total_metres -- Total amount from the sale
  total_amount = 36000 := by
  sorry

end cloth_sale_total_amount_l2010_201053


namespace specific_value_is_165_l2010_201048

-- Declare x as a specific number and its value
def x : ℕ := 11

-- Declare the specific value as 15 times x
def specific_value : ℕ := 15 * x

-- The theorem to prove
theorem specific_value_is_165 : specific_value = 165 := by
  sorry

end specific_value_is_165_l2010_201048


namespace no_same_last_four_digits_pow_l2010_201065

theorem no_same_last_four_digits_pow (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (5^n % 10000) ≠ (6^m % 10000) :=
by sorry

end no_same_last_four_digits_pow_l2010_201065
