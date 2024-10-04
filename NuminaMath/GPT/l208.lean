import Mathlib

namespace basketball_starting_lineups_l208_208191

noncomputable def choose (n k : ℕ) : ℕ :=
nat.factorial n / (nat.factorial k * nat.factorial (n - k))

theorem basketball_starting_lineups :
  let total_players := 12 in
  let choose_forwards := choose total_players 2 in
  let choose_guards := choose (total_players - 2) 2 in
  let choose_center := choose (total_players - 4) 1 in
  choose_forwards * choose_guards * choose_center = 23760 := by
  sorry

end basketball_starting_lineups_l208_208191


namespace farey_sequence_problem_l208_208105

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l208_208105


namespace common_difference_is_4_l208_208653

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {a_4 a_5 S_6 : ℝ}
variable {d : ℝ}

-- Definitions of conditions given in the problem
def a4_cond : a_4 = a_n 4 := sorry
def a5_cond : a_5 = a_n 5 := sorry
def sum_six : S_6 = (6/2) * (2 * a_n 1 + 5 * d) := sorry
def term_sum : a_4 + a_5 = 24 := sorry

-- Proof statement
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l208_208653


namespace ratio_division_l208_208789

theorem ratio_division
  (A B C : ℕ)
  (h : (A : ℚ) / B = 3 / 2 ∧ (B : ℚ) / C = 1 / 3) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 :=
by
  sorry

end ratio_division_l208_208789


namespace average_speeds_equation_l208_208379

theorem average_speeds_equation (x : ℝ) (hx : 0 < x) : 
  10 / x - 7 / (1.4 * x) = 10 / 60 :=
by
  sorry

end average_speeds_equation_l208_208379


namespace any_nat_as_fraction_form_l208_208872

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l208_208872


namespace circle_ways_l208_208481

noncomputable def count3ConsecutiveCircles : ℕ :=
  let longSideWays := 1 + 2 + 3 + 4 + 5 + 6
  let perpendicularWays := (4 + 4 + 4 + 3 + 2 + 1) * 2
  longSideWays + perpendicularWays

theorem circle_ways : count3ConsecutiveCircles = 57 := by
  sorry

end circle_ways_l208_208481


namespace peter_flight_distance_l208_208667

theorem peter_flight_distance :
  ∀ (distance_spain_to_russia distance_spain_to_germany : ℕ),
  distance_spain_to_russia = 7019 →
  distance_spain_to_germany = 1615 →
  (distance_spain_to_russia - distance_spain_to_germany) + 2 * distance_spain_to_germany = 8634 :=
by
  intros distance_spain_to_russia distance_spain_to_germany h1 h2
  rw [h1, h2]
  sorry

end peter_flight_distance_l208_208667


namespace team_card_sending_l208_208566

theorem team_card_sending (x : ℕ) (h : x * (x - 1) = 56) : x * (x - 1) = 56 := 
by 
  sorry

end team_card_sending_l208_208566


namespace speed_of_A_l208_208314
-- Import necessary library

-- Define conditions
def initial_distance : ℝ := 25  -- initial distance between A and B
def speed_B : ℝ := 13  -- speed of B in kmph
def meeting_time : ℝ := 1  -- time duration in hours

-- The speed of A which is to be proven
def speed_A : ℝ := 12

-- The theorem to be proved
theorem speed_of_A (d : ℝ) (vB : ℝ) (t : ℝ) (vA : ℝ) : d = 25 → vB = 13 → t = 1 → 
  d = vA * t + vB * t → vA = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Enforcing the statement to be proved
  have := Eq.symm h4
  simp [speed_A, *] at *
  sorry

end speed_of_A_l208_208314


namespace time_to_cross_lake_one_direction_l208_208425

-- Definitions for our conditions
def cost_per_hour := 10
def total_cost_round_trip := 80

-- Statement we want to prove
theorem time_to_cross_lake_one_direction : (total_cost_round_trip / cost_per_hour) / 2 = 4 :=
  by
  sorry

end time_to_cross_lake_one_direction_l208_208425


namespace cylinder_volume_l208_208490

theorem cylinder_volume (r h : ℝ) (radius_is_2 : r = 2) (height_is_3 : h = 3) :
  π * r^2 * h = 12 * π :=
by
  rw [radius_is_2, height_is_3]
  sorry

end cylinder_volume_l208_208490


namespace simplify_abs_expression_l208_208124

theorem simplify_abs_expression (x : ℝ) : 
  |2*x + 1| - |x - 3| + |x - 6| = 
  if x < -1/2 then -2*x + 2 
  else if x < 3 then 2*x + 4 
  else if x < 6 then 10 
  else 2*x - 2 :=
by 
  sorry

end simplify_abs_expression_l208_208124


namespace large_hotdogs_sold_l208_208886

theorem large_hotdogs_sold (total_hodogs : ℕ) (small_hotdogs : ℕ) (h1 : total_hodogs = 79) (h2 : small_hotdogs = 58) : 
  total_hodogs - small_hotdogs = 21 :=
by
  sorry

end large_hotdogs_sold_l208_208886


namespace no_zero_root_l208_208908

theorem no_zero_root (x : ℝ) :
  (¬ (∃ x : ℝ, (4 * x ^ 2 - 3 = 49) ∧ x = 0)) ∧
  (¬ (∃ x : ℝ, (x ^ 2 - x - 20 = 0) ∧ x = 0)) :=
by
  sorry

end no_zero_root_l208_208908


namespace original_number_l208_208710

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l208_208710


namespace sally_initial_poems_l208_208541

theorem sally_initial_poems (recited: ℕ) (forgotten: ℕ) (h1 : recited = 3) (h2 : forgotten = 5) : 
  recited + forgotten = 8 := 
by
  sorry

end sally_initial_poems_l208_208541


namespace luka_age_difference_l208_208652

theorem luka_age_difference (a l : ℕ) (h1 : a = 8) (h2 : ∀ m : ℕ, m = 6 → l = m + 4) : l - a = 2 :=
by
  -- Assume Aubrey's age is 8
  have ha : a = 8 := h1
  -- Assume Max's age at Aubrey's 8th birthday is 6
  have hl : l = 10 := h2 6 rfl
  -- Hence, Luka is 2 years older than Aubrey
  sorry

end luka_age_difference_l208_208652


namespace janice_weekly_earnings_l208_208967

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l208_208967


namespace right_triangle_short_leg_l208_208367

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l208_208367


namespace perimeters_ratio_l208_208157

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l208_208157


namespace find_black_balls_l208_208337

-- Define the conditions given in the problem.
def initial_balls : ℕ := 10
def all_red_balls (p_red : ℝ) : Prop := p_red = 1
def equal_red_black (p_red : ℝ) (p_black : ℝ) : Prop := p_red = 0.5 ∧ p_black = 0.5
def with_green_balls (p_red : ℝ) (green_balls : ℕ) : Prop := green_balls = 2 ∧ p_red = 0.7

-- Define the total probability condition
def total_probability (p_red : ℝ) (p_green : ℝ) (p_black : ℝ) : Prop :=
  p_red + p_green + p_black = 1

-- The final statement to prove
theorem find_black_balls :
  ∃ black_balls : ℕ,
    initial_balls = 10 ∧
    (∃ p_red : ℝ, all_red_balls p_red) ∧
    (∃ p_red p_black : ℝ, equal_red_black p_red p_black) ∧
    (∃ p_red : ℝ, ∃ green_balls : ℕ, with_green_balls p_red green_balls) ∧
    (∃ p_red p_green p_black : ℝ, total_probability p_red p_green p_black) ∧
    black_balls = 1 :=
sorry

end find_black_balls_l208_208337


namespace least_k_cubed_divisible_by_168_l208_208020

theorem least_k_cubed_divisible_by_168 : ∃ k : ℤ, (k ^ 3) % 168 = 0 ∧ ∀ n : ℤ, (n ^ 3) % 168 = 0 → k ≤ n :=
sorry

end least_k_cubed_divisible_by_168_l208_208020


namespace chosen_number_l208_208017

theorem chosen_number (x: ℤ) (h: 2 * x - 152 = 102) : x = 127 :=
by
  sorry

end chosen_number_l208_208017


namespace min_value_diff_l208_208773

theorem min_value_diff (P Q F1 F : ℝ × ℝ)
  (hP_on_ellipse : P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1)
  (hQ_proj_line : ∃ t : ℝ, Q = (P.1 + 4 * t, P.2 + 3 * t) ∧ 4 * P.1 + 3 * P.2 = 21)
  (hFocus : F = (2, 0)) :
  ∃ Q, P.1 ^ 2 / 4 + P.2 ^ 2 / 3 = 1 ∧
       Q = (P.1 + t * 4, P.2 + t * 3) ∧ 
       4 * Q.1 + 3 * Q.2 - 21 = 0 ∧ 
       min_val = 1 :=
sorry

end min_value_diff_l208_208773


namespace economic_rationale_education_policy_l208_208952

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l208_208952


namespace area_of_circle_diameter_7_5_l208_208011

theorem area_of_circle_diameter_7_5 :
  ∃ (A : ℝ), (A = 14.0625 * Real.pi) ↔ (∃ (d : ℝ), d = 7.5 ∧ A = Real.pi * (d / 2) ^ 2) :=
by
  sorry

end area_of_circle_diameter_7_5_l208_208011


namespace simplified_expression_l208_208731

variable (x y : ℝ)

theorem simplified_expression (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / ((-4 / 15) * Real.sqrt (y / x)) * ((-5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by
  sorry

end simplified_expression_l208_208731


namespace algebraic_expression_solution_l208_208429

theorem algebraic_expression_solution
  (a b : ℝ)
  (h : -2 * a + 3 * b = 10) :
  9 * b - 6 * a + 2 = 32 :=
by 
  -- We would normally provide the proof here
  sorry

end algebraic_expression_solution_l208_208429


namespace relationship_ab_l208_208632

noncomputable def a : ℝ := Real.log 243 / Real.log 5
noncomputable def b : ℝ := Real.log 27 / Real.log 3

theorem relationship_ab : a = (5 / 3) * b := sorry

end relationship_ab_l208_208632


namespace rectangle_fraction_l208_208293

noncomputable def side_of_square : ℝ := Real.sqrt 900
noncomputable def radius_of_circle : ℝ := side_of_square
noncomputable def area_of_rectangle : ℝ := 120
noncomputable def breadth_of_rectangle : ℝ := 10
noncomputable def length_of_rectangle : ℝ := area_of_rectangle / breadth_of_rectangle
noncomputable def fraction : ℝ := length_of_rectangle / radius_of_circle

theorem rectangle_fraction :
  (length_of_rectangle / radius_of_circle) = (2 / 5) :=
by
  sorry

end rectangle_fraction_l208_208293


namespace min_side_length_l208_208586

def table_diagonal (w h : ℕ) : ℕ :=
  Nat.sqrt (w * w + h * h)

theorem min_side_length (w h : ℕ) (S : ℕ) (dw : w = 9) (dh : h = 12) (dS : S = 15) :
  S >= table_diagonal w h :=
by
  sorry

end min_side_length_l208_208586


namespace simplify_fraction_l208_208277

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208277


namespace simplify_fraction_l208_208280

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208280


namespace min_cubes_needed_l208_208579

def minimum_cubes_for_views (front_view side_view : ℕ) : ℕ :=
  4

theorem min_cubes_needed (front_view_cond side_view_cond : ℕ) :
  front_view_cond = 2 ∧ side_view_cond = 3 → minimum_cubes_for_views front_view_cond side_view_cond = 4 :=
by
  intro h
  cases h
  -- Proving the condition based on provided views
  sorry

end min_cubes_needed_l208_208579


namespace rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l208_208223

variable (x : ℚ)

-- Polynomial 1
def polynomial1 := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_polynomial1 :
  (polynomial1 (-1) = 0) ∧
  (polynomial1 2 = 0) ∧
  (polynomial1 (-2) = 0) ∧
  (polynomial1 4 = 0) :=
sorry

-- Polynomial 2
def polynomial2 := 8*x^3 - 20*x^2 - 2*x + 5

theorem rational_roots_polynomial2 :
  (polynomial2 (1/2) = 0) ∧
  (polynomial2 (-1/2) = 0) ∧
  (polynomial2 (5/2) = 0) :=
sorry

-- Polynomial 3
def polynomial3 := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem rational_roots_polynomial3 :
  (polynomial3 (-1/2) = 0) ∧
  (polynomial3 (1/2) = 0) ∧
  (polynomial3 1 = 0) ∧
  (polynomial3 3 = 0) :=
sorry

end rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l208_208223


namespace xiaoning_comprehensive_score_l208_208444

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l208_208444


namespace parabola_intersects_x_axis_l208_208572

theorem parabola_intersects_x_axis 
  (a c : ℝ) 
  (h : ∃ x : ℝ, x = 1 ∧ (a * x^2 + x + c = 0)) : 
  a + c = -1 :=
sorry

end parabola_intersects_x_axis_l208_208572


namespace students_play_neither_sport_l208_208091

def total_students : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def both_players : ℕ := 10

theorem students_play_neither_sport :
  total_students - (hockey_players + basketball_players - both_players) = 4 :=
by
  sorry

end students_play_neither_sport_l208_208091


namespace intersection_of_M_and_N_l208_208933

theorem intersection_of_M_and_N (x : ℝ) :
  {x | x > 1} ∩ {x | x^2 - 2 * x < 0} = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l208_208933


namespace quadratic_solution_1_quadratic_solution_2_l208_208837

theorem quadratic_solution_1 (x : ℝ) :
  x^2 + 3 * x - 1 = 0 ↔ (x = (-3 + Real.sqrt 13) / 2) ∨ (x = (-3 - Real.sqrt 13) / 2) :=
by
  sorry

theorem quadratic_solution_2 (x : ℝ) :
  (x - 2)^2 = 2 * (x - 2) ↔ (x = 2) ∨ (x = 4) :=
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l208_208837


namespace parallel_planes_of_skew_lines_l208_208980

variables {Plane : Type*} {Line : Type*}
variables (α β : Plane)
variables (a b : Line)

-- Conditions
def is_parallel (p1 p2 : Plane) : Prop := sorry -- Parallel planes relation
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- Line in plane relation
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- Line parallel to plane relation
def is_skew_lines (l1 l2 : Line) : Prop := sorry -- Skew lines relation

-- Theorem to prove
theorem parallel_planes_of_skew_lines 
  (h1 : line_in_plane a α)
  (h2 : line_in_plane b β)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α)
  (h5 : is_skew_lines a b) :
  is_parallel α β :=
sorry

end parallel_planes_of_skew_lines_l208_208980


namespace largest_k_l208_208114

def S : Set ℕ := {x | x > 0 ∧ x ≤ 100}

def satisfies_property (A B : Set ℕ) : Prop :=
  ∃ x ∈ A ∩ B, ∀ y ∈ A ∪ B, x ≠ y

theorem largest_k (k : ℕ) : 
  (∃ subsets : Finset (Set ℕ), 
    (subsets.card = k) ∧ 
    (∀ {A B : Set ℕ}, A ∈ subsets ∧ B ∈ subsets ∧ A ≠ B → 
      ¬(A ∩ B = ∅) ∧ satisfies_property A B)) →
  k ≤ 2^99 - 1 := sorry

end largest_k_l208_208114


namespace ratio_of_perimeters_l208_208144

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l208_208144


namespace inequality_positive_reals_l208_208233

theorem inequality_positive_reals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a^2 + 1 / b^2 + 8 * a * b ≥ 8) ∧ (1 / a^2 + 1 / b^2 + 8 * a * b = 8 → a = b ∧ a = 1/2) :=
by
  sorry

end inequality_positive_reals_l208_208233


namespace find_a_b_l208_208043

theorem find_a_b (a b : ℤ) (h: 4 * a^2 + 3 * b^2 + 10 * a * b = 144) :
    (a = 2 ∧ b = 4) :=
by {
  sorry
}

end find_a_b_l208_208043


namespace find_min_difference_l208_208101

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l208_208101


namespace sum_of_a_l208_208235

def f (n : ℕ) : ℝ := n^2 * Real.cos (n * Real.pi)

def a (n : ℕ) : ℝ := f n + f (n + 1)

theorem sum_of_a {n : ℕ} (h : n = 100) : ∑ i in Finset.range 100, a (i + 1) = -550 :=
by
  sorry

end sum_of_a_l208_208235


namespace number_of_black_and_white_films_l208_208317

theorem number_of_black_and_white_films (B x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h_fraction : (6 * y : ℚ) / ((y / (x : ℚ))/100 * (B : ℚ) + 6 * y) = 20 / 21) :
  B = 30 * x :=
sorry

end number_of_black_and_white_films_l208_208317


namespace total_bins_l208_208893

-- Definition of the problem conditions
def road_length : ℕ := 400
def placement_interval : ℕ := 20
def bins_per_side : ℕ := (road_length / placement_interval) - 1

-- Statement of the problem
theorem total_bins : 2 * bins_per_side = 38 := by
  sorry

end total_bins_l208_208893


namespace sufficient_condition_for_beta_l208_208523

theorem sufficient_condition_for_beta (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (x ≤ m)) → (3 ≤ m) :=
by
  sorry

end sufficient_condition_for_beta_l208_208523


namespace lemmings_distance_average_l208_208739

noncomputable def diagonal_length (side: ℝ) : ℝ :=
  Real.sqrt (side^2 + side^2)

noncomputable def fraction_traveled (side: ℝ) (distance: ℝ) : ℝ :=
  distance / (Real.sqrt 2 * side)

noncomputable def final_coordinates (side: ℝ) (distance1: ℝ) (angle: ℝ) (distance2: ℝ) : (ℝ × ℝ) :=
  let frac := fraction_traveled side distance1
  let initial_pos := (frac * side, frac * side)
  let move_dist := distance2 * (Real.sqrt 2 / 2)
  (initial_pos.1 + move_dist, initial_pos.2 + move_dist)

noncomputable def average_shortest_distances (side: ℝ) (coords: ℝ × ℝ) : ℝ :=
  let x_dist := min coords.1 (side - coords.1)
  let y_dist := min coords.2 (side - coords.2)
  (x_dist + (side - x_dist) + y_dist + (side - y_dist)) / 4

theorem lemmings_distance_average :
  let side := 15
  let distance1 := 9.3
  let angle := 45 / 180 * Real.pi -- convert to radians
  let distance2 := 3
  let coords := final_coordinates side distance1 angle distance2
  average_shortest_distances side coords = 7.5 :=
by
  sorry

end lemmings_distance_average_l208_208739


namespace sum_of_integers_l208_208840

variable (p q r s : ℤ)

theorem sum_of_integers :
  (p - q + r = 7) →
  (q - r + s = 8) →
  (r - s + p = 4) →
  (s - p + q = 1) →
  p + q + r + s = 20 := by
  intros h1 h2 h3 h4
  sorry

end sum_of_integers_l208_208840


namespace negation_of_proposition_l208_208415

theorem negation_of_proposition (x y : ℝ) :
  (¬ (x + y = 1 → xy ≤ 1)) ↔ (x + y ≠ 1 → xy > 1) :=
by 
  sorry

end negation_of_proposition_l208_208415


namespace carolyn_removal_sum_correct_l208_208543

-- Define the initial conditions
def n : Nat := 10
def initialList : List Nat := List.range (n + 1)  -- equals [0, 1, 2, ..., 10]

-- Given that Carolyn removes specific numbers based on the game rules
def carolynRemovals : List Nat := [6, 10, 8]

-- Sum of numbers removed by Carolyn
def carolynRemovalSum : Nat := carolynRemovals.sum

-- Theorem stating the sum of numbers removed by Carolyn
theorem carolyn_removal_sum_correct : carolynRemovalSum = 24 := by
  sorry

end carolyn_removal_sum_correct_l208_208543


namespace probability_no_adjacent_seats_l208_208533

theorem probability_no_adjacent_seats :
  let total_ways := Nat.choose 10 3
  let adjacent_2_1 := 9 * 8
  let adjacent_3 := 8
  let ways_adjacent := adjacent_2_1 + adjacent_3
  let p_none_adjacent := 1 - (ways_adjacent / total_ways : ℚ)
  p_none_adjacent = 1 / 3 :=
by
  let total_ways := Nat.choose 10 3
  let adjacent_2_1 := 9 * 8
  let adjacent_3 := 8
  let ways_adjacent := adjacent_2_1 + adjacent_3
  let p_none_adjacent := 1 - (ways_adjacent / total_ways : ℚ)
  have h1 : total_ways = 120 := by simp
  have h2 : adjacent_2_1 = 72 := by simp
  have h3 : adjacent_3 = 8 := by simp
  have h4 : ways_adjacent = 80 := by simp [h2, h3]
  have h5 : p_none_adjacent = (120 - 80) / 120 := by simp [h4, h1]
  simp [h5]
  norm_num [h1]
  done

end probability_no_adjacent_seats_l208_208533


namespace problem_l208_208781

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then Real.log x / Real.log 3 + m else 1 / 2017

theorem problem (m := -2) (h_root : f 3 m = 0):
  f (f 6 m - 2) m = 1 / 2017 :=
by
  sorry

end problem_l208_208781


namespace quadratic_solution_difference_l208_208418

theorem quadratic_solution_difference (x : ℝ) :
  ∀ x : ℝ, (x^2 - 5*x + 15 = x + 55) → (∃ a b : ℝ, a ≠ b ∧ x^2 - 6*x - 40 = 0 ∧ abs (a - b) = 14) :=
by
  sorry

end quadratic_solution_difference_l208_208418


namespace mutually_exclusive_event_is_D_l208_208949

namespace Problem

def event_A (n : ℕ) (defective : ℕ) : Prop := defective ≥ 2
def mutually_exclusive_event (n : ℕ) : Prop := (∀ (defective : ℕ), defective ≤ 1) ↔ (∀ (defective : ℕ), defective ≥ 2 → false)

theorem mutually_exclusive_event_is_D (n : ℕ) : mutually_exclusive_event n := 
by 
  sorry

end Problem

end mutually_exclusive_event_is_D_l208_208949


namespace max_cables_cut_l208_208861

theorem max_cables_cut (computers cables clusters : ℕ) (h_computers : computers = 200) (h_cables : cables = 345) (h_clusters : clusters = 8) :
  ∃ k : ℕ, k = cables - (computers - clusters + 1) ∧ k = 153 :=
by
  sorry

end max_cables_cut_l208_208861


namespace sum_remainders_eq_two_l208_208181

theorem sum_remainders_eq_two (a b c : ℤ) (h_a : a % 24 = 10) (h_b : b % 24 = 4) (h_c : c % 24 = 12) :
  (a + b + c) % 24 = 2 :=
by
  sorry

end sum_remainders_eq_two_l208_208181


namespace farey_sequence_problem_l208_208106

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l208_208106


namespace domain_transform_l208_208777

variable (f : ℝ → ℝ)

theorem domain_transform (h : ∀ x, -1 ≤ x ∧ x ≤ 4 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f y = 2 * x - 1 :=
sorry

end domain_transform_l208_208777


namespace no_integer_solutions_system_l208_208831

theorem no_integer_solutions_system :
  ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := 
sorry

end no_integer_solutions_system_l208_208831


namespace complement_intersection_l208_208658

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 2, 3} → 
  B = {2, 3, 4} → 
  (U \ (A ∩ B) = {1, 4, 5}) := 
by
  sorry

end complement_intersection_l208_208658


namespace find_number_l208_208264

theorem find_number (x : ℝ) (h : 0.3 * x - (1 / 3) * (0.3 * x) = 36) : x = 180 :=
sorry

end find_number_l208_208264


namespace a6_minus_b6_divisible_by_9_l208_208362

theorem a6_minus_b6_divisible_by_9 {a b : ℤ} (h₁ : a % 3 ≠ 0) (h₂ : b % 3 ≠ 0) : (a ^ 6 - b ^ 6) % 9 = 0 := 
sorry

end a6_minus_b6_divisible_by_9_l208_208362


namespace third_side_not_twelve_l208_208188

theorem third_side_not_twelve (x : ℕ) (h1 : x > 5) (h2 : x < 11) (h3 : x % 2 = 0) : x ≠ 12 :=
by
  -- The proof is omitted
  sorry

end third_side_not_twelve_l208_208188


namespace payback_time_l208_208393

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l208_208393


namespace abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l208_208730

theorem abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one (x : ℝ) :
  |x| < 1 → x^3 < 1 ∧ (x^3 < 1 → |x| < 1 → False) :=
by
  sorry

end abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l208_208730


namespace sum_of_squares_l208_208928

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2) : 
  ∃ e f : ℕ, a^2 + n * b^2 = e^2 + f^2 :=
by
  sorry

-- Theorem parameters and logical flow explained:

-- a, b, n : ℕ                  -- Natural number inputs
-- h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2  -- Condition given in the problem that a^2 + 2nb^2 is a perfect square
-- Prove that there exist natural numbers e and f such that a^2 + nb^2 = e^2 + f^2

end sum_of_squares_l208_208928


namespace probability_of_first_three_red_cards_l208_208575

def total_cards : ℕ := 104
def suits : ℕ := 4
def cards_per_suit : ℕ := 26
def red_suits : ℕ := 2
def black_suits : ℕ := 2
def total_red_cards : ℕ := 52
def total_black_cards : ℕ := 52

noncomputable def probability_first_three_red : ℚ :=
  (total_red_cards / total_cards) * ((total_red_cards - 1) / (total_cards - 1)) * ((total_red_cards - 2) / (total_cards - 2))

theorem probability_of_first_three_red_cards :
  probability_first_three_red = 425 / 3502 :=
sorry

end probability_of_first_three_red_cards_l208_208575


namespace daily_practice_hours_l208_208197

-- Define the conditions as given in the problem
def total_hours_practiced_this_week : ℕ := 36
def total_days_in_week : ℕ := 7
def days_could_not_practice : ℕ := 1
def actual_days_practiced := total_days_in_week - days_could_not_practice

-- State the theorem including the question and the correct answer, given the conditions
theorem daily_practice_hours :
  total_hours_practiced_this_week / actual_days_practiced = 6 := 
by
  sorry

end daily_practice_hours_l208_208197


namespace chocolate_chips_per_family_member_l208_208387

def total_family_members : ℕ := 4
def batches_choco_chip : ℕ := 3
def batches_double_choco_chip : ℕ := 2
def batches_white_choco_chip : ℕ := 1
def cookies_per_batch_choco_chip : ℕ := 12
def cookies_per_batch_double_choco_chip : ℕ := 10
def cookies_per_batch_white_choco_chip : ℕ := 15
def choco_chips_per_cookie_choco_chip : ℕ := 2
def choco_chips_per_cookie_double_choco_chip : ℕ := 4
def choco_chips_per_cookie_white_choco_chip : ℕ := 3

theorem chocolate_chips_per_family_member :
  (batches_choco_chip * cookies_per_batch_choco_chip * choco_chips_per_cookie_choco_chip +
   batches_double_choco_chip * cookies_per_batch_double_choco_chip * choco_chips_per_cookie_double_choco_chip +
   batches_white_choco_chip * cookies_per_batch_white_choco_chip * choco_chips_per_cookie_white_choco_chip) / 
   total_family_members = 49 :=
by
  sorry

end chocolate_chips_per_family_member_l208_208387


namespace fractions_equivalent_iff_x_eq_zero_l208_208842

theorem fractions_equivalent_iff_x_eq_zero (x : ℝ) (h : (x + 1) / (x + 3) = 1 / 3) : x = 0 :=
by
  sorry

end fractions_equivalent_iff_x_eq_zero_l208_208842


namespace smallest_value_of_abs_sum_l208_208705

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l208_208705


namespace right_triangle_short_leg_l208_208368

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l208_208368


namespace find_parallel_line_through_point_l208_208162

-- Definition of a point in Cartesian coordinates
structure Point :=
(x : ℝ)
(y : ℝ)

-- Definition of a line in slope-intercept form
def line (a b c : ℝ) : Prop := ∀ p : Point, a * p.x + b * p.y + c = 0

-- Conditions provided in the problem
def P : Point := ⟨-1, 3⟩
def line1 : Prop := line 1 (-2) 3
def parallel_line (c : ℝ) : Prop := line 1 (-2) c

-- Theorem to prove
theorem find_parallel_line_through_point : parallel_line 7 :=
sorry

end find_parallel_line_through_point_l208_208162


namespace max_value_of_z_l208_208640

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y - 5 ≥ 0) (h2 : x - 2 * y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∃ x y, x + y = 9 :=
by {
  sorry
}

end max_value_of_z_l208_208640


namespace find_min_difference_l208_208104

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l208_208104


namespace distance_between_midpoints_l208_208449

-- Conditions
def AA' := 68 -- in centimeters
def BB' := 75 -- in centimeters
def CC' := 112 -- in centimeters
def DD' := 133 -- in centimeters

-- Question: Prove the distance between the midpoints of A'C' and B'D' is 14 centimeters
theorem distance_between_midpoints :
  let midpoint_A'C' := (AA' + CC') / 2
  let midpoint_B'D' := (BB' + DD') / 2
  (midpoint_B'D' - midpoint_A'C' = 14) :=
by
  sorry

end distance_between_midpoints_l208_208449


namespace problem_part_1_problem_part_2_l208_208785

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

noncomputable def tan_2x_when_parallel (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Prop :=
    Real.tan (2 * x) = 12 / 5

noncomputable def f (x : ℝ) : ℝ :=
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2

def range_f_on_interval : Prop :=
  ∀ x ∈ Set.Icc (-Real.pi / 2) 0, -Real.sqrt 2 / 2 ≤ f x ∧ f x ≤ 1 / 2

theorem problem_part_1 (x : ℝ) (h : (Real.sin x + 3 / 2 * Real.cos x = 0)) : Real.tan (2 * x) = 12 / 5 :=
by
  sorry

theorem problem_part_2 : range_f_on_interval :=
by
  sorry

end problem_part_1_problem_part_2_l208_208785


namespace inequality_abc_l208_208475

open Real

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / sqrt (a^2 + 8 * b * c)) + (b / sqrt (b^2 + 8 * c * a)) + (c / sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_abc_l208_208475


namespace payback_time_l208_208392

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l208_208392


namespace arithmetic_mean_q_r_l208_208547

theorem arithmetic_mean_q_r (p q r : ℝ) (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) (h3 : r - p = 34) : (q + r) / 2 = 27 :=
sorry

end arithmetic_mean_q_r_l208_208547


namespace quadratic_two_equal_real_roots_l208_208063

theorem quadratic_two_equal_real_roots (m : ℝ) :
  (∃ (x : ℝ), x^2 + m * x + m = 0 ∧ ∀ (y : ℝ), x = y → x^2 + m * y + m = 0) →
  (m = 0 ∨ m = 4) :=
by {
  sorry
}

end quadratic_two_equal_real_roots_l208_208063


namespace initial_price_of_article_l208_208048

theorem initial_price_of_article (P : ℝ) (h : 0.4025 * P = 620) : P = 620 / 0.4025 :=
by
  sorry

end initial_price_of_article_l208_208048


namespace xiao_ming_percentile_l208_208509

theorem xiao_ming_percentile (total_students : ℕ) (rank : ℕ) 
  (h1 : total_students = 48) (h2 : rank = 5) :
  ∃ p : ℕ, (p = 90 ∨ p = 91) ∧ (43 < (p * total_students) / 100) ∧ ((p * total_students) / 100 ≤ 44) :=
by
  sorry

end xiao_ming_percentile_l208_208509


namespace janice_total_earnings_l208_208960

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l208_208960


namespace largest_fraction_sum_l208_208596

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l208_208596


namespace solution_set_suff_not_necessary_l208_208680

theorem solution_set_suff_not_necessary (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end solution_set_suff_not_necessary_l208_208680


namespace values_of_x_in_range_l208_208080

open Real

noncomputable def count_x_values : Nat :=
  let lower_bound := -25
  let upper_bound := 105
  let count_k := ((upper_bound / π).floor - (lower_bound / π).ceil).succ
  count_k

theorem values_of_x_in_range :
  ∃ k_values : FinSet ℤ, 
  (∀ k ∈ k_values, -25 < k * π ∧ k * π < 105) ∧
  (∀ x, x ∈ (Set.Ioo (-25) 105) → (cos x)^2 + 2 * (sin x)^2 = 1 → x = k_values.to_list.length ∧
   k_values.to_list.length = 41) :=
sorry

end values_of_x_in_range_l208_208080


namespace pam_walked_1683_miles_l208_208121

noncomputable def pam_miles_walked 
    (pedometer_limit : ℕ)
    (initial_reading : ℕ)
    (flips : ℕ)
    (final_reading : ℕ)
    (steps_per_mile : ℕ)
    : ℕ :=
  (pedometer_limit + 1) * flips + final_reading / steps_per_mile

theorem pam_walked_1683_miles
    (pedometer_limit : ℕ := 49999)
    (initial_reading : ℕ := 0)
    (flips : ℕ := 50)
    (final_reading : ℕ := 25000)
    (steps_per_mile : ℕ := 1500) 
    : pam_miles_walked pedometer_limit initial_reading flips final_reading steps_per_mile = 1683 := 
  sorry

end pam_walked_1683_miles_l208_208121


namespace simplify_fraction_l208_208284

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l208_208284


namespace find_c_l208_208527

-- Define that \( r \) and \( s \) are roots of \( 2x^2 - 4x - 5 \)
variables (r s : ℚ)
-- Condition: sum of roots \( r + s = 2 \)
axiom sum_of_roots : r + s = 2
-- Condition: product of roots \( rs = -5/2 \)
axiom product_of_roots : r * s = -5 / 2

-- Definition of \( c \) based on the roots \( r-3 \) and \( s-3 \)
def c : ℚ := (r - 3) * (s - 3)

-- The theorem to be proved
theorem find_c : c = 1 / 2 :=
by
  sorry

end find_c_l208_208527


namespace equation_of_line_l208_208851

theorem equation_of_line (θ : ℝ) (b : ℝ) :
  θ = 135 ∧ b = -1 → (∀ x y : ℝ, x + y + 1 = 0) :=
by
  sorry

end equation_of_line_l208_208851


namespace ratio_of_areas_is_one_ninth_l208_208136

-- Define the side lengths of Square A and Square B
variables (x : ℝ)
def side_length_a := x
def side_length_b := 3 * x

-- Define the areas of Square A and Square B
def area_a := side_length_a x * side_length_a x
def area_b := side_length_b x * side_length_b x

-- The theorem to prove the ratio of areas
theorem ratio_of_areas_is_one_ninth : (area_a x) / (area_b x) = (1 / 9) :=
by sorry

end ratio_of_areas_is_one_ninth_l208_208136


namespace max_xy_value_l208_208237

theorem max_xy_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 2) : xy ≤ 1 / 2 := 
by
  sorry

end max_xy_value_l208_208237


namespace no_integer_solutions_l208_208441

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^3 + 21 * y^2 + 5 = 0 :=
by {
  sorry
}

end no_integer_solutions_l208_208441


namespace jack_mopping_time_l208_208383

-- Definitions for the conditions
def bathroom_area : ℝ := 24
def kitchen_area : ℝ := 80
def mopping_rate : ℝ := 8

-- The proof problem: Prove Jack will spend 13 minutes mopping
theorem jack_mopping_time : (bathroom_area + kitchen_area) / mopping_rate = 13 := by
  sorry

end jack_mopping_time_l208_208383


namespace shorter_leg_of_right_triangle_l208_208372

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l208_208372


namespace line_parallel_to_x_axis_l208_208766

variable (k : ℝ)

theorem line_parallel_to_x_axis :
  let point1 := (3, 2 * k + 1)
  let point2 := (8, 4 * k - 5)
  (point1.2 = point2.2) ↔ (k = 3) :=
by
  sorry

end line_parallel_to_x_axis_l208_208766


namespace probability_of_3_consecutive_heads_in_4_tosses_l208_208737

theorem probability_of_3_consecutive_heads_in_4_tosses :
  (∃ p : ℚ, p = 3/16 ∧ probability (λ s : vector bool 4, is_3_consecutive_heads s) = p) :=
sorry

def is_3_consecutive_heads (v : vector bool 4) : Prop :=
  (v.head = tt ∧ v.tail.head = tt ∧ v.tail.tail.head = tt) ∨
  (v.head = tt ∧ v.tail.head = tt ∧ v.tail.tail.tail.head = tt) ∨
  (v.tail.head = tt ∧ v.tail.tail.head = tt ∧ v.tail.tail.tail.head = tt)

end probability_of_3_consecutive_heads_in_4_tosses_l208_208737


namespace problem1_problem2_l208_208835

-- Problem 1
theorem problem1 (x : ℚ) (h : x = -1/3) : 6 * x^2 + 5 * x^2 - 2 * (3 * x - 2 * x^2) = 11 / 3 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℚ) (ha : a = -2) (hb : b = -1) : 5 * a^2 - a * b - 2 * (3 * a * b - (a * b - 2 * a^2)) = -6 :=
by sorry

end problem1_problem2_l208_208835


namespace average_in_all_6_subjects_l208_208747

-- Definitions of the conditions
def average_in_5_subjects : ℝ := 74
def marks_in_6th_subject : ℝ := 104
def num_subjects_total : ℝ := 6

-- Proof that the average in all 6 subjects is 79
theorem average_in_all_6_subjects :
  (average_in_5_subjects * 5 + marks_in_6th_subject) / num_subjects_total = 79 := by
  sorry

end average_in_all_6_subjects_l208_208747


namespace probability_PORTLAND_l208_208520

noncomputable def probability_dock : ℚ :=
  1 / Nat.choose 4 2

noncomputable def probability_plants : ℚ :=
  2 / Nat.choose 6 4

noncomputable def probability_hero : ℚ :=
  3 / Nat.choose 4 3

noncomputable def total_probability : ℚ :=
  probability_dock * probability_plants * probability_hero

theorem probability_PORTLAND : total_probability = 1 / 40 :=
by
  sorry

end probability_PORTLAND_l208_208520


namespace insphere_touches_centers_of_faces_l208_208213

/--
Given a regular tetrahedron whose faces are all equilateral triangles, 
prove that the insphere touches each triangular face at its center.
-/
theorem insphere_touches_centers_of_faces (T : regular_tetrahedron) :
  ∀ face ∈ T.faces, T.insphere.touches face (T.face_center face) :=
sorry

end insphere_touches_centers_of_faces_l208_208213


namespace equation_true_l208_208047

variables {AB BC CD AD AC BD : ℝ}

theorem equation_true :
  (AD * BC + AB * CD = AC * BD) ∧
  (AD * BC - AB * CD ≠ AC * BD) ∧
  (AB * BC + AC * CD ≠ AC * BD) ∧
  (AB * BC - AC * CD ≠ AC * BD) :=
by
  sorry

end equation_true_l208_208047


namespace continuous_stripe_probability_l208_208917

-- Definitions based on conditions from a)
def total_possible_combinations : ℕ := 4^6

def favorable_outcomes : ℕ := 12

def probability_of_continuous_stripe : ℚ := favorable_outcomes / total_possible_combinations

-- The theorem equivalent to prove the given problem
theorem continuous_stripe_probability :
  probability_of_continuous_stripe = 3 / 1024 :=
by
  sorry

end continuous_stripe_probability_l208_208917


namespace cost_to_fly_A_to_B_l208_208120

noncomputable def flight_cost (distance : ℕ) : ℕ := (distance * 10 / 100) + 100

theorem cost_to_fly_A_to_B :
  flight_cost 3250 = 425 :=
by
  sorry

end cost_to_fly_A_to_B_l208_208120


namespace original_number_value_l208_208716

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l208_208716


namespace original_number_solution_l208_208720

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l208_208720


namespace value_of_k_l208_208174

theorem value_of_k :
  ∃ k, k = 2 ∧ (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 5 ∧
                ∀ (s t : ℕ), (s, t) ∈ pairs → s = k * t) :=
by 
sorry

end value_of_k_l208_208174


namespace slope_angle_of_perpendicular_line_l208_208859

theorem slope_angle_of_perpendicular_line (h : ∀ x, x = (π / 3)) : ∀ θ, θ = (π / 2) := 
by 
  -- Placeholder for the proof
  sorry

end slope_angle_of_perpendicular_line_l208_208859


namespace find_ellipse_equation_sum_of_slopes_constant_l208_208604

noncomputable def ellipse_pass_through (A : ℝ × ℝ) (a b : ℝ) (e : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 
  (A.fst ^ 2 / a ^ 2 + A.snd ^ 2 / b ^ 2 = 1) ∧ 
  (e = Real.sqrt 2 / 2)

theorem find_ellipse_equation {A : ℝ × ℝ} {a b : ℝ} (h : ellipse_pass_through A a b (Real.sqrt 2 / 2)) :
  a = Real.sqrt 2 ∧ b = 1 :=
sorry

def line_passing_through (P : ℝ × ℝ) (k : ℝ) : Prop := 
  ∃ Q : ℝ × ℝ, P ≠ Q ∧ 
  ∃ m n : ℝ, m ≠ 0 ∧ P = (m, k * (m - 1) + 1) ∧ Q = (n, k * (n - 1) + 1)

theorem sum_of_slopes_constant {a b : ℝ} (h₁ : a = Real.sqrt 2) (h₂ : b = 1) (k : ℝ) :
  let E : ℝ × ℝ → Prop := λ (P : ℝ × ℝ), 
    (P.fst ^ 2 / h₁ ^ 2 + P.snd ^ 2 / h₂ ^ 2 = 1)
  ∀ (P Q : ℝ × ℝ), 
    E P → E Q → P ≠ (0, -1) → Q ≠ (0, -1) →
    line_passing_through (1, 1) k →
    (P = (x1, y1)) → (Q = (x2, y2)) →
    ((y1 + 1) / x1 + (y2 + 1) / x2 = 2) :=
sorry

end find_ellipse_equation_sum_of_slopes_constant_l208_208604


namespace chord_length_l208_208994

noncomputable def circle_center (c: ℝ × ℝ) (r: ℝ): Prop := 
  ∃ x y: ℝ, 
    (x - c.1)^2 + (y - c.2)^2 = r^2

noncomputable def line_equation (a b c: ℝ): Prop := 
  ∀ x y: ℝ, 
    a*x + b*y + c = 0

theorem chord_length (a: ℝ): 
  circle_center (2, 1) 2 ∧ line_equation a 1 (-5) ∧
  ∃(chord_len: ℝ), chord_len = 4 → 
  a = 2 :=
by
  sorry

end chord_length_l208_208994


namespace perimeter_ratio_of_squares_l208_208139

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l208_208139


namespace largest_of_sums_l208_208597

noncomputable def a1 := (1 / 4 : ℚ) + (1 / 5 : ℚ)
noncomputable def a2 := (1 / 4 : ℚ) + (1 / 6 : ℚ)
noncomputable def a3 := (1 / 4 : ℚ) + (1 / 3 : ℚ)
noncomputable def a4 := (1 / 4 : ℚ) + (1 / 8 : ℚ)
noncomputable def a5 := (1 / 4 : ℚ) + (1 / 7 : ℚ)

theorem largest_of_sums :
  max a1 (max a2 (max a3 (max a4 a5))) = 7 / 12 :=
by sorry

end largest_of_sums_l208_208597


namespace A_walking_speed_l208_208046

-- Definition for the conditions
def A_speed (v : ℝ) : Prop := 
  ∃ (t : ℝ), 120 = 20 * (t - 6) ∧ 120 = v * t

-- The main theorem to prove the question
theorem A_walking_speed : ∀ (v : ℝ), A_speed v → v = 10 :=
by
  intros v h
  sorry

end A_walking_speed_l208_208046


namespace set_of_integers_between_10_and_16_l208_208918

theorem set_of_integers_between_10_and_16 :
  {x : ℤ | 10 < x ∧ x < 16} = {11, 12, 13, 14, 15} :=
by
  sorry

end set_of_integers_between_10_and_16_l208_208918


namespace find_x_l208_208159

theorem find_x (p q r x : ℝ) (h1 : (p + q + r) / 3 = 4) (h2 : (p + q + r + x) / 4 = 5) : x = 8 :=
sorry

end find_x_l208_208159


namespace page_number_counted_twice_l208_208295

theorem page_number_counted_twice {n x : ℕ} (h₁ : n = 70) (h₂ : x > 0) (h₃ : x ≤ n) (h₄ : 2550 = n * (n + 1) / 2 + x) : x = 65 :=
by {
  sorry
}

end page_number_counted_twice_l208_208295


namespace ratio_of_frank_to_joystick_l208_208564

-- Define the costs involved
def cost_table : ℕ := 140
def cost_chair : ℕ := 100
def cost_joystick : ℕ := 20
def diff_spent : ℕ := 30

-- Define the payments
def F_j := 5
def E_j := 15

-- The ratio we need to prove
def ratio_frank_to_total_joystick (F_j : ℕ) (total_joystick : ℕ) : (ℕ × ℕ) :=
  (F_j / Nat.gcd F_j total_joystick, total_joystick / Nat.gcd F_j total_joystick)

theorem ratio_of_frank_to_joystick :
  let F_j := 5
  let total_joystick := 20
  ratio_frank_to_total_joystick F_j total_joystick = (1, 4) := by
  sorry

end ratio_of_frank_to_joystick_l208_208564


namespace half_percent_to_decimal_l208_208436

def percent_to_decimal (x : ℚ) : ℚ := x / 100

theorem half_percent_to_decimal : percent_to_decimal (1 / 2) = 0.005 :=
by
  sorry

end half_percent_to_decimal_l208_208436


namespace weight_of_new_boy_l208_208992

theorem weight_of_new_boy (W : ℕ) (original_weight : ℕ) (total_new_weight : ℕ)
  (h_original_avg : original_weight = 5 * 35)
  (h_new_avg : total_new_weight = 6 * 36)
  (h_new_weight : total_new_weight = original_weight + W) :
  W = 41 := by
  sorry

end weight_of_new_boy_l208_208992


namespace smallest_integer_satisfying_conditions_l208_208428

theorem smallest_integer_satisfying_conditions :
  ∃ M : ℕ, M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 ∧ M % 12 = 11 ∧ M = 27719 := by
  sorry

end smallest_integer_satisfying_conditions_l208_208428


namespace probability_no_defective_pens_selected_l208_208947

noncomputable def probability_not_defective (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) : ℚ :=
  let non_defective_pens := total_pens - defective_pens
  let first_prob := non_defective_pens / total_pens
  let remaining_pens := total_pens - 1
  let remaining_non_defective_pens := non_defective_pens - 1
  let second_prob := remaining_non_defective_pens / remaining_pens
  first_prob * second_prob

theorem probability_no_defective_pens_selected :
  probability_not_defective 12 3 2 = 6 / 11 :=
by
  sorry

end probability_no_defective_pens_selected_l208_208947


namespace solution_set_empty_l208_208298

-- Define the quadratic polynomial
def quadratic (x : ℝ) : ℝ := x^2 - 2*x + 3

-- State the theorem that the solution set of the given inequality is empty
theorem solution_set_empty : ∀ x : ℝ, quadratic x < 0 → false :=
by
  intro x
  unfold quadratic
  sorry

end solution_set_empty_l208_208298


namespace find_number_of_dogs_l208_208118

variables (D P S : ℕ)
theorem find_number_of_dogs (h1 : D = 2 * P) (h2 : P = 2 * S) (h3 : 4 * D + 4 * P + 2 * S = 510) :
  D = 60 := 
sorry

end find_number_of_dogs_l208_208118


namespace Janice_earnings_l208_208965

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l208_208965


namespace jebb_take_home_pay_is_4620_l208_208979

noncomputable def gross_salary : ℤ := 6500
noncomputable def federal_tax (income : ℤ) : ℤ :=
  let tax1 := min income 2000 * 10 / 100
  let tax2 := min (max (income - 2000) 0) 2000 * 15 / 100
  let tax3 := max (income - 4000) 0 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def health_insurance : ℤ := 300
noncomputable def retirement_contribution (income : ℤ) : ℤ := income * 7 / 100

noncomputable def total_deductions (income : ℤ) : ℤ :=
  federal_tax income + health_insurance + retirement_contribution income

noncomputable def take_home_pay (income : ℤ) : ℤ :=
  income - total_deductions income

theorem jebb_take_home_pay_is_4620 : take_home_pay gross_salary = 4620 := by
  sorry

end jebb_take_home_pay_is_4620_l208_208979


namespace not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l208_208669

theorem not_divisible_by_5_square_plus_or_minus_1_divisible_by_5 (a : ℤ) (h : a % 5 ≠ 0) :
  (a^2 + 1) % 5 = 0 ∨ (a^2 - 1) % 5 = 0 :=
by
  sorry

end not_divisible_by_5_square_plus_or_minus_1_divisible_by_5_l208_208669


namespace min_value_fractions_l208_208524

open Real

theorem min_value_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) :
  3 ≤ (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a)) :=
sorry

end min_value_fractions_l208_208524


namespace series_largest_prime_factor_of_111_l208_208200

def series := [368, 689, 836]  -- given sequence series

def div_condition (n : Nat) := 
  ∃ k : Nat, n = 111 * k

def largest_prime_factor (n : Nat) (p : Nat) := 
  Prime p ∧ ∀ q : Nat, Prime q → q ∣ n → q ≤ p

theorem series_largest_prime_factor_of_111 :
  largest_prime_factor 111 37 := 
by
  sorry

end series_largest_prime_factor_of_111_l208_208200


namespace add_in_base_7_l208_208343

theorem add_in_base_7 (X Y : ℕ) (h1 : (X + 5) % 7 = 0) (h2 : (Y + 2) % 7 = X) : X + Y = 2 :=
by
  sorry

end add_in_base_7_l208_208343


namespace roots_of_equation_l208_208556

theorem roots_of_equation :
  (∃ (x_1 x_2 : ℝ), x_1 > x_2 ∧ (∀ x, x^2 - |x-1| - 1 = 0 ↔ x = x_1 ∨ x = x_2)) :=
sorry

end roots_of_equation_l208_208556


namespace borel_cantelli_variant_l208_208540

variables {Ω : Type*} {A : ℕ → Set Ω} {P : MeasureTheory.Measure Ω}

-- Defining the given conditions
def condition1 : Prop := ∀ ε > 0, ∃ N, ∀ n ≥ N, P (A n) < ε
def condition2 : Prop := ∑' n, P (A n \ A (n + 1)) < ∞

-- The main theorem statement
theorem borel_cantelli_variant (h1 : condition1) (h2 : condition2) :
  P (Set.Union (λ n, Set.Inter (Set.Icc n (∞)) A)) = 0 :=
sorry

end borel_cantelli_variant_l208_208540


namespace Sandy_record_age_l208_208688

theorem Sandy_record_age :
  ∀ (current_length age_years : ℕ) (goal_length : ℕ) (growth_rate tenths_per_month : ℕ),
  goal_length = 26 ∧
  current_length = 2 ∧
  age_years = 12 ∧
  growth_rate = 10 *
  tenths_per_month / 10 →
  age_years + (goal_length - current_length) * 10 / growth_rate = 32 :=
by
  intros,
  sorry

end Sandy_record_age_l208_208688


namespace fg_of_neg5_eq_484_l208_208499

def f (x : Int) : Int := x * x
def g (x : Int) : Int := 6 * x + 8

theorem fg_of_neg5_eq_484 : f (g (-5)) = 484 := 
  sorry

end fg_of_neg5_eq_484_l208_208499


namespace balcony_more_than_orchestra_l208_208204

variables (O B : ℕ) (H1 : O + B = 380) (H2 : 12 * O + 8 * B = 3320)

theorem balcony_more_than_orchestra : B - O = 240 :=
by sorry

end balcony_more_than_orchestra_l208_208204


namespace min_value_of_f_l208_208698

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l208_208698


namespace smallest_r_l208_208623

variables (p q r s : ℤ)

-- Define the conditions
def condition1 : Prop := p + 3 = q - 1
def condition2 : Prop := p + 3 = r + 5
def condition3 : Prop := p + 3 = s - 2

-- Prove that r is the smallest
theorem smallest_r (h1 : condition1 p q) (h2 : condition2 p r) (h3 : condition3 p s) : r < p ∧ r < q ∧ r < s :=
sorry

end smallest_r_l208_208623


namespace part_a_part_b_l208_208018

-- Part (a)
theorem part_a (x : ℝ) (h : x > 0) : x^3 - 3*x ≥ -2 :=
sorry

-- Part (b)
theorem part_b (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) + 2 * ((y / (x * z)) + (z / (x * y)) + (x / (y * z))) ≥ 9 :=
sorry

end part_a_part_b_l208_208018


namespace intersection_points_calculation_l208_208239

-- Define the quadratic function and related functions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def u (a b c x : ℝ) : ℝ := - f a b c (-x)
def v (a b c x : ℝ) : ℝ := f a b c (x + 1)

-- Define the number of intersection points
def m : ℝ := 1
def n : ℝ := 0

-- The proof goal
theorem intersection_points_calculation (a b c : ℝ) : 7 * m + 3 * n = 7 :=
by sorry

end intersection_points_calculation_l208_208239


namespace min_abs_sum_is_5_l208_208701

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l208_208701


namespace largest_fraction_l208_208958

theorem largest_fraction (d x : ℕ) 
  (h1: (2 * x / d) + (3 * x / d) + (4 * x / d) = 10 / 11)
  (h2: d = 11 * x) : (4 / 11 : ℚ) = (4 * x / d : ℚ) :=
by
  sorry

end largest_fraction_l208_208958


namespace vector_product_magnitude_l208_208501

noncomputable def vector_magnitude (a b : ℝ) (theta : ℝ) : ℝ :=
  abs a * abs b * Real.sin theta

theorem vector_product_magnitude 
  (a b : ℝ) 
  (theta : ℝ) 
  (ha : abs a = 4) 
  (hb : abs b = 3) 
  (h_dot : a * b = -2) 
  (theta_range : 0 ≤ theta ∧ theta ≤ Real.pi)
  (cos_theta : Real.cos theta = -1/6) 
  (sin_theta : Real.sin theta = Real.sqrt 35 / 6) :
  vector_magnitude a b theta = 2 * Real.sqrt 35 :=
sorry

end vector_product_magnitude_l208_208501


namespace titu_andreescu_inequality_l208_208732

theorem titu_andreescu_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 :=
sorry

end titu_andreescu_inequality_l208_208732


namespace log_max_reciprocal_min_l208_208068

open Real

-- Definitions for the conditions
variables (x y : ℝ)
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ 2 * x + 5 * y = 20

-- Theorem statement for the first question
theorem log_max (x y : ℝ) (h : conditions x y) : log x + log y ≤ 1 :=
sorry

-- Theorem statement for the second question
theorem reciprocal_min (x y : ℝ) (h : conditions x y) : (1 / x) + (1 / y) ≥ (7 + 2 * sqrt 10) / 20 :=
sorry

end log_max_reciprocal_min_l208_208068


namespace probability_blue_face_l208_208053

theorem probability_blue_face :
  (3 / 6 : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end probability_blue_face_l208_208053


namespace optimal_route_l208_208887

-- Define the probabilities of no traffic jam on each road segment.
def P_AC : ℚ := 9 / 10
def P_CD : ℚ := 14 / 15
def P_DB : ℚ := 5 / 6
def P_CF : ℚ := 9 / 10
def P_FB : ℚ := 15 / 16
def P_AE : ℚ := 9 / 10
def P_EF : ℚ := 9 / 10
def P_FB2 : ℚ := 19 / 20  -- Alias for repeated probability

-- Define the probability of encountering a traffic jam on a route
def prob_traffic_jam (p_no_jam : ℚ) : ℚ := 1 - p_no_jam

-- Define the probabilities of encountering a traffic jam along each route.
def P_ACDB_jam : ℚ := prob_traffic_jam (P_AC * P_CD * P_DB)
def P_ACFB_jam : ℚ := prob_traffic_jam (P_AC * P_CF * P_FB)
def P_AEFB_jam : ℚ := prob_traffic_jam (P_AE * P_EF * P_FB2)

-- State the theorem to prove the optimal route
theorem optimal_route : P_ACDB_jam < P_ACFB_jam ∧ P_ACDB_jam < P_AEFB_jam :=
by { sorry }

end optimal_route_l208_208887


namespace annie_blocks_walked_l208_208332

theorem annie_blocks_walked (x : ℕ) (h1 : 7 * 2 = 14) (h2 : 2 * x + 14 = 24) : x = 5 :=
by
  sorry

end annie_blocks_walked_l208_208332


namespace simplify_fraction_l208_208276

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208276


namespace remaining_subtasks_l208_208517

def total_problems : ℝ := 72.0
def finished_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks :
    (total_problems * subtasks_per_problem - finished_problems * subtasks_per_problem) = 200 := 
by
  sorry

end remaining_subtasks_l208_208517


namespace simplify_fraction_l208_208279

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208279


namespace triangle_centroid_altitude_l208_208957

/-- In triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25, the length of GQ where Q 
    is the foot of the altitude from the centroid G to the side YZ is 56/25. -/
theorem triangle_centroid_altitude :
  let XY := 7
  let XZ := 24
  let YZ := 25
  let GQ := 56 / 25
  GQ = (56 : ℝ) / 25 :=
by
  -- proof goes here
  sorry

end triangle_centroid_altitude_l208_208957


namespace train_length_l208_208435

theorem train_length
  (S : ℝ)  -- speed of the train in meters per second
  (L : ℝ)  -- length of the train in meters
  (h1 : L = S * 20)
  (h2 : L + 500 = S * 40) :
  L = 500 := 
sorry

end train_length_l208_208435


namespace simplify_expression_l208_208270

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l208_208270


namespace evaluate_expression_l208_208906

theorem evaluate_expression : 
  ( (2^12)^2 - (2^10)^2 ) / ( (2^11)^2 - (2^9)^2 ) = 4 :=
by
  sorry

end evaluate_expression_l208_208906


namespace range_of_a_l208_208943

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 :=
by
  sorry

end range_of_a_l208_208943


namespace male_athletes_drawn_l208_208378

theorem male_athletes_drawn (total_males : ℕ) (total_females : ℕ) (total_sample : ℕ)
  (h_males : total_males = 20) (h_females : total_females = 10) (h_sample : total_sample = 6) :
  (total_sample * total_males) / (total_males + total_females) = 4 := 
  by
  sorry

end male_athletes_drawn_l208_208378


namespace expected_number_of_remaining_bullets_l208_208327

-- Conditions given in the problem
def hit_rate : ℝ := 0.6
def total_bullets : ℕ := 4
def probability_of_miss : ℝ := 1 - hit_rate

-- Definition of possible values of remaining bullets and their probabilities
def prob_ξ (k : ℕ) : ℝ :=
  match k with
  | 0 => probability_of_miss ^ 3 * hit_rate
  | 1 => probability_of_miss ^ 2 * hit_rate * probability_of_miss
  | 2 => probability_of_miss * hit_rate * probability_of_miss
  | 3 => hit_rate
  | _ => 0

-- Expected value calculation
def expected_ξ : ℝ :=
  (0 * prob_ξ 0) + (1 * prob_ξ 1) + (2 * prob_ξ 2) + (3 * prob_ξ 3)

-- The proposition to prove
theorem expected_number_of_remaining_bullets : expected_ξ = 2.376 := by
  sorry

end expected_number_of_remaining_bullets_l208_208327


namespace total_books_bought_l208_208180

-- Let x be the number of math books and y be the number of history books
variables (x y : ℕ)

-- Conditions
def math_book_cost := 4
def history_book_cost := 5
def total_price := 368
def num_math_books := 32

-- The total number of books bought is the sum of the number of math books and history books, which should result in 80
theorem total_books_bought : 
  y * history_book_cost + num_math_books * math_book_cost = total_price → 
  x = num_math_books → 
  x + y = 80 :=
by
  sorry

end total_books_bought_l208_208180


namespace proportion_of_adopted_kittens_l208_208976

-- Define the relevant objects and conditions in Lean
def breeding_rabbits : ℕ := 10
def kittens_first_spring := 10 * breeding_rabbits -- 100 kittens
def kittens_second_spring : ℕ := 60
def adopted_first_spring (P : ℝ) := 100 * P
def returned_first_spring : ℕ := 5
def adopted_second_spring : ℕ := 4
def total_rabbits_in_house (P : ℝ) :=
  breeding_rabbits + (kittens_first_spring - adopted_first_spring P + returned_first_spring) +
  (kittens_second_spring - adopted_second_spring)

theorem proportion_of_adopted_kittens : ∃ (P : ℝ), total_rabbits_in_house P = 121 ∧ P = 0.5 :=
by
  use 0.5
  -- Proof part (with "sorry" to skip the detailed proof)
  sorry

end proportion_of_adopted_kittens_l208_208976


namespace idempotent_mappings_count_l208_208035

noncomputable def num_idempotent_mappings (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), Nat.choose n k * k^(n - k)

theorem idempotent_mappings_count (X : Type) [Fintype X] [DecidableEq X] (f : X → X)
  (h : ∀ x, f (f x) = f x) (n : ℕ) (hn : Fintype.card X = n) :
  ∃ num_mappings : ℕ, num_mappings = num_idempotent_mappings n :=
by
  use num_idempotent_mappings n
  sorry

end idempotent_mappings_count_l208_208035


namespace ab_inequality_l208_208122

theorem ab_inequality
  {a b : ℝ}
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_b_sum : a + b = 2) :
  ∀ n : ℕ, 2 ≤ n → (a^n + 1) * (b^n + 1) ≥ 4 :=
by
  sorry

end ab_inequality_l208_208122


namespace calculate_jessie_points_l208_208843

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l208_208843


namespace eccentricity_of_given_ellipse_l208_208161

noncomputable def ellipse_eccentricity (φ : Real) : Real :=
  let x := 3 * Real.cos φ
  let y := 5 * Real.sin φ
  let a := 5
  let b := 3
  let c := Real.sqrt (a * a - b * b)
  c / a

theorem eccentricity_of_given_ellipse (φ : Real) :
  ellipse_eccentricity φ = 4 / 5 :=
sorry

end eccentricity_of_given_ellipse_l208_208161


namespace min_bottles_needed_l208_208598

theorem min_bottles_needed (fluid_ounces_needed : ℝ) (bottle_size_ml : ℝ) (conversion_factor : ℝ) :
  fluid_ounces_needed = 60 ∧ bottle_size_ml = 250 ∧ conversion_factor = 33.8 →
  ∃ (n : ℕ), n = 8 ∧ (fluid_ounces_needed / conversion_factor * 1000 / bottle_size_ml) <= ↑n :=
by
  sorry

end min_bottles_needed_l208_208598


namespace intersection_complement_l208_208492

open Set

/-- The universal set U as the set of all real numbers -/
def U : Set ℝ := @univ ℝ

/-- The set M -/
def M : Set ℝ := {-1, 0, 1}

/-- The set N defined by the equation x^2 + x = 0 -/
def N : Set ℝ := {x | x^2 + x = 0}

/-- The complement of set N in the universal set U -/
def C_U_N : Set ℝ := {x ∈ U | x ≠ -1 ∧ x ≠ 0}

theorem intersection_complement :
  M ∩ C_U_N = {1} :=
by
  sorry

end intersection_complement_l208_208492


namespace tangent_line_through_origin_l208_208684

noncomputable def curve (x : ℝ) : ℝ := Real.exp (x - 1) + x

theorem tangent_line_through_origin :
  ∃ k : ℝ, k = 2 ∧ ∀ x y : ℝ, (y = k * x) ↔ (∃ m : ℝ, curve m = m + Real.exp (m - 1) ∧ (curve m) = (m + Real.exp (m - 1)) ∧ k = (Real.exp (m - 1) + 1) ∧ y = k * x ∧ y = 2*x) :=
by 
  sorry

end tangent_line_through_origin_l208_208684


namespace greatest_k_inequality_l208_208225

theorem greatest_k_inequality :
  ∃ k : ℕ, k = 13 ∧ ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a * b * c = 1 → 
  (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4) :=
sorry

end greatest_k_inequality_l208_208225


namespace olaf_total_cars_l208_208662

noncomputable def olaf_initial_cars : ℕ := 150
noncomputable def uncle_cars : ℕ := 5
noncomputable def grandpa_cars : ℕ := 2 * uncle_cars
noncomputable def dad_cars : ℕ := 10
noncomputable def mum_cars : ℕ := dad_cars + 5
noncomputable def auntie_cars : ℕ := 6
noncomputable def liam_cars : ℕ := dad_cars / 2
noncomputable def emma_cars : ℕ := uncle_cars / 3
noncomputable def grandma_cars : ℕ := 3 * auntie_cars

noncomputable def total_gifts : ℕ := 
  grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars + liam_cars + emma_cars + grandma_cars

noncomputable def total_cars_after_gifts : ℕ := olaf_initial_cars + total_gifts

theorem olaf_total_cars : total_cars_after_gifts = 220 := by
  sorry

end olaf_total_cars_l208_208662


namespace silk_diameter_scientific_notation_l208_208404

-- Definition of the given condition
def silk_diameter := 0.000014 

-- The goal to be proved
theorem silk_diameter_scientific_notation : silk_diameter = 1.4 * 10^(-5) := 
by 
  sorry

end silk_diameter_scientific_notation_l208_208404


namespace simplify_expression_l208_208271

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l208_208271


namespace quotient_of_sum_l208_208440

theorem quotient_of_sum (a b c x y z : ℝ)
  (h1 : a^2 + b^2 + c^2 = 25)
  (h2 : x^2 + y^2 + z^2 = 36)
  (h3 : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = 5 / 6 :=
by
  sorry

end quotient_of_sum_l208_208440


namespace correct_propositions_l208_208627

variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Assume basic predicates for lines and planes
variable (parallel : Line → Line → Prop)
variable (perp : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (planar_parallel : Plane → Plane → Prop)

-- Stating the theorem to be proved
theorem correct_propositions :
  (parallel m n ∧ perp m α → perp n α) ∧ 
  (planar_parallel α β ∧ parallel m n ∧ perp m α → perp n β) :=
by
  sorry

end correct_propositions_l208_208627


namespace max_gold_coins_l208_208309

theorem max_gold_coins (k : ℕ) (n : ℕ) (h : n = 13 * k + 3 ∧ n < 150) : n = 146 :=
by 
  sorry

end max_gold_coins_l208_208309


namespace permissible_m_values_l208_208768

theorem permissible_m_values :
  ∀ (m : ℕ) (a : ℝ), 
  (∃ k, 2 ≤ k ∧ k ≤ 4 ∧ (3 / (6 / (2 * m + 1)) ≤ k)) → m = 2 ∨ m = 3 :=
by
  sorry

end permissible_m_values_l208_208768


namespace number_of_good_subsets_of_S_l208_208656

open Finset

def S : Finset ℕ := (finset.range 50).image (λ n, n + 1)

def is_good (s : Finset ℕ) : Prop := s.card = 3 ∧ (s.sum id) % 3 = 0

def count_good_subsets : ℕ := (S.subsets_of_card 3).filter is_good |>.card

theorem number_of_good_subsets_of_S :
  count_good_subsets = 6544 := by
  sorry

end number_of_good_subsets_of_S_l208_208656


namespace checkered_board_cut_l208_208611

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l208_208611


namespace valid_combinations_count_l208_208894

theorem valid_combinations_count : 
  let wrapping_paper_count := 10
  let ribbon_count := 3
  let gift_card_count := 5
  let invalid_combinations := 1 -- red ribbon with birthday card
  let total_combinations := wrapping_paper_count * ribbon_count * gift_card_count
  total_combinations - invalid_combinations = 149 := 
by 
  sorry

end valid_combinations_count_l208_208894


namespace isosceles_triangle_perimeter_l208_208639

variable (a b c : ℝ) (h_iso : a = b ∨ a = c ∨ b = c) (h_a : a = 6) (h_b : b = 6) (h_c : c = 3)
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem isosceles_triangle_perimeter : a + b + c = 15 :=
by 
  -- Given definitions and triangle inequality
  have h_valid : a = 6 ∧ b = 6 ∧ c = 3 := ⟨h_a, h_b, h_c⟩
  sorry

end isosceles_triangle_perimeter_l208_208639


namespace squirrel_walnuts_l208_208315

theorem squirrel_walnuts :
  let boy_gathered := 6
  let boy_dropped := 1
  let initial_in_burrow := 12
  let girl_brought := 5
  let girl_ate := 2
  initial_in_burrow + (boy_gathered - boy_dropped) + girl_brought - girl_ate = 20 :=
by
  sorry

end squirrel_walnuts_l208_208315


namespace population_growth_l208_208477

theorem population_growth (scale_factor1 scale_factor2 : ℝ)
    (h1 : scale_factor1 = 1.2)
    (h2 : scale_factor2 = 1.26) :
    (scale_factor1 * scale_factor2) - 1 = 0.512 :=
by
  sorry

end population_growth_l208_208477


namespace total_rainfall_l208_208916

theorem total_rainfall (R1 R2 : ℝ) (h1 : R2 = 1.5 * R1) (h2 : R2 = 15) : R1 + R2 = 25 := 
by
  sorry

end total_rainfall_l208_208916


namespace triangle_inequality_x_values_l208_208580

theorem triangle_inequality_x_values :
  {x : ℕ | 1 ≤ x ∧ x < 14} = {x | x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13} :=
  by
    sorry

end triangle_inequality_x_values_l208_208580


namespace economic_rationale_education_policy_l208_208953

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l208_208953


namespace quadratic_difference_l208_208972

theorem quadratic_difference (f : ℝ → ℝ) (hpoly : ∃ c d e : ℤ, ∀ x, f x = c*x^2 + d*x + e) 
(h : f (Real.sqrt 3) - f (Real.sqrt 2) = 4) : 
f (Real.sqrt 10) - f (Real.sqrt 7) = 12 := sorry

end quadratic_difference_l208_208972


namespace savings_by_paying_cash_l208_208827

def cash_price := 400
def down_payment := 120
def monthly_installment := 30
def number_of_months := 12

theorem savings_by_paying_cash :
  let total_cost_plan := down_payment + (monthly_installment * number_of_months) in
  let savings := total_cost_plan - cash_price in
  savings = 80 := 
by
  let total_cost_plan := down_payment + monthly_installment * number_of_months
  let savings := total_cost_plan - cash_price
  sorry

end savings_by_paying_cash_l208_208827


namespace average_incorrect_l208_208182

theorem average_incorrect : ¬( (1 + 1 + 0 + 2 + 4) / 5 = 2) :=
by {
  sorry
}

end average_incorrect_l208_208182


namespace probability_of_collinear_dots_in_5x5_grid_l208_208956

-- The Lean code to represent the problem and statement 
theorem probability_of_collinear_dots_in_5x5_grid :
  let total_dots := 25
  let dots_chosen := 4
  let collinear_sets := 14
  let total_combinations := Nat.choose total_dots dots_chosen
  let probability := (collinear_sets : ℚ) / total_combinations in
  probability = 7 / 6325 :=
begin
  sorry
end

end probability_of_collinear_dots_in_5x5_grid_l208_208956


namespace population_net_increase_per_day_l208_208878

theorem population_net_increase_per_day (birth_rate death_rate : ℚ) (seconds_per_day : ℕ) (net_increase : ℚ) :
  birth_rate = 7 / 2 ∧
  death_rate = 2 / 2 ∧
  seconds_per_day = 24 * 60 * 60 ∧
  net_increase = (birth_rate - death_rate) * seconds_per_day →
  net_increase = 216000 := 
by
  sorry

end population_net_increase_per_day_l208_208878


namespace two_digit_number_reversed_l208_208012

theorem two_digit_number_reversed :
  ∃ (x y : ℕ), (10 * x + y = 73) ∧ (10 * x + y = 2 * (10 * y + x) - 1) ∧ (x < 10) ∧ (y < 10) := 
by
  sorry

end two_digit_number_reversed_l208_208012


namespace determine_constants_l208_208338

structure Vector2D :=
(x : ℝ)
(y : ℝ)

def a := 11 / 20
def b := -7 / 20

def v1 : Vector2D := ⟨3, 2⟩
def v2 : Vector2D := ⟨-1, 6⟩
def v3 : Vector2D := ⟨2, -1⟩

def linear_combination (v1 v2 : Vector2D) (a b : ℝ) : Vector2D :=
  ⟨a * v1.x + b * v2.x, a * v1.y + b * v2.y⟩

theorem determine_constants (a b : ℝ) :
  ∃ (a b : ℝ), linear_combination v1 v2 a b = v3 :=
by
  use (11 / 20)
  use (-7 / 20)
  sorry

end determine_constants_l208_208338


namespace polynomial_inequality_l208_208395

noncomputable def F (x a_3 a_2 a_1 k : ℝ) : ℝ :=
  x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + k^4

theorem polynomial_inequality 
  (p k : ℝ) 
  (a_3 a_2 a_1 : ℝ) 
  (h_p : 0 < p) 
  (h_k : 0 < k) 
  (h_roots : ∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧
    F (-x1) a_3 a_2 a_1 k = 0 ∧
    F (-x2) a_3 a_2 a_1 k = 0 ∧
    F (-x3) a_3 a_2 a_1 k = 0 ∧
    F (-x4) a_3 a_2 a_1 k = 0) :
  F p a_3 a_2 a_1 k ≥ (p + k)^4 := 
sorry

end polynomial_inequality_l208_208395


namespace max_distance_eq_of_l1_l208_208245

noncomputable def equation_of_l1 (l1 l2 : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (2, 4) ∧ -- Points A and B
  l1 A.1 = A.2 ∧ l2 B.1 = B.2 ∧ -- l1 passes through A and l2 passes through B
  (∀ (x : ℝ), l1 x - l2 x = 1) ∧ -- l1 and l2 are parallel (constant difference in y-values)
  (∃ (c : ℝ), ∀ (x : ℝ), l1 x = -x + c ∧ l2 x = -x + c + 1) -- distance maximized

theorem max_distance_eq_of_l1 : 
  ∃ (l1 l2 : ℝ → ℝ), equation_of_l1 l1 l2 (1, 3) (2, 4) ∧
  ∀ (x : ℝ), l1 x = -x + 4 := 
sorry

end max_distance_eq_of_l1_l208_208245


namespace probability_third_draw_first_class_expected_value_first_class_in_10_draws_l208_208363

-- Define the problem with products
structure Products where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

-- Given products configuration
def products : Products := { total := 5, first_class := 3, second_class := 2 }

-- Probability calculation without replacement
-- Define the event of drawing
def draw_without_replacement (p : Products) (draws : ℕ) (desired_event : ℕ -> Bool) : ℚ := 
  if draws = 3 ∧ desired_event 3 ∧ ¬ desired_event 1 ∧ ¬ desired_event 2 then
    (2 / 5) * ((1 : ℚ) / 4) * (3 / 3)
  else 
    0

-- Define desired_event for the specific problem
def desired_event (n : ℕ) : Bool := 
  match n with
  | 3 => true
  | _ => false

-- The first problem's proof statement
theorem probability_third_draw_first_class : draw_without_replacement products 3 desired_event = 1 / 10 := sorry

-- Expected value calculation with replacement
-- Binomial distribution to find expected value
def expected_value_with_replacement (p : Products) (draws : ℕ) : ℚ :=
  draws * (p.first_class / p.total)

-- The second problem's proof statement
theorem expected_value_first_class_in_10_draws : expected_value_with_replacement products 10 = 6 := sorry

end probability_third_draw_first_class_expected_value_first_class_in_10_draws_l208_208363


namespace age_of_B_l208_208889

theorem age_of_B (A B C : ℕ) (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : A + B + C = 37) : B = 14 :=
by sorry

end age_of_B_l208_208889


namespace ratio_identity_l208_208227

-- Given system of equations
def system_of_equations (k : ℚ) (x y z : ℚ) :=
  x + k * y + 2 * z = 0 ∧
  2 * x + k * y + 3 * z = 0 ∧
  3 * x + 5 * y + 4 * z = 0

-- Prove that for k = -7/5, the system has a nontrivial solution and 
-- that the ratio xz / y^2 equals -25
theorem ratio_identity (x y z : ℚ) (k : ℚ) (h : system_of_equations k x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  k = -7 / 5 → x * z / y^2 = -25 :=
by
  sorry

end ratio_identity_l208_208227


namespace triangle_properties_l208_208045

-- Define the given sides of the triangle
def a := 6
def b := 8
def c := 10

-- Define necessary parameters and properties
def isRightTriangle (a b c : Nat) : Prop := a^2 + b^2 = c^2
def area (a b : Nat) : Nat := (a * b) / 2
def semiperimeter (a b c : Nat) : Nat := (a + b + c) / 2
def inradius (A s : Nat) : Nat := A / s
def circumradius (c : Nat) : Nat := c / 2

-- The theorem statement
theorem triangle_properties :
  isRightTriangle a b c ∧
  area a b = 24 ∧
  semiperimeter a b c = 12 ∧
  inradius (area a b) (semiperimeter a b c) = 2 ∧
  circumradius c = 5 :=
by
  sorry

end triangle_properties_l208_208045


namespace triangle_angle_C_l208_208811

theorem triangle_angle_C (A B C : ℝ) (sin cos : ℝ → ℝ) 
  (h1 : 3 * sin A + 4 * cos B = 6)
  (h2 : 4 * sin B + 3 * cos A = 1)
  (triangle_sum : A + B + C = 180) :
  C = 30 :=
by
  sorry

end triangle_angle_C_l208_208811


namespace point_not_in_second_quadrant_l208_208250

theorem point_not_in_second_quadrant (m : ℝ) : ¬ (m^2 + m ≤ 0 ∧ m - 1 ≥ 0) :=
by
  sorry

end point_not_in_second_quadrant_l208_208250


namespace daily_production_l208_208885

-- Definitions based on conditions
def weekly_production : ℕ := 3400
def working_days_in_week : ℕ := 5

-- Statement to prove the number of toys produced each day
theorem daily_production : (weekly_production / working_days_in_week) = 680 :=
by
  sorry

end daily_production_l208_208885


namespace expectation_of_X_is_15_max_P_a_le_X_le_b_additional_workers_needed_l208_208735

noncomputable def repair_data := [9, 15, 12, 18, 12, 18, 9, 9, 24, 12, 12, 24, 15, 15, 15, 12, 15, 15, 15, 24]

def X_distribution_table : List (ℚ × ℚ) :=
[(9, 3/20), (12, 5/20), (15, 7/20), (18, 2/20), (24, 3/20)]

def expectation_X : ℚ :=
9 * (3/20) + 12 * (5/20) + 15 * (7/20) + 18 * (2/20) + 24 * (3/20)

theorem expectation_of_X_is_15 : expectation_X = 15 :=
by
  sorry

def P_a_le_X_le_b (a b : ℕ) : ℚ :=
if (a, b) = (9, 15) then (3/20) + (5/20) + (7/20)
else if (a, b) = (12, 18) then (5/20) + (7/20) + (2/20)
else if (a, b) = (18, 24) then (2/20) + (3/20)
else 0

theorem max_P_a_le_X_le_b : 
  ∃ a b : ℕ, b - a = 6 ∧ P_a_le_X_le_b a b = 3/4 :=
by
  use 9, 15
  split
  . rfl
  . sorry

def num_of_workers_needed (E : ℚ) (max_per_worker : ℚ) : ℚ :=
E / max_per_worker

theorem additional_workers_needed : 
  num_of_workers_needed 15 4 - 2 = 2 :=
by
  sorry

end expectation_of_X_is_15_max_P_a_le_X_le_b_additional_workers_needed_l208_208735


namespace simplify_and_evaluate_l208_208406

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  ( (x + 3) / x - 1 ) / ( (x^2 - 1) / (x^2 + x) ) = Real.sqrt 3 :=
by
  sorry

end simplify_and_evaluate_l208_208406


namespace hotdog_eating_ratio_l208_208803

variable (rate_first rate_second rate_third total_hotdogs time_minutes : ℕ)
variable (rate_ratio : ℕ)

def rate_first_eq : rate_first = 10 := by sorry
def rate_second_eq : rate_second = 3 * rate_first := by sorry
def total_hotdogs_eq : total_hotdogs = 300 := by sorry
def time_minutes_eq : time_minutes = 5 := by sorry
def rate_third_eq : rate_third = total_hotdogs / time_minutes := by sorry

theorem hotdog_eating_ratio :
  rate_ratio = rate_third / rate_second :=
  by sorry

end hotdog_eating_ratio_l208_208803


namespace shaina_chocolate_amount_l208_208651

variable (total_chocolate : ℚ) (num_piles : ℕ) (fraction_kept : ℚ)
variable (eq_total_chocolate : total_chocolate = 72 / 7)
variable (eq_num_piles : num_piles = 6)
variable (eq_fraction_kept : fraction_kept = 1 / 3)

theorem shaina_chocolate_amount :
  (total_chocolate / num_piles) * (1 - fraction_kept) = 8 / 7 :=
by
  sorry

end shaina_chocolate_amount_l208_208651


namespace decrement_from_each_observation_l208_208856

theorem decrement_from_each_observation (n : Nat) (mean_original mean_updated decrement : ℝ)
  (h1 : n = 50)
  (h2 : mean_original = 200)
  (h3 : mean_updated = 191)
  (h4 : decrement = 9) :
  (mean_original - mean_updated) * (n : ℝ) / n = decrement :=
by
  sorry

end decrement_from_each_observation_l208_208856


namespace reading_order_l208_208804

theorem reading_order (a b c d : ℝ) 
  (h1 : a + c = b + d) 
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end reading_order_l208_208804


namespace min_abs_sum_l208_208695

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l208_208695


namespace simplify_fraction_l208_208273

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208273


namespace blue_tint_percentage_in_new_mixture_l208_208882

-- Define the conditions given in the problem
def original_volume : ℝ := 40
def blue_tint_percentage : ℝ := 0.20
def added_blue_tint_volume : ℝ := 8

-- Calculate the original blue tint volume
def original_blue_tint_volume := blue_tint_percentage * original_volume

-- Calculate the new blue tint volume after adding more blue tint
def new_blue_tint_volume := original_blue_tint_volume + added_blue_tint_volume

-- Calculate the new total volume of the mixture
def new_total_volume := original_volume + added_blue_tint_volume

-- Define the expected result in percentage
def expected_blue_tint_percentage : ℝ := 33.3333

-- Statement to prove
theorem blue_tint_percentage_in_new_mixture :
  (new_blue_tint_volume / new_total_volume) * 100 = expected_blue_tint_percentage :=
sorry

end blue_tint_percentage_in_new_mixture_l208_208882


namespace regular_12gon_symmetry_and_angle_l208_208892

theorem regular_12gon_symmetry_and_angle :
  ∀ (L R : ℕ), 
  (L = 12) ∧ (R = 30) → 
  (L + R = 42) :=
by
  -- placeholder for the actual proof
  sorry

end regular_12gon_symmetry_and_angle_l208_208892


namespace num_values_sum_l208_208823

noncomputable def g : ℝ → ℝ :=
sorry

theorem num_values_sum (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (g x + y) = g (x + y) + x * g y - 2 * x * y - 2 * x + 2) :
  ∃ n s : ℕ, (n = 1 ∧ s = 3 ∧ n * s = 3) :=
sorry

end num_values_sum_l208_208823


namespace gcd_of_three_numbers_l208_208677

theorem gcd_of_three_numbers :
  Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
sorry

end gcd_of_three_numbers_l208_208677


namespace sum_of_fractions_irreducible_l208_208681

noncomputable def is_irreducible (num denom : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ num ∧ d ∣ denom → d = 1

theorem sum_of_fractions_irreducible (a b : ℕ) (h_coprime : Nat.gcd a b = 1) :
  is_irreducible (2 * a + b) (a * (a + b)) :=
by
  sorry

end sum_of_fractions_irreducible_l208_208681


namespace find_original_number_l208_208711

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l208_208711


namespace speed_of_stream_l208_208322

variable (b s : ℝ)

-- Conditions:
def downstream_eq : Prop := 90 = (b + s) * 3
def upstream_eq : Prop := 72 = (b - s) * 3

-- Goal:
theorem speed_of_stream (h1 : downstream_eq b s) (h2 : upstream_eq b s) : s = 3 :=
by
  sorry

end speed_of_stream_l208_208322


namespace total_gain_loss_is_correct_l208_208890

noncomputable def total_gain_loss_percentage 
    (cost1 cost2 cost3 : ℝ) 
    (gain1 gain2 gain3 : ℝ) : ℝ :=
  let total_cost := cost1 + cost2 + cost3
  let gain_amount1 := cost1 * gain1
  let loss_amount2 := cost2 * gain2
  let gain_amount3 := cost3 * gain3
  let net_gain_loss := (gain_amount1 + gain_amount3) - loss_amount2
  (net_gain_loss / total_cost) * 100

theorem total_gain_loss_is_correct :
  total_gain_loss_percentage 
    675958 995320 837492 0.11 (-0.11) 0.15 = 3.608 := 
sorry

end total_gain_loss_is_correct_l208_208890


namespace rest_days_in_1200_days_l208_208466

noncomputable def rest_days_coinciding (n : ℕ) : ℕ :=
  if h : n > 0 then (n / 6) else 0

theorem rest_days_in_1200_days :
  rest_days_coinciding 1200 = 200 :=
by
  sorry

end rest_days_in_1200_days_l208_208466


namespace inequality_abc_l208_208772

theorem inequality_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by {
  sorry
}

end inequality_abc_l208_208772


namespace min_value_expression_l208_208236

theorem min_value_expression (x : ℚ) : ∃ x : ℚ, (2 * x - 5)^2 + 18 = 18 :=
by {
  use 2.5,
  sorry
}

end min_value_expression_l208_208236


namespace cassandra_makes_four_pies_l208_208051

-- Define the number of dozens and respective apples per dozen
def dozens : ℕ := 4
def apples_per_dozen : ℕ := 12

-- Define the total number of apples
def total_apples : ℕ := dozens * apples_per_dozen

-- Define apples per slice and slices per pie
def apples_per_slice : ℕ := 2
def slices_per_pie : ℕ := 6

-- Calculate the number of slices and number of pies based on conditions
def total_slices : ℕ := total_apples / apples_per_slice
def total_pies : ℕ := total_slices / slices_per_pie

-- Prove that the number of pies is 4
theorem cassandra_makes_four_pies : total_pies = 4 := by
  sorry

end cassandra_makes_four_pies_l208_208051


namespace probability_each_university_at_least_one_admission_l208_208923

def total_students := 4
def total_universities := 3

theorem probability_each_university_at_least_one_admission :
  ∃ (p : ℚ), p = 4 / 9 :=
by
  sorry

end probability_each_university_at_least_one_admission_l208_208923


namespace martys_journey_length_l208_208977

theorem martys_journey_length (x : ℝ) (h1 : x / 4 + 30 + x / 3 = x) : x = 72 :=
sorry

end martys_journey_length_l208_208977


namespace students_not_solving_any_problem_l208_208024

variable (A_0 A_1 A_2 A_3 A_4 A_5 A_6 : ℕ)

-- Given conditions
def number_of_students := 2006
def condition_1 := A_1 = 4 * A_2
def condition_2 := A_2 = 4 * A_3
def condition_3 := A_3 = 4 * A_4
def condition_4 := A_4 = 4 * A_5
def condition_5 := A_5 = 4 * A_6
def total_students := A_0 + A_1 = 2006

-- The final statement to be proven
theorem students_not_solving_any_problem : 
  (A_1 = 4 * A_2) →
  (A_2 = 4 * A_3) →
  (A_3 = 4 * A_4) →
  (A_4 = 4 * A_5) →
  (A_5 = 4 * A_6) →
  (A_0 + A_1 = 2006) →
  (A_0 = 982) :=
by
  intro h1 h2 h3 h4 h5 h6
  -- Proof should go here
  sorry

end students_not_solving_any_problem_l208_208024


namespace min_value_of_function_l208_208294

theorem min_value_of_function : 
  ∃ x > 2, ∀ y > 2, (y + 1 / (y - 2)) ≥ 4 ∧ (x + 1 / (x - 2)) = 4 := 
by sorry

end min_value_of_function_l208_208294


namespace relationship_between_x_y_z_l208_208485

theorem relationship_between_x_y_z (x y z : ℕ) (a b c d : ℝ)
  (h1 : x ≤ y ∧ y ≤ z)
  (h2 : (x:ℝ)^a = 70^d ∧ (y:ℝ)^b = 70^d ∧ (z:ℝ)^c = 70^d)
  (h3 : 1/a + 1/b + 1/c = 1/d) :
  x + y = z := 
sorry

end relationship_between_x_y_z_l208_208485


namespace value_2_std_dev_less_than_mean_l208_208546

def mean : ℝ := 16.5
def std_dev : ℝ := 1.5

theorem value_2_std_dev_less_than_mean :
  mean - 2 * std_dev = 13.5 := by
  sorry

end value_2_std_dev_less_than_mean_l208_208546


namespace library_visit_period_l208_208969

noncomputable def dance_class_days := 6
noncomputable def karate_class_days := 12
noncomputable def common_days := 36

theorem library_visit_period (library_days : ℕ) 
  (hdance : ∀ (n : ℕ), n * dance_class_days = common_days)
  (hkarate : ∀ (n : ℕ), n * karate_class_days = common_days)
  (hcommon : ∀ (n : ℕ), n * library_days = common_days) : 
  library_days = 18 := 
sorry

end library_visit_period_l208_208969


namespace solve_for_x_l208_208708

theorem solve_for_x (x : ℕ) : (1 : ℚ) / 2 = x / 8 → x = 4 := by
  sorry

end solve_for_x_l208_208708


namespace part_I_part_II_l208_208795

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 6)

theorem part_I :
  ∃ A ω ϕ, A > 0 ∧ ω > 0 ∧ 0 ≤ ϕ ∧ ϕ < π ∧ (∀ x, f x = 2 * Real.sin (2 * x + π / 6)) ∧ 
           (∀ x ∈ set.Ici 0, f(x) <= 2) :=
by
  use 2, 2, π / 6
  sorry

theorem part_II {A B C b : ℝ} (hA: A > 0) (hf : f B = 1) (hb : b = 1) (triangle_ABC : ∀x ∈ set.Ici 0, 0 < A ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) :
  1 + Real.sqrt 3 ≤ (2 * Real.sin (A + π / 6) + 1) ∧ (2 * Real.sin (A + π / 6) + 1) ≤ 3 :=
by
  sorry

end part_I_part_II_l208_208795


namespace average_speed_rest_of_trip_l208_208031

variable (v : ℝ) -- The average speed for the rest of the trip
variable (d1 : ℝ := 30 * 5) -- Distance for the first part of the trip
variable (t1 : ℝ := 5) -- Time for the first part of the trip
variable (t_total : ℝ := 7.5) -- Total time for the trip
variable (avg_total : ℝ := 34) -- Average speed for the entire trip

def total_distance := avg_total * t_total
def d2 := total_distance - d1
def t2 := t_total - t1

theorem average_speed_rest_of_trip : 
  v = 42 :=
by
  let distance_rest := d2
  let time_rest := t2
  have v_def : v = distance_rest / time_rest := by sorry
  have v_value : v = 42 := by sorry
  exact v_value

end average_speed_rest_of_trip_l208_208031


namespace al_original_portion_l208_208900

theorem al_original_portion {a b c d : ℕ} 
  (h1 : a + b + c + d = 2000)
  (h2 : a - 150 + 3 * b + 3 * c + d - 50 = 2500)
  (h3 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  a = 450 :=
sorry

end al_original_portion_l208_208900


namespace blue_pill_cost_l208_208583

theorem blue_pill_cost
  (days : ℕ)
  (total_cost : ℤ)
  (cost_diff : ℤ)
  (daily_cost : ℤ)
  (y : ℤ) : 
  days = 21 →
  total_cost = 966 →
  cost_diff = 2 →
  daily_cost = total_cost / days →
  daily_cost = 46 →
  2 * y - cost_diff = daily_cost →
  y = 24 := 
by
  intros days_eq total_cost_eq cost_diff_eq daily_cost_eq d_cost_eq daily_eq_46;
  sorry

end blue_pill_cost_l208_208583


namespace average_marks_physics_mathematics_l208_208044

theorem average_marks_physics_mathematics {P C M : ℕ} (h1 : P + C + M = 180) (h2 : P = 140) (h3 : P + C = 140) : 
  (P + M) / 2 = 90 := by
  sorry

end average_marks_physics_mathematics_l208_208044


namespace overall_loss_is_correct_l208_208042

-- Define the conditions
def worth_of_stock : ℝ := 17500
def percent_stock_sold_at_profit : ℝ := 0.20
def profit_rate : ℝ := 0.10
def percent_stock_sold_at_loss : ℝ := 0.80
def loss_rate : ℝ := 0.05

-- Define the calculations based on the conditions
def worth_sold_at_profit : ℝ := percent_stock_sold_at_profit * worth_of_stock
def profit_amount : ℝ := profit_rate * worth_sold_at_profit

def worth_sold_at_loss : ℝ := percent_stock_sold_at_loss * worth_of_stock
def loss_amount : ℝ := loss_rate * worth_sold_at_loss

-- Define the overall loss amount
def overall_loss : ℝ := loss_amount - profit_amount

-- Theorem to prove that the calculated overall loss amount matches the expected loss amount
theorem overall_loss_is_correct :
  overall_loss = 350 :=
by
  sorry

end overall_loss_is_correct_l208_208042


namespace farey_sequence_problem_l208_208108

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l208_208108


namespace find_a_l208_208812

noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ

-- Define the given conditions
def b : ℝ := 4 * Real.sqrt 6
def B : ℝ := Real.pi / 3  -- 60 degrees in radians
def A : ℝ := Real.pi / 4  -- 45 degrees in radians

-- Define the law of sines equation
lemma law_of_sines (a : ℝ) : 
    a / sin A = b / sin B :=
begin
    sorry
end

-- Prove that a is equal to 8
theorem find_a (a : ℝ) (h : a / sin A = b / sin B) : a = 8 :=
begin
    sorry
end

end find_a_l208_208812


namespace classify_discuss_l208_208381

theorem classify_discuss (a b c : ℚ) (h : a * b * c > 0) : 
  (|a| / a + |b| / b + |c| / c = 3) ∨ (|a| / a + |b| / b + |c| / c = -1) :=
sorry

end classify_discuss_l208_208381


namespace direction_vector_l208_208891

open Matrix BigOperators

noncomputable def P : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    [1 / 10, 1 / 20, 1 / 5],
    [1 / 20, 1 / 5, 2 / 5],
    [1 / 5, 2 / 5, 4 / 5]
  ]

def v : Fin 3 → ℤ := ![2, 1, 10]

theorem direction_vector (a b c : ℤ) (g : Fin 3 → ℤ)
  (hₐ : a > 0) (h_gcd : Int.gcd (Int.gcd (Int.gcd a (Int.gcd b c)) 1) = 1)
  (h_Pv : ∀ (i : Fin 3), (P i).sum (λ j x, x * Int.cast (g j)) = Int.cast (g i)) : 
  g = v :=
sorry

end direction_vector_l208_208891


namespace simplify_fraction_l208_208285

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l208_208285


namespace find_annual_interest_rate_l208_208202

theorem find_annual_interest_rate (A P : ℝ) (n t : ℕ) (r : ℝ) :
  A = P * (1 + r / n)^(n * t) →
  A = 5292 →
  P = 4800 →
  n = 1 →
  t = 2 →
  r = 0.05 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end find_annual_interest_rate_l208_208202


namespace max_consecutive_sum_lt_1000_l208_208007

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l208_208007


namespace no_tangential_triangle_exists_l208_208780

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C2
def C2 (a b x y : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Additional condition that the point (1, 1) lies on C2
def point_on_C2 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  (1^2) / (a^2) + (1^2) / (b^2) = 1

-- The theorem to prove
theorem no_tangential_triangle_exists (a b : ℝ) (h : a > b ∧ b > 0) :
  point_on_C2 a b h →
  ¬ ∃ (A B C : ℝ × ℝ), 
    (C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2) ∧ 
    (C2 a b A.1 A.2 h ∧ C2 a b B.1 B.2 h ∧ C2 a b C.1 C.2 h) :=
by sorry

end no_tangential_triangle_exists_l208_208780


namespace find_theta_perpendicular_l208_208937

theorem find_theta_perpendicular (θ : ℝ) (hθ : 0 < θ ∧ θ < π)
  (a b : ℝ × ℝ) (ha : a = (Real.sin θ, 1)) (hb : b = (2 * Real.cos θ, -1))
  (hperp : a.fst * b.fst + a.snd * b.snd = 0) : θ = π / 4 :=
by
  -- Proof would be written here
  sorry

end find_theta_perpendicular_l208_208937


namespace trapezoid_CD_length_l208_208514

/-- In trapezoid ABCD with AD parallel to BC and diagonals intersecting:
  - BD = 2
  - ∠DBC = 36°
  - ∠BDA = 72°
  - The ratio BC : AD = 5 : 3

We are to show that the length of CD is 4/3. --/
theorem trapezoid_CD_length
  {A B C D : Type}
  (BD : ℝ) (DBC : ℝ) (BDA : ℝ) (BC_over_AD : ℝ)
  (AD_parallel_BC : Prop) (diagonals_intersect : Prop)
  (hBD : BD = 2) 
  (hDBC : DBC = 36) 
  (hBDA : BDA = 72)
  (hBC_over_AD : BC_over_AD = 5 / 3) 
  :  CD = 4 / 3 :=
by
  sorry

end trapezoid_CD_length_l208_208514


namespace wholesale_price_l208_208457

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l208_208457


namespace initial_scooter_value_l208_208685

theorem initial_scooter_value (V : ℝ) 
    (h : (9 / 16) * V = 22500) : V = 40000 :=
sorry

end initial_scooter_value_l208_208685


namespace polygon_sides_l208_208489

open Real

theorem polygon_sides (n : ℕ) : 
  (∀ (angle : ℝ), angle = 40 → n * angle = 360) → n = 9 := by
  intro h
  have h₁ := h 40 rfl
  sorry

end polygon_sides_l208_208489


namespace simplify_fraction_l208_208274

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208274


namespace cost_large_bulb_l208_208867

def small_bulbs : Nat := 3
def cost_small_bulb : Nat := 8
def total_budget : Nat := 60
def amount_left : Nat := 24

theorem cost_large_bulb (cost_large_bulb : Nat) :
  total_budget - amount_left - small_bulbs * cost_small_bulb = cost_large_bulb →
  cost_large_bulb = 12 := by
  sorry

end cost_large_bulb_l208_208867


namespace intersection_A_B_l208_208626

def A (x : ℝ) : Prop := x > 3
def B (x : ℝ) : Prop := x ≤ 4

theorem intersection_A_B : {x | A x} ∩ {x | B x} = {x | 3 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_l208_208626


namespace value_of_fraction_of_power_l208_208423

-- Define the values in the problem
def a : ℝ := 6
def b : ℝ := 30

-- The problem asks us to prove
theorem value_of_fraction_of_power : 
  (1 / 3) * (a ^ b) = 2 * (a ^ (b - 1)) :=
by
  -- Initial Setup
  let c := (1 / 3) * (a ^ b)
  let d := 2 * (a ^ (b - 1))
  -- The main claim
  show c = d
  sorry

end value_of_fraction_of_power_l208_208423


namespace number_of_students_l208_208089

theorem number_of_students (x : ℕ) (h : x * (x - 1) = 210) : x = 15 := 
by sorry

end number_of_students_l208_208089


namespace necessary_and_sufficient_condition_for_x2_ne_y2_l208_208419

theorem necessary_and_sufficient_condition_for_x2_ne_y2 (x y : ℤ) :
  (x ^ 2 ≠ y ^ 2) ↔ (x ≠ y ∧ x ≠ -y) :=
by
  sorry

end necessary_and_sufficient_condition_for_x2_ne_y2_l208_208419


namespace prime_base_values_l208_208468

theorem prime_base_values :
  ∀ p : ℕ, Prime p →
    (2 * p^3 + p^2 + 6 + 4 * p^2 + p + 4 + 2 * p^2 + p + 5 + 2 * p^2 + 2 * p + 2 + 9 =
     4 * p^2 + 3 * p + 3 + 5 * p^2 + 7 * p + 2 + 3 * p^2 + 2 * p + 1) →
    false :=
by {
  sorry
}

end prime_base_values_l208_208468


namespace sum_of_fifths_divisible_by_30_l208_208409

open BigOperators

theorem sum_of_fifths_divisible_by_30 {a : ℕ → ℕ} {n : ℕ} 
  (h : 30 ∣ ∑ i in Finset.range n, a i) : 
  30 ∣ ∑ i in Finset.range n, (a i) ^ 5 := 
by sorry

end sum_of_fifths_divisible_by_30_l208_208409


namespace triangle_side_length_l208_208624

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 20)
  (h2 : (1 / 2) * b * c * (Real.sin (Real.pi / 3)) = 10 * Real.sqrt 3) : a = 7 :=
sorry

end triangle_side_length_l208_208624


namespace number_of_zeros_of_f_in_interval_l208_208416

theorem number_of_zeros_of_f_in_interval :
  let f (x : ℝ) := 2^x + x^3 - 2
   in (∃! x ∈ Ioo 0 1, f x = 0) :=
by 
  sorry

end number_of_zeros_of_f_in_interval_l208_208416


namespace evaluate_expression_l208_208758

theorem evaluate_expression : 
  let a := 3 
  let b := 2 
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := 
by
  let a := 3
  let b := 2
  sorry

end evaluate_expression_l208_208758


namespace maximize_profit_l208_208447

noncomputable section

-- Definitions of parameters
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 200
def daily_cost : ℝ := 450
def price_min : ℝ := 30
def price_max : ℝ := 60

-- Function for daily profit
def daily_profit (x : ℝ) : ℝ := (x - 30) * daily_sales_volume x - daily_cost

-- Theorem statement
theorem maximize_profit :
  let max_profit_price := 60
  let max_profit_value := 1950
  30 ≤ max_profit_price ∧ max_profit_price ≤ 60 ∧
  daily_profit max_profit_price = max_profit_value :=
by
  sorry

end maximize_profit_l208_208447


namespace bracelets_count_l208_208591

-- Define the conditions
def stones_total : Nat := 36
def stones_per_bracelet : Nat := 12

-- Define the theorem statement
theorem bracelets_count : stones_total / stones_per_bracelet = 3 := by
  sorry

end bracelets_count_l208_208591


namespace solution_l208_208783

def system (a b : ℝ) : Prop :=
  (2 * a + b = 3) ∧ (a - b = 1)

theorem solution (a b : ℝ) (h: system a b) : a + 2 * b = 2 :=
by
  cases h with
  | intro h1 h2 => sorry

end solution_l208_208783


namespace peter_investment_time_l208_208119

variables (P A1 A2 : ℝ) (r t : ℝ)
hypothesis P_eq : P = 710
hypothesis A1_eq : A1 = 815
hypothesis A2_eq : A2 = 850
hypothesis simple_interest_david : A2 = P + (P * r * 4)
hypothesis simple_interest_peter : A1 = P + (P * r * t)

theorem peter_investment_time : t = 3 :=
begin
  have P_eq := P_eq,
  have A1_eq := A1_eq,
  have A2_eq := A2_eq,
  have r_eq : r = 0.05,
  {
    -- From A2_eq and simple_interest_david
    sorry,
  },
  have t_eq : t = 3,
  {
    -- From A1_eq and r_eq and simple_interest_peter
    sorry,
  },
  exact t_eq,
end

end peter_investment_time_l208_208119


namespace greatest_integer_b_l208_208608

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ 0) ↔ b ≤ 6 := 
by
  sorry

end greatest_integer_b_l208_208608


namespace exists_powers_mod_eq_l208_208670

theorem exists_powers_mod_eq (N : ℕ) (A : ℤ) : ∃ r s : ℕ, r ≠ s ∧ (A ^ r - A ^ s) % N = 0 :=
sorry

end exists_powers_mod_eq_l208_208670


namespace first_day_of_month_l208_208412

-- Define the days of the week as an enumeration
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Given condition
def eighteenth_day_is : Day := Wednesday

-- Theorem to prove
theorem first_day_of_month : 
  (eighteenth_day_is = Wednesday) → (eighteenth_day_is = Wednesday) := 
by
  intros h,
  -- We will need to establish the reverse order from 18th day is Wednesday back to 1st day is Sunday
  sorry

end first_day_of_month_l208_208412


namespace ac_le_bc_l208_208232

theorem ac_le_bc (a b c : ℝ) (h: a > b): ∃ c, ac * c ≤ bc * c := by
  sorry

end ac_le_bc_l208_208232


namespace area_and_cost_of_path_l208_208876

variables (length_field width_field path_width : ℝ) (cost_per_sq_m : ℝ)

noncomputable def area_of_path (length_field width_field path_width : ℝ) : ℝ :=
  let total_length := length_field + 2 * path_width
  let total_width := width_field + 2 * path_width
  let area_with_path := total_length * total_width
  let area_grass_field := length_field * width_field
  area_with_path - area_grass_field

noncomputable def cost_of_path (area_of_path cost_per_sq_m : ℝ) : ℝ :=
  area_of_path * cost_per_sq_m

theorem area_and_cost_of_path
  (length_field width_field path_width : ℝ)
  (cost_per_sq_m : ℝ)
  (h_length_field : length_field = 75)
  (h_width_field : width_field = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_sq_m : cost_per_sq_m = 10) :
  area_of_path length_field width_field path_width = 675 ∧
  cost_of_path (area_of_path length_field width_field path_width) cost_per_sq_m = 6750 :=
by
  rw [h_length_field, h_width_field, h_path_width, h_cost_per_sq_m]
  simp [area_of_path, cost_of_path]
  sorry

end area_and_cost_of_path_l208_208876


namespace product_of_two_numbers_l208_208683

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 120) :
  x * y = -20 :=
sorry

end product_of_two_numbers_l208_208683


namespace simple_interest_proof_l208_208427

def simple_interest (P R T: ℝ) : ℝ :=
  P * R * T

theorem simple_interest_proof :
  simple_interest 810 (4.783950617283951 / 100) 4 = 154.80 :=
by
  sorry

end simple_interest_proof_l208_208427


namespace table_tennis_team_l208_208093

theorem table_tennis_team : 
  ∃ n : ℕ, n = 48 ∧
  ∃ (Player : Type) (veteran new : Finset Player) (chosen ranked: Finset Player) (position_no : chosen → ℕ),
  veteran.card = 2 ∧ 
  new.card = 3 ∧ 
  (chosen ⊆ veteran ∪ new) ∧
  chosen.card = 3 ∧ 
  (∃ p ∈ chosen, p ∈ veteran) ∧ 
  ((∃ p1 p2 : chosen, position_no p1 = 1 ∧ position_no p2 = 2 ∧ (p1 ∈ new ∨ p2 ∈ new)) ∧ 
  position_no '' chosen = {1, 2, 3}) := 
begin
  sorry,
end

end table_tennis_team_l208_208093


namespace calories_per_person_l208_208650

theorem calories_per_person (oranges : ℕ) (pieces_per_orange : ℕ) (people : ℕ) (calories_per_orange : ℕ) :
  oranges = 5 →
  pieces_per_orange = 8 →
  people = 4 →
  calories_per_orange = 80 →
  (oranges * pieces_per_orange) / people * ((oranges * calories_per_orange) / (oranges * pieces_per_orange)) = 100 :=
by
  intros h_oranges h_pieces_per_orange h_people h_calories_per_orange
  sorry

end calories_per_person_l208_208650


namespace number_of_students_is_four_l208_208160

-- Definitions from the conditions
def average_weight_decrease := 8
def replaced_student_weight := 96
def new_student_weight := 64
def weight_decrease := replaced_student_weight - new_student_weight

-- Goal: Prove that the number of students is 4
theorem number_of_students_is_four
  (average_weight_decrease: ℕ)
  (replaced_student_weight new_student_weight: ℕ)
  (weight_decrease: ℕ) :
  weight_decrease / average_weight_decrease = 4 := 
by
  sorry

end number_of_students_is_four_l208_208160


namespace square_area_l208_208206

noncomputable def side_length_x (x : ℚ) : Prop :=
5 * x - 20 = 30 - 4 * x

noncomputable def side_length_s : ℚ :=
70 / 9

noncomputable def area_of_square : ℚ :=
(side_length_s)^2

theorem square_area (x : ℚ) (h : side_length_x x) : area_of_square = 4900 / 81 :=
sorry

end square_area_l208_208206


namespace Kato_finishes_first_l208_208666

-- Define constants and variables from the problem conditions
def Kato_total_pages : ℕ := 10
def Kato_lines_per_page : ℕ := 20
def Gizi_lines_per_page : ℕ := 30
def conversion_ratio : ℚ := 3 / 4
def initial_pages_written_by_Kato : ℕ := 4
def initial_additional_lines_by_Kato : ℚ := 2.5
def Kato_to_Gizi_writing_ratio : ℚ := 3 / 4

-- Calculate total lines in Kato's manuscript
def Kato_total_lines : ℕ := Kato_total_pages * Kato_lines_per_page

-- Convert Kato's lines to Gizi's format
def Kato_lines_in_Gizi_format : ℚ := Kato_total_lines * conversion_ratio

-- Calculate total pages Gizi needs to type
def Gizi_total_pages : ℚ := Kato_lines_in_Gizi_format / Gizi_lines_per_page

-- Calculate initial lines by Kato before Gizi starts typing
def initial_lines_by_Kato : ℚ := initial_pages_written_by_Kato * Kato_lines_per_page + initial_additional_lines_by_Kato

-- Lines Kato writes for every page Gizi types including setup time consideration
def additional_lines_by_Kato_per_Gizi_page : ℚ := Gizi_lines_per_page * Kato_to_Gizi_writing_ratio + initial_additional_lines_by_Kato / Gizi_total_pages

-- Calculate total lines Kato writes while Gizi finishes 5 pages
def final_lines_by_Kato : ℚ := additional_lines_by_Kato_per_Gizi_page * Gizi_total_pages

-- Remaining lines after initial setup for Kato
def remaining_lines_by_Kato_after_initial : ℚ := Kato_total_lines - initial_lines_by_Kato

-- Final proof statement
theorem Kato_finishes_first : final_lines_by_Kato ≥ remaining_lines_by_Kato_after_initial :=
by sorry

end Kato_finishes_first_l208_208666


namespace max_omega_l208_208796

theorem max_omega (ω : ℕ) (T : ℝ) (h₁ : T = 2 * Real.pi / ω) (h₂ : 1 < T) (h₃ : T < 3) : ω = 6 :=
sorry

end max_omega_l208_208796


namespace right_triangle_of_angle_condition_l208_208545

-- Defining the angles of the triangle
variables (α β γ : ℝ)

-- Defining the condition where the sum of angles in a triangle is 180 degrees
def sum_of_angles_in_triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Defining the given condition 
def angle_condition (γ α β : ℝ) : Prop :=
  γ = α + β

-- Stating the theorem to be proved
theorem right_triangle_of_angle_condition (α β γ : ℝ) :
  sum_of_angles_in_triangle α β γ → angle_condition γ α β → γ = 90 :=
by
  intro hsum hcondition
  sorry

end right_triangle_of_angle_condition_l208_208545


namespace team_total_points_l208_208693

-- Definition of Wade's average points per game
def wade_avg_points_per_game := 20

-- Definition of teammates' average points per game
def teammates_avg_points_per_game := 40

-- Definition of the number of games
def number_of_games := 5

-- The total points calculation problem
theorem team_total_points 
  (Wade_avg : wade_avg_points_per_game = 20)
  (Teammates_avg : teammates_avg_points_per_game = 40)
  (Games : number_of_games = 5) :
  5 * wade_avg_points_per_game + 5 * teammates_avg_points_per_game = 300 := 
by 
  -- The proof is omitted and marked as sorry
  sorry

end team_total_points_l208_208693


namespace p_iff_q_l208_208771

theorem p_iff_q (a b : ℝ) : (a > b) ↔ (a^3 > b^3) :=
sorry

end p_iff_q_l208_208771


namespace area_of_section_ABD_l208_208576
-- Import everything from the Mathlib library

-- Define the conditions
def is_equilateral_triangle (a b c : ℝ) (ABC_angle : ℝ) : Prop := 
  a = b ∧ b = c ∧ ABC_angle = 60

def plane_angle (angle : ℝ) : Prop := 
  angle = 35 + 18/60

def volume_of_truncated_pyramid (volume : ℝ) : Prop := 
  volume = 15

-- The main theorem based on the above conditions
theorem area_of_section_ABD
  (a b c ABC_angle : ℝ)
  (S : ℝ)
  (V : ℝ)
  (h1 : is_equilateral_triangle a b c ABC_angle)
  (h2 : plane_angle S)
  (h3 : volume_of_truncated_pyramid V) :
  ∃ (area : ℝ), area = 16.25 :=
by
  -- skipping the proof
  sorry

end area_of_section_ABD_l208_208576


namespace ten_percent_of_fifty_percent_of_five_hundred_l208_208313

theorem ten_percent_of_fifty_percent_of_five_hundred :
  0.10 * (0.50 * 500) = 25 :=
by
  sorry

end ten_percent_of_fifty_percent_of_five_hundred_l208_208313


namespace proof_problem_l208_208622

theorem proof_problem
  (a b : ℝ)
  (h1 : a = -(-3))
  (h2 : b = - (- (1 / 2))⁻¹)
  (m n : ℝ) :
  (|m - a| + |n + b| = 0) → (a = 3 ∧ b = -2 ∧ m = 3 ∧ n = -2) :=
by {
  sorry
}

end proof_problem_l208_208622


namespace wheel_revolutions_l208_208582

theorem wheel_revolutions (r_course r_wheel : ℝ) (laps : ℕ) (C_course C_wheel : ℝ) (d_total : ℝ) :
  r_course = 7 →
  r_wheel = 5 →
  laps = 15 →
  C_course = 2 * Real.pi * r_course →
  d_total = laps * C_course →
  C_wheel = 2 * Real.pi * r_wheel →
  ((d_total) / (C_wheel)) = 21 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end wheel_revolutions_l208_208582


namespace bottles_per_case_correct_l208_208883

-- Define the conditions given in the problem
def daily_bottle_production : ℕ := 120000
def number_of_cases_needed : ℕ := 10000

-- Define the expected answer
def bottles_per_case : ℕ := 12

-- The statement we need to prove
theorem bottles_per_case_correct :
  daily_bottle_production / number_of_cases_needed = bottles_per_case :=
by
  -- Leap of logic: actually solving this for correctness is here considered a leap
  sorry

end bottles_per_case_correct_l208_208883


namespace quadratic_roots_bc_minus_two_l208_208797

theorem quadratic_roots_bc_minus_two (b c : ℝ) 
  (h1 : 1 + -2 = -b) 
  (h2 : 1 * -2 = c) : b * c = -2 :=
by 
  sorry

end quadratic_roots_bc_minus_two_l208_208797


namespace matching_jelly_bean_probability_l208_208209

-- define Ava's jelly beans
def ava_jelly_beans : List (String × ℕ) := [
  ("green", 2),
  ("red", 2)
]

-- define Ben's jelly beans
def ben_jelly_beans : List (String × ℕ) := [
  ("green", 2),
  ("yellow", 3),
  ("red", 3)
]

-- calculate total jelly beans Ava has
def ava_total : ℕ := ava_jelly_beans.sum (fun (_, n) => n)

-- calculate total jelly beans Ben has
def ben_total : ℕ := ben_jelly_beans.sum (fun (_, n) => n)

-- calculate the probability of seeing "green" for Ava
def ava_green_probability : ℚ := (ava_jelly_beans.find_exc (fun (c, _) => c = "green")).2  / ava_total

-- calculate the probability of seeing "red" for Ava
def ava_red_probability : ℚ := (ava_jelly_beans.find_exc (fun (c, _) => c = "red")).2  / ava_total

-- calculate the probability of seeing "green" for Ben
def ben_green_probability : ℚ := (ben_jelly_beans.find_exc (fun (c, _) => c = "green")).2  / ben_total

-- calculate the probability of seeing "red" for Ben
def ben_red_probability : ℚ := (ben_jelly_beans.find_exc (fun (c, _) => c = "red")).2  / ben_total

-- define the event of color matching
def probability_of_matching_colors : ℚ := (ava_green_probability * ben_green_probability) + (ava_red_probability * ben_red_probability)

--state the theorem to be proved
theorem matching_jelly_bean_probability : 
  probability_of_matching_colors = 5 / 16 := 
by
  sorry -- Proof goes here

end matching_jelly_bean_probability_l208_208209


namespace min_value_3x_2y_l208_208930

theorem min_value_3x_2y (x y : ℝ) (h1: x > 0) (h2 : y > 0) (h3 : x = 4 * x * y - 2 * y) :
  3 * x + 2 * y >= 2 + Real.sqrt 3 :=
by
  sorry

end min_value_3x_2y_l208_208930


namespace abc_correct_and_c_not_true_l208_208478

theorem abc_correct_and_c_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a^2 > b^2 ∧ ab > b^2 ∧ (1/(a+b) > 1/a) ∧ ¬(1/a < 1/b) :=
  sorry

end abc_correct_and_c_not_true_l208_208478


namespace max_value_of_function_l208_208352

theorem max_value_of_function (x : ℝ) (h : x < 1 / 2) : 
  ∃ y, y = 2 * x + 1 / (2 * x - 1) ∧ y ≤ -1 :=
by
  sorry

end max_value_of_function_l208_208352


namespace range_of_a_l208_208619

-- Assuming all necessary imports and definitions are included

variable {R : Type} [LinearOrderedField R]

def satisfies_conditions (f : R → R) (a : R) : Prop :=
  (∀ x, f (1 + x) = f (1 - x)) ∧
  (∀ x y, 1 ≤ x → x < y → f x < f y) ∧
  (∀ x, (1/2 : R) ≤ x ∧ x ≤ 1 → f (a * x) < f (x - 1))

theorem range_of_a (f : R → R) (a : R) :
  satisfies_conditions f a → 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l208_208619


namespace constants_solution_l208_208224

theorem constants_solution : ∀ (x : ℝ), x ≠ 0 ∧ x^2 ≠ 2 →
  (2 * x^2 - 5 * x + 1) / (x^3 - 2 * x) = (-1 / 2) / x + (2.5 * x - 5) / (x^2 - 2) := by
  intros x hx
  sorry

end constants_solution_l208_208224


namespace gcd_lcm_product_l208_208062

theorem gcd_lcm_product (a b : ℕ) (ha : a = 225) (hb : b = 252) :
  Nat.gcd a b * Nat.lcm a b = 56700 := by
  sorry

end gcd_lcm_product_l208_208062


namespace tangent_through_points_l208_208259

theorem tangent_through_points :
  ∀ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ, y₁ = x₁^2 + 1 → y₂ = x₂^2 + 1 → 
    (2 * x₁ * (x₂ - x₁) + y₁ = 0 → x₂ = -x₁) ∧ 
    (2 * x₂ * (x₁ - x₂) + y₂ = 0 → x₁ = -x₂)) →
  (x₁ = 1 / Real.sqrt 3 ∧ x₂ = -1 / Real.sqrt 3 ∧
   (x₁^2 + 1 = (1 / 3) + 1) ∧ (x₂^2 + 1 = (1 / 3) + 1)) :=
by
  sorry

end tangent_through_points_l208_208259


namespace required_run_rate_l208_208568

theorem required_run_rate
  (run_rate_first_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_first : ℕ)
  (overs_remaining : ℕ)
  (H_run_rate_10_overs : run_rate_first_10_overs = 3.2)
  (H_target_runs : target_runs = 222)
  (H_overs_first : overs_first = 10)
  (H_overs_remaining : overs_remaining = 40) :
  ((target_runs - run_rate_first_10_overs * overs_first) / overs_remaining) = 4.75 := 
by
  sorry

end required_run_rate_l208_208568


namespace geometric_series_sum_eq_l208_208305

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end geometric_series_sum_eq_l208_208305


namespace new_girl_weight_l208_208729

theorem new_girl_weight (W : ℝ) (h : (W + 24) / 8 = W / 8 + 3) :
  (W + 24) - (W - 70) = 94 :=
by
  sorry

end new_girl_weight_l208_208729


namespace find_original_number_l208_208717

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l208_208717


namespace math_problem_l208_208790

theorem math_problem (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 :=
by
  sorry

end math_problem_l208_208790


namespace emily_fishes_correct_l208_208757

/-- Given conditions:
1. Emily caught 4 trout weighing 2 pounds each.
2. Emily caught 3 catfish weighing 1.5 pounds each.
3. Bluegills weigh 2.5 pounds each.
4. Emily caught a total of 25 pounds of fish. -/
def emilyCatches : Prop :=
  ∃ (trout_count catfish_count bluegill_count : ℕ)
    (trout_weight catfish_weight bluegill_weight total_weight : ℝ),
    trout_count = 4 ∧ catfish_count = 3 ∧ 
    trout_weight = 2 ∧ catfish_weight = 1.5 ∧ 
    bluegill_weight = 2.5 ∧ 
    total_weight = 25 ∧
    (total_weight = (trout_count * trout_weight) + (catfish_count * catfish_weight) + (bluegill_count * bluegill_weight)) ∧
    bluegill_count = 5

theorem emily_fishes_correct : emilyCatches := by
  sorry

end emily_fishes_correct_l208_208757


namespace compare_logs_l208_208479

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log 3

theorem compare_logs : b > c ∧ c > a :=
by
  sorry

end compare_logs_l208_208479


namespace five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l208_208866

theorem five_digit_numbers_greater_than_20314_and_formable_with_0_to_5 :
  (∃ (f : Fin 6 → Fin 5) (n : ℕ), 
    (n = 120 * 3 + 24 * 4 + 6 * 3 - 1) ∧
    (n = 473) ∧ 
    (∀ (x : Fin 6), f x = 0 ∨ f x = 1 ∨ f x = 2 ∨ f x = 3 ∨ f x = 4 ∨ f x = 5) ∧
    (∀ (i j : Fin 5), i ≠ j → f i ≠ f j)) :=
sorry

end five_digit_numbers_greater_than_20314_and_formable_with_0_to_5_l208_208866


namespace max_consecutive_sum_le_1000_l208_208004

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l208_208004


namespace no_real_x_for_sqrt_l208_208228

theorem no_real_x_for_sqrt :
  ¬ ∃ x : ℝ, - (x^2 + 2 * x + 5) ≥ 0 :=
sorry

end no_real_x_for_sqrt_l208_208228


namespace evaluate_expression_l208_208473

def x : ℝ := 2
def y : ℝ := 4

theorem evaluate_expression : y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l208_208473


namespace compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l208_208525

def a_n (p n : ℕ) : ℕ := (2 * n + 1) ^ p
def b_n (p n : ℕ) : ℕ := (2 * n) ^ p + (2 * n - 1) ^ p

theorem compare_magnitude_p2_for_n1 :
  b_n 2 1 < a_n 2 1 := sorry

theorem compare_magnitude_p2_for_n2 :
  b_n 2 2 = a_n 2 2 := sorry

theorem compare_magnitude_p2_for_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  b_n 2 n > a_n 2 n := sorry

theorem compare_magnitude_p_eq_n_for_all_n (n : ℕ) :
  a_n n n ≥ b_n n n := sorry

end compare_magnitude_p2_for_n1_compare_magnitude_p2_for_n2_compare_magnitude_p2_for_n_ge_3_compare_magnitude_p_eq_n_for_all_n_l208_208525


namespace probability_of_pulling_blue_ball_l208_208301

def given_conditions (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ) :=
  total_balls = 15 ∧ initial_blue_balls = 7 ∧ blue_balls_removed = 3

theorem probability_of_pulling_blue_ball
  (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ)
  (hc : given_conditions total_balls initial_blue_balls blue_balls_removed) :
  ((initial_blue_balls - blue_balls_removed) / (total_balls - blue_balls_removed) : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_pulling_blue_ball_l208_208301


namespace compute_k_l208_208974

noncomputable def tan_inverse (k : ℝ) : ℝ := Real.arctan k

theorem compute_k (x k : ℝ) (hx1 : Real.tan x = 2 / 3) (hx2 : Real.tan (3 * x) = 3 / 5) : k = 2 / 3 := sorry

end compute_k_l208_208974


namespace intersection_A1_B1_complement_A1_B1_union_A2_B2_l208_208571

-- Problem 1: Intersection and Complement
def setA1 : Set ℕ := {x : ℕ | x > 0 ∧ x < 9}
def setB1 : Set ℕ := {1, 2, 3}

theorem intersection_A1_B1 : (setA1 ∩ setB1) = {1, 2, 3} := by
  sorry

theorem complement_A1_B1 : {x : ℕ | x ∈ setA1 ∧ x ∉ setB1} = {4, 5, 6, 7, 8} := by
  sorry

-- Problem 2: Union
def setA2 : Set ℝ := {x : ℝ | -3 < x ∧ x < 1}
def setB2 : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem union_A2_B2 : (setA2 ∪ setB2) = {x : ℝ | (-3 < x ∧ x < 1) ∨ (2 < x ∧ x < 10)} := by
  sorry

end intersection_A1_B1_complement_A1_B1_union_A2_B2_l208_208571


namespace monotonicity_and_extrema_l208_208055

noncomputable def f (x : ℝ) := (2 * x) / (x + 1)

theorem monotonicity_and_extrema :
  (∀ x1 x2 : ℝ, 3 ≤ x1 → x1 < x2 → x2 ≤ 5 → f x1 < f x2) ∧
  (f 3 = 5 / 4) ∧
  (f 5 = 3 / 2) :=
by
  sorry

end monotonicity_and_extrema_l208_208055


namespace find_functional_equation_solutions_l208_208919

theorem find_functional_equation_solutions :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, x > 0 → y > 0 → f x * f (y * f x) = f (x + y)) →
    (∃ a > 0, ∀ x > 0, f x = 1 / (1 + a * x) ∨ ∀ x > 0, f x = 1)) :=
by
  sorry

end find_functional_equation_solutions_l208_208919


namespace find_number_l208_208028

theorem find_number (x : ℝ) (h : 0.80 * 40 = (4/5) * x + 16) : x = 20 :=
by sorry

end find_number_l208_208028


namespace total_weekly_allowance_l208_208190

theorem total_weekly_allowance
  (total_students : ℕ)
  (students_6dollar : ℕ)
  (students_4dollar : ℕ)
  (students_7dollar : ℕ)
  (allowance_6dollar : ℕ)
  (allowance_4dollar : ℕ)
  (allowance_7dollar : ℕ)
  (days_in_week : ℕ) :
  total_students = 100 →
  students_6dollar = 60 →
  students_4dollar = 25 →
  students_7dollar = 15 →
  allowance_6dollar = 6 →
  allowance_4dollar = 4 →
  allowance_7dollar = 7 →
  days_in_week = 7 →
  (students_6dollar * allowance_6dollar + students_4dollar * allowance_4dollar + students_7dollar * allowance_7dollar) * days_in_week = 3955 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_weekly_allowance_l208_208190


namespace solution_set_of_inequality_l208_208558

theorem solution_set_of_inequality :
  ∀ (x : ℝ), abs (2 * x + 1) < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_of_inequality_l208_208558


namespace total_jokes_l208_208815

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l208_208815


namespace y_relationship_l208_208484

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (hA : y1 = -7 * x1 + 14) 
  (hB : y2 = -7 * x2 + 14) 
  (hC : y3 = -7 * x3 + 14) 
  (hx : x1 > x3 ∧ x3 > x2) : y1 < y3 ∧ y3 < y2 :=
by
  sorry

end y_relationship_l208_208484


namespace sin_cos_sum_eq_l208_208075

theorem sin_cos_sum_eq (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2)
  (h : (Real.sin θ) - 2 * (Real.cos θ) = 0) :
  (Real.sin θ + Real.cos θ) = (3 * Real.sqrt 5) / 5 := by
  sorry

end sin_cos_sum_eq_l208_208075


namespace tank_volume_ratio_l208_208959

theorem tank_volume_ratio (A B : ℝ) 
    (h : (3 / 4) * A = (5 / 8) * B) : A / B = 6 / 5 := 
by 
  sorry

end tank_volume_ratio_l208_208959


namespace sale_in_first_month_l208_208319

theorem sale_in_first_month 
  (sale_2 : ℝ) (sale_3 : ℝ) (sale_4 : ℝ) (sale_5 : ℝ) (sale_6 : ℝ) (avg_sale : ℝ)
  (h_sale_2 : sale_2 = 5366) (h_sale_3 : sale_3 = 5808) 
  (h_sale_4 : sale_4 = 5399) (h_sale_5 : sale_5 = 6124) 
  (h_sale_6 : sale_6 = 4579) (h_avg_sale : avg_sale = 5400) :
  ∃ (sale_1 : ℝ), sale_1 = 5124 :=
by
  let total_sales := avg_sale * 6
  let known_sales := sale_2 + sale_3 + sale_4 + sale_5 + sale_6
  have h_total_sales : total_sales = 32400 := by sorry
  have h_known_sales : known_sales = 27276 := by sorry
  let sale_1 := total_sales - known_sales
  use sale_1
  have h_sale_1 : sale_1 = 5124 := by sorry
  exact h_sale_1

end sale_in_first_month_l208_208319


namespace batsman_average_increase_l208_208442

theorem batsman_average_increase
  (A : ℕ)  -- Assume the initial average is a non-negative integer
  (h1 : 11 * A + 70 = 12 * (A + 3))  -- Condition derived from the problem
  : A + 3 = 37 := 
by {
  -- The actual proof would go here, but is replaced by sorry to skip the proof
  sorry
}

end batsman_average_increase_l208_208442


namespace find_other_integer_l208_208339

theorem find_other_integer (x y : ℤ) (h1 : 4 * x + 3 * y = 140) (h2 : x = 20 ∨ y = 20) : x = 20 ∧ y = 20 :=
by
  sorry

end find_other_integer_l208_208339


namespace no_square_number_divisible_by_six_in_range_l208_208058

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (6 ∣ x) ∧ (50 < x) ∧ (x < 120) :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l208_208058


namespace M_is_set_of_positive_rationals_le_one_l208_208257

def M : Set ℚ := {x | 0 < x ∧ x ≤ 1}

axiom contains_one (M : Set ℚ) : 1 ∈ M

axiom closed_under_operations (M : Set ℚ) :
  ∀ x ∈ M, (1 / (1 + x) ∈ M) ∧ (x / (1 + x) ∈ M)

theorem M_is_set_of_positive_rationals_le_one :
  M = {x | 0 < x ∧ x ≤ 1} :=
sorry

end M_is_set_of_positive_rationals_le_one_l208_208257


namespace NumberOfRootsForEquation_l208_208813

noncomputable def numRootsAbsEq : ℕ :=
  let f := (fun x : ℝ => abs (abs (abs (abs (x - 1) - 9) - 9) - 3))
  let roots : List ℝ := [27, -25, 11, -9, 9, -7]
  roots.length

theorem NumberOfRootsForEquation : numRootsAbsEq = 6 := by
  sorry

end NumberOfRootsForEquation_l208_208813


namespace classical_prob_exp_is_exp1_l208_208585

-- Define the conditions under which an experiment is a classical probability model
def classical_probability_model (experiment : String) : Prop :=
  match experiment with
  | "exp1" => true  -- experiment ①: finite outcomes and equal likelihood
  | "exp2" => false -- experiment ②: infinite outcomes
  | "exp3" => false -- experiment ③: unequal likelihood
  | "exp4" => false -- experiment ④: infinite outcomes
  | _ => false

theorem classical_prob_exp_is_exp1 : classical_probability_model "exp1" = true ∧
                                      classical_probability_model "exp2" = false ∧
                                      classical_probability_model "exp3" = false ∧
                                      classical_probability_model "exp4" = false :=
by
  sorry

end classical_prob_exp_is_exp1_l208_208585


namespace square_perimeter_ratio_l208_208149

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l208_208149


namespace enthalpy_of_formation_C6H6_l208_208186

theorem enthalpy_of_formation_C6H6 :
  ∀ (enthalpy_C2H2 : ℝ) (enthalpy_C6H6 : ℝ)
  (enthalpy_C6H6_C6H6 : ℝ) (Hess_law : Prop),
  (enthalpy_C2H2 = 226.7) →
  (enthalpy_C6H6 = 631.1) →
  (enthalpy_C6H6_C6H6 = -33.9) →
  Hess_law →
  -- Using the given conditions to accumulate the enthalpy change for the formation of C6H6.
  ∃ Q_formation : ℝ, Q_formation = -82.9 := by
  sorry

end enthalpy_of_formation_C6H6_l208_208186


namespace cost_of_TOP_book_l208_208207

theorem cost_of_TOP_book (T : ℝ) (h1 : T = 8)
  (abc_cost : ℝ := 23)
  (top_books_sold : ℝ := 13)
  (abc_books_sold : ℝ := 4)
  (earnings_difference : ℝ := 12)
  (h2 : top_books_sold * T - abc_books_sold * abc_cost = earnings_difference) :
  T = 8 := 
by
  sorry

end cost_of_TOP_book_l208_208207


namespace quadratic_roots_in_range_l208_208927

theorem quadratic_roots_in_range (a : ℝ) (α β : ℝ)
  (h_eq : ∀ x : ℝ, x^2 + (a^2 + 1) * x + a - 2 = 0)
  (h_root1 : α > 1)
  (h_root2 : β < -1)
  (h_viete_sum : α + β = -(a^2 + 1))
  (h_viete_prod : α * β = a - 2) :
  0 < a ∧ a < 2 :=
  sorry

end quadratic_roots_in_range_l208_208927


namespace any_nat_as_fraction_form_l208_208871

theorem any_nat_as_fraction_form (n : ℕ) : ∃ (x y : ℕ), x = n^3 ∧ y = n^2 ∧ (x^3 / y^4 : ℝ) = n :=
by
  sorry

end any_nat_as_fraction_form_l208_208871


namespace ratio_of_perimeters_l208_208147

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l208_208147


namespace distance_from_edge_to_bottom_l208_208402

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end distance_from_edge_to_bottom_l208_208402


namespace number_of_groups_of_oranges_l208_208560

-- Defining the conditions
def total_oranges : ℕ := 356
def oranges_per_group : ℕ := 2

-- The proof statement
theorem number_of_groups_of_oranges : total_oranges / oranges_per_group = 178 := 
by 
  sorry

end number_of_groups_of_oranges_l208_208560


namespace angles_sum_132_l208_208253

theorem angles_sum_132
  (D E F p q : ℝ)
  (hD : D = 38)
  (hE : E = 58)
  (hF : F = 36)
  (five_sided_angle_sum : D + E + (360 - p) + 90 + (126 - q) = 540) : 
  p + q = 132 := 
by
  sorry

end angles_sum_132_l208_208253


namespace evaluate_expression_l208_208057

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l208_208057


namespace intersection_A_B_l208_208934

/-- Definition of set A -/
def A : Set ℕ := {1, 2, 3, 4}

/-- Definition of set B -/
def B : Set ℕ := {x | x > 2}

/-- The theorem to prove the intersection of sets A and B -/
theorem intersection_A_B : A ∩ B = {3, 4} :=
by
  sorry

end intersection_A_B_l208_208934


namespace simplify_expression_l208_208267

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l208_208267


namespace lucy_needs_more_distance_l208_208826

noncomputable def mary_distance : ℝ := (3 / 8) * 24
noncomputable def edna_distance : ℝ := (2 / 3) * mary_distance
noncomputable def lucy_distance : ℝ := (5 / 6) * edna_distance

theorem lucy_needs_more_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end lucy_needs_more_distance_l208_208826


namespace three_digit_integers_count_l208_208629

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_perfect_square_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 4

def is_allowed_digit (d : ℕ) : Prop :=
  is_prime_digit d ∨ is_perfect_square_digit d

theorem three_digit_integers_count : ∃ n : ℕ, n = 216 ∧ 
  (∀ x y z : ℕ, x ≠ 0 → is_allowed_digit x → is_allowed_digit y → is_allowed_digit z → 
   (x * 100 + y * 10 + z) > 99 ∧ (x * 100 + y * 10 + z) < 1000 → 
   (nat.card {n : ℕ | n = x * 100 + y * 10 + z ∧ is_allowed_digit x ∧ is_allowed_digit y ∧ is_allowed_digit z}) = 216) :=
begin
  sorry
end

end three_digit_integers_count_l208_208629


namespace tim_interest_rate_l208_208262

theorem tim_interest_rate
  (r : ℝ)
  (h1 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000))
  (h2 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000) + 23.5) : 
  r = 0.1 :=
by
  sorry

end tim_interest_rate_l208_208262


namespace average_age_after_swap_l208_208848

theorem average_age_after_swap :
  let initial_average_age := 28
  let num_people_initial := 8
  let person_leaving_age := 20
  let person_entering_age := 25
  let initial_total_age := initial_average_age * num_people_initial
  let total_age_after_leaving := initial_total_age - person_leaving_age
  let total_age_final := total_age_after_leaving + person_entering_age
  let num_people_final := 8
  initial_average_age / num_people_initial = 28 ->
  total_age_final / num_people_final = 28.625 :=
by
  intros
  sorry

end average_age_after_swap_l208_208848


namespace limit_of_sequence_z_l208_208539

open Nat Real

noncomputable def sequence_z (n : ℕ) : ℝ :=
  -3 + (-1)^n / (n^2 : ℝ)

theorem limit_of_sequence_z :
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, abs (sequence_z n + 3) < ε :=
by
  sorry

end limit_of_sequence_z_l208_208539


namespace total_cans_collected_l208_208536

theorem total_cans_collected :
  let cans_in_first_bag := 5
  let cans_in_second_bag := 7
  let cans_in_third_bag := 12
  let cans_in_fourth_bag := 4
  let cans_in_fifth_bag := 8
  let cans_in_sixth_bag := 10
  let cans_in_seventh_bag := 15
  let cans_in_eighth_bag := 6
  let cans_in_ninth_bag := 5
  let cans_in_tenth_bag := 13
  let total_cans := cans_in_first_bag + cans_in_second_bag + cans_in_third_bag + cans_in_fourth_bag + cans_in_fifth_bag + cans_in_sixth_bag + cans_in_seventh_bag + cans_in_eighth_bag + cans_in_ninth_bag + cans_in_tenth_bag
  total_cans = 85 :=
by
  sorry

end total_cans_collected_l208_208536


namespace min_b1_b2_l208_208217

-- Define the sequence recurrence relation
def sequence_recurrence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2011) / (1 + b (n + 1))

-- Problem statement: Prove the minimum value of b₁ + b₂ is 2012
theorem min_b1_b2 (b : ℕ → ℕ) (h : ∀ n ≥ 1, 0 < b n) (rec : sequence_recurrence b) :
  b 1 + b 2 ≥ 2012 :=
sorry

end min_b1_b2_l208_208217


namespace circle_transformation_l208_208849

theorem circle_transformation (c : ℝ × ℝ) (v : ℝ × ℝ) (h_center : c = (8, -3)) (h_vector : v = (2, -5)) :
  let reflected := (c.2, c.1)
  let translated := (reflected.1 + v.1, reflected.2 + v.2)
  translated = (-1, 3) :=
by
  sorry

end circle_transformation_l208_208849


namespace percent_females_employed_l208_208310

noncomputable def employed_percent (population: ℕ) : ℚ := 0.60
noncomputable def employed_males_percent (population: ℕ) : ℚ := 0.48

theorem percent_females_employed (population: ℕ) : ((employed_percent population) - (employed_males_percent population)) / (employed_percent population) = 0.20 :=
by
  sorry

end percent_females_employed_l208_208310


namespace determine_k_l208_208946

theorem determine_k (k : ℝ) :
  (∀ x : ℝ, ((k^2 - 9) * x^2 - (2 * (k + 1)) * x + 1 = 0 → x ∈ {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ∨ a = (2*(k+1) + √4 * (k+1)^2 - 4 * (k^2-9) ) / (2*(k^2-9)) ∨ a = (2 * (k+1) - √4 * (k+1)^2 - 4 *(k^2-9)) / (2 *(k^2-9))) → x ∈  {a : ℝ | a = - (2 * (k+1)) / (2 * (k^2-9)} ) ∨ x ∈  {a : ℝ | a = (2*(k+1) + 0 ) / (2 * (k^2-9))} ∨ a = (2*(k+1) - 0 ) / (2 *(k^2-9)} )  → 
  k = 3 ∨ k = -3 ∨ k = -5 := sorry

end determine_k_l208_208946


namespace trapezium_area_l208_208426

theorem trapezium_area (a b h : ℝ) (h_a : a = 4) (h_b : b = 5) (h_h : h = 6) :
  (1 / 2 * (a + b) * h) = 27 :=
by
  rw [h_a, h_b, h_h]
  norm_num

end trapezium_area_l208_208426


namespace percent_motorists_exceeding_speed_limit_l208_208664

-- Definitions based on conditions:
def total_motorists := 100
def percent_receiving_tickets := 10
def percent_exceeding_no_ticket := 50

-- The Lean 4 statement to prove the question
theorem percent_motorists_exceeding_speed_limit :
  (percent_receiving_tickets + (percent_receiving_tickets * percent_exceeding_no_ticket / 100)) = 20 :=
by
  sorry

end percent_motorists_exceeding_speed_limit_l208_208664


namespace find_length_QR_l208_208834

-- Conditions
variables {D E F Q R : Type} [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace Q] [MetricSpace R]
variables {DE EF DF QR : ℝ} (tangent : Q = E ∧ R = D)
variables (t₁ : de = 5) (t₂ : ef = 12) (t₃ : df = 13)

-- Problem: Prove that QR = 5 given the conditions.
theorem find_length_QR : QR = 5 :=
sorry

end find_length_QR_l208_208834


namespace part1_part2_l208_208302

open ProbabilityTheory MeasureTheory

noncomputable theory

namespace FireCompetition

variables {Ω : Type*} [ProbabilitySpace Ω]
variables {A1 A2 B1 B2 : Event Ω}
variables (PA1 PA2 PB1 PB2 : ℚ)

-- Given conditions
def P_A1 : Prop := PA1 = 4 / 5
def P_B1 : Prop := PB1 = 3 / 5
def P_A2 : Prop := PA2 = 2 / 3
def P_B2 : Prop := PB2 = 3 / 4

-- Independence condition
def independence : Prop := Independent (A1 :: B1 :: A2 :: B2 :: [])

-- Proof of Part 1: Probability of A winning exactly one round
theorem part1 (h1 : P_A1) (h2 : P_A2) (ind : independence) :
  (Prob (A1 \ (A2 ∩ ¬(A2)))).Val + (Prob ((¬A1) ∩ A2)).Val = 2 / 5 :=
sorry

-- Proof of Part 2: Probability that at least one of A or B wins the competition
theorem part2 (h1 : P_A1) (h2 : P_A2) (h3 : P_B1) (h4 : P_B2) (ind : independence) :
  (1 - Prob ((¬A1 ∩ ¬A2) ∩ (¬B1 ∩ ¬B2))).Val = 223 / 300 :=
sorry

end FireCompetition

end part1_part2_l208_208302


namespace expression_evaluation_l208_208336

theorem expression_evaluation :
  (1007 * (((7/4 : ℚ) / (3/4) + (3 / (9/4)) + (1/3)) /
    ((1 + 2 + 3 + 4 + 5) * 5 - 22)) / 19) = (4 : ℚ) :=
by
  sorry

end expression_evaluation_l208_208336


namespace square_perimeter_ratio_l208_208152

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l208_208152


namespace chord_existence_l208_208864

theorem chord_existence (O M : ℝ × ℝ) (R l : ℝ) :
  let OM := dist O M
  let d := sqrt (R^2 - (l/2)^2)
  l ≤ 2*R → OM ≤ R →
  (OM < d → ¬ ∃ A B : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2) ∧
  (OM = d → ∃! A B : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2) ∧
  (OM > d → ∃ A B C D : ℝ × ℝ, dist O A = R ∧ dist O B = R ∧ dist A B = l ∧ M = (A + B) / 2 ∧ dist O C = R ∧ dist O D = R ∧ dist C D = l ∧ M = (C + D) / 2 ∧ A ≠ C) :=
by {
  intros O M R l OM d hl hOM,
  split,
  { intro h1,
    sorry },
  split,
  { intro h2,
    sorry },
  { intro h3,
    sorry }
}

end chord_existence_l208_208864


namespace problem_statement_l208_208657

def T (m : ℕ) : ℕ := sorry
def H (m : ℕ) : ℕ := sorry

def p (m k : ℕ) : ℝ := 
  if k % 2 = 1 then 0 else sorry

theorem problem_statement (m : ℕ) : p m 0 ≥ p (m + 1) 0 := sorry

end problem_statement_l208_208657


namespace janice_weekly_earnings_l208_208966

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l208_208966


namespace solution_set_of_inequality_l208_208345

theorem solution_set_of_inequality :
  { x : ℝ | 2 * x^2 - x - 3 > 0 } = { x : ℝ | x > 3 / 2 ∨ x < -1 } :=
sorry

end solution_set_of_inequality_l208_208345


namespace perfect_cubes_count_l208_208940

theorem perfect_cubes_count : 
  Nat.card {n : ℕ | n^3 > 500 ∧ n^3 < 2000} = 5 :=
by
  sorry

end perfect_cubes_count_l208_208940


namespace max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l208_208921

open Real

theorem max_min_2sinx_minus_3 : 
  ∀ x : ℝ, 
    -5 ≤ 2 * sin x - 3 ∧ 
    2 * sin x - 3 ≤ -1 :=
by sorry

theorem max_min_7_fourth_sinx_minus_sinx_squared : 
  ∀ x : ℝ, 
    -1/4 ≤ (7/4 + sin x - sin x ^ 2) ∧ 
    (7/4 + sin x - sin x ^ 2) ≤ 2 :=
by sorry

end max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l208_208921


namespace train_time_to_B_l208_208990

theorem train_time_to_B (T : ℝ) (M : ℝ) :
  (∃ (D : ℝ), (T + 5) * (D + M) / T = 6 * M ∧ 2 * D = 5 * M) → T = 7 :=
by
  sorry

end train_time_to_B_l208_208990


namespace area_of_right_isosceles_triangle_l208_208060

def is_right_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

theorem area_of_right_isosceles_triangle (a b c : ℝ) (h : is_right_isosceles a b c) (h_hypotenuse : c = 10) :
  1/2 * a * b = 25 :=
by
  sorry

end area_of_right_isosceles_triangle_l208_208060


namespace intersection_A_B_l208_208531

open Set

variable (x : ℝ)

def setA : Set ℝ := {x | x^2 - 3 * x ≤ 0}
def setB : Set ℝ := {1, 2}

theorem intersection_A_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_A_B_l208_208531


namespace min_x2_y2_of_product_eq_zero_l208_208437

theorem min_x2_y2_of_product_eq_zero (x y : ℝ) (h : (x + 8) * (y - 8) = 0) : x^2 + y^2 = 64 :=
sorry

end min_x2_y2_of_product_eq_zero_l208_208437


namespace first_shaded_square_ensuring_all_columns_l208_208201

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def shaded_squares_in_columns (k : ℕ) : Prop :=
  ∀ j : ℕ, j < 7 → ∃ n : ℕ, triangular_number n % 7 = j ∧ triangular_number n ≤ k

theorem first_shaded_square_ensuring_all_columns:
  shaded_squares_in_columns 55 :=
by
  sorry

end first_shaded_square_ensuring_all_columns_l208_208201


namespace coefficient_of_q_l208_208808

theorem coefficient_of_q (q' : ℤ → ℤ) (h : ∀ q, q' q = 3 * q - 3) (h₁ : q' (q' 4) = 72) : 
  ∀ q, q' q = 3 * q - 3 :=
  sorry

end coefficient_of_q_l208_208808


namespace max_non_disjoint_3_element_subsets_l208_208522

open Finset

variable {α : Type} 

-- Definition of a set with n elements 
def X (n : ℕ) : Finset (Fin n) := univ

-- The statement of the problem
theorem max_non_disjoint_3_element_subsets (n : ℕ) (hn : n ≥ 6) :
  ∃ (F : Finset (Finset (Fin n))), 
    (∀ A ∈ F, A.card = 3) ∧ 
    (∀ A B ∈ F, A ≠ B → A ∩ B ≠ ∅) ∧ 
    F.card = choose (n - 1) 2 :=
sorry

end max_non_disjoint_3_element_subsets_l208_208522


namespace min_abs_sum_is_5_l208_208700

noncomputable def min_abs_sum (x : ℝ) : ℝ :=
  |x + 1| + |x + 3| + |x + 6|

theorem min_abs_sum_is_5 : ∃ x : ℝ, (∀ y : ℝ, min_abs_sum y ≥ min_abs_sum x) ∧ min_abs_sum x = 5 :=
by
  use -3
  sorry

end min_abs_sum_is_5_l208_208700


namespace geometric_sequence_common_ratio_l208_208794

theorem geometric_sequence_common_ratio (a q : ℝ) (h : a = a * q / (1 - q)) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l208_208794


namespace find_x_l208_208791

theorem find_x : ∃ (x : ℚ), (3 * x - 5) / 7 = 15 ∧ x = 110 / 3 := by
  sorry

end find_x_l208_208791


namespace max_quotient_l208_208357

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : (b / a) ≤ 15 :=
  sorry

end max_quotient_l208_208357


namespace tan_sum_simplification_l208_208542

theorem tan_sum_simplification :
  (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 4)) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  sorry

end tan_sum_simplification_l208_208542


namespace max_remaining_area_l208_208452

theorem max_remaining_area (original_area : ℕ) (rec1 : ℕ × ℕ) (rec2 : ℕ × ℕ) (rec3 : ℕ × ℕ)
  (rec4 : ℕ × ℕ) (total_area_cutout : ℕ):
  original_area = 132 →
  rec1 = (1, 4) →
  rec2 = (2, 2) →
  rec3 = (2, 3) →
  rec4 = (2, 3) →
  total_area_cutout = 20 →
  original_area - total_area_cutout = 112 :=
by
  intros
  sorry

end max_remaining_area_l208_208452


namespace find_asterisk_value_l208_208722

theorem find_asterisk_value :
  ∃ x : ℤ, (x / 21) * (42 / 84) = 1 ↔ x = 21 :=
by
  sorry

end find_asterisk_value_l208_208722


namespace crayons_problem_l208_208194

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l208_208194


namespace roses_cut_l208_208424

variable (initial final : ℕ) -- Declare variables for initial and final numbers of roses

-- Define the theorem stating the solution
theorem roses_cut (h1 : initial = 6) (h2 : final = 16) : final - initial = 10 :=
sorry -- Use sorry to skip the proof

end roses_cut_l208_208424


namespace product_of_solutions_l208_208941

theorem product_of_solutions (x : ℝ) (h : |(18 / x) - 6| = 3) : 2 * 6 = 12 :=
by
  sorry

end product_of_solutions_l208_208941


namespace square_perimeter_ratio_l208_208153

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l208_208153


namespace molecular_physics_statements_l208_208901

theorem molecular_physics_statements :
  (¬A) ∧ B ∧ C ∧ D :=
by sorry

end molecular_physics_statements_l208_208901


namespace box_office_collection_l208_208989

open Nat

/-- Define the total tickets sold -/
def total_tickets : ℕ := 1500

/-- Define the price of an adult ticket -/
def price_adult_ticket : ℕ := 12

/-- Define the price of a student ticket -/
def price_student_ticket : ℕ := 6

/-- Define the number of student tickets sold -/
def student_tickets : ℕ := 300

/-- Define the number of adult tickets sold -/
def adult_tickets : ℕ := total_tickets - student_tickets

/-- Define the revenue from adult tickets -/
def revenue_adult_tickets : ℕ := adult_tickets * price_adult_ticket

/-- Define the revenue from student tickets -/
def revenue_student_tickets : ℕ := student_tickets * price_student_ticket

/-- Define the total amount collected -/
def total_amount_collected : ℕ := revenue_adult_tickets + revenue_student_tickets

/-- Theorem to prove the total amount collected at the box office -/
theorem box_office_collection : total_amount_collected = 16200 := by
  sorry

end box_office_collection_l208_208989


namespace checkered_board_cut_l208_208612

def can_cut_equal_squares (n : ℕ) : Prop :=
  n % 5 = 0 ∧ n > 5

theorem checkered_board_cut (n : ℕ) (h : n % 5 = 0 ∧ n > 5) :
  ∃ m, n^2 = 5 * m :=
by
  sorry

end checkered_board_cut_l208_208612


namespace car_trip_time_difference_l208_208734

theorem car_trip_time_difference
  (average_speed : ℝ)
  (distance1 distance2 : ℝ)
  (speed_60_mph : average_speed = 60)
  (dist1_540 : distance1 = 540)
  (dist2_510 : distance2 = 510) :
  ((distance1 - distance2) / average_speed) * 60 = 30 := by
  sorry

end car_trip_time_difference_l208_208734


namespace blisters_on_rest_of_body_l208_208904

theorem blisters_on_rest_of_body (blisters_per_arm total_blisters : ℕ) (h1 : blisters_per_arm = 60) (h2 : total_blisters = 200) : 
  total_blisters - 2 * blisters_per_arm = 80 :=
by {
  -- The proof can be written here
  sorry
}

end blisters_on_rest_of_body_l208_208904


namespace interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l208_208461

noncomputable def principal_first_year : ℝ := 9000
noncomputable def interest_rate_first_year : ℝ := 0.09
noncomputable def principal_second_year : ℝ := principal_first_year * (1 + interest_rate_first_year)
noncomputable def interest_rate_second_year : ℝ := 0.105
noncomputable def principal_third_year : ℝ := principal_second_year * (1 + interest_rate_second_year)
noncomputable def interest_rate_third_year : ℝ := 0.085

noncomputable def compute_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

theorem interest_first_year_correct :
  compute_interest principal_first_year interest_rate_first_year = 810 := by
  sorry

theorem interest_second_year_correct :
  compute_interest principal_second_year interest_rate_second_year = 1034.55 := by
  sorry

theorem interest_third_year_correct :
  compute_interest principal_third_year interest_rate_third_year = 922.18 := by
  sorry

end interest_first_year_correct_interest_second_year_correct_interest_third_year_correct_l208_208461


namespace probability_both_girls_l208_208736

def club_probability (total_members girls chosen_members : ℕ) : ℚ :=
  (Nat.choose girls chosen_members : ℚ) / (Nat.choose total_members chosen_members : ℚ)

theorem probability_both_girls (H1 : total_members = 12) (H2 : girls = 7) (H3 : chosen_members = 2) :
  club_probability 12 7 2 = 7 / 22 :=
by {
  sorry
}

end probability_both_girls_l208_208736


namespace min_abs_sum_l208_208694

noncomputable def abs (x : ℝ) : ℝ := if x < 0 then -x else x

theorem min_abs_sum :
  ∃ (x : ℝ), (abs (x + 1) + abs (x + 3) + abs (x + 6)) = 7 :=
by {
  sorry
}

end min_abs_sum_l208_208694


namespace problem1_problem2_l208_208824

-- Problem 1
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  a * b + b * c + c * a ≤ 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b) :
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

end problem1_problem2_l208_208824


namespace AM_GM_inequality_AM_GM_equality_l208_208100

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) :=
by
  sorry

theorem AM_GM_equality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c :=
by
  sorry

end AM_GM_inequality_AM_GM_equality_l208_208100


namespace goldfish_graph_discrete_points_l208_208938

theorem goldfish_graph_discrete_points : 
  ∀ n : ℤ, 1 ≤ n ∧ n ≤ 10 → ∃ C : ℤ, C = 20 * n + 10 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 10 ∧ m ≠ n) → C ≠ (20 * m + 10) :=
by
  sorry

end goldfish_graph_discrete_points_l208_208938


namespace min_visible_pairs_l208_208618

-- Define the problem conditions
def bird_circle_flock (P : ℕ) : Prop :=
  P = 155

def mutual_visibility_condition (θ : ℝ) : Prop :=
  θ ≤ 10

-- Define the minimum number of mutually visible pairs
def min_mutual_visible_pairs (P_pairs : ℕ) : Prop :=
  P_pairs = 270

-- The main theorem statement
theorem min_visible_pairs (n : ℕ) (θ : ℝ) (P_pairs : ℕ)
  (H1 : bird_circle_flock n)
  (H2 : mutual_visibility_condition θ) :
  min_mutual_visible_pairs P_pairs :=
by
  sorry

end min_visible_pairs_l208_208618


namespace meteorological_forecasts_inaccuracy_l208_208660

theorem meteorological_forecasts_inaccuracy :
  let pA_accurate := 0.8
  let pB_accurate := 0.7
  let pA_inaccurate := 1 - pA_accurate
  let pB_inaccurate := 1 - pB_accurate
  pA_inaccurate * pB_inaccurate = 0.06 :=
by
  sorry

end meteorological_forecasts_inaccuracy_l208_208660


namespace tan_half_angle_second_quadrant_l208_208654

variables (θ : ℝ) (k : ℤ)
open Real

theorem tan_half_angle_second_quadrant (h : (π / 2) + 2 * k * π < θ ∧ θ < π + 2 * k * π) : 
  tan (θ / 2) > 1 := 
sorry

end tan_half_angle_second_quadrant_l208_208654


namespace work_completion_time_l208_208433

noncomputable def rate_b : ℝ := 1 / 24
noncomputable def rate_a : ℝ := 2 * rate_b
noncomputable def combined_rate : ℝ := rate_a + rate_b
noncomputable def completion_time : ℝ := 1 / combined_rate

theorem work_completion_time :
  completion_time = 8 :=
by
  sorry

end work_completion_time_l208_208433


namespace kosher_clients_count_l208_208117

def T := 30
def V := 7
def VK := 3
def Neither := 18

theorem kosher_clients_count (K : ℕ) : T - Neither = V + K - VK → K = 8 :=
by
  intro h
  sorry

end kosher_clients_count_l208_208117


namespace xyz_sum_eq_eleven_l208_208775

theorem xyz_sum_eq_eleven (x y z : ℝ) (h : (x - 2)^2 + (y - 3)^2 + (z - 6)^2 = 0) : x + y + z = 11 :=
sorry

end xyz_sum_eq_eleven_l208_208775


namespace determine_original_volume_of_tank_l208_208895

noncomputable def salt_volume (x : ℝ) := 0.20 * x
noncomputable def new_volume_after_evaporation (x : ℝ) := (3 / 4) * x
noncomputable def new_volume_after_additions (x : ℝ) := (3 / 4) * x + 6 + 12
noncomputable def new_salt_after_addition (x : ℝ) := 0.20 * x + 12
noncomputable def resulting_salt_concentration (x : ℝ) := (0.20 * x + 12) / ((3 / 4) * x + 18)

theorem determine_original_volume_of_tank (x : ℝ) :
  resulting_salt_concentration x = 1 / 3 → x = 120 := 
by 
  sorry

end determine_original_volume_of_tank_l208_208895


namespace arithmetic_sequence_binomial_expansion_l208_208675

open Nat

theorem arithmetic_sequence_binomial_expansion :
  ∃ (n r : ℕ), 
    n = 8 ∧ 
    (∀ x : ℝ, 
      let a := (binom n 0 : ℝ),
      let b := (binom n 1) * (1 / 2 : ℝ),
      let c := (binom n 2) * (1 / 2)^2,
      b - a = c - b ∧
        ((a = 1) ∧ 
         (b = (n : ℝ) / 2) ∧ 
         (c = (n : ℝ) * (n - 1) / 8) ∧
         r = 2 ∧
         (x ^ (n - 4 * r) / 6 = 1) ∧
         (binom n r) * (1 / 2)^r = 14)) :=
sorry

end arithmetic_sequence_binomial_expansion_l208_208675


namespace water_tank_full_capacity_l208_208318

-- Define the conditions
variable {C x : ℝ}
variable (h1 : x / C = 1 / 3)
variable (h2 : (x + 6) / C = 1 / 2)

-- Prove that C = 36
theorem water_tank_full_capacity : C = 36 :=
by
  sorry

end water_tank_full_capacity_l208_208318


namespace math_problem_l208_208646

noncomputable def triangle_conditions (a b c A B C : ℝ) := 
  (2 * b - c) / a = (Real.cos C) / (Real.cos A) ∧ 
  a = Real.sqrt 5 ∧
  1 / 2 * b * c * (Real.sin A) = Real.sqrt 3 / 2

theorem math_problem (a b c A B C : ℝ) (h : triangle_conditions a b c A B C) :
  A = π / 3 ∧ a + b + c = Real.sqrt 5 + Real.sqrt 11 :=
by
  sorry

end math_problem_l208_208646


namespace min_x_plus_y_l208_208086

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y = 9 :=
sorry

end min_x_plus_y_l208_208086


namespace nonagon_distinct_diagonals_l208_208496

theorem nonagon_distinct_diagonals : 
  let n := 9 in
  ∃ (d : ℕ), d = (n * (n - 3)) / 2 ∧ d = 27 :=
by
  sorry

end nonagon_distinct_diagonals_l208_208496


namespace find_a_ge_3_l208_208229

open Real Function

noncomputable def trigonometric_inequality (a : ℝ) :=
  ∀ θ ∈ Ico 0 (π / 2),
    sqrt 2 * (2 * a + 3) * cos (θ - π / 4) / (6 / (sin θ + cos θ)) <
    2 * sin (2 * θ) - 3 * a + 6

theorem find_a_ge_3 : 
  ∀ (a : ℝ), trigonometric_inequality a → a > 3 :=
sorry

end find_a_ge_3_l208_208229


namespace bacteria_growth_rate_l208_208192

theorem bacteria_growth_rate (P : ℝ) (r : ℝ) : 
  (P * r ^ 25 = 2 * (P * r ^ 24) ) → r = 2 :=
by sorry

end bacteria_growth_rate_l208_208192


namespace not_perfect_square_l208_208749

theorem not_perfect_square (n : ℕ) (h₁ : 100 + 200 = 300) (h₂ : ¬(300 % 9 = 0)) : ¬(∃ m : ℕ, n = m * m) :=
by
  intros
  sorry

end not_perfect_square_l208_208749


namespace min_abs_sum_is_two_l208_208696

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l208_208696


namespace number_is_43_l208_208164

theorem number_is_43 (m : ℕ) : (m > 30 ∧ m < 50) ∧ Nat.Prime m ∧ m % 12 = 7 ↔ m = 43 :=
by
  sorry

end number_is_43_l208_208164


namespace boys_girls_dance_l208_208588

theorem boys_girls_dance (b g : ℕ) 
  (h : ∀ n, (n <= b) → (n + 7) ≤ g) 
  (hb_lasts : b + 7 = g) :
  b = g - 7 := by
  sorry

end boys_girls_dance_l208_208588


namespace find_x_l208_208307

noncomputable def value_of_x (x : ℝ) := (5 * x) ^ 4 = (15 * x) ^ 3

theorem find_x : ∀ (x : ℝ), (value_of_x x) ∧ (x ≠ 0) → x = 27 / 5 :=
by
  intro x
  intro h
  sorry

end find_x_l208_208307


namespace Janice_earnings_l208_208963

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l208_208963


namespace solve_abs_inequality_l208_208130

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l208_208130


namespace farey_sequence_problem_l208_208107

theorem farey_sequence_problem (p q : ℕ) (hpq_pos : 0 < p ∧ 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 5 / 8) 
  (h_minimal_q : ∀ p' q', 0 < p' ∧ 0 < q' → 3 / 5 < p' / q' → p' / q' < 5 / 8 → q' ≥ q) : 
  q - p = 5 := 
sorry

end farey_sequence_problem_l208_208107


namespace sum_of_legs_of_similar_larger_triangle_l208_208692

-- Define the conditions for the problem
def smaller_triangle_area : ℝ := 10
def larger_triangle_area : ℝ := 400
def smaller_triangle_hypotenuse : ℝ := 10

-- Define the correct answer (sum of the lengths of the legs of the larger triangle)
def sum_of_legs_of_larger_triangle : ℝ := 88.55

-- State the Lean theorem
theorem sum_of_legs_of_similar_larger_triangle :
  (∀ (A B C a b c : ℝ), 
    a * b / 2 = smaller_triangle_area ∧ 
    c = smaller_triangle_hypotenuse ∧
    C * C / 4 = larger_triangle_area / smaller_triangle_area ∧
    A / a = B / b ∧ 
    A^2 + B^2 = C^2 → 
    A + B = sum_of_legs_of_larger_triangle) :=
  by sorry

end sum_of_legs_of_similar_larger_triangle_l208_208692


namespace max_consecutive_integers_sum_lt_1000_l208_208010

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l208_208010


namespace preservation_interval_l208_208635

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * (x - 1)^2 + 1

theorem preservation_interval {a b : ℝ} (h_dom_range : (∀ y, y ∈ set.Icc a b → (∃ x, x ∈ set.Icc a b ∧ f x = y))) :
  set.Icc a b = set.Icc 1 3 :=
begin
  -- We assume the domain and range conditions
  -- We need to prove that the preservation interval is [1, 3]
  sorry
end

end preservation_interval_l208_208635


namespace no_integer_soln_x_y_l208_208671

theorem no_integer_soln_x_y (x y : ℤ) : x^2 + 5 ≠ y^3 := 
sorry

end no_integer_soln_x_y_l208_208671


namespace find_a_l208_208778

def tangent_condition (x a : ℝ) : Prop := 2 * x - (Real.log x + a) + 1 = 0

def slope_condition (x : ℝ) : Prop := 2 = 1 / x

theorem find_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ tangent_condition x a ∧ slope_condition x) →
  a = -2 * Real.log 2 :=
by
  intro h
  sorry

end find_a_l208_208778


namespace solve_inequality_l208_208673

open Real

noncomputable def expression (x : ℝ) : ℝ :=
  (sqrt (x^2 - 4*x + 3) + 1) * log x / (log 2 * 5) + (1 / x) * (sqrt (8 * x - 2 * x^2 - 6) + 1)

theorem solve_inequality :
  ∃ x : ℝ, x = 1 ∧
    (x > 0) ∧
    (x^2 - 4 * x + 3 ≥ 0) ∧
    (8 * x - 2 * x^2 - 6 ≥ 0) ∧
    expression x ≤ 0 :=
by
  sorry

end solve_inequality_l208_208673


namespace num_trombone_players_l208_208806

def weight_per_trumpet := 5
def weight_per_clarinet := 5
def weight_per_trombone := 10
def weight_per_tuba := 20
def weight_per_drum := 15

def num_trumpets := 6
def num_clarinets := 9
def num_tubas := 3
def num_drummers := 2
def total_weight := 245

theorem num_trombone_players : 
  let weight_trumpets := num_trumpets * weight_per_trumpet
  let weight_clarinets := num_clarinets * weight_per_clarinet
  let weight_tubas := num_tubas * weight_per_tuba
  let weight_drums := num_drummers * weight_per_drum
  let weight_others := weight_trumpets + weight_clarinets + weight_tubas + weight_drums
  let weight_trombones := total_weight - weight_others
  weight_trombones / weight_per_trombone = 8 :=
by
  sorry

end num_trombone_players_l208_208806


namespace determine_angles_l208_208218

theorem determine_angles 
  (small_angle1 : ℝ) 
  (small_angle2 : ℝ) 
  (large_angle1 : ℝ) 
  (large_angle2 : ℝ) 
  (triangle_sum_property : ∀ a b c : ℝ, a + b + c = 180) 
  (exterior_angle_property : ∀ a c : ℝ, a + c = 180) :
  (small_angle1 = 70) → 
  (small_angle2 = 180 - 130) → 
  (large_angle1 = 45) → 
  (large_angle2 = 50) → 
  ∃ α β : ℝ, α = 120 ∧ β = 85 :=
by
  intros h1 h2 h3 h4
  sorry

end determine_angles_l208_208218


namespace no_solution_of_abs_sum_l208_208312

theorem no_solution_of_abs_sum (a : ℝ) : (∀ x : ℝ, |x - 2| + |x + 3| < a → false) ↔ a ≤ 5 := sorry

end no_solution_of_abs_sum_l208_208312


namespace perimeters_ratio_l208_208156

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l208_208156


namespace total_amount_of_check_l208_208316

def numParts : Nat := 59
def price50DollarPart : Nat := 50
def price20DollarPart : Nat := 20
def num50DollarParts : Nat := 40

theorem total_amount_of_check : (num50DollarParts * price50DollarPart + (numParts - num50DollarParts) * price20DollarPart) = 2380 := by
  sorry

end total_amount_of_check_l208_208316


namespace arithmetic_evaluation_l208_208464

theorem arithmetic_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := 
by
  sorry

end arithmetic_evaluation_l208_208464


namespace intersection_singleton_one_l208_208929

-- Define sets A and B according to the given conditions
def setA : Set ℤ := { x | 0 < x ∧ x < 4 }
def setB : Set ℤ := { x | (x+1)*(x-2) < 0 }

-- Statement to prove A ∩ B = {1}
theorem intersection_singleton_one : setA ∩ setB = {1} :=
by 
  sorry

end intersection_singleton_one_l208_208929


namespace marys_garbage_bill_is_correct_l208_208978

noncomputable def calculate_garbage_bill :=
  let weekly_trash_bin_cost := 2 * 10
  let weekly_recycling_bin_cost := 1 * 5
  let weekly_green_waste_bin_cost := 1 * 3
  let total_weekly_cost := weekly_trash_bin_cost + weekly_recycling_bin_cost + weekly_green_waste_bin_cost
  let monthly_bin_cost := total_weekly_cost * 4
  let base_monthly_cost := monthly_bin_cost + 15
  let discount := base_monthly_cost * 0.18
  let discounted_cost := base_monthly_cost - discount
  let fines := 20 + 10
  discounted_cost + fines

theorem marys_garbage_bill_is_correct :
  calculate_garbage_bill = 134.14 := 
  by {
  sorry
  }

end marys_garbage_bill_is_correct_l208_208978


namespace min_abs_sum_l208_208703

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l208_208703


namespace cannot_contain_point_1997_0_l208_208084

variable {m b : ℝ}

theorem cannot_contain_point_1997_0 (h : m * b > 0) : ¬ (0 = 1997 * m + b) := sorry

end cannot_contain_point_1997_0_l208_208084


namespace domain_of_log_sqrt_l208_208850

noncomputable def domain_of_function := {x : ℝ | (2 * x - 1 > 0) ∧ (2 * x - 1 ≠ 1) ∧ (3 * x - 2 > 0)}

theorem domain_of_log_sqrt : domain_of_function = {x : ℝ | (2 / 3 < x ∧ x < 1) ∨ (1 < x)} :=
by sorry

end domain_of_log_sqrt_l208_208850


namespace solve_for_r_l208_208847

variable (n : ℝ) (r : ℝ)

theorem solve_for_r (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : r = (n * (1 + Real.sqrt 3)) / 2 :=
by
  sorry

end solve_for_r_l208_208847


namespace probability_of_five_dice_all_same_l208_208726

theorem probability_of_five_dice_all_same : 
  (6 / (6 ^ 5) = 1 / 1296) :=
by
  sorry

end probability_of_five_dice_all_same_l208_208726


namespace butterfat_mixture_l208_208727

/-
  Given:
  - 8 gallons of milk with 40% butterfat
  - x gallons of milk with 10% butterfat
  - Resulting mixture with 20% butterfat

  Prove:
  - x = 16 gallons
-/

theorem butterfat_mixture (x : ℝ) : 
  (0.40 * 8 + 0.10 * x) / (8 + x) = 0.20 → x = 16 := 
by
  sorry

end butterfat_mixture_l208_208727


namespace find_a_10_l208_208645

/-- 
a_n is an arithmetic sequence
-/
def a (n : ℕ) : ℝ := sorry

/-- 
Given conditions:
- Condition 1: a_2 + a_5 = 19
- Condition 2: S_5 = 40, where S_5 is the sum of the first five terms
-/
axiom condition1 : a 2 + a 5 = 19
axiom condition2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 40

noncomputable def a_10 : ℝ := a 10

theorem find_a_10 : a_10 = 29 :=
by
  sorry

end find_a_10_l208_208645


namespace solve_system_of_inequalities_l208_208134

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l208_208134


namespace twenty_percent_greater_l208_208183

theorem twenty_percent_greater (x : ℕ) : 
  x = 80 + (20 * 80 / 100) → x = 96 :=
by
  sorry

end twenty_percent_greater_l208_208183


namespace cups_remaining_l208_208385

-- Each definition only directly appears in the conditions problem
def required_cups : ℕ := 7
def added_cups : ℕ := 3

-- The proof problem capturing Joan needs to add 4 more cups of flour.
theorem cups_remaining : required_cups - added_cups = 4 := 
by
  -- The proof is skipped using sorry.
  sorry

end cups_remaining_l208_208385


namespace age_of_b_l208_208724

variable (a b c d : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : d = b / 2)
variable (h4 : a + b + c + d = 44)

theorem age_of_b : b = 14 :=
by 
  sorry

end age_of_b_l208_208724


namespace simplify_expression_l208_208269

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l208_208269


namespace distance_between_planes_l208_208607

def plane1 (x y z : ℝ) := 3 * x - y + z - 3 = 0
def plane2 (x y z : ℝ) := 6 * x - 2 * y + 2 * z + 4 = 0

theorem distance_between_planes :
  ∃ d : ℝ, d = (5 * Real.sqrt 11) / 11 ∧ 
            ∀ x y z : ℝ, plane1 x y z → plane2 x y z → d = (5 * Real.sqrt 11) / 11 :=
sorry

end distance_between_planes_l208_208607


namespace area_of_triangle_formed_by_intercepts_l208_208471

theorem area_of_triangle_formed_by_intercepts :
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := [-3, 4]
  let y_intercept := 48
  let base := 7
  let height := 48
  let area := (1 / 2 : ℝ) * base * height
  area = 168 :=
by
  sorry

end area_of_triangle_formed_by_intercepts_l208_208471


namespace find_k_l208_208296

-- The function that computes the sum of the digits for the known form of the product (9 * 999...9) with k digits.
def sum_of_digits (k : ℕ) : ℕ :=
  8 + 9 * (k - 1) + 1

theorem find_k (k : ℕ) : sum_of_digits k = 2000 ↔ k = 222 := by
  sorry

end find_k_l208_208296


namespace smallest_q_p_difference_l208_208109

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l208_208109


namespace bridge_length_proof_l208_208742

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 49.9960003199744
noncomputable def train_speed_kmph : ℝ := 18
noncomputable def conversion_factor : ℝ := 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * conversion_factor
noncomputable def total_distance : ℝ := train_speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem bridge_length_proof : bridge_length = 149.980001599872 := 
by 
  sorry

end bridge_length_proof_l208_208742


namespace family_gathering_l208_208092

theorem family_gathering : 
  ∃ (total_people oranges bananas apples : ℕ), 
    total_people = 20 ∧ 
    oranges = total_people / 2 ∧ 
    bananas = (total_people - oranges) / 2 ∧ 
    apples = total_people - oranges - bananas ∧ 
    oranges < total_people ∧ 
    total_people - oranges = 10 :=
by
  sorry

end family_gathering_l208_208092


namespace journey_total_time_l208_208630

theorem journey_total_time (speed1 time1 speed2 total_distance : ℕ) 
  (h1 : speed1 = 40) 
  (h2 : time1 = 3) 
  (h3 : speed2 = 60) 
  (h4 : total_distance = 240) : 
  time1 + (total_distance - speed1 * time1) / speed2 = 5 := 
by 
  sorry

end journey_total_time_l208_208630


namespace max_value_AMC_l208_208820

theorem max_value_AMC (A M C : ℕ) (h : A + M + C = 15) : 
  2 * (A * M * C) + A * M + M * C + C * A ≤ 325 := 
sorry

end max_value_AMC_l208_208820


namespace susie_pizza_sales_l208_208410

theorem susie_pizza_sales :
  ∃ x : ℕ, 
    (24 * 3 + 15 * x = 117) ∧ 
    x = 3 := 
by
  sorry

end susie_pizza_sales_l208_208410


namespace given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l208_208398

theorem given_conditions_implies_a1d1_a2d2_a3d3_eq_zero
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, 
    x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
    (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - x + 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 0 :=
by
  sorry

end given_conditions_implies_a1d1_a2d2_a3d3_eq_zero_l208_208398


namespace value_of_fraction_l208_208290

theorem value_of_fraction (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (b / c = 2005) ∧ (c / b = 2005)) : (b + c) / (a + b) = 2005 :=
by
  sorry

end value_of_fraction_l208_208290


namespace geometric_sum_n_is_4_l208_208169

theorem geometric_sum_n_is_4 
  (a r : ℚ) (n : ℕ) (S_n : ℚ) 
  (h1 : a = 1) 
  (h2 : r = 1 / 4) 
  (h3 : S_n = (a * (1 - r^n)) / (1 - r)) 
  (h4 : S_n = 85 / 64) : 
  n = 4 := 
sorry

end geometric_sum_n_is_4_l208_208169


namespace g_neither_even_nor_odd_l208_208472

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1

theorem g_neither_even_nor_odd : ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = -g (-x)) := 
by sorry

end g_neither_even_nor_odd_l208_208472


namespace sandy_will_be_32_l208_208687

  -- Define the conditions
  def world_record_length : ℝ := 26
  def sandy_current_length : ℝ := 2
  def monthly_growth_rate : ℝ := 0.1
  def sandy_current_age : ℝ := 12

  -- Define the annual growth rate calculation
  def annual_growth_rate : ℝ := monthly_growth_rate * 12

  -- Define total growth needed
  def total_growth_needed : ℝ := world_record_length - sandy_current_length

  -- Define the years needed to grow the fingernails to match the world record
  def years_needed : ℝ := total_growth_needed / annual_growth_rate

  -- Define Sandy's age when she achieves the world record length
  def sandy_age_when_record_achieved : ℝ := sandy_current_age + years_needed

  -- The statement we want to prove
  theorem sandy_will_be_32 :
    sandy_age_when_record_achieved = 32 :=
  by
    -- Placeholder proof
    sorry
  
end sandy_will_be_32_l208_208687


namespace Janice_earnings_l208_208964

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end Janice_earnings_l208_208964


namespace fabian_total_cost_l208_208342

def mouse_cost : ℕ := 20

def keyboard_cost : ℕ := 2 * mouse_cost

def headphones_cost : ℕ := mouse_cost + 15

def usb_hub_cost : ℕ := 36 - mouse_cost

def total_cost : ℕ := mouse_cost + keyboard_cost + headphones_cost + usb_hub_cost

theorem fabian_total_cost : total_cost = 111 := 
by 
  unfold total_cost mouse_cost keyboard_cost headphones_cost usb_hub_cost
  sorry

end fabian_total_cost_l208_208342


namespace inequality_among_positives_l208_208115

theorem inequality_among_positives
  (a b x y z : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) :
  (x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y)) ≥ (3 / (a + b)) :=
by
  sorry

end inequality_among_positives_l208_208115


namespace AM_GM_Inequality_equality_condition_l208_208973

-- Given conditions
variables (n : ℕ) (a b : ℝ)

-- Assumptions
lemma condition_n : 0 < n := sorry
lemma condition_a : 0 < a := sorry
lemma condition_b : 0 < b := sorry

-- Statement
theorem AM_GM_Inequality :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2 ^ (n + 1) :=
sorry

-- Equality condition
theorem equality_condition :
  (1 + a / b) ^ n + (1 + b / a) ^ n = 2 ^ (n + 1) ↔ a = b :=
sorry

end AM_GM_Inequality_equality_condition_l208_208973


namespace sum_of_reciprocals_l208_208172

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) : 
  (1 / x) + (1 / y) = 4 :=
by
  sorry

end sum_of_reciprocals_l208_208172


namespace min_disks_required_l208_208573

/-- A structure to hold information about the file storage problem -/
structure FileStorageConditions where
  total_files : ℕ
  disk_capacity : ℝ
  num_files_1_6MB : ℕ
  num_files_1MB : ℕ
  num_files_0_5MB : ℕ

/-- Define specific conditions given in the problem -/
def storage_conditions : FileStorageConditions := {
  total_files := 42,
  disk_capacity := 2.88,
  num_files_1_6MB := 8,
  num_files_1MB := 16,
  num_files_0_5MB := 18 -- Derived from total_files - num_files_1_6MB - num_files_1MB
}

/-- Theorem stating the minimum number of disks required to store all files is 16 -/
theorem min_disks_required (c : FileStorageConditions)
  (h1 : c.total_files = 42)
  (h2 : c.disk_capacity = 2.88)
  (h3 : c.num_files_1_6MB = 8)
  (h4 : c.num_files_1MB = 16)
  (h5 : c.num_files_0_5MB = 18) :
  ∃ n : ℕ, n = 16 := by
  sorry

end min_disks_required_l208_208573


namespace call_duration_l208_208255

def initial_credit : ℝ := 30
def cost_per_minute : ℝ := 0.16
def remaining_credit : ℝ := 26.48

theorem call_duration :
  (initial_credit - remaining_credit) / cost_per_minute = 22 := 
sorry

end call_duration_l208_208255


namespace area_three_arcs_correct_l208_208417

noncomputable def area_of_three_arcs (BC : ℝ) (X midpoint of AB : Prop) (Y midpoint of AC : Prop) (A midpoint of BC : Prop) : ℝ :=
  -- Mention that BC = 1
  let BC_length := (BC = 1) in
  -- Use the given and derived lengths and angles to calculate the area
  let angle_BAC := (36 : ℝ) * real.pi / 180 in
  let angle_BXC := (72 : ℝ) * real.pi / 180 in
  let AC := (sqrt 5 + 1) / 2 in
  let T_1 := real.pi / 20 * (3 + sqrt 5) in
  let T_2 := (3 * real.pi / 10) - (sqrt (10 + 2 * sqrt 5)) / 8 in
  let area := T_1 + 2 * T_2 in
  area

theorem area_three_arcs_correct :
  ∀ BC X midpoint of AB Y midpoint of AC A midpoint of BC,
    BC = 1 →
    area_of_three_arcs BC X Y A = 1.756 :=
by
  intros
  sorry

end area_three_arcs_correct_l208_208417


namespace simplify_fraction_l208_208283

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l208_208283


namespace maria_spent_60_dollars_l208_208026

theorem maria_spent_60_dollars :
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  true
    → total_cost = 60 := 
by 
  intros
  let cost_per_flower := 6
  let roses := 7
  let daisies := 3
  let total_flowers := roses + daisies
  let total_cost := total_flowers * cost_per_flower
  sorry

end maria_spent_60_dollars_l208_208026


namespace bacteria_elimination_l208_208504

theorem bacteria_elimination (d N : ℕ) (hN : N = 50 - 6 * (d - 1)) (hCondition : N ≤ 0) : d = 10 :=
by
  -- We can straightforwardly combine the given conditions and derive the required theorem.
  sorry

end bacteria_elimination_l208_208504


namespace find_p_q_l208_208529

theorem find_p_q 
  (p q: ℚ)
  (a : ℚ × ℚ × ℚ × ℚ := (4, p, -2, 1))
  (b : ℚ × ℚ × ℚ × ℚ := (3, 2, q, -1))
  (orthogonal : (4 * 3 + p * 2 + (-2) * q + 1 * (-1) = 0))
  (equal_magnitudes : (4^2 + p^2 + (-2)^2 + 1^2 = 3^2 + 2^2 + q^2 + (-1)^2))
  : p = -93/44 ∧ q = 149/44 := 
  by 
    sorry

end find_p_q_l208_208529


namespace jack_and_jill_meet_distance_l208_208518

theorem jack_and_jill_meet_distance :
  ∃ t : ℝ, t = 15 / 60 ∧ 14 * t ≤ 4 ∧ 15 * (t - 15 / 60) ≤ 4 ∧
  ( 14 * t - 4 + 18 * (t - 2 / 7) = 15 * (t - 15 / 60) ∨ 15 * (t - 15 / 60) = 4 - 18 * (t - 2 / 7) ) ∧
  4 - 15 * (t - 15 / 60) = 851 / 154 :=
sorry

end jack_and_jill_meet_distance_l208_208518


namespace function_has_one_root_l208_208553

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem function_has_one_root : ∃! x : ℝ, f x = 0 :=
by
  -- Indicate that we haven't included the proof
  sorry

end function_has_one_root_l208_208553


namespace horse_goat_sheep_consumption_l208_208888

theorem horse_goat_sheep_consumption :
  (1 / (1 / (1 : ℝ) + 1 / 2 + 1 / 3)) = 6 / 11 :=
by
  sorry

end horse_goat_sheep_consumption_l208_208888


namespace first_day_is_sunday_l208_208413

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end first_day_is_sunday_l208_208413


namespace xiaoning_pe_comprehensive_score_l208_208445

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l208_208445


namespace total_weight_of_three_packages_l208_208261

theorem total_weight_of_three_packages (a b c d : ℝ)
  (h1 : a + b = 162)
  (h2 : b + c = 164)
  (h3 : c + a = 168) :
  a + b + c = 247 :=
sorry

end total_weight_of_three_packages_l208_208261


namespace shopkeepers_total_profit_percentage_l208_208740

noncomputable def calculateProfitPercentage : ℝ :=
  let oranges := 1000
  let bananas := 800
  let apples := 750
  let rotten_oranges_percentage := 0.12
  let rotten_bananas_percentage := 0.05
  let rotten_apples_percentage := 0.10
  let profit_oranges_percentage := 0.20
  let profit_bananas_percentage := 0.25
  let profit_apples_percentage := 0.15
  let cost_per_orange := 2.5
  let cost_per_banana := 1.5
  let cost_per_apple := 2.0

  let rotten_oranges := rotten_oranges_percentage * oranges
  let rotten_bananas := rotten_bananas_percentage * bananas
  let rotten_apples := rotten_apples_percentage * apples

  let good_oranges := oranges - rotten_oranges
  let good_bananas := bananas - rotten_bananas
  let good_apples := apples - rotten_apples

  let cost_oranges := cost_per_orange * oranges
  let cost_bananas := cost_per_banana * bananas
  let cost_apples := cost_per_apple * apples

  let total_cost := cost_oranges + cost_bananas + cost_apples

  let selling_price_oranges := cost_per_orange * (1 + profit_oranges_percentage) * good_oranges
  let selling_price_bananas := cost_per_banana * (1 + profit_bananas_percentage) * good_bananas
  let selling_price_apples := cost_per_apple * (1 + profit_apples_percentage) * good_apples

  let total_selling_price := selling_price_oranges + selling_price_bananas + selling_price_apples

  let total_profit := total_selling_price - total_cost

  (total_profit / total_cost) * 100

theorem shopkeepers_total_profit_percentage :
  calculateProfitPercentage = 8.03 := sorry

end shopkeepers_total_profit_percentage_l208_208740


namespace sum_YNRB_l208_208809

theorem sum_YNRB :
  ∃ (R Y B N : ℕ),
    (RY = 10 * R + Y) ∧
    (BY = 10 * B + Y) ∧
    (111 * N = (10 * R + Y) * (10 * B + Y)) →
    (Y + N + R + B = 21) :=
sorry

end sum_YNRB_l208_208809


namespace Raja_and_Ram_together_l208_208123

def RajaDays : ℕ := 12
def RamDays : ℕ := 6

theorem Raja_and_Ram_together (W : ℕ) : 
  let RajaRate := W / RajaDays
  let RamRate := W / RamDays
  let CombinedRate := RajaRate + RamRate 
  let DaysTogether := W / CombinedRate 
  DaysTogether = 4 := 
by
  sorry

end Raja_and_Ram_together_l208_208123


namespace max_consecutive_sum_lt_1000_l208_208006

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l208_208006


namespace factorize_expression_l208_208474

variable (a : ℝ)

theorem factorize_expression : a^3 + 4 * a^2 + 4 * a = a * (a + 2)^2 := by
  sorry

end factorize_expression_l208_208474


namespace sum_of_all_possible_values_is_correct_l208_208857

noncomputable def M_sum_of_all_possible_values (a b c M : ℝ) : Prop :=
  M = a * b * c ∧ M = 8 * (a + b + c) ∧ c = a + b ∧ b = 2 * a

theorem sum_of_all_possible_values_is_correct :
  ∃ M, (∃ a b c, M_sum_of_all_possible_values a b c M) ∧ M = 96 * Real.sqrt 2 := by
  sorry

end sum_of_all_possible_values_is_correct_l208_208857


namespace kx2_kx_1_pos_l208_208187

theorem kx2_kx_1_pos (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end kx2_kx_1_pos_l208_208187


namespace number_of_valid_integers_l208_208787

def count_valid_numbers : Nat :=
  let one_digit_count : Nat := 6
  let two_digit_count : Nat := 6 * 6
  let three_digit_count : Nat := 6 * 6 * 6
  one_digit_count + two_digit_count + three_digit_count

theorem number_of_valid_integers :
  count_valid_numbers = 258 :=
sorry

end number_of_valid_integers_l208_208787


namespace solve_inequality_2_star_x_l208_208348

theorem solve_inequality_2_star_x :
  ∀ x : ℝ, 
  6 < (2 * x - 2 - x + 3) ∧ (2 * x - 2 - x + 3) < 7 ↔ 5 < x ∧ x < 6 :=
by sorry

end solve_inequality_2_star_x_l208_208348


namespace selecting_elements_l208_208476

theorem selecting_elements (P Q S : ℕ) (a : ℕ) 
    (h1 : P = Nat.choose 17 (2 * a - 1))
    (h2 : Q = Nat.choose 17 (2 * a))
    (h3 : S = Nat.choose 18 12) :
    P + Q = S → (a = 3 ∨ a = 6) :=
by
  sorry

end selecting_elements_l208_208476


namespace floor_ineq_l208_208349

theorem floor_ineq (x y : ℝ) : 
  Int.floor (2 * x) + Int.floor (2 * y) ≥ Int.floor x + Int.floor y + Int.floor (x + y) := 
sorry

end floor_ineq_l208_208349


namespace sin_A_triangle_abc_l208_208800

open Real

noncomputable def sin_A_tri (A B C : ℝ) (a b c : ℝ) (h1 : B = π / 4) (h2 : (c * sin A) = 1 / 3 * c) : Real :=
  sin A

theorem sin_A_triangle_abc (a b c : ℝ) (A B C : ℝ) 
  (h_B_eq : B = π / 4) 
  (h_height : (a * sin A = 1/3 * a))
   : sin A = 3 * sqrt 10 / 10 := 
by
  sorry

end sin_A_triangle_abc_l208_208800


namespace xiao_ming_shopping_l208_208829

theorem xiao_ming_shopping :
  ∃ x : ℕ, x ≤ 16 ∧ 6 * x ≤ 100 ∧ 100 - 6 * x = 28 :=
by
  -- Given that:
  -- 1. x is the same amount spent in each of the six stores.
  -- 2. Total money spent, 6 * x, must be less than or equal to 100.
  -- 3. We seek to prove that Xiao Ming has 28 yuan left.
  sorry

end xiao_ming_shopping_l208_208829


namespace part_a_part_b_part_c_l208_208394

noncomputable def probability_of_meeting : ℚ :=
  let total_area := 60 * 60
  let meeting_area := 2 * (1/2 * 50 * 50)
  meeting_area / total_area

theorem part_a : probability_of_meeting = 11 / 36 := by
  sorry

noncomputable def probability_of_meeting_b : ℚ :=
  let total_area := 30 * 60
  let meeting_area := 2 * (1/2 * 20 * 30 - (1/2 * 10 * 10))
  meeting_area / total_area

theorem part_b : probability_of_meeting_b = 1 / 6 := by
  sorry

noncomputable def probability_of_meeting_c : ℚ :=
  let total_area := 50 * 60
  let meeting_area := (40 * 10 + 1/2 * 10 * 10 + 1/2 * 10 * 10)
  meeting_area / total_area

theorem part_c : probability_of_meeting_c = 3 / 200 := by
  sorry

end part_a_part_b_part_c_l208_208394


namespace find_z_l208_208926

open Complex

noncomputable def sqrt_five : ℝ := Real.sqrt 5

theorem find_z (z : ℂ) 
  (hz1 : z.re < 0) 
  (hz2 : z.im > 0) 
  (h_modulus : abs z = 3) 
  (h_real_part : z.re = -sqrt_five) : 
  z = -sqrt_five + 2 * I :=
by
  sorry

end find_z_l208_208926


namespace range_of_m_l208_208071

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
sorry

end range_of_m_l208_208071


namespace original_number_l208_208713

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l208_208713


namespace simplify_expression_l208_208268

theorem simplify_expression : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by {
  sorry
}

end simplify_expression_l208_208268


namespace inequality_proof_l208_208353

variable {a b c : ℝ}

theorem inequality_proof (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2*Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := 
sorry

end inequality_proof_l208_208353


namespace sum_of_arithmetic_sequence_l208_208300

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) (S5 : S 5 = 30) (S10 : S 10 = 110) : S 15 = 240 :=
by
  sorry

end sum_of_arithmetic_sequence_l208_208300


namespace num_rectangular_tables_l208_208320

theorem num_rectangular_tables (R : ℕ) 
  (rectangular_tables_seat : R * 10 = 70) :
  R = 7 := by
  sorry

end num_rectangular_tables_l208_208320


namespace alcohol_percentage_in_second_vessel_l208_208898

open Real

theorem alcohol_percentage_in_second_vessel (x : ℝ) (h : (0.2 * 2) + (0.01 * x * 6) = 8 * 0.28) : 
  x = 30.666666666666668 :=
by 
  sorry

end alcohol_percentage_in_second_vessel_l208_208898


namespace janice_total_earnings_l208_208961

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l208_208961


namespace min_abs_sum_l208_208702

theorem min_abs_sum (x : ℝ) : 
  (∃ x, (∀ y, ∥ y + 1 ∥ + ∥ y + 3 ∥ + ∥ y + 6 ∥ ≥ ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥) ∧ 
        ∥ x + 1 ∥ + ∥ x + 3 ∥ + ∥ x + 6 ∥ = 5) := 
sorry

end min_abs_sum_l208_208702


namespace rest_area_location_l208_208852

theorem rest_area_location : 
  ∃ (rest_area_milepost : ℕ), 
    let first_exit := 23
    let seventh_exit := 95
    let distance := seventh_exit - first_exit
    let halfway_distance := distance / 2
    rest_area_milepost = first_exit + halfway_distance :=
by
  sorry

end rest_area_location_l208_208852


namespace original_price_l208_208555

theorem original_price (P : ℝ) (h1 : P + 0.10 * P = 330) : P = 300 := 
by
  sorry

end original_price_l208_208555


namespace carmen_sold_1_box_of_fudge_delights_l208_208465

noncomputable def boxes_of_fudge_delights (total_earned: ℝ) (samoas_price: ℝ) (thin_mints_price: ℝ) (fudge_delights_price: ℝ) (sugar_cookies_price: ℝ) (samoas_sold: ℝ) (thin_mints_sold: ℝ) (sugar_cookies_sold: ℝ): ℝ :=
  let samoas_total := samoas_price * samoas_sold
  let thin_mints_total := thin_mints_price * thin_mints_sold
  let sugar_cookies_total := sugar_cookies_price * sugar_cookies_sold
  let other_cookies_total := samoas_total + thin_mints_total + sugar_cookies_total
  (total_earned - other_cookies_total) / fudge_delights_price

theorem carmen_sold_1_box_of_fudge_delights: boxes_of_fudge_delights 42 4 3.5 5 2 3 2 9 = 1 :=
by
  sorry

end carmen_sold_1_box_of_fudge_delights_l208_208465


namespace remainder_when_squared_mod_seven_l208_208532

theorem remainder_when_squared_mod_seven
  (x y : ℤ) (k m : ℤ)
  (hx : x = 52 * k + 19)
  (hy : 3 * y = 7 * m + 5) :
  ((x + 2 * y)^2 % 7) = 1 := by
  sorry

end remainder_when_squared_mod_seven_l208_208532


namespace andrei_club_visits_l208_208570

theorem andrei_club_visits (d c : ℕ) (h : 15 * d + 11 * c = 115) : d + c = 9 :=
by
  sorry

end andrei_club_visits_l208_208570


namespace relationship_abc_l208_208480

open Real

variable {x : ℝ}
variable (a b c : ℝ)
variable (h1 : 0 < x ∧ x ≤ 1)
variable (h2 : a = (sin x / x) ^ 2)
variable (h3 : b = sin x / x)
variable (h4 : c = sin (x^2) / x^2)

theorem relationship_abc (h1 : 0 < x ∧ x ≤ 1) (h2 : a = (sin x / x) ^ 2) (h3 : b = sin x / x) (h4 : c = sin (x^2) / x^2) :
  a < b ∧ b ≤ c :=
sorry

end relationship_abc_l208_208480


namespace log_eq_15_given_log_base3_x_eq_5_l208_208621

variable (x : ℝ)
variable (log_base3_x : ℝ)
variable (h : log_base3_x = 5)

theorem log_eq_15_given_log_base3_x_eq_5 (h : log_base3_x = 5) : log_base3_x * 3 = 15 :=
by
  sorry

end log_eq_15_given_log_base3_x_eq_5_l208_208621


namespace jonas_shoes_l208_208521

theorem jonas_shoes (socks pairs_of_pants t_shirts shoes : ℕ) (new_socks : ℕ) (h1 : socks = 20) (h2 : pairs_of_pants = 10) (h3 : t_shirts = 10) (h4 : new_socks = 35 ∧ (socks + new_socks = 35)) :
  shoes = 35 :=
by
  sorry

end jonas_shoes_l208_208521


namespace number_of_valid_m_values_l208_208954

/--
In the coordinate plane, construct a right triangle with its legs parallel to the x and y axes, and with the medians on its legs lying on the lines y = 3x + 1 and y = mx + 2. 
Prove that the number of values for the constant m such that this triangle exists is 2.
-/
theorem number_of_valid_m_values : 
  ∃ (m : ℝ), 
    (∃ (a b : ℝ), 
      (∀ D E : ℝ × ℝ, D = (a / 2, 0) ∧ E = (0, b / 2) →
      D.2 = 3 * D.1 + 1 ∧ 
      E.2 = m * E.1 + 2)) → 
    (number_of_solutions_for_m = 2) 
  :=
sorry

end number_of_valid_m_values_l208_208954


namespace abs_inequality_solution_l208_208133

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l208_208133


namespace sister_age_is_one_l208_208674

variable (B S : ℕ)

theorem sister_age_is_one (h : B = B * S) : S = 1 :=
by {
  sorry
}

end sister_age_is_one_l208_208674


namespace bobby_weekly_salary_l208_208589

variable (S : ℝ)
variables (federal_tax : ℝ) (state_tax : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) (city_fee : ℝ) (net_paycheck : ℝ)

def bobby_salary_equation := 
  S - (federal_tax * S) - (state_tax * S) - health_insurance - life_insurance - city_fee = net_paycheck

theorem bobby_weekly_salary 
  (S : ℝ) 
  (federal_tax : ℝ := 1/3) 
  (state_tax : ℝ := 0.08) 
  (health_insurance : ℝ := 50) 
  (life_insurance : ℝ := 20) 
  (city_fee : ℝ := 10) 
  (net_paycheck : ℝ := 184) 
  (valid_solution : bobby_salary_equation S (1/3) 0.08 50 20 10 184) : 
  S = 450.03 := 
  sorry

end bobby_weekly_salary_l208_208589


namespace g_difference_l208_208655

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 3) * (n + 5) + 2

theorem g_difference (s : ℕ) : g s - g (s - 1) = (3 * s^2 + 9 * s + 8) / 4 :=
by
  -- skip the proof
  sorry

end g_difference_l208_208655


namespace alcohol_solution_l208_208016

/-- 
A 40-liter solution of alcohol and water is 5 percent alcohol. If 3.5 liters of alcohol and 6.5 liters of water are added to this solution, 
what percent of the solution produced is alcohol? 
-/
theorem alcohol_solution (original_volume : ℝ) (original_percent_alcohol : ℝ)
                        (added_alcohol : ℝ) (added_water : ℝ) :
  original_volume = 40 →
  original_percent_alcohol = 5 →
  added_alcohol = 3.5 →
  added_water = 6.5 →
  (100 * (original_volume * original_percent_alcohol / 100 + added_alcohol) / (original_volume + added_alcohol + added_water)) = 11 := 
by 
  intros h1 h2 h3 h4
  sorry

end alcohol_solution_l208_208016


namespace Michael_rides_six_miles_l208_208676

theorem Michael_rides_six_miles
  (rate : ℝ)
  (time : ℝ)
  (interval_time : ℝ)
  (interval_distance : ℝ)
  (intervals : ℝ)
  (total_distance : ℝ) :
  rate = 1.5 ∧ time = 40 ∧ interval_time = 10 ∧ interval_distance = 1.5 ∧ intervals = time / interval_time ∧ total_distance = intervals * interval_distance →
  total_distance = 6 :=
by
  intros h
  -- Placeholder for the proof
  sorry

end Michael_rides_six_miles_l208_208676


namespace xiao_gao_actual_score_l208_208944

-- Definitions from the conditions:
def standard_score : ℕ := 80
def xiao_gao_recorded_score : ℤ := 12

-- Proof problem statement:
theorem xiao_gao_actual_score : (standard_score : ℤ) + xiao_gao_recorded_score = 92 :=
by
  sorry

end xiao_gao_actual_score_l208_208944


namespace solve_abs_equation_l208_208288

theorem solve_abs_equation (x : ℝ) :
  (|2 * x + 1| - |x - 5| = 6) ↔ (x = -12 ∨ x = 10 / 3) :=
by sorry

end solve_abs_equation_l208_208288


namespace opposite_of_2023_is_neg_2023_l208_208554

def opposite_of (x : Int) : Int := -x

theorem opposite_of_2023_is_neg_2023 : opposite_of 2023 = -2023 :=
by
  sorry

end opposite_of_2023_is_neg_2023_l208_208554


namespace xiaoning_pe_comprehensive_score_l208_208446

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end xiaoning_pe_comprehensive_score_l208_208446


namespace nests_count_l208_208689

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l208_208689


namespace pyramid_surface_area_l208_208909

theorem pyramid_surface_area (base_edge volume : ℝ)
  (h_base_edge : base_edge = 1)
  (h_volume : volume = 1) :
  let height := 3
  let slant_height := Real.sqrt (9.25)
  let base_area := base_edge * base_edge
  let lateral_area := 4 * (1 / 2 * base_edge * slant_height)
  let total_surface_area := base_area + lateral_area
  total_surface_area = 7.082 :=
by
  sorry

end pyramid_surface_area_l208_208909


namespace highest_value_meter_l208_208032

theorem highest_value_meter (A B C : ℝ) 
  (h_avg : (A + B + C) / 3 = 6)
  (h_A_min : A = 2)
  (h_B_min : B = 2) : C = 14 :=
by {
  sorry
}

end highest_value_meter_l208_208032


namespace sue_answer_is_106_l208_208203

-- Definitions based on conditions
def ben_step1 (x : ℕ) : ℕ := x * 3
def ben_step2 (x : ℕ) : ℕ := ben_step1 x + 2
def ben_step3 (x : ℕ) : ℕ := ben_step2 x * 2

def sue_step1 (y : ℕ) : ℕ := y + 3
def sue_step2 (y : ℕ) : ℕ := sue_step1 y - 2
def sue_step3 (y : ℕ) : ℕ := sue_step2 y * 2

-- Ben starts with the number 8
def ben_number : ℕ := 8

-- Ben gives the number to Sue
def given_to_sue : ℕ := ben_step3 ben_number

-- Lean statement to prove
theorem sue_answer_is_106 : sue_step3 given_to_sue = 106 :=
by
  sorry

end sue_answer_is_106_l208_208203


namespace line_tangent_to_parabola_l208_208087

theorem line_tangent_to_parabola (c : ℝ) : (∀ (x y : ℝ), 2 * x - y + c = 0 ∧ x^2 = 4 * y) → c = -4 := by
  sorry

end line_tangent_to_parabola_l208_208087


namespace Jessie_points_l208_208846

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l208_208846


namespace sum_of_integers_ending_in_2_between_100_and_500_l208_208050

theorem sum_of_integers_ending_in_2_between_100_and_500 :
  let s : List ℤ := List.range' 102 400 10
  let sum_of_s := s.sum
  sum_of_s = 11880 :=
by
  sorry

end sum_of_integers_ending_in_2_between_100_and_500_l208_208050


namespace number_divisible_by_75_l208_208511

def is_two_digit (x : ℕ) := x >= 10 ∧ x < 100

theorem number_divisible_by_75 {a b : ℕ} (h1 : a * b = 35) (h2 : is_two_digit (10 * a + b)) : (10 * a + b) % 75 = 0 :=
sorry

end number_divisible_by_75_l208_208511


namespace average_age_decrease_l208_208549

theorem average_age_decrease (N T : ℕ) (h₁ : (T : ℝ) / N - 3 = (T - 30 : ℝ) / N) : N = 10 :=
sorry

end average_age_decrease_l208_208549


namespace least_number_of_stamps_l208_208256

def min_stamps (x y : ℕ) : ℕ := x + y

theorem least_number_of_stamps {x y : ℕ} (h : 5 * x + 7 * y = 50) 
  : min_stamps x y = 8 :=
sorry

end least_number_of_stamps_l208_208256


namespace arithmetic_sequence_common_difference_and_m_l208_208907

theorem arithmetic_sequence_common_difference_and_m (S : ℕ → ℤ) (a : ℕ → ℤ) (m d : ℕ) 
(h1 : S (m-1) = -2) (h2 : S m = 0) (h3 : S (m+1) = 3) :
  d = 1 ∧ m = 5 :=
by sorry

end arithmetic_sequence_common_difference_and_m_l208_208907


namespace find_x_for_parallel_vectors_l208_208786

-- Definitions for the given conditions
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- The proof statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : parallel a (b x)) : x = 6 :=
  sorry

end find_x_for_parallel_vectors_l208_208786


namespace length_of_QR_l208_208647

theorem length_of_QR {P Q R N : Type} 
  (PQ PR QR : ℝ) (QN NR PN : ℝ)
  (h1 : PQ = 5)
  (h2 : PR = 10)
  (h3 : QN = 3 * NR)
  (h4 : PN = 6)
  (h5 : QR = QN + NR) :
  QR = 724 / 3 :=
by sorry

end length_of_QR_l208_208647


namespace months_to_save_l208_208552

/-- The grandfather saves 530 yuan from his pension every month. -/
def savings_per_month : ℕ := 530

/-- The price of the smartphone is 2000 yuan. -/
def smartphone_price : ℕ := 2000

/-- The number of months needed to save enough money to buy the smartphone. -/
def months_needed : ℕ := smartphone_price / savings_per_month

/-- Proof that the number of months needed is 4. -/
theorem months_to_save : months_needed = 4 :=
by
  sorry

end months_to_save_l208_208552


namespace crayons_left_l208_208691

theorem crayons_left (initial_crayons : ℕ) (crayons_taken : ℕ) : initial_crayons = 7 → crayons_taken = 3 → initial_crayons - crayons_taken = 4 :=
by
  sorry

end crayons_left_l208_208691


namespace selection_competition_l208_208578

variables (p q r : Prop)

theorem selection_competition 
  (h1 : p ∨ q) 
  (h2 : ¬ (p ∧ q)) 
  (h3 : ¬ q ∧ r) : p ∧ ¬ q ∧ r :=
by
  sorry

end selection_competition_l208_208578


namespace equation_of_hyperbola_l208_208354

-- Definitions for conditions

def center_at_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def focus_point (focus : ℝ × ℝ) : Prop :=
  focus = (Real.sqrt 2, 0)

def distance_to_asymptote (focus : ℝ × ℝ) (distance : ℝ) : Prop :=
  -- Placeholder for the actual distance calculation
  distance = 1 -- The given distance condition in the problem

-- The mathematical proof problem statement

theorem equation_of_hyperbola :
  center_at_origin (0,0) ∧
  focus_point (Real.sqrt 2, 0) ∧
  distance_to_asymptote (Real.sqrt 2, 0) 1 → 
    ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
    (a^2 + b^2 = 2) ∧ (a^2 = 1) ∧ (b^2 = 1) ∧ 
    (∀ x y : ℝ, b^2*y^2 = x^2 - a^2*y^2 → (y = 0 ∧ x^2 = 1)) :=
sorry

end equation_of_hyperbola_l208_208354


namespace alcohol_solution_volume_l208_208189

theorem alcohol_solution_volume (V : ℝ) (h1 : 0.42 * V = 0.33 * (V + 3)) : V = 11 :=
by
  sorry

end alcohol_solution_volume_l208_208189


namespace problem_solution_l208_208502

noncomputable def quadratic_symmetric_b (a : ℝ) : ℝ :=
  2 * (1 - a)

theorem problem_solution (a : ℝ) (h1 : quadratic_symmetric_b a = 6) :
  b = 6 :=
by
  sorry

end problem_solution_l208_208502


namespace problem_solution_l208_208067

-- Definitions and Assumptions
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, f x - (deriv^[2]) f x > 0)

-- Statement to Prove
theorem problem_solution : e * f 2015 > f 2016 :=
by
  sorry

end problem_solution_l208_208067


namespace wrong_conclusion_l208_208064

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem wrong_conclusion {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : 2 * a + b = 0) (h₂ : a + b + c = 3) (h₃ : 4 * a + 2 * b + c = 8) :
  quadratic a b c (-1) ≠ 0 :=
sorry

end wrong_conclusion_l208_208064


namespace trains_same_distance_at_meeting_l208_208896

theorem trains_same_distance_at_meeting
  (d v : ℝ) (h_d : 0 < d) (h_v : 0 < v) :
  ∃ t : ℝ, v * t + v * (t - 1) = d ∧ 
  v * t = (d + v) / 2 ∧ 
  d - (v * (t - 1)) = (d + v) / 2 :=
by
  sorry

end trains_same_distance_at_meeting_l208_208896


namespace value_of_a_c_l208_208163

theorem value_of_a_c {a b c d : ℝ} :
  (∀ x y : ℝ, y = -|x - a| + b → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) ∧
  (∀ x y : ℝ, y = |x - c| - d → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) →
  a + c = 8 :=
by
  sorry

end value_of_a_c_l208_208163


namespace prop_logic_example_l208_208636

theorem prop_logic_example (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end prop_logic_example_l208_208636


namespace arithmetic_sequence_sum_20_l208_208513

open BigOperators

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + (a 1 - a 0)

theorem arithmetic_sequence_sum_20 {a : ℕ → ℤ} (h_arith : is_arithmetic_sequence a)
    (h1 : a 0 + a 1 + a 2 = -24)
    (h18 : a 17 + a 18 + a 19 = 78) :
    ∑ i in Finset.range 20, a i = 180 :=
sorry

end arithmetic_sequence_sum_20_l208_208513


namespace solve_squares_and_circles_l208_208023

theorem solve_squares_and_circles (x y : ℝ) :
  (5 * x + 2 * y = 39) ∧ (3 * x + 3 * y = 27) → (x = 7) ∧ (y = 2) :=
by
  intro h
  sorry

end solve_squares_and_circles_l208_208023


namespace total_value_of_remaining_books_l208_208590

-- initial definitions
def total_books : ℕ := 55
def hardback_books : ℕ := 10
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

-- calculate remaining books
def remaining_books : ℕ := total_books - books_sold

-- calculate remaining hardback and paperback books
def remaining_hardback_books : ℕ := hardback_books
def remaining_paperback_books : ℕ := remaining_books - remaining_hardback_books

-- calculate total values
def remaining_hardback_value : ℕ := remaining_hardback_books * hardback_price
def remaining_paperback_value : ℕ := remaining_paperback_books * paperback_price

-- total value of remaining books
def total_remaining_value : ℕ := remaining_hardback_value + remaining_paperback_value

theorem total_value_of_remaining_books : total_remaining_value = 510 := by
  -- calculation steps are skipped as instructed
  sorry

end total_value_of_remaining_books_l208_208590


namespace positive_solution_sqrt_eq_l208_208763

theorem positive_solution_sqrt_eq (y : ℝ) (hy_pos : 0 < y) : 
    (∃ a, a = y ∧ a^2 = y * a) ∧ (∃ b, b = y ∧ b^2 = y + b) ∧ y = 2 :=
by 
  sorry

end positive_solution_sqrt_eq_l208_208763


namespace possible_landing_l208_208506

-- There are 1985 airfields
def num_airfields : ℕ := 1985

-- 50 airfields where planes could potentially land
def num_land_airfields : ℕ := 50

-- Define the structure of the problem
structure AirfieldSetup :=
  (airfields : Fin num_airfields → Fin num_land_airfields)

-- There exists a configuration such that the conditions are met
theorem possible_landing : ∃ (setup : AirfieldSetup), 
  (∀ i : Fin num_airfields, -- For each airfield
    ∃ j : Fin num_land_airfields, -- There exists a landing airfield
    setup.airfields i = j) -- The plane lands at this airfield.
:=
sorry

end possible_landing_l208_208506


namespace elizabeth_net_profit_l208_208220

noncomputable section

def net_profit : ℝ :=
  let cost_bag_1 := 2.5
  let cost_bag_2 := 3.5
  let total_cost := 10 * cost_bag_1 + 10 * cost_bag_2
  let selling_price := 6.0
  let sold_bags_1_no_discount := 7 * selling_price
  let sold_bags_2_no_discount := 8 * selling_price
  let discount_1 := 0.2
  let discount_2 := 0.3
  let discounted_price_1 := selling_price * (1 - discount_1)
  let discounted_price_2 := selling_price * (1 - discount_2)
  let sold_bags_1_with_discount := 3 * discounted_price_1
  let sold_bags_2_with_discount := 2 * discounted_price_2
  let total_revenue := sold_bags_1_no_discount + sold_bags_2_no_discount + sold_bags_1_with_discount + sold_bags_2_with_discount
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.8 := by
  sorry

end elizabeth_net_profit_l208_208220


namespace boys_and_girls_l208_208508

theorem boys_and_girls (x y : ℕ) (h1 : x + y = 21) (h2 : 5 * x + 2 * y = 69) : x = 9 ∧ y = 12 :=
by 
  sorry

end boys_and_girls_l208_208508


namespace tan_150_degree_is_correct_l208_208600

noncomputable def tan_150_degree_is_negative_sqrt_3_div_3 : Prop :=
  let theta := Real.pi * 150 / 180
  let ref_angle := Real.pi * 30 / 180
  let cos_150 := -Real.cos ref_angle
  let sin_150 := Real.sin ref_angle
  Real.tan theta = -Real.sqrt 3 / 3

theorem tan_150_degree_is_correct :
  tan_150_degree_is_negative_sqrt_3_div_3 :=
by
  sorry

end tan_150_degree_is_correct_l208_208600


namespace score_order_l208_208364

variable (A B C D : ℕ)

theorem score_order (h1 : A + B = C + D) (h2 : C + A > B + D) (h3 : C > A + B) :
  (C > A ∧ A > B ∧ B > D) :=
by
  sorry

end score_order_l208_208364


namespace explain_education_policy_l208_208950

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l208_208950


namespace ratio_large_to_small_l208_208738

-- Definitions of the conditions
def total_fries_sold : ℕ := 24
def small_fries_sold : ℕ := 4
def large_fries_sold : ℕ := total_fries_sold - small_fries_sold

-- The proof goal
theorem ratio_large_to_small : large_fries_sold / small_fries_sold = 5 :=
by
  -- Mathematical steps would go here, but we skip with sorry
  sorry

end ratio_large_to_small_l208_208738


namespace edges_after_truncation_l208_208755

-- Define a regular tetrahedron with 4 vertices and 6 edges
structure Tetrahedron :=
  (vertices : ℕ)
  (edges : ℕ)

-- Initial regular tetrahedron
def initial_tetrahedron : Tetrahedron :=
  { vertices := 4, edges := 6 }

-- Function to calculate the number of edges after truncating vertices
def truncated_edges (t : Tetrahedron) (vertex_truncations : ℕ) (new_edges_per_vertex : ℕ) : ℕ :=
  vertex_truncations * new_edges_per_vertex

-- Given a regular tetrahedron and the truncation process
def resulting_edges (t : Tetrahedron) (vertex_truncations : ℕ) :=
  truncated_edges t vertex_truncations 3

-- Problem statement: Proving the resulting figure has 12 edges
theorem edges_after_truncation :
  resulting_edges initial_tetrahedron 4 = 12 :=
  sorry

end edges_after_truncation_l208_208755


namespace one_cow_one_bag_l208_208184

def husk_eating (C B D : ℕ) : Prop :=
  C * D / B = D

theorem one_cow_one_bag (C B D n : ℕ) (h : husk_eating C B D) (hC : C = 46) (hB : B = 46) (hD : D = 46) : n = D :=
by
  rw [hC, hB, hD] at h
  sorry

end one_cow_one_bag_l208_208184


namespace shorter_leg_of_right_triangle_l208_208374

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l208_208374


namespace complex_shape_perimeter_l208_208252

theorem complex_shape_perimeter :
  ∃ h : ℝ, 12 * h - 20 = 95 ∧
  (24 + ((230 / 12) - 2) + 10 : ℝ) = 51.1667 :=
by
  sorry

end complex_shape_perimeter_l208_208252


namespace shaded_region_area_l208_208609

def area_of_square (side : ℕ) : ℕ := side * side

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

def combined_area_of_triangles (base height : ℕ) : ℕ := 2 * area_of_triangle base height

def shaded_area (square_side : ℕ) (triangle_base triangle_height : ℕ) : ℕ :=
  area_of_square square_side - combined_area_of_triangles triangle_base triangle_height

theorem shaded_region_area (h₁ : area_of_square 40 = 1600)
                          (h₂ : area_of_triangle 30 30 = 450)
                          (h₃ : combined_area_of_triangles 30 30 = 900) :
  shaded_area 40 30 30 = 700 :=
by
  sorry

end shaded_region_area_l208_208609


namespace area_of_isosceles_triangle_PQR_l208_208644

noncomputable def area_of_triangle (P Q R : ℝ) (PQ PR QR PS QS SR : ℝ) : Prop :=
PQ = 17 ∧ PR = 17 ∧ QR = 16 ∧ PS = 15 ∧ QS = 8 ∧ SR = 8 →
(1 / 2) * QR * PS = 120

theorem area_of_isosceles_triangle_PQR :
  ∀ (P Q R : ℝ), 
  ∀ (PQ PR QR PS QS SR : ℝ), 
  PQ = 17 → PR = 17 → QR = 16 → PS = 15 → QS = 8 → SR = 8 →
  area_of_triangle P Q R PQ PR QR PS QS SR := 
by
  intros P Q R PQ PR QR PS QS SR hPQ hPR hQR hPS hQS hSR
  unfold area_of_triangle
  simp [hPQ, hPR, hQR, hPS, hQS, hSR]
  sorry

end area_of_isosceles_triangle_PQR_l208_208644


namespace fifteenth_term_is_143_l208_208998

noncomputable def first_term : ℕ := 3
noncomputable def second_term : ℕ := 13
noncomputable def third_term : ℕ := 23
noncomputable def common_difference : ℕ := second_term - first_term
noncomputable def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

theorem fifteenth_term_is_143 :
  nth_term 15 = 143 := by
  sorry

end fifteenth_term_is_143_l208_208998


namespace train_journey_time_l208_208434

theorem train_journey_time :
  ∃ T : ℝ, (30 : ℝ) / 60 = (7 / 6 * T) - T ∧ T = 3 :=
by
  sorry

end train_journey_time_l208_208434


namespace max_consecutive_integers_lt_1000_l208_208001

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l208_208001


namespace darnel_difference_l208_208469

theorem darnel_difference (sprint_1 jog_1 sprint_2 jog_2 sprint_3 jog_3 : ℝ)
  (h_sprint_1 : sprint_1 = 0.8932)
  (h_jog_1 : jog_1 = 0.7683)
  (h_sprint_2 : sprint_2 = 0.9821)
  (h_jog_2 : jog_2 = 0.4356)
  (h_sprint_3 : sprint_3 = 1.2534)
  (h_jog_3 : jog_3 = 0.6549) :
  (sprint_1 + sprint_2 + sprint_3 - (jog_1 + jog_2 + jog_3)) = 1.2699 := by
  sorry

end darnel_difference_l208_208469


namespace union_of_A_and_B_l208_208486

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 4, 5} := 
by
  sorry

end union_of_A_and_B_l208_208486


namespace retailer_markup_percentage_l208_208577

-- Definitions of initial conditions
def CP : ℝ := 100
def intended_profit_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25
def actual_profit_percentage : ℝ := 0.2375

-- Proving the retailer marked his goods at 65% above the cost price
theorem retailer_markup_percentage : ∃ (MP : ℝ), ((0.75 * MP - CP) / CP) * 100 = actual_profit_percentage * 100 ∧ ((MP - CP) / CP) * 100 = 65 := 
by
  -- The mathematical proof steps mean to be filled here  
  sorry

end retailer_markup_percentage_l208_208577


namespace largest_sum_l208_208593

theorem largest_sum :
  let s1 := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let s2 := (1 : ℚ) / 3 + (1 : ℚ) / 5
  let s3 := (1 : ℚ) / 3 + (1 : ℚ) / 2
  let s4 := (1 : ℚ) / 3 + (1 : ℚ) / 9
  let s5 := (1 : ℚ) / 3 + (1 : ℚ) / 6
  in max s1 (max s2 (max s3 (max s4 s5))) = (5 : ℚ) / 6 := by
  sorry

end largest_sum_l208_208593


namespace shorter_leg_in_right_triangle_l208_208369

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l208_208369


namespace repayment_is_correct_l208_208448

noncomputable def repayment_amount (a r : ℝ) : ℝ := a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1)

theorem repayment_is_correct (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  repayment_amount a r = a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1) :=
by
  sorry

end repayment_is_correct_l208_208448


namespace even_perfect_square_factors_l208_208078

theorem even_perfect_square_factors : 
  (∃ count : ℕ, count = 3 * 2 * 3 ∧ 
    (∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ b ≤ 3 ∧ c % 2 = 0 ∧ c ≤ 4) → 
      (2^a * 7^b * 3^c ∣ 2^6 * 7^3 * 3^4))) :=
sorry

end even_perfect_square_factors_l208_208078


namespace min_value_of_f_l208_208699

def f (x : ℝ) := abs (x + 1) + abs (x + 3) + abs (x + 6)

theorem min_value_of_f : ∃ (x : ℝ), f x = 5 :=
by
  use -3
  simp [f]
  sorry

end min_value_of_f_l208_208699


namespace part_a_part_b_l208_208981

def P (m n : ℕ) : ℕ := m^2003 * n^2017 - m^2017 * n^2003

theorem part_a (m n : ℕ) : P m n % 24 = 0 := 
by sorry

theorem part_b : ∃ (m n : ℕ), P m n % 7 ≠ 0 :=
by sorry

end part_a_part_b_l208_208981


namespace intercepts_correct_l208_208754

-- Define the equation of the line
def line_eq (x y : ℝ) := 5 * x - 2 * y - 10 = 0

-- Define the intercepts
def x_intercept : ℝ := 2
def y_intercept : ℝ := -5

-- Prove that the intercepts are as stated
theorem intercepts_correct :
  (∃ x, line_eq x 0 ∧ x = x_intercept) ∧
  (∃ y, line_eq 0 y ∧ y = y_intercept) :=
by
  sorry

end intercepts_correct_l208_208754


namespace board_tiling_condition_l208_208614

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l208_208614


namespace find_min_difference_l208_208102

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l208_208102


namespace cube_volume_and_diagonal_l208_208707

theorem cube_volume_and_diagonal (A : ℝ) (s : ℝ) (V : ℝ) (d : ℝ) 
  (h1 : A = 864)
  (h2 : 6 * s^2 = A)
  (h3 : V = s^3)
  (h4 : d = s * Real.sqrt 3) :
  V = 1728 ∧ d = 12 * Real.sqrt 3 :=
by 
  sorry

end cube_volume_and_diagonal_l208_208707


namespace ratio_of_perimeters_l208_208146

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l208_208146


namespace number_of_zeros_l208_208165

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - Real.log x / Real.log 2

theorem number_of_zeros : (∀ x > 0, f(x) < f(x + 1)) 
  ∧ f(1) > 0 
  ∧ f(2) < 0 
  → ∃! x, 0 < x ∧ f(x) = 0 :=
sorry

end number_of_zeros_l208_208165


namespace base_conversion_problem_l208_208993

theorem base_conversion_problem (b : ℕ) (h : b^2 + 2 * b - 25 = 0) : b = 3 :=
sorry

end base_conversion_problem_l208_208993


namespace find_min_difference_l208_208103

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l208_208103


namespace ratio_students_preference_l208_208642

theorem ratio_students_preference
  (total_students : ℕ)
  (mac_preference : ℕ)
  (windows_preference : ℕ)
  (no_preference : ℕ)
  (students_equally_preferred_both : ℕ)
  (h_total : total_students = 210)
  (h_mac : mac_preference = 60)
  (h_windows : windows_preference = 40)
  (h_no_pref : no_preference = 90)
  (h_students_equally : students_equally_preferred_both = total_students - (mac_preference + windows_preference + no_preference)) :
  (students_equally_preferred_both : ℚ) / mac_preference = 1 / 3 := 
by
  sorry

end ratio_students_preference_l208_208642


namespace subscription_total_amount_l208_208460

theorem subscription_total_amount 
  (A B C : ℝ)
  (profit_C profit_total : ℝ)
  (subscription_A subscription_B subscription_C : ℝ)
  (subscription_total : ℝ)
  (hA : subscription_A = subscription_B + 4000)
  (hB : subscription_B = subscription_C + 5000)
  (hc_share : profit_C = 8400)
  (total_profit : profit_total = 35000)
  (h_ratio : profit_C / profit_total = subscription_C / subscription_total)
  (h_subs : subscription_total = subscription_A + subscription_B + subscription_C)
  : subscription_total = 50000 := 
sorry

end subscription_total_amount_l208_208460


namespace nests_count_l208_208690

theorem nests_count :
  ∃ (N : ℕ), (6 = N + 3) ∧ (N = 3) :=
by
  sorry

end nests_count_l208_208690


namespace sufficient_not_necessary_condition_l208_208246

theorem sufficient_not_necessary_condition (x : ℝ) : (x^2 - 2 * x < 0) → (|x - 1| < 2) ∧ ¬( (|x - 1| < 2) → (x^2 - 2 * x < 0)) :=
by sorry

end sufficient_not_necessary_condition_l208_208246


namespace largest_divisor_of_even_diff_squares_l208_208971

theorem largest_divisor_of_even_diff_squares (m n : ℤ) (h_m_even : ∃ k : ℤ, m = 2 * k) (h_n_even : ∃ k : ℤ, n = 2 * k) (h_n_lt_m : n < m) : 
  ∃ d : ℤ, d = 16 ∧ ∀ p : ℤ, (p ∣ (m^2 - n^2)) → p ≤ d :=
sorry

end largest_divisor_of_even_diff_squares_l208_208971


namespace exists_consecutive_nat_with_integer_quotient_l208_208056

theorem exists_consecutive_nat_with_integer_quotient :
  ∃ n : ℕ, (n + 1) / n = 2 :=
by
  sorry

end exists_consecutive_nat_with_integer_quotient_l208_208056


namespace _l208_208041

noncomputable theorem meagre_sets_cardinality {n : ℕ} (hn : n ≥ 2) (S : Finset (Fin n × Fin n)) (hS : ∀ i, ∃ j, (i, j) ∈ S ∧ ∀ j, ∃ i, (i, j) ∈ S) :
  ∃ m n_permutations, m = n ∧ n_permutations = nat.factorial n := by
sorrry

noncomputable theorem fat_sets_cardinality {n : ℕ} (hn : n ≥ 2) (S : Finset (Fin n × Fin n)) (hS : ∀ i, ∃ j, (i, j) ∈ S ∧ ∀ j, ∃ i, (i, j) ∈ S) :
  ∃ M n_squared, M = 2 * n - 2 ∧ n_squared = n ^ 2 := by
sorrry

end _l208_208041


namespace solve_for_x_l208_208287

theorem solve_for_x (x : ℝ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -(2 / 11) :=
by
  sorry

end solve_for_x_l208_208287


namespace solve_inequality_l208_208128

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l208_208128


namespace nonagon_diagonals_l208_208493

def convex_nonagon_diagonals : Prop :=
∀ (n : ℕ), n = 9 → (n * (n - 3)) / 2 = 27

theorem nonagon_diagonals : convex_nonagon_diagonals :=
by {
  sorry,
}

end nonagon_diagonals_l208_208493


namespace evaluation_of_expression_l208_208221

theorem evaluation_of_expression: 
  (3^10 + 3^7) / (3^10 - 3^7) = 14 / 13 := 
  sorry

end evaluation_of_expression_l208_208221


namespace f_continuous_on_interval_f_not_bounded_variation_l208_208258

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x * Real.sin (1 / x)

theorem f_continuous_on_interval : ContinuousOn f (Set.Icc 0 1) :=
sorry

theorem f_not_bounded_variation : ¬ BoundedVariationOn f (Set.Icc 0 1) :=
sorry

end f_continuous_on_interval_f_not_bounded_variation_l208_208258


namespace compute_ab_l208_208179

namespace MathProof

variable {a b : ℝ}

theorem compute_ab (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := 
by
  sorry

end MathProof

end compute_ab_l208_208179


namespace number_triangle_value_of_n_l208_208810

theorem number_triangle_value_of_n:
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = 2022 ∧ (∃ n : ℕ, n > 0 ∧ n^2 ∣ 2022 ∧ n = 1) :=
by sorry

end number_triangle_value_of_n_l208_208810


namespace c_geq_one_l208_208399

variable {α : Type*} [LinearOrderedField α]

theorem c_geq_one
  (a : ℕ → α)
  (c : α)
  (h1 : ∀ i : ℕ, 0 < i → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, 0 < i → 0 < j → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 :=
sorry

end c_geq_one_l208_208399


namespace solve_for_a_l208_208638

-- Define the sets M and N as given in the problem
def M : Set ℝ := {x : ℝ | x^2 + 6 * x - 16 = 0}
def N (a : ℝ) : Set ℝ := {x : ℝ | x * a - 3 = 0}

-- Define the proof statement
theorem solve_for_a (a : ℝ) : (N a ⊆ M) ↔ (a = 0 ∨ a = 3/2 ∨ a = -3/8) :=
by
  -- The proof would go here
  sorry

end solve_for_a_l208_208638


namespace calculate_jessie_points_l208_208844

theorem calculate_jessie_points (total_points : ℕ) (some_players_points : ℕ) (players : ℕ) :
  total_points = 311 →
  some_players_points = 188 →
  players = 3 →
  (total_points - some_players_points) / players = 41 :=
by
  intros
  sorry

end calculate_jessie_points_l208_208844


namespace average_rate_first_half_80_l208_208661

theorem average_rate_first_half_80
    (total_distance : ℝ)
    (average_rate_trip : ℝ)
    (distance_first_half : ℝ)
    (time_first_half : ℝ)
    (time_second_half : ℝ)
    (time_total : ℝ)
    (R : ℝ)
    (H1 : total_distance = 640)
    (H2 : average_rate_trip = 40)
    (H3 : distance_first_half = total_distance / 2)
    (H4 : time_first_half = distance_first_half / R)
    (H5 : time_second_half = 3 * time_first_half)
    (H6 : time_total = time_first_half + time_second_half)
    (H7 : average_rate_trip = total_distance / time_total) :
    R = 80 := 
by 
  -- Given conditions
  sorry

end average_rate_first_half_80_l208_208661


namespace prob_all_green_is_1_div_30_l208_208648

-- Definitions as per the conditions
def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples  : ℕ := 4

def choose (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.descFactorial n k / (Nat.factorial k) else 0

-- Question part: probability calculation
def prob_all_green (n total green k : ℕ) (h_total : total = 10) (h_green : green = 4) (h_k : k = 3) : ℚ :=
  (choose green k) / (choose total k)

-- Statement to be proved
theorem prob_all_green_is_1_div_30 : (prob_all_green 3 total_apples green_apples 3 rfl rfl rfl) = 1/30 :=
  by sorry

end prob_all_green_is_1_div_30_l208_208648


namespace farm_field_area_l208_208500

variable (A D : ℕ)

theorem farm_field_area
  (h1 : 160 * D = A)
  (h2 : 85 * (D + 2) + 40 = A) :
  A = 480 :=
by
  sorry

end farm_field_area_l208_208500


namespace solve_fraction_eq_for_x_l208_208125

theorem solve_fraction_eq_for_x (x : ℝ) (hx : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end solve_fraction_eq_for_x_l208_208125


namespace two_p_plus_q_l208_208634

variable {p q : ℚ}  -- Variables are rationals

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by sorry

end two_p_plus_q_l208_208634


namespace multiple_time_second_artifact_is_three_l208_208814

-- Define the conditions as Lean definitions
def months_in_year : ℕ := 12
def total_time_both_artifacts_years : ℕ := 10
def total_time_first_artifact_months : ℕ := 6 + 24

-- Convert total time of both artifacts from years to months
def total_time_both_artifacts_months : ℕ := total_time_both_artifacts_years * months_in_year

-- Define the time for the second artifact
def time_second_artifact_months : ℕ :=
  total_time_both_artifacts_months - total_time_first_artifact_months

-- Define the sought multiple
def multiple_second_first : ℕ :=
  time_second_artifact_months / total_time_first_artifact_months

-- The theorem stating the required proof
theorem multiple_time_second_artifact_is_three :
  multiple_second_first = 3 :=
by
  sorry

end multiple_time_second_artifact_is_three_l208_208814


namespace geometric_sequence_sums_l208_208779

open Real

theorem geometric_sequence_sums (S T R : ℝ)
  (h1 : ∃ a r, S = a * (1 + r))
  (h2 : ∃ a r, T = a * (1 + r + r^2 + r^3))
  (h3 : ∃ a r, R = a * (1 + r + r^2 + r^3 + r^4 + r^5)) :
  S^2 + T^2 = S * (T + R) :=
by
  sorry

end geometric_sequence_sums_l208_208779


namespace repeatingDecimal_to_frac_l208_208759

noncomputable def repeatingDecimalToFrac : ℚ :=
  3 + 3 * (2 / 99 : ℚ)

theorem repeatingDecimal_to_frac :
  repeatingDecimalToFrac = 101 / 33 :=
by {
  sorry
}

end repeatingDecimal_to_frac_l208_208759


namespace pressure_on_trapezoidal_dam_l208_208914

noncomputable def water_pressure_on_trapezoidal_dam (ρ g h a b : ℝ) : ℝ :=
  ρ * g * (h^2) * (2 * a + b) / 6

theorem pressure_on_trapezoidal_dam
  (ρ g h a b : ℝ) : water_pressure_on_trapezoidal_dam ρ g h a b = ρ * g * (h^2) * (2 * a + b) / 6 := by
  sorry

end pressure_on_trapezoidal_dam_l208_208914


namespace nonagon_diagonals_count_l208_208497

theorem nonagon_diagonals_count (n : ℕ) (h1 : n = 9) : 
  let diagonals_per_vertex := n - 3 in
  let naive_count := n * diagonals_per_vertex in
  let distinct_diagonals := naive_count / 2 in
  distinct_diagonals = 27 :=
by
  sorry

end nonagon_diagonals_count_l208_208497


namespace terminating_decimals_nat_l208_208606

theorem terminating_decimals_nat (n : ℕ) (h1 : ∃ a b : ℕ, n = 2^a * 5^b)
  (h2 : ∃ c d : ℕ, n + 1 = 2^c * 5^d) : n = 1 ∨ n = 4 :=
by
  sorry

end terminating_decimals_nat_l208_208606


namespace distance_between_foci_of_hyperbola_l208_208061

theorem distance_between_foci_of_hyperbola (a b c : ℝ) : (x^2 - y^2 = 4) → (a = 2) → (b = 0) → (c = Real.sqrt (4 + 0)) → 
    dist (2, 0) (-2, 0) = 4 :=
by
  sorry

end distance_between_foci_of_hyperbola_l208_208061


namespace max_consecutive_integers_sum_lt_1000_l208_208008

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l208_208008


namespace mirror_area_correct_l208_208663

-- Given conditions
def outer_length : ℕ := 80
def outer_width : ℕ := 60
def frame_width : ℕ := 10

-- Deriving the dimensions of the mirror
def mirror_length : ℕ := outer_length - 2 * frame_width
def mirror_width : ℕ := outer_width - 2 * frame_width

-- Statement: Prove that the area of the mirror is 2400 cm^2
theorem mirror_area_correct : mirror_length * mirror_width = 2400 := by
  -- Proof should go here
  sorry

end mirror_area_correct_l208_208663


namespace james_total_earnings_l208_208384

-- Assume the necessary info for January, February, and March earnings
-- Definitions given as conditions in a)
def January_earnings : ℝ := 4000

def February_earnings : ℝ := January_earnings * 1.5 * 1.2

def March_earnings : ℝ := February_earnings * 0.8

-- The total earnings to be calculated
def Total_earnings : ℝ := January_earnings + February_earnings + March_earnings

-- Prove the total earnings is $16960
theorem james_total_earnings : Total_earnings = 16960 := by
  sorry

end james_total_earnings_l208_208384


namespace job_positions_growth_rate_l208_208036

theorem job_positions_growth_rate (x : ℝ) :
  1501 * (1 + x) ^ 2 = 1815 := sorry

end job_positions_growth_rate_l208_208036


namespace compare_two_and_neg_three_l208_208599

theorem compare_two_and_neg_three (h1 : 2 > 0) (h2 : -3 < 0) : 2 > -3 :=
by
  sorry

end compare_two_and_neg_three_l208_208599


namespace find_f_neg1_l208_208361

theorem find_f_neg1 {f : ℝ → ℝ} (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 := 
by 
  -- skipping the proof: 
  sorry

end find_f_neg1_l208_208361


namespace value_of_expression_l208_208013

theorem value_of_expression : (165^2 - 153^2) / 12 = 318 := by
  sorry

end value_of_expression_l208_208013


namespace perimeter_ratio_of_squares_l208_208141

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l208_208141


namespace perimeter_of_original_rectangle_l208_208453

-- Define the rectangle's dimensions based on the given condition
def length_of_rectangle := 2 * 8 -- because it forms two squares of side 8 cm each
def width_of_rectangle := 8 -- side of the squares

-- Using the formula for the perimeter of a rectangle: P = 2 * (length + width)
def perimeter_of_rectangle := 2 * (length_of_rectangle + width_of_rectangle)

-- The statement we need to prove
theorem perimeter_of_original_rectangle : perimeter_of_rectangle = 48 := by
  sorry

end perimeter_of_original_rectangle_l208_208453


namespace positional_relationship_l208_208096

theorem positional_relationship (r PO QO : ℝ) (h_r : r = 6) (h_PO : PO = 4) (h_QO : QO = 6) :
  (PO < r) ∧ (QO = r) :=
by
  sorry

end positional_relationship_l208_208096


namespace triangle_ratio_l208_208515

noncomputable def triangle_problem (BC AC : ℝ) (angleC : ℝ) : ℝ :=
  let CD := AC / 2
  let BD := BC - CD
  let HD := BD / 2
  let AD := (3^(1/2)) * CD
  let AH := AD - HD
  (AH / HD)

theorem triangle_ratio (BC AC : ℝ) (angleC : ℝ) (h1 : BC = 6) (h2 : AC = 3 * Real.sqrt 3) (h3 : angleC = Real.pi / 6) :
  triangle_problem BC AC angleC = -2 - Real.sqrt 3 :=
by
  sorry  

end triangle_ratio_l208_208515


namespace effective_percentage_change_l208_208193

def original_price (P : ℝ) : ℝ := P
def annual_sale_discount (P : ℝ) : ℝ := 0.70 * P
def clearance_event_discount (P : ℝ) : ℝ := 0.80 * (annual_sale_discount P)
def sales_tax (P : ℝ) : ℝ := 1.10 * (clearance_event_discount P)

theorem effective_percentage_change (P : ℝ) :
  (sales_tax P) = 0.616 * P := by
  sorry

end effective_percentage_change_l208_208193


namespace tan_of_angle_l208_208932

noncomputable def tan_val (α : ℝ) : ℝ := Real.tan α

theorem tan_of_angle (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.cos (2 * α) = -3 / 5) :
  tan_val α = -2 := by
  sorry

end tan_of_angle_l208_208932


namespace sum_of_first_five_terms_sequence_l208_208557

-- Definitions derived from conditions
def seventh_term : ℤ := 4
def eighth_term : ℤ := 10
def ninth_term : ℤ := 16

-- The main theorem statement
theorem sum_of_first_five_terms_sequence : 
  ∃ (a d : ℤ), 
    a + 6 * d = seventh_term ∧
    a + 7 * d = eighth_term ∧
    a + 8 * d = ninth_term ∧
    (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = -100) :=
by
  sorry

end sum_of_first_five_terms_sequence_l208_208557


namespace find_number_l208_208019

theorem find_number (N : ℝ) (h : 0.6 * (3 / 5) * N = 36) : N = 100 :=
by sorry

end find_number_l208_208019


namespace condition_2_3_implies_f_x1_greater_f_x2_l208_208072

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * Real.cos x

theorem condition_2_3_implies_f_x1_greater_f_x2 
(x1 x2 : ℝ) (h1 : -2 * Real.pi / 3 ≤ x1 ∧ x1 ≤ 2 * Real.pi / 3) 
(h2 : -2 * Real.pi / 3 ≤ x2 ∧ x2 ≤ 2 * Real.pi / 3) 
(hx1_sq_gt_x2_sq : x1^2 > x2^2) (hx1_gt_abs_x2 : x1 > |x2|) : 
  f x1 > f x2 := 
sorry

end condition_2_3_implies_f_x1_greater_f_x2_l208_208072


namespace stan_water_intake_l208_208289

-- Define the constants and parameters given in the conditions
def words_per_minute : ℕ := 50
def pages : ℕ := 5
def words_per_page : ℕ := 400
def water_per_hour : ℚ := 15  -- use rational numbers for precise division

-- Define the derived quantities from the conditions
def total_words : ℕ := pages * words_per_page
def total_minutes : ℕ := total_words / words_per_minute
def water_per_minute : ℚ := water_per_hour / 60

-- State the theorem
theorem stan_water_intake : 10 = total_minutes * water_per_minute := by
  sorry

end stan_water_intake_l208_208289


namespace possible_ages_l208_208884

-- Define the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 3}

-- Condition: The age must start with "211"
def starting_sequence : List ℕ := [2, 1, 1]

-- Calculate the count of possible ages
def count_ages : ℕ :=
  let remaining_digits := [2, 2, 1, 3]
  let total_permutations := Nat.factorial 4
  let repetitions := Nat.factorial 2
  total_permutations / repetitions

theorem possible_ages : count_ages = 12 := by
  -- Proof should go here but it's omitted according to instructions.
  sorry

end possible_ages_l208_208884


namespace abs_inequality_solution_l208_208127

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l208_208127


namespace initial_markup_percentage_l208_208324

theorem initial_markup_percentage (C : ℝ) (M : ℝ) 
  (h1 : ∀ S_1 : ℝ, S_1 = C * (1 + M))
  (h2 : ∀ S_2 : ℝ, S_2 = C * (1 + M) * 1.25)
  (h3 : ∀ S_3 : ℝ, S_3 = C * (1 + M) * 1.25 * 0.94)
  (h4 : ∀ S_3 : ℝ, S_3 = C * 1.41) : 
  M = 0.2 :=
by
  sorry

end initial_markup_percentage_l208_208324


namespace value_of_a_plus_b_l208_208244

variable {F : Type} [Field F]

theorem value_of_a_plus_b (a b : F) (h1 : ∀ x, x ≠ 0 → a + b / x = 2 ↔ x = -2)
                                      (h2 : ∀ x, x ≠ 0 → a + b / x = 6 ↔ x = -6) :
  a + b = 20 :=
sorry

end value_of_a_plus_b_l208_208244


namespace fraction_one_two_three_sum_l208_208801

def fraction_one_bedroom : ℝ := 0.12
def fraction_two_bedroom : ℝ := 0.26
def fraction_three_bedroom : ℝ := 0.38
def fraction_four_bedroom : ℝ := 0.24

theorem fraction_one_two_three_sum :
  fraction_one_bedroom + fraction_two_bedroom + fraction_three_bedroom = 0.76 :=
by
  sorry

end fraction_one_two_three_sum_l208_208801


namespace number_of_workers_l208_208430

theorem number_of_workers (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 350000) : W = 1000 :=
sorry

end number_of_workers_l208_208430


namespace marina_more_fudge_l208_208260

theorem marina_more_fudge (h1 : 4.5 * 16 = 72)
                          (h2 : 4 * 16 - 6 = 58) :
                          72 - 58 = 14 := by
  sorry

end marina_more_fudge_l208_208260


namespace roden_gold_fish_count_l208_208982

theorem roden_gold_fish_count
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (gold_fish : ℕ)
  (h1 : total_fish = 22)
  (h2 : blue_fish = 7)
  (h3 : total_fish = blue_fish + gold_fish) : gold_fish = 15 :=
by
  sorry

end roden_gold_fish_count_l208_208982


namespace find_parenthesis_value_l208_208358

theorem find_parenthesis_value (x : ℝ) (h : x * (-2/3) = 2) : x = -3 :=
by
  sorry

end find_parenthesis_value_l208_208358


namespace product_of_x_y_l208_208805

variable (x y : ℝ)

-- Condition: EF = GH
def EF_eq_GH := (x^2 + 2 * x - 8 = 45)

-- Condition: FG = EH
def FG_eq_EH := (y^2 + 8 * y + 16 = 36)

-- Condition: y > 0
def y_pos := (y > 0)

theorem product_of_x_y : EF_eq_GH x ∧ FG_eq_EH y ∧ y_pos y → 
  x * y = -2 + 6 * Real.sqrt 6 :=
sorry

end product_of_x_y_l208_208805


namespace shorter_leg_in_right_triangle_l208_208371

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l208_208371


namespace compute_xy_l208_208584

theorem compute_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x ^ (Real.sqrt y) = 27) (h2 : (Real.sqrt x) ^ y = 9) :
  x * y = 12 * Real.sqrt 3 :=
sorry

end compute_xy_l208_208584


namespace line_through_point_equal_intercepts_locus_equidistant_lines_l208_208880

theorem line_through_point_equal_intercepts (x y : ℝ) (hx : x = 1) (hy : y = 3) :
  (∃ k : ℝ, y = k * x ∧ k = 3) ∨ (∃ a : ℝ, x + y = a ∧ a = 4) :=
sorry

theorem locus_equidistant_lines (x y : ℝ) :
  ∀ (a b : ℝ), (2 * x + 3 * y - a = 0) ∧ (4 * x + 6 * y + b = 0) →
  ∀ b : ℝ, |b + 10| = |b - 8| → b = -9 → 
  4 * x + 6 * y - 9 = 0 :=
sorry

end line_through_point_equal_intercepts_locus_equidistant_lines_l208_208880


namespace find_k_and_b_l208_208088

theorem find_k_and_b (k b : ℝ) :
  (∃ P Q : ℝ × ℝ, P ≠ Q ∧
  ((P.1 - 1)^2 + P.2^2 = 1) ∧ 
  ((Q.1 - 1)^2 + Q.2^2 = 1) ∧ 
  (P.2 = k * P.1) ∧ 
  (Q.2 = k * Q.1) ∧ 
  (P.1 - P.2 + b = 0) ∧ 
  (Q.1 - Q.2 + b = 0) ∧ 
  ((P.1 + Q.1) / 2 = (P.2 + Q.2) / 2)) →
  k = -1 ∧ b = -1 :=
sorry

end find_k_and_b_l208_208088


namespace janice_weekly_earnings_l208_208968

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end janice_weekly_earnings_l208_208968


namespace manufacturer_l208_208450

-- Let x be the manufacturer's suggested retail price
variable (x : ℝ)

-- Regular discount range from 10% to 30%
def regular_discount (d : ℝ) : Prop := d >= 0.10 ∧ d <= 0.30

-- Additional discount during sale 
def additional_discount : ℝ := 0.20

-- The final discounted price is $16.80
def final_price (x : ℝ) : Prop := ∃ d, regular_discount d ∧ 0.80 * ((1 - d) * x) = 16.80

theorem manufacturer's_suggested_retail_price :
  final_price x → x = 30 := by
  sorry

end manufacturer_l208_208450


namespace count_perfect_cubes_between_500_and_2000_l208_208939

theorem count_perfect_cubes_between_500_and_2000 : ∃ count : ℕ, count = 5 ∧ (∀ n, 500 < n^3 ∧ n^3 < 2000 → (8 ≤ n ∧ n ≤ 12)) :=
by
  existsi 5
  split
  {
    sorry,  -- Proof that count = 5
    sorry,  -- Proof that for any n, if 500 < n^3 and n^3 < 2000 then 8 <= n <= 12
  }

end count_perfect_cubes_between_500_and_2000_l208_208939


namespace y_percentage_of_8950_l208_208799

noncomputable def x := 0.18 * 4750
noncomputable def y := 1.30 * x
theorem y_percentage_of_8950 : (y / 8950) * 100 = 12.42 := 
by 
  -- proof steps are omitted
  sorry

end y_percentage_of_8950_l208_208799


namespace not_possible_to_tile_l208_208748

theorem not_possible_to_tile 
    (m n : ℕ) (a b : ℕ)
    (h_m : m = 2018)
    (h_n : n = 2020)
    (h_a : a = 5)
    (h_b : b = 8) :
    ¬ ∃ k : ℕ, k * (a * b) = m * n := by
sorry

end not_possible_to_tile_l208_208748


namespace no_positive_integer_solutions_l208_208830

theorem no_positive_integer_solutions (x y : ℕ) (h : x > 0 ∧ y > 0) : x^2 + (x+1)^2 ≠ y^4 + (y+1)^4 :=
by
  intro h1
  sorry

end no_positive_integer_solutions_l208_208830


namespace simplify_fraction_l208_208282

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l208_208282


namespace solve_quartic_equation_l208_208920

theorem solve_quartic_equation :
  (∃ x : ℝ, x > 0 ∧ 
    (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 60 * x - 12) * (x ^ 2 + 30 * x + 6) ∧ 
    ∃ y1 y2 : ℝ, y1 + y2 = 60 ∧ (x^2 - 60 * x - 12 = 0)) → 
    x = 30 + Real.sqrt 912 :=
sorry

end solve_quartic_equation_l208_208920


namespace max_sum_of_S_l208_208299

open Set

def disjoint_subset_sum_condition (S : Set ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ⊆ S → B ⊆ S → Disjoint A B → (A.sum id ≠ B.sum id)

theorem max_sum_of_S :
  ∃ S : Set ℕ, (∀ x ∈ S, x ≤ 15) ∧ disjoint_subset_sum_condition S ∧ S.sum id = 61 :=
  sorry

end max_sum_of_S_l208_208299


namespace estimate_total_observations_in_interval_l208_208641

def total_observations : ℕ := 1000
def sample_size : ℕ := 50
def frequency_in_sample : ℝ := 0.12

theorem estimate_total_observations_in_interval : 
  frequency_in_sample * (total_observations : ℝ) = 120 :=
by
  -- conditions defined above
  -- use given frequency to estimate the total observations in the interval
  -- actual proof omitted
  sorry

end estimate_total_observations_in_interval_l208_208641


namespace largest_sum_l208_208594

theorem largest_sum :
  let s1 := (1 : ℚ) / 3 + (1 : ℚ) / 4
  let s2 := (1 : ℚ) / 3 + (1 : ℚ) / 5
  let s3 := (1 : ℚ) / 3 + (1 : ℚ) / 2
  let s4 := (1 : ℚ) / 3 + (1 : ℚ) / 9
  let s5 := (1 : ℚ) / 3 + (1 : ℚ) / 6
  in max s1 (max s2 (max s3 (max s4 s5))) = (5 : ℚ) / 6 := by
  sorry

end largest_sum_l208_208594


namespace toys_ratio_l208_208746

theorem toys_ratio (k A M T : ℕ) (h1 : M = 6) (h2 : A = k * M) (h3 : A = T - 2) (h4 : A + M + T = 56):
  A / M = 4 :=
by
  sorry

end toys_ratio_l208_208746


namespace students_got_off_the_bus_l208_208176

theorem students_got_off_the_bus
    (original_students : ℕ)
    (students_left : ℕ)
    (h_original : original_students = 10)
    (h_left : students_left = 7) :
    original_students - students_left = 3 :=
by {
  sorry
}

end students_got_off_the_bus_l208_208176


namespace trigonometric_identity_l208_208487

theorem trigonometric_identity (alpha : ℝ) (h : Real.tan alpha = 2 * Real.tan (π / 5)) :
  (Real.cos (alpha - 3 * π / 10) / Real.sin (alpha - π / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l208_208487


namespace geometric_sequence_sum_ratio_l208_208397

noncomputable def a (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * q^n

-- Sum of the first 'n' terms of a geometric sequence
noncomputable def S (n : ℕ) (a1 q : ℝ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a1 q : ℝ) 
  (h : 8 * (a 11 a1 q) = (a 14 a1 q)) :
  (S 4 a1 q) / (S 2 a1 q) = 5 :=
by
  sorry

end geometric_sequence_sum_ratio_l208_208397


namespace circumcircles_common_point_l208_208776

noncomputable theory

variables {A B C D E F O : Type} [euclidean_space ℝ A B C D E F O]

def lines_intersect_four_triangles (A B C D E F : euclidean_space ℝ A B C D E F) : Prop :=
∃ (AE AF ED FB : set (euclidean_space ℝ A B C D E F)),
  -- Points of intersection
  AE ∩ AF = {A} ∧
  ED ∩ FB = {D} ∧
  -- Formation of triangles
  triangle A B F ∧ triangle A E D ∧ triangle B C E ∧ triangle D C F
  
def circumcircles_concur (A B C D E F O : euclidean_space ℝ A B C D E F) : Prop :=
∃ (O : euclidean_space ℝ A B C D E F),
  -- Circumcircles concurrence definitions
  circumcircle (triangle A B F) O ∧ circumcircle (triangle A E D) O ∧
  circumcircle (triangle B C E) O ∧ circumcircle (triangle D C F) O

theorem circumcircles_common_point
  (A B C D E F : euclidean_space ℝ A B C D E F) :
  lines_intersect_four_triangles A B C D E F → circumcircles_concur A B C D E F :=
begin
  intros h,
  sorry, -- The proof goes here
end

end circumcircles_common_point_l208_208776


namespace distance_to_school_l208_208386

variables (d : ℝ)
def jog_rate := 5
def bus_rate := 30
def total_time := 1 

theorem distance_to_school :
  (d / jog_rate) + (d / bus_rate) = total_time ↔ d = 30 / 7 :=
by
  sorry

end distance_to_school_l208_208386


namespace sum_of_variables_l208_208616

theorem sum_of_variables (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2 * x + 4 * y - 6 * z + 14 = 0) : x + y + z = 2 :=
sorry

end sum_of_variables_l208_208616


namespace carbon_paper_count_l208_208038

theorem carbon_paper_count (x : ℕ) (sheets : ℕ) (copies : ℕ) (h1 : sheets = 3) (h2 : copies = 2) :
  x = 1 :=
sorry

end carbon_paper_count_l208_208038


namespace sum_of_reciprocals_l208_208422

variable (x y : ℝ)

theorem sum_of_reciprocals (h1 : x + y = 10) (h2 : x * y = 20) : 1 / x + 1 / y = 1 / 2 :=
by
  sorry

end sum_of_reciprocals_l208_208422


namespace extremum_f_at_one_range_a_monotonic_f_l208_208116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.log x + Real.log x / x

theorem extremum_f_at_one (a : ℝ) (h : a = -1 / 2) : f a 1 = 0 :=
by
  -- Proof for extremum of f at x=1 when a=-1/2
  sorry

theorem range_a_monotonic_f (a : ℝ) : (∀ x > 0, (2 * a * x + 1 - Real.log x) / x^2 ≥ 0) ↔ (a ≥ 1 / (2 * Real.exp 2)) :=
by
  -- Proof defining the range of a for monotonic increase of f on its domain
  sorry

end extremum_f_at_one_range_a_monotonic_f_l208_208116


namespace option_C_is_correct_l208_208308

theorem option_C_is_correct :
  (-3 - (-2) ≠ -5) ∧
  (-|(-1:ℝ)/3| + 1 ≠ 4/3) ∧
  (4 - 4 / 2 = 2) ∧
  (3^2 / 6 * (1/6) ≠ 9) :=
by
  -- Proof omitted
  sorry

end option_C_is_correct_l208_208308


namespace find_a_l208_208610

theorem find_a (a : ℝ) :
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 :=
sorry

end find_a_l208_208610


namespace problem1_problem2_problem3_l208_208620

-- Definition of the sequence
def a (n : ℕ) (k : ℚ) : ℚ := (k * n - 3) / (n - 3 / 2)

-- The first condition proof problem
theorem problem1 (k : ℚ) : (∀ n : ℕ, a n k = (a (n + 1) k + a (n - 1) k) / 2) → k = 2 :=
sorry

-- The second condition proof problem
theorem problem2 (k : ℚ) : 
  k ≠ 2 → 
  (if k > 2 then (a 1 k < k ∧ a 2 k = max (a 1 k) (a 2 k))
   else if k < 2 then (a 2 k < k ∧ a 1 k = max (a 1 k) (a 2 k))
   else False) :=
sorry

-- The third condition proof problem
theorem problem3 (k : ℚ) : 
  (∀ n : ℕ, n > 0 → a n k > (k * 2^n + (-1)^n) / 2^n) → 
  101 / 48 < k ∧ k < 13 / 6 :=
sorry

end problem1_problem2_problem3_l208_208620


namespace count_eligible_three_digit_numbers_l208_208628

def is_eligible_digit (d : Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem count_eligible_three_digit_numbers : 
  (∃ n : Nat, 100 ≤ n ∧ n < 1000 ∧
  (∀ d : Nat, d ∈ [n / 100, (n / 10) % 10, n % 10] → is_eligible_digit d)) →
  ∃ count : Nat, count = 343 :=
by
  sorry

end count_eligible_three_digit_numbers_l208_208628


namespace geese_percentage_non_ducks_l208_208819

theorem geese_percentage_non_ducks :
  let total_birds := 100
  let geese := 0.20 * total_birds
  let swans := 0.30 * total_birds
  let herons := 0.15 * total_birds
  let ducks := 0.25 * total_birds
  let pigeons := 0.10 * total_birds
  let non_duck_birds := total_birds - ducks
  (geese / non_duck_birds) * 100 = 27 := 
by
  sorry

end geese_percentage_non_ducks_l208_208819


namespace sequences_no_two_heads_follow_each_other_l208_208033

open Nat

-- Definitions only based on conditions provided
def valid_sequences (n : ℕ) (f : ℕ) : ℕ :=
  binomial (n - f + 1) f

def count_valid_sequences (n : ℕ) : ℕ :=
  (range (n.div 2 + 1)).sum (λ f, valid_sequences n f)

theorem sequences_no_two_heads_follow_each_other (n : ℕ) (h : n = 12) : count_valid_sequences n = 377 :=
by
  rw h
  -- no proof needed
  sorry

end sequences_no_two_heads_follow_each_other_l208_208033


namespace square_perimeter_ratio_l208_208151

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l208_208151


namespace crayons_problem_l208_208195

theorem crayons_problem 
  (total_crayons : ℕ)
  (red_crayons : ℕ)
  (blue_crayons : ℕ)
  (green_crayons : ℕ)
  (pink_crayons : ℕ)
  (h1 : total_crayons = 24)
  (h2 : red_crayons = 8)
  (h3 : blue_crayons = 6)
  (h4 : green_crayons = 2 / 3 * blue_crayons)
  (h5 : pink_crayons = total_crayons - red_crayons - blue_crayons - green_crayons) :
  pink_crayons = 6 :=
by
  sorry

end crayons_problem_l208_208195


namespace square_perimeter_ratio_l208_208150

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l208_208150


namespace ratio_daves_bench_to_weight_l208_208753

variables (wD bM bD bC : ℝ)

def daves_weight := wD = 175
def marks_bench_press := bM = 55
def marks_comparison_to_craig := bM = bC - 50
def craigs_comparison_to_dave := bC = 0.20 * bD

theorem ratio_daves_bench_to_weight
  (h1 : daves_weight wD)
  (h2 : marks_bench_press bM)
  (h3 : marks_comparison_to_craig bM bC)
  (h4 : craigs_comparison_to_dave bC bD) :
  (bD / wD) = 3 :=
by
  rw [daves_weight] at h1
  rw [marks_bench_press] at h2
  rw [marks_comparison_to_craig] at h3
  rw [craigs_comparison_to_dave] at h4
  -- Now we have:
  -- 1. wD = 175
  -- 2. bM = 55
  -- 3. bM = bC - 50
  -- 4. bC = 0.20 * bD
  -- We proceed to solve:
  sorry

end ratio_daves_bench_to_weight_l208_208753


namespace sum_every_second_term_is_1010_l208_208040

def sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_every_second_term_is_1010 :
  ∃ (x1 : ℤ) (d : ℤ) (S : ℤ), 
  (sequence_sum 2020 x1 d = 6060) ∧
  (d = 2) ∧
  (S = (1010 : ℤ)) ∧ 
  (2 * S + 4040 = 6060) :=
  sorry

end sum_every_second_term_is_1010_l208_208040


namespace max_consecutive_integers_lt_1000_l208_208000

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l208_208000


namespace positive_integer_solution_of_inequality_l208_208679

theorem positive_integer_solution_of_inequality :
  {x : ℕ // 0 < x ∧ x < 2} → x = 1 :=
by
  sorry

end positive_integer_solution_of_inequality_l208_208679


namespace arithmetic_progression_integers_l208_208902

theorem arithmetic_progression_integers 
  (d : ℤ) (a : ℤ) (h_d_pos : d > 0)
  (h_progression : ∀ i j : ℤ, i ≠ j → ∃ k : ℤ, a * (a + i * d) = a + k * d)
  : ∀ n : ℤ, ∃ m : ℤ, a + n * d = m :=
by
  sorry

end arithmetic_progression_integers_l208_208902


namespace smallest_q_p_difference_l208_208112

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l208_208112


namespace augmented_matrix_correct_l208_208991

-- Define the system of linear equations as a pair of equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x + y = 1) ∧ (3 * x - 2 * y = 0)

-- Define what it means to be the correct augmented matrix for the system
def is_augmented_matrix (A : Matrix (Fin 2) (Fin 3) ℝ) : Prop :=
  A = ![
    ![2, 1, 1],
    ![3, -2, 0]
  ]

-- The theorem states that the augmented matrix of the given system of equations is the specified matrix
theorem augmented_matrix_correct :
  ∃ x y : ℝ, system_of_equations x y ∧ is_augmented_matrix ![
    ![2, 1, 1],
    ![3, -2, 0]
  ] :=
sorry

end augmented_matrix_correct_l208_208991


namespace board_game_cost_correct_l208_208910

-- Definitions
def jump_rope_cost : ℕ := 7
def ball_cost : ℕ := 4
def saved_money : ℕ := 6
def gift_money : ℕ := 13
def needed_money : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_money + gift_money

-- Total cost of all items
def total_cost : ℕ := total_money + needed_money

-- Combined cost of jump rope and ball
def combined_cost_jump_rope_ball : ℕ := jump_rope_cost + ball_cost

-- Cost of the board game
def board_game_cost : ℕ := total_cost - combined_cost_jump_rope_ball

-- Theorem to prove
theorem board_game_cost_correct : board_game_cost = 12 :=
by 
  -- Proof omitted
  sorry

end board_game_cost_correct_l208_208910


namespace interest_after_5_years_l208_208659

noncomputable def initial_amount : ℝ := 2000
noncomputable def interest_rate : ℝ := 0.08
noncomputable def duration : ℕ := 5
noncomputable def final_amount : ℝ := initial_amount * (1 + interest_rate) ^ duration
noncomputable def interest_earned : ℝ := final_amount - initial_amount

theorem interest_after_5_years : interest_earned = 938.66 := by
  sorry

end interest_after_5_years_l208_208659


namespace Hannah_total_spent_l208_208077

def rides_cost (total_money : ℝ) : ℝ :=
  0.35 * total_money

def games_cost (total_money : ℝ) : ℝ :=
  0.25 * total_money

def food_and_souvenirs_cost : ℝ :=
  7 + 4 + 5 + 6

def total_spent (total_money : ℝ) : ℝ :=
  rides_cost total_money + games_cost total_money + food_and_souvenirs_cost

theorem Hannah_total_spent (total_money : ℝ) (h : total_money = 80) :
  total_spent total_money = 70 :=
by
  rw [total_spent, h, rides_cost, games_cost]
  norm_num
  sorry

end Hannah_total_spent_l208_208077


namespace first_day_exceeds_200_l208_208365

def bacteria_count (n : ℕ) : ℕ := 4 * 3^n

def exceeds_200 (n : ℕ) : Prop := bacteria_count n > 200

theorem first_day_exceeds_200 : ∃ n, exceeds_200 n ∧ ∀ m < n, ¬ exceeds_200 m :=
by sorry

end first_day_exceeds_200_l208_208365


namespace expected_difference_zero_l208_208210

/-- Define the probabilities for rolling a prime number leading to unsweetened cereal and a composite
    number leading to sweetened cereal on an 8-sided die.
    - Primes (unsweetened): {2, 3, 5}
    - Composites (sweetened): {4, 6, 8}
    - No cereal: 7
    - Reroll: 1
--/

def prob_unsweetened := 3 / 7  -- probability of rolling 2, 3, or 5
def prob_sweetened := 3 / 7    -- probability of rolling 4, 6, or 8
def days_in_year := 365
def expected_days_unsweetened := prob_unsweetened * days_in_year
def expected_days_sweetened := prob_sweetened * days_in_year

theorem expected_difference_zero : expected_days_unsweetened - expected_days_sweetened = 0 :=
by
  sorry

end expected_difference_zero_l208_208210


namespace power_division_calculation_l208_208211

theorem power_division_calculation :
  ( ( 5^13 / 5^11 )^2 * 5^2 ) / 2^5 = 15625 / 32 :=
by
  sorry

end power_division_calculation_l208_208211


namespace inequality_sqrt_l208_208538

theorem inequality_sqrt (m n : ℕ) (h : m < n) : 
  (m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n)) :=
by
  sorry

end inequality_sqrt_l208_208538


namespace ray_climbing_stairs_l208_208721

theorem ray_climbing_stairs (n : ℕ) (h1 : n % 4 = 3) (h2 : n % 5 = 2) (h3 : 10 < n) : n = 27 :=
sorry

end ray_climbing_stairs_l208_208721


namespace entrants_total_l208_208177

theorem entrants_total (N : ℝ) (h1 : N > 800)
  (h2 : 0.35 * N = NumFemales)
  (h3 : 0.65 * N = NumMales)
  (h4 : NumMales - NumFemales = 252) :
  N = 840 := 
sorry

end entrants_total_l208_208177


namespace eval_expression_l208_208113

def square_avg (a b : ℚ) : ℚ := (a^2 + b^2) / 2
def custom_avg (a b c : ℚ) : ℚ := (a + b + 2 * c) / 3

theorem eval_expression : 
  custom_avg (custom_avg 2 (-1) 1) (square_avg 2 3) 1 = 19 / 6 :=
by
  sorry

end eval_expression_l208_208113


namespace lisa_flight_time_l208_208975

noncomputable def distance : ℝ := 519.5
noncomputable def speed : ℝ := 54.75
noncomputable def time : ℝ := 9.49

theorem lisa_flight_time : distance / speed = time :=
by
  sorry

end lisa_flight_time_l208_208975


namespace rectangle_area_increase_l208_208138

theorem rectangle_area_increase (L W : ℝ) (h1: L > 0) (h2: W > 0) :
   let original_area := L * W
   let new_length := 1.20 * L
   let new_width := 1.20 * W
   let new_area := new_length * new_width
   let percentage_increase := ((new_area - original_area) / original_area) * 100
   percentage_increase = 44 :=
by
  sorry

end rectangle_area_increase_l208_208138


namespace smallest_number_of_fruits_l208_208178

theorem smallest_number_of_fruits (N : ℕ) :
  (∃ x : ℕ, 3 * x + 1 = (8 * N - 56) / 27) → N = 79 :=
by sorry

end smallest_number_of_fruits_l208_208178


namespace perimeters_ratio_l208_208154

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l208_208154


namespace perimeter_ratio_of_squares_l208_208143

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l208_208143


namespace g_60_l208_208853

noncomputable def g : ℝ → ℝ :=
sorry

axiom g_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y

axiom g_45 : g 45 = 15

theorem g_60 : g 60 = 11.25 :=
by
  sorry

end g_60_l208_208853


namespace explain_education_policy_l208_208951

theorem explain_education_policy :
  ∃ (reason1 reason2 : String), reason1 ≠ reason2 ∧
    (reason1 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason2 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions")
    ∨
    (reason2 = "International Agreements: Favorable foreign credit terms or reciprocal educational benefits" ∧
     reason1 = "Addressing Demographic Changes: Attracting educated youth for future economic contributions") :=
by
  sorry

end explain_education_policy_l208_208951


namespace lcm_of_two_numbers_l208_208728
-- Importing the math library

-- Define constants and variables
variables (A B LCM HCF : ℕ)

-- Given conditions
def product_condition : Prop := A * B = 17820
def hcf_condition : Prop := HCF = 12
def lcm_condition : Prop := LCM = Nat.lcm A B

-- Theorem to prove
theorem lcm_of_two_numbers : product_condition A B ∧ hcf_condition HCF →
                              lcm_condition A B LCM →
                              LCM = 1485 := 
by
  sorry

end lcm_of_two_numbers_l208_208728


namespace sequence_inequality_l208_208400

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_condition : ∀ n m, a (n + m) ≤ a n + a m) :
  ∀ n m, n ≥ m → a n ≤ m * a 1 + ((n / m) - 1) * a m := by
  sorry

end sequence_inequality_l208_208400


namespace ants_no_collision_probability_l208_208836

noncomputable def probability_no_collision 
    (initial_ants : Finset Vertex)
    (graph : Vertex → Finset Vertex)
    (move_probability : Π v : Vertex, Finset (graph v) → Probability)
    (no_collision_prob : ℚ)
    (conditions : 
      -- Adjacent vertices conditions:
      (graph A = {B, C, D, E, F}) ∧ 
      (graph F = {A, B, C}) ∧
      (graph B = {A, C, D, E, F}) ∧
      (graph C = {A, B, D, E, F}) ∧
      (graph D = {A, B, C, E}) ∧
      (graph E = {A, B, C, D})) : Prop :=
    -- Only check initial setup and probability, assuming basic Lean/FOL non-constructive probability is used
    initial_ants = {A, B, C, D, E, F} ∧
    no_collision_prob = 240 / 15625 ∧ 
    true -- Placeholder for potential other prob. calculations within full proof framework

-- The goal is to show:
theorem ants_no_collision_probability : 
    probability_no_collision initial_ants graph move_probability (240 / 15625) 
    (begin 
      split, -- conditions for connections
      all_goals { try {assumption} },
    end) :=
    sorry

end ants_no_collision_probability_l208_208836


namespace triangle_classification_l208_208074

def is_obtuse_triangle (a b c : ℕ) : Prop :=
c^2 > a^2 + b^2 ∧ a < b ∧ b < c

def is_right_triangle (a b c : ℕ) : Prop :=
c^2 = a^2 + b^2 ∧ a < b ∧ b < c

def is_acute_triangle (a b c : ℕ) : Prop :=
c^2 < a^2 + b^2 ∧ a < b ∧ b < c

theorem triangle_classification :
    is_acute_triangle 10 12 14 ∧ 
    is_right_triangle 10 24 26 ∧ 
    is_obtuse_triangle 4 6 8 :=
by 
  sorry

end triangle_classification_l208_208074


namespace sandy_total_money_received_l208_208403

def sandy_saturday_half_dollars := 17
def sandy_sunday_half_dollars := 6
def half_dollar_value : ℝ := 0.50

theorem sandy_total_money_received :
  (sandy_saturday_half_dollars * half_dollar_value) +
  (sandy_sunday_half_dollars * half_dollar_value) = 11.50 :=
by
  sorry

end sandy_total_money_received_l208_208403


namespace trapezoid_height_l208_208854

theorem trapezoid_height (AD BC : ℝ) (AB CD : ℝ) (h₁ : AD = 25) (h₂ : BC = 4) (h₃ : AB = 20) (h₄ : CD = 13) : ∃ h : ℝ, h = 12 :=
by
  -- Definitions
  let AD := 25
  let BC := 4
  let AB := 20
  let CD := 13
  
  sorry

end trapezoid_height_l208_208854


namespace revenue_from_full_price_tickets_l208_208198

noncomputable def full_price_ticket_revenue (f h p : ℕ) : ℕ := f * p

theorem revenue_from_full_price_tickets (f h p : ℕ) (total_tickets total_revenue : ℕ) 
  (tickets_eq : f + h = total_tickets)
  (revenue_eq : f * p + h * (p / 2) = total_revenue) 
  (total_tickets_value : total_tickets = 180)
  (total_revenue_value : total_revenue = 2652) :
  full_price_ticket_revenue f h p = 984 :=
by {
  sorry
}

end revenue_from_full_price_tickets_l208_208198


namespace eve_distance_ran_more_l208_208222

variable (ran walked : ℝ)

def eve_distance_difference (ran walked : ℝ) : ℝ :=
  ran - walked

theorem eve_distance_ran_more :
  eve_distance_difference 0.7 0.6 = 0.1 :=
by
  sorry

end eve_distance_ran_more_l208_208222


namespace obtuse_angle_in_triangle_l208_208913

-- Defining points A, B, and C
def A : (ℝ × ℝ) := (1, 2)
def B : (ℝ × ℝ) := (-3, 4)
def C : (ℝ × ℝ) := (0, -2)

-- Function to calculate the square of the distance between two points
def dist_sq (p1 p2: ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the square of the lengths of sides AB, BC, and AC
def AB_sq : ℝ := dist_sq A B
def BC_sq : ℝ := dist_sq B C
def AC_sq : ℝ := dist_sq A C

-- Main theorem
theorem obtuse_angle_in_triangle :
  BC_sq > AB_sq + AC_sq :=
by
  -- Skipping the proof
  sorry

end obtuse_angle_in_triangle_l208_208913


namespace solve_inequality_l208_208129

theorem solve_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ (x ∈ Ioo (-6.5) 3.5) := 
sorry

end solve_inequality_l208_208129


namespace max_value_fraction_l208_208488

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (∀ x y : ℝ, (0 < x → 0 < y → (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3)) :=
by
  sorry

end max_value_fraction_l208_208488


namespace sequence_product_l208_208605

theorem sequence_product :
  (∏ n in finset.range 5, (1 / (3 ^ (2 * n + 1))) * (3 ^ (2 * n + 2))) = 243 :=
by
  sorry

end sequence_product_l208_208605


namespace river_current_speed_l208_208454

theorem river_current_speed 
  (downstream_distance upstream_distance still_water_speed : ℝ)
  (H1 : still_water_speed = 20)
  (H2 : downstream_distance = 100)
  (H3 : upstream_distance = 60)
  (H4 : (downstream_distance / (still_water_speed + x)) = (upstream_distance / (still_water_speed - x)))
  : x = 5 :=
by
  sorry

end river_current_speed_l208_208454


namespace distance_between_cities_l208_208996

noncomputable def distance_A_to_B : ℕ := 180
noncomputable def distance_B_to_A : ℕ := 150
noncomputable def total_distance : ℕ := distance_A_to_B + distance_B_to_A

theorem distance_between_cities : total_distance = 330 := by
  sorry

end distance_between_cities_l208_208996


namespace sum_of_cubes_of_roots_l208_208238

theorem sum_of_cubes_of_roots (r1 r2 r3 : ℂ) (h1 : r1 + r2 + r3 = 3) (h2 : r1 * r2 + r1 * r3 + r2 * r3 = 0) (h3 : r1 * r2 * r3 = -1) : 
  r1^3 + r2^3 + r3^3 = 24 :=
  sorry

end sum_of_cubes_of_roots_l208_208238


namespace board_tiling_condition_l208_208613

-- Define the problem in Lean

theorem board_tiling_condition (n : ℕ) : 
  (∃ m : ℕ, n * n = m + 4 * m) ↔ (∃ k : ℕ, n = 5 * k ∧ n > 5) := by 
sorry

end board_tiling_condition_l208_208613


namespace billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l208_208562

theorem billy_avoids_swimming_n_eq_2022 :
  ∀ n : ℕ, n = 2022 → (∃ (strategy : ℕ → ℕ), ∀ k, strategy (2022 + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_odd_n (n : ℕ) (h : n > 10 ∧ n % 2 = 1) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_even_n (n : ℕ) (h : n > 10 ∧ n % 2 = 0) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

end billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l208_208562


namespace face_value_is_100_l208_208328

-- Definitions based on conditions
def faceValue (F : ℝ) : Prop :=
  let discountedPrice := 0.92 * F
  let brokerageFee := 0.002 * discountedPrice
  let totalCostPrice := discountedPrice + brokerageFee
  totalCostPrice = 92.2

-- The proof statement in Lean
theorem face_value_is_100 : ∃ F : ℝ, faceValue F ∧ F = 100 :=
by
  use 100
  unfold faceValue
  simp
  norm_num
  sorry

end face_value_is_100_l208_208328


namespace probability_C_calc_l208_208029

noncomputable section

-- Define the given probabilities
def prob_A : ℚ := 3 / 8
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 3 / 16
def prob_D : ℚ := prob_C

-- The sum of probabilities equals 1
theorem probability_C_calc :
  prob_A + prob_B + prob_C + prob_D = 1 :=
by
  -- Simplifying directly, we can assert the correctness of given prob_C
  sorry

end probability_C_calc_l208_208029


namespace correct_factorization_l208_208015

theorem correct_factorization (a x m : ℝ) :
  (ax^2 - a = a * (x^2 - 1)) ∨
  (m^3 + m = m * (m^2 + 1)) ∨
  (x^2 + 2*x - 3 = x*(x+2) - 3) ∨
  (x^2 + 2*x - 3 = (x-3)*(x+1)) :=
by sorry

end correct_factorization_l208_208015


namespace area_union_after_rotation_l208_208516

-- Define the sides of the triangle
def PQ : ℝ := 11
def QR : ℝ := 13
def PR : ℝ := 12

-- Define the condition that H is the centroid of the triangle PQR
def centroid (P Q R H : ℝ × ℝ) : Prop := sorry -- This definition would require geometric relationships.

-- Statement to prove the area of the union of PQR and P'Q'R' after 180° rotation about H.
theorem area_union_after_rotation (P Q R H : ℝ × ℝ) (hPQ : dist P Q = PQ) (hQR : dist Q R = QR) (hPR : dist P R = PR) (hH : centroid P Q R H) : 
  let s := (PQ + QR + PR) / 2
  let area_PQR := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  2 * area_PQR = 12 * Real.sqrt 105 :=
sorry

end area_union_after_rotation_l208_208516


namespace set_intersection_complement_equiv_l208_208401

open Set

variable {α : Type*}
variable {x : α}

def U : Set ℝ := univ
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {x | x^2 < 1}

theorem set_intersection_complement_equiv :
  M ∩ (U \ N) = {x | 1 ≤ x} :=
by
  sorry

end set_intersection_complement_equiv_l208_208401


namespace fixed_point_difference_l208_208625

noncomputable def func (a x : ℝ) : ℝ := a^x + Real.log a

theorem fixed_point_difference (a m n : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (func a 0 = n) ∧ (y = func a x → (x = m) ∧ (y = n)) → (m - n = -2) :=
by 
  intro h
  sorry

end fixed_point_difference_l208_208625


namespace unique_a_exists_iff_n_eq_two_l208_208761

theorem unique_a_exists_iff_n_eq_two (n : ℕ) (h1 : 1 < n) : 
  (∃ a : ℕ, 0 < a ∧ a ≤ n! ∧ n! ∣ a^n + 1 ∧ ∀ b : ℕ, (0 < b ∧ b ≤ n! ∧ n! ∣ b^n + 1) → b = a) ↔ n = 2 := 
by {
  sorry
}

end unique_a_exists_iff_n_eq_two_l208_208761


namespace license_plate_increase_factor_l208_208081

def old_plate_count : ℕ := 26^2 * 10^3
def new_plate_count : ℕ := 26^4 * 10^4
def increase_factor : ℕ := new_plate_count / old_plate_count

theorem license_plate_increase_factor : increase_factor = 2600 :=
by
  unfold increase_factor
  rw [old_plate_count, new_plate_count]
  norm_num
  sorry

end license_plate_increase_factor_l208_208081


namespace sandy_age_when_record_l208_208686

noncomputable def calc_age (record_length current_length monthly_growth_rate age : ℕ) : ℕ :=
  let yearly_growth_rate := monthly_growth_rate * 12
  let needed_length := record_length - current_length
  let years_needed := needed_length / yearly_growth_rate
  age + years_needed

theorem sandy_age_when_record (record_length current_length monthly_growth_rate age : ℕ) :
  record_length = 26 →
  current_length = 2 →
  monthly_growth_rate = 1 →
  age = 12 →
  calc_age record_length current_length monthly_growth_rate age = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  unfold calc_age
  simp
  sorry

end sandy_age_when_record_l208_208686


namespace shorter_leg_of_right_triangle_l208_208373

theorem shorter_leg_of_right_triangle {a b : ℕ} (h : nat.gcd a b = 1) 
  (h_right_triangle : a^2 + b^2 = 65^2) : a = 25 ∨ b = 25 :=
by sorry

end shorter_leg_of_right_triangle_l208_208373


namespace complement_U_A_correct_l208_208784

-- Step 1: Define the universal set U
def U (x : ℝ) := x > 0

-- Step 2: Define the set A
def A (x : ℝ) := 0 < x ∧ x < 1

-- Step 3: Define the complement of A in U
def complement_U_A (x : ℝ) := U x ∧ ¬ A x

-- Step 4: Define the expected complement
def expected_complement (x : ℝ) := x ≥ 1

-- Step 5: The proof problem statement
theorem complement_U_A_correct (x : ℝ) : complement_U_A x = expected_complement x := by
  sorry

end complement_U_A_correct_l208_208784


namespace find_n_l208_208706

theorem find_n
  (n : ℤ)
  (h : n + (n + 1) + (n + 2) + (n + 3) = 30) :
  n = 6 :=
by
  sorry

end find_n_l208_208706


namespace total_jokes_proof_l208_208817

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l208_208817


namespace num_of_consec_int_sum_18_l208_208079

theorem num_of_consec_int_sum_18 : 
  ∃! (a n : ℕ), n ≥ 3 ∧ (n * (2 * a + n - 1)) = 36 :=
sorry

end num_of_consec_int_sum_18_l208_208079


namespace simplify_expression_l208_208985

theorem simplify_expression :
  (18 / 17) * (13 / 24) * (68 / 39) = 1 := 
by
  sorry

end simplify_expression_l208_208985


namespace ladder_wood_sufficiency_l208_208649

theorem ladder_wood_sufficiency
  (total_wood : ℝ)
  (rung_length_in: ℝ)
  (rung_distance_in: ℝ)
  (ladder_height_ft: ℝ)
  (total_wood_ft : total_wood = 300)
  (rung_length_ft : rung_length_in = 18 / 12)
  (rung_distance_ft : rung_distance_in = 6 / 12)
  (ladder_height_ft : ladder_height_ft = 50) :
  (∃ wood_needed : ℝ, wood_needed ≤ total_wood ∧ total_wood - wood_needed = 162.5) :=
sorry

end ladder_wood_sufficiency_l208_208649


namespace total_number_of_questions_l208_208505

theorem total_number_of_questions (type_a_problems type_b_problems : ℕ) 
(time_spent_type_a time_spent_type_b : ℕ) 
(total_exam_time : ℕ) 
(h1 : type_a_problems = 50) 
(h2 : time_spent_type_a = 2 * time_spent_type_b) 
(h3 : time_spent_type_a * type_a_problems = 72) 
(h4 : total_exam_time = 180) :
type_a_problems + type_b_problems = 200 := 
by
  sorry

end total_number_of_questions_l208_208505


namespace correct_statement_exam_l208_208802

theorem correct_statement_exam 
  (students_participated : ℕ)
  (students_sampled : ℕ)
  (statement1 : Bool)
  (statement2 : Bool)
  (statement3 : Bool)
  (statement4 : Bool)
  (cond1 : students_participated = 70000)
  (cond2 : students_sampled = 1000)
  (cond3 : statement1 = False)
  (cond4 : statement2 = False)
  (cond5 : statement3 = False)
  (cond6 : statement4 = True) :
  statement4 = True := 
sorry

end correct_statement_exam_l208_208802


namespace fifth_number_in_eighth_row_l208_208535

theorem fifth_number_in_eighth_row : 
  (∀ n : ℕ, ∃ k : ℕ, k = n * n ∧ 
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
      k - (n - m) = 54 → m = 5 ∧ n = 8) := by sorry

end fifth_number_in_eighth_row_l208_208535


namespace alpha_beta_sum_l208_208602

variable (α β : ℝ)

theorem alpha_beta_sum (h : ∀ x, (x - α) / (x + β) = (x^2 - 64 * x + 992) / (x^2 + 56 * x - 3168)) :
  α + β = 82 :=
sorry

end alpha_beta_sum_l208_208602


namespace find_g_product_l208_208822

theorem find_g_product 
  (x1 x2 x3 x4 x5 : ℝ)
  (h_root1 : x1^5 - x1^3 + 1 = 0)
  (h_root2 : x2^5 - x2^3 + 1 = 0)
  (h_root3 : x3^5 - x3^3 + 1 = 0)
  (h_root4 : x4^5 - x4^3 + 1 = 0)
  (h_root5 : x5^5 - x5^3 + 1 = 0)
  (g : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2 - 3) :
  g x1 * g x2 * g x3 * g x4 * g x5 = 107 := 
sorry

end find_g_product_l208_208822


namespace original_price_discount_l208_208987

theorem original_price_discount (P : ℝ) (h : 0.90 * P = 450) : P = 500 :=
by
  sorry

end original_price_discount_l208_208987


namespace exists_multiple_with_all_digits_l208_208266

theorem exists_multiple_with_all_digits (n : ℕ) :
  ∃ m : ℕ, (m % n = 0) ∧ (∀ d : ℕ, d < 10 → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_multiple_with_all_digits_l208_208266


namespace train_speed_l208_208329

theorem train_speed (distance : ℝ) (time : ℝ) (distance_eq : distance = 270) (time_eq : time = 9)
  : (distance / time) * (3600 / 1000) = 108 :=
by 
  sorry

end train_speed_l208_208329


namespace find_x_l208_208085

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end find_x_l208_208085


namespace complement_is_correct_l208_208935

-- Define the universal set U and set M
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

-- Define the complement of M with respect to U
def complement_U (U M : Set ℕ) : Set ℕ := {x ∈ U | x ∉ M}

-- State the theorem to be proved
theorem complement_is_correct : complement_U U M = {3, 5, 6} :=
by
  sorry

end complement_is_correct_l208_208935


namespace range_of_function_l208_208297

theorem range_of_function (y : ℝ) (t: ℝ) (x : ℝ) (h_t : t = x^2 - 1) (h_domain : t ∈ Set.Ici (-1)) :
  ∃ (y_set : Set ℝ), ∀ y ∈ y_set, y = (1/3)^t ∧ y_set = Set.Ioo 0 3 ∨ y_set = Set.Icc 0 3 := by
  sorry

end range_of_function_l208_208297


namespace abs_inequality_solution_l208_208132

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end abs_inequality_solution_l208_208132


namespace find_ab_l208_208503

-- Define the "¤" operation
def op (x y : ℝ) := (x + y)^2 - (x - y)^2

-- The Lean 4 theorem statement
theorem find_ab (a b : ℝ) (h : op a b = 24) : a * b = 6 := 
by
  -- We leave the proof as an exercise
  sorry

end find_ab_l208_208503


namespace find_number_l208_208431

theorem find_number (x : ℕ) (h : 8 * x = 64) : x = 8 :=
sorry

end find_number_l208_208431


namespace time_to_pay_back_l208_208390

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l208_208390


namespace range_x_minus_y_l208_208097

-- Definition of the curve in polar coordinates
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta + 2 * Real.sin theta

-- Conversion to rectangular coordinates
noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 2 * y

-- The final Lean 4 statement
theorem range_x_minus_y (x y : ℝ) (h : curve_rectangular x y) :
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10 :=
sorry

end range_x_minus_y_l208_208097


namespace simplify_polynomial_l208_208405

theorem simplify_polynomial (s : ℝ) :
  (2 * s ^ 2 + 5 * s - 3) - (2 * s ^ 2 + 9 * s - 6) = -4 * s + 3 :=
by 
  sorry

end simplify_polynomial_l208_208405


namespace incorrect_expressions_l208_208137

theorem incorrect_expressions (x y : ℚ) (h : x / y = 2 / 5) :
    (x + 3 * y) / x ≠ 17 / 2 ∧ (x - y) / y ≠ 3 / 5 :=
by
  sorry

end incorrect_expressions_l208_208137


namespace average_of_solutions_l208_208451

theorem average_of_solutions (a b : ℝ) :
  (∃ x1 x2 : ℝ, 3 * a * x1^2 - 6 * a * x1 + 2 * b = 0 ∧
                3 * a * x2^2 - 6 * a * x2 + 2 * b = 0 ∧
                x1 ≠ x2) →
  (1 + 1) / 2 = 1 :=
by
  intros
  sorry

end average_of_solutions_l208_208451


namespace dot_product_eq_one_l208_208798

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_eq_one_l208_208798


namespace degrees_to_radians_l208_208215

theorem degrees_to_radians (deg: ℝ) (h : deg = 120) : deg * (π / 180) = 2 * π / 3 :=
by
  simp [h]
  sorry

end degrees_to_radians_l208_208215


namespace power_multiplication_l208_208905

theorem power_multiplication :
  3^5 * 6^5 = 1889568 :=
by
  sorry

end power_multiplication_l208_208905


namespace total_spending_is_correct_l208_208881

def total_spending : ℝ :=
  let meal_expenses_10 := 10 * 18
  let meal_expenses_5 := 5 * 25
  let total_meal_expenses := meal_expenses_10 + meal_expenses_5
  let service_charge := 50
  let total_before_discount := total_meal_expenses + service_charge
  let discount := 0.05 * total_meal_expenses
  let total_after_discount := total_before_discount - discount
  let tip := 0.10 * total_before_discount
  total_after_discount + tip

theorem total_spending_is_correct : total_spending = 375.25 :=
by
  sorry

end total_spending_is_correct_l208_208881


namespace unique_real_root_of_quadratic_l208_208945

theorem unique_real_root_of_quadratic (k : ℝ) :
  (∃ a : ℝ, ∀ b : ℝ, ((k^2 - 9) * b^2 - 2 * (k + 1) * b + 1 = 0 → b = a)) ↔ (k = 3 ∨ k = -3 ∨ k = -5) :=
by
  sorry

end unique_real_root_of_quadratic_l208_208945


namespace height_of_parallelogram_l208_208344

theorem height_of_parallelogram
  (A B H : ℝ)
  (h1 : A = 480)
  (h2 : B = 32)
  (h3 : A = B * H) : 
  H = 15 := sorry

end height_of_parallelogram_l208_208344


namespace distinct_dress_designs_l208_208196

theorem distinct_dress_designs : 
  let num_colors := 5
  let num_patterns := 6
  num_colors * num_patterns = 30 :=
by
  sorry

end distinct_dress_designs_l208_208196


namespace any_nat_representation_as_fraction_l208_208870

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l208_208870


namespace find_y_l208_208346

theorem find_y (Y : ℝ) (h : (200 + 200 / Y) * Y = 18200) : Y = 90 :=
by
  sorry

end find_y_l208_208346


namespace smallest_m_integral_roots_l208_208868

theorem smallest_m_integral_roots (m : ℕ) : 
  (∃ p q : ℤ, (10 * p * p - ↑m * p + 360 = 0) ∧ (p + q = m / 10) ∧ (p * q = 36) ∧ (p % q = 0 ∨ q % p = 0)) → 
  m = 120 :=
by
sorry

end smallest_m_integral_roots_l208_208868


namespace original_number_l208_208709

theorem original_number (x : ℝ) (h : 1000 * x = 3 / x) : x = (real.sqrt 30) / 100 :=
by sorry

end original_number_l208_208709


namespace probability_sum_15_pair_octahedral_dice_l208_208550

theorem probability_sum_15_pair_octahedral_dice (n : ℕ) (h1 : n = 8) :
  (∃ (A B : fin n → fin n), (A + B = 15) ∧ ((A, B) = (7, 8) ∨ (A, B) = (8, 7))) → 
  (2 / (n * n) = 1 / 32) :=
by
  sorry

end probability_sum_15_pair_octahedral_dice_l208_208550


namespace geom_seq_inv_sum_eq_l208_208095

noncomputable def geom_seq (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * r^n

theorem geom_seq_inv_sum_eq
    (a_1 r : ℚ)
    (h_sum : geom_seq a_1 r 0 + geom_seq a_1 r 1 + geom_seq a_1 r 2 + geom_seq a_1 r 3 = 15/8)
    (h_prod : geom_seq a_1 r 1 * geom_seq a_1 r 2 = -9/8) :
  1 / geom_seq a_1 r 0 + 1 / geom_seq a_1 r 1 + 1 / geom_seq a_1 r 2 + 1 / geom_seq a_1 r 3 = -5/3 :=
sorry

end geom_seq_inv_sum_eq_l208_208095


namespace min_value_expression_l208_208617

noncomputable def f (t : ℝ) : ℝ :=
  (1 / (t + 1)) + (2 * t / (2 * t + 1))

theorem min_value_expression (x y : ℝ) (h : x * y > 0) :
  ∃ t, (x / y = t) ∧ t > 0 ∧ f t = 4 - 2 * Real.sqrt 2 := 
  sorry

end min_value_expression_l208_208617


namespace points_on_line_l208_208059

theorem points_on_line : 
    ∀ (P : ℝ × ℝ),
      (P = (1, 2) ∨ P = (0, 0) ∨ P = (2, 4) ∨ P = (5, 10) ∨ P = (-1, -2))
      → (∃ m b, m = 2 ∧ b = 0 ∧ P.2 = m * P.1 + b) :=
by
  sorry

end points_on_line_l208_208059


namespace at_least_one_fraction_lt_two_l208_208925

theorem at_least_one_fraction_lt_two 
  (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : 2 < x + y) : 
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_fraction_lt_two_l208_208925


namespace nonagon_diagonals_count_eq_27_l208_208494

theorem nonagon_diagonals_count_eq_27 :
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  distinct_diagonals = 27 :=
by
  let vertices := 9
  let connections_per_vertex := vertices - 3
  let total_raw_count_diagonals := vertices * connections_per_vertex
  let distinct_diagonals := total_raw_count_diagonals / 2
  have : distinct_diagonals = 27 := sorry
  exact this

end nonagon_diagonals_count_eq_27_l208_208494


namespace shorter_leg_of_right_triangle_l208_208375

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l208_208375


namespace chess_tournament_participants_l208_208507

theorem chess_tournament_participants (n : ℕ) 
  (h : (n * (n - 1)) / 2 = 15) : n = 6 :=
sorry

end chess_tournament_participants_l208_208507


namespace symmetrical_shapes_congruent_l208_208563

theorem symmetrical_shapes_congruent
  (shapes : Type)
  (is_symmetrical : shapes → shapes → Prop)
  (congruent : shapes → shapes → Prop)
  (symmetrical_implies_equal_segments : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (segment : ℝ), segment_s1 = segment_s2)
  (symmetrical_implies_equal_angles : ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → ∀ (angle : ℝ), angle_s1 = angle_s2) :
  ∀ (s1 s2 : shapes), is_symmetrical s1 s2 → congruent s1 s2 :=
by
  sorry

end symmetrical_shapes_congruent_l208_208563


namespace number_exceeds_25_percent_by_150_l208_208569

theorem number_exceeds_25_percent_by_150 (x : ℝ) : (0.25 * x + 150 = x) → x = 200 :=
by
  sorry

end number_exceeds_25_percent_by_150_l208_208569


namespace clock_hand_positions_l208_208214

theorem clock_hand_positions : ∃ n : ℕ, n = 143 ∧ 
  (∀ t : ℝ, let hour_pos := t / 12
            let min_pos := t
            let switched_hour_pos := t
            let switched_min_pos := t / 12
            hour_pos = switched_min_pos ∧ min_pos = switched_hour_pos ↔
            ∃ k : ℤ, t = k / 11) :=
by sorry

end clock_hand_positions_l208_208214


namespace remainder_of_4n_minus_6_l208_208793

theorem remainder_of_4n_minus_6 (n : ℕ) (h : n % 9 = 5) : (4 * n - 6) % 9 = 5 :=
sorry

end remainder_of_4n_minus_6_l208_208793


namespace option_A_correct_l208_208530

variable (f g : ℝ → ℝ)

-- Given conditions
axiom cond1 : ∀ x : ℝ, f x - g (4 - x) = 2
axiom cond2 : ∀ x : ℝ, deriv g x = deriv f (x - 2)
axiom cond3 : ∀ x : ℝ, f (x + 2) = - f (- x - 2)

theorem option_A_correct : ∀ x : ℝ, f (4 + x) + f (- x) = 0 :=
by
  -- Proving the theorem
  sorry

end option_A_correct_l208_208530


namespace max_marks_is_400_l208_208877

theorem max_marks_is_400 :
  ∃ M : ℝ, (0.30 * M = 120) ∧ (M = 400) := 
by 
  sorry

end max_marks_is_400_l208_208877


namespace find_f_24_25_26_l208_208414

-- Given conditions
def homogeneous (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (n a b c : ℤ), f (n * a) (n * b) (n * c) = n * f a b c

def shift_invariance (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c n : ℤ), f (a + n) (b + n) (c + n) = f a b c + n

def symmetry (f : ℤ → ℤ → ℤ → ℝ) : Prop :=
  ∀ (a b c : ℤ), f a b c = f c b a

-- Proving the required value under the conditions
theorem find_f_24_25_26 (f : ℤ → ℤ → ℤ → ℝ)
  (homo : homogeneous f) 
  (shift : shift_invariance f) 
  (symm : symmetry f) : 
  f 24 25 26 = 25 := 
sorry

end find_f_24_25_26_l208_208414


namespace minEmployees_correct_l208_208205

noncomputable def minEmployees (seaTurtles birdMigration bothTurtlesBirds turtlesPlants allThree : ℕ) : ℕ :=
  let onlySeaTurtles := seaTurtles - (bothTurtlesBirds + turtlesPlants - allThree)
  let onlyBirdMigration := birdMigration - (bothTurtlesBirds + allThree - turtlesPlants)
  onlySeaTurtles + onlyBirdMigration + bothTurtlesBirds + turtlesPlants + allThree

theorem minEmployees_correct :
  minEmployees 120 90 30 50 15 = 245 := by
  sorry

end minEmployees_correct_l208_208205


namespace compute_value_l208_208333

theorem compute_value
  (x y z : ℝ)
  (h1 : (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -9)
  (h2 : (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 15) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 13.5 :=
by
  sorry

end compute_value_l208_208333


namespace simplify_fraction_l208_208275

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208275


namespace max_consecutive_sum_le_1000_l208_208002

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l208_208002


namespace function_evaluation_l208_208359

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x ^ 2 + 1) : 
  ∀ x : ℝ, f x = 2 * x ^ 2 - 4 * x + 3 :=
sorry

end function_evaluation_l208_208359


namespace maximum_value_x_2y_2z_l208_208821

noncomputable def max_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : ℝ :=
  x + 2*y + 2*z

theorem maximum_value_x_2y_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) : 
  max_sum x y z h ≤ 15 :=
sorry

end maximum_value_x_2y_2z_l208_208821


namespace initial_liquid_A_quantity_l208_208030

theorem initial_liquid_A_quantity
  (x : ℝ)
  (init_A init_B init_C : ℝ)
  (removed_A removed_B removed_C : ℝ)
  (added_B added_C : ℝ)
  (new_A new_B new_C : ℝ)
  (h1 : init_A / init_B = 7 / 5)
  (h2 : init_A / init_C = 7 / 3)
  (h3 : init_A + init_B + init_C = 15 * x)
  (h4 : removed_A = 7 / 15 * 9)
  (h5 : removed_B = 5 / 15 * 9)
  (h6 : removed_C = 3 / 15 * 9)
  (h7 : new_A = init_A - removed_A)
  (h8 : new_B = init_B - removed_B + added_B)
  (h9 : new_C = init_C - removed_C + added_C)
  (h10 : new_A / (new_B + new_C) = 7 / 10)
  (h11 : added_B = 6)
  (h12 : added_C = 3) : 
  init_A = 35.7 :=
sorry

end initial_liquid_A_quantity_l208_208030


namespace AM_minus_GM_lower_bound_l208_208936

theorem AM_minus_GM_lower_bound (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : 
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := 
by {
  sorry -- Proof to be filled in
}

end AM_minus_GM_lower_bound_l208_208936


namespace incorrect_inequality_l208_208231

theorem incorrect_inequality (a b c : ℝ) (h : a > b) : ¬ (forall c, a * c > b * c) :=
by
  intro h'
  have h'' := h' c
  sorry

end incorrect_inequality_l208_208231


namespace abs_inequality_solution_l208_208126

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l208_208126


namespace simplify_fraction_l208_208286

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end simplify_fraction_l208_208286


namespace train_boarding_probability_l208_208897

theorem train_boarding_probability :
  (0.5 / 5) = 1 / 10 :=
by sorry

end train_boarding_probability_l208_208897


namespace smallest_value_of_abs_sum_l208_208704

theorem smallest_value_of_abs_sum : ∃ x : ℝ, (x = -3) ∧ ( ∀ y : ℝ, |y + 1| + |y + 3| + |y + 6| ≥ 5 ) :=
by
  use -3
  split
  . exact rfl
  . intro y
    have h1 : |y + 1| + |y + 3| + |y + 6| = sorry,
    sorry

end smallest_value_of_abs_sum_l208_208704


namespace alice_vs_bob_payment_multiple_l208_208331

theorem alice_vs_bob_payment_multiple :
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  total_alice_payment / bob_payment = 9 := by
  -- define the variables as per the conditions
  let alice_acorns := 3600
  let price_per_acorn := 15
  let bob_payment := 6000
  let total_alice_payment := alice_acorns * price_per_acorn
  -- define the target statement
  show total_alice_payment / bob_payment = 9
  sorry

end alice_vs_bob_payment_multiple_l208_208331


namespace smallest_q_p_difference_l208_208111

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l208_208111


namespace part1_proof_part2_proof_l208_208356

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - 1| - |x - m|

theorem part1_proof : ∀ x, f x 2 ≥ 1 ↔ x ≥ 2 :=
by 
  sorry

theorem part2_proof : (∀ x : ℝ, f x m ≤ 5) → (-4 ≤ m ∧ m ≤ 6) :=
by
  sorry

end part1_proof_part2_proof_l208_208356


namespace tina_took_away_2_oranges_l208_208865

-- Definition of the problem
def oranges_taken_away (x : ℕ) : Prop :=
  let original_oranges := 5
  let tangerines_left := 17 - 10 
  let oranges_left := original_oranges - x
  tangerines_left = oranges_left + 4 

-- The statement that needs to be proven
theorem tina_took_away_2_oranges : oranges_taken_away 2 :=
  sorry

end tina_took_away_2_oranges_l208_208865


namespace stormi_mowing_charge_l208_208838

theorem stormi_mowing_charge (cars_washed : ℕ) (car_wash_price : ℕ) (lawns_mowed : ℕ) (bike_cost : ℕ) (money_needed_more : ℕ) 
  (total_from_cars : ℕ := cars_washed * car_wash_price)
  (total_earned : ℕ := bike_cost - money_needed_more)
  (earned_from_lawns : ℕ := total_earned - total_from_cars) :
  cars_washed = 3 → car_wash_price = 10 → lawns_mowed = 2 → bike_cost = 80 → money_needed_more = 24 → earned_from_lawns / lawns_mowed = 13 := 
by
  sorry

end stormi_mowing_charge_l208_208838


namespace total_valid_votes_l208_208643

theorem total_valid_votes (V : ℝ) (H_majority : 0.70 * V - 0.30 * V = 188) : V = 470 :=
by
  sorry

end total_valid_votes_l208_208643


namespace largest_fraction_sum_l208_208595

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end largest_fraction_sum_l208_208595


namespace find_A_l208_208839

variable (x A B C : ℝ)

theorem find_A :
  (∃ A B C : ℝ, (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → 
  (1 / (x^3 + 2 * x^2 - 19 * x - 30) = 
  (A / (x + 3)) + (B / (x - 2)) + (C / (x - 2)^2)) ∧ 
  A = 1 / 25)) :=
by
  sorry

end find_A_l208_208839


namespace determine_a_value_l208_208265

noncomputable def rectangle_value (a : ℝ) : ℝ := (1 + real.sqrt 33) / 2 + 8

theorem determine_a_value :
  ∃ a : ℝ, 
    -- Conditions
    (∃ x : ℝ, ∃ y : ℝ, y = real.log a x ∧ x * 4 = 32 ∧ (x + 8) * real.log a (x + 8) = 2 * real.log a x ∧ (x + 8, y + 4)) ∧ 
    -- Conclusion
    a = real.root 4 (rectangle_value a) :=
begin
  sorry
end

end determine_a_value_l208_208265


namespace geometric_series_sum_l208_208306

theorem geometric_series_sum :
  let a_0 := 1 / 2
  let r := 1 / 2
  let n := 5
  let sum := ∑ i in Finset.range(n), a_0 * r^i
  sum = 31 / 32 :=
by
  sorry

end geometric_series_sum_l208_208306


namespace sequence_general_term_l208_208770

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n + 1) : a n = n * n :=
by
  sorry

end sequence_general_term_l208_208770


namespace negation_of_universal_proposition_l208_208678

theorem negation_of_universal_proposition :
  (¬ ∀ x > 1, (1 / 2)^x < 1 / 2) ↔ (∃ x > 1, (1 / 2)^x ≥ 1 / 2) :=
sorry

end negation_of_universal_proposition_l208_208678


namespace infinite_coprime_binom_l208_208528

theorem infinite_coprime_binom (k l : ℕ) (hk : k > 0) (hl : l > 0) : 
  ∃ᶠ m in atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
sorry

end infinite_coprime_binom_l208_208528


namespace remainder_of_8x_minus_5_l208_208565

theorem remainder_of_8x_minus_5 (x : ℕ) (h : x % 15 = 7) : (8 * x - 5) % 15 = 6 :=
by
  sorry

end remainder_of_8x_minus_5_l208_208565


namespace find_larger_number_l208_208942

theorem find_larger_number (x y : ℕ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : x + y = 54) : 
  y = 30 :=
sorry

end find_larger_number_l208_208942


namespace greatest_large_chips_l208_208862

theorem greatest_large_chips (s l p : ℕ) (h1 : s + l = 80) (h2 : s = l + p) (hp : Nat.Prime p) : l ≤ 39 :=
by
  sorry

end greatest_large_chips_l208_208862


namespace time_to_pay_back_l208_208388

def total_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

def monthly_profit := monthly_revenue - monthly_expenses

theorem time_to_pay_back : 
  (total_cost / monthly_profit) = 10 := 
by
  -- Definition of monthly_profit 
  have monthly_profit_def : monthly_profit = 4000 - 1500 := rfl
  rw [monthly_profit_def]
  
  -- Performing the division
  show (25000 / 2500) = 10
  apply div_eq_of_eq_mul
  norm_num
  sorry

end time_to_pay_back_l208_208388


namespace train_on_time_speed_l208_208743

theorem train_on_time_speed :
  ∃ (v : ℝ), (70 / v) + 0.25 = 70 / 35 ∧ v = 40 :=
by
  existsi (40 : ℝ)
  split
  calc
    70 / 40 + 0.25 = 1.75 + 0.25 : by norm_num
                ... = 2           : by norm_num
  sorry

end train_on_time_speed_l208_208743


namespace ending_number_of_range_divisible_by_five_l208_208559

theorem ending_number_of_range_divisible_by_five
  (first_number : ℕ)
  (number_of_terms : ℕ)
  (h_first : first_number = 15)
  (h_terms : number_of_terms = 10)
  : ∃ ending_number : ℕ, ending_number = first_number + 5 * (number_of_terms - 1) := 
by
  sorry

end ending_number_of_range_divisible_by_five_l208_208559


namespace Abby_wins_if_N_2011_Brian_wins_in_31_cases_l208_208899

-- Definitions and assumptions directly from the problem conditions
inductive Player
| Abby
| Brian

def game_condition (N : ℕ) : Prop :=
  ∀ (p : Player), 
    (p = Player.Abby → (∃ k, N = 2 * k + 1)) ∧ 
    (p = Player.Brian → (∃ k, N = 2 * (2^k - 1))) -- This encodes the winning state conditions for simplicity

-- Part (a)
theorem Abby_wins_if_N_2011 : game_condition 2011 :=
by
  sorry

-- Part (b)
theorem Brian_wins_in_31_cases : 
  (∃ S : Finset ℕ, (∀ N ∈ S, N ≤ 2011 ∧ game_condition N) ∧ S.card = 31) :=
by
  sorry

end Abby_wins_if_N_2011_Brian_wins_in_31_cases_l208_208899


namespace arithmetic_sequence_sum_l208_208171

/-
The sum of the first 20 terms of the arithmetic sequence 8, 5, 2, ... is -410.
-/

theorem arithmetic_sequence_sum :
  let a : ℤ := 8
  let d : ℤ := -3
  let n : ℤ := 20
  let S_n : ℤ := n * a + (d * n * (n - 1)) / 2
  S_n = -410 := by
  sorry

end arithmetic_sequence_sum_l208_208171


namespace simplify_fraction_l208_208281

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208281


namespace prob_A_and_B_succeed_prob_vaccine_A_successful_l208_208751

-- Define the probabilities of success for Company A, Company B, and Company C
def P_A := (2 : ℚ) / 3
def P_B := (1 : ℚ) / 2
def P_C := (3 : ℚ) / 5

-- Define the theorem statements

-- Theorem for the probability that both Company A and Company B succeed
theorem prob_A_and_B_succeed : P_A * P_B = 1 / 3 := by
  sorry

-- Theorem for the probability that vaccine A is successfully developed
theorem prob_vaccine_A_successful : 1 - ((1 - P_A) * (1 - P_B)) = 5 / 6 := by
  sorry

end prob_A_and_B_succeed_prob_vaccine_A_successful_l208_208751


namespace probability_units_digit_prime_correct_l208_208741

noncomputable def probability_units_digit_prime : ℚ :=
  let primes := {2, 3, 5, 7}
  let total_outcomes := 10
  primes.card / total_outcomes

theorem probability_units_digit_prime_correct :
  probability_units_digit_prime = 2 / 5 := by
  sorry

end probability_units_digit_prime_correct_l208_208741


namespace problem_equiv_proof_l208_208767

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1)

theorem problem_equiv_proof :
  (2 ^ a + 2 ^ b ≥ 2 * Real.sqrt 2) ∧
  (Real.log a / Real.log 2 + Real.log b / Real.log 2 ≤ -2) ∧
  (a ^ 2 + b ^ 2 ≥ 1 / 2) :=
by
  sorry

end problem_equiv_proof_l208_208767


namespace solve_for_x_l208_208407

theorem solve_for_x (x y : ℕ) (h₁ : 9 ^ y = 3 ^ x) (h₂ : y = 6) : x = 12 :=
by
  sorry

end solve_for_x_l208_208407


namespace time_to_pay_back_l208_208391

-- Definitions for conditions
def initial_cost : ℕ := 25000
def monthly_revenue : ℕ := 4000
def monthly_expenses : ℕ := 1500
def monthly_profit : ℕ := monthly_revenue - monthly_expenses

-- Theorem statement
theorem time_to_pay_back : initial_cost / monthly_profit = 10 := by
  -- Skipping the proof here
  sorry

end time_to_pay_back_l208_208391


namespace parametric_equation_of_line_passing_through_M_l208_208321

theorem parametric_equation_of_line_passing_through_M (
  t : ℝ
) : 
    ∃ x y : ℝ, 
      x = 1 + (t * (Real.cos (Real.pi / 3))) ∧ 
      y = 5 + (t * (Real.sin (Real.pi / 3))) ∧ 
      x = 1 + (1/2) * t ∧ 
      y = 5 + (Real.sqrt 3 / 2) * t := 
by
  sorry

end parametric_equation_of_line_passing_through_M_l208_208321


namespace triangle_angle_sum_l208_208094

theorem triangle_angle_sum (x : ℝ) :
    let angle1 : ℝ := 40
    let angle2 : ℝ := 4 * x
    let angle3 : ℝ := 3 * x
    angle1 + angle2 + angle3 = 180 -> x = 20 := 
sorry

end triangle_angle_sum_l208_208094


namespace john_pays_more_than_jane_l208_208970

noncomputable def original_price : ℝ := 34.00
noncomputable def discount : ℝ := 0.10
noncomputable def tip_percent : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price - (discount * original_price)
noncomputable def john_tip : ℝ := tip_percent * original_price
noncomputable def john_total : ℝ := discounted_price + john_tip
noncomputable def jane_tip : ℝ := tip_percent * discounted_price
noncomputable def jane_total : ℝ := discounted_price + jane_tip

theorem john_pays_more_than_jane : john_total - jane_total = 0.51 := by
  sorry

end john_pays_more_than_jane_l208_208970


namespace chef_earns_less_than_manager_l208_208463

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.22

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 :=
by
  sorry

end chef_earns_less_than_manager_l208_208463


namespace nonrational_ab_l208_208788

theorem nonrational_ab {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
    ¬(∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) := by
  sorry

end nonrational_ab_l208_208788


namespace repeating_decimals_subtraction_l208_208212

/--
Calculate the value of 0.\overline{234} - 0.\overline{567} - 0.\overline{891}.
Express your answer as a fraction in its simplest form.

Shown that:
Let x = 0.\overline{234}, y = 0.\overline{567}, z = 0.\overline{891},
Then 0.\overline{234} - 0.\overline{567} - 0.\overline{891} = -1224/999
-/
theorem repeating_decimals_subtraction : 
  let x : ℚ := 234 / 999
  let y : ℚ := 567 / 999
  let z : ℚ := 891 / 999
  x - y - z = -1224 / 999 := 
by
  sorry

end repeating_decimals_subtraction_l208_208212


namespace weeks_to_save_l208_208924

-- Define the conditions as given in the problem
def cost_of_bike : ℕ := 600
def gift_from_parents : ℕ := 60
def gift_from_uncle : ℕ := 40
def gift_from_sister : ℕ := 20
def gift_from_friend : ℕ := 30
def weekly_earnings : ℕ := 18

-- Total gift money
def total_gift_money : ℕ := gift_from_parents + gift_from_uncle + gift_from_sister + gift_from_friend

-- Total money after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_gift_money + weekly_earnings * x

-- Main theorem statement
theorem weeks_to_save (x : ℕ) : total_money_after_weeks x = cost_of_bike → x = 25 := by
  sorry

end weeks_to_save_l208_208924


namespace time_to_pay_back_l208_208389

def total_cost : ℝ := 25000
def monthly_revenue : ℝ := 4000
def monthly_expenses : ℝ := 1500

def monthly_profit := monthly_revenue - monthly_expenses

theorem time_to_pay_back : 
  (total_cost / monthly_profit) = 10 := 
by
  -- Definition of monthly_profit 
  have monthly_profit_def : monthly_profit = 4000 - 1500 := rfl
  rw [monthly_profit_def]
  
  -- Performing the division
  show (25000 / 2500) = 10
  apply div_eq_of_eq_mul
  norm_num
  sorry

end time_to_pay_back_l208_208389


namespace average_age_without_teacher_l208_208548

theorem average_age_without_teacher 
  (A : ℕ) 
  (h : 15 * A + 26 = 16 * (A + 1)) : 
  A = 10 :=
sorry

end average_age_without_teacher_l208_208548


namespace perimeter_ratio_of_squares_l208_208142

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l208_208142


namespace xiaoMing_better_performance_l208_208874

-- Definitions based on conditions
def xiaoMing_scores : List Float := [90, 67, 90, 92, 96]
def xiaoLiang_scores : List Float := [87, 62, 90, 92, 92]

-- Definitions of average and variance calculation
def average (scores : List Float) : Float :=
  (scores.sum) / (scores.length.toFloat)

def variance (scores : List Float) : Float :=
  let avg := average scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length.toFloat)

-- Prove that Xiao Ming's performance is better than Xiao Liang's.
theorem xiaoMing_better_performance :
  average xiaoMing_scores > average xiaoLiang_scores ∧ variance xiaoMing_scores < variance xiaoLiang_scores :=
by
  sorry

end xiaoMing_better_performance_l208_208874


namespace sum_remainder_l208_208304

theorem sum_remainder (n : ℕ) (h : n = 102) :
  ((n * (n + 1) / 2) % 5250) = 3 :=
by
  sorry

end sum_remainder_l208_208304


namespace monotonicity_of_f_exists_a_for_min_g_l208_208073

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + a * x + a / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := log x + a / x - 2

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, 0 < f' x a) ∨ 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ 
   (∀ x ∈ (Ioo 0 x1 ⊔ Ioc x2 ∞), f' x a > 0) ∧ 
   (∀ x ∈ Ioo x1 x2, f' x a < 0)) :=
sorry

theorem exists_a_for_min_g : 
  ∃ a : ℝ, 0 < a ∧ (∀ x ∈ Ioc (0:ℝ) (exp 2), g x a ≥ 2) ∧ g (exp 2) a = 2 :=
sorry

end monotonicity_of_f_exists_a_for_min_g_l208_208073


namespace figure_F12_diamonds_l208_208241

-- Definitions based on conditions
def initial_diamonds : ℕ := 3

def added_diamonds (n : ℕ) : ℕ := 8 * n

def total_diamonds : ℕ → ℕ
| 1 := initial_diamonds
| n + 1 := total_diamonds n + added_diamonds (n + 1)

-- Proposition to be proved
theorem figure_F12_diamonds : total_diamonds 12 = 619 :=
sorry

end figure_F12_diamonds_l208_208241


namespace pipe_fill_rate_l208_208581

theorem pipe_fill_rate 
  (C : ℝ) (t : ℝ) (capacity : C = 4000) (time_to_fill : t = 300) :
  (3/4 * C / t) = 10 := 
by 
  sorry

end pipe_fill_rate_l208_208581


namespace perimeters_ratio_l208_208158

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l208_208158


namespace total_cost_l208_208455

/-- There are two types of discs, one costing 10.50 and another costing 8.50.
You bought a total of 10 discs, out of which 6 are priced at 8.50.
The task is to determine the total amount spent. -/
theorem total_cost (price1 price2 : ℝ) (num1 num2 : ℕ) 
  (h1 : price1 = 10.50) (h2 : price2 = 8.50) 
  (h3 : num1 = 6) (h4 : num2 = 10) 
  (h5 : num2 - num1 = 4) : 
  (num1 * price2 + (num2 - num1) * price1) = 93.00 := 
by
  sorry

end total_cost_l208_208455


namespace find_multiplier_l208_208014

theorem find_multiplier (n x : ℤ) (h1: n = 12) (h2: 4 * n - 3 = (n - 7) * x) : x = 9 :=
by {
  sorry
}

end find_multiplier_l208_208014


namespace functional_equality_l208_208760

theorem functional_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f y + x ^ 2 + 1) + 2 * x = y + (f (x + 1)) ^ 2) →
  (∀ x : ℝ, f x = x) := 
by
  intro h
  sorry

end functional_equality_l208_208760


namespace geometric_series_sum_infinity_l208_208170

theorem geometric_series_sum_infinity (a₁ : ℝ) (q : ℝ) (S₆ S₃ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : S₆ / S₃ = 7 / 8)
  (h₃ : S₆ = a₁ * (1 - q ^ 6) / (1 - q))
  (h₄ : S₃ = a₁ * (1 - q ^ 3) / (1 - q)) :
  ∑' i : ℕ, a₁ * q ^ i = 2 := by
  sorry

end geometric_series_sum_infinity_l208_208170


namespace probability_participation_on_both_days_l208_208347

-- Definitions based on conditions
def total_students := 5
def total_combinations := 2^total_students
def same_day_scenarios := 2
def favorable_outcomes := total_combinations - same_day_scenarios

-- Theorem statement
theorem probability_participation_on_both_days :
  (favorable_outcomes / total_combinations : ℚ) = 15 / 16 :=
by
  sorry

end probability_participation_on_both_days_l208_208347


namespace total_right_handed_players_l208_208311

-- Defining the conditions and the given values
def total_players : ℕ := 61
def throwers : ℕ := 37
def non_throwers : ℕ := total_players - throwers
def left_handed_non_throwers : ℕ := non_throwers / 3
def right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers

-- The proof goal
theorem total_right_handed_players 
  (h1 : total_players = 61)
  (h2 : throwers = 37)
  (h3 : non_throwers = total_players - throwers)
  (h4 : left_handed_non_throwers = non_throwers / 3)
  (h5 : right_handed_non_throwers = non_throwers - left_handed_non_throwers)
  (h6 : left_handed_non_throwers * 3 = non_throwers)
  : throwers + right_handed_non_throwers = 53 :=
sorry

end total_right_handed_players_l208_208311


namespace susan_spent_75_percent_l208_208544

variables (B b s : ℝ)

-- Conditions
def condition1 : Prop := b = 0.25 * (B - 3 * s)
def condition2 : Prop := s = 0.10 * (B - 2 * b)

-- Theorem
theorem susan_spent_75_percent (h1 : condition1 B b s) (h2 : condition2 B b s) : b + s = 0.75 * B := 
sorry

end susan_spent_75_percent_l208_208544


namespace associate_professors_bring_one_chart_l208_208049

theorem associate_professors_bring_one_chart
(A B C : ℕ) (h1 : 2 * A + B = 7) (h2 : A * C + 2 * B = 11) (h3 : A + B = 6) : C = 1 :=
by sorry

end associate_professors_bring_one_chart_l208_208049


namespace graph_fixed_point_l208_208782

theorem graph_fixed_point {a : ℝ} (h₁ : a > 0) (h₂ : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ ∀ x : ℝ, y = a^(x + 2) - 2 ↔ (x, y) = A := 
by 
  sorry

end graph_fixed_point_l208_208782


namespace steve_total_money_l208_208408

theorem steve_total_money
    (nickels : ℕ)
    (dimes : ℕ)
    (nickel_value : ℕ := 5)
    (dime_value : ℕ := 10)
    (cond1 : nickels = 2)
    (cond2 : dimes = nickels + 4) 
    : (nickels * nickel_value + dimes * dime_value) = 70 := by
  sorry

end steve_total_money_l208_208408


namespace purple_chip_count_l208_208021

theorem purple_chip_count :
  ∃ (x : ℕ), (x > 5) ∧ (x < 11) ∧
  (∃ (blue green purple red : ℕ),
    (2^6) * (5^2) * 11 * 7 = (blue * 1) * (green * 5) * (purple * x) * (red * 11) ∧ purple = 1) :=
sorry

end purple_chip_count_l208_208021


namespace sequence_eq_third_term_l208_208875

theorem sequence_eq_third_term 
  (p : ℤ → ℤ)
  (a : ℕ → ℤ)
  (n : ℕ) (h₁ : n > 2)
  (h₂ : a 2 = p (a 1))
  (h₃ : a 3 = p (a 2))
  (h₄ : ∀ k, 4 ≤ k ∧ k ≤ n → a k = p (a (k - 1)))
  (h₅ : a 1 = p (a n))
  : a 1 = a 3 :=
sorry

end sequence_eq_third_term_l208_208875


namespace music_tool_cost_l208_208519

noncomputable def flute_cost : ℝ := 142.46
noncomputable def song_book_cost : ℝ := 7
noncomputable def total_spent : ℝ := 158.35

theorem music_tool_cost :
    total_spent - (flute_cost + song_book_cost) = 8.89 :=
by
  sorry

end music_tool_cost_l208_208519


namespace solve_system_of_inequalities_l208_208135

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l208_208135


namespace speed_of_stream_l208_208723

def boatSpeedDownstream (V_b V_s : ℝ) : ℝ :=
  V_b + V_s

def boatSpeedUpstream (V_b V_s : ℝ) : ℝ :=
  V_b - V_s

theorem speed_of_stream (V_b V_s : ℝ) (h1 : V_b + V_s = 25) (h2 : V_b - V_s = 5) : V_s = 10 :=
by {
  sorry
}

end speed_of_stream_l208_208723


namespace interest_rate_second_part_l208_208983

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end interest_rate_second_part_l208_208983


namespace total_jokes_l208_208816

theorem total_jokes (jessy_jokes_saturday : ℕ) (alan_jokes_saturday : ℕ) 
  (jessy_next_saturday : ℕ) (alan_next_saturday : ℕ) (total_jokes_so_far : ℕ) :
  jessy_jokes_saturday = 11 → 
  alan_jokes_saturday = 7 → 
  jessy_next_saturday = 11 * 2 → 
  alan_next_saturday = 7 * 2 → 
  total_jokes_so_far = (jessy_jokes_saturday + alan_jokes_saturday) + (jessy_next_saturday + alan_next_saturday) → 
  total_jokes_so_far = 54 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_jokes_l208_208816


namespace quadratic_completes_square_l208_208167

theorem quadratic_completes_square (b c : ℤ) :
  (∃ b c : ℤ, (∀ x : ℤ, x^2 - 12 * x + 49 = (x + b)^2 + c) ∧ b + c = 7) :=
sorry

end quadratic_completes_square_l208_208167


namespace find_original_number_l208_208718

theorem find_original_number (x : ℝ) (hx : 0 < x) 
  (h : 1000 * x = 3 * (1 / x)) : x ≈ 0.01732 :=
by 
  sorry

end find_original_number_l208_208718


namespace monotone_decreasing_on_1_infty_extremum_on_2_6_l208_208491

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Part (1): Monotonicity of f(x) on (1,+∞)
theorem monotone_decreasing_on_1_infty : ∀ x1 x2 : ℝ, 1 < x1 → 1 < x2 → x1 < x2 → f x1 > f x2 := by
  sorry

-- Part (2): Extremum of f(x) on [2,6]
theorem extremum_on_2_6 :
  (∀ x ∈ Icc 2 6, f 2 ≥ f x ∧ f x ≥ f 6) := by
  sorry

#check f
#check monotone_decreasing_on_1_infty
#check extremum_on_2_6

end monotone_decreasing_on_1_infty_extremum_on_2_6_l208_208491


namespace ratio_of_perimeters_l208_208148

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l208_208148


namespace athlete_D_is_selected_l208_208173

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected_l208_208173


namespace parallel_line_through_point_l208_208997

theorem parallel_line_through_point (x y : ℝ) (m b : ℝ) (h₁ : y = -3 * x + b) (h₂ : x = 2) (h₃ : y = 1) :
  b = 7 :=
by
  -- x, y are components of the point P (2,1)
  -- equation of line parallel to y = -3x + 2 has slope -3 but different y-intercept
  -- y = -3x + b is the general form, and must pass through (2,1) => 1 = -3*2 + b
  -- Therefore, b must be 7
  sorry

end parallel_line_through_point_l208_208997


namespace rate_of_rainfall_is_one_l208_208382

variable (R : ℝ)
variable (h1 : 2 + 4 * R + 4 * 3 = 18)

theorem rate_of_rainfall_is_one : R = 1 :=
by
  sorry

end rate_of_rainfall_is_one_l208_208382


namespace complex_number_first_quadrant_l208_208995

theorem complex_number_first_quadrant (z : ℂ) (h : z = (i - 1) / i) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y > 0 := 
sorry

end complex_number_first_quadrant_l208_208995


namespace sin_inequality_iff_angle_inequality_l208_208744

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end sin_inequality_iff_angle_inequality_l208_208744


namespace card_combination_exists_l208_208833

theorem card_combination_exists : ∃ (R B G : ℕ), 
  R + B + G = 20 ∧ 2 ≤ R ∧ 3 ≤ B ∧ 1 ≤ G ∧ 3*R + 5*B + 7*G = 84 := 
by
  use 16, 3, 1
  simp [Nat.add_comm] -- simplifying helps to demonstrate the example solution.
  split; norm_num
  split; norm_num
  split; norm_num
  simp [Nat.add_comm, Nat.mul_comm]
  split; norm_num
  simp [Nat.add_comm, Nat.mul_comm]
sorry

end card_combination_exists_l208_208833


namespace shortest_side_of_similar_triangle_l208_208325

theorem shortest_side_of_similar_triangle (h1 : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (h2 : 15^2 + b^2 = 34^2) (h3 : ∃ (k : ℝ), k = 68 / 34) :
  ∃ s : ℝ, s = 2 * Real.sqrt 931 :=
by
  sorry

end shortest_side_of_similar_triangle_l208_208325


namespace total_kids_in_camp_l208_208175

-- Definitions from the conditions
variables (X : ℕ)
def kids_going_to_soccer_camp := X / 2
def kids_going_to_soccer_camp_morning := kids_going_to_soccer_camp / 4
def kids_going_to_soccer_camp_afternoon := kids_going_to_soccer_camp - kids_going_to_soccer_camp_morning

-- Given condition that 750 kids are going to soccer camp in the afternoon
axiom h : kids_going_to_soccer_camp_afternoon X = 750

-- The statement to prove that X = 2000
theorem total_kids_in_camp : X = 2000 :=
sorry

end total_kids_in_camp_l208_208175


namespace polynomial_multiplication_l208_208341

theorem polynomial_multiplication :
  (5 * X^2 + 3 * X - 4) * (2 * X^3 + X^2 - X + 1) = 
  (10 * X^5 + 11 * X^4 - 10 * X^3 - 2 * X^2 + 7 * X - 4) := 
by {
  sorry
}

end polynomial_multiplication_l208_208341


namespace max_min_diff_eq_four_l208_208774

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * |x - a|

theorem max_min_diff_eq_four (a : ℝ) (h_a : a ≥ 2) : 
    let M := max (f a (-1)) (f a 1)
    let m := min (f a (-1)) (f a 1)
    M - m = 4 :=
by
  sorry

end max_min_diff_eq_four_l208_208774


namespace maximize_takehome_pay_l208_208090

noncomputable def tax_initial (income : ℝ) : ℝ :=
  if income ≤ 20000 then 0.10 * income else 2000 + 0.05 * ((income - 20000) / 10000) * income

noncomputable def tax_beyond (income : ℝ) : ℝ :=
  (income - 20000) * ((0.005 * ((income - 20000) / 10000)) * income)

noncomputable def tax_total (income : ℝ) : ℝ :=
  if income ≤ 20000 then tax_initial income else tax_initial 20000 + tax_beyond income

noncomputable def takehome_pay_function (income : ℝ) : ℝ :=
  income - tax_total income

theorem maximize_takehome_pay : ∃ x, takehome_pay_function x = takehome_pay_function 30000 := 
sorry

end maximize_takehome_pay_l208_208090


namespace cuboid_volume_l208_208185

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by
  sorry

end cuboid_volume_l208_208185


namespace years_ago_twice_age_l208_208421

variables (H J x : ℕ)

def henry_age : ℕ := 20
def jill_age : ℕ := 13

axiom age_sum : H + J = 33
axiom age_difference : H - x = 2 * (J - x)

theorem years_ago_twice_age (H := henry_age) (J := jill_age) : x = 6 :=
by sorry

end years_ago_twice_age_l208_208421


namespace problem_intersecting_lines_l208_208855

theorem problem_intersecting_lines (c d : ℝ) :
  (3 : ℝ) = (1 / 3 : ℝ) * (6 : ℝ) + c ∧ (6 : ℝ) = (1 / 3 : ℝ) * (3 : ℝ) + d → c + d = 6 :=
by
  intros h
  sorry

end problem_intersecting_lines_l208_208855


namespace num_of_ordered_pairs_l208_208039

theorem num_of_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a)
(h4 : (a-2)*(b-2) = (ab / 2)) : (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
by
  sorry

end num_of_ordered_pairs_l208_208039


namespace perimeter_ratio_of_squares_l208_208140

theorem perimeter_ratio_of_squares (a b : ℝ) (ha : a = 49) (hb : b = 64) :
  real.sqrt a / real.sqrt b = 7 / 8 :=
by
  rw [ha, hb]
  calc
  real.sqrt 49 / real.sqrt 64 = 7 / 8 : sorry

end perimeter_ratio_of_squares_l208_208140


namespace original_number_solution_l208_208719

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end original_number_solution_l208_208719


namespace problem_statement_l208_208873

-- Define what it means to be a quadratic equation
def is_quadratic (eqn : String) : Prop :=
  -- In the context of this solution, we'll define a quadratic equation as one
  -- that fits the form ax^2 + bx + c = 0 where a, b, c are constants and a ≠ 0.
  eqn = "x^2 - 2 = 0"

-- We need to formulate a theorem that checks the validity of which equation is quadratic.
theorem problem_statement :
  is_quadratic "x^2 - 2 = 0" :=
sorry

end problem_statement_l208_208873


namespace wholesale_price_l208_208456

theorem wholesale_price (R W : ℝ) (h1 : R = 1.80 * W) (h2 : R = 36) : W = 20 :=
by
  sorry 

end wholesale_price_l208_208456


namespace david_money_left_l208_208054

theorem david_money_left (S : ℤ) (h1 : S - 800 = 1800 - S) : 1800 - S = 500 :=
by
  sorry

end david_money_left_l208_208054


namespace diff_cubes_square_of_squares_l208_208534

theorem diff_cubes_square_of_squares {x y : ℤ} (h1 : (x + 1) ^ 3 - x ^ 3 = y ^ 2) :
  ∃ (a b : ℤ), y = a ^ 2 + b ^ 2 ∧ a = b + 1 :=
sorry

end diff_cubes_square_of_squares_l208_208534


namespace shorter_leg_in_right_triangle_l208_208370

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end shorter_leg_in_right_triangle_l208_208370


namespace rectangle_perimeter_l208_208037

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b) = 4 * (2 * a + 2 * b) - 12) :
    (2 * (a + b) = 72) ∨ (2 * (a + b) = 100) := by
  sorry

end rectangle_perimeter_l208_208037


namespace incorrect_inequality_l208_208230

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ∃ c : ℝ, ac = bc :=
by
  have h1 : ¬(ac > bc) := by
    let c := 0
    show ac = bc 
    sorry

  exact ⟨0, h1⟩

end incorrect_inequality_l208_208230


namespace parallelogram_coincides_l208_208462

-- Define the shapes
inductive Shape
| Parallelogram
| EquilateralTriangle
| IsoscelesRightTriangle
| RegularPentagon

open Shape

-- Define the property of coinciding with itself after 180 degrees rotation
def coincides_after_180_degrees_rotation (shape : Shape) : Prop :=
  match shape with
  | Parallelogram => true
  | EquilateralTriangle => false
  | IsoscelesRightTriangle => false
  | RegularPentagon => false

-- The theorem stating the Parallelogram coincides with itself after 180 degrees rotation
theorem parallelogram_coincides : coincides_after_180_degrees_rotation Parallelogram :=
  by
    exact true.intro

end parallelogram_coincides_l208_208462


namespace sufficient_but_not_necessary_condition_l208_208498

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

-- This definition states that both f and g are either odd or even functions
def is_odd_or_even (f g : ℝ → ℝ) : Prop := 
  (is_odd f ∧ is_odd g) ∨ (is_even f ∧ is_even g)

theorem sufficient_but_not_necessary_condition (f g : ℝ → ℝ)
  (h : is_odd_or_even f g) : 
  ¬(is_odd f ∧ is_odd g) → is_even_function (f * g) :=
sorry

end sufficient_but_not_necessary_condition_l208_208498


namespace distinctDiagonalsConvexNonagon_l208_208495

theorem distinctDiagonalsConvexNonagon : 
  ∀ (P : Type) [fintype P] [decidable_eq P] (vertices : finset P) (h : vertices.card = 9), 
  let n := vertices.card in
  let diagonals := (n * (n - 3)) / 2 in
  diagonals = 27 :=
by
  intros
  let n := vertices.card
  have keyIdentity : (n * (n - 3)) / 2 = 27 := sorry
  exact keyIdentity

end distinctDiagonalsConvexNonagon_l208_208495


namespace ratio_of_investments_l208_208725

variable (A B C : ℝ) (k : ℝ)

-- Conditions
def investments_ratio := (6 * k + 5 * k + 4 * k = 7250) ∧ (5 * k - 6 * k = 250)

-- Theorem we need to prove
theorem ratio_of_investments (h : investments_ratio k) : (A / B = 6 / 5) ∧ (B / C = 5 / 4) := 
  sorry

end ratio_of_investments_l208_208725


namespace find_smaller_number_l208_208420

-- Define the conditions
def sum_of_numbers (x y : ℕ) := x + y = 70
def second_number_relation (x y : ℕ) := y = 3 * x + 10

-- Define the problem statement
theorem find_smaller_number (x y : ℕ) (h1 : sum_of_numbers x y) (h2 : second_number_relation x y) : x = 15 :=
sorry

end find_smaller_number_l208_208420


namespace part_a_part_b_l208_208668

variable {A : Type*} [Ring A]

def B (A : Type*) [Ring A] : Set A :=
  {a | a^2 = 1}

variable (a : A) (b : B A)

theorem part_a (a : A) (b : A) (h : b ∈ B A) : a * b - b * a = b * a * b - a := by
  sorry

theorem part_b (A : Type*) [Ring A] (h : ∀ x : A, x^2 = 0 -> x = 0) : Group (B A) := by
  sorry

end part_a_part_b_l208_208668


namespace total_jokes_proof_l208_208818

-- Definitions of the conditions
def jokes_jessy_last_saturday : Nat := 11
def jokes_alan_last_saturday : Nat := 7
def jokes_jessy_next_saturday : Nat := 2 * jokes_jessy_last_saturday
def jokes_alan_next_saturday : Nat := 2 * jokes_alan_last_saturday

-- Sum of jokes over two Saturdays
def total_jokes : Nat := (jokes_jessy_last_saturday + jokes_alan_last_saturday) + (jokes_jessy_next_saturday + jokes_alan_next_saturday)

-- The proof problem
theorem total_jokes_proof : total_jokes = 54 := 
by
  sorry

end total_jokes_proof_l208_208818


namespace cheese_bread_grams_l208_208025

/-- Each 100 grams of cheese bread costs 3.20 BRL and corresponds to 10 pieces. 
Each person eats, on average, 5 pieces of cheese bread. Including the professor,
there are 16 students, 1 monitor, and 5 parents, making a total of 23 people. 
The precision of the bakery's scale is 100 grams. -/
theorem cheese_bread_grams : (5 * 23 / 10) * 100 = 1200 := 
by
  sorry

end cheese_bread_grams_l208_208025


namespace maximum_value_minimum_value_l208_208070

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def check_digits (N M : ℕ) (a b c d e f g h : ℕ) : Prop :=
  N = 1000 * a + 100 * b + 10 * c + d ∧
  M = 1000 * e + 100 * f + 10 * g + h ∧
  a ≠ e ∧
  b ≠ f ∧
  c ≠ g ∧
  d ≠ h ∧
  a ≠ f ∧
  a ≠ g ∧
  a ≠ h ∧
  b ≠ e ∧
  b ≠ g ∧
  b ≠ h ∧
  c ≠ e ∧
  c ≠ f ∧
  c ≠ h ∧
  d ≠ e ∧
  d ≠ f ∧
  d ≠ g

theorem maximum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 15000 :=
by
  intros
  sorry

theorem minimum_value (N M a b c d e f g h : ℕ) :
  is_four_digit_number N →
  is_four_digit_number M →
  check_digits N M a b c d e f g h →
  N - M = 1994 →
  N + M = 4998 :=
by
  intros
  sorry

end maximum_value_minimum_value_l208_208070


namespace john_ingrid_combined_weighted_average_tax_rate_l208_208099

noncomputable def john_employment_income : ℕ := 57000
noncomputable def john_employment_tax_rate : ℚ := 0.30
noncomputable def john_rental_income : ℕ := 11000
noncomputable def john_rental_tax_rate : ℚ := 0.25

noncomputable def ingrid_employment_income : ℕ := 72000
noncomputable def ingrid_employment_tax_rate : ℚ := 0.40
noncomputable def ingrid_investment_income : ℕ := 4500
noncomputable def ingrid_investment_tax_rate : ℚ := 0.15

noncomputable def combined_weighted_average_tax_rate : ℚ :=
  let john_total_tax := john_employment_income * john_employment_tax_rate + john_rental_income * john_rental_tax_rate
  let john_total_income := john_employment_income + john_rental_income
  let ingrid_total_tax := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate
  let ingrid_total_income := ingrid_employment_income + ingrid_investment_income
  let combined_total_tax := john_total_tax + ingrid_total_tax
  let combined_total_income := john_total_income + ingrid_total_income
  (combined_total_tax / combined_total_income) * 100

theorem john_ingrid_combined_weighted_average_tax_rate :
  combined_weighted_average_tax_rate = 34.14 := by
  sorry

end john_ingrid_combined_weighted_average_tax_rate_l208_208099


namespace sqrt_diff_eq_neg_sixteen_l208_208335

theorem sqrt_diff_eq_neg_sixteen : 
  (Real.sqrt (16 - 8 * Real.sqrt 2) - Real.sqrt (16 + 8 * Real.sqrt 2)) = -16 := 
  sorry

end sqrt_diff_eq_neg_sixteen_l208_208335


namespace animal_costs_l208_208587

theorem animal_costs (S K L : ℕ) (h1 : K = 4 * S) (h2 : L = 4 * K) (h3 : S + 2 * K + L = 200) :
  S = 8 ∧ K = 32 ∧ L = 128 :=
by
  sorry

end animal_costs_l208_208587


namespace translation_graph_pass_through_point_l208_208512

theorem translation_graph_pass_through_point :
  (∃ a : ℝ, (∀ x y : ℝ, y = -2 * x + 1 - 3 → y = 3 → x = a) → a = -5/2) :=
sorry

end translation_graph_pass_through_point_l208_208512


namespace james_gave_away_one_bag_l208_208098

theorem james_gave_away_one_bag (initial_marbles : ℕ) (bags : ℕ) (marbles_left : ℕ) (h1 : initial_marbles = 28) (h2 : bags = 4) (h3 : marbles_left = 21) : (initial_marbles / bags) = (initial_marbles - marbles_left) / (initial_marbles / bags) :=
by
  sorry

end james_gave_away_one_bag_l208_208098


namespace simplify_fraction_l208_208999

theorem simplify_fraction :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 :=
by
  sorry

end simplify_fraction_l208_208999


namespace geometric_sequence_q_l208_208955

theorem geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 4 + a 8 = 8) :
  q = 1 :=
by
  sorry

end geometric_sequence_q_l208_208955


namespace borrowing_methods_count_l208_208168

open Finset Nat

-- Definitions for the problem
def physics_books : ℕ := 3
def history_books : ℕ := 2
def mathematics_books : ℕ := 4
def science_students : ℕ := 4
def liberal_arts_students : ℕ := 3

-- Statement of the problem
theorem borrowing_methods_count :
  let category1 := (choose liberal_arts_students 1) * ((choose mathematics_books 1) + (choose mathematics_books 2) + (choose mathematics_books 3))
  let category2 := (choose liberal_arts_students 2) * ((choose mathematics_books 1) + (choose mathematics_books 2))
  let category3 := (choose liberal_arts_students 3) * (choose mathematics_books 1)
  category1 + category2 + category3 = 76 :=
by
  sorry

end borrowing_methods_count_l208_208168


namespace boys_girls_rel_l208_208510

theorem boys_girls_rel (b g : ℕ) (h : g = 7 + 2 * (b - 1)) : b = (g - 5) / 2 := 
by sorry

end boys_girls_rel_l208_208510


namespace unique_root_of_quadratic_eq_l208_208858

theorem unique_root_of_quadratic_eq (a b c : ℝ) (d : ℝ) 
  (h_seq : b = a - d ∧ c = a - 2 * d) 
  (h_nonneg : a ≥ b ∧ b ≥ c ∧ c ≥ 0) 
  (h_discriminant : (-(a - d))^2 - 4 * a * (a - 2 * d) = 0) :
  ∃ x : ℝ, (ax^2 - bx + c = 0) ∧ x = 1 / 2 :=
by
  sorry

end unique_root_of_quadratic_eq_l208_208858


namespace arithmetic_sequence_sum_l208_208380

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 1)
  (h2 : a 2 + a 3 = 9) :
  a 4 + a 5 = 17 :=
sorry

end arithmetic_sequence_sum_l208_208380


namespace true_for_2_and_5_l208_208752

theorem true_for_2_and_5 (x : ℝ) : ((x - 2) * (x - 5) = 0) ↔ (x = 2 ∨ x = 5) :=
by
  sorry

end true_for_2_and_5_l208_208752


namespace tan_diff_eq_sqrt_three_l208_208615

open Real

theorem tan_diff_eq_sqrt_three (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : cos α * cos β = 1 / 6) (h5 : sin α * sin β = 1 / 3) : 
  tan (β - α) = sqrt 3 := by
  sorry

end tan_diff_eq_sqrt_three_l208_208615


namespace apples_total_l208_208911

theorem apples_total :
  ∀ (Marin David Amanda : ℕ),
  Marin = 6 →
  David = 2 * Marin →
  Amanda = David + 5 →
  Marin + David + Amanda = 35 :=
by
  intros Marin David Amanda hMarin hDavid hAmanda
  sorry

end apples_total_l208_208911


namespace original_number_l208_208714

theorem original_number :
  ∃ x : ℝ, 0 < x ∧ (move_decimal_point x 3 = 3 / x) ∧ x = sqrt 30 / 100 :=
sorry

noncomputable def move_decimal_point (x : ℝ) (places : ℕ) : ℝ := x * 10^places

end original_number_l208_208714


namespace problem_1_problem_2_l208_208340

noncomputable def poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 := sorry

theorem problem_1 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  a₁ + a₂ + a₃ + a₄ = -80 :=
sorry

theorem problem_2 (poly_expansion : (2 * x - 3) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) :
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 625 :=
sorry

end problem_1_problem_2_l208_208340


namespace any_nat_representation_as_fraction_l208_208869

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l208_208869


namespace monotonic_increasing_condition_l208_208292

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → y a x₁ ≤ y a x₂) ↔ 
  (a = 0 ∨ a > 0) :=
sorry

end monotonic_increasing_condition_l208_208292


namespace savings_by_paying_cash_l208_208828

theorem savings_by_paying_cash
  (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (number_of_months : ℕ)
  (h1 : cash_price = 400) (h2 : down_payment = 120) (h3 : monthly_payment = 30) (h4 : number_of_months = 12) :
  cash_price + (monthly_payment * number_of_months - down_payment) - cash_price = 80 :=
by
  sorry

end savings_by_paying_cash_l208_208828


namespace smallest_part_proportional_l208_208082

/-- If we divide 124 into three parts proportional to 2, 1/2, and 1/4,
    prove that the smallest part is 124 / 11. -/
theorem smallest_part_proportional (x : ℝ) 
  (h : 2 * x + (1 / 2) * x + (1 / 4) * x = 124) : 
  (1 / 4) * x = 124 / 11 :=
sorry

end smallest_part_proportional_l208_208082


namespace weight_of_new_person_l208_208291

-- Definition of the problem
def average_weight_increases (W : ℝ) (N : ℝ) : Prop :=
  let increase := 2.5
  W - 45 + N = W + 8 * increase

-- The main statement we need to prove
theorem weight_of_new_person (W : ℝ) : ∃ N, average_weight_increases W N ∧ N = 65 := 
by
  use 65
  unfold average_weight_increases
  sorry

end weight_of_new_person_l208_208291


namespace smallest_q_p_difference_l208_208110

theorem smallest_q_p_difference :
  ∃ (p q : ℕ), 
  (3 : ℚ) / 5 < p / q ∧ p / q < (5 : ℚ) / 8 ∧
  ∀ (r : ℕ), (3 : ℚ) / 5 < r / q ∧ r / q < (5 : ℚ) / 8 → p = r ∧ q = 13 →
  q - p = 5 :=
by {
  -- proof goes here
  sorry
}

end smallest_q_p_difference_l208_208110


namespace hot_dogs_total_l208_208208

theorem hot_dogs_total (D : ℕ)
  (h1 : 9 = 2 * D + D + 3) :
  (2 * D + 9 + D = 15) :=
by sorry

end hot_dogs_total_l208_208208


namespace max_consecutive_integers_sum_lt_1000_l208_208009

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l208_208009


namespace union_of_P_and_Q_l208_208065

noncomputable def P : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_of_P_and_Q :
  P ∪ Q = {x | -1 < x ∧ x < 2} :=
sorry

end union_of_P_and_Q_l208_208065


namespace max_consecutive_sum_lt_1000_l208_208005

theorem max_consecutive_sum_lt_1000 : ∃ (n : ℕ), (∀ (m : ℕ), m > n → (m * (m + 1)) / 2 ≥ 1000) ∧ (∀ (k : ℕ), k ≤ n → (k * (k + 1)) / 2 < 1000) :=
begin
  sorry,
end

end max_consecutive_sum_lt_1000_l208_208005


namespace successive_product_4160_l208_208438

theorem successive_product_4160 (n : ℕ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_4160_l208_208438


namespace min_ab_eq_4_l208_208637

theorem min_ab_eq_4 (a b : ℝ) (h : 4 / a + 1 / b = Real.sqrt (a * b)) : a * b ≥ 4 :=
sorry

end min_ab_eq_4_l208_208637


namespace trapezoid_perimeter_area_sum_l208_208482

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

noncomputable def perimeter (vertices : List (Real × Real)) : Real :=
  match vertices with
  | [a, b, c, d] => (distance a b) + (distance b c) + (distance c d) + (distance d a)
  | _ => 0

noncomputable def area_trapezoid (b1 b2 h : Real) : Real :=
  0.5 * (b1 + b2) * h

theorem trapezoid_perimeter_area_sum
  (A B C D : Real × Real)
  (h_AB : A = (2, 3))
  (h_BC : B = (7, 3))
  (h_CD : C = (9, 7))
  (h_DA : D = (0, 7)) :
  let perimeter := perimeter [A, B, C, D]
  let area := area_trapezoid (distance C D) (distance A B) (C.2 - B.2)
  perimeter + area = 42 + 4 * Real.sqrt 5 :=
by
  sorry

end trapezoid_perimeter_area_sum_l208_208482


namespace necessary_and_sufficient_condition_l208_208526

variables (x y : ℝ)

theorem necessary_and_sufficient_condition (h1 : x > y) (h2 : 1/x > 1/y) : x * y < 0 :=
sorry

end necessary_and_sufficient_condition_l208_208526


namespace dog_food_bags_needed_l208_208574

theorem dog_food_bags_needed
  (cup_weight: ℝ)
  (dogs: ℕ)
  (cups_per_day: ℕ)
  (days_in_month: ℕ)
  (bag_weight: ℝ)
  (hcw: cup_weight = 1/4)
  (hd: dogs = 2)
  (hcd: cups_per_day = 6 * 2)
  (hdm: days_in_month = 30)
  (hbw: bag_weight = 20) :
  (dogs * cups_per_day * days_in_month * cup_weight) / bag_weight = 9 :=
by
  sorry

end dog_food_bags_needed_l208_208574


namespace probability_of_one_machine_maintenance_l208_208863

theorem probability_of_one_machine_maintenance :
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444 :=
by {
  let pA := 0.1
  let pB := 0.2
  let pC := 0.4
  let qA := 1 - pA
  let qB := 1 - pB
  let qC := 1 - pC
  show (pA * qB * qC) + (qA * pB * qC) + (qA * qB * pC) = 0.444
  sorry
}

end probability_of_one_machine_maintenance_l208_208863


namespace correct_idiom_l208_208561

-- Define the conditions given in the problem
def context := "The vast majority of office clerks read a significant amount of materials"
def idiom_usage := "to say _ of additional materials"

-- Define the proof problem
theorem correct_idiom (context: String) (idiom_usage: String) : idiom_usage.replace "_ of additional materials" "nothing of newspapers and magazines" = "to say nothing of newspapers and magazines" :=
sorry

end correct_idiom_l208_208561


namespace prob_both_even_correct_l208_208022

-- Define the dice and verify their properties
def die1 := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def die2 := {n : ℕ // n ≥ 1 ∧ n ≤ 7}

-- Define the sets of even numbers for both dice
def even_die1 (n : die1) : Prop := n.1 % 2 = 0
def even_die2 (n : die2) : Prop := n.1 % 2 = 0

-- Define the probabilities of rolling an even number on each die
def prob_even_die1 := 3 / 6
def prob_even_die2 := 3 / 7

-- Calculate the combined probability
def prob_both_even := prob_even_die1 * prob_even_die2

-- The theorem stating the probability of both dice rolling even is 3/14
theorem prob_both_even_correct : prob_both_even = 3 / 14 :=
by
  -- Proof is omitted
  sorry

end prob_both_even_correct_l208_208022


namespace sum_of_ages_3_years_hence_l208_208682

theorem sum_of_ages_3_years_hence (A B C D S : ℝ) (h1 : A = 2 * B) (h2 : C = A / 2) (h3 : D = A - C) (h_sum : A + B + C + D = S) : 
  (A + 3) + (B + 3) + (C + 3) + (D + 3) = S + 12 :=
by sorry

end sum_of_ages_3_years_hence_l208_208682


namespace triangle_centroid_property_l208_208254

def distance_sq (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem triangle_centroid_property
  (A B C P : ℝ × ℝ)
  (G : ℝ × ℝ)
  (hG : G = ( (A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3 )) :
  distance_sq A P + distance_sq B P + distance_sq C P = 
  distance_sq A G + distance_sq B G + distance_sq C G + 3 * distance_sq G P :=
by
  sorry

end triangle_centroid_property_l208_208254


namespace simplify_fraction_l208_208278

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208278


namespace trash_can_ratio_l208_208458

theorem trash_can_ratio (streets_trash_cans total_trash_cans : ℕ) 
(h_streets : streets_trash_cans = 14) 
(h_total : total_trash_cans = 42) : 
(total_trash_cans - streets_trash_cans) / streets_trash_cans = 2 :=
by {
  sorry
}

end trash_can_ratio_l208_208458


namespace max_consecutive_sum_le_1000_l208_208003

theorem max_consecutive_sum_le_1000 : 
  ∃ (n : ℕ), (∀ m : ℕ, m > n → ∑ k in finset.range (m + 1), k > 1000) ∧
             ∑ k in finset.range (n + 1), k ≤ 1000 :=
by
  sorry

end max_consecutive_sum_le_1000_l208_208003


namespace interest_rate_proven_l208_208984

structure InvestmentProblem where
  P : ℝ  -- Principal amount
  A : ℝ  -- Accumulated amount
  n : ℕ  -- Number of times interest is compounded per year
  t : ℕ  -- Time in years
  rate : ℝ  -- Interest rate per annum (to be proven)

noncomputable def solve_interest_rate (ip : InvestmentProblem) : ℝ :=
  let half_yearly_rate := ip.rate / 2 / 100
  let amount_formula := ip.P * (1 + half_yearly_rate)^(ip.n * ip.t)
  half_yearly_rate

theorem interest_rate_proven :
  ∀ (P A : ℝ) (n t : ℕ), 
  P = 6000 → 
  A = 6615 → 
  n = 2 → 
  t = 1 → 
  solve_interest_rate {P := P, A := A, n := n, t := t, rate := 10.0952} = 10.0952 := 
by 
  intros
  rw [solve_interest_rate]
  sorry

end interest_rate_proven_l208_208984


namespace simplify_fraction_l208_208272

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := 
by
  sorry

end simplify_fraction_l208_208272


namespace rectangle_length_width_ratio_l208_208764

-- Define the side lengths of the small squares and the large square
variables (s : ℝ)

-- Define the dimensions of the large square and the rectangle
def large_square_side : ℝ := 5 * s
def rectangle_length : ℝ := 5 * s
def rectangle_width : ℝ := s

-- State and prove the theorem
theorem rectangle_length_width_ratio : rectangle_length s / rectangle_width s = 5 :=
by sorry

end rectangle_length_width_ratio_l208_208764


namespace max_point_h_l208_208551

-- Definitions of the linear functions f and g
def f (x : ℝ) : ℝ := 2 * x + 2
def g (x : ℝ) : ℝ := -x - 3

-- The product of f(x) and g(x)
def h (x : ℝ) : ℝ := f x * g x

-- Statement: Prove that x = -2 is the maximum point of h(x)
theorem max_point_h : ∃ x_max : ℝ, h x_max = (-2) :=
by
  -- skipping the proof
  sorry

end max_point_h_l208_208551


namespace perimeters_ratio_l208_208155

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l208_208155


namespace solve_for_F_l208_208355

theorem solve_for_F (F C : ℝ) (h₁ : C = 4 / 7 * (F - 40)) (h₂ : C = 25) : F = 83.75 :=
sorry

end solve_for_F_l208_208355


namespace shorter_leg_of_right_triangle_l208_208377

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l208_208377


namespace original_rectangle_area_l208_208199

-- Define the original rectangle sides, square side, and perimeters of rectangles adjacent to the square
variables {a b x : ℝ}
variable (h1 : a + x = 10)
variable (h2 : b + x = 8)

-- Define the area calculation
def area (a b : ℝ) := a * b

-- The area of the original rectangle should be 80 cm²
theorem original_rectangle_area : area (10 - x) (8 - x) = 80 := by
  sorry

end original_rectangle_area_l208_208199


namespace not_possible_2002_pieces_l208_208034

theorem not_possible_2002_pieces (k : ℤ) : ¬ (1 + 7 * k = 2002) :=
by
  sorry

end not_possible_2002_pieces_l208_208034


namespace alien_collected_95_units_l208_208745

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  match n with
  | 235 => 2 * 6^2 + 3 * 6^1 + 5 * 6^0
  | _ => 0

theorem alien_collected_95_units : convert_base_six_to_ten 235 = 95 := by
  sorry

end alien_collected_95_units_l208_208745


namespace right_triangle_short_leg_l208_208366

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l208_208366


namespace area_of_region_bounded_by_sec_and_csc_l208_208762

theorem area_of_region_bounded_by_sec_and_csc (x y : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 0 ≤ x ∧ 0 ≤ y) → 
  (∃ (area : ℝ), area = 1) :=
by 
  sorry

end area_of_region_bounded_by_sec_and_csc_l208_208762


namespace solve_abs_inequality_l208_208131

theorem solve_abs_inequality (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) ↔ 
      (x ∈ Ioo (-13 / 2) (-3) 
     ∨ x ∈ Ico (-3) 2 
     ∨ x ∈ Icc 2 (7 / 2)) :=
by
  sorry

end solve_abs_inequality_l208_208131


namespace batsman_average_l208_208432

variable (x : ℝ)

theorem batsman_average (h1 : ∀ x, 11 * x + 55 = 12 * (x + 1)) : 
  x = 43 → (x + 1 = 44) :=
by
  sorry

end batsman_average_l208_208432


namespace f_transform_l208_208243

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 4 * x - 5

theorem f_transform (x h : ℝ) : 
  f (x + h) - f x = 6 * x ^ 2 - 6 * x + 6 * x * h + 2 * h ^ 2 - 3 * h + 4 := 
by
  sorry

end f_transform_l208_208243


namespace solve_abs_inequality_l208_208922

theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |3 * x - 6| ∧ |3 * x - 6| ≤ 15 ↔ (-3 ≤ x ∧ x ≤ 4 / 3) ∨ (8 / 3 ≤ x ∧ x ≤ 7) := 
sorry

end solve_abs_inequality_l208_208922


namespace length_BF_l208_208567

-- Define the geometrical configuration
structure Point :=
  (x : ℝ) (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 6 4.8
def C := Point.mk 12 0
def D := Point.mk 3 (-6)
def E := Point.mk 3 0
def F := Point.mk 6 0

-- Define given conditions
def AE := (3 : ℝ)
def CE := (9 : ℝ)
def DE := (6 : ℝ)
def AC := AE + CE

theorem length_BF : (BF = (72 / 7 : ℝ)) :=
by
  sorry

end length_BF_l208_208567


namespace positive_difference_balances_l208_208750

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem positive_difference_balances :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 15
  let A_cedric := cedric_balance P r_cedric t
  let A_daniel := daniel_balance P r_daniel t
  (A_daniel - A_cedric) = 11632.65 :=
by
  sorry

end positive_difference_balances_l208_208750


namespace arithmetic_sequence_a15_l208_208807

theorem arithmetic_sequence_a15 {a : ℕ → ℝ} (d : ℝ) (a7 a23 : ℝ) 
    (h1 : a 7 = 8) (h2 : a 23 = 22) : 
    a 15 = 15 := 
by
  sorry

end arithmetic_sequence_a15_l208_208807


namespace probability_of_prime_sum_less_than_30_l208_208219

open scoped BigOperators

noncomputable def ten_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def is_prime_sum_less_than_30 (a b : ℕ) : Prop :=
  Nat.Prime (a + b) ∧ (a + b) < 30

theorem probability_of_prime_sum_less_than_30 :
  (ten_primes.card.choose 2) = 45 ∧
  ((ten_primes.filter (λ ab : ℕ × ℕ, is_prime_sum_less_than_30 ab.1 ab.2)).card : ℚ) / ten_primes.card.choose 2 = 4 / 45 :=
sorry

end probability_of_prime_sum_less_than_30_l208_208219


namespace diane_initial_amount_l208_208915

theorem diane_initial_amount
  (X : ℝ)        -- the amount Diane started with
  (won_amount : ℝ := 65)
  (total_loss : ℝ := 215)
  (owing_friends : ℝ := 50)
  (final_amount := X + won_amount - total_loss - owing_friends) :
  X = 100 := 
by 
  sorry

end diane_initial_amount_l208_208915


namespace xiaoning_comprehensive_score_l208_208443

theorem xiaoning_comprehensive_score
  (max_score : ℕ := 100)
  (midterm_weight : ℝ := 0.3)
  (final_weight : ℝ := 0.7)
  (midterm_score : ℕ := 80)
  (final_score : ℕ := 90) :
  (midterm_score * midterm_weight + final_score * final_weight) = 87 :=
by
  sorry

end xiaoning_comprehensive_score_l208_208443


namespace arithmetic_sequence_T_n_bound_l208_208483

open Nat

theorem arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h2 : a 2 = 6) (h3_h6 : a 3 + a 6 = 27) :
  (∀ n, a n = 3 * n) := 
by
  sorry

theorem T_n_bound (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) (m : ℝ) (h_general_term : ∀ n, a n = 3 * n) 
  (h_S_n : ∀ n, S n = n^2 + n) (h_T_n : ∀ n, T n = (S n : ℝ) / (3 * (2 : ℝ)^(n-1)))
  (h_bound : ∀ n > 0, T n ≤ m) : 
  m ≥ 3/2 :=
by
  sorry

end arithmetic_sequence_T_n_bound_l208_208483


namespace simplify_expression_l208_208986

theorem simplify_expression (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) (hx1 : x ≠ 1) : 
  (1 + 1 / (x - 2)) / ((x^2 - 2 * x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by
  sorry

end simplify_expression_l208_208986


namespace cos_angle_sum_eq_negative_sqrt_10_div_10_l208_208350

theorem cos_angle_sum_eq_negative_sqrt_10_div_10 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (α + π / 4) = - Real.sqrt 10 / 10 := by
  sorry

end cos_angle_sum_eq_negative_sqrt_10_div_10_l208_208350


namespace shared_vertex_probability_correct_m_n_sum_l208_208601

-- Define the grid and the properties
def side_length : ℕ := 5
def total_triangles : ℕ := 55 -- Sum of squares of first 5 natural numbers

-- Number of ways to choose 2 triangles out of total_triangles
def total_ways_to_choose_two : ℕ := (total_triangles * (total_triangles - 1)) / 2

-- Number of pairs that share exactly one vertex
def num_pairs_sharing_one_vertex : ℕ := 450 -- Calculated from cases in solution

-- Probability in simplified form as a rational number
def shared_vertex_probability : ℚ := num_pairs_sharing_one_vertex /. total_ways_to_choose_two

-- Fractions are relatively prime
def m : ℕ := 10
def n : ℕ := 33

theorem shared_vertex_probability_correct :
  shared_vertex_probability = 10 /. 33 := sorry

theorem m_n_sum :
  m + n = 43 := by
  exact rfl

end shared_vertex_probability_correct_m_n_sum_l208_208601


namespace rice_difference_on_15th_and_first_10_squares_l208_208263

-- Definitions
def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ := 
  (3 * (3^n - 1)) / (3 - 1)

-- Theorem statement
theorem rice_difference_on_15th_and_first_10_squares :
  grains_on_square 15 - sum_first_n_squares 10 = 14260335 :=
by
  sorry

end rice_difference_on_15th_and_first_10_squares_l208_208263


namespace proof_5x_plus_4_l208_208360

variable (x : ℝ)

-- Given condition
def condition := 5 * x - 8 = 15 * x + 12

-- Required proof
theorem proof_5x_plus_4 (h : condition x) : 5 * (x + 4) = 10 :=
by {
  sorry
}

end proof_5x_plus_4_l208_208360


namespace circle_area_l208_208247

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = 2 * r) : π * r^2 = 2 := by
  sorry

end circle_area_l208_208247


namespace place_b_left_of_a_forms_correct_number_l208_208083

noncomputable def form_three_digit_number (a b : ℕ) : ℕ :=
  100 * b + a

theorem place_b_left_of_a_forms_correct_number (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 1 ≤ b ∧ b < 10) :
  form_three_digit_number a b = 100 * b + a :=
by sorry

end place_b_left_of_a_forms_correct_number_l208_208083


namespace find_root_and_m_l208_208931

theorem find_root_and_m (x₁ m : ℝ) (h₁ : -2 * x₁ = 2) (h₂ : x^2 + m * x + 2 = 0) : x₁ = -1 ∧ m = 3 := 
by 
  -- Proof omitted
  sorry

end find_root_and_m_l208_208931


namespace find_original_number_l208_208712

noncomputable def original_number (x : ℝ) : Prop :=
  (0 < x) ∧ (1000 * x = 3 / x)

theorem find_original_number : ∃ x : ℝ, original_number x ∧ x = real.sqrt 30 / 100 :=
by
  sorry

end find_original_number_l208_208712


namespace vec_subtraction_l208_208076

variables (a b : Prod ℝ ℝ)
def vec1 : Prod ℝ ℝ := (1, 2)
def vec2 : Prod ℝ ℝ := (3, 1)

theorem vec_subtraction : (2 * (vec1.fst, vec1.snd) - (vec2.fst, vec2.snd)) = (-1, 3) := by
  -- Proof here, skipped
  sorry

end vec_subtraction_l208_208076


namespace apples_purchased_l208_208303

variable (A : ℕ) -- Let A be the number of kg of apples purchased.

-- Conditions
def cost_of_apples (A : ℕ) : ℕ := 70 * A
def cost_of_mangoes : ℕ := 45 * 9
def total_amount_paid : ℕ := 965

-- Theorem to prove that A == 8
theorem apples_purchased
  (h : cost_of_apples A + cost_of_mangoes = total_amount_paid) :
  A = 8 := by
sorry

end apples_purchased_l208_208303


namespace x1_x2_product_l208_208792

theorem x1_x2_product (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1^2 - 2006 * x1 = 1) (h3 : x2^2 - 2006 * x2 = 1) : x1 * x2 = -1 := 
by
  sorry

end x1_x2_product_l208_208792


namespace Jessie_points_l208_208845

theorem Jessie_points (total_points team_points : ℕ) (players_points : ℕ) (P Q R : ℕ) (eq1 : total_points = 311) (eq2 : players_points = 188) (eq3 : team_points - players_points = 3 * P) (eq4 : P = Q) (eq5 : Q = R) : Q = 41 :=
by
  sorry

end Jessie_points_l208_208845


namespace space_diagonal_of_prism_l208_208240

theorem space_diagonal_of_prism (l w h : ℝ) (hl : l = 2) (hw : w = 3) (hh : h = 4) :
  (l ^ 2 + w ^ 2 + h ^ 2).sqrt = Real.sqrt 29 :=
by
  rw [hl, hw, hh]
  sorry

end space_diagonal_of_prism_l208_208240


namespace represent_380000_in_scientific_notation_l208_208411

theorem represent_380000_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 380000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.8 ∧ n = 5 :=
by
  sorry

end represent_380000_in_scientific_notation_l208_208411


namespace sum_difference_4041_l208_208765

def sum_of_first_n_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_difference_4041 :
  sum_of_first_n_integers 2021 - sum_of_first_n_integers 2019 = 4041 :=
by
  sorry

end sum_difference_4041_l208_208765


namespace tire_usage_l208_208733

theorem tire_usage (total_distance : ℕ) (num_tires : ℕ) (active_tires : ℕ) 
  (h1 : total_distance = 45000) 
  (h2 : num_tires = 5) 
  (h3 : active_tires = 4) 
  (equal_usage : (total_distance * active_tires) / num_tires = 36000) : 
  (∀ tire, tire < num_tires → used_miles_per_tire = 36000) := 
by
  sorry

end tire_usage_l208_208733


namespace glass_panels_in_neighborhood_l208_208323

def total_glass_panels_in_neighborhood := 
  let double_windows_downstairs : ℕ := 6
  let glass_panels_per_double_window_downstairs : ℕ := 4
  let single_windows_upstairs : ℕ := 8
  let glass_panels_per_single_window_upstairs : ℕ := 3
  let bay_windows : ℕ := 2
  let glass_panels_per_bay_window : ℕ := 6
  let houses : ℕ := 10

  let glass_panels_in_one_house : ℕ := 
    (double_windows_downstairs * glass_panels_per_double_window_downstairs) +
    (single_windows_upstairs * glass_panels_per_single_window_upstairs) +
    (bay_windows * glass_panels_per_bay_window)

  houses * glass_panels_in_one_house

theorem glass_panels_in_neighborhood : total_glass_panels_in_neighborhood = 600 := by
  -- Calculation steps skipped
  sorry

end glass_panels_in_neighborhood_l208_208323


namespace simple_interest_sum_l208_208439

theorem simple_interest_sum (P_SI : ℕ) :
  let P_CI := 5000
  let r_CI := 12
  let t_CI := 2
  let r_SI := 10
  let t_SI := 5
  let CI := (P_CI * (1 + r_CI / 100)^t_CI - P_CI)
  let SI := CI / 2
  (P_SI * r_SI * t_SI / 100 = SI) -> 
  P_SI = 1272 := by {
  sorry
}

end simple_interest_sum_l208_208439


namespace collinear_points_sum_l208_208603

theorem collinear_points_sum (x y : ℝ) : 
  (∃ a b : ℝ, a * x + b * 3 + (1 - a - b) * 2 = a * x + b * y + (1 - a - b) * y ∧ 
               a * y + b * 4 + (1 - a - b) * y = a * x + b * y + (1 - a - b) * x) → 
  x = 2 → y = 4 → x + y = 6 :=
by sorry

end collinear_points_sum_l208_208603


namespace find_t_l208_208948

theorem find_t : ∀ (p j t x y a b c : ℝ),
  j = 0.75 * p →
  j = 0.80 * t →
  t = p - (t/100) * p →
  x = 0.10 * t →
  y = 0.50 * j →
  x + y = 12 →
  a = x + y →
  b = 0.15 * a →
  c = 2 * b →
  t = 24 := 
by
  intros p j t x y a b c hjp hjt htp hxt hyy hxy ha hb hc
  sorry

end find_t_l208_208948


namespace subway_train_speed_l208_208860

theorem subway_train_speed (s : ℕ) (h1 : 0 ≤ s ∧ s ≤ 7) (h2 : s^2 + 2*s = 63) : s = 7 :=
by
  sorry

end subway_train_speed_l208_208860


namespace infinite_series_sum_l208_208052

theorem infinite_series_sum :
  (∑' n : ℕ, if h : n ≠ 0 then 1 / (n * (n + 1) * (n + 3)) else 0) = 5 / 36 := by
  sorry

end infinite_series_sum_l208_208052


namespace base_number_is_two_l208_208633

theorem base_number_is_two (a : ℝ) (x : ℕ) (h1 : x = 14) (h2 : a^x - a^(x - 2) = 3 * a^12) : a = 2 := by
  sorry

end base_number_is_two_l208_208633


namespace infinite_rel_prime_set_of_form_2n_minus_3_l208_208832

theorem infinite_rel_prime_set_of_form_2n_minus_3 : ∃ S : Set ℕ, (∀ x ∈ S, ∃ n : ℕ, x = 2^n - 3) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧ S.Infinite := 
by
  sorry

end infinite_rel_prime_set_of_form_2n_minus_3_l208_208832


namespace money_inequality_l208_208672

-- Definitions and conditions
variables (a b : ℝ)
axiom cond1 : 6 * a + b > 78
axiom cond2 : 4 * a - b = 42

-- Theorem that encapsulates the problem and required proof
theorem money_inequality (a b : ℝ) (h1: 6 * a + b > 78) (h2: 4 * a - b = 42) : a > 12 ∧ b > 6 :=
  sorry

end money_inequality_l208_208672


namespace not_all_x_heart_x_eq_0_l208_208216

def heartsuit (x y : ℝ) : ℝ := abs (x + y)

theorem not_all_x_heart_x_eq_0 :
  ¬ (∀ x : ℝ, heartsuit x x = 0) :=
by sorry

end not_all_x_heart_x_eq_0_l208_208216


namespace alpha_beta_squared_l208_208631

section
variables (α β : ℝ)
-- Given conditions
def is_root (a b : ℝ) : Prop :=
  a + b = 2 ∧ a * b = -1 ∧ (∀ x : ℝ, x^2 - 2 * x - 1 = 0 → x = a ∨ x = b)

-- The theorem to prove
theorem alpha_beta_squared (h: is_root α β) : α^2 + β^2 = 6 :=
sorry
end

end alpha_beta_squared_l208_208631


namespace janice_total_earnings_l208_208962

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end janice_total_earnings_l208_208962


namespace p_sufficient_not_necessary_q_l208_208769

-- Define the conditions p and q
def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

-- Prove the relationship between p and q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l208_208769


namespace tan_alpha_value_l208_208066

theorem tan_alpha_value
  (α : ℝ)
  (h_cos : Real.cos α = -4/5)
  (h_range : (Real.pi / 2) < α ∧ α < Real.pi) :
  Real.tan α = -3/4 := by
  sorry

end tan_alpha_value_l208_208066


namespace ratio_of_perimeters_l208_208145

theorem ratio_of_perimeters (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 / b^2 = 49 / 64) : a / b = 7 / 8 :=
by
  sorry

end ratio_of_perimeters_l208_208145


namespace smallest_square_area_l208_208226

theorem smallest_square_area (n : ℕ) (h : ∃ m : ℕ, 14 * n = m ^ 2) : n = 14 :=
sorry

end smallest_square_area_l208_208226


namespace find_m_l208_208242

theorem find_m (m : ℝ) (x : ℝ) (y : ℝ) (h_eq_parabola : y = m * x^2)
  (h_directrix : y = 1 / 8) : m = -2 :=
by
  sorry

end find_m_l208_208242


namespace paper_holes_symmetric_l208_208467

-- Define the initial conditions
def folded_paper : Type := sorry -- Specific structure to represent the paper and its folds

def paper_fold_bottom_to_top (paper : folded_paper) : folded_paper := sorry
def paper_fold_right_half_to_left (paper : folded_paper) : folded_paper := sorry
def paper_fold_diagonal (paper : folded_paper) : folded_paper := sorry

-- Define a function that represents punching a hole near the folded edge
def punch_hole_near_folded_edge (paper : folded_paper) : folded_paper := sorry

-- Initial paper
def initial_paper : folded_paper := sorry

-- Folded and punched paper
def paper_after_folds_and_punch : folded_paper :=
  punch_hole_near_folded_edge (
    paper_fold_diagonal (
      paper_fold_right_half_to_left (
        paper_fold_bottom_to_top initial_paper)))

-- Unfolding the paper
def unfold_diagonal (paper : folded_paper) : folded_paper := sorry
def unfold_right_half (paper : folded_paper) : folded_paper := sorry
def unfold_bottom_to_top (paper : folded_paper) : folded_paper := sorry

def paper_after_unfolding : folded_paper :=
  unfold_bottom_to_top (
    unfold_right_half (
      unfold_diagonal paper_after_folds_and_punch))

-- Definition of hole pattern 'eight_symmetric_holes'
def eight_symmetric_holes (paper : folded_paper) : Prop := sorry

-- The proof problem
theorem paper_holes_symmetric :
  eight_symmetric_holes paper_after_unfolding := sorry

end paper_holes_symmetric_l208_208467


namespace sum_of_areas_S5_l208_208251

-- Definition of the points and their coordinates
def point (n : ℕ) (hn : n > 0) : ℝ × ℝ := (n, 2 / n)

-- Area of the triangle formed by the line passing through P_n and P_{n+1} with the coordinate axes
def triangleArea (n : ℕ) (hn : n > 0) : ℝ :=
  let y_intercept := (2 / n) + (2 / (n + 1))
  let x_intercept := 2 * n + 1
  (1 / 2) * y_intercept * x_intercept

-- Sum of the areas of the first n triangles
def sumOfAreas (n : ℕ) : ℝ :=
  ∑ i in finset.range n, triangleArea (i + 1) (nat.succ_pos i)

-- Theorem statement
theorem sum_of_areas_S5 : sumOfAreas 5 = 125 / 6 := sorry

end sum_of_areas_S5_l208_208251


namespace zoo_animal_difference_l208_208459

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := 1 / 2 * (parrots + snakes)
  let zebras := elephants - 3
  monkeys - zebras = 35 :=
by
  sorry

end zoo_animal_difference_l208_208459


namespace parabola_translation_l208_208330

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 5
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

-- Statement of the translation problem in Lean 4
theorem parabola_translation :
  ∀ x : ℝ, g x = f (x + 2) - 3 := 
sorry

end parabola_translation_l208_208330


namespace original_number_value_l208_208715

noncomputable def orig_number_condition (x : ℝ) : Prop :=
  1000 * x = 3 * (1 / x)

theorem original_number_value : ∃ x : ℝ, 0 < x ∧ orig_number_condition x ∧ x = √(30) / 100 :=
begin
  -- the proof
  sorry
end

end original_number_value_l208_208715


namespace pulley_distance_l208_208903

theorem pulley_distance (r₁ r₂ d l : ℝ):
    r₁ = 10 →
    r₂ = 6 →
    l = 30 →
    (d = 2 * Real.sqrt 229) :=
by
    intros h₁ h₂ h₃
    sorry

end pulley_distance_l208_208903


namespace am_gm_four_vars_l208_208879

theorem am_gm_four_vars {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / a + 1 / b + 1 / c + 1 / d) ≥ 16 :=
by
  sorry

end am_gm_four_vars_l208_208879


namespace original_price_of_table_l208_208988

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end original_price_of_table_l208_208988


namespace calculate_value_l208_208592

theorem calculate_value : (24 + 12) / ((5 - 3) * 2) = 9 := by 
  sorry

end calculate_value_l208_208592


namespace norm_photos_l208_208027

-- Define variables for the number of photos taken by Lisa, Mike, and Norm.
variables {L M N : ℕ}

-- Define the given conditions as hypotheses.
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := N = 2 * L + 10

-- State the problem in Lean: we want to prove that the number of photos Norm took is 110.
theorem norm_photos (L M N : ℕ) (h1 : condition1 L M N) (h2 : condition2 L N) : N = 110 :=
by
  sorry

end norm_photos_l208_208027


namespace higher_amount_is_sixty_l208_208537

theorem higher_amount_is_sixty (R : ℕ) (n : ℕ) (H : ℝ) 
  (h1 : 2000 = 40 * n + H * R)
  (h2 : 1800 = 40 * (n + 10) + H * (R - 10)) :
  H = 60 :=
by
  sorry

end higher_amount_is_sixty_l208_208537


namespace track_width_eight_l208_208326

theorem track_width_eight (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 16 * Real.pi) : r1 - r2 = 8 := 
sorry

end track_width_eight_l208_208326


namespace A_div_B_l208_208912

noncomputable def A : ℝ := 
  ∑' n, if n % 2 = 0 ∧ n % 4 ≠ 0 then 1 / (n:ℝ)^2 else 0

noncomputable def B : ℝ := 
  ∑' n, if n % 4 = 0 then (-1)^(n / 4 + 1) * 1 / (n:ℝ)^2 else 0

theorem A_div_B : A / B = 17 := by
  sorry

end A_div_B_l208_208912


namespace square_paintings_size_l208_208470

theorem square_paintings_size (total_area : ℝ) (small_paintings_count : ℕ) (small_painting_area : ℝ) 
                              (large_painting_area : ℝ) (square_paintings_count : ℕ) (square_paintings_total_area : ℝ) : 
  total_area = small_paintings_count * small_painting_area + large_painting_area + square_paintings_total_area → 
  square_paintings_count = 3 → 
  small_paintings_count = 4 → 
  small_painting_area = 2 * 3 → 
  large_painting_area = 10 * 15 → 
  square_paintings_total_area = 3 * 6^2 → 
  ∃ side_length, side_length^2 = (square_paintings_total_area / square_paintings_count) ∧ side_length = 6 := 
by
  intro h_total h_square_count h_small_count h_small_area h_large_area h_square_total 
  use 6
  sorry

end square_paintings_size_l208_208470


namespace n_power_of_two_if_2_pow_n_plus_one_odd_prime_l208_208841

-- Definition: a positive integer n is a power of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Theorem: if 2^n +1 is an odd prime, then n must be a power of 2
theorem n_power_of_two_if_2_pow_n_plus_one_odd_prime (n : ℕ) (hp : Prime (2^n + 1)) (hn : Odd (2^n + 1)) : is_power_of_two n :=
by
  sorry

end n_power_of_two_if_2_pow_n_plus_one_odd_prime_l208_208841


namespace shorter_leg_of_right_triangle_l208_208376

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end shorter_leg_of_right_triangle_l208_208376


namespace monotonic_exponential_decreasing_l208_208234

variable (a : ℝ) (f : ℝ → ℝ)

theorem monotonic_exponential_decreasing {m n : ℝ}
  (h0 : a = (Real.sqrt 5 - 1) / 2)
  (h1 : ∀ x, f x = a^x)
  (h2 : 0 < a ∧ a < 1)
  (h3 : f m > f n) :
  m < n :=
sorry

end monotonic_exponential_decreasing_l208_208234


namespace not_product_of_two_integers_l208_208396

theorem not_product_of_two_integers (n : ℕ) (hn : n > 0) :
  ∀ t k : ℕ, t * (t + k) = n^2 + n + 1 → k ≥ 2 * Nat.sqrt n :=
by
  sorry

end not_product_of_two_integers_l208_208396


namespace range_of_a_l208_208248

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, x ≥ 1 → x^2 + a * x + 9 ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l208_208248


namespace distance_yolkino_palkino_l208_208665

theorem distance_yolkino_palkino (d_1 d_2 : ℕ) (h : ∀ k : ℕ, d_1 + d_2 = 13) : 
  ∀ k : ℕ, d_1 + d_2 = 13 → (d_1 + d_2 = 13) :=
by
  sorry

end distance_yolkino_palkino_l208_208665


namespace line_equation_l208_208351

noncomputable def arithmetic_sequence (n : ℕ) (a_1 d : ℝ) : ℝ :=
  a_1 + (n - 1) * d

theorem line_equation
  (a_2 a_4 a_5 : ℝ)
  (a_2_cond : a_2 = arithmetic_sequence 2 a_1 d)
  (a_4_cond : a_4 = arithmetic_sequence 4 a_1 d)
  (a_5_cond : a_5 = arithmetic_sequence 5 a_1 d)
  (sum_cond : a_2 + a_4 = 12)
  (a_5_val : a_5 = 10)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * y = 0 ↔ (x - 0)^2 + (y - 1)^2 = 1)
  : ∃ (line : ℝ → ℝ → Prop), line x y ↔ (6 * x - y + 1 = 0) :=
by
  sorry

end line_equation_l208_208351


namespace initial_matchsticks_l208_208756

-- Define the problem conditions
def matchsticks_elvis := 4
def squares_elvis := 5
def matchsticks_ralph := 8
def squares_ralph := 3
def matchsticks_left := 6

-- Calculate the total matchsticks used by Elvis and Ralph
def total_used_elvis := matchsticks_elvis * squares_elvis
def total_used_ralph := matchsticks_ralph * squares_ralph
def total_used := total_used_elvis + total_used_ralph

-- The proof statement
theorem initial_matchsticks (matchsticks_elvis squares_elvis matchsticks_ralph squares_ralph matchsticks_left : ℕ) : total_used + matchsticks_left = 50 := 
by
  sorry

end initial_matchsticks_l208_208756


namespace percentage_increase_in_population_due_to_birth_is_55_l208_208166

/-- The initial population at the start of the period is 100,000 people. -/
def initial_population : ℕ := 100000

/-- The period of observation is 10 years. -/
def period : ℕ := 10

/-- The number of people leaving the area each year due to emigration is 2000. -/
def emigration_per_year : ℕ := 2000

/-- The number of people coming into the area each year due to immigration is 2500. -/
def immigration_per_year : ℕ := 2500

/-- The population at the end of the period is 165,000 people. -/
def final_population : ℕ := 165000

/-- The net migration per year is calculated by subtracting emigration from immigration. -/
def net_migration_per_year : ℕ := immigration_per_year - emigration_per_year

/-- The total net migration over the period is obtained by multiplying net migration per year by the number of years. -/
def net_migration_over_period : ℕ := net_migration_per_year * period

/-- The total population increase is the difference between the final and initial population. -/
def total_population_increase : ℕ := final_population - initial_population

/-- The increase in population due to birth is calculated by subtracting net migration over the period from the total population increase. -/
def increase_due_to_birth : ℕ := total_population_increase - net_migration_over_period

/-- The percentage increase in population due to birth is calculated by dividing the increase due to birth by the initial population, and then multiplying by 100 to convert to percentage. -/
def percentage_increase_due_to_birth : ℕ := (increase_due_to_birth * 100) / initial_population

/-- The final Lean statement to prove. -/
theorem percentage_increase_in_population_due_to_birth_is_55 :
  percentage_increase_due_to_birth = 55 := by
sorry

end percentage_increase_in_population_due_to_birth_is_55_l208_208166


namespace cyclic_sum_fraction_ge_one_l208_208825

theorem cyclic_sum_fraction_ge_one (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (hineq : (a/(b+c+1) + b/(c+a+1) + c/(a+b+1)) ≤ 1) :
  (1/(b+c+1) + 1/(c+a+1) + 1/(a+b+1)) ≥ 1 :=
by sorry

end cyclic_sum_fraction_ge_one_l208_208825


namespace modulus_of_z_equals_sqrt2_l208_208069

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := 2 * i + (2 / (1 + i))

theorem modulus_of_z_equals_sqrt2 : complex.abs z = real.sqrt 2 :=
by sorry

end modulus_of_z_equals_sqrt2_l208_208069


namespace min_abs_sum_is_two_l208_208697

theorem min_abs_sum_is_two : ∃ x ∈ set.Ioo (- ∞) (∞), ∀ y ∈ set.Ioo (- ∞) (∞), (|y + 1| + |y + 3| + |y + 6| ≥ 2) ∧ (|x + 1| + |x + 3| + |x + 6| = 2) := sorry

end min_abs_sum_is_two_l208_208697


namespace std_dev_samples_l208_208249

def sample_A := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def sample_B := [84, 86, 86, 88, 88, 88, 90, 90, 90, 90]

noncomputable def std_dev (l : List ℕ) :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  let variance := (l.map (λ x => (x - mean) * (x - mean))).sum / n
  variance.sqrt

theorem std_dev_samples :
  std_dev sample_A = std_dev sample_B := 
sorry

end std_dev_samples_l208_208249


namespace billy_free_time_l208_208334

theorem billy_free_time
  (play_time_percentage : ℝ := 0.75)
  (read_pages_per_hour : ℝ := 60)
  (book_pages : ℝ := 80)
  (number_of_books : ℝ := 3)
  (read_percentage : ℝ := 1 - play_time_percentage)
  (total_pages : ℝ := number_of_books * book_pages)
  (read_time_hours : ℝ := total_pages / read_pages_per_hour)
  (free_time_hours : ℝ := read_time_hours / read_percentage) :
  free_time_hours = 16 := 
sorry

end billy_free_time_l208_208334
