import Mathlib

namespace solve_for_x_l1779_177987

-- Assumptions and conditions of the problem
def a : ℚ := 4 / 7
def b : ℚ := 1 / 5
def c : ℚ := 12
def d : ℚ := 105

-- The statement of the problem
theorem solve_for_x (x : ℚ) (h : a * b * x = c) : x = d :=
by sorry

end solve_for_x_l1779_177987


namespace greatest_possible_large_chips_l1779_177996

theorem greatest_possible_large_chips : 
  ∃ s l p: ℕ, s + l = 60 ∧ s = l + 2 * p ∧ Prime p ∧ l = 28 :=
by
  sorry

end greatest_possible_large_chips_l1779_177996


namespace fixed_point_C_D_intersection_l1779_177945

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ P.2 ≠ 0

noncomputable def line_CD_fixed_point (t : ℝ) (C D : ℝ × ℝ) : Prop :=
  let x1 := (36 - 2 * t^2) / (18 + t^2)
  let y1 := (12 * t) / (18 + t^2)
  let x2 := (2 * t^2 - 4) / (2 + t^2)
  let y2 := -(4 * t) / (t^2 + 2)
  C = (x1, y1) ∧ D = (x2, y2) →
  let k_CD := (4 * t) / (6 - t^2)
  ∀ (x y : ℝ), y + (4 * t) / (t^2 + 2) = k_CD * (x - (2 * t^2 - 4) / (t^2 + 2)) →
  y = 0 → x = 1

theorem fixed_point_C_D_intersection :
  ∀ (t : ℝ) (C D : ℝ × ℝ), point_on_line (4, t) →
  ellipse_equation C.1 C.2 →
  ellipse_equation D.1 D.2 →
  line_CD_fixed_point t C D :=
by
  intros t C D point_on_line_P ellipse_C ellipse_D
  sorry

end fixed_point_C_D_intersection_l1779_177945


namespace probability_C_calc_l1779_177950

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

end probability_C_calc_l1779_177950


namespace harold_catches_up_at_12_miles_l1779_177964

/-- 
Proof Problem: Given that Adrienne starts walking from X to Y at 3 miles per hour and one hour later Harold starts walking from X to Y at 4 miles per hour, prove that Harold covers 12 miles when he catches up to Adrienne.
-/
theorem harold_catches_up_at_12_miles :
  (∀ (T : ℕ), (ad_distance : ℕ) = 3 * (T + 1) → (ha_distance : ℕ) = 4 * T → ad_distance = ha_distance) →
  (∃ T : ℕ, ha_distance = 12) :=
by
  sorry

end harold_catches_up_at_12_miles_l1779_177964


namespace eccentricity_of_ellipse_l1779_177921

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
                                 (h_angle : Real.cos (Real.pi / 6) = b / a) :
    eccentricity a b = (Real.sqrt 6) / 3 := by
  sorry

end eccentricity_of_ellipse_l1779_177921


namespace science_fair_unique_students_l1779_177990

/-!
# Problem statement:
At Euclid Middle School, there are three clubs participating in the Science Fair: the Robotics Club, the Astronomy Club, and the Chemistry Club.
There are 15 students in the Robotics Club, 10 students in the Astronomy Club, and 12 students in the Chemistry Club.
Assuming 2 students are members of all three clubs, prove that the total number of unique students participating in the Science Fair is 33.
-/

theorem science_fair_unique_students (R A C : ℕ) (all_three : ℕ) (hR : R = 15) (hA : A = 10) (hC : C = 12) (h_all_three : all_three = 2) :
    R + A + C - 2 * all_three = 33 :=
by
  -- Proof goes here
  sorry

end science_fair_unique_students_l1779_177990


namespace integer_values_of_b_l1779_177963

theorem integer_values_of_b (b : ℤ) :
  (∃ x : ℤ, x^3 + 2*x^2 + b*x + 18 = 0) ↔ 
  b = -21 ∨ b = 19 ∨ b = -17 ∨ b = -4 ∨ b = 3 :=
by
  sorry

end integer_values_of_b_l1779_177963


namespace part1_part2_l1779_177975

-- Definitions for the sides and the target equations
def triangleSides (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.sin (C / 2) ^ 2 + c * Real.sin (B / 2) ^ 2 = a / 2

-- The first part of the problem
theorem part1 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  b + c = 2 * a :=
  sorry

-- The second part of the problem
theorem part2 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  A ≤ π / 3 :=
  sorry

end part1_part2_l1779_177975


namespace find_rectangle_pairs_l1779_177957

theorem find_rectangle_pairs (w l : ℕ) (hw : w > 0) (hl : l > 0) (h : w * l = 18) : 
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) ∨
  (w, l) = (6, 3) ∨ (w, l) = (9, 2) ∨ (w, l) = (18, 1) :=
by
  sorry

end find_rectangle_pairs_l1779_177957


namespace integer_solution_count_eq_eight_l1779_177940

theorem integer_solution_count_eq_eight : ∃ S : Finset (ℤ × ℤ), (∀ s ∈ S, 2 * s.1 ^ 2 + s.1 * s.2 - s.2 ^ 2 = 14 ∧ (s.1 = s.1 ∧ s.2 = s.2)) ∧ S.card = 8 :=
by
  sorry

end integer_solution_count_eq_eight_l1779_177940


namespace elberta_money_l1779_177953

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) 
  (h1 : granny_smith = 120) 
  (h2 : anjou = granny_smith / 4) 
  (h3 : elberta = anjou + 5) : 
  elberta = 35 :=
by {
  sorry
}

end elberta_money_l1779_177953


namespace ratio_in_range_l1779_177986

theorem ratio_in_range {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end ratio_in_range_l1779_177986


namespace gcd_polynomial_l1779_177999

theorem gcd_polynomial (b : ℤ) (h : b % 2 = 0 ∧ 1171 ∣ b) : 
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 :=
sorry

end gcd_polynomial_l1779_177999


namespace arman_sister_age_l1779_177971

-- Define the conditions
variables (S : ℝ) -- Arman's sister's age four years ago
variable (A : ℝ) -- Arman's age four years ago

-- Given conditions as hypotheses
axiom h1 : A = 6 * S -- Arman is six times older than his sister
axiom h2 : A + 8 = 40 -- In 4 years, Arman's age will be 40 (hence, A in 4 years should be A + 8)

-- Main theorem to prove
theorem arman_sister_age (h1 : A = 6 * S) (h2 : A + 8 = 40) : S = 16 / 3 :=
by
  sorry

end arman_sister_age_l1779_177971


namespace jana_height_l1779_177931

theorem jana_height (Jess_height : ℕ) (h1 : Jess_height = 72) 
  (Kelly_height : ℕ) (h2 : Kelly_height = Jess_height - 3) 
  (Jana_height : ℕ) (h3 : Jana_height = Kelly_height + 5) : 
  Jana_height = 74 := by
  subst h1
  subst h2
  subst h3
  sorry

end jana_height_l1779_177931


namespace average_is_4_l1779_177965

theorem average_is_4 (p q r s : ℝ) (h : (5 / 4) * (p + q + r + s) = 20) : 
  (p + q + r + s) / 4 = 4 := 
by 
  sorry 

end average_is_4_l1779_177965


namespace min_stamps_for_target_value_l1779_177935

theorem min_stamps_for_target_value :
  ∃ (c f : ℕ), 5 * c + 7 * f = 50 ∧ ∀ (c' f' : ℕ), 5 * c' + 7 * f' = 50 → c + f ≤ c' + f' → c + f = 8 :=
by
  sorry

end min_stamps_for_target_value_l1779_177935


namespace solve_for_x_l1779_177924

theorem solve_for_x (h_perimeter_square : ∀(s : ℝ), 4 * s = 64)
  (h_height_triangle : ∀(h : ℝ), h = 48)
  (h_area_equal : ∀(s h x : ℝ), s * s = 1/2 * h * x) : 
  x = 32 / 3 := by
  sorry

end solve_for_x_l1779_177924


namespace cube_sum_div_by_nine_l1779_177912

theorem cube_sum_div_by_nine (n : ℕ) (hn : 0 < n) : (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 := by sorry

end cube_sum_div_by_nine_l1779_177912


namespace decreasing_function_range_l1779_177955

theorem decreasing_function_range (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end decreasing_function_range_l1779_177955


namespace intersected_squares_and_circles_l1779_177978

def is_intersected_by_line (p q : ℕ) : Prop :=
  p = q

def total_intersections : ℕ := 504 * 2

theorem intersected_squares_and_circles :
  total_intersections = 1008 :=
by
  sorry

end intersected_squares_and_circles_l1779_177978


namespace owen_wins_with_n_bullseyes_l1779_177944

-- Define the parameters and conditions
def initial_score_lead : ℕ := 60
def total_shots : ℕ := 120
def bullseye_points : ℕ := 9
def minimum_points_per_shot : ℕ := 3
def max_points_per_shot : ℕ := 9
def n : ℕ := 111

-- Define the condition for Owen's winning requirement
theorem owen_wins_with_n_bullseyes :
  6 * 111 + 360 > 1020 :=
by
  sorry

end owen_wins_with_n_bullseyes_l1779_177944


namespace combination_2586_1_eq_2586_l1779_177906

noncomputable def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_2586_1_eq_2586 : combination 2586 1 = 2586 := by
  sorry

end combination_2586_1_eq_2586_l1779_177906


namespace penalty_kicks_calculation_l1779_177966

def totalPlayers := 24
def goalkeepers := 4
def nonGoalkeeperShootsAgainstOneGoalkeeper := totalPlayers - 1
def totalPenaltyKicks := goalkeepers * nonGoalkeeperShootsAgainstOneGoalkeeper

theorem penalty_kicks_calculation : totalPenaltyKicks = 92 := by
  sorry

end penalty_kicks_calculation_l1779_177966


namespace sin_2x_and_tan_fraction_l1779_177959

open Real

theorem sin_2x_and_tan_fraction (x : ℝ) (h : sin (π + x) + cos (π + x) = 1 / 2) :
  (sin (2 * x) = -3 / 4) ∧ ((1 + tan x) / (sin x * cos (x - π / 4)) = -8 * sqrt 2 / 3) :=
by
  sorry

end sin_2x_and_tan_fraction_l1779_177959


namespace range_of_k_l1779_177949

theorem range_of_k (a k : ℝ) : 
  (∀ x y : ℝ, y^2 - x * y + 2 * x + k = 0 → (x = a ∧ y = -a)) →
  k ≤ 1/2 :=
by sorry

end range_of_k_l1779_177949


namespace min_value_expression_l1779_177901

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 4*x + 2) * (y^2 + 4*y + 2) * (z^2 + 4*z + 2) / (x * y * z) ≥ 216 :=
by
  sorry

end min_value_expression_l1779_177901


namespace bob_hair_growth_time_l1779_177942

theorem bob_hair_growth_time (initial_length final_length growth_rate monthly_to_yearly_conversion : ℝ) 
  (initial_cut : initial_length = 6) 
  (current_length : final_length = 36) 
  (growth_per_month : growth_rate = 0.5) 
  (months_in_year : monthly_to_yearly_conversion = 12) : 
  (final_length - initial_length) / (growth_rate * monthly_to_yearly_conversion) = 5 :=
by
  sorry

end bob_hair_growth_time_l1779_177942


namespace yellow_percentage_l1779_177915

theorem yellow_percentage (s w : ℝ) 
  (h_cross : w * w + 4 * w * (s - 2 * w) = 0.49 * s * s) : 
  (w / s) ^ 2 = 0.2514 :=
by
  sorry

end yellow_percentage_l1779_177915


namespace polar_distance_to_axis_l1779_177992

theorem polar_distance_to_axis (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = Real.pi / 6) : 
  ρ * Real.sin θ = 1 := 
by
  rw [hρ, hθ]
  -- The remaining proof steps would go here
  sorry

end polar_distance_to_axis_l1779_177992


namespace transition_to_modern_population_reproduction_l1779_177916

-- Defining the conditions as individual propositions
def A : Prop := ∃ (m b : ℝ), m < 0 ∧ b = 0
def B : Prop := ∃ (m b : ℝ), m < 0 ∧ b < 0
def C : Prop := ∃ (m b : ℝ), m > 0 ∧ b = 0
def D : Prop := ∃ (m b : ℝ), m > 0 ∧ b > 0

-- Defining the question as a property marking the transition from traditional to modern types of population reproduction
def Q : Prop := B

-- The proof problem
theorem transition_to_modern_population_reproduction :
  Q = B :=
by
  sorry

end transition_to_modern_population_reproduction_l1779_177916


namespace hockey_season_length_l1779_177946

theorem hockey_season_length (total_games_per_month : ℕ) (total_games_season : ℕ) 
  (h1 : total_games_per_month = 13) (h2 : total_games_season = 182) : 
  total_games_season / total_games_per_month = 14 := 
by 
  sorry

end hockey_season_length_l1779_177946


namespace average_interest_rate_l1779_177907

theorem average_interest_rate (I : ℝ) (r1 r2 : ℝ) (y : ℝ)
  (h0 : I = 6000)
  (h1 : r1 = 0.05)
  (h2 : r2 = 0.07)
  (h3 : 0.05 * (6000 - y) = 0.07 * y) :
  ((r1 * (I - y) + r2 * y) / I) = 0.05833 :=
by
  sorry

end average_interest_rate_l1779_177907


namespace henry_age_is_29_l1779_177961

-- Definitions and conditions
variable (Henry_age Jill_age : ℕ)

-- Condition 1: Sum of the present age of Henry and Jill is 48
def sum_of_ages : Prop := Henry_age + Jill_age = 48

-- Condition 2: Nine years ago, Henry was twice the age of Jill
def age_relation_nine_years_ago : Prop := Henry_age - 9 = 2 * (Jill_age - 9)

-- Theorem to prove
theorem henry_age_is_29 (H: ℕ) (J: ℕ)
  (h1 : sum_of_ages H J) 
  (h2 : age_relation_nine_years_ago H J) : H = 29 :=
by
  sorry

end henry_age_is_29_l1779_177961


namespace least_positive_integer_n_l1779_177919

theorem least_positive_integer_n : ∃ (n : ℕ), (1 / (n : ℝ) - 1 / (n + 1) < 1 / 100) ∧ ∀ m, m < n → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 100) :=
sorry

end least_positive_integer_n_l1779_177919


namespace six_diggers_five_hours_l1779_177910

theorem six_diggers_five_hours (holes_per_hour_per_digger : ℝ) 
  (h1 : 3 * holes_per_hour_per_digger * 3 = 3) :
  6 * (holes_per_hour_per_digger) * 5 = 10 :=
by
  -- The proof will go here, but we only need to state the theorem
  sorry

end six_diggers_five_hours_l1779_177910


namespace trays_from_first_table_is_23_l1779_177977

-- Definitions of conditions
def trays_per_trip : ℕ := 7
def trips_made : ℕ := 4
def trays_from_second_table : ℕ := 5

-- Total trays carried
def total_trays_carried : ℕ := trays_per_trip * trips_made

-- Number of trays picked from first table
def trays_from_first_table : ℕ :=
  total_trays_carried - trays_from_second_table

-- Theorem stating that the number of trays picked up from the first table is 23
theorem trays_from_first_table_is_23 : trays_from_first_table = 23 := by
  sorry

end trays_from_first_table_is_23_l1779_177977


namespace intersection_of_lines_l1779_177991

theorem intersection_of_lines :
  ∃ (x y : ℝ), (8 * x + 5 * y = 40) ∧ (3 * x - 10 * y = 15) ∧ (x = 5) ∧ (y = 0) := 
by 
  sorry

end intersection_of_lines_l1779_177991


namespace tile_floor_multiple_of_seven_l1779_177937

theorem tile_floor_multiple_of_seven (n : ℕ) (a : ℕ)
  (h1 : n * n = 7 * a)
  (h2 : 4 * a / 7 + 3 * a / 7 = a) :
  ∃ k : ℕ, n = 7 * k := by
  sorry

end tile_floor_multiple_of_seven_l1779_177937


namespace tom_initial_foreign_exchange_l1779_177970

theorem tom_initial_foreign_exchange (x : ℝ) (y₀ y₁ y₂ y₃ y₄ : ℝ) :
  y₀ = x / 2 - 5 ∧
  y₁ = y₀ / 2 - 5 ∧
  y₂ = y₁ / 2 - 5 ∧
  y₃ = y₂ / 2 - 5 ∧
  y₄ = y₃ / 2 - 5 ∧
  y₄ - 5 = 100
  → x = 3355 :=
by
  intro h
  sorry

end tom_initial_foreign_exchange_l1779_177970


namespace eleven_pow_2010_mod_19_l1779_177973

theorem eleven_pow_2010_mod_19 : (11 ^ 2010) % 19 = 3 := sorry

end eleven_pow_2010_mod_19_l1779_177973


namespace find_angle_2_l1779_177984

theorem find_angle_2 (angle1 : ℝ) (angle2 : ℝ) 
  (h1 : angle1 = 60) 
  (h2 : angle1 + angle2 = 180) : 
  angle2 = 120 := 
by
  sorry

end find_angle_2_l1779_177984


namespace complement_union_l1779_177902

open Set

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3}
noncomputable def C_UA : Set ℕ := U \ A

-- Statement to prove
theorem complement_union (U A B C_UA : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3, 5})
  (hB : B = {2, 3}) 
  (hCUA : C_UA = U \ A) : 
  (C_UA ∪ B) = {2, 3, 4} := 
sorry

end complement_union_l1779_177902


namespace neg_p_equiv_exists_leq_l1779_177913

-- Define the given proposition p
def p : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- State the equivalence we need to prove
theorem neg_p_equiv_exists_leq :
  ¬ p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0 :=
by {
  sorry  -- Proof is skipped as per instructions
}

end neg_p_equiv_exists_leq_l1779_177913


namespace f_neg2_minus_f_neg3_l1779_177920

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3_l1779_177920


namespace systematic_sampling_condition_l1779_177988

theorem systematic_sampling_condition (population sample_size total_removed segments individuals_per_segment : ℕ) 
  (h_population : population = 1650)
  (h_sample_size : sample_size = 35)
  (h_total_removed : total_removed = 5)
  (h_segments : segments = sample_size)
  (h_individuals_per_segment : individuals_per_segment = (population - total_removed) / sample_size)
  (h_modulo : population % sample_size = total_removed)
  :
  total_removed = 5 ∧ segments = 35 ∧ individuals_per_segment = 47 := 
by
  sorry

end systematic_sampling_condition_l1779_177988


namespace greater_number_is_25_l1779_177981

theorem greater_number_is_25 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
sorry

end greater_number_is_25_l1779_177981


namespace sum_of_roots_l1779_177951

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) →
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0) →
  a + b + c + d = 8 :=
by
  sorry

end sum_of_roots_l1779_177951


namespace inequality_x2_y4_z6_l1779_177941

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l1779_177941


namespace subject_selection_ways_l1779_177939

theorem subject_selection_ways :
  let compulsory := 3 -- Chinese, Mathematics, English
  let choose_one := 2
  let choose_two := 6
  compulsory + choose_one * choose_two = 12 :=
by
  sorry

end subject_selection_ways_l1779_177939


namespace freshman_class_count_l1779_177956

theorem freshman_class_count : ∃ n : ℤ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧ n = 49 := by
  sorry

end freshman_class_count_l1779_177956


namespace soap_box_width_l1779_177943

theorem soap_box_width
  (carton_length : ℝ) (carton_width : ℝ) (carton_height : ℝ)
  (box_length : ℝ) (box_height : ℝ) (max_boxes : ℝ) (carton_volume : ℝ)
  (box_volume : ℝ) (W : ℝ) : 
  carton_length = 25 →
  carton_width = 42 →
  carton_height = 60 →
  box_length = 6 →
  box_height = 6 →
  max_boxes = 250 →
  carton_volume = carton_length * carton_width * carton_height →
  box_volume = box_length * W * box_height →
  max_boxes * box_volume = carton_volume →
  W = 7 :=
sorry

end soap_box_width_l1779_177943


namespace tan_pi_minus_alpha_l1779_177993

theorem tan_pi_minus_alpha (α : ℝ) (h : Real.tan (Real.pi - α) = -2) : 
  (1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5 / 2) :=
by
  sorry

end tan_pi_minus_alpha_l1779_177993


namespace train_passes_jogger_in_time_l1779_177903

def jogger_speed_kmh : ℝ := 8
def train_speed_kmh : ℝ := 60
def initial_distance_m : ℝ := 360
def train_length_m : ℝ := 200

noncomputable def jogger_speed_ms : ℝ := jogger_speed_kmh * 1000 / 3600
noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
noncomputable def relative_speed_ms : ℝ := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m
noncomputable def passing_time_s : ℝ := total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_time :
  passing_time_s = 38.75 := by
  sorry

end train_passes_jogger_in_time_l1779_177903


namespace total_sounds_produced_l1779_177938

-- Defining the total number of nails for one customer and the number of customers
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3

-- Proving the total number of nail trimming sounds for 3 customers = 60
theorem total_sounds_produced : nails_per_person * number_of_customers = 60 := by
  sorry

end total_sounds_produced_l1779_177938


namespace bucket_fill_proof_l1779_177983

variables (x y : ℕ)
def tank_capacity : ℕ := 4 * x

theorem bucket_fill_proof (hx: y = x + 4) (hy: 4 * x = 3 * y): tank_capacity x = 48 :=
by {
  -- Proof steps will be here, but are elided for now
  sorry 
}

end bucket_fill_proof_l1779_177983


namespace percentage_calculation_l1779_177904

def percentage_less_than_50000_towns : Float := 85

def percentage_less_than_20000_towns : Float := 20
def percentage_20000_to_49999_towns : Float := 65

theorem percentage_calculation :
  percentage_less_than_50000_towns = percentage_less_than_20000_towns + percentage_20000_to_49999_towns :=
by
  sorry

end percentage_calculation_l1779_177904


namespace total_money_shared_l1779_177905

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l1779_177905


namespace arithmetic_mean_is_ten_l1779_177928

theorem arithmetic_mean_is_ten (a b x : ℝ) (h₁ : a = 4) (h₂ : b = 16) (h₃ : x = (a + b) / 2) : x = 10 :=
by
  sorry

end arithmetic_mean_is_ten_l1779_177928


namespace greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l1779_177922

theorem greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30 :
  ∃ d, d ∣ 480 ∧ d < 60 ∧ d ∣ 90 ∧ (∀ e, e ∣ 480 → e < 60 → e ∣ 90 → e ≤ d) ∧ d = 30 :=
sorry

end greatest_divisor_of_480_less_than_60_and_factor_of_90_is_30_l1779_177922


namespace average_salary_all_workers_l1779_177960

theorem average_salary_all_workers 
  (n : ℕ) (avg_salary_technicians avg_salary_rest total_avg_salary : ℝ)
  (h1 : n = 7) 
  (h2 : avg_salary_technicians = 8000) 
  (h3 : avg_salary_rest = 6000)
  (h4 : total_avg_salary = avg_salary_technicians) : 
  total_avg_salary = 8000 :=
by sorry

end average_salary_all_workers_l1779_177960


namespace smallest_norm_of_v_l1779_177958

variables (v : ℝ × ℝ)

def vector_condition (v : ℝ × ℝ) : Prop :=
  ‖(v.1 - 2, v.2 + 4)‖ = 10

theorem smallest_norm_of_v
  (hv : vector_condition v) :
  ‖v‖ ≥ 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_of_v_l1779_177958


namespace john_final_push_time_l1779_177930

theorem john_final_push_time :
  ∃ t : ℝ, (∀ (d_j d_s : ℝ), d_j = 4.2 * t ∧ d_s = 3.7 * t ∧ (d_j = d_s + 14)) → t = 28 :=
by
  sorry

end john_final_push_time_l1779_177930


namespace minimize_y_l1779_177926

noncomputable def y (x a b c : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + (x - c)^2

theorem minimize_y (a b c : ℝ) : ∃ x : ℝ, (∀ x0 : ℝ, y x a b c ≤ y x0 a b c) ∧ x = (a + b + c) / 3 :=
by
  sorry

end minimize_y_l1779_177926


namespace strictly_increasing_interval_l1779_177962

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

theorem strictly_increasing_interval :
  (∀ k : ℤ, ∀ x : ℝ, 
    (2 * k * Real.pi - 5 * Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) 
    → (f x) < (f (x + 1))) :=
by 
  sorry

end strictly_increasing_interval_l1779_177962


namespace Milly_study_time_l1779_177918

theorem Milly_study_time :
  let math_time := 60
  let geo_time := math_time / 2
  let mean_time := (math_time + geo_time) / 2
  let total_study_time := math_time + geo_time + mean_time
  total_study_time = 135 := by
  sorry

end Milly_study_time_l1779_177918


namespace option_d_l1779_177936

variable {R : Type*} [LinearOrderedField R]

theorem option_d (a b c d : R) (h1 : a > b) (h2 : c > d) : a - d > b - c := 
by 
  sorry

end option_d_l1779_177936


namespace mario_age_difference_l1779_177909

variable (Mario_age Maria_age : ℕ)

def age_conditions (Mario_age Maria_age difference : ℕ) : Prop :=
  Mario_age + Maria_age = 7 ∧
  Mario_age = 4 ∧
  Mario_age - Maria_age = difference

theorem mario_age_difference : ∃ (difference : ℕ), age_conditions 4 (4 - difference) difference ∧ difference = 1 := by
  sorry

end mario_age_difference_l1779_177909


namespace measure_of_y_l1779_177947

theorem measure_of_y (y : ℕ) (h₁ : 40 + 2 * y + y = 180) : y = 140 / 3 :=
by
  sorry

end measure_of_y_l1779_177947


namespace brothers_percentage_fewer_trees_l1779_177932

theorem brothers_percentage_fewer_trees (total_trees initial_days brother_days : ℕ) (trees_per_day : ℕ) (total_brother_trees : ℕ) (percentage_fewer : ℕ):
  initial_days = 2 →
  brother_days = 3 →
  trees_per_day = 20 →
  total_trees = 196 →
  total_brother_trees = total_trees - (trees_per_day * initial_days) →
  percentage_fewer = ((total_brother_trees / brother_days - trees_per_day) * 100) / trees_per_day →
  percentage_fewer = 60 :=
by
  sorry

end brothers_percentage_fewer_trees_l1779_177932


namespace correct_operation_l1779_177908

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l1779_177908


namespace find_cos_A_l1779_177968

theorem find_cos_A
  (A C : ℝ)
  (AB CD : ℝ)
  (AD BC : ℝ)
  (α : ℝ)
  (h1 : A = C)
  (h2 : AB = 150)
  (h3 : CD = 150)
  (h4 : AD ≠ BC)
  (h5 : AB + BC + CD + AD = 560)
  (h6 : A = α)
  (h7 : C = α)
  (BD₁ BD₂ : ℝ)
  (h8 : BD₁^2 = AD^2 + 150^2 - 2 * 150 * AD * Real.cos α)
  (h9 : BD₂^2 = BC^2 + 150^2 - 2 * 150 * BC * Real.cos α)
  (h10 : BD₁ = BD₂) :
  Real.cos A = 13 / 15 := 
sorry

end find_cos_A_l1779_177968


namespace remainder_of_2_pow_2018_plus_1_mod_2018_l1779_177994

theorem remainder_of_2_pow_2018_plus_1_mod_2018 : (2 ^ 2018 + 1) % 2018 = 2 := by
  sorry

end remainder_of_2_pow_2018_plus_1_mod_2018_l1779_177994


namespace mark_jump_rope_hours_l1779_177972

theorem mark_jump_rope_hours 
    (record : ℕ := 54000)
    (jump_per_second : ℕ := 3)
    (seconds_per_hour : ℕ := 3600)
    (total_jumps_to_break_record : ℕ := 54001)
    (jumps_per_hour : ℕ := jump_per_second * seconds_per_hour) 
    (hours_needed : ℕ := total_jumps_to_break_record / jumps_per_hour) 
    (round_up : ℕ := if total_jumps_to_break_record % jumps_per_hour = 0 then hours_needed else hours_needed + 1) :
    round_up = 5 :=
sorry

end mark_jump_rope_hours_l1779_177972


namespace part1_part2_l1779_177917

def star (a b c d : ℝ) : ℝ := a * c - b * d

-- Part (1)
theorem part1 : star (-4) 3 2 (-6) = 10 := by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : ∀ x : ℝ, star x (2 * x - 1) (m * x + 1) m = 0 → (m ≠ 0 → (((1 - 2 * m) ^ 2 - 4 * m * m) ≥ 0))) :
  (m ≤ 1 / 4 ∨ m < 0) ∧ m ≠ 0 := by
  sorry

end part1_part2_l1779_177917


namespace unique_ordered_triple_l1779_177900

theorem unique_ordered_triple : 
  ∃! (x y z : ℝ), x + y = 4 ∧ xy - z^2 = 4 ∧ x = 2 ∧ y = 2 ∧ z = 0 :=
by
  sorry

end unique_ordered_triple_l1779_177900


namespace total_cleaning_time_l1779_177969

-- Definition for the problem conditions
def time_to_clean_egg (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ) : ℕ :=
  (num_eggs * seconds_per_egg) / seconds_per_minute

def time_to_clean_toilet_paper (minutes_per_roll : ℕ) (num_rolls : ℕ) : ℕ :=
  num_rolls * minutes_per_roll

-- Main statement to prove the total cleaning time
theorem total_cleaning_time
  (seconds_per_egg : ℕ) (num_eggs : ℕ) (seconds_per_minute : ℕ)
  (minutes_per_roll : ℕ) (num_rolls : ℕ) :
  seconds_per_egg = 15 →
  num_eggs = 60 →
  seconds_per_minute = 60 →
  minutes_per_roll = 30 →
  num_rolls = 7 →
  time_to_clean_egg seconds_per_egg num_eggs seconds_per_minute +
  time_to_clean_toilet_paper minutes_per_roll num_rolls = 225 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cleaning_time_l1779_177969


namespace complement_P_subset_PQ_intersection_PQ_eq_Q_l1779_177979

open Set

variable {R : Type*} [OrderedCommRing R]

def P (x : R) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (m x : R) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem complement_P : (compl (setOf P)) = {x | x < -2} ∪ {x | x > 10} :=
by {
  sorry
}

theorem subset_PQ (m : R) : (∀ x, P x → Q m x) ↔ m ≥ 9 :=
by {
  sorry
}

theorem intersection_PQ_eq_Q (m : R) : (∀ x, Q m x → P x) ↔ m ≤ 9 :=
by {
  sorry
}

end complement_P_subset_PQ_intersection_PQ_eq_Q_l1779_177979


namespace walt_total_invested_l1779_177985

-- Given Conditions
def invested_at_seven : ℝ := 5500
def total_interest : ℝ := 970
def interest_rate_seven : ℝ := 0.07
def interest_rate_nine : ℝ := 0.09

-- Define the total amount invested
noncomputable def total_invested : ℝ := 12000

-- Prove the total amount invested
theorem walt_total_invested :
  interest_rate_seven * invested_at_seven + interest_rate_nine * (total_invested - invested_at_seven) = total_interest :=
by
  -- The proof goes here
  sorry

end walt_total_invested_l1779_177985


namespace sum_of_numbers_is_919_l1779_177989

-- Problem Conditions
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def is_three_digit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999
def satisfies_equation (x y : ℕ) : Prop := 1000 * x + y = 11 * x * y

-- Main Statement
theorem sum_of_numbers_is_919 (x y : ℕ) 
  (h1 : is_two_digit x) 
  (h2 : is_three_digit y) 
  (h3 : satisfies_equation x y) : 
  x + y = 919 := 
sorry

end sum_of_numbers_is_919_l1779_177989


namespace θ_values_l1779_177923

-- Define the given conditions
def terminal_side_coincides (θ : ℝ) : Prop :=
  ∃ k : ℤ, 7 * θ = θ + 360 * k

def θ_in_range (θ : ℝ) : Prop :=
  0 ≤ θ ∧ θ < 360

-- The main theorem
theorem θ_values (θ : ℝ) (h_terminal : terminal_side_coincides θ) (h_range : θ_in_range θ) :
  θ = 0 ∨ θ = 60 ∨ θ = 120 ∨ θ = 180 ∨ θ = 240 ∨ θ = 300 :=
sorry

end θ_values_l1779_177923


namespace initial_mixture_amount_l1779_177974

/-- A solution initially contains an unknown amount of a mixture consisting of 15% sodium chloride
(NaCl), 30% potassium chloride (KCl), 35% sugar, and 20% water. To this mixture, 50 grams of sodium chloride
and 80 grams of potassium chloride are added. If the new salt content of the solution (NaCl and KCl combined)
is 47.5%, how many grams of the mixture were present initially?

Given:
  * The initial mixture consists of 15% NaCl and 30% KCl.
  * 50 grams of NaCl and 80 grams of KCl are added.
  * The new mixture has 47.5% NaCl and KCl combined.
  
Prove that the initial amount of the mixture was 2730 grams. -/
theorem initial_mixture_amount
    (x : ℝ)
    (h_initial_mixture : 0.15 * x + 50 + 0.30 * x + 80 = 0.475 * (x + 130)) :
    x = 2730 := by
  sorry

end initial_mixture_amount_l1779_177974


namespace find_correct_fraction_l1779_177911

theorem find_correct_fraction
  (mistake_frac : ℚ) (n : ℕ) (delta : ℚ)
  (correct_frac : ℚ) (number : ℕ)
  (h1 : mistake_frac = 5 / 6)
  (h2 : number = 288)
  (h3 : mistake_frac * number = correct_frac * number + delta)
  (h4 : delta = 150) :
  correct_frac = 5 / 32 :=
by
  sorry

end find_correct_fraction_l1779_177911


namespace perimeter_C_l1779_177980

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l1779_177980


namespace George_spending_l1779_177997

theorem George_spending (B m s : ℝ) (h1 : m = 0.25 * (B - s)) (h2 : s = 0.05 * (B - m)) : 
  (m + s) / B = 1 := 
by
  sorry

end George_spending_l1779_177997


namespace initial_amount_is_3_l1779_177929

-- Define the initial amount of water in the bucket
def initial_water_amount (total water_added : ℝ) : ℝ :=
  total - water_added

-- Define the variables
def total : ℝ := 9.8
def water_added : ℝ := 6.8

-- State the problem
theorem initial_amount_is_3 : initial_water_amount total water_added = 3 := 
  by
    sorry

end initial_amount_is_3_l1779_177929


namespace problem_conditions_equation_right_triangle_vertex_coordinates_l1779_177995

theorem problem_conditions_equation : 
  ∃ (a b c : ℝ), a = -1 ∧ b = -2 ∧ c = 3 ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - (-(x + 1))^2 + 4) ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - x^2 - 2 * x + 3)
:= sorry

theorem right_triangle_vertex_coordinates :
  ∀ x y : ℝ, x = -1 ∧ 
  (y = -2 ∨ y = 4 ∨ y = (3 + (17:ℝ).sqrt) / 2 ∨ y = (3 - (17:ℝ).sqrt) / 2)
  ∧ 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (-3, 0)
  let C : ℝ × ℝ := (0, 3)
  let P : ℝ × ℝ := (x, y)
  let BC : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let PB : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let PC : ℝ := (P.1 - C.1)^2 + (P.2 - C.2)^2
  (BC + PB = PC ∨ BC + PC = PB ∨ PB + PC = BC)
:= sorry

end problem_conditions_equation_right_triangle_vertex_coordinates_l1779_177995


namespace contrapositive_of_square_inequality_l1779_177927

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x > y → x^2 > y^2) ↔ (x^2 ≤ y^2 → x ≤ y) :=
sorry

end contrapositive_of_square_inequality_l1779_177927


namespace ordered_pairs_unique_solution_l1779_177933

theorem ordered_pairs_unique_solution :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = Real.sqrt 2 :=
by
  sorry

end ordered_pairs_unique_solution_l1779_177933


namespace roshini_spent_on_sweets_l1779_177954

variable (initial_amount friends_amount total_friends_amount sweets_amount : ℝ)

noncomputable def Roshini_conditions (initial_amount friends_amount total_friends_amount sweets_amount : ℝ) :=
  initial_amount = 10.50 ∧ friends_amount = 6.80 ∧ sweets_amount = 3.70 ∧ 2 * 3.40 = 6.80

theorem roshini_spent_on_sweets :
  ∀ (initial_amount friends_amount total_friends_amount sweets_amount : ℝ),
    Roshini_conditions initial_amount friends_amount total_friends_amount sweets_amount →
    initial_amount - friends_amount = sweets_amount :=
by
  intros initial_amount friends_amount total_friends_amount sweets_amount h
  cases h
  sorry

end roshini_spent_on_sweets_l1779_177954


namespace combined_wattage_l1779_177952

theorem combined_wattage (w1 w2 w3 w4 : ℕ) (h1 : w1 = 60) (h2 : w2 = 80) (h3 : w3 = 100) (h4 : w4 = 120) :
  let nw1 := w1 + w1 / 4
  let nw2 := w2 + w2 / 4
  let nw3 := w3 + w3 / 4
  let nw4 := w4 + w4 / 4
  nw1 + nw2 + nw3 + nw4 = 450 :=
by
  sorry

end combined_wattage_l1779_177952


namespace inequality_solution_set_l1779_177976

theorem inequality_solution_set
  (a b c m n : ℝ) (h : a ≠ 0) 
  (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ m < x ∧ x < n)
  (h2 : 0 < m)
  (h3 : ∀ x : ℝ, cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) :
  (cx^2 + bx + a < 0 ↔ (x < 1 / n ∨ 1 / m < x)) := 
sorry

end inequality_solution_set_l1779_177976


namespace solve_for_m_l1779_177948

noncomputable def has_positive_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (m / (x - 3) - 1 / (3 - x) = 2)

theorem solve_for_m (m : ℝ) : has_positive_root m → m = -1 :=
sorry

end solve_for_m_l1779_177948


namespace no_positive_integral_solution_l1779_177998

theorem no_positive_integral_solution :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ p : ℕ, Prime p ∧ n^2 - 45 * n + 520 = p :=
by {
  -- Since we only need the statement, we'll introduce the necessary steps without the full proof
  sorry
}

end no_positive_integral_solution_l1779_177998


namespace angle_BAO_eq_angle_CAH_l1779_177934

noncomputable def is_triangle (A B C : Type) : Prop := sorry
noncomputable def orthocenter (A B C H : Type) : Prop := sorry
noncomputable def circumcenter (A B C O : Type) : Prop := sorry
noncomputable def angle (A B C : Type) : Type := sorry

theorem angle_BAO_eq_angle_CAH (A B C H O : Type) 
  (hABC : is_triangle A B C)
  (hH : orthocenter A B C H)
  (hO : circumcenter A B C O):
  angle B A O = angle C A H := 
  sorry

end angle_BAO_eq_angle_CAH_l1779_177934


namespace probability_two_females_one_male_l1779_177967

theorem probability_two_females_one_male :
  let total_contestants := 8
  let num_females := 5
  let num_males := 3
  let choose3 := Nat.choose total_contestants 3
  let choose2f := Nat.choose num_females 2
  let choose1m := Nat.choose num_males 1
  let favorable_outcomes := choose2f * choose1m
  choose3 ≠ 0 → (favorable_outcomes / choose3 : ℚ) = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l1779_177967


namespace negation_at_most_three_l1779_177925

theorem negation_at_most_three :
  ¬ (∀ n : ℕ, n ≤ 3) ↔ (∃ n : ℕ, n ≥ 4) :=
by
  sorry

end negation_at_most_three_l1779_177925


namespace inverse_function_domain_l1779_177982

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem inverse_function_domain :
  ∃ (g : ℝ → ℝ), (∀ x, 0 ≤ x → f (g x) = x) ∧ (∀ y, 0 ≤ y → g (f y) = y) ∧ (∀ x, 0 ≤ x ↔ 0 ≤ g x) :=
by
  sorry

end inverse_function_domain_l1779_177982


namespace robot_distance_proof_l1779_177914

noncomputable def distance (south1 south2 south3 east1 east2 : ℝ) : ℝ :=
  Real.sqrt ((south1 + south2 + south3)^2 + (east1 + east2)^2)

theorem robot_distance_proof :
  distance 1.2 1.8 1.0 1.0 2.0 = 5.0 :=
by
  sorry

end robot_distance_proof_l1779_177914
