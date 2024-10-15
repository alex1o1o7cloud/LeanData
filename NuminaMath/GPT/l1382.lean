import Mathlib

namespace NUMINAMATH_GPT_train_journey_duration_l1382_138207

def battery_lifespan (talk_time standby_time : ℝ) :=
  talk_time <= 6 ∧ standby_time <= 210

def full_battery_usage (total_time : ℝ) :=
  (total_time / 2) / 6 + (total_time / 2) / 210 = 1

theorem train_journey_duration (t : ℝ) (h1 : battery_lifespan (t / 2) (t / 2)) (h2 : full_battery_usage t) :
  t = 35 / 3 :=
sorry

end NUMINAMATH_GPT_train_journey_duration_l1382_138207


namespace NUMINAMATH_GPT_solve_for_y_l1382_138211

noncomputable def solve_quadratic := {y : ℂ // 4 + 3 * y^2 = 0.7 * y - 40}

theorem solve_for_y : 
  ∃ y : ℂ, (y = 0.1167 + 3.8273 * Complex.I ∨ y = 0.1167 - 3.8273 * Complex.I) ∧
            (4 + 3 * y^2 = 0.7 * y - 40) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1382_138211


namespace NUMINAMATH_GPT_problem_solution_l1382_138255

theorem problem_solution (x : ℝ) :
          ((3 * x - 4) * (x + 5) ≠ 0) → 
          (10 * x^3 + 20 * x^2 - 75 * x - 105) / ((3 * x - 4) * (x + 5)) < 5 ↔ 
          (x ∈ Set.Ioo (-5 : ℝ) (-1) ∪ Set.Ioi (4 / 3)) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1382_138255


namespace NUMINAMATH_GPT_number_of_common_tangents_l1382_138236

theorem number_of_common_tangents 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (circle2 : ∀ x y : ℝ, 2 * y^2 - 6 * x - 8 * y + 9 = 0) : 
  ∃ n : ℕ, n = 3 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_number_of_common_tangents_l1382_138236


namespace NUMINAMATH_GPT_proof_mod_55_l1382_138201

theorem proof_mod_55 (M : ℕ) (h1 : M % 5 = 3) (h2 : M % 11 = 9) : M % 55 = 53 := 
  sorry

end NUMINAMATH_GPT_proof_mod_55_l1382_138201


namespace NUMINAMATH_GPT_exists_coprime_integers_divisible_l1382_138243

theorem exists_coprime_integers_divisible {a b p : ℤ} : ∃ k l : ℤ, gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end NUMINAMATH_GPT_exists_coprime_integers_divisible_l1382_138243


namespace NUMINAMATH_GPT_largest_angle_in_ratio_3_4_5_l1382_138217

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_ratio_3_4_5_l1382_138217


namespace NUMINAMATH_GPT_bound_diff_sqrt_two_l1382_138246

theorem bound_diff_sqrt_two (a b k m : ℝ) (h : ∀ x ∈ Set.Icc a b, abs (x^2 - k * x - m) ≤ 1) : b - a ≤ 2 * Real.sqrt 2 := sorry

end NUMINAMATH_GPT_bound_diff_sqrt_two_l1382_138246


namespace NUMINAMATH_GPT_correct_sum_of_integers_l1382_138209

theorem correct_sum_of_integers (x y : ℕ) (h1 : x - y = 4) (h2 : x * y = 192) : x + y = 28 := by
  sorry

end NUMINAMATH_GPT_correct_sum_of_integers_l1382_138209


namespace NUMINAMATH_GPT_more_crayons_than_erasers_l1382_138274

theorem more_crayons_than_erasers
  (E : ℕ) (C : ℕ) (C_left : ℕ) (E_left : ℕ)
  (hE : E = 457) (hC : C = 617) (hC_left : C_left = 523) (hE_left : E_left = E) :
  C_left - E_left = 66 := 
by
  sorry

end NUMINAMATH_GPT_more_crayons_than_erasers_l1382_138274


namespace NUMINAMATH_GPT_tournament_committee_count_l1382_138282

-- Given conditions
def num_teams : ℕ := 5
def members_per_team : ℕ := 8
def committee_size : ℕ := 11
def nonhost_member_selection (n : ℕ) : ℕ := (n.choose 2) -- Selection of 2 members from non-host teams
def host_member_selection (n : ℕ) : ℕ := (n.choose 2)   -- Selection of 2 members from the remaining members of the host team; captain not considered in this choose as it's already selected

-- The total number of ways to form the required tournament committee
def total_committee_selections : ℕ :=
  num_teams * host_member_selection 7 * (nonhost_member_selection 8)^4

-- Proof stating the solution to the problem
theorem tournament_committee_count :
  total_committee_selections = 64534080 := by
  sorry

end NUMINAMATH_GPT_tournament_committee_count_l1382_138282


namespace NUMINAMATH_GPT_person_reaches_before_bus_l1382_138263

theorem person_reaches_before_bus (dist : ℝ) (speed1 speed2 : ℝ) (miss_time_minutes : ℝ) :
  dist = 2.2 → speed1 = 3 → speed2 = 6 → miss_time_minutes = 12 →
  ((60 : ℝ) * (dist/speed1) - miss_time_minutes) - ((60 : ℝ) * (dist/speed2)) = 10 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_person_reaches_before_bus_l1382_138263


namespace NUMINAMATH_GPT_ratio_problem_l1382_138284

variable (a b c d : ℚ)

theorem ratio_problem
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l1382_138284


namespace NUMINAMATH_GPT_train_length_l1382_138265

theorem train_length (speed_kmph : ℝ) (cross_time_sec : ℝ) (train_length : ℝ) :
  speed_kmph = 60 → cross_time_sec = 12 → train_length = 200.04 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1382_138265


namespace NUMINAMATH_GPT_problem_cos_tan_half_l1382_138257

open Real

theorem problem_cos_tan_half
  (α : ℝ)
  (hcos : cos α = -4/5)
  (hquad : π < α ∧ α < 3 * π / 2) :
  (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 :=
  sorry

end NUMINAMATH_GPT_problem_cos_tan_half_l1382_138257


namespace NUMINAMATH_GPT_part_a_part_b_l1382_138283

variable (p : ℕ → ℕ)
axiom primes_sequence : ∀ n, (∀ m < p n, m ∣ p n → m = 1 ∨ m = p n) ∧ p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 5 ∧ p 4 = 7 ∧ p 5 = 11

theorem part_a (n : ℕ) (h : n ≥ 5) : p n > 2 * n := 
  by sorry

theorem part_b (n : ℕ) : p n > 3 * n ↔ n ≥ 12 := 
  by sorry

end NUMINAMATH_GPT_part_a_part_b_l1382_138283


namespace NUMINAMATH_GPT_points_on_same_sphere_l1382_138256

-- Define the necessary structures and assumptions
variables {P : Type*} [MetricSpace P]

-- Definitions of spheres and points
structure Sphere (P : Type*) [MetricSpace P] :=
(center : P)
(radius : ℝ)
(positive_radius : 0 < radius)

def symmetric_point (S A1 : P) : P := sorry -- definition to get the symmetric point A2

-- Given conditions
variables (S A B C A1 B1 C1 A2 B2 C2 : P)
variable (omega : Sphere P)
variable (Omega : Sphere P)
variable (M_S_A : P) -- midpoint of SA
variable (M_S_B : P) -- midpoint of SB
variable (M_S_C : P) -- midpoint of SC

-- Assertions of conditions
axiom sphere_through_vertex : omega.center = S
axiom first_intersections : omega.radius = dist S A1 ∧ omega.radius = dist S B1 ∧ omega.radius = dist S C1
axiom omega_Omega_intersection : ∃ (circle_center : P) (plane_parallel_to_ABC : P), true-- some conditions indicating intersection
axiom symmetric_points_A1_A2 : A2 = symmetric_point S A1
axiom symmetric_points_B1_B2 : B2 = symmetric_point S B1
axiom symmetric_points_C1_C2 : C2 = symmetric_point S C1

-- The theorem to prove
theorem points_on_same_sphere : ∃ (sphere : Sphere P), 
  (dist sphere.center A) = sphere.radius ∧ 
  (dist sphere.center B) = sphere.radius ∧ 
  (dist sphere.center C) = sphere.radius ∧ 
  (dist sphere.center A2) = sphere.radius ∧ 
  (dist sphere.center B2) = sphere.radius ∧ 
  (dist sphere.center C2) = sphere.radius := 
sorry

end NUMINAMATH_GPT_points_on_same_sphere_l1382_138256


namespace NUMINAMATH_GPT_line_not_in_first_quadrant_l1382_138260

theorem line_not_in_first_quadrant (t : ℝ) : 
  (∀ x y : ℝ, ¬ ((0 < x ∧ 0 < y) ∧ (2 * t - 3) * x + y + 6 = 0)) ↔ t ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_line_not_in_first_quadrant_l1382_138260


namespace NUMINAMATH_GPT_cuboids_painted_l1382_138223

-- Let's define the conditions first
def faces_per_cuboid : ℕ := 6
def total_faces_painted : ℕ := 36

-- Now, we state the theorem we want to prove
theorem cuboids_painted (n : ℕ) (h : total_faces_painted = n * faces_per_cuboid) : n = 6 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_cuboids_painted_l1382_138223


namespace NUMINAMATH_GPT_exponent_sum_l1382_138214

variables (a : ℝ) (m n : ℝ)

theorem exponent_sum (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end NUMINAMATH_GPT_exponent_sum_l1382_138214


namespace NUMINAMATH_GPT_equation_of_parallel_line_l1382_138208

-- Definitions for conditions from the problem
def point_A : ℝ × ℝ := (3, 2)
def line_eq (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def parallel_slope : ℝ := -4

-- Proof problem statement
theorem equation_of_parallel_line (x y : ℝ) :
  (∃ (m b : ℝ), m = parallel_slope ∧ b = 2 + 4 * 3 ∧ y = m * (x - 3) + b) →
  4 * x + y - 14 = 0 :=
sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l1382_138208


namespace NUMINAMATH_GPT_mr_a_net_gain_l1382_138240

theorem mr_a_net_gain 
  (initial_value : ℝ)
  (sale_profit_percentage : ℝ)
  (buyback_loss_percentage : ℝ)
  (final_sale_price : ℝ) 
  (buyback_price : ℝ)
  (net_gain : ℝ) :
  initial_value = 12000 →
  sale_profit_percentage = 0.15 →
  buyback_loss_percentage = 0.12 →
  final_sale_price = initial_value * (1 + sale_profit_percentage) →
  buyback_price = final_sale_price * (1 - buyback_loss_percentage) →
  net_gain = final_sale_price - buyback_price →
  net_gain = 1656 :=
by
  sorry

end NUMINAMATH_GPT_mr_a_net_gain_l1382_138240


namespace NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l1382_138266

structure Rectangle :=
  (w x y z : ℕ)

def RectangleA := Rectangle.mk 5 1 9 2
def RectangleB := Rectangle.mk 2 0 6 3
def RectangleC := Rectangle.mk 6 7 4 1
def RectangleD := Rectangle.mk 8 4 3 5
def RectangleE := Rectangle.mk 7 3 8 0

-- Problem Statement: Given these rectangles, prove that the top leftmost rectangle is B.
theorem top_leftmost_rectangle_is_B 
  (A : Rectangle := RectangleA)
  (B : Rectangle := RectangleB)
  (C : Rectangle := RectangleC)
  (D : Rectangle := RectangleD)
  (E : Rectangle := RectangleE) : 
  B = Rectangle.mk 2 0 6 3 := 
sorry

end NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l1382_138266


namespace NUMINAMATH_GPT_line_through_point_parallel_l1382_138237

theorem line_through_point_parallel (p : ℝ × ℝ) (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0)
  (hp : a * p.1 + b * p.2 + c = 0) :
  ∃ k : ℝ, a * p.1 + b * p.2 + k = 0 :=
by
  use - (a * p.1 + b * p.2)
  sorry

end NUMINAMATH_GPT_line_through_point_parallel_l1382_138237


namespace NUMINAMATH_GPT_determine_f_peak_tourism_season_l1382_138218

noncomputable def f (n : ℕ) : ℝ := 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300

theorem determine_f :
  (∀ n : ℕ, f n = 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300) ∧
  (f 8 - f 2 = 400) ∧
  (f 2 = 100) :=
sorry

theorem peak_tourism_season (n : ℤ) :
  (6 ≤ n ∧ n ≤ 10) ↔ (200 * Real.cos (((Real.pi / 6) * n) + 2 * Real.pi / 3) + 300 >= 400) :=
sorry

end NUMINAMATH_GPT_determine_f_peak_tourism_season_l1382_138218


namespace NUMINAMATH_GPT_contestant_wins_quiz_l1382_138250

noncomputable def winProbability : ℚ :=
  let p_correct := (1 : ℚ) / 3
  let p_wrong := (2 : ℚ) / 3
  let binom := Nat.choose  -- binomial coefficient function
  ((binom 4 2 * (p_correct ^ 2) * (p_wrong ^ 2)) +
   (binom 4 3 * (p_correct ^ 3) * (p_wrong ^ 1)) +
   (binom 4 4 * (p_correct ^ 4) * (p_wrong ^ 0)))

theorem contestant_wins_quiz :
  winProbability = 11 / 27 :=
by
  simp [winProbability, Nat.choose]
  norm_num
  done

end NUMINAMATH_GPT_contestant_wins_quiz_l1382_138250


namespace NUMINAMATH_GPT_cuckoo_sounds_from_10_to_16_l1382_138200

-- Define a function for the cuckoo sounds per hour considering the clock
def cuckoo_sounds (h : ℕ) : ℕ :=
  if h ≤ 12 then h else h - 12

-- Define the total number of cuckoo sounds from 10:00 to 16:00
def total_cuckoo_sounds : ℕ :=
  (List.range' 10 (16 - 10 + 1)).map cuckoo_sounds |>.sum

theorem cuckoo_sounds_from_10_to_16 : total_cuckoo_sounds = 43 := by
  sorry

end NUMINAMATH_GPT_cuckoo_sounds_from_10_to_16_l1382_138200


namespace NUMINAMATH_GPT_train_speed_l1382_138270

theorem train_speed (train_length bridge_length cross_time : ℝ)
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : cross_time = 25) :
  (train_length + bridge_length) / cross_time = 16 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1382_138270


namespace NUMINAMATH_GPT_cosine_60_degrees_l1382_138247

theorem cosine_60_degrees : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_cosine_60_degrees_l1382_138247


namespace NUMINAMATH_GPT_cos_double_angle_nonpositive_l1382_138219

theorem cos_double_angle_nonpositive (α β : ℝ) (φ : ℝ) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := 
sorry

end NUMINAMATH_GPT_cos_double_angle_nonpositive_l1382_138219


namespace NUMINAMATH_GPT_find_length_of_AC_in_triangle_ABC_l1382_138294

noncomputable def length_AC_in_triangle_ABC
  (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3) :
  ℝ :=
  let cos_B := Real.cos (Real.pi / 3)
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_B
  Real.sqrt AC_squared

theorem find_length_of_AC_in_triangle_ABC :
  ∃ AC : ℝ, ∀ (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3),
    length_AC_in_triangle_ABC AB BC angle_B h_AB h_BC h_angle_B = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_find_length_of_AC_in_triangle_ABC_l1382_138294


namespace NUMINAMATH_GPT_volume_is_120_l1382_138224

namespace volume_proof

-- Definitions from the given conditions
variables (a b c : ℝ)
axiom ab_relation : a * b = 48
axiom bc_relation : b * c = 20
axiom ca_relation : c * a = 15

-- Goal to prove
theorem volume_is_120 : a * b * c = 120 := by
  sorry

end volume_proof

end NUMINAMATH_GPT_volume_is_120_l1382_138224


namespace NUMINAMATH_GPT_avg_gpa_8th_graders_l1382_138221

theorem avg_gpa_8th_graders :
  ∀ (GPA_6th GPA_8th : ℝ),
    GPA_6th = 93 →
    (∀ GPA_7th : ℝ, GPA_7th = GPA_6th + 2 →
    (GPA_6th + GPA_7th + GPA_8th) / 3 = 93 →
    GPA_8th = 91) :=
by
  intros GPA_6th GPA_8th h1 GPA_7th h2 h3
  sorry

end NUMINAMATH_GPT_avg_gpa_8th_graders_l1382_138221


namespace NUMINAMATH_GPT_inequality_proof_l1382_138288

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + 1 / b^3 - 1) * (b^3 + 1 / c^3 - 1) * (c^3 + 1 / a^3 - 1) ≤ (a * b * c + 1 / (a * b * c) - 1)^3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1382_138288


namespace NUMINAMATH_GPT_molly_age_l1382_138253

theorem molly_age
  (S M : ℕ)
  (h_ratio : S / M = 4 / 3)
  (h_sandy_future : S + 6 = 42)
  : M = 27 :=
sorry

end NUMINAMATH_GPT_molly_age_l1382_138253


namespace NUMINAMATH_GPT_team_X_finishes_with_more_points_than_Y_l1382_138244

-- Define the number of teams and games played
def numberOfTeams : ℕ := 8
def gamesPerTeam : ℕ := numberOfTeams - 1

-- Define the probability of winning (since each team has a 50% chance to win any game)
def probOfWin : ℝ := 0.5

-- Define the event that team X finishes with more points than team Y
noncomputable def probXFinishesMorePointsThanY : ℝ := 1 / 2

-- Statement to be proved: 
theorem team_X_finishes_with_more_points_than_Y :
  (∃ p : ℝ, p = probXFinishesMorePointsThanY) :=
sorry

end NUMINAMATH_GPT_team_X_finishes_with_more_points_than_Y_l1382_138244


namespace NUMINAMATH_GPT_round_nearest_hundredth_problem_l1382_138230

noncomputable def round_nearest_hundredth (x : ℚ) : ℚ :=
  let shifted := x * 100
  let rounded := if (shifted - shifted.floor) < 0.5 then shifted.floor else shifted.ceil
  rounded / 100

theorem round_nearest_hundredth_problem :
  let A := 34.561
  let B := 34.558
  let C := 34.5539999
  let D := 34.5601
  let E := 34.56444
  round_nearest_hundredth A = 34.56 ∧
  round_nearest_hundredth B = 34.56 ∧
  round_nearest_hundredth C ≠ 34.56 ∧
  round_nearest_hundredth D = 34.56 ∧
  round_nearest_hundredth E = 34.56 :=
sorry

end NUMINAMATH_GPT_round_nearest_hundredth_problem_l1382_138230


namespace NUMINAMATH_GPT_pencil_fraction_white_part_l1382_138204

theorem pencil_fraction_white_part
  (L : ℝ )
  (H1 : L = 9.333333333333332)
  (H2 : (1 / 8) * L + (7 / 12 * 7 / 8) * (7 / 8) * L + W * (7 / 8) * L = L) :
  W = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_pencil_fraction_white_part_l1382_138204


namespace NUMINAMATH_GPT_find_four_digit_squares_l1382_138264

theorem find_four_digit_squares (N : ℕ) (a b : ℕ) 
    (h1 : 100 ≤ N ∧ N < 10000)
    (h2 : 10 ≤ a ∧ a < 100)
    (h3 : 0 ≤ b ∧ b < 100)
    (h4 : N = 100 * a + b)
    (h5 : N = (a + b) ^ 2) : 
    N = 9801 ∨ N = 3025 ∨ N = 2025 :=
    sorry

end NUMINAMATH_GPT_find_four_digit_squares_l1382_138264


namespace NUMINAMATH_GPT_value_of_y_l1382_138298

theorem value_of_y (x y z : ℕ) (h1 : 3 * x = 3 / 4 * y) (h2 : x + z = 24) (h3 : z = 8) : y = 64 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_value_of_y_l1382_138298


namespace NUMINAMATH_GPT_complex_z_calculation_l1382_138273

theorem complex_z_calculation (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^99 + z^100 + z^101 + z^102 + z^103 = 1 + z :=
sorry

end NUMINAMATH_GPT_complex_z_calculation_l1382_138273


namespace NUMINAMATH_GPT_train_length_l1382_138227

theorem train_length (L : ℕ) (V : ℕ) (platform_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
    (h1 : V = L / time_pole) 
    (h2 : V = (L + platform_length) / time_platform) :
    L = 300 := 
by 
  -- The proof can be filled here
  sorry

end NUMINAMATH_GPT_train_length_l1382_138227


namespace NUMINAMATH_GPT_num_students_only_math_l1382_138220

def oakwood_ninth_grade_problem 
  (total_students: ℕ)
  (students_in_math: ℕ)
  (students_in_foreign_language: ℕ)
  (students_in_science: ℕ)
  (students_in_all_three: ℕ)
  (students_total_from_ie: ℕ) :=
  (total_students = 120) ∧
  (students_in_math = 85) ∧
  (students_in_foreign_language = 65) ∧
  (students_in_science = 75) ∧
  (students_in_all_three = 20) ∧
  total_students = students_in_math + students_in_foreign_language + students_in_science 
  - (students_total_from_ie) + students_in_all_three - (students_in_all_three)

theorem num_students_only_math 
  (total_students: ℕ := 120)
  (students_in_math: ℕ := 85)
  (students_in_foreign_language: ℕ := 65)
  (students_in_science: ℕ := 75)
  (students_in_all_three: ℕ := 20)
  (students_total_from_ie: ℕ := 45) :
  oakwood_ninth_grade_problem total_students students_in_math students_in_foreign_language students_in_science students_in_all_three students_total_from_ie →
  ∃ (students_only_math: ℕ), students_only_math = 75 :=
by
  sorry

end NUMINAMATH_GPT_num_students_only_math_l1382_138220


namespace NUMINAMATH_GPT_lake_coverage_day_17_l1382_138242

-- Define the state of lake coverage as a function of day
def lake_coverage (day : ℕ) : ℝ :=
  if day ≤ 20 then 2 ^ (day - 20) else 0

-- Prove that on day 17, the lake was covered by 12.5% algae
theorem lake_coverage_day_17 : lake_coverage 17 = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_lake_coverage_day_17_l1382_138242


namespace NUMINAMATH_GPT_sum_of_ages_26_l1382_138292

-- Define an age predicate to manage the three ages
def is_sum_of_ages (kiana twin : ℕ) : Prop :=
  kiana < twin ∧ twin * twin * kiana = 180 ∧ (kiana + twin + twin = 26)

theorem sum_of_ages_26 : 
  ∃ (kiana twin : ℕ), is_sum_of_ages kiana twin :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_ages_26_l1382_138292


namespace NUMINAMATH_GPT_balance_scale_comparison_l1382_138285

theorem balance_scale_comparison :
  (4 / 3) * Real.pi * (8 : ℝ)^3 > (4 / 3) * Real.pi * (3 : ℝ)^3 + (4 / 3) * Real.pi * (5 : ℝ)^3 :=
by
  sorry

end NUMINAMATH_GPT_balance_scale_comparison_l1382_138285


namespace NUMINAMATH_GPT_inequality_proof_l1382_138261

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : 
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9 * x * y * z ≥ 9 * (x * y + y * z + z * x) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1382_138261


namespace NUMINAMATH_GPT_evaluate_expression_l1382_138280

def a : ℕ := 3
def b : ℕ := 2

theorem evaluate_expression : (a^2 * a^5) / (b^2 / b^3) = 4374 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1382_138280


namespace NUMINAMATH_GPT_least_n_froods_score_l1382_138272

theorem least_n_froods_score (n : ℕ) : (n * (n + 1) / 2 > 12 * n) ↔ (n > 23) := 
by 
  sorry

end NUMINAMATH_GPT_least_n_froods_score_l1382_138272


namespace NUMINAMATH_GPT_repeating_decimal_product_l1382_138278

def repeating_decimal_12 := 12 / 99
def repeating_decimal_34 := 34 / 99

theorem repeating_decimal_product : (repeating_decimal_12 * repeating_decimal_34) = 136 / 3267 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l1382_138278


namespace NUMINAMATH_GPT_man_son_ratio_in_two_years_l1382_138269

noncomputable def man_and_son_age_ratio (M S : ℕ) (h1 : M = S + 25) (h2 : S = 23) : ℕ × ℕ :=
  let S_in_2_years := S + 2
  let M_in_2_years := M + 2
  (M_in_2_years / S_in_2_years, S_in_2_years / S_in_2_years)

theorem man_son_ratio_in_two_years : man_and_son_age_ratio 48 23 (by norm_num) (by norm_num) = (2, 1) :=
  sorry

end NUMINAMATH_GPT_man_son_ratio_in_two_years_l1382_138269


namespace NUMINAMATH_GPT_number_of_sets_X_l1382_138234

noncomputable def finite_set_problem (M A B : Finset ℕ) : Prop :=
  (M.card = 10) ∧ 
  (A ⊆ M) ∧ 
  (B ⊆ M) ∧ 
  (A ∩ B = ∅) ∧ 
  (A.card = 2) ∧ 
  (B.card = 3) ∧ 
  (∃ (X : Finset ℕ), X ⊆ M ∧ ¬(A ⊆ X) ∧ ¬(B ⊆ X))

theorem number_of_sets_X (M A B : Finset ℕ) (h : finite_set_problem M A B) : 
  ∃ n : ℕ, n = 672 := 
sorry

end NUMINAMATH_GPT_number_of_sets_X_l1382_138234


namespace NUMINAMATH_GPT_one_cow_one_bag_l1382_138241

theorem one_cow_one_bag {days_per_bag : ℕ} (h : 50 * days_per_bag = 50 * 50) : days_per_bag = 50 :=
by
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_l1382_138241


namespace NUMINAMATH_GPT_all_inequalities_hold_l1382_138258

variables (a b c x y z : ℝ)

-- Conditions
def condition1 : Prop := x^2 < a^2
def condition2 : Prop := y^2 < b^2
def condition3 : Prop := z^2 < c^2

-- Inequalities to prove
def inequality1 : Prop := x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a^2 * b^2 + b^2 * c^2 + c^2 * a^2
def inequality2 : Prop := x^4 + y^4 + z^4 < a^4 + b^4 + c^4
def inequality3 : Prop := x^2 * y^2 * z^2 < a^2 * b^2 * c^2

theorem all_inequalities_hold (h1 : condition1 a x) (h2 : condition2 b y) (h3 : condition3 c z) :
  inequality1 a b c x y z ∧ inequality2 a b c x y z ∧ inequality3 a b c x y z := by
  sorry

end NUMINAMATH_GPT_all_inequalities_hold_l1382_138258


namespace NUMINAMATH_GPT_negation_universal_prop_l1382_138215

theorem negation_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end NUMINAMATH_GPT_negation_universal_prop_l1382_138215


namespace NUMINAMATH_GPT_yellow_paint_quarts_l1382_138271

theorem yellow_paint_quarts (ratio_r : ℕ) (ratio_y : ℕ) (ratio_w : ℕ) (qw : ℕ) : 
  ratio_r = 5 → ratio_y = 3 → ratio_w = 7 → qw = 21 → (qw * ratio_y) / ratio_w = 9 :=
by
  -- No proof required, inserting sorry to indicate missing proof
  sorry

end NUMINAMATH_GPT_yellow_paint_quarts_l1382_138271


namespace NUMINAMATH_GPT_expected_value_linear_combination_l1382_138287

variable (ξ η : ℝ)
variable (E : ℝ → ℝ)
axiom E_lin (a b : ℝ) (X Y : ℝ) : E (a * X + b * Y) = a * E X + b * E Y

axiom E_ξ : E ξ = 10
axiom E_η : E η = 3

theorem expected_value_linear_combination : E (3 * ξ + 5 * η) = 45 := by
  sorry

end NUMINAMATH_GPT_expected_value_linear_combination_l1382_138287


namespace NUMINAMATH_GPT_probability_of_queen_after_first_queen_l1382_138251

-- Define the standard deck
def standard_deck : Finset (Fin 54) := Finset.univ

-- Define the event of drawing the first queen
def first_queen (deck : Finset (Fin 54)) : Prop := -- placeholder defining first queen draw
  sorry

-- Define the event of drawing a queen immediately after the first queen
def queen_after_first_queen (deck : Finset (Fin 54)) : Prop :=
  sorry

-- Define the probability of an event given a condition
noncomputable def probability (event : Prop) (condition : Prop) : ℚ :=
  sorry

-- Main theorem statement
theorem probability_of_queen_after_first_queen : probability 
  (queen_after_first_queen standard_deck) (first_queen standard_deck) = 2/27 :=
sorry

end NUMINAMATH_GPT_probability_of_queen_after_first_queen_l1382_138251


namespace NUMINAMATH_GPT_solutions_of_system_l1382_138212

theorem solutions_of_system :
  ∀ (x y : ℝ), (x - 2 * y = 1) ∧ (x^3 - 8 * y^3 - 6 * x * y = 1) ↔ y = (x - 1) / 2 :=
by
  -- Since this is a statement-only task, the detailed proof is omitted.
  -- Insert actual proof here.
  sorry

end NUMINAMATH_GPT_solutions_of_system_l1382_138212


namespace NUMINAMATH_GPT_measure_of_angle_Z_l1382_138267

theorem measure_of_angle_Z (X Y Z : ℝ) (h_sum : X + Y + Z = 180) (h_XY : X + Y = 80) : Z = 100 := 
by
  -- The proof is not required.
  sorry

end NUMINAMATH_GPT_measure_of_angle_Z_l1382_138267


namespace NUMINAMATH_GPT_quadratic_solution_symmetry_l1382_138205

variable (a b c n : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : a * (-5)^2 + b * (-5) + c = -2.79)
variable (h₂ : a * 1^2 + b * 1 + c = -2.79)
variable (h₃ : a * 2^2 + b * 2 + c = 0)
variable (h₄ : a * 3^2 + b * 3 + c = n)

theorem quadratic_solution_symmetry :
  (x = 3 ∨ x = -7) ↔ (a * x^2 + b * x + c = n) :=
sorry

end NUMINAMATH_GPT_quadratic_solution_symmetry_l1382_138205


namespace NUMINAMATH_GPT_trader_loss_percentage_l1382_138286

theorem trader_loss_percentage :
  let SP := 325475
  let gain := 14 / 100
  let loss := 14 / 100
  let CP1 := SP / (1 + gain)
  let CP2 := SP / (1 - loss)
  let TCP := CP1 + CP2
  let TSP := SP + SP
  let profit_or_loss := TSP - TCP
  let profit_or_loss_percentage := (profit_or_loss / TCP) * 100
  profit_or_loss_percentage = -1.958 :=
by
  sorry

end NUMINAMATH_GPT_trader_loss_percentage_l1382_138286


namespace NUMINAMATH_GPT_problem_Ashwin_Sah_l1382_138297

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end NUMINAMATH_GPT_problem_Ashwin_Sah_l1382_138297


namespace NUMINAMATH_GPT_geometric_seq_a6_value_l1382_138268

theorem geometric_seq_a6_value 
    (a : ℕ → ℝ) 
    (q : ℝ) 
    (h_q_pos : q > 0)
    (h_a_pos : ∀ n, a n > 0)
    (h_a2 : a 2 = 1)
    (h_a8_eq : a 8 = a 6 + 2 * a 4) : 
    a 6 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_seq_a6_value_l1382_138268


namespace NUMINAMATH_GPT_groom_dog_time_l1382_138231

theorem groom_dog_time :
  ∃ (D : ℝ), (5 * D + 3 * 0.5 = 14) ∧ (D = 2.5) :=
by
  sorry

end NUMINAMATH_GPT_groom_dog_time_l1382_138231


namespace NUMINAMATH_GPT_yanni_money_left_in_cents_l1382_138225

-- Conditions
def initial_money : ℝ := 0.85
def money_from_mother : ℝ := 0.40
def money_found : ℝ := 0.50
def cost_per_toy : ℝ := 1.60
def number_of_toys : ℕ := 3
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Prove
theorem yanni_money_left_in_cents : 
  (initial_money + money_from_mother + money_found) * 100 = 175 :=
by
  sorry

end NUMINAMATH_GPT_yanni_money_left_in_cents_l1382_138225


namespace NUMINAMATH_GPT_revenue_highest_visitors_is_48_thousand_l1382_138295

-- Define the frequencies for each day
def freq_Oct_1 : ℝ := 0.05
def freq_Oct_2 : ℝ := 0.08
def freq_Oct_3 : ℝ := 0.09
def freq_Oct_4 : ℝ := 0.13
def freq_Oct_5 : ℝ := 0.30
def freq_Oct_6 : ℝ := 0.15
def freq_Oct_7 : ℝ := 0.20

-- Define the revenue on October 1st
def revenue_Oct_1 : ℝ := 80000

-- Define the revenue is directly proportional to the frequency of visitors
def avg_daily_visitor_spending_is_constant := true

-- The goal is to prove that the revenue on the day with the highest frequency is 48 thousand yuan
theorem revenue_highest_visitors_is_48_thousand :
  avg_daily_visitor_spending_is_constant →
  revenue_Oct_1 / freq_Oct_1 = x / freq_Oct_5 →
  x = 48000 :=
by
  sorry

end NUMINAMATH_GPT_revenue_highest_visitors_is_48_thousand_l1382_138295


namespace NUMINAMATH_GPT_problem1_problem2_l1382_138290

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x - 1

noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem1 (a : ℝ) (h1 : 2 / Real.exp 2 < a) (h2 : a < 1 / Real.exp 1) :
  ∃ (x1 x2 : ℝ), (0 < x1 ∧ x1 < 2) ∧ (0 < x2 ∧ x2 < 2) ∧ x1 ≠ x2 ∧ g x1 = a ∧ g x2 = a :=
sorry

theorem problem2 : ∀ x > 0, f x + 2 / (Real.exp 1 * g x) > 0 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1382_138290


namespace NUMINAMATH_GPT_maryann_free_time_l1382_138275

theorem maryann_free_time
    (x : ℕ)
    (expensive_time : ℕ := 8)
    (friends : ℕ := 3)
    (total_time : ℕ := 42)
    (lockpicking_time : 3 * (x + expensive_time) = total_time) : 
    x = 6 :=
by
  sorry

end NUMINAMATH_GPT_maryann_free_time_l1382_138275


namespace NUMINAMATH_GPT_is_odd_function_l1382_138228

def f (x : ℝ) : ℝ := x^3 - x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_is_odd_function_l1382_138228


namespace NUMINAMATH_GPT_production_days_l1382_138249

theorem production_days (n : ℕ) (h₁ : (50 * n + 95) / (n + 1) = 55) : 
    n = 8 := 
    sorry

end NUMINAMATH_GPT_production_days_l1382_138249


namespace NUMINAMATH_GPT_C_necessary_but_not_sufficient_for_A_l1382_138226

variable {A B C : Prop}

-- Given conditions
def sufficient_not_necessary (h : A → B) (hn : ¬(B → A)) := h
def necessary_sufficient := B ↔ C

-- Prove that C is a necessary but not sufficient condition for A
theorem C_necessary_but_not_sufficient_for_A (h₁ : A → B) (hn : ¬(B → A)) (h₂ : B ↔ C) : (C → A) ∧ ¬(A → C) :=
  by
  sorry

end NUMINAMATH_GPT_C_necessary_but_not_sufficient_for_A_l1382_138226


namespace NUMINAMATH_GPT_circle_intersection_value_l1382_138289

theorem circle_intersection_value {x1 y1 x2 y2 : ℝ} 
  (h_circle : x1^2 + y1^2 = 4)
  (h_non_negative : x1 ≥ 0 ∧ y1 ≥ 0 ∧ x2 ≥ 0 ∧ y2 ≥ 0)
  (h_symmetric : x1 = y2 ∧ x2 = y1) :
  x1^2 + x2^2 = 4 := 
by
  sorry

end NUMINAMATH_GPT_circle_intersection_value_l1382_138289


namespace NUMINAMATH_GPT_find_positive_x_l1382_138238

theorem find_positive_x (x : ℝ) (h1 : x * ⌊x⌋ = 72) (h2 : x > 0) : x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_positive_x_l1382_138238


namespace NUMINAMATH_GPT_problem_l1382_138232

def a := 1 / 4
def b := 1 / 2
def c := -3 / 4

def a_n (n : ℕ) : ℚ := 2 * n + 1
def S_n (n : ℕ) : ℚ := (n + 2) * n
def f (n : ℕ) : ℚ := 4 * a * n^2 + (4 * a + 2 * b) * n + (a + b + c)

theorem problem : ∀ n : ℕ, f n = S_n n := by
  sorry

end NUMINAMATH_GPT_problem_l1382_138232


namespace NUMINAMATH_GPT_expression_calculation_l1382_138291

theorem expression_calculation : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 28 * 21^1005 :=
by
  sorry

end NUMINAMATH_GPT_expression_calculation_l1382_138291


namespace NUMINAMATH_GPT_convex_polyhedron_has_triangular_face_l1382_138210

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end NUMINAMATH_GPT_convex_polyhedron_has_triangular_face_l1382_138210


namespace NUMINAMATH_GPT_pages_read_in_a_year_l1382_138216

-- Definition of the problem conditions
def novels_per_month := 4
def pages_per_novel := 200
def months_per_year := 12

-- Theorem statement corresponding to the problem
theorem pages_read_in_a_year (h1 : novels_per_month = 4) (h2 : pages_per_novel = 200) (h3 : months_per_year = 12) : 
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by
  sorry

end NUMINAMATH_GPT_pages_read_in_a_year_l1382_138216


namespace NUMINAMATH_GPT_problem1_problem2_l1382_138248

-- Problem 1
theorem problem1 (α : ℝ) (h : 2 * Real.sin α - Real.cos α = 0) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10 / 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) :
  (Real.sin x ^ 3 + Real.sin x * Real.cos x ^ 2) / (1 - Real.tan x) = 7 * Real.sqrt 2 / 60 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1382_138248


namespace NUMINAMATH_GPT_part1_part2_l1382_138279

-- Definition of the conditions given
def february_parcels : ℕ := 200000
def april_parcels : ℕ := 338000
def monthly_growth_rate : ℝ := 0.3

-- Problem 1: Proving the monthly growth rate is 0.3
theorem part1 (x : ℝ) (h : february_parcels * (1 + x)^2 = april_parcels) : x = monthly_growth_rate :=
  sorry

-- Problem 2: Proving the number of parcels in May is less than 450,000 with the given growth rate
theorem part2 (h : monthly_growth_rate = 0.3 ) : february_parcels * (1 + monthly_growth_rate)^3 < 450000 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1382_138279


namespace NUMINAMATH_GPT_angle_ABC_tangent_circle_l1382_138233

theorem angle_ABC_tangent_circle 
  (BAC ACB : ℝ)
  (h1 : BAC = 70)
  (h2 : ACB = 45)
  (D : Type)
  (incenter : ∀ D : Type, Prop)  -- Represent the condition that D is the incenter
  : ∃ ABC : ℝ, ABC = 65 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_tangent_circle_l1382_138233


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1382_138259

theorem ball_hits_ground_time (t : ℝ) : 
  (∃ t : ℝ, -10 * t^2 + 40 * t + 50 = 0 ∧ t ≥ 0) → t = 5 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1382_138259


namespace NUMINAMATH_GPT_largest_possible_value_n_l1382_138299

theorem largest_possible_value_n (n : ℕ) (h : ∀ m : ℕ, m ≠ n → n % m = 0 → m ≤ 35) : n = 35 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_n_l1382_138299


namespace NUMINAMATH_GPT_total_race_time_l1382_138262

theorem total_race_time 
  (num_runners : ℕ) 
  (first_five_time : ℕ) 
  (additional_time : ℕ) 
  (total_runners : ℕ) 
  (num_first_five : ℕ)
  (num_last_three : ℕ) 
  (total_expected_time : ℕ) 
  (h1 : num_runners = 8) 
  (h2 : first_five_time = 8) 
  (h3 : additional_time = 2) 
  (h4 : num_first_five = 5)
  (h5 : num_last_three = num_runners - num_first_five)
  (h6 : total_runners = num_first_five + num_last_three)
  (h7 : 5 * first_five_time + 3 * (first_five_time + additional_time) = total_expected_time)
  : total_expected_time = 70 := 
by
  sorry

end NUMINAMATH_GPT_total_race_time_l1382_138262


namespace NUMINAMATH_GPT_client_dropped_off_phones_l1382_138281

def initial_phones : ℕ := 15
def repaired_phones : ℕ := 3
def coworker_phones : ℕ := 9

theorem client_dropped_off_phones (x : ℕ) : 
  initial_phones - repaired_phones + x = 2 * coworker_phones → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_client_dropped_off_phones_l1382_138281


namespace NUMINAMATH_GPT_find_a_max_and_min_values_l1382_138202

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + 3*x + a + 1)

theorem find_a (a : ℝ) : (f' a 0) = 2 → a = 1 :=
by {
  -- Proof omitted
  sorry
}

theorem max_and_min_values (a : ℝ) :
  (a = 1) →
  (Real.exp (-2) * (4 - 2 + 1) = (3 / Real.exp 2)) ∧
  (Real.exp (-1) * (1 - 1 + 1) = (1 / Real.exp 1)) ∧
  (Real.exp 2 * (4 + 2 + 1) = (7 * Real.exp 2)) :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_find_a_max_and_min_values_l1382_138202


namespace NUMINAMATH_GPT_factor_exp_l1382_138203

variable (x : ℤ)

theorem factor_exp : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_exp_l1382_138203


namespace NUMINAMATH_GPT_isosceles_triangle_condition_l1382_138252

-- Theorem statement
theorem isosceles_triangle_condition (N : ℕ) (h : N > 2) : 
  (∃ N1 : ℕ, N = N1 ∧ N1 = 10) ∨ (∃ N2 : ℕ, N = N2 ∧ N2 = 11) :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_condition_l1382_138252


namespace NUMINAMATH_GPT_geometric_sequence_l1382_138235

theorem geometric_sequence (q : ℝ) (a : ℕ → ℝ) (h1 : q > 0) (h2 : a 2 = 1)
  (h3 : a 2 * a 10 = 2 * (a 5)^2) : ∀ n, a n = 2^((n-2:ℝ)/2) := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_l1382_138235


namespace NUMINAMATH_GPT_sqrt_mul_eq_6_l1382_138239

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_mul_eq_6_l1382_138239


namespace NUMINAMATH_GPT_greatest_x_integer_l1382_138254

theorem greatest_x_integer (x : ℤ) (h : ∃ n : ℤ, x^2 + 2 * x + 7 = (x - 4) * n) : x ≤ 35 :=
sorry

end NUMINAMATH_GPT_greatest_x_integer_l1382_138254


namespace NUMINAMATH_GPT_correct_option_l1382_138245

variable (a : ℤ)

theorem correct_option :
  (-2 * a^2)^3 = -8 * a^6 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1382_138245


namespace NUMINAMATH_GPT_B_alone_finishes_in_21_days_l1382_138276

theorem B_alone_finishes_in_21_days (W_A W_B : ℝ) (h1 : W_A = 0.5 * W_B) (h2 : W_A + W_B = 1 / 14) : W_B = 1 / 21 :=
by sorry

end NUMINAMATH_GPT_B_alone_finishes_in_21_days_l1382_138276


namespace NUMINAMATH_GPT_intersection_M_N_l1382_138296

def M := {y : ℝ | y <= 4}
def N := {x : ℝ | x > 0}

theorem intersection_M_N : {x : ℝ | x > 0} ∩ {y : ℝ | y <= 4} = {z : ℝ | 0 < z ∧ z <= 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1382_138296


namespace NUMINAMATH_GPT_evaluate_expression_l1382_138229

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1382_138229


namespace NUMINAMATH_GPT_value_of_m_l1382_138206

def f (x m : ℝ) : ℝ := x^2 - 2 * x + m
def g (x m : ℝ) : ℝ := x^2 - 2 * x + 2 * m + 8

theorem value_of_m (m : ℝ) : (3 * f 5 m = g 5 m) → m = -22 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_m_l1382_138206


namespace NUMINAMATH_GPT_subtract_fractions_l1382_138293

theorem subtract_fractions : (18 / 42 - 3 / 8) = 3 / 56 :=
by
  sorry

end NUMINAMATH_GPT_subtract_fractions_l1382_138293


namespace NUMINAMATH_GPT_pieces_per_block_is_32_l1382_138222

-- Define the number of pieces of junk mail given to each house
def pieces_per_house : ℕ := 8

-- Define the number of houses in each block
def houses_per_block : ℕ := 4

-- Calculate the total number of pieces of junk mail given to each block
def total_pieces_per_block : ℕ := pieces_per_house * houses_per_block

-- Prove that the total number of pieces of junk mail given to each block is 32
theorem pieces_per_block_is_32 : total_pieces_per_block = 32 := 
by sorry

end NUMINAMATH_GPT_pieces_per_block_is_32_l1382_138222


namespace NUMINAMATH_GPT_complex_z_eq_neg_i_l1382_138277

theorem complex_z_eq_neg_i (z : ℂ) (i : ℂ) (h1 : i * z = 1) (hi : i^2 = -1) : z = -i :=
sorry

end NUMINAMATH_GPT_complex_z_eq_neg_i_l1382_138277


namespace NUMINAMATH_GPT_triangle_is_right_l1382_138213

theorem triangle_is_right {A B C : ℝ} (h : A + B + C = 180) (h1 : A = B + C) : A = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_l1382_138213
