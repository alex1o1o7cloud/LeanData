import Mathlib

namespace NUMINAMATH_GPT_simplest_fraction_l1483_148330

theorem simplest_fraction (x y : ℝ) (h1 : 2 * x ≠ 0) (h2 : x + y ≠ 0) :
  let A := (2 * x) / (4 * x^2)
  let B := (x^2 + y^2) / (x + y)
  let C := (x^2 + 2 * x + 1) / (x + 1)
  let D := (x^2 - 4) / (x + 2)
  B = (x^2 + y^2) / (x + y) ∧
  A ≠ (2 * x) / (4 * x^2) ∧
  C ≠ (x^2 + 2 * x + 1) / (x + 1) ∧
  D ≠ (x^2 - 4) / (x + 2) := sorry

end NUMINAMATH_GPT_simplest_fraction_l1483_148330


namespace NUMINAMATH_GPT_perpendicular_lines_l1483_148360

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + y + 1 = 0) ∧ (∀ x y : ℝ, x + a * y + 3 = 0) ∧ (∀ A1 B1 A2 B2 : ℝ, A1 * A2 + B1 * B2 = 0) →
  a = -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1483_148360


namespace NUMINAMATH_GPT_number_of_gigs_played_l1483_148316

-- Definitions based on given conditions
def earnings_per_member : ℕ := 20
def number_of_members : ℕ := 4
def total_earnings : ℕ := 400

-- Proof statement in Lean 4
theorem number_of_gigs_played : (total_earnings / (earnings_per_member * number_of_members)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_gigs_played_l1483_148316


namespace NUMINAMATH_GPT_a_2017_eq_2_l1483_148392

variable (n : ℕ)
variable (S : ℕ → ℤ)

/-- Define the sequence sum Sn -/
def S_n (n : ℕ) : ℤ := 2 * n - 1

/-- Define the sequence term an -/
def a_n (n : ℕ) : ℤ := S_n n - S_n (n - 1)

theorem a_2017_eq_2 : a_n 2017 = 2 := 
by
  have hSn : ∀ n, S_n n = (2 * n - 1) := by intro; simp [S_n] 
  have ha : ∀ n, a_n n = (S_n n - S_n (n - 1)) := by intro; simp [a_n]
  simp only [ha, hSn] 
  sorry

end NUMINAMATH_GPT_a_2017_eq_2_l1483_148392


namespace NUMINAMATH_GPT_playback_methods_proof_l1483_148381

/-- A TV station continuously plays 5 advertisements, consisting of 3 different commercial advertisements
and 2 different Olympic promotional advertisements. The requirements are:
  1. The last advertisement must be an Olympic promotional advertisement.
  2. The 2 Olympic promotional advertisements can be played consecutively.
-/
def number_of_playback_methods (commercials olympics: ℕ) (last_ad_olympic: Bool) (olympics_consecutive: Bool) : ℕ :=
  if commercials = 3 ∧ olympics = 2 ∧ last_ad_olympic ∧ olympics_consecutive then 36 else 0

theorem playback_methods_proof :
  number_of_playback_methods 3 2 true true = 36 := by
  sorry

end NUMINAMATH_GPT_playback_methods_proof_l1483_148381


namespace NUMINAMATH_GPT_xyz_sum_divisible_l1483_148334

-- Define variables and conditions
variable (p x y z : ℕ) [Fact (Prime p)]
variable (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
variable (h_eq1 : x^3 % p = y^3 % p)
variable (h_eq2 : y^3 % p = z^3 % p)

-- Theorem statement
theorem xyz_sum_divisible (p x y z : ℕ) [Fact (Prime p)]
  (h1 : 0 < x) (h2 : x < y) (h3 : y < z) (h4 : z < p)
  (h_eq1 : x^3 % p = y^3 % p)
  (h_eq2 : y^3 % p = z^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := 
  sorry

end NUMINAMATH_GPT_xyz_sum_divisible_l1483_148334


namespace NUMINAMATH_GPT_turn_all_black_l1483_148329

def invertColor (v : Vertex) (G : Graph) : Graph := sorry

theorem turn_all_black (G : Graph) (n : ℕ) (whiteBlack : Vertex → Bool) :
  (∀ v : Vertex, whiteBlack v = false) :=
by
 -- Providing the base case for induction
  induction n with 
  | zero => sorry -- The base case for graphs with one vertex
  | succ n ih =>
    -- Inductive step: assume true for graph with n vertices and prove for graph with n+1 vertices
    sorry

end NUMINAMATH_GPT_turn_all_black_l1483_148329


namespace NUMINAMATH_GPT_solve_equation_l1483_148358

theorem solve_equation : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1483_148358


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l1483_148364

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 + 4 * x + k = 0) ↔ k ≤ 4 := by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l1483_148364


namespace NUMINAMATH_GPT_red_balls_in_bag_l1483_148311

theorem red_balls_in_bag (total_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) :
  total_balls = 60 → 
  white_balls = 22 → 
  green_balls = 18 → 
  yellow_balls = 8 → 
  purple_balls = 7 → 
  prob_neither_red_nor_purple = 0.8 → 
  ( ∃ (red_balls : ℕ), red_balls = 5 ) :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  sorry

end NUMINAMATH_GPT_red_balls_in_bag_l1483_148311


namespace NUMINAMATH_GPT_truck_distance_l1483_148301

theorem truck_distance (d: ℕ) (g: ℕ) (eff: ℕ) (new_g: ℕ) (total_distance: ℕ)
  (h1: d = 300) (h2: g = 10) (h3: eff = d / g) (h4: new_g = 15) (h5: total_distance = eff * new_g):
  total_distance = 450 :=
sorry

end NUMINAMATH_GPT_truck_distance_l1483_148301


namespace NUMINAMATH_GPT_fraction_simplification_l1483_148359

theorem fraction_simplification :
  (36 / 19) * (57 / 40) * (95 / 171) = (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1483_148359


namespace NUMINAMATH_GPT_pens_sold_during_promotion_l1483_148355

theorem pens_sold_during_promotion (x y n : ℕ) 
  (h_profit: 12 * x + 7 * y = 2011)
  (h_n: n = 2 * x + y) : 
  n = 335 := by
  sorry

end NUMINAMATH_GPT_pens_sold_during_promotion_l1483_148355


namespace NUMINAMATH_GPT_bag_with_cracks_number_l1483_148303

def marbles : List ℕ := [18, 19, 21, 23, 25, 34]

def total_marbles : ℕ := marbles.sum

def modulo_3 (n : ℕ) : ℕ := n % 3

theorem bag_with_cracks_number :
  ∃ (c : ℕ), c ∈ marbles ∧ 
    (total_marbles - c) % 3 = 0 ∧
    c = 23 :=
by 
  sorry

end NUMINAMATH_GPT_bag_with_cracks_number_l1483_148303


namespace NUMINAMATH_GPT_find_matrix_triples_elements_l1483_148399

theorem find_matrix_triples_elements (M A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ (a b c d : ℝ), A = ![![a, b], ![c, d]] -> M * A = ![![3 * a, 3 * b], ![3 * c, 3 * d]]) :
  M = ![![3, 0], ![0, 3]] :=
by
  sorry

end NUMINAMATH_GPT_find_matrix_triples_elements_l1483_148399


namespace NUMINAMATH_GPT_genevieve_errors_fixed_l1483_148357

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end NUMINAMATH_GPT_genevieve_errors_fixed_l1483_148357


namespace NUMINAMATH_GPT_curve_representation_l1483_148309

def curve_set (x y : Real) : Prop := 
  ((x + y - 1) * Real.sqrt (x^2 + y^2 - 4) = 0)

def line_set (x y : Real) : Prop :=
  (x + y - 1 = 0) ∧ (x^2 + y^2 ≥ 4)

def circle_set (x y : Real) : Prop :=
  (x^2 + y^2 = 4)

theorem curve_representation (x y : Real) :
  curve_set x y ↔ (line_set x y ∨ circle_set x y) :=
sorry

end NUMINAMATH_GPT_curve_representation_l1483_148309


namespace NUMINAMATH_GPT_cracked_seashells_zero_l1483_148313

/--
Tom found 15 seashells, and Fred found 43 seashells. After cleaning, it was discovered that Fred had 28 more seashells than Tom. Prove that the number of cracked seashells is 0.
-/
theorem cracked_seashells_zero
(Tom_seashells : ℕ)
(Fred_seashells : ℕ)
(cracked_seashells : ℕ)
(Tom_after_cleaning : ℕ := Tom_seashells - cracked_seashells)
(Fred_after_cleaning : ℕ := Fred_seashells - cracked_seashells)
(h1 : Tom_seashells = 15)
(h2 : Fred_seashells = 43)
(h3 : Fred_after_cleaning = Tom_after_cleaning + 28) :
  cracked_seashells = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_cracked_seashells_zero_l1483_148313


namespace NUMINAMATH_GPT_building_height_l1483_148379

theorem building_height
    (flagpole_height : ℝ)
    (flagpole_shadow_length : ℝ)
    (building_shadow_length : ℝ)
    (h : ℝ)
    (h_eq : flagpole_height / flagpole_shadow_length = h / building_shadow_length)
    (flagpole_height_eq : flagpole_height = 18)
    (flagpole_shadow_length_eq : flagpole_shadow_length = 45)
    (building_shadow_length_eq : building_shadow_length = 65) :
  h = 26 := by
  sorry

end NUMINAMATH_GPT_building_height_l1483_148379


namespace NUMINAMATH_GPT_sum_of_fractions_le_half_l1483_148343

theorem sum_of_fractions_le_half {a b c : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 1) :
  1 / (a^2 + 2 * b^2 + 3) + 1 / (b^2 + 2 * c^2 + 3) + 1 / (c^2 + 2 * a^2 + 3) ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_le_half_l1483_148343


namespace NUMINAMATH_GPT_length_of_bridge_l1483_148384

theorem length_of_bridge (speed : ℝ) (time_min : ℝ) (length : ℝ)
  (h_speed : speed = 5) (h_time : time_min = 15) :
  length = 1250 :=
sorry

end NUMINAMATH_GPT_length_of_bridge_l1483_148384


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1483_148383

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  ((x > 1) ∨ (y > 2)) → (x + y > 3) ∧ ¬((x > 1) ∨ (y > 2) ↔ (x + y > 3)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1483_148383


namespace NUMINAMATH_GPT_cosine_F_in_triangle_DEF_l1483_148322

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end NUMINAMATH_GPT_cosine_F_in_triangle_DEF_l1483_148322


namespace NUMINAMATH_GPT_students_attending_swimming_class_l1483_148390

theorem students_attending_swimming_class 
  (total_students : ℕ) 
  (chess_percentage : ℕ) 
  (swimming_percentage : ℕ) 
  (number_of_students : ℕ)
  (chess_students := chess_percentage * total_students / 100)
  (swimming_students := swimming_percentage * chess_students / 100) 
  (condition1 : total_students = 2000)
  (condition2 : chess_percentage = 10)
  (condition3 : swimming_percentage = 50)
  (condition4 : number_of_students = chess_students) :
  swimming_students = 100 := 
by 
  sorry

end NUMINAMATH_GPT_students_attending_swimming_class_l1483_148390


namespace NUMINAMATH_GPT_mrs_hilt_apple_pies_l1483_148312

-- Given definitions
def total_pies := 30 * 5
def pecan_pies := 16

-- The number of apple pies
def apple_pies := total_pies - pecan_pies

-- The proof statement
theorem mrs_hilt_apple_pies : apple_pies = 134 :=
by
  sorry -- Proof step to be filled

end NUMINAMATH_GPT_mrs_hilt_apple_pies_l1483_148312


namespace NUMINAMATH_GPT_product_of_roots_eq_neg25_l1483_148388

theorem product_of_roots_eq_neg25 : 
  ∀ (x : ℝ), 24 * x^2 + 36 * x - 600 = 0 → x * (x - ((-36 - 24 * x)/24)) = -25 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_neg25_l1483_148388


namespace NUMINAMATH_GPT_minutes_in_hours_l1483_148307

theorem minutes_in_hours (h : ℝ) (m : ℝ) (H : h = 3.5) (M : m = 60) : h * m = 210 := by
  sorry

end NUMINAMATH_GPT_minutes_in_hours_l1483_148307


namespace NUMINAMATH_GPT_linear_system_solution_l1483_148351

theorem linear_system_solution (k x y : ℝ) (h₁ : x + y = 5 * k) (h₂ : x - y = 9 * k) (h₃ : 2 * x + 3 * y = 6) :
  k = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_linear_system_solution_l1483_148351


namespace NUMINAMATH_GPT_find_YW_in_triangle_l1483_148353

theorem find_YW_in_triangle
  (X Y Z W : Type)
  (d_XZ d_YZ d_XW d_CW : ℝ)
  (h_XZ : d_XZ = 10)
  (h_YZ : d_YZ = 10)
  (h_XW : d_XW = 12)
  (h_CW : d_CW = 5) : 
  YW = 29 / 12 :=
sorry

end NUMINAMATH_GPT_find_YW_in_triangle_l1483_148353


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1483_148349

theorem inscribed_circle_radius (A p r s : ℝ) (h₁ : A = 2 * p) (h₂ : p = 2 * s) (h₃ : A = r * s) : r = 4 :=
by sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1483_148349


namespace NUMINAMATH_GPT_mark_initial_kept_percentage_l1483_148393

-- Defining the conditions
def initial_friends : Nat := 100
def remaining_friends : Nat := 70
def percentage_contacted (P : ℝ) := 100 - P
def percentage_responded : ℝ := 0.5

-- Theorem statement: Mark initially kept 40% of his friends
theorem mark_initial_kept_percentage (P : ℝ) : 
  (P / 100 * initial_friends) + (percentage_contacted P / 100 * initial_friends * percentage_responded) = remaining_friends → 
  P = 40 := by
  sorry

end NUMINAMATH_GPT_mark_initial_kept_percentage_l1483_148393


namespace NUMINAMATH_GPT_find_cans_lids_l1483_148325

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end NUMINAMATH_GPT_find_cans_lids_l1483_148325


namespace NUMINAMATH_GPT_sum_of_first_8_terms_l1483_148308

theorem sum_of_first_8_terms (seq : ℕ → ℝ) (q : ℝ) (h_q : q = 2) 
  (h_sum_first_4 : seq 0 + seq 1 + seq 2 + seq 3 = 1) 
  (h_geom : ∀ n, seq (n + 1) = q * seq n) : 
  seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6 + seq 7 = 17 := 
sorry

end NUMINAMATH_GPT_sum_of_first_8_terms_l1483_148308


namespace NUMINAMATH_GPT_jogger_distance_ahead_l1483_148340

theorem jogger_distance_ahead
  (train_speed_km_hr : ℝ) (jogger_speed_km_hr : ℝ)
  (train_length_m : ℝ) (time_seconds : ℝ)
  (relative_speed_m_s : ℝ) (distance_covered_m : ℝ)
  (D : ℝ)
  (h1 : train_speed_km_hr = 45)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_length_m = 100)
  (h4 : time_seconds = 25)
  (h5 : relative_speed_m_s = 36 * (5/18))
  (h6 : distance_covered_m = 10 * 25)
  (h7 : D + train_length_m = distance_covered_m) :
  D = 150 :=
by sorry

end NUMINAMATH_GPT_jogger_distance_ahead_l1483_148340


namespace NUMINAMATH_GPT_fishing_problem_l1483_148306

theorem fishing_problem
  (everyday : ℕ)
  (every_other_day : ℕ)
  (every_three_days : ℕ)
  (yesterday_fishing : ℕ)
  (today_fishing : ℕ)
  (h_everyday : everyday = 7)
  (h_every_other_day : every_other_day = 8)
  (h_every_three_days : every_three_days = 3)
  (h_yesterday_fishing : yesterday_fishing = 12)
  (h_today_fishing : today_fishing = 10) :
  (every_three_days + everyday + (every_other_day - (yesterday_fishing - everyday))) = 15 := by
  sorry

end NUMINAMATH_GPT_fishing_problem_l1483_148306


namespace NUMINAMATH_GPT_circle_area_circle_circumference_l1483_148317

section CircleProperties

variable (r : ℝ) -- Define the radius of the circle as a real number

-- State the theorem for the area of the circle
theorem circle_area (A : ℝ) : A = π * r^2 :=
sorry

-- State the theorem for the circumference of the circle
theorem circle_circumference (C : ℝ) : C = 2 * π * r :=
sorry

end CircleProperties

end NUMINAMATH_GPT_circle_area_circle_circumference_l1483_148317


namespace NUMINAMATH_GPT_find_angle_x_l1483_148305

theorem find_angle_x (angle_ABC angle_BAC angle_BCA angle_DCE angle_CED x : ℝ)
  (h1 : angle_ABC + angle_BAC + angle_BCA = 180)
  (h2 : angle_ABC = 70) 
  (h3 : angle_BAC = 50)
  (h4 : angle_DCE + angle_CED = 90)
  (h5 : angle_DCE = angle_BCA) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_x_l1483_148305


namespace NUMINAMATH_GPT_remainder_when_divided_by_23_l1483_148378

theorem remainder_when_divided_by_23 (y : ℕ) (h : y % 276 = 42) : y % 23 = 19 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_23_l1483_148378


namespace NUMINAMATH_GPT_number_value_l1483_148376

theorem number_value (N : ℝ) (h : 0.40 * N = 180) : 
  (1/4) * (1/3) * (2/5) * N = 15 :=
by
  -- assume the conditions have been stated correctly
  sorry

end NUMINAMATH_GPT_number_value_l1483_148376


namespace NUMINAMATH_GPT_probability_sum_divisible_by_3_l1483_148324

theorem probability_sum_divisible_by_3:
  ∀ (n a b c : ℕ), a + b + c = n →
  4 * (a^3 + b^3 + c^3 + 6 * a * b * c) ≥ (a + b + c)^3 :=
by 
  intros n a b c habc_eq_n
  sorry

end NUMINAMATH_GPT_probability_sum_divisible_by_3_l1483_148324


namespace NUMINAMATH_GPT_second_crane_height_l1483_148315

noncomputable def height_of_second_crane : ℝ :=
  let crane1 := 228
  let building1 := 200
  let building2 := 100
  let crane3 := 147
  let building3 := 140
  let avg_building_height := (building1 + building2 + building3) / 3
  let avg_crane_height := avg_building_height * 1.13
  let h := (avg_crane_height * 3) - (crane1 - building1 + crane3 - building3) + building2
  h

theorem second_crane_height : height_of_second_crane = 122 := 
  sorry

end NUMINAMATH_GPT_second_crane_height_l1483_148315


namespace NUMINAMATH_GPT_inverse_proportion_comparison_l1483_148320

theorem inverse_proportion_comparison (y1 y2 : ℝ) 
  (h1 : y1 = - 6 / 2)
  (h2 : y2 = - 6 / -1) : 
  y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_comparison_l1483_148320


namespace NUMINAMATH_GPT_florist_total_roses_l1483_148370

-- Define the known quantities
def originalRoses : ℝ := 37.0
def firstPick : ℝ := 16.0
def secondPick : ℝ := 19.0

-- The theorem stating the total number of roses
theorem florist_total_roses : originalRoses + firstPick + secondPick = 72.0 :=
  sorry

end NUMINAMATH_GPT_florist_total_roses_l1483_148370


namespace NUMINAMATH_GPT_minimize_a_plus_b_l1483_148336

theorem minimize_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 4 * a + b = 30) :
  a + b = 9 → (a, b) = (7, 2) := sorry

end NUMINAMATH_GPT_minimize_a_plus_b_l1483_148336


namespace NUMINAMATH_GPT_students_wearing_other_colors_l1483_148363

-- Definitions according to the problem conditions
def total_students : ℕ := 900
def percentage_blue : ℕ := 44
def percentage_red : ℕ := 28
def percentage_green : ℕ := 10

-- Goal: Prove the number of students who wear other colors
theorem students_wearing_other_colors :
  (total_students * (100 - (percentage_blue + percentage_red + percentage_green))) / 100 = 162 :=
by
  -- Skipping the proof steps with sorry
  sorry

end NUMINAMATH_GPT_students_wearing_other_colors_l1483_148363


namespace NUMINAMATH_GPT_isabella_paintable_area_l1483_148398

def total_paintable_area : ℕ :=
  let room1_area := 2 * (14 * 9) + 2 * (12 * 9) - 70
  let room2_area := 2 * (13 * 9) + 2 * (11 * 9) - 70
  let room3_area := 2 * (15 * 9) + 2 * (10 * 9) - 70
  let room4_area := 4 * (12 * 9) - 70
  room1_area + room2_area + room3_area + room4_area

theorem isabella_paintable_area : total_paintable_area = 1502 := by
  sorry

end NUMINAMATH_GPT_isabella_paintable_area_l1483_148398


namespace NUMINAMATH_GPT_pq_conditions_l1483_148382

theorem pq_conditions (p q : ℝ) (hp : p > 1) (hq : q > 1) (hq_inverse : 1 / p + 1 / q = 1) (hpq : p * q = 9) :
  (p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨ (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2) :=
  sorry

end NUMINAMATH_GPT_pq_conditions_l1483_148382


namespace NUMINAMATH_GPT_problem_A_inter_B_empty_l1483_148380

section

def set_A : Set ℝ := {x | |x| ≥ 2}
def set_B : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_A_inter_B_empty : set_A ∩ set_B = ∅ := 
  sorry

end

end NUMINAMATH_GPT_problem_A_inter_B_empty_l1483_148380


namespace NUMINAMATH_GPT_number_of_boys_l1483_148348

theorem number_of_boys
  (M W B : Nat)
  (total_earnings wages_of_men earnings_of_men : Nat)
  (num_men_eq_women : 5 * M = W)
  (num_men_eq_boys : 5 * M = B)
  (earnings_eq_90 : total_earnings = 90)
  (men_wages_6 : wages_of_men = 6)
  (men_earnings_eq_30 : earnings_of_men = M * wages_of_men) : 
  B = 5 := 
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l1483_148348


namespace NUMINAMATH_GPT_racers_meet_at_start_again_l1483_148369

-- We define the conditions as given
def RacingMagic_time := 60
def ChargingBull_time := 60 * 60 / 40 -- 90 seconds
def SwiftShadow_time := 80
def SpeedyStorm_time := 100

-- Prove the LCM of their lap times is 3600 seconds,
-- which is equivalent to 60 minutes.
theorem racers_meet_at_start_again :
  Nat.lcm (Nat.lcm (Nat.lcm RacingMagic_time ChargingBull_time) SwiftShadow_time) SpeedyStorm_time = 3600 ∧
  3600 / 60 = 60 := by
  sorry

end NUMINAMATH_GPT_racers_meet_at_start_again_l1483_148369


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1483_148338

-- lean statement
theorem triangle_is_isosceles (a b c : ℝ) (C : ℝ) (h : a = 2 * b * Real.cos C) : 
  ∃ k : ℝ, a = k ∧ b = k := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1483_148338


namespace NUMINAMATH_GPT_thomas_score_l1483_148318

def average (scores : List ℕ) : ℚ := scores.sum / scores.length

variable (scores : List ℕ)

theorem thomas_score (h_length : scores.length = 19)
                     (h_avg_before : average scores = 78)
                     (h_avg_after : average ((98 :: scores)) = 79) :
  let thomas_score := 98
  thomas_score = 98 := sorry

end NUMINAMATH_GPT_thomas_score_l1483_148318


namespace NUMINAMATH_GPT_prove_minimality_of_smallest_three_digit_multiple_of_17_l1483_148319

def smallest_three_digit_multiple_of_17: ℕ :=
  102

theorem prove_minimality_of_smallest_three_digit_multiple_of_17 (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k ∧ 100 ≤ 17 * k ∧ 17 * k < 1000) 
  → n = 102 :=
    by
      sorry

end NUMINAMATH_GPT_prove_minimality_of_smallest_three_digit_multiple_of_17_l1483_148319


namespace NUMINAMATH_GPT_hyperbola_sqrt3_eccentricity_l1483_148367

noncomputable def hyperbola_eccentricity (m : ℝ) : ℝ :=
  let a := 2
  let b := m
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_sqrt3_eccentricity (m : ℝ) (h_m_pos : 0 < m) (h_slope : m = 2 * Real.sqrt 2) :
  hyperbola_eccentricity m = Real.sqrt 3 :=
by
  unfold hyperbola_eccentricity
  rw [h_slope]
  simp
  sorry

end NUMINAMATH_GPT_hyperbola_sqrt3_eccentricity_l1483_148367


namespace NUMINAMATH_GPT_dart_board_probability_l1483_148300

variable {s : ℝ} (hexagon_area : ℝ := (3 * Real.sqrt 3) / 2 * s^2) (center_hexagon_area : ℝ := (3 * Real.sqrt 3) / 8 * s^2)

theorem dart_board_probability (s : ℝ) (P : ℝ) (h : P = center_hexagon_area / hexagon_area) :
  P = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_dart_board_probability_l1483_148300


namespace NUMINAMATH_GPT_find_circle_center_l1483_148374

theorem find_circle_center
  (x y : ℝ)
  (h1 : 5 * x - 4 * y = 10)
  (h2 : 3 * x - y = 0)
  : x = -10 / 7 ∧ y = -30 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_circle_center_l1483_148374


namespace NUMINAMATH_GPT_find_alpha_plus_beta_l1483_148368

variable (α β : ℝ)

def condition_1 : Prop := α^3 - 3*α^2 + 5*α = 1
def condition_2 : Prop := β^3 - 3*β^2 + 5*β = 5

theorem find_alpha_plus_beta (h1 : condition_1 α) (h2 : condition_2 β) : α + β = 2 := 
  sorry

end NUMINAMATH_GPT_find_alpha_plus_beta_l1483_148368


namespace NUMINAMATH_GPT_vector_parallel_l1483_148361

theorem vector_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (3 * (2 * x + 1) - 4 * (2 - x) = 0) → (x = 1 / 2) :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_vector_parallel_l1483_148361


namespace NUMINAMATH_GPT_initial_books_count_l1483_148302

-- Definitions in conditions
def books_sold : ℕ := 42
def books_left : ℕ := 66

-- The theorem to prove the initial books count
theorem initial_books_count (initial_books : ℕ) : initial_books = books_sold + books_left :=
  by sorry

end NUMINAMATH_GPT_initial_books_count_l1483_148302


namespace NUMINAMATH_GPT_b_can_finish_work_in_15_days_l1483_148346

theorem b_can_finish_work_in_15_days (W : ℕ) (r_A : ℕ) (r_B : ℕ) (h1 : r_A = W / 21) (h2 : 10 * r_B + 7 * r_A / 21 = W) : r_B = W / 15 :=
by sorry

end NUMINAMATH_GPT_b_can_finish_work_in_15_days_l1483_148346


namespace NUMINAMATH_GPT_candy_cost_correct_l1483_148344

-- Given conditions:
def given_amount : ℝ := 1.00
def change_received : ℝ := 0.46

-- Define candy cost based on given conditions
def candy_cost : ℝ := given_amount - change_received

-- Statement to be proved
theorem candy_cost_correct : candy_cost = 0.54 := 
by
  sorry

end NUMINAMATH_GPT_candy_cost_correct_l1483_148344


namespace NUMINAMATH_GPT_royal_children_count_l1483_148332

theorem royal_children_count :
  ∀ (d n : ℕ), 
    d ≥ 1 → 
    n = 35 / (d + 1) →
    (d + 3) ≤ 20 →
    (d + 3 = 7 ∨ d + 3 = 9) :=
by
  intros d n H1 H2 H3
  sorry

end NUMINAMATH_GPT_royal_children_count_l1483_148332


namespace NUMINAMATH_GPT_gcd_g50_g51_l1483_148365

-- Define the polynomial g(x)
def g (x : ℤ) : ℤ := x^2 + x + 2023

-- State the theorem with necessary conditions
theorem gcd_g50_g51 : Int.gcd (g 50) (g 51) = 17 :=
by
  -- Goals and conditions stated
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_gcd_g50_g51_l1483_148365


namespace NUMINAMATH_GPT_denominator_of_fractions_l1483_148371

theorem denominator_of_fractions (y a : ℝ) (hy : y > 0) 
  (h : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_denominator_of_fractions_l1483_148371


namespace NUMINAMATH_GPT_tiles_painted_in_15_minutes_l1483_148354

open Nat

theorem tiles_painted_in_15_minutes:
  let don_rate := 3
  let ken_rate := don_rate + 2
  let laura_rate := 2 * ken_rate
  let kim_rate := laura_rate - 3
  don_rate + ken_rate + laura_rate + kim_rate == 25 → 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = 375 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tiles_painted_in_15_minutes_l1483_148354


namespace NUMINAMATH_GPT_functional_relationship_l1483_148347

-- Define the conditions and question for Scenario ①
def scenario1 (x y k : ℝ) (h1 : k ≠ 0) : Prop :=
  y = k / x

-- Define the conditions and question for Scenario ②
def scenario2 (n S k : ℝ) (h2 : k ≠ 0) : Prop :=
  S = k / n

-- Define the conditions and question for Scenario ③
def scenario3 (t s k : ℝ) (h3 : k ≠ 0) : Prop :=
  s = k * t

-- The main theorem
theorem functional_relationship (x y n S t s k : ℝ) (h1 : k ≠ 0) :
  (scenario1 x y k h1) ∧ (scenario2 n S k h1) ∧ ¬(scenario3 t s k h1) := 
sorry

end NUMINAMATH_GPT_functional_relationship_l1483_148347


namespace NUMINAMATH_GPT_probability_X_eq_2_l1483_148337

namespace Hypergeometric

def combin (n k : ℕ) : ℕ := n.choose k

noncomputable def hypergeometric (N M n k : ℕ) : ℚ :=
  (combin M k * combin (N - M) (n - k)) / combin N n

theorem probability_X_eq_2 :
  hypergeometric 8 5 3 2 = 15 / 28 := by
  sorry

end Hypergeometric

end NUMINAMATH_GPT_probability_X_eq_2_l1483_148337


namespace NUMINAMATH_GPT_f_increasing_f_at_2_solve_inequality_l1483_148323

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a + f b - 1
axiom f_pos (x : ℝ) (h : x > 0) : f x > 1
axiom f_at_4 : f 4 = 5

theorem f_increasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

theorem f_at_2 : f 2 = 3 :=
sorry

theorem solve_inequality (m : ℝ) : f (3 * m^2 - m - 2) < 3 ↔ -1 < m ∧ m < 4 / 3 :=
sorry

end NUMINAMATH_GPT_f_increasing_f_at_2_solve_inequality_l1483_148323


namespace NUMINAMATH_GPT_pages_to_read_tomorrow_l1483_148314

-- Define the problem setup
def total_pages : ℕ := 100
def pages_yesterday : ℕ := 35
def pages_today : ℕ := pages_yesterday - 5

-- Define the total pages read after two days
def pages_read_in_two_days : ℕ := pages_yesterday + pages_today

-- Define the number of pages left to read
def pages_left_to_read (total_pages read_so_far : ℕ) : ℕ := total_pages - read_so_far

-- Prove that the number of pages to read tomorrow is 35
theorem pages_to_read_tomorrow :
  pages_left_to_read total_pages pages_read_in_two_days = 35 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_pages_to_read_tomorrow_l1483_148314


namespace NUMINAMATH_GPT_solve_for_x_l1483_148341

theorem solve_for_x (x : ℝ) (h : 40 / x - 1 = 19) : x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1483_148341


namespace NUMINAMATH_GPT_latest_time_for_60_degrees_l1483_148366

def temperature_at_time (t : ℝ) : ℝ :=
  -2 * t^2 + 16 * t + 40

theorem latest_time_for_60_degrees (t : ℝ) :
  temperature_at_time t = 60 → t = 5 :=
sorry

end NUMINAMATH_GPT_latest_time_for_60_degrees_l1483_148366


namespace NUMINAMATH_GPT_number_of_fours_is_even_l1483_148345

theorem number_of_fours_is_even 
  (x y z : ℕ) 
  (h1 : x + y + z = 80) 
  (h2 : 3 * x + 4 * y + 5 * z = 276) : 
  Even y :=
by
  sorry

end NUMINAMATH_GPT_number_of_fours_is_even_l1483_148345


namespace NUMINAMATH_GPT_total_paths_from_X_to_Z_l1483_148333

variable (X Y Z : Type)
variables (f : X → Y → Z)
variables (g : X → Z)

-- Conditions
def paths_X_to_Y : ℕ := 3
def paths_Y_to_Z : ℕ := 4
def direct_paths_X_to_Z : ℕ := 1

-- Proof problem statement
theorem total_paths_from_X_to_Z : paths_X_to_Y * paths_Y_to_Z + direct_paths_X_to_Z = 13 := sorry

end NUMINAMATH_GPT_total_paths_from_X_to_Z_l1483_148333


namespace NUMINAMATH_GPT_equal_roots_quadratic_k_eq_one_l1483_148350

theorem equal_roots_quadratic_k_eq_one
  (k : ℝ)
  (h : ∃ x : ℝ, x^2 - 2 * x + k == 0 ∧ x^2 - 2 * x + k == 0) :
  k = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_equal_roots_quadratic_k_eq_one_l1483_148350


namespace NUMINAMATH_GPT_grandmother_cheapest_option_l1483_148372

-- Conditions definition
def cost_of_transportation : Nat := 200
def berries_collected : Nat := 5
def market_price_berries : Nat := 150
def price_sugar : Nat := 54
def amount_jam_from_1kg_berries_sugar : ℚ := 1.5
def cost_ready_made_jam_per_kg : Nat := 220

-- Calculations
def cost_per_kg_berries : ℚ := cost_of_transportation / berries_collected
def cost_bought_berries : Nat := market_price_berries
def total_cost_1kg_self_picked : ℚ := cost_per_kg_berries + price_sugar
def total_cost_1kg_bought : Nat := cost_bought_berries + price_sugar
def total_cost_1_5kg_self_picked : ℚ := total_cost_1kg_self_picked
def total_cost_1_5kg_bought : ℚ := total_cost_1kg_bought
def total_cost_1_5kg_ready_made : ℚ := cost_ready_made_jam_per_kg * amount_jam_from_1kg_berries_sugar

theorem grandmother_cheapest_option :
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_bought ∧ 
  total_cost_1_5kg_self_picked ≤ total_cost_1_5kg_ready_made :=
  by
    sorry

end NUMINAMATH_GPT_grandmother_cheapest_option_l1483_148372


namespace NUMINAMATH_GPT_convex_polygon_from_non_overlapping_rectangles_is_rectangle_l1483_148326

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

end NUMINAMATH_GPT_convex_polygon_from_non_overlapping_rectangles_is_rectangle_l1483_148326


namespace NUMINAMATH_GPT_problem1_problem2_l1483_148342

-- Definitions for first problem
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem for first problem
theorem problem1 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : ∀ x, -3 ≤ x → x ≤ 3) (h : f (m + 1) > f (2 * m - 1)) :
  -1 ≤ m ∧ m < 2 :=
sorry

-- Definitions for second problem
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem for second problem
theorem problem2 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : odd_function f) (h3 : f 2 = 1) (h4 : ∀ x, -3 ≤ x → x ≤ 3) :
  ∀ x, f (x + 1) + 1 > 0 ↔ -3 < x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1483_148342


namespace NUMINAMATH_GPT_sqrt_37_between_6_and_7_l1483_148389

theorem sqrt_37_between_6_and_7 : 6 < Real.sqrt 37 ∧ Real.sqrt 37 < 7 := 
by 
  have h₁ : Real.sqrt 36 = 6 := by sorry
  have h₂ : Real.sqrt 49 = 7 := by sorry
  sorry

end NUMINAMATH_GPT_sqrt_37_between_6_and_7_l1483_148389


namespace NUMINAMATH_GPT_laura_house_distance_l1483_148335

-- Definitions based on conditions
def x : Real := 10  -- Distance from Laura's house to her school in miles

def distance_to_school_per_day := 2 * x
def school_days_per_week := 5
def distance_to_school_per_week := school_days_per_week * distance_to_school_per_day

def distance_to_supermarket := x + 10
def supermarket_trips_per_week := 2
def distance_to_supermarket_per_trip := 2 * distance_to_supermarket
def distance_to_supermarket_per_week := supermarket_trips_per_week * distance_to_supermarket_per_trip

def total_distance_per_week := 220

-- The proof statement
theorem laura_house_distance :
  distance_to_school_per_week + distance_to_supermarket_per_week = total_distance_per_week ∧ x = 10 := by
  sorry

end NUMINAMATH_GPT_laura_house_distance_l1483_148335


namespace NUMINAMATH_GPT_sequences_properties_l1483_148321

-- Definitions for properties P and P'
def is_property_P (seq : List ℕ) : Prop := sorry
def is_property_P' (seq : List ℕ) : Prop := sorry

-- Define sequences
def sequence1 := [1, 2, 3, 1]
def sequence2 := [1, 234, 5]  -- Extend as needed

-- Conditions
def bn_is_permutation_of_an (a b : List ℕ) : Prop := sorry -- Placeholder for permutation check

-- Main Statement 
theorem sequences_properties :
  is_property_P sequence1 ∧
  is_property_P' sequence2 := 
by
  sorry

-- Additional theorem to check permutation if needed
-- theorem permutation_check :
--  bn_is_permutation_of_an sequence1 sequence2 :=
-- by
--  sorry

end NUMINAMATH_GPT_sequences_properties_l1483_148321


namespace NUMINAMATH_GPT_triangle_area_ABC_l1483_148387

variable {A : Prod ℝ ℝ}
variable {B : Prod ℝ ℝ}
variable {C : Prod ℝ ℝ}

noncomputable def area_of_triangle (A B C : Prod ℝ ℝ ) : ℝ :=
  (1 / 2) * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))

theorem triangle_area_ABC : 
  ∀ {A B C : Prod ℝ ℝ}, 
  A = (2, 3) → 
  B = (5, 7) → 
  C = (6, 1) → 
  area_of_triangle A B C = 11 
:= by
  intros
  subst_vars
  simp [area_of_triangle]
  sorry

end NUMINAMATH_GPT_triangle_area_ABC_l1483_148387


namespace NUMINAMATH_GPT_valid_combination_exists_l1483_148397

def exists_valid_combination : Prop :=
  ∃ (a: Fin 7 → ℤ), (a 0 = 1) ∧
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 4) ∧ 
  (a 4 = 5) ∧ (a 5 = 6) ∧ (a 6 = 7) ∧
  ((a 0 = a 1 + a 2 + a 3 + a 4 - a 5 - a 6))

theorem valid_combination_exists :
  exists_valid_combination :=
by
  sorry

end NUMINAMATH_GPT_valid_combination_exists_l1483_148397


namespace NUMINAMATH_GPT_slope_of_tangent_line_at_A_l1483_148395

noncomputable def f (x : ℝ) := x^2 + 3 * x

def derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (sorry : ℝ)  -- Placeholder for the definition of the derivative

theorem slope_of_tangent_line_at_A : 
  derivative_at f 1 = 5 := 
sorry

end NUMINAMATH_GPT_slope_of_tangent_line_at_A_l1483_148395


namespace NUMINAMATH_GPT_birds_reduction_on_third_day_l1483_148362

theorem birds_reduction_on_third_day
  {a b c : ℕ} 
  (h1 : a = 300)
  (h2 : b = 2 * a)
  (h3 : c = 1300)
  : (b - (c - (a + b))) = 200 :=
by sorry

end NUMINAMATH_GPT_birds_reduction_on_third_day_l1483_148362


namespace NUMINAMATH_GPT_Tonya_buys_3_lego_sets_l1483_148304

-- Definitions based on conditions
def num_sisters : Nat := 2
def num_dolls : Nat := 4
def price_per_doll : Nat := 15
def price_per_lego_set : Nat := 20

-- The amount of money spent on each sister should be the same
def amount_spent_on_younger_sister := num_dolls * price_per_doll
def amount_spent_on_older_sister := (amount_spent_on_younger_sister / price_per_lego_set)

-- Proof statement
theorem Tonya_buys_3_lego_sets : amount_spent_on_older_sister = 3 :=
by
  sorry

end NUMINAMATH_GPT_Tonya_buys_3_lego_sets_l1483_148304


namespace NUMINAMATH_GPT_max_min_x_plus_y_on_circle_l1483_148373

-- Define the conditions
def polar_eq (ρ θ : Real) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the standard form of the circle
def circle_eq (x y : Real) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations of the circle
def parametric_eq (α : Real) (x y : Real) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Define the problem in Lean
theorem max_min_x_plus_y_on_circle :
  (∀ (ρ θ : Real), polar_eq ρ θ → circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  (∀ (α : Real), parametric_eq α (2 + Real.sqrt 2 * Real.cos α) (2 + Real.sqrt 2 * Real.sin α)) →
  (∀ (P : Real × Real), circle_eq P.1 P.2 → 2 ≤ P.1 + P.2 ∧ P.1 + P.2 ≤ 6) :=
by
  intros hpolar hparam P hcircle
  sorry

end NUMINAMATH_GPT_max_min_x_plus_y_on_circle_l1483_148373


namespace NUMINAMATH_GPT_ann_age_l1483_148327

theorem ann_age {a b y : ℕ} (h1 : a + b = 44) (h2 : y = a - b) (h3 : b = a / 2 + 2 * (a - b)) : a = 24 :=
by
  sorry

end NUMINAMATH_GPT_ann_age_l1483_148327


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1483_148375

variable {U : Set ℤ}
variable {A : Set ℤ}

theorem complement_of_A_in_U (hU : U = {-1, 0, 1}) (hA : A = {0, 1}) : U \ A = {-1} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1483_148375


namespace NUMINAMATH_GPT_two_buckets_have_40_liters_l1483_148352

def liters_in_jug := 5
def jugs_in_bucket := 4
def liters_in_bucket := liters_in_jug * jugs_in_bucket
def buckets := 2

theorem two_buckets_have_40_liters :
  buckets * liters_in_bucket = 40 :=
by
  sorry

end NUMINAMATH_GPT_two_buckets_have_40_liters_l1483_148352


namespace NUMINAMATH_GPT_syllogism_error_l1483_148328

-- Definitions based on conditions from a)
def major_premise (a: ℝ) : Prop := a^2 > 0

def minor_premise (a: ℝ) : Prop := true

-- Theorem stating that the conclusion does not necessarily follow
theorem syllogism_error (a : ℝ) (h_minor : minor_premise a) : ¬major_premise 0 :=
by
  sorry

end NUMINAMATH_GPT_syllogism_error_l1483_148328


namespace NUMINAMATH_GPT_min_value_of_vector_sum_l1483_148339

noncomputable def min_vector_sum_magnitude (P Q: (ℝ×ℝ)) : ℝ :=
  let x := P.1
  let y := P.2
  let a := Q.1
  let b := Q.2
  Real.sqrt ((x + a)^2 + (y + b)^2)

theorem min_value_of_vector_sum :
  ∃ P Q, 
  (P.1 - 2)^2 + (P.2 - 2)^2 = 1 ∧ 
  Q.1 + Q.2 = 1 ∧ 
  min_vector_sum_magnitude P Q = (5 * Real.sqrt 2 - 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_vector_sum_l1483_148339


namespace NUMINAMATH_GPT_range_of_m_l1483_148386

open Set

variable {m : ℝ}

def A : Set ℝ := { x | x^2 < 16 }
def B (m : ℝ) : Set ℝ := { x | x < m }

theorem range_of_m (h : A ∩ B m = A) : 4 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1483_148386


namespace NUMINAMATH_GPT_abc_values_l1483_148356

theorem abc_values (a b c : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hab : b = a^2 / (2 - a^2)) 
  (hbc : c = b^2 / (2 - b^2)) 
  (hca : a = c^2 / (2 - c^2)) : 
  a + b + c = 6 ∨ a + b + c = -4 ∨ a + b + c = -6 :=
sorry

end NUMINAMATH_GPT_abc_values_l1483_148356


namespace NUMINAMATH_GPT_john_must_work_10_more_days_l1483_148331

-- Define the conditions as hypotheses
def total_days_worked := 10
def total_earnings := 250
def desired_total_earnings := total_earnings * 2
def daily_earnings := total_earnings / total_days_worked

-- Theorem that needs to be proved
theorem john_must_work_10_more_days:
  (desired_total_earnings / daily_earnings) - total_days_worked = 10 := by
  sorry

end NUMINAMATH_GPT_john_must_work_10_more_days_l1483_148331


namespace NUMINAMATH_GPT_journey_distance_last_day_l1483_148310

theorem journey_distance_last_day (S₆ : ℕ) (q : ℝ) (n : ℕ) (a₁ : ℝ) : 
  S₆ = 378 ∧ q = 1 / 2 ∧ n = 6 ∧ S₆ = a₁ * (1 - q^n) / (1 - q)
  → a₁ * q^(n - 1) = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_journey_distance_last_day_l1483_148310


namespace NUMINAMATH_GPT_largest_divisor_of_consecutive_even_product_l1483_148385

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℤ, k = 24 ∧ 
  (2 * n) * (2 * n + 2) * (2 * n + 4) % k = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_consecutive_even_product_l1483_148385


namespace NUMINAMATH_GPT_set_intersection_complement_eq_l1483_148396

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

noncomputable def complement (U B : Set ℕ) : Set ℕ := { x ∈ U | x ∉ B }

theorem set_intersection_complement_eq : (A ∩ (complement U B)) = {1, 3} := 
by 
  sorry

end NUMINAMATH_GPT_set_intersection_complement_eq_l1483_148396


namespace NUMINAMATH_GPT_binary_to_decimal_1100_l1483_148394

-- Define the binary number 1100
def binary_1100 : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0

-- State the theorem that we need to prove
theorem binary_to_decimal_1100 : binary_1100 = 12 := by
  rw [binary_1100]
  sorry

end NUMINAMATH_GPT_binary_to_decimal_1100_l1483_148394


namespace NUMINAMATH_GPT_smallest_b_for_factorization_l1483_148377

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end NUMINAMATH_GPT_smallest_b_for_factorization_l1483_148377


namespace NUMINAMATH_GPT_shaded_area_l1483_148391

theorem shaded_area (d : ℝ) (k : ℝ) (π : ℝ) (r : ℝ)
  (h_diameter : d = 6) 
  (h_radius_large : k = 5)
  (h_small_radius: r = d / 2) :
  ((π * (k * r)^2) - (π * r^2)) = 216 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l1483_148391
