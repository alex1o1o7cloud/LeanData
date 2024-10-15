import Mathlib

namespace NUMINAMATH_GPT_SummitAcademy_Contestants_l927_92757

theorem SummitAcademy_Contestants (s j : ℕ)
  (h1 : s > 0)
  (h2 : j > 0)
  (hs : (1 / 3 : ℚ) * s = (3 / 4 : ℚ) * j) :
  s = (9 / 4 : ℚ) * j :=
sorry

end NUMINAMATH_GPT_SummitAcademy_Contestants_l927_92757


namespace NUMINAMATH_GPT_leah_earned_initially_l927_92707

noncomputable def initial_money (x : ℝ) : Prop :=
  let amount_after_milkshake := (6 / 7) * x
  let amount_left_wallet := (3 / 7) * x
  amount_left_wallet = 12

theorem leah_earned_initially (x : ℝ) (h : initial_money x) : x = 28 :=
by
  sorry

end NUMINAMATH_GPT_leah_earned_initially_l927_92707


namespace NUMINAMATH_GPT_downstream_speed_l927_92729

noncomputable def speed_downstream (Vu Vs : ℝ) : ℝ :=
  2 * Vs - Vu

theorem downstream_speed (Vu Vs : ℝ) (hVu : Vu = 30) (hVs : Vs = 45) :
  speed_downstream Vu Vs = 60 := by
  rw [hVu, hVs]
  dsimp [speed_downstream]
  linarith

end NUMINAMATH_GPT_downstream_speed_l927_92729


namespace NUMINAMATH_GPT_max_value_a_n_l927_92728

noncomputable def a_seq : ℕ → ℕ
| 0     => 0  -- By Lean's 0-based indexing, a_1 corresponds to a_seq 1
| 1     => 3
| (n+2) => a_seq (n+1) + 1

def S_n (n : ℕ) : ℕ := (n * (n + 5)) / 2

theorem max_value_a_n : 
  ∃ n : ℕ, S_n n = 2023 ∧ a_seq n = 73 :=
by
  sorry

end NUMINAMATH_GPT_max_value_a_n_l927_92728


namespace NUMINAMATH_GPT_find_f_l927_92712

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x / (a * x + b)

theorem find_f (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f 2 a b = 1) (h₂ : ∃! x, f x a b = x) :
  f x (1/2) 1 = 2 * x / (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_find_f_l927_92712


namespace NUMINAMATH_GPT_increasing_iff_a_ge_half_l927_92794

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x ^ 3 + (1 / 2) * (a - 1) * x ^ 2 + a * x + 1

theorem increasing_iff_a_ge_half (a : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (2 * x ^ 2 + (a - 1) * x + a) ≥ 0) ↔ a ≥ -1 / 2 :=
sorry

end NUMINAMATH_GPT_increasing_iff_a_ge_half_l927_92794


namespace NUMINAMATH_GPT_clara_weight_l927_92784

-- Define the weights of Alice and Clara
variables (a c : ℕ)

-- Define the conditions given in the problem
def condition1 := a + c = 240
def condition2 := c - a = c / 3

-- The theorem to prove Clara's weight given the conditions
theorem clara_weight : condition1 a c → condition2 a c → c = 144 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_clara_weight_l927_92784


namespace NUMINAMATH_GPT_selection_probability_equal_l927_92733

theorem selection_probability_equal :
  let n := 2012
  let eliminated := 12
  let remaining := n - eliminated
  let selected := 50
  let probability := (remaining / n) * (selected / remaining)
  probability = 25 / 1006 :=
by
  sorry

end NUMINAMATH_GPT_selection_probability_equal_l927_92733


namespace NUMINAMATH_GPT_sqrt_expression_range_l927_92779

theorem sqrt_expression_range (x : ℝ) : x + 3 ≥ 0 ∧ x ≠ 0 ↔ x ≥ -3 ∧ x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_range_l927_92779


namespace NUMINAMATH_GPT_intersection_complement_eq_l927_92732

open Set

variable (U M N : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  N ∩ (U \ M) = {3, 5} := by 
sorry

end NUMINAMATH_GPT_intersection_complement_eq_l927_92732


namespace NUMINAMATH_GPT_simplify_expression_l927_92791

theorem simplify_expression (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + ((a + 1) / (1 - a)))) = (1 + a) / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l927_92791


namespace NUMINAMATH_GPT_chocolate_factory_production_l927_92725

theorem chocolate_factory_production
  (candies_per_hour : ℕ)
  (total_candies : ℕ)
  (days : ℕ)
  (total_hours : ℕ := total_candies / candies_per_hour)
  (hours_per_day : ℕ := total_hours / days)
  (h1 : candies_per_hour = 50)
  (h2 : total_candies = 4000)
  (h3 : days = 8) :
  hours_per_day = 10 := by
  sorry

end NUMINAMATH_GPT_chocolate_factory_production_l927_92725


namespace NUMINAMATH_GPT_geometric_progression_product_sum_sumrecip_l927_92719

theorem geometric_progression_product_sum_sumrecip (P S S' : ℝ) (n : ℕ)
  (hP : P = a ^ n * r ^ ((n * (n - 1)) / 2))
  (hS : S = a * (1 - r ^ n) / (1 - r))
  (hS' : S' = (r ^ n - 1) / (a * (r - 1))) :
  P = (S / S') ^ (1 / 2 * n) :=
  sorry

end NUMINAMATH_GPT_geometric_progression_product_sum_sumrecip_l927_92719


namespace NUMINAMATH_GPT_least_number_of_pennies_l927_92740

theorem least_number_of_pennies (a : ℕ) :
  (a ≡ 1 [MOD 7]) ∧ (a ≡ 0 [MOD 3]) → a = 15 := by
  sorry

end NUMINAMATH_GPT_least_number_of_pennies_l927_92740


namespace NUMINAMATH_GPT_num_ways_arrange_l927_92772

open Finset

def valid_combinations : Finset (Finset Nat) :=
  { {2, 5, 11, 3}, {3, 5, 6, 2}, {3, 6, 11, 5}, {5, 6, 11, 2} }

theorem num_ways_arrange : valid_combinations.card = 4 :=
  by
    sorry  -- proof of the statement

end NUMINAMATH_GPT_num_ways_arrange_l927_92772


namespace NUMINAMATH_GPT_slope_of_line_I_l927_92711

-- Line I intersects y = 1 at point P
def intersects_y_eq_one (I P : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, P (x, 1) ↔ I (x, y) ∧ y = 1

-- Line I intersects x - y - 7 = 0 at point Q
def intersects_x_minus_y_eq_seven (I Q : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, Q (x, y) ↔ I (x, y) ∧ x - y - 7 = 0

-- The coordinates of the midpoint of segment PQ are (1, -1)
def midpoint_eq (P Q : ℝ × ℝ) : Prop :=
∃ x1 y1 x2 y2 : ℝ,
  P = (x1, y1) ∧ Q = (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)

-- We need to show that the slope of line I is -2/3
def slope_of_I (I : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
∀ x y : ℝ, I (x, y) → y + 1 = k * (x - 1)

theorem slope_of_line_I :
  ∃ I P Q : (ℝ × ℝ → Prop),
    intersects_y_eq_one I P ∧
    intersects_x_minus_y_eq_seven I Q ∧
    (∃ x1 y1 x2 y2 : ℝ, P (x1, y1) ∧ Q (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)) →
    slope_of_I I (-2/3) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_I_l927_92711


namespace NUMINAMATH_GPT_student_correct_answers_l927_92793

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 60) (h2 : 4 * C - W = 140) : C = 40 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l927_92793


namespace NUMINAMATH_GPT_time_to_cross_lake_one_direction_l927_92797

-- Definitions for our conditions
def cost_per_hour := 10
def total_cost_round_trip := 80

-- Statement we want to prove
theorem time_to_cross_lake_one_direction : (total_cost_round_trip / cost_per_hour) / 2 = 4 :=
  by
  sorry

end NUMINAMATH_GPT_time_to_cross_lake_one_direction_l927_92797


namespace NUMINAMATH_GPT_probability_red_or_blue_l927_92726

noncomputable def total_marbles : ℕ := 100

noncomputable def probability_white : ℚ := 1 / 4

noncomputable def probability_green : ℚ := 1 / 5

theorem probability_red_or_blue :
  (1 - (probability_white + probability_green)) = 11 / 20 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_probability_red_or_blue_l927_92726


namespace NUMINAMATH_GPT_complex_z_1000_l927_92781

open Complex

theorem complex_z_1000 (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (Real.pi * 5 / 180)) :
  z^(1000 : ℕ) + (z^(1000 : ℕ))⁻¹ = 2 * Real.cos (Real.pi * 20 / 180) :=
sorry

end NUMINAMATH_GPT_complex_z_1000_l927_92781


namespace NUMINAMATH_GPT_renaldo_distance_l927_92720

theorem renaldo_distance (R : ℕ) (h : R + (1/3 : ℝ) * R + 7 = 27) : R = 15 :=
by sorry

end NUMINAMATH_GPT_renaldo_distance_l927_92720


namespace NUMINAMATH_GPT_number_of_friends_l927_92759

def initial_candies : ℕ := 10
def additional_candies : ℕ := 4
def total_candies : ℕ := initial_candies + additional_candies
def candies_per_friend : ℕ := 2

theorem number_of_friends : total_candies / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_friends_l927_92759


namespace NUMINAMATH_GPT_ellipse_center_x_coordinate_l927_92749

theorem ellipse_center_x_coordinate (C : ℝ × ℝ)
  (h1 : C.1 = 3)
  (h2 : 4 ≤ C.2 ∧ C.2 ≤ 12)
  (hx : ∃ F1 F2 : ℝ × ℝ, F1 = (3, 4) ∧ F2 = (3, 12)
    ∧ (F1.1 = F2.1 ∧ F1.2 < F2.2)
    ∧ C = ((F1.1 + F2.1)/2, (F1.2 + F2.2)/2))
  (tangent : ∀ P : ℝ × ℝ, (P.1 - 0) * (P.2 - 0) = 0)
  (ellipse : ∃ a b : ℝ, a > 0 ∧ b > 0
    ∧ ∀ P : ℝ × ℝ,
      (P.1 - C.1)^2/a^2 + (P.2 - C.2)^2/b^2 = 1) :
   C.1 = 3 := sorry

end NUMINAMATH_GPT_ellipse_center_x_coordinate_l927_92749


namespace NUMINAMATH_GPT_max_non_overlapping_areas_l927_92704

theorem max_non_overlapping_areas (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 4 * n + 4 := 
sorry

end NUMINAMATH_GPT_max_non_overlapping_areas_l927_92704


namespace NUMINAMATH_GPT_f_increasing_l927_92770

noncomputable def f (x : Real) : Real := (2 * Real.exp x) / (1 + Real.exp x) + 1/2

theorem f_increasing : ∀ x y : Real, x < y → f x < f y := 
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_f_increasing_l927_92770


namespace NUMINAMATH_GPT_standard_circle_eq_l927_92705

noncomputable def circle_equation : String :=
  "The standard equation of the circle whose center lies on the line y = -4x and is tangent to the line x + y - 1 = 0 at point P(3, -2) is (x - 1)^2 + (y + 4)^2 = 8"

theorem standard_circle_eq
  (center_x : ℝ)
  (center_y : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (point : ℝ × ℝ)
  (eqn_line : ∀ x y, tangent_line x y ↔ x + y - 1 = 0)
  (center_on_line : ∀ x y, y = -4 * x → center_y = y)
  (point_on_tangent : point = (3, -2))
  (tangent_at_point : tangent_line (point.1) (point.2)) :
  (center_x = 1 ∧ center_y = -4 ∧ (∃ r : ℝ, r = 2 * Real.sqrt 2)) →
  (∀ x y, (x - 1)^2 + (y + 4)^2 = 8) := by
  sorry

end NUMINAMATH_GPT_standard_circle_eq_l927_92705


namespace NUMINAMATH_GPT_count_numbers_with_cube_root_lt_8_l927_92751

theorem count_numbers_with_cube_root_lt_8 : 
  ∀ n : ℕ, (n > 0) → (n < 8^3) → n ≤ 8^3 - 1 :=
by
  -- We need to prove that the count of such numbers is 511
  sorry

end NUMINAMATH_GPT_count_numbers_with_cube_root_lt_8_l927_92751


namespace NUMINAMATH_GPT_ratio_of_toys_l927_92730

theorem ratio_of_toys (total_toys : ℕ) (num_friends : ℕ) (toys_D : ℕ) 
  (h1 : total_toys = 118) 
  (h2 : num_friends = 4) 
  (h3 : toys_D = total_toys / num_friends) : 
  (toys_D / total_toys : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_toys_l927_92730


namespace NUMINAMATH_GPT_alarm_clock_shows_noon_in_14_minutes_l927_92739

-- Definitions based on given problem conditions
def clockRunsSlow (clock_time real_time : ℕ) : Prop :=
  clock_time = real_time * 56 / 60

def timeSinceSet : ℕ := 210 -- 3.5 hours in minutes
def correctClockShowsNoon : ℕ := 720 -- Noon in minutes (12*60)

-- Main statement to prove
theorem alarm_clock_shows_noon_in_14_minutes :
  ∃ minutes : ℕ, clockRunsSlow (timeSinceSet * 56 / 60) timeSinceSet ∧ correctClockShowsNoon - (480 + timeSinceSet * 56 / 60) = minutes ∧ minutes = 14 := 
by
  sorry

end NUMINAMATH_GPT_alarm_clock_shows_noon_in_14_minutes_l927_92739


namespace NUMINAMATH_GPT_emma_final_amount_l927_92790

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end NUMINAMATH_GPT_emma_final_amount_l927_92790


namespace NUMINAMATH_GPT_min_value_expression_eq_2sqrt3_l927_92727

noncomputable def min_value_expression (c d : ℝ) : ℝ :=
  c^2 + d^2 + 4 / c^2 + 2 * d / c

theorem min_value_expression_eq_2sqrt3 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, (∀ d : ℝ, min_value_expression c d ≥ y) ∧ y = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_expression_eq_2sqrt3_l927_92727


namespace NUMINAMATH_GPT_new_boarder_ratio_l927_92723

structure School where
  initial_boarders : ℕ
  day_students : ℕ
  boarders_ratio : ℚ

theorem new_boarder_ratio (S : School) (additional_boarders : ℕ) :
  S.initial_boarders = 60 →
  S.boarders_ratio = 2 / 5 →
  additional_boarders = 15 →
  S.day_students = (60 * 5) / 2 →
  (S.initial_boarders + additional_boarders) / S.day_students = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_new_boarder_ratio_l927_92723


namespace NUMINAMATH_GPT_no_separation_sister_chromatids_first_meiotic_l927_92774

-- Definitions for the steps happening during the first meiotic division
def first_meiotic_division :=
  ∃ (prophase_I : Prop) (metaphase_I : Prop) (anaphase_I : Prop) (telophase_I : Prop),
    prophase_I ∧ metaphase_I ∧ anaphase_I ∧ telophase_I

def pairing_homologous_chromosomes (prophase_I : Prop) := prophase_I
def crossing_over (prophase_I : Prop) := prophase_I
def separation_homologous_chromosomes (anaphase_I : Prop) := anaphase_I
def separation_sister_chromatids (mitosis : Prop) (second_meiotic_division : Prop) :=
  mitosis ∨ second_meiotic_division

-- Theorem to prove that the separation of sister chromatids does not occur during the first meiotic division
theorem no_separation_sister_chromatids_first_meiotic
  (prophase_I metaphase_I anaphase_I telophase_I mitosis second_meiotic_division : Prop)
  (h1: first_meiotic_division)
  (h2 : pairing_homologous_chromosomes prophase_I)
  (h3 : crossing_over prophase_I)
  (h4 : separation_homologous_chromosomes anaphase_I)
  (h5 : separation_sister_chromatids mitosis second_meiotic_division) : 
  ¬ separation_sister_chromatids prophase_I anaphase_I :=
by
  sorry

end NUMINAMATH_GPT_no_separation_sister_chromatids_first_meiotic_l927_92774


namespace NUMINAMATH_GPT_distance_from_apex_l927_92717

theorem distance_from_apex (A B : ℝ)
  (h_A : A = 216 * Real.sqrt 3)
  (h_B : B = 486 * Real.sqrt 3)
  (distance_planes : ℝ)
  (h_distance_planes : distance_planes = 8) :
  ∃ h : ℝ, h = 24 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_apex_l927_92717


namespace NUMINAMATH_GPT_find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l927_92778

variables {m : ℝ} 
def point_on_y_axis (P : (ℝ × ℝ)) := P = (0, -3)
def point_distance_to_y_axis (P : (ℝ × ℝ)) := P = (6, 0) ∨ P = (-6, -6)
def point_in_third_quadrant_and_equidistant (P : (ℝ × ℝ)) := P = (-6, -6)

theorem find_coords_of_P_cond1 (P : ℝ × ℝ) (h : 2 * m + 4 = 0) : point_on_y_axis P ↔ P = (0, -3) :=
by {
  sorry
}

theorem find_coords_of_P_cond2 (P : ℝ × ℝ) (h : abs (2 * m + 4) = 6) : point_distance_to_y_axis P ↔ (P = (6, 0) ∨ P = (-6, -6)) :=
by {
  sorry
}

theorem find_coords_of_P_cond3 (P : ℝ × ℝ) (h1 : 2 * m + 4 < 0) (h2 : m - 1 < 0) (h3 : abs (2 * m + 4) = abs (m - 1)) : point_in_third_quadrant_and_equidistant P ↔ P = (-6, -6) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_coords_of_P_cond1_find_coords_of_P_cond2_find_coords_of_P_cond3_l927_92778


namespace NUMINAMATH_GPT_lower_side_length_is_correct_l927_92710

noncomputable def length_of_lower_side
  (a b h : ℝ) (A : ℝ) 
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62) : ℝ :=
b

theorem lower_side_length_is_correct
  (a b h : ℝ) (A : ℝ)
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62)
  (ha : A = (1/2) * (a + b) * h) : b = 17.65 :=
by
  sorry

end NUMINAMATH_GPT_lower_side_length_is_correct_l927_92710


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l927_92737

theorem solve_equation_1 (x : ℝ) :
  x^2 - 10 * x + 16 = 0 → x = 8 ∨ x = 2 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 3) = 6 - 2 * x → x = 3 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l927_92737


namespace NUMINAMATH_GPT_farmhands_work_hours_l927_92785

def apples_per_pint (variety: String) : ℕ :=
  match variety with
  | "golden_delicious" => 20
  | "pink_lady" => 40
  | _ => 0

def total_apples_for_pints (pints: ℕ) : ℕ :=
  (apples_per_pint "golden_delicious") * pints + (apples_per_pint "pink_lady") * pints

def apples_picked_per_hour_per_farmhand : ℕ := 240

def num_farmhands : ℕ := 6

def total_apples_picked_per_hour : ℕ :=
  num_farmhands * apples_picked_per_hour_per_farmhand

def ratio_golden_to_pink : ℕ × ℕ := (1, 2)

def haley_cider_pints : ℕ := 120

def hours_worked (pints: ℕ) (picked_per_hour: ℕ): ℕ :=
  (total_apples_for_pints pints) / picked_per_hour

theorem farmhands_work_hours :
  hours_worked haley_cider_pints total_apples_picked_per_hour = 5 := by
  sorry

end NUMINAMATH_GPT_farmhands_work_hours_l927_92785


namespace NUMINAMATH_GPT_cube_edge_length_eq_six_l927_92713

theorem cube_edge_length_eq_six {s : ℝ} (h : s^3 = 6 * s^2) : s = 6 :=
sorry

end NUMINAMATH_GPT_cube_edge_length_eq_six_l927_92713


namespace NUMINAMATH_GPT_find_x_l927_92745

theorem find_x (x : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - i) * (Complex.ofReal x + i) = 1 + i) : x = 0 :=
by sorry

end NUMINAMATH_GPT_find_x_l927_92745


namespace NUMINAMATH_GPT_sequence_problem_l927_92741

variable {n : ℕ}

-- We define the arithmetic sequence conditions
noncomputable def a_n : ℕ → ℕ
| n => 2 * n + 1

-- Conditions that the sequence must satisfy
axiom a_3_eq_7 : a_n 3 = 7
axiom a_5_a_7_eq_26 : a_n 5 + a_n 7 = 26

-- Define the sum of the sequence
noncomputable def S_n (n : ℕ) := n^2 + 2 * n

-- Define the sequence b_n
noncomputable def b_n (n : ℕ) := 1 / (a_n n ^ 2 - 1 : ℝ)

-- Define the sum of the sequence b_n
noncomputable def T_n (n : ℕ) := (n / (4 * (n + 1)) : ℝ)

-- The main theorem to prove
theorem sequence_problem :
  (a_n n = 2 * n + 1) ∧ (S_n n = n^2 + 2 * n) ∧ (T_n n = n / (4 * (n + 1))) :=
  sorry

end NUMINAMATH_GPT_sequence_problem_l927_92741


namespace NUMINAMATH_GPT_ratio_area_A_to_C_l927_92763

noncomputable def side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area (side : ℕ) : ℕ :=
  side * side

theorem ratio_area_A_to_C : 
  let A_perimeter := 16
  let B_perimeter := 40
  let C_perimeter := 2 * A_perimeter
  let side_A := side_length A_perimeter
  let side_C := side_length C_perimeter
  let area_A := area side_A
  let area_C := area side_C
  (area_A : ℚ) / area_C = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_A_to_C_l927_92763


namespace NUMINAMATH_GPT_initial_money_amount_l927_92753

theorem initial_money_amount 
  (X : ℝ) 
  (h : 0.70 * X = 350) : 
  X = 500 := 
sorry

end NUMINAMATH_GPT_initial_money_amount_l927_92753


namespace NUMINAMATH_GPT_max_additional_bags_correct_l927_92769

-- Definitions from conditions
def num_people : ℕ := 6
def bags_per_person : ℕ := 5
def weight_per_bag : ℕ := 50
def max_plane_capacity : ℕ := 6000

-- Derived definitions from conditions
def total_bags : ℕ := num_people * bags_per_person
def total_weight_of_bags : ℕ := total_bags * weight_per_bag
def remaining_capacity : ℕ := max_plane_capacity - total_weight_of_bags 
def max_additional_bags : ℕ := remaining_capacity / weight_per_bag

-- Theorem statement
theorem max_additional_bags_correct : max_additional_bags = 90 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_max_additional_bags_correct_l927_92769


namespace NUMINAMATH_GPT_suff_cond_iff_lt_l927_92782

variable (a b : ℝ)

-- Proving that (a - b) a^2 < 0 is a sufficient but not necessary condition for a < b
theorem suff_cond_iff_lt (h : (a - b) * a^2 < 0) : a < b :=
by {
  sorry
}

end NUMINAMATH_GPT_suff_cond_iff_lt_l927_92782


namespace NUMINAMATH_GPT_lines_parallel_to_skew_are_skew_or_intersect_l927_92756

-- Define skew lines conditions in space
def skew_lines (l1 l2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ¬ (∀ t1 t2 : ℝ, l1 t1 = l2 t2) ∧ ¬ (∃ d : ℝ × ℝ × ℝ, ∀ t : ℝ, l1 t + d = l2 t)

-- Define parallel lines condition in space
def parallel_lines (m l : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∃ v : ℝ × ℝ × ℝ, ∀ t1 t2 : ℝ, m t1 = l t2 + v

-- Define the relationship to check between lines
def relationship (m1 m2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  (∃ t1 t2 : ℝ, m1 t1 = m2 t2) ∨ skew_lines m1 m2

-- The main theorem statement
theorem lines_parallel_to_skew_are_skew_or_intersect
  {l1 l2 m1 m2 : ℝ → ℝ × ℝ × ℝ}
  (h_skew: skew_lines l1 l2)
  (h_parallel_1: parallel_lines m1 l1)
  (h_parallel_2: parallel_lines m2 l2) :
  relationship m1 m2 :=
by
  sorry

end NUMINAMATH_GPT_lines_parallel_to_skew_are_skew_or_intersect_l927_92756


namespace NUMINAMATH_GPT_a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l927_92783

theorem a1_minus_2a2_plus_3a3_minus_4a4_eq_48:
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (∀ x : ℝ, (1 + 2 * x) ^ 4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 = 48 :=
by
  sorry

end NUMINAMATH_GPT_a1_minus_2a2_plus_3a3_minus_4a4_eq_48_l927_92783


namespace NUMINAMATH_GPT_total_rings_is_19_l927_92747

-- Definitions based on the problem conditions
def rings_on_first_day : Nat := 8
def rings_on_second_day : Nat := 6
def rings_on_third_day : Nat := 5

-- Total rings calculation
def total_rings : Nat := rings_on_first_day + rings_on_second_day + rings_on_third_day

-- Proof statement
theorem total_rings_is_19 : total_rings = 19 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_rings_is_19_l927_92747


namespace NUMINAMATH_GPT_determine_c_l927_92777

theorem determine_c (c d : ℝ) (hc : c < 0) (hd : d > 0) (hamp : ∀ x, y = c * Real.cos (d * x) → |y| ≤ 3) :
  c = -3 :=
sorry

end NUMINAMATH_GPT_determine_c_l927_92777


namespace NUMINAMATH_GPT_ratio_pentagon_side_length_to_rectangle_width_l927_92735

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_pentagon_side_length_to_rectangle_width_l927_92735


namespace NUMINAMATH_GPT_initial_milk_amount_l927_92758

theorem initial_milk_amount (d : ℚ) (r : ℚ) (T : ℚ) 
  (hd : d = 0.4) 
  (hr : r = 0.69) 
  (h_remaining : r = (1 - d) * T) : 
  T = 1.15 := 
  sorry

end NUMINAMATH_GPT_initial_milk_amount_l927_92758


namespace NUMINAMATH_GPT_female_employees_l927_92709

theorem female_employees (total_employees male_employees : ℕ) 
  (advanced_degree_male_adv: ℝ) (advanced_degree_female_adv: ℝ) (prob: ℝ) 
  (h1 : total_employees = 450) 
  (h2 : male_employees = 300)
  (h3 : advanced_degree_male_adv = 0.10) 
  (h4 : advanced_degree_female_adv = 0.40)
  (h5 : prob = 0.4) : 
  ∃ F : ℕ, 0.10 * male_employees + (advanced_degree_female_adv * F + (1 - advanced_degree_female_adv) * F) / total_employees = prob ∧ F = 150 :=
by
  sorry

end NUMINAMATH_GPT_female_employees_l927_92709


namespace NUMINAMATH_GPT_calculate_value_l927_92799

theorem calculate_value : 12 * ((1/3 : ℝ) + (1/4) - (1/12))⁻¹ = 24 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l927_92799


namespace NUMINAMATH_GPT_sum_even_and_multiples_of_5_l927_92743

def num_even_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 5 -- even digits: {0, 2, 4, 6, 8}
  thousands * hundreds * tens * units

def num_multiples_of_5_four_digit : ℕ :=
  let thousands := 9 -- thousands place cannot be zero
  let hundreds := 10
  let tens := 10
  let units := 2 -- multiples of 5 digits: {0, 5}
  thousands * hundreds * tens * units

theorem sum_even_and_multiples_of_5 : num_even_four_digit + num_multiples_of_5_four_digit = 6300 := by
  sorry

end NUMINAMATH_GPT_sum_even_and_multiples_of_5_l927_92743


namespace NUMINAMATH_GPT_sum_of_scores_l927_92742

/-- Prove that given the conditions on Bill, John, and Sue's scores, the total sum of the scores of the three students is 160. -/
theorem sum_of_scores (B J S : ℕ) (h1 : B = J + 20) (h2 : B = S / 2) (h3 : B = 45) : B + J + S = 160 :=
sorry

end NUMINAMATH_GPT_sum_of_scores_l927_92742


namespace NUMINAMATH_GPT_new_profit_percentage_l927_92731

theorem new_profit_percentage (P : ℝ) (h1 : 1.10 * P = 990) (h2 : 0.90 * P * (1 + 0.30) = 1053) : 0.30 = 0.30 :=
by sorry

end NUMINAMATH_GPT_new_profit_percentage_l927_92731


namespace NUMINAMATH_GPT_range_of_m_l927_92724

theorem range_of_m (m : ℝ) (h : 2 * m + 3 < 4) : m < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l927_92724


namespace NUMINAMATH_GPT_determine_functions_l927_92748

noncomputable def satisfies_condition (f : ℕ → ℕ) : Prop :=
∀ (n p : ℕ), Prime p → (f n)^p % f p = n % f p

theorem determine_functions :
  ∀ (f : ℕ → ℕ),
  satisfies_condition f →
  f = id ∨
  (∀ p: ℕ, Prime p → f p = 1) ∨
  (f 2 = 2 ∧ (∀ p: ℕ, Prime p → p > 2 → f p = 1) ∧ ∀ n: ℕ, f n % 2 = n % 2) :=
by
  intros f h1
  sorry

end NUMINAMATH_GPT_determine_functions_l927_92748


namespace NUMINAMATH_GPT_quadratic_roots_expression_eq_zero_l927_92787

theorem quadratic_roots_expression_eq_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * x^2 + b * x + c = 0)
  (x1 x2 : ℝ)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (s1 s2 s3 : ℝ)
  (h_s1 : s1 = x1 + x2)
  (h_s2 : s2 = x1^2 + x2^2)
  (h_s3 : s3 = x1^3 + x2^3) :
  a * s3 + b * s2 + c * s1 = 0 := sorry

end NUMINAMATH_GPT_quadratic_roots_expression_eq_zero_l927_92787


namespace NUMINAMATH_GPT_volume_of_S_l927_92700

-- Define the region S in terms of the conditions
def region_S (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1.5 ∧ 
  abs x + abs y ≤ 1 ∧ 
  abs z ≤ 0.5

-- Define the volume calculation function
noncomputable def volume_S : ℝ :=
  sorry -- This is where the computation/theorem proving for volume would go

-- The theorem stating the volume of S
theorem volume_of_S : volume_S = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_volume_of_S_l927_92700


namespace NUMINAMATH_GPT_sailboat_rental_cost_l927_92798

-- Define the conditions
def rental_per_hour_ski := 80
def hours_per_day := 3
def days := 2
def cost_ski := (hours_per_day * days * rental_per_hour_ski)
def additional_cost := 120

-- Statement to prove
theorem sailboat_rental_cost :
  ∃ (S : ℕ), cost_ski = S + additional_cost → S = 360 := by
  sorry

end NUMINAMATH_GPT_sailboat_rental_cost_l927_92798


namespace NUMINAMATH_GPT_quadratic_roots_unique_pair_l927_92703

theorem quadratic_roots_unique_pair (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h_root1 : p * q = q)
  (h_root2 : p + q = -p)
  (h_rel : q = -2 * p) : 
(p, q) = (1, -2) :=
  sorry

end NUMINAMATH_GPT_quadratic_roots_unique_pair_l927_92703


namespace NUMINAMATH_GPT_minimum_sum_of_dimensions_of_box_l927_92752

theorem minimum_sum_of_dimensions_of_box (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 2310) :
  a + b + c ≥ 52 :=
sorry

end NUMINAMATH_GPT_minimum_sum_of_dimensions_of_box_l927_92752


namespace NUMINAMATH_GPT_total_digits_first_2003_even_integers_l927_92722

theorem total_digits_first_2003_even_integers : 
  let even_integers := (List.range' 1 (2003 * 2)).filter (λ n => n % 2 = 0)
  let one_digit_count := List.filter (λ n => n < 10) even_integers |>.length
  let two_digit_count := List.filter (λ n => 10 ≤ n ∧ n < 100) even_integers |>.length
  let three_digit_count := List.filter (λ n => 100 ≤ n ∧ n < 1000) even_integers |>.length
  let four_digit_count := List.filter (λ n => 1000 ≤ n) even_integers |>.length
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4
  total_digits = 7460 :=
by
  sorry

end NUMINAMATH_GPT_total_digits_first_2003_even_integers_l927_92722


namespace NUMINAMATH_GPT_fermat_numbers_pairwise_coprime_l927_92714

theorem fermat_numbers_pairwise_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (2 ^ (2 ^ i) + 1) (2 ^ (2 ^ j) + 1) = 1 :=
sorry

end NUMINAMATH_GPT_fermat_numbers_pairwise_coprime_l927_92714


namespace NUMINAMATH_GPT_smallest_n_Sn_pos_l927_92762

theorem smallest_n_Sn_pos {a : ℕ → ℤ} (S : ℕ → ℤ) 
  (h1 : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1))
  (h2 : ∀ n, (n ≠ 5 → S n > S 5))
  (h3 : |a 5| > |a 6|) :
  ∃ n : ℕ, S n > 0 ∧ ∀ m < n, S m ≤ 0 :=
by 
  -- Actual proof steps would go here.
  sorry

end NUMINAMATH_GPT_smallest_n_Sn_pos_l927_92762


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l927_92796

theorem asymptotes_of_hyperbola 
  (x y : ℝ)
  (h : x^2 / 4 - y^2 / 36 = 1) : 
  (y = 3 * x) ∨ (y = -3 * x) :=
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l927_92796


namespace NUMINAMATH_GPT_find_natural_numbers_l927_92746

theorem find_natural_numbers :
  ∃ (x y : ℕ), 
    x * y - (x + y) = Nat.gcd x y + Nat.lcm x y ∧ 
    ((x = 6 ∧ y = 3) ∨ (x = 6 ∧ y = 4) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 6)) := 
by 
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l927_92746


namespace NUMINAMATH_GPT_Randy_blocks_used_l927_92706

theorem Randy_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks_used : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks_used = (blocks_tower + blocks_house) → total_blocks_used = 80 :=
by
  sorry

end NUMINAMATH_GPT_Randy_blocks_used_l927_92706


namespace NUMINAMATH_GPT_find_x_l927_92721

theorem find_x :
  ∀ (x : ℝ), 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 → x = 13.26 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l927_92721


namespace NUMINAMATH_GPT_rectangle_area_k_l927_92744

theorem rectangle_area_k (d : ℝ) (x : ℝ) (h_ratio : 5 * x > 0 ∧ 2 * x > 0) (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (∃ (h : k = 10 / 29), (5 * x) * (2 * x) = k * d^2) := by
  use 10 / 29
  sorry

end NUMINAMATH_GPT_rectangle_area_k_l927_92744


namespace NUMINAMATH_GPT_smallest_range_possible_l927_92708

-- Definition of the problem conditions
def seven_observations (x1 x2 x3 x4 x5 x6 x7 : ℝ) :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 9 ∧
  x4 = 10

noncomputable def smallest_range : ℝ :=
  5

-- Lean statement asserting the proof problem
theorem smallest_range_possible (x1 x2 x3 x4 x5 x6 x7 : ℝ) (h : seven_observations x1 x2 x3 x4 x5 x6 x7) :
  ∃ x1' x2' x3' x4' x5' x6' x7', seven_observations x1' x2' x3' x4' x5' x6' x7' ∧ (x7' - x1') = smallest_range :=
sorry

end NUMINAMATH_GPT_smallest_range_possible_l927_92708


namespace NUMINAMATH_GPT_factorize_1_factorize_2_l927_92701

variable {a x y : ℝ}

theorem factorize_1 : 2 * a * x^2 - 8 * a * x * y + 8 * a * y^2 = 2 * a * (x - 2 * y)^2 := 
by
  sorry

theorem factorize_2 : 6 * x * y^2 - 9 * x^2 * y - y^3 = -y * (3 * x - y)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_l927_92701


namespace NUMINAMATH_GPT_gnomes_cannot_cross_l927_92788

theorem gnomes_cannot_cross :
  ∀ (gnomes : List ℕ), 
    (∀ g, g ∈ gnomes → g ∈ (List.range 100).map (λ x => x + 1)) →
    List.sum gnomes = 5050 → 
    ∀ (boat_capacity : ℕ), boat_capacity = 100 →
    ∀ (k : ℕ), (200 * (k + 1) - k^2 = 10100) → false :=
by
  intros gnomes H_weights H_sum boat_capacity H_capacity k H_equation
  sorry

end NUMINAMATH_GPT_gnomes_cannot_cross_l927_92788


namespace NUMINAMATH_GPT_man_speed_upstream_l927_92764

def man_speed_still_water : ℕ := 50
def speed_downstream : ℕ := 80

theorem man_speed_upstream : (man_speed_still_water - (speed_downstream - man_speed_still_water)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_upstream_l927_92764


namespace NUMINAMATH_GPT_factor_expression_l927_92755

theorem factor_expression (x : ℝ) :
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) :=
  sorry

end NUMINAMATH_GPT_factor_expression_l927_92755


namespace NUMINAMATH_GPT_find_k_l927_92767

theorem find_k (x y z k : ℝ) (h1 : 5 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 9 / (z - y)) : k = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l927_92767


namespace NUMINAMATH_GPT_average_k_l927_92768

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end NUMINAMATH_GPT_average_k_l927_92768


namespace NUMINAMATH_GPT_points_lie_on_ellipse_l927_92771

open Real

noncomputable def curve_points_all_lie_on_ellipse (s: ℝ) : Prop :=
  let x := 2 * cos s + 2 * sin s
  let y := 4 * (cos s - sin s)
  (x^2 / 8 + y^2 / 32 = 1)

-- Below statement defines the theorem we aim to prove:
theorem points_lie_on_ellipse (s: ℝ) : curve_points_all_lie_on_ellipse s :=
sorry -- This "sorry" is to indicate that the proof is omitted.

end NUMINAMATH_GPT_points_lie_on_ellipse_l927_92771


namespace NUMINAMATH_GPT_smallest_n_for_y_n_integer_l927_92792

noncomputable def y (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then (5 : ℝ)^(1/3) else
  if n = 2 then ((5 : ℝ)^(1/3))^((5 : ℝ)^(1/3)) else
  y (n-1)^((5 : ℝ)^(1/3))

theorem smallest_n_for_y_n_integer : ∃ n : ℕ, y n = 5 ∧ ∀ m < n, y m ≠ ((⌊y m⌋:ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_y_n_integer_l927_92792


namespace NUMINAMATH_GPT_proposition_p_and_not_q_is_true_l927_92715

-- Define proposition p
def p : Prop := ∀ x > 0, Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : Real, a > b → a^2 > b^2

-- State the theorem to be proven in Lean
theorem proposition_p_and_not_q_is_true : p ∧ ¬q :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_proposition_p_and_not_q_is_true_l927_92715


namespace NUMINAMATH_GPT_savings_correct_l927_92789

noncomputable def school_price_math : Float := 45
noncomputable def school_price_science : Float := 60
noncomputable def school_price_literature : Float := 35

noncomputable def discount_math : Float := 0.20
noncomputable def discount_science : Float := 0.25
noncomputable def discount_literature : Float := 0.15

noncomputable def tax_school : Float := 0.07
noncomputable def tax_alt : Float := 0.06
noncomputable def shipping_alt : Float := 10

noncomputable def alt_price_math : Float := (school_price_math * (1 - discount_math)) * (1 + tax_alt)
noncomputable def alt_price_science : Float := (school_price_science * (1 - discount_science)) * (1 + tax_alt)
noncomputable def alt_price_literature : Float := (school_price_literature * (1 - discount_literature)) * (1 + tax_alt)

noncomputable def total_alt_cost : Float := alt_price_math + alt_price_science + alt_price_literature + shipping_alt

noncomputable def school_price_math_tax : Float := school_price_math * (1 + tax_school)
noncomputable def school_price_science_tax : Float := school_price_science * (1 + tax_school)
noncomputable def school_price_literature_tax : Float := school_price_literature * (1 + tax_school)

noncomputable def total_school_cost : Float := school_price_math_tax + school_price_science_tax + school_price_literature_tax

noncomputable def savings : Float := total_school_cost - total_alt_cost

theorem savings_correct : savings = 22.40 := by
  sorry

end NUMINAMATH_GPT_savings_correct_l927_92789


namespace NUMINAMATH_GPT_geometric_seq_a3_l927_92734

theorem geometric_seq_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 6 = a 3 * r^3)
  (h2 : a 9 = a 3 * r^6)
  (h3 : a 6 = 6)
  (h4 : a 9 = 9) : 
  a 3 = 4 := 
sorry

end NUMINAMATH_GPT_geometric_seq_a3_l927_92734


namespace NUMINAMATH_GPT_problem_solution_l927_92754

noncomputable def solve_equation (x : ℝ) : Prop :=
  x ≠ 4 ∧ (x + 36 / (x - 4) = -9)

theorem problem_solution : {x : ℝ | solve_equation x} = {0, -5} :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l927_92754


namespace NUMINAMATH_GPT_value_of_y_at_64_l927_92773

theorem value_of_y_at_64 (x y k : ℝ) (h1 : y = k * x^(1/3)) (h2 : 8^(1/3) = 2) (h3 : y = 4 ∧ x = 8):
  y = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_y_at_64_l927_92773


namespace NUMINAMATH_GPT_n_fraction_of_sum_l927_92738

theorem n_fraction_of_sum (n S : ℝ) (h1 : n = S / 5) (h2 : S ≠ 0) :
  n = 1 / 6 * ((S + (S / 5))) :=
by
  sorry

end NUMINAMATH_GPT_n_fraction_of_sum_l927_92738


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_div_4_l927_92716

open Real

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : (cos α * 2 + (-1) * sin α = 0)) : 
  tan (α - π / 4) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_div_4_l927_92716


namespace NUMINAMATH_GPT_range_of_t_minus_1_over_t_minus_3_l927_92795

variable {f : ℝ → ℝ}

-- Function conditions: monotonically decreasing and odd
axiom f_mono_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition on the real number t
variable {t : ℝ}
axiom f_condition : f (t^2 - 2 * t) + f (-3) > 0

-- Question: Prove the range of (t-1)/(t-3)
theorem range_of_t_minus_1_over_t_minus_3 (h : -1 < t ∧ t < 3) : 
  ((t - 1) / (t - 3)) < 1/2 :=
  sorry

end NUMINAMATH_GPT_range_of_t_minus_1_over_t_minus_3_l927_92795


namespace NUMINAMATH_GPT_units_digit_2_pow_2130_l927_92750

theorem units_digit_2_pow_2130 : (Nat.pow 2 2130) % 10 = 4 :=
by sorry

end NUMINAMATH_GPT_units_digit_2_pow_2130_l927_92750


namespace NUMINAMATH_GPT_first_group_people_count_l927_92780

def group_ice_cream (P : ℕ) : Prop :=
  let total_days_per_person1 := P * 10
  let total_days_per_person2 := 5 * 16
  total_days_per_person1 = total_days_per_person2

theorem first_group_people_count 
  (P : ℕ) 
  (H1 : group_ice_cream P) : 
  P = 8 := 
sorry

end NUMINAMATH_GPT_first_group_people_count_l927_92780


namespace NUMINAMATH_GPT_jack_runs_faster_than_paul_l927_92765

noncomputable def convert_km_hr_to_m_s (v : ℝ) : ℝ :=
  v * (1000 / 3600)

noncomputable def speed_difference : ℝ :=
  let v_J_km_hr := 20.62665  -- Jack's speed in km/hr
  let v_J_m_s := convert_km_hr_to_m_s v_J_km_hr  -- Jack's speed in m/s
  let distance := 1000  -- distance in meters
  let time_J := distance / v_J_m_s  -- Jack's time in seconds
  let time_P := time_J + 1.5  -- Paul's time in seconds
  let v_P_m_s := distance / time_P  -- Paul's speed in m/s
  let speed_diff_m_s := v_J_m_s - v_P_m_s  -- speed difference in m/s
  let speed_diff_km_hr := speed_diff_m_s * (3600 / 1000)  -- convert to km/hr
  speed_diff_km_hr

theorem jack_runs_faster_than_paul : speed_difference = 0.18225 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_jack_runs_faster_than_paul_l927_92765


namespace NUMINAMATH_GPT_river_depth_ratio_l927_92776

-- Definitions based on the conditions
def depthMidMay : ℝ := 5
def increaseMidJune : ℝ := 10
def depthMidJune : ℝ := depthMidMay + increaseMidJune
def depthMidJuly : ℝ := 45

-- The theorem based on the question and correct answer
theorem river_depth_ratio : depthMidJuly / depthMidJune = 3 := by 
  -- Proof skipped for illustration purposes
  sorry

end NUMINAMATH_GPT_river_depth_ratio_l927_92776


namespace NUMINAMATH_GPT_largest_common_value_less_than_1000_l927_92736

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a = 999 ∧ (∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 8 * m) ∧ a < 1000 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_value_less_than_1000_l927_92736


namespace NUMINAMATH_GPT_chord_length_of_intersecting_line_and_circle_l927_92702

theorem chord_length_of_intersecting_line_and_circle :
  ∀ (x y : ℝ), (3 * x + 4 * y - 5 = 0) ∧ (x^2 + y^2 = 4) →
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_chord_length_of_intersecting_line_and_circle_l927_92702


namespace NUMINAMATH_GPT_unique_solution_implies_a_eq_pm_b_l927_92718

theorem unique_solution_implies_a_eq_pm_b 
  (a b : ℝ) 
  (h_nonzero_a : a ≠ 0) 
  (h_nonzero_b : b ≠ 0) 
  (h_unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) : 
  a = b ∨ a = -b :=
sorry

end NUMINAMATH_GPT_unique_solution_implies_a_eq_pm_b_l927_92718


namespace NUMINAMATH_GPT_num_cows_l927_92766

-- Define the context
variable (C H L Heads : ℕ)

-- Define the conditions
axiom condition1 : L = 2 * Heads + 8
axiom condition2 : L = 4 * C + 2 * H
axiom condition3 : Heads = C + H

-- State the goal
theorem num_cows : C = 4 := by
  sorry

end NUMINAMATH_GPT_num_cows_l927_92766


namespace NUMINAMATH_GPT_number_one_half_more_equals_twenty_five_percent_less_l927_92775

theorem number_one_half_more_equals_twenty_five_percent_less (n : ℤ) : 
    (80 - 0.25 * 80 = 60) → ((3 / 2 : ℚ) * n = 60) → (n = 40) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_number_one_half_more_equals_twenty_five_percent_less_l927_92775


namespace NUMINAMATH_GPT_growth_factor_condition_l927_92760

open BigOperators

theorem growth_factor_condition {n : ℕ} (h : ∏ i in Finset.range n, (i + 2) / (i + 1) = 50) : n = 49 := by
  sorry

end NUMINAMATH_GPT_growth_factor_condition_l927_92760


namespace NUMINAMATH_GPT_max_integer_value_of_expression_l927_92761

theorem max_integer_value_of_expression (x : ℝ) :
  ∃ M : ℤ, M = 15 ∧ ∀ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) ≤ M :=
sorry

end NUMINAMATH_GPT_max_integer_value_of_expression_l927_92761


namespace NUMINAMATH_GPT_min_pounds_of_beans_l927_92786

theorem min_pounds_of_beans : 
  ∃ (b : ℕ), (∀ (r : ℝ), (r ≥ 8 + b / 3 ∧ r ≤ 3 * b) → b ≥ 3) :=
sorry

end NUMINAMATH_GPT_min_pounds_of_beans_l927_92786
