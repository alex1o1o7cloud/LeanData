import Mathlib

namespace NUMINAMATH_GPT_Rebecca_eggs_l1050_105026

/-- Rebecca has 6 marbles -/
def M : ℕ := 6

/-- Rebecca has 14 more eggs than marbles -/
def E : ℕ := M + 14

/-- Rebecca has 20 eggs -/
theorem Rebecca_eggs : E = 20 := by
  sorry

end NUMINAMATH_GPT_Rebecca_eggs_l1050_105026


namespace NUMINAMATH_GPT_age_ratio_in_2_years_is_2_1_l1050_105047

-- Define the ages and conditions
def son_age (current_year : ℕ) : ℕ := 20
def man_age (current_year : ℕ) : ℕ := son_age current_year + 22

def son_age_in_2_years (current_year : ℕ) : ℕ := son_age current_year + 2
def man_age_in_2_years (current_year : ℕ) : ℕ := man_age current_year + 2

-- The theorem stating the ratio of the man's age to the son's age in two years is 2:1
theorem age_ratio_in_2_years_is_2_1 (current_year : ℕ) :
  man_age_in_2_years current_year = 2 * son_age_in_2_years current_year :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_in_2_years_is_2_1_l1050_105047


namespace NUMINAMATH_GPT_solve_inequality_l1050_105093

variable {a x : ℝ}

theorem solve_inequality (h : a > 0) : 
  (ax^2 - (a + 1)*x + 1 < 0) ↔ 
    (if 0 < a ∧ a < 1 then 1 < x ∧ x < 1/a else 
     if a = 1 then false else 
     if a > 1 then 1/a < x ∧ x < 1 else true) :=
  sorry

end NUMINAMATH_GPT_solve_inequality_l1050_105093


namespace NUMINAMATH_GPT_number_of_white_balls_l1050_105096

-- Definition of conditions
def red_balls : ℕ := 4
def frequency_of_red_balls : ℝ := 0.25
def total_balls (white_balls : ℕ) : ℕ := red_balls + white_balls

-- Proving the number of white balls given the conditions
theorem number_of_white_balls (x : ℕ) :
  (red_balls : ℝ) / total_balls x = frequency_of_red_balls → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_balls_l1050_105096


namespace NUMINAMATH_GPT_vehicle_distance_traveled_l1050_105082

theorem vehicle_distance_traveled 
  (perimeter_back : ℕ) (perimeter_front : ℕ) (revolution_difference : ℕ)
  (R : ℕ)
  (h1 : perimeter_back = 9)
  (h2 : perimeter_front = 7)
  (h3 : revolution_difference = 10)
  (h4 : (R * perimeter_back) = ((R + revolution_difference) * perimeter_front)) :
  (R * perimeter_back) = 315 :=
by
  -- Prove that the distance traveled by the vehicle is 315 feet
  -- given the conditions and the hypothesis.
  sorry

end NUMINAMATH_GPT_vehicle_distance_traveled_l1050_105082


namespace NUMINAMATH_GPT_trivia_team_students_l1050_105023

def total_students (not_picked groups students_per_group: ℕ) :=
  not_picked + groups * students_per_group

theorem trivia_team_students (not_picked groups students_per_group: ℕ) (h_not_picked: not_picked = 10) (h_groups: groups = 8) (h_students_per_group: students_per_group = 6) :
  total_students not_picked groups students_per_group = 58 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_students_l1050_105023


namespace NUMINAMATH_GPT_total_length_of_sticks_l1050_105050

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_length_of_sticks_l1050_105050


namespace NUMINAMATH_GPT_smallest_base_10_integer_exists_l1050_105011

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_10_integer_exists_l1050_105011


namespace NUMINAMATH_GPT_largest_value_p_l1050_105088

theorem largest_value_p 
  (p q r : ℝ) 
  (h1 : p + q + r = 10) 
  (h2 : p * q + p * r + q * r = 25) :
  p ≤ 20 / 3 :=
sorry

end NUMINAMATH_GPT_largest_value_p_l1050_105088


namespace NUMINAMATH_GPT_lemuel_total_points_l1050_105035

theorem lemuel_total_points (two_point_shots : ℕ) (three_point_shots : ℕ) (points_from_two : ℕ) (points_from_three : ℕ) :
  two_point_shots = 7 →
  three_point_shots = 3 →
  points_from_two = 2 →
  points_from_three = 3 →
  two_point_shots * points_from_two + three_point_shots * points_from_three = 23 :=
by
  sorry

end NUMINAMATH_GPT_lemuel_total_points_l1050_105035


namespace NUMINAMATH_GPT_range_of_m_l1050_105005

theorem range_of_m (a b m : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_eq : a * b = a + b + 3) (h_ineq : a * b ≥ m) : m ≤ 9 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1050_105005


namespace NUMINAMATH_GPT_P_eq_Q_at_x_l1050_105075

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 2
def Q (x : ℝ) : ℝ := 0

theorem P_eq_Q_at_x :
  ∃ x : ℝ, P x = Q x ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_P_eq_Q_at_x_l1050_105075


namespace NUMINAMATH_GPT_convert_neg_300_deg_to_rad_l1050_105054

theorem convert_neg_300_deg_to_rad :
  -300 * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_convert_neg_300_deg_to_rad_l1050_105054


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l1050_105049

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
    (h1 : q > 0)
    (h2 : 2 * a 3 = a 5 - 3 * a 4) 
    (h3 : a 2 * a 4 * a 6 = 64) 
    (h4 : ∀ n, S_n n = (1 - q^n) / (1 - q) * a 1) :
    q = 2 ∧ (∀ n, S_n n = (2^n - 1) / 2) := 
  by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l1050_105049


namespace NUMINAMATH_GPT_aira_rubber_bands_l1050_105016

variable (S A J : ℕ)

-- Conditions
def conditions (S A J : ℕ) : Prop :=
  S = A + 5 ∧ A = J - 1 ∧ S + A + J = 18

-- Proof problem
theorem aira_rubber_bands (S A J : ℕ) (h : conditions S A J) : A = 4 :=
by
  -- introduce the conditions
  obtain ⟨h₁, h₂, h₃⟩ := h
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_aira_rubber_bands_l1050_105016


namespace NUMINAMATH_GPT_jason_pokemon_cards_l1050_105072

-- Conditions
def initial_cards : ℕ := 13
def cards_given : ℕ := 9

-- Proof Statement
theorem jason_pokemon_cards (initial_cards cards_given : ℕ) : initial_cards - cards_given = 4 :=
by
  sorry

end NUMINAMATH_GPT_jason_pokemon_cards_l1050_105072


namespace NUMINAMATH_GPT_nine_points_unit_square_l1050_105087

theorem nine_points_unit_square :
  ∀ (points : List (ℝ × ℝ)), points.length = 9 → 
  (∀ (x : ℝ × ℝ), x ∈ points → 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1) → 
  ∃ (A B C : ℝ × ℝ), A ∈ points ∧ B ∈ points ∧ C ∈ points ∧ 
  (1 / 8 : ℝ) ≤ abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_nine_points_unit_square_l1050_105087


namespace NUMINAMATH_GPT_omicron_variant_diameter_in_scientific_notation_l1050_105009

/-- Converting a number to scientific notation. -/
def to_scientific_notation (d : ℝ) (a : ℝ) (n : ℤ) : Prop :=
  d = a * 10 ^ n

theorem omicron_variant_diameter_in_scientific_notation :
  to_scientific_notation 0.00000011 1.1 (-7) :=
by
  sorry

end NUMINAMATH_GPT_omicron_variant_diameter_in_scientific_notation_l1050_105009


namespace NUMINAMATH_GPT_difference_q_r_l1050_105012

theorem difference_q_r (x : ℝ) (p q r : ℝ) 
  (h1 : 7 * x - 3 * x = 3600) 
  (h2 : q = 7 * x) 
  (h3 : r = 12 * x) :
  r - q = 4500 := 
sorry

end NUMINAMATH_GPT_difference_q_r_l1050_105012


namespace NUMINAMATH_GPT_women_lawyers_percentage_l1050_105004

-- Define the conditions of the problem
variable {T : ℝ} (h1 : 0.80 * T = 0.80 * T)                          -- Placeholder for group size, not necessarily used directly
variable (h2 : 0.32 = 0.80 * L)                                       -- Given condition of the problem: probability of selecting a woman lawyer

-- Define the theorem to be proven
theorem women_lawyers_percentage (h2 : 0.32 = 0.80 * L) : L = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_women_lawyers_percentage_l1050_105004


namespace NUMINAMATH_GPT_football_field_width_l1050_105066

theorem football_field_width (length : ℕ) (total_distance : ℕ) (laps : ℕ) (width : ℕ) 
  (h1 : length = 100) (h2 : total_distance = 1800) (h3 : laps = 6) :
  width = 50 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_football_field_width_l1050_105066


namespace NUMINAMATH_GPT_parallelepiped_volume_l1050_105069

noncomputable def volume_of_parallelepiped (a : ℝ) : ℝ :=
  (a^3 * Real.sqrt 2) / 2

theorem parallelepiped_volume (a : ℝ) (h_pos : 0 < a) :
  volume_of_parallelepiped a = (a^3 * Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_volume_l1050_105069


namespace NUMINAMATH_GPT_exp_inequality_solution_l1050_105046

theorem exp_inequality_solution (x : ℝ) (h : 1 < Real.exp x ∧ Real.exp x < 2) : 0 < x ∧ x < Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_exp_inequality_solution_l1050_105046


namespace NUMINAMATH_GPT_gcd_765432_654321_l1050_105064

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := 
  sorry

end NUMINAMATH_GPT_gcd_765432_654321_l1050_105064


namespace NUMINAMATH_GPT_approx_d_l1050_105019

noncomputable def close_approx_d : ℝ :=
  let d := (69.28 * (0.004)^3 - Real.log 27) / (0.03 * Real.cos (55 * Real.pi / 180))
  d

theorem approx_d : |close_approx_d + 191.297| < 0.001 :=
  by
    -- Proof goes here.
    sorry

end NUMINAMATH_GPT_approx_d_l1050_105019


namespace NUMINAMATH_GPT_race_placement_l1050_105028

def finished_places (nina zoey sam liam vince : ℕ) : Prop :=
  nina = 12 ∧
  sam = nina + 1 ∧
  zoey = nina - 2 ∧
  liam = zoey - 3 ∧
  vince = liam + 2 ∧
  vince = nina - 3

theorem race_placement (nina zoey sam liam vince : ℕ) :
  finished_places nina zoey sam liam vince →
  nina = 12 →
  sam = 13 →
  zoey = 10 →
  liam = 7 →
  vince = 5 →
  (8 ≠ sam ∧ 8 ≠ nina ∧ 8 ≠ zoey ∧ 8 ≠ liam ∧ 8 ≠ jodi ∧ 8 ≠ vince) := by
  sorry

end NUMINAMATH_GPT_race_placement_l1050_105028


namespace NUMINAMATH_GPT_greatest_possible_value_of_x_l1050_105025

theorem greatest_possible_value_of_x
    (x : ℕ)
    (h1 : x > 0)
    (h2 : x % 4 = 0)
    (h3 : x^3 < 8000) :
    x ≤ 16 :=
    sorry

end NUMINAMATH_GPT_greatest_possible_value_of_x_l1050_105025


namespace NUMINAMATH_GPT_rate_of_interest_l1050_105068

theorem rate_of_interest (P T SI: ℝ) (h1 : P = 2500) (h2 : T = 5) (h3 : SI = P - 2000) (h4 : SI = (P * R * T) / 100):
  R = 4 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1050_105068


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1050_105024

/-- Given an arithmetic sequence {a_n} and the first term a_1 = -2010, 
and given that the average of the first 2009 terms minus the average of the first 2007 terms equals 2,
prove that the sum of the first 2011 terms S_2011 equals 0. -/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith_seq : ∃ d, ∀ n, a n = a 1 + (n - 1) * d)
  (h_Sn : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d)
  (h_a1 : a 1 = -2010)
  (h_avg_diff : (S 2009) / 2009 - (S 2007) / 2007 = 2) :
  S 2011 = 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1050_105024


namespace NUMINAMATH_GPT_union_complements_l1050_105080

open Set

variable (U : Set ℕ) (A B : Set ℕ)

-- Define the conditions
def condition_U : U = {1, 2, 3, 4, 5} := by
  sorry

def condition_A : A = {1, 2, 3} := by
  sorry

def condition_B : B = {2, 3, 4} := by
  sorry

-- Prove that (complement_U A) ∪ (complement_U B) = {1, 4, 5}
theorem union_complements :
  (U \ A) ∪ (U \ B) = {1, 4, 5} := by
  sorry

end NUMINAMATH_GPT_union_complements_l1050_105080


namespace NUMINAMATH_GPT_range_of_y_l1050_105032

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 120) : y ∈ Set.Ioo (-11 : ℝ) (-10 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_y_l1050_105032


namespace NUMINAMATH_GPT_thor_hammer_weight_exceeds_2000_l1050_105020

/--  The Mighty Thor uses a hammer that doubles in weight each day as he trains.
      Starting on the first day with a hammer that weighs 7 pounds, prove that
      on the 10th day the hammer's weight exceeds 2000 pounds. 
-/
theorem thor_hammer_weight_exceeds_2000 :
  ∃ n : ℕ, 7 * 2^(n - 1) > 2000 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_thor_hammer_weight_exceeds_2000_l1050_105020


namespace NUMINAMATH_GPT_correct_answer_l1050_105015

theorem correct_answer (x : ℤ) (h : (x - 11) / 5 = 31) : (x - 5) / 11 = 15 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_l1050_105015


namespace NUMINAMATH_GPT_tan_alpha_eq_one_l1050_105098

open Real

theorem tan_alpha_eq_one (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (h_cos_sin_eq : cos (α + β) = sin (α - β)) : tan α = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_l1050_105098


namespace NUMINAMATH_GPT_bicycle_trip_length_l1050_105074

def total_distance (days1 day1 miles1 day2 miles2: ℕ) : ℕ :=
  days1 * miles1 + day2 * miles2

theorem bicycle_trip_length :
  total_distance 12 12 1 6 = 150 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_trip_length_l1050_105074


namespace NUMINAMATH_GPT_tessa_needs_more_apples_l1050_105045

/-- Tessa starts with 4 apples.
    Anita gives her 5 more apples.
    She needs 10 apples to make a pie.
    Prove that she needs 1 more apple to make the pie.
-/
theorem tessa_needs_more_apples:
  ∀ initial_apples extra_apples total_needed extra_needed: ℕ,
    initial_apples = 4 → extra_apples = 5 → total_needed = 10 →
    extra_needed = total_needed - (initial_apples + extra_apples) →
    extra_needed = 1 :=
by
  intros initial_apples extra_apples total_needed extra_needed hi he ht heq
  rw [hi, he, ht] at heq
  simp at heq
  assumption

end NUMINAMATH_GPT_tessa_needs_more_apples_l1050_105045


namespace NUMINAMATH_GPT_cos_240_eq_negative_half_l1050_105013

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_240_eq_negative_half_l1050_105013


namespace NUMINAMATH_GPT_final_score_l1050_105052

-- Definitions based on the conditions
def bullseye_points : ℕ := 50
def miss_points : ℕ := 0
def half_bullseye_points : ℕ := bullseye_points / 2

-- Statement to prove
theorem final_score : bullseye_points + miss_points + half_bullseye_points = 75 :=
by
  sorry

end NUMINAMATH_GPT_final_score_l1050_105052


namespace NUMINAMATH_GPT_yoghurt_cost_1_l1050_105014

theorem yoghurt_cost_1 :
  ∃ y : ℝ,
  (∀ (ice_cream_cartons yoghurt_cartons : ℕ) (ice_cream_cost_one_carton : ℝ) (yoghurt_cost_one_carton : ℝ),
    ice_cream_cartons = 19 →
    yoghurt_cartons = 4 →
    ice_cream_cost_one_carton = 7 →
    (19 * 7 = 133) →  -- total ice cream cost
    (133 - 129 = 4) → -- Total yogurt cost
    (4 = 4 * y) →    -- Yoghurt cost equation
    y = 1) :=
sorry

end NUMINAMATH_GPT_yoghurt_cost_1_l1050_105014


namespace NUMINAMATH_GPT_platform_length_is_correct_l1050_105091

def speed_kmph : ℝ := 72
def seconds_to_cross_platform : ℝ := 26
def train_length_m : ℝ := 270.0416

noncomputable def length_of_platform : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * seconds_to_cross_platform
  total_distance - train_length_m

theorem platform_length_is_correct : 
  length_of_platform = 249.9584 := 
by
  sorry

end NUMINAMATH_GPT_platform_length_is_correct_l1050_105091


namespace NUMINAMATH_GPT_candy_pieces_per_pile_l1050_105021

theorem candy_pieces_per_pile :
  ∀ (total_candies eaten_candies num_piles pieces_per_pile : ℕ),
    total_candies = 108 →
    eaten_candies = 36 →
    num_piles = 8 →
    pieces_per_pile = (total_candies - eaten_candies) / num_piles →
    pieces_per_pile = 9 :=
by
  intros total_candies eaten_candies num_piles pieces_per_pile
  sorry

end NUMINAMATH_GPT_candy_pieces_per_pile_l1050_105021


namespace NUMINAMATH_GPT_cube_root_of_neg_125_l1050_105089

theorem cube_root_of_neg_125 : (-5)^3 = -125 := 
by sorry

end NUMINAMATH_GPT_cube_root_of_neg_125_l1050_105089


namespace NUMINAMATH_GPT_more_volunteers_needed_l1050_105033

theorem more_volunteers_needed
    (required_volunteers : ℕ)
    (students_per_class : ℕ)
    (num_classes : ℕ)
    (teacher_volunteers : ℕ)
    (total_volunteers : ℕ) :
    required_volunteers = 50 →
    students_per_class = 5 →
    num_classes = 6 →
    teacher_volunteers = 13 →
    total_volunteers = (students_per_class * num_classes) + teacher_volunteers →
    (required_volunteers - total_volunteers) = 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_more_volunteers_needed_l1050_105033


namespace NUMINAMATH_GPT_cost_per_ream_is_27_l1050_105070

-- Let ream_sheets be the number of sheets in one ream.
def ream_sheets : ℕ := 500

-- Let total_sheets be the total number of sheets needed.
def total_sheets : ℕ := 5000

-- Let total_cost be the total cost to buy the total number of sheets.
def total_cost : ℕ := 270

-- We need to prove that the cost per ream (in dollars) is 27.
theorem cost_per_ream_is_27 : (total_cost / (total_sheets / ream_sheets)) = 27 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_ream_is_27_l1050_105070


namespace NUMINAMATH_GPT_min_value_l1050_105065

theorem min_value (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by sorry

end NUMINAMATH_GPT_min_value_l1050_105065


namespace NUMINAMATH_GPT_range_of_c_l1050_105053

theorem range_of_c (c : ℝ) :
  (c^2 - 5 * c + 7 > 1 ∧ (|2 * c - 1| ≤ 1)) ∨ ((c^2 - 5 * c + 7 ≤ 1) ∧ |2 * c - 1| > 1) ↔ (0 ≤ c ∧ c ≤ 1) ∨ (2 ≤ c ∧ c ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_c_l1050_105053


namespace NUMINAMATH_GPT_fill_in_square_l1050_105043

theorem fill_in_square (x y : ℝ) (h : 4 * x^2 * (81 / 4 * x * y) = 81 * x^3 * y) : (81 / 4 * x * y) = (81 / 4 * x * y) :=
by
  sorry

end NUMINAMATH_GPT_fill_in_square_l1050_105043


namespace NUMINAMATH_GPT_geometric_sequence_quadratic_roots_l1050_105034

theorem geometric_sequence_quadratic_roots
    (a b : ℝ)
    (h_geometric : ∃ q : ℝ, b = 2 * q ∧ a = 2 * q^2) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + (1 / 3) = 0 ∧ a * x2^2 + b * x2 + (1 / 3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_quadratic_roots_l1050_105034


namespace NUMINAMATH_GPT_chord_square_length_l1050_105073

/-- Given three circles with radii 4, 8, and 16, such that the first two are externally tangent to each other and both are internally tangent to the third, if a chord in the circle with radius 16 is a common external tangent to the other two circles, then the square of the length of this chord is 7616/9. -/
theorem chord_square_length (r1 r2 r3 : ℝ) (h1 : r1 = 4) (h2 : r2 = 8) (h3 : r3 = 16)
  (tangent_condition : ∀ (O4 O8 O16 : ℝ), O4 = r1 + r2 ∧ O8 = r2 + r3 ∧ O16 = r1 + r3) :
  (16^2 - (20/3)^2) * 4 = 7616 / 9 :=
by
  sorry

end NUMINAMATH_GPT_chord_square_length_l1050_105073


namespace NUMINAMATH_GPT_measure_of_angle_f_l1050_105077

theorem measure_of_angle_f (angle_D angle_E angle_F : ℝ)
  (h1 : angle_D = 75)
  (h2 : angle_E = 4 * angle_F + 30)
  (h3 : angle_D + angle_E + angle_F = 180) : 
  angle_F = 15 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_f_l1050_105077


namespace NUMINAMATH_GPT_quotient_base6_division_l1050_105079

theorem quotient_base6_division :
  let a := 2045
  let b := 14
  let base := 6
  a / b = 51 :=
by
  sorry

end NUMINAMATH_GPT_quotient_base6_division_l1050_105079


namespace NUMINAMATH_GPT_parabola_focus_l1050_105062

theorem parabola_focus (a b c : ℝ) (h_eq : ∀ x : ℝ, 2 * x^2 + 8 * x - 1 = a * (x + b)^2 + c) :
  ∃ focus : ℝ × ℝ, focus = (-2, -71 / 8) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l1050_105062


namespace NUMINAMATH_GPT_apples_in_basket_l1050_105071

-- Define the conditions in Lean
def four_times_as_many_apples (O A : ℕ) : Prop :=
  A = 4 * O

def emiliano_consumes (O A : ℕ) : Prop :=
  (2/3 : ℚ) * O + (2/3 : ℚ) * A = 50

-- Formulate the main proposition to prove there are 60 apples
theorem apples_in_basket (O A : ℕ) (h1 : four_times_as_many_apples O A) (h2 : emiliano_consumes O A) : A = 60 := 
by
  sorry

end NUMINAMATH_GPT_apples_in_basket_l1050_105071


namespace NUMINAMATH_GPT_prove_correct_option_C_l1050_105086

theorem prove_correct_option_C (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 :=
by sorry

end NUMINAMATH_GPT_prove_correct_option_C_l1050_105086


namespace NUMINAMATH_GPT_part_a_solution_exists_l1050_105027

theorem part_a_solution_exists : ∃ (x y : ℕ), x^2 - y^2 = 31 ∧ x = 16 ∧ y = 15 := 
by 
  sorry

end NUMINAMATH_GPT_part_a_solution_exists_l1050_105027


namespace NUMINAMATH_GPT_calculate_new_volume_l1050_105008

noncomputable def volume_of_sphere_with_increased_radius
  (initial_surface_area : ℝ) (radius_increase : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((Real.sqrt (initial_surface_area / (4 * Real.pi)) + radius_increase) ^ 3)

theorem calculate_new_volume :
  volume_of_sphere_with_increased_radius 400 (2) = 2304 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_calculate_new_volume_l1050_105008


namespace NUMINAMATH_GPT_find_x_plus_y_l1050_105099

-- Define the initial assumptions and conditions
variables {x y : ℝ}
axiom geom_sequence : 1 > 0 ∧ x > 0 ∧ y > 0 ∧ 3 > 0 ∧ 1 * x = y
axiom arith_sequence : 2 * y = x + 3

-- Prove that x + y = 15 / 4
theorem find_x_plus_y : x + y = 15 / 4 := sorry

end NUMINAMATH_GPT_find_x_plus_y_l1050_105099


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1050_105018

-- Define sets M and N
def M : Set ℕ := {0, 2, 3, 4}
def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

-- State the problem as a theorem
theorem intersection_of_M_and_N : (M ∩ N) = {0, 4} :=
by
    sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1050_105018


namespace NUMINAMATH_GPT_AM_GM_Inequality_equality_condition_l1050_105076

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

end NUMINAMATH_GPT_AM_GM_Inequality_equality_condition_l1050_105076


namespace NUMINAMATH_GPT_rotation_of_unit_circle_l1050_105010

open Real

noncomputable def rotated_coordinates (θ : ℝ) : ℝ × ℝ :=
  ( -sin θ, cos θ )

theorem rotation_of_unit_circle (θ : ℝ) (k : ℤ) (h : θ ≠ k * π + π / 2) :
  let A := (cos θ, sin θ)
  let O := (0, 0)
  let B := rotated_coordinates (θ)
  B = (-sin θ, cos θ) :=
sorry

end NUMINAMATH_GPT_rotation_of_unit_circle_l1050_105010


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1050_105051

theorem value_of_x_plus_y 
  (x y : ℝ) 
  (h1 : -x = 3) 
  (h2 : |y| = 5) : 
  x + y = 2 ∨ x + y = -8 := 
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1050_105051


namespace NUMINAMATH_GPT_positive_integers_mod_l1050_105006

theorem positive_integers_mod (n : ℕ) (h : n > 0) :
  ∃! (x : ℕ), x < 10^n ∧ x^2 % 10^n = x % 10^n :=
sorry

end NUMINAMATH_GPT_positive_integers_mod_l1050_105006


namespace NUMINAMATH_GPT_emily_spending_l1050_105084

theorem emily_spending : ∀ {x : ℝ}, (x + 2 * x + 3 * x = 120) → (x = 20) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_emily_spending_l1050_105084


namespace NUMINAMATH_GPT_binom_1294_2_l1050_105063

def combination (n k : Nat) := n.choose k

theorem binom_1294_2 : combination 1294 2 = 836161 := by
  sorry

end NUMINAMATH_GPT_binom_1294_2_l1050_105063


namespace NUMINAMATH_GPT_C_work_completion_l1050_105092

theorem C_work_completion (A_completion_days B_completion_days AB_completion_days : ℕ)
  (A_cond : A_completion_days = 8)
  (B_cond : B_completion_days = 12)
  (AB_cond : AB_completion_days = 4) :
  ∃ (C_completion_days : ℕ), C_completion_days = 24 := 
by
  sorry

end NUMINAMATH_GPT_C_work_completion_l1050_105092


namespace NUMINAMATH_GPT_JacksonsGrade_l1050_105017

theorem JacksonsGrade : 
  let hours_playing_video_games := 12
  let hours_studying := (1 / 3) * hours_playing_video_games
  let hours_kindness := (1 / 4) * hours_playing_video_games
  let grade_initial := 0
  let grade_per_hour_studying := 20
  let grade_per_hour_kindness := 40
  let grade_from_studying := grade_per_hour_studying * hours_studying
  let grade_from_kindness := grade_per_hour_kindness * hours_kindness
  let total_grade := grade_initial + grade_from_studying + grade_from_kindness
  total_grade = 200 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_JacksonsGrade_l1050_105017


namespace NUMINAMATH_GPT_factor_expression_l1050_105029

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) :=
sorry

end NUMINAMATH_GPT_factor_expression_l1050_105029


namespace NUMINAMATH_GPT_plumber_total_cost_l1050_105057

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end NUMINAMATH_GPT_plumber_total_cost_l1050_105057


namespace NUMINAMATH_GPT_man_work_days_l1050_105042

variable (W : ℝ) -- Denoting the amount of work by W

-- Defining the work rate variables
variables (M Wm B : ℝ)

-- Conditions from the problem:
-- Combined work rate of man, woman, and boy together completes the work in 3 days
axiom combined_work_rate : M + Wm + B = W / 3
-- Woman completes the work alone in 18 days
axiom woman_work_rate : Wm = W / 18
-- Boy completes the work alone in 9 days
axiom boy_work_rate : B = W / 9

-- The goal is to prove the man takes 6 days to complete the work alone
theorem man_work_days : (W / M) = 6 :=
by
  sorry

end NUMINAMATH_GPT_man_work_days_l1050_105042


namespace NUMINAMATH_GPT_garden_perimeter_ratio_l1050_105000

theorem garden_perimeter_ratio (side_length : ℕ) (tripled_side_length : ℕ) (original_perimeter : ℕ) (new_perimeter : ℕ) (ratio : ℚ) :
  side_length = 50 →
  tripled_side_length = 3 * side_length →
  original_perimeter = 4 * side_length →
  new_perimeter = 4 * tripled_side_length →
  ratio = original_perimeter / new_perimeter →
  ratio = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_garden_perimeter_ratio_l1050_105000


namespace NUMINAMATH_GPT_integral_2x_plus_3_squared_l1050_105058

open Real

-- Define the function to be integrated
def f (x : ℝ) := (2 * x + 3) ^ 2

-- State the theorem for the indefinite integral
theorem integral_2x_plus_3_squared :
  ∃ C : ℝ, ∫ x, f x = (1 / 6) * (2 * x + 3) ^ 3 + C :=
by
  sorry

end NUMINAMATH_GPT_integral_2x_plus_3_squared_l1050_105058


namespace NUMINAMATH_GPT_exponential_function_decreasing_l1050_105094

theorem exponential_function_decreasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (0 < a ∧ a < 1) → ¬ (∀ x : ℝ, x > 0 → a ^ x > 0) :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_decreasing_l1050_105094


namespace NUMINAMATH_GPT_cos_double_angle_l1050_105083

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1050_105083


namespace NUMINAMATH_GPT_polynomial_value_given_cond_l1050_105067

variable (x : ℝ)
theorem polynomial_value_given_cond :
  (x^2 - (5/2) * x = 6) →
  2 * x^2 - 5 * x + 6 = 18 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_given_cond_l1050_105067


namespace NUMINAMATH_GPT_simplify_expression_l1050_105041

theorem simplify_expression : 4 * (8 - 2 + 3) - 7 = 29 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l1050_105041


namespace NUMINAMATH_GPT_scott_sold_40_cups_of_smoothies_l1050_105030

theorem scott_sold_40_cups_of_smoothies
  (cost_smoothie : ℕ)
  (cost_cake : ℕ)
  (num_cakes : ℕ)
  (total_revenue : ℕ)
  (h1 : cost_smoothie = 3)
  (h2 : cost_cake = 2)
  (h3 : num_cakes = 18)
  (h4 : total_revenue = 156) :
  ∃ x : ℕ, (cost_smoothie * x + cost_cake * num_cakes = total_revenue ∧ x = 40) := 
sorry

end NUMINAMATH_GPT_scott_sold_40_cups_of_smoothies_l1050_105030


namespace NUMINAMATH_GPT_linear_function_is_C_l1050_105081

theorem linear_function_is_C :
  ∀ (f : ℤ → ℤ), (f = (λ x => 2 * x^2 - 1) ∨ f = (λ x => -1/x) ∨ f = (λ x => (x+1)/3) ∨ f = (λ x => 3 * x + 2 * x^2 - 1)) →
  (f = (λ x => (x+1)/3)) ↔ 
  (∃ (m b : ℤ), ∀ x : ℤ, f x = m * x + b) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_is_C_l1050_105081


namespace NUMINAMATH_GPT_area_of_rectangle_l1050_105060

def length : ℕ := 4
def width : ℕ := 2

theorem area_of_rectangle : length * width = 8 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1050_105060


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1050_105048

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 16) (h2 : (1 / x) = 3 * (1 / y)) : 
  x + y = (16 * Real.sqrt 3) / 3 := 
sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1050_105048


namespace NUMINAMATH_GPT_one_in_set_A_l1050_105078

theorem one_in_set_A : 1 ∈ {x | x ≥ -1} :=
sorry

end NUMINAMATH_GPT_one_in_set_A_l1050_105078


namespace NUMINAMATH_GPT_mosquitoes_required_l1050_105002

theorem mosquitoes_required
  (blood_loss_to_cause_death : Nat)
  (drops_per_mosquito_A : Nat)
  (drops_per_mosquito_B : Nat)
  (drops_per_mosquito_C : Nat)
  (n : Nat) :
  blood_loss_to_cause_death = 15000 →
  drops_per_mosquito_A = 20 →
  drops_per_mosquito_B = 25 →
  drops_per_mosquito_C = 30 →
  75 * n = blood_loss_to_cause_death →
  n = 200 := by
  sorry

end NUMINAMATH_GPT_mosquitoes_required_l1050_105002


namespace NUMINAMATH_GPT_inequality_solution_range_l1050_105007

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_inequality_solution_range_l1050_105007


namespace NUMINAMATH_GPT_quadratic_graph_y1_lt_y2_l1050_105095

theorem quadratic_graph_y1_lt_y2 (x1 x2 : ℝ) (h1 : -x1^2 = y1) (h2 : -x2^2 = y2) (h3 : x1 * x2 > x2^2) : y1 < y2 :=
  sorry

end NUMINAMATH_GPT_quadratic_graph_y1_lt_y2_l1050_105095


namespace NUMINAMATH_GPT_find_coords_C_l1050_105085

-- Define the coordinates of given points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- The proof problem wrapped in a lean theorem
theorem find_coords_C (C : ℝ × ℝ) 
  (h1 : AB = AC) (h2 : (D.1, D.2) = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) :
  C = (-1, 5) :=
sorry

end NUMINAMATH_GPT_find_coords_C_l1050_105085


namespace NUMINAMATH_GPT_find_second_sum_l1050_105044

theorem find_second_sum (x : ℝ) (total_sum : ℝ) (h : total_sum = 2691) 
  (h1 : (24 * x) / 100 = 15 * (total_sum - x) / 100) : total_sum - x = 1656 :=
by
  sorry

end NUMINAMATH_GPT_find_second_sum_l1050_105044


namespace NUMINAMATH_GPT_graph_intersect_points_l1050_105090

-- Define f as a function defined on all real numbers and invertible
variable (f : ℝ → ℝ) (hf : Function.Injective f)

-- Define the theorem to find the number of intersection points
theorem graph_intersect_points : 
  ∃ (n : ℕ), n = 3 ∧ ∃ (x : ℝ), (f (x^2) = f (x^6)) :=
  by
    -- Outline sketch: We aim to show there are 3 real solutions satisfying the equation
    -- The proof here is skipped, hence we put sorry
    sorry

end NUMINAMATH_GPT_graph_intersect_points_l1050_105090


namespace NUMINAMATH_GPT_blue_eyed_among_blondes_l1050_105056

variable (l g b a : ℝ)

-- Given: The proportion of blondes among blue-eyed people is greater than the proportion of blondes among all people.
axiom given_condition : a / g > b / l

-- Prove: The proportion of blue-eyed people among blondes is greater than the proportion of blue-eyed people among all people.
theorem blue_eyed_among_blondes (l g b a : ℝ) (h : a / g > b / l) : a / b > g / l :=
by
  sorry

end NUMINAMATH_GPT_blue_eyed_among_blondes_l1050_105056


namespace NUMINAMATH_GPT_robotics_club_students_l1050_105097

theorem robotics_club_students (total cs e both neither : ℕ) 
  (h1 : total = 80)
  (h2 : cs = 52)
  (h3 : e = 38)
  (h4 : both = 25)
  (h5 : neither = total - (cs - both + e - both + both)) :
  neither = 15 :=
by
  sorry

end NUMINAMATH_GPT_robotics_club_students_l1050_105097


namespace NUMINAMATH_GPT_math_problem_l1050_105036

theorem math_problem 
  (x y z : ℚ)
  (h1 : 4 * x - 5 * y - z = 0)
  (h2 : x + 5 * y - 18 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 3622 / 9256 := 
sorry

end NUMINAMATH_GPT_math_problem_l1050_105036


namespace NUMINAMATH_GPT_triangle_inequality_condition_l1050_105003

theorem triangle_inequality_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (ineq : 6 * (a * b + b * c + c * a) > 5 * (a ^ 2 + b ^ 2 + c ^ 2)) : 
  (a < b + c ∧ b < a + c ∧ c < a + b) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_condition_l1050_105003


namespace NUMINAMATH_GPT_find_j_value_l1050_105039

variable {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def polynomial_has_four_distinct_real_roots_in_arithmetic_progression
(p : Polynomial R) : Prop :=
∃ a d : R, p.roots.toFinset = {a, a + d, a + 2*d, a + 3*d} ∧
a ≠ a + d ∧ a ≠ a + 2*d ∧ a ≠ a + 3*d ∧ a + d ≠ a + 2*d ∧
a + d ≠ a + 3*d ∧ a + 2*d ≠ a + 3*d

-- The main theorem statement
theorem find_j_value (k : R) 
  (h : polynomial_has_four_distinct_real_roots_in_arithmetic_progression 
  (Polynomial.X^4 + Polynomial.C j * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C 900)) :
  j = -900 :=
sorry

end NUMINAMATH_GPT_find_j_value_l1050_105039


namespace NUMINAMATH_GPT_part_a_part_b_l1050_105037

-- Part (a): Proving that 91 divides n^37 - n for all integers n
theorem part_a (n : ℤ) : 91 ∣ (n ^ 37 - n) := 
sorry

-- Part (b): Finding the largest k that divides n^37 - n for all integers n is 3276
theorem part_b (n : ℤ) : ∀ k : ℤ, (k > 0) → (∀ n : ℤ, k ∣ (n ^ 37 - n)) → k ≤ 3276 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1050_105037


namespace NUMINAMATH_GPT_apples_total_l1050_105001

theorem apples_total (apples_per_person : ℝ) (number_of_people : ℝ) (h_apples : apples_per_person = 15.0) (h_people : number_of_people = 3.0) : 
  apples_per_person * number_of_people = 45.0 := by
  sorry

end NUMINAMATH_GPT_apples_total_l1050_105001


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_formula_l1050_105038

theorem arithmetic_geometric_sequence_formula :
  ∃ (a d : ℝ), (3 * a = 6) ∧
  ((5 - d) * (15 + d) = 64) ∧
  (∀ (n : ℕ), n ≥ 3 → (∃ (b_n : ℝ), b_n = 2 ^ (n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_formula_l1050_105038


namespace NUMINAMATH_GPT_option_c_correct_l1050_105022

theorem option_c_correct (x y : ℝ) : 3 * x^2 * y + 2 * y * x^2 = 5 * x^2 * y :=
by {
  sorry
}

end NUMINAMATH_GPT_option_c_correct_l1050_105022


namespace NUMINAMATH_GPT_polygonal_pyramid_faces_l1050_105040

/-- A polygonal pyramid is a three-dimensional solid. Its base is a regular polygon. Each of the vertices of the polygonal base is connected to a single point, called the apex. The sum of the number of edges and the number of vertices of a particular polygonal pyramid is 1915. This theorem states that the number of faces of this pyramid is 639. -/
theorem polygonal_pyramid_faces (n : ℕ) (hn : 2 * n + (n + 1) = 1915) : n + 1 = 639 :=
by
  sorry

end NUMINAMATH_GPT_polygonal_pyramid_faces_l1050_105040


namespace NUMINAMATH_GPT_Sammy_has_8_bottle_caps_l1050_105031

def Billie_caps : Nat := 2
def Janine_caps (B : Nat) : Nat := 3 * B
def Sammy_caps (J : Nat) : Nat := J + 2

theorem Sammy_has_8_bottle_caps : 
  Sammy_caps (Janine_caps Billie_caps) = 8 := 
by
  sorry

end NUMINAMATH_GPT_Sammy_has_8_bottle_caps_l1050_105031


namespace NUMINAMATH_GPT_count_whole_numbers_in_interval_l1050_105055

open Real

theorem count_whole_numbers_in_interval : 
  ∃ n : ℕ, n = 5 ∧ ∀ x : ℕ, (sqrt 7 < x ∧ x < exp 2) ↔ (3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end NUMINAMATH_GPT_count_whole_numbers_in_interval_l1050_105055


namespace NUMINAMATH_GPT_ordered_pair_correct_l1050_105061

def find_ordered_pair (s m : ℚ) : Prop :=
  (∀ t : ℚ, (∃ x y : ℚ, x = -3 + t * m ∧ y = s + t * (-7) ∧ y = (3/4) * x + 5))
  ∧ s = 11/4 ∧ m = -28/3

theorem ordered_pair_correct :
  find_ordered_pair (11/4) (-28/3) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_correct_l1050_105061


namespace NUMINAMATH_GPT_max_abs_sum_l1050_105059

theorem max_abs_sum (a b c : ℝ) (h : ∀ x, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  |a| + |b| + |c| ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_abs_sum_l1050_105059
