import Mathlib

namespace NUMINAMATH_GPT_find_missing_number_l479_47975

theorem find_missing_number (x : ℝ)
  (h1 : (x + 42 + 78 + 104) / 4 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) :
  x = 74 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l479_47975


namespace NUMINAMATH_GPT_find_an_find_n_l479_47992

noncomputable def a_n (n : ℕ) : ℤ := 12 + (n - 1) * 2

noncomputable def S_n (n : ℕ) : ℤ := n * 12 + (n * (n - 1) / 2) * 2

theorem find_an (n : ℕ) : a_n n = 2 * n + 10 :=
by sorry

theorem find_n (n : ℕ) (S_n : ℤ) : S_n = 242 → n = 11 :=
by sorry

end NUMINAMATH_GPT_find_an_find_n_l479_47992


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_9_terms_l479_47986

-- Define the odd function and its properties
variables {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = -f (x)) 
          (h2 : ∀ x y, x < y → f x < f y)

-- Define the shifted function g
noncomputable def g (x : ℝ) := f (x - 5)

-- Define the arithmetic sequence with non-zero common difference
variables {a : ℕ → ℝ} (d : ℝ) (h3 : d ≠ 0) 
          (h4 : ∀ n, a (n + 1) = a n + d)

-- Condition given by the problem
variable (h5 : g (a 1) + g (a 9) = 0)

-- Proof obligation
theorem sum_of_arithmetic_sequence_9_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 45 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_9_terms_l479_47986


namespace NUMINAMATH_GPT_stewart_farm_sheep_count_l479_47972

theorem stewart_farm_sheep_count 
  (S H : ℕ) 
  (ratio : S * 7 = 4 * H)
  (food_per_horse : H * 230 = 12880) : 
  S = 32 := 
sorry

end NUMINAMATH_GPT_stewart_farm_sheep_count_l479_47972


namespace NUMINAMATH_GPT_pyramid_height_l479_47999

theorem pyramid_height (lateral_edge : ℝ) (h : ℝ) (equilateral_angles : ℝ × ℝ × ℝ) (lateral_edge_length : lateral_edge = 3)
  (lateral_faces_are_equilateral : equilateral_angles = (60, 60, 60)) :
  h = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_pyramid_height_l479_47999


namespace NUMINAMATH_GPT_problem_statement_l479_47998

theorem problem_statement (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
(a + b = 2) ∧ ¬( (a^2 + a > 2) ∧ (b^2 + b > 2) ) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l479_47998


namespace NUMINAMATH_GPT_arithmetic_progression_l479_47969

-- Define the general formula for the nth term of an arithmetic progression
def nth_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the conditions given in the problem
def condition1 (a1 d : ℤ) : Prop := nth_term a1 d 13 = 3 * nth_term a1 d 3
def condition2 (a1 d : ℤ) : Prop := nth_term a1 d 18 = 2 * nth_term a1 d 7 + 8

-- The main proof problem statement
theorem arithmetic_progression (a1 d : ℤ) (h1 : condition1 a1 d) (h2 : condition2 a1 d) : a1 = 12 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_l479_47969


namespace NUMINAMATH_GPT_share_of_a_l479_47971

variables (A B C : ℝ)

def conditions :=
  A = (2 / 3) * (B + C) ∧
  B = (2 / 3) * (A + C) ∧
  A + B + C = 700

theorem share_of_a (h : conditions A B C) : A = 280 :=
by { sorry }

end NUMINAMATH_GPT_share_of_a_l479_47971


namespace NUMINAMATH_GPT_parallel_and_through_point_l479_47955

-- Defining the given line
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Defining the target line passing through the point (0, 4)
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the point (0, 4)
def point : ℝ × ℝ := (0, 4)

-- Prove that line2 passes through the point (0, 4) and is parallel to line1
theorem parallel_and_through_point (x y : ℝ) 
  (h1 : line1 x y) 
  : line2 (point.fst) (point.snd) := by
  sorry

end NUMINAMATH_GPT_parallel_and_through_point_l479_47955


namespace NUMINAMATH_GPT_cube_inequality_l479_47923

theorem cube_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^3 + b^3) / 2 ≥ ((a + b) / 2)^3 :=
by 
  sorry

end NUMINAMATH_GPT_cube_inequality_l479_47923


namespace NUMINAMATH_GPT_similar_triangle_leg_l479_47959

theorem similar_triangle_leg (x : Real) : 
  (12 / x = 9 / 7) → x = 84 / 9 := by
  intro h
  sorry

end NUMINAMATH_GPT_similar_triangle_leg_l479_47959


namespace NUMINAMATH_GPT_max_soap_boxes_l479_47901

theorem max_soap_boxes 
  (base_width base_length top_width top_length height soap_width soap_length soap_height max_weight soap_weight : ℝ)
  (h_base_dims : base_width = 25)
  (h_base_len : base_length = 42)
  (h_top_width : top_width = 20)
  (h_top_length : top_length = 35)
  (h_height : height = 60)
  (h_soap_width : soap_width = 7)
  (h_soap_length : soap_length = 6)
  (h_soap_height : soap_height = 10)
  (h_max_weight : max_weight = 150)
  (h_soap_weight : soap_weight = 3) :
  (50 = 
    min 
      (⌊top_width / soap_width⌋ * ⌊top_length / soap_length⌋ * ⌊height / soap_height⌋)
      (⌊max_weight / soap_weight⌋)) := by sorry

end NUMINAMATH_GPT_max_soap_boxes_l479_47901


namespace NUMINAMATH_GPT_effective_weight_lowered_l479_47979

theorem effective_weight_lowered 
    (num_weight_plates : ℕ) 
    (weight_per_plate : ℝ) 
    (increase_percentage : ℝ) 
    (total_weight_without_technology : ℝ) 
    (additional_weight : ℝ) 
    (effective_weight_lowering : ℝ) 
    (h1 : num_weight_plates = 10)
    (h2 : weight_per_plate = 30)
    (h3 : increase_percentage = 0.20)
    (h4 : total_weight_without_technology = num_weight_plates * weight_per_plate)
    (h5 : additional_weight = increase_percentage * total_weight_without_technology)
    (h6 : effective_weight_lowering = total_weight_without_technology + additional_weight) :
    effective_weight_lowering = 360 := 
by
  sorry

end NUMINAMATH_GPT_effective_weight_lowered_l479_47979


namespace NUMINAMATH_GPT_sum_of_three_numbers_l479_47932

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 :=
by sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l479_47932


namespace NUMINAMATH_GPT_max_value_fractions_l479_47903

noncomputable def maxFractions (a b c : ℝ) : ℝ :=
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c)

theorem max_value_fractions (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
    (h_sum : a + b + c = 2) :
    maxFractions a b c ≤ 1 ∧ 
    (a = 2 / 3 ∧ b = 2 / 3 ∧ c = 2 / 3 → maxFractions a b c = 1) := 
  by
    sorry

end NUMINAMATH_GPT_max_value_fractions_l479_47903


namespace NUMINAMATH_GPT_minute_hand_angle_l479_47905

theorem minute_hand_angle (minutes_slow : ℕ) (total_minutes : ℕ) (full_rotation : ℝ) (h1 : minutes_slow = 5) (h2 : total_minutes = 60) (h3 : full_rotation = 2 * Real.pi) : 
  (minutes_slow / total_minutes : ℝ) * full_rotation = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_angle_l479_47905


namespace NUMINAMATH_GPT_car_fewer_minutes_than_bus_l479_47907

-- Conditions translated into Lean definitions
def bus_time_to_beach : ℕ := 40
def car_round_trip_time : ℕ := 70

-- Derived condition
def car_one_way_time : ℕ := car_round_trip_time / 2

-- Theorem statement to be proven
theorem car_fewer_minutes_than_bus : car_one_way_time = bus_time_to_beach - 5 := by
  -- This is the placeholder for the proof
  sorry

end NUMINAMATH_GPT_car_fewer_minutes_than_bus_l479_47907


namespace NUMINAMATH_GPT_difference_q_r_share_l479_47914

theorem difference_q_r_share (x : ℝ) (h1 : 7 * x - 3 * x = 2800) :
  12 * x - 7 * x = 3500 :=
by
  sorry

end NUMINAMATH_GPT_difference_q_r_share_l479_47914


namespace NUMINAMATH_GPT_average_cd_e_l479_47949

theorem average_cd_e (c d e : ℝ) (h : (4 + 6 + 9 + c + d + e) / 6 = 20) : 
    (c + d + e) / 3 = 101 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_cd_e_l479_47949


namespace NUMINAMATH_GPT_part1_values_correct_estimated_students_correct_l479_47942

def students_data : List ℕ :=
  [30, 60, 70, 10, 30, 115, 70, 60, 75, 90, 15, 70, 40, 75, 105, 80, 60, 30, 70, 45]

def total_students := 200

def categorized_counts := (2, 5, 10, 3) -- (0 ≤ t < 30, 30 ≤ t < 60, 60 ≤ t < 90, 90 ≤ t < 120)

def mean := 60

def median := 65

def mode := 70

theorem part1_values_correct :
  let a := 5
  let b := 3
  let c := 65
  let d := 70
  categorized_counts = (2, a, 10, b) ∧ mean = 60 ∧ median = c ∧ mode = d := by {
  -- Proof will be provided here
  sorry
}

theorem estimated_students_correct :
  let at_least_avg := 130
  at_least_avg = (total_students * 13 / 20) := by {
  -- Proof will be provided here
  sorry
}

end NUMINAMATH_GPT_part1_values_correct_estimated_students_correct_l479_47942


namespace NUMINAMATH_GPT_podcast_ratio_l479_47917

theorem podcast_ratio
  (total_drive_time : ℕ)
  (first_podcast : ℕ)
  (third_podcast : ℕ)
  (fourth_podcast : ℕ)
  (next_podcast : ℕ)
  (second_podcast : ℕ) :
  total_drive_time = 360 →
  first_podcast = 45 →
  third_podcast = 105 →
  fourth_podcast = 60 →
  next_podcast = 60 →
  second_podcast = total_drive_time - (first_podcast + third_podcast + fourth_podcast + next_podcast) →
  second_podcast / first_podcast = 2 :=
by
  sorry

end NUMINAMATH_GPT_podcast_ratio_l479_47917


namespace NUMINAMATH_GPT_students_didnt_like_food_l479_47952

theorem students_didnt_like_food (total_students : ℕ) (liked_food : ℕ) (didnt_like_food : ℕ) 
  (h1 : total_students = 814) (h2 : liked_food = 383) 
  : didnt_like_food = total_students - liked_food := 
by 
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_students_didnt_like_food_l479_47952


namespace NUMINAMATH_GPT_solve_inequality_system_l479_47987

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l479_47987


namespace NUMINAMATH_GPT_aluminum_foil_thickness_l479_47911

-- Define the variables and constants
variables (d l m w t : ℝ)

-- Define the conditions
def density_condition : Prop := d = m / (l * w * t)
def volume_formula : Prop := t = m / (d * l * w)

-- The theorem to prove
theorem aluminum_foil_thickness (h1 : density_condition d l m w t) : volume_formula d l m w t :=
sorry

end NUMINAMATH_GPT_aluminum_foil_thickness_l479_47911


namespace NUMINAMATH_GPT_solve_system_l479_47921

-- Define the conditions
def system_of_equations (x y : ℝ) : Prop :=
  (x + y = 8) ∧ (2 * x - y = 7)

-- Define the proof problem statement
theorem solve_system : 
  system_of_equations 5 3 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_solve_system_l479_47921


namespace NUMINAMATH_GPT_slices_left_for_phill_correct_l479_47983

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end NUMINAMATH_GPT_slices_left_for_phill_correct_l479_47983


namespace NUMINAMATH_GPT_juan_ran_80_miles_l479_47936

def speed : Real := 10 -- miles per hour
def time : Real := 8   -- hours

theorem juan_ran_80_miles :
  speed * time = 80 := 
by
  sorry

end NUMINAMATH_GPT_juan_ran_80_miles_l479_47936


namespace NUMINAMATH_GPT_interest_rate_l479_47967

theorem interest_rate (part1_amount part2_amount total_amount total_income : ℝ) (interest_rate1 interest_rate2 : ℝ) :
  part1_amount = 2000 →
  part2_amount = total_amount - part1_amount →
  interest_rate2 = 6 →
  total_income = (part1_amount * interest_rate1 / 100) + (part2_amount * interest_rate2 / 100) →
  total_amount = 2500 →
  total_income = 130 →
  interest_rate1 = 5 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_interest_rate_l479_47967


namespace NUMINAMATH_GPT_find_constants_l479_47918

theorem find_constants (a b : ℝ) (h₀ : ∀ x : ℝ, (x^3 + 3*a*x^2 + b*x + a^2 = 0 → x = -1)) :
    a = 2 ∧ b = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l479_47918


namespace NUMINAMATH_GPT_no_real_b_for_line_to_vertex_of_parabola_l479_47985

theorem no_real_b_for_line_to_vertex_of_parabola : 
  ¬ ∃ b : ℝ, ∃ x : ℝ, y = x + b ∧ y = x^2 + b^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_b_for_line_to_vertex_of_parabola_l479_47985


namespace NUMINAMATH_GPT_stratified_sampling_pines_l479_47954

def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

theorem stratified_sampling_pines :
  sample_size * pine_saplings / total_saplings = 20 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_pines_l479_47954


namespace NUMINAMATH_GPT_largest_possible_package_l479_47993

/-- Alice, Bob, and Carol bought certain numbers of markers and the goal is to find the greatest number of markers per package. -/
def alice_markers : Nat := 60
def bob_markers : Nat := 36
def carol_markers : Nat := 48

theorem largest_possible_package :
  Nat.gcd (Nat.gcd alice_markers bob_markers) carol_markers = 12 :=
sorry

end NUMINAMATH_GPT_largest_possible_package_l479_47993


namespace NUMINAMATH_GPT_part_a_part_b_l479_47982

-- Define the system of equations
def system_of_equations (x y z p : ℝ) :=
  x^2 - 3 * y + p = z ∧ y^2 - 3 * z + p = x ∧ z^2 - 3 * x + p = y

-- Part (a) proof problem statement
theorem part_a (p : ℝ) (hp : p ≥ 4) :
  (p > 4 → ¬ ∃ (x y z : ℝ), system_of_equations x y z p) ∧
  (p = 4 → ∀ (x y z : ℝ), system_of_equations x y z 4 → x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

-- Part (b) proof problem statement
theorem part_b (p : ℝ) (hp : 1 < p ∧ p < 4) :
  ∀ (x y z : ℝ), system_of_equations x y z p → x = y ∧ y = z :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l479_47982


namespace NUMINAMATH_GPT_calculate_probability_l479_47913

-- Definitions
def total_coins : ℕ := 16  -- Total coins (3 pennies + 5 nickels + 8 dimes)
def draw_coins : ℕ := 8    -- Coins drawn
def successful_outcomes : ℕ := 321  -- Number of successful outcomes
def total_outcomes : ℕ := Nat.choose total_coins draw_coins  -- Total number of ways to choose draw_coins from total_coins

-- Question statement in Lean 4: Probability of drawing coins worth at least 75 cents
theorem calculate_probability : (successful_outcomes : ℝ) / (total_outcomes : ℝ) = 321 / 12870 := by
  sorry

end NUMINAMATH_GPT_calculate_probability_l479_47913


namespace NUMINAMATH_GPT_find_value_of_x8_plus_x4_plus_1_l479_47908

theorem find_value_of_x8_plus_x4_plus_1 (x : ℂ) (hx : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 :=
sorry

end NUMINAMATH_GPT_find_value_of_x8_plus_x4_plus_1_l479_47908


namespace NUMINAMATH_GPT_greatest_b_not_in_range_l479_47933

theorem greatest_b_not_in_range (b : ℤ) : ∀ x : ℝ, ¬ (x^2 + (b : ℝ) * x + 20 = -9) ↔ b ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_b_not_in_range_l479_47933


namespace NUMINAMATH_GPT_variance_of_scores_l479_47943

open Real

def scores : List ℝ := [30, 26, 32, 27, 35]
noncomputable def average (s : List ℝ) : ℝ := s.sum / s.length
noncomputable def variance (s : List ℝ) : ℝ :=
  (s.map (λ x => (x - average s) ^ 2)).sum / s.length

theorem variance_of_scores :
  variance scores = 54 / 5 := 
by
  sorry

end NUMINAMATH_GPT_variance_of_scores_l479_47943


namespace NUMINAMATH_GPT_present_population_l479_47984

-- Definitions
def initial_population : ℕ := 1200
def first_year_increase_rate : ℝ := 0.25
def second_year_increase_rate : ℝ := 0.30

-- Problem Statement
theorem present_population (initial_population : ℕ) 
    (first_year_increase_rate second_year_increase_rate : ℝ) : 
    initial_population = 1200 → 
    first_year_increase_rate = 0.25 → 
    second_year_increase_rate = 0.30 →
    ∃ current_population : ℕ, current_population = 1950 :=
by
  intros h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_present_population_l479_47984


namespace NUMINAMATH_GPT_UPOMB_position_l479_47997

-- Define the set of letters B, M, O, P, and U
def letters : List Char := ['B', 'M', 'O', 'P', 'U']

-- Define the word UPOMB
def word := "UPOMB"

-- Define a function that calculates the factorial of a number
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to calculate the position of a word in the alphabetical permutations of a list of characters
def word_position (w : String) (chars : List Char) : Nat :=
  let rec aux (w : List Char) (remaining : List Char) : Nat :=
    match w with
    | [] => 1
    | c :: cs =>
      let before_count := remaining.filter (· < c) |>.length
      let rest_count := factorial (remaining.length - 1)
      before_count * rest_count + aux cs (remaining.erase c)
  aux w.data chars

-- The desired theorem statement
theorem UPOMB_position : word_position word letters = 119 := by
  sorry

end NUMINAMATH_GPT_UPOMB_position_l479_47997


namespace NUMINAMATH_GPT_zero_not_in_range_of_g_l479_47976

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_zero_not_in_range_of_g_l479_47976


namespace NUMINAMATH_GPT_find_h_l479_47977

theorem find_h (h : ℝ) :
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ -(x - h)^2 = -1) → (h = 2 ∨ h = 8) :=
by sorry

end NUMINAMATH_GPT_find_h_l479_47977


namespace NUMINAMATH_GPT_min_x_plus_y_of_positive_l479_47974

open Real

theorem min_x_plus_y_of_positive (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_x_plus_y_of_positive_l479_47974


namespace NUMINAMATH_GPT_mass_of_cork_l479_47945

theorem mass_of_cork (ρ_p ρ_w ρ_s : ℝ) (m_p x : ℝ) :
  ρ_p = 2.15 * 10^4 → 
  ρ_w = 2.4 * 10^2 →
  ρ_s = 4.8 * 10^2 →
  m_p = 86.94 →
  x = 2.4 * 10^2 * (m_p / ρ_p) →
  x = 85 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mass_of_cork_l479_47945


namespace NUMINAMATH_GPT_ribbons_purple_l479_47931

theorem ribbons_purple (total_ribbons : ℕ) (yellow_ribbons purple_ribbons orange_ribbons black_ribbons : ℕ)
  (h1 : yellow_ribbons = total_ribbons / 4)
  (h2 : purple_ribbons = total_ribbons / 3)
  (h3 : orange_ribbons = total_ribbons / 6)
  (h4 : black_ribbons = 40)
  (h5 : yellow_ribbons + purple_ribbons + orange_ribbons + black_ribbons = total_ribbons) :
  purple_ribbons = 53 :=
by
  sorry

end NUMINAMATH_GPT_ribbons_purple_l479_47931


namespace NUMINAMATH_GPT_sin_pi_over_six_l479_47968

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sin_pi_over_six_l479_47968


namespace NUMINAMATH_GPT_moles_of_C6H6_l479_47956

def balanced_reaction (a b c d : ℕ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ a + b + c + d = 4

theorem moles_of_C6H6 (a b c d : ℕ) (h_balanced : balanced_reaction a b c d) :
  a = 1 := 
by 
  sorry

end NUMINAMATH_GPT_moles_of_C6H6_l479_47956


namespace NUMINAMATH_GPT_multiple_of_six_as_four_cubes_integer_as_five_cubes_l479_47947

-- Part (a)
theorem multiple_of_six_as_four_cubes (n : ℤ) : ∃ a b c d : ℤ, 6 * n = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 :=
by
  sorry

-- Part (b)
theorem integer_as_five_cubes (k : ℤ) : ∃ a b c d e : ℤ, k = a ^ 3 + b ^ 3 + c ^ 3 + d ^ 3 + e ^ 3 :=
by
  have h := multiple_of_six_as_four_cubes
  sorry

end NUMINAMATH_GPT_multiple_of_six_as_four_cubes_integer_as_five_cubes_l479_47947


namespace NUMINAMATH_GPT_intersection_A_B_l479_47927

-- Defining set A condition
def A : Set ℝ := {x | x - 1 < 2}

-- Defining set B condition
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- The goal to prove
theorem intersection_A_B : {x | x > 0 ∧ x < 3} = (A ∩ { x | 0 < x ∧ x < 8 }) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l479_47927


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l479_47991

/-
In an arithmetic sequence, if the sum of terms \( a_2 + a_3 + a_4 + a_5 + a_6 = 90 \), 
prove that \( a_1 + a_7 = 36 \).
-/

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_sum : a 2 + a 3 + a 4 + a 5 + a 6 = 90) :
  a 1 + a 7 = 36 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l479_47991


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l479_47937

-- Given {a_n} is an arithmetic sequence, and a_2 + a_3 + a_{10} + a_{11} = 40, prove a_6 + a_7 = 20
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 40) :
  a 6 + a 7 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l479_47937


namespace NUMINAMATH_GPT_cape_may_vs_daytona_shark_sightings_diff_l479_47904

-- Definitions based on the conditions
def total_shark_sightings := 40
def cape_may_sightings : ℕ := 24
def daytona_beach_sightings : ℕ := total_shark_sightings - cape_may_sightings

-- The main theorem stating the problem in Lean
theorem cape_may_vs_daytona_shark_sightings_diff :
  (2 * daytona_beach_sightings - cape_may_sightings) = 8 := by
  sorry

end NUMINAMATH_GPT_cape_may_vs_daytona_shark_sightings_diff_l479_47904


namespace NUMINAMATH_GPT_shop_discount_percentage_l479_47900

-- Definitions based on conditions
def original_price := 800
def price_paid := 560
def discount_amount := original_price - price_paid
def percentage_discount := (discount_amount / original_price) * 100

-- Proposition to prove
theorem shop_discount_percentage : percentage_discount = 30 := by
  sorry

end NUMINAMATH_GPT_shop_discount_percentage_l479_47900


namespace NUMINAMATH_GPT_man_was_absent_for_days_l479_47934

theorem man_was_absent_for_days
  (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 10 * x - 2 * y = 216) :
  y = 7 :=
by
  sorry

end NUMINAMATH_GPT_man_was_absent_for_days_l479_47934


namespace NUMINAMATH_GPT_original_cost_of_luxury_bag_l479_47912

theorem original_cost_of_luxury_bag (SP : ℝ) (profit_margin : ℝ) (original_cost : ℝ) 
  (h1 : SP = 3450) (h2 : profit_margin = 0.15) (h3 : SP = original_cost * (1 + profit_margin)) : 
  original_cost = 3000 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_luxury_bag_l479_47912


namespace NUMINAMATH_GPT_shop_sold_price_l479_47910

noncomputable def clock_selling_price (C : ℝ) : ℝ :=
  let buy_back_price := 0.60 * C
  let maintenance_cost := 0.10 * buy_back_price
  let total_spent := buy_back_price + maintenance_cost
  let selling_price := 1.80 * total_spent
  selling_price

theorem shop_sold_price (C : ℝ) (h1 : C - 0.60 * C = 100) :
  clock_selling_price C = 297 := by
  sorry

end NUMINAMATH_GPT_shop_sold_price_l479_47910


namespace NUMINAMATH_GPT_abcd_product_l479_47988

noncomputable def A := (Real.sqrt 3000 + Real.sqrt 3001)
noncomputable def B := (-Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def C := (Real.sqrt 3000 - Real.sqrt 3001)
noncomputable def D := (Real.sqrt 3001 - Real.sqrt 3000)

theorem abcd_product :
  A * B * C * D = -1 :=
by
  sorry

end NUMINAMATH_GPT_abcd_product_l479_47988


namespace NUMINAMATH_GPT_product_equals_eight_l479_47973

theorem product_equals_eight : (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7) = 8 := 
sorry

end NUMINAMATH_GPT_product_equals_eight_l479_47973


namespace NUMINAMATH_GPT_expression_simplifies_to_36_l479_47939

theorem expression_simplifies_to_36 (x : ℝ) : (x + 1)^2 + 2 * (x + 1) * (5 - x) + (5 - x)^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_36_l479_47939


namespace NUMINAMATH_GPT_proof_problem_l479_47922

variable (γ θ α : ℝ)
variable (x y : ℝ)

def condition1 := x = γ * Real.sin ((θ - α) / 2)
def condition2 := y = γ * Real.sin ((θ + α) / 2)

theorem proof_problem
  (h1 : condition1 γ θ α x)
  (h2 : condition2 γ θ α y)
  : x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * (Real.sin α)^2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l479_47922


namespace NUMINAMATH_GPT_expansion_abs_coeff_sum_l479_47929

theorem expansion_abs_coeff_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - x)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 32 :=
by
  sorry

end NUMINAMATH_GPT_expansion_abs_coeff_sum_l479_47929


namespace NUMINAMATH_GPT_line_parallel_not_coincident_l479_47948

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_not_coincident_l479_47948


namespace NUMINAMATH_GPT_find_min_value_l479_47940

theorem find_min_value (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y = 2 → c = 1 / 2 ∧ (8 / ((x + 2) * (y + 4))) ≥ c) :=
  sorry

end NUMINAMATH_GPT_find_min_value_l479_47940


namespace NUMINAMATH_GPT_train_speed_l479_47964

/-- Define the lengths of the train and the bridge and the time taken to cross the bridge. --/
def len_train : ℕ := 360
def len_bridge : ℕ := 240
def time_minutes : ℕ := 4
def time_seconds : ℕ := 240 -- 4 minutes converted to seconds

/-- Define the speed calculation based on the given domain. --/
def total_distance : ℕ := len_train + len_bridge
def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

/-- The statement to prove that the speed of the train is 2.5 m/s. --/
theorem train_speed :
  speed total_distance time_seconds = 2.5 := sorry

end NUMINAMATH_GPT_train_speed_l479_47964


namespace NUMINAMATH_GPT_time_increases_with_water_speed_increase_l479_47906

variable (S : ℝ) -- Total distance
variable (V : ℝ) -- Speed of the ferry in still water
variable (V1 V2 : ℝ) -- Speed of the water flow before and after increase

-- Ensure realistic conditions
axiom V_pos : 0 < V
axiom V1_pos : 0 < V1
axiom V2_pos : 0 < V2
axiom V1_less_V : V1 < V
axiom V2_less_V : V2 < V
axiom V1_less_V2 : V1 < V2

theorem time_increases_with_water_speed_increase :
  (S / (V + V1) + S / (V - V1)) < (S / (V + V2) + S / (V - V2)) :=
sorry

end NUMINAMATH_GPT_time_increases_with_water_speed_increase_l479_47906


namespace NUMINAMATH_GPT_max_n_l479_47920

noncomputable def a (n : ℕ) : ℕ := n

noncomputable def b (n : ℕ) : ℕ := 2 ^ a n

theorem max_n (n : ℕ) (h1 : a 2 = 2) (h2 : ∀ n, b n = 2 ^ a n)
  (h3 : b 4 = 4 * b 2) : n ≤ 9 :=
by 
  sorry

end NUMINAMATH_GPT_max_n_l479_47920


namespace NUMINAMATH_GPT_Todd_time_correct_l479_47960

theorem Todd_time_correct :
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  Todd_time = 88 :=
by
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  sorry

end NUMINAMATH_GPT_Todd_time_correct_l479_47960


namespace NUMINAMATH_GPT_find_largest_number_among_three_l479_47996

noncomputable def A (B : ℝ) := 2 * B - 43
noncomputable def C (A : ℝ) := 0.5 * A + 5

-- The main statement to be proven
theorem find_largest_number_among_three : 
  ∃ (A B C : ℝ), 
  A + B + C = 50 ∧ 
  A = 2 * B - 43 ∧ 
  C = 0.5 * A + 5 ∧ 
  max A (max B C) = 27.375 :=
by
  sorry

end NUMINAMATH_GPT_find_largest_number_among_three_l479_47996


namespace NUMINAMATH_GPT_solution_set_l479_47961

theorem solution_set :
  {p : ℝ × ℝ | (p.1^2 + 3 * p.1 * p.2 + 2 * p.2^2) * (p.1^2 * p.2^2 - 1) = 0} =
  {p : ℝ × ℝ | p.2 = -p.1 / 2} ∪
  {p : ℝ × ℝ | p.2 = -p.1} ∪
  {p : ℝ × ℝ | p.2 = -1 / p.1} ∪
  {p : ℝ × ℝ | p.2 = 1 / p.1} :=
by sorry

end NUMINAMATH_GPT_solution_set_l479_47961


namespace NUMINAMATH_GPT_Cody_spent_25_tickets_on_beanie_l479_47965

-- Introducing the necessary definitions and assumptions
variable (x : ℕ)

-- Define the conditions translated from the problem statement
def initial_tickets := 49
def tickets_left (x : ℕ) := initial_tickets - x + 6

-- State the main problem as Theorem
theorem Cody_spent_25_tickets_on_beanie (H : tickets_left x = 30) : x = 25 := by
  sorry

end NUMINAMATH_GPT_Cody_spent_25_tickets_on_beanie_l479_47965


namespace NUMINAMATH_GPT_pascal_fifth_number_in_row_15_l479_47926

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end NUMINAMATH_GPT_pascal_fifth_number_in_row_15_l479_47926


namespace NUMINAMATH_GPT_gcd_228_1995_base3_to_base6_conversion_l479_47902

-- Proof Problem 1: GCD of 228 and 1995 is 57
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

-- Proof Problem 2: Converting base-3 number 11102 to base-6
theorem base3_to_base6_conversion : Nat.ofDigits 6 [3, 1, 5] = Nat.ofDigits 10 [1, 1, 1, 0, 2] :=
by
  sorry

end NUMINAMATH_GPT_gcd_228_1995_base3_to_base6_conversion_l479_47902


namespace NUMINAMATH_GPT_sale_price_is_correct_l479_47909

def initial_price : ℝ := 560
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.30
def discount3 : ℝ := 0.15
def tax_rate : ℝ := 0.12

noncomputable def final_price : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let price_after_third_discount := price_after_second_discount * (1 - discount3)
  let price_after_tax := price_after_third_discount * (1 + tax_rate)
  price_after_tax

theorem sale_price_is_correct :
  final_price = 298.55 :=
sorry

end NUMINAMATH_GPT_sale_price_is_correct_l479_47909


namespace NUMINAMATH_GPT_smallest_and_largest_values_l479_47944

theorem smallest_and_largest_values (x : ℕ) (h : x < 100) :
  (x ≡ 2 [MOD 3]) ∧ (x ≡ 2 [MOD 4]) ∧ (x ≡ 2 [MOD 5]) ↔ (x = 2 ∨ x = 62) :=
by
  sorry

end NUMINAMATH_GPT_smallest_and_largest_values_l479_47944


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l479_47981

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

theorem minimum_reciprocal_sum (a m n : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : f a (-1) = -1) (h₄ : m + n = 2) (h₅ : 0 < m) (h₆ : 0 < n) :
  (1 / m) + (1 / n) = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_l479_47981


namespace NUMINAMATH_GPT_find_y_l479_47924

theorem find_y (y : ℕ) (h : 4 ^ 12 = 64 ^ y) : y = 4 :=
sorry

end NUMINAMATH_GPT_find_y_l479_47924


namespace NUMINAMATH_GPT_red_trace_larger_sphere_area_l479_47938

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end NUMINAMATH_GPT_red_trace_larger_sphere_area_l479_47938


namespace NUMINAMATH_GPT_work_done_by_student_l479_47995

theorem work_done_by_student
  (M : ℝ)  -- mass of the student
  (m : ℝ)  -- mass of the stone
  (h : ℝ)  -- height from which the stone is thrown
  (L : ℝ)  -- distance on the ice where the stone lands
  (g : ℝ)  -- acceleration due to gravity
  (t : ℝ := Real.sqrt (2 * h / g))  -- time it takes for the stone to hit the ice derived from free fall equation
  (Vk : ℝ := L / t)  -- initial speed of the stone derived from horizontal motion
  (Vu : ℝ := m / M * Vk)  -- initial speed of the student derived from conservation of momentum
  : (1/2 * m * Vk^2 + (1/2) * M * Vu^2) = 126.74 :=
by
  sorry

end NUMINAMATH_GPT_work_done_by_student_l479_47995


namespace NUMINAMATH_GPT_problem_l479_47957

theorem problem (a b : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 2 → ax^2 - bx + 2 < 0) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_problem_l479_47957


namespace NUMINAMATH_GPT_polynomial_identity_l479_47990

noncomputable def p (x : ℝ) : ℝ := x 

theorem polynomial_identity (p : ℝ → ℝ) (h : ∀ q : ℝ → ℝ, ∀ x : ℝ, p (q x) = q (p x)) : 
  (∀ x : ℝ, p x = x) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_identity_l479_47990


namespace NUMINAMATH_GPT_usual_time_is_12_l479_47925

variable (S T : ℕ)

theorem usual_time_is_12 (h1: S > 0) (h2: 5 * (T + 3) = 4 * T) : T = 12 := 
by 
  sorry

end NUMINAMATH_GPT_usual_time_is_12_l479_47925


namespace NUMINAMATH_GPT_f_g_2_equals_169_l479_47919

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + x + 3

-- The theorem statement
theorem f_g_2_equals_169 : f (g 2) = 169 :=
by
  sorry

end NUMINAMATH_GPT_f_g_2_equals_169_l479_47919


namespace NUMINAMATH_GPT_problem_l479_47930

theorem problem (a : ℝ) :
  (∀ x : ℝ, (x > 1 ↔ (x - 1 > 0 ∧ 2 * x - a > 0))) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_l479_47930


namespace NUMINAMATH_GPT_stickers_distribution_l479_47951

-- Define the mathematical problem: distributing 10 stickers among 5 sheets with each sheet getting at least one sticker.

def partitions_count (n k : ℕ) : ℕ := sorry

theorem stickers_distribution (n : ℕ) (k : ℕ) (h₁ : n = 10) (h₂ : k = 5) :
  partitions_count (n - k) k = 7 := by
  sorry

end NUMINAMATH_GPT_stickers_distribution_l479_47951


namespace NUMINAMATH_GPT_simplify_expression_l479_47958

theorem simplify_expression (x y z : ℝ) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (hx2 : x ≠ 2) (hy3 : y ≠ 3) (hz5 : z ≠ 5) :
  ( ( (x - 2) / (3 - z) * ( (y - 3) / (5 - x) ) * ( (z - 5) / (2 - y) ) ) ^ 2 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l479_47958


namespace NUMINAMATH_GPT_arithmetic_mean_is_b_l479_47962

variable (x a b : ℝ)
variable (hx : x ≠ 0)
variable (hb : b ≠ 0)

theorem arithmetic_mean_is_b : (1 / 2 : ℝ) * ((x * b + a) / x + (x * b - a) / x) = b :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_b_l479_47962


namespace NUMINAMATH_GPT_sandy_paints_area_l479_47978

-- Definition of the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 5
def door_height : ℝ := 1
def door_length : ℝ := 6.5

-- Areas computation
def wall_area : ℝ := wall_height * wall_length
def window_area : ℝ := window_height * window_length
def door_area : ℝ := door_height * door_length

-- Area to be painted
def area_not_painted : ℝ := window_area + door_area
def area_to_be_painted : ℝ := wall_area - area_not_painted

-- The theorem to prove
theorem sandy_paints_area : area_to_be_painted = 128.5 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_sandy_paints_area_l479_47978


namespace NUMINAMATH_GPT_floor_of_ten_times_expected_value_of_fourth_largest_l479_47946

theorem floor_of_ten_times_expected_value_of_fourth_largest : 
  let n := 90
  let m := 5
  let k := 4
  let E := (k * (n + 1)) / (m + 1)
  ∀ (X : Fin m → Fin n) (h : ∀ i j : Fin m, i ≠ j → X i ≠ X j), 
  Nat.floor (10 * E) = 606 := 
by
  sorry

end NUMINAMATH_GPT_floor_of_ten_times_expected_value_of_fourth_largest_l479_47946


namespace NUMINAMATH_GPT_quadratic_roots_inequality_solution_set_l479_47935

-- Problem 1 statement
theorem quadratic_roots : 
  (∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := 
by
  sorry

-- Problem 2 statement
theorem inequality_solution_set :
  (∀ x : ℝ, (x - 2 * (x - 1) ≤ 1 ∧ (1 + x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_inequality_solution_set_l479_47935


namespace NUMINAMATH_GPT_number_of_birdhouses_l479_47963

-- Definitions for the conditions
def cost_per_nail : ℝ := 0.05
def cost_per_plank : ℝ := 3.0
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def total_cost : ℝ := 88.0

-- Total cost calculation per birdhouse
def cost_per_birdhouse := planks_per_birdhouse * cost_per_plank + nails_per_birdhouse * cost_per_nail

-- Proving that the number of birdhouses is 4
theorem number_of_birdhouses : total_cost / cost_per_birdhouse = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_birdhouses_l479_47963


namespace NUMINAMATH_GPT_average_price_mixed_sugar_l479_47970

def average_selling_price_per_kg (weightA weightB weightC costA costB costC : ℕ) := 
  (costA * weightA + costB * weightB + costC * weightC) / (weightA + weightB + weightC : ℚ)

theorem average_price_mixed_sugar : 
  average_selling_price_per_kg 3 2 5 28 20 12 = 18.4 := 
by
  sorry

end NUMINAMATH_GPT_average_price_mixed_sugar_l479_47970


namespace NUMINAMATH_GPT_original_population_divisor_l479_47966

theorem original_population_divisor (a b c : ℕ) (ha : ∃ a, ∃ b, ∃ c, a^2 + 120 = b^2 ∧ b^2 + 80 = c^2) :
  7 ∣ a :=
by
  sorry

end NUMINAMATH_GPT_original_population_divisor_l479_47966


namespace NUMINAMATH_GPT_sum_of_x_intersections_l479_47941

theorem sum_of_x_intersections (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 + x) = f (3 - x))
  (m : ℕ) (xs : Fin m → ℝ) (ys : Fin m → ℝ)
  (h_intersection : ∀ i : Fin m, f (xs i) = |(xs i)^2 - 4 * (xs i) - 3|) :
  (Finset.univ.sum fun i => xs i) = 2 * m :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_intersections_l479_47941


namespace NUMINAMATH_GPT_fg_of_5_eq_163_l479_47994

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem fg_of_5_eq_163 : f (g 5) = 163 :=
by
  sorry

end NUMINAMATH_GPT_fg_of_5_eq_163_l479_47994


namespace NUMINAMATH_GPT_discount_percentage_is_20_l479_47950

theorem discount_percentage_is_20
  (regular_price_per_shirt : ℝ) (number_of_shirts : ℝ) (total_sale_price : ℝ)
  (h₁ : regular_price_per_shirt = 50) (h₂ : number_of_shirts = 6) (h₃ : total_sale_price = 240) :
  ( ( (regular_price_per_shirt * number_of_shirts - total_sale_price) / (regular_price_per_shirt * number_of_shirts) ) * 100 ) = 20 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_is_20_l479_47950


namespace NUMINAMATH_GPT_brett_total_miles_l479_47916

def miles_per_hour : ℕ := 75
def hours_driven : ℕ := 12

theorem brett_total_miles : miles_per_hour * hours_driven = 900 := 
by 
  sorry

end NUMINAMATH_GPT_brett_total_miles_l479_47916


namespace NUMINAMATH_GPT_second_group_children_is_16_l479_47989

def cases_purchased : ℕ := 13
def bottles_per_case : ℕ := 24
def camp_days : ℕ := 3
def first_group_children : ℕ := 14
def third_group_children : ℕ := 12
def bottles_per_child_per_day : ℕ := 3
def additional_bottles_needed : ℕ := 255

def fourth_group_children (x : ℕ) : ℕ := (14 + x + 12) / 2
def total_initial_bottles : ℕ := cases_purchased * bottles_per_case
def total_children (x : ℕ) : ℕ := 14 + x + 12 + fourth_group_children x 

def total_consumption (x : ℕ) : ℕ := (total_children x) * bottles_per_child_per_day * camp_days
def total_bottles_needed : ℕ := total_initial_bottles + additional_bottles_needed

theorem second_group_children_is_16 :
  ∃ x : ℕ, total_consumption x = total_bottles_needed ∧ x = 16 :=
by
  sorry

end NUMINAMATH_GPT_second_group_children_is_16_l479_47989


namespace NUMINAMATH_GPT_cakes_served_dinner_l479_47980

def total_cakes_today : Nat := 15
def cakes_served_lunch : Nat := 6

theorem cakes_served_dinner : total_cakes_today - cakes_served_lunch = 9 :=
by
  -- Define what we need to prove
  sorry -- to skip the proof

end NUMINAMATH_GPT_cakes_served_dinner_l479_47980


namespace NUMINAMATH_GPT_elderly_sample_correct_l479_47953

-- Conditions
def young_employees : ℕ := 300
def middle_aged_employees : ℕ := 150
def elderly_employees : ℕ := 100
def total_employees : ℕ := young_employees + middle_aged_employees + elderly_employees
def sample_size : ℕ := 33
def elderly_sample (total : ℕ) (elderly : ℕ) (sample : ℕ) : ℕ := (sample * elderly) / total

-- Statement to prove
theorem elderly_sample_correct :
  elderly_sample total_employees elderly_employees sample_size = 6 := 
by
  sorry

end NUMINAMATH_GPT_elderly_sample_correct_l479_47953


namespace NUMINAMATH_GPT_find_r_s_l479_47915

noncomputable def r_s_proof_problem (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : Prop :=
(r, s) = (4, 5)

theorem find_r_s (r s : ℕ) (h1 : Nat.gcd r s = 1) (h2 : (r : ℝ) / s = (2 * (Real.sqrt 2 + Real.sqrt 10)) / (5 * Real.sqrt (3 + Real.sqrt 5))) : r_s_proof_problem r s h1 h2 :=
sorry

end NUMINAMATH_GPT_find_r_s_l479_47915


namespace NUMINAMATH_GPT_cost_of_5_dozen_l479_47928

noncomputable def price_per_dozen : ℝ :=
  24 / 3

noncomputable def cost_before_tax (num_dozen : ℝ) : ℝ :=
  num_dozen * price_per_dozen

noncomputable def cost_after_tax (num_dozen : ℝ) : ℝ :=
  (1 + 0.10) * cost_before_tax num_dozen

theorem cost_of_5_dozen :
  cost_after_tax 5 = 44 := 
sorry

end NUMINAMATH_GPT_cost_of_5_dozen_l479_47928
