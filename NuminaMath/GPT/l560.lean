import Mathlib

namespace NUMINAMATH_GPT_programmer_debugging_hours_l560_56019

theorem programmer_debugging_hours
    (total_hours : ℕ)
    (flow_chart_fraction : ℚ)
    (coding_fraction : ℚ)
    (meeting_fraction : ℚ)
    (flow_chart_hours : ℚ)
    (coding_hours : ℚ)
    (meeting_hours : ℚ)
    (debugging_hours : ℚ)
    (H1 : total_hours = 192)
    (H2 : flow_chart_fraction = 3 / 10)
    (H3 : coding_fraction = 3 / 8)
    (H4 : meeting_fraction = 1 / 5)
    (H5 : flow_chart_hours = flow_chart_fraction * total_hours)
    (H6 : coding_hours = coding_fraction * total_hours)
    (H7 : meeting_hours = meeting_fraction * total_hours)
    (H8 : debugging_hours = total_hours - (flow_chart_hours + coding_hours + meeting_hours))
    :
    debugging_hours = 24 :=
by 
  sorry

end NUMINAMATH_GPT_programmer_debugging_hours_l560_56019


namespace NUMINAMATH_GPT_boundary_points_distance_probability_l560_56018

theorem boundary_points_distance_probability
  (a b c : ℕ)
  (h1 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → (|x - y| ≥ 1 / 2 → True))
  (h2 : ∀ (x y : ℝ), x ∈ [0, 4] → y ∈ [0, 4] → True)
  (h3 : ∃ a b c : ℕ, a - b * Real.pi = 2 ∧ c = 4 ∧ Int.gcd (Int.ofNat a) (Int.gcd (Int.ofNat b) (Int.ofNat c)) = 1) :
  (a + b + c = 62) := sorry

end NUMINAMATH_GPT_boundary_points_distance_probability_l560_56018


namespace NUMINAMATH_GPT_minimal_volume_block_l560_56025

theorem minimal_volume_block (l m n : ℕ) (h : (l - 1) * (m - 1) * (n - 1) = 297) : l * m * n = 192 :=
sorry

end NUMINAMATH_GPT_minimal_volume_block_l560_56025


namespace NUMINAMATH_GPT_cannot_cover_chessboard_with_one_corner_removed_l560_56021

theorem cannot_cover_chessboard_with_one_corner_removed :
  ¬ (∃ (f : Fin (8*8 - 1) → Fin (64-1) × Fin (64-1)), 
        (∀ (i j : Fin (64-1)), 
          i ≠ j → f i ≠ f j) ∧ 
        (∀ (i : Fin (8 * 8 - 1)), 
          (f i).fst + (f i).snd = 2)) :=
by
  sorry

end NUMINAMATH_GPT_cannot_cover_chessboard_with_one_corner_removed_l560_56021


namespace NUMINAMATH_GPT_length_imaginary_axis_hyperbola_l560_56032

theorem length_imaginary_axis_hyperbola : 
  ∀ (a b : ℝ), (a = 2) → (b = 1) → 
  (∀ x y : ℝ, (y^2 / a^2 - x^2 = 1) → 2 * b = 2) :=
by intros a b ha hb x y h; sorry

end NUMINAMATH_GPT_length_imaginary_axis_hyperbola_l560_56032


namespace NUMINAMATH_GPT_no_positive_integer_triples_l560_56044

theorem no_positive_integer_triples (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : ¬ (x^2 + y^2 + 41 = 2^n) :=
  sorry

end NUMINAMATH_GPT_no_positive_integer_triples_l560_56044


namespace NUMINAMATH_GPT_portion_spent_in_second_store_l560_56066

theorem portion_spent_in_second_store (M : ℕ) (X : ℕ) (H : M = 180)
  (H1 : M - (M / 2 + 14) = 76)
  (H2 : X + 16 = 76)
  (H3 : M = (M / 2 + 14) + (X + 16)) :
  (X : ℚ) / M = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_portion_spent_in_second_store_l560_56066


namespace NUMINAMATH_GPT_sum_of_cos_series_l560_56060

theorem sum_of_cos_series :
  6 * Real.cos (18 * Real.pi / 180) + 2 * Real.cos (36 * Real.pi / 180) + 
  4 * Real.cos (54 * Real.pi / 180) + 6 * Real.cos (72 * Real.pi / 180) + 
  8 * Real.cos (90 * Real.pi / 180) + 10 * Real.cos (108 * Real.pi / 180) + 
  12 * Real.cos (126 * Real.pi / 180) + 14 * Real.cos (144 * Real.pi / 180) + 
  16 * Real.cos (162 * Real.pi / 180) + 18 * Real.cos (180 * Real.pi / 180) + 
  20 * Real.cos (198 * Real.pi / 180) + 22 * Real.cos (216 * Real.pi / 180) + 
  24 * Real.cos (234 * Real.pi / 180) + 26 * Real.cos (252 * Real.pi / 180) + 
  28 * Real.cos (270 * Real.pi / 180) + 30 * Real.cos (288 * Real.pi / 180) + 
  32 * Real.cos (306 * Real.pi / 180) + 34 * Real.cos (324 * Real.pi / 180) + 
  36 * Real.cos (342 * Real.pi / 180) + 38 * Real.cos (360 * Real.pi / 180) = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cos_series_l560_56060


namespace NUMINAMATH_GPT_benjamin_weekly_walks_l560_56084

def walking_miles_in_week
  (work_days_per_week : ℕ)
  (work_distance_per_day : ℕ)
  (dog_walks_per_day : ℕ)
  (dog_walk_distance : ℕ)
  (best_friend_visits_per_week : ℕ)
  (best_friend_distance : ℕ)
  (store_visits_per_week : ℕ)
  (store_distance : ℕ)
  (hike_distance_per_week : ℕ) : ℕ :=
  (work_days_per_week * work_distance_per_day) +
  (dog_walks_per_day * dog_walk_distance * 7) +
  (best_friend_visits_per_week * (best_friend_distance * 2)) +
  (store_visits_per_week * (store_distance * 2)) +
  hike_distance_per_week

theorem benjamin_weekly_walks :
  walking_miles_in_week 5 (8 * 2) 2 3 1 5 2 4 10 = 158 := 
  by
    sorry

end NUMINAMATH_GPT_benjamin_weekly_walks_l560_56084


namespace NUMINAMATH_GPT_range_of_a_l560_56047

noncomputable def curve_y (a : ℝ) (x : ℝ) : ℝ := (a - 3) * x^3 + Real.log x
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ := x^3 - a * x^2 - 3 * x + 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ deriv (curve_y a) x = 0) ∧
  (∀ x ∈ Set.Icc (1 : ℝ) 2, 0 ≤ deriv (function_f a) x) → a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l560_56047


namespace NUMINAMATH_GPT_problem_travel_time_with_current_l560_56040

theorem problem_travel_time_with_current
  (D r c : ℝ) (t : ℝ)
  (h1 : (r - c) ≠ 0)
  (h2 : D / (r - c) = 60 / 7)
  (h3 : D / r = t - 7)
  (h4 : D / (r + c) = t)
  : t = 3 + 9 / 17 := 
sorry

end NUMINAMATH_GPT_problem_travel_time_with_current_l560_56040


namespace NUMINAMATH_GPT_min_value_quadratic_l560_56058

theorem min_value_quadratic (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 1) :
  (∀ x, (a * x^2 + 2 * x + b = 0) → x = -1 / a) →
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ (∀ a b, a > b → b > 0 → a * b = 1 →
     c ≤ (a^2 + b^2) / (a - b)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l560_56058


namespace NUMINAMATH_GPT_total_paintable_area_l560_56039

-- Define the dimensions of a bedroom
def bedroom_length : ℕ := 10
def bedroom_width : ℕ := 12
def bedroom_height : ℕ := 9

-- Define the non-paintable area per bedroom
def non_paintable_area_per_bedroom : ℕ := 74

-- Number of bedrooms
def number_of_bedrooms : ℕ := 4

-- The total paintable area that we need to prove
theorem total_paintable_area : 
  4 * (2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height) - non_paintable_area_per_bedroom) = 1288 := 
by
  sorry

end NUMINAMATH_GPT_total_paintable_area_l560_56039


namespace NUMINAMATH_GPT_spending_example_l560_56095

theorem spending_example (X : ℝ) (h₁ : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end NUMINAMATH_GPT_spending_example_l560_56095


namespace NUMINAMATH_GPT_f_properties_l560_56022

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end NUMINAMATH_GPT_f_properties_l560_56022


namespace NUMINAMATH_GPT_fred_current_dimes_l560_56052

-- Definitions based on the conditions
def original_dimes : ℕ := 7
def borrowed_dimes : ℕ := 3

-- The theorem to prove
theorem fred_current_dimes : original_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_GPT_fred_current_dimes_l560_56052


namespace NUMINAMATH_GPT_diana_age_l560_56002

open Classical

theorem diana_age :
  ∃ (D : ℚ), (∃ (C E : ℚ), C = 4 * D ∧ E = D + 5 ∧ C = E) ∧ D = 5/3 :=
by
  -- Definitions and conditions are encapsulated in the existential quantifiers and the proof concludes with D = 5/3.
  sorry

end NUMINAMATH_GPT_diana_age_l560_56002


namespace NUMINAMATH_GPT_colby_mangoes_l560_56011

def mangoes_still_have (t m k : ℕ) : ℕ :=
  let r1 := t - m
  let r2 := r1 / 2
  let r3 := r1 - r2
  r3 * k

theorem colby_mangoes (t m k : ℕ) (h_t : t = 60) (h_m : m = 20) (h_k : k = 8) :
  mangoes_still_have t m k = 160 :=
by
  sorry

end NUMINAMATH_GPT_colby_mangoes_l560_56011


namespace NUMINAMATH_GPT_minimum_value_of_y_l560_56043

theorem minimum_value_of_y : ∀ x : ℝ, ∃ y : ℝ, (y = 3 * x^2 + 6 * x + 9) → y ≥ 6 :=
by
  intro x
  use (3 * (x + 1)^2 + 6)
  intro h
  sorry

end NUMINAMATH_GPT_minimum_value_of_y_l560_56043


namespace NUMINAMATH_GPT_range_of_a_in_third_quadrant_l560_56057

def pointInThirdQuadrant (x y : ℝ) := x < 0 ∧ y < 0

theorem range_of_a_in_third_quadrant (a : ℝ) (M : ℝ × ℝ) 
  (hM : M = (-1, a-1)) (hThirdQuad : pointInThirdQuadrant M.1 M.2) : 
  a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_third_quadrant_l560_56057


namespace NUMINAMATH_GPT_conversion_rates_l560_56046

noncomputable def teamADailyConversionRate (a b : ℝ) := 1.2 * b
noncomputable def teamBDailyConversionRate (a b : ℝ) := b

theorem conversion_rates (total_area : ℝ) (b : ℝ) (h1 : total_area = 1500) (h2 : b = 50) 
    (h3 : teamADailyConversionRate 1500 b * b = 1.2) 
    (h4 : teamBDailyConversionRate 1500 b = b) 
    (h5 : (1500 / teamBDailyConversionRate 1500 b) - 5 = 1500 / teamADailyConversionRate 1500 b) :
  teamADailyConversionRate 1500 b = 60 ∧ teamBDailyConversionRate 1500 b = 50 := 
by
  sorry

end NUMINAMATH_GPT_conversion_rates_l560_56046


namespace NUMINAMATH_GPT_cl_mass_percentage_in_ccl4_l560_56053

noncomputable def mass_percentage_of_cl_in_ccl4 : ℝ :=
  let mass_C : ℝ := 12.01
  let mass_Cl : ℝ := 35.45
  let num_Cl : ℝ := 4
  let total_mass_Cl : ℝ := num_Cl * mass_Cl
  let total_mass_CCl4 : ℝ := mass_C + total_mass_Cl
  (total_mass_Cl / total_mass_CCl4) * 100

theorem cl_mass_percentage_in_ccl4 :
  abs (mass_percentage_of_cl_in_ccl4 - 92.19) < 0.01 := 
sorry

end NUMINAMATH_GPT_cl_mass_percentage_in_ccl4_l560_56053


namespace NUMINAMATH_GPT_cute_angle_of_isosceles_cute_triangle_l560_56024

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end NUMINAMATH_GPT_cute_angle_of_isosceles_cute_triangle_l560_56024


namespace NUMINAMATH_GPT_cuboid_breadth_l560_56069

theorem cuboid_breadth (l h A : ℝ) (w : ℝ) :
  l = 8 ∧ h = 12 ∧ A = 960 → 2 * (l * w + l * h + w * h) = A → w = 19.2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cuboid_breadth_l560_56069


namespace NUMINAMATH_GPT_find_m_for_line_passing_through_circle_center_l560_56010

theorem find_m_for_line_passing_through_circle_center :
  ∀ (m : ℝ), (∀ (x y : ℝ), 2 * x + y + m = 0 ↔ (x - 1)^2 + (y + 2)^2 = 5) → m = 0 :=
by
  intro m
  intro h
  -- Here we construct that the center (1, -2) must lie on the line 2x + y + m = 0
  -- using the given condition of the circle center.
  have center := h 1 (-2)
  -- solving for the equation at the point (1, -2) must yield m = 0
  sorry

end NUMINAMATH_GPT_find_m_for_line_passing_through_circle_center_l560_56010


namespace NUMINAMATH_GPT_ethanol_combustion_heat_l560_56068

theorem ethanol_combustion_heat (Q : Real) :
  (∃ (m : Real), m = 0.1 ∧ (∀ (n : Real), n = 1 → Q * n / m = 10 * Q)) :=
by
  sorry

end NUMINAMATH_GPT_ethanol_combustion_heat_l560_56068


namespace NUMINAMATH_GPT_B_work_time_l560_56096

theorem B_work_time :
  (∀ A_efficiency : ℝ, A_efficiency = 1 / 12 → ∀ B_efficiency : ℝ, B_efficiency = A_efficiency * 1.2 → (1 / B_efficiency = 10)) :=
by
  intros A_efficiency A_efficiency_eq B_efficiency B_efficiency_eq
  sorry

end NUMINAMATH_GPT_B_work_time_l560_56096


namespace NUMINAMATH_GPT_distance_of_course_l560_56077

-- Definitions
def teamESpeed : ℕ := 20
def teamASpeed : ℕ := teamESpeed + 5

-- Time taken by Team E
variable (tE : ℕ)

-- Distance calculation
def teamEDistance : ℕ := teamESpeed * tE
def teamADistance : ℕ := teamASpeed * (tE - 3)

-- Proof statement
theorem distance_of_course (tE : ℕ) (h : teamEDistance tE = teamADistance tE) : teamEDistance tE = 300 :=
sorry

end NUMINAMATH_GPT_distance_of_course_l560_56077


namespace NUMINAMATH_GPT_length_of_BC_l560_56042

theorem length_of_BC (BD CD : ℝ) (h1 : BD = 3 + 3 * BD) (h2 : CD = 2 + 2 * CD) (h3 : 4 * BD + 3 * CD + 5 = 20) : 2 * CD + 2 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_BC_l560_56042


namespace NUMINAMATH_GPT_son_and_daughter_current_ages_l560_56062

theorem son_and_daughter_current_ages
  (father_age_now : ℕ)
  (son_age_5_years_ago : ℕ)
  (daughter_age_5_years_ago : ℝ)
  (h_father_son_birth : father_age_now - (son_age_5_years_ago + 5) = (son_age_5_years_ago + 5))
  (h_father_daughter_birth : father_age_now - (daughter_age_5_years_ago + 5) = (daughter_age_5_years_ago + 5))
  (h_daughter_half_son_5_years_ago : daughter_age_5_years_ago = son_age_5_years_ago / 2) :
  son_age_5_years_ago + 5 = 12 ∧ daughter_age_5_years_ago + 5 = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_son_and_daughter_current_ages_l560_56062


namespace NUMINAMATH_GPT_quadratic_roots_condition_l560_56048

theorem quadratic_roots_condition (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 0) :
  ¬ ((∃ x y : ℝ, ax^2 + 2*x + 1 = 0 ∧ ax^2 + 2*y + 1 = 0 ∧ x*y < 0) ↔
     (a > 0 ∧ a ≠ 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_condition_l560_56048


namespace NUMINAMATH_GPT_percentage_error_l560_56035

-- Define the conditions
def actual_side (a : ℝ) := a
def measured_side (a : ℝ) := 1.05 * a
def actual_area (a : ℝ) := a^2
def calculated_area (a : ℝ) := (1.05 * a)^2

-- Define the statement that we need to prove
theorem percentage_error (a : ℝ) (h : a > 0) :
  (calculated_area a - actual_area a) / actual_area a * 100 = 10.25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_error_l560_56035


namespace NUMINAMATH_GPT_john_marble_choices_l560_56013

open Nat

theorem john_marble_choices :
  (choose 4 2) * (choose 12 3) = 1320 :=
by
  sorry

end NUMINAMATH_GPT_john_marble_choices_l560_56013


namespace NUMINAMATH_GPT_correct_option_is_B_l560_56092

-- Define the operations as hypotheses
def option_A (a : ℤ) : Prop := (a^2 + a^3 = a^5)
def option_B (a : ℤ) : Prop := ((a^2)^3 = a^6)
def option_C (a : ℤ) : Prop := (a^2 * a^3 = a^6)
def option_D (a : ℤ) : Prop := (6 * a^6 - 2 * a^3 = 3 * a^3)

-- Prove that option B is correct
theorem correct_option_is_B (a : ℤ) : option_B a :=
by
  unfold option_B
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l560_56092


namespace NUMINAMATH_GPT_percentage_of_orange_and_watermelon_juice_l560_56020

-- Define the total volume of the drink
def total_volume := 150

-- Define the volume of grape juice in the drink
def grape_juice_volume := 45

-- Define the percentage calculation for grape juice
def grape_juice_percentage := (grape_juice_volume / total_volume) * 100

-- Define the remaining percentage that is made of orange and watermelon juices
def remaining_percentage := 100 - grape_juice_percentage

-- Define the percentage of orange and watermelon juice being the same
def orange_and_watermelon_percentage := remaining_percentage / 2

theorem percentage_of_orange_and_watermelon_juice : 
  orange_and_watermelon_percentage = 35 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_percentage_of_orange_and_watermelon_juice_l560_56020


namespace NUMINAMATH_GPT_seq_a_n_a_4_l560_56075

theorem seq_a_n_a_4 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ n : ℕ, a (n+1) = 2 * a n) ∧ (a 4 = 8) :=
sorry

end NUMINAMATH_GPT_seq_a_n_a_4_l560_56075


namespace NUMINAMATH_GPT_find_line_equation_l560_56093

open Real

noncomputable def line_equation (x y : ℝ) (k : ℝ) : ℝ := k * x - y + 4 - 3 * k

noncomputable def distance_to_line (x1 y1 k : ℝ) : ℝ :=
  abs (k * x1 - y1 + 4 - 3 * k) / sqrt (k^2 + 1)

theorem find_line_equation :
  (∃ k : ℝ, (k = 2 ∨ k = -2 / 3) ∧
    (∀ x y, (x, y) = (3, 4) → (2 * x - y - 2 = 0 ∨ 2 * x + 3 * y - 18 = 0)))
    ∧ (line_equation (-2) 2 2 = line_equation 4 (-2) 2)
    ∧ (line_equation (-2) 2 (-2 / 3) = line_equation 4 (-2) (-2 / 3)) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l560_56093


namespace NUMINAMATH_GPT_problem1_l560_56087

theorem problem1 (a : ℝ) (x : ℝ) (h : a > 0) : |x - (1/a)| + |x + a| ≥ 2 :=
sorry

end NUMINAMATH_GPT_problem1_l560_56087


namespace NUMINAMATH_GPT_number_of_symmetric_subsets_l560_56078

def has_integer_solutions (m : ℤ) : Prop :=
  ∃ x y : ℤ, x * y = -36 ∧ x + y = -m

def M : Set ℤ :=
  {m | has_integer_solutions m}

def is_symmetric_subset (A : Set ℤ) : Prop :=
  A ⊆ M ∧ ∀ a ∈ A, -a ∈ A

theorem number_of_symmetric_subsets :
  (∃ A : Set ℤ, is_symmetric_subset A ∧ A ≠ ∅) →
  (∃ n : ℕ, n = 31) :=
by
  sorry

end NUMINAMATH_GPT_number_of_symmetric_subsets_l560_56078


namespace NUMINAMATH_GPT_rods_in_one_mile_l560_56051

-- Define the conditions as assumptions in Lean

-- 1. 1 mile = 8 furlongs
def mile_to_furlong : ℕ := 8

-- 2. 1 furlong = 220 paces
def furlong_to_pace : ℕ := 220

-- 3. 1 pace = 0.2 rods
def pace_to_rod : ℝ := 0.2

-- Define the statement to be proven
theorem rods_in_one_mile : (mile_to_furlong * furlong_to_pace * pace_to_rod) = 352 := by
  sorry

end NUMINAMATH_GPT_rods_in_one_mile_l560_56051


namespace NUMINAMATH_GPT_angles_in_first_or_third_quadrant_l560_56001

noncomputable def angles_first_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}

noncomputable def angles_third_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}

theorem angles_in_first_or_third_quadrant :
  ∃ S1 S2 : Set ℝ, 
    (S1 = {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}) ∧
    (S2 = {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}) ∧
    (angles_first_quadrant_set = S1 ∧ angles_third_quadrant_set = S2)
  :=
sorry

end NUMINAMATH_GPT_angles_in_first_or_third_quadrant_l560_56001


namespace NUMINAMATH_GPT_batsman_average_19th_inning_l560_56017

theorem batsman_average_19th_inning (initial_avg : ℝ) 
    (scored_19th_inning : ℝ) 
    (new_avg : ℝ) 
    (h1 : scored_19th_inning = 100) 
    (h2 : new_avg = initial_avg + 2)
    (h3 : new_avg = (18 * initial_avg + 100) / 19) :
    new_avg = 64 :=
by
  have h4 : initial_avg = 62 := by
    sorry
  sorry

end NUMINAMATH_GPT_batsman_average_19th_inning_l560_56017


namespace NUMINAMATH_GPT_initial_amount_l560_56014

variable (X : ℝ)

/--
An individual deposited 20% of 25% of 30% of their initial amount into their bank account.
If the deposited amount is Rs. 750, prove that their initial amount was Rs. 50000.
-/
theorem initial_amount (h : (0.2 * 0.25 * 0.3 * X) = 750) : X = 50000 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l560_56014


namespace NUMINAMATH_GPT_triangle_obtuse_l560_56036

-- We need to set up the definitions for angles and their relationships in triangles.

variable {A B C : ℝ} -- representing the angles of the triangle in radians

structure Triangle (A B C : ℝ) : Prop where
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  sum_to_pi : A + B + C = Real.pi -- representing the sum of angles in a triangle

-- Definition to state the condition in the problem
def triangle_condition (A B C : ℝ) : Prop :=
  Triangle A B C ∧ (Real.cos A * Real.cos B - Real.sin A * Real.sin B > 0)

-- Theorem to prove the triangle is obtuse under the given condition
theorem triangle_obtuse {A B C : ℝ} (h : triangle_condition A B C) : ∃ C', C' = C ∧ C' > Real.pi / 2 :=
sorry

end NUMINAMATH_GPT_triangle_obtuse_l560_56036


namespace NUMINAMATH_GPT_factorization_mn_l560_56028

variable (m n : ℝ) -- Declare m and n as arbitrary real numbers.

theorem factorization_mn (m n : ℝ) : m^2 - m * n = m * (m - n) := by
  sorry

end NUMINAMATH_GPT_factorization_mn_l560_56028


namespace NUMINAMATH_GPT_sam_walking_speed_l560_56080

variable (s : ℝ)
variable (t : ℝ)
variable (fred_speed : ℝ := 2)
variable (sam_distance : ℝ := 25)
variable (total_distance : ℝ := 35)

theorem sam_walking_speed :
  (total_distance - sam_distance) = fred_speed * t ∧
  sam_distance = s * t →
  s = 5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_sam_walking_speed_l560_56080


namespace NUMINAMATH_GPT_optimal_play_winner_l560_56026

-- Definitions for the conditions
def chessboard_size (K N : ℕ) : Prop := True
def rook_initial_position (K N : ℕ) : (ℕ × ℕ) :=
  (K, N)
def move (r : ℕ × ℕ) (direction : ℕ) : (ℕ × ℕ) :=
  if direction = 0 then (r.1 - 1, r.2)
  else (r.1, r.2 - 1)
def rook_cannot_move (r : ℕ × ℕ) : Prop :=
  r.1 = 0 ∨ r.2 = 0

-- Theorem to prove the winner given the conditions
theorem optimal_play_winner (K N : ℕ) :
  (K = N → ∃ player : ℕ, player = 2) ∧ (K ≠ N → ∃ player : ℕ, player = 1) :=
by
  sorry

end NUMINAMATH_GPT_optimal_play_winner_l560_56026


namespace NUMINAMATH_GPT_max_trading_cards_l560_56015

variable (money : ℝ) (cost_per_card : ℝ) (max_cards : ℕ)

theorem max_trading_cards (h_money : money = 9) (h_cost : cost_per_card = 1) : max_cards ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_trading_cards_l560_56015


namespace NUMINAMATH_GPT_composite_dice_product_probability_l560_56071

theorem composite_dice_product_probability :
  let outcomes := 6 ^ 4
  let non_composite_ways := 13
  let composite_probability := 1 - non_composite_ways / outcomes
  composite_probability = 1283 / 1296 :=
by
  sorry

end NUMINAMATH_GPT_composite_dice_product_probability_l560_56071


namespace NUMINAMATH_GPT_M_gt_N_l560_56097

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2*x + 2*y - 2

theorem M_gt_N : M x y > N x y :=
by
  sorry

end NUMINAMATH_GPT_M_gt_N_l560_56097


namespace NUMINAMATH_GPT_savings_is_22_77_cents_per_egg_l560_56099

-- Defining the costs and discount condition
def cost_per_large_egg_StoreA : ℚ := 0.55
def cost_per_extra_large_egg_StoreA : ℚ := 0.65
def discounted_cost_of_three_trays_large_StoreB : ℚ := 38
def total_eggs_in_three_trays : ℕ := 90

-- Savings calculation
def savings_per_egg : ℚ := (cost_per_extra_large_egg_StoreA - (discounted_cost_of_three_trays_large_StoreB / total_eggs_in_three_trays)) * 100

-- The statement to prove
theorem savings_is_22_77_cents_per_egg : savings_per_egg = 22.77 :=
by
  -- Here the proof would go, but we are omitting it with sorry
  sorry

end NUMINAMATH_GPT_savings_is_22_77_cents_per_egg_l560_56099


namespace NUMINAMATH_GPT_domain_of_inverse_function_l560_56065

noncomputable def log_inverse_domain : Set ℝ :=
  {y | y ≥ 5}

theorem domain_of_inverse_function :
  ∀ y, y ∈ log_inverse_domain ↔ ∃ x, x ≥ 3 ∧ y = 4 + Real.logb 2 (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_inverse_function_l560_56065


namespace NUMINAMATH_GPT_lisa_investment_in_stocks_l560_56050

-- Definitions for the conditions
def total_investment (r : ℝ) : Prop := r + 7 * r = 200000
def stock_investment (r : ℝ) : ℝ := 7 * r

-- Given the conditions, we need to prove the amount invested in stocks
theorem lisa_investment_in_stocks (r : ℝ) (h : total_investment r) : stock_investment r = 175000 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_lisa_investment_in_stocks_l560_56050


namespace NUMINAMATH_GPT_temperature_decrease_time_l560_56041

theorem temperature_decrease_time
  (T_initial T_final T_per_hour : ℤ)
  (h_initial : T_initial = -5)
  (h_final : T_final = -25)
  (h_decrease : T_per_hour = -5) :
  (T_final - T_initial) / T_per_hour = 4 := by
sorry

end NUMINAMATH_GPT_temperature_decrease_time_l560_56041


namespace NUMINAMATH_GPT_sacks_per_day_l560_56055

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (harvest_rate : ℕ)
  (h1 : total_sacks = 498)
  (h2 : days = 6)
  (h3 : harvest_rate = total_sacks / days) :
  harvest_rate = 83 := by
  sorry

end NUMINAMATH_GPT_sacks_per_day_l560_56055


namespace NUMINAMATH_GPT_lcm_220_504_l560_56027

/-- The least common multiple of 220 and 504 is 27720. -/
theorem lcm_220_504 : Nat.lcm 220 504 = 27720 :=
by
  -- This is the final statement of the theorem. The proof is not provided and marked with 'sorry'.
  sorry

end NUMINAMATH_GPT_lcm_220_504_l560_56027


namespace NUMINAMATH_GPT_sum_of_interior_diagonals_l560_56007

theorem sum_of_interior_diagonals (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 50) (h2 : x * y + y * z + z * x = 47) : 
  4 * Real.sqrt (x^2 + y^2 + z^2) = 20 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_interior_diagonals_l560_56007


namespace NUMINAMATH_GPT_estimate_probability_concave_l560_56034

noncomputable def times_thrown : ℕ := 1000
noncomputable def frequency_convex : ℝ := 0.44

theorem estimate_probability_concave :
  (1 - frequency_convex) = 0.56 := by
  sorry

end NUMINAMATH_GPT_estimate_probability_concave_l560_56034


namespace NUMINAMATH_GPT_servings_in_box_l560_56000

-- Define amounts
def total_cereal : ℕ := 18
def per_serving : ℕ := 2

-- Define the statement to prove
theorem servings_in_box : total_cereal / per_serving = 9 :=
by
  sorry

end NUMINAMATH_GPT_servings_in_box_l560_56000


namespace NUMINAMATH_GPT_average_age_of_students_l560_56073

theorem average_age_of_students :
  (8 * 14 + 6 * 16 + 17) / 15 = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_students_l560_56073


namespace NUMINAMATH_GPT_total_formula_portions_l560_56029

def puppies : ℕ := 7
def feedings_per_day : ℕ := 3
def days : ℕ := 5

theorem total_formula_portions : 
  (feedings_per_day * days * puppies = 105) := 
by
  sorry

end NUMINAMATH_GPT_total_formula_portions_l560_56029


namespace NUMINAMATH_GPT_minimum_value_on_line_l560_56049

theorem minimum_value_on_line : ∃ (x y : ℝ), (x + y = 4) ∧ (∀ x' y', (x' + y' = 4) → (x^2 + y^2 ≤ x'^2 + y'^2)) ∧ (x^2 + y^2 = 8) :=
sorry

end NUMINAMATH_GPT_minimum_value_on_line_l560_56049


namespace NUMINAMATH_GPT_husband_monthly_savings_l560_56064

theorem husband_monthly_savings :
  let wife_weekly_savings := 100
  let weeks_in_month := 4
  let months := 4
  let total_weeks := weeks_in_month * months
  let wife_savings := wife_weekly_savings * total_weeks
  let stock_price := 50
  let number_of_shares := 25
  let invested_half := stock_price * number_of_shares
  let total_savings := invested_half * 2
  let husband_savings := total_savings - wife_savings
  let monthly_husband_savings := husband_savings / months
  monthly_husband_savings = 225 := 
by 
  sorry

end NUMINAMATH_GPT_husband_monthly_savings_l560_56064


namespace NUMINAMATH_GPT_equation_solution_l560_56090

theorem equation_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (2 * x / (x - 2) - 2 = 1 / (x * (x - 2))) ↔ x = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l560_56090


namespace NUMINAMATH_GPT_gcd_lcm_product_l560_56030

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 45) :
  Nat.gcd a b * Nat.lcm a b = 1350 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l560_56030


namespace NUMINAMATH_GPT_sugar_water_sweeter_l560_56082

variable (a b m : ℝ)
variable (a_pos : a > 0) (b_gt_a : b > a) (m_pos : m > 0)

theorem sugar_water_sweeter : (a + m) / (b + m) > a / b :=
by
  sorry

end NUMINAMATH_GPT_sugar_water_sweeter_l560_56082


namespace NUMINAMATH_GPT_road_completion_days_l560_56089

variable (L : ℕ) (M_1 : ℕ) (W_1 : ℕ) (t1 : ℕ) (M_2 : ℕ)

theorem road_completion_days : L = 10 ∧ M_1 = 30 ∧ W_1 = 2 ∧ t1 = 5 ∧ M_2 = 60 → D = 15 :=
by
  sorry

end NUMINAMATH_GPT_road_completion_days_l560_56089


namespace NUMINAMATH_GPT_part1_l560_56054

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 :=
by
  sorry

end NUMINAMATH_GPT_part1_l560_56054


namespace NUMINAMATH_GPT_tangent_parabola_line_l560_56063

theorem tangent_parabola_line (a : ℝ) :
  (∃ x0 : ℝ, ax0^2 + 3 = 2 * x0 + 1) ∧ (∀ x : ℝ, a * x^2 - 2 * x + 2 = 0 → x = x0) → a = 1/2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tangent_parabola_line_l560_56063


namespace NUMINAMATH_GPT_option_C_is_proposition_l560_56005

def is_proposition (s : Prop) : Prop := ∃ p : Prop, s = p

theorem option_C_is_proposition : is_proposition (4 + 3 = 8) := sorry

end NUMINAMATH_GPT_option_C_is_proposition_l560_56005


namespace NUMINAMATH_GPT_squares_count_correct_l560_56059

-- Assuming basic setup and coordinate system.
def is_valid_point (x y : ℕ) : Prop :=
  x ≤ 8 ∧ y ≤ 8

-- Checking if a point (a, b) in the triangle as described.
def is_in_triangle (a b : ℕ) : Prop :=
  0 ≤ b ∧ b ≤ a ∧ a ≤ 4

-- Function derived from the solution detailing the number of such squares.
def count_squares (a b : ℕ) : ℕ :=
  -- Placeholder to represent the derived formula - to be replaced with actual derivation function
  (9 - a + b) * (a + b + 1) - 1

-- Statement to prove
theorem squares_count_correct (a b : ℕ) (h : is_in_triangle a b) :
  ∃ n, n = count_squares a b := 
sorry

end NUMINAMATH_GPT_squares_count_correct_l560_56059


namespace NUMINAMATH_GPT_triangle_area_of_parabola_hyperbola_l560_56076

-- Definitions for parabola and hyperbola
def parabola_directrix (a : ℕ) (x y : ℝ) : Prop := x^2 = 16 * y
def hyperbola_asymptotes (a b : ℕ) (x y : ℝ) : Prop := x^2 / (a^2) - y^2 / (b^2) = 1

-- Theorem stating the area of the triangle formed by the intersections of the asymptotes with the directrix
theorem triangle_area_of_parabola_hyperbola (a b : ℕ) (h : a = 1) (h' : b = 1) : 
  ∃ (area : ℝ), area = 16 :=
sorry

end NUMINAMATH_GPT_triangle_area_of_parabola_hyperbola_l560_56076


namespace NUMINAMATH_GPT_right_triangle_second_arm_square_l560_56023

theorem right_triangle_second_arm_square :
  ∀ (k : ℤ) (a : ℤ) (c : ℤ) (b : ℤ),
  a = 2 * k + 1 → 
  c = 2 * k + 3 → 
  a^2 + b^2 = c^2 → 
  b^2 ≠ a * c ∧ b^2 ≠ (c / a) ∧ b^2 ≠ (a + c) ∧ b^2 ≠ (c - a) :=
by sorry

end NUMINAMATH_GPT_right_triangle_second_arm_square_l560_56023


namespace NUMINAMATH_GPT_train_speed_fraction_l560_56081

theorem train_speed_fraction (T : ℝ) (hT : T = 3) : T / (T + 0.5) = 6 / 7 := by
  sorry

end NUMINAMATH_GPT_train_speed_fraction_l560_56081


namespace NUMINAMATH_GPT_factorial_divisibility_l560_56086

theorem factorial_divisibility 
  (n k : ℕ) 
  (p : ℕ) 
  [hp : Fact (Nat.Prime p)] 
  (h1 : 0 < n) 
  (h2 : 0 < k) 
  (h3 : p ^ k ∣ n!) : 
  (p! ^ k ∣ n!) :=
sorry

end NUMINAMATH_GPT_factorial_divisibility_l560_56086


namespace NUMINAMATH_GPT_average_mark_first_class_l560_56070

theorem average_mark_first_class (A : ℝ)
  (class1_students class2_students : ℝ)
  (avg2 combined_avg total_students total_marks_combined : ℝ)
  (h1 : class1_students = 22)
  (h2 : class2_students = 28)
  (h3 : avg2 = 60)
  (h4 : combined_avg = 51.2)
  (h5 : total_students = class1_students + class2_students)
  (h6 : total_marks_combined = total_students * combined_avg)
  (h7 : 22 * A + 28 * avg2 = total_marks_combined) :
  A = 40 :=
by
  sorry

end NUMINAMATH_GPT_average_mark_first_class_l560_56070


namespace NUMINAMATH_GPT_avg_daily_production_l560_56009

theorem avg_daily_production (x y : ℕ) (h1 : x + y = 350) (h2 : 2 * x - y = 250) : x = 200 ∧ y = 150 := 
by
  sorry

end NUMINAMATH_GPT_avg_daily_production_l560_56009


namespace NUMINAMATH_GPT_claire_crafting_hours_l560_56045

theorem claire_crafting_hours (H1 : 24 = 24) (H2 : 8 = 8) (H3 : 4 = 4) (H4 : 2 = 2):
  let total_hours_per_day := 24
  let sleep_hours := 8
  let cleaning_hours := 4
  let cooking_hours := 2
  let working_hours := total_hours_per_day - sleep_hours
  let remaining_hours := working_hours - (cleaning_hours + cooking_hours)
  let crafting_hours := remaining_hours / 2
  crafting_hours = 5 :=
by
  sorry

end NUMINAMATH_GPT_claire_crafting_hours_l560_56045


namespace NUMINAMATH_GPT_brendan_yards_per_week_l560_56088

def original_speed_flat : ℝ := 8  -- Brendan's speed on flat terrain in yards/day
def improvement_flat : ℝ := 0.5   -- Lawn mower improvement on flat terrain (50%)
def reduction_uneven : ℝ := 0.35  -- Speed reduction on uneven terrain (35%)
def days_flat : ℝ := 4            -- Days on flat terrain
def days_uneven : ℝ := 3          -- Days on uneven terrain

def improved_speed_flat : ℝ := original_speed_flat * (1 + improvement_flat)
def speed_uneven : ℝ := improved_speed_flat * (1 - reduction_uneven)

def total_yards_week : ℝ := (improved_speed_flat * days_flat) + (speed_uneven * days_uneven)

theorem brendan_yards_per_week : total_yards_week = 71.4 :=
sorry

end NUMINAMATH_GPT_brendan_yards_per_week_l560_56088


namespace NUMINAMATH_GPT_geom_seq_sum_eq_six_l560_56012

theorem geom_seq_sum_eq_six 
    (a : ℕ → ℝ) 
    (r : ℝ) 
    (h_geom : ∀ n, a (n + 1) = a n * r) 
    (h_pos : ∀ n, a n > 0)
    (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) 
    : a 5 + a 7 = 6 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_eq_six_l560_56012


namespace NUMINAMATH_GPT_line_equation_l560_56037

theorem line_equation (P A B : ℝ × ℝ) (h1 : P = (-1, 3)) (h2 : A = (1, 2)) (h3 : B = (3, 1)) :
  ∃ c : ℝ, (x - 2*y + c = 0) ∧ (4*x - 2*y - 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l560_56037


namespace NUMINAMATH_GPT_channel_width_at_top_l560_56016

theorem channel_width_at_top 
  (area : ℝ) (bottom_width : ℝ) (depth : ℝ) 
  (H1 : bottom_width = 6) 
  (H2 : area = 630) 
  (H3 : depth = 70) : 
  ∃ w : ℝ, (∃ H : w + 6 > 0, area = 1 / 2 * (w + bottom_width) * depth) ∧ w = 12 :=
by
  sorry

end NUMINAMATH_GPT_channel_width_at_top_l560_56016


namespace NUMINAMATH_GPT_fib_100_mod_5_l560_56031

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end NUMINAMATH_GPT_fib_100_mod_5_l560_56031


namespace NUMINAMATH_GPT_work_completed_together_l560_56061

theorem work_completed_together (A_days B_days : ℕ) (hA : A_days = 40) (hB : B_days = 60) : 
  1 / (1 / (A_days: ℝ) + 1 / (B_days: ℝ)) = 24 :=
by
  sorry

end NUMINAMATH_GPT_work_completed_together_l560_56061


namespace NUMINAMATH_GPT_max_wrappers_l560_56091

-- Definitions for the conditions
def total_wrappers : ℕ := 49
def andy_wrappers : ℕ := 34

-- The problem statement to prove
theorem max_wrappers : total_wrappers - andy_wrappers = 15 :=
by
  sorry

end NUMINAMATH_GPT_max_wrappers_l560_56091


namespace NUMINAMATH_GPT_problem_a_l560_56008

theorem problem_a (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  Int.floor (5 * x) + Int.floor (5 * y) ≥ Int.floor (3 * x + y) + Int.floor (3 * y + x) :=
sorry

end NUMINAMATH_GPT_problem_a_l560_56008


namespace NUMINAMATH_GPT_fewer_females_than_males_l560_56038

theorem fewer_females_than_males 
  (total_students : ℕ)
  (female_students : ℕ)
  (h_total : total_students = 280)
  (h_female : female_students = 127) :
  total_students - female_students - female_students = 26 := by
  sorry

end NUMINAMATH_GPT_fewer_females_than_males_l560_56038


namespace NUMINAMATH_GPT_half_abs_diff_of_squares_l560_56094

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end NUMINAMATH_GPT_half_abs_diff_of_squares_l560_56094


namespace NUMINAMATH_GPT_cab_driver_income_l560_56003

theorem cab_driver_income (x : ℕ) 
  (h₁ : (45 + x + 60 + 65 + 70) / 5 = 58) : x = 50 := 
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_cab_driver_income_l560_56003


namespace NUMINAMATH_GPT_translate_one_chapter_in_three_hours_l560_56098

-- Definitions representing the conditions:
def jun_seok_time : ℝ := 4
def yoon_yeol_time : ℝ := 12

-- Question and Correct answer as a statement:
theorem translate_one_chapter_in_three_hours :
  (1 / (1 / jun_seok_time + 1 / yoon_yeol_time)) = 3 := by
sorry

end NUMINAMATH_GPT_translate_one_chapter_in_three_hours_l560_56098


namespace NUMINAMATH_GPT_max_consecutive_sum_le_1000_l560_56072

theorem max_consecutive_sum_le_1000 : 
  ∃ n : ℕ, (∀ m : ℕ, m ≤ n → (m * (m + 1)) / 2 < 1000) ∧ ¬∃ n' : ℕ, n < n' ∧ (n' * (n' + 1)) / 2 < 1000 :=
sorry

end NUMINAMATH_GPT_max_consecutive_sum_le_1000_l560_56072


namespace NUMINAMATH_GPT_quadratic_has_minimum_l560_56083

theorem quadratic_has_minimum 
  (a b : ℝ) (h : a ≠ 0) (g : ℝ → ℝ) 
  (H : ∀ x, g x = a * x^2 + b * x + (b^2 / a)) :
  ∃ x₀, ∀ x, g x ≥ g x₀ :=
by sorry

end NUMINAMATH_GPT_quadratic_has_minimum_l560_56083


namespace NUMINAMATH_GPT_methane_production_proof_l560_56079

noncomputable def methane_production
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : Prop :=
  methane_formed = 3

theorem methane_production_proof 
  (C H : ℕ)
  (methane_formed : ℕ)
  (h_formula : ∀ c h, c = 1 ∧ h = 2)
  (h_initial_conditions : C = 3 ∧ H = 6)
  (h_reaction : ∀ (c h m : ℕ), c = 1 ∧ h = 2 → m = 1) : methane_production C H methane_formed h_formula h_initial_conditions h_reaction :=
by {
  sorry
}

end NUMINAMATH_GPT_methane_production_proof_l560_56079


namespace NUMINAMATH_GPT_problem_statement_l560_56067

def S (a b : ℤ) : ℤ := 4 * a + 6 * b
def T (a b : ℤ) : ℤ := 2 * a - 3 * b

theorem problem_statement : T (S 8 3) 4 = 88 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l560_56067


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l560_56056

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l560_56056


namespace NUMINAMATH_GPT_friends_popcorn_l560_56085

theorem friends_popcorn (pieces_per_serving : ℕ) (jared_count : ℕ) (total_servings : ℕ) (jared_friends : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_count = 90)
  (h3 : total_servings = 9)
  (h4 : jared_friends = 3) :
  (total_servings * pieces_per_serving - jared_count) / jared_friends = 60 := by
  sorry

end NUMINAMATH_GPT_friends_popcorn_l560_56085


namespace NUMINAMATH_GPT_amc_proposed_by_Dorlir_Ahmeti_Albania_l560_56033

-- Define the problem statement, encapsulating the conditions and the final inequality.
theorem amc_proposed_by_Dorlir_Ahmeti_Albania
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cond : a * b + b * c + c * a = 3) :
  (a / Real.sqrt (a^3 + 5) + b / Real.sqrt (b^3 + 5) + c / Real.sqrt (c^3 + 5) ≤ Real.sqrt 6 / 2) := 
by 
  sorry -- Proof steps go here, which are omitted as per the requirement.

end NUMINAMATH_GPT_amc_proposed_by_Dorlir_Ahmeti_Albania_l560_56033


namespace NUMINAMATH_GPT_find_base_b_l560_56006

theorem find_base_b : ∃ b : ℕ, (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 ∧ b = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_base_b_l560_56006


namespace NUMINAMATH_GPT_player_A_wins_iff_n_is_odd_l560_56004

-- Definitions of the problem conditions
structure ChessboardGame (n : ℕ) :=
  (stones : ℕ := 99)
  (playerA_first : Prop := true)
  (turns : ℕ := n * 99)

-- Statement of the problem
theorem player_A_wins_iff_n_is_odd (n : ℕ) (g : ChessboardGame n) : 
  PlayerA_has_winning_strategy ↔ n % 2 = 1 := 
sorry

end NUMINAMATH_GPT_player_A_wins_iff_n_is_odd_l560_56004


namespace NUMINAMATH_GPT_speed_of_train_is_correct_l560_56074

noncomputable def speedOfTrain := 
  let lengthOfTrain : ℝ := 800 -- length of the train in meters
  let timeToCrossMan : ℝ := 47.99616030717543 -- time in seconds to cross the man
  let speedOfMan : ℝ := 5 * (1000 / 3600) -- speed of the man in m/s (conversion from km/hr to m/s)
  let relativeSpeed : ℝ := lengthOfTrain / timeToCrossMan -- relative speed of the train
  let speedOfTrainInMS : ℝ := relativeSpeed + speedOfMan -- speed of the train in m/s
  let speedOfTrainInKMHR : ℝ := speedOfTrainInMS * (3600 / 1000) -- speed in km/hr
  64.9848 -- result is approximately 64.9848 km/hr

theorem speed_of_train_is_correct :
  speedOfTrain = 64.9848 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_train_is_correct_l560_56074
