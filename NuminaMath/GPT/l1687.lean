import Mathlib

namespace NUMINAMATH_GPT_sam_pens_count_l1687_168720

-- Lean 4 statement
theorem sam_pens_count :
  ∃ (black_pens blue_pens pencils red_pens : ℕ),
    (black_pens = blue_pens + 10) ∧
    (blue_pens = 2 * pencils) ∧
    (pencils = 8) ∧
    (red_pens = pencils - 2) ∧
    (black_pens + blue_pens + red_pens = 48) :=
by {
  sorry
}

end NUMINAMATH_GPT_sam_pens_count_l1687_168720


namespace NUMINAMATH_GPT_quotient_of_division_l1687_168717

theorem quotient_of_division (L : ℕ) (S : ℕ) (Q : ℕ) (h1 : L = 1631) (h2 : L - S = 1365) (h3 : L = S * Q + 35) :
  Q = 6 :=
by
  sorry

end NUMINAMATH_GPT_quotient_of_division_l1687_168717


namespace NUMINAMATH_GPT_mary_average_speed_l1687_168780

noncomputable def trip_distance : ℝ := 1.5 + 1.5
noncomputable def trip_time_minutes : ℝ := 45 + 15
noncomputable def trip_time_hours : ℝ := trip_time_minutes / 60

theorem mary_average_speed :
  (trip_distance / trip_time_hours) = 3 := by
  sorry

end NUMINAMATH_GPT_mary_average_speed_l1687_168780


namespace NUMINAMATH_GPT_john_will_lose_weight_in_80_days_l1687_168783

-- Assumptions based on the problem conditions
def calories_eaten : ℕ := 1800
def calories_burned : ℕ := 2300
def calories_to_lose_one_pound : ℕ := 4000
def pounds_to_lose : ℕ := 10

-- Definition of the net calories burned per day
def net_calories_burned_per_day : ℕ := calories_burned - calories_eaten

-- Definition of total calories to lose the target weight
def total_calories_to_lose_target_weight (pounds_to_lose : ℕ) : ℕ :=
  calories_to_lose_one_pound * pounds_to_lose

-- Definition of days to lose the target weight
def days_to_lose_weight (target_calories : ℕ) (daily_net_calories : ℕ) : ℕ :=
  target_calories / daily_net_calories

-- Prove that John will lose 10 pounds in 80 days
theorem john_will_lose_weight_in_80_days :
  days_to_lose_weight (total_calories_to_lose_target_weight pounds_to_lose) net_calories_burned_per_day = 80 := by
  sorry

end NUMINAMATH_GPT_john_will_lose_weight_in_80_days_l1687_168783


namespace NUMINAMATH_GPT_sequence_mono_iff_b_gt_neg3_l1687_168739

theorem sequence_mono_iff_b_gt_neg3 (b : ℝ) : 
  (∀ n : ℕ, 1 ≤ n → (n + 1) ^ 2 + b * (n + 1) > n ^ 2 + b * n) → b > -3 := 
by
  sorry

end NUMINAMATH_GPT_sequence_mono_iff_b_gt_neg3_l1687_168739


namespace NUMINAMATH_GPT_number_of_sides_l1687_168781

theorem number_of_sides (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_sides_l1687_168781


namespace NUMINAMATH_GPT_third_side_triangle_l1687_168733

theorem third_side_triangle (a : ℝ) :
  (5 < a ∧ a < 13) → (a = 8) :=
sorry

end NUMINAMATH_GPT_third_side_triangle_l1687_168733


namespace NUMINAMATH_GPT_magnitude_b_magnitude_c_area_l1687_168775

-- Define the triangle ABC and parameters
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi_div2 : A < Real.pi / 2)
variables (triangle_condition : a = Real.sqrt 15) (sin_A : Real.sin A = 1 / 4)

-- Problem 1
theorem magnitude_b (cos_B : Real.cos B = Real.sqrt 5 / 3) :
  b = (8 * Real.sqrt 15) / 3 := by
  sorry

-- Problem 2
theorem magnitude_c_area (b_eq_4a : b = 4 * a) :
  c = 15 ∧ (1 / 2 * b * c * Real.sin A = (15 / 2) * Real.sqrt 15) := by
  sorry

end NUMINAMATH_GPT_magnitude_b_magnitude_c_area_l1687_168775


namespace NUMINAMATH_GPT_monotonic_decreasing_interval_l1687_168718

open Real

noncomputable def decreasing_interval (k: ℤ): Set ℝ :=
  {x | k * π - π / 3 < x ∧ x < k * π + π / 6 }

theorem monotonic_decreasing_interval (k : ℤ) :
  ∀ x, x ∈ decreasing_interval k ↔ (k * π - π / 3 < x ∧ x < k * π + π / 6) :=
by 
  intros x
  sorry

end NUMINAMATH_GPT_monotonic_decreasing_interval_l1687_168718


namespace NUMINAMATH_GPT_find_parameters_infinite_solutions_l1687_168785

def system_has_infinite_solutions (a b : ℝ) :=
  ∀ x y : ℝ, 2 * (a - b) * x + 6 * y = a ∧ 3 * b * x + (a - b) * b * y = 1

theorem find_parameters_infinite_solutions :
  ∀ (a b : ℝ), 
  system_has_infinite_solutions a b ↔ 
    (a = (3 + Real.sqrt 17) / 2 ∧ b = (Real.sqrt 17 - 3) / 2) ∨
    (a = (3 - Real.sqrt 17) / 2 ∧ b = (-3 - Real.sqrt 17) / 2) ∨
    (a = -2 ∧ b = 1) ∨
    (a = -1 ∧ b = 2) :=
sorry

end NUMINAMATH_GPT_find_parameters_infinite_solutions_l1687_168785


namespace NUMINAMATH_GPT_circle_equation_tangent_line1_tangent_line2_l1687_168738

-- Definitions of points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (2, 0)
def P : ℝ × ℝ := (2, 3)

-- Equation for the circle given the point constraints
def circle_eq : Prop := 
  ∀ x y : ℝ, ((x - 1)^2 + y^2 = 1) ↔ ((x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 0))

-- Equations for the tangent lines passing through point P and tangent to the circle
def tangent_eq1 : Prop := 
  P.1 = 2

def tangent_eq2 : Prop :=
  4 * P.1 - 3 * P.2 + 1 = 0

-- Statements to be proven
theorem circle_equation : circle_eq := 
  sorry 

theorem tangent_line1 : tangent_eq1 := 
  sorry 

theorem tangent_line2 : tangent_eq2 := 
  sorry 

end NUMINAMATH_GPT_circle_equation_tangent_line1_tangent_line2_l1687_168738


namespace NUMINAMATH_GPT_sum_of_number_and_reverse_divisible_by_11_l1687_168782

theorem sum_of_number_and_reverse_divisible_by_11 (A B : ℕ) (hA : 0 ≤ A) (hA9 : A ≤ 9) (hB : 0 ≤ B) (hB9 : B ≤ 9) :
  11 ∣ ((10 * A + B) + (10 * B + A)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_number_and_reverse_divisible_by_11_l1687_168782


namespace NUMINAMATH_GPT_correct_addition_result_l1687_168776

-- Definitions corresponding to the conditions
def mistaken_addend := 240
def correct_addend := 420
def incorrect_sum := 390

-- The proof statement
theorem correct_addition_result : 
  (incorrect_sum - mistaken_addend + correct_addend) = 570 :=
by
  sorry

end NUMINAMATH_GPT_correct_addition_result_l1687_168776


namespace NUMINAMATH_GPT_roll_probability_l1687_168791

noncomputable def probability_allison_rolls_greater : ℚ :=
  let p_brian := 5 / 6  -- Probability of Brian rolling 5 or lower
  let p_noah := 1       -- Probability of Noah rolling 5 or lower (since all faces roll 5 or lower)
  p_brian * p_noah

theorem roll_probability :
  probability_allison_rolls_greater = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_roll_probability_l1687_168791


namespace NUMINAMATH_GPT_least_clock_equivalent_l1687_168799

def clock_equivalent (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + 12 * k = b

theorem least_clock_equivalent (h : ℕ) (hh : h > 3) (hq : clock_equivalent h (h * h)) :
  h = 4 :=
by
  sorry

end NUMINAMATH_GPT_least_clock_equivalent_l1687_168799


namespace NUMINAMATH_GPT_find_20_paise_coins_l1687_168721

theorem find_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7100) : x = 200 :=
by
  -- Given the conditions, we need to prove x = 200.
  -- Steps and proofs are omitted here.
  sorry

end NUMINAMATH_GPT_find_20_paise_coins_l1687_168721


namespace NUMINAMATH_GPT_steve_speed_ratio_l1687_168768

variable (distance : ℝ)
variable (total_time : ℝ)
variable (speed_back : ℝ)
variable (speed_to : ℝ)

noncomputable def speed_ratio (distance : ℝ) (total_time : ℝ) (speed_back : ℝ) : ℝ := 
  let time_to := total_time - distance / speed_back
  let speed_to := distance / time_to
  speed_back / speed_to

theorem steve_speed_ratio (h1 : distance = 10) (h2 : total_time = 6) (h3 : speed_back = 5) :
  speed_ratio distance total_time speed_back = 2 := by
  sorry

end NUMINAMATH_GPT_steve_speed_ratio_l1687_168768


namespace NUMINAMATH_GPT_correct_option_l1687_168798

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg_real (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ > f x₂

theorem correct_option (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_decr : is_decreasing_on_nonneg_real f) :
  f 2 < f (-1) ∧ f (-1) < f 0 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1687_168798


namespace NUMINAMATH_GPT_pencils_ratio_l1687_168730

theorem pencils_ratio (T S Ti : ℕ) 
  (h1 : T = 6 * S)
  (h2 : T = 12)
  (h3 : Ti = 16) : Ti / S = 8 := by
  sorry

end NUMINAMATH_GPT_pencils_ratio_l1687_168730


namespace NUMINAMATH_GPT_no_real_roots_l1687_168788

theorem no_real_roots (k : ℝ) (h : k ≠ 0) : ¬∃ x : ℝ, x^2 + k * x + 3 * k^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_l1687_168788


namespace NUMINAMATH_GPT_point_on_coordinate_axes_l1687_168748

-- Definitions and assumptions from the problem conditions
variables {a b : ℝ}

-- The theorem statement asserts that point M(a, b) must be located on the coordinate axes given ab = 0
theorem point_on_coordinate_axes (h : a * b = 0) : 
  (a = 0) ∨ (b = 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_coordinate_axes_l1687_168748


namespace NUMINAMATH_GPT_complex_distance_l1687_168764

theorem complex_distance (i : Complex) (h : i = Complex.I) :
  Complex.abs (3 / (2 - i)^2) = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_complex_distance_l1687_168764


namespace NUMINAMATH_GPT_length_of_common_chord_l1687_168700

theorem length_of_common_chord (x y : ℝ) :
  (x + 1)^2 + (y - 3)^2 = 9 ∧ x^2 + y^2 - 4 * x + 2 * y - 11 = 0 → 
  ∃ l : ℝ, l = 24 / 5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_common_chord_l1687_168700


namespace NUMINAMATH_GPT_max_product_l1687_168794

-- Problem statement: Define the conditions and the conclusion
theorem max_product (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 4) : mn ≤ 4 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_max_product_l1687_168794


namespace NUMINAMATH_GPT_daily_construction_areas_minimum_area_A_must_build_l1687_168797

-- Definitions based on conditions and questions
variable {area : ℕ}
variable {daily_A : ℕ}
variable {daily_B : ℕ}
variable (h_area : area = 5100)
variable (h_A_B_diff : daily_A = daily_B + 2)
variable (h_A_days : 900 / daily_A = 720 / daily_B)

-- Proof statements for the questions in the problem
theorem daily_construction_areas (daily_B : ℕ) (daily_A : ℕ) :
  daily_B = 8 ∧ daily_A = 10 :=
by sorry

theorem minimum_area_A_must_build (daily_A : ℕ) (daily_B : ℕ) (area_A : ℕ) :
  (area_A ≥ 2 * (5100 - area_A)) → (area_A ≥ 3400) :=
by sorry

end NUMINAMATH_GPT_daily_construction_areas_minimum_area_A_must_build_l1687_168797


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1687_168710

theorem perfect_square_trinomial (a b m : ℝ) :
  (∃ x : ℝ, a^2 + mab + b^2 = (x + b)^2 ∨ a^2 + mab + b^2 = (x - b)^2) ↔ (m = 2 ∨ m = -2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1687_168710


namespace NUMINAMATH_GPT_operation_value_l1687_168753

def operation (a b : ℤ) : ℤ := 3 * a - 3 * b + 4

theorem operation_value : operation 6 8 = -2 := by
  sorry

end NUMINAMATH_GPT_operation_value_l1687_168753


namespace NUMINAMATH_GPT_cosine_difference_formula_l1687_168747

theorem cosine_difference_formula
  (α : ℝ)
  (h1 : 0 < α)
  (h2 : α < (Real.pi / 2))
  (h3 : Real.tan α = 2) :
  Real.cos (α - (Real.pi / 4)) = (3 * Real.sqrt 10) / 10 := 
by
  sorry

end NUMINAMATH_GPT_cosine_difference_formula_l1687_168747


namespace NUMINAMATH_GPT_negation_proof_l1687_168703

theorem negation_proof : ¬ (∃ x : ℝ, (x ≤ -1) ∨ (x ≥ 2)) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 := 
by 
  -- proof skipped
  sorry

end NUMINAMATH_GPT_negation_proof_l1687_168703


namespace NUMINAMATH_GPT_correct_reflection_l1687_168761

section
variable {Point : Type}
variables (PQ : Point → Point → Prop) (shaded_figure : Point → Prop)
variables (A B C D E : Point → Prop)

-- Condition: The line segment PQ is the axis of reflection.
-- Condition: The shaded figure is positioned above the line PQ and touches it at two points.
-- Define the reflection operation (assuming definitions for points and reflections are given).

def reflected (fig : Point → Prop) (axis : Point → Point → Prop) : Point → Prop := sorry  -- Define properly

-- The correct answer: The reflected figure should match figure (A).
theorem correct_reflection :
  reflected shaded_figure PQ = A :=
sorry
end

end NUMINAMATH_GPT_correct_reflection_l1687_168761


namespace NUMINAMATH_GPT_max_value_fraction_l1687_168779

theorem max_value_fraction (x : ℝ) : x ≠ 0 → 1 / (x^4 + 4*x^2 + 2 + 8/x^2 + 16/x^4) ≤ 1 / 31 :=
by sorry

end NUMINAMATH_GPT_max_value_fraction_l1687_168779


namespace NUMINAMATH_GPT_units_digit_sum_factorials_500_l1687_168778

-- Define the unit digit computation function
def unit_digit (n : ℕ) : ℕ := n % 10

-- Define the factorial function
def fact : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * fact n

-- Define the sum of factorials from 1 to n
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ i => fact i)

-- Define the problem statement
theorem units_digit_sum_factorials_500 : unit_digit (sum_factorials 500) = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_sum_factorials_500_l1687_168778


namespace NUMINAMATH_GPT_translation_2_units_left_l1687_168743

-- Define the initial parabola
def parabola1 (x : ℝ) : ℝ := x^2 + 1

-- Define the translated parabola
def parabola2 (x : ℝ) : ℝ := x^2 + 4 * x + 5

-- State that parabola2 is obtained by translating parabola1
-- And prove that this translation is 2 units to the left
theorem translation_2_units_left :
  ∀ x : ℝ, parabola2 x = parabola1 (x + 2) := 
by
  sorry

end NUMINAMATH_GPT_translation_2_units_left_l1687_168743


namespace NUMINAMATH_GPT_perpendicular_line_equation_l1687_168772

theorem perpendicular_line_equation (x y : ℝ) :
  (2, -1) ∈ ({ p : ℝ × ℝ | p.1 * 2 + p.2 * 1 - 3 = 0 }) ∧ 
  (∀ p : ℝ × ℝ, (p.1 * 2 + p.2 * (-4) + 5 = 0) → (p.2 * 1 + p.1 * 2 = 0)) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_equation_l1687_168772


namespace NUMINAMATH_GPT_smallest_whole_number_divisible_by_8_leaves_remainder_1_l1687_168736

theorem smallest_whole_number_divisible_by_8_leaves_remainder_1 :
  ∃ (n : ℕ), n ≡ 1 [MOD 2] ∧ n ≡ 1 [MOD 3] ∧ n ≡ 1 [MOD 4] ∧ n ≡ 1 [MOD 5] ∧ n ≡ 1 [MOD 7] ∧ n % 8 = 0 ∧ n = 7141 :=
by
  sorry

end NUMINAMATH_GPT_smallest_whole_number_divisible_by_8_leaves_remainder_1_l1687_168736


namespace NUMINAMATH_GPT_solve_equation_l1687_168724

theorem solve_equation (x : ℝ) (h : (3 * x) / (x + 1) = 9 / (x + 1)) : x = 3 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1687_168724


namespace NUMINAMATH_GPT_arabella_dance_steps_l1687_168796

theorem arabella_dance_steps :
  exists T1 T2 T3 : ℕ,
    T1 = 30 ∧
    T3 = T1 + T2 ∧
    T1 + T2 + T3 = 90 ∧
    (T2 / T1 : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arabella_dance_steps_l1687_168796


namespace NUMINAMATH_GPT_min_transfers_to_uniform_cards_l1687_168715

theorem min_transfers_to_uniform_cards (n : ℕ) (h : n = 101) (s : Fin n) :
  ∃ k : ℕ, (∀ s1 s2 : Fin n → ℕ, 
    (∀ i, s1 i = i + 1) ∧ (∀ j, s2 j = 51) → -- Initial and final conditions
    k ≤ 42925) := 
sorry

end NUMINAMATH_GPT_min_transfers_to_uniform_cards_l1687_168715


namespace NUMINAMATH_GPT_vector_product_magnitude_l1687_168758

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

end NUMINAMATH_GPT_vector_product_magnitude_l1687_168758


namespace NUMINAMATH_GPT_hours_spent_gaming_l1687_168704

def total_hours_in_day : ℕ := 24

def sleeping_fraction : ℚ := 1/3

def studying_fraction : ℚ := 3/4

def gaming_fraction : ℚ := 1/4

theorem hours_spent_gaming :
  let sleeping_hours := total_hours_in_day * sleeping_fraction
  let remaining_hours_after_sleeping := total_hours_in_day - sleeping_hours
  let studying_hours := remaining_hours_after_sleeping * studying_fraction
  let remaining_hours_after_studying := remaining_hours_after_sleeping - studying_hours
  remaining_hours_after_studying * gaming_fraction = 1 :=
by
  sorry

end NUMINAMATH_GPT_hours_spent_gaming_l1687_168704


namespace NUMINAMATH_GPT_arithmetic_seq_a12_l1687_168767

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Prove that a_12 = 12 given the conditions
theorem arithmetic_seq_a12 :
  ∃ a₁, (arithmetic_seq a₁ 2 2 = -8) → (arithmetic_seq a₁ 2 12 = 12) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a12_l1687_168767


namespace NUMINAMATH_GPT_eventually_repeating_last_two_digits_l1687_168731

theorem eventually_repeating_last_two_digits (K : ℕ) : ∃ N : ℕ, ∃ t : ℕ, 
    (∃ s : ℕ, t = s * 77 + N) ∨ (∃ u : ℕ, t = u * 54 + N) ∧ (t % 100) / 10 = (t % 100) % 10 :=
sorry

end NUMINAMATH_GPT_eventually_repeating_last_two_digits_l1687_168731


namespace NUMINAMATH_GPT_time_to_pass_platform_l1687_168770

-- Definitions based on the conditions
def train_length : ℕ := 1500 -- (meters)
def tree_crossing_time : ℕ := 120 -- (seconds)
def platform_length : ℕ := 500 -- (meters)

-- Define the train's speed
def train_speed := train_length / tree_crossing_time

-- Define the total distance the train needs to cover to pass the platform
def total_distance := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : 
  total_distance / train_speed = 160 :=
by sorry

end NUMINAMATH_GPT_time_to_pass_platform_l1687_168770


namespace NUMINAMATH_GPT_reciprocal_problem_l1687_168762

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 5) : 150 * (x⁻¹) = 240 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_problem_l1687_168762


namespace NUMINAMATH_GPT_downstream_distance_l1687_168744

theorem downstream_distance
    (speed_still_water : ℝ)
    (current_rate : ℝ)
    (travel_time_minutes : ℝ)
    (h_still_water : speed_still_water = 20)
    (h_current_rate : current_rate = 4)
    (h_travel_time : travel_time_minutes = 24) :
    (speed_still_water + current_rate) * (travel_time_minutes / 60) = 9.6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_downstream_distance_l1687_168744


namespace NUMINAMATH_GPT_kai_ice_plate_division_l1687_168701

-- Define the "L"-shaped ice plate with given dimensions
structure LShapedIcePlate (a : ℕ) :=
(horiz_length : ℕ)
(vert_length : ℕ)
(horiz_eq_vert : horiz_length = a ∧ vert_length = a)

-- Define the correctness of dividing the L-shaped plate into four equal parts
def can_be_divided_into_four_equal_parts (a : ℕ) (piece : LShapedIcePlate a) : Prop :=
∃ cut_points_v1 cut_points_v2 cut_points_h1 cut_points_h2,
  -- The cut points for vertical and horizontal cuts to turn the large "L" shape into four smaller "L" shapes
  piece.horiz_length = cut_points_v1 + cut_points_v2 ∧
  piece.vert_length = cut_points_h1 + cut_points_h2 ∧
  cut_points_v1 = a / 2 ∧ cut_points_v2 = a - a / 2 ∧
  cut_points_h1 = a / 2 ∧ cut_points_h2 = a - a / 2

-- Prove the main theorem
theorem kai_ice_plate_division (a : ℕ) (h : a > 0) (plate : LShapedIcePlate a) : 
  can_be_divided_into_four_equal_parts a plate :=
sorry

end NUMINAMATH_GPT_kai_ice_plate_division_l1687_168701


namespace NUMINAMATH_GPT_find_k_value_l1687_168706

theorem find_k_value (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ∧
  (∀ x1 x2 : ℝ, (k * x1^2 + 4 * x1 + 4 = 0 ∧ k * x2^2 + 4 * x2 + 4 = 0) → x1 = x2) →
  (k = 0 ∨ k = 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_k_value_l1687_168706


namespace NUMINAMATH_GPT_guess_probability_greater_than_two_thirds_l1687_168725

theorem guess_probability_greater_than_two_thirds :
  (1335 : ℝ) / 2002 > 2 / 3 :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_guess_probability_greater_than_two_thirds_l1687_168725


namespace NUMINAMATH_GPT_question_eq_answer_l1687_168755

theorem question_eq_answer (w x y z k : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2520) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 24 :=
sorry

end NUMINAMATH_GPT_question_eq_answer_l1687_168755


namespace NUMINAMATH_GPT_unique_positive_integer_solution_l1687_168769

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 2652 := sorry

end NUMINAMATH_GPT_unique_positive_integer_solution_l1687_168769


namespace NUMINAMATH_GPT_problem_equivalent_l1687_168729

variable (f : ℝ → ℝ)

theorem problem_equivalent (h₁ : ∀ x, deriv f x = deriv (deriv f) x)
                            (h₂ : ∀ x, deriv (deriv f) x < f x) : 
                            f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := sorry

end NUMINAMATH_GPT_problem_equivalent_l1687_168729


namespace NUMINAMATH_GPT_y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l1687_168745

variable (x y : ℝ)

-- Condition: y is defined as a function of x
def y_def := y = 2 * x + 5

-- Theorem: y > 0 if and only if x > -5/2
theorem y_positive_if_and_only_if_x_greater_than_negative_five_over_two 
  (h : y_def x y) : y > 0 ↔ x > -5 / 2 := by sorry

end NUMINAMATH_GPT_y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l1687_168745


namespace NUMINAMATH_GPT_joe_paint_usage_l1687_168750

noncomputable def paint_used_after_four_weeks : ℝ := 
  let total_paint := 480
  let first_week_paint := (1/5) * total_paint
  let second_week_paint := (1/6) * (total_paint - first_week_paint)
  let third_week_paint := (1/7) * (total_paint - first_week_paint - second_week_paint)
  let fourth_week_paint := (2/9) * (total_paint - first_week_paint - second_week_paint - third_week_paint)
  first_week_paint + second_week_paint + third_week_paint + fourth_week_paint

theorem joe_paint_usage :
  abs (paint_used_after_four_weeks - 266.66) < 0.01 :=
sorry

end NUMINAMATH_GPT_joe_paint_usage_l1687_168750


namespace NUMINAMATH_GPT_circle_radius_l1687_168732

/-- Let a circle have a maximum distance of 11 cm and a minimum distance of 5 cm from a point P.
Prove that the radius of the circle can be either 3 cm or 8 cm. -/
theorem circle_radius (max_dist min_dist : ℕ) (h_max : max_dist = 11) (h_min : min_dist = 5) :
  (∃ r : ℕ, r = 3 ∨ r = 8) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l1687_168732


namespace NUMINAMATH_GPT_rectangle_area_from_square_area_and_proportions_l1687_168784

theorem rectangle_area_from_square_area_and_proportions :
  ∃ (a b w : ℕ), a = 16 ∧ b = 3 * w ∧ w = Int.natAbs (Int.sqrt a) ∧ w * b = 48 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_from_square_area_and_proportions_l1687_168784


namespace NUMINAMATH_GPT_rhombus_area_l1687_168749

theorem rhombus_area (d1 d2 : ℝ) (θ : ℝ) (h1 : d1 = 8) (h2 : d2 = 10) (h3 : Real.sin θ = 3 / 5) : 
  (1 / 2) * d1 * d2 * Real.sin θ = 24 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1687_168749


namespace NUMINAMATH_GPT_stickers_at_end_of_week_l1687_168766

theorem stickers_at_end_of_week (initial_stickers earned_stickers total_stickers : Nat) :
  initial_stickers = 39 →
  earned_stickers = 22 →
  total_stickers = initial_stickers + earned_stickers →
  total_stickers = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_stickers_at_end_of_week_l1687_168766


namespace NUMINAMATH_GPT_lcm_14_18_20_l1687_168757

theorem lcm_14_18_20 : Nat.lcm (Nat.lcm 14 18) 20 = 1260 :=
by
  -- Define the prime factorizations
  have fact_14 : 14 = 2 * 7 := by norm_num
  have fact_18 : 18 = 2 * 3^2 := by norm_num
  have fact_20 : 20 = 2^2 * 5 := by norm_num
  
  -- Calculate the LCM based on the highest powers of each prime
  have lcm : Nat.lcm (Nat.lcm 14 18) 20 = 2^2 * 3^2 * 5 * 7 :=
    by
      sorry -- Proof details are not required

  -- Final verification that this calculation matches 1260
  exact lcm

end NUMINAMATH_GPT_lcm_14_18_20_l1687_168757


namespace NUMINAMATH_GPT_investment_share_l1687_168759

variable (P_investment Q_investment : ℝ)

theorem investment_share (h1 : Q_investment = 60000) (h2 : P_investment / Q_investment = 2 / 3) : P_investment = 40000 := by
  sorry

end NUMINAMATH_GPT_investment_share_l1687_168759


namespace NUMINAMATH_GPT_base4_base9_digit_difference_l1687_168754

theorem base4_base9_digit_difference (n : ℕ) (h1 : n = 523) (h2 : ∀ (k : ℕ), 4^(k - 1) ≤ n -> n < 4^k -> k = 5)
  (h3 : ∀ (k : ℕ), 9^(k - 1) ≤ n -> n < 9^k -> k = 3) : (5 - 3 = 2) :=
by
  -- Let's provide our specific instantiations for h2 and h3
  have base4_digits := h2 5;
  have base9_digits := h3 3;
  -- Clear sorry
  rfl

end NUMINAMATH_GPT_base4_base9_digit_difference_l1687_168754


namespace NUMINAMATH_GPT_probability_of_two_accurate_forecasts_l1687_168713

noncomputable def event_A : Type := {forecast : ℕ | forecast = 1}

def prob_A : ℝ := 0.9
def prob_A' : ℝ := 1 - prob_A

-- Define that there are 3 independent trials
def num_forecasts : ℕ := 3

-- Given
def probability_two_accurate (x : ℕ) : ℝ :=
if x = 2 then 3 * (prob_A^2 * prob_A') else 0

-- Statement to be proved
theorem probability_of_two_accurate_forecasts : probability_two_accurate 2 = 0.243 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_probability_of_two_accurate_forecasts_l1687_168713


namespace NUMINAMATH_GPT_sequence_term_l1687_168708

theorem sequence_term (x : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, 2 / x n = 1 / x (n - 1) + 1 / x (n + 1))
  (h₁ : x 2 = 2 / 3)
  (h₂ : x 4 = 2 / 5) :
  x 10 = 2 / 11 := 
sorry

end NUMINAMATH_GPT_sequence_term_l1687_168708


namespace NUMINAMATH_GPT_surface_area_of_circumscribing_sphere_l1687_168714

theorem surface_area_of_circumscribing_sphere :
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  A = 17 * Real.pi :=
by
  let l := 2
  let h := 3
  let d := Real.sqrt (l^2 + l^2 + h^2)
  let r := d / 2
  let A := 4 * Real.pi * r^2
  show A = 17 * Real.pi
  sorry

end NUMINAMATH_GPT_surface_area_of_circumscribing_sphere_l1687_168714


namespace NUMINAMATH_GPT_empty_atm_l1687_168777

theorem empty_atm (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 9 < b 9)
    (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ 8 → a k ≠ b k) 
    (n : ℕ) (h₀ : n = 1) : 
    ∃ (sequence : ℕ → ℕ), (∀ i, sequence i ≤ n) → (∀ k, ∃ i, k > i → sequence k = 0) :=
sorry

end NUMINAMATH_GPT_empty_atm_l1687_168777


namespace NUMINAMATH_GPT_value_of_x_l1687_168795

theorem value_of_x (x : ℝ) : (3 * x + 5) / 7 = 13 → x = 86 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_x_l1687_168795


namespace NUMINAMATH_GPT_no_valid_prime_pairs_l1687_168751

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_valid_prime_pairs :
  ∀ x y : ℕ, is_prime x → is_prime y → y < x → x ≤ 200 → (x % y = 0) → ((x +1) % (y +1) = 0) → false :=
by
  sorry

end NUMINAMATH_GPT_no_valid_prime_pairs_l1687_168751


namespace NUMINAMATH_GPT_product_of_roots_l1687_168789

theorem product_of_roots (a b c : ℂ) (h_roots : 3 * (Polynomial.C a) * (Polynomial.C b) * (Polynomial.C c) = -7) :
  a * b * c = -7 / 3 :=
by sorry

end NUMINAMATH_GPT_product_of_roots_l1687_168789


namespace NUMINAMATH_GPT_jeff_current_cats_l1687_168728

def initial_cats : ℕ := 20
def monday_found_kittens : ℕ := 2 + 3
def monday_stray_cats : ℕ := 4
def tuesday_injured_cats : ℕ := 1
def tuesday_health_issues_cats : ℕ := 2
def tuesday_family_cats : ℕ := 3
def wednesday_adopted_cats : ℕ := 4 * 2
def wednesday_pregnant_cats : ℕ := 2
def thursday_adopted_cats : ℕ := 3
def thursday_donated_cats : ℕ := 3
def friday_adopted_cats : ℕ := 2
def friday_found_cats : ℕ := 3

theorem jeff_current_cats : 
  initial_cats 
  + monday_found_kittens + monday_stray_cats 
  + (tuesday_injured_cats + tuesday_health_issues_cats + tuesday_family_cats)
  + (wednesday_pregnant_cats - wednesday_adopted_cats)
  + (thursday_donated_cats - thursday_adopted_cats)
  + (friday_found_cats - friday_adopted_cats) 
  = 30 := by
  sorry

end NUMINAMATH_GPT_jeff_current_cats_l1687_168728


namespace NUMINAMATH_GPT_total_heads_of_cabbage_l1687_168774

-- Problem definition for the first patch
def first_patch : ℕ := 12 * 15

-- Problem definition for the second patch
def second_patch : ℕ := 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24

-- Problem statement
theorem total_heads_of_cabbage : first_patch + second_patch = 316 := by
  sorry

end NUMINAMATH_GPT_total_heads_of_cabbage_l1687_168774


namespace NUMINAMATH_GPT_infinite_series_sum_l1687_168737

theorem infinite_series_sum : 
  ∑' k : ℕ, (8 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1))) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1687_168737


namespace NUMINAMATH_GPT_product_of_two_consecutive_integers_sum_lt_150_l1687_168726

theorem product_of_two_consecutive_integers_sum_lt_150 :
  ∃ (n : Nat), n * (n + 1) = 5500 ∧ 2 * n + 1 < 150 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_consecutive_integers_sum_lt_150_l1687_168726


namespace NUMINAMATH_GPT_chloe_total_score_l1687_168793

def points_per_treasure : ℕ := 9
def treasures_first_level : ℕ := 6
def treasures_second_level : ℕ := 3

def score_first_level : ℕ := treasures_first_level * points_per_treasure
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := score_first_level + score_second_level

theorem chloe_total_score : total_score = 81 := by
  sorry

end NUMINAMATH_GPT_chloe_total_score_l1687_168793


namespace NUMINAMATH_GPT_percentage_students_50_59_is_10_71_l1687_168705

theorem percentage_students_50_59_is_10_71 :
  let n_90_100 := 3
  let n_80_89 := 6
  let n_70_79 := 8
  let n_60_69 := 4
  let n_50_59 := 3
  let n_below_50 := 4
  let total_students := n_90_100 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_below_50
  let fraction := (n_50_59 : ℚ) / total_students
  let percentage := (fraction * 100)
  percentage = 10.71 := by sorry

end NUMINAMATH_GPT_percentage_students_50_59_is_10_71_l1687_168705


namespace NUMINAMATH_GPT_max_min_values_f_decreasing_interval_f_l1687_168723

noncomputable def a : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := ((a.1 * (b x).1) + (a.2 * (b x).2)) + 2

theorem max_min_values_f (k : ℤ) :
  (∃ (x1 : ℝ), (x1 = 2 * k * Real.pi + Real.pi / 6) ∧ f x1 = 3) ∧
  (∃ (x2 : ℝ), (x2 = 2 * k * Real.pi - 5 * Real.pi / 6) ∧ f x2 = 1) := 
sorry

theorem decreasing_interval_f :
  ∀ x, (Real.pi / 6 ≤ x ∧ x ≤ 7 * Real.pi / 6) → (∀ y, f x ≥ f y → x ≤ y) := 
sorry

end NUMINAMATH_GPT_max_min_values_f_decreasing_interval_f_l1687_168723


namespace NUMINAMATH_GPT_ages_l1687_168787

-- Definitions of ages
variables (S M : ℕ) -- S: son's current age, M: mother's current age

-- Given conditions
def father_age : ℕ := 44
def son_father_relationship (S : ℕ) : Prop := father_age = S + S
def son_mother_relationship (S M : ℕ) : Prop := (S - 5) = (M - 10)

-- Theorem to prove the ages
theorem ages (S M : ℕ) (h1 : son_father_relationship S) (h2 : son_mother_relationship S M) :
  S = 22 ∧ M = 27 :=
by 
  sorry

end NUMINAMATH_GPT_ages_l1687_168787


namespace NUMINAMATH_GPT_difference_between_length_and_breadth_l1687_168716

theorem difference_between_length_and_breadth (L W : ℝ) (h1 : W = 1/2 * L) (h2 : L * W = 800) : L - W = 20 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_length_and_breadth_l1687_168716


namespace NUMINAMATH_GPT_right_triangle_point_selection_l1687_168711

theorem right_triangle_point_selection : 
  let n := 200 
  let rows := 2
  (rows * (n - 22 + 1)) + 2 * (rows * (n - 122 + 1)) + (n * (2 * (n - 1))) = 80268 := 
by 
  let rows := 2
  let n := 200
  let case1a := rows * (n - 22 + 1)
  let case1b := 2 * (rows * (n - 122 + 1))
  let case2 := n * (2 * (n - 1))
  have h : case1a + case1b + case2 = 80268 := by sorry
  exact h

end NUMINAMATH_GPT_right_triangle_point_selection_l1687_168711


namespace NUMINAMATH_GPT_smallest_sum_xy_l1687_168709

theorem smallest_sum_xy (x y : ℕ) (hx : x ≠ y) (h : 0 < x ∧ 0 < y) (hxy : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) :
  x + y = 64 :=
sorry

end NUMINAMATH_GPT_smallest_sum_xy_l1687_168709


namespace NUMINAMATH_GPT_rectangular_prism_parallel_edges_l1687_168760

theorem rectangular_prism_parallel_edges (length width height : ℕ) (h1 : length ≠ width) (h2 : width ≠ height) (h3 : length ≠ height) : 
  ∃ pairs : ℕ, pairs = 6 := by
  sorry

end NUMINAMATH_GPT_rectangular_prism_parallel_edges_l1687_168760


namespace NUMINAMATH_GPT_length_of_diagonal_AC_l1687_168765

-- Definitions based on the conditions
variable (AB BC CD DA AC : ℝ)
variable (angle_ADC : ℝ)

-- Conditions
def conditions : Prop :=
  AB = 12 ∧ BC = 12 ∧ CD = 15 ∧ DA = 15 ∧ angle_ADC = 120

theorem length_of_diagonal_AC (h : conditions AB BC CD DA angle_ADC) : AC = 15 :=
sorry

end NUMINAMATH_GPT_length_of_diagonal_AC_l1687_168765


namespace NUMINAMATH_GPT_determine_multiplier_l1687_168712

theorem determine_multiplier (x : ℝ) : 125 * x - 138 = 112 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_multiplier_l1687_168712


namespace NUMINAMATH_GPT_frosting_time_difference_l1687_168727

def normally_frost_time_per_cake := 5
def sprained_frost_time_per_cake := 8
def number_of_cakes := 10

theorem frosting_time_difference :
  (sprained_frost_time_per_cake * number_of_cakes) -
  (normally_frost_time_per_cake * number_of_cakes) = 30 :=
by
  sorry

end NUMINAMATH_GPT_frosting_time_difference_l1687_168727


namespace NUMINAMATH_GPT_determinant_of_given_matrix_l1687_168752

-- Define the given matrix
def given_matrix (z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![z + 2, z, z], ![z, z + 3, z], ![z, z, z + 4]]

-- Define the proof statement
theorem determinant_of_given_matrix (z : ℂ) : Matrix.det (given_matrix z) = 22 * z + 24 :=
by
  sorry

end NUMINAMATH_GPT_determinant_of_given_matrix_l1687_168752


namespace NUMINAMATH_GPT_inverse_proportion_quad_l1687_168790

theorem inverse_proportion_quad (k : ℝ) : (∀ x : ℝ, x > 0 → (k + 1) / x < 0) ∧ (∀ x : ℝ, x < 0 → (k + 1) / x > 0) ↔ k < -1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_quad_l1687_168790


namespace NUMINAMATH_GPT_range_of_a_l1687_168741

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem range_of_a {a : ℝ} :
  (∀ x > 1, f a x > 1) → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1687_168741


namespace NUMINAMATH_GPT_find_number_l1687_168763

theorem find_number (x : ℝ) : (0.75 * x = 0.45 * 1500 + 495) -> x = 1560 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1687_168763


namespace NUMINAMATH_GPT_sum_of_distinct_abc_eq_roots_l1687_168746

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 * ((x + 2*y)^2 - y^2 + x - 1)

-- Main theorem statement
theorem sum_of_distinct_abc_eq_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h1 : f a (b+c) = f b (c+a)) (h2 : f b (c+a) = f c (a+b)) :
  a + b + c = (1 + Real.sqrt 5) / 2 ∨ a + b + c = (1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_abc_eq_roots_l1687_168746


namespace NUMINAMATH_GPT_friend_gives_30_l1687_168756

noncomputable def total_earnings := 10 + 30 + 50 + 40 + 70

noncomputable def equal_share := total_earnings / 5

noncomputable def contribution_of_highest_earner := 70

noncomputable def amount_to_give := contribution_of_highest_earner - equal_share

theorem friend_gives_30 : amount_to_give = 30 := by
  sorry

end NUMINAMATH_GPT_friend_gives_30_l1687_168756


namespace NUMINAMATH_GPT_value_of_2a_minus_b_minus_4_l1687_168786

theorem value_of_2a_minus_b_minus_4 (a b : ℝ) (h : 2 * a - b = 2) : 2 * a - b - 4 = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_2a_minus_b_minus_4_l1687_168786


namespace NUMINAMATH_GPT_average_price_of_remaining_cans_l1687_168735

theorem average_price_of_remaining_cans (price_all price_returned : ℕ) (average_all average_returned : ℚ) 
    (h1 : price_all = 6) (h2 : average_all = 36.5) (h3 : price_returned = 2) (h4 : average_returned = 49.5) : 
    (price_all - price_returned) ≠ 0 → 
    4 * 30 = 6 * 36.5 - 2 * 49.5 :=
by
  intros hne
  sorry

end NUMINAMATH_GPT_average_price_of_remaining_cans_l1687_168735


namespace NUMINAMATH_GPT_smallest_gcd_bc_l1687_168722

theorem smallest_gcd_bc (a b c : ℕ) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) : Nat.gcd b c = 1 :=
sorry

end NUMINAMATH_GPT_smallest_gcd_bc_l1687_168722


namespace NUMINAMATH_GPT_acquaintances_at_ends_equal_l1687_168792

theorem acquaintances_at_ends_equal 
  (n : ℕ) -- number of participants
  (a b : ℕ → ℕ) -- functions which return the number of acquaintances before/after for each participant
  (h_ai_bi : ∀ (i : ℕ), 1 < i ∧ i < n → a i = b i) -- condition for participants except first and last
  (h_a1 : a 1 = 0) -- the first person has no one before them
  (h_bn : b n = 0) -- the last person has no one after them
  :
  a n = b 1 :=
by
  sorry

end NUMINAMATH_GPT_acquaintances_at_ends_equal_l1687_168792


namespace NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_l1687_168734

theorem gcd_of_polynomial_and_multiple (b : ℕ) (hb : 714 ∣ b) : 
  Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 102) b = 102 := by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomial_and_multiple_l1687_168734


namespace NUMINAMATH_GPT_train_route_l1687_168740

-- Definition of letter positions
def letter_position : Char → Nat
| 'A' => 1
| 'B' => 2
| 'K' => 11
| 'L' => 12
| 'U' => 21
| 'V' => 22
| _ => 0

-- Definition of decode function
def decode (s : List Nat) : String :=
match s with
| [21, 2, 12, 21] => "Baku"
| [21, 22, 12, 21] => "Ufa"
| _ => ""

-- Assert encoded strings
def departure_encoded : List Nat := [21, 2, 12, 21]
def arrival_encoded : List Nat := [21, 22, 12, 21]

-- Theorem statement
theorem train_route :
  decode departure_encoded = "Ufa" ∧ decode arrival_encoded = "Baku" :=
by
  sorry

end NUMINAMATH_GPT_train_route_l1687_168740


namespace NUMINAMATH_GPT_child_support_calculation_l1687_168742

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end NUMINAMATH_GPT_child_support_calculation_l1687_168742


namespace NUMINAMATH_GPT_max_distance_from_center_of_square_l1687_168773

theorem max_distance_from_center_of_square :
  let A := (0, 0)
  let B := (1, 0)
  let C := (1, 1)
  let D := (0, 1)
  let O := (0.5, 0.5)
  ∃ P : ℝ × ℝ, 
  (let u := dist P A
   let v := dist P B
   let w := dist P C
   u^2 + v^2 + w^2 = 2)
  → dist O P = (1 + 2 * Real.sqrt 2) / (3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_max_distance_from_center_of_square_l1687_168773


namespace NUMINAMATH_GPT_expr_B_not_simplified_using_difference_of_squares_l1687_168771

def expr_A (x y : ℝ) := (-x - y) * (-x + y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (y + x) * (x - y)
def expr_D (x y : ℝ) := (y - x) * (x + y)

theorem expr_B_not_simplified_using_difference_of_squares (x y : ℝ) :
  ∃ x y, ¬ ∃ a b, expr_B x y = a^2 - b^2 :=
sorry

end NUMINAMATH_GPT_expr_B_not_simplified_using_difference_of_squares_l1687_168771


namespace NUMINAMATH_GPT_jellybean_ratio_l1687_168702

theorem jellybean_ratio (gigi_je : ℕ) (rory_je : ℕ) (lorelai_je : ℕ) (h_gigi : gigi_je = 15) (h_rory : rory_je = gigi_je + 30) (h_lorelai : lorelai_je = 180) : lorelai_je / (rory_je + gigi_je) = 3 :=
by
  -- Introduce the given hypotheses
  rw [h_gigi, h_rory, h_lorelai]
  -- Simplify the expression
  sorry

end NUMINAMATH_GPT_jellybean_ratio_l1687_168702


namespace NUMINAMATH_GPT_compute_ratio_l1687_168707

theorem compute_ratio (x y z a : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x + y + z = a) (h5 : a ≠ 0) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 1 / 3 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_compute_ratio_l1687_168707


namespace NUMINAMATH_GPT_least_common_denominator_l1687_168719

-- Define the list of numbers
def numbers : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the least common multiple function
noncomputable def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Define the main theorem
theorem least_common_denominator : lcm_list numbers = 2520 := 
  by sorry

end NUMINAMATH_GPT_least_common_denominator_l1687_168719
