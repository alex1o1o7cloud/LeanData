import Mathlib

namespace NUMINAMATH_GPT_max_pieces_l485_48576

theorem max_pieces (plywood_width plywood_height piece_width piece_height : ℕ)
  (h_plywood : plywood_width = 22) (h_plywood_height : plywood_height = 15)
  (h_piece : piece_width = 3) (h_piece_height : piece_height = 5) :
  (plywood_width * plywood_height) / (piece_width * piece_height) = 22 := by
  sorry

end NUMINAMATH_GPT_max_pieces_l485_48576


namespace NUMINAMATH_GPT_base8_246_is_166_in_base10_l485_48504

def convert_base8_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10;
  let d1 := (n / 10) % 10;
  let d2 := (n / 100) % 10;
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

theorem base8_246_is_166_in_base10 : convert_base8_to_base10 246 = 166 :=
  sorry

end NUMINAMATH_GPT_base8_246_is_166_in_base10_l485_48504


namespace NUMINAMATH_GPT_costume_total_cost_l485_48565

variable (friends : ℕ) (cost_per_costume : ℕ) 

theorem costume_total_cost (h1 : friends = 8) (h2 : cost_per_costume = 5) : friends * cost_per_costume = 40 :=
by {
  sorry -- We omit the proof, as instructed.
}

end NUMINAMATH_GPT_costume_total_cost_l485_48565


namespace NUMINAMATH_GPT_sin_45_degrees_l485_48598

noncomputable def Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem sin_45_degrees : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end NUMINAMATH_GPT_sin_45_degrees_l485_48598


namespace NUMINAMATH_GPT_rectangle_R2_area_l485_48545

theorem rectangle_R2_area
  (side1_R1 : ℝ) (area_R1 : ℝ) (diag_R2 : ℝ)
  (h_side1_R1 : side1_R1 = 4)
  (h_area_R1 : area_R1 = 32)
  (h_diag_R2 : diag_R2 = 20) :
  ∃ (area_R2 : ℝ), area_R2 = 160 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_R2_area_l485_48545


namespace NUMINAMATH_GPT_math_problem_l485_48536

theorem math_problem (a : ℝ) (h : a^2 - 4 * a + 3 = 0) (h_ne : a ≠ 2 ∧ a ≠ 3 ∧ a ≠ -3) :
  (9 - 3 * a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -3 / 8 :=
sorry

end NUMINAMATH_GPT_math_problem_l485_48536


namespace NUMINAMATH_GPT_multiples_of_6_or_8_under_201_not_both_l485_48571

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_6_or_8_under_201_not_both_l485_48571


namespace NUMINAMATH_GPT_lavinias_son_older_than_daughter_l485_48587

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end NUMINAMATH_GPT_lavinias_son_older_than_daughter_l485_48587


namespace NUMINAMATH_GPT_triangle_area_correct_l485_48577

noncomputable def triangle_area_given_conditions (a b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = c + 4 ∧ b = c + 2 ∧ Real.cos A = -1/2 then
  1/2 * b * c * Real.sin A
  else 0

theorem triangle_area_correct :
  ∀ (a b c : ℝ), ∀ A : ℝ, a = c + 4 → b = c + 2 → Real.cos A = -1/2 → 
  triangle_area_given_conditions a b c A = 15 * Real.sqrt 3 / 4 :=
by
  intros a b c A ha hb hc
  simp [triangle_area_given_conditions, ha, hb, hc]
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l485_48577


namespace NUMINAMATH_GPT_smallest_n_terminating_decimal_l485_48551

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (∀ m : ℕ, (∃ k : ℕ, m + 150 = 2^k * (5^0))
  → (m ≥ n)) ∧ (n > 0) ∧ (∃ k : ℕ, n + 150 = 2^k * (5^0)) := 
sorry

end NUMINAMATH_GPT_smallest_n_terminating_decimal_l485_48551


namespace NUMINAMATH_GPT_two_bishops_placement_l485_48524

theorem two_bishops_placement :
  let squares := 64
  let white_squares := 32
  let black_squares := 32
  let first_bishop_white_positions := 32
  let second_bishop_black_positions := 32 - 8
  first_bishop_white_positions * second_bishop_black_positions = 768 := by
  sorry

end NUMINAMATH_GPT_two_bishops_placement_l485_48524


namespace NUMINAMATH_GPT_solution_set_abs_inequality_l485_48570

theorem solution_set_abs_inequality :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

end NUMINAMATH_GPT_solution_set_abs_inequality_l485_48570


namespace NUMINAMATH_GPT_symmetric_axis_parabola_l485_48560

theorem symmetric_axis_parabola (h k : ℝ) (x : ℝ) :
  (∀ x, y = (x - h)^2 + k) → h = 2 → (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_axis_parabola_l485_48560


namespace NUMINAMATH_GPT_johann_ate_ten_oranges_l485_48568

variable (x : ℕ)
variable (y : ℕ)

def johann_initial_oranges := 60

def johann_remaining_after_eating := johann_initial_oranges - x

def johann_remaining_after_theft := (johann_remaining_after_eating / 2)

def johann_remaining_after_return := johann_remaining_after_theft + 5

theorem johann_ate_ten_oranges (h : johann_remaining_after_return = 30) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_johann_ate_ten_oranges_l485_48568


namespace NUMINAMATH_GPT_balance_rearrangement_vowels_at_end_l485_48558

theorem balance_rearrangement_vowels_at_end : 
  let vowels := ['A', 'A', 'E'];
  let consonants := ['B', 'L', 'N', 'C'];
  (Nat.factorial 3 / Nat.factorial 2) * Nat.factorial 4 = 72 :=
by
  sorry

end NUMINAMATH_GPT_balance_rearrangement_vowels_at_end_l485_48558


namespace NUMINAMATH_GPT_surface_area_of_sphere_with_diameter_4_l485_48531

theorem surface_area_of_sphere_with_diameter_4 :
    let diameter := 4
    let radius := diameter / 2
    let surface_area := 4 * Real.pi * radius^2
    surface_area = 16 * Real.pi :=
by
  -- Sorry is used in place of the actual proof.
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_with_diameter_4_l485_48531


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l485_48585

theorem arithmetic_sequence_sum (a b : ℤ) (h1 : 10 - 3 = 7)
  (h2 : a = 10 + 7) (h3 : b = 24 + 7) : a + b = 48 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l485_48585


namespace NUMINAMATH_GPT_ratio_of_ages_l485_48595

theorem ratio_of_ages (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = S + 20) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l485_48595


namespace NUMINAMATH_GPT_find_student_ticket_price_l485_48523

variable (S : ℝ)
variable (student_tickets non_student_tickets total_tickets : ℕ)
variable (non_student_ticket_price total_revenue : ℝ)

theorem find_student_ticket_price 
  (h1 : student_tickets = 90)
  (h2 : non_student_tickets = 60)
  (h3 : total_tickets = student_tickets + non_student_tickets)
  (h4 : non_student_ticket_price = 8)
  (h5 : total_revenue = 930)
  (h6 : 90 * S + 60 * non_student_ticket_price = total_revenue) : 
  S = 5 := 
sorry

end NUMINAMATH_GPT_find_student_ticket_price_l485_48523


namespace NUMINAMATH_GPT_smallest_integer_in_correct_range_l485_48516

theorem smallest_integer_in_correct_range :
  ∃ (n : ℤ), n > 1 ∧ n % 3 = 1 ∧ n % 5 = 1 ∧ n % 8 = 1 ∧ n % 7 = 2 ∧ 161 ≤ n ∧ n ≤ 200 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_correct_range_l485_48516


namespace NUMINAMATH_GPT_alien_saturday_sequence_l485_48505

def a_1 : String := "A"
def a_2 : String := "AY"
def a_3 : String := "AYYA"
def a_4 : String := "AYYAYAAY"

noncomputable def a_5 : String := a_4 ++ "YAAYAYYA"
noncomputable def a_6 : String := a_5 ++ "YAAYAYYAAAYAYAAY"

theorem alien_saturday_sequence : 
  a_6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" :=
sorry

end NUMINAMATH_GPT_alien_saturday_sequence_l485_48505


namespace NUMINAMATH_GPT_expression_equivalence_l485_48507

def algebraicExpression : String := "5 - 4a"
def wordExpression : String := "the difference of 5 and 4 times a"

theorem expression_equivalence : algebraicExpression = wordExpression := 
sorry

end NUMINAMATH_GPT_expression_equivalence_l485_48507


namespace NUMINAMATH_GPT_initially_calculated_average_weight_l485_48543

theorem initially_calculated_average_weight (n : ℕ) (misread_diff correct_avg_weight : ℝ)
  (hn : n = 20) (hmisread_diff : misread_diff = 10) (hcorrect_avg_weight : correct_avg_weight = 58.9) :
  ((correct_avg_weight * n - misread_diff) / n) = 58.4 :=
by
  rw [hn, hmisread_diff, hcorrect_avg_weight]
  sorry

end NUMINAMATH_GPT_initially_calculated_average_weight_l485_48543


namespace NUMINAMATH_GPT_star_difference_l485_48578

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by
  sorry

end NUMINAMATH_GPT_star_difference_l485_48578


namespace NUMINAMATH_GPT_sequence_sum_l485_48527

theorem sequence_sum (x y : ℕ) 
  (r : ℚ) 
  (h1 : 4 * r = 1) 
  (h2 : x = 256 * r)
  (h3 : y = x * r): 
  x + y = 80 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_sum_l485_48527


namespace NUMINAMATH_GPT_sum_of_powers_of_4_l485_48553

theorem sum_of_powers_of_4 : 4^0 + 4^1 + 4^2 + 4^3 = 85 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_4_l485_48553


namespace NUMINAMATH_GPT_harvest_duration_l485_48540

theorem harvest_duration (total_earnings earnings_per_week : ℕ) (h1 : total_earnings = 1216) (h2 : earnings_per_week = 16) :
  total_earnings / earnings_per_week = 76 :=
by
  sorry

end NUMINAMATH_GPT_harvest_duration_l485_48540


namespace NUMINAMATH_GPT_expression_for_f_l485_48569

theorem expression_for_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 - x - 2) : ∀ x : ℤ, f x = x^2 - 3 * x := 
by
  sorry

end NUMINAMATH_GPT_expression_for_f_l485_48569


namespace NUMINAMATH_GPT_probability_of_drawing_diamond_or_ace_l485_48548

-- Define the number of diamonds
def numDiamonds : ℕ := 13

-- Define the number of other Aces
def numOtherAces : ℕ := 3

-- Define the total number of cards in the deck
def totalCards : ℕ := 52

-- Define the number of desirable outcomes (either diamonds or Aces)
def numDesirableOutcomes : ℕ := numDiamonds + numOtherAces

-- Define the probability of drawing a diamond or an Ace
def desiredProbability : ℚ := numDesirableOutcomes / totalCards

theorem probability_of_drawing_diamond_or_ace :
  desiredProbability = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_diamond_or_ace_l485_48548


namespace NUMINAMATH_GPT_quadratic_distinct_roots_example_l485_48591

theorem quadratic_distinct_roots_example {b c : ℝ} (hb : b = 1) (hc : c = 0) :
    (b^2 - 4 * c) > 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_example_l485_48591


namespace NUMINAMATH_GPT_shaded_area_proof_l485_48567

-- Given Definitions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width
def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Conditions
def grid_area : ℕ :=
  rectangle_area 2 3 + rectangle_area 3 4 + rectangle_area 4 5

def unshaded_triangle_area : ℕ := triangle_area 12 4

-- Question
def shaded_area : ℕ := grid_area - unshaded_triangle_area

-- Proof statement
theorem shaded_area_proof : shaded_area = 14 := by
  sorry

end NUMINAMATH_GPT_shaded_area_proof_l485_48567


namespace NUMINAMATH_GPT_measure_8_cm_measure_5_cm_1_measure_5_cm_2_l485_48582

theorem measure_8_cm:
  ∃ n : ℕ, n * (11 - 7) = 8 := by
  sorry

theorem measure_5_cm_1:
  ∃ x : ℕ, ∃ y : ℕ, x * ((11 - 7) * 2) - y * 7 = 5 := by
  sorry

theorem measure_5_cm_2:
  3 * 11 - 4 * 7 = 5 := by
  sorry

end NUMINAMATH_GPT_measure_8_cm_measure_5_cm_1_measure_5_cm_2_l485_48582


namespace NUMINAMATH_GPT_find_matrix_N_l485_48539

-- Define the given matrix equation
def condition (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N ^ 3 - 3 * N ^ 2 + 4 * N = ![![8, 16], ![4, 8]]

-- State the theorem
theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) (h : condition N) :
  N = ![![2, 4], ![1, 2]] :=
sorry

end NUMINAMATH_GPT_find_matrix_N_l485_48539


namespace NUMINAMATH_GPT_crickets_needed_to_reach_11_l485_48528

theorem crickets_needed_to_reach_11 (collected_crickets : ℕ) (wanted_crickets : ℕ) 
                                     (h : collected_crickets = 7) (h2 : wanted_crickets = 11) :
  wanted_crickets - collected_crickets = 4 :=
sorry

end NUMINAMATH_GPT_crickets_needed_to_reach_11_l485_48528


namespace NUMINAMATH_GPT_fraction_of_fraction_of_fraction_l485_48529

theorem fraction_of_fraction_of_fraction (a b c d : ℝ) (h₁ : a = 1/5) (h₂ : b = 1/3) (h₃ : c = 1/6) (h₄ : d = 90) :
  (a * b * c * d) = 1 :=
by
  rw [h₁, h₂, h₃, h₄]
  simp
  sorry -- To indicate that the proof is missing

end NUMINAMATH_GPT_fraction_of_fraction_of_fraction_l485_48529


namespace NUMINAMATH_GPT_find_incorrect_value_of_observation_l485_48500

noncomputable def incorrect_observation_value (mean1 : ℝ) (mean2 : ℝ) (n : ℕ) : ℝ :=
  let old_sum := mean1 * n
  let new_sum := mean2 * n
  let correct_value := 45
  let incorrect_value := (old_sum - new_sum + correct_value)
  (incorrect_value / -1)

theorem find_incorrect_value_of_observation :
  incorrect_observation_value 36 36.5 50 = 20 :=
by
  -- By the problem setup, incorrect_observation_value 36 36.5 50 is as defined in the proof steps.
  -- As per the proof steps and calculation, incorrect_observation_value 36 36.5 50 should evaluate to 20.
  sorry

end NUMINAMATH_GPT_find_incorrect_value_of_observation_l485_48500


namespace NUMINAMATH_GPT_algebra_simplification_l485_48572

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end NUMINAMATH_GPT_algebra_simplification_l485_48572


namespace NUMINAMATH_GPT_smallest_total_students_l485_48502

theorem smallest_total_students (n : ℕ) (h1 : 25 * n % 100 = 0)
  (h2 : 10 * n % 4 = 0)
  (h3 : ∃ (y z : ℕ), y = 3 * z / 2 ∧ (y + z - n / 40 = n / 4)) :
  ∃ k : ℕ, n = 200 * k :=
by
  sorry

end NUMINAMATH_GPT_smallest_total_students_l485_48502


namespace NUMINAMATH_GPT_triangle_area_l485_48520

theorem triangle_area (a b : ℝ) (h1 : b = (24 / a)) (h2 : 3 * 4 + a * (12 / a) = 12) : b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l485_48520


namespace NUMINAMATH_GPT_ones_digit_8_power_32_l485_48518

theorem ones_digit_8_power_32 : (8^32) % 10 = 6 :=
by sorry

end NUMINAMATH_GPT_ones_digit_8_power_32_l485_48518


namespace NUMINAMATH_GPT_solve_for_q_l485_48547

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_q_l485_48547


namespace NUMINAMATH_GPT_students_calculation_l485_48574

variable (students_boys students_playing_soccer students_not_playing_soccer girls_not_playing_soccer : ℕ)
variable (percentage_boys_play_soccer : ℚ)

def students_not_playing_sum (students_boys_not_playing : ℕ) : ℕ :=
  students_boys_not_playing + girls_not_playing_soccer

def total_students (students_not_playing_sum students_playing_soccer : ℕ) : ℕ :=
  students_not_playing_sum + students_playing_soccer

theorem students_calculation 
  (H1 : students_boys = 312)
  (H2 : students_playing_soccer = 250)
  (H3 : percentage_boys_play_soccer = 0.86)
  (H4 : girls_not_playing_soccer = 73)
  (H5 : percentage_boys_play_soccer * students_playing_soccer = 215)
  (H6 : students_boys - 215 = 97)
  (H7 : students_not_playing_sum 97 = 170)
  (H8 : total_students 170 250 = 420) : ∃ total, total = 420 :=
by 
  existsi total_students 170 250
  exact H8

end NUMINAMATH_GPT_students_calculation_l485_48574


namespace NUMINAMATH_GPT_golden_section_point_l485_48525

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_section_point (AB AP PB : ℝ)
  (h1 : AP + PB = AB)
  (h2 : AB = 5)
  (h3 : (AB / AP) = (AP / PB))
  (h4 : AP > PB) :
  AP = (5 * Real.sqrt 5 - 5) / 2 :=
by sorry

end NUMINAMATH_GPT_golden_section_point_l485_48525


namespace NUMINAMATH_GPT_arithmetic_mean_is_one_l485_48521

theorem arithmetic_mean_is_one (x a : ℝ) (hx : x ≠ 0) (hx2a : x^2 ≠ a) :
  (1 / 2 * ((x^2 + a) / x^2 + (x^2 - a) / x^2) = 1) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_one_l485_48521


namespace NUMINAMATH_GPT_mini_bottles_needed_to_fill_jumbo_l485_48566

def mini_bottle_capacity : ℕ := 45
def jumbo_bottle_capacity : ℕ := 600

-- The problem statement expressed as a Lean theorem.
theorem mini_bottles_needed_to_fill_jumbo :
  (jumbo_bottle_capacity + mini_bottle_capacity - 1) / mini_bottle_capacity = 14 :=
by
  sorry

end NUMINAMATH_GPT_mini_bottles_needed_to_fill_jumbo_l485_48566


namespace NUMINAMATH_GPT_percentage_saving_l485_48599

theorem percentage_saving 
  (p_coat p_pants : ℝ)
  (d_coat d_pants : ℝ)
  (h_coat : p_coat = 100)
  (h_pants : p_pants = 50)
  (h_d_coat : d_coat = 0.30)
  (h_d_pants : d_pants = 0.40) :
  (p_coat * d_coat + p_pants * d_pants) / (p_coat + p_pants) = 0.333 :=
by
  sorry

end NUMINAMATH_GPT_percentage_saving_l485_48599


namespace NUMINAMATH_GPT_lines_parallel_or_coincident_l485_48557

/-- Given lines l₁ and l₂ with certain properties,
    prove that they are either parallel or coincident. -/
theorem lines_parallel_or_coincident
  (P Q : ℝ × ℝ)
  (hP : P = (-2, -1))
  (hQ : Q = (3, -6))
  (h_slope1 : ∀ θ, θ = 135 → Real.tan (θ * (Real.pi / 180)) = -1)
  (h_slope2 : (Q.2 - P.2) / (Q.1 - P.1) = -1) : 
  true :=
by sorry

end NUMINAMATH_GPT_lines_parallel_or_coincident_l485_48557


namespace NUMINAMATH_GPT_abscissa_midpoint_range_l485_48534

-- Definitions based on the given conditions.
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 6
def on_circle (x y : ℝ) : Prop := circle_eq x y
def chord_length (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 2)^2
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0
def on_line (x y : ℝ) : Prop := line_eq x y
def segment_length (P Q : ℝ × ℝ) : Prop := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4
def acute_angle (P Q G : ℝ × ℝ) : Prop := -- definition of acute angle condition
  sorry -- placeholder for the actual definition

-- The proof statement.
theorem abscissa_midpoint_range {A B P Q G M : ℝ × ℝ}
  (h_A_on_circle : on_circle A.1 A.2)
  (h_B_on_circle : on_circle B.1 B.2)
  (h_AB_length : chord_length A B)
  (h_P_on_line : on_line P.1 P.2)
  (h_Q_on_line : on_line Q.1 Q.2)
  (h_PQ_length : segment_length P Q)
  (h_angle_acute : acute_angle P Q G)
  (h_G_mid : G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (h_M_mid : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1 < 0) ∨ (M.1 > 3) :=
sorry

end NUMINAMATH_GPT_abscissa_midpoint_range_l485_48534


namespace NUMINAMATH_GPT_unique_solution_for_equation_l485_48580

theorem unique_solution_for_equation (a b c d : ℝ) 
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h : ∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) :
  a = 2 ^ (1 / 672) ∧ b = -2 * 2 ^ (1 / 672) ∧ c = -4 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_for_equation_l485_48580


namespace NUMINAMATH_GPT_area_difference_l485_48581

-- Define the original and new rectangle dimensions
def original_rect_area (length width : ℕ) : ℕ := length * width
def new_rect_area (length width : ℕ) : ℕ := (length - 2) * (width + 2)

-- Define the problem statement
theorem area_difference (a : ℕ) : new_rect_area a 5 - original_rect_area a 5 = 2 * a - 14 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_area_difference_l485_48581


namespace NUMINAMATH_GPT_strategy_for_antonio_l485_48538

-- We define the concept of 'winning' and 'losing' positions
def winning_position (m n : ℕ) : Prop :=
  ¬ (m % 2 = 0 ∧ n % 2 = 0)

-- Now create the main theorem
theorem strategy_for_antonio (m n : ℕ) : winning_position m n ↔ 
  (¬(m % 2 = 0 ∧ n % 2 = 0)) :=
by
  unfold winning_position
  sorry

end NUMINAMATH_GPT_strategy_for_antonio_l485_48538


namespace NUMINAMATH_GPT_emily_original_salary_l485_48596

def original_salary_emily (num_employees : ℕ) (original_employee_salary new_employee_salary new_salary_emily : ℕ) : ℕ :=
  new_salary_emily + (new_employee_salary - original_employee_salary) * num_employees

theorem emily_original_salary :
  original_salary_emily 10 20000 35000 850000 = 1000000 :=
by
  sorry

end NUMINAMATH_GPT_emily_original_salary_l485_48596


namespace NUMINAMATH_GPT_product_two_digit_numbers_l485_48508

theorem product_two_digit_numbers (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 777) : (a = 21 ∧ b = 37) ∨ (a = 37 ∧ b = 21) := 
  sorry

end NUMINAMATH_GPT_product_two_digit_numbers_l485_48508


namespace NUMINAMATH_GPT_shaded_area_fraction_l485_48544

theorem shaded_area_fraction (ABCD_area : ℝ) (shaded_square1_area : ℝ) (shaded_rectangle_area : ℝ) (shaded_square2_area : ℝ) (total_shaded_area : ℝ)
  (h_ABCD : ABCD_area = 36) 
  (h_shaded_square1 : shaded_square1_area = 4)
  (h_shaded_rectangle : shaded_rectangle_area = 12)
  (h_shaded_square2 : shaded_square2_area = 36)
  (h_total_shaded : total_shaded_area = 16) :
  (total_shaded_area / ABCD_area) = 4 / 9 :=
by 
  simp [h_ABCD, h_total_shaded]
  sorry

end NUMINAMATH_GPT_shaded_area_fraction_l485_48544


namespace NUMINAMATH_GPT_people_on_bus_before_stop_l485_48594

variable (P_before P_after P_got_on : ℕ)
variable (h1 : P_got_on = 13)
variable (h2 : P_after = 17)

theorem people_on_bus_before_stop : P_before = 4 :=
by
  -- Given that P_after = 17 and P_got_on = 13
  -- We need to prove P_before = P_after - P_got_on = 4
  sorry

end NUMINAMATH_GPT_people_on_bus_before_stop_l485_48594


namespace NUMINAMATH_GPT_log_sum_equality_l485_48588

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_log_sum_equality_l485_48588


namespace NUMINAMATH_GPT_parabola_vertex_l485_48522

-- Define the condition: the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 4 * y + 3 * x + 1 = 0

-- Define the statement: prove that the vertex of the parabola is (1, -2)
theorem parabola_vertex :
  parabola_equation 1 (-2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l485_48522


namespace NUMINAMATH_GPT_estate_value_l485_48542

theorem estate_value (E : ℝ) (x : ℝ) (y: ℝ) (z: ℝ) 
  (h1 : 9 * x = 3 / 4 * E) 
  (h2 : z = 8 * x) 
  (h3 : y = 600) 
  (h4 : E = z + 9 * x + y):
  E = 1440 := 
sorry

end NUMINAMATH_GPT_estate_value_l485_48542


namespace NUMINAMATH_GPT_pyramid_edges_sum_l485_48597

noncomputable def sum_of_pyramid_edges (s : ℝ) (h : ℝ) : ℝ :=
  let diagonal := s * Real.sqrt 2
  let half_diagonal := diagonal / 2
  let slant_height := Real.sqrt (half_diagonal^2 + h^2)
  4 * s + 4 * slant_height

theorem pyramid_edges_sum
  (s : ℝ) (h : ℝ)
  (hs : s = 15)
  (hh : h = 15) :
  sum_of_pyramid_edges s h = 135 :=
sorry

end NUMINAMATH_GPT_pyramid_edges_sum_l485_48597


namespace NUMINAMATH_GPT_color_of_182nd_marble_l485_48515

-- conditions
def pattern_length : ℕ := 15
def blue_length : ℕ := 6
def red_length : ℕ := 5
def green_length : ℕ := 4

def marble_color (n : ℕ) : String :=
  let cycle_pos := n % pattern_length
  if cycle_pos < blue_length then
    "blue"
  else if cycle_pos < blue_length + red_length then
    "red"
  else
    "green"

theorem color_of_182nd_marble : marble_color 182 = "blue" :=
by
  sorry

end NUMINAMATH_GPT_color_of_182nd_marble_l485_48515


namespace NUMINAMATH_GPT_multiplicative_inverse_l485_48573

def A : ℕ := 123456
def B : ℕ := 171428
def mod_val : ℕ := 1000000
def sum_A_B : ℕ := A + B
def N : ℕ := 863347

theorem multiplicative_inverse : (sum_A_B * N) % mod_val = 1 :=
by
  -- diverting proof with sorry since proof steps aren't the focus
  sorry

end NUMINAMATH_GPT_multiplicative_inverse_l485_48573


namespace NUMINAMATH_GPT_point_C_values_l485_48510

variable (B C : ℝ)
variable (distance_BC : ℝ)
variable (hB : B = 3)
variable (hDistance : distance_BC = 2)

theorem point_C_values (hBC : abs (C - B) = distance_BC) : (C = 1 ∨ C = 5) := 
by
  sorry

end NUMINAMATH_GPT_point_C_values_l485_48510


namespace NUMINAMATH_GPT_cost_of_pencil_and_pen_l485_48513

variable (p q : ℝ)

axiom condition1 : 4 * p + 3 * q = 4.20
axiom condition2 : 3 * p + 4 * q = 4.55

theorem cost_of_pencil_and_pen : p + q = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pencil_and_pen_l485_48513


namespace NUMINAMATH_GPT_expand_product_l485_48532

theorem expand_product (x : ℝ) :
  (x + 4) * (x - 5) = x^2 - x - 20 :=
by
  -- The proof will use algebraic identities and simplifications.
  sorry

end NUMINAMATH_GPT_expand_product_l485_48532


namespace NUMINAMATH_GPT_f_eq_four_or_seven_l485_48535

noncomputable def f (a b : ℕ) : ℚ := (a^2 + a * b + b^2) / (a * b - 1)

theorem f_eq_four_or_seven (a b : ℕ) (h : a > 0) (h1 : b > 0) (h2 : a * b ≠ 1) : 
  f a b = 4 ∨ f a b = 7 := 
sorry

end NUMINAMATH_GPT_f_eq_four_or_seven_l485_48535


namespace NUMINAMATH_GPT_find_AE_l485_48509

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end NUMINAMATH_GPT_find_AE_l485_48509


namespace NUMINAMATH_GPT_percent_of_total_is_correct_l485_48589

theorem percent_of_total_is_correct :
  (6.620000000000001 / 100 * 1000 = 66.2) :=
by
  sorry

end NUMINAMATH_GPT_percent_of_total_is_correct_l485_48589


namespace NUMINAMATH_GPT_average_annual_growth_rate_in_2014_and_2015_l485_48554

noncomputable def average_annual_growth_rate (p2013 p2015 : ℝ) (x : ℝ) : Prop :=
  p2013 * (1 + x)^2 = p2015

theorem average_annual_growth_rate_in_2014_and_2015 :
  average_annual_growth_rate 6.4 10 0.25 :=
by
  unfold average_annual_growth_rate
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_in_2014_and_2015_l485_48554


namespace NUMINAMATH_GPT_find_c_of_binomial_square_l485_48584

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end NUMINAMATH_GPT_find_c_of_binomial_square_l485_48584


namespace NUMINAMATH_GPT_not_all_zero_iff_at_least_one_non_zero_l485_48526

theorem not_all_zero_iff_at_least_one_non_zero (a b c : ℝ) : ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_not_all_zero_iff_at_least_one_non_zero_l485_48526


namespace NUMINAMATH_GPT_num_points_common_to_graphs_l485_48503

theorem num_points_common_to_graphs :
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  ∀ (x y : ℝ), ((2 * x - y + 3 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + y - 3 = 0 ∨ 3 * x - 4 * y + 8 = 0)) →
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 :=
sorry

end NUMINAMATH_GPT_num_points_common_to_graphs_l485_48503


namespace NUMINAMATH_GPT_find_A_coordinates_l485_48552

-- Given conditions
variable (B : (ℝ × ℝ)) (hB1 : B = (1, 2))

-- Definitions to translate problem conditions into Lean
def symmetric_y (P B : ℝ × ℝ) : Prop :=
  P.1 = -B.1 ∧ P.2 = B.2

def symmetric_x (A P : ℝ × ℝ) : Prop :=
  A.1 = P.1 ∧ A.2 = -P.2

-- Theorem statement
theorem find_A_coordinates (A P B : ℝ × ℝ) (hB1 : B = (1, 2))
    (h_symm_y: symmetric_y P B) (h_symm_x: symmetric_x A P) : 
    A = (-1, -2) :=
by
  sorry

end NUMINAMATH_GPT_find_A_coordinates_l485_48552


namespace NUMINAMATH_GPT_zoo_animal_difference_l485_48506

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  monkeys - zebras = 35 := by
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := elephants - 3
  show monkeys - zebras = 35
  sorry

end NUMINAMATH_GPT_zoo_animal_difference_l485_48506


namespace NUMINAMATH_GPT_fraction_equality_l485_48564

variable (a_n b_n : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Conditions
axiom S_T_ratio (n : ℕ) : T_n n ≠ 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)
axiom Sn_def (n : ℕ) : S_n n = n / 2 * (2 * a_n 0 + (n - 1) * (a_n 1 - a_n 0))
axiom Tn_def (n : ℕ) : T_n n = n / 2 * (2 * b_n 0 + (n - 1) * (b_n 1 - b_n 0))
axiom an_def (n : ℕ) : a_n n = a_n 0 + n * (a_n 1 - a_n 0)
axiom bn_def (n : ℕ) : b_n n = b_n 0 + n * (b_n 1 - b_n 0)

-- Proof statement
theorem fraction_equality :
  (b_n 3 + b_n 18) ≠ 0 → (b_n 6 + b_n 15) ≠ 0 →
  (a_n 10 / (b_n 3 + b_n 18) + a_n 11 / (b_n 6 + b_n 15)) = (41 / 78) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l485_48564


namespace NUMINAMATH_GPT_min_value_l485_48546

theorem min_value (x : ℝ) (h : 0 < x) : x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 :=
sorry

end NUMINAMATH_GPT_min_value_l485_48546


namespace NUMINAMATH_GPT_runway_show_total_time_l485_48501

-- Define the conditions
def time_per_trip : Nat := 2
def num_models : Nat := 6
def trips_bathing_suits_per_model : Nat := 2
def trips_evening_wear_per_model : Nat := 3
def trips_per_model : Nat := trips_bathing_suits_per_model + trips_evening_wear_per_model
def total_trips : Nat := trips_per_model * num_models

-- State the theorem
theorem runway_show_total_time : total_trips * time_per_trip = 60 := by
  -- fill in the proof here
  sorry

end NUMINAMATH_GPT_runway_show_total_time_l485_48501


namespace NUMINAMATH_GPT_officeEmployees_l485_48590

noncomputable def totalEmployees 
  (averageSalaryAll : ℝ) 
  (averageSalaryOfficers : ℝ) 
  (averageSalaryManagers : ℝ) 
  (averageSalaryWorkers : ℝ) 
  (numOfficers : ℕ) 
  (numManagers : ℕ) 
  (numWorkers : ℕ) : ℕ := 
  if (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
      = (numOfficers + numManagers + numWorkers) * averageSalaryAll 
  then numOfficers + numManagers + numWorkers 
  else 0

theorem officeEmployees
  (averageSalaryAll : ℝ)
  (averageSalaryOfficers : ℝ)
  (averageSalaryManagers : ℝ)
  (averageSalaryWorkers : ℝ)
  (numOfficers : ℕ)
  (numManagers : ℕ)
  (numWorkers : ℕ) :
  averageSalaryAll = 720 →
  averageSalaryOfficers = 1320 →
  averageSalaryManagers = 840 →
  averageSalaryWorkers = 600 →
  numOfficers = 10 →
  numManagers = 20 →
  (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
    = (numOfficers + numManagers + numWorkers) * averageSalaryAll →
  totalEmployees averageSalaryAll averageSalaryOfficers averageSalaryManagers averageSalaryWorkers numOfficers numManagers numWorkers = 100 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6] at h7
  rw [h1]
  simp [totalEmployees, h7]
  sorry

end NUMINAMATH_GPT_officeEmployees_l485_48590


namespace NUMINAMATH_GPT_part1_part2_l485_48517
open Real

-- Part 1
theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  0 < (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ∧
  (sqrt (1 + x) + sqrt (1 - x) + 2) * (sqrt (1 - x^2) + 1) ≤ 8 := 
sorry

-- Part 2
theorem part2 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) :
  ∃ β > 0, β = 4 ∧ sqrt (1 + x) + sqrt (1 - x) ≤ 2 - x^2 / β :=
sorry

end NUMINAMATH_GPT_part1_part2_l485_48517


namespace NUMINAMATH_GPT_gcd_of_45_and_75_l485_48586

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_of_45_and_75_l485_48586


namespace NUMINAMATH_GPT_inversely_proportional_ratio_l485_48541

theorem inversely_proportional_ratio (x y x1 x2 y1 y2 : ℝ) 
  (h_inv_prop : x * y = x1 * y2) 
  (h_ratio : x1 / x2 = 3 / 5) 
  (x1_nonzero : x1 ≠ 0) 
  (x2_nonzero : x2 ≠ 0) 
  (y1_nonzero : y1 ≠ 0) 
  (y2_nonzero : y2 ≠ 0) : 
  y1 / y2 = 5 / 3 := 
sorry

end NUMINAMATH_GPT_inversely_proportional_ratio_l485_48541


namespace NUMINAMATH_GPT_simplify_cube_root_l485_48562

theorem simplify_cube_root (a : ℝ) (h : 0 ≤ a) : (a * a^(1/2))^(1/3) = a^(1/2) :=
sorry

end NUMINAMATH_GPT_simplify_cube_root_l485_48562


namespace NUMINAMATH_GPT_coin_flip_probability_l485_48555

theorem coin_flip_probability (p : ℝ) 
  (h : p^2 + (1 - p)^2 = 4 * p * (1 - p)) : 
  p = (3 + Real.sqrt 3) / 6 :=
sorry

end NUMINAMATH_GPT_coin_flip_probability_l485_48555


namespace NUMINAMATH_GPT_ratio_of_constants_l485_48514

theorem ratio_of_constants (a b c: ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : c = b / a) : c = 1 / 16 :=
by sorry

end NUMINAMATH_GPT_ratio_of_constants_l485_48514


namespace NUMINAMATH_GPT_find_a_l485_48511

-- Definitions of the conditions
variables {a b c : ℤ} 

-- Theorem statement
theorem find_a (h1: a + b = c) (h2: b + c = 7) (h3: c = 4) : a = 1 :=
by
  -- Using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_find_a_l485_48511


namespace NUMINAMATH_GPT_hex_B1C_base10_l485_48561

theorem hex_B1C_base10 : (11 * 16^2 + 1 * 16^1 + 12 * 16^0) = 2844 :=
by
  sorry

end NUMINAMATH_GPT_hex_B1C_base10_l485_48561


namespace NUMINAMATH_GPT_last_integer_in_geometric_sequence_l485_48559

theorem last_integer_in_geometric_sequence (a : ℕ) (r : ℚ) (h_a : a = 2048000) (h_r : r = 1/2) : 
  ∃ n : ℕ, (a : ℚ) * (r^n : ℚ) = 125 := 
by
  sorry

end NUMINAMATH_GPT_last_integer_in_geometric_sequence_l485_48559


namespace NUMINAMATH_GPT_zero_intersections_l485_48556

noncomputable def Line : Type := sorry  -- Define Line as a type
noncomputable def is_skew (a b : Line) : Prop := sorry  -- Predicate for skew lines
noncomputable def is_common_perpendicular (EF a b : Line) : Prop := sorry  -- Predicate for common perpendicular
noncomputable def is_parallel (l EF : Line) : Prop := sorry  -- Predicate for parallel lines
noncomputable def count_intersections (l a b : Line) : ℕ := sorry  -- Function to count intersections

theorem zero_intersections (EF a b l : Line) 
  (h_skew : is_skew a b) 
  (h_common_perpendicular : is_common_perpendicular EF a b)
  (h_parallel : is_parallel l EF) : 
  count_intersections l a b = 0 := 
sorry

end NUMINAMATH_GPT_zero_intersections_l485_48556


namespace NUMINAMATH_GPT_lioness_hyena_age_ratio_l485_48575

variables {k H : ℕ}

-- Conditions
def lioness_age (lioness_age hyena_age : ℕ) : Prop := ∃ k, lioness_age = k * hyena_age
def lioness_is_12 (lioness_age : ℕ) : Prop := lioness_age = 12
def baby_age (mother_age baby_age : ℕ) : Prop := baby_age = mother_age / 2
def baby_ages_sum_in_5_years (baby_l_age baby_h_age sum : ℕ) : Prop := 
  (baby_l_age + 5) + (baby_h_age + 5) = sum

-- The statement to be proved
theorem lioness_hyena_age_ratio (H : ℕ)
  (h1 : lioness_age 12 H) 
  (h2 : baby_age 12 6) 
  (h3 : baby_age H (H / 2)) 
  (h4 : baby_ages_sum_in_5_years 6 (H / 2) 19) : 12 / H = 2 := 
sorry

end NUMINAMATH_GPT_lioness_hyena_age_ratio_l485_48575


namespace NUMINAMATH_GPT_totalTrianglesInFigure_l485_48549

-- Definition of the problem involving a rectangle with subdivisions creating triangles
def numberOfTrianglesInRectangle : Nat :=
  let smallestTriangles := 24   -- Number of smallest triangles
  let nextSizeTriangles1 := 8   -- Triangles formed by combining smallest triangles
  let nextSizeTriangles2 := 12
  let nextSizeTriangles3 := 16
  let largestTriangles := 4
  smallestTriangles + nextSizeTriangles1 + nextSizeTriangles2 + nextSizeTriangles3 + largestTriangles

-- The Lean 4 theorem statement, stating that the total number of triangles equals 64
theorem totalTrianglesInFigure : numberOfTrianglesInRectangle = 64 := 
by
  sorry

end NUMINAMATH_GPT_totalTrianglesInFigure_l485_48549


namespace NUMINAMATH_GPT_probability_point_between_lines_l485_48533

theorem probability_point_between_lines {x y : ℝ} :
  (∀ x, y = -2 * x + 8) →
  (∀ x, y = -3 * x + 8) →
  0.33 = 0.33 :=
by
  intro hl hm
  sorry

end NUMINAMATH_GPT_probability_point_between_lines_l485_48533


namespace NUMINAMATH_GPT_total_canoes_built_l485_48519

def geometric_sum (a r n : ℕ) : ℕ :=
  a * ((r^n - 1) / (r - 1))

theorem total_canoes_built : geometric_sum 10 3 7 = 10930 := 
  by
    -- The proof will go here.
    sorry

end NUMINAMATH_GPT_total_canoes_built_l485_48519


namespace NUMINAMATH_GPT_simplify_expression_l485_48592

variable {a b : ℝ}

theorem simplify_expression {a b : ℝ} (h : |2 - a + b| + (ab + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l485_48592


namespace NUMINAMATH_GPT_cyclist_speed_l485_48550

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem cyclist_speed :
  ∀ (d t : ℝ), 
  (d / 10 = t + 1) → 
  (d / 15 = t - 1) →
  required_speed d t = 12 := 
by
  intros d t h1 h2
  sorry

end NUMINAMATH_GPT_cyclist_speed_l485_48550


namespace NUMINAMATH_GPT_min_value_a_plus_2b_l485_48530

theorem min_value_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 20) : a + 2 * b = 4 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_plus_2b_l485_48530


namespace NUMINAMATH_GPT_Sam_age_l485_48512

theorem Sam_age (S D : ℕ) (h1 : S + D = 54) (h2 : S = D / 2) : S = 18 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_Sam_age_l485_48512


namespace NUMINAMATH_GPT_no_such_function_l485_48537

theorem no_such_function (f : ℕ → ℕ) : ¬ (∀ n : ℕ, f (f n) = n + 2019) :=
sorry

end NUMINAMATH_GPT_no_such_function_l485_48537


namespace NUMINAMATH_GPT_arithmetic_sequence_middle_term_l485_48579

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end NUMINAMATH_GPT_arithmetic_sequence_middle_term_l485_48579


namespace NUMINAMATH_GPT_janet_time_per_post_l485_48593

/-- Janet gets paid $0.25 per post she checks. She earns $90 per hour. 
    Prove that it takes her 10 seconds to check a post. -/
theorem janet_time_per_post
  (payment_per_post : ℕ → ℝ)
  (hourly_pay : ℝ)
  (posts_checked_hourly : ℕ)
  (secs_per_post : ℝ) :
  payment_per_post 1 = 0.25 →
  hourly_pay = 90 →
  hourly_pay = payment_per_post (posts_checked_hourly) →
  secs_per_post = 10 :=
sorry

end NUMINAMATH_GPT_janet_time_per_post_l485_48593


namespace NUMINAMATH_GPT_maggie_earnings_l485_48583

def subscriptions_to_parents := 4
def subscriptions_to_grandfather := 1
def subscriptions_to_next_door_neighbor := 2
def subscriptions_to_another_neighbor := 2 * subscriptions_to_next_door_neighbor
def subscription_rate := 5

theorem maggie_earnings : 
  (subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door_neighbor + subscriptions_to_another_neighbor) * subscription_rate = 55 := 
by
  sorry

end NUMINAMATH_GPT_maggie_earnings_l485_48583


namespace NUMINAMATH_GPT_find_divisor_l485_48563

noncomputable def divisor_of_nearest_divisible (a b : ℕ) (d : ℕ) : ℕ :=
  if h : b % d = 0 ∧ (b - a < d) then d else 0

theorem find_divisor (a b : ℕ) (d : ℕ) (h1 : b = 462) (h2 : a = 457)
  (h3 : b % d = 0) (h4 : b - a < d) :
  d = 5 :=
sorry

end NUMINAMATH_GPT_find_divisor_l485_48563
