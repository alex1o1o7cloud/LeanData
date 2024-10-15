import Mathlib

namespace NUMINAMATH_GPT_tangent_line_through_external_point_l375_37527

theorem tangent_line_through_external_point (x y : ℝ) (h_circle : x^2 + y^2 = 1) (P : ℝ × ℝ) (h_P : P = (1, 2)) : 
  (∃ k : ℝ, (y = 2 + k * (x - 1)) ∧ (x = 1 ∨ (3 * x - 4 * y + 5 = 0))) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_external_point_l375_37527


namespace NUMINAMATH_GPT_simplify_expression_l375_37588

theorem simplify_expression (p : ℝ) : 
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l375_37588


namespace NUMINAMATH_GPT_gcd_of_polynomial_l375_37516

theorem gcd_of_polynomial (b : ℕ) (hb : b % 780 = 0) : Nat.gcd (5 * b^3 + 2 * b^2 + 6 * b + 65) b = 65 := by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomial_l375_37516


namespace NUMINAMATH_GPT_salary_problem_l375_37589

theorem salary_problem
  (A B : ℝ)
  (h1 : A + B = 3000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 2250 :=
sorry

end NUMINAMATH_GPT_salary_problem_l375_37589


namespace NUMINAMATH_GPT_cylinder_height_l375_37547

theorem cylinder_height
  (V : ℝ → ℝ → ℝ) 
  (π : ℝ)
  (r h : ℝ)
  (vol_increase_height : ℝ)
  (vol_increase_radius : ℝ)
  (h_increase : ℝ)
  (r_increase : ℝ)
  (original_radius : ℝ) :
  V r h = π * r^2 * h → 
  vol_increase_height = π * r^2 * h_increase →
  vol_increase_radius = π * ((r + r_increase)^2 - r^2) * h →
  r = original_radius →
  vol_increase_height = 72 * π →
  vol_increase_radius = 72 * π →
  original_radius = 3 →
  r_increase = 2 →
  h_increase = 2 →
  h = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_l375_37547


namespace NUMINAMATH_GPT_sum_of_circle_areas_constant_l375_37581

theorem sum_of_circle_areas_constant (r OP : ℝ) (h1 : 0 < r) (h2 : 0 ≤ OP ∧ OP < r) 
  (a' b' c' : ℝ) (h3 : a'^2 + b'^2 + c'^2 = OP^2) :
  ∃ (a b c : ℝ), (a^2 + b^2 + c^2 = 3 * r^2 - OP^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_circle_areas_constant_l375_37581


namespace NUMINAMATH_GPT_no_such_quadratics_l375_37585

theorem no_such_quadratics :
  ¬ ∃ (a b c : ℤ), ∃ (x1 x2 x3 x4 : ℤ),
    (a * x1 * x2 = c ∧ a * (x1 + x2) = -b) ∧
    ((a + 1) * x3 * x4 = c + 1 ∧ (a + 1) * (x3 + x4) = -(b + 1)) :=
sorry

end NUMINAMATH_GPT_no_such_quadratics_l375_37585


namespace NUMINAMATH_GPT_find_X_sum_coordinates_l375_37501

/- Define points and their coordinates -/
variables (X Y Z : ℝ × ℝ)
variable  (XY XZ ZY : ℝ)
variable  (k : ℝ)
variable  (hxz : XZ = (3/4) * XY)
variable  (hzy : ZY = (1/4) * XY)
variable  (hy : Y = (2, 9))
variable  (hz : Z = (1, 5))

/-- Lean 4 statement for the proof problem -/
theorem find_X_sum_coordinates :
  (Y.1 = 2) ∧ (Y.2 = 9) ∧ (Z.1 = 1) ∧ (Z.2 = 5) ∧
  XZ = (3/4) * XY ∧ ZY = (1/4) * XY →
  (X.1 + X.2) = -9 := 
by
  sorry

end NUMINAMATH_GPT_find_X_sum_coordinates_l375_37501


namespace NUMINAMATH_GPT_odd_periodic_function_value_l375_37582

theorem odd_periodic_function_value
  (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = - f x)
  (periodic_f : ∀ x, f (x + 3) = f x)
  (bounded_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f 8.5 = -1 :=
sorry

end NUMINAMATH_GPT_odd_periodic_function_value_l375_37582


namespace NUMINAMATH_GPT_sales_this_month_l375_37537

-- Define the given conditions
def price_large := 60
def price_small := 30
def num_large_last_month := 8
def num_small_last_month := 4

-- Define the computation of total sales for last month
def sales_last_month : ℕ :=
  price_large * num_large_last_month + price_small * num_small_last_month

-- State the theorem to prove the sales this month
theorem sales_this_month : sales_last_month * 2 = 1200 :=
by
  -- Proof will follow, for now we use sorry as a placeholder
  sorry

end NUMINAMATH_GPT_sales_this_month_l375_37537


namespace NUMINAMATH_GPT_problem_x_sq_plus_y_sq_l375_37522

variables {x y : ℝ}

theorem problem_x_sq_plus_y_sq (h₁ : x - y = 12) (h₂ : x * y = 9) : x^2 + y^2 = 162 := 
sorry

end NUMINAMATH_GPT_problem_x_sq_plus_y_sq_l375_37522


namespace NUMINAMATH_GPT_probability_A_and_B_selected_l375_37511

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_and_B_selected_l375_37511


namespace NUMINAMATH_GPT_negation_of_P_l375_37598

-- Define the proposition P
def P (x : ℝ) : Prop := x^2 = 1 → x = 1

-- Define the negation of the proposition P
def neg_P (x : ℝ) : Prop := x^2 ≠ 1 → x ≠ 1

theorem negation_of_P (x : ℝ) : ¬P x ↔ neg_P x := by
  sorry

end NUMINAMATH_GPT_negation_of_P_l375_37598


namespace NUMINAMATH_GPT_problem_statement_l375_37504

noncomputable def f (x : ℝ) := 2 * x + 3
noncomputable def g (x : ℝ) := 3 * x - 2

theorem problem_statement : (f (g (f 3)) / g (f (g 3))) = 53 / 49 :=
by
  -- The proof is not provided as requested.
  sorry

end NUMINAMATH_GPT_problem_statement_l375_37504


namespace NUMINAMATH_GPT_jenna_water_cups_l375_37572

theorem jenna_water_cups (O S W : ℕ) (h1 : S = 3 * O) (h2 : W = 3 * S) (h3 : O = 4) : W = 36 :=
by
  sorry

end NUMINAMATH_GPT_jenna_water_cups_l375_37572


namespace NUMINAMATH_GPT_arithmetic_sequence_term_difference_l375_37593

theorem arithmetic_sequence_term_difference :
  let a : ℕ := 3
  let d : ℕ := 6
  let t1 := a + 1499 * d
  let t2 := a + 1503 * d
  t2 - t1 = 24 :=
    by
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_difference_l375_37593


namespace NUMINAMATH_GPT_smallest_integer_value_l375_37566

theorem smallest_integer_value (n : ℤ) : ∃ (n : ℤ), n = 5 ∧ n^2 - 11*n + 28 < 0 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_smallest_integer_value_l375_37566


namespace NUMINAMATH_GPT_MrsYoung_puzzle_complete_l375_37550

theorem MrsYoung_puzzle_complete :
  let total_pieces := 500
  let children := 4
  let pieces_per_child := total_pieces / children
  let minutes := 120
  let pieces_Reyn := (25 * (minutes / 30))
  let pieces_Rhys := 2 * pieces_Reyn
  let pieces_Rory := 3 * pieces_Reyn
  let pieces_Rina := 4 * pieces_Reyn
  let total_pieces_placed := pieces_Reyn + pieces_Rhys + pieces_Rory + pieces_Rina
  total_pieces_placed >= total_pieces :=
by
  sorry

end NUMINAMATH_GPT_MrsYoung_puzzle_complete_l375_37550


namespace NUMINAMATH_GPT_length_of_chord_EF_l375_37536

noncomputable def chord_length (theta_1 theta_2 : ℝ) : ℝ :=
  let x_1 := 2 * Real.cos theta_1
  let y_1 := Real.sin theta_1
  let x_2 := 2 * Real.cos theta_2
  let y_2 := Real.sin theta_2
  Real.sqrt ((x_2 - x_1)^2 + (y_2 - y_1)^2)

theorem length_of_chord_EF :
  ∀ (theta_1 theta_2 : ℝ), 
  (2 * Real.cos theta_1) + (Real.sin theta_1) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_2) + (Real.sin theta_2) + Real.sqrt 3 = 0 →
  (2 * Real.cos theta_1)^2 + 4 * (Real.sin theta_1)^2 = 4 →
  (2 * Real.cos theta_2)^2 + 4 * (Real.sin theta_2)^2 = 4 →
  chord_length theta_1 theta_2 = 8 / 5 :=
by
  intros theta_1 theta_2 h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_length_of_chord_EF_l375_37536


namespace NUMINAMATH_GPT_min_value_a_plus_9b_l375_37515

theorem min_value_a_plus_9b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 1 / b = 1) : a + 9 * b ≥ 16 :=
  sorry

end NUMINAMATH_GPT_min_value_a_plus_9b_l375_37515


namespace NUMINAMATH_GPT_joan_original_seashells_l375_37565

theorem joan_original_seashells (a b total: ℕ) (h1 : a = 63) (h2 : b = 16) (h3: total = a + b) : total = 79 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_joan_original_seashells_l375_37565


namespace NUMINAMATH_GPT_jim_catches_up_to_cara_l375_37592

noncomputable def time_to_catch_up (jim_speed: ℝ) (cara_speed: ℝ) (initial_time: ℝ) (stretch_time: ℝ) : ℝ :=
  let initial_distance_jim := jim_speed * initial_time
  let initial_distance_cara := cara_speed * initial_time
  let added_distance_cara := cara_speed * stretch_time
  let distance_gap := added_distance_cara
  let relative_speed := jim_speed - cara_speed
  distance_gap / relative_speed

theorem jim_catches_up_to_cara :
  time_to_catch_up 6 5 (30/60) (18/60) * 60 = 90 :=
by
  sorry

end NUMINAMATH_GPT_jim_catches_up_to_cara_l375_37592


namespace NUMINAMATH_GPT_janet_pairs_of_2_l375_37569

def total_pairs (x y z : ℕ) : Prop := x + y + z = 18

def total_cost (x y z : ℕ) : Prop := 2 * x + 5 * y + 7 * z = 60

theorem janet_pairs_of_2 (x y z : ℕ) (h1 : total_pairs x y z) (h2 : total_cost x y z) (hz : z = 3) : x = 12 :=
by
  -- Proof is currently skipped
  sorry

end NUMINAMATH_GPT_janet_pairs_of_2_l375_37569


namespace NUMINAMATH_GPT_weight_loss_in_april_l375_37576

-- Definitions based on given conditions
def total_weight_to_lose : ℕ := 10
def march_weight_loss : ℕ := 3
def may_weight_loss : ℕ := 3

-- Theorem statement
theorem weight_loss_in_april :
  total_weight_to_lose = march_weight_loss + 4 + may_weight_loss := 
sorry

end NUMINAMATH_GPT_weight_loss_in_april_l375_37576


namespace NUMINAMATH_GPT_final_weights_are_correct_l375_37540

-- Definitions of initial weights and reduction percentages per week
def initial_weight_A : ℝ := 300
def initial_weight_B : ℝ := 450
def initial_weight_C : ℝ := 600
def initial_weight_D : ℝ := 750

def reduction_A_week1 : ℝ := 0.20 * initial_weight_A
def reduction_B_week1 : ℝ := 0.15 * initial_weight_B
def reduction_C_week1 : ℝ := 0.30 * initial_weight_C
def reduction_D_week1 : ℝ := 0.25 * initial_weight_D

def weight_A_after_week1 : ℝ := initial_weight_A - reduction_A_week1
def weight_B_after_week1 : ℝ := initial_weight_B - reduction_B_week1
def weight_C_after_week1 : ℝ := initial_weight_C - reduction_C_week1
def weight_D_after_week1 : ℝ := initial_weight_D - reduction_D_week1

def reduction_A_week2 : ℝ := 0.25 * weight_A_after_week1
def reduction_B_week2 : ℝ := 0.30 * weight_B_after_week1
def reduction_C_week2 : ℝ := 0.10 * weight_C_after_week1
def reduction_D_week2 : ℝ := 0.20 * weight_D_after_week1

def weight_A_after_week2 : ℝ := weight_A_after_week1 - reduction_A_week2
def weight_B_after_week2 : ℝ := weight_B_after_week1 - reduction_B_week2
def weight_C_after_week2 : ℝ := weight_C_after_week1 - reduction_C_week2
def weight_D_after_week2 : ℝ := weight_D_after_week1 - reduction_D_week2

def reduction_A_week3 : ℝ := 0.15 * weight_A_after_week2
def reduction_B_week3 : ℝ := 0.10 * weight_B_after_week2
def reduction_C_week3 : ℝ := 0.20 * weight_C_after_week2
def reduction_D_week3 : ℝ := 0.30 * weight_D_after_week2

def weight_A_after_week3 : ℝ := weight_A_after_week2 - reduction_A_week3
def weight_B_after_week3 : ℝ := weight_B_after_week2 - reduction_B_week3
def weight_C_after_week3 : ℝ := weight_C_after_week2 - reduction_C_week3
def weight_D_after_week3 : ℝ := weight_D_after_week2 - reduction_D_week3

def reduction_A_week4 : ℝ := 0.10 * weight_A_after_week3
def reduction_B_week4 : ℝ := 0.20 * weight_B_after_week3
def reduction_C_week4 : ℝ := 0.25 * weight_C_after_week3
def reduction_D_week4 : ℝ := 0.15 * weight_D_after_week3

def final_weight_A : ℝ := weight_A_after_week3 - reduction_A_week4
def final_weight_B : ℝ := weight_B_after_week3 - reduction_B_week4
def final_weight_C : ℝ := weight_C_after_week3 - reduction_C_week4
def final_weight_D : ℝ := weight_D_after_week3 - reduction_D_week4

theorem final_weights_are_correct :
  final_weight_A = 137.7 ∧ 
  final_weight_B = 192.78 ∧ 
  final_weight_C = 226.8 ∧ 
  final_weight_D = 267.75 :=
by
  unfold final_weight_A final_weight_B final_weight_C final_weight_D
  sorry

end NUMINAMATH_GPT_final_weights_are_correct_l375_37540


namespace NUMINAMATH_GPT_tan_alpha_eq_one_third_l375_37518

variable (α : ℝ)

theorem tan_alpha_eq_one_third (h : Real.tan (α + Real.pi / 4) = 2) : Real.tan α = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_one_third_l375_37518


namespace NUMINAMATH_GPT_numBaskets_l375_37549

noncomputable def numFlowersInitial : ℕ := 5 + 5
noncomputable def numFlowersAfterGrowth : ℕ := numFlowersInitial + 20
noncomputable def numFlowersFinal : ℕ := numFlowersAfterGrowth - 10
noncomputable def flowersPerBasket : ℕ := 4

theorem numBaskets : numFlowersFinal / flowersPerBasket = 5 := 
by
  sorry

end NUMINAMATH_GPT_numBaskets_l375_37549


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l375_37559

theorem lcm_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 6) (h_product : a * b = 432) :
  Nat.lcm a b = 72 :=
by 
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l375_37559


namespace NUMINAMATH_GPT_max_gcd_a_is_25_l375_37512

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 100 + n^2 + 2 * n

-- Define the gcd function
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

-- Define the theorem to prove the maximum value of d_n as 25
theorem max_gcd_a_is_25 : ∃ n : ℕ, d n = 25 := 
sorry

end NUMINAMATH_GPT_max_gcd_a_is_25_l375_37512


namespace NUMINAMATH_GPT_continuity_at_x0_l375_37571

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4
def x0 := 3

theorem continuity_at_x0 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_x0_l375_37571


namespace NUMINAMATH_GPT_triangle_problem_l375_37591

noncomputable def triangle_sum : Real := sorry

theorem triangle_problem
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (hA : A = π / 6) -- A = 30 degrees
  (h_a : a = Real.sqrt 3) -- a = √3
  (h_law_of_sines : ∀ (x : ℝ), x = 2 * triangle_sum * Real.sin x) -- Law of Sines
  (h_sin_30 : Real.sin (π / 6) = 1 / 2) -- sin 30 degrees = 1/2
  : (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) 
  = 2 * Real.sqrt 3 := sorry

end NUMINAMATH_GPT_triangle_problem_l375_37591


namespace NUMINAMATH_GPT_number_of_hexagons_l375_37595

-- Definitions based on conditions
def num_pentagons : ℕ := 12

-- Based on the problem statement, the goal is to prove that the number of hexagons is 20
theorem number_of_hexagons (h : num_pentagons = 12) : ∃ (num_hexagons : ℕ), num_hexagons = 20 :=
by {
  -- proof would be here
  sorry
}

end NUMINAMATH_GPT_number_of_hexagons_l375_37595


namespace NUMINAMATH_GPT_parallel_and_perpendicular_implies_perpendicular_l375_37590

variables (l : Line) (α β : Plane)

axiom line_parallel_plane (l : Line) (π : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (π : Plane) : Prop
axiom planes_are_perpendicular (π₁ π₂ : Plane) : Prop

theorem parallel_and_perpendicular_implies_perpendicular
  (h1 : line_parallel_plane l α)
  (h2 : line_perpendicular_plane l β) 
  : planes_are_perpendicular α β :=
sorry

end NUMINAMATH_GPT_parallel_and_perpendicular_implies_perpendicular_l375_37590


namespace NUMINAMATH_GPT_simplify_fraction_l375_37505

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (10 * x * y^2) / (5 * x * y) = 2 * y := 
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l375_37505


namespace NUMINAMATH_GPT_calculate_expr_equals_243_l375_37521

theorem calculate_expr_equals_243 :
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049 = 243) :=
by
  sorry

end NUMINAMATH_GPT_calculate_expr_equals_243_l375_37521


namespace NUMINAMATH_GPT_number_of_people_l375_37509

theorem number_of_people (n k : ℕ) (h₁ : k * n * (n - 1) = 440) : n = 11 :=
sorry

end NUMINAMATH_GPT_number_of_people_l375_37509


namespace NUMINAMATH_GPT_sum_of_digits_of_fraction_repeating_decimal_l375_37564

theorem sum_of_digits_of_fraction_repeating_decimal :
  (exists (c d : ℕ), (4 / 13 : ℚ) = c * 0.1 + d * 0.01 ∧ (c + d) = 3) :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_fraction_repeating_decimal_l375_37564


namespace NUMINAMATH_GPT_prime_factor_of_reversed_difference_l375_37517

theorem prime_factor_of_reversed_difference (A B C : ℕ) (hA : A ≠ C) (hA_d : 1 ≤ A ∧ A ≤ 9) (hB_d : 0 ≤ B ∧ B ≤ 9) (hC_d : 1 ≤ C ∧ C ≤ 9) :
  ∃ p, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 11 := 
by
  sorry

end NUMINAMATH_GPT_prime_factor_of_reversed_difference_l375_37517


namespace NUMINAMATH_GPT_friends_meet_probability_l375_37573

noncomputable def probability_of_meeting :=
  let duration_total := 60 -- Total duration from 14:00 to 15:00 in minutes
  let duration_meeting := 30 -- Duration they can meet from 14:00 to 14:30 in minutes
  duration_meeting / duration_total

theorem friends_meet_probability : probability_of_meeting = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_friends_meet_probability_l375_37573


namespace NUMINAMATH_GPT_cost_pants_shirt_l375_37514

variable (P S C : ℝ)

theorem cost_pants_shirt (h1 : P + C = 244) (h2 : C = 5 * S) (h3 : C = 180) : P + S = 100 := by
  sorry

end NUMINAMATH_GPT_cost_pants_shirt_l375_37514


namespace NUMINAMATH_GPT_lcm_of_numbers_l375_37524

theorem lcm_of_numbers (x : Nat) (h_ratio : x ≠ 0) (h_hcf : Nat.gcd (5 * x) (Nat.gcd (7 * x) (9 * x)) = 11) :
    Nat.lcm (5 * x) (Nat.lcm (7 * x) (9 * x)) = 99 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_l375_37524


namespace NUMINAMATH_GPT_train_speed_is_36_0036_kmph_l375_37502

noncomputable def train_length : ℝ := 130
noncomputable def bridge_length : ℝ := 150
noncomputable def crossing_time : ℝ := 27.997760179185665
noncomputable def speed_in_kmph : ℝ := (train_length + bridge_length) / crossing_time * 3.6

theorem train_speed_is_36_0036_kmph :
  abs (speed_in_kmph - 36.0036) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_is_36_0036_kmph_l375_37502


namespace NUMINAMATH_GPT_num_ints_between_sqrt2_and_sqrt32_l375_37586

theorem num_ints_between_sqrt2_and_sqrt32 : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ k : ℤ, (2 ≤ k) ∧ (k ≤ 5)) :=
by
  sorry

end NUMINAMATH_GPT_num_ints_between_sqrt2_and_sqrt32_l375_37586


namespace NUMINAMATH_GPT_total_time_to_climb_seven_flights_l375_37575

-- Define the conditions
def first_flight_time : ℕ := 15
def difference_between_flights : ℕ := 10
def num_of_flights : ℕ := 7

-- Define the sum of an arithmetic series function
def arithmetic_series_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the theorem
theorem total_time_to_climb_seven_flights :
  arithmetic_series_sum first_flight_time difference_between_flights num_of_flights = 315 :=
by
  sorry

end NUMINAMATH_GPT_total_time_to_climb_seven_flights_l375_37575


namespace NUMINAMATH_GPT_neg_parallelogram_is_rhombus_l375_37556

def parallelogram_is_rhombus := true

theorem neg_parallelogram_is_rhombus : ¬ parallelogram_is_rhombus := by
  sorry

end NUMINAMATH_GPT_neg_parallelogram_is_rhombus_l375_37556


namespace NUMINAMATH_GPT_initial_percentage_reduction_l375_37532

theorem initial_percentage_reduction
  (x: ℕ)
  (h1: ∀ P: ℝ, P * (1 - x / 100) * 0.85 * 1.5686274509803921 = P) :
  x = 25 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_reduction_l375_37532


namespace NUMINAMATH_GPT_ratio_B_to_C_l375_37578

-- Definitions for conditions
def total_amount : ℕ := 1440
def B_amt : ℕ := 270
def A_amt := (1 / 3) * B_amt
def C_amt := total_amount - A_amt - B_amt

-- Theorem statement
theorem ratio_B_to_C : (B_amt : ℚ) / C_amt = 1 / 4 :=
  by
    sorry

end NUMINAMATH_GPT_ratio_B_to_C_l375_37578


namespace NUMINAMATH_GPT_minimum_value_l375_37551

open Classical

variable {a b c : ℝ}

theorem minimum_value (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a + b + c = 4) :
  36 ≤ (9 / a) + (16 / b) + (25 / c) :=
sorry

end NUMINAMATH_GPT_minimum_value_l375_37551


namespace NUMINAMATH_GPT_task_candy_distribution_l375_37561

noncomputable def candy_distribution_eq_eventually (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ m : ℕ, ∀ j : ℕ, m ≥ k → a (j + m * n) = a (0 + m * n)

theorem task_candy_distribution :
  ∀ n : ℕ, n > 0 →
  ∀ a : ℕ → ℕ,
  (∀ i : ℕ, a i = if a i % 2 = 1 then (a i) + 1 else a i) →
  (∀ i : ℕ, a (i + 1) = a i / 2 + a (i - 1) / 2) →
  candy_distribution_eq_eventually n a :=
by
  intros n n_positive a h_even h_transfer
  sorry

end NUMINAMATH_GPT_task_candy_distribution_l375_37561


namespace NUMINAMATH_GPT_f_2009_l375_37587

noncomputable def f : ℝ → ℝ := sorry -- This will be defined by the conditions.

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + 6) = f x + f 3
axiom f_one : f 1 = 2

theorem f_2009 : f 2009 = 2 :=
by {
  -- The proof would go here, summarizing the logical steps derived in the previous sections.
  sorry
}

end NUMINAMATH_GPT_f_2009_l375_37587


namespace NUMINAMATH_GPT_binom_14_11_l375_37507

open Nat

theorem binom_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end NUMINAMATH_GPT_binom_14_11_l375_37507


namespace NUMINAMATH_GPT_fraction_married_men_l375_37500

-- Define the problem conditions
def num_faculty : ℕ := 100
def women_perc : ℕ := 60
def married_perc : ℕ := 60
def single_men_perc : ℚ := 3/4

-- We need to calculate the fraction of men who are married.
theorem fraction_married_men :
  (60 : ℚ) / 100 = women_perc / num_faculty →
  (60 : ℚ) / 100 = married_perc / num_faculty →
  (3/4 : ℚ) = single_men_perc →
  ∃ (fraction : ℚ), fraction = 1/4 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_fraction_married_men_l375_37500


namespace NUMINAMATH_GPT_chocolates_per_small_box_l375_37580

/-- A large box contains 19 small boxes and each small box contains a certain number of chocolate bars.
There are 475 chocolate bars in the large box. --/
def number_of_chocolate_bars_per_small_box : Prop :=
  ∃ x : ℕ, 475 = 19 * x ∧ x = 25

theorem chocolates_per_small_box : number_of_chocolate_bars_per_small_box :=
by
  sorry -- proof is skipped

end NUMINAMATH_GPT_chocolates_per_small_box_l375_37580


namespace NUMINAMATH_GPT_eva_marks_difference_l375_37535

theorem eva_marks_difference 
    (m2 : ℕ) (a2 : ℕ) (s2 : ℕ) (total_marks : ℕ)
    (h_m2 : m2 = 80) (h_a2 : a2 = 90) (h_s2 : s2 = 90) (h_total_marks : total_marks = 485)
    (m1 a1 s1 : ℕ)
    (h_m1 : m1 = m2 + 10)
    (h_a1 : a1 = a2 - 15)
    (h_s1 : s1 = s2 - 1 / 3 * s2)
    (total_semesters : ℕ)
    (h_total_semesters : total_semesters = m1 + a1 + s1 + m2 + a2 + s2)
    : m1 = m2 + 10 := by
  sorry

end NUMINAMATH_GPT_eva_marks_difference_l375_37535


namespace NUMINAMATH_GPT_find_xnp_l375_37570

theorem find_xnp (x n p : ℕ) (h1 : 0 < x) (h2 : 0 < n) (h3 : Nat.Prime p) 
                  (h4 : 2 * x^3 + x^2 + 10 * x + 5 = 2 * p^n) : x + n + p = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_xnp_l375_37570


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l375_37525

theorem necessary_but_not_sufficient (a b : ℕ) : 
  (a ≠ 1 ∨ b ≠ 2) → ¬ (a + b = 3) → ¬(a = 1 ∧ b = 2) ∧ ((a = 1 ∧ b = 2) → (a + b = 3)) := sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l375_37525


namespace NUMINAMATH_GPT_rabbit_population_2002_l375_37563

theorem rabbit_population_2002 :
  ∃ (x : ℕ) (k : ℝ), 
    (180 - 50 = k * x) ∧ 
    (255 - 75 = k * 180) ∧ 
    x = 130 :=
by
  sorry

end NUMINAMATH_GPT_rabbit_population_2002_l375_37563


namespace NUMINAMATH_GPT_chuck_total_play_area_l375_37560

noncomputable def chuck_play_area (leash_radius : ℝ) : ℝ :=
  let middle_arc_area := (1 / 2) * Real.pi * leash_radius^2
  let corner_arc_area := 2 * (1 / 4) * Real.pi * leash_radius^2
  middle_arc_area + corner_arc_area

theorem chuck_total_play_area (leash_radius : ℝ) (shed_width shed_length : ℝ) 
  (h_radius : leash_radius = 4) (h_width : shed_width = 4) (h_length : shed_length = 6) :
  chuck_play_area leash_radius = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_chuck_total_play_area_l375_37560


namespace NUMINAMATH_GPT_mean_of_other_two_numbers_l375_37506

-- Definitions based on conditions in the problem.
def mean_of_four (numbers : List ℕ) : ℝ := 2187.25
def sum_of_numbers : ℕ := 1924 + 2057 + 2170 + 2229 + 2301 + 2365
def sum_of_four_numbers : ℝ := 4 * 2187.25
def sum_of_two_numbers := sum_of_numbers - sum_of_four_numbers

-- Theorem to assert the mean of the other two numbers.
theorem mean_of_other_two_numbers : (4297 / 2) = 2148.5 := by
  sorry

end NUMINAMATH_GPT_mean_of_other_two_numbers_l375_37506


namespace NUMINAMATH_GPT_geometric_sequence_sum_l375_37531

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q
def cond1 := a 0 + a 1 = 3
def cond2 := a 2 + a 3 = 12
def cond3 := is_geometric_sequence a

theorem geometric_sequence_sum :
  cond1 a →
  cond2 a →
  cond3 a q →
  a 4 + a 5 = 48 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l375_37531


namespace NUMINAMATH_GPT_multiples_of_4_in_sequence_l375_37544

-- Define the arithmetic sequence terms
def nth_term (a d n : ℤ) : ℤ := a + (n - 1) * d

-- Define the conditions
def cond_1 : ℤ := 200 -- first term
def cond_2 : ℤ := -6 -- common difference
def smallest_term : ℤ := 2

-- Define the count of terms function
def num_terms (a d min : ℤ) : ℤ := (a - min) / -d + 1

-- The total number of terms in the sequence
def total_terms : ℤ := num_terms cond_1 cond_2 smallest_term

-- Define a function to get the ith term that is a multiple of 4
def ith_multiple_of_4 (n : ℤ) : ℤ := cond_1 + 18 * (n - 1)

-- Define the count of multiples of 4 within the given number of terms
def count_multiples_of_4 (total : ℤ) : ℤ := (total / 3) + 1

-- Final theorem statement
theorem multiples_of_4_in_sequence : count_multiples_of_4 total_terms = 12 := sorry

end NUMINAMATH_GPT_multiples_of_4_in_sequence_l375_37544


namespace NUMINAMATH_GPT_min_forget_all_three_l375_37555

theorem min_forget_all_three (total_students students_forgot_gloves students_forgot_scarves students_forgot_hats : ℕ) (h_total : total_students = 60) (h_gloves : students_forgot_gloves = 55) (h_scarves : students_forgot_scarves = 52) (h_hats : students_forgot_hats = 50) :
  ∃ min_students_forget_three, min_students_forget_three = total_students - (total_students - students_forgot_gloves + total_students - students_forgot_scarves + total_students - students_forgot_hats) :=
by
  use 37
  sorry

end NUMINAMATH_GPT_min_forget_all_three_l375_37555


namespace NUMINAMATH_GPT_find_a_l375_37562

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l375_37562


namespace NUMINAMATH_GPT_smallest_b_value_l375_37529

theorem smallest_b_value (a b : ℕ) (h1 : a - b = 8) (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 3 := sorry

end NUMINAMATH_GPT_smallest_b_value_l375_37529


namespace NUMINAMATH_GPT_min_value_inequality_l375_37543

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 / x + 1 / y) * (4 * x + y) ≥ 9 ∧ ((1 / x + 1 / y) * (4 * x + y) = 9 ↔ y / x = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l375_37543


namespace NUMINAMATH_GPT_distance_from_A_to_origin_l375_37584

open Real

theorem distance_from_A_to_origin 
  (x1 y1 : ℝ)
  (hx1 : y1^2 = 4 * x1)
  (hratio : (x1 + 1) / abs y1 = 5 / 4)
  (hAF_gt_2 : dist (x1, y1) (1, 0) > 2) : 
  dist (x1, y1) (0, 0) = 4 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_distance_from_A_to_origin_l375_37584


namespace NUMINAMATH_GPT_polynomial_decomposition_l375_37528

theorem polynomial_decomposition :
  (x^3 - 2*x^2 + 3*x + 5) = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 :=
by sorry

end NUMINAMATH_GPT_polynomial_decomposition_l375_37528


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_is_integer_l375_37596

theorem radius_of_inscribed_circle_is_integer 
  (a b c : ℤ) 
  (h_pythagorean : c^2 = a^2 + b^2) 
  : ∃ r : ℤ, r = (a + b - c) / 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_is_integer_l375_37596


namespace NUMINAMATH_GPT_find_second_dimension_l375_37554

variable (l h w : ℕ)
variable (cost_per_sqft total_cost : ℕ)
variable (surface_area : ℕ)

def insulation_problem_conditions (l : ℕ) (h : ℕ) (cost_per_sqft : ℕ) (total_cost : ℕ) (w : ℕ) (surface_area : ℕ) : Prop :=
  l = 4 ∧ h = 3 ∧ cost_per_sqft = 20 ∧ total_cost = 1880 ∧ surface_area = (2 * l * w + 2 * l * h + 2 * w * h)

theorem find_second_dimension (l h w : ℕ) (cost_per_sqft total_cost surface_area : ℕ) :
  insulation_problem_conditions l h cost_per_sqft total_cost w surface_area →
  surface_area = 94 →
  w = 5 :=
by
  intros
  simp [insulation_problem_conditions] at *
  sorry

end NUMINAMATH_GPT_find_second_dimension_l375_37554


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l375_37519

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 2 → x^2 - a > 0) → (a < 2) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l375_37519


namespace NUMINAMATH_GPT_some_students_are_not_club_members_l375_37534

variable (U : Type) -- U represents the universe of students and club members
variables (Student ClubMember StudyLate : U → Prop)

-- Conditions derived from the problem
axiom h1 : ∃ s, Student s ∧ ¬ StudyLate s -- Some students do not study late
axiom h2 : ∀ c, ClubMember c → StudyLate c -- All club members study late

theorem some_students_are_not_club_members :
  ∃ s, Student s ∧ ¬ ClubMember s :=
by
  sorry

end NUMINAMATH_GPT_some_students_are_not_club_members_l375_37534


namespace NUMINAMATH_GPT_order_of_a_b_c_l375_37553

noncomputable def ln : ℝ → ℝ := Real.log
noncomputable def a : ℝ := ln 3 / 3
noncomputable def b : ℝ := ln 5 / 5
noncomputable def c : ℝ := ln 6 / 6

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l375_37553


namespace NUMINAMATH_GPT_min_value_fraction_l375_37574

theorem min_value_fraction (a b : ℝ) (n : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_sum : a + b = 2) : 
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l375_37574


namespace NUMINAMATH_GPT_calculate_a_minus_b_l375_37546

theorem calculate_a_minus_b (a b c : ℝ) (h1 : a - b - c = 3) (h2 : a - b + c = 11) : a - b = 7 :=
by 
  -- The proof would be fleshed out here.
  sorry

end NUMINAMATH_GPT_calculate_a_minus_b_l375_37546


namespace NUMINAMATH_GPT_simplify_expression_l375_37539

theorem simplify_expression (x : ℝ) : 2 * (x - 3) - (-x + 4) = 3 * x - 10 :=
by
  -- The proof is omitted, so use sorry to skip it
  sorry

end NUMINAMATH_GPT_simplify_expression_l375_37539


namespace NUMINAMATH_GPT_stratified_sampling_city_B_l375_37510

theorem stratified_sampling_city_B (sales_points_A : ℕ) (sales_points_B : ℕ) (sales_points_C : ℕ) (total_sales_points : ℕ) (sample_size : ℕ)
(h_total : total_sales_points = 450)
(h_sample : sample_size = 90)
(h_sales_points_A : sales_points_A = 180)
(h_sales_points_B : sales_points_B = 150)
(h_sales_points_C : sales_points_C = 120) :
  (sample_size * sales_points_B / total_sales_points) = 30 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_city_B_l375_37510


namespace NUMINAMATH_GPT_term_300_is_neg_8_l375_37513

noncomputable def geom_seq (a r : ℤ) : ℕ → ℤ
| 0       => a
| (n + 1) => r * geom_seq a r n

-- First term and second term are given as conditions.
def a1 : ℤ := 8
def a2 : ℤ := -8

-- Define the common ratio based on the conditions
def r : ℤ := a2 / a1

-- Theorem stating the 300th term is -8
theorem term_300_is_neg_8 : geom_seq a1 r 299 = -8 :=
by
  have h_r : r = -1 := by
    rw [r, a2, a1]
    norm_num
  rw [h_r]
  sorry

end NUMINAMATH_GPT_term_300_is_neg_8_l375_37513


namespace NUMINAMATH_GPT_no_nat_pairs_divisibility_l375_37538

theorem no_nat_pairs_divisibility (a b : ℕ) (hab : b^a ∣ a^b - 1) : false :=
sorry

end NUMINAMATH_GPT_no_nat_pairs_divisibility_l375_37538


namespace NUMINAMATH_GPT_values_of_x_and_y_l375_37542

theorem values_of_x_and_y (x y : ℝ) (h1 : x - y > x + 1) (h2 : x + y < y - 2) : x < -2 ∧ y < -1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_values_of_x_and_y_l375_37542


namespace NUMINAMATH_GPT_value_of_expression_l375_37526

variables {x1 x2 x3 x4 x5 x6 : ℝ}

theorem value_of_expression
  (h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 = 1)
  (h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 = 14)
  (h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 = 135) :
  16 * x1 + 25 * x2 + 36 * x3 + 49 * x4 + 64 * x5 + 81 * x6 = 832 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l375_37526


namespace NUMINAMATH_GPT_odds_of_picking_blue_marble_l375_37523

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end NUMINAMATH_GPT_odds_of_picking_blue_marble_l375_37523


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l375_37568

theorem cost_price_of_computer_table (CP SP : ℝ) 
  (h1 : SP = CP * 1.15) 
  (h2 : SP = 5750) 
  : CP = 5000 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l375_37568


namespace NUMINAMATH_GPT_clock_rings_in_a_day_l375_37557

theorem clock_rings_in_a_day (intervals : ℕ) (hours_in_a_day : ℕ) (time_between_rings : ℕ) : 
  intervals = hours_in_a_day / time_between_rings + 1 → intervals = 7 :=
sorry

end NUMINAMATH_GPT_clock_rings_in_a_day_l375_37557


namespace NUMINAMATH_GPT_sequence_formula_l375_37541

theorem sequence_formula (a : ℕ → ℤ) (h0 : a 0 = 1) (h1 : a 1 = 5)
    (h_rec : ∀ n, n ≥ 2 → a n = (2 * (a (n - 1))^2 - 3 * (a (n - 1)) - 9) / (2 * a (n - 2))) :
  ∀ n, a n = 2^(n + 2) - 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_formula_l375_37541


namespace NUMINAMATH_GPT_product_zero_when_a_is_2_l375_37545

theorem product_zero_when_a_is_2 : 
  ∀ (a : ℤ), a = 2 → (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  intros a ha
  sorry

end NUMINAMATH_GPT_product_zero_when_a_is_2_l375_37545


namespace NUMINAMATH_GPT_adding_sugar_increases_sweetness_l375_37579

theorem adding_sugar_increases_sweetness 
  (a b m : ℝ) (hb : b > a) (ha : a > 0) (hm : m > 0) : 
  (a / b) < (a + m) / (b + m) := 
by
  sorry

end NUMINAMATH_GPT_adding_sugar_increases_sweetness_l375_37579


namespace NUMINAMATH_GPT_log_sum_identity_l375_37548

-- Prove that: lg 8 + 3 * lg 5 = 3

noncomputable def common_logarithm (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum_identity : 
    common_logarithm 8 + 3 * common_logarithm 5 = 3 := 
by
  sorry

end NUMINAMATH_GPT_log_sum_identity_l375_37548


namespace NUMINAMATH_GPT_identify_person_l375_37520

variable (Person : Type) (Tweedledum Tralyalya : Person)
variable (has_black_card : Person → Prop)
variable (statement_true : Person → Prop)
variable (statement_made_by : Person)

-- Condition: The statement made: "Either I am Tweedledum, or I have a card of a black suit in my pocket."
def statement (p : Person) : Prop := p = Tweedledum ∨ has_black_card p

-- Condition: Anyone with a black card making a true statement is not possible.
axiom black_card_truth_contradiction : ∀ p : Person, has_black_card p → ¬ statement_true p

theorem identify_person :
statement_made_by = Tralyalya ∧ ¬ has_black_card statement_made_by :=
by
  sorry

end NUMINAMATH_GPT_identify_person_l375_37520


namespace NUMINAMATH_GPT_parallelepiped_inequality_l375_37599

theorem parallelepiped_inequality (a b c d : ℝ) (h : d^2 = a^2 + b^2 + c^2 + 2 * (a * b + a * c + b * c)) :
  a^2 + b^2 + c^2 ≥ (1 / 3) * d^2 :=
by
  sorry

end NUMINAMATH_GPT_parallelepiped_inequality_l375_37599


namespace NUMINAMATH_GPT_value_of_x_l375_37503

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l375_37503


namespace NUMINAMATH_GPT_value_of_x_l375_37597

theorem value_of_x (x : ℝ) (a : ℝ) (h1 : x ^ 2 * 8 ^ 3 / 256 = a) (h2 : a = 450) : x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l375_37597


namespace NUMINAMATH_GPT_solve_expression_hundreds_digit_l375_37567

def factorial (n : ℕ) : ℕ :=
  Nat.factorial n

def div_mod (a b m : ℕ) : ℕ :=
  (a / b) % m

def hundreds_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem solve_expression_hundreds_digit :
  hundreds_digit (div_mod (factorial 17) 5 1000 - div_mod (factorial 10) 2 1000) = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_expression_hundreds_digit_l375_37567


namespace NUMINAMATH_GPT_trapezoid_area_l375_37508

theorem trapezoid_area (x : ℝ) :
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  area = 9 * x^2 / 2 :=
by
  -- Definitions based on conditions
  let base1 := 5 * x
  let base2 := 4 * x
  let height := x
  let area := height * (base1 + base2) / 2
  -- Proof of the theorem, currently omitted
  sorry

end NUMINAMATH_GPT_trapezoid_area_l375_37508


namespace NUMINAMATH_GPT_find_value_of_x_l375_37594

theorem find_value_of_x (x : ℝ) : (45 * x = 0.4 * 900) -> x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_value_of_x_l375_37594


namespace NUMINAMATH_GPT_find_b_l375_37533

def perpendicular_vectors (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_b (b : ℝ) :
  perpendicular_vectors ⟨-5, 11⟩ ⟨b, 3⟩ →
  b = 33 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l375_37533


namespace NUMINAMATH_GPT_inequality_x_y_l375_37530

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end NUMINAMATH_GPT_inequality_x_y_l375_37530


namespace NUMINAMATH_GPT_two_digit_integers_remainder_3_count_l375_37583

theorem two_digit_integers_remainder_3_count :
  ∃ (n : ℕ) (a b : ℕ), a = 10 ∧ b = 99 ∧ (∀ k : ℕ, (a ≤ k ∧ k ≤ b) ↔ (∃ n : ℕ, k = 7 * n + 3 ∧ n ≥ 1 ∧ n ≤ 13)) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_integers_remainder_3_count_l375_37583


namespace NUMINAMATH_GPT_length_of_opposite_leg_l375_37552

noncomputable def hypotenuse_length : Real := 18

noncomputable def angle_deg : Real := 30

theorem length_of_opposite_leg (h : Real) (angle : Real) (condition1 : h = hypotenuse_length) (condition2 : angle = angle_deg) : 
 ∃ x : Real, 2 * x = h ∧ angle = 30 → x = 9 := 
by
  sorry

end NUMINAMATH_GPT_length_of_opposite_leg_l375_37552


namespace NUMINAMATH_GPT_proportion_solve_x_l375_37558

theorem proportion_solve_x :
  (0.75 / x = 5 / 7) → x = 1.05 :=
by
  sorry

end NUMINAMATH_GPT_proportion_solve_x_l375_37558


namespace NUMINAMATH_GPT_total_customers_in_line_l375_37577

-- Define the number of people behind the first person
def people_behind := 11

-- Define the total number of people in line
def people_in_line : Nat := people_behind + 1

-- Prove the total number of people in line is 12
theorem total_customers_in_line : people_in_line = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_customers_in_line_l375_37577
