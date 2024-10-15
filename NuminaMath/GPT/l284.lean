import Mathlib

namespace NUMINAMATH_GPT_total_employees_in_buses_l284_28441

-- Definitions from conditions
def busCapacity : ℕ := 150
def percentageFull1 : ℕ := 60
def percentageFull2 : ℕ := 70

-- Proving the total number of employees
theorem total_employees_in_buses : 
  (percentageFull1 * busCapacity / 100) + (percentageFull2 * busCapacity / 100) = 195 := 
by
  sorry

end NUMINAMATH_GPT_total_employees_in_buses_l284_28441


namespace NUMINAMATH_GPT_solution_correctness_l284_28481

theorem solution_correctness:
  ∀ (x1 : ℝ) (θ : ℝ), (θ = (5 * Real.pi / 13)) →
  (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) →
  ∃ (x2 : ℝ), (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ 
  (Real.sin x1 - 2 * Real.sin (x2 + θ) = -1) :=
by 
  intros x1 θ hθ hx1;
  sorry

end NUMINAMATH_GPT_solution_correctness_l284_28481


namespace NUMINAMATH_GPT_average_carnations_l284_28443

theorem average_carnations (c1 c2 c3 n : ℕ) (h1 : c1 = 9) (h2 : c2 = 14) (h3 : c3 = 13) (h4 : n = 3) :
  (c1 + c2 + c3) / n = 12 :=
by
  sorry

end NUMINAMATH_GPT_average_carnations_l284_28443


namespace NUMINAMATH_GPT_slope_intercept_condition_l284_28434

theorem slope_intercept_condition (m b : ℚ) (h_m : m = 1/3) (h_b : b = -3/4) : -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_GPT_slope_intercept_condition_l284_28434


namespace NUMINAMATH_GPT_quadratic_roots_shifted_l284_28419

theorem quadratic_roots_shifted (a b c : ℝ) (r s : ℝ) 
  (h1 : 4 * r ^ 2 + 2 * r - 9 = 0) 
  (h2 : 4 * s ^ 2 + 2 * s - 9 = 0) :
  c = 51 / 4 := by
  sorry

end NUMINAMATH_GPT_quadratic_roots_shifted_l284_28419


namespace NUMINAMATH_GPT_sum_faces_edges_vertices_l284_28410

def faces : Nat := 6
def edges : Nat := 12
def vertices : Nat := 8

theorem sum_faces_edges_vertices : faces + edges + vertices = 26 :=
by
  sorry

end NUMINAMATH_GPT_sum_faces_edges_vertices_l284_28410


namespace NUMINAMATH_GPT_max_of_x_l284_28403

theorem max_of_x (x y z : ℝ) (h1 : x + y + z = 7) (h2 : xy + xz + yz = 10) : x ≤ 3 := by
  sorry

end NUMINAMATH_GPT_max_of_x_l284_28403


namespace NUMINAMATH_GPT_mean_of_three_numbers_l284_28429

theorem mean_of_three_numbers (a : Fin 12 → ℕ) (x y z : ℕ) 
  (h1 : (Finset.univ.sum a) / 12 = 40)
  (h2 : ((Finset.univ.sum a) + x + y + z) / 15 = 50) :
  (x + y + z) / 3 = 90 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_three_numbers_l284_28429


namespace NUMINAMATH_GPT_equation_of_parallel_line_l284_28426

theorem equation_of_parallel_line {x y : ℝ} :
  (∃ b : ℝ, ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 + b = 0)) ↔ 
  (∃ b : ℝ, b = -2 ∧ ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 - 2 = 0)) := 
by 
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l284_28426


namespace NUMINAMATH_GPT_average_weight_decrease_l284_28454

theorem average_weight_decrease 
  (weight_old_student : ℝ := 92) 
  (weight_new_student : ℝ := 72) 
  (number_of_students : ℕ := 5) : 
  (weight_old_student - weight_new_student) / ↑number_of_students = 4 :=
by 
  sorry

end NUMINAMATH_GPT_average_weight_decrease_l284_28454


namespace NUMINAMATH_GPT_polar_line_through_centers_l284_28485

-- Definition of the given circles in polar coordinates
def Circle1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def Circle2 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Statement of the problem
theorem polar_line_through_centers (ρ θ : ℝ) :
  (∃ c1 c2 : ℝ × ℝ, Circle1 c1.fst c1.snd ∧ Circle2 c2.fst c2.snd ∧ θ = Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_polar_line_through_centers_l284_28485


namespace NUMINAMATH_GPT_two_digit_numbers_satisfying_l284_28446

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_numbers_satisfying (n : ℕ) : 
  is_two_digit n → n = P n + S n ↔ (n % 10 = 9) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_satisfying_l284_28446


namespace NUMINAMATH_GPT_parallel_vectors_l284_28491

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_b : b = (-1, m)) (h_parallel : ∃ k : ℝ, b = k • a) : m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_vectors_l284_28491


namespace NUMINAMATH_GPT_lowest_fraction_of_job_in_one_hour_l284_28422

-- Define the rates at which each person can work
def rate_A : ℚ := 1/3
def rate_B : ℚ := 1/4
def rate_C : ℚ := 1/6

-- Define the combined rates for each pair of people
def combined_rate_AB : ℚ := rate_A + rate_B
def combined_rate_AC : ℚ := rate_A + rate_C
def combined_rate_BC : ℚ := rate_B + rate_C

-- The Lean 4 statement to prove
theorem lowest_fraction_of_job_in_one_hour : min combined_rate_AB (min combined_rate_AC combined_rate_BC) = 5/12 :=
by 
  -- Here we state that the minimum combined rate is 5/12
  sorry

end NUMINAMATH_GPT_lowest_fraction_of_job_in_one_hour_l284_28422


namespace NUMINAMATH_GPT_remaining_budget_l284_28458

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end NUMINAMATH_GPT_remaining_budget_l284_28458


namespace NUMINAMATH_GPT_integer_values_of_f_l284_28496

theorem integer_values_of_f (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a * b ≠ 1) : 
  ∃ k ∈ ({4, 7} : Finset ℕ), 
    (a^2 + b^2 + a * b) / (a * b - 1) = k := 
by
  sorry

end NUMINAMATH_GPT_integer_values_of_f_l284_28496


namespace NUMINAMATH_GPT_quadratic_b_value_l284_28435

theorem quadratic_b_value (b m : ℝ) (h_b_pos : 0 < b) (h_quad_form : ∀ x, x^2 + b * x + 108 = (x + m)^2 - 4)
  (h_m_pos_sqrt : m = 4 * Real.sqrt 7 ∨ m = -4 * Real.sqrt 7) : b = 8 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_b_value_l284_28435


namespace NUMINAMATH_GPT_period_of_sine_plus_cosine_l284_28498

noncomputable def period_sine_cosine_sum (b : ℝ) : ℝ :=
  2 * Real.pi / b

theorem period_of_sine_plus_cosine (b : ℝ) (hb : b = 3) :
  period_sine_cosine_sum b = 2 * Real.pi / 3 :=
by
  rw [hb]
  apply rfl

end NUMINAMATH_GPT_period_of_sine_plus_cosine_l284_28498


namespace NUMINAMATH_GPT_geometric_sequence_value_l284_28414

theorem geometric_sequence_value 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_condition : a 4 * a 6 * a 8 * a 10 * a 12 = 32) :
  (a 10 ^ 2) / (a 12) = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_value_l284_28414


namespace NUMINAMATH_GPT_range_of_a_for_three_zeros_l284_28462

noncomputable def has_three_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
  (x₁^3 + a * x₁ + 2 = 0) ∧
  (x₂^3 + a * x₂ + 2 = 0) ∧
  (x₃^3 + a * x₃ + 2 = 0)

theorem range_of_a_for_three_zeros (a : ℝ) : has_three_zeros a ↔ a < -3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_three_zeros_l284_28462


namespace NUMINAMATH_GPT_field_dimension_solution_l284_28438

theorem field_dimension_solution (m : ℤ) (H1 : (3 * m + 11) * m = 100) : m = 5 :=
sorry

end NUMINAMATH_GPT_field_dimension_solution_l284_28438


namespace NUMINAMATH_GPT_teachers_on_field_trip_l284_28448

-- Definitions for conditions in the problem
def number_of_students := 12
def cost_per_student_ticket := 1
def cost_per_adult_ticket := 3
def total_cost_of_tickets := 24

-- Main statement
theorem teachers_on_field_trip :
  ∃ (T : ℕ), number_of_students * cost_per_student_ticket + T * cost_per_adult_ticket = total_cost_of_tickets ∧ T = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_teachers_on_field_trip_l284_28448


namespace NUMINAMATH_GPT_inequality_solution_l284_28423

theorem inequality_solution (x : ℝ) : (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) ∨ x = -3 :=
by
sorry

end NUMINAMATH_GPT_inequality_solution_l284_28423


namespace NUMINAMATH_GPT_find_integers_in_range_l284_28439

theorem find_integers_in_range :
  ∀ x : ℤ,
  (20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = 19) ↔
  x = 24 ∨ x = 29 ∨ x = 34 ∨ x = 39 ∨ x = 44 ∨ x = 49 :=
by sorry

end NUMINAMATH_GPT_find_integers_in_range_l284_28439


namespace NUMINAMATH_GPT_quadratic_function_properties_l284_28431

theorem quadratic_function_properties :
  ∃ a : ℝ, ∃ f : ℝ → ℝ,
    (∀ x : ℝ, f x = a * (x + 1) ^ 2 - 2) ∧
    (f 1 = 10) ∧
    (f (-1) = -2) ∧
    (∀ x : ℝ, x > -1 → f x ≥ f (-1))
:=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l284_28431


namespace NUMINAMATH_GPT_checkerboard_disc_coverage_l284_28474

/-- A circular disc with a diameter of 5 units is placed on a 10 x 10 checkerboard with each square having a side length of 1 unit such that the centers of both the disc and the checkerboard coincide.
    Prove that the number of checkerboard squares that are completely covered by the disc is 36. -/
theorem checkerboard_disc_coverage :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let side_length : ℝ := 1
  let board_size : ℕ := 10
  let disc_center : ℝ × ℝ := (board_size / 2, board_size / 2)
  ∃ (count : ℕ), count = 36 := 
  sorry

end NUMINAMATH_GPT_checkerboard_disc_coverage_l284_28474


namespace NUMINAMATH_GPT_minimum_guests_needed_l284_28467

theorem minimum_guests_needed (total_food : ℕ) (max_food_per_guest : ℕ) (guests_needed : ℕ) : 
  total_food = 323 → max_food_per_guest = 2 → guests_needed = Nat.ceil (323 / 2) → guests_needed = 162 :=
by
  intros
  sorry

end NUMINAMATH_GPT_minimum_guests_needed_l284_28467


namespace NUMINAMATH_GPT_actual_time_before_storm_is_18_18_l284_28406

theorem actual_time_before_storm_is_18_18 :
  ∃ h m : ℕ, (h = 18) ∧ (m = 18) ∧ 
            ((09 = (if h == 0 then 1 else h - 1) ∨ 09 = (if h == 23 then 0 else h + 1)) ∧ 
             (09 = (if m == 0 then 1 else m - 1) ∨ 09 = (if m == 59 then 0 else m + 1))) := 
  sorry

end NUMINAMATH_GPT_actual_time_before_storm_is_18_18_l284_28406


namespace NUMINAMATH_GPT_length_DE_l284_28497

open Classical

noncomputable def triangle_base_length (ABC_base : ℝ) : ℝ :=
15

noncomputable def is_parallel (DE BC : ℝ) : Prop :=
DE = BC

noncomputable def area_ratio (triangle_small triangle_large : ℝ) : ℝ :=
0.25

theorem length_DE 
  (ABC_base : ℝ)
  (DE : ℝ)
  (BC : ℝ)
  (triangle_small : ℝ)
  (triangle_large : ℝ)
  (h_base : triangle_base_length ABC_base = 15)
  (h_parallel : is_parallel DE BC)
  (h_area : area_ratio triangle_small triangle_large = 0.25)
  (h_similar : true):
  DE = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_length_DE_l284_28497


namespace NUMINAMATH_GPT_solve_equation_125_eq_5_25_exp_x_min_2_l284_28447

theorem solve_equation_125_eq_5_25_exp_x_min_2 :
    ∃ x : ℝ, 125 = 5 * (25 : ℝ)^(x - 2) ∧ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_125_eq_5_25_exp_x_min_2_l284_28447


namespace NUMINAMATH_GPT_rhombus_longest_diagonal_l284_28466

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 192) (h_ratio : ratio = 4 / 3) :
  ∃ d1 d2 : ℝ, d1 / d2 = 4 / 3 ∧ (d1 * d2) / 2 = 192 ∧ d1 = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_longest_diagonal_l284_28466


namespace NUMINAMATH_GPT_example_theorem_l284_28455

noncomputable def P (A : Set ℕ) : ℝ := sorry

variable (A1 A2 A3 : Set ℕ)

axiom prob_A1 : P A1 = 0.2
axiom prob_A2 : P A2 = 0.3
axiom prob_A3 : P A3 = 0.5

theorem example_theorem : P (A1 ∪ A2) ≤ 0.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_example_theorem_l284_28455


namespace NUMINAMATH_GPT_converse_of_squared_positive_is_negative_l284_28499

theorem converse_of_squared_positive_is_negative (x : ℝ) :
  (∀ x : ℝ, x < 0 → x^2 > 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
sorry

end NUMINAMATH_GPT_converse_of_squared_positive_is_negative_l284_28499


namespace NUMINAMATH_GPT_emily_necklaces_for_friends_l284_28489

theorem emily_necklaces_for_friends (n b B : ℕ)
  (h1 : n = 26)
  (h2 : b = 2)
  (h3 : B = 52)
  (h4 : n * b = B) : 
  n = 26 :=
by
  sorry

end NUMINAMATH_GPT_emily_necklaces_for_friends_l284_28489


namespace NUMINAMATH_GPT_find_slope_l3_l284_28468

/-- Conditions --/
def line1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line2 (x y : ℝ) : Prop := y = 2
def A : Prod ℝ ℝ := (0, -3)
def area_ABC : ℝ := 5

noncomputable def B : Prod ℝ ℝ := (2, 2)  -- Simultaneous solution of line1 and line2

theorem find_slope_l3 (C : ℝ × ℝ) (slope_l3 : ℝ) :
  line2 C.1 C.2 ∧
  ((0 : ℝ), -3) ∈ {p : ℝ × ℝ | line1 p.1 p.2 → line2 p.1 p.2 } ∧
  C.2 = 2 ∧
  0 ≤ slope_l3 ∧
  area_ABC = 5 →
  slope_l3 = 5 / 4 :=
sorry

end NUMINAMATH_GPT_find_slope_l3_l284_28468


namespace NUMINAMATH_GPT_hyperbola_standard_eq_line_eq_AB_l284_28476

noncomputable def fixed_points : (Real × Real) × (Real × Real) := ((-Real.sqrt 2, 0.0), (Real.sqrt 2, 0.0))

def locus_condition (P : Real × Real) (F1 F2 : Real × Real) : Prop :=
  abs (dist P F2 - dist P F1) = 2

def curve_E (P : Real × Real) : Prop :=
  (P.1 < 0) ∧ (P.1 * P.1 - P.2 * P.2 = 1)

theorem hyperbola_standard_eq :
  ∃ P : Real × Real, locus_condition P (fixed_points.1) (fixed_points.2) ↔ curve_E P :=
sorry

def line_intersects_hyperbola (P : Real × Real) (k : Real) : Prop :=
  P.2 = k * P.1 - 1 ∧ curve_E P

def dist_A_B (A B : Real × Real) : Real :=
  dist A B

theorem line_eq_AB :
  ∃ k : Real, k = -Real.sqrt 5 / 2 ∧
              ∃ A B : Real × Real, line_intersects_hyperbola A k ∧ 
              line_intersects_hyperbola B k ∧ 
              dist_A_B A B = 6 * Real.sqrt 3 ∧
              ∀ x y : Real, y = k * x - 1 ↔ x * (Real.sqrt 5/2) + y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_eq_line_eq_AB_l284_28476


namespace NUMINAMATH_GPT_inequality_holds_for_positive_reals_l284_28450

theorem inequality_holds_for_positive_reals (x y : ℝ) (m n : ℤ) 
  (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  (1 - x^n)^m + (1 - y^m)^n ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_reals_l284_28450


namespace NUMINAMATH_GPT_num_teachers_in_Oxford_High_School_l284_28416

def classes : Nat := 15
def students_per_class : Nat := 20
def principals : Nat := 1
def total_people : Nat := 349

theorem num_teachers_in_Oxford_High_School : 
  ∃ (teachers : Nat), teachers = total_people - (classes * students_per_class + principals) :=
by
  use 48
  sorry

end NUMINAMATH_GPT_num_teachers_in_Oxford_High_School_l284_28416


namespace NUMINAMATH_GPT_work_days_B_l284_28464

theorem work_days_B (A B: ℕ) (work_per_day_B: ℕ) (total_days : ℕ) (total_units : ℕ) :
  (A = 2 * B) → (work_per_day_B = 1) → (total_days = 36) → (B = 1) → (total_units = total_days * (A + B)) → 
  total_units / work_per_day_B = 108 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_work_days_B_l284_28464


namespace NUMINAMATH_GPT_AH_HD_ratio_l284_28484

-- Given conditions
variables {A B C H D : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited D]
variables (BC : ℝ) (AC : ℝ) (angle_C : ℝ)
-- We assume the values provided in the problem
variables (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4)

-- Altitudes and orthocenter assumption, representing intersections at orthocenter H
variables (A D H : Type) -- Points to represent A, D, and orthocenter H

noncomputable def AH_H_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) : ℝ :=
  if BC = 6 ∧ AC = 4 * Real.sqrt 2 ∧ angle_C = Real.pi / 4 then 2 else 0

-- We need to prove the ratio AH:HD equals 2 given the conditions
theorem AH_HD_ratio (BC AC : ℝ) (angle_C : ℝ)
  (BC_eq : BC = 6) (AC_eq : AC = 4 * Real.sqrt 2) (angle_C_eq : angle_C = Real.pi / 4) :
  AH_H_ratio BC AC angle_C BC_eq AC_eq angle_C_eq = 2 :=
by {
  -- the statement will be proved here
  sorry
}

end NUMINAMATH_GPT_AH_HD_ratio_l284_28484


namespace NUMINAMATH_GPT_slope_of_monotonically_decreasing_function_l284_28409

theorem slope_of_monotonically_decreasing_function
  (k b : ℝ)
  (H : ∀ x₁ x₂, x₁ ≤ x₂ → k * x₁ + b ≥ k * x₂ + b) : k < 0 := sorry

end NUMINAMATH_GPT_slope_of_monotonically_decreasing_function_l284_28409


namespace NUMINAMATH_GPT_zoes_apartment_number_units_digit_is_1_l284_28400

-- Defining the conditions as the initial problem does
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def has_digit_two (n : ℕ) : Prop :=
  n / 10 = 2 ∨ n % 10 = 2

def three_out_of_four (n : ℕ) : Prop :=
  (is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ ¬ has_digit_two n) ∨
  (is_square n ∧ is_odd n ∧ ¬ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (is_square n ∧ ¬ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n) ∨
  (¬ is_square n ∧ is_odd n ∧ is_divisible_by_3 n ∧ has_digit_two n)

theorem zoes_apartment_number_units_digit_is_1 : ∃ n : ℕ, is_two_digit_number n ∧ three_out_of_four n ∧ n % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_zoes_apartment_number_units_digit_is_1_l284_28400


namespace NUMINAMATH_GPT_visitors_equal_cats_l284_28477

-- Definition for conditions
def visitors_pets_cats (V C : ℕ) : Prop :=
  (∃ P : ℕ, P = 3 * V ∧ P = 3 * C)

-- Statement of the proof problem
theorem visitors_equal_cats {V C : ℕ}
  (h : visitors_pets_cats V C) : V = C :=
by sorry

end NUMINAMATH_GPT_visitors_equal_cats_l284_28477


namespace NUMINAMATH_GPT_annika_return_time_l284_28490

-- Define the rate at which Annika hikes.
def hiking_rate := 10 -- minutes per kilometer

-- Define the distances mentioned in the problem.
def initial_distance_east := 2.5 -- kilometers
def total_distance_east := 3.5 -- kilometers

-- Define the time calculations.
def additional_distance_east := total_distance_east - initial_distance_east

-- Calculate the total time required for Annika to get back to the start.
theorem annika_return_time (rate : ℝ) (initial_dist : ℝ) (total_dist : ℝ) (additional_dist : ℝ) : 
  initial_dist = 2.5 → total_dist = 3.5 → rate = 10 → additional_dist = total_dist - initial_dist → 
  (2.5 * rate + additional_dist * rate * 2) = 45 :=
by
-- Since this is just the statement and no proof is needed, we use sorry
sorry

end NUMINAMATH_GPT_annika_return_time_l284_28490


namespace NUMINAMATH_GPT_total_children_l284_28408

-- Definitions for the conditions in the problem
def boys : ℕ := 19
def girls : ℕ := 41

-- Theorem stating the total number of children is 60
theorem total_children : boys + girls = 60 :=
by
  -- calculation done to show steps, but not necessary for the final statement
  sorry

end NUMINAMATH_GPT_total_children_l284_28408


namespace NUMINAMATH_GPT_unique_solution_a_eq_sqrt3_l284_28442

theorem unique_solution_a_eq_sqrt3 (a : ℝ) :
  (∃! x : ℝ, x^2 - a * |x| + a^2 - 3 = 0) ↔ a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_unique_solution_a_eq_sqrt3_l284_28442


namespace NUMINAMATH_GPT_quadrilateral_area_correct_l284_28449

noncomputable def area_of_quadrilateral (n : ℕ) (hn : n > 0) : ℚ :=
  (2 * n^3) / (4 * n^2 - 1)

theorem quadrilateral_area_correct (n : ℕ) (hn : n > 0) :
  ∃ area : ℚ, area = (2 * n^3) / (4 * n^2 - 1) :=
by
  use area_of_quadrilateral n hn
  sorry

end NUMINAMATH_GPT_quadrilateral_area_correct_l284_28449


namespace NUMINAMATH_GPT_amc_problem_l284_28452

theorem amc_problem (a b : ℕ) (h : ∀ n : ℕ, 0 < n → a^n + n ∣ b^n + n) : a = b :=
sorry

end NUMINAMATH_GPT_amc_problem_l284_28452


namespace NUMINAMATH_GPT_compare_exponents_l284_28494

noncomputable def a : ℝ := 20 ^ 22
noncomputable def b : ℝ := 21 ^ 21
noncomputable def c : ℝ := 22 ^ 20

theorem compare_exponents : a > b ∧ b > c :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_exponents_l284_28494


namespace NUMINAMATH_GPT_good_eggs_collected_l284_28457

/-- 
Uncle Ben has 550 chickens on his farm, consisting of 49 roosters and the rest being hens. 
Out of these hens, there are three types:
1. Type A: 25 hens do not lay eggs at all.
2. Type B: 155 hens lay 2 eggs per day.
3. Type C: The remaining hens lay 4 eggs every three days.

Moreover, Uncle Ben found that 3% of the eggs laid by Type B and Type C hens go bad before being collected. 
Prove that the total number of good eggs collected by Uncle Ben after one day is 716.
-/
theorem good_eggs_collected 
    (total_chickens : ℕ) (roosters : ℕ) (typeA_hens : ℕ) (typeB_hens : ℕ) 
    (typeB_eggs_per_day : ℕ) (typeC_eggs_per_3days : ℕ) (percent_bad_eggs : ℚ) :
  total_chickens = 550 →
  roosters = 49 →
  typeA_hens = 25 →
  typeB_hens = 155 →
  typeB_eggs_per_day = 2 →
  typeC_eggs_per_3days = 4 →
  percent_bad_eggs = 0.03 →
  (total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day) - 
  round (percent_bad_eggs * ((total_chickens - roosters - typeA_hens - typeB_hens) * (typeC_eggs_per_3days / 3) + (typeB_hens * typeB_eggs_per_day))) = 716 :=
by
  intros
  sorry

end NUMINAMATH_GPT_good_eggs_collected_l284_28457


namespace NUMINAMATH_GPT_distance_from_D_to_plane_B1EF_l284_28421

theorem distance_from_D_to_plane_B1EF :
  let D := (0, 0, 0)
  let B₁ := (1, 1, 1)
  let E := (1, 1/2, 0)
  let F := (1/2, 1, 0)
  ∃ (d : ℝ), d = 1 := by
  sorry

end NUMINAMATH_GPT_distance_from_D_to_plane_B1EF_l284_28421


namespace NUMINAMATH_GPT_alice_study_time_for_average_75_l284_28471

variable (study_time : ℕ → ℚ)
variable (score : ℕ → ℚ)

def inverse_relation := ∀ n, study_time n * score n = 120

theorem alice_study_time_for_average_75
  (inverse_relation : inverse_relation study_time score)
  (study_time_1 : study_time 1 = 2)
  (score_1 : score 1 = 60)
  : study_time 2 = 4/3 := by
  sorry

end NUMINAMATH_GPT_alice_study_time_for_average_75_l284_28471


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l284_28453

noncomputable def f (x : ℝ) : ℝ := sorry

theorem monotonic_increasing_interval :
  (∀ x Δx : ℝ, 0 < x → 0 < Δx → 
  (f (x + Δx) - f x) / Δx = (2 / (Real.sqrt (x + Δx) + Real.sqrt x)) - (1 / (x^2 + x * Δx))) →
  ∀ x : ℝ, 1 < x → (∃ ε > 0, ∀ y, x < y ∧ y < x + ε → f y > f x) :=
by
  intro hyp
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l284_28453


namespace NUMINAMATH_GPT_anusha_receives_84_l284_28405

-- Define the conditions as given in the problem
def anusha_amount (A : ℕ) (B : ℕ) (E : ℕ) : Prop :=
  12 * A = 8 * B ∧ 12 * A = 6 * E ∧ A + B + E = 378

-- Lean statement to prove the amount Anusha gets is 84
theorem anusha_receives_84 (A B E : ℕ) (h : anusha_amount A B E) : A = 84 :=
sorry

end NUMINAMATH_GPT_anusha_receives_84_l284_28405


namespace NUMINAMATH_GPT_whale_tongue_weight_difference_l284_28420

noncomputable def tongue_weight_blue_whale_kg : ℝ := 2700
noncomputable def tongue_weight_fin_whale_kg : ℝ := 1800
noncomputable def kg_to_pounds : ℝ := 2.20462
noncomputable def ton_to_pounds : ℝ := 2000

noncomputable def tongue_weight_blue_whale_tons := (tongue_weight_blue_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def tongue_weight_fin_whale_tons := (tongue_weight_fin_whale_kg * kg_to_pounds) / ton_to_pounds
noncomputable def weight_difference_tons := tongue_weight_blue_whale_tons - tongue_weight_fin_whale_tons

theorem whale_tongue_weight_difference :
  weight_difference_tons = 0.992079 :=
by
  sorry

end NUMINAMATH_GPT_whale_tongue_weight_difference_l284_28420


namespace NUMINAMATH_GPT_total_unique_handshakes_l284_28432

def num_couples := 8
def num_individuals := num_couples * 2
def potential_handshakes_per_person := num_individuals - 1 - 1
def total_handshakes := num_individuals * potential_handshakes_per_person / 2

theorem total_unique_handshakes : total_handshakes = 112 := sorry

end NUMINAMATH_GPT_total_unique_handshakes_l284_28432


namespace NUMINAMATH_GPT_total_lambs_l284_28445

-- Defining constants
def Merry_lambs : ℕ := 10
def Brother_lambs : ℕ := Merry_lambs + 3

-- Proving the total number of lambs
theorem total_lambs : Merry_lambs + Brother_lambs = 23 :=
  by
    -- The actual proof is omitted and a placeholder is put instead
    sorry

end NUMINAMATH_GPT_total_lambs_l284_28445


namespace NUMINAMATH_GPT_camille_total_birds_count_l284_28461

theorem camille_total_birds_count :
  let cardinals := 3
  let robins := 4 * cardinals
  let blue_jays := 2 * cardinals
  let sparrows := 3 * cardinals + 1
  let pigeons := 3 * blue_jays
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry

end NUMINAMATH_GPT_camille_total_birds_count_l284_28461


namespace NUMINAMATH_GPT_product_of_solutions_abs_eq_l284_28486

theorem product_of_solutions_abs_eq (x1 x2 : ℝ) (h1 : |2 * x1 - 1| + 4 = 24) (h2 : |2 * x2 - 1| + 4 = 24) : x1 * x2 = -99.75 := 
sorry

end NUMINAMATH_GPT_product_of_solutions_abs_eq_l284_28486


namespace NUMINAMATH_GPT_find_k_l284_28483

theorem find_k (k : ℕ) : (1/2)^18 * (1/81)^k = (1/18)^18 → k = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_k_l284_28483


namespace NUMINAMATH_GPT_minimum_value_reciprocals_l284_28415

theorem minimum_value_reciprocals (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : 2 / Real.sqrt (a^2 + 4 * b^2) = Real.sqrt 2) :
  (1 / a^2 + 1 / b^2) = 9 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_reciprocals_l284_28415


namespace NUMINAMATH_GPT_solve_linear_equation_one_variable_with_parentheses_l284_28463

/--
Theorem: Solving a linear equation in one variable that contains parentheses
is equivalent to the process of:
1. Removing the parentheses,
2. Moving terms,
3. Combining like terms, and
4. Making the coefficient of the unknown equal to 1.

Given: a linear equation in one variable that contains parentheses
Prove: The process of solving it is to remove the parentheses, move terms, combine like terms, and make the coefficient of the unknown equal to 1.
-/
theorem solve_linear_equation_one_variable_with_parentheses
  (eq : String) :
  ∃ instructions : String,
    instructions = "remove the parentheses; move terms; combine like terms; make the coefficient of the unknown equal to 1" :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_equation_one_variable_with_parentheses_l284_28463


namespace NUMINAMATH_GPT_shelves_of_mystery_books_l284_28469

theorem shelves_of_mystery_books (total_books : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ) (M : ℕ) 
  (h_total_books : total_books = 54) 
  (h_picture_shelves : picture_shelves = 4) 
  (h_books_per_shelf : books_per_shelf = 6)
  (h_mystery_books : total_books - picture_shelves * books_per_shelf = M * books_per_shelf) :
  M = 5 :=
by
  sorry

end NUMINAMATH_GPT_shelves_of_mystery_books_l284_28469


namespace NUMINAMATH_GPT_min_value_expression_l284_28440

theorem min_value_expression (k x y z : ℝ) (hk : 0 < k) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x_min y_min z_min : ℝ, (0 < x_min) ∧ (0 < y_min) ∧ (0 < z_min) ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    k * (4 * z / (2 * x + y) + 4 * x / (y + 2 * z) + y / (x + z))
    ≥ 3 * k) ∧
  k * (4 * z_min / (2 * x_min + y_min) + 4 * x_min / (y_min + 2 * z_min) + y_min / (x_min + z_min)) = 3 * k :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l284_28440


namespace NUMINAMATH_GPT_train_length_l284_28437

theorem train_length (L : ℕ) 
  (h_tree : L / 120 = L / 200 * 200) 
  (h_platform : (L + 800) / 200 = L / 120) : 
  L = 1200 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l284_28437


namespace NUMINAMATH_GPT_speed_of_other_train_l284_28444

theorem speed_of_other_train (len1 len2 time : ℝ) (v1 v_other : ℝ) :
  len1 = 200 ∧ len2 = 300 ∧ time = 17.998560115190788 ∧ v1 = 40 →
  v_other = ((len1 + len2) / 1000) / (time / 3600) - v1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_speed_of_other_train_l284_28444


namespace NUMINAMATH_GPT_total_team_points_l284_28436

theorem total_team_points :
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  (A + B + C + D + E + F + G + H = 22) :=
by
  let A := 2
  let B := 9
  let C := 4
  let D := -3
  let E := 7
  let F := 0
  let G := 5
  let H := -2
  sorry

end NUMINAMATH_GPT_total_team_points_l284_28436


namespace NUMINAMATH_GPT_initial_bacteria_count_l284_28456

theorem initial_bacteria_count (n : ℕ) : 
  (n * 4^10 = 4194304) → n = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_bacteria_count_l284_28456


namespace NUMINAMATH_GPT_average_employees_per_week_l284_28428

variable (x : ℕ)

theorem average_employees_per_week (h1 : x + 200 > x)
                                   (h2 : x < 200)
                                   (h3 : 2 * 200 = 400) :
  (x + 200 + x + 200 + 200 + 400) / 4 = 250 := by
  sorry

end NUMINAMATH_GPT_average_employees_per_week_l284_28428


namespace NUMINAMATH_GPT_div_factorial_result_l284_28495

-- Define the given condition
def ten_fact : ℕ := 3628800

-- Define four factorial
def four_fact : ℕ := 4 * 3 * 2 * 1

-- State the theorem to be proved
theorem div_factorial_result : ten_fact / four_fact = 151200 :=
by
  -- Sorry is used to skip the proof, only the statement is provided
  sorry

end NUMINAMATH_GPT_div_factorial_result_l284_28495


namespace NUMINAMATH_GPT_minimize_material_used_l284_28459

theorem minimize_material_used (r h : ℝ) (V : ℝ) (S : ℝ) 
  (volume_formula : π * r^2 * h = V) (volume_given : V = 27 * π) :
  ∃ r, r = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_material_used_l284_28459


namespace NUMINAMATH_GPT_find_k_value_l284_28475

theorem find_k_value (k : ℚ) :
  (∀ x y : ℚ, (x = 1/3 ∧ y = -8 → -3/4 - 3 * k * x = 7 * y)) → k = 55.25 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l284_28475


namespace NUMINAMATH_GPT_value_of_xyz_l284_28482

theorem value_of_xyz (x y z : ℂ) 
  (h1 : x * y + 5 * y = -20)
  (h2 : y * z + 5 * z = -20)
  (h3 : z * x + 5 * x = -20) :
  x * y * z = 80 := 
by
  sorry

end NUMINAMATH_GPT_value_of_xyz_l284_28482


namespace NUMINAMATH_GPT_cows_in_group_l284_28417

variable (c h : ℕ)

theorem cows_in_group (hcow : 4 * c + 2 * h = 2 * (c + h) + 18) : c = 9 := 
by 
  sorry

end NUMINAMATH_GPT_cows_in_group_l284_28417


namespace NUMINAMATH_GPT_at_least_one_gt_one_of_sum_gt_two_l284_28412

theorem at_least_one_gt_one_of_sum_gt_two (x y : ℝ) (h : x + y > 2) : x > 1 ∨ y > 1 := 
by sorry

end NUMINAMATH_GPT_at_least_one_gt_one_of_sum_gt_two_l284_28412


namespace NUMINAMATH_GPT_exponent_problem_l284_28465

variable (x m n : ℝ)
variable (h1 : x^m = 3)
variable (h2 : x^n = 5)

theorem exponent_problem : x^(2 * m - 3 * n) = 9 / 125 :=
by 
  sorry

end NUMINAMATH_GPT_exponent_problem_l284_28465


namespace NUMINAMATH_GPT_smallest_prime_divisor_524_plus_718_l284_28488

theorem smallest_prime_divisor_524_plus_718 (x y : ℕ) (h1 : x = 5 ^ 24) (h2 : y = 7 ^ 18) :
  ∃ p : ℕ, Nat.Prime p ∧ p = 2 ∧ p ∣ (x + y) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_divisor_524_plus_718_l284_28488


namespace NUMINAMATH_GPT_grocer_display_proof_l284_28402

-- Define the arithmetic sequence conditions
def num_cans_in_display (n : ℕ) : Prop :=
  let a := 1
  let d := 2
  (n * n = 225) 

-- Prove the total weight is 1125 kg
def total_weight_supported (weight_per_can : ℕ) (total_cans : ℕ) : Prop :=
  (total_cans * weight_per_can = 1125)

-- State the main theorem combining the two proofs.
theorem grocer_display_proof (n weight_per_can total_cans : ℕ) :
  num_cans_in_display n → total_weight_supported weight_per_can total_cans → 
  n = 15 ∧ total_cans * weight_per_can = 1125 :=
by {
  sorry
}

end NUMINAMATH_GPT_grocer_display_proof_l284_28402


namespace NUMINAMATH_GPT_find_b_l284_28418

open Real

theorem find_b (b : ℝ) (h : b + ⌈b⌉ = 21.5) : b = 10.5 :=
sorry

end NUMINAMATH_GPT_find_b_l284_28418


namespace NUMINAMATH_GPT_infinite_series_sum_l284_28424

theorem infinite_series_sum :
  (∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1)) = 5 / 16 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l284_28424


namespace NUMINAMATH_GPT_unique_plants_count_1320_l284_28472

open Set

variable (X Y Z : Finset ℕ)

def total_plants_X : ℕ := 600
def total_plants_Y : ℕ := 480
def total_plants_Z : ℕ := 420
def shared_XY : ℕ := 60
def shared_YZ : ℕ := 70
def shared_XZ : ℕ := 80
def shared_XYZ : ℕ := 30

theorem unique_plants_count_1320 : X.card = total_plants_X →
                                Y.card = total_plants_Y →
                                Z.card = total_plants_Z →
                                (X ∩ Y).card = shared_XY →
                                (Y ∩ Z).card = shared_YZ →
                                (X ∩ Z).card = shared_XZ →
                                (X ∩ Y ∩ Z).card = shared_XYZ →
                                (X ∪ Y ∪ Z).card = 1320 := 
by {
  sorry
}

end NUMINAMATH_GPT_unique_plants_count_1320_l284_28472


namespace NUMINAMATH_GPT_convex_k_gons_count_l284_28425

noncomputable def number_of_convex_k_gons (n k : ℕ) : ℕ :=
  if h : n ≥ 2 * k then
    n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k))
  else
    0

theorem convex_k_gons_count (n k : ℕ) (h : n ≥ 2 * k) :
  number_of_convex_k_gons n k = n * Nat.factorial (n - k - 1) / (k * Nat.factorial k * Nat.factorial (n - 2 * k)) :=
by
  sorry

end NUMINAMATH_GPT_convex_k_gons_count_l284_28425


namespace NUMINAMATH_GPT_acetone_mass_percentage_O_l284_28427

-- Definition of atomic masses
def atomic_mass_C := 12.01
def atomic_mass_H := 1.008
def atomic_mass_O := 16.00

-- Definition of the molar mass of acetone
def molar_mass_acetone := (3 * atomic_mass_C) + (6 * atomic_mass_H) + atomic_mass_O

-- Definition of mass percentage of oxygen in acetone
def mass_percentage_O_acetone := (atomic_mass_O / molar_mass_acetone) * 100

theorem acetone_mass_percentage_O : mass_percentage_O_acetone = 27.55 := by sorry

end NUMINAMATH_GPT_acetone_mass_percentage_O_l284_28427


namespace NUMINAMATH_GPT_alfred_gain_percent_l284_28413

-- Definitions based on the conditions
def purchase_price : ℝ := 4700
def repair_costs : ℝ := 800
def selling_price : ℝ := 6000

-- Lean statement to prove gain percent
theorem alfred_gain_percent :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 9.09 := by
  sorry

end NUMINAMATH_GPT_alfred_gain_percent_l284_28413


namespace NUMINAMATH_GPT_converse_opposite_l284_28460

theorem converse_opposite (x y : ℝ) : (x + y = 0) → (y = -x) :=
by
  sorry

end NUMINAMATH_GPT_converse_opposite_l284_28460


namespace NUMINAMATH_GPT_work_required_to_pump_liquid_l284_28480

/-- Calculation of work required to pump a liquid of density ρ out of a parabolic boiler. -/
theorem work_required_to_pump_liquid
  (ρ g H a : ℝ)
  (h_pos : 0 < H)
  (a_pos : 0 < a) :
  ∃ (A : ℝ), A = (π * ρ * g * H^3) / (6 * a^2) :=
by
  -- TODO: Provide the proof.
  sorry

end NUMINAMATH_GPT_work_required_to_pump_liquid_l284_28480


namespace NUMINAMATH_GPT_popton_school_bus_total_toes_l284_28411

-- Define the number of toes per hand for each race
def toes_per_hand_hoopit : ℕ := 3
def toes_per_hand_neglart : ℕ := 2
def toes_per_hand_zentorian : ℕ := 4

-- Define the number of hands for each race
def hands_per_hoopit : ℕ := 4
def hands_per_neglart : ℕ := 5
def hands_per_zentorian : ℕ := 6

-- Define the number of students from each race on the bus
def num_hoopits : ℕ := 7
def num_neglarts : ℕ := 8
def num_zentorians : ℕ := 5

-- Calculate the total number of toes on the bus
def total_toes_on_bus : ℕ :=
  num_hoopits * (toes_per_hand_hoopit * hands_per_hoopit) +
  num_neglarts * (toes_per_hand_neglart * hands_per_neglart) +
  num_zentorians * (toes_per_hand_zentorian * hands_per_zentorian)

-- Theorem stating the number of toes on the bus
theorem popton_school_bus_total_toes : total_toes_on_bus = 284 :=
by
  sorry

end NUMINAMATH_GPT_popton_school_bus_total_toes_l284_28411


namespace NUMINAMATH_GPT_kaleb_gave_boxes_l284_28473

theorem kaleb_gave_boxes (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (given_boxes : ℕ)
  (h1 : total_boxes = 14) 
  (h2 : pieces_per_box = 6) 
  (h3 : pieces_left = 54) :
  given_boxes = 5 :=
by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_kaleb_gave_boxes_l284_28473


namespace NUMINAMATH_GPT_trajectory_center_of_C_number_of_lines_l_l284_28401

noncomputable def trajectory_equation : Prop :=
  ∃ (a b : ℝ), a = 4 ∧ b^2 = 12 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def line_count : Prop :=
  ∀ (k m : ℤ), 
  ∃ (num_lines : ℕ), 
  (∀ (x : ℝ), (3 + 4 * k^2) * x^2 + 8 * k * m * x + 4 * m^2 - 48 = 0 → num_lines = 9 ∨ num_lines = 0) ∧
  (∀ (x : ℝ), (3 - k^2) * x^2 - 2 * k * m * x - m^2 - 12 = 0 → num_lines = 9 ∨ num_lines = 0)

theorem trajectory_center_of_C :
  trajectory_equation :=
sorry

theorem number_of_lines_l :
  line_count :=
sorry

end NUMINAMATH_GPT_trajectory_center_of_C_number_of_lines_l_l284_28401


namespace NUMINAMATH_GPT_average_growth_rate_l284_28433

theorem average_growth_rate (x : ℝ) :
  (7200 * (1 + x)^2 = 8712) → x = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_average_growth_rate_l284_28433


namespace NUMINAMATH_GPT_alpha_identity_l284_28430

theorem alpha_identity (α : ℝ) (hα : α ≠ 0) (h_tan : Real.tan α = -α) : 
    (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_alpha_identity_l284_28430


namespace NUMINAMATH_GPT_tax_calculation_l284_28493

theorem tax_calculation 
  (total_earnings : ℕ) 
  (deductions : ℕ) 
  (tax_paid : ℕ) 
  (tax_rate_10 : ℚ) 
  (tax_rate_20 : ℚ) 
  (taxable_income : ℕ)
  (X : ℕ)
  (h_total_earnings : total_earnings = 100000)
  (h_deductions : deductions = 30000)
  (h_tax_paid : tax_paid = 12000)
  (h_tax_rate_10 : tax_rate_10 = 10 / 100)
  (h_tax_rate_20 : tax_rate_20 = 20 / 100)
  (h_taxable_income : taxable_income = total_earnings - deductions)
  (h_tax_equation : tax_paid = (tax_rate_10 * X) + (tax_rate_20 * (taxable_income - X))) :
  X = 20000 := 
sorry

end NUMINAMATH_GPT_tax_calculation_l284_28493


namespace NUMINAMATH_GPT_find_triples_l284_28487

theorem find_triples (a b c : ℕ) (h₁ : a ≥ b) (h₂ : b ≥ c) (h₃ : a^3 + 9 * b^2 + 9 * c + 7 = 1997) :
  (a = 10 ∧ b = 10 ∧ c = 10) :=
by sorry

end NUMINAMATH_GPT_find_triples_l284_28487


namespace NUMINAMATH_GPT_polygon_sides_l284_28492

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 360) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l284_28492


namespace NUMINAMATH_GPT_perpendicular_lines_l284_28407

theorem perpendicular_lines (a : ℝ) : (x + 2*y + 1 = 0) ∧ (ax + y - 2 = 0) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l284_28407


namespace NUMINAMATH_GPT_trajectory_equation_minimum_AB_l284_28451

/-- Let a moving circle \( C \) passes through the point \( F(0, 1) \).
    The center of the circle \( C \), denoted as \( (x, y) \), is above the \( x \)-axis and the
    distance from \( (x, y) \) to \( F \) is greater than its distance to the \( x \)-axis by 1.
    We aim to prove that the trajectory of the center is \( x^2 = 4y \). -/
theorem trajectory_equation {x y : ℝ} (h : y > 0) (hCF : Real.sqrt (x^2 + (y - 1)^2) - y = 1) : 
  x^2 = 4 * y :=
sorry

/-- Suppose \( A \) and \( B \) are two distinct points on the curve \( x^2 = 4y \). 
    The tangents at \( A \) and \( B \) intersect at \( P \), and \( AP \perp BP \). 
    Then the minimum value of \( |AB| \) is 4. -/
theorem minimum_AB {x₁ x₂ : ℝ} 
  (h₁ : y₁ = (x₁^2) / 4) (h₂ : y₂ = (x₂^2) / 4)
  (h_perp : x₁ * x₂ = -4) : 
  ∃ (d : ℝ), d ≥ 0 ∧ d = 4 :=
sorry

end NUMINAMATH_GPT_trajectory_equation_minimum_AB_l284_28451


namespace NUMINAMATH_GPT_flat_fee_l284_28479

theorem flat_fee (f n : ℝ) (h1 : f + 4 * n = 320) (h2 : f + 7 * n = 530) : f = 40 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_flat_fee_l284_28479


namespace NUMINAMATH_GPT_smallest_number_of_players_l284_28478

theorem smallest_number_of_players :
  ∃ n, n ≡ 1 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 4 [MOD 6] ∧ ∃ m, n = m * m ∧ ∀ k, (k ≡ 1 [MOD 3] ∧ k ≡ 2 [MOD 4] ∧ k ≡ 4 [MOD 6] ∧ ∃ m, k = m * m) → k ≥ n :=
sorry

end NUMINAMATH_GPT_smallest_number_of_players_l284_28478


namespace NUMINAMATH_GPT_maximum_value_is_16_l284_28470

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
(x^2 - 2 * x * y + 2 * y^2) * (x^2 - 2 * x * z + 2 * z^2) * (y^2 - 2 * y * z + 2 * z^2)

theorem maximum_value_is_16 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  maximum_value x y z ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_is_16_l284_28470


namespace NUMINAMATH_GPT_solution_set_inequality_l284_28404

theorem solution_set_inequality (a b : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → x^2 + a * x + b ≤ 0) :
  a * b = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_inequality_l284_28404
