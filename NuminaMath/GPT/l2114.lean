import Mathlib

namespace NUMINAMATH_GPT_solve_triangle_l2114_211469

variable {A B C : ℝ}
variable {a b c : ℝ}

noncomputable def sin_B_plus_pi_four (a b c : ℝ) : ℝ :=
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  let sin_B := Real.sqrt (1 - cos_B^2)
  sin_B * Real.sqrt 2 / 2 + cos_B * Real.sqrt 2 / 2

theorem solve_triangle 
  (a b c : ℝ)
  (h1 : b = 2 * Real.sqrt 5)
  (h2 : c = 3)
  (h3 : 3 * a * (a^2 + b^2 - c^2) / (2 * a * b) = 2 * c * (b^2 + c^2 - a^2) / (2 * b * c)) :
  a = Real.sqrt 5 ∧ 
  sin_B_plus_pi_four a b c = Real.sqrt 10 / 10 :=
by 
  sorry

end NUMINAMATH_GPT_solve_triangle_l2114_211469


namespace NUMINAMATH_GPT_div_seven_and_sum_factors_l2114_211417

theorem div_seven_and_sum_factors (a b c : ℤ) (h : (a = 0 ∨ b = 0 ∨ c = 0) ∧ ¬(a = 0 ∧ b = 0 ∧ c = 0)) :
  ∃ k : ℤ, (a + b + c)^7 - a^7 - b^7 - c^7 = k * 7 * (a + b) * (b + c) * (c + a) :=
by
  sorry

end NUMINAMATH_GPT_div_seven_and_sum_factors_l2114_211417


namespace NUMINAMATH_GPT_jumping_bug_ways_l2114_211432

-- Define the problem with given conditions and required answer
theorem jumping_bug_ways :
  let starting_position := 0
  let ending_position := 3
  let jumps := 5
  let jump_options := [1, -1]
  (∃ (jump_seq : Fin jumps → ℤ), (∀ i, jump_seq i ∈ jump_options ∧ (List.sum (List.ofFn jump_seq) = ending_position)) ∧
  (List.count (-1) (List.ofFn jump_seq) = 1)) →
  (∃ n : ℕ, n = 5) :=
by
  sorry  -- Proof to be completed

end NUMINAMATH_GPT_jumping_bug_ways_l2114_211432


namespace NUMINAMATH_GPT_cylinder_original_radius_l2114_211411

theorem cylinder_original_radius
  (r : ℝ)
  (h_original : ℝ := 4)
  (h_increased : ℝ := 3 * h_original)
  (volume_eq : π * (r + 8)^2 * h_original = π * r^2 * h_increased) :
  r = 4 + 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_cylinder_original_radius_l2114_211411


namespace NUMINAMATH_GPT_problem_statement_l2114_211427

variable (x y z a b c : ℝ)

-- Conditions
def condition1 := x / a + y / b + z / c = 5
def condition2 := a / x + b / y + c / z = 0

-- Proof statement
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end NUMINAMATH_GPT_problem_statement_l2114_211427


namespace NUMINAMATH_GPT_avg_monthly_bill_over_6_months_l2114_211403

theorem avg_monthly_bill_over_6_months :
  ∀ (avg_first_4_months avg_last_2_months : ℝ), 
  avg_first_4_months = 30 → 
  avg_last_2_months = 24 → 
  (4 * avg_first_4_months + 2 * avg_last_2_months) / 6 = 28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_avg_monthly_bill_over_6_months_l2114_211403


namespace NUMINAMATH_GPT_total_hours_charged_l2114_211406

variable (K P M : ℕ)

theorem total_hours_charged (h1 : P = 2 * K) (h2 : P = M / 3) (h3 : M = K + 80) : K + P + M = 144 := 
by
  sorry

end NUMINAMATH_GPT_total_hours_charged_l2114_211406


namespace NUMINAMATH_GPT_terminal_side_in_third_quadrant_l2114_211481

-- Define the conditions
def sin_condition (α : Real) : Prop := Real.sin α < 0
def tan_condition (α : Real) : Prop := Real.tan α > 0

-- State the theorem
theorem terminal_side_in_third_quadrant (α : Real) (h1 : sin_condition α) (h2 : tan_condition α) : α ∈ Set.Ioo (π / 2) π :=
  sorry

end NUMINAMATH_GPT_terminal_side_in_third_quadrant_l2114_211481


namespace NUMINAMATH_GPT_at_most_n_diameters_l2114_211479

theorem at_most_n_diameters {n : ℕ} (h : n ≥ 3) (points : Fin n → ℝ × ℝ) (d : ℝ) 
  (hd : ∀ i j, dist (points i) (points j) ≤ d) :
  ∃ (diameters : Fin n → Fin n), 
    (∀ i, dist (points i) (points (diameters i)) = d) ∧
    (∀ i j, (dist (points i) (points j) = d) → 
      (∃ k, k = i ∨ k = j → diameters k = if k = i then j else i)) :=
sorry

end NUMINAMATH_GPT_at_most_n_diameters_l2114_211479


namespace NUMINAMATH_GPT_find_plane_through_points_and_perpendicular_l2114_211471

-- Definitions for points and plane conditions
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def point1 : Point3D := ⟨2, -2, 2⟩
def point2 : Point3D := ⟨0, 2, -1⟩

def normal_vector_of_given_plane : Point3D := ⟨2, -1, 2⟩

-- Lean 4 statement
theorem find_plane_through_points_and_perpendicular :
  ∃ (A B C D : ℤ), 
  (∀ (p : Point3D), (p = point1 ∨ p = point2) → A * p.x + B * p.y + C * p.z + D = 0) ∧
  (A * 2 + B * -1 + C * 2 = 0) ∧ 
  A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (A = 5 ∧ B = -2 ∧ C = 6 ∧ D = -26) :=
by
  sorry

end NUMINAMATH_GPT_find_plane_through_points_and_perpendicular_l2114_211471


namespace NUMINAMATH_GPT_number_of_chairs_in_first_row_l2114_211409

-- Define the number of chairs in each row
def chairs_in_second_row := 23
def chairs_in_third_row := 32
def chairs_in_fourth_row := 41
def chairs_in_fifth_row := 50
def chairs_in_sixth_row := 59

-- Define the pattern increment
def increment := 9

-- Define a function to calculate the number of chairs in a given row, given the increment pattern
def chairs_in_row (n : Nat) : Nat :=
if n = 1 then (chairs_in_second_row - increment)
else if n = 2 then chairs_in_second_row
else if n = 3 then chairs_in_third_row
else if n = 4 then chairs_in_fourth_row
else if n = 5 then chairs_in_fifth_row
else if n = 6 then chairs_in_sixth_row
else chairs_in_second_row + (n - 2) * increment

-- The theorem to prove: The number of chairs in the first row is 14
theorem number_of_chairs_in_first_row : chairs_in_row 1 = 14 :=
  by sorry

end NUMINAMATH_GPT_number_of_chairs_in_first_row_l2114_211409


namespace NUMINAMATH_GPT_possible_values_of_a_l2114_211402

variable (a : ℝ)
def A : Set ℝ := { x | x^2 ≠ 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem possible_values_of_a (h : (A ∪ B a) = A) : a = 1 ∨ a = -1 ∨ a = 0 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l2114_211402


namespace NUMINAMATH_GPT_tangent_line_at_point_e_tangent_line_from_origin_l2114_211404

-- Problem 1
theorem tangent_line_at_point_e (x y : ℝ) (h : y = Real.exp x) (h_e : x = Real.exp 1) :
    (Real.exp x) * x - y - Real.exp (x + 1) = 0 :=
sorry

-- Problem 2
theorem tangent_line_from_origin (x y : ℝ) (h : y = Real.exp x) :
    x = 1 →  Real.exp x * x - y = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_e_tangent_line_from_origin_l2114_211404


namespace NUMINAMATH_GPT_maximum_side_length_of_triangle_l2114_211452

theorem maximum_side_length_of_triangle (a b c : ℕ) (h_diff: a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_perimeter: a + b + c = 30)
  (h_triangle_inequality_1: a + b > c) 
  (h_triangle_inequality_2: a + c > b) 
  (h_triangle_inequality_3: b + c > a) : 
  c ≤ 14 :=
sorry

end NUMINAMATH_GPT_maximum_side_length_of_triangle_l2114_211452


namespace NUMINAMATH_GPT_boxes_needed_l2114_211405

-- Define the given conditions

def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def total_pencils : ℕ := red_pencils + blue_pencils + green_pencils + yellow_pencils
def pencils_per_box : ℕ := 20

-- Lean theorem statement to prove the number of boxes needed is 8

theorem boxes_needed : total_pencils / pencils_per_box = 8 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_boxes_needed_l2114_211405


namespace NUMINAMATH_GPT_last_two_digits_of_power_sequence_l2114_211435

noncomputable def power_sequence (n : ℕ) : ℤ :=
  (Int.sqrt 29 + Int.sqrt 21)^(2 * n) + (Int.sqrt 29 - Int.sqrt 21)^(2 * n)

theorem last_two_digits_of_power_sequence :
  (power_sequence 992) % 100 = 71 := by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_power_sequence_l2114_211435


namespace NUMINAMATH_GPT_max_value_f_diff_l2114_211407

open Real

noncomputable def f (A ω : ℝ) (x : ℝ) := A * sin (ω * x + π / 6) - 1

theorem max_value_f_diff {A ω : ℝ} (hA : A > 0) (hω : ω > 0)
  (h_sym : (π / 2) = π / (2 * ω))
  (h_initial : f A ω (π / 6) = 1) :
  ∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) →
  (f A ω x1 - f A ω x2 ≤ 3) :=
sorry

end NUMINAMATH_GPT_max_value_f_diff_l2114_211407


namespace NUMINAMATH_GPT_quadratic_root_in_l2114_211464

variable (a b c m : ℝ)

theorem quadratic_root_in (ha : a > 0) (hm : m > 0) 
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_root_in_l2114_211464


namespace NUMINAMATH_GPT_apples_for_48_oranges_l2114_211470

theorem apples_for_48_oranges (o a : ℕ) (h : 8 * o = 6 * a) (ho : o = 48) : a = 36 :=
by
  sorry

end NUMINAMATH_GPT_apples_for_48_oranges_l2114_211470


namespace NUMINAMATH_GPT_books_a_count_l2114_211454

theorem books_a_count (A B : ℕ) (h1 : A + B = 20) (h2 : A = B + 4) : A = 12 :=
by
  sorry

end NUMINAMATH_GPT_books_a_count_l2114_211454


namespace NUMINAMATH_GPT_son_work_rate_l2114_211475

noncomputable def man_work_rate := 1/10
noncomputable def combined_work_rate := 1/5

theorem son_work_rate :
  ∃ S : ℝ, man_work_rate + S = combined_work_rate ∧ S = 1/10 := sorry

end NUMINAMATH_GPT_son_work_rate_l2114_211475


namespace NUMINAMATH_GPT_complex_sum_abs_eq_1_or_3_l2114_211478

open Complex

theorem complex_sum_abs_eq_1_or_3
  (a b c : ℂ)
  (ha : abs a = 1)
  (hb : abs b = 1)
  (hc : abs c = 1)
  (h : a^3 / (b * c) + b^3 / (a * c) + c^3 / (a * b) = 1) :
  ∃ r : ℝ, (r = 1 ∨ r = 3) ∧ abs (a + b + c) = r :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_complex_sum_abs_eq_1_or_3_l2114_211478


namespace NUMINAMATH_GPT_four_digit_palindrome_square_count_l2114_211422

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end NUMINAMATH_GPT_four_digit_palindrome_square_count_l2114_211422


namespace NUMINAMATH_GPT_Archer_catch_total_fish_l2114_211445

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end NUMINAMATH_GPT_Archer_catch_total_fish_l2114_211445


namespace NUMINAMATH_GPT_rate_of_mangoes_is_60_l2114_211426

-- Define the conditions
def kg_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 9
def total_paid : ℕ := 1100

-- Define the cost of grapes and total cost
def cost_of_grapes : ℕ := kg_grapes * rate_per_kg_grapes
def cost_of_mangoes : ℕ := total_paid - cost_of_grapes
def rate_per_kg_mangoes : ℕ := cost_of_mangoes / kg_mangoes

-- Prove that the rate of mangoes per kg is 60
theorem rate_of_mangoes_is_60 : rate_per_kg_mangoes = 60 := by
  -- Here we would provide the proof
  sorry

end NUMINAMATH_GPT_rate_of_mangoes_is_60_l2114_211426


namespace NUMINAMATH_GPT_find_abc_l2114_211440

theorem find_abc (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h_eq : 10 * a + 11 * b + c = 25) : a = 0 ∧ b = 2 ∧ c = 3 := 
sorry

end NUMINAMATH_GPT_find_abc_l2114_211440


namespace NUMINAMATH_GPT_find_carbon_atoms_l2114_211408

variable (n : ℕ)
variable (molecular_weight : ℝ := 124.0)
variable (weight_Cu : ℝ := 63.55)
variable (weight_C : ℝ := 12.01)
variable (weight_O : ℝ := 16.00)
variable (num_Cu : ℕ := 1)
variable (num_O : ℕ := 3)

theorem find_carbon_atoms 
  (h : molecular_weight = (num_Cu * weight_Cu) + (n * weight_C) + (num_O * weight_O)) : 
  n = 1 :=
sorry

end NUMINAMATH_GPT_find_carbon_atoms_l2114_211408


namespace NUMINAMATH_GPT_projection_of_a_on_b_l2114_211499

theorem projection_of_a_on_b (a b : ℝ) (θ : ℝ) 
  (ha : |a| = 2) 
  (hb : |b| = 1)
  (hθ : θ = 60) : 
  (|a| * Real.cos (θ * Real.pi / 180)) = 1 := 
sorry

end NUMINAMATH_GPT_projection_of_a_on_b_l2114_211499


namespace NUMINAMATH_GPT_positive_difference_of_numbers_l2114_211400

theorem positive_difference_of_numbers (x : ℝ) (h : (30 + x) / 2 = 34) : abs (x - 30) = 8 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_numbers_l2114_211400


namespace NUMINAMATH_GPT_no_such_n_exists_l2114_211460

theorem no_such_n_exists : ∀ (n : ℕ), n ≥ 1 → ¬ Prime (n^n - 4 * n + 3) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_no_such_n_exists_l2114_211460


namespace NUMINAMATH_GPT_solve_for_x_l2114_211439

theorem solve_for_x : (1 / 3 - 1 / 4) * 2 = 1 / 6 :=
by
  -- Sorry is used to skip the proof; the proof steps are not included.
  sorry

end NUMINAMATH_GPT_solve_for_x_l2114_211439


namespace NUMINAMATH_GPT_molecular_weight_of_N2O5_l2114_211430

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_atoms_N : ℕ := 2
def num_atoms_O : ℕ := 5
def molecular_weight_N2O5 : ℝ := (num_atoms_N * atomic_weight_N) + (num_atoms_O * atomic_weight_O)

theorem molecular_weight_of_N2O5 : molecular_weight_N2O5 = 108.02 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_N2O5_l2114_211430


namespace NUMINAMATH_GPT_x_varies_as_z_raised_to_n_power_l2114_211455

noncomputable def x_varies_as_cube_of_y (k y : ℝ) : ℝ := k * y ^ 3
noncomputable def y_varies_as_cube_root_of_z (j z : ℝ) : ℝ := j * z ^ (1/3 : ℝ)

theorem x_varies_as_z_raised_to_n_power (k j z : ℝ) :
  ∃ n : ℝ, x_varies_as_cube_of_y k (y_varies_as_cube_root_of_z j z) = (k * j^3) * z ^ n ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_x_varies_as_z_raised_to_n_power_l2114_211455


namespace NUMINAMATH_GPT_fraction_to_decimal_terminating_l2114_211449

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_terminating_l2114_211449


namespace NUMINAMATH_GPT_find_divisor_l2114_211448

theorem find_divisor : ∃ D : ℕ, 14698 = (D * 89) + 14 ∧ D = 165 :=
by
  use 165
  sorry

end NUMINAMATH_GPT_find_divisor_l2114_211448


namespace NUMINAMATH_GPT_general_form_of_line_l2114_211498

theorem general_form_of_line (x y : ℝ) 
  (passes_through_A : ∃ y, 2 = y)          -- Condition 1: passes through A(-2, 2)
  (same_y_intercept : ∃ y, 6 = y)          -- Condition 2: same y-intercept as y = x + 6
  : 2 * x - y + 6 = 0 := 
sorry

end NUMINAMATH_GPT_general_form_of_line_l2114_211498


namespace NUMINAMATH_GPT_nearest_integer_to_power_sum_l2114_211457

theorem nearest_integer_to_power_sum :
  let x := (3 + Real.sqrt 5)
  Int.floor ((x ^ 4) + 1 / 2) = 752 :=
by
  sorry

end NUMINAMATH_GPT_nearest_integer_to_power_sum_l2114_211457


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2114_211447

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) (S_n : ℕ → ℝ)
  (h₁ : S_n 3 = a₁ + a₁ * q + a₁ * q ^ 2)
  (h₂ : S_n 2 = a₁ + a₁ * q)
  (h₃ : S_n 3 / S_n 2 = 3 / 2) :
  q = 1 ∨ q = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2114_211447


namespace NUMINAMATH_GPT_square_ratios_l2114_211425

/-- 
  Given two squares with areas ratio 16:49, 
  prove that the ratio of their perimeters is 4:7,
  and the ratio of the sum of their perimeters to the sum of their areas is 84:13.
-/
theorem square_ratios (s₁ s₂ : ℝ) 
  (h₁ : s₁^2 / s₂^2 = 16 / 49) :
  (s₁ / s₂ = 4 / 7) ∧ ((4 * (s₁ + s₂)) / (s₁^2 + s₂^2) = 84 / 13) :=
by {
  sorry
}

end NUMINAMATH_GPT_square_ratios_l2114_211425


namespace NUMINAMATH_GPT_find_m_l2114_211423

-- Define the function and conditions
def power_function (x : ℝ) (m : ℕ) : ℝ := x^(m - 2)

theorem find_m (m : ℕ) (x : ℝ) (h1 : 0 < m) (h2 : power_function 0 m = 0 → false) : m = 1 ∨ m = 2 :=
by
  sorry -- Skip the proof

end NUMINAMATH_GPT_find_m_l2114_211423


namespace NUMINAMATH_GPT_xy_z_eq_inv_sqrt2_l2114_211465

noncomputable def f (t : ℝ) : ℝ := (Real.sqrt 2) * t + 1 / ((Real.sqrt 2) * t)

theorem xy_z_eq_inv_sqrt2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (Real.sqrt 2) * x + 1 / ((Real.sqrt 2) * x) 
      + (Real.sqrt 2) * y + 1 / ((Real.sqrt 2) * y) 
      + (Real.sqrt 2) * z + 1 / ((Real.sqrt 2) * z) 
      = 6 - 2 * (Real.sqrt (2 * x)) * abs (y - z) 
            - (Real.sqrt (2 * y)) * (x - z) ^ 2 
            - (Real.sqrt (2 * z)) * (Real.sqrt (abs (x - y)))) :
  x = y ∧ y = z ∧ z = 1 / (Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_xy_z_eq_inv_sqrt2_l2114_211465


namespace NUMINAMATH_GPT_relay_go_match_outcomes_l2114_211472

theorem relay_go_match_outcomes : (Nat.choose 14 7) = 3432 := by
  sorry

end NUMINAMATH_GPT_relay_go_match_outcomes_l2114_211472


namespace NUMINAMATH_GPT_total_duration_in_seconds_l2114_211466

theorem total_duration_in_seconds :
  let hours_in_seconds := 2 * 3600
  let minutes_in_seconds := 45 * 60
  let extra_seconds := 30
  hours_in_seconds + minutes_in_seconds + extra_seconds = 9930 := by
  sorry

end NUMINAMATH_GPT_total_duration_in_seconds_l2114_211466


namespace NUMINAMATH_GPT_integral_sign_negative_l2114_211436

open Topology

-- Define the problem
theorem integral_sign_negative {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) (h_lt : ∀ x ∈ Set.Icc a b, f x < 0) (h_ab : a < b) :
  ∫ x in a..b, f x < 0 := 
sorry

end NUMINAMATH_GPT_integral_sign_negative_l2114_211436


namespace NUMINAMATH_GPT_gcd_fx_x_l2114_211459

-- Let x be an instance of ℤ
variable (x : ℤ)

-- Define that x is a multiple of 46200
def is_multiple_of_46200 := ∃ k : ℤ, x = 46200 * k

-- Define the function f(x) = (3x + 5)(5x + 3)(11x + 6)(x + 11)
def f (x : ℤ) := (3 * x + 5) * (5 * x + 3) * (11 * x + 6) * (x + 11)

-- The statement to prove
theorem gcd_fx_x (h : is_multiple_of_46200 x) : Int.gcd (f x) x = 990 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_gcd_fx_x_l2114_211459


namespace NUMINAMATH_GPT_area_y_eq_x2_y_eq_x3_l2114_211415

noncomputable section

open Real

def area_closed_figure_between_curves : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (x^2 - x^3)

theorem area_y_eq_x2_y_eq_x3 :
  area_closed_figure_between_curves = 1 / 12 := by
  sorry

end NUMINAMATH_GPT_area_y_eq_x2_y_eq_x3_l2114_211415


namespace NUMINAMATH_GPT_find_t_l2114_211480

variables (c o u n t s : ℕ)

theorem find_t (h1 : c + o = u) 
               (h2 : u + n = t)
               (h3 : t + c = s)
               (h4 : o + n + s = 18)
               (hz : c > 0) (ho : o > 0) (hu : u > 0) (hn : n > 0) (ht : t > 0) (hs : s > 0) : 
               t = 9 := 
by
  sorry

end NUMINAMATH_GPT_find_t_l2114_211480


namespace NUMINAMATH_GPT_apples_remaining_l2114_211477

-- Define the initial condition of the number of apples on the tree
def initial_apples : ℕ := 7

-- Define the number of apples picked by Rachel
def picked_apples : ℕ := 4

-- Proof goal: the number of apples remaining on the tree is 3
theorem apples_remaining : (initial_apples - picked_apples = 3) :=
sorry

end NUMINAMATH_GPT_apples_remaining_l2114_211477


namespace NUMINAMATH_GPT_find_n_l2114_211486

-- Define the variables d, Q, r, m, and n
variables (d Q r m n : ℝ)

-- Define the conditions Q = d / ((1 + r)^n - m) and m < (1 + r)^n
def conditions (d Q r m n : ℝ) : Prop :=
  Q = d / ((1 + r)^n - m) ∧ m < (1 + r)^n

theorem find_n (d Q r m : ℝ) (h : conditions d Q r m n) : 
  n = (Real.log (d / Q + m)) / (Real.log (1 + r)) :=
sorry

end NUMINAMATH_GPT_find_n_l2114_211486


namespace NUMINAMATH_GPT_slope_of_line_l2114_211450

theorem slope_of_line (x : ℝ) : (2 * x + 1) = 2 :=
by sorry

end NUMINAMATH_GPT_slope_of_line_l2114_211450


namespace NUMINAMATH_GPT_find_complex_number_l2114_211412

open Complex

theorem find_complex_number (z : ℂ) (hz : z + Complex.abs z = Complex.ofReal 2 + 8 * Complex.I) : 
z = -15 + 8 * Complex.I := by sorry

end NUMINAMATH_GPT_find_complex_number_l2114_211412


namespace NUMINAMATH_GPT_simplify_and_find_ab_ratio_l2114_211474

-- Given conditions
def given_expression (k : ℤ) : ℤ := 10 * k + 15

-- Simplified form
def simplified_form (k : ℤ) : ℤ := 2 * k + 3

-- Proof problem statement
theorem simplify_and_find_ab_ratio
  (k : ℤ) :
  let a := 2
  let b := 3
  (10 * k + 15) / 5 = 2 * k + 3 → 
  (a:ℚ) / (b:ℚ) = 2 / 3 := sorry

end NUMINAMATH_GPT_simplify_and_find_ab_ratio_l2114_211474


namespace NUMINAMATH_GPT_value_of_y_l2114_211485

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l2114_211485


namespace NUMINAMATH_GPT_inverse_of_11_mod_1021_l2114_211443

theorem inverse_of_11_mod_1021 : ∃ x : ℕ, x < 1021 ∧ 11 * x ≡ 1 [MOD 1021] := by
  use 557
  -- We leave the proof as an exercise.
  sorry

end NUMINAMATH_GPT_inverse_of_11_mod_1021_l2114_211443


namespace NUMINAMATH_GPT_Maria_waist_size_correct_l2114_211442

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end NUMINAMATH_GPT_Maria_waist_size_correct_l2114_211442


namespace NUMINAMATH_GPT_problem_solution_l2114_211461

variable {x y z : ℝ}

/-- Suppose that x, y, and z are three positive numbers that satisfy the given conditions.
    Prove that z + 1/y = 13/77. --/
theorem problem_solution (h1 : x * y * z = 1)
                         (h2 : x + 1 / z = 8)
                         (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := 
  sorry

end NUMINAMATH_GPT_problem_solution_l2114_211461


namespace NUMINAMATH_GPT_sum_areas_of_circles_l2114_211493

theorem sum_areas_of_circles (r s t : ℝ) 
  (h1 : r + s = 6)
  (h2 : r + t = 8)
  (h3 : s + t = 10) : 
  r^2 * Real.pi + s^2 * Real.pi + t^2 * Real.pi = 56 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_sum_areas_of_circles_l2114_211493


namespace NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l2114_211462

theorem ratio_of_first_term_to_common_difference (a d : ℕ) (h : 15 * a + 105 * d = 3 * (5 * a + 10 * d)) : a = 5 * d :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_first_term_to_common_difference_l2114_211462


namespace NUMINAMATH_GPT_permutation_sum_eq_744_l2114_211431

open Nat

theorem permutation_sum_eq_744 (n : ℕ) (h1 : n ≠ 0) (h2 : n + 3 ≤ 2 * n) (h3 : n + 1 ≤ 4) :
  choose (2 * n) (n + 3) + choose 4 (n + 1) = 744 := by
  sorry

end NUMINAMATH_GPT_permutation_sum_eq_744_l2114_211431


namespace NUMINAMATH_GPT_Olivia_steps_l2114_211441

def round_to_nearest_ten (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem Olivia_steps :
  let x := 57 + 68
  let y := x - 15
  round_to_nearest_ten y = 110 := 
by
  sorry

end NUMINAMATH_GPT_Olivia_steps_l2114_211441


namespace NUMINAMATH_GPT_solve_for_n_l2114_211437

-- Define the equation as a Lean expression
def equation (n : ℚ) : Prop :=
  (2 - n) / (n + 1) + (2 * n - 4) / (2 - n) = 1

theorem solve_for_n : ∃ n : ℚ, equation n ∧ n = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_solve_for_n_l2114_211437


namespace NUMINAMATH_GPT_not_divisible_by_q_plus_one_l2114_211497

theorem not_divisible_by_q_plus_one (q : ℕ) (hq_odd : q % 2 = 1) (hq_gt_two : q > 2) :
  ¬ (q + 1) ∣ ((q + 1) ^ ((q - 1) / 2) + 2) :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_q_plus_one_l2114_211497


namespace NUMINAMATH_GPT_sum_first_six_terms_arithmetic_seq_l2114_211496

theorem sum_first_six_terms_arithmetic_seq :
  ∃ a_1 d : ℤ, (a_1 + 3 * d = 7) ∧ (a_1 + 4 * d = 12) ∧ (a_1 + 5 * d = 17) ∧ 
  (6 * (2 * a_1 + 5 * d) / 2 = 27) :=
by
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_arithmetic_seq_l2114_211496


namespace NUMINAMATH_GPT_range_of_a_l2114_211456

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * a * x - 2 else x + 36 / x - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 2 a) ↔ (2 ≤ a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2114_211456


namespace NUMINAMATH_GPT_ratio_value_l2114_211416

theorem ratio_value (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_value_l2114_211416


namespace NUMINAMATH_GPT_box_cookies_count_l2114_211495

theorem box_cookies_count (cookies_per_bag : ℕ) (cookies_per_box : ℕ) :
  cookies_per_bag = 7 →
  8 * cookies_per_box = 9 * cookies_per_bag + 33 →
  cookies_per_box = 12 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_box_cookies_count_l2114_211495


namespace NUMINAMATH_GPT_geometric_sequence_l2114_211473

open Nat

-- Define the sequence and conditions for the problem
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {m p : ℕ}
variable (h1 : a 1 ≠ 0)
variable (h2 : ∀ n : ℕ, 2 * S (n + 1) - 3 * S n = 2 * a 1)
variable (h3 : S 0 = 0)
variable (h4 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
variable (h5 : a 1 ≥ m^(p-1))
variable (h6 : a p ≤ (m+1)^(p-1))

-- The theorem that we need to prove
theorem geometric_sequence (n : ℕ) : 
  (exists r : ℕ → ℕ, ∀ k : ℕ, a (k + 1) = r (k + 1) * a k) ∧ 
  (∀ k : ℕ, a k = sorry) := sorry

end NUMINAMATH_GPT_geometric_sequence_l2114_211473


namespace NUMINAMATH_GPT_find_smaller_number_l2114_211458

theorem find_smaller_number (a b : ℕ) (h1 : a + b = 45) (h2 : b = 4 * a) : a = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l2114_211458


namespace NUMINAMATH_GPT_acute_angle_sine_l2114_211414

theorem acute_angle_sine (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 0.58) : (π / 6) < α ∧ α < (π / 4) :=
by
  sorry

end NUMINAMATH_GPT_acute_angle_sine_l2114_211414


namespace NUMINAMATH_GPT_n_mod_5_division_of_grid_l2114_211490

theorem n_mod_5_division_of_grid (n : ℕ) :
  (∃ m : ℕ, n^2 = 4 + 5 * m) ↔ n % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_n_mod_5_division_of_grid_l2114_211490


namespace NUMINAMATH_GPT_smallest_prime_angle_l2114_211491

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_angle :
  ∃ (x : ℕ), is_prime x ∧ is_prime (2 * x) ∧ x + 2 * x = 90 ∧ x = 29 :=
by sorry

end NUMINAMATH_GPT_smallest_prime_angle_l2114_211491


namespace NUMINAMATH_GPT_final_temperature_l2114_211468

variable (initial_temp : ℝ := 40)
variable (double_temp : ℝ := initial_temp * 2)
variable (reduce_by_dad : ℝ := double_temp - 30)
variable (reduce_by_mother : ℝ := reduce_by_dad * 0.70)
variable (increase_by_sister : ℝ := reduce_by_mother + 24)

theorem final_temperature : increase_by_sister = 59 := by
  sorry

end NUMINAMATH_GPT_final_temperature_l2114_211468


namespace NUMINAMATH_GPT_cost_of_lunch_l2114_211413

-- Define the conditions: total amount and tip percentage
def total_amount : ℝ := 72.6
def tip_percentage : ℝ := 0.20

-- Define the proof problem
theorem cost_of_lunch (C : ℝ) (h : C + tip_percentage * C = total_amount) : C = 60.5 := 
sorry

end NUMINAMATH_GPT_cost_of_lunch_l2114_211413


namespace NUMINAMATH_GPT_correct_relationship_in_triangle_l2114_211492

theorem correct_relationship_in_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A + B) = Real.sin C :=
sorry

end NUMINAMATH_GPT_correct_relationship_in_triangle_l2114_211492


namespace NUMINAMATH_GPT_exists_visible_point_l2114_211482

open Nat -- to use natural numbers and their operations

def is_visible (x y : ℤ) : Prop :=
  Int.gcd x y = 1

theorem exists_visible_point (n : ℕ) (hn : n > 0) :
  ∃ a b : ℤ, is_visible a b ∧
  ∀ (P : ℤ × ℤ), (P ≠ (a, b) → (Int.sqrt ((P.fst - a) * (P.fst - a) + (P.snd - b) * (P.snd - b)) > n)) :=
sorry

end NUMINAMATH_GPT_exists_visible_point_l2114_211482


namespace NUMINAMATH_GPT_lcm_of_40_90_150_l2114_211428

-- Definition to calculate the Least Common Multiple of three numbers
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- Definitions for the given numbers
def n1 : ℕ := 40
def n2 : ℕ := 90
def n3 : ℕ := 150

-- The statement of the proof problem
theorem lcm_of_40_90_150 : lcm3 n1 n2 n3 = 1800 := by
  sorry

end NUMINAMATH_GPT_lcm_of_40_90_150_l2114_211428


namespace NUMINAMATH_GPT_quadratics_common_root_square_sum_6_l2114_211444

theorem quadratics_common_root_square_sum_6
  (a b c : ℝ)
  (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_common_root_1: ∃ x1, x1^2 + a * x1 + b = 0 ∧ x1^2 + b * x1 + c = 0)
  (h_common_root_2: ∃ x2, x2^2 + b * x2 + c = 0 ∧ x2^2 + c * x2 + a = 0)
  (h_common_root_3: ∃ x3, x3^2 + c * x3 + a = 0 ∧ x3^2 + a * x3 + b = 0) :
  a^2 + b^2 + c^2 = 6 :=
sorry

end NUMINAMATH_GPT_quadratics_common_root_square_sum_6_l2114_211444


namespace NUMINAMATH_GPT_magnitude_of_parallel_vector_l2114_211419

theorem magnitude_of_parallel_vector {x : ℝ} 
  (h_parallel : 2 / x = -1 / 3) : 
  (Real.sqrt (x^2 + 3^2)) = 3 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_magnitude_of_parallel_vector_l2114_211419


namespace NUMINAMATH_GPT_remainder_29_169_1990_mod_11_l2114_211453

theorem remainder_29_169_1990_mod_11 :
  (29 * 169 ^ 1990) % 11 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_29_169_1990_mod_11_l2114_211453


namespace NUMINAMATH_GPT_initial_amount_invested_l2114_211421

-- Conditions
def initial_investment : ℝ := 367.36
def annual_interest_rate : ℝ := 0.08
def accumulated_amount : ℝ := 500
def years : ℕ := 4

-- Required to prove that the initial investment satisfies the given equation
theorem initial_amount_invested :
  initial_investment * (1 + annual_interest_rate) ^ years = accumulated_amount :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_invested_l2114_211421


namespace NUMINAMATH_GPT_ferry_q_more_time_l2114_211484

variables (speed_ferry_p speed_ferry_q distance_ferry_p distance_ferry_q time_ferry_p time_ferry_q : ℕ)
  -- Conditions given in the problem
  (h1 : speed_ferry_p = 8)
  (h2 : time_ferry_p = 2)
  (h3 : distance_ferry_p = speed_ferry_p * time_ferry_p)
  (h4 : distance_ferry_q = 3 * distance_ferry_p)
  (h5 : speed_ferry_q = speed_ferry_p + 4)
  (h6 : time_ferry_q = distance_ferry_q / speed_ferry_q)

theorem ferry_q_more_time : time_ferry_q - time_ferry_p = 2 :=
by
  sorry

end NUMINAMATH_GPT_ferry_q_more_time_l2114_211484


namespace NUMINAMATH_GPT_compute_a_sq_sub_b_sq_l2114_211438

variables {a b : (ℝ × ℝ)}

-- Conditions
axiom a_nonzero : a ≠ (0, 0)
axiom b_nonzero : b ≠ (0, 0)
axiom a_add_b_eq_neg3_6 : a + b = (-3, 6)
axiom a_sub_b_eq_neg3_2 : a - b = (-3, 2)

-- Question and the correct answer
theorem compute_a_sq_sub_b_sq : (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 :=
by sorry

end NUMINAMATH_GPT_compute_a_sq_sub_b_sq_l2114_211438


namespace NUMINAMATH_GPT_weight_difference_l2114_211483

theorem weight_difference (brown black white grey : ℕ) 
  (h_brown : brown = 4)
  (h_white : white = 2 * brown)
  (h_grey : grey = black - 2)
  (avg_weight : (brown + black + white + grey) / 4 = 5): 
  (black - brown) = 1 := by
  sorry

end NUMINAMATH_GPT_weight_difference_l2114_211483


namespace NUMINAMATH_GPT_abc_inequality_l2114_211429

theorem abc_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < c) 
  (h4 : a * b * c = 1) 
  : 
  (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) := 
by 
  sorry

end NUMINAMATH_GPT_abc_inequality_l2114_211429


namespace NUMINAMATH_GPT_fill_bucket_completely_l2114_211463

theorem fill_bucket_completely (t : ℕ) : (2/3 : ℚ) * t = 100 → t = 150 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fill_bucket_completely_l2114_211463


namespace NUMINAMATH_GPT_line_through_point_l2114_211433

-- Definitions for conditions
def point : (ℝ × ℝ) := (1, 2)

-- Function to check if a line equation holds for the given form 
def is_line_eq (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

-- Main Lean theorem statement
theorem line_through_point (a b c : ℝ) :
  (∃ a b c, (is_line_eq a b c 1 2) ∧ 
           ((a = 1 ∧ b = 1 ∧ c = -3) ∨ (a = 2 ∧ b = -1 ∧ c = 0))) :=
sorry

end NUMINAMATH_GPT_line_through_point_l2114_211433


namespace NUMINAMATH_GPT_trisha_take_home_pay_l2114_211418

def hourly_wage : ℝ := 15
def hours_per_week : ℝ := 40
def weeks_per_year : ℝ := 52
def tax_rate : ℝ := 0.2

def annual_gross_pay : ℝ := hourly_wage * hours_per_week * weeks_per_year
def amount_withheld : ℝ := tax_rate * annual_gross_pay
def annual_take_home_pay : ℝ := annual_gross_pay - amount_withheld

theorem trisha_take_home_pay :
  annual_take_home_pay = 24960 := 
by
  sorry

end NUMINAMATH_GPT_trisha_take_home_pay_l2114_211418


namespace NUMINAMATH_GPT_aluminum_phosphate_molecular_weight_l2114_211476

theorem aluminum_phosphate_molecular_weight :
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  (Al + P + 4 * O) = 121.95 :=
by
  let Al := 26.98
  let P := 30.97
  let O := 16.00
  sorry

end NUMINAMATH_GPT_aluminum_phosphate_molecular_weight_l2114_211476


namespace NUMINAMATH_GPT_value_of_p_l2114_211446

theorem value_of_p (a : ℕ → ℚ) (m : ℕ) (p : ℚ)
  (h1 : a 1 = 111)
  (h2 : a 2 = 217)
  (h3 : ∀ n : ℕ, 3 ≤ n ∧ n ≤ m → a n = a (n - 2) - (n - p) / a (n - 1))
  (h4 : m = 220) :
  p = 110 / 109 :=
by
  sorry

end NUMINAMATH_GPT_value_of_p_l2114_211446


namespace NUMINAMATH_GPT_rain_third_day_l2114_211401

theorem rain_third_day (rain_day1 rain_day2 rain_day3 : ℕ)
  (h1 : rain_day1 = 4)
  (h2 : rain_day2 = 5 * rain_day1)
  (h3 : rain_day3 = (rain_day1 + rain_day2) - 6) : 
  rain_day3 = 18 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_rain_third_day_l2114_211401


namespace NUMINAMATH_GPT_arithmetic_mean_difference_l2114_211494

-- Definitions and conditions
variable (p q r : ℝ)
variable (h1 : (p + q) / 2 = 10)
variable (h2 : (q + r) / 2 = 26)

-- Theorem statement
theorem arithmetic_mean_difference : r - p = 32 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_mean_difference_l2114_211494


namespace NUMINAMATH_GPT_derivative_y_eq_l2114_211424

noncomputable def y (x : ℝ) : ℝ := 
  (3 / 2) * Real.log (Real.tanh (x / 2)) + Real.cosh x - (Real.cosh x) / (2 * (Real.sinh x)^2)

theorem derivative_y_eq :
  (deriv y x) = (Real.cosh x)^4 / (Real.sinh x)^3 :=
sorry

end NUMINAMATH_GPT_derivative_y_eq_l2114_211424


namespace NUMINAMATH_GPT_erwin_chocolates_weeks_l2114_211420

-- Define weekdays chocolates and weekends chocolates
def weekdays_chocolates := 2
def weekends_chocolates := 1

-- Define the total chocolates Erwin ate
def total_chocolates := 24

-- Define the number of weekdays and weekend days in a week
def weekdays := 5
def weekends := 2

-- Define the total chocolates Erwin eats in a week
def chocolates_per_week : Nat := (weekdays * weekdays_chocolates) + (weekends * weekends_chocolates)

-- Prove that Erwin finishes all chocolates in 2 weeks
theorem erwin_chocolates_weeks : (total_chocolates / chocolates_per_week) = 2 := by
  sorry

end NUMINAMATH_GPT_erwin_chocolates_weeks_l2114_211420


namespace NUMINAMATH_GPT_largest_even_of_sum_140_l2114_211488

theorem largest_even_of_sum_140 :
  ∃ (n : ℕ), 2 * n + 2 * (n + 1) + 2 * (n + 2) + 2 * (n + 3) = 140 ∧ 2 * (n + 3) = 38 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_of_sum_140_l2114_211488


namespace NUMINAMATH_GPT_total_cookies_needed_l2114_211487

-- Define the conditions
def cookies_per_person : ℝ := 24.0
def number_of_people : ℝ := 6.0

-- Define the goal
theorem total_cookies_needed : cookies_per_person * number_of_people = 144.0 :=
by
  sorry

end NUMINAMATH_GPT_total_cookies_needed_l2114_211487


namespace NUMINAMATH_GPT_bananas_to_oranges_equivalence_l2114_211467

noncomputable def bananas_to_apples (bananas apples : ℕ) : Prop :=
  4 * apples = 3 * bananas

noncomputable def apples_to_oranges (apples oranges : ℕ) : Prop :=
  5 * oranges = 2 * apples

theorem bananas_to_oranges_equivalence (x y : ℕ) (hx : bananas_to_apples 24 x) (hy : apples_to_oranges x y) :
  y = 72 / 10 := by
  sorry

end NUMINAMATH_GPT_bananas_to_oranges_equivalence_l2114_211467


namespace NUMINAMATH_GPT_find_fx_plus_1_l2114_211434

theorem find_fx_plus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x - 1) = x^2 + 4 * x - 5) : 
  ∀ x : ℤ, f (x + 1) = x^2 + 8 * x + 7 :=
sorry

end NUMINAMATH_GPT_find_fx_plus_1_l2114_211434


namespace NUMINAMATH_GPT_probability_at_least_3_out_of_6_babies_speak_l2114_211410

noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

noncomputable def prob_at_least_k (total : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob total i p)

theorem probability_at_least_3_out_of_6_babies_speak :
  prob_at_least_k 6 3 (2/5) = 7120/15625 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_3_out_of_6_babies_speak_l2114_211410


namespace NUMINAMATH_GPT_add_fractions_l2114_211489

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end NUMINAMATH_GPT_add_fractions_l2114_211489


namespace NUMINAMATH_GPT_number_of_tie_games_l2114_211451

def total_games (n_teams: ℕ) (games_per_matchup: ℕ) : ℕ :=
  (n_teams * (n_teams - 1) / 2) * games_per_matchup

def theoretical_max_points (total_games: ℕ) (points_per_win: ℕ): ℕ :=
  total_games * points_per_win

def actual_total_points (lions: ℕ) (tigers: ℕ) (mounties: ℕ) (royals: ℕ): ℕ :=
  lions + tigers + mounties + royals

def tie_games (theoretical_points: ℕ) (actual_points: ℕ) (points_per_tie: ℕ): ℕ :=
  (theoretical_points - actual_points) / points_per_tie

theorem number_of_tie_games
  (n_teams: ℕ)
  (games_per_matchup: ℕ)
  (points_per_win: ℕ)
  (points_per_tie: ℕ)
  (lions: ℕ)
  (tigers: ℕ)
  (mounties: ℕ)
  (royals: ℕ)
  (h_teams: n_teams = 4)
  (h_games: games_per_matchup = 4)
  (h_points_win: points_per_win = 3)
  (h_points_tie: points_per_tie = 2)
  (h_lions: lions = 22)
  (h_tigers: tigers = 19)
  (h_mounties: mounties = 14)
  (h_royals: royals = 12) :
  tie_games (theoretical_max_points (total_games n_teams games_per_matchup) points_per_win) 
  (actual_total_points lions tigers mounties royals) points_per_tie = 5 :=
by
  rw [h_teams, h_games, h_points_win, h_points_tie, h_lions, h_tigers, h_mounties, h_royals]
  simp [total_games, theoretical_max_points, actual_total_points, tie_games]
  sorry

end NUMINAMATH_GPT_number_of_tie_games_l2114_211451
