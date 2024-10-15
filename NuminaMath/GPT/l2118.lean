import Mathlib

namespace NUMINAMATH_GPT_renata_donation_l2118_211878

variable (D L : ℝ)

theorem renata_donation : ∃ D : ℝ, 
  (10 - D + 90 - L - 2 + 65 = 94) ↔ D = 4 :=
by
  sorry

end NUMINAMATH_GPT_renata_donation_l2118_211878


namespace NUMINAMATH_GPT_max_value_squared_of_ratio_l2118_211877

-- Definition of positive real numbers with given conditions
variables (a b x y : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 

-- Main statement
theorem max_value_squared_of_ratio 
  (h_ge : a ≥ b)
  (h_eq_1 : a ^ 2 + y ^ 2 = b ^ 2 + x ^ 2)
  (h_eq_2 : b ^ 2 + x ^ 2 = (a - x) ^ 2 + (b + y) ^ 2)
  (h_range_x : 0 ≤ x ∧ x < a)
  (h_range_y : 0 ≤ y ∧ y < b)
  (h_additional_x : x = a - 2 * b)
  (h_additional_y : y = b / 2) : 
  (a / b) ^ 2 = 4 / 9 := 
sorry

end NUMINAMATH_GPT_max_value_squared_of_ratio_l2118_211877


namespace NUMINAMATH_GPT_min_quadratic_expression_l2118_211833

theorem min_quadratic_expression:
  ∀ x : ℝ, x = 3 → (x^2 - 6 * x + 5 = -4) :=
by
  sorry

end NUMINAMATH_GPT_min_quadratic_expression_l2118_211833


namespace NUMINAMATH_GPT_solve_equation_l2118_211851

-- Definitions based on the conditions
def equation (a b c d : ℕ) : Prop :=
  2^a * 3^b - 5^c * 7^d = 1

def nonnegative_integers (a b c d : ℕ) : Prop := 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof to show the exact solutions
theorem solve_equation :
  (∃ (a b c d : ℕ), nonnegative_integers a b c d ∧ equation a b c d) ↔ 
  ( (1, 0, 0, 0) = (1, 0, 0, 0) ∨ (3, 0, 0, 1) = (3, 0, 0, 1) ∨ 
    (1, 1, 1, 0) = (1, 1, 1, 0) ∨ (2, 2, 1, 1) = (2, 2, 1, 1) ) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2118_211851


namespace NUMINAMATH_GPT_length_width_difference_l2118_211841

noncomputable def width : ℝ := Real.sqrt (588 / 8)
noncomputable def length : ℝ := 4 * width
noncomputable def difference : ℝ := length - width

theorem length_width_difference : difference = 25.722 := by
  sorry

end NUMINAMATH_GPT_length_width_difference_l2118_211841


namespace NUMINAMATH_GPT_classroom_books_l2118_211806

theorem classroom_books (students_group1 students_group2 books_per_student_group1 books_per_student_group2 books_brought books_lost : ℕ)
  (h1 : students_group1 = 20)
  (h2 : books_per_student_group1 = 15)
  (h3 : students_group2 = 25)
  (h4 : books_per_student_group2 = 10)
  (h5 : books_brought = 30)
  (h6 : books_lost = 7) :
  (students_group1 * books_per_student_group1 + students_group2 * books_per_student_group2 + books_brought - books_lost) = 573 := by
  sorry

end NUMINAMATH_GPT_classroom_books_l2118_211806


namespace NUMINAMATH_GPT_hexagon_area_within_rectangle_of_5x4_l2118_211884

-- Define the given conditions
def is_rectangle (length width : ℝ) := length > 0 ∧ width > 0

def vertices_touch_midpoints (length width : ℝ) (hexagon_area : ℝ) : Prop :=
  let rectangle_area := length * width
  let triangle_area := (1 / 2) * (length / 2) * (width / 2)
  let total_triangle_area := 4 * triangle_area
  rectangle_area - total_triangle_area = hexagon_area

-- Formulate the main statement to be proved
theorem hexagon_area_within_rectangle_of_5x4 : 
  vertices_touch_midpoints 5 4 10 := 
by
  -- Proof is omitted for this theorem
  sorry

end NUMINAMATH_GPT_hexagon_area_within_rectangle_of_5x4_l2118_211884


namespace NUMINAMATH_GPT_sin_A_plus_B_lt_sin_A_add_sin_B_l2118_211850

variable {A B : ℝ}
variable (A_pos : 0 < A)
variable (B_pos : 0 < B)
variable (AB_sum_pi : A + B < π)

theorem sin_A_plus_B_lt_sin_A_add_sin_B (a b : ℝ) (h1 : a = Real.sin (A + B)) (h2 : b = Real.sin A + Real.sin B) : 
  a < b := by
  sorry

end NUMINAMATH_GPT_sin_A_plus_B_lt_sin_A_add_sin_B_l2118_211850


namespace NUMINAMATH_GPT_largest_common_divisor_414_345_l2118_211831

theorem largest_common_divisor_414_345 : ∃ d, d ∣ 414 ∧ d ∣ 345 ∧ 
                                      (∀ e, e ∣ 414 ∧ e ∣ 345 → e ≤ d) ∧ d = 69 :=
by 
  sorry

end NUMINAMATH_GPT_largest_common_divisor_414_345_l2118_211831


namespace NUMINAMATH_GPT_rotate_and_translate_line_l2118_211849

theorem rotate_and_translate_line :
  let initial_line (x : ℝ) := 3 * x
  let rotated_line (x : ℝ) := - (1 / 3) * x
  let translated_line (x : ℝ) := - (1 / 3) * (x - 1)

  ∀ x : ℝ, translated_line x = - (1 / 3) * x + (1 / 3) := 
by
  intros
  simp
  sorry

end NUMINAMATH_GPT_rotate_and_translate_line_l2118_211849


namespace NUMINAMATH_GPT_probability_white_given_black_drawn_l2118_211801

-- Definitions based on the conditions
def num_white : ℕ := 3
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

def P (n : ℕ) : ℚ := n / total_balls

-- Event A: drawing a black ball on the first draw
def PA : ℚ := P num_black

-- Event B: drawing a white ball on the second draw
def PB_given_A : ℚ := num_white / (total_balls - 1)

-- Theorem statement
theorem probability_white_given_black_drawn :
  (PA * PB_given_A) / PA = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_white_given_black_drawn_l2118_211801


namespace NUMINAMATH_GPT_evaluate_expression_l2118_211862

theorem evaluate_expression : 
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = (137 / 52) :=
by
  -- We need to evaluate from the innermost part to the outermost,
  -- as noted in the problem statement and solution steps.
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2118_211862


namespace NUMINAMATH_GPT_convention_handshakes_l2118_211821

-- Introducing the conditions
def companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_reps : ℕ := companies * reps_per_company
def shakes_per_rep : ℕ := total_reps - 1 - (reps_per_company - 1)
def handshakes : ℕ := (total_reps * shakes_per_rep) / 2

-- Statement of the proof
theorem convention_handshakes : handshakes = 160 :=
by
  sorry  -- Proof is not required in this task.

end NUMINAMATH_GPT_convention_handshakes_l2118_211821


namespace NUMINAMATH_GPT_evaluate_ninth_roots_of_unity_product_l2118_211873

theorem evaluate_ninth_roots_of_unity_product : 
  (3 - Complex.exp (2 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (4 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (6 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (8 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (10 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (12 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (14 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (16 * Real.pi * Complex.I / 9)) 
  = 9841 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_ninth_roots_of_unity_product_l2118_211873


namespace NUMINAMATH_GPT_tangent_line_equation_even_derived_l2118_211800

def f (x a : ℝ) : ℝ := x^3 + (a - 2) * x^2 + a * x - 1

def f' (x a : ℝ) : ℝ := 3 * x^2 + 2 * (a - 2) * x + a

theorem tangent_line_equation_even_derived (a : ℝ) (h : ∀ x : ℝ, f' x a = f' (-x) a) :
  5 * 1 - (f 1 a) - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_even_derived_l2118_211800


namespace NUMINAMATH_GPT_parabola_focus_to_directrix_distance_correct_l2118_211802

def parabola_focus_to_directrix_distance (a : ℕ) (y x : ℝ) : Prop :=
  y^2 = 2 * x → a = 2 →  1 = 1

theorem parabola_focus_to_directrix_distance_correct :
  ∀ (a : ℕ) (y x : ℝ), parabola_focus_to_directrix_distance a y x :=
by
  unfold parabola_focus_to_directrix_distance
  intros
  sorry

end NUMINAMATH_GPT_parabola_focus_to_directrix_distance_correct_l2118_211802


namespace NUMINAMATH_GPT_arith_seq_geom_seq_l2118_211828

theorem arith_seq_geom_seq (a : ℕ → ℝ) (d : ℝ) (h : d ≠ 0) 
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : (a 9)^2 = a 5 * a 15) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_arith_seq_geom_seq_l2118_211828


namespace NUMINAMATH_GPT_sum_roots_l2118_211811

theorem sum_roots :
  (∀ (x : ℂ), (3 * x^3 - 2 * x^2 + 4 * x - 15 = 0) → 
              x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  (∀ (x : ℂ), (4 * x^3 - 16 * x^2 - 28 * x + 35 = 0) → 
              x = y₁ ∨ x = y₂ ∨ x = y₃) →
  (x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 14 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_roots_l2118_211811


namespace NUMINAMATH_GPT_gasoline_amount_added_l2118_211836

noncomputable def initial_fill (capacity : ℝ) : ℝ := (3 / 4) * capacity
noncomputable def final_fill (capacity : ℝ) : ℝ := (9 / 10) * capacity
noncomputable def gasoline_added (capacity : ℝ) : ℝ := final_fill capacity - initial_fill capacity

theorem gasoline_amount_added :
  ∀ (capacity : ℝ), capacity = 24 → gasoline_added capacity = 3.6 :=
  by
    intros capacity h
    rw [h]
    have initial_fill_24 : initial_fill 24 = 18 := by norm_num [initial_fill]
    have final_fill_24 : final_fill 24 = 21.6 := by norm_num [final_fill]
    have gasoline_added_24 : gasoline_added 24 = 3.6 :=
      by rw [gasoline_added, initial_fill_24, final_fill_24]; norm_num
    exact gasoline_added_24

end NUMINAMATH_GPT_gasoline_amount_added_l2118_211836


namespace NUMINAMATH_GPT_hyperbola_eccentricity_eq_two_l2118_211840

theorem hyperbola_eccentricity_eq_two :
  (∀ x y : ℝ, ((x^2 / 2) - (y^2 / 6) = 1) → 
    let a_squared := 2
    let b_squared := 6
    let a := Real.sqrt a_squared
    let b := Real.sqrt b_squared
    let e := Real.sqrt (1 + b_squared / a_squared)
    e = 2) := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_eq_two_l2118_211840


namespace NUMINAMATH_GPT_fault_line_movement_l2118_211899

theorem fault_line_movement (total_movement: ℝ) (past_year: ℝ) (prev_year: ℝ) (total_eq: total_movement = 6.5) (past_eq: past_year = 1.25) :
  prev_year = 5.25 := by
  sorry

end NUMINAMATH_GPT_fault_line_movement_l2118_211899


namespace NUMINAMATH_GPT_length_of_bridge_is_correct_l2118_211839

noncomputable def length_of_inclined_bridge (initial_speed : ℕ) (time : ℕ) (acceleration : ℕ) : ℚ :=
  (1 / 60) * (time * initial_speed + (time * (time - 1)) / 2)

theorem length_of_bridge_is_correct : 
  length_of_inclined_bridge 10 18 1 = 5.55 := 
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_is_correct_l2118_211839


namespace NUMINAMATH_GPT_find_ages_l2118_211886

theorem find_ages (F S : ℕ) (h1 : F + 2 * S = 110) (h2 : 3 * F = 186) :
  F = 62 ∧ S = 24 := by
  sorry

end NUMINAMATH_GPT_find_ages_l2118_211886


namespace NUMINAMATH_GPT_jellybean_ratio_l2118_211826

-- Define the conditions
def Matilda_jellybeans := 420
def Steve_jellybeans := 84
def Matt_jellybeans := 10 * Steve_jellybeans

-- State the theorem to prove the ratio
theorem jellybean_ratio : (Matilda_jellybeans : Nat) / (Matt_jellybeans : Nat) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_ratio_l2118_211826


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l2118_211823

theorem simplify_and_evaluate_expr (x y : ℚ) (h1 : x = -3/8) (h2 : y = 4) :
  (x - 2 * y) ^ 2 + (x - 2 * y) * (x + 2 * y) - 2 * x * (x - y) = 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l2118_211823


namespace NUMINAMATH_GPT_move_symmetric_point_left_l2118_211879

-- Define the original point and the operations
def original_point : ℝ × ℝ := (-2, 3)

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

-- Prove the resulting point after the operations
theorem move_symmetric_point_left : move_left (symmetric_point original_point) 2 = (0, -3) :=
by
  sorry

end NUMINAMATH_GPT_move_symmetric_point_left_l2118_211879


namespace NUMINAMATH_GPT_three_digit_number_divisible_by_7_l2118_211890

theorem three_digit_number_divisible_by_7
  (a b : ℕ)
  (h1 : (a + b) % 7 = 0) :
  (100 * a + 10 * b + a) % 7 = 0 :=
sorry

end NUMINAMATH_GPT_three_digit_number_divisible_by_7_l2118_211890


namespace NUMINAMATH_GPT_find_W_l2118_211892

def digit_sum_eq (X Y Z W : ℕ) : Prop := X * 10 + Y + Z * 10 + X = W * 10 + X
def digit_diff_eq (X Y Z : ℕ) : Prop := X * 10 + Y - (Z * 10 + X) = X
def is_digit (n : ℕ) : Prop := n < 10

theorem find_W (X Y Z W : ℕ) (h1 : digit_sum_eq X Y Z W) (h2 : digit_diff_eq X Y Z) 
  (hX : is_digit X) (hY : is_digit Y) (hZ : is_digit Z) (hW : is_digit W) : W = 0 := 
sorry

end NUMINAMATH_GPT_find_W_l2118_211892


namespace NUMINAMATH_GPT_second_person_days_l2118_211875

theorem second_person_days (h1 : 2 * (1 : ℝ) / 8 = 1) 
                           (h2 : 1 / 24 + x / 24 = 1 / 8) : x = 1 / 12 :=
sorry

end NUMINAMATH_GPT_second_person_days_l2118_211875


namespace NUMINAMATH_GPT_problem_solution_set_l2118_211860

-- Definitions and conditions according to the given problem
def odd_function_domain := {x : ℝ | x ≠ 0}
def function_condition1 (f : ℝ → ℝ) (x : ℝ) : Prop := x > 0 → deriv f x < (3 * f x) / x
def function_condition2 (f : ℝ → ℝ) : Prop := f 1 = 1 / 2
def function_condition3 (f : ℝ → ℝ) : Prop := ∀ x, f (2 * x) = 2 * f x

-- Main proof statement
theorem problem_solution_set (f : ℝ → ℝ)
  (odd_function : ∀ x, f (-x) = -f x)
  (dom : ∀ x, x ∈ odd_function_domain → f x ≠ 0)
  (cond1 : ∀ x, function_condition1 f x)
  (cond2 : function_condition2 f)
  (cond3 : function_condition3 f) :
  {x : ℝ | f x / (4 * x) < 2 * x^2} = {x : ℝ | x < -1 / 4} ∪ {x : ℝ | x > 1 / 4} :=
sorry

end NUMINAMATH_GPT_problem_solution_set_l2118_211860


namespace NUMINAMATH_GPT_rectangle_area_l2118_211852

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 40) : l * b = 75 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2118_211852


namespace NUMINAMATH_GPT_ramesh_paid_price_l2118_211844

theorem ramesh_paid_price {P : ℝ} (h1 : P = 18880 / 1.18) : 
  (0.80 * P + 125 + 250) = 13175 :=
by sorry

end NUMINAMATH_GPT_ramesh_paid_price_l2118_211844


namespace NUMINAMATH_GPT_sequence_sum_l2118_211837

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 1

-- Define the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- State the theorem
theorem sequence_sum : (b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) + b_n (a_n 5) + b_n (a_n 6)) = 126 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_l2118_211837


namespace NUMINAMATH_GPT_min_value_symmetry_l2118_211863

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_symmetry (a b c : ℝ) (h_a : a > 0) (h_symm : ∀ x : ℝ, quadratic a b c (2 + x) = quadratic a b c (2 - x)) : 
  quadratic a b c 2 < quadratic a b c 1 ∧ quadratic a b c 1 < quadratic a b c 4 := 
sorry

end NUMINAMATH_GPT_min_value_symmetry_l2118_211863


namespace NUMINAMATH_GPT_scout_weekend_earnings_l2118_211865

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end NUMINAMATH_GPT_scout_weekend_earnings_l2118_211865


namespace NUMINAMATH_GPT_Olivia_spent_25_dollars_l2118_211819

theorem Olivia_spent_25_dollars
    (initial_amount : ℕ)
    (final_amount : ℕ)
    (spent_amount : ℕ)
    (h_initial : initial_amount = 54)
    (h_final : final_amount = 29)
    (h_spent : spent_amount = initial_amount - final_amount) :
    spent_amount = 25 := by
  sorry

end NUMINAMATH_GPT_Olivia_spent_25_dollars_l2118_211819


namespace NUMINAMATH_GPT_range_of_a_l2118_211861

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := {x | abs (x - 2) ≤ a}
def set_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

lemma disjoint_sets (A B : Set ℝ) : A ∩ B = ∅ :=
  sorry

theorem range_of_a (h : set_A a ∩ set_B = ∅) : a < 1 :=
  by
  sorry

end NUMINAMATH_GPT_range_of_a_l2118_211861


namespace NUMINAMATH_GPT_cube_surface_area_l2118_211832

theorem cube_surface_area (a : ℕ) (h : a = 2) : 6 * a^2 = 24 := 
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2118_211832


namespace NUMINAMATH_GPT_correct_system_l2118_211855

def system_of_equations (x y : ℤ) : Prop :=
  (5 * x + 45 = y) ∧ (7 * x - 3 = y)

theorem correct_system : ∃ x y : ℤ, system_of_equations x y :=
sorry

end NUMINAMATH_GPT_correct_system_l2118_211855


namespace NUMINAMATH_GPT_total_cats_in_meow_and_paw_l2118_211882

-- Define the conditions
def CatsInCatCafeCool : Nat := 5
def CatsInCatCafePaw : Nat := 2 * CatsInCatCafeCool
def CatsInCatCafeMeow : Nat := 3 * CatsInCatCafePaw

-- Define the total number of cats in Cat Cafe Meow and Cat Cafe Paw
def TotalCats : Nat := CatsInCatCafeMeow + CatsInCatCafePaw

-- The theorem stating the problem
theorem total_cats_in_meow_and_paw : TotalCats = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_cats_in_meow_and_paw_l2118_211882


namespace NUMINAMATH_GPT_find_a9_for_geo_seq_l2118_211870

noncomputable def geo_seq_a_3_a_13_positive_common_ratio_2 (a_3 a_9 a_13 : ℕ) : Prop :=
  (a_3 * a_13 = 16) ∧ (a_3 > 0) ∧ (a_9 > 0) ∧ (a_13 > 0) ∧ (forall (n₁ n₂ : ℕ), a_9 = a_3 * 2 ^ 6)

theorem find_a9_for_geo_seq (a_3 a_9 a_13 : ℕ) 
  (h : geo_seq_a_3_a_13_positive_common_ratio_2 a_3 a_9 a_13) :
  a_9 = 8 :=
  sorry

end NUMINAMATH_GPT_find_a9_for_geo_seq_l2118_211870


namespace NUMINAMATH_GPT_least_t_geometric_progression_exists_l2118_211812

open Real

theorem least_t_geometric_progression_exists :
  ∃ (t : ℝ),
  (∃ (α : ℝ), 0 < α ∧ α < π / 3 ∧
             (arcsin (sin α) = α ∧
              arcsin (sin (3 * α)) = 3 * α ∧
              arcsin (sin (8 * α)) = 8 * α) ∧
              (arcsin (sin (t * α)) = (some_ratio) * (arcsin (sin (8 * α))) )) ∧ 
   0 < t := 
by 
  sorry

end NUMINAMATH_GPT_least_t_geometric_progression_exists_l2118_211812


namespace NUMINAMATH_GPT_toy_cost_l2118_211885

-- Definitions based on the conditions in part a)
def initial_amount : ℕ := 57
def spent_amount : ℕ := 49
def remaining_amount : ℕ := initial_amount - spent_amount
def number_of_toys : ℕ := 2

-- Statement to prove that each toy costs 4 dollars
theorem toy_cost :
  (remaining_amount / number_of_toys) = 4 :=
by
  sorry

end NUMINAMATH_GPT_toy_cost_l2118_211885


namespace NUMINAMATH_GPT_seconds_in_minutes_l2118_211883

-- Define the concepts of minutes and seconds
def minutes (m : ℝ) : ℝ := m

def seconds (s : ℝ) : ℝ := s

-- Define the given values
def conversion_factor : ℝ := 60 -- seconds in one minute

def given_minutes : ℝ := 12.5

-- State the theorem
theorem seconds_in_minutes : seconds (given_minutes * conversion_factor) = 750 := 
by
sorry

end NUMINAMATH_GPT_seconds_in_minutes_l2118_211883


namespace NUMINAMATH_GPT_integer_satisfies_inequality_l2118_211868

theorem integer_satisfies_inequality (n : ℤ) : 
  (3 : ℚ) / 10 < n / 20 ∧ n / 20 < 2 / 5 → n = 7 :=
sorry

end NUMINAMATH_GPT_integer_satisfies_inequality_l2118_211868


namespace NUMINAMATH_GPT_ratio_of_lengths_l2118_211848

variables (l1 l2 l3 : ℝ)

theorem ratio_of_lengths (h1 : l2 = (1/2) * (l1 + l3))
                         (h2 : l2^3 = (6/13) * (l1^3 + l3^3)) :
  (l1 / l3) = (7 / 5) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_lengths_l2118_211848


namespace NUMINAMATH_GPT_product_of_solutions_l2118_211895

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end NUMINAMATH_GPT_product_of_solutions_l2118_211895


namespace NUMINAMATH_GPT_quadratic_no_rational_solution_l2118_211807

theorem quadratic_no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ∀ (x : ℚ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_rational_solution_l2118_211807


namespace NUMINAMATH_GPT_hyperbola_equation_foci_shared_l2118_211889

theorem hyperbola_equation_foci_shared :
  ∃ m : ℝ, (∃ c : ℝ, c = 2 * Real.sqrt 2 ∧ 
              ∃ a b : ℝ, a^2 = 12 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2) ∧ 
    (c = 2 * Real.sqrt 2 → (∃ a b : ℝ, a^2 = m ∧ b^2 = m - 8 ∧ c^2 = a^2 + b^2)) → 
  (∃ m : ℝ, m = 7) := 
sorry

end NUMINAMATH_GPT_hyperbola_equation_foci_shared_l2118_211889


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l2118_211880

theorem cyclic_sum_inequality
  (a b c d e : ℝ)
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ e)
  (h6 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l2118_211880


namespace NUMINAMATH_GPT_problem_statement_l2118_211824

open Set

variable (a : ℕ)
variable (A : Set ℕ := {2, 3, 4})
variable (B : Set ℕ := {a + 2, a})

theorem problem_statement (hB : B ⊆ A) : (A \ B) = {3} :=
sorry

end NUMINAMATH_GPT_problem_statement_l2118_211824


namespace NUMINAMATH_GPT_average_speed_round_trip_l2118_211898

noncomputable def average_speed (d : ℝ) (v_to v_from : ℝ) : ℝ :=
  let time_to := d / v_to
  let time_from := d / v_from
  let total_time := time_to + time_from
  let total_distance := 2 * d
  total_distance / total_time

theorem average_speed_round_trip (d : ℝ) :
  average_speed d 60 40 = 48 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l2118_211898


namespace NUMINAMATH_GPT_skating_minutes_needed_l2118_211829

-- Define the conditions
def minutes_per_day (day: ℕ) : ℕ :=
  if day ≤ 4 then 80 else if day ≤ 6 then 100 else 0

-- Define total skating time up to 6 days
def total_time_six_days := (4 * 80) + (2 * 100)

-- Prove that Gage needs to skate 180 minutes on the seventh day
theorem skating_minutes_needed : 
  (total_time_six_days + x = 7 * 100) → x = 180 :=
by sorry

end NUMINAMATH_GPT_skating_minutes_needed_l2118_211829


namespace NUMINAMATH_GPT_empty_solution_set_implies_a_range_l2118_211842

def f (a x: ℝ) := x^2 + (1 - a) * x - a

theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬ (f a (f a x) < 0)) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by
  sorry

end NUMINAMATH_GPT_empty_solution_set_implies_a_range_l2118_211842


namespace NUMINAMATH_GPT_greatest_integer_radius_l2118_211815

theorem greatest_integer_radius (r : ℝ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
sorry

end NUMINAMATH_GPT_greatest_integer_radius_l2118_211815


namespace NUMINAMATH_GPT_triangle_isosceles_l2118_211893

-- Definitions involved: Triangle, Circumcircle, Angle Bisector, Isosceles Triangle
universe u

structure Triangle (α : Type u) :=
  (A B C : α)

structure Circumcircle (α : Type u) :=
  (triangle : Triangle α)

structure AngleBisector (α : Type u) :=
  (A : α)
  (triangle : Triangle α)

def IsoscelesTriangle {α : Type u} (P Q R : α) : Prop :=
  ∃ (p₁ p₂ p₃ : α), (p₁ = P ∧ p₂ = Q ∧ p₃ = R) ∧
                  ((∃ θ₁ θ₂, θ₁ + θ₂ = 90) → (∃ θ₃ θ₂, θ₃ + θ₂ = 90))

theorem triangle_isosceles {α : Type u} (T : Triangle α) (S : α)
  (h1 : Circumcircle α) (h2 : AngleBisector α) :
  IsoscelesTriangle T.B T.C S :=
by
  sorry

end NUMINAMATH_GPT_triangle_isosceles_l2118_211893


namespace NUMINAMATH_GPT_distance_between_ann_and_glenda_l2118_211816

def ann_distance : ℝ := 
  let speed1 := 6
  let time1 := 1
  let speed2 := 8
  let time2 := 1
  let break1 := 0
  let speed3 := 4
  let time3 := 1
  speed1 * time1 + speed2 * time2 + break1 * 0 + speed3 * time3

def glenda_distance : ℝ := 
  let speed1 := 8
  let time1 := 1
  let speed2 := 5
  let time2 := 1
  let break1 := 0
  let speed3 := 9
  let back_time := 0.5
  let back_distance := speed3 * back_time
  let continue_time := 0.5
  let continue_distance := speed3 * continue_time
  speed1 * time1 + speed2 * time2 + break1 * 0 + (-back_distance) + continue_distance

theorem distance_between_ann_and_glenda : 
  ann_distance + glenda_distance = 35.5 := 
by 
  sorry

end NUMINAMATH_GPT_distance_between_ann_and_glenda_l2118_211816


namespace NUMINAMATH_GPT_value_of_a_l2118_211867

def star (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem value_of_a (a : ℝ) (h : star a 3 = 15) : a = 11 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2118_211867


namespace NUMINAMATH_GPT_element_in_set_l2118_211876

theorem element_in_set : 1 ∈ ({0, 1} : Set ℕ) := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_element_in_set_l2118_211876


namespace NUMINAMATH_GPT_find_m_l2118_211803

-- Definitions for the sets
def setA (x : ℝ) : Prop := -2 < x ∧ x < 8
def setB (m : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3

-- Condition on the intersection
def intersection (m : ℝ) (a b : ℝ) (x : ℝ) : Prop := 2 * m - 1 < x ∧ x < m + 3 ∧ -2 < x ∧ x < 8

-- Theorem statement
theorem find_m (m a b : ℝ) (h₀ : b - a = 3) (h₁ : ∀ x, intersection m a b x ↔ (a < x ∧ x < b)) : m = -2 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l2118_211803


namespace NUMINAMATH_GPT_weight_of_each_bag_of_flour_l2118_211881

-- Definitions based on the given conditions
def cookies_eaten_by_Jim : ℕ := 15
def cookies_left : ℕ := 105
def total_cookies : ℕ := cookies_eaten_by_Jim + cookies_left

def cookies_per_dozen : ℕ := 12
def pounds_per_dozen : ℕ := 2

def dozens_of_cookies := total_cookies / cookies_per_dozen
def total_pounds_of_flour := dozens_of_cookies * pounds_per_dozen

def bags_of_flour : ℕ := 4

-- Question to be proved
theorem weight_of_each_bag_of_flour : total_pounds_of_flour / bags_of_flour = 5 := by
  sorry

end NUMINAMATH_GPT_weight_of_each_bag_of_flour_l2118_211881


namespace NUMINAMATH_GPT_find_ratio_l2118_211817
   
   -- Given Conditions
   variable (S T F : ℝ)
   variable (H1 : 30 + S + T + F = 450)
   variable (H2 : S > 30)
   variable (H3 : T > S)
   variable (H4 : F > T)
   
   -- The goal is to find the ratio S / 30
   theorem find_ratio :
     ∃ r : ℝ, r = S / 30 ↔ false :=
   by
     sorry
   
end NUMINAMATH_GPT_find_ratio_l2118_211817


namespace NUMINAMATH_GPT_circle_radius_tangent_to_parabola_l2118_211853

theorem circle_radius_tangent_to_parabola (a : ℝ) (b r : ℝ) :
  (∀ x : ℝ, y = 4 * x ^ 2) ∧ 
  (b = a ^ 2 / 4) ∧ 
  (∀ x : ℝ, x ^ 2 + (4 * x ^ 2 - b) ^ 2 = r ^ 2)  → 
  r = a ^ 2 / 4 := 
  sorry

end NUMINAMATH_GPT_circle_radius_tangent_to_parabola_l2118_211853


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2118_211825

theorem distance_between_foci_of_hyperbola {x y : ℝ} (h : x ^ 2 - 4 * y ^ 2 = 4) :
  ∃ c : ℝ, 2 * c = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2118_211825


namespace NUMINAMATH_GPT_number_of_cases_for_Ds_hearts_l2118_211869

theorem number_of_cases_for_Ds_hearts (hA : 5 ≤ 13) (hB : 4 ≤ 13) (dist : 52 % 4 = 0) : 
  ∃ n, n = 5 ∧ 0 ≤ n ∧ n ≤ 13 := sorry

end NUMINAMATH_GPT_number_of_cases_for_Ds_hearts_l2118_211869


namespace NUMINAMATH_GPT_students_and_ticket_price_l2118_211891

theorem students_and_ticket_price (students teachers ticket_price : ℕ) 
  (h1 : students % 5 = 0)
  (h2 : (students + teachers) * (ticket_price / 2) = 1599)
  (h3 : ∃ n, ticket_price = 2 * n) 
  (h4 : teachers = 1) :
  students = 40 ∧ ticket_price = 78 := 
by
  sorry

end NUMINAMATH_GPT_students_and_ticket_price_l2118_211891


namespace NUMINAMATH_GPT_bisector_length_is_correct_l2118_211843

noncomputable def length_of_bisector_of_angle_C
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) : ℝ := 3.2

theorem bisector_length_is_correct
    (BC AC : ℝ)
    (angleC : ℝ)
    (hBC : BC = 5)
    (hAC : AC = 7)
    (hAngleC: angleC = 80) :
    length_of_bisector_of_angle_C BC AC angleC hBC hAC hAngleC = 3.2 := by
  sorry

end NUMINAMATH_GPT_bisector_length_is_correct_l2118_211843


namespace NUMINAMATH_GPT_bird_needs_more_twigs_l2118_211854

variable (base_twigs : ℕ := 12)
variable (additional_twigs_per_base : ℕ := 6)
variable (fraction_dropped : ℚ := 1/3)

theorem bird_needs_more_twigs (tree_dropped : ℕ) : 
  tree_dropped = (additional_twigs_per_base * base_twigs) * 1/3 →
  (base_twigs * additional_twigs_per_base - tree_dropped) = 48 :=
by
  sorry

end NUMINAMATH_GPT_bird_needs_more_twigs_l2118_211854


namespace NUMINAMATH_GPT_camel_cost_l2118_211846

variables {C H O E G Z : ℕ} 

-- conditions
axiom h1 : 10 * C = 24 * H
axiom h2 : 16 * H = 4 * O
axiom h3 : 6 * O = 4 * E
axiom h4 : 3 * E = 15 * G
axiom h5 : 8 * G = 20 * Z
axiom h6 : 12 * E = 180000

-- goal
theorem camel_cost : C = 6000 :=
by sorry

end NUMINAMATH_GPT_camel_cost_l2118_211846


namespace NUMINAMATH_GPT_megan_picture_shelves_l2118_211894

def books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 8
def total_books : ℕ := 70
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := total_books - total_mystery_books
def picture_shelves : ℕ := total_picture_books / books_per_shelf

theorem megan_picture_shelves : picture_shelves = 2 := 
by sorry

end NUMINAMATH_GPT_megan_picture_shelves_l2118_211894


namespace NUMINAMATH_GPT_solve_for_x_l2118_211822

theorem solve_for_x : ∃ x : ℝ, (9 - x) ^ 2 = x ^ 2 ∧ x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2118_211822


namespace NUMINAMATH_GPT_number_of_neighborhoods_l2118_211805

def street_lights_per_side : ℕ := 250
def roads_per_neighborhood : ℕ := 4
def total_street_lights : ℕ := 20000

theorem number_of_neighborhoods : 
  (total_street_lights / (2 * street_lights_per_side * roads_per_neighborhood)) = 10 :=
by
  -- proof to show that the number of neighborhoods is 10
  sorry

end NUMINAMATH_GPT_number_of_neighborhoods_l2118_211805


namespace NUMINAMATH_GPT_total_growing_space_correct_l2118_211856

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end NUMINAMATH_GPT_total_growing_space_correct_l2118_211856


namespace NUMINAMATH_GPT_penny_nickel_dime_heads_probability_l2118_211808

def num_successful_outcomes : Nat :=
1 * 1 * 1 * 2

def total_possible_outcomes : Nat :=
2 ^ 4

def probability_event : ℚ :=
num_successful_outcomes / total_possible_outcomes

theorem penny_nickel_dime_heads_probability :
  probability_event = 1 / 8 := 
by
  sorry

end NUMINAMATH_GPT_penny_nickel_dime_heads_probability_l2118_211808


namespace NUMINAMATH_GPT_functional_equation_solution_l2118_211897

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y) + f (f x + f y) = y * f x + f (x + f y)) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l2118_211897


namespace NUMINAMATH_GPT_forty_percent_of_number_l2118_211809

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 10) : 0.40 * N = 120 :=
sorry

end NUMINAMATH_GPT_forty_percent_of_number_l2118_211809


namespace NUMINAMATH_GPT_scientific_notation_l2118_211874

-- Given radius of a water molecule
def radius_of_water_molecule := 0.00000000192

-- Required scientific notation
theorem scientific_notation : radius_of_water_molecule = 1.92 * 10 ^ (-9) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l2118_211874


namespace NUMINAMATH_GPT_negation_of_universal_l2118_211872

variable (f : ℝ → ℝ) (m : ℝ)

theorem negation_of_universal :
  (∀ x : ℝ, f x ≥ m) → ¬ (∀ x : ℝ, f x ≥ m) → ∃ x : ℝ, f x < m :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_l2118_211872


namespace NUMINAMATH_GPT_max_integer_solutions_l2118_211827

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end NUMINAMATH_GPT_max_integer_solutions_l2118_211827


namespace NUMINAMATH_GPT_xiaodong_sister_age_correct_l2118_211857

/-- Let's define the conditions as Lean definitions -/
def sister_age := 13
def xiaodong_age := sister_age - 8
def sister_age_in_3_years := sister_age + 3
def xiaodong_age_in_3_years := xiaodong_age + 3

/-- We need to prove that in 3 years, the sister's age will be twice Xiaodong's age -/
theorem xiaodong_sister_age_correct :
  (sister_age_in_3_years = 2 * xiaodong_age_in_3_years) → sister_age = 13 :=
by
  sorry

end NUMINAMATH_GPT_xiaodong_sister_age_correct_l2118_211857


namespace NUMINAMATH_GPT_podcast_length_l2118_211859

theorem podcast_length (x : ℝ) (hx : x + 2 * x + 1.75 + 1 + 1 = 6) : x = 0.75 :=
by {
  -- We do not need the proof steps here
  sorry
}

end NUMINAMATH_GPT_podcast_length_l2118_211859


namespace NUMINAMATH_GPT_probability_divisibility_9_correct_l2118_211896

-- Define the set S
def S : Set ℕ := { n | ∃ a b: ℕ, 0 ≤ a ∧ a < 40 ∧ 0 ≤ b ∧ b < 40 ∧ a ≠ b ∧ n = 2^a + 2^b }

-- Define the criteria for divisibility by 9
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

-- Define the total size of set S
def size_S : ℕ := 780  -- as calculated from combination

-- Count valid pairs (a, b) such that 2^a + 2^b is divisible by 9
def valid_pairs : ℕ := 133  -- as calculated from summation

-- Define the probability
def probability_divisible_by_9 : ℕ := valid_pairs / size_S

-- The proof statement
theorem probability_divisibility_9_correct:
  (valid_pairs : ℚ) / (size_S : ℚ) = 133 / 780 := sorry

end NUMINAMATH_GPT_probability_divisibility_9_correct_l2118_211896


namespace NUMINAMATH_GPT_original_price_l2118_211820

theorem original_price (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 60) 
  (h2 : rate_of_profit = 0.20) 
  (h3 : SP = CP * (1 + rate_of_profit)) : CP = 50 := by
  sorry

end NUMINAMATH_GPT_original_price_l2118_211820


namespace NUMINAMATH_GPT_problem_l2118_211818

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

variables (f : ℝ → ℝ)
variables (h_odd : is_odd_function f)
variables (h_f1 : f 1 = 5)
variables (h_period : ∀ x, f (x + 4) = -f x)

-- Prove that f(2012) + f(2015) = -5
theorem problem :
  f 2012 + f 2015 = -5 :=
sorry

end NUMINAMATH_GPT_problem_l2118_211818


namespace NUMINAMATH_GPT_ratio_hexagon_octagon_l2118_211838

noncomputable def ratio_of_areas (s : ℝ) :=
  let A1 := s / (2 * Real.tan (Real.pi / 6))
  let H1 := s / (2 * Real.sin (Real.pi / 6))
  let area1 := Real.pi * (H1^2 - A1^2)
  let A2 := s / (2 * Real.tan (Real.pi / 8))
  let H2 := s / (2 * Real.sin (Real.pi / 8))
  let area2 := Real.pi * (H2^2 - A2^2)
  area1 / area2

theorem ratio_hexagon_octagon (s : ℝ) (h : s = 3) : ratio_of_areas s = 49 / 25 :=
  sorry

end NUMINAMATH_GPT_ratio_hexagon_octagon_l2118_211838


namespace NUMINAMATH_GPT_determine_g_l2118_211845

theorem determine_g (t : ℝ) : ∃ (g : ℝ → ℝ), (∀ x y, y = 2 * x - 40 ∧ y = 20 * t - 14 → g t = 10 * t + 13) :=
by
  sorry

end NUMINAMATH_GPT_determine_g_l2118_211845


namespace NUMINAMATH_GPT_smallest_y_condition_l2118_211887

theorem smallest_y_condition : ∃ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 ∧ y = 167 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_y_condition_l2118_211887


namespace NUMINAMATH_GPT_find_parabola_coeffs_l2118_211814

def parabola_vertex_form (a b c : ℝ) : Prop :=
  ∃ k:ℝ, k = c - b^2 / (4*a) ∧ k = 3

def parabola_through_point (a b c : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, x = 0 ∧ y = 1 ∧  y = a * x^2 + b * x + c

theorem find_parabola_coeffs :
  ∃ a b c : ℝ, parabola_vertex_form a b c ∧ parabola_through_point a b c ∧
  a = -1/2 ∧ b = 2 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_parabola_coeffs_l2118_211814


namespace NUMINAMATH_GPT_find_quadruple_l2118_211866

theorem find_quadruple :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a^3 + b^4 + c^5 = d^11 ∧ a * b * c < 10^5 :=
sorry

end NUMINAMATH_GPT_find_quadruple_l2118_211866


namespace NUMINAMATH_GPT_rajas_monthly_income_l2118_211810

theorem rajas_monthly_income (I : ℝ) (h : 0.6 * I + 0.1 * I + 0.1 * I + 5000 = I) : I = 25000 :=
sorry

end NUMINAMATH_GPT_rajas_monthly_income_l2118_211810


namespace NUMINAMATH_GPT_cards_choice_ways_l2118_211858

theorem cards_choice_ways (S : List Char) (cards : Finset (Char × ℕ)) :
  (∀ c ∈ cards, c.1 ∈ S) ∧
  (∀ (c1 c2 : Char × ℕ), c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.1 ≠ c2.1) ∧
  (∃ c ∈ cards, c.2 = 1 ∧ c.1 = 'H') →
  (∃ c ∈ cards, c.2 = 1) →
  ∃ (ways : ℕ), ways = 1014 := 
sorry

end NUMINAMATH_GPT_cards_choice_ways_l2118_211858


namespace NUMINAMATH_GPT_functional_equation_g_l2118_211847

variable (g : ℝ → ℝ)
variable (f : ℝ)
variable (h : ℝ)

theorem functional_equation_g (H1 : ∀ x y : ℝ, g (x + y) = g x * g y)
                            (H2 : g 3 = 4) :
                            g 6 = 16 := 
by
  sorry

end NUMINAMATH_GPT_functional_equation_g_l2118_211847


namespace NUMINAMATH_GPT_maxwell_walking_speed_l2118_211834

open Real

theorem maxwell_walking_speed (v : ℝ) : 
  (∀ (v : ℝ), (4 * v + 6 * 3 = 34)) → v = 4 :=
by
  intros
  have h1 : 4 * v + 18 = 34 := by sorry
  have h2 : 4 * v = 16 := by sorry
  have h3 : v = 4 := by sorry
  exact h3

end NUMINAMATH_GPT_maxwell_walking_speed_l2118_211834


namespace NUMINAMATH_GPT_product_of_102_and_27_l2118_211813

theorem product_of_102_and_27 : 102 * 27 = 2754 :=
by
  sorry

end NUMINAMATH_GPT_product_of_102_and_27_l2118_211813


namespace NUMINAMATH_GPT_xy_sum_is_2_l2118_211864

theorem xy_sum_is_2 (x y : ℝ) (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := 
by
  sorry

end NUMINAMATH_GPT_xy_sum_is_2_l2118_211864


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2118_211871

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → |x| > 1) ∧ (|x| > 1 → (x > 1 ∨ x < -1)) ∧ ¬(|x| > 1 → x > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l2118_211871


namespace NUMINAMATH_GPT_tan_increasing_interval_l2118_211888

noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  {x | (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12)}

theorem tan_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12) ↔ 
    (∃ y, y = (2 * x + Real.pi / 3) ∧ Real.tan y > Real.tan (2 * x + Real.pi / 3 - 1e-6)) :=
sorry

end NUMINAMATH_GPT_tan_increasing_interval_l2118_211888


namespace NUMINAMATH_GPT_problem1_problem2_l2118_211830

-- Problem 1
theorem problem1 : 3^2 * (-1 + 3) - (-16) / 8 = 20 :=
by decide  -- automatically prove simple arithmetic

-- Problem 2
variables {x : ℝ} (hx1 : x ≠ 1) (hx2 : x ≠ -1)

theorem problem2 : ((x^2 / (x + 1)) - (1 / (x + 1))) * (x + 1) / (x - 1) = x + 1 :=
by sorry  -- proof to be completed

end NUMINAMATH_GPT_problem1_problem2_l2118_211830


namespace NUMINAMATH_GPT_probability_X_eq_4_l2118_211835

-- Define the number of students and boys
def total_students := 15
def total_boys := 7
def selected_students := 10

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := n.choose k

-- Calculate the probability
def P_X_eq_4 := (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students

-- The statement to be proven
theorem probability_X_eq_4 :
  P_X_eq_4 = (binomial_coeff total_boys 4 * binomial_coeff (total_students - total_boys) 6) / binomial_coeff total_students selected_students := by
  sorry

end NUMINAMATH_GPT_probability_X_eq_4_l2118_211835


namespace NUMINAMATH_GPT_pyramid_base_side_length_l2118_211804

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (side_length : ℝ)
  (h1 : area_lateral_face = 144)
  (h2 : slant_height = 24)
  (h3 : 144 = 0.5 * side_length * 24) : 
  side_length = 12 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_side_length_l2118_211804
