import Mathlib

namespace NUMINAMATH_GPT_find_X_l347_34797

theorem find_X (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 :=
sorry

end NUMINAMATH_GPT_find_X_l347_34797


namespace NUMINAMATH_GPT_solution_set_of_inequality_l347_34781

theorem solution_set_of_inequality (x : ℝ) : (1 / 2 < x ∧ x < 1) ↔ (x / (2 * x - 1) > 1) :=
by { sorry }

end NUMINAMATH_GPT_solution_set_of_inequality_l347_34781


namespace NUMINAMATH_GPT_distance_gracie_joe_l347_34736

noncomputable def distance_between_points := Real.sqrt (5^2 + (-1)^2)
noncomputable def joe_point := Complex.mk 3 (-4)
noncomputable def gracie_point := Complex.mk (-2) (-3)

theorem distance_gracie_joe : Complex.abs (joe_point - gracie_point) = distance_between_points := by 
  sorry

end NUMINAMATH_GPT_distance_gracie_joe_l347_34736


namespace NUMINAMATH_GPT_triangle_inequality_at_vertex_l347_34769

-- Define the edge lengths of the tetrahedron and the common vertex label
variables {a b c d e f S : ℝ}

-- Conditions for the edge lengths and vertex label
axiom edge_lengths :
  a + b + c = S ∧
  a + d + e = S ∧
  b + d + f = S ∧
  c + e + f = S

-- The theorem to be proven
theorem triangle_inequality_at_vertex :
  a + b + c = S →
  a + d + e = S →
  b + d + f = S →
  c + e + f = S →
  (a ≤ b + c) ∧
  (b ≤ c + a) ∧
  (c ≤ a + b) ∧
  (a ≤ d + e) ∧
  (d ≤ e + a) ∧
  (e ≤ a + d) ∧
  (b ≤ d + f) ∧
  (d ≤ f + b) ∧
  (f ≤ b + d) ∧
  (c ≤ e + f) ∧
  (e ≤ f + c) ∧
  (f ≤ c + e) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_at_vertex_l347_34769


namespace NUMINAMATH_GPT_monotonicity_tangent_intersection_points_l347_34714

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end NUMINAMATH_GPT_monotonicity_tangent_intersection_points_l347_34714


namespace NUMINAMATH_GPT_least_value_sum_l347_34726

theorem least_value_sum (x y z : ℤ) (h : (x - 10) * (y - 5) * (z - 2) = 1000) : x + y + z = 92 :=
sorry

end NUMINAMATH_GPT_least_value_sum_l347_34726


namespace NUMINAMATH_GPT_num_divisible_by_both_digits_l347_34794

theorem num_divisible_by_both_digits : 
  ∃ n, n = 14 ∧ ∀ (d : ℕ), (d ≥ 10 ∧ d < 100) → 
      (∀ a b, (d = 10 * a + b) → d % a = 0 ∧ d % b = 0 → (a = b ∨ a * 2 = b ∨ a * 5 = b)) :=
sorry

end NUMINAMATH_GPT_num_divisible_by_both_digits_l347_34794


namespace NUMINAMATH_GPT_product_of_consecutive_integers_l347_34751

theorem product_of_consecutive_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_less : a < b) :
  ∃ (x y : ℕ), x ≠ y ∧ x * y % (a * b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_l347_34751


namespace NUMINAMATH_GPT_problem_statement_l347_34738

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l347_34738


namespace NUMINAMATH_GPT_min_books_borrowed_l347_34709

theorem min_books_borrowed
  (total_students : ℕ)
  (students_no_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (avg_books_per_student : ℝ)
  (total_students_eq : total_students = 40)
  (students_no_books_eq : students_no_books = 2)
  (students_one_book_eq : students_one_book = 12)
  (students_two_books_eq : students_two_books = 13)
  (avg_books_per_student_eq : avg_books_per_student = 2) :
  ∀ min_books_borrowed : ℕ, 
    (total_students * avg_books_per_student = 80) → 
    (students_one_book * 1 + students_two_books * 2 ≤ 38) → 
    (total_students - students_no_books - students_one_book - students_two_books = 13) →
    min_books_borrowed * 13 = 42 → 
    min_books_borrowed = 4 :=
by
  intros min_books_borrowed total_books_eq books_count_eq remaining_students_eq total_min_books_eq
  sorry

end NUMINAMATH_GPT_min_books_borrowed_l347_34709


namespace NUMINAMATH_GPT_number_of_cows_l347_34741

theorem number_of_cows (D C : ℕ) (h1 : 2 * D + 4 * C = 40 + 2 * (D + C)) : C = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_cows_l347_34741


namespace NUMINAMATH_GPT_eval_abs_a_plus_b_l347_34782

theorem eval_abs_a_plus_b (a b : ℤ) (x : ℤ) 
(h : (7 * x - a) ^ 2 = 49 * x ^ 2 - b * x + 9) : |a + b| = 45 :=
sorry

end NUMINAMATH_GPT_eval_abs_a_plus_b_l347_34782


namespace NUMINAMATH_GPT_m_le_n_l347_34758

def polygon : Type := sorry  -- A placeholder definition for polygon.

variables (M : polygon) -- The polygon \( M \)
def max_non_overlapping_circles (M : polygon) : ℕ := sorry -- The maximum number of non-overlapping circles with diameter 1 inside \( M \).
def min_covering_circles (M : polygon) : ℕ := sorry -- The minimum number of circles with radius 1 required to cover \( M \).

theorem m_le_n (M : polygon) : min_covering_circles M ≤ max_non_overlapping_circles M :=
sorry

end NUMINAMATH_GPT_m_le_n_l347_34758


namespace NUMINAMATH_GPT_expression_is_composite_l347_34732

theorem expression_is_composite (a b : ℕ) : ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 4 * a^2 + 4 * a * b + 4 * a + 2 * b + 1 = m * n := 
by 
  sorry

end NUMINAMATH_GPT_expression_is_composite_l347_34732


namespace NUMINAMATH_GPT_exists_t_for_f_inequality_l347_34762

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1) ^ 2

theorem exists_t_for_f_inequality :
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f (x + t) ≤ x := by
  sorry

end NUMINAMATH_GPT_exists_t_for_f_inequality_l347_34762


namespace NUMINAMATH_GPT_no_solution_for_x_l347_34796

open Real

theorem no_solution_for_x (m : ℝ) : ¬ ∃ x : ℝ, (sin (3 * x) * cos (↑60 - x) + 1) / (sin (↑60 - 7 * x) - cos (↑30 + x) + m) = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_x_l347_34796


namespace NUMINAMATH_GPT_find_primes_l347_34730

-- Definition of being a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

-- Lean 4 statement of the problem
theorem find_primes (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 → p = 5 ∧ q = 3 ∧ r = 19 := 
by
  sorry

end NUMINAMATH_GPT_find_primes_l347_34730


namespace NUMINAMATH_GPT_sphere_surface_area_l347_34757

theorem sphere_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (32 / 3) * π) : 4 * π * R^2 = 16 * π :=
sorry

end NUMINAMATH_GPT_sphere_surface_area_l347_34757


namespace NUMINAMATH_GPT_rod_sliding_friction_l347_34744

noncomputable def coefficient_of_friction (mg : ℝ) (F : ℝ) (α : ℝ) := 
  (F * Real.cos α - 6 * mg * Real.sin α) / (6 * mg)

theorem rod_sliding_friction
  (α : ℝ)
  (hα : α = 85 * Real.pi / 180)
  (mg : ℝ)
  (hmg_pos : 0 < mg)
  (F : ℝ)
  (hF : F = (mg - 6 * mg * Real.cos 85) / Real.sin 85) :
  coefficient_of_friction mg F α = 0.08 := 
by
  simp [coefficient_of_friction, hα, hF, Real.cos, Real.sin]
  sorry

end NUMINAMATH_GPT_rod_sliding_friction_l347_34744


namespace NUMINAMATH_GPT_concatenated_natural_irrational_l347_34703

def concatenated_natural_decimal : ℝ := 0.1234567891011121314151617181920 -- and so on

theorem concatenated_natural_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ concatenated_natural_decimal = p / q :=
sorry

end NUMINAMATH_GPT_concatenated_natural_irrational_l347_34703


namespace NUMINAMATH_GPT_intersection_M_N_l347_34771

def I : Set ℤ := {0, -1, -2, -3, -4}
def M : Set ℤ := {0, -1, -2}
def N : Set ℤ := {0, -3, -4}

theorem intersection_M_N : M ∩ N = {0} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l347_34771


namespace NUMINAMATH_GPT_compute_ab_l347_34766

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end NUMINAMATH_GPT_compute_ab_l347_34766


namespace NUMINAMATH_GPT_major_premise_incorrect_l347_34756

theorem major_premise_incorrect (a b : ℝ) (h : a > b) : ¬ (a^2 > b^2) :=
by {
  sorry
}

end NUMINAMATH_GPT_major_premise_incorrect_l347_34756


namespace NUMINAMATH_GPT_increasing_interval_l347_34712

noncomputable def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2

theorem increasing_interval :
  is_monotonic_increasing y π (2 * π) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_increasing_interval_l347_34712


namespace NUMINAMATH_GPT_xiaoMing_xiaoHong_diff_university_l347_34731

-- Definitions based on problem conditions
inductive Student
| XiaoMing
| XiaoHong
| StudentC
| StudentD
deriving DecidableEq

inductive University
| A
| B
deriving DecidableEq

-- Definition for the problem
def num_ways_diff_university : Nat :=
  4 -- The correct answer based on the solution steps

-- Problem statement
theorem xiaoMing_xiaoHong_diff_university :
  let students := [Student.XiaoMing, Student.XiaoHong, Student.StudentC, Student.StudentD]
  let universities := [University.A, University.B]
  (∃ (assign : Student → University),
    assign Student.XiaoMing ≠ assign Student.XiaoHong ∧
    (assign Student.StudentC ≠ assign Student.StudentD ∨
     assign Student.XiaoMing ≠ assign Student.StudentD ∨
     assign Student.XiaoHong ≠ assign Student.StudentC ∨
     assign Student.XiaoMing ≠ assign Student.StudentC)) →
  num_ways_diff_university = 4 :=
by
  sorry

end NUMINAMATH_GPT_xiaoMing_xiaoHong_diff_university_l347_34731


namespace NUMINAMATH_GPT_collinear_points_l347_34765

theorem collinear_points (k : ℝ) (OA OB OC : ℝ × ℝ) 
  (hOA : OA = (1, -3)) 
  (hOB : OB = (2, -1))
  (hOC : OC = (k + 1, k - 2))
  (h_collinear : ∃ t : ℝ, OC - OA = t • (OB - OA)) : 
  k = 1 :=
by
  have := h_collinear
  sorry

end NUMINAMATH_GPT_collinear_points_l347_34765


namespace NUMINAMATH_GPT_smallest_x_plus_y_l347_34778

theorem smallest_x_plus_y (x y : ℕ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x^2 - 29 * y^2 = 1) : x + y = 11621 := 
sorry

end NUMINAMATH_GPT_smallest_x_plus_y_l347_34778


namespace NUMINAMATH_GPT_total_snowfall_l347_34708

variable (morning_snowfall : ℝ) (afternoon_snowfall : ℝ)

theorem total_snowfall {morning_snowfall afternoon_snowfall : ℝ} (h_morning : morning_snowfall = 0.12) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 :=
sorry

end NUMINAMATH_GPT_total_snowfall_l347_34708


namespace NUMINAMATH_GPT_distinct_positive_integers_solution_l347_34780

theorem distinct_positive_integers_solution (x y : ℕ) (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h : 1 / x + 1 / y = 2 / 7) : (x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4) :=
by
  sorry -- proof to be filled in.

end NUMINAMATH_GPT_distinct_positive_integers_solution_l347_34780


namespace NUMINAMATH_GPT_same_type_sqrt_l347_34723

theorem same_type_sqrt (x : ℝ) : (x = 2 * Real.sqrt 3) ↔
  (x = Real.sqrt (1/3)) ∨
  (¬(x = Real.sqrt 8) ∧ ¬(x = Real.sqrt 18) ∧ ¬(x = Real.sqrt 9)) :=
by
  sorry

end NUMINAMATH_GPT_same_type_sqrt_l347_34723


namespace NUMINAMATH_GPT_order_of_a_add_b_sub_b_l347_34742

variable (a b : ℚ)

theorem order_of_a_add_b_sub_b (hb : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end NUMINAMATH_GPT_order_of_a_add_b_sub_b_l347_34742


namespace NUMINAMATH_GPT_fraction_problem_l347_34728

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_problem_l347_34728


namespace NUMINAMATH_GPT_joanna_marbles_l347_34737

theorem joanna_marbles (m n : ℕ) (h1 : m * n = 720) (h2 : m > 1) (h3 : n > 1) :
  ∃ (count : ℕ), count = 28 :=
by
  -- Use the properties of divisors and conditions to show that there are 28 valid pairs (m, n).
  sorry

end NUMINAMATH_GPT_joanna_marbles_l347_34737


namespace NUMINAMATH_GPT_min_value_fraction_l347_34746

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ x₀, (2 * x₀ - 2) * (-2 * x₀ + a) = -1) : 
  ∃ a b, a + b = 5 / 2 → a > 0 → b > 0 → 
  (∀ a b, (1 / a + 4 / b) ≥ 18 / 5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l347_34746


namespace NUMINAMATH_GPT_focus_of_parabola_l347_34784

theorem focus_of_parabola (x y : ℝ) (h : x^2 = -y) : (0, -1/4) = (0, -1/4) :=
by sorry

end NUMINAMATH_GPT_focus_of_parabola_l347_34784


namespace NUMINAMATH_GPT_phung_more_than_chiu_l347_34772

theorem phung_more_than_chiu
  (C P H : ℕ)
  (h1 : C = 56)
  (h2 : H = P + 5)
  (h3 : C + P + H = 205) :
  P - C = 16 :=
by
  sorry

end NUMINAMATH_GPT_phung_more_than_chiu_l347_34772


namespace NUMINAMATH_GPT_prime_numbers_satisfying_condition_l347_34752

theorem prime_numbers_satisfying_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_prime_numbers_satisfying_condition_l347_34752


namespace NUMINAMATH_GPT_factor_expression_l347_34735

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l347_34735


namespace NUMINAMATH_GPT_age_difference_is_18_l347_34743

def difference_in_ages (X Y Z : ℕ) : ℕ := (X + Y) - (Y + Z)
def younger_by_eighteen (X Z : ℕ) : Prop := Z = X - 18

theorem age_difference_is_18 (X Y Z : ℕ) (h : younger_by_eighteen X Z) : difference_in_ages X Y Z = 18 := by
  sorry

end NUMINAMATH_GPT_age_difference_is_18_l347_34743


namespace NUMINAMATH_GPT_correct_option_is_A_l347_34775

def a (n : ℕ) : ℤ :=
  match n with
  | 1 => -3
  | 2 => 7
  | _ => 0  -- This is just a placeholder for other values

def optionA (n : ℕ) : ℤ := (-1)^n * (4*n - 1)
def optionB (n : ℕ) : ℤ := (-1)^n * (4*n + 1)
def optionC (n : ℕ) : ℤ := 4*n - 7
def optionD (n : ℕ) : ℤ := (-1)^(n + 1) * (4*n - 1)

theorem correct_option_is_A :
  (a 1 = -3) ∧ (a 2 = 7) ∧
  (optionA 1 = -3 ∧ optionA 2 = 7) ∧
  ¬(optionB 1 = -3 ∧ optionB 2 = 7) ∧
  ¬(optionC 1 = -3 ∧ optionC 2 = 7) ∧
  ¬(optionD 1 = -3 ∧ optionD 2 = 7) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_is_A_l347_34775


namespace NUMINAMATH_GPT_find_x1_l347_34755

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/4) 
  : x1 = 3/4 := 
sorry

end NUMINAMATH_GPT_find_x1_l347_34755


namespace NUMINAMATH_GPT_unit_digit_G_1000_l347_34710

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem unit_digit_G_1000 : (G 1000) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_unit_digit_G_1000_l347_34710


namespace NUMINAMATH_GPT_find_a1_l347_34750

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  a 6 = 9 →
  a 3 = 3 * a 2 →
  a 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l347_34750


namespace NUMINAMATH_GPT_problem_exists_integers_a_b_c_d_l347_34700

theorem problem_exists_integers_a_b_c_d :
  ∃ (a b c d : ℤ), 
  |a| > 1000000 ∧ |b| > 1000000 ∧ |c| > 1000000 ∧ |d| > 1000000 ∧
  (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) + 1 / (d:ℚ) = 1 / (a * b * c * d : ℚ)) :=
sorry

end NUMINAMATH_GPT_problem_exists_integers_a_b_c_d_l347_34700


namespace NUMINAMATH_GPT_find_three_digit_number_l347_34747

/-- 
  Define the three-digit number abc and show that for some digit d in the range of 1 to 9,
  the conditions are satisfied.
-/
theorem find_three_digit_number
  (a b c d : ℕ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1 ≤ d ∧ d ≤ 9)
  (h_abc : 100 * a + 10 * b + c = 627)
  (h_bcd : 100 * b + 10 * c + d = 627 * a)
  (h_1a4d : 1040 + 100 * a + d = 627 * a)
  : 100 * a + 10 * b + c = 627 := 
sorry

end NUMINAMATH_GPT_find_three_digit_number_l347_34747


namespace NUMINAMATH_GPT_find_missing_dimension_of_carton_l347_34786

-- Definition of given dimensions and conditions
def carton_length : ℕ := 25
def carton_width : ℕ := 48
def soap_length : ℕ := 8
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 300
def soap_volume : ℕ := soap_length * soap_width * soap_height
def total_carton_volume : ℕ := max_soap_boxes * soap_volume

-- The main statement to prove
theorem find_missing_dimension_of_carton (h : ℕ) (volume_eq : carton_length * carton_width * h = total_carton_volume) : h = 60 :=
sorry

end NUMINAMATH_GPT_find_missing_dimension_of_carton_l347_34786


namespace NUMINAMATH_GPT_dima_can_find_heavy_ball_l347_34793

noncomputable def find_heavy_ball
  (balls : Fin 9) -- 9 balls, indexed from 0 to 8 representing the balls 1 to 9
  (heavy : Fin 9) -- One of the balls is heavier
  (weigh : Fin 9 → Fin 9 → Ordering) -- A function that compares two groups of balls and gives an Ordering: .lt, .eq, or .gt
  (predetermined_sets : List (Fin 9 × Fin 9)) -- A list of tuples representing balls on each side for the two weighings
  (valid_sets : predetermined_sets.length ≤ 2) : Prop := -- Not more than two weighings
  ∃ idx : Fin 9, idx = heavy -- Need to prove that we can always find the heavier ball

theorem dima_can_find_heavy_ball :
  ∀ (balls : Fin 9) (heavy : Fin 9)
    (weigh : Fin 9 → Fin 9 → Ordering)
    (predetermined_sets : List (Fin 9 × Fin 9))
    (valid_sets : predetermined_sets.length ≤ 2),
  find_heavy_ball balls heavy weigh predetermined_sets valid_sets :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_dima_can_find_heavy_ball_l347_34793


namespace NUMINAMATH_GPT_estimate_white_balls_l347_34749

theorem estimate_white_balls
  (total_balls : ℕ)
  (trials : ℕ)
  (white_draws : ℕ)
  (proportion_white : ℚ)
  (hw : total_balls = 10)
  (ht : trials = 400)
  (hd : white_draws = 240)
  (hprop : proportion_white = 0.6) :
  ∃ x : ℕ, x = 6 :=
by
  sorry

end NUMINAMATH_GPT_estimate_white_balls_l347_34749


namespace NUMINAMATH_GPT_Jose_got_5_questions_wrong_l347_34768

def Jose_questions_wrong (M J A : ℕ) : Prop :=
  M = J - 20 ∧
  J = A + 40 ∧
  M + J + A = 210 ∧
  (50 * 2 = 100) ∧
  (100 - J) / 2 = 5

theorem Jose_got_5_questions_wrong (M J A : ℕ) (h1 : M = J - 20) (h2 : J = A + 40) (h3 : M + J + A = 210) : 
  Jose_questions_wrong M J A :=
by
  sorry

end NUMINAMATH_GPT_Jose_got_5_questions_wrong_l347_34768


namespace NUMINAMATH_GPT_participation_increase_closest_to_10_l347_34739

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end NUMINAMATH_GPT_participation_increase_closest_to_10_l347_34739


namespace NUMINAMATH_GPT_total_actions_135_l347_34785

theorem total_actions_135
  (y : ℕ) -- represents the total number of actions
  (h1 : y ≥ 10) -- since there are at least 10 initial comments
  (h2 : ∀ (likes dislikes : ℕ), likes + dislikes = y - 10) -- total votes exclude neutral comments
  (score_eq : ∀ (likes dislikes : ℕ), 70 * dislikes = 30 * likes)
  (score_50 : ∀ (likes dislikes : ℕ), 50 = likes - dislikes) :
  y = 135 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_actions_135_l347_34785


namespace NUMINAMATH_GPT_d_not_unique_minimum_l347_34727

noncomputable def d (n : ℕ) (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (Finset.min' (Finset.univ.image (λ i => abs (x i - t))) sorry + 
  Finset.max' (Finset.univ.image (λ i => abs (x i - t))) sorry) / 2

theorem d_not_unique_minimum (n : ℕ) (x : Fin n → ℝ) :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ d n x t1 = d n x t2 := sorry

end NUMINAMATH_GPT_d_not_unique_minimum_l347_34727


namespace NUMINAMATH_GPT_only_n_divides_2_to_n_minus_1_l347_34791

theorem only_n_divides_2_to_n_minus_1 (n : ℕ) (h1 : n > 0) : n ∣ (2^n - 1) ↔ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_only_n_divides_2_to_n_minus_1_l347_34791


namespace NUMINAMATH_GPT_inequality_cubed_l347_34753

theorem inequality_cubed (a b : ℝ) (h : a < b ∧ b < 0) : a^3 ≤ b^3 :=
sorry

end NUMINAMATH_GPT_inequality_cubed_l347_34753


namespace NUMINAMATH_GPT_theater_rows_25_l347_34783

theorem theater_rows_25 (n : ℕ) (x : ℕ) (k : ℕ) (h : n = 1000) (h1 : k > 16) (h2 : (2 * x + k) * (k + 1) = 2000) : (k + 1) = 25 :=
by
  -- The proof goes here, which we omit for the problem statement.
  sorry

end NUMINAMATH_GPT_theater_rows_25_l347_34783


namespace NUMINAMATH_GPT_solve_equation_l347_34720

theorem solve_equation (x : ℝ) :
  (1 / (x^2 + 17 * x - 8) + 1 / (x^2 + 4 * x - 8) + 1 / (x^2 - 9 * x - 8) = 0) →
  (x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l347_34720


namespace NUMINAMATH_GPT_union_of_A_B_l347_34767

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_B :
  A ∪ B = {x | x ≥ -1} := by
  sorry

end NUMINAMATH_GPT_union_of_A_B_l347_34767


namespace NUMINAMATH_GPT_maximum_n_for_positive_sum_l347_34725

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :=
  S n > 0

-- Definition of the arithmetic sequence properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d
  
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
variable (h₁ : a 1 > 0)
variable (h₅ : a 2016 + a 2017 > 0)
variable (h₆ : a 2016 * a 2017 < 0)

-- Add the definition of the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Prove the final statement
theorem maximum_n_for_positive_sum : max_n_for_positive_sum a S 4032 :=
by
  -- conditions to use in the proof
  have h₁ : a 1 > 0 := sorry
  have h₅ : a 2016 + a 2017 > 0 := sorry
  have h₆ : a 2016 * a 2017 < 0 := sorry
  -- positively bounded sum
  let Sn := sum_of_first_n_terms a
  -- proof to utilize Lean's capabilities, replace with actual proof later
  sorry

end NUMINAMATH_GPT_maximum_n_for_positive_sum_l347_34725


namespace NUMINAMATH_GPT_unique_solution_f_eq_x_l347_34792

theorem unique_solution_f_eq_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^2 + y + f y) = 2 * y + f x ^ 2) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_unique_solution_f_eq_x_l347_34792


namespace NUMINAMATH_GPT_sign_of_c_l347_34799

theorem sign_of_c (a b c : ℝ) (h1 : (a * b / c) < 0) (h2 : (a * b) < 0) : c > 0 :=
sorry

end NUMINAMATH_GPT_sign_of_c_l347_34799


namespace NUMINAMATH_GPT_total_candy_count_l347_34777

def numberOfRedCandies : ℕ := 145
def numberOfBlueCandies : ℕ := 3264
def totalNumberOfCandies : ℕ := numberOfRedCandies + numberOfBlueCandies

theorem total_candy_count :
  totalNumberOfCandies = 3409 :=
by
  unfold totalNumberOfCandies
  unfold numberOfRedCandies
  unfold numberOfBlueCandies
  sorry

end NUMINAMATH_GPT_total_candy_count_l347_34777


namespace NUMINAMATH_GPT_min_value_reciprocal_l347_34716

variable {a b : ℝ}

theorem min_value_reciprocal (h1 : a * b > 0) (h2 : a + 4 * b = 1) : 
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 9) := 
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_l347_34716


namespace NUMINAMATH_GPT_number_of_classes_l347_34718

theorem number_of_classes (x : ℕ) (h : x * (x - 1) = 20) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_classes_l347_34718


namespace NUMINAMATH_GPT_relationship_S_T_l347_34729

def S (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := 2^n - (-1)^n

theorem relationship_S_T (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → S n < T n) ∧ (n % 2 = 0 → S n > T n) :=
by
  sorry

end NUMINAMATH_GPT_relationship_S_T_l347_34729


namespace NUMINAMATH_GPT_square_field_area_l347_34707

theorem square_field_area (s : ℕ) (area cost_per_meter total_cost gate_width : ℕ):
  area = s^2 →
  cost_per_meter = 2 →
  total_cost = 1332 →
  gate_width = 1 →
  (4 * s - 2 * gate_width) * cost_per_meter = total_cost →
  area = 27889 :=
by
  intros h_area h_cost_per_meter h_total_cost h_gate_width h_equation
  sorry

end NUMINAMATH_GPT_square_field_area_l347_34707


namespace NUMINAMATH_GPT_days_B_to_finish_work_l347_34711

-- Definition of work rates based on the conditions
def work_rate_A (A_days: ℕ) : ℚ := 1 / A_days
def work_rate_B (B_days: ℕ) : ℚ := 1 / B_days

-- Theorem that encapsulates the problem statement
theorem days_B_to_finish_work (A_days B_days together_days : ℕ) (work_rate_A_eq : work_rate_A 4 = 1/4) (work_rate_B_eq : work_rate_B 12 = 1/12) : 
  ∀ (remaining_work: ℚ), remaining_work = 1 - together_days * (work_rate_A 4 + work_rate_B 12) → 
  (remaining_work / (work_rate_B 12)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_days_B_to_finish_work_l347_34711


namespace NUMINAMATH_GPT_tina_brownies_per_meal_l347_34798

-- Define the given conditions
def total_brownies : ℕ := 24
def days : ℕ := 5
def meals_per_day : ℕ := 2
def brownies_by_husband_per_day : ℕ := 1
def total_brownies_shared_with_guests : ℕ := 4
def total_brownies_left : ℕ := 5

-- Conjecture: How many brownies did Tina have with each meal
theorem tina_brownies_per_meal :
  (total_brownies 
  - (brownies_by_husband_per_day * days) 
  - total_brownies_shared_with_guests 
  - total_brownies_left)
  / (days * meals_per_day) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tina_brownies_per_meal_l347_34798


namespace NUMINAMATH_GPT_calculate_expression_l347_34724

theorem calculate_expression : -4^2 * (-1)^2022 = -16 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l347_34724


namespace NUMINAMATH_GPT_total_legs_correct_l347_34789

variable (a b : ℕ)

def total_legs (a b : ℕ) : ℕ := 2 * a + 4 * b

theorem total_legs_correct (a b : ℕ) : total_legs a b = 2 * a + 4 * b :=
by sorry

end NUMINAMATH_GPT_total_legs_correct_l347_34789


namespace NUMINAMATH_GPT_possible_value_of_2n_plus_m_l347_34719

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end NUMINAMATH_GPT_possible_value_of_2n_plus_m_l347_34719


namespace NUMINAMATH_GPT_intersection_complement_l347_34774

open Set

variable (U A B : Set ℕ)

-- Definitions based on conditions given in the problem
def universal_set : Set ℕ := {1, 2, 3, 4, 5}
def set_A : Set ℕ := {2, 4}
def set_B : Set ℕ := {4, 5}

-- Proof statement
theorem intersection_complement :
  A = set_A → 
  B = set_B → 
  U = universal_set → 
  (A ∩ (U \ B)) = {2} := 
by
  intros hA hB hU
  sorry

end NUMINAMATH_GPT_intersection_complement_l347_34774


namespace NUMINAMATH_GPT_Eval_trig_exp_l347_34764

theorem Eval_trig_exp :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_Eval_trig_exp_l347_34764


namespace NUMINAMATH_GPT_complement_set_l347_34790

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem complement_set :
  U = univ ∧ M = {x | x^2 - 2 * x ≤ 0} → (U \ M) = {x | x < 0 ∨ x > 2} :=
by
  intros
  sorry

end NUMINAMATH_GPT_complement_set_l347_34790


namespace NUMINAMATH_GPT_probability_white_given_red_l347_34717

-- Define the total number of balls initially
def total_balls := 10

-- Define the number of red balls, white balls, and black balls
def red_balls := 3
def white_balls := 2
def black_balls := 5

-- Define the event A: Picking a red ball on the first draw
def event_A := red_balls

-- Define the event B: Picking a white ball on the second draw
-- Number of balls left after picking one red ball
def remaining_balls_after_A := total_balls - 1

-- Define the event AB: Picking a red ball first and then a white ball
def event_AB := red_balls * white_balls

-- Calculate the probability P(B|A)
def P_B_given_A := event_AB / (event_A * remaining_balls_after_A)

-- Prove the probability of picking a white ball on the second draw given that the first ball picked is a red ball
theorem probability_white_given_red : P_B_given_A = (2 / 9) := by
  sorry

end NUMINAMATH_GPT_probability_white_given_red_l347_34717


namespace NUMINAMATH_GPT_sum_of_monomials_is_monomial_l347_34701

variable (a b : ℕ)

theorem sum_of_monomials_is_monomial (m n : ℕ) (h : ∃ k : ℕ, 2 * a^m * b^n + a * b^3 = k * a^1 * b^3) :
  m = 1 ∧ n = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_monomials_is_monomial_l347_34701


namespace NUMINAMATH_GPT_cyclic_quadrilateral_condition_l347_34702

-- Definitions of the points and sides of the triangle
variables (A B C S E F : Type) 

-- Assume S is the centroid of triangle ABC
def is_centroid (A B C S : Type) : Prop := 
  -- actual centralized definition here (omitted)
  sorry

-- Assume E is the midpoint of side AB
def is_midpoint (A B E : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume F is the midpoint of side AC
def is_midpoint_AC (A C F : Type) : Prop := 
  -- actual midpoint definition here (omitted)
  sorry 

-- Assume a quadrilateral AESF
def is_cyclic (A E S F : Type) : Prop :=
  -- actual cyclic definition here (omitted)
  sorry 

theorem cyclic_quadrilateral_condition 
  (A B C S E F : Type)
  (a b c : ℝ) 
  (h1 : is_centroid A B C S)
  (h2 : is_midpoint A B E) 
  (h3 : is_midpoint_AC A C F) :
  is_cyclic A E S F ↔ (c^2 + b^2 = 2 * a^2) :=
sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_condition_l347_34702


namespace NUMINAMATH_GPT_find_multiple_l347_34788

variable (P W : ℕ)
variable (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
variable (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2)

theorem find_multiple (P W : ℕ) (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
                      (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_l347_34788


namespace NUMINAMATH_GPT_prize_distribution_l347_34704

theorem prize_distribution 
  (total_winners : ℕ)
  (score1 score2 score3 : ℕ)
  (total_points : ℕ) 
  (winners1 winners2 winners3 : ℕ) :
  total_winners = 5 →
  score1 = 20 →
  score2 = 19 →
  score3 = 18 →
  total_points = 94 →
  score1 * winners1 + score2 * winners2 + score3 * winners3 = total_points →
  winners1 + winners2 + winners3 = total_winners →
  winners1 = 1 ∧ winners2 = 2 ∧ winners3 = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_prize_distribution_l347_34704


namespace NUMINAMATH_GPT_calculate_expression_l347_34787

noncomputable def expr1 : ℝ := (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3)
noncomputable def expr2 : ℝ := (2 * Real.sqrt 2 - 1) ^ 2
noncomputable def combined_expr : ℝ := expr1 + expr2

-- We need to prove the main statement
theorem calculate_expression : combined_expr = 8 - 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l347_34787


namespace NUMINAMATH_GPT_blue_paint_quantity_l347_34779

-- Conditions
def paint_ratio (r b y w : ℕ) : Prop := r = 2 * w / 4 ∧ b = 3 * w / 4 ∧ y = 1 * w / 4 ∧ w = 4 * (r + b + y + w) / 10

-- Given
def quart_white_paint : ℕ := 16

-- Prove that Victor should use 12 quarts of blue paint
theorem blue_paint_quantity (r b y w : ℕ) (h : paint_ratio r b y w) (hw : w = quart_white_paint) : 
  b = 12 := by
  sorry

end NUMINAMATH_GPT_blue_paint_quantity_l347_34779


namespace NUMINAMATH_GPT_was_not_speeding_l347_34734

theorem was_not_speeding (x s : ℝ) (s_obs : ℝ := 26.5) (x_limit : ℝ := 120)
  (brake_dist_eq : s = 0.01 * x + 0.002 * x^2) : s_obs < 30 → x ≤ x_limit :=
sorry

end NUMINAMATH_GPT_was_not_speeding_l347_34734


namespace NUMINAMATH_GPT_solve_fraction_identity_l347_34795

theorem solve_fraction_identity (x : ℝ) (hx : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_identity_l347_34795


namespace NUMINAMATH_GPT_sum_of_remainders_eq_11_mod_13_l347_34745

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_eq_11_mod_13_l347_34745


namespace NUMINAMATH_GPT_rhombus_area_l347_34759

noncomputable def sqrt125 : ℝ := Real.sqrt 125

theorem rhombus_area 
  (p q : ℝ) 
  (h1 : p < q) 
  (h2 : p + 8 = q) 
  (h3 : ∀ a b : ℝ, a^2 + b^2 = 125 ↔ 2*a = p ∧ 2*b = q) : 
  p*q/2 = 60.5 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l347_34759


namespace NUMINAMATH_GPT_piastres_in_6th_purse_l347_34713

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end NUMINAMATH_GPT_piastres_in_6th_purse_l347_34713


namespace NUMINAMATH_GPT_range_of_m_l347_34748

theorem range_of_m (m : ℝ) (x : ℝ) (h_eq : m / (x - 2) = 3) (h_pos : x > 0) : m > -6 ∧ m ≠ 0 := 
sorry

end NUMINAMATH_GPT_range_of_m_l347_34748


namespace NUMINAMATH_GPT_factorize_x4_minus_16_factorize_trinomial_l347_34715

-- For problem 1: Factorization of \( x^4 - 16 \)
theorem factorize_x4_minus_16 (x : ℝ) : 
  x^4 - 16 = (x - 2) * (x + 2) * (x^2 + 4) := 
sorry

-- For problem 2: Factorization of \( -9x^2y + 12xy^2 - 4y^3 \)
theorem factorize_trinomial (x y : ℝ) : 
  -9 * x^2 * y + 12 * x * y^2 - 4 * y^3 = -y * (3 * x - 2 * y)^2 := 
sorry

end NUMINAMATH_GPT_factorize_x4_minus_16_factorize_trinomial_l347_34715


namespace NUMINAMATH_GPT_test_scores_ordering_l347_34721

variable (M Q S Z K : ℕ)
variable (M_thinks_lowest : M > K)
variable (Q_thinks_same : Q = K)
variable (S_thinks_not_highest : S < K)
variable (Z_thinks_not_middle : (Z < S ∨ Z > M))

theorem test_scores_ordering : (Z < S) ∧ (S < Q) ∧ (Q < M) := by
  -- proof
  sorry

end NUMINAMATH_GPT_test_scores_ordering_l347_34721


namespace NUMINAMATH_GPT_goods_amount_decreased_initial_goods_amount_total_fees_l347_34760

-- Define the conditions as variables
def tonnages : List Int := [31, -31, -16, 34, -38, -20]
def final_goods : Int := 430
def fee_per_ton : Int := 5

-- Prove that the amount of goods in the warehouse has decreased
theorem goods_amount_decreased : (tonnages.sum < 0) := by
  sorry

-- Prove the initial amount of goods in the warehouse
theorem initial_goods_amount : (final_goods + tonnages.sum = 470) := by
  sorry

-- Prove the total loading and unloading fees
theorem total_fees : (tonnages.map Int.natAbs).sum * fee_per_ton = 850 := by
  sorry

end NUMINAMATH_GPT_goods_amount_decreased_initial_goods_amount_total_fees_l347_34760


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l347_34761

theorem sum_of_distinct_integers 
  (a b c d e : ℤ)
  (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60)
  (h2 : (7 - a) ≠ (7 - b) ∧ (7 - a) ≠ (7 - c) ∧ (7 - a) ≠ (7 - d) ∧ (7 - a) ≠ (7 - e))
  (h3 : (7 - b) ≠ (7 - c) ∧ (7 - b) ≠ (7 - d) ∧ (7 - b) ≠ (7 - e))
  (h4 : (7 - c) ≠ (7 - d) ∧ (7 - c) ≠ (7 - e))
  (h5 : (7 - d) ≠ (7 - e)) : 
  a + b + c + d + e = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l347_34761


namespace NUMINAMATH_GPT_ab_range_l347_34705

theorem ab_range (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 - x^2|)
  (h_a_lt_b : 0 < a ∧ a < b) (h_fa_eq_fb : f a = f b) :
  0 < a * b ∧ a * b < 2 := 
by
  sorry

end NUMINAMATH_GPT_ab_range_l347_34705


namespace NUMINAMATH_GPT_not_divisible_by_1955_l347_34776

theorem not_divisible_by_1955 (n : ℤ) : ¬ ∃ k : ℤ, (n^2 + n + 1) = 1955 * k :=
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_1955_l347_34776


namespace NUMINAMATH_GPT_roots_of_polynomial_in_range_l347_34773

theorem roots_of_polynomial_in_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 * x2 = m^2 - 2 ∧ (x1 + x2) = -(m - 1)) 
  -> 0 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_in_range_l347_34773


namespace NUMINAMATH_GPT_tic_tac_toe_lines_l347_34763

theorem tic_tac_toe_lines (n : ℕ) (h_pos : 0 < n) : 
  ∃ lines : ℕ, lines = (5^n - 3^n) / 2 :=
sorry

end NUMINAMATH_GPT_tic_tac_toe_lines_l347_34763


namespace NUMINAMATH_GPT_gcd_lcm_identity_l347_34706

variables {n m k : ℕ}

/-- Given positive integers n, m, and k such that n divides lcm(m, k) 
    and m divides lcm(n, k), we prove that n * gcd(m, k) = m * gcd(n, k). -/
theorem gcd_lcm_identity (n_pos : 0 < n) (m_pos : 0 < m) (k_pos : 0 < k) 
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) :
  n * Nat.gcd m k = m * Nat.gcd n k :=
sorry

end NUMINAMATH_GPT_gcd_lcm_identity_l347_34706


namespace NUMINAMATH_GPT_triangle_third_side_l347_34754

theorem triangle_third_side (DE DF : ℝ) (E F : ℝ) (EF : ℝ) 
    (h₁ : DE = 7) 
    (h₂ : DF = 21) 
    (h₃ : E = 3 * F) : EF = 14 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_third_side_l347_34754


namespace NUMINAMATH_GPT_find_n_for_geometric_series_l347_34722

theorem find_n_for_geometric_series
  (n : ℝ)
  (a1 : ℝ := 12)
  (a2 : ℝ := 4)
  (r1 : ℝ)
  (S1 : ℝ)
  (b1 : ℝ := 12)
  (b2 : ℝ := 4 + n)
  (r2 : ℝ)
  (S2 : ℝ) :
  (r1 = a2 / a1) →
  (S1 = a1 / (1 - r1)) →
  (S2 = 4 * S1) →
  (r2 = b2 / b1) →
  (S2 = b1 / (1 - r2)) →
  n = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_n_for_geometric_series_l347_34722


namespace NUMINAMATH_GPT_sum_of_cubes_l347_34740

-- Definitions
noncomputable def p : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def r : ℂ := sorry

-- Roots conditions
axiom h_root_p : p^3 - 2 * p^2 + 3 * p - 4 = 0
axiom h_root_q : q^3 - 2 * q^2 + 3 * q - 4 = 0
axiom h_root_r : r^3 - 2 * r^2 + 3 * r - 4 = 0

-- Vieta's conditions
axiom h_sum : p + q + r = 2
axiom h_product_pairs : p * q + q * r + r * p = 3
axiom h_product : p * q * r = 4

-- Goal
theorem sum_of_cubes : p^3 + q^3 + r^3 = 2 :=
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l347_34740


namespace NUMINAMATH_GPT_system1_l347_34733

theorem system1 {x y : ℝ} 
  (h1 : x + y = 3) 
  (h2 : x - y = 1) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_system1_l347_34733


namespace NUMINAMATH_GPT_arithmetic_progression_ratio_l347_34770

variable {α : Type*} [LinearOrder α] [Field α]

theorem arithmetic_progression_ratio (a d : α) (h : 15 * a + 105 * d = 3 * (8 * a + 28 * d)) : a / d = 7 / 3 := 
by sorry

end NUMINAMATH_GPT_arithmetic_progression_ratio_l347_34770
