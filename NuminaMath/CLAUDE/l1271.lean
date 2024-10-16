import Mathlib

namespace NUMINAMATH_CALUDE_total_shaded_area_l1271_127172

/-- The total shaded area of two squares with inscribed circles -/
theorem total_shaded_area (small_side large_side small_radius large_radius : ℝ)
  (h1 : small_side = 6)
  (h2 : large_side = 12)
  (h3 : small_radius = 3)
  (h4 : large_radius = 6) :
  (small_side ^ 2 - π * small_radius ^ 2) + (large_side ^ 2 - π * large_radius ^ 2) = 180 - 45 * π := by
  sorry

end NUMINAMATH_CALUDE_total_shaded_area_l1271_127172


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1271_127144

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 4 * y = -7) ∧ 
    (6 * x - 5 * y = 9) ∧ 
    (x = 71 / 9) ∧ 
    (y = 23 / 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1271_127144


namespace NUMINAMATH_CALUDE_certain_number_addition_l1271_127128

theorem certain_number_addition (x : ℝ) (h : 5 * x = 60) : x + 34 = 46 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_addition_l1271_127128


namespace NUMINAMATH_CALUDE_max_value_expression_l1271_127171

theorem max_value_expression (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) ≤ 3/2 * (c^2 + d^2)) ∧
  (∃ y : ℝ, 3 * (c - y) * (y + Real.sqrt (y^2 + d^2)) = 3/2 * (c^2 + d^2)) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l1271_127171


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1271_127105

/-- The eccentricity of a hyperbola with special properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (x = c ∨ x = -c) → 
    (y = b^2 / a ∨ y = -b^2 / a)) →
  2 * c = 2 * b^2 / a →
  c^2 = a^2 * (e^2 - 1) →
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1271_127105


namespace NUMINAMATH_CALUDE_disprove_combined_average_formula_l1271_127104

theorem disprove_combined_average_formula :
  ∃ (a b : ℕ+), a ≠ b ∧
    ∀ (m n : ℕ+), m ≠ n →
      (m.val * a.val + n.val * b.val) / (m.val + n.val) ≠ (a.val + b.val) / 2 := by
  sorry

end NUMINAMATH_CALUDE_disprove_combined_average_formula_l1271_127104


namespace NUMINAMATH_CALUDE_angle_four_value_l1271_127110

theorem angle_four_value (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 + 70 + 40 = 180)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_four_value_l1271_127110


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l1271_127164

/-- Calculates the required reduction in food consumption per student to maintain
    the same total cost given a decrease in student population and an increase in food price. -/
theorem food_consumption_reduction
  (initial_students : ℕ)
  (initial_price : ℝ)
  (student_decrease_rate : ℝ)
  (price_increase_rate : ℝ)
  (h1 : student_decrease_rate = 0.1)
  (h2 : price_increase_rate = 0.2)
  (h3 : initial_students > 0)
  (h4 : initial_price > 0) :
  let new_students : ℝ := initial_students * (1 - student_decrease_rate)
  let new_price : ℝ := initial_price * (1 + price_increase_rate)
  let consumption_ratio : ℝ := (initial_students * initial_price) / (new_students * new_price)
  abs (1 - consumption_ratio - 0.0741) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_food_consumption_reduction_l1271_127164


namespace NUMINAMATH_CALUDE_problem_solution_l1271_127151

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^4 + 3*y^3 + 10) / 7 = 283/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1271_127151


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1271_127167

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {3, 4, 5}

theorem complement_union_theorem :
  (Set.compl A ∩ U) ∪ B = {1, 3, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1271_127167


namespace NUMINAMATH_CALUDE_sqrt_less_implies_less_l1271_127161

theorem sqrt_less_implies_less (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a < Real.sqrt b → a < b :=
by sorry

end NUMINAMATH_CALUDE_sqrt_less_implies_less_l1271_127161


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1271_127115

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 2) ↔ (∃ x : ℝ, x < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1271_127115


namespace NUMINAMATH_CALUDE_sodas_bought_example_l1271_127125

/-- Given a total cost, sandwich price, number of sandwiches, and soda price,
    calculate the number of sodas bought. -/
def sodas_bought (total_cost sandwich_price num_sandwiches soda_price : ℚ) : ℚ :=
  (total_cost - num_sandwiches * sandwich_price) / soda_price

theorem sodas_bought_example : 
  sodas_bought 8.38 2.45 2 0.87 = 4 := by sorry

end NUMINAMATH_CALUDE_sodas_bought_example_l1271_127125


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1271_127184

theorem inequality_solution_range (a : ℝ) : 
  (∀ x, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1271_127184


namespace NUMINAMATH_CALUDE_weighted_power_inequality_l1271_127102

theorem weighted_power_inequality (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p := by
  sorry

end NUMINAMATH_CALUDE_weighted_power_inequality_l1271_127102


namespace NUMINAMATH_CALUDE_set_intersection_union_problem_l1271_127153

theorem set_intersection_union_problem (a b : ℝ) :
  let M : Set ℝ := {3, 2^a}
  let N : Set ℝ := {a, b}
  (M ∩ N = {2}) → (M ∪ N = {1, 2, 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_union_problem_l1271_127153


namespace NUMINAMATH_CALUDE_bennys_work_days_l1271_127195

/-- Given that Benny worked 3 hours a day for a total of 18 hours,
    prove that he worked for 6 days. -/
theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 18 → days * hours_per_day = total_hours → days = 6 := by
  sorry


end NUMINAMATH_CALUDE_bennys_work_days_l1271_127195


namespace NUMINAMATH_CALUDE_largest_among_three_l1271_127129

theorem largest_among_three (sin2 log132 log1213 : ℝ) 
  (h1 : 0 < sin2 ∧ sin2 < 1)
  (h2 : log132 < 0)
  (h3 : log1213 > 1) :
  log1213 = max sin2 (max log132 log1213) :=
sorry

end NUMINAMATH_CALUDE_largest_among_three_l1271_127129


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1271_127131

/-- A line passing through two points (2, 9) and (4, 13) intersects the y-axis at (0, 5) -/
theorem line_intersection_y_axis :
  ∀ (f : ℝ → ℝ),
  (f 2 = 9) →
  (f 4 = 13) →
  (∀ x y, f x = y ↔ y = 2*x + 5) →
  f 0 = 5 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1271_127131


namespace NUMINAMATH_CALUDE_count_values_for_sum_20_main_theorem_l1271_127159

def count_integer_values (n : ℕ) : ℕ :=
  (Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))).card

theorem count_values_for_sum_20 :
  count_integer_values 20 = 6 :=
sorry

theorem main_theorem :
  ∃ (S : Finset ℤ),
    S.card = 6 ∧
    ∀ (a b c : ℕ),
      a > 0 → b > 0 → c > 0 →
      a + b + c = 20 →
      (a + b : ℤ) / (c : ℤ) ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_values_for_sum_20_main_theorem_l1271_127159


namespace NUMINAMATH_CALUDE_hobby_store_sales_l1271_127156

/-- The combined sales of trading cards in June and July -/
def combined_sales (normal_sales : ℕ) (june_extra : ℕ) : ℕ :=
  (normal_sales + june_extra) + normal_sales

/-- Theorem stating the combined sales of trading cards in June and July -/
theorem hobby_store_sales : combined_sales 21122 3922 = 46166 := by
  sorry

end NUMINAMATH_CALUDE_hobby_store_sales_l1271_127156


namespace NUMINAMATH_CALUDE_jelly_beans_problem_l1271_127143

theorem jelly_beans_problem (b c : ℕ) : 
  b = 3 * c →                   -- Initially, blueberry count is 3 times cherry count
  b - 20 = 4 * (c - 20) →       -- After eating 20 of each, blueberry count is 4 times cherry count
  b = 180                       -- Prove that initial blueberry count was 180
  := by sorry

end NUMINAMATH_CALUDE_jelly_beans_problem_l1271_127143


namespace NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l1271_127162

theorem sqrt_720_equals_12_sqrt_5 : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_equals_12_sqrt_5_l1271_127162


namespace NUMINAMATH_CALUDE_bisection_method_step_next_interval_is_1_5_to_2_l1271_127145

def f (x : ℝ) := x^3 - x - 5

theorem bisection_method_step (a b : ℝ) (hab : a < b) (hf : f a * f b < 0) :
  let m := (a + b) / 2
  (f a * f m < 0 ∧ (m, b) = (1.5, 2)) ∨
  (f m * f b < 0 ∧ (a, m) = (1.5, 2)) :=
sorry

theorem next_interval_is_1_5_to_2 :
  let a := 1
  let b := 2
  let m := (a + b) / 2
  m = 1.5 ∧ f a * f b < 0 →
  (1.5, 2) = (let m := (a + b) / 2; if f a * f m < 0 then (a, m) else (m, b)) :=
sorry

end NUMINAMATH_CALUDE_bisection_method_step_next_interval_is_1_5_to_2_l1271_127145


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1271_127191

-- Define the circles M₁ and M₂
def M₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def M₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the trajectory of the center of the moving circle M
def trajectory (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- State the theorem
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
    (∃ (R : ℝ), 
      (∀ (x₁ y₁ : ℝ), M₁ x₁ y₁ → (x - x₁)^2 + (y - y₁)^2 = (1 + R)^2) ∧
      (∀ (x₂ y₂ : ℝ), M₂ x₂ y₂ → (x - x₂)^2 + (y - y₂)^2 = (3 - R)^2)) →
    trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1271_127191


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_3_l1271_127177

theorem least_prime_factor_of_5_4_minus_5_3 :
  Nat.minFac (5^4 - 5^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_4_minus_5_3_l1271_127177


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_closed_interval_l1271_127168

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x ≥ 3, y = Real.log (x + 1) / Real.log (1/2)}
def N : Set ℝ := {x | x^2 + 2*x - 3 ≤ 0}

-- State the theorem
theorem M_intersect_N_eq_closed_interval :
  M ∩ N = Set.Icc (-3) (-2) := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_closed_interval_l1271_127168


namespace NUMINAMATH_CALUDE_definite_integral_x_cubed_l1271_127141

theorem definite_integral_x_cubed : ∫ (x : ℝ) in (-1)..(1), x^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_x_cubed_l1271_127141


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1271_127140

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ 
      (∀ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 → 
        (|k₁| + |k₂| ≥ |((P.2) / (P.1 - a))| + |((P.2) / (P.1 + a))|))) →
  (∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0 ∧ |k₁| + |k₂| = 1) →
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1271_127140


namespace NUMINAMATH_CALUDE_crate_height_difference_l1271_127136

/-- The height difference between two crate packing methods for cylindrical pipes -/
theorem crate_height_difference (n : ℕ) (d : ℝ) :
  let h_direct := n * d
  let h_staggered := (n / 2) * (d + d * Real.sqrt 3 / 2)
  n = 200 ∧ d = 12 →
  h_direct - h_staggered = 120 - 60 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_crate_height_difference_l1271_127136


namespace NUMINAMATH_CALUDE_intersection_M_N_l1271_127108

def M : Set ℝ := {x : ℝ | |x - 1| ≤ 1}
def N : Set ℝ := {x : ℝ | Real.log x > 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1271_127108


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l1271_127122

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Generates the next triangle in the sequence based on the current triangle -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨1003, 1004, 1005, sorry⟩
  | n + 1 => (triangleSequence n).bind nextTriangle

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last existing triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 753 / 128 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l1271_127122


namespace NUMINAMATH_CALUDE_diamond_calculation_l1271_127155

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a + 1 / b

-- Theorem statement
theorem diamond_calculation :
  (diamond (diamond 3 4) 5) - (diamond 3 (diamond 4 5)) = 89 / 420 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l1271_127155


namespace NUMINAMATH_CALUDE_E_parity_2023_2024_2025_l1271_127106

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 2) + E (n + 1) + E n

theorem E_parity_2023_2024_2025 :
  (E 2023 % 2, E 2024 % 2, E 2025 % 2) = (1, 1, 1) := by
  sorry

end NUMINAMATH_CALUDE_E_parity_2023_2024_2025_l1271_127106


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1271_127174

theorem cubic_root_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 7*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 22 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1271_127174


namespace NUMINAMATH_CALUDE_quadratic_form_equivalence_l1271_127188

theorem quadratic_form_equivalence : ∀ x : ℝ, x^2 + 6*x - 2 = (x + 3)^2 - 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_equivalence_l1271_127188


namespace NUMINAMATH_CALUDE_bianca_deleted_files_l1271_127117

/-- The number of pictures Bianca deleted -/
def pictures : ℕ := 5

/-- The number of songs Bianca deleted -/
def songs : ℕ := 12

/-- The number of text files Bianca deleted -/
def text_files : ℕ := 10

/-- The number of video files Bianca deleted -/
def video_files : ℕ := 6

/-- The total number of files Bianca deleted -/
def total_files : ℕ := pictures + songs + text_files + video_files

theorem bianca_deleted_files : total_files = 33 := by
  sorry

end NUMINAMATH_CALUDE_bianca_deleted_files_l1271_127117


namespace NUMINAMATH_CALUDE_diophantine_equation_equivalence_l1271_127198

/-- Given non-square integers a and b, the existence of a non-trivial integer solution
    to x^2 - ay^2 - bz^2 + abw^2 = 0 is equivalent to the existence of a non-trivial
    integer solution to x^2 - ay^2 - bz^2 = 0 -/
theorem diophantine_equation_equivalence (a b : ℤ) 
  (ha : ¬ ∃ (n : ℤ), n^2 = a) (hb : ¬ ∃ (n : ℤ), n^2 = b) :
  (∃ (x y z w : ℤ), x^2 - a*y^2 - b*z^2 + a*b*w^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)) ↔
  (∃ (x y z : ℤ), x^2 - a*y^2 - b*z^2 = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by sorry


end NUMINAMATH_CALUDE_diophantine_equation_equivalence_l1271_127198


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1271_127138

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), A = 16 ∧ B = 4 ∧ C = -16 ∧
  ∀ (x : ℝ), x ≠ 2 → x ≠ 4 →
    8 * x^2 / ((x - 4) * (x - 2)^3) = A / (x - 4) + B / (x - 2) + C / (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1271_127138


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l1271_127181

theorem largest_prime_divisor_test (m : ℕ) : 
  700 ≤ m → m ≤ 750 → 
  (∀ p : ℕ, p.Prime → p ≤ 23 → m % p ≠ 0) → 
  m.Prime :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l1271_127181


namespace NUMINAMATH_CALUDE_greatest_three_digit_sum_with_reversal_l1271_127109

/-- Reverses a three-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

/-- Checks if a number is three-digit -/
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem greatest_three_digit_sum_with_reversal :
  ∀ n : ℕ, isThreeDigit n → n + reverse n ≤ 1211 → n ≤ 952 := by
  sorry

#check greatest_three_digit_sum_with_reversal

end NUMINAMATH_CALUDE_greatest_three_digit_sum_with_reversal_l1271_127109


namespace NUMINAMATH_CALUDE_anna_final_mark_l1271_127160

/-- Calculates the final mark given term mark, exam mark, and their respective weights -/
def calculate_final_mark (term_mark : ℝ) (exam_mark : ℝ) (term_weight : ℝ) (exam_weight : ℝ) : ℝ :=
  term_mark * term_weight + exam_mark * exam_weight

/-- Anna's final mark calculation -/
theorem anna_final_mark :
  calculate_final_mark 80 90 0.7 0.3 = 83 := by
  sorry

#eval calculate_final_mark 80 90 0.7 0.3

end NUMINAMATH_CALUDE_anna_final_mark_l1271_127160


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_l1271_127107

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  (Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2)) ≤ 9 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_l1271_127107


namespace NUMINAMATH_CALUDE_equation_solution_l1271_127179

theorem equation_solution : 
  ∃! x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1271_127179


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1271_127137

-- Define the function f
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = -120 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1271_127137


namespace NUMINAMATH_CALUDE_candy_chocolate_choices_l1271_127165

theorem candy_chocolate_choices (num_candies num_chocolates : ℕ) : 
  num_candies = 2 → num_chocolates = 3 → num_candies + num_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_choices_l1271_127165


namespace NUMINAMATH_CALUDE_some_negative_numbers_satisfy_inequality_l1271_127126

theorem some_negative_numbers_satisfy_inequality :
  (∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0) ↔
  (∃ x₀ : ℝ, x₀ < 0 ∧ (1 + x₀) * (1 - 9 * x₀) > 0) :=
by sorry

end NUMINAMATH_CALUDE_some_negative_numbers_satisfy_inequality_l1271_127126


namespace NUMINAMATH_CALUDE_points_ABD_collinear_l1271_127112

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define points A, B, C, D
variable (A B C D : V)

-- State the theorem
theorem points_ABD_collinear
  (h_not_collinear : ¬ ∃ (k : ℝ), a = k • b)
  (h_AB : B - A = a + 2 • b)
  (h_BC : C - B = -3 • a + 7 • b)
  (h_CD : D - C = 4 • a - 5 • b) :
  ∃ (k : ℝ), D - A = k • (B - A) :=
sorry

end NUMINAMATH_CALUDE_points_ABD_collinear_l1271_127112


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1271_127157

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B : Set ℝ := {x | -2 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1271_127157


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1271_127147

theorem arithmetic_calculations :
  (58 + 15 * 4 = 118) ∧
  (216 - 72 / 8 = 207) ∧
  ((358 - 295) / 7 = 9) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1271_127147


namespace NUMINAMATH_CALUDE_triangle_equations_l1271_127158

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)  -- Side lengths
  (A B C : ℝ)  -- Angles in radians
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- Side lengths are positive
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)  -- Angles are positive
  (h7 : A + B + C = π)  -- Sum of angles is π

-- Define the theorem
theorem triangle_equations (t : Triangle) (h : t.A = π/3) :
  t.a * Real.sin t.C - Real.sqrt 3 * t.c * Real.cos t.A = 0 ∧
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) ∧
  Real.sqrt 3 * t.b * Real.sin t.A - t.a * Real.cos t.C = (t.c + t.b) * Real.cos t.A :=
by sorry

end NUMINAMATH_CALUDE_triangle_equations_l1271_127158


namespace NUMINAMATH_CALUDE_total_balls_in_bag_l1271_127166

/-- The number of balls of each color in the bag -/
structure BagContents where
  white : ℕ
  green : ℕ
  yellow : ℕ
  red : ℕ
  purple : ℕ

/-- The probability of choosing a ball that is neither red nor purple -/
def prob_not_red_or_purple (bag : BagContents) : ℚ :=
  (bag.white + bag.green + bag.yellow : ℚ) / (bag.white + bag.green + bag.yellow + bag.red + bag.purple)

/-- The theorem stating the total number of balls in the bag -/
theorem total_balls_in_bag (bag : BagContents) 
  (h1 : bag.white = 10)
  (h2 : bag.green = 30)
  (h3 : bag.yellow = 10)
  (h4 : bag.red = 47)
  (h5 : bag.purple = 3)
  (h6 : prob_not_red_or_purple bag = 1/2) :
  bag.white + bag.green + bag.yellow + bag.red + bag.purple = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_bag_l1271_127166


namespace NUMINAMATH_CALUDE_ethan_present_count_l1271_127189

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- The difference in presents between Alissa and Ethan -/
def present_difference : ℕ := 22

theorem ethan_present_count : ethan_presents = alissa_presents - present_difference := by
  sorry

end NUMINAMATH_CALUDE_ethan_present_count_l1271_127189


namespace NUMINAMATH_CALUDE_number_equality_l1271_127111

theorem number_equality (T : ℝ) : (1/3 : ℝ) * (1/8 : ℝ) * T = (1/4 : ℝ) * (1/6 : ℝ) * 150 → T = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1271_127111


namespace NUMINAMATH_CALUDE_angle_E_measure_l1271_127187

-- Define the hexagon and its angles
def Hexagon (A B C D E F : ℝ) : Prop :=
  -- Convexity is implied by the sum of angles being 720°
  A + B + C + D + E + F = 720 ∧
  -- Angles A, C, and D are congruent
  A = C ∧ A = D ∧
  -- Angle B is 20 degrees more than angle A
  B = A + 20 ∧
  -- Angles E and F are congruent
  E = F ∧
  -- Angle A is 30 degrees less than angle E
  A + 30 = E

-- Theorem statement
theorem angle_E_measure (A B C D E F : ℝ) :
  Hexagon A B C D E F → E = 158 := by
  sorry

end NUMINAMATH_CALUDE_angle_E_measure_l1271_127187


namespace NUMINAMATH_CALUDE_max_first_term_arithmetic_progression_l1271_127199

def arithmetic_progression (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n+1 => arithmetic_progression a₁ d n + d

def sum_arithmetic_progression (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem max_first_term_arithmetic_progression 
  (a₁ : ℚ) (d : ℚ) 
  (h₁ : ∃ (n : ℕ), sum_arithmetic_progression a₁ d 4 = n)
  (h₂ : ∃ (m : ℕ), sum_arithmetic_progression a₁ d 7 = m)
  (h₃ : a₁ ≤ 2/3) :
  a₁ ≤ 9/14 :=
sorry

end NUMINAMATH_CALUDE_max_first_term_arithmetic_progression_l1271_127199


namespace NUMINAMATH_CALUDE_circle_distance_characterization_l1271_127139

/-- Given two concentric circles C and S centered at P with radii r and s respectively,
    where s < r, and B is a point within S, this theorem characterizes the set of
    points A such that the distance from A to B is less than the distance from A
    to any point on circle C. -/
theorem circle_distance_characterization
  (P B : EuclideanSpace ℝ (Fin 2))  -- Points in 2D real Euclidean space
  (r s : ℝ)  -- Radii of circles C and S
  (h_s_lt_r : s < r)  -- Condition that s < r
  (h_B_in_S : ‖B - P‖ ≤ s)  -- B is within circle S
  (A : EuclideanSpace ℝ (Fin 2))  -- Arbitrary point A
  : (∀ (C : EuclideanSpace ℝ (Fin 2)), ‖C - P‖ = r → ‖A - B‖ < ‖A - C‖) ↔
    ‖A - B‖ < r - s :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_characterization_l1271_127139


namespace NUMINAMATH_CALUDE_expression_value_l1271_127119

theorem expression_value : (49 + 5)^2 - (5^2 + 49^2) = 490 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1271_127119


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1271_127123

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1271_127123


namespace NUMINAMATH_CALUDE_interior_lattice_points_collinear_l1271_127176

/-- A lattice point in the plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle in the plane -/
structure Triangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Check if a point is inside a triangle -/
def isInside (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if a point is on the boundary of a triangle -/
def isOnBoundary (p : LatticePoint) (t : Triangle) : Prop := sorry

/-- Check if points are collinear -/
def areCollinear (points : List LatticePoint) : Prop := sorry

/-- The main theorem -/
theorem interior_lattice_points_collinear (T : Triangle) :
  (∀ p : LatticePoint, isOnBoundary p T → (p = T.A ∨ p = T.B ∨ p = T.C)) →
  (∃! (points : List LatticePoint), points.length = 4 ∧ 
    (∀ p ∈ points, isInside p T) ∧
    (∀ p : LatticePoint, isInside p T → p ∈ points)) →
  ∃ (points : List LatticePoint), points.length = 4 ∧
    (∀ p ∈ points, isInside p T) ∧ areCollinear points :=
by sorry


end NUMINAMATH_CALUDE_interior_lattice_points_collinear_l1271_127176


namespace NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1271_127133

/-- Given a banker's gain and true discount on a bill due in 1 year,
    calculate the rate of interest per annum. -/
def calculate_interest_rate (bankers_gain : ℚ) (true_discount : ℚ) : ℚ :=
  bankers_gain / true_discount

/-- Theorem stating that for the given banker's gain and true discount,
    the calculated interest rate is 12% -/
theorem interest_rate_is_twelve_percent :
  let bankers_gain : ℚ := 78/10
  let true_discount : ℚ := 65
  calculate_interest_rate bankers_gain true_discount = 12/100 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_twelve_percent_l1271_127133


namespace NUMINAMATH_CALUDE_system_solvable_iff_a_in_range_l1271_127101

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*(a - x - y) = 64 ∧
  y = 8 * Real.sin (x - 2*b) - 6 * Real.cos (x - 2*b)

-- Theorem statement
theorem system_solvable_iff_a_in_range :
  ∀ a : ℝ, (∃ b x y : ℝ, system a b x y) ↔ -18 ≤ a ∧ a ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_system_solvable_iff_a_in_range_l1271_127101


namespace NUMINAMATH_CALUDE_exists_good_interval_and_fixed_point_l1271_127103

/-- Definition of a good interval for a function f on [a, b] --/
def is_good_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b) ∨ 
  (∀ x ∈ Set.Icc a b, f x ∉ Set.Icc a b)

/-- Main theorem --/
theorem exists_good_interval_and_fixed_point 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ a b : ℝ, a < b → f a - f b > b - a) : 
  (∃ c d : ℝ, c < d ∧ is_good_interval f c d) ∧ 
  (∃ x₀ : ℝ, f x₀ = x₀) := by
  sorry


end NUMINAMATH_CALUDE_exists_good_interval_and_fixed_point_l1271_127103


namespace NUMINAMATH_CALUDE_pet_store_birds_l1271_127120

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 36 := by sorry

end NUMINAMATH_CALUDE_pet_store_birds_l1271_127120


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solution_l1271_127183

theorem cubic_equation_integer_solution (m : ℤ) :
  (∃ x : ℤ, x^3 - m*x^2 + m*x - (m^2 + 1) = 0) ↔ (m = -3 ∨ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solution_l1271_127183


namespace NUMINAMATH_CALUDE_skidding_distance_speed_relation_l1271_127135

theorem skidding_distance_speed_relation 
  (a b : ℝ) 
  (h1 : b = a * 60^2) 
  (h2 : 3 * b = a * x^2) : 
  x = 60 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_skidding_distance_speed_relation_l1271_127135


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1271_127121

theorem fraction_sum_equals_decimal : 
  2/10 - 5/100 + 3/1000 + 8/10000 = 0.1538 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1271_127121


namespace NUMINAMATH_CALUDE_redToGreenGrapeRatio_l1271_127193

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  raspberries : ℕ
  greenGrapes : ℕ
  redGrapes : ℕ

/-- The properties of our specific fruit salad --/
def fruitSaladProperties (fs : FruitSalad) : Prop :=
  fs.raspberries = fs.greenGrapes - 5 ∧
  fs.raspberries + fs.greenGrapes + fs.redGrapes = 102 ∧
  fs.redGrapes = 67

/-- The theorem stating the ratio of red grapes to green grapes --/
theorem redToGreenGrapeRatio (fs : FruitSalad) 
  (h : fruitSaladProperties fs) : 
  fs.redGrapes * 20 = fs.greenGrapes * 67 := by
  sorry

#check redToGreenGrapeRatio

end NUMINAMATH_CALUDE_redToGreenGrapeRatio_l1271_127193


namespace NUMINAMATH_CALUDE_duck_race_charity_l1271_127197

/-- The amount of money raised in a charity duck race -/
def charity_amount (regular_price : ℝ) (large_price : ℝ) (regular_sold : ℕ) (large_sold : ℕ) : ℝ :=
  regular_price * (regular_sold : ℝ) + large_price * (large_sold : ℝ)

/-- Theorem stating the amount raised in the specific duck race -/
theorem duck_race_charity : 
  charity_amount 3 5 221 185 = 1588 := by
  sorry

end NUMINAMATH_CALUDE_duck_race_charity_l1271_127197


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_three_l1271_127152

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - 2m - 3) + (m + 1)i is a purely imaginary number, m = 3. -/
theorem purely_imaginary_implies_m_eq_three (m : ℝ) :
  is_purely_imaginary ((m^2 - 2*m - 3 : ℝ) + (m + 1)*I) → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_m_eq_three_l1271_127152


namespace NUMINAMATH_CALUDE_equality_or_sum_zero_l1271_127134

theorem equality_or_sum_zero (a b c d : ℝ) :
  (a + b) / (b + c) = (c + d) / (d + a) →
  (a = c ∨ a + b + c + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_equality_or_sum_zero_l1271_127134


namespace NUMINAMATH_CALUDE_triangle_angle_b_l1271_127118

/-- In a triangle ABC, given that a cos B - b cos A = c and C = π/5, prove that B = 3π/10 -/
theorem triangle_angle_b (a b c A B C : ℝ) : 
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_l1271_127118


namespace NUMINAMATH_CALUDE_min_value_cosine_function_l1271_127113

theorem min_value_cosine_function :
  ∀ x : ℝ, 2 * Real.cos x - 1 ≥ -3 ∧ ∃ x : ℝ, 2 * Real.cos x - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cosine_function_l1271_127113


namespace NUMINAMATH_CALUDE_price_difference_l1271_127185

/-- Calculates the difference between the final retail price and the average price
    customers paid for the first 150 garments under a special pricing scheme. -/
theorem price_difference (original_price : ℝ) (first_increase : ℝ) (second_increase : ℝ)
  (special_rate1 : ℝ) (special_rate2 : ℝ) (special_quantity1 : ℕ) (special_quantity2 : ℕ)
  (h1 : original_price = 50)
  (h2 : first_increase = 0.3)
  (h3 : second_increase = 0.15)
  (h4 : special_rate1 = 0.7)
  (h5 : special_rate2 = 0.85)
  (h6 : special_quantity1 = 50)
  (h7 : special_quantity2 = 100) :
  let final_price := original_price * (1 + first_increase) * (1 + second_increase)
  let special_price1 := final_price * special_rate1
  let special_price2 := final_price * special_rate2
  let total_special_price := special_price1 * special_quantity1 + special_price2 * special_quantity2
  let avg_special_price := total_special_price / (special_quantity1 + special_quantity2)
  final_price - avg_special_price = 14.95 := by
sorry

end NUMINAMATH_CALUDE_price_difference_l1271_127185


namespace NUMINAMATH_CALUDE_equation_solution_l1271_127192

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem equation_solution (a : ℕ+) :
  (∃ n : ℕ+, 7 * a * n - 3 * factorial n = 2020) ↔ (a = 68 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1271_127192


namespace NUMINAMATH_CALUDE_percentage_of_men_l1271_127194

/-- Represents the composition of employees in a company -/
structure Company where
  men : ℝ
  women : ℝ
  men_french : ℝ
  women_french : ℝ

/-- The company satisfies the given conditions -/
def valid_company (c : Company) : Prop :=
  c.men + c.women = 100 ∧
  c.men_french = 0.6 * c.men ∧
  c.women_french = 0.35 * c.women ∧
  c.men_french + c.women_french = 50

/-- The theorem stating that 60% of the company employees are men -/
theorem percentage_of_men (c : Company) (h : valid_company c) : c.men = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_l1271_127194


namespace NUMINAMATH_CALUDE_cyclic_inequality_l1271_127170

theorem cyclic_inequality (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l1271_127170


namespace NUMINAMATH_CALUDE_inverse_x_equals_three_l1271_127100

theorem inverse_x_equals_three (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 + 1/27 = x*y) : 1/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_x_equals_three_l1271_127100


namespace NUMINAMATH_CALUDE_apple_price_theorem_l1271_127169

/-- The original price per kilogram of apples -/
def original_price : ℝ := 5

/-- The discounted price for 10 kilograms of apples -/
def discounted_total : ℝ := 30

/-- The number of kilograms of apples -/
def kg_amount : ℝ := 10

/-- The discount percentage -/
def discount_percent : ℝ := 40

/-- Theorem: The original price per kilogram of apples is $5 -/
theorem apple_price_theorem :
  original_price = discounted_total / (kg_amount * (1 - discount_percent / 100)) :=
sorry

end NUMINAMATH_CALUDE_apple_price_theorem_l1271_127169


namespace NUMINAMATH_CALUDE_coordinate_change_l1271_127146

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def is_basis (v : Fin 3 → V) : Prop :=
  LinearIndependent ℝ v ∧ Submodule.span ℝ (Set.range v) = ⊤

theorem coordinate_change
  (a b c : V)
  (h1 : is_basis (![a, b, c]))
  (h2 : is_basis (![a - b, a + b, c]))
  (p : V)
  (h3 : p = 4 • a + 2 • b + (-1) • c) :
  ∃ (x y z : ℝ), p = x • (a - b) + y • (a + b) + z • c ∧ x = 1 ∧ y = 3 ∧ z = -1 :=
sorry

end NUMINAMATH_CALUDE_coordinate_change_l1271_127146


namespace NUMINAMATH_CALUDE_garrett_roses_count_l1271_127116

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The difference in the number of roses between Mrs. Santiago and Mrs. Garrett -/
def difference : ℕ := 34

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := santiago_roses - difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_roses_count_l1271_127116


namespace NUMINAMATH_CALUDE_f_properties_l1271_127149

noncomputable def f (x : ℝ) : ℝ := 1/2 * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 7/4
  let min_value : ℝ := (5 + Real.sqrt 3) / 4
  let interval : Set ℝ := Set.Icc (Real.pi / 12) (Real.pi / 4)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ t : ℝ, t > 0 → (∀ x : ℝ, f (x + t) = f x) → t ≥ period) ∧
  (∃ x ∈ interval, f x = max_value ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, f x = min_value ∧ ∀ y ∈ interval, f y ≥ f x) ∧
  (f (Real.pi / 6) = max_value) ∧
  (f (Real.pi / 12) = min_value) ∧
  (f (Real.pi / 4) = min_value) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1271_127149


namespace NUMINAMATH_CALUDE_sum_in_first_quadrant_l1271_127180

/-- Given complex numbers z₁ and z₂, prove that their sum is in the first quadrant -/
theorem sum_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 1 - I) : 
  let z := z₁ + z₂
  (z.re > 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_sum_in_first_quadrant_l1271_127180


namespace NUMINAMATH_CALUDE_at_least_one_negative_l1271_127182

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l1271_127182


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l1271_127132

-- Define a three-digit palindrome
def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 100 * a + 10 * b + a

-- Define the set of all three-digit palindromes
def three_digit_palindromes : Set ℕ :=
  {n : ℕ | is_three_digit_palindrome n}

-- Statement to prove
theorem greatest_common_factor_of_three_digit_palindromes :
  ∃ g : ℕ, g > 0 ∧ 
    (∀ n ∈ three_digit_palindromes, g ∣ n) ∧
    (∀ d : ℕ, d > 0 → (∀ n ∈ three_digit_palindromes, d ∣ n) → d ≤ g) ∧
    g = 101 :=
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_three_digit_palindromes_l1271_127132


namespace NUMINAMATH_CALUDE_original_number_before_increase_l1271_127196

theorem original_number_before_increase (x : ℝ) : x * 1.5 = 525 → x = 350 := by
  sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l1271_127196


namespace NUMINAMATH_CALUDE_book_cost_price_l1271_127127

/-- The cost price of a book sold for $200 with a 20% profit is $166.67 -/
theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 200 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 166.67 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l1271_127127


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1271_127114

theorem complex_expression_simplification (i : ℂ) (h : i^2 = -1) :
  i * (1 - i) - 1 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1271_127114


namespace NUMINAMATH_CALUDE_word_permutations_l1271_127124

-- Define the number of distinct letters in the word
def num_distinct_letters : ℕ := 6

-- Define the function to calculate factorial
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem word_permutations :
  factorial num_distinct_letters = 720 := by
  sorry

end NUMINAMATH_CALUDE_word_permutations_l1271_127124


namespace NUMINAMATH_CALUDE_domino_arrangements_equal_combinations_l1271_127142

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with length 2 and width 1 -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- The number of distinct domino arrangements on a grid -/
def num_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem domino_arrangements_equal_combinations (g : Grid) (d : Domino) :
  g.width = 6 →
  g.height = 5 →
  d.length = 2 →
  d.width = 1 →
  num_arrangements g d 5 = choose 9 5 := by sorry

end NUMINAMATH_CALUDE_domino_arrangements_equal_combinations_l1271_127142


namespace NUMINAMATH_CALUDE_original_banana_count_l1271_127190

/-- The number of bananas Willie and Charles originally had together -/
def total_bananas (willie_bananas : ℝ) (charles_bananas : ℝ) : ℝ :=
  willie_bananas + charles_bananas

/-- Theorem stating that Willie and Charles originally had 83.0 bananas together -/
theorem original_banana_count : total_bananas 48.0 35.0 = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_original_banana_count_l1271_127190


namespace NUMINAMATH_CALUDE_remainder_three_pow_twenty_mod_seven_l1271_127130

theorem remainder_three_pow_twenty_mod_seven : 3^20 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_pow_twenty_mod_seven_l1271_127130


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1271_127154

theorem square_sum_theorem (x y : ℝ) (h1 : (x + y)^2 = 25) (h2 : x * y = 6) : x^2 + y^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1271_127154


namespace NUMINAMATH_CALUDE_function_values_l1271_127175

def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_values (a b c : ℝ) :
  (∀ x, f x a b c ≤ f (-1) a b c) ∧
  (f (-1) a b c = 7) ∧
  (∀ x, f x a b c ≥ f 3 a b c) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f 3 a b c = -25 :=
by sorry

end NUMINAMATH_CALUDE_function_values_l1271_127175


namespace NUMINAMATH_CALUDE_fraction_power_product_l1271_127178

theorem fraction_power_product :
  (1 / 3 : ℚ) ^ 4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1271_127178


namespace NUMINAMATH_CALUDE_servant_cash_payment_l1271_127150

-- Define the problem parameters
def annual_cash_salary : ℕ := 90
def turban_price : ℕ := 70
def months_worked : ℕ := 9
def months_per_year : ℕ := 12

-- Define the theorem
theorem servant_cash_payment :
  let total_annual_salary := annual_cash_salary + turban_price
  let proportion_worked := months_worked / months_per_year
  let earned_amount := (proportion_worked * total_annual_salary : ℚ).floor
  earned_amount - turban_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_servant_cash_payment_l1271_127150


namespace NUMINAMATH_CALUDE_more_male_students_l1271_127163

theorem more_male_students (total : ℕ) (female : ℕ) (h1 : total = 280) (h2 : female = 127) :
  total - female - female = 26 := by
  sorry

end NUMINAMATH_CALUDE_more_male_students_l1271_127163


namespace NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l1271_127148

theorem unique_solution_for_diophantine_equation :
  ∃! (a b : ℕ), 
    Nat.Prime a ∧ 
    b > 0 ∧ 
    9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) ∧
    a = 251 ∧ 
    b = 7 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l1271_127148


namespace NUMINAMATH_CALUDE_xyz_equality_l1271_127173

theorem xyz_equality (x y z : ℕ+) (a b c d : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z)
  (h3 : (x : ℝ) ^ a = (y : ℝ) ^ b)
  (h4 : (y : ℝ) ^ b = (z : ℝ) ^ c)
  (h5 : (z : ℝ) ^ c = 70 ^ d)
  (h6 : 1 / a + 1 / b + 1 / c = 1 / d) :
  x + y = z := by sorry

end NUMINAMATH_CALUDE_xyz_equality_l1271_127173


namespace NUMINAMATH_CALUDE_adjacent_different_colors_l1271_127186

/-- Represents a square on the grid -/
structure Square where
  row : Fin 10
  col : Fin 10

/-- Represents the color of a piece -/
inductive Color
  | White
  | Black

/-- Represents the state of the grid at any point in the process -/
def GridState := Square → Option Color

/-- Represents a single step in the replacement process -/
structure ReplacementStep where
  removed : Square
  placed : Square

/-- The sequence of replacement steps -/
def ReplacementSequence := List ReplacementStep

/-- Two squares are adjacent if they share a common edge -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ (s1.col.val + 1 = s2.col.val ∨ s2.col.val + 1 = s1.col.val)) ∨
  (s1.col = s2.col ∧ (s1.row.val + 1 = s2.row.val ∨ s2.row.val + 1 = s1.row.val))

/-- The initial state of the grid with 91 white pieces -/
def initialState : GridState :=
  sorry

/-- The state of the grid after applying a sequence of replacement steps -/
def applyReplacements (initial : GridState) (steps : ReplacementSequence) : GridState :=
  sorry

/-- Theorem: There exists a point in the replacement process where two adjacent squares have different colored pieces -/
theorem adjacent_different_colors (steps : ReplacementSequence) :
  ∃ (partialSteps : ReplacementSequence) (s1 s2 : Square),
    partialSteps.length < steps.length ∧
    adjacent s1 s2 ∧
    let state := applyReplacements initialState partialSteps
    (state s1).isSome ∧ (state s2).isSome ∧ state s1 ≠ state s2 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_different_colors_l1271_127186
