import Mathlib

namespace NUMINAMATH_CALUDE_intersection_M_N_l4144_414461

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {y | ∃ x ∈ (Set.Ioo 0 2), y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4144_414461


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l4144_414419

theorem incorrect_inequality_transformation :
  ¬(∀ (a b c : ℝ), a * c > b * c → a > b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l4144_414419


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4144_414487

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4144_414487


namespace NUMINAMATH_CALUDE_order_of_abc_l4144_414495

theorem order_of_abc (a b c : ℝ) : 
  a = 2/21 → b = Real.log 1.1 → c = 21/220 → a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l4144_414495


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4144_414418

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l4144_414418


namespace NUMINAMATH_CALUDE_xy_equals_three_l4144_414451

theorem xy_equals_three (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_three_l4144_414451


namespace NUMINAMATH_CALUDE_roots_eighth_power_sum_l4144_414432

theorem roots_eighth_power_sum (x y : ℝ) : 
  x^2 - 2*x*Real.sqrt 2 + 1 = 0 ∧ 
  y^2 - 2*y*Real.sqrt 2 + 1 = 0 ∧ 
  x ≠ y → 
  x^8 + y^8 = 1154 := by sorry

end NUMINAMATH_CALUDE_roots_eighth_power_sum_l4144_414432


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l4144_414489

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 2; 2, 1]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-1, 2; 3, -4]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![5, -6; 1, 0]

theorem matrix_multiplication_result : A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l4144_414489


namespace NUMINAMATH_CALUDE_greatest_k_value_l4144_414402

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 := by
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l4144_414402


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_orthocenter_on_incircle_ratio_l4144_414421

/-- An isosceles triangle with sides a, a, and b -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0

/-- The orthocenter of a triangle -/
def orthocenter (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- The incircle of a triangle -/
def incircle (t : IsoscelesTriangle) : Set (ℝ × ℝ) := sorry

/-- Predicate to check if a point lies on a set -/
def lies_on (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop := p ∈ s

theorem isosceles_triangle_with_orthocenter_on_incircle_ratio
  (t : IsoscelesTriangle)
  (h : lies_on (orthocenter t) (incircle t)) :
  t.a / t.b = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_orthocenter_on_incircle_ratio_l4144_414421


namespace NUMINAMATH_CALUDE_points_form_hyperbola_l4144_414472

/-- The set of points (x,y) defined by x = 2cosh(t) and y = 4sinh(t) for real t forms a hyperbola -/
theorem points_form_hyperbola :
  ∀ (t x y : ℝ), x = 2 * Real.cosh t ∧ y = 4 * Real.sinh t →
  x^2 / 4 - y^2 / 16 = 1 := by
sorry

end NUMINAMATH_CALUDE_points_form_hyperbola_l4144_414472


namespace NUMINAMATH_CALUDE_h_value_l4144_414463

theorem h_value (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 5 ∧ y^2 - 4*h*y = 5 ∧ x^2 + y^2 = 34) → 
  |h| = Real.sqrt (3/2) := by
sorry

end NUMINAMATH_CALUDE_h_value_l4144_414463


namespace NUMINAMATH_CALUDE_common_chord_length_l4144_414427

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0
def circle_C2 (x y : ℝ) : Prop := (x + 3/2)^2 + (y - 3/2)^2 = 11/2

-- Theorem statement
theorem common_chord_length :
  ∃ (a b c d : ℝ),
    (circle_C1 a b ∧ circle_C1 c d) ∧
    (circle_C2 a b ∧ circle_C2 c d) ∧
    (a ≠ c ∨ b ≠ d) ∧
    Real.sqrt ((a - c)^2 + (b - d)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_length_l4144_414427


namespace NUMINAMATH_CALUDE_bart_mixtape_length_l4144_414465

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (first_side_songs : ℕ) (second_side_songs : ℕ) (song_length : ℕ) : ℕ :=
  (first_side_songs + second_side_songs) * song_length

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_length_l4144_414465


namespace NUMINAMATH_CALUDE_termite_ridden_homes_l4144_414459

theorem termite_ridden_homes (total_homes : ℝ) (termite_ridden : ℝ) 
  (h1 : termite_ridden > 0) 
  (h2 : termite_ridden / total_homes ≤ 1) 
  (h3 : (3/4) * (termite_ridden / total_homes) = 1/4) : 
  termite_ridden / total_homes = 1/3 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_homes_l4144_414459


namespace NUMINAMATH_CALUDE_divisors_product_18_l4144_414481

def divisors_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisors_product_18 : divisors_product 18 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_divisors_product_18_l4144_414481


namespace NUMINAMATH_CALUDE_circle_tangent_origin_l4144_414464

-- Define the circle equation
def circle_equation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

-- Define the tangency condition
def tangent_at_origin (D E F : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  ∀ (x y : ℝ), circle_equation x y D E F → x^2 + y^2 ≥ r^2 ∧
  circle_equation 0 0 D E F

-- Theorem statement
theorem circle_tangent_origin (D E F : ℝ) :
  tangent_at_origin D E F → D = 0 ∧ F = 0 ∧ E ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_origin_l4144_414464


namespace NUMINAMATH_CALUDE_tank_capacity_l4144_414401

theorem tank_capacity : ∃ (capacity : ℚ), 
  capacity > 0 ∧ 
  (1/3 : ℚ) * capacity + 180 = (2/3 : ℚ) * capacity ∧ 
  capacity = 540 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l4144_414401


namespace NUMINAMATH_CALUDE_total_stickers_l4144_414470

def initial_stickers : Float := 20.0
def bought_stickers : Float := 26.0
def birthday_stickers : Float := 20.0
def sister_gift : Float := 6.0
def mother_gift : Float := 58.0

theorem total_stickers : 
  initial_stickers + bought_stickers + birthday_stickers + sister_gift + mother_gift = 130.0 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l4144_414470


namespace NUMINAMATH_CALUDE_inequality_system_unique_solution_l4144_414476

/-- A system of inequalities with parameter a and variable x -/
structure InequalitySystem (a : ℝ) :=
  (x : ℤ)
  (ineq1 : x^3 + 3*x^2 - x - 3 > 0)
  (ineq2 : x^2 - 2*a*x - 1 ≤ 0)
  (a_pos : a > 0)

/-- The theorem stating the range of a for which the system has exactly one integer solution -/
theorem inequality_system_unique_solution :
  ∀ a : ℝ, (∃! s : InequalitySystem a, True) ↔ 3/4 ≤ a ∧ a < 4/3 :=
sorry

end NUMINAMATH_CALUDE_inequality_system_unique_solution_l4144_414476


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4144_414490

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1)*x + a ≤ 0}
  (a > 1 → S = {x : ℝ | 1 ≤ x ∧ x ≤ a}) ∧
  (a = 1 → S = {x : ℝ | x = 1}) ∧
  (a < 1 → S = {x : ℝ | a ≤ x ∧ x ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4144_414490


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l4144_414411

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 20 m x 10 m x 5 m is 1000 cubic meters -/
theorem pond_volume_calculation : pond_volume 20 10 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l4144_414411


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4144_414485

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  (1 / x + 3 / y) ≥ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4144_414485


namespace NUMINAMATH_CALUDE_ratio_to_ten_l4144_414417

theorem ratio_to_ten : ∃ x : ℚ, (15 : ℚ) / 1 = x / 10 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_ten_l4144_414417


namespace NUMINAMATH_CALUDE_probability_different_tens_proof_l4144_414484

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_numbers : ℕ := 70

/-- The number of different tens digits available in the range 10 to 79. -/
def available_tens_digits : ℕ := 7

/-- The number of integers for each tens digit. -/
def numbers_per_tens : ℕ := 10

/-- The number of integers to be chosen. -/
def chosen_count : ℕ := 7

/-- The probability of selecting 7 different integers from the range 10 to 79 (inclusive)
    such that each has a different tens digit. -/
def probability_different_tens : ℚ := 10000000 / 93947434

theorem probability_different_tens_proof :
  (numbers_per_tens ^ chosen_count : ℚ) / (total_numbers.choose chosen_count) = probability_different_tens :=
sorry

end NUMINAMATH_CALUDE_probability_different_tens_proof_l4144_414484


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l4144_414447

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Icc 0 2, 
    ∀ y ∈ Set.Icc 0 2, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l4144_414447


namespace NUMINAMATH_CALUDE_no_nonzero_perfect_square_in_sequence_l4144_414446

theorem no_nonzero_perfect_square_in_sequence
  (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k, a k = 1988 ∨ b k = 1988) :
  ∀ k n, n ≠ 0 → a k ≠ n^2 :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_perfect_square_in_sequence_l4144_414446


namespace NUMINAMATH_CALUDE_different_color_probability_l4144_414450

theorem different_color_probability : 
  let total_balls : ℕ := 5
  let white_balls : ℕ := 2
  let black_balls : ℕ := 3
  let probability_different_colors : ℚ := 12 / 25
  (white_balls + black_balls = total_balls) →
  (probability_different_colors = 
    (white_balls * black_balls + black_balls * white_balls) / (total_balls * total_balls)) :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l4144_414450


namespace NUMINAMATH_CALUDE_solve_turtle_problem_l4144_414492

def turtle_problem (owen_initial : ℕ) (johanna_difference : ℕ) : Prop :=
  let johanna_initial : ℕ := owen_initial - johanna_difference
  let owen_after_month : ℕ := owen_initial * 2
  let johanna_after_loss : ℕ := johanna_initial / 2
  let owen_final : ℕ := owen_after_month + johanna_after_loss
  owen_final = 50

theorem solve_turtle_problem :
  turtle_problem 21 5 := by sorry

end NUMINAMATH_CALUDE_solve_turtle_problem_l4144_414492


namespace NUMINAMATH_CALUDE_balloon_permutations_l4144_414453

def balloon_arrangements : ℕ := 1260

theorem balloon_permutations :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  let unique_letters : ℕ := 3
  (total_letters = repeated_l + repeated_o + unique_letters) →
  (balloon_arrangements = (Nat.factorial total_letters) / ((Nat.factorial repeated_l) * (Nat.factorial repeated_o))) :=
by sorry

end NUMINAMATH_CALUDE_balloon_permutations_l4144_414453


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l4144_414475

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := a n

theorem fifth_term_of_arithmetic_sequence
  (a : ℕ → ℤ) (h : ArithmeticSequence a)
  (h10 : nthTerm a 10 = 15)
  (h12 : nthTerm a 12 = 21) :
  nthTerm a 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l4144_414475


namespace NUMINAMATH_CALUDE_triangle_side_length_l4144_414469

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Conditions
  (a = Real.sqrt 3) →
  (Real.sin B = 1 / 2) →
  (C = π / 6) →
  -- Sum of angles in a triangle is π
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Conclusion
  b = 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4144_414469


namespace NUMINAMATH_CALUDE_positive_integer_equation_l4144_414467

theorem positive_integer_equation (m n p : ℕ+) : 
  3 * m.val + 3 / (n.val + 1 / p.val) = 17 → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_equation_l4144_414467


namespace NUMINAMATH_CALUDE_equilateral_condition_obtuse_condition_two_triangles_condition_l4144_414429

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the properties we need to prove
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

def hasTwoConfigurations (a b : ℝ) (B : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), C₁ ≠ C₂ ∧
    (∃ (t₁ t₂ : Triangle), 
      t₁.a = a ∧ t₁.b = b ∧ t₁.B = B ∧ t₁.C = C₁ ∧
      t₂.a = a ∧ t₂.b = b ∧ t₂.B = B ∧ t₂.C = C₂)

-- State the theorems
theorem equilateral_condition (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : t.B = Real.pi / 3) : 
  isEquilateral t := by sorry

theorem obtuse_condition (t : Triangle) 
  (h : Real.cos t.A^2 + Real.sin t.B^2 + Real.sin t.C^2 < 1) : 
  isObtuse t := by sorry

theorem two_triangles_condition :
  hasTwoConfigurations 4 2 (25 * Real.pi / 180) := by sorry

end NUMINAMATH_CALUDE_equilateral_condition_obtuse_condition_two_triangles_condition_l4144_414429


namespace NUMINAMATH_CALUDE_kath_group_admission_cost_l4144_414405

/-- Calculates the total admission cost for a group watching a movie before 6 P.M. -/
def total_admission_cost (regular_price : ℕ) (discount : ℕ) (num_people : ℕ) : ℕ :=
  (regular_price - discount) * num_people

/-- The total admission cost for Kath's group is $30 -/
theorem kath_group_admission_cost :
  let regular_price := 8
  let discount := 3
  let num_people := 6
  total_admission_cost regular_price discount num_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_kath_group_admission_cost_l4144_414405


namespace NUMINAMATH_CALUDE_line_through_point_l4144_414442

/-- 
Given a line with equation 3ax + (2a+1)y = 3a+3 that passes through the point (3, -9),
prove that a = -1.
-/
theorem line_through_point (a : ℝ) : 
  (3 * a * 3 + (2 * a + 1) * (-9) = 3 * a + 3) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l4144_414442


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4144_414441

theorem quadratic_equation_solution (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - b = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 2*y - b = 0 ∧ y = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4144_414441


namespace NUMINAMATH_CALUDE_rectangle_path_ratio_l4144_414415

/-- Represents a rectangle on a lattice grid --/
structure LatticeRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the number of shortest paths between opposite corners of a rectangle --/
def shortestPaths (rect : LatticeRectangle) : ℕ :=
  Nat.choose (rect.width + rect.height) rect.width

/-- Theorem: For a rectangle with height = k * width, the number of paths starting vertically
    is k times the number of paths starting horizontally --/
theorem rectangle_path_ratio {k : ℕ} (rect : LatticeRectangle) 
    (h : rect.height = k * rect.width) :
  shortestPaths ⟨rect.height, rect.width⟩ = k * shortestPaths ⟨rect.width, rect.height⟩ := by
  sorry

#check rectangle_path_ratio

end NUMINAMATH_CALUDE_rectangle_path_ratio_l4144_414415


namespace NUMINAMATH_CALUDE_sum_factorials_perfect_square_l4144_414420

-- Define the sum of factorials from 1! to n!
def sumFactorials (n : ℕ) : ℕ := (Finset.range n).sum (λ i => Nat.factorial (i + 1))

-- Define a predicate for perfect squares
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

-- Theorem statement
theorem sum_factorials_perfect_square :
  ∀ n : ℕ, n > 0 → (isPerfectSquare (sumFactorials n) ↔ n = 1 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_factorials_perfect_square_l4144_414420


namespace NUMINAMATH_CALUDE_hcf_problem_l4144_414434

theorem hcf_problem (a b hcf : ℕ) (h1 : a = 391) (h2 : a ≥ b) 
  (h3 : ∃ (lcm : ℕ), lcm = hcf * 16 * 17 ∧ lcm = a * b / hcf) : hcf = 23 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l4144_414434


namespace NUMINAMATH_CALUDE_johnny_age_multiple_l4144_414426

theorem johnny_age_multiple (current_age : ℕ) (m : ℕ+) : current_age = 8 →
  (current_age + 2 : ℕ) = m * (current_age - 3) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_johnny_age_multiple_l4144_414426


namespace NUMINAMATH_CALUDE_figure_2010_squares_l4144_414466

/-- The number of squares in a figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 1 + 4 * (n - 1)

/-- The theorem stating that Figure 2010 contains 8037 squares -/
theorem figure_2010_squares : num_squares 2010 = 8037 := by
  sorry

end NUMINAMATH_CALUDE_figure_2010_squares_l4144_414466


namespace NUMINAMATH_CALUDE_third_chest_coin_difference_l4144_414473

theorem third_chest_coin_difference (total_gold total_silver : ℕ) 
  (x1 y1 x2 y2 x3 y3 : ℕ) : 
  total_gold = 40 →
  total_silver = 40 →
  x1 + x2 + x3 = total_gold →
  y1 + y2 + y3 = total_silver →
  x1 = y1 + 7 →
  y2 = x2 - 15 →
  y3 - x3 = 22 :=
by sorry

end NUMINAMATH_CALUDE_third_chest_coin_difference_l4144_414473


namespace NUMINAMATH_CALUDE_potion_original_price_l4144_414480

/-- The original price of a potion, given that the discounted price is one-fifth of the original. -/
def original_price (discounted_price : ℝ) : ℝ := 5 * discounted_price

/-- Theorem stating that the original price of the potion is $40, given the conditions. -/
theorem potion_original_price : original_price 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_potion_original_price_l4144_414480


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_ten_l4144_414430

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_ten :
  ∃ X : ℕ, X > 0 ∧ 
  (∃ T : ℕ, T > 0 ∧ is_binary_number T ∧ T = 10 * X) ∧
  (∀ Y : ℕ, Y > 0 → 
    (∃ S : ℕ, S > 0 ∧ is_binary_number S ∧ S = 10 * Y) → 
    X ≤ Y) ∧
  X = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_ten_l4144_414430


namespace NUMINAMATH_CALUDE_orange_juice_glasses_l4144_414491

theorem orange_juice_glasses (total_juice : ℕ) (juice_per_glass : ℕ) (h1 : total_juice = 153) (h2 : juice_per_glass = 30) :
  ∃ (num_glasses : ℕ), num_glasses * juice_per_glass ≥ total_juice ∧
  ∀ (m : ℕ), m * juice_per_glass ≥ total_juice → m ≥ num_glasses :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_glasses_l4144_414491


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l4144_414410

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  population : ℕ
  sampleSize : ℕ
  sampleSize_le_population : sampleSize ≤ population

/-- The probability of an individual being selected in a systematic sampling -/
def selectionProbability (s : SystematicSampling) : ℚ :=
  s.sampleSize / s.population

theorem systematic_sampling_probability 
  (s : SystematicSampling) 
  (h1 : s.population = 121) 
  (h2 : s.sampleSize = 12) : 
  selectionProbability s = 12 / 121 := by
  sorry

#check systematic_sampling_probability

end NUMINAMATH_CALUDE_systematic_sampling_probability_l4144_414410


namespace NUMINAMATH_CALUDE_projection_matrix_l4144_414468

def P : Matrix (Fin 2) (Fin 2) ℚ := !![965/1008, 18/41; 19/34, 23/41]

theorem projection_matrix : P * P = P := by sorry

end NUMINAMATH_CALUDE_projection_matrix_l4144_414468


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l4144_414497

/-- The circle equation: x^2 + y^2 - 2x - 2y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y = 0

/-- The line equation: x + y + 2 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

/-- The maximum distance from a point on the circle to the line is 3√2 -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_equation x y ∧
  (∀ (a b : ℝ), circle_equation a b →
    Real.sqrt ((x - a)^2 + (y - b)^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ (p q : ℝ), circle_equation p q ∧
    Real.sqrt ((x - p)^2 + (y - q)^2) = 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l4144_414497


namespace NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l4144_414414

theorem cubic_root_sum_inverse_squares : 
  ∀ (a b c : ℝ), 
  (a^3 - 6*a^2 - a + 3 = 0) → 
  (b^3 - 6*b^2 - b + 3 = 0) → 
  (c^3 - 6*c^2 - c + 3 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 37/9) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_inverse_squares_l4144_414414


namespace NUMINAMATH_CALUDE_parabola_translation_l4144_414404

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Translates a parabola horizontally -/
def translate_x (p : Parabola) (dx : ℝ) : Parabola :=
  { f := fun x => p.f (x - dx) }

/-- Translates a parabola vertically -/
def translate_y (p : Parabola) (dy : ℝ) : Parabola :=
  { f := fun x => p.f x + dy }

/-- The original parabola y = x^2 + 3 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 + 3 }

/-- The resulting parabola after translation -/
def resulting_parabola : Parabola :=
  { f := fun x => (x+3)^2 - 1 }

theorem parabola_translation :
  (translate_y (translate_x original_parabola 3) (-4)).f =
  resulting_parabola.f := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4144_414404


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l4144_414428

theorem diophantine_equation_solution (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l4144_414428


namespace NUMINAMATH_CALUDE_wonderful_class_size_l4144_414471

/-- Represents the number of students in Mrs. Wonderful's class -/
def class_size : ℕ := 18

/-- Represents the number of girls in the class -/
def girls : ℕ := class_size / 2 - 2

/-- Represents the number of boys in the class -/
def boys : ℕ := girls + 4

/-- The total number of jelly beans Mrs. Wonderful brought -/
def total_jelly_beans : ℕ := 420

/-- The number of jelly beans left after distribution -/
def remaining_jelly_beans : ℕ := 6

/-- Theorem stating that the given conditions result in 18 students -/
theorem wonderful_class_size : 
  (3 * girls * girls + 2 * boys * boys = total_jelly_beans - remaining_jelly_beans) ∧
  (boys = girls + 4) ∧
  (class_size = girls + boys) := by sorry

end NUMINAMATH_CALUDE_wonderful_class_size_l4144_414471


namespace NUMINAMATH_CALUDE_sum_sequence_equality_l4144_414433

theorem sum_sequence_equality (M : ℤ) : 
  1499 + 1497 + 1495 + 1493 + 1491 = 7500 - M → M = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_sequence_equality_l4144_414433


namespace NUMINAMATH_CALUDE_weekly_rental_cost_l4144_414403

/-- The weekly rental cost of a parking space, given the monthly cost,
    yearly savings, and number of months and weeks in a year. -/
theorem weekly_rental_cost (monthly_cost : ℕ) (yearly_savings : ℕ) 
                            (months_per_year : ℕ) (weeks_per_year : ℕ) :
  monthly_cost = 42 →
  yearly_savings = 16 →
  months_per_year = 12 →
  weeks_per_year = 52 →
  (monthly_cost * months_per_year + yearly_savings) / weeks_per_year = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_weekly_rental_cost_l4144_414403


namespace NUMINAMATH_CALUDE_eighty_one_power_ten_equals_three_power_q_l4144_414440

theorem eighty_one_power_ten_equals_three_power_q (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end NUMINAMATH_CALUDE_eighty_one_power_ten_equals_three_power_q_l4144_414440


namespace NUMINAMATH_CALUDE_jean_sale_savings_l4144_414474

/-- Represents the total savings during a jean sale -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) (fox_quantity pony_quantity : ℕ) : ℚ :=
  (fox_price * fox_quantity * fox_discount / 100) + (pony_price * pony_quantity * pony_discount / 100)

/-- Theorem stating the total savings during the jean sale -/
theorem jean_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 13.999999999999993
  let fox_discount : ℚ := 22 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 864 / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_jean_sale_savings_l4144_414474


namespace NUMINAMATH_CALUDE_macaroons_remaining_l4144_414424

def remaining_macaroons (initial_red : ℕ) (initial_green : ℕ) (eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  (initial_red - eaten_red) + (initial_green - eaten_green)

theorem macaroons_remaining :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroons_remaining_l4144_414424


namespace NUMINAMATH_CALUDE_min_ratio_cone_cylinder_volumes_l4144_414412

/-- The minimum ratio of the volume of a cone to the volume of a cylinder, 
    both circumscribed around the same sphere, is 4/3. -/
theorem min_ratio_cone_cylinder_volumes : ℝ := by
  -- Let r be the radius of the sphere
  -- Let h be the height of the cone
  -- Let a be the radius of the base of the cone
  -- The cylinder has height 2r and radius r
  -- The ratio of volumes is (π * a^2 * h / 3) / (π * r^2 * 2r)
  -- We need to prove that the minimum value of this ratio is 4/3
  sorry

end NUMINAMATH_CALUDE_min_ratio_cone_cylinder_volumes_l4144_414412


namespace NUMINAMATH_CALUDE_continued_fraction_sum_l4144_414488

theorem continued_fraction_sum (w x y : ℕ+) :
  (97 : ℚ) / 19 = w + 1 / (x + 1 / y) →
  (w : ℕ) + x + y = 16 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_sum_l4144_414488


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_six_l4144_414431

theorem sum_of_A_and_C_is_six (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 2 →
  A + C = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_six_l4144_414431


namespace NUMINAMATH_CALUDE_impossible_all_coeffs_roots_l4144_414496

/-- Given n > 1 monic quadratic polynomials and 2n distinct coefficients,
    prove that not all coefficients can be roots of the polynomials. -/
theorem impossible_all_coeffs_roots (n : ℕ) (a b : Fin n → ℝ) 
    (h_n : n > 1)
    (h_distinct : ∀ (i j : Fin n), i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)
    (h_poly : ∀ (i : Fin n), ∃ (x : ℝ), x^2 - a i * x + b i = 0) :
    ¬(∀ (i : Fin n), (∃ (j : Fin n), a i^2 - a j * a i + b j = 0) ∧
                     (∃ (k : Fin n), b i^2 - a k * b i + b k = 0)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_all_coeffs_roots_l4144_414496


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l4144_414400

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  sphere_diameter = small_cube_edge * Real.sqrt 3 →
  small_cube_edge^3 = 192 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l4144_414400


namespace NUMINAMATH_CALUDE_zoo_visit_cost_l4144_414444

/-- Calculates the total cost of a zoo visit for a group with a discount applied -/
theorem zoo_visit_cost 
  (num_children num_adults num_seniors : ℕ)
  (child_price adult_price senior_price : ℚ)
  (discount_rate : ℚ)
  (h_children : num_children = 6)
  (h_adults : num_adults = 10)
  (h_seniors : num_seniors = 4)
  (h_child_price : child_price = 12)
  (h_adult_price : adult_price = 20)
  (h_senior_price : senior_price = 15)
  (h_discount : discount_rate = 0.15) :
  (num_children : ℚ) * child_price + 
  (num_adults : ℚ) * adult_price + 
  (num_seniors : ℚ) * senior_price - 
  ((num_children : ℚ) * child_price + 
   (num_adults : ℚ) * adult_price + 
   (num_seniors : ℚ) * senior_price) * discount_rate = 282.20 := by
sorry

end NUMINAMATH_CALUDE_zoo_visit_cost_l4144_414444


namespace NUMINAMATH_CALUDE_find_S_l4144_414449

theorem find_S : ∃ S : ℚ, (1/3 : ℚ) * (1/8 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l4144_414449


namespace NUMINAMATH_CALUDE_factor_divisor_statements_l4144_414435

theorem factor_divisor_statements : 
  (∃ n : ℤ, 24 = 4 * n) ∧ 
  (∃ n : ℤ, 209 = 19 * n) ∧ 
  ¬(∃ n : ℤ, 63 = 19 * n) ∧
  (∃ n : ℤ, 180 = 9 * n) := by
sorry

end NUMINAMATH_CALUDE_factor_divisor_statements_l4144_414435


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4144_414439

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 2 = 1 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 10 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 10 = 9 → n ≤ m :=
by
  sorry

#eval 59 % 2  -- Expected output: 1
#eval 59 % 3  -- Expected output: 2
#eval 59 % 4  -- Expected output: 3
#eval 59 % 10 -- Expected output: 9

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4144_414439


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l4144_414498

/-- The smallest positive integer n such that (x+1)^n - 1 is divisible by x^2 + 1 modulo 3 -/
def smallest_n : ℕ := 8

/-- The divisor polynomial -/
def divisor_poly (x : ℤ) : ℤ := x^2 + 1

/-- The dividend polynomial -/
def dividend_poly (x : ℤ) (n : ℕ) : ℤ := (x + 1)^n - 1

/-- Divisibility modulo 3 -/
def is_divisible_mod_3 (a b : ℤ → ℤ) : Prop :=
  ∃ (p q : ℤ → ℤ), ∀ x, a x = b x * p x + 3 * q x

theorem smallest_n_divisible :
  (∀ n < smallest_n, ¬ is_divisible_mod_3 (dividend_poly · n) divisor_poly) ∧
  is_divisible_mod_3 (dividend_poly · smallest_n) divisor_poly :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l4144_414498


namespace NUMINAMATH_CALUDE_min_dot_product_of_vectors_l4144_414482

/-- Given plane vectors AC and BD, prove the minimum value of AB · CD -/
theorem min_dot_product_of_vectors (A B C D : ℝ × ℝ) : 
  (C.1 - A.1 = 1 ∧ C.2 - A.2 = 2) →  -- AC = (1, 2)
  (D.1 - B.1 = -2 ∧ D.2 - B.2 = 2) →  -- BD = (-2, 2)
  ∃ (min : ℝ), min = -9/4 ∧ 
    ∀ (AB CD : ℝ × ℝ), 
      AB.1 = B.1 - A.1 ∧ AB.2 = B.2 - A.2 →
      CD.1 = D.1 - C.1 ∧ CD.2 = D.2 - C.2 →
      AB.1 * CD.1 + AB.2 * CD.2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_of_vectors_l4144_414482


namespace NUMINAMATH_CALUDE_rationalize_denominator_l4144_414409

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -12 ∧ B = 7 ∧ C = 9 ∧ D = 13 ∧ E = 5 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l4144_414409


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l4144_414458

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l4144_414458


namespace NUMINAMATH_CALUDE_largest_number_with_properties_l4144_414452

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that extracts a two-digit number from adjacent digits in a larger number -/
def twoDigitNumber (n : ℕ) (i : ℕ) : ℕ :=
  (n / 10^i % 100)

/-- A function that checks if all two-digit numbers formed by adjacent digits are prime -/
def allTwoDigitPrime (n : ℕ) : Prop :=
  ∀ i : ℕ, i < (Nat.digits 10 n).length - 1 → isPrime (twoDigitNumber n i)

/-- A function that checks if all two-digit prime numbers formed are distinct -/
def allTwoDigitPrimeDistinct (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → j < (Nat.digits 10 n).length - 1 → 
    twoDigitNumber n i ≠ twoDigitNumber n j

/-- The main theorem stating that 617371311979 is the largest number satisfying the conditions -/
theorem largest_number_with_properties :
  (∀ m : ℕ, m > 617371311979 → 
    ¬(allTwoDigitPrime m ∧ allTwoDigitPrimeDistinct m)) ∧
  (allTwoDigitPrime 617371311979 ∧ allTwoDigitPrimeDistinct 617371311979) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_properties_l4144_414452


namespace NUMINAMATH_CALUDE_magic_square_x_value_l4144_414437

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (s : MagicSquare) : ℕ := s.a + s.b + s.c

/-- Predicate for a valid magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  -- All rows have the same sum
  s.a + s.b + s.c = magicSum s ∧
  s.d + s.e + s.f = magicSum s ∧
  s.g + s.h + s.i = magicSum s ∧
  -- All columns have the same sum
  s.a + s.d + s.g = magicSum s ∧
  s.b + s.e + s.h = magicSum s ∧
  s.c + s.f + s.i = magicSum s ∧
  -- Both diagonals have the same sum
  s.a + s.e + s.i = magicSum s ∧
  s.c + s.e + s.g = magicSum s

theorem magic_square_x_value (s : MagicSquare) 
  (h1 : isMagicSquare s)
  (h2 : s.b = 19 ∧ s.e = 15 ∧ s.h = 11)  -- Second column condition
  (h3 : s.b = 19 ∧ s.c = 14)  -- First row condition
  (h4 : s.e = 15 ∧ s.i = 12)  -- Diagonal condition
  : s.g = 18 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l4144_414437


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4144_414456

theorem complex_equation_solution (z : ℂ) : (1 - 2*I)*z = Complex.abs (3 + 4*I) → z = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4144_414456


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_theorem_l4144_414479

theorem pencil_eraser_cost_theorem :
  ∃ (p e : ℕ), p > e ∧ p > 0 ∧ e > 0 ∧ 15 * p + 5 * e = 200 ∧ p + e = 18 := by
  sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_theorem_l4144_414479


namespace NUMINAMATH_CALUDE_dividend_divisor_properties_l4144_414494

theorem dividend_divisor_properties : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 957 ∧ divisor = 75 ∧
  (dividend / divisor = (divisor / 10 + divisor % 10)) ∧
  (dividend % divisor = 57) ∧
  ((dividend % divisor) * (dividend / divisor) + divisor = 759) := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_properties_l4144_414494


namespace NUMINAMATH_CALUDE_boxes_in_case_l4144_414445

/-- Given that George has 12 blocks in total, each box holds 6 blocks,
    and George has 2 boxes of blocks, prove that there are 2 boxes in a case. -/
theorem boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) (boxes_of_blocks : ℕ) : 
  total_blocks = 12 → blocks_per_box = 6 → boxes_of_blocks = 2 → 
  (total_blocks / blocks_per_box : ℕ) = boxes_of_blocks := by
  sorry

#check boxes_in_case

end NUMINAMATH_CALUDE_boxes_in_case_l4144_414445


namespace NUMINAMATH_CALUDE_movie_only_attendance_l4144_414438

/-- Represents the number of students attending different activities --/
structure ActivityAttendance where
  total : ℕ
  picnic : ℕ
  games : ℕ
  movie_and_picnic : ℕ
  movie_and_games : ℕ
  picnic_and_games : ℕ
  all_activities : ℕ

/-- The given conditions for the problem --/
def given_conditions : ActivityAttendance :=
  { total := 31
  , picnic := 20
  , games := 5
  , movie_and_picnic := 4
  , movie_and_games := 2
  , picnic_and_games := 0
  , all_activities := 2
  }

/-- Theorem stating that the number of students meeting for the movie only is 12 --/
theorem movie_only_attendance (conditions : ActivityAttendance) : 
  conditions.total - (conditions.picnic + conditions.games - conditions.movie_and_picnic - conditions.movie_and_games - conditions.picnic_and_games + conditions.all_activities) = 12 :=
by sorry

end NUMINAMATH_CALUDE_movie_only_attendance_l4144_414438


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l4144_414483

/-- The number of dinners sold on each day of the week --/
structure DinnerSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the dinner sales problem --/
def dinner_problem (sales : DinnerSales) : Prop :=
  sales.monday = 40 ∧
  sales.tuesday = sales.monday + 40 ∧
  sales.thursday = sales.wednesday + 3 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203

/-- The theorem stating the ratio of Wednesday's sales to Tuesday's sales --/
theorem wednesday_to_tuesday_ratio (sales : DinnerSales) 
  (h : dinner_problem sales) : 
  (sales.wednesday : ℚ) / sales.tuesday = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_wednesday_to_tuesday_ratio_l4144_414483


namespace NUMINAMATH_CALUDE_given_curve_is_circle_l4144_414425

-- Define a polar coordinate
def PolarCoordinate := ℝ × ℝ

-- Define a circle in terms of its radius
def Circle (radius : ℝ) := {p : PolarCoordinate | p.2 = radius}

-- Define the curve given by the equation r = 5
def GivenCurve := {p : PolarCoordinate | p.2 = 5}

-- Theorem statement
theorem given_curve_is_circle : GivenCurve = Circle 5 := by
  sorry

end NUMINAMATH_CALUDE_given_curve_is_circle_l4144_414425


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4144_414486

theorem sufficient_not_necessary (x a : ℝ) (h : x > 0) :
  (a = 4 → ∀ x > 0, x + a / x ≥ 4) ∧
  ¬(∀ x > 0, x + a / x ≥ 4 → a = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4144_414486


namespace NUMINAMATH_CALUDE_roy_julia_age_difference_l4144_414448

theorem roy_julia_age_difference :
  ∀ (R J K : ℕ) (x : ℕ),
    R = J + x →  -- Roy is x years older than Julia
    R = K + x / 2 →  -- Roy is half of x years older than Kelly
    R + 4 = 2 * (J + 4) →  -- In 4 years, Roy will be twice as old as Julia
    (R + 4) * (K + 4) = 108 →  -- In 4 years, Roy's age multiplied by Kelly's age would be 108
    x = 6 :=  -- The difference between Roy's and Julia's ages is 6 years
by sorry

end NUMINAMATH_CALUDE_roy_julia_age_difference_l4144_414448


namespace NUMINAMATH_CALUDE_vector_equality_implies_coordinates_l4144_414455

/-- Given four points A, B, C, D in a plane, where vector AB equals vector CD,
    prove that the coordinates of C and D satisfy specific values. -/
theorem vector_equality_implies_coordinates (A B C D : ℝ × ℝ) :
  A = (1, 2) →
  B = (5, 4) →
  C.2 = 3 →
  D.1 = -3 →
  B.1 - A.1 = D.1 - C.1 →
  B.2 - A.2 = D.2 - C.2 →
  C.1 = -7 ∧ D.2 = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_implies_coordinates_l4144_414455


namespace NUMINAMATH_CALUDE_constant_term_expansion_l4144_414416

theorem constant_term_expansion (x : ℝ) : 
  ∃ (c : ℝ), (x + 1/x + 2)^4 = c + (terms_with_x : ℝ) ∧ c = 70 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l4144_414416


namespace NUMINAMATH_CALUDE_impossibility_of_identical_remainders_l4144_414478

theorem impossibility_of_identical_remainders :
  ¬ ∃ (a : Fin 100 → ℕ) (r : ℕ),
    r ≠ 0 ∧
    ∀ i : Fin 100, a i % a (i.succ) = r :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_identical_remainders_l4144_414478


namespace NUMINAMATH_CALUDE_eddie_number_l4144_414406

theorem eddie_number (n : ℕ) (m : ℕ) (h1 : n ≥ 40) (h2 : n % 5 = 0) (h3 : n % m = 0) :
  (∀ k : ℕ, k ≥ 40 ∧ k % 5 = 0 ∧ ∃ j : ℕ, k % j = 0 → k ≥ n) →
  n = 40 ∧ m = 2 := by
sorry

end NUMINAMATH_CALUDE_eddie_number_l4144_414406


namespace NUMINAMATH_CALUDE_red_candies_count_l4144_414462

theorem red_candies_count (green blue : ℕ) (prob_blue : ℚ) (red : ℕ) : 
  green = 5 → blue = 3 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  red = 4 := by
  sorry

end NUMINAMATH_CALUDE_red_candies_count_l4144_414462


namespace NUMINAMATH_CALUDE_lab_expense_ratio_l4144_414408

/-- Given a laboratory budget and expenses, prove the ratio of test tube cost to flask cost -/
theorem lab_expense_ratio (total_budget flask_cost remaining : ℚ) : 
  total_budget = 325 →
  flask_cost = 150 →
  remaining = 25 →
  ∃ (test_tube_cost : ℚ),
    total_budget = flask_cost + test_tube_cost + (test_tube_cost / 2) + remaining →
    test_tube_cost / flask_cost = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_lab_expense_ratio_l4144_414408


namespace NUMINAMATH_CALUDE_sum_of_specific_values_is_zero_l4144_414493

-- Define an odd function f on ℝ
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property f(x+1) = -f(x-1)
def hasFunctionalProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f (x - 1)

-- Theorem statement
theorem sum_of_specific_values_is_zero
  (f : ℝ → ℝ)
  (h1 : isOddFunction f)
  (h2 : hasFunctionalProperty f) :
  f 0 + f 1 + f 2 + f 3 + f 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_specific_values_is_zero_l4144_414493


namespace NUMINAMATH_CALUDE_max_b_in_box_l4144_414457

theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c < b →
  b < a →
  b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l4144_414457


namespace NUMINAMATH_CALUDE_process_output_for_4_l4144_414454

/-- A function representing the process described in the flowchart --/
def process (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the process outputs 3 when given input 4 --/
theorem process_output_for_4 : process 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_process_output_for_4_l4144_414454


namespace NUMINAMATH_CALUDE_prob_at_least_three_speak_l4144_414499

/-- The probability of a single baby speaking -/
def p : ℚ := 1/5

/-- The number of babies in the cluster -/
def n : ℕ := 7

/-- The probability that exactly k out of n babies will speak -/
def prob_exactly (k : ℕ) : ℚ :=
  (n.choose k) * (1 - p)^(n - k) * p^k

/-- The probability that at least 3 out of 7 babies will speak -/
theorem prob_at_least_three_speak : 
  1 - (prob_exactly 0 + prob_exactly 1 + prob_exactly 2) = 45349/78125 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_three_speak_l4144_414499


namespace NUMINAMATH_CALUDE_blue_balls_in_box_l4144_414407

theorem blue_balls_in_box (purple_balls yellow_balls min_tries : ℕ) 
  (h1 : purple_balls = 7)
  (h2 : yellow_balls = 11)
  (h3 : min_tries = 19) :
  ∃! blue_balls : ℕ, 
    blue_balls > 0 ∧ 
    purple_balls + yellow_balls + blue_balls = min_tries :=
by
  sorry

end NUMINAMATH_CALUDE_blue_balls_in_box_l4144_414407


namespace NUMINAMATH_CALUDE_taxi_driver_probability_l4144_414436

-- Define the number of checkpoints
def num_checkpoints : ℕ := 6

-- Define the probability of encountering a red light at each checkpoint
def red_light_prob : ℚ := 1/3

-- Define the probability of passing exactly two checkpoints before encountering a red light
def pass_two_checkpoints_prob : ℚ := 4/27

-- State the theorem
theorem taxi_driver_probability :
  ∀ (n : ℕ) (p : ℚ),
  n = num_checkpoints →
  p = red_light_prob →
  pass_two_checkpoints_prob = (1 - p) * (1 - p) * p :=
by sorry

end NUMINAMATH_CALUDE_taxi_driver_probability_l4144_414436


namespace NUMINAMATH_CALUDE_cow_ratio_l4144_414460

theorem cow_ratio (total : ℕ) (females males : ℕ) : 
  total = 300 →
  females + males = total →
  females = 2 * (females / 2) →
  males = 2 * (males / 2) →
  females / 2 = males / 2 + 50 →
  females = 2 * males :=
by
  sorry

end NUMINAMATH_CALUDE_cow_ratio_l4144_414460


namespace NUMINAMATH_CALUDE_pyramid_solution_l4144_414443

/-- Represents the structure of the number pyramid --/
structure NumberPyramid where
  row2_1 : ℕ
  row2_2 : ℕ → ℕ
  row2_3 : ℕ → ℕ
  row3_1 : ℕ → ℕ
  row3_2 : ℕ → ℕ
  row4   : ℕ → ℕ

/-- The specific number pyramid instance from the problem --/
def problemPyramid : NumberPyramid := {
  row2_1 := 11
  row2_2 := λ x => 6 + x
  row2_3 := λ x => x + 7
  row3_1 := λ x => 11 + (6 + x)
  row3_2 := λ x => (6 + x) + (x + 7)
  row4   := λ x => (11 + (6 + x)) + ((6 + x) + (x + 7))
}

/-- The theorem stating that x = 10 in this specific number pyramid --/
theorem pyramid_solution :
  ∃ x : ℕ, problemPyramid.row4 x = 60 ∧ x = 10 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_solution_l4144_414443


namespace NUMINAMATH_CALUDE_tangent_line_problem_l4144_414477

theorem tangent_line_problem (a : ℝ) :
  (∃ l : Set (ℝ × ℝ),
    -- l is a line
    (∃ m k : ℝ, l = {(x, y) | y = m*x + k}) ∧
    -- l passes through (1,0)
    (1, 0) ∈ l ∧
    -- l is tangent to y = x^3
    (∃ x₀ y₀ : ℝ, (x₀, y₀) ∈ l ∧ y₀ = x₀^3 ∧ m = 3*x₀^2) ∧
    -- l is tangent to y = ax^2 + (15/4)x - 9
    (∃ x₁ y₁ : ℝ, (x₁, y₁) ∈ l ∧ y₁ = a*x₁^2 + (15/4)*x₁ - 9 ∧ m = 2*a*x₁ + 15/4)) →
  a = -25/64 ∨ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l4144_414477


namespace NUMINAMATH_CALUDE_headphones_to_case_ratio_l4144_414413

def phone_cost : ℚ := 1000
def contract_cost_per_month : ℚ := 200
def case_cost_percentage : ℚ := 20 / 100
def total_first_year_cost : ℚ := 3700

def case_cost : ℚ := phone_cost * case_cost_percentage
def contract_cost_year : ℚ := contract_cost_per_month * 12
def headphones_cost : ℚ := total_first_year_cost - (phone_cost + case_cost + contract_cost_year)

theorem headphones_to_case_ratio :
  headphones_cost / case_cost = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_headphones_to_case_ratio_l4144_414413


namespace NUMINAMATH_CALUDE_base_b_square_l4144_414423

theorem base_b_square (b : ℕ) : 
  (b + 5)^2 = 4*b^2 + 3*b + 6 → b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_b_square_l4144_414423


namespace NUMINAMATH_CALUDE_average_is_eight_l4144_414422

def number_of_bags : ℕ := 5

def brown_mms : List ℕ := [9, 12, 8, 8, 3]

def average_brown_mms : ℚ := (brown_mms.sum : ℚ) / number_of_bags

theorem average_is_eight : average_brown_mms = 8 := by sorry

end NUMINAMATH_CALUDE_average_is_eight_l4144_414422
