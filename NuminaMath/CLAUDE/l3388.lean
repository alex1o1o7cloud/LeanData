import Mathlib

namespace NUMINAMATH_CALUDE_total_books_is_42_l3388_338838

-- Define the initial number of books on each shelf
def initial_books_shelf1 : ℕ := 9
def initial_books_shelf2 : ℕ := 0
def initial_books_shelf3 : ℕ := initial_books_shelf1 + (initial_books_shelf1 * 3 / 10)
def initial_books_shelf4 : ℕ := initial_books_shelf3 / 2

-- Define the number of books added to each shelf
def added_books_shelf1 : ℕ := 10
def added_books_shelf4 : ℕ := 5

-- Define the total number of books after additions
def total_books : ℕ := 
  (initial_books_shelf1 + added_books_shelf1) +
  initial_books_shelf2 +
  initial_books_shelf3 +
  (initial_books_shelf4 + added_books_shelf4)

-- Theorem statement
theorem total_books_is_42 : total_books = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_books_is_42_l3388_338838


namespace NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l3388_338889

theorem sum_of_two_smallest_numbers : ∀ (a b c : ℕ), 
  a = 10 ∧ b = 11 ∧ c = 12 → 
  min a (min b c) + min (max a b) (min b c) = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_smallest_numbers_l3388_338889


namespace NUMINAMATH_CALUDE_peculiar_quadratic_minimum_l3388_338857

/-- A quadratic polynomial q(x) = x^2 + bx + c is peculiar if q(q(x)) = 0 has exactly four real roots, including a triple root. -/
def IsPeculiar (q : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, (∀ x, q x = x^2 + b*x + c) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, (∀ x, q (q x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
  (r₁ = r₂ ∧ r₂ = r₃ ∧ r₃ ≠ r₄))

theorem peculiar_quadratic_minimum :
  ∃! q : ℝ → ℝ, IsPeculiar q ∧
  (∀ p : ℝ → ℝ, IsPeculiar p → q 0 ≤ p 0) ∧
  (∀ x, q x = x^2 - 1/2) ∧
  q 0 = -1/2 := by sorry

end NUMINAMATH_CALUDE_peculiar_quadratic_minimum_l3388_338857


namespace NUMINAMATH_CALUDE_exponential_properties_l3388_338847

theorem exponential_properties (a : ℝ) (x y : ℝ) 
  (hx : a^x = 2) (hy : a^y = 3) : 
  a^(x + y) = 6 ∧ a^(2*x - 3*y) = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_exponential_properties_l3388_338847


namespace NUMINAMATH_CALUDE_jakes_friend_candy_and_euros_l3388_338861

/-- Proves the number of candies Jake's friend can purchase and the amount in Euros he will receive --/
theorem jakes_friend_candy_and_euros :
  let feeding_allowance : ℝ := 4
  let fraction_given : ℝ := 1/4
  let candy_price : ℝ := 0.2
  let discount : ℝ := 0.15
  let exchange_rate : ℝ := 0.85
  
  let money_given := feeding_allowance * fraction_given
  let discounted_price := candy_price * (1 - discount)
  let candies_purchasable := ⌊money_given / discounted_price⌋
  let euros_received := money_given * exchange_rate
  
  (candies_purchasable = 5) ∧ (euros_received = 0.85) :=
by
  sorry

#check jakes_friend_candy_and_euros

end NUMINAMATH_CALUDE_jakes_friend_candy_and_euros_l3388_338861


namespace NUMINAMATH_CALUDE_handshake_arrangement_count_l3388_338834

/-- Represents a handshaking arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j)

/-- The number of valid handshaking arrangements for 12 people -/
def M : ℕ := sorry

/-- Theorem stating that the number of handshaking arrangements is congruent to 340 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 340 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_handshake_arrangement_count_l3388_338834


namespace NUMINAMATH_CALUDE_probability_not_red_special_cube_l3388_338893

structure Cube where
  total_faces : ℕ
  green_faces : ℕ
  blue_faces : ℕ
  red_faces : ℕ

def probability_not_red (c : Cube) : ℚ :=
  (c.green_faces + c.blue_faces : ℚ) / c.total_faces

theorem probability_not_red_special_cube :
  let c : Cube := {
    total_faces := 6,
    green_faces := 3,
    blue_faces := 2,
    red_faces := 1
  }
  probability_not_red c = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_special_cube_l3388_338893


namespace NUMINAMATH_CALUDE_ps_length_is_eight_l3388_338872

/-- Triangle PQR with given side lengths and angle bisector PS -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- PS is the angle bisector of ∠PQR -/
  PS_is_angle_bisector : Bool

/-- The theorem stating that PS = 8 in the given triangle -/
theorem ps_length_is_eight (t : TrianglePQR) 
  (h1 : t.PQ = 8)
  (h2 : t.QR = 15)
  (h3 : t.PR = 17)
  (h4 : t.PS_is_angle_bisector = true) :
  ∃ PS : ℝ, PS = 8 ∧ PS > 0 := by
  sorry


end NUMINAMATH_CALUDE_ps_length_is_eight_l3388_338872


namespace NUMINAMATH_CALUDE_inequality_proof_l3388_338821

theorem inequality_proof (x y : ℝ) (n : ℕ+) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (x^n.val / (1 - x^2) + y^n.val / (1 - y^2)) ≥ ((x^n.val + y^n.val) / (1 - x*y)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3388_338821


namespace NUMINAMATH_CALUDE_sqrt_division_minus_abs_l3388_338868

theorem sqrt_division_minus_abs : Real.sqrt 63 / Real.sqrt 7 - |(-4)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_minus_abs_l3388_338868


namespace NUMINAMATH_CALUDE_median_mean_difference_l3388_338830

theorem median_mean_difference (x : ℤ) (a : ℤ) : 
  x > 0 → x + a > 0 → x + 4 > 0 → x + 7 > 0 → x + 37 > 0 →
  (x + (x + a) + (x + 4) + (x + 7) + (x + 37)) / 5 = (x + 4) + 6 →
  a = 2 := by sorry

end NUMINAMATH_CALUDE_median_mean_difference_l3388_338830


namespace NUMINAMATH_CALUDE_shopping_tax_calculation_l3388_338820

/-- Calculates the final tax percentage given spending percentages and tax rates --/
def final_tax_percentage (clothing_percent : ℝ) (food_percent : ℝ) (electronics_percent : ℝ) 
  (other_percent : ℝ) (clothing_tax : ℝ) (food_tax : ℝ) (electronics_tax : ℝ) 
  (other_tax : ℝ) (loyalty_discount : ℝ) : ℝ :=
  let total_tax := clothing_percent * clothing_tax + food_percent * food_tax + 
                   electronics_percent * electronics_tax + other_percent * other_tax
  let discounted_tax := total_tax * (1 - loyalty_discount)
  discounted_tax * 100

theorem shopping_tax_calculation :
  final_tax_percentage 0.4 0.25 0.2 0.15 0.05 0.02 0.1 0.08 0.03 = 5.529 := by
  sorry

end NUMINAMATH_CALUDE_shopping_tax_calculation_l3388_338820


namespace NUMINAMATH_CALUDE_parallelogram_area_12_48_l3388_338856

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters -/
theorem parallelogram_area_12_48 : parallelogram_area 12 48 = 576 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_12_48_l3388_338856


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3388_338810

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, -2; -3, 6]
  Matrix.det A = 36 := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l3388_338810


namespace NUMINAMATH_CALUDE_fruit_basket_count_l3388_338814

/-- The number of ways to choose items from a set of n items -/
def choiceCount (n : ℕ) : ℕ := n + 1

/-- The total number of fruit baskets including the empty basket -/
def totalBaskets (appleCount orangeCount : ℕ) : ℕ :=
  choiceCount appleCount * choiceCount orangeCount

/-- The number of valid fruit baskets (excluding the empty basket) -/
def validBaskets (appleCount orangeCount : ℕ) : ℕ :=
  totalBaskets appleCount orangeCount - 1

theorem fruit_basket_count :
  validBaskets 7 12 = 103 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l3388_338814


namespace NUMINAMATH_CALUDE_subtraction_puzzle_l3388_338809

theorem subtraction_puzzle (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9)
  (h4 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) % 10 = 2)
  (h5 : b = c - 1)
  (h6 : (100 * a + 10 * b + c) - (100 * c + 10 * b + a) / 100 = 8) :
  a = 0 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_puzzle_l3388_338809


namespace NUMINAMATH_CALUDE_hoseok_minyoung_problem_l3388_338806

/-- Given a line of students, calculate the number of students between two specified positions. -/
def students_between (total : ℕ) (right_pos : ℕ) (left_pos : ℕ) : ℕ :=
  left_pos - (total - right_pos + 1) - 1

/-- Theorem: In a line of 13 students, with one student 9th from the right and another 8th from the left, 
    there are 2 students between them. -/
theorem hoseok_minyoung_problem :
  students_between 13 9 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_minyoung_problem_l3388_338806


namespace NUMINAMATH_CALUDE_fraction_equality_l3388_338840

theorem fraction_equality : (25 + 15) / (5 - 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3388_338840


namespace NUMINAMATH_CALUDE_blue_marbles_probability_l3388_338897

theorem blue_marbles_probability (red green blue white : ℕ) : 
  red = 3 → green = 4 → blue = 8 → white = 5 →
  (blue * (blue - 1)) / ((red + green + blue + white) * (red + green + blue + white - 1)) = 14 / 95 := by
sorry

end NUMINAMATH_CALUDE_blue_marbles_probability_l3388_338897


namespace NUMINAMATH_CALUDE_circle_radius_l3388_338884

/-- The radius of a circle described by the equation x² + y² + 12 = 10x - 6y is √22. -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 + 12 = 10*x - 6*y) → ∃ (center_x center_y : ℝ), 
    ∀ (point_x point_y : ℝ), 
      (point_x - center_x)^2 + (point_y - center_y)^2 = 22 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_l3388_338884


namespace NUMINAMATH_CALUDE_simplify_expression_l3388_338860

theorem simplify_expression (m : ℝ) (h : m < 1) :
  (m - 1) * Real.sqrt (-1 / (m - 1)) = -Real.sqrt (1 - m) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3388_338860


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3388_338852

-- Define the polyhedron P
structure Polyhedron :=
  (vertices : Finset (Fin 9))
  (edges : Finset (Fin 9 × Fin 9))
  (base : Finset (Fin 7))
  (apex1 : Fin 9)
  (apex2 : Fin 9)

-- Define the coloring of edges
def Coloring (P : Polyhedron) := (Fin 9 × Fin 9) → Bool

-- Define a valid coloring
def ValidColoring (P : Polyhedron) (c : Coloring P) : Prop :=
  ∀ e ∈ P.edges, c e = true ∨ c e = false

-- Define a monochromatic triangle
def MonochromaticTriangle (P : Polyhedron) (c : Coloring P) : Prop :=
  ∃ (v1 v2 v3 : Fin 9), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    (v1, v2) ∈ P.edges ∧ (v2, v3) ∈ P.edges ∧ (v1, v3) ∈ P.edges ∧
    c (v1, v2) = c (v2, v3) ∧ c (v2, v3) = c (v1, v3)

-- The main theorem
theorem monochromatic_triangle_exists (P : Polyhedron) (c : Coloring P)
    (h_valid : ValidColoring P c) :
    MonochromaticTriangle P c := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3388_338852


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l3388_338886

/-- Calculates the total loan amount given the down payment, monthly payment, and loan duration in years. -/
def totalLoanAmount (downPayment : ℕ) (monthlyPayment : ℕ) (years : ℕ) : ℕ :=
  downPayment + monthlyPayment * (years * 12)

/-- Theorem stating that a loan with a $10,000 down payment and $600 monthly payments for 5 years totals $46,000. -/
theorem loan_amount_calculation :
  totalLoanAmount 10000 600 5 = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l3388_338886


namespace NUMINAMATH_CALUDE_inverse_as_linear_combination_l3388_338843

def M : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 0, 4]

theorem inverse_as_linear_combination :
  ∃ (a b : ℚ), M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) ∧ 
  a = -1/12 ∧ b = 7/12 := by
sorry

end NUMINAMATH_CALUDE_inverse_as_linear_combination_l3388_338843


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3388_338867

theorem simplify_and_evaluate : 
  let x : ℚ := 1/2
  5 * x^2 - (x^2 - 2*(2*x - 3)) = -3 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3388_338867


namespace NUMINAMATH_CALUDE_single_digit_addition_l3388_338896

theorem single_digit_addition (A : ℕ) : 
  A < 10 → -- A is a single digit number
  10 * A + A + 10 * A + A = 132 → -- AA + AA = 132
  A = 6 := by sorry

end NUMINAMATH_CALUDE_single_digit_addition_l3388_338896


namespace NUMINAMATH_CALUDE_floor_neg_seven_fourths_l3388_338888

theorem floor_neg_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_neg_seven_fourths_l3388_338888


namespace NUMINAMATH_CALUDE_problem_statement_l3388_338858

theorem problem_statement (x y : ℝ) (hx : x = Real.sqrt 2 + 1) (hy : y = Real.sqrt 2 - 1) :
  (x + y) * (x - y) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3388_338858


namespace NUMINAMATH_CALUDE_special_function_increasing_l3388_338819

/-- A function satisfying the given properties -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  pos_gt_one : ∀ x > 0, f x > 1
  multiplicative : ∀ x y, f (x + y) = f x * f y

/-- Theorem: f is increasing on ℝ -/
theorem special_function_increasing (f : ℝ → ℝ) [SpecialFunction f] :
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_special_function_increasing_l3388_338819


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3388_338890

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the y-axis reflection function
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the problem statement
theorem sum_of_coordinates_after_reflection :
  let C : Point := (3, 8)
  let D : Point := reflect_y C
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3388_338890


namespace NUMINAMATH_CALUDE_set_operations_l3388_338862

def U : Set ℕ := {x | x ≤ 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 2, 4, 6}

theorem set_operations :
  (A ∩ B = {2, 4}) ∧
  (U \ (A ∪ B) = {0, 3, 7}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3388_338862


namespace NUMINAMATH_CALUDE_fraction_irreducible_l3388_338822

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l3388_338822


namespace NUMINAMATH_CALUDE_min_all_correct_questions_l3388_338816

/-- Represents the number of questions all students answered correctly -/
def all_correct (total : ℕ) (correct : List ℕ) : ℕ :=
  total - (correct.map (λ x => total - x)).sum

/-- The main theorem -/
theorem min_all_correct_questions :
  let total := 15
  let correct := [11, 12, 13, 14]
  all_correct total correct = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_all_correct_questions_l3388_338816


namespace NUMINAMATH_CALUDE_max_distance_complex_l3388_338849

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs ((1 + 2*Complex.I)*z^3 - z^6) ≤ Real.sqrt 5 + 1 ∧
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs ((1 + 2*Complex.I)*z₀^3 - z₀^6) = Real.sqrt 5 + 1 :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l3388_338849


namespace NUMINAMATH_CALUDE_acid_concentration_theorem_l3388_338877

def acid_concentration_problem (acid1 acid2 acid3 : ℝ) 
  (conc1 conc2 : ℝ) : Prop :=
  let water1 := acid1 / conc1 - acid1
  let water2 := acid2 / conc2 - acid2
  let total_water := water1 + water2
  let conc3 := acid3 / (acid3 + total_water)
  acid1 = 10 ∧ 
  acid2 = 20 ∧ 
  acid3 = 30 ∧ 
  conc1 = 0.05 ∧ 
  conc2 = 70/300 ∧ 
  conc3 = 0.105

theorem acid_concentration_theorem : 
  acid_concentration_problem 10 20 30 0.05 (70/300) :=
sorry

end NUMINAMATH_CALUDE_acid_concentration_theorem_l3388_338877


namespace NUMINAMATH_CALUDE_work_completion_time_l3388_338881

theorem work_completion_time (b_time : ℝ) (joint_work_time : ℝ) (work_completed : ℝ) (a_time : ℝ) : 
  b_time = 20 →
  joint_work_time = 2 →
  work_completed = 0.2333333333333334 →
  joint_work_time * ((1 / a_time) + (1 / b_time)) = work_completed →
  a_time = 15 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3388_338881


namespace NUMINAMATH_CALUDE_rectangular_plot_ratio_l3388_338835

/-- A rectangular plot with given area and breadth -/
structure RectangularPlot where
  area : ℝ
  breadth : ℝ
  length_multiple : ℕ
  area_eq : area = breadth * (breadth * length_multiple)

/-- The ratio of length to breadth for a rectangular plot -/
def length_breadth_ratio (plot : RectangularPlot) : ℚ :=
  plot.length_multiple

theorem rectangular_plot_ratio (plot : RectangularPlot) 
  (h1 : plot.area = 432)
  (h2 : plot.breadth = 12) :
  length_breadth_ratio plot = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_ratio_l3388_338835


namespace NUMINAMATH_CALUDE_expression_simplification_l3388_338863

theorem expression_simplification (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3388_338863


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3388_338887

/-- The curve function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x - 4

theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let M : ℝ × ℝ := (1, 1)
  let m : ℝ := f' M.1  -- Slope of the tangent line at M
  let line (x y : ℝ) := 2 * x - y + 4 = 0
  (∀ x y, line x y ↔ y - P.2 = m * (x - P.1)) ∧  -- Point-slope form
  (f M.1 = M.2) ∧  -- M is on the curve
  (f' M.1 = m)  -- Slope at M equals the derivative
  := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3388_338887


namespace NUMINAMATH_CALUDE_S_maximized_l3388_338801

/-- The sum of the first n terms of an arithmetic sequence with general term a_n = -2n + 25 -/
def S (n : ℕ) : ℝ := 24 * n - n^2

/-- The value of n that maximizes S(n) -/
def n_max : ℕ := 11

/-- Theorem stating that S(n) is maximized when n = n_max -/
theorem S_maximized : ∀ k : ℕ, S k ≤ S n_max := by sorry

end NUMINAMATH_CALUDE_S_maximized_l3388_338801


namespace NUMINAMATH_CALUDE_functional_sequence_a10_l3388_338811

/-- A sequence satisfying a functional equation -/
def FunctionalSequence (a : ℕ+ → ℤ) : Prop :=
  ∀ p q : ℕ+, a (p + q) = a p + a q

theorem functional_sequence_a10 (a : ℕ+ → ℤ) 
  (h1 : FunctionalSequence a) (h2 : a 2 = -6) : 
  a 10 = -30 := by sorry

end NUMINAMATH_CALUDE_functional_sequence_a10_l3388_338811


namespace NUMINAMATH_CALUDE_similarity_condition_l3388_338829

theorem similarity_condition (a b : ℝ) :
  (∃ h : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, h x = y) ∧ 
    (∀ x₁ x₂ : ℝ, h x₁ = h x₂ → x₁ = x₂) ∧
    (∀ x : ℝ, h (x^2 + a*x + b) = (h x)^2)) →
  b = a*(a + 2)/4 := by
sorry

end NUMINAMATH_CALUDE_similarity_condition_l3388_338829


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3388_338826

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > y) (h2 : y > 0) 
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) : x + y ≥ 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3388_338826


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3388_338865

theorem solve_complex_equation (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (a - i)^2 = 2*i) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3388_338865


namespace NUMINAMATH_CALUDE_absolute_difference_sum_product_l3388_338841

theorem absolute_difference_sum_product (x y : ℝ) (hx : x = 12) (hy : y = 18) :
  |x - y| * (x + y) = 180 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_sum_product_l3388_338841


namespace NUMINAMATH_CALUDE_total_skips_eq_2450_l3388_338880

/-- Represents the number of skips completed by a person given their skipping rate and duration. -/
def skips_completed (rate : ℚ) (duration : ℚ) : ℚ := rate * duration

/-- Calculates the total number of skips completed by Roberto, Valerie, and Lucas. -/
def total_skips : ℚ :=
  let roberto_rate : ℚ := 4200 / 60  -- skips per minute
  let valerie_rate : ℚ := 80         -- skips per minute
  let lucas_rate : ℚ := 150 / 5      -- skips per minute
  let roberto_duration : ℚ := 15     -- minutes
  let valerie_duration : ℚ := 10     -- minutes
  let lucas_duration : ℚ := 20       -- minutes
  skips_completed roberto_rate roberto_duration +
  skips_completed valerie_rate valerie_duration +
  skips_completed lucas_rate lucas_duration

theorem total_skips_eq_2450 : total_skips = 2450 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_eq_2450_l3388_338880


namespace NUMINAMATH_CALUDE_square_sum_pattern_l3388_338812

theorem square_sum_pattern : 
  (1^2 + 3^2 = 10) → (2^2 + 4^2 = 20) → (3^2 + 5^2 = 34) → (4^2 + 6^2 = 52) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_pattern_l3388_338812


namespace NUMINAMATH_CALUDE_ice_cream_cost_l3388_338864

theorem ice_cream_cost (pierre_scoops mom_scoops : ℕ) (total_bill : ℚ) 
  (h1 : pierre_scoops = 3)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (scoop_cost : ℚ), scoop_cost * (pierre_scoops + mom_scoops : ℚ) = total_bill ∧ scoop_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l3388_338864


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3388_338870

theorem division_remainder_problem : ∃ (x : ℕ+), 
  19250 % x.val = 11 ∧ 
  20302 % x.val = 3 ∧ 
  x.val = 53 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3388_338870


namespace NUMINAMATH_CALUDE_system_is_linear_l3388_338894

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and x and y are variables. -/
def IsLinearEquation (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, eq x y ↔ a * x + b * y = c

/-- A system of two linear equations is a pair of linear equations in two variables. -/
def IsSystemOfTwoLinearEquations (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  IsLinearEquation eq1 ∧ IsLinearEquation eq2

/-- The given system of equations. -/
def System : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) :=
  (fun x y ↦ x + y = 2, fun _ y ↦ y = 3)

theorem system_is_linear : IsSystemOfTwoLinearEquations System.1 System.2 := by
  sorry

#check system_is_linear

end NUMINAMATH_CALUDE_system_is_linear_l3388_338894


namespace NUMINAMATH_CALUDE_part_one_part_two_l3388_338800

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  t.a * (1 + Real.cos t.C / 2) = (Real.sqrt 3 * t.c * Real.sin t.A) / 2

-- Part 1: Prove that C = 2π/3
theorem part_one (t : Triangle) (h : condition t) : t.C = 2 * Real.pi / 3 := by
  sorry

-- Part 2: Prove that maximum area occurs when a = √2
theorem part_two (t : Triangle) (h1 : condition t) (h2 : t.c = Real.sqrt 6) :
  ∃ (max_area : ℝ), ∀ (t' : Triangle), condition t' → t'.c = Real.sqrt 6 →
    (1/2 * t'.a * t'.b * Real.sin t'.C) ≤ max_area ∧
    (1/2 * t'.a * t'.b * Real.sin t'.C = max_area → t'.a = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3388_338800


namespace NUMINAMATH_CALUDE_quadratic_substitution_roots_l3388_338844

/-- Given a quadratic equation ax^2 + bx + c = 0, this theorem proves the conditions for equal
    product of roots after substitution and the sum of all roots in those cases. -/
theorem quadratic_substitution_roots (a b c : ℝ) (h : a ≠ 0) :
  ∃ k : ℝ, 
    (k = 0 ∨ k = -b/a) ∧ 
    (∀ y : ℝ, c/a = (a*k^2 + b*k + c)/a) ∧
    ((k = 0 → ((-b/a) + (-b/a) = -2*b/a)) ∧ 
     (k = -b/a → ((-b/a) + (b/a) = 0))) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_substitution_roots_l3388_338844


namespace NUMINAMATH_CALUDE_square_circles_l3388_338874

/-- A square in a plane -/
structure Square where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to check if a circle's diameter has endpoints as vertices of the square -/
def is_valid_circle (s : Square) (c : Circle) : Prop :=
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ s.vertices ∧ v2 ∈ s.vertices ∧
    v1 ≠ v2 ∧
    c.center = ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2) ∧
    c.radius = Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) / 2

/-- The main theorem -/
theorem square_circles (s : Square) :
  ∃! (circles : Finset Circle), circles.card = 2 ∧
    ∀ c ∈ circles, is_valid_circle s c ∧
    ∀ c, is_valid_circle s c → c ∈ circles :=
  sorry


end NUMINAMATH_CALUDE_square_circles_l3388_338874


namespace NUMINAMATH_CALUDE_adjacent_probability_l3388_338846

/-- The number of people -/
def total_people : ℕ := 9

/-- The number of rows -/
def num_rows : ℕ := 3

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 3

/-- The probability of two specific people sitting next to each other in the same row -/
def probability_adjacent : ℚ := 2 / 9

theorem adjacent_probability :
  probability_adjacent = (2 : ℚ) / (total_people : ℚ) := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_l3388_338846


namespace NUMINAMATH_CALUDE_product_value_l3388_338813

theorem product_value (x : ℝ) (h : Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) :
  (6 + x) * (21 - x) = 1369 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_value_l3388_338813


namespace NUMINAMATH_CALUDE_total_lost_words_l3388_338851

/-- Represents the number of letters in the language --/
def total_letters : ℕ := 100

/-- Represents the number of forbidden letters --/
def forbidden_letters : ℕ := 6

/-- Calculates the number of lost one-letter words --/
def lost_one_letter_words : ℕ := forbidden_letters

/-- Calculates the number of lost two-letter words with forbidden first letter --/
def lost_two_letter_first : ℕ := forbidden_letters * total_letters

/-- Calculates the number of lost two-letter words with forbidden second letter --/
def lost_two_letter_second : ℕ := total_letters * forbidden_letters

/-- Calculates the number of lost two-letter words with both letters forbidden --/
def lost_two_letter_both : ℕ := forbidden_letters * forbidden_letters

/-- Calculates the total number of lost two-letter words --/
def lost_two_letter_words : ℕ := lost_two_letter_first + lost_two_letter_second - lost_two_letter_both

/-- Theorem stating the total number of lost words --/
theorem total_lost_words :
  lost_one_letter_words + lost_two_letter_words = 1170 := by sorry

end NUMINAMATH_CALUDE_total_lost_words_l3388_338851


namespace NUMINAMATH_CALUDE_a_investment_l3388_338871

/-- Calculates the investment of partner A in a business partnership --/
def calculate_investment_A (investment_B investment_C total_profit profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_B + investment_C + profit_share_A * (investment_B + investment_C) / (total_profit - profit_share_A)
  profit_share_A * total_investment / total_profit

/-- Theorem stating that A's investment is 6300 given the problem conditions --/
theorem a_investment (investment_B investment_C total_profit profit_share_A : ℚ)
  (hB : investment_B = 4200)
  (hC : investment_C = 10500)
  (hProfit : total_profit = 12100)
  (hShareA : profit_share_A = 3630) :
  calculate_investment_A investment_B investment_C total_profit profit_share_A = 6300 := by
  sorry

#eval calculate_investment_A 4200 10500 12100 3630

end NUMINAMATH_CALUDE_a_investment_l3388_338871


namespace NUMINAMATH_CALUDE_range_of_a_l3388_338804

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (Real.log x - x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ -Real.exp 1) →
  (∃ x > 0, f a x = -Real.exp 1) →
  a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3388_338804


namespace NUMINAMATH_CALUDE_wallet_value_l3388_338832

def total_bills : ℕ := 12
def five_dollar_bills : ℕ := 4
def five_dollar_value : ℕ := 5
def ten_dollar_value : ℕ := 10

theorem wallet_value :
  (five_dollar_bills * five_dollar_value) +
  ((total_bills - five_dollar_bills) * ten_dollar_value) = 100 :=
by sorry

end NUMINAMATH_CALUDE_wallet_value_l3388_338832


namespace NUMINAMATH_CALUDE_intersection_sum_l3388_338833

theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 4 + a) ∧ (4 = (1/3) * 2 + b) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l3388_338833


namespace NUMINAMATH_CALUDE_irrational_ratio_transformation_l3388_338842

theorem irrational_ratio_transformation : ∃ x y : ℝ, 
  (Irrational x) ∧ (Irrational y) ∧ (x ≠ y) ∧ ((7 + x) / (11 + y) = 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_irrational_ratio_transformation_l3388_338842


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3388_338807

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3388_338807


namespace NUMINAMATH_CALUDE_product_mod_twelve_l3388_338825

theorem product_mod_twelve : (95 * 97) % 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twelve_l3388_338825


namespace NUMINAMATH_CALUDE_f_neg_one_gt_f_two_l3388_338803

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Condition 1: y = f(x+1) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

-- Condition 2: f(x) is an increasing function on the interval [1, +∞)
def is_increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

-- Theorem statement
theorem f_neg_one_gt_f_two 
  (h1 : is_even_shifted f) 
  (h2 : is_increasing_on_interval f) : 
  f (-1) > f 2 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_one_gt_f_two_l3388_338803


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l3388_338818

theorem not_divisible_by_nine (n : ℕ) : ¬ (9 ∣ (n^3 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l3388_338818


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l3388_338815

theorem max_value_of_sum_products (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 200) :
  ab + bc + cd ≤ 10000 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l3388_338815


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l3388_338848

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, y^2 = x^3 + (x + 1)^2 ↔ (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l3388_338848


namespace NUMINAMATH_CALUDE_bc_length_l3388_338805

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isObtuseTriangle (t : Triangle) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem bc_length (ABC : Triangle) 
  (h1 : isObtuseTriangle ABC)
  (h2 : triangleArea ABC = 10 * Real.sqrt 3)
  (h3 : distance ABC.A ABC.B = 5)
  (h4 : distance ABC.A ABC.C = 8) :
  distance ABC.B ABC.C = Real.sqrt 129 := by sorry

end NUMINAMATH_CALUDE_bc_length_l3388_338805


namespace NUMINAMATH_CALUDE_cookies_baked_l3388_338875

/-- Given 5 pans of cookies with 8 cookies per pan, prove that the total number of cookies is 40. -/
theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 5) (h2 : cookies_per_pan = 8) :
  pans * cookies_per_pan = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookies_baked_l3388_338875


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l3388_338895

/-- Given an arithmetic sequence with 7 terms, first term 10, and common difference 12,
    prove that the average of all terms is 46. -/
theorem arithmetic_sequence_average : 
  let n : ℕ := 7
  let a : ℕ := 10
  let d : ℕ := 12
  let sequence := (fun i => a + d * (i - 1))
  let sum := (sequence 1 + sequence n) * n / 2
  (sum : ℚ) / n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l3388_338895


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_sides_sum_l3388_338876

/-- A parallelogram with vertices A, B, C, and D. -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : (A.1 - B.1, A.2 - B.2) = (D.1 - C.1, D.2 - C.2) ∧ 
                      (A.1 - D.1, A.2 - D.2) = (B.1 - C.1, B.2 - C.2))

/-- The squared distance between two points in ℝ² -/
def dist_squared (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The sum of the squares of the diagonals of a parallelogram 
    is equal to the sum of the squares of its four sides -/
theorem parallelogram_diagonals_sides_sum (P : Parallelogram) : 
  dist_squared P.A P.C + dist_squared P.B P.D = 
  dist_squared P.A P.B + dist_squared P.B P.C + 
  dist_squared P.C P.D + dist_squared P.D P.A :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_sides_sum_l3388_338876


namespace NUMINAMATH_CALUDE_triangle_sine_theorem_l3388_338859

theorem triangle_sine_theorem (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ) :
  area = 36 →
  side = 12 →
  median = 10 →
  area = 1/2 * side * median * Real.sin θ →
  0 < θ →
  θ < π/2 →
  Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_theorem_l3388_338859


namespace NUMINAMATH_CALUDE_incorrect_proportion_l3388_338837

theorem incorrect_proportion (a b m n : ℝ) (h : a * b = m * n) :
  ¬(m / a = n / b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_proportion_l3388_338837


namespace NUMINAMATH_CALUDE_parabola_h_value_l3388_338831

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Checks if a point (x, y) lies on the parabola -/
def Parabola.contains (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

theorem parabola_h_value (p : Parabola) :
  p.a < 0 →
  0 < p.h →
  p.h < 6 →
  p.contains 0 4 →
  p.contains 6 5 →
  p.h = 4 := by
  sorry

#check parabola_h_value

end NUMINAMATH_CALUDE_parabola_h_value_l3388_338831


namespace NUMINAMATH_CALUDE_largest_prime_factor_f9_div_f3_l3388_338869

def f (n : ℕ) : ℕ := (3^n + 1) / 2

theorem largest_prime_factor_f9_div_f3 :
  let ratio : ℕ := f 9 / f 3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ ratio ∧ p = 37 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ ratio → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_f9_div_f3_l3388_338869


namespace NUMINAMATH_CALUDE_lemniscate_polar_to_rect_l3388_338850

/-- The lemniscate equation in polar coordinates -/
def lemniscate_polar (r φ a : ℝ) : Prop :=
  r^2 = 2 * a^2 * Real.cos (2 * φ)

/-- The lemniscate equation in rectangular coordinates -/
def lemniscate_rect (x y a : ℝ) : Prop :=
  (x^2 + y^2)^2 = 2 * a^2 * (x^2 - y^2)

/-- Theorem stating that the rectangular equation represents the lemniscate -/
theorem lemniscate_polar_to_rect (a : ℝ) :
  ∀ (x y r φ : ℝ), 
    x = r * Real.cos φ →
    y = r * Real.sin φ →
    lemniscate_polar r φ a →
    lemniscate_rect x y a :=
by
  sorry

end NUMINAMATH_CALUDE_lemniscate_polar_to_rect_l3388_338850


namespace NUMINAMATH_CALUDE_class_size_l3388_338824

/-- The number of students in a class with French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french - both) + (german - both) + both + neither

/-- Theorem: The total number of students in the class is 87 -/
theorem class_size : total_students 41 22 9 33 = 87 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3388_338824


namespace NUMINAMATH_CALUDE_escalator_travel_time_l3388_338845

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_travel_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 250) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_travel_time_l3388_338845


namespace NUMINAMATH_CALUDE_kyle_practice_time_l3388_338853

/-- Kyle's daily basketball practice schedule -/
def KylePractice : Prop :=
  ∃ (total_time shooting_time running_time weightlifting_time : ℕ),
    -- Total practice time
    total_time = shooting_time + running_time + weightlifting_time
    -- Half time spent shooting
    ∧ 2 * shooting_time = total_time
    -- Running time is twice weightlifting time
    ∧ running_time = 2 * weightlifting_time
    -- Weightlifting time is 20 minutes
    ∧ weightlifting_time = 20
    -- Total time in hours is 2
    ∧ total_time = 120

/-- Theorem: Kyle's daily basketball practice is 2 hours -/
theorem kyle_practice_time : KylePractice := by
  sorry

end NUMINAMATH_CALUDE_kyle_practice_time_l3388_338853


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l3388_338899

theorem sum_of_squares_representation : 
  (((17 ^ 2 + 19 ^ 2) / 2) ^ 2 : ℕ) = 260 ^ 2 + 195 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l3388_338899


namespace NUMINAMATH_CALUDE_sum_of_specific_terms_l3388_338866

/-- The sum of the 7th to 10th terms of a sequence defined by S_n = 2n^2 - 3n + 1 is 116 -/
theorem sum_of_specific_terms (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, S n = 2 * n^2 - 3 * n + 1) →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  a 7 + a 8 + a 9 + a 10 = 116 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_terms_l3388_338866


namespace NUMINAMATH_CALUDE_regular_pentagons_are_similar_l3388_338879

/-- A regular pentagon is a polygon with 5 sides of equal length and 5 equal angles -/
structure RegularPentagon where
  side_length : ℝ
  angle_measure : ℝ
  side_length_pos : side_length > 0
  angle_measure_pos : angle_measure > 0
  angle_sum : angle_measure * 5 = 540

/-- Two shapes are similar if they have the same shape but not necessarily the same size -/
def AreSimilar (p1 p2 : RegularPentagon) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ p2.side_length = k * p1.side_length

/-- Theorem: Any two regular pentagons are similar -/
theorem regular_pentagons_are_similar (p1 p2 : RegularPentagon) : AreSimilar p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagons_are_similar_l3388_338879


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3388_338891

/-- Proposition P: For any real number x, ax^2 + ax + 1 > 0 always holds -/
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0

/-- Proposition Q: The equation x^2 - x + a = 0 has real roots -/
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ := {a : ℝ | a < 0 ∨ (0 < a ∧ a < 4)}

theorem range_of_a_theorem :
  ∀ a : ℝ, (¬(P a ∧ Q a) ∧ (P a ∨ Q a)) ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3388_338891


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3388_338808

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 6 * q^4 * Real.sqrt (6 * q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3388_338808


namespace NUMINAMATH_CALUDE_discount_calculation_l3388_338823

/-- Calculates the discounted price of an item given the original price and discount rate. -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proves that a 20% discount on a $120 item results in a price of $96. -/
theorem discount_calculation :
  let original_price : ℝ := 120
  let discount_rate : ℝ := 0.2
  discounted_price original_price discount_rate = 96 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3388_338823


namespace NUMINAMATH_CALUDE_average_cost_before_gratuity_l3388_338878

/-- Proves that for a group of 7 people with a total bill of $840 including 20% gratuity,
    the average cost per person before gratuity is $100. -/
theorem average_cost_before_gratuity 
  (num_people : ℕ) 
  (total_bill : ℝ) 
  (gratuity_rate : ℝ) :
  num_people = 7 →
  total_bill = 840 →
  gratuity_rate = 0.20 →
  (total_bill / (1 + gratuity_rate)) / num_people = 100 := by
sorry

end NUMINAMATH_CALUDE_average_cost_before_gratuity_l3388_338878


namespace NUMINAMATH_CALUDE_books_together_l3388_338882

/-- The number of books Tim and Mike have together -/
def total_books (tim_books mike_books : ℕ) : ℕ := tim_books + mike_books

/-- Theorem stating that Tim and Mike have 42 books together -/
theorem books_together : total_books 22 20 = 42 := by
  sorry

end NUMINAMATH_CALUDE_books_together_l3388_338882


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l3388_338802

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 2}

-- Define set B (domain of the function)
def B : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem 1
theorem union_A_B_when_a_is_one :
  A 1 ∪ B = {x | -1 < x ∧ x < 3} := by sorry

-- Theorem 2
theorem intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three (a : ℝ) :
  A a ∩ B = ∅ ↔ a ≤ -3 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_intersection_A_B_empty_iff_a_leq_neg_three_or_geq_three_l3388_338802


namespace NUMINAMATH_CALUDE_relationship_of_values_l3388_338839

/-- An odd function f defined on ℝ satisfying f(x) + xf'(x) < 0 for x < 0 -/
class OddDecreasingFunction (f : ℝ → ℝ) : Prop where
  odd : ∀ x, f (-x) = -f x
  decreasing : ∀ x < 0, f x + x * (deriv f x) < 0

/-- The main theorem stating the relationship between πf(π), (-2)f(-2), and f(1) -/
theorem relationship_of_values (f : ℝ → ℝ) [OddDecreasingFunction f] :
  π * f π > (-2) * f (-2) ∧ (-2) * f (-2) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_of_values_l3388_338839


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_inequality_condition_l3388_338828

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

-- Part 1
theorem min_value_when_a_is_one :
  ∃ m : ℝ, (∀ x : ℝ, f x 1 ≥ m) ∧ (∃ x : ℝ, f x 1 = m) ∧ m = 2 :=
sorry

-- Part 2
theorem inequality_condition :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x a > x^2 - b + 1) →
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_inequality_condition_l3388_338828


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3388_338817

theorem smallest_whole_number_above_sum : ℕ := by
  let sum := 3 + 1/3 + 4 + 1/4 + 5 + 1/6 + 6 + 1/7
  have h1 : sum < 19 := by sorry
  have h2 : sum > 18 := by sorry
  exact 19

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l3388_338817


namespace NUMINAMATH_CALUDE_fraction_power_product_one_l3388_338898

theorem fraction_power_product_one : (9 / 8 : ℚ) ^ 4 * (8 / 9 : ℚ) ^ 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_one_l3388_338898


namespace NUMINAMATH_CALUDE_art_to_maths_ratio_is_one_to_one_l3388_338855

/-- Represents the school supplies problem --/
structure SchoolSupplies where
  total_budget : ℕ
  maths_books : ℕ
  maths_book_price : ℕ
  science_books_diff : ℕ
  science_book_price : ℕ
  music_books_cost : ℕ

/-- The ratio of art books to maths books is 1:1 --/
def art_to_maths_ratio (s : SchoolSupplies) : Prop :=
  let total_spent := s.maths_books * s.maths_book_price + 
                     (s.maths_books + s.science_books_diff) * s.science_book_price + 
                     s.maths_books * s.maths_book_price + 
                     s.music_books_cost
  total_spent ≤ s.total_budget ∧ 
  (s.maths_books : ℚ) / s.maths_books = 1

/-- The main theorem stating that the ratio of art books to maths books is 1:1 --/
theorem art_to_maths_ratio_is_one_to_one (s : SchoolSupplies) 
  (h : s = { total_budget := 500,
             maths_books := 4,
             maths_book_price := 20,
             science_books_diff := 6,
             science_book_price := 10,
             music_books_cost := 160 }) : 
  art_to_maths_ratio s := by
  sorry


end NUMINAMATH_CALUDE_art_to_maths_ratio_is_one_to_one_l3388_338855


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l3388_338827

/-- Proves that adding 70 ounces of a 60% salt solution to 70 ounces of a 20% salt solution
    results in a mixture that is 40% salt. -/
theorem salt_mixture_proof :
  let initial_amount : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_amount : ℝ := 70
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let total_amount : ℝ := initial_amount + added_amount
  let total_salt : ℝ := initial_amount * initial_concentration + added_amount * added_concentration
  total_salt / total_amount = final_concentration := by
  sorry

#check salt_mixture_proof

end NUMINAMATH_CALUDE_salt_mixture_proof_l3388_338827


namespace NUMINAMATH_CALUDE_age_difference_l3388_338854

theorem age_difference (a b c : ℕ) : 
  b = 18 →
  b = 2 * c →
  a + b + c = 47 →
  a = b + 2 :=
by sorry

end NUMINAMATH_CALUDE_age_difference_l3388_338854


namespace NUMINAMATH_CALUDE_binomial_multiply_three_l3388_338873

theorem binomial_multiply_three : 3 * Nat.choose 9 5 = 378 := by sorry

end NUMINAMATH_CALUDE_binomial_multiply_three_l3388_338873


namespace NUMINAMATH_CALUDE_exists_arrangement_for_23_l3388_338883

/-- Recursive definition of the sequence F_i -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence satisfying the required property -/
theorem exists_arrangement_for_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ F 1 = 1 ∧ 
  (∀ n : ℕ, n ≥ 2 → F n = 3 * F (n - 1) - F (n - 2)) ∧
  F 12 % 23 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_for_23_l3388_338883


namespace NUMINAMATH_CALUDE_triangle_problem_l3388_338836

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  t.c / (t.b - t.a) + Real.sin t.B / (Real.sin t.C + Real.sin t.A) = -1

/-- The law of sines for a triangle -/
def law_of_sines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B ∧ t.b / Real.sin t.B = t.c / Real.sin t.C

/-- The cosine rule for a triangle -/
def cosine_rule (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2 * t.b * t.c * Real.cos t.A

theorem triangle_problem (t : Triangle) 
  (h1 : given_condition t) 
  (h2 : law_of_sines t) 
  (h3 : cosine_rule t)
  (h4 : t.a = 4 * Real.sqrt 13)
  (h5 : t.c = 12) :
  t.A = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.b * t.c * Real.sin t.A = 12 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l3388_338836


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l3388_338885

theorem road_trip_time_calculation (dist_wa_id : ℝ) (dist_id_nv : ℝ) (speed_wa_id : ℝ) (speed_id_nv : ℝ)
  (h1 : dist_wa_id = 640)
  (h2 : dist_id_nv = 550)
  (h3 : speed_wa_id = 80)
  (h4 : speed_id_nv = 50)
  (h5 : speed_wa_id > 0)
  (h6 : speed_id_nv > 0) :
  dist_wa_id / speed_wa_id + dist_id_nv / speed_id_nv = 19 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_time_calculation_l3388_338885


namespace NUMINAMATH_CALUDE_set_equality_implies_p_equals_three_l3388_338892

theorem set_equality_implies_p_equals_three (p : ℝ) : 
  let U : Set ℝ := {x | x^2 - 3*x + 2 = 0}
  let A : Set ℝ := {x | x^2 - p*x + 2 = 0}
  (U \ A = ∅) → p = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_p_equals_three_l3388_338892
