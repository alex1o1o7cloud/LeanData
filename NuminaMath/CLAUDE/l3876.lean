import Mathlib

namespace NUMINAMATH_CALUDE_prism_volume_l3876_387602

/-- The volume of a right prism with an equilateral triangular base -/
theorem prism_volume (a : ℝ) (h : ℝ) : 
  a = 5 → -- Side length of the equilateral triangle base
  (a * h * 2 + a^2 * Real.sqrt 3 / 4) = 40 → -- Sum of areas of three adjacent faces
  a * a * Real.sqrt 3 / 4 * h = 625 / 160 * (3 - Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_l3876_387602


namespace NUMINAMATH_CALUDE_triangle_properties_l3876_387683

/-- Given a triangle ABC with specific properties, prove its angle A and area. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) : 
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C →
  -- Sum of angles is π
  A + B + C = π →
  -- Side lengths satisfy triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  Real.sin A + Real.sqrt 3 * Real.cos A = 2 →
  a = 2 →
  B = π / 4 →
  -- Prove angle A and area
  A = π / 6 ∧ 
  (1/2 : ℝ) * a * b * Real.sin C = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3876_387683


namespace NUMINAMATH_CALUDE_constant_function_proof_l3876_387618

theorem constant_function_proof (f : ℝ → ℝ) 
  (h_continuous : Continuous f) 
  (h_condition : ∀ (x : ℝ) (t : ℝ), t ≥ 0 → f x = f (Real.exp t * x)) : 
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l3876_387618


namespace NUMINAMATH_CALUDE_concentric_circles_radius_change_l3876_387614

theorem concentric_circles_radius_change (R_o R_i : ℝ) 
  (h1 : R_o = 6)
  (h2 : R_i = 4)
  (h3 : R_o > R_i)
  (h4 : 0 < R_i)
  (h5 : 0 < R_o) :
  let A_original := π * (R_o^2 - R_i^2)
  let R_i_new := R_i * 0.75
  let A_new := A_original * 3.6
  ∃ x : ℝ, 
    (π * ((R_o * (1 + x/100))^2 - R_i_new^2) = A_new) ∧
    x = 50 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radius_change_l3876_387614


namespace NUMINAMATH_CALUDE_equation_solutions_count_l3876_387678

theorem equation_solutions_count :
  let f : ℝ → ℝ := λ θ => (Real.sin θ ^ 2 - 1) * (2 * Real.sin θ ^ 2 - 1)
  ∃! (s : Finset ℝ), s.card = 6 ∧ 
    (∀ θ ∈ s, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0) ∧
    (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ f θ = 0 → θ ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l3876_387678


namespace NUMINAMATH_CALUDE_investment_calculation_l3876_387663

/-- Given two investors P and Q, where the profit is divided in the ratio 4:6 and P invested 60000, 
    prove that Q invested 90000. -/
theorem investment_calculation (P Q : ℕ) (profit_ratio : Rat) (P_investment : ℕ) : 
  profit_ratio = 4 / 6 →
  P_investment = 60000 →
  Q = 90000 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l3876_387663


namespace NUMINAMATH_CALUDE_two_digit_sum_theorem_l3876_387617

def is_valid_set (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (10 * a + b) + (10 * a + c) + (10 * b + a) + (10 * b + c) + (10 * c + a) + (10 * c + b) = 484

def valid_sets : List (Fin 10 × Fin 10 × Fin 10) :=
  [(9, 4, 9), (9, 5, 8), (9, 6, 7), (8, 6, 8), (8, 7, 7)]

theorem two_digit_sum_theorem (a b c : ℕ) :
  is_valid_set a b c →
  (a, b, c) ∈ valid_sets.map (fun (x, y, z) => (x.val, y.val, z.val)) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_theorem_l3876_387617


namespace NUMINAMATH_CALUDE_jerry_added_six_figures_l3876_387627

/-- Given that Jerry initially had 4 action figures and ended up with 10 action figures in total,
    prove that he added 6 action figures. -/
theorem jerry_added_six_figures (initial : ℕ) (total : ℕ) (added : ℕ)
    (h1 : initial = 4)
    (h2 : total = 10)
    (h3 : total = initial + added) :
  added = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerry_added_six_figures_l3876_387627


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3876_387611

theorem polynomial_evaluation :
  ∀ y : ℝ, y > 0 → y^2 - 3*y - 9 = 0 → y^3 - 3*y^2 - 9*y + 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3876_387611


namespace NUMINAMATH_CALUDE_condition_one_condition_two_l3876_387640

-- Define set A
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Define set B
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- Theorem for condition 1
theorem condition_one (a : ℝ) : A a ∩ B = A a → a < -3 ∨ a > 3 := by
  sorry

-- Theorem for condition 2
theorem condition_two (a : ℝ) : (A a ∩ B).Nonempty → a < -1 ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_condition_one_condition_two_l3876_387640


namespace NUMINAMATH_CALUDE_rectilinear_polygon_odd_area_l3876_387696

/-- A rectilinear polygon with integer vertex coordinates and odd side lengths -/
structure RectilinearPolygon where
  vertices : List (Int × Int)
  sides_parallel_to_axes : Bool
  all_sides_odd_length : Bool

/-- The area of a rectilinear polygon -/
noncomputable def area (p : RectilinearPolygon) : ℝ :=
  sorry

/-- A predicate to check if a number is odd -/
def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem rectilinear_polygon_odd_area
  (p : RectilinearPolygon)
  (h_sides : p.vertices.length = 100)
  (h_parallel : p.sides_parallel_to_axes = true)
  (h_odd_sides : p.all_sides_odd_length = true) :
  is_odd (Int.floor (area p)) :=
sorry

end NUMINAMATH_CALUDE_rectilinear_polygon_odd_area_l3876_387696


namespace NUMINAMATH_CALUDE_complement_of_union_l3876_387639

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3}

theorem complement_of_union (U A B : Set Nat) 
  (hU : U = {0, 1, 2, 3, 4})
  (hA : A = {0, 1, 3})
  (hB : B = {2, 3}) :
  (U \ (A ∪ B)) = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l3876_387639


namespace NUMINAMATH_CALUDE_angle_properties_l3876_387660

/-- Given that the terminal side of angle α passes through point P(5a, -12a) where a < 0,
    prove that tan α = -12/5 and sin α + cos α = 7/13 -/
theorem angle_properties (a : ℝ) (α : ℝ) (h : a < 0) :
  let x := 5 * a
  let y := -12 * a
  let r := Real.sqrt (x^2 + y^2)
  (Real.tan α = -12/5) ∧ (Real.sin α + Real.cos α = 7/13) := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l3876_387660


namespace NUMINAMATH_CALUDE_boot_price_calculation_l3876_387667

theorem boot_price_calculation (discount_percent : ℝ) (discounted_price : ℝ) : 
  discount_percent = 20 → discounted_price = 72 → 
  discounted_price / (1 - discount_percent / 100) = 90 := by
sorry

end NUMINAMATH_CALUDE_boot_price_calculation_l3876_387667


namespace NUMINAMATH_CALUDE_problem_solution_l3876_387651

/-- Given that 2x^5 - x^3 + 4x^2 + 3x - 5 + g(x) = 7x^3 - 4x + 2,
    prove that g(x) = -2x^5 + 6x^3 - 4x^2 - x + 7 -/
theorem problem_solution (x : ℝ) :
  let g : ℝ → ℝ := λ x => -2*x^5 + 6*x^3 - 4*x^2 - x + 7
  2*x^5 - x^3 + 4*x^2 + 3*x - 5 + g x = 7*x^3 - 4*x + 2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3876_387651


namespace NUMINAMATH_CALUDE_min_value_of_a_l3876_387686

theorem min_value_of_a (x a : ℝ) : 
  (∃ x, |x - 1| + |x + a| ≤ 8) → a ≥ -9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3876_387686


namespace NUMINAMATH_CALUDE_inverse_composition_l3876_387600

-- Define the functions h and k
noncomputable def h : ℝ → ℝ := sorry
noncomputable def k : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_composition (x : ℝ) : 
  (h⁻¹ ∘ k) x = 3 * x - 4 → k⁻¹ (h 8) = 8 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l3876_387600


namespace NUMINAMATH_CALUDE_line_always_intersects_ellipse_iff_m_in_range_l3876_387668

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the ellipse equation
def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

-- Theorem statement
theorem line_always_intersects_ellipse_iff_m_in_range :
  ∀ m : ℝ, (∀ k : ℝ, ∃ x y : ℝ, line k x = y ∧ ellipse m x y) ↔ 
  (m ≥ 1 ∧ m ≠ 5) :=
sorry

end NUMINAMATH_CALUDE_line_always_intersects_ellipse_iff_m_in_range_l3876_387668


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3876_387699

theorem partial_fraction_decomposition :
  ∀ (x : ℝ) (P Q R : ℚ),
    P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 →
    x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
    (x^2 - 9) / ((x - 1)*(x - 4)*(x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3876_387699


namespace NUMINAMATH_CALUDE_marbles_theorem_l3876_387619

def marbles_problem (total : ℕ) (colors : ℕ) (red_lost : ℕ) : ℕ :=
  let marbles_per_color := total / colors
  let red_remaining := marbles_per_color - red_lost
  let blue_remaining := marbles_per_color - (2 * red_lost)
  let yellow_remaining := marbles_per_color - (3 * red_lost)
  red_remaining + blue_remaining + yellow_remaining

theorem marbles_theorem :
  marbles_problem 72 3 5 = 42 := by sorry

end NUMINAMATH_CALUDE_marbles_theorem_l3876_387619


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l3876_387605

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l3876_387605


namespace NUMINAMATH_CALUDE_lemonade_problem_l3876_387649

theorem lemonade_problem (lemons_for_60 : ℕ) (gallons : ℕ) (lemon_cost : ℚ) :
  lemons_for_60 = 36 →
  gallons = 15 →
  lemon_cost = 1/2 →
  (lemons_for_60 * gallons) / 60 = 9 ∧
  (lemons_for_60 * gallons) / 60 * lemon_cost = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_problem_l3876_387649


namespace NUMINAMATH_CALUDE_paper_sheets_calculation_l3876_387622

theorem paper_sheets_calculation (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  sheets_per_student = 5 →
  num_classes * students_per_class * sheets_per_student = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_sheets_calculation_l3876_387622


namespace NUMINAMATH_CALUDE_samples_left_over_proof_l3876_387661

/-- Calculates the number of samples left over given the number of samples per box,
    number of boxes opened, and number of customers who tried a sample. -/
def samples_left_over (samples_per_box : ℕ) (boxes_opened : ℕ) (customers : ℕ) : ℕ :=
  samples_per_box * boxes_opened - customers

/-- Proves that given 20 samples per box, 12 boxes opened, and 235 customers,
    the number of samples left over is 5. -/
theorem samples_left_over_proof :
  samples_left_over 20 12 235 = 5 := by
  sorry

end NUMINAMATH_CALUDE_samples_left_over_proof_l3876_387661


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3876_387635

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 3 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) * 3) / 2

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces in a triangle with n rows of rods -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l3876_387635


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3876_387654

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 4 + a 7 = 48) →
  (a 2 + a 5 + a 8 = 40) →
  (a 3 + a 6 + a 9 = 32) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3876_387654


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_l3876_387664

/-- The number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zero : ℕ := 531441

/-- The number of 6-digit numbers with at least one zero -/
def six_digit_numbers_with_zero : ℕ := total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_zero_count : six_digit_numbers_with_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_count_l3876_387664


namespace NUMINAMATH_CALUDE_derivative_of_y_l3876_387691

noncomputable def y (x : ℝ) : ℝ := 
  -1 / (3 * Real.sin x ^ 3) - 1 / Real.sin x + 1 / 2 * Real.log ((1 + Real.sin x) / (1 - Real.sin x))

theorem derivative_of_y (x : ℝ) (h : x ∉ Set.range (fun n => n * π)) :
  deriv y x = 1 / (Real.cos x * Real.sin x ^ 4) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_y_l3876_387691


namespace NUMINAMATH_CALUDE_no_real_solutions_l3876_387685

theorem no_real_solutions : ¬∃ (x : ℝ), x + 64 / (x + 3) = -13 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3876_387685


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l3876_387603

-- Define the triangular number function
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem for the 30th triangular number
theorem thirtieth_triangular_number : triangularNumber 30 = 465 := by
  sorry

-- Theorem for the sum of 30th and 29th triangular numbers
theorem sum_of_30th_and_29th_triangular_numbers :
  triangularNumber 30 + triangularNumber 29 = 900 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_sum_of_30th_and_29th_triangular_numbers_l3876_387603


namespace NUMINAMATH_CALUDE_certain_number_value_l3876_387628

-- Define the operation #
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem certain_number_value :
  ∀ x : ℝ, hash x 6 = 48 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l3876_387628


namespace NUMINAMATH_CALUDE_gasoline_tank_problem_l3876_387681

/-- Proves properties of a gasoline tank given initial and final fill levels -/
theorem gasoline_tank_problem (x : ℚ) 
  (h1 : 5/6 * x - 2/3 * x = 18) 
  (h2 : x > 0) : 
  x = 108 ∧ 18 * 4 = 72 := by
  sorry

#check gasoline_tank_problem

end NUMINAMATH_CALUDE_gasoline_tank_problem_l3876_387681


namespace NUMINAMATH_CALUDE_largest_number_l3876_387655

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -3 → b = 0 → c = Real.sqrt 5 → d = 2 → 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3876_387655


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3876_387647

theorem imaginary_part_of_complex_number : 
  Complex.im ((2 : ℂ) + Complex.I * Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l3876_387647


namespace NUMINAMATH_CALUDE_common_tangent_theorem_l3876_387698

/-- The value of 'a' for which the graphs of f(x) = ln(x) and g(x) = x^2 + ax 
    have a common tangent line parallel to y = x -/
def tangent_condition (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ 
    (1 / x₁ = 1) ∧ 
    (2 * x₂ + a = 1) ∧ 
    (x₂^2 + a * x₂ = x₂ - 1)

theorem common_tangent_theorem :
  ∀ a : ℝ, tangent_condition a → (a = 3 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_theorem_l3876_387698


namespace NUMINAMATH_CALUDE_count_b_k_divisible_by_11_l3876_387634

-- Define b_k as a function that takes k and returns the concatenated number
def b (k : ℕ) : ℕ := sorry

-- Define a function to count how many b_k are divisible by 11 for 1 ≤ k ≤ 50
def count_divisible_by_11 : ℕ := sorry

-- Theorem stating that the count of b_k divisible by 11 for 1 ≤ k ≤ 50 is equal to X
theorem count_b_k_divisible_by_11 : count_divisible_by_11 = X := by sorry

end NUMINAMATH_CALUDE_count_b_k_divisible_by_11_l3876_387634


namespace NUMINAMATH_CALUDE_line_intercepts_l3876_387674

/-- Given a line with equation 4x + 6y = 24, prove its x-intercept and y-intercept -/
theorem line_intercepts (x y : ℝ) :
  4 * x + 6 * y = 24 →
  (x = 6 ∧ y = 0) ∨ (x = 0 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_l3876_387674


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l3876_387662

theorem chocolate_bar_cost (total_bars : ℕ) (unsold_bars : ℕ) (total_amount : ℚ) :
  total_bars = 8 →
  unsold_bars = 3 →
  total_amount = 20 →
  (total_bars - unsold_bars : ℚ) * (total_amount / (total_bars - unsold_bars : ℚ)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l3876_387662


namespace NUMINAMATH_CALUDE_greatest_third_side_l3876_387672

theorem greatest_third_side (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (c : ℕ), c = 16 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
sorry

end NUMINAMATH_CALUDE_greatest_third_side_l3876_387672


namespace NUMINAMATH_CALUDE_f_properties_l3876_387692

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then x^2 + 4*x + 3 else Real.log (x - 1) + 1

-- Theorem statement
theorem f_properties :
  (f (Real.exp 1 + 1) = 2) ∧
  (Set.range f = Set.Ici (-1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3876_387692


namespace NUMINAMATH_CALUDE_sunglasses_sign_cost_l3876_387601

theorem sunglasses_sign_cost (selling_price cost_price : ℕ) (pairs_sold : ℕ) : 
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  (pairs_sold * (selling_price - cost_price)) / 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_sign_cost_l3876_387601


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3876_387673

/-- A quadratic function with specific properties -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≤ f 3) ∧
    (f 3 = 10) ∧
    (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4)

/-- Theorem stating that f(5) = 0 for the specified quadratic function -/
theorem quadratic_function_property (f : ℝ → ℝ) (h : quadratic_function f) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3876_387673


namespace NUMINAMATH_CALUDE_area_of_four_squares_l3876_387666

/-- The area of a shape composed of four identical squares with side length 3 cm is 36 cm² -/
theorem area_of_four_squares (side_length : ℝ) (h1 : side_length = 3) : 
  4 * (side_length ^ 2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_squares_l3876_387666


namespace NUMINAMATH_CALUDE_fisher_algebra_eligibility_l3876_387669

/-- Determines if a student is eligible for algebra based on their quarterly scores -/
def isEligible (q1 q2 q3 q4 : ℚ) : Prop :=
  (q1 + q2 + q3 + q4) / 4 ≥ 83

/-- Fisher's minimum required score for the 4th quarter -/
def fisherMinScore : ℚ := 98

theorem fisher_algebra_eligibility :
  ∀ q4 : ℚ,
  isEligible 82 77 75 q4 ↔ q4 ≥ fisherMinScore :=
by sorry

#check fisher_algebra_eligibility

end NUMINAMATH_CALUDE_fisher_algebra_eligibility_l3876_387669


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l3876_387638

theorem inverse_proposition_false : 
  ¬(∀ (a b c : ℝ), a > b → a / (c^2) > b / (c^2)) := by
sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l3876_387638


namespace NUMINAMATH_CALUDE_a_in_A_sufficient_not_necessary_for_a_in_B_l3876_387626

def A : Set ℝ := {1, 2, 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem a_in_A_sufficient_not_necessary_for_a_in_B :
  (∀ a, a ∈ A → a ∈ B) ∧ (∃ a, a ∈ B ∧ a ∉ A) := by sorry

end NUMINAMATH_CALUDE_a_in_A_sufficient_not_necessary_for_a_in_B_l3876_387626


namespace NUMINAMATH_CALUDE_inequality_proof_l3876_387608

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_abc : a + b + c = 1) :
  (7 + 2*b) / (1 + a) + (7 + 2*c) / (1 + b) + (7 + 2*a) / (1 + c) ≥ 69/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3876_387608


namespace NUMINAMATH_CALUDE_constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l3876_387630

/-- Represents an algebraic term --/
inductive Term
  | Constant (n : ℕ)
  | Variable (name : String)
  | Product (terms : List Term)

/-- Defines when two terms are like terms --/
def areLikeTerms (t1 t2 : Term) : Prop :=
  match t1, t2 with
  | Term.Constant _, Term.Constant _ => True
  | Term.Variable x, Term.Variable y => x = y
  | Term.Product l1, Term.Product l2 => l1 = l2
  | _, _ => False

/-- Theorem stating that constants are like terms --/
theorem constants_are_like_terms (a b : ℕ) :
  areLikeTerms (Term.Constant a) (Term.Constant b) := by sorry

/-- Theorem stating that terms with different variables are not like terms --/
theorem different_variables_not_like_terms (x y : String) (h : x ≠ y) :
  ¬ areLikeTerms (Term.Variable x) (Term.Variable y) := by sorry

/-- Theorem stating that terms with different exponents are not like terms --/
theorem different_exponents_not_like_terms (x : String) (a b : ℕ) (h : a ≠ b) :
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Constant a]) 
    (Term.Product [Term.Variable x, Term.Constant b]) := by sorry

/-- Main theorem combining the results for the given problem --/
theorem like_terms_classification 
  (a b : ℕ) 
  (x y z : String) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : x ≠ z) :
  areLikeTerms (Term.Constant a) (Term.Constant b) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable y, Term.Variable x]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable y, Term.Variable z]) ∧
  ¬ areLikeTerms 
    (Term.Product [Term.Variable x, Term.Variable y])
    (Term.Product [Term.Variable x, Term.Variable y, Term.Variable z]) := by sorry

end NUMINAMATH_CALUDE_constants_are_like_terms_different_variables_not_like_terms_different_exponents_not_like_terms_like_terms_classification_l3876_387630


namespace NUMINAMATH_CALUDE_non_zero_terms_count_l3876_387646

/-- The expression to be expanded and simplified -/
def expression (x : ℝ) : ℝ := (x - 3) * (x^2 + 5*x + 8) + 2 * (x^3 + 3*x^2 - x - 4)

/-- The expanded and simplified form of the expression -/
def simplified_expression (x : ℝ) : ℝ := 3*x^3 + 8*x^2 - 9*x - 32

/-- Theorem stating that the number of non-zero terms in the simplified expression is 4 -/
theorem non_zero_terms_count : 
  (∃ (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0), 
    ∀ x, simplified_expression x = a*x^3 + b*x^2 + c*x + d) ∧
  (∀ (a b c d e : ℝ), ¬(∀ x, simplified_expression x = a*x^4 + b*x^3 + c*x^2 + d*x + e)) :=
sorry

end NUMINAMATH_CALUDE_non_zero_terms_count_l3876_387646


namespace NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_six_fifths_l3876_387697

theorem tan_theta_two_implies_expression_equals_six_fifths (θ : Real) 
  (h : Real.tan θ = 2) : 
  (Real.sin θ * (1 + Real.sin (2 * θ))) / 
  (Real.sqrt 2 * Real.cos (θ - π / 4)) = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_two_implies_expression_equals_six_fifths_l3876_387697


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387658

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 120 → b = 160 → c^2 = a^2 + b^2 → c = 200 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3876_387658


namespace NUMINAMATH_CALUDE_first_number_equation_l3876_387643

theorem first_number_equation : ∃ x : ℝ, 
  x + 17.0005 - 9.1103 = 20.011399999999995 ∧ 
  x = 12.121199999999995 := by sorry

end NUMINAMATH_CALUDE_first_number_equation_l3876_387643


namespace NUMINAMATH_CALUDE_num_subsets_eq_two_pow_l3876_387615

/-- The number of subsets of a finite set -/
def num_subsets (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of subsets of a set with n elements is 2^n -/
theorem num_subsets_eq_two_pow (n : ℕ) : num_subsets n = 2^n := by
  sorry

end NUMINAMATH_CALUDE_num_subsets_eq_two_pow_l3876_387615


namespace NUMINAMATH_CALUDE_independence_test_not_always_correct_l3876_387620

-- Define what an independence test is
def IndependenceTest : Type := sorry

-- Define a function that represents the conclusion of an independence test
def conclusion (test : IndependenceTest) : Prop := sorry

-- Theorem stating that the conclusion of an independence test is not always correct
theorem independence_test_not_always_correct :
  ¬ (∀ (test : IndependenceTest), conclusion test) := by sorry

end NUMINAMATH_CALUDE_independence_test_not_always_correct_l3876_387620


namespace NUMINAMATH_CALUDE_alloy_composition_l3876_387641

/-- Proves that the amount of the first alloy used is 15 kg given the specified conditions -/
theorem alloy_composition (x : ℝ) : 
  (0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) → x = 15 :=
by sorry

end NUMINAMATH_CALUDE_alloy_composition_l3876_387641


namespace NUMINAMATH_CALUDE_point_comparison_l3876_387693

/-- Given points in a 2D coordinate system, prove that a > c -/
theorem point_comparison (a b c d e f : ℝ) : 
  b > 0 →  -- (a, b) is above x-axis
  d > 0 →  -- (c, d) is above x-axis
  f < 0 →  -- (e, f) is below x-axis
  a > 0 →  -- (a, b) is to the right of y-axis
  c > 0 →  -- (c, d) is to the right of y-axis
  e < 0 →  -- (e, f) is to the left of y-axis
  a > c →  -- (a, b) is horizontally farther from y-axis than (c, d)
  b > d →  -- (a, b) is vertically farther from x-axis than (c, d)
  a > c :=
by sorry

end NUMINAMATH_CALUDE_point_comparison_l3876_387693


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3876_387624

theorem root_equation_implies_expression_value (x₀ : ℝ) (h : x₀ > 0) :
  x₀^3 * Real.exp (x₀ - 4) + 2 * Real.log x₀ - 4 = 0 →
  Real.exp ((4 - x₀) / 2) + 2 * Real.log x₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3876_387624


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3876_387607

/-- A line passing through the point (-3, -1) and parallel to x - 3y - 1 = 0 has the equation x - 3y = 0 -/
theorem parallel_line_through_point : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ x - 3*y = 0) →
    (-3, -1) ∈ l →
    (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = t ∧ y = (t - 1) / 3) →
    True :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3876_387607


namespace NUMINAMATH_CALUDE_remainder_987654_div_6_l3876_387687

theorem remainder_987654_div_6 : 987654 % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987654_div_6_l3876_387687


namespace NUMINAMATH_CALUDE_intersection_points_l3876_387659

-- Define the polar equations
def line_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 4
def curve_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the constraints
def valid_polar_coord (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_points :
  ∀ ρ θ, valid_polar_coord ρ θ →
    (line_equation ρ θ ∧ curve_equation ρ θ) →
    ((ρ = 4 ∧ θ = 0) ∨ (ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l3876_387659


namespace NUMINAMATH_CALUDE_min_value_interval_min_value_interval_converse_l3876_387690

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

/-- The theorem stating the possible values of a -/
theorem min_value_interval (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) →
  a = 2 ∨ a = -10 := by
  sorry

/-- The converse theorem -/
theorem min_value_interval_converse :
  ∀ a : ℝ, (a = 2 ∨ a = -10) →
  (∀ x ∈ Set.Icc a (a + 6), f x ≥ 9) ∧
  (∃ x ∈ Set.Icc a (a + 6), f x = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_interval_min_value_interval_converse_l3876_387690


namespace NUMINAMATH_CALUDE_robin_total_bottles_l3876_387677

/-- The total number of water bottles Robin drank throughout the day -/
def total_bottles (morning afternoon evening night : ℕ) : ℕ :=
  morning + afternoon + evening + night

/-- Theorem stating that Robin drank 24 bottles in total -/
theorem robin_total_bottles : 
  total_bottles 7 9 5 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_robin_total_bottles_l3876_387677


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_165_l3876_387676

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Property of combination numbers -/
axiom comb_property (n m : ℕ) : binomial n (m-1) + binomial n m = binomial (n+1) m

/-- Special case of combination numbers -/
axiom comb_special_case : binomial 2 2 = binomial 3 3

/-- The sum of squares of binomial coefficients from C(2,2) to C(10,2) -/
def sum_of_squares : ℕ := 
  binomial 2 2 + binomial 3 2 + binomial 4 2 + binomial 5 2 + 
  binomial 6 2 + binomial 7 2 + binomial 8 2 + binomial 9 2 + binomial 10 2

theorem sum_of_squares_equals_165 : sum_of_squares = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_165_l3876_387676


namespace NUMINAMATH_CALUDE_grain_movement_representation_l3876_387644

-- Define the type for grain movement
inductive GrainMovement
  | arrival
  | departure

-- Define a function to represent the sign of grain movement
def signOfMovement (g : GrainMovement) : Int :=
  match g with
  | GrainMovement.arrival => 1
  | GrainMovement.departure => -1

-- Define the theorem
theorem grain_movement_representation :
  ∀ (quantity : ℕ),
  (signOfMovement GrainMovement.arrival * quantity = 30) →
  (signOfMovement GrainMovement.departure * quantity = -30) :=
by
  sorry


end NUMINAMATH_CALUDE_grain_movement_representation_l3876_387644


namespace NUMINAMATH_CALUDE_annual_growth_rate_l3876_387682

theorem annual_growth_rate (initial_amount final_amount : ℝ) (h : initial_amount * (1 + 0.125)^2 = final_amount) :
  ∃ (rate : ℝ), initial_amount * (1 + rate)^2 = final_amount ∧ rate = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l3876_387682


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3876_387648

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 4 + a 7 = 10 →
  a 3 + a 6 + a 9 = 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3876_387648


namespace NUMINAMATH_CALUDE_six_factorial_divisors_l3876_387657

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define function to count positive divisors
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem six_factorial_divisors :
  count_divisors (factorial 6) = 30 := by
  sorry

end NUMINAMATH_CALUDE_six_factorial_divisors_l3876_387657


namespace NUMINAMATH_CALUDE_coloring_theorem_l3876_387623

/-- A coloring of natural numbers using k colors -/
def Coloring (k : ℕ) := ℕ → Fin k

/-- Proposition: For any coloring of natural numbers using k colors,
    there exist four distinct natural numbers a, b, c, d of the same color
    satisfying the required properties. -/
theorem coloring_theorem (k : ℕ) (coloring : Coloring k) :
  ∃ (a b c d : ℕ) (color : Fin k),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    coloring a = color ∧ coloring b = color ∧ coloring c = color ∧ coloring d = color ∧
    a * d = b * c ∧
    (∃ m : ℕ, b = a * 2^m) ∧
    (∃ n : ℕ, c = a * 3^n) :=
  sorry

end NUMINAMATH_CALUDE_coloring_theorem_l3876_387623


namespace NUMINAMATH_CALUDE_overlapping_squares_area_l3876_387642

theorem overlapping_squares_area (side_length : ℝ) (rotation_angle : ℝ) : 
  side_length = 12 →
  rotation_angle = 30 * π / 180 →
  ∃ (common_area : ℝ), common_area = 48 * Real.sqrt 3 ∧
    common_area = 2 * (1/2 * side_length * (side_length / Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_overlapping_squares_area_l3876_387642


namespace NUMINAMATH_CALUDE_divisibility_by_x_squared_minus_one_cubed_l3876_387637

theorem divisibility_by_x_squared_minus_one_cubed (n : ℕ) :
  ∃ P : Polynomial ℚ, 
    X^(4*n+2) - (2*n+1) * X^(2*n+2) + (2*n+1) * X^(2*n) - 1 = 
    (X^2 - 1)^3 * P :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_x_squared_minus_one_cubed_l3876_387637


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_equals_rectangle_width_l3876_387612

theorem equilateral_triangle_height_equals_rectangle_width (w : ℝ) :
  let rectangle_area := 2 * w^2
  let triangle_side := (2 * w^2 * 4 / Real.sqrt 3).sqrt
  let triangle_height := triangle_side * Real.sqrt 3 / 2
  triangle_height = w * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_equals_rectangle_width_l3876_387612


namespace NUMINAMATH_CALUDE_meeting_at_64th_lamp_l3876_387621

def meet_point (total_intervals : ℕ) (petya_progress : ℕ) (vasya_progress : ℕ) : ℕ :=
  3 * petya_progress + 1

theorem meeting_at_64th_lamp (total_lamps : ℕ) (petya_at : ℕ) (vasya_at : ℕ) 
  (h1 : total_lamps = 100)
  (h2 : petya_at = 22)
  (h3 : vasya_at = 88) :
  meet_point (total_lamps - 1) (petya_at - 1) (total_lamps - vasya_at) = 64 := by
  sorry

end NUMINAMATH_CALUDE_meeting_at_64th_lamp_l3876_387621


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3876_387684

theorem sqrt_inequality : Real.sqrt 6 - Real.sqrt 5 > 2 * Real.sqrt 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3876_387684


namespace NUMINAMATH_CALUDE_john_shopping_expense_l3876_387656

/-- Given John's shopping scenario, prove the amount spent on pants. -/
theorem john_shopping_expense (tshirt_count : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) 
  (h1 : tshirt_count = 3)
  (h2 : tshirt_price = 20)
  (h3 : total_spent = 110) :
  total_spent - (tshirt_count * tshirt_price) = 50 := by
  sorry

end NUMINAMATH_CALUDE_john_shopping_expense_l3876_387656


namespace NUMINAMATH_CALUDE_fish_population_estimate_l3876_387680

/-- Estimate the number of fish in a reservoir using the capture-recapture method. -/
theorem fish_population_estimate
  (M : ℕ) -- Number of fish initially captured, marked, and released
  (m : ℕ) -- Number of fish captured in the second round
  (n : ℕ) -- Number of marked fish found in the second capture
  (h1 : M > 0)
  (h2 : m > 0)
  (h3 : n > 0)
  (h4 : n ≤ m)
  (h5 : n ≤ M) :
  ∃ x : ℚ, x = (M * m : ℚ) / n ∧ x > 0 :=
sorry

end NUMINAMATH_CALUDE_fish_population_estimate_l3876_387680


namespace NUMINAMATH_CALUDE_vector_sum_theorem_l3876_387631

def vector_a : ℝ × ℝ × ℝ := (2, -3, 4)
def vector_b : ℝ × ℝ × ℝ := (-5, 1, 6)
def vector_c : ℝ × ℝ × ℝ := (3, 0, -2)

theorem vector_sum_theorem :
  vector_a.1 + vector_b.1 + vector_c.1 = 0 ∧
  vector_a.2.1 + vector_b.2.1 + vector_c.2.1 = -2 ∧
  vector_a.2.2 + vector_b.2.2 + vector_c.2.2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_theorem_l3876_387631


namespace NUMINAMATH_CALUDE_chip_consumption_theorem_l3876_387604

/-- Calculates the total number of bags of chips consumed in a week -/
def weekly_chip_consumption (breakfast_bags : ℕ) (lunch_bags : ℕ) (days_in_week : ℕ) : ℕ :=
  let dinner_bags := 2 * lunch_bags
  let daily_consumption := breakfast_bags + lunch_bags + dinner_bags
  daily_consumption * days_in_week

/-- Theorem stating that consuming 1 bag for breakfast, 2 for lunch, and doubling lunch for dinner
    every day for a week results in 49 bags consumed -/
theorem chip_consumption_theorem :
  weekly_chip_consumption 1 2 7 = 49 := by
  sorry

#eval weekly_chip_consumption 1 2 7

end NUMINAMATH_CALUDE_chip_consumption_theorem_l3876_387604


namespace NUMINAMATH_CALUDE_expressions_equality_l3876_387688

/-- 
Theorem: The expressions 2a+3bc and (a+b)(2a+c) are equal if and only if a+b+c = 2.
-/
theorem expressions_equality (a b c : ℝ) : 2*a + 3*b*c = (a+b)*(2*a+c) ↔ a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l3876_387688


namespace NUMINAMATH_CALUDE_line_direction_vector_l3876_387675

/-- Given a line passing through two points and a direction vector, prove the scalar value. -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 2) →
  p2 = (2, -3) →
  (a, -2) = (p2.1 - p1.1, p2.2 - p1.2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3876_387675


namespace NUMINAMATH_CALUDE_log_xy_value_l3876_387689

theorem log_xy_value (x y : ℝ) (hxy3 : Real.log (x * y^3) = 1) (hx2y : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l3876_387689


namespace NUMINAMATH_CALUDE_number_of_employees_l3876_387679

def average_salary_without_manager : ℝ := 1200
def average_salary_with_manager : ℝ := 1300
def manager_salary : ℝ := 3300

theorem number_of_employees : 
  ∃ (E : ℕ), 
    (E * average_salary_without_manager + manager_salary) / (E + 1) = average_salary_with_manager ∧ 
    E = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l3876_387679


namespace NUMINAMATH_CALUDE_line_E_passes_through_points_l3876_387606

def point := ℝ × ℝ

-- Define the line equations
def line_A (p : point) : Prop := 3 * p.1 - 2 * p.2 + 1 = 0
def line_B (p : point) : Prop := 4 * p.1 - 5 * p.2 + 13 = 0
def line_C (p : point) : Prop := 5 * p.1 + 2 * p.2 - 17 = 0
def line_D (p : point) : Prop := p.1 + 7 * p.2 - 24 = 0
def line_E (p : point) : Prop := p.1 - 4 * p.2 + 10 = 0

-- Define the given point and the endpoints of the line segment
def given_point : point := (4, 3)
def segment_start : point := (2, 7)
def segment_end : point := (8, -2)

-- Define the trisection points
def trisection_point1 : point := (4, 4)
def trisection_point2 : point := (6, 1)

-- Theorem statement
theorem line_E_passes_through_points :
  (line_E given_point ∨ line_E trisection_point1 ∨ line_E trisection_point2) ∧
  ¬(line_A given_point ∨ line_A trisection_point1 ∨ line_A trisection_point2) ∧
  ¬(line_B given_point ∨ line_B trisection_point1 ∨ line_B trisection_point2) ∧
  ¬(line_C given_point ∨ line_C trisection_point1 ∨ line_C trisection_point2) ∧
  ¬(line_D given_point ∨ line_D trisection_point1 ∨ line_D trisection_point2) :=
by sorry

end NUMINAMATH_CALUDE_line_E_passes_through_points_l3876_387606


namespace NUMINAMATH_CALUDE_perfect_square_pair_iff_in_solution_set_l3876_387645

/-- A pair of integers (a, b) satisfies the perfect square property if
    a^2 + 4b and b^2 + 4a are both perfect squares. -/
def PerfectSquarePair (a b : ℤ) : Prop :=
  ∃ (m n : ℤ), a^2 + 4*b = m^2 ∧ b^2 + 4*a = n^2

/-- The set of solutions for the perfect square pair problem. -/
def SolutionSet : Set (ℤ × ℤ) :=
  {p | ∃ (k : ℤ), p = (k^2, 0) ∨ p = (0, k^2) ∨ p = (k, 1-k) ∨
                   p = (-6, -5) ∨ p = (-5, -6) ∨ p = (-4, -4)}

/-- The main theorem stating that a pair (a, b) satisfies the perfect square property
    if and only if it belongs to the solution set. -/
theorem perfect_square_pair_iff_in_solution_set (a b : ℤ) :
  PerfectSquarePair a b ↔ (a, b) ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_pair_iff_in_solution_set_l3876_387645


namespace NUMINAMATH_CALUDE_central_academy_olympiad_l3876_387629

theorem central_academy_olympiad (j s : ℕ) (hj : j > 0) (hs : s > 0) : 
  (3 * j : ℚ) / 7 = (6 * s : ℚ) / 7 → j = 2 * s := by
  sorry

end NUMINAMATH_CALUDE_central_academy_olympiad_l3876_387629


namespace NUMINAMATH_CALUDE_hcf_problem_l3876_387613

theorem hcf_problem (a b : ℕ) (h : ℕ) : 
  (max a b = 600) →
  (∃ (k : ℕ), lcm a b = h * 11 * 12) →
  gcd a b = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3876_387613


namespace NUMINAMATH_CALUDE_heathers_oranges_l3876_387632

theorem heathers_oranges (initial remaining taken : ℕ) : 
  remaining = initial - taken → 
  taken = 35 → 
  remaining = 25 → 
  initial = 60 := by sorry

end NUMINAMATH_CALUDE_heathers_oranges_l3876_387632


namespace NUMINAMATH_CALUDE_p_and_q_iff_a_in_range_l3876_387665

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x, x^2 + 2*a*x + a + 2 = 0

def q (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- State the theorem
theorem p_and_q_iff_a_in_range (a : ℝ) : 
  (p a ∧ q a) ↔ a ∈ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_p_and_q_iff_a_in_range_l3876_387665


namespace NUMINAMATH_CALUDE_square_of_6y_minus_2_l3876_387695

-- Define the condition
def satisfies_equation (y : ℝ) : Prop := 3 * y^2 + 2 = 5 * y + 7

-- State the theorem
theorem square_of_6y_minus_2 (y : ℝ) (h : satisfies_equation y) : (6 * y - 2)^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_square_of_6y_minus_2_l3876_387695


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3876_387671

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 243 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3876_387671


namespace NUMINAMATH_CALUDE_car_A_time_l3876_387653

/-- Proves that Car A takes 8 hours to reach its destination given the specified conditions -/
theorem car_A_time (speed_A speed_B time_B : ℝ) (ratio : ℝ) : 
  speed_A = 50 →
  speed_B = 25 →
  time_B = 4 →
  ratio = 4 →
  speed_A * (ratio * speed_B * time_B) / speed_A = 8 :=
by
  sorry

#check car_A_time

end NUMINAMATH_CALUDE_car_A_time_l3876_387653


namespace NUMINAMATH_CALUDE_dartboard_area_ratio_l3876_387609

theorem dartboard_area_ratio :
  let outer_square_side : ℝ := 4
  let inner_square_side : ℝ := 2
  let triangle_leg : ℝ := 1 / Real.sqrt 2
  let s : ℝ := (1 / 2) * triangle_leg * triangle_leg
  let p : ℝ := (1 / 2) * (inner_square_side + outer_square_side) * (outer_square_side / 2 - triangle_leg)
  p / s = 12 := by sorry

end NUMINAMATH_CALUDE_dartboard_area_ratio_l3876_387609


namespace NUMINAMATH_CALUDE_water_to_pool_volume_l3876_387694

/-- Proves that one gallon of water fills 1 cubic foot of Jerry's pool --/
theorem water_to_pool_volume 
  (total_water : ℝ) 
  (drinking_cooking : ℝ) 
  (shower_water : ℝ) 
  (pool_length pool_width pool_height : ℝ) 
  (num_showers : ℕ) 
  (h1 : total_water = 1000) 
  (h2 : drinking_cooking = 100) 
  (h3 : shower_water = 20) 
  (h4 : pool_length = 10 ∧ pool_width = 10 ∧ pool_height = 6) 
  (h5 : num_showers = 15) : 
  (total_water - drinking_cooking - num_showers * shower_water) / (pool_length * pool_width * pool_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_to_pool_volume_l3876_387694


namespace NUMINAMATH_CALUDE_car_trip_duration_l3876_387652

/-- Proves that a car trip with given conditions has a total duration of 8 hours -/
theorem car_trip_duration (initial_speed : ℝ) (initial_time : ℝ) (later_speed : ℝ) (avg_speed : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 6)
  (h3 : later_speed = 46)
  (h4 : avg_speed = 34) :
  ∃ (total_time : ℝ), 
    (initial_speed * initial_time + later_speed * (total_time - initial_time)) / total_time = avg_speed ∧
    total_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_car_trip_duration_l3876_387652


namespace NUMINAMATH_CALUDE_jills_uphill_speed_l3876_387650

/-- Jill's speed running up the hill -/
def uphill_speed : ℝ := 9

/-- Jill's speed running down the hill -/
def downhill_speed : ℝ := 12

/-- Hill height in feet -/
def hill_height : ℝ := 900

/-- Total time for running up and down the hill in seconds -/
def total_time : ℝ := 175

theorem jills_uphill_speed :
  (hill_height / uphill_speed + hill_height / downhill_speed = total_time) ∧
  (uphill_speed > 0) ∧
  (downhill_speed > 0) ∧
  (hill_height > 0) ∧
  (total_time > 0) := by
  sorry

end NUMINAMATH_CALUDE_jills_uphill_speed_l3876_387650


namespace NUMINAMATH_CALUDE_train_speeds_l3876_387625

-- Define the problem parameters
def distance : ℝ := 450
def time : ℝ := 5
def speed_difference : ℝ := 6

-- Define the theorem
theorem train_speeds (slower_speed faster_speed : ℝ) : 
  slower_speed > 0 ∧ 
  faster_speed = slower_speed + speed_difference ∧
  distance = (slower_speed + faster_speed) * time →
  slower_speed = 42 ∧ faster_speed = 48 := by
sorry

end NUMINAMATH_CALUDE_train_speeds_l3876_387625


namespace NUMINAMATH_CALUDE_arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l3876_387670

-- 1
theorem arrangement_two_rows (n : ℕ) (m : ℕ) (h : n + m = 7) :
  (Nat.factorial 7) = 5040 :=
sorry

-- 2
theorem arrangement_person_not_at_ends (n : ℕ) (h : n = 7) :
  5 * (Nat.factorial 6) = 3600 :=
sorry

-- 3
theorem arrangement_girls_together (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 4) = 576 :=
sorry

-- 4
theorem arrangement_boys_not_adjacent (boys girls : ℕ) (h1 : boys = 3) (h2 : girls = 4) :
  (Nat.factorial 4) * (Nat.factorial 5 / Nat.factorial 2) = 1440 :=
sorry

end NUMINAMATH_CALUDE_arrangement_two_rows_arrangement_person_not_at_ends_arrangement_girls_together_arrangement_boys_not_adjacent_l3876_387670


namespace NUMINAMATH_CALUDE_fraction_numerator_proof_l3876_387610

theorem fraction_numerator_proof (x : ℚ) : 
  (x / (4 * x + 4) = 3 / 7) → x = -12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_proof_l3876_387610


namespace NUMINAMATH_CALUDE_sum_proper_divisors_81_l3876_387636

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ d ≠ n}

theorem sum_proper_divisors_81 :
  (Finset.sum (Finset.filter (· ≠ 81) (Finset.range 82)) (λ x => if x ∣ 81 then x else 0)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_81_l3876_387636


namespace NUMINAMATH_CALUDE_coins_missing_l3876_387616

theorem coins_missing (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let found := (3 : ℚ) / 4 * lost
  let remaining := x - lost + found
  (x - remaining) / x = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_coins_missing_l3876_387616


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l3876_387633

/-- Proves that the number of orange juice bottles is 42 given the conditions of the problem -/
theorem orange_juice_bottles (orange_cost apple_cost total_bottles total_cost : ℚ)
  (h1 : orange_cost = 70/100)
  (h2 : apple_cost = 60/100)
  (h3 : total_bottles = 70)
  (h4 : total_cost = 4620/100)
  (h5 : ∃ (orange apple : ℚ), orange + apple = total_bottles ∧ 
                               orange * orange_cost + apple * apple_cost = total_cost) :
  ∃ (orange : ℚ), orange = 42 ∧ 
    ∃ (apple : ℚ), orange + apple = total_bottles ∧ 
                    orange * orange_cost + apple * apple_cost = total_cost :=
by sorry


end NUMINAMATH_CALUDE_orange_juice_bottles_l3876_387633
