import Mathlib

namespace NUMINAMATH_CALUDE_carpet_dimensions_l288_28828

/-- Represents a rectangular carpet with integral side lengths -/
structure Carpet where
  width : ℕ
  length : ℕ

/-- Represents a rectangular room -/
structure Room where
  width : ℕ
  length : ℕ

/-- Checks if a carpet fits perfectly in a room (diagonally) -/
def fitsInRoom (c : Carpet) (r : Room) : Prop :=
  c.width ^ 2 + c.length ^ 2 = r.width ^ 2 + r.length ^ 2

theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 38 →
    r2.width = 50 →
    r1.length = r2.length →
    fitsInRoom c r1 →
    fitsInRoom c r2 →
    c.width = 25 ∧ c.length = 50 := by
  sorry


end NUMINAMATH_CALUDE_carpet_dimensions_l288_28828


namespace NUMINAMATH_CALUDE_ellipse_intersection_l288_28804

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
  Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
  Real.sqrt ((0 - f₁.1)^2 + (0 - f₁.2)^2) +
  Real.sqrt ((0 - f₂.1)^2 + (0 - f₂.2)^2)

theorem ellipse_intersection :
  let f₁ : ℝ × ℝ := (0, 5)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (28/9, 0)
  is_ellipse f₁ f₂ (0, 0) → is_ellipse f₁ f₂ p :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l288_28804


namespace NUMINAMATH_CALUDE_green_blue_difference_l288_28866

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the count of disks for each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of disks -/
def totalDisks (counts : DiskCounts) : ℕ :=
  counts.blue + counts.yellow + counts.green

/-- Checks if the given counts match the specified ratio -/
def matchesRatio (counts : DiskCounts) (blueRatio yellowRatio greenRatio : ℕ) : Prop :=
  counts.blue * yellowRatio = counts.yellow * blueRatio ∧
  counts.blue * greenRatio = counts.green * blueRatio

theorem green_blue_difference (counts : DiskCounts) :
  totalDisks counts = 72 →
  matchesRatio counts 3 7 8 →
  counts.green - counts.blue = 20 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l288_28866


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l288_28864

def quadratic_equation (b c : ℝ) := fun x : ℝ => x^2 + b*x + c

def roots (f : ℝ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  f r₁ = 0 ∧ f r₂ = 0

theorem correct_quadratic_equation :
  ∃ (b₁ c₁ b₂ c₂ : ℝ),
    roots (quadratic_equation b₁ c₁) 5 3 ∧
    roots (quadratic_equation b₂ c₂) (-7) (-2) ∧
    b₁ = -8 ∧
    c₂ = 14 →
    quadratic_equation (-8) 14 = quadratic_equation b₁ c₂ :=
sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l288_28864


namespace NUMINAMATH_CALUDE_modulus_z_is_sqrt_5_l288_28865

theorem modulus_z_is_sqrt_5 (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_z_is_sqrt_5_l288_28865


namespace NUMINAMATH_CALUDE_fraction_equality_l288_28817

theorem fraction_equality (x y : ℝ) (h : x / 2 = y / 5) : x / y = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l288_28817


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l288_28897

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 9 * a + 18 = 0) →
  (3 * b^3 - 6 * b^2 + 9 * b + 18 = 0) →
  (3 * c^3 - 6 * c^2 + 9 * c + 18 = 0) →
  a^2 + b^2 + c^2 = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l288_28897


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l288_28889

theorem brown_eyed_brunettes (total : ℕ) (blonde_blue : ℕ) (brunette : ℕ) (brown : ℕ)
  (h1 : total = 50)
  (h2 : blonde_blue = 14)
  (h3 : brunette = 31)
  (h4 : brown = 18) :
  brunette + blonde_blue - (total - brown) = 13 := by
  sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l288_28889


namespace NUMINAMATH_CALUDE_power_of_fraction_five_sixths_fourth_l288_28824

theorem power_of_fraction_five_sixths_fourth : (5 / 6 : ℚ) ^ 4 = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_five_sixths_fourth_l288_28824


namespace NUMINAMATH_CALUDE_last_three_nonzero_digits_of_80_factorial_l288_28854

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Returns the last three nonzero digits of a natural number -/
def lastThreeNonzeroDigits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_nonzero_digits_of_80_factorial :
  lastThreeNonzeroDigits (factorial 80) = 712 := by
  sorry

end NUMINAMATH_CALUDE_last_three_nonzero_digits_of_80_factorial_l288_28854


namespace NUMINAMATH_CALUDE_furniture_cost_price_sum_l288_28880

theorem furniture_cost_price_sum (sp1 sp2 sp3 sp4 : ℕ) 
  (h1 : sp1 = 3000) (h2 : sp2 = 2400) (h3 : sp3 = 12000) (h4 : sp4 = 18000) : 
  (sp1 / 120 * 100 + sp2 / 120 * 100 + sp3 / 120 * 100 + sp4 / 120 * 100 : ℕ) = 29500 := by
  sorry

#check furniture_cost_price_sum

end NUMINAMATH_CALUDE_furniture_cost_price_sum_l288_28880


namespace NUMINAMATH_CALUDE_acute_angle_cosine_difference_l288_28818

theorem acute_angle_cosine_difference (α : Real) : 
  0 < α → α < π / 2 →  -- acute angle condition
  3 * Real.sin α = Real.tan α →  -- given equation
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_cosine_difference_l288_28818


namespace NUMINAMATH_CALUDE_min_value_problem_l288_28825

theorem min_value_problem (a : ℝ) (h_a : a > 0) :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ y ≥ a * (x - 3) ∧
    (∀ x' y' : ℝ, x' ≥ 1 → x' + y' ≤ 3 → y' ≥ a * (x' - 3) → 2 * x' + y' ≥ 2 * x + y) ∧
    2 * x + y = 1) →
  a = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_problem_l288_28825


namespace NUMINAMATH_CALUDE_student_distribution_ways_l288_28848

def num_universities : ℕ := 8
def num_students : ℕ := 3
def num_selected_universities : ℕ := 2

theorem student_distribution_ways :
  (num_students.choose 1) * (num_selected_universities.choose 2) * (num_universities.choose 2) = 168 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_ways_l288_28848


namespace NUMINAMATH_CALUDE_ellipses_same_foci_l288_28853

/-- Given two ellipses with equations x²/9 + y²/4 = 1 and x²/(9-k) + y²/(4-k) = 1,
    where k < 4, prove that they have the same foci. -/
theorem ellipses_same_foci (k : ℝ) (h : k < 4) :
  let e1 := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  let e2 := {(x, y) : ℝ × ℝ | x^2 / (9 - k) + y^2 / (4 - k) = 1}
  let foci1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  let foci2 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  foci1 = foci2 := by
sorry


end NUMINAMATH_CALUDE_ellipses_same_foci_l288_28853


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l288_28847

theorem sqrt_difference_approximation : 
  |Real.sqrt 75 - Real.sqrt 72 - 0.17| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l288_28847


namespace NUMINAMATH_CALUDE_product_and_difference_equation_l288_28883

theorem product_and_difference_equation (n v : ℝ) : 
  n = -4.5 → 10 * n = v - 2 * n → v = -9 := by sorry

end NUMINAMATH_CALUDE_product_and_difference_equation_l288_28883


namespace NUMINAMATH_CALUDE_twenty_percent_of_three_and_three_quarters_l288_28893

theorem twenty_percent_of_three_and_three_quarters :
  (20 : ℚ) / 100 * (15 : ℚ) / 4 = (3 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_twenty_percent_of_three_and_three_quarters_l288_28893


namespace NUMINAMATH_CALUDE_soccer_team_physics_count_l288_28816

theorem soccer_team_physics_count (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → 
  math = 10 → 
  both = 4 → 
  math + physics - both = total → 
  physics = 9 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_physics_count_l288_28816


namespace NUMINAMATH_CALUDE_equation_solution_l288_28890

theorem equation_solution :
  ∃ x : ℚ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) ∧ (x = 7 / 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l288_28890


namespace NUMINAMATH_CALUDE_unique_prime_pair_with_prime_root_l288_28857

theorem unique_prime_pair_with_prime_root :
  ∃! (m n : ℕ), Prime m ∧ Prime n ∧
  (∃ x : ℕ, Prime x ∧ x^2 - m*x - n = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_with_prime_root_l288_28857


namespace NUMINAMATH_CALUDE_pythagorean_cube_equation_solutions_l288_28832

theorem pythagorean_cube_equation_solutions :
  ∀ a b c : ℕ+,
    a^2 + b^2 = c^2 ∧ a^3 + b^3 + 1 = (c - 1)^3 →
    ((a = 6 ∧ b = 8 ∧ c = 10) ∨ (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_cube_equation_solutions_l288_28832


namespace NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l288_28811

theorem consecutive_cubes_divisibility (a : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * a * (a^2 + 2) = 3 * a * k₁ ∧ 3 * a * (a^2 + 2) = 9 * k₂ := by
  sorry

end NUMINAMATH_CALUDE_consecutive_cubes_divisibility_l288_28811


namespace NUMINAMATH_CALUDE_line_classification_l288_28830

-- Define the coordinate plane
def CoordinatePlane : Type := ℝ × ℝ

-- Define an integer point
def IntegerPoint (p : CoordinatePlane) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define a line on the coordinate plane
def Line : Type := CoordinatePlane → Prop

-- Define set I as the set of all lines
def I : Set Line := Set.univ

-- Define set M as the set of lines passing through exactly one integer point
def M : Set Line :=
  {l : Line | ∃! (p : CoordinatePlane), IntegerPoint p ∧ l p}

-- Define set N as the set of lines passing through no integer points
def N : Set Line :=
  {l : Line | ∀ (p : CoordinatePlane), l p → ¬IntegerPoint p}

-- Define set P as the set of lines passing through infinitely many integer points
def P : Set Line :=
  {l : Line | ∀ (n : ℕ), ∃ (S : Finset CoordinatePlane),
    Finset.card S = n ∧ (∀ (p : CoordinatePlane), p ∈ S → IntegerPoint p ∧ l p)}

theorem line_classification :
  (M ∪ N ∪ P = I) ∧ (N ≠ ∅) ∧ (M ≠ ∅) ∧ (P ≠ ∅) := by sorry

end NUMINAMATH_CALUDE_line_classification_l288_28830


namespace NUMINAMATH_CALUDE_sign_of_a_equals_sign_of_r_l288_28895

-- Define the variables and their properties
variable (x y : ℝ → ℝ) -- x and y are real-valued functions
variable (r : ℝ) -- r is the correlation coefficient
variable (a b : ℝ) -- a and b are coefficients in the regression line equation

-- Define the linear relationship and regression line
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ t, y t = m * (x t) + c

-- Define the regression line equation
def regression_line (x y : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ t, y t = a * (x t) + b

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  ∃ (cov_xy std_x std_y : ℝ), r = cov_xy / (std_x * std_y) ∧ std_x > 0 ∧ std_y > 0

-- State the theorem
theorem sign_of_a_equals_sign_of_r
  (h_linear : linear_relationship x y)
  (h_regression : regression_line x y a b)
  (h_correlation : correlation_coefficient x y r) :
  (a > 0 ↔ r > 0) ∧ (a < 0 ↔ r < 0) :=
sorry

end NUMINAMATH_CALUDE_sign_of_a_equals_sign_of_r_l288_28895


namespace NUMINAMATH_CALUDE_equation_solution_l288_28882

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25 + x) ∧ (x = -60) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l288_28882


namespace NUMINAMATH_CALUDE_matrix_power_four_l288_28801

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; -1, 0]

theorem matrix_power_four : A^4 = !![(-1 : ℤ), (-1 : ℤ); (1 : ℤ), (0 : ℤ)] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l288_28801


namespace NUMINAMATH_CALUDE_complement_N_subset_complement_M_l288_28836

/-- The set of real numbers -/
def R : Set ℝ := Set.univ

/-- The set M defined as {x | 0 < x < 2} -/
def M : Set ℝ := {x | 0 < x ∧ x < 2}

/-- The set N defined as {x | x^2 + x - 6 ≤ 0} -/
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

/-- Theorem stating that the complement of N is a subset of the complement of M -/
theorem complement_N_subset_complement_M : (R \ N) ⊆ (R \ M) := by
  sorry

end NUMINAMATH_CALUDE_complement_N_subset_complement_M_l288_28836


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l288_28899

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial expansion function
def expandPolynomial (a b : ℝ) (n : ℕ) : (ℕ → ℝ) := sorry

-- Theorem statement
theorem coefficient_x_cubed_expansion :
  let expansion := expandPolynomial 1 (-1) 5
  let coefficient_x_cubed := (expansion 3) + (expansion 1)
  coefficient_x_cubed = -15 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l288_28899


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l288_28842

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 180)
  (x_smallest : x ≤ y ∧ x ≤ z)
  (y_largest : y ≥ x ∧ y ≥ z)
  (n_def : n = 8 * x)
  (y_def : y = n + 10)
  (z_def : z = n - 10) :
  x * y * z = (180 / 17) * ((1440 / 17)^2 - 100) := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l288_28842


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l288_28821

def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

theorem geometric_sequence_property (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) :
  (is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ = a₂ * a₃) ∧
  (∃ b₁ b₂ b₃ b₄ : ℝ, b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ b₃ ≠ 0 ∧ b₄ ≠ 0 ∧
    b₁ * b₄ = b₂ * b₃ ∧ ¬is_geometric_sequence b₁ b₂ b₃ b₄) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l288_28821


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l288_28881

/-- Represents an arithmetic sequence of three real numbers. -/
structure ArithmeticSequence (α : Type*) [LinearOrderedField α] where
  p : α
  q : α
  r : α
  is_arithmetic : q - r = p - q
  decreasing : p ≥ q ∧ q ≥ r
  nonnegative : r ≥ 0

/-- The theorem stating the properties of the quadratic equation and its root. -/
theorem quadratic_root_theorem (α : Type*) [LinearOrderedField α] 
  (seq : ArithmeticSequence α) : 
  (∃ x y : α, x = 2 * y ∧ 
   seq.p * x^2 + seq.q * x + seq.r = 0 ∧ 
   seq.p * y^2 + seq.q * y + seq.r = 0) → 
  (∃ y : α, y = -1/6 ∧ seq.p * y^2 + seq.q * y + seq.r = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l288_28881


namespace NUMINAMATH_CALUDE_pond_length_l288_28803

/-- Given a rectangular field with length 20 m and width 10 m, containing a square pond
    whose area is 1/8 of the field's area, the length of the pond is 5 m. -/
theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 20 →
  field_width = 10 →
  field_length = 2 * field_width →
  pond_area = (1 / 8) * (field_length * field_width) →
  Real.sqrt pond_area = 5 := by
  sorry


end NUMINAMATH_CALUDE_pond_length_l288_28803


namespace NUMINAMATH_CALUDE_lottery_winnings_l288_28892

theorem lottery_winnings (total_given : ℝ) (num_students : ℕ) (fraction : ℝ) :
  total_given = 15525 →
  num_students = 100 →
  fraction = 1 / 1000 →
  ∃ winnings : ℝ, winnings = 155250 ∧ total_given = num_students * (fraction * winnings) :=
by sorry

end NUMINAMATH_CALUDE_lottery_winnings_l288_28892


namespace NUMINAMATH_CALUDE_ahmed_orange_trees_count_l288_28861

-- Define the number of apple and orange trees for Hassan
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2

-- Define the number of apple trees for Ahmed
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

-- Define the total number of trees for Hassan
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Define the relationship between Ahmed's and Hassan's total trees
def ahmed_total_trees (ahmed_orange_trees : ℕ) : ℕ := 
  ahmed_apple_trees + ahmed_orange_trees

-- Theorem stating that Ahmed has 8 orange trees
theorem ahmed_orange_trees_count : 
  ∃ (x : ℕ), ahmed_total_trees x = hassan_total_trees + 9 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_orange_trees_count_l288_28861


namespace NUMINAMATH_CALUDE_brett_marbles_difference_l288_28807

/-- The number of red marbles Brett has -/
def red_marbles : ℕ := 6

/-- The number of blue marbles Brett has -/
def blue_marbles : ℕ := 5 * red_marbles

/-- The difference between blue and red marbles -/
def marble_difference : ℕ := blue_marbles - red_marbles

theorem brett_marbles_difference : marble_difference = 24 := by
  sorry

end NUMINAMATH_CALUDE_brett_marbles_difference_l288_28807


namespace NUMINAMATH_CALUDE_parabola_vertex_and_focus_l288_28810

/-- A parabola is defined by the equation x = (1/8) * y^2 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = (1/8) * p.2^2}

/-- The vertex of a parabola is the point where it turns -/
def Vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The focus of a parabola is a fixed point used in its geometric definition -/
def Focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem parabola_vertex_and_focus :
  Vertex Parabola = (0, 0) ∧ Focus Parabola = (1/2, 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_focus_l288_28810


namespace NUMINAMATH_CALUDE_max_value_expression_l288_28826

theorem max_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  a * b * Real.sqrt 3 + 3 * b * c ≤ 2 ∧ ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 2 ∧ a * b * Real.sqrt 3 + 3 * b * c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l288_28826


namespace NUMINAMATH_CALUDE_desks_per_row_l288_28808

theorem desks_per_row (total_students : ℕ) (restroom_students : ℕ) (rows : ℕ) :
  total_students = 23 →
  restroom_students = 2 →
  rows = 4 →
  let absent_students := 3 * restroom_students - 1
  let present_students := total_students - restroom_students - absent_students
  let total_desks := (3 * present_students) / 2
  total_desks / rows = 6 :=
by sorry

end NUMINAMATH_CALUDE_desks_per_row_l288_28808


namespace NUMINAMATH_CALUDE_rosa_peach_apple_difference_l288_28885

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 17
def steven_apples : ℕ := 16

-- Define Jake's peaches and apples in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples : ℕ := steven_apples + 8

-- Define Rosa's peaches and apples
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples : ℕ := steven_apples / 2

-- Theorem to prove
theorem rosa_peach_apple_difference : rosa_peaches - rosa_apples = 25 := by
  sorry

end NUMINAMATH_CALUDE_rosa_peach_apple_difference_l288_28885


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l288_28875

theorem arithmetic_expression_equality : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l288_28875


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l288_28819

theorem least_four_digit_multiple : ∀ n : ℕ, 
  (1000 ≤ n ∧ n < 10000) → -- four-digit positive integer
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) → -- divisible by 3, 5, and 7
  1050 ≤ n := by
  sorry

#check least_four_digit_multiple

end NUMINAMATH_CALUDE_least_four_digit_multiple_l288_28819


namespace NUMINAMATH_CALUDE_larry_channels_l288_28827

/-- The number of channels Larry has after all changes --/
def final_channels (initial : ℕ) (removed1 removed2 added1 added2 added3 : ℕ) : ℕ :=
  initial - removed1 + added1 - removed2 + added2 + added3

/-- Theorem stating that Larry's final number of channels is 147 --/
theorem larry_channels : 
  final_channels 150 20 12 10 8 7 = 147 := by
  sorry

end NUMINAMATH_CALUDE_larry_channels_l288_28827


namespace NUMINAMATH_CALUDE_multiple_in_selection_l288_28841

theorem multiple_in_selection (S : Finset ℕ) : 
  S ⊆ Finset.range 100 → S.card = 51 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ (k : ℕ), b = k * a :=
sorry

end NUMINAMATH_CALUDE_multiple_in_selection_l288_28841


namespace NUMINAMATH_CALUDE_evaluate_expression_l288_28800

theorem evaluate_expression : 
  3999^3 - 2 * 3998 * 3999^2 - 2 * 3998^2 * 3999 + 3997^3 = 95806315 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l288_28800


namespace NUMINAMATH_CALUDE_cube_root_sum_l288_28823

theorem cube_root_sum (a : ℝ) (h : a^3 = 7) :
  (0.007 : ℝ)^(1/3) + 7000^(1/3) = 10.1 * a := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_l288_28823


namespace NUMINAMATH_CALUDE_clerk_salary_l288_28874

theorem clerk_salary (manager_salary : ℝ) (num_managers : ℕ) (num_clerks : ℕ) (total_salary : ℝ) :
  manager_salary = 5 →
  num_managers = 2 →
  num_clerks = 3 →
  total_salary = 16 →
  ∃ (clerk_salary : ℝ), clerk_salary = 2 ∧ total_salary = num_managers * manager_salary + num_clerks * clerk_salary :=
by
  sorry

end NUMINAMATH_CALUDE_clerk_salary_l288_28874


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l288_28833

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2*x + 1

-- Define the property of having a non-empty solution set
def has_solution (a : ℝ) : Prop := ∃ x, f a x < 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∀ a, has_solution a → a ≤ 1) ∧
  ¬(∀ a, a ≤ 1 → has_solution a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l288_28833


namespace NUMINAMATH_CALUDE_smallest_divisible_by_9_l288_28858

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def insert_digit (a b d : ℕ) : ℕ := a * 10 + d * 10 + b

theorem smallest_divisible_by_9 :
  ∀ d : ℕ, d ≥ 3 →
    is_divisible_by_9 (insert_digit 761 829 d) →
    insert_digit 761 829 3 ≤ insert_digit 761 829 d :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_9_l288_28858


namespace NUMINAMATH_CALUDE_equation_solution_l288_28851

theorem equation_solution (x : ℝ) : 
  x = 46 →
  (8 / (Real.sqrt (x - 10) - 10) + 
   2 / (Real.sqrt (x - 10) - 5) + 
   9 / (Real.sqrt (x - 10) + 5) + 
   15 / (Real.sqrt (x - 10) + 10) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l288_28851


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l288_28877

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  (∃ k : ℕ, k = 30 ∧ 
   (∀ m : ℕ, 2010^m ∣ factorial 2010 → m ≤ k) ∧
   2010^k ∣ factorial 2010) ∧
  2010 = 2 * 3 * 5 * 67 := by
sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l288_28877


namespace NUMINAMATH_CALUDE_parallel_vectors_component_l288_28849

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b,
    then the first component of a must be -1. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = Real.sqrt 3 ∧ b.1 = Real.sqrt 3 ∧ b.2 = -3 ∧
  ∃ (k : ℝ), a = k • b →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_component_l288_28849


namespace NUMINAMATH_CALUDE_upper_bound_for_expression_l288_28850

theorem upper_bound_for_expression (n : ℤ) : 
  (∃ ub : ℤ, 
    (ub = 40) ∧ 
    (∀ m : ℤ, 1 < 4*m + 7 → 4*m + 7 < ub) ∧
    (∃! (l : List ℤ), l.length = 10 ∧ 
      (∀ k : ℤ, k ∈ l ↔ (1 < 4*k + 7 ∧ 4*k + 7 < ub)))) :=
by sorry

end NUMINAMATH_CALUDE_upper_bound_for_expression_l288_28850


namespace NUMINAMATH_CALUDE_equation_solution_l288_28829

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3)}

theorem equation_solution :
  {(a, b, c) : ℕ × ℕ × ℕ | (c - 1) * (a * b - b - a) = a + b - 2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l288_28829


namespace NUMINAMATH_CALUDE_expression_equals_one_l288_28843

theorem expression_equals_one : 
  (50^2 - 9^2) / (40^2 - 8^2) * ((40 - 8) * (40 + 8)) / ((50 - 9) * (50 + 9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l288_28843


namespace NUMINAMATH_CALUDE_unique_natural_number_with_special_division_property_l288_28814

theorem unique_natural_number_with_special_division_property :
  ∃! (n : ℕ), ∃ (a b : ℕ),
    n = 12 * b + a ∧
    n = 10 * a + b ∧
    a ≤ 11 ∧
    b ≤ 9 ∧
    n = 119 := by
  sorry

end NUMINAMATH_CALUDE_unique_natural_number_with_special_division_property_l288_28814


namespace NUMINAMATH_CALUDE_van_distance_theorem_l288_28867

def distance_covered (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  new_speed * (initial_time * time_ratio)

theorem van_distance_theorem (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) :
  initial_time = 5 →
  new_speed = 80 →
  time_ratio = 3/2 →
  distance_covered initial_time new_speed time_ratio = 600 := by
    sorry

end NUMINAMATH_CALUDE_van_distance_theorem_l288_28867


namespace NUMINAMATH_CALUDE_only_two_is_possible_l288_28856

/-- Represents a triangular grid with 9 cells -/
def TriangularGrid := Fin 9 → ℤ

/-- Represents a move on the triangular grid -/
inductive Move
| add (i j : Fin 9) : Move
| subtract (i j : Fin 9) : Move

/-- Applies a move to the grid -/
def applyMove (grid : TriangularGrid) (move : Move) : TriangularGrid :=
  match move with
  | Move.add i j => 
      fun k => if k = i ∨ k = j then grid k + 1 else grid k
  | Move.subtract i j => 
      fun k => if k = i ∨ k = j then grid k - 1 else grid k

/-- Checks if two cells are adjacent in the triangular grid -/
def isAdjacent (i j : Fin 9) : Prop := sorry

/-- Checks if a grid contains consecutive natural numbers from n to n+8 -/
def containsConsecutiveNumbers (grid : TriangularGrid) (n : ℕ) : Prop :=
  ∃ (perm : Fin 9 → Fin 9), ∀ i : Fin 9, grid (perm i) = n + i

/-- The main theorem stating that n = 2 is the only solution -/
theorem only_two_is_possible :
  ∀ (n : ℕ),
    (∃ (grid : TriangularGrid) (moves : List Move),
      (∀ i : Fin 9, grid i = 0) ∧
      (∀ move ∈ moves, ∃ i j, move = Move.add i j ∨ move = Move.subtract i j) ∧
      (∀ move ∈ moves, ∃ i j, isAdjacent i j) ∧
      (containsConsecutiveNumbers (moves.foldl applyMove grid) n)) ↔
    n = 2 := by
  sorry


end NUMINAMATH_CALUDE_only_two_is_possible_l288_28856


namespace NUMINAMATH_CALUDE_combinations_equal_thirty_l288_28886

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 3

/-- The number of finish type options available -/
def num_finishes : ℕ := 2

/-- The total number of combinations of color, painting method, and finish type -/
def total_combinations : ℕ := num_colors * num_methods * num_finishes

/-- Theorem stating that the total number of combinations is 30 -/
theorem combinations_equal_thirty : total_combinations = 30 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_thirty_l288_28886


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l288_28879

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 20 ∧ 1 / x₀ + 1 / y₀ = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l288_28879


namespace NUMINAMATH_CALUDE_quadratic_sum_l288_28839

/-- Given a quadratic polynomial 12x^2 + 144x + 1728, when written in the form a(x+b)^2+c
    where a, b, and c are constants, prove that a + b + c = 1314 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ (a + b + c = 1314) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l288_28839


namespace NUMINAMATH_CALUDE_max_equilateral_triangle_area_in_rectangle_l288_28822

theorem max_equilateral_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 10 ∧ b = 11 →
  ∃ (area : ℝ),
  area = 221 * Real.sqrt 3 - 330 ∧
  (∀ (triangle_area : ℝ),
    (∃ (x y : ℝ),
      0 ≤ x ∧ x ≤ a ∧
      0 ≤ y ∧ y ≤ b ∧
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ area) :=
by sorry

end NUMINAMATH_CALUDE_max_equilateral_triangle_area_in_rectangle_l288_28822


namespace NUMINAMATH_CALUDE_orthic_similarity_condition_l288_28869

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The sequence of orthic triangles starting from an initial triangle -/
def orthicSequence (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℕ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
| 0 => t
| n + 1 => orthicTriangle (orthicSequence t n)

/-- Two triangles are similar -/
def areSimilar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem orthic_similarity_condition (n : ℕ) (p : RegularPolygon n) :
  (∃ (v1 v2 v3 : Fin n) (k : ℕ),
    areSimilar
      (p.vertices v1, p.vertices v2, p.vertices v3)
      (orthicSequence (p.vertices v1, p.vertices v2, p.vertices v3) k))
  ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_orthic_similarity_condition_l288_28869


namespace NUMINAMATH_CALUDE_apple_pie_problem_l288_28838

def max_pies (total_apples unripe_apples apples_per_pie : ℕ) : ℕ :=
  (total_apples - unripe_apples) / apples_per_pie

theorem apple_pie_problem :
  max_pies 34 6 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_problem_l288_28838


namespace NUMINAMATH_CALUDE_A_equality_l288_28873

/-- The number of integer tuples (x₁, x₂, ..., xₖ) satisfying the given conditions -/
def A (n k r : ℕ+) : ℕ := sorry

/-- The theorem stating the equality of A for different arguments -/
theorem A_equality (s t : ℕ+) (hs : s ≥ 2) (ht : t ≥ 2) :
  A (s * t) s t = A (s * (t - 1)) s t ∧ A (s * t) s t = A ((s - 1) * t) s t :=
sorry

end NUMINAMATH_CALUDE_A_equality_l288_28873


namespace NUMINAMATH_CALUDE_complex_function_evaluation_l288_28898

theorem complex_function_evaluation : 
  let z : ℂ := (Complex.I + 1) / (Complex.I - 1)
  let f : ℂ → ℂ := fun x ↦ x^2 - x + 1
  f z = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_function_evaluation_l288_28898


namespace NUMINAMATH_CALUDE_johnny_guitar_picks_l288_28835

theorem johnny_guitar_picks (total red blue yellow : ℕ) : 
  total > 0 → 
  2 * red = total → 
  3 * blue = total → 
  yellow = total - red - blue → 
  blue = 12 → 
  yellow = 6 := by
sorry

end NUMINAMATH_CALUDE_johnny_guitar_picks_l288_28835


namespace NUMINAMATH_CALUDE_max_consecutive_common_divisor_l288_28868

def a (n : ℕ) : ℤ :=
  if 7 ∣ n then n^6 - 2017 else (n^6 - 2017) / 7

theorem max_consecutive_common_divisor :
  (∃ k : ℕ, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) ∧
  (¬∃ k > 2, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_common_divisor_l288_28868


namespace NUMINAMATH_CALUDE_matthews_cracker_distribution_l288_28846

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_person : ℕ) (num_friends : ℕ) : 
  total_crackers = 36 → 
  crackers_per_person = 2 → 
  total_crackers = num_friends * crackers_per_person → 
  num_friends = 18 := by
sorry

end NUMINAMATH_CALUDE_matthews_cracker_distribution_l288_28846


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l288_28852

theorem inscribed_circle_radius (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 8^2) →
  (A₂ = (A₁ + (A₁ + A₂)) / 2) →
  A₁ = π * ((8 * Real.sqrt 3) / 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l288_28852


namespace NUMINAMATH_CALUDE_suzanne_reading_l288_28876

theorem suzanne_reading (total_pages : ℕ) (extra_pages : ℕ) (pages_left : ℕ) 
  (h1 : total_pages = 64)
  (h2 : extra_pages = 16)
  (h3 : pages_left = 18) :
  ∃ (monday_pages : ℕ), 
    monday_pages + (monday_pages + extra_pages) = total_pages - pages_left ∧ 
    monday_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_suzanne_reading_l288_28876


namespace NUMINAMATH_CALUDE_sin_EAF_value_l288_28820

/-- A rectangle ABCD with E and F trisecting CD -/
structure RectangleWithTrisection where
  /-- Point A of the rectangle -/
  A : ℝ × ℝ
  /-- Point B of the rectangle -/
  B : ℝ × ℝ
  /-- Point C of the rectangle -/
  C : ℝ × ℝ
  /-- Point D of the rectangle -/
  D : ℝ × ℝ
  /-- Point E trisecting CD -/
  E : ℝ × ℝ
  /-- Point F trisecting CD -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : (A.1 = D.1) ∧ (B.1 = C.1) ∧ (A.2 = B.2) ∧ (C.2 = D.2)
  /-- AB = 8 -/
  AB_length : (B.1 - A.1) = 8
  /-- BC = 6 -/
  BC_length : (B.2 - C.2) = 6
  /-- E and F trisect CD -/
  trisection : (E.1 - C.1) = (2/3) * (D.1 - C.1) ∧ (F.1 - C.1) = (1/3) * (D.1 - C.1)

/-- The sine of angle EAF in the given rectangle with trisection -/
def sin_EAF (r : RectangleWithTrisection) : ℝ :=
  sorry

/-- Theorem stating that sin ∠EAF = 12√13 / 194 -/
theorem sin_EAF_value (r : RectangleWithTrisection) : 
  sin_EAF r = 12 * Real.sqrt 13 / 194 :=
sorry

end NUMINAMATH_CALUDE_sin_EAF_value_l288_28820


namespace NUMINAMATH_CALUDE_altitude_length_l288_28891

-- Define the right triangle DEF
def RightTriangleDEF (DE DF EF : ℝ) : Prop :=
  DE = 15 ∧ DF = 9 ∧ EF = 12 ∧ DE^2 = DF^2 + EF^2

-- Define the altitude from F to DE
def Altitude (DE DF EF h : ℝ) : Prop :=
  h * DE = 2 * (1/2 * DF * EF)

-- Theorem statement
theorem altitude_length (DE DF EF h : ℝ) 
  (hTriangle : RightTriangleDEF DE DF EF) 
  (hAltitude : Altitude DE DF EF h) : 
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_altitude_length_l288_28891


namespace NUMINAMATH_CALUDE_jam_distribution_and_consumption_l288_28840

/-- Represents the amount of jam and consumption rate for each person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- Proves the correct distribution and consumption rates of jam for Ponchik and Syropchik -/
theorem jam_distribution_and_consumption 
  (total_jam : ℝ)
  (ponchik_hypothetical_days : ℝ)
  (syropchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syropchik : syropchik_hypothetical_days = 20)
  : ∃ (ponchik syropchik : JamConsumption),
    ponchik.amount + syropchik.amount = total_jam ∧
    ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
    syropchik.amount / ponchik_hypothetical_days = ponchik.rate ∧
    ponchik.amount / syropchik_hypothetical_days = syropchik.rate ∧
    ponchik.amount = 40 ∧
    syropchik.amount = 60 ∧
    ponchik.rate = 4/3 ∧
    syropchik.rate = 2 := by
  sorry


end NUMINAMATH_CALUDE_jam_distribution_and_consumption_l288_28840


namespace NUMINAMATH_CALUDE_euler_formula_third_quadrant_l288_28815

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem euler_formula_third_quadrant (θ : ℝ) (k : ℤ) :
  (2 * k * Real.pi + Real.pi / 2 < θ) ∧ (θ ≤ 2 * k * Real.pi + 2 * Real.pi / 3) →
  third_quadrant (cexp (2 * θ * Complex.I)) :=
sorry

end NUMINAMATH_CALUDE_euler_formula_third_quadrant_l288_28815


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l288_28863

/-- Recursive function to calculate the number of valid arrangements -/
def validArrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people around the table -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing for n people -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  (validArrangements n : ℚ) / (totalOutcomes n : ℚ)

theorem no_adjacent_standing_probability :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l288_28863


namespace NUMINAMATH_CALUDE_limit_exponential_function_l288_28872

theorem limit_exponential_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |((2 * Real.exp (x - 1) - 1) ^ ((3 * x - 1) / (x - 1))) - Real.exp 4| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_exponential_function_l288_28872


namespace NUMINAMATH_CALUDE_factorial_power_of_two_l288_28862

theorem factorial_power_of_two (k : ℕ) :
  ∀ n m : ℕ, (2^k).factorial = 2^n * m ↔
  ∃ t : ℕ, n = 2^k - 1 - t ∧ m = (2^k).factorial / 2^(2^k - 1 - t) := by
  sorry

end NUMINAMATH_CALUDE_factorial_power_of_two_l288_28862


namespace NUMINAMATH_CALUDE_first_day_exceeding_150_l288_28871

def paperclips : ℕ → ℕ
  | 0 => 5  -- Monday (day 1)
  | n + 1 => 2 * paperclips n + 2

theorem first_day_exceeding_150 :
  ∃ n : ℕ, paperclips n > 150 ∧ ∀ m : ℕ, m < n → paperclips m ≤ 150 ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_150_l288_28871


namespace NUMINAMATH_CALUDE_residue_mod_14_l288_28805

theorem residue_mod_14 : (320 * 16 - 28 * 5 + 7) % 14 = 3 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_14_l288_28805


namespace NUMINAMATH_CALUDE_checkerboard_coverage_l288_28859

/-- Represents a checkerboard --/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  removed_squares : ℕ

/-- Checks if a checkerboard can be covered by dominoes --/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem stating which boards can be covered --/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ 
  (board ≠ ⟨4, 4, 1⟩ ∧ board ≠ ⟨3, 7, 0⟩ ∧ board ≠ ⟨7, 3, 0⟩) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_l288_28859


namespace NUMINAMATH_CALUDE_inequality_proof_l288_28845

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/3)^(1/3)) 
  (hb : b = Real.log (1/2)) 
  (hc : c = Real.log (1/4) / Real.log (1/3)) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l288_28845


namespace NUMINAMATH_CALUDE_orange_juice_bottles_l288_28834

/-- The number of fluid ounces Christine must buy -/
def min_fl_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 250

/-- The number of fluid ounces in 1 liter -/
def fl_oz_per_liter : ℝ := 33.8

/-- The smallest number of bottles Christine could buy -/
def min_bottles : ℕ := 8

theorem orange_juice_bottles :
  ∃ (n : ℕ), n = min_bottles ∧
  n * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz ∧
  ∀ (m : ℕ), m * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_bottles_l288_28834


namespace NUMINAMATH_CALUDE_simple_interest_problem_l288_28802

theorem simple_interest_problem (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2000 → P = 2500 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l288_28802


namespace NUMINAMATH_CALUDE_average_temperature_problem_l288_28888

theorem average_temperature_problem (T₁ T₂ T₃ T₄ T₅ : ℚ) : 
  (T₁ + T₂ + T₃ + T₄) / 4 = 58 →
  T₁ / T₅ = 7 / 8 →
  T₅ = 32 →
  (T₂ + T₃ + T₄ + T₅) / 4 = 59 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_problem_l288_28888


namespace NUMINAMATH_CALUDE_canada_moose_population_l288_28860

/-- The moose population in Canada, in millions -/
def moose_population : ℝ := 1

/-- The beaver population in Canada, in millions -/
def beaver_population : ℝ := 2 * moose_population

/-- The human population in Canada, in millions -/
def human_population : ℝ := 38

theorem canada_moose_population :
  (beaver_population = 2 * moose_population) →
  (human_population = 19 * beaver_population) →
  (human_population = 38) →
  moose_population = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_canada_moose_population_l288_28860


namespace NUMINAMATH_CALUDE_max_additional_license_plates_l288_28831

def initial_first_set : Finset Char := {'C', 'H', 'L', 'P', 'R'}
def initial_second_set : Finset Char := {'A', 'I', 'O'}
def initial_third_set : Finset Char := {'D', 'M', 'N', 'T'}

def initial_combinations : ℕ := initial_first_set.card * initial_second_set.card * initial_third_set.card

def max_additional_combinations : ℕ := 
  (initial_first_set.card * (initial_second_set.card + 2) * initial_third_set.card) - initial_combinations

theorem max_additional_license_plates : max_additional_combinations = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_additional_license_plates_l288_28831


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l288_28837

theorem product_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (sum_squares_eq : x^2 + y^2 = 120) : 
  x * y = -20 := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l288_28837


namespace NUMINAMATH_CALUDE_max_median_length_l288_28894

theorem max_median_length (a b c m : ℝ) (hA : Real.cos A = 15/17) (ha : a = 2) :
  m ≤ 4 ∧ ∃ (b c : ℝ), m = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_median_length_l288_28894


namespace NUMINAMATH_CALUDE_group_size_problem_l288_28812

theorem group_size_problem (T : ℕ) (L : ℕ) : 
  T > 90 →  -- Total number of people is greater than 90
  L = T - 90 →  -- Number of people under 20 is the total minus 90
  (L : ℚ) / T = 2/5 →  -- Probability of selecting someone under 20 is 0.4
  T = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_group_size_problem_l288_28812


namespace NUMINAMATH_CALUDE_percentage_of_S_grades_l288_28887

def grading_scale (score : ℕ) : String :=
  if 95 ≤ score ∧ score ≤ 100 then "S"
  else if 88 ≤ score ∧ score < 95 then "A"
  else if 80 ≤ score ∧ score < 88 then "B"
  else if 72 ≤ score ∧ score < 80 then "C"
  else if 65 ≤ score ∧ score < 72 then "D"
  else "F"

def scores : List ℕ := [95, 88, 70, 100, 75, 90, 80, 77, 67, 78, 85, 65, 72, 82, 96]

theorem percentage_of_S_grades :
  (scores.filter (λ score => grading_scale score = "S")).length / scores.length * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_S_grades_l288_28887


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_constant_equality_condition_l288_28813

theorem inequality_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) :=
by sorry

theorem largest_constant :
  ∀ C : ℝ, (∀ x₁ x₂ x₃ x₄ x₅ x₆ : ℝ, (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ C * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂))) → C ≤ 3 :=
by sorry

theorem equality_condition (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 = 3 * (x₁*(x₂ + x₃) + x₂*(x₃ + x₄) + x₃*(x₄ + x₅) + x₄*(x₅ + x₆) + x₅*(x₆ + x₁) + x₆*(x₁ + x₂)) ↔
  x₁ + x₄ = x₂ + x₅ ∧ x₂ + x₅ = x₃ + x₆ :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_constant_equality_condition_l288_28813


namespace NUMINAMATH_CALUDE_solve_for_y_l288_28896

theorem solve_for_y (t : ℝ) (x y : ℝ) : 
  x = 3 - 2*t → y = 5*t + 3 → x = -7 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l288_28896


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l288_28870

theorem det_trig_matrix_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l288_28870


namespace NUMINAMATH_CALUDE_largest_angle_right_triangle_l288_28878

/-- A right triangle with acute angles in the ratio 8:1 has its largest angle measuring 90 degrees. -/
theorem largest_angle_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_acute_ratio : a / b = 8 ∨ b / a = 8) : max a (max b c) = 90 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_right_triangle_l288_28878


namespace NUMINAMATH_CALUDE_conic_intersection_lines_concurrent_l288_28809

-- Define the type for a conic
def Conic := Type

-- Define the type for a point
def Point := Type

-- Define the type for a line
def Line := Type

-- Define a function to check if a point is on a conic
def point_on_conic (p : Point) (c : Conic) : Prop := sorry

-- Define a function to create a line from two points
def line_through_points (p q : Point) : Line := sorry

-- Define a function to check if three lines are concurrent
def are_concurrent (l₁ l₂ l₃ : Line) : Prop := sorry

-- Define the theorem
theorem conic_intersection_lines_concurrent 
  (𝓔₁ 𝓔₂ 𝓔₃ : Conic) 
  (A B : Point) 
  (h_common : point_on_conic A 𝓔₁ ∧ point_on_conic A 𝓔₂ ∧ point_on_conic A 𝓔₃ ∧
              point_on_conic B 𝓔₁ ∧ point_on_conic B 𝓔₂ ∧ point_on_conic B 𝓔₃)
  (C D E F G H : Point)
  (h_intersections : point_on_conic C 𝓔₁ ∧ point_on_conic C 𝓔₂ ∧
                     point_on_conic D 𝓔₁ ∧ point_on_conic D 𝓔₂ ∧
                     point_on_conic E 𝓔₁ ∧ point_on_conic E 𝓔₃ ∧
                     point_on_conic F 𝓔₁ ∧ point_on_conic F 𝓔₃ ∧
                     point_on_conic G 𝓔₂ ∧ point_on_conic G 𝓔₃ ∧
                     point_on_conic H 𝓔₂ ∧ point_on_conic H 𝓔₃)
  (ℓ₁₂ := line_through_points C D)
  (ℓ₁₃ := line_through_points E F)
  (ℓ₂₃ := line_through_points G H) :
  are_concurrent ℓ₁₂ ℓ₁₃ ℓ₂₃ := by
  sorry

end NUMINAMATH_CALUDE_conic_intersection_lines_concurrent_l288_28809


namespace NUMINAMATH_CALUDE_annie_initial_money_l288_28844

def hamburger_price : ℕ := 4
def cheeseburger_price : ℕ := 5
def fries_price : ℕ := 3
def milkshake_price : ℕ := 5
def smoothie_price : ℕ := 6

def hamburger_count : ℕ := 8
def cheeseburger_count : ℕ := 5
def fries_count : ℕ := 3
def milkshake_count : ℕ := 6
def smoothie_count : ℕ := 4

def discount : ℕ := 10
def money_left : ℕ := 45

def total_cost : ℕ := 
  hamburger_price * hamburger_count +
  cheeseburger_price * cheeseburger_count +
  fries_price * fries_count +
  milkshake_price * milkshake_count +
  smoothie_price * smoothie_count

def discounted_cost : ℕ := total_cost - discount

theorem annie_initial_money : 
  discounted_cost + money_left = 155 := by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l288_28844


namespace NUMINAMATH_CALUDE_max_n_for_300_triangles_max_n_is_102_l288_28806

/-- Represents a convex polygon with interior points -/
structure ConvexPolygon where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ -- number of interior points
  no_collinear : Prop -- property that no three points are collinear

/-- The number of triangles formed in a convex polygon with interior points -/
def num_triangles (p : ConvexPolygon) : ℕ :=
  p.n + p.interior_points + 198

/-- Theorem stating the maximum value of n for which no more than 300 triangles can be formed -/
theorem max_n_for_300_triangles (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  p.n ≤ 102 := by
  sorry

/-- The maximum value of n is indeed 102 -/
theorem max_n_is_102 (p : ConvexPolygon) 
  (h1 : p.interior_points = 100) 
  (h2 : num_triangles p ≤ 300) : 
  ∃ (q : ConvexPolygon), q.n = 102 ∧ q.interior_points = 100 ∧ num_triangles q = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_n_for_300_triangles_max_n_is_102_l288_28806


namespace NUMINAMATH_CALUDE_distance_calculation_l288_28884

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 34

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 4

/-- Time Brad runs before meeting Maxwell, in hours -/
def brad_time : ℝ := 3

theorem distance_calculation :
  distance_between_homes = maxwell_speed * maxwell_time + brad_speed * brad_time :=
by sorry

end NUMINAMATH_CALUDE_distance_calculation_l288_28884


namespace NUMINAMATH_CALUDE_exists_point_X_l288_28855

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :=
  ∃ (X : ℝ × ℝ),
    -- X is on the circle
    (X.1 - circle.center.1)^2 + (X.2 - circle.center.2)^2 = circle.radius^2 ∧
    -- Define lines AX and BX
    let AX : Line := ⟨A, X⟩
    let BX : Line := ⟨B, X⟩
    -- C and D are intersections of AX and BX with the circle
    ∃ (C D : ℝ × ℝ),
      -- C and D are on the circle
      (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2 ∧
      (D.1 - circle.center.1)^2 + (D.2 - circle.center.2)^2 = circle.radius^2 ∧
      -- C is on AX, D is on BX
      (C.2 - A.2) * (X.1 - A.1) = (C.1 - A.1) * (X.2 - A.2) ∧
      (D.2 - B.2) * (X.1 - B.1) = (D.1 - B.1) * (X.2 - B.2) ∧
      -- CD is parallel to MN
      (C.2 - D.2) * (MN.point2.1 - MN.point1.1) = (C.1 - D.1) * (MN.point2.2 - MN.point1.2)

-- Theorem statement
theorem exists_point_X (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :
  problem_setup A B circle MN :=
sorry

end NUMINAMATH_CALUDE_exists_point_X_l288_28855
