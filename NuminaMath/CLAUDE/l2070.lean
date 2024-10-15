import Mathlib

namespace NUMINAMATH_CALUDE_inequality_constraint_l2070_207065

theorem inequality_constraint (a : ℝ) : 
  (∀ x : ℝ, x > 1 → (a + 1) / x + Real.log x > a) → a ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_l2070_207065


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l2070_207098

theorem integer_solutions_of_system (x y : ℤ) : 
  (4 * x^2 = y^2 + 2*y + 1 + 3 ∧ 
   (2*x)^2 - (y + 1)^2 = 3 ∧ 
   (2*x - y - 1) * (2*x + y + 1) = 3) ↔ 
  ((x = 1 ∧ y = 0) ∨ (x = 1 ∧ y = -2) ∨ (x = -1 ∧ y = 0) ∨ (x = -1 ∧ y = -2)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l2070_207098


namespace NUMINAMATH_CALUDE_circle_center_l2070_207089

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem circle_center : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = 4) ∧ h = 2 ∧ k = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l2070_207089


namespace NUMINAMATH_CALUDE_function_properties_l2070_207097

noncomputable def f (b c x : ℝ) : ℝ := (2 * x^2 + b * x + c) / (x^2 + 1)

noncomputable def F (b c : ℝ) (x : ℝ) : ℝ := Real.log (f b c x) / Real.log 10

theorem function_properties (b c : ℝ) 
  (h_b : b < 0) 
  (h_range : Set.range (f b c) = Set.Icc 1 3) :
  (b = -2 ∧ c = 2) ∧ 
  (∀ x y : ℝ, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → F b c x < F b c y) ∧
  (∀ t : ℝ, Real.log (7/5) / Real.log 10 ≤ F b c (|t - 1/6| - |t + 1/6|) ∧ 
            F b c (|t - 1/6| - |t + 1/6|) ≤ Real.log (13/5) / Real.log 10) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2070_207097


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2070_207077

theorem infinite_solutions_diophantine_equation (n : ℤ) :
  ∃ (S : Set (ℕ × ℕ × ℕ)), Set.Infinite S ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S → (x^2 : ℤ) + y^2 - z^2 = n :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2070_207077


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2070_207026

theorem increasing_function_condition (f : ℝ → ℝ) (h : Monotone f) :
  ∀ a b : ℝ, a + b > 0 ↔ f a + f b > f (-a) + f (-b) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2070_207026


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2070_207045

theorem perfect_square_condition (A B : ℤ) : 
  (800 < A ∧ A < 1300) → 
  B > 1 → 
  A = B^4 → 
  (∃ n : ℤ, A = n^2) ↔ (B = 5 ∨ B = 6) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2070_207045


namespace NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l2070_207044

/-- Represents a cubic structure composed of unit cubes -/
structure CubicStructure where
  size : ℕ
  deriving Repr

/-- Calculates the maximum number of visible unit cubes from a single point outside the cube -/
def maxVisibleUnitCubes (c : CubicStructure) : ℕ :=
  3 * c.size^2 - 3 * (c.size - 1) + 1

/-- Theorem stating that for a 12 × 12 × 12 cube, the maximum number of visible unit cubes is 400 -/
theorem max_visible_cubes_12x12x12 :
  maxVisibleUnitCubes ⟨12⟩ = 400 := by
  sorry

#eval maxVisibleUnitCubes ⟨12⟩

end NUMINAMATH_CALUDE_max_visible_cubes_12x12x12_l2070_207044


namespace NUMINAMATH_CALUDE_specific_cube_figure_surface_area_l2070_207000

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  edge_length : ℝ

/-- Calculate the surface area of a cube figure -/
def surface_area (figure : CubeFigure) : ℝ :=
  sorry

/-- Theorem: The surface area of a specific cube figure is 32 square units -/
theorem specific_cube_figure_surface_area :
  let figure : CubeFigure := { num_cubes := 9, edge_length := 1 }
  surface_area figure = 32 := by sorry

end NUMINAMATH_CALUDE_specific_cube_figure_surface_area_l2070_207000


namespace NUMINAMATH_CALUDE_square_areas_tiles_l2070_207006

theorem square_areas_tiles (x : ℝ) : 
  x > 0 ∧ 
  x^2 + (x + 12)^2 = 2120 → 
  x = 26 ∧ x + 12 = 38 := by
sorry

end NUMINAMATH_CALUDE_square_areas_tiles_l2070_207006


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2070_207004

theorem complex_magnitude_one (r : ℝ) (z : ℂ) (h1 : |r| < 4) (h2 : z + 1/z + 2 = r) : 
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2070_207004


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2070_207074

theorem complex_equation_solution (a b c : ℕ+) 
  (h : (a - b * Complex.I) ^ 2 + c = 13 - 8 * Complex.I) :
  a = 2 ∧ b = 2 ∧ c = 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2070_207074


namespace NUMINAMATH_CALUDE_work_multiple_l2070_207081

/-- Given that P persons can complete a work W in 12 days, 
    and mP persons can complete half of the work (W/2) in 3 days,
    prove that the multiple m is 2. -/
theorem work_multiple (P : ℕ) (W : ℝ) (m : ℝ) 
  (h1 : P > 0) (h2 : W > 0) (h3 : m > 0)
  (complete_full : P * 12 * (W / (P * 12)) = W)
  (complete_half : m * P * 3 * (W / (2 * m * P * 3)) = W / 2) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_multiple_l2070_207081


namespace NUMINAMATH_CALUDE_coin_distribution_six_boxes_l2070_207020

def coinDistribution (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | m + 1 => 2 * coinDistribution m

theorem coin_distribution_six_boxes :
  coinDistribution 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_six_boxes_l2070_207020


namespace NUMINAMATH_CALUDE_jean_spots_on_sides_l2070_207021

/-- Represents the number of spots on different parts of Jean the jaguar. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backAndHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the distribution of spots. -/
theorem jean_spots_on_sides (j : JeanSpots) 
  (h1 : j.upperTorso = j.total / 2)
  (h2 : j.backAndHindquarters = j.total / 3)
  (h3 : j.sides = j.total - j.upperTorso - j.backAndHindquarters)
  (h4 : j.upperTorso = 30) :
  j.sides = 10 := by
  sorry

end NUMINAMATH_CALUDE_jean_spots_on_sides_l2070_207021


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2070_207047

/-- The equation represents a circle if and only if this condition holds -/
def is_circle (a : ℝ) : Prop := 4 + 4 - 4*a > 0

/-- The condition we're examining -/
def condition (a : ℝ) : Prop := a ≤ 2

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_circle a → condition a) ∧
  ¬(∀ a : ℝ, condition a → is_circle a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2070_207047


namespace NUMINAMATH_CALUDE_xyz_value_l2070_207018

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2070_207018


namespace NUMINAMATH_CALUDE_probability_of_arrangement_l2070_207054

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

theorem probability_of_arrangement (total_children num_girls num_boys : ℕ) 
  (h1 : total_children = 20)
  (h2 : num_girls = 11)
  (h3 : num_boys = 9)
  (h4 : total_children = num_girls + num_boys) :
  (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose 20 9 = 
    (Nat.choose 14 9 + 6 * Nat.choose 13 8) / Nat.choose total_children num_boys := by
  sorry

end NUMINAMATH_CALUDE_probability_of_arrangement_l2070_207054


namespace NUMINAMATH_CALUDE_polynomial_simplification_and_evaluation_l2070_207041

theorem polynomial_simplification_and_evaluation (a b : ℝ) :
  (-3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3) ∧
  (a - 2 * b = -1 → -3 * (a - 2 * b)^6 + 5 * (a - 2 * b)^3 = -8) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_and_evaluation_l2070_207041


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_unique_position_2005_l2070_207003

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- The theorem stating that the 669th term of the sequence is 2005 -/
theorem arithmetic_sequence_2005 : arithmetic_sequence 669 = 2005 := by
  sorry

/-- The theorem stating that 669 is the unique position where the sequence equals 2005 -/
theorem unique_position_2005 : ∀ n : ℕ, arithmetic_sequence n = 2005 ↔ n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_unique_position_2005_l2070_207003


namespace NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2070_207075

/-- The y-coordinate of the point on the y-axis equidistant from C(-3, 0) and D(-2, 5) is 2 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 : ℝ)^2 + 0^2 + 0^2 + y^2 = (-2 : ℝ)^2 + 5^2 + 0^2 + (y - 5)^2) ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_y_coordinate_l2070_207075


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l2070_207068

theorem binomial_coefficient_22_5 
  (h1 : Nat.choose 20 3 = 1140)
  (h2 : Nat.choose 20 4 = 4845)
  (h3 : Nat.choose 20 5 = 15504) : 
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l2070_207068


namespace NUMINAMATH_CALUDE_wall_painting_contribution_l2070_207096

/-- Calculates the individual contribution for a wall painting project --/
theorem wall_painting_contribution
  (total_area : ℝ)
  (coverage_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (num_coats : ℕ)
  (h_total_area : total_area = 1600)
  (h_coverage : coverage_per_gallon = 400)
  (h_cost : cost_per_gallon = 45)
  (h_coats : num_coats = 2) :
  (total_area / coverage_per_gallon * cost_per_gallon * num_coats) / 2 = 180 := by
  sorry

#check wall_painting_contribution

end NUMINAMATH_CALUDE_wall_painting_contribution_l2070_207096


namespace NUMINAMATH_CALUDE_exam_mean_score_l2070_207071

/-- Given an exam score distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean score is 76. -/
theorem exam_mean_score (mean std_dev : ℝ)
  (h1 : mean - 2 * std_dev = 60)
  (h2 : mean + 3 * std_dev = 100) :
  mean = 76 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2070_207071


namespace NUMINAMATH_CALUDE_triangle_problem_l2070_207033

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = Real.sqrt 3)
  (h2 : t.a + t.c = 4)
  (h3 : Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = 13 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2070_207033


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2070_207027

/-- Calculates the total cups of ingredients in a recipe given the ratio and amount of sugar -/
def total_cups (butter_ratio : ℚ) (flour_ratio : ℚ) (sugar_ratio : ℚ) (sugar_cups : ℚ) : ℚ :=
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let part_size := sugar_cups / sugar_ratio
  (total_ratio * part_size)

/-- Proves that for a recipe with butter:flour:sugar ratio of 1:5:3 and 6 cups of sugar, 
    the total amount of ingredients is 18 cups -/
theorem recipe_total_cups : 
  total_cups 1 5 3 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2070_207027


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2070_207048

theorem polynomial_evaluation : 
  103^5 - 5 * 103^4 + 10 * 103^3 - 10 * 103^2 + 5 * 103 - 1 = 11036846832 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2070_207048


namespace NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2070_207099

-- Define a triangle
structure Triangle where
  sides : Fin 3 → ℝ
  angles : Fin 3 → ℝ

-- Define properties of triangles
def Triangle.isScalene (t : Triangle) : Prop :=
  ∀ i j : Fin 3, i ≠ j → t.sides i ≠ t.sides j

def Triangle.isEquilateral (t : Triangle) : Prop :=
  ∀ i j : Fin 3, t.sides i = t.sides j

def Triangle.isRight (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i = 90

-- Theorem: A scalene equilateral triangle is impossible
theorem no_scalene_equilateral_triangle :
  ¬∃ t : Triangle, t.isScalene ∧ t.isEquilateral :=
sorry

-- Theorem: A right equilateral triangle is impossible
theorem no_right_equilateral_triangle :
  ¬∃ t : Triangle, t.isRight ∧ t.isEquilateral :=
sorry

end NUMINAMATH_CALUDE_no_scalene_equilateral_triangle_no_right_equilateral_triangle_l2070_207099


namespace NUMINAMATH_CALUDE_cube_inscribed_in_sphere_l2070_207055

theorem cube_inscribed_in_sphere (edge_length : ℝ) (sphere_area : ℝ) : 
  edge_length = Real.sqrt 2 →
  sphere_area = 6 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_cube_inscribed_in_sphere_l2070_207055


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2070_207008

theorem modulus_of_complex_number (z : ℂ) (h : z = 3 - 2*Complex.I) : Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2070_207008


namespace NUMINAMATH_CALUDE_added_balls_relationship_l2070_207040

/-- Proves the relationship between added white and black balls to maintain a specific probability -/
theorem added_balls_relationship (x y : ℕ) : 
  (x + 2 : ℚ) / (5 + x + y : ℚ) = 1/3 → y = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_added_balls_relationship_l2070_207040


namespace NUMINAMATH_CALUDE_specific_cube_surface_area_l2070_207066

/-- Calculates the total surface area of a cube with holes -/
def cubeWithHolesSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (numHoles : ℕ) : ℝ :=
  let originalSurfaceArea := 6 * cubeEdge^2
  let holeArea := numHoles * holeEdge^2
  let newExposedArea := numHoles * 4 * cubeEdge * holeEdge
  originalSurfaceArea - holeArea + newExposedArea

/-- Theorem stating the surface area of a specific cube with holes -/
theorem specific_cube_surface_area :
  cubeWithHolesSurfaceArea 5 2 3 = 258 := by
  sorry

#eval cubeWithHolesSurfaceArea 5 2 3

end NUMINAMATH_CALUDE_specific_cube_surface_area_l2070_207066


namespace NUMINAMATH_CALUDE_unique_solution_l2070_207007

/-- Represents a pair of digits in base r -/
structure DigitPair (r : ℕ) where
  first : ℕ
  second : ℕ
  h_first : first < r
  h_second : second < r

/-- Constructs a number from repeating a digit pair n times in base r -/
def construct_number (r : ℕ) (pair : DigitPair r) (n : ℕ) : ℕ :=
  pair.first * r + pair.second

/-- Checks if a number consists of only ones in base r -/
def all_ones (r : ℕ) (x : ℕ) : Prop :=
  ∀ k, (x / r^k) % r = 1 ∨ (x / r^k) = 0

theorem unique_solution :
  ∀ (r : ℕ) (x : ℕ) (n : ℕ) (pair : DigitPair r),
    2 ≤ r →
    r ≤ 70 →
    x = construct_number r pair n →
    all_ones r (x^2) →
    (r = 7 ∧ x = 26) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2070_207007


namespace NUMINAMATH_CALUDE_last_digit_of_power_tower_plus_one_l2070_207011

theorem last_digit_of_power_tower_plus_one :
  (2^(2^1989) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_tower_plus_one_l2070_207011


namespace NUMINAMATH_CALUDE_fiona_hoodies_l2070_207037

theorem fiona_hoodies (total : ℕ) (casey_extra : ℕ) : 
  total = 8 → casey_extra = 2 → ∃ (fiona : ℕ), 
    fiona + (fiona + casey_extra) = total ∧ fiona = 3 := by
  sorry

end NUMINAMATH_CALUDE_fiona_hoodies_l2070_207037


namespace NUMINAMATH_CALUDE_small_cakes_needed_l2070_207052

/-- Prove that given the conditions, the number of small cakes needed is 630 --/
theorem small_cakes_needed (helpers : ℕ) (large_cakes_needed : ℕ) (hours : ℕ)
  (large_cakes_per_hour : ℕ) (small_cakes_per_hour : ℕ) :
  helpers = 10 →
  large_cakes_needed = 20 →
  hours = 3 →
  large_cakes_per_hour = 2 →
  small_cakes_per_hour = 35 →
  (helpers * hours * small_cakes_per_hour) - 
  (large_cakes_needed * small_cakes_per_hour * hours / large_cakes_per_hour) = 630 := by
  sorry

#check small_cakes_needed

end NUMINAMATH_CALUDE_small_cakes_needed_l2070_207052


namespace NUMINAMATH_CALUDE_max_area_rectangular_frame_l2070_207067

/-- Represents the maximum area of a rectangular frame given budget constraints. -/
theorem max_area_rectangular_frame :
  ∃ (L W : ℕ),
    (3 * L + 5 * W ≤ 100) ∧
    (∀ (L' W' : ℕ), (3 * L' + 5 * W' ≤ 100) → L * W ≥ L' * W') ∧
    L * W = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_frame_l2070_207067


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2070_207086

/-- Given a real number a, f(x) = ax - ln x, and l is the tangent line to f at (1, f(1)),
    prove that the y-intercept of l is 1. -/
theorem tangent_line_y_intercept (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x - Real.log x
  let f' : ℝ → ℝ := λ x => a - 1 / x
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, f 1)
  let l : ℝ → ℝ := λ x => slope * (x - point.1) + point.2
  l 0 = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2070_207086


namespace NUMINAMATH_CALUDE_range_of_f_l2070_207049

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Define the domain
def D : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ D, f x = y} = {y | 2 ≤ y ∧ y < 11} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2070_207049


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2070_207019

theorem trigonometric_expression_equals_one : 
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l2070_207019


namespace NUMINAMATH_CALUDE_marcus_baseball_cards_l2070_207035

/-- Given that Carter has 152 baseball cards and Marcus has 58 more than Carter,
    prove that Marcus has 210 baseball cards. -/
theorem marcus_baseball_cards :
  let carter_cards : ℕ := 152
  let difference : ℕ := 58
  let marcus_cards : ℕ := carter_cards + difference
  marcus_cards = 210 := by sorry

end NUMINAMATH_CALUDE_marcus_baseball_cards_l2070_207035


namespace NUMINAMATH_CALUDE_system_solution_l2070_207076

theorem system_solution (x y m : ℝ) : 
  (3 * x + 2 * y = 4 * m - 5 ∧ 
   2 * x + 3 * y = m ∧ 
   x + y = 2) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2070_207076


namespace NUMINAMATH_CALUDE_investment_average_rate_l2070_207073

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) 
  (h1 : total = 6000)
  (h2 : rate1 = 0.035)
  (h3 : rate2 = 0.055)
  (h4 : ∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) :
  ∃ avg_rate : ℝ, abs (avg_rate - 0.043) < 0.0001 ∧ 
  avg_rate * total = rate1 * (total - x) + rate2 * x :=
sorry

end NUMINAMATH_CALUDE_investment_average_rate_l2070_207073


namespace NUMINAMATH_CALUDE_exists_valid_nail_configuration_l2070_207042

/-- Represents a nail configuration for hanging a painting -/
structure NailConfiguration where
  nails : Fin 4 → Unit

/-- Represents the state of the painting (hanging or fallen) -/
inductive PaintingState
  | Hanging
  | Fallen

/-- Determines the state of the painting given a nail configuration and a set of removed nails -/
def paintingState (config : NailConfiguration) (removed : Set (Fin 4)) : PaintingState :=
  sorry

/-- Theorem stating the existence of a nail configuration satisfying the given conditions -/
theorem exists_valid_nail_configuration :
  ∃ (config : NailConfiguration),
    (∀ (i : Fin 4), paintingState config {i} = PaintingState.Hanging) ∧
    (∀ (i j : Fin 4), i ≠ j → paintingState config {i, j} = PaintingState.Fallen) :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_nail_configuration_l2070_207042


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2070_207016

/-- An isosceles triangle with two sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base : ℝ),
  base > 0 →
  7 + 7 + base = 23 →
  base = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2070_207016


namespace NUMINAMATH_CALUDE_longitude_latitude_unique_identification_l2070_207090

/-- A point on the Earth's surface --/
structure EarthPoint where
  longitude : Real
  latitude : Real

/-- Function to determine if a description can uniquely identify a point --/
def canUniquelyIdentify (description : EarthPoint → Prop) : Prop :=
  ∀ (p1 p2 : EarthPoint), description p1 → description p2 → p1 = p2

/-- Theorem stating that longitude and latitude can uniquely identify a point --/
theorem longitude_latitude_unique_identification :
  canUniquelyIdentify (λ p : EarthPoint => p.longitude = 118 ∧ p.latitude = 40) :=
sorry

end NUMINAMATH_CALUDE_longitude_latitude_unique_identification_l2070_207090


namespace NUMINAMATH_CALUDE_cubic_third_root_l2070_207085

theorem cubic_third_root (a b : ℚ) :
  (∀ x : ℚ, a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 8/3) →
  (a * (-1)^3 + (a + 4*b) * (-1)^2 + (b - 5*a) * (-1) + (10 - a) = 0) →
  (a * 4^3 + (a + 4*b) * 4^2 + (b - 5*a) * 4 + (10 - a) = 0) →
  ∃ x : ℚ, x = 8/3 ∧ a * x^3 + (a + 4*b) * x^2 + (b - 5*a) * x + (10 - a) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_third_root_l2070_207085


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2070_207083

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : 5 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.tan α = -4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2070_207083


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l2070_207078

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def left_append_2 (n : ℕ) : ℕ := 2000 + n

def right_append_2 (n : ℕ) : ℕ := n * 10 + 2

theorem three_digit_number_proof :
  ∃! n : ℕ, is_three_digit n ∧ has_distinct_digits n ∧
  (left_append_2 n - right_append_2 n = 945 ∨ right_append_2 n - left_append_2 n = 945) ∧
  n = 327 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l2070_207078


namespace NUMINAMATH_CALUDE_two_color_similar_ngons_l2070_207062

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define similarity between two n-gons
def AreSimilarNGons (n : ℕ) (k : ℝ) (ngon1 ngon2 : Fin n → Point) : Prop :=
  ∃ (center : Point), ∀ (i : Fin n),
    let p1 := ngon1 i
    let p2 := ngon2 i
    (p2.x - center.x)^2 + (p2.y - center.y)^2 = k^2 * ((p1.x - center.x)^2 + (p1.y - center.y)^2)

theorem two_color_similar_ngons 
  (n : ℕ) 
  (h_n : n ≥ 3) 
  (k : ℝ) 
  (h_k : k > 0 ∧ k ≠ 1) 
  (coloring : Coloring) :
  ∃ (ngon1 ngon2 : Fin n → Point),
    AreSimilarNGons n k ngon1 ngon2 ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon1 i) = c)) ∧
    (∃ (c : Color), (∀ (i : Fin n), coloring (ngon2 i) = c)) :=
by sorry

end NUMINAMATH_CALUDE_two_color_similar_ngons_l2070_207062


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_three_l2070_207032

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_neg_two_eq_neg_three
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_pos : ∀ x > 0, f x = x^2 - 1) :
  f (-2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_three_l2070_207032


namespace NUMINAMATH_CALUDE_triangle_uniqueness_l2070_207061

/-- Given two excircle radii and an altitude of a triangle, 
    the triangle is uniquely determined iff the altitude is not 
    equal to the harmonic mean of the two radii. -/
theorem triangle_uniqueness (ρa ρb mc : ℝ) (h_pos : ρa > 0 ∧ ρb > 0 ∧ mc > 0) :
  ∃! (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (a + b > c ∧ b + c > a ∧ c + a > b) ∧
    (ρa = (a + b + c) / (2 * (b + c))) ∧
    (ρb = (a + b + c) / (2 * (c + a))) ∧
    (mc = 2 * (a * b * c) / ((a + b + c) * c)) ↔ 
  mc ≠ 2 * ρa * ρb / (ρa + ρb) := by
sorry

end NUMINAMATH_CALUDE_triangle_uniqueness_l2070_207061


namespace NUMINAMATH_CALUDE_solution_set_characterization_l2070_207043

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 - (a + 1) * x + 1 < 0

def solution_set (a : ℝ) : Set ℝ :=
  {x | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) (h : a > 0) :
  (a = 2 → solution_set a = Set.Ioo (1/2) 1) ∧
  (0 < a ∧ a < 1 → solution_set a = Set.Ioo 1 (1/a)) ∧
  (a = 1 → solution_set a = ∅) ∧
  (a > 1 → solution_set a = Set.Ioo (1/a) 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l2070_207043


namespace NUMINAMATH_CALUDE_inequality_proof_l2070_207025

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_sum : d * e * f + d * e + e * f + f * d = 4) : 
  ((a + b) * d * e + (b + c) * e * f + (c + a) * f * d)^2 ≥ 
  12 * (a * b * d * e + b * c * e * f + c * a * f * d) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2070_207025


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2070_207030

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 4}
def B : Set Nat := {2, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2070_207030


namespace NUMINAMATH_CALUDE_matrix_determinant_l2070_207051

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 0, 1; -5, 5, -4; 3, 3, 6]
  Matrix.det A = 96 := by sorry

end NUMINAMATH_CALUDE_matrix_determinant_l2070_207051


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2070_207082

theorem smallest_integer_with_remainders : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 6 = 3 ∧
  ∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2070_207082


namespace NUMINAMATH_CALUDE_overall_class_average_l2070_207079

-- Define the percentages of each group
def group1_percent : Real := 0.20
def group2_percent : Real := 0.50
def group3_percent : Real := 1 - group1_percent - group2_percent

-- Define the test averages for each group
def group1_average : Real := 80
def group2_average : Real := 60
def group3_average : Real := 40

-- Define the overall class average
def class_average : Real :=
  group1_percent * group1_average +
  group2_percent * group2_average +
  group3_percent * group3_average

-- Theorem statement
theorem overall_class_average :
  class_average = 58 := by
  sorry

end NUMINAMATH_CALUDE_overall_class_average_l2070_207079


namespace NUMINAMATH_CALUDE_shortest_ant_path_equals_slant_edge_l2070_207053

/-- Represents a regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  slantEdgeLength : ℝ
  dihedralAngle : ℝ

/-- The shortest path for an ant to visit all slant edges and return to the starting point -/
def shortestAntPath (pyramid : RegularHexagonalPyramid) : ℝ :=
  pyramid.slantEdgeLength

theorem shortest_ant_path_equals_slant_edge 
  (pyramid : RegularHexagonalPyramid) 
  (h1 : pyramid.dihedralAngle = 10) : 
  shortestAntPath pyramid = pyramid.slantEdgeLength :=
sorry

end NUMINAMATH_CALUDE_shortest_ant_path_equals_slant_edge_l2070_207053


namespace NUMINAMATH_CALUDE_units_digit_expression_l2070_207056

def units_digit (n : ℤ) : ℕ :=
  (n % 10).toNat

theorem units_digit_expression : units_digit ((8 * 23 * 1982 - 8^3) + 8) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_expression_l2070_207056


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2070_207060

-- Define the interval [1,2]
def I : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 2 }

-- Define the proposition
def P (a : ℝ) : Prop := ∀ x ∈ I, x^2 - a ≤ 0

-- Define the sufficient condition
def S (a : ℝ) : Prop := a ≥ 5

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ a, S a → P a) ∧ (∃ a, P a ∧ ¬S a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2070_207060


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2070_207028

theorem quadratic_equation_solution (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, k * x^2 + x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (x₁ + x₂)^2 + x₁ * x₂ = 4 →
  k = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2070_207028


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l2070_207087

theorem complex_number_real_imag_equal (a : ℝ) : 
  let z : ℂ := (6 + a * Complex.I) / (3 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l2070_207087


namespace NUMINAMATH_CALUDE_power_of_128_fourth_sevenths_l2070_207058

theorem power_of_128_fourth_sevenths (h : 128 = 2^7) : (128 : ℝ)^(4/7) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_fourth_sevenths_l2070_207058


namespace NUMINAMATH_CALUDE_equation_solution_l2070_207084

theorem equation_solution (x : ℝ) : (x - 3)^2 = x^2 - 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2070_207084


namespace NUMINAMATH_CALUDE_polynomial_identity_result_l2070_207064

theorem polynomial_identity_result : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a₀ + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_result_l2070_207064


namespace NUMINAMATH_CALUDE_greatest_x_value_l2070_207022

theorem greatest_x_value : ∃ (x_max : ℚ), 
  (∀ x : ℚ, ((4*x - 16) / (3*x - 4))^2 + ((4*x - 16) / (3*x - 4)) = 6 → x ≤ x_max) ∧
  ((4*x_max - 16) / (3*x_max - 4))^2 + ((4*x_max - 16) / (3*x_max - 4)) = 6 ∧
  x_max = 28/13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2070_207022


namespace NUMINAMATH_CALUDE_sams_new_crime_books_l2070_207038

theorem sams_new_crime_books 
  (used_adventure : ℝ) 
  (used_mystery : ℝ) 
  (total_books : ℝ) 
  (h1 : used_adventure = 13.0)
  (h2 : used_mystery = 17.0)
  (h3 : total_books = 45.0) :
  total_books - (used_adventure + used_mystery) = 15.0 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_crime_books_l2070_207038


namespace NUMINAMATH_CALUDE_gold_bar_value_proof_l2070_207088

/-- The value of one bar of gold -/
def gold_bar_value : ℝ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold Legacy and Aleena have together -/
def total_value : ℝ := 17600

theorem gold_bar_value_proof : 
  gold_bar_value * (legacy_bars + aleena_bars : ℝ) = total_value := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_value_proof_l2070_207088


namespace NUMINAMATH_CALUDE_car_speed_problem_l2070_207029

theorem car_speed_problem (distance_AB : ℝ) (speed_A : ℝ) (time : ℝ) (final_distance : ℝ) :
  distance_AB = 300 →
  speed_A = 40 →
  time = 2 →
  final_distance = 100 →
  (∃ speed_B : ℝ, speed_B = 140 ∨ speed_B = 60) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2070_207029


namespace NUMINAMATH_CALUDE_tower_combinations_l2070_207094

theorem tower_combinations (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) : 
  (Nat.choose n k) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l2070_207094


namespace NUMINAMATH_CALUDE_number_relationship_l2070_207046

theorem number_relationship : 4^(3/10) < 8^(1/4) ∧ 8^(1/4) < 3^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l2070_207046


namespace NUMINAMATH_CALUDE_not_divisible_by_nine_l2070_207012

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem not_divisible_by_nine : ¬(∃ k : ℕ, 48767621 = 9 * k) :=
  by
  have h1 : ∀ n : ℕ, (∃ k : ℕ, n = 9 * k) ↔ (∃ m : ℕ, sum_of_digits n = 9 * m) := by sorry
  have h2 : sum_of_digits 48767621 = 41 := by sorry
  have h3 : ¬(∃ m : ℕ, 41 = 9 * m) := by sorry
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_nine_l2070_207012


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2070_207002

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 * x + 9) = 12 → x = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2070_207002


namespace NUMINAMATH_CALUDE_farmland_cleanup_theorem_l2070_207024

/-- Calculates the remaining area to be cleaned given the total area and cleaned areas -/
def remaining_area (total : Float) (lizzie : Float) (hilltown : Float) (green_valley : Float) : Float :=
  total - (lizzie + hilltown + green_valley)

/-- Theorem stating that the remaining area to be cleaned is 2442.38 square feet -/
theorem farmland_cleanup_theorem :
  remaining_area 9500.0 2534.1 2675.95 1847.57 = 2442.38 := by
  sorry

end NUMINAMATH_CALUDE_farmland_cleanup_theorem_l2070_207024


namespace NUMINAMATH_CALUDE_cosine_sum_special_angle_l2070_207031

/-- 
If the terminal side of angle θ passes through the point (3, -4), 
then cos(θ + π/4) = 7√2/10.
-/
theorem cosine_sum_special_angle (θ : ℝ) : 
  (3 : ℝ) * Real.cos θ = 3 ∧ (3 : ℝ) * Real.sin θ = -4 → 
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_special_angle_l2070_207031


namespace NUMINAMATH_CALUDE_milan_bill_cost_l2070_207080

/-- The total cost of a long distance phone bill given the monthly fee, per-minute cost, and minutes used. -/
def total_cost (monthly_fee : ℚ) (per_minute_cost : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + per_minute_cost * minutes_used

/-- Proof that Milan's long distance bill is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let per_minute_cost : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  total_cost monthly_fee per_minute_cost minutes_used = 2336 / 100 := by
  sorry

#eval total_cost 2 (12/100) 178

end NUMINAMATH_CALUDE_milan_bill_cost_l2070_207080


namespace NUMINAMATH_CALUDE_aunt_age_l2070_207001

/-- Proves that given Cori is 3 years old today, and in 5 years she will be one-third the age of her aunt, her aunt's current age is 19 years. -/
theorem aunt_age (cori_age : ℕ) (aunt_age : ℕ) : 
  cori_age = 3 → 
  (cori_age + 5 : ℕ) = (aunt_age + 5) / 3 → 
  aunt_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_aunt_age_l2070_207001


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l2070_207014

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_satisfies_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l2070_207014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2070_207093

/-- An arithmetic sequence with common difference d and special properties. -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  t : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d
  h3 : a 1 + t^2 = a 2 + t^3
  h4 : a 2 + t^3 = a 3 + t

/-- The theorem stating the properties of the arithmetic sequence. -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.t = -1/2 ∧ seq.d = 3/8 ∧
  (∃ (m p r : ℕ), m < p ∧ p < r ∧
    seq.a m - 2*seq.t^m = seq.a p - 2*seq.t^p ∧
    seq.a p - 2*seq.t^p = seq.a r - 2*seq.t^r ∧
    seq.a r - 2*seq.t^r = 0 ∧
    m = 1 ∧ p = 3 ∧ r = 4) ∧
  (∀ n : ℕ, seq.a n = 3/8 * n - 11/8) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2070_207093


namespace NUMINAMATH_CALUDE_inspector_ratio_l2070_207095

/-- Represents the daily production of a workshop relative to the first workshop -/
structure WorkshopProduction where
  relative_production : ℚ

/-- Represents an inspector group -/
structure InspectorGroup where
  num_inspectors : ℕ

/-- Represents the factory setup and inspection process -/
structure Factory where
  workshops : Fin 6 → WorkshopProduction
  initial_products : ℚ
  inspector_speed : ℚ
  group_a : InspectorGroup
  group_b : InspectorGroup

/-- The theorem stating the ratio of inspectors in group A to group B -/
theorem inspector_ratio (f : Factory) : 
  f.workshops 0 = ⟨1⟩ ∧ 
  f.workshops 1 = ⟨1⟩ ∧ 
  f.workshops 2 = ⟨1⟩ ∧ 
  f.workshops 3 = ⟨1⟩ ∧ 
  f.workshops 4 = ⟨3/4⟩ ∧ 
  f.workshops 5 = ⟨8/3⟩ ∧
  (6 * (f.workshops 0).relative_production + 
   6 * (f.workshops 1).relative_production + 
   6 * (f.workshops 2).relative_production + 
   3 * f.initial_products = 6 * f.inspector_speed * f.group_a.num_inspectors) ∧
  (2 * (f.workshops 3).relative_production + 
   2 * (f.workshops 4).relative_production + 
   2 * f.initial_products = 2 * f.inspector_speed * f.group_b.num_inspectors) ∧
  (6 * (f.workshops 5).relative_production + 
   f.initial_products = 4 * f.inspector_speed * f.group_b.num_inspectors) →
  f.group_a.num_inspectors * 19 = f.group_b.num_inspectors * 18 :=
by sorry

end NUMINAMATH_CALUDE_inspector_ratio_l2070_207095


namespace NUMINAMATH_CALUDE_total_attendees_l2070_207070

/-- The admission fee for children in dollars -/
def child_fee : ℚ := 1.5

/-- The admission fee for adults in dollars -/
def adult_fee : ℚ := 4

/-- The total amount collected in dollars -/
def total_collected : ℚ := 5050

/-- The number of children who attended -/
def num_children : ℕ := 700

/-- The number of adults who attended -/
def num_adults : ℕ := 1500

/-- Theorem: The total number of people who entered the fair is 2200 -/
theorem total_attendees : num_children + num_adults = 2200 := by
  sorry

end NUMINAMATH_CALUDE_total_attendees_l2070_207070


namespace NUMINAMATH_CALUDE_length_of_PQ_l2070_207010

-- Define the points
variable (P Q R S : ℝ × ℝ)

-- Define the distances between points
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- Define isosceles triangle
def isIsosceles (A B C : ℝ × ℝ) : Prop :=
  distance A B = distance A C

-- Define perimeter of a triangle
def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance B C + distance C A

-- State the theorem
theorem length_of_PQ (P Q R S : ℝ × ℝ) :
  isIsosceles P Q R →
  isIsosceles Q R S →
  perimeter Q R S = 24 →
  perimeter P Q R = 23 →
  distance Q R = 10 →
  distance P Q = 6.5 := by sorry

end NUMINAMATH_CALUDE_length_of_PQ_l2070_207010


namespace NUMINAMATH_CALUDE_alfred_incurred_loss_no_gain_percent_l2070_207017

/-- Represents the financial transaction of buying and selling a scooter --/
structure ScooterTransaction where
  purchase_price : ℝ
  repair_cost : ℝ
  taxes_and_fees : ℝ
  accessories_cost : ℝ
  selling_price : ℝ

/-- Calculates the total cost of the scooter transaction --/
def total_cost (t : ScooterTransaction) : ℝ :=
  t.purchase_price + t.repair_cost + t.taxes_and_fees + t.accessories_cost

/-- Theorem stating that Alfred incurred a loss on the scooter transaction --/
theorem alfred_incurred_loss (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  total_cost t > t.selling_price := by
  sorry

/-- Corollary stating that there is no gain percent as Alfred incurred a loss --/
theorem no_gain_percent (t : ScooterTransaction) 
  (h1 : t.purchase_price = 4700)
  (h2 : t.repair_cost = 800)
  (h3 : t.taxes_and_fees = 300)
  (h4 : t.accessories_cost = 250)
  (h5 : t.selling_price = 6000) :
  ¬∃ (gain_percent : ℝ), gain_percent > 0 ∧ t.selling_price = total_cost t * (1 + gain_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_alfred_incurred_loss_no_gain_percent_l2070_207017


namespace NUMINAMATH_CALUDE_triangle_sine_law_l2070_207023

theorem triangle_sine_law (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  A = π / 3 →
  a = Real.sqrt 3 →
  c / Real.sin C = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_law_l2070_207023


namespace NUMINAMATH_CALUDE_alpha_minus_beta_range_l2070_207057

theorem alpha_minus_beta_range (α β : Real) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∃ (x : Real), x = α - β ∧ -3*π/2 ≤ x ∧ x ≤ 0 ∧
  ∀ (y : Real), (-3*π/2 ≤ y ∧ y ≤ 0) → ∃ (α' β' : Real), 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ y = α' - β' :=
by
  sorry

end NUMINAMATH_CALUDE_alpha_minus_beta_range_l2070_207057


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2070_207039

/-- The range of m for which the quadratic equation (m-1)x^2 + 2x + 1 = 0 has two distinct real roots -/
theorem quadratic_distinct_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 + 2 * x + 1 = 0 ∧ (m - 1) * y^2 + 2 * y + 1 = 0) ↔ 
  (m < 2 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l2070_207039


namespace NUMINAMATH_CALUDE_base6_addition_puzzle_l2070_207069

/-- Converts a base 6 number to base 10 -/
def base6_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 36 + tens * 6 + ones

/-- Converts a base 10 number to base 6 -/
def base10_to_base6 (n : Nat) : Nat × Nat × Nat :=
  let hundreds := n / 36
  let tens := (n % 36) / 6
  let ones := n % 6
  (hundreds, tens, ones)

theorem base6_addition_puzzle :
  ∃ (S H E : Nat),
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧
    S < 6 ∧ H < 6 ∧ E < 6 ∧
    S ≠ H ∧ S ≠ E ∧ H ≠ E ∧
    base6_to_base10 S H E + base6_to_base10 0 H E = base6_to_base10 S E S ∧
    S = 4 ∧ H = 1 ∧ E = 2 ∧
    base10_to_base6 (S + H + E) = (0, 1, 1) := by
  sorry

#eval base10_to_base6 7  -- Expected output: (0, 1, 1)

end NUMINAMATH_CALUDE_base6_addition_puzzle_l2070_207069


namespace NUMINAMATH_CALUDE_rod_length_l2070_207072

theorem rod_length (pieces : ℝ) (piece_length : ℝ) (h1 : pieces = 118.75) (h2 : piece_length = 0.40) :
  pieces * piece_length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l2070_207072


namespace NUMINAMATH_CALUDE_amusement_park_spending_l2070_207063

theorem amusement_park_spending (initial_amount snack_cost : ℕ) : 
  initial_amount = 100 →
  snack_cost = 20 →
  initial_amount - (snack_cost + 3 * snack_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l2070_207063


namespace NUMINAMATH_CALUDE_solve_equation_l2070_207092

theorem solve_equation (x : ℝ) :
  Real.sqrt ((3 / x) + 3 * x) = 3 →
  x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2070_207092


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2070_207036

theorem fraction_evaluation : (5 * 6 + 4) / 8 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2070_207036


namespace NUMINAMATH_CALUDE_range_of_a_l2070_207034

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x + 2/x + a ≥ 0) → a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2070_207034


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_50_l2070_207005

theorem last_three_digits_of_7_to_50 : 7^50 % 1000 = 991 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_50_l2070_207005


namespace NUMINAMATH_CALUDE_max_difference_PA_PB_l2070_207015

/-- Curve C₂ -/
def C₂ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Point A on the negative x-axis -/
def A : ℝ × ℝ := (-2, 0)

/-- Given point B -/
def B : ℝ × ℝ := (1, 1)

/-- Distance squared between two points -/
def dist_squared (p₁ p₂ : ℝ × ℝ) : ℝ :=
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2

theorem max_difference_PA_PB :
  ∃ (max : ℝ), max = 2 + 2 * Real.sqrt 39 ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
  dist_squared P A - dist_squared P B ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_difference_PA_PB_l2070_207015


namespace NUMINAMATH_CALUDE_coin_flip_problem_l2070_207059

theorem coin_flip_problem (n : ℕ) : 
  (1 + n : ℚ) / 2^n = 3/16 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l2070_207059


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2070_207050

/-- Proves that the speed of a boat in still water is 16 km/hr, given the rate of the stream and the time and distance traveled downstream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 8)
  (h3 : downstream_distance = 168) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 16 ∧ 
    downstream_distance = (still_water_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2070_207050


namespace NUMINAMATH_CALUDE_halloween_candy_proof_l2070_207013

/-- Represents the number of candy pieces Debby's sister had -/
def sisters_candy : ℕ := 42

theorem halloween_candy_proof :
  let debbys_candy : ℕ := 32
  let eaten_candy : ℕ := 35
  let remaining_candy : ℕ := 39
  debbys_candy + sisters_candy - eaten_candy = remaining_candy :=
by
  sorry

#check halloween_candy_proof

end NUMINAMATH_CALUDE_halloween_candy_proof_l2070_207013


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2070_207009

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 < 0 ∧ ¬(|x| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2070_207009


namespace NUMINAMATH_CALUDE_snake_diet_l2070_207091

/-- The number of birds each snake eats per day in a forest ecosystem -/
def birds_per_snake (beetles_per_bird : ℕ) (snakes_per_jaguar : ℕ) (num_jaguars : ℕ) (total_beetles : ℕ) : ℕ :=
  (total_beetles / beetles_per_bird) / (num_jaguars * snakes_per_jaguar)

/-- Theorem stating that each snake eats 3 birds per day in the given ecosystem -/
theorem snake_diet :
  birds_per_snake 12 5 6 1080 = 3 := by
  sorry

#eval birds_per_snake 12 5 6 1080

end NUMINAMATH_CALUDE_snake_diet_l2070_207091
