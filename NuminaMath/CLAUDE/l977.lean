import Mathlib

namespace NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l977_97774

/-- The set of solutions to the quadratic equation ax^2 - ax + 1 = 0 -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 = 0}

/-- The theorem stating that A is empty if and only if a is in [0, 4) -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l977_97774


namespace NUMINAMATH_CALUDE_only_B_is_equation_l977_97762

-- Define what an equation is
def is_equation (e : String) : Prop :=
  ∃ (lhs rhs : String), e = lhs ++ "=" ++ rhs

-- Define the given expressions
def expr_A : String := "x-6"
def expr_B : String := "3r+y=5"
def expr_C : String := "-3+x>-2"
def expr_D : String := "4/6=2/3"

-- Theorem statement
theorem only_B_is_equation :
  is_equation expr_B ∧
  ¬is_equation expr_A ∧
  ¬is_equation expr_C ∧
  ¬is_equation expr_D :=
by sorry

end NUMINAMATH_CALUDE_only_B_is_equation_l977_97762


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l977_97754

theorem replaced_girl_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 94)
  (h3 : avg_increase = 3) : 
  ∃ (old_weight : ℝ), 
    old_weight = new_weight - (n * avg_increase) ∧ 
    old_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_replaced_girl_weight_l977_97754


namespace NUMINAMATH_CALUDE_lemonade_second_intermission_l977_97778

theorem lemonade_second_intermission 
  (total : ℝ) 
  (first : ℝ) 
  (third : ℝ) 
  (h1 : total = 0.9166666666666666) 
  (h2 : first = 0.25) 
  (h3 : third = 0.25) : 
  total - (first + third) = 0.4166666666666666 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_second_intermission_l977_97778


namespace NUMINAMATH_CALUDE_largest_number_in_set_l977_97715

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -2 * a = max (-2 * a) (max (5 * a) (max (36 / a) (max (a ^ 3) 2))) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l977_97715


namespace NUMINAMATH_CALUDE_circles_are_separate_l977_97748

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Equation of the first circle: x² + y² = r₁² -/
  c1 : ℝ → ℝ → Prop
  /-- Equation of the second circle: (x-a)² + (y-b)² = r₂² -/
  c2 : ℝ → ℝ → Prop

/-- Definition of separate circles -/
def are_separate (circles : TwoCircles) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ),
    (∀ x y, circles.c1 x y ↔ (x - x₁)^2 + (y - y₁)^2 = r₁^2) ∧
    (∀ x y, circles.c2 x y ↔ (x - x₂)^2 + (y - y₂)^2 = r₂^2) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ + r₂)^2

/-- The two given circles are separate -/
theorem circles_are_separate : are_separate { 
  c1 := λ x y => x^2 + y^2 = 1,
  c2 := λ x y => (x-3)^2 + (y-4)^2 = 9
} := by sorry

end NUMINAMATH_CALUDE_circles_are_separate_l977_97748


namespace NUMINAMATH_CALUDE_abs_value_sum_diff_l977_97793

theorem abs_value_sum_diff (a b c : ℝ) : 
  (|a| = 1) → (|b| = 2) → (|c| = 3) → (a > b) → (b > c) → 
  (a + b - c = 2 ∨ a + b - c = 0) := by
sorry

end NUMINAMATH_CALUDE_abs_value_sum_diff_l977_97793


namespace NUMINAMATH_CALUDE_percentage_problem_l977_97797

theorem percentage_problem (x : ℝ) (h : 0.8 * x = 240) : 0.2 * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l977_97797


namespace NUMINAMATH_CALUDE_exists_point_on_h_with_sum_40_l977_97721

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the function h in terms of g
def h (g : ℝ → ℝ) (x : ℝ) : ℝ := (g x - 2)^2

-- Theorem statement
theorem exists_point_on_h_with_sum_40 (g : ℝ → ℝ) (h : ℝ → ℝ) 
  (h_def : ∀ x, h x = (g x - 2)^2) (g_val : g 4 = 8) :
  ∃ x y, h x = y ∧ x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_exists_point_on_h_with_sum_40_l977_97721


namespace NUMINAMATH_CALUDE_parentheses_removal_l977_97716

theorem parentheses_removal (x y : ℝ) : x - 2 * (y - 1) = x - 2 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l977_97716


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l977_97713

/-- Given that z = (a - i) / (2 - i) is a pure imaginary number, prove that a = -1/2 --/
theorem pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (2 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l977_97713


namespace NUMINAMATH_CALUDE_number_division_problem_l977_97796

theorem number_division_problem (x : ℝ) : x / 0.04 = 200.9 → x = 8.036 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l977_97796


namespace NUMINAMATH_CALUDE_xy_value_l977_97737

theorem xy_value (x y : ℂ) (h : (1 - Complex.I) * x + (1 + Complex.I) * y = 2) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l977_97737


namespace NUMINAMATH_CALUDE_polynomial_equality_l977_97714

theorem polynomial_equality (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l977_97714


namespace NUMINAMATH_CALUDE_fraction_equals_five_l977_97788

theorem fraction_equals_five (a b : ℕ+) (k : ℕ+) 
  (h : (a.val^2 + b.val^2 : ℚ) / (a.val * b.val - 1) = k.val) : k = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_five_l977_97788


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l977_97794

theorem sum_of_tenth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l977_97794


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l977_97722

theorem base_10_to_base_7 : 
  (1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0 : ℕ) = 2468 := by
  sorry

#eval 1 * 7^4 + 0 * 7^3 + 2 * 7^2 + 2 * 7^1 + 4 * 7^0

end NUMINAMATH_CALUDE_base_10_to_base_7_l977_97722


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l977_97723

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the focus to the asymptote is equal to the length of the real axis,
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  let focus_to_asymptote := (b * c) / Real.sqrt (a^2 + b^2)
  focus_to_asymptote = 2 * a →
  c / a = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l977_97723


namespace NUMINAMATH_CALUDE_ball_probability_problem_l977_97744

theorem ball_probability_problem (R B : ℕ) : 
  (R * (R - 1)) / ((R + B) * (R + B - 1)) = 2/7 →
  (2 * R * B) / ((R + B) * (R + B - 1)) = 1/2 →
  R = 105 ∧ B = 91 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_problem_l977_97744


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l977_97720

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 → 
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l977_97720


namespace NUMINAMATH_CALUDE_quadratic_factorization_l977_97712

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l977_97712


namespace NUMINAMATH_CALUDE_greater_solution_of_quadratic_l977_97777

theorem greater_solution_of_quadratic (x : ℝ) :
  x^2 - 5*x - 84 = 0 → x ≤ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_greater_solution_of_quadratic_l977_97777


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l977_97702

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  let arithmetic_mean := (reciprocals.sum) / 4
  arithmetic_mean = 247 / 840 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l977_97702


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l977_97760

theorem geometric_series_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 10)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10))) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l977_97760


namespace NUMINAMATH_CALUDE_hcd_7350_150_minus_12_l977_97718

theorem hcd_7350_150_minus_12 : Nat.gcd 7350 150 - 12 = 138 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7350_150_minus_12_l977_97718


namespace NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l977_97786

/-- A parallelogram with vertices at (0, 0), (7, 0), (3, 5), and (10, 5) has an area of 35 square units. -/
theorem parallelogram_area : ℝ → Prop := fun area =>
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (7, 0)
  let v3 : ℝ × ℝ := (3, 5)
  let v4 : ℝ × ℝ := (10, 5)
  area = 35

/-- The proof of the parallelogram area theorem. -/
theorem parallelogram_area_proof : parallelogram_area 35 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_parallelogram_area_proof_l977_97786


namespace NUMINAMATH_CALUDE_sum_of_cubes_equality_l977_97750

theorem sum_of_cubes_equality (a b c : ℝ) 
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : a^3 + b^3 + c^3 = 3*a*b*c) : 
  a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equality_l977_97750


namespace NUMINAMATH_CALUDE_weight_of_B_l977_97736

/-- Given the weights of four people A, B, C, and D, prove that B weighs 50 kg. -/
theorem weight_of_B (W_A W_B W_C W_D : ℝ) : W_B = 50 :=
  by
  have h1 : W_A + W_B + W_C + W_D = 240 := by sorry
  have h2 : W_A + W_B = 110 := by sorry
  have h3 : W_B + W_C = 100 := by sorry
  have h4 : W_C + W_D = 130 := by sorry
  sorry

#check weight_of_B

end NUMINAMATH_CALUDE_weight_of_B_l977_97736


namespace NUMINAMATH_CALUDE_square_area_ratio_sqrt_l977_97790

theorem square_area_ratio_sqrt (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  Real.sqrt ((side_c ^ 2) / (side_d ^ 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_sqrt_l977_97790


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l977_97729

theorem ceiling_fraction_evaluation :
  (⌈(25 / 11 : ℚ) - ⌈(35 / 25 : ℚ)⌉⌉ : ℚ) /
  (⌈(35 / 11 : ℚ) + ⌈(11 * 25 / 35 : ℚ)⌉⌉ : ℚ) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l977_97729


namespace NUMINAMATH_CALUDE_infinite_solutions_l977_97767

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation we want to prove holds for infinitely many n
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem infinite_solutions :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → equation n :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_l977_97767


namespace NUMINAMATH_CALUDE_serving_size_is_six_ounces_l977_97740

-- Define the given constants
def concentrate_cans : ℕ := 12
def water_cans_per_concentrate : ℕ := 4
def ounces_per_can : ℕ := 12
def total_servings : ℕ := 120

-- Define the theorem
theorem serving_size_is_six_ounces :
  let total_cans := concentrate_cans * (water_cans_per_concentrate + 1)
  let total_ounces := total_cans * ounces_per_can
  let serving_size := total_ounces / total_servings
  serving_size = 6 := by sorry

end NUMINAMATH_CALUDE_serving_size_is_six_ounces_l977_97740


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_l977_97706

theorem sum_of_A_and_C (A B C : ℕ) : A = 238 → A = B + 143 → C = B + 304 → A + C = 637 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_l977_97706


namespace NUMINAMATH_CALUDE_max_sum_roots_l977_97738

/-- Given real numbers b and c, and function f(x) = x^2 + bx + c,
    if f(f(x)) = 0 has exactly three different real roots,
    then the maximum value of the sum of the roots of f(x) is 1/2. -/
theorem max_sum_roots (b c : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0) →
  (∃ (α β : ℝ), f α = 0 ∧ f β = 0 ∧ α + β = -b) →
  (∀ (x y : ℝ), f x = 0 ∧ f y = 0 → x + y ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_roots_l977_97738


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l977_97732

theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, mx + 2 > 0 ↔ x < 2) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_value_l977_97732


namespace NUMINAMATH_CALUDE_step_height_calculation_step_height_proof_l977_97753

theorem step_height_calculation (num_flights : ℕ) (flight_height : ℕ) (total_steps : ℕ) (inches_per_foot : ℕ) : ℕ :=
  let total_height_feet := num_flights * flight_height
  let total_height_inches := total_height_feet * inches_per_foot
  total_height_inches / total_steps

theorem step_height_proof :
  step_height_calculation 9 10 60 12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_step_height_calculation_step_height_proof_l977_97753


namespace NUMINAMATH_CALUDE_bug_crawl_tiles_l977_97763

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  length : Nat
  width : Nat
  tileSize : Nat
  totalTiles : Nat

/-- Calculates the number of tiles a bug crosses when crawling diagonally across the floor. -/
def tilesTraversed (floor : TiledFloor) : Nat :=
  floor.length + floor.width - 1

/-- Theorem stating the number of tiles crossed by a bug on a specific floor. -/
theorem bug_crawl_tiles (floor : TiledFloor) 
  (h1 : floor.length = 17)
  (h2 : floor.width = 10)
  (h3 : floor.tileSize = 1)
  (h4 : floor.totalTiles = 170) :
  tilesTraversed floor = 26 := by
  sorry

#eval tilesTraversed { length := 17, width := 10, tileSize := 1, totalTiles := 170 }

end NUMINAMATH_CALUDE_bug_crawl_tiles_l977_97763


namespace NUMINAMATH_CALUDE_sum_digits_base5_588_l977_97719

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_digits_base5_588 :
  sumDigits (toBase5 588) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_digits_base5_588_l977_97719


namespace NUMINAMATH_CALUDE_union_A_B_l977_97768

def A : Set ℝ := {-1, 1}
def B : Set ℝ := {x | x^2 + x - 2 = 0}

theorem union_A_B : A ∪ B = {-2, -1, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_A_B_l977_97768


namespace NUMINAMATH_CALUDE_subset_complement_implies_a_negative_l977_97710

theorem subset_complement_implies_a_negative 
  (I : Set ℝ) 
  (A B : Set ℝ) 
  (a : ℝ) 
  (h_I : I = Set.univ) 
  (h_A : A = {x : ℝ | x ≤ a + 1}) 
  (h_B : B = {x : ℝ | x ≥ 1}) 
  (h_subset : A ⊆ (I \ B)) : 
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_subset_complement_implies_a_negative_l977_97710


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l977_97707

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- Calculates the surface area of a rectangular prism --/
def surfaceArea (prism : RectangularPrism) : ℝ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Represents the result of cutting a unit cube from a rectangular prism --/
structure CutPrism where
  original : RectangularPrism
  cut_from_corner : Bool

/-- Calculates the surface area of a prism after a unit cube is cut from it --/
def surfaceAreaAfterCut (cut : CutPrism) : ℝ :=
  surfaceArea cut.original

theorem surface_area_unchanged (cut : CutPrism) :
  surfaceArea cut.original = surfaceAreaAfterCut cut :=
sorry

#check surface_area_unchanged

end NUMINAMATH_CALUDE_surface_area_unchanged_l977_97707


namespace NUMINAMATH_CALUDE_length_of_chord_line_equation_l977_97759

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (1,0)
def line_through_P (m b : ℝ) (x y : ℝ) : Prop := y = m*(x - 1)

-- Define the intersection points of the line and parabola
def intersection_points (m b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | parabola x y ∧ line_through_P m b x y}

-- Part 1: Length of chord AB when slope is 1
theorem length_of_chord (A B : ℝ × ℝ) :
  A ∈ intersection_points 1 0 →
  B ∈ intersection_points 1 0 →
  A ≠ B →
  ‖A - B‖ = 2 * Real.sqrt 6 :=
sorry

-- Part 2: Equation of line when PA = -2PB
theorem line_equation (A B : ℝ × ℝ) (m : ℝ) :
  A ∈ intersection_points m 0 →
  B ∈ intersection_points m 0 →
  A ≠ B →
  (A.1 - 1, A.2) = (-2 * (B.1 - 1), -2 * B.2) →
  (m = 1/2 ∨ m = -1/2) :=
sorry

end NUMINAMATH_CALUDE_length_of_chord_line_equation_l977_97759


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l977_97784

theorem sum_of_cubes_divisibility (n : ℤ) : 
  ∃ (k₁ k₂ : ℤ), 3 * n * (n^2 + 2) = 3 * k₁ ∧ 3 * n * (n^2 + 2) = 9 * k₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisibility_l977_97784


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l977_97783

/-- Parabola type representing y = ax^2 + bx --/
structure Parabola where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- Point type representing (x, y) coordinates --/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_relationship (p : Parabola) (m n t : ℝ) :
  3 * p.a + p.b > 0 →
  p.a + p.b < 0 →
  Point.mk (-3) m ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 2 n ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 4 t ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  n < t ∧ t < m := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l977_97783


namespace NUMINAMATH_CALUDE_second_team_odd_second_team_odd_approx_l977_97704

/-- Calculates the odd for the second team in a four-team soccer bet -/
theorem second_team_odd (odd1 odd3 odd4 bet_amount expected_winnings : ℝ) : ℝ :=
  let total_odds := expected_winnings / bet_amount
  let second_team_odd := total_odds / (odd1 * odd3 * odd4)
  second_team_odd

/-- The calculated odd for the second team is approximately 5.23 -/
theorem second_team_odd_approx :
  let odd1 : ℝ := 1.28
  let odd3 : ℝ := 3.25
  let odd4 : ℝ := 2.05
  let bet_amount : ℝ := 5.00
  let expected_winnings : ℝ := 223.0072
  abs (second_team_odd odd1 odd3 odd4 bet_amount expected_winnings - 5.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_second_team_odd_second_team_odd_approx_l977_97704


namespace NUMINAMATH_CALUDE_matt_profit_l977_97725

/-- Represents a baseball card collection --/
structure CardCollection where
  count : ℕ
  value : ℕ

/-- Calculates the total value of a card collection --/
def totalValue (c : CardCollection) : ℕ := c.count * c.value

/-- Represents a trade transaction --/
structure Trade where
  givenCards : List CardCollection
  receivedCards : List CardCollection

/-- Calculates the profit from a trade --/
def tradeProfitᵢ (t : Trade) : ℤ :=
  (t.receivedCards.map totalValue).sum - (t.givenCards.map totalValue).sum

/-- The initial card collection --/
def initialCollection : CardCollection := ⟨8, 6⟩

/-- The four trades Matt made --/
def trades : List Trade := [
  ⟨[⟨2, 6⟩], [⟨3, 2⟩, ⟨1, 9⟩]⟩,
  ⟨[⟨1, 2⟩, ⟨1, 6⟩], [⟨2, 5⟩, ⟨1, 8⟩]⟩,
  ⟨[⟨1, 5⟩, ⟨1, 9⟩], [⟨3, 3⟩, ⟨1, 10⟩, ⟨1, 1⟩]⟩,
  ⟨[⟨2, 3⟩, ⟨1, 8⟩], [⟨2, 7⟩, ⟨1, 4⟩]⟩
]

/-- Calculates the total profit from all trades --/
def totalProfit : ℤ := (trades.map tradeProfitᵢ).sum

theorem matt_profit : totalProfit = 23 := by
  sorry

end NUMINAMATH_CALUDE_matt_profit_l977_97725


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l977_97766

/-- Configuration of semicircles and inscribed circle -/
structure CircleConfiguration where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- Conditions for the circle configuration -/
def valid_configuration (c : CircleConfiguration) : Prop :=
  c.R = 18 ∧ c.r = 9 ∧ c.x > 0 ∧ c.x < c.R ∧
  (c.R - c.x)^2 - c.x^2 = (c.r + c.x)^2 - c.x^2

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius (c : CircleConfiguration) 
  (h : valid_configuration c) : c.x = 8 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l977_97766


namespace NUMINAMATH_CALUDE_rectangle_width_equals_three_l977_97709

theorem rectangle_width_equals_three (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 9)
  (h2 : rect_length = 27)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_three_l977_97709


namespace NUMINAMATH_CALUDE_retailer_profit_is_ten_percent_l977_97700

/-- Calculates the profit percentage for a retailer selling pens --/
def profit_percentage (buy_quantity : ℕ) (buy_price : ℕ) (discount : ℚ) : ℚ :=
  let cost_price := buy_price
  let selling_price_per_pen := 1 - discount
  let total_selling_price := buy_quantity * selling_price_per_pen
  let profit := total_selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percentage is 10% for the given conditions --/
theorem retailer_profit_is_ten_percent :
  profit_percentage 40 36 (1/100) = 10 := by
  sorry

#eval profit_percentage 40 36 (1/100)

end NUMINAMATH_CALUDE_retailer_profit_is_ten_percent_l977_97700


namespace NUMINAMATH_CALUDE_a_1992_b_1992_values_l977_97789

def a : ℕ → ℤ
| 0 => 0
| (n + 1) => 2 * a n - a (n - 1) + 2

def b : ℕ → ℤ
| 0 => 8
| (n + 1) => 2 * b n - b (n - 1)

axiom square_sum : ∀ n > 0, ∃ k : ℤ, a n ^ 2 + b n ^ 2 = k ^ 2

theorem a_1992_b_1992_values : 
  (a 1992 = 1992^2 ∧ b 1992 = 7976) ∨ (a 1992 = 1992^2 ∧ b 1992 = -7960) := by
  sorry

end NUMINAMATH_CALUDE_a_1992_b_1992_values_l977_97789


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l977_97795

/-- A baking recipe with flour and sugar -/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- The amount of flour Mary has already added -/
def flour_added : ℕ := 2

/-- The amount of flour Mary still needs to add -/
def flour_to_add : ℕ := 7

/-- The recipe Mary is using -/
def marys_recipe : Recipe := {
  flour := flour_added + flour_to_add,
  sugar := 3
}

/-- Theorem: The total amount of flour in the recipe is equal to the sum of the flour already added and the flour to be added -/
theorem recipe_flour_amount : marys_recipe.flour = flour_added + flour_to_add := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l977_97795


namespace NUMINAMATH_CALUDE_sequence_determination_l977_97756

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  let n := (p - 1) / 2
  ∀ (a : Fin n → ℕ), 
    (∀ i : Fin n, a i ∈ Finset.range n.succ) →
    Function.Injective a →
    (∀ i j : Fin n, i ≠ j → ∃ (r : ℕ), (a i * a j) % p = r) →
    ∃! (b : Fin n → ℕ), ∀ i : Fin n, a i = b i :=
by sorry

end NUMINAMATH_CALUDE_sequence_determination_l977_97756


namespace NUMINAMATH_CALUDE_cat_food_cans_per_package_l977_97769

theorem cat_food_cans_per_package (cat_packages : ℕ) (dog_packages : ℕ) (dog_cans_per_package : ℕ) (cat_dog_difference : ℕ) :
  cat_packages = 9 →
  dog_packages = 7 →
  dog_cans_per_package = 5 →
  cat_dog_difference = 55 →
  ∃ (cat_cans_per_package : ℕ),
    cat_cans_per_package * cat_packages = dog_cans_per_package * dog_packages + cat_dog_difference ∧
    cat_cans_per_package = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_food_cans_per_package_l977_97769


namespace NUMINAMATH_CALUDE_area_constants_sum_l977_97746

/-- Represents a grid with squares and overlapping circles -/
structure GridWithCircles where
  grid_size : Nat
  square_size : ℝ
  circle_diameter : ℝ
  circle_center_distance : ℝ

/-- Calculates the constants C and D for the area of visible shaded region -/
def calculate_area_constants (g : GridWithCircles) : ℝ × ℝ :=
  sorry

/-- The theorem stating that C + D = 150 for the given configuration -/
theorem area_constants_sum (g : GridWithCircles) 
  (h1 : g.grid_size = 4)
  (h2 : g.square_size = 3)
  (h3 : g.circle_diameter = 6)
  (h4 : g.circle_center_distance = 3) :
  let (C, D) := calculate_area_constants g
  C + D = 150 :=
sorry

end NUMINAMATH_CALUDE_area_constants_sum_l977_97746


namespace NUMINAMATH_CALUDE_square_of_negative_square_l977_97727

theorem square_of_negative_square (x : ℝ) : (-x^2)^2 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l977_97727


namespace NUMINAMATH_CALUDE_artistic_parents_l977_97792

theorem artistic_parents (total : ℕ) (dad : ℕ) (mom : ℕ) (both : ℕ) : 
  total = 40 → dad = 18 → mom = 20 → both = 11 →
  total - (dad + mom - both) = 13 := by
sorry

end NUMINAMATH_CALUDE_artistic_parents_l977_97792


namespace NUMINAMATH_CALUDE_bike_trip_distance_l977_97770

/-- Calculates the total distance traveled given outbound and return times and average speed -/
def total_distance (outbound_time return_time : ℚ) (average_speed : ℚ) : ℚ :=
  let total_time := (outbound_time + return_time) / 60
  total_time * average_speed

/-- Proves that the total distance traveled is 4 miles given the specified conditions -/
theorem bike_trip_distance :
  let outbound_time : ℚ := 15
  let return_time : ℚ := 25
  let average_speed : ℚ := 6
  total_distance outbound_time return_time average_speed = 4 := by
  sorry

#eval total_distance 15 25 6

end NUMINAMATH_CALUDE_bike_trip_distance_l977_97770


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l977_97758

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l977_97758


namespace NUMINAMATH_CALUDE_fundraising_theorem_l977_97799

def fundraising_problem (goal ken_amount : ℕ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_raised := ken_amount + mary_amount + scott_amount
  (mary_amount = 5 * ken_amount) ∧
  (mary_amount = 3 * scott_amount) ∧
  (ken_amount = 600) ∧
  (total_raised - goal = 600)

theorem fundraising_theorem : fundraising_problem 4000 600 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_theorem_l977_97799


namespace NUMINAMATH_CALUDE_inequality_proof_l977_97717

theorem inequality_proof (n : ℕ+) : (2*n+1)^(n:ℕ) ≥ (2*n)^(n:ℕ) + (2*n-1)^(n:ℕ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l977_97717


namespace NUMINAMATH_CALUDE_max_product_of_three_integers_l977_97773

/-- 
Given three integers where two are equal and their sum is 2000,
prove that their maximum product is 8000000000/27.
-/
theorem max_product_of_three_integers (x y z : ℤ) : 
  x = y ∧ x + y + z = 2000 → 
  x * y * z ≤ 8000000000 / 27 := by
sorry

end NUMINAMATH_CALUDE_max_product_of_three_integers_l977_97773


namespace NUMINAMATH_CALUDE_lunes_area_equals_rectangle_area_l977_97705

/-- Given a rectangle with sides a and b, with half-circles drawn outward on each side
    and a circumscribing circle, the area of the lunes (crescent shapes) is equal to
    the area of the rectangle. -/
theorem lunes_area_equals_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let semicircle_area := π * (a^2 + b^2) / 4
  let circumscribed_circle_area := π * (a^2 + b^2) / 4
  let rectangle_area := a * b
  let lunes_area := semicircle_area + rectangle_area - circumscribed_circle_area
  lunes_area = rectangle_area :=
by sorry

end NUMINAMATH_CALUDE_lunes_area_equals_rectangle_area_l977_97705


namespace NUMINAMATH_CALUDE_max_square_plots_l977_97781

/-- Represents the dimensions of the park -/
structure ParkDimensions where
  width : ℕ
  length : ℕ

/-- Represents the constraints for the park division -/
structure ParkConstraints where
  dimensions : ParkDimensions
  pathwayMaterial : ℕ

/-- Calculates the number of square plots given the number of plots along the width -/
def calculatePlots (n : ℕ) : ℕ := n * (2 * n)

/-- Calculates the total length of pathways given the number of plots along the width -/
def calculatePathwayLength (n : ℕ) : ℕ := 120 * n - 90

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (constraints : ParkConstraints) 
  (h1 : constraints.dimensions.width = 30)
  (h2 : constraints.dimensions.length = 60)
  (h3 : constraints.pathwayMaterial = 2010) :
  ∃ (n : ℕ), calculatePlots n = 578 ∧ 
             calculatePathwayLength n ≤ constraints.pathwayMaterial ∧
             ∀ (m : ℕ), m > n → calculatePathwayLength m > constraints.pathwayMaterial :=
  by sorry


end NUMINAMATH_CALUDE_max_square_plots_l977_97781


namespace NUMINAMATH_CALUDE_average_water_added_l977_97761

def water_day1 : ℝ := 318
def water_day2 : ℝ := 312
def water_day3_morning : ℝ := 180
def water_day3_afternoon : ℝ := 162
def num_days : ℝ := 3

theorem average_water_added (water_day1 water_day2 water_day3_morning water_day3_afternoon num_days : ℝ) :
  (water_day1 + water_day2 + water_day3_morning + water_day3_afternoon) / num_days = 324 := by
  sorry

end NUMINAMATH_CALUDE_average_water_added_l977_97761


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l977_97703

theorem smallest_four_digit_divisible_by_4_and_5 :
  ∃ n : ℕ, 
    (1000 ≤ n ∧ n < 10000) ∧  -- four-digit number
    (n % 4 = 0) ∧             -- divisible by 4
    (n % 5 = 0) ∧             -- divisible by 5
    (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) ∧ (m % 4 = 0) ∧ (m % 5 = 0) → n ≤ m) ∧  -- smallest such number
    n = 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_4_and_5_l977_97703


namespace NUMINAMATH_CALUDE_polynomial_expansion_l977_97779

-- Define the polynomials
def p (z : ℝ) : ℝ := 3 * z^2 + 4 * z - 5
def q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the expanded result
def expanded_result (z : ℝ) : ℝ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- Theorem statement
theorem polynomial_expansion (z : ℝ) : p z * q z = expanded_result z := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l977_97779


namespace NUMINAMATH_CALUDE_final_amount_is_correct_l977_97734

/-- Calculates the final amount paid for a shopping trip with specific discounts and promotions -/
def calculate_final_amount (jimmy_shorts : ℕ) (jimmy_short_price : ℚ) 
                           (irene_shirts : ℕ) (irene_shirt_price : ℚ) 
                           (senior_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let jimmy_total := jimmy_shorts * jimmy_short_price
  let irene_total := irene_shirts * irene_shirt_price
  let jimmy_discounted := (jimmy_shorts / 3) * 2 * jimmy_short_price
  let irene_discounted := ((irene_shirts / 3) * 2 + irene_shirts % 3) * irene_shirt_price
  let total_before_discount := jimmy_discounted + irene_discounted
  let discount_amount := total_before_discount * senior_discount
  let total_after_discount := total_before_discount - discount_amount
  let tax_amount := total_after_discount * sales_tax
  total_after_discount + tax_amount

/-- Theorem stating that the final amount paid is $76.55 -/
theorem final_amount_is_correct : 
  calculate_final_amount 3 15 5 17 (1/10) (1/20) = 76.55 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_is_correct_l977_97734


namespace NUMINAMATH_CALUDE_circumcircles_intersect_at_common_point_l977_97747

-- Define the basic structures
structure Point : Type := (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

structure Circle : Type :=
  (center : Point) (radius : ℝ)

-- Define the properties and conditions
def is_acute_triangle (t : Triangle) : Prop := sorry

def are_not_equal (p q : Point) : Prop := sorry

def is_midpoint (m : Point) (p q : Point) : Prop := sorry

def is_midpoint_of_minor_arc (m : Point) (a b c : Point) : Prop := sorry

def is_midpoint_of_major_arc (n : Point) (a b c : Point) : Prop := sorry

def is_incenter (w : Point) (t : Triangle) : Prop := sorry

def is_excenter (x : Point) (t : Triangle) (v : Point) : Prop := sorry

def circumcircle (t : Triangle) : Circle := sorry

def circles_intersect_at_point (c₁ c₂ c₃ : Circle) (p : Point) : Prop := sorry

-- State the theorem
theorem circumcircles_intersect_at_common_point
  (A B C D E F M N W X Y Z : Point) :
  is_acute_triangle (Triangle.mk A B C) →
  are_not_equal A B →
  are_not_equal A C →
  is_midpoint D B C →
  is_midpoint E C A →
  is_midpoint F A B →
  is_midpoint_of_minor_arc M B C A →
  is_midpoint_of_major_arc N B A C →
  is_incenter W (Triangle.mk D E F) →
  is_excenter X (Triangle.mk D E F) D →
  is_excenter Y (Triangle.mk D E F) E →
  is_excenter Z (Triangle.mk D E F) F →
  ∃ (P : Point),
    circles_intersect_at_point
      (circumcircle (Triangle.mk A B C))
      (circumcircle (Triangle.mk W N X))
      (circumcircle (Triangle.mk Y M Z))
      P :=
by
  sorry

end NUMINAMATH_CALUDE_circumcircles_intersect_at_common_point_l977_97747


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l977_97749

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [12, 25, 39, 48]
  (numbers.sum / numbers.length : ℝ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l977_97749


namespace NUMINAMATH_CALUDE_rectangular_floor_tiles_l977_97752

theorem rectangular_floor_tiles (width : ℕ) (length : ℕ) (diagonal_tiles : ℕ) :
  (2 * width = 3 * length) →  -- length-to-width ratio is 3:2
  (diagonal_tiles * diagonal_tiles = 13 * width * width) →  -- diagonal covers whole number of tiles
  (2 * diagonal_tiles - 1 = 45) →  -- total tiles on both diagonals is 45
  (width * length = 245) :=  -- total tiles covering the floor
by sorry

end NUMINAMATH_CALUDE_rectangular_floor_tiles_l977_97752


namespace NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l977_97787

theorem cos_arithmetic_sequence_product (a₁ : ℝ) : 
  let a : ℕ+ → ℝ := λ n => a₁ + (2 * π / 3) * (n.val - 1)
  let S : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (a n)}
  (∃ a b : ℝ, S = {a, b} ∧ a ≠ b) → 
  ∃ a b : ℝ, S = {a, b} ∧ a * b = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l977_97787


namespace NUMINAMATH_CALUDE_minimum_fencing_cost_theorem_l977_97741

/-- Represents the cost per linear foot for different fencing materials -/
structure FencingMaterial where
  wood : ℝ
  chainLink : ℝ
  iron : ℝ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minimumFencingCost (area : ℝ) (uncoveredSide : ℝ) (materials : FencingMaterial) : ℝ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem minimum_fencing_cost_theorem :
  let area : ℝ := 680
  let uncoveredSide : ℝ := 34
  let materials : FencingMaterial := { wood := 5, chainLink := 7, iron := 10 }
  minimumFencingCost area uncoveredSide materials = 438 := by
  sorry

end NUMINAMATH_CALUDE_minimum_fencing_cost_theorem_l977_97741


namespace NUMINAMATH_CALUDE_complex_number_location_l977_97739

def is_in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = 1 + Complex.I) :
  is_in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_complex_number_location_l977_97739


namespace NUMINAMATH_CALUDE_exponent_multiplication_l977_97764

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l977_97764


namespace NUMINAMATH_CALUDE_sum_of_roots_is_nine_l977_97791

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property of f
def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

-- Define a property for f having exactly three distinct real roots
def has_three_distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, (f x = 0 ∧ f y = 0 ∧ f z = 0) ∧ 
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧
  (∀ w : ℝ, f w = 0 → w = x ∨ w = y ∨ w = z)

-- Theorem statement
theorem sum_of_roots_is_nine (f : ℝ → ℝ) 
  (h1 : is_symmetric_about_3 f) 
  (h2 : has_three_distinct_real_roots f) : 
  ∃ x y z : ℝ, f x = 0 ∧ f y = 0 ∧ f z = 0 ∧ x + y + z = 9 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_nine_l977_97791


namespace NUMINAMATH_CALUDE_money_never_equal_l977_97708

/-- Represents the amount of money in Kiriels and Dariels -/
structure Money where
  kiriels : ℕ
  dariels : ℕ

/-- Represents a currency exchange operation -/
inductive Exchange
  | KirielToDariel : ℕ → Exchange
  | DarielToKiriel : ℕ → Exchange

/-- Applies a single exchange operation to a Money value -/
def applyExchange (m : Money) (e : Exchange) : Money :=
  match e with
  | Exchange.KirielToDariel n => 
      ⟨m.kiriels - n, m.dariels + 10 * n⟩
  | Exchange.DarielToKiriel n => 
      ⟨m.kiriels + 10 * n, m.dariels - n⟩

/-- Applies a sequence of exchanges to an initial Money value -/
def applyExchanges (initial : Money) : List Exchange → Money
  | [] => initial
  | e :: es => applyExchanges (applyExchange initial e) es

theorem money_never_equal :
  ∀ (exchanges : List Exchange),
    let final := applyExchanges ⟨0, 1⟩ exchanges
    final.kiriels ≠ final.dariels :=
  sorry


end NUMINAMATH_CALUDE_money_never_equal_l977_97708


namespace NUMINAMATH_CALUDE_gold_rod_weight_sum_l977_97735

theorem gold_rod_weight_sum (a : Fin 5 → ℝ) :
  (∀ i j : Fin 5, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 0 = 4 →                                            -- first term is 4
  a 4 = 2 →                                            -- last term is 2
  a 1 + a 3 = 6 :=                                     -- sum of second and fourth terms is 6
by sorry

end NUMINAMATH_CALUDE_gold_rod_weight_sum_l977_97735


namespace NUMINAMATH_CALUDE_problem_statement_l977_97775

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a-b, 0} → a^2019 + b^2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l977_97775


namespace NUMINAMATH_CALUDE_charlie_has_largest_answer_l977_97782

def alice_calc (start : ℕ) : ℕ := ((start - 3) * 3) + 5

def bob_calc (start : ℕ) : ℕ := ((start * 3) - 3) + 5

def charlie_calc (start : ℕ) : ℕ := ((start - 3) + 5) * 3

theorem charlie_has_largest_answer (start : ℕ) (h : start = 15) :
  charlie_calc start > alice_calc start ∧ charlie_calc start > bob_calc start := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_largest_answer_l977_97782


namespace NUMINAMATH_CALUDE_segment_point_relation_l977_97730

/-- Given a segment AB of length 2 and a point P on AB such that AP² = AB · PB, prove that AP = √5 - 1 -/
theorem segment_point_relation (A B P : ℝ) : 
  (0 ≤ P - A) ∧ (P - A ≤ B - A) ∧  -- P is on segment AB
  (B - A = 2) ∧                    -- AB = 2
  ((P - A)^2 = (B - A) * (B - P))  -- AP² = AB · PB
  → P - A = Real.sqrt 5 - 1 := by sorry

end NUMINAMATH_CALUDE_segment_point_relation_l977_97730


namespace NUMINAMATH_CALUDE_distribution_theorem_l977_97701

/-- The number of ways to distribute 5 students into 3 groups (A, B, C),
    where group A has at least 2 students and groups B and C each have at least 1 student. -/
def distribution_schemes : ℕ := 80

/-- The total number of students -/
def total_students : ℕ := 5

/-- The number of groups -/
def num_groups : ℕ := 3

/-- The minimum number of students in group A -/
def min_group_a : ℕ := 2

/-- The minimum number of students in groups B and C -/
def min_group_bc : ℕ := 1

theorem distribution_theorem :
  (∀ (scheme : Fin total_students → Fin num_groups),
    (∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅)) →
  (Fintype.card {scheme : Fin total_students → Fin num_groups |
    ∃ (a b c : Finset (Fin total_students)),
      a.card ≥ min_group_a ∧
      b.card ≥ min_group_bc ∧
      c.card ≥ min_group_bc ∧
      a ∪ b ∪ c = Finset.univ ∧
      a ∩ b = ∅ ∧ b ∩ c = ∅ ∧ a ∩ c = ∅}) = distribution_schemes :=
by sorry

end NUMINAMATH_CALUDE_distribution_theorem_l977_97701


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l977_97755

/-- Represents the outcome of rolling two 6-sided dice -/
def DiceRoll := Fin 6 × Fin 6

/-- Calculates the sum of a dice roll -/
def sum_roll (roll : DiceRoll) : Nat :=
  (roll.1.val + 1) + (roll.2.val + 1)

/-- Represents the sample space of all possible dice rolls -/
def sample_space : Finset DiceRoll :=
  Finset.product (Finset.univ : Finset (Fin 6)) (Finset.univ : Finset (Fin 6))

/-- Checks if the area of a circle is less than its circumference given its diameter -/
def area_less_than_circumference (d : Nat) : Bool :=
  d * d < 4 * d

/-- The set of favorable outcomes -/
def favorable_outcomes : Finset DiceRoll :=
  sample_space.filter (λ roll => area_less_than_circumference (sum_roll roll))

/-- The probability of the area being less than the circumference -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_is_one_twelfth : probability = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l977_97755


namespace NUMINAMATH_CALUDE_percentage_problem_l977_97765

theorem percentage_problem (N : ℝ) : 
  (0.4 * N = 4/5 * 25 + 4) → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l977_97765


namespace NUMINAMATH_CALUDE_total_balls_count_l977_97785

/-- The number of different colors of balls -/
def num_colors : ℕ := 10

/-- The number of balls for each color -/
def balls_per_color : ℕ := 35

/-- The total number of balls -/
def total_balls : ℕ := num_colors * balls_per_color

theorem total_balls_count : total_balls = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l977_97785


namespace NUMINAMATH_CALUDE_solution_set_and_range_l977_97743

def f (a x : ℝ) : ℝ := -x^2 + a*x + 4

def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem solution_set_and_range :
  (∀ x ∈ Set.Icc (-1 : ℝ) ((Real.sqrt 17 - 1) / 2), f 1 x ≥ g x) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f 1 x > g x) ∧
  (∀ a ∈ Set.Icc (-1 : ℝ) 1, ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≥ g x) ∧
  (∀ a < -1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) ∧
  (∀ a > 1, ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x < g x) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l977_97743


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l977_97771

theorem sum_of_five_consecutive_even_integers (a : ℤ) : 
  (a + (a + 4) = 150) → (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 385) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l977_97771


namespace NUMINAMATH_CALUDE_zero_points_product_bound_l977_97751

noncomputable def f (a x : ℝ) : ℝ := |Real.log x / Real.log a| - (1/2)^x

theorem zero_points_product_bound (a x₁ x₂ : ℝ) 
  (ha : a > 0 ∧ a ≠ 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) : 
  0 < x₁ * x₂ ∧ x₁ * x₂ < 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_points_product_bound_l977_97751


namespace NUMINAMATH_CALUDE_complex_solutions_of_x_squared_equals_negative_four_l977_97757

theorem complex_solutions_of_x_squared_equals_negative_four :
  ∀ x : ℂ, x^2 = -4 ↔ x = 2*I ∨ x = -2*I :=
sorry

end NUMINAMATH_CALUDE_complex_solutions_of_x_squared_equals_negative_four_l977_97757


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_simplification_l977_97731

theorem sin_cos_pi_12_simplification :
  1/2 * Real.sin (π/12) * Real.cos (π/12) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_simplification_l977_97731


namespace NUMINAMATH_CALUDE_parallelepiped_construction_impossible_l977_97772

/-- Represents the five shapes of blocks -/
inductive BlockShape
  | I
  | L
  | T
  | Plus
  | J

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ

/-- Represents the construction requirements -/
structure ConstructionRequirements where
  total_blocks : ℕ
  shapes : List BlockShape
  volume : ℕ

/-- Checks if a parallelepiped satisfies the edge conditions -/
def valid_edges (p : Parallelepiped) : Prop :=
  p.length > 1 ∧ p.width > 1 ∧ p.height > 1

/-- Checks if a parallelepiped can be constructed with given requirements -/
def can_construct (p : Parallelepiped) (req : ConstructionRequirements) : Prop :=
  p.volume = req.volume ∧ valid_edges p

/-- Main theorem: Impossibility of constructing the required parallelepiped -/
theorem parallelepiped_construction_impossible (req : ConstructionRequirements) :
  req.total_blocks = 48 ∧ 
  req.shapes = [BlockShape.I, BlockShape.L, BlockShape.T, BlockShape.Plus, BlockShape.J] ∧
  req.volume = 1990 →
  ¬∃ (p : Parallelepiped), can_construct p req :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_construction_impossible_l977_97772


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l977_97711

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (a ≠ 5 ∧ b ≠ -5) → a + b ≠ 0) ∧
  ¬(∀ a b : ℝ, a + b ≠ 0 → (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l977_97711


namespace NUMINAMATH_CALUDE_arctan_sum_of_roots_l977_97726

theorem arctan_sum_of_roots (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  x₂^2 - x₂ * Real.sin (3 * π / 5) + Real.cos (3 * π / 5) = 0 →
  Real.arctan x₁ + Real.arctan x₂ = π / 5 := by
sorry

end NUMINAMATH_CALUDE_arctan_sum_of_roots_l977_97726


namespace NUMINAMATH_CALUDE_randy_money_left_l977_97776

theorem randy_money_left (initial_amount : ℝ) (lunch_cost : ℝ) (ice_cream_fraction : ℝ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_fraction = 1/4 →
  initial_amount - lunch_cost - (initial_amount - lunch_cost) * ice_cream_fraction = 15 := by
sorry

end NUMINAMATH_CALUDE_randy_money_left_l977_97776


namespace NUMINAMATH_CALUDE_deer_bridge_problem_l977_97728

theorem deer_bridge_problem (y : ℚ) : 
  (3 * (3 * (3 * y - 50) - 50) - 50) * 4 - 50 = 0 ∧ y > 0 → y = 425 / 18 := by
  sorry

end NUMINAMATH_CALUDE_deer_bridge_problem_l977_97728


namespace NUMINAMATH_CALUDE_roots_expression_l977_97733

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 + p*α - 2 = 0)
  (hβ : β^2 + p*β - 2 = 0)
  (hγ : γ^2 + q*γ - 2 = 0)
  (hδ : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (q^2 - p^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l977_97733


namespace NUMINAMATH_CALUDE_non_intercept_line_conditions_l977_97780

/-- A line that cannot be converted to intercept form -/
def NonInterceptLine (m : ℝ) : Prop :=
  ∃ (x y : ℝ), m * (x + y - 1) + (3 * y - 4 * x + 5) = 0 ∧
  ((m - 4 = 0) ∨ (m + 3 = 0) ∨ (-m + 5 = 0))

/-- The theorem stating the conditions for a line that cannot be converted to intercept form -/
theorem non_intercept_line_conditions :
  ∀ m : ℝ, NonInterceptLine m ↔ (m = 4 ∨ m = -3 ∨ m = 5) :=
by sorry

end NUMINAMATH_CALUDE_non_intercept_line_conditions_l977_97780


namespace NUMINAMATH_CALUDE_aluminum_sulfide_weight_l977_97798

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The weight of a given number of moles of a compound -/
def molar_weight (moles molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

theorem aluminum_sulfide_weight :
  let al_weight : ℝ := 26.98
  let s_weight : ℝ := 32.06
  let moles : ℝ := 4
  molar_weight moles (molecular_weight al_weight s_weight) = 600.56 := by
sorry

end NUMINAMATH_CALUDE_aluminum_sulfide_weight_l977_97798


namespace NUMINAMATH_CALUDE_array_sum_mod_1004_l977_97745

/-- Represents the sum of all terms in a 1/q-array as described in the problem -/
def array_sum (q : ℕ) : ℚ :=
  (3 * q^2 : ℚ) / ((3*q - 1) * (q - 1))

/-- The theorem stating that the sum of all terms in a 1/1004-array is congruent to 1 modulo 1004 -/
theorem array_sum_mod_1004 :
  ∃ (n : ℕ), array_sum 1004 = (n * 1004 + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_array_sum_mod_1004_l977_97745


namespace NUMINAMATH_CALUDE_overlapping_strips_area_l977_97742

theorem overlapping_strips_area (left_length right_length total_length : ℝ)
  (left_only_area right_only_area : ℝ) :
  left_length = 9 →
  right_length = 7 →
  total_length = 16 →
  left_length + right_length = total_length →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ),
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlapping_strips_area_l977_97742


namespace NUMINAMATH_CALUDE_price_reduction_equation_l977_97724

/-- Represents the price reduction scenario for a certain type of chip -/
theorem price_reduction_equation (initial_price : ℝ) (final_price : ℝ) (x : ℝ) :
  initial_price = 400 →
  final_price = 144 →
  0 < x →
  x < 1 →
  initial_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l977_97724
