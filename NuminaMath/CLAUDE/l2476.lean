import Mathlib

namespace NUMINAMATH_CALUDE_count_bijections_on_three_element_set_l2476_247600

def S : Finset ℕ := {1, 2, 3}

theorem count_bijections_on_three_element_set :
  Fintype.card { f : S → S | Function.Bijective f } = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_bijections_on_three_element_set_l2476_247600


namespace NUMINAMATH_CALUDE_wire_sharing_l2476_247673

/-- Given a wire of total length 150 cm, where one person's share is 16 cm shorter than the other's,
    prove that the shorter share is 67 cm. -/
theorem wire_sharing (total_length : ℕ) (difference : ℕ) (seokgi_share : ℕ) (yeseul_share : ℕ) :
  total_length = 150 ∧ difference = 16 ∧ seokgi_share + yeseul_share = total_length ∧ 
  yeseul_share = seokgi_share + difference → seokgi_share = 67 :=
by sorry

end NUMINAMATH_CALUDE_wire_sharing_l2476_247673


namespace NUMINAMATH_CALUDE_marigold_fraction_l2476_247614

/-- Represents the composition of flowers in a bouquet -/
structure Bouquet where
  yellow_daisies : ℚ
  white_daisies : ℚ
  yellow_marigolds : ℚ
  white_marigolds : ℚ

/-- The conditions of the flower bouquet problem -/
def bouquet_conditions (b : Bouquet) : Prop :=
  -- Half of the yellow flowers are daisies
  b.yellow_daisies = b.yellow_marigolds ∧
  -- Two-thirds of the white flowers are marigolds
  b.white_marigolds = 2 * b.white_daisies ∧
  -- Four-sevenths of the flowers are yellow
  b.yellow_daisies + b.yellow_marigolds = (4:ℚ)/7 * (b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds) ∧
  -- All fractions are non-negative
  0 ≤ b.yellow_daisies ∧ 0 ≤ b.white_daisies ∧ 0 ≤ b.yellow_marigolds ∧ 0 ≤ b.white_marigolds ∧
  -- The sum of all fractions is 1
  b.yellow_daisies + b.white_daisies + b.yellow_marigolds + b.white_marigolds = 1

/-- The theorem stating that marigolds constitute 4/7 of the flowers -/
theorem marigold_fraction (b : Bouquet) (h : bouquet_conditions b) :
  b.yellow_marigolds + b.white_marigolds = (4:ℚ)/7 := by
  sorry

end NUMINAMATH_CALUDE_marigold_fraction_l2476_247614


namespace NUMINAMATH_CALUDE_second_number_proof_l2476_247687

theorem second_number_proof (N : ℕ) : 
  (N % 144 = 29) → (6215 % 144 = 23) → N = 6365 :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l2476_247687


namespace NUMINAMATH_CALUDE_jason_earnings_l2476_247684

/-- Represents the earnings of a person given their initial and final amounts -/
def earnings (initial final : ℕ) : ℕ := final - initial

theorem jason_earnings :
  let fred_initial : ℕ := 49
  let jason_initial : ℕ := 3
  let fred_final : ℕ := 112
  let jason_final : ℕ := 63
  earnings jason_initial jason_final = 60 := by
sorry

end NUMINAMATH_CALUDE_jason_earnings_l2476_247684


namespace NUMINAMATH_CALUDE_odd_function_extension_l2476_247663

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension {f : ℝ → ℝ} 
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = lg (x + 1)) :
  ∀ x < 0, f x = -lg (-x + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2476_247663


namespace NUMINAMATH_CALUDE_quadratic_properties_l2476_247669

def quadratic_function (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem quadratic_properties :
  (quadratic_function (-1) = 0) ∧
  (∀ x : ℝ, quadratic_function (1 + x) = quadratic_function (1 - x)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2476_247669


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2476_247647

theorem algebraic_expression_value (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a * b = 3) : 
  a * b^2 - a^2 * b = -15 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2476_247647


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l2476_247621

theorem last_two_nonzero_digits_of_80_factorial (n : ℕ) : n = 80 → 
  ∃ k : ℕ, n.factorial = 100 * k + 12 ∧ k % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_last_two_nonzero_digits_of_80_factorial_l2476_247621


namespace NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l2476_247637

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 - 2*a*x + a^2 + a - 2

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x = 0

-- Define the proposition q
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 3

theorem quadratic_roots_and_inequality (a m : ℝ) :
  (¬ p a → a > 2) ∧
  ((∀ m, p a → q m a) ∧ (∃ m, q m a ∧ ¬ p a) → m ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_inequality_l2476_247637


namespace NUMINAMATH_CALUDE_truck_sand_theorem_l2476_247662

/-- The amount of sand remaining on a truck after transit -/
def sand_remaining (initial : ℝ) (lost : ℝ) : ℝ := initial - lost

/-- Theorem: The amount of sand remaining on the truck is 1.7 pounds -/
theorem truck_sand_theorem (initial : ℝ) (lost : ℝ) 
  (h1 : initial = 4.1)
  (h2 : lost = 2.4) : 
  sand_remaining initial lost = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_truck_sand_theorem_l2476_247662


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l2476_247607

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence_solution :
  ∀ x y z : ℝ,
  is_arithmetic_sequence x y z →
  x + y + z = -3 →
  is_geometric_sequence (x + y) (y + z) (z + x) →
  ((x = -1 ∧ y = -1 ∧ z = -1) ∨ (x = -7 ∧ y = -1 ∧ z = 5)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l2476_247607


namespace NUMINAMATH_CALUDE_books_per_shelf_l2476_247618

theorem books_per_shelf
  (mystery_shelves : ℕ)
  (picture_shelves : ℕ)
  (total_books : ℕ)
  (h1 : mystery_shelves = 5)
  (h2 : picture_shelves = 4)
  (h3 : total_books = 54)
  (h4 : total_books % (mystery_shelves + picture_shelves) = 0)  -- Ensures even distribution
  : total_books / (mystery_shelves + picture_shelves) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2476_247618


namespace NUMINAMATH_CALUDE_shirt_ratio_l2476_247643

theorem shirt_ratio : 
  ∀ (steven andrew brian : ℕ),
  steven = 4 * andrew →
  brian = 3 →
  steven = 72 →
  andrew / brian = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_shirt_ratio_l2476_247643


namespace NUMINAMATH_CALUDE_magic_8_ball_theorem_l2476_247630

def magic_8_ball_probability : ℚ := 181440 / 823543

theorem magic_8_ball_theorem (n : ℕ) (k : ℕ) (p : ℚ) :
  n = 7 →
  k = 4 →
  p = 3 / 7 →
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k) = magic_8_ball_probability :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_theorem_l2476_247630


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2476_247677

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4 * Real.sqrt 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 2 * Real.sqrt 2) ∧ (y = 2 * Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2476_247677


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2476_247622

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2476_247622


namespace NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2476_247660

theorem solution_set_linear_inequalities :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_linear_inequalities_l2476_247660


namespace NUMINAMATH_CALUDE_sqrt_four_equals_two_l2476_247636

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_equals_two_l2476_247636


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2476_247648

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2*x < 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {-2, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2476_247648


namespace NUMINAMATH_CALUDE_anna_basketball_score_product_l2476_247634

def first_10_games : List ℕ := [5, 7, 9, 2, 6, 10, 5, 7, 8, 4]

theorem anna_basketball_score_product :
  ∀ (game11 game12 : ℕ),
  game11 < 15 ∧ game12 < 15 →
  (List.sum first_10_games + game11) % 11 = 0 →
  (List.sum first_10_games + game11 + game12) % 12 = 0 →
  game11 * game12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_anna_basketball_score_product_l2476_247634


namespace NUMINAMATH_CALUDE_decimal_conversion_and_addition_l2476_247661

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def binary_to_decimal (b : List Bool) : ℕ :=
  sorry

def binary_add (a b : List Bool) : List Bool :=
  sorry

theorem decimal_conversion_and_addition :
  let binary_45 := decimal_to_binary 45
  let binary_3 := decimal_to_binary 3
  let sum := binary_add binary_45 binary_3
  binary_to_decimal sum = 48 := by
  sorry

end NUMINAMATH_CALUDE_decimal_conversion_and_addition_l2476_247661


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2476_247631

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Real.log (1 + x) ≥ a * x / (1 + x)) →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2476_247631


namespace NUMINAMATH_CALUDE_segment_multiplication_l2476_247681

-- Define a segment as a pair of points
def Segment (α : Type*) := α × α

-- Define the length of a segment
def length {α : Type*} (s : Segment α) : ℝ := sorry

-- Define the multiplication of a segment by a scalar
def scaleSegment {α : Type*} (s : Segment α) (n : ℕ) : Segment α := sorry

-- Theorem statement
theorem segment_multiplication {α : Type*} (AB : Segment α) (n : ℕ) :
  ∃ (AC : Segment α), length AC = n * length AB :=
sorry

end NUMINAMATH_CALUDE_segment_multiplication_l2476_247681


namespace NUMINAMATH_CALUDE_four_digit_sum_l2476_247635

theorem four_digit_sum (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 810 →
  1000 ≤ a * 1000 + b * 100 + c * 10 + d →
  a * 1000 + b * 100 + c * 10 + d < 10000 →
  a + b + c + d = 23 := by
sorry

end NUMINAMATH_CALUDE_four_digit_sum_l2476_247635


namespace NUMINAMATH_CALUDE_percentage_greater_l2476_247670

theorem percentage_greater (A B : ℝ) (y : ℝ) (h1 : A > B) (h2 : B > 0) : 
  let C := A + B
  y = 100 * ((C - B) / B) → y = 100 * (A / B) := by
sorry

end NUMINAMATH_CALUDE_percentage_greater_l2476_247670


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l2476_247685

/-- Given a triangle ABC with side lengths, prove that the perimeter of the inner triangle
    formed by lines parallel to each side is equal to the length of side AB. -/
theorem inner_triangle_perimeter (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_x_lt_c : x < c) (h_y_lt_a : y < a) (h_z_lt_b : z < b)
  (h_prop_x : x / c = (c - x) / a) (h_prop_y : y / a = (a - y) / b) (h_prop_z : z / b = (b - z) / c) :
  x / c * a + y / a * b + z / b * c = a := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l2476_247685


namespace NUMINAMATH_CALUDE_exists_face_sum_gt_25_l2476_247642

/-- Represents a cube with labeled edges -/
structure LabeledCube where
  edges : Fin 12 → ℕ
  edge_sum : ∀ i : Fin 12, edges i ∈ Finset.range 13 \ {0}

/-- Represents a face of the cube -/
def Face := Fin 4 → Fin 12

/-- The sum of the numbers on the edges of a face -/
def face_sum (c : LabeledCube) (f : Face) : ℕ :=
  (Finset.range 4).sum (λ i => c.edges (f i))

/-- Theorem: There exists a face with sum greater than 25 -/
theorem exists_face_sum_gt_25 (c : LabeledCube) : 
  ∃ f : Face, face_sum c f > 25 := by
  sorry


end NUMINAMATH_CALUDE_exists_face_sum_gt_25_l2476_247642


namespace NUMINAMATH_CALUDE_minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l2476_247620

/-- Given n sticks of lengths 1, 2, ..., n, this function returns the minimum number
    of sticks that need to be broken in half to form a square. If it's possible to
    form a square without breaking any sticks, it returns 0. -/
def minBrokenSticks (n : ℕ) : ℕ :=
  if n = 12 then 2
  else if n = 15 then 0
  else sorry

theorem minBrokenSticks_correct :
  (minBrokenSticks 12 = 2) ∧ (minBrokenSticks 15 = 0) := by sorry

/-- Function to check if it's possible to form a square from n sticks of lengths 1, 2, ..., n
    without breaking any sticks -/
def canFormSquare (n : ℕ) : Prop :=
  ∃ (a b c d : List ℕ), 
    (a ++ b ++ c ++ d).sum = n * (n + 1) / 2 ∧
    (∀ x ∈ a ++ b ++ c ++ d, x ≤ n) ∧
    a.sum = b.sum ∧ b.sum = c.sum ∧ c.sum = d.sum

theorem canFormSquare_15 : canFormSquare 15 := by sorry

theorem not_canFormSquare_12 : ¬ canFormSquare 12 := by sorry

end NUMINAMATH_CALUDE_minBrokenSticks_correct_canFormSquare_15_not_canFormSquare_12_l2476_247620


namespace NUMINAMATH_CALUDE_set_M_properties_l2476_247694

-- Define the set M
variable (M : Set ℝ)

-- Define the properties of M
variable (h_nonempty : M.Nonempty)
variable (h_two : 2 ∈ M)
variable (h_diff : ∀ x y, x ∈ M → y ∈ M → x - y ∈ M)

-- Theorem statement
theorem set_M_properties :
  (0 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → x + y ∈ M) ∧
  (∀ x, x ∈ M → x ≠ 0 → x ≠ 1 → (1 / (x * (x - 1))) ∈ M) :=
sorry

end NUMINAMATH_CALUDE_set_M_properties_l2476_247694


namespace NUMINAMATH_CALUDE_circumcircle_diameter_l2476_247699

theorem circumcircle_diameter (a b c : ℝ) (θ : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : Real.cos θ = 1/3) :
  let d := max a (max b c)
  2 * d / Real.sin θ = 9 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_diameter_l2476_247699


namespace NUMINAMATH_CALUDE_solution_set_eq_neg_one_one_l2476_247609

-- Define the solution set of x^2 - 1 = 0
def solution_set : Set ℝ := {x : ℝ | x^2 - 1 = 0}

-- Theorem stating that the solution set is exactly {-1, 1}
theorem solution_set_eq_neg_one_one : solution_set = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_neg_one_one_l2476_247609


namespace NUMINAMATH_CALUDE_ones_digit_sum_powers_2011_l2476_247605

theorem ones_digit_sum_powers_2011 : ∃ n : ℕ, n < 10 ∧ (1^2011 + 2^2011 + 3^2011 + 4^2011 + 5^2011 + 6^2011 + 7^2011 + 8^2011 + 9^2011 + 10^2011) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_sum_powers_2011_l2476_247605


namespace NUMINAMATH_CALUDE_symmetry_oyz_coordinates_l2476_247659

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The Oyz plane -/
def Oyz : Set Point3D :=
  {p : Point3D | p.x = 0}

/-- Symmetry with respect to the Oyz plane -/
def symmetricOyz (a b : Point3D) : Prop :=
  b.x = -a.x ∧ b.y = a.y ∧ b.z = a.z

theorem symmetry_oyz_coordinates :
  let a : Point3D := ⟨3, 4, 5⟩
  let b : Point3D := ⟨-3, 4, 5⟩
  symmetricOyz a b := by sorry

end NUMINAMATH_CALUDE_symmetry_oyz_coordinates_l2476_247659


namespace NUMINAMATH_CALUDE_parabola_intercept_problem_l2476_247675

/-- Given two parabolas with specific properties, prove that h = 36 -/
theorem parabola_intercept_problem :
  ∀ (h j k : ℤ),
  (∀ x, ∃ y, y = 3 * (x - h)^2 + j) →
  (∀ x, ∃ y, y = 2 * (x - h)^2 + k) →
  (3 * h^2 + j = 2013) →
  (2 * h^2 + k = 2014) →
  (∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ 3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0) →
  (∃ x3 x4 : ℤ, x3 > 0 ∧ x4 > 0 ∧ 2 * (x3 - h)^2 + k = 0 ∧ 2 * (x4 - h)^2 + k = 0) →
  h = 36 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intercept_problem_l2476_247675


namespace NUMINAMATH_CALUDE_max_profit_at_18_profit_maximized_at_18_l2476_247690

-- Define the profit function
def profit (x : ℝ) : ℝ := -0.5 * x^2 + 18 * x - 20

-- Theorem statement
theorem max_profit_at_18 :
  ∃ (x_max : ℝ), x_max > 0 ∧ 
  (∀ (x : ℝ), x > 0 → profit x ≤ profit x_max) ∧
  x_max = 18 ∧ profit x_max = 142 := by
  sorry

-- Additional theorem to show that 18 is indeed the maximizer
theorem profit_maximized_at_18 :
  ∀ (x : ℝ), x > 0 → profit x ≤ profit 18 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_at_18_profit_maximized_at_18_l2476_247690


namespace NUMINAMATH_CALUDE_small_pizza_slices_l2476_247674

/-- The number of large pizzas --/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas --/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a large pizza --/
def slices_per_large_pizza : ℕ := 16

/-- The total number of slices eaten --/
def total_slices_eaten : ℕ := 48

/-- Theorem: The number of slices in a small pizza is 8 --/
theorem small_pizza_slices : 
  ∃ (slices_per_small_pizza : ℕ), 
    slices_per_small_pizza * num_small_pizzas + 
    slices_per_large_pizza * num_large_pizzas = total_slices_eaten ∧
    slices_per_small_pizza = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_slices_l2476_247674


namespace NUMINAMATH_CALUDE_expected_value_is_four_thirds_l2476_247626

/-- The expected value of a biased coin flip --/
def expected_value_biased_coin : ℚ :=
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let value_heads : ℤ := 5
  let value_tails : ℤ := -6
  p_heads * value_heads + p_tails * value_tails

/-- Theorem: The expected value of the biased coin flip is 4/3 --/
theorem expected_value_is_four_thirds :
  expected_value_biased_coin = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_four_thirds_l2476_247626


namespace NUMINAMATH_CALUDE_square_ending_theorem_l2476_247638

theorem square_ending_theorem (n : ℤ) :
  (∀ d : ℕ, d ∈ Finset.range 9 → (n^2 : ℤ) % 10000 ≠ d * 1111) ∧
  ((∃ d : ℕ, d ∈ Finset.range 9 ∧ (n^2 : ℤ) % 1000 = d * 111) → (n^2 : ℤ) % 1000 = 444) :=
by sorry

end NUMINAMATH_CALUDE_square_ending_theorem_l2476_247638


namespace NUMINAMATH_CALUDE_loan_repayment_months_l2476_247658

/-- Represents the monthly income in ten thousands of yuan -/
def monthlyIncome : ℕ → ℚ
  | 0 => 20  -- First month's income
  | n + 1 => if n < 5 then monthlyIncome n * 1.2 else monthlyIncome n + 2

/-- Calculates the cumulative income up to month n -/
def cumulativeIncome (n : ℕ) : ℚ :=
  (List.range n).map monthlyIncome |>.sum

/-- The loan amount in ten thousands of yuan -/
def loanAmount : ℚ := 400

theorem loan_repayment_months :
  (∀ k < 10, cumulativeIncome k < loanAmount) ∧
  cumulativeIncome 10 ≥ loanAmount := by
  sorry

#eval cumulativeIncome 10  -- For verification

end NUMINAMATH_CALUDE_loan_repayment_months_l2476_247658


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2476_247692

theorem expression_simplification_and_evaluation :
  let x : ℤ := -3
  let y : ℤ := -2
  3 * x^2 * y - (2 * x^2 * y - (2 * x * y - x^2 * y) - 4 * x^2 * y) - x * y = -66 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2476_247692


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2476_247696

def hyperbola (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 / n = 1

def tangent_line (m n : ℝ) (x y : ℝ) : Prop :=
  2 * m * x - n * y + 2 = 0

def asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

theorem hyperbola_asymptotes (m n : ℝ) :
  (∀ x y, hyperbola m n x y) →
  (∀ x y, tangent_line m n x y) →
  (∀ x y, asymptote (Real.sqrt 2) x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2476_247696


namespace NUMINAMATH_CALUDE_smallest_four_divisors_sum_of_squares_l2476_247615

theorem smallest_four_divisors_sum_of_squares (n : ℕ+) 
  (d1 d2 d3 d4 : ℕ+) 
  (h_div : ∀ m : ℕ+, m ∣ n → m ≥ d1 ∧ m ≥ d2 ∧ m ≥ d3 ∧ m ≥ d4)
  (h_order : d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  (h_sum : n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_smallest_four_divisors_sum_of_squares_l2476_247615


namespace NUMINAMATH_CALUDE_stock_worth_l2476_247693

theorem stock_worth (X : ℝ) : 
  (0.1 * X * 1.2 + 0.9 * X * 0.95 = X - 400) → X = 16000 := by sorry

end NUMINAMATH_CALUDE_stock_worth_l2476_247693


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2476_247667

theorem polynomial_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2476_247667


namespace NUMINAMATH_CALUDE_triangle_side_difference_bound_l2476_247612

/-- Given a triangle ABC with side lengths a, b, c and corresponding opposite angles A, B, C,
    prove that if a = 1 and C - B = π/2, then √2/2 < c - b < 1 -/
theorem triangle_side_difference_bound (a b c A B C : Real) : 
  a = 1 → 
  C - B = π / 2 → 
  0 < A ∧ 0 < B ∧ 0 < C → 
  A + B + C = π → 
  a / (Real.sin A) = b / (Real.sin B) → 
  a / (Real.sin A) = c / (Real.sin C) → 
  Real.sqrt 2 / 2 < c - b ∧ c - b < 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_bound_l2476_247612


namespace NUMINAMATH_CALUDE_amy_picture_files_l2476_247644

theorem amy_picture_files (music_files : ℝ) (video_files : ℝ) (total_files : ℕ) : 
  music_files = 4.0 →
  video_files = 21.0 →
  total_files = 48 →
  (total_files : ℝ) - (music_files + video_files) = 23 := by
sorry

end NUMINAMATH_CALUDE_amy_picture_files_l2476_247644


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2476_247610

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : 
  x^3 - 1/x^3 = 125 := by sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l2476_247610


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l2476_247639

/-- A quadratic function with no real roots has a coefficient greater than 1 -/
theorem quadratic_no_roots (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l2476_247639


namespace NUMINAMATH_CALUDE_square_area_ratio_l2476_247682

theorem square_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := Real.sqrt ((4 / 5) * r ^ 2)
  let s2 := Real.sqrt (2 * r ^ 2)
  (s1 ^ 2) / (s2 ^ 2) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2476_247682


namespace NUMINAMATH_CALUDE_ten_team_round_robin_l2476_247604

def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

theorem ten_team_round_robin :
  roundRobinGames 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_ten_team_round_robin_l2476_247604


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2476_247654

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 10

theorem circle_center_coordinates :
  CircleCenter 2 1 := by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2476_247654


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2476_247627

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def has_seven_consecutive_nonprimes (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, i ≥ k ∧ i < k + 7 → ¬(is_prime i)

theorem smallest_prime_after_seven_nonprimes :
  (is_prime 97) ∧ 
  (has_seven_consecutive_nonprimes 90) ∧
  (∀ p : ℕ, p < 97 → ¬(is_prime p ∧ has_seven_consecutive_nonprimes (p - 7))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l2476_247627


namespace NUMINAMATH_CALUDE_tile_count_equivalence_l2476_247656

theorem tile_count_equivalence (area : ℝ) : 
  area = (0.3 : ℝ)^2 * 720 → area = (0.4 : ℝ)^2 * 405 := by
  sorry

end NUMINAMATH_CALUDE_tile_count_equivalence_l2476_247656


namespace NUMINAMATH_CALUDE_alexis_isabella_shopping_ratio_l2476_247676

theorem alexis_isabella_shopping_ratio : 
  let alexis_pants : ℕ := 21
  let alexis_dresses : ℕ := 18
  let isabella_total : ℕ := 13
  (alexis_pants + alexis_dresses) / isabella_total = 3 :=
by sorry

end NUMINAMATH_CALUDE_alexis_isabella_shopping_ratio_l2476_247676


namespace NUMINAMATH_CALUDE_correct_calculation_of_one_fifth_sum_of_acute_angles_l2476_247683

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

-- Theorem statement
theorem correct_calculation_of_one_fifth_sum_of_acute_angles 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  18 < (1/5) * (α + β) ∧ 
  (1/5) * (α + β) < 54 ∧ 
  (42 ∈ {17, 42, 56, 73} ∩ Set.Icc 18 54) ∧ 
  ({17, 42, 56, 73} ∩ Set.Icc 18 54 = {42}) :=
sorry

end NUMINAMATH_CALUDE_correct_calculation_of_one_fifth_sum_of_acute_angles_l2476_247683


namespace NUMINAMATH_CALUDE_root_sum_relation_l2476_247691

/-- The polynomial x^3 - 4x^2 + 7x - 10 -/
def p (x : ℝ) : ℝ := x^3 - 4*x^2 + 7*x - 10

/-- The sum of the k-th powers of the roots of p -/
def t (k : ℕ) : ℝ := sorry

theorem root_sum_relation :
  ∃ (u v w : ℝ), p u = 0 ∧ p v = 0 ∧ p w = 0 ∧
  (∀ k, t k = u^k + v^k + w^k) ∧
  t 0 = 3 ∧ t 1 = 4 ∧ t 2 = 10 ∧
  (∃ (d e f : ℝ), ∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) →
  ∃ (d e f : ℝ), (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2)) ∧ d + e + f = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_relation_l2476_247691


namespace NUMINAMATH_CALUDE_rectangle_area_is_9000_l2476_247603

/-- A rectangle WXYZ with given coordinates -/
structure Rectangle where
  W : ℝ × ℝ
  X : ℝ × ℝ
  Z : ℝ × ℤ

/-- The area of a rectangle WXYZ -/
def rectangleArea (r : Rectangle) : ℝ := sorry

/-- The theorem stating that the area of the given rectangle is 9000 -/
theorem rectangle_area_is_9000 (r : Rectangle) 
  (h1 : r.W = (2, 3))
  (h2 : r.X = (302, 23))
  (h3 : r.Z.1 = 4) :
  rectangleArea r = 9000 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_is_9000_l2476_247603


namespace NUMINAMATH_CALUDE_triangle_relation_angle_C_measure_l2476_247646

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

theorem triangle_relation (t : Triangle)
  (h : Real.sin (2 * t.A + t.B) / Real.sin t.A = 2 + 2 * Real.cos (t.A + t.B)) :
  t.b = 2 * t.a := by sorry

theorem angle_C_measure (t : Triangle)
  (h1 : t.b = 2 * t.a)
  (h2 : t.c = Real.sqrt 7 * t.a) :
  t.C = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_relation_angle_C_measure_l2476_247646


namespace NUMINAMATH_CALUDE_planes_perpendicular_from_lines_l2476_247613

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the perpendicular relation between planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_from_lines
  (α β : Plane) (a b : Line)
  (h1 : perpendicular_line_plane a α)
  (h2 : perpendicular_line_plane b β)
  (h3 : perpendicular_line_line a b) :
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_from_lines_l2476_247613


namespace NUMINAMATH_CALUDE_bad_carrots_count_l2476_247623

/-- The number of bad carrots in Vanessa's garden -/
def bad_carrots (vanessa_carrots mother_carrots good_carrots : ℕ) : ℕ :=
  vanessa_carrots + mother_carrots - good_carrots

theorem bad_carrots_count : bad_carrots 17 14 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l2476_247623


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2476_247641

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7*x

-- Theorem statement
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 582 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2476_247641


namespace NUMINAMATH_CALUDE_smallest_positive_integer_to_multiple_of_five_l2476_247665

theorem smallest_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (725 + m) % 5 = 0 → m ≥ n) ∧ (725 + n) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_to_multiple_of_five_l2476_247665


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2476_247619

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 1 → (1 / (x - 1) ≥ -1 ↔ x ∈ Set.Ici 1 ∪ Set.Iic 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2476_247619


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2476_247657

theorem cubic_polynomial_integer_root 
  (d e : ℚ) 
  (h1 : ∃ x : ℝ, x^3 + d*x + e = 0 ∧ x = 2 - Real.sqrt 5)
  (h2 : ∃ n : ℤ, n^3 + d*n + e = 0) :
  ∃ n : ℤ, n^3 + d*n + e = 0 ∧ n = -4 :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l2476_247657


namespace NUMINAMATH_CALUDE_second_third_smallest_average_l2476_247640

theorem second_third_smallest_average (a b c d e : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- five different positive integers
  (a + b + c + d + e : ℚ) / 5 = 5 ∧  -- average is 5
  ∀ x y z w v : ℕ+, x < y ∧ y < z ∧ z < w ∧ w < v → 
    (x + y + z + w + v : ℚ) / 5 = 5 → (v - x : ℚ) ≤ (e - a) →  -- difference is maximized
  (b + c : ℚ) / 2 = 5/2 :=  -- average of second and third smallest is 2.5
sorry

end NUMINAMATH_CALUDE_second_third_smallest_average_l2476_247640


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2476_247689

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The terms a_2, (1/2)a_3, a_1 form an arithmetic sequence. -/
def ArithmeticSubsequence (a : ℕ → ℝ) : Prop :=
  a 2 - (1/2 * a 3) = (1/2 * a 3) - a 1

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSubsequence a →
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2476_247689


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l2476_247645

-- Part 1
theorem simplify_sqrt_fraction : (1/2) * Real.sqrt (4/7) = Real.sqrt 7 / 7 := by sorry

-- Part 2
theorem simplify_sqrt_difference_squares : Real.sqrt (20^2 - 15^2) = 5 * Real.sqrt 7 := by sorry

-- Part 3
theorem simplify_sqrt_fraction_product : Real.sqrt ((32 * 9) / 25) = (12 * Real.sqrt 2) / 5 := by sorry

-- Part 4
theorem simplify_sqrt_decimal : Real.sqrt 22.5 = (3 * Real.sqrt 10) / 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_simplify_sqrt_difference_squares_simplify_sqrt_fraction_product_simplify_sqrt_decimal_l2476_247645


namespace NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_three_l2476_247617

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), c ≠ 0 ∧ a.1 = c * b.1 ∧ a.2 = c * b.2

/-- Given two vectors a and b, where a depends on k, prove that k = 3 when a and b are parallel -/
theorem vector_parallel_implies_k_equals_three (k : ℝ) :
  let a : ℝ × ℝ := (2 - k, 3)
  let b : ℝ × ℝ := (2, -6)
  parallel a b → k = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_parallel_implies_k_equals_three_l2476_247617


namespace NUMINAMATH_CALUDE_only_first_statement_true_l2476_247602

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- Axioms for parallel and perpendicular relations
axiom parallel_transitive {l1 l2 l3 : Line} : parallel l1 l2 → parallel l2 l3 → parallel l1 l3
axiom perpendicular_not_parallel {l1 l2 : Line} : perpendicular l1 l2 → ¬ parallel l1 l2
axiom plane_perpendicular_not_parallel {p1 p2 : Plane} : plane_perpendicular p1 p2 → ¬ plane_parallel p1 p2

-- The main theorem
theorem only_first_statement_true 
  (a b c : Line) (α β γ : Plane) 
  (h_distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (parallel a b ∧ parallel b c → parallel a c) ∧
  ¬(perpendicular a b ∧ perpendicular b c → parallel a c) ∧
  ¬(plane_perpendicular α β ∧ plane_perpendicular β γ → plane_parallel α γ) ∧
  ¬(plane_perpendicular α β ∧ plane_intersection α β = a ∧ perpendicular b a → line_perpendicular_to_plane b β) :=
sorry

end NUMINAMATH_CALUDE_only_first_statement_true_l2476_247602


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2476_247653

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2476_247653


namespace NUMINAMATH_CALUDE_factorization_equality_l2476_247625

theorem factorization_equality (x y : ℝ) :
  (2*x - y) * (x + 3*y) - (2*x + 3*y) * (y - 2*x) = 3 * (2*x - y) * (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2476_247625


namespace NUMINAMATH_CALUDE_scientific_notation_of_44_3_million_l2476_247629

theorem scientific_notation_of_44_3_million : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_44_3_million_l2476_247629


namespace NUMINAMATH_CALUDE_board_number_is_91_l2476_247671

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def does_not_contain_seven (n : ℕ) : Prop :=
  ¬ (∃ d, d ∈ n.digits 10 ∧ d = 7)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem board_number_is_91 
  (n : ℕ) 
  (x : ℕ) 
  (h_consecutive : ∀ i < n, is_two_digit (x / 10^i % 100))
  (h_descending : ∀ i < n - 1, x / 10^i % 100 > x / 10^(i+1) % 100)
  (h_last_digit : does_not_contain_seven (x % 100))
  (h_prime_factors : ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ x = p * q ∧ q = p + 4) :
  x = 91 :=
sorry

end NUMINAMATH_CALUDE_board_number_is_91_l2476_247671


namespace NUMINAMATH_CALUDE_max_weighing_ways_exists_89_ways_l2476_247655

/-- Represents the set of weights as powers of 2 up to 2^9 (512) -/
def weights : Finset ℕ := Finset.range 10

/-- The number of ways a weight P can be measured using weights up to 2^n -/
def K (n : ℕ) (P : ℕ) : ℕ := sorry

/-- The maximum number of ways any weight can be measured using weights up to 2^n -/
def K_max (n : ℕ) : ℕ := sorry

/-- Theorem stating that no load can be weighed in more than 89 different ways -/
theorem max_weighing_ways : K_max 9 ≤ 89 := sorry

/-- Theorem stating that there exists a load that can be weighed in exactly 89 different ways -/
theorem exists_89_ways : ∃ P : ℕ, K 9 P = 89 := sorry

end NUMINAMATH_CALUDE_max_weighing_ways_exists_89_ways_l2476_247655


namespace NUMINAMATH_CALUDE_division_problem_l2476_247666

theorem division_problem (x : ℝ) : 
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2476_247666


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2476_247608

def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2476_247608


namespace NUMINAMATH_CALUDE_fraction_simplification_l2476_247688

theorem fraction_simplification : 3 / (2 - 3/4) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2476_247688


namespace NUMINAMATH_CALUDE_chewing_gum_price_l2476_247650

def currency_denominations : List Nat := [1, 5, 10, 20, 50, 100]

def is_valid_payment (price : Nat) (payment1 payment2 : Nat) : Prop :=
  payment1 > price ∧ payment2 > price ∧
  ∃ (exchange : Nat), exchange ≤ payment1 ∧ exchange ≤ payment2 ∧
    payment1 - exchange + (payment2 - price) = price ∧
    payment2 - (payment2 - price) + exchange = price

def exists_valid_payments (price : Nat) : Prop :=
  ∃ (payment1 payment2 : Nat),
    payment1 ∈ currency_denominations ∧
    payment2 ∈ currency_denominations ∧
    is_valid_payment price payment1 payment2

theorem chewing_gum_price :
  ¬ exists_valid_payments 2 ∧
  ¬ exists_valid_payments 6 ∧
  ¬ exists_valid_payments 7 ∧
  exists_valid_payments 8 :=
by sorry

end NUMINAMATH_CALUDE_chewing_gum_price_l2476_247650


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_proof_l2476_247698

/-- The smallest number of eggs given the conditions -/
def smallest_number_of_eggs : ℕ := 137

/-- The number of containers with 9 eggs -/
def containers_with_nine : ℕ := 3

/-- The capacity of a full container -/
def container_capacity : ℕ := 10

theorem smallest_number_of_eggs_proof :
  ∀ n : ℕ,
  n > 130 ∧
  n = container_capacity * (n / container_capacity) - containers_with_nine →
  n ≥ smallest_number_of_eggs :=
by
  sorry

#check smallest_number_of_eggs_proof

end NUMINAMATH_CALUDE_smallest_number_of_eggs_proof_l2476_247698


namespace NUMINAMATH_CALUDE_divisors_of_eight_factorial_greater_than_seven_factorial_l2476_247606

theorem divisors_of_eight_factorial_greater_than_seven_factorial :
  (Finset.filter (fun d => d > Nat.factorial 7 ∧ Nat.factorial 8 % d = 0) (Finset.range (Nat.factorial 8 + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_eight_factorial_greater_than_seven_factorial_l2476_247606


namespace NUMINAMATH_CALUDE_banquet_solution_l2476_247632

def banquet_problem (total_attendees : ℕ) (resident_price : ℚ) (non_resident_price : ℚ) (total_revenue : ℚ) : Prop :=
  ∃ (residents : ℕ),
    residents ≤ total_attendees ∧
    residents * resident_price + (total_attendees - residents) * non_resident_price = total_revenue

theorem banquet_solution :
  banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) →
  ∃ (residents : ℕ), residents = 220 ∧ banquet_problem 586 (12.95 : ℚ) (17.95 : ℚ) (9423.70 : ℚ) :=
by
  sorry

#check banquet_solution

end NUMINAMATH_CALUDE_banquet_solution_l2476_247632


namespace NUMINAMATH_CALUDE_fish_tagging_problem_l2476_247601

/-- The number of fish initially tagged in a pond -/
def initially_tagged (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ) : ℕ :=
  (tagged_in_catch * total_fish) / catch_size

theorem fish_tagging_problem (total_fish : ℕ) (catch_size : ℕ) (tagged_in_catch : ℕ)
  (h1 : total_fish = 1500)
  (h2 : catch_size = 50)
  (h3 : tagged_in_catch = 2) :
  initially_tagged total_fish catch_size tagged_in_catch = 60 := by
  sorry

end NUMINAMATH_CALUDE_fish_tagging_problem_l2476_247601


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l2476_247628

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : sumOfDigitsUpTo 5000 = 229450 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_5000_l2476_247628


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l2476_247679

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y ≥ 0 ∧ y^2 = 4 → y = x :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_four_l2476_247679


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l2476_247616

theorem percentage_equation_solution : 
  ∃ x : ℝ, 45 * x = (35 / 100) * 900 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l2476_247616


namespace NUMINAMATH_CALUDE_christinas_age_problem_l2476_247680

theorem christinas_age_problem (C : ℝ) (Y : ℝ) :
  (C + 5 = Y / 2) →
  (21 = (3 / 5) * C) →
  Y = 80 := by
sorry

end NUMINAMATH_CALUDE_christinas_age_problem_l2476_247680


namespace NUMINAMATH_CALUDE_square_root_five_expansion_l2476_247668

theorem square_root_five_expansion 
  (a b m n : ℤ) 
  (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) : 
  a = m^2 + 5*n^2 ∧ b = 2*m*n := by
sorry

end NUMINAMATH_CALUDE_square_root_five_expansion_l2476_247668


namespace NUMINAMATH_CALUDE_expression_simplification_l2476_247678

theorem expression_simplification (m : ℝ) (h : m^2 + 3*m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6*m) / (m + 2 - 5 / (m - 2)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2476_247678


namespace NUMINAMATH_CALUDE_daves_tiling_area_l2476_247697

theorem daves_tiling_area (total_area : ℝ) (clara_ratio : ℕ) (dave_ratio : ℕ) 
  (h1 : total_area = 330)
  (h2 : clara_ratio = 4)
  (h3 : dave_ratio = 7) : 
  (dave_ratio : ℝ) / ((clara_ratio : ℝ) + (dave_ratio : ℝ)) * total_area = 210 :=
by sorry

end NUMINAMATH_CALUDE_daves_tiling_area_l2476_247697


namespace NUMINAMATH_CALUDE_spider_journey_l2476_247664

theorem spider_journey (r : ℝ) (final_leg : ℝ) : r = 75 ∧ final_leg = 90 →
  2 * r + r + final_leg = 315 := by
  sorry

end NUMINAMATH_CALUDE_spider_journey_l2476_247664


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l2476_247686

/-- 
Given a man's upstream and downstream rowing speeds, 
calculate his speed in still water.
-/
theorem mans_speed_in_still_water 
  (upstream_speed downstream_speed : ℝ) 
  (h1 : upstream_speed = 20) 
  (h2 : downstream_speed = 60) : 
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

#check mans_speed_in_still_water

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l2476_247686


namespace NUMINAMATH_CALUDE_positive_integer_sum_with_square_is_thirty_l2476_247611

theorem positive_integer_sum_with_square_is_thirty (P : ℕ+) : P^2 + P = 30 → P = 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_sum_with_square_is_thirty_l2476_247611


namespace NUMINAMATH_CALUDE_toy_store_optimization_l2476_247633

/-- Toy store profit optimization problem --/
theorem toy_store_optimization :
  let initial_price : ℝ := 120
  let initial_cost : ℝ := 80
  let initial_sales : ℝ := 20
  let price_reduction (x : ℝ) := x
  let sales_increase (x : ℝ) := 2 * x
  let new_price (x : ℝ) := initial_price - price_reduction x
  let new_sales (x : ℝ) := initial_sales + sales_increase x
  let profit (x : ℝ) := (new_price x - initial_cost) * new_sales x

  -- Daily sales function
  ∀ x, new_sales x = 20 + 2*x ∧

  -- Profit function and domain
  (∀ x, profit x = -2*x^2 + 60*x + 800) ∧
  (∀ x, 0 < x → x ≤ 40 → new_price x ≥ initial_cost) ∧

  -- Maximum profit
  ∃ x, 0 < x ∧ x ≤ 40 ∧ 
    profit x = 1250 ∧
    (∀ y, 0 < y → y ≤ 40 → profit y ≤ profit x) ∧
    new_price x = 105 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_optimization_l2476_247633


namespace NUMINAMATH_CALUDE_pencil_cost_l2476_247672

/-- The cost of a pencil given total money and number of pencils that can be bought --/
theorem pencil_cost (total_money : ℚ) (num_pencils : ℕ) (h : total_money = 50 ∧ num_pencils = 10) : 
  total_money / num_pencils = 5 := by
  sorry

#check pencil_cost

end NUMINAMATH_CALUDE_pencil_cost_l2476_247672


namespace NUMINAMATH_CALUDE_unique_abcabc_cube_minus_square_l2476_247652

/-- A number is of the form abcabc if it equals 1001 * (100a + 10b + c) for some digits a, b, c -/
def is_abcabc (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 1001 * (100 * a + 10 * b + c)

/-- The main theorem stating that 78 is the unique positive integer x 
    such that x^3 - x^2 is a six-digit number of the form abcabc -/
theorem unique_abcabc_cube_minus_square :
  ∃! (x : ℕ), x > 0 ∧ 100000 ≤ x^3 - x^2 ∧ x^3 - x^2 < 1000000 ∧ is_abcabc (x^3 - x^2) ∧ x = 78 :=
sorry

end NUMINAMATH_CALUDE_unique_abcabc_cube_minus_square_l2476_247652


namespace NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l2476_247649

-- Define the sets A and B
def A : Set ℝ := {x | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem for part (1)
theorem subset_condition (a : ℝ) : B a ⊆ A → a ∈ Set.Iic 3 := by sorry

-- Theorem for part (2)
theorem nonempty_intersection (a : ℝ) : (A ∩ B a).Nonempty → a ∈ Set.Ioi (5/2) := by sorry

end NUMINAMATH_CALUDE_subset_condition_nonempty_intersection_l2476_247649


namespace NUMINAMATH_CALUDE_divisibility_problem_l2476_247695

theorem divisibility_problem :
  ∃ k : ℕ, (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) = k * (2^4 * 5^7 * 2003) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2476_247695


namespace NUMINAMATH_CALUDE_solution_implies_k_value_l2476_247624

theorem solution_implies_k_value (x y k : ℝ) :
  x = -3 → y = 2 → 2 * x + k * y = 0 → k = 3 := by sorry

end NUMINAMATH_CALUDE_solution_implies_k_value_l2476_247624


namespace NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l2476_247651

theorem consecutive_squares_not_equal_consecutive_fourth_powers :
  ∀ x y : ℕ+, x^2 + (x + 1)^2 ≠ y^4 + (y + 1)^4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_squares_not_equal_consecutive_fourth_powers_l2476_247651
