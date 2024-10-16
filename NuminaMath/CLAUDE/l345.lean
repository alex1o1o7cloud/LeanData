import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l345_34550

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the function f
def f (M : ℝ × ℝ) (ABC : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define the theorem
theorem min_value_theorem (ABC : Triangle) :
  let ⟨A, B, C⟩ := ABC
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 2 * Real.sqrt 3 →
  Real.cos (Real.arccos ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / 
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))) = Real.cos (π/6) →
  ∀ M : ℝ × ℝ, 
    let (m, n, p) := f M ABC
    m = 1/2 →
    (∃ x y : ℝ, n = x ∧ p = y ∧ 
      (∀ x' y' : ℝ, 1/x' + 4/y' ≥ 1/x + 4/y → x' = x ∧ y' = y) →
      1/x + 4/y = 18) := by sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_min_value_theorem_l345_34550


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l345_34506

/-- 
If 3x^2 - 7x + m = 0 is a quadratic equation with exactly one solution for x, 
then m = 49/12.
-/
theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℚ, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l345_34506


namespace NUMINAMATH_CALUDE_triangle_existence_l345_34556

-- Define the basic types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the function to check if a line cuts off equal segments from a point on two other lines
def cutsEqualSegments (l : Line) (A : Point) (AB AC : Line) : Prop :=
  ∃ (P Q : Point), isPointOnLine P AB ∧ isPointOnLine Q AC ∧
    isPointOnLine P l ∧ isPointOnLine Q l ∧
    (P.x - A.x)^2 + (P.y - A.y)^2 = (Q.x - A.x)^2 + (Q.y - A.y)^2

-- State the theorem
theorem triangle_existence 
  (A O : Point) 
  (l : Line) 
  (h_euler : isPointOnLine O l)
  (h_equal_segments : ∃ (AB AC : Line), cutsEqualSegments l A AB AC) :
  ∃ (T : Triangle), T.A = A ∧ 
    isPointOnLine T.B l ∧ isPointOnLine T.C l ∧
    (T.B.x - O.x)^2 + (T.B.y - O.y)^2 = (T.C.x - O.x)^2 + (T.C.y - O.y)^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l345_34556


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l345_34560

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 → b = 21 → h^2 = a^2 + b^2 → h = Real.sqrt 666 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l345_34560


namespace NUMINAMATH_CALUDE_strawberry_problem_l345_34597

theorem strawberry_problem (initial : Float) (eaten : Float) (remaining : Float) : 
  initial = 78.0 → eaten = 42.0 → remaining = initial - eaten → remaining = 36.0 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_problem_l345_34597


namespace NUMINAMATH_CALUDE_volume_ratio_cube_sphere_l345_34505

/-- The ratio of the volume of a cube to the volume of a sphere -/
theorem volume_ratio_cube_sphere (cube_edge : Real) (other_cube_edge : Real) : 
  cube_edge = 4 → other_cube_edge = 3 →
  (cube_edge ^ 3) / ((4/3) * π * (other_cube_edge ^ 3)) = 16 / (9 * π) := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cube_sphere_l345_34505


namespace NUMINAMATH_CALUDE_weight_gain_proof_l345_34579

/-- Calculates the final weight after muscle and fat gain -/
def final_weight (initial_weight : ℝ) (muscle_gain_percent : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percent
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that given the specified conditions, the final weight is 150 kg -/
theorem weight_gain_proof :
  final_weight 120 0.2 0.25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_weight_gain_proof_l345_34579


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l345_34535

/-- Given two concentric circles x and y, if the probability of a randomly selected point 
    inside circle x being outside circle y is 0.9722222222222222, then the ratio of the 
    radius of circle x to the radius of circle y is 6. -/
theorem concentric_circles_ratio (x y : Real) (h : x > y) 
    (prob : (x^2 - y^2) / x^2 = 0.9722222222222222) : x / y = 6 := by
  sorry


end NUMINAMATH_CALUDE_concentric_circles_ratio_l345_34535


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l345_34591

theorem geometric_sequence_first_term (a r : ℝ) : 
  (a * r^2 = 3) → (a * r^4 = 27) → (a = Real.sqrt 9 ∨ a = -Real.sqrt 9) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l345_34591


namespace NUMINAMATH_CALUDE_squares_property_l345_34511

theorem squares_property (a b c : ℕ) 
  (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ (w x y z : ℕ), 
    a * b = w^2 ∧ 
    b * c = x^2 ∧ 
    c * a = y^2 ∧ 
    a * b + b * c + c * a = z^2 := by
  sorry

end NUMINAMATH_CALUDE_squares_property_l345_34511


namespace NUMINAMATH_CALUDE_sqrt_three_multiplication_l345_34510

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_multiplication_l345_34510


namespace NUMINAMATH_CALUDE_f_monotone_range_l345_34532

/-- Definition of the function f --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

/-- The main theorem --/
theorem f_monotone_range (a : ℝ) :
  (a > 0 ∧ a ≠ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_range_l345_34532


namespace NUMINAMATH_CALUDE_height_lateral_edge_ratio_is_correct_l345_34513

/-- Regular quadrilateral pyramid with vertex P and square base ABCD -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : RegularQuadPyramid

/-- The ratio of height to lateral edge in a regular quadrilateral pyramid
    where the intersecting plane creates a cross-section with half the area of the base -/
def height_to_lateral_edge_ratio (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

/-- Theorem stating the ratio of height to lateral edge -/
theorem height_lateral_edge_ratio_is_correct (p : RegularQuadPyramid) (plane : IntersectingPlane) :
  height_to_lateral_edge_ratio p plane = (1 + Real.sqrt 33) / 8 :=
sorry

end NUMINAMATH_CALUDE_height_lateral_edge_ratio_is_correct_l345_34513


namespace NUMINAMATH_CALUDE_brent_kitkat_count_l345_34531

/-- The number of Kit-Kat bars Brent received -/
def kitKat : ℕ := sorry

/-- The number of Hershey kisses Brent received -/
def hersheyKisses : ℕ := 3 * kitKat

/-- The number of Nerds boxes Brent received -/
def nerds : ℕ := 8

/-- The initial number of lollipops Brent received -/
def initialLollipops : ℕ := 11

/-- The number of Baby Ruths Brent had -/
def babyRuths : ℕ := 10

/-- The number of Reese Peanut butter cups Brent had -/
def reeseCups : ℕ := babyRuths / 2

/-- The number of lollipops Brent gave to his sister -/
def givenLollipops : ℕ := 5

/-- The total number of candy pieces Brent had left -/
def totalCandyLeft : ℕ := 49

theorem brent_kitkat_count : kitKat = 5 := by
  sorry

end NUMINAMATH_CALUDE_brent_kitkat_count_l345_34531


namespace NUMINAMATH_CALUDE_possible_m_values_l345_34528

theorem possible_m_values (x m a b : ℤ) : 
  (∀ x, x^2 + m*x - 14 = (x + a) * (x + b)) → 
  (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
sorry

end NUMINAMATH_CALUDE_possible_m_values_l345_34528


namespace NUMINAMATH_CALUDE_emily_team_size_l345_34592

/-- Represents the number of players on Emily's team -/
def number_of_players : ℕ := sorry

/-- The total points scored by the team -/
def total_points : ℕ := 39

/-- The points scored by Emily -/
def emily_points : ℕ := 23

/-- The points scored by each other player -/
def points_per_other_player : ℕ := 2

theorem emily_team_size :
  number_of_players = 9 ∧
  total_points = emily_points + (number_of_players - 1) * points_per_other_player :=
sorry

end NUMINAMATH_CALUDE_emily_team_size_l345_34592


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l345_34529

theorem unique_solution_xyz (x y z : ℝ) 
  (eq1 : x + y = 4)
  (eq2 : x * y - z^2 = 4) :
  x = 2 ∧ y = 2 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l345_34529


namespace NUMINAMATH_CALUDE_unique_prime_product_sum_of_cubes_l345_34547

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- Product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is expressible as the sum of two positive cubes -/
def isSumOfTwoCubes (n : ℕ) : Prop := ∃ a b : ℕ+, n = a^3 + b^3

theorem unique_prime_product_sum_of_cubes :
  ∀ k : ℕ+, (k = 1 ↔ isSumOfTwoCubes (primeProduct k)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_sum_of_cubes_l345_34547


namespace NUMINAMATH_CALUDE_regular_milk_consumption_l345_34558

def total_milk : ℝ := 0.6
def soy_milk : ℝ := 0.1

theorem regular_milk_consumption : total_milk - soy_milk = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_regular_milk_consumption_l345_34558


namespace NUMINAMATH_CALUDE_root_is_factor_l345_34561

theorem root_is_factor (P : ℝ → ℝ) (a : ℝ) :
  (P a = 0) → ∃ Q : ℝ → ℝ, ∀ x, P x = (x - a) * Q x := by
  sorry

end NUMINAMATH_CALUDE_root_is_factor_l345_34561


namespace NUMINAMATH_CALUDE_max_value_of_f_l345_34548

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l345_34548


namespace NUMINAMATH_CALUDE_mans_walking_speed_l345_34590

theorem mans_walking_speed (woman_speed : ℝ) (passing_wait_time : ℝ) (catch_up_time : ℝ) :
  woman_speed = 25 →
  passing_wait_time = 5 / 60 →
  catch_up_time = 20 / 60 →
  ∃ (man_speed : ℝ),
    woman_speed * passing_wait_time = man_speed * (passing_wait_time + catch_up_time) ∧
    man_speed = 25 / 4 := by
  sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l345_34590


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l345_34539

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l345_34539


namespace NUMINAMATH_CALUDE_cos_sixty_degrees_l345_34501

theorem cos_sixty_degrees : Real.cos (60 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sixty_degrees_l345_34501


namespace NUMINAMATH_CALUDE_frustum_slant_height_l345_34564

/-- 
Given a cone cut by a plane parallel to its base forming a frustum:
- r: radius of the upper base of the frustum
- 4r: radius of the lower base of the frustum
- 3: slant height of the removed cone
- h: slant height of the frustum

Prove that h = 9
-/
theorem frustum_slant_height (r : ℝ) (h : ℝ) : h / (4 * r) = (h + 3) / (5 * r) → h = 9 := by
  sorry

end NUMINAMATH_CALUDE_frustum_slant_height_l345_34564


namespace NUMINAMATH_CALUDE_expression_simplification_l345_34571

theorem expression_simplification (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l345_34571


namespace NUMINAMATH_CALUDE_sum_of_roots_l345_34594

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - a^2 + a - 5 = 0)
  (hb : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l345_34594


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l345_34516

theorem quadratic_roots_problem (x y : ℝ) : 
  x + y = 6 → 
  |x - y| = 8 → 
  x^2 - 6*x - 7 = 0 ∧ y^2 - 6*y - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l345_34516


namespace NUMINAMATH_CALUDE_anne_sweettarts_distribution_l345_34562

theorem anne_sweettarts_distribution (total_sweettarts : ℕ) (sweettarts_per_friend : ℕ) (num_friends : ℕ) : 
  total_sweettarts = 15 → 
  sweettarts_per_friend = 5 → 
  total_sweettarts = sweettarts_per_friend * num_friends → 
  num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_anne_sweettarts_distribution_l345_34562


namespace NUMINAMATH_CALUDE_symmetric_sequence_sum_is_n_squared_l345_34554

/-- The sum of the symmetric sequence 1+2+3+...+(n-1)+n+(n+1)+n+...+3+2+1 -/
def symmetricSequenceSum (n : ℕ) : ℕ :=
  (List.range n).sum + n + (n + 1) + (List.range n).sum

/-- Theorem: The sum of the symmetric sequence is equal to n^2 -/
theorem symmetric_sequence_sum_is_n_squared (n : ℕ) :
  symmetricSequenceSum n = n^2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sequence_sum_is_n_squared_l345_34554


namespace NUMINAMATH_CALUDE_smallest_N_l345_34500

/-- Represents a point in the square array -/
structure Point where
  row : Fin 4
  col : Nat

/-- The first numbering scheme (left to right, top to bottom) -/
def x (p : Point) (N : Nat) : Nat :=
  p.row.val * N + p.col

/-- The second numbering scheme (top to bottom, left to right) -/
def y (p : Point) : Nat :=
  (p.col - 1) * 4 + p.row.val + 1

/-- The theorem stating the smallest possible value of N -/
theorem smallest_N : ∃ (N : Nat) (p₁ p₂ p₃ p₄ : Point),
  N > 0 ∧
  p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧
  p₁.col > 0 ∧ p₂.col > 0 ∧ p₃.col > 0 ∧ p₄.col > 0 ∧
  p₁.col ≤ N ∧ p₂.col ≤ N ∧ p₃.col ≤ N ∧ p₄.col ≤ N ∧
  x p₁ N = y p₃ ∧
  x p₂ N = y p₁ ∧
  x p₃ N = y p₄ ∧
  x p₄ N = y p₂ ∧
  (∀ (M : Nat) (q₁ q₂ q₃ q₄ : Point),
    M > 0 ∧
    q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧
    q₁.col > 0 ∧ q₂.col > 0 ∧ q₃.col > 0 ∧ q₄.col > 0 ∧
    q₁.col ≤ M ∧ q₂.col ≤ M ∧ q₃.col ≤ M ∧ q₄.col ≤ M ∧
    x q₁ M = y q₃ ∧
    x q₂ M = y q₁ ∧
    x q₃ M = y q₄ ∧
    x q₄ M = y q₂ →
    N ≤ M) ∧
  N = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_l345_34500


namespace NUMINAMATH_CALUDE_composite_function_properties_l345_34563

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_domain : ∀ x, x ∈ Set.univ)
variable (h_increasing : ∀ x y, 2 < x ∧ x < y ∧ y < 6 → f x < f y)

-- Define the composite function g
def g (x : ℝ) := f (2 - x)

-- Theorem statement
theorem composite_function_properties :
  (∀ x y, 4 < x ∧ x < y ∧ y < 8 → g x < g y) ∧
  (∀ x, g x = g (4 - x)) :=
sorry

end NUMINAMATH_CALUDE_composite_function_properties_l345_34563


namespace NUMINAMATH_CALUDE_square_difference_formula_l345_34566

theorem square_difference_formula : 15^2 - 2*(15*5) + 5^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l345_34566


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l345_34551

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ a = 25 * r ∧ 7/9 = a * r) : 
  a = 5 * Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l345_34551


namespace NUMINAMATH_CALUDE_divisors_of_100n4_l345_34569

-- Define a function to count divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Define the condition from the problem
def condition (n : ℕ) : Prop :=
  n > 0 ∧ count_divisors (90 * n^2) = 90

-- Theorem statement
theorem divisors_of_100n4 (n : ℕ) (h : condition n) :
  count_divisors (100 * n^4) = 245 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_100n4_l345_34569


namespace NUMINAMATH_CALUDE_mrs_brier_bakes_18_muffins_l345_34570

/-- The number of muffins baked by Mrs. Brier's class -/
def mrs_brier_muffins : ℕ := 55 - (20 + 17)

/-- Theorem stating that Mrs. Brier's class bakes 18 muffins -/
theorem mrs_brier_bakes_18_muffins :
  mrs_brier_muffins = 18 := by sorry

end NUMINAMATH_CALUDE_mrs_brier_bakes_18_muffins_l345_34570


namespace NUMINAMATH_CALUDE_root_range_l345_34576

theorem root_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1) := by
sorry

end NUMINAMATH_CALUDE_root_range_l345_34576


namespace NUMINAMATH_CALUDE_geometric_sequence_r_value_l345_34549

/-- A geometric sequence with sum of first n terms S_n = 2 * 3^n + r -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), n ≥ 1 → a (n + 1) = q * a n

/-- Sum of first n terms of the sequence -/
def S (n : ℕ) (r : ℝ) : ℝ := 2 * 3^n + r

/-- First term of the sequence -/
def a₁ (r : ℝ) : ℝ := S 1 r

theorem geometric_sequence_r_value
  (a : ℕ → ℝ) (r : ℝ)
  (h_geo : GeometricSequence a)
  (h_sum : ∀ n, n ≥ 1 → a n = S n r - S (n-1) r) :
  r = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_r_value_l345_34549


namespace NUMINAMATH_CALUDE_maria_chocolate_chip_cookies_l345_34557

/-- Calculates the number of chocolate chip cookies Maria had -/
def chocolateChipCookies (cookiesPerBag : ℕ) (oatmealCookies : ℕ) (numBags : ℕ) : ℕ :=
  cookiesPerBag * numBags - oatmealCookies

/-- Proves that Maria had 5 chocolate chip cookies -/
theorem maria_chocolate_chip_cookies :
  chocolateChipCookies 8 19 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_chocolate_chip_cookies_l345_34557


namespace NUMINAMATH_CALUDE_roots_of_equation_l345_34522

theorem roots_of_equation (x : ℝ) :
  (x^2 - 5*x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l345_34522


namespace NUMINAMATH_CALUDE_sin_negative_seventeen_pi_thirds_l345_34555

theorem sin_negative_seventeen_pi_thirds :
  Real.sin (-17 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_seventeen_pi_thirds_l345_34555


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l345_34572

theorem consecutive_integers_product (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∀ (start : ℕ), ∃ (x y : ℕ), 
    start < x ∧ x ≤ start + b - 1 ∧
    start < y ∧ y ≤ start + b - 1 ∧
    x ≠ y ∧
    (a * b) ∣ (x * y) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l345_34572


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_six_l345_34575

/-- A three-digit number in the form 2a3 -/
def number_2a3 (a : ℕ) : ℕ := 200 + 10 * a + 3

/-- A three-digit number in the form 5b9 -/
def number_5b9 (b : ℕ) : ℕ := 500 + 10 * b + 9

/-- Proposition: If 2a3 + 326 = 5b9 and 5b9 is a multiple of 9, then a + b = 6 -/
theorem sum_of_a_and_b_is_six (a b : ℕ) :
  number_2a3 a + 326 = number_5b9 b →
  (∃ k : ℕ, number_5b9 b = 9 * k) →
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_six_l345_34575


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l345_34536

theorem sqrt_expression_simplification :
  Real.sqrt 24 - 3 * Real.sqrt (1/6) + Real.sqrt 6 = (5 * Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l345_34536


namespace NUMINAMATH_CALUDE_megacorp_fine_l345_34518

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculate the profit percentage for a given day -/
def profitPercentage (d : Day) : ℝ :=
  match d with
  | Day.Monday => 0.1
  | Day.Tuesday => 0.2
  | Day.Wednesday => 0.15
  | Day.Thursday => 0.25
  | Day.Friday => 0.3
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Daily earnings from mining -/
def miningEarnings : ℝ := 3000000

/-- Daily earnings from oil refining -/
def oilRefiningEarnings : ℝ := 5000000

/-- Monthly expenses -/
def monthlyExpenses : ℝ := 30000000

/-- Tax rate on profits -/
def taxRate : ℝ := 0.35

/-- Fine rate on annual profits -/
def fineRate : ℝ := 0.01

/-- Number of days in a month (assumed average) -/
def daysInMonth : ℕ := 30

/-- Number of months in a year -/
def monthsInYear : ℕ := 12

/-- Calculate MegaCorp's fine -/
def calculateFine : ℝ :=
  let dailyEarnings := miningEarnings + oilRefiningEarnings
  let weeklyProfits := (List.sum (List.map (fun d => dailyEarnings * profitPercentage d) [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]))
  let monthlyRevenue := dailyEarnings * daysInMonth
  let monthlyProfits := monthlyRevenue - monthlyExpenses - (taxRate * (weeklyProfits * 4))
  let annualProfits := monthlyProfits * monthsInYear
  fineRate * annualProfits

theorem megacorp_fine : calculateFine = 23856000 := by sorry


end NUMINAMATH_CALUDE_megacorp_fine_l345_34518


namespace NUMINAMATH_CALUDE_fair_attendance_l345_34567

/-- The total attendance at a fair over three years -/
def total_attendance (this_year : ℕ) : ℕ :=
  let next_year := 2 * this_year
  let last_year := next_year - 200
  last_year + this_year + next_year

/-- Theorem: The total attendance over three years is 2800 -/
theorem fair_attendance : total_attendance 600 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l345_34567


namespace NUMINAMATH_CALUDE_class_size_l345_34530

theorem class_size (N : ℕ) (S D B : ℕ) : 
  S = (3 * N) / 5 →  -- 3/5 of the class swims
  D = (3 * N) / 5 →  -- 3/5 of the class dances
  B = 5 →            -- 5 pupils both swim and dance
  N = S + D - B →    -- Total is sum of swimmers and dancers minus overlap
  N = 25 := by sorry

end NUMINAMATH_CALUDE_class_size_l345_34530


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l345_34533

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 5 ∧ n ≡ -7458 [ZMOD 6] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l345_34533


namespace NUMINAMATH_CALUDE_chord_length_l345_34598

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a - b*t ∧ y = c + b*t}
  let chord := circle ∩ line
  r = 2 ∧ a = 2 ∧ b = 1/2 ∧ c = -1 →
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l345_34598


namespace NUMINAMATH_CALUDE_expected_value_of_special_die_l345_34578

def die_faces : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]

theorem expected_value_of_special_die :
  (List.sum die_faces) / (List.length die_faces : ℚ) = 650 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_special_die_l345_34578


namespace NUMINAMATH_CALUDE_range_of_M_l345_34588

theorem range_of_M (x y : ℝ) (h : x^2 + x*y + y^2 = 2) : 
  let M := x^2 - x*y + y^2
  2/3 ≤ M ∧ M ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_range_of_M_l345_34588


namespace NUMINAMATH_CALUDE_only_statement1_is_true_l345_34542

-- Define the statements
def statement1 (a x y : ℝ) : Prop := a * (x - y) = a * x - a * y
def statement2 (a x y : ℝ) : Prop := a^(x - y) = a^x - a^y
def statement3 (x y : ℝ) : Prop := x > y ∧ y > 0 → Real.log (x - y) = Real.log x - Real.log y
def statement4 (x y : ℝ) : Prop := x > 0 ∧ y > 0 → Real.log x / Real.log y = Real.log x - Real.log y
def statement5 (a x y : ℝ) : Prop := a * (x * y) = (a * x) * (a * y)

-- Theorem stating that only statement1 is true among all statements
theorem only_statement1_is_true :
  (∀ a x y : ℝ, statement1 a x y) ∧
  (∃ a x y : ℝ, ¬ statement2 a x y) ∧
  (∃ x y : ℝ, ¬ statement3 x y) ∧
  (∃ x y : ℝ, ¬ statement4 x y) ∧
  (∃ a x y : ℝ, ¬ statement5 a x y) :=
sorry

end NUMINAMATH_CALUDE_only_statement1_is_true_l345_34542


namespace NUMINAMATH_CALUDE_no_integer_pairs_sqrt_l345_34544

theorem no_integer_pairs_sqrt : 
  ¬∃ (x y : ℕ), 0 < x ∧ x < y ∧ Real.sqrt 2756 = Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_sqrt_l345_34544


namespace NUMINAMATH_CALUDE_wendys_recycling_points_l345_34503

/-- Given that Wendy earns 5 points per recycled bag, had 11 bags, and didn't recycle 2 bags,
    prove that she earned 45 points. -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    (total_bags - unrecycled_bags) * points_per_bag = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_wendys_recycling_points_l345_34503


namespace NUMINAMATH_CALUDE_tims_movie_marathon_duration_l345_34534

/-- The duration of Tim's movie marathon --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_factor : ℝ) (third_movie_offset : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_factor)
  let third_movie := first_movie + second_movie - third_movie_offset
  first_movie + second_movie + third_movie

/-- Theorem stating the duration of Tim's specific movie marathon --/
theorem tims_movie_marathon_duration :
  movie_marathon_duration 2 0.5 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tims_movie_marathon_duration_l345_34534


namespace NUMINAMATH_CALUDE_horner_v1_for_f_at_neg_two_l345_34514

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : List ℝ := [2, 0.3, 1, 0, 6, -5, 1]

/-- Theorem: v1 in Horner's method for f(x) at x = -2 is -7 -/
theorem horner_v1_for_f_at_neg_two :
  (horner (f.tail) (-2) : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_horner_v1_for_f_at_neg_two_l345_34514


namespace NUMINAMATH_CALUDE_simplest_fraction_C_l345_34540

def is_simplest_fraction (num : ℚ → ℚ) (denom : ℚ → ℚ) : Prop :=
  ∀ a : ℚ, ∀ k : ℚ, k ≠ 0 → num a / denom a = (k * num a) / (k * denom a) → k = 1 ∨ k = -1

theorem simplest_fraction_C :
  is_simplest_fraction (λ a => 2 * a) (λ a => 2 - a) :=
sorry

end NUMINAMATH_CALUDE_simplest_fraction_C_l345_34540


namespace NUMINAMATH_CALUDE_probability_three_students_same_canteen_l345_34502

/-- The probability of all three students going to the same canteen -/
def probability_same_canteen (num_canteens : ℕ) (num_students : ℕ) : ℚ :=
  if num_canteens = 2 ∧ num_students = 3 then
    1 / 4
  else
    0

/-- Theorem: The probability of all three students going to the same canteen is 1/4 -/
theorem probability_three_students_same_canteen :
  probability_same_canteen 2 3 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_students_same_canteen_l345_34502


namespace NUMINAMATH_CALUDE_rectangular_field_area_l345_34565

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 15) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) : a * b = 120 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l345_34565


namespace NUMINAMATH_CALUDE_jennifer_cookie_sales_l345_34593

theorem jennifer_cookie_sales (kim_sales : ℕ) (jennifer_extra : ℕ) 
  (h1 : kim_sales = 54)
  (h2 : jennifer_extra = 17) : 
  kim_sales + jennifer_extra = 71 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_cookie_sales_l345_34593


namespace NUMINAMATH_CALUDE_sum_w_y_l345_34581

theorem sum_w_y (w x y z : ℚ) 
  (eq1 : w * x * y = 10)
  (eq2 : w * y * z = 5)
  (eq3 : w * x * z = 45)
  (eq4 : x * y * z = 12) :
  w + y = 19/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_w_y_l345_34581


namespace NUMINAMATH_CALUDE_circles_tangent_radius_l345_34587

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 8*y - 5 = 0
def circle_O2 (x y r : ℝ) : Prop := (x+2)^2 + y^2 = r^2

-- Theorem statement
theorem circles_tangent_radius (r : ℝ) :
  (r > 0) →
  (∃! p : ℝ × ℝ, circle_O1 p.1 p.2 ∧ circle_O2 p.1 p.2 r) →
  r = 1 ∨ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_radius_l345_34587


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l345_34538

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly (f : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k : ℕ, k > n → f k = 0

theorem polynomial_value_at_two
  (f : ℕ → ℕ)
  (h_poly : NonNegIntPoly f)
  (h_one : f 1 = 6)
  (h_seven : f 7 = 3438) :
  f 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l345_34538


namespace NUMINAMATH_CALUDE_price_is_four_l345_34507

/-- The price per bag of leaves for Bob and Johnny's leaf raking business -/
def price_per_bag (monday_bags : ℕ) (tuesday_bags : ℕ) (wednesday_bags : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (monday_bags + tuesday_bags + wednesday_bags)

/-- Theorem stating that the price per bag is $4 given the conditions -/
theorem price_is_four :
  price_per_bag 5 3 9 68 = 4 := by
  sorry

end NUMINAMATH_CALUDE_price_is_four_l345_34507


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l345_34553

theorem inequality_solution_sets (a b : ℝ) : 
  (∀ x : ℝ, ax - b > 0 ↔ x < 3) →
  (∀ x : ℝ, (b*x^2 + a) / (x + 1) > 0 ↔ x < -1 ∨ (-1/3 < x ∧ x < 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l345_34553


namespace NUMINAMATH_CALUDE_two_pairs_probability_l345_34519

-- Define the total number of socks
def total_socks : ℕ := 10

-- Define the number of colors
def num_colors : ℕ := 5

-- Define the number of socks per color
def socks_per_color : ℕ := 2

-- Define the number of socks drawn
def socks_drawn : ℕ := 4

-- Define the probability of drawing two pairs of different colors
def prob_two_pairs : ℚ := 1 / 21

-- Theorem statement
theorem two_pairs_probability :
  (Nat.choose num_colors 2) / (Nat.choose total_socks socks_drawn) = prob_two_pairs :=
sorry

end NUMINAMATH_CALUDE_two_pairs_probability_l345_34519


namespace NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l345_34526

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Christen peeled 21 potatoes in the given scenario -/
theorem christen_peeled_21_potatoes :
  let scenario : PotatoPeeling := {
    initial_potatoes := 60,
    homer_rate := 4,
    christen_rate := 6,
    time_before_christen := 6
  }
  christenPeeledPotatoes scenario = 21 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_21_potatoes_l345_34526


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l345_34508

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has left -/
def seashells_left : ℕ := 17

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_eq_sum : 
  total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l345_34508


namespace NUMINAMATH_CALUDE_animal_feet_count_animal_feet_theorem_l345_34573

theorem animal_feet_count (total_heads : Nat) (hen_count : Nat) : Nat :=
  let cow_count := total_heads - hen_count
  let hen_feet := hen_count * 2
  let cow_feet := cow_count * 4
  hen_feet + cow_feet

theorem animal_feet_theorem (total_heads : Nat) (hen_count : Nat) 
  (h1 : total_heads = 44) (h2 : hen_count = 18) : 
  animal_feet_count total_heads hen_count = 140 := by
  sorry

end NUMINAMATH_CALUDE_animal_feet_count_animal_feet_theorem_l345_34573


namespace NUMINAMATH_CALUDE_complex_equation_solution_l345_34585

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2*Complex.I → z = -1 + (3/2)*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l345_34585


namespace NUMINAMATH_CALUDE_num_factors_1320_eq_32_l345_34512

/-- The number of distinct positive factors of 1320 -/
def num_factors_1320 : ℕ := sorry

/-- 1320 is a positive integer -/
axiom h_1320_positive : 1320 > 0

/-- Theorem: The number of distinct positive factors of 1320 is 32 -/
theorem num_factors_1320_eq_32 : num_factors_1320 = 32 := by sorry

end NUMINAMATH_CALUDE_num_factors_1320_eq_32_l345_34512


namespace NUMINAMATH_CALUDE_periodic_function_value_l345_34568

def f (x : ℝ) : ℝ := sorry

theorem periodic_function_value (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 2) = f x) →
  (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 4^x - 1) →
  f (-5.5) = 1 := by sorry

end NUMINAMATH_CALUDE_periodic_function_value_l345_34568


namespace NUMINAMATH_CALUDE_lines_theorem_l345_34586

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2*x + 3*y - 5 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₃ (x y : ℝ) : Prop := 2*x + y - 5 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define parallel and perpendicular lines
def parallel_line (x y : ℝ) : Prop := 2*x + y - 3 = 0
def perpendicular_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

theorem lines_theorem :
  (∃ (x y : ℝ), l₁ x y ∧ l₂ x y) →  -- P exists as intersection of l₁ and l₂
  (parallel_line (P.1) (P.2)) ∧    -- Parallel line passes through P
  (∀ (x y : ℝ), parallel_line x y → (∃ (k : ℝ), y - P.2 = k * (x - P.1) ∧ y = -2*x + 5)) ∧  -- Parallel line is parallel to l₃
  (perpendicular_line (P.1) (P.2)) ∧  -- Perpendicular line passes through P
  (∀ (x y : ℝ), perpendicular_line x y → 
    (∃ (k₁ k₂ : ℝ), y - P.2 = k₁ * (x - P.1) ∧ y = k₂ * x - 5/2 ∧ k₁ * k₂ = -1)) -- Perpendicular line is perpendicular to l₃
  := by sorry

end NUMINAMATH_CALUDE_lines_theorem_l345_34586


namespace NUMINAMATH_CALUDE_student_a_score_l345_34583

/-- Calculates the score for a test based on the given grading method -/
def calculate_score (total_questions : ℕ) (correct_answers : ℕ) : ℕ :=
  let incorrect_answers := total_questions - correct_answers
  correct_answers - 2 * incorrect_answers

/-- Theorem stating that the score for the given conditions is 61 -/
theorem student_a_score :
  let total_questions : ℕ := 100
  let correct_answers : ℕ := 87
  calculate_score total_questions correct_answers = 61 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l345_34583


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l345_34580

theorem logarithm_sum_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) +
   1 / (Real.log 2 / Real.log 8 + 1) +
   1 / (Real.log 9 / Real.log 18 + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l345_34580


namespace NUMINAMATH_CALUDE_marble_problem_l345_34504

theorem marble_problem (r b : ℕ) : 
  (r > 0) →
  (b > 0) →
  ((r - 1 : ℚ) / (r + b - 1) = 1 / 7) →
  (r / (r + b - 2 : ℚ) = 1 / 5) →
  (r + b = 22) :=
by sorry

end NUMINAMATH_CALUDE_marble_problem_l345_34504


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l345_34545

/-- Calculates the total water capacity for a farmer's trucks -/
theorem farmer_water_capacity 
  (num_trucks : ℕ) 
  (tanks_per_truck : ℕ) 
  (tank_capacity : ℕ) 
  (h1 : num_trucks = 5)
  (h2 : tanks_per_truck = 4)
  (h3 : tank_capacity = 200) : 
  num_trucks * tanks_per_truck * tank_capacity = 4000 := by
  sorry

#check farmer_water_capacity

end NUMINAMATH_CALUDE_farmer_water_capacity_l345_34545


namespace NUMINAMATH_CALUDE_movie_duration_l345_34521

theorem movie_duration (flight_duration : ℕ) (tv_time : ℕ) (sleep_time : ℕ) (remaining_time : ℕ) (num_movies : ℕ) :
  flight_duration = 600 →
  tv_time = 75 →
  sleep_time = 270 →
  remaining_time = 45 →
  num_movies = 2 →
  ∃ (movie_duration : ℕ),
    flight_duration = tv_time + sleep_time + num_movies * movie_duration + remaining_time ∧
    movie_duration = 105 := by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l345_34521


namespace NUMINAMATH_CALUDE_x4_plus_81_factorization_l345_34525

theorem x4_plus_81_factorization (x : ℝ) :
  x^4 + 81 = (x^2 - 3*x + 4.5) * (x^2 + 3*x + 4.5) := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_81_factorization_l345_34525


namespace NUMINAMATH_CALUDE_edward_sold_games_l345_34527

theorem edward_sold_games (initial_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : initial_games = 35)
  (h2 : boxes = 2)
  (h3 : games_per_box = 8) :
  initial_games - (boxes * games_per_box) = 19 := by
  sorry

end NUMINAMATH_CALUDE_edward_sold_games_l345_34527


namespace NUMINAMATH_CALUDE_dice_sum_product_l345_34596

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 144 →
  a + b + c + d ≠ 18 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_product_l345_34596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l345_34524

/-- An arithmetic sequence with a_3 = 0 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 3 = 0

theorem arithmetic_sequence_geometric_mean 
  (a : ℕ → ℝ) (k : ℕ) 
  (h_arith : arithmetic_sequence a) 
  (h_geom_mean : a k ^ 2 = a 6 * a (k + 6)) : 
  k = 9 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l345_34524


namespace NUMINAMATH_CALUDE_sum_of_like_monomials_l345_34577

/-- The sum of like monomials -/
theorem sum_of_like_monomials (m n : ℕ) :
  (2 * n : ℤ) * X^(m + 2) * Y^7 + (-4 * m : ℤ) * X^4 * Y^(3 * n - 2) = -2 * X^4 * Y^7 :=
by
  sorry

#check sum_of_like_monomials

end NUMINAMATH_CALUDE_sum_of_like_monomials_l345_34577


namespace NUMINAMATH_CALUDE_cylinder_cone_volume_ratio_l345_34541

theorem cylinder_cone_volume_ratio (h r_cylinder r_cone : ℝ) 
  (h_positive : h > 0)
  (r_cylinder_positive : r_cylinder > 0)
  (r_cone_positive : r_cone > 0)
  (cross_section_equal : h * (2 * r_cylinder) = h * r_cone) :
  (π * r_cylinder^2 * h) / ((1/3) * π * r_cone^2 * h) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_cylinder_cone_volume_ratio_l345_34541


namespace NUMINAMATH_CALUDE_linear_function_property_l345_34509

/-- A linear function is a function f : ℝ → ℝ such that f(ax + by) = af(x) + bf(y) for all x, y ∈ ℝ and a, b ∈ ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y a b : ℝ), f (a * x + b * y) = a * f x + b * f y

theorem linear_function_property (f : ℝ → ℝ) (h_linear : LinearFunction f) 
    (h_given : f 10 - f 4 = 20) : f 16 - f 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l345_34509


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_6_l345_34543

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def digit_sum (n : ℕ) : ℕ := 
  let digits := n.digits 10
  digits.sum

theorem five_digit_multiple_of_6 (n : ℕ) : 
  (∃ d : ℕ, n = 84370 + d ∧ d < 10) → 
  is_multiple_of_6 n → 
  (n.mod 10 = 2 ∨ n.mod 10 = 8) :=
sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_6_l345_34543


namespace NUMINAMATH_CALUDE_decline_type_composition_l345_34559

/-- Represents the age composition of a population --/
inductive AgeComposition
  | Growth
  | Stable
  | Decline

/-- Represents the relative distribution of age groups in a population --/
structure PopulationDistribution where
  young : ℕ
  adult : ℕ
  elderly : ℕ

/-- Determines the age composition based on the population distribution --/
def determineAgeComposition (pop : PopulationDistribution) : AgeComposition :=
  sorry

/-- Theorem stating that a population with fewer young individuals and more adults and elderly individuals has a decline type age composition --/
theorem decline_type_composition (pop : PopulationDistribution) 
  (h1 : pop.young < pop.adult)
  (h2 : pop.young < pop.elderly) :
  determineAgeComposition pop = AgeComposition.Decline :=
sorry

end NUMINAMATH_CALUDE_decline_type_composition_l345_34559


namespace NUMINAMATH_CALUDE_machine_count_l345_34537

theorem machine_count (hours_R hours_S total_hours : ℕ) (h1 : hours_R = 36) (h2 : hours_S = 9) (h3 : total_hours = 12) :
  ∃ (n : ℕ), n > 0 ∧ n * (1 / hours_R + 1 / hours_S) = 1 / total_hours ∧ n = 15 :=
by sorry

end NUMINAMATH_CALUDE_machine_count_l345_34537


namespace NUMINAMATH_CALUDE_four_digit_odd_divisible_by_digits_l345_34523

def is_odd (n : Nat) : Prop := ∃ k, n = 2 * k + 1

def is_four_digit (n : Nat) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_of (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

theorem four_digit_odd_divisible_by_digits :
  ∀ n : Nat,
  is_four_digit n →
  (let d := digits_of n
   d.length = 4 ∧
   (∀ x ∈ d, is_odd x) ∧
   d.toFinset.card = 4 ∧
   (∀ x ∈ d, n % x = 0)) →
  n ∈ [1395, 1935, 3195, 3915, 9135, 9315] := by
sorry

end NUMINAMATH_CALUDE_four_digit_odd_divisible_by_digits_l345_34523


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_711_1033_l345_34582

theorem smallest_multiple_of_5_711_1033 :
  ∃ (n : ℕ), n > 0 ∧ 
  5 ∣ n ∧ 711 ∣ n ∧ 1033 ∣ n ∧ 
  (∀ m : ℕ, m > 0 → 5 ∣ m → 711 ∣ m → 1033 ∣ m → n ≤ m) ∧
  n = 3683445 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_711_1033_l345_34582


namespace NUMINAMATH_CALUDE_hash_2_3_4_l345_34515

/-- The # operation defined on real numbers -/
def hash (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem stating that hash(2, 3, 4) = -15 -/
theorem hash_2_3_4 : hash 2 3 4 = -15 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l345_34515


namespace NUMINAMATH_CALUDE_base8_addition_subtraction_l345_34584

/-- Converts a base-8 number represented as a list of digits to its decimal (base-10) equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a decimal (base-10) number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base8_addition_subtraction :
  decimalToBase8 ((base8ToDecimal [1, 7, 6] + base8ToDecimal [4, 5]) - base8ToDecimal [6, 3]) = [1, 5, 1] := by
  sorry

end NUMINAMATH_CALUDE_base8_addition_subtraction_l345_34584


namespace NUMINAMATH_CALUDE_charles_watercolor_pictures_after_work_l345_34552

/-- Represents the number of pictures drawn on a specific type of paper -/
structure PictureCount where
  regular : ℕ
  watercolor : ℕ

/-- Represents the initial paper count and pictures drawn on different occasions -/
structure DrawingData where
  initialRegular : ℕ
  initialWatercolor : ℕ
  todayPictures : PictureCount
  yesterdayBeforeWork : ℕ
  remainingRegular : ℕ

/-- Calculates the number of watercolor pictures drawn after work yesterday -/
def watercolorPicturesAfterWork (data : DrawingData) : ℕ :=
  data.initialWatercolor - data.todayPictures.watercolor -
  (data.yesterdayBeforeWork - (data.initialRegular - data.todayPictures.regular - data.remainingRegular))

/-- Theorem stating that Charles drew 6 watercolor pictures after work yesterday -/
theorem charles_watercolor_pictures_after_work :
  let data : DrawingData := {
    initialRegular := 10,
    initialWatercolor := 10,
    todayPictures := { regular := 4, watercolor := 2 },
    yesterdayBeforeWork := 6,
    remainingRegular := 2
  }
  watercolorPicturesAfterWork data = 6 := by sorry

end NUMINAMATH_CALUDE_charles_watercolor_pictures_after_work_l345_34552


namespace NUMINAMATH_CALUDE_sum_of_P_and_R_is_eight_l345_34517

theorem sum_of_P_and_R_is_eight :
  ∀ (P Q R S : ℕ),
    P ∈ ({1, 2, 3, 5} : Set ℕ) →
    Q ∈ ({1, 2, 3, 5} : Set ℕ) →
    R ∈ ({1, 2, 3, 5} : Set ℕ) →
    S ∈ ({1, 2, 3, 5} : Set ℕ) →
    P ≠ Q → P ≠ R → P ≠ S → Q ≠ R → Q ≠ S → R ≠ S →
    (P : ℚ) / Q - (R : ℚ) / S = 2 →
    P + R = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_P_and_R_is_eight_l345_34517


namespace NUMINAMATH_CALUDE_cake_serving_solution_l345_34520

/-- Represents the number of cakes served for each type --/
structure CakeCount where
  chocolate : ℕ
  vanilla : ℕ
  strawberry : ℕ

/-- Represents the conditions of the cake serving problem --/
def cake_serving_conditions (c : CakeCount) : Prop :=
  c.chocolate = 2 * c.vanilla ∧
  c.strawberry = c.chocolate / 2 ∧
  c.vanilla = 12 + 18

/-- Theorem stating the correct number of cakes served for each type --/
theorem cake_serving_solution :
  ∃ c : CakeCount, cake_serving_conditions c ∧ 
    c.chocolate = 60 ∧ c.vanilla = 30 ∧ c.strawberry = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_serving_solution_l345_34520


namespace NUMINAMATH_CALUDE_square_root_of_25_l345_34546

theorem square_root_of_25 : ∀ x : ℝ, x^2 = 25 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_square_root_of_25_l345_34546


namespace NUMINAMATH_CALUDE_correct_factorization_l345_34574

theorem correct_factorization (x : ℝ) : x^2 - 0.01 = (x + 0.1) * (x - 0.1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l345_34574


namespace NUMINAMATH_CALUDE_condition_relationship_l345_34599

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a > 0 ∧ b > 0 → a * b > 0) ∧  -- A is necessary for B
  (∃ a b, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) -- A is not sufficient for B
  := by sorry

end NUMINAMATH_CALUDE_condition_relationship_l345_34599


namespace NUMINAMATH_CALUDE_matrix_power_four_l345_34589

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-14), -6; 3, (-17)] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l345_34589


namespace NUMINAMATH_CALUDE_f_properties_l345_34595

def f (x : ℝ) : ℝ := |x| - 1

theorem f_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_properties_l345_34595
