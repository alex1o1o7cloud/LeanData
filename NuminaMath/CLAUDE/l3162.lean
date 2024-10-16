import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3162_316241

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |4^x₀ - a * 2^x₀ + 1| ≤ 2^(x₀ + 1)) ↔ 
  a ∈ Set.Icc 0 (9/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3162_316241


namespace NUMINAMATH_CALUDE_cube_surface_area_l3162_316257

/-- Given a cube with volume 3375 cubic centimeters, its surface area is 1350 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 3375 → 
  volume = side ^ 3 → 
  surface_area = 6 * side ^ 2 → 
  surface_area = 1350 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3162_316257


namespace NUMINAMATH_CALUDE_second_x_intercept_of_quadratic_l3162_316275

/-- Given a quadratic function with vertex at (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the second x-intercept is 9. -/
theorem second_x_intercept_of_quadratic 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = -3) : 
  ∃ x, f x = 0 ∧ x = 9 := by
sorry

end NUMINAMATH_CALUDE_second_x_intercept_of_quadratic_l3162_316275


namespace NUMINAMATH_CALUDE_inequality_not_true_l3162_316211

theorem inequality_not_true (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ¬((2 * a * b) / (a + b) > Real.sqrt (a * b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_true_l3162_316211


namespace NUMINAMATH_CALUDE_function_always_positive_l3162_316205

-- Define a function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State that f' is the derivative of f
variable (hf' : ∀ x, HasDerivAt f (f' x) x)

-- State the given condition
variable (h : ∀ x, 2 * f x + x * f' x > x^2)

-- Theorem to prove
theorem function_always_positive : ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l3162_316205


namespace NUMINAMATH_CALUDE_min_value_problem_l3162_316262

open Real

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : log 2 * x + log 8 * y = log 2) : 
  (∀ a b : ℝ, a > 0 → b > 0 → log 2 * a + log 8 * b = log 2 → 1/x + 1/y ≤ 1/a + 1/b) ∧ 
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ log 2 * c + log 8 * d = log 2 ∧ 1/c + 1/d = 4 + 2 * sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3162_316262


namespace NUMINAMATH_CALUDE_nested_root_simplification_l3162_316213

theorem nested_root_simplification (b : ℝ) (h : b > 0) :
  (((b^16)^(1/8))^(1/4))^3 * (((b^16)^(1/4))^(1/8))^3 = b^3 := by sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l3162_316213


namespace NUMINAMATH_CALUDE_a_left_after_three_days_l3162_316222

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 21

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 28

/-- The number of days B worked alone to complete the remaining work -/
def b_remaining_days : ℝ := 21

/-- The number of days A worked before leaving -/
def x : ℝ := 3

theorem a_left_after_three_days :
  (x / (a_days⁻¹ + b_days⁻¹)⁻¹) + (b_remaining_days / b_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_a_left_after_three_days_l3162_316222


namespace NUMINAMATH_CALUDE_fraction_before_simplification_l3162_316201

theorem fraction_before_simplification
  (n d : ℕ)  -- n and d are the numerator and denominator before simplification
  (h1 : n + d = 80)  -- sum of numerator and denominator is 80
  (h2 : n / d = 3 / 7)  -- fraction simplifies to 3/7
  : n = 24 ∧ d = 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_before_simplification_l3162_316201


namespace NUMINAMATH_CALUDE_total_cost_is_183_l3162_316280

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklaceCapacity : ℕ
  currentNecklaces : ℕ
  ringCapacity : ℕ
  currentRings : ℕ
  braceletCapacity : ℕ
  currentBracelets : ℕ
  necklacePrice : ℕ
  ringPrice : ℕ
  braceletPrice : ℕ

/-- Calculates the total cost to fill the displays --/
def totalCost (store : JewelryStore) : ℕ :=
  (store.necklaceCapacity - store.currentNecklaces) * store.necklacePrice +
  (store.ringCapacity - store.currentRings) * store.ringPrice +
  (store.braceletCapacity - store.currentBracelets) * store.braceletPrice

/-- The specific jewelry store in the problem --/
def problemStore : JewelryStore := {
  necklaceCapacity := 12
  currentNecklaces := 5
  ringCapacity := 30
  currentRings := 18
  braceletCapacity := 15
  currentBracelets := 8
  necklacePrice := 4
  ringPrice := 10
  braceletPrice := 5
}

/-- Theorem stating that the total cost to fill the displays is $183 --/
theorem total_cost_is_183 : totalCost problemStore = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_183_l3162_316280


namespace NUMINAMATH_CALUDE_tangent_angle_parabola_l3162_316240

/-- The angle of inclination of the tangent to y = x^2 at (1/2, 1/4) is 45° -/
theorem tangent_angle_parabola : 
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1/2
  let y₀ : ℝ := 1/4
  let m := (deriv f) x₀
  let θ := Real.arctan m
  θ = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_parabola_l3162_316240


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l3162_316261

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for blue, False for red

/-- The volume of a tetrahedron formed by four vertices of a cube -/
def tetrahedronVolume (c : ColoredCube) (v1 v2 v3 v4 : Fin 8) : ℝ :=
  sorry

/-- Theorem: The volume of the tetrahedron formed by blue vertices in a cube with side length 8 is 170⅔ -/
theorem blue_tetrahedron_volume (c : ColoredCube) 
  (h1 : c.sideLength = 8)
  (h2 : ∀ i j : Fin 8, i ≠ j → c.vertexColors i ≠ c.vertexColors j) :
  ∃ v1 v2 v3 v4 : Fin 8, 
    (c.vertexColors v1 ∧ c.vertexColors v2 ∧ c.vertexColors v3 ∧ c.vertexColors v4) ∧
    tetrahedronVolume c v1 v2 v3 v4 = 170 + 2/3 :=
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l3162_316261


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3162_316288

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3162_316288


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3162_316218

def A : Set ℝ := {x | x - 6 < 0}
def B : Set ℝ := {-3, 5, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3162_316218


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3162_316252

-- Define a rectangle with integer side lengths
def Rectangle := { l : ℕ // l > 0 } × { w : ℕ // w > 0 }

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.1 + r.2)

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.1 * r.2

-- Theorem statement
theorem rectangle_area_difference :
  ∃ (max_area min_area : ℕ),
    (∀ r : Rectangle, perimeter r = 60 → area r ≤ max_area) ∧
    (∀ r : Rectangle, perimeter r = 60 → area r ≥ min_area) ∧
    (∃ r1 r2 : Rectangle, perimeter r1 = 60 ∧ perimeter r2 = 60 ∧ 
      area r1 = max_area ∧ area r2 = min_area) ∧
    max_area - min_area = 196 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3162_316252


namespace NUMINAMATH_CALUDE_houses_in_block_l3162_316233

/-- Proves that the number of houses in a block is 12, given the conditions of mail distribution. -/
theorem houses_in_block (total_mail : ℕ) (mail_per_house : ℕ) (skip_pattern : ℕ) :
  total_mail = 128 →
  mail_per_house = 16 →
  skip_pattern = 3 →
  ∃ (houses : ℕ), houses = 12 ∧ 
    houses * (skip_pattern - 1) * mail_per_house = total_mail * (skip_pattern - 1) / skip_pattern :=
by sorry

end NUMINAMATH_CALUDE_houses_in_block_l3162_316233


namespace NUMINAMATH_CALUDE_divisors_of_eight_n_cubed_l3162_316248

theorem divisors_of_eight_n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 17) :
  (Nat.divisors (8 * n^3)).card = 196 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_eight_n_cubed_l3162_316248


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3162_316273

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n.choose 2) = 45 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3162_316273


namespace NUMINAMATH_CALUDE_rihanna_remaining_money_l3162_316296

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount mango_price juice_price mango_count juice_count : ℕ) : ℕ :=
  initial_amount - (mango_price * mango_count + juice_price * juice_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money :
  remaining_money 50 3 3 6 6 = 14 := by
  sorry

#eval remaining_money 50 3 3 6 6

end NUMINAMATH_CALUDE_rihanna_remaining_money_l3162_316296


namespace NUMINAMATH_CALUDE_total_rent_is_6500_l3162_316249

/-- Represents the grazing data for a milkman -/
structure GrazingData where
  cows : ℕ
  months : ℕ

/-- Calculates the total rent of a pasture given grazing data and one milkman's rent -/
def totalRent (a b c d : GrazingData) (aRent : ℕ) : ℕ :=
  let totalCowMonths := a.cows * a.months + b.cows * b.months + c.cows * c.months + d.cows * d.months
  let rentPerCowMonth := aRent / (a.cows * a.months)
  rentPerCowMonth * totalCowMonths

/-- Theorem stating that the total rent is 6500 given the problem conditions -/
theorem total_rent_is_6500 :
  let a : GrazingData := ⟨24, 3⟩
  let b : GrazingData := ⟨10, 5⟩
  let c : GrazingData := ⟨35, 4⟩
  let d : GrazingData := ⟨21, 3⟩
  let aRent : ℕ := 1440
  totalRent a b c d aRent = 6500 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_6500_l3162_316249


namespace NUMINAMATH_CALUDE_water_bottles_duration_l3162_316242

theorem water_bottles_duration (total_bottles : ℕ) (bottles_per_day : ℕ) (duration : ℕ) : 
  total_bottles = 153 → bottles_per_day = 9 → duration = 17 → 
  total_bottles = bottles_per_day * duration := by
sorry

end NUMINAMATH_CALUDE_water_bottles_duration_l3162_316242


namespace NUMINAMATH_CALUDE_monomino_position_l3162_316219

/-- Represents a position on an 8x8 board -/
def Position := Fin 8 × Fin 8

/-- Represents a tromino (3x1 rectangle) -/
def Tromino := List Position

/-- Represents a monomino (1x1 square) -/
def Monomino := Position

/-- Represents a coloring of the board -/
def Coloring := Position → Fin 3

/-- The first coloring pattern -/
def coloring1 : Coloring := sorry

/-- The second coloring pattern -/
def coloring2 : Coloring := sorry

/-- Checks if a tromino is valid (covers exactly one square of each color) -/
def isValidTromino (t : Tromino) (c : Coloring) : Prop := sorry

/-- Checks if a set of trominos and a monomino form a valid covering of the board -/
def isValidCovering (trominos : List Tromino) (monomino : Monomino) : Prop := sorry

theorem monomino_position (trominos : List Tromino) (monomino : Monomino) :
  isValidCovering trominos monomino →
  coloring1 monomino = 1 ∧ coloring2 monomino = 1 := by sorry

end NUMINAMATH_CALUDE_monomino_position_l3162_316219


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3162_316228

/-- Calculate the interest rate given principal, time, and simple interest -/
theorem simple_interest_rate (principal time simple_interest : ℝ) :
  principal > 0 ∧ time > 0 ∧ simple_interest > 0 →
  (simple_interest * 100) / (principal * time) = 9 ∧
  principal = 8965 ∧ time = 5 ∧ simple_interest = 4034.25 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3162_316228


namespace NUMINAMATH_CALUDE_quadrilateral_not_necessarily_plane_l3162_316234

-- Define the types of figures we're considering
inductive Figure
| Triangle
| Trapezoid
| Parallelogram
| Quadrilateral

-- Define what it means for a figure to be a plane figure
def is_plane_figure (f : Figure) : Prop :=
  match f with
  | Figure.Triangle => True
  | Figure.Trapezoid => True
  | Figure.Parallelogram => True
  | Figure.Quadrilateral => false

-- Theorem stating that only quadrilaterals are not necessarily plane figures
theorem quadrilateral_not_necessarily_plane :
  ∀ f : Figure, ¬(is_plane_figure f) ↔ f = Figure.Quadrilateral :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_not_necessarily_plane_l3162_316234


namespace NUMINAMATH_CALUDE_power_of_three_difference_l3162_316216

theorem power_of_three_difference : 3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_difference_l3162_316216


namespace NUMINAMATH_CALUDE_part1_part2_l3162_316263

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (1 / (2^x - 1) + a / 2)

-- Part 1: If f(x) is even and f(1) = 3/2, then f(-1) = 3/2
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → 
  f a 1 = 3/2 → 
  f a (-1) = 3/2 := by sorry

-- Part 2: If a = 1, then f(x) is an even function
theorem part2 : 
  ∀ x : ℝ, x ≠ 0 → f 1 x = f 1 (-x) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3162_316263


namespace NUMINAMATH_CALUDE_g_of_3_equals_19_l3162_316232

def g (x : ℝ) : ℝ := x^2 + 3*x + 1

theorem g_of_3_equals_19 : g 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_19_l3162_316232


namespace NUMINAMATH_CALUDE_f_is_quasi_even_l3162_316208

/-- A function is quasi-even if f(-x) = f(x) only for a finite number of non-zero arguments x. -/
def QuasiEven (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), ∀ x ≠ 0, f (-x) = f x ↔ x ∈ S

/-- The function f(x) = x³ - 2x -/
def f (x : ℝ) : ℝ := x^3 - 2*x

/-- Theorem: f(x) = x³ - 2x is a quasi-even function -/
theorem f_is_quasi_even : QuasiEven f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quasi_even_l3162_316208


namespace NUMINAMATH_CALUDE_expression_evaluation_l3162_316209

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  (2 * x^2 - 2 * y^2) - 3 * (x^2 * y^2 + x^2) + 3 * (x^2 * y^2 + y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3162_316209


namespace NUMINAMATH_CALUDE_subtraction_to_perfect_square_l3162_316255

theorem subtraction_to_perfect_square : ∃ n : ℕ, (92555 : ℕ) - 139 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_to_perfect_square_l3162_316255


namespace NUMINAMATH_CALUDE_ellipse_properties_l3162_316295

-- Define the ellipses
def ellipse_N (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

def ellipse_M (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem ellipse_properties (a b : ℝ) :
  (∀ x y, ellipse_M a b x y ↔ ellipse_N x y) →  -- M and N share foci
  (ellipse_M a b 0 2) →  -- M passes through (0,2)
  (∃ A B : ℝ × ℝ, 
    ellipse_M a b A.1 A.2 ∧ 
    ellipse_M a b B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A.1 > B.1) →  -- A and B are intersections of M and y = x + 2, with A to the right of B
  (2 * a = 4 * Real.sqrt 2) ∧  -- Length of major axis is 4√2
  (∃ A B : ℝ × ℝ, 
    ellipse_M a b A.1 A.2 ∧ 
    ellipse_M a b B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    A.1 > B.1 ∧ 
    A.1 * B.1 + A.2 * B.2 = -4/3) :=  -- Dot product of OA and OB is -4/3
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3162_316295


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l3162_316229

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is on a line segment
def isOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define a function to check if two line segments are parallel
def areParallel (A B : ℝ × ℝ) (C D : ℝ × ℝ) : Prop := sorry

-- Define a function to count valid triangles
def countValidTriangles (ABC X'Y'Z' : Triangle) : ℕ := sorry

-- Define a function to count valid triangles including extensions
def countValidTrianglesWithExtensions (ABC X'Y'Z' : Triangle) : ℕ := sorry

theorem triangle_construction_theorem (ABC X'Y'Z' : Triangle) :
  (countValidTriangles ABC X'Y'Z' = 2 ∨
   countValidTriangles ABC X'Y'Z' = 1 ∨
   countValidTriangles ABC X'Y'Z' = 0) ∧
  countValidTrianglesWithExtensions ABC X'Y'Z' = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l3162_316229


namespace NUMINAMATH_CALUDE_odd_function_property_l3162_316259

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  f 2016 = 2 → f (-2016) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3162_316259


namespace NUMINAMATH_CALUDE_system_solution_l3162_316292

theorem system_solution (x y z : ℝ) : 
  (5*x + 7*y) / (x + y) = 6 ∧
  3*(z - x) / (x - y + z) = 1 ∧
  (2*x + 3*y - z) / (x/2 + 3) = 4 →
  x = 8 ∧ y = 8 ∧ z = 12 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3162_316292


namespace NUMINAMATH_CALUDE_problem_statement_l3162_316235

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + |x^2 - 1|

def A (a : ℝ) : Set ℝ := {x | f a x ≤ 0}

theorem problem_statement (a : ℝ) :
  (∀ x ∈ A a, x ∈ Set.Icc 0 3) →
  -1 < a ∧ a ≤ 11/5 ∧
  (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧ g a x₁ = 0 ∧ g a x₂ = 0) →
  1 + Real.sqrt 3 < a ∧ a < 19/5 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3162_316235


namespace NUMINAMATH_CALUDE_sum_of_roots_l3162_316251

theorem sum_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x / Real.sqrt y - y / Real.sqrt x = 7 / 12)
  (h2 : x - y = 7) : x + y = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3162_316251


namespace NUMINAMATH_CALUDE_range_of_a_l3162_316217

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - a)^2 < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : 1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3162_316217


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3162_316210

theorem similar_triangle_perimeter (a b c : ℝ) (h1 : a = 12) (h2 : b = 12) (h3 : c = 15) 
  (h4 : a = b) (h5 : c ≥ a) (h6 : c ≥ b) (long_side : ℝ) (h7 : long_side = 45) : 
  (long_side / c) * (a + b + c) = 117 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3162_316210


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3162_316258

theorem x_plus_y_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x / 3 = y^2) (h2 : x / 9 = 9*y) : x + y = 2214 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3162_316258


namespace NUMINAMATH_CALUDE_soda_preference_count_l3162_316284

theorem soda_preference_count (total_surveyed : ℕ) (soda_angle : ℝ) 
  (h1 : total_surveyed = 600)
  (h2 : soda_angle = 108) : 
  ℕ := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_count_l3162_316284


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3162_316276

theorem sum_of_fractions : 2 / 5 + 3 / 8 = 31 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3162_316276


namespace NUMINAMATH_CALUDE_six_bottle_caps_cost_l3162_316299

/-- The cost of a given number of bottle caps -/
def bottle_cap_cost (num_caps : ℕ) (cost_per_cap : ℕ) : ℕ :=
  num_caps * cost_per_cap

/-- Theorem: The cost of 6 bottle caps at $2 each is $12 -/
theorem six_bottle_caps_cost : bottle_cap_cost 6 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_six_bottle_caps_cost_l3162_316299


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3162_316265

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3162_316265


namespace NUMINAMATH_CALUDE_complex_number_problem_l3162_316269

open Complex

theorem complex_number_problem :
  let z : ℂ := (1 - I)^2 + 3 + 6*I
  ∃ (a b : ℝ),
    z = 3 + 4*I ∧
    Complex.abs z = 5 ∧
    z^2 + a*z + b = -8 + 20*I ∧
    a = -1 ∧
    b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3162_316269


namespace NUMINAMATH_CALUDE_intersected_prisms_count_l3162_316297

def small_prism_dimensions : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

def cube_edge_length : ℕ := 90

def count_intersected_prisms (dimensions : Fin 3 → ℕ) (edge_length : ℕ) : ℕ :=
  sorry

theorem intersected_prisms_count :
  count_intersected_prisms small_prism_dimensions cube_edge_length = 66 := by sorry

end NUMINAMATH_CALUDE_intersected_prisms_count_l3162_316297


namespace NUMINAMATH_CALUDE_jamie_water_bottle_limit_l3162_316244

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def bathroom_limit : ℕ := 32

/-- The amount of milk Jamie consumed -/
def milk_consumed : ℕ := 8

/-- The amount of grape juice Jamie consumed -/
def grape_juice_consumed : ℕ := 16

/-- The amount Jamie can drink from her water bottle during the test -/
def water_bottle_limit : ℕ := bathroom_limit - (milk_consumed + grape_juice_consumed)

theorem jamie_water_bottle_limit :
  water_bottle_limit = 8 :=
by sorry

end NUMINAMATH_CALUDE_jamie_water_bottle_limit_l3162_316244


namespace NUMINAMATH_CALUDE_expression_simplification_l3162_316202

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2*x + 1)) = 1 + Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3162_316202


namespace NUMINAMATH_CALUDE_circle_centers_distance_l3162_316254

-- Define the circles
def circle_O₁ : ℝ := 3
def circle_O₂ : ℝ := 7

-- Define the condition of at most one common point
def at_most_one_common_point (d : ℝ) : Prop :=
  d ≥ circle_O₁ + circle_O₂ ∨ d ≤ abs (circle_O₁ - circle_O₂)

-- State the theorem
theorem circle_centers_distance (d : ℝ) :
  at_most_one_common_point d → d ≠ 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l3162_316254


namespace NUMINAMATH_CALUDE_problem_solution_l3162_316279

theorem problem_solution (x y : ℝ) 
  (eq1 : x + 2 * y = 1) 
  (eq2 : 2 * x - 3 * y = 2) : 
  (2 * x + 4 * y - 2) / 2 + (6 * x - 9 * y) / 3 = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3162_316279


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3162_316266

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The specific function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  x^2 + (a - 1) * x + a

theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, IsEven (f a) → a = 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l3162_316266


namespace NUMINAMATH_CALUDE_atop_difference_l3162_316298

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y - 3 * x

-- State the theorem
theorem atop_difference : atop 8 5 - atop 5 8 = -9 := by
  sorry

end NUMINAMATH_CALUDE_atop_difference_l3162_316298


namespace NUMINAMATH_CALUDE_book_distribution_count_l3162_316274

def distribute_books (total_books : ℕ) (min_per_location : ℕ) : ℕ :=
  (total_books - 2 * min_per_location + 1)

theorem book_distribution_count :
  distribute_books 8 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_book_distribution_count_l3162_316274


namespace NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l3162_316204

theorem sum_of_gcd_and_lcm : Nat.gcd 15 45 + Nat.lcm 15 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_gcd_and_lcm_l3162_316204


namespace NUMINAMATH_CALUDE_distance_lima_caracas_l3162_316270

theorem distance_lima_caracas : 
  let caracas : ℂ := 0
  let lima : ℂ := 960 + 1280 * Complex.I
  Complex.abs (lima - caracas) = 1600 := by
sorry

end NUMINAMATH_CALUDE_distance_lima_caracas_l3162_316270


namespace NUMINAMATH_CALUDE_least_three_digit_11_heavy_l3162_316200

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_three_digit_11_heavy : ∀ n : ℕ, 100 ≤ n ∧ n < 108 → ¬(is_11_heavy n) ∧ is_11_heavy 108 := by
  sorry

#check least_three_digit_11_heavy

end NUMINAMATH_CALUDE_least_three_digit_11_heavy_l3162_316200


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_l3162_316277

theorem unique_divisor_with_remainder (n : ℕ) (r : ℕ) : 
  (∃! u : ℕ, u > r ∧ n % u = r) ↔ n = 11 ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_l3162_316277


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l3162_316285

def total_marbles : ℕ := 12
def special_marbles : ℕ := 3
def marbles_to_choose : ℕ := 4

theorem marble_selection_theorem :
  (special_marbles) * (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_theorem_l3162_316285


namespace NUMINAMATH_CALUDE_fraction_simplification_l3162_316243

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x - 2) / (x^2 - 2*x + 1) / (x / (x - 1)) + 1 / (x^2 - x) = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3162_316243


namespace NUMINAMATH_CALUDE_dart_probability_l3162_316231

/-- The probability of a dart landing within a circle inscribed in a regular hexagonal dartboard -/
theorem dart_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let circle_area := π * s^2
  circle_area / hexagon_area = 2 * π / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l3162_316231


namespace NUMINAMATH_CALUDE_average_salary_l3162_316290

def salary_A : ℕ := 10000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def num_individuals : ℕ := 5

theorem average_salary :
  (salary_A + salary_B + salary_C + salary_D + salary_E) / num_individuals = 8600 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_l3162_316290


namespace NUMINAMATH_CALUDE_vector_properties_l3162_316283

/-- Given two vectors in ℝ², prove dot product and parallelism properties -/
theorem vector_properties (a b : ℝ × ℝ) (h1 : a = (1, -2)) (h2 : b = (-3, 2)) :
  (a + b) • (a - b) = -8 ∧
  ∃ k : ℝ, k = (1 : ℝ) / 3 ∧ ∃ c : ℝ, c ≠ 0 ∧ k • a + b = c • (a - 3 • b) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l3162_316283


namespace NUMINAMATH_CALUDE_bill_john_score_difference_l3162_316250

-- Define the scores as natural numbers
def bill_score : ℕ := 45
def sue_score : ℕ := bill_score * 2
def john_score : ℕ := 160 - bill_score - sue_score

-- Theorem statement
theorem bill_john_score_difference :
  bill_score > john_score ∧
  bill_score = sue_score / 2 ∧
  bill_score + john_score + sue_score = 160 →
  bill_score - john_score = 20 := by
sorry

end NUMINAMATH_CALUDE_bill_john_score_difference_l3162_316250


namespace NUMINAMATH_CALUDE_cube_painting_theorem_l3162_316214

/-- The number of faces on a cube -/
def num_faces : ℕ := 6

/-- The number of available colors -/
def num_colors : ℕ := 7

/-- The number of symmetries of a cube -/
def cube_symmetries : ℕ := 24

/-- The number of ways to paint a cube with different colors on each face -/
def paint_cube_ways : ℕ := (num_colors.factorial) / (num_colors - num_faces).factorial

/-- The number of distinct ways to paint a cube considering symmetries -/
def distinct_paint_ways : ℕ := paint_cube_ways / cube_symmetries

theorem cube_painting_theorem : distinct_paint_ways = 210 := by
  sorry

end NUMINAMATH_CALUDE_cube_painting_theorem_l3162_316214


namespace NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3162_316286

theorem least_positive_integer_satisfying_congruences : ∃ x : ℕ, x > 0 ∧
  ((x : ℤ) + 127 ≡ 53 [ZMOD 15]) ∧
  ((x : ℤ) + 104 ≡ 76 [ZMOD 7]) ∧
  (∀ y : ℕ, y > 0 →
    ((y : ℤ) + 127 ≡ 53 [ZMOD 15]) →
    ((y : ℤ) + 104 ≡ 76 [ZMOD 7]) →
    x ≤ y) ∧
  x = 91 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_satisfying_congruences_l3162_316286


namespace NUMINAMATH_CALUDE_kittens_given_to_jessica_l3162_316226

theorem kittens_given_to_jessica (initial_kittens : ℕ) (kittens_to_sara : ℕ) (kittens_left : ℕ) :
  initial_kittens = 18 → kittens_to_sara = 6 → kittens_left = 9 →
  initial_kittens - kittens_to_sara - kittens_left = 3 :=
by sorry

end NUMINAMATH_CALUDE_kittens_given_to_jessica_l3162_316226


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3162_316212

theorem absolute_value_equation_solutions :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 2 ∨ x = 8/3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3162_316212


namespace NUMINAMATH_CALUDE_one_third_of_eleven_y_plus_three_l3162_316267

theorem one_third_of_eleven_y_plus_three (y : ℝ) : (1 / 3) * (11 * y + 3) = (11 * y / 3) + 1 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_eleven_y_plus_three_l3162_316267


namespace NUMINAMATH_CALUDE_product_of_ten_fractions_is_one_tenth_l3162_316291

theorem product_of_ten_fractions_is_one_tenth : 
  ∃ (a b c d e f g h i j : ℚ), 
    (0 < a ∧ a < 1) ∧ 
    (0 < b ∧ b < 1) ∧ 
    (0 < c ∧ c < 1) ∧ 
    (0 < d ∧ d < 1) ∧ 
    (0 < e ∧ e < 1) ∧ 
    (0 < f ∧ f < 1) ∧ 
    (0 < g ∧ g < 1) ∧ 
    (0 < h ∧ h < 1) ∧ 
    (0 < i ∧ i < 1) ∧ 
    (0 < j ∧ j < 1) ∧ 
    a * b * c * d * e * f * g * h * i * j = 1 / 10 :=
by sorry


end NUMINAMATH_CALUDE_product_of_ten_fractions_is_one_tenth_l3162_316291


namespace NUMINAMATH_CALUDE_factorial_sum_ratio_l3162_316223

theorem factorial_sum_ratio (N : ℕ) (h : N > 0) : 
  (Nat.factorial (N + 1) + Nat.factorial (N - 1)) / Nat.factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3*N^2 + 2*N) := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_ratio_l3162_316223


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l3162_316264

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Calculates the total compensation based on the given rates and hours -/
def calculateTotalCompensation (c : BusDriverCompensation) : ℝ :=
  c.regularRate * c.regularHours + c.overtimeRate * c.overtimeHours

/-- Theorem stating that the bus driver's regular rate is $16 per hour -/
theorem bus_driver_regular_rate :
  ∃ (c : BusDriverCompensation),
    c.regularHours = 40 ∧
    c.overtimeHours = 8 ∧
    c.overtimeRate = 1.75 * c.regularRate ∧
    c.totalCompensation = 864 ∧
    calculateTotalCompensation c = c.totalCompensation ∧
    c.regularRate = 16 := by
  sorry


end NUMINAMATH_CALUDE_bus_driver_regular_rate_l3162_316264


namespace NUMINAMATH_CALUDE_georges_expenses_l3162_316245

theorem georges_expenses (B : ℝ) (B_pos : B > 0) : ∃ (m s : ℝ),
  m = 0.25 * (B - s) ∧
  s = 0.05 * (B - m) ∧
  m + s = B :=
by sorry

end NUMINAMATH_CALUDE_georges_expenses_l3162_316245


namespace NUMINAMATH_CALUDE_balloon_descent_rate_l3162_316230

/-- Hot air balloon ascent and descent rates problem -/
theorem balloon_descent_rate 
  (rise_rate : ℝ) 
  (total_pull_time : ℝ) 
  (release_time : ℝ) 
  (max_height : ℝ) 
  (h_rise_rate : rise_rate = 50)
  (h_total_pull_time : total_pull_time = 30)
  (h_release_time : release_time = 10)
  (h_max_height : max_height = 1400) :
  ∃ (descent_rate : ℝ),
    max_height = rise_rate * total_pull_time - descent_rate * release_time ∧
    descent_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloon_descent_rate_l3162_316230


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l3162_316237

def min_sugar_purchase (s : ℝ) : Prop :=
  ∀ f : ℝ, (f ≥ 8 + (3/4) * s) ∧ (f ≤ 3 * s) → s ≥ 4

theorem betty_sugar_purchase :
  ∃ s : ℝ, min_sugar_purchase s ∧ ∀ t : ℝ, t < s → ¬ min_sugar_purchase t :=
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l3162_316237


namespace NUMINAMATH_CALUDE_min_value_inequality_l3162_316220

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, |a - x| + |x + b| + c ≥ 1) →
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3162_316220


namespace NUMINAMATH_CALUDE_lucille_house_height_difference_l3162_316225

theorem lucille_house_height_difference (h1 h2 h3 : ℝ) :
  h1 = 80 ∧ h2 = 70 ∧ h3 = 99 →
  ((h1 + h2 + h3) / 3) - h1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_lucille_house_height_difference_l3162_316225


namespace NUMINAMATH_CALUDE_range_of_a_l3162_316215

-- Define the conditions P and Q
def P (x : ℝ) : Prop := x > 1 ∨ x < -3
def Q (x a : ℝ) : Prop := x > a

-- Define the relationship between P and Q
def Q_sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, Q x a → P x) ∧ ∃ x, P x ∧ ¬Q x a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  Q_sufficient_not_necessary a → a ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l3162_316215


namespace NUMINAMATH_CALUDE_ratio_of_fifteenth_terms_l3162_316239

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  first_term : ℚ
  common_difference : ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.first_term + (n - 1) * seq.common_difference) / 2

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem ratio_of_fifteenth_terms
  (seq1 seq2 : ArithmeticSequence)
  (h : ∀ n : ℕ, sum_n_terms seq1 n / sum_n_terms seq2 n = (5 * n + 3) / (3 * n + 35)) :
  nth_term seq1 15 / nth_term seq2 15 = 59 / 57 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_fifteenth_terms_l3162_316239


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3162_316246

theorem max_sum_of_factors (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2023 →
  A + B + C ≤ 297 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3162_316246


namespace NUMINAMATH_CALUDE_find_b_value_l3162_316256

theorem find_b_value (a b c : ℚ) 
  (sum_eq : a + b + c = 150)
  (eq_condition : (a + 10) = (b - 3) ∧ (b - 3) = 4 * c) : 
  b = 655 / 9 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3162_316256


namespace NUMINAMATH_CALUDE_mark_ordered_100_nuggets_l3162_316271

/-- The number of chicken nuggets in a box -/
def nuggets_per_box : ℕ := 20

/-- The cost of one box of chicken nuggets in dollars -/
def cost_per_box : ℕ := 4

/-- The amount Mark paid for chicken nuggets in dollars -/
def mark_paid : ℕ := 20

/-- The number of chicken nuggets Mark ordered -/
def mark_ordered : ℕ := (mark_paid / cost_per_box) * nuggets_per_box

theorem mark_ordered_100_nuggets : mark_ordered = 100 := by
  sorry

end NUMINAMATH_CALUDE_mark_ordered_100_nuggets_l3162_316271


namespace NUMINAMATH_CALUDE_simplify_expression_l3162_316224

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3162_316224


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3162_316294

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (2 / (x + 3))
  else if x < -3 then Int.floor (2 / (x + 3))
  else 0  -- This value doesn't matter as g is undefined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3162_316294


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3162_316207

theorem largest_multiple_of_8_under_100 : 
  ∃ n : ℕ, n * 8 = 96 ∧ 
  96 < 100 ∧ 
  ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3162_316207


namespace NUMINAMATH_CALUDE_mary_fruit_expenses_l3162_316268

theorem mary_fruit_expenses :
  let berries_cost : ℚ := 1108 / 100
  let apples_cost : ℚ := 1433 / 100
  let peaches_cost : ℚ := 931 / 100
  berries_cost + apples_cost + peaches_cost = 3472 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_expenses_l3162_316268


namespace NUMINAMATH_CALUDE_sum_subtract_problem_l3162_316238

theorem sum_subtract_problem : (34.5 + 15.2) - 0.7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_subtract_problem_l3162_316238


namespace NUMINAMATH_CALUDE_solution_for_equation_l3162_316282

noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem solution_for_equation (x : ℝ) (h1 : 0 < x) (h2 : x ≠ 1) :
  x^2 * log x 27 * log 9 x = x + 4 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_for_equation_l3162_316282


namespace NUMINAMATH_CALUDE_time_after_1450_minutes_l3162_316278

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

def midnight : Time where
  hours := 0
  minutes := 0
  h_valid := by simp
  m_valid := by simp

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := (totalMinutes / 60) % 24,
    minutes := totalMinutes % 60,
    h_valid := by sorry
    m_valid := by sorry }

theorem time_after_1450_minutes :
  addMinutes midnight 1450 = { hours := 0, minutes := 10, h_valid := by simp, m_valid := by simp } :=
by sorry

end NUMINAMATH_CALUDE_time_after_1450_minutes_l3162_316278


namespace NUMINAMATH_CALUDE_greatest_n_is_5_l3162_316287

/-- A coloring of a board is a function that assigns a color to each square. -/
def Coloring (m n : ℕ) := Fin m → Fin n → Fin 3

/-- A board has a valid coloring if no rectangle has all four corners of the same color. -/
def ValidColoring (m n : ℕ) (c : Coloring m n) : Prop :=
  ∀ (r1 r2 : Fin m) (c1 c2 : Fin n),
    r1 ≠ r2 → c1 ≠ c2 →
    (c r1 c1 = c r1 c2 ∧ c r1 c1 = c r2 c1 ∧ c r1 c1 = c r2 c2) → False

/-- The main theorem: The greatest possible value of n is 5. -/
theorem greatest_n_is_5 :
  (∃ (c : Coloring 6 4), ValidColoring 6 4 c) ∧
  (∀ n > 5, ¬∃ (c : Coloring (n+1) (n-1)), ValidColoring (n+1) (n-1) c) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_is_5_l3162_316287


namespace NUMINAMATH_CALUDE_paise_to_rupees_l3162_316293

/-- Proves that if 0.5% of a equals 80 paise, then a equals 160 rupees. -/
theorem paise_to_rupees (a : ℝ) : (0.005 * a = 0.8) → a = 160 := by
  sorry

end NUMINAMATH_CALUDE_paise_to_rupees_l3162_316293


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l3162_316236

theorem volunteer_arrangement_count (n : ℕ) (k : ℕ) (h1 : n = 7) (h2 : k = 3) :
  Nat.choose n k * Nat.choose (n - k) k = 140 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l3162_316236


namespace NUMINAMATH_CALUDE_spring_mass_for_32cm_l3162_316253

/-- Represents the relationship between spring length and mass -/
def spring_length (initial_length : ℝ) (extension_rate : ℝ) (mass : ℝ) : ℝ :=
  initial_length + extension_rate * mass

/-- Theorem: For a spring with initial length 18 cm and extension rate 2 cm/kg,
    a length of 32 cm corresponds to a mass of 7 kg -/
theorem spring_mass_for_32cm :
  spring_length 18 2 7 = 32 :=
by sorry

end NUMINAMATH_CALUDE_spring_mass_for_32cm_l3162_316253


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3162_316260

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) → 
  (3 * q^2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = -58 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3162_316260


namespace NUMINAMATH_CALUDE_work_completion_time_l3162_316272

theorem work_completion_time
  (x_alone_time : ℝ)
  (y_alone_time : ℝ)
  (x_alone_days : ℝ)
  (h1 : x_alone_time = 20)
  (h2 : y_alone_time = 12)
  (h3 : x_alone_days = 4)
  : ∃ (total_time : ℝ), total_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3162_316272


namespace NUMINAMATH_CALUDE_complex_equality_sum_l3162_316206

theorem complex_equality_sum (a b : ℝ) (h : a - 2 * Complex.I = b + a * Complex.I) : a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l3162_316206


namespace NUMINAMATH_CALUDE_partner_a_share_l3162_316281

/-- Calculates the share of profit for a partner in a partnership --/
def calculate_share (investment_a investment_b investment_c profit_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let total_profit := (profit_b * total_investment) / investment_b
  (investment_a * total_profit) / total_investment

theorem partner_a_share :
  let investment_a : ℚ := 7000
  let investment_b : ℚ := 11000
  let investment_c : ℚ := 18000
  let profit_b : ℚ := 2200
  calculate_share investment_a investment_b investment_c profit_b = 1400 := by
  sorry

#eval calculate_share 7000 11000 18000 2200

end NUMINAMATH_CALUDE_partner_a_share_l3162_316281


namespace NUMINAMATH_CALUDE_counterexample_exists_l3162_316203

theorem counterexample_exists : ∃ p : ℕ, 
  Nat.Prime p ∧ 
  Odd p ∧ 
  ¬(Nat.Prime (p^2 - 2) ∧ Odd (p^2 - 2)) ∧ 
  p = 11 :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3162_316203


namespace NUMINAMATH_CALUDE_students_present_l3162_316247

theorem students_present (total : ℕ) (absent_percent : ℚ) (present : ℕ) : 
  total = 50 → 
  absent_percent = 12 / 100 → 
  present = total - (total * (absent_percent : ℚ)).floor → 
  present = 44 := by
sorry

end NUMINAMATH_CALUDE_students_present_l3162_316247


namespace NUMINAMATH_CALUDE_water_removal_for_concentration_l3162_316227

theorem water_removal_for_concentration (initial_volume : ℝ) (final_concentration : ℝ) 
  (water_removed : ℝ) : 
  initial_volume = 21 ∧ 
  final_concentration = 60 ∧ 
  water_removed = 7 → 
  water_removed = initial_volume - (initial_volume * (initial_volume * final_concentration) / 
    (100 * (initial_volume - water_removed))) / 100 :=
by sorry

end NUMINAMATH_CALUDE_water_removal_for_concentration_l3162_316227


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l3162_316221

/-- Given two concentric circles with radii a and b (a > b), 
    if the area of the ring between them is 16π,
    then the length of a chord of the larger circle 
    that is tangent to the smaller circle is 8. -/
theorem chord_length_concentric_circles 
  (a b : ℝ) (h1 : a > b) (h2 : a^2 - b^2 = 16) : 
  ∃ c : ℝ, c = 8 ∧ c^2 = 4 * (a^2 - b^2) := by
sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l3162_316221


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l3162_316289

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical shift
def vertical_shift : ℝ := 5

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola x + vertical_shift

-- Theorem stating that the shifted parabola is equivalent to y = x^2 + 5
theorem shifted_parabola_equation :
  ∀ x : ℝ, shifted_parabola x = x^2 + 5 := by sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l3162_316289
