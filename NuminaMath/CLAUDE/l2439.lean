import Mathlib

namespace NUMINAMATH_CALUDE_max_shadow_area_l2439_243974

/-- Regular tetrahedron with edge length a -/
structure Tetrahedron where
  a : ℝ
  a_pos : a > 0

/-- Cube with edge length a -/
structure Cube where
  a : ℝ
  a_pos : a > 0

/-- The maximum shadow area of a regular tetrahedron and a cube -/
theorem max_shadow_area 
  (t : Tetrahedron) 
  (c : Cube) 
  (light_zenith : True) -- Light source is directly above
  : 
  (∃ (shadow_area_tetra : ℝ), 
    shadow_area_tetra ≤ t.a^2 / 2 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_tetra) ∧
  (∃ (shadow_area_cube : ℝ), 
    shadow_area_cube ≤ c.a^2 * Real.sqrt 3 / 3 ∧ 
    ∀ (other_area : ℝ), other_area ≤ shadow_area_cube) :=
sorry

end NUMINAMATH_CALUDE_max_shadow_area_l2439_243974


namespace NUMINAMATH_CALUDE_problem_solution_l2439_243953

def f (x : ℝ) : ℝ := |x| - |2*x - 1|

def M : Set ℝ := {x | f x > -1}

theorem problem_solution :
  (M = Set.Ioo 0 2) ∧
  (∀ a ∈ M,
    (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
    (a = 1 → a^2 - a + 1 = 1/a) ∧
    (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2439_243953


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2439_243959

/-- The equation of a line perpendicular to x - 3y + 2 = 0 and passing through (1, -2) -/
theorem perpendicular_line_equation :
  let l₁ : ℝ → ℝ → Prop := λ x y => x - 3 * y + 2 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => 3 * x + y - 1 = 0
  let P : ℝ × ℝ := (1, -2)
  (∀ x y, l₁ x y → (3 * x + y = 0 → False)) ∧ 
  l₂ P.1 P.2 ∧
  ∀ x y, l₂ x y → (x - 3 * y = 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2439_243959


namespace NUMINAMATH_CALUDE_wyatt_remaining_money_l2439_243976

/-- Calculates the remaining money after shopping, given the initial amount,
    costs of items, quantities, and discount rate. -/
def remaining_money (initial_amount : ℚ)
                    (bread_cost loaves orange_juice_cost cartons
                     cookie_cost boxes apple_cost pounds
                     chocolate_cost bars : ℚ)
                    (discount_rate : ℚ) : ℚ :=
  let total_cost := bread_cost * loaves +
                    orange_juice_cost * cartons +
                    cookie_cost * boxes +
                    apple_cost * pounds +
                    chocolate_cost * bars
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that Wyatt has $127.60 left after shopping -/
theorem wyatt_remaining_money :
  remaining_money 200
                  6.50 5 3.25 4 2.10 7 1.75 3 2.50 6
                  0.10 = 127.60 := by
  sorry

end NUMINAMATH_CALUDE_wyatt_remaining_money_l2439_243976


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2439_243922

/-- Given an ellipse C and a line l, prove that under certain conditions, 
    a point derived from l lies on a specific circle. -/
theorem ellipse_line_intersection (a b : ℝ) (k m : ℝ) : 
  a > b ∧ b > 0 ∧ 
  (a^2 - b^2) / a^2 = 3 / 4 ∧
  b = 1 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧
    x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    (y₁ * x₂) / (x₁ * y₂) = 5 / 4) →
  m^2 + k^2 = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2439_243922


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l2439_243983

-- Define the GCF (Greatest Common Factor) function
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the LCM (Least Common Multiple) function
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_of_lcms : GCF (LCM 15 21) (LCM 10 14) = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l2439_243983


namespace NUMINAMATH_CALUDE_incorrect_inequality_l2439_243934

theorem incorrect_inequality : ¬((-5.2 : ℚ) > -5.1) := by sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l2439_243934


namespace NUMINAMATH_CALUDE_b_8_equals_162_l2439_243900

/-- Given sequences {aₙ} and {bₙ} satisfying the specified conditions, b₈ equals 162 -/
theorem b_8_equals_162 (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a n) * (a (n + 1)) = 3^n)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a n) + (a (n + 1)) = b n)
  (h4 : ∀ n : ℕ, n ≥ 1 → (a n)^2 - (b n) * (a n) + 3^n = 0) :
  b 8 = 162 := by
  sorry

end NUMINAMATH_CALUDE_b_8_equals_162_l2439_243900


namespace NUMINAMATH_CALUDE_cubic_root_product_l2439_243932

theorem cubic_root_product (a b c : ℝ) : 
  (3 * a^3 - 9 * a^2 + a - 5 = 0) ∧ 
  (3 * b^3 - 9 * b^2 + b - 5 = 0) ∧ 
  (3 * c^3 - 9 * c^2 + c - 5 = 0) → 
  a * b * c = 5/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_product_l2439_243932


namespace NUMINAMATH_CALUDE_orange_shells_count_l2439_243954

theorem orange_shells_count 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (blue_shells : ℕ)
  (h1 : total_shells = 65)
  (h2 : purple_shells = 13)
  (h3 : pink_shells = 8)
  (h4 : yellow_shells = 18)
  (h5 : blue_shells = 12)
  (h6 : (total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells)) * 100 = total_shells * 35) :
  total_shells - (purple_shells + pink_shells + yellow_shells + blue_shells) = 14 := by
sorry

end NUMINAMATH_CALUDE_orange_shells_count_l2439_243954


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l2439_243949

theorem tangent_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (40 * π / 180) =
  (Real.sin (50 * π / 180) * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.cos (20 * π / 180) * Real.cos (30 * π / 180))) /
  (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l2439_243949


namespace NUMINAMATH_CALUDE_inequality_implies_k_range_l2439_243937

/-- The natural logarithm function -/
noncomputable def f (x : ℝ) : ℝ := Real.log x

/-- The exponential function -/
noncomputable def g (x : ℝ) : ℝ := Real.exp x

/-- The main theorem -/
theorem inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x * f x - k * (x + 1) * f (g (x - 1)) ≤ 0) →
  k ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_implies_k_range_l2439_243937


namespace NUMINAMATH_CALUDE_oranges_for_24_apples_value_l2439_243958

/-- The number of oranges that can be bought for the price of 24 apples -/
def oranges_for_24_apples (apple_price banana_price cucumber_price orange_price : ℚ) : ℚ :=
  24 * apple_price / orange_price

/-- Theorem stating the number of oranges that can be bought for the price of 24 apples -/
theorem oranges_for_24_apples_value
  (h1 : 12 * apple_price = 6 * banana_price)
  (h2 : 3 * banana_price = 5 * cucumber_price)
  (h3 : 2 * cucumber_price = orange_price)
  : oranges_for_24_apples apple_price banana_price cucumber_price orange_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_oranges_for_24_apples_value_l2439_243958


namespace NUMINAMATH_CALUDE_complex_arithmetic_problem_l2439_243912

theorem complex_arithmetic_problem : ((6^2 - 4^2) + 2)^3 / 2 = 5324 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_problem_l2439_243912


namespace NUMINAMATH_CALUDE_magic_rectangle_unique_z_l2439_243931

/-- Represents a 3x3 magic rectangle with some fixed values --/
structure MagicRectangle where
  x : ℕ+
  y : ℕ+
  u : ℕ+
  z : ℕ+

/-- The sum of each row and column in the magic rectangle --/
def row_col_sum (m : MagicRectangle) : ℕ :=
  3 + m.x + 21

/-- The magic rectangle property: all rows and columns have the same sum --/
def is_magic_rectangle (m : MagicRectangle) : Prop :=
  (row_col_sum m = m.y + 25 + m.z) ∧
  (row_col_sum m = 15 + m.u + 4) ∧
  (row_col_sum m = 3 + m.y + 15) ∧
  (row_col_sum m = m.x + 25 + m.u) ∧
  (row_col_sum m = 21 + m.z + 4)

theorem magic_rectangle_unique_z :
  ∀ m : MagicRectangle, is_magic_rectangle m → m.z = 20 :=
by sorry

end NUMINAMATH_CALUDE_magic_rectangle_unique_z_l2439_243931


namespace NUMINAMATH_CALUDE_incorrect_combination_not_equivalent_l2439_243997

-- Define the polynomial
def original_polynomial (a b : ℚ) : ℚ := 2 * a * b - 4 * a^2 - 5 * a * b + 9 * a^2

-- Define the incorrect combination
def incorrect_combination (a b : ℚ) : ℚ := (2 * a * b - 5 * a * b) - (4 * a^2 + 9 * a^2)

-- Theorem stating that the incorrect combination is not equivalent to the original polynomial
theorem incorrect_combination_not_equivalent :
  ∃ a b : ℚ, original_polynomial a b ≠ incorrect_combination a b :=
sorry

end NUMINAMATH_CALUDE_incorrect_combination_not_equivalent_l2439_243997


namespace NUMINAMATH_CALUDE_hockey_season_length_l2439_243957

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The total number of hockey games in the season -/
def total_games : ℕ := 182

/-- The number of months in the hockey season -/
def season_length : ℕ := total_games / games_per_month

theorem hockey_season_length :
  season_length = 14 :=
sorry

end NUMINAMATH_CALUDE_hockey_season_length_l2439_243957


namespace NUMINAMATH_CALUDE_base_12_remainder_l2439_243948

/-- Converts a base-12 number to base-10 --/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The problem statement --/
theorem base_12_remainder (digits : List Nat) (h : digits = [3, 4, 7, 2]) :
  base12ToBase10 digits % 10 = 5 := by
  sorry

#eval base12ToBase10 [3, 4, 7, 2]

end NUMINAMATH_CALUDE_base_12_remainder_l2439_243948


namespace NUMINAMATH_CALUDE_simplify_expression_l2439_243961

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2439_243961


namespace NUMINAMATH_CALUDE_peter_has_25_candies_l2439_243996

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Peter has -/
def peter_candies : ℕ := shared_candies * 3 - mark_candies - john_candies

theorem peter_has_25_candies : peter_candies = 25 := by
  sorry

end NUMINAMATH_CALUDE_peter_has_25_candies_l2439_243996


namespace NUMINAMATH_CALUDE_opposite_sign_quadratic_solution_l2439_243965

theorem opposite_sign_quadratic_solution :
  ∀ m n : ℝ,
  (|2*m + n| + Real.sqrt (3*n + 12) = 0) →
  (m = 2 ∧ n = -4) ∧
  (∀ x : ℝ, m*x^2 + 4*n*x - 2 = 0 ↔ x = 4 + Real.sqrt 17 ∨ x = 4 - Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_quadratic_solution_l2439_243965


namespace NUMINAMATH_CALUDE_negation_of_all_divisible_by_seven_are_odd_l2439_243913

theorem negation_of_all_divisible_by_seven_are_odd :
  (¬ ∀ n : ℤ, 7 ∣ n → Odd n) ↔ (∃ n : ℤ, 7 ∣ n ∧ ¬ Odd n) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_divisible_by_seven_are_odd_l2439_243913


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2439_243908

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((5432 + y) % 5 = 0 ∧ (5432 + y) % 6 = 0 ∧ (5432 + y) % 7 = 0 ∧ (5432 + y) % 11 = 0 ∧ (5432 + y) % 13 = 0)) ∧
  ((5432 + x) % 5 = 0 ∧ (5432 + x) % 6 = 0 ∧ (5432 + x) % 7 = 0 ∧ (5432 + x) % 11 = 0 ∧ (5432 + x) % 13 = 0) →
  x = 24598 :=
by sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2439_243908


namespace NUMINAMATH_CALUDE_cylinder_section_volume_l2439_243981

/-- Represents a cylinder -/
structure Cylinder where
  base_area : ℝ
  height : ℝ

/-- Represents a plane cutting the cylinder -/
structure CuttingPlane where
  not_parallel_to_base : Bool
  not_intersect_base : Bool

/-- Represents the section of the cylinder cut by the plane -/
structure CylinderSection where
  cylinder : Cylinder
  cutting_plane : CuttingPlane
  segment_height : ℝ

/-- The volume of a cylinder section -/
def section_volume (s : CylinderSection) : ℝ :=
  s.cylinder.base_area * s.segment_height

theorem cylinder_section_volume 
  (s : CylinderSection) 
  (h1 : s.cutting_plane.not_parallel_to_base = true) 
  (h2 : s.cutting_plane.not_intersect_base = true) : 
  ∃ (v : ℝ), v = section_volume s :=
sorry

end NUMINAMATH_CALUDE_cylinder_section_volume_l2439_243981


namespace NUMINAMATH_CALUDE_checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l2439_243943

/-- A checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- The property of a checkerboard being completely coverable by dominoes -/
def is_coverable (board : Checkerboard) : Prop :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating that a checkerboard is coverable iff its area is even -/
theorem checkerboard_coverable_iff_even_area (board : Checkerboard) :
  is_coverable board ↔ Even (board.rows * board.cols) :=
sorry

/-- Function to check if a checkerboard is coverable -/
def check_coverable (board : Checkerboard) : Bool :=
  (board.rows * board.cols) % 2 = 0

/-- Theorem stating which of the given checkerboards are not coverable -/
theorem uncoverable_checkerboards :
  let boards := [
    Checkerboard.mk 4 4,
    Checkerboard.mk 5 5,
    Checkerboard.mk 5 7,
    Checkerboard.mk 6 6,
    Checkerboard.mk 7 3
  ]
  (boards.filter (λ b => ¬check_coverable b)).map (λ b => (b.rows, b.cols)) =
    [(5, 5), (5, 7), (7, 3)] :=
sorry

end NUMINAMATH_CALUDE_checkerboard_coverable_iff_even_area_uncoverable_checkerboards_l2439_243943


namespace NUMINAMATH_CALUDE_inscribed_hexagon_radius_equation_l2439_243939

/-- A hexagon inscribed in a circle with specific side lengths -/
structure InscribedHexagon where
  r : ℝ  -- radius of the circumscribed circle
  side1 : ℝ  -- length of two sides
  side2 : ℝ  -- length of two other sides
  side3 : ℝ  -- length of the remaining two sides
  h1 : side1 = 1
  h2 : side2 = 2
  h3 : side3 = 3

/-- The radius of the circumscribed circle satisfies a specific equation -/
theorem inscribed_hexagon_radius_equation (h : InscribedHexagon) : 
  2 * h.r^3 - 7 * h.r - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_radius_equation_l2439_243939


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l2439_243903

theorem tangent_sum_identity : 
  Real.sqrt 3 * Real.tan (12 * π / 180) + 
  Real.sqrt 3 * Real.tan (18 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (18 * π / 180) = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l2439_243903


namespace NUMINAMATH_CALUDE_equation_satisfaction_l2439_243936

theorem equation_satisfaction (a b c : ℤ) :
  a = c ∧ b - 1 = a →
  a * (a - b) + b * (b - c) + c * (c - a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfaction_l2439_243936


namespace NUMINAMATH_CALUDE_total_medicine_boxes_l2439_243902

def vitamins : ℕ := 472
def supplements : ℕ := 288

theorem total_medicine_boxes : vitamins + supplements = 760 := by
  sorry

end NUMINAMATH_CALUDE_total_medicine_boxes_l2439_243902


namespace NUMINAMATH_CALUDE_root_product_equals_64_l2439_243924

theorem root_product_equals_64 : 
  (256 : ℝ) ^ (1/4 : ℝ) * (64 : ℝ) ^ (1/3 : ℝ) * (16 : ℝ) ^ (1/2 : ℝ) = 64 := by
sorry

end NUMINAMATH_CALUDE_root_product_equals_64_l2439_243924


namespace NUMINAMATH_CALUDE_problem_statement_l2439_243984

-- Define the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

-- State the theorem
theorem problem_statement (a b : ℕ → ℝ) :
  geometric_sequence a →
  a 3 * a 11 = 4 * a 7 →
  arithmetic_sequence b →
  a 7 = b 7 →
  b 5 + b 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2439_243984


namespace NUMINAMATH_CALUDE_fraction_equality_l2439_243994

theorem fraction_equality (a b : ℚ) 
  (h1 : a = 1/2) 
  (h2 : b = 2/3) : 
  (6*a + 18*b) / (12*a + 6*b) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2439_243994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2439_243918

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (h : a₁ = 165 ∧ aₙ = 45 ∧ d = -5) :
  (a₁ - aₙ) / (-d) + 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2439_243918


namespace NUMINAMATH_CALUDE_monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l2439_243925

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp (2 * x) + (1 - x) * Real.exp x + a

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.exp (2 * x) + (1 - x) * Real.exp x - Real.exp x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f' a x * Real.exp (2 - x)

-- Statement 1
theorem monotonicity_of_g :
  let a := Real.exp (-2) / 2
  ∀ x y, x < 2 → y > 2 → g a x > g a 2 ∧ g a 2 < g a y := by sorry

-- Statement 2
theorem range_of_a_for_two_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  0 < a ∧ a < 1 / (2 * Real.exp 1) := by sorry

-- Statement 3
theorem inequality_for_extreme_points :
  ∀ a x₁ x₂, x₁ < x₂ →
  (f' a x₁ = 0 ∧ f' a x₂ = 0) →
  x₁ + 2 * x₂ > 3 := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_of_g_range_of_a_for_two_extreme_points_inequality_for_extreme_points_l2439_243925


namespace NUMINAMATH_CALUDE_unique_divisor_of_18_l2439_243956

def divides (m n : ℕ) : Prop := ∃ k, n = m * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_divisor_of_18 : ∃! a : ℕ, 
  divides 3 a ∧ 
  divides a 18 ∧ 
  Even (sum_of_digits a) := by sorry

end NUMINAMATH_CALUDE_unique_divisor_of_18_l2439_243956


namespace NUMINAMATH_CALUDE_triangle_properties_l2439_243966

/-- Given a triangle ABC where sides a and b are roots of x^2 - 2√3x + 2 = 0,
    and cos(A + B) = 1/2, prove the following properties -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (∃ x y : ℝ, x^2 - 2 * Real.sqrt 3 * x + 2 = 0 ∧ y^2 - 2 * Real.sqrt 3 * y + 2 = 0 ∧ x = a ∧ y = b) →
  Real.cos (A + B) = 1/2 →
  C = Real.pi * 2/3 ∧
  (a^2 + b^2 - 2*a*b*Real.cos C) = 10 ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2439_243966


namespace NUMINAMATH_CALUDE_both_questions_correct_percentage_l2439_243951

theorem both_questions_correct_percentage
  (p_first : ℝ)
  (p_second : ℝ)
  (p_neither : ℝ)
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 :=
by
  sorry

end NUMINAMATH_CALUDE_both_questions_correct_percentage_l2439_243951


namespace NUMINAMATH_CALUDE_x_range_l2439_243911

-- Define the condition
def satisfies_equation (x y : ℝ) : Prop :=
  x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0

-- Define the theorem
theorem x_range (x y : ℝ) (h : satisfies_equation x y) :
  14 - 2 * Real.sqrt 13 ≤ x ∧ x ≤ 14 + 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_x_range_l2439_243911


namespace NUMINAMATH_CALUDE_box_surface_area_l2439_243940

/-- Calculates the surface area of the interior of a box formed by removing square corners from a rectangular sheet and folding up the remaining flaps. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let base_length := sheet_length - 2 * corner_size
  let base_width := sheet_width - 2 * corner_size
  let base_area := base_length * base_width
  let side_area1 := 2 * (base_length * corner_size)
  let side_area2 := 2 * (base_width * corner_size)
  base_area + side_area1 + side_area2

/-- The surface area of the interior of the box is 812 square units. -/
theorem box_surface_area :
  interior_surface_area 28 36 7 = 812 :=
by sorry

end NUMINAMATH_CALUDE_box_surface_area_l2439_243940


namespace NUMINAMATH_CALUDE_linear_system_solution_l2439_243947

theorem linear_system_solution (x y : ℝ) 
  (eq1 : x + 4*y = 5) 
  (eq2 : 5*x + 6*y = 7) : 
  3*x + 5*y = 6 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2439_243947


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l2439_243991

/-- Given a paint mixture with ratio blue:green:white as 3:3:5, 
    prove that using 15 quarts of white paint requires 9 quarts of green paint -/
theorem paint_mixture_ratio (blue green white : ℚ) : 
  blue / green = 1 →
  green / white = 3 / 5 →
  white = 15 →
  green = 9 :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l2439_243991


namespace NUMINAMATH_CALUDE_digit_37_is_1_l2439_243952

/-- The decimal representation of 1/7 has a repeating cycle of 6 digits: 142857 -/
def decimal_rep_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

/-- The length of the repeating cycle in the decimal representation of 1/7 -/
def cycle_length : Nat := decimal_rep_of_one_seventh.length

/-- The 37th digit after the decimal point in the decimal representation of 1/7 -/
def digit_37 : Nat := decimal_rep_of_one_seventh[(37 - 1) % cycle_length]

theorem digit_37_is_1 : digit_37 = 1 := by sorry

end NUMINAMATH_CALUDE_digit_37_is_1_l2439_243952


namespace NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l2439_243942

theorem m_equals_one_sufficient_not_necessary (m : ℝ) :
  (m = 1 → |m| = 1) ∧ (∃ m : ℝ, |m| = 1 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_m_equals_one_sufficient_not_necessary_l2439_243942


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2439_243979

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_x_l2439_243979


namespace NUMINAMATH_CALUDE_altons_rental_cost_l2439_243915

/-- Calculates the weekly rental cost for Alton's business. -/
theorem altons_rental_cost
  (daily_earnings : ℝ)  -- Alton's daily earnings
  (weekly_profit : ℝ)   -- Alton's weekly profit
  (h1 : daily_earnings = 8)  -- Alton earns $8 per day
  (h2 : weekly_profit = 36)  -- Alton's total profit every week is $36
  : ∃ (rental_cost : ℝ), rental_cost = 20 ∧ 7 * daily_earnings - rental_cost = weekly_profit :=
by
  sorry

#check altons_rental_cost

end NUMINAMATH_CALUDE_altons_rental_cost_l2439_243915


namespace NUMINAMATH_CALUDE_jade_transactions_l2439_243905

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 17 →
  jade = 83 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l2439_243905


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2439_243928

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 6 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2439_243928


namespace NUMINAMATH_CALUDE_total_salmon_count_l2439_243993

/-- Represents the count of male and female salmon for a species -/
structure SalmonCount where
  males : Nat
  females : Nat

/-- Calculates the total number of salmon for a given species -/
def totalForSpecies (count : SalmonCount) : Nat :=
  count.males + count.females

/-- The counts for each salmon species -/
def chinookCount : SalmonCount := { males := 451228, females := 164225 }
def sockeyeCount : SalmonCount := { males := 212001, females := 76914 }
def cohoCount : SalmonCount := { males := 301008, females := 111873 }
def pinkCount : SalmonCount := { males := 518001, females := 182945 }
def chumCount : SalmonCount := { males := 230023, females := 81321 }

/-- Theorem stating that the total number of salmon across all species is 2,329,539 -/
theorem total_salmon_count : 
  totalForSpecies chinookCount + 
  totalForSpecies sockeyeCount + 
  totalForSpecies cohoCount + 
  totalForSpecies pinkCount + 
  totalForSpecies chumCount = 2329539 := by
  sorry

end NUMINAMATH_CALUDE_total_salmon_count_l2439_243993


namespace NUMINAMATH_CALUDE_expression_evaluation_l2439_243916

theorem expression_evaluation : 
  (2 + 1/4)^(1/2) - 0.3^0 - 16^(-3/4) = 3/8 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2439_243916


namespace NUMINAMATH_CALUDE_square_route_distance_l2439_243969

/-- Represents a square route with given side length -/
structure SquareRoute where
  side_length : ℝ

/-- Calculates the total distance traveled in a square route -/
def total_distance (route : SquareRoute) : ℝ :=
  4 * route.side_length

/-- Theorem: The total distance traveled in a square route with sides of 2000 km is 8000 km -/
theorem square_route_distance :
  let route := SquareRoute.mk 2000
  total_distance route = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_route_distance_l2439_243969


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2439_243946

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2439_243946


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l2439_243988

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 * n * (n - 2) * (n - 4)) / 3 → n = 18 :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l2439_243988


namespace NUMINAMATH_CALUDE_exponential_equation_comparison_l2439_243977

theorem exponential_equation_comparison
  (c d k n : ℝ)
  (hc : c > 0)
  (hk : k > 0)
  (hd : d ≠ 0)
  (hn : n ≠ 0) :
  (Real.log c / Real.log k < d / n) ↔
  ((1 / d) * Real.log (1 / c) < (1 / n) * Real.log (1 / k)) :=
sorry

end NUMINAMATH_CALUDE_exponential_equation_comparison_l2439_243977


namespace NUMINAMATH_CALUDE_clock_problem_l2439_243941

/-- Represents a time on a 12-hour digital clock -/
structure Time where
  hour : Nat
  minute : Nat
  second : Nat
  deriving Repr

/-- Adds a duration to a given time -/
def addDuration (t : Time) (hours minutes seconds : Nat) : Time :=
  sorry

/-- Calculates the sum of hour, minute, and second values of a time -/
def timeSum (t : Time) : Nat :=
  sorry

theorem clock_problem :
  let initial_time : Time := ⟨3, 0, 0⟩
  let final_time := addDuration initial_time 85 58 30
  final_time = ⟨4, 58, 30⟩ ∧ timeSum final_time = 92 := by sorry

end NUMINAMATH_CALUDE_clock_problem_l2439_243941


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2439_243967

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 24*r^2 + 50*r - 24 = 0 →
  s^3 - 24*s^2 + 50*s - 24 = 0 →
  t^3 - 24*t^2 + 50*t - 24 = 0 →
  (r / (1/r + s*t)) + (s / (1/s + t*r)) + (t / (1/t + r*s)) = 19.04 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2439_243967


namespace NUMINAMATH_CALUDE_train_speed_theorem_l2439_243970

def train_problem (train1_length train2_length : Real) 
                  (train2_speed : Real) 
                  (clear_time : Real) : Prop :=
  let total_length := train1_length + train2_length
  let total_length_km := total_length / 1000
  let clear_time_hours := clear_time / 3600
  let relative_speed := total_length_km / clear_time_hours
  let train1_speed := relative_speed - train2_speed
  train1_speed = 42.0012

theorem train_speed_theorem : 
  train_problem 160 320 30 23.998 := by
  sorry

#check train_speed_theorem

end NUMINAMATH_CALUDE_train_speed_theorem_l2439_243970


namespace NUMINAMATH_CALUDE_students_painting_l2439_243921

theorem students_painting (green red both : ℕ) 
  (h1 : green = 52)
  (h2 : red = 56)
  (h3 : both = 38) :
  green + red - both = 70 := by
  sorry

end NUMINAMATH_CALUDE_students_painting_l2439_243921


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2439_243963

theorem arithmetic_expression_equality : 10 - 9 + 8 * (7 - 6) + 5 * 4 - 3 + 2 - 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2439_243963


namespace NUMINAMATH_CALUDE_inequality_proof_l2439_243923

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2439_243923


namespace NUMINAMATH_CALUDE_normal_price_after_discounts_l2439_243960

theorem normal_price_after_discounts (price : ℝ) : 
  price * (1 - 0.1) * (1 - 0.2) = 144 → price = 200 := by
  sorry

end NUMINAMATH_CALUDE_normal_price_after_discounts_l2439_243960


namespace NUMINAMATH_CALUDE_expression_value_l2439_243989

theorem expression_value (a b : ℝ) (h : 3 * (a - 2) = 2 * (2 * b - 4)) :
  9 * a^2 - 24 * a * b + 16 * b^2 + 25 = 29 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2439_243989


namespace NUMINAMATH_CALUDE_oranges_remaining_proof_l2439_243973

/-- The number of oranges Michaela needs to eat to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both Michaela and Cassandra eat until they're full -/
def remaining_oranges : ℕ := total_oranges - (michaela_oranges + cassandra_oranges)

theorem oranges_remaining_proof : remaining_oranges = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_proof_l2439_243973


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2439_243914

/-- Given a polynomial g(x) = 3x^4 - 20x^3 + 30x^2 - 35x - 75, prove that g(6) = 363 -/
theorem polynomial_evaluation (x : ℝ) :
  let g := fun x => 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75
  g 6 = 363 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2439_243914


namespace NUMINAMATH_CALUDE_inequality_theorem_l2439_243930

theorem inequality_theorem (a b : ℝ) : 
  |2*a - 2| < |a - 4| → |2*b - 2| < |b - 4| → 2*|a + b| < |4 + a*b| := by
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2439_243930


namespace NUMINAMATH_CALUDE_diana_box_capacity_l2439_243917

/-- Represents a box with dimensions and jellybean capacity -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℕ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.height * b.width * b.length

/-- Theorem: A box with triple height, double width, and quadruple length of Bert's box
    that holds 150 jellybeans will hold 3600 jellybeans -/
theorem diana_box_capacity (bert_box : Box)
    (h1 : bert_box.capacity = 150)
    (diana_box : Box)
    (h2 : diana_box.height = 3 * bert_box.height)
    (h3 : diana_box.width = 2 * bert_box.width)
    (h4 : diana_box.length = 4 * bert_box.length) :
    diana_box.capacity = 3600 := by
  sorry


end NUMINAMATH_CALUDE_diana_box_capacity_l2439_243917


namespace NUMINAMATH_CALUDE_no_m_exists_for_equality_necessary_but_not_sufficient_condition_l2439_243995

-- Define set P
def P : Set ℝ := {x : ℝ | x^2 - 8*x - 20 ≤ 0}

-- Define set S as a function of m
def S (m : ℝ) : Set ℝ := {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for part I
theorem no_m_exists_for_equality :
  ¬ ∃ m : ℝ, P = S m :=
sorry

-- Theorem for part II
theorem necessary_but_not_sufficient_condition :
  ∀ m : ℝ, m ≤ 3 → (P ⊆ S m ∧ P ≠ S m) :=
sorry

end NUMINAMATH_CALUDE_no_m_exists_for_equality_necessary_but_not_sufficient_condition_l2439_243995


namespace NUMINAMATH_CALUDE_arrangeStudents_eq_288_l2439_243933

/-- The number of ways to arrange 6 students in a 2x3 grid with constraints -/
def arrangeStudents : ℕ :=
  let totalStudents : ℕ := 6
  let rows : ℕ := 2
  let columns : ℕ := 3
  let positionsForA : ℕ := totalStudents
  let positionsForB : ℕ := 2  -- Not in same row or column as A
  let remainingStudents : ℕ := totalStudents - 2
  positionsForA * positionsForB * Nat.factorial remainingStudents

theorem arrangeStudents_eq_288 : arrangeStudents = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangeStudents_eq_288_l2439_243933


namespace NUMINAMATH_CALUDE_tens_digit_of_17_to_1993_l2439_243992

theorem tens_digit_of_17_to_1993 : ∃ n : ℕ, 17^1993 ≡ 30 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_17_to_1993_l2439_243992


namespace NUMINAMATH_CALUDE_q_factor_change_l2439_243944

theorem q_factor_change (w h z : ℝ) (q : ℝ → ℝ → ℝ → ℝ) 
  (hq : q w h z = 5 * w / (4 * h * z^2)) :
  q (4*w) (2*h) (3*z) = (2/9) * q w h z := by
sorry

end NUMINAMATH_CALUDE_q_factor_change_l2439_243944


namespace NUMINAMATH_CALUDE_exists_factorial_with_124_zeros_l2439_243986

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

/-- There exists a positive integer n such that n! has exactly 124 trailing zeros -/
theorem exists_factorial_with_124_zeros : ∃ n : ℕ, n > 0 ∧ trailingZeros n = 124 := by
  sorry

end NUMINAMATH_CALUDE_exists_factorial_with_124_zeros_l2439_243986


namespace NUMINAMATH_CALUDE_range_of_a_l2439_243978

/-- The exponential function f(x) = (2a - 6)^x is monotonically decreasing on ℝ -/
def P (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (2*a - 6)^x > (2*a - 6)^y

/-- Both real roots of the equation x^2 - 3ax + 2a^2 + 1 = 0 are greater than 3 -/
def Q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 3*a*x + 2*a^2 + 1 = 0 → x > 3

theorem range_of_a (a : ℝ) (h1 : a > 3) (h2 : a ≠ 4) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) ↔ a ≥ 3.5 ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2439_243978


namespace NUMINAMATH_CALUDE_basketball_free_throws_count_l2439_243907

/-- Represents the scoring of a basketball team -/
structure BasketballScore where
  two_points : ℕ
  three_points : ℕ
  free_throws : ℕ

/-- Calculates the total score of a basketball team -/
def total_score (score : BasketballScore) : ℕ :=
  2 * score.two_points + 3 * score.three_points + score.free_throws

/-- Theorem: Given the conditions, the number of free throws is 12 -/
theorem basketball_free_throws_count 
  (score : BasketballScore) 
  (h1 : 3 * score.three_points = 2 * (2 * score.two_points))
  (h2 : score.free_throws = score.two_points + 1)
  (h3 : total_score score = 79) : 
  score.free_throws = 12 := by
  sorry

#check basketball_free_throws_count

end NUMINAMATH_CALUDE_basketball_free_throws_count_l2439_243907


namespace NUMINAMATH_CALUDE_half_day_division_ways_l2439_243962

/-- The number of ways to express 43200 as a product of two positive integers -/
def num_factor_pairs : ℕ := 72

/-- Half a day in seconds -/
def half_day_seconds : ℕ := 43200

theorem half_day_division_ways :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = half_day_seconds) (Finset.product (Finset.range (half_day_seconds + 1)) (Finset.range (half_day_seconds + 1)))).card = num_factor_pairs :=
sorry

end NUMINAMATH_CALUDE_half_day_division_ways_l2439_243962


namespace NUMINAMATH_CALUDE_angle_three_times_complement_l2439_243906

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_three_times_complement_l2439_243906


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2439_243990

/-- A line in the form y - 2 = mx + m passes through the point (-1, 2) for all values of m -/
theorem line_passes_through_fixed_point (m : ℝ) :
  let line := fun (x y : ℝ) => y - 2 = m * x + m
  line (-1) 2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2439_243990


namespace NUMINAMATH_CALUDE_coin_flip_result_l2439_243975

/-- A fair coin is flipped 100 times with 48 heads. -/
structure CoinFlip where
  total_flips : ℕ
  heads_count : ℕ
  is_fair : Bool

/-- The frequency of heads in a coin flip experiment. -/
def frequency (cf : CoinFlip) : ℚ :=
  cf.heads_count / cf.total_flips

/-- The theoretical probability of heads for a fair coin. -/
def fair_coin_probability : ℚ := 1 / 2

/-- Theorem stating the frequency and probability for the given coin flip experiment. -/
theorem coin_flip_result (cf : CoinFlip) 
  (h1 : cf.total_flips = 100)
  (h2 : cf.heads_count = 48)
  (h3 : cf.is_fair = true) :
  frequency cf = 48 / 100 ∧ fair_coin_probability = 1 / 2 := by
  sorry

#eval (48 : ℚ) / 100  -- Should output 0.48
#eval (1 : ℚ) / 2     -- Should output 0.5

end NUMINAMATH_CALUDE_coin_flip_result_l2439_243975


namespace NUMINAMATH_CALUDE_new_teacher_age_proof_l2439_243968

/-- The number of teachers initially -/
def initial_teachers : ℕ := 20

/-- The average age of initial teachers -/
def initial_average_age : ℕ := 49

/-- The number of teachers after a new teacher joins -/
def final_teachers : ℕ := 21

/-- The new average age after a new teacher joins -/
def final_average_age : ℕ := 48

/-- The age of the new teacher -/
def new_teacher_age : ℕ := 28

theorem new_teacher_age_proof :
  initial_teachers * initial_average_age + new_teacher_age = final_teachers * final_average_age :=
sorry

end NUMINAMATH_CALUDE_new_teacher_age_proof_l2439_243968


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2439_243935

theorem cafeteria_apples (initial_apples : ℕ) (bought_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 17)
  (h2 : bought_apples = 23)
  (h3 : final_apples = 38) :
  initial_apples - (initial_apples - (final_apples - bought_apples)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2439_243935


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l2439_243998

theorem fraction_addition_simplification : (2 : ℚ) / 5 + (3 : ℚ) / 15 = (3 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l2439_243998


namespace NUMINAMATH_CALUDE_village_foods_lettuce_price_l2439_243938

/-- The price of a head of lettuce at Village Foods -/
def lettuce_price : ℝ := 1

/-- The number of customers per month -/
def customers_per_month : ℕ := 500

/-- The number of lettuce heads each customer buys -/
def lettuce_per_customer : ℕ := 2

/-- The number of tomatoes each customer buys -/
def tomatoes_per_customer : ℕ := 4

/-- The price of each tomato -/
def tomato_price : ℝ := 0.5

/-- The total monthly sales of lettuce and tomatoes -/
def total_monthly_sales : ℝ := 2000

theorem village_foods_lettuce_price :
  lettuce_price = 1 ∧
  customers_per_month * lettuce_per_customer * lettuce_price +
  customers_per_month * tomatoes_per_customer * tomato_price = total_monthly_sales :=
by sorry

end NUMINAMATH_CALUDE_village_foods_lettuce_price_l2439_243938


namespace NUMINAMATH_CALUDE_total_cookies_count_l2439_243950

/-- Given 286 bags of cookies with 452 cookies in each bag, 
    prove that the total number of cookies is 129,272. -/
theorem total_cookies_count (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 286) (h2 : cookies_per_bag = 452) : 
  bags * cookies_per_bag = 129272 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_count_l2439_243950


namespace NUMINAMATH_CALUDE_expression_simplification_l2439_243999

theorem expression_simplification (x : ℝ) : 
  3 * x - 5 * (2 + x) + 6 * (2 - x) - 7 * (2 + 3 * x) = -29 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2439_243999


namespace NUMINAMATH_CALUDE_late_fisherman_arrival_day_l2439_243901

/-- Represents the day of the week when the late fisherman arrived -/
inductive ArrivalDay
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Calculates the number of days the late fisherman fished -/
def daysLateArrivalFished (d : ArrivalDay) : Nat :=
  match d with
  | .Monday => 5
  | .Tuesday => 4
  | .Wednesday => 3
  | .Thursday => 2
  | .Friday => 1

theorem late_fisherman_arrival_day :
  ∃ (n : Nat) (d : ArrivalDay),
    n > 0 ∧
    50 * n + 10 * (daysLateArrivalFished d) = 370 ∧
    d = ArrivalDay.Thursday :=
by sorry

end NUMINAMATH_CALUDE_late_fisherman_arrival_day_l2439_243901


namespace NUMINAMATH_CALUDE_cards_per_page_l2439_243929

theorem cards_per_page (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 3) 
  (h2 : old_cards = 9) 
  (h3 : pages = 4) : 
  (new_cards + old_cards) / pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_per_page_l2439_243929


namespace NUMINAMATH_CALUDE_abs_two_over_one_plus_i_l2439_243909

theorem abs_two_over_one_plus_i : Complex.abs (2 / (1 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_over_one_plus_i_l2439_243909


namespace NUMINAMATH_CALUDE_company_c_cheapest_l2439_243955

-- Define the pricing structures for each company
def company_a_cost (miles : ℝ) : ℝ :=
  2.10 + 0.40 * (miles * 5 - 1)

def company_b_cost (miles : ℝ) : ℝ :=
  3.00 + 0.50 * (miles * 4 - 1)

def company_c_cost (miles : ℝ) : ℝ :=
  1.50 * miles + 2.00

-- Theorem statement
theorem company_c_cheapest :
  let journey_length : ℝ := 8
  company_c_cost journey_length < company_a_cost journey_length ∧
  company_c_cost journey_length < company_b_cost journey_length :=
by sorry

end NUMINAMATH_CALUDE_company_c_cheapest_l2439_243955


namespace NUMINAMATH_CALUDE_riverside_denial_rate_l2439_243987

theorem riverside_denial_rate (total_kids : ℕ) (riverside_kids : ℕ) (westside_kids : ℕ) (mountaintop_kids : ℕ)
  (westside_denial_rate : ℚ) (mountaintop_denial_rate : ℚ) (kids_admitted : ℕ) :
  total_kids = riverside_kids + westside_kids + mountaintop_kids →
  total_kids = 260 →
  riverside_kids = 120 →
  westside_kids = 90 →
  mountaintop_kids = 50 →
  westside_denial_rate = 7/10 →
  mountaintop_denial_rate = 1/2 →
  kids_admitted = 148 →
  (riverside_kids - (total_kids - kids_admitted - (westside_denial_rate * westside_kids).num
    - (mountaintop_denial_rate * mountaintop_kids).num)) / riverside_kids = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_riverside_denial_rate_l2439_243987


namespace NUMINAMATH_CALUDE_special_function_property_l2439_243919

/-- A function from positive reals to positive reals satisfying the given condition -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x + f y) ≥ f (x + y) + f y)

/-- The main theorem -/
theorem special_function_property (f : ℝ → ℝ) (h : SpecialFunction f) :
  ∀ x > 0, f x > x :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l2439_243919


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l2439_243982

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_expression_equals_one :
  lg 2 * lg 2 + lg 2 * lg 5 + lg 5 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l2439_243982


namespace NUMINAMATH_CALUDE_equation_solution_l2439_243945

theorem equation_solution :
  ∃! x : ℝ, (3 / (x - 3) = 4 / (x - 4)) ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2439_243945


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2439_243972

/-- Given a parabola and two points on it, prove the y-intercept of the line through these points -/
theorem parabola_line_intersection (A B : ℝ × ℝ) (a : ℝ) : 
  (A.1^2 = A.2) →  -- A is on the parabola
  (B.1^2 = B.2) →  -- B is on the parabola
  (A.1 < 0) →  -- A is on the left side of y-axis
  (B.1 > 0) →  -- B is on the right side of y-axis
  (∃ k : ℝ, A.2 = k * A.1 + a ∧ B.2 = k * B.1 + a) →  -- Line AB has equation y = kx + a
  (A.1 * B.1 + A.2 * B.2 > 0) →  -- ∠AOB is acute
  a > 1 := by
sorry


end NUMINAMATH_CALUDE_parabola_line_intersection_l2439_243972


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_l2439_243910

theorem quadratic_equation_sum (x q t : ℝ) : 
  (9 * x^2 - 36 * x - 81 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_l2439_243910


namespace NUMINAMATH_CALUDE_square_perimeter_equal_area_rectangle_l2439_243980

theorem square_perimeter_equal_area_rectangle (l w : ℝ) (h1 : l = 1024) (h2 : w = 1) :
  let rectangle_area := l * w
  let square_side := (rectangle_area).sqrt
  let square_perimeter := 4 * square_side
  square_perimeter = 128 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_equal_area_rectangle_l2439_243980


namespace NUMINAMATH_CALUDE_sofia_running_time_l2439_243904

/-- The time Sofia takes to complete 8 laps on a track with given conditions -/
theorem sofia_running_time (laps : ℕ) (track_length : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ)
  (h1 : laps = 8)
  (h2 : track_length = 300)
  (h3 : first_half_speed = 5)
  (h4 : second_half_speed = 6) :
  let time_per_lap := track_length / (2 * first_half_speed) + track_length / (2 * second_half_speed)
  let total_time := laps * time_per_lap
  total_time = 440 := by sorry

#eval (7 * 60 + 20 : ℕ) -- Evaluates to 440, confirming 7 minutes and 20 seconds

end NUMINAMATH_CALUDE_sofia_running_time_l2439_243904


namespace NUMINAMATH_CALUDE_perimeter_of_rectangle_l2439_243985

def rhombus_in_rectangle (WE XF EG FH : ℝ) : Prop :=
  WE = 10 ∧ XF = 25 ∧ EG = 20 ∧ FH = 50

theorem perimeter_of_rectangle (WE XF EG FH : ℝ) 
  (h : rhombus_in_rectangle WE XF EG FH) : 
  ∃ (perimeter : ℝ), perimeter = 53 * Real.sqrt 29 - 73 := by
  sorry

#check perimeter_of_rectangle

end NUMINAMATH_CALUDE_perimeter_of_rectangle_l2439_243985


namespace NUMINAMATH_CALUDE_carlos_singles_percentage_l2439_243971

/-- Represents the hit statistics for Carlos during the baseball season -/
structure HitStats :=
  (total_hits : ℕ)
  (home_runs : ℕ)
  (triples : ℕ)
  (doubles : ℕ)
  (strikeouts : ℕ)

/-- Calculates the percentage of singles among successful hits -/
def percentage_singles (stats : HitStats) : ℚ :=
  let successful_hits := stats.total_hits - stats.strikeouts
  let non_single_hits := stats.home_runs + stats.triples + stats.doubles
  let singles := successful_hits - non_single_hits
  (singles : ℚ) / (successful_hits : ℚ) * 100

/-- The hit statistics for Carlos -/
def carlos_stats : HitStats :=
  { total_hits := 50
  , home_runs := 4
  , triples := 2
  , doubles := 8
  , strikeouts := 6 }

/-- Theorem stating that the percentage of singles for Carlos is approximately 68.18% -/
theorem carlos_singles_percentage :
  abs (percentage_singles carlos_stats - 68.18) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_carlos_singles_percentage_l2439_243971


namespace NUMINAMATH_CALUDE_equation_solutions_l2439_243927

def equation (x : ℝ) : Prop :=
  6 / (Real.sqrt (x - 8) - 9) + 1 / (Real.sqrt (x - 8) - 4) + 
  7 / (Real.sqrt (x - 8) + 4) + 12 / (Real.sqrt (x - 8) + 9) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {17, 44} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2439_243927


namespace NUMINAMATH_CALUDE_candy_sales_l2439_243964

theorem candy_sales (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_sales_l2439_243964


namespace NUMINAMATH_CALUDE_journey_duration_l2439_243920

/-- Represents the journey of a spaceship --/
structure SpaceshipJourney where
  initial_travel : ℕ
  initial_break : ℕ
  second_travel : ℕ
  second_break : ℕ
  subsequent_travel : ℕ
  subsequent_break : ℕ
  total_non_moving : ℕ

/-- Calculates the total journey time for a spaceship --/
def total_journey_time (j : SpaceshipJourney) : ℕ :=
  let remaining_break := j.total_non_moving - j.initial_break - j.second_break
  let subsequent_segments := remaining_break / j.subsequent_break
  j.initial_travel + j.initial_break + j.second_travel + j.second_break +
  subsequent_segments * (j.subsequent_travel + j.subsequent_break)

/-- Theorem stating that the journey takes 72 hours --/
theorem journey_duration (j : SpaceshipJourney)
  (h1 : j.initial_travel = 10)
  (h2 : j.initial_break = 3)
  (h3 : j.second_travel = 10)
  (h4 : j.second_break = 1)
  (h5 : j.subsequent_travel = 11)
  (h6 : j.subsequent_break = 1)
  (h7 : j.total_non_moving = 8) :
  total_journey_time j = 72 := by
  sorry


end NUMINAMATH_CALUDE_journey_duration_l2439_243920


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2439_243926

/-- Ellipse with foci on y-axis and center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line intersecting the ellipse -/
structure Line where
  k : ℝ
  m : ℝ

/-- Definition of the problem setup -/
def EllipseProblem (E : Ellipse) (l : Line) : Prop :=
  -- Eccentricity is √3/2
  E.a / (E.a ^ 2 - E.b ^ 2).sqrt = 2 / Real.sqrt 3 ∧
  -- Perimeter of quadrilateral is 4√5
  4 * (E.a ^ 2 + E.b ^ 2).sqrt = 4 * Real.sqrt 5 ∧
  -- Line intersects ellipse at two points
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (l.k * x₁ + l.m) ^ 2 / (4 : ℝ) + x₁ ^ 2 = 1 ∧
    (l.k * x₂ + l.m) ^ 2 / (4 : ℝ) + x₂ ^ 2 = 1 ∧
  -- AP = 3PB condition
  x₁ = -3 * x₂

/-- Main theorem to prove -/
theorem ellipse_intersection_theorem (E : Ellipse) (l : Line) 
  (h : EllipseProblem E l) : 
  1 < l.m ^ 2 ∧ l.m ^ 2 < 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l2439_243926
