import Mathlib

namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_175_l1488_148836

theorem smallest_prime_factor_of_175 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 175 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 175 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_175_l1488_148836


namespace NUMINAMATH_CALUDE_solve_equation_l1488_148843

-- Define the operation "*"
def star (a b : ℝ) : ℝ := 2 * a - b

-- Theorem statement
theorem solve_equation (x : ℝ) (h : star x (star 2 1) = 3) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1488_148843


namespace NUMINAMATH_CALUDE_water_depth_multiple_l1488_148887

/-- Given Dean's height and the water depth, prove that the multiple of Dean's height
    representing the water depth is 10. -/
theorem water_depth_multiple (dean_height water_depth : ℝ) 
  (h1 : dean_height = 6)
  (h2 : water_depth = 60) :
  water_depth / dean_height = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_multiple_l1488_148887


namespace NUMINAMATH_CALUDE_math_test_questions_l1488_148833

/-- Proves that the total number of questions in a math test is 60 -/
theorem math_test_questions : ∃ N : ℕ,
  (N : ℚ) * (80 : ℚ) / 100 + 35 - N / 2 = N - 7 ∧
  N = 60 := by
  sorry

end NUMINAMATH_CALUDE_math_test_questions_l1488_148833


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1488_148819

theorem complex_modulus_equality (x y : ℝ) (i : ℂ) (h : i * i = -1) 
  (eq : x + 3 * i = 2 + y * i) : Complex.abs (x + y * i) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1488_148819


namespace NUMINAMATH_CALUDE_f_2009_equals_one_l1488_148852

-- Define the function f
axiom f : ℝ → ℝ

-- Define the conditions
axiom func_prop : ∀ x y : ℝ, f (x * y) = f x * f y
axiom f0_nonzero : f 0 ≠ 0

-- State the theorem
theorem f_2009_equals_one : f 2009 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_equals_one_l1488_148852


namespace NUMINAMATH_CALUDE_triathlon_speed_l1488_148837

/-- Triathlon problem -/
theorem triathlon_speed (total_time swim_dist swim_speed run_dist run_speed rest_time bike_dist : ℝ) 
  (h_total : total_time = 2.25)
  (h_swim : swim_dist = 0.5)
  (h_swim_speed : swim_speed = 2)
  (h_run : run_dist = 4)
  (h_run_speed : run_speed = 8)
  (h_rest : rest_time = 1/6)
  (h_bike : bike_dist = 20) :
  bike_dist / (total_time - (swim_dist / swim_speed + run_dist / run_speed + rest_time)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_speed_l1488_148837


namespace NUMINAMATH_CALUDE_randy_tower_blocks_l1488_148857

/-- 
Given:
- Randy has 90 blocks in total
- He uses 89 blocks to build a house
- He uses some blocks to build a tower
- He used 26 more blocks for the house than for the tower

Prove that Randy used 63 blocks to build the tower.
-/
theorem randy_tower_blocks : 
  ∀ (total house tower : ℕ),
  total = 90 →
  house = 89 →
  house = tower + 26 →
  tower = 63 := by
sorry

end NUMINAMATH_CALUDE_randy_tower_blocks_l1488_148857


namespace NUMINAMATH_CALUDE_erik_money_left_l1488_148891

-- Define the problem parameters
def initial_money : ℚ := 86
def bread_price : ℚ := 3
def juice_price : ℚ := 6
def eggs_price : ℚ := 4
def chocolate_price : ℚ := 2
def apples_price : ℚ := 1.25
def grapes_price : ℚ := 2.50

def bread_quantity : ℕ := 3
def juice_quantity : ℕ := 3
def eggs_quantity : ℕ := 2
def chocolate_quantity : ℕ := 5
def apples_quantity : ℚ := 4
def grapes_quantity : ℚ := 1.5

def bread_eggs_discount : ℚ := 0.10
def other_items_discount : ℚ := 0.05
def sales_tax_rate : ℚ := 0.06

-- Define the theorem
theorem erik_money_left : 
  let total_cost := bread_price * bread_quantity + juice_price * juice_quantity + 
                    eggs_price * eggs_quantity + chocolate_price * chocolate_quantity + 
                    apples_price * apples_quantity + grapes_price * grapes_quantity
  let bread_eggs_cost := bread_price * bread_quantity + eggs_price * eggs_quantity
  let other_items_cost := total_cost - bread_eggs_cost
  let discounted_cost := total_cost - (bread_eggs_cost * bread_eggs_discount) - 
                         (other_items_cost * other_items_discount)
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  initial_money - final_cost = 32.78 := by
  sorry


end NUMINAMATH_CALUDE_erik_money_left_l1488_148891


namespace NUMINAMATH_CALUDE_ic_train_speed_ratio_l1488_148806

theorem ic_train_speed_ratio :
  ∀ (u v : ℝ), u > 0 → v > 0 →
  (u / v = ((u + v) / (u - v))) →
  (u / v = 1 + Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ic_train_speed_ratio_l1488_148806


namespace NUMINAMATH_CALUDE_centroid_of_V_l1488_148818

-- Define the region V
def V : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs p.1 ≤ p.2 ∧ p.2 ≤ abs p.1 + 3 ∧ p.2 ≤ 4}

-- Define the centroid of a region
def centroid (S : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem centroid_of_V :
  centroid V = (0, 2.31) := by
  sorry

end NUMINAMATH_CALUDE_centroid_of_V_l1488_148818


namespace NUMINAMATH_CALUDE_g_lower_bound_l1488_148844

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem g_lower_bound : ∀ x > 0, g x > 4/3 := by
  sorry

end NUMINAMATH_CALUDE_g_lower_bound_l1488_148844


namespace NUMINAMATH_CALUDE_max_area_at_150_l1488_148849

/-- Represents a rectangular pasture with a fence on three sides and a barn on the fourth side. -/
structure Pasture where
  fenceLength : ℝ  -- Total length of fence available
  barnLength : ℝ   -- Length of the barn side

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn. -/
def Pasture.area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fenceLength - 2 * x)

/-- Theorem stating that the maximum area of the pasture occurs when the side parallel to the barn is 150 feet. -/
theorem max_area_at_150 (p : Pasture) (h1 : p.fenceLength = 300) (h2 : p.barnLength = 350) :
  ∃ (x : ℝ), x > 0 ∧ x < p.barnLength ∧
  (∀ (y : ℝ), y > 0 → y < p.barnLength → p.area x ≥ p.area y) ∧
  p.fenceLength - 2 * x = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_area_at_150_l1488_148849


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1488_148810

/-- The trajectory of the midpoint of a chord on a circle -/
theorem midpoint_trajectory (k x y : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Line equation
    (k * x₁ - y₁ + 1 = 0) ∧ (k * x₂ - y₂ + 1 = 0) ∧
    -- Circle equation
    (x₁^2 + y₁^2 = 1) ∧ (x₂^2 + y₂^2 = 1) ∧
    -- (x, y) is the midpoint of (x₁, y₁) and (x₂, y₂)
    (x = (x₁ + x₂) / 2) ∧ (y = (y₁ + y₂) / 2)) →
  x^2 + y^2 - y = 0 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1488_148810


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1488_148847

theorem system_of_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 4 * y = -7) ∧ 
    (6 * x - 5 * y = 9) ∧ 
    (x = 71 / 9) ∧ 
    (y = 23 / 3) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1488_148847


namespace NUMINAMATH_CALUDE_candy_box_capacity_l1488_148801

theorem candy_box_capacity (dan_capacity : ℕ) (dan_height dan_width dan_length : ℝ) 
  (ella_height ella_width ella_length : ℝ) :
  dan_capacity = 150 →
  ella_height = 3 * dan_height →
  ella_width = 3 * dan_width →
  ella_length = 3 * dan_length →
  ⌊(ella_height * ella_width * ella_length) / (dan_height * dan_width * dan_length) * dan_capacity⌋ = 4050 :=
by sorry

end NUMINAMATH_CALUDE_candy_box_capacity_l1488_148801


namespace NUMINAMATH_CALUDE_expression_value_l1488_148805

theorem expression_value (z : ℝ) : (1 : ℝ)^(6*z-3) / (7⁻¹ + 4⁻¹) = 28/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1488_148805


namespace NUMINAMATH_CALUDE_tiffany_treasures_l1488_148840

theorem tiffany_treasures (points_per_treasure : ℕ) (first_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 6 →
  first_level_treasures = 3 →
  total_score = 48 →
  (total_score - points_per_treasure * first_level_treasures) / points_per_treasure = 5 :=
by sorry

end NUMINAMATH_CALUDE_tiffany_treasures_l1488_148840


namespace NUMINAMATH_CALUDE_triangle_ratio_l1488_148884

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*sin(A) - b*sin(B) = 4c*sin(C) and cos(A) = -1/4, then b/c = 6 -/
theorem triangle_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a * Real.sin A - b * Real.sin B = 4 * c * Real.sin C →
  Real.cos A = -1/4 →
  b / c = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1488_148884


namespace NUMINAMATH_CALUDE_volume_of_solid_l1488_148803

-- Define the region S
def S : Set (ℝ × ℝ) :=
  {(x, y) | |9 - x| + y ≤ 12 ∧ 3 * y - x ≥ 18}

-- Define the line of revolution
def revolution_line (x y : ℝ) : Prop :=
  3 * y - x = 18

-- Define the volume of the solid
def solid_volume (S : Set (ℝ × ℝ)) (line : ℝ → ℝ → Prop) : ℝ :=
  -- This is a placeholder for the actual volume calculation
  sorry

-- Theorem statement
theorem volume_of_solid :
  solid_volume S revolution_line = 135 * Real.pi / (8 * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_CALUDE_volume_of_solid_l1488_148803


namespace NUMINAMATH_CALUDE_prince_gvidon_descendants_l1488_148850

/-- The total number of descendants of Prince Gvidon -/
def total_descendants : ℕ := 189

/-- The number of sons Prince Gvidon had -/
def initial_sons : ℕ := 3

/-- The number of descendants who had two sons each -/
def descendants_with_sons : ℕ := 93

/-- The number of sons each descendant with sons had -/
def sons_per_descendant : ℕ := 2

theorem prince_gvidon_descendants :
  total_descendants = initial_sons + descendants_with_sons * sons_per_descendant :=
by sorry

end NUMINAMATH_CALUDE_prince_gvidon_descendants_l1488_148850


namespace NUMINAMATH_CALUDE_lemonade_scaling_l1488_148829

/-- Lemonade recipe and scaling -/
theorem lemonade_scaling (lemons : ℕ) (sugar : ℚ) :
  (30 : ℚ) / 40 = lemons / 10 →
  (2 : ℚ) / 5 = sugar / 10 →
  lemons = 8 ∧ sugar = 4 := by
  sorry

#check lemonade_scaling

end NUMINAMATH_CALUDE_lemonade_scaling_l1488_148829


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1488_148854

theorem sphere_surface_area (r₁ r₂ d R : ℝ) : 
  r₁ > 0 → r₂ > 0 → d > 0 → R > 0 →
  r₁^2 * π = 9 * π →
  r₂^2 * π = 16 * π →
  d = 1 →
  R^2 = r₂^2 + (R - d)^2 →
  R^2 = r₁^2 + R^2 →
  4 * π * R^2 = 100 * π := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1488_148854


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1488_148824

/-- A right triangle with perimeter 60 and area 48 has a hypotenuse of length 28.4 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a^2 + b^2 = c^2 ∧  -- right triangle (Pythagorean theorem)
  a + b + c = 60 ∧  -- perimeter is 60
  (1/2) * a * b = 48 ∧  -- area is 48
  c = 28.4 :=  -- hypotenuse is 28.4
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1488_148824


namespace NUMINAMATH_CALUDE_absolute_difference_equals_one_l1488_148811

theorem absolute_difference_equals_one (x y : ℝ) :
  |x| - |y| = 1 ↔
  ((y = x - 1 ∧ x ≥ 1) ∨
   (y = 1 - x ∧ x ≥ 1) ∨
   (y = -x - 1 ∧ x ≤ -1) ∨
   (y = x + 1 ∧ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_difference_equals_one_l1488_148811


namespace NUMINAMATH_CALUDE_prime_divisibility_l1488_148808

theorem prime_divisibility (p a b : ℤ) (hp : Prime p) (hp_not_3 : p ≠ 3)
  (h_sum : p ∣ (a + b)) (h_cube_sum : p^2 ∣ (a^3 + b^3)) :
  p^2 ∣ (a + b) ∨ p^3 ∣ (a^3 + b^3) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1488_148808


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1488_148864

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean_triple : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_triple_l1488_148864


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l1488_148882

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) →
  b ∈ (Set.Ioo 0 1) →
  c ∈ (Set.Ioo 0 1) →
  a + b + c = 1 →
  3*a + 2*b = 2 →
  ∀ x y : ℝ, x ∈ (Set.Ioo 0 1) → y ∈ (Set.Ioo 0 1) → x + y < 1 → x * y ≤ a * b →
  a * b ≤ 1/6 :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l1488_148882


namespace NUMINAMATH_CALUDE_square_dissection_interior_rectangle_l1488_148898

-- Define a rectangle type
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

-- Define the square dissection
def SquareDissection (n : ℕ) (rectangles : Finset Rectangle) : Prop :=
  n > 1 ∧
  rectangles.card = n ∧
  (∀ r ∈ rectangles, r.x ≥ 0 ∧ r.y ≥ 0 ∧ r.x + r.width ≤ 1 ∧ r.y + r.height ≤ 1) ∧
  (∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 →
    ∃ r ∈ rectangles, r.x < x ∧ x < r.x + r.width ∧ r.y < y ∧ y < r.y + r.height)

-- Define an interior rectangle
def InteriorRectangle (r : Rectangle) : Prop :=
  r.x > 0 ∧ r.y > 0 ∧ r.x + r.width < 1 ∧ r.y + r.height < 1

-- The theorem to be proved
theorem square_dissection_interior_rectangle
  (n : ℕ) (rectangles : Finset Rectangle) (h : SquareDissection n rectangles) :
  ∃ r ∈ rectangles, InteriorRectangle r := by
  sorry

end NUMINAMATH_CALUDE_square_dissection_interior_rectangle_l1488_148898


namespace NUMINAMATH_CALUDE_packs_per_box_l1488_148855

/-- Given that Jenny sold 24.0 boxes of Trefoils and 192 packs in total,
    prove that there are 8 packs in each box. -/
theorem packs_per_box (boxes : ℝ) (total_packs : ℕ) 
    (h1 : boxes = 24.0) 
    (h2 : total_packs = 192) : 
  (total_packs : ℝ) / boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_packs_per_box_l1488_148855


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l1488_148822

def expression (x : ℝ) : ℝ := 6 * (x - 2 * x^3) - 5 * (2 * x^2 - 3 * x^3 + 2 * x^4) + 3 * (3 * x^2 - 2 * x^6)

theorem coefficient_of_x_squared :
  ∃ (a b c d e f : ℝ), 
    (∀ x, expression x = a * x + (-1) * x^2 + c * x^3 + d * x^4 + e * x^6 + f) :=
by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l1488_148822


namespace NUMINAMATH_CALUDE_calculate_expression_l1488_148812

theorem calculate_expression : 18 * 35 + 45 * 18 - 18 * 10 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1488_148812


namespace NUMINAMATH_CALUDE_tangent_and_chord_l1488_148800

noncomputable section

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - y = 0

-- Define the point P
structure Point_P where
  x : ℝ
  y : ℝ
  on_line : line_l x y

-- Define the tangent property
def is_tangent (x y : ℝ) : Prop := ∃ (t : ℝ), circle_M (x + t) (y + 2*t)

-- Main theorem
theorem tangent_and_chord :
  ∃ (P : Point_P),
    (∃ (A B : ℝ × ℝ),
      is_tangent (A.1 - P.x) (A.2 - P.y) ∧
      is_tangent (B.1 - P.x) (B.2 - P.y) ∧
      (A.1 - P.x) * (B.1 - P.x) + (A.2 - P.y) * (B.2 - P.y) = 
        ((A.1 - P.x)^2 + (A.2 - P.y)^2)^(1/2) * ((B.1 - P.x)^2 + (B.2 - P.y)^2)^(1/2) / 2) ∧
    ((P.x = 2 ∧ P.y = 4) ∨ (P.x = 6/5 ∧ P.y = 12/5)) ∧
    (∃ (C : ℝ × ℝ),
      (C.1 - P.x)^2 + (C.2 - P.y)^2 = (0 - P.x)^2 + (4 - P.y)^2 ∧
      ∃ (D : ℝ × ℝ),
        circle_M D.1 D.2 ∧
        (D.1 - C.1) * (1/2 - C.1) + (D.2 - C.2) * (15/4 - C.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_chord_l1488_148800


namespace NUMINAMATH_CALUDE_base_7_digits_of_1234_l1488_148873

theorem base_7_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_7_digits_of_1234_l1488_148873


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1488_148831

theorem coin_flip_probability : 
  let n : ℕ := 5  -- number of coins
  let k : ℕ := 3  -- minimum number of heads we're interested in
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := (Finset.range (n - k + 1)).sum (λ i => Nat.choose n (k + i))
  (favorable_outcomes : ℚ) / total_outcomes = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1488_148831


namespace NUMINAMATH_CALUDE_two_digit_numbers_divisibility_l1488_148823

/-- The number of digits in the numbers we're considering -/
def n : ℕ := 2019

/-- The set of possible digits -/
def digits : Set ℕ := {d | 0 ≤ d ∧ d ≤ 9}

/-- A function that counts the number of n-digit numbers made of 2 different digits -/
noncomputable def count_two_digit_numbers (n : ℕ) : ℕ :=
  sorry

/-- The highest power of 3 that divides a natural number -/
noncomputable def highest_power_of_three (m : ℕ) : ℕ :=
  sorry

theorem two_digit_numbers_divisibility :
  highest_power_of_three (count_two_digit_numbers n) = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_divisibility_l1488_148823


namespace NUMINAMATH_CALUDE_anthony_percentage_more_than_mabel_l1488_148894

theorem anthony_percentage_more_than_mabel :
  ∀ (mabel anthony cal jade : ℕ),
    mabel = 90 →
    cal = (2 * anthony) / 3 →
    jade = cal + 18 →
    jade = 84 →
    (anthony : ℚ) / mabel = 11 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_anthony_percentage_more_than_mabel_l1488_148894


namespace NUMINAMATH_CALUDE_sum_of_ages_at_milestone_l1488_148814

-- Define the ages of Hans, Josiah, and Julia
def hans_age : ℕ := 15
def josiah_age : ℕ := 3 * hans_age
def julia_age : ℕ := hans_age - 5

-- Define Julia's age when Hans was born
def julia_age_at_hans_birth : ℕ := julia_age / 2

-- Define Josiah's age when Julia was half her current age
def josiah_age_at_milestone : ℕ := josiah_age - hans_age - julia_age_at_hans_birth

-- Theorem statement
theorem sum_of_ages_at_milestone : 
  josiah_age_at_milestone + julia_age_at_hans_birth + 0 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_at_milestone_l1488_148814


namespace NUMINAMATH_CALUDE_total_distance_eight_points_circle_l1488_148820

/-- The total distance traveled by 8 points on a circle visiting non-adjacent points -/
theorem total_distance_eight_points_circle (r : ℝ) (h : r = 40) :
  let n := 8
  let distance_two_apart := r * Real.sqrt 2
  let distance_three_apart := r * Real.sqrt (2 + Real.sqrt 2)
  let distance_four_apart := 2 * r
  let single_point_distance := 4 * distance_two_apart + 2 * distance_three_apart + distance_four_apart
  n * single_point_distance = 1280 * Real.sqrt 2 + 640 * Real.sqrt (2 + Real.sqrt 2) + 640 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_eight_points_circle_l1488_148820


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_negative_four_l1488_148804

/-- Represents a parabola and its transformations -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Applies transformations to a parabola -/
def transform (p : Parabola) : Parabola :=
  { a := -p.a,  -- 180-degree rotation
    h := p.h - 4,  -- 4-unit left shift
    k := p.k - 3 } -- 3-unit downward shift

/-- Calculates the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := -2 * p.h

/-- Theorem: The sum of zeros of the transformed parabola is -4 -/
theorem sum_of_zeros_is_negative_four :
  let original := Parabola.mk 1 2 3
  let transformed := transform original
  sumOfZeros transformed = -4 := by sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_negative_four_l1488_148804


namespace NUMINAMATH_CALUDE_equation_solution_l1488_148870

theorem equation_solution :
  ∃ (x : ℚ), (x + 36) / 3 = (7 - 2*x) / 6 ∧ x = -65 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1488_148870


namespace NUMINAMATH_CALUDE_moles_of_Cu_CN_2_formed_l1488_148853

/-- Represents a chemical species in a reaction -/
inductive Species
| HCN
| CuSO4
| Cu_CN_2
| H2SO4

/-- Represents the coefficients of a balanced chemical equation -/
structure BalancedEquation :=
(reactants : Species → ℕ)
(products : Species → ℕ)

/-- Represents the available moles of each species -/
structure AvailableMoles :=
(moles : Species → ℝ)

def reaction : BalancedEquation :=
{ reactants := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
, products := λ s => match s with
  | Species.Cu_CN_2 => 1
  | Species.H2SO4 => 1
  | _ => 0
}

def available : AvailableMoles :=
{ moles := λ s => match s with
  | Species.HCN => 2
  | Species.CuSO4 => 1
  | _ => 0
}

/-- Calculates the moles of product formed based on the limiting reactant -/
def moles_of_product (eq : BalancedEquation) (avail : AvailableMoles) (product : Species) : ℝ :=
sorry

theorem moles_of_Cu_CN_2_formed :
  moles_of_product reaction available Species.Cu_CN_2 = 1 :=
sorry

end NUMINAMATH_CALUDE_moles_of_Cu_CN_2_formed_l1488_148853


namespace NUMINAMATH_CALUDE_distance_is_134_div_7_l1488_148862

/-- The distance from a point to a plane defined by three points -/
def distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The points given in the problem -/
def M₀ : ℝ × ℝ × ℝ := (-13, -8, 16)
def M₁ : ℝ × ℝ × ℝ := (1, 2, 0)
def M₂ : ℝ × ℝ × ℝ := (3, 0, -3)
def M₃ : ℝ × ℝ × ℝ := (5, 2, 6)

/-- The theorem stating that the distance is equal to 134/7 -/
theorem distance_is_134_div_7 : distance_point_to_plane M₀ M₁ M₂ M₃ = 134 / 7 := by sorry

end NUMINAMATH_CALUDE_distance_is_134_div_7_l1488_148862


namespace NUMINAMATH_CALUDE_coprime_20172019_l1488_148888

theorem coprime_20172019 :
  (Nat.gcd 20172019 20172017 = 1) ∧
  (Nat.gcd 20172019 20172018 = 1) ∧
  (Nat.gcd 20172019 20172020 = 1) ∧
  (Nat.gcd 20172019 20172021 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coprime_20172019_l1488_148888


namespace NUMINAMATH_CALUDE_square_side_length_l1488_148834

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 1 / 9 → side ^ 2 = area → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1488_148834


namespace NUMINAMATH_CALUDE_floor_a_equals_four_l1488_148876

theorem floor_a_equals_four (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x ≥ 0) (h3 : y ≥ 0) (h4 : z ≥ 0) :
  let a := Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1)
  ⌊a⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_a_equals_four_l1488_148876


namespace NUMINAMATH_CALUDE_tangent_line_at_x_squared_l1488_148839

theorem tangent_line_at_x_squared (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := 2 * x₀
  (λ x ↦ m * (x - x₀) + y₀) = (λ x ↦ 4 * x - 4) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_squared_l1488_148839


namespace NUMINAMATH_CALUDE_power_of_two_divides_factorial_l1488_148827

theorem power_of_two_divides_factorial (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
sorry

end NUMINAMATH_CALUDE_power_of_two_divides_factorial_l1488_148827


namespace NUMINAMATH_CALUDE_intersection_distance_l1488_148802

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the directrix of C
def directrix (x : ℝ) : Prop :=
  x = -2

-- Define the intersection points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (-2, -3)

-- State the theorem
theorem intersection_distance :
  ellipse A.1 A.2 ∧
  ellipse B.1 B.2 ∧
  directrix A.1 ∧
  directrix B.1 ∧
  (∃ (x : ℝ), x > 0 ∧ ellipse x 0 ∧ parabola x 0) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l1488_148802


namespace NUMINAMATH_CALUDE_expand_product_l1488_148860

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1488_148860


namespace NUMINAMATH_CALUDE_jung_mi_number_problem_l1488_148858

theorem jung_mi_number_problem :
  ∃ x : ℚ, (-4/5) * (x + (-2/3)) = -1/2 ∧ x = 31/24 := by
  sorry

end NUMINAMATH_CALUDE_jung_mi_number_problem_l1488_148858


namespace NUMINAMATH_CALUDE_isabel_candy_count_l1488_148816

/-- The total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (from_friend : ℕ) (from_cousin : ℕ) : ℕ :=
  initial + from_friend + from_cousin

/-- Theorem stating the total number of candy pieces Isabel has -/
theorem isabel_candy_count :
  ∀ x : ℕ, total_candy 216 137 x = 353 + x :=
by sorry

end NUMINAMATH_CALUDE_isabel_candy_count_l1488_148816


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1488_148893

theorem pure_imaginary_complex_number (a : ℝ) :
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1488_148893


namespace NUMINAMATH_CALUDE_jenny_distance_difference_l1488_148865

theorem jenny_distance_difference : 
  ∀ (run_distance walk_distance : ℝ),
    run_distance = 0.6 →
    walk_distance = 0.4 →
    run_distance - walk_distance = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l1488_148865


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1488_148872

theorem complex_magnitude_problem (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1488_148872


namespace NUMINAMATH_CALUDE_investment_interest_proof_l1488_148890

/-- Calculates the total interest earned on an investment with compound interest. -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the total interest earned on a $2000 investment at 8% annual interest
    compounded annually for 5 years is approximately $938.656. -/
theorem investment_interest_proof :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.656) < 0.001 := by
  sorry

#eval totalInterestEarned 2000 0.08 5

end NUMINAMATH_CALUDE_investment_interest_proof_l1488_148890


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l1488_148848

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculate the total hours worked in a week -/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculate the hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's actual work schedule -/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 360 }

/-- Theorem: Sheila's hourly wage is $10 -/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_sheila_hourly_wage_l1488_148848


namespace NUMINAMATH_CALUDE_total_marks_difference_l1488_148877

theorem total_marks_difference (P C M : ℝ) 
  (h1 : P + C + M > P) 
  (h2 : (C + M) / 2 = 75) : 
  P + C + M - P = 150 := by
sorry

end NUMINAMATH_CALUDE_total_marks_difference_l1488_148877


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1488_148863

theorem algebraic_expression_value (x y : ℝ) : 
  5 * x^2 - 4 * x * y - 1 = -11 → -10 * x^2 + 8 * x * y + 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1488_148863


namespace NUMINAMATH_CALUDE_largest_box_volume_l1488_148880

/-- The volume of the largest rectangular parallelopiped that can be enclosed in a cylindrical container with a hemispherical lid. -/
theorem largest_box_volume (total_height radius : ℝ) (h_total_height : total_height = 60) (h_radius : radius = 30) :
  let cylinder_height : ℝ := total_height - radius
  let box_base_side : ℝ := 2 * radius
  let box_height : ℝ := cylinder_height
  let box_volume : ℝ := box_base_side^2 * box_height
  box_volume = 108000 := by
  sorry

#check largest_box_volume

end NUMINAMATH_CALUDE_largest_box_volume_l1488_148880


namespace NUMINAMATH_CALUDE_derivative_difference_bound_l1488_148842

variable (f : ℝ → ℝ) (M : ℝ)

theorem derivative_difference_bound
  (h_diff : Differentiable ℝ f)
  (h_pos : M > 0)
  (h_bound : ∀ x t : ℝ, |f (x + t) - 2 * f x + f (x - t)| ≤ M * t^2) :
  ∀ x t : ℝ, |deriv f (x + t) - deriv f x| ≤ M * |t| :=
by sorry

end NUMINAMATH_CALUDE_derivative_difference_bound_l1488_148842


namespace NUMINAMATH_CALUDE_carla_water_calculation_l1488_148889

/-- The amount of water Carla needs to bring for her animals -/
def water_needed (pig_count : ℕ) (horse_count : ℕ) (pig_water : ℕ) (chicken_tank : ℕ) : ℕ :=
  let pig_total := pig_count * pig_water
  let horse_total := horse_count * (2 * pig_water)
  pig_total + horse_total + chicken_tank

/-- Theorem stating the total amount of water Carla needs -/
theorem carla_water_calculation :
  water_needed 8 10 3 30 = 114 := by
  sorry

end NUMINAMATH_CALUDE_carla_water_calculation_l1488_148889


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1488_148813

theorem sum_of_squares_of_roots (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → a^2 + b^2 = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1488_148813


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1488_148883

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 7 * Complex.I
  let z₂ : ℂ := 2 - 7 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = -90 / 53 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1488_148883


namespace NUMINAMATH_CALUDE_range_of_a_for_union_equality_intersection_A_B_union_A_B_l1488_148896

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 6}

-- State the theorem
theorem range_of_a_for_union_equality :
  ∀ a : ℝ, (A ∪ C a = C a) ↔ (2 ≤ a ∧ a < 3) :=
by sorry

-- Additional theorems for intersection and union of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} :=
by sorry

theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_union_equality_intersection_A_B_union_A_B_l1488_148896


namespace NUMINAMATH_CALUDE_fraction_problem_l1488_148871

theorem fraction_problem (a b c d : ℚ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 6) :
  d / a = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1488_148871


namespace NUMINAMATH_CALUDE_project_over_budget_proof_l1488_148846

/-- Calculates the amount a project is over budget given the total budget, 
    number of months, months passed, and actual expenditure. -/
def project_over_budget (total_budget : ℚ) (num_months : ℕ) 
                        (months_passed : ℕ) (actual_expenditure : ℚ) : ℚ :=
  actual_expenditure - (total_budget / num_months) * months_passed

/-- Proves that given the specific conditions of the problem, 
    the project is over budget by $280. -/
theorem project_over_budget_proof : 
  project_over_budget 12600 12 6 6580 = 280 := by
  sorry

#eval project_over_budget 12600 12 6 6580

end NUMINAMATH_CALUDE_project_over_budget_proof_l1488_148846


namespace NUMINAMATH_CALUDE_s₂_is_zero_l1488_148832

-- Define the polynomial division operation
def poly_div (p q : ℝ → ℝ) : (ℝ → ℝ) × ℝ := sorry

-- Define p₁(x) and s₁
def p₁_and_s₁ : (ℝ → ℝ) × ℝ := poly_div (λ x => x^6) (λ x => x - 1/2)

def p₁ : ℝ → ℝ := (p₁_and_s₁.1)
def s₁ : ℝ := (p₁_and_s₁.2)

-- Define p₂(x) and s₂
def p₂_and_s₂ : (ℝ → ℝ) × ℝ := poly_div p₁ (λ x => x - 1/2)

def p₂ : ℝ → ℝ := (p₂_and_s₂.1)
def s₂ : ℝ := (p₂_and_s₂.2)

-- The theorem to prove
theorem s₂_is_zero : s₂ = 0 := by sorry

end NUMINAMATH_CALUDE_s₂_is_zero_l1488_148832


namespace NUMINAMATH_CALUDE_proposition_ranges_l1488_148867

def prop_p (m : ℝ) : Prop :=
  ∀ x : ℝ, -3 < x ∧ x < 1 → x^2 + 4*x + 9 - m > 0

def prop_q (m : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x^2 - 2*m*x + 1 < 0

theorem proposition_ranges (m : ℝ) :
  (prop_p m ↔ m < 5) ∧
  (prop_p m ≠ prop_q m ↔ m ≤ 1 ∨ m ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_proposition_ranges_l1488_148867


namespace NUMINAMATH_CALUDE_vector_equality_l1488_148845

/-- Given two vectors in ℝ², prove that if their sum and difference have equal magnitudes, 
    then the second component of the second vector must be 3/2. -/
theorem vector_equality (a b : ℝ × ℝ) (h : ‖a + b‖ = ‖a - b‖) 
    (ha : a = (1, 2)) (hb : b.1 = -3) : b.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_l1488_148845


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1488_148861

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : a₃ = 11) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 119 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1488_148861


namespace NUMINAMATH_CALUDE_final_values_l1488_148828

def program_execution (a b : Int) : Int × Int :=
  let a' := a + b
  let b' := a' - b
  (a', b')

theorem final_values : program_execution 1 3 = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_final_values_l1488_148828


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1488_148885

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, |x - 1| < 2 → x^2 - 5*x - 6 < 0) ∧
  (∃ x : ℝ, x^2 - 5*x - 6 < 0 ∧ ¬(|x - 1| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1488_148885


namespace NUMINAMATH_CALUDE_sin_m_eq_cos_810_l1488_148886

theorem sin_m_eq_cos_810 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.sin (m * π / 180) = Real.cos (810 * π / 180)) :
  m = 0 ∨ m = 180 := by
  sorry

end NUMINAMATH_CALUDE_sin_m_eq_cos_810_l1488_148886


namespace NUMINAMATH_CALUDE_animal_distance_calculation_l1488_148869

/-- Calculates the total distance covered by a fox, rabbit, and deer running at their maximum speeds for 120 minutes. -/
theorem animal_distance_calculation :
  let fox_speed : ℝ := 50  -- km/h
  let rabbit_speed : ℝ := 60  -- km/h
  let deer_speed : ℝ := 80  -- km/h
  let time_hours : ℝ := 120 / 60  -- Convert 120 minutes to hours
  let fox_distance := fox_speed * time_hours
  let rabbit_distance := rabbit_speed * time_hours
  let deer_distance := deer_speed * time_hours
  let total_distance := fox_distance + rabbit_distance + deer_distance
  total_distance = 380  -- km
  := by sorry

end NUMINAMATH_CALUDE_animal_distance_calculation_l1488_148869


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1488_148826

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planes_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → planes_perpendicular α β :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l1488_148826


namespace NUMINAMATH_CALUDE_least_cans_required_l1488_148856

def maaza_volume : ℕ := 20
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_cans_required :
  let gcd := Nat.gcd (Nat.gcd maaza_volume pepsi_volume) sprite_volume
  maaza_volume / gcd + pepsi_volume / gcd + sprite_volume / gcd = 133 := by
  sorry

end NUMINAMATH_CALUDE_least_cans_required_l1488_148856


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l1488_148825

/-- The property that a function f: ℕ → ℕ must satisfy -/
def SatisfiesProperty (f : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (f n) + f (f (f n)) = 3 * n

/-- Theorem stating that the identity function is the only function satisfying the property -/
theorem unique_satisfying_function :
  ∀ f : ℕ → ℕ, SatisfiesProperty f → f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l1488_148825


namespace NUMINAMATH_CALUDE_remaining_money_after_expenses_l1488_148874

def rent : ℝ := 1200
def salary : ℝ := 5000

theorem remaining_money_after_expenses :
  let food_and_travel := 2 * rent
  let shared_rent := rent / 2
  let total_expenses := food_and_travel + shared_rent
  salary - total_expenses = 2000 := by sorry

end NUMINAMATH_CALUDE_remaining_money_after_expenses_l1488_148874


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1488_148879

def number := 102 * 104 * 107 * 108

theorem distinct_prime_factors_count :
  Nat.card (Nat.factors number).toFinset = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1488_148879


namespace NUMINAMATH_CALUDE_remainder_three_power_2010_mod_8_l1488_148835

theorem remainder_three_power_2010_mod_8 : 3^2010 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_2010_mod_8_l1488_148835


namespace NUMINAMATH_CALUDE_p_plus_q_equals_42_l1488_148851

theorem p_plus_q_equals_42 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 4 → P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4)) →
  P + Q = 42 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_42_l1488_148851


namespace NUMINAMATH_CALUDE_least_candies_eleven_candies_maria_candies_l1488_148859

theorem least_candies (c : ℕ) : c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 → c ≥ 11 :=
by sorry

theorem eleven_candies : 11 % 3 = 2 ∧ 11 % 4 = 3 ∧ 11 % 6 = 5 :=
by sorry

theorem maria_candies : ∃ (c : ℕ), c > 0 ∧ c % 3 = 2 ∧ c % 4 = 3 ∧ c % 6 = 5 ∧ c = 11 :=
by sorry

end NUMINAMATH_CALUDE_least_candies_eleven_candies_maria_candies_l1488_148859


namespace NUMINAMATH_CALUDE_yellow_marble_probability_l1488_148817

/-- Represents a bag of marbles with counts for different colors -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the probability of drawing a yellow marble as the last marble
    given the contents of bags A, B, C, and D and the described drawing process -/
def yellowProbability (bagA bagB bagC bagD : Bag) : ℚ :=
  let totalA := bagA.white + bagA.black
  let totalB := bagB.yellow + bagB.blue
  let totalC := bagC.yellow + bagC.blue
  let totalD := bagD.yellow + bagD.blue
  
  let probWhiteA := bagA.white / totalA
  let probBlackA := bagA.black / totalA
  let probYellowB := bagB.yellow / totalB
  let probBlueC := bagC.blue / totalC
  let probYellowC := bagC.yellow / totalC
  let probYellowD := bagD.yellow / totalD
  
  probWhiteA * probYellowB + 
  probBlackA * probBlueC * probYellowD + 
  probBlackA * probYellowC

theorem yellow_marble_probability :
  let bagA : Bag := { white := 5, black := 6 }
  let bagB : Bag := { yellow := 8, blue := 6 }
  let bagC : Bag := { yellow := 3, blue := 7 }
  let bagD : Bag := { yellow := 1, blue := 4 }
  yellowProbability bagA bagB bagC bagD = 136 / 275 := by
  sorry

end NUMINAMATH_CALUDE_yellow_marble_probability_l1488_148817


namespace NUMINAMATH_CALUDE_cubic_factorization_l1488_148830

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x + 3)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1488_148830


namespace NUMINAMATH_CALUDE_cupcake_production_difference_l1488_148841

def cupcake_difference (betty_rate : ℕ) (dora_rate : ℕ) (total_time : ℕ) (break_time : ℕ) : ℕ :=
  (dora_rate * total_time) - (betty_rate * (total_time - break_time))

theorem cupcake_production_difference :
  cupcake_difference 10 8 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_production_difference_l1488_148841


namespace NUMINAMATH_CALUDE_intersection_M_N_l1488_148875

def M : Set ℝ := {-4, -3, -2, -1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 + 3*x < 0}

theorem intersection_M_N : M ∩ N = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1488_148875


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l1488_148878

theorem complex_modulus_theorem : Complex.abs (-6 + (9/4) * Complex.I) = (Real.sqrt 657) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l1488_148878


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1488_148838

/-- A curve y = sin(x + φ) is symmetric about the y-axis if and only if sin(x + φ) = sin(-x + φ) for all x ∈ ℝ -/
def symmetric_about_y_axis (φ : ℝ) : Prop :=
  ∀ x : ℝ, Real.sin (x + φ) = Real.sin (-x + φ)

/-- φ = π/2 is a sufficient condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_condition (φ : ℝ) (h : φ = π / 2) : symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is not a necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem not_necessary_condition : ∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ := by
  sorry

/-- φ = π/2 is a sufficient but not necessary condition for y = sin(x + φ) to be symmetric about the y-axis -/
theorem sufficient_but_not_necessary : 
  (∀ φ : ℝ, φ = π / 2 → symmetric_about_y_axis φ) ∧ 
  (∃ φ : ℝ, φ ≠ π / 2 ∧ symmetric_about_y_axis φ) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1488_148838


namespace NUMINAMATH_CALUDE_student_selection_permutation_l1488_148892

theorem student_selection_permutation :
  (Nat.factorial 6) / (Nat.factorial 4) = 30 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_permutation_l1488_148892


namespace NUMINAMATH_CALUDE_goldfish_red_balls_l1488_148866

/-- Given a fish tank with goldfish and platyfish, prove the number of red balls each goldfish plays with -/
theorem goldfish_red_balls 
  (total_balls : ℕ) 
  (num_goldfish : ℕ) 
  (num_platyfish : ℕ) 
  (white_balls_per_platyfish : ℕ) 
  (h1 : total_balls = 80) 
  (h2 : num_goldfish = 3) 
  (h3 : num_platyfish = 10) 
  (h4 : white_balls_per_platyfish = 5) : 
  (total_balls - num_platyfish * white_balls_per_platyfish) / num_goldfish = 10 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_red_balls_l1488_148866


namespace NUMINAMATH_CALUDE_exactly_one_incorrect_l1488_148807

-- Define the statements
def statement1 : Prop := ∀ (P : ℝ → Prop), (∀ x, P x) ↔ ¬(∃ x, ¬(P x))

def statement2 : Prop := ∀ (p q : Prop), ¬(p ∨ q) → (¬p ∧ ¬q)

def statement3 : Prop := ∀ (m n : ℝ), 
  (m * n > 0 → (∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (m > 0 ∧ n > 0 ∧ m ≠ n))) ∧
  (¬(∀ x y : ℝ, m * x^2 + n * y^2 = 1 ↔ (m > 0 ∧ n > 0 ∧ m ≠ n)) → m * n ≤ 0)

-- Theorem to prove
theorem exactly_one_incorrect : 
  (statement1 ∧ statement2 ∧ ¬statement3) ∨
  (statement1 ∧ ¬statement2 ∧ statement3) ∨
  (¬statement1 ∧ statement2 ∧ statement3) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_incorrect_l1488_148807


namespace NUMINAMATH_CALUDE_penny_fountain_problem_l1488_148821

theorem penny_fountain_problem (rachelle gretchen rocky : ℕ) : 
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  rachelle + gretchen + rocky = 300 :=
by sorry

end NUMINAMATH_CALUDE_penny_fountain_problem_l1488_148821


namespace NUMINAMATH_CALUDE_lunch_cost_before_tip_l1488_148895

/-- Given a 20% tip and a total spending of $60.6, prove that the original cost of the lunch before the tip was $50.5. -/
theorem lunch_cost_before_tip (tip_percentage : Real) (total_spent : Real) (lunch_cost : Real) : 
  tip_percentage = 0.20 →
  total_spent = 60.6 →
  lunch_cost * (1 + tip_percentage) = total_spent →
  lunch_cost = 50.5 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_before_tip_l1488_148895


namespace NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l1488_148899

/-- The cost of the first book -/
def book1_cost : ℚ := 13

/-- The cost of the second book -/
def book2_cost : ℚ := 15

/-- The cost of the third and fourth books -/
def book34_cost : ℚ := 10

/-- The discount rate applied to the first two books -/
def discount_rate : ℚ := 1/4

/-- The free shipping threshold -/
def free_shipping_threshold : ℚ := 50

/-- Calculate the discounted price of a book -/
def apply_discount (price : ℚ) : ℚ :=
  price * (1 - discount_rate)

/-- Calculate the total cost of all four books with discounts applied -/
def total_cost : ℚ :=
  apply_discount book1_cost + apply_discount book2_cost + 2 * book34_cost

/-- The theorem stating the additional amount needed for free shipping -/
theorem additional_amount_for_free_shipping :
  free_shipping_threshold - total_cost = 9 := by sorry

end NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l1488_148899


namespace NUMINAMATH_CALUDE_folded_area_ratio_paper_folding_problem_l1488_148809

/-- Represents a rectangular piece of paper. -/
structure Paper where
  length : ℝ
  width : ℝ
  area : ℝ
  widthIsSquareRootTwo : width = Real.sqrt 2 * length
  areaIsLengthTimesWidth : area = length * width

/-- Represents the paper after folding. -/
structure FoldedPaper where
  original : Paper
  foldedArea : ℝ

/-- The ratio of the folded area to the original area is (16 - √6) / 16. -/
theorem folded_area_ratio (p : Paper) (fp : FoldedPaper) 
    (h : fp.original = p) : 
    fp.foldedArea / p.area = (16 - Real.sqrt 6) / 16 := by
  sorry

/-- Main theorem stating the result of the problem. -/
theorem paper_folding_problem :
  ∃ (p : Paper) (fp : FoldedPaper), 
    fp.original = p ∧ fp.foldedArea / p.area = (16 - Real.sqrt 6) / 16 := by
  sorry

end NUMINAMATH_CALUDE_folded_area_ratio_paper_folding_problem_l1488_148809


namespace NUMINAMATH_CALUDE_function_has_two_zeros_l1488_148815

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem function_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_function_has_two_zeros_l1488_148815


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l1488_148897

theorem sum_of_solutions_eq_eight : 
  ∃ (x y : ℝ), x * (x - 8) = 7 ∧ y * (y - 8) = 7 ∧ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_eight_l1488_148897


namespace NUMINAMATH_CALUDE_no_feasible_distribution_no_feasible_distribution_proof_l1488_148881

/-- Represents a cricket player with their initial average runs and desired increase --/
structure Player where
  initialAvg : ℕ
  desiredIncrease : ℕ

/-- Theorem stating that no feasible distribution exists for the given problem --/
theorem no_feasible_distribution 
  (playerA : Player) 
  (playerB : Player) 
  (playerC : Player) 
  (totalRunsLimit : ℕ) : Prop :=
  playerA.initialAvg = 32 ∧ 
  playerA.desiredIncrease = 4 ∧
  playerB.initialAvg = 45 ∧ 
  playerB.desiredIncrease = 5 ∧
  playerC.initialAvg = 55 ∧ 
  playerC.desiredIncrease = 6 ∧
  totalRunsLimit = 250 →
  ¬∃ (runsA runsB runsC : ℕ),
    (runsA + runsB + runsC ≤ totalRunsLimit) ∧
    ((playerA.initialAvg * 10 + runsA) / 11 ≥ playerA.initialAvg + playerA.desiredIncrease) ∧
    ((playerB.initialAvg * 10 + runsB) / 11 ≥ playerB.initialAvg + playerB.desiredIncrease) ∧
    ((playerC.initialAvg * 10 + runsC) / 11 ≥ playerC.initialAvg + playerC.desiredIncrease)

/-- The proof of the theorem --/
theorem no_feasible_distribution_proof : no_feasible_distribution 
  { initialAvg := 32, desiredIncrease := 4 }
  { initialAvg := 45, desiredIncrease := 5 }
  { initialAvg := 55, desiredIncrease := 6 }
  250 := by
  sorry

end NUMINAMATH_CALUDE_no_feasible_distribution_no_feasible_distribution_proof_l1488_148881


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1488_148868

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + a₃ + a₅ = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1488_148868
