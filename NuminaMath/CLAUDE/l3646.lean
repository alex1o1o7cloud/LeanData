import Mathlib

namespace NUMINAMATH_CALUDE_recycled_cans_count_l3646_364680

def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 5 then 0
  else (initial_cans / 5) + recycle_cans (initial_cans / 5)

theorem recycled_cans_count :
  recycle_cans 3125 = 781 :=
by sorry

end NUMINAMATH_CALUDE_recycled_cans_count_l3646_364680


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3646_364608

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 2 * x - 5) - (2 * x^3 + x^2 + 3 * x + 7) = x^3 + 3 * x^2 - x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3646_364608


namespace NUMINAMATH_CALUDE_complex_number_problem_l3646_364689

def i : ℂ := Complex.I

theorem complex_number_problem (z : ℂ) (h : (1 + 2*i)*z = 3 - 4*i) : 
  z.im = -2 ∧ Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3646_364689


namespace NUMINAMATH_CALUDE_gear_angular_speed_relationship_l3646_364639

theorem gear_angular_speed_relationship 
  (x y z : ℕ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (ω₁ ω₂ ω₃ : ℝ) 
  (h₁ : 2 * x * ω₁ = 3 * y * ω₂) 
  (h₂ : 3 * y * ω₂ = 4 * z * ω₃) :
  ∃ (k : ℝ), k > 0 ∧ 
    ω₁ = k * (2 * z / x) ∧
    ω₂ = k * (4 * z / (3 * y)) ∧
    ω₃ = k :=
by sorry

end NUMINAMATH_CALUDE_gear_angular_speed_relationship_l3646_364639


namespace NUMINAMATH_CALUDE_rectangle_from_isosceles_120_l3646_364663

/-- An isosceles triangle with a vertex angle of 120 degrees --/
structure IsoscelesTriangle120 where
  -- We represent the triangle by its side lengths
  base : ℝ
  leg : ℝ
  base_positive : 0 < base
  leg_positive : 0 < leg
  vertex_angle : Real.cos (120 * π / 180) = (base^2 - 2 * leg^2) / (2 * leg^2)

/-- A rectangle formed by isosceles triangles --/
structure RectangleFromTriangles where
  width : ℝ
  height : ℝ
  triangles : List IsoscelesTriangle120
  width_positive : 0 < width
  height_positive : 0 < height

/-- Theorem stating that it's possible to form a rectangle from isosceles triangles with 120° vertex angle --/
theorem rectangle_from_isosceles_120 : 
  ∃ (r : RectangleFromTriangles), r.triangles.length > 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_from_isosceles_120_l3646_364663


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3646_364633

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3646_364633


namespace NUMINAMATH_CALUDE_function_properties_l3646_364658

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (5 - x) = f (5 + x))
    (h3 : ¬ (∀ x y, f x = f y)) :
    IsEven f ∧ IsPeriodic f 10 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3646_364658


namespace NUMINAMATH_CALUDE_river_current_speed_l3646_364601

/-- The speed of a river's current given swimmer's performance -/
theorem river_current_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : distance = 7) 
  (h2 : time = 3.684210526315789) 
  (h3 : still_water_speed = 4.4) : 
  ∃ current_speed : ℝ, 
    current_speed = 2.5 ∧ 
    distance / time = still_water_speed - current_speed := by
  sorry

#check river_current_speed

end NUMINAMATH_CALUDE_river_current_speed_l3646_364601


namespace NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l3646_364684

theorem hcf_of_ratio_and_lcm (a b : ℕ+) : 
  (a : ℚ) / b = 4 / 5 → 
  Nat.lcm a b = 80 → 
  Nat.gcd a b = 2 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_ratio_and_lcm_l3646_364684


namespace NUMINAMATH_CALUDE_remainder_of_1531_base12_div_8_l3646_364622

/-- Represents a base-12 number as a list of digits (least significant first) -/
def Base12 := List Nat

/-- Converts a base-12 number to base-10 -/
def toBase10 (n : Base12) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 number 1531 -/
def num : Base12 := [1, 3, 5, 1]

theorem remainder_of_1531_base12_div_8 :
  toBase10 num % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1531_base12_div_8_l3646_364622


namespace NUMINAMATH_CALUDE_max_value_theorem_l3646_364657

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 7 + 9 * y * z ≤ (1/2) * Real.sqrt 88 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3646_364657


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3646_364624

theorem sum_of_solutions : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, x > 0 ∧ x ≤ 30 ∧ (17*(4*x - 3) % 10 = 34 % 10)) ∧
  (∀ x : ℕ, x > 0 ∧ x ≤ 30 ∧ (17*(4*x - 3) % 10 = 34 % 10) → x ∈ S) ∧
  (Finset.sum S id = 51) := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3646_364624


namespace NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l3646_364636

theorem or_and_not_implies_false_and_true (p q : Prop) :
  (p ∨ q) → (¬p) → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implies_false_and_true_l3646_364636


namespace NUMINAMATH_CALUDE_total_pages_read_l3646_364654

/-- Represents the number of pages in each chapter of the book --/
def pages_per_chapter : ℕ := 40

/-- Represents the number of chapters Mitchell read before 4 o'clock --/
def chapters_before_4 : ℕ := 10

/-- Represents the number of pages Mitchell read from the 11th chapter at 4 o'clock --/
def pages_at_4 : ℕ := 20

/-- Represents the number of additional chapters Mitchell read after 4 o'clock --/
def chapters_after_4 : ℕ := 2

/-- Theorem stating that the total number of pages Mitchell read is 500 --/
theorem total_pages_read : 
  pages_per_chapter * chapters_before_4 + 
  pages_at_4 + 
  pages_per_chapter * chapters_after_4 = 500 := by
sorry

end NUMINAMATH_CALUDE_total_pages_read_l3646_364654


namespace NUMINAMATH_CALUDE_product_expansion_l3646_364676

theorem product_expansion (x : ℝ) : (2 + 3 * x) * (-2 + 3 * x) = 9 * x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3646_364676


namespace NUMINAMATH_CALUDE_cos_30_degree_calculation_l3646_364677

theorem cos_30_degree_calculation : 
  |Real.sqrt 3 - 1| - 2 * (Real.sqrt 3 / 2) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_30_degree_calculation_l3646_364677


namespace NUMINAMATH_CALUDE_no_zeros_of_g_l3646_364694

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x)/x > 0 for all x ≠ 0,
    the function g(x) = f(x) + 1/x has no zeros. -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x ≠ 0, deriv f x + f x / x > 0) :
    ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_of_g_l3646_364694


namespace NUMINAMATH_CALUDE_grass_weeds_count_l3646_364681

/-- Represents the number of weeds in different areas of the garden -/
structure GardenWeeds where
  flower_bed : ℕ
  vegetable_patch : ℕ
  grass : ℕ

/-- Represents Lucille's earnings and expenses -/
structure LucilleFinances where
  cents_per_weed : ℕ
  soda_cost : ℕ
  remaining_cents : ℕ

def calculate_grass_weeds (garden : GardenWeeds) (finances : LucilleFinances) : ℕ :=
  garden.grass

theorem grass_weeds_count 
  (garden : GardenWeeds) 
  (finances : LucilleFinances) 
  (h1 : garden.flower_bed = 11)
  (h2 : garden.vegetable_patch = 14)
  (h3 : finances.cents_per_weed = 6)
  (h4 : finances.soda_cost = 99)
  (h5 : finances.remaining_cents = 147)
  : calculate_grass_weeds garden finances = 32 := by
  sorry

#eval calculate_grass_weeds 
  { flower_bed := 11, vegetable_patch := 14, grass := 32 } 
  { cents_per_weed := 6, soda_cost := 99, remaining_cents := 147 }

end NUMINAMATH_CALUDE_grass_weeds_count_l3646_364681


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3646_364690

/-- Given a square with side length 2 and four congruent isosceles triangles constructed with 
    their bases on the sides of the square, if the sum of the areas of the four isosceles 
    triangles is equal to the area of the square, then the length of one of the two congruent 
    sides of one isosceles triangle is √17/2. -/
theorem isosceles_triangle_side_length (square_side : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) :
  square_side = 2 →
  triangle_base = square_side →
  (4 * (1/2 * triangle_base * triangle_height)) = square_side^2 →
  ∃ (triangle_side : ℝ), 
    triangle_side^2 = (triangle_base/2)^2 + triangle_height^2 ∧ 
    triangle_side = Real.sqrt 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3646_364690


namespace NUMINAMATH_CALUDE_woman_birth_year_l3646_364653

/-- A woman born in the second half of the 19th century was x years old in the year x^2. -/
theorem woman_birth_year :
  ∃ x : ℕ,
    (1850 ≤ x^2 - x) ∧
    (x^2 - x < 1900) ∧
    (x^2 = x + 1892) :=
by sorry

end NUMINAMATH_CALUDE_woman_birth_year_l3646_364653


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l3646_364693

def days_to_read (n : ℕ) : ℕ := n + 1

def total_days (n : ℕ) : ℕ := 
  n * (2 * 2 + (n - 1) * 1) / 2

def weekday (start_day : ℕ) (days_passed : ℕ) : ℕ := 
  (start_day + days_passed) % 7

theorem zoey_reading_schedule :
  let number_of_books : ℕ := 20
  let friday : ℕ := 5
  (total_days number_of_books = 230) ∧ 
  (weekday friday (total_days number_of_books) = 4) := by
  sorry

#check zoey_reading_schedule

end NUMINAMATH_CALUDE_zoey_reading_schedule_l3646_364693


namespace NUMINAMATH_CALUDE_ellipse_product_l3646_364642

/-- Given an ellipse with center O, major axis AB, minor axis CD, and focus F,
    prove that if OF = 8 and the diameter of the inscribed circle of triangle OCF is 4,
    then (AB)(CD) = 240 -/
theorem ellipse_product (O A B C D F : ℝ × ℝ) : 
  let OA := dist O A
  let OB := dist O B
  let OC := dist O C
  let OD := dist O D
  let OF := dist O F
  let a := OA
  let b := OC
  let inscribed_diameter := 4
  (OA = OB) →  -- A and B are equidistant from O (major axis)
  (OC = OD) →  -- C and D are equidistant from O (minor axis)
  (a > b) →    -- major axis is longer than minor axis
  (OF = 8) →   -- given condition
  (b + OF - a = inscribed_diameter / 2) →  -- inradius formula for triangle OCF
  (2 * a) * (2 * b) = 240 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_product_l3646_364642


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3646_364603

/-- Given a cube with volume 8x cubic units and surface area 4x square units, x = 216 -/
theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 4*x) → x = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3646_364603


namespace NUMINAMATH_CALUDE_roberts_pencils_l3646_364628

-- Define the price of a pencil in cents
def pencil_price : ℕ := 20

-- Define the number of pencils Tolu wants
def tolu_pencils : ℕ := 3

-- Define the number of pencils Melissa wants
def melissa_pencils : ℕ := 2

-- Define the total amount spent by all students in cents
def total_spent : ℕ := 200

-- Theorem to prove Robert's number of pencils
theorem roberts_pencils : 
  ∃ (robert_pencils : ℕ), 
    pencil_price * (tolu_pencils + melissa_pencils + robert_pencils) = total_spent ∧
    robert_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_roberts_pencils_l3646_364628


namespace NUMINAMATH_CALUDE_ab_value_l3646_364626

theorem ab_value (a b : ℝ) (h1 : a - b = 8) (h2 : a^2 + b^2 = 164) : a * b = 50 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3646_364626


namespace NUMINAMATH_CALUDE_circle_exists_with_conditions_l3646_364612

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a circle in 3D space
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D  -- Normal vector to the plane of the circle

def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

def circle_touches_plane (c : Circle3D) (p : Plane3D) : Prop := sorry

def circle_passes_through_points (c : Circle3D) (p1 p2 : Point3D) : Prop := sorry

theorem circle_exists_with_conditions 
  (p1 p2 : Point3D) 
  (projection_plane : Plane3D) : 
  ∃ (c : Circle3D), 
    circle_passes_through_points c p1 p2 ∧ 
    angle_between_planes (Plane3D.mk c.normal.x c.normal.y c.normal.z 0) projection_plane = π/3 ∧
    circle_touches_plane c projection_plane := by
  sorry

end NUMINAMATH_CALUDE_circle_exists_with_conditions_l3646_364612


namespace NUMINAMATH_CALUDE_quadratic_positive_iff_not_one_l3646_364668

/-- Given a quadratic function f(x) = x^2 + bx + 1 where f(-1) = f(3),
    prove that f(x) > 0 if and only if x ≠ 1 -/
theorem quadratic_positive_iff_not_one (b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^2 + b*x + 1)
    (h2 : f (-1) = f 3) :
    ∀ x, f x > 0 ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_positive_iff_not_one_l3646_364668


namespace NUMINAMATH_CALUDE_tile_arrangements_l3646_364632

/-- The number of distinguishable arrangements of tiles -/
def num_arrangements (brown purple green yellow : ℕ) : ℕ :=
  Nat.factorial (brown + purple + green + yellow) /
  (Nat.factorial brown * Nat.factorial purple * Nat.factorial green * Nat.factorial yellow)

/-- Theorem stating that the number of distinguishable arrangements
    of 1 brown, 1 purple, 2 green, and 3 yellow tiles is 420 -/
theorem tile_arrangements :
  num_arrangements 1 1 2 3 = 420 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l3646_364632


namespace NUMINAMATH_CALUDE_quadratic_coefficient_inequalities_l3646_364688

theorem quadratic_coefficient_inequalities
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_real_roots : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) :
  min a (min b c) ≤ (1/4) * (a + b + c) ∧
  max a (max b c) ≥ (4/9) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_inequalities_l3646_364688


namespace NUMINAMATH_CALUDE_mandy_toys_count_l3646_364627

theorem mandy_toys_count (mandy anna amanda : ℕ) 
  (h1 : anna = 3 * mandy)
  (h2 : amanda = anna + 2)
  (h3 : mandy + anna + amanda = 142) :
  mandy = 20 := by
sorry

end NUMINAMATH_CALUDE_mandy_toys_count_l3646_364627


namespace NUMINAMATH_CALUDE_password_unique_l3646_364615

def is_valid_password (n : ℕ) : Prop :=
  -- The password is an eight-digit number
  100000000 > n ∧ n ≥ 10000000 ∧
  -- The password is a multiple of both 3 and 25
  n % 3 = 0 ∧ n % 25 = 0 ∧
  -- The password is between 20,000,000 and 30,000,000
  n > 20000000 ∧ n < 30000000 ∧
  -- The millions place and the hundred thousands place digits are the same
  (n / 1000000) % 10 = (n / 100000) % 10 ∧
  -- The hundreds digit is 2 less than the ten thousands digit
  (n / 100) % 10 + 2 = (n / 10000) % 10 ∧
  -- The digits in the hundred thousands, ten thousands, and thousands places form a three-digit number
  -- which, when divided by the two-digit number formed by the digits in the ten millions and millions places,
  -- gives a quotient of 25
  ((n / 100000) % 1000) / ((n / 1000000) % 100) = 25

theorem password_unique : ∀ n : ℕ, is_valid_password n ↔ n = 26650350 := by
  sorry

end NUMINAMATH_CALUDE_password_unique_l3646_364615


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3646_364695

/-- A line passing through (0, 7) perpendicular to 2x - 6y - 14 = 0 has equation y + 3x - 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), y = m * x + b ∧ (0, 7) ∈ {(x, y) | y = m * x + b}) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 2 * x₁ - 6 * y₁ - 14 = 0 ∧ 2 * x₂ - 6 * y₂ - 14 = 0 → 
    (y₂ - y₁) * (y - 7) = -(x₂ - x₁) * x) → 
  y + 3 * x - 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3646_364695


namespace NUMINAMATH_CALUDE_division_problem_l3646_364652

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3646_364652


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l3646_364682

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, ¬(a > 1) ∧ 1 / a < 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l3646_364682


namespace NUMINAMATH_CALUDE_quadratic_intercepts_l3646_364613

/-- A quadratic function. -/
structure QuadraticFunction where
  f : ℝ → ℝ

/-- The x-intercepts of two quadratic functions. -/
structure XIntercepts where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ

/-- The problem statement. -/
theorem quadratic_intercepts 
  (f g : QuadraticFunction) 
  (x : XIntercepts) 
  (h1 : ∀ x, g.f x = -f.f (120 - x))
  (h2 : ∃ v, g.f v = f.f v ∧ ∀ x, f.f x ≤ f.f v)
  (h3 : x.x₁ < x.x₂ ∧ x.x₂ < x.x₃ ∧ x.x₃ < x.x₄)
  (h4 : x.x₃ - x.x₂ = 160) :
  x.x₄ - x.x₁ = 640 + 320 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intercepts_l3646_364613


namespace NUMINAMATH_CALUDE_inequality_proof_l3646_364600

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1 + 9*a^2)/(1 + 2*a + 2*b^2 + 2*c^2) + 
  (1 + 9*b^2)/(1 + 2*b + 2*c^2 + 2*a^2) + 
  (1 + 9*c^2)/(1 + 2*c + 2*a^2 + 2*b^2) < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3646_364600


namespace NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l3646_364655

theorem maximize_x_cube_y_fourth (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 40) :
  x^3 * y^4 ≤ 24^3 * 32^4 ∧ x^3 * y^4 = 24^3 * 32^4 ↔ x = 24 ∧ y = 32 := by
  sorry

end NUMINAMATH_CALUDE_maximize_x_cube_y_fourth_l3646_364655


namespace NUMINAMATH_CALUDE_baseball_bat_price_l3646_364648

-- Define the prices and quantities
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def price_difference : ℝ := 237

-- Define the theorem
theorem baseball_bat_price :
  ∃ (bat_price : ℝ),
    (basketball_price * basketball_quantity) =
    (baseball_price * baseball_quantity + bat_price + price_difference) ∧
    bat_price = 18 := by
  sorry

end NUMINAMATH_CALUDE_baseball_bat_price_l3646_364648


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3646_364697

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3646_364697


namespace NUMINAMATH_CALUDE_max_value_of_f_l3646_364696

noncomputable def f (x : ℝ) : ℝ := x^3 - 15*x^2 - 33*x + 6

theorem max_value_of_f :
  ∃ (M : ℝ), M = 23 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3646_364696


namespace NUMINAMATH_CALUDE_complex_norm_problem_l3646_364630

theorem complex_norm_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 6)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_norm_problem_l3646_364630


namespace NUMINAMATH_CALUDE_bird_families_flew_away_l3646_364620

/-- Given the initial number of bird families and the number of families left,
    calculate the number of families that flew away. -/
theorem bird_families_flew_away (initial : ℕ) (left : ℕ) (flew_away : ℕ) 
    (h1 : initial = 41)
    (h2 : left = 14)
    (h3 : flew_away = initial - left) :
  flew_away = 27 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_flew_away_l3646_364620


namespace NUMINAMATH_CALUDE_number_of_employees_l3646_364614

-- Define the given constants
def average_salary : ℚ := 1500
def salary_increase : ℚ := 100
def manager_salary : ℚ := 3600

-- Define the number of employees (excluding manager) as a variable
variable (n : ℚ)

-- Define the theorem
theorem number_of_employees : 
  (n * average_salary + manager_salary) / (n + 1) = average_salary + salary_increase →
  n = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_employees_l3646_364614


namespace NUMINAMATH_CALUDE_man_running_opposite_direction_l3646_364698

/-- Proves that the relative speed between a train and a man is equal to the sum of their speeds,
    indicating that the man is running in the opposite direction to the train. -/
theorem man_running_opposite_direction
  (train_length : ℝ)
  (train_speed : ℝ)
  (man_speed : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 110)
  (h2 : train_speed = 40 * 1000 / 3600)
  (h3 : man_speed = 4 * 1000 / 3600)
  (h4 : passing_time = 9) :
  train_length / passing_time = train_speed + man_speed := by
  sorry

#check man_running_opposite_direction

end NUMINAMATH_CALUDE_man_running_opposite_direction_l3646_364698


namespace NUMINAMATH_CALUDE_min_sum_a7_a14_l3646_364671

/-- An arithmetic sequence of positive real numbers -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, 0 < a n ∧ ∃ d : ℝ, a (n + 1) = a n + d

/-- The theorem stating the minimum value of a_7 + a_14 in the given sequence -/
theorem min_sum_a7_a14 (a : ℕ → ℝ) (h_arith : ArithmeticSequence a) (h_prod : a 1 * a 20 = 100) :
  ∀ x y : ℝ, x = a 7 ∧ y = a 14 → x + y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_sum_a7_a14_l3646_364671


namespace NUMINAMATH_CALUDE_ratio_of_segments_l3646_364635

-- Define the points on a line
variable (E F G H : ℝ)

-- Define the conditions
variable (h1 : E < F)
variable (h2 : F < G)
variable (h3 : G < H)
variable (h4 : F - E = 3)
variable (h5 : G - F = 8)
variable (h6 : H - E = 23)

-- Theorem statement
theorem ratio_of_segments :
  (G - E) / (H - F) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l3646_364635


namespace NUMINAMATH_CALUDE_dilation_complex_mapping_l3646_364605

theorem dilation_complex_mapping :
  let center : ℂ := 2 - 3*I
  let scale_factor : ℝ := 3
  let original : ℂ := (5 - 4*I) / 3
  let image : ℂ := 1 + 2*I
  (image - center) = scale_factor • (original - center) := by sorry

end NUMINAMATH_CALUDE_dilation_complex_mapping_l3646_364605


namespace NUMINAMATH_CALUDE_inequality_holds_iff_p_le_eight_l3646_364674

theorem inequality_holds_iff_p_le_eight (p : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < π/2 → (1 + 1/Real.sin x)^3 ≥ p/(Real.tan x)^2) ↔ p ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_p_le_eight_l3646_364674


namespace NUMINAMATH_CALUDE_equation_solution_l3646_364646

open Real

theorem equation_solution (x : ℝ) :
  0 < x ∧ x < π →
  ((Real.sqrt 2014 - Real.sqrt 2013) ^ (tan x)^2 + 
   (Real.sqrt 2014 + Real.sqrt 2013) ^ (-(tan x)^2) = 
   2 * (Real.sqrt 2014 - Real.sqrt 2013)^3) ↔ 
  (x = π/3 ∨ x = 2*π/3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3646_364646


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_l3646_364610

theorem sqrt_meaningful_iff (x : ℝ) : Real.sqrt (x - 1/2) ≥ 0 ↔ x ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_l3646_364610


namespace NUMINAMATH_CALUDE_max_distance_42000km_l3646_364617

/-- Represents the maximum distance a car can travel with tire switching -/
def maxDistanceWithTireSwitch (frontTireLife rear_tire_life : ℕ) : ℕ :=
  min frontTireLife rear_tire_life

/-- Theorem stating the maximum distance a car can travel with specific tire lifespans -/
theorem max_distance_42000km (frontTireLife rearTireLife : ℕ) 
  (h1 : frontTireLife = 42000)
  (h2 : rearTireLife = 56000) :
  maxDistanceWithTireSwitch frontTireLife rearTireLife = 42000 :=
by
  sorry

#eval maxDistanceWithTireSwitch 42000 56000

end NUMINAMATH_CALUDE_max_distance_42000km_l3646_364617


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3646_364604

theorem algebraic_expression_value (a : ℝ) (h : 2 * a^2 - 3 * a = 1) :
  9 * a + 7 - 6 * a^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3646_364604


namespace NUMINAMATH_CALUDE_division_problem_l3646_364644

theorem division_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 725 →
  quotient = 20 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3646_364644


namespace NUMINAMATH_CALUDE_steve_long_letter_time_l3646_364660

/-- Represents the writing habits of Steve --/
structure WritingHabits where
  days_between_letters : ℕ
  minutes_per_regular_letter : ℕ
  minutes_per_page : ℕ
  long_letter_time_factor : ℕ
  total_pages_per_month : ℕ
  days_in_month : ℕ

/-- Calculates the time spent on the long letter at the end of the month --/
def long_letter_time (habits : WritingHabits) : ℕ :=
  let regular_letters := habits.days_in_month / habits.days_between_letters
  let pages_per_regular_letter := habits.minutes_per_regular_letter / habits.minutes_per_page
  let regular_letter_pages := regular_letters * pages_per_regular_letter
  let long_letter_pages := habits.total_pages_per_month - regular_letter_pages
  long_letter_pages * (habits.minutes_per_page * habits.long_letter_time_factor)

/-- Theorem stating that Steve spends 80 minutes writing the long letter --/
theorem steve_long_letter_time :
  ∃ (habits : WritingHabits),
    habits.days_between_letters = 3 ∧
    habits.minutes_per_regular_letter = 20 ∧
    habits.minutes_per_page = 10 ∧
    habits.long_letter_time_factor = 2 ∧
    habits.total_pages_per_month = 24 ∧
    habits.days_in_month = 30 ∧
    long_letter_time habits = 80 := by
  sorry

end NUMINAMATH_CALUDE_steve_long_letter_time_l3646_364660


namespace NUMINAMATH_CALUDE_complement_P_correct_l3646_364669

/-- The set P defined as {x | x² < 1} -/
def P : Set ℝ := {x | x^2 < 1}

/-- The complement of P in ℝ -/
def complement_P : Set ℝ := {x | x ≤ -1 ∨ x ≥ 1}

theorem complement_P_correct : 
  {x : ℝ | x ∉ P} = complement_P := by
  sorry

end NUMINAMATH_CALUDE_complement_P_correct_l3646_364669


namespace NUMINAMATH_CALUDE_train_length_calculation_l3646_364640

/-- The length of the train in meters -/
def train_length : ℝ := 1200

/-- The time (in seconds) it takes for the train to cross a tree -/
def tree_crossing_time : ℝ := 120

/-- The time (in seconds) it takes for the train to pass a platform -/
def platform_crossing_time : ℝ := 160

/-- The length of the platform in meters -/
def platform_length : ℝ := 400

theorem train_length_calculation :
  train_length = 1200 ∧
  (train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time) :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l3646_364640


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3646_364664

/-- 
A quadratic function y = kx^2 + 2x + 1 intersects the x-axis at two points 
if and only if k < 1 and k ≠ 0
-/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 + 2 * x₁ + 1 = 0 ∧ k * x₂^2 + 2 * x₂ + 1 = 0) ↔
  (k < 1 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l3646_364664


namespace NUMINAMATH_CALUDE_min_value_problem_l3646_364649

theorem min_value_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) :
  22.75 ≤ a + 3 * b + 2 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 
    0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧
    1 / (a₀ + 3) + 1 / (b₀ + 3) + 1 / (c₀ + 3) = 1 / 4 ∧
    a₀ + 3 * b₀ + 2 * c₀ = 22.75 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3646_364649


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3646_364659

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 150th term of the specific arithmetic sequence -/
def term_150 : ℝ :=
  arithmetic_sequence 3 4 150

theorem arithmetic_sequence_150th_term :
  term_150 = 599 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l3646_364659


namespace NUMINAMATH_CALUDE_expression_equivalence_l3646_364647

theorem expression_equivalence (a b c : ℝ) (hc : c ≠ 0) :
  ((a - 0.07 * a) + 0.05 * b) / c = (0.93 * a + 0.05 * b) / c := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3646_364647


namespace NUMINAMATH_CALUDE_evaluate_expression_l3646_364686

theorem evaluate_expression : (33 + 12)^2 - (12^2 + 33^2) = 792 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3646_364686


namespace NUMINAMATH_CALUDE_sports_participation_l3646_364672

theorem sports_participation (B C S Ba : ℕ)
  (BC BS BBa CS CBa SBa BCSL : ℕ)
  (h1 : B = 12)
  (h2 : C = 10)
  (h3 : S = 9)
  (h4 : Ba = 6)
  (h5 : BC = 5)
  (h6 : BS = 4)
  (h7 : BBa = 3)
  (h8 : CS = 2)
  (h9 : CBa = 3)
  (h10 : SBa = 2)
  (h11 : BCSL = 1) :
  B + C + S + Ba - BC - BS - BBa - CS - CBa - SBa + BCSL = 19 := by
  sorry

end NUMINAMATH_CALUDE_sports_participation_l3646_364672


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3646_364685

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 1 →
  a 1 + a 3 + a 5 = 21 →
  a 2 + a 4 + a 6 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3646_364685


namespace NUMINAMATH_CALUDE_total_sales_correct_l3646_364616

/-- Represents the sales data for a house --/
structure HouseSale where
  boxes : Nat
  price_per_box : Float
  discount_rate : Float
  discount_threshold : Nat

/-- Represents the sales data for the neighbor --/
structure NeighborSale where
  boxes : Nat
  price_per_box : Float
  exchange_rate : Float

/-- Calculates the total sales in US dollars --/
def calculate_total_sales (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) : Float :=
  sorry

/-- The main theorem to prove --/
theorem total_sales_correct (green : HouseSale) (yellow : HouseSale) (brown : HouseSale) (neighbor : NeighborSale) (tax_rate : Float) :
  let green_sale := HouseSale.mk 3 4 0.1 2
  let yellow_sale := HouseSale.mk 3 (13/3) 0 0
  let brown_sale := HouseSale.mk 9 2 0.05 3
  let neighbor_sale := NeighborSale.mk 3 4.5 1.1
  calculate_total_sales green_sale yellow_sale brown_sale neighbor_sale 0.07 = 57.543 :=
  sorry

end NUMINAMATH_CALUDE_total_sales_correct_l3646_364616


namespace NUMINAMATH_CALUDE_coin_distribution_l3646_364656

/-- The number of ways to distribute n identical objects into k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem coin_distribution : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_coin_distribution_l3646_364656


namespace NUMINAMATH_CALUDE_ball_distribution_ratio_l3646_364666

/-- The number of balls -/
def n : ℕ := 25

/-- The number of bins -/
def m : ℕ := 5

/-- The probability of distributing n balls into m bins such that
    one bin has 3 balls, another has 7 balls, and the other three have 5 balls each -/
noncomputable def p : ℝ := sorry

/-- The probability of distributing n balls equally into m bins (5 balls each) -/
noncomputable def q : ℝ := sorry

/-- Theorem stating that the ratio of p to q is 12 -/
theorem ball_distribution_ratio : p / q = 12 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_ratio_l3646_364666


namespace NUMINAMATH_CALUDE_proposition_four_l3646_364645

theorem proposition_four (p q : Prop) :
  (p → q) ∧ ¬(q → p) → (¬p → ¬q) ∧ ¬(¬q → ¬p) :=
sorry


end NUMINAMATH_CALUDE_proposition_four_l3646_364645


namespace NUMINAMATH_CALUDE_remaining_money_for_seat_and_tape_l3646_364692

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_money_for_seat_and_tape :
  initial_amount - (frame_cost + wheel_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_for_seat_and_tape_l3646_364692


namespace NUMINAMATH_CALUDE_dividend_percentage_calculation_l3646_364623

/-- Calculate the dividend percentage of shares -/
theorem dividend_percentage_calculation 
  (cost_price : ℝ) 
  (desired_interest_rate : ℝ) 
  (market_value : ℝ) : 
  cost_price = 60 →
  desired_interest_rate = 12 / 100 →
  market_value = 45 →
  (market_value * desired_interest_rate) / cost_price * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_dividend_percentage_calculation_l3646_364623


namespace NUMINAMATH_CALUDE_cookie_ratio_l3646_364678

def cookie_problem (initial_white : ℕ) (black_white_difference : ℕ) (remaining_total : ℕ) : Prop :=
  let initial_black : ℕ := initial_white + black_white_difference
  let eaten_black : ℕ := initial_black / 2
  let remaining_black : ℕ := initial_black - eaten_black
  let remaining_white : ℕ := remaining_total - remaining_black
  let eaten_white : ℕ := initial_white - remaining_white
  (eaten_white : ℚ) / initial_white = 3 / 4

theorem cookie_ratio :
  cookie_problem 80 50 85 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l3646_364678


namespace NUMINAMATH_CALUDE_three_teacher_student_pairs_arrangements_l3646_364611

def teacher_student_arrangements (n : ℕ) : ℕ :=
  n.factorial * (2^n)

theorem three_teacher_student_pairs_arrangements :
  teacher_student_arrangements 3 = 48 := by
sorry

end NUMINAMATH_CALUDE_three_teacher_student_pairs_arrangements_l3646_364611


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l3646_364662

theorem average_of_three_numbers (N : ℕ) : 
  15 < N ∧ N < 23 ∧ Even N → 
  (∃ x, x = (8 + 12 + N) / 3 ∧ (x = 12 ∨ x = 14)) :=
sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l3646_364662


namespace NUMINAMATH_CALUDE_parallel_line_length_l3646_364637

/-- Given a triangle with base 15 inches and two parallel lines dividing it into three equal areas,
    the length of the parallel line closer to the base is 5√3 inches. -/
theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 15 →
  (parallel_line / base)^2 = 1/3 →
  parallel_line = 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l3646_364637


namespace NUMINAMATH_CALUDE_mark_has_10_fewer_cards_l3646_364606

/-- The number of Pokemon cards each person has. -/
structure CardCounts where
  lloyd : ℕ
  mark : ℕ
  michael : ℕ

/-- The conditions of the Pokemon card problem. -/
def PokemonCardProblem (c : CardCounts) : Prop :=
  c.mark = 3 * c.lloyd ∧
  c.mark < c.michael ∧
  c.michael = 100 ∧
  c.lloyd + c.mark + c.michael + 80 = 300

/-- The theorem stating that Mark has 10 fewer cards than Michael. -/
theorem mark_has_10_fewer_cards (c : CardCounts) 
  (h : PokemonCardProblem c) : c.michael - c.mark = 10 := by
  sorry

end NUMINAMATH_CALUDE_mark_has_10_fewer_cards_l3646_364606


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3646_364634

theorem baseball_card_value_decrease : ∀ (initial_value : ℝ),
  initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.4)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.46 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3646_364634


namespace NUMINAMATH_CALUDE_scientific_notation_138000_l3646_364679

theorem scientific_notation_138000 : 
  138000 = 1.38 * (10 : ℝ)^5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_138000_l3646_364679


namespace NUMINAMATH_CALUDE_calculate_difference_l3646_364602

theorem calculate_difference (x y z : ℝ) (hx : x = 40) (hy : y = 20) (hz : z = 5) :
  0.8 * (3 * (2 * x))^2 - 0.8 * Real.sqrt ((y / 4)^3 * z^3) = 45980 := by
  sorry

end NUMINAMATH_CALUDE_calculate_difference_l3646_364602


namespace NUMINAMATH_CALUDE_equivalence_sqrt_and_fraction_l3646_364691

theorem equivalence_sqrt_and_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt a + 1 > Real.sqrt b) ↔ 
  (∀ x > 1, a * x + x / (x - 1) > b) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_sqrt_and_fraction_l3646_364691


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3646_364673

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 5 = 28 →
  a 3 + a 4 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3646_364673


namespace NUMINAMATH_CALUDE_monica_has_eight_cookies_left_l3646_364618

/-- The number of cookies left for Monica --/
def cookies_left_for_monica (total_cookies : ℕ) (father_cookies : ℕ) : ℕ :=
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  total_cookies - (father_cookies + mother_cookies + brother_cookies)

/-- Theorem stating that Monica has 8 cookies left --/
theorem monica_has_eight_cookies_left :
  cookies_left_for_monica 30 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_monica_has_eight_cookies_left_l3646_364618


namespace NUMINAMATH_CALUDE_first_degree_function_theorem_l3646_364643

/-- A first-degree function from ℝ to ℝ -/
structure FirstDegreeFunction where
  f : ℝ → ℝ
  k : ℝ
  b : ℝ
  h : ∀ x, f x = k * x + b
  k_nonzero : k ≠ 0

/-- Theorem: If f is a first-degree function satisfying f(f(x)) = 4x + 9 for all x,
    then f(x) = 2x + 3 or f(x) = -2x - 9 -/
theorem first_degree_function_theorem (f : FirstDegreeFunction) 
  (h : ∀ x, f.f (f.f x) = 4 * x + 9) :
  (∀ x, f.f x = 2 * x + 3) ∨ (∀ x, f.f x = -2 * x - 9) := by
  sorry

end NUMINAMATH_CALUDE_first_degree_function_theorem_l3646_364643


namespace NUMINAMATH_CALUDE_max_volume_open_top_box_l3646_364641

/-- Given a square sheet metal of width 60 cm, the maximum volume of an open-top box 
    with a square base that can be created from it is 16000 cm³. -/
theorem max_volume_open_top_box (sheet_width : ℝ) (h : sheet_width = 60) :
  ∃ (x : ℝ), 0 < x ∧ x < sheet_width / 2 ∧
  (∀ (y : ℝ), 0 < y → y < sheet_width / 2 → 
    x * (sheet_width - 2 * x)^2 ≥ y * (sheet_width - 2 * y)^2) ∧
  x * (sheet_width - 2 * x)^2 = 16000 :=
by sorry


end NUMINAMATH_CALUDE_max_volume_open_top_box_l3646_364641


namespace NUMINAMATH_CALUDE_remainder_problem_l3646_364638

theorem remainder_problem (a b : ℕ) (h1 : 3 * a > b) (h2 : a % 5 = 1) (h3 : b % 5 = 4) :
  (3 * a - b) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3646_364638


namespace NUMINAMATH_CALUDE_linear_function_quadrant_slope_l3646_364650

/-- A linear function passing through the first, second, and third quadrants has a slope between 0 and 2 -/
theorem linear_function_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + (2 - k)) →
  (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + (2 - k)) →
  (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + (2 - k)) →
  (∃ x₃ y₃ : ℝ, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + (2 - k)) →
  0 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrant_slope_l3646_364650


namespace NUMINAMATH_CALUDE_car_cyclist_problem_solution_l3646_364619

/-- Represents the speeds and meeting point of a car and cyclist problem -/
structure CarCyclistProblem where
  car_speed : ℝ
  cyclist_speed : ℝ
  meeting_distance_from_A : ℝ

/-- Checks if the given speeds and meeting point satisfy the problem conditions -/
def is_valid_solution (p : CarCyclistProblem) : Prop :=
  let total_distance := 80
  let time_to_meet := 1.5
  let distance_after_one_hour := 24
  let car_distance_one_hour := p.car_speed
  let cyclist_distance_one_hour := p.cyclist_speed
  let car_total_distance := p.car_speed * time_to_meet
  let cyclist_total_distance := p.cyclist_speed * 1.25  -- Cyclist rests for 1 hour

  -- Condition 1: After one hour, they are 24 km apart
  (total_distance - (car_distance_one_hour + cyclist_distance_one_hour) = distance_after_one_hour) ∧
  -- Condition 2: They meet after 90 minutes
  (car_total_distance + cyclist_total_distance = total_distance) ∧
  -- Condition 3: Meeting point is correct
  (p.meeting_distance_from_A = car_total_distance)

/-- The theorem stating that the given solution satisfies the problem conditions -/
theorem car_cyclist_problem_solution :
  is_valid_solution ⟨40, 16, 60⟩ := by
  sorry

end NUMINAMATH_CALUDE_car_cyclist_problem_solution_l3646_364619


namespace NUMINAMATH_CALUDE_complex_root_ratio_l3646_364670

theorem complex_root_ratio (m n : ℝ) : 
  (Complex.I * 2 + 1) ^ 2 + m * (Complex.I * 2 + 1) + n = 0 → m / n = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_ratio_l3646_364670


namespace NUMINAMATH_CALUDE_sequence_classification_l3646_364607

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = aⁿ - 1 (a is a non-zero real number),
    the sequence {aₙ} is either an arithmetic sequence or a geometric sequence. -/
theorem sequence_classification (a : ℝ) (ha : a ≠ 0) :
  let S : ℕ → ℝ := λ n => a^n - 1
  let a_seq : ℕ → ℝ := λ n => S n - S (n-1)
  (∀ n : ℕ, n > 1 → a_seq (n+1) - a_seq n = 0) ∨
  (∀ n : ℕ, n > 2 → a_seq (n+1) / a_seq n = a) :=
by sorry

end NUMINAMATH_CALUDE_sequence_classification_l3646_364607


namespace NUMINAMATH_CALUDE_trees_survived_vs_died_l3646_364675

theorem trees_survived_vs_died (initial_trees : ℕ) (trees_died : ℕ) : 
  initial_trees = 13 → trees_died = 6 → (initial_trees - trees_died) - trees_died = 1 := by
  sorry

end NUMINAMATH_CALUDE_trees_survived_vs_died_l3646_364675


namespace NUMINAMATH_CALUDE_age_of_other_man_l3646_364683

/-- Given a group of men where two are replaced by women, prove the age of a specific man. -/
theorem age_of_other_man
  (n : ℕ)  -- Total number of people
  (m : ℕ)  -- Number of men initially
  (w : ℕ)  -- Number of women replacing men
  (age_increase : ℝ)  -- Increase in average age
  (known_man_age : ℝ)  -- Age of the known man
  (women_avg_age : ℝ)  -- Average age of the women
  (h1 : n = 8)  -- Total number of people is 8
  (h2 : m = 8)  -- Initial number of men is 8
  (h3 : w = 2)  -- Number of women replacing men is 2
  (h4 : age_increase = 2)  -- Average age increases by 2 years
  (h5 : known_man_age = 24)  -- One man is 24 years old
  (h6 : women_avg_age = 30)  -- Average age of women is 30 years
  : ∃ (other_man_age : ℝ), other_man_age = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l3646_364683


namespace NUMINAMATH_CALUDE_fraction_of_savings_used_for_bills_l3646_364629

def weekly_savings : ℚ := 25
def weeks_saved : ℕ := 6
def dad_contribution : ℚ := 70
def coat_cost : ℚ := 170

theorem fraction_of_savings_used_for_bills :
  let total_savings := weekly_savings * weeks_saved
  let remaining_cost := coat_cost - dad_contribution
  let amount_for_bills := total_savings - remaining_cost
  amount_for_bills / total_savings = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_savings_used_for_bills_l3646_364629


namespace NUMINAMATH_CALUDE_odot_calculation_l3646_364667

def odot (a b : ℝ) : ℝ := a * b + (a - b)

theorem odot_calculation : odot (odot 3 2) 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_odot_calculation_l3646_364667


namespace NUMINAMATH_CALUDE_black_area_calculation_l3646_364687

theorem black_area_calculation (large_side : ℝ) (small_side : ℝ) :
  large_side = 12 →
  small_side = 5 →
  large_side^2 - 2 * small_side^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_black_area_calculation_l3646_364687


namespace NUMINAMATH_CALUDE_remainder_theorem_l3646_364661

theorem remainder_theorem (A : ℕ) 
  (h1 : A % 1981 = 35) 
  (h2 : A % 1982 = 35) : 
  A % 14 = 7 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3646_364661


namespace NUMINAMATH_CALUDE_bird_weights_solution_l3646_364665

theorem bird_weights_solution :
  ∃! (A B V G : ℕ+),
    A + B + V + G = 32 ∧
    V < G ∧
    V + G < B ∧
    A < V + B ∧
    G + B < A + V ∧
    A = 13 ∧ B = 10 ∧ V = 4 ∧ G = 5 :=
by sorry

end NUMINAMATH_CALUDE_bird_weights_solution_l3646_364665


namespace NUMINAMATH_CALUDE_problem_solution_l3646_364631

theorem problem_solution : ∃ N : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  N / sum = 2 * diff ∧ N % sum = 80 ∧ N = 220080 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3646_364631


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l3646_364609

/-- The number of fruit types available to Joe -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has per day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruits

/-- The probability of choosing the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of not eating at least two different kinds of fruits -/
def prob_not_varied : ℚ := num_fruits * prob_same_fruit

/-- The probability of eating at least two different kinds of fruits -/
def prob_varied : ℚ := 1 - prob_not_varied

theorem joe_fruit_probability : prob_varied = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l3646_364609


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l3646_364625

theorem triangle_angle_problem (x : ℝ) : 
  x > 0 ∧ 
  x + 2*x + 40 = 180 → 
  x = 140/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l3646_364625


namespace NUMINAMATH_CALUDE_inscribed_circle_triangle_shortest_side_l3646_364699

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  s1 : ℝ
  /-- The length of the second segment of the divided side -/
  s2 : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ

/-- Theorem stating that for a triangle with an inscribed circle of radius 5 units
    that divides one side into segments of 4 and 10 units, the shortest side is 30 units -/
theorem inscribed_circle_triangle_shortest_side
  (t : InscribedCircleTriangle)
  (h1 : t.r = 5)
  (h2 : t.s1 = 4)
  (h3 : t.s2 = 10) :
  t.shortest_side = 30 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_triangle_shortest_side_l3646_364699


namespace NUMINAMATH_CALUDE_congruence_from_equation_l3646_364621

theorem congruence_from_equation (a b : ℕ+) (h : a^(b : ℕ) - b^(a : ℕ) = 1008) :
  a ≡ b [ZMOD 1008] := by
  sorry

end NUMINAMATH_CALUDE_congruence_from_equation_l3646_364621


namespace NUMINAMATH_CALUDE_true_absolute_error_example_l3646_364651

/-- The true absolute error of a₀ with respect to a -/
def trueAbsoluteError (a₀ a : ℝ) : ℝ := |a - a₀|

theorem true_absolute_error_example : 
  trueAbsoluteError 245.2 246 = 0.8 := by sorry

end NUMINAMATH_CALUDE_true_absolute_error_example_l3646_364651
