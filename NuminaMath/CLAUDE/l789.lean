import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_253_247_l789_78971

theorem difference_of_squares_253_247 : 253^2 - 247^2 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_253_247_l789_78971


namespace NUMINAMATH_CALUDE_four_terms_after_substitution_l789_78910

-- Define the expression with a variable for the asterisk
def expression (a : ℝ → ℝ) : ℝ → ℝ := λ x => (x^4 - 3)^2 + (x^3 + a x)^2

-- Define the proposed replacement for the asterisk
def replacement : ℝ → ℝ := λ x => x^3 + 3*x

-- The theorem to prove
theorem four_terms_after_substitution :
  ∃ (c₁ c₂ c₃ c₄ : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (∀ x, expression replacement x = c₁ * x^n₁ + c₂ * x^n₂ + c₃ * x^n₃ + c₄ * x^n₄) ∧
    (n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₃ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄) :=
by
  sorry

end NUMINAMATH_CALUDE_four_terms_after_substitution_l789_78910


namespace NUMINAMATH_CALUDE_solution_in_interval_l789_78913

theorem solution_in_interval :
  ∃ x₀ : ℝ, (Real.log x₀ + x₀ = 4) ∧ (2 < x₀) ∧ (x₀ < 3) := by
  sorry

#check solution_in_interval

end NUMINAMATH_CALUDE_solution_in_interval_l789_78913


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_is_200_l789_78917

/-- The area of an octagon inscribed in a square, where each vertex of the octagon
    bisects the sides of the square and the perimeter of the square is 80 centimeters. -/
def inscribedOctagonArea (square_perimeter : ℝ) (octagon_bisects_square : Prop) : ℝ :=
  sorry

/-- Theorem stating that the area of the inscribed octagon is 200 square centimeters. -/
theorem inscribed_octagon_area_is_200 :
  inscribedOctagonArea 80 true = 200 := by sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_is_200_l789_78917


namespace NUMINAMATH_CALUDE_area_TURS_l789_78986

/-- Rectangle PQRS with trapezoid TURS inside -/
structure Geometry where
  /-- Width of rectangle PQRS -/
  width : ℝ
  /-- Height of rectangle PQRS -/
  height : ℝ
  /-- Area of rectangle PQRS -/
  area_PQRS : ℝ
  /-- Distance of T from S -/
  ST_distance : ℝ
  /-- Distance of U from R -/
  UR_distance : ℝ
  /-- Width is 6 units -/
  width_eq : width = 6
  /-- Height is 4 units -/
  height_eq : height = 4
  /-- Area of PQRS is 24 square units -/
  area_eq : area_PQRS = 24
  /-- ST distance is 1 unit -/
  ST_eq : ST_distance = 1
  /-- UR distance is 1 unit -/
  UR_eq : UR_distance = 1

/-- The area of trapezoid TURS is 20 square units -/
theorem area_TURS (g : Geometry) : Real.sqrt ((g.width - 2 * g.ST_distance) * g.height + g.ST_distance * g.height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_area_TURS_l789_78986


namespace NUMINAMATH_CALUDE_parallel_line_through_A_l789_78991

-- Define the point A
def A : ℝ × ℝ × ℝ := (-2, 3, 1)

-- Define the planes that form the given line
def plane1 (x y z : ℝ) : Prop := x - 2*y - z - 2 = 0
def plane2 (x y z : ℝ) : Prop := 2*x + 3*y - z + 1 = 0

-- Define the direction vector of the given line
def direction_vector : ℝ × ℝ × ℝ := (5, -1, 7)

-- Define the equation of the parallel line passing through A
def parallel_line (x y z : ℝ) : Prop :=
  (x + 2) / 5 = (y - 3) / (-1) ∧ (y - 3) / (-1) = (z - 1) / 7

-- Theorem statement
theorem parallel_line_through_A :
  ∀ (x y z : ℝ), 
    (∃ (t : ℝ), x = -2 + 5*t ∧ y = 3 - t ∧ z = 1 + 7*t) →
    parallel_line x y z :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_A_l789_78991


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l789_78990

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents an arrangement of rectangles -/
structure Arrangement where
  height : ℝ
  width : ℝ

/-- Theorem: It is possible to arrange five identical rectangles with perimeter 10
    to form a single rectangle with perimeter 22 -/
theorem rectangle_arrangement_exists : ∃ (small : Rectangle) (arr : Arrangement),
  perimeter small = 10 ∧
  arr.height = 5 * small.length ∧
  arr.width = small.width ∧
  2 * (arr.height + arr.width) = 22 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l789_78990


namespace NUMINAMATH_CALUDE_increasing_quadratic_max_value_function_inequality_positive_reals_l789_78902

-- Statement 1
theorem increasing_quadratic (x : ℝ) (h : x > 0) :
  Monotone (fun x => 2 * x^2 + x + 1) := by sorry

-- Statement 2
theorem max_value_function (x : ℝ) (h : x > 0) :
  (2 - 3*x - 4/x) ≤ (2 - 4*Real.sqrt 3) := by sorry

-- Statement 3
theorem inequality_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z := by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_max_value_function_inequality_positive_reals_l789_78902


namespace NUMINAMATH_CALUDE_angle_between_vectors_l789_78938

/-- The angle between two planar vectors -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = -1)  -- dot product condition
  (h2 : a.1^2 + a.2^2 = 4)           -- |a| = 2 condition
  (h3 : b.1^2 + b.2^2 = 1)           -- |b| = 1 condition
  : angle_between a b = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l789_78938


namespace NUMINAMATH_CALUDE_xy_inequality_l789_78962

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : x > 0) (hy : y > 0) :
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) := by
  sorry

end NUMINAMATH_CALUDE_xy_inequality_l789_78962


namespace NUMINAMATH_CALUDE_product_of_reversed_digits_l789_78933

theorem product_of_reversed_digits (A B : ℕ) (k : ℕ) : 
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 →
  (10 * A + B) * (10 * B + A) = k →
  (k + 1) % 101 = 0 →
  k = 403 := by
sorry

end NUMINAMATH_CALUDE_product_of_reversed_digits_l789_78933


namespace NUMINAMATH_CALUDE_profit_increase_l789_78972

theorem profit_increase (initial_profit : ℝ) (x : ℝ) : 
  -- Conditions
  (initial_profit * (1 + x / 100) * 0.8 * 1.5 = initial_profit * 1.6200000000000001) →
  -- Conclusion
  x = 35 := by
sorry

end NUMINAMATH_CALUDE_profit_increase_l789_78972


namespace NUMINAMATH_CALUDE_rulers_remaining_l789_78958

theorem rulers_remaining (initial_rulers : ℕ) (removed_rulers : ℕ) : 
  initial_rulers = 14 → removed_rulers = 11 → initial_rulers - removed_rulers = 3 :=
by sorry

end NUMINAMATH_CALUDE_rulers_remaining_l789_78958


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l789_78985

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (4 + ((x^6 - 3*x^3 + 2) / (3*x^3))^2) = (x^6 - 3*x^3 + 2) / (3*x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l789_78985


namespace NUMINAMATH_CALUDE_van_speed_ratio_l789_78966

theorem van_speed_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ)
  (h1 : distance = 465)
  (h2 : original_time = 5)
  (h3 : new_speed = 62)
  : (distance / new_speed) / original_time = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_van_speed_ratio_l789_78966


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l789_78930

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l789_78930


namespace NUMINAMATH_CALUDE_surface_area_after_removing_corners_l789_78965

/-- Represents the dimensions of a cube in centimeters -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (d : CubeDimensions) : ℝ :=
  6 * d.length * d.width

/-- Represents the problem setup -/
structure CubeWithCornersRemoved where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions

/-- The main theorem to be proved -/
theorem surface_area_after_removing_corners
  (c : CubeWithCornersRemoved)
  (h1 : c.originalCube.length = 4)
  (h2 : c.originalCube.width = 4)
  (h3 : c.originalCube.height = 4)
  (h4 : c.cornerCube.length = 2)
  (h5 : c.cornerCube.width = 2)
  (h6 : c.cornerCube.height = 2) :
  surfaceArea c.originalCube = 96 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_after_removing_corners_l789_78965


namespace NUMINAMATH_CALUDE_residue_calculation_l789_78964

theorem residue_calculation : (207 * 13 - 18 * 8 + 5) % 16 = 8 := by
  sorry

end NUMINAMATH_CALUDE_residue_calculation_l789_78964


namespace NUMINAMATH_CALUDE_lineup_selection_theorem_l789_78929

/-- The number of ways to select a lineup of 6 players from a team of 15 players -/
def lineup_selection_ways : ℕ := 3603600

/-- The size of the basketball team -/
def team_size : ℕ := 15

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 6

theorem lineup_selection_theorem :
  (Finset.range team_size).card.factorial / 
  ((team_size - positions_to_fill).factorial) = lineup_selection_ways :=
sorry

end NUMINAMATH_CALUDE_lineup_selection_theorem_l789_78929


namespace NUMINAMATH_CALUDE_tan_double_angle_l789_78961

theorem tan_double_angle (x : ℝ) (h : Real.tan x = 2) : Real.tan (2 * x) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l789_78961


namespace NUMINAMATH_CALUDE_range_of_f_l789_78946

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {π/4, Real.arctan 2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l789_78946


namespace NUMINAMATH_CALUDE_min_h_10_l789_78909

/-- A function is stringent if f(x) + f(y) > 2y^2 for all positive integers x and y -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > 2 * y.val ^ 2

/-- The sum of h from 1 to 15 -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_10 (h : ℕ+ → ℤ) (stringent_h : Stringent h) 
    (min_sum : ∀ g : ℕ+ → ℤ, Stringent g → SumH g ≥ SumH h) : 
    h ⟨10, by norm_num⟩ ≥ 136 := by
  sorry

end NUMINAMATH_CALUDE_min_h_10_l789_78909


namespace NUMINAMATH_CALUDE_orange_banana_ratio_l789_78923

/-- Proves that the ratio of oranges to bananas is 2:1 given the problem conditions --/
theorem orange_banana_ratio :
  ∀ (orange_price pear_price banana_price : ℚ),
  pear_price - orange_price = banana_price →
  orange_price + pear_price = 120 →
  pear_price = 90 →
  200 * banana_price + (24000 - 200 * banana_price) / orange_price = 400 →
  (24000 - 200 * banana_price) / orange_price / 200 = 2 :=
by
  sorry

#check orange_banana_ratio

end NUMINAMATH_CALUDE_orange_banana_ratio_l789_78923


namespace NUMINAMATH_CALUDE_union_of_sets_l789_78977

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l789_78977


namespace NUMINAMATH_CALUDE_intersection_A_B_is_positive_reals_l789_78903

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x + 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_A_B_is_positive_reals :
  A ∩ B = Set.Ioo 0 (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_A_B_is_positive_reals_l789_78903


namespace NUMINAMATH_CALUDE_andrew_toast_count_l789_78941

/-- The cost of breakfast for Dale and Andrew -/
def total_cost : ℕ := 15

/-- The cost of a slice of toast -/
def toast_cost : ℕ := 1

/-- The cost of an egg -/
def egg_cost : ℕ := 3

/-- The number of slices of toast Dale had -/
def dale_toast : ℕ := 2

/-- The number of eggs Dale had -/
def dale_eggs : ℕ := 2

/-- The number of eggs Andrew had -/
def andrew_eggs : ℕ := 2

/-- The number of slices of toast Andrew had -/
def andrew_toast : ℕ := 1

theorem andrew_toast_count :
  total_cost = 
    dale_toast * toast_cost + dale_eggs * egg_cost + 
    andrew_toast * toast_cost + andrew_eggs * egg_cost :=
by sorry

end NUMINAMATH_CALUDE_andrew_toast_count_l789_78941


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l789_78992

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back --/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back --/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 15)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  oranges_to_put_back fs = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l789_78992


namespace NUMINAMATH_CALUDE_homework_completion_difference_l789_78976

-- Define the efficiency rates and Tim's completion time
def samuel_efficiency : ℝ := 0.90
def sarah_efficiency : ℝ := 0.75
def tim_efficiency : ℝ := 0.80
def tim_completion_time : ℝ := 45

-- Define the theorem
theorem homework_completion_difference :
  let base_time := tim_completion_time / tim_efficiency
  let samuel_time := base_time / samuel_efficiency
  let sarah_time := base_time / sarah_efficiency
  sarah_time - samuel_time = 12.5 := by
sorry

end NUMINAMATH_CALUDE_homework_completion_difference_l789_78976


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l789_78973

/-- Given two rectangles with equal area, where one rectangle has dimensions 12 inches by 15 inches,
    and the other rectangle has a length of 6 inches, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 6)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l789_78973


namespace NUMINAMATH_CALUDE_sara_onions_l789_78924

theorem sara_onions (sally_onions fred_onions total_onions : ℕ) 
  (h1 : sally_onions = 5)
  (h2 : fred_onions = 9)
  (h3 : total_onions = 18)
  : total_onions - (sally_onions + fred_onions) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_onions_l789_78924


namespace NUMINAMATH_CALUDE_pool_and_deck_area_l789_78935

/-- Given a rectangular pool with dimensions 10 feet by 12 feet and a surrounding deck
    with a uniform width of 4 feet, the total area of the pool and deck is 360 square feet. -/
theorem pool_and_deck_area :
  let pool_length : ℕ := 12
  let pool_width : ℕ := 10
  let deck_width : ℕ := 4
  let total_length : ℕ := pool_length + 2 * deck_width
  let total_width : ℕ := pool_width + 2 * deck_width
  total_length * total_width = 360 := by
  sorry

end NUMINAMATH_CALUDE_pool_and_deck_area_l789_78935


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l789_78911

theorem simplify_complex_fraction (b : ℝ) (h : b ≠ 2) :
  2 - (1 / (1 + b / (2 - b))) = 1 + b / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l789_78911


namespace NUMINAMATH_CALUDE_platform_crossing_time_l789_78995

def train_speed : Real := 36  -- km/h
def time_to_cross_pole : Real := 12  -- seconds
def time_to_cross_platform : Real := 44.99736021118311  -- seconds

theorem platform_crossing_time :
  time_to_cross_platform = 44.99736021118311 :=
by sorry

end NUMINAMATH_CALUDE_platform_crossing_time_l789_78995


namespace NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l789_78948

theorem sqrt_x_minus_8_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 8) ↔ x ≥ 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_8_meaningful_l789_78948


namespace NUMINAMATH_CALUDE_triangle_theorem_l789_78996

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle)
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * Real.cos t.C)
  (h2 : t.a + t.b = 4)
  (h3 : t.c = 2) :
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l789_78996


namespace NUMINAMATH_CALUDE_parallelogram_area_28_32_l789_78955

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters -/
theorem parallelogram_area_28_32 : parallelogramArea 28 32 = 896 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_28_32_l789_78955


namespace NUMINAMATH_CALUDE_value_of_b_l789_78905

theorem value_of_b (a b : ℝ) (h1 : a = 5) (h2 : a^2 + a*b = 60) : b = 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l789_78905


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l789_78997

def rectangle_width : ℝ := 20
def rectangle_height : ℝ := 15
def circle_diameter : ℝ := 8

theorem greatest_distance_between_circle_centers : 
  ∃ (d : ℝ), d = Real.sqrt 193 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    0 ≤ x₁ ∧ x₁ ≤ rectangle_width ∧
    0 ≤ y₁ ∧ y₁ ≤ rectangle_height ∧
    0 ≤ x₂ ∧ x₂ ≤ rectangle_width ∧
    0 ≤ y₂ ∧ y₂ ≤ rectangle_height ∧
    circle_diameter / 2 ≤ x₁ ∧ x₁ ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y₁ ∧ y₁ ≤ rectangle_height - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ x₂ ∧ x₂ ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y₂ ∧ y₂ ≤ rectangle_height - circle_diameter / 2 →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l789_78997


namespace NUMINAMATH_CALUDE_stock_market_value_l789_78940

theorem stock_market_value 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_yield : ℝ) 
  (h1 : dividend_rate = 0.07) 
  (h2 : market_yield = 0.10) : 
  (dividend_rate * face_value) / market_yield = 0.7 * face_value := by
  sorry

end NUMINAMATH_CALUDE_stock_market_value_l789_78940


namespace NUMINAMATH_CALUDE_contacts_in_second_box_l789_78939

/-- The number of contacts in the first box -/
def first_box_contacts : ℕ := 50

/-- The price of the first box in cents -/
def first_box_price : ℕ := 2500

/-- The price of the second box in cents -/
def second_box_price : ℕ := 3300

/-- The number of contacts that equal $1 worth in the chosen box -/
def contacts_per_dollar : ℕ := 3

/-- The number of contacts in the second box -/
def second_box_contacts : ℕ := 99

theorem contacts_in_second_box :
  (first_box_price / first_box_contacts > second_box_price / second_box_contacts) ∧
  (second_box_price / second_box_contacts = 100 / contacts_per_dollar) →
  second_box_contacts = 99 := by
  sorry

end NUMINAMATH_CALUDE_contacts_in_second_box_l789_78939


namespace NUMINAMATH_CALUDE_average_attendance_theorem_l789_78963

/-- Represents the attendance data for a week -/
structure WeekAttendance where
  totalStudents : ℕ
  mondayAbsence : ℚ
  tuesdayAbsence : ℚ
  wednesdayAbsence : ℚ
  thursdayAbsence : ℚ
  fridayAbsence : ℚ

/-- Calculates the average number of students present in a week -/
def averageAttendance (w : WeekAttendance) : ℚ :=
  let mondayPresent := w.totalStudents * (1 - w.mondayAbsence)
  let tuesdayPresent := w.totalStudents * (1 - w.tuesdayAbsence)
  let wednesdayPresent := w.totalStudents * (1 - w.wednesdayAbsence)
  let thursdayPresent := w.totalStudents * (1 - w.thursdayAbsence)
  let fridayPresent := w.totalStudents * (1 - w.fridayAbsence)
  (mondayPresent + tuesdayPresent + wednesdayPresent + thursdayPresent + fridayPresent) / 5

theorem average_attendance_theorem (w : WeekAttendance) 
    (h1 : w.totalStudents = 50)
    (h2 : w.mondayAbsence = 1/10)
    (h3 : w.tuesdayAbsence = 3/25)
    (h4 : w.wednesdayAbsence = 3/20)
    (h5 : w.thursdayAbsence = 1/12.5)
    (h6 : w.fridayAbsence = 1/20) : 
  averageAttendance w = 45 := by
sorry

end NUMINAMATH_CALUDE_average_attendance_theorem_l789_78963


namespace NUMINAMATH_CALUDE_sine_intersection_theorem_l789_78927

theorem sine_intersection_theorem (a b : ℕ) (h : a ≠ b) :
  ∃ c : ℕ, c ≠ a ∧ c ≠ b ∧
  ∀ x : ℝ, Real.sin (a * x) = Real.sin (b * x) →
            Real.sin (c * x) = Real.sin (a * x) :=
by sorry

end NUMINAMATH_CALUDE_sine_intersection_theorem_l789_78927


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l789_78957

-- Define the operation ⊗ (using ⊗ instead of ⭐ as it's more readily available)
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt y))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : bowtie 3 h = 5 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l789_78957


namespace NUMINAMATH_CALUDE_no_real_solutions_l789_78950

theorem no_real_solutions :
  ¬∃ (x : ℝ), (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l789_78950


namespace NUMINAMATH_CALUDE_no_integer_solution_l789_78982

theorem no_integer_solution : ¬ ∃ (a k : ℤ), 2 * a^2 - 7 * k + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l789_78982


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l789_78974

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 110 → 
  train_speed_kmph = 36 → 
  bridge_length = 170 → 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry


end NUMINAMATH_CALUDE_train_bridge_crossing_time_l789_78974


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l789_78999

/-- The mapping f: ℝ² → ℝ² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2,1) is the pre-image of (3,1) under the mapping f -/
theorem preimage_of_3_1 : f (2, 1) = (3, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l789_78999


namespace NUMINAMATH_CALUDE_angle_C_in_similar_triangles_l789_78932

-- Define the triangles and their properties
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

-- Theorem statement
theorem angle_C_in_similar_triangles (ABC DEF : Triangle) 
  (h1 : similar ABC DEF) (h2 : ABC.A = 30) (h3 : DEF.B = 30) : ABC.C = 120 := by
  sorry


end NUMINAMATH_CALUDE_angle_C_in_similar_triangles_l789_78932


namespace NUMINAMATH_CALUDE_sum_fusion_2020_l789_78926

/-- A number is a sum fusion number if it's equal to the square difference of two consecutive even numbers. -/
def IsSumFusionNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2

/-- 2020 is a sum fusion number. -/
theorem sum_fusion_2020 : IsSumFusionNumber 2020 := by
  sorry

#check sum_fusion_2020

end NUMINAMATH_CALUDE_sum_fusion_2020_l789_78926


namespace NUMINAMATH_CALUDE_triangle_sin_c_max_l789_78943

/-- Given a triangle ABC with sides a, b, c, prove that if the dot products of its vectors
    satisfy the given condition, then sin C is less than or equal to √7/3 -/
theorem triangle_sin_c_max (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_dot_product : b * c * (b^2 + c^2 - a^2) + 2 * a * c * (a^2 + c^2 - b^2) = 
                   3 * a * b * (a^2 + b^2 - c^2)) :
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sin_c_max_l789_78943


namespace NUMINAMATH_CALUDE_dans_candy_purchase_l789_78959

def candy_problem (initial_money remaining_money candy_cost : ℕ) : Prop :=
  let spent_money := initial_money - remaining_money
  let num_candy_bars := spent_money / candy_cost
  num_candy_bars = 1

theorem dans_candy_purchase :
  candy_problem 4 1 3 := by sorry

end NUMINAMATH_CALUDE_dans_candy_purchase_l789_78959


namespace NUMINAMATH_CALUDE_greatest_number_of_factors_l789_78989

/-- The greatest number of positive factors for b^n given the conditions -/
def max_factors : ℕ := 561

/-- b is a positive integer less than or equal to 15 -/
def b : ℕ := 12

/-- n is a perfect square less than or equal to 16 -/
def n : ℕ := 16

/-- Theorem stating that max_factors is the greatest number of positive factors of b^n -/
theorem greatest_number_of_factors :
  ∀ (b' n' : ℕ), 
    b' > 0 → b' ≤ 15 → 
    n' > 0 → ∃ (k : ℕ), n' = k^2 → n' ≤ 16 →
    (Nat.factors (b'^n')).length ≤ max_factors :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_factors_l789_78989


namespace NUMINAMATH_CALUDE_concave_probability_is_one_third_l789_78920

/-- A digit is a natural number between 4 and 8 inclusive -/
def Digit : Type := { n : ℕ // 4 ≤ n ∧ n ≤ 8 }

/-- A three-digit number is a tuple of three digits -/
def ThreeDigitNumber : Type := Digit × Digit × Digit

/-- A concave number is a three-digit number where the first and third digits are greater than the second -/
def is_concave (n : ThreeDigitNumber) : Prop :=
  let (a, b, c) := n
  a.val > b.val ∧ c.val > b.val

/-- The set of all possible three-digit numbers with distinct digits from {4,5,6,7,8} -/
def all_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The set of all concave numbers from all_numbers -/
def concave_numbers : Finset ThreeDigitNumber :=
  sorry

/-- The probability of a randomly chosen three-digit number being concave -/
def concave_probability : ℚ :=
  (Finset.card concave_numbers : ℚ) / (Finset.card all_numbers : ℚ)

theorem concave_probability_is_one_third :
  concave_probability = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_concave_probability_is_one_third_l789_78920


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l789_78954

/-- A line passing through (1,2) with equal x and y intercepts -/
structure Line where
  slope : ℝ
  intercept : ℝ
  passes_through_one_two : 2 = slope * 1 + intercept
  equal_intercepts : intercept = slope * intercept

/-- A point (a,b) on the line -/
structure Point (l : Line) where
  a : ℝ
  b : ℝ
  on_line : b = l.slope * a + l.intercept

/-- The theorem statement -/
theorem min_value_of_exponential_sum (l : Line) (p : Point l) 
  (h_non_zero : l.intercept ≠ 0) :
  (3 : ℝ) ^ p.a + (3 : ℝ) ^ p.b ≥ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l789_78954


namespace NUMINAMATH_CALUDE_perimeter_is_158_l789_78944

/-- Represents a tile in the figure -/
structure Tile where
  width : ℕ := 2
  height : ℕ := 4

/-- Represents the figure composed of tiles -/
structure Figure where
  tiles : List Tile
  horizontalEdges : ℕ := 45
  verticalEdges : ℕ := 34

/-- Calculates the perimeter of the figure -/
def calculatePerimeter (f : Figure) : ℕ :=
  (f.horizontalEdges + f.verticalEdges) * 2

/-- Theorem stating that the perimeter of the specific figure is 158 -/
theorem perimeter_is_158 (f : Figure) : calculatePerimeter f = 158 := by
  sorry

#check perimeter_is_158

end NUMINAMATH_CALUDE_perimeter_is_158_l789_78944


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l789_78969

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 8 * x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x > 5/3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l789_78969


namespace NUMINAMATH_CALUDE_tims_contribution_l789_78918

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the final number of bales
def final_bales : ℕ := 54

-- Define Tim's contribution
def tims_bales : ℕ := final_bales - initial_bales

-- Theorem to prove
theorem tims_contribution : tims_bales = 26 := by sorry

end NUMINAMATH_CALUDE_tims_contribution_l789_78918


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l789_78928

-- Define propositions p and q
variable (p q : Prop)

-- Define Proposition A
def PropA : Prop := p → q

-- Define Proposition B
def PropB : Prop := p ↔ q

-- Theorem to prove
theorem prop_a_necessary_not_sufficient_for_prop_b :
  (PropA p q → PropB p q) ∧ ¬(PropB p q → PropA p q) :=
sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l789_78928


namespace NUMINAMATH_CALUDE_work_completion_time_l789_78968

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 9

/-- The number of days x needs to finish the remaining work after y left -/
def x_remaining : ℝ := 8

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 20

theorem work_completion_time :
  x_days = 20 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l789_78968


namespace NUMINAMATH_CALUDE_gate_width_scientific_notation_l789_78914

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  prop : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem gate_width_scientific_notation :
  toScientificNotation 0.000000007 = ScientificNotation.mk 7 (-9) sorry := by
  sorry

end NUMINAMATH_CALUDE_gate_width_scientific_notation_l789_78914


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l789_78978

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ

/-- The roots of a quadratic polynomial -/
structure Roots where
  x₁ : ℤ
  x₂ : ℤ

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℤ) : Prop :=
  ∃ m k : ℤ, m ≠ 1 ∧ m ≠ -1 ∧ k ≠ 1 ∧ k ≠ -1 ∧ n = m * k

/-- Main theorem -/
theorem quadratic_roots_imply_composite
  (p : QuadraticPolynomial)
  (r : Roots)
  (h₁ : r.x₁ * r.x₁ + p.a * r.x₁ + p.b = 0)
  (h₂ : r.x₂ * r.x₂ + p.a * r.x₂ + p.b = 0)
  (h₃ : |r.x₁| > 2)
  (h₄ : |r.x₂| > 2) :
  IsComposite (p.a + p.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_composite_l789_78978


namespace NUMINAMATH_CALUDE_julians_comic_book_pages_l789_78907

theorem julians_comic_book_pages 
  (frames_per_page : ℝ) 
  (total_frames : ℕ) 
  (h1 : frames_per_page = 143.0) 
  (h2 : total_frames = 1573) : 
  ⌊(total_frames : ℝ) / frames_per_page⌋ = 11 := by
sorry

end NUMINAMATH_CALUDE_julians_comic_book_pages_l789_78907


namespace NUMINAMATH_CALUDE_max_value_w_l789_78951

theorem max_value_w (p q : ℝ) (w : ℝ) 
  (hw : w = Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q))
  (h1 : 2 * p - q ≥ 0)
  (h2 : 3 * q - 2 * p ≥ 0)
  (h3 : 6 - 2 * q ≥ 0) :
  w ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_w_l789_78951


namespace NUMINAMATH_CALUDE_fruit_cost_difference_l789_78942

theorem fruit_cost_difference : 
  let grapes_kg : ℝ := 7
  let grapes_price : ℝ := 70
  let grapes_discount : ℝ := 0.10
  let grapes_tax : ℝ := 0.05

  let mangoes_kg : ℝ := 9
  let mangoes_price : ℝ := 55
  let mangoes_discount : ℝ := 0.05
  let mangoes_tax : ℝ := 0.07

  let apples_kg : ℝ := 5
  let apples_price : ℝ := 40
  let apples_discount : ℝ := 0.08
  let apples_tax : ℝ := 0.03

  let oranges_kg : ℝ := 3
  let oranges_price : ℝ := 30
  let oranges_discount : ℝ := 0.15
  let oranges_tax : ℝ := 0.06

  let mangoes_cost := mangoes_kg * mangoes_price * (1 - mangoes_discount) * (1 + mangoes_tax)
  let apples_cost := apples_kg * apples_price * (1 - apples_discount) * (1 + apples_tax)

  mangoes_cost - apples_cost = 313.6475 := by sorry

end NUMINAMATH_CALUDE_fruit_cost_difference_l789_78942


namespace NUMINAMATH_CALUDE_rectangular_solid_width_l789_78931

theorem rectangular_solid_width (length depth surface_area : ℝ) : 
  length = 10 →
  depth = 6 →
  surface_area = 408 →
  surface_area = 2 * length * width + 2 * length * depth + 2 * width * depth →
  width = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_width_l789_78931


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l789_78912

/-- A right prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightPrism (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C A₁ B₁ C₁ : V)
  (is_right_prism : sorry)

/-- Points P and P₁ on edges BB₁ and CC₁ respectively -/
structure PrismWithPoints (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] extends RightPrism V :=
  (P P₁ : V)
  (P_on_BB₁ : sorry)
  (P₁_on_CC₁ : sorry)
  (ratio_condition : sorry)

/-- The dihedral angle between two planes -/
def dihedral_angle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (plane1 plane2 : Set V) : ℝ := sorry

/-- Theorem stating the properties of the tetrahedron AA₁PP₁ -/
theorem tetrahedron_properties 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (prism : PrismWithPoints V) :
  let A := prism.A
  let A₁ := prism.A₁
  let P := prism.P
  let P₁ := prism.P₁
  (dihedral_angle V {A, P₁, P} {A, A₁, P} = π / 2) ∧ 
  (dihedral_angle V {A₁, P, P₁} {A₁, A, P} = π / 2) ∧
  (dihedral_angle V {A, P, P₁} {A, A₁, P} + 
   dihedral_angle V {A, P, P₁} {A₁, P, P₁} + 
   dihedral_angle V {A₁, P₁, P} {A, A₁, P₁} = π) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l789_78912


namespace NUMINAMATH_CALUDE_not_perfect_square_l789_78936

theorem not_perfect_square (n : ℕ) (d : ℕ) (h : d > 0) (h' : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), x^2 = n^2 + d := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l789_78936


namespace NUMINAMATH_CALUDE_emily_walks_farther_l789_78983

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_detours : List ℕ := [15, 20, 10, 0, 5]
def emily_detours : List ℕ := [10, 25, 10, 15, 10]

def total_distance (base : ℕ) (detours : List ℕ) : ℕ :=
  (base * 10 + detours.sum) * 2

theorem emily_walks_farther :
  total_distance emily_base_distance emily_detours -
  total_distance troy_base_distance troy_detours = 270 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l789_78983


namespace NUMINAMATH_CALUDE_base_of_equation_l789_78919

theorem base_of_equation (x y : ℕ) (base : ℝ) : 
  x = 9 → 
  x - y = 9 → 
  (base ^ x) * (4 ^ y) = 19683 → 
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_base_of_equation_l789_78919


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_read_l789_78908

/-- Represents the 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : Nat
  total_movies : Nat
  books_read : Nat
  movies_watched : Nat

/-- Theorem: In the 'crazy silly school' series, if all available books and movies are consumed,
    and the number of movies watched is 2 more than the number of books read,
    then the number of books read is 8. -/
theorem crazy_silly_school_books_read
  (css : CrazySillySchool)
  (h1 : css.total_books = 8)
  (h2 : css.total_movies = 10)
  (h3 : css.movies_watched = css.books_read + 2)
  (h4 : css.books_read = css.total_books)
  (h5 : css.movies_watched = css.total_movies) :
  css.books_read = 8 := by
  sorry

#check crazy_silly_school_books_read

end NUMINAMATH_CALUDE_crazy_silly_school_books_read_l789_78908


namespace NUMINAMATH_CALUDE_problem_statement_l789_78975

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 10) :
  x^2*y + x*y^2 = 500/121 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l789_78975


namespace NUMINAMATH_CALUDE_difference_value_l789_78900

theorem difference_value (n : ℚ) : n = 45 → (n / 3 - 5 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_value_l789_78900


namespace NUMINAMATH_CALUDE_apple_banana_ratio_l789_78993

theorem apple_banana_ratio (n : ℕ) : 
  (3 * n + 2 * n = 72) → False :=
by
  sorry

end NUMINAMATH_CALUDE_apple_banana_ratio_l789_78993


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_div_five_l789_78980

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ m ∣ n

theorem smallest_two_digit_prime_with_reversed_composite_div_five :
  ∃ (n : ℕ),
    n ≥ 20 ∧ n < 30 ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n) % 5 = 0 ∧
    ∀ (m : ℕ), m ≥ 20 ∧ m < n →
      ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ (reverse_digits m) % 5 = 0) :=
  by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_div_five_l789_78980


namespace NUMINAMATH_CALUDE_population_change_l789_78960

theorem population_change (P : ℝ) : 
  (P * 1.05 * 0.95 = 9975) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l789_78960


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l789_78925

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin of the coordinate system -/
def origin : Point := ⟨0, 0⟩

/-- The coordinate system -/
structure CoordinateSystem where
  origin : Point

/-- A point's coordinates with respect to a coordinate system -/
def coordinates (p : Point) (cs : CoordinateSystem) : Point :=
  ⟨p.x - cs.origin.x, p.y - cs.origin.y⟩

theorem coordinates_wrt_origin (A : Point) (cs : CoordinateSystem) :
  A.x = -1 ∧ A.y = 2 → coordinates A cs = A :=
by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l789_78925


namespace NUMINAMATH_CALUDE_purple_ribbons_l789_78921

theorem purple_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 6 →
  black = 40 →
  yellow + purple + orange + black = total →
  purple = 53 := by
sorry

end NUMINAMATH_CALUDE_purple_ribbons_l789_78921


namespace NUMINAMATH_CALUDE_sisters_portions_l789_78906

/-- Represents the types of granola bars --/
inductive BarType
  | ChocolateChip
  | OatAndHoney
  | PeanutButter

/-- Represents the number of bars of each type --/
structure BarCounts where
  chocolateChip : ℕ
  oatAndHoney : ℕ
  peanutButter : ℕ

/-- Represents the initial distribution of bars --/
def initialDistribution : BarCounts :=
  { chocolateChip := 8, oatAndHoney := 6, peanutButter := 6 }

/-- Represents the bars set aside daily --/
def dailySetAside : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 2, peanutButter := 2 }

/-- Represents the bars left after setting aside --/
def afterSetAside : BarCounts :=
  { chocolateChip := 5, oatAndHoney := 4, peanutButter := 4 }

/-- Represents the bars traded --/
def traded : BarCounts :=
  { chocolateChip := 2, oatAndHoney := 4, peanutButter := 0 }

/-- Represents the bars left after trading --/
def afterTrading : BarCounts :=
  { chocolateChip := 3, oatAndHoney := 0, peanutButter := 3 }

/-- Represents the whole bars given to each sister --/
def wholeBarsGiven (type : BarType) : ℕ := 2

/-- Theorem: Each sister receives 2.5 portions of their preferred granola bar type --/
theorem sisters_portions (type : BarType) : 
  (wholeBarsGiven type : ℚ) + (1 : ℚ) / 2 = (5 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sisters_portions_l789_78906


namespace NUMINAMATH_CALUDE_discount_percentage_decrease_l789_78915

theorem discount_percentage_decrease (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_decrease_l789_78915


namespace NUMINAMATH_CALUDE_add_seconds_theorem_l789_78994

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (8:00:00 a.m.) -/
def initialTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 102930

/-- The expected final time (12:35:30) -/
def expectedFinalTime : Time :=
  { hours := 12, minutes := 35, seconds := 30 }

theorem add_seconds_theorem :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_add_seconds_theorem_l789_78994


namespace NUMINAMATH_CALUDE_max_profit_at_45_l789_78945

/-- Represents the daily profit function for selling children's toys -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1000 * x - 21000

/-- Represents the cost price of each toy -/
def cost_price : ℝ := 30

/-- Represents the maximum allowed profit margin -/
def max_profit_margin : ℝ := 0.5

/-- Represents the minimum selling price -/
def min_selling_price : ℝ := 35

/-- Represents the maximum selling price based on the profit margin constraint -/
def max_selling_price : ℝ := cost_price * (1 + max_profit_margin)

theorem max_profit_at_45 :
  ∃ (max_profit : ℝ),
    (∀ x : ℝ, min_selling_price ≤ x ∧ x ≤ max_selling_price →
      profit_function x ≤ profit_function max_selling_price) ∧
    profit_function max_selling_price = max_profit ∧
    max_profit = 3750 := by
  sorry

#eval profit_function max_selling_price

end NUMINAMATH_CALUDE_max_profit_at_45_l789_78945


namespace NUMINAMATH_CALUDE_max_value_fraction_l789_78979

theorem max_value_fraction (x y : ℝ) (hx : -4 ≤ x ∧ x ≤ -2) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ a b, -4 ≤ a ∧ a ≤ -2 ∧ 3 ≤ b ∧ b ≤ 5 → (a + b) / a ≤ (x + y) / x) →
  (x + y) / x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l789_78979


namespace NUMINAMATH_CALUDE_profit_revenue_relationship_l789_78947

/-- Represents the financial data of a company over two years -/
structure CompanyFinancials where
  prevRevenue : ℝ
  prevProfit : ℝ
  currRevenue : ℝ
  currProfit : ℝ

/-- The theorem stating the relationship between profits and revenues -/
theorem profit_revenue_relationship (c : CompanyFinancials)
  (h1 : c.currRevenue = 0.8 * c.prevRevenue)
  (h2 : c.currProfit = 0.2 * c.currRevenue)
  (h3 : c.currProfit = 1.6000000000000003 * c.prevProfit) :
  c.prevProfit / c.prevRevenue = 0.1 := by
  sorry

#check profit_revenue_relationship

end NUMINAMATH_CALUDE_profit_revenue_relationship_l789_78947


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l789_78953

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x + 8

-- Theorem statement
theorem extreme_value_and_range :
  (f 2 = -8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f x ∧ f x ≤ 24) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = -8) ∧
  (∃ x ∈ Set.Icc (-3) 3, f x = 24) :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l789_78953


namespace NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l789_78956

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the first derivative of a sequence
def FirstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

-- Define the k-th derivative of a sequence
def KthDerivative (a : Sequence) : ℕ → Sequence
  | 0 => a
  | k + 1 => FirstDerivative (KthDerivative a k)

-- Define a good sequence
def IsGoodSequence (a : Sequence) : Prop :=
  ∀ k n, KthDerivative a k n > 0

-- Theorem statement
theorem product_of_good_sequences_is_good
  (a b : Sequence)
  (ha : IsGoodSequence a)
  (hb : IsGoodSequence b) :
  IsGoodSequence (λ n => a n * b n) :=
by sorry

end NUMINAMATH_CALUDE_product_of_good_sequences_is_good_l789_78956


namespace NUMINAMATH_CALUDE_arrangement_count_is_48_l789_78952

/-- The number of ways to arrange 5 different items into two rows -/
def arrangement_count : ℕ := 48

/-- The total number of items -/
def total_items : ℕ := 5

/-- The minimum number of items in each row -/
def min_items_per_row : ℕ := 2

/-- The number of items that must be in the front row -/
def fixed_front_items : ℕ := 2

/-- Theorem stating that the number of arrangements is 48 -/
theorem arrangement_count_is_48 :
  arrangement_count = 48 ∧
  total_items = 5 ∧
  min_items_per_row = 2 ∧
  fixed_front_items = 2 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_is_48_l789_78952


namespace NUMINAMATH_CALUDE_smallest_c_is_correct_l789_78937

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- The smallest positive real value c such that d(n) ≤ c * √n for all positive integers n -/
noncomputable def smallest_c : ℝ := Real.sqrt 3

theorem smallest_c_is_correct :
  (∀ n : ℕ+, (num_divisors n : ℝ) ≤ smallest_c * Real.sqrt n) ∧
  (∀ c : ℝ, 0 < c → c < smallest_c →
    ∃ n : ℕ+, (num_divisors n : ℝ) > c * Real.sqrt n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_is_correct_l789_78937


namespace NUMINAMATH_CALUDE_greyson_fuel_expense_l789_78904

/-- Calculates the total fuel expense for a week given the number of refills and cost per refill -/
def total_fuel_expense (num_refills : ℕ) (cost_per_refill : ℕ) : ℕ :=
  num_refills * cost_per_refill

/-- Proves that Greyson's total fuel expense for the week is $40 -/
theorem greyson_fuel_expense :
  total_fuel_expense 4 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_greyson_fuel_expense_l789_78904


namespace NUMINAMATH_CALUDE_mike_is_18_l789_78949

-- Define Mike's age and his uncle's age
def mike_age : ℕ := sorry
def uncle_age : ℕ := sorry

-- Define the conditions
axiom age_difference : mike_age = uncle_age - 18
axiom sum_of_ages : mike_age + uncle_age = 54

-- Theorem to prove
theorem mike_is_18 : mike_age = 18 := by sorry

end NUMINAMATH_CALUDE_mike_is_18_l789_78949


namespace NUMINAMATH_CALUDE_alices_lawn_area_l789_78988

/-- Represents a rectangular lawn with fence posts -/
structure Lawn :=
  (total_posts : ℕ)
  (post_spacing : ℕ)
  (long_side_posts : ℕ)
  (short_side_posts : ℕ)

/-- Calculates the area of the lawn given its specifications -/
def lawn_area (l : Lawn) : ℕ :=
  (l.post_spacing * (l.short_side_posts - 1)) * (l.post_spacing * (l.long_side_posts - 1))

/-- Theorem stating the area of Alice's lawn -/
theorem alices_lawn_area :
  ∀ (l : Lawn),
  l.total_posts = 24 →
  l.post_spacing = 5 →
  l.long_side_posts = 3 * l.short_side_posts →
  2 * (l.long_side_posts + l.short_side_posts - 2) = l.total_posts →
  lawn_area l = 825 := by
  sorry

#check alices_lawn_area

end NUMINAMATH_CALUDE_alices_lawn_area_l789_78988


namespace NUMINAMATH_CALUDE_percentage_problem_l789_78922

theorem percentage_problem (number : ℝ) (P : ℝ) : number = 15 → P = 20 / 100 * number + 47 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l789_78922


namespace NUMINAMATH_CALUDE_inequality_proof_l789_78970

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) 
  (h4 : x > y) (h5 : y > z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((c*z + a*x) * (c*x + a*z)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l789_78970


namespace NUMINAMATH_CALUDE_julie_landscaping_rate_l789_78934

/-- Julie's landscaping business problem -/
theorem julie_landscaping_rate :
  ∀ (mowing_rate : ℝ),
  let weeding_rate : ℝ := 8
  let mowing_hours : ℝ := 25
  let weeding_hours : ℝ := 3
  let total_earnings : ℝ := 248
  (2 * (mowing_rate * mowing_hours + weeding_rate * weeding_hours) = total_earnings) →
  mowing_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_julie_landscaping_rate_l789_78934


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l789_78901

theorem quadratic_root_implies_coefficient (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 - Complex.I) ^ 2 + a * (1 - Complex.I) + 2 = 0 →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l789_78901


namespace NUMINAMATH_CALUDE_alex_peeled_22_potatoes_l789_78981

/-- The number of potatoes Alex peeled -/
def alexPotatoes (totalPotatoes : ℕ) (homerRate alextRate : ℕ) (alexJoinTime : ℕ) : ℕ :=
  let homerPotatoes := homerRate * alexJoinTime
  let remainingPotatoes := totalPotatoes - homerPotatoes
  let combinedRate := homerRate + alextRate
  let remainingTime := remainingPotatoes / combinedRate
  alextRate * remainingTime

theorem alex_peeled_22_potatoes :
  alexPotatoes 60 4 6 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_alex_peeled_22_potatoes_l789_78981


namespace NUMINAMATH_CALUDE_radius_vector_coordinates_l789_78984

/-- Given a point M with coordinates (-2, 5, 0) in a rectangular coordinate system,
    prove that the coordinates of its radius vector OM are equal to (-2, 5, 0). -/
theorem radius_vector_coordinates (M : ℝ × ℝ × ℝ) (h : M = (-2, 5, 0)) :
  M = (-2, 5, 0) := by sorry

end NUMINAMATH_CALUDE_radius_vector_coordinates_l789_78984


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l789_78987

/-- Given two right circular cylinders with identical volumes, where the radius of the second cylinder
    is 20% more than the radius of the first, prove that the height of the first cylinder
    is 44% more than the height of the second cylinder. -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) : 
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  (π * r₁^2 * h₁ = π * r₂^2 * h₂) →  -- Volumes are equal
  (r₂ = 1.2 * r₁) →                  -- Second radius is 20% more than the first
  (h₁ = 1.44 * h₂) :=                -- First height is 44% more than the second
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l789_78987


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_sin_60_l789_78998

theorem reciprocal_of_negative_sin_60 :
  ((-Real.sin (π / 3))⁻¹) = -(2 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_sin_60_l789_78998


namespace NUMINAMATH_CALUDE_moon_earth_distance_in_scientific_notation_l789_78916

/-- The average distance from the moon to the earth in meters -/
def moon_earth_distance : ℝ := 384000000

/-- The scientific notation representation of the moon-earth distance -/
def moon_earth_distance_scientific : ℝ := 3.84 * (10 ^ 8)

theorem moon_earth_distance_in_scientific_notation :
  moon_earth_distance = moon_earth_distance_scientific := by
  sorry

end NUMINAMATH_CALUDE_moon_earth_distance_in_scientific_notation_l789_78916


namespace NUMINAMATH_CALUDE_fraction_subtraction_result_l789_78967

theorem fraction_subtraction_result : (18 : ℚ) / 42 - 3 / 8 - 1 / 12 = -5 / 168 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_result_l789_78967
