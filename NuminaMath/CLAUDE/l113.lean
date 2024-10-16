import Mathlib

namespace NUMINAMATH_CALUDE_janet_total_earnings_l113_11302

/-- Calculates Janet's total earnings from exterminator work and sculpture sales -/
def janet_earnings (exterminator_rate : ℝ) (sculpture_rate : ℝ) (hours_worked : ℝ) 
                   (sculpture1_weight : ℝ) (sculpture2_weight : ℝ) : ℝ :=
  exterminator_rate * hours_worked + 
  sculpture_rate * (sculpture1_weight + sculpture2_weight)

/-- Proves that Janet's total earnings are $1640 given the specified conditions -/
theorem janet_total_earnings :
  janet_earnings 70 20 20 5 7 = 1640 := by
  sorry

end NUMINAMATH_CALUDE_janet_total_earnings_l113_11302


namespace NUMINAMATH_CALUDE_selling_price_with_gain_l113_11303

/-- Given an article with a cost price where a $10 gain represents a 10% gain, 
    prove that the selling price is $110. -/
theorem selling_price_with_gain (cost_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : 10 / cost_price = 0.1) : 
  cost_price + 10 = 110 := by
  sorry

#check selling_price_with_gain

end NUMINAMATH_CALUDE_selling_price_with_gain_l113_11303


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l113_11352

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 5

def ring_arrangements (n k : ℕ) : ℕ := (n.choose k) * k.factorial

def finger_distributions (m n : ℕ) : ℕ := (m + n - 1).choose n

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange * 
  finger_distributions (rings_to_arrange + number_of_fingers - 1) number_of_fingers = 31752000 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l113_11352


namespace NUMINAMATH_CALUDE_fraction_pure_fuji_l113_11315

-- Define the total number of trees
def total_trees : ℕ := 180

-- Define the number of pure Fuji trees
def pure_fuji : ℕ := 135

-- Define the number of pure Gala trees
def pure_gala : ℕ := 27

-- Define the number of cross-pollinated trees
def cross_pollinated : ℕ := 18

-- Define the cross-pollination rate
def cross_pollination_rate : ℚ := 1/10

-- Theorem stating the fraction of pure Fuji trees
theorem fraction_pure_fuji :
  (pure_fuji : ℚ) / total_trees = 3/4 :=
by
  sorry

-- Conditions from the problem
axiom condition1 : pure_fuji + cross_pollinated = 153
axiom condition2 : (cross_pollinated : ℚ) / total_trees = cross_pollination_rate
axiom condition3 : total_trees = pure_fuji + pure_gala + cross_pollinated

end NUMINAMATH_CALUDE_fraction_pure_fuji_l113_11315


namespace NUMINAMATH_CALUDE_stone_123_is_12_l113_11316

/-- Represents the counting pattern on a circle of stones -/
def stone_count (n : ℕ) : ℕ := 
  let cycle := 28
  n % cycle

/-- The original position of a stone given its count number -/
def original_position (count : ℕ) : ℕ :=
  if count ≤ 15 then count
  else 16 - (count - 15)

theorem stone_123_is_12 : original_position (stone_count 123) = 12 := by sorry

end NUMINAMATH_CALUDE_stone_123_is_12_l113_11316


namespace NUMINAMATH_CALUDE_second_polygon_sides_l113_11391

theorem second_polygon_sides
  (perimeter_equal : ℝ → ℝ → ℕ → ℕ → Prop)
  (first_sides : ℕ)
  (side_length_ratio : ℝ)
  (second_sides : ℕ) :
  perimeter_equal (3 * side_length_ratio) side_length_ratio first_sides second_sides →
  first_sides = 50 →
  second_sides = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l113_11391


namespace NUMINAMATH_CALUDE_factor_expression_l113_11331

theorem factor_expression (b : ℝ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l113_11331


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l113_11347

/-- Triangle ABC with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : (ℝ × ℝ) := (4, 0)
  B : (ℝ × ℝ) := (6, 7)
  C : (ℝ × ℝ) := (0, 3)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.altitudeFromB (t : Triangle) : LineEquation :=
  { a := 3, b := 2, c := -12 }

def Triangle.medianFromB (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (t.altitudeFromB = { a := 3, b := 2, c := -12 }) ∧
  (t.medianFromB = { a := 5, b := 1, c := -20 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l113_11347


namespace NUMINAMATH_CALUDE_find_true_product_l113_11398

theorem find_true_product (a b : ℕ) : 
  b = 2 * a →
  136 * (10 * b + a) = 136 * (10 * a + b) + 1224 →
  136 * (10 * a + b) = 1632 := by
sorry

end NUMINAMATH_CALUDE_find_true_product_l113_11398


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l113_11369

theorem triangle_abc_is_right_triangle (AB AC BC : ℝ) 
  (h1 : AB = 1) (h2 : AC = 2) (h3 : BC = Real.sqrt 5) : 
  AB ^ 2 + AC ^ 2 = BC ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l113_11369


namespace NUMINAMATH_CALUDE_gcd_and_bezout_identity_l113_11368

theorem gcd_and_bezout_identity :
  ∃ (d u v : ℤ), Int.gcd 663 182 = d ∧ d = 663 * u + 182 * v ∧ d = 13 :=
by sorry

end NUMINAMATH_CALUDE_gcd_and_bezout_identity_l113_11368


namespace NUMINAMATH_CALUDE_equation_unique_solution_l113_11325

/-- The function representing the left-hand side of the equation -/
def f (y : ℝ) : ℝ := (30 * y + (30 * y + 25) ^ (1/3)) ^ (1/3)

/-- The theorem stating that the equation has a unique solution -/
theorem equation_unique_solution :
  ∃! y : ℝ, f y = 15 ∧ y = 335/3 := by sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l113_11325


namespace NUMINAMATH_CALUDE_parabolas_intersection_l113_11359

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 4

-- Define a function to check if a point is on both parabolas
def is_intersection (x y : ℝ) : Prop :=
  parabola1 x = y ∧ parabola2 x = y

-- Theorem statement
theorem parabolas_intersection :
  (is_intersection (-3) 25) ∧ 
  (is_intersection 1 1) ∧
  (∀ x y : ℝ, is_intersection x y → (x = -3 ∧ y = 25) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l113_11359


namespace NUMINAMATH_CALUDE_fence_construction_l113_11318

/-- A fence construction problem -/
theorem fence_construction (panels : ℕ) (sheets_per_panel : ℕ) (beams_per_panel : ℕ) 
  (rods_per_sheet : ℕ) (total_rods : ℕ) :
  panels = 10 →
  sheets_per_panel = 3 →
  beams_per_panel = 2 →
  rods_per_sheet = 10 →
  total_rods = 380 →
  (total_rods - panels * sheets_per_panel * rods_per_sheet) / (panels * beams_per_panel) = 4 :=
by sorry

end NUMINAMATH_CALUDE_fence_construction_l113_11318


namespace NUMINAMATH_CALUDE_max_value_theorem_l113_11387

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ -1 + (3*Real.sqrt 2)/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l113_11387


namespace NUMINAMATH_CALUDE_decimal_to_binary_75_l113_11350

theorem decimal_to_binary_75 : 
  (75 : ℕ) = 1 * 2^6 + 0 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_75_l113_11350


namespace NUMINAMATH_CALUDE_expression_value_l113_11305

theorem expression_value (x : ℤ) (h : x = -3) : x^2 - 4*(x - 5) = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l113_11305


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l113_11355

/-- The sum of the digits of 10^91 + 100 is 2 -/
theorem sum_of_digits_of_large_number : ∃ (n : ℕ), n = 10^91 + 100 ∧ (n.digits 10).sum = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l113_11355


namespace NUMINAMATH_CALUDE_equal_angles_not_always_vertical_l113_11394

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the equality of angles
def angle_equal (a b : Angle) : Prop := a = b

-- Theorem stating that equal angles are not necessarily vertical angles
theorem equal_angles_not_always_vertical :
  ∃ (a b : Angle), angle_equal a b ∧ ¬(are_vertical_angles a b) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_not_always_vertical_l113_11394


namespace NUMINAMATH_CALUDE_gcd_72_120_l113_11304

theorem gcd_72_120 : Nat.gcd 72 120 = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_72_120_l113_11304


namespace NUMINAMATH_CALUDE_combined_experience_is_68_l113_11380

/-- Calculates the combined experience of James, John, and Mike -/
def combinedExperience (james_current : ℕ) (years_ago : ℕ) (john_multiplier : ℕ) (john_when_mike_started : ℕ) : ℕ :=
  let james_past := james_current - years_ago
  let john_past := john_multiplier * james_past
  let john_current := john_past + years_ago
  let mike_experience := john_current - john_when_mike_started
  james_current + john_current + mike_experience

/-- The combined experience of James, John, and Mike is 68 years -/
theorem combined_experience_is_68 :
  combinedExperience 20 8 2 16 = 68 := by
  sorry

end NUMINAMATH_CALUDE_combined_experience_is_68_l113_11380


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l113_11365

/-- Given that the solution set of ax² + bx + c > 0 is {x | 1 < x < 2},
    prove that the solution set of cx² + bx + a < 0 is {x | x < 1/2 or x > 1} -/
theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < 2) :
  ∀ x : ℝ, c*x^2 + b*x + a < 0 ↔ x < 1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l113_11365


namespace NUMINAMATH_CALUDE_total_weight_of_goods_l113_11397

theorem total_weight_of_goods (x : ℝ) 
  (h1 : (x - 10) / 7 = (x + 5) / 8) : x = 115 := by
  sorry

#check total_weight_of_goods

end NUMINAMATH_CALUDE_total_weight_of_goods_l113_11397


namespace NUMINAMATH_CALUDE_particle_position_at_5pm_l113_11340

-- Define the particle's position as a function of time
def particle_position (t : ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem particle_position_at_5pm :
  -- Given conditions
  (particle_position 7 = (1, 2)) →
  (particle_position 9 = (3, -2)) →
  -- Constant speed along a straight line (slope remains constant)
  (∀ t₁ t₂ t₃ t₄ : ℝ, t₁ ≠ t₂ ∧ t₃ ≠ t₄ →
    (particle_position t₂).1 - (particle_position t₁).1 ≠ 0 →
    ((particle_position t₂).2 - (particle_position t₁).2) / ((particle_position t₂).1 - (particle_position t₁).1) =
    ((particle_position t₄).2 - (particle_position t₃).2) / ((particle_position t₄).1 - (particle_position t₃).1)) →
  -- Conclusion
  particle_position 17 = (11, -18) :=
by sorry

end NUMINAMATH_CALUDE_particle_position_at_5pm_l113_11340


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l113_11366

theorem polynomial_divisibility (a : ℤ) : 
  (∃ q : Polynomial ℤ, X^6 - 33•X + 20 = (X^2 - X + a•1) * q) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l113_11366


namespace NUMINAMATH_CALUDE_price_increase_theorem_l113_11310

theorem price_increase_theorem (original_price : ℝ) (original_price_pos : original_price > 0) :
  let price_a := original_price * 1.2 * 1.15
  let price_b := original_price * 1.3 * 0.9
  let price_c := original_price * 1.25 * 1.2
  let total_increase := (price_a + price_b + price_c) - 3 * original_price
  let percent_increase := total_increase / (3 * original_price) * 100
  percent_increase = 35 := by
sorry

end NUMINAMATH_CALUDE_price_increase_theorem_l113_11310


namespace NUMINAMATH_CALUDE_knicks_win_probability_l113_11313

/-- The probability of winning a single game for the Heat -/
def p : ℚ := 3/5

/-- The probability of winning a single game for the Knicks -/
def q : ℚ := 1 - p

/-- The number of games needed to win the tournament -/
def games_to_win : ℕ := 3

/-- The total number of games in the tournament -/
def total_games : ℕ := 5

/-- The probability of the Knicks winning the tournament in exactly 5 games -/
def knicks_win_prob : ℚ :=
  (Nat.choose 4 2 : ℚ) * q^2 * p^2 * q

theorem knicks_win_probability :
  knicks_win_prob = 432/3125 :=
sorry

end NUMINAMATH_CALUDE_knicks_win_probability_l113_11313


namespace NUMINAMATH_CALUDE_bricklayer_solution_l113_11300

/-- Represents the problem of two bricklayers building a wall -/
structure BricklayerProblem where
  -- Total number of bricks in the wall
  total_bricks : ℕ
  -- Time taken by the first bricklayer alone (in hours)
  time_first : ℕ
  -- Time taken by the second bricklayer alone (in hours)
  time_second : ℕ
  -- Reduction in combined output (in bricks per hour)
  output_reduction : ℕ
  -- Time taken when working together (in hours)
  time_together : ℕ

/-- The theorem stating the solution to the bricklayer problem -/
theorem bricklayer_solution (problem : BricklayerProblem) :
  problem.time_first = 8 →
  problem.time_second = 12 →
  problem.output_reduction = 15 →
  problem.time_together = 6 →
  problem.total_bricks = 360 := by
  sorry

#check bricklayer_solution

end NUMINAMATH_CALUDE_bricklayer_solution_l113_11300


namespace NUMINAMATH_CALUDE_lucy_calculation_mistake_l113_11327

theorem lucy_calculation_mistake (a b c : ℝ) 
  (h1 : a / (b * c) = 4)
  (h2 : (a / b) / c = 12)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a / b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_lucy_calculation_mistake_l113_11327


namespace NUMINAMATH_CALUDE_simplify_star_expression_l113_11345

/-- Custom binary operation ※ for rational numbers -/
def star (a b : ℚ) : ℚ := 2 * a - b

/-- Theorem stating the equivalence of the expression and its simplified form -/
theorem simplify_star_expression (x y : ℚ) : 
  star (star (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
sorry

end NUMINAMATH_CALUDE_simplify_star_expression_l113_11345


namespace NUMINAMATH_CALUDE_brick_width_calculation_l113_11376

theorem brick_width_calculation (courtyard_length courtyard_width : ℝ)
                                (brick_length : ℝ)
                                (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.1 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by
  sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l113_11376


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l113_11386

theorem gcd_lcm_sum : Nat.gcd 54 24 + Nat.lcm 48 18 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l113_11386


namespace NUMINAMATH_CALUDE_pizza_meat_distribution_l113_11307

/-- Pizza meat distribution problem -/
theorem pizza_meat_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) 
  (h1 : pepperoni = 30)
  (h2 : ham = 2 * pepperoni)
  (h3 : sausage = pepperoni + 12)
  (h4 : slices = 6)
  : (pepperoni + ham + sausage) / slices = 22 := by
  sorry

end NUMINAMATH_CALUDE_pizza_meat_distribution_l113_11307


namespace NUMINAMATH_CALUDE_smallest_stable_triangle_side_l113_11356

/-- A stable triangle is a scalene triangle with positive integer side lengths that are multiples of 5, 80, and 112 respectively. -/
def StableTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- scalene
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive
  ∃ (x y z : ℕ), a = 5 * x ∧ b = 80 * y ∧ c = 112 * z  -- multiples of 5, 80, 112

/-- The smallest possible side length in any stable triangle is 20. -/
theorem smallest_stable_triangle_side : 
  (∃ (a b c : ℕ), StableTriangle a b c) → 
  (∀ (a b c : ℕ), StableTriangle a b c → min a (min b c) ≥ 20) ∧
  (∃ (a b c : ℕ), StableTriangle a b c ∧ min a (min b c) = 20) :=
sorry

end NUMINAMATH_CALUDE_smallest_stable_triangle_side_l113_11356


namespace NUMINAMATH_CALUDE_sum_of_smaller_angles_is_180_l113_11319

/-- A convex pentagon with all diagonals drawn. -/
structure ConvexPentagonWithDiagonals where
  -- We don't need to define the specific properties here, just the structure

/-- The sum of the smaller angles formed by intersecting diagonals in a convex pentagon. -/
def sumOfSmallerAngles (p : ConvexPentagonWithDiagonals) : ℝ := sorry

/-- Theorem: The sum of the smaller angles formed by intersecting diagonals in a convex pentagon is always 180°. -/
theorem sum_of_smaller_angles_is_180 (p : ConvexPentagonWithDiagonals) :
  sumOfSmallerAngles p = 180 := by sorry

end NUMINAMATH_CALUDE_sum_of_smaller_angles_is_180_l113_11319


namespace NUMINAMATH_CALUDE_area_above_line_is_two_thirds_l113_11308

/-- A square in a 2D plane -/
structure Square where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- A line in a 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square -/
def square_area (s : Square) : ℝ :=
  let (x1, y1) := s.bottom_left
  let (x2, y2) := s.top_right
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the region above a line in a square -/
noncomputable def area_above_line (s : Square) (l : Line) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area above the specified line is 2/3 of the square's area -/
theorem area_above_line_is_two_thirds (s : Square) (l : Line) : 
  s.bottom_left = (2, 1) ∧ 
  s.top_right = (5, 4) ∧ 
  l.point1 = (2, 1) ∧ 
  l.point2 = (5, 3) → 
  area_above_line s l = (2/3) * square_area s := by
  sorry


end NUMINAMATH_CALUDE_area_above_line_is_two_thirds_l113_11308


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l113_11309

theorem square_sum_equals_two (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l113_11309


namespace NUMINAMATH_CALUDE_collinear_points_sum_l113_11323

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), p2 = (t1 • p1 + (1 - t1) • p3) ∧ p3 = (t2 • p1 + (1 - t2) • p2)

/-- If the points (1, a, b), (a, 2, b), and (a, b, 3) are collinear in 3-space, then a + b = 4. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (1, a, b) (a, 2, b) (a, b, 3) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l113_11323


namespace NUMINAMATH_CALUDE_four_inch_cube_multi_painted_l113_11353

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : ℕ
  painted_faces : ℕ
  hsl : side_length = n
  hpf : painted_faces = 6

/-- Represents a smaller cube cut from a larger cube -/
structure SmallCube where
  painted_faces : ℕ

/-- Function to count cubes with at least two painted faces -/
def count_multi_painted_cubes (n : ℕ) : ℕ :=
  8 + 12 * (n - 2)

/-- Theorem statement -/
theorem four_inch_cube_multi_painted (c : Cube 4) :
  count_multi_painted_cubes c.side_length = 40 :=
sorry

end NUMINAMATH_CALUDE_four_inch_cube_multi_painted_l113_11353


namespace NUMINAMATH_CALUDE_bakers_ovens_l113_11344

/-- Baker's bread production problem -/
theorem bakers_ovens :
  let loaves_per_hour_per_oven : ℕ := 5
  let weekday_hours : ℕ := 5
  let weekday_count : ℕ := 5
  let weekend_hours : ℕ := 2
  let weekend_count : ℕ := 2
  let weeks : ℕ := 3
  let total_loaves : ℕ := 1740
  
  let weekly_hours := weekday_hours * weekday_count + weekend_hours * weekend_count
  let weekly_loaves_per_oven := weekly_hours * loaves_per_hour_per_oven
  let total_loaves_per_oven := weekly_loaves_per_oven * weeks
  
  total_loaves / total_loaves_per_oven = 4 := by
  sorry


end NUMINAMATH_CALUDE_bakers_ovens_l113_11344


namespace NUMINAMATH_CALUDE_square_real_implies_a_zero_l113_11326

theorem square_real_implies_a_zero (a : ℝ) : 
  (Complex.I * a + 2) ^ 2 ∈ Set.range Complex.ofReal → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_real_implies_a_zero_l113_11326


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l113_11360

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  1 / (x - 1) - 2 / (x^2 - 1) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l113_11360


namespace NUMINAMATH_CALUDE_function_composition_l113_11390

theorem function_composition (f : ℝ → ℝ) :
  (∀ x, f (x - 2) = 3 * x - 5) → (∀ x, f x = 3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l113_11390


namespace NUMINAMATH_CALUDE_total_age_is_23_l113_11389

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : ryanne_age = hezekiah_age + 7)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) :
  hezekiah_age + ryanne_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end NUMINAMATH_CALUDE_total_age_is_23_l113_11389


namespace NUMINAMATH_CALUDE_car_speed_problem_l113_11321

theorem car_speed_problem (x : ℝ) :
  x > 0 →
  (x + 60) / 2 = 75 →
  x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l113_11321


namespace NUMINAMATH_CALUDE_constraint_implies_sum_equals_nine_l113_11312

open Real

/-- The maximum value of xy + xz + yz given the constraint -/
noncomputable def N : ℝ := sorry

/-- The minimum value of xy + xz + yz given the constraint -/
noncomputable def n : ℝ := sorry

/-- Theorem stating that N + 8n = 9 given the constraint -/
theorem constraint_implies_sum_equals_nine :
  ∀ x y z : ℝ, 3 * (x + y + z) = x^2 + y^2 + z^2 → N + 8 * n = 9 := by
  sorry

end NUMINAMATH_CALUDE_constraint_implies_sum_equals_nine_l113_11312


namespace NUMINAMATH_CALUDE_two_numbers_between_4_and_16_l113_11373

theorem two_numbers_between_4_and_16 :
  ∃ (a b : ℝ), 
    4 < a ∧ a < b ∧ b < 16 ∧
    (b - a = a - 4) ∧
    (b * b = a * 16) ∧
    a + b = 20 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_between_4_and_16_l113_11373


namespace NUMINAMATH_CALUDE_triangle_side_length_l113_11333

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : B = π / 6) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l113_11333


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l113_11375

theorem solution_set_reciprocal_inequality (x : ℝ) :
  1 / x ≤ 1 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l113_11375


namespace NUMINAMATH_CALUDE_number_equation_solution_l113_11367

theorem number_equation_solution : ∃ x : ℝ, 46 + 3 * x = 109 ∧ x = 21 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l113_11367


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l113_11335

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x0 y0, x0 + y0 = 36 ∧ x0 = 3 * y0 ∧ x0 * y0 = k) :
  x = -6 → y = -40.5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l113_11335


namespace NUMINAMATH_CALUDE_travel_agency_choice_l113_11384

-- Define the cost functions for each travel agency
def costA (x : ℝ) : ℝ := 2000 * x * 0.75
def costB (x : ℝ) : ℝ := 2000 * (x - 1) * 0.8

-- Define the theorem
theorem travel_agency_choice (x : ℝ) (h1 : 10 ≤ x) (h2 : x ≤ 25) :
  (10 ≤ x ∧ x < 16 → costB x < costA x) ∧
  (x = 16 → costA x = costB x) ∧
  (16 < x ∧ x ≤ 25 → costA x < costB x) :=
sorry

end NUMINAMATH_CALUDE_travel_agency_choice_l113_11384


namespace NUMINAMATH_CALUDE_w_squared_value_l113_11392

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 9)*(w + 6)) :
  w^2 = 57.5 - 0.5 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l113_11392


namespace NUMINAMATH_CALUDE_triangle_properties_l113_11349

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  -- Given condition
  c * Real.sin ((A + C) / 2) = b * Real.sin C →
  -- BD is altitude from B to AC, BD = 1, b = √3
  b = Real.sqrt 3 →
  ∃ (D : Real),
    D > 0 ∧
    D * Real.sin B = 1 →
    -- Prove:
    -- 1. Angle B = π/3
    B = π / 3 ∧
    -- 2. Perimeter of triangle ABC = 3 + √3
    a + b + c = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l113_11349


namespace NUMINAMATH_CALUDE_fib_105_mod_7_l113_11301

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- The period of Fibonacci sequence modulo 7 -/
def fib_mod7_period : ℕ := 16

theorem fib_105_mod_7 : fib 104 % 7 = 2 := by
  sorry

#eval fib 104 % 7

end NUMINAMATH_CALUDE_fib_105_mod_7_l113_11301


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l113_11346

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∀ n : ℕ, n > 0 → 
  (¬ is_prime n) → 
  (¬ is_square n) → 
  (smallest_prime_factor n ≥ 60) → 
  n ≥ 4087 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l113_11346


namespace NUMINAMATH_CALUDE_book_price_increase_l113_11317

theorem book_price_increase (original_price : ℝ) : 
  original_price * (1 + 0.6) = 480 → original_price = 300 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_l113_11317


namespace NUMINAMATH_CALUDE_max_triangle_side_l113_11385

theorem max_triangle_side (a : ℕ) : 
  (3 + 8 > a ∧ 3 + a > 8 ∧ 8 + a > 3) → a ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_l113_11385


namespace NUMINAMATH_CALUDE_solution_set_contains_two_and_zero_l113_11377

/-- The solution set of the inequality (1+k²)x ≤ k⁴+4 with respect to x -/
def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (1 + k^2) * x ≤ k^4 + 4}

/-- For any real constant k, both 2 and 0 are in the solution set M -/
theorem solution_set_contains_two_and_zero :
  ∀ k : ℝ, (2 ∈ M k) ∧ (0 ∈ M k) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_contains_two_and_zero_l113_11377


namespace NUMINAMATH_CALUDE_car_distance_traveled_l113_11311

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) :
  time = 11 →
  speed = 65 →
  distance = speed * time →
  distance = 715 := by sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l113_11311


namespace NUMINAMATH_CALUDE_new_person_weight_l113_11363

/-- Given a group of 8 people, when one person weighing 20 kg is replaced by a new person,
    and the average weight increases by 2.5 kg, the weight of the new person is 40 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_removed : Real) (avg_increase : Real) :
  initial_count = 8 →
  weight_removed = 20 →
  avg_increase = 2.5 →
  (initial_count : Real) * avg_increase + weight_removed = 40 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l113_11363


namespace NUMINAMATH_CALUDE_university_distribution_l113_11371

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of universities --/
def num_universities : ℕ := 5

/-- The number of students that should choose Peking University --/
def students_to_peking : ℕ := 2

/-- The number of ways to distribute students among universities with the given constraints --/
def distribution_count : ℕ := 640

theorem university_distribution :
  (num_students.choose students_to_peking) * 
  (num_universities - 1) ^ (num_students - students_to_peking) = 
  distribution_count := by sorry

end NUMINAMATH_CALUDE_university_distribution_l113_11371


namespace NUMINAMATH_CALUDE_xy_value_l113_11388

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 15/8) :
  x * y = Real.sqrt 17 / 4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l113_11388


namespace NUMINAMATH_CALUDE_manganese_percentage_after_iron_addition_l113_11351

theorem manganese_percentage_after_iron_addition 
  (initial_mixture_mass : ℝ)
  (initial_manganese_percentage : ℝ)
  (added_iron_mass : ℝ)
  (h1 : initial_mixture_mass = 1)
  (h2 : initial_manganese_percentage = 20)
  (h3 : added_iron_mass = 1)
  : (initial_manganese_percentage / 100 * initial_mixture_mass) / 
    (initial_mixture_mass + added_iron_mass) * 100 = 10 := by
  sorry

#check manganese_percentage_after_iron_addition

end NUMINAMATH_CALUDE_manganese_percentage_after_iron_addition_l113_11351


namespace NUMINAMATH_CALUDE_original_paint_intensity_l113_11361

/-- Given a paint mixture scenario, prove that the original paint intensity was 50%. -/
theorem original_paint_intensity
  (replacement_intensity : ℝ)
  (final_intensity : ℝ)
  (replaced_fraction : ℝ)
  (h1 : replacement_intensity = 25)
  (h2 : final_intensity = 30)
  (h3 : replaced_fraction = 0.8)
  : (1 - replaced_fraction) * 50 + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#check original_paint_intensity

end NUMINAMATH_CALUDE_original_paint_intensity_l113_11361


namespace NUMINAMATH_CALUDE_max_perimeter_constrained_quadrilateral_l113_11337

/-- A convex quadrilateral with specific side and diagonal constraints -/
structure ConstrainedQuadrilateral where
  -- Two sides are equal to 1
  side1 : ℝ
  side2 : ℝ
  side1_eq_one : side1 = 1
  side2_eq_one : side2 = 1
  -- Other sides and diagonals are not greater than 1
  side3 : ℝ
  side4 : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  side3_le_one : side3 ≤ 1
  side4_le_one : side4 ≤ 1
  diagonal1_le_one : diagonal1 ≤ 1
  diagonal2_le_one : diagonal2 ≤ 1
  -- Convexity condition (simplified for this problem)
  is_convex : diagonal1 + diagonal2 > side1 + side3

/-- The maximum perimeter of a constrained quadrilateral -/
theorem max_perimeter_constrained_quadrilateral (q : ConstrainedQuadrilateral) :
  q.side1 + q.side2 + q.side3 + q.side4 ≤ 2 + 4 * Real.sin (15 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_constrained_quadrilateral_l113_11337


namespace NUMINAMATH_CALUDE_m_range_l113_11382

-- Define the condition function
def condition (x m : ℝ) : Prop := x^2 + m*x - 2*m^2 < 0

-- Define the sufficient condition
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, sufficient_condition x → condition x m) →
  (∃ x, condition x m ∧ ¬sufficient_condition x) →
  m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l113_11382


namespace NUMINAMATH_CALUDE_farmer_apples_l113_11395

/-- Given a farmer with 127 apples who gives 88 apples to his neighbor,
    prove that the farmer will have 39 apples remaining. -/
theorem farmer_apples : 
  let initial_apples : ℕ := 127
  let given_apples : ℕ := 88
  initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l113_11395


namespace NUMINAMATH_CALUDE_initial_medium_size_shoes_l113_11357

/-- Given a shoe shop's inventory and sales data, prove the initial number of medium-size shoes. -/
theorem initial_medium_size_shoes
  (large_size : Nat) -- Initial number of large-size shoes
  (small_size : Nat) -- Initial number of small-size shoes
  (sold : Nat) -- Number of shoes sold
  (remaining : Nat) -- Number of shoes remaining after sale
  (h1 : large_size = 22)
  (h2 : small_size = 24)
  (h3 : sold = 83)
  (h4 : remaining = 13)
  (h5 : ∃ M : Nat, large_size + M + small_size = sold + remaining) :
  ∃ M : Nat, M = 26 ∧ large_size + M + small_size = sold + remaining :=
by sorry


end NUMINAMATH_CALUDE_initial_medium_size_shoes_l113_11357


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l113_11362

theorem termite_ridden_not_collapsing (total_homes : ℕ) 
  (termite_ridden : ℚ) (collapsing_ratio : ℚ) :
  termite_ridden = 1 / 3 →
  collapsing_ratio = 7 / 10 →
  (termite_ridden - termite_ridden * collapsing_ratio) = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l113_11362


namespace NUMINAMATH_CALUDE_coffee_discount_difference_l113_11396

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

theorem coffee_discount_difference :
  let sequence1 := (initial_price - fixed_discount) * (1 - percentage_discount)
  let sequence2 := (initial_price * (1 - percentage_discount)) - fixed_discount
  sequence1 - sequence2 = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_coffee_discount_difference_l113_11396


namespace NUMINAMATH_CALUDE_j_travel_time_l113_11314

/-- Given two travelers J and L, where:
  * J takes 45 minutes less time than L to travel 45 miles
  * J travels 1/2 mile per hour faster than L
  * y is J's rate of speed in miles per hour
Prove that J's time to travel 45 miles is equal to 45/y -/
theorem j_travel_time (y : ℝ) (h1 : y > 0) : ∃ (t_j t_l : ℝ),
  t_j = 45 / y ∧
  t_l = 45 / (y - 1/2) ∧
  t_l - t_j = 3/4 :=
sorry

end NUMINAMATH_CALUDE_j_travel_time_l113_11314


namespace NUMINAMATH_CALUDE_parabola_c_value_l113_11374

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^2 + b * x + c

theorem parabola_c_value :
  ∀ b c : ℝ, 
  Parabola b c 2 = 12 ∧ 
  Parabola b c 4 = 44 →
  c = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l113_11374


namespace NUMINAMATH_CALUDE_max_distance_between_points_l113_11339

/-- Given vector OA = (1, -1) and |OA| = |OB|, the maximum value of |AB| is 2√2. -/
theorem max_distance_between_points (OA OB : ℝ × ℝ) : 
  OA = (1, -1) → 
  Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) = Real.sqrt ((OB.1 ^ 2) + (OB.2 ^ 2)) →
  (∃ (AB : ℝ × ℝ), AB = OB - OA ∧ 
    Real.sqrt ((AB.1 ^ 2) + (AB.2 ^ 2)) ≤ 2 * Real.sqrt 2 ∧
    ∃ (OB' : ℝ × ℝ), Real.sqrt ((OB'.1 ^ 2) + (OB'.2 ^ 2)) = Real.sqrt ((OA.1 ^ 2) + (OA.2 ^ 2)) ∧
      let AB' := OB' - OA
      Real.sqrt ((AB'.1 ^ 2) + (AB'.2 ^ 2)) = 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_between_points_l113_11339


namespace NUMINAMATH_CALUDE_teena_speed_calculation_l113_11348

/-- Teena's speed in miles per hour -/
def teena_speed : ℝ := 55

/-- Loe's speed in miles per hour -/
def loe_speed : ℝ := 40

/-- Initial distance Teena is behind Loe in miles -/
def initial_distance_behind : ℝ := 7.5

/-- Time after which Teena is ahead of Loe in hours -/
def time_elapsed : ℝ := 1.5

/-- Distance Teena is ahead of Loe after time_elapsed in miles -/
def final_distance_ahead : ℝ := 15

theorem teena_speed_calculation :
  teena_speed * time_elapsed = 
    initial_distance_behind + final_distance_ahead + (loe_speed * time_elapsed) := by
  sorry

end NUMINAMATH_CALUDE_teena_speed_calculation_l113_11348


namespace NUMINAMATH_CALUDE_simplify_expression_l113_11383

theorem simplify_expression (x : ℝ) : 8*x - 3 + 2*x - 7 + 4*x + 15 = 14*x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l113_11383


namespace NUMINAMATH_CALUDE_product_of_powers_l113_11372

theorem product_of_powers : 3^2 * 5^2 * 7 * 11^2 = 190575 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l113_11372


namespace NUMINAMATH_CALUDE_no_x_squared_term_l113_11343

/-- 
Given the algebraic expression (x-2)(ax²-x+1), this theorem states that
the coefficient of x² in the expanded form is zero if and only if a = -1/2.
-/
theorem no_x_squared_term (x a : ℝ) : 
  (x - 2) * (a * x^2 - x + 1) = a * x^3 + 3 * x - 2 ↔ a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l113_11343


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l113_11399

theorem min_sum_with_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = x * y → a + b ≤ x + y ∧ a + b = 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l113_11399


namespace NUMINAMATH_CALUDE_expression_evaluation_l113_11320

theorem expression_evaluation :
  let x : ℝ := (Real.pi - 3) ^ 0
  let y : ℝ := (-1/3)⁻¹
  ((2*x - y)^2 - (y + 2*x) * (y - 2*x)) / (-1/2 * x) = -40 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l113_11320


namespace NUMINAMATH_CALUDE_hyperbola_fixed_point_l113_11330

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the focal distance
def focal_distance : ℝ := 4

-- Define the left focus
def left_focus : ℝ × ℝ := (-2, 0)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

-- Define the vector from F to a point
def vector_FP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - left_focus.1, P.2 - left_focus.2)

-- State the theorem
theorem hyperbola_fixed_point :
  ∃ M : ℝ × ℝ,
    M.2 = 0 ∧
    (∀ P Q : ℝ × ℝ,
      hyperbola P.1 P.2 → hyperbola Q.1 Q.2 →
      (∃ t : ℝ, P.2 - M.2 = t * (P.1 - M.1) ∧ Q.2 - M.2 = t * (Q.1 - M.1)) →
      dot_product (vector_FP P) (vector_FP Q) = 1) ∧
    M = (-3 - Real.sqrt 3, 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_fixed_point_l113_11330


namespace NUMINAMATH_CALUDE_probability_of_specific_outcome_l113_11341

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of six coins -/
structure SixCoins :=
  (penny : CoinFlip)
  (nickel : CoinFlip)
  (dime : CoinFlip)
  (quarter : CoinFlip)
  (halfDollar : CoinFlip)
  (oneDollar : CoinFlip)

/-- The total number of possible outcomes when flipping six coins -/
def totalOutcomes : Nat := 64

/-- A specific outcome we're interested in -/
def specificOutcome : SixCoins :=
  { penny := CoinFlip.Heads,
    nickel := CoinFlip.Heads,
    dime := CoinFlip.Heads,
    quarter := CoinFlip.Tails,
    halfDollar := CoinFlip.Tails,
    oneDollar := CoinFlip.Tails }

/-- The probability of getting the specific outcome when flipping six coins -/
theorem probability_of_specific_outcome :
  (1 : ℚ) / totalOutcomes = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_outcome_l113_11341


namespace NUMINAMATH_CALUDE_average_problem_l113_11336

theorem average_problem (x : ℝ) : 
  let numbers := [54, 55, 57, 58, 59, 62, 62, 63, x]
  (numbers.sum / numbers.length : ℝ) = 60 → x = 70 := by
sorry

end NUMINAMATH_CALUDE_average_problem_l113_11336


namespace NUMINAMATH_CALUDE_bridge_length_l113_11354

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l113_11354


namespace NUMINAMATH_CALUDE_carpet_dimensions_l113_11328

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  width : ℝ

/-- Represents a rectangular carpet --/
structure Carpet where
  length : ℝ
  width : ℝ

/-- Checks if a carpet fits in a room such that each corner touches a different wall --/
def fits_in_room (c : Carpet) (r : Room) : Prop :=
  ∃ (α b : ℝ),
    α + (c.length / c.width) * b = r.length ∧
    (c.length / c.width) * α + b = r.width ∧
    c.width^2 = α^2 + b^2

/-- The main theorem to prove --/
theorem carpet_dimensions :
  ∃ (c : Carpet),
    fits_in_room c { length := 38, width := 55 } ∧
    fits_in_room c { length := 50, width := 55 } ∧
    c.length = 50 ∧
    c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carpet_dimensions_l113_11328


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l113_11379

theorem rectangular_plot_dimensions (length width : ℝ) : 
  length = 58 →
  (4 * width + 2 * length) * 26.5 = 5300 →
  length - width = 37 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l113_11379


namespace NUMINAMATH_CALUDE_workers_not_worked_days_l113_11332

/-- Represents the payment schedule for workers over a 30-day period --/
structure WorkerSchedule where
  total_days : ℕ
  daily_wage : ℕ
  daily_penalty : ℕ
  days_not_worked : ℕ
  h_total_days : total_days = 30
  h_daily_wage : daily_wage = 100
  h_daily_penalty : daily_penalty = 25
  h_no_earnings : daily_wage * (total_days - days_not_worked) = daily_penalty * days_not_worked

/-- Theorem stating that under the given conditions, workers did not work for 24 days --/
theorem workers_not_worked_days (schedule : WorkerSchedule) : schedule.days_not_worked = 24 := by
  sorry


end NUMINAMATH_CALUDE_workers_not_worked_days_l113_11332


namespace NUMINAMATH_CALUDE_divisibility_by_hundred_l113_11364

theorem divisibility_by_hundred (N : ℕ) : 
  N = 2^5 * 3^2 * 7 * 75 → 100 ∣ N := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_hundred_l113_11364


namespace NUMINAMATH_CALUDE_negation_equivalence_l113_11322

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l113_11322


namespace NUMINAMATH_CALUDE_equation_solution_l113_11334

theorem equation_solution : 
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l113_11334


namespace NUMINAMATH_CALUDE_sum_of_divisors_420_l113_11381

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (·∣n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_420 : sum_of_divisors 420 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_420_l113_11381


namespace NUMINAMATH_CALUDE_wall_width_proof_l113_11358

theorem wall_width_proof (width height length : ℝ) 
  (height_def : height = 6 * width)
  (length_def : length = 7 * height)
  (volume_def : width * height * length = 86436) :
  width = 7 :=
by sorry

end NUMINAMATH_CALUDE_wall_width_proof_l113_11358


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l113_11306

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 ≥ 121 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ θ φ : ℝ, (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_trig_expression_achievable_l113_11306


namespace NUMINAMATH_CALUDE_f_properties_l113_11342

-- Define the function f(x) = x³ - 3x²
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- State the theorem
theorem f_properties :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧  -- f is increasing on (-∞, 0)
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧  -- f is decreasing on (0, 2)
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧  -- f is increasing on (2, +∞)
  (∀ x, x ≠ 0 → f x ≤ f 0) ∧  -- f(0) is a local maximum
  (∀ x, x ≠ 2 → f x ≥ f 2) ∧  -- f(2) is a local minimum
  f 0 = 0 ∧  -- value at x = 0
  f 2 = -4  -- value at x = 2
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l113_11342


namespace NUMINAMATH_CALUDE_prime_sum_seven_power_l113_11393

theorem prime_sum_seven_power (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 7 → (p^q = 32 ∨ p^q = 25) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_seven_power_l113_11393


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l113_11338

def C : Set Nat := {47, 49, 51, 53, 55}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧
  (∀ (m : Nat), m ∈ C → (Nat.minFac n ≤ Nat.minFac m)) ∧
  n = 51 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l113_11338


namespace NUMINAMATH_CALUDE_book_price_percentage_l113_11370

theorem book_price_percentage (suggested_retail_price : ℝ) : 
  suggested_retail_price > 0 →
  let marked_price := 0.6 * suggested_retail_price
  let alice_paid := 0.6 * marked_price
  alice_paid / suggested_retail_price = 0.36 := by
  sorry

end NUMINAMATH_CALUDE_book_price_percentage_l113_11370


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l113_11378

theorem unique_solution_for_equation : 
  ∀ (n k : ℕ), 2023 + 2^n = k^2 ↔ n = 1 ∧ k = 45 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l113_11378


namespace NUMINAMATH_CALUDE_initial_men_count_l113_11324

/-- Given provisions that last 15 days for an initial group of men and 12.5 days when 200 more men join,
    prove that the initial number of men is 1000. -/
theorem initial_men_count (M : ℕ) (P : ℝ) : 
  (P / (15 * M) = P / (12.5 * (M + 200))) → M = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l113_11324


namespace NUMINAMATH_CALUDE_right_triangle_angles_l113_11329

theorem right_triangle_angles (A B C : Real) (h1 : A + B + C = 180) (h2 : C = 90) (h3 : A = 50) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l113_11329
