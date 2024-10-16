import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_equation_from_parameters_l446_44670

/-- Represents an ellipse with axes aligned to the coordinate system -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  e : ℝ  -- Eccentricity
  h : 0 < a ∧ 0 < b ∧ 0 ≤ e ∧ e < 1  -- Constraints on a, b, and e

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

theorem ellipse_equation_from_parameters :
  ∀ E : Ellipse,
    E.e = 2/3 →
    E.b = 4 * Real.sqrt 5 →
    (∀ x y : ℝ, ellipse_equation E x y ↔ x^2/144 + y^2/80 = 1) ∨
    (∀ x y : ℝ, ellipse_equation E x y ↔ y^2/144 + x^2/80 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_parameters_l446_44670


namespace NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_plus_3_l446_44693

theorem gcd_n_cube_minus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 - 27) (n + 3) = if (n + 3) % 9 = 0 then 9 else 1 := by sorry

end NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_plus_3_l446_44693


namespace NUMINAMATH_CALUDE_percentage_before_break_l446_44642

/-- Given a total number of pages and the number of pages to read after a break,
    calculate the percentage of pages that must be read before the break. -/
theorem percentage_before_break (total_pages : ℕ) (pages_after_break : ℕ) 
    (h1 : total_pages = 30) (h2 : pages_after_break = 9) : 
    (((total_pages - pages_after_break : ℚ) / total_pages) * 100 = 70) := by
  sorry

end NUMINAMATH_CALUDE_percentage_before_break_l446_44642


namespace NUMINAMATH_CALUDE_equation_solutions_l446_44641

def equation (x : ℝ) : Prop :=
  (17 * x - x^2) / (x + 2) * (x + (17 - x) / (x + 2)) = 48

theorem equation_solutions :
  {x : ℝ | equation x} = {3, 4, -10 + 4 * Real.sqrt 21, -10 - 4 * Real.sqrt 21} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l446_44641


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l446_44662

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove the common difference is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of S_n for arithmetic sequence
  (h2 : S 4 = 3 * S 2)  -- Given condition
  (h3 : a 7 = 15)  -- Given condition
  : ∃ d : ℝ, ∀ n, a (n + 1) - a n = d ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l446_44662


namespace NUMINAMATH_CALUDE_clock_sale_correct_l446_44643

/-- Represents the clock selling scenario --/
structure ClockSale where
  originalCost : ℝ
  collectorPrice : ℝ
  buybackPrice : ℝ
  finalPrice : ℝ

/-- The clock sale scenario satisfying all given conditions --/
def clockScenario : ClockSale :=
  { originalCost := 250,
    collectorPrice := 300,
    buybackPrice := 150,
    finalPrice := 270 }

/-- Theorem stating that the given scenario satisfies all conditions and results in the correct final price --/
theorem clock_sale_correct (c : ClockSale) (h : c = clockScenario) : 
  c.collectorPrice = c.originalCost * 1.2 ∧ 
  c.buybackPrice = c.collectorPrice * 0.5 ∧
  c.originalCost - c.buybackPrice = 100 ∧
  c.finalPrice = c.buybackPrice * 1.8 := by
  sorry

#check clock_sale_correct

end NUMINAMATH_CALUDE_clock_sale_correct_l446_44643


namespace NUMINAMATH_CALUDE_real_roots_of_x_squared_minus_four_l446_44661

theorem real_roots_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_of_x_squared_minus_four_l446_44661


namespace NUMINAMATH_CALUDE_arrange_five_balls_three_boxes_l446_44629

/-- The number of ways to put n distinguishable objects into k distinguishable containers -/
def arrange_objects (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem arrange_five_balls_three_boxes : arrange_objects 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_arrange_five_balls_three_boxes_l446_44629


namespace NUMINAMATH_CALUDE_smaug_copper_coins_l446_44683

/-- Represents the number of coins of each type in Smaug's hoard -/
structure DragonHoard where
  gold : ℕ
  silver : ℕ
  copper : ℕ

/-- Calculates the total value of a hoard in copper coins -/
def hoardValue (h : DragonHoard) (silverValue copperValue : ℕ) : ℕ :=
  h.gold * silverValue * copperValue + h.silver * copperValue + h.copper

/-- Theorem stating that Smaug has 33 copper coins -/
theorem smaug_copper_coins :
  ∃ (h : DragonHoard),
    h.gold = 100 ∧
    h.silver = 60 ∧
    hoardValue h 3 8 = 2913 ∧
    h.copper = 33 := by
  sorry

end NUMINAMATH_CALUDE_smaug_copper_coins_l446_44683


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_6_to_1993_l446_44617

theorem rightmost_three_digits_of_6_to_1993 :
  6^1993 ≡ 296 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_6_to_1993_l446_44617


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l446_44650

theorem expression_simplification_and_evaluation (x : ℝ) :
  (x + 2) * (x - 3) - x * (2 * x - 1) = -x^2 - 6 ∧
  ((2 : ℝ) + 2) * ((2 : ℝ) - 3) - (2 : ℝ) * (2 * (2 : ℝ) - 1) = -10 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l446_44650


namespace NUMINAMATH_CALUDE_curve_self_intersection_l446_44665

/-- The x-coordinate of a point on the curve for a given t -/
def x (t : ℝ) : ℝ := 2 * t^2 - 3

/-- The y-coordinate of a point on the curve for a given t -/
def y (t : ℝ) : ℝ := 2 * t^4 - 9 * t^2 + 6

/-- Theorem stating that (-1, -1) is a self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ x t₁ = x t₂ ∧ y t₁ = y t₂ ∧ x t₁ = -1 ∧ y t₁ = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l446_44665


namespace NUMINAMATH_CALUDE_parabola_rotation_l446_44619

/-- A parabola is defined by its coefficients a, h, and k in the form y = a(x - h)² + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Rotate a parabola by 180° around the origin -/
def rotate180 (p : Parabola) : Parabola :=
  { a := -p.a, h := -p.h, k := -p.k }

theorem parabola_rotation :
  let p := Parabola.mk 2 1 2
  rotate180 p = Parabola.mk (-2) (-1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_rotation_l446_44619


namespace NUMINAMATH_CALUDE_trophy_cost_l446_44664

theorem trophy_cost (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  let total_cents : ℕ := 1000 * x + 9990 + y
  (72 ∣ total_cents) →
  (total_cents : ℚ) / (72 * 100) = 11.11 := by
sorry

end NUMINAMATH_CALUDE_trophy_cost_l446_44664


namespace NUMINAMATH_CALUDE_divisibility_implies_one_l446_44672

theorem divisibility_implies_one (n : ℕ+) (h : n ∣ 2^n.val - 1) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_one_l446_44672


namespace NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l446_44666

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  a : Point2D
  b : Point2D
  c : Point2D
  d : Point2D

/-- Represents an oblique projection transformation -/
def ObliqueProjection := Point2D → Point2D

/-- Checks if four points form a parallelogram -/
def isParallelogram (a b c d : Point2D) : Prop := sorry

/-- The theorem stating that oblique projection preserves parallelograms -/
theorem oblique_projection_preserves_parallelogram 
  (p : Parallelogram) (proj : ObliqueProjection) :
  let p' := Parallelogram.mk 
    (proj p.a) (proj p.b) (proj p.c) (proj p.d)
  isParallelogram p'.a p'.b p'.c p'.d := by sorry

end NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l446_44666


namespace NUMINAMATH_CALUDE_average_extra_chores_l446_44682

/-- Proves that given the specified conditions, the average number of extra chores per week is 15 -/
theorem average_extra_chores
  (fixed_allowance : ℝ)
  (extra_chore_pay : ℝ)
  (total_weeks : ℕ)
  (total_earned : ℝ)
  (h1 : fixed_allowance = 20)
  (h2 : extra_chore_pay = 1.5)
  (h3 : total_weeks = 10)
  (h4 : total_earned = 425) :
  (total_earned / total_weeks - fixed_allowance) / extra_chore_pay = 15 := by
  sorry

#check average_extra_chores

end NUMINAMATH_CALUDE_average_extra_chores_l446_44682


namespace NUMINAMATH_CALUDE_line_equation_proof_l446_44656

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The projection point P(-2,1) -/
def projection_point : Point := ⟨-2, 1⟩

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The line perpendicular to a given line passing through a point -/
def perpendicular_line_through_point (l : Line) (p : Point) : Line :=
  ⟨l.b, -l.a, l.a * p.y - l.b * p.x⟩

theorem line_equation_proof (L : Line) : 
  (point_on_line projection_point L) ∧ 
  (perpendicular L (line_through_points origin projection_point)) →
  L = ⟨2, -1, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_line_equation_proof_l446_44656


namespace NUMINAMATH_CALUDE_opposite_reciprocal_equation_l446_44690

theorem opposite_reciprocal_equation (a b c d : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposites
  (h2 : c * d = 1)  -- c and d are reciprocals
  : (a + b)^2 - 3*(c*d)^4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_equation_l446_44690


namespace NUMINAMATH_CALUDE_problem_1_l446_44659

theorem problem_1 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l446_44659


namespace NUMINAMATH_CALUDE_oil_volume_in_tank_l446_44680

/-- The volume of oil in a cylindrical tank with given dimensions and mixture ratio -/
theorem oil_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) (fill_percentage : ℝ) 
  (oil_ratio : ℝ) (water_ratio : ℝ) (h_height : tank_height = 8) 
  (h_diameter : tank_diameter = 3) (h_fill : fill_percentage = 0.75) 
  (h_ratio : oil_ratio / (oil_ratio + water_ratio) = 3 / 10) : 
  (fill_percentage * π * (tank_diameter / 2)^2 * tank_height) * (oil_ratio / (oil_ratio + water_ratio)) = 4.05 * π := by
sorry

end NUMINAMATH_CALUDE_oil_volume_in_tank_l446_44680


namespace NUMINAMATH_CALUDE_upstream_speed_l446_44695

theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45) 
  (h2 : speed_downstream = 53) : 
  speed_still - (speed_downstream - speed_still) = 37 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_l446_44695


namespace NUMINAMATH_CALUDE_brown_ball_weight_calculation_l446_44675

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := total_weight - blue_ball_weight

theorem brown_ball_weight_calculation :
  brown_ball_weight = 3.12 := by sorry

end NUMINAMATH_CALUDE_brown_ball_weight_calculation_l446_44675


namespace NUMINAMATH_CALUDE_system_equations_solutions_l446_44651

theorem system_equations_solutions (a x y : ℝ) : 
  (x - y = 2*a + 1 ∧ 2*x + 3*y = 9*a - 8) →
  ((x = y → a = -1/2) ∧
   (x > 0 ∧ y < 0 ∧ x + y = 0 → a = 3/4)) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_solutions_l446_44651


namespace NUMINAMATH_CALUDE_total_revenue_equals_8189_35_l446_44630

-- Define the types of ground beef
structure GroundBeef where
  regular : ℝ
  lean : ℝ
  extraLean : ℝ

-- Define the prices
def regularPrice : ℝ := 3.50
def leanPrice : ℝ := 4.25
def extraLeanPrice : ℝ := 5.00

-- Define the sales for each day
def mondaySales : GroundBeef := { regular := 198.5, lean := 276.2, extraLean := 150.7 }
def tuesdaySales : GroundBeef := { regular := 210, lean := 420, extraLean := 150 }
def wednesdaySales : GroundBeef := { regular := 230, lean := 324.6, extraLean := 120.4 }

-- Define the discount for Tuesday
def tuesdayDiscount : ℝ := 0.1

-- Define the sale price for lean ground beef on Wednesday
def wednesdayLeanSalePrice : ℝ := 3.75

-- Function to calculate revenue for a single day
def calculateDayRevenue (sales : GroundBeef) (regularPrice leanPrice extraLeanPrice : ℝ) : ℝ :=
  sales.regular * regularPrice + sales.lean * leanPrice + sales.extraLean * extraLeanPrice

-- Theorem statement
theorem total_revenue_equals_8189_35 :
  let mondayRevenue := calculateDayRevenue mondaySales regularPrice leanPrice extraLeanPrice
  let tuesdayRevenue := calculateDayRevenue tuesdaySales (regularPrice * (1 - tuesdayDiscount)) (leanPrice * (1 - tuesdayDiscount)) (extraLeanPrice * (1 - tuesdayDiscount))
  let wednesdayRevenue := calculateDayRevenue wednesdaySales regularPrice wednesdayLeanSalePrice extraLeanPrice
  mondayRevenue + tuesdayRevenue + wednesdayRevenue = 8189.35 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_equals_8189_35_l446_44630


namespace NUMINAMATH_CALUDE_johns_journey_distance_l446_44623

/-- Calculates the total distance traveled by John given his journey conditions -/
def total_distance (
  initial_driving_speed : ℝ)
  (initial_driving_time : ℝ)
  (second_driving_speed : ℝ)
  (second_driving_time : ℝ)
  (biking_speed : ℝ)
  (biking_time : ℝ)
  (walking_speed : ℝ)
  (walking_time : ℝ) : ℝ :=
  initial_driving_speed * initial_driving_time +
  second_driving_speed * second_driving_time +
  biking_speed * biking_time +
  walking_speed * walking_time

/-- Theorem stating that John's total travel distance is 179 miles -/
theorem johns_journey_distance : 
  total_distance 55 2 45 1 15 1.5 3 0.5 = 179 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l446_44623


namespace NUMINAMATH_CALUDE_number_equation_solution_l446_44635

theorem number_equation_solution : 
  ∃ x : ℝ, (2 * x = 3 * x - 25) ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l446_44635


namespace NUMINAMATH_CALUDE_xy_value_l446_44600

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l446_44600


namespace NUMINAMATH_CALUDE_power_function_m_value_l446_44663

/-- A power function that passes through (2, 16) and (1/2, m) -/
def power_function (x : ℝ) : ℝ := x ^ 4

theorem power_function_m_value :
  let f := power_function
  (f 2 = 16) ∧ (∃ m, f (1/2) = m) →
  f (1/2) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_power_function_m_value_l446_44663


namespace NUMINAMATH_CALUDE_sasha_picked_24_leaves_l446_44649

/-- The number of apple trees along the road. -/
def apple_trees : ℕ := 17

/-- The number of poplar trees along the road. -/
def poplar_trees : ℕ := 20

/-- The index of the apple tree from which Sasha starts picking leaves. -/
def start_tree : ℕ := 8

/-- The total number of trees along the road. -/
def total_trees : ℕ := apple_trees + poplar_trees

/-- The number of leaves Sasha picked. -/
def leaves_picked : ℕ := total_trees - (start_tree - 1)

theorem sasha_picked_24_leaves : leaves_picked = 24 := by
  sorry

end NUMINAMATH_CALUDE_sasha_picked_24_leaves_l446_44649


namespace NUMINAMATH_CALUDE_amount_in_scientific_notation_l446_44636

-- Define the amount in yuan
def amount : ℕ := 25000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 10)

-- Theorem statement
theorem amount_in_scientific_notation :
  (amount : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_amount_in_scientific_notation_l446_44636


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l446_44692

theorem right_triangle_acute_angles (α : Real) 
  (h1 : 0 < α ∧ α < 90) 
  (h2 : (90 - α / 2) / (45 + α / 2) = 13 / 17) : 
  α = 63 ∧ 90 - α = 27 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l446_44692


namespace NUMINAMATH_CALUDE_either_shooter_hits_probability_l446_44612

-- Define the probabilities for shooters A and B
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Define the probability that either A or B hits the target
def prob_either_hits : ℝ := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

-- Theorem statement
theorem either_shooter_hits_probability :
  prob_either_hits = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_either_shooter_hits_probability_l446_44612


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l446_44627

/-- A line l passes through point A(t,0) and is tangent to the curve y = x^2 with an angle of inclination of 45° -/
theorem tangent_line_intersection (t : ℝ) : 
  (∃ (m : ℝ), 
    -- The line passes through (t, 0)
    (t - m) * (m^2 - 0) = (1 - 0) * (0 - t) ∧ 
    -- The line is tangent to y = x^2 at (m, m^2)
    2 * m = 1 ∧ 
    -- The angle of inclination is 45°
    (m^2 - 0) / (m - t) = 1) → 
  t = 1/4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l446_44627


namespace NUMINAMATH_CALUDE_constant_term_expansion_l446_44611

theorem constant_term_expansion (b : ℝ) (h : b = -1/2) :
  let c := 6 * b^2
  c = 3/2 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l446_44611


namespace NUMINAMATH_CALUDE_standard_deviation_is_2_l446_44653

def data : List ℝ := [51, 54, 55, 57, 53]

theorem standard_deviation_is_2 :
  let mean := (data.sum) / (data.length : ℝ)
  let variance := (data.map (λ x => (x - mean) ^ 2)).sum / (data.length : ℝ)
  Real.sqrt variance = 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_is_2_l446_44653


namespace NUMINAMATH_CALUDE_simplify_expression_l446_44696

theorem simplify_expression (a b : ℝ) :
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) = 30 * a + 39 * b + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l446_44696


namespace NUMINAMATH_CALUDE_common_tangents_count_l446_44639

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define a function to count common tangent lines
noncomputable def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l446_44639


namespace NUMINAMATH_CALUDE_deepak_third_period_profit_l446_44673

def anand_investment : ℕ := 22500
def deepak_investment : ℕ := 35000
def total_investment : ℕ := anand_investment + deepak_investment

def first_period_profit : ℕ := 9600
def second_period_profit : ℕ := 12800
def third_period_profit : ℕ := 18000

def profit_share (investment : ℕ) (total_profit : ℕ) : ℚ :=
  (investment : ℚ) / (total_investment : ℚ) * (total_profit : ℚ)

theorem deepak_third_period_profit :
  profit_share deepak_investment third_period_profit = 10960 := by
  sorry

end NUMINAMATH_CALUDE_deepak_third_period_profit_l446_44673


namespace NUMINAMATH_CALUDE_chess_sets_problem_l446_44697

theorem chess_sets_problem (x : ℕ) (y : ℕ) : 
  (x > 0) →
  (y > 0) →
  (16 * x = y * ((16 * x) / y)) →
  ((16 * x) / y + 2) * (y - 10) = 16 * x →
  ((16 * x) / y + 4) * (y - 16) = 16 * x →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_sets_problem_l446_44697


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l446_44691

theorem simplify_sqrt_sum : 
  Real.sqrt 0.5 + Real.sqrt (0.5 + 1.5) + Real.sqrt (0.5 + 1.5 + 2.5) + 
  Real.sqrt (0.5 + 1.5 + 2.5 + 3.5) = Real.sqrt 0.5 + 3 * Real.sqrt 2 + Real.sqrt 4.5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l446_44691


namespace NUMINAMATH_CALUDE_probability_triangle_or_circle_l446_44667

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) (squares : ℕ)
  (h_total : total = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3)
  (h_sum : triangles + circles + squares = total) :
  (triangles + circles : ℚ) / total = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_triangle_or_circle_l446_44667


namespace NUMINAMATH_CALUDE_lucys_journey_l446_44688

theorem lucys_journey (total : ℚ) 
  (h1 : total / 4 + 25 + total / 6 = total) : total = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_lucys_journey_l446_44688


namespace NUMINAMATH_CALUDE_men_in_first_group_l446_44626

/-- The number of days taken by the first group to complete the job -/
def days_group1 : ℕ := 15

/-- The number of men in the second group -/
def men_group2 : ℕ := 25

/-- The number of days taken by the second group to complete the job -/
def days_group2 : ℕ := 24

/-- The amount of work done is the product of the number of workers and the number of days they work -/
def work_done (men : ℕ) (days : ℕ) : ℕ := men * days

/-- The theorem stating that the number of men in the first group is 40 -/
theorem men_in_first_group : 
  ∃ (men_group1 : ℕ), 
    men_group1 = 40 ∧ 
    work_done men_group1 days_group1 = work_done men_group2 days_group2 :=
by sorry

end NUMINAMATH_CALUDE_men_in_first_group_l446_44626


namespace NUMINAMATH_CALUDE_square_difference_l446_44646

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l446_44646


namespace NUMINAMATH_CALUDE_inverse_proportion_change_l446_44620

/-- Given positive numbers x and y that are inversely proportional, prove that when x doubles, y decreases by 50% -/
theorem inverse_proportion_change (x y x' y' k : ℝ) :
  x > 0 →
  y > 0 →
  x * y = k →
  x' = 2 * x →
  x' * y' = k →
  y' / y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_change_l446_44620


namespace NUMINAMATH_CALUDE_special_polyhedron_property_l446_44658

-- Define the structure of our polyhedron
structure Polyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  T : ℕ  -- number of triangular faces
  P : ℕ  -- number of square faces

-- Define the properties of our specific polyhedron
def SpecialPolyhedron (poly : Polyhedron) : Prop :=
  poly.V - poly.E + poly.F = 2 ∧  -- Euler's formula
  poly.F = 40 ∧                   -- Total number of faces
  poly.T + poly.P = poly.F ∧      -- Faces are either triangles or squares
  poly.T = 1 ∧                    -- Number of triangular faces at a vertex
  poly.P = 3 ∧                    -- Number of square faces at a vertex
  poly.E = (3 * poly.T + 4 * poly.P) / 2  -- Edge calculation

-- Theorem statement
theorem special_polyhedron_property (poly : Polyhedron) 
  (h : SpecialPolyhedron poly) : 
  100 * poly.P + 10 * poly.T + poly.V = 351 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_property_l446_44658


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l446_44676

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 0.1) : 
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l446_44676


namespace NUMINAMATH_CALUDE_ball_probabilities_l446_44602

/-- Represents the number of balls of each color in the bag -/
structure BallCounts where
  red : ℕ
  yellow : ℕ
  white : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (counts : BallCounts) : ℕ :=
  counts.red + counts.yellow + counts.white

/-- Calculates the probability of picking a ball of a specific color -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  (favorable : ℚ) / (total : ℚ)

theorem ball_probabilities (initial : BallCounts)
    (h_initial : initial = ⟨10, 6, 4⟩) :
  let total := totalBalls initial
  (probability initial.white total = 1/5) ∧
  (probability (initial.red + initial.yellow) total = 4/5) ∧
  (probability (initial.white - 2) (total - 4) = 1/8) := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l446_44602


namespace NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l446_44689

/-- The function f(x) = x(x-c)^2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- f has a local maximum at x = 2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≤ f c 2

theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by sorry

end NUMINAMATH_CALUDE_local_max_implies_c_eq_6_l446_44689


namespace NUMINAMATH_CALUDE_complex_equation_solution_l446_44647

theorem complex_equation_solution : 
  ∃ (z : ℂ), z / (1 - Complex.I) = Complex.I ^ 2019 → z = -1 - Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l446_44647


namespace NUMINAMATH_CALUDE_f_sum_inequality_l446_44657

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_sum_inequality (x : ℝ) :
  (f x + f (x - 1/2) > 1) ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_inequality_l446_44657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l446_44632

/-- An arithmetic sequence with sum of first n terms Sₙ -/
structure ArithmeticSequence where
  S : ℕ → ℝ
  is_arithmetic : ∃ (a₁ d : ℝ), ∀ n : ℕ, S n = n * a₁ + (n * (n - 1) / 2) * d

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = 0 ∧ seq.S 15 = 25

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ a₅ : ℝ, a₅ = -1/3 ∧ ∀ n : ℕ, seq.S n = n * a₅ + (n * (n - 1) / 2) * (2/3)) ∧ 
  (∀ n : ℕ, seq.S n ≥ seq.S 5) ∧
  (∃ min_value : ℝ, min_value = -49 ∧ ∀ n : ℕ, n * seq.S n ≥ min_value) ∧
  (¬∃ max_value : ℝ, ∀ n : ℕ, seq.S n / n ≤ max_value) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l446_44632


namespace NUMINAMATH_CALUDE_problem_solution_l446_44605

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 - x + 2 < 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 ≥ 1

-- Theorem statement
theorem problem_solution :
  (¬p) ∧ q :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l446_44605


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l446_44652

/-- A circle is tangent to two parallel lines and its center lies on a third line -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  (3 * x + 4 * y = 40) ∧ 
  (3 * x + 4 * y = -20) ∧ 
  (x - 3 * y = 0) →
  x = 30 / 13 ∧ y = 10 / 13 := by
  sorry

#check circle_tangent_to_parallel_lines

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l446_44652


namespace NUMINAMATH_CALUDE_peanuts_in_box_l446_44638

/-- The number of peanuts in a box after adding more -/
theorem peanuts_in_box (initial : ℕ) (added : ℕ) : 
  initial = 4 → added = 12 → initial + added = 16 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l446_44638


namespace NUMINAMATH_CALUDE_movie_shelf_problem_l446_44684

/-- The minimum number of additional movies needed to satisfy the conditions -/
def additional_movies_needed (current_movies : ℕ) (num_shelves : ℕ) : ℕ :=
  let target := num_shelves * (current_movies / num_shelves + 1)
  target - current_movies

theorem movie_shelf_problem :
  let current_movies := 9
  let num_shelves := 2
  let result := additional_movies_needed current_movies num_shelves
  (result = 1 ∧
   (current_movies + result) % 2 = 0 ∧
   (current_movies + result) / num_shelves % 2 = 1 ∧
   ∀ (shelf : ℕ), shelf < num_shelves →
     (current_movies + result) / num_shelves = (current_movies + result - shelf * ((current_movies + result) / num_shelves)) / (num_shelves - shelf)) :=
by sorry

end NUMINAMATH_CALUDE_movie_shelf_problem_l446_44684


namespace NUMINAMATH_CALUDE_square_root_of_1024_l446_44685

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l446_44685


namespace NUMINAMATH_CALUDE_distribute_10_8_l446_44607

/-- The number of ways to distribute n distinct objects into k distinct groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Stirling number of the second kind: the number of ways to partition
    a set of n elements into k non-empty subsets -/
def stirling2 (n k : ℕ) : ℕ := sorry

theorem distribute_10_8 :
  distribute 10 8 = 30240000 := by sorry

end NUMINAMATH_CALUDE_distribute_10_8_l446_44607


namespace NUMINAMATH_CALUDE_angle_between_lines_l446_44616

theorem angle_between_lines (r₁ r₂ r₃ : ℝ) (shaded_ratio : ℝ) :
  r₁ = 3 ∧ r₂ = 2 ∧ r₃ = 1 ∧ shaded_ratio = 8/13 →
  ∃ θ : ℝ, 
    θ > 0 ∧ 
    θ < π/2 ∧
    (6 * θ + 4 * π = 24 * π / 7) ∧
    θ = π/7 :=
sorry

end NUMINAMATH_CALUDE_angle_between_lines_l446_44616


namespace NUMINAMATH_CALUDE_count_triples_satisfying_equation_l446_44622

theorem count_triples_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      let (x, y, z) := t
      (x^y)^z = 64 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (Finset.product (Finset.range 64) (Finset.product (Finset.range 64) (Finset.range 64)))).card
  ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_count_triples_satisfying_equation_l446_44622


namespace NUMINAMATH_CALUDE_intersection_equality_subset_relation_l446_44608

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x | -1-2*a ≤ x ∧ x ≤ a-2}

-- Theorem for part (1)
theorem intersection_equality (a : ℝ) : A ∩ B a = A ↔ a ≥ 7 := by sorry

-- Theorem for part (2)
theorem subset_relation (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ a < 1/3 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_subset_relation_l446_44608


namespace NUMINAMATH_CALUDE_segment_ratios_l446_44618

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove the ratios of certain segments. -/
theorem segment_ratios (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 3) (h_QR : R - Q = 7) (h_PS : S - P = 20) :
    (R - P) / (S - Q) = 10 / 17 ∧ (S - P) / (Q - P) = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_segment_ratios_l446_44618


namespace NUMINAMATH_CALUDE_equation_holds_l446_44621

theorem equation_holds (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l446_44621


namespace NUMINAMATH_CALUDE_cakes_served_during_lunch_l446_44625

/-- Given that a restaurant served some cakes for lunch and dinner, with a total of 15 cakes served today and 9 cakes served during dinner, prove that 6 cakes were served during lunch. -/
theorem cakes_served_during_lunch (total_cakes dinner_cakes lunch_cakes : ℕ) 
  (h1 : total_cakes = 15)
  (h2 : dinner_cakes = 9)
  (h3 : total_cakes = lunch_cakes + dinner_cakes) :
  lunch_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_served_during_lunch_l446_44625


namespace NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l446_44633

/-- The length of the path traveled by the center of a quarter-circle when rolled -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / Real.pi) :
  let path_length := 3 * (Real.pi * r / 2)
  path_length = 4.5 := by sorry

end NUMINAMATH_CALUDE_quarter_circle_roll_path_length_l446_44633


namespace NUMINAMATH_CALUDE_shanghai_score_is_75_l446_44613

/-- The score of the Shanghai team in the basketball game -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team in the basketball game -/
def beijing_score : ℕ := shanghai_score - 10

/-- Yao Ming's score in the basketball game -/
def yao_ming_score : ℕ := 30

theorem shanghai_score_is_75 :
  (shanghai_score - beijing_score = 10) ∧
  (shanghai_score + beijing_score = 5 * yao_ming_score - 10) →
  shanghai_score = 75 := by
sorry

end NUMINAMATH_CALUDE_shanghai_score_is_75_l446_44613


namespace NUMINAMATH_CALUDE_second_quadrant_condition_l446_44644

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m - 2) (m + 1)

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

-- State the theorem
theorem second_quadrant_condition (m : ℝ) : 
  in_second_quadrant (z m) ↔ -1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_second_quadrant_condition_l446_44644


namespace NUMINAMATH_CALUDE_chocolate_probability_theorem_not_always_between_probabilities_l446_44671

structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total_pos : total > 0

def probability (box : ChocolateBox) : ℚ :=
  box.white / box.total

theorem chocolate_probability_theorem 
  (box1 box2 : ChocolateBox) :
  ∃ (combined : ChocolateBox),
    probability combined > min (probability box1) (probability box2) ∧
    probability combined < max (probability box1) (probability box2) ∧
    combined.white = box1.white + box2.white ∧
    combined.total = box1.total + box2.total :=
sorry

theorem not_always_between_probabilities 
  (box1 box2 : ChocolateBox) :
  ¬ ∀ (combined : ChocolateBox),
    (combined.white = box1.white + box2.white ∧
     combined.total = box1.total + box2.total) →
    (probability combined > min (probability box1) (probability box2) ∧
     probability combined < max (probability box1) (probability box2)) :=
sorry

end NUMINAMATH_CALUDE_chocolate_probability_theorem_not_always_between_probabilities_l446_44671


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l446_44628

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l446_44628


namespace NUMINAMATH_CALUDE_initial_cards_calculation_l446_44655

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem initial_cards_calculation :
  initial_cards = cards_given_away + cards_left :=
by sorry

end NUMINAMATH_CALUDE_initial_cards_calculation_l446_44655


namespace NUMINAMATH_CALUDE_quadratic_root_difference_squares_l446_44699

theorem quadratic_root_difference_squares (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - b*x₁ + 12 = 0 ∧ x₂^2 - b*x₂ + 12 = 0 ∧ x₁^2 - x₂^2 = 7) → 
  b = 7 ∨ b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_squares_l446_44699


namespace NUMINAMATH_CALUDE_square_min_rotation_l446_44694

/-- A square is a geometric shape with four equal sides and four right angles. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- The minimum rotation angle for a square to coincide with itself. -/
def minRotationAngle (s : Square) : ℝ := 90

/-- Theorem stating that the minimum rotation angle for a square to coincide with itself is 90 degrees. -/
theorem square_min_rotation (s : Square) : minRotationAngle s = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_min_rotation_l446_44694


namespace NUMINAMATH_CALUDE_lanie_hourly_rate_l446_44606

/-- Calculates the hourly rate given the fraction of hours worked, total hours, and weekly salary -/
def hourly_rate (fraction_worked : ℚ) (total_hours : ℕ) (weekly_salary : ℕ) : ℚ :=
  weekly_salary / (fraction_worked * total_hours)

/-- Proves that given the specified conditions, the hourly rate is $15 -/
theorem lanie_hourly_rate :
  let fraction_worked : ℚ := 4/5
  let total_hours : ℕ := 40
  let weekly_salary : ℕ := 480
  hourly_rate fraction_worked total_hours weekly_salary = 15 := by
sorry

end NUMINAMATH_CALUDE_lanie_hourly_rate_l446_44606


namespace NUMINAMATH_CALUDE_complex_exponential_conjugate_l446_44686

theorem complex_exponential_conjugate (α β : ℝ) : 
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_conjugate_l446_44686


namespace NUMINAMATH_CALUDE_drama_club_pets_l446_44660

theorem drama_club_pets (S : Finset ℕ) (R G : Finset ℕ) : 
  Finset.card S = 50 → 
  (∀ s ∈ S, s ∈ R ∨ s ∈ G) → 
  Finset.card R = 35 → 
  Finset.card G = 40 → 
  Finset.card (R ∩ G) = 25 := by
sorry

end NUMINAMATH_CALUDE_drama_club_pets_l446_44660


namespace NUMINAMATH_CALUDE_yella_computer_usage_l446_44674

/-- Yella's computer usage problem -/
theorem yella_computer_usage 
  (last_week_usage : ℕ) 
  (this_week_first_4_days : ℕ) 
  (this_week_last_3_days : ℕ) 
  (next_week_weekday_classes : ℕ) 
  (next_week_weekday_gaming : ℕ) 
  (next_week_weekend_usage : ℕ) : 
  last_week_usage = 91 →
  this_week_first_4_days = 8 →
  this_week_last_3_days = 10 →
  next_week_weekday_classes = 5 →
  next_week_weekday_gaming = 3 →
  next_week_weekend_usage = 12 →
  (last_week_usage - (4 * this_week_first_4_days + 3 * this_week_last_3_days) = 29) ∧
  (last_week_usage - (5 * (next_week_weekday_classes + next_week_weekday_gaming) + 2 * next_week_weekend_usage) = 27) :=
by sorry

end NUMINAMATH_CALUDE_yella_computer_usage_l446_44674


namespace NUMINAMATH_CALUDE_modulus_of_z_l446_44624

/-- The modulus of the complex number z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_z (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l446_44624


namespace NUMINAMATH_CALUDE_right_triangle_circle_intersection_l446_44609

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circle
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point D
def D : ℝ × ℝ := sorry

-- Define the properties of the triangle and circle
def is_right_triangle (t : Triangle) : Prop :=
  sorry

def circle_intersects_BC (t : Triangle) (c : Circle) : Prop :=
  sorry

def AC_is_diameter (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Theorem statement
theorem right_triangle_circle_intersection 
  (t : Triangle) (c : Circle) :
  is_right_triangle t →
  circle_intersects_BC t c →
  AC_is_diameter t c →
  t.A.1 - t.B.1 = 18 →
  t.A.1 - t.C.1 = 30 →
  D.1 - t.B.1 = 14.4 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_intersection_l446_44609


namespace NUMINAMATH_CALUDE_fraction_of_seats_sold_l446_44614

/-- Proves that the fraction of seats sold is 0.75 given the auditorium layout and earnings --/
theorem fraction_of_seats_sold (rows : ℕ) (seats_per_row : ℕ) (ticket_price : ℚ) (total_earnings : ℚ) :
  rows = 20 →
  seats_per_row = 10 →
  ticket_price = 10 →
  total_earnings = 1500 →
  (total_earnings / ticket_price) / (rows * seats_per_row : ℚ) = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_seats_sold_l446_44614


namespace NUMINAMATH_CALUDE_triangle_square_distance_l446_44601

-- Define the triangle ABF
def Triangle (A B F : ℝ × ℝ) : Prop :=
  ∃ (x y z : ℝ), 
    (A.1 - F.1)^2 + (A.2 - F.2)^2 = x^2 ∧
    (B.1 - F.1)^2 + (B.2 - F.2)^2 = y^2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = z^2

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ (s : ℝ),
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = s^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = s^2 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = s^2 ∧
    (A.1 - D.1)^2 + (A.2 - D.2)^2 = s^2

-- Define the circumcenter of a square
def Circumcenter (E A B C D : ℝ × ℝ) : Prop :=
  (E.1 - A.1)^2 + (E.2 - A.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 ∧
  (E.1 - B.1)^2 + (E.2 - B.2)^2 = (E.1 - C.1)^2 + (E.2 - C.2)^2 ∧
  (E.1 - C.1)^2 + (E.2 - C.2)^2 = (E.1 - D.1)^2 + (E.2 - D.2)^2

theorem triangle_square_distance 
  (A B C D E F : ℝ × ℝ)
  (h1 : Triangle A B F)
  (h2 : Square A B C D)
  (h3 : Circumcenter E A B C D)
  (h4 : (A.1 - F.1)^2 + (A.2 - F.2)^2 = 36)
  (h5 : (B.1 - F.1)^2 + (B.2 - F.2)^2 = 64)
  (h6 : (A.1 - B.1) * (F.1 - B.1) + (A.2 - B.2) * (F.2 - B.2) = 0) :
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 98 :=
by sorry

end NUMINAMATH_CALUDE_triangle_square_distance_l446_44601


namespace NUMINAMATH_CALUDE_john_ducks_count_l446_44687

/-- Proves that John bought 30 ducks given the problem conditions -/
theorem john_ducks_count :
  let cost_per_duck : ℕ := 10
  let weight_per_duck : ℕ := 4
  let price_per_pound : ℕ := 5
  let total_profit : ℕ := 300
  let num_ducks : ℕ := (total_profit / (weight_per_duck * price_per_pound - cost_per_duck))
  num_ducks = 30 := by sorry

end NUMINAMATH_CALUDE_john_ducks_count_l446_44687


namespace NUMINAMATH_CALUDE_min_blocks_correct_l446_44634

/-- A list of positive integer weights representing ice blocks -/
def IceBlocks := List Nat

/-- Predicate to check if a list of weights can satisfy any demand (p, q) where p + q ≤ 2016 -/
def CanSatisfyDemand (blocks : IceBlocks) : Prop :=
  ∀ p q : Nat, p + q ≤ 2016 → ∃ (subsetP subsetQ : List Nat),
    subsetP.Disjoint subsetQ ∧
    subsetP.sum = p ∧
    subsetQ.sum = q ∧
    (subsetP ++ subsetQ).Sublist blocks

/-- The minimum number of ice blocks needed -/
def MinBlocks : Nat := 18

/-- Theorem stating that MinBlocks is the minimum number of ice blocks needed -/
theorem min_blocks_correct :
  (∃ (blocks : IceBlocks), blocks.length = MinBlocks ∧ blocks.all (· > 0) ∧ CanSatisfyDemand blocks) ∧
  (∀ (blocks : IceBlocks), blocks.length < MinBlocks → ¬CanSatisfyDemand blocks) := by
  sorry

#check min_blocks_correct

end NUMINAMATH_CALUDE_min_blocks_correct_l446_44634


namespace NUMINAMATH_CALUDE_jack_letters_difference_l446_44610

theorem jack_letters_difference (morning_emails morning_letters afternoon_emails afternoon_letters : ℕ) :
  morning_emails = 6 →
  morning_letters = 8 →
  afternoon_emails = 2 →
  afternoon_letters = 7 →
  morning_letters - afternoon_letters = 1 := by
  sorry

end NUMINAMATH_CALUDE_jack_letters_difference_l446_44610


namespace NUMINAMATH_CALUDE_length_AB_trajectory_C_l446_44654

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 5*y^2 = 5

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ellipse_foci : A = (-2, 0) ∧ B = (2, 0))
  (angle_relation : ∀ (θA θB θC : ℝ), 
    Real.sin θB - Real.sin θA = Real.sin θC → 
    θA + θB + θC = π)

-- Statement 1: Length of AB is 4
theorem length_AB (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = 4 :=
sorry

-- Statement 2: Trajectory of C
theorem trajectory_C (t : Triangle) (x y : ℝ) :
  (∃ (C : ℝ × ℝ), t.C = C ∧ x > 1) →
  (x^2 - y^2/3 = 1) :=
sorry

end NUMINAMATH_CALUDE_length_AB_trajectory_C_l446_44654


namespace NUMINAMATH_CALUDE_negation_square_nonnegative_l446_44678

theorem negation_square_nonnegative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_square_nonnegative_l446_44678


namespace NUMINAMATH_CALUDE_equation_solution_l446_44615

theorem equation_solution (x : ℝ) (h : x * (x - 1) ≠ 0) :
  (x / (x - 1) - 2 / x = 1) ↔ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l446_44615


namespace NUMINAMATH_CALUDE_sin_120_degrees_l446_44631

theorem sin_120_degrees : 
  ∃ (Q : ℝ × ℝ) (E : ℝ × ℝ),
    (Q.1^2 + Q.2^2 = 1) ∧  -- Q is on the unit circle
    (Real.cos (2*π/3) = Q.1 ∧ Real.sin (2*π/3) = Q.2) ∧  -- Q is at 120°
    (E.2 = 0 ∧ (Q.1 - E.1) * (Q.1 - E.1) + Q.2 * Q.2 = (Q.1 - E.1)^2) →  -- E is the foot of perpendicular
    Real.sin (2*π/3) = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l446_44631


namespace NUMINAMATH_CALUDE_present_difference_l446_44677

/-- The number of presents Santana buys for her brothers in a year -/
def presentCount : ℕ → ℕ
| 1 => 3  -- March (first half)
| 2 => 1  -- October (second half)
| 3 => 1  -- November (second half)
| 4 => 2  -- December (second half)
| _ => 0

/-- The total number of brothers Santana has -/
def totalBrothers : ℕ := 7

/-- The number of presents bought in the first half of the year -/
def firstHalfPresents : ℕ := presentCount 1

/-- The number of presents bought in the second half of the year -/
def secondHalfPresents : ℕ := presentCount 2 + presentCount 3 + presentCount 4 + totalBrothers

theorem present_difference : secondHalfPresents - firstHalfPresents = 8 := by
  sorry

end NUMINAMATH_CALUDE_present_difference_l446_44677


namespace NUMINAMATH_CALUDE_calculation_proof_l446_44668

theorem calculation_proof : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l446_44668


namespace NUMINAMATH_CALUDE_food_for_six_days_is_87_l446_44637

/-- Represents the daily food consumption for Joy's foster dogs -/
def daily_food_consumption : ℚ :=
  -- Mom's food
  (1.5 * 3) +
  -- First two puppies
  (2 * (1/2 * 3)) +
  -- Next two puppies
  (2 * (3/4 * 2)) +
  -- Last puppy
  (1 * 4)

/-- The total amount of food needed for 6 days -/
def total_food_for_six_days : ℚ := daily_food_consumption * 6

/-- Theorem stating that the total food needed for 6 days is 87 cups -/
theorem food_for_six_days_is_87 : total_food_for_six_days = 87 := by sorry

end NUMINAMATH_CALUDE_food_for_six_days_is_87_l446_44637


namespace NUMINAMATH_CALUDE_total_jogging_time_l446_44640

-- Define the number of weekdays
def weekdays : ℕ := 5

-- Define the regular jogging time per day in minutes
def regular_time : ℕ := 30

-- Define the extra time jogged on Tuesday in minutes
def extra_tuesday : ℕ := 5

-- Define the extra time jogged on Friday in minutes
def extra_friday : ℕ := 25

-- Define the total jogging time for the week in minutes
def total_time : ℕ := weekdays * regular_time + extra_tuesday + extra_friday

-- Theorem: The total jogging time for the week is equal to 3 hours
theorem total_jogging_time : total_time / 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_total_jogging_time_l446_44640


namespace NUMINAMATH_CALUDE_triangle_side_b_l446_44648

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_b (t : Triangle) 
  (h1 : t.a = Real.sqrt 3)
  (h2 : t.A = 60 * π / 180)
  (h3 : t.C = 75 * π / 180)
  : t.b = Real.sqrt 2 := by
  sorry

-- Note: We use radians for angles in Lean, so we convert degrees to radians

end NUMINAMATH_CALUDE_triangle_side_b_l446_44648


namespace NUMINAMATH_CALUDE_rhombus_area_from_overlapping_strips_l446_44698

/-- The area of a rhombus formed by two overlapping strips -/
theorem rhombus_area_from_overlapping_strips (β : Real) (h : β ≠ 0) : 
  let strip_width : Real := 2
  let diagonal1 : Real := strip_width
  let diagonal2 : Real := strip_width / Real.sin β
  let rhombus_area : Real := (1 / 2) * diagonal1 * diagonal2
  rhombus_area = 2 / Real.sin β :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_from_overlapping_strips_l446_44698


namespace NUMINAMATH_CALUDE__l446_44681

def smallest_angle_theorem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 2) (hc : ‖c‖ = 5) (habc : a + b + c = 0) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π := by sorry

end NUMINAMATH_CALUDE__l446_44681


namespace NUMINAMATH_CALUDE_students_taking_history_l446_44645

theorem students_taking_history 
  (total_students : ℕ) 
  (statistics_students : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total_students = 89)
  (h_statistics : statistics_students = 32)
  (h_history_or_stats : history_or_statistics = 59)
  (h_history_not_stats : history_not_statistics = 27) :
  ∃ history_students : ℕ, history_students = 54 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_history_l446_44645


namespace NUMINAMATH_CALUDE_negative_two_triangle_five_l446_44679

/-- Definition of the triangle operation for rational numbers -/
def triangle (a b : ℚ) : ℚ := a * b + b - a

/-- Theorem stating that (-2) triangle 5 equals -3 -/
theorem negative_two_triangle_five : triangle (-2) 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_triangle_five_l446_44679


namespace NUMINAMATH_CALUDE_average_weight_increase_l446_44604

/-- Proves that replacing a person weighing 45 kg with a person weighing 93 kg
    in a group of 8 people increases the average weight by 6 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let group_size : ℕ := 8
  let old_weight : ℝ := 45
  let new_weight : ℝ := 93
  let weight_difference : ℝ := new_weight - old_weight
  let average_increase : ℝ := weight_difference / group_size
  average_increase = 6 := by
  sorry

#check average_weight_increase

end NUMINAMATH_CALUDE_average_weight_increase_l446_44604


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l446_44603

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₀ := 3
  let r := 3 * x^2
  geometric_sequence a₀ r 4 = 243 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l446_44603


namespace NUMINAMATH_CALUDE_equivalent_statements_l446_44669

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l446_44669
