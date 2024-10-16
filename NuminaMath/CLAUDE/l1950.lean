import Mathlib

namespace NUMINAMATH_CALUDE_marked_angle_is_fifteen_degrees_l1950_195083

-- Define the figure with three squares
structure ThreeSquaresFigure where
  -- Angles are represented in degrees
  angle_C1OC2 : ℝ
  angle_A2OA3 : ℝ
  angle_C3OA1 : ℝ

-- Define the properties of the figure
def is_valid_three_squares_figure (f : ThreeSquaresFigure) : Prop :=
  f.angle_C1OC2 = 30 ∧
  f.angle_A2OA3 = 45 ∧
  -- Additional properties of squares (all right angles)
  -- Assuming O is the center where all squares meet
  ∃ (angle_C1OA1 angle_C2OA2 angle_C3OA3 : ℝ),
    angle_C1OA1 = 90 ∧ angle_C2OA2 = 90 ∧ angle_C3OA3 = 90

-- Theorem statement
theorem marked_angle_is_fifteen_degrees (f : ThreeSquaresFigure) 
  (h : is_valid_three_squares_figure f) : f.angle_C3OA1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marked_angle_is_fifteen_degrees_l1950_195083


namespace NUMINAMATH_CALUDE_solve_cube_equation_l1950_195081

theorem solve_cube_equation : ∃ x : ℝ, (x - 3)^3 = (1/27)⁻¹ ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_cube_equation_l1950_195081


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l1950_195049

theorem fraction_equality_implies_value (a : ℝ) : 
  a / (a + 45) = 0.82 → a = 205 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l1950_195049


namespace NUMINAMATH_CALUDE_smallest_gamma_for_integer_solution_l1950_195064

theorem smallest_gamma_for_integer_solution :
  ∃ (γ : ℕ), γ > 0 ∧
  (∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ) = 4 * Real.sqrt 2)) ∧
  (∀ (γ' : ℕ), 0 < γ' ∧ γ' < γ →
    ¬∃ (x : ℕ), (Real.sqrt x - Real.sqrt (24 * γ') = 4 * Real.sqrt 2)) ∧
  γ = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gamma_for_integer_solution_l1950_195064


namespace NUMINAMATH_CALUDE_dolphin_count_dolphin_count_proof_l1950_195067

theorem dolphin_count : ℕ → Prop :=
  fun total_dolphins =>
    let fully_trained := total_dolphins / 4
    let remaining := total_dolphins - fully_trained
    let in_training := (2 * remaining) / 3
    let untrained := remaining - in_training
    (fully_trained = total_dolphins / 4) ∧
    (in_training = (2 * remaining) / 3) ∧
    (untrained = 5) →
    total_dolphins = 20

-- The proof goes here
theorem dolphin_count_proof : dolphin_count 20 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_count_dolphin_count_proof_l1950_195067


namespace NUMINAMATH_CALUDE_line_equations_l1950_195078

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the point A
def A : Point := (1, -3)

-- Define the reference line
def reference_line : Line := λ x y ↦ 2*x - y + 4

-- Define the properties of lines l and m
def parallel (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y = k * l2 x y
def perpendicular (l1 l2 : Line) : Prop := ∃ k : ℝ, ∀ x y, l1 x y * l2 x y = -k

-- Define the y-intercept of a line
def y_intercept (l : Line) : ℝ := l 0 1

-- State the theorem
theorem line_equations (l m : Line) : 
  (∃ k : ℝ, l A.fst A.snd = 0) →  -- l passes through A
  parallel l reference_line →     -- l is parallel to reference_line
  perpendicular l m →             -- m is perpendicular to l
  y_intercept m = 3 →             -- m has y-intercept 3
  (∀ x y, l x y = 2*x - y - 5) ∧  -- equation of l
  (∀ x y, m x y = x + 2*y - 6)    -- equation of m
  := by sorry

end NUMINAMATH_CALUDE_line_equations_l1950_195078


namespace NUMINAMATH_CALUDE_alloy_density_l1950_195099

theorem alloy_density (gold_density copper_density silver_density alloy_density : ℝ)
  (hg : gold_density = 10)
  (hc : copper_density = 5)
  (hs : silver_density = 7)
  (ha : alloy_density = 9) :
  let gold_parts : ℝ := 6
  let copper_parts : ℝ := 1
  let silver_parts : ℝ := 1
  (gold_density * gold_parts + copper_density * copper_parts + silver_density * silver_parts) /
    (gold_parts + copper_parts + silver_parts) = alloy_density :=
by sorry

end NUMINAMATH_CALUDE_alloy_density_l1950_195099


namespace NUMINAMATH_CALUDE_store_revenue_l1950_195057

theorem store_revenue (december : ℝ) (november january : ℝ)
  (h1 : november = (3/5) * december)
  (h2 : january = (1/3) * november) :
  december = (5/2) * ((november + january) / 2) :=
by sorry

end NUMINAMATH_CALUDE_store_revenue_l1950_195057


namespace NUMINAMATH_CALUDE_y1_gt_y2_l1950_195023

/-- A quadratic function with a positive leading coefficient and symmetric axis at x = 1 -/
structure SymmetricQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  sym_axis : b = -2 * a

/-- The y-coordinate of the quadratic function at a given x -/
def y_coord (q : SymmetricQuadratic) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_gt_y2 (q : SymmetricQuadratic) (y₁ y₂ : ℝ)
  (h1 : y_coord q (-1) = y₁)
  (h2 : y_coord q 2 = y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_gt_y2_l1950_195023


namespace NUMINAMATH_CALUDE_truth_probability_l1950_195052

theorem truth_probability (pA pB pAB : ℝ) : 
  pA = 0.7 →
  pAB = 0.42 →
  pAB = pA * pB →
  pB = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_truth_probability_l1950_195052


namespace NUMINAMATH_CALUDE_simplify_expression_l1950_195080

theorem simplify_expression : (576 : ℝ) ^ (1/4) * (216 : ℝ) ^ (1/2) = 72 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1950_195080


namespace NUMINAMATH_CALUDE_min_value_a_b_squared_l1950_195077

/-- Given that the ratio of the absolute values of the coefficients of x² and x³ terms
    in the expansion of (1/a + ax)⁵ - (1/b + bx)⁵ is 1:6, 
    the minimum value of a² + b² is 12 -/
theorem min_value_a_b_squared (a b : ℝ) (h : ∃ k : ℝ, k > 0 ∧ 
  |5 * (1/a^2 - 1/b^2)| = k ∧ |10 * (a - b)| = 6*k) : 
  a^2 + b^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_b_squared_l1950_195077


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1950_195086

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 4
  let downstream_distance : ℝ := 5.133333333333334
  let downstream_time : ℝ := 14 / 60
  ∃ v : ℝ, v > 0 ∧ (v + current_speed) * downstream_time = downstream_distance ∧ v = 18 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1950_195086


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1950_195096

theorem circle_radius_proof (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 5^2) →
  (A₂ = (A₁ + A₂) / 2) →
  ∃ r : ℝ, r > 0 ∧ A₁ = π * r^2 ∧ r = 5 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1950_195096


namespace NUMINAMATH_CALUDE_average_cost_is_1_85_l1950_195034

/-- Calculates the average cost per fruit given the prices and quantities of fruits, applying special offers --/
def average_cost_per_fruit (apple_price banana_price orange_price : ℚ) 
  (apple_qty banana_qty orange_qty : ℕ) : ℚ :=
  let apple_cost := apple_price * (apple_qty.div 10 * 10)
  let banana_cost := banana_price * banana_qty
  let orange_cost := orange_price * (orange_qty.div 3 * 3)
  let total_cost := apple_cost + banana_cost + orange_cost
  let total_fruits := apple_qty + banana_qty + orange_qty
  total_cost / total_fruits

/-- The average cost per fruit is $1.85 given the specified prices, quantities, and offers --/
theorem average_cost_is_1_85 :
  average_cost_per_fruit 2 1 3 12 4 4 = 37/20 := by
  sorry

end NUMINAMATH_CALUDE_average_cost_is_1_85_l1950_195034


namespace NUMINAMATH_CALUDE_factor_expression_l1950_195048

theorem factor_expression (a b c : ℝ) : 
  ((a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1950_195048


namespace NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1950_195009

theorem smallest_k_for_no_real_roots : 
  ∃ (k : ℤ), k = 1 ∧ 
  (∀ (x : ℝ), (k + 1 : ℝ) * x^2 - (6 * k + 2 : ℝ) * x + (3 * k + 2 : ℝ) ≠ 0) ∧
  (∀ (j : ℤ), j < k → ∃ (x : ℝ), (j + 1 : ℝ) * x^2 - (6 * j + 2 : ℝ) * x + (3 * j + 2 : ℝ) = 0) :=
by sorry

#check smallest_k_for_no_real_roots

end NUMINAMATH_CALUDE_smallest_k_for_no_real_roots_l1950_195009


namespace NUMINAMATH_CALUDE_smallest_z_satisfying_conditions_l1950_195066

theorem smallest_z_satisfying_conditions : ∃ (z : ℕ), z = 10 ∧ 
  (∀ (x y : ℕ), x > 0 ∧ y > 0 →
    (27 ^ z) * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
    x + y = z ∧
    x * y < z ^ 2) ∧
  (∀ (z' : ℕ), z' < z →
    ¬(∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
      (27 ^ z') * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
      x + y = z' ∧
      x * y < z' ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_satisfying_conditions_l1950_195066


namespace NUMINAMATH_CALUDE_water_consumption_correct_l1950_195091

/-- Water consumption per person per year in cubic meters for different regions -/
structure WaterConsumption where
  west : ℝ
  nonWest : ℝ
  russia : ℝ

/-- Given water consumption data -/
def givenData : WaterConsumption :=
  { west := 21428
    nonWest := 26848.55
    russia := 302790.13 }

/-- Theorem stating that the given water consumption data is correct -/
theorem water_consumption_correct (data : WaterConsumption) :
  data.west = givenData.west ∧
  data.nonWest = givenData.nonWest ∧
  data.russia = givenData.russia :=
by sorry

#check water_consumption_correct

end NUMINAMATH_CALUDE_water_consumption_correct_l1950_195091


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1950_195006

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_about_x_axis (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 = x2 ∧ y1 = -y2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_x_axis (a - 1) 5 2 (b - 1) →
  (a + b) ^ 2005 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1950_195006


namespace NUMINAMATH_CALUDE_tracy_candies_l1950_195059

theorem tracy_candies (initial_candies : ℕ) : 
  (∃ (sister_took : ℕ),
    initial_candies > 0 ∧
    sister_took ≥ 2 ∧ 
    sister_took ≤ 6 ∧
    (initial_candies * 3 / 4) * 2 / 3 - 40 - sister_took = 10) →
  initial_candies = 108 :=
by sorry

end NUMINAMATH_CALUDE_tracy_candies_l1950_195059


namespace NUMINAMATH_CALUDE_cos_A_minus_sin_C_range_l1950_195056

theorem cos_A_minus_sin_C_range (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = 2 * b * Real.sin A → -- Given condition
  -Real.sqrt 3 / 2 < Real.cos A - Real.sin C ∧ 
  Real.cos A - Real.sin C < 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_A_minus_sin_C_range_l1950_195056


namespace NUMINAMATH_CALUDE_parabola_sum_l1950_195013

/-- A parabola with coefficients p, q, and r. -/
structure Parabola where
  p : ℚ
  q : ℚ
  r : ℚ

/-- The y-coordinate of a point on the parabola given its x-coordinate. -/
def Parabola.y_coord (para : Parabola) (x : ℚ) : ℚ :=
  para.p * x^2 + para.q * x + para.r

theorem parabola_sum (para : Parabola) 
    (vertex : para.y_coord 3 = -2)
    (point : para.y_coord 6 = 5) :
    para.p + para.q + para.r = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l1950_195013


namespace NUMINAMATH_CALUDE_complex_number_system_l1950_195000

theorem complex_number_system (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 4)
  (h_prod_sum : a * b + b * c + c * a = 4)
  (h_prod : a * b * c = 4) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_system_l1950_195000


namespace NUMINAMATH_CALUDE_world_cup_2006_matches_l1950_195062

/- Define the tournament structure -/
structure WorldCup where
  totalTeams : Nat
  numGroups : Nat
  teamsPerGroup : Nat
  advancingTeams : Nat

/- Define the calculation of total matches -/
def totalMatches (wc : WorldCup) : Nat :=
  let groupStageMatches := wc.numGroups * (wc.teamsPerGroup.choose 2)
  let knockoutMatches := wc.advancingTeams - 1
  groupStageMatches + knockoutMatches

/- Theorem statement -/
theorem world_cup_2006_matches (wc : WorldCup) 
  (h1 : wc.totalTeams = 32)
  (h2 : wc.numGroups = 8)
  (h3 : wc.teamsPerGroup = 4)
  (h4 : wc.advancingTeams = 16) :
  totalMatches wc = 64 := by
  sorry


end NUMINAMATH_CALUDE_world_cup_2006_matches_l1950_195062


namespace NUMINAMATH_CALUDE_pascal_triangle_elements_l1950_195003

/-- The number of elements in a single row of Pascal's Triangle -/
def elementsInRow (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def totalElementsPascal (n : ℕ) : ℕ := triangularNumber n

theorem pascal_triangle_elements :
  totalElementsPascal 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_elements_l1950_195003


namespace NUMINAMATH_CALUDE_not_even_not_odd_composition_l1950_195031

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Statement of the theorem
theorem not_even_not_odd_composition (f : ℝ → ℝ) (c : ℝ) (h : OddFunction f) :
  ¬ (EvenFunction (fun x ↦ f (f (x + c)))) ∧ ¬ (OddFunction (fun x ↦ f (f (x + c)))) :=
sorry

end NUMINAMATH_CALUDE_not_even_not_odd_composition_l1950_195031


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_system_l1950_195082

/-- Given two linear functions and their intersection point, prove that the point satisfies a specific system of equations -/
theorem intersection_point_satisfies_system (a b : ℝ) : 
  (∃ x y : ℝ, y = 3 * x + 6 ∧ y = 2 * x - 4 ∧ x = a ∧ y = b) →
  (3 * a - b = -6 ∧ 2 * a - b - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_system_l1950_195082


namespace NUMINAMATH_CALUDE_barn_paint_area_l1950_195068

/-- Calculates the total area to be painted for a rectangular barn with given dimensions and windows. -/
def total_paint_area (width length height window_width window_height window_count : ℕ) : ℕ :=
  let wall_area_1 := 2 * (width * height)
  let wall_area_2 := 2 * (length * height - window_width * window_height * window_count)
  let ceiling_area := width * length
  2 * (wall_area_1 + wall_area_2) + ceiling_area

/-- The total area to be painted for the given barn is 780 sq yd. -/
theorem barn_paint_area :
  total_paint_area 12 15 6 2 3 2 = 780 :=
by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l1950_195068


namespace NUMINAMATH_CALUDE_square_of_sum_l1950_195012

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_l1950_195012


namespace NUMINAMATH_CALUDE_simplify_expression_l1950_195002

theorem simplify_expression (w : ℝ) : 3*w + 4 - 2*w - 5 + 6*w + 7 - 3*w - 9 = 4*w - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1950_195002


namespace NUMINAMATH_CALUDE_period_of_f_l1950_195039

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

theorem period_of_f (f : ℝ → ℝ) (h : has_property f) : is_periodic f 4 := by
  sorry

end NUMINAMATH_CALUDE_period_of_f_l1950_195039


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l1950_195069

/-- Represents the total number of handshakes at the event -/
def total_handshakes : ℕ := 300

/-- Calculates the number of athlete handshakes given the total number of athletes -/
def athlete_handshakes (n : ℕ) : ℕ := (3 * n * n) / 4

/-- Calculates the number of coach handshakes given the total number of athletes -/
def coach_handshakes (n : ℕ) : ℕ := n

/-- Theorem stating the minimum number of coach handshakes -/
theorem min_coach_handshakes :
  ∃ n : ℕ, 
    athlete_handshakes n + coach_handshakes n = total_handshakes ∧
    coach_handshakes n = 20 ∧
    ∀ m : ℕ, 
      athlete_handshakes m + coach_handshakes m = total_handshakes →
      coach_handshakes m ≥ coach_handshakes n :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l1950_195069


namespace NUMINAMATH_CALUDE_steven_more_peaches_l1950_195054

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 17

/-- The number of apples Steven has -/
def steven_apples : ℕ := 16

/-- Jake has 6 fewer peaches than Steven -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has 8 more apples than Steven -/
def jake_apples : ℕ := steven_apples + 8

/-- Theorem: Steven has 1 more peach than apples -/
theorem steven_more_peaches : steven_peaches - steven_apples = 1 := by
  sorry

end NUMINAMATH_CALUDE_steven_more_peaches_l1950_195054


namespace NUMINAMATH_CALUDE_complex_multiplication_l1950_195008

def i : ℂ := Complex.I

theorem complex_multiplication :
  (6 - 3 * i) * (-7 + 2 * i) = -36 + 33 * i :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1950_195008


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_edge_sum_equality_l1950_195035

/-- 
Given a rectangular parallelepiped with edge lengths a, b, and c,
the sum of the squares of its four space diagonals is equal to
the sum of the squares of all its edges.
-/
theorem parallelepiped_diagonal_edge_sum_equality 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_edge_sum_equality_l1950_195035


namespace NUMINAMATH_CALUDE_bridge_length_l1950_195070

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 90)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  train_speed_kmh * (1000 / 3600) * crossing_time - train_length = 285 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1950_195070


namespace NUMINAMATH_CALUDE_trip_time_difference_l1950_195033

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 240)
  (h2 : distance2 = 420)
  (h3 : speed = 60) :
  distance2 / speed - distance1 / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l1950_195033


namespace NUMINAMATH_CALUDE_percentage_of_b_grades_l1950_195095

def scores : List Nat := [91, 68, 59, 99, 82, 88, 86, 79, 72, 60, 87, 85, 83, 76, 81, 93, 65, 89, 78, 74]

def is_grade_b (score : Nat) : Bool :=
  83 ≤ score ∧ score ≤ 92

def count_grade_b (scores : List Nat) : Nat :=
  scores.filter is_grade_b |>.length

theorem percentage_of_b_grades (scores : List Nat) :
  scores.length = 20 →
  (count_grade_b scores : Rat) / scores.length * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_b_grades_l1950_195095


namespace NUMINAMATH_CALUDE_sin_120_degrees_l1950_195016

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_120_degrees_l1950_195016


namespace NUMINAMATH_CALUDE_distance_to_third_side_l1950_195072

/-- Represents a point inside an equilateral triangle -/
structure PointInTriangle where
  /-- Distance to the first side -/
  d1 : ℝ
  /-- Distance to the second side -/
  d2 : ℝ
  /-- Distance to the third side -/
  d3 : ℝ
  /-- The sum of distances equals the triangle's height -/
  sum_eq_height : d1 + d2 + d3 = 5 * Real.sqrt 3

/-- Theorem: In an equilateral triangle with side length 10, if a point inside
    has distances 1 and 3 to two sides, its distance to the third side is 5√3 - 4 -/
theorem distance_to_third_side
  (P : PointInTriangle)
  (h1 : P.d1 = 1)
  (h2 : P.d2 = 3) :
  P.d3 = 5 * Real.sqrt 3 - 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_third_side_l1950_195072


namespace NUMINAMATH_CALUDE_point_location_implies_coordinate_signs_l1950_195090

/-- A point in a 2D coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is to the right of the y-axis -/
def isRightOfYAxis (p : Point) : Prop := p.x > 0

/-- Predicate to check if a point is below the x-axis -/
def isBelowXAxis (p : Point) : Prop := p.y < 0

/-- Theorem stating that if a point is to the right of the y-axis and below the x-axis,
    then its x-coordinate is positive and y-coordinate is negative -/
theorem point_location_implies_coordinate_signs (p : Point) :
  isRightOfYAxis p → isBelowXAxis p → p.x > 0 ∧ p.y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_location_implies_coordinate_signs_l1950_195090


namespace NUMINAMATH_CALUDE_bank_comparison_l1950_195025

/-- Calculates the annual yield given a quarterly interest rate -/
def annual_yield_quarterly (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate) ^ 4 - 1

/-- Calculates the annual yield given an annual interest rate -/
def annual_yield_annual (annual_rate : ℝ) : ℝ :=
  annual_rate

theorem bank_comparison (bank1_quarterly_rate : ℝ) (bank2_annual_rate : ℝ)
    (h1 : bank1_quarterly_rate = 0.8)
    (h2 : bank2_annual_rate = -9) :
    annual_yield_quarterly bank1_quarterly_rate > annual_yield_annual bank2_annual_rate := by
  sorry

#eval annual_yield_quarterly 0.8
#eval annual_yield_annual (-9)

end NUMINAMATH_CALUDE_bank_comparison_l1950_195025


namespace NUMINAMATH_CALUDE_num_choices_eq_ten_l1950_195092

/-- The number of science subjects -/
def num_science : ℕ := 3

/-- The number of humanities subjects -/
def num_humanities : ℕ := 3

/-- The total number of subjects to choose from -/
def total_subjects : ℕ := num_science + num_humanities

/-- The number of subjects that must be chosen -/
def subjects_to_choose : ℕ := 3

/-- The minimum number of science subjects that must be chosen -/
def min_science : ℕ := 2

/-- The function that calculates the number of ways to choose subjects -/
def num_choices : ℕ := sorry

theorem num_choices_eq_ten : num_choices = 10 := by sorry

end NUMINAMATH_CALUDE_num_choices_eq_ten_l1950_195092


namespace NUMINAMATH_CALUDE_triangle_ratio_range_l1950_195047

theorem triangle_ratio_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Side lengths are positive
  a / Real.sin A = b / Real.sin B ∧  -- Law of sines
  b / Real.sin B = c / Real.sin C ∧  -- Law of sines
  -Real.cos B / Real.cos C = (2 * a + b) / c  -- Given condition
  →
  1 < (a + b) / c ∧ (a + b) / c ≤ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_range_l1950_195047


namespace NUMINAMATH_CALUDE_min_value_xyz_min_value_exact_l1950_195005

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → x^3 * y^2 * z ≤ a^3 * b^2 * c :=
by sorry

theorem min_value_exact (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 1/a + 1/b + 1/c = 9 ∧ a^3 * b^2 * c = 1/46656 :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_min_value_exact_l1950_195005


namespace NUMINAMATH_CALUDE_function_value_at_negative_half_l1950_195021

theorem function_value_at_negative_half (a : ℝ) (f : ℝ → ℝ) :
  0 < a →
  a ≠ 1 →
  (∀ x, f x = a^x) →
  f 2 = 81 →
  f (-1/2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_half_l1950_195021


namespace NUMINAMATH_CALUDE_exactly_two_pairs_exist_l1950_195032

/-- Two lines in the xy-plane -/
structure TwoLines where
  a : ℝ
  d : ℝ

/-- The condition for two lines to be identical -/
def are_identical (l : TwoLines) : Prop :=
  ∀ x y : ℝ, (4 * x + l.a * y + l.d = 0) ↔ (l.d * x - 3 * y + 15 = 0)

/-- The theorem stating that there are exactly two pairs (a, d) satisfying the condition -/
theorem exactly_two_pairs_exist :
  ∃! (s : Finset TwoLines), s.card = 2 ∧ (∀ l ∈ s, are_identical l) ∧
    (∀ l : TwoLines, are_identical l → l ∈ s) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_pairs_exist_l1950_195032


namespace NUMINAMATH_CALUDE_calculate_expression_solve_equation_l1950_195073

-- Problem 1
theorem calculate_expression : -3^2 + 5 * (-8/5) - (-4)^2 / (-8) = -13 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ x : ℚ, (x + 1) / 2 - 2 = x / 4 ∧ x = -4/3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_equation_l1950_195073


namespace NUMINAMATH_CALUDE_fred_total_cents_l1950_195085

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of dimes Fred has -/
def fred_dimes : ℕ := 9

/-- Theorem: Fred's total cents is 90 -/
theorem fred_total_cents : fred_dimes * dime_value = 90 := by
  sorry

end NUMINAMATH_CALUDE_fred_total_cents_l1950_195085


namespace NUMINAMATH_CALUDE_average_scissors_after_changes_l1950_195004

/-- Represents a drawer with scissors and pencils -/
structure Drawer where
  scissors : ℕ
  pencils : ℕ

/-- Calculates the average number of scissors in the drawers -/
def averageScissors (drawers : List Drawer) : ℚ :=
  (drawers.map (·.scissors)).sum / drawers.length

theorem average_scissors_after_changes : 
  let initialDrawers : List Drawer := [
    { scissors := 39, pencils := 22 },
    { scissors := 27, pencils := 54 },
    { scissors := 45, pencils := 33 }
  ]
  let scissorsAdded : List ℕ := [13, 7, 10]
  let finalDrawers := List.zipWith 
    (fun d a => { scissors := d.scissors + a, pencils := d.pencils }) 
    initialDrawers 
    scissorsAdded
  averageScissors finalDrawers = 47 := by
  sorry

end NUMINAMATH_CALUDE_average_scissors_after_changes_l1950_195004


namespace NUMINAMATH_CALUDE_unknown_number_is_nine_l1950_195076

def first_number : ℝ := 4.2

def second_number : ℝ := first_number + 2

def third_number : ℝ := first_number + 4

def unknown_number : ℝ := 9 * first_number - 2 * third_number - 2 * second_number

theorem unknown_number_is_nine : unknown_number = 9 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_is_nine_l1950_195076


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l1950_195026

theorem quadratic_equation_1 (x : ℝ) : 5 * x^2 = 40 * x → x = 0 ∨ x = 8 := by
  sorry

#check quadratic_equation_1

end NUMINAMATH_CALUDE_quadratic_equation_1_l1950_195026


namespace NUMINAMATH_CALUDE_equation_solution_l1950_195055

theorem equation_solution : ∃! x : ℚ, 5 * (x - 4) = 3 * (3 - 3 * x) + 6 ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1950_195055


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1950_195027

theorem cubic_polynomial_integer_root
  (p q : ℚ)
  (h1 : ∃ x : ℝ, x^3 + p*x + q = 0 ∧ x = 2 - Real.sqrt 5)
  (h2 : ∃ n : ℤ, n^3 + p*n + q = 0) :
  ∃ n : ℤ, n^3 + p*n + q = 0 ∧ n = -4 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1950_195027


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l1950_195050

/-- Given two vectors m and n in ℝ², if m + n is perpendicular to m, then the second component of n is -3. -/
theorem vector_perpendicular_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  (m + n) • m = 0 →
  a = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l1950_195050


namespace NUMINAMATH_CALUDE_pumpkins_eaten_by_rabbits_l1950_195038

/-- Represents the number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- Represents the number of pumpkins Sara has left after rabbits ate some -/
def remaining_pumpkins : ℕ := 20

/-- Represents the number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

/-- Theorem stating that the number of pumpkins eaten by rabbits is the difference between
    the initial number of pumpkins and the remaining number of pumpkins -/
theorem pumpkins_eaten_by_rabbits :
  eaten_pumpkins = initial_pumpkins - remaining_pumpkins :=
by
  sorry

end NUMINAMATH_CALUDE_pumpkins_eaten_by_rabbits_l1950_195038


namespace NUMINAMATH_CALUDE_grunters_win_probability_l1950_195071

theorem grunters_win_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 6) :
  p^n = 64/729 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l1950_195071


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l1950_195044

def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def K_count : ℕ := 1
def Br_count : ℕ := 1
def O_count : ℕ := 3

def molecular_weight : ℝ :=
  K_count * atomic_weight_K +
  Br_count * atomic_weight_Br +
  O_count * atomic_weight_O

theorem compound_molecular_weight :
  molecular_weight = 167.00 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l1950_195044


namespace NUMINAMATH_CALUDE_brenda_remaining_mice_l1950_195097

def total_baby_mice : ℕ := 3 * 8

def mice_given_to_robbie : ℕ := total_baby_mice / 6

def mice_sold_to_pet_store : ℕ := 3 * mice_given_to_robbie

def remaining_after_pet_store : ℕ := total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store)

def mice_sold_as_feeder : ℕ := remaining_after_pet_store / 2

theorem brenda_remaining_mice :
  total_baby_mice - (mice_given_to_robbie + mice_sold_to_pet_store + mice_sold_as_feeder) = 4 := by
  sorry

end NUMINAMATH_CALUDE_brenda_remaining_mice_l1950_195097


namespace NUMINAMATH_CALUDE_pyramid_angle_ratio_relationship_l1950_195089

/-- A pyramid with all lateral faces forming the same angle with the base -/
structure Pyramid where
  base_area : ℝ
  lateral_angle : ℝ
  total_to_base_ratio : ℝ

/-- The angle formed by the lateral faces with the base of the pyramid -/
def lateral_angle (p : Pyramid) : ℝ := p.lateral_angle

/-- The ratio of the total surface area to the base area of the pyramid -/
def total_to_base_ratio (p : Pyramid) : ℝ := p.total_to_base_ratio

/-- Theorem stating the relationship between the lateral angle and the total-to-base area ratio -/
theorem pyramid_angle_ratio_relationship (p : Pyramid) :
  lateral_angle p = Real.arccos (4 / (total_to_base_ratio p - 1)) ∧
  total_to_base_ratio p > 5 := by sorry

end NUMINAMATH_CALUDE_pyramid_angle_ratio_relationship_l1950_195089


namespace NUMINAMATH_CALUDE_total_toys_is_15_8_l1950_195084

-- Define the initial number of toys and daily changes
def initial_toys : ℝ := 5.3
def tuesday_remaining_percent : ℝ := 0.605
def tuesday_new_toys : ℝ := 3.6
def wednesday_loss_percent : ℝ := 0.502
def wednesday_new_toys : ℝ := 2.4
def thursday_loss_percent : ℝ := 0.308
def thursday_new_toys : ℝ := 4.5

-- Define the function to calculate the total number of toys
def total_toys : ℝ :=
  let tuesday_toys := initial_toys * tuesday_remaining_percent + tuesday_new_toys
  let wednesday_toys := tuesday_toys * (1 - wednesday_loss_percent) + wednesday_new_toys
  let thursday_toys := wednesday_toys * (1 - thursday_loss_percent) + thursday_new_toys
  let lost_tuesday := initial_toys - initial_toys * tuesday_remaining_percent
  let lost_wednesday := tuesday_toys - tuesday_toys * (1 - wednesday_loss_percent)
  let lost_thursday := wednesday_toys - wednesday_toys * (1 - thursday_loss_percent)
  thursday_toys + lost_tuesday + lost_wednesday + lost_thursday

-- Theorem statement
theorem total_toys_is_15_8 : total_toys = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_total_toys_is_15_8_l1950_195084


namespace NUMINAMATH_CALUDE_find_A_l1950_195079

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def round_down_hundreds (n : ℕ) : ℕ := (n / 100) * 100

theorem find_A (A : ℕ) :
  is_single_digit A →
  is_three_digit (A * 100 + 27) →
  round_down_hundreds (A * 100 + 27) = 200 →
  A = 2 := by sorry

end NUMINAMATH_CALUDE_find_A_l1950_195079


namespace NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l1950_195060

theorem cinnamon_nutmeg_difference :
  let cinnamon : Float := 0.6666666666666666
  let nutmeg : Float := 0.5
  cinnamon - nutmeg = 0.1666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l1950_195060


namespace NUMINAMATH_CALUDE_polynomial_value_l1950_195074

theorem polynomial_value (x : ℝ) (h : (Real.sqrt 3 + 1) * x = Real.sqrt 3 - 1) :
  x^4 - 5*x^3 + 6*x^2 - 5*x + 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l1950_195074


namespace NUMINAMATH_CALUDE_emily_cookies_l1950_195040

theorem emily_cookies (e : ℚ) 
  (total : ℚ) 
  (h1 : total = e + 3*e + 2*(3*e) + 4*(2*(3*e))) 
  (h2 : total = 90) : e = 45/17 := by
  sorry

end NUMINAMATH_CALUDE_emily_cookies_l1950_195040


namespace NUMINAMATH_CALUDE_only_valid_pair_l1950_195063

-- Define the animals
inductive Animal : Type
  | Lion : Animal
  | Tiger : Animal
  | Leopard : Animal
  | Elephant : Animal

-- Define a pair of animals
def AnimalPair := (Animal × Animal)

-- Define the conditions
def ValidPair (pair : AnimalPair) : Prop :=
  let (a1, a2) := pair
  -- Two different animals are sent
  (a1 ≠ a2) ∧
  -- If Lion is sent, Tiger must be sent
  ((a1 = Animal.Lion ∨ a2 = Animal.Lion) → (a1 = Animal.Tiger ∨ a2 = Animal.Tiger)) ∧
  -- If Leopard is not sent, Tiger cannot be sent
  ((a1 ≠ Animal.Leopard ∧ a2 ≠ Animal.Leopard) → (a1 ≠ Animal.Tiger ∧ a2 ≠ Animal.Tiger)) ∧
  -- If Leopard participates, Elephant is not sent
  ((a1 = Animal.Leopard ∨ a2 = Animal.Leopard) → (a1 ≠ Animal.Elephant ∧ a2 ≠ Animal.Elephant))

-- Theorem: The only valid pair is Tiger and Leopard
theorem only_valid_pair :
  ∀ (pair : AnimalPair), ValidPair pair ↔ pair = (Animal.Tiger, Animal.Leopard) ∨ pair = (Animal.Leopard, Animal.Tiger) :=
by sorry

end NUMINAMATH_CALUDE_only_valid_pair_l1950_195063


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_equals_three_l1950_195075

-- Define the function f(x) = ax³ + bx - 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 1

-- Define the derivative of f
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

theorem tangent_line_implies_b_minus_a_equals_three (a b : ℝ) :
  f_derivative a b 1 = 1 ∧ f a b 1 = 1 → b - a = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_b_minus_a_equals_three_l1950_195075


namespace NUMINAMATH_CALUDE_mean_proportional_of_segments_l1950_195045

theorem mean_proportional_of_segments (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∃ c : ℝ, c ^ 2 = a * b ∧ c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_of_segments_l1950_195045


namespace NUMINAMATH_CALUDE_angle_equality_in_right_triangle_l1950_195020

theorem angle_equality_in_right_triangle (D E F : Real) (angle_D angle_E angle_3 angle_4 : Real) :
  angle_E = 90 →
  angle_D = 70 →
  angle_3 = angle_4 →
  angle_3 + angle_4 = 180 - angle_E - angle_D →
  angle_4 = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_in_right_triangle_l1950_195020


namespace NUMINAMATH_CALUDE_average_and_difference_l1950_195058

theorem average_and_difference (y : ℝ) : 
  (35 + y) / 2 = 44 → |35 - y| = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l1950_195058


namespace NUMINAMATH_CALUDE_james_muffins_count_l1950_195029

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The factor by which James baked more muffins than Arthur -/
def james_factor : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_factor

theorem james_muffins_count : james_muffins = 1380 := by
  sorry

end NUMINAMATH_CALUDE_james_muffins_count_l1950_195029


namespace NUMINAMATH_CALUDE_meals_sold_equals_twelve_l1950_195061

/-- Represents the number of meals sold during lunch in a restaurant -/
def meals_sold_during_lunch (lunch_meals : ℕ) (dinner_prep : ℕ) (dinner_available : ℕ) : ℕ :=
  lunch_meals + dinner_prep - dinner_available

/-- Theorem stating that the number of meals sold during lunch is 12 -/
theorem meals_sold_equals_twelve : 
  meals_sold_during_lunch 17 5 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meals_sold_equals_twelve_l1950_195061


namespace NUMINAMATH_CALUDE_exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l1950_195010

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The sample space of all possible outcomes when drawing two balls -/
def sampleSpace : Finset Outcome :=
  sorry

/-- The event of having exactly one black ball -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event of having exactly two red balls -/
def exactlyTwoRed (outcome : Outcome) : Prop :=
  outcome.first = Color.Red ∧ outcome.second = Color.Red

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are complementary if one of them always occurs -/
def complementary (e1 e2 : Outcome → Prop) : Prop :=
  ∀ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoRed ∧
  ¬complementary exactlyOneBlack exactlyTwoRed :=
sorry

end NUMINAMATH_CALUDE_exactly_one_black_and_exactly_two_red_mutually_exclusive_but_not_complementary_l1950_195010


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1950_195015

theorem max_rectangle_area (perimeter : ℕ) (h_perimeter : perimeter = 156) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1521 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1950_195015


namespace NUMINAMATH_CALUDE_worm_pages_in_four_volumes_l1950_195017

/-- Represents a collection of book volumes -/
structure BookCollection where
  num_volumes : ℕ
  pages_per_volume : ℕ

/-- Calculates the number of pages a worm burrows through in a book collection -/
def worm_burrowed_pages (books : BookCollection) : ℕ :=
  (books.num_volumes - 2) * books.pages_per_volume

/-- Theorem stating the number of pages a worm burrows through in a specific book collection -/
theorem worm_pages_in_four_volumes :
  let books : BookCollection := ⟨4, 200⟩
  worm_burrowed_pages books = 400 := by sorry

end NUMINAMATH_CALUDE_worm_pages_in_four_volumes_l1950_195017


namespace NUMINAMATH_CALUDE_framed_painting_ratio_l1950_195041

theorem framed_painting_ratio : 
  let painting_width : ℝ := 20
  let painting_height : ℝ := 30
  let frame_side_width : ℝ := 2  -- This is the solution, but we don't use it in the statement
  let frame_top_bottom_width := 3 * frame_side_width
  let framed_width := painting_width + 2 * frame_side_width
  let framed_height := painting_height + 2 * frame_top_bottom_width
  (framed_width * framed_height - painting_width * painting_height = painting_width * painting_height) →
  (min framed_width framed_height) / (max framed_width framed_height) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_framed_painting_ratio_l1950_195041


namespace NUMINAMATH_CALUDE_second_train_speed_l1950_195042

/-- Proves that the speed of the second train is 16 km/hr given the problem conditions -/
theorem second_train_speed (speed1 : ℝ) (total_distance : ℝ) (distance_difference : ℝ) :
  speed1 = 20 →
  total_distance = 450 →
  distance_difference = 50 →
  ∃ (speed2 : ℝ) (time : ℝ),
    speed2 > 0 ∧
    time > 0 ∧
    speed1 * time = speed2 * time + distance_difference ∧
    speed1 * time + speed2 * time = total_distance ∧
    speed2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l1950_195042


namespace NUMINAMATH_CALUDE_fraction_simplification_l1950_195051

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 8/12) :
  (6*x + 8*y) / (48*x*y) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1950_195051


namespace NUMINAMATH_CALUDE_graces_nickels_l1950_195028

theorem graces_nickels (dimes : ℕ) (nickels : ℕ) : 
  dimes = 10 →
  dimes * 10 + nickels * 5 = 150 →
  nickels = 10 := by
sorry

end NUMINAMATH_CALUDE_graces_nickels_l1950_195028


namespace NUMINAMATH_CALUDE_three_digit_sum_divisible_by_11_l1950_195014

theorem three_digit_sum_divisible_by_11 (a b : ℕ) : 
  (400 + 10*a + 3) + 984 = 1300 + 10*b + 7 →
  (1300 + 10*b + 7) % 11 = 0 →
  a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_divisible_by_11_l1950_195014


namespace NUMINAMATH_CALUDE_quadratic_sequence_l1950_195087

/-- Given a quadratic equation with real roots and a specific condition, 
    prove the relation between consecutive terms and the geometric nature of a derived sequence. -/
theorem quadratic_sequence (n : ℕ+) (a : ℕ+ → ℝ) (α β : ℝ) 
  (h1 : a n * α^2 - a (n + 1) * α + 1 = 0)
  (h2 : a n * β^2 - a (n + 1) * β + 1 = 0)
  (h3 : 6 * α - 2 * α * β + 6 * β = 3) :
  (∀ m : ℕ+, a (m + 1) = 1/2 * a m + 1/3) ∧ 
  (∃ r : ℝ, ∀ m : ℕ+, a (m + 1) - 2/3 = r * (a m - 2/3)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_l1950_195087


namespace NUMINAMATH_CALUDE_total_weight_juvenile_female_muscovy_l1950_195036

/-- Given a pond with ducks, calculate the total weight of juvenile female Muscovy ducks -/
theorem total_weight_juvenile_female_muscovy (total_ducks : ℕ) 
  (muscovy_percentage mallard_percentage : ℚ)
  (female_muscovy_percentage : ℚ) 
  (juvenile_female_muscovy_percentage : ℚ)
  (avg_weight_juvenile_female_muscovy : ℚ) :
  total_ducks = 120 →
  muscovy_percentage = 45/100 →
  mallard_percentage = 35/100 →
  female_muscovy_percentage = 60/100 →
  juvenile_female_muscovy_percentage = 30/100 →
  avg_weight_juvenile_female_muscovy = 7/2 →
  ∃ (weight : ℚ), weight = 63/2 ∧ 
    weight = (total_ducks : ℚ) * muscovy_percentage * female_muscovy_percentage * 
             juvenile_female_muscovy_percentage * avg_weight_juvenile_female_muscovy :=
by sorry

end NUMINAMATH_CALUDE_total_weight_juvenile_female_muscovy_l1950_195036


namespace NUMINAMATH_CALUDE_max_value_abc_max_value_abc_attained_l1950_195001

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  a^3 * b^2 * c^2 ≤ 432 / 7^7 := by
  sorry

theorem max_value_abc_attained (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 1 ∧ a₀^3 * b₀^2 * c₀^2 = 432 / 7^7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_max_value_abc_attained_l1950_195001


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l1950_195022

/-- The number of candies Bobby eats per day from Monday through Friday -/
def daily_candies : ℕ := 2

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of weeks it takes Bobby to finish the packets -/
def num_weeks : ℕ := 3

/-- The number of candies Bobby eats on weekend days -/
def weekend_candies : ℕ := 1

theorem bobby_candy_consumption :
  daily_candies * 5 * num_weeks + weekend_candies * 2 * num_weeks = num_packets * candies_per_packet :=
sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l1950_195022


namespace NUMINAMATH_CALUDE_yuri_puppies_count_l1950_195093

/-- The number of puppies Yuri adopted in the first week -/
def first_week : ℕ := 20

/-- The number of puppies Yuri adopted in the second week -/
def second_week : ℕ := (2 * first_week) / 5

/-- The number of puppies Yuri adopted in the third week -/
def third_week : ℕ := (3 * second_week) / 8

/-- The number of puppies Yuri adopted in the fourth week -/
def fourth_week : ℕ := 2 * second_week

/-- The number of puppies Yuri adopted in the fifth week -/
def fifth_week : ℕ := first_week + 10

/-- The number of puppies Yuri adopted in the sixth week -/
def sixth_week : ℕ := 2 * third_week - 5

/-- The number of puppies Yuri adopted in the seventh week -/
def seventh_week : ℕ := 2 * sixth_week

/-- The number of puppies Yuri adopted in half of the eighth week -/
def eighth_week_half : ℕ := (5 * seventh_week) / 6

/-- The total number of puppies Yuri adopted -/
def total_puppies : ℕ := first_week + second_week + third_week + fourth_week + 
                         fifth_week + sixth_week + seventh_week + eighth_week_half

theorem yuri_puppies_count : total_puppies = 81 := by
  sorry

end NUMINAMATH_CALUDE_yuri_puppies_count_l1950_195093


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_main_theorem_l1950_195007

theorem geometric_sum_remainder (n : ℕ) (a r : ℤ) (m : ℕ) (h : m > 0) :
  (a * (r^n - 1) / (r - 1)) % m = ((a * (r^n % m - 1)) / (r - 1)) % m :=
sorry

theorem main_theorem :
  (((3^1005 - 1) / 2) : ℤ) % 500 = 121 :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_main_theorem_l1950_195007


namespace NUMINAMATH_CALUDE_problem_solution_l1950_195019

theorem problem_solution (a : ℝ) (f g : ℝ → ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, f x = x^2 + 9)
  (h3 : ∀ x, g x = x^2 - 3)
  (h4 : f (g a) = 9) :
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1950_195019


namespace NUMINAMATH_CALUDE_fred_paper_count_l1950_195065

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets + received_sheets - given_sheets =
  initial_sheets + received_sheets - given_sheets :=
by sorry

end NUMINAMATH_CALUDE_fred_paper_count_l1950_195065


namespace NUMINAMATH_CALUDE_calculation_proof_l1950_195098

theorem calculation_proof : (1000 : ℤ) * 7 / 10 * 17 * (5^2) = 297500 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1950_195098


namespace NUMINAMATH_CALUDE_max_m_value_l1950_195053

def f (x : ℝ) : ℝ := |x + 1| + |1 - 2*x|

theorem max_m_value (a b : ℝ) (h1 : 0 < b) (h2 : b < 1/2) (h3 : 1/2 < a) 
  (h4 : f a = 3 * f b) :
  ∃ m : ℤ, (∀ n : ℤ, a^2 + b^2 > ↑n → n ≤ m) ∧ a^2 + b^2 > ↑m :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l1950_195053


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1950_195011

def U : Finset Nat := {1, 2, 3, 4, 5}
def M : Finset Nat := {1, 4, 5}
def N : Finset Nat := {1, 3}

theorem intersection_with_complement : M ∩ (U \ N) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1950_195011


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1950_195094

theorem solve_linear_equation (n : ℚ) (h : 2 * n + 5 = 16) : 2 * n - 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1950_195094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1950_195037

theorem arithmetic_sequence_term_count 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ) 
  (h1 : a = 7) 
  (h2 : d = 2) 
  (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : 
  n = 70 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1950_195037


namespace NUMINAMATH_CALUDE_no_linear_factor_l1950_195088

/-- The polynomial p(x, y, z) = x^2 - y^2 + z^2 - 2yz + 2x - 3y + z -/
def p (x y z : ℤ) : ℤ := x^2 - y^2 + z^2 - 2*y*z + 2*x - 3*y + z

/-- Theorem stating that p(x, y, z) cannot be factored with a linear integer factor -/
theorem no_linear_factor :
  ¬ ∃ (a b c d : ℤ) (q : ℤ → ℤ → ℤ → ℤ),
    ∀ x y z, p x y z = (a*x + b*y + c*z + d) * q x y z :=
by sorry

end NUMINAMATH_CALUDE_no_linear_factor_l1950_195088


namespace NUMINAMATH_CALUDE_tylers_age_l1950_195018

theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  T + B + S = 25 → 
  S = B + 1 → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l1950_195018


namespace NUMINAMATH_CALUDE_cube_root_sum_equals_five_l1950_195030

theorem cube_root_sum_equals_five :
  (Real.rpow (25 + 10 * Real.sqrt 5) (1/3 : ℝ)) + (Real.rpow (25 - 10 * Real.sqrt 5) (1/3 : ℝ)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_sum_equals_five_l1950_195030


namespace NUMINAMATH_CALUDE_house_numbering_proof_l1950_195043

theorem house_numbering_proof :
  (2 * 169^2 - 1 = 239^2) ∧ (2 * (288^2 + 288) = 408^2) := by
  sorry

end NUMINAMATH_CALUDE_house_numbering_proof_l1950_195043


namespace NUMINAMATH_CALUDE_not_all_trihedral_angles_form_equilateral_triangles_l1950_195046

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- Represents a plane intersecting a trihedral angle -/
structure Intersection where
  angle : TrihedralAngle
  plane : Unit  -- We don't need to define the plane explicitly for this problem

/-- Predicate to check if an intersection forms an equilateral triangle -/
def forms_equilateral_triangle (i : Intersection) : Prop :=
  -- This would involve complex geometric calculations in reality
  sorry

/-- Theorem stating that not all trihedral angles can be intersected to form equilateral triangles -/
theorem not_all_trihedral_angles_form_equilateral_triangles :
  ∃ (t : TrihedralAngle), ∀ (p : Unit), ¬(forms_equilateral_triangle ⟨t, p⟩) :=
sorry

end NUMINAMATH_CALUDE_not_all_trihedral_angles_form_equilateral_triangles_l1950_195046


namespace NUMINAMATH_CALUDE_ending_number_proof_l1950_195024

theorem ending_number_proof (n : ℕ) : 
  (n > 100) ∧ 
  (∃ (count : ℕ), count = 33 ∧ 
    (∀ k : ℕ, 100 < k ∧ k ≤ n ∧ k % 3 = 0 → 
      ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i)) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ (count : ℕ), count = 33 ∧ 
      (∀ k : ℕ, 100 < k ∧ k ≤ m ∧ k % 3 = 0 → 
        ∃ i : ℕ, i ≤ count ∧ k = 100 + 3 * i))) →
  n = 198 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l1950_195024
