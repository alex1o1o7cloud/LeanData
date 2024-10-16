import Mathlib

namespace NUMINAMATH_CALUDE_nephews_count_l794_79448

/-- The number of nephews Alden and Vihaan have altogether -/
def total_nephews (alden_past : ℕ) (increase : ℕ) : ℕ :=
  let alden_current := 2 * alden_past
  let vihaan := alden_current + increase
  alden_current + vihaan

/-- Theorem stating the total number of nephews Alden and Vihaan have -/
theorem nephews_count : total_nephews 50 60 = 260 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l794_79448


namespace NUMINAMATH_CALUDE_f_of_five_equals_62_l794_79435

/-- Given a function f(x) = 2x^2 + y where f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_62 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
sorry

end NUMINAMATH_CALUDE_f_of_five_equals_62_l794_79435


namespace NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l794_79422

/-- The number of sides in the regular polygon -/
def n : ℕ := 21

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle (in degrees) for which a regular n-gon has rotational symmetry -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 21-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is equal to 38 -/
theorem regular_21gon_symmetry_sum :
  (L n : ℚ) + R n = 38 := by sorry

end NUMINAMATH_CALUDE_regular_21gon_symmetry_sum_l794_79422


namespace NUMINAMATH_CALUDE_binomial_probability_ge_two_l794_79414

/-- A random variable following a Binomial distribution B(10, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function of ξ -/
def pmf (k : ℕ) : ℝ := sorry

/-- The cumulative distribution function of ξ -/
def cdf (k : ℕ) : ℝ := sorry

theorem binomial_probability_ge_two :
  (1 - cdf 1) = 1013 / 1024 :=
sorry

end NUMINAMATH_CALUDE_binomial_probability_ge_two_l794_79414


namespace NUMINAMATH_CALUDE_floor_sum_abcd_l794_79439

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2500) (h2 : c^2 + d^2 = 2500) (h3 : a*c + b*d = 1500) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_abcd_l794_79439


namespace NUMINAMATH_CALUDE_number_division_property_l794_79454

theorem number_division_property : ∃ (n : ℕ), 
  let sum := 2468 + 1375
  let diff := 2468 - 1375
  n = 12609027 ∧
  n / sum = 3 * diff ∧
  n % sum = 150 ∧
  (n - 150) / sum = 5 * diff :=
by sorry

end NUMINAMATH_CALUDE_number_division_property_l794_79454


namespace NUMINAMATH_CALUDE_max_triangle_area_l794_79417

/-- The ellipse E defined by x²/13 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 13 + p.2^2 / 4 = 1}

/-- The left focus F₁ of the ellipse -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus F₂ of the ellipse -/
def F₂ : ℝ × ℝ := sorry

/-- A point P on the ellipse, not coinciding with left and right vertices -/
def P : ℝ × ℝ := sorry

/-- The area of triangle F₂PF₁ -/
def triangleArea (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating that the maximum area of triangle F₂PF₁ is 6 -/
theorem max_triangle_area :
  ∀ p ∈ Ellipse, p ≠ F₁ ∧ p ≠ F₂ → triangleArea p ≤ 6 ∧ ∃ q ∈ Ellipse, triangleArea q = 6 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l794_79417


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l794_79481

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 1 -/
def reflect_line_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The composition of two reflections -/
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line_y_eq_x_minus_1 (reflect_y_axis p)

theorem double_reflection_of_D :
  double_reflection (7, 0) = (1, -8) := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l794_79481


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l794_79411

def i : ℂ := Complex.I

theorem sum_of_powers_of_i : i^14760 + i^14761 + i^14762 + i^14763 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l794_79411


namespace NUMINAMATH_CALUDE_distance_to_y_axis_reflection_l794_79412

/-- The distance between a point and its reflection over the y-axis -/
theorem distance_to_y_axis_reflection (a b : ℝ) : 
  Real.sqrt (((-a) - a)^2 + (b - b)^2) = 2 * |a| := by
  sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_reflection_l794_79412


namespace NUMINAMATH_CALUDE_rectangle_area_l794_79413

/-- Rectangle ABCD with point E on AB and point F on AC -/
structure RectangleConfig where
  /-- Length of side AD -/
  a : ℝ
  /-- Point A -/
  A : ℝ × ℝ
  /-- Point B -/
  B : ℝ × ℝ
  /-- Point C -/
  C : ℝ × ℝ
  /-- Point D -/
  D : ℝ × ℝ
  /-- Point E -/
  E : ℝ × ℝ
  /-- Point F -/
  F : ℝ × ℝ
  /-- ABCD is a rectangle -/
  is_rectangle : A.1 = D.1 ∧ A.2 = B.2 ∧ B.1 = C.1 ∧ C.2 = D.2
  /-- AB = 2 × AD -/
  ab_twice_ad : B.1 - A.1 = 2 * a
  /-- E is the midpoint of AB -/
  e_midpoint : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  /-- F is on AC -/
  f_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))
  /-- F is on DE -/
  f_on_de : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F = (D.1 + s * (E.1 - D.1), D.2 + s * (E.2 - D.2))
  /-- Area of quadrilateral BFED is 50 -/
  area_bfed : abs ((B.1 - F.1) * (E.2 - D.2) - (B.2 - F.2) * (E.1 - D.1)) / 2 = 50

/-- The area of rectangle ABCD is 300 -/
theorem rectangle_area (config : RectangleConfig) : (config.B.1 - config.A.1) * (config.B.2 - config.D.2) = 300 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l794_79413


namespace NUMINAMATH_CALUDE_passengers_per_bus_l794_79406

def total_people : ℕ := 1230
def num_buses : ℕ := 26

theorem passengers_per_bus :
  (total_people / num_buses : ℕ) = 47 := by sorry

end NUMINAMATH_CALUDE_passengers_per_bus_l794_79406


namespace NUMINAMATH_CALUDE_score_difference_is_1_25_l794_79409

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.20, 75),
  (0.25, 85),
  (0.30, 95),
  (0.10, 100)
]

-- Calculate the mean score
def mean_score : ℝ := 
  (score_distribution.map (λ p => p.1 * p.2)).sum

-- Define the median score
def median_score : ℝ := 85

-- Theorem statement
theorem score_difference_is_1_25 : 
  median_score - mean_score = 1.25 := by sorry

end NUMINAMATH_CALUDE_score_difference_is_1_25_l794_79409


namespace NUMINAMATH_CALUDE_oil_ratio_proof_l794_79434

theorem oil_ratio_proof (small_tank_capacity large_tank_capacity initial_large_tank_oil additional_oil_needed : ℕ) 
  (h1 : small_tank_capacity = 4000)
  (h2 : large_tank_capacity = 20000)
  (h3 : initial_large_tank_oil = 3000)
  (h4 : additional_oil_needed = 4000)
  (h5 : initial_large_tank_oil + (small_tank_capacity - (small_tank_capacity - x)) + additional_oil_needed = large_tank_capacity / 2)
  : (small_tank_capacity - (small_tank_capacity - x)) / small_tank_capacity = 3 / 4 := by
  sorry

#check oil_ratio_proof

end NUMINAMATH_CALUDE_oil_ratio_proof_l794_79434


namespace NUMINAMATH_CALUDE_initial_workers_count_l794_79499

/-- Represents the work done in digging a hole -/
def work (workers : ℕ) (hours : ℕ) (depth : ℕ) : ℕ := workers * hours * depth

theorem initial_workers_count :
  ∀ (W : ℕ),
  (∃ (k : ℕ), k > 0 ∧
    work W 8 30 = k * 30 ∧
    work (W + 35) 6 40 = k * 40) →
  W = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_workers_count_l794_79499


namespace NUMINAMATH_CALUDE_female_salmon_count_l794_79486

theorem female_salmon_count (male_salmon : ℕ) (total_salmon : ℕ) 
  (h1 : male_salmon = 712261)
  (h2 : total_salmon = 971639) :
  total_salmon - male_salmon = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l794_79486


namespace NUMINAMATH_CALUDE_square_sum_problem_l794_79472

theorem square_sum_problem (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) :
  a^2 + b^2 = 40 := by sorry

end NUMINAMATH_CALUDE_square_sum_problem_l794_79472


namespace NUMINAMATH_CALUDE_number_of_walls_proof_correct_l794_79489

/-- Proves that the number of walls in a room is 5, given specific conditions about wall size and painting time. -/
theorem number_of_walls (wall_width : ℝ) (wall_height : ℝ) (painting_rate : ℝ) (total_time : ℝ) (spare_time : ℝ) : ℕ :=
  by
  -- Define the given conditions
  have h1 : wall_width = 2 := by sorry
  have h2 : wall_height = 3 := by sorry
  have h3 : painting_rate = 1 / 10 := by sorry  -- 1 square meter per 10 minutes
  have h4 : total_time = 10 := by sorry  -- 10 hours
  have h5 : spare_time = 5 := by sorry  -- 5 hours

  -- Calculate the number of walls
  let wall_area := wall_width * wall_height
  let available_time := total_time - spare_time
  let paintable_area := available_time * 60 * painting_rate  -- Convert hours to minutes
  let number_of_walls := ⌊paintable_area / wall_area⌋  -- Floor division

  -- Prove that the number of walls is 5
  sorry

/-- The number of walls in the room -/
def solution : ℕ := 5

/-- Proves that the calculated number of walls matches the solution -/
theorem proof_correct : number_of_walls 2 3 (1/10) 10 5 = solution := by sorry

end NUMINAMATH_CALUDE_number_of_walls_proof_correct_l794_79489


namespace NUMINAMATH_CALUDE_prob_one_red_ball_eq_one_third_l794_79476

/-- The probability of drawing exactly one red ball from a bag -/
def prob_one_red_ball (red_balls black_balls : ℕ) : ℚ :=
  red_balls / (red_balls + black_balls)

/-- Theorem: The probability of drawing exactly one red ball from a bag
    containing 2 red balls and 4 black balls is 1/3 -/
theorem prob_one_red_ball_eq_one_third :
  prob_one_red_ball 2 4 = 1/3 := by sorry

end NUMINAMATH_CALUDE_prob_one_red_ball_eq_one_third_l794_79476


namespace NUMINAMATH_CALUDE_g_expression_l794_79438

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the relationship between f and g
def g_relation (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_expression (g : ℝ → ℝ) (h : g_relation g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l794_79438


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l794_79487

theorem consecutive_integers_problem (n : ℕ) (avg : ℚ) (max : ℕ) 
  (h_consecutive : ∃ (start : ℤ), ∀ i : ℕ, i < n → start + i ∈ (Set.range (fun i => start + i) : Set ℤ))
  (h_average : avg = (↑(n * (2 * max - n + 1)) / (2 * n) : ℚ))
  (h_max : max = 36)
  (h_avg : avg = 33) :
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l794_79487


namespace NUMINAMATH_CALUDE_school_bus_problem_l794_79428

theorem school_bus_problem (total_distance bus_speed walking_speed rest_time : ℝ) 
  (h_total : total_distance = 21)
  (h_bus : bus_speed = 60)
  (h_walk : walking_speed = 4)
  (h_rest : rest_time = 1/6) :
  ∃ (x : ℝ), 
    (x = 19 ∧ total_distance - x = 2) ∧ 
    ((total_distance - x) / walking_speed + rest_time = (total_distance + x) / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_school_bus_problem_l794_79428


namespace NUMINAMATH_CALUDE_bookstore_sales_l794_79401

/-- Calculates the number of bookmarks sold given the number of books sold and the ratio of books to bookmarks. -/
def bookmarks_sold (books : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) : ℕ :=
  (books * bookmark_ratio) / book_ratio

/-- Theorem stating that given 72 books sold and a 9:2 ratio of books to bookmarks, 16 bookmarks were sold. -/
theorem bookstore_sales : bookmarks_sold 72 9 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_l794_79401


namespace NUMINAMATH_CALUDE_pencil_cost_l794_79456

/-- The cost of a pencil given initial and remaining amounts -/
theorem pencil_cost (initial : ℕ) (remaining : ℕ) (h : initial = 15 ∧ remaining = 4) :
  initial - remaining = 11 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l794_79456


namespace NUMINAMATH_CALUDE_fencing_cost_is_5300_l794_79408

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (length width fenceCostPerMeter : ℝ) : ℝ :=
  2 * (length + width) * fenceCostPerMeter

/-- Theorem: The total cost of fencing the given rectangular plot is $5300 -/
theorem fencing_cost_is_5300 :
  let length : ℝ := 70
  let width : ℝ := 30
  let fenceCostPerMeter : ℝ := 26.50
  totalFencingCost length width fenceCostPerMeter = 5300 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_5300_l794_79408


namespace NUMINAMATH_CALUDE_smallest_angle_in_ratio_quadrilateral_l794_79466

/-- 
Given a quadrilateral where the measures of interior angles are in the ratio 1:2:3:4,
prove that the measure of the smallest interior angle is 36°.
-/
theorem smallest_angle_in_ratio_quadrilateral : 
  ∀ (a b c d : ℝ),
  a > 0 → b > 0 → c > 0 → d > 0 →
  b = 2*a → c = 3*a → d = 4*a →
  a + b + c + d = 360 →
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_ratio_quadrilateral_l794_79466


namespace NUMINAMATH_CALUDE_park_outer_diameter_l794_79474

/-- Represents the diameter of the outer boundary of a circular park with concentric sections -/
def outer_diameter (fountain_diameter : ℝ) (garden_width : ℝ) (path_width : ℝ) : ℝ :=
  fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating the diameter of the outer boundary of the jogging path -/
theorem park_outer_diameter :
  outer_diameter 14 12 10 = 58 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l794_79474


namespace NUMINAMATH_CALUDE_no_valid_covering_exists_l794_79468

/-- Represents a unit square in the chain --/
structure UnitSquare where
  id : Nat
  left_neighbor : Option Nat
  right_neighbor : Option Nat

/-- Represents the chain of squares --/
def SquareChain := List UnitSquare

/-- Represents a vertex on the cube --/
structure CubeVertex where
  x : Fin 4
  y : Fin 4
  z : Fin 4

/-- Represents the 3x3x3 cube --/
def Cube := Set CubeVertex

/-- A covering is a mapping from squares to positions on the cube surface --/
def Covering := UnitSquare → Option CubeVertex

/-- Checks if a covering is valid according to the problem constraints --/
def is_valid_covering (chain : SquareChain) (cube : Cube) (covering : Covering) : Prop :=
  sorry

/-- The main theorem stating that no valid covering exists --/
theorem no_valid_covering_exists (chain : SquareChain) (cube : Cube) :
  chain.length = 54 → ¬∃ (covering : Covering), is_valid_covering chain cube covering :=
sorry

end NUMINAMATH_CALUDE_no_valid_covering_exists_l794_79468


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l794_79452

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l794_79452


namespace NUMINAMATH_CALUDE_max_rooms_needed_l794_79465

/-- Represents a group of football fans -/
structure FanGroup where
  team : Fin 3
  gender : Bool
  count : Nat

/-- The maximum number of fans that can be accommodated in one room -/
def maxFansPerRoom : Nat := 3

/-- The total number of football fans -/
def totalFans : Nat := 100

/-- Calculates the number of rooms needed for a given fan group -/
def roomsNeeded (group : FanGroup) : Nat :=
  (group.count + maxFansPerRoom - 1) / maxFansPerRoom

/-- Theorem stating the maximum number of rooms needed -/
theorem max_rooms_needed (fans : List FanGroup) 
  (h1 : fans.length = 6)
  (h2 : fans.foldl (λ acc g => acc + g.count) 0 = totalFans) : 
  (fans.foldl (λ acc g => acc + roomsNeeded g) 0) ≤ 37 := by
  sorry

end NUMINAMATH_CALUDE_max_rooms_needed_l794_79465


namespace NUMINAMATH_CALUDE_circle_point_x_value_l794_79473

/-- Given a circle in the xy-plane with diameter endpoints (-3,0) and (21,0),
    if the point (x,12) is on the circle, then x = 9. -/
theorem circle_point_x_value (x : ℝ) : 
  let center : ℝ × ℝ := ((21 - 3) / 2 + -3, 0)
  let radius : ℝ := (21 - (-3)) / 2
  ((x - center.1)^2 + (12 - center.2)^2 = radius^2) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l794_79473


namespace NUMINAMATH_CALUDE_integers_between_cubes_l794_79498

theorem integers_between_cubes : ∃ n : ℕ, n = 26 ∧ 
  n = (⌊(9.3 : ℝ)^3⌋ - ⌈(9.2 : ℝ)^3⌉ + 1) := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l794_79498


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l794_79479

/-- The ratio of average speed to still water speed for a boat trip --/
theorem boat_speed_ratio 
  (still_water_speed : ℝ) 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : still_water_speed = 20)
  (h2 : current_speed = 4)
  (h3 : downstream_distance = 5)
  (h4 : upstream_distance = 3) :
  let downstream_speed := still_water_speed + current_speed
  let upstream_speed := still_water_speed - current_speed
  let total_distance := downstream_distance + upstream_distance
  let total_time := downstream_distance / downstream_speed + upstream_distance / upstream_speed
  let average_speed := total_distance / total_time
  average_speed / still_water_speed = 96 / 95 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l794_79479


namespace NUMINAMATH_CALUDE_inequality_proof_l794_79426

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / c) + (a * c / b) + (b * c / a) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l794_79426


namespace NUMINAMATH_CALUDE_g_max_value_l794_79424

/-- The function g(x) = 4x - x^3 -/
def g (x : ℝ) : ℝ := 4 * x - x^3

/-- The maximum value of g(x) on [0, 2] is 8√3/9 -/
theorem g_max_value : 
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = (8 * Real.sqrt 3) / 9 := by
sorry

end NUMINAMATH_CALUDE_g_max_value_l794_79424


namespace NUMINAMATH_CALUDE_unique_solution_trig_equation_l794_79482

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 115 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trig_equation_l794_79482


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_max_perimeter_l794_79458

theorem smallest_whole_number_above_max_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  57 > 7 + 21 + s ∧ 
  ∀ n : ℕ, n < 57 → ∃ s' : ℝ, 
    s' > 0 ∧ 
    s' + 7 > 21 ∧ 
    s' + 21 > 7 ∧ 
    7 + 21 > s' ∧ 
    n ≤ 7 + 21 + s' :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_max_perimeter_l794_79458


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l794_79484

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l794_79484


namespace NUMINAMATH_CALUDE_prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l794_79494

/-- Represents the total number of circular parts -/
def total_parts : ℕ := 100

/-- Represents the number of parts with qualified diameters -/
def qualified_diameter : ℕ := 98

/-- Represents the number of parts with qualified smoothness -/
def qualified_smoothness : ℕ := 96

/-- Represents the number of parts with both qualified diameter and smoothness -/
def qualified_both : ℕ := 94

/-- Calculates the probability of qualified diameter given qualified smoothness -/
def prob_diameter_given_smoothness : ℚ :=
  qualified_both / qualified_smoothness

/-- Calculates the probability of qualified smoothness given qualified diameter -/
def prob_smoothness_given_diameter : ℚ :=
  qualified_both / qualified_diameter

theorem prob_diameter_given_smoothness_correct :
  prob_diameter_given_smoothness = 94 / 96 := by sorry

theorem prob_smoothness_given_diameter_correct :
  prob_smoothness_given_diameter = 94 / 98 := by sorry

end NUMINAMATH_CALUDE_prob_diameter_given_smoothness_correct_prob_smoothness_given_diameter_correct_l794_79494


namespace NUMINAMATH_CALUDE_cos_sin_transformation_l794_79447

theorem cos_sin_transformation (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 12) + π / 3) :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_transformation_l794_79447


namespace NUMINAMATH_CALUDE_guitar_difference_is_three_l794_79477

/-- The number of fewer 8 string guitars compared to normal guitars -/
def guitar_difference : ℕ :=
  let num_basses : ℕ := 3
  let strings_per_bass : ℕ := 4
  let num_normal_guitars : ℕ := 2 * num_basses
  let strings_per_normal_guitar : ℕ := 6
  let strings_per_8string_guitar : ℕ := 8
  let total_strings : ℕ := 72
  let normal_guitar_strings : ℕ := num_normal_guitars * strings_per_normal_guitar
  let bass_strings : ℕ := num_basses * strings_per_bass
  let remaining_strings : ℕ := total_strings - (normal_guitar_strings + bass_strings)
  let num_8string_guitars : ℕ := remaining_strings / strings_per_8string_guitar
  num_normal_guitars - num_8string_guitars

theorem guitar_difference_is_three :
  guitar_difference = 3 := by sorry

end NUMINAMATH_CALUDE_guitar_difference_is_three_l794_79477


namespace NUMINAMATH_CALUDE_jeromes_contact_list_ratio_l794_79444

/-- Proves that the ratio of out of school friends to classmates is 1:2 given the conditions in Jerome's contact list problem -/
theorem jeromes_contact_list_ratio : 
  ∀ (out_of_school_friends : ℕ),
    20 + out_of_school_friends + 2 + 1 = 33 →
    out_of_school_friends = 10 ∧ 
    (out_of_school_friends : ℚ) / 20 = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_jeromes_contact_list_ratio_l794_79444


namespace NUMINAMATH_CALUDE_gmat_exam_problem_l794_79423

theorem gmat_exam_problem (total : ℕ) (h_total : total > 0) :
  let first_correct := (80 : ℚ) / 100 * total
  let second_correct := (75 : ℚ) / 100 * total
  let neither_correct := (5 : ℚ) / 100 * total
  let both_correct := first_correct + second_correct - total + neither_correct
  (both_correct / total) = (60 : ℚ) / 100 := by
sorry

end NUMINAMATH_CALUDE_gmat_exam_problem_l794_79423


namespace NUMINAMATH_CALUDE_odd_prime_sqrt_sum_l794_79430

theorem odd_prime_sqrt_sum (p : ℕ) : 
  Prime p ↔ (∃ m : ℕ, ∃ n : ℕ, Real.sqrt m + Real.sqrt (m + p) = n) ∧ Odd p := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sqrt_sum_l794_79430


namespace NUMINAMATH_CALUDE_last_three_digits_perfect_square_l794_79471

theorem last_three_digits_perfect_square (n : ℕ) : 
  ∃ (m : ℕ), m * m % 1000 = 689 ∧ 
  ∀ (k : ℕ), k * k % 1000 ≠ 759 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_perfect_square_l794_79471


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l794_79493

/-- The probability of getting exactly k successes in n independent Bernoulli trials 
    with probability p of success on each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 4 heads in 6 independent coin flips -/
def probability_4_heads_in_6_flips (p : ℝ) : ℝ :=
  binomial_probability 6 4 p

theorem unfair_coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : probability_4_heads_in_6_flips p = 500 / 2187) : 
  p = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l794_79493


namespace NUMINAMATH_CALUDE_winnie_lollipops_l794_79404

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops :
  lollipops_kept 432 14 = 12 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l794_79404


namespace NUMINAMATH_CALUDE_work_earnings_equality_l794_79403

theorem work_earnings_equality (t : ℝ) : 
  (t + 2) * (4 * t - 2) = (4 * t - 7) * (t + 3) + 3 → t = 14 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l794_79403


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l794_79497

theorem absolute_value_inequality (x : ℝ) :
  |2 * x + 6| < 10 ↔ -8 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l794_79497


namespace NUMINAMATH_CALUDE_total_accessories_count_l794_79419

def dresses_per_day_first_period : ℕ := 4
def days_first_period : ℕ := 10
def dresses_per_day_second_period : ℕ := 5
def days_second_period : ℕ := 3
def ribbons_per_dress : ℕ := 3
def buttons_per_dress : ℕ := 2
def lace_trims_per_dress : ℕ := 1

theorem total_accessories_count :
  (dresses_per_day_first_period * days_first_period +
   dresses_per_day_second_period * days_second_period) *
  (ribbons_per_dress + buttons_per_dress + lace_trims_per_dress) = 330 := by
sorry

end NUMINAMATH_CALUDE_total_accessories_count_l794_79419


namespace NUMINAMATH_CALUDE_tournament_distributions_l794_79440

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

theorem tournament_distributions :
  total_distributions = 32 :=
sorry

end NUMINAMATH_CALUDE_tournament_distributions_l794_79440


namespace NUMINAMATH_CALUDE_seven_students_distribution_l794_79457

/-- The number of ways to distribute n students into two dormitories,
    with each dormitory having at least m students -/
def distribution_count (n : ℕ) (m : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 112 ways to distribute 7 students into two dormitories,
    with each dormitory having at least 2 students -/
theorem seven_students_distribution :
  distribution_count 7 2 = 112 := by
  sorry

end NUMINAMATH_CALUDE_seven_students_distribution_l794_79457


namespace NUMINAMATH_CALUDE_min_value_theorem_l794_79490

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  ∃ (m : ℝ), m = 105 ∧ ∀ z, x^2 + y^2 - x*y ≥ z → z ≤ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l794_79490


namespace NUMINAMATH_CALUDE_sophie_cookies_count_l794_79420

def cupcake_price : ℚ := 2
def doughnut_price : ℚ := 1
def apple_pie_slice_price : ℚ := 2
def cookie_price : ℚ := 0.6

def num_cupcakes : ℕ := 5
def num_doughnuts : ℕ := 6
def num_apple_pie_slices : ℕ := 4

def total_spent : ℚ := 33

theorem sophie_cookies_count :
  ∃ (num_cookies : ℕ),
    num_cookies = 15 ∧
    total_spent = 
      (num_cupcakes : ℚ) * cupcake_price +
      (num_doughnuts : ℚ) * doughnut_price +
      (num_apple_pie_slices : ℚ) * apple_pie_slice_price +
      (num_cookies : ℚ) * cookie_price :=
by sorry

end NUMINAMATH_CALUDE_sophie_cookies_count_l794_79420


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_12_mod_19_l794_79460

theorem largest_four_digit_congruent_to_12_mod_19 : ∀ n : ℕ,
  n < 10000 → n ≡ 12 [ZMOD 19] → n ≤ 9987 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_12_mod_19_l794_79460


namespace NUMINAMATH_CALUDE_triangle_side_length_l794_79478

-- Define the triangle ABC
def triangle (A B C : ℝ × ℝ) : Prop :=
  -- Add conditions to ensure it's a valid triangle
  true

-- Define the angle measure in degrees
def angle_measure (A B C : ℝ × ℝ) : ℝ := 
  -- Add definition for angle measure
  0

-- Define the distance between two points
def distance (A B : ℝ × ℝ) : ℝ :=
  -- Add definition for distance
  0

theorem triangle_side_length 
  (A B C : ℝ × ℝ) 
  (h_triangle : triangle A B C)
  (h_angle_B : angle_measure A B C = 45)
  (h_angle_C : angle_measure B C A = 80)
  (h_side_AC : distance A C = 5) :
  distance B C = (10 * Real.sin (55 * π / 180)) / Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l794_79478


namespace NUMINAMATH_CALUDE_sector_radius_l794_79429

/-- Given a circular sector with area 7 square centimeters and arc length 3.5 cm,
    prove that the radius of the circle is 4 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
    (h_area : area = 7) 
    (h_arc_length : arc_length = 3.5) 
    (h_sector_area : area = (arc_length * radius) / 2) : radius = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l794_79429


namespace NUMINAMATH_CALUDE_max_store_visits_l794_79475

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : unique_visitors = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors * 2 ≤ total_visits) 
  (h6 : ∀ v, v ≤ unique_visitors → v ≥ 1) : 
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
  (∀ v, v ≤ unique_visitors → v ≤ max_visits) ∧ max_visits = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_store_visits_l794_79475


namespace NUMINAMATH_CALUDE_iron_bar_width_is_48_l794_79467

-- Define the dimensions of the iron bar
def iron_bar_length : ℝ := 12
def iron_bar_height : ℝ := 6

-- Define the number of iron bars and iron balls
def num_iron_bars : ℕ := 10
def num_iron_balls : ℕ := 720

-- Define the volume of each iron ball
def iron_ball_volume : ℝ := 8

-- Theorem statement
theorem iron_bar_width_is_48 (w : ℝ) :
  (num_iron_bars : ℝ) * (iron_bar_length * w * iron_bar_height) =
  (num_iron_balls : ℝ) * iron_ball_volume →
  w = 48 := by sorry

end NUMINAMATH_CALUDE_iron_bar_width_is_48_l794_79467


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_nine_between_90_and_200_l794_79462

theorem unique_square_divisible_by_nine_between_90_and_200 :
  ∃! y : ℕ, 
    90 < y ∧ 
    y < 200 ∧ 
    ∃ n : ℕ, y = n^2 ∧ 
    ∃ k : ℕ, y = 9 * k :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_nine_between_90_and_200_l794_79462


namespace NUMINAMATH_CALUDE_total_cost_is_88_l794_79464

/-- The cost of a single pair of jeans in dollars -/
def jeans_cost : ℝ := 14.50

/-- The cost of a single shirt in dollars -/
def shirt_cost : ℝ := 9.50

/-- The cost of a single jacket in dollars -/
def jacket_cost : ℝ := 21.00

/-- The number of pairs of jeans John buys -/
def jeans_quantity : ℕ := 2

/-- The number of shirts John buys -/
def shirt_quantity : ℕ := 4

/-- The number of jackets John buys -/
def jacket_quantity : ℕ := 1

/-- The total original cost of John's purchase -/
def total_cost : ℝ := jeans_cost * jeans_quantity + shirt_cost * shirt_quantity + jacket_cost * jacket_quantity

theorem total_cost_is_88 : total_cost = 88 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_88_l794_79464


namespace NUMINAMATH_CALUDE_basketball_team_chemistry_count_l794_79425

theorem basketball_team_chemistry_count :
  ∀ (total_players physics_players both_players chemistry_players : ℕ),
    total_players = 15 →
    physics_players = 8 →
    both_players = 3 →
    physics_players + chemistry_players - both_players = total_players →
    chemistry_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_chemistry_count_l794_79425


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l794_79407

theorem opposite_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := 3/4
  (∀ z : ℚ, z + x = 0 ↔ z = y) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_fourths_l794_79407


namespace NUMINAMATH_CALUDE_shortest_side_length_l794_79431

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The length of the shortest side of the triangle -/
  shortest_side : ℝ
  /-- Condition: The radius is positive -/
  r_pos : r > 0
  /-- Condition: The segments are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Condition: The shortest side is positive -/
  shortest_side_pos : shortest_side > 0

/-- Theorem: In a triangle with an inscribed circle of radius 5 units, 
    where one side is divided into segments of 9 and 5 units by the point of tangency, 
    the length of the shortest side is 16 units. -/
theorem shortest_side_length (t : InscribedCircleTriangle) 
    (h1 : t.r = 5)
    (h2 : t.a = 9)
    (h3 : t.b = 5) : 
  t.shortest_side = 16 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l794_79431


namespace NUMINAMATH_CALUDE_min_value_of_expression_l794_79437

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0)
  (hx : x₁^2 - 4*a*x₁ + 3*a^2 = 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 = 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), y₁^2 - 4*a*y₁ + 3*a^2 = 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 = 0 → 
  y₁ + y₂ + a / (y₁ * y₂) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l794_79437


namespace NUMINAMATH_CALUDE_teacher_age_l794_79495

theorem teacher_age (n : ℕ) (initial_avg : ℚ) (transfer_age : ℕ) (new_avg : ℚ) :
  n = 45 →
  initial_avg = 14 →
  transfer_age = 15 →
  new_avg = 14.66 →
  let remaining_students := n - 1
  let total_age := n * initial_avg
  let remaining_age := total_age - transfer_age
  let teacher_age := (remaining_students + 1) * new_avg - remaining_age
  (∀ p : ℕ, Prime p → p > teacher_age → p ≥ 17) ∧
  Prime 17 ∧
  17 > teacher_age :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l794_79495


namespace NUMINAMATH_CALUDE_prob_sum_div_three_is_seven_ninths_l794_79427

/-- Represents a biased die where even numbers are twice as likely as odd numbers -/
structure BiasedDie where
  /-- Probability of rolling an odd number -/
  odd_prob : ℝ
  /-- Probability of rolling an even number -/
  even_prob : ℝ
  /-- The probabilities sum to 1 -/
  sum_to_one : odd_prob + even_prob = 1
  /-- Even numbers are twice as likely as odd numbers -/
  even_twice_odd : even_prob = 2 * odd_prob

/-- The probability that the sum of three rolls of a biased die is divisible by 3 -/
def prob_sum_div_three (d : BiasedDie) : ℝ :=
  d.even_prob^3 + d.odd_prob^3 + 3 * d.even_prob^2 * d.odd_prob

/-- Theorem: The probability that the sum of three rolls of the biased die is divisible by 3 is 7/9 -/
theorem prob_sum_div_three_is_seven_ninths (d : BiasedDie) :
    prob_sum_div_three d = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_div_three_is_seven_ninths_l794_79427


namespace NUMINAMATH_CALUDE_zhang_apple_sales_l794_79415

/-- Represents the number of apples Zhang needs to sell to earn a specific profit -/
def apples_to_sell (buy_price : ℚ) (sell_price : ℚ) (target_profit : ℚ) : ℚ :=
  target_profit / (sell_price - buy_price)

/-- Theorem stating the number of apples Zhang needs to sell to earn 15 yuan -/
theorem zhang_apple_sales : 
  let buy_price : ℚ := 1 / 4  -- 4 apples for 1 yuan
  let sell_price : ℚ := 2 / 5 -- 5 apples for 2 yuan
  let target_profit : ℚ := 15
  apples_to_sell buy_price sell_price target_profit = 100 :=
by
  sorry

#eval apples_to_sell (1/4) (2/5) 15

end NUMINAMATH_CALUDE_zhang_apple_sales_l794_79415


namespace NUMINAMATH_CALUDE_factorial_minus_one_mod_930_l794_79405

theorem factorial_minus_one_mod_930 : (Nat.factorial 30 - 1) % 930 = 29 := by
  sorry

end NUMINAMATH_CALUDE_factorial_minus_one_mod_930_l794_79405


namespace NUMINAMATH_CALUDE_bottles_sold_wed_to_sun_is_250_l794_79491

/-- Represents the inventory and sales of hand sanitizer bottles at Danivan Drugstore --/
structure DrugstoreInventory where
  initial_inventory : ℕ
  monday_sales : ℕ
  tuesday_sales : ℕ
  saturday_delivery : ℕ
  final_inventory : ℕ

/-- Calculates the number of bottles sold from Wednesday to Sunday --/
def bottles_sold_wed_to_sun (d : DrugstoreInventory) : ℕ :=
  d.initial_inventory - d.monday_sales - d.tuesday_sales + d.saturday_delivery - d.final_inventory

/-- Theorem stating that the number of bottles sold from Wednesday to Sunday is 250 --/
theorem bottles_sold_wed_to_sun_is_250 (d : DrugstoreInventory) 
    (h1 : d.initial_inventory = 4500)
    (h2 : d.monday_sales = 2445)
    (h3 : d.tuesday_sales = 900)
    (h4 : d.saturday_delivery = 650)
    (h5 : d.final_inventory = 1555) :
    bottles_sold_wed_to_sun d = 250 := by
  sorry

end NUMINAMATH_CALUDE_bottles_sold_wed_to_sun_is_250_l794_79491


namespace NUMINAMATH_CALUDE_negative_division_equals_nine_l794_79483

theorem negative_division_equals_nine : (-81) / (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_equals_nine_l794_79483


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l794_79451

theorem cube_volume_from_surface_area :
  ∀ s V : ℝ,
  (6 * s^2 = 864) →  -- Surface area condition
  (V = s^3) →        -- Volume definition
  V = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l794_79451


namespace NUMINAMATH_CALUDE_cos_135_degrees_l794_79463

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l794_79463


namespace NUMINAMATH_CALUDE_cubic_sum_inequality_l794_79418

theorem cubic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) ≥ (x*y + y*z + z*x) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_inequality_l794_79418


namespace NUMINAMATH_CALUDE_total_sides_of_dice_l794_79450

/-- The number of dice each person brought -/
def dice_per_person : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def number_of_people : ℕ := 2

/-- Theorem: The total number of sides on all dice brought by two people, 
    each bringing 4 six-sided dice, is 48. -/
theorem total_sides_of_dice : 
  number_of_people * dice_per_person * sides_per_die = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_sides_of_dice_l794_79450


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l794_79445

theorem rectangle_shorter_side 
  (area : ℝ) 
  (perimeter : ℝ) 
  (h_area : area = 91) 
  (h_perimeter : perimeter = 40) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    width = 7 ∧ 
    width ≤ length := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l794_79445


namespace NUMINAMATH_CALUDE_pressure_volume_inverse_proportionality_l794_79459

/-- Given inverse proportionality of pressure and volume, prove that if the initial pressure is 8 kPa
    at 3.5 liters, then the pressure at 7 liters is 4 kPa. -/
theorem pressure_volume_inverse_proportionality
  (pressure volume : ℝ → ℝ) -- Pressure and volume as functions of time
  (t₀ t₁ : ℝ) -- Initial and final times
  (h_inverse_prop : ∀ t, pressure t * volume t = pressure t₀ * volume t₀) -- Inverse proportionality
  (h_init_volume : volume t₀ = 3.5)
  (h_init_pressure : pressure t₀ = 8)
  (h_final_volume : volume t₁ = 7) :
  pressure t₁ = 4 := by
  sorry

end NUMINAMATH_CALUDE_pressure_volume_inverse_proportionality_l794_79459


namespace NUMINAMATH_CALUDE_prime_sum_power_implies_three_power_l794_79453

theorem prime_sum_power_implies_three_power (n : ℕ) : 
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_power_implies_three_power_l794_79453


namespace NUMINAMATH_CALUDE_weight_ratio_l794_79449

def student_weight : ℝ := 79
def total_weight : ℝ := 116
def weight_loss : ℝ := 5

def sister_weight : ℝ := total_weight - student_weight
def student_new_weight : ℝ := student_weight - weight_loss

theorem weight_ratio : student_new_weight / sister_weight = 2 := by sorry

end NUMINAMATH_CALUDE_weight_ratio_l794_79449


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l794_79455

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l794_79455


namespace NUMINAMATH_CALUDE_bus_fare_cost_l794_79461

/-- Represents the cost of a bus fare for one person one way -/
def bus_fare : ℝ := 1.5

/-- Represents the cost of zoo entry for one person -/
def zoo_entry : ℝ := 5

/-- Represents the total money brought -/
def total_money : ℝ := 40

/-- Represents the money left after zoo entry and bus fare -/
def money_left : ℝ := 24

/-- Proves that the bus fare cost per person one way is $1.50 -/
theorem bus_fare_cost : 
  2 * zoo_entry + 4 * bus_fare = total_money - money_left :=
by sorry

end NUMINAMATH_CALUDE_bus_fare_cost_l794_79461


namespace NUMINAMATH_CALUDE_terrell_new_lifts_count_l794_79492

/-- The number of times Terrell must lift the new weights to match or exceed his original total weight -/
def min_lifts_for_equal_weight (original_weight : ℕ) (original_reps : ℕ) (new_weight : ℕ) : ℕ :=
  let original_total := 2 * original_weight * original_reps
  let new_total_per_rep := 2 * new_weight
  ((original_total + new_total_per_rep - 1) / new_total_per_rep : ℕ)

/-- Theorem stating that Terrell needs at least 14 lifts with the new weights -/
theorem terrell_new_lifts_count :
  min_lifts_for_equal_weight 25 10 18 = 14 := by
  sorry

#eval min_lifts_for_equal_weight 25 10 18

end NUMINAMATH_CALUDE_terrell_new_lifts_count_l794_79492


namespace NUMINAMATH_CALUDE_simultaneous_ring_theorem_l794_79442

def bell_ring_time (start_hour start_minute : ℕ) (interval_minutes : ℕ) : ℕ × ℕ := sorry

def next_simultaneous_ring 
  (start_hour start_minute : ℕ) 
  (interval1 interval2 interval3 : ℕ) : ℕ × ℕ := sorry

theorem simultaneous_ring_theorem 
  (h1 : interval1 = 18)
  (h2 : interval2 = 24)
  (h3 : interval3 = 30)
  (h4 : start_hour = 10)
  (h5 : start_minute = 0) :
  next_simultaneous_ring start_hour start_minute interval1 interval2 interval3 = (16, 0) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_ring_theorem_l794_79442


namespace NUMINAMATH_CALUDE_horner_v3_calculation_l794_79496

/-- Horner's method V₃ calculation for a specific polynomial -/
theorem horner_v3_calculation (x : ℝ) (h : x = 4) : 
  let f := fun (x : ℝ) => 4*x^6 + 3*x^5 + 4*x^4 + 2*x^3 + 5*x^2 - 7*x + 9
  let v3 := (4*x + 3)*x + 4
  v3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_horner_v3_calculation_l794_79496


namespace NUMINAMATH_CALUDE_unique_triplet_l794_79485

theorem unique_triplet : ∃! (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b * c + 1) % a = 0 ∧
  (a * c + 1) % b = 0 ∧
  (a * b + 1) % c = 0 ∧
  a = 2 ∧ b = 3 ∧ c = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_triplet_l794_79485


namespace NUMINAMATH_CALUDE_degree_of_g_l794_79443

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9*x^5 + 5*x^4 + 2*x^2 - x + 6

-- Define the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ c : ℝ, ∀ x : ℝ, f x + g x = c) →  -- degree of f(x) + g(x) is 0
  (∃ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₅ ≠ 0 ∧ 
    ∀ x : ℝ, g x = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →  -- g(x) is a polynomial of degree 5
  true :=
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l794_79443


namespace NUMINAMATH_CALUDE_rebecca_income_percentage_l794_79400

def rebecca_income : ℕ := 15000
def jimmy_income : ℕ := 18000
def income_increase : ℕ := 7000

def new_rebecca_income : ℕ := rebecca_income + income_increase
def combined_income : ℕ := new_rebecca_income + jimmy_income

theorem rebecca_income_percentage :
  (new_rebecca_income : ℚ) / (combined_income : ℚ) = 55 / 100 := by sorry

end NUMINAMATH_CALUDE_rebecca_income_percentage_l794_79400


namespace NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l794_79470

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (2*x + 3)^8 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8) →
  a + a₂ + a₄ + a₆ + a₈ = 3281 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_indexed_coefficients_l794_79470


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_and_house_l794_79469

theorem blocks_used_for_tower_and_house : 
  let total_blocks : ℕ := 58
  let tower_blocks : ℕ := 27
  let house_blocks : ℕ := 53
  tower_blocks + house_blocks = 80 :=
by sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_and_house_l794_79469


namespace NUMINAMATH_CALUDE_complex_sum_problem_l794_79488

theorem complex_sum_problem (a b c d e f : ℝ) : 
  b = 1 → 
  e = -a - c → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) = -Complex.I → 
  d + f = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l794_79488


namespace NUMINAMATH_CALUDE_race_track_width_l794_79416

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 880 →
  outer_radius = 165.0563499208679 →
  ∃ width : ℝ, (abs (width - 25.049) < 0.001 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_l794_79416


namespace NUMINAMATH_CALUDE_probability_endpoints_of_edge_is_four_fifths_l794_79410

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count_per_vertex : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- The probability of choosing two vertices that are endpoints of an edge -/
def probability_endpoints_of_edge (o : RegularOctahedron) : ℚ :=
  (4 : ℚ) / 5

/-- Theorem: The probability of randomly choosing two vertices of a regular octahedron 
    that are endpoints of an edge is 4/5 -/
theorem probability_endpoints_of_edge_is_four_fifths (o : RegularOctahedron) :
  probability_endpoints_of_edge o = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_endpoints_of_edge_is_four_fifths_l794_79410


namespace NUMINAMATH_CALUDE_kitchen_renovation_rate_l794_79402

/-- The hourly rate for professionals renovating Kamil's kitchen -/
def hourly_rate (professionals : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (professionals * hours_per_day * days)

/-- Theorem stating the hourly rate for the kitchen renovation professionals -/
theorem kitchen_renovation_rate : 
  hourly_rate 2 6 7 1260 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_renovation_rate_l794_79402


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l794_79446

theorem fraction_sum_equality : 
  (-3 : ℚ) / 20 + 5 / 200 - 7 / 2000 * 2 = -132 / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l794_79446


namespace NUMINAMATH_CALUDE_seven_trees_planting_methods_l794_79480

/-- The number of ways to plant n trees in a row, choosing from plane trees and willow trees,
    such that no two adjacent trees are both willows. -/
def valid_planting_methods (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 34 valid planting methods for 7 trees. -/
theorem seven_trees_planting_methods :
  valid_planting_methods 7 = 34 :=
sorry

end NUMINAMATH_CALUDE_seven_trees_planting_methods_l794_79480


namespace NUMINAMATH_CALUDE_anderson_shirts_theorem_l794_79432

theorem anderson_shirts_theorem (total_clothing pieces_of_trousers : ℕ) 
  (h1 : total_clothing = 934)
  (h2 : pieces_of_trousers = 345) :
  total_clothing - pieces_of_trousers = 589 := by
  sorry

end NUMINAMATH_CALUDE_anderson_shirts_theorem_l794_79432


namespace NUMINAMATH_CALUDE_prime_roots_equation_l794_79433

theorem prime_roots_equation (p q : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ 
   x ≠ y ∧
   (p * x^2 - q * x + 1985 = 0) ∧ 
   (p * y^2 - q * y + 1985 = 0)) →
  12 * p^2 + q = 414 := by
sorry

end NUMINAMATH_CALUDE_prime_roots_equation_l794_79433


namespace NUMINAMATH_CALUDE_sports_cards_pages_l794_79421

/-- Calculates the number of pages needed for a given number of cards and cards per page -/
def pagesNeeded (cards : ℕ) (cardsPerPage : ℕ) : ℕ :=
  (cards + cardsPerPage - 1) / cardsPerPage

theorem sports_cards_pages : 
  let baseballCards := 12
  let baseballCardsPerPage := 4
  let basketballCards := 14
  let basketballCardsPerPage := 3
  let soccerCards := 7
  let soccerCardsPerPage := 5
  (pagesNeeded baseballCards baseballCardsPerPage) +
  (pagesNeeded basketballCards basketballCardsPerPage) +
  (pagesNeeded soccerCards soccerCardsPerPage) = 10 := by
  sorry


end NUMINAMATH_CALUDE_sports_cards_pages_l794_79421


namespace NUMINAMATH_CALUDE_binomial_12_11_l794_79441

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_11_l794_79441


namespace NUMINAMATH_CALUDE_correct_sampling_order_l794_79436

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

-- Define the properties of each sampling method
def isSimpleRandom (method : SamplingMethod) : Prop :=
  method = SamplingMethod.SimpleRandom

def isSystematic (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Systematic

def isStratified (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified

-- Define the properties of the given methods
def method1Properties (method : SamplingMethod) : Prop :=
  isSimpleRandom method

def method2Properties (method : SamplingMethod) : Prop :=
  isSystematic method

def method3Properties (method : SamplingMethod) : Prop :=
  isStratified method

-- Theorem statement
theorem correct_sampling_order :
  ∃ (m1 m2 m3 : SamplingMethod),
    method1Properties m1 ∧
    method2Properties m2 ∧
    method3Properties m3 ∧
    m1 = SamplingMethod.SimpleRandom ∧
    m2 = SamplingMethod.Systematic ∧
    m3 = SamplingMethod.Stratified :=
by
  sorry

end NUMINAMATH_CALUDE_correct_sampling_order_l794_79436
