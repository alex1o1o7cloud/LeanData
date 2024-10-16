import Mathlib

namespace NUMINAMATH_CALUDE_jade_lego_tower_level_width_l3415_341515

/-- Calculates the width of each level in Jade's Lego tower -/
theorem jade_lego_tower_level_width 
  (initial_pieces : ℕ) 
  (levels : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : initial_pieces = 100)
  (h2 : levels = 11)
  (h3 : remaining_pieces = 23) :
  (initial_pieces - remaining_pieces) / levels = 7 := by
  sorry

end NUMINAMATH_CALUDE_jade_lego_tower_level_width_l3415_341515


namespace NUMINAMATH_CALUDE_shortest_paths_count_l3415_341573

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the grid and gas station locations -/
structure Grid where
  width : ℕ
  height : ℕ
  gasStations : List Point

/-- Represents the problem setup -/
structure ProblemSetup where
  grid : Grid
  start : Point
  finish : Point
  refuelDistance : ℕ

/-- Calculates the number of shortest paths between two points on a grid -/
def numberOfShortestPaths (start : Point) (finish : Point) : ℕ :=
  sorry

/-- Checks if a path is valid given the refuel constraints -/
def isValidPath (path : List Point) (gasStations : List Point) (refuelDistance : ℕ) : Bool :=
  sorry

/-- Main theorem: The number of shortest paths from A to B with refueling constraints is 24 -/
theorem shortest_paths_count (setup : ProblemSetup) : 
  (numberOfShortestPaths setup.start setup.finish) = 24 :=
sorry

end NUMINAMATH_CALUDE_shortest_paths_count_l3415_341573


namespace NUMINAMATH_CALUDE_angles_on_y_axis_l3415_341571

def terminal_side_on_y_axis (θ : Real) : Prop :=
  ∃ n : Int, θ = n * Real.pi + Real.pi / 2

theorem angles_on_y_axis :
  {θ : Real | terminal_side_on_y_axis θ} = {θ : Real | ∃ n : Int, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end NUMINAMATH_CALUDE_angles_on_y_axis_l3415_341571


namespace NUMINAMATH_CALUDE_polygon_with_60_degree_exterior_angles_has_6_sides_l3415_341547

-- Define a polygon type
structure Polygon where
  sides : ℕ
  exteriorAngle : ℝ

-- Theorem statement
theorem polygon_with_60_degree_exterior_angles_has_6_sides :
  ∀ p : Polygon, p.exteriorAngle = 60 → p.sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_60_degree_exterior_angles_has_6_sides_l3415_341547


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3415_341551

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.sqrt 3 * a * Real.sin C - c * (2 + Real.cos A) = 0 →
  a = Real.sqrt 13 →
  Real.sin C = 3 * Real.sin B →
  a ≥ b ∧ a ≥ c →
  (A = 2 * π / 3 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3415_341551


namespace NUMINAMATH_CALUDE_game_points_proof_l3415_341585

def points_earned (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem game_points_proof :
  points_earned 7 8 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_game_points_proof_l3415_341585


namespace NUMINAMATH_CALUDE_total_games_is_30_l3415_341562

/-- The number of Monopoly games won by Betsy, Helen, and Susan -/
def monopoly_games (betsy helen susan : ℕ) : Prop :=
  betsy = 5 ∧ helen = 2 * betsy ∧ susan = 3 * betsy

/-- The total number of games won by all three players -/
def total_games (betsy helen susan : ℕ) : ℕ :=
  betsy + helen + susan

/-- Theorem stating that the total number of games won is 30 -/
theorem total_games_is_30 :
  ∀ betsy helen susan : ℕ,
  monopoly_games betsy helen susan →
  total_games betsy helen susan = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_total_games_is_30_l3415_341562


namespace NUMINAMATH_CALUDE_system_solution_unique_l3415_341577

theorem system_solution_unique (x y : ℝ) : 
  (x - y = -5 ∧ 3*x + 2*y = 10) ↔ (x = 0 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3415_341577


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3415_341505

theorem fraction_evaluation : 
  (20-18+16-14+12-10+8-6+4-2) / (2-4+6-8+10-12+14-16+18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3415_341505


namespace NUMINAMATH_CALUDE_smallest_difference_l3415_341593

def Digits : Finset Nat := {0, 2, 4, 5, 7}

def is_valid_arrangement (a b c d x y z : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧
  x ∈ Digits ∧ y ∈ Digits ∧ z ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ x ∧ a ≠ y ∧ a ≠ z ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ x ∧ b ≠ y ∧ b ≠ z ∧
  c ≠ d ∧ c ≠ x ∧ c ≠ y ∧ c ≠ z ∧
  d ≠ x ∧ d ≠ y ∧ d ≠ z ∧
  x ≠ y ∧ x ≠ z ∧
  y ≠ z ∧
  a ≠ 0 ∧ x ≠ 0

def difference (a b c d x y z : Nat) : Nat :=
  1000 * a + 100 * b + 10 * c + d - (100 * x + 10 * y + z)

theorem smallest_difference :
  ∀ a b c d x y z,
    is_valid_arrangement a b c d x y z →
    difference a b c d x y z ≥ 1325 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_l3415_341593


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l3415_341587

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers can form a triangle -/
def canFormTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

/-- The main theorem stating that the smallest perimeter of a scalene triangle
    with consecutive odd prime side lengths and a prime perimeter is 23 -/
theorem smallest_prime_perimeter_triangle :
  ∀ a b c : ℕ,
  areConsecutiveOddPrimes a b c →
  canFormTriangle a b c →
  isPrime (a + b + c) →
  a + b + c ≥ 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l3415_341587


namespace NUMINAMATH_CALUDE_division_of_fractions_l3415_341597

theorem division_of_fractions : 
  (5 / 6 : ℚ) / (7 / 9 : ℚ) / (11 / 13 : ℚ) = 195 / 154 := by sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3415_341597


namespace NUMINAMATH_CALUDE_vector_perpendicular_implies_x_value_l3415_341563

/-- Given vectors a and b in R^2, if a is perpendicular to 2a + b, then the x-coordinate of b is 10 -/
theorem vector_perpendicular_implies_x_value (a b : ℝ × ℝ) :
  a = (-2, 3) →
  b.2 = -2 →
  (a.1 * (2 * a.1 + b.1) + a.2 * (2 * a.2 + b.2) = 0) →
  b.1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_perpendicular_implies_x_value_l3415_341563


namespace NUMINAMATH_CALUDE_find_m_l3415_341566

theorem find_m : ∃ m : ℚ, m * 9999 = 624877405 ∧ m = 62493.5 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3415_341566


namespace NUMINAMATH_CALUDE_road_length_theorem_l3415_341523

/-- Represents the distance between two markers on the road. -/
structure MarkerDistance where
  fromA : ℕ
  fromB : ℕ

/-- The road between cities A and B -/
structure Road where
  length : ℕ
  marker1 : MarkerDistance
  marker2 : MarkerDistance

/-- Conditions for a valid road configuration -/
def isValidRoad (r : Road) : Prop :=
  (r.marker1.fromA + r.marker1.fromB = r.length) ∧
  (r.marker2.fromA + r.marker2.fromB = r.length) ∧
  (r.marker2.fromA = r.marker1.fromA + 10) ∧
  ((r.marker1.fromA = 2 * r.marker1.fromB ∨ r.marker1.fromB = 2 * r.marker1.fromA) ∧
   (r.marker2.fromA = 3 * r.marker2.fromB ∨ r.marker2.fromB = 3 * r.marker2.fromA))

theorem road_length_theorem :
  ∀ r : Road, isValidRoad r → (r.length = 120 ∨ r.length = 24) ∧
  (∀ d : ℕ, d ≠ 120 ∧ d ≠ 24 → ¬∃ r' : Road, r'.length = d ∧ isValidRoad r') :=
by sorry

end NUMINAMATH_CALUDE_road_length_theorem_l3415_341523


namespace NUMINAMATH_CALUDE_tiling_problem_l3415_341545

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a square tile -/
structure TileDimensions where
  side : ℕ

/-- Calculates the number of tiles needed to cover a room -/
def calculate_tiles (room : RoomDimensions) (border_tile : TileDimensions) (inner_tile : TileDimensions) : ℕ :=
  sorry

/-- Theorem statement for the tiling problem -/
theorem tiling_problem (room : RoomDimensions) (border_tile : TileDimensions) (inner_tile : TileDimensions) :
  room.length = 24 ∧ room.width = 18 ∧ border_tile.side = 2 ∧ inner_tile.side = 1 →
  calculate_tiles room border_tile inner_tile = 318 :=
by sorry

end NUMINAMATH_CALUDE_tiling_problem_l3415_341545


namespace NUMINAMATH_CALUDE_plane_ticket_price_is_800_l3415_341510

/-- Represents the luggage and ticket pricing scenario -/
structure LuggagePricing where
  totalWeight : ℕ
  freeAllowance : ℕ
  excessChargeRate : ℚ
  luggageTicketPrice : ℕ

/-- Calculates the plane ticket price based on the given luggage pricing scenario -/
def planeTicketPrice (scenario : LuggagePricing) : ℕ :=
  sorry

/-- Theorem stating that the plane ticket price is 800 yuan for the given scenario -/
theorem plane_ticket_price_is_800 :
  planeTicketPrice ⟨30, 20, 3/200, 120⟩ = 800 :=
sorry

end NUMINAMATH_CALUDE_plane_ticket_price_is_800_l3415_341510


namespace NUMINAMATH_CALUDE_expression_value_l3415_341588

theorem expression_value : 4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 8000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3415_341588


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3415_341557

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The asymptotic line equations of a hyperbola -/
def asymptoticLines (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- Theorem stating that the given standard equation implies the asymptotic lines,
    but not necessarily vice versa -/
theorem hyperbola_asymptotes (h : Hyperbola) :
  (h.a = 4 ∧ h.b = 3 → ∀ x y, standardEquation h x y → asymptoticLines h x y) ∧
  ¬(∀ h : Hyperbola, (∀ x y, asymptoticLines h x y → standardEquation h x y)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3415_341557


namespace NUMINAMATH_CALUDE_function_extrema_product_l3415_341578

theorem function_extrema_product (a b : Real) :
  let f := fun x => a - Real.sqrt 3 * Real.tan (2 * x)
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) b, f x ≥ 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 7) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) b, f x = 3) →
  a * b = Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_product_l3415_341578


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3415_341594

theorem quadratic_roots_condition (n : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + n*x + 9 = 0 ∧ y^2 + n*y + 9 = 0) ↔ 
  n < -6 ∨ n > 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3415_341594


namespace NUMINAMATH_CALUDE_jake_has_eight_peaches_l3415_341543

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference in peaches between Steven and Jake -/
def difference : ℕ := 7

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - difference

theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eight_peaches_l3415_341543


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3415_341539

/-- Calculate the average speed of a round trip flight with wind vectors -/
theorem round_trip_average_speed 
  (speed_to_mother : ℝ) 
  (tailwind_speed : ℝ) 
  (tailwind_angle : ℝ) 
  (speed_to_home : ℝ) 
  (headwind_speed : ℝ) 
  (headwind_angle : ℝ) 
  (h1 : speed_to_mother = 96) 
  (h2 : tailwind_speed = 12) 
  (h3 : tailwind_angle = 30 * π / 180) 
  (h4 : speed_to_home = 88) 
  (h5 : headwind_speed = 15) 
  (h6 : headwind_angle = 60 * π / 180) : 
  ∃ (average_speed : ℝ), 
    abs (average_speed - 93.446) < 0.001 ∧ 
    average_speed = (
      (speed_to_mother + tailwind_speed * Real.cos tailwind_angle) + 
      (speed_to_home - headwind_speed * Real.cos headwind_angle)
    ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l3415_341539


namespace NUMINAMATH_CALUDE_intersection_of_given_sets_l3415_341529

theorem intersection_of_given_sets :
  let A : Set ℕ := {1, 3, 4}
  let B : Set ℕ := {3, 4, 5}
  A ∩ B = {3, 4} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_given_sets_l3415_341529


namespace NUMINAMATH_CALUDE_bob_water_percentage_l3415_341511

-- Define the water requirements for each crop
def water_corn : ℕ := 20
def water_cotton : ℕ := 80
def water_beans : ℕ := 2 * water_corn

-- Define the acreage for each farmer
def bob_corn : ℕ := 3
def bob_cotton : ℕ := 9
def bob_beans : ℕ := 12

def brenda_corn : ℕ := 6
def brenda_cotton : ℕ := 7
def brenda_beans : ℕ := 14

def bernie_corn : ℕ := 2
def bernie_cotton : ℕ := 12

-- Calculate water usage for each farmer
def bob_water : ℕ := bob_corn * water_corn + bob_cotton * water_cotton + bob_beans * water_beans
def brenda_water : ℕ := brenda_corn * water_corn + brenda_cotton * water_cotton + brenda_beans * water_beans
def bernie_water : ℕ := bernie_corn * water_corn + bernie_cotton * water_cotton

-- Calculate total water usage
def total_water : ℕ := bob_water + brenda_water + bernie_water

-- Define the theorem
theorem bob_water_percentage :
  (bob_water : ℚ) / total_water * 100 = 36 := by sorry

end NUMINAMATH_CALUDE_bob_water_percentage_l3415_341511


namespace NUMINAMATH_CALUDE_total_is_700_l3415_341567

/-- The number of magazines Marie sold -/
def magazines : ℕ := 425

/-- The number of newspapers Marie sold -/
def newspapers : ℕ := 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := magazines + newspapers

/-- Proof that the total number of reading materials sold is 700 -/
theorem total_is_700 : total_reading_materials = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_is_700_l3415_341567


namespace NUMINAMATH_CALUDE_flower_count_l3415_341533

theorem flower_count : 
  ∀ (flowers bees : ℕ), 
    bees = 3 → 
    bees = flowers - 2 → 
    flowers = 5 := by sorry

end NUMINAMATH_CALUDE_flower_count_l3415_341533


namespace NUMINAMATH_CALUDE_parabola_max_value_l3415_341534

theorem parabola_max_value (x : ℝ) : 
  ∃ (max : ℝ), max = 6 ∧ ∀ y : ℝ, y = -3 * x^2 + 6 → y ≤ max :=
sorry

end NUMINAMATH_CALUDE_parabola_max_value_l3415_341534


namespace NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l3415_341535

-- Define the concept of parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the concept of corresponding angles
def corresponding_angles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the concept of equal angles
def angle_equal (a1 a2 : Angle) : Prop := sorry

-- The theorem to be proved
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  parallel l1 l2 → corresponding_angles a1 a2 l1 l2 → angle_equal a1 a2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_corresponding_angles_l3415_341535


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l3415_341569

theorem quadratic_polynomial_problem : ∃ (q : ℝ → ℝ),
  (∀ x, q x = -4.5 * x^2 - 13.5 * x + 81) ∧
  q (-6) = 0 ∧
  q 3 = 0 ∧
  q 4 = -45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l3415_341569


namespace NUMINAMATH_CALUDE_race_finish_difference_l3415_341514

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  distance : ℝ

/-- Represents the race with three runners -/
structure Race where
  runner1 : Runner
  runner2 : Runner
  runner3 : Runner
  constant_speed : Prop

/-- The difference in distance between two runners at the finish line -/
def distance_difference (r1 r2 : Runner) : ℝ :=
  r1.distance - r2.distance

/-- The theorem statement -/
theorem race_finish_difference (race : Race) 
  (h1 : distance_difference race.runner1 race.runner2 = 2)
  (h2 : distance_difference race.runner1 race.runner3 = 4)
  (h3 : race.constant_speed) :
  distance_difference race.runner2 race.runner3 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_difference_l3415_341514


namespace NUMINAMATH_CALUDE_discount_calculation_l3415_341501

/-- The discount calculation problem --/
theorem discount_calculation (original_cost spent : ℝ) 
  (h1 : original_cost = 35)
  (h2 : spent = 18) : 
  original_cost - spent = 17 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3415_341501


namespace NUMINAMATH_CALUDE_comparison_and_inequality_l3415_341580

theorem comparison_and_inequality (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (2 * x^2 + y^2 > x^2 + x * y) ∧ (Real.sqrt 6 - Real.sqrt 5 < 2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_comparison_and_inequality_l3415_341580


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l3415_341550

theorem greatest_common_divisor_of_differences : Nat.gcd (858 - 794) (Nat.gcd (1351 - 858) (1351 - 794)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_differences_l3415_341550


namespace NUMINAMATH_CALUDE_roots_difference_l3415_341576

theorem roots_difference (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 → 
  x₂^2 + x₂ - 3 = 0 → 
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_l3415_341576


namespace NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l3415_341558

/-- Calculates the profit percentage of a dishonest dealer. -/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_positive : 0 < claimed_weight)
  (h_weights : actual_weight = 575 ∧ claimed_weight = 1000) :
  (claimed_weight - actual_weight) / claimed_weight * 100 = 42.5 :=
by
  sorry

#check dishonest_dealer_profit_percentage

end NUMINAMATH_CALUDE_dishonest_dealer_profit_percentage_l3415_341558


namespace NUMINAMATH_CALUDE_divisible_by_seven_pair_l3415_341503

theorem divisible_by_seven_pair : ∃! (x y : ℕ), x < 10 ∧ y < 10 ∧
  (1000 + 100 * x + 10 * y + 2) % 7 = 0 ∧
  (1000 * x + 120 + y) % 7 = 0 ∧
  x = 6 ∧ y = 5 := by sorry

end NUMINAMATH_CALUDE_divisible_by_seven_pair_l3415_341503


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3415_341572

theorem arithmetic_calculations : 
  ((1 : ℝ) + 4 - (-7) + (-8) = 3) ∧ 
  (-8.9 - (-4.7) + 7.5 = 3.3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3415_341572


namespace NUMINAMATH_CALUDE_choco_given_away_l3415_341564

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the number of dozens of oatmeal raisin cookies baked. -/
def oatmeal_baked : ℚ := 3

/-- Represents the number of dozens of sugar cookies baked. -/
def sugar_baked : ℚ := 2

/-- Represents the number of dozens of chocolate chip cookies baked. -/
def choco_baked : ℚ := 4

/-- Represents the number of dozens of oatmeal raisin cookies given away. -/
def oatmeal_given : ℚ := 2

/-- Represents the number of dozens of sugar cookies given away. -/
def sugar_given : ℚ := 3/2

/-- Represents the total number of cookies Ann keeps. -/
def cookies_kept : ℕ := 36

/-- Theorem stating the number of dozens of chocolate chip cookies given away. -/
theorem choco_given_away : 
  (oatmeal_baked * dozen + sugar_baked * dozen + choco_baked * dozen - 
   oatmeal_given * dozen - sugar_given * dozen - cookies_kept) / dozen = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_choco_given_away_l3415_341564


namespace NUMINAMATH_CALUDE_annas_initial_candies_l3415_341559

/-- Given that Anna receives some candies from Larry and ends up with a total number of candies,
    this theorem proves how many candies Anna started with. -/
theorem annas_initial_candies
  (candies_from_larry : ℕ)
  (total_candies : ℕ)
  (h1 : candies_from_larry = 86)
  (h2 : total_candies = 91)
  : total_candies - candies_from_larry = 5 := by
  sorry

end NUMINAMATH_CALUDE_annas_initial_candies_l3415_341559


namespace NUMINAMATH_CALUDE_correct_derivatives_l3415_341537

open Real

theorem correct_derivatives :
  (∀ x : ℝ, deriv (λ x => (2 * x) / (x^2 + 1)) x = (2 - 2 * x^2) / (x^2 + 1)^2) ∧
  (∀ x : ℝ, deriv (λ x => exp (3 * x + 1)) x = 3 * exp (3 * x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_correct_derivatives_l3415_341537


namespace NUMINAMATH_CALUDE_prob_at_least_two_same_l3415_341509

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of at least two dice showing the same number when rolling four fair 8-sided dice -/
theorem prob_at_least_two_same (num_sides : ℕ) (num_dice : ℕ) : 
  num_sides = 8 → num_dice = 4 → 
  (1 - (num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3)) / (num_sides ^ num_dice)) = 151/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_same_l3415_341509


namespace NUMINAMATH_CALUDE_two_never_appears_l3415_341536

/-- Represents a move in the game -/
def Move (s : List Int) : List Int :=
  -- Implementation details omitted
  sorry

/-- Represents the state of the board after any number of moves -/
inductive BoardState
| initial (n : Nat) : BoardState
| after_move (prev : BoardState) : BoardState

/-- The sequence of numbers on the board -/
def board_sequence (state : BoardState) : List Int :=
  match state with
  | BoardState.initial n => List.range (2*n) -- Simplified representation
  | BoardState.after_move prev => Move (board_sequence prev)

/-- Theorem stating that 2 never appears after any number of moves -/
theorem two_never_appears (n : Nat) (state : BoardState) : 
  2 ∉ board_sequence state :=
sorry

end NUMINAMATH_CALUDE_two_never_appears_l3415_341536


namespace NUMINAMATH_CALUDE_cone_slant_height_l3415_341524

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 5) (h2 : csa = 157.07963267948966) :
  csa / (Real.pi * r) = 10 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3415_341524


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_x_l3415_341596

/-- Proves that the percentage of alcohol by volume in solution x is 10% -/
theorem alcohol_percentage_solution_x :
  ∀ (x y : ℝ),
  y = 0.30 →
  450 * y + 300 * x = 0.22 * (450 + 300) →
  x = 0.10 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_x_l3415_341596


namespace NUMINAMATH_CALUDE_valid_selections_count_l3415_341528

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of ways to select 4 students from 7, where at least one of A and B participates,
    and when both participate, their speeches are not adjacent -/
def valid_selections : ℕ := sorry

theorem valid_selections_count : valid_selections = 600 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l3415_341528


namespace NUMINAMATH_CALUDE_square_last_digit_six_implies_second_last_odd_l3415_341504

theorem square_last_digit_six_implies_second_last_odd (n : ℕ) : 
  n^2 % 100 ≥ 6 ∧ n^2 % 100 < 16 → (n^2 / 10) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_last_digit_six_implies_second_last_odd_l3415_341504


namespace NUMINAMATH_CALUDE_sundae_price_l3415_341590

/-- Proves that the price of each sundae is $0.60 given the conditions of the catering order --/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) :
  ice_cream_bars = 200 →
  sundaes = 200 →
  total_price = 200 →
  ice_cream_price = 0.4 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.6 :=
by
  sorry

#eval (200 : ℚ) - 200 * 0.4  -- Expected output: 120
#eval 120 / 200              -- Expected output: 0.6

end NUMINAMATH_CALUDE_sundae_price_l3415_341590


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3415_341532

theorem sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hp_neq_3 : p ≠ 3) :
  ∃ a b c : ℕ, 4 * p^2 + 1 = a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3415_341532


namespace NUMINAMATH_CALUDE_a_fraction_is_one_third_l3415_341584

-- Define the partnership
structure Partnership :=
  (total_capital : ℝ)
  (a_fraction : ℝ)
  (b_fraction : ℝ)
  (c_fraction : ℝ)
  (d_fraction : ℝ)
  (total_profit : ℝ)
  (a_profit : ℝ)

-- Define the conditions
def partnership_conditions (p : Partnership) : Prop :=
  p.b_fraction = 1/4 ∧
  p.c_fraction = 1/5 ∧
  p.d_fraction = 1 - (p.a_fraction + p.b_fraction + p.c_fraction) ∧
  p.total_profit = 2445 ∧
  p.a_profit = 815 ∧
  p.a_profit / p.total_profit = p.a_fraction

-- Theorem statement
theorem a_fraction_is_one_third (p : Partnership) :
  partnership_conditions p → p.a_fraction = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_a_fraction_is_one_third_l3415_341584


namespace NUMINAMATH_CALUDE_keaton_ladder_climbs_l3415_341591

/-- Proves that Keaton climbed the ladder 20 times given the problem conditions -/
theorem keaton_ladder_climbs : 
  let keaton_ladder_height : ℕ := 30 * 12  -- 30 feet in inches
  let reece_ladder_height : ℕ := (30 - 4) * 12  -- 26 feet in inches
  let reece_climbs : ℕ := 15
  let total_length : ℕ := 11880  -- in inches
  ∃ (keaton_climbs : ℕ), 
    keaton_climbs * keaton_ladder_height + reece_climbs * reece_ladder_height = total_length ∧ 
    keaton_climbs = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_keaton_ladder_climbs_l3415_341591


namespace NUMINAMATH_CALUDE_james_black_spools_l3415_341555

/-- Represents the number of spools of yarn needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- Represents the number of spools of red yarn James has -/
def red_spools : ℕ := 12

/-- Represents the number of spools of blue yarn James has -/
def blue_spools : ℕ := 6

/-- Represents the number of berets James can make -/
def total_berets : ℕ := 11

/-- Calculates the number of black yarn spools James has -/
def black_spools : ℕ := 
  spools_per_beret * total_berets - (red_spools + blue_spools)

theorem james_black_spools : black_spools = 15 := by
  sorry

end NUMINAMATH_CALUDE_james_black_spools_l3415_341555


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l3415_341540

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (red_cards : Nat)
  (black_cards : Nat)

/-- The probability of drawing a red card first and then a black card -/
def probability_red_then_black (d : Deck) : Rat :=
  (d.red_cards * d.black_cards) / (d.cards * (d.cards - 1))

/-- Theorem statement for the probability in a standard 52-card deck -/
theorem probability_in_standard_deck :
  let d : Deck := ⟨52, 26, 26⟩
  probability_red_then_black d = 13 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l3415_341540


namespace NUMINAMATH_CALUDE_boys_without_calculators_l3415_341556

/-- Given a class with boys and girls, and information about calculator possession,
    prove that the number of boys without calculators is 5. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calc : ℕ)
  (girls_with_calc : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calc = 30)
  (h3 : girls_with_calc = 15) :
  total_boys - (total_with_calc - girls_with_calc) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l3415_341556


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3415_341599

/-- A function that generates the nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ :=
  10 * n - 5

/-- Theorem stating that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3415_341599


namespace NUMINAMATH_CALUDE_gross_profit_percentage_l3415_341512

theorem gross_profit_percentage 
  (selling_price : ℝ) 
  (wholesale_cost : ℝ) 
  (h1 : selling_price = 28) 
  (h2 : wholesale_cost = 25) : 
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 12 := by
sorry

end NUMINAMATH_CALUDE_gross_profit_percentage_l3415_341512


namespace NUMINAMATH_CALUDE_fraction_proof_l3415_341502

theorem fraction_proof (N : ℝ) (h1 : N = 150) (h2 : N - (3/5) * N = 60) : (3 : ℝ) / 5 = (3 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l3415_341502


namespace NUMINAMATH_CALUDE_max_value_nonnegative_inequality_condition_l3415_341526

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a

theorem max_value_nonnegative (a : ℝ) :
  ∀ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) → f a x₀ ≥ 0 := by sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x + Real.exp (x - 1) ≥ 1) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_max_value_nonnegative_inequality_condition_l3415_341526


namespace NUMINAMATH_CALUDE_log_difference_equals_negative_nine_l3415_341592

theorem log_difference_equals_negative_nine :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 81 / Real.log 3) = -9 := by
sorry

end NUMINAMATH_CALUDE_log_difference_equals_negative_nine_l3415_341592


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3415_341531

theorem modulus_of_complex_fraction (z : ℂ) :
  z = (5 : ℂ) / (1 - 2 * Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3415_341531


namespace NUMINAMATH_CALUDE_marbles_remaining_l3415_341549

theorem marbles_remaining (total : ℕ) (given_to_theresa : ℚ) (given_to_elliot : ℚ) :
  total = 100 →
  given_to_theresa = 25 / 100 →
  given_to_elliot = 10 / 100 →
  total - (total * given_to_theresa).floor - (total * given_to_elliot).floor = 65 := by
sorry

end NUMINAMATH_CALUDE_marbles_remaining_l3415_341549


namespace NUMINAMATH_CALUDE_probability_sum_six_three_dice_l3415_341508

/-- A function that returns the number of ways to roll a sum of 6 with three dice -/
def waysToRollSixWithThreeDice : ℕ :=
  -- We don't implement the function, just declare it
  sorry

/-- The total number of possible outcomes when rolling three six-sided dice -/
def totalOutcomes : ℕ := 6^3

/-- The probability of rolling a sum of 6 with three fair six-sided dice -/
theorem probability_sum_six_three_dice :
  (waysToRollSixWithThreeDice : ℚ) / totalOutcomes = 5 / 108 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_three_dice_l3415_341508


namespace NUMINAMATH_CALUDE_height_is_four_l3415_341525

/-- The configuration of squares with a small square of area 1 -/
structure SquareConfiguration where
  /-- The side length of the second square -/
  a : ℝ
  /-- The height to be determined -/
  h : ℝ
  /-- The small square has area 1 -/
  small_square_area : 1 = 1
  /-- The equation relating the squares and height -/
  square_relation : 1 + a + 3 = a + h

/-- The theorem stating that h = 4 in the given square configuration -/
theorem height_is_four (config : SquareConfiguration) : config.h = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_is_four_l3415_341525


namespace NUMINAMATH_CALUDE_exists_grid_with_partitions_l3415_341581

/-- A cell in the grid --/
structure Cell :=
  (x : Nat) (y : Nat)

/-- A shape in the grid --/
structure Shape :=
  (cells : List Cell)

/-- The grid --/
def Grid := List Cell

/-- Predicate to check if a shape is valid (contains 5 cells) --/
def isValidShape5 (s : Shape) : Prop :=
  s.cells.length = 5

/-- Predicate to check if a shape is valid (contains 4 cells) --/
def isValidShape4 (s : Shape) : Prop :=
  s.cells.length = 4

/-- Predicate to check if shapes are equal (up to rotation and flipping) --/
def areShapesEqual (s1 s2 : Shape) : Prop :=
  sorry  -- Implementation of shape equality check

/-- Theorem stating the existence of a grid with the required properties --/
theorem exists_grid_with_partitions :
  ∃ (g : Grid) (partition1 partition2 : List Shape),
    g.length = 20 ∧
    partition1.length = 4 ∧
    (∀ s ∈ partition1, isValidShape5 s) ∧
    (∀ i j, i < partition1.length → j < partition1.length → i ≠ j →
      areShapesEqual (partition1.get ⟨i, sorry⟩) (partition1.get ⟨j, sorry⟩)) ∧
    partition2.length = 5 ∧
    (∀ s ∈ partition2, isValidShape4 s) ∧
    (∀ i j, i < partition2.length → j < partition2.length → i ≠ j →
      areShapesEqual (partition2.get ⟨i, sorry⟩) (partition2.get ⟨j, sorry⟩)) :=
by
  sorry


end NUMINAMATH_CALUDE_exists_grid_with_partitions_l3415_341581


namespace NUMINAMATH_CALUDE_unique_integer_square_Q_l3415_341570

/-- Q is a function that maps an integer to an integer -/
def Q (x : ℤ) : ℤ := x^4 + 4*x^3 + 6*x^2 - x + 41

/-- There exists exactly one integer x such that Q(x) is a perfect square -/
theorem unique_integer_square_Q : ∃! x : ℤ, ∃ y : ℤ, Q x = y^2 := by sorry

end NUMINAMATH_CALUDE_unique_integer_square_Q_l3415_341570


namespace NUMINAMATH_CALUDE_calculator_game_sum_l3415_341583

/-- Represents the operation to be performed on a calculator --/
inductive Operation
  | Square
  | Negate

/-- Performs the specified operation on a number --/
def applyOperation (op : Operation) (x : Int) : Int :=
  match op with
  | Operation.Square => x * x
  | Operation.Negate => -x

/-- Determines the operation for the third calculator based on the pass number --/
def thirdOperation (pass : Nat) : Operation :=
  if pass % 2 = 0 then Operation.Negate else Operation.Square

/-- Performs one round of operations on the three calculators --/
def performRound (a b c : Int) (pass : Nat) : (Int × Int × Int) :=
  (applyOperation Operation.Square a,
   applyOperation Operation.Square b,
   applyOperation (thirdOperation pass) c)

/-- Performs n rounds of operations on the three calculators --/
def performNRounds (n : Nat) (a b c : Int) : (Int × Int × Int) :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => 
    let (a', b', c') := performRound a b c n
    performNRounds n a' b' c'

theorem calculator_game_sum :
  let (a, b, c) := performNRounds 50 1 0 (-1)
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l3415_341583


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l3415_341538

/-- The number of bacteria after a given number of 30-second intervals -/
def bacteria_count (initial : ℕ) (intervals : ℕ) : ℕ :=
  initial * 4^intervals

/-- The theorem stating the initial number of bacteria -/
theorem initial_bacteria_count : 
  ∃ (initial : ℕ), bacteria_count initial 8 = 1048576 ∧ initial = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l3415_341538


namespace NUMINAMATH_CALUDE_sqrt_4_equals_2_l3415_341506

theorem sqrt_4_equals_2 : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_equals_2_l3415_341506


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3415_341544

theorem trig_expression_equality : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let tan30 : ℝ := Real.sqrt 3 / 3
  let sin60 : ℝ := Real.sqrt 3 / 2
  4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3415_341544


namespace NUMINAMATH_CALUDE_abs_diff_squares_over_sum_squares_l3415_341507

theorem abs_diff_squares_over_sum_squares (a b : ℝ) 
  (h : (a * b) / (a^2 + b^2) = 1/4) : 
  |a^2 - b^2| / (a^2 + b^2) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_squares_over_sum_squares_l3415_341507


namespace NUMINAMATH_CALUDE_exactly_two_valid_pairs_l3415_341574

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def is_valid_pair (n m : ℕ+) : Prop := sum_factorials n.val = m.val ^ 2

theorem exactly_two_valid_pairs :
  ∃! (s : Finset (ℕ+ × ℕ+)), s.card = 2 ∧ ∀ (p : ℕ+ × ℕ+), p ∈ s ↔ is_valid_pair p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_pairs_l3415_341574


namespace NUMINAMATH_CALUDE_function_composition_equality_l3415_341548

theorem function_composition_equality (b : ℝ) (h1 : b > 0) : 
  let g : ℝ → ℝ := λ x ↦ b * x^2 - Real.cos (π * x)
  g (g 1) = -Real.cos π → b = 1 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3415_341548


namespace NUMINAMATH_CALUDE_shyne_plants_l3415_341541

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow -/
def total_plants : ℕ := eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets

theorem shyne_plants : total_plants = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l3415_341541


namespace NUMINAMATH_CALUDE_chocolates_difference_l3415_341553

theorem chocolates_difference (robert_chocolates nickel_chocolates : ℕ) 
  (h1 : robert_chocolates = 7)
  (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 := by
sorry

end NUMINAMATH_CALUDE_chocolates_difference_l3415_341553


namespace NUMINAMATH_CALUDE_equation_solution_l3415_341560

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 16) - (8 * Real.cos (π / 6)) / Real.sqrt (x + 16) = 4) ∧
  (x = (2 + 2 * Real.sqrt (1 + Real.sqrt 3))^2 - 16) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3415_341560


namespace NUMINAMATH_CALUDE_conditional_probability_coin_flips_l3415_341589

-- Define the sample space for two coin flips
def CoinFlip := Bool × Bool

-- Define the probability measure
noncomputable def P : Set CoinFlip → ℝ := sorry

-- Define event A: heads on the first flip
def A : Set CoinFlip := {x | x.1 = true}

-- Define event B: heads on the second flip
def B : Set CoinFlip := {x | x.2 = true}

-- Define the intersection of events A and B
def AB : Set CoinFlip := A ∩ B

-- State the theorem
theorem conditional_probability_coin_flips :
  P B / P A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_conditional_probability_coin_flips_l3415_341589


namespace NUMINAMATH_CALUDE_greatest_power_of_8_dividing_20_factorial_l3415_341586

theorem greatest_power_of_8_dividing_20_factorial :
  (∃ n : ℕ+, 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) ∧
  (∃ n : ℕ+, n.val = 6 ∧ 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_8_dividing_20_factorial_l3415_341586


namespace NUMINAMATH_CALUDE_thomas_blocks_total_l3415_341568

theorem thomas_blocks_total (stack1 stack2 stack3 stack4 stack5 : ℕ) : 
  stack1 = 7 →
  stack2 = stack1 + 3 →
  stack3 = stack2 - 6 →
  stack4 = stack3 + 10 →
  stack5 = 2 * stack2 →
  stack1 + stack2 + stack3 + stack4 + stack5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_thomas_blocks_total_l3415_341568


namespace NUMINAMATH_CALUDE_function_satisfies_condition_l3415_341500

/-- The function f(x) = 1/x - x satisfies the given condition for all x₁, x₂ in (0, +∞) where x₁ ≠ x₂ -/
theorem function_satisfies_condition :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ →
  (x₁ - x₂) * ((1 / x₁ - x₁) - (1 / x₂ - x₂)) < 0 := by
  sorry


end NUMINAMATH_CALUDE_function_satisfies_condition_l3415_341500


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3415_341565

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_derivative_at_2 : 
  deriv f 2 = (1 - Real.log 2) / 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3415_341565


namespace NUMINAMATH_CALUDE_quadratic_function_coefficients_l3415_341522

/-- Given a quadratic function f(x) = 2(x-3)^2 + 2, prove that it can be expressed
    as ax^2 + bx + c where a = 2, b = -12, and c = 20 -/
theorem quadratic_function_coefficients :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 2 * (x - 3)^2 + 2) ∧
    (∃ (a b c : ℝ), a = 2 ∧ b = -12 ∧ c = 20 ∧ 
      ∀ x, f x = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_coefficients_l3415_341522


namespace NUMINAMATH_CALUDE_tianjin_population_scientific_notation_l3415_341516

/-- The population of Tianjin -/
def tianjin_population : ℕ := 13860000

/-- Scientific notation representation of Tianjin's population -/
def tianjin_scientific : ℝ := 1.386 * (10 ^ 7)

/-- Theorem stating that the population of Tianjin in scientific notation is correct -/
theorem tianjin_population_scientific_notation :
  (tianjin_population : ℝ) = tianjin_scientific :=
by sorry

end NUMINAMATH_CALUDE_tianjin_population_scientific_notation_l3415_341516


namespace NUMINAMATH_CALUDE_prime_sequence_extension_l3415_341519

theorem prime_sequence_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_extension_l3415_341519


namespace NUMINAMATH_CALUDE_no_distinct_positive_roots_l3415_341552

theorem no_distinct_positive_roots :
  ∀ (b c : ℤ), 0 ≤ b ∧ b ≤ 5 ∧ -10 ≤ c ∧ c ≤ 10 →
  ¬∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + b * x₁ + c = 0 ∧ x₂^2 + b * x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_positive_roots_l3415_341552


namespace NUMINAMATH_CALUDE_flower_bed_path_area_l3415_341527

/-- The area of a circular ring around a flower bed -/
theorem flower_bed_path_area (circumference : Real) (path_width : Real) : 
  circumference = 314 → path_width = 2 →
  let inner_radius := circumference / (2 * Real.pi)
  let outer_radius := inner_radius + path_width
  abs (Real.pi * (outer_radius^2 - inner_radius^2) - 640.56) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_flower_bed_path_area_l3415_341527


namespace NUMINAMATH_CALUDE_no_valid_equation_l3415_341517

/-- Represents a letter in the equation -/
structure Letter where
  value : Nat
  property : value < 10

/-- Represents a two-digit number as a pair of letters -/
structure TwoDigitNumber where
  tens : Letter
  ones : Letter
  different : tens ≠ ones

/-- Represents the equation АБ×ВГ = ДДЕЕ -/
structure Equation where
  ab : TwoDigitNumber
  vg : TwoDigitNumber
  d : Letter
  e : Letter
  different_letters : ab.tens ≠ ab.ones ∧ ab.tens ≠ vg.tens ∧ ab.tens ≠ vg.ones ∧
                      ab.ones ≠ vg.tens ∧ ab.ones ≠ vg.ones ∧ vg.tens ≠ vg.ones ∧
                      d ≠ e
  valid_multiplication : ab.tens.value * 10 + ab.ones.value *
                         (vg.tens.value * 10 + vg.ones.value) =
                         d.value * 1000 + d.value * 100 + e.value * 10 + e.value

theorem no_valid_equation : ¬ ∃ (eq : Equation), True := by
  sorry

end NUMINAMATH_CALUDE_no_valid_equation_l3415_341517


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l3415_341554

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < m + 2}
def B : Set ℝ := {x | -4 < x ∧ x < 4}

-- Theorem for part (I)
theorem intersection_and_union_when_m_equals_3 :
  (A 3 ∩ B = {x | 1 < x ∧ x < 4}) ∧
  (A 3 ∪ B = {x | -4 < x ∧ x < 5}) := by
  sorry

-- Theorem for part (II)
theorem subset_of_complement_iff_m_in_range :
  ∀ m : ℝ, A m ⊆ Bᶜ ↔ m ≤ -6 ∨ m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_equals_3_subset_of_complement_iff_m_in_range_l3415_341554


namespace NUMINAMATH_CALUDE_cycling_route_length_l3415_341595

/-- The total length of a rectangular cycling route -/
def total_length (upper_horizontal : ℝ) (left_vertical : ℝ) : ℝ :=
  2 * (upper_horizontal + left_vertical)

/-- Theorem: The total length of the cycling route is 52 km -/
theorem cycling_route_length :
  let upper_horizontal := 4 + 7 + 2
  let left_vertical := 6 + 7
  total_length upper_horizontal left_vertical = 52 := by
  sorry

end NUMINAMATH_CALUDE_cycling_route_length_l3415_341595


namespace NUMINAMATH_CALUDE_toy_price_problem_l3415_341598

theorem toy_price_problem (num_toys : ℕ) (sixth_toy_price : ℝ) (new_average : ℝ) :
  num_toys = 5 →
  sixth_toy_price = 16 →
  new_average = 11 →
  (num_toys : ℝ) * (num_toys + 1 : ℝ)⁻¹ * (num_toys * new_average - sixth_toy_price) = 10 :=
by sorry

end NUMINAMATH_CALUDE_toy_price_problem_l3415_341598


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3415_341579

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  B = π / 4 →
  (1 / 2) * a * b * Real.sin C = 2 →
  b = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3415_341579


namespace NUMINAMATH_CALUDE_blocks_shared_l3415_341546

theorem blocks_shared (start_blocks end_blocks : ℝ) (h1 : start_blocks = 86.0) (h2 : end_blocks = 127) : 
  end_blocks - start_blocks = 41 := by
sorry

end NUMINAMATH_CALUDE_blocks_shared_l3415_341546


namespace NUMINAMATH_CALUDE_first_shirt_costs_15_l3415_341518

/-- The cost of the first shirt given the conditions of the problem -/
def first_shirt_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + 6

/-- The total cost of both shirts -/
def total_cost (second_shirt_cost : ℝ) : ℝ :=
  second_shirt_cost + first_shirt_cost second_shirt_cost

theorem first_shirt_costs_15 :
  ∃ (second_shirt_cost : ℝ),
    first_shirt_cost second_shirt_cost = 15 ∧
    total_cost second_shirt_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_first_shirt_costs_15_l3415_341518


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3415_341542

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (s : ℝ) (r : ℝ) 
  (h_s : s = 10) (h_r : r = 3) : 
  let square_area := s^2
  let triangle_area := 8 * (1/2 * s/2 * (r * Real.sqrt 3))
  let sector_area := 4 * (1/12 * Real.pi * r^2)
  square_area - triangle_area - sector_area = 100 - 60 * Real.sqrt 3 - 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l3415_341542


namespace NUMINAMATH_CALUDE_final_number_independent_of_operations_l3415_341582

/-- Represents the state of the blackboard with counts of 0, 1, and 2 -/
structure Board :=
  (count_0 : ℕ)
  (count_1 : ℕ)
  (count_2 : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | replace_0_1_with_2
  | replace_1_2_with_0
  | replace_0_2_with_1

/-- Applies an operation to the board -/
def apply_operation (b : Board) (op : Operation) : Board :=
  match op with
  | Operation.replace_0_1_with_2 => ⟨b.count_0 - 1, b.count_1 - 1, b.count_2 + 1⟩
  | Operation.replace_1_2_with_0 => ⟨b.count_0 + 1, b.count_1 - 1, b.count_2 - 1⟩
  | Operation.replace_0_2_with_1 => ⟨b.count_0 - 1, b.count_1 + 1, b.count_2 - 1⟩

/-- Checks if the board has only one number left -/
def is_final (b : Board) : Prop :=
  (b.count_0 = 1 ∧ b.count_1 = 0 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 1 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 0 ∧ b.count_2 = 1)

/-- The final number on the board -/
def final_number (b : Board) : ℕ :=
  if b.count_0 = 1 then 0
  else if b.count_1 = 1 then 1
  else 2

/-- Theorem: The final number is determined by initial parity, regardless of operations -/
theorem final_number_independent_of_operations (initial : Board) 
  (ops1 ops2 : List Operation) (h1 : is_final (ops1.foldl apply_operation initial))
  (h2 : is_final (ops2.foldl apply_operation initial)) :
  final_number (ops1.foldl apply_operation initial) = 
  final_number (ops2.foldl apply_operation initial) :=
sorry

end NUMINAMATH_CALUDE_final_number_independent_of_operations_l3415_341582


namespace NUMINAMATH_CALUDE_tenth_term_value_l3415_341575

def sequence_term (n : ℕ+) : ℚ :=
  (-1)^(n + 1 : ℕ) * (2 * n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem tenth_term_value : sequence_term 10 = -19 / 101 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l3415_341575


namespace NUMINAMATH_CALUDE_trees_died_in_typhoon_l3415_341513

theorem trees_died_in_typhoon (initial_trees : ℕ) (remaining_trees : ℕ) : 
  initial_trees = 20 → remaining_trees = 4 → initial_trees - remaining_trees = 16 := by
  sorry

end NUMINAMATH_CALUDE_trees_died_in_typhoon_l3415_341513


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l3415_341530

theorem smallest_four_digit_divisible_by_43 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 → n ≥ 1032 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_43_l3415_341530


namespace NUMINAMATH_CALUDE_beach_waders_l3415_341521

/-- Proves that 3 people from the first row got up to wade in the water, given the conditions of the beach scenario. -/
theorem beach_waders (first_row : ℕ) (second_row : ℕ) (third_row : ℕ) 
  (h1 : first_row = 24)
  (h2 : second_row = 20)
  (h3 : third_row = 18)
  (h4 : ∃ x : ℕ, first_row - x + (second_row - 5) + third_row = 54) :
  ∃ x : ℕ, x = 3 ∧ first_row - x + (second_row - 5) + third_row = 54 :=
by sorry

end NUMINAMATH_CALUDE_beach_waders_l3415_341521


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3415_341561

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

/-- Theorem: The number of diagonals in a nonagon is 27 -/
theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3415_341561


namespace NUMINAMATH_CALUDE_pig_price_calculation_l3415_341520

/-- Given the total cost of 3 pigs and 10 hens, and the average price of a hen,
    calculate the average price of a pig. -/
theorem pig_price_calculation (total_cost hen_price : ℚ) 
    (h1 : total_cost = 1200)
    (h2 : hen_price = 30) : 
    (total_cost - 10 * hen_price) / 3 = 300 :=
by sorry

end NUMINAMATH_CALUDE_pig_price_calculation_l3415_341520
