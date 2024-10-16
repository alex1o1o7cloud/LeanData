import Mathlib

namespace NUMINAMATH_CALUDE_max_whole_nine_one_number_l2598_259862

def is_whole_nine_one_number (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 4 ∧
  1 ≤ b ∧ b ≤ 9 ∧
  1 ≤ c ∧ c ≤ 9 ∧
  1 ≤ d ∧ d ≤ 9 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  ∃ k : ℕ, k * (2 * b + c) = 4 * a + 2 * d

def M (a b c d : ℕ) : ℕ :=
  2000 * a + 100 * b + 10 * c + d

theorem max_whole_nine_one_number :
  ∀ a b c d : ℕ,
    is_whole_nine_one_number a b c d →
    M a b c d ≤ 7524 :=
  sorry

end NUMINAMATH_CALUDE_max_whole_nine_one_number_l2598_259862


namespace NUMINAMATH_CALUDE_prism_volume_l2598_259804

/-- The volume of a right rectangular prism with face areas 10, 14, and 35 square inches is 70 cubic inches. -/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 10) 
  (area2 : w * h = 14) 
  (area3 : l * h = 35) : 
  l * w * h = 70 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2598_259804


namespace NUMINAMATH_CALUDE_polyhedron_edge_intersection_l2598_259821

/-- A polyhedron with a specified number of edges. -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron. -/
structure IntersectingPlane where
  intersected_edges : ℕ

/-- Theorem about the maximum number of intersected edges for different types of polyhedra. -/
theorem polyhedron_edge_intersection
  (p : Polyhedron)
  (h : p.edges = 100) :
  ∃ (convex_max non_convex_max : ℕ),
    -- For a convex polyhedron, the maximum number of intersected edges is 66
    (∀ (plane : IntersectingPlane), plane.intersected_edges ≤ convex_max) ∧
    convex_max = 66 ∧
    -- For a non-convex polyhedron, there exists a configuration where 96 edges can be intersected
    (∃ (plane : IntersectingPlane), plane.intersected_edges = non_convex_max) ∧
    non_convex_max = 96 ∧
    -- For any polyhedron, it's impossible to intersect all 100 edges
    (∀ (plane : IntersectingPlane), plane.intersected_edges < p.edges) :=
by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edge_intersection_l2598_259821


namespace NUMINAMATH_CALUDE_extreme_value_of_f_l2598_259882

-- Define the function f
def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the solution set condition
def solution_set (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f x < 0 ↔ (x < m + 1 ∧ x ≠ m)

-- Theorem statement
theorem extreme_value_of_f (a b c m : ℝ) :
  solution_set (f · a b c) m →
  ∃ x, f x a b c = -4/27 ∧ ∀ y, f y a b c ≥ -4/27 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_of_f_l2598_259882


namespace NUMINAMATH_CALUDE_group_size_proof_l2598_259861

theorem group_size_proof (total_paise : ℕ) (h : total_paise = 4624) :
  ∃ n : ℕ, n * n = total_paise ∧ n = 68 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l2598_259861


namespace NUMINAMATH_CALUDE_right_triangle_area_l2598_259847

/-- Given a right-angled triangle with perimeter 18 and sum of squares of side lengths 128, its area is 9. -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 18 →
  a^2 + b^2 + c^2 = 128 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 9 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2598_259847


namespace NUMINAMATH_CALUDE_megan_total_songs_l2598_259826

/-- Calculates the total number of songs bought given the initial number of albums,
    the number of albums removed, and the number of songs per album. -/
def total_songs (initial_albums : ℕ) (removed_albums : ℕ) (songs_per_album : ℕ) : ℕ :=
  (initial_albums - removed_albums) * songs_per_album

/-- Proves that the total number of songs bought is correct for Megan's scenario. -/
theorem megan_total_songs :
  total_songs 8 2 7 = 42 :=
by sorry

end NUMINAMATH_CALUDE_megan_total_songs_l2598_259826


namespace NUMINAMATH_CALUDE_power_equation_solution_l2598_259897

theorem power_equation_solution (m : ℕ) : 8^36 * 6^21 = 3 * 24^m → m = 43 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2598_259897


namespace NUMINAMATH_CALUDE_fraction_equality_l2598_259830

theorem fraction_equality (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2598_259830


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2598_259816

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 + 4 * x + 1 < 0) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2598_259816


namespace NUMINAMATH_CALUDE_subtract_from_forty_squared_l2598_259855

theorem subtract_from_forty_squared (n : ℕ) (h : n = 40 - 1) : n^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_forty_squared_l2598_259855


namespace NUMINAMATH_CALUDE_min_trips_required_l2598_259881

def trays_per_trip : ℕ := 9
def trays_table1 : ℕ := 17
def trays_table2 : ℕ := 55

def total_trays : ℕ := trays_table1 + trays_table2

theorem min_trips_required : (total_trays + trays_per_trip - 1) / trays_per_trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_trips_required_l2598_259881


namespace NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l2598_259889

/-- Given a chef cooking potatoes with the following conditions:
  - The total number of potatoes to cook is 15
  - The number of potatoes already cooked is 6
  - Each potato takes 8 minutes to cook
  This theorem proves that the time required to cook the remaining potatoes is 72 minutes. -/
theorem cooking_time_for_remaining_potatoes :
  let total_potatoes : ℕ := 15
  let cooked_potatoes : ℕ := 6
  let cooking_time_per_potato : ℕ := 8
  let remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
  remaining_potatoes * cooking_time_per_potato = 72 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_for_remaining_potatoes_l2598_259889


namespace NUMINAMATH_CALUDE_min_distance_after_nine_minutes_l2598_259818

/-- Represents the robot's position on a 2D grid -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the possible directions the robot can face -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a turn the robot can make -/
inductive Turn
  | Left
  | Right

/-- The distance the robot travels in one minute -/
def distancePerMinute : ℕ := 10

/-- The total number of minutes the robot moves -/
def totalMinutes : ℕ := 9

/-- A function to calculate the Manhattan distance between two positions -/
def manhattanDistance (p1 p2 : Position) : ℕ :=
  (Int.natAbs (p1.x - p2.x)) + (Int.natAbs (p1.y - p2.y))

/-- A function to simulate the robot's movement for one minute -/
def moveOneMinute (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.North => { x := pos.x, y := pos.y + distancePerMinute }
  | Direction.East => { x := pos.x + distancePerMinute, y := pos.y }
  | Direction.South => { x := pos.x, y := pos.y - distancePerMinute }
  | Direction.West => { x := pos.x - distancePerMinute, y := pos.y }

/-- The theorem stating that the minimum distance from the starting point after 9 minutes is 10 meters -/
theorem min_distance_after_nine_minutes :
  ∃ (finalPos : Position),
    manhattanDistance { x := 0, y := 0 } finalPos = distancePerMinute ∧
    (∀ (pos : Position),
      (∃ (moves : List Turn),
        moves.length = totalMinutes - 1 ∧
        -- First move is always East
        (let firstMove := moveOneMinute { x := 0, y := 0 } Direction.East
        -- Subsequent moves can involve turns
        let finalPosition := moves.foldl
          (fun acc turn =>
            let newDir := match turn with
              | Turn.Left => Direction.North  -- Simplified turn logic
              | Turn.Right => Direction.South
            moveOneMinute acc newDir)
          firstMove
        finalPosition = pos)) →
      manhattanDistance { x := 0, y := 0 } pos ≥ distancePerMinute) :=
sorry

end NUMINAMATH_CALUDE_min_distance_after_nine_minutes_l2598_259818


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2598_259886

/-- Given 6 people with an average weight of 156 lbs, if a 7th person enters and
    the new average weight becomes 151 lbs, then the weight of the 7th person is 121 lbs. -/
theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (seventh_person_weight : ℝ) :
  initial_people = 6 →
  initial_avg_weight = 156 →
  new_avg_weight = 151 →
  (initial_people * initial_avg_weight + seventh_person_weight) / (initial_people + 1) = new_avg_weight →
  seventh_person_weight = 121 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2598_259886


namespace NUMINAMATH_CALUDE_train_crossing_time_l2598_259811

theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 → 
  train_speed_kmh = 36 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2598_259811


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2598_259895

theorem inequality_solution_set : 
  {x : ℝ | x + 2 < 1} = {x : ℝ | x < -1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2598_259895


namespace NUMINAMATH_CALUDE_total_candies_in_store_l2598_259825

def chocolate_boxes : List Nat := [200, 320, 500, 500, 768, 768]
def candy_tubs : List Nat := [1380, 1150, 1150, 1720]

theorem total_candies_in_store : 
  (chocolate_boxes.sum + candy_tubs.sum) = 8456 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_in_store_l2598_259825


namespace NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l2598_259883

theorem tangent_lines_parallel_to_given_line :
  ∃ (x₁ x₂ : ℝ),
    -- The two points lie on the curve
    x₁^3 + x₁ - 2 = 4 * x₁ - 4 ∧
    x₂^3 + x₂ - 2 = 4 * x₂ ∧
    -- The derivative at these points equals the slope of the given line
    3 * x₁^2 + 1 = 4 ∧
    3 * x₂^2 + 1 = 4 ∧
    -- The tangent lines are different
    x₁ ≠ x₂ :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_parallel_to_given_line_l2598_259883


namespace NUMINAMATH_CALUDE_triple_equality_l2598_259815

theorem triple_equality (a b c : ℝ) : 
  a * (b^2 + c) = c * (c + a * b) →
  b * (c^2 + a) = a * (a + b * c) →
  c * (a^2 + b) = b * (b + c * a) →
  a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_triple_equality_l2598_259815


namespace NUMINAMATH_CALUDE_remainder_two_power_200_minus_3_mod_7_l2598_259863

theorem remainder_two_power_200_minus_3_mod_7 : 
  (2^200 - 3) % 7 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_two_power_200_minus_3_mod_7_l2598_259863


namespace NUMINAMATH_CALUDE_least_product_of_reciprocal_sum_l2598_259807

theorem least_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (9 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (9 : ℚ)⁻¹ → (a * b : ℕ) ≤ (c * d : ℕ)) ∧ 
  (a * b : ℕ) = 144 := by
sorry

end NUMINAMATH_CALUDE_least_product_of_reciprocal_sum_l2598_259807


namespace NUMINAMATH_CALUDE_hexagon_side_length_l2598_259887

/-- The length of one side of a regular hexagon with perimeter 43.56 -/
theorem hexagon_side_length : ∃ (s : ℝ), s > 0 ∧ s * 6 = 43.56 ∧ s = 7.26 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l2598_259887


namespace NUMINAMATH_CALUDE_output_is_27_l2598_259854

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 ≤ 22 then
    step1 + 8
  else
    step1 + 3

theorem output_is_27 : function_machine 12 = 27 := by
  sorry

end NUMINAMATH_CALUDE_output_is_27_l2598_259854


namespace NUMINAMATH_CALUDE_bicycle_not_in_motion_time_l2598_259872

-- Define the constants
def total_distance : ℝ := 22.5
def bert_ride_speed : ℝ := 8
def bert_walk_speed : ℝ := 5
def al_walk_speed : ℝ := 4
def al_ride_speed : ℝ := 10

-- Define the theorem
theorem bicycle_not_in_motion_time :
  ∃ (x : ℝ),
    (x / bert_ride_speed + (total_distance - x) / bert_walk_speed =
     x / al_walk_speed + (total_distance - x) / al_ride_speed) ∧
    ((x / al_walk_speed - x / bert_ride_speed) * 60 = 75) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_not_in_motion_time_l2598_259872


namespace NUMINAMATH_CALUDE_parabola_slope_AF_l2598_259845

-- Define the parabola
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem parabola_slope_AF (C : Parabola) (A F : Point) :
  A.x = -2 ∧ A.y = 3 ∧  -- A is (-2, 3)
  A.x = -C.p/2 ∧        -- A is on the directrix
  F.x = C.p/2 ∧ F.y = 0 -- F is the focus
  →
  (F.y - A.y) / (F.x - A.x) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_slope_AF_l2598_259845


namespace NUMINAMATH_CALUDE_imaginary_unit_calculation_l2598_259838

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_calculation : i * (1 + i)^2 = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_unit_calculation_l2598_259838


namespace NUMINAMATH_CALUDE_remainder_theorem_l2598_259878

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') : 
  P % (2 * D * D') = D * R' + R := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2598_259878


namespace NUMINAMATH_CALUDE_candy_box_price_l2598_259874

/-- Proves that the current price of a candy box is 15 pounds given the conditions -/
theorem candy_box_price (
  soda_price : ℝ)
  (candy_increase : ℝ)
  (soda_increase : ℝ)
  (original_total : ℝ)
  (h1 : soda_price = 6)
  (h2 : candy_increase = 0.25)
  (h3 : soda_increase = 0.50)
  (h4 : original_total = 16) :
  ∃ (candy_price : ℝ), candy_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_price_l2598_259874


namespace NUMINAMATH_CALUDE_shortest_side_length_l2598_259852

/-- A triangle with an inscribed circle -/
structure InscribedCircleTriangle where
  /-- The length of the first segment of the partitioned side -/
  segment1 : ℝ
  /-- The length of the second segment of the partitioned side -/
  segment2 : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The angle opposite the partitioned side in radians -/
  opposite_angle : ℝ

/-- The theorem stating the length of the shortest side of the triangle -/
theorem shortest_side_length (t : InscribedCircleTriangle)
  (h1 : t.segment1 = 7)
  (h2 : t.segment2 = 9)
  (h3 : t.radius = 5)
  (h4 : t.opposite_angle = π / 3) :
  ∃ (shortest_side : ℝ), shortest_side = 20 * (2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_length_l2598_259852


namespace NUMINAMATH_CALUDE_tennis_racket_weight_tennis_racket_weight_proof_l2598_259809

theorem tennis_racket_weight : ℝ → ℝ → Prop :=
  fun (racket_weight bicycle_weight : ℝ) =>
    (10 * racket_weight = 8 * bicycle_weight) →
    (4 * bicycle_weight = 120) →
    racket_weight = 24

-- Proof
theorem tennis_racket_weight_proof :
  ∃ (racket_weight bicycle_weight : ℝ),
    tennis_racket_weight racket_weight bicycle_weight :=
by
  sorry

end NUMINAMATH_CALUDE_tennis_racket_weight_tennis_racket_weight_proof_l2598_259809


namespace NUMINAMATH_CALUDE_boys_without_calculators_l2598_259857

/-- Proves that the number of boys who didn't bring calculators is 8 -/
theorem boys_without_calculators (total_students : ℕ) (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ)
  (h1 : total_students = 40)
  (h2 : total_boys = 20)
  (h3 : students_with_calculators = 30)
  (h4 : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 8 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l2598_259857


namespace NUMINAMATH_CALUDE_line_slope_l2598_259836

/-- The slope of the line x - √3y + 1 = 0 is 1/√3 -/
theorem line_slope (x y : ℝ) : x - Real.sqrt 3 * y + 1 = 0 → (y - 1) / x = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2598_259836


namespace NUMINAMATH_CALUDE_tom_barbados_trip_cost_l2598_259846

/-- The total cost for Tom's trip to Barbados -/
def total_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (trip_cost : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  out_of_pocket_medical + trip_cost

/-- Theorem stating the total cost for Tom's trip to Barbados -/
theorem tom_barbados_trip_cost :
  total_cost 10 45 250 0.8 1200 = 1340 := by
  sorry

end NUMINAMATH_CALUDE_tom_barbados_trip_cost_l2598_259846


namespace NUMINAMATH_CALUDE_strawberry_sugar_purchase_strategy_l2598_259860

theorem strawberry_sugar_purchase_strategy :
  -- Define constants
  let discount_threshold : ℝ := 1000
  let discount_rate : ℝ := 0.5
  let budget : ℝ := 1200
  let strawberry_price : ℝ := 300
  let sugar_price : ℝ := 30
  let strawberry_amount : ℝ := 4
  let sugar_amount : ℝ := 6

  -- Define purchase strategy
  let first_purchase_strawberry : ℝ := 3
  let first_purchase_sugar : ℝ := 4
  let second_purchase_strawberry : ℝ := strawberry_amount - first_purchase_strawberry
  let second_purchase_sugar : ℝ := sugar_amount - first_purchase_sugar

  -- Calculate costs
  let first_purchase_cost : ℝ := first_purchase_strawberry * strawberry_price + first_purchase_sugar * sugar_price
  let second_purchase_full_price : ℝ := second_purchase_strawberry * strawberry_price + second_purchase_sugar * sugar_price
  let second_purchase_discounted : ℝ := second_purchase_full_price * (1 - discount_rate)
  let total_cost : ℝ := first_purchase_cost + second_purchase_discounted

  -- Theorem statement
  (first_purchase_cost ≥ discount_threshold) →
  (total_cost ≤ budget) ∧
  (first_purchase_strawberry + second_purchase_strawberry = strawberry_amount) ∧
  (first_purchase_sugar + second_purchase_sugar = sugar_amount) :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sugar_purchase_strategy_l2598_259860


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2598_259810

/-- The distance between two people walking in opposite directions for 2 hours -/
theorem distance_after_two_hours 
  (jay_speed : ℝ) 
  (paul_speed : ℝ) 
  (h1 : jay_speed = 0.8 / 15) 
  (h2 : paul_speed = 3 / 30) 
  : jay_speed * 120 + paul_speed * 120 = 18.4 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l2598_259810


namespace NUMINAMATH_CALUDE_expression_always_positive_l2598_259849

theorem expression_always_positive (a b : ℝ) : a^2 + b^2 + 4*b - 2*a + 6 > 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_always_positive_l2598_259849


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2598_259898

theorem sum_of_roots_cubic_equation : 
  let p (x : ℝ) := 3 * x^3 - 15 * x^2 - 36 * x + 7
  ∃ r s t : ℝ, (p r = 0 ∧ p s = 0 ∧ p t = 0) ∧ (r + s + t = 5) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l2598_259898


namespace NUMINAMATH_CALUDE_batsman_80_run_innings_l2598_259812

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding a score -/
def newAverage (stats : BatsmanStats) (score : ℕ) : ℚ :=
  (stats.totalRuns + score) / (stats.innings + 1)

theorem batsman_80_run_innings :
  ∀ (stats : BatsmanStats),
    stats.average = 46 →
    newAverage stats 80 = 48 →
    stats.innings = 16 :=
by sorry

end NUMINAMATH_CALUDE_batsman_80_run_innings_l2598_259812


namespace NUMINAMATH_CALUDE_exactly_two_approve_probability_l2598_259822

def approval_rate : ℝ := 0.8
def num_voters : ℕ := 4
def num_approving : ℕ := 2

def probability_exactly_two_approve : ℝ := 
  (Nat.choose num_voters num_approving) * (approval_rate ^ num_approving) * ((1 - approval_rate) ^ (num_voters - num_approving))

theorem exactly_two_approve_probability :
  probability_exactly_two_approve = 0.1536 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_approve_probability_l2598_259822


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l2598_259834

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 10*z^5 + A*z^4 + (-88)*z^3 + C*z^2 + D*z + 16 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 10 := by
  sorry

#check polynomial_coefficient_B

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l2598_259834


namespace NUMINAMATH_CALUDE_parallelogram_symmetry_l2598_259884

-- Define a parallelogram
structure Parallelogram where
  -- Add necessary fields to define a parallelogram
  -- This is a simplified representation
  centrally_symmetric : Bool
  axially_symmetric : Bool

-- Theorem statement
theorem parallelogram_symmetry (p : Parallelogram) : p.centrally_symmetric ∧ ¬p.axially_symmetric := by
  sorry

-- Note: The actual implementation would require more detailed definitions and properties of parallelograms

end NUMINAMATH_CALUDE_parallelogram_symmetry_l2598_259884


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l2598_259829

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := 2 * Q.quadrilateral_faces
  total_line_segments - Q.edges - face_diagonals

/-- The specific polyhedron Q described in the problem -/
def Q : ConvexPolyhedron :=
  { vertices := 30
  , edges := 72
  , faces := 44
  , triangular_faces := 30
  , quadrilateral_faces := 14 }

theorem space_diagonals_of_Q :
  space_diagonals Q = 335 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l2598_259829


namespace NUMINAMATH_CALUDE_volunteer_hours_per_time_l2598_259802

/-- The number of times John volunteers per month -/
def volunteering_frequency : ℕ := 2

/-- The total number of hours John volunteers per year -/
def total_hours_per_year : ℕ := 72

/-- The number of months in a year -/
def months_per_year : ℕ := 12

/-- Theorem stating how many hours John volunteers at a time -/
theorem volunteer_hours_per_time :
  total_hours_per_year / (volunteering_frequency * months_per_year) = 3 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_hours_per_time_l2598_259802


namespace NUMINAMATH_CALUDE_three_true_propositions_l2598_259800

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_true_propositions
  (a : Line3D) (α β : Plane3D) (h_diff : α ≠ β) :
  (perpendicular a α ∧ perpendicular a β → parallel α β) ∧
  (perpendicular a α ∧ parallel α β → perpendicular a β) ∧
  (perpendicular a β ∧ parallel α β → perpendicular a α) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l2598_259800


namespace NUMINAMATH_CALUDE_two_flies_problem_l2598_259885

/-- Two flies crawling on a wall problem -/
theorem two_flies_problem (d v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t1 := 2 * d / v
  let t2 := 5 * d / (2 * v)
  let avg_speed1 := 2 * d / t1
  let avg_speed2 := 2 * d / t2
  t1 < t2 ∧ avg_speed1 > avg_speed2 := by
  sorry

#check two_flies_problem

end NUMINAMATH_CALUDE_two_flies_problem_l2598_259885


namespace NUMINAMATH_CALUDE_inequality_proof_l2598_259820

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥
  2 * a / (b + c) + 2 * b / (c + a) + 2 * c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2598_259820


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l2598_259896

/-- A positive geometric sequence satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 1 / a 0
  condition : a 8 = a 6 + 2 * a 4

/-- The theorem statement -/
theorem geometric_sequence_minimum (seq : GeometricSequence) :
  (∃ m n : ℕ, Real.sqrt (seq.a m * seq.a n) = Real.sqrt 2 * seq.a 1) →
  (∀ m n : ℕ, 1 / m + 9 / n ≥ 4) ∧
  (∃ m n : ℕ, 1 / m + 9 / n = 4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l2598_259896


namespace NUMINAMATH_CALUDE_alcohol_mixture_concentration_l2598_259866

/-- Represents an alcohol solution with a volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Represents a mixture of two alcohol solutions -/
def AlcoholMixture (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :=
  s1.volume * s1.concentration + s2.volume * s2.concentration = 
    (s1.volume + s2.volume) * finalConcentration

theorem alcohol_mixture_concentration 
  (s1 s2 : AlcoholSolution) (finalConcentration : ℝ) :
  s1.volume = 75 →
  s2.volume = 125 →
  s2.concentration = 0.12 →
  finalConcentration = 0.15 →
  AlcoholMixture s1 s2 finalConcentration →
  s1.concentration = 0.20 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_concentration_l2598_259866


namespace NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l2598_259839

theorem rectangle_to_square_area_ratio (a : ℝ) (a_pos : 0 < a) : 
  let square_side := a
  let square_diagonal := a * Real.sqrt 2
  let rectangle_length := square_diagonal
  let rectangle_width := square_side
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  rectangle_area / square_area = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_area_ratio_l2598_259839


namespace NUMINAMATH_CALUDE_parking_lot_cars_l2598_259828

/-- Given a parking lot with observed wheels and wheels per car, calculate the number of cars -/
def number_of_cars (total_wheels : ℕ) (wheels_per_car : ℕ) : ℕ :=
  total_wheels / wheels_per_car

/-- Theorem: In a parking lot with 48 observed wheels and 4 wheels per car, there are 12 cars -/
theorem parking_lot_cars : number_of_cars 48 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l2598_259828


namespace NUMINAMATH_CALUDE_mom_bought_14_packages_l2598_259831

/-- The number of packages Mom bought -/
def num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Proof that Mom bought 14 packages of white t-shirts -/
theorem mom_bought_14_packages :
  num_packages 70 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_14_packages_l2598_259831


namespace NUMINAMATH_CALUDE_expression_simplification_l2598_259880

theorem expression_simplification (a : ℝ) (h : a = 4) :
  (1 - (a + 1) / a) / ((a^2 - 1) / (a^2 - a)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2598_259880


namespace NUMINAMATH_CALUDE_decimal_point_shift_l2598_259824

theorem decimal_point_shift (x : ℝ) : x - x / 10 = 37.35 → x = 41.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l2598_259824


namespace NUMINAMATH_CALUDE_system_solution_l2598_259853

theorem system_solution (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin a - x₁ * Real.cos a) ∧
     x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (2*a) - x₁ * Real.sin (2*a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin a - x₂ * Real.cos a) ∧
     x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (2*a) - x₂ * Real.sin (2*a))) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 24^2) →
  (∃ n : ℤ, a = π/6 + (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 - (2/3) * Real.arctan (5/12) + (2*π*n)/3 ∨
            a = π/6 + (2*π*n)/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2598_259853


namespace NUMINAMATH_CALUDE_unique_prime_divisibility_l2598_259832

theorem unique_prime_divisibility : 
  ∀ p : ℕ, Prime p → 
  (p = 3 ↔ 
    ∃! a : ℕ, a ∈ Finset.range p ∧ 
    p ∣ (a^3 - 3*a + 1)) := by sorry

end NUMINAMATH_CALUDE_unique_prime_divisibility_l2598_259832


namespace NUMINAMATH_CALUDE_ch4_moles_formed_l2598_259893

/-- Represents the balanced chemical equation: Be2C + 4 H2O → 2 Be(OH)2 + 3 CH4 -/
structure ChemicalEquation where
  be2c_coeff : ℚ
  h2o_coeff : ℚ
  beoh2_coeff : ℚ
  ch4_coeff : ℚ

/-- Represents the available moles of reactants -/
structure AvailableReactants where
  be2c_moles : ℚ
  h2o_moles : ℚ

/-- Calculates the moles of CH4 formed based on the chemical equation and available reactants -/
def moles_ch4_formed (equation : ChemicalEquation) (reactants : AvailableReactants) : ℚ :=
  min
    (reactants.be2c_moles * equation.ch4_coeff / equation.be2c_coeff)
    (reactants.h2o_moles * equation.ch4_coeff / equation.h2o_coeff)

theorem ch4_moles_formed
  (equation : ChemicalEquation)
  (reactants : AvailableReactants)
  (h_equation : equation = ⟨1, 4, 2, 3⟩)
  (h_reactants : reactants = ⟨3, 12⟩) :
  moles_ch4_formed equation reactants = 9 := by
  sorry

end NUMINAMATH_CALUDE_ch4_moles_formed_l2598_259893


namespace NUMINAMATH_CALUDE_game_result_l2598_259803

/-- Represents the state of the game, with each player's money in pence -/
structure GameState where
  adams : ℚ
  baker : ℚ
  carter : ℚ
  dobson : ℚ
  edwards : ℚ
  francis : ℚ
  gudgeon : ℚ

/-- Doubles the money of all players except the winner -/
def double_others (state : GameState) (winner : Fin 7) : GameState :=
  match winner with
  | 0 => ⟨state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 1 => ⟨2*state.adams, state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 2 => ⟨2*state.adams, 2*state.baker, state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 3 => ⟨2*state.adams, 2*state.baker, 2*state.carter, state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 4 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 5 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, state.francis, 2*state.gudgeon⟩
  | 6 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, state.gudgeon⟩

/-- Plays the game for all seven rounds -/
def play_game (initial_state : GameState) : GameState :=
  (List.range 7).foldl (fun state i => double_others state i) initial_state

/-- The main theorem to prove -/
theorem game_result (initial_state : GameState) 
  (h1 : initial_state.adams = 1/2)
  (h2 : initial_state.baker = 1/4)
  (h3 : initial_state.carter = 1/4)
  (h4 : initial_state.dobson = 1/4)
  (h5 : initial_state.edwards = 1/4)
  (h6 : initial_state.francis = 1/4)
  (h7 : initial_state.gudgeon = 1/4) :
  let final_state := play_game initial_state
  final_state.adams = 32 ∧
  final_state.baker = 32 ∧
  final_state.carter = 32 ∧
  final_state.dobson = 32 ∧
  final_state.edwards = 32 ∧
  final_state.francis = 32 ∧
  final_state.gudgeon = 32 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2598_259803


namespace NUMINAMATH_CALUDE_dust_particles_problem_l2598_259892

theorem dust_particles_problem (initial_dust : ℕ) : 
  (initial_dust / 10 + 223 = 331) → initial_dust = 1080 := by
  sorry

end NUMINAMATH_CALUDE_dust_particles_problem_l2598_259892


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2598_259851

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2598_259851


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2598_259867

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 31 →
  (total_players - throwers) % 3 = 0 →
  57 = throwers + (total_players - throwers) * 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2598_259867


namespace NUMINAMATH_CALUDE_lucy_popsicle_purchase_l2598_259871

theorem lucy_popsicle_purchase (lucy_money : ℕ) (popsicle_cost : ℕ) : 
  lucy_money = 2540 → popsicle_cost = 175 → (lucy_money / popsicle_cost : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_lucy_popsicle_purchase_l2598_259871


namespace NUMINAMATH_CALUDE_jewelry_store_restocking_l2598_259868

/-- A jewelry store restocking problem -/
theorem jewelry_store_restocking
  (necklace_capacity : ℕ)
  (current_necklaces : ℕ)
  (current_rings : ℕ)
  (bracelet_capacity : ℕ)
  (current_bracelets : ℕ)
  (necklace_cost : ℕ)
  (ring_cost : ℕ)
  (bracelet_cost : ℕ)
  (total_cost : ℕ)
  (h1 : necklace_capacity = 12)
  (h2 : current_necklaces = 5)
  (h3 : current_rings = 18)
  (h4 : bracelet_capacity = 15)
  (h5 : current_bracelets = 8)
  (h6 : necklace_cost = 4)
  (h7 : ring_cost = 10)
  (h8 : bracelet_cost = 5)
  (h9 : total_cost = 183) :
  ∃ (ring_capacity : ℕ), ring_capacity = 30 ∧
    (necklace_capacity - current_necklaces) * necklace_cost +
    (ring_capacity - current_rings) * ring_cost +
    (bracelet_capacity - current_bracelets) * bracelet_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_jewelry_store_restocking_l2598_259868


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_l2598_259877

theorem power_two_greater_than_square (n : ℕ) (h : n > 5) : 2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_l2598_259877


namespace NUMINAMATH_CALUDE_total_campers_rowing_l2598_259835

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 35)
  (h2 : afternoon_campers = 27) :
  morning_campers + afternoon_campers = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l2598_259835


namespace NUMINAMATH_CALUDE_two_lines_theorem_l2598_259856

/-- Two lines in a 2D plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → ℝ → Prop

/-- The given lines -/
def given_lines : TwoLines where
  l₁ := fun x y ↦ 2 * x + y + 4 = 0
  l₂ := fun a x y ↦ a * x + 4 * y + 1 = 0

/-- Perpendicularity condition -/
def perpendicular (lines : TwoLines) (a : ℝ) : Prop :=
  ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y

/-- Parallelism condition -/
def parallel (lines : TwoLines) (a : ℝ) : Prop :=
  ∀ x y, lines.l₁ x y ↔ ∃ k, lines.l₂ a (x + k) (y + k)

/-- Main theorem -/
theorem two_lines_theorem (lines : TwoLines) :
  (∃ a, perpendicular lines a → 
    ∃ x y, lines.l₁ x y ∧ lines.l₂ a x y ∧ x = -3/2 ∧ y = -1) ∧
  (∃ a, parallel lines a → 
    ∃ d, d = (3 * Real.sqrt 5) / 4 ∧ 
      ∀ x₁ y₁ x₂ y₂, lines.l₁ x₁ y₁ → lines.l₂ a x₂ y₂ → 
        ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2) :=
by sorry

end NUMINAMATH_CALUDE_two_lines_theorem_l2598_259856


namespace NUMINAMATH_CALUDE_second_person_speed_l2598_259817

/-- Given two people moving in opposite directions, this theorem proves
    the speed of the second person given the conditions of the problem. -/
theorem second_person_speed
  (time : ℝ)
  (distance : ℝ)
  (speed1 : ℝ)
  (h1 : time = 4)
  (h2 : distance = 36)
  (h3 : speed1 = 6)
  (h4 : distance = time * (speed1 + speed2)) :
  speed2 = 3 :=
sorry

end NUMINAMATH_CALUDE_second_person_speed_l2598_259817


namespace NUMINAMATH_CALUDE_transport_cost_tripled_bags_reduced_weight_l2598_259808

/-- The cost of transporting cement bags -/
def transport_cost (bags : ℕ) (weight : ℚ) : ℚ :=
  (6000 : ℚ) * bags * weight / (80 * 50)

/-- Theorem: The cost of transporting 240 bags weighing 30 kgs each is $10800 -/
theorem transport_cost_tripled_bags_reduced_weight :
  transport_cost 240 30 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_tripled_bags_reduced_weight_l2598_259808


namespace NUMINAMATH_CALUDE_eggs_leftover_l2598_259819

def david_eggs : ℕ := 45
def ella_eggs : ℕ := 58
def fiona_eggs : ℕ := 29
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + ella_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_leftover_l2598_259819


namespace NUMINAMATH_CALUDE_find_b_l2598_259891

theorem find_b (b c : ℝ) : 
  (∀ x, (3 * x^2 - 4 * x + 5/2) * (2 * x^2 + b * x + c) = 
        6 * x^4 - 11 * x^3 + 13 * x^2 - 15/2 * x + 10/2) → 
  b = -1 := by
sorry

end NUMINAMATH_CALUDE_find_b_l2598_259891


namespace NUMINAMATH_CALUDE_max_snack_bags_l2598_259806

def granola_bars : ℕ := 24
def dried_fruit : ℕ := 36
def nuts : ℕ := 60

theorem max_snack_bags : 
  ∃ (n : ℕ), n > 0 ∧ 
  granola_bars % n = 0 ∧ 
  dried_fruit % n = 0 ∧ 
  nuts % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (granola_bars % m = 0 ∧ dried_fruit % m = 0 ∧ nuts % m = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_max_snack_bags_l2598_259806


namespace NUMINAMATH_CALUDE_max_area_difference_l2598_259827

-- Define a rectangle with integer dimensions
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Theorem statement
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 180 ∧ 
    perimeter r2 = 180 ∧ 
    (∀ (r : Rectangle), perimeter r = 180 → 
      area r1 - area r2 ≥ area r1 - area r ∧ 
      area r1 - area r2 ≥ area r - area r2) ∧
    area r1 - area r2 = 1936 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_l2598_259827


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l2598_259823

/-- Proves that -0.000008691 is equal to -8.691×10^(-6) in scientific notation -/
theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  -0.000008691 = a * 10^n ∧ 
  1 ≤ |a| ∧ 
  |a| < 10 ∧ 
  a = -8.691 ∧ 
  n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l2598_259823


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2598_259843

theorem complex_magnitude_problem (z : ℂ) (h : (Complex.I - 2) * z = 4 + 3 * Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2598_259843


namespace NUMINAMATH_CALUDE_prob_odd_add_only_prob_odd_with_multiply_l2598_259879

-- Define the calculator operations
inductive Operation
| Add
| Multiply

-- Define the calculator state
structure CalculatorState where
  display : ℕ
  lastOp : Option Operation

-- Define the probability of getting an odd number
def probOdd (ops : List Operation) : ℚ :=
  sorry

-- Theorem for part (a)
theorem prob_odd_add_only :
  ∀ (n : ℕ), probOdd (List.replicate n Operation.Add) = 1/2 :=
sorry

-- Theorem for part (b)
theorem prob_odd_with_multiply (n : ℕ) :
  probOdd (List.cons Operation.Multiply (List.replicate n Operation.Add)) < 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_odd_add_only_prob_odd_with_multiply_l2598_259879


namespace NUMINAMATH_CALUDE_three_fourths_to_sixth_power_l2598_259844

theorem three_fourths_to_sixth_power : (3 / 4 : ℚ) ^ 6 = 729 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_three_fourths_to_sixth_power_l2598_259844


namespace NUMINAMATH_CALUDE_notebook_cost_l2598_259858

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 30 →
  total_cost = 1584 →
  ∃ (students_bought notebooks_per_student cost_per_notebook : Nat),
    students_bought = 20 ∧
    students_bought * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook ≥ notebooks_per_student ∧
    cost_per_notebook = 11 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2598_259858


namespace NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l2598_259888

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 5*x - 20 = 4*x + 25

-- Define a function to represent the sum of solutions
def sum_of_solutions : ℝ := 9

-- Theorem statement
theorem sum_of_quadratic_solutions :
  ∃ (x₁ x₂ : ℝ), 
    quadratic_equation x₁ ∧ 
    quadratic_equation x₂ ∧ 
    x₁ ≠ x₂ ∧
    x₁ + x₂ = sum_of_solutions :=
sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_solutions_l2598_259888


namespace NUMINAMATH_CALUDE_final_amount_calculation_l2598_259869

/-- Calculates the final amount after two years of compound interest with different rates for each year -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that the final amount after two years is approximately 5518.80 Rs -/
theorem final_amount_calculation :
  ∃ ε > 0, |final_amount 5253 0.02 0.03 - 5518.80| < ε :=
by
  sorry

#eval final_amount 5253 0.02 0.03

end NUMINAMATH_CALUDE_final_amount_calculation_l2598_259869


namespace NUMINAMATH_CALUDE_f_is_even_function_l2598_259841

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def f (x : ℝ) : ℝ := x^2

theorem f_is_even_function : is_even_function f := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_function_l2598_259841


namespace NUMINAMATH_CALUDE_percentage_problem_l2598_259840

theorem percentage_problem (x p : ℝ) (h1 : 0.25 * x = (p/100) * 500 - 5) (h2 : x = 180) : p = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2598_259840


namespace NUMINAMATH_CALUDE_least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l2598_259890

theorem least_number_of_trees (n : ℕ) : n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_four_five_six : 4 ∣ 60 ∧ 5 ∣ 60 ∧ 6 ∣ 60 := by
  sorry

theorem least_number_of_trees_is_sixty :
  ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 5 ∣ m ∧ 6 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_trees_sixty_divisible_by_four_five_six_least_number_of_trees_is_sixty_l2598_259890


namespace NUMINAMATH_CALUDE_corey_candies_l2598_259873

theorem corey_candies :
  let total_candies : ℝ := 66.5
  let tapanga_extra : ℝ := 8.25
  let corey_candies : ℝ := (total_candies - tapanga_extra) / 2
  corey_candies = 29.125 :=
by
  sorry

end NUMINAMATH_CALUDE_corey_candies_l2598_259873


namespace NUMINAMATH_CALUDE_fermat_prime_equation_solutions_l2598_259850

/-- A Fermat's Prime is a prime number of the form 2^α + 1, for α a positive integer -/
def IsFermatPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ α : ℕ, α > 0 ∧ p = 2^α + 1

/-- The main theorem statement -/
theorem fermat_prime_equation_solutions :
  ∀ p n k : ℕ,
  p > 0 ∧ n > 0 ∧ k > 0 →
  IsFermatPrime p →
  p^n + n = (n+1)^k →
  (p = 3 ∧ n = 1 ∧ k = 2) ∨ (p = 5 ∧ n = 2 ∧ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_equation_solutions_l2598_259850


namespace NUMINAMATH_CALUDE_simplify_expression_l2598_259870

theorem simplify_expression (z : ℝ) : (5 - 2*z) - (4 + 5*z) = 1 - 7*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2598_259870


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l2598_259814

theorem geometric_mean_problem (a b c : ℝ) 
  (h1 : b^2 = a*c)  -- b is the geometric mean of a and c
  (h2 : a*b*c = 27) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l2598_259814


namespace NUMINAMATH_CALUDE_quadratic_max_condition_l2598_259864

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

-- Define the maximum value function
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

-- Theorem statement
theorem quadratic_max_condition (t : ℝ) :
  (∀ x : ℝ, t ≤ x ∧ x ≤ t + 2 → f x ≤ y_max t) →
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ f x = y_max t) →
  t ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_condition_l2598_259864


namespace NUMINAMATH_CALUDE_pole_length_after_cut_l2598_259833

theorem pole_length_after_cut (original_length : ℝ) (cut_percentage : ℝ) (new_length : ℝ) : 
  original_length = 20 →
  cut_percentage = 30 →
  new_length = original_length * (1 - cut_percentage / 100) →
  new_length = 14 :=
by sorry

end NUMINAMATH_CALUDE_pole_length_after_cut_l2598_259833


namespace NUMINAMATH_CALUDE_half_three_abs_diff_squares_l2598_259801

theorem half_three_abs_diff_squares : (1/2 : ℝ) * 3 * |20^2 - 15^2| = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_half_three_abs_diff_squares_l2598_259801


namespace NUMINAMATH_CALUDE_cheryl_tournament_cost_l2598_259805

/-- Calculates the total amount Cheryl pays for a golf tournament given her expenses -/
def tournament_cost (electricity_bill : ℕ) (phone_bill_difference : ℕ) (tournament_percentage : ℕ) : ℕ :=
  let phone_bill := electricity_bill + phone_bill_difference
  let tournament_additional_cost := phone_bill * tournament_percentage / 100
  phone_bill + tournament_additional_cost

/-- Proves that Cheryl pays $1440 for the golf tournament given the specified conditions -/
theorem cheryl_tournament_cost :
  tournament_cost 800 400 20 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_tournament_cost_l2598_259805


namespace NUMINAMATH_CALUDE_total_animals_seen_l2598_259842

/-- Represents the number of animals Erica saw on Saturday -/
def saturday_animals : ℕ := 3 + 2

/-- Represents the number of animals Erica saw on Sunday -/
def sunday_animals : ℕ := 2 + 5

/-- Represents the number of animals Erica saw on Monday -/
def monday_animals : ℕ := 5 + 3

/-- Theorem stating that the total number of animals Erica saw is 20 -/
theorem total_animals_seen : saturday_animals + sunday_animals + monday_animals = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_seen_l2598_259842


namespace NUMINAMATH_CALUDE_prob_A3_given_white_l2598_259813

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def total_urns : ℕ := 12

def urns : Fin 4 → (ℕ × Urn)
  | 0 => (6, ⟨3, 4⟩)  -- A₁
  | 1 => (3, ⟨2, 8⟩)  -- A₂
  | 2 => (2, ⟨6, 1⟩)  -- A₃
  | 3 => (1, ⟨4, 3⟩)  -- A₄

def prob_select_urn (i : Fin 4) : ℚ :=
  (urns i).1 / total_urns

def prob_white_given_urn (i : Fin 4) : ℚ :=
  (urns i).2.white / ((urns i).2.white + (urns i).2.black)

def prob_white : ℚ :=
  Finset.sum Finset.univ (λ i => prob_select_urn i * prob_white_given_urn i)

theorem prob_A3_given_white :
  (prob_select_urn 2 * prob_white_given_urn 2) / prob_white = 30 / 73 := by
  sorry

end NUMINAMATH_CALUDE_prob_A3_given_white_l2598_259813


namespace NUMINAMATH_CALUDE_sqrt_x_cubed_sqrt_x_l2598_259899

theorem sqrt_x_cubed_sqrt_x (x : ℝ) (hx : x > 0) : Real.sqrt (x^3 * Real.sqrt x) = x^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_cubed_sqrt_x_l2598_259899


namespace NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l2598_259894

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling the 12-sided die -/
def expected_value : ℚ := (Finset.sum twelve_sided_die (fun i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_of_twelve_sided_die : expected_value = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_twelve_sided_die_l2598_259894


namespace NUMINAMATH_CALUDE_f_three_zeros_implies_a_gt_sqrt_two_l2598_259876

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a^2*x - 4*a

/-- Theorem stating that if f has three zero points and a > 0, then a > √2 -/
theorem f_three_zeros_implies_a_gt_sqrt_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) :
  a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_three_zeros_implies_a_gt_sqrt_two_l2598_259876


namespace NUMINAMATH_CALUDE_fraction_comparison_l2598_259859

theorem fraction_comparison : (22222222221 : ℚ) / 22222222223 > (33333333331 : ℚ) / 33333333334 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2598_259859


namespace NUMINAMATH_CALUDE_closest_point_is_vertex_l2598_259875

/-- Given a parabola y² = -2x and a point A(m, 0), if the point on the parabola 
closest to A is the vertex of the parabola, then m ∈ [-1, +∞). -/
theorem closest_point_is_vertex (m : ℝ) : 
  (∀ x y : ℝ, y^2 = -2*x → 
    (∀ x' y' : ℝ, y'^2 = -2*x' → (x' - m)^2 + y'^2 ≥ (x - m)^2 + y^2) → 
    x = 0 ∧ y = 0) → 
  m ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_closest_point_is_vertex_l2598_259875


namespace NUMINAMATH_CALUDE_initial_sodium_chloride_percentage_l2598_259837

/-- Proves that given a tank with 10,000 gallons of solution, if 5,500 gallons of water evaporate
    and the remaining solution is 11.11111111111111% sodium chloride, then the initial percentage
    of sodium chloride was 5%. -/
theorem initial_sodium_chloride_percentage
  (initial_volume : ℝ)
  (evaporated_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 10000)
  (h2 : evaporated_volume = 5500)
  (h3 : final_percentage = 11.11111111111111)
  (h4 : final_percentage = (100 * initial_volume * (initial_percentage / 100)) /
                           (initial_volume - evaporated_volume)) :
  initial_percentage = 5 :=
by sorry

end NUMINAMATH_CALUDE_initial_sodium_chloride_percentage_l2598_259837


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l2598_259848

theorem quadratic_roots_existence : 
  (∃ x : ℝ, x^2 + x = 0) ∧ 
  (∃ x : ℝ, 5*x^2 - 4*x - 1 = 0) ∧ 
  (∃ x : ℝ, 3*x^2 - 4*x + 1 = 0) ∧ 
  (∀ x : ℝ, 4*x^2 - 5*x + 2 ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l2598_259848


namespace NUMINAMATH_CALUDE_problem_1_l2598_259865

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3*x)^2 - 4*(x^3)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2598_259865
