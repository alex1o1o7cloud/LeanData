import Mathlib

namespace candy_bars_purchased_l2875_287509

theorem candy_bars_purchased (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 20 →
  candy_bar_cost = 2 →
  remaining_amount = 12 →
  (initial_amount - remaining_amount) / candy_bar_cost = 4 := by
sorry

end candy_bars_purchased_l2875_287509


namespace fred_earnings_l2875_287546

/-- Represents Fred's chore earnings --/
def chore_earnings (initial_amount final_amount : ℕ) 
  (car_wash_price lawn_mow_price dog_walk_price : ℕ)
  (cars_washed lawns_mowed dogs_walked : ℕ) : Prop :=
  final_amount - initial_amount = 
    car_wash_price * cars_washed + 
    lawn_mow_price * lawns_mowed + 
    dog_walk_price * dogs_walked

/-- Theorem stating that Fred's earnings from chores match the difference in his money --/
theorem fred_earnings :
  chore_earnings 23 86 5 10 3 4 3 7 := by
  sorry

end fred_earnings_l2875_287546


namespace airplane_passengers_l2875_287524

theorem airplane_passengers (total_passengers men : ℕ) 
  (h1 : total_passengers = 80)
  (h2 : men = 30)
  (h3 : ∃ women : ℕ, men = women ∧ men + women + (total_passengers - (men + women)) = total_passengers) :
  total_passengers - 2 * men = 20 := by
  sorry

end airplane_passengers_l2875_287524


namespace sqrt_equation_solution_l2875_287547

theorem sqrt_equation_solution (z : ℚ) : 
  Real.sqrt (5 - 4 * z + 1) = 7 → z = -43 / 4 := by
  sorry

end sqrt_equation_solution_l2875_287547


namespace sadie_homework_problem_l2875_287528

/-- The total number of math homework problems Sadie has for the week. -/
def total_problems : ℕ := 140

/-- The number of solving linear equations problems Sadie has. -/
def linear_equations_problems : ℕ := 28

/-- Theorem stating that the total number of math homework problems is 140,
    given the conditions from the problem. -/
theorem sadie_homework_problem :
  (total_problems : ℝ) * 0.4 * 0.5 = linear_equations_problems :=
by sorry

end sadie_homework_problem_l2875_287528


namespace weight_change_l2875_287570

theorem weight_change (w : ℝ) (hw : w > 0) : w * 0.8 * 1.3 * 0.8 * 1.1 < w := by
  sorry

#check weight_change

end weight_change_l2875_287570


namespace prob_red_white_blue_eq_two_fifty_five_l2875_287584

/-- The number of red marbles initially in the bag -/
def red_marbles : ℕ := 4

/-- The number of white marbles initially in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles initially in the bag -/
def blue_marbles : ℕ := 2

/-- The total number of marbles initially in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The probability of drawing a red marble first, then a white marble, then a blue marble -/
def prob_red_white_blue : ℚ :=
  (red_marbles : ℚ) / total_marbles *
  (white_marbles : ℚ) / (total_marbles - 1) *
  (blue_marbles : ℚ) / (total_marbles - 2)

theorem prob_red_white_blue_eq_two_fifty_five :
  prob_red_white_blue = 2 / 55 := by sorry

end prob_red_white_blue_eq_two_fifty_five_l2875_287584


namespace inequality_not_always_true_l2875_287590

theorem inequality_not_always_true (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z > 0) :
  ¬ (∀ x y z : ℝ, x > 0 → y > 0 → x > y → z > 0 → |x/z - y/z| = (x-y)/z) :=
sorry

end inequality_not_always_true_l2875_287590


namespace negative_reciprocal_inequality_l2875_287587

theorem negative_reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  -1/a < -1/b := by
  sorry

end negative_reciprocal_inequality_l2875_287587


namespace product_inequality_l2875_287579

theorem product_inequality (a b c d : ℝ) 
  (sum_eq : a + d = b + c) 
  (abs_ineq : |a - d| < |b - c|) : 
  a * d > b * c := by
  sorry

end product_inequality_l2875_287579


namespace mean_motorcycles_rainy_days_l2875_287594

def sunny_car_counts : List ℝ := [30, 14, 14, 21, 25]
def sunny_motorcycle_counts : List ℝ := [5, 2, 4, 1, 3]
def rainy_car_counts : List ℝ := [40, 20, 17, 31, 30]
def rainy_motorcycle_counts : List ℝ := [2, 1, 1, 0, 2]

theorem mean_motorcycles_rainy_days :
  (rainy_motorcycle_counts.sum / rainy_motorcycle_counts.length : ℝ) = 1.2 := by
  sorry

end mean_motorcycles_rainy_days_l2875_287594


namespace other_donation_is_100_l2875_287544

/-- Represents the fundraiser for basketball equipment -/
structure Fundraiser where
  goal : ℕ
  bronze_donation : ℕ
  silver_donation : ℕ
  bronze_count : ℕ
  silver_count : ℕ
  other_count : ℕ
  final_day_goal : ℕ

/-- Calculates the amount donated by the family with another status -/
def other_donation (f : Fundraiser) : ℕ :=
  f.goal - (f.bronze_donation * f.bronze_count + f.silver_donation * f.silver_count + f.final_day_goal)

/-- Theorem stating that the family with another status donated $100 -/
theorem other_donation_is_100 (f : Fundraiser)
  (h1 : f.goal = 750)
  (h2 : f.bronze_donation = 25)
  (h3 : f.silver_donation = 50)
  (h4 : f.bronze_count = 10)
  (h5 : f.silver_count = 7)
  (h6 : f.other_count = 1)
  (h7 : f.final_day_goal = 50) :
  other_donation f = 100 := by
  sorry

end other_donation_is_100_l2875_287544


namespace tank_capacity_tank_capacity_1440_l2875_287589

/-- Given a tank with a leak and an inlet pipe, prove its capacity. -/
theorem tank_capacity (leak_time : ℝ) (inlet_rate : ℝ) (combined_time : ℝ) : ℝ :=
  let leak_rate := 1 / leak_time
  let inlet_rate_hourly := inlet_rate * 60
  let combined_rate := 1 / combined_time
  let capacity := (inlet_rate_hourly - combined_rate) / (leak_rate - combined_rate)
  by
    -- Assumptions
    have h1 : leak_time = 6 := by sorry
    have h2 : inlet_rate = 6 := by sorry
    have h3 : combined_time = 12 := by sorry
    
    -- Proof
    sorry

/-- The main theorem stating the tank's capacity. -/
theorem tank_capacity_1440 : tank_capacity 6 6 12 = 1440 := by sorry

end tank_capacity_tank_capacity_1440_l2875_287589


namespace rationalize_denominator_l2875_287521

theorem rationalize_denominator : (45 : ℝ) / Real.sqrt 45 = 3 * Real.sqrt 5 := by sorry

end rationalize_denominator_l2875_287521


namespace givenCurve_is_parabola_l2875_287505

/-- A curve in 2D space represented by parametric equations -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Definition of a parabola in standard form -/
def IsParabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given parametric curve -/
def givenCurve : ParametricCurve where
  x := λ t => t
  y := λ t => t^2 + 1

/-- Theorem stating that the given curve is a parabola -/
theorem givenCurve_is_parabola :
  IsParabola (λ x => givenCurve.y (givenCurve.x⁻¹ x)) :=
sorry

end givenCurve_is_parabola_l2875_287505


namespace tiles_per_row_l2875_287585

-- Define the area of the room
def room_area : ℝ := 144

-- Define the side length of a tile in meters
def tile_side : ℝ := 0.3

-- Theorem statement
theorem tiles_per_row (room_area : ℝ) (tile_side : ℝ) :
  room_area = 144 ∧ tile_side = 0.3 →
  (Real.sqrt room_area / tile_side : ℝ) = 40 := by
  sorry

end tiles_per_row_l2875_287585


namespace arithmetic_sequence_problem_l2875_287556

/-- An arithmetic sequence with positive terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a n > 0

/-- The property that three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_sum : a 2 + a 3 + a 4 = 15)
    (h_geom : GeometricSequence (a 1 + 2) (a 3 + 4) (a 6 + 16)) :
    a 10 = 19 := by
  sorry

end arithmetic_sequence_problem_l2875_287556


namespace temperature_at_speed_0_4_l2875_287507

/-- The temperature in degrees Celsius given the speed of sound in meters per second -/
def temperature (v : ℝ) : ℝ := 15 * v^2

/-- Theorem: When the speed of sound is 0.4 m/s, the temperature is 2.4°C -/
theorem temperature_at_speed_0_4 : temperature 0.4 = 2.4 := by
  sorry

end temperature_at_speed_0_4_l2875_287507


namespace indefinite_game_l2875_287551

/-- Represents the web structure --/
structure Web where
  rings : ℕ
  radii : ℕ
  rings_ge_two : rings ≥ 2
  radii_ge_three : radii ≥ 3

/-- Represents the game state --/
inductive GameState
  | Ongoing
  | ButterflyWins
  | SpiderWins

/-- Defines the game outcome --/
def gameOutcome (web : Web) : GameState :=
  if web.radii % 2 = 0 ∧ web.rings ≥ web.radii / 2 then
    GameState.Ongoing
  else if web.radii % 2 = 1 ∧ web.rings ≥ (web.radii - 1) / 2 then
    GameState.Ongoing
  else
    GameState.Ongoing -- We use Ongoing as a placeholder, as the actual outcome might depend on the players' strategies

/-- Theorem stating that under certain conditions, the game continues indefinitely --/
theorem indefinite_game (web : Web) :
  (web.radii % 2 = 0 → web.rings ≥ web.radii / 2) ∧
  (web.radii % 2 = 1 → web.rings ≥ (web.radii - 1) / 2) →
  gameOutcome web = GameState.Ongoing :=
by
  sorry

#check indefinite_game

end indefinite_game_l2875_287551


namespace theodore_sturgeon_collection_hardcovers_l2875_287513

/-- Given a collection of books with two price options and a total cost,
    calculate the number of books purchased at the higher price. -/
def hardcover_count (total_volumes : ℕ) (paperback_price hardcover_price : ℕ) (total_cost : ℕ) : ℕ :=
  let h := (2 * total_cost - paperback_price * total_volumes) / (2 * (hardcover_price - paperback_price))
  h

/-- Theorem stating that given the specific conditions of the problem,
    the number of hardcover books purchased is 6. -/
theorem theodore_sturgeon_collection_hardcovers :
  hardcover_count 12 15 30 270 = 6 := by
  sorry

end theodore_sturgeon_collection_hardcovers_l2875_287513


namespace perpendicular_vectors_k_value_l2875_287566

/-- Given vectors a and b, if they are perpendicular, then k = 3 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let a : ℝ × ℝ := (2*k - 3, -6)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → k = 3 := by
  sorry

end perpendicular_vectors_k_value_l2875_287566


namespace line_y_axis_intersection_l2875_287512

/-- A line passing through two given points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 4 ∧ y₁ = 20 ∧ x₂ = -6 ∧ y₂ = -2 →
  ∃ (y : ℝ), y = 11.2 ∧ 
    (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end line_y_axis_intersection_l2875_287512


namespace boy_age_problem_l2875_287550

theorem boy_age_problem (current_age : ℕ) (years_ago : ℕ) : 
  current_age = 10 →
  current_age = 2 * (current_age - years_ago) →
  years_ago = 5 :=
by
  sorry

end boy_age_problem_l2875_287550


namespace flower_bed_area_and_perimeter_l2875_287548

/-- Represents a rectangular flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Calculate the area of a rectangular flower bed -/
def area (fb : FlowerBed) : ℝ :=
  fb.length * fb.width

/-- Calculate the perimeter of a rectangular flower bed -/
def perimeter (fb : FlowerBed) : ℝ :=
  2 * (fb.length + fb.width)

theorem flower_bed_area_and_perimeter :
  let fb : FlowerBed := { length := 60, width := 45 }
  area fb = 2700 ∧ perimeter fb = 210 := by
  sorry

end flower_bed_area_and_perimeter_l2875_287548


namespace spiral_grid_second_row_sum_l2875_287537

/-- Represents a position in the grid -/
structure Position :=
  (x : Fin 15)
  (y : Fin 15)

/-- Represents the spiral grid -/
def SpiralGrid := Fin 15 → Fin 15 → Nat

/-- Creates a spiral grid according to the problem description -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Returns the center position of the grid -/
def centerPosition : Position :=
  ⟨7, 7⟩

/-- Checks if a given position is in the second row from the top -/
def isSecondRow (pos : Position) : Prop :=
  pos.y = 1

/-- Returns the maximum value in the second row -/
def maxSecondRow (grid : SpiralGrid) : Nat :=
  sorry

/-- Returns the minimum value in the second row -/
def minSecondRow (grid : SpiralGrid) : Nat :=
  sorry

theorem spiral_grid_second_row_sum :
  let grid := createSpiralGrid
  maxSecondRow grid + minSecondRow grid = 367 :=
sorry

end spiral_grid_second_row_sum_l2875_287537


namespace weight_of_a_l2875_287596

theorem weight_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 80 →
  (a + b + c + d) / 4 = 82 →
  e = d + 3 →
  (b + c + d + e) / 4 = 81 →
  a = 95 := by
sorry

end weight_of_a_l2875_287596


namespace interval_relationship_l2875_287552

theorem interval_relationship : 
  (∀ x, 2 < x ∧ x < 3 → 1 < x ∧ x < 5) ∧ 
  ¬(∀ x, 1 < x ∧ x < 5 → 2 < x ∧ x < 3) := by
  sorry

end interval_relationship_l2875_287552


namespace teacher_age_l2875_287504

theorem teacher_age (num_students : ℕ) (avg_age_students : ℕ) (avg_increase : ℕ) : 
  num_students = 22 →
  avg_age_students = 21 →
  avg_increase = 1 →
  (num_students * avg_age_students + 44) / (num_students + 1) = avg_age_students + avg_increase :=
by sorry

end teacher_age_l2875_287504


namespace intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l2875_287583

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def N (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

theorem intersection_singleton_implies_k_negative_one :
  (∃! x, x ∈ M ∩ N k) → k = -1 :=
by sorry

theorem intersection_and_union_when_k_is_two :
  M ∩ N 2 = {x | -1 ≤ x ∧ x ≤ 2} ∧ M ∪ N 2 = {x | x ≤ 5} :=
by sorry

end intersection_singleton_implies_k_negative_one_intersection_and_union_when_k_is_two_l2875_287583


namespace cube_and_sphere_volume_l2875_287577

theorem cube_and_sphere_volume (cube_volume : Real) (sphere_volume : Real) : 
  cube_volume = 8 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end cube_and_sphere_volume_l2875_287577


namespace solve_for_y_l2875_287560

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 7) : y = 55/2 := by
  sorry

end solve_for_y_l2875_287560


namespace stationery_solution_l2875_287593

/-- Represents a pack of stationery -/
structure StationeryPack where
  sheets : ℕ
  envelopes : ℕ

/-- The problem setup -/
def stationeryProblem (pack : StationeryPack) : Prop :=
  ∃ (jack_leftover_sheets tom_leftover_envelopes : ℕ),
    -- Jack uses all envelopes and has 90 sheets left
    pack.sheets - 2 * pack.envelopes = jack_leftover_sheets ∧
    jack_leftover_sheets = 90 ∧
    -- Tom uses all sheets and has 30 envelopes left
    pack.sheets = 4 * (pack.envelopes - tom_leftover_envelopes) ∧
    tom_leftover_envelopes = 30

/-- The theorem to prove -/
theorem stationery_solution :
  ∃ (pack : StationeryPack),
    stationeryProblem pack ∧
    pack.sheets = 120 ∧
    pack.envelopes = 30 := by
  sorry

end stationery_solution_l2875_287593


namespace george_remaining_eggs_l2875_287503

/-- Calculates the remaining number of eggs given the initial inventory and sold amount. -/
def remaining_eggs (cases : ℕ) (boxes_per_case : ℕ) (eggs_per_box : ℕ) (boxes_sold : ℕ) : ℕ :=
  cases * boxes_per_case * eggs_per_box - boxes_sold * eggs_per_box

/-- Proves that George has 648 eggs remaining after selling 3 boxes. -/
theorem george_remaining_eggs :
  remaining_eggs 7 12 8 3 = 648 := by
  sorry

end george_remaining_eggs_l2875_287503


namespace opposite_to_silver_is_pink_l2875_287564

-- Define the colors
inductive Color
  | Pink
  | Teal
  | Maroon
  | Lilac
  | Silver
  | Crimson

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  faces : List Face
  hinged : List (Face × Face)

-- Define the property of opposite faces
def areOpposite (c : Cube) (f1 f2 : Face) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ f1 ≠ f2

-- State the theorem
theorem opposite_to_silver_is_pink (c : Cube) :
  (∃ f1 f2 : Face, f1.color = Color.Silver ∧ f2.color = Color.Pink ∧ areOpposite c f1 f2) :=
by sorry

end opposite_to_silver_is_pink_l2875_287564


namespace division_result_l2875_287555

theorem division_result : (4.036 : ℝ) / 0.02 = 201.8 := by
  sorry

end division_result_l2875_287555


namespace sqrt_sum_inequality_l2875_287562

theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) ≤ 2 * Real.sqrt 2 := by
  sorry

end sqrt_sum_inequality_l2875_287562


namespace sin_thirteen_pi_sixths_l2875_287506

theorem sin_thirteen_pi_sixths : Real.sin (13 * Real.pi / 6) = 1 / 2 := by
  sorry

end sin_thirteen_pi_sixths_l2875_287506


namespace bobby_candy_problem_l2875_287568

/-- Proves that Bobby ate 6 pieces of candy initially -/
theorem bobby_candy_problem :
  ∀ (initial_candy : ℕ) (eaten_initially : ℕ) (eaten_later : ℕ) (remaining_candy : ℕ),
    initial_candy = 22 →
    eaten_later = 5 →
    remaining_candy = 8 →
    initial_candy - (eaten_initially + eaten_initially / 2 + eaten_later) = remaining_candy →
    eaten_initially = 6 := by
  sorry

end bobby_candy_problem_l2875_287568


namespace joyce_apples_to_larry_l2875_287510

/-- The number of apples Joyce gave to Larry -/
def apples_given_to_larry (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem joyce_apples_to_larry : 
  apples_given_to_larry 75 23 = 52 := by
  sorry

end joyce_apples_to_larry_l2875_287510


namespace polynomial_expansion_l2875_287572

theorem polynomial_expansion (z : ℝ) : 
  (3*z^2 + 4*z - 5) * (4*z^3 - 3*z + 2) = 
  12*z^5 + 16*z^4 - 29*z^3 - 6*z^2 + 23*z - 10 := by
  sorry

end polynomial_expansion_l2875_287572


namespace total_points_scored_l2875_287520

/-- Given a player who played 10.0 games and scored 12 points in each game,
    the total points scored is 120. -/
theorem total_points_scored (games : ℝ) (points_per_game : ℕ) : 
  games = 10.0 → points_per_game = 12 → games * (points_per_game : ℝ) = 120 := by
  sorry

end total_points_scored_l2875_287520


namespace f_neg_one_eq_one_fifteenth_l2875_287536

/-- The function f satisfying the given equation for all x -/
noncomputable def f : ℝ → ℝ := 
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) - 1) / (x^(2^5 - 1) - 1)

/-- Theorem stating that f(-1) = 1/15 -/
theorem f_neg_one_eq_one_fifteenth : f (-1) = 1 / 15 := by
  sorry

end f_neg_one_eq_one_fifteenth_l2875_287536


namespace line_through_points_l2875_287539

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given distinct vectors a and b, and a scalar k, 
    prove that k*a + (1/2)*b lies on the line through a and b 
    if and only if k = 1/2 -/
theorem line_through_points (a b : V) (k : ℝ) 
    (h_distinct : a ≠ b) : 
    (∃ t : ℝ, k • a + (1/2) • b = a + t • (b - a)) ↔ k = 1/2 := by
  sorry

end line_through_points_l2875_287539


namespace equation_solution_l2875_287540

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), 
    x₁ > 0 ∧ x₂ > 0 ∧
    (1/3 * (4 * x₁^2 - 2) = (x₁^2 - 60*x₁ - 15) * (x₁^2 + 30*x₁ + 3)) ∧
    (1/3 * (4 * x₂^2 - 2) = (x₂^2 - 60*x₂ - 15) * (x₂^2 + 30*x₂ + 3)) ∧
    x₁ = 30 + Real.sqrt 917 ∧
    x₂ = -15 + Real.sqrt 8016 / 6 ∧
    ∀ (y : ℝ), 
      y > 0 ∧ (1/3 * (4 * y^2 - 2) = (y^2 - 60*y - 15) * (y^2 + 30*y + 3)) →
      (y = x₁ ∨ y = x₂) :=
by sorry

end equation_solution_l2875_287540


namespace total_matting_cost_l2875_287545

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a room with its dimensions and matting cost -/
structure Room where
  dimensions : RoomDimensions
  mattingCostPerSquareMeter : ℝ

/-- Calculates the floor area of a room -/
def floorArea (room : Room) : ℝ :=
  room.dimensions.length * room.dimensions.width

/-- Calculates the matting cost for a room -/
def mattingCost (room : Room) : ℝ :=
  floorArea room * room.mattingCostPerSquareMeter

/-- The three rooms in the house -/
def hall : Room :=
  { dimensions := { length := 20, width := 15, height := 5 },
    mattingCostPerSquareMeter := 40 }

def bedroom : Room :=
  { dimensions := { length := 10, width := 5, height := 4 },
    mattingCostPerSquareMeter := 35 }

def study : Room :=
  { dimensions := { length := 8, width := 6, height := 3 },
    mattingCostPerSquareMeter := 45 }

/-- Theorem: The total cost of matting for all three rooms is 15910 -/
theorem total_matting_cost :
  mattingCost hall + mattingCost bedroom + mattingCost study = 15910 := by
  sorry

end total_matting_cost_l2875_287545


namespace rectangular_sheet_area_l2875_287541

theorem rectangular_sheet_area :
  ∀ (area_small area_large total_area : ℝ),
  area_large = 4 * area_small →
  area_large - area_small = 2208 →
  total_area = area_small + area_large →
  total_area = 3680 :=
by
  sorry

end rectangular_sheet_area_l2875_287541


namespace margo_irma_pairing_probability_l2875_287569

/-- Represents the number of students in the class -/
def class_size : ℕ := 40

/-- Represents the probability of Margo being paired with Irma -/
def probability_paired_with_irma : ℚ := 1 / 39

/-- Theorem stating that the probability of Margo being paired with Irma is 1/39 -/
theorem margo_irma_pairing_probability :
  probability_paired_with_irma = 1 / (class_size - 1) :=
by sorry

end margo_irma_pairing_probability_l2875_287569


namespace die_roll_probability_l2875_287599

/-- The probability of rolling a different number on a six-sided die -/
def p_different : ℚ := 5 / 6

/-- The probability of rolling the same number on a six-sided die -/
def p_same : ℚ := 1 / 6

/-- The number of rolls before the final roll -/
def n : ℕ := 9

theorem die_roll_probability :
  p_different ^ n * p_same = (5^8 : ℚ) / (6^9 : ℚ) := by
  sorry

end die_roll_probability_l2875_287599


namespace least_product_of_primes_above_30_l2875_287518

theorem least_product_of_primes_above_30 :
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    p * q = 1147 ∧
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 30 → s > 30 → r ≠ s → r * s ≥ 1147 :=
by sorry

end least_product_of_primes_above_30_l2875_287518


namespace paragraph_writing_time_l2875_287514

/-- Represents the time in minutes for various writing assignments -/
structure WritingTimes where
  short_answer : ℕ  -- Time for one short-answer question
  essay : ℕ         -- Time for one essay
  total : ℕ         -- Total homework time
  paragraph : ℕ     -- Time for one paragraph (to be proved)

/-- Represents the number of assignments -/
structure AssignmentCounts where
  essays : ℕ
  paragraphs : ℕ
  short_answers : ℕ

theorem paragraph_writing_time 
  (wt : WritingTimes) 
  (ac : AssignmentCounts) 
  (h1 : wt.short_answer = 3)
  (h2 : wt.essay = 60)
  (h3 : wt.total = 4 * 60)
  (h4 : ac.essays = 2)
  (h5 : ac.paragraphs = 5)
  (h6 : ac.short_answers = 15)
  (h7 : wt.total = ac.essays * wt.essay + ac.paragraphs * wt.paragraph + ac.short_answers * wt.short_answer) :
  wt.paragraph = 15 := by
  sorry

end paragraph_writing_time_l2875_287514


namespace train_passing_time_l2875_287522

/-- Proves that a train passing a platform in given time and speed will pass a stationary point in approximately 20 seconds -/
theorem train_passing_time (platform_length : ℝ) (platform_passing_time : ℝ) (train_speed_kmh : ℝ) 
  (h1 : platform_length = 360.0288)
  (h2 : platform_passing_time = 44)
  (h3 : train_speed_kmh = 54) : 
  ∃ (time : ℝ), abs (time - 20) < 0.01 := by
  sorry

end train_passing_time_l2875_287522


namespace rational_irrational_relations_l2875_287559

theorem rational_irrational_relations (m n : ℚ) :
  (((m - 3) * Real.sqrt 6 + n - 3 = 0) → Real.sqrt (m * n) = 3 ∨ Real.sqrt (m * n) = -3) ∧
  ((∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ (2 + Real.sqrt 3) * m - (1 - Real.sqrt 3) * n = 5) → 
   ∃ x : ℝ, m^2 = x ∧ n^2 = x ∧ x = 25/9) :=
by sorry

end rational_irrational_relations_l2875_287559


namespace turning_to_similar_section_is_random_event_l2875_287534

/-- Represents the event of turning to a similar section in a textbook -/
def turning_to_similar_section : Type := Unit

/-- Defines the properties of the event -/
class EventProperties (α : Type) where
  not_guaranteed : ∀ (x y : α), x ≠ y → True
  possible : ∃ (x : α), True
  not_certain : ¬ (∀ (x : α), True)
  not_impossible : ∃ (x : α), True
  not_predictable : ∀ (x : α), ¬ (∀ (y : α), x = y)

/-- Defines a random event -/
class RandomEvent (α : Type) extends EventProperties α

/-- Theorem stating that turning to a similar section is a random event -/
theorem turning_to_similar_section_is_random_event :
  RandomEvent turning_to_similar_section :=
sorry

end turning_to_similar_section_is_random_event_l2875_287534


namespace quarters_in_jar_l2875_287535

def pennies : ℕ := 123
def nickels : ℕ := 85
def dimes : ℕ := 35
def half_dollars : ℕ := 15
def dollar_coins : ℕ := 5
def family_members : ℕ := 8
def ice_cream_cost : ℚ := 4.5
def leftover : ℚ := 0.97

def total_other_coins : ℚ := 
  pennies * 0.01 + nickels * 0.05 + dimes * 0.1 + half_dollars * 0.5 + dollar_coins * 1.0

theorem quarters_in_jar : 
  ∃ (quarters : ℕ), 
    (quarters : ℚ) * 0.25 + total_other_coins = 
      family_members * ice_cream_cost + leftover ∧ 
    quarters = 140 := by sorry

end quarters_in_jar_l2875_287535


namespace magnitude_of_mn_l2875_287501

/-- Given vectors and conditions, prove the magnitude of MN --/
theorem magnitude_of_mn (a b c : ℝ × ℝ) (x y : ℝ) : 
  a = (2, -1) →
  b = (x, -2) →
  c = (3, y) →
  ∃ (k : ℝ), a = k • b →  -- a is parallel to b
  (a + b) • (b - c) = 0 →  -- (a + b) is perpendicular to (b - c)
  ‖(y - x, x - y)‖ = 8 * Real.sqrt 2 := by
  sorry

#check magnitude_of_mn

end magnitude_of_mn_l2875_287501


namespace opposite_sign_fractions_l2875_287575

theorem opposite_sign_fractions (x : ℚ) : 
  x = 7/5 → ((x - 1) / 2) * ((x - 2) / 3) < 0 := by sorry

end opposite_sign_fractions_l2875_287575


namespace linear_equation_values_l2875_287525

/-- Given that x^(a-2) - 2y^(a-b+5) = 1 is a linear equation in x and y, prove that a = 3 and b = 7 -/
theorem linear_equation_values (a b : ℤ) : 
  (∀ x y : ℝ, ∃ m n c : ℝ, x^(a-2) - 2*y^(a-b+5) = m*x + n*y + c) → 
  a = 3 ∧ b = 7 := by
sorry

end linear_equation_values_l2875_287525


namespace family_ages_solution_l2875_287597

/-- Represents the ages of a father and his two children -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  daughter : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.father + ages.son + ages.daughter = 110 ∧
  ages.son = ages.daughter ∧
  3 * ages.father = 186

/-- The theorem to be proved -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ 
    ages.father = 62 ∧ ages.son = 24 ∧ ages.daughter = 24 := by
  sorry

end family_ages_solution_l2875_287597


namespace isosceles_triangle_angles_l2875_287561

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the property of being isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define the exterior angle of A
def exteriorAngleA (t : Triangle) : ℝ := 180 - t.A

-- Theorem statement
theorem isosceles_triangle_angles (t : Triangle) :
  exteriorAngleA t = 110 →
  isIsosceles t →
  t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry


end isosceles_triangle_angles_l2875_287561


namespace train_length_l2875_287591

/-- Given a train that crosses a platform in 39 seconds and a signal pole in 18 seconds,
    where the platform is 350 meters long, prove that the length of the train is 300 meters. -/
theorem train_length (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (platform_length : ℕ)
    (h1 : platform_crossing_time = 39)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 350) :
    let train_length := (platform_crossing_time * platform_length) / (platform_crossing_time - pole_crossing_time)
    train_length = 300 := by
  sorry

end train_length_l2875_287591


namespace exists_winning_strategy_2019_not_exists_winning_strategy_2020_l2875_287542

/-- Represents the state of the game at any point -/
structure GameState where
  piles : List Nat
  bag : Nat

/-- Represents a valid move in the game -/
inductive Move
  | Split (pile : Nat) (split1 : Nat) (split2 : Nat)

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.Split pile split1 split2 =>
      pile ∈ state.piles ∧
      pile ≥ 4 ∧
      split1 > 0 ∧
      split2 > 0 ∧
      split1 + split2 = pile - 1

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Split pile split1 split2 =>
      { piles := state.piles.filter (· ≠ pile) ++ [split1, split2],
        bag := state.bag + 1 }

/-- Checks if the game is in a winning state -/
def isWinningState (state : GameState) : Prop :=
  state.piles.all (· = 3)

/-- Defines a game strategy as a function that selects a move given a game state -/
def Strategy := GameState → Option Move

/-- Theorem: When N = 2019, there exists a winning strategy -/
theorem exists_winning_strategy_2019 :
  ∃ (strategy : Strategy),
    let initialState : GameState := { piles := [2019], bag := 0 }
    ∃ (finalState : GameState),
      (∀ (state : GameState),
        state.piles.sum + state.bag = 2019 →
        match strategy state with
        | some move => isValidMove state move
        | none => isWinningState state) ∧
      isWinningState finalState :=
sorry

/-- Theorem: When N = 2020, there does not exist a winning strategy -/
theorem not_exists_winning_strategy_2020 :
  ¬∃ (strategy : Strategy),
    let initialState : GameState := { piles := [2020], bag := 0 }
    ∃ (finalState : GameState),
      (∀ (state : GameState),
        state.piles.sum + state.bag = 2020 →
        match strategy state with
        | some move => isValidMove state move
        | none => isWinningState state) ∧
      isWinningState finalState :=
sorry

end exists_winning_strategy_2019_not_exists_winning_strategy_2020_l2875_287542


namespace binary_multiplication_theorem_l2875_287526

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec to_bits (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
    to_bits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, false, false, true, false, true]  -- 1010011₂
  binary_to_decimal a * binary_to_decimal b = binary_to_decimal c := by
  sorry

end binary_multiplication_theorem_l2875_287526


namespace graph_translation_l2875_287595

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define k as a positive real number
variable (k : ℝ)
variable (h : k > 0)

-- State the theorem
theorem graph_translation (x y : ℝ) : 
  y = f (x + k) ↔ y = f ((x + k) - k) :=
sorry

end graph_translation_l2875_287595


namespace no_m_for_all_x_range_for_m_in_interval_l2875_287538

-- Part 1
theorem no_m_for_all_x : ∀ m : ℝ, ∃ x : ℝ, 2 * x - 1 ≤ m * (x^2 - 1) := by sorry

-- Part 2
def inequality_set (m : ℝ) : Set ℝ := {x | 2 * x - 1 > m * (x^2 - 1)}

theorem range_for_m_in_interval :
  ∀ m ∈ Set.Icc (-2 : ℝ) 2,
  inequality_set m = Set.Ioo (((-1 : ℝ) + Real.sqrt 7) / 2) ((1 + Real.sqrt 3) / 2) := by sorry

end no_m_for_all_x_range_for_m_in_interval_l2875_287538


namespace parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l2875_287553

/-- Represents a parabola and an ellipse in a 2D coordinate system -/
structure ParabolaEllipseSystem where
  m : ℝ
  n : ℝ
  hm : m > 0
  hn : n > 0

/-- The parabola equation: mx + ny² = 0 -/
def parabola_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x + sys.n * y^2 = 0

/-- The ellipse equation: mx² + ny² = 1 -/
def ellipse_equation (sys : ParabolaEllipseSystem) (x y : ℝ) : Prop :=
  sys.m * x^2 + sys.n * y^2 = 1

/-- Theorem stating that the parabola opens to the left -/
theorem parabola_opens_left (sys : ParabolaEllipseSystem) :
  ∀ x y, parabola_equation sys x y → x ≤ 0 :=
sorry

/-- Theorem stating that the equation represents an ellipse -/
theorem is_ellipse (sys : ParabolaEllipseSystem) :
  ∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1 :=
sorry

/-- Main theorem: The system represents a left-opening parabola and an ellipse -/
theorem parabola_ellipse_system_correct (sys : ParabolaEllipseSystem) :
  (∀ x y, parabola_equation sys x y → x ≤ 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ ∀ x y, ellipse_equation sys x y ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end parabola_opens_left_is_ellipse_parabola_ellipse_system_correct_l2875_287553


namespace ted_peeling_time_l2875_287523

/-- The time it takes Julie to peel potatoes individually (in hours) -/
def julie_time : ℝ := 10

/-- The time Julie and Ted work together (in hours) -/
def together_time : ℝ := 4

/-- The time it takes Julie to complete the task after Ted leaves (in hours) -/
def julie_remaining_time : ℝ := 0.9999999999999998

/-- The time it takes Ted to peel potatoes individually (in hours) -/
def ted_time : ℝ := 8

/-- Theorem stating that given the conditions, Ted's individual time to peel potatoes is 8 hours -/
theorem ted_peeling_time :
  (together_time * (1 / julie_time + 1 / ted_time)) + (julie_remaining_time * (1 / julie_time)) = 1 :=
sorry

end ted_peeling_time_l2875_287523


namespace chess_tournament_matches_l2875_287549

/-- The number of matches in a round-robin tournament with n players -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament_matches :
  num_matches 10 = 45 := by sorry

end chess_tournament_matches_l2875_287549


namespace profit_percent_calculation_l2875_287529

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/82 - 1) * 100 := by
  sorry

#eval (100/82 - 1) * 100 -- This will output approximately 21.95

end profit_percent_calculation_l2875_287529


namespace simplify_expression_1_simplify_expression_2_l2875_287517

-- Expression 1
theorem simplify_expression_1 (a b : ℝ) :
  -2 * a^2 * b - 3 * a * b^2 + 3 * a^2 * b - 4 * a * b^2 = a^2 * b - 7 * a * b^2 := by
  sorry

-- Expression 2
theorem simplify_expression_2 (x y z : ℝ) :
  2 * (x * y * z - 3 * x) + 5 * (2 * x - 3 * x * y * z) = 4 * x - 13 * x * y * z := by
  sorry

end simplify_expression_1_simplify_expression_2_l2875_287517


namespace total_points_after_perfect_games_l2875_287511

/-- The number of points in a perfect score -/
def perfect_score : ℕ := 21

/-- The number of consecutive perfect games -/
def consecutive_games : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games :
  perfect_score * consecutive_games = 63 := by
  sorry

end total_points_after_perfect_games_l2875_287511


namespace product_of_terms_l2875_287571

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = r * b n

/-- The main theorem -/
theorem product_of_terms (a b : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  3 * a 1 - (a 8)^2 + 3 * a 15 = 0 →
  a 8 = b 10 →
  b 3 * b 17 = 36 := by
  sorry

end product_of_terms_l2875_287571


namespace dr_strange_food_choices_l2875_287588

/-- Represents the number of food items and days --/
def n : ℕ := 12

/-- Represents the ways to choose food items each day --/
def choices : ℕ → ℕ
  | 0 => 2  -- First day has 2 choices
  | i => 2  -- Each subsequent day has 2 choices

/-- The total number of ways to choose food items over n days --/
def totalWays : ℕ := 2^n

theorem dr_strange_food_choices :
  totalWays = 2048 := by sorry

end dr_strange_food_choices_l2875_287588


namespace subtracted_number_l2875_287581

theorem subtracted_number (x : ℕ) (some_number : ℕ) 
  (h1 : x = 88320) 
  (h2 : x + 1315 + 9211 - some_number = 11901) : 
  some_number = 86945 := by
  sorry

end subtracted_number_l2875_287581


namespace rectangle_area_equals_perimeter_l2875_287527

theorem rectangle_area_equals_perimeter (b : ℝ) (h1 : b > 0) :
  let l := 3 * b
  let area := l * b
  let perimeter := 2 * (l + b)
  area = perimeter → b = 8/3 ∧ l = 8 := by
sorry

end rectangle_area_equals_perimeter_l2875_287527


namespace problem_solution_l2875_287598

-- Define the function f(x) = ax^3 + bx^2
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

-- Define the derivative of f
def f_deriv (a b x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x

theorem problem_solution :
  ∀ (a b : ℝ),
    (f a b 1 = 4 ∧ f_deriv a b 1 = 9) →
    (a = 1 ∧ b = 3) ∧
    ∀ (m : ℝ),
      (∀ x ∈ Set.Icc m (m + 1), f_deriv 1 3 x ≥ 0) →
      (m ≥ 0 ∨ m ≤ -3) :=
by sorry

end problem_solution_l2875_287598


namespace each_person_receives_eight_doughnuts_l2875_287554

/-- The number of doughnuts each person receives when Samuel and Cathy share their doughnuts. -/
def doughnuts_per_person : ℕ :=
  let samuel_doughnuts : ℕ := 2 * 12
  let cathy_doughnuts : ℕ := 4 * 12
  let total_doughnuts : ℕ := samuel_doughnuts + cathy_doughnuts
  let total_people : ℕ := 10
  let dieting_friends : ℕ := 1
  let sharing_people : ℕ := total_people - dieting_friends
  total_doughnuts / sharing_people

/-- Theorem stating that each person receives 8 doughnuts. -/
theorem each_person_receives_eight_doughnuts : doughnuts_per_person = 8 := by
  sorry

end each_person_receives_eight_doughnuts_l2875_287554


namespace line_slope_intercept_product_l2875_287543

/-- Given a line passing through points (0, -2) and (1, 1), 
    prove that the product of its slope and y-intercept equals -6 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x : ℝ, b = -2 ∧ m * 0 + b = -2) →  -- line passes through (0, -2)
    (∀ x : ℝ, m * 1 + b = 1) →            -- line passes through (1, 1)
    m * b = -6 := by
  sorry

end line_slope_intercept_product_l2875_287543


namespace dan_bought_five_notebooks_l2875_287565

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  totalSpent : ℕ
  backpackCost : ℕ
  penCost : ℕ
  pencilCost : ℕ
  notebookCost : ℕ

/-- Calculates the number of notebooks bought -/
def notebooksBought (supplies : SchoolSupplies) : ℕ :=
  (supplies.totalSpent - (supplies.backpackCost + supplies.penCost + supplies.pencilCost)) / supplies.notebookCost

/-- Theorem stating that Dan bought 5 notebooks -/
theorem dan_bought_five_notebooks (supplies : SchoolSupplies)
  (h1 : supplies.totalSpent = 32)
  (h2 : supplies.backpackCost = 15)
  (h3 : supplies.penCost = 1)
  (h4 : supplies.pencilCost = 1)
  (h5 : supplies.notebookCost = 3) :
  notebooksBought supplies = 5 := by
  sorry

end dan_bought_five_notebooks_l2875_287565


namespace circle_tangent_and_chord_l2875_287574

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 25

-- Define point P
def P : ℝ × ℝ := (3, 4)

-- Define the tangent line l
def l (x y : ℝ) : Prop := 3*x + 4*y - 25 = 0

-- Define line m
def m (x y : ℝ) : Prop := x = 3 ∨ 7*x - 24*y + 75 = 0

-- Theorem statement
theorem circle_tangent_and_chord :
  (∀ x y, C x y → l x y → (x, y) = P) ∧
  (∀ x y, m x y → 
    (∃ x1 y1 x2 y2, C x1 y1 ∧ C x2 y2 ∧ m x1 y1 ∧ m x2 y2 ∧ 
     (x1 - x2)^2 + (y1 - y2)^2 = 64) ∧
    (x, y) = P) := by sorry

end circle_tangent_and_chord_l2875_287574


namespace event_B_more_likely_l2875_287531

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event A: some number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by
  sorry

end event_B_more_likely_l2875_287531


namespace summer_sun_salutations_l2875_287558

/-- The number of sun salutations Summer performs each weekday -/
def sun_salutations_per_weekday : ℕ :=
  1300 / (365 / 7 * 5)

/-- Theorem stating that Summer performs 5 sun salutations each weekday -/
theorem summer_sun_salutations :
  sun_salutations_per_weekday = 5 := by
  sorry

end summer_sun_salutations_l2875_287558


namespace twenty_five_percent_less_than_80_l2875_287586

theorem twenty_five_percent_less_than_80 (x : ℝ) : 
  (60 : ℝ) = 80 * (3/4) → x + x/4 = 60 → x = 48 := by sorry

end twenty_five_percent_less_than_80_l2875_287586


namespace unique_solution_for_F_l2875_287532

/-- The function F as defined in the problem -/
def F (a b c : ℝ) : ℝ := a * b^3 + c

/-- Theorem stating that -5/19 is the unique solution for a in F(a, 2, 3) = F(a, 3, 8) -/
theorem unique_solution_for_F :
  ∃! a : ℝ, F a 2 3 = F a 3 8 ∧ a = -5/19 := by
  sorry

end unique_solution_for_F_l2875_287532


namespace range_of_x_l2875_287508

theorem range_of_x (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  (∃ x : ℝ, ∀ y : ℝ, (∃ a' b' c' : ℝ, a'^2 + 2*b'^2 + 3*c'^2 = 6 ∧ a' + 2*b' + 3*c' > |y + 1|) ↔ -7 < y ∧ y < 5) :=
sorry

end range_of_x_l2875_287508


namespace binomial_expansion_sum_l2875_287567

theorem binomial_expansion_sum (a b c d e f : ℤ) : 
  (∀ x : ℤ, (x - 2)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  16*(a + b) + 4*(c + d) + (e + f) = -256 := by
sorry

end binomial_expansion_sum_l2875_287567


namespace sum_in_range_l2875_287578

theorem sum_in_range : 
  let sum := 3 + 3/8 + 4 + 2/5 + 6 + 1/11
  13 < sum ∧ sum < 14 := by
  sorry

end sum_in_range_l2875_287578


namespace gcd_of_117_and_182_l2875_287533

theorem gcd_of_117_and_182 :
  Nat.gcd 117 182 = 13 := by
  sorry

end gcd_of_117_and_182_l2875_287533


namespace graph_is_pair_of_straight_lines_l2875_287580

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_straight_lines :
  ∃ (f g : ℝ → ℝ), 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end graph_is_pair_of_straight_lines_l2875_287580


namespace quadratic_fit_coefficient_l2875_287516

/-- Given three points (1, y₁), (2, y₂), and (3, y₃), the coefficient 'a' of the 
    quadratic equation y = ax² + bx + c that best fits these points is equal to 
    (y₃ - 2y₂ + y₁) / 2. -/
theorem quadratic_fit_coefficient (y₁ y₂ y₃ : ℝ) : 
  ∃ (a b c : ℝ), 
    (a * 1^2 + b * 1 + c = y₁) ∧ 
    (a * 2^2 + b * 2 + c = y₂) ∧ 
    (a * 3^2 + b * 3 + c = y₃) ∧ 
    (a = (y₃ - 2 * y₂ + y₁) / 2) := by
  sorry


end quadratic_fit_coefficient_l2875_287516


namespace new_person_weight_is_81_l2875_287557

/-- The weight of a new person replacing one in a group, given the average weight increase --/
def new_person_weight (n : ℕ) (avg_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + n * avg_increase

/-- Theorem: The weight of the new person is 81 kg --/
theorem new_person_weight_is_81 :
  new_person_weight 8 2 65 = 81 := by
  sorry

end new_person_weight_is_81_l2875_287557


namespace problem_solution_l2875_287573

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- The absolute value of m is 2
  : (a + b) / (4 * m) + 2 * m^2 - 3 * c * d = 5 := by
  sorry

end problem_solution_l2875_287573


namespace prime_square_remainders_mod_180_l2875_287592

theorem prime_square_remainders_mod_180 :
  ∃ (S : Finset Nat), 
    (∀ p : Nat, Prime p → p > 5 → ∃ r ∈ S, p^2 % 180 = r) ∧ 
    S.card = 2 := by
  sorry

end prime_square_remainders_mod_180_l2875_287592


namespace problem_statement_l2875_287519

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end problem_statement_l2875_287519


namespace vector_addition_and_scalar_multiplication_l2875_287563

/-- Given vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_addition_and_scalar_multiplication (a b : ℝ × ℝ × ℝ) :
  a = (3, -2, 1) →
  b = (-2, 4, 0) →
  a + 2 • b = (-1, 6, 1) := by
sorry

end vector_addition_and_scalar_multiplication_l2875_287563


namespace prime_divisor_form_l2875_287500

theorem prime_divisor_form (n : ℕ) (q : ℕ) (h_prime : Nat.Prime q) (h_divides : q ∣ 2^(2^n) + 1) :
  ∃ x : ℤ, q = 2^(n + 1) * x + 1 :=
sorry

end prime_divisor_form_l2875_287500


namespace min_omega_value_l2875_287576

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) : 
  (∀ x, 2 * Real.cos (ω * (x - π/5) + π/5) = 2 * Real.sin (ω * x + π/5)) →
  ω ≥ 5/2 ∧ (∀ ω' > 0, (∀ x, 2 * Real.cos (ω' * (x - π/5) + π/5) = 2 * Real.sin (ω' * x + π/5)) → ω' ≥ ω) :=
by sorry

end min_omega_value_l2875_287576


namespace orchestra_overlap_l2875_287502

theorem orchestra_overlap (total : ℕ) (violin : ℕ) (keyboard : ℕ) (neither : ℕ) : 
  total = 42 → violin = 25 → keyboard = 22 → neither = 3 →
  violin + keyboard - (total - neither) = 8 :=
by sorry

end orchestra_overlap_l2875_287502


namespace perpendicular_slope_l2875_287530

theorem perpendicular_slope (x y : ℝ) :
  let original_line := {(x, y) | 5 * x - 2 * y = 10}
  let original_slope : ℝ := 5 / 2
  let perpendicular_slope : ℝ := -1 / original_slope
  perpendicular_slope = -2 / 5 := by sorry

end perpendicular_slope_l2875_287530


namespace exists_injection_with_property_l2875_287582

-- Define the set A as a finite type
variable {A : Type} [Finite A]

-- Define the set S as a predicate on triples of elements from A
variable (S : A → A → A → Prop)

-- State the conditions on S
variable (h1 : ∀ a b c : A, S a b c ↔ S b c a)
variable (h2 : ∀ a b c : A, S a b c ↔ ¬S c b a)
variable (h3 : ∀ a b c d : A, (S a b c ∧ S c d a) ↔ (S b c d ∧ S d a b))

-- State the theorem
theorem exists_injection_with_property :
  ∃ g : A → ℝ, Function.Injective g ∧
    ∀ a b c : A, g a < g b ∧ g b < g c → S a b c :=
sorry

end exists_injection_with_property_l2875_287582


namespace special_triangle_b_value_l2875_287515

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = 4 ∧ t.b + t.c = 6 ∧ t.b < t.c ∧ Real.cos t.A = 1/2

theorem special_triangle_b_value (t : Triangle) (h : SpecialTriangle t) : t.b = 5/2 := by
  sorry

end special_triangle_b_value_l2875_287515
