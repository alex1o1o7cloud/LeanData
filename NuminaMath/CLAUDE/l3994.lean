import Mathlib

namespace NUMINAMATH_CALUDE_max_guaranteed_rectangle_area_l3994_399443

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a rectangle on the chessboard -/
structure Rectangle :=
  (top_left : Nat × Nat)
  (width : Nat)
  (height : Nat)

/-- Check if a rectangle fits on the chessboard without overlapping removed squares -/
def Rectangle.fits (board : Chessboard) (rect : Rectangle) : Prop :=
  rect.top_left.1 + rect.width ≤ board.size ∧
  rect.top_left.2 + rect.height ≤ board.size ∧
  ∀ x y, rect.top_left.1 ≤ x ∧ x < rect.top_left.1 + rect.width ∧
         rect.top_left.2 ≤ y ∧ y < rect.top_left.2 + rect.height →
         (x, y) ∉ board.removed

/-- The main theorem -/
theorem max_guaranteed_rectangle_area (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.removed.card = 8) : 
  (∀ n > 8, ∃ rect : Rectangle, rect.width * rect.height = n → ¬rect.fits board) ∧ 
  (∃ rect : Rectangle, rect.width * rect.height = 8 ∧ rect.fits board) :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_rectangle_area_l3994_399443


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l3994_399476

-- Part 1
theorem inequality_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := by sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_one : a + b + c = 1) :
  a * b + b * c + a * c ≤ 1 / 3 := by sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l3994_399476


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l3994_399469

theorem complex_square_root_of_negative_four (z : ℂ) : 
  z^2 = -4 ∧ z.im > 0 → z = 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_four_l3994_399469


namespace NUMINAMATH_CALUDE_find_N_l3994_399483

theorem find_N (a b c N : ℚ) 
  (sum_eq : a + b + c = 120)
  (a_eq : a - 10 = N)
  (b_eq : 10 * b = N)
  (c_eq : c - 10 = N) :
  N = 1100 / 21 := by
sorry

end NUMINAMATH_CALUDE_find_N_l3994_399483


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l3994_399423

theorem rectangular_prism_width 
  (length : ℝ) 
  (height : ℝ) 
  (diagonal : ℝ) 
  (width : ℝ) 
  (h1 : length = 5) 
  (h2 : height = 8) 
  (h3 : diagonal = 10) 
  (h4 : diagonal ^ 2 = length ^ 2 + width ^ 2 + height ^ 2) : 
  width = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l3994_399423


namespace NUMINAMATH_CALUDE_num_distinguishable_triangles_l3994_399496

/-- Represents the number of available colors for triangles -/
def numColors : ℕ := 8

/-- Represents the number of corner triangles in the large triangle -/
def numCorners : ℕ := 3

/-- Represents the number of triangles between center and corner -/
def numBetween : ℕ := 1

/-- Represents the number of center triangles -/
def numCenter : ℕ := 1

/-- Calculates the number of ways to choose corner colors -/
def cornerColorings : ℕ := 
  numColors + (numColors.choose 1 * (numColors - 1).choose 1) + numColors.choose numCorners

/-- Theorem: The number of distinguishable large equilateral triangles is 7680 -/
theorem num_distinguishable_triangles : 
  cornerColorings * numColors^(numBetween + numCenter) = 7680 := by sorry

end NUMINAMATH_CALUDE_num_distinguishable_triangles_l3994_399496


namespace NUMINAMATH_CALUDE_inequality_proof_l3994_399433

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b^3 * c)) - (a / b^2) ≥ (c / b) - (c^2 / a) ∧
  ((a^2 / (b^3 * c)) - (a / b^2) = (c / b) - (c^2 / a) ↔ a = b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3994_399433


namespace NUMINAMATH_CALUDE_paint_left_is_four_liters_l3994_399414

/-- The amount of paint Dexter used in gallons -/
def dexter_paint : ℚ := 3/8

/-- The amount of paint Jay used in gallons -/
def jay_paint : ℚ := 5/8

/-- The conversion factor from gallons to liters -/
def gallon_to_liter : ℚ := 4

/-- The total amount of paint in gallons -/
def total_paint : ℚ := 2

theorem paint_left_is_four_liters : 
  (total_paint * gallon_to_liter) - ((dexter_paint + jay_paint) * gallon_to_liter) = 4 := by
  sorry

end NUMINAMATH_CALUDE_paint_left_is_four_liters_l3994_399414


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3994_399480

theorem minimum_value_theorem (x : ℝ) (h : x > 4) :
  (x + 11) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 15 ∧
  (∃ x₀ > 4, (x₀ + 11) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 15 ∧ x₀ = 19) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3994_399480


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3994_399401

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 seats -/
def are_adjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 seats -/
def are_across (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Checks if a seating arrangement is valid according to the problem constraints -/
def is_valid_arrangement (arr : SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12) : Prop :=
  ∀ i j : Fin 12,
    (i ≠ j) →
    (¬are_adjacent (arr i) (arr j)) ∧
    (¬are_across (arr i) (arr j)) ∧
    (∀ k : Fin 6, (couples k).1 ≠ i ∨ (couples k).2 ≠ j)

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12),
    (∀ arr ∈ arrangements, is_valid_arrangement arr couples) ∧
    arrangements.card = 1440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3994_399401


namespace NUMINAMATH_CALUDE_wire_resistance_theorem_l3994_399418

/-- The resistance of a wire loop -/
def wire_loop_resistance (R : ℝ) : ℝ := R

/-- The distance between points A and B -/
def distance_AB : ℝ := 2

/-- The resistance of one meter of wire -/
def wire_resistance_per_meter (R : ℝ) : ℝ := R

/-- Theorem: The resistance of one meter of wire is equal to the total resistance of the wire loop -/
theorem wire_resistance_theorem (R : ℝ) :
  wire_loop_resistance R = wire_resistance_per_meter R :=
by sorry

end NUMINAMATH_CALUDE_wire_resistance_theorem_l3994_399418


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3994_399472

theorem adult_ticket_cost (num_adults : ℕ) (child_ticket_cost : ℚ) (total_receipts : ℚ) :
  num_adults = 152 →
  child_ticket_cost = 5/2 →
  total_receipts = 1026 →
  ∃ adult_ticket_cost : ℚ,
    adult_ticket_cost * num_adults + child_ticket_cost * (num_adults / 2) = total_receipts ∧
    adult_ticket_cost = 11/2 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3994_399472


namespace NUMINAMATH_CALUDE_smallest_x_value_l3994_399413

theorem smallest_x_value (x : ℝ) : 
  (2 * x^2 + 24 * x - 60 = x * (x + 13)) → x ≥ -15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l3994_399413


namespace NUMINAMATH_CALUDE_max_stamps_problem_l3994_399477

/-- The maximum number of stamps that can be bought -/
def max_stamps (initial_money : ℕ) (bus_ticket_cost : ℕ) (stamp_price : ℕ) : ℕ :=
  ((initial_money * 100 - bus_ticket_cost) / stamp_price : ℕ)

/-- Theorem: Given $50 initial money, 180 cents bus ticket cost, and 45 cents stamp price,
    the maximum number of stamps that can be bought is 107 -/
theorem max_stamps_problem : max_stamps 50 180 45 = 107 := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_problem_l3994_399477


namespace NUMINAMATH_CALUDE_smallest_multiple_l3994_399417

theorem smallest_multiple (x : ℕ) : x = 432 ↔ 
  (x > 0 ∧ 500 * x % 864 = 0 ∧ ∀ y : ℕ, y > 0 → 500 * y % 864 = 0 → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3994_399417


namespace NUMINAMATH_CALUDE_larger_number_problem_l3994_399463

theorem larger_number_problem (x y : ℝ) : 
  y = 2 * x - 3 → x + y = 51 → max x y = 33 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3994_399463


namespace NUMINAMATH_CALUDE_sue_dog_walking_charge_l3994_399409

/-- The amount Sue charged per dog for walking --/
def sue_charge_per_dog (perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed : ℚ) : ℚ :=
  let christian_total := christian_initial + christian_yards * christian_yard_price
  let initial_total := christian_total + sue_initial
  let needed := perfume_cost - initial_total
  let sue_earned := needed - additional_needed
  sue_earned / sue_dogs

theorem sue_dog_walking_charge 
  (perfume_cost : ℚ)
  (christian_initial : ℚ)
  (sue_initial : ℚ)
  (christian_yards : ℚ)
  (christian_yard_price : ℚ)
  (sue_dogs : ℚ)
  (additional_needed : ℚ)
  (h1 : perfume_cost = 50)
  (h2 : christian_initial = 5)
  (h3 : sue_initial = 7)
  (h4 : christian_yards = 4)
  (h5 : christian_yard_price = 5)
  (h6 : sue_dogs = 6)
  (h7 : additional_needed = 6) :
  sue_charge_per_dog perfume_cost christian_initial sue_initial christian_yards christian_yard_price sue_dogs additional_needed = 2 :=
by sorry

end NUMINAMATH_CALUDE_sue_dog_walking_charge_l3994_399409


namespace NUMINAMATH_CALUDE_min_time_is_200_minutes_l3994_399429

/-- Represents the travel problem between two cities -/
structure TravelProblem where
  distance : ℝ
  num_people : ℕ
  num_bicycles : ℕ
  cyclist_speed : ℝ
  pedestrian_speed : ℝ

/-- Calculates the minimum travel time for the given problem -/
def min_travel_time (problem : TravelProblem) : ℝ :=
  sorry

/-- Theorem stating that the minimum travel time for the given problem is 200 minutes -/
theorem min_time_is_200_minutes :
  let problem : TravelProblem := {
    distance := 45,
    num_people := 3,
    num_bicycles := 2,
    cyclist_speed := 15,
    pedestrian_speed := 5
  }
  min_travel_time problem = 200 / 60 := by sorry

end NUMINAMATH_CALUDE_min_time_is_200_minutes_l3994_399429


namespace NUMINAMATH_CALUDE_sum_lent_is_400_l3994_399447

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem sum_lent_is_400 :
  let rate : ℚ := 4
  let time : ℚ := 8
  let principal : ℚ := 400
  simpleInterest principal rate time = principal - 272 :=
by
  sorry

#check sum_lent_is_400

end NUMINAMATH_CALUDE_sum_lent_is_400_l3994_399447


namespace NUMINAMATH_CALUDE_sum_of_abs_values_l3994_399493

theorem sum_of_abs_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a < b) → (a + b = 1 ∨ a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abs_values_l3994_399493


namespace NUMINAMATH_CALUDE_train_length_l3994_399467

/-- The length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  bridge_length = 265 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3994_399467


namespace NUMINAMATH_CALUDE_total_amount_cows_and_goats_l3994_399461

/-- The total amount spent on cows and goats -/
def total_amount (num_cows num_goats cow_price goat_price : ℕ) : ℕ :=
  num_cows * cow_price + num_goats * goat_price

/-- Theorem: The total amount spent on 2 cows at Rs. 460 each and 8 goats at Rs. 60 each is Rs. 1400 -/
theorem total_amount_cows_and_goats :
  total_amount 2 8 460 60 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_cows_and_goats_l3994_399461


namespace NUMINAMATH_CALUDE_boat_current_rate_l3994_399403

/-- Proves that given a boat with a speed of 20 km/hr in still water,
    traveling 10 km downstream in 24 minutes, the rate of the current is 5 km/hr. -/
theorem boat_current_rate :
  let boat_speed : ℝ := 20 -- km/hr
  let downstream_distance : ℝ := 10 -- km
  let downstream_time : ℝ := 24 / 60 -- hr (24 minutes converted to hours)
  ∃ current_rate : ℝ,
    (boat_speed + current_rate) * downstream_time = downstream_distance ∧
    current_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_current_rate_l3994_399403


namespace NUMINAMATH_CALUDE_lakes_country_islands_l3994_399421

/-- A connected planar graph representing the lakes and canals system -/
structure LakeSystem where
  V : ℕ  -- number of vertices (lakes)
  E : ℕ  -- number of edges (canals)
  is_connected : Bool
  is_planar : Bool

/-- The number of islands in a lake system -/
def num_islands (sys : LakeSystem) : ℕ :=
  sys.V - sys.E + 2 - 1

/-- Theorem stating the number of islands in the given lake system -/
theorem lakes_country_islands (sys : LakeSystem) 
  (h1 : sys.V = 7)
  (h2 : sys.E = 10)
  (h3 : sys.is_connected = true)
  (h4 : sys.is_planar = true) :
  num_islands sys = 4 := by
  sorry

#eval num_islands ⟨7, 10, true, true⟩

end NUMINAMATH_CALUDE_lakes_country_islands_l3994_399421


namespace NUMINAMATH_CALUDE_min_value_of_f_l3994_399488

/-- The function f(x) = 3x^2 - 18x + 2023 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

theorem min_value_of_f :
  ∃ (m : ℝ), m = 1996 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3994_399488


namespace NUMINAMATH_CALUDE_power_simplification_l3994_399458

theorem power_simplification (x : ℝ) : (5 * x^4)^3 = 125 * x^12 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l3994_399458


namespace NUMINAMATH_CALUDE_parabola_transformation_l3994_399410

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := -x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 - 3

/-- Theorem stating that the transformed parabola is equivalent to
    shifting the original parabola 2 units right and 3 units down -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 2) - 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l3994_399410


namespace NUMINAMATH_CALUDE_equilibrium_exists_l3994_399460

/-- Represents the equilibrium state of two connected vessels with different liquids -/
def EquilibriumState (H : ℝ) : Prop :=
  ∃ (h_water h_gasoline : ℝ),
    -- Initial conditions
    0 < H ∧
    -- Valve position
    0.15 * H < 0.9 * H ∧
    -- Initial liquid levels
    0.9 * H = 0.9 * H ∧
    -- Densities
    let ρ_water : ℝ := 1000
    let ρ_gasoline : ℝ := 600
    -- Equilibrium condition
    ρ_water * (0.75 * H - (0.9 * H - h_water)) = 
      ρ_water * (h_water - 0.15 * H) + ρ_gasoline * (H - h_water) ∧
    -- Final water level
    h_water = 0.69 * H ∧
    -- Final gasoline level
    h_gasoline = H

/-- Theorem stating that the equilibrium state exists for any positive vessel height -/
theorem equilibrium_exists (H : ℝ) (h_pos : 0 < H) : EquilibriumState H := by
  sorry

#check equilibrium_exists

end NUMINAMATH_CALUDE_equilibrium_exists_l3994_399460


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3994_399466

def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

def symmetric_point (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem symmetric_circle_equation : 
  ∀ x y : ℝ, original_circle (symmetric_point x y).1 (symmetric_point x y).2 ↔ x^2 + y^2 - 4*y = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3994_399466


namespace NUMINAMATH_CALUDE_integer_roots_quadratic_l3994_399412

theorem integer_roots_quadratic (m n : ℤ) : 
  (∃ x y : ℤ, (2*m - 3)*(n - 1)*x^2 + (2*m - 3)*(n - 1)*(m - n - 4)*x - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               (2*m - 3)*(n - 1)*y^2 + (2*m - 3)*(n - 1)*(m - n - 4)*y - 2*(2*m - 3)*(n - 1)*(m - n - 2) - 1 = 0 ∧
               x ≠ y) ↔
  ((m = 2 ∧ n = 2) ∨ (m = 2 ∧ n = 0)) :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_quadratic_l3994_399412


namespace NUMINAMATH_CALUDE_caramel_chews_theorem_l3994_399449

/-- Represents the distribution of candy bags -/
structure CandyDistribution where
  totalCandies : ℕ
  totalBags : ℕ
  heartsCount : ℕ
  kissesCount : ℕ
  jelliesCount : ℕ
  heartsExtra : ℕ
  jelliesMultiplier : ℚ

/-- Calculates the number of candies in caramel chews bags -/
def caramelChewsCandies (d : CandyDistribution) : ℕ :=
  let remainingBags := d.totalBags - (d.heartsCount + d.kissesCount + d.jelliesCount)
  let baseCandy := (d.totalCandies - d.heartsCount * d.heartsExtra) / d.totalBags
  remainingBags * baseCandy

/-- Theorem stating that for the given distribution, caramel chews bags contain 44 candies -/
theorem caramel_chews_theorem (d : CandyDistribution) 
  (h1 : d.totalCandies = 500)
  (h2 : d.totalBags = 20)
  (h3 : d.heartsCount = 6)
  (h4 : d.kissesCount = 8)
  (h5 : d.jelliesCount = 4)
  (h6 : d.heartsExtra = 2)
  (h7 : d.jelliesMultiplier = 3/2) :
  caramelChewsCandies d = 44 := by
  sorry

end NUMINAMATH_CALUDE_caramel_chews_theorem_l3994_399449


namespace NUMINAMATH_CALUDE_chocolate_milk_ounces_l3994_399462

/-- The number of ounces of milk in each glass of chocolate milk. -/
def milk_per_glass : ℚ := 13/2

/-- The number of ounces of chocolate syrup in each glass of chocolate milk. -/
def syrup_per_glass : ℚ := 3/2

/-- The total number of ounces in each glass of chocolate milk. -/
def total_per_glass : ℚ := milk_per_glass + syrup_per_glass

/-- Theorem stating that each glass of chocolate milk contains 8 ounces. -/
theorem chocolate_milk_ounces : total_per_glass = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_ounces_l3994_399462


namespace NUMINAMATH_CALUDE_function_range_l3994_399475

/-- Given a function f(x) = x³ - 3a²x + a where a > 0, 
    if its maximum value is positive and its minimum value is negative, 
    then a > √2/2 -/
theorem function_range (a : ℝ) (h1 : a > 0) 
  (f : ℝ → ℝ) (h2 : ∀ x, f x = x^3 - 3*a^2*x + a) 
  (h3 : ∃ M, ∀ x, f x ≤ M ∧ M > 0)  -- maximum value is positive
  (h4 : ∃ m, ∀ x, f x ≥ m ∧ m < 0)  -- minimum value is negative
  : a > Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l3994_399475


namespace NUMINAMATH_CALUDE_committee_formation_count_l3994_399425

/-- The number of ways to choose a committee under given conditions -/
def committee_formations (total_boys : ℕ) (total_girls : ℕ) (committee_size : ℕ) 
  (boys_with_event_planning : ℕ) (girls_with_leadership : ℕ) : ℕ :=
  let boys_to_choose := committee_size / 2
  let girls_to_choose := committee_size / 2
  let remaining_boys := total_boys - boys_with_event_planning
  let remaining_girls := total_girls - girls_with_leadership
  (Nat.choose remaining_boys (boys_to_choose - 1)) * 
  (Nat.choose remaining_girls (girls_to_choose - 1))

/-- Theorem stating the number of ways to form the committee -/
theorem committee_formation_count :
  committee_formations 8 6 8 1 1 = 350 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_count_l3994_399425


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l3994_399491

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_with_median (P Q R M : ℝ × ℝ) : 
  let PQ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let PR := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let PM := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let QR := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  let s := (PQ + PR + QR) / 2
  let area := Real.sqrt (s * (s - PQ) * (s - PR) * (s - QR))
  PQ = 9 →
  PR = 17 →
  PM = 13 →
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  area = A :=
by sorry

#check triangle_area_with_median

end NUMINAMATH_CALUDE_triangle_area_with_median_l3994_399491


namespace NUMINAMATH_CALUDE_min_value_theorem_l3994_399404

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (8^a * 2^b)) : 
  ∃ (min_val : ℝ), min_val = 5 + 2 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (8^x * 2^y) → 
      1/x + 2/y ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3994_399404


namespace NUMINAMATH_CALUDE_brown_class_points_l3994_399435

theorem brown_class_points (william_points mr_adams_points daniel_points : ℕ)
  (mean_points : ℚ) (total_classes : ℕ) :
  william_points = 50 →
  mr_adams_points = 57 →
  daniel_points = 57 →
  mean_points = 53.3 →
  total_classes = 4 →
  ∃ (brown_points : ℕ),
    (william_points + mr_adams_points + daniel_points + brown_points) / total_classes = mean_points ∧
    brown_points = 49 :=
by sorry

end NUMINAMATH_CALUDE_brown_class_points_l3994_399435


namespace NUMINAMATH_CALUDE_one_real_root_l3994_399428

-- Define the determinant function
def det (x c b d : ℝ) : ℝ := x^3 + (c^2 + d^2) * x

-- State the theorem
theorem one_real_root
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  ∃! x : ℝ, det x c b d = 0 :=
sorry

end NUMINAMATH_CALUDE_one_real_root_l3994_399428


namespace NUMINAMATH_CALUDE_van_rental_equation_l3994_399406

theorem van_rental_equation (x : ℕ+) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_van_rental_equation_l3994_399406


namespace NUMINAMATH_CALUDE_scooter_initial_cost_l3994_399438

theorem scooter_initial_cost (P : ℝ) : 
  (P + 300) * 1.1 = 1320 → P = 900 := by sorry

end NUMINAMATH_CALUDE_scooter_initial_cost_l3994_399438


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3994_399420

theorem geometric_series_sum (x : ℝ) :
  (|x| < 1) →
  (∑' n, x^n = 16) →
  x = 15/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3994_399420


namespace NUMINAMATH_CALUDE_child_ticket_cost_l3994_399419

theorem child_ticket_cost 
  (adult_price : ℕ) 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_attendance : ℕ) 
  (h1 : adult_price = 9)
  (h2 : total_tickets = 225)
  (h3 : total_revenue = 1875)
  (h4 : adult_attendance = 175) :
  ∃ (child_price : ℕ), 
    child_price * (total_tickets - adult_attendance) + 
    adult_price * adult_attendance = total_revenue ∧ 
    child_price = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l3994_399419


namespace NUMINAMATH_CALUDE_characterization_of_p_l3994_399489

/-- The polynomial equation in x with parameter p -/
def f (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + x^2 + 3*p*x + 1

/-- A function has at least two distinct positive real roots -/
def has_two_distinct_positive_roots (g : ℝ → ℝ) : Prop :=
  ∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ g x = 0 ∧ g y = 0

/-- The main theorem: characterization of p for which f has at least two distinct positive real roots -/
theorem characterization_of_p (p : ℝ) : 
  has_two_distinct_positive_roots (f p) ↔ p < 1/4 := by sorry

end NUMINAMATH_CALUDE_characterization_of_p_l3994_399489


namespace NUMINAMATH_CALUDE_kenny_trumpet_practice_l3994_399487

def basketball_hours : ℕ := 10

def running_hours (b : ℕ) : ℕ := 2 * b

def trumpet_hours (r : ℕ) : ℕ := 2 * r

def total_practice_hours (b r t : ℕ) : ℕ := b + r + t

theorem kenny_trumpet_practice (x y : ℕ) :
  let b := basketball_hours
  let r := running_hours b
  let t := trumpet_hours r
  total_practice_hours b r t = x + y →
  t = 40 := by
sorry

end NUMINAMATH_CALUDE_kenny_trumpet_practice_l3994_399487


namespace NUMINAMATH_CALUDE_jacket_trouser_combinations_l3994_399440

theorem jacket_trouser_combinations (jacket_styles : ℕ) (trouser_colors : ℕ) : 
  jacket_styles = 4 → trouser_colors = 3 → jacket_styles * trouser_colors = 12 := by
  sorry

end NUMINAMATH_CALUDE_jacket_trouser_combinations_l3994_399440


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3994_399411

theorem last_two_digits_sum (n : ℕ) : n = 23 →
  (7^n + 13^n) % 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3994_399411


namespace NUMINAMATH_CALUDE_smart_car_competition_probability_l3994_399481

/-- The probability of selecting exactly 4 girls when randomly choosing 10 people
    from a group of 15 people (7 girls and 8 boys) -/
theorem smart_car_competition_probability :
  let total_members : ℕ := 15
  let girls : ℕ := 7
  let boys : ℕ := total_members - girls
  let selected : ℕ := 10
  let prob_four_girls := (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected
  prob_four_girls = (Nat.choose girls 4 * Nat.choose boys 6 : ℚ) / Nat.choose total_members selected :=
by sorry

end NUMINAMATH_CALUDE_smart_car_competition_probability_l3994_399481


namespace NUMINAMATH_CALUDE_larger_integer_value_l3994_399498

theorem larger_integer_value (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 3 / 2 → 
  (a : ℕ) * b = 216 → 
  max a b = 18 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3994_399498


namespace NUMINAMATH_CALUDE_cat_addition_l3994_399442

/-- Proves that buying more cats results in the correct total number of cats. -/
theorem cat_addition (initial_cats bought_cats : ℕ) :
  initial_cats = 11 →
  bought_cats = 43 →
  initial_cats + bought_cats = 54 := by
  sorry

end NUMINAMATH_CALUDE_cat_addition_l3994_399442


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l3994_399446

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((c^2 = a^2 + b^2) ∨ (b^2 = a^2 + c^2)) → 
  ((d^2 = b^2 - a^2) ∨ (b^2 = a^2 + d^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l3994_399446


namespace NUMINAMATH_CALUDE_curve_properties_l3994_399499

/-- The curve function -/
def curve (c : ℝ) (x : ℝ) : ℝ := c * x^4 + x^2 - c

theorem curve_properties :
  ∀ (c : ℝ),
  -- The points (1, 1) and (-1, 1) lie on the curve for all values of c
  curve c 1 = 1 ∧ curve c (-1) = 1 ∧
  -- When c = -1/4, the curve is tangent to the line y = x at the point (1, 1)
  (let c := -1/4
   curve c 1 = 1 ∧ (deriv (curve c)) 1 = 1) ∧
  -- The curve intersects the line y = x at the points (1, 1) and (-1 + √2, -1 + √2)
  (∃ (x : ℝ), x ≠ 1 ∧ curve (-1/4) x = x ∧ x = -1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l3994_399499


namespace NUMINAMATH_CALUDE_two_negative_roots_l3994_399490

/-- The polynomial function we're analyzing -/
def f (q : ℝ) (x : ℝ) : ℝ := x^4 + 2*q*x^3 - 3*x^2 + 2*q*x + 1

/-- Theorem stating that for any q < 1/4, the equation f q x = 0 has at least two distinct negative real roots -/
theorem two_negative_roots (q : ℝ) (h : q < 1/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ f q x₁ = 0 ∧ f q x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_negative_roots_l3994_399490


namespace NUMINAMATH_CALUDE_f_properties_l3994_399485

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem f_properties :
  let max_value : ℝ := 1 / Real.exp 1
  ∀ (x₁ x₂ x₀ m : ℝ),
  (∀ x > 0, f x = (Real.log x) / x) →
  (∀ x > 0, f x ≤ max_value) →
  (f (Real.exp 1) = max_value) →
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f (Real.exp 1 + x) > f (Real.exp 1 - x)) →
  (f x₁ = m) →
  (f x₂ = m) →
  (x₀ = (x₁ + x₂) / 2) →
  (deriv f x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3994_399485


namespace NUMINAMATH_CALUDE_binding_cost_per_manuscript_l3994_399497

/-- Proves that the binding cost per manuscript is $5 given the specified conditions. -/
theorem binding_cost_per_manuscript
  (num_manuscripts : ℕ)
  (pages_per_manuscript : ℕ)
  (copy_cost_per_page : ℚ)
  (total_cost : ℚ)
  (h1 : num_manuscripts = 10)
  (h2 : pages_per_manuscript = 400)
  (h3 : copy_cost_per_page = 5 / 100)
  (h4 : total_cost = 250) :
  (total_cost - (num_manuscripts * pages_per_manuscript * copy_cost_per_page)) / num_manuscripts = 5 :=
by sorry

end NUMINAMATH_CALUDE_binding_cost_per_manuscript_l3994_399497


namespace NUMINAMATH_CALUDE_soccer_most_popular_l3994_399422

-- Define the list of sports
inductive Sport
  | Hockey
  | Basketball
  | Soccer
  | Volleyball
  | Badminton

-- Function to get the number of students for each sport
def students_playing (s : Sport) : ℕ :=
  match s with
  | Sport.Hockey => 30
  | Sport.Basketball => 40
  | Sport.Soccer => 50
  | Sport.Volleyball => 35
  | Sport.Badminton => 25

-- Theorem: Soccer has the highest number of students
theorem soccer_most_popular (s : Sport) : 
  students_playing Sport.Soccer ≥ students_playing s :=
sorry

end NUMINAMATH_CALUDE_soccer_most_popular_l3994_399422


namespace NUMINAMATH_CALUDE_subSubfaces_12_9_l3994_399451

/-- The number of k-dimensional sub-subfaces in an n-dimensional cube -/
def subSubfaces (n k : ℕ) : ℕ := 2^(n - k) * (Nat.choose n k)

/-- Theorem: The number of 9-dimensional sub-subfaces in a 12-dimensional cube is 1760 -/
theorem subSubfaces_12_9 : subSubfaces 12 9 = 1760 := by
  sorry

end NUMINAMATH_CALUDE_subSubfaces_12_9_l3994_399451


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_ratio_l3994_399454

theorem bicycle_helmet_cost_ratio :
  ∀ (bicycle_cost helmet_cost : ℕ),
    helmet_cost = 40 →
    bicycle_cost + helmet_cost = 240 →
    ∃ (m : ℕ), bicycle_cost = m * helmet_cost →
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_ratio_l3994_399454


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3994_399426

/-- Returns true if the given number is a palindrome in the specified base -/
def isPalindrome (n : ℕ) (base : ℕ) : Bool := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∃ (n : ℕ), n > 7 ∧
    isPalindrome n 3 = true ∧
    isPalindrome n 5 = true ∧
    (∀ (m : ℕ), m > 7 ∧ m < n →
      isPalindrome m 3 = false ∨ isPalindrome m 5 = false) ∧
    n = 26 := by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3994_399426


namespace NUMINAMATH_CALUDE_square_root_problem_l3994_399441

theorem square_root_problem (x : ℝ) : (3/5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3994_399441


namespace NUMINAMATH_CALUDE_list_length_difference_l3994_399415

/-- 
Given two lists of integers, where the second list contains all elements of the first list 
plus one additional element, prove that the difference in their lengths is 1.
-/
theorem list_length_difference (list1 list2 : List Int) (h : ∀ x, x ∈ list1 → x ∈ list2) 
  (h_additional : ∃ y, y ∈ list2 ∧ y ∉ list1) : 
  list2.length - list1.length = 1 := by
  sorry

end NUMINAMATH_CALUDE_list_length_difference_l3994_399415


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3994_399455

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 12 + a 3 * a 5 = 15 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3994_399455


namespace NUMINAMATH_CALUDE_box_volume_yards_l3994_399465

/-- Conversion factor from cubic feet to cubic yards -/
def cubic_feet_to_yards : ℝ := 27

/-- Volume of the box in cubic feet -/
def box_volume_feet : ℝ := 216

/-- Theorem stating that the volume of the box in cubic yards is 8 -/
theorem box_volume_yards : (box_volume_feet / cubic_feet_to_yards) = 8 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_yards_l3994_399465


namespace NUMINAMATH_CALUDE_scout_profit_is_250_l3994_399464

/-- Calculates the profit for a scout troop selling candy bars -/
def scout_profit (num_bars : ℕ) (buy_rate : ℚ) (sell_rate : ℚ) : ℚ :=
  let cost_per_bar := 3 / (6 : ℚ)
  let sell_per_bar := 2 / (3 : ℚ)
  let total_cost := (num_bars : ℚ) * cost_per_bar
  let total_revenue := (num_bars : ℚ) * sell_per_bar
  total_revenue - total_cost

/-- The profit for a scout troop selling 1500 candy bars is $250 -/
theorem scout_profit_is_250 :
  scout_profit 1500 (3/6) (2/3) = 250 := by
  sorry

end NUMINAMATH_CALUDE_scout_profit_is_250_l3994_399464


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3994_399424

/-- The roots of the quadratic equation x^2 - 7x + 12 = 0 -/
def roots : Set ℝ := {x : ℝ | x^2 - 7*x + 12 = 0}

/-- An isosceles triangle with two sides from the roots set -/
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 ∈ roots) ∨ (side1 = side3 ∧ side2 ∈ roots) ∨ (side2 = side3 ∧ side1 ∈ roots)
  sides_from_roots : {side1, side2, side3} ∩ roots = {side1, side2} ∨ {side1, side2, side3} ∩ roots = {side1, side3} ∨ {side1, side2, side3} ∩ roots = {side2, side3}

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of the isosceles triangle is either 10 or 11 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3994_399424


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3994_399470

noncomputable def f (x : ℝ) := Real.sin x - x

theorem solution_set_of_inequality (x : ℝ) :
  f (x + 2) + f (1 - 2*x) < 0 ↔ x < 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3994_399470


namespace NUMINAMATH_CALUDE_hotel_rooms_booked_l3994_399456

theorem hotel_rooms_booked (single_room_cost double_room_cost total_revenue double_rooms : ℕ)
  (h1 : single_room_cost = 35)
  (h2 : double_room_cost = 60)
  (h3 : total_revenue = 14000)
  (h4 : double_rooms = 196)
  : ∃ single_rooms : ℕ, single_rooms + double_rooms = 260 ∧ 
    single_room_cost * single_rooms + double_room_cost * double_rooms = total_revenue := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_booked_l3994_399456


namespace NUMINAMATH_CALUDE_smallest_n_ending_same_as_n_squared_l3994_399459

theorem smallest_n_ending_same_as_n_squared : 
  ∃ (N : ℕ), 
    N > 0 ∧ 
    (N % 1000 = N^2 % 1000) ∧ 
    (N ≥ 100) ∧
    (∀ (M : ℕ), M > 0 ∧ M < N → (M % 1000 ≠ M^2 % 1000 ∨ M < 100)) ∧ 
    N = 376 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_ending_same_as_n_squared_l3994_399459


namespace NUMINAMATH_CALUDE_contrapositive_not_always_false_l3994_399437

theorem contrapositive_not_always_false :
  ∃ (p q : Prop), (p → q) ∧ ¬(¬q → ¬p) → False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_not_always_false_l3994_399437


namespace NUMINAMATH_CALUDE_blocks_added_l3994_399444

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) :
  final_blocks - initial_blocks = 30 := by
sorry

end NUMINAMATH_CALUDE_blocks_added_l3994_399444


namespace NUMINAMATH_CALUDE_average_income_problem_l3994_399450

theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (N + O) / 2 = 6250 →
  M = 4000 →
  (M + O) / 2 = 5200 := by
  sorry

end NUMINAMATH_CALUDE_average_income_problem_l3994_399450


namespace NUMINAMATH_CALUDE_tensor_inequality_range_l3994_399432

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem tensor_inequality_range (a : ℝ) : 
  (∀ x : ℝ, tensor (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2) (3/2) :=
sorry

end NUMINAMATH_CALUDE_tensor_inequality_range_l3994_399432


namespace NUMINAMATH_CALUDE_marathon_average_time_l3994_399400

def casey_time : ℝ := 6

theorem marathon_average_time : 
  let zendaya_time := casey_time + (1/3 * casey_time)
  let total_time := casey_time + zendaya_time
  let average_time := total_time / 2
  average_time = 7 := by sorry

end NUMINAMATH_CALUDE_marathon_average_time_l3994_399400


namespace NUMINAMATH_CALUDE_trig_identities_l3994_399473

/-- Theorem: Trigonometric identities for specific angles -/
theorem trig_identities :
  (∃ (x y : ℝ), x = 263 * π / 180 ∧ y = 203 * π / 180 ∧
    Real.cos x * Real.cos y + Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1/2) ∧
  (∃ (z : ℝ), z = 8 * π / 180 ∧
    (Real.cos (7 * π / 180) - Real.sin (15 * π / 180) * Real.sin z) / Real.cos z =
    (Real.sqrt 6 + Real.sqrt 2) / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3994_399473


namespace NUMINAMATH_CALUDE_valid_placements_correct_l3994_399427

/-- Represents a chess piece type -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Represents the size of the chessboard -/
def boardSize : Nat := 8

/-- Calculates the number of ways to place two identical pieces of the given type on an 8x8 chessboard such that they do not capture each other -/
def validPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1972
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1980

/-- Theorem stating the correct number of valid placements for each piece type -/
theorem valid_placements_correct :
  (validPlacements ChessPiece.Rook = 1568) ∧
  (validPlacements ChessPiece.King = 1806) ∧
  (validPlacements ChessPiece.Bishop = 1972) ∧
  (validPlacements ChessPiece.Knight = 1848) ∧
  (validPlacements ChessPiece.Queen = 1980) :=
by sorry

end NUMINAMATH_CALUDE_valid_placements_correct_l3994_399427


namespace NUMINAMATH_CALUDE_quadratic_value_at_4_l3994_399416

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_value_at_4 
  (a b c : ℝ) 
  (h_max : ∃ (k : ℝ), quadratic a b c k = 5 ∧ ∀ x, quadratic a b c x ≤ 5)
  (h_max_at_3 : quadratic a b c 3 = 5)
  (h_at_0 : quadratic a b c 0 = -13) :
  quadratic a b c 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_4_l3994_399416


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3994_399430

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) 
  (h1 : contained_in m α) 
  (h2 : parallel α β) : 
  line_parallel_to_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l3994_399430


namespace NUMINAMATH_CALUDE_intersection_value_l3994_399436

theorem intersection_value (m n : ℝ) (h1 : n = 3 / m) (h2 : n = m + 1) :
  (m - n)^2 * (1 / n - 1 / m) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l3994_399436


namespace NUMINAMATH_CALUDE_cube_root_of_64_l3994_399457

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l3994_399457


namespace NUMINAMATH_CALUDE_valid_seating_count_l3994_399468

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 chairs -/
def isAdjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 chairs -/
def isAcross (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Represents a valid seating arrangement for 6 married couples -/
def ValidSeating (s : SeatingArrangement) : Prop :=
  ∀ i j : Fin 12,
    -- Men and women alternate
    (i.val % 2 = 0 → s i < 6) ∧
    (i.val % 2 = 1 → s i ≥ 6) ∧
    -- No one sits next to or across from their spouse
    (s i < 6 ∧ s j ≥ 6 ∧ s i + 6 = s j →
      ¬(isAdjacent i j ∨ isAcross i j))

/-- The number of valid seating arrangements -/
def numValidSeatings : ℕ := sorry

theorem valid_seating_count : numValidSeatings = 5184 := by sorry

end NUMINAMATH_CALUDE_valid_seating_count_l3994_399468


namespace NUMINAMATH_CALUDE_polynomial_lower_bound_l3994_399495

theorem polynomial_lower_bound (x : ℝ) : x^4 - 4*x^3 + 8*x^2 - 8*x + 5 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_lower_bound_l3994_399495


namespace NUMINAMATH_CALUDE_monotonic_increasing_sequence_l3994_399405

/-- A sequence {a_n} with general term a_n = n^2 + bn is monotonically increasing if and only if b > -3 -/
theorem monotonic_increasing_sequence (b : ℝ) :
  (∀ n : ℕ, (n : ℝ)^2 + b * n < ((n + 1) : ℝ)^2 + b * (n + 1)) ↔ b > -3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_sequence_l3994_399405


namespace NUMINAMATH_CALUDE_circle_intersection_distance_l3994_399492

-- Define the circles and their properties
variable (r R : ℝ)
variable (d : ℝ)

-- Hypotheses
variable (h1 : r > 0)
variable (h2 : R > 0)
variable (h3 : r < R)
variable (h4 : d > 0)

-- Define the intersection property
variable (intersection : ∃ (x : ℝ × ℝ), (x.1^2 + x.2^2 = r^2) ∧ ((x.1 - d)^2 + x.2^2 = R^2))

-- Theorem statement
theorem circle_intersection_distance : R - r < d ∧ d < r + R := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_distance_l3994_399492


namespace NUMINAMATH_CALUDE_horner_v5_equals_761_l3994_399434

def f (x : ℝ) : ℝ := 3 * x^9 + 3 * x^6 + 5 * x^4 + x^3 + 7 * x^2 + 3 * x + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

def coefficients : List ℝ := [1, 3, 7, 1, 5, 0, 3, 0, 0, 3]

theorem horner_v5_equals_761 :
  let x : ℝ := 3
  let v₅ := (horner_method (coefficients.take 6) x)
  v₅ = 761 := by sorry

end NUMINAMATH_CALUDE_horner_v5_equals_761_l3994_399434


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l3994_399471

/-- Represents a six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n ≤ 999999 }

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n ≤ 99999 }

/-- Function that removes one digit from a six-digit number to form a five-digit number -/
def removeOneDigit (n : SixDigitNumber) : FiveDigitNumber :=
  sorry

/-- The problem statement -/
theorem unique_six_digit_number :
  ∃! (n : SixDigitNumber), 
    ∀ (m : FiveDigitNumber), 
      (m = removeOneDigit n) → (n.val - m.val = 654321) := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l3994_399471


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_dormitory_to_city_distance_proof_l3994_399431

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun total_distance =>
    (1/6 : ℝ) * total_distance +
    (1/4 : ℝ) * total_distance +
    (1/3 : ℝ) * total_distance +
    10 +
    (1/12 : ℝ) * total_distance = total_distance →
    total_distance = 60

-- The proof is omitted
theorem dormitory_to_city_distance_proof : dormitory_to_city_distance 60 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_dormitory_to_city_distance_proof_l3994_399431


namespace NUMINAMATH_CALUDE_equation_solution_l3994_399482

theorem equation_solution : ∃! x : ℝ, (x + 1 ≠ 0 ∧ 2*x - 1 ≠ 0) ∧ (2 / (x + 1) = 3 / (2*x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3994_399482


namespace NUMINAMATH_CALUDE_parallelogram_revolution_surface_area_l3994_399474

/-- The surface area of a solid of revolution formed by rotating a parallelogram -/
theorem parallelogram_revolution_surface_area
  (p d : ℝ)
  (perimeter_positive : p > 0)
  (diagonal_positive : d > 0) :
  let perimeter := 2 * p
  let diagonal := d
  let surface_area := 2 * Real.pi * d * p
  surface_area = 2 * Real.pi * diagonal * (perimeter / 2) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_revolution_surface_area_l3994_399474


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3994_399448

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (Complex.I - 1) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3994_399448


namespace NUMINAMATH_CALUDE_roots_properties_l3994_399478

theorem roots_properties (x : ℝ) : 
  (x^2 - 7 * |x| + 6 = 0) → 
  (∃ (roots : Finset ℝ), 
    (∀ r ∈ roots, r^2 - 7 * |r| + 6 = 0) ∧ 
    (Finset.sum roots id = 0) ∧ 
    (Finset.prod roots id = 36)) :=
by sorry

end NUMINAMATH_CALUDE_roots_properties_l3994_399478


namespace NUMINAMATH_CALUDE_sandys_savings_ratio_l3994_399453

/-- The ratio of Sandy's savings this year to last year -/
theorem sandys_savings_ratio (S1 D1 : ℝ) (S1_pos : 0 < S1) (D1_pos : 0 < D1) :
  let Y := 0.06 * S1 + 0.08 * D1
  let X := 0.099 * S1 + 0.126 * D1
  X / Y = (0.099 + 0.126) / (0.06 + 0.08) := by
  sorry

end NUMINAMATH_CALUDE_sandys_savings_ratio_l3994_399453


namespace NUMINAMATH_CALUDE_investment_problem_l3994_399407

/-- Proves that the total investment amount is $5,400 given the problem conditions -/
theorem investment_problem (total : ℝ) (amount_at_8_percent : ℝ) (amount_at_10_percent : ℝ)
  (h1 : amount_at_8_percent = 3000)
  (h2 : total = amount_at_8_percent + amount_at_10_percent)
  (h3 : amount_at_8_percent * 0.08 = amount_at_10_percent * 0.10) :
  total = 5400 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3994_399407


namespace NUMINAMATH_CALUDE_closed_set_properties_l3994_399445

-- Define what it means for a set to be closed
def is_closed (M : Set ℤ) : Prop :=
  ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-2, -1, 0, 1, 2}
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set of positive integers
def positive_integers : Set ℤ := {n : ℤ | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed M) ∧
  (¬ is_closed positive_integers) ∧
  (is_closed M_3k) ∧
  (∃ A₁ A₂ : Set ℤ, is_closed A₁ ∧ is_closed A₂ ∧ ¬ is_closed (A₁ ∪ A₂)) := by
  sorry

end NUMINAMATH_CALUDE_closed_set_properties_l3994_399445


namespace NUMINAMATH_CALUDE_real_roots_condition_l3994_399479

theorem real_roots_condition (a b : ℝ) : 
  (∃ x : ℝ, (1 - a * x) / (1 + a * x) * Real.sqrt ((1 + b * x) / (1 - b * x)) = 1) ↔ 
  (1 / 2 < a / b ∧ a / b < 1) :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_l3994_399479


namespace NUMINAMATH_CALUDE_min_value_neg_half_l3994_399486

/-- A function f with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

/-- The maximum value of f on (0, +∞) -/
def max_value : ℝ := 5

/-- Theorem: The minimum value of f on (-∞, 0) is -1 -/
theorem min_value_neg_half (a b : ℝ) :
  (∀ x > 0, f a b x ≤ max_value) →
  (∃ x > 0, f a b x = max_value) →
  (∀ x < 0, f a b x ≥ -1) ∧
  (∃ x < 0, f a b x = -1) :=
sorry

end NUMINAMATH_CALUDE_min_value_neg_half_l3994_399486


namespace NUMINAMATH_CALUDE_distance_to_point_l3994_399494

theorem distance_to_point : Real.sqrt ((-12 - 0)^2 + (16 - 0)^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_l3994_399494


namespace NUMINAMATH_CALUDE_janous_inequality_l3994_399439

theorem janous_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l3994_399439


namespace NUMINAMATH_CALUDE_trapezoid_area_l3994_399402

/-- The area of a trapezoid given the areas of the triangles adjacent to its bases -/
theorem trapezoid_area (K₁ K₂ : ℝ) (h₁ : K₁ > 0) (h₂ : K₂ > 0) :
  ∃ (A : ℝ), A = K₁ + K₂ + 2 * Real.sqrt (K₁ * K₂) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3994_399402


namespace NUMINAMATH_CALUDE_bens_class_girls_l3994_399452

theorem bens_class_girls (total : ℕ) (girl_ratio boy_ratio : ℕ) (h1 : total = 35) (h2 : girl_ratio = 3) (h3 : boy_ratio = 4) :
  ∃ (girls boys : ℕ), girls + boys = total ∧ girls * boy_ratio = boys * girl_ratio ∧ girls = 15 := by
sorry

end NUMINAMATH_CALUDE_bens_class_girls_l3994_399452


namespace NUMINAMATH_CALUDE_petes_son_age_l3994_399484

/-- Given Pete's current age and the relationship between Pete's and his son's ages in 4 years,
    this theorem proves the current age of Pete's son. -/
theorem petes_son_age (pete_age : ℕ) (h : pete_age = 35) :
  ∃ (son_age : ℕ), son_age = 9 ∧ pete_age + 4 = 3 * (son_age + 4) := by
  sorry

end NUMINAMATH_CALUDE_petes_son_age_l3994_399484


namespace NUMINAMATH_CALUDE_absolute_value_of_x_minus_five_l3994_399408

theorem absolute_value_of_x_minus_five (x : ℝ) (h : x = 4) : |x - 5| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_x_minus_five_l3994_399408
