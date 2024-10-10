import Mathlib

namespace oncoming_train_speed_l873_87303

/-- Given two trains passing each other, calculate the speed of the oncoming train -/
theorem oncoming_train_speed
  (v₁ : ℝ)  -- Speed of the passenger's train in km/h
  (l : ℝ)   -- Length of the oncoming train in meters
  (t : ℝ)   -- Time taken for the oncoming train to pass in seconds
  (h₁ : v₁ = 40)  -- The speed of the passenger's train is 40 km/h
  (h₂ : l = 75)   -- The length of the oncoming train is 75 meters
  (h₃ : t = 3)    -- The time taken to pass is 3 seconds
  : ∃ v₂ : ℝ, v₂ = 50 ∧ l / 1000 = (v₁ + v₂) * (t / 3600) :=
sorry

end oncoming_train_speed_l873_87303


namespace hexagon_perimeter_l873_87373

/-- An equilateral hexagon with specific properties -/
structure EquilateralHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that three nonadjacent interior angles are 60°
  angle_property : True
  -- The area of the hexagon is 9√3
  area_eq : side^2 * Real.sqrt 3 = 9 * Real.sqrt 3

/-- The perimeter of an equilateral hexagon is 18 given the specified conditions -/
theorem hexagon_perimeter (h : EquilateralHexagon) : h.side * 6 = 18 := by
  sorry

end hexagon_perimeter_l873_87373


namespace missing_consonants_fraction_l873_87388

theorem missing_consonants_fraction 
  (total_letters : ℕ) 
  (total_vowels : ℕ) 
  (total_missing : ℕ) 
  (missing_vowels : ℕ) 
  (h1 : total_letters = 26) 
  (h2 : total_vowels = 5) 
  (h3 : total_missing = 5) 
  (h4 : missing_vowels = 2) :
  (total_missing - missing_vowels) / (total_letters - total_vowels) = 1 / 7 := by
sorry

end missing_consonants_fraction_l873_87388


namespace recipe_flour_amount_l873_87309

/-- The amount of flour in cups that Mary has already added to the recipe. -/
def flour_already_added : ℕ := 2

/-- The amount of flour in cups that Mary needs to add to the recipe. -/
def flour_to_be_added : ℕ := 5

/-- The total amount of flour in cups that the recipe calls for. -/
def total_flour : ℕ := flour_already_added + flour_to_be_added

theorem recipe_flour_amount :
  total_flour = 7 :=
by sorry

end recipe_flour_amount_l873_87309


namespace probability_tamika_greater_carlos_l873_87353

def tamika_set : Finset ℕ := {8, 9, 10, 11}
def carlos_set : Finset ℕ := {3, 5, 6, 7}

def tamika_result (a b : ℕ) : ℕ := a + b

def carlos_result (a b : ℕ) : ℕ := a * b - 2

def valid_pair (s : Finset ℕ) (a b : ℕ) : Prop :=
  a ∈ s ∧ b ∈ s ∧ a ≠ b

def favorable_outcomes : ℕ := 26
def total_outcomes : ℕ := 36

theorem probability_tamika_greater_carlos :
  (↑favorable_outcomes / ↑total_outcomes : ℚ) = 13 / 18 := by sorry

end probability_tamika_greater_carlos_l873_87353


namespace triangle_side_length_l873_87393

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define a median
def is_median (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ × ℝ), m = ((t.B.1 + t.C.1) / 2, (t.B.2 + t.C.2) / 2) ∧ M = m

theorem triangle_side_length (t : Triangle) :
  length t.A t.B = 7 →
  length t.B t.C = 10 →
  (∃ (M : ℝ × ℝ), is_median t M ∧ length t.A M = 5) →
  length t.A t.C = Real.sqrt 51 :=
sorry

end triangle_side_length_l873_87393


namespace tan_fifteen_pi_fourths_l873_87378

theorem tan_fifteen_pi_fourths : Real.tan (15 * π / 4) = -1 := by
  sorry

end tan_fifteen_pi_fourths_l873_87378


namespace exponent_equality_l873_87306

theorem exponent_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 7) : x = 14 := by
  sorry

end exponent_equality_l873_87306


namespace least_difference_l873_87307

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem least_difference (x y z : ℕ) : 
  x < y → 
  y < z → 
  y - x > 5 → 
  Even x → 
  x % 3 = 0 → 
  Odd y → 
  Odd z → 
  is_prime y → 
  y > 20 → 
  z % 5 = 0 → 
  1 < x → 
  x < 30 → 
  (∀ x' y' z' : ℕ, 
    x' < y' → 
    y' < z' → 
    y' - x' > 5 → 
    Even x' → 
    x' % 3 = 0 → 
    Odd y' → 
    Odd z' → 
    is_prime y' → 
    y' > 20 → 
    z' % 5 = 0 → 
    1 < x' → 
    x' < 30 → 
    z - x ≤ z' - x') → 
  z - x = 19 := by
sorry

end least_difference_l873_87307


namespace quadratic_root_range_l873_87398

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x, x^2 + a*x - 2 = 0) ∧ 
  (x₁^2 + a*x₁ - 2 = 0) ∧ 
  (x₂^2 + a*x₂ - 2 = 0) ∧ 
  (x₁ < 1) ∧ (1 < x₂) →
  a < 1 :=
sorry

end quadratic_root_range_l873_87398


namespace exploration_writing_ratio_l873_87361

theorem exploration_writing_ratio :
  let exploring_time : ℝ := 3
  let book_writing_time : ℝ := 0.5
  let total_time : ℝ := 5
  let notes_writing_time : ℝ := total_time - exploring_time - book_writing_time
  (notes_writing_time / exploring_time = 1 / 2) := by
sorry

end exploration_writing_ratio_l873_87361


namespace circus_ticket_cost_l873_87310

/-- The cost of tickets for a group attending a circus -/
def ticket_cost (adult_price : ℚ) (child_price : ℚ) (num_adults : ℕ) (num_children : ℕ) : ℚ :=
  (adult_price * num_adults) + (child_price * num_children)

/-- Theorem: The total cost of tickets for 2 adults at $44.00 each and 5 children at $28.00 each is $228.00 -/
theorem circus_ticket_cost :
  ticket_cost 44 28 2 5 = 228 := by
  sorry

end circus_ticket_cost_l873_87310


namespace debby_water_bottles_l873_87350

theorem debby_water_bottles (initial_bottles : ℕ) (days : ℕ) (remaining_bottles : ℕ) 
  (h1 : initial_bottles = 264)
  (h2 : days = 11)
  (h3 : remaining_bottles = 99) :
  (initial_bottles - remaining_bottles) / days = 15 := by
  sorry

end debby_water_bottles_l873_87350


namespace tom_dimes_count_l873_87384

/-- The number of dimes Tom initially had -/
def initial_dimes : ℕ := 15

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad : ℕ := 33

/-- The total number of dimes Tom has after receiving dimes from his dad -/
def total_dimes : ℕ := initial_dimes + dimes_from_dad

theorem tom_dimes_count : total_dimes = 48 := by
  sorry

end tom_dimes_count_l873_87384


namespace michael_cleaning_count_l873_87394

/-- The number of times Michael takes a bath per week -/
def baths_per_week : ℕ := 2

/-- The number of times Michael takes a shower per week -/
def showers_per_week : ℕ := 1

/-- The number of weeks in the given time period -/
def weeks : ℕ := 52

/-- The total number of times Michael cleans himself in the given time period -/
def total_cleanings : ℕ := weeks * (baths_per_week + showers_per_week)

theorem michael_cleaning_count : total_cleanings = 156 := by
  sorry

end michael_cleaning_count_l873_87394


namespace fibonacci_square_equality_l873_87321

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_square_equality :
  ∃! n : ℕ, n > 0 ∧ fib n = n^2 ∧ n = 12 := by sorry

end fibonacci_square_equality_l873_87321


namespace original_number_proof_l873_87377

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 →
  y = 30.333333333333332 →
  y > x →
  x + y = 41.6 := by
sorry

end original_number_proof_l873_87377


namespace right_triangle_height_l873_87311

theorem right_triangle_height (a b h : ℝ) (h₁ : a > 0) (h₂ : b > 0) : 
  a = 1 → b = 4 → h^2 = a * b → h = 2 := by sorry

end right_triangle_height_l873_87311


namespace binary_arithmetic_equality_l873_87347

def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => 2 * acc + c.toNat - '0'.toNat) 0

theorem binary_arithmetic_equality : 
  let a := binary_to_nat "1011101"
  let b := binary_to_nat "1101"
  let c := binary_to_nat "101010"
  let d := binary_to_nat "110"
  let result := binary_to_nat "1110111100"
  ((a + b) * c) / d = result := by
  sorry

end binary_arithmetic_equality_l873_87347


namespace necklace_beads_l873_87385

theorem necklace_beads (total : ℕ) (amethyst : ℕ) (amber : ℕ) (turquoise : ℕ) :
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  total = amethyst + amber + turquoise →
  turquoise = 19 := by
sorry

end necklace_beads_l873_87385


namespace river_road_bus_car_ratio_l873_87338

/-- The ratio of buses to cars on River Road -/
def busCarRatio (numBuses : ℕ) (numCars : ℕ) : ℚ :=
  numBuses / numCars

theorem river_road_bus_car_ratio : 
  let numCars : ℕ := 60
  let numBuses : ℕ := numCars - 40
  busCarRatio numBuses numCars = 1 / 3 := by
sorry

end river_road_bus_car_ratio_l873_87338


namespace f_derivative_at_zero_l873_87339

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end f_derivative_at_zero_l873_87339


namespace digit_puzzle_solution_l873_87300

/-- Represents a digit in base 10 -/
def Digit := Fin 10

/-- Checks if all elements in a list are distinct -/
def all_distinct (l : List Digit) : Prop :=
  ∀ i j, i ≠ j → l.get i ≠ l.get j

/-- Converts a pair of digits to a two-digit number -/
def to_number (tens digit : Digit) : Nat :=
  10 * tens.val + digit.val

/-- The main theorem -/
theorem digit_puzzle_solution (Y E M T : Digit) 
  (h_distinct : all_distinct [Y, E, M, T])
  (h_equation : to_number Y E * to_number M E = to_number T T * 101) :
  E.val + M.val + T.val + Y.val = 4 := by
  sorry

end digit_puzzle_solution_l873_87300


namespace computer_price_problem_l873_87387

theorem computer_price_problem (sticker_price : ℝ) : 
  (0.85 * sticker_price - 50 = 0.7 * sticker_price - 20) → 
  sticker_price = 200 := by
sorry

end computer_price_problem_l873_87387


namespace vector_sum_problem_l873_87396

/-- Given two vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_sum_problem (a b : Fin 3 → ℝ) 
  (ha : a = ![1, 2, 3]) 
  (hb : b = ![-1, 0, 1]) : 
  a + 2 • b = ![-1, 2, 5] := by
  sorry

end vector_sum_problem_l873_87396


namespace michelle_gas_usage_l873_87312

theorem michelle_gas_usage (initial_gas final_gas : ℚ) : 
  initial_gas = 1/2 → 
  final_gas = 1/6 → 
  initial_gas - final_gas = 1/3 :=
by sorry

end michelle_gas_usage_l873_87312


namespace product_unit_digit_is_one_l873_87331

def unit_digit (n : ℕ) : ℕ := n % 10

def numbers : List ℕ := [7858413, 10864231, 45823797, 97833129, 51679957, 
                         38213827, 75946153, 27489543, 94837311, 37621597]

theorem product_unit_digit_is_one :
  unit_digit (numbers.prod) = 1 := by
  sorry

#check product_unit_digit_is_one

end product_unit_digit_is_one_l873_87331


namespace amanda_friends_count_l873_87314

def total_tickets : ℕ := 80
def tickets_per_friend : ℕ := 4
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_friends_count :
  ∃ (friends : ℕ), 
    friends * tickets_per_friend + second_day_tickets + third_day_tickets = total_tickets ∧
    friends = 5 := by
  sorry

end amanda_friends_count_l873_87314


namespace sum_of_cubes_l873_87379

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 := by
  sorry

end sum_of_cubes_l873_87379


namespace angle_range_in_triangle_l873_87301

open Real

theorem angle_range_in_triangle (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) :
  π / 2 < A ∧ A < 3 * π / 4 := by
  sorry

end angle_range_in_triangle_l873_87301


namespace point_coordinates_l873_87360

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The distance between a point and the x-axis. -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- The distance between a point and the y-axis. -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: A point P in the fourth quadrant with distance 2 to the x-axis
    and distance 3 to the y-axis has coordinates (3, -2). -/
theorem point_coordinates (P : Point)
    (h1 : FourthQuadrant P)
    (h2 : DistanceToXAxis P = 2)
    (h3 : DistanceToYAxis P = 3) :
    P = Point.mk 3 (-2) := by
  sorry

end point_coordinates_l873_87360


namespace initial_alcohol_percentage_l873_87359

/-- Initial percentage of alcohol in the solution -/
def initial_percentage : ℝ := 5

/-- Initial volume of the solution in liters -/
def initial_volume : ℝ := 40

/-- Volume of alcohol added in liters -/
def added_alcohol : ℝ := 3.5

/-- Volume of water added in liters -/
def added_water : ℝ := 6.5

/-- Final percentage of alcohol in the solution -/
def final_percentage : ℝ := 11

theorem initial_alcohol_percentage :
  initial_percentage = 5 :=
by
  have h1 : initial_volume + added_alcohol + added_water = 50 := by sorry
  have h2 : (initial_percentage / 100) * initial_volume + added_alcohol =
            (final_percentage / 100) * (initial_volume + added_alcohol + added_water) := by sorry
  sorry

end initial_alcohol_percentage_l873_87359


namespace cuboid_volume_l873_87323

/-- A cuboid with integer edge lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℕ :=
  c.length * c.width * c.height

/-- Theorem: The volume of a cuboid with edges 6 cm, 5 cm, and 6 cm is 180 cubic centimeters -/
theorem cuboid_volume : volume { length := 6, width := 5, height := 6 } = 180 := by
  sorry

end cuboid_volume_l873_87323


namespace paint_containers_left_over_l873_87374

/-- Calculates the number of paint containers left over after repainting a bathroom --/
theorem paint_containers_left_over 
  (initial_containers : ℕ) 
  (total_walls : ℕ) 
  (tiled_walls : ℕ) 
  (ceiling_containers : ℕ) 
  (gradient_containers_per_wall : ℕ) : 
  initial_containers = 16 →
  total_walls = 4 →
  tiled_walls = 1 →
  ceiling_containers = 1 →
  gradient_containers_per_wall = 1 →
  initial_containers - 
    (ceiling_containers + 
     (total_walls - tiled_walls) * (1 + gradient_containers_per_wall)) = 11 := by
  sorry


end paint_containers_left_over_l873_87374


namespace perimeter_F₂MN_is_8_l873_87320

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the foci F₁ and F₂
variable (F₁ F₂ : ℝ × ℝ)

-- Define points M and N on the ellipse
variable (M N : ℝ × ℝ)

-- Axiom: F₁ and F₂ are foci of the ellipse C
axiom foci_of_C : ∀ (x y : ℝ), C x y → ∃ (a : ℝ), dist (x, y) F₁ + dist (x, y) F₂ = 2 * a

-- Axiom: M and N are on the ellipse C
axiom M_on_C : C M.1 M.2
axiom N_on_C : C N.1 N.2

-- Axiom: M, N, and F₁ are collinear
axiom collinear_MNF₁ : ∃ (t : ℝ), N = F₁ + t • (M - F₁)

-- Theorem: The perimeter of triangle F₂MN is 8
theorem perimeter_F₂MN_is_8 : dist M N + dist M F₂ + dist N F₂ = 8 := by sorry

end perimeter_F₂MN_is_8_l873_87320


namespace basketball_team_chances_l873_87308

/-- The starting percentage for making the basketball team for a 66-inch tall player -/
def starting_percentage : ℝ := 10

/-- The increase in percentage chance per inch above 66 inches -/
def increase_per_inch : ℝ := 10

/-- The height of the player with known chances -/
def known_height : ℝ := 68

/-- The chances of making the team for the player with known height -/
def known_chances : ℝ := 30

/-- The baseline height for the starting percentage -/
def baseline_height : ℝ := 66

theorem basketball_team_chances :
  starting_percentage =
    known_chances - (increase_per_inch * (known_height - baseline_height)) :=
by sorry

end basketball_team_chances_l873_87308


namespace product_of_decimals_l873_87371

theorem product_of_decimals : 0.3 * 0.7 = 0.21 := by
  sorry

end product_of_decimals_l873_87371


namespace probability_of_red_ball_l873_87316

theorem probability_of_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 5 →
  (red_balls : ℚ) / total_balls = 2 / 7 := by
  sorry

end probability_of_red_ball_l873_87316


namespace triangle_equality_l873_87381

/-- Given a triangle ABC with sides a, b, and c satisfying a^2 + b^2 + c^2 = ab + bc + ac,
    prove that the triangle is equilateral. -/
theorem triangle_equality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
    (eq : a^2 + b^2 + c^2 = a*b + b*c + a*c) : a = b ∧ b = c := by
  sorry

end triangle_equality_l873_87381


namespace magnitude_of_vector_combination_l873_87329

/-- Given two plane vectors a and b with the angle between them π/2 and magnitudes 1,
    prove that the magnitude of 3a - 2b is 1. -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- angle between a and b is π/2
  (a.1^2 + a.2^2 = 1) →  -- |a| = 1
  (b.1^2 + b.2^2 = 1) →  -- |b| = 1
  ((3*a.1 - 2*b.1)^2 + (3*a.2 - 2*b.2)^2 = 1) := by
sorry

end magnitude_of_vector_combination_l873_87329


namespace sqrt_200_simplification_l873_87367

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end sqrt_200_simplification_l873_87367


namespace lcm_of_five_numbers_l873_87332

theorem lcm_of_five_numbers :
  Nat.lcm 456 (Nat.lcm 783 (Nat.lcm 935 (Nat.lcm 1024 1297))) = 2308474368000 := by
  sorry

end lcm_of_five_numbers_l873_87332


namespace total_candies_l873_87348

/-- The number of candies in each gift box -/
def candies_per_box : ℕ := 156

/-- The number of children receiving gift boxes -/
def num_children : ℕ := 20

/-- Theorem: The total number of candies needed is 3120 -/
theorem total_candies : candies_per_box * num_children = 3120 := by
  sorry

end total_candies_l873_87348


namespace product_ab_l873_87319

theorem product_ab (a b : ℝ) (h1 : a - b = 2) (h2 : a^2 + b^2 = 25) : a * b = 21 / 2 := by
  sorry

end product_ab_l873_87319


namespace clock_angle_at_15_40_clock_angle_at_15_40_is_130_l873_87341

/-- The angle between clock hands at 15:40 --/
theorem clock_angle_at_15_40 : ℝ :=
  let minutes_past_hour : ℝ := 40
  let hours_past_12 : ℝ := 3
  let minutes_per_hour : ℝ := 60
  let degrees_per_circle : ℝ := 360
  let hours_per_revolution : ℝ := 12

  let minute_hand_angle : ℝ := (minutes_past_hour / minutes_per_hour) * degrees_per_circle
  let hour_hand_angle : ℝ := (hours_past_12 / hours_per_revolution +
                              minutes_past_hour / (minutes_per_hour * hours_per_revolution)) *
                             degrees_per_circle

  let angle_between : ℝ := |minute_hand_angle - hour_hand_angle|

  130 -- The actual proof is omitted

theorem clock_angle_at_15_40_is_130 : clock_angle_at_15_40 = 130 := by
  sorry -- Proof is omitted

end clock_angle_at_15_40_clock_angle_at_15_40_is_130_l873_87341


namespace percentage_of_green_leaves_l873_87305

/-- Given a collection of leaves with known properties, prove the percentage of green leaves. -/
theorem percentage_of_green_leaves 
  (total_leaves : ℕ) 
  (brown_percentage : ℚ) 
  (yellow_leaves : ℕ) 
  (h1 : total_leaves = 25)
  (h2 : brown_percentage = 1/5)
  (h3 : yellow_leaves = 15) :
  (total_leaves - (brown_percentage * total_leaves).num - yellow_leaves : ℚ) / total_leaves = 1/5 := by
  sorry

end percentage_of_green_leaves_l873_87305


namespace inequality_solution_l873_87399

theorem inequality_solution (x : ℝ) : 
  (x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 0 ↔ x ≤ 0 ∧ x ≠ 1 ∧ x ≠ 2 :=
by sorry

end inequality_solution_l873_87399


namespace grid_product_problem_l873_87368

theorem grid_product_problem (x y : ℚ) 
  (h1 : x * 3 = y) 
  (h2 : 7 * y = 350) : 
  x = 50 / 3 := by
  sorry

end grid_product_problem_l873_87368


namespace prob_sum_le_4_l873_87344

/-- The number of possible outcomes for a single die. -/
def die_outcomes : ℕ := 6

/-- The set of all possible outcomes when throwing two dice. -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range die_outcomes) (Finset.range die_outcomes)

/-- The set of favorable outcomes where the sum is less than or equal to 4. -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 ≤ 4)

/-- The probability of the sum of two dice being less than or equal to 4. -/
theorem prob_sum_le_4 :
    (favorable_outcomes.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry

end prob_sum_le_4_l873_87344


namespace arithmetic_evaluation_l873_87327

theorem arithmetic_evaluation : 2 * 7 + 9 * 4 - 6 * 5 + 8 * 3 = 44 := by
  sorry

end arithmetic_evaluation_l873_87327


namespace simplify_expression_l873_87330

theorem simplify_expression (a b : ℝ) : a + b - (a - b) = 2 * b := by
  sorry

end simplify_expression_l873_87330


namespace student_arrangement_probabilities_l873_87362

/-- Represents the probability of various arrangements of 4 students in a row. -/
structure StudentArrangementProbabilities where
  /-- The total number of possible arrangements for 4 students. -/
  total_arrangements : ℕ
  /-- The number of arrangements where a specific student is at one end. -/
  student_at_end : ℕ
  /-- The number of arrangements where two specific students are at both ends. -/
  two_students_at_ends : ℕ

/-- Theorem stating the probabilities of various student arrangements. -/
theorem student_arrangement_probabilities 
  (probs : StudentArrangementProbabilities)
  (h1 : probs.total_arrangements = 24)
  (h2 : probs.student_at_end = 12)
  (h3 : probs.two_students_at_ends = 4) :
  let p1 := probs.student_at_end / probs.total_arrangements
  let p2 := probs.two_students_at_ends / probs.total_arrangements
  let p3 := 1 - (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  let p4 := (probs.total_arrangements - probs.student_at_end - probs.student_at_end + probs.two_students_at_ends) / probs.total_arrangements
  (p1 = 1/2) ∧ 
  (p2 = 1/6) ∧ 
  (p3 = 5/6) ∧ 
  (p4 = 1/6) := by
  sorry

end student_arrangement_probabilities_l873_87362


namespace roots_of_equation_l873_87392

theorem roots_of_equation : 
  {x : ℝ | x * (x + 5)^3 * (5 - x) = 0} = {-5, 0, 5} := by
sorry

end roots_of_equation_l873_87392


namespace function_equation_implies_zero_l873_87397

/-- A function satisfying f(x + |y|) = f(|x|) + f(y) for all real x and y is identically zero. -/
theorem function_equation_implies_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x + |y|) = f (|x|) + f y) : 
    ∀ x : ℝ, f x = 0 := by
  sorry

end function_equation_implies_zero_l873_87397


namespace perpendicular_lines_m_value_l873_87358

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0,
    this function returns true if they are perpendicular. -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The slope-intercept form of l₁: (m-2)x + 3y + 2m = 0 -/
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m - 2) * x + 3 * y + 2 * m = 0

/-- The slope-intercept form of l₂: x + my + 6 = 0 -/
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  x + m * y + 6 = 0

theorem perpendicular_lines_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, are_perpendicular (m - 2) 3 1 m) → m = 1/2 := by
  sorry

end perpendicular_lines_m_value_l873_87358


namespace minimum_games_for_winning_percentage_l873_87386

theorem minimum_games_for_winning_percentage (N : ℕ) : 
  (∀ k : ℕ, k < N → (3 + k : ℚ) / (4 + k) < 4/5) ∧ 
  (3 + N : ℚ) / (4 + N) ≥ 4/5 → 
  N = 1 :=
by sorry

end minimum_games_for_winning_percentage_l873_87386


namespace milk_drinking_problem_l873_87328

theorem milk_drinking_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (max_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  max_fraction = 1/3 →
  max_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 :=
by
  sorry

end milk_drinking_problem_l873_87328


namespace max_sections_five_l873_87317

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end max_sections_five_l873_87317


namespace crayons_per_box_l873_87375

theorem crayons_per_box (total_crayons : ℕ) (num_boxes : ℕ) (h1 : total_crayons = 35) (h2 : num_boxes = 7) :
  total_crayons / num_boxes = 5 := by
sorry

end crayons_per_box_l873_87375


namespace subtracted_value_l873_87395

theorem subtracted_value (N : ℝ) (x : ℝ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 14) / 10 = 4) : 
  x = 5 := by
  sorry

end subtracted_value_l873_87395


namespace equation_solutions_l873_87334

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (2 * (x₁ - 3) = 3 * x₁ * (x₁ - 3) ∧ x₁ = 3) ∧ 
                (2 * (x₂ - 3) = 3 * x₂ * (x₂ - 3) ∧ x₂ = 2/3)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ + 1 = 0 ∧ y₁ = 1) ∧ 
                (2 * y₂^2 - 3 * y₂ + 1 = 0 ∧ y₂ = 1/2)) := by
  sorry


end equation_solutions_l873_87334


namespace expand_expression_l873_87380

theorem expand_expression (x : ℝ) : (3*x^2 + 2*x - 4)*(x - 3) = 3*x^3 - 7*x^2 - 10*x + 12 := by
  sorry

end expand_expression_l873_87380


namespace symmetry_plane_arrangement_l873_87343

/-- A symmetry plane of a body. -/
structure SymmetryPlane where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A body with symmetry planes. -/
structure Body where
  symmetry_planes : List SymmetryPlane
  exactly_three_planes : symmetry_planes.length = 3

/-- Angle between two symmetry planes. -/
def angle_between (p1 p2 : SymmetryPlane) : ℝ :=
  sorry

/-- Predicate to check if two planes are perpendicular. -/
def are_perpendicular (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 90

/-- Predicate to check if two planes intersect at 60 degrees. -/
def intersect_at_60 (p1 p2 : SymmetryPlane) : Prop :=
  angle_between p1 p2 = 60

/-- Theorem stating the possible arrangements of symmetry planes. -/
theorem symmetry_plane_arrangement (b : Body) :
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    are_perpendicular p1 p2) ∨
  (∀ (p1 p2 : SymmetryPlane), p1 ∈ b.symmetry_planes → p2 ∈ b.symmetry_planes → p1 ≠ p2 →
    intersect_at_60 p1 p2) :=
  sorry

end symmetry_plane_arrangement_l873_87343


namespace candy_problem_l873_87365

theorem candy_problem (initial_candy : ℕ) (talitha_took : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 349)
  (h2 : talitha_took = 108)
  (h3 : remaining_candy = 88) :
  initial_candy - talitha_took - remaining_candy = 153 :=
by
  sorry

end candy_problem_l873_87365


namespace square_plot_poles_l873_87322

/-- The number of fence poles needed for a square plot -/
def total_poles (poles_per_side : ℕ) : ℕ :=
  poles_per_side * 4 - 4

/-- Theorem: For a square plot with 27 fence poles on each side, 
    the total number of poles needed is 104 -/
theorem square_plot_poles : total_poles 27 = 104 := by
  sorry

end square_plot_poles_l873_87322


namespace distance_point_to_line_polar_l873_87313

/-- The distance from a point in polar coordinates to a line in polar form -/
theorem distance_point_to_line_polar (ρ_A : ℝ) (θ_A : ℝ) (k : ℝ) :
  let l : ℝ × ℝ → Prop := λ (ρ, θ) ↦ 2 * ρ * Real.sin (θ - π/4) = Real.sqrt 2
  let A : ℝ × ℝ := (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)
  let d := abs (A.1 - A.2 + 1) / Real.sqrt 2
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 → d = 5 * Real.sqrt 2 / 2 :=
by sorry


end distance_point_to_line_polar_l873_87313


namespace euler_dedekind_divisibility_l873_87369

-- Define the Euler totient function
def Φ : ℕ → ℕ := sorry

-- Define the Dedekind's totient function
def Ψ : ℕ → ℕ := sorry

-- Define the set of numbers of the form 2^n₁, 2^n₁3^n₂, or 2^n₁5^n₂
def S : Set ℕ :=
  {n : ℕ | n = 1 ∨ (∃ n₁ n₂ : ℕ, n = 2^n₁ ∨ n = 2^n₁ * 3^n₂ ∨ n = 2^n₁ * 5^n₂)}

-- State the theorem
theorem euler_dedekind_divisibility (n : ℕ) :
  (n ∈ S) ↔ (Φ n ∣ n + Ψ n) := by sorry

end euler_dedekind_divisibility_l873_87369


namespace dog_eaten_cost_calculation_l873_87352

def cake_cost (flour_cost sugar_cost butter_cost eggs_cost : ℚ) : ℚ :=
  flour_cost + sugar_cost + butter_cost + eggs_cost

def dog_eaten_cost (total_cost : ℚ) (total_slices mother_eaten_slices : ℕ) : ℚ :=
  (total_cost * (total_slices - mother_eaten_slices : ℚ)) / total_slices

theorem dog_eaten_cost_calculation :
  let flour_cost : ℚ := 4
  let sugar_cost : ℚ := 2
  let butter_cost : ℚ := 2.5
  let eggs_cost : ℚ := 0.5
  let total_slices : ℕ := 6
  let mother_eaten_slices : ℕ := 2
  let total_cost := cake_cost flour_cost sugar_cost butter_cost eggs_cost
  dog_eaten_cost total_cost total_slices mother_eaten_slices = 6 := by
  sorry

#eval dog_eaten_cost (cake_cost 4 2 2.5 0.5) 6 2

end dog_eaten_cost_calculation_l873_87352


namespace max_working_groups_l873_87335

theorem max_working_groups (total_instructors : ℕ) (group_size : ℕ) (max_membership : ℕ) :
  total_instructors = 36 →
  group_size = 4 →
  max_membership = 2 →
  (∃ (n : ℕ), n ≤ 18 ∧ 
    n * group_size ≤ total_instructors * max_membership ∧
    ∀ (m : ℕ), m > n → m * group_size > total_instructors * max_membership) :=
by sorry

end max_working_groups_l873_87335


namespace total_distance_walked_and_run_l873_87363

/-- Calculates the total distance traveled when walking and running at different speeds for different durations. -/
theorem total_distance_walked_and_run 
  (walk_time : ℝ) (walk_speed : ℝ) (run_time : ℝ) (run_speed : ℝ) :
  walk_time = 60 →  -- 60 minutes walking
  walk_speed = 3 →  -- 3 mph walking speed
  run_time = 45 →   -- 45 minutes running
  run_speed = 8 →   -- 8 mph running speed
  (walk_time + run_time) / 60 = 1.75 →  -- Total time in hours
  walk_time / 60 * walk_speed + run_time / 60 * run_speed = 9 := by
  sorry

#check total_distance_walked_and_run

end total_distance_walked_and_run_l873_87363


namespace square_difference_l873_87333

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 := by
  sorry

end square_difference_l873_87333


namespace tan_half_angle_special_point_l873_87318

/-- 
If the terminal side of angle α passes through the point (-1, 2),
then tan(α/2) = (1 + √5) / 2.
-/
theorem tan_half_angle_special_point (α : Real) :
  (Real.cos α = -1 / Real.sqrt 5 ∧ Real.sin α = 2 / Real.sqrt 5) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end tan_half_angle_special_point_l873_87318


namespace units_digit_of_23_power_23_l873_87346

theorem units_digit_of_23_power_23 : (23^23) % 10 = 7 := by
  sorry

end units_digit_of_23_power_23_l873_87346


namespace max_intersections_three_circles_one_line_l873_87354

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between a line and three circles -/
def max_line_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and one line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_line_circle_intersections = 12 := by
  sorry

end max_intersections_three_circles_one_line_l873_87354


namespace average_speed_two_segments_l873_87390

/-- Given a 100-mile trip where the first 50 miles are traveled at 20 mph
    and the remaining 50 miles at 50 mph, prove that the average speed
    for the entire trip is 100 / (50/20 + 50/50) miles per hour. -/
theorem average_speed_two_segments (total_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ)
  (first_speed : ℝ) (second_speed : ℝ)
  (h1 : total_distance = 100)
  (h2 : first_segment = 50)
  (h3 : second_segment = 50)
  (h4 : first_speed = 20)
  (h5 : second_speed = 50)
  (h6 : total_distance = first_segment + second_segment) :
  (total_distance / (first_segment / first_speed + second_segment / second_speed)) =
  100 / (50 / 20 + 50 / 50) :=
by sorry

end average_speed_two_segments_l873_87390


namespace mistake_correction_l873_87357

theorem mistake_correction (x : ℝ) : 8 * x + 8 = 56 → x / 8 = 0.75 := by
  sorry

end mistake_correction_l873_87357


namespace smallest_multiple_l873_87349

theorem smallest_multiple (n : ℕ) : n = 1628 ↔ 
  (∃ k : ℕ, n = 37 * k) ∧ 
  (∃ m : ℕ, n - 3 = 101 * m) ∧ 
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 37 * k) ∧ (∃ m : ℕ, x - 3 = 101 * m))) :=
by sorry

end smallest_multiple_l873_87349


namespace minimum_g_5_l873_87304

def Tenuous (f : ℕ → ℤ) : Prop :=
  ∀ x y : ℕ, x > 0 ∧ y > 0 → f x + f y > y^2

def SumOfG (g : ℕ → ℤ) : ℤ :=
  (List.range 10).map (λ i => g (i + 1)) |>.sum

theorem minimum_g_5 (g : ℕ → ℤ) (h_tenuous : Tenuous g) 
    (h_min : ∀ g' : ℕ → ℤ, Tenuous g' → SumOfG g ≤ SumOfG g') : 
  g 5 ≥ 49 := by
  sorry

end minimum_g_5_l873_87304


namespace farey_sequence_properties_l873_87324

/-- Farey sequence of order n -/
def farey_sequence (n : ℕ) : List (ℚ) := sorry

/-- Sum of numerators in a Farey sequence -/
def sum_numerators (seq : List ℚ) : ℚ := sorry

/-- Sum of denominators in a Farey sequence -/
def sum_denominators (seq : List ℚ) : ℚ := sorry

/-- Sum of fractions in a Farey sequence -/
def sum_fractions (seq : List ℚ) : ℚ := sorry

theorem farey_sequence_properties (n : ℕ) :
  let seq := farey_sequence n
  (sum_denominators seq = 2 * sum_numerators seq) ∧
  (sum_fractions seq = (seq.length : ℚ) / 2) := by sorry

end farey_sequence_properties_l873_87324


namespace wooden_block_surface_area_l873_87302

theorem wooden_block_surface_area (A₁ A₂ A₃ A₄ A₅ A₆ A₇ : ℕ) 
  (h₁ : A₁ = 148)
  (h₂ : A₂ = 46)
  (h₃ : A₃ = 72)
  (h₄ : A₄ = 28)
  (h₅ : A₅ = 88)
  (h₆ : A₆ = 126)
  (h₇ : A₇ = 58) :
  ∃ A₈ : ℕ, A₈ = 22 ∧ A₁ + A₂ + A₃ + A₄ - (A₅ + A₆ + A₇) = A₈ :=
by sorry

end wooden_block_surface_area_l873_87302


namespace frank_apples_l873_87340

theorem frank_apples (frank : ℕ) (susan : ℕ) : 
  susan = 3 * frank →  -- Susan picked 3 times as many apples as Frank
  (2 * frank / 3 + 3 * susan / 2 : ℚ) = 78 →  -- Remaining apples after Frank sold 1/3 and Susan gave out 1/2
  frank = 36 := by
sorry

end frank_apples_l873_87340


namespace new_speed_calculation_l873_87315

theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) 
  (h1 : distance = 469)
  (h2 : original_time = 6)
  (h3 : original_time > 0) :
  let new_time := original_time * (3/2)
  let new_speed := distance / new_time
  new_speed = distance / (original_time * (3/2)) := by
sorry

end new_speed_calculation_l873_87315


namespace derivative_at_one_equals_one_l873_87325

theorem derivative_at_one_equals_one 
  (f : ℝ → ℝ) 
  (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^3 - (deriv f 1) * x^2 + 1) : 
  deriv f 1 = 1 := by
  sorry

end derivative_at_one_equals_one_l873_87325


namespace square_side_length_l873_87337

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
  sorry

end square_side_length_l873_87337


namespace exists_N_average_twelve_l873_87345

theorem exists_N_average_twelve : ∃ N : ℝ, 11 < N ∧ N < 21 ∧ (8 + 12 + N) / 3 = 12 := by
  sorry

end exists_N_average_twelve_l873_87345


namespace expression_equality_l873_87382

theorem expression_equality (x y z : ℝ) 
  (h1 : x * y = 6)
  (h2 : x - z = 2)
  (h3 : x + y + z = 9) :
  x / y - z / x - z^2 / (x * y) = 2 := by
  sorry

end expression_equality_l873_87382


namespace complex_magnitude_problem_l873_87336

def i : ℂ := Complex.I

theorem complex_magnitude_problem (z : ℂ) (h : z * (i + 1) = i) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_problem_l873_87336


namespace sufficient_not_necessary_condition_l873_87326

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  ¬(∀ a b, a > b → a > b + 1) :=
sorry

end sufficient_not_necessary_condition_l873_87326


namespace product_is_very_large_l873_87342

theorem product_is_very_large : 
  (3 + 2) * 
  (3^2 + 2^2) * 
  (3^4 + 2^4) * 
  (3^8 + 2^8) * 
  (3^16 + 2^16) * 
  (3^32 + 2^32) * 
  (3^64 + 2^64) > 10^400 := by
sorry

end product_is_very_large_l873_87342


namespace complement_of_46_35_l873_87383

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60, minutes := totalMinutes % 60 }

/-- The theorem stating that the complement of 46°35' is 43°25' -/
theorem complement_of_46_35 :
  complement { degrees := 46, minutes := 35 } = { degrees := 43, minutes := 25 } := by
  sorry

end complement_of_46_35_l873_87383


namespace abc_equation_solutions_l873_87355

/-- Given integers a, b, c ≥ 2, prove that a b c - 1 = (a - 1)(b - 1)(c - 1) 
    if and only if (a, b, c) is a permutation of (2, 2, 2), (2, 2, 4), (2, 4, 8), or (3, 5, 15) -/
theorem abc_equation_solutions (a b c : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) ↔ 
  List.Perm [a, b, c] [2, 2, 2] ∨ 
  List.Perm [a, b, c] [2, 2, 4] ∨ 
  List.Perm [a, b, c] [2, 4, 8] ∨ 
  List.Perm [a, b, c] [3, 5, 15] :=
by sorry


end abc_equation_solutions_l873_87355


namespace three_intersection_points_l873_87391

/-- The number of distinct points satisfying the given equations -/
def num_intersection_points : ℕ := 3

/-- First equation -/
def equation1 (x y : ℝ) : Prop :=
  (x + y - 7) * (2*x - 3*y + 7) = 0

/-- Second equation -/
def equation2 (x y : ℝ) : Prop :=
  (x - y + 3) * (3*x + 2*y - 18) = 0

/-- Theorem stating that there are exactly 3 distinct points satisfying both equations -/
theorem three_intersection_points :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = num_intersection_points ∧
    ∀ p ∈ points, equation1 p.1 p.2 ∧ equation2 p.1 p.2 :=
sorry

end three_intersection_points_l873_87391


namespace complex_fraction_equality_l873_87372

theorem complex_fraction_equality : ∃ (i : ℂ), i^2 = -1 ∧ i / (1 + Real.sqrt 3 * i) = (Real.sqrt 3 / 4 : ℂ) + (1 / 4 : ℂ) * i := by
  sorry

end complex_fraction_equality_l873_87372


namespace divisible_by_six_sum_powers_divisible_by_seven_l873_87356

-- Part (a)
theorem divisible_by_six (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) := by
  sorry

-- Part (b)
theorem sum_powers_divisible_by_seven :
  7 ∣ (1^2015 + 2^2015 + 3^2015 + 4^2015 + 5^2015 + 6^2015) := by
  sorry

end divisible_by_six_sum_powers_divisible_by_seven_l873_87356


namespace s_range_l873_87389

theorem s_range (a b c : ℝ) 
  (ha : 1/2 ≤ a ∧ a ≤ 1) 
  (hb : 1/2 ≤ b ∧ b ≤ 1) 
  (hc : 1/2 ≤ c ∧ c ≤ 1) : 
  let s := (a + b) / (1 + c) + (b + c) / (1 + a) + (c + a) / (1 + b)
  2 ≤ s ∧ s ≤ 3 := by
sorry

end s_range_l873_87389


namespace paint_cost_decrease_l873_87351

theorem paint_cost_decrease (canvas_original : ℝ) (paint_original : ℝ) 
  (h1 : paint_original = 4 * canvas_original)
  (h2 : canvas_original > 0)
  (h3 : paint_original > 0) :
  let canvas_new := 0.6 * canvas_original
  let total_original := paint_original + canvas_original
  let total_new := 0.4400000000000001 * total_original
  ∃ (paint_new : ℝ), paint_new = 0.4 * paint_original ∧ total_new = paint_new + canvas_new :=
by sorry

end paint_cost_decrease_l873_87351


namespace initial_water_temp_l873_87370

-- Define the constants
def total_time : ℕ := 73
def temp_increase_per_minute : ℕ := 3
def boiling_point : ℕ := 212
def pasta_cooking_time : ℕ := 12

-- Define the theorem
theorem initial_water_temp (mixing_time : ℕ) (boiling_time : ℕ) 
  (h1 : mixing_time = pasta_cooking_time / 3)
  (h2 : boiling_time = total_time - (pasta_cooking_time + mixing_time))
  (h3 : boiling_point = temp_increase_per_minute * boiling_time + 41) :
  41 = boiling_point - temp_increase_per_minute * boiling_time :=
by sorry

end initial_water_temp_l873_87370


namespace max_value_x_2y_plus_1_l873_87376

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ z, |x - 2*y + 1| ≤ z ↔ M ≤ z) :=
sorry

end max_value_x_2y_plus_1_l873_87376


namespace problem_solution_l873_87366

theorem problem_solution (x : ℕ+) : 
  x^2 + 4*x + 29 = x*(4*x + 9) + 13 → x = 2 := by
sorry

end problem_solution_l873_87366


namespace division_multiplication_result_l873_87364

theorem division_multiplication_result : 
  let x : ℝ := 6.5
  let y : ℝ := (x / 6) * 12
  y = 13 := by sorry

end division_multiplication_result_l873_87364
