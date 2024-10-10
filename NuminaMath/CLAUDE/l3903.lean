import Mathlib

namespace shaded_area_theorem_l3903_390394

/-- The area of the region covered by two identical squares overlapping to form a regular octagon
    but not covered by a circle, given the circle's radius and π value. -/
theorem shaded_area_theorem (R : ℝ) (π : ℝ) (h1 : R = 60) (h2 : π = 3.14) :
  let total_square_area := 2 * R * R
  let circle_area := π * R * R
  total_square_area - circle_area = 3096 := by sorry

end shaded_area_theorem_l3903_390394


namespace rectangle_area_l3903_390313

theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 200 → l * w = 1600 := by sorry

end rectangle_area_l3903_390313


namespace diamond_eight_five_l3903_390324

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := (a + b) * ((a - b)^2)

-- Theorem statement
theorem diamond_eight_five : diamond 8 5 = 117 := by
  sorry

end diamond_eight_five_l3903_390324


namespace arithmetic_calculation_l3903_390338

theorem arithmetic_calculation : 10 * (1/8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 := by
  sorry

end arithmetic_calculation_l3903_390338


namespace complex_equation_solution_l3903_390389

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = 10 → z = 1 - 3*I := by
  sorry

end complex_equation_solution_l3903_390389


namespace not_proportional_l3903_390372

-- Define the notion of direct proportionality
def is_directly_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, y t = k * x t

-- Define the notion of inverse proportionality
def is_inversely_proportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

-- Define our equation
def our_equation (x y : ℝ) : Prop :=
  2 * x + 3 * y = 6

-- Theorem statement
theorem not_proportional :
  ¬ (∃ x y : ℝ → ℝ, (∀ t : ℝ, our_equation (x t) (y t)) ∧
    (is_directly_proportional x y ∨ is_inversely_proportional x y)) :=
sorry

end not_proportional_l3903_390372


namespace product_of_digits_for_non_divisible_by_five_l3903_390312

def numbers : List Nat := [4750, 4760, 4775, 4785, 4790]

def is_divisible_by_five (n : Nat) : Bool :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_for_non_divisible_by_five :
  ∃ n ∈ numbers, ¬is_divisible_by_five n ∧ 
    units_digit n * tens_digit n = 0 :=
by sorry

end product_of_digits_for_non_divisible_by_five_l3903_390312


namespace complex_repairs_is_two_l3903_390360

/-- Represents Jim's bike shop operations for a month --/
structure BikeShop where
  tire_repair_price : ℕ
  tire_repair_cost : ℕ
  tire_repairs_count : ℕ
  complex_repair_price : ℕ
  complex_repair_cost : ℕ
  retail_profit : ℕ
  fixed_expenses : ℕ
  total_profit : ℕ

/-- Calculates the number of complex repairs given the shop's operations --/
def complex_repairs_count (shop : BikeShop) : ℕ :=
  sorry

/-- Theorem stating that the number of complex repairs is 2 --/
theorem complex_repairs_is_two (shop : BikeShop) 
  (h1 : shop.tire_repair_price = 20)
  (h2 : shop.tire_repair_cost = 5)
  (h3 : shop.tire_repairs_count = 300)
  (h4 : shop.complex_repair_price = 300)
  (h5 : shop.complex_repair_cost = 50)
  (h6 : shop.retail_profit = 2000)
  (h7 : shop.fixed_expenses = 4000)
  (h8 : shop.total_profit = 3000) :
  complex_repairs_count shop = 2 := by
  sorry

end complex_repairs_is_two_l3903_390360


namespace two_friend_visits_count_l3903_390319

/-- Represents a friend with a visitation period -/
structure Friend where
  period : ℕ

/-- Calculates the number of days in a given period where exactly two out of three friends visit -/
def countTwoFriendVisits (f1 f2 f3 : Friend) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 27 days in a 365-day period 
    where two out of three friends visit, given their visitation periods -/
theorem two_friend_visits_count : 
  let max : Friend := { period := 5 }
  let nora : Friend := { period := 6 }
  let olivia : Friend := { period := 7 }
  countTwoFriendVisits max nora olivia 365 = 27 := by
  sorry

end two_friend_visits_count_l3903_390319


namespace sequence_and_inequality_problem_l3903_390359

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def positive_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_and_inequality_problem
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_pos : positive_sequence b)
  (h_a1 : a 1 = 2)
  (h_b1 : b 1 = 3)
  (h_sum1 : a 3 + b 5 = 56)
  (h_sum2 : a 5 + b 3 = 26)
  (h_ineq : ∀ n : ℕ, n > 0 → ∀ x : ℝ, -x^2 + 3*x ≤ (2 * b n) / (2 * ↑n + 1)) :
  (∀ n : ℕ, a n = 3 * ↑n - 1) ∧
  (∀ n : ℕ, b n = 3 * 2^(n-1)) ∧
  (∀ x : ℝ, (-x^2 + 3*x ≤ 2) ↔ (x ≥ 2 ∨ x ≤ 1)) :=
sorry

end sequence_and_inequality_problem_l3903_390359


namespace correct_allocation_plans_l3903_390335

/-- Represents the number of factories --/
def num_factories : Nat := 4

/-- Represents the number of classes --/
def num_classes : Nat := 3

/-- Represents the requirement that at least one factory must have a class --/
def must_have_class : Nat := 1

/-- The number of different allocation plans --/
def allocation_plans : Nat := 57

/-- Theorem stating that the number of allocation plans is correct --/
theorem correct_allocation_plans :
  (num_factories = 4) →
  (num_classes = 3) →
  (must_have_class = 1) →
  (allocation_plans = 57) := by
  sorry

end correct_allocation_plans_l3903_390335


namespace elephant_count_theorem_l3903_390378

/-- The total number of elephants in two parks, given the number in one park
    and a multiplier for the other park. -/
def total_elephants (park1_count : ℕ) (multiplier : ℕ) : ℕ :=
  park1_count + multiplier * park1_count

/-- Theorem stating that the total number of elephants in two parks is 280,
    given that one park has 70 elephants and the other has 3 times as many. -/
theorem elephant_count_theorem :
  total_elephants 70 3 = 280 := by
  sorry

end elephant_count_theorem_l3903_390378


namespace correct_inequalities_l3903_390382

/-- 
Given a student's estimated scores in Chinese and Mathematics after a mock final exam,
this theorem proves that the correct system of inequalities representing the situation is
x > 85 and y ≥ 80, where x is the Chinese score and y is the Mathematics score.
-/
theorem correct_inequalities (x y : ℝ) 
  (h1 : x > 85)  -- Chinese score is higher than 85 points
  (h2 : y ≥ 80)  -- Mathematics score is not less than 80 points
  : x > 85 ∧ y ≥ 80 := by
  sorry

end correct_inequalities_l3903_390382


namespace cubic_equation_roots_l3903_390311

/-- The equation x³ - x - 2/(3√3) = 0 has exactly three real roots: 2/√3, -1/√3, and -1/√3 -/
theorem cubic_equation_roots :
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, x^3 - x - 2/(3*Real.sqrt 3) = 0 ∧ 
  (2/Real.sqrt 3 ∈ s ∧ -1/Real.sqrt 3 ∈ s) :=
sorry

end cubic_equation_roots_l3903_390311


namespace even_odd_sum_difference_l3903_390340

-- Define the sum of the first n even numbers
def sumEven (n : ℕ) : ℕ := n * (n + 1)

-- Define the sum of the first n odd numbers
def sumOdd (n : ℕ) : ℕ := n^2

-- State the theorem
theorem even_odd_sum_difference : sumEven 100 - sumOdd 100 = 100 := by
  sorry

end even_odd_sum_difference_l3903_390340


namespace intersection_M_N_l3903_390343

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end intersection_M_N_l3903_390343


namespace airport_gate_probability_l3903_390307

/-- The number of gates in the airport -/
def num_gates : ℕ := 16

/-- The distance between adjacent gates in feet -/
def distance_between_gates : ℕ := 75

/-- The maximum distance Dina is willing to walk in feet -/
def max_walking_distance : ℕ := 300

/-- The probability of walking 300 feet or less to the new gate -/
def probability_short_walk : ℚ := 8/15

theorem airport_gate_probability :
  let total_possibilities := num_gates * (num_gates - 1)
  let gates_within_distance := 2 * (max_walking_distance / distance_between_gates)
  let favorable_outcomes := num_gates * gates_within_distance
  (favorable_outcomes : ℚ) / total_possibilities = probability_short_walk :=
sorry

end airport_gate_probability_l3903_390307


namespace x_is_integer_l3903_390369

theorem x_is_integer (x : ℝ) 
  (h1 : ∃ n : ℤ, x^1960 - x^1919 = n)
  (h2 : ∃ m : ℤ, x^2001 - x^1960 = m)
  (h3 : ∃ k : ℤ, x^2001 - x^1919 = k) : 
  ∃ z : ℤ, x = z := by
sorry

end x_is_integer_l3903_390369


namespace min_additional_squares_for_symmetry_l3903_390334

-- Define a point in the grid
structure Point where
  x : Nat
  y : Nat

-- Define the grid size
def gridSize : Nat := 6

-- Define the initially shaded squares
def initialShaded : List Point := [
  { x := 1, y := 1 },
  { x := 1, y := 6 },
  { x := 6, y := 1 },
  { x := 3, y := 4 }
]

-- Function to check if a point is within the grid
def inGrid (p : Point) : Bool :=
  1 ≤ p.x ∧ p.x ≤ gridSize ∧ 1 ≤ p.y ∧ p.y ≤ gridSize

-- Function to check if a set of points has both horizontal and vertical symmetry
def hasSymmetry (points : List Point) : Bool :=
  sorry

-- The main theorem
theorem min_additional_squares_for_symmetry :
  ∃ (additionalPoints : List Point),
    additionalPoints.length = 4 ∧
    (∀ p ∈ additionalPoints, inGrid p) ∧
    hasSymmetry (initialShaded ++ additionalPoints) ∧
    (∀ (otherPoints : List Point),
      otherPoints.length < 4 →
      ¬ hasSymmetry (initialShaded ++ otherPoints)) :=
  sorry

end min_additional_squares_for_symmetry_l3903_390334


namespace executive_board_selection_l3903_390322

theorem executive_board_selection (n : ℕ) (r : ℕ) : n = 12 ∧ r = 5 → Nat.choose n r = 792 := by
  sorry

end executive_board_selection_l3903_390322


namespace base9_813_equals_base3_220110_l3903_390320

/-- Converts a base-9 number to base-3 --/
def base9_to_base3 (n : ℕ) : ℕ :=
  sorry

/-- Theorem: 813 in base 9 is equal to 220110 in base 3 --/
theorem base9_813_equals_base3_220110 : base9_to_base3 813 = 220110 := by
  sorry

end base9_813_equals_base3_220110_l3903_390320


namespace divisibility_of_square_sum_minus_2017_l3903_390326

theorem divisibility_of_square_sum_minus_2017 (n : ℕ) : 
  ∃ x y : ℤ, (n : ℤ) ∣ (x^2 + y^2 - 2017) := by
sorry

end divisibility_of_square_sum_minus_2017_l3903_390326


namespace joy_reading_rate_l3903_390371

/-- Represents Joy's reading rate in pages per hour -/
def reading_rate (pages_per_20min : ℕ) (pages_per_5hours : ℕ) : ℚ :=
  (pages_per_20min * 3)

/-- Theorem stating Joy's reading rate is 24 pages per hour -/
theorem joy_reading_rate :
  reading_rate 8 120 = 24 := by sorry

end joy_reading_rate_l3903_390371


namespace river_current_speed_proof_l3903_390366

def river_current_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  let current_speed := 4
  current_speed

theorem river_current_speed_proof (boat_speed distance total_time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : distance = 60)
  (h3 : total_time = 6.25) :
  river_current_speed boat_speed distance total_time = 4 := by
  sorry

#check river_current_speed_proof

end river_current_speed_proof_l3903_390366


namespace sum_and_difference_bounds_l3903_390309

theorem sum_and_difference_bounds (a b : ℝ) 
  (ha : 60 ≤ a ∧ a ≤ 84) (hb : 28 ≤ b ∧ b ≤ 33) : 
  (88 ≤ a + b ∧ a + b ≤ 117) ∧ (27 ≤ a - b ∧ a - b ≤ 56) := by
  sorry

end sum_and_difference_bounds_l3903_390309


namespace circle_construction_l3903_390376

/-- Given four lines intersecting at a point with 45° angles between them, and a circle
    intersecting these lines such that two opposite chords have lengths a and b,
    and one chord is three times the length of its opposite chord,
    the circle's center (u, v) and radius r satisfy specific equations. -/
theorem circle_construction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (u v r : ℝ),
    u^2 = (a^2 - b^2) / 8 + Real.sqrt (((a^2 - b^2) / 8)^2 + ((a^2 + b^2) / 10)^2) ∧
    v^2 = r^2 - a^2 / 4 ∧
    r^2 = (u^2 + v^2) / 2 + (a^2 + b^2) / 8 :=
by sorry

end circle_construction_l3903_390376


namespace opposite_of_negative_2023_l3903_390379

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by sorry

end opposite_of_negative_2023_l3903_390379


namespace face_vertex_assignment_l3903_390361

-- Define a planar bipartite graph
class PlanarBipartiteGraph (G : Type) where
  -- Add necessary properties for planar bipartite graphs
  is_planar : Bool
  is_bipartite : Bool

-- Define faces and vertices of a graph
def faces (G : Type) [PlanarBipartiteGraph G] : Set G := sorry
def vertices (G : Type) [PlanarBipartiteGraph G] : Set G := sorry

-- Theorem statement
theorem face_vertex_assignment {G : Type} [PlanarBipartiteGraph G] :
  ∃ f : faces G → vertices G, Function.Injective f :=
sorry

end face_vertex_assignment_l3903_390361


namespace smaller_number_problem_l3903_390352

theorem smaller_number_problem (x y : ℤ) : 
  x + y = 64 → y = x + 12 → x = 26 := by
  sorry

end smaller_number_problem_l3903_390352


namespace point_C_values_l3903_390336

-- Define the points on the number line
def point_A : ℝ := 2
def point_B : ℝ := -4

-- Define the property of equal distances between adjacent points
def equal_distances (c : ℝ) : Prop :=
  abs (point_A - point_B) = abs (point_B - c) ∧ 
  abs (point_A - point_B) = abs (point_A - c)

-- Theorem statement
theorem point_C_values : 
  ∀ c : ℝ, equal_distances c → (c = -10 ∨ c = 8) :=
by sorry

end point_C_values_l3903_390336


namespace digits_at_1100_to_1102_l3903_390318

/-- Represents a list of integers starting with 2 in increasing order -/
def listStartingWith2 : List ℕ := sorry

/-- Returns the nth digit in the concatenated string of all numbers in the list -/
def nthDigit (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1100th, 1101st, and 1102nd digits are 2, 1, and 9 respectively -/
theorem digits_at_1100_to_1102 :
  (nthDigit 1100 = 2) ∧ (nthDigit 1101 = 1) ∧ (nthDigit 1102 = 9) := by sorry

end digits_at_1100_to_1102_l3903_390318


namespace rationalize_denominator_sqrt3_minus2_l3903_390373

theorem rationalize_denominator_sqrt3_minus2 :
  1 / (Real.sqrt 3 - 2) = -Real.sqrt 3 - 2 := by
  sorry

end rationalize_denominator_sqrt3_minus2_l3903_390373


namespace correct_equation_l3903_390329

theorem correct_equation (a b : ℝ) : 5 * a^2 * b - 6 * a^2 * b = -a^2 * b := by
  sorry

end correct_equation_l3903_390329


namespace theater_ticket_sales_l3903_390314

/-- Calculates the total amount collected from ticket sales given the number of adults and children, and their respective ticket prices. -/
def totalTicketSales (numAdults numChildren adultPrice childPrice : ℕ) : ℕ :=
  numAdults * adultPrice + numChildren * childPrice

/-- Theorem stating that given the specific conditions of the problem, the total ticket sales amount to $246. -/
theorem theater_ticket_sales :
  let adultPrice : ℕ := 11
  let childPrice : ℕ := 10
  let totalAttendees : ℕ := 23
  let numChildren : ℕ := 7
  let numAdults : ℕ := totalAttendees - numChildren
  totalTicketSales numAdults numChildren adultPrice childPrice = 246 :=
by
  sorry

#check theater_ticket_sales

end theater_ticket_sales_l3903_390314


namespace value_of_expression_l3903_390399

theorem value_of_expression (x : ℝ) (h : 3 * x^2 - 2 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2/3) = 3 := by
  sorry

end value_of_expression_l3903_390399


namespace first_day_distance_l3903_390321

/-- Proves the distance covered on the first day of a three-day hike -/
theorem first_day_distance (total_distance : ℝ) (second_day : ℝ) (third_day : ℝ)
  (h1 : total_distance = 50)
  (h2 : second_day = total_distance / 2)
  (h3 : third_day = 15)
  : total_distance - second_day - third_day = 10 := by
  sorry

end first_day_distance_l3903_390321


namespace prism_volume_l3903_390332

/-- A right rectangular prism with face areas 12, 18, and 24 square inches has a volume of 72 cubic inches -/
theorem prism_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0)
  (area1 : l * w = 12) (area2 : w * h = 18) (area3 : l * h = 24) :
  l * w * h = 72 := by sorry

end prism_volume_l3903_390332


namespace profit_is_12_5_l3903_390301

/-- Calculates the profit per piece given the purchase price, markup percentage, and discount percentage. -/
def calculate_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let initial_price := purchase_price * (1 + markup_percent)
  let final_price := initial_price * (1 - discount_percent)
  final_price - purchase_price

/-- Theorem stating that the profit per piece is 12.5 yuan under the given conditions. -/
theorem profit_is_12_5 :
  calculate_profit 100 0.25 0.1 = 12.5 := by
  sorry

#eval calculate_profit 100 0.25 0.1

end profit_is_12_5_l3903_390301


namespace ellipse_eccentricity_at_min_mn_l3903_390397

/-- Given that 1/m + 2/n = 1 with m > 0 and n > 0, prove that the eccentricity of the ellipse
    x²/m² + y²/n² = 1 is √3/2 when mn takes its minimum value. -/
theorem ellipse_eccentricity_at_min_mn (m n : ℝ) 
  (h1 : m > 0) (h2 : n > 0) (h3 : 1/m + 2/n = 1) : 
  let e := Real.sqrt (1 - (min m n)^2 / (max m n)^2)
  ∃ (x : ℝ), (x = mn) ∧ (∀ y : ℝ, y = m*n → x ≤ y) → e = Real.sqrt 3 / 2 := by
  sorry

end ellipse_eccentricity_at_min_mn_l3903_390397


namespace min_value_expression_l3903_390388

theorem min_value_expression (x y z : ℝ) 
  (hx : -1/2 < x ∧ x < 1/2) 
  (hy : -1/2 < y ∧ y < 1/2) 
  (hz : -1/2 < z ∧ z < 1/2) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + 
  (1 / ((1 + x) * (1 + y) * (1 + z))) + 
  1/2 ≥ 5/2 ∧ 
  (1 / ((1 - 0) * (1 - 0) * (1 - 0))) + 
  (1 / ((1 + 0) * (1 + 0) * (1 + 0))) + 
  1/2 = 5/2 := by
sorry

end min_value_expression_l3903_390388


namespace smallest_perfect_square_sum_l3903_390358

def consecutive_sum (n : ℕ) : ℕ := 10 * (2 * n + 19)

theorem smallest_perfect_square_sum :
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), consecutive_sum m = k^2) ∧
    (∃ (k : ℕ), consecutive_sum n = k^2) ∧
    consecutive_sum n = 1000 :=
sorry

end smallest_perfect_square_sum_l3903_390358


namespace tournament_result_l3903_390303

/-- Represents a tennis tournament with the given rules --/
structure TennisTournament where
  participants : ℕ
  points_for_win : ℕ
  points_for_loss : ℕ

/-- Calculates the number of participants finishing with a given number of points --/
def participants_with_points (t : TennisTournament) (points : ℕ) : ℕ :=
  Nat.choose (Nat.log 2 t.participants) points

theorem tournament_result (t : TennisTournament) 
  (h1 : t.participants = 512)
  (h2 : t.points_for_win = 1)
  (h3 : t.points_for_loss = 0) :
  participants_with_points t 6 = 84 := by
  sorry

end tournament_result_l3903_390303


namespace boris_candy_problem_l3903_390300

/-- Given the initial candy count, amount eaten by daughter, number of bowls, 
    and final count in one bowl, calculate how many pieces Boris took from each bowl. -/
theorem boris_candy_problem (initial_candy : ℕ) (daughter_ate : ℕ) (num_bowls : ℕ) (final_bowl_count : ℕ)
  (h1 : initial_candy = 100)
  (h2 : daughter_ate = 8)
  (h3 : num_bowls = 4)
  (h4 : final_bowl_count = 20)
  (h5 : num_bowls > 0) :
  let remaining_candy := initial_candy - daughter_ate
  let candy_per_bowl := remaining_candy / num_bowls
  candy_per_bowl - final_bowl_count = 3 := by sorry

end boris_candy_problem_l3903_390300


namespace gasoline_tank_capacity_l3903_390375

theorem gasoline_tank_capacity : 
  ∀ (capacity : ℝ),
  (5/6 : ℝ) * capacity - (1/3 : ℝ) * capacity = 20 →
  capacity = 40 := by
sorry

end gasoline_tank_capacity_l3903_390375


namespace list_price_calculation_l3903_390337

theorem list_price_calculation (list_price : ℝ) : 
  (list_price ≥ 0) →
  (0.15 * (list_price - 15) = 0.25 * (list_price - 25)) →
  list_price = 40 := by
  sorry

end list_price_calculation_l3903_390337


namespace min_value_theorem_l3903_390341

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 4 / b = 1) :
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), 2 / a + 2 * b ≥ x → m ≤ x :=
by sorry

end min_value_theorem_l3903_390341


namespace smallest_x_with_given_remainders_l3903_390362

theorem smallest_x_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 7 = 6 ∧ 
  x % 8 = 7 ∧
  ∀ (y : ℕ), y > 0 → 
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → 
    x ≤ y ∧ 
  x = 167 := by
sorry

end smallest_x_with_given_remainders_l3903_390362


namespace proportional_scaling_l3903_390357

/-- Proportional scaling of a rectangle -/
theorem proportional_scaling (w h new_w : ℝ) (hw : w > 0) (hh : h > 0) (hnew_w : new_w > 0) :
  let scale_factor := new_w / w
  let new_h := h * scale_factor
  w = 3 ∧ h = 2 ∧ new_w = 12 → new_h = 8 := by sorry

end proportional_scaling_l3903_390357


namespace equation_roots_property_l3903_390386

-- Define the equation and its properties
def equation (m : ℤ) (x : ℤ) : Prop := x^2 + (m + 1) * x - 2 = 0

-- Define the roots
def is_root (m α β : ℤ) : Prop :=
  equation m (α + 1) ∧ equation m (β + 1) ∧ α < β ∧ m ≠ 0

-- Define d
def d (α β : ℤ) : ℤ := β - α

-- Theorem statement
theorem equation_roots_property :
  ∀ m α β : ℤ, is_root m α β → m = -2 ∧ d α β = 3 := by sorry

end equation_roots_property_l3903_390386


namespace quadratic_inequality_solution_set_l3903_390345

theorem quadratic_inequality_solution_set
  (a b c α β : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (h3 : α > 0)
  (h4 : a < 0)
  (h5 : α + β = -b / a)
  (h6 : α * β = c / a) :
  ∀ x, c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α :=
by sorry

end quadratic_inequality_solution_set_l3903_390345


namespace quadratic_inequality_range_l3903_390305

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ a ∈ Set.Ioc (-4) 0 := by
  sorry

end quadratic_inequality_range_l3903_390305


namespace midpoint_on_yaxis_product_l3903_390348

/-- Given a function f(x) = a^x where a > 0 and a ≠ 1, if the midpoint of the line segment
    with endpoints (x₁, f(x₁)) and (x₂, f(x₂)) is on the y-axis, then f(x₁) · f(x₂) = 1 -/
theorem midpoint_on_yaxis_product (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) (ha_ne_one : a ≠ 1) 
  (h_midpoint : x₁ + x₂ = 0) : 
  (a^x₁) * (a^x₂) = 1 := by
  sorry

end midpoint_on_yaxis_product_l3903_390348


namespace fifteenth_student_age_l3903_390344

/-- Represents a class of students with given age statistics -/
structure ClassStats where
  total_students : Nat
  class_average : Float
  group1_size : Nat
  group1_average : Float
  group2_size : Nat
  group3_size : Nat
  group3_average : Float
  remaining_boys_average : Float

/-- Theorem stating the age of the 15th student given the class statistics -/
theorem fifteenth_student_age (stats : ClassStats) 
  (h1 : stats.total_students = 15)
  (h2 : stats.class_average = 15.2)
  (h3 : stats.group1_size = 5)
  (h4 : stats.group1_average = 14)
  (h5 : stats.group2_size = 4)
  (h6 : stats.group3_size = 3)
  (h7 : stats.group3_average = 16.6)
  (h8 : stats.remaining_boys_average = 15.4)
  (h9 : stats.group1_size + stats.group2_size + stats.group3_size + 3 = stats.total_students) :
  ∃ (age : Float), age = 15.7 ∧ age = (stats.class_average * stats.total_students.toFloat
                                      - stats.group1_average * stats.group1_size.toFloat
                                      - stats.group3_average * stats.group3_size.toFloat
                                      - stats.remaining_boys_average * 3)
                                      / stats.group2_size.toFloat :=
by sorry


end fifteenth_student_age_l3903_390344


namespace floor_ceiling_sum_l3903_390339

theorem floor_ceiling_sum : ⌊(-3.72 : ℝ)⌋ + ⌈(34.1 : ℝ)⌉ = 31 := by sorry

end floor_ceiling_sum_l3903_390339


namespace arithmetic_sequence_first_term_l3903_390306

/-- An arithmetic sequence {aₙ} satisfying aₙ₊₁ + aₙ = 4n for all n has a₁ = 1 -/
theorem arithmetic_sequence_first_term (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)  -- arithmetic sequence condition
  (h_sum : ∀ n, a (n + 1) + a n = 4 * n)                    -- given condition
  : a 1 = 1 := by
  sorry

end arithmetic_sequence_first_term_l3903_390306


namespace parallelogram_uniqueness_l3903_390351

/-- Represents a parallelogram in 2D space -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a point in 2D space -/
structure Point :=
  (x y : ℝ)

/-- The measure of an angle in radians -/
def Angle := ℝ

/-- The length of a line segment -/
def Length := ℝ

/-- Checks if two parallelograms are congruent -/
def are_congruent (p1 p2 : Parallelogram) : Prop :=
  sorry

/-- Constructs a parallelogram given the required parameters -/
def construct_parallelogram (α ε : Angle) (bd : Length) : Parallelogram :=
  sorry

/-- Theorem stating the uniqueness of the constructed parallelogram -/
theorem parallelogram_uniqueness (α ε : Angle) (bd : Length) :
  ∀ p1 p2 : Parallelogram,
    (p1 = construct_parallelogram α ε bd) →
    (p2 = construct_parallelogram α ε bd) →
    are_congruent p1 p2 :=
  sorry

end parallelogram_uniqueness_l3903_390351


namespace triangle_height_calculation_l3903_390370

theorem triangle_height_calculation (base area height : Real) : 
  base = 8.4 → area = 24.36 → area = (base * height) / 2 → height = 5.8 := by
  sorry

end triangle_height_calculation_l3903_390370


namespace initial_average_weight_l3903_390374

theorem initial_average_weight (a b c d e : ℝ) : 
  -- Initial conditions
  (a + b + c) / 3 = (a + b + c) / 3 →
  -- Adding packet d
  (a + b + c + d) / 4 = 80 →
  -- Replacing a with e
  (b + c + d + e) / 4 = 79 →
  -- Relationship between d and e
  e = d + 3 →
  -- Weight of packet a
  a = 75 →
  -- Conclusion: initial average weight
  (a + b + c) / 3 = 84 := by
sorry


end initial_average_weight_l3903_390374


namespace tank_capacity_l3903_390367

/-- The capacity of a tank given specific inlet and outlet pipe rates --/
theorem tank_capacity 
  (outlet_time : ℝ) 
  (inlet_rate1 : ℝ) 
  (inlet_rate2 : ℝ) 
  (extended_time : ℝ) 
  (h1 : outlet_time = 10) 
  (h2 : inlet_rate1 = 4) 
  (h3 : inlet_rate2 = 6) 
  (h4 : extended_time = 8) : 
  ∃ (capacity : ℝ), 
    capacity = 13500 ∧ 
    capacity / outlet_time - (inlet_rate1 * 60 + inlet_rate2 * 60) = capacity / (outlet_time + extended_time) :=
by sorry

end tank_capacity_l3903_390367


namespace inequality_proof_l3903_390377

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 9 / 4) : 
  a^3 + b^3 + c^3 > a * Real.sqrt (b + c) + b * Real.sqrt (c + a) + c * Real.sqrt (a + b) := by
  sorry

end inequality_proof_l3903_390377


namespace initial_men_count_l3903_390393

/-- Represents the work completion scenario in a garment industry -/
structure WorkScenario where
  men : ℕ
  hours_per_day : ℕ
  days : ℕ

/-- Calculates the total man-hours for a given work scenario -/
def total_man_hours (scenario : WorkScenario) : ℕ :=
  scenario.men * scenario.hours_per_day * scenario.days

/-- The initial work scenario -/
def initial_scenario (initial_men : ℕ) : WorkScenario :=
  { men := initial_men, hours_per_day := 8, days := 10 }

/-- The second work scenario -/
def second_scenario : WorkScenario :=
  { men := 8, hours_per_day := 15, days := 8 }

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : ∃ (initial_men : ℕ), 
  initial_men = 12 ∧ 
  total_man_hours (initial_scenario initial_men) = total_man_hours second_scenario :=
sorry

end initial_men_count_l3903_390393


namespace crayons_left_l3903_390325

/-- Represents the number of crayons Mary has -/
structure Crayons where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons -/
def total_crayons (c : Crayons) : Nat :=
  c.green + c.blue

/-- Represents the number of crayons Mary gives away -/
structure CrayonsGiven where
  green : Nat
  blue : Nat

/-- Calculates the total number of crayons given away -/
def total_given (g : CrayonsGiven) : Nat :=
  g.green + g.blue

/-- Theorem: Mary has 9 crayons left after giving some away -/
theorem crayons_left (initial : Crayons) (given : CrayonsGiven) 
  (h1 : initial.green = 5)
  (h2 : initial.blue = 8)
  (h3 : given.green = 3)
  (h4 : given.blue = 1) :
  total_crayons initial - total_given given = 9 := by
  sorry


end crayons_left_l3903_390325


namespace martha_butterflies_l3903_390383

def butterfly_collection (blue yellow black : ℕ) : Prop :=
  blue = 2 * yellow ∧ black = 5 ∧ blue = 4

theorem martha_butterflies :
  ∀ blue yellow black : ℕ,
  butterfly_collection blue yellow black →
  blue + yellow + black = 11 :=
by
  sorry

end martha_butterflies_l3903_390383


namespace smallest_group_size_exists_group_size_l3903_390333

theorem smallest_group_size (n : ℕ) : 
  (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) → n ≥ 169 :=
by sorry

theorem exists_group_size : 
  ∃ n : ℕ, (n % 6 = 1) ∧ (n % 9 = 3) ∧ (n % 8 = 5) ∧ n = 169 :=
by sorry

end smallest_group_size_exists_group_size_l3903_390333


namespace quadratic_inequality_range_l3903_390308

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a ≤ 1 := by
  sorry

end quadratic_inequality_range_l3903_390308


namespace matching_probability_is_four_fifteenths_l3903_390349

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeanDistribution where
  blue : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeanDistribution.total (d : JellyBeanDistribution) : ℕ :=
  d.blue + d.red + d.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeanDistribution :=
  { blue := 1, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeanDistribution :=
  { blue := 2, red := 1, yellow := 2 }

/-- Calculates the probability of matching colors -/
def probability_matching_colors (d1 d2 : JellyBeanDistribution) : ℚ :=
  (d1.blue * d2.blue + d1.red * d2.red : ℚ) / ((d1.total * d2.total) : ℚ)

theorem matching_probability_is_four_fifteenths :
  probability_matching_colors abe bob = 4 / 15 := by
  sorry

end matching_probability_is_four_fifteenths_l3903_390349


namespace cubic_expression_value_l3903_390330

theorem cubic_expression_value (r s : ℝ) : 
  3 * r^2 - 4 * r - 7 = 0 →
  3 * s^2 - 4 * s - 7 = 0 →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 := by
  sorry

end cubic_expression_value_l3903_390330


namespace simplify_expression_l3903_390347

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 0) :
  a - b + (2 * b^2) / (a + b) = (a^2 + b^2) / (a + b) := by
  sorry

end simplify_expression_l3903_390347


namespace min_value_product_l3903_390365

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x + 3 * y) * (y + 3 * z) * (2 * x * z + 1) ≥ 24 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    (x₀ + 3 * y₀) * (y₀ + 3 * z₀) * (2 * x₀ * z₀ + 1) = 24 * Real.sqrt 2 :=
sorry

end min_value_product_l3903_390365


namespace parakeets_fed_sixty_cups_l3903_390304

/-- The number of parakeets fed with a given amount of bird seed -/
def parakeets_fed (cups : ℕ) : ℕ :=
  sorry

theorem parakeets_fed_sixty_cups :
  parakeets_fed 60 = 20 :=
by
  sorry

/-- Assumption: 30 cups of bird seed feed 10 parakeets for 5 days -/
axiom feed_ratio : parakeets_fed 30 = 10

/-- The number of parakeets fed is directly proportional to the amount of bird seed -/
axiom linear_feed : ∀ (c₁ c₂ : ℕ), c₁ ≠ 0 → c₂ ≠ 0 →
  (parakeets_fed c₁ : ℚ) / c₁ = (parakeets_fed c₂ : ℚ) / c₂

end parakeets_fed_sixty_cups_l3903_390304


namespace gcd_of_45_75_90_l3903_390317

theorem gcd_of_45_75_90 : Nat.gcd 45 (Nat.gcd 75 90) = 15 := by
  sorry

end gcd_of_45_75_90_l3903_390317


namespace jennifer_remaining_money_l3903_390398

def initial_amount : ℚ := 90

def sandwich_fraction : ℚ := 1/5
def museum_fraction : ℚ := 1/6
def book_fraction : ℚ := 1/2

def remaining_amount : ℚ := initial_amount - (sandwich_fraction * initial_amount + museum_fraction * initial_amount + book_fraction * initial_amount)

theorem jennifer_remaining_money :
  remaining_amount = 12 := by sorry

end jennifer_remaining_money_l3903_390398


namespace quadratic_function_range_l3903_390354

theorem quadratic_function_range (a b : ℝ) : 
  (∀ x ∈ Set.Ioo 2 5, a * x^2 + b * x + 2 > 0) →
  (a * 1^2 + b * 1 + 2 = 1) →
  a > 3 - 2 * Real.sqrt 2 :=
sorry

end quadratic_function_range_l3903_390354


namespace machinery_cost_l3903_390310

def total_amount : ℝ := 7428.57
def raw_materials : ℝ := 5000
def cash_percentage : ℝ := 0.30

theorem machinery_cost :
  ∃ (machinery : ℝ),
    machinery = total_amount - raw_materials - (cash_percentage * total_amount) ∧
    machinery = 200 := by
  sorry

end machinery_cost_l3903_390310


namespace f_is_quadratic_l3903_390368

/-- Definition of a quadratic equation with one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 4x - x² -/
def f (x : ℝ) : ℝ := 4 * x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l3903_390368


namespace solve_equation_l3903_390391

theorem solve_equation : ∃ x : ℝ, 0.035 * x = 42 ∧ x = 1200 := by
  sorry

end solve_equation_l3903_390391


namespace negation_of_existence_l3903_390315

variable (a : ℝ)

theorem negation_of_existence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end negation_of_existence_l3903_390315


namespace points_per_bag_l3903_390331

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : 
  total_bags = 11 → unrecycled_bags = 2 → total_points = 45 → 
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end points_per_bag_l3903_390331


namespace system_solution_l3903_390381

theorem system_solution (a b : ℝ) : 
  (∃ x y : ℝ, a * x - y = 4 ∧ 3 * x + b * y = 4 ∧ x = 2 ∧ y = -2) →
  a + b = 2 := by
sorry

end system_solution_l3903_390381


namespace birthday_celebration_attendees_l3903_390350

theorem birthday_celebration_attendees :
  ∀ (n : ℕ), 
  (12 * (n + 2) = 16 * n) → 
  n = 6 := by
sorry

end birthday_celebration_attendees_l3903_390350


namespace zero_not_in_range_of_g_l3903_390387

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(2 : ℝ) / (x + 3)⌉ 
  else if x < -3 then ⌊(2 : ℝ) / (x + 3)⌋ 
  else 0  -- This value doesn't matter as g is not defined at x = -3

-- Theorem statement
theorem zero_not_in_range_of_g : ¬ ∃ (x : ℝ), g x = 0 := by
  sorry

end zero_not_in_range_of_g_l3903_390387


namespace walking_meeting_point_l3903_390323

/-- Represents the meeting of two people walking towards each other --/
theorem walking_meeting_point (total_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) 
  (deceleration_a : ℝ) (acceleration_b : ℝ) (h : ℕ) :
  total_distance = 100 ∧ 
  speed_a = 5 ∧ 
  speed_b = 4 ∧ 
  deceleration_a = 0.4 ∧ 
  acceleration_b = 0.5 →
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 + 
  (h : ℝ) * (2 * speed_b + (h - 1) * acceleration_b) / 2 = total_distance ∧ 
  (h : ℝ) * (2 * speed_a - (h - 1) * deceleration_a) / 2 = 
  total_distance / 2 - 31 := by
  sorry

#check walking_meeting_point

end walking_meeting_point_l3903_390323


namespace identity_function_unique_l3903_390384

def PositiveNat := {n : ℕ // n > 0}

theorem identity_function_unique 
  (f : PositiveNat → PositiveNat) 
  (h : ∀ (m n : PositiveNat), ∃ (k : ℕ), k * (m.val^2 + (f n).val) = m.val * (f m).val + n.val) : 
  ∀ (n : PositiveNat), f n = n :=
sorry

end identity_function_unique_l3903_390384


namespace factorization_equality_l3903_390346

theorem factorization_equality (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factorization_equality_l3903_390346


namespace triangle_projection_types_l3903_390353

-- Define the possible projection types
inductive ProjectionType
  | Angle
  | Strip
  | TwoAnglesJoined
  | Triangle
  | AngleWithInfiniteFigure

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to determine if a point is on a plane
def isPointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Function to determine if three points are collinear
def areCollinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), p3.x - p1.x = t * (p2.x - p1.x) ∧
              p3.y - p1.y = t * (p2.y - p1.y) ∧
              p3.z - p1.z = t * (p2.z - p1.z)

-- Define the projection function
def project (triangle : Triangle3D) (O : Point3D) (P : Plane3D) : ProjectionType :=
  sorry -- Actual implementation would go here

-- The main theorem
theorem triangle_projection_types 
  (triangle : Triangle3D) 
  (O : Point3D) 
  (P : Plane3D) 
  (h1 : ¬ isPointOnPlane O (Plane3D.mk 0 0 0 0)) -- O is not in the plane of the triangle
  (h2 : ¬ areCollinear triangle.A triangle.B triangle.C) -- ABC is a valid triangle
  : ∃ (projType : ProjectionType), project triangle O P = projType ∧ 
    (projType = ProjectionType.Angle ∨ 
     projType = ProjectionType.Strip ∨ 
     projType = ProjectionType.TwoAnglesJoined ∨ 
     projType = ProjectionType.Triangle ∨ 
     projType = ProjectionType.AngleWithInfiniteFigure) :=
  sorry


end triangle_projection_types_l3903_390353


namespace modulo_residue_problem_l3903_390392

theorem modulo_residue_problem :
  (312 + 6 * 51 + 8 * 187 + 5 * 34) % 17 = 2 := by
  sorry

end modulo_residue_problem_l3903_390392


namespace smallest_constant_inequality_l3903_390395

theorem smallest_constant_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b / (a + b + 2 * c) + b * c / (b + c + 2 * a) + c * a / (c + a + 2 * b) ≤ (1/4) * (a + b + c)) ∧
  ∀ k : ℝ, k > 0 → k < 1/4 →
    ∃ a' b' c' : ℝ, a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
      a' * b' / (a' + b' + 2 * c') + b' * c' / (b' + c' + 2 * a') + c' * a' / (c' + a' + 2 * b') > k * (a' + b' + c') :=
by sorry

end smallest_constant_inequality_l3903_390395


namespace convention_handshakes_l3903_390302

-- Define the number of companies and representatives per company
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4

-- Define the total number of people
def total_people : ℕ := num_companies * reps_per_company

-- Define the number of handshakes per person
def handshakes_per_person : ℕ := total_people - reps_per_company

-- Theorem statement
theorem convention_handshakes : 
  (total_people * handshakes_per_person) / 2 = 160 := by
  sorry


end convention_handshakes_l3903_390302


namespace group_size_before_new_member_l3903_390390

theorem group_size_before_new_member 
  (avg_after : ℚ) 
  (new_member_amount : ℚ) 
  (avg_before : ℚ) 
  (h1 : avg_after = 20)
  (h2 : new_member_amount = 56)
  (h3 : avg_before = 14) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * avg_before + new_member_amount = (n + 1 : ℚ) * avg_after ∧
    n = 6 :=
by sorry

end group_size_before_new_member_l3903_390390


namespace intersection_S_T_l3903_390380

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0}

def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by
  sorry

end intersection_S_T_l3903_390380


namespace candy_difference_is_twenty_l3903_390342

/-- The number of candies Bryan has compared to Ben -/
def candy_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  λ bryan_skittles bryan_gummy bryan_lollipops ben_mm ben_jelly ben_lollipops =>
    (bryan_skittles + bryan_gummy + bryan_lollipops) - (ben_mm + ben_jelly + ben_lollipops)

theorem candy_difference_is_twenty :
  candy_difference 50 30 15 20 45 10 = 20 := by
  sorry

end candy_difference_is_twenty_l3903_390342


namespace mix_buyer_probability_l3903_390328

theorem mix_buyer_probability (total : ℕ) (cake muffin cookie : ℕ) 
  (cake_muffin cake_cookie muffin_cookie : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_cake : cake = 70)
  (h_muffin : muffin = 60)
  (h_cookie : cookie = 50)
  (h_cake_muffin : cake_muffin = 25)
  (h_cake_cookie : cake_cookie = 15)
  (h_muffin_cookie : muffin_cookie = 10)
  (h_all_three : all_three = 5) : 
  (total - (cake + muffin + cookie - cake_muffin - cake_cookie - muffin_cookie + all_three)) / total = 1 / 10 := by
  sorry

end mix_buyer_probability_l3903_390328


namespace ten_machines_four_minutes_l3903_390396

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute := (420 * machines) / 6
  bottles_per_minute * minutes

/-- Theorem stating that 10 machines produce 2800 bottles in 4 minutes -/
theorem ten_machines_four_minutes :
  bottles_produced 10 4 = 2800 := by
  sorry

#eval bottles_produced 10 4

end ten_machines_four_minutes_l3903_390396


namespace probability_of_specific_combination_l3903_390355

def shirts : ℕ := 6
def shorts : ℕ := 7
def socks : ℕ := 8
def hats : ℕ := 3
def total_items : ℕ := shirts + shorts + socks + hats
def items_chosen : ℕ := 4

theorem probability_of_specific_combination :
  (shirts.choose 1 * shorts.choose 1 * socks.choose 1 * hats.choose 1) / total_items.choose items_chosen = 144 / 1815 :=
sorry

end probability_of_specific_combination_l3903_390355


namespace smallest_prime_after_six_nonprimes_l3903_390364

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if there are six consecutive nonprime numbers before a given number -/
def hasSixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 97 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 97 ∧ hasSixConsecutiveNonprimes 97 ∧
  ∀ m : ℕ, m < 97 → ¬(isPrime m ∧ hasSixConsecutiveNonprimes m) :=
sorry

end smallest_prime_after_six_nonprimes_l3903_390364


namespace sector_central_angle_l3903_390363

/-- Given a sector with arc length 2π cm and radius 2 cm, its central angle is π radians. -/
theorem sector_central_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2 * Real.pi) (h2 : radius = 2) :
  arc_length / radius = Real.pi := by
  sorry

end sector_central_angle_l3903_390363


namespace least_valid_number_l3903_390327

def is_valid (n : ℕ) : Prop :=
  n % 11 = 0 ∧
  n % 2 = 1 ∧
  n % 3 = 1 ∧
  n % 4 = 1 ∧
  n % 5 = 1 ∧
  n % 7 = 1

theorem least_valid_number : ∀ m : ℕ, m < 2521 → ¬(is_valid m) ∧ is_valid 2521 :=
sorry

end least_valid_number_l3903_390327


namespace smallest_n_l3903_390356

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_n : 
  let n : ℕ := 9075
  ∀ m : ℕ, m > 0 → 
    (is_factor (5^2) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (3^3) (m * (2^5) * (6^2) * (7^3) * (13^4)) ∧
     is_factor (11^2) (m * (2^5) * (6^2) * (7^3) * (13^4))) →
    m ≥ n := by
  sorry

end smallest_n_l3903_390356


namespace rugby_team_average_weight_l3903_390316

theorem rugby_team_average_weight 
  (initial_players : ℕ) 
  (new_player_weight : ℝ) 
  (new_average_weight : ℝ) : 
  initial_players = 20 ∧ 
  new_player_weight = 210 ∧ 
  new_average_weight = 181.42857142857142 → 
  (initial_players * (new_average_weight * (initial_players + 1) - new_player_weight)) / initial_players = 180 := by
sorry

end rugby_team_average_weight_l3903_390316


namespace no_good_number_l3903_390385

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem no_good_number :
  ¬ ∃ n : ℕ, 
    is_divisible_by_sum_of_digits n ∧
    is_divisible_by_sum_of_digits (n + 1) ∧
    is_divisible_by_sum_of_digits (n + 2) ∧
    is_divisible_by_sum_of_digits (n + 3) :=
sorry

end no_good_number_l3903_390385
