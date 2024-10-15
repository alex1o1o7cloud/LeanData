import Mathlib

namespace NUMINAMATH_CALUDE_waysToSum1800_eq_45651_l3460_346005

/-- The number of ways to write 1800 as the sum of ones, twos, and threes, ignoring order -/
def waysToSum1800 : ℕ := sorry

/-- The target number we're considering -/
def targetNumber : ℕ := 1800

theorem waysToSum1800_eq_45651 : waysToSum1800 = 45651 := by sorry

end NUMINAMATH_CALUDE_waysToSum1800_eq_45651_l3460_346005


namespace NUMINAMATH_CALUDE_unique_solution_for_m_squared_minus_eight_equals_three_to_n_l3460_346033

theorem unique_solution_for_m_squared_minus_eight_equals_three_to_n :
  ∀ m n : ℕ, m^2 - 8 = 3^n ↔ m = 3 ∧ n = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_m_squared_minus_eight_equals_three_to_n_l3460_346033


namespace NUMINAMATH_CALUDE_train_speed_l3460_346023

/-- Given a train of length 180 meters that crosses a stationary point in 6 seconds,
    prove that its speed is 30 meters per second. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 180) (h2 : time = 6) :
  length / time = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3460_346023


namespace NUMINAMATH_CALUDE_angle_measure_l3460_346021

-- Define a type for angles
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := a1 = a2

-- Define complementary angle
def complementary_angle (a : Angle) : Angle :=
  ⟨90 - a.degrees, 60 - a.minutes⟩

-- Theorem statement
theorem angle_measure :
  ∀ (angle1 angle2 : Angle),
  vertical_angles angle1 angle2 →
  complementary_angle angle1 = ⟨79, 32⟩ →
  angle2 = ⟨100, 28⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3460_346021


namespace NUMINAMATH_CALUDE_inequality_solution_l3460_346063

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^3 - 1) > 0 ↔ x < -3 ∨ (-3 < x ∧ x < 1) ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3460_346063


namespace NUMINAMATH_CALUDE_paper_length_calculation_l3460_346013

/-- Calculates the length of paper wrapped around a tube -/
theorem paper_length_calculation 
  (paper_width : ℝ) 
  (initial_diameter : ℝ) 
  (final_diameter : ℝ) 
  (num_layers : ℕ) 
  (h1 : paper_width = 4)
  (h2 : initial_diameter = 4)
  (h3 : final_diameter = 16)
  (h4 : num_layers = 500) :
  (π * num_layers * (initial_diameter + final_diameter) / 2) / 100 = 50 * π := by
sorry

end NUMINAMATH_CALUDE_paper_length_calculation_l3460_346013


namespace NUMINAMATH_CALUDE_jinho_remaining_money_l3460_346075

theorem jinho_remaining_money (initial_amount : ℕ) (eraser_cost pencil_cost : ℕ) 
  (eraser_count pencil_count : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 2500 →
  eraser_cost = 120 →
  pencil_cost = 350 →
  eraser_count = 5 →
  pencil_count = 3 →
  remaining_amount = initial_amount - (eraser_cost * eraser_count + pencil_cost * pencil_count) →
  remaining_amount = 850 := by
sorry

end NUMINAMATH_CALUDE_jinho_remaining_money_l3460_346075


namespace NUMINAMATH_CALUDE_negation_equivalence_l3460_346099

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3460_346099


namespace NUMINAMATH_CALUDE_insect_legs_l3460_346041

theorem insect_legs (num_insects : ℕ) (total_legs : ℕ) (h1 : num_insects = 8) (h2 : total_legs = 48) :
  total_legs / num_insects = 6 := by
  sorry

end NUMINAMATH_CALUDE_insect_legs_l3460_346041


namespace NUMINAMATH_CALUDE_sphere_radius_equals_seven_l3460_346010

/-- Given a sphere and a right circular cylinder where:
    1. The surface area of the sphere equals the curved surface area of the cylinder
    2. The height of the cylinder is 14 cm
    3. The diameter of the cylinder is 14 cm
    This theorem proves that the radius of the sphere is 7 cm. -/
theorem sphere_radius_equals_seven (r : ℝ) :
  (4 * Real.pi * r^2 = 2 * Real.pi * 7 * 14) →
  r = 7 := by
  sorry

#check sphere_radius_equals_seven

end NUMINAMATH_CALUDE_sphere_radius_equals_seven_l3460_346010


namespace NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l3460_346019

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probability_divisible_by_10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1 ∧
  ∀ t : Segment, probability_divisible_by_10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probability_divisible_by_10 s = 1/19 ∧
  ∀ t : Segment, probability_divisible_by_10 t = 0 ∨ probability_divisible_by_10 t ≥ 1/19 :=
sorry

end NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l3460_346019


namespace NUMINAMATH_CALUDE_safe_journey_exists_l3460_346073

-- Define the duration of the journey
def road_duration : ℕ := 4
def trail_duration : ℕ := 4

-- Define the eruption patterns
def crater1_cycle : ℕ := 18
def crater2_cycle : ℕ := 10

-- Define the safety condition
def is_safe (t : ℕ) : Prop :=
  (t % crater1_cycle ≠ 0) ∧ 
  ((t % crater2_cycle ≠ 0) → (t < road_duration ∨ t ≥ road_duration + trail_duration))

-- Theorem statement
theorem safe_journey_exists :
  ∃ start : ℕ, 
    (∀ t : ℕ, t ≥ start ∧ t < start + 2 * (road_duration + trail_duration) → is_safe t) :=
sorry

end NUMINAMATH_CALUDE_safe_journey_exists_l3460_346073


namespace NUMINAMATH_CALUDE_equal_distribution_probability_l3460_346070

/-- Represents a player in the game -/
inductive Player : Type
| Alice : Player
| Bob : Player
| Charlie : Player
| Dana : Player

/-- The state of the game is represented by the money each player has -/
def GameState := Player → ℕ

/-- The initial state of the game where each player has 1 dollar -/
def initialState : GameState := fun _ => 1

/-- A single turn of the game where a player gives 1 dollar to another randomly chosen player -/
def turn (state : GameState) : GameState := sorry

/-- The probability that after 40 turns, each player has 1 dollar -/
def probabilityEqualDistribution (n : ℕ) : ℝ :=
  sorry

/-- The main theorem stating that the probability of equal distribution after 40 turns is 1/9 -/
theorem equal_distribution_probability :
  probabilityEqualDistribution 40 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_probability_l3460_346070


namespace NUMINAMATH_CALUDE_water_servings_difference_l3460_346060

/-- Proves the difference in servings for Simeon's water consumption --/
theorem water_servings_difference (total_water : ℕ) (old_serving : ℕ) (new_serving : ℕ)
  (h1 : total_water = 64)
  (h2 : old_serving = 8)
  (h3 : new_serving = 16)
  (h4 : old_serving > 0)
  (h5 : new_serving > 0) :
  (total_water / old_serving) - (total_water / new_serving) = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_servings_difference_l3460_346060


namespace NUMINAMATH_CALUDE_brandons_cash_sales_l3460_346090

theorem brandons_cash_sales (total_sales : ℝ) (credit_sales_fraction : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_sales_fraction = 2/5 →
  cash_sales = total_sales * (1 - credit_sales_fraction) →
  cash_sales = 48 := by
sorry

end NUMINAMATH_CALUDE_brandons_cash_sales_l3460_346090


namespace NUMINAMATH_CALUDE_interest_rate_is_10_percent_l3460_346089

/-- Calculates the simple interest rate given principal, amount, and time. -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: Given the conditions, the interest rate is 10%. -/
theorem interest_rate_is_10_percent :
  let principal : ℚ := 750
  let amount : ℚ := 900
  let time : ℕ := 2
  calculate_interest_rate principal amount time = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_10_percent_l3460_346089


namespace NUMINAMATH_CALUDE_sam_remaining_seashells_l3460_346093

def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

theorem sam_remaining_seashells : 
  initial_seashells - seashells_given_away = 17 := by sorry

end NUMINAMATH_CALUDE_sam_remaining_seashells_l3460_346093


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l3460_346026

/-- Calculates the total cost of stationery given the number of boxes of pencils,
    pencils per box, cost per pencil, and cost per pen. -/
def total_stationery_cost (boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) (pen_cost : ℕ) : ℕ :=
  let total_pencils := boxes * pencils_per_box
  let total_pens := 2 * total_pencils + 300
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery under the given conditions is $18,300. -/
theorem stationery_cost_theorem :
  total_stationery_cost 15 80 4 5 = 18300 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l3460_346026


namespace NUMINAMATH_CALUDE_bowling_team_size_l3460_346034

/-- The number of original players in a bowling team -/
def original_players : ℕ := 7

/-- The original average weight of the team in kg -/
def original_avg : ℚ := 94

/-- The weight of the first new player in kg -/
def new_player1 : ℚ := 110

/-- The weight of the second new player in kg -/
def new_player2 : ℚ := 60

/-- The new average weight of the team after adding two players, in kg -/
def new_avg : ℚ := 92

theorem bowling_team_size :
  (original_avg * original_players + new_player1 + new_player2) / (original_players + 2) = new_avg :=
sorry

end NUMINAMATH_CALUDE_bowling_team_size_l3460_346034


namespace NUMINAMATH_CALUDE_min_phi_for_even_sine_l3460_346095

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem min_phi_for_even_sine (ω φ : ℝ) (h_omega : ω ≠ 0) (h_phi : φ > 0) 
  (h_even : IsEven (fun x ↦ 2 * Real.sin (ω * x + φ))) :
  ∃ (k : ℤ), φ = k * Real.pi + Real.pi / 2 ∧ 
  ∀ (m : ℤ), (m * Real.pi + Real.pi / 2 > 0) → (k * Real.pi + Real.pi / 2 ≤ m * Real.pi + Real.pi / 2) :=
sorry

end NUMINAMATH_CALUDE_min_phi_for_even_sine_l3460_346095


namespace NUMINAMATH_CALUDE_union_A_B_l3460_346058

/-- Set A is defined as the set of real numbers between -2 and 3 inclusive -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- Set B is defined as the set of positive real numbers -/
def B : Set ℝ := {x | x > 0}

/-- The union of sets A and B is equal to the set of real numbers greater than or equal to -2 -/
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_union_A_B_l3460_346058


namespace NUMINAMATH_CALUDE_number_of_students_in_section_B_l3460_346044

/-- Given a class with two sections A and B, prove the number of students in section B -/
theorem number_of_students_in_section_B 
  (students_A : ℕ) 
  (avg_weight_A : ℚ) 
  (avg_weight_B : ℚ) 
  (avg_weight_total : ℚ) 
  (h1 : students_A = 50)
  (h2 : avg_weight_A = 50)
  (h3 : avg_weight_B = 70)
  (h4 : avg_weight_total = 61.67) :
  ∃ (students_B : ℕ), students_B = 70 := by
sorry

end NUMINAMATH_CALUDE_number_of_students_in_section_B_l3460_346044


namespace NUMINAMATH_CALUDE_total_combinations_l3460_346022

/-- The number of students -/
def num_students : ℕ := 20

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The minimum number of members in each group -/
def min_members_per_group : ℕ := 3

/-- The number of topics -/
def num_topics : ℕ := 5

/-- The number of ways to divide students into groups -/
def group_formations : ℕ := 165

/-- The number of ways to assign topics to groups -/
def topic_assignments : ℕ := 120

theorem total_combinations : 
  (group_formations * topic_assignments = 19800) ∧ 
  (num_students ≥ num_groups * min_members_per_group) ∧
  (num_topics > num_groups) :=
sorry

end NUMINAMATH_CALUDE_total_combinations_l3460_346022


namespace NUMINAMATH_CALUDE_coin_diameter_is_14_l3460_346080

/-- The diameter of a coin given its radius -/
def coin_diameter (radius : ℝ) : ℝ := 2 * radius

/-- Theorem: The diameter of a coin with radius 7 cm is 14 cm -/
theorem coin_diameter_is_14 : coin_diameter 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_coin_diameter_is_14_l3460_346080


namespace NUMINAMATH_CALUDE_same_speed_is_two_l3460_346064

-- Define Jack's speed function
def jack_speed (x : ℝ) : ℝ := x^2 - 7*x - 18

-- Define Jill's distance function
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72

-- Define Jill's time function
def jill_time (x : ℝ) : ℝ := x + 8

-- Theorem statement
theorem same_speed_is_two :
  ∀ x : ℝ, 
  x ≠ -8 →  -- Ensure division by zero is avoided
  (jill_distance x) / (jill_time x) = jack_speed x →
  jack_speed x = 2 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_is_two_l3460_346064


namespace NUMINAMATH_CALUDE_problem_solution_l3460_346025

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt x else |Real.sin x|

theorem problem_solution (a : ℝ) :
  f a = (1/2) → (a = (1/4) ∨ a = -π/6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3460_346025


namespace NUMINAMATH_CALUDE_supermarket_eggs_l3460_346078

/-- Represents the number of egg cartons in the supermarket -/
def num_cartons : ℕ := 28

/-- Represents the length of the egg array in each carton -/
def carton_length : ℕ := 33

/-- Represents the width of the egg array in each carton -/
def carton_width : ℕ := 4

/-- Calculates the total number of eggs in the supermarket -/
def total_eggs : ℕ := num_cartons * carton_length * carton_width

/-- Theorem stating that the total number of eggs in the supermarket is 3696 -/
theorem supermarket_eggs : total_eggs = 3696 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_eggs_l3460_346078


namespace NUMINAMATH_CALUDE_parabola_intersection_l3460_346028

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ := {x | 2*x^2 + 3*x - 4 = x^2 + 2*x + 1}

/-- The y-coordinate of the intersection points -/
def intersection_y : ℝ := 4.5

/-- The first parabola -/
def parabola1 (x : ℝ) : ℝ := 2*x^2 + 3*x - 4

/-- The second parabola -/
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem parabola_intersection :
  intersection_x = {(-1 + Real.sqrt 21) / 2, (-1 - Real.sqrt 21) / 2} ∧
  ∀ x ∈ intersection_x, parabola1 x = intersection_y ∧ parabola2 x = intersection_y :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3460_346028


namespace NUMINAMATH_CALUDE_point_on_bisector_coordinates_l3460_346036

/-- A point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The bisector of the first and third quadrants -/
def firstThirdQuadrantBisector (p : Point) : Prop :=
  p.x = p.y

/-- Point P with coordinates (a, 2a-1) -/
def P (a : ℝ) : Point :=
  { x := a, y := 2 * a - 1 }

/-- Theorem stating that if P(a) is on the bisector, its coordinates are (1, 1) -/
theorem point_on_bisector_coordinates :
  ∀ a : ℝ, firstThirdQuadrantBisector (P a) → P a = { x := 1, y := 1 } :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_bisector_coordinates_l3460_346036


namespace NUMINAMATH_CALUDE_peter_erasers_l3460_346052

theorem peter_erasers (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l3460_346052


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocals_l3460_346040

theorem inverse_sum_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a⁻¹ + 3 * b⁻¹)⁻¹ = a * b / (2 * b + 3 * a) :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocals_l3460_346040


namespace NUMINAMATH_CALUDE_max_dominoes_9x10_board_l3460_346084

/-- Represents a chessboard with given dimensions -/
structure Chessboard where
  rows : ℕ
  cols : ℕ

/-- Represents a domino with given dimensions -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of dominoes that can be placed on a chessboard -/
def max_dominoes (board : Chessboard) (domino : Domino) : ℕ :=
  sorry

/-- Theorem stating the maximum number of 6x1 dominoes on a 9x10 chessboard -/
theorem max_dominoes_9x10_board :
  let board := Chessboard.mk 9 10
  let domino := Domino.mk 6 1
  max_dominoes board domino = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_dominoes_9x10_board_l3460_346084


namespace NUMINAMATH_CALUDE_regular_hexagonal_prism_sum_l3460_346049

/-- A regular hexagonal prism -/
structure RegularHexagonalPrism where
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ

/-- The sum of faces, edges, and vertices of a regular hexagonal prism is 38 -/
theorem regular_hexagonal_prism_sum (prism : RegularHexagonalPrism) :
  prism.faces + prism.edges + prism.vertices = 38 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagonal_prism_sum_l3460_346049


namespace NUMINAMATH_CALUDE_initial_men_count_l3460_346079

/-- The number of men initially colouring the cloth -/
def M : ℕ := sorry

/-- The length of cloth coloured by M men in 2 days -/
def initial_cloth_length : ℝ := 48

/-- The time taken by M men to colour the initial cloth length -/
def initial_time : ℝ := 2

/-- The length of cloth coloured by 8 men in 0.75 days -/
def new_cloth_length : ℝ := 36

/-- The time taken by 8 men to colour the new cloth length -/
def new_time : ℝ := 0.75

/-- The number of men in the new scenario -/
def new_men : ℕ := 8

theorem initial_men_count : M = 4 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l3460_346079


namespace NUMINAMATH_CALUDE_largest_beta_exponent_l3460_346042

open Real

/-- Given a sequence of points in a plane with specific distance properties, 
    this theorem proves the largest possible exponent β for which r_n ≥ Cn^β holds. -/
theorem largest_beta_exponent 
  (O : ℝ × ℝ) 
  (P : ℕ → ℝ × ℝ) 
  (r : ℕ → ℝ) 
  (α : ℝ) 
  (h_alpha : 0 < α ∧ α < 1)
  (h_distance : ∀ n m : ℕ, n ≠ m → dist (P n) (P m) ≥ (r n) ^ α)
  (h_r_increasing : ∀ n : ℕ, r n ≤ r (n + 1))
  (h_r_def : ∀ n : ℕ, dist O (P n) = r n) :
  ∃ (C : ℝ) (h_C : C > 0), ∀ n : ℕ, r n ≥ C * n ^ (1 / (2 * (1 - α))) ∧ 
  ∀ β : ℝ, (∃ (D : ℝ) (h_D : D > 0), ∀ n : ℕ, r n ≥ D * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end NUMINAMATH_CALUDE_largest_beta_exponent_l3460_346042


namespace NUMINAMATH_CALUDE_net_gain_proof_l3460_346003

def initial_value : ℝ := 15000

def first_sale (value : ℝ) : ℝ := value * 1.2
def second_sale (value : ℝ) : ℝ := value * 0.85
def third_sale (value : ℝ) : ℝ := value * 1.1
def fourth_sale (value : ℝ) : ℝ := value * 0.95

def total_expense (initial : ℝ) : ℝ :=
  second_sale (first_sale initial) + fourth_sale (third_sale (second_sale (first_sale initial)))

def total_income (initial : ℝ) : ℝ :=
  first_sale initial + third_sale (second_sale (first_sale initial))

theorem net_gain_proof :
  total_income initial_value - total_expense initial_value = 3541.50 := by
  sorry

end NUMINAMATH_CALUDE_net_gain_proof_l3460_346003


namespace NUMINAMATH_CALUDE_initial_population_proof_l3460_346059

/-- Proves that the initial population is 10000 given the conditions --/
theorem initial_population_proof (P : ℝ) : 
  (P * (1 + 0.2)^2 = 14400) → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_initial_population_proof_l3460_346059


namespace NUMINAMATH_CALUDE_negation_of_all_even_divisible_by_two_l3460_346046

theorem negation_of_all_even_divisible_by_two :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, ¬(2 ∣ n) ∧ ¬(Even n)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_even_divisible_by_two_l3460_346046


namespace NUMINAMATH_CALUDE_problem_statement_l3460_346054

theorem problem_statement : 
  (∃ x : ℝ, x - 2 > 0) ∧ ¬(∀ x : ℝ, Real.sqrt x < x) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3460_346054


namespace NUMINAMATH_CALUDE_zoo_visitors_saturday_l3460_346065

theorem zoo_visitors_saturday (friday_visitors : ℕ) (saturday_multiplier : ℕ) : 
  friday_visitors = 3575 →
  saturday_multiplier = 5 →
  friday_visitors * saturday_multiplier = 17875 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_saturday_l3460_346065


namespace NUMINAMATH_CALUDE_platform_length_l3460_346047

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_cross_time : ℝ) (man_cross_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_cross_time = 30 →
  man_cross_time = 17 →
  (train_speed * platform_cross_time) - (train_speed * man_cross_time) = 260 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3460_346047


namespace NUMINAMATH_CALUDE_equation_solution_l3460_346011

theorem equation_solution : ∃! x : ℚ, x - 5/6 = 7/18 - x/4 ∧ x = 44/45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3460_346011


namespace NUMINAMATH_CALUDE_equal_roots_cubic_l3460_346018

theorem equal_roots_cubic (m n : ℝ) (h : n ≠ 0) :
  ∃ (x : ℝ), (x^3 + m*x - n = 0 ∧ n*x^3 - 2*m^2*x^2 - 5*m*n*x - 2*m^3 - n^2 = 0) →
  ∃ (a : ℝ), a = (n/2)^(1/3) ∧ 
  (∀ y : ℝ, y^3 + m*y - n = 0 ↔ y = a ∨ y = a ∨ y = -2*a) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_cubic_l3460_346018


namespace NUMINAMATH_CALUDE_phone_number_proof_l3460_346092

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def transform (n : ℕ) : ℕ :=
  2 * 10000000 + (n / 100000) * 1000000 + 800000 + (n % 100000)

theorem phone_number_proof (x : ℕ) (h1 : is_six_digit x) (h2 : transform x = 81 * x) :
  x = 260000 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_proof_l3460_346092


namespace NUMINAMATH_CALUDE_arctan_sum_three_four_l3460_346020

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_four_l3460_346020


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l3460_346057

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the directrix line
def directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the property of the moving circle
def circle_property (center : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 
  (center.1 - F.1)^2 + (center.2 - F.2)^2 = r^2 ∧
  ∃ (p : ℝ × ℝ), p ∈ directrix ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

-- Theorem statement
theorem trajectory_is_parabola :
  ∀ (center : ℝ × ℝ), circle_property center → center.2^2 = 4 * center.1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l3460_346057


namespace NUMINAMATH_CALUDE_simplify_expression_l3460_346050

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3460_346050


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3460_346068

/-- Given that x^4 varies inversely with the fourth root of w, 
    prove that when x = 6, w = 1/4096, given that x = 3 when w = 16 -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^4 * w^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = k) : 
  x = 6 → w = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3460_346068


namespace NUMINAMATH_CALUDE_ordering_theorem_l3460_346001

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def monotonically_decreasing_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f y < f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem ordering_theorem (h1 : monotonically_decreasing_neg f) (h2 : even_function f) :
  f (-1) < f 9 ∧ f 9 < f 13 := by
  sorry

end NUMINAMATH_CALUDE_ordering_theorem_l3460_346001


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3460_346038

theorem inequality_solution_set (x : ℝ) :
  x ≠ 2 ∧ x ≠ -9/2 →
  ((x + 1) / (x + 2) > (3*x + 4) / (2*x + 9)) ↔
  (x ∈ Set.Ioo (-9/2 : ℝ) (-2) ∪ Set.Ioo ((1 - Real.sqrt 5) / 2) ((1 + Real.sqrt 5) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3460_346038


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3460_346012

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimming problem, the swimmer's speed in still water is 5.5 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true = 35 / 5)   -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5)  -- Upstream condition
  : s.swimmer = 5.5 := by
  sorry

#eval 5.5  -- To check if the value 5.5 is recognized correctly

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3460_346012


namespace NUMINAMATH_CALUDE_good_numbers_exist_l3460_346008

def has_no_repeating_digits (n : ℕ) : Prop :=
  (n / 10) % 10 ≠ n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def increase_digits (n : ℕ) : ℕ :=
  ((n / 10) + 1) * 10 + ((n % 10) + 1)

theorem good_numbers_exist : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10 ≤ n₁ ∧ n₁ < 100 ∧
  10 ≤ n₂ ∧ n₂ < 100 ∧
  has_no_repeating_digits n₁ ∧
  has_no_repeating_digits n₂ ∧
  n₁ % sum_of_digits n₁ = 0 ∧
  n₂ % sum_of_digits n₂ = 0 ∧
  has_no_repeating_digits (increase_digits n₁) ∧
  has_no_repeating_digits (increase_digits n₂) ∧
  (increase_digits n₁) % sum_of_digits (increase_digits n₁) = 0 ∧
  (increase_digits n₂) % sum_of_digits (increase_digits n₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_good_numbers_exist_l3460_346008


namespace NUMINAMATH_CALUDE_f_min_value_a_value_l3460_346030

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (x : ℝ), ∀ (y : ℝ), f x ≤ f y ∧ f x = 3 := by sorry

-- Define the solution set condition
def solution_set_condition (a : ℝ) (m n : ℝ) : Prop :=
  (∀ x, m < x ∧ x < n ↔ f x + x - a < 0) ∧ n - m = 6

-- Theorem for the value of a
theorem a_value : ∀ (m n : ℝ), solution_set_condition 8 m n := by sorry

end NUMINAMATH_CALUDE_f_min_value_a_value_l3460_346030


namespace NUMINAMATH_CALUDE_genesis_work_hours_l3460_346071

/-- The number of hours Genesis worked per day on the new project -/
def hoursPerDayNewProject : ℕ := 6

/-- The number of weeks Genesis worked on the new project -/
def weeksNewProject : ℕ := 3

/-- The number of hours Genesis worked per day on the additional task -/
def hoursPerDayAdditionalTask : ℕ := 3

/-- The number of weeks Genesis worked on the additional task -/
def weeksAdditionalTask : ℕ := 2

/-- The number of days in a week -/
def daysPerWeek : ℕ := 7

/-- The total number of hours Genesis worked during the entire period -/
def totalHoursWorked : ℕ :=
  hoursPerDayNewProject * weeksNewProject * daysPerWeek +
  hoursPerDayAdditionalTask * weeksAdditionalTask * daysPerWeek

theorem genesis_work_hours : totalHoursWorked = 168 := by
  sorry

end NUMINAMATH_CALUDE_genesis_work_hours_l3460_346071


namespace NUMINAMATH_CALUDE_some_number_value_l3460_346087

theorem some_number_value (x : ℝ) : 60 + 5 * 12 / (180 / x) = 61 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3460_346087


namespace NUMINAMATH_CALUDE_rotated_angle_measure_l3460_346016

/-- Given an initial angle of 50 degrees that is rotated 580 degrees clockwise,
    the resulting acute angle is 90 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℕ) : 
  initial_angle = 50 → 
  rotation = 580 → 
  (initial_angle + rotation) % 360 = 270 → 
  360 - ((initial_angle + rotation) % 360) = 90 :=
by sorry

end NUMINAMATH_CALUDE_rotated_angle_measure_l3460_346016


namespace NUMINAMATH_CALUDE_greatest_common_divisor_546_126_under_30_l3460_346007

theorem greatest_common_divisor_546_126_under_30 : 
  ∃ (n : ℕ), n = 21 ∧ 
  n ∣ 546 ∧ 
  n < 30 ∧ 
  n ∣ 126 ∧
  ∀ (m : ℕ), m ∣ 546 → m < 30 → m ∣ 126 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_546_126_under_30_l3460_346007


namespace NUMINAMATH_CALUDE_invertible_elements_and_inverses_l3460_346031

-- Define the invertible elements and their inverses for modulo 8
def invertible_mod_8 : Set ℤ := {1, 3, 5, 7}
def inverse_mod_8 : ℤ → ℤ
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | _ => 0  -- Default case for non-invertible elements

-- Define the invertible elements and their inverses for modulo 9
def invertible_mod_9 : Set ℤ := {1, 2, 4, 5, 7, 8}
def inverse_mod_9 : ℤ → ℤ
  | 1 => 1
  | 2 => 5
  | 4 => 7
  | 5 => 2
  | 7 => 4
  | 8 => 8
  | _ => 0  -- Default case for non-invertible elements

theorem invertible_elements_and_inverses :
  (∀ x ∈ invertible_mod_8, (x * inverse_mod_8 x) % 8 = 1) ∧
  (∀ x ∈ invertible_mod_9, (x * inverse_mod_9 x) % 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_invertible_elements_and_inverses_l3460_346031


namespace NUMINAMATH_CALUDE_equation_solution_l3460_346024

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ↔ x = -2/3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3460_346024


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3460_346015

theorem sum_of_coefficients_equals_one (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^11 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                           a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3460_346015


namespace NUMINAMATH_CALUDE_ratio_chain_l3460_346072

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l3460_346072


namespace NUMINAMATH_CALUDE_intersection_limit_l3460_346088

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4

-- Define the horizontal line function
def g (m : ℝ) (x : ℝ) : ℝ := m

-- Define L(m) as the x-coordinate of the left endpoint of intersection
noncomputable def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)

-- Define r as a function of m
noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

-- Theorem statement
theorem intersection_limit :
  ∀ ε > 0, ∃ δ > 0, ∀ m : ℝ, 0 < |m| ∧ |m| < δ ∧ -4 < m ∧ m < 4 →
    |r m - (1/2)| < ε :=
sorry

end NUMINAMATH_CALUDE_intersection_limit_l3460_346088


namespace NUMINAMATH_CALUDE_tournament_probability_l3460_346048

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), m % 2 = 1 ∧ unique_wins_probability = (m : ℚ) / 2^409 :=
sorry

end NUMINAMATH_CALUDE_tournament_probability_l3460_346048


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3460_346066

/-- Proves that the speed of a boat in still water is 57 kmph given the conditions -/
theorem boat_speed_in_still_water : 
  ∀ (t : ℝ) (Vb : ℝ),
    t > 0 →  -- time taken to row downstream is positive
    Vb > 19 →  -- boat speed in still water is greater than stream speed
    (Vb - 19) * (2 * t) = (Vb + 19) * t →  -- equation based on distance = speed * time
    Vb = 57 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3460_346066


namespace NUMINAMATH_CALUDE_probability_adjacent_vertices_decagon_l3460_346014

/-- A decagon is a polygon with 10 vertices -/
def Decagon := Fin 10

/-- Two vertices in a decagon are adjacent if their indices differ by 1 (mod 10) -/
def adjacent (a b : Decagon) : Prop :=
  (a.val + 1) % 10 = b.val ∨ (b.val + 1) % 10 = a.val

/-- The total number of ways to choose 2 distinct vertices from a decagon -/
def total_choices : ℕ := 10 * 9 / 2

/-- The number of ways to choose 2 adjacent vertices from a decagon -/
def adjacent_choices : ℕ := 10

theorem probability_adjacent_vertices_decagon :
  (adjacent_choices : ℚ) / total_choices = 2 / 9 := by
  sorry

#eval (adjacent_choices : ℚ) / total_choices

end NUMINAMATH_CALUDE_probability_adjacent_vertices_decagon_l3460_346014


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3460_346077

theorem cos_alpha_value (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.cos α = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3460_346077


namespace NUMINAMATH_CALUDE_quadratic_roots_subset_l3460_346043

/-- Set A is defined as the solution set of x^2 + ax + b = 0 -/
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

/-- Set B is defined as {1, 2} -/
def B : Set ℝ := {1, 2}

/-- The theorem states that given the conditions, (a, b) must be one of the three specified pairs -/
theorem quadratic_roots_subset (a b : ℝ) : 
  A a b ⊆ B ∧ A a b ≠ ∅ → 
  ((a = -2 ∧ b = 1) ∨ (a = -4 ∧ b = 4) ∨ (a = -3 ∧ b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_subset_l3460_346043


namespace NUMINAMATH_CALUDE_trigonometric_properties_l3460_346055

theorem trigonometric_properties :
  (∀ α : Real, 0 < α ∧ α < Real.pi / 2 → Real.sin α > 0) ∧
  (∃ α : Real, 0 < α ∧ α < Real.pi / 2 ∧ Real.cos (2 * α) > 0) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_properties_l3460_346055


namespace NUMINAMATH_CALUDE_polynomial_value_l3460_346085

theorem polynomial_value : (3 : ℝ)^6 - 7 * 3 = 708 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3460_346085


namespace NUMINAMATH_CALUDE_special_parallelogram_side_ratio_l3460_346029

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- Adjacent sides of the parallelogram
  a : ℝ
  b : ℝ
  -- Diagonals of the parallelogram
  d1 : ℝ
  d2 : ℝ
  -- Conditions
  a_pos : 0 < a
  b_pos : 0 < b
  d1_pos : 0 < d1
  d2_pos : 0 < d2
  acute_angle : Real.cos (60 * π / 180) = 1 / 2
  diag_ratio : d1^2 / d2^2 = 1 / 3
  diag1_eq : d1^2 = a^2 + b^2 - a * b
  diag2_eq : d2^2 = a^2 + b^2 + a * b

/-- Theorem: In a special parallelogram, the ratio of adjacent sides is 1:1 -/
theorem special_parallelogram_side_ratio (p : SpecialParallelogram) : p.a = p.b := by
  sorry

end NUMINAMATH_CALUDE_special_parallelogram_side_ratio_l3460_346029


namespace NUMINAMATH_CALUDE_trapezoid_lower_side_length_l3460_346096

/-- Proves that the length of the lower side of a trapezoid is 17.65 cm given specific conditions -/
theorem trapezoid_lower_side_length 
  (height : ℝ) 
  (area : ℝ) 
  (side_difference : ℝ) 
  (h1 : height = 5.2)
  (h2 : area = 100.62)
  (h3 : side_difference = 3.4) : 
  ∃ (lower_side : ℝ), lower_side = 17.65 ∧ 
  area = (1/2) * (lower_side + (lower_side + side_difference)) * height :=
sorry

end NUMINAMATH_CALUDE_trapezoid_lower_side_length_l3460_346096


namespace NUMINAMATH_CALUDE_janets_freelance_rate_janets_freelance_rate_is_33_75_l3460_346061

/-- Calculates Janet's hourly rate as a freelancer given her current job details and additional costs --/
theorem janets_freelance_rate (current_hourly_rate : ℝ) 
  (weekly_hours : ℝ) (weeks_per_month : ℝ) (extra_fica_per_week : ℝ) 
  (healthcare_premium : ℝ) (additional_monthly_income : ℝ) : ℝ :=
  let current_monthly_income := current_hourly_rate * weekly_hours * weeks_per_month
  let additional_costs := extra_fica_per_week * weeks_per_month + healthcare_premium
  let freelance_income := current_monthly_income + additional_monthly_income
  let net_freelance_income := freelance_income - additional_costs
  let monthly_hours := weekly_hours * weeks_per_month
  net_freelance_income / monthly_hours

/-- Proves that Janet's freelance hourly rate is $33.75 given the specified conditions --/
theorem janets_freelance_rate_is_33_75 : 
  janets_freelance_rate 30 40 4 25 400 1100 = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_janets_freelance_rate_janets_freelance_rate_is_33_75_l3460_346061


namespace NUMINAMATH_CALUDE_min_sum_of_product_144_l3460_346098

theorem min_sum_of_product_144 (a b : ℤ) (h : a * b = 144) :
  ∀ (x y : ℤ), x * y = 144 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_144_l3460_346098


namespace NUMINAMATH_CALUDE_set_intersection_problem_l3460_346002

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem set_intersection_problem : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l3460_346002


namespace NUMINAMATH_CALUDE_triangle_side_length_l3460_346004

theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 5 →
  Real.cos (Y - Z) = 21 / 32 →
  x^2 = 47.75 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3460_346004


namespace NUMINAMATH_CALUDE_solve_for_c_l3460_346006

theorem solve_for_c : ∃ C : ℝ, (4 * C + 5 = 25) ∧ (C = 5) := by sorry

end NUMINAMATH_CALUDE_solve_for_c_l3460_346006


namespace NUMINAMATH_CALUDE_problem_solution_l3460_346094

theorem problem_solution (x y : ℝ) (h1 : x^(2*y) = 9) (h2 : x = 3) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3460_346094


namespace NUMINAMATH_CALUDE_largest_multiple_of_45_with_nine_and_zero_m_div_45_l3460_346035

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def only_nine_and_zero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 9 ∨ d = 0

/-- The largest positive integer that is a multiple of 45 and consists only of digits 9 and 0 -/
def m : ℕ := 99990

theorem largest_multiple_of_45_with_nine_and_zero :
  m % 45 = 0 ∧
  only_nine_and_zero m ∧
  ∀ n : ℕ, n % 45 = 0 → only_nine_and_zero n → n ≤ m :=
sorry

theorem m_div_45 : m / 45 = 2222 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_45_with_nine_and_zero_m_div_45_l3460_346035


namespace NUMINAMATH_CALUDE_smallest_side_range_l3460_346081

theorem smallest_side_range (c : ℝ) (a b d : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) (h4 : d > 0) 
  (h5 : a + b + d = c) (h6 : d = 2 * a) (h7 : a ≤ b) (h8 : a ≤ d) : 
  c / 6 < a ∧ a < c / 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_side_range_l3460_346081


namespace NUMINAMATH_CALUDE_trackball_mice_count_l3460_346069

theorem trackball_mice_count (total : ℕ) (wireless_ratio : ℚ) (optical_ratio : ℚ) :
  total = 80 →
  wireless_ratio = 1/2 →
  optical_ratio = 1/4 →
  (wireless_ratio + optical_ratio + (1 - wireless_ratio - optical_ratio) : ℚ) = 1 →
  ↑total * (1 - wireless_ratio - optical_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_trackball_mice_count_l3460_346069


namespace NUMINAMATH_CALUDE_train_arrival_time_l3460_346091

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem train_arrival_time 
  (departure : Time)
  (journey_duration : Nat)
  (h1 : departure = { hours := 9, minutes := 45 })
  (h2 : journey_duration = 15) :
  addMinutes departure journey_duration = { hours := 10, minutes := 0 } :=
sorry

end NUMINAMATH_CALUDE_train_arrival_time_l3460_346091


namespace NUMINAMATH_CALUDE_merchant_max_profit_optimal_selling_price_l3460_346076

/-- Represents the merchant's profit function -/
def profit (x : ℝ) : ℝ := -10 * x^2 + 80 * x + 200

/-- The optimal price increase that maximizes profit -/
def optimal_increase : ℝ := 4

/-- The maximum achievable profit -/
def max_profit : ℝ := 360

theorem merchant_max_profit :
  (∀ x, 0 ≤ x → x < 10 → profit x ≤ max_profit) ∧
  profit optimal_increase = max_profit :=
sorry

theorem optimal_selling_price :
  optimal_increase + 10 = 14 :=
sorry

end NUMINAMATH_CALUDE_merchant_max_profit_optimal_selling_price_l3460_346076


namespace NUMINAMATH_CALUDE_tournament_matches_l3460_346082

def matches_in_group (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tournament_matches : 
  let group_a_players : ℕ := 6
  let group_b_players : ℕ := 5
  matches_in_group group_a_players + matches_in_group group_b_players = 25 := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_l3460_346082


namespace NUMINAMATH_CALUDE_janes_quiz_mean_l3460_346045

theorem janes_quiz_mean : 
  let scores : List ℝ := [86, 91, 89, 95, 88, 94]
  (scores.sum / scores.length : ℝ) = 90.5 := by sorry

end NUMINAMATH_CALUDE_janes_quiz_mean_l3460_346045


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3460_346039

/-- A hyperbola is defined by its standard equation parameters a and b,
    where the equation is (x²/a² - y²/b² = 1) or (y²/a² - x²/b² = 1) -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  isVertical : Bool

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space, defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point lies on a hyperbola -/
def pointOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  if h.isVertical then
    p.y^2 / h.a^2 - p.x^2 / h.b^2 = 1
  else
    p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Checks if a line is an asymptote of a hyperbola -/
def isAsymptote (h : Hyperbola) (l : Line) : Prop :=
  if h.isVertical then
    l.slope = h.a / h.b ∨ l.slope = -h.a / h.b
  else
    l.slope = h.b / h.a ∨ l.slope = -h.b / h.a

theorem hyperbola_standard_equation
  (h : Hyperbola)
  (p : Point)
  (l : Line)
  (h_point : pointOnHyperbola h p)
  (h_asymptote : isAsymptote h l)
  (h_p_coords : p.x = 1 ∧ p.y = 2 * Real.sqrt 2)
  (h_l_equation : l.slope = 2 ∧ l.yIntercept = 0) :
  h.a = 2 ∧ h.b = 1 ∧ h.isVertical = true :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3460_346039


namespace NUMINAMATH_CALUDE_ratio_after_adding_water_l3460_346074

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratio (m : Mixture) : ℝ × ℝ :=
  (m.alcohol, m.water)

/-- Adds water to a mixture -/
def add_water (m : Mixture) (amount : ℝ) : Mixture :=
  { alcohol := m.alcohol, water := m.water + amount }

/-- The initial mixture -/
def initial_mixture : Mixture :=
  { alcohol := 4, water := 3 }

/-- The amount of water added -/
def water_added : ℝ := 8

/-- Theorem stating that adding water changes the ratio to 4:11 -/
theorem ratio_after_adding_water :
  ratio (add_water initial_mixture water_added) = (4, 11) := by
  sorry

end NUMINAMATH_CALUDE_ratio_after_adding_water_l3460_346074


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l3460_346062

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a binary representation (list of bits). -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  let a := [true, false, true, true]   -- 1101₂
  let b := [true, true, true]          -- 111₂
  let c := [false, true, false, true]  -- 1010₂
  let d := [true, false, false, true]  -- 1001₂
  let result := [true, true, true, false, false, true] -- 100111₂
  binary_to_nat a + binary_to_nat b - binary_to_nat c + binary_to_nat d =
  binary_to_nat result :=
by
  sorry


end NUMINAMATH_CALUDE_binary_addition_subtraction_l3460_346062


namespace NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l3460_346051

-- Define the necessary structures
structure Parallelogram :=
  (diagonals_bisect : Bool)

structure Rhombus :=
  (is_parallelogram : Bool)
  (diagonals_bisect : Bool)

-- State the theorem
theorem rhombus_diagonals_bisect :
  (∀ p : Parallelogram, p.diagonals_bisect = true) →
  (∀ r : Rhombus, r.is_parallelogram = true) →
  (∀ r : Rhombus, r.diagonals_bisect = true) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l3460_346051


namespace NUMINAMATH_CALUDE_odd_scripts_in_final_state_l3460_346053

/-- Represents the state of the box of scripts -/
structure ScriptBox where
  total : Nat
  odd : Nat
  even : Nat

/-- The procedure of selecting and manipulating scripts -/
def select_and_manipulate (box : ScriptBox) : ScriptBox :=
  sorry

/-- Represents the final state of the box -/
def final_state (initial : ScriptBox) : ScriptBox :=
  sorry

theorem odd_scripts_in_final_state :
  ∀ (initial : ScriptBox),
    initial.total = 4032 →
    initial.odd = initial.total / 2 →
    initial.even = initial.total / 2 →
    let final := final_state initial
    final.total = 3 →
    final.odd > 0 →
    final.even > 0 →
    final.odd = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_scripts_in_final_state_l3460_346053


namespace NUMINAMATH_CALUDE_perpendicular_probability_l3460_346097

/-- A square is a shape with 4 vertices -/
structure Square where
  vertices : Finset (ℕ × ℕ)
  vertex_count : vertices.card = 4

/-- A line in a square is defined by two distinct vertices -/
structure Line (s : Square) where
  v1 : s.vertices
  v2 : s.vertices
  distinct : v1 ≠ v2

/-- Two lines are perpendicular if they form a right angle -/
def perpendicular (s : Square) (l1 l2 : Line s) : Prop := sorry

/-- The total number of possible line pairs in a square -/
def total_line_pairs (s : Square) : ℕ := sorry

/-- The number of perpendicular line pairs in a square -/
def perpendicular_line_pairs (s : Square) : ℕ := sorry

/-- The theorem to be proved -/
theorem perpendicular_probability (s : Square) : 
  (perpendicular_line_pairs s : ℚ) / (total_line_pairs s : ℚ) = 5 / 18 := sorry

end NUMINAMATH_CALUDE_perpendicular_probability_l3460_346097


namespace NUMINAMATH_CALUDE_angle_of_inclination_at_max_area_l3460_346032

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y = (k - 1) * x + 2

-- Define the circle equation
def circle_equation (k x y : ℝ) : Prop := x^2 + y^2 + k*x + 2*y + k^2 = 0

-- Define the condition for maximum area of the circle
def max_area_condition (k : ℝ) : Prop := k = 0

-- Theorem statement
theorem angle_of_inclination_at_max_area (k : ℝ) :
  max_area_condition k →
  ∃ (x y : ℝ), line_equation k x y ∧ circle_equation k x y →
  Real.arctan (-1) = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_CALUDE_angle_of_inclination_at_max_area_l3460_346032


namespace NUMINAMATH_CALUDE_school_population_l3460_346017

theorem school_population (girls : ℕ) (boys : ℕ) (teachers : ℕ) (staff : ℕ)
  (h1 : girls = 542)
  (h2 : boys = 387)
  (h3 : teachers = 45)
  (h4 : staff = 27) :
  girls + boys + teachers + staff = 1001 := by
sorry

end NUMINAMATH_CALUDE_school_population_l3460_346017


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l3460_346083

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) where
  radius_larger : R > r

/-- Point on a circle -/
structure PointOnCircle (center : ℝ × ℝ) (radius : ℝ) where
  point : ℝ × ℝ
  on_circle : (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

/-- Theorem about the sum of squared distances and the locus of midpoint -/
theorem concentric_circles_theorem
  (R r : ℝ) (h : ConcentricCircles R r)
  (O : ℝ × ℝ) -- Center of the circles
  (P : PointOnCircle O r) -- Fixed point on smaller circle
  (B : PointOnCircle O R) -- Moving point on larger circle
  (A : PointOnCircle O r) -- Point on smaller circle determined by perpendicular line from P to BP
  (C : PointOnCircle O R) -- Intersection of BP with larger circle
  : 
  -- Part 1: Sum of squared distances
  (B.point.1 - C.point.1)^2 + (B.point.2 - C.point.2)^2 +
  (C.point.1 - A.point.1)^2 + (C.point.2 - A.point.2)^2 +
  (A.point.1 - B.point.1)^2 + (A.point.2 - B.point.2)^2 = 6 * R^2 + 2 * r^2
  ∧
  -- Part 2: Locus of midpoint of AB
  ∃ (Q : ℝ × ℝ),
    Q = ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2) ∧
    (Q.1 - (O.1 + P.point.1) / 2)^2 + (Q.2 - (O.2 + P.point.2) / 2)^2 = (R / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l3460_346083


namespace NUMINAMATH_CALUDE_real_part_of_z_is_two_l3460_346027

theorem real_part_of_z_is_two : Complex.re (((Complex.I - 1)^2 + 1) / Complex.I^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_two_l3460_346027


namespace NUMINAMATH_CALUDE_extra_flowers_count_l3460_346086

def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4
def used_flowers : ℕ := 19

def total_picked : ℕ := tulips + roses + daisies + lilies

theorem extra_flowers_count : total_picked - used_flowers = 8 := by
  sorry

end NUMINAMATH_CALUDE_extra_flowers_count_l3460_346086


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3460_346000

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (average_books : ℚ) (h1 : total_students = 25) (h2 : zero_books = 3) (h3 : one_book = 10) 
  (h4 : two_books = 4) (h5 : average_books = 5/2) : ℕ :=
  let total_books := (total_students : ℚ) * average_books
  let accounted_students := zero_books + one_book + two_books
  let remaining_students := total_students - accounted_students
  let accounted_books := one_book * 1 + two_books * 2
  let remaining_books := total_books - accounted_books
  let min_books_per_remaining := 3
  24

end NUMINAMATH_CALUDE_max_books_borrowed_l3460_346000


namespace NUMINAMATH_CALUDE_pat_earned_stickers_l3460_346037

/-- The number of stickers Pat had at the beginning of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by sorry

end NUMINAMATH_CALUDE_pat_earned_stickers_l3460_346037


namespace NUMINAMATH_CALUDE_arrange_seven_books_three_identical_l3460_346056

/-- The number of ways to arrange books with some identical copies -/
def arrange_books (total : ℕ) (identical : ℕ) : ℕ :=
  Nat.factorial total / Nat.factorial identical

/-- Theorem: Arranging 7 books with 3 identical copies yields 840 possibilities -/
theorem arrange_seven_books_three_identical :
  arrange_books 7 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_arrange_seven_books_three_identical_l3460_346056


namespace NUMINAMATH_CALUDE_inequality_proof_l3460_346009

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) : a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3460_346009


namespace NUMINAMATH_CALUDE_complex_equation_real_part_l3460_346067

theorem complex_equation_real_part : 
  ∀ z : ℂ, (1 + Complex.I) * z = Complex.I → Complex.re z = (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_l3460_346067
