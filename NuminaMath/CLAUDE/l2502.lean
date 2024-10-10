import Mathlib

namespace intersection_integer_iff_k_valid_l2502_250275

/-- A point in the Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Check if a point is the intersection of two lines -/
def is_intersection (p : Point) (k : ℤ) : Prop :=
  p.y = p.x - 2 ∧ p.y = k * p.x + k

/-- The set of valid k values -/
def valid_k : Set ℤ := {-2, 0, 2, 4}

/-- Main theorem: The intersection is an integer point iff k is in the valid set -/
theorem intersection_integer_iff_k_valid (k : ℤ) :
  (∃ p : Point, is_intersection p k) ↔ k ∈ valid_k :=
sorry

end intersection_integer_iff_k_valid_l2502_250275


namespace ticket_sales_total_l2502_250250

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (adult_price child_price : ℕ) (total_tickets children_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - children_tickets
  adult_price * adult_tickets + child_price * children_tickets

/-- Theorem stating that the total money collected is $104 -/
theorem ticket_sales_total : 
  total_money_collected 6 4 21 11 = 104 := by
  sorry

end ticket_sales_total_l2502_250250


namespace decrease_xyz_squared_l2502_250295

theorem decrease_xyz_squared (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  x' * y' * z' ^ 2 = 0.1296 * x * y * z ^ 2 := by
sorry

end decrease_xyz_squared_l2502_250295


namespace least_divisible_by_first_ten_l2502_250253

theorem least_divisible_by_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∧
  n = 2520 := by
  sorry

end least_divisible_by_first_ten_l2502_250253


namespace time_to_put_30_toys_is_14_minutes_l2502_250266

/-- The time required to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_cycle : ℕ) (toys_out_per_cycle : ℕ) (cycle_duration : ℕ) : ℚ :=
  let net_increase := toys_in_per_cycle - toys_out_per_cycle
  let cycles_needed := (total_toys - toys_in_per_cycle) / net_increase
  let total_seconds := cycles_needed * cycle_duration + cycle_duration
  total_seconds / 60

/-- Theorem: The time to put 30 toys in the box is 14 minutes -/
theorem time_to_put_30_toys_is_14_minutes :
  time_to_put_toys_in_box 30 3 2 30 = 14 := by
  sorry

end time_to_put_30_toys_is_14_minutes_l2502_250266


namespace energetic_time_proof_l2502_250222

def initial_speed : ℝ := 25
def tired_speed : ℝ := 15
def rest_time : ℝ := 0.5
def total_distance : ℝ := 132
def total_time : ℝ := 8

theorem energetic_time_proof :
  ∃ x : ℝ, 
    x ≥ 0 ∧
    x ≤ total_time - rest_time ∧
    initial_speed * x + tired_speed * (total_time - rest_time - x) = total_distance ∧
    x = 39 / 20 := by
  sorry

end energetic_time_proof_l2502_250222


namespace number_puzzle_l2502_250299

theorem number_puzzle : ∃ x : ℝ, 3 * (x + 2) = 24 + x ∧ x = 9 := by sorry

end number_puzzle_l2502_250299


namespace third_number_proof_l2502_250269

theorem third_number_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 48 → 
  (a + b) / 2 = 56 → 
  c = 32 := by
sorry

end third_number_proof_l2502_250269


namespace square_root_of_factorial_fraction_l2502_250232

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 126) = 24 * Real.sqrt 5 := by
  sorry

end square_root_of_factorial_fraction_l2502_250232


namespace frog_weight_ratio_l2502_250240

/-- The ratio of the weight of the largest frog to the smallest frog is 10 -/
theorem frog_weight_ratio :
  ∀ (small_frog large_frog : ℝ),
  large_frog = 120 →
  large_frog = small_frog + 108 →
  large_frog / small_frog = 10 := by
sorry

end frog_weight_ratio_l2502_250240


namespace arcsin_of_one_equals_pi_div_two_l2502_250238

theorem arcsin_of_one_equals_pi_div_two : Real.arcsin 1 = π / 2 := by
  sorry

end arcsin_of_one_equals_pi_div_two_l2502_250238


namespace quadratic_roots_properties_l2502_250278

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 5*x₁ - 3 = 0) 
  (h₂ : x₂^2 - 5*x₂ - 3 = 0) :
  (x₁^2 + x₂^2 = 31) ∧ (1/x₁ - 1/x₂ = Real.sqrt 37 / 3) := by
  sorry

end quadratic_roots_properties_l2502_250278


namespace factorial_ratio_equals_twelve_l2502_250281

theorem factorial_ratio_equals_twelve : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 5) = 12 := by
  sorry

end factorial_ratio_equals_twelve_l2502_250281


namespace james_amy_balloon_difference_l2502_250265

/-- 
Given that James has 232 balloons and Amy has 101 balloons, 
prove that James has 131 more balloons than Amy.
-/
theorem james_amy_balloon_difference : 
  let james_balloons : ℕ := 232
  let amy_balloons : ℕ := 101
  james_balloons - amy_balloons = 131 := by
sorry

end james_amy_balloon_difference_l2502_250265


namespace football_players_count_l2502_250273

theorem football_players_count (total_players cricket_players hockey_players softball_players : ℕ) 
  (h1 : total_players = 59)
  (h2 : cricket_players = 16)
  (h3 : hockey_players = 12)
  (h4 : softball_players = 13) :
  total_players - (cricket_players + hockey_players + softball_players) = 18 := by
  sorry

end football_players_count_l2502_250273


namespace complex_number_location_l2502_250256

theorem complex_number_location :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end complex_number_location_l2502_250256


namespace polynomial_factorization_l2502_250287

theorem polynomial_factorization (x y : ℝ) :
  -6 * x^2 * y + 12 * x * y^2 - 3 * x * y = -3 * x * y * (2 * x - 4 * y + 1) := by
  sorry

end polynomial_factorization_l2502_250287


namespace white_square_area_main_white_square_area_l2502_250263

-- Define the cube's side length
def cubeSide : ℝ := 12

-- Define the total amount of blue paint
def totalBluePaint : ℝ := 432

-- Define the number of faces on a cube
def numFaces : ℕ := 6

-- Theorem statement
theorem white_square_area (cubeSide : ℝ) (totalBluePaint : ℝ) (numFaces : ℕ) :
  cubeSide > 0 →
  totalBluePaint > 0 →
  numFaces = 6 →
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

-- Main theorem using the defined constants
theorem main_white_square_area : 
  let totalSurfaceArea := numFaces * cubeSide * cubeSide
  let bluePaintPerFace := totalBluePaint / numFaces
  let whiteSquareArea := cubeSide * cubeSide - bluePaintPerFace
  whiteSquareArea = 72 := by
  sorry

end white_square_area_main_white_square_area_l2502_250263


namespace odd_squares_sum_power_of_two_l2502_250213

theorem odd_squares_sum_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ x y : ℤ, Odd x ∧ Odd y ∧ x^2 + 7*y^2 = 2^n := by
  sorry

end odd_squares_sum_power_of_two_l2502_250213


namespace proposition_equivalences_and_set_equality_l2502_250217

-- Define the proposition
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1 ∨ x = 2

-- Define the sets P and S
def setP : Set ℝ := {x | -1 < x ∧ x < 3}
def setS (a : ℝ) : Set ℝ := {x | x^2 + (a+1)*x + a < 0}

theorem proposition_equivalences_and_set_equality :
  (∀ x, Q x → P x) ∧
  (∀ x, ¬(P x) → ¬(Q x)) ∧
  (∀ x, ¬(Q x) → ¬(P x)) ∧
  ∃ a, setP = setS a ∧ a = -3 := by sorry

end proposition_equivalences_and_set_equality_l2502_250217


namespace union_of_P_and_Q_l2502_250243

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {2, 4}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3, 4} := by sorry

end union_of_P_and_Q_l2502_250243


namespace odd_cycle_existence_l2502_250282

/-- A graph is a structure with vertices and edges. -/
structure Graph (V : Type) :=
  (edges : V → V → Prop)

/-- The degree of a vertex in a graph is the number of edges incident to it. -/
def degree {V : Type} (G : Graph V) (v : V) : ℕ := sorry

/-- The minimum degree of a graph is the minimum of the degrees of all vertices. -/
def min_degree {V : Type} (G : Graph V) : ℕ := sorry

/-- A path in a graph is a sequence of vertices where each adjacent pair is connected by an edge. -/
def is_path {V : Type} (G : Graph V) (p : List V) : Prop := sorry

/-- A cycle in a graph is a path that starts and ends at the same vertex. -/
def is_cycle {V : Type} (G : Graph V) (c : List V) : Prop := sorry

/-- The length of a cycle is the number of edges in the cycle. -/
def cycle_length {V : Type} (c : List V) : ℕ := sorry

/-- A theorem stating that any graph with minimum degree at least 3 contains an odd cycle. -/
theorem odd_cycle_existence {V : Type} (G : Graph V) :
  min_degree G ≥ 3 → ∃ c : List V, is_cycle G c ∧ Odd (cycle_length c) := by
  sorry

end odd_cycle_existence_l2502_250282


namespace cistern_problem_l2502_250283

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

theorem cistern_problem :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  cistern_wet_surface_area length width depth = 49 := by
  sorry

end cistern_problem_l2502_250283


namespace least_divisor_for_perfect_square_l2502_250239

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (16800 / n) ∧ 
  ∀ m : ℕ, m > 0 → is_perfect_square (16800 / m) → n ≤ m :=
by sorry

end least_divisor_for_perfect_square_l2502_250239


namespace quadratic_roots_product_l2502_250204

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p^2 + 5 * p - 7 = 0) → 
  (3 * q^2 + 5 * q - 7 = 0) → 
  (p - 2) * (q - 2) = 5 := by
sorry

end quadratic_roots_product_l2502_250204


namespace range_of_m_l2502_250259

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -7 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 7}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 3*m - 2}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 4 :=
sorry

end range_of_m_l2502_250259


namespace price_reduction_problem_l2502_250261

/-- The price reduction problem -/
theorem price_reduction_problem (reduced_price : ℝ) (extra_oil : ℝ) (total_money : ℝ) 
  (h1 : reduced_price = 15)
  (h2 : extra_oil = 6)
  (h3 : total_money = 900) :
  let original_price := total_money / (total_money / reduced_price - extra_oil)
  let percentage_reduction := (original_price - reduced_price) / original_price * 100
  ∃ (ε : ℝ), ε > 0 ∧ abs (percentage_reduction - 10) < ε :=
by sorry

end price_reduction_problem_l2502_250261


namespace line_segment_proportion_l2502_250201

theorem line_segment_proportion (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a / c = c / b → a = 4 → b = 9 → c = 6 := by
  sorry

end line_segment_proportion_l2502_250201


namespace coin_flip_probability_l2502_250272

theorem coin_flip_probability : 
  let p_heads : ℝ := 1/2  -- probability of getting heads on a single flip
  let n : ℕ := 5  -- number of flips
  let target_sequence := List.replicate 4 true ++ [false]  -- HTTT (true for heads, false for tails)
  
  (target_sequence.map (fun h => if h then p_heads else 1 - p_heads)).prod = 1/32 :=
by sorry

end coin_flip_probability_l2502_250272


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l2502_250242

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -16
  let b : ℝ := 48
  let c : ℝ := -75
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 3 :=
by sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_equation_l2502_250242


namespace calculate_expression_l2502_250264

theorem calculate_expression : 
  |-Real.sqrt 3| + (1/2)⁻¹ + (Real.pi + 1)^0 - Real.tan (60 * π / 180) = 3 := by
  sorry

end calculate_expression_l2502_250264


namespace supplementary_to_complementary_ratio_l2502_250223

/-- 
Given an angle of 45 degrees, prove that the ratio of its supplementary angle 
to its complementary angle is 3:1.
-/
theorem supplementary_to_complementary_ratio 
  (angle : ℝ) 
  (h_angle : angle = 45) 
  (h_supplementary : ℝ → ℝ → Prop)
  (h_complementary : ℝ → ℝ → Prop)
  (h_supp_def : ∀ x y, h_supplementary x y ↔ x + y = 180)
  (h_comp_def : ∀ x y, h_complementary x y ↔ x + y = 90) :
  (180 - angle) / (90 - angle) = 3 := by
sorry

end supplementary_to_complementary_ratio_l2502_250223


namespace factorization_equality_l2502_250252

theorem factorization_equality (a b : ℝ) : 2*a - 8*a*b^2 = 2*a*(1-2*b)*(1+2*b) := by
  sorry

end factorization_equality_l2502_250252


namespace truck_catch_up_time_is_fifteen_l2502_250276

/-- Represents a vehicle with a constant speed -/
structure Vehicle where
  speed : ℝ

/-- Represents the state of the vehicles at a given time -/
structure VehicleState where
  bus : Vehicle
  truck : Vehicle
  car : Vehicle
  time : ℝ
  busTruckDistance : ℝ
  truckCarDistance : ℝ

/-- The initial state of the vehicles -/
def initialState : VehicleState := sorry

/-- The state after the car catches up with the truck -/
def carTruckCatchUpState : VehicleState := sorry

/-- The state after the car catches up with the bus -/
def carBusCatchUpState : VehicleState := sorry

/-- The state after the truck catches up with the bus -/
def truckBusCatchUpState : VehicleState := sorry

/-- The time it takes for the truck to catch up with the bus after the car catches up with the bus -/
def truckCatchUpTime : ℝ := truckBusCatchUpState.time - carBusCatchUpState.time

theorem truck_catch_up_time_is_fifteen :
  truckCatchUpTime = 15 := by sorry

end truck_catch_up_time_is_fifteen_l2502_250276


namespace min_value_f_l2502_250229

/-- The function f(x) = (x^2 + 2) / x has a minimum value of 2√2 for x > 1 -/
theorem min_value_f (x : ℝ) (h : x > 1) : (x^2 + 2) / x ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_f_l2502_250229


namespace range_of_a_l2502_250244

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ - 1 < 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  a ∈ Set.Icc (-1) 1 ∪ Set.Ioi 3 :=
sorry

end range_of_a_l2502_250244


namespace square_side_length_l2502_250216

theorem square_side_length (perimeter : ℚ) (h : perimeter = 12 / 25) :
  perimeter / 4 = 12 / 100 := by
  sorry

end square_side_length_l2502_250216


namespace vector_magnitude_problem_l2502_250211

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : angle_between a b = Real.pi / 4)
  (h2 : Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2)) = Real.sqrt 2)
  (h3 : Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2)) = 3) :
  Real.sqrt (((2 * a.1 - b.1) ^ 2) + ((2 * a.2 - b.2) ^ 2)) = Real.sqrt 5 := by
  sorry

end vector_magnitude_problem_l2502_250211


namespace expression_evaluation_l2502_250247

theorem expression_evaluation :
  let x : ℚ := -1/3
  (3*x + 2) * (3*x - 2) - 5*x*(x - 1) - (2*x - 1)^2 = -8 := by sorry

end expression_evaluation_l2502_250247


namespace tan_double_angle_l2502_250241

theorem tan_double_angle (α : Real) :
  (2 * Real.cos α + Real.sin α) / (Real.cos α - 2 * Real.sin α) = -1 →
  Real.tan (2 * α) = -3/4 := by
  sorry

end tan_double_angle_l2502_250241


namespace negation_of_absolute_value_inequality_l2502_250270

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x + 1| ≥ 0) ↔ (∃ x : ℝ, |x + 1| < 0) := by sorry

end negation_of_absolute_value_inequality_l2502_250270


namespace tshirts_bought_l2502_250285

/-- Given the price conditions for pants and t-shirts, 
    prove the number of t-shirts that can be bought with 800 Rs. -/
theorem tshirts_bought (pants_price t_shirt_price : ℕ) : 
  (3 * pants_price + 6 * t_shirt_price = 1500) →
  (pants_price + 12 * t_shirt_price = 1500) →
  (800 / t_shirt_price = 8) := by
  sorry

end tshirts_bought_l2502_250285


namespace min_value_theorem_equality_condition_unique_minimum_l2502_250249

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 10*x + 100/x^3 ≥ 40 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 10*x + 100/x^3 = 40 := by
  sorry

theorem unique_minimum (x : ℝ) (h1 : x > 0) (h2 : x^2 + 10*x + 100/x^3 = 40) : x = 2 := by
  sorry

end min_value_theorem_equality_condition_unique_minimum_l2502_250249


namespace exists_self_intersecting_net_l2502_250218

/-- A tetrahedron is represented by its four vertices in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A net of a tetrahedron is represented by the 2D coordinates of its vertices -/
structure TetrahedronNet where
  vertices : Fin 4 → ℝ × ℝ

/-- A function that determines if a tetrahedron net self-intersects -/
def self_intersects (net : TetrahedronNet) : Prop :=
  sorry

/-- A function that cuts a tetrahedron along three edges not belonging to the same face -/
def cut_tetrahedron (t : Tetrahedron) : TetrahedronNet :=
  sorry

/-- The main theorem: there exists a tetrahedron whose net self-intersects -/
theorem exists_self_intersecting_net :
  ∃ t : Tetrahedron, self_intersects (cut_tetrahedron t) :=
sorry

end exists_self_intersecting_net_l2502_250218


namespace intersection_of_three_lines_l2502_250215

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y k : ℚ) : 
  (y = 4*x - 2) ∧ (y = -3*x + 9) ∧ (y = 2*x + k) → k = 8/7 := by
  sorry

end intersection_of_three_lines_l2502_250215


namespace complex_number_problem_l2502_250258

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property of being a purely imaginary number
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the property of being a real number
def isRealNumber (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem complex_number_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isRealNumber ((z + 2) / (1 - i))) : 
  z = -2 * i := by
  sorry

end complex_number_problem_l2502_250258


namespace rectangle_width_l2502_250289

theorem rectangle_width (a w l d : ℝ) : 
  a > 0 → w > 0 → l > 0 → d > 0 →
  a = w * l →
  d^2 = w^2 + l^2 →
  a = 12 →
  d = 5 →
  w = 3 := by
sorry

end rectangle_width_l2502_250289


namespace iphone_discount_l2502_250220

theorem iphone_discount (iphone_price iwatch_price iwatch_discount cashback_rate final_price : ℝ) :
  iphone_price = 800 →
  iwatch_price = 300 →
  iwatch_discount = 0.1 →
  cashback_rate = 0.02 →
  final_price = 931 →
  ∃ (iphone_discount : ℝ),
    iphone_discount = 0.15 ∧
    final_price = (1 - cashback_rate) * (iphone_price * (1 - iphone_discount) + iwatch_price * (1 - iwatch_discount)) :=
by sorry

end iphone_discount_l2502_250220


namespace genevieve_coffee_consumption_l2502_250291

/-- Proves that Genevieve drank 6 pints of coffee given the conditions -/
theorem genevieve_coffee_consumption 
  (total_coffee : ℚ) 
  (num_thermoses : ℕ) 
  (genevieve_thermoses : ℕ) 
  (h1 : total_coffee = 4.5) 
  (h2 : num_thermoses = 18) 
  (h3 : genevieve_thermoses = 3) 
  (h4 : ∀ g : ℚ, g * 8 = g * (8 : ℚ)) -- Conversion from gallons to pints
  : (total_coffee * 8 * genevieve_thermoses) / num_thermoses = 6 := by
  sorry

end genevieve_coffee_consumption_l2502_250291


namespace cinnamon_swirls_theorem_l2502_250208

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- The total number of cinnamon swirl pieces prepared -/
def total_pieces : ℕ := num_people * janes_pieces

theorem cinnamon_swirls_theorem :
  total_pieces = 12 :=
sorry

end cinnamon_swirls_theorem_l2502_250208


namespace village_population_calculation_l2502_250286

def initial_population : ℕ := 3161
def death_rate : ℚ := 5 / 100
def leaving_rate : ℚ := 15 / 100

theorem village_population_calculation :
  let remaining_after_deaths := initial_population - Int.floor (↑initial_population * death_rate)
  let final_population := remaining_after_deaths - Int.floor (↑remaining_after_deaths * leaving_rate)
  final_population = 2553 := by
  sorry

end village_population_calculation_l2502_250286


namespace grocery_store_soda_bottles_l2502_250226

/-- 
Given a grocery store with regular and diet soda bottles, this theorem proves 
the number of diet soda bottles, given the number of regular soda bottles and 
the difference between regular and diet soda bottles.
-/
theorem grocery_store_soda_bottles 
  (regular_soda : ℕ) 
  (difference : ℕ) 
  (h1 : regular_soda = 67)
  (h2 : regular_soda = difference + diet_soda) : 
  diet_soda = 9 := by
  sorry

end grocery_store_soda_bottles_l2502_250226


namespace y_value_proof_l2502_250205

theorem y_value_proof (y : ℝ) (h_pos : y > 0) 
  (h_eq : Real.sqrt (12 * y) * Real.sqrt (5 * y) * Real.sqrt (7 * y) * Real.sqrt (21 * y) = 21) : 
  y = 1 / Real.rpow 20 (1/4) :=
sorry

end y_value_proof_l2502_250205


namespace square_diff_fourth_power_l2502_250257

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_diff_fourth_power_l2502_250257


namespace toys_ratio_l2502_250207

def num_friends : ℕ := 4
def total_toys : ℕ := 118

theorem toys_ratio : 
  ∃ (toys_to_B : ℕ), 
    toys_to_B * num_friends = total_toys ∧ 
    (toys_to_B : ℚ) / total_toys = 1 / 4 := by
  sorry

end toys_ratio_l2502_250207


namespace polynomial_value_l2502_250296

/-- Given that ax³ + bx + 1 = 2023 when x = 1, prove that ax³ + bx - 2 = -2024 when x = -1 -/
theorem polynomial_value (a b : ℝ) : 
  (a * 1^3 + b * 1 + 1 = 2023) → (a * (-1)^3 + b * (-1) - 2 = -2024) := by
sorry

end polynomial_value_l2502_250296


namespace perpendicular_lines_a_value_l2502_250271

/-- Given two perpendicular lines (3a+2)x+(1-4a)y+8=0 and (5a-2)x+(a+4)y-7=0, prove that a = 0 or a = 12/11 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  ((3*a+2) * (5*a-2) + (1-4*a) * (a+4) = 0) → (a = 0 ∨ a = 12/11) := by
  sorry

end perpendicular_lines_a_value_l2502_250271


namespace arithmetic_computation_l2502_250221

theorem arithmetic_computation : -7 * 5 - (-4 * -2) + (-9 * -6) = 11 := by
  sorry

end arithmetic_computation_l2502_250221


namespace product_of_roots_l2502_250277

theorem product_of_roots (x : ℝ) : 
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r)) →
  (∃ p q r : ℝ, x^3 - 15*x^2 + 75*x - 36 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 36) :=
by sorry

end product_of_roots_l2502_250277


namespace quadratic_constant_term_l2502_250251

theorem quadratic_constant_term (m : ℝ) : 
  (∀ x, m * x^2 + 2 * x + m^2 - 1 = 0) → m = 1 ∨ m = -1 := by
  sorry

end quadratic_constant_term_l2502_250251


namespace square_sum_of_coefficients_l2502_250233

theorem square_sum_of_coefficients (a b c : ℝ) : 
  36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c)^2 →
  a^2 + b^2 + c^2 = 14 := by
sorry

end square_sum_of_coefficients_l2502_250233


namespace sqrt_product_equals_120_sqrt_3_l2502_250262

theorem sqrt_product_equals_120_sqrt_3 : 
  Real.sqrt 75 * Real.sqrt 48 * Real.sqrt 12 = 120 * Real.sqrt 3 := by
  sorry

end sqrt_product_equals_120_sqrt_3_l2502_250262


namespace andrew_grapes_purchase_l2502_250231

/-- The quantity of grapes (in kg) purchased by Andrew -/
def grapes_quantity : ℕ := sorry

/-- The cost of grapes per kg -/
def grapes_cost_per_kg : ℕ := 54

/-- The quantity of mangoes (in kg) purchased by Andrew -/
def mangoes_quantity : ℕ := 10

/-- The cost of mangoes per kg -/
def mangoes_cost_per_kg : ℕ := 62

/-- The total amount paid by Andrew -/
def total_paid : ℕ := 1376

theorem andrew_grapes_purchase :
  grapes_quantity * grapes_cost_per_kg + 
  mangoes_quantity * mangoes_cost_per_kg = total_paid ∧
  grapes_quantity = 14 := by sorry

end andrew_grapes_purchase_l2502_250231


namespace inverse_sum_reciprocal_l2502_250298

theorem inverse_sum_reciprocal (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x⁻¹ + y⁻¹)⁻¹ = (x * y) / (x + y) := by
  sorry

end inverse_sum_reciprocal_l2502_250298


namespace locus_C_is_ellipse_l2502_250293

/-- Circle O₁ with equation (x-1)² + y² = 1 -/
def circle_O₁ : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 1}

/-- Circle O₂ with equation (x+1)² + y² = 16 -/
def circle_O₂ : Set (ℝ × ℝ) :=
  {p | (p.1 + 1)^2 + p.2^2 = 16}

/-- The set of points P(x, y) that represent the center of circle C -/
def locus_C : Set (ℝ × ℝ) :=
  {p | ∃ r > 0,
    (∀ q ∈ circle_O₁, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r + 1)^2) ∧
    (∀ q ∈ circle_O₂, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (4 - r)^2)}

/-- Theorem stating that the locus of the center of circle C is an ellipse -/
theorem locus_C_is_ellipse : ∃ a b c d e f : ℝ,
  a > 0 ∧ b^2 < 4 * a * c ∧
  locus_C = {p | a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 + f = 0} :=
sorry

end locus_C_is_ellipse_l2502_250293


namespace min_sum_of_product_2010_l2502_250227

theorem min_sum_of_product_2010 :
  ∃ (min : ℕ), min = 78 ∧
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    a * b * c = 2010 →
    a + b + c ≥ min :=
by sorry

end min_sum_of_product_2010_l2502_250227


namespace necessary_but_not_sufficient_condition_l2502_250230

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (|x + 1| ≤ 4) → (-6 ≤ x ∧ x ≤ 3) ∧
  ∃ y : ℝ, -6 ≤ y ∧ y ≤ 3 ∧ |y + 1| > 4 :=
by sorry

end necessary_but_not_sufficient_condition_l2502_250230


namespace linear_function_domain_range_l2502_250255

def LinearFunction (k b : ℚ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_domain_range 
  (k b : ℚ) 
  (h_domain : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → LinearFunction k b x ∈ Set.Icc (-4 : ℝ) 1) 
  (h_range : Set.Icc (-4 : ℝ) 1 ⊆ Set.range (LinearFunction k b)) :
  (k = 5/6 ∧ b = -3/2) ∨ (k = -5/6 ∧ b = -3/2) :=
sorry

end linear_function_domain_range_l2502_250255


namespace example_theorem_l2502_250212

-- Define the necessary types and structures

-- State the theorem
theorem example_theorem (hypothesis1 : Type) (hypothesis2 : Type) : conclusion_type :=
  -- The proof would go here, but we're using sorry as requested
  sorry

-- Additional definitions or lemmas if needed

end example_theorem_l2502_250212


namespace max_product_sum_1998_l2502_250228

theorem max_product_sum_1998 :
  ∀ x y : ℤ, x + y = 1998 → x * y ≤ 998001 :=
by
  sorry

end max_product_sum_1998_l2502_250228


namespace calculate_interest_rate_l2502_250290

/-- Given a total sum and two parts with specific interest conditions, 
    calculate the interest rate for the second part. -/
theorem calculate_interest_rate 
  (total_sum : ℚ) 
  (second_part : ℚ) 
  (first_part_years : ℚ) 
  (first_part_rate : ℚ) 
  (second_part_years : ℚ) 
  (h1 : total_sum = 2730) 
  (h2 : second_part = 1680) 
  (h3 : first_part_years = 8) 
  (h4 : first_part_rate = 3 / 100) 
  (h5 : second_part_years = 3) 
  (h6 : (total_sum - second_part) * first_part_rate * first_part_years = 
        second_part * (second_part_years * x) ) :
  x = 5 / 100 := by
  sorry

end calculate_interest_rate_l2502_250290


namespace smallest_five_digit_multiple_of_3_and_4_l2502_250280

theorem smallest_five_digit_multiple_of_3_and_4 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  3 ∣ n ∧ 
  4 ∣ n ∧ 
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) → 3 ∣ m → 4 ∣ m → m ≥ n) ∧
  n = 10008 :=
sorry

end smallest_five_digit_multiple_of_3_and_4_l2502_250280


namespace chinese_character_number_puzzle_l2502_250202

theorem chinese_character_number_puzzle :
  ∃! (A B C D : Nat),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    A * 10 + B = 19 ∧
    C * 10 + D = 62 ∧
    (A * 1000 + B * 100 + C * 10 + D) - (A * 1000 + A * 100 + B * 10 + B) = 124 := by
  sorry

end chinese_character_number_puzzle_l2502_250202


namespace coin_black_region_probability_l2502_250235

/-- The probability of a coin partially covering a black region on a specially painted square. -/
theorem coin_black_region_probability : 
  let square_side : ℝ := 10
  let triangle_leg : ℝ := 3
  let diamond_side : ℝ := 3 * Real.sqrt 2
  let coin_diameter : ℝ := 2
  let valid_region_side : ℝ := square_side - coin_diameter
  let valid_region_area : ℝ := valid_region_side ^ 2
  let triangle_area : ℝ := 1/2 * triangle_leg ^ 2
  let diamond_area : ℝ := diamond_side ^ 2
  let overlap_area : ℝ := 48 + 4 * Real.sqrt 2 + 2 * Real.pi
  overlap_area / valid_region_area = (48 + 4 * Real.sqrt 2 + 2 * Real.pi) / 64 := by
  sorry

end coin_black_region_probability_l2502_250235


namespace equation_system_solutions_l2502_250294

/-- The system of equations has two types of solutions:
    1. (3, 5, 7, 9)
    2. (t, -t, t, -t) for any real t -/
theorem equation_system_solutions :
  ∀ (a b c d : ℝ),
    (a * b + a * c = 3 * b + 3 * c) ∧
    (b * c + b * d = 5 * c + 5 * d) ∧
    (a * c + c * d = 7 * a + 7 * d) ∧
    (a * d + b * d = 9 * a + 9 * b) →
    ((a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨
     (∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t)) :=
by sorry

end equation_system_solutions_l2502_250294


namespace milk_added_to_full_can_l2502_250224

/-- Represents the contents of a can with milk and water -/
structure Can where
  milk : ℝ
  water : ℝ

/-- Represents the ratios of milk to water -/
structure Ratio where
  milk : ℝ
  water : ℝ

def Can.ratio (can : Can) : Ratio :=
  { milk := can.milk, water := can.water }

def Can.total (can : Can) : ℝ :=
  can.milk + can.water

theorem milk_added_to_full_can 
  (initial_ratio : Ratio) 
  (final_ratio : Ratio) 
  (capacity : ℝ) :
  initial_ratio.milk / initial_ratio.water = 4 / 3 →
  final_ratio.milk / final_ratio.water = 2 / 1 →
  capacity = 36 →
  ∃ (initial_can final_can : Can),
    initial_can.ratio = initial_ratio ∧
    final_can.ratio = final_ratio ∧
    final_can.total = capacity ∧
    final_can.water = initial_can.water ∧
    final_can.milk - initial_can.milk = 72 / 7 := by
  sorry

end milk_added_to_full_can_l2502_250224


namespace angle_bisector_theorem_bisector_proportion_l2502_250225

/-- Represents a triangle with side lengths and an angle bisector -/
structure BisectedTriangle where
  -- Side lengths
  p : ℝ
  q : ℝ
  r : ℝ
  -- Length of angle bisector segments
  u : ℝ
  v : ℝ
  -- Conditions
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r
  triangle_ineq : p < q + r ∧ q < p + r ∧ r < p + q
  bisector_sum : u + v = p

/-- The angle bisector theorem holds for this triangle -/
theorem angle_bisector_theorem (t : BisectedTriangle) : t.u / t.q = t.v / t.r := sorry

/-- The main theorem: proving the proportion involving v and r -/
theorem bisector_proportion (t : BisectedTriangle) : t.v / t.r = t.p / (t.q + t.r) := by
  sorry

end angle_bisector_theorem_bisector_proportion_l2502_250225


namespace intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l2502_250245

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 12 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 3*a + 2}

-- Part 1
theorem intersection_A_complement_B_when_a_is_1 :
  A ∩ (Set.univ \ B 1) = {x | -2 ≤ x ∧ x ≤ 0} ∪ {x | 5 ≤ x ∧ x ≤ 6} := by sorry

-- Part 2
theorem A_intersect_B_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ∈ Set.Iic (-3/2) ∪ Set.Icc (-1) (4/3) := by sorry


end intersection_A_complement_B_when_a_is_1_A_intersect_B_equals_B_l2502_250245


namespace train_speed_calculation_l2502_250246

-- Define the length of the train in meters
def train_length : ℝ := 310

-- Define the length of the platform in meters
def platform_length : ℝ := 210

-- Define the time taken to cross the platform in seconds
def crossing_time : ℝ := 26

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_calculation :
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / crossing_time
  let speed_kmhr := speed_ms * conversion_factor
  speed_kmhr = 72 := by sorry

end train_speed_calculation_l2502_250246


namespace ios_department_larger_l2502_250297

/-- Represents the number of developers in the Android department -/
def android_devs : ℕ := sorry

/-- Represents the number of developers in the iOS department -/
def ios_devs : ℕ := sorry

/-- The total number of messages sent equals the total number of messages received -/
axiom message_balance : 7 * android_devs + 15 * ios_devs = 15 * android_devs + 9 * ios_devs

theorem ios_department_larger : ios_devs > android_devs := by
  sorry

end ios_department_larger_l2502_250297


namespace sector_perimeter_l2502_250254

/-- The perimeter of a circular sector with a central angle of 180 degrees and a radius of 28.000000000000004 cm is 143.96459430079216 cm. -/
theorem sector_perimeter : 
  let r : ℝ := 28.000000000000004
  let θ : ℝ := 180
  let arc_length : ℝ := (θ / 360) * 2 * Real.pi * r
  let perimeter : ℝ := arc_length + 2 * r
  perimeter = 143.96459430079216 := by sorry

end sector_perimeter_l2502_250254


namespace common_root_quadratic_l2502_250234

theorem common_root_quadratic (a b : ℝ) : 
  (∃! t : ℝ, t^2 + a*t + b = 0 ∧ t^2 + b*t + a = 0) → 
  (a + b + 1 = 0 ∧ a ≠ b) :=
by sorry

end common_root_quadratic_l2502_250234


namespace max_value_implies_a_l2502_250248

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -9 * x^2 - 6 * a * x + 2 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x ≤ -3) ∧
  (∃ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f a x = -3) →
  a = Real.sqrt 6 + 2 :=
by sorry

end max_value_implies_a_l2502_250248


namespace sqrt_sum_inequality_l2502_250209

theorem sqrt_sum_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq : a + b = c + d) (ineq : a < c ∧ c ≤ d ∧ d < b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt c + Real.sqrt d := by
  sorry

end sqrt_sum_inequality_l2502_250209


namespace kelly_games_l2502_250210

theorem kelly_games (games_given_away : ℕ) (games_left : ℕ) : games_given_away = 91 → games_left = 92 → games_given_away + games_left = 183 :=
by
  sorry

end kelly_games_l2502_250210


namespace circle_diameter_problem_l2502_250288

/-- Given two circles A and C inside a larger circle B, prove the diameter of A -/
theorem circle_diameter_problem (R B r : ℝ) : 
  R = 10 → -- Radius of circle B is 10 cm (half the diameter of 20 cm)
  100 * Real.pi - 2 * Real.pi * r^2 = 5 * (Real.pi * r^2) → -- Ratio of shaded area to area of A is 5:1
  (2 * r : ℝ) = 2 * Real.sqrt (100 / 7) := by
  sorry

#check circle_diameter_problem

end circle_diameter_problem_l2502_250288


namespace circle_radius_from_arc_and_angle_l2502_250237

/-- Given an arc length of 4 and a central angle of 2 radians, the radius of the circle is 2. -/
theorem circle_radius_from_arc_and_angle (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) 
    (h1 : arc_length = 4)
    (h2 : central_angle = 2)
    (h3 : arc_length = radius * central_angle) : 
  radius = 2 := by
  sorry

end circle_radius_from_arc_and_angle_l2502_250237


namespace halfway_between_one_eighth_and_one_third_l2502_250267

theorem halfway_between_one_eighth_and_one_third : 
  (1 / 8 : ℚ) + ((1 / 3 : ℚ) - (1 / 8 : ℚ)) / 2 = 11 / 48 := by
  sorry

end halfway_between_one_eighth_and_one_third_l2502_250267


namespace coneSurface_is_cone_l2502_250236

/-- A surface in spherical coordinates (ρ, θ, φ) defined by ρ = c sin φ, where c is a positive constant -/
def coneSurface (c : ℝ) (h : c > 0) (ρ θ φ : ℝ) : Prop :=
  ρ = c * Real.sin φ

/-- The shape described by the coneSurface is a cone -/
theorem coneSurface_is_cone (c : ℝ) (h : c > 0) :
  ∃ (cone : Set (ℝ × ℝ × ℝ)), ∀ (ρ θ φ : ℝ),
    coneSurface c h ρ θ φ ↔ (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) ∈ cone :=
sorry

end coneSurface_is_cone_l2502_250236


namespace intersection_A_B_union_complement_B_A_l2502_250203

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define the universal set R (real numbers)
def R : Set ℝ := univ

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 2 < x ∧ x < 6} := by sorry

-- Theorem for the union of complement of B and A
theorem union_complement_B_A : (R \ B) ∪ A = {x | x < 6 ∨ 9 ≤ x} := by sorry

end intersection_A_B_union_complement_B_A_l2502_250203


namespace intersection_complement_M_and_N_l2502_250200

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

theorem intersection_complement_M_and_N : 
  (Set.univ \ M) ∩ N = Set.Ioo 0 1 := by sorry

end intersection_complement_M_and_N_l2502_250200


namespace order_of_abc_l2502_250279

theorem order_of_abc : 
  let a : ℝ := 2017^0
  let b : ℝ := 2015 * 2017 - 2016^2
  let c : ℝ := (-2/3)^2016 * (3/2)^2017
  b < a ∧ a < c := by sorry

end order_of_abc_l2502_250279


namespace ryan_learning_time_l2502_250206

/-- Represents the time Ryan spends on learning languages in hours -/
structure LearningTime where
  total : ℝ
  english : ℝ
  chinese : ℝ

/-- Theorem: Given Ryan's total learning time and English learning time, 
    prove that his Chinese learning time is the difference -/
theorem ryan_learning_time (rt : LearningTime) 
  (h1 : rt.total = 3) 
  (h2 : rt.english = 2) 
  (h3 : rt.total = rt.english + rt.chinese) : 
  rt.chinese = 1 := by
sorry

end ryan_learning_time_l2502_250206


namespace double_base_cost_increase_l2502_250219

/-- The cost function for a given base value -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem stating that doubling the base value results in a cost that is 1600% of the original -/
theorem double_base_cost_increase (t : ℝ) (b : ℝ) :
  cost t (2 * b) = 16 * cost t b :=
by sorry

end double_base_cost_increase_l2502_250219


namespace students_count_l2502_250268

/-- The total number of students in an arrangement of rows -/
def totalStudents (rows : ℕ) (studentsPerRow : ℕ) (lastRowStudents : ℕ) : ℕ :=
  (rows - 1) * studentsPerRow + lastRowStudents

/-- Theorem: Given 8 rows of students, where 7 rows have 6 students each 
    and the last row has 5 students, the total number of students is 47. -/
theorem students_count : totalStudents 8 6 5 = 47 := by
  sorry

end students_count_l2502_250268


namespace smallest_n_for_integer_T_l2502_250214

def K' : ℚ := 1/1 + 1/2 + 1/3 + 1/4 + 1/5

def T (n : ℕ) : ℚ := n * (5^(n-1)) * K'

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_T :
  ∀ n : ℕ, n > 0 → (is_integer (T n) ↔ n ≥ 24) ∧
  ∀ m : ℕ, m < 24 → ¬ is_integer (T m) :=
sorry

end smallest_n_for_integer_T_l2502_250214


namespace complex_number_in_third_quadrant_l2502_250292

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - Complex.I)^2 / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end complex_number_in_third_quadrant_l2502_250292


namespace amusing_numbers_l2502_250260

def is_amusing (x : Nat) : Prop :=
  (1000 ≤ x ∧ x < 10000) ∧
  ∃ y : Nat, (1000 ≤ y ∧ y < 10000) ∧
  (y % x = 0) ∧
  (∀ i : Fin 4,
    let x_digit := (x / (10 ^ i.val)) % 10
    let y_digit := (y / (10 ^ i.val)) % 10
    (x_digit = 0 ∧ y_digit = 1) ∨
    (x_digit = 9 ∧ y_digit = 8) ∨
    (x_digit ≠ 0 ∧ x_digit ≠ 9 ∧ (y_digit = x_digit - 1 ∨ y_digit = x_digit + 1)))

theorem amusing_numbers :
  is_amusing 1111 ∧ is_amusing 1091 ∧ is_amusing 1109 ∧ is_amusing 1089 :=
sorry

end amusing_numbers_l2502_250260


namespace darnel_sprint_distance_l2502_250274

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.13 →
  jogged_distance + additional_sprint = 0.88 := by
  sorry

end darnel_sprint_distance_l2502_250274


namespace pine_cones_on_roof_l2502_250284

/-- Calculates the weight of pine cones on a roof given the number of trees, 
    pine cones per tree, percentage on roof, and weight per pine cone. -/
theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (cones_per_tree : ℕ) 
  (percent_on_roof : ℚ) 
  (weight_per_cone : ℕ) 
  (h1 : num_trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : percent_on_roof = 30 / 100)
  (h4 : weight_per_cone = 4) :
  (num_trees * cones_per_tree : ℚ) * percent_on_roof * weight_per_cone = 1920 :=
sorry

end pine_cones_on_roof_l2502_250284
