import Mathlib

namespace NUMINAMATH_CALUDE_equation_solutions_l4113_411364

theorem equation_solutions : 
  let solutions : List ℂ := [
    4 + Complex.I * Real.sqrt 6,
    4 - Complex.I * Real.sqrt 6,
    4 + Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 + Real.sqrt 433),
    4 + Complex.I * Real.sqrt (21 - Real.sqrt 433),
    4 - Complex.I * Real.sqrt (21 - Real.sqrt 433)
  ]
  ∀ x ∈ solutions, (x - 2)^6 + (x - 6)^6 = 32 ∧
  ∀ x : ℂ, (x - 2)^6 + (x - 6)^6 = 32 → x ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4113_411364


namespace NUMINAMATH_CALUDE_sin_cos_equation_solvability_l4113_411334

theorem sin_cos_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solvability_l4113_411334


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_simplification_l4113_411397

theorem sqrt_sum_quotient_simplification :
  (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_simplification_l4113_411397


namespace NUMINAMATH_CALUDE_quadratic_root_existence_l4113_411391

/-- Given a quadratic function f(x) = x^2 + x + m where m is positive,
    if f(t) < 0 for some real t, then f has a root in the interval (t, t+1) -/
theorem quadratic_root_existence (m : ℝ) (t : ℝ) (h_m : m > 0) :
  let f : ℝ → ℝ := λ x ↦ x^2 + x + m
  f t < 0 → ∃ x : ℝ, t < x ∧ x < t + 1 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_existence_l4113_411391


namespace NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_C_l4113_411378

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem triangle_side_a (t : Triangle) (hb : t.b = 2) (hB : t.B = π/6) (hC : t.C = 3*π/4) :
  t.a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem triangle_angle_C (t : Triangle) (hS : t.a * t.b * Real.sin t.C / 2 = (t.a^2 + t.b^2 - t.c^2) / 4) :
  t.C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_C_l4113_411378


namespace NUMINAMATH_CALUDE_gloria_pine_trees_l4113_411300

def cabin_price : ℕ := 129000
def initial_cash : ℕ := 150
def cypress_trees : ℕ := 20
def maple_trees : ℕ := 24
def cypress_price : ℕ := 100
def maple_price : ℕ := 300
def pine_price : ℕ := 200
def leftover_cash : ℕ := 350

def pine_trees : ℕ := 600

theorem gloria_pine_trees :
  ∃ (total_raised : ℕ),
    total_raised = cabin_price + leftover_cash ∧
    total_raised = initial_cash + cypress_trees * cypress_price + maple_trees * maple_price + pine_trees * pine_price :=
by sorry

end NUMINAMATH_CALUDE_gloria_pine_trees_l4113_411300


namespace NUMINAMATH_CALUDE_smallest_constant_for_ratio_difference_l4113_411344

theorem smallest_constant_for_ratio_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∃ (i j k l : Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a₁ / a₂ - a₃ / a₄| ≤ (1/2 : ℝ)) ∧
  (∀ C < (1/2 : ℝ), ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    ∀ (i j k l : Fin 5), i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
      |b₁ / b₂ - b₃ / b₄| > C) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_for_ratio_difference_l4113_411344


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l4113_411372

-- Define the quadratic polynomial p(x)
def p (x : ℚ) : ℚ := (12/7) * x^2 + (36/7) * x - 216/7

-- Theorem stating that p(x) satisfies the given conditions
theorem quadratic_polynomial_satisfies_conditions :
  p (-6) = 0 ∧ p 3 = 0 ∧ p 1 = -24 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l4113_411372


namespace NUMINAMATH_CALUDE_coefficient_of_x_fourth_l4113_411351

theorem coefficient_of_x_fourth (x : ℝ) : 
  let expr := 5*(x^4 - 2*x^5) + 3*(x^2 - 3*x^4 + 2*x^6) - (2*x^5 - 3*x^4)
  ∃ (a b c d e f : ℝ), expr = a*x^6 + b*x^5 + (-1)*x^4 + d*x^3 + e*x^2 + f*x + c :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_fourth_l4113_411351


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_size_l4113_411347

theorem choir_arrangement (n : ℕ) : (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
sorry

theorem min_choir_size : ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_size_l4113_411347


namespace NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l4113_411358

theorem quadratic_vertex_ordinate (b c : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁^2 + b*x₁ + c = 2017) ∧ (x₂^2 + b*x₂ + c = 2017)) →
  (-(b^2 - 4*c) / 4 = -1016064) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_ordinate_l4113_411358


namespace NUMINAMATH_CALUDE_y_coord_at_neg_three_l4113_411317

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  max_value : ℝ
  max_x : ℝ
  point_zero : ℝ
  has_max : max_value = 7
  max_at : max_x = -2
  passes_zero : a * 0^2 + b * 0 + c = point_zero
  passes_zero_value : point_zero = -15

/-- The y-coordinate of a point on the quadratic function -/
def y_coord (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- The theorem stating the y-coordinate at x = -3 is 1.5 -/
theorem y_coord_at_neg_three (f : QuadraticFunction) : y_coord f (-3) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_y_coord_at_neg_three_l4113_411317


namespace NUMINAMATH_CALUDE_similar_triangles_area_l4113_411333

-- Define the triangles and their properties
def Triangle : Type := Unit

def similar (t1 t2 : Triangle) : Prop := sorry

def similarityRatio (t1 t2 : Triangle) : ℚ := sorry

def area (t : Triangle) : ℝ := sorry

-- State the theorem
theorem similar_triangles_area 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) 
  (h_area_ABC : area ABC = 3) : 
  area DEF = 12 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_area_l4113_411333


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l4113_411330

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  ∃ (min : ℝ), min = 18/7 ∧ x^2 + y^2 + z^2 ≥ min ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = 6 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l4113_411330


namespace NUMINAMATH_CALUDE_square_side_length_l4113_411388

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 1 / 9 ∧ area = side ^ 2 → side = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4113_411388


namespace NUMINAMATH_CALUDE_circle_area_polar_l4113_411360

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is 25π/4 -/
theorem circle_area_polar (θ : ℝ) (r : ℝ → ℝ) : 
  (r θ = 4 * Real.cos θ - 3 * Real.sin θ) → 
  (∃ c : ℝ × ℝ, ∃ radius : ℝ, 
    (∀ x y : ℝ, (x - c.1)^2 + (y - c.2)^2 = radius^2 ↔ 
      ∃ θ : ℝ, x = r θ * Real.cos θ ∧ y = r θ * Real.sin θ) ∧
    π * radius^2 = 25 * π / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_area_polar_l4113_411360


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4113_411350

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
   x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) → 
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4113_411350


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_three_plus_sqrt_three_l4113_411349

theorem trigonometric_sum_equals_three_plus_sqrt_three :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  3 * tan_30 + 6 * sin_30 = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_three_plus_sqrt_three_l4113_411349


namespace NUMINAMATH_CALUDE_system_solution_l4113_411362

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) :
  x^2 + y^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l4113_411362


namespace NUMINAMATH_CALUDE_grid_solution_l4113_411382

/-- Represents a 3x3 grid of integers -/
structure Grid :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : Int)

/-- Checks if the middle number in each row is the sum of the numbers at its ends -/
def rowSumsValid (g : Grid) : Prop :=
  g.a12 = g.a11 + g.a13 ∧ g.a22 = g.a21 + g.a23 ∧ g.a32 = g.a31 + g.a33

/-- Checks if the sums of the numbers on both diagonals are equal -/
def diagonalSumsEqual (g : Grid) : Prop :=
  g.a11 + g.a22 + g.a33 = g.a13 + g.a22 + g.a31

/-- The theorem stating the solution to the grid problem -/
theorem grid_solution :
  ∀ (g : Grid),
    g.a11 = 4 ∧ g.a12 = 12 ∧ g.a13 = 8 ∧ g.a21 = 10 →
    rowSumsValid g →
    diagonalSumsEqual g →
    g.a22 = 3 ∧ g.a23 = 9 ∧ g.a31 = -3 ∧ g.a32 = -2 ∧ g.a33 = 1 :=
by sorry

end NUMINAMATH_CALUDE_grid_solution_l4113_411382


namespace NUMINAMATH_CALUDE_thirty_switch_network_connections_l4113_411361

/-- Represents a network of switches with their connections. -/
structure SwitchNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  no_multiple_connections : Bool

/-- Calculates the total number of connections in the network. -/
def total_connections (network : SwitchNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2

/-- Theorem stating that a network of 30 switches, each connected to 4 others,
    has 60 total connections. -/
theorem thirty_switch_network_connections :
  let network := SwitchNetwork.mk 30 4 true
  total_connections network = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_switch_network_connections_l4113_411361


namespace NUMINAMATH_CALUDE_smallest_n_for_20_colors_l4113_411319

/-- Represents a ball with a color -/
structure Ball :=
  (color : Nat)

/-- Represents a circular arrangement of balls -/
def CircularArrangement := List Ball

/-- Checks if a sequence of balls has at least k different colors -/
def hasAtLeastKColors (sequence : List Ball) (k : Nat) : Prop :=
  (sequence.map Ball.color).toFinset.card ≥ k

theorem smallest_n_for_20_colors 
  (total_balls : Nat) 
  (num_colors : Nat) 
  (balls_per_color : Nat) 
  (h1 : total_balls = 1000) 
  (h2 : num_colors = 40) 
  (h3 : balls_per_color = 25) 
  (h4 : total_balls = num_colors * balls_per_color) :
  ∃ (n : Nat), 
    (∀ (arrangement : CircularArrangement), 
      arrangement.length = total_balls → 
      ∃ (subsequence : List Ball), 
        subsequence.length = n ∧ 
        hasAtLeastKColors subsequence 20) ∧
    (∀ (m : Nat), m < n → 
      ∃ (arrangement : CircularArrangement), 
        arrangement.length = total_balls ∧ 
        ∀ (subsequence : List Ball), 
          subsequence.length = m → 
          ¬(hasAtLeastKColors subsequence 20)) ∧
    n = 352 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_20_colors_l4113_411319


namespace NUMINAMATH_CALUDE_find_a_value_l4113_411339

/-- Given sets A and B, prove that a is either -2/3 or -7/4 -/
theorem find_a_value (x : ℝ) (a : ℝ) : 
  let A : Set ℝ := {1, 2, x^2 - 5*x + 9}
  let B : Set ℝ := {3, x^2 + a*x + a}
  A = {1, 2, 3} → 2 ∈ B → (a = -2/3 ∨ a = -7/4) := by
sorry

end NUMINAMATH_CALUDE_find_a_value_l4113_411339


namespace NUMINAMATH_CALUDE_calculate_expression_l4113_411390

theorem calculate_expression : 3000 * (3000 ^ 2999) * 2 ^ 3000 = 3000 ^ 3000 * 2 ^ 3000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4113_411390


namespace NUMINAMATH_CALUDE_bumper_car_line_count_l4113_411370

/-- Calculates the final number of people in line for bumper cars after several changes --/
def final_line_count (initial : ℕ) (left1 left2 left3 joined1 joined2 joined3 : ℕ) : ℕ :=
  initial - left1 + joined1 - left2 + joined2 - left3 + joined3

/-- Theorem stating the final number of people in line for the given scenario --/
theorem bumper_car_line_count : 
  final_line_count 31 15 8 7 12 18 25 = 56 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_count_l4113_411370


namespace NUMINAMATH_CALUDE_all_radii_equal_l4113_411316

/-- A circle with radius 2 cm -/
structure Circle :=
  (radius : ℝ)
  (h : radius = 2)

/-- Any radius of the circle is 2 cm -/
theorem all_radii_equal (c : Circle) (r : ℝ) (h : r = c.radius) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_all_radii_equal_l4113_411316


namespace NUMINAMATH_CALUDE_custom_op_result_l4113_411368

/-- Custom operation ※ -/
def custom_op (a b m n : ℕ) : ℕ := (a^b)^m + (b^a)^n

theorem custom_op_result : ∃ (m n : ℕ), 
  (custom_op 1 4 m n = 10) ∧ 
  (custom_op 2 2 m n = 15) ∧ 
  (4^(2*m + n - 1) = 81) := by sorry

end NUMINAMATH_CALUDE_custom_op_result_l4113_411368


namespace NUMINAMATH_CALUDE_intersection_P_Q_l4113_411329

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | ∃ y, Real.log (2 - x) = y}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l4113_411329


namespace NUMINAMATH_CALUDE_yoongi_flowers_l4113_411318

def flowers_problem (initial : ℕ) (to_eunji : ℕ) (to_yuna : ℕ) : Prop :=
  initial - (to_eunji + to_yuna) = 12

theorem yoongi_flowers : flowers_problem 28 7 9 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_flowers_l4113_411318


namespace NUMINAMATH_CALUDE_rice_dumpling_costs_l4113_411313

theorem rice_dumpling_costs (total_cost_honey : ℝ) (total_cost_date : ℝ) 
  (cost_diff : ℝ) (h1 : total_cost_honey = 1300) (h2 : total_cost_date = 1000) 
  (h3 : cost_diff = 0.6) :
  ∃ (cost_date cost_honey : ℝ),
    cost_date = 2 ∧ 
    cost_honey = 2.6 ∧
    cost_honey = cost_date + cost_diff ∧
    total_cost_honey / cost_honey = total_cost_date / cost_date :=
by
  sorry

end NUMINAMATH_CALUDE_rice_dumpling_costs_l4113_411313


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l4113_411363

/-- Represents the distances walked by Spencer -/
structure WalkDistances where
  total : ℝ
  libraryToPostOffice : ℝ
  postOfficeToHome : ℝ

/-- Calculates the distance from house to library -/
def distanceHouseToLibrary (w : WalkDistances) : ℝ :=
  w.total - w.libraryToPostOffice - w.postOfficeToHome

/-- Theorem stating that the distance from house to library is 0.3 miles -/
theorem spencer_walk_distance (w : WalkDistances) 
  (h_total : w.total = 0.8)
  (h_lib_post : w.libraryToPostOffice = 0.1)
  (h_post_home : w.postOfficeToHome = 0.4) : 
  distanceHouseToLibrary w = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l4113_411363


namespace NUMINAMATH_CALUDE_bee_puzzle_l4113_411394

theorem bee_puzzle (B : ℕ) 
  (h1 : B > 0)
  (h2 : B % 5 = 0)
  (h3 : B % 3 = 0)
  (h4 : B = B / 5 + B / 3 + 3 * (B / 3 - B / 5) + 1) :
  B = 15 := by
sorry

end NUMINAMATH_CALUDE_bee_puzzle_l4113_411394


namespace NUMINAMATH_CALUDE_temperature_at_4km_l4113_411304

/-- Calculates the temperature at a given altitude based on the ground temperature and temperature drop rate. -/
def temperature_at_altitude (ground_temp : ℝ) (temp_drop_rate : ℝ) (altitude : ℝ) : ℝ :=
  ground_temp - temp_drop_rate * altitude

/-- Proves that the temperature at an altitude of 4 kilometers is -5°C given the specified conditions. -/
theorem temperature_at_4km (ground_temp : ℝ) (temp_drop_rate : ℝ) 
  (h1 : ground_temp = 15)
  (h2 : temp_drop_rate = 5) : 
  temperature_at_altitude ground_temp temp_drop_rate 4 = -5 := by
  sorry

#eval temperature_at_altitude 15 5 4

end NUMINAMATH_CALUDE_temperature_at_4km_l4113_411304


namespace NUMINAMATH_CALUDE_glucose_solution_volume_l4113_411395

/-- Given a glucose solution where 45 cubic centimeters contain 6.75 grams of glucose,
    prove that the volume containing 15 grams of glucose is 100 cubic centimeters. -/
theorem glucose_solution_volume (volume : ℝ) (glucose_mass : ℝ) :
  (45 : ℝ) / volume = 6.75 / glucose_mass →
  glucose_mass = 15 →
  volume = 100 := by
  sorry

end NUMINAMATH_CALUDE_glucose_solution_volume_l4113_411395


namespace NUMINAMATH_CALUDE_probability_one_from_each_name_l4113_411377

def total_cards : ℕ := 12
def alice_letters : ℕ := 5
def bob_letters : ℕ := 7

theorem probability_one_from_each_name :
  let prob_alice_then_bob := (alice_letters : ℚ) / total_cards * bob_letters / (total_cards - 1)
  let prob_bob_then_alice := (bob_letters : ℚ) / total_cards * alice_letters / (total_cards - 1)
  prob_alice_then_bob + prob_bob_then_alice = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_from_each_name_l4113_411377


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4113_411367

/-- An isosceles triangle with perimeter 20 and leg length 7 has a base length of 6 -/
theorem isosceles_triangle_base_length : ∀ (base leg : ℝ),
  leg = 7 → base + 2 * leg = 20 → base = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4113_411367


namespace NUMINAMATH_CALUDE_expression_simplification_l4113_411303

theorem expression_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - b^2 - a^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) =
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l4113_411303


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4113_411353

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x - 2) ↔ q ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4113_411353


namespace NUMINAMATH_CALUDE_a_value_l4113_411348

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {3^a, b}

theorem a_value (a b : ℝ) :
  A a ∪ B a b = {-1, 0, 1} → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l4113_411348


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4113_411326

/-- An arithmetic sequence with non-zero terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≠ 0) ∧
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_condition : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0) :
  a 7 = 4 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4113_411326


namespace NUMINAMATH_CALUDE_equation_solution_range_l4113_411325

theorem equation_solution_range (a : ℝ) : 
  ∃ x : ℝ, 
    ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ∧ 
    ((2 - a) * x - 3 > 0) → 
    a < -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l4113_411325


namespace NUMINAMATH_CALUDE_diagonal_length_specific_hexagon_l4113_411381

/-- A regular hexagon inscribed in a circle with alternating side lengths --/
structure AlternatingHexagon where
  -- The length of the shorter sides
  short_side : ℝ
  -- The length of the longer sides
  long_side : ℝ
  -- Assumption that the hexagon is regular and inscribed in a circle
  is_regular_inscribed : True

/-- The length of a diagonal in an alternating hexagon --/
def diagonal_length (h : AlternatingHexagon) : ℝ :=
  sorry

/-- Theorem: The diagonal length of a specific alternating hexagon is 7√3 --/
theorem diagonal_length_specific_hexagon :
  let h : AlternatingHexagon := ⟨5, 7, trivial⟩
  diagonal_length h = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_length_specific_hexagon_l4113_411381


namespace NUMINAMATH_CALUDE_drug_storage_temperature_range_l4113_411327

/-- Given a drug with a storage temperature of 20 ± 2 (°C), 
    the difference between the highest and lowest suitable storage temperatures is 4°C -/
theorem drug_storage_temperature_range : 
  let recommended_temp : ℝ := 20
  let tolerance : ℝ := 2
  let highest_temp := recommended_temp + tolerance
  let lowest_temp := recommended_temp - tolerance
  highest_temp - lowest_temp = 4 := by
  sorry

end NUMINAMATH_CALUDE_drug_storage_temperature_range_l4113_411327


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_less_than_neg_thirteen_half_l4113_411392

theorem proposition_false_iff_a_less_than_neg_thirteen_half :
  (∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) = false ↔ a < -13/2 :=
sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_less_than_neg_thirteen_half_l4113_411392


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l4113_411357

theorem quadratic_coefficient (b : ℝ) : 
  ((-14 : ℝ)^2 + b * (-14) + 49 = 0) → b = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l4113_411357


namespace NUMINAMATH_CALUDE_water_purification_minimum_processes_l4113_411359

theorem water_purification_minimum_processes : ∃ n : ℕ,
  (∀ m : ℕ, m < n → (0.8 ^ m : ℝ) ≥ 0.05) ∧
  (0.8 ^ n : ℝ) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_water_purification_minimum_processes_l4113_411359


namespace NUMINAMATH_CALUDE_beach_towel_laundry_loads_l4113_411312

theorem beach_towel_laundry_loads 
  (num_families : ℕ) 
  (people_per_family : ℕ) 
  (vacation_days : ℕ) 
  (towels_per_person_per_day : ℕ) 
  (towels_per_load : ℕ) 
  (h1 : num_families = 3) 
  (h2 : people_per_family = 4) 
  (h3 : vacation_days = 7) 
  (h4 : towels_per_person_per_day = 1) 
  (h5 : towels_per_load = 14) : 
  (num_families * people_per_family * vacation_days * towels_per_person_per_day + towels_per_load - 1) / towels_per_load = 6 := by
  sorry

end NUMINAMATH_CALUDE_beach_towel_laundry_loads_l4113_411312


namespace NUMINAMATH_CALUDE_factorial_division_l4113_411336

theorem factorial_division : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l4113_411336


namespace NUMINAMATH_CALUDE_max_a_value_l4113_411384

theorem max_a_value (a : ℝ) : 
  a > 0 →
  (∀ x ∈ Set.Icc 1 2, ∃ y, y = x - a / x) →
  (∀ M : ℝ × ℝ, M.1 ∈ Set.Icc 1 2 → M.2 = M.1 - a / M.1 → 
    ∀ N : ℝ × ℝ, N.1 = M.1 ∧ 
    N.2 = (1 + a / 2) * (M.1 - 1) + (1 - a) → 
    (M.2 - N.2)^2 ≤ 1) →
  a ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l4113_411384


namespace NUMINAMATH_CALUDE_equation_solution_l4113_411314

theorem equation_solution : 
  ∃ y : ℚ, (7 * y / (y + 5) - 4 / (y + 5) = 2 / (y + 5) + 1 / 2) ∧ y = 17 / 13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4113_411314


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l4113_411375

/-- Given that the coefficient of x^3y^3 in the expansion of (x+ay)^6 is -160, prove that a = -2 -/
theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 = -160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l4113_411375


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l4113_411374

def p (x : ℝ) : ℝ := 8 * x^4 + 26 * x^3 - 65 * x^2 + 24 * x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l4113_411374


namespace NUMINAMATH_CALUDE_b_contribution_is_90000_l4113_411354

/-- Represents the business partnership between A and B --/
structure Partnership where
  a_investment : ℕ  -- A's initial investment
  b_join_time : ℕ  -- Time when B joins (in months)
  total_time : ℕ   -- Total investment period (in months)
  profit_ratio_a : ℕ  -- A's part in profit ratio
  profit_ratio_b : ℕ  -- B's part in profit ratio

/-- Calculates B's contribution given the partnership details --/
def calculate_b_contribution (p : Partnership) : ℕ :=
  -- Placeholder for the actual calculation
  0

/-- Theorem stating that B's contribution is 90000 given the specific partnership details --/
theorem b_contribution_is_90000 :
  let p : Partnership := {
    a_investment := 35000,
    b_join_time := 5,
    total_time := 12,
    profit_ratio_a := 2,
    profit_ratio_b := 3
  }
  calculate_b_contribution p = 90000 := by
  sorry


end NUMINAMATH_CALUDE_b_contribution_is_90000_l4113_411354


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4113_411320

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - (11/17) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4113_411320


namespace NUMINAMATH_CALUDE_range_of_a_l4113_411315

open Set Real

def A : Set ℝ := {x | (x - 1) * (x - 2) ≥ 0}

theorem range_of_a (a : ℝ) :
  (A ∪ {x | x ≥ a} = univ) → a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4113_411315


namespace NUMINAMATH_CALUDE_subtraction_problem_solution_l4113_411365

theorem subtraction_problem_solution :
  ∀ h t u : ℕ,
  h > u →
  h < 10 ∧ t < 10 ∧ u < 10 →
  (100 * h + 10 * t + u) - (100 * t + 10 * h + u) = 553 →
  h = 9 ∧ t = 4 ∧ u = 3 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_solution_l4113_411365


namespace NUMINAMATH_CALUDE_milk_water_mixture_l4113_411332

/-- Given a mixture of milk and water with an initial ratio of 6:3 and a final ratio of 6:5 after
    adding 10 liters of water, the original quantity of milk is 30 liters. -/
theorem milk_water_mixture (milk : ℝ) (water : ℝ) : 
  milk / water = 6 / 3 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l4113_411332


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l4113_411307

theorem rectangle_area_problem :
  ∃ (length width : ℕ+), 
    (length : ℝ) * (width : ℝ) = ((length : ℝ) + 3) * ((width : ℝ) - 1) ∧
    (length : ℝ) * (width : ℝ) = ((length : ℝ) - 3) * ((width : ℝ) + 2) ∧
    (length : ℝ) * (width : ℝ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l4113_411307


namespace NUMINAMATH_CALUDE_quadratic_polynomial_roots_l4113_411302

theorem quadratic_polynomial_roots (x y : ℝ) (t : ℝ → ℝ) : 
  x + y = 12 → x * (3 * y) = 108 → 
  (∀ r, t r = 0 ↔ r = x ∨ r = y) → 
  t = fun r ↦ r^2 - 12*r + 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_roots_l4113_411302


namespace NUMINAMATH_CALUDE_price_difference_proof_l4113_411399

def shop_x_price : ℚ := 1.25
def shop_y_price : ℚ := 2.75
def num_copies : ℕ := 40

theorem price_difference_proof :
  (shop_y_price * num_copies) - (shop_x_price * num_copies) = 60 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_proof_l4113_411399


namespace NUMINAMATH_CALUDE_kids_at_camp_l4113_411323

theorem kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) 
  (h1 : total_kids = 313473) 
  (h2 : kids_at_home = 274865) : 
  total_kids - kids_at_home = 38608 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_l4113_411323


namespace NUMINAMATH_CALUDE_floor_of_5_7_l4113_411387

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l4113_411387


namespace NUMINAMATH_CALUDE_gain_percent_problem_l4113_411342

def gain_percent (gain : ℚ) (cost_price : ℚ) : ℚ :=
  (gain / cost_price) * 100

theorem gain_percent_problem (gain : ℚ) (cost_price : ℚ) 
  (h1 : gain = 70 / 100)  -- 70 paise = 0.70 rupees
  (h2 : cost_price = 70) : 
  gain_percent gain cost_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_problem_l4113_411342


namespace NUMINAMATH_CALUDE_smallest_positive_value_36k_minus_5m_l4113_411355

theorem smallest_positive_value_36k_minus_5m (k m : ℕ+) :
  (∀ n : ℕ+, 36^(k : ℕ) - 5^(m : ℕ) ≠ n) →
  (36^(k : ℕ) - 5^(m : ℕ) = 11 ∨ 36^(k : ℕ) - 5^(m : ℕ) > 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_value_36k_minus_5m_l4113_411355


namespace NUMINAMATH_CALUDE_point_trajectory_l4113_411346

/-- The trajectory of a point P satisfying |PA| + |PB| = 5, where A(0,0) and B(5,0) are fixed points -/
theorem point_trajectory (P : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (5, 0)
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 5 →
  P.2 = 0 ∧ 0 ≤ P.1 ∧ P.1 ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_point_trajectory_l4113_411346


namespace NUMINAMATH_CALUDE_trajectory_equation_l4113_411337

theorem trajectory_equation (a b x y : ℝ) : 
  a^2 + b^2 = 100 →  -- Line segment length is 10
  x = a / 5 →        -- AM = 4MB implies x = a/(1+4)
  y = 4*b / 5 →      -- AM = 4MB implies y = 4b/(1+4)
  16*x^2 + y^2 = 64  -- Trajectory equation
:= by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l4113_411337


namespace NUMINAMATH_CALUDE_one_absent_one_present_probability_l4113_411309

theorem one_absent_one_present_probability 
  (p_absent : ℝ) 
  (h_absent : p_absent = 1 / 20) : 
  let p_present := 1 - p_absent
  2 * (p_absent * p_present) = 19 / 200 := by
sorry

end NUMINAMATH_CALUDE_one_absent_one_present_probability_l4113_411309


namespace NUMINAMATH_CALUDE_some_number_value_l4113_411373

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l4113_411373


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l4113_411383

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) :
  i^14761 + i^14762 + i^14763 + i^14764 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l4113_411383


namespace NUMINAMATH_CALUDE_silva_family_zoo_cost_l4113_411328

/-- Calculates the total cost of zoo tickets for a family group -/
def total_zoo_cost (senior_ticket_cost : ℚ) (child_discount : ℚ) (senior_discount : ℚ) : ℚ :=
  let full_price := senior_ticket_cost / (1 - senior_discount)
  let child_price := full_price * (1 - child_discount)
  3 * senior_ticket_cost + 3 * full_price + 3 * child_price

/-- Theorem stating the total cost for the Silva family zoo trip -/
theorem silva_family_zoo_cost :
  total_zoo_cost 7 (4/10) (3/10) = 69 := by
  sorry

#eval total_zoo_cost 7 (4/10) (3/10)

end NUMINAMATH_CALUDE_silva_family_zoo_cost_l4113_411328


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l4113_411380

theorem gcd_special_numbers : Nat.gcd 33333333 777777777 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l4113_411380


namespace NUMINAMATH_CALUDE_shadow_problem_l4113_411308

/-- Given a cube with edge length 2 cm and a light source y cm above an upper vertex
    casting a shadow with area 175 sq cm (excluding the area beneath the cube),
    prove that the greatest integer less than or equal to 100y is 333. -/
theorem shadow_problem (y : ℝ) : 
  (2 : ℝ) > 0 ∧ 
  y > 0 ∧ 
  175 = (Real.sqrt 179 - 2)^2 →
  ⌊100 * y⌋ = 333 := by
  sorry

end NUMINAMATH_CALUDE_shadow_problem_l4113_411308


namespace NUMINAMATH_CALUDE_total_investment_sum_l4113_411396

/-- Proves that the total sum of investments is 6358 given the specified conditions --/
theorem total_investment_sum (raghu_investment : ℝ) 
  (h1 : raghu_investment = 2200)
  (h2 : ∃ trishul_investment : ℝ, trishul_investment = raghu_investment * 0.9)
  (h3 : ∃ vishal_investment : ℝ, vishal_investment = trishul_investment * 1.1) :
  ∃ total_investment : ℝ, total_investment = raghu_investment + trishul_investment + vishal_investment ∧ 
  total_investment = 6358 :=
by sorry

end NUMINAMATH_CALUDE_total_investment_sum_l4113_411396


namespace NUMINAMATH_CALUDE_parabola_one_intersection_l4113_411331

/-- A parabola that intersects the x-axis at exactly one point -/
def one_intersection_parabola (c : ℝ) : Prop :=
  ∃! x, x^2 + x + c = 0

/-- The theorem stating that the parabola y = x^2 + x + c intersects 
    the x-axis at exactly one point when c = 1/4 -/
theorem parabola_one_intersection :
  one_intersection_parabola (1/4 : ℝ) ∧ 
  ∀ c : ℝ, one_intersection_parabola c → c = 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_one_intersection_l4113_411331


namespace NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l4113_411305

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a^2 > a ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_a_gt_1_sufficient_not_necessary_for_a_sq_gt_a_l4113_411305


namespace NUMINAMATH_CALUDE_factor_problems_l4113_411393

theorem factor_problems : 
  (∃ n : ℤ, 25 = 5 * n) ∧ (∃ m : ℤ, 200 = 10 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_problems_l4113_411393


namespace NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l4113_411352

theorem max_stamps_for_50_dollars (stamp_price : ℕ) (available_amount : ℕ) : 
  stamp_price = 37 → available_amount = 5000 → 
  (∃ (n : ℕ), n * stamp_price ≤ available_amount ∧ 
  ∀ (m : ℕ), m * stamp_price ≤ available_amount → m ≤ n) → 
  (∃ (max_stamps : ℕ), max_stamps = 135) := by
  sorry

end NUMINAMATH_CALUDE_max_stamps_for_50_dollars_l4113_411352


namespace NUMINAMATH_CALUDE_amusement_park_spending_l4113_411340

/-- Calculates the total amount spent by a group of children at an amusement park -/
def total_spent (num_children : ℕ) 
  (ferris_wheel_cost ferris_wheel_riders : ℕ)
  (roller_coaster_cost roller_coaster_riders : ℕ)
  (merry_go_round_cost : ℕ)
  (bumper_cars_cost bumper_cars_riders : ℕ)
  (ice_cream_cost ice_cream_eaters : ℕ)
  (hot_dog_cost hot_dog_eaters : ℕ)
  (pizza_cost pizza_eaters : ℕ) : ℕ :=
  ferris_wheel_cost * ferris_wheel_riders +
  roller_coaster_cost * roller_coaster_riders +
  merry_go_round_cost * num_children +
  bumper_cars_cost * bumper_cars_riders +
  ice_cream_cost * ice_cream_eaters +
  hot_dog_cost * hot_dog_eaters +
  pizza_cost * pizza_eaters

/-- Theorem stating that the total amount spent by the group is $170 -/
theorem amusement_park_spending :
  total_spent 8 5 5 7 3 3 4 6 8 5 6 4 4 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l4113_411340


namespace NUMINAMATH_CALUDE_perimeter_difference_l4113_411310

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Represents Figure 1: a 3x6 rectangle --/
def figure1 : ℕ × ℕ := (3, 6)

/-- Represents Figure 2: a 2x7 rectangle with an additional square --/
def figure2 : ℕ × ℕ := (2, 7)

/-- The additional perimeter contributed by the extra square in Figure 2 --/
def extraSquarePerimeter : ℕ := 3

theorem perimeter_difference :
  rectanglePerimeter figure2.1 figure2.2 + extraSquarePerimeter -
  rectanglePerimeter figure1.1 figure1.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_l4113_411310


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4113_411321

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4113_411321


namespace NUMINAMATH_CALUDE_remaining_distance_l4113_411343

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 768) :
  total_distance - driven_distance = 432 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l4113_411343


namespace NUMINAMATH_CALUDE_ticket_distribution_theorem_l4113_411306

/-- The number of ways to distribute 3 different tickets to 3 students out of a group of 10 -/
def ticket_distribution_ways : ℕ := 10 * 9 * 8

/-- Theorem: The number of ways to distribute 3 different tickets to 3 students out of a group of 10 is 720 -/
theorem ticket_distribution_theorem : ticket_distribution_ways = 720 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_theorem_l4113_411306


namespace NUMINAMATH_CALUDE_square_side_length_l4113_411366

theorem square_side_length (perimeter : ℝ) (area : ℝ) (h1 : perimeter = 44) (h2 : area = 121) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4113_411366


namespace NUMINAMATH_CALUDE_smallest_a2_l4113_411356

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ a 2 > 0 ∧
  ∀ n ∈ Finset.range 7, a (n + 2) * a n * a (n - 1) = a (n + 2) + a n + a (n - 1)

def no_extension (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, x * a 8 * a 7 ≠ x + a 8 + a 7

theorem smallest_a2 (a : ℕ → ℝ) (h1 : sequence_property a) (h2 : no_extension a) :
  a 2 = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_a2_l4113_411356


namespace NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_not_zero_l4113_411322

-- Define the complex number i
def i : ℂ := Complex.I

-- Define x and y
def x : ℂ := i
def y : ℂ := -i

-- State the theorem
theorem x_fourth_plus_y_fourth_not_zero : x^4 + y^4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_y_fourth_not_zero_l4113_411322


namespace NUMINAMATH_CALUDE_gold_cube_side_length_l4113_411345

/-- Proves that a gold cube with given parameters has a side length of 6 cm -/
theorem gold_cube_side_length (L : ℝ) 
  (density : ℝ) (buy_price : ℝ) (sell_factor : ℝ) (profit : ℝ) :
  density = 19 →
  buy_price = 60 →
  sell_factor = 1.5 →
  profit = 123120 →
  profit = (sell_factor * buy_price * density * L^3) - (buy_price * density * L^3) →
  L = 6 :=
by sorry

end NUMINAMATH_CALUDE_gold_cube_side_length_l4113_411345


namespace NUMINAMATH_CALUDE_business_partnership_ratio_l4113_411301

/-- Represents the capital and profit distribution of a business partnership --/
structure BusinessPartnership where
  a : ℝ  -- Capital of partner a
  b : ℝ  -- Capital of partner b
  c : ℝ  -- Capital of partner c
  k : ℝ  -- Multiplier relating a's capital to b's capital
  total_profit : ℝ
  b_share : ℝ

/-- Theorem representing the business partnership problem --/
theorem business_partnership_ratio 
  (bp : BusinessPartnership)
  (h1 : bp.b = 4 * bp.c)
  (h2 : 2 * bp.a = bp.k * bp.b)
  (h3 : bp.total_profit = 16500)
  (h4 : bp.b_share = 6000)
  (h5 : bp.b / (bp.a + bp.b + bp.c) = bp.b_share / bp.total_profit) :
  2 * bp.a / bp.b = 3 := by
    sorry

#check business_partnership_ratio

end NUMINAMATH_CALUDE_business_partnership_ratio_l4113_411301


namespace NUMINAMATH_CALUDE_plastic_bag_co2_calculation_l4113_411369

/-- The amount of carbon dioxide released by a canvas bag in pounds -/
def canvas_bag_co2 : ℝ := 600

/-- The number of plastic bags used per shopping trip -/
def bags_per_trip : ℕ := 8

/-- The number of shopping trips needed for the canvas bag to become the lower-carbon solution -/
def trips_for_breakeven : ℕ := 300

/-- The number of ounces in a pound -/
def ounces_per_pound : ℝ := 16

/-- The amount of carbon dioxide released by each plastic bag in ounces -/
def plastic_bag_co2 : ℝ := 4

theorem plastic_bag_co2_calculation :
  plastic_bag_co2 = canvas_bag_co2 * ounces_per_pound / (bags_per_trip * trips_for_breakeven) :=
by sorry

end NUMINAMATH_CALUDE_plastic_bag_co2_calculation_l4113_411369


namespace NUMINAMATH_CALUDE_square_sum_xy_l4113_411385

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 35)
  (h2 : y * (x + y) = 77) : 
  (x + y)^2 = 112 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l4113_411385


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4113_411376

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 4*x^3 - 6*x^3 + 8*x^3 = 
  -3 + 23*x - x^2 + 6*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4113_411376


namespace NUMINAMATH_CALUDE_dog_weight_l4113_411324

theorem dog_weight (k d r : ℚ) 
  (total_weight : d + k + r = 40)
  (dog_rabbit_twice_kitten : d + r = 2 * k)
  (dog_kitten_equals_rabbit : d + k = r) : 
  d = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_l4113_411324


namespace NUMINAMATH_CALUDE_eugene_toothpick_boxes_l4113_411341

def toothpicks_per_card : ℕ := 64
def total_cards : ℕ := 52
def unused_cards : ℕ := 23
def toothpicks_per_box : ℕ := 550

theorem eugene_toothpick_boxes : 
  ∃ (boxes : ℕ), 
    boxes = (((total_cards - unused_cards) * toothpicks_per_card + toothpicks_per_box - 1) / toothpicks_per_box : ℕ) ∧ 
    boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_eugene_toothpick_boxes_l4113_411341


namespace NUMINAMATH_CALUDE_ellipse_equation_proof_l4113_411398

theorem ellipse_equation_proof (a b : ℝ) : 
  (∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0) → -- ellipse passes through (2, 0)
  (a^2 - b^2 = 2) → -- ellipse shares focus with hyperbola x² - y² = 1
  (a^2 = 4 ∧ b^2 = 2) → -- derived from the conditions
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ellipse_equation_proof_l4113_411398


namespace NUMINAMATH_CALUDE_french_toast_slices_l4113_411335

/- Define the problem parameters -/
def weeks_per_year : ℕ := 52
def days_per_week : ℕ := 2
def loaves_used : ℕ := 26
def slices_per_loaf : ℕ := 12
def slices_for_daughters : ℕ := 1

/- Define the function to calculate slices per person -/
def slices_per_person : ℚ :=
  let total_slices := loaves_used * slices_per_loaf
  let total_days := weeks_per_year * days_per_week
  let slices_per_day := total_slices / total_days
  let slices_for_parents := slices_per_day - slices_for_daughters
  slices_for_parents / 2

/- State the theorem -/
theorem french_toast_slices :
  slices_per_person = 1 := by sorry

end NUMINAMATH_CALUDE_french_toast_slices_l4113_411335


namespace NUMINAMATH_CALUDE_parallel_vectors_t_value_l4113_411338

def vector_a : Fin 2 → ℝ := ![(-1), 3]
def vector_b (t : ℝ) : Fin 2 → ℝ := ![1, t]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ i, u i = k * v i

theorem parallel_vectors_t_value :
  ∀ t : ℝ, parallel vector_a (vector_b t) → t = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_t_value_l4113_411338


namespace NUMINAMATH_CALUDE_dallas_current_age_l4113_411371

/-- Proves Dallas's current age given the relationships between family members' ages --/
theorem dallas_current_age (dallas_last_year darcy_last_year darcy_current dexter_current : ℕ) 
  (h1 : dallas_last_year = 3 * darcy_last_year)
  (h2 : darcy_current = 2 * dexter_current)
  (h3 : dexter_current = 8) :
  dallas_last_year + 1 = 46 := by
  sorry

#check dallas_current_age

end NUMINAMATH_CALUDE_dallas_current_age_l4113_411371


namespace NUMINAMATH_CALUDE_consecutive_even_product_divisibility_l4113_411386

theorem consecutive_even_product_divisibility (n : ℤ) (h : Even n) :
  ∃ k : ℤ, n * (n + 2) * (n + 4) * (n + 6) = 48 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_divisibility_l4113_411386


namespace NUMINAMATH_CALUDE_min_value_quadratic_l4113_411311

theorem min_value_quadratic (x y : ℝ) (h : x^2 + x*y + y^2 = 3) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ z, z = x^2 - x*y + y^2 → z ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l4113_411311


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_equal_distance_l4113_411379

-- Define the point P as a function of a
def P (a : ℝ) : ℝ × ℝ := (2*a - 3, a + 6)

-- Part 1
theorem point_on_x_axis (a : ℝ) : 
  P a = (-15, 0) ↔ (P a).2 = 0 :=
sorry

-- Part 2
theorem point_in_second_quadrant_equal_distance (a : ℝ) :
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ abs (P a).1 = (P a).2 → a^2003 + 2024 = 2023 :=
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_in_second_quadrant_equal_distance_l4113_411379


namespace NUMINAMATH_CALUDE_building_has_at_least_43_floors_l4113_411389

/-- Represents a building with apartments -/
structure Building where
  apartments_per_floor : ℕ
  kolya_floor : ℕ
  kolya_apartment : ℕ
  vasya_floor : ℕ
  vasya_apartment : ℕ

/-- The specific building described in the problem -/
def problem_building : Building :=
  { apartments_per_floor := 4
  , kolya_floor := 5
  , kolya_apartment := 83
  , vasya_floor := 3
  , vasya_apartment := 169
  }

/-- Calculates the minimum number of floors in the building -/
def min_floors (b : Building) : ℕ :=
  ((b.vasya_apartment - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that the building has at least 43 floors -/
theorem building_has_at_least_43_floors :
  min_floors problem_building ≥ 43 := by
  sorry


end NUMINAMATH_CALUDE_building_has_at_least_43_floors_l4113_411389
