import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l3730_373089

-- Define the expression as a function of x and some_expression
def f (x : ℝ) (some_expression : ℝ) : ℝ :=
  |x - 4| + |x + 6| + |some_expression|

-- State the theorem
theorem min_value_of_some_expression :
  (∃ (some_expression : ℝ), ∀ (x : ℝ), f x some_expression ≥ 11) →
  (∃ (some_expression : ℝ), (∀ (x : ℝ), f x some_expression ≥ 11) ∧ |some_expression| = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_some_expression_l3730_373089


namespace NUMINAMATH_CALUDE_max_infected_population_l3730_373097

/-- A graph representing the CMI infection spread --/
structure InfectionGraph where
  V : Type*  -- Set of vertices (people)
  E : V → V → Prop  -- Edge relation (friendship)
  degree_bound : ∀ v : V, (∃ n : ℕ, n ≤ 3 ∧ (∃ (l : List V), l.length = n ∧ (∀ u ∈ l, E v u)))

/-- The infection state of the graph over time --/
def InfectionState (G : InfectionGraph) := ℕ → G.V → Prop

/-- The initial infection state --/
def initial_infection (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∃ (infected : Finset G.V), infected.card = 2023 ∧ 
    ∀ v, S 0 v ↔ v ∈ infected

/-- The infection spread rule --/
def infection_rule (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ t v, S (t + 1) v ↔ 
    S t v ∨ (∃ (u₁ u₂ : G.V), u₁ ≠ u₂ ∧ G.E v u₁ ∧ G.E v u₂ ∧ S t u₁ ∧ S t u₂)

/-- Everyone eventually gets infected --/
def all_infected (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ v, ∃ t, S t v

/-- The main theorem --/
theorem max_infected_population (G : InfectionGraph) (S : InfectionState G) 
  (h_initial : initial_infection G S)
  (h_rule : infection_rule G S)
  (h_all : all_infected G S) :
  ∀ n : ℕ, (∃ f : G.V → Fin n, Function.Injective f) → n ≤ 4043 := by
  sorry

end NUMINAMATH_CALUDE_max_infected_population_l3730_373097


namespace NUMINAMATH_CALUDE_five_in_range_of_quadratic_l3730_373042

theorem five_in_range_of_quadratic (b : ℝ) : ∃ x : ℝ, x^2 + b*x + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_in_range_of_quadratic_l3730_373042


namespace NUMINAMATH_CALUDE_min_lines_is_seven_l3730_373083

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- The quadrants a line passes through --/
def quadrants (l : Line) : Set (Fin 4) :=
  sorry

/-- The minimum number of lines needed to ensure two lines pass through the same quadrants --/
def min_lines_same_quadrants : ℕ :=
  sorry

/-- Theorem stating that the minimum number of lines is 7 --/
theorem min_lines_is_seven : min_lines_same_quadrants = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_is_seven_l3730_373083


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3730_373005

theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3730_373005


namespace NUMINAMATH_CALUDE_adjacent_sum_9_is_30_l3730_373036

def divisors_of_216 : List ℕ := [2, 3, 4, 6, 8, 9, 12, 18, 24, 27, 36, 54, 72, 108, 216]

def valid_arrangement (arr : List ℕ) : Prop :=
  ∀ i j, i ≠ j → (arr.get! i).gcd (arr.get! j) > 1

def adjacent_sum_9 (arr : List ℕ) : ℕ :=
  let idx := arr.indexOf 9
  (arr.get! ((idx - 1 + arr.length) % arr.length)) + (arr.get! ((idx + 1) % arr.length))

theorem adjacent_sum_9_is_30 :
  ∃ arr : List ℕ, arr.Perm divisors_of_216 ∧ valid_arrangement arr ∧ adjacent_sum_9 arr = 30 :=
sorry

end NUMINAMATH_CALUDE_adjacent_sum_9_is_30_l3730_373036


namespace NUMINAMATH_CALUDE_bananas_permutations_l3730_373020

/-- The number of distinct permutations of a word with repeated letters -/
def distinct_permutations (total : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial total / (List.prod (List.map Nat.factorial repetitions))

/-- Theorem: The number of distinct permutations of BANANAS is 420 -/
theorem bananas_permutations :
  distinct_permutations 7 [3, 2] = 420 := by
  sorry

#eval distinct_permutations 7 [3, 2]

end NUMINAMATH_CALUDE_bananas_permutations_l3730_373020


namespace NUMINAMATH_CALUDE_simplify_fraction_solve_inequality_system_l3730_373032

-- Problem 1
theorem simplify_fraction (m n : ℝ) (hm : m ≠ 0) (hmn : 9*m^2 ≠ 4*n^2) :
  (1/(3*m-2*n) - 1/(3*m+2*n)) / (m*n/((9*m^2)-(4*n^2))) = 4/m := by sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (3*x + 10 > 5*x - 2*(5-x) ∧ (x+3)/5 > 1-x) ↔ (1/3 < x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_solve_inequality_system_l3730_373032


namespace NUMINAMATH_CALUDE_probability_at_most_one_defective_is_five_sevenths_l3730_373007

def total_products : ℕ := 8
def defective_products : ℕ := 3
def drawn_products : ℕ := 3

def probability_at_most_one_defective : ℚ :=
  (Nat.choose (total_products - defective_products) drawn_products +
   Nat.choose (total_products - defective_products) (drawn_products - 1) * 
   Nat.choose defective_products 1) /
  Nat.choose total_products drawn_products

theorem probability_at_most_one_defective_is_five_sevenths :
  probability_at_most_one_defective = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_most_one_defective_is_five_sevenths_l3730_373007


namespace NUMINAMATH_CALUDE_carpet_width_l3730_373030

/-- Proves that given a room 15 meters long and 6 meters wide, carpeted at a cost of 30 paise per meter for a total of Rs. 36, the width of the carpet used is 800 centimeters. -/
theorem carpet_width (room_length : ℝ) (room_breadth : ℝ) (carpet_cost_paise : ℝ) (total_cost_rupees : ℝ) :
  room_length = 15 →
  room_breadth = 6 →
  carpet_cost_paise = 30 →
  total_cost_rupees = 36 →
  ∃ (carpet_width : ℝ), carpet_width = 800 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_l3730_373030


namespace NUMINAMATH_CALUDE_function_property_l3730_373035

theorem function_property (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f x + f y = f (x + 1) + f (y - 1))
  (h1 : f 2016 = 6102)
  (h2 : f 6102 = 2016) :
  f 1 = 8117 := by
sorry

end NUMINAMATH_CALUDE_function_property_l3730_373035


namespace NUMINAMATH_CALUDE_polygon_sides_l3730_373033

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) → 
  ((n - 2) * 180 + 360 = 1800) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3730_373033


namespace NUMINAMATH_CALUDE_cross_pollinated_percentage_l3730_373037

theorem cross_pollinated_percentage
  (total : ℕ)  -- Total number of trees
  (fuji : ℕ)   -- Number of pure Fuji trees
  (gala : ℕ)   -- Number of pure Gala trees
  (cross : ℕ)  -- Number of cross-pollinated trees
  (h1 : total = fuji + gala + cross)  -- Total trees equation
  (h2 : fuji + cross = 221)           -- Pure Fuji + Cross-pollinated
  (h3 : fuji = (3 * total) / 4)       -- 3/4 of all trees are pure Fuji
  (h4 : gala = 39)                    -- Number of pure Gala trees
  : (cross : ℚ) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cross_pollinated_percentage_l3730_373037


namespace NUMINAMATH_CALUDE_second_store_unload_percentage_l3730_373080

def initial_load : ℝ := 50000
def first_unload_percent : ℝ := 0.1
def remaining_after_deliveries : ℝ := 36000

theorem second_store_unload_percentage :
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  let unloaded_at_second := remaining_after_first - remaining_after_deliveries
  (unloaded_at_second / remaining_after_first) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_store_unload_percentage_l3730_373080


namespace NUMINAMATH_CALUDE_means_and_sum_of_squares_l3730_373049

theorem means_and_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_means_and_sum_of_squares_l3730_373049


namespace NUMINAMATH_CALUDE_alice_wrong_questions_l3730_373012

/-- Represents the number of questions a person got wrong in the test. -/
structure TestResult where
  wrong : ℕ

/-- Represents the test results for Alice, Beth, Charlie, Daniel, and Ellen. -/
structure TestResults where
  alice : TestResult
  beth : TestResult
  charlie : TestResult
  daniel : TestResult
  ellen : TestResult

/-- The theorem stating that Alice got 9 questions wrong given the conditions. -/
theorem alice_wrong_questions (results : TestResults) : results.alice.wrong = 9 :=
  by
  have h1 : results.alice.wrong + results.beth.wrong = results.charlie.wrong + results.daniel.wrong + results.ellen.wrong :=
    sorry
  have h2 : results.alice.wrong + results.daniel.wrong = results.beth.wrong + results.charlie.wrong + 3 :=
    sorry
  have h3 : results.charlie.wrong = 6 :=
    sorry
  have h4 : results.daniel.wrong = 8 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_alice_wrong_questions_l3730_373012


namespace NUMINAMATH_CALUDE_triangle_height_relationship_l3730_373026

/-- Given two triangles A and B, proves the relationship between their heights
    when their bases and areas are related. -/
theorem triangle_height_relationship (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let base_A := 1.2 * b
  let area_B := (1 / 2) * b * h
  let area_A := 0.9975 * area_B
  let height_A := (2 * area_A) / base_A
  height_A / h = 0.83125 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_relationship_l3730_373026


namespace NUMINAMATH_CALUDE_fruit_sales_revenue_l3730_373091

theorem fruit_sales_revenue : 
  let original_lemon_price : ℝ := 8
  let original_grape_price : ℝ := 7
  let lemon_price_increase : ℝ := 4
  let grape_price_increase : ℝ := lemon_price_increase / 2
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let new_lemon_price : ℝ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℝ := original_grape_price + grape_price_increase
  let total_revenue : ℝ := (↑num_lemons * new_lemon_price) + (↑num_grapes * new_grape_price)
  total_revenue = 2220 := by
sorry

end NUMINAMATH_CALUDE_fruit_sales_revenue_l3730_373091


namespace NUMINAMATH_CALUDE_arun_age_proof_l3730_373079

theorem arun_age_proof (A S G M : ℕ) : 
  A - 6 = 18 * G →
  G + 2 = M →
  M = 5 →
  S = A - 8 →
  A = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_proof_l3730_373079


namespace NUMINAMATH_CALUDE_lcm_18_35_l3730_373071

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_35_l3730_373071


namespace NUMINAMATH_CALUDE_distance_at_16_00_l3730_373015

/-- Represents the distance to Moscow at a given time -/
structure DistanceAtTime where
  time : ℕ  -- Time in hours since 12:00
  lowerBound : ℚ
  upperBound : ℚ

/-- The problem statement -/
theorem distance_at_16_00 
  (d12 : DistanceAtTime) 
  (d13 : DistanceAtTime)
  (d15 : DistanceAtTime)
  (h_constant_speed : ∀ t₁ t₂, d12.time ≤ t₁ → t₁ < t₂ → t₂ ≤ d15.time → 
    (d12.lowerBound - d15.upperBound) / (d15.time - d12.time) ≤ 
    (d12.upperBound - d15.lowerBound) / (d15.time - d12.time))
  (h_d12 : d12.time = 0 ∧ d12.lowerBound = 81.5 ∧ d12.upperBound = 82.5)
  (h_d13 : d13.time = 1 ∧ d13.lowerBound = 70.5 ∧ d13.upperBound = 71.5)
  (h_d15 : d15.time = 3 ∧ d15.lowerBound = 45.5 ∧ d15.upperBound = 46.5) :
  ∃ (d : ℚ), d = 34 ∧ 
    (d12.lowerBound - d) / 4 = (d12.upperBound - d) / 4 ∧
    (d13.lowerBound - d) / 3 = (d13.upperBound - d) / 3 ∧
    (d15.lowerBound - d) / 1 = (d15.upperBound - d) / 1 :=
sorry

end NUMINAMATH_CALUDE_distance_at_16_00_l3730_373015


namespace NUMINAMATH_CALUDE_thirty_seventh_digit_of_1_17_l3730_373076

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℕ → ℕ
| 0 => 0
| 1 => 5
| 2 => 8
| 3 => 8
| 4 => 2
| 5 => 3
| 6 => 5
| 7 => 2
| 8 => 9
| 9 => 4
| 10 => 1
| 11 => 1
| 12 => 7
| 13 => 6
| 14 => 4
| 15 => 7
| n + 16 => decimal_rep_1_17 n

-- Define the period of the decimal representation
def period : ℕ := 16

-- Theorem statement
theorem thirty_seventh_digit_of_1_17 :
  decimal_rep_1_17 ((37 - 1) % period) = 8 := by
  sorry

end NUMINAMATH_CALUDE_thirty_seventh_digit_of_1_17_l3730_373076


namespace NUMINAMATH_CALUDE_tangent_line_at_minus_one_l3730_373060

def curve (x : ℝ) : ℝ := x^3

theorem tangent_line_at_minus_one : 
  let p : ℝ × ℝ := (-1, -1)
  let m : ℝ := 3 * p.1^2
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  ∀ x, tangent_line x = 3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_minus_one_l3730_373060


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3730_373067

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 5 * i) / (2 + 7 * i) = Complex.mk (-29/53) (-31/53) :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3730_373067


namespace NUMINAMATH_CALUDE_triangle_special_angle_l3730_373086

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a^2 + b^2 - √2ab = c^2, then the measure of angle C is π/4 -/
theorem triangle_special_angle (a b c : ℝ) (h : a^2 + b^2 - Real.sqrt 2 * a * b = c^2) :
  let angle_C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  angle_C = π / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_special_angle_l3730_373086


namespace NUMINAMATH_CALUDE_range_of_t_l3730_373006

/-- A function f(x) = x^2 - 2tx + 1 that is decreasing on (-∞, 1] -/
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

/-- The theorem stating the range of t given the conditions -/
theorem range_of_t (t : ℝ) : 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f t x > f t y) →  -- f is decreasing on (-∞, 1]
  (∀ x₁ ∈ Set.Icc 0 (t+1), ∀ x₂ ∈ Set.Icc 0 (t+1), |f t x₁ - f t x₂| ≤ 2) →  -- |f(x₁) - f(x₂)| ≤ 2
  t ∈ Set.Icc 1 (Real.sqrt 2) :=  -- t ∈ [1, √2]
sorry

end NUMINAMATH_CALUDE_range_of_t_l3730_373006


namespace NUMINAMATH_CALUDE_waiter_dishes_served_l3730_373011

/-- Calculates the total number of dishes served by a waiter --/
def total_dishes_served (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (courses_per_woman : ℕ) (courses_per_man : ℕ) 
  (shared_courses_women : ℕ) (shared_courses_men : ℕ) : ℕ :=
  let dishes_per_table := 
    (women_per_table * courses_per_woman - shared_courses_women) +
    (men_per_table * courses_per_man - shared_courses_men)
  num_tables * dishes_per_table

/-- Theorem stating the total number of dishes served under given conditions --/
theorem waiter_dishes_served : 
  total_dishes_served 7 7 2 3 4 1 2 = 182 := by
  sorry

end NUMINAMATH_CALUDE_waiter_dishes_served_l3730_373011


namespace NUMINAMATH_CALUDE_largest_prime_in_equation_l3730_373081

theorem largest_prime_in_equation (x : ℤ) (n : ℕ) (p : ℕ) 
  (hp : Nat.Prime p) (heq : 7 * x^2 - 44 * x + 12 = p^n) :
  p ≤ 47 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_in_equation_l3730_373081


namespace NUMINAMATH_CALUDE_angle_A_measure_l3730_373029

-- Define the measure of angles A and B
def measure_A : ℝ := sorry
def measure_B : ℝ := sorry

-- Define the conditions
axiom supplementary : measure_A + measure_B = 180
axiom relation : measure_A = 3 * measure_B

-- Theorem to prove
theorem angle_A_measure : measure_A = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l3730_373029


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3730_373066

/-- Atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Deuterium in g/mol -/
def atomic_weight_D : ℝ := 2.01

/-- Number of Barium atoms in the compound -/
def num_Ba : ℕ := 2

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- Number of regular Hydrogen atoms in the compound -/
def num_H : ℕ := 4

/-- Number of Deuterium atoms in the compound -/
def num_D : ℕ := 1

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  (num_Ba : ℝ) * atomic_weight_Ba +
  (num_O : ℝ) * atomic_weight_O +
  (num_H : ℝ) * atomic_weight_H +
  (num_D : ℝ) * atomic_weight_D

theorem molecular_weight_calculation :
  molecular_weight = 328.71 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3730_373066


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_4_l3730_373073

theorem nearest_integer_to_3_plus_sqrt2_to_4 :
  ∃ (n : ℤ), n = 386 ∧ ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 2)^4 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^4 - m| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_to_4_l3730_373073


namespace NUMINAMATH_CALUDE_animal_count_l3730_373095

theorem animal_count (dogs cats frogs : ℕ) : 
  cats = (80 * dogs) / 100 →
  frogs = 2 * dogs →
  frogs = 160 →
  dogs + cats + frogs = 304 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l3730_373095


namespace NUMINAMATH_CALUDE_rogers_new_crayons_l3730_373059

/-- Given that Roger has 4 used crayons, 8 broken crayons, and a total of 14 crayons,
    prove that the number of new crayons is 2. -/
theorem rogers_new_crayons (used : ℕ) (broken : ℕ) (total : ℕ) (new : ℕ) :
  used = 4 →
  broken = 8 →
  total = 14 →
  new + used + broken = total →
  new = 2 := by
  sorry

end NUMINAMATH_CALUDE_rogers_new_crayons_l3730_373059


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l3730_373077

/-- The number of ways to choose 3 distinct positions from a group of n people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem states that choosing 3 distinct positions from 6 people results in 120 ways -/
theorem three_positions_from_six_people :
  choose_three_positions 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l3730_373077


namespace NUMINAMATH_CALUDE_smallest_k_satisfying_condition_l3730_373027

def S (n : ℕ) : ℤ := 2 * n^2 - 15 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem smallest_k_satisfying_condition : 
  (∀ k < 6, a k + a (k + 1) ≤ 12) ∧ 
  (a 6 + a 7 > 12) := by sorry

end NUMINAMATH_CALUDE_smallest_k_satisfying_condition_l3730_373027


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l3730_373001

/-- Given a triangle ABC with A(0,6), B(0,0), C(8,0), 
    D is the midpoint of AB, 
    E is on BC such that BE is one-third of BC,
    F is the intersection of AE and CD.
    Prove that the sum of x and y coordinates of F is 56/11. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (0, 6) →
  B = (0, 0) →
  C = (8, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E.1 = B.1 + (C.1 - B.1) / 3 →
  E.2 = B.2 + (C.2 - B.2) / 3 →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 56 / 11 := by
  sorry


end NUMINAMATH_CALUDE_intersection_coordinate_sum_l3730_373001


namespace NUMINAMATH_CALUDE_yunas_marbles_l3730_373017

/-- Yuna's marble problem -/
theorem yunas_marbles (M : ℕ) : 
  (((M - 12 + 5) / 2 : ℚ) + 3 : ℚ) = 17 → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_yunas_marbles_l3730_373017


namespace NUMINAMATH_CALUDE_min_phase_shift_l3730_373050

theorem min_phase_shift (x φ : ℝ) : 
  (∀ x, 2 * Real.sin (x + π/6 - φ) = 2 * Real.sin (x - π/3)) →
  (φ > 0 → φ ≥ π/2) ∧ 
  (∃ φ₀ > 0, ∀ x, 2 * Real.sin (x + π/6 - φ₀) = 2 * Real.sin (x - π/3) ∧ φ₀ = π/2) := by
  sorry

#check min_phase_shift

end NUMINAMATH_CALUDE_min_phase_shift_l3730_373050


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_nine_l3730_373082

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The set of all possible sums when rolling two dice -/
def possible_sums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of sums greater than 9 -/
def sums_greater_than_nine : Set ℕ := {10, 11, 12}

/-- The number of favorable outcomes (sums greater than 9) -/
def favorable_outcomes : ℕ := 6

/-- Theorem: The probability of rolling two dice and getting a sum greater than 9 is 1/6 -/
theorem probability_sum_greater_than_nine :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_nine_l3730_373082


namespace NUMINAMATH_CALUDE_negation_of_unique_solution_l3730_373014

theorem negation_of_unique_solution (a b : ℝ) (h : a ≠ 0) :
  ¬(∃! x : ℝ, a * x = b) ↔ (¬∃ x : ℝ, a * x = b) ∨ (∃ x y : ℝ, x ≠ y ∧ a * x = b ∧ a * y = b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_unique_solution_l3730_373014


namespace NUMINAMATH_CALUDE_table_height_l3730_373045

/-- Given three rectangular boxes (blue, red, and green) and their height relationships
    with a table, prove that the height of the table is 91 cm. -/
theorem table_height
  (h b r g : ℝ)
  (eq1 : h + b - g = 111)
  (eq2 : h + r - b = 80)
  (eq3 : h + g - r = 82) :
  h = 91 := by
sorry

end NUMINAMATH_CALUDE_table_height_l3730_373045


namespace NUMINAMATH_CALUDE_trig_identity_l3730_373063

theorem trig_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3730_373063


namespace NUMINAMATH_CALUDE_triangle_perimeter_not_55_l3730_373024

theorem triangle_perimeter_not_55 (a b x : ℝ) : 
  a = 18 → b = 10 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) → 
  a + b + x ≠ 55 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_not_55_l3730_373024


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3730_373048

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^2 - 2ax + a^2 -/
theorem line_through_parabola_vertex :
  ∃! a : ℝ, ∀ x y : ℝ,
    (y = x + a) ∧ (y = x^2 - 2*a*x + a^2) →
    (x = a ∧ y = 0) := by sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3730_373048


namespace NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l3730_373075

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bound by two circles and the x-axis -/
def areaRegion (c1 c2 : Circle) : ℝ :=
  sorry

theorem area_between_circles_and_xaxis :
  let c1 : Circle := { center := (3, 3), radius := 3 }
  let c2 : Circle := { center := (9, 3), radius := 3 }
  areaRegion c1 c2 = 18 - (9 * Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_and_xaxis_l3730_373075


namespace NUMINAMATH_CALUDE_ticket_sales_result_l3730_373016

/-- Represents a section in the stadium -/
structure Section where
  name : String
  seats : Nat
  price : Nat

/-- Represents the stadium configuration -/
def Stadium : List Section := [
  ⟨"A", 40, 10⟩,
  ⟨"B", 30, 15⟩,
  ⟨"C", 25, 20⟩
]

/-- Theorem stating the result of the ticket sales -/
theorem ticket_sales_result 
  (children : Nat) (adults : Nat) (seniors : Nat)
  (h1 : children = 52)
  (h2 : adults = 29)
  (h3 : seniors = 15)
  (h4 : children + adults + seniors = Stadium.foldr (fun s acc => s.seats + acc) 0 + 1) :
  (∀ s : Section, s ∈ Stadium → 
    (if s.name = "A" then adults + seniors else children) ≥ s.seats) ∧
  (Stadium.foldr (fun s acc => s.seats * s.price + acc) 0 = 1350) := by
  sorry

#check ticket_sales_result

end NUMINAMATH_CALUDE_ticket_sales_result_l3730_373016


namespace NUMINAMATH_CALUDE_division_remainder_l3730_373062

theorem division_remainder : ∃ (q r : ℕ), 1620 = (1620 - 1365) * q + r ∧ r < (1620 - 1365) ∧ r = 90 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l3730_373062


namespace NUMINAMATH_CALUDE_isabel_birthday_money_l3730_373021

/-- The amount of money Isabel received for her birthday -/
def birthday_money : ℕ := sorry

/-- The cost of each toy -/
def toy_cost : ℕ := 2

/-- The number of toys Isabel could buy -/
def toys_bought : ℕ := 7

/-- Theorem stating that Isabel's birthday money is equal to the total cost of the toys she could buy -/
theorem isabel_birthday_money :
  birthday_money = toy_cost * toys_bought :=
sorry

end NUMINAMATH_CALUDE_isabel_birthday_money_l3730_373021


namespace NUMINAMATH_CALUDE_chicken_pasta_orders_count_l3730_373023

def chicken_pasta_pieces : ℕ := 2
def barbecue_chicken_pieces : ℕ := 3
def fried_chicken_dinner_pieces : ℕ := 8
def fried_chicken_dinner_orders : ℕ := 2
def barbecue_chicken_orders : ℕ := 3
def total_chicken_pieces : ℕ := 37

theorem chicken_pasta_orders_count : 
  ∃ (chicken_pasta_orders : ℕ), 
    chicken_pasta_orders * chicken_pasta_pieces + 
    barbecue_chicken_orders * barbecue_chicken_pieces + 
    fried_chicken_dinner_orders * fried_chicken_dinner_pieces = 
    total_chicken_pieces ∧ 
    chicken_pasta_orders = 6 := by
  sorry

end NUMINAMATH_CALUDE_chicken_pasta_orders_count_l3730_373023


namespace NUMINAMATH_CALUDE_blue_surface_area_fraction_l3730_373096

theorem blue_surface_area_fraction (edge_length : ℕ) (small_cube_count : ℕ) 
  (green_count : ℕ) (blue_count : ℕ) :
  edge_length = 4 →
  small_cube_count = 64 →
  green_count = 44 →
  blue_count = 20 →
  (∃ (blue_exposed : ℕ), 
    blue_exposed ≤ blue_count ∧ 
    blue_exposed * 1 = (edge_length ^ 2 * 6) / 8) :=
by sorry

end NUMINAMATH_CALUDE_blue_surface_area_fraction_l3730_373096


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l3730_373040

theorem sqrt_sum_equals_thirteen_sixths : 
  Real.sqrt (9 / 4) + Real.sqrt (4 / 9) = 13 / 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_thirteen_sixths_l3730_373040


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_l3730_373003

theorem quadratic_roots_negative (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (p > 5/9 ∧ p ≤ 1) ∨ p ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_l3730_373003


namespace NUMINAMATH_CALUDE_circle_on_y_axis_through_point_one_two_l3730_373018

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_on_y_axis_through_point_one_two :
  ∃ (c : Circle),
    c.center.1 = 0 ∧
    c.radius = 1 ∧
    circle_equation c 1 2 ∧
    ∀ (x y : ℝ), circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_on_y_axis_through_point_one_two_l3730_373018


namespace NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l3730_373056

/-- Given real functions f and g defined on the entire real line, 
    satisfying certain conditions, prove that |g(y)| ≤ 1 for all y -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ) 
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l3730_373056


namespace NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3730_373051

theorem product_from_lcm_and_gcd (a b : ℕ+) : 
  Nat.lcm a b = 72 → Nat.gcd a b = 6 → a * b = 432 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_and_gcd_l3730_373051


namespace NUMINAMATH_CALUDE_distance_AB_is_70_l3730_373057

/-- The distance between two points A and B, given specific travel conditions of two couriers --/
def distance_AB : ℝ := by sorry

theorem distance_AB_is_70 :
  let t₁ := 14 -- Travel time for first courier in hours
  let d := 10 -- Distance behind A where second courier starts in km
  let x := distance_AB -- Distance from A to B in km
  let v₁ := x / t₁ -- Speed of first courier
  let v₂ := (x + d) / t₁ -- Speed of second courier
  let t₁_20 := 20 / v₁ -- Time for first courier to travel 20 km
  let t₂_20 := 20 / v₂ -- Time for second courier to travel 20 km
  t₁_20 = t₂_20 + 0.5 → -- Second courier is half hour faster over 20 km
  x = 70 := by sorry

end NUMINAMATH_CALUDE_distance_AB_is_70_l3730_373057


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l3730_373054

theorem fraction_multiplication_addition : (2 : ℚ) / 9 * 5 / 8 + 1 / 4 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l3730_373054


namespace NUMINAMATH_CALUDE_fair_coin_heads_prob_equals_frequency_l3730_373087

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  total_tosses : ℕ
  heads_count : ℕ
  heads_frequency : ℝ

/-- Defines what it means for an experiment to be valid -/
def is_valid_experiment (e : CoinTossExperiment) : Prop :=
  e.total_tosses > 0 ∧ 
  e.heads_count ≤ e.total_tosses ∧ 
  e.heads_frequency = (e.heads_count : ℝ) / (e.total_tosses : ℝ)

/-- The probability of a fair coin landing heads up -/
def fair_coin_heads_probability : ℝ := 0.5005

/-- Pearson's experiment data -/
def pearson_experiment : CoinTossExperiment := {
  total_tosses := 24000,
  heads_count := 12012,
  heads_frequency := 0.5005
}

/-- Theorem stating that the probability of a fair coin landing heads up
    is equal to the frequency observed in Pearson's large-scale experiment -/
theorem fair_coin_heads_prob_equals_frequency 
  (h_valid : is_valid_experiment pearson_experiment)
  (h_large : pearson_experiment.total_tosses ≥ 10000) :
  fair_coin_heads_probability = pearson_experiment.heads_frequency := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_heads_prob_equals_frequency_l3730_373087


namespace NUMINAMATH_CALUDE_students_per_group_l3730_373078

theorem students_per_group (total_students : Nat) (num_teachers : Nat) 
  (h1 : total_students = 256) 
  (h2 : num_teachers = 8) 
  (h3 : num_teachers > 0) : 
  total_students / num_teachers = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_per_group_l3730_373078


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3730_373094

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 3*x^2 + 1)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (15^2) + (5^2) = 375 := by sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l3730_373094


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3730_373055

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define the angle measure function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_measure_in_triangle (A B C P : ℝ × ℝ) :
  Triangle A B C →
  angle_measure A B C = 40 →
  angle_measure A C B = 40 →
  angle_measure P A C = 20 →
  angle_measure P C B = 30 →
  angle_measure P B C = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3730_373055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3730_373092

/-- An arithmetic sequence with given third and eleventh terms -/
def ArithmeticSequence (a₃ a₁₁ : ℚ) :=
  ∃ (a₁ d : ℚ), a₃ = a₁ + 2 * d ∧ a₁₁ = a₁ + 10 * d

/-- Theorem stating the first term and common difference of the sequence -/
theorem arithmetic_sequence_solution :
  ∀ (a₃ a₁₁ : ℚ), a₃ = 3 ∧ a₁₁ = 15 →
  ArithmeticSequence a₃ a₁₁ →
  ∃ (a₁ d : ℚ), a₁ = 0 ∧ d = 3/2 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3730_373092


namespace NUMINAMATH_CALUDE_perpendicular_line_through_M_l3730_373028

-- Define the line l: 2x - y - 4 = 0
def line_l (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define point M as the intersection of line l with the x-axis
def point_M : ℝ × ℝ := (2, 0)

-- Define the perpendicular line: x + 2y - 2 = 0
def perp_line (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem statement
theorem perpendicular_line_through_M :
  (perp_line (point_M.1) (point_M.2)) ∧
  (∀ x y : ℝ, line_l x y → perp_line x y → 
    (y - point_M.2) * (x - point_M.1) = -(2 * (x - point_M.1) * (y - point_M.2))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_M_l3730_373028


namespace NUMINAMATH_CALUDE_shorter_side_length_l3730_373039

theorem shorter_side_length (a b : ℕ) : 
  a > b →                 -- Ensure a is the longer side
  2 * a + 2 * b = 48 →    -- Perimeter condition
  a * b = 140 →           -- Area condition
  b = 10 := by            -- Conclusion: shorter side is 10 feet
sorry

end NUMINAMATH_CALUDE_shorter_side_length_l3730_373039


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3730_373009

theorem inequality_solution_set (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3730_373009


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3730_373010

/-- Given a sphere with surface area 256π cm², its volume is 2048π/3 cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ,
  (4 : ℝ) * π * r^2 = 256 * π →
  (4 : ℝ) / 3 * π * r^3 = 2048 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3730_373010


namespace NUMINAMATH_CALUDE_joey_study_time_l3730_373061

/-- Calculates the total study time for Joey's SAT exam -/
def total_study_time (weekday_hours_per_night : ℕ) (weekday_nights : ℕ) 
  (weekend_hours_per_day : ℕ) (weekend_days : ℕ) (weeks_until_exam : ℕ) : ℕ :=
  ((weekday_hours_per_night * weekday_nights + weekend_hours_per_day * weekend_days) 
    * weeks_until_exam)

/-- Proves that Joey will spend 96 hours studying for his SAT exam -/
theorem joey_study_time : 
  total_study_time 2 5 3 2 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_joey_study_time_l3730_373061


namespace NUMINAMATH_CALUDE_square_park_area_l3730_373098

theorem square_park_area (side_length : ℝ) (h : side_length = 200) :
  side_length * side_length = 40000 := by
  sorry

end NUMINAMATH_CALUDE_square_park_area_l3730_373098


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3730_373065

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := a * b / 2
  let area2 := a * Real.sqrt (b^2 - a^2) / 2
  min area1 area2 = 6 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3730_373065


namespace NUMINAMATH_CALUDE_fifteen_times_fifteen_l3730_373046

theorem fifteen_times_fifteen : 
  ∀ n : ℕ, n = 15 → 15 * n = 225 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_times_fifteen_l3730_373046


namespace NUMINAMATH_CALUDE_number_1991_position_l3730_373025

/-- Represents a row in the number array -/
structure NumberArrayRow where
  startNumber : Nat
  length : Nat

/-- Defines the pattern of the number array -/
def numberArrayPattern (row : Nat) : NumberArrayRow :=
  { startNumber := row * 10,
    length := if row < 10 then row else 10 + (row - 10) * 10 }

/-- Checks if a number appears in a specific row and position -/
def appearsInRowAndPosition (n : Nat) (row : Nat) (position : Nat) : Prop :=
  let arrayRow := numberArrayPattern row
  n ≥ arrayRow.startNumber ∧ 
  n < arrayRow.startNumber + arrayRow.length ∧
  n = arrayRow.startNumber + position - 1

/-- Theorem stating that 1991 appears in the 199th row and 2nd position -/
theorem number_1991_position :
  appearsInRowAndPosition 1991 199 2 := by
  sorry


end NUMINAMATH_CALUDE_number_1991_position_l3730_373025


namespace NUMINAMATH_CALUDE_trapezoid_max_segment_length_l3730_373044

/-- Given a trapezoid with sum of bases equal to 4, the maximum length of a segment
    passing through the intersection of diagonals and parallel to bases is 2. -/
theorem trapezoid_max_segment_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (s : ℝ), s ≤ 2 ∧ 
  ∀ (t : ℝ), (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4),
    t = (2 * x * y) / (x + y)) → t ≤ s :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_max_segment_length_l3730_373044


namespace NUMINAMATH_CALUDE_jacks_total_money_l3730_373008

/-- Calculates the total amount of money in dollars given an amount in dollars and euros, with a fixed exchange rate. -/
def total_money_in_dollars (dollars : ℕ) (euros : ℕ) (exchange_rate : ℕ) : ℕ :=
  dollars + euros * exchange_rate

/-- Theorem stating that Jack's total money in dollars is 117 given the problem conditions. -/
theorem jacks_total_money :
  total_money_in_dollars 45 36 2 = 117 := by
  sorry

end NUMINAMATH_CALUDE_jacks_total_money_l3730_373008


namespace NUMINAMATH_CALUDE_polynomial_roots_l3730_373052

theorem polynomial_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 + 2*p*x^3 - x^2 + 2*p*x + 1 = 0 ∧ 
    y^4 + 2*p*y^3 - y^2 + 2*p*y + 1 = 0) ↔ 
  -3/4 ≤ p ∧ p ≤ -1/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3730_373052


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_equality_l3730_373013

theorem quadratic_inequality_implies_equality (x : ℝ) :
  -2 * x^2 + 5 * x - 2 > 0 →
  Real.sqrt (4 * x^2 - 4 * x + 1) + 2 * abs (x - 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_equality_l3730_373013


namespace NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3730_373069

theorem reciprocal_of_two_thirds : 
  (2 : ℚ) / 3 * (3 : ℚ) / 2 = 1 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_two_thirds_l3730_373069


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3730_373022

theorem polynomial_division_theorem :
  ∃ (α β r : ℝ), ∀ z : ℝ,
    4 * z^4 - 3 * z^3 + 5 * z^2 - 7 * z + 6 =
    (4 * z + 7) * (z^3 - 2.5 * z^2 + α * z + β) + r :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3730_373022


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l3730_373000

def total_cost (adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost : ℚ)
               (num_adults num_kids num_exclusive_dishes : ℕ)
               (discount_rate sales_tax_rate service_charge_rate : ℚ)
               (exclusive_dish_charge : ℚ) : ℚ :=
  let subtotal := adult_meal_cost * num_adults +
                  adult_drink_cost * num_adults +
                  kid_drink_cost * num_kids +
                  dessert_cost * (num_adults + num_kids) +
                  exclusive_dish_charge * num_exclusive_dishes
  let discounted_subtotal := subtotal * (1 - discount_rate)
  let with_tax := discounted_subtotal * (1 + sales_tax_rate)
  let final_total := with_tax * (1 + service_charge_rate)
  final_total

theorem restaurant_bill_calculation :
  let adult_meal_cost : ℚ := 12
  let adult_drink_cost : ℚ := 2.5
  let kid_drink_cost : ℚ := 1.5
  let dessert_cost : ℚ := 4
  let num_adults : ℕ := 7
  let num_kids : ℕ := 4
  let num_exclusive_dishes : ℕ := 3
  let discount_rate : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.075
  let service_charge_rate : ℚ := 0.15
  let exclusive_dish_charge : ℚ := 3
  total_cost adult_meal_cost adult_drink_cost kid_drink_cost dessert_cost
             num_adults num_kids num_exclusive_dishes
             discount_rate sales_tax_rate service_charge_rate
             exclusive_dish_charge = 178.57 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l3730_373000


namespace NUMINAMATH_CALUDE_bridge_building_time_l3730_373068

/-- If 60 workers can build a bridge in 8 days, then 40 workers can build the same bridge in 12 days, given that all workers work at the same rate. -/
theorem bridge_building_time 
  (work : ℝ) -- Total amount of work required to build the bridge
  (rate : ℝ) -- Rate of work per worker per day
  (h1 : work = 60 * rate * 8) -- 60 workers complete the bridge in 8 days
  (h2 : rate > 0) -- Workers have a positive work rate
  : work = 40 * rate * 12 := by
  sorry

end NUMINAMATH_CALUDE_bridge_building_time_l3730_373068


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_given_hcf_and_lcm_factors_l3730_373070

theorem sum_of_numbers_with_given_hcf_and_lcm_factors
  (a b : ℕ+)
  (h_hcf : Nat.gcd a b = 23)
  (h_lcm : Nat.lcm a b = 81328) :
  a + b = 667 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_given_hcf_and_lcm_factors_l3730_373070


namespace NUMINAMATH_CALUDE_john_smith_payment_l3730_373084

def number_of_cakes : ℕ := 3
def cost_per_cake : ℕ := 12
def number_of_people_splitting_cost : ℕ := 2

theorem john_smith_payment (total_cost : ℕ) (johns_share : ℕ) : 
  total_cost = number_of_cakes * cost_per_cake →
  johns_share = total_cost / number_of_people_splitting_cost →
  johns_share = 18 := by
sorry

end NUMINAMATH_CALUDE_john_smith_payment_l3730_373084


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l3730_373047

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l3730_373047


namespace NUMINAMATH_CALUDE_find_divisor_l3730_373064

theorem find_divisor : ∃ (x : ℕ), x > 0 ∧ 190 = 9 * x + 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_find_divisor_l3730_373064


namespace NUMINAMATH_CALUDE_point_A_coordinates_l3730_373043

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
def lies_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- The coordinates of point A as a function of x -/
def point_A (x : ℝ) : ℝ × ℝ := (2 - x, x + 3)

/-- Theorem stating that if point A lies on the x-axis, its coordinates are (5, 0) -/
theorem point_A_coordinates :
  ∃ x : ℝ, lies_on_x_axis (point_A x) → point_A x = (5, 0) := by
  sorry


end NUMINAMATH_CALUDE_point_A_coordinates_l3730_373043


namespace NUMINAMATH_CALUDE_problem_solution_l3730_373019

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2*a| + |x - a|

theorem problem_solution :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x : ℝ, f a x ≥ |x - 2*a| + |x - a|) →
  (∀ x : ℝ, (f 1 x > 2 ↔ x < 1/2 ∨ x > 5/2)) ∧
  (∀ b : ℝ, b ≠ 0 → f a b ≥ f a a) ∧
  (∀ b : ℝ, b ≠ 0 → (f a b = f a a ↔ (2*a - b)*(b - a) ≥ 0 ∨ 2*a - b = 0 ∨ b - a = 0)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3730_373019


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l3730_373034

/-- The equation |y-3| = √((x+4)² + (y-1)²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2) ↔ y = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l3730_373034


namespace NUMINAMATH_CALUDE_main_theorem_l3730_373041

/-- A nondecreasing function satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∧
  (∀ x y : ℝ, f (f x) + f y = f (x + f y) + 1)

/-- The set of all solutions to the functional equation. -/
def SolutionSet : Set (ℝ → ℝ) :=
  {f | FunctionalEquation f ∧
    (∀ x, f x = 1) ∨
    (∀ x, f x = x + 1) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌊n * x + α⌋ + 1)) ∨
    (∃ n : ℕ+, ∃ α : ℝ, 0 ≤ α ∧ α < 1 ∧ 
      (∀ x, f x = (1 / n) * ⌈n * x - α⌉ + 1))}

/-- The main theorem stating that the SolutionSet contains all solutions to the functional equation. -/
theorem main_theorem : ∀ f : ℝ → ℝ, FunctionalEquation f → f ∈ SolutionSet := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3730_373041


namespace NUMINAMATH_CALUDE_age_problem_l3730_373002

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 52) : 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3730_373002


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l3730_373058

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l3730_373058


namespace NUMINAMATH_CALUDE_three_numbers_sequence_l3730_373053

theorem three_numbers_sequence (x y z : ℝ) : 
  (x + y + z = 35 ∧ 
   2 * y = x + z + 1 ∧ 
   y^2 = (x + 3) * z) → 
  ((x = 15 ∧ y = 12 ∧ z = 8) ∨ 
   (x = 5 ∧ y = 12 ∧ z = 18)) := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sequence_l3730_373053


namespace NUMINAMATH_CALUDE_rice_weight_qualification_l3730_373090

def weight_range (x : ℝ) : Prop := 9.9 ≤ x ∧ x ≤ 10.1

theorem rice_weight_qualification :
  ¬(weight_range 9.09) ∧
  weight_range 9.99 ∧
  weight_range 10.01 ∧
  weight_range 10.09 :=
by sorry

end NUMINAMATH_CALUDE_rice_weight_qualification_l3730_373090


namespace NUMINAMATH_CALUDE_multiply_and_distribute_l3730_373072

theorem multiply_and_distribute (a b : ℝ) : -a * b * (-b + 1) = a * b^2 - a * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_distribute_l3730_373072


namespace NUMINAMATH_CALUDE_tim_stacked_bales_l3730_373004

theorem tim_stacked_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : final_bales = 82) :
  final_bales - initial_bales = 54 := by
  sorry

end NUMINAMATH_CALUDE_tim_stacked_bales_l3730_373004


namespace NUMINAMATH_CALUDE_line_contains_point_l3730_373088

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y,
    where k is a real number. The line contains the point (3,-2) if and only if k = -2. -/
theorem line_contains_point (k : ℝ) : 2 - k * 3 = -4 * (-2) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l3730_373088


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l3730_373074

theorem fraction_irreducibility (n : ℕ) : 
  Irreducible ((2 * n^2 + 11 * n - 18) / (n + 7)) ↔ n % 3 = 0 ∨ n % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l3730_373074


namespace NUMINAMATH_CALUDE_prob_three_primes_six_dice_l3730_373093

/-- The probability of rolling a prime number on a 10-sided die -/
def prob_prime_10 : ℚ := 2 / 5

/-- The probability of not rolling a prime number on a 10-sided die -/
def prob_not_prime_10 : ℚ := 3 / 5

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

theorem prob_three_primes_six_dice : 
  (choose_3_from_6 : ℚ) * prob_prime_10^3 * prob_not_prime_10^3 = 4320 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_primes_six_dice_l3730_373093


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l3730_373099

theorem subtraction_of_large_numbers :
  1000000000000 - 888777888777 = 111222111223 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l3730_373099


namespace NUMINAMATH_CALUDE_base5_divisible_by_31_l3730_373085

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Checks if a number is divisible by 31 --/
def isDivisibleBy31 (n : ℕ) : Prop := ∃ k : ℕ, n = 31 * k

/-- The main theorem --/
theorem base5_divisible_by_31 (x : ℕ) : 
  x < 5 → (isDivisibleBy31 (base5ToBase10 3 4 x 1) ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_base5_divisible_by_31_l3730_373085


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l3730_373031

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y, a * x + 2 * y + 1 = 0) →
  (∀ x y, x + y - 2 = 0) →
  (∀ x₁ y₁ x₂ y₂, a * x₁ + 2 * y₁ + 1 = 0 ∧ x₂ + y₂ - 2 = 0 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (y₂ - y₁)) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l3730_373031


namespace NUMINAMATH_CALUDE_factorial_multiple_of_eight_l3730_373038

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_multiple_of_eight (n : ℕ) :
  (∃ k : ℕ, factorial n = 8 * k) → n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_multiple_of_eight_l3730_373038
