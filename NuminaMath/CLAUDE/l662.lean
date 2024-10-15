import Mathlib

namespace NUMINAMATH_CALUDE_fraction_squared_times_32_equals_8_l662_66297

theorem fraction_squared_times_32_equals_8 : ∃ f : ℚ, f^2 * 32 = 2^3 ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_times_32_equals_8_l662_66297


namespace NUMINAMATH_CALUDE_max_puzzle_sets_l662_66273

/-- Represents the number of puzzles in a set -/
structure PuzzleSet where
  logic : ℕ
  visual : ℕ
  word : ℕ

/-- Checks if a PuzzleSet is valid according to the given conditions -/
def isValidSet (s : PuzzleSet) : Prop :=
  s.logic + s.visual + s.word ≥ 5 ∧ 2 * s.visual = s.logic

/-- The theorem to be proved -/
theorem max_puzzle_sets :
  ∀ (n : ℕ),
    (∃ (s : PuzzleSet),
      isValidSet s ∧
      n * s.logic ≤ 30 ∧
      n * s.visual ≤ 18 ∧
      n * s.word ≤ 12 ∧
      n * s.logic + n * s.visual + n * s.word = 30 + 18 + 12) →
    n ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_max_puzzle_sets_l662_66273


namespace NUMINAMATH_CALUDE_book_collection_ratio_l662_66241

theorem book_collection_ratio (first_week : ℕ) (total : ℕ) : 
  first_week = 9 → total = 99 → 
  (total - first_week) / first_week = 10 := by
sorry

end NUMINAMATH_CALUDE_book_collection_ratio_l662_66241


namespace NUMINAMATH_CALUDE_min_trig_expression_l662_66293

theorem min_trig_expression (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.sin x + 1 / Real.sin x)^3 + (Real.cos x + 1 / Real.cos x)^3 ≥ 729 * Real.sqrt 2 / 16 := by
  sorry

end NUMINAMATH_CALUDE_min_trig_expression_l662_66293


namespace NUMINAMATH_CALUDE_segment_sum_after_n_halvings_sum_after_million_halvings_l662_66254

/-- The sum of numbers on a segment after n halvings -/
def segmentSum (n : ℕ) : ℕ :=
  3^n + 1

/-- Theorem: The sum of numbers on a segment after n halvings is 3^n + 1 -/
theorem segment_sum_after_n_halvings (n : ℕ) :
  segmentSum n = 3^n + 1 := by
  sorry

/-- Corollary: The sum after one million halvings -/
theorem sum_after_million_halvings :
  segmentSum 1000000 = 3^1000000 + 1 := by
  sorry

end NUMINAMATH_CALUDE_segment_sum_after_n_halvings_sum_after_million_halvings_l662_66254


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l662_66245

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 4 / 3) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l662_66245


namespace NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l662_66290

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + 1

theorem derivative_even_implies_a_zero (a : ℝ) :
  (∀ x : ℝ, f_derivative a x = f_derivative a (-x)) →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_a_zero_l662_66290


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l662_66209

variables (a b x y : ℝ)

theorem complex_fraction_simplification :
  (a * x * (3 * a^2 * x^2 + 5 * b^2 * y^2) + b * y * (2 * a^2 * x^2 + 4 * b^2 * y^2)) / (a * x + b * y) = 3 * a^2 * x^2 + 4 * b^2 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l662_66209


namespace NUMINAMATH_CALUDE_red_crayons_per_person_l662_66263

def initial_rulers : ℕ := 11
def initial_crayons : ℕ := 34
def tim_added_rulers : ℕ := 14
def jane_removed_crayons : ℕ := 20
def jane_added_blue_crayons : ℕ := 8
def number_of_people : ℕ := 3

def total_red_crayons : ℕ := 2 * jane_added_blue_crayons

theorem red_crayons_per_person :
  total_red_crayons / number_of_people = 5 := by sorry

end NUMINAMATH_CALUDE_red_crayons_per_person_l662_66263


namespace NUMINAMATH_CALUDE_smallest_impossible_score_l662_66268

def dart_scores : Set ℕ := {0, 1, 3, 7, 8, 12}

def is_valid_sum (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ dart_scores ∧ b ∈ dart_scores ∧ c ∈ dart_scores ∧ a + b + c = n

theorem smallest_impossible_score :
  (∀ m : ℕ, m < 22 → is_valid_sum m) ∧ ¬is_valid_sum 22 :=
sorry

end NUMINAMATH_CALUDE_smallest_impossible_score_l662_66268


namespace NUMINAMATH_CALUDE_corn_planting_bags_used_l662_66237

/-- Represents the corn planting scenario with given conditions -/
structure CornPlanting where
  kids : ℕ
  earsPerRow : ℕ
  seedsPerEar : ℕ
  seedsPerBag : ℕ
  payPerRow : ℚ
  dinnerCost : ℚ

/-- Calculates the number of bags of corn seeds used by each kid -/
def bagsUsedPerKid (cp : CornPlanting) : ℚ :=
  let totalEarned := 2 * cp.dinnerCost
  let rowsPlanted := totalEarned / cp.payPerRow
  let seedsPerRow := cp.earsPerRow * cp.seedsPerEar
  let totalSeeds := rowsPlanted * seedsPerRow
  totalSeeds / cp.seedsPerBag

/-- Theorem stating that each kid used 140 bags of corn seeds -/
theorem corn_planting_bags_used
  (cp : CornPlanting)
  (h1 : cp.kids = 4)
  (h2 : cp.earsPerRow = 70)
  (h3 : cp.seedsPerEar = 2)
  (h4 : cp.seedsPerBag = 48)
  (h5 : cp.payPerRow = 3/2)
  (h6 : cp.dinnerCost = 36) :
  bagsUsedPerKid cp = 140 := by
  sorry

end NUMINAMATH_CALUDE_corn_planting_bags_used_l662_66237


namespace NUMINAMATH_CALUDE_grocery_store_costs_l662_66235

/-- Grocery store daily operation costs problem -/
theorem grocery_store_costs (total_costs : ℝ) (employees_salary_ratio : ℝ) (delivery_costs_ratio : ℝ)
  (h1 : total_costs = 4000)
  (h2 : employees_salary_ratio = 2 / 5)
  (h3 : delivery_costs_ratio = 1 / 4) :
  total_costs - (employees_salary_ratio * total_costs + delivery_costs_ratio * (total_costs - employees_salary_ratio * total_costs)) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l662_66235


namespace NUMINAMATH_CALUDE_cube_edge_multiple_l662_66271

/-- The ratio of the volumes of two cubes -/
def volume_ratio : ℝ := 0.03703703703703703

/-- Theorem: If the ratio of the volume of cube Q to the volume of cube P is 0.03703703703703703,
    and the length of an edge of cube P is some multiple k of the length of an edge of cube Q,
    then k = 3. -/
theorem cube_edge_multiple (q p k : ℝ) (hq : q > 0) (hp : p > 0) (hk : k > 0)
  (h_edge : p = k * q) (h_volume : q^3 / p^3 = volume_ratio) : k = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_multiple_l662_66271


namespace NUMINAMATH_CALUDE_largest_z_value_l662_66236

theorem largest_z_value (x y z : ℝ) : 
  x + y + z = 5 → 
  x * y + y * z + x * z = 3 → 
  z ≤ 13/3 :=
by sorry

end NUMINAMATH_CALUDE_largest_z_value_l662_66236


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l662_66253

theorem largest_divisor_of_five_consecutive_integers (n : ℤ) :
  ∃ (k : ℤ), k * 60 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ (m : ℤ), m > 60 → ¬∃ (j : ℤ), j * m = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l662_66253


namespace NUMINAMATH_CALUDE_workshop_average_age_l662_66283

theorem workshop_average_age 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (num_elderly : ℕ) (avg_age_elderly : ℝ)
  (h1 : num_females = 8)
  (h2 : avg_age_females = 34)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 32)
  (h5 : num_elderly = 5)
  (h6 : avg_age_elderly = 60) :
  let total_people := num_females + num_males + num_elderly
  let total_age := num_females * avg_age_females + num_males * avg_age_males + num_elderly * avg_age_elderly
  total_age / total_people = 38.24 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_age_l662_66283


namespace NUMINAMATH_CALUDE_race_earnings_theorem_l662_66232

/-- Represents the race parameters and results -/
structure RaceData where
  duration : ℕ         -- Race duration in minutes
  lap_distance : ℕ     -- Distance of one lap in meters
  certificate_rate : ℚ -- Gift certificate rate in dollars per 100 meters
  winner_laps : ℕ      -- Number of laps run by the winner

/-- Calculates the average earnings per minute for the race winner -/
def average_earnings_per_minute (data : RaceData) : ℚ :=
  (data.winner_laps * data.lap_distance * data.certificate_rate) / (100 * data.duration)

/-- Theorem stating that for the given race conditions, the average earnings per minute is $7 -/
theorem race_earnings_theorem (data : RaceData) 
  (h1 : data.duration = 12)
  (h2 : data.lap_distance = 100)
  (h3 : data.certificate_rate = 7/2)
  (h4 : data.winner_laps = 24) :
  average_earnings_per_minute data = 7 := by
  sorry

end NUMINAMATH_CALUDE_race_earnings_theorem_l662_66232


namespace NUMINAMATH_CALUDE_simplified_fraction_value_l662_66226

theorem simplified_fraction_value (k : ℝ) : 
  ∃ (a b : ℤ), (10 * k + 15) / 5 = a * k + b → a / b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_value_l662_66226


namespace NUMINAMATH_CALUDE_percentage_comparisons_l662_66276

theorem percentage_comparisons (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  (x / y) * 100 = 80 ∧
  ((y - x) / x) * 100 = 25 ∧
  ((y - x) / y) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_comparisons_l662_66276


namespace NUMINAMATH_CALUDE_inequality_proof_l662_66292

theorem inequality_proof (a b : ℝ) : (6*a - 3*b - 3) * (a^2 + a^2*b - 2*a^3) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l662_66292


namespace NUMINAMATH_CALUDE_expression_evaluation_l662_66217

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  5 * (a^2 + b) - 2 * (b + 2 * a^2) + 2 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l662_66217


namespace NUMINAMATH_CALUDE_minimum_buses_required_l662_66243

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 535) (h2 : bus_capacity = 45) :
  ∃ (num_buses : ℕ), num_buses * bus_capacity ≥ total_students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ num_buses ∧
  num_buses = 12 :=
sorry

end NUMINAMATH_CALUDE_minimum_buses_required_l662_66243


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l662_66265

theorem binomial_floor_divisibility (p n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_p : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) :=
sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l662_66265


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l662_66288

theorem consecutive_integer_averages (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (7 * c + 21) / 7) →
  ((7 * d + 21) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l662_66288


namespace NUMINAMATH_CALUDE_equality_multiplication_l662_66286

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end NUMINAMATH_CALUDE_equality_multiplication_l662_66286


namespace NUMINAMATH_CALUDE_rationalize_denominator_l662_66285

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = A + B * Real.sqrt C ∧ A = -2 ∧ B = -1 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l662_66285


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l662_66242

/-- Calculates the total cost of concert tickets --/
def concertTicketCost (generalAdmissionPrice : ℚ) (vipPrice : ℚ) (premiumPrice : ℚ)
                      (generalAdmissionQuantity : ℕ) (vipQuantity : ℕ) (premiumQuantity : ℕ)
                      (generalAdmissionDiscount : ℚ) (vipDiscount : ℚ) : ℚ :=
  let generalAdmissionCost := generalAdmissionPrice * generalAdmissionQuantity * (1 - generalAdmissionDiscount)
  let vipCost := vipPrice * vipQuantity * (1 - vipDiscount)
  let premiumCost := premiumPrice * premiumQuantity
  generalAdmissionCost + vipCost + premiumCost

theorem concert_ticket_cost :
  concertTicketCost 6 10 15 6 2 1 (1/10) (3/20) = 644/10 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l662_66242


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l662_66249

theorem sin_2alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π/4) π) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (π/4 - α)) : 
  Real.sin (2 * α) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l662_66249


namespace NUMINAMATH_CALUDE_polynomial_expansion_l662_66275

theorem polynomial_expansion (x : ℝ) :
  (7 * x + 5) * (3 * x^2 - 2 * x + 4) = 21 * x^3 + x^2 + 18 * x + 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l662_66275


namespace NUMINAMATH_CALUDE_a_formula_a_2_2_l662_66225

/-- The number of ordered subset groups with empty intersection -/
def a (i j : ℕ+) : ℕ :=
  (2^j.val - 1)^i.val

/-- The theorem stating the formula for a(i,j) -/
theorem a_formula (i j : ℕ+) :
  a i j = (Finset.univ.filter (fun s : Finset (Fin i.val) => s.card > 0)).card ^ j.val :=
by sorry

/-- Specific case for a(2,2) -/
theorem a_2_2 : a 2 2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_a_formula_a_2_2_l662_66225


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l662_66278

/-- Given a geometric sequence {a_n} with common ratio q and a_2 = 8, prove that a_6 = 128 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 2 = 8) 
  (h3 : ∀ n : ℕ, a (n + 1) = q * a n) : a 6 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l662_66278


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l662_66246

theorem stones_per_bracelet (total_stones : Float) (num_bracelets : Float) 
  (h1 : total_stones = 88.0) 
  (h2 : num_bracelets = 8.0) : 
  total_stones / num_bracelets = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l662_66246


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l662_66284

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 5 * p * (2^(p+1) - 1) = k^2 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l662_66284


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l662_66289

/-- The average speed of a round trip where:
    - The total distance is 2m meters (m meters each way)
    - The northward journey is at 3 minutes per mile
    - The southward journey is at 3 miles per minute
    - 1 mile = 1609.34 meters
-/
theorem round_trip_average_speed (m : ℝ) :
  let meters_per_mile : ℝ := 1609.34
  let north_speed : ℝ := 1 / 3 -- miles per minute
  let south_speed : ℝ := 3 -- miles per minute
  let total_distance : ℝ := 2 * m / meters_per_mile -- in miles
  let north_time : ℝ := m / (meters_per_mile * north_speed) -- in minutes
  let south_time : ℝ := m / (meters_per_mile * south_speed) -- in minutes
  let total_time : ℝ := north_time + south_time -- in minutes
  let average_speed : ℝ := total_distance / (total_time / 60) -- in miles per hour
  average_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l662_66289


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l662_66229

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 - 3*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 3

theorem cubic_function_extrema_difference (a b : ℝ) :
  f' a (-1) = 0 →
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧ 
    f a b x_max - f a b x_min = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l662_66229


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l662_66216

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 190*x - 7800

/-- The optimal selling price that maximizes profit -/
def optimal_price : ℝ := 95

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, 60 ≤ x ∧ x ≤ 130 → profit_function x ≤ profit_function optimal_price :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_l662_66216


namespace NUMINAMATH_CALUDE_non_attacking_knights_count_l662_66233

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Checks if two positions are distinct --/
def are_distinct (p1 p2 : Position) : Prop :=
  p1 ≠ p2

/-- Calculates the square of the distance between two positions --/
def distance_squared (p1 p2 : Position) : Nat :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if two knights attack each other --/
def knights_attack (p1 p2 : Position) : Prop :=
  distance_squared p1 p2 = 5

/-- Counts the number of ways to place two knights that do not attack each other --/
def count_non_attacking_placements (board : Chessboard) : Nat :=
  sorry

theorem non_attacking_knights_count :
  ∀ (board : Chessboard),
    board.size = 8 →
    count_non_attacking_placements board = 1848 :=
by sorry

end NUMINAMATH_CALUDE_non_attacking_knights_count_l662_66233


namespace NUMINAMATH_CALUDE_system_solution_l662_66255

theorem system_solution (x y : ℝ) (h1 : 2*x + 3*y = 5) (h2 : 3*x + 2*y = 10) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l662_66255


namespace NUMINAMATH_CALUDE_wire_cutting_l662_66212

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l662_66212


namespace NUMINAMATH_CALUDE_factor_expression_l662_66227

theorem factor_expression (x : ℝ) : 63 * x + 42 = 21 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l662_66227


namespace NUMINAMATH_CALUDE_f_of_2_eq_6_l662_66281

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- Theorem: f(2) = 6 -/
theorem f_of_2_eq_6 : f 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_eq_6_l662_66281


namespace NUMINAMATH_CALUDE_equation_not_equivalent_to_expression_with_unknown_l662_66248

-- Define what an expression is
def Expression : Type := Unit

-- Define what an unknown is
def Unknown : Type := Unit

-- Define what an equation is
def Equation : Type := Unit

-- Define a property for expressions that contain unknowns
def contains_unknown (e : Expression) : Prop := sorry

-- Define the property that an equation contains unknowns
axiom equation_contains_unknown : ∀ (eq : Equation), ∃ (u : Unknown), contains_unknown eq

-- Theorem to prove
theorem equation_not_equivalent_to_expression_with_unknown : 
  ¬(∀ (e : Expression), contains_unknown e → ∃ (eq : Equation), e = eq) :=
sorry

end NUMINAMATH_CALUDE_equation_not_equivalent_to_expression_with_unknown_l662_66248


namespace NUMINAMATH_CALUDE_same_gender_leaders_count_l662_66228

/-- Represents the number of ways to select a captain and co-captain of the same gender
    from a team with an equal number of men and women. -/
def select_same_gender_leaders (team_size : ℕ) : ℕ :=
  2 * (team_size * (team_size - 1))

/-- Theorem: In a team of 12 men and 12 women, there are 264 ways to select
    a captain and co-captain of the same gender. -/
theorem same_gender_leaders_count :
  select_same_gender_leaders 12 = 264 := by
  sorry

#eval select_same_gender_leaders 12

end NUMINAMATH_CALUDE_same_gender_leaders_count_l662_66228


namespace NUMINAMATH_CALUDE_equation_rearrangement_l662_66299

theorem equation_rearrangement (s P k c n : ℝ) 
  (h : P = s / ((1 + k)^n + c)) 
  (h_pos : s > 0) 
  (h_k_pos : k > -1) 
  (h_P_pos : P > 0) 
  (h_denom_pos : (s/P) - c > 0) : 
  n = (Real.log ((s/P) - c)) / (Real.log (1 + k)) := by
sorry

end NUMINAMATH_CALUDE_equation_rearrangement_l662_66299


namespace NUMINAMATH_CALUDE_expression_simplification_l662_66231

theorem expression_simplification (x : ℝ) (h : x = 4) :
  (x^2 - 4*x + 4) / (x^2 - 1) / (1 - 3 / (x + 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l662_66231


namespace NUMINAMATH_CALUDE_lateral_surface_area_rotated_unit_square_l662_66247

/-- The lateral surface area of a cylinder formed by rotating a square with area 1 around one of its sides. -/
theorem lateral_surface_area_rotated_unit_square : 
  ∀ (square_area : ℝ) (cylinder_height : ℝ) (cylinder_base_circumference : ℝ),
    square_area = 1 →
    cylinder_height = Real.sqrt square_area →
    cylinder_base_circumference = Real.sqrt square_area →
    cylinder_height * cylinder_base_circumference = 1 := by
  sorry

#check lateral_surface_area_rotated_unit_square

end NUMINAMATH_CALUDE_lateral_surface_area_rotated_unit_square_l662_66247


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l662_66287

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

theorem two_digit_reverse_sum (n : ℕ) :
  is_two_digit n →
  (n : ℤ) - (reverse_digits n : ℤ) = 7 * ((n / 10 : ℤ) + (n % 10 : ℤ)) →
  (n : ℕ) + reverse_digits n = 99 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l662_66287


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l662_66222

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 80)
  (h3 : S = a / (1 - r)) :
  a = 60 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l662_66222


namespace NUMINAMATH_CALUDE_hyperbola_equation_l662_66213

/-- The trajectory of a point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4, 0) and F₂(4, 0) are fixed points -/
def hyperbola_trajectory (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  d P F₂ - d P F₁ = 4 →
  P.1^2 / 4 - P.2^2 / 12 = 1 ∧ P.1 ≤ -2

theorem hyperbola_equation : 
  ∀ P : ℝ × ℝ, hyperbola_trajectory P :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l662_66213


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l662_66272

/-- Represents a circle placement arrangement in a square -/
structure CircleArrangement where
  n : ℕ  -- Side length of the square
  num_circles : ℕ  -- Number of circles placed

/-- Checks if a circle arrangement is valid -/
def is_valid_arrangement (arr : CircleArrangement) : Prop :=
  arr.n ≥ 8 ∧ arr.num_circles > arr.n^2

/-- Theorem stating the existence of a valid circle arrangement -/
theorem exists_valid_arrangement : 
  ∃ (arr : CircleArrangement), is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l662_66272


namespace NUMINAMATH_CALUDE_solution_to_system_l662_66244

theorem solution_to_system (x y : ℝ) : 
  x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2 →
  (1 / x - 1 / (2 * y) = 2 * y^4 - 2 * x^4) ∧
  (1 / x + 1 / (2 * y) = (3 * x^2 + y^2) * (x^2 + 3 * y^2)) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l662_66244


namespace NUMINAMATH_CALUDE_hexagon_not_possible_after_cut_l662_66218

-- Define a polygon
structure Polygon :=
  (sides : ℕ)
  (sides_ge_3 : sides ≥ 3)

-- Define the operation of cutting off a corner
def cut_corner (p : Polygon) : Polygon :=
  ⟨p.sides - 1, by sorry⟩

-- Theorem statement
theorem hexagon_not_possible_after_cut (p : Polygon) :
  (cut_corner p).sides = 4 → p.sides ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_not_possible_after_cut_l662_66218


namespace NUMINAMATH_CALUDE_range_of_a_l662_66214

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ∧
  (∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y) →
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l662_66214


namespace NUMINAMATH_CALUDE_median_salary_is_manager_salary_l662_66204

/-- Represents a job position with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company -/
def companyPositions : List Position :=
  [{ title := "CEO", count := 1, salary := 140000 },
   { title := "Senior Manager", count := 4, salary := 95000 },
   { title := "Manager", count := 13, salary := 78000 },
   { title := "Assistant Manager", count := 7, salary := 55000 },
   { title := "Clerk", count := 38, salary := 25000 }]

/-- The total number of employees in the company -/
def totalEmployees : Nat :=
  companyPositions.foldl (fun acc pos => acc + pos.count) 0

theorem median_salary_is_manager_salary :
  medianSalary companyPositions = 78000 ∧ totalEmployees = 63 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_manager_salary_l662_66204


namespace NUMINAMATH_CALUDE_arrangement_theorem_l662_66203

def number_of_arrangements (n m : ℕ) : ℕ := Nat.choose n m * Nat.factorial m

theorem arrangement_theorem : number_of_arrangements 6 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l662_66203


namespace NUMINAMATH_CALUDE_library_fine_fifth_day_l662_66219

def fine_calculation (initial_fine : Float) (increase : Float) (days : Nat) : Float :=
  let rec calc_fine (current_fine : Float) (day : Nat) : Float :=
    if day = 0 then
      current_fine
    else
      let increased := current_fine + increase
      let doubled := current_fine * 2
      calc_fine (min increased doubled) (day - 1)
  calc_fine initial_fine days

theorem library_fine_fifth_day :
  fine_calculation 0.07 0.30 4 = 0.86 := by
  sorry

end NUMINAMATH_CALUDE_library_fine_fifth_day_l662_66219


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l662_66252

theorem sum_of_cubes_and_cube_of_sum : (5 + 7)^3 + (5^3 + 7^3) = 2196 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l662_66252


namespace NUMINAMATH_CALUDE_root_power_equality_l662_66223

theorem root_power_equality (x₀ : ℝ) (h : x₀^11 + x₀^7 + x₀^3 = 1) :
  x₀^4 + x₀^3 - 1 = x₀^15 := by
  sorry

end NUMINAMATH_CALUDE_root_power_equality_l662_66223


namespace NUMINAMATH_CALUDE_exists_n_no_rational_solution_l662_66260

-- Define a quadratic polynomial with real coefficients
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

-- State the theorem
theorem exists_n_no_rational_solution (a b c : ℝ) :
  ∃ n : ℕ, ∀ x : ℚ, QuadraticPolynomial a b c x ≠ (1 : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_exists_n_no_rational_solution_l662_66260


namespace NUMINAMATH_CALUDE_biased_coin_probability_l662_66250

theorem biased_coin_probability : ∃ (h : ℝ),
  (0 < h ∧ h < 1) ∧
  (Nat.choose 6 2 * h^2 * (1-h)^4 = Nat.choose 6 3 * h^3 * (1-h)^3) ∧
  (Nat.choose 6 4 * h^4 * (1-h)^2 = 19440 / 117649) :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l662_66250


namespace NUMINAMATH_CALUDE_line_slope_problem_l662_66261

/-- Given a line passing through points (-1, -4) and (5, k) with slope k, prove that k = 4/5 -/
theorem line_slope_problem (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l662_66261


namespace NUMINAMATH_CALUDE_bus_speed_problem_l662_66280

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := 30

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when both are driving west, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if they drove towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_problem :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l662_66280


namespace NUMINAMATH_CALUDE_men_in_second_group_l662_66211

/-- Given the conditions of the problem, prove that the number of men in the second group is 9 -/
theorem men_in_second_group : 
  let first_group_men : ℕ := 4
  let first_group_hours_per_day : ℕ := 10
  let first_group_earnings : ℕ := 1200
  let second_group_hours_per_day : ℕ := 6
  let second_group_earnings : ℕ := 1620
  let days_per_week : ℕ := 7
  
  ∃ (second_group_men : ℕ),
    second_group_men * second_group_hours_per_day * days_per_week * first_group_earnings = 
    first_group_men * first_group_hours_per_day * days_per_week * second_group_earnings ∧
    second_group_men = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_men_in_second_group_l662_66211


namespace NUMINAMATH_CALUDE_average_of_XYZ_l662_66294

theorem average_of_XYZ (X Y Z : ℝ) 
  (eq1 : 2001 * Z - 4002 * X = 8008)
  (eq2 : 2001 * Y + 5005 * X = 10010) : 
  (X + Y + Z) / 3 = 0.1667 * X + 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_XYZ_l662_66294


namespace NUMINAMATH_CALUDE_inequality_solution_set_l662_66298

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l662_66298


namespace NUMINAMATH_CALUDE_trig_identity_l662_66239

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l662_66239


namespace NUMINAMATH_CALUDE_x_equation_implies_a_plus_b_l662_66201

theorem x_equation_implies_a_plus_b (x : ℝ) (a b : ℕ+) :
  x^2 + 5*x + 5/x + 1/x^2 = 34 →
  x = a + Real.sqrt b →
  (a : ℝ) + b = 5 := by sorry

end NUMINAMATH_CALUDE_x_equation_implies_a_plus_b_l662_66201


namespace NUMINAMATH_CALUDE_g_sum_zero_l662_66215

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_zero_l662_66215


namespace NUMINAMATH_CALUDE_min_green_beads_l662_66266

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  total_sum : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads
    satisfying the given conditions is 27. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) :
  n.green ≥ 27 := by sorry

end NUMINAMATH_CALUDE_min_green_beads_l662_66266


namespace NUMINAMATH_CALUDE_dinas_crayons_l662_66230

theorem dinas_crayons (wanda_crayons : ℕ) (total_crayons : ℕ) (dina_crayons : ℕ) :
  wanda_crayons = 62 →
  total_crayons = 116 →
  total_crayons = wanda_crayons + dina_crayons + (dina_crayons - 2) →
  dina_crayons = 28 := by
  sorry

end NUMINAMATH_CALUDE_dinas_crayons_l662_66230


namespace NUMINAMATH_CALUDE_fourth_number_is_ten_l662_66206

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number_is_ten (a : ℕ → ℕ) 
  (h_seq : sequence_property a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_is_ten_l662_66206


namespace NUMINAMATH_CALUDE_binary_to_decimal_11110_l662_66259

/-- Converts a binary digit to its decimal value -/
def binaryToDecimal (digit : Nat) (position : Nat) : Nat :=
  digit * 2^position

/-- The binary representation of the number -/
def binaryNumber : List Nat := [1, 1, 1, 1, 0]

/-- The decimal representation of the binary number -/
def decimalRepresentation : Nat :=
  (List.enumFrom 0 binaryNumber).map (fun (pos, digit) => binaryToDecimal digit pos) |>.sum

/-- Theorem stating that the decimal representation of "11110" is 30 -/
theorem binary_to_decimal_11110 :
  decimalRepresentation = 30 := by sorry

end NUMINAMATH_CALUDE_binary_to_decimal_11110_l662_66259


namespace NUMINAMATH_CALUDE_race_track_width_proof_l662_66270

def inner_circumference : Real := 440
def outer_radius : Real := 84.02817496043394

theorem race_track_width_proof :
  let inner_radius := inner_circumference / (2 * Real.pi)
  let width := outer_radius - inner_radius
  ∃ ε > 0, abs (width - 14.021) < ε :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_proof_l662_66270


namespace NUMINAMATH_CALUDE_pop_spent_15_l662_66205

def cereal_spending (pop crackle snap : ℝ) : Prop :=
  pop + crackle + snap = 150 ∧
  snap = 2 * crackle ∧
  crackle = 3 * pop

theorem pop_spent_15 :
  ∃ (pop crackle snap : ℝ), cereal_spending pop crackle snap ∧ pop = 15 := by
  sorry

end NUMINAMATH_CALUDE_pop_spent_15_l662_66205


namespace NUMINAMATH_CALUDE_sum_of_numbers_leq_threshold_l662_66262

theorem sum_of_numbers_leq_threshold : 
  let numbers : List ℚ := [8/10, 1/2, 9/10]
  let threshold : ℚ := 4/10
  (numbers.filter (λ x => x ≤ threshold)).sum = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_leq_threshold_l662_66262


namespace NUMINAMATH_CALUDE_min_value_problem_l662_66267

theorem min_value_problem (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 75) : 
  y₁^2 + 2 * y₂^2 + 3 * y₃^2 ≥ 5625 / 29 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 2 * y₂'^2 + 3 * y₃'^2 = 5625 / 29 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 2 * y₁' + 3 * y₂' + 4 * y₃' = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l662_66267


namespace NUMINAMATH_CALUDE_area_of_triangle_is_5_l662_66258

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Define the triangle formed by the line and coordinate axes
def triangle_area : ℝ := 5

-- Theorem statement
theorem area_of_triangle_is_5 :
  triangle_area = 5 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_is_5_l662_66258


namespace NUMINAMATH_CALUDE_unhappy_no_skills_no_skills_purple_l662_66220

/-- Represents the properties of a snake --/
structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

/-- Tom's collection of snakes --/
def toms_snakes : Finset Snake := sorry

/-- The number of snakes in Tom's collection --/
axiom total_snakes : toms_snakes.card = 17

/-- The number of purple snakes --/
axiom purple_snakes : (toms_snakes.filter (fun s => s.purple)).card = 5

/-- All purple snakes are unhappy --/
axiom purple_unhappy : ∀ s ∈ toms_snakes, s.purple → ¬s.happy

/-- The number of happy snakes --/
axiom happy_snakes : (toms_snakes.filter (fun s => s.happy)).card = 7

/-- All happy snakes can add and subtract --/
axiom happy_skills : ∀ s ∈ toms_snakes, s.happy → s.can_add ∧ s.can_subtract

/-- No purple snakes can add or subtract --/
axiom purple_no_skills : ∀ s ∈ toms_snakes, s.purple → ¬s.can_add ∧ ¬s.can_subtract

theorem unhappy_no_skills :
  ∀ s ∈ toms_snakes, ¬s.happy → ¬s.can_add ∨ ¬s.can_subtract :=
sorry

theorem no_skills_purple :
  ∀ s ∈ toms_snakes, ¬s.can_add ∧ ¬s.can_subtract → s.purple :=
sorry

end NUMINAMATH_CALUDE_unhappy_no_skills_no_skills_purple_l662_66220


namespace NUMINAMATH_CALUDE_max_dot_product_in_triangle_l662_66240

theorem max_dot_product_in_triangle (A B C P : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BAC := Real.arccos ((AB^2 + AC^2 - (B.1 - C.1)^2 - (B.2 - C.2)^2) / (2 * AB * AC))
  let AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  let dot_product := PB.1 * PC.1 + PB.2 * PC.2
  AB = 3 ∧ AC = 4 ∧ BAC = π/3 ∧ AP = 2 →
  ∃ P_max : ℝ × ℝ, dot_product ≤ 10 + 2 * Real.sqrt 37 ∧
            ∃ P_actual : ℝ × ℝ, dot_product = 10 + 2 * Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_in_triangle_l662_66240


namespace NUMINAMATH_CALUDE_tuna_sales_problem_l662_66279

/-- The number of packs of tuna fish sold per hour during the peak season -/
def peak_packs_per_hour : ℕ := 6

/-- The price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- The number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- The additional revenue made during the high season compared to the low season, in dollars -/
def additional_revenue : ℕ := 1800

/-- The number of packs of tuna fish sold per hour during the low season -/
def low_season_packs : ℕ := 4

theorem tuna_sales_problem :
  peak_packs_per_hour * price_per_pack * hours_per_day =
  low_season_packs * price_per_pack * hours_per_day + additional_revenue := by
  sorry

end NUMINAMATH_CALUDE_tuna_sales_problem_l662_66279


namespace NUMINAMATH_CALUDE_stairs_distance_l662_66274

theorem stairs_distance (total_time speed_up speed_down : ℝ) 
  (h_total_time : total_time = 4)
  (h_speed_up : speed_up = 2)
  (h_speed_down : speed_down = 3)
  (h_distance_diff : ∀ d : ℝ, d / speed_up + (d + 2) / speed_down = total_time) :
  ∃ d : ℝ, d + 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_stairs_distance_l662_66274


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l662_66296

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l662_66296


namespace NUMINAMATH_CALUDE_marbles_per_customer_l662_66234

theorem marbles_per_customer 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 400) 
  (h2 : num_customers = 20) 
  (h3 : remaining_marbles = 100) :
  (initial_marbles - remaining_marbles) / num_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_marbles_per_customer_l662_66234


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_power_of_two_l662_66202

theorem binomial_coefficient_modulo_power_of_two 
  (n : ℕ) (r : ℕ) (h_r_odd : Odd r) :
  ∃ i : ℕ, i < 2^n ∧ Nat.choose (2^n + i) i ≡ r [MOD 2^(n+1)] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_power_of_two_l662_66202


namespace NUMINAMATH_CALUDE_complex_equation_solution_l662_66224

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l662_66224


namespace NUMINAMATH_CALUDE_pells_equation_unique_solution_l662_66256

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℕ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

-- The main theorem
theorem pells_equation_unique_solution (x₀ y₀ x y : ℕ) :
  fundamental_solution x₀ y₀ →
  x^2 - 2003 * y^2 = 1 →
  x > 0 →
  y > 0 →
  all_prime_factors_divide x x₀ →
  x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_pells_equation_unique_solution_l662_66256


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l662_66264

/-- Represents a rectangular grid with alternating painted columns and rows -/
structure PaintedGrid where
  rows : Nat
  cols : Nat
  unpaintedCells : Nat

/-- Checks if the grid dimensions are valid (odd number of rows and columns) -/
def PaintedGrid.isValid (grid : PaintedGrid) : Prop :=
  grid.rows % 2 = 1 ∧ grid.cols % 2 = 1

/-- Calculates the number of painted cells in the grid -/
def PaintedGrid.paintedCells (grid : PaintedGrid) : Nat :=
  grid.rows * grid.cols - grid.unpaintedCells

/-- Theorem: If a valid painted grid has 74 unpainted cells, 
    then the number of painted cells is either 301 or 373 -/
theorem painted_cells_theorem (grid : PaintedGrid) :
  grid.isValid ∧ grid.unpaintedCells = 74 →
  grid.paintedCells = 301 ∨ grid.paintedCells = 373 := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l662_66264


namespace NUMINAMATH_CALUDE_impossible_arrangement_l662_66269

theorem impossible_arrangement :
  ¬ ∃ (grid : Matrix (Fin 6) (Fin 7) ℕ),
    (∀ i j, grid i j ∈ Set.range (fun n => n + 1) ∩ Set.Icc 1 42) ∧
    (∀ i j, ∃! p, grid p j = grid i j) ∧
    (∀ i j, Even (grid i j + grid (i + 1) j)) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l662_66269


namespace NUMINAMATH_CALUDE_right_triangle_condition_l662_66221

theorem right_triangle_condition (a d : ℝ) (ha : a > 0) (hd : d > 1) :
  (a * d^2)^2 = a^2 + (a * d)^2 ↔ d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l662_66221


namespace NUMINAMATH_CALUDE_point_P_coordinates_l662_66238

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def is_on_segment (P M N : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • M + t • N

def vector_eq (P M N : ℝ × ℝ) : Prop :=
  (N.1 - P.1, N.2 - P.2) = (-2 * (M.1 - P.1), -2 * (M.2 - P.2))

theorem point_P_coordinates :
  ∀ P : ℝ × ℝ, is_on_segment P M N ∧ vector_eq P M N → P = (2, 4) :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l662_66238


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l662_66200

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f (x : ℝ) (h : x ≥ -1) :
  f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4 := by sorry

-- Define n as the minimum value of f(x)
def n : ℝ := 4

-- Theorem for the minimum value of 2a + b
theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * n * a * b = a + 2 * b) :
  2 * a + b ≥ 9/8 := by sorry

-- Theorem stating that 9/8 is indeed the minimum value
theorem min_value_2a_plus_b_is_9_8 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * n * a * b = a + 2 * b ∧ 2 * a + b = 9/8 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l662_66200


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l662_66210

-- Define a plane
variable (α : Set (ℝ × ℝ × ℝ))

-- Define a line
variable (l : Set (ℝ × ℝ × ℝ))

-- Define a predicate for a line being in a plane
def LineInPlane (line : Set (ℝ × ℝ × ℝ)) (plane : Set (ℝ × ℝ × ℝ)) : Prop :=
  line ⊆ plane

-- Define a predicate for two lines being perpendicular
def Perpendicular (line1 line2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Definition of perpendicularity

-- Theorem statement
theorem exists_perpendicular_line (α : Set (ℝ × ℝ × ℝ)) (l : Set (ℝ × ℝ × ℝ)) :
  ∃ m : Set (ℝ × ℝ × ℝ), LineInPlane m α ∧ Perpendicular m l :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l662_66210


namespace NUMINAMATH_CALUDE_infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l662_66257

theorem infinitely_many_n_factorial_divisible_by_n_cubed_minus_one :
  {n : ℕ+ | (n.val.factorial : ℤ) % (n.val ^ 3 - 1) = 0}.Infinite :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_factorial_divisible_by_n_cubed_minus_one_l662_66257


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l662_66251

theorem arithmetic_equalities : 
  (-20 + (-14) - (-18) - 13 = -29) ∧ 
  ((-2) * 3 + (-5) - 4 / (-1/2) = -3) ∧ 
  ((-3/8 - 1/6 + 3/4) * (-24) = -5) ∧ 
  (-81 / (9/4) * |(-4/9)| - (-3)^3 / 27 = -15) := by sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l662_66251


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_area_l662_66277

/-- Given a line passing through (a, 0) and intersecting the y-axis in the first quadrant,
    forming a triangular region with area T, prove that the equation of this line is
    2Tx + a²y - 2aT = 0 -/
theorem line_equation_from_triangle_area (a T : ℝ) (h_a : a ≠ 0) (h_T : T > 0) :
  ∃ (m b : ℝ),
    (∀ x y : ℝ, y = m * x + b → (x = a ∧ y = 0) ∨ (x = 0 ∧ y > 0)) ∧
    (1/2 * a * b = T) ∧
    (∀ x y : ℝ, 2 * T * x + a^2 * y - 2 * a * T = 0 ↔ y = m * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_area_l662_66277


namespace NUMINAMATH_CALUDE_inequality_proof_l662_66291

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (1 + Real.sqrt x)^2) + (1 / (1 + Real.sqrt y)^2) ≥ 2 / (x + y + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l662_66291


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l662_66208

/-- Represents the number of employees in each age group -/
structure AgeGroup where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of employees in each age group -/
structure AgeRatio where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample size for each age group -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (ratio : AgeRatio) : AgeGroup :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := sampleSize * ratio.middleAged / totalRatio,
    young := sampleSize * ratio.young / totalRatio,
    elderly := sampleSize * ratio.elderly / totalRatio }

theorem correct_stratified_sample :
  let totalPopulation : ℕ := 3200
  let sampleSize : ℕ := 400
  let ratio : AgeRatio := { middleAged := 5, young := 3, elderly := 2 }
  let sample : AgeGroup := stratifiedSample totalPopulation sampleSize ratio
  sample.middleAged = 200 ∧ sample.young = 120 ∧ sample.elderly = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_l662_66208


namespace NUMINAMATH_CALUDE_kyle_corn_purchase_l662_66295

-- Define the problem parameters
def total_pounds : ℝ := 30
def total_cost : ℝ := 22.50
def corn_price : ℝ := 1.05
def beans_price : ℝ := 0.55

-- Define the theorem
theorem kyle_corn_purchase :
  ∃ (corn beans : ℝ),
    corn + beans = total_pounds ∧
    corn_price * corn + beans_price * beans = total_cost ∧
    corn = 12 := by
  sorry

end NUMINAMATH_CALUDE_kyle_corn_purchase_l662_66295


namespace NUMINAMATH_CALUDE_quiz_correct_answers_l662_66282

theorem quiz_correct_answers (cherry kim nicole : ℕ) 
  (h1 : nicole + 3 = kim)
  (h2 : kim = cherry + 8)
  (h3 : cherry = 17) : 
  nicole = 22 := by
sorry

end NUMINAMATH_CALUDE_quiz_correct_answers_l662_66282


namespace NUMINAMATH_CALUDE_remove_one_gives_average_eight_point_five_l662_66207

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_one_gives_average_eight_point_five :
  average (remove_number original_list 1) = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_eight_point_five_l662_66207
