import Mathlib

namespace table_tennis_equation_l512_51208

/-- Represents a table tennis competition -/
structure TableTennisCompetition where
  teams : ℕ
  totalMatches : ℕ
  pairPlaysOneMatch : Bool

/-- The equation for the number of matches in a table tennis competition -/
def matchEquation (c : TableTennisCompetition) : Prop :=
  c.teams * (c.teams - 1) = c.totalMatches * 2

/-- Theorem stating the correct equation for the given competition conditions -/
theorem table_tennis_equation (c : TableTennisCompetition) 
  (h1 : c.pairPlaysOneMatch = true) 
  (h2 : c.totalMatches = 28) : 
  matchEquation c := by
  sorry

end table_tennis_equation_l512_51208


namespace prism_faces_count_l512_51289

/-- A prism is a polyhedron with two congruent polygonal bases and rectangular lateral faces. -/
structure Prism where
  /-- The number of sides in each base of the prism -/
  base_sides : ℕ
  /-- The number of vertices of the prism -/
  vertices : ℕ
  /-- The number of edges of the prism -/
  edges : ℕ
  /-- The number of faces of the prism -/
  faces : ℕ
  /-- The sum of vertices and edges is 40 -/
  sum_condition : vertices + edges = 40
  /-- The number of vertices is twice the number of base sides -/
  vertices_def : vertices = 2 * base_sides
  /-- The number of edges is thrice the number of base sides -/
  edges_def : edges = 3 * base_sides
  /-- The number of faces is 2 more than the number of base sides -/
  faces_def : faces = base_sides + 2

/-- Theorem: A prism with 40 as the sum of its edges and vertices has 10 faces -/
theorem prism_faces_count (p : Prism) : p.faces = 10 := by
  sorry


end prism_faces_count_l512_51289


namespace existence_of_abc_l512_51297

theorem existence_of_abc (n k : ℕ) (h1 : n > 20) (h2 : k > 1) (h3 : k^2 ∣ n) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a := by
sorry

end existence_of_abc_l512_51297


namespace rational_representation_l512_51294

theorem rational_representation (q : ℚ) (hq : q > 0) :
  ∃ (a b c d : ℕ+), q = (a^2021 + b^2023) / (c^2022 + d^2024) := by
  sorry

end rational_representation_l512_51294


namespace function_properties_imply_solution_set_l512_51275

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def satisfies_negation_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def matches_linear_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (1/2) * x

def solution_set (k : ℤ) : ℝ := 4 * k - 1

theorem function_properties_imply_solution_set 
  (f : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : satisfies_negation_property f) 
  (h3 : matches_linear_on_interval f) :
  ∀ x, f x = -(1/2) ↔ ∃ k : ℤ, x = solution_set k :=
sorry

end function_properties_imply_solution_set_l512_51275


namespace basketball_expected_score_l512_51287

def expected_score (p_in : ℝ) (p_out : ℝ) (n_in : ℕ) (n_out : ℕ) (points_in : ℕ) (points_out : ℕ) : ℝ :=
  (p_in * n_in * points_in) + (p_out * n_out * points_out)

theorem basketball_expected_score :
  expected_score 0.7 0.4 10 5 2 3 = 20 := by
  sorry

end basketball_expected_score_l512_51287


namespace black_tiles_201_implies_total_4624_l512_51259

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- Calculates the number of black tiles on the floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  3 * floor.side_length - 3

/-- Calculates the total number of tiles on the floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If there are 201 black tiles, then the total number of tiles is 4624 -/
theorem black_tiles_201_implies_total_4624 :
  ∀ (floor : SquareFloor), black_tiles floor = 201 → total_tiles floor = 4624 :=
by
  sorry

end black_tiles_201_implies_total_4624_l512_51259


namespace inequalities_properties_l512_51219

theorem inequalities_properties (a b : ℝ) (h : a < b ∧ b < 0) : 
  abs a > abs b ∧ 
  1 / a > 1 / b ∧ 
  a / b + b / a > 2 ∧ 
  a ^ 2 > b ^ 2 := by
sorry

end inequalities_properties_l512_51219


namespace bus_fraction_proof_l512_51273

def total_distance : ℝ := 30.000000000000007

theorem bus_fraction_proof :
  let distance_by_foot : ℝ := (1/3) * total_distance
  let distance_by_car : ℝ := 2
  let distance_by_bus : ℝ := total_distance - distance_by_foot - distance_by_car
  distance_by_bus / total_distance = 3/5 := by sorry

end bus_fraction_proof_l512_51273


namespace number_division_problem_l512_51239

theorem number_division_problem (x : ℚ) : x / 2 = 100 + x / 5 → x = 1000 / 3 := by
  sorry

end number_division_problem_l512_51239


namespace triangle_properties_l512_51274

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  2 * Real.cos (t.A + t.B) = -1

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = Real.pi / 3 ∧
  t.c = Real.sqrt 6 ∧
  (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 :=
sorry

end triangle_properties_l512_51274


namespace jelly_bracelet_cost_l512_51228

def friends : List String := ["Jessica", "Tori", "Lily", "Patrice"]

def total_spent : ℕ := 44

def total_bracelets : ℕ := (friends.map String.length).sum

theorem jelly_bracelet_cost :
  total_spent / total_bracelets = 2 := by sorry

end jelly_bracelet_cost_l512_51228


namespace greatest_five_digit_divisible_by_sum_of_digits_l512_51245

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Checks if a number is divisible by the sum of its digits -/
def isDivisibleBySumOfDigits (n : ℕ) : Prop :=
  n % sumOfDigits n = 0

/-- Theorem: 99972 is the greatest five-digit number divisible by the sum of its digits -/
theorem greatest_five_digit_divisible_by_sum_of_digits :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ isDivisibleBySumOfDigits n → n ≤ 99972 :=
by sorry

end greatest_five_digit_divisible_by_sum_of_digits_l512_51245


namespace total_earnings_l512_51291

/-- Calculate total earnings from selling candied apples and grapes -/
theorem total_earnings (num_apples : ℕ) (price_apple : ℚ) 
                       (num_grapes : ℕ) (price_grape : ℚ) : 
  num_apples = 15 → 
  price_apple = 2 → 
  num_grapes = 12 → 
  price_grape = (3/2) → 
  (num_apples : ℚ) * price_apple + (num_grapes : ℚ) * price_grape = 48 := by
sorry

end total_earnings_l512_51291


namespace repeating_decimal_subtraction_l512_51221

/-- Represents a repeating decimal with a 4-digit repetend -/
def RepeatingDecimal (a b c d : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + d : ℚ) / 9999

/-- The problem statement -/
theorem repeating_decimal_subtraction :
  RepeatingDecimal 4 5 6 7 - RepeatingDecimal 1 2 3 4 - RepeatingDecimal 2 3 4 5 = 988 / 9999 := by
  sorry

end repeating_decimal_subtraction_l512_51221


namespace library_book_increase_l512_51234

theorem library_book_increase (N : ℕ) : 
  N > 0 ∧ 
  (N * 1.004 * 1.008 : ℝ) < 50000 →
  ⌊(N * 1.004 * 1.008 - N * 1.004 : ℝ)⌋ = 251 :=
by sorry

end library_book_increase_l512_51234


namespace linear_function_m_range_l512_51258

/-- A linear function y = (m-1)x + (4m-3) whose graph lies in the first, second, and fourth quadrants -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + (4 * m - 3)

/-- The slope of the linear function is negative -/
def slope_negative (m : ℝ) : Prop := m - 1 < 0

/-- The y-intercept of the linear function is positive -/
def y_intercept_positive (m : ℝ) : Prop := 4 * m - 3 > 0

/-- The graph of the linear function lies in the first, second, and fourth quadrants -/
def graph_in_first_second_fourth_quadrants (m : ℝ) : Prop :=
  slope_negative m ∧ y_intercept_positive m

theorem linear_function_m_range :
  ∀ m : ℝ, graph_in_first_second_fourth_quadrants m → 3/4 < m ∧ m < 1 :=
by sorry

end linear_function_m_range_l512_51258


namespace shorter_worm_length_l512_51255

/-- Given two worms where one is 0.8 inches long and the other is 0.7 inches longer,
    prove that the length of the shorter worm is 0.8 inches. -/
theorem shorter_worm_length (worm1 worm2 : ℝ) 
  (h1 : worm1 = 0.8)
  (h2 : worm2 = worm1 + 0.7) :
  min worm1 worm2 = 0.8 := by
  sorry

end shorter_worm_length_l512_51255


namespace lisa_packing_peanuts_l512_51278

/-- The amount of packing peanuts needed for a large order in grams -/
def large_order_peanuts : ℕ := 200

/-- The amount of packing peanuts needed for a small order in grams -/
def small_order_peanuts : ℕ := 50

/-- The number of large orders Lisa has sent -/
def large_orders : ℕ := 3

/-- The number of small orders Lisa has sent -/
def small_orders : ℕ := 4

/-- The total amount of packing peanuts used by Lisa -/
def total_peanuts : ℕ := large_order_peanuts * large_orders + small_order_peanuts * small_orders

theorem lisa_packing_peanuts : total_peanuts = 800 := by
  sorry

end lisa_packing_peanuts_l512_51278


namespace temperature_difference_product_product_of_possible_P_values_l512_51283

theorem temperature_difference_product (P : ℝ) : 
  (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4) →
  (P = 17 ∨ P = 9) :=
by sorry

theorem product_of_possible_P_values : 
  (∃ P : ℝ, (∃ X D : ℝ, 
    X = D + P ∧ 
    |((D + P) - 8) - (D + 5)| = 4)) →
  17 * 9 = 153 :=
by sorry

end temperature_difference_product_product_of_possible_P_values_l512_51283


namespace complex_equation_solution_l512_51280

theorem complex_equation_solution (x y : ℝ) (i : ℂ) (h : i * i = -1) :
  (2 * x - 1 : ℂ) + i = y - (3 - y) * i →
  x = 5 / 2 ∧ y = 4 := by
  sorry

end complex_equation_solution_l512_51280


namespace smallest_sum_reciprocals_l512_51256

theorem smallest_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 13) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 13 ∧ a + b = 196 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 13 → c + d ≥ 196 :=
sorry

end smallest_sum_reciprocals_l512_51256


namespace ellipse_eccentricity_l512_51237

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    if the triangle formed by its left vertex, right vertex, and top vertex
    is an isosceles triangle with base angle 30°, then its eccentricity is √6/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (b / a = Real.sqrt 3 / 3) → 
  Real.sqrt (1 - (b / a)^2) = Real.sqrt 6 / 3 := by
  sorry

end ellipse_eccentricity_l512_51237


namespace strawberries_in_buckets_l512_51267

theorem strawberries_in_buckets (total_strawberries : ℕ) (num_buckets : ℕ) (removed_per_bucket : ℕ) :
  total_strawberries = 300 →
  num_buckets = 5 →
  removed_per_bucket = 20 →
  (total_strawberries / num_buckets) - removed_per_bucket = 40 :=
by
  sorry

end strawberries_in_buckets_l512_51267


namespace train_tickets_theorem_l512_51266

/-- Calculates the number of different tickets needed for a train route -/
def number_of_tickets (intermediate_stops : ℕ) : ℕ :=
  intermediate_stops * (intermediate_stops + 3) + 2

/-- Theorem stating that a train route with 5 intermediate stops requires 42 different tickets -/
theorem train_tickets_theorem :
  number_of_tickets 5 = 42 := by
  sorry

end train_tickets_theorem_l512_51266


namespace circles_configuration_l512_51265

/-- Two circles with radii r₁ and r₂, and distance d between their centers,
    are in the "one circle inside the other" configuration if d < |r₁ - r₂| -/
def CircleInsideOther (r₁ r₂ d : ℝ) : Prop :=
  d < |r₁ - r₂|

/-- Given two circles with radii 1 and 5, and distance 3 between their centers,
    prove that one circle is inside the other -/
theorem circles_configuration :
  CircleInsideOther 1 5 3 := by
sorry

end circles_configuration_l512_51265


namespace binomial_constant_term_l512_51272

/-- The constant term in the binomial expansion of (x - a/(3x))^8 -/
def constantTerm (a : ℝ) : ℝ := ((-1)^6 * a^6) * (Nat.choose 8 6)

theorem binomial_constant_term (a : ℝ) : 
  constantTerm a = 28 → a = 1 ∨ a = -1 := by
  sorry

end binomial_constant_term_l512_51272


namespace cost_price_calculation_l512_51285

theorem cost_price_calculation (marked_price : ℝ) (selling_price_percent : ℝ) (profit_percent : ℝ) :
  marked_price = 62.5 →
  selling_price_percent = 0.95 →
  profit_percent = 1.25 →
  ∃ (cost_price : ℝ), cost_price = 47.5 ∧ 
    selling_price_percent * marked_price = profit_percent * cost_price :=
by
  sorry

end cost_price_calculation_l512_51285


namespace solution_set_equality_l512_51246

theorem solution_set_equality : 
  {x : ℝ | (x - 3) * (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end solution_set_equality_l512_51246


namespace earn_twelve_points_l512_51288

/-- Calculates the points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_not_defeated : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_not_defeated) * points_per_enemy

/-- Theorem: In the given scenario, the player earns 12 points --/
theorem earn_twelve_points :
  points_earned 6 2 3 = 12 := by
  sorry

end earn_twelve_points_l512_51288


namespace area_increase_l512_51293

/-- A rectangle with perimeter 160 meters -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 80

/-- The change in area when increasing both sides by 10 meters -/
def area_change (rect : Rectangle) : ℝ :=
  (rect.length + 10) * (rect.width + 10) - rect.length * rect.width

theorem area_increase (rect : Rectangle) : area_change rect = 900 := by
  sorry

end area_increase_l512_51293


namespace yellow_parrots_count_l512_51281

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (yellow_fraction : ℚ) : 
  total = 120 →
  red_fraction = 2/3 →
  yellow_fraction = 1 - red_fraction →
  (yellow_fraction * total : ℚ) = 40 := by
  sorry

end yellow_parrots_count_l512_51281


namespace complex_product_real_solutions_l512_51214

theorem complex_product_real_solutions (x : ℝ) : 
  (Complex.I : ℂ) * ((x + Complex.I) * ((x + 3 : ℝ) + 2 * Complex.I) * ((x + 5 : ℝ) - Complex.I)).im = 0 ↔ 
  x = -1.5 ∨ x = -5 := by
sorry

end complex_product_real_solutions_l512_51214


namespace markers_problem_l512_51232

theorem markers_problem (initial_markers : ℕ) (markers_per_box : ℕ) (final_markers : ℕ) :
  initial_markers = 32 →
  markers_per_box = 9 →
  final_markers = 86 →
  (final_markers - initial_markers) / markers_per_box = 6 :=
by
  sorry

end markers_problem_l512_51232


namespace stratified_sampling_grade10_l512_51224

theorem stratified_sampling_grade10 (total_students : ℕ) (sample_size : ℕ) (grade10_in_sample : ℕ) :
  total_students = 1800 →
  sample_size = 90 →
  grade10_in_sample = 42 →
  (grade10_in_sample : ℚ) / (sample_size : ℚ) = (840 : ℚ) / (total_students : ℚ) :=
by sorry

end stratified_sampling_grade10_l512_51224


namespace events_mutually_exclusive_not_complementary_l512_51298

-- Define the sample space
def sampleSpace : ℕ := 10 -- (5 choose 2)

-- Define the events
def exactlyOneMale (outcome : ℕ) : Prop := sorry
def exactlyTwoFemales (outcome : ℕ) : Prop := sorry

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

-- Define complementary events
def complementary (e1 e2 : ℕ → Prop) : Prop :=
  ∀ outcome, e1 outcome ↔ ¬(e2 outcome)

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneMale exactlyTwoFemales ∧
  ¬(complementary exactlyOneMale exactlyTwoFemales) := by
  sorry

end events_mutually_exclusive_not_complementary_l512_51298


namespace negation_of_every_scientist_is_curious_l512_51215

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for being a scientist and being curious
variable (scientist : U → Prop)
variable (curious : U → Prop)

-- State the theorem
theorem negation_of_every_scientist_is_curious :
  (¬ ∀ x, scientist x → curious x) ↔ (∃ x, scientist x ∧ ¬ curious x) :=
sorry

end negation_of_every_scientist_is_curious_l512_51215


namespace expression_equality_l512_51242

theorem expression_equality : 4⁻¹ - Real.sqrt (1/16) + (3 - Real.sqrt 2)^0 = 1 := by sorry

end expression_equality_l512_51242


namespace boys_age_problem_l512_51279

theorem boys_age_problem (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 29 →
  age1 = age2 →
  age3 = 11 →
  age1 = 9 ∧ age2 = 9 := by
sorry

end boys_age_problem_l512_51279


namespace spice_difference_total_l512_51243

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.35

theorem spice_difference_total : 
  abs (cinnamon - nutmeg) + abs (nutmeg - ginger) + abs (cinnamon - ginger) = 0.64 := by
  sorry

end spice_difference_total_l512_51243


namespace point_not_in_transformed_plane_l512_51263

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def applySimiliarity (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies a plane equation -/
def satisfiesPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Main theorem: Point A does not belong to the image of plane a under the similarity transformation -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := 5, y := 0, z := -6 }
  let a : Plane := { a := 6, b := -1, c := -1, d := 7 }
  let k : ℝ := 2/7
  let a_transformed := applySimiliarity a k
  ¬ satisfiesPlane A a_transformed :=
by sorry

end point_not_in_transformed_plane_l512_51263


namespace square_root_difference_l512_51231

theorem square_root_difference : Real.sqrt (49 + 36) - Real.sqrt (36 - 0) = 4 := by
  sorry

end square_root_difference_l512_51231


namespace calculation_proof_l512_51227

theorem calculation_proof : -1^4 - (1 - 0.5) * (2 - (-3)^2) = 5/2 := by
  sorry

end calculation_proof_l512_51227


namespace anna_bob_matches_l512_51213

/-- The number of players in the chess tournament -/
def total_players : ℕ := 12

/-- The number of players in each match -/
def players_per_match : ℕ := 6

/-- The number of players to choose after Anna and Bob are selected -/
def players_to_choose : ℕ := players_per_match - 2

/-- The number of remaining players after Anna and Bob are selected -/
def remaining_players : ℕ := total_players - 2

/-- The number of matches where Anna and Bob play together -/
def matches_together : ℕ := Nat.choose remaining_players players_to_choose

theorem anna_bob_matches :
  matches_together = 210 := by sorry

end anna_bob_matches_l512_51213


namespace floor_plus_self_unique_solution_l512_51241

theorem floor_plus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ + s = 20.75 := by
  sorry

end floor_plus_self_unique_solution_l512_51241


namespace village_population_l512_51229

/-- If 80% of a village's population is 64,000, then the total population is 80,000. -/
theorem village_population (population : ℕ) (h : (80 : ℕ) * population = 100 * 64000) :
  population = 80000 := by
  sorry

end village_population_l512_51229


namespace gcd_consecutive_pairs_l512_51299

theorem gcd_consecutive_pairs (m n : ℕ) (h : m > n) :
  (∀ k : ℕ, k ∈ Finset.range (m - n) → Nat.gcd (n + k + 1) (m + k + 1) = 1) ↔ m = n + 1 := by
  sorry

end gcd_consecutive_pairs_l512_51299


namespace ellipse_properties_l512_51248

/-- An ellipse with center at the origin, foci on the x-axis, and eccentricity 1/2 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : c / a = 1 / 2
  h_rel : a^2 = b^2 + c^2

/-- A line passing through a point with a given angle -/
structure TangentLine where
  c : ℝ
  angle : ℝ
  h_angle : angle = π / 3

/-- The main theorem about the ellipse and its properties -/
theorem ellipse_properties (E : Ellipse) (L : TangentLine) :
  (E.a = 2 ∧ E.b = Real.sqrt 3 ∧ E.c = 1) ∧
  (∀ k m : ℝ, ∃ x y : ℝ,
    x^2 / 4 + y^2 / 3 = 1 ∧
    y = k * x + m ∧
    (x - 2)^2 + y^2 = 4 →
    k * (2 / 7) + m = 0) :=
sorry

end ellipse_properties_l512_51248


namespace matching_probability_theorem_l512_51286

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.yellow + jb.red

/-- Abe's jelly beans -/
def abe : JellyBeans :=
  { green := 2, yellow := 0, red := 3 }

/-- Bob's jelly beans -/
def bob : JellyBeans :=
  { green := 2, yellow := 2, red := 3 }

/-- Calculates the probability of matching colors -/
def matchingProbability (person1 person2 : JellyBeans) : ℚ :=
  (person1.green * person2.green + person1.red * person2.red : ℚ) /
  ((person1.total * person2.total) : ℚ)

theorem matching_probability_theorem :
  matchingProbability abe bob = 13 / 35 := by
  sorry

end matching_probability_theorem_l512_51286


namespace self_inverse_cube_mod_15_l512_51252

theorem self_inverse_cube_mod_15 (a : ℤ) (h : a * a ≡ 1 [ZMOD 15]) :
  a^3 ≡ 1 [ZMOD 15] := by
  sorry

end self_inverse_cube_mod_15_l512_51252


namespace sqrt_difference_equality_l512_51250

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 25) = Real.sqrt 170 - Real.sqrt 11 := by
  sorry

end sqrt_difference_equality_l512_51250


namespace complex_equation_solution_l512_51223

theorem complex_equation_solution (z : ℂ) : (z - Complex.I) * Complex.I = 2 + Complex.I → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l512_51223


namespace no_real_roots_l512_51225

theorem no_real_roots : ¬∃ x : ℝ, x + 2 * Real.sqrt (x - 5) = 6 := by sorry

end no_real_roots_l512_51225


namespace sphere_radii_difference_l512_51261

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h₁ : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h₂ : 2 * π * r₁ + 2 * π * r₂ = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end sphere_radii_difference_l512_51261


namespace lunks_needed_for_apples_l512_51277

-- Define the exchange rates
def lunks_to_kunks (l : ℚ) : ℚ := l * (2/4)
def kunks_to_apples (k : ℚ) : ℚ := k * (5/3)

-- Theorem statement
theorem lunks_needed_for_apples (n : ℚ) : 
  kunks_to_apples (lunks_to_kunks 18) = 15 := by
  sorry

#check lunks_needed_for_apples

end lunks_needed_for_apples_l512_51277


namespace parallel_vectors_k_value_l512_51226

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, a.1 = t * b.1 ∧ a.2 = t * b.2

theorem parallel_vectors_k_value :
  ∀ k : ℝ, 
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (k, 4)
  parallel a b → k = -2 := by
sorry

end parallel_vectors_k_value_l512_51226


namespace intersecting_squares_area_difference_l512_51222

theorem intersecting_squares_area_difference : 
  let s1 : ℕ := 12
  let s2 : ℕ := 9
  let s3 : ℕ := 7
  let s4 : ℕ := 3
  s1^2 + s3^2 - (s2^2 + s4^2) = 103 := by
  sorry

end intersecting_squares_area_difference_l512_51222


namespace triple_sharp_72_l512_51238

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.4 * N + 3

-- Theorem statement
theorem triple_sharp_72 : sharp (sharp (sharp 72)) = 9.288 := by
  sorry

end triple_sharp_72_l512_51238


namespace jeremy_oranges_l512_51210

theorem jeremy_oranges (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) : 
  tuesday = 3 * monday →
  wednesday = 70 →
  monday + tuesday + wednesday = 470 →
  monday = 100 := by
sorry

end jeremy_oranges_l512_51210


namespace f_comparison_and_max_value_l512_51249

noncomputable def f (x : ℝ) : ℝ := -2 * Real.sin x - Real.cos (2 * x)

theorem f_comparison_and_max_value :
  (f (π / 4) > f (π / 6)) ∧
  (∀ x : ℝ, f x ≤ -3/2) ∧
  (∃ x : ℝ, f x = -3/2) :=
by sorry

end f_comparison_and_max_value_l512_51249


namespace chinese_dream_speech_competition_l512_51205

theorem chinese_dream_speech_competition :
  let num_contestants : ℕ := 4
  let num_topics : ℕ := 4
  let num_topics_used : ℕ := 3
  
  (num_topics.choose 1) * (num_topics_used ^ num_contestants) = 324 :=
by sorry

end chinese_dream_speech_competition_l512_51205


namespace unique_solution_l512_51260

theorem unique_solution (x y z : ℝ) 
  (hx : x > 5) (hy : y > 5) (hz : z > 5)
  (h : ((x + 3)^2 / (y + z - 3)) + ((y + 5)^2 / (z + x - 5)) + ((z + 7)^2 / (x + y - 7)) = 45) :
  x = 15 ∧ y = 15 ∧ z = 15 := by
sorry

end unique_solution_l512_51260


namespace second_day_hours_proof_l512_51268

/-- Represents the number of hours worked on the second day -/
def hours_second_day : ℕ := 8

/-- The hourly rate paid to each worker -/
def hourly_rate : ℕ := 10

/-- The total payment received by both workers -/
def total_payment : ℕ := 660

/-- The number of hours worked on the first day -/
def hours_first_day : ℕ := 10

/-- The number of hours worked on the third day -/
def hours_third_day : ℕ := 15

/-- The number of workers -/
def num_workers : ℕ := 2

theorem second_day_hours_proof :
  hours_second_day * num_workers * hourly_rate +
  hours_first_day * num_workers * hourly_rate +
  hours_third_day * num_workers * hourly_rate = total_payment :=
by sorry

end second_day_hours_proof_l512_51268


namespace problem_solution_l512_51254

def f (x a : ℝ) := |2*x - a| + |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x (-1) ≤ 2 ↔ x ∈ Set.Icc (-1/2) (1/2)) ∧
  (Set.Icc (1/2) 1 ⊆ {x : ℝ | f x a ≤ |2*x + 1|} → a ∈ Set.Icc 0 3) :=
by sorry

end problem_solution_l512_51254


namespace base_subtraction_l512_51247

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_subtraction : 
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_7_number := [5, 6, 1]  -- 165 in base 7 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_7_number 7) = 169 := by
  sorry

end base_subtraction_l512_51247


namespace fraction_change_l512_51244

/-- Given a fraction 3/4, if we increase the numerator by 12% and decrease the denominator by 2%,
    the resulting fraction is approximately 0.8571. -/
theorem fraction_change (ε : ℝ) (h_ε : ε > 0) :
  ∃ (new_fraction : ℝ),
    (3 * (1 + 0.12)) / (4 * (1 - 0.02)) = new_fraction ∧
    |new_fraction - 0.8571| < ε :=
by sorry

end fraction_change_l512_51244


namespace concert_attendance_l512_51271

theorem concert_attendance (adult_price child_price total_collected : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 3)
  (h3 : total_collected = 6000)
  (h4 : ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected) :
  ∃ (total : ℕ), total = 1500 ∧ 
    ∃ (a c : ℕ), c = 3 * a ∧ adult_price * a + child_price * c = total_collected ∧ 
    total = a + c := by
  sorry

end concert_attendance_l512_51271


namespace coloring_book_coupons_l512_51276

theorem coloring_book_coupons 
  (initial_stock : ℝ) 
  (books_sold : ℝ) 
  (coupons_per_book : ℝ) 
  (h1 : initial_stock = 40.0) 
  (h2 : books_sold = 20.0) 
  (h3 : coupons_per_book = 4.0) : 
  (initial_stock - books_sold) * coupons_per_book = 80.0 := by
  sorry

end coloring_book_coupons_l512_51276


namespace circle_regions_l512_51217

/-- The number of regions into which n circles divide a plane, 
    where each pair of circles intersects and no three circles 
    intersect at the same point. -/
def f (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating the properties of the function f -/
theorem circle_regions (n : ℕ) : 
  n > 0 → 
  (f 3 = 8) ∧ 
  (f n = n^2 - n + 2) := by
  sorry

end circle_regions_l512_51217


namespace square_sum_plus_product_squares_l512_51270

theorem square_sum_plus_product_squares : (3 + 9)^2 + 3^2 * 9^2 = 873 := by
  sorry

end square_sum_plus_product_squares_l512_51270


namespace money_distribution_l512_51201

/-- Given three people A, B, and C with a total of 600 Rs between them,
    where B and C together have 450 Rs, and C has 100 Rs,
    prove that A and C together have 250 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 600 →
  B + C = 450 →
  C = 100 →
  A + C = 250 := by
sorry

end money_distribution_l512_51201


namespace parallel_lines_a_value_l512_51257

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal or both lines are vertical -/
def parallel (l1 l2 : Line) : Prop :=
  (l1.b ≠ 0 ∧ l2.b ≠ 0 ∧ l1.a / l1.b = l2.a / l2.b) ∨
  (l1.b = 0 ∧ l2.b = 0)

theorem parallel_lines_a_value (a : ℝ) :
  let l1 : Line := ⟨a, 2, a⟩
  let l2 : Line := ⟨3*a, a-1, 7⟩
  parallel l1 l2 → a = 0 ∨ a = 7 := by
  sorry

end parallel_lines_a_value_l512_51257


namespace max_pots_is_ten_l512_51209

/-- Represents the number of items Susan can buy -/
structure Purchase where
  pins : ℕ
  pans : ℕ
  pots : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  3 * p.pins + 4 * p.pans + 9 * p.pots

/-- Checks if a purchase is valid according to the problem constraints -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pins ≥ 1 ∧ p.pans ≥ 1 ∧ p.pots ≥ 1 ∧ totalCost p = 100

/-- Theorem stating that the maximum number of pots Susan can buy is 10 -/
theorem max_pots_is_ten :
  ∀ p : Purchase, isValidPurchase p → p.pots ≤ 10 ∧ 
  ∃ q : Purchase, isValidPurchase q ∧ q.pots = 10 :=
sorry

end max_pots_is_ten_l512_51209


namespace canal_construction_l512_51200

/-- Canal construction problem -/
theorem canal_construction 
  (total_length : ℝ) 
  (team_b_extra : ℝ) 
  (time_ratio : ℝ) 
  (cost_a : ℝ) 
  (cost_b : ℝ) 
  (total_days : ℕ) 
  (h1 : total_length = 1650)
  (h2 : team_b_extra = 30)
  (h3 : time_ratio = 3/2)
  (h4 : cost_a = 90000)
  (h5 : cost_b = 120000)
  (h6 : total_days = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧ 
    rate_b = 90 ∧ 
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    total_length / rate_a = time_ratio * (total_length / rate_b) ∧
    ∃ (days_a_alone : ℝ),
      0 ≤ days_a_alone ∧ 
      days_a_alone ≤ total_days ∧
      rate_a * days_a_alone + (rate_a + rate_b) * (total_days - days_a_alone) = total_length ∧
      total_cost = cost_a * days_a_alone + (cost_a + cost_b) * (total_days - days_a_alone) :=
by sorry

end canal_construction_l512_51200


namespace power_equation_solution_l512_51218

theorem power_equation_solution : 2^90 * 8^90 = 64^(90 - 30) := by sorry

end power_equation_solution_l512_51218


namespace inequality_solution_m_range_l512_51290

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + m - 1

theorem inequality_solution (x : ℝ) :
  (m = -1 ∧ x ≥ 1) ∨
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) ↔
  f m x ≥ (m + 1) * x := by sorry

theorem m_range :
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f m x ≥ 0) → m ≥ 1 := by sorry

end inequality_solution_m_range_l512_51290


namespace eighth_term_of_geometric_sequence_l512_51235

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem eighth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_3 : a 3 = 3)
  (h_6 : a 6 = 24) :
  a 8 = 96 := by
sorry

end eighth_term_of_geometric_sequence_l512_51235


namespace prob_all_cats_before_lunch_l512_51206

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of animals -/
def totalAnimals : ℕ := 7

/-- The number of cats -/
def numCats : ℕ := 2

/-- The number of dogs -/
def numDogs : ℕ := 5

/-- The number of animals to be groomed before lunch -/
def numGroomed : ℕ := 4

/-- The probability of grooming all cats before lunch -/
def probAllCats : ℚ := (choose numDogs (numGroomed - numCats)) / (choose totalAnimals numGroomed)

theorem prob_all_cats_before_lunch : probAllCats = 2/7 := by sorry

end prob_all_cats_before_lunch_l512_51206


namespace beth_crayon_count_l512_51262

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayon_count : total_crayons = 46 := by
  sorry

end beth_crayon_count_l512_51262


namespace problem_statement_l512_51207

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.tan x = 1

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - 3*x + 2 < 0 ↔ 1 < x ∧ x < 2

-- Theorem statement
theorem problem_statement : (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ ¬(¬p ∨ ¬q) := by
  sorry

end problem_statement_l512_51207


namespace sculpture_and_base_height_l512_51233

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total height of a sculpture and its base in inches -/
def total_height (sculpture_feet : ℕ) (sculpture_inches : ℕ) (base_inches : ℕ) : ℕ :=
  feet_to_inches sculpture_feet + sculpture_inches + base_inches

/-- Theorem stating that a sculpture of 2 feet 10 inches on an 8-inch base has a total height of 42 inches -/
theorem sculpture_and_base_height :
  total_height 2 10 8 = 42 := by
  sorry

end sculpture_and_base_height_l512_51233


namespace units_digit_G_100_l512_51236

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 2

-- Define a function to get the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_100 : units_digit (G 100) = 0 := by
  sorry

end units_digit_G_100_l512_51236


namespace all_roots_real_l512_51220

/-- The polynomial x^4 - 4x^3 + 6x^2 - 4x + 1 -/
def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x + 1

/-- Theorem stating that all roots of the polynomial are real -/
theorem all_roots_real : ∀ x : ℂ, p x.re = 0 → x.im = 0 := by
  sorry

end all_roots_real_l512_51220


namespace white_balls_count_l512_51251

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  green = 18 ∧
  yellow = 17 ∧
  red = 3 ∧
  purple = 1 ∧
  prob = 95 / 100 ∧
  prob = (total - (red + purple)) / total →
  total - (green + yellow + red + purple) = 21 :=
by sorry

end white_balls_count_l512_51251


namespace set_equality_implies_a_equals_three_l512_51202

theorem set_equality_implies_a_equals_three (a : ℝ) : 
  ({0, 1, a^2} : Set ℝ) = ({1, 0, 2*a + 3} : Set ℝ) → a = 3 :=
by sorry

end set_equality_implies_a_equals_three_l512_51202


namespace solve_exponential_equation_l512_51292

theorem solve_exponential_equation :
  ∃ y : ℝ, (40 : ℝ)^3 = 8^y ∧ y = 3 := by
  sorry

end solve_exponential_equation_l512_51292


namespace valid_count_is_48_l512_51253

/-- Represents a three-digit number with the last two digits being the same -/
structure ThreeDigitNumber where
  first : Nat
  last : Nat
  first_is_digit : first ≤ 9
  last_is_digit : last ≤ 9

/-- Checks if a ThreeDigitNumber is valid according to the problem conditions -/
def isValid (n : ThreeDigitNumber) : Prop :=
  (100 * n.first + 11 * n.last) % 3 = 0 ∧
  n.first + 2 * n.last ≤ 18

/-- The count of valid ThreeDigitNumbers -/
def validCount : Nat :=
  (ThreeDigitNumber.mk 1 0 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 3 (by norm_num) (by norm_num) ::
   ThreeDigitNumber.mk 1 6 (by norm_num) (by norm_num) ::
   -- ... (other valid ThreeDigitNumbers)
   []).length

theorem valid_count_is_48 : validCount = 48 := by
  sorry

#eval validCount

end valid_count_is_48_l512_51253


namespace digit_replacement_theorem_l512_51269

theorem digit_replacement_theorem : ∃ (x y z w : ℕ), 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 ∧
  42 * (10 * x + 8) = 2000 + 100 * y + 10 * z + w ∧
  (x + y + z + w) % 2 = 1 ∧
  2000 ≤ 42 * (10 * x + 8) ∧ 42 * (10 * x + 8) < 3000 :=
by sorry

end digit_replacement_theorem_l512_51269


namespace equation_solution_l512_51284

theorem equation_solution :
  let s : ℚ := 20
  let r : ℚ := 270 / 7
  2 * (r - 45) / 3 = (3 * s - 2 * r) / 4 := by
  sorry

end equation_solution_l512_51284


namespace f_at_one_l512_51282

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem f_at_one : f 1 = 2 := by sorry

end f_at_one_l512_51282


namespace inverse_proportion_points_relation_l512_51264

theorem inverse_proportion_points_relation :
  ∀ x₁ x₂ x₃ : ℝ,
  (2 = 8 / x₁) →
  (-1 = 8 / x₂) →
  (4 = 8 / x₃) →
  (x₁ > x₃ ∧ x₃ > x₂) :=
by sorry

end inverse_proportion_points_relation_l512_51264


namespace expression_simplification_l512_51203

theorem expression_simplification (y : ℝ) : 
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 := by
  sorry

end expression_simplification_l512_51203


namespace ellipse_equation_from_line_through_focus_and_vertex_l512_51230

/-- Represents an ellipse in standard form -/
structure StandardEllipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: If a line with equation x - 2y + 2 = 0 passes through a focus and a vertex of an ellipse,
    then the standard equation of the ellipse is either x²/5 + y² = 1 or x²/4 + y²/5 = 1 -/
theorem ellipse_equation_from_line_through_focus_and_vertex 
  (l : Line) 
  (hl : l.a = 1 ∧ l.b = -2 ∧ l.c = 2) 
  (passes_through_focus_and_vertex : ∃ (e : StandardEllipse), 
    (∃ (x y : ℝ), x - 2*y + 2 = 0 ∧ 
      ((x = e.a ∧ y = 0) ∨ (x = 0 ∧ y = e.b) ∨ (x = -e.a ∧ y = 0) ∨ (x = 0 ∧ y = -e.b)) ∧
      ((x^2 / e.a^2 + y^2 / e.b^2 = 1) ∨ (y^2 / e.a^2 + x^2 / e.b^2 = 1)))) :
  ∃ (e : StandardEllipse), (e.a^2 = 5 ∧ e.b^2 = 1) ∨ (e.a^2 = 4 ∧ e.b^2 = 5) := by
  sorry

end ellipse_equation_from_line_through_focus_and_vertex_l512_51230


namespace jerry_added_two_figures_l512_51211

/-- The number of action figures Jerry added to his shelf -/
def action_figures_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem: Jerry added 2 action figures to his shelf -/
theorem jerry_added_two_figures : action_figures_added 8 10 = 2 := by
  sorry

end jerry_added_two_figures_l512_51211


namespace opposite_equal_roots_iff_l512_51216

/-- The equation has roots that are numerically equal but of opposite signs -/
def has_opposite_equal_roots (d e f n : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ 0 ∧ y₂ = -y₁ ∧
  (y₁^2 + 2*d*y₁) / (e*y₁ + f) = n / (n - 2) ∧
  (y₂^2 + 2*d*y₂) / (e*y₂ + f) = n / (n - 2)

/-- The main theorem -/
theorem opposite_equal_roots_iff (d e f : ℝ) :
  ∀ n : ℝ, has_opposite_equal_roots d e f n ↔ n = 4*d / (2*d - e) :=
sorry

end opposite_equal_roots_iff_l512_51216


namespace parabola_rotation_180_l512_51212

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Rotates a parabola 180° around its vertex -/
def rotate_180 (p : Parabola) : Parabola :=
  { a := -p.a, b := p.b }

theorem parabola_rotation_180 (p : Parabola) (h : p = { a := 1/2, b := 1 }) :
  rotate_180 p = { a := -1/2, b := 1 } := by
  sorry

end parabola_rotation_180_l512_51212


namespace cow_husk_consumption_l512_51295

/-- Given that 40 cows eat 40 bags of husk in 40 days, 
    prove that one cow will eat one bag of husk in 40 days. -/
theorem cow_husk_consumption (cows bags days : ℕ) 
  (h : cows = 40 ∧ bags = 40 ∧ days = 40) : 
  (cows * bags) / (cows * days) = 1 := by
  sorry

end cow_husk_consumption_l512_51295


namespace romeo_chocolate_bars_l512_51204

theorem romeo_chocolate_bars : 
  ∀ (buy_cost sell_total packaging_cost profit num_bars : ℕ),
    buy_cost = 5 →
    sell_total = 90 →
    packaging_cost = 2 →
    profit = 55 →
    num_bars * (buy_cost + packaging_cost) + profit = sell_total →
    num_bars = 5 := by
  sorry

end romeo_chocolate_bars_l512_51204


namespace sum_congruence_l512_51240

theorem sum_congruence : ∃ k : ℤ, (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) = 16 * k + 2 := by
  sorry

end sum_congruence_l512_51240


namespace cheesecake_price_per_slice_l512_51296

/-- Represents the price of a cheesecake slice -/
def price_per_slice (slices_per_pie : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (slices_per_pie * pies_sold)

/-- Proves that the price per slice of cheesecake is $7 -/
theorem cheesecake_price_per_slice :
  price_per_slice 6 7 294 = 7 := by
  sorry

end cheesecake_price_per_slice_l512_51296
