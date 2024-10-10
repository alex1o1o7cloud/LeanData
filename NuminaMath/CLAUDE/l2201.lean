import Mathlib

namespace milk_production_l2201_220177

/-- Milk production calculation -/
theorem milk_production
  (m n p x q r : ℝ)
  (h1 : m > 0)
  (h2 : p > 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ m)
  : (q * r * (m + 0.2 * x) * n) / (m * p) =
    q * r * ((m - x) * (n / (m * p)) + x * (1.2 * n / (m * p))) :=
by sorry

end milk_production_l2201_220177


namespace factorization_of_difference_of_squares_l2201_220154

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end factorization_of_difference_of_squares_l2201_220154


namespace min_buses_for_field_trip_l2201_220135

/-- The minimum number of buses needed to transport students for a field trip. -/
def min_buses (total_students : ℕ) (bus_capacity : ℕ) (min_buses : ℕ) : ℕ :=
  max (min_buses) (((total_students + bus_capacity - 1) / bus_capacity) : ℕ)

/-- Theorem stating the minimum number of buses needed for the given conditions. -/
theorem min_buses_for_field_trip :
  min_buses 500 45 2 = 12 := by
  sorry

end min_buses_for_field_trip_l2201_220135


namespace square_and_sqrt_preserve_geometric_sequence_l2201_220162

-- Define the domain (−∞,0)∪(0,+∞)
def NonZeroReals : Set ℝ := {x : ℝ | x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the "preserving geometric sequence" property
def PreservingGeometricSequence (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, IsGeometricSequence a → IsGeometricSequence (fun n ↦ f (a n))

-- State the theorem
theorem square_and_sqrt_preserve_geometric_sequence :
  (PreservingGeometricSequence (fun x ↦ x^2)) ∧
  (PreservingGeometricSequence (fun x ↦ Real.sqrt (abs x))) :=
sorry

end square_and_sqrt_preserve_geometric_sequence_l2201_220162


namespace range_of_a_l2201_220170

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ (a > 8 ∨ a < -2) :=
by sorry

end range_of_a_l2201_220170


namespace sallys_car_fuel_efficiency_l2201_220152

/-- Calculates the fuel efficiency of Sally's car given her trip expenses and savings --/
theorem sallys_car_fuel_efficiency :
  ∀ (savings : ℝ) (parking : ℝ) (entry : ℝ) (meal : ℝ) (distance : ℝ) (gas_price : ℝ) (additional_savings : ℝ),
    savings = 28 →
    parking = 10 →
    entry = 55 →
    meal = 25 →
    distance = 165 →
    gas_price = 3 →
    additional_savings = 95 →
    (2 * distance) / ((savings + additional_savings - (parking + entry + meal)) / gas_price) = 30 :=
by
  sorry

end sallys_car_fuel_efficiency_l2201_220152


namespace square_difference_theorem_l2201_220189

theorem square_difference_theorem : (13 + 8)^2 - (13 - 8)^2 = 416 := by
  sorry

end square_difference_theorem_l2201_220189


namespace complex_division_proof_l2201_220137

theorem complex_division_proof : ∀ (i : ℂ), i^2 = -1 → (1 : ℂ) / (1 + i) = (1 - i) / 2 := by sorry

end complex_division_proof_l2201_220137


namespace deposit_calculation_l2201_220163

theorem deposit_calculation (total_price : ℝ) (deposit_percentage : ℝ) (remaining_amount : ℝ) 
  (h1 : deposit_percentage = 0.1)
  (h2 : remaining_amount = 720)
  (h3 : total_price * (1 - deposit_percentage) = remaining_amount) :
  total_price * deposit_percentage = 80 := by
  sorry

end deposit_calculation_l2201_220163


namespace three_correct_probability_l2201_220191

/-- The probability of exactly 3 out of 5 packages being delivered to their correct houses -/
def probability_three_correct (n : ℕ) : ℚ :=
  if n = 5 then 1 / 12 else 0

/-- Theorem stating the probability of exactly 3 out of 5 packages being delivered correctly -/
theorem three_correct_probability :
  probability_three_correct 5 = 1 / 12 :=
by sorry

end three_correct_probability_l2201_220191


namespace min_distance_vectors_l2201_220196

/-- Given planar vectors a and b with an angle of 120° between them and a dot product of -1,
    the minimum value of |a - b| is √6. -/
theorem min_distance_vectors (a b : ℝ × ℝ) : 
  (Real.cos (120 * π / 180) = -1/2) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  (∀ c d : ℝ × ℝ, c.1 * d.1 + c.2 * d.2 = -1 → 
    Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≥ Real.sqrt 6) ∧
  (∃ e f : ℝ × ℝ, e.1 * f.1 + e.2 * f.2 = -1 ∧ 
    Real.sqrt ((e.1 - f.1)^2 + (e.2 - f.2)^2) = Real.sqrt 6) := by
  sorry

end min_distance_vectors_l2201_220196


namespace freight_train_speed_proof_l2201_220140

-- Define the total distance between points A and B
def total_distance : ℝ := 460

-- Define the time it takes for the trains to meet
def meeting_time : ℝ := 2

-- Define the speed of the passenger train
def passenger_train_speed : ℝ := 120

-- Define the speed of the freight train (to be proven)
def freight_train_speed : ℝ := 110

-- Theorem statement
theorem freight_train_speed_proof :
  total_distance = (passenger_train_speed + freight_train_speed) * meeting_time :=
by sorry

end freight_train_speed_proof_l2201_220140


namespace expand_expression_l2201_220153

theorem expand_expression (x y : ℝ) :
  -2 * (4 * x^3 - 3 * x * y + 5) = -8 * x^3 + 6 * x * y - 10 := by
  sorry

end expand_expression_l2201_220153


namespace abs_sum_iff_positive_l2201_220173

theorem abs_sum_iff_positive (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end abs_sum_iff_positive_l2201_220173


namespace layla_nahima_score_difference_l2201_220192

theorem layla_nahima_score_difference :
  ∀ (total_points layla_score nahima_score : ℕ),
    total_points = 112 →
    layla_score = 70 →
    total_points = layla_score + nahima_score →
    layla_score - nahima_score = 28 :=
by
  sorry

end layla_nahima_score_difference_l2201_220192


namespace always_negative_quadratic_function_l2201_220106

/-- The function f(x) = kx^2 - kx - 1 is always negative if and only if -4 < k ≤ 0 -/
theorem always_negative_quadratic_function (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ -4 < k ∧ k ≤ 0 := by sorry

end always_negative_quadratic_function_l2201_220106


namespace masha_number_l2201_220198

theorem masha_number (x y : ℕ) : 
  (x + y = 2002 ∨ x * y = 2002) →
  (∀ a : ℕ, (a + y = 2002 ∨ a * y = 2002) → ∃ b ≠ y, (a + b = 2002 ∨ a * b = 2002)) →
  (∀ a : ℕ, (x + a = 2002 ∨ x * a = 2002) → ∃ b ≠ x, (b + a = 2002 ∨ b * a = 2002)) →
  max x y = 1001 :=
by sorry

end masha_number_l2201_220198


namespace probability_specific_arrangement_l2201_220199

def num_letters : ℕ := 8

theorem probability_specific_arrangement (n : ℕ) (h : n = num_letters) :
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 40320 :=
sorry

end probability_specific_arrangement_l2201_220199


namespace bankers_discount_problem_l2201_220197

/-- Proves that given a sum S where the banker's discount is 18 and the true discount is 15, S equals 75 -/
theorem bankers_discount_problem (S : ℝ) 
  (h1 : 18 = 15 + (15^2 / S)) : S = 75 := by
  sorry

end bankers_discount_problem_l2201_220197


namespace regular_hexagon_interior_angle_l2201_220108

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_regular_hexagon : ℝ := 120

/-- Theorem: The measure of each interior angle of a regular hexagon is 120 degrees -/
theorem regular_hexagon_interior_angle :
  interior_angle_regular_hexagon = 120 := by
  sorry

end regular_hexagon_interior_angle_l2201_220108


namespace max_value_a_l2201_220116

theorem max_value_a (a b c d : ℕ+) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) 
  (h4 : Even c) (h5 : d < 150) : a ≤ 8924 := by
  sorry

end max_value_a_l2201_220116


namespace tiger_catch_deer_distance_l2201_220113

/-- The distance a tiger needs to run to catch a deer given their speeds and initial separation -/
theorem tiger_catch_deer_distance 
  (tiger_leaps_behind : ℕ)
  (tiger_leaps_per_minute : ℕ)
  (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ)
  (deer_meters_per_leap : ℕ)
  (h1 : tiger_leaps_behind = 50)
  (h2 : tiger_leaps_per_minute = 5)
  (h3 : deer_leaps_per_minute = 4)
  (h4 : tiger_meters_per_leap = 8)
  (h5 : deer_meters_per_leap = 5) :
  (tiger_leaps_behind * tiger_meters_per_leap * tiger_leaps_per_minute) /
  (tiger_leaps_per_minute * tiger_meters_per_leap - deer_leaps_per_minute * deer_meters_per_leap) = 800 :=
by sorry

end tiger_catch_deer_distance_l2201_220113


namespace complex_number_real_condition_l2201_220132

theorem complex_number_real_condition (a : ℝ) : 
  (2 * Complex.I - a / (1 - Complex.I)).im = 0 → a = 4 := by
  sorry

end complex_number_real_condition_l2201_220132


namespace melanie_missed_games_l2201_220105

/-- The number of football games Melanie missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Melanie missed 4 games given the conditions -/
theorem melanie_missed_games :
  let total_games : ℕ := 7
  let attended_games : ℕ := 3
  games_missed total_games attended_games = 4 := by
  sorry


end melanie_missed_games_l2201_220105


namespace wilson_theorem_plus_one_l2201_220126

theorem wilson_theorem_plus_one (p : Nat) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  (p - 1).factorial + 1 ∣ p := by
  sorry

end wilson_theorem_plus_one_l2201_220126


namespace sum_of_extrema_l2201_220122

/-- A function f(x) = 2x³ - ax² + 1 with exactly one zero in (0, +∞) -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^3 - a * x^2 + 1

/-- The property that f has exactly one zero in (0, +∞) -/
def has_one_zero (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f a x = 0

/-- The theorem stating that if f has one zero in (0, +∞), then the sum of its max and min on [-1, 1] is -3 -/
theorem sum_of_extrema (a : ℝ) (h : has_one_zero a) :
  (⨆ x ∈ Set.Icc (-1) 1, f a x) + (⨅ x ∈ Set.Icc (-1) 1, f a x) = -3 :=
sorry

end sum_of_extrema_l2201_220122


namespace sum_of_reciprocals_of_roots_l2201_220118

theorem sum_of_reciprocals_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 26*r₁ + 12 = 0 → 
  r₂^2 - 26*r₂ + 12 = 0 → 
  r₁ ≠ r₂ →
  (1/r₁ + 1/r₂) = 13/6 := by
sorry

end sum_of_reciprocals_of_roots_l2201_220118


namespace temperature_height_relationship_l2201_220149

/-- The temperature-height relationship function -/
def t (h : ℝ) : ℝ := 20 - 6 * h

/-- The set of given data points -/
def data_points : List (ℝ × ℝ) := [(0, 20), (1, 14), (2, 8), (3, 2), (4, -4)]

/-- Theorem stating that the function t accurately describes the temperature-height relationship -/
theorem temperature_height_relationship :
  ∀ (point : ℝ × ℝ), point ∈ data_points → t point.1 = point.2 := by
  sorry

end temperature_height_relationship_l2201_220149


namespace at_least_three_to_six_colorings_l2201_220174

/-- Represents the colors that can be used to color the hexagons -/
inductive Color
| Red
| Yellow
| Green
| Blue

/-- Represents a hexagon in the figure -/
structure Hexagon where
  color : Color

/-- Represents the central hexagon and its six adjacent hexagons -/
structure CentralHexagonWithAdjacent where
  center : Hexagon
  adjacent : Fin 6 → Hexagon

/-- Two hexagons are considered adjacent if they share a side -/
def areAdjacent (h1 h2 : Hexagon) : Prop := sorry

/-- A coloring is valid if no two adjacent hexagons have the same color -/
def isValidColoring (config : CentralHexagonWithAdjacent) : Prop :=
  config.center.color = Color.Red ∧
  ∀ i j : Fin 6, i ≠ j →
    config.adjacent i ≠ config.adjacent j ∧
    config.adjacent i ≠ config.center ∧
    config.adjacent j ≠ config.center

/-- The number of valid colorings for the central hexagon and its adjacent hexagons -/
def numValidColorings : ℕ := sorry

theorem at_least_three_to_six_colorings :
  numValidColorings ≥ 3^6 := by sorry

end at_least_three_to_six_colorings_l2201_220174


namespace trisection_intersection_l2201_220100

theorem trisection_intersection (f : ℝ → ℝ) (A B C E : ℝ × ℝ) :
  f = (λ x => Real.exp x) →
  A = (0, 1) →
  B = (3, Real.exp 3) →
  C.1 = 1 →
  C.2 = 2/3 * A.2 + 1/3 * B.2 →
  E.2 = C.2 →
  f E.1 = E.2 →
  E.1 = Real.log ((2 + Real.exp 3) / 3) := by
sorry

end trisection_intersection_l2201_220100


namespace factorial_square_root_squared_l2201_220176

theorem factorial_square_root_squared : (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 := by
  sorry

end factorial_square_root_squared_l2201_220176


namespace megan_carrots_l2201_220175

theorem megan_carrots (initial_carrots thrown_out_carrots next_day_carrots : ℕ) :
  initial_carrots ≥ thrown_out_carrots →
  initial_carrots - thrown_out_carrots + next_day_carrots =
    initial_carrots + next_day_carrots - thrown_out_carrots :=
by sorry

end megan_carrots_l2201_220175


namespace fourth_year_area_l2201_220138

def initial_area : ℝ := 10000
def annual_increase : ℝ := 0.2

def area_after_n_years (n : ℕ) : ℝ :=
  initial_area * (1 + annual_increase) ^ n

theorem fourth_year_area :
  area_after_n_years 3 = 17280 :=
by sorry

end fourth_year_area_l2201_220138


namespace f_composition_equals_result_l2201_220128

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_equals_result : 
  f (f (f (f (1 + 2*I)))) = (23882205 - 24212218*I)^3 := by sorry

end f_composition_equals_result_l2201_220128


namespace distance_between_lines_l2201_220157

/-- Represents a circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- Length of the first chord -/
  chord1_length : ℝ
  /-- Length of the second chord -/
  chord2_length : ℝ
  /-- Length of the third chord -/
  chord3_length : ℝ
  /-- The first and second chords have equal length -/
  chord1_eq_chord2 : chord1_length = chord2_length
  /-- The first chord has length 40 -/
  chord1_is_40 : chord1_length = 40
  /-- The third chord has length 36 -/
  chord3_is_36 : chord3_length = 36

/-- Theorem stating that the distance between adjacent parallel lines is 1.5 -/
theorem distance_between_lines (c : CircleWithParallelLines) : c.line_distance = 1.5 := by
  sorry

end distance_between_lines_l2201_220157


namespace vector_calculation_l2201_220112

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_calculation : 
  (2 : ℝ) • vector_a - vector_b = (5, 8) := by sorry

end vector_calculation_l2201_220112


namespace largest_quantity_l2201_220167

def D : ℚ := 2008/2007 + 2008/2009
def E : ℚ := 2008/2009 + 2010/2009
def F : ℚ := 2009/2008 + 2009/2010 - 1/2009

theorem largest_quantity : D > E ∧ D > F := by
  sorry

end largest_quantity_l2201_220167


namespace oprah_car_collection_l2201_220120

/-- The number of cars in Oprah's collection -/
def total_cars : ℕ := 3500

/-- The average number of cars Oprah gives away per year -/
def cars_given_per_year : ℕ := 50

/-- The number of years it takes to reduce the collection -/
def years_to_reduce : ℕ := 60

/-- The number of cars left after giving away -/
def cars_left : ℕ := 500

theorem oprah_car_collection :
  total_cars = cars_left + cars_given_per_year * years_to_reduce :=
by sorry

end oprah_car_collection_l2201_220120


namespace spanish_not_german_students_l2201_220114

theorem spanish_not_german_students (total : ℕ) (both : ℕ) (spanish : ℕ) (german : ℕ) : 
  total = 30 →
  both = 2 →
  spanish = 3 * german →
  spanish + german - both = total →
  spanish - both = 20 :=
by
  sorry

end spanish_not_german_students_l2201_220114


namespace faster_train_speed_l2201_220148

/-- Proves the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed
  (speed_diff : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : speed_diff = 36)
  (h2 : faster_train_length = 120)
  (h3 : crossing_time = 12)
  : ∃ (faster_speed : ℝ), faster_speed = 72 :=
by
  sorry

end faster_train_speed_l2201_220148


namespace fraction_relation_l2201_220142

theorem fraction_relation (x y z w : ℚ) 
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 := by
sorry

end fraction_relation_l2201_220142


namespace a_3_equals_1_l2201_220172

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n - 3

theorem a_3_equals_1 (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 1 = 7) : 
  a 3 = 1 := by
  sorry

end a_3_equals_1_l2201_220172


namespace ball_drawing_theorem_l2201_220110

/-- The number of distinct red balls in the bag -/
def num_red_balls : ℕ := 4

/-- The number of distinct white balls in the bag -/
def num_white_balls : ℕ := 6

/-- The score for drawing a red ball -/
def red_score : ℕ := 2

/-- The score for drawing a white ball -/
def white_score : ℕ := 1

/-- The number of ways to draw 4 balls such that the number of red balls is not less than the number of white balls -/
def ways_to_draw_4_balls : ℕ := 115

/-- The number of ways to draw 5 balls such that the total score is at least 7 points -/
def ways_to_draw_5_balls_score_7_plus : ℕ := 186

/-- The number of ways to arrange 5 drawn balls (with a score of 8 points) such that only two red balls are adjacent -/
def ways_to_arrange_5_balls_score_8 : ℕ := 4320

theorem ball_drawing_theorem : 
  ways_to_draw_4_balls = 115 ∧ 
  ways_to_draw_5_balls_score_7_plus = 186 ∧ 
  ways_to_arrange_5_balls_score_8 = 4320 := by
  sorry

end ball_drawing_theorem_l2201_220110


namespace road_travel_cost_l2201_220103

/-- The cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost (lawn_length lawn_width road_width : ℕ) (cost_per_sqm : ℚ) : 
  lawn_length = 80 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  cost_per_sqm = 2 → 
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * cost_per_sqm = 2600 := by
  sorry

end road_travel_cost_l2201_220103


namespace cube_sum_eq_neg_26_l2201_220168

/-- ω is a nonreal complex number that is a cube root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a nonreal cube root of unity -/
axiom ω_cube_root : ω ^ 3 = 1 ∧ ω ≠ 1

/-- The main theorem to prove -/
theorem cube_sum_eq_neg_26 :
  (1 + ω + 2 * ω^2)^3 + (1 - 2*ω + ω^2)^3 = -26 :=
sorry

end cube_sum_eq_neg_26_l2201_220168


namespace same_color_probability_l2201_220121

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def num_draws : ℕ := 3

theorem same_color_probability :
  let prob_same_color := (marbles_per_color / total_marbles) ^ num_draws * 3
  prob_same_color = 1 / 9 := by
  sorry

end same_color_probability_l2201_220121


namespace student_a_score_l2201_220115

def final_score (total_questions : ℕ) (correct_answers : ℕ) : ℤ :=
  correct_answers - 2 * (total_questions - correct_answers)

theorem student_a_score :
  final_score 100 93 = 79 := by
  sorry

end student_a_score_l2201_220115


namespace min_balls_to_draw_l2201_220130

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The actual counts of balls in the box -/
def initialCounts : BallCounts :=
  { red := 35, green := 27, yellow := 22, blue := 18, white := 15, black := 12 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 20

/-- Theorem stating the minimum number of balls to draw to guarantee the target count -/
theorem min_balls_to_draw (counts : BallCounts) (target : Nat) :
  counts = initialCounts → target = targetCount →
  (∃ (n : Nat), n = 103 ∧
    (∀ (m : Nat), m < n →
      ¬∃ (color : Nat), color ≥ target ∧
        (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
         color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black)) ∧
    (∃ (color : Nat), color ≥ target ∧
      (color ≤ counts.red ∨ color ≤ counts.green ∨ color ≤ counts.yellow ∨
       color ≤ counts.blue ∨ color ≤ counts.white ∨ color ≤ counts.black))) :=
by sorry

end min_balls_to_draw_l2201_220130


namespace divisible_by_six_ratio_l2201_220194

theorem divisible_by_six_ratio (n : ℕ) : n = 120 →
  (Finset.filter (fun x => x % 6 = 0) (Finset.range (n + 1))).card / (n + 1 : ℚ) = 1 / 6 := by
  sorry

end divisible_by_six_ratio_l2201_220194


namespace middle_number_proof_l2201_220181

theorem middle_number_proof (a b c : ℕ) 
  (h_order : a < b ∧ b < c)
  (h_sum1 : a + b = 15)
  (h_sum2 : a + c = 20)
  (h_sum3 : b + c = 25) :
  b = 10 := by
  sorry

end middle_number_proof_l2201_220181


namespace sqrt_2_4_3_6_5_2_l2201_220150

theorem sqrt_2_4_3_6_5_2 : Real.sqrt (2^4 * 3^6 * 5^2) = 540 := by
  sorry

end sqrt_2_4_3_6_5_2_l2201_220150


namespace inverse_f_at_negative_seven_sixtyfourth_l2201_220187

noncomputable def f (x : ℝ) : ℝ := (x^7 - 1) / 4

theorem inverse_f_at_negative_seven_sixtyfourth :
  f⁻¹ (-7/64) = (9/16)^(1/7) :=
by sorry

end inverse_f_at_negative_seven_sixtyfourth_l2201_220187


namespace multiplication_properties_l2201_220117

theorem multiplication_properties :
  (∃ (p q : Nat), Prime p ∧ Prime q ∧ ¬(Prime (p * q))) ∧
  (∀ (a b : Int), ∃ (c : Int), (a^2) * (b^2) = c^2) ∧
  (∀ (m n : Int), Odd m → Odd n → Odd (m * n)) ∧
  (∀ (x y : Int), Even x → Even y → Even (x * y)) :=
by sorry

#check multiplication_properties

end multiplication_properties_l2201_220117


namespace p_adic_valuation_factorial_formula_l2201_220195

/-- The sum of digits of n in base p -/
def sum_of_digits (n : ℕ) (p : ℕ) : ℕ :=
  sorry

/-- The p-adic valuation of n! -/
def p_adic_valuation_factorial (p : ℕ) (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The p-adic valuation of n! equals (n - s(n)) / (p - 1) -/
theorem p_adic_valuation_factorial_formula (p : ℕ) (n : ℕ) (hp : Prime p) (hn : n > 0) :
  p_adic_valuation_factorial p n = (n - sum_of_digits n p) / (p - 1) :=
sorry

end p_adic_valuation_factorial_formula_l2201_220195


namespace necessary_but_not_sufficient_l2201_220141

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (∀ a b, (a - b) * a^2 < 0 → a < b) ∧ 
  (∃ a b, a < b ∧ (a - b) * a^2 ≥ 0) :=
sorry

end necessary_but_not_sufficient_l2201_220141


namespace total_produce_cost_l2201_220143

/-- Calculates the total cost of produce given specific quantities and pricing conditions -/
theorem total_produce_cost (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                           (grape_boxes : ℕ) (grape_weight : ℚ) (grape_price : ℚ)
                           (apples : ℕ) (apple_price : ℚ)
                           (carrot_bags : ℕ) (carrot_orig_price : ℚ) (carrot_discount : ℚ)
                           (strawberry_pounds : ℕ) (strawberry_orig_price : ℚ) (strawberry_discount : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_weight = 2.2 ∧ grape_price = 2.5 ∧
  apples = 700 ∧ apple_price = 0.5 ∧
  carrot_bags = 100 ∧ carrot_orig_price = 2 ∧ carrot_discount = 0.25 ∧
  strawberry_pounds = 120 ∧ strawberry_orig_price = 3.5 ∧ strawberry_discount = 0.15 →
  (asparagus_bundles : ℚ) * asparagus_price +
  (grape_boxes : ℚ) * grape_weight * grape_price +
  ((apples / 3) * 2 : ℚ) * apple_price +
  (carrot_bags : ℚ) * carrot_orig_price * (1 - carrot_discount) +
  (strawberry_pounds : ℚ) * strawberry_orig_price * (1 - strawberry_discount) = 1140.5 := by
sorry

end total_produce_cost_l2201_220143


namespace james_remaining_money_l2201_220125

def weekly_allowance : ℕ := 10
def saving_weeks : ℕ := 4
def video_game_fraction : ℚ := 1/2
def book_fraction : ℚ := 1/4

theorem james_remaining_money :
  let total_savings := weekly_allowance * saving_weeks
  let after_video_game := total_savings * (1 - video_game_fraction)
  let book_cost := after_video_game * book_fraction
  let remaining := after_video_game - book_cost
  remaining = 15 := by sorry

end james_remaining_money_l2201_220125


namespace three_solutions_imply_b_neg_c_zero_l2201_220123

theorem three_solutions_imply_b_neg_c_zero
  (f : ℝ → ℝ)
  (b c : ℝ)
  (h1 : ∀ x, f x = x^2 + b * |x| + c)
  (h2 : ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0)
  (h3 : ∀ (w v : ℝ), f w = 0 ∧ f v = 0 ∧ w ≠ v → w = x ∨ w = y ∨ w = z ∨ v = x ∨ v = y ∨ v = z) :
  b < 0 ∧ c = 0 := by
sorry

end three_solutions_imply_b_neg_c_zero_l2201_220123


namespace binomial_not_divisible_by_prime_l2201_220109

theorem binomial_not_divisible_by_prime (p : ℕ) (n : ℕ) : 
  Prime p → 
  (∀ m : ℕ, m ≤ n → ¬(p ∣ Nat.choose n m)) ↔ 
  ∃ k s : ℕ, n = s * p^k - 1 ∧ 1 ≤ s ∧ s ≤ p :=
by sorry

end binomial_not_divisible_by_prime_l2201_220109


namespace chocolate_division_l2201_220145

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (multiply_factor : ℕ) 
  (h1 : total_chocolate = 60 / 7)
  (h2 : num_piles = 5)
  (h3 : multiply_factor = 3) :
  (total_chocolate / num_piles) * multiply_factor = 36 / 7 := by
  sorry

end chocolate_division_l2201_220145


namespace justin_total_pages_justin_first_book_pages_justin_second_book_pages_l2201_220188

/-- Represents the reading schedule for a week -/
structure ReadingSchedule where
  firstBookDay1 : ℕ
  secondBookDay1 : ℕ
  firstBookIncrement : ℕ → ℕ
  secondBookIncrement : ℕ
  firstBookBreakDay : ℕ
  secondBookBreakDay : ℕ

/-- Calculates the total pages read for both books in a week -/
def totalPagesRead (schedule : ReadingSchedule) : ℕ := 
  let firstBookPages := schedule.firstBookDay1 + 
    (schedule.firstBookDay1 * 2) + 
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 3) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 4) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 5) +
    (schedule.firstBookDay1 * 2 + schedule.firstBookIncrement 6)
  let secondBookPages := schedule.secondBookDay1 + 
    (schedule.secondBookDay1 + schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 2 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 3 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 4 * schedule.secondBookIncrement) +
    (schedule.secondBookDay1 + 5 * schedule.secondBookIncrement)
  firstBookPages + secondBookPages

/-- Justin's reading schedule -/
def justinSchedule : ReadingSchedule := {
  firstBookDay1 := 10,
  secondBookDay1 := 15,
  firstBookIncrement := λ n => 5 * (n - 2),
  secondBookIncrement := 3,
  firstBookBreakDay := 7,
  secondBookBreakDay := 4
}

/-- Theorem stating that Justin reads 295 pages in total -/
theorem justin_total_pages : totalPagesRead justinSchedule = 295 := by
  sorry

/-- Theorem stating that Justin reads 160 pages of the first book -/
theorem justin_first_book_pages : 
  justinSchedule.firstBookDay1 + 
  (justinSchedule.firstBookDay1 * 2) + 
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 3) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 4) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 5) +
  (justinSchedule.firstBookDay1 * 2 + justinSchedule.firstBookIncrement 6) = 160 := by
  sorry

/-- Theorem stating that Justin reads 135 pages of the second book -/
theorem justin_second_book_pages :
  justinSchedule.secondBookDay1 + 
  (justinSchedule.secondBookDay1 + justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 2 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 3 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 4 * justinSchedule.secondBookIncrement) +
  (justinSchedule.secondBookDay1 + 5 * justinSchedule.secondBookIncrement) = 135 := by
  sorry

end justin_total_pages_justin_first_book_pages_justin_second_book_pages_l2201_220188


namespace arithmetic_geometric_mean_inequality_l2201_220146

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧ ((a + b) / 2 = Real.sqrt (a * b) ↔ a = b) := by
  sorry

end arithmetic_geometric_mean_inequality_l2201_220146


namespace intersection_point_of_lines_l2201_220158

theorem intersection_point_of_lines (x y : ℚ) : 
  (5 * x - 2 * y = 4) ∧ (3 * x + 4 * y = 16) ↔ x = 24/13 ∧ y = 34/13 := by
  sorry

end intersection_point_of_lines_l2201_220158


namespace common_root_of_quadratics_l2201_220178

theorem common_root_of_quadratics (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ b * x^2 + c * x + a = 0 → x = 1 := by
  sorry

end common_root_of_quadratics_l2201_220178


namespace solution_satisfies_system_l2201_220127

theorem solution_satisfies_system :
  let x : ℚ := 130/161
  let y : ℚ := 76/23
  let z : ℚ := 3
  (7 * x - 3 * y + 2 * z = 4) ∧
  (4 * y - x - 5 * z = -3) ∧
  (3 * x + 2 * y - z = 7) := by
  sorry

end solution_satisfies_system_l2201_220127


namespace simplify_and_evaluate_evaluate_at_four_l2201_220124

theorem simplify_and_evaluate (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (((2 * x + 2) / (x^2 - 1) + 1) / ((x + 1) / (x^2 - 2*x + 1))) = x - 1 :=
by sorry

theorem evaluate_at_four : 
  (((2 * 4 + 2) / (4^2 - 1) + 1) / ((4 + 1) / (4^2 - 2*4 + 1))) = 3 :=
by sorry

end simplify_and_evaluate_evaluate_at_four_l2201_220124


namespace arctan_equation_solution_l2201_220101

theorem arctan_equation_solution (x : ℝ) : 
  Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 ↔ 
  x = Real.sqrt ((1 + Real.sqrt 5) / 2) ∨ x = -Real.sqrt ((1 + Real.sqrt 5) / 2) :=
by sorry

end arctan_equation_solution_l2201_220101


namespace task_probability_l2201_220107

theorem task_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 5/8) 
  (h2 : p2 = 3/5) 
  (h3 : p3 = 7/10) 
  (h4 : p4 = 9/12) : 
  p1 * (1 - p2) * (1 - p3) * p4 = 9/160 := by
  sorry

end task_probability_l2201_220107


namespace total_value_is_correct_l2201_220151

/-- The number of £5 notes issued by the Bank of England -/
def num_notes : ℕ := 440000000

/-- The face value of each note in pounds -/
def face_value : ℕ := 5

/-- The total face value of all notes in pounds -/
def total_value : ℕ := num_notes * face_value

/-- Theorem: The total face value of all notes is £2,200,000,000 -/
theorem total_value_is_correct : total_value = 2200000000 := by
  sorry

end total_value_is_correct_l2201_220151


namespace total_passengers_per_hour_l2201_220169

/-- Calculates the total number of different passengers stepping on and off trains at a station within an hour -/
theorem total_passengers_per_hour 
  (train_interval : ℕ) 
  (passengers_leaving : ℕ) 
  (passengers_boarding : ℕ) 
  (hour_in_minutes : ℕ) :
  train_interval = 5 →
  passengers_leaving = 200 →
  passengers_boarding = 320 →
  hour_in_minutes = 60 →
  (hour_in_minutes / train_interval) * (passengers_leaving + passengers_boarding) = 6240 := by
  sorry

end total_passengers_per_hour_l2201_220169


namespace parallel_vectors_k_value_l2201_220179

/-- Given vectors a and b in ℝ², if k*a + b is parallel to a + 3*b, then k = 1/3 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ k • a + b = t • (a + 3 • b)) :
  k = 1/3 := by
  sorry

end parallel_vectors_k_value_l2201_220179


namespace parabola_translation_l2201_220183

/-- Given a parabola y = x^2 + bx + c that is translated 3 units right and 4 units down
    to become y = x^2 - 2x + 2, prove that b = 4 and c = 9 -/
theorem parabola_translation (b c : ℝ) : 
  (∀ x y : ℝ, y = x^2 + b*x + c ↔ y + 4 = (x - 3)^2 - 2*(x - 3) + 2) →
  b = 4 ∧ c = 9 := by sorry

end parabola_translation_l2201_220183


namespace prob_white_then_red_is_four_fifteenths_l2201_220119

/-- Represents the number of red marbles in the bag -/
def red_marbles : ℕ := 4

/-- Represents the number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- Represents the total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles

/-- The probability of drawing a white marble first and a red marble second -/
def prob_white_then_red : ℚ :=
  (white_marbles : ℚ) / total_marbles * red_marbles / (total_marbles - 1)

theorem prob_white_then_red_is_four_fifteenths :
  prob_white_then_red = 4 / 15 := by
  sorry

end prob_white_then_red_is_four_fifteenths_l2201_220119


namespace rectangle_difference_l2201_220156

/-- A rectangle with given perimeter and area -/
structure Rectangle where
  length : ℝ
  breadth : ℝ
  perimeter_eq : length + breadth = 93
  area_eq : length * breadth = 2030

/-- The difference between length and breadth of a rectangle with perimeter 186m and area 2030m² is 23m -/
theorem rectangle_difference (rect : Rectangle) : rect.length - rect.breadth = 23 := by
  sorry

end rectangle_difference_l2201_220156


namespace hyperbola_equation_from_focus_and_midpoint_l2201_220147

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

theorem hyperbola_equation_from_focus_and_midpoint 
  (h : Hyperbola)
  (focus : Point)
  (midpoint : Point)
  (h_focus : focus.x = -2 ∧ focus.y = 0)
  (h_midpoint : midpoint.x = -3 ∧ midpoint.y = -1)
  (h_intersect : ∃ (A B : Point), 
    hyperbola_equation h A ∧ 
    hyperbola_equation h B ∧
    (A.x + B.x) / 2 = midpoint.x ∧
    (A.y + B.y) / 2 = midpoint.y) :
  h.a^2 = 3 ∧ h.b^2 = 1 := by
  sorry

end hyperbola_equation_from_focus_and_midpoint_l2201_220147


namespace fourth_number_proof_l2201_220160

theorem fourth_number_proof (n : ℝ) (h1 : n = 27) : 
  let numbers : List ℝ := [3, 16, 33, n + 1]
  (numbers.sum / numbers.length = 20) → (n + 1 = 28) := by
sorry

end fourth_number_proof_l2201_220160


namespace circles_common_points_l2201_220166

/-- Two circles with radii 2 and 3 have common points if and only if 
    the distance between their centers is between 1 and 5 (inclusive). -/
theorem circles_common_points (d : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 ∧ (x - d)^2 + y^2 = 9) ↔ 1 ≤ d ∧ d ≤ 5 :=
sorry

end circles_common_points_l2201_220166


namespace least_n_with_gcd_conditions_l2201_220184

theorem least_n_with_gcd_conditions (n : ℕ) : 
  n > 1500 ∧ 
  Nat.gcd 40 (n + 105) = 10 ∧ 
  Nat.gcd (n + 40) 105 = 35 ∧
  (∀ m : ℕ, m > 1500 → Nat.gcd 40 (m + 105) = 10 → Nat.gcd (m + 40) 105 = 35 → m ≥ n) →
  n = 1511 := by
sorry

end least_n_with_gcd_conditions_l2201_220184


namespace expected_draws_no_ugly_l2201_220190

def bag_total : ℕ := 20
def blue_marbles : ℕ := 9
def ugly_marbles : ℕ := 10
def special_marbles : ℕ := 1

def prob_blue : ℚ := blue_marbles / bag_total
def prob_special : ℚ := special_marbles / bag_total

theorem expected_draws_no_ugly : 
  let p := prob_blue
  let q := prob_special
  (∑' k : ℕ, k * (1 / (1 - p)) * p^(k-1) * q) = 20 / 11 :=
sorry

end expected_draws_no_ugly_l2201_220190


namespace permutation_difference_divisibility_l2201_220171

/-- For any integer n > 2 and any two permutations of {0, 1, ..., n-1},
    there exist distinct indices i and j such that n divides (aᵢ * bᵢ - aⱼ * bⱼ). -/
theorem permutation_difference_divisibility (n : ℕ) (hn : n > 2)
  (a b : Fin n → Fin n) (ha : Function.Bijective a) (hb : Function.Bijective b) :
  ∃ (i j : Fin n), i ≠ j ∧ (n : ℤ) ∣ (a i * b i - a j * b j) :=
sorry

end permutation_difference_divisibility_l2201_220171


namespace unique_solution_mn_l2201_220129

theorem unique_solution_mn : ∃! (m n : ℕ+), 18 * m * n = 63 - 9 * m - 3 * n ∧ m = 7 ∧ n = 4 := by
  sorry

end unique_solution_mn_l2201_220129


namespace broken_bulbs_in_foyer_l2201_220131

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3 / 5

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1 / 3

/-- The number of light bulbs not broken in both the foyer and kitchen -/
def total_not_broken : ℕ := 34

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

theorem broken_bulbs_in_foyer :
  foyer_broken = 10 := by sorry

end broken_bulbs_in_foyer_l2201_220131


namespace equation_solution_l2201_220159

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) ↔ x = 3/2 := by
sorry

end equation_solution_l2201_220159


namespace fourth_root_equation_solution_l2201_220136

theorem fourth_root_equation_solution :
  ∃! x : ℚ, (62 - 3*x)^(1/4) + (38 + 3*x)^(1/4) = 5 := by
  sorry

end fourth_root_equation_solution_l2201_220136


namespace negation_square_nonnegative_l2201_220134

theorem negation_square_nonnegative (x : ℝ) : 
  ¬(x ≥ 0 → x^2 > 0) ↔ (x < 0 → x^2 ≤ 0) :=
sorry

end negation_square_nonnegative_l2201_220134


namespace spade_problem_l2201_220102

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_problem : spade 5 (spade 3 9) = 1 := by sorry

end spade_problem_l2201_220102


namespace min_difference_gcd3_lcm135_l2201_220182

def min_difference_with_gcd_lcm : ℕ → ℕ → ℕ → ℕ → ℕ := sorry

theorem min_difference_gcd3_lcm135 :
  min_difference_with_gcd_lcm 3 135 = 12 :=
by sorry

end min_difference_gcd3_lcm135_l2201_220182


namespace inequality_proof_l2201_220180

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1 / 3) ∧ 
  (a^2 / b + b^2 / c + c^2 / a ≥ 1) :=
by sorry

end inequality_proof_l2201_220180


namespace minimum_value_problems_l2201_220193

theorem minimum_value_problems :
  (∀ x > 0, x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1)) ∧
  (∀ m > 0, (m^2 + 5*m + 12) / m ≥ 4 * Real.sqrt 3 + 5) := by
  sorry

end minimum_value_problems_l2201_220193


namespace tangent_points_line_passes_through_fixed_point_l2201_220185

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Tangent line from a point to a parabola -/
def tangent_line (p : Parabola) (m : Point) : Line :=
  sorry

/-- Tangent point of a line to a parabola -/
def tangent_point (p : Parabola) (l : Line) : Point :=
  sorry

/-- Given a parabola C, a line l, and a point M on l, prove that the line AB 
    formed by the tangent points A and B of the tangent lines from M to C 
    always passes through a fixed point -/
theorem tangent_points_line_passes_through_fixed_point 
  (C : Parabola) 
  (l : Line) 
  (M : Point) 
  (m : ℝ) 
  (h1 : C.equation = fun x y => x^2 = 4*y) 
  (h2 : l.equation = fun x y => y = -m) 
  (h3 : m > 0) 
  (h4 : l.equation M.x M.y) :
  let t1 := tangent_line C M
  let t2 := tangent_line C M
  let A := tangent_point C t1
  let B := tangent_point C t2
  let AB : Line := sorry
  AB.equation 0 m := by sorry

end tangent_points_line_passes_through_fixed_point_l2201_220185


namespace equation_solution_exists_l2201_220161

theorem equation_solution_exists : ∃ (x y : ℕ), x^9 = 2013 * y^10 := by
  sorry

end equation_solution_exists_l2201_220161


namespace distinct_feeding_sequences_l2201_220111

def number_of_pairs : ℕ := 5

def feeding_sequence (n : ℕ) : ℕ := 
  match n with
  | 0 => 1  -- The first animal (male lion) is fixed
  | 1 => number_of_pairs  -- First choice of female
  | k => if k % 2 = 0 then number_of_pairs - k / 2 else number_of_pairs - (k - 1) / 2

theorem distinct_feeding_sequences :
  (List.range (2 * number_of_pairs)).foldl (fun acc i => acc * feeding_sequence i) 1 = 2880 := by
  sorry

end distinct_feeding_sequences_l2201_220111


namespace equation_solution_l2201_220104

theorem equation_solution : ∃ y : ℝ, (3 * y + 7 * y = 282 - 8 * (y - 3)) ∧ y = 17 := by
  sorry

end equation_solution_l2201_220104


namespace ellie_bike_oil_needed_l2201_220133

/-- The amount of oil needed to fix a bicycle --/
def oil_needed (oil_per_wheel : ℕ) (oil_for_rest : ℕ) (num_wheels : ℕ) : ℕ :=
  oil_per_wheel * num_wheels + oil_for_rest

/-- Theorem: The total amount of oil needed to fix Ellie's bike is 25ml --/
theorem ellie_bike_oil_needed :
  oil_needed 10 5 2 = 25 := by
  sorry

end ellie_bike_oil_needed_l2201_220133


namespace zoey_reading_schedule_l2201_220144

def days_to_read (n : ℕ) : ℕ := 2 * n - 1

def total_days (num_books : ℕ) : ℕ := num_books^2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem zoey_reading_schedule :
  let num_books := 18
  let start_day := 1  -- Monday
  let total_reading_days := total_days num_books
  day_of_week start_day total_reading_days = 3  -- Wednesday
:= by sorry

end zoey_reading_schedule_l2201_220144


namespace expression_value_l2201_220164

theorem expression_value : -20 + 8 * (5^2 - 3) = 156 := by
  sorry

end expression_value_l2201_220164


namespace sufficient_not_necessary_l2201_220165

theorem sufficient_not_necessary :
  (∃ x : ℝ, x < -1 ∧ x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end sufficient_not_necessary_l2201_220165


namespace quadratic_is_square_of_binomial_l2201_220139

theorem quadratic_is_square_of_binomial (r : ℝ) (hr : r ≠ 0) :
  ∃ (p q : ℝ), ∀ x, r^2 * x^2 - 20 * x + 100 / r^2 = (p * x + q)^2 := by
  sorry

end quadratic_is_square_of_binomial_l2201_220139


namespace money_calculation_l2201_220186

/-- Calculates the total amount of money given the number of 50 and 500 rupee notes -/
def totalMoney (n50 : ℕ) (n500 : ℕ) : ℕ :=
  50 * n50 + 500 * n500

/-- Proves that the total amount of money is 10350 rupees given the specified conditions -/
theorem money_calculation :
  ∀ (n50 n500 : ℕ),
    n50 = 37 →
    n50 + n500 = 54 →
    totalMoney n50 n500 = 10350 := by
  sorry

#eval totalMoney 37 17  -- Should output 10350

end money_calculation_l2201_220186


namespace sixth_term_of_geometric_sequence_l2201_220155

/-- Given a geometric sequence with 8 terms where the first term is 3 and the last term is 39366,
    prove that the 6th term is 23328. -/
theorem sixth_term_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : 
  (∀ n, a n = 3 * r^(n-1)) →  -- Geometric sequence definition
  a 8 = 39366 →              -- Last term condition
  a 6 = 23328 :=             -- Theorem to prove
by
  sorry

#check sixth_term_of_geometric_sequence

end sixth_term_of_geometric_sequence_l2201_220155
