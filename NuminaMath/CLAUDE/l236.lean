import Mathlib

namespace walking_probability_is_four_sevenths_l236_23606

/-- The number of bus stops -/
def num_stops : ℕ := 15

/-- The distance between adjacent stops in feet -/
def distance_between_stops : ℕ := 100

/-- The maximum walking distance in feet -/
def max_walking_distance : ℕ := 500

/-- The probability of walking 500 feet or less between two randomly chosen stops -/
def walking_probability : ℚ :=
  let total_possibilities := num_stops * (num_stops - 1)
  let favorable_outcomes := 120  -- This is derived from the problem, not the solution
  favorable_outcomes / total_possibilities

theorem walking_probability_is_four_sevenths :
  walking_probability = 4 / 7 := by sorry

end walking_probability_is_four_sevenths_l236_23606


namespace number_in_bases_is_61_l236_23636

/-- Represents a number in different bases -/
def NumberInBases (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    (0 ≤ a ∧ a < 6) ∧
    (0 ≤ b ∧ b < 6) ∧
    n = 36 * a + 6 * b + a ∧
    n = 15 * b + a

theorem number_in_bases_is_61 :
  ∃ (n : ℕ), NumberInBases n ∧ n = 61 :=
sorry

end number_in_bases_is_61_l236_23636


namespace triangle_inequality_l236_23639

/-- A complete graph K_n with n vertices, where each edge is colored either red, green, or blue. -/
structure ColoredCompleteGraph (n : ℕ) where
  n_ge_3 : n ≥ 3

/-- The number of triangles in K_n with all edges of the same color. -/
def monochromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- The number of triangles in K_n with all edges of different colors. -/
def trichromatic_triangles (G : ColoredCompleteGraph n) : ℕ := sorry

/-- Theorem stating the relationship between monochromatic and trichromatic triangles. -/
theorem triangle_inequality (G : ColoredCompleteGraph n) :
  trichromatic_triangles G ≤ 2 * monochromatic_triangles G + n * (n - 1) / 3 := by sorry

end triangle_inequality_l236_23639


namespace complex_fraction_equals_i_l236_23672

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_i : (1 + i^2017) / (1 - i) = i := by sorry

end complex_fraction_equals_i_l236_23672


namespace arithmetic_sequence_general_term_l236_23685

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  a_1_eq_2 : a 1 = 2
  geometric : a 1 * a 4 = a 2 * a 2  -- a_1, a_2, a_4 form a geometric sequence

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end arithmetic_sequence_general_term_l236_23685


namespace quadratic_properties_l236_23660

/-- The quadratic function y = mx^2 - x - m + 1 where m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - x - m + 1

theorem quadratic_properties (m : ℝ) (hm : m ≠ 0) :
  (∀ x, f m x = 0 → x = 1 ∨ x = (1 - m) / m) ∧
  (m < 0 → ∀ a b, f m a = 0 → f m b = 0 → a ≠ b → |a - b| > 2) ∧
  (m > 1 → ∀ x > 1, ∀ y > x, f m y > f m x) :=
by sorry

end quadratic_properties_l236_23660


namespace expression_equality_l236_23690

theorem expression_equality (x : ℝ) : x*(x*(x*(3-2*x)-4)+8)+3*x^2 = -2*x^4 + 3*x^3 - x^2 + 8*x := by
  sorry

end expression_equality_l236_23690


namespace candidate_count_l236_23604

theorem candidate_count (total_selections : ℕ) (h : total_selections = 90) : 
  ∃ n : ℕ, n * (n - 1) = total_selections ∧ n = 10 := by
  sorry

end candidate_count_l236_23604


namespace hike_length_l236_23633

/-- Represents a four-day hike with given conditions -/
structure FourDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  first_two_days : day1 + day2 = 24
  second_third_avg : (day2 + day3) / 2 = 15
  last_two_days : day3 + day4 = 32
  first_third_days : day1 + day3 = 28

/-- The total length of the hike is 56 miles -/
theorem hike_length (h : FourDayHike) : h.day1 + h.day2 + h.day3 + h.day4 = 56 := by
  sorry

end hike_length_l236_23633


namespace train_crossing_time_l236_23688

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_length : ℝ) (platform_length : ℝ) :
  train_speed_kmph = 72 →
  train_length = 280.0416 →
  platform_length = 240 →
  (train_length + platform_length) / (train_speed_kmph * (1 / 3.6)) = 26.00208 := by
  sorry

#check train_crossing_time

end train_crossing_time_l236_23688


namespace power_function_through_point_l236_23664

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x^a) → f 2 = 8 → a = 3 := by
  sorry

end power_function_through_point_l236_23664


namespace basketball_score_possibilities_count_basketball_scores_l236_23642

def basketball_scores (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => 3 * k + 2 * (n - k)) (Finset.range (n + 1))

theorem basketball_score_possibilities :
  basketball_scores 5 = {10, 11, 12, 13, 14, 15} :=
sorry

theorem count_basketball_scores :
  (basketball_scores 5).card = 6 :=
sorry

end basketball_score_possibilities_count_basketball_scores_l236_23642


namespace unique_n_existence_and_value_l236_23601

theorem unique_n_existence_and_value : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 75 := by
  sorry

end unique_n_existence_and_value_l236_23601


namespace inequality_proof_l236_23625

theorem inequality_proof (a b : ℝ) (θ : ℝ) : 
  abs a + abs b ≤ 
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ∧
  Real.sqrt (a^2 * Real.cos θ^2 + b^2 * Real.sin θ^2) + 
  Real.sqrt (a^2 * Real.sin θ^2 + b^2 * Real.cos θ^2) ≤ 
  Real.sqrt (2 * (a^2 + b^2)) :=
by sorry

end inequality_proof_l236_23625


namespace find_y_l236_23654

theorem find_y (c d : ℝ) (y : ℝ) (h1 : d > 0) : 
  ((3 * c) ^ (3 * d) = c^d * y^d) → y = 27 * c^2 := by
sorry

end find_y_l236_23654


namespace intersection_point_is_minus_one_minus_one_l236_23670

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 7 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Theorem stating that (-1, -1) is the unique intersection point
theorem intersection_point_is_minus_one_minus_one :
  ∃! (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -1 ∧ y = -1 := by sorry

end intersection_point_is_minus_one_minus_one_l236_23670


namespace unique_solution_l236_23600

def complex_number (a : ℝ) : ℂ := Complex.mk (a^2 - 2) (3*a - 4)

theorem unique_solution :
  ∃! a : ℝ,
    (complex_number a).re = (complex_number a).im ∧
    (complex_number a).re < 0 ∧
    (complex_number a).im < 0 :=
by
  sorry

end unique_solution_l236_23600


namespace sum_of_distinct_integers_l236_23649

theorem sum_of_distinct_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120 →
  a + b + c + d + e = 33 := by
  sorry

end sum_of_distinct_integers_l236_23649


namespace cubic_root_sum_squared_l236_23655

theorem cubic_root_sum_squared (p q r t : ℝ) : 
  (p + q + r = 8) →
  (p * q + p * r + q * r = 14) →
  (p * q * r = 2) →
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) →
  (t^4 - 16*t^2 - 12*t = -8) := by sorry

end cubic_root_sum_squared_l236_23655


namespace base8_subtraction_to_base4_l236_23630

/-- Converts a number from base 8 to base 10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : ℕ) : List ℕ := sorry

/-- Subtracts two numbers in base 8 --/
def subtractBase8 (a b : ℕ) : ℕ := sorry

theorem base8_subtraction_to_base4 :
  let a := 643
  let b := 257
  let result := subtractBase8 a b
  base10ToBase4 (base8ToBase10 result) = [3, 3, 1, 1, 0] := by sorry

end base8_subtraction_to_base4_l236_23630


namespace compound_molecular_weight_l236_23698

/-- Represents the atomic weight of an element in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Al" => 26.98
  | "O"  => 16.00
  | "C"  => 12.01
  | "N"  => 14.01
  | "H"  => 1.008
  | _    => 0  -- Default case, though not used in this problem

/-- Calculates the molecular weight of a compound given its composition -/
def molecular_weight (Al O C N H : ℕ) : ℝ :=
  Al * atomic_weight "Al" +
  O  * atomic_weight "O"  +
  C  * atomic_weight "C"  +
  N  * atomic_weight "N"  +
  H  * atomic_weight "H"

/-- Theorem stating that the molecular weight of the given compound is 146.022 amu -/
theorem compound_molecular_weight :
  molecular_weight 2 3 1 2 4 = 146.022 := by
  sorry

end compound_molecular_weight_l236_23698


namespace fish_eaten_ratio_l236_23687

def total_rocks : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit : ℕ := 2

def rocks_eaten : ℕ := total_rocks - rocks_left + rocks_spit

theorem fish_eaten_ratio :
  (rocks_eaten : ℚ) / total_rocks = 1 / 2 := by
  sorry

end fish_eaten_ratio_l236_23687


namespace line_intercept_sum_l236_23696

/-- A line in the x-y plane -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- The y-intercept of a line -/
def y_intercept (l : Line) : ℚ := l.y_intercept

/-- The sum of x-intercept and y-intercept of a line -/
def intercept_sum (l : Line) : ℚ := x_intercept l + y_intercept l

theorem line_intercept_sum :
  ∃ (l : Line), l.slope = -3 ∧ l.y_intercept = -13 ∧ intercept_sum l = -52/3 := by
  sorry

end line_intercept_sum_l236_23696


namespace eighteen_times_two_minus_four_l236_23614

theorem eighteen_times_two_minus_four (x : ℝ) : x * 2 = 18 → x - 4 = 5 := by
  sorry

end eighteen_times_two_minus_four_l236_23614


namespace symmetric_points_on_circle_l236_23669

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 1 = 0

-- Define the line equation
def line_equation (x y : ℝ) (c : ℝ) : Prop :=
  2*x + y + c = 0

-- Theorem statement
theorem symmetric_points_on_circle (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_equation x₁ y₁ ∧
    circle_equation x₂ y₂ ∧
    (∃ (x_mid y_mid : ℝ),
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 ∧
      line_equation x_mid y_mid c)) →
  c = 1 :=
by
  sorry

end symmetric_points_on_circle_l236_23669


namespace zongzi_pricing_and_purchase_l236_23697

-- Define the total number of zongzi and total cost
def total_zongzi : ℕ := 1100
def total_cost : ℚ := 3000

-- Define the price ratio between type A and B
def price_ratio : ℚ := 1.2

-- Define the new total number of zongzi and budget
def new_total_zongzi : ℕ := 2600
def new_budget : ℚ := 7000

-- Define the unit prices of type A and B zongzi
def unit_price_B : ℚ := 2.5
def unit_price_A : ℚ := 3

-- Define the maximum number of type A zongzi in the second scenario
def max_type_A : ℕ := 1000

theorem zongzi_pricing_and_purchase :
  -- The cost of purchasing type A is the same as type B
  (total_cost / 2) / unit_price_A = (total_cost / 2) / unit_price_B ∧
  -- The unit price of type A is 1.2 times the unit price of type B
  unit_price_A = price_ratio * unit_price_B ∧
  -- The total number of zongzi purchased is 1100
  (total_cost / 2) / unit_price_A + (total_cost / 2) / unit_price_B = total_zongzi ∧
  -- The maximum number of type A zongzi in the second scenario is 1000
  max_type_A * unit_price_A + (new_total_zongzi - max_type_A) * unit_price_B ≤ new_budget ∧
  ∀ n : ℕ, n > max_type_A → n * unit_price_A + (new_total_zongzi - n) * unit_price_B > new_budget :=
by sorry

end zongzi_pricing_and_purchase_l236_23697


namespace star_18_6_l236_23638

/-- The star operation defined for integers -/
def star (a b : ℤ) : ℚ := a - a / b

/-- Theorem stating that 18 ★ 6 = 15 -/
theorem star_18_6 : star 18 6 = 15 := by sorry

end star_18_6_l236_23638


namespace complex_square_root_l236_23675

theorem complex_square_root : 
  ∃ (z₁ z₂ : ℂ), z₁^2 = -45 + 28*I ∧ z₂^2 = -45 + 28*I ∧ 
  z₁ = 2 + 7*I ∧ z₂ = -2 - 7*I ∧
  ∀ (z : ℂ), z^2 = -45 + 28*I → z = z₁ ∨ z = z₂ := by
sorry

end complex_square_root_l236_23675


namespace parabola_single_intersection_l236_23684

/-- A parabola defined by y = x^2 + 2x + c + 1 -/
def parabola (c : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + c + 1

/-- A horizontal line defined by y = 1 -/
def line : ℝ → ℝ := λ _ => 1

/-- The condition for the parabola to intersect the line at only one point -/
def single_intersection (c : ℝ) : Prop :=
  ∃! x, parabola c x = line x

theorem parabola_single_intersection :
  ∀ c : ℝ, single_intersection c ↔ c = 1 := by sorry

end parabola_single_intersection_l236_23684


namespace intersection_empty_union_real_l236_23647

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 2*a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 1}

-- Theorem for part I
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem for part II
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end intersection_empty_union_real_l236_23647


namespace john_earnings_l236_23645

/-- Calculates the money earned by John for repairing cars -/
def money_earned (total_cars : ℕ) (standard_cars : ℕ) (standard_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  let remaining_cars := total_cars - standard_cars
  let standard_total_time := standard_cars * standard_time
  let remaining_time := remaining_cars * (standard_time + standard_time / 2)
  let total_time := standard_total_time + remaining_time
  let total_hours := (total_time + 59) / 60  -- Ceiling division
  hourly_rate * total_hours

/-- Theorem stating that John earns $80 for repairing the cars -/
theorem john_earnings : money_earned 5 3 40 20 = 80 := by
  sorry

end john_earnings_l236_23645


namespace least_colors_for_hidden_edges_l236_23626

/-- The size of the grid (both width and height) -/
def gridSize : ℕ := 7

/-- The total number of edges in the grid -/
def totalEdges : ℕ := 2 * gridSize * (gridSize - 1)

/-- The expected number of hidden edges given N colors -/
def expectedHiddenEdges (N : ℕ) : ℚ := totalEdges / N

/-- Theorem stating the least N for which the expected number of hidden edges is less than 3 -/
theorem least_colors_for_hidden_edges :
  ∀ N : ℕ, N ≥ 29 ↔ expectedHiddenEdges N < 3 :=
sorry

end least_colors_for_hidden_edges_l236_23626


namespace concert_hall_audience_l236_23631

theorem concert_hall_audience (total_seats : ℕ) 
  (h_total : total_seats = 1260)
  (h_glasses : (7 : ℚ) / 18 * total_seats = number_with_glasses)
  (h_male_no_glasses : (6 : ℚ) / 11 * (total_seats - number_with_glasses) = number_male_no_glasses) :
  number_male_no_glasses = 420 := by
  sorry

end concert_hall_audience_l236_23631


namespace fuel_tank_ethanol_percentage_l236_23663

/-- Fuel tank problem -/
theorem fuel_tank_ethanol_percentage
  (tank_capacity : ℝ)
  (fuel_a_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_ethanol_percentage = 12 / 100)
  (h3 : total_ethanol = 30)
  (h4 : fuel_a_volume = 106) :
  (total_ethanol - fuel_a_ethanol_percentage * fuel_a_volume) / (tank_capacity - fuel_a_volume) = 16 / 100 := by
sorry

end fuel_tank_ethanol_percentage_l236_23663


namespace plate_arrangement_circular_table_l236_23624

def plate_arrangement (b r g o y : ℕ) : ℕ :=
  let total := b + r + g + o + y
  let all_arrangements := Nat.factorial (total - 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial g * Nat.factorial o * Nat.factorial y)
  let adjacent_green := Nat.factorial (total - g + 1) / (Nat.factorial b * Nat.factorial r * Nat.factorial o * Nat.factorial y) * Nat.factorial g
  all_arrangements - adjacent_green

theorem plate_arrangement_circular_table :
  plate_arrangement 6 3 3 2 2 = 
    Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2) - 
    (Nat.factorial 14 / (Nat.factorial 6 * Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1) * Nat.factorial 3) :=
by
  sorry

end plate_arrangement_circular_table_l236_23624


namespace discount_comparison_l236_23620

/-- The cost difference between Option 2 and Option 1 for buying suits and ties -/
def cost_difference (x : ℝ) : ℝ :=
  (3600 + 36*x) - (40*x + 3200)

theorem discount_comparison (x : ℝ) (h : x > 20) :
  cost_difference x ≥ 0 ∧ cost_difference 30 > 0 := by
  sorry

#eval cost_difference 30

end discount_comparison_l236_23620


namespace train_length_calculation_l236_23615

/-- Represents the properties of a train and its movement --/
structure Train where
  length : ℝ
  speed : ℝ
  platform_crossing_time : ℝ
  pole_crossing_time : ℝ
  platform_length : ℝ

/-- Theorem stating the length of the train given specific conditions --/
theorem train_length_calculation (t : Train)
  (h1 : t.platform_crossing_time = 39)
  (h2 : t.pole_crossing_time = 16)
  (h3 : t.platform_length = 431.25)
  (h4 : t.length = t.speed * t.pole_crossing_time)
  (h5 : t.length + t.platform_length = t.speed * t.platform_crossing_time) :
  t.length = 6890 / 23 := by
  sorry

#check train_length_calculation

end train_length_calculation_l236_23615


namespace roots_and_element_imply_value_l236_23674

theorem roots_and_element_imply_value (a : ℝ) :
  let A := {x : ℝ | (x - a) * (x - a + 1) = 0}
  2 ∈ A → (a = 2 ∨ a = 3) := by
  sorry

end roots_and_element_imply_value_l236_23674


namespace power_function_property_l236_23693

/-- Given a function f(x) = x^α where f(2) = 4, prove that f(-1) = 1 -/
theorem power_function_property (α : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x ^ α) 
  (h2 : f 2 = 4) : 
  f (-1) = 1 := by
sorry

end power_function_property_l236_23693


namespace sin_2017pi_over_6_l236_23681

theorem sin_2017pi_over_6 : Real.sin ((2017 * π) / 6) = 1 / 2 := by
  sorry

end sin_2017pi_over_6_l236_23681


namespace n_minus_m_not_odd_l236_23680

theorem n_minus_m_not_odd (n m : ℤ) (h : Even (n^2 - m^2)) : ¬Odd (n - m) := by
  sorry

end n_minus_m_not_odd_l236_23680


namespace never_equal_amounts_l236_23641

/-- Represents the currencies in Dillie and Dallie -/
inductive Currency
| Diller
| Daller

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillers : ℕ)
  (dallers : ℕ)

/-- Represents a currency exchange -/
inductive Exchange
| ToDallers
| ToDillers

/-- The exchange rate from dillers to dallers -/
def dillerToDallerRate : ℕ := 10

/-- The exchange rate from dallers to dillers -/
def dallerToDillerRate : ℕ := 10

/-- Perform a single exchange -/
def performExchange (state : MoneyState) (exchange : Exchange) : MoneyState :=
  match exchange with
  | Exchange.ToDallers => 
      { dillers := state.dillers / dillerToDallerRate,
        dallers := state.dallers + state.dillers * dillerToDallerRate }
  | Exchange.ToDillers => 
      { dillers := state.dillers + state.dallers * dallerToDillerRate,
        dallers := state.dallers / dallerToDillerRate }

/-- The initial state of the financier's money -/
def initialState : MoneyState := { dillers := 1, dallers := 0 }

/-- The main theorem to prove -/
theorem never_equal_amounts (exchanges : List Exchange) :
  let finalState := exchanges.foldl performExchange initialState
  finalState.dillers ≠ finalState.dallers :=
sorry

end never_equal_amounts_l236_23641


namespace sine_inequality_solution_l236_23619

theorem sine_inequality_solution (x y : Real) :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), 
   ∀ y ∈ Set.Icc 0 (2 * Real.pi), 
   Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔ 
  y ∈ Set.Icc 0 Real.pi :=
sorry

end sine_inequality_solution_l236_23619


namespace total_spears_l236_23627

/-- The number of spears that can be made from one sapling -/
def spears_per_sapling : ℕ := 3

/-- The number of spears that can be made from one log -/
def spears_per_log : ℕ := 9

/-- The number of saplings available -/
def num_saplings : ℕ := 6

/-- The number of logs available -/
def num_logs : ℕ := 1

/-- Theorem: The total number of spears Marcy can make is 27 -/
theorem total_spears : 
  spears_per_sapling * num_saplings + spears_per_log * num_logs = 27 := by
  sorry


end total_spears_l236_23627


namespace opposite_plus_two_equals_zero_l236_23695

theorem opposite_plus_two_equals_zero (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  sorry

end opposite_plus_two_equals_zero_l236_23695


namespace geometric_sequence_sum_l236_23662

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a₁ = 3
  a 4 = 24 →                                 -- a₄ = 24
  a 3 + a 4 + a 5 = 84 :=                    -- prove a₃ + a₄ + a₅ = 84
by
  sorry

end geometric_sequence_sum_l236_23662


namespace chord_equation_l236_23640

/-- Given a circle and a chord, prove the equation of the chord --/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - P.1)^2 + (y - P.2)^2) →
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = (x - Q.1)^2 + (y - Q.2)^2) →
  (P.1 + Q.1) / 2 = 1 →
  (P.2 + Q.2) / 2 = 2 →
  ∃ (k : ℝ), ∀ (x y : ℝ), (y - P.2) = k * (x - P.1) ∧ (y - Q.2) = k * (x - Q.1) →
    x + 2*y - 5 = 0 := by
  sorry

end chord_equation_l236_23640


namespace sum_product_theorem_l236_23643

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 12)
  (eq4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := by
sorry

end sum_product_theorem_l236_23643


namespace defective_units_shipped_percentage_l236_23682

/-- Given that 6% of units produced are defective and 0.24% of total units
    are defective and shipped for sale, prove that 4% of defective units
    are shipped for sale. -/
theorem defective_units_shipped_percentage
  (total_defective_percent : ℝ)
  (defective_shipped_percent : ℝ)
  (h1 : total_defective_percent = 6)
  (h2 : defective_shipped_percent = 0.24) :
  defective_shipped_percent / total_defective_percent * 100 = 4 := by
  sorry

end defective_units_shipped_percentage_l236_23682


namespace seventh_degree_equation_reduction_l236_23637

theorem seventh_degree_equation_reduction (a b : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = x^7 - 7*a*x^5 + 14*a^2*x^3 - 7*a^3*x - b) →
    (∃ α β : ℝ, α * β = a ∧ α^7 + β^7 = b ∧ f α = 0 ∧ f β = 0) :=
by sorry

end seventh_degree_equation_reduction_l236_23637


namespace intersection_of_A_and_B_l236_23673

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 3} := by sorry

end intersection_of_A_and_B_l236_23673


namespace subset_implication_l236_23603

theorem subset_implication (A B : Set ℕ) (a : ℕ) :
  A = {1, a} ∧ B = {1, 2, 3} →
  (a = 3 → A ⊆ B) := by
  sorry

end subset_implication_l236_23603


namespace airplane_distance_difference_l236_23646

/-- Theorem: Distance difference for an airplane flying with and against wind -/
theorem airplane_distance_difference (a : ℝ) : 
  let windless_speed : ℝ := a
  let wind_speed : ℝ := 20
  let time_without_wind : ℝ := 4
  let time_against_wind : ℝ := 3
  windless_speed * time_without_wind - (windless_speed - wind_speed) * time_against_wind = a + 60 := by
  sorry

end airplane_distance_difference_l236_23646


namespace square_area_ratio_l236_23629

theorem square_area_ratio (y : ℝ) (hy : y > 0) : 
  (y^2) / ((3*y)^2) = 1/9 := by sorry

end square_area_ratio_l236_23629


namespace product_of_consecutive_integers_l236_23644

theorem product_of_consecutive_integers (n : ℕ) : 
  n = 5 → (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) = 30240 := by
  sorry

end product_of_consecutive_integers_l236_23644


namespace monomial_exponents_l236_23609

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def are_like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ i, m1 i = m2 i

theorem monomial_exponents (a b : ℕ) :
  are_like_terms (fun i => if i = 0 then a + 1 else if i = 1 then 3 else 0)
                 (fun i => if i = 0 then 2 else if i = 1 then b else 0) →
  a = 1 ∧ b = 3 := by
  sorry

end monomial_exponents_l236_23609


namespace combination_problem_classification_l236_23689

-- Define a type for the scenarios
inductive Scenario
| sets_two_elements
| round_robin_tournament
| two_digit_number_formation
| two_digit_number_no_repeat

-- Define what it means for a scenario to be a combination problem
def is_combination_problem (s : Scenario) : Prop :=
  match s with
  | Scenario.sets_two_elements => True
  | Scenario.round_robin_tournament => True
  | Scenario.two_digit_number_formation => False
  | Scenario.two_digit_number_no_repeat => False

-- Theorem statement
theorem combination_problem_classification :
  (is_combination_problem Scenario.sets_two_elements) ∧
  (is_combination_problem Scenario.round_robin_tournament) ∧
  (¬ is_combination_problem Scenario.two_digit_number_formation) ∧
  (¬ is_combination_problem Scenario.two_digit_number_no_repeat) := by
  sorry


end combination_problem_classification_l236_23689


namespace circle_equation_proof_l236_23623

theorem circle_equation_proof (x y : ℝ) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 4*x + 6*y - 3 = 0
  let M : ℝ × ℝ := (-1, 1)
  let new_circle : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 3)^2 = 25
  (∃ h k r : ℝ, ∀ x y : ℝ, C x y ↔ (x - h)^2 + (y - k)^2 = r^2) →
  (new_circle M.1 M.2) ∧
  (∀ x y : ℝ, C x y ↔ (x - 2)^2 + (y + 3)^2 = r^2) →
  (∀ x y : ℝ, new_circle x y ↔ (x - 2)^2 + (y + 3)^2 = 25) :=
by sorry


end circle_equation_proof_l236_23623


namespace folded_paper_corner_distance_l236_23678

/-- Represents a square sheet of paper with white front and black back -/
structure Paper where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

/-- Represents the folded state of the paper -/
structure FoldedPaper where
  paper : Paper
  fold_length : ℝ
  black_area : ℝ
  white_area : ℝ
  black_twice_white : black_area = 2 * white_area
  areas_sum : black_area + white_area = paper.area

/-- The theorem to be proved -/
theorem folded_paper_corner_distance 
  (p : Paper) 
  (fp : FoldedPaper) 
  (h_area : p.area = 18) 
  (h_fp_paper : fp.paper = p) :
  Real.sqrt 2 * fp.fold_length = 4 * Real.sqrt 3 := by
  sorry

end folded_paper_corner_distance_l236_23678


namespace right_triangles_on_circle_l236_23632

/-- The number of right-angled triangles formed by 2n equally spaced points on a circle -/
theorem right_triangles_on_circle (n : ℕ) (h : n > 1) :
  (number_of_right_triangles : ℕ) = 2 * n * (n - 1) :=
by sorry

end right_triangles_on_circle_l236_23632


namespace combined_height_problem_l236_23653

/-- Given that Mr. Martinez is two feet taller than Chiquita and Chiquita is 5 feet tall,
    prove that their combined height is 12 feet. -/
theorem combined_height_problem (chiquita_height : ℝ) (martinez_height : ℝ) :
  chiquita_height = 5 →
  martinez_height = chiquita_height + 2 →
  chiquita_height + martinez_height = 12 :=
by sorry

end combined_height_problem_l236_23653


namespace jane_drawing_paper_l236_23651

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets : ℕ := 28

/-- The number of old, yellow sheets of drawing paper Jane has -/
def yellow_sheets : ℕ := 27

/-- The total number of drawing paper sheets Jane has -/
def total_sheets : ℕ := brown_sheets + yellow_sheets

theorem jane_drawing_paper : total_sheets = 55 := by
  sorry

end jane_drawing_paper_l236_23651


namespace negation_of_existential_proposition_l236_23634

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by
  sorry

end negation_of_existential_proposition_l236_23634


namespace max_values_on_unit_circle_l236_23657

theorem max_values_on_unit_circle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 = 1) :
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a + b ≥ x + y) ∧
  (a + b ≤ Real.sqrt 2) ∧
  (∀ x y, x > 0 → y > 0 → x^2 + y^2 = 1 → a * b ≥ x * y) ∧
  (a * b ≤ 1/2) := by
sorry


end max_values_on_unit_circle_l236_23657


namespace paris_saturday_study_hours_l236_23691

/-- The number of hours Paris studies on Saturdays during the semester -/
def saturday_study_hours (
  semester_weeks : ℕ)
  (weekday_study_hours : ℕ)
  (sunday_study_hours : ℕ)
  (total_study_hours : ℕ) : ℕ :=
  total_study_hours - (semester_weeks * 5 * weekday_study_hours) - (semester_weeks * sunday_study_hours)

/-- Theorem stating that Paris studies 60 hours on Saturdays during the semester -/
theorem paris_saturday_study_hours :
  saturday_study_hours 15 3 5 360 = 60 := by
  sorry


end paris_saturday_study_hours_l236_23691


namespace atLeastOneMale_and_allFemales_mutuallyExclusive_l236_23616

/-- Represents the outcome of selecting 2 students from the group -/
inductive Selection
| TwoMales
| OneMaleOneFemale
| TwoFemales

/-- The sample space of all possible selections -/
def sampleSpace : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale, Selection.TwoFemales}

/-- The event "At least 1 male student" -/
def atLeastOneMale : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale}

/-- The event "All female students" -/
def allFemales : Set Selection :=
  {Selection.TwoFemales}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set Selection) : Prop :=
  A ∩ B = ∅

theorem atLeastOneMale_and_allFemales_mutuallyExclusive :
  mutuallyExclusive atLeastOneMale allFemales :=
sorry

end atLeastOneMale_and_allFemales_mutuallyExclusive_l236_23616


namespace dot_product_range_l236_23652

/-- The ellipse equation -/
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 16) + (P.2^2 / 15) = 1

/-- The circle equation -/
def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

/-- Definition of a diameter of the circle -/
def is_diameter (E F : ℝ × ℝ) : Prop :=
  is_on_circle E ∧ is_on_circle F ∧ 
  (E.1 + F.1 = 2) ∧ (E.2 + F.2 = 0)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem dot_product_range (P E F : ℝ × ℝ) :
  is_on_ellipse P → is_diameter E F →
  5 ≤ dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ∧
  dot_product (E.1 - P.1, E.2 - P.2) (F.1 - P.1, F.2 - P.2) ≤ 21 :=
by sorry

end dot_product_range_l236_23652


namespace abs_equation_solution_l236_23661

theorem abs_equation_solution :
  ∃! y : ℝ, |y - 6| + 3*y = 12 :=
by
  -- The unique solution is y = 3
  use 3
  sorry

end abs_equation_solution_l236_23661


namespace emma_remaining_amount_l236_23658

def calculate_remaining_amount (initial_amount furniture_cost fraction_given : ℚ) : ℚ :=
  let remaining_after_furniture := initial_amount - furniture_cost
  let amount_given := fraction_given * remaining_after_furniture
  remaining_after_furniture - amount_given

theorem emma_remaining_amount :
  calculate_remaining_amount 2000 400 (3/4) = 400 := by
  sorry

end emma_remaining_amount_l236_23658


namespace power_of_three_difference_l236_23622

theorem power_of_three_difference : 3^(1+3+4) - (3^1 + 3^3 + 3^4) = 6450 := by
  sorry

end power_of_three_difference_l236_23622


namespace digit_150_is_7_l236_23665

/-- The decimal representation of 17/70 -/
def decimal_rep : ℚ := 17 / 70

/-- The length of the repeating sequence in the decimal representation of 17/70 -/
def repeat_length : ℕ := 6

/-- The nth digit after the decimal point in the decimal representation of 17/70 -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_is_7 : nth_digit 150 = 7 := by sorry

end digit_150_is_7_l236_23665


namespace license_plate_count_l236_23648

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The number of possible letters -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_in_plate : ℕ := 5

/-- The number of letters in a license plate -/
def letters_in_plate : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_in_plate + 1

/-- The total number of distinct license plates -/
def total_plates : ℕ := block_positions * (num_digits ^ digits_in_plate) * (num_letters ^ letters_in_plate)

theorem license_plate_count : total_plates = 105456000 := by sorry

end license_plate_count_l236_23648


namespace unique_solution_l236_23612

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end unique_solution_l236_23612


namespace birth_year_problem_l236_23605

theorem birth_year_problem (x : ℕ) : 
  (1800 ≤ x^2 - x) ∧ (x^2 - x < 1850) ∧ (x^2 = x + 1806) → x^2 - x = 1806 :=
by sorry

end birth_year_problem_l236_23605


namespace kids_difference_l236_23694

/-- The number of kids Julia played with on Monday and Tuesday, and the difference between them. -/
def tag_game (monday tuesday : ℕ) : Prop :=
  monday = 16 ∧ tuesday = 4 ∧ monday - tuesday = 12

/-- Theorem stating the difference in the number of kids Julia played with. -/
theorem kids_difference : ∃ (monday tuesday : ℕ), tag_game monday tuesday :=
  sorry

end kids_difference_l236_23694


namespace vector_subtraction_l236_23683

/-- Given vectors a and b in ℝ², prove that a - 2b = (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end vector_subtraction_l236_23683


namespace triangle_segment_length_l236_23602

structure Triangle :=
  (A B C : ℝ × ℝ)

def angleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry  -- Definition of angle bisector

theorem triangle_segment_length 
  (ABC : Triangle) 
  (D : ℝ × ℝ) 
  (h_bisector : angleBisector ABC D)
  (h_AD : dist D ABC.A = 15)
  (h_DC : dist D ABC.C = 45)
  (h_DB : dist D ABC.B = 24) :
  dist ABC.A ABC.B = 39 :=
sorry

#check triangle_segment_length

end triangle_segment_length_l236_23602


namespace nutmeg_amount_l236_23677

theorem nutmeg_amount (cinnamon : Float) (difference : Float) (nutmeg : Float) : 
  cinnamon = 0.67 → 
  difference = 0.17 →
  cinnamon = nutmeg + difference →
  nutmeg = 0.50 := by
  sorry

end nutmeg_amount_l236_23677


namespace quadratic_inequality_solution_set_l236_23656

-- Define the set of real numbers between -3 and 2
def OpenInterval : Set ℝ := {x | -3 < x ∧ x < 2}

-- Define the quadratic function ax^2 + bx + c
def QuadraticF (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reversed quadratic function cx^2 + bx + a
def ReversedQuadraticF (a b c : ℝ) (x : ℝ) : ℝ := c * x^2 + b * x + a

-- Define the solution set of the reversed quadratic inequality
def ReversedSolutionSet : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x ∈ OpenInterval, QuadraticF a b c x > 0) :
  ∀ x, ReversedQuadraticF a b c x > 0 ↔ x ∈ ReversedSolutionSet :=
sorry

end quadratic_inequality_solution_set_l236_23656


namespace multiple_of_smaller_number_l236_23621

theorem multiple_of_smaller_number (s l : ℝ) (h1 : s + l = 24) (h2 : s = 10) : ∃ m : ℝ, m * s = 5 * l ∧ m = 7 := by
  sorry

end multiple_of_smaller_number_l236_23621


namespace tom_balloons_l236_23610

theorem tom_balloons (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 30 → given = 16 → remaining = initial - given → remaining = 14 := by
  sorry

end tom_balloons_l236_23610


namespace trig_product_equality_l236_23679

theorem trig_product_equality : 
  Real.sin (4 * Real.pi / 3) * Real.cos (5 * Real.pi / 6) * Real.tan (-4 * Real.pi / 3) = -3 * Real.sqrt 3 / 4 := by
  sorry

end trig_product_equality_l236_23679


namespace three_objects_five_containers_l236_23607

/-- The number of ways to place n distinct objects into m distinct containers -/
def placement_count (n m : ℕ) : ℕ := m^n

/-- Theorem: Placing 3 distinct objects into 5 distinct containers results in 125 different arrangements -/
theorem three_objects_five_containers : placement_count 3 5 = 125 := by
  sorry

end three_objects_five_containers_l236_23607


namespace angle_bisector_inequality_l236_23699

-- Define a triangle with side lengths and angle bisectors
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_a : t_a = (b * c * (a + c - b).sqrt) / (b + c)
  bisector_b : t_b = (a * c * (b + c - a).sqrt) / (a + c)

-- State the theorem
theorem angle_bisector_inequality (t : Triangle) : (t.t_a + t.t_b) / (t.a + t.b) < 4/3 := by
  sorry

end angle_bisector_inequality_l236_23699


namespace fraction_simplification_l236_23611

theorem fraction_simplification (x y z : ℝ) (h : x + y + z = 3) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (x*y + y*z + z*x) / (9 - 2*(x*y + y*z + z*x)) :=
by sorry

end fraction_simplification_l236_23611


namespace exists_universal_program_l236_23628

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a labyrinth configuration --/
def Labyrinth := Fin 8 → Fin 8 → Bool

/-- Represents a move command --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a program as a list of commands --/
def Program := List Command

/-- Checks if a square is accessible in the labyrinth --/
def isAccessible (lab : Labyrinth) (pos : Position) : Bool :=
  lab pos.x pos.y

/-- Executes a single command on a position in a labyrinth --/
def executeCommand (lab : Labyrinth) (pos : Position) (cmd : Command) : Position :=
  sorry

/-- Executes a program on a position in a labyrinth --/
def executeProgram (lab : Labyrinth) (pos : Position) (prog : Program) : Position :=
  sorry

/-- Checks if a program visits all accessible squares in a labyrinth from a given starting position --/
def visitsAllAccessible (lab : Labyrinth) (start : Position) (prog : Program) : Prop :=
  sorry

/-- The main theorem: there exists a program that visits all accessible squares
    for any labyrinth and starting position --/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (lab : Labyrinth) (start : Position),
      visitsAllAccessible lab start prog :=
sorry

end exists_universal_program_l236_23628


namespace original_number_proof_l236_23618

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 10^4 * x = 4 * (1/x)) : x = 0.02 := by
  sorry

end original_number_proof_l236_23618


namespace b_score_is_93_l236_23668

/-- Represents the scores of five people in an exam -/
structure ExamScores where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The average score of all five people is 90 -/
def average_all (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C + scores.D + scores.E) / 5 = 90

/-- The average score of A, B, and C is 86 -/
def average_ABC (scores : ExamScores) : Prop :=
  (scores.A + scores.B + scores.C) / 3 = 86

/-- The average score of B, D, and E is 95 -/
def average_BDE (scores : ExamScores) : Prop :=
  (scores.B + scores.D + scores.E) / 3 = 95

/-- Theorem: Given the conditions, B's score is 93 -/
theorem b_score_is_93 (scores : ExamScores) 
  (h1 : average_all scores) 
  (h2 : average_ABC scores) 
  (h3 : average_BDE scores) : 
  scores.B = 93 := by
  sorry

end b_score_is_93_l236_23668


namespace rectangle_area_stage_7_l236_23635

/-- The side length of each square in inches -/
def square_side : ℝ := 4

/-- The number of squares at Stage 7 -/
def num_squares : ℕ := 7

/-- The area of the rectangle at Stage 7 in square inches -/
def rectangle_area : ℝ := (square_side ^ 2) * num_squares

/-- Theorem: The area of the rectangle at Stage 7 is 112 square inches -/
theorem rectangle_area_stage_7 : rectangle_area = 112 := by
  sorry

end rectangle_area_stage_7_l236_23635


namespace multiple_birth_statistics_l236_23666

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1250 →
  quadruplets = 3 * quintuplets →
  triplets = 2 * quadruplets →
  twins = 2 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 6250 / 59 := by
  sorry

end multiple_birth_statistics_l236_23666


namespace jerry_butterflies_l236_23686

/-- The number of butterflies Jerry originally had -/
def original_butterflies : ℕ := 93

/-- The number of butterflies Jerry let go -/
def butterflies_let_go : ℕ := 11

/-- The number of butterflies Jerry has left -/
def butterflies_left : ℕ := 82

/-- Theorem: Jerry originally had 93 butterflies -/
theorem jerry_butterflies : original_butterflies = butterflies_let_go + butterflies_left := by
  sorry

end jerry_butterflies_l236_23686


namespace reduce_to_single_digit_l236_23692

/-- Represents the operation of splitting digits and summing -/
def digitSplitSum (n : ℕ) : ℕ → ℕ :=
  sorry

/-- Predicate for a number being single-digit -/
def isSingleDigit (n : ℕ) : Prop :=
  n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) :
  ∃ (seq : Fin 16 → ℕ), seq 0 = N ∧ isSingleDigit (seq 15) ∧
  ∀ i : Fin 15, seq (i + 1) = digitSplitSum (seq i) (seq i) :=
sorry

end reduce_to_single_digit_l236_23692


namespace trapezium_side_length_l236_23608

/-- Given a trapezium with area 342 cm², one parallel side of 14 cm, and height 18 cm,
    prove that the length of the other parallel side is 24 cm. -/
theorem trapezium_side_length (area : ℝ) (side1 : ℝ) (height : ℝ) (side2 : ℝ) :
  area = 342 →
  side1 = 14 →
  height = 18 →
  area = (1 / 2) * (side1 + side2) * height →
  side2 = 24 := by
  sorry

end trapezium_side_length_l236_23608


namespace part_one_part_two_l236_23676

-- Define the function f
def f (a : ℝ) (n : ℕ+) (x : ℝ) : ℝ := a * x^n.val * (1 - x)

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x > 0, f a 2 x ≤ 4/27) ∧ (∃ x > 0, f a 2 x = 4/27) → a = 1 := by sorry

-- Part 2
theorem part_two (n : ℕ+) (m : ℝ) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x ≠ y ∧ f 1 n x = m ∧ f 1 n y = m) →
  0 < m ∧ m < (n.val ^ n.val : ℝ) / ((n.val + 1 : ℕ) ^ (n.val + 1)) := by sorry

end part_one_part_two_l236_23676


namespace pauls_coupon_percentage_l236_23659

def initial_cost : ℝ := 350
def store_discount_percent : ℝ := 20
def final_price : ℝ := 252

theorem pauls_coupon_percentage :
  ∃ (coupon_percent : ℝ),
    final_price = initial_cost * (1 - store_discount_percent / 100) * (1 - coupon_percent / 100) ∧
    coupon_percent = 10 := by
  sorry

end pauls_coupon_percentage_l236_23659


namespace team_b_four_wins_prob_l236_23613

/-- Represents a team in the tournament -/
inductive Team
  | A
  | B
  | C

/-- The probability of one team beating another -/
def beat_prob (winner loser : Team) : ℝ :=
  match winner, loser with
  | Team.A, Team.B => 0.4
  | Team.B, Team.C => 0.5
  | Team.C, Team.A => 0.6
  | _, _ => 0 -- For other combinations, we set probability to 0

/-- The probability of Team B winning four consecutive matches -/
def team_b_four_wins : ℝ :=
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C) * 
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C)

theorem team_b_four_wins_prob : team_b_four_wins = 0.09 := by
  sorry

end team_b_four_wins_prob_l236_23613


namespace divisibility_implication_l236_23650

theorem divisibility_implication (k : ℕ) : 
  (∃ k, 7^17 + 17 * 3 - 1 = 9 * k) → 
  (∃ m, 7^18 + 18 * 3 - 1 = 9 * m) := by
sorry

end divisibility_implication_l236_23650


namespace odd_function_geometric_sequence_l236_23667

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else -(Real.log (-x) + a / (-x))

theorem odd_function_geometric_sequence (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    x₁ + x₄ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧
      f a x₂ / f a x₁ = q ∧
      f a x₃ / f a x₂ = q ∧
      f a x₄ / f a x₃ = q) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
sorry

end odd_function_geometric_sequence_l236_23667


namespace solve_slurpee_problem_l236_23617

def slurpee_problem (money_given : ℝ) (change_received : ℝ) (num_slurpees : ℕ) : Prop :=
  let total_spent := money_given - change_received
  let cost_per_slurpee := total_spent / num_slurpees
  cost_per_slurpee = 2

theorem solve_slurpee_problem :
  slurpee_problem 20 8 6 := by
  sorry

end solve_slurpee_problem_l236_23617


namespace sqrt_six_equals_r_squared_minus_five_over_two_l236_23671

theorem sqrt_six_equals_r_squared_minus_five_over_two :
  Real.sqrt 6 = ((Real.sqrt 2 + Real.sqrt 3)^2 - 5) / 2 := by sorry

end sqrt_six_equals_r_squared_minus_five_over_two_l236_23671
