import Mathlib

namespace NUMINAMATH_CALUDE_golden_ratio_cosine_l4009_400950

theorem golden_ratio_cosine (golden_ratio : ℝ) (h1 : golden_ratio = (Real.sqrt 5 - 1) / 2) 
  (h2 : golden_ratio = 2 * Real.sin (18 * π / 180)) : 
  Real.cos (36 * π / 180) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_cosine_l4009_400950


namespace NUMINAMATH_CALUDE_stating_min_time_to_find_faulty_bulb_l4009_400941

/-- Represents the time in seconds for a single bulb operation (screwing or unscrewing) -/
def bulb_operation_time : ℕ := 10

/-- Represents the total number of bulbs in the series -/
def total_bulbs : ℕ := 4

/-- Represents the number of spare bulbs available -/
def spare_bulbs : ℕ := 1

/-- Represents the number of faulty bulbs in the series -/
def faulty_bulbs : ℕ := 1

/-- 
Theorem stating that the minimum time to identify a faulty bulb 
in a series of 4 bulbs is 60 seconds, given the conditions of the problem.
-/
theorem min_time_to_find_faulty_bulb : 
  (bulb_operation_time * 2 * (total_bulbs - 1) : ℕ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_time_to_find_faulty_bulb_l4009_400941


namespace NUMINAMATH_CALUDE_john_ate_half_package_l4009_400926

/-- Calculates the fraction of a candy package eaten based on servings, calories per serving, and calories consumed. -/
def fraction_eaten (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving)

/-- Proves that John ate half of the candy package -/
theorem john_ate_half_package : 
  let servings : ℕ := 3
  let calories_per_serving : ℕ := 120
  let calories_consumed : ℕ := 180
  fraction_eaten servings calories_per_serving calories_consumed = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_john_ate_half_package_l4009_400926


namespace NUMINAMATH_CALUDE_remainder_7835_mod_11_l4009_400972

theorem remainder_7835_mod_11 : 7835 % 11 = (7 + 8 + 3 + 5) % 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7835_mod_11_l4009_400972


namespace NUMINAMATH_CALUDE_next_simultaneous_visit_l4009_400957

def visit_interval_1 : ℕ := 6
def visit_interval_2 : ℕ := 8
def visit_interval_3 : ℕ := 9

theorem next_simultaneous_visit :
  Nat.lcm (Nat.lcm visit_interval_1 visit_interval_2) visit_interval_3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_visit_l4009_400957


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l4009_400985

theorem students_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 40 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l4009_400985


namespace NUMINAMATH_CALUDE_zero_product_property_l4009_400977

theorem zero_product_property (a b : ℝ) : a * b = 0 → a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_product_property_l4009_400977


namespace NUMINAMATH_CALUDE_sequence_floor_representation_l4009_400921

theorem sequence_floor_representation 
  (a : Fin 1999 → ℕ) 
  (h : ∀ i j : Fin 1999, i + j < 1999 → 
    a i + a 1 ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1) : 
  ∃ x : ℝ, ∀ n : Fin 1999, a n = ⌊n * x⌋ := by
sorry

end NUMINAMATH_CALUDE_sequence_floor_representation_l4009_400921


namespace NUMINAMATH_CALUDE_can_add_flights_to_5000_l4009_400954

/-- A graph representing cities and flights --/
structure CityGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  edge_symmetric : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ {a}, (a, a) ∉ edges

/-- The number of cities --/
def num_cities : Nat := 998

/-- Check if the graph satisfies the flight laws --/
def satisfies_laws (g : CityGraph) : Prop :=
  (g.vertices.card = num_cities) ∧
  (∀ k : Finset Nat, k ⊆ g.vertices →
    (g.edges.filter (fun e => e.1 ∈ k ∧ e.2 ∈ k)).card ≤ 5 * k.card + 10)

/-- The theorem to be proved --/
theorem can_add_flights_to_5000 (g : CityGraph) (h : satisfies_laws g) :
  ∃ g' : CityGraph, satisfies_laws g' ∧
    g.edges ⊆ g'.edges ∧
    g'.edges.card = 5000 := by
  sorry

end NUMINAMATH_CALUDE_can_add_flights_to_5000_l4009_400954


namespace NUMINAMATH_CALUDE_largest_reciprocal_l4009_400936

theorem largest_reciprocal (a b c d e : ℚ) : 
  a = 3/7 → b = 1/2 → c = 3/4 → d = 4 → e = 100 → 
  (1/a > 1/b ∧ 1/a > 1/c ∧ 1/a > 1/d ∧ 1/a > 1/e) := by
  sorry

end NUMINAMATH_CALUDE_largest_reciprocal_l4009_400936


namespace NUMINAMATH_CALUDE_probability_123_in_10_rolls_l4009_400981

theorem probability_123_in_10_rolls (n : ℕ) (h : n = 10) :
  let total_outcomes := 6^n
  let favorable_outcomes := 8 * 6^7 - 15 * 6^4 + 4 * 6
  (favorable_outcomes : ℚ) / total_outcomes = 2220072 / 6^10 :=
by sorry

end NUMINAMATH_CALUDE_probability_123_in_10_rolls_l4009_400981


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4009_400982

theorem sum_of_three_numbers : 0.8 + (1 / 2 : ℚ) + 0.9 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4009_400982


namespace NUMINAMATH_CALUDE_two_numbers_difference_l4009_400901

theorem two_numbers_difference (x y : ℝ) : x + y = 55 ∧ x = 35 → x - y = 15 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l4009_400901


namespace NUMINAMATH_CALUDE_line_segment_slope_l4009_400911

theorem line_segment_slope (m n p : ℝ) : 
  (m = 4 * n + 5) → 
  (m + 2 = 4 * (n + p) + 5) → 
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_line_segment_slope_l4009_400911


namespace NUMINAMATH_CALUDE_xiao_ming_running_time_l4009_400944

theorem xiao_ming_running_time (track_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : track_length = 360)
  (h2 : speed1 = 5)
  (h3 : speed2 = 4) :
  let avg_speed := (speed1 + speed2) / 2
  let total_time := track_length / avg_speed
  let half_distance := track_length / 2
  let second_half_time := half_distance / speed2
  second_half_time = 44 := by
sorry

end NUMINAMATH_CALUDE_xiao_ming_running_time_l4009_400944


namespace NUMINAMATH_CALUDE_cara_right_neighbors_l4009_400963

/-- The number of Cara's friends -/
def num_friends : ℕ := 7

/-- The number of different friends who can sit immediately to Cara's right -/
def num_right_neighbors : ℕ := num_friends

theorem cara_right_neighbors :
  num_right_neighbors = num_friends :=
by sorry

end NUMINAMATH_CALUDE_cara_right_neighbors_l4009_400963


namespace NUMINAMATH_CALUDE_three_numbers_sum_6_product_4_l4009_400908

theorem three_numbers_sum_6_product_4 :
  ∀ a b c : ℕ,
  a + b + c = 6 →
  a * b * c = 4 →
  ((a = 1 ∧ b = 1 ∧ c = 4) ∨
   (a = 1 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_6_product_4_l4009_400908


namespace NUMINAMATH_CALUDE_rectangle_y_value_l4009_400993

/-- Rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 5 * r.y

theorem rectangle_y_value (r : Rectangle) (h_area : area r = 35) : r.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l4009_400993


namespace NUMINAMATH_CALUDE_parking_lot_buses_l4009_400953

/-- The total number of buses in a parking lot after more buses arrive -/
def total_buses (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 7 initial buses and 6 additional buses, the total is 13 -/
theorem parking_lot_buses : total_buses 7 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_buses_l4009_400953


namespace NUMINAMATH_CALUDE_athlete_track_arrangements_l4009_400927

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two in their numbered tracks -/
def athleteArrangements : ℕ := 20

/-- The number of ways to choose 2 items from a set of 5 -/
def choose5_2 : ℕ := 10

/-- The number of derangements of 3 objects -/
def derangement3 : ℕ := 2

theorem athlete_track_arrangements :
  athleteArrangements = choose5_2 * derangement3 :=
sorry

end NUMINAMATH_CALUDE_athlete_track_arrangements_l4009_400927


namespace NUMINAMATH_CALUDE_max_students_distribution_l4009_400930

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1001) (h2 : pencils = 910) :
  (∃ (students pen_per_student pencil_per_student : ℕ),
    students * pen_per_student = pens ∧
    students * pencil_per_student = pencils ∧
    ∀ s : ℕ, s * pen_per_student = pens → s * pencil_per_student = pencils → s ≤ students) ↔
  students = Nat.gcd pens pencils :=
sorry

end NUMINAMATH_CALUDE_max_students_distribution_l4009_400930


namespace NUMINAMATH_CALUDE_sqrt8_same_type_as_sqrt2_l4009_400903

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define a function to check if a number is of the same type of quadratic root as √2
def same_type_as_sqrt2 (n : ℕ) : Prop :=
  ¬ (is_perfect_square n) ∧ ∃ k : ℕ, n = 2 * k ∧ ¬ (is_perfect_square k)

-- Theorem statement
theorem sqrt8_same_type_as_sqrt2 :
  same_type_as_sqrt2 8 ∧
  ¬ (same_type_as_sqrt2 4) ∧
  ¬ (same_type_as_sqrt2 12) ∧
  ¬ (same_type_as_sqrt2 24) :=
sorry

end NUMINAMATH_CALUDE_sqrt8_same_type_as_sqrt2_l4009_400903


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l4009_400964

theorem five_digit_divisible_by_nine :
  ∃ (n : ℕ), 
    n < 10 ∧ 
    (35000 + n * 100 + 72) % 9 = 0 ∧
    (3 + 5 + n + 7 + 2) % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l4009_400964


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4009_400948

theorem geometric_series_sum : 
  let s := ∑' k, (3^k : ℝ) / (9^k - 1)
  s = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4009_400948


namespace NUMINAMATH_CALUDE_line_segment_can_have_specific_length_l4009_400922

/-- A line segment is a geometric object with a measurable, finite length. -/
structure LineSegment where
  length : ℝ
  length_positive : length > 0

/-- Theorem: A line segment can have a specific, finite length (e.g., 0.7 meters). -/
theorem line_segment_can_have_specific_length : ∃ (s : LineSegment), s.length = 0.7 :=
sorry

end NUMINAMATH_CALUDE_line_segment_can_have_specific_length_l4009_400922


namespace NUMINAMATH_CALUDE_lamp_position_probability_l4009_400915

/-- The probability that a randomly chosen point on a line segment of length 6
    is at least 2 units away from both endpoints is 1/3. -/
theorem lamp_position_probability : 
  let total_length : ℝ := 6
  let min_distance : ℝ := 2
  let favorable_length : ℝ := total_length - 2 * min_distance
  favorable_length / total_length = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_lamp_position_probability_l4009_400915


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4009_400951

/-- The sum of the infinite series ∑(n=0 to ∞) (2^n / 5^n) is equal to 5/3 -/
theorem geometric_series_sum : 
  let a : ℕ → ℝ := λ n => (2 : ℝ)^n
  (∑' n, a n / 5^n) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4009_400951


namespace NUMINAMATH_CALUDE_motorcyclist_meets_cyclist1_l4009_400978

/-- Represents the time in minutes for two entities to meet or overtake each other. -/
structure MeetingTime where
  time : ℝ
  time_positive : time > 0

/-- Represents an entity moving on the circular highway. -/
structure Entity where
  speed : ℝ
  direction : Bool  -- True for one direction, False for the opposite

/-- The circular highway setup with four entities. -/
structure CircularHighway where
  runner : Entity
  cyclist1 : Entity
  cyclist2 : Entity
  motorcyclist : Entity
  runner_cyclist2_meeting : MeetingTime
  runner_cyclist1_overtake : MeetingTime
  motorcyclist_cyclist2_overtake : MeetingTime
  highway_length : ℝ
  highway_length_positive : highway_length > 0

  runner_direction : runner.direction = true
  cyclist1_direction : cyclist1.direction = true
  cyclist2_direction : cyclist2.direction = false
  motorcyclist_direction : motorcyclist.direction = false

  runner_cyclist2_meeting_time : runner_cyclist2_meeting.time = 12
  runner_cyclist1_overtake_time : runner_cyclist1_overtake.time = 20
  motorcyclist_cyclist2_overtake_time : motorcyclist_cyclist2_overtake.time = 5

/-- The theorem stating that the motorcyclist meets the first cyclist every 3 minutes. -/
theorem motorcyclist_meets_cyclist1 (h : CircularHighway) :
  ∃ (t : MeetingTime), t.time = 3 ∧
    h.highway_length / t.time = h.motorcyclist.speed + h.cyclist1.speed :=
sorry

end NUMINAMATH_CALUDE_motorcyclist_meets_cyclist1_l4009_400978


namespace NUMINAMATH_CALUDE_pen_discount_theorem_l4009_400906

theorem pen_discount_theorem (marked_price : ℝ) :
  let purchase_quantity : ℕ := 60
  let purchase_price_in_pens : ℕ := 46
  let profit_percent : ℝ := 29.130434782608695

  let cost_price : ℝ := marked_price * purchase_price_in_pens
  let selling_price : ℝ := cost_price * (1 + profit_percent / 100)
  let selling_price_per_pen : ℝ := selling_price / purchase_quantity
  let discount : ℝ := marked_price - selling_price_per_pen
  let discount_percent : ℝ := (discount / marked_price) * 100

  discount_percent = 1 := by sorry

end NUMINAMATH_CALUDE_pen_discount_theorem_l4009_400906


namespace NUMINAMATH_CALUDE_pencil_price_l4009_400916

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pen_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 510 →
  pen_price = 12 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_price_l4009_400916


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4009_400923

def A : Set ℤ := {1, 3}

def B : Set ℤ := {x | 0 < Real.log (x + 1) ∧ Real.log (x + 1) < 1/2}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4009_400923


namespace NUMINAMATH_CALUDE_least_n_divisibility_l4009_400974

theorem least_n_divisibility : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k = 0) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ k ≤ n + 1 ∧ (n - 1)^2 % k ≠ 0) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → 
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k = 0) ∨
    (∀ (k : ℕ), k ≥ 1 ∧ k ≤ m + 1 → (m - 1)^2 % k ≠ 0)) ∧
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_least_n_divisibility_l4009_400974


namespace NUMINAMATH_CALUDE_remaining_average_l4009_400940

theorem remaining_average (n : ℕ) (total_avg : ℚ) (partial_avg : ℚ) :
  n = 10 →
  total_avg = 80 →
  partial_avg = 58 →
  ∃ (m : ℕ), m = 6 ∧
    (n * total_avg - m * partial_avg) / (n - m) = 113 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l4009_400940


namespace NUMINAMATH_CALUDE_find_b_value_l4009_400995

theorem find_b_value (x b : ℝ) (h1 : 5 * x + 3 = b * x - 22) (h2 : x = 5) : b = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l4009_400995


namespace NUMINAMATH_CALUDE_oliver_used_30_tickets_l4009_400955

/-- The number of tickets Oliver used at the town carnival -/
def olivers_tickets (ferris_wheel_rides bumper_car_rides tickets_per_ride : ℕ) : ℕ :=
  (ferris_wheel_rides + bumper_car_rides) * tickets_per_ride

/-- Theorem: Oliver used 30 tickets at the town carnival -/
theorem oliver_used_30_tickets :
  olivers_tickets 7 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oliver_used_30_tickets_l4009_400955


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4009_400939

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149)
  (sum_of_products : a*b + b*c + c*a = 70) :
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4009_400939


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l4009_400917

/-- Given 4 non-negative integers representing ages, if their mean is 8 and
    their median is 5, then the sum of the smallest and largest of these
    integers is 22. -/
theorem cousins_ages_sum (a b c d : ℕ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d →  -- Sorted in ascending order
  (a + b + c + d) / 4 = 8 →  -- Mean is 8
  (b + c) / 2 = 5 →  -- Median is 5
  a + d = 22 := by sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l4009_400917


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l4009_400937

/-- The function f(x) = ln x - 3/x has a zero point in the interval (2, 3) -/
theorem zero_point_in_interval (f : ℝ → ℝ) :
  (∀ x > 0, f x = Real.log x - 3 / x) →
  (∀ x > 0, StrictMono f) →
  ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l4009_400937


namespace NUMINAMATH_CALUDE_range_of_m_l4009_400914

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m : 
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) → 
  (∀ m : ℝ, (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) ↔ (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4009_400914


namespace NUMINAMATH_CALUDE_shortest_paths_count_l4009_400931

/-- The number of shortest paths on a chess board from (0,0) to (m,n) -/
def numShortestPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- Theorem: The number of shortest paths from (0,0) to (m,n) on a chess board,
    where movement is restricted to coordinate axis directions and
    direction changes only at integer coordinates, is equal to (m+n choose m) -/
theorem shortest_paths_count (m n : ℕ) :
  numShortestPaths m n = Nat.choose (m + n) m := by
  sorry

end NUMINAMATH_CALUDE_shortest_paths_count_l4009_400931


namespace NUMINAMATH_CALUDE_cows_bought_is_two_l4009_400913

/-- The number of cows bought given the total cost, number of goats, and average prices. -/
def number_of_cows (total_cost goats goat_price cow_price : ℕ) : ℕ :=
  (total_cost - goats * goat_price) / cow_price

/-- Theorem stating that the number of cows bought is 2 under the given conditions. -/
theorem cows_bought_is_two :
  number_of_cows 1500 10 70 400 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cows_bought_is_two_l4009_400913


namespace NUMINAMATH_CALUDE_project_completion_equation_l4009_400925

theorem project_completion_equation (x : ℝ) : 
  (x > 17) →  -- Ensure x - 17 is positive
  (1 / (x + 15) + 1 / (x + 36) = 1 / (x - 17)) := by
sorry

end NUMINAMATH_CALUDE_project_completion_equation_l4009_400925


namespace NUMINAMATH_CALUDE_average_problem_l4009_400969

theorem average_problem (x : ℝ) : 
  (15 + 25 + 35 + x) / 4 = 30 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l4009_400969


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l4009_400929

theorem no_linear_term_condition (a : ℝ) : 
  (∀ x : ℝ, ∃ b c : ℝ, (x + a) * (x - 1/2) = x^2 + b + c * x) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l4009_400929


namespace NUMINAMATH_CALUDE_ratio_of_squares_l4009_400909

theorem ratio_of_squares (x y : ℝ) (h : x^2 = 8*y^2 - 224) :
  x/y = Real.sqrt (8 - 224/y^2) :=
by sorry

end NUMINAMATH_CALUDE_ratio_of_squares_l4009_400909


namespace NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l4009_400994

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

-- Define the property of having two zeros
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Theorem statement
theorem two_zeros_iff_a_positive :
  ∀ a : ℝ, has_two_zeros (f a) ↔ a > 0 :=
sorry

end NUMINAMATH_CALUDE_two_zeros_iff_a_positive_l4009_400994


namespace NUMINAMATH_CALUDE_crayons_per_pack_l4009_400996

/-- Given that Nancy bought a total of 615 crayons in 41 packs,
    prove that there were 15 crayons in each pack. -/
theorem crayons_per_pack :
  ∀ (total_crayons : ℕ) (num_packs : ℕ),
    total_crayons = 615 →
    num_packs = 41 →
    total_crayons / num_packs = 15 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_pack_l4009_400996


namespace NUMINAMATH_CALUDE_tan_theta_value_l4009_400943

theorem tan_theta_value (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l4009_400943


namespace NUMINAMATH_CALUDE_sam_cleaner_meet_twice_l4009_400904

/-- Represents the movement of Sam and the street cleaner on a path with benches --/
structure PathMovement where
  sam_speed : ℝ
  cleaner_speed : ℝ
  bench_distance : ℝ
  cleaner_stop_time : ℝ

/-- Calculates the number of times Sam and the cleaner meet --/
def number_of_meetings (movement : PathMovement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : PathMovement :=
  { sam_speed := 3
  , cleaner_speed := 9
  , bench_distance := 300
  , cleaner_stop_time := 40 }

/-- Theorem stating that Sam and the cleaner meet exactly twice --/
theorem sam_cleaner_meet_twice :
  number_of_meetings problem_scenario = 2 := by
  sorry

end NUMINAMATH_CALUDE_sam_cleaner_meet_twice_l4009_400904


namespace NUMINAMATH_CALUDE_quadratic_roots_l4009_400975

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (k + 2) * x + 2 * k

theorem quadratic_roots (k : ℝ) :
  (quadratic k 1 = 0 → k = 1 ∧ quadratic k 2 = 0) ∧
  (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4009_400975


namespace NUMINAMATH_CALUDE_select_team_count_l4009_400928

/-- The number of ways to select a team of 8 members (4 boys and 4 girls) from a group of 10 boys and 12 girls -/
def selectTeam (totalBoys : ℕ) (totalGirls : ℕ) (teamSize : ℕ) (boysInTeam : ℕ) (girlsInTeam : ℕ) : ℕ :=
  Nat.choose totalBoys boysInTeam * Nat.choose totalGirls girlsInTeam

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem select_team_count :
  selectTeam 10 12 8 4 4 = 103950 := by
  sorry

end NUMINAMATH_CALUDE_select_team_count_l4009_400928


namespace NUMINAMATH_CALUDE_money_sharing_l4009_400980

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda * 5 = ben * 3 →
  carlos * 5 = ben * 12 →
  ben = 25 →
  total = 100 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l4009_400980


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l4009_400992

theorem algebraic_expression_equality (a b : ℝ) (h : 5 * a + 3 * b = -4) :
  2 * (a + b) + 4 * (2 * a + b) - 10 = -18 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l4009_400992


namespace NUMINAMATH_CALUDE_distance_A_to_B_l4009_400924

/-- Prove that the distance from A to B is 510 km given the travel conditions -/
theorem distance_A_to_B : 
  ∀ (d_AB : ℝ) (d_AC : ℝ) (t_E t_F : ℝ) (speed_ratio : ℝ),
  d_AC = 300 →
  t_E = 3 →
  t_F = 4 →
  speed_ratio = 2.2666666666666666 →
  (d_AB / t_E) / (d_AC / t_F) = speed_ratio →
  d_AB = 510 := by
sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l4009_400924


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l4009_400945

theorem rectangular_plot_breadth (length breadth area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 972 →
  breadth = 18 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l4009_400945


namespace NUMINAMATH_CALUDE_triangle_properties_l4009_400920

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.b - t.c) / t.a = Real.cos t.C / Real.cos t.A) 
  (h2 : t.a = Real.sqrt 5) 
  (h3 : (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2) :
  t.A = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 5 + Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l4009_400920


namespace NUMINAMATH_CALUDE_barrel_capacity_l4009_400949

theorem barrel_capacity (original_amount : ℝ) (capacity : ℝ) : 
  (original_amount = 3 / 5 * capacity) →
  (original_amount - 18 = 0.6 * original_amount) →
  (capacity = 75) :=
by sorry

end NUMINAMATH_CALUDE_barrel_capacity_l4009_400949


namespace NUMINAMATH_CALUDE_abs_geq_ax_implies_abs_a_leq_one_l4009_400910

theorem abs_geq_ax_implies_abs_a_leq_one (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_geq_ax_implies_abs_a_leq_one_l4009_400910


namespace NUMINAMATH_CALUDE_normal_distribution_standard_deviations_l4009_400961

/-- Proves that for a normal distribution with mean 14.0 and standard deviation 1.5,
    the value 11 is exactly 2 standard deviations less than the mean. -/
theorem normal_distribution_standard_deviations (μ σ x : ℝ) 
  (h_mean : μ = 14.0)
  (h_std_dev : σ = 1.5)
  (h_value : x = 11.0) :
  (μ - x) / σ = 2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_standard_deviations_l4009_400961


namespace NUMINAMATH_CALUDE_inequality_not_always_holds_l4009_400960

theorem inequality_not_always_holds (a b : ℝ) (h : a > b) :
  ¬ ∀ c : ℝ, a * c > b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_holds_l4009_400960


namespace NUMINAMATH_CALUDE_fraction_chain_l4009_400947

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by sorry

end NUMINAMATH_CALUDE_fraction_chain_l4009_400947


namespace NUMINAMATH_CALUDE_polygon_diagonals_l4009_400984

theorem polygon_diagonals (n : ℕ) (d : ℕ) : n = 17 ∧ d = 104 →
  (n - 1) * (n - 4) / 2 = d := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l4009_400984


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l4009_400967

/-- Given two lines in the xy-plane, this function checks if they are symmetric with respect to the y-axis -/
def symmetric_about_y_axis (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, line1 x y ↔ line2 (-x) y

/-- The original line: 3x - 4y + 5 = 0 -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The symmetric line: 3x + 4y + 22 = 0 -/
def symmetric_line (x y : ℝ) : Prop := 3 * x + 4 * y + 22 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to the y-axis -/
theorem symmetry_about_y_axis : symmetric_about_y_axis original_line symmetric_line := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l4009_400967


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l4009_400973

/-- The inverse relationship between y^5 and z^(1/5) -/
def inverse_relation (y z : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ y^5 * z^(1/5) = k

theorem inverse_variation_problem (y₁ y₂ z₁ z₂ : ℝ) 
  (h1 : inverse_relation y₁ z₁)
  (h2 : inverse_relation y₂ z₂)
  (h3 : y₁ = 3)
  (h4 : z₁ = 8)
  (h5 : y₂ = 6) :
  z₂ = 1 / 1048576 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l4009_400973


namespace NUMINAMATH_CALUDE_cubic_sum_implies_square_sum_less_than_one_l4009_400965

theorem cubic_sum_implies_square_sum_less_than_one
  (x y : ℝ)
  (x_pos : 0 < x)
  (y_pos : 0 < y)
  (h : x^3 + y^3 = x - y) :
  x^2 + y^2 < 1 :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_implies_square_sum_less_than_one_l4009_400965


namespace NUMINAMATH_CALUDE_shara_shell_count_l4009_400968

/-- Calculates the total number of shells Shara has after her vacation. -/
def total_shells (initial_shells : ℕ) (shells_per_day : ℕ) (days : ℕ) (fourth_day_shells : ℕ) : ℕ :=
  initial_shells + shells_per_day * days + fourth_day_shells

/-- Theorem stating that Shara has 41 shells after her vacation. -/
theorem shara_shell_count : 
  total_shells 20 5 3 6 = 41 := by
  sorry

end NUMINAMATH_CALUDE_shara_shell_count_l4009_400968


namespace NUMINAMATH_CALUDE_extreme_value_condition_l4009_400959

-- Define the function f(x)
def f (m n : ℝ) (x : ℝ) : ℝ := x^3 + 3*m*x^2 + n*x + m^2

-- Define the derivative of f(x)
def f_prime (m n : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*m*x + n

-- Theorem statement
theorem extreme_value_condition (m n : ℝ) :
  f m n (-1) = 0 ∧ f_prime m n (-1) = 0 → m + n = 11 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l4009_400959


namespace NUMINAMATH_CALUDE_math_competition_problem_l4009_400933

theorem math_competition_problem (n : ℕ) (S : Finset (Finset (Fin 6))) :
  (∀ (i j : Fin 6), i ≠ j → (S.filter (λ s => i ∈ s ∧ j ∈ s)).card > (2 * S.card) / 5) →
  (∀ s ∈ S, s.card ≤ 5) →
  (S.filter (λ s => s.card = 5)).card ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_math_competition_problem_l4009_400933


namespace NUMINAMATH_CALUDE_james_vote_percentage_l4009_400966

theorem james_vote_percentage (total_votes : ℕ) (john_votes : ℕ) (third_candidate_extra : ℕ) :
  total_votes = 1150 →
  john_votes = 150 →
  third_candidate_extra = 150 →
  let third_candidate_votes := john_votes + third_candidate_extra
  let remaining_votes := total_votes - john_votes
  let james_votes := total_votes - (john_votes + third_candidate_votes)
  james_votes / remaining_votes = 7 / 10 := by
  sorry

#check james_vote_percentage

end NUMINAMATH_CALUDE_james_vote_percentage_l4009_400966


namespace NUMINAMATH_CALUDE_max_product_l4009_400979

def digits : List Nat := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 5 1 9 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l4009_400979


namespace NUMINAMATH_CALUDE_num_paths_A_to_B_l4009_400962

/-- Represents the number of red arrows from Point A -/
def num_red_arrows : ℕ := 3

/-- Represents the number of blue arrows connected to each red arrow -/
def blue_per_red : ℕ := 2

/-- Represents the number of green arrows connected to each blue arrow -/
def green_per_blue : ℕ := 2

/-- Represents the number of orange arrows connected to each green arrow -/
def orange_per_green : ℕ := 1

/-- Represents the number of ways to reach each blue arrow from a red arrow -/
def ways_to_blue : ℕ := 3

/-- Represents the number of ways to reach each green arrow from a blue arrow -/
def ways_to_green : ℕ := 4

/-- Represents the number of ways to reach each orange arrow from a green arrow -/
def ways_to_orange : ℕ := 5

/-- Theorem stating that the number of paths from A to B is 1440 -/
theorem num_paths_A_to_B : 
  num_red_arrows * blue_per_red * green_per_blue * orange_per_green * 
  ways_to_blue * ways_to_green * ways_to_orange = 1440 := by
  sorry

#check num_paths_A_to_B

end NUMINAMATH_CALUDE_num_paths_A_to_B_l4009_400962


namespace NUMINAMATH_CALUDE_positive_real_array_inequalities_l4009_400983

theorem positive_real_array_inequalities
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0)
  (h1 : (x₁^2 - x₃*x₅)*(x₂^2 - x₃*x₅) ≤ 0)
  (h2 : (x₂^2 - x₄*x₁)*(x₃^2 - x₄*x₁) ≤ 0)
  (h3 : (x₃^2 - x₅*x₂)*(x₄^2 - x₅*x₂) ≤ 0)
  (h4 : (x₄^2 - x₁*x₃)*(x₅^2 - x₁*x₃) ≤ 0)
  (h5 : (x₅^2 - x₂*x₄)*(x₁^2 - x₂*x₄) ≤ 0) :
  x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅ := by
sorry

end NUMINAMATH_CALUDE_positive_real_array_inequalities_l4009_400983


namespace NUMINAMATH_CALUDE_solution_set_of_increasing_function_l4009_400918

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the solution set of |f(x+1)| < 1 is (-1, 2). -/
theorem solution_set_of_increasing_function (f : ℝ → ℝ) 
  (h_incr : StrictMono f) (h_f0 : f 0 = -1) (h_f3 : f 3 = 1) :
  {x : ℝ | |f (x + 1)| < 1} = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_increasing_function_l4009_400918


namespace NUMINAMATH_CALUDE_rainy_days_exist_l4009_400905

/-- Represents the number of rainy days given the conditions of Mo's drinking habits -/
def rainy_days (n d T H P : ℤ) : Prop :=
  ∃ (R : ℤ),
    (1 ≤ d) ∧ (d ≤ 31) ∧
    (T = 3 * (d - R)) ∧
    (H = n * R) ∧
    (T = H + P) ∧
    (R = (3 * d - P) / (n + 3)) ∧
    (0 ≤ R) ∧ (R ≤ d)

/-- Theorem stating the existence of R satisfying the conditions -/
theorem rainy_days_exist (n d T H P : ℤ) (h1 : 1 ≤ d) (h2 : d ≤ 31) 
  (h3 : T = 3 * (d - (3 * d - P) / (n + 3))) 
  (h4 : H = n * ((3 * d - P) / (n + 3))) 
  (h5 : T = H + P)
  (h6 : (3 * d - P) % (n + 3) = 0)
  (h7 : 0 ≤ (3 * d - P) / (n + 3))
  (h8 : (3 * d - P) / (n + 3) ≤ d) :
  rainy_days n d T H P :=
by
  sorry


end NUMINAMATH_CALUDE_rainy_days_exist_l4009_400905


namespace NUMINAMATH_CALUDE_tv_sale_value_change_l4009_400988

theorem tv_sale_value_change 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 10) 
  (h2 : sales_increase_percent = 85) : 
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_quantity := original_quantity * (1 + sales_increase_percent / 100)
  let original_value := original_price * original_quantity
  let new_value := new_price * new_quantity
  (new_value - original_value) / original_value * 100 = 66.5 := by
sorry

end NUMINAMATH_CALUDE_tv_sale_value_change_l4009_400988


namespace NUMINAMATH_CALUDE_limit_implies_a_range_l4009_400935

/-- If the limit of 3^n / (3^(n+1) + (a+1)^n) as n approaches infinity is 1/3, 
    then a is in the open interval (-4, 2) -/
theorem limit_implies_a_range (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |3^n / (3^(n+1) + (a+1)^n) - 1/3| < ε) →
  a ∈ Set.Ioo (-4 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_limit_implies_a_range_l4009_400935


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_one_l4009_400956

theorem sqrt_expression_equals_one :
  (Real.sqrt 24 - Real.sqrt 216) / Real.sqrt 6 + 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_one_l4009_400956


namespace NUMINAMATH_CALUDE_race_distance_l4009_400912

theorem race_distance (a_finish_time : ℝ) (time_diff : ℝ) (distance_diff : ℝ) :
  a_finish_time = 3 →
  time_diff = 7 →
  distance_diff = 56 →
  ∃ (total_distance : ℝ),
    total_distance = 136 ∧
    (total_distance / a_finish_time) * time_diff = distance_diff :=
by sorry

end NUMINAMATH_CALUDE_race_distance_l4009_400912


namespace NUMINAMATH_CALUDE_average_donation_l4009_400907

def donations : List ℝ := [10, 12, 13.5, 40.8, 19.3, 20.8, 25, 16, 30, 30]

theorem average_donation : (donations.sum / donations.length) = 21.74 := by
  sorry

end NUMINAMATH_CALUDE_average_donation_l4009_400907


namespace NUMINAMATH_CALUDE_proportional_segments_l4009_400900

theorem proportional_segments (a b c d : ℝ) : 
  a = 3 ∧ d = 4 ∧ c = 6 ∧ (a / b = c / d) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l4009_400900


namespace NUMINAMATH_CALUDE_complement_of_A_l4009_400987

def A : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}

theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_l4009_400987


namespace NUMINAMATH_CALUDE_asymptotes_of_specific_hyperbola_l4009_400946

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The equation of a hyperbola in standard form -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y = h.b / h.a * x ∨ y = -h.b / h.a * x

/-- The specific hyperbola we're interested in -/
def specific_hyperbola : Hyperbola where
  a := 1
  b := 2
  h_pos := by simp

theorem asymptotes_of_specific_hyperbola :
  ∀ x y : ℝ, asymptote_equation specific_hyperbola x y ↔ (y = 2*x ∨ y = -2*x) :=
sorry

end NUMINAMATH_CALUDE_asymptotes_of_specific_hyperbola_l4009_400946


namespace NUMINAMATH_CALUDE_distinct_subsets_removal_l4009_400971

theorem distinct_subsets_removal (n : ℕ) (X : Finset ℕ) (A : Fin n → Finset ℕ) 
  (h1 : n ≥ 2) 
  (h2 : X.card = n) 
  (h3 : ∀ i : Fin n, A i ⊆ X) 
  (h4 : ∀ i j : Fin n, i ≠ j → A i ≠ A j) :
  ∃ x ∈ X, ∀ i j : Fin n, i ≠ j → A i \ {x} ≠ A j \ {x} := by
  sorry

end NUMINAMATH_CALUDE_distinct_subsets_removal_l4009_400971


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l4009_400986

theorem triangle_max_perimeter (A B C : ℝ) (a b c : ℝ) :
  A = 2 * π / 3 →
  a = 3 →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = 2 * Real.sin (A / 2) * Real.sin (B / 2) / Real.sin ((A + B) / 2) →
  b = 2 * Real.sin (B / 2) * Real.sin (C / 2) / Real.sin ((B + C) / 2) →
  c = 2 * Real.sin (C / 2) * Real.sin (A / 2) / Real.sin ((C + A) / 2) →
  (∀ B' C' a' b' c',
    A + B' + C' = π →
    a' > 0 ∧ b' > 0 ∧ c' > 0 →
    a' = 2 * Real.sin (A / 2) * Real.sin (B' / 2) / Real.sin ((A + B') / 2) →
    b' = 2 * Real.sin (B' / 2) * Real.sin (C' / 2) / Real.sin ((B' + C') / 2) →
    c' = 2 * Real.sin (C' / 2) * Real.sin (A / 2) / Real.sin ((C' + A) / 2) →
    a' + b' + c' ≤ a + b + c) →
  a + b + c = 3 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l4009_400986


namespace NUMINAMATH_CALUDE_david_twice_rosy_age_l4009_400958

/-- The number of years it will take for David to be twice as old as Rosy -/
def years_until_twice_age : ℕ :=
  sorry

/-- David's current age -/
def david_age : ℕ :=
  sorry

/-- Rosy's current age -/
def rosy_age : ℕ :=
  12

theorem david_twice_rosy_age :
  (david_age = rosy_age + 18) →
  (david_age + years_until_twice_age = 2 * (rosy_age + years_until_twice_age)) →
  years_until_twice_age = 6 :=
by sorry

end NUMINAMATH_CALUDE_david_twice_rosy_age_l4009_400958


namespace NUMINAMATH_CALUDE_correct_match_probability_l4009_400989

/-- The number of celebrities and baby pictures -/
def n : ℕ := 4

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := n.factorial

/-- The number of correct arrangements -/
def correct_arrangements : ℕ := 1

/-- The probability of correctly matching all celebrities to their baby pictures -/
def probability : ℚ := correct_arrangements / total_arrangements

theorem correct_match_probability :
  probability = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_correct_match_probability_l4009_400989


namespace NUMINAMATH_CALUDE_officer_selection_count_l4009_400976

/-- The number of ways to choose officers from a club -/
def choose_officers (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else (n - k + 1).factorial / (n - k).factorial

/-- Theorem: Choosing 5 officers from 15 members results in 360,360 possibilities -/
theorem officer_selection_count :
  choose_officers 15 5 = 360360 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_count_l4009_400976


namespace NUMINAMATH_CALUDE_zero_point_location_l4009_400932

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 4*x + a

-- State the theorem
theorem zero_point_location (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (0 < a) → (a < 2) →
  (f a x₁ = 0) → (f a x₂ = 0) → (f a x₃ = 0) →
  (x₁ < x₂) → (x₂ < x₃) →
  (0 < x₂) ∧ (x₂ < 1) := by
  sorry

end NUMINAMATH_CALUDE_zero_point_location_l4009_400932


namespace NUMINAMATH_CALUDE_finite_triples_sum_reciprocals_l4009_400999

theorem finite_triples_sum_reciprocals :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c = (1 : ℚ) / 1000 →
    (a, b, c) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_finite_triples_sum_reciprocals_l4009_400999


namespace NUMINAMATH_CALUDE_jane_rounds_played_l4009_400970

-- Define the parameters of the game
def points_per_round : ℕ := 10
def final_points : ℕ := 60
def lost_points : ℕ := 20

-- Define the theorem
theorem jane_rounds_played :
  (final_points + lost_points) / points_per_round = 8 :=
by sorry

end NUMINAMATH_CALUDE_jane_rounds_played_l4009_400970


namespace NUMINAMATH_CALUDE_multiplicative_inverse_137_mod_391_l4009_400991

theorem multiplicative_inverse_137_mod_391 :
  ∃ x : ℕ, x < 391 ∧ (137 * x) % 391 = 1 ∧ x = 294 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_137_mod_391_l4009_400991


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l4009_400919

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

/-- Applies similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem -/
theorem point_not_on_transformed_plane :
  let originalPlane : Plane := { a := 1, b := -1, c := -1, d := -1 }
  let k : ℝ := 4
  let transformedPlane := transformPlane originalPlane k
  let point : Point3D := { x := 7, y := 0, z := -1 }
  ¬(pointOnPlane point transformedPlane) := by
  sorry


end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l4009_400919


namespace NUMINAMATH_CALUDE_henry_total_score_l4009_400990

def geography_score : ℝ := 50
def math_score : ℝ := 70
def english_score : ℝ := 66
def science_score : ℝ := 84
def french_score : ℝ := 75

def geography_weight : ℝ := 0.25
def math_weight : ℝ := 0.20
def english_weight : ℝ := 0.20
def science_weight : ℝ := 0.15
def french_weight : ℝ := 0.10

def history_score : ℝ :=
  geography_score * geography_weight +
  math_score * math_weight +
  english_score * english_weight +
  science_score * science_weight +
  french_score * french_weight

def total_score : ℝ :=
  geography_score + math_score + english_score + science_score + french_score + history_score

theorem henry_total_score :
  total_score = 404.8 := by
  sorry

end NUMINAMATH_CALUDE_henry_total_score_l4009_400990


namespace NUMINAMATH_CALUDE_line_perpendicular_parallel_implies_planes_perpendicular_l4009_400997

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_parallel_implies_planes_perpendicular
  (m : Line) (α β : Plane) :
  perpendicular m β → parallel m α → perpendicularPlanes α β := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_parallel_implies_planes_perpendicular_l4009_400997


namespace NUMINAMATH_CALUDE_magical_coin_expected_winnings_l4009_400952

/-- Represents the outcomes of the magical coin flip -/
inductive Outcome
  | Heads
  | Tails
  | Edge
  | Disappear

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 3/8
  | Outcome.Tails => 1/4
  | Outcome.Edge => 1/8
  | Outcome.Disappear => 1/4

/-- The winnings (or losses) for each outcome -/
def winnings (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 2
  | Outcome.Tails => 5
  | Outcome.Edge => -2
  | Outcome.Disappear => -6

/-- The expected winnings of flipping the magical coin -/
def expected_winnings : ℚ :=
  (probability Outcome.Heads * winnings Outcome.Heads) +
  (probability Outcome.Tails * winnings Outcome.Tails) +
  (probability Outcome.Edge * winnings Outcome.Edge) +
  (probability Outcome.Disappear * winnings Outcome.Disappear)

theorem magical_coin_expected_winnings :
  expected_winnings = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_magical_coin_expected_winnings_l4009_400952


namespace NUMINAMATH_CALUDE_roots_sum_squares_and_product_l4009_400938

theorem roots_sum_squares_and_product (α β : ℝ) : 
  (2 * α^2 - α - 4 = 0) → (2 * β^2 - β - 4 = 0) → α^2 + α*β + β^2 = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_squares_and_product_l4009_400938


namespace NUMINAMATH_CALUDE_prob_sum_three_dice_l4009_400902

/-- The probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The desired sum on the top faces -/
def desired_sum : ℕ := 3

/-- The probability of rolling the desired sum on all dice -/
def prob_desired_sum : ℚ := single_die_prob ^ num_dice

theorem prob_sum_three_dice (h : desired_sum = 3 ∧ num_dice = 3) :
  prob_desired_sum = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_dice_l4009_400902


namespace NUMINAMATH_CALUDE_jimmy_stair_climb_time_l4009_400998

def stairClimbTime (n : ℕ) : ℕ :=
  let baseTime := 25
  let increment := 7
  let flightTimes := List.range n |>.map (λ i => baseTime + i * increment)
  let totalFlightTime := flightTimes.sum
  let stopTime := (n - 1) / 2 * 10
  totalFlightTime + stopTime

theorem jimmy_stair_climb_time :
  stairClimbTime 7 = 342 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climb_time_l4009_400998


namespace NUMINAMATH_CALUDE_ball_box_problem_l4009_400934

theorem ball_box_problem (num_balls : ℕ) (X : ℕ) (h1 : num_balls = 25) 
  (h2 : num_balls - 20 = X - num_balls) : X = 30 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_problem_l4009_400934


namespace NUMINAMATH_CALUDE_chocolate_price_after_discount_l4009_400942

/-- The final price of a chocolate after discount -/
def final_price (original_cost discount : ℚ) : ℚ :=
  original_cost - discount

/-- Theorem: The final price of a chocolate with original cost $2 and discount $0.57 is $1.43 -/
theorem chocolate_price_after_discount :
  final_price 2 0.57 = 1.43 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_price_after_discount_l4009_400942
