import Mathlib

namespace NUMINAMATH_CALUDE_point_on_x_axis_l2147_214721

theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  (P.2 = 0) → P = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2147_214721


namespace NUMINAMATH_CALUDE_binary_1010_eq_10_l2147_214713

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₍₂₎ -/
def binary_1010 : List Bool := [false, true, false, true]

theorem binary_1010_eq_10 : binary_to_decimal binary_1010 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_eq_10_l2147_214713


namespace NUMINAMATH_CALUDE_complement_of_A_l2147_214780

def U : Set ℕ := {x | 0 < x ∧ x < 8}
def A : Set ℕ := {2, 4, 5}

theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2147_214780


namespace NUMINAMATH_CALUDE_first_price_increase_l2147_214752

theorem first_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.15 = 1.38 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_price_increase_l2147_214752


namespace NUMINAMATH_CALUDE_divisibility_of_power_difference_l2147_214779

theorem divisibility_of_power_difference (a b : ℕ) (h : a + b = 61) :
  (61 : ℤ) ∣ (a^100 - b^100) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_power_difference_l2147_214779


namespace NUMINAMATH_CALUDE_f_monotonicity_and_negativity_l2147_214795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

theorem f_monotonicity_and_negativity (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y → f a x < f a y) ∨
  (∃ c, c > 0 ∧ 
    (∀ x y, 0 < x ∧ x < y ∧ y < c → f a x < f a y) ∧
    (∀ x y, c < x ∧ x < y → f a y < f a x)) ∧
  (∀ x, x > 0 → f a x < 0) ↔ a > (Real.exp 1)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_negativity_l2147_214795


namespace NUMINAMATH_CALUDE_transport_probabilities_theorem_l2147_214724

structure TransportProbabilities where
  plane : ℝ
  ship : ℝ
  train : ℝ
  car : ℝ
  sum_to_one : plane + ship + train + car = 1
  all_nonnegative : plane ≥ 0 ∧ ship ≥ 0 ∧ train ≥ 0 ∧ car ≥ 0

def prob_train_or_plane (p : TransportProbabilities) : ℝ :=
  p.train + p.plane

def prob_not_ship (p : TransportProbabilities) : ℝ :=
  1 - p.ship

theorem transport_probabilities_theorem (p : TransportProbabilities)
    (h1 : p.plane = 0.2)
    (h2 : p.ship = 0.3)
    (h3 : p.train = 0.4)
    (h4 : p.car = 0.1) :
    prob_train_or_plane p = 0.6 ∧ prob_not_ship p = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_transport_probabilities_theorem_l2147_214724


namespace NUMINAMATH_CALUDE_connie_marbles_l2147_214770

/-- Given that Juan has 25 more marbles than Connie and Juan has 64 marbles, 
    prove that Connie has 39 marbles. -/
theorem connie_marbles (connie juan : ℕ) 
  (h1 : juan = connie + 25) 
  (h2 : juan = 64) : 
  connie = 39 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l2147_214770


namespace NUMINAMATH_CALUDE_sarah_friends_count_l2147_214715

/-- The number of friends Sarah brought into the bedroom -/
def friends_with_sarah (total_people bedroom_people living_room_people : ℕ) : ℕ :=
  total_people - (bedroom_people + living_room_people)

theorem sarah_friends_count :
  ∀ (total_people bedroom_people living_room_people : ℕ),
  total_people = 15 →
  bedroom_people = 3 →
  living_room_people = 8 →
  friends_with_sarah total_people bedroom_people living_room_people = 4 := by
sorry

end NUMINAMATH_CALUDE_sarah_friends_count_l2147_214715


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l2147_214785

/-- The number of different Spiderman comic books --/
def spiderman_comics : ℕ := 8

/-- The number of different Archie comic books --/
def archie_comics : ℕ := 6

/-- The number of different Garfield comic books --/
def garfield_comics : ℕ := 7

/-- The number of ways to arrange the comic books --/
def arrange_comics : ℕ := spiderman_comics.factorial * (archie_comics - 1).factorial * garfield_comics.factorial * 2

theorem comic_arrangement_count :
  arrange_comics = 4864460800 :=
by sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l2147_214785


namespace NUMINAMATH_CALUDE_machine_work_time_l2147_214788

/-- The number of shirts made by the machine today -/
def shirts_today : ℕ := 8

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 2

/-- The number of minutes the machine worked today -/
def minutes_worked : ℕ := shirts_today / shirts_per_minute

theorem machine_work_time : minutes_worked = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_l2147_214788


namespace NUMINAMATH_CALUDE_marys_green_beans_weight_l2147_214774

/-- Proves that the weight of green beans is 4 pounds given the conditions of Mary's grocery shopping. -/
theorem marys_green_beans_weight (bag_capacity : ℝ) (milk_weight : ℝ) (remaining_space : ℝ) :
  bag_capacity = 20 →
  milk_weight = 6 →
  remaining_space = 2 →
  ∃ (green_beans_weight : ℝ),
    green_beans_weight + milk_weight + 2 * green_beans_weight = bag_capacity - remaining_space ∧
    green_beans_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_marys_green_beans_weight_l2147_214774


namespace NUMINAMATH_CALUDE_luke_game_points_l2147_214718

theorem luke_game_points (points_per_round : ℕ) (num_rounds : ℕ) (total_points : ℕ) : 
  points_per_round = 327 → num_rounds = 193 → total_points = points_per_round * num_rounds → total_points = 63111 := by
  sorry

end NUMINAMATH_CALUDE_luke_game_points_l2147_214718


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l2147_214776

theorem fraction_less_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l2147_214776


namespace NUMINAMATH_CALUDE_merger_proportion_l2147_214797

/-- Represents the proportion of managers in a company -/
def ManagerProportion := Fin 101 → ℚ

/-- Represents the proportion of employees from one company in a merged company -/
def MergedProportion := Fin 101 → ℚ

theorem merger_proportion 
  (company_a_managers : ManagerProportion)
  (company_b_managers : ManagerProportion)
  (merged_managers : ManagerProportion)
  (h1 : company_a_managers 10 = 1)
  (h2 : company_b_managers 30 = 1)
  (h3 : merged_managers 25 = 1) :
  ∃ (result : MergedProportion), result 25 = 1 :=
sorry

end NUMINAMATH_CALUDE_merger_proportion_l2147_214797


namespace NUMINAMATH_CALUDE_gcd_lcm_pairs_l2147_214709

theorem gcd_lcm_pairs :
  (Nat.gcd 6 12 = 6 ∧ Nat.lcm 6 12 = 12) ∧
  (Nat.gcd 7 8 = 1 ∧ Nat.lcm 7 8 = 56) ∧
  (Nat.gcd 15 20 = 5 ∧ Nat.lcm 15 20 = 60) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_pairs_l2147_214709


namespace NUMINAMATH_CALUDE_doughnuts_eaten_l2147_214773

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of doughnuts in the box initially -/
def initial_doughnuts : ℕ := 2 * dozen

/-- The number of doughnuts remaining -/
def remaining_doughnuts : ℕ := 16

/-- The number of doughnuts eaten by the family -/
def eaten_doughnuts : ℕ := initial_doughnuts - remaining_doughnuts

theorem doughnuts_eaten : eaten_doughnuts = 8 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_eaten_l2147_214773


namespace NUMINAMATH_CALUDE_pentagon_h_coordinate_l2147_214702

structure Pentagon where
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ
  I : ℝ × ℝ
  J : ℝ × ℝ

def has_vertical_symmetry (p : Pentagon) : Prop := sorry

def area (p : Pentagon) : ℝ := sorry

theorem pentagon_h_coordinate (p : Pentagon) 
  (sym : has_vertical_symmetry p)
  (coords : p.F = (0, 0) ∧ p.G = (0, 6) ∧ p.H.1 = 3 ∧ p.J = (6, 0))
  (total_area : area p = 60) :
  p.H.2 = 14 := by sorry

end NUMINAMATH_CALUDE_pentagon_h_coordinate_l2147_214702


namespace NUMINAMATH_CALUDE_simplify_fraction_l2147_214708

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2147_214708


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2147_214736

theorem missing_fraction_sum (x : ℚ) : 
  (1/3 : ℚ) + (1/2 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (-17/30 : ℚ) = 
  (13333333333333333 : ℚ) / 100000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2147_214736


namespace NUMINAMATH_CALUDE_three_m_plus_n_equals_46_l2147_214738

theorem three_m_plus_n_equals_46 (m n : ℕ) 
  (h1 : m > n) 
  (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 
  3 * m + n = 46 := by
sorry

end NUMINAMATH_CALUDE_three_m_plus_n_equals_46_l2147_214738


namespace NUMINAMATH_CALUDE_river_width_l2147_214775

/-- Given a river with the following properties:
  * The river is 4 meters deep
  * The river flows at a rate of 6 kilometers per hour
  * The volume of water flowing into the sea is 26000 cubic meters per minute
  Prove that the width of the river is 65 meters. -/
theorem river_width (depth : ℝ) (flow_rate : ℝ) (volume : ℝ) :
  depth = 4 →
  flow_rate = 6 →
  volume = 26000 →
  (volume / (depth * (flow_rate * 1000 / 60))) = 65 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l2147_214775


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2147_214767

theorem tan_45_degrees_equals_one : 
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l2147_214767


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2147_214733

theorem complex_number_theorem (z : ℂ) :
  (∃ (k : ℝ), z / 4 = k * I) →
  Complex.abs z = 2 * Real.sqrt 5 →
  z = 2 * I ∨ z = -2 * I := by sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2147_214733


namespace NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l2147_214763

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    from 6 available topics is 5/6 -/
theorem prob_different_topics_is_five_sixths :
  prob_different_topics = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_different_topics_is_five_sixths_l2147_214763


namespace NUMINAMATH_CALUDE_power_difference_mod_eight_l2147_214731

theorem power_difference_mod_eight : 
  (47^1235 - 22^1235) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_mod_eight_l2147_214731


namespace NUMINAMATH_CALUDE_race_finish_orders_l2147_214764

theorem race_finish_orders (n : ℕ) : n = 4 → Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_finish_orders_l2147_214764


namespace NUMINAMATH_CALUDE_point_distance_inequality_l2147_214717

/-- Given points A(0,2), B(0,1), and D(t,0) with t > 0, and M(x,y) on line segment AD,
    if |AM| ≤ 2|BM| always holds, then t ≥ 2√3/3. -/
theorem point_distance_inequality (t : ℝ) (h_t : t > 0) :
  (∀ x y : ℝ, y = (2*t - 2*x)/t →
    x^2 + (y - 2)^2 ≤ 4 * (x^2 + (y - 1)^2)) →
  t ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_point_distance_inequality_l2147_214717


namespace NUMINAMATH_CALUDE_early_arrival_equals_walking_time_l2147_214705

/-- Represents the scenario of a man meeting his wife while walking home from the train station. -/
structure Scenario where
  /-- The time (in minutes) saved by meeting on the way compared to usual arrival time. -/
  time_saved : ℕ
  /-- The time (in minutes) the man spent walking before meeting his wife. -/
  walking_time : ℕ
  /-- The time (in minutes) the wife would normally drive to the station. -/
  normal_driving_time : ℕ
  /-- Assumption that the normal driving time is the difference between walking time and time saved. -/
  h_normal_driving : normal_driving_time = walking_time - time_saved

/-- Theorem stating that the time the man arrived early at the station equals his walking time. -/
theorem early_arrival_equals_walking_time (s : Scenario) :
  s.walking_time = s.walking_time :=
by sorry

#check early_arrival_equals_walking_time

end NUMINAMATH_CALUDE_early_arrival_equals_walking_time_l2147_214705


namespace NUMINAMATH_CALUDE_inequality_proof_l2147_214759

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2147_214759


namespace NUMINAMATH_CALUDE_rachel_winter_clothing_l2147_214732

theorem rachel_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 7 → scarves_per_box = 3 → mittens_per_box = 4 → 
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 49 := by
  sorry

end NUMINAMATH_CALUDE_rachel_winter_clothing_l2147_214732


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2147_214749

theorem sum_remainder_mod_seven
  (a b c : ℕ)
  (ha : 0 < a ∧ a < 7)
  (hb : 0 < b ∧ b < 7)
  (hc : 0 < c ∧ c < 7)
  (h1 : a * b * c % 7 = 1)
  (h2 : 4 * c % 7 = 3)
  (h3 : 5 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l2147_214749


namespace NUMINAMATH_CALUDE_not_perfect_square_different_parity_l2147_214722

theorem not_perfect_square_different_parity (a b : ℤ) 
  (h : a % 2 ≠ b % 2) : 
  ¬∃ (k : ℤ), (a + 3*b) * (5*a + 7*b) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_different_parity_l2147_214722


namespace NUMINAMATH_CALUDE_max_value_of_complex_number_l2147_214796

theorem max_value_of_complex_number (z : ℂ) : 
  Complex.abs (z - (3 - I)) = 2 → 
  (∀ w : ℂ, Complex.abs (w - (3 - I)) = 2 → Complex.abs (w + (1 + I)) ≤ Complex.abs (z + (1 + I))) → 
  Complex.abs (z + (1 + I)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_number_l2147_214796


namespace NUMINAMATH_CALUDE_similar_rectangle_ratio_l2147_214760

/-- Given a rectangle with length 40 meters and width 20 meters, 
    prove that a similar smaller rectangle with an area of 200 square meters 
    has dimensions that are 1/2 of the larger rectangle's dimensions. -/
theorem similar_rectangle_ratio (big_length big_width small_area : ℝ) 
  (h1 : big_length = 40)
  (h2 : big_width = 20)
  (h3 : small_area = 200)
  (h4 : small_area = (big_length * r) * (big_width * r)) 
  (r : ℝ) : r = 1 / 2 := by
  sorry

#check similar_rectangle_ratio

end NUMINAMATH_CALUDE_similar_rectangle_ratio_l2147_214760


namespace NUMINAMATH_CALUDE_candy_distribution_l2147_214782

theorem candy_distribution (bags : ℝ) (total_candy : ℕ) (h1 : bags = 15.0) (h2 : total_candy = 75) :
  (total_candy : ℝ) / bags = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2147_214782


namespace NUMINAMATH_CALUDE_larger_pile_size_l2147_214711

/-- Given two piles of toys where the total number is 120 and the larger pile
    is twice as big as the smaller pile, the number of toys in the larger pile is 80. -/
theorem larger_pile_size (small : ℕ) (large : ℕ) : 
  small + large = 120 → large = 2 * small → large = 80 := by
  sorry

end NUMINAMATH_CALUDE_larger_pile_size_l2147_214711


namespace NUMINAMATH_CALUDE_traffic_light_color_change_probability_l2147_214700

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle time of the traffic light -/
def totalCycleTime (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def colorChangeInterval (cycle : TrafficLightCycle) (observationTime : ℕ) : ℕ :=
  3 * observationTime

/-- Theorem: The probability of observing a color change in a randomly selected 
    4-second interval of a traffic light cycle is 12/85 -/
theorem traffic_light_color_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 40)
  (h2 : cycle.yellow = 5)
  (h3 : cycle.red = 40)
  (observationTime : ℕ)
  (h4 : observationTime = 4) :
  (colorChangeInterval cycle observationTime : ℚ) / (totalCycleTime cycle) = 12 / 85 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_color_change_probability_l2147_214700


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2147_214734

/-- Calculates the number of sampled students within a given interval using systematic sampling -/
def sampledStudentsInInterval (totalStudents : ℕ) (sampleSize : ℕ) (intervalStart : ℕ) (intervalEnd : ℕ) : ℕ :=
  let intervalSize := intervalEnd - intervalStart + 1
  let samplingInterval := totalStudents / sampleSize
  intervalSize / samplingInterval

theorem systematic_sampling_theorem :
  sampledStudentsInInterval 1221 37 496 825 = 10 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2147_214734


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l2147_214743

theorem range_of_a_for_false_proposition :
  ∀ (a : ℝ),
    (¬ ∃ (x₀ : ℝ), x₀^2 + a*x₀ - 4*a < 0) ↔
    (a ∈ Set.Icc (-16 : ℝ) 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l2147_214743


namespace NUMINAMATH_CALUDE_last_twelve_average_l2147_214748

theorem last_twelve_average (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (thirteenth_result : ℝ) :
  total_count = 25 →
  total_average = 18 →
  first_twelve_average = 10 →
  thirteenth_result = 90 →
  (total_count * total_average - 12 * first_twelve_average - thirteenth_result) / 12 = 20 := by
sorry

end NUMINAMATH_CALUDE_last_twelve_average_l2147_214748


namespace NUMINAMATH_CALUDE_legacy_cleaning_time_l2147_214751

/-- The number of floors in the building -/
def num_floors : ℕ := 4

/-- The number of rooms per floor -/
def rooms_per_floor : ℕ := 10

/-- Legacy's hourly rate in dollars -/
def hourly_rate : ℕ := 15

/-- Total earnings from cleaning all floors in dollars -/
def total_earnings : ℕ := 3600

/-- Time to clean one room in hours -/
def time_per_room : ℚ := 6

theorem legacy_cleaning_time :
  time_per_room = (total_earnings : ℚ) / (hourly_rate * num_floors * rooms_per_floor : ℚ) :=
sorry

end NUMINAMATH_CALUDE_legacy_cleaning_time_l2147_214751


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2147_214768

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : Real :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_range (t : Triangle) 
  (h1 : t.B = π/3) 
  (h2 : t.b = 2 * Real.sqrt 3) 
  (h3 : t.A > 0) 
  (h4 : t.C > 0) 
  (h5 : t.A + t.B + t.C = π) :
  4 * Real.sqrt 3 < perimeter t ∧ perimeter t ≤ 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2147_214768


namespace NUMINAMATH_CALUDE_complement_intersection_M_N_l2147_214784

def M : Set ℝ := {x : ℝ | x ≥ 1}
def N : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection_M_N :
  (M ∩ N)ᶜ = {x : ℝ | x < 1 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_M_N_l2147_214784


namespace NUMINAMATH_CALUDE_S_infinite_l2147_214725

/-- The number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (n : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The main theorem stating that S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l2147_214725


namespace NUMINAMATH_CALUDE_smallest_perfect_square_tiling_l2147_214729

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length. -/
structure Square where
  side : ℕ

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- The area of a square. -/
def Square.area (s : Square) : ℕ := s.side * s.side

/-- A rectangle fits in a square if its width and height are both less than or equal to the square's side length. -/
def fits_in (r : Rectangle) (s : Square) : Prop := r.width ≤ s.side ∧ r.height ≤ s.side

/-- A square is perfectly tiled by rectangles if the sum of the areas of the rectangles equals the area of the square. -/
def perfectly_tiled (s : Square) (rs : List Rectangle) : Prop :=
  (rs.map Rectangle.area).sum = s.area

theorem smallest_perfect_square_tiling :
  ∃ (s : Square) (rs : List Rectangle),
    (∀ r ∈ rs, r.width = 3 ∧ r.height = 4) ∧
    perfectly_tiled s rs ∧
    (∀ r ∈ rs, fits_in r s) ∧
    rs.length = 12 ∧
    s.side = 12 ∧
    (∀ (s' : Square) (rs' : List Rectangle),
      (∀ r ∈ rs', r.width = 3 ∧ r.height = 4) →
      perfectly_tiled s' rs' →
      (∀ r ∈ rs', fits_in r s') →
      s'.side ≥ s.side) := by
  sorry

#check smallest_perfect_square_tiling

end NUMINAMATH_CALUDE_smallest_perfect_square_tiling_l2147_214729


namespace NUMINAMATH_CALUDE_product_sign_implication_l2147_214789

theorem product_sign_implication (a b c d : ℝ) :
  (a * b * c * d < 0) →
  (a > 0) →
  (b > c) →
  (d < 0) →
  ((0 < c ∧ c < b) ∨ (c < b ∧ b < 0)) :=
by sorry

end NUMINAMATH_CALUDE_product_sign_implication_l2147_214789


namespace NUMINAMATH_CALUDE_remainder_double_n_l2147_214714

theorem remainder_double_n (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_double_n_l2147_214714


namespace NUMINAMATH_CALUDE_positive_number_equation_l2147_214750

theorem positive_number_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^a) (h4 : b = 3*a) : a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_l2147_214750


namespace NUMINAMATH_CALUDE_min_gcd_of_primes_squared_minus_one_l2147_214791

theorem min_gcd_of_primes_squared_minus_one (p q : Nat) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hp_gt_100 : p > 100) (hq_gt_100 : q > 100) : 
  Nat.gcd (p^2 - 1) (q^2 - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_gcd_of_primes_squared_minus_one_l2147_214791


namespace NUMINAMATH_CALUDE_students_per_school_is_247_l2147_214728

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := 6175

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := total_students / num_schools

/-- Theorem stating that the number of students in each elementary school is 247 -/
theorem students_per_school_is_247 : students_per_school = 247 := by sorry

end NUMINAMATH_CALUDE_students_per_school_is_247_l2147_214728


namespace NUMINAMATH_CALUDE_num_possible_lists_eq_1728_l2147_214716

/-- The number of balls in the bin -/
def num_balls : ℕ := 12

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The number of possible lists when drawing with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 1728 -/
theorem num_possible_lists_eq_1728 : num_possible_lists = 1728 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_eq_1728_l2147_214716


namespace NUMINAMATH_CALUDE_first_term_value_l2147_214735

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_value 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_first_term_value_l2147_214735


namespace NUMINAMATH_CALUDE_certain_number_value_l2147_214707

theorem certain_number_value : ∃ x : ℝ, 25 * x = 675 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2147_214707


namespace NUMINAMATH_CALUDE_ted_banana_purchase_l2147_214723

/-- The number of oranges Ted needs to purchase -/
def num_oranges : ℕ := 10

/-- The cost of one banana in dollars -/
def banana_cost : ℚ := 2

/-- The cost of one orange in dollars -/
def orange_cost : ℚ := 3/2

/-- The total cost of the fruits in dollars -/
def total_cost : ℚ := 25

/-- The number of bananas Ted needs to purchase -/
def num_bananas : ℕ := 5

theorem ted_banana_purchase :
  num_bananas * banana_cost + num_oranges * orange_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_ted_banana_purchase_l2147_214723


namespace NUMINAMATH_CALUDE_circle_centers_distance_bound_l2147_214758

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Sum of reciprocals of distances between circle centers -/
def sum_reciprocal_distances (circles : List Circle) : ℝ := sorry

/-- No line meets more than two circles -/
def no_line_meets_more_than_two (circles : List Circle) : Prop := sorry

theorem circle_centers_distance_bound (n : ℕ) (circles : List Circle) 
  (h1 : circles.length = n)
  (h2 : ∀ c ∈ circles, c.radius = 1)
  (h3 : no_line_meets_more_than_two circles) :
  sum_reciprocal_distances circles ≤ (n - 1 : ℝ) * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_centers_distance_bound_l2147_214758


namespace NUMINAMATH_CALUDE_pizza_combinations_l2147_214719

def number_of_toppings : ℕ := 8

theorem pizza_combinations : 
  (number_of_toppings) +                    -- one-topping pizzas
  (number_of_toppings.choose 2) +           -- two-topping pizzas
  (number_of_toppings.choose 3) = 92 :=     -- three-topping pizzas
by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l2147_214719


namespace NUMINAMATH_CALUDE_john_memory_card_cost_l2147_214769

/-- The number of pictures John takes per day -/
def pictures_per_day : ℕ := 10

/-- The number of years John has been taking pictures -/
def years : ℕ := 3

/-- The number of images a memory card can store -/
def images_per_card : ℕ := 50

/-- The cost of each memory card in dollars -/
def cost_per_card : ℕ := 60

/-- The number of days in a year (assuming no leap years) -/
def days_per_year : ℕ := 365

theorem john_memory_card_cost :
  (years * days_per_year * pictures_per_day / images_per_card) * cost_per_card = 13140 :=
sorry

end NUMINAMATH_CALUDE_john_memory_card_cost_l2147_214769


namespace NUMINAMATH_CALUDE_unique_prime_with_same_remainder_l2147_214745

theorem unique_prime_with_same_remainder : 
  ∃! n : ℕ, 
    Prime n ∧ 
    200 < n ∧ 
    n < 300 ∧ 
    ∃ r : ℕ, n % 7 = r ∧ n % 9 = r :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_with_same_remainder_l2147_214745


namespace NUMINAMATH_CALUDE_owner_away_time_l2147_214756

/-- Calculates the time an owner was away based on cat's kibble consumption --/
def time_away (initial_kibble : ℝ) (remaining_kibble : ℝ) (consumption_rate : ℝ) : ℝ :=
  (initial_kibble - remaining_kibble) * consumption_rate

/-- Theorem stating that given the conditions, the owner was away for 8 hours --/
theorem owner_away_time (cat_consumption_rate : ℝ) (initial_kibble : ℝ) (remaining_kibble : ℝ)
  (h1 : cat_consumption_rate = 4) -- Cat eats 1 pound every 4 hours
  (h2 : initial_kibble = 3) -- Bowl filled with 3 pounds initially
  (h3 : remaining_kibble = 1) -- 1 pound remains when owner returns
  : time_away initial_kibble remaining_kibble cat_consumption_rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_owner_away_time_l2147_214756


namespace NUMINAMATH_CALUDE_solution_pairs_l2147_214740

theorem solution_pairs (x y : ℝ) (hxy : x ≠ y) 
  (eq1 : x^100 - y^100 = 2^99 * (x - y))
  (eq2 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2147_214740


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2147_214744

-- Define set A
def A : Set ℝ := {x | |x - 1| < 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.log (x^2 + 1)}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2147_214744


namespace NUMINAMATH_CALUDE_expression_evaluation_l2147_214794

theorem expression_evaluation (x y z : ℝ) : 
  (x - (y + z)) - ((x + y) - 2*z) = -2*y - 3*z := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2147_214794


namespace NUMINAMATH_CALUDE_builder_wage_is_100_l2147_214710

/-- The daily wage of a builder given the construction rates and total cost -/
def builder_daily_wage (builders_per_floor : ℕ) (days_per_floor : ℕ) 
  (total_builders : ℕ) (total_houses : ℕ) (floors_per_house : ℕ) 
  (total_cost : ℕ) : ℚ :=
  (total_cost : ℚ) / (total_builders * total_houses * floors_per_house * days_per_floor : ℚ)

theorem builder_wage_is_100 :
  builder_daily_wage 3 30 6 5 6 270000 = 100 := by sorry

end NUMINAMATH_CALUDE_builder_wage_is_100_l2147_214710


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l2147_214746

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Three terms form an arithmetic sequence -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  arithmetic_sequence (a 4) (a 5) (a 6) →
  q = 1 ∨ q = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l2147_214746


namespace NUMINAMATH_CALUDE_unique_x_exists_l2147_214766

theorem unique_x_exists : ∃! x : ℝ, x > 0 ∧ x * ↑(⌊x⌋) = 50 ∧ |x - 7.142857| < 0.000001 := by sorry

end NUMINAMATH_CALUDE_unique_x_exists_l2147_214766


namespace NUMINAMATH_CALUDE_weekend_getaway_cost_sharing_l2147_214777

/-- A weekend getaway cost-sharing problem -/
theorem weekend_getaway_cost_sharing 
  (henry_paid linda_paid jack_paid : ℝ)
  (h l : ℝ)
  (henry_paid_amount : henry_paid = 120)
  (linda_paid_amount : linda_paid = 150)
  (jack_paid_amount : jack_paid = 210)
  (total_cost : henry_paid + linda_paid + jack_paid = henry_paid + linda_paid + jack_paid)
  (even_split : (henry_paid + linda_paid + jack_paid) / 3 = henry_paid + h)
  (even_split' : (henry_paid + linda_paid + jack_paid) / 3 = linda_paid + l)
  : h - l = 30 := by sorry

end NUMINAMATH_CALUDE_weekend_getaway_cost_sharing_l2147_214777


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2147_214793

theorem gcd_of_three_numbers : Nat.gcd 17934 (Nat.gcd 23526 51774) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2147_214793


namespace NUMINAMATH_CALUDE_new_student_weight_l2147_214792

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_avg = 14.6 →
  (initial_count : ℝ) * initial_avg + (initial_count + 1 : ℝ) * new_avg - (initial_count : ℝ) * initial_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l2147_214792


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2147_214737

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The main theorem -/
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) : 
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2147_214737


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2147_214753

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_average_age : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 25)
  (h3 : team_average_age = 23)
  (h4 : ∃ (wicket_keeper_age : ℕ), 
    wicket_keeper_age > captain_age ∧ 
    (team_size : ℝ) * team_average_age = 
      (team_size - 2 : ℝ) * (team_average_age - 1) + captain_age + wicket_keeper_age) :
  ∃ (wicket_keeper_age : ℕ), wicket_keeper_age = captain_age + 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2147_214753


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2147_214762

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define the interval
def I : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem quadratic_max_value (t : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ I → f t x ≤ m) ∧
  (t < 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ -2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = -2*t + 2)) ∧
  (t = 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2)) ∧
  (t > 0 → (∀ (x : ℝ), x ∈ I → f t x ≤ 2*t + 2) ∧ (∃ (x : ℝ), x ∈ I ∧ f t x = 2*t + 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2147_214762


namespace NUMINAMATH_CALUDE_mans_downstream_speed_l2147_214754

theorem mans_downstream_speed 
  (upstream_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : upstream_speed = 8) 
  (h2 : stream_speed = 2.5) : 
  upstream_speed + 2 * stream_speed = 13 := by
  sorry

end NUMINAMATH_CALUDE_mans_downstream_speed_l2147_214754


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l2147_214783

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (∃ k : ℕ, x = 6 * k) ∧ 
  50 ≤ x ∧ x ≤ 150 :=
by sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l2147_214783


namespace NUMINAMATH_CALUDE_meditation_time_per_week_l2147_214706

/-- Calculates the total hours spent meditating in a week given the daily meditation time in minutes -/
def weekly_meditation_hours (daily_minutes : ℕ) : ℚ :=
  (daily_minutes : ℚ) * 7 / 60

theorem meditation_time_per_week :
  weekly_meditation_hours (30 * 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_meditation_time_per_week_l2147_214706


namespace NUMINAMATH_CALUDE_quadratic_equation_problem1_quadratic_equation_problem2_l2147_214786

-- Problem 1
theorem quadratic_equation_problem1 (x : ℝ) :
  (x - 5)^2 - 16 = 0 ↔ x = 9 ∨ x = 1 := by sorry

-- Problem 2
theorem quadratic_equation_problem2 (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem1_quadratic_equation_problem2_l2147_214786


namespace NUMINAMATH_CALUDE_ai_chip_pass_rate_below_threshold_l2147_214704

-- Define the probabilities for intelligent testing indicators
def p_safety : ℚ := 49/50
def p_energy : ℚ := 48/49
def p_performance : ℚ := 47/48

-- Define the probability of passing manual testing
def p_manual : ℚ := 49/50

-- Define the number of chips selected for manual testing
def n_chips : ℕ := 50

-- Theorem statement
theorem ai_chip_pass_rate_below_threshold :
  let p_intelligent := p_safety * p_energy * p_performance
  let p_overall := p_intelligent * p_manual
  p_overall < 93/100 := by
  sorry

end NUMINAMATH_CALUDE_ai_chip_pass_rate_below_threshold_l2147_214704


namespace NUMINAMATH_CALUDE_trapezoid_fg_squared_l2147_214781

/-- Represents a trapezoid EFGH with specific properties -/
structure Trapezoid where
  /-- Length of EF -/
  ef : ℝ
  /-- Length of EH -/
  eh : ℝ
  /-- FG is perpendicular to EF and GH -/
  fg_perpendicular : Bool
  /-- Diagonals EG and FH are perpendicular -/
  diagonals_perpendicular : Bool

/-- Theorem about the length of FG in a specific trapezoid -/
theorem trapezoid_fg_squared (t : Trapezoid) 
  (h1 : t.ef = 3)
  (h2 : t.eh = Real.sqrt 2001)
  (h3 : t.fg_perpendicular = true)
  (h4 : t.diagonals_perpendicular = true) :
  ∃ (fg : ℝ), fg^2 = (9 + 3 * Real.sqrt 7977) / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_fg_squared_l2147_214781


namespace NUMINAMATH_CALUDE_sum_of_solutions_x_squared_36_l2147_214730

theorem sum_of_solutions_x_squared_36 (x : ℝ) (h : x^2 = 36) :
  ∃ (y : ℝ), y^2 = 36 ∧ x + y = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_x_squared_36_l2147_214730


namespace NUMINAMATH_CALUDE_isabellas_haircut_l2147_214799

/-- Given an initial hair length and an amount cut off, 
    calculate the resulting hair length after a haircut. -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem: Isabella's hair length after the haircut is 9 inches. -/
theorem isabellas_haircut : hair_length_after_cut 18 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircut_l2147_214799


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l2147_214798

theorem wall_painting_fraction :
  let total_wall : ℚ := 1
  let matilda_half : ℚ := 1/2
  let ellie_half : ℚ := 1/2
  let matilda_painted : ℚ := matilda_half * (1/2)
  let ellie_painted : ℚ := ellie_half * (1/3)
  matilda_painted + ellie_painted = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l2147_214798


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2147_214742

/-- The perimeter of a semicircle with radius r is equal to 2r + πr -/
theorem semicircle_perimeter (r : ℝ) (h : r = 35) :
  ∃ P : ℝ, P = 2 * r + π * r := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2147_214742


namespace NUMINAMATH_CALUDE_rabbit_storage_l2147_214703

/-- Represents the number of items stored per hole for each animal -/
structure StorageRate where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- Represents the number of holes dug by each animal -/
structure Holes where
  rabbit : ℕ
  deer : ℕ
  fox : ℕ

/-- The main theorem stating that given the conditions, the rabbit stored 60 items -/
theorem rabbit_storage (rate : StorageRate) (holes : Holes) : 
  rate.rabbit = 4 →
  rate.deer = 5 →
  rate.fox = 7 →
  rate.rabbit * holes.rabbit = rate.deer * holes.deer →
  rate.rabbit * holes.rabbit = rate.fox * holes.fox →
  holes.deer = holes.rabbit - 3 →
  holes.fox = holes.deer + 2 →
  rate.rabbit * holes.rabbit = 60 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_storage_l2147_214703


namespace NUMINAMATH_CALUDE_distance_city_A_to_C_l2147_214778

/-- Prove the distance between city A and city C given travel times and speeds -/
theorem distance_city_A_to_C 
  (time_Eddy : ℝ) 
  (time_Freddy : ℝ) 
  (distance_AB : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : time_Eddy = 3) 
  (h2 : time_Freddy = 4) 
  (h3 : distance_AB = 600) 
  (h4 : speed_ratio = 1.7391304347826086) : 
  ∃ distance_AC : ℝ, distance_AC = 460 := by
  sorry

end NUMINAMATH_CALUDE_distance_city_A_to_C_l2147_214778


namespace NUMINAMATH_CALUDE_vector_equation_solution_l2147_214765

variable (V : Type*) [AddCommGroup V] [Module ℝ V]
variable (a : V)

theorem vector_equation_solution (x : V) (h : 2 • x - 3 • (x - 2 • a) = 0) : x = 6 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l2147_214765


namespace NUMINAMATH_CALUDE_square_sum_equals_eight_l2147_214720

theorem square_sum_equals_eight (m : ℝ) 
  (h : (2018 + m) * (2020 + m) = 2) : 
  (2018 + m)^2 + (2020 + m)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_eight_l2147_214720


namespace NUMINAMATH_CALUDE_slope_of_line_AB_l2147_214741

/-- Given points A(2, 0) and B(3, √3), prove that the slope of line AB is √3 -/
theorem slope_of_line_AB (A B : ℝ × ℝ) : 
  A = (2, 0) → B = (3, Real.sqrt 3) → (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_AB_l2147_214741


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2147_214726

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y x₀ y₀ : ℝ) : 
  (x₀^2 + y₀^2 = 1) →  -- P is on the unit circle
  (x = (x₀ + 3) / 2) →  -- x-coordinate of midpoint M
  (y = y₀ / 2) →  -- y-coordinate of midpoint M
  ((2*x - 3)^2 + 4*y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2147_214726


namespace NUMINAMATH_CALUDE_P_inter_Q_equiv_l2147_214701

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem P_inter_Q_equiv : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_P_inter_Q_equiv_l2147_214701


namespace NUMINAMATH_CALUDE_jason_music_store_expenses_l2147_214772

theorem jason_music_store_expenses :
  let flute : ℝ := 142.46
  let music_tool : ℝ := 8.89
  let song_book : ℝ := 7.00
  let flute_case : ℝ := 35.25
  let music_stand : ℝ := 12.15
  let cleaning_kit : ℝ := 14.99
  let sheet_protectors : ℝ := 3.29
  flute + music_tool + song_book + flute_case + music_stand + cleaning_kit + sheet_protectors = 224.03 := by
  sorry

end NUMINAMATH_CALUDE_jason_music_store_expenses_l2147_214772


namespace NUMINAMATH_CALUDE_hockey_league_teams_l2147_214755

/-- The number of teams in a hockey league -/
def num_teams : ℕ := 15

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season -/
def total_games : ℕ := 1050

/-- Theorem stating that the number of teams is correct given the conditions -/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) / 2) * games_per_pair = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l2147_214755


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l2147_214761

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - 2*a)^2 = 5

-- Define the points A, B, and P
def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (1, 6)
def point_P : ℝ × ℝ := (-1, 3)

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def tangent_line_2 (x y : ℝ) : Prop := x + 2*y - 5 = 0

theorem circle_and_tangent_lines :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), circle_C x y → (x - 2)^2 + (y - 4)^2 = 5) ∧
  (∀ (x y : ℝ), (tangent_line_1 x y ∨ tangent_line_2 x y) →
    (∃ (t : ℝ), circle_C (t*x + (1-t)*point_P.1) (t*y + (1-t)*point_P.2)) ∧
    (∀ (s : ℝ), s ≠ t → ¬ circle_C (s*x + (1-s)*point_P.1) (s*y + (1-s)*point_P.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l2147_214761


namespace NUMINAMATH_CALUDE_center_cell_value_l2147_214757

theorem center_cell_value (a b c d e f g h i : ℝ) 
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ i > 0)
  (row_products : a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1)
  (col_products : a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1)
  (square_products : a * d * e * b = 2 ∧ b * e * f * c = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2) :
  e = 1 := by
sorry

end NUMINAMATH_CALUDE_center_cell_value_l2147_214757


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2147_214727

theorem problem_solving_probability 
  (prob_A prob_B : ℚ) 
  (h_A : prob_A = 2/3) 
  (h_B : prob_B = 3/4) : 
  1 - (1 - prob_A) * (1 - prob_B) = 11/12 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2147_214727


namespace NUMINAMATH_CALUDE_dans_team_total_games_l2147_214712

/-- Represents a baseball team's game results -/
structure BaseballTeam where
  wins : ℕ
  losses : ℕ

/-- The total number of games played by a baseball team -/
def total_games (team : BaseballTeam) : ℕ :=
  team.wins + team.losses

/-- Theorem: Dan's high school baseball team played 18 games in total -/
theorem dans_team_total_games :
  ∃ (team : BaseballTeam), team.wins = 15 ∧ team.losses = 3 ∧ total_games team = 18 :=
sorry

end NUMINAMATH_CALUDE_dans_team_total_games_l2147_214712


namespace NUMINAMATH_CALUDE_multiplication_difference_l2147_214747

theorem multiplication_difference (number : ℕ) (correct_multiplier : ℕ) (mistaken_multiplier : ℕ) :
  number = 135 →
  correct_multiplier = 43 →
  mistaken_multiplier = 34 →
  (number * correct_multiplier) - (number * mistaken_multiplier) = 1215 := by
sorry

end NUMINAMATH_CALUDE_multiplication_difference_l2147_214747


namespace NUMINAMATH_CALUDE_distinct_remainders_mod_14_l2147_214771

theorem distinct_remainders_mod_14 : ∃ (a b c d e : ℕ),
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  1 ≤ d ∧ d ≤ 13 ∧
  1 ≤ e ∧ e ≤ 13 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (a * b) % 14 ≠ (a * c) % 14 ∧
  (a * b) % 14 ≠ (a * d) % 14 ∧
  (a * b) % 14 ≠ (a * e) % 14 ∧
  (a * b) % 14 ≠ (b * c) % 14 ∧
  (a * b) % 14 ≠ (b * d) % 14 ∧
  (a * b) % 14 ≠ (b * e) % 14 ∧
  (a * b) % 14 ≠ (c * d) % 14 ∧
  (a * b) % 14 ≠ (c * e) % 14 ∧
  (a * b) % 14 ≠ (d * e) % 14 ∧
  (a * c) % 14 ≠ (a * d) % 14 ∧
  (a * c) % 14 ≠ (a * e) % 14 ∧
  (a * c) % 14 ≠ (b * c) % 14 ∧
  (a * c) % 14 ≠ (b * d) % 14 ∧
  (a * c) % 14 ≠ (b * e) % 14 ∧
  (a * c) % 14 ≠ (c * d) % 14 ∧
  (a * c) % 14 ≠ (c * e) % 14 ∧
  (a * c) % 14 ≠ (d * e) % 14 ∧
  (a * d) % 14 ≠ (a * e) % 14 ∧
  (a * d) % 14 ≠ (b * c) % 14 ∧
  (a * d) % 14 ≠ (b * d) % 14 ∧
  (a * d) % 14 ≠ (b * e) % 14 ∧
  (a * d) % 14 ≠ (c * d) % 14 ∧
  (a * d) % 14 ≠ (c * e) % 14 ∧
  (a * d) % 14 ≠ (d * e) % 14 ∧
  (a * e) % 14 ≠ (b * c) % 14 ∧
  (a * e) % 14 ≠ (b * d) % 14 ∧
  (a * e) % 14 ≠ (b * e) % 14 ∧
  (a * e) % 14 ≠ (c * d) % 14 ∧
  (a * e) % 14 ≠ (c * e) % 14 ∧
  (a * e) % 14 ≠ (d * e) % 14 ∧
  (b * c) % 14 ≠ (b * d) % 14 ∧
  (b * c) % 14 ≠ (b * e) % 14 ∧
  (b * c) % 14 ≠ (c * d) % 14 ∧
  (b * c) % 14 ≠ (c * e) % 14 ∧
  (b * c) % 14 ≠ (d * e) % 14 ∧
  (b * d) % 14 ≠ (b * e) % 14 ∧
  (b * d) % 14 ≠ (c * d) % 14 ∧
  (b * d) % 14 ≠ (c * e) % 14 ∧
  (b * d) % 14 ≠ (d * e) % 14 ∧
  (b * e) % 14 ≠ (c * d) % 14 ∧
  (b * e) % 14 ≠ (c * e) % 14 ∧
  (b * e) % 14 ≠ (d * e) % 14 ∧
  (c * d) % 14 ≠ (c * e) % 14 ∧
  (c * d) % 14 ≠ (d * e) % 14 ∧
  (c * e) % 14 ≠ (d * e) % 14 :=
by sorry

end NUMINAMATH_CALUDE_distinct_remainders_mod_14_l2147_214771


namespace NUMINAMATH_CALUDE_block_partition_l2147_214790

theorem block_partition (n : ℕ) (weights : List ℕ) : 
  weights.length = n →
  (∀ w ∈ weights, 1 ≤ w ∧ w < n) →
  weights.sum < 2 * n →
  ∃ (subset : List ℕ), subset ⊆ weights ∧ subset.sum = n := by
  sorry

end NUMINAMATH_CALUDE_block_partition_l2147_214790


namespace NUMINAMATH_CALUDE_average_age_across_rooms_l2147_214787

theorem average_age_across_rooms (room_a_people room_b_people room_c_people : ℕ)
                                 (room_a_avg room_b_avg room_c_avg : ℚ)
                                 (h1 : room_a_people = 8)
                                 (h2 : room_b_people = 5)
                                 (h3 : room_c_people = 7)
                                 (h4 : room_a_avg = 35)
                                 (h5 : room_b_avg = 30)
                                 (h6 : room_c_avg = 25) :
  (room_a_people * room_a_avg + room_b_people * room_b_avg + room_c_people * room_c_avg) /
  (room_a_people + room_b_people + room_c_people : ℚ) = 30.25 := by
  sorry

end NUMINAMATH_CALUDE_average_age_across_rooms_l2147_214787


namespace NUMINAMATH_CALUDE_amys_remaining_money_is_56_04_l2147_214739

/-- Calculates the amount of money Amy has left after her purchases --/
def amys_remaining_money (initial_amount : ℚ) (doll_price : ℚ) (doll_count : ℕ)
  (board_game_price : ℚ) (board_game_count : ℕ) (comic_book_price : ℚ) (comic_book_count : ℕ)
  (board_game_discount : ℚ) (sales_tax : ℚ) : ℚ :=
  let doll_cost := doll_price * doll_count
  let board_game_cost := board_game_price * board_game_count
  let comic_book_cost := comic_book_price * comic_book_count
  let discounted_board_game_cost := board_game_cost * (1 - board_game_discount)
  let total_before_tax := doll_cost + discounted_board_game_cost + comic_book_cost
  let total_after_tax := total_before_tax * (1 + sales_tax)
  initial_amount - total_after_tax

/-- Theorem stating that Amy has $56.04 left after her purchases --/
theorem amys_remaining_money_is_56_04 :
  amys_remaining_money 100 1.25 3 12.75 2 3.50 4 0.10 0.08 = 56.04 := by
  sorry

end NUMINAMATH_CALUDE_amys_remaining_money_is_56_04_l2147_214739
