import Mathlib

namespace product_sum_base_c_l1754_175453

def base_c_to_decimal (n : ℕ) (c : ℕ) : ℕ := c + n

def decimal_to_base_c (n : ℕ) (c : ℕ) : ℕ := n - c

theorem product_sum_base_c (c : ℕ) : 
  (base_c_to_decimal 12 c) * (base_c_to_decimal 14 c) * (base_c_to_decimal 18 c) = 
    5 * c^3 + 3 * c^2 + 2 * c + 0 →
  decimal_to_base_c (base_c_to_decimal 12 c + base_c_to_decimal 14 c + 
                     base_c_to_decimal 18 c + base_c_to_decimal 20 c) c = 40 :=
by sorry

end product_sum_base_c_l1754_175453


namespace addition_and_subtraction_of_integers_and_fractions_l1754_175401

theorem addition_and_subtraction_of_integers_and_fractions :
  (1 : ℤ) * 17 + (-12) = 5 ∧ -((1 : ℚ) / 7) - (-(6 / 7)) = 5 / 7 := by
  sorry

end addition_and_subtraction_of_integers_and_fractions_l1754_175401


namespace equilateral_triangle_product_l1754_175497

/-- Given that (0,0), (a,11), and (b,37) form an equilateral triangle, prove that ab = 315 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((a + 11 * Complex.I) * (Complex.exp (Complex.I * Real.pi / 3)) = b + 37 * Complex.I) →
  a * b = 315 := by
  sorry

end equilateral_triangle_product_l1754_175497


namespace smallest_sum_is_26_l1754_175483

def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧ (1978^m) % 1000 = (1978^n) % 1000

theorem smallest_sum_is_26 :
  ∃ (m n : ℕ), is_valid_pair m n ∧ m + n = 26 ∧
  ∀ (m' n' : ℕ), is_valid_pair m' n' → m' + n' ≥ 26 :=
sorry

end smallest_sum_is_26_l1754_175483


namespace ternary_to_decimal_l1754_175467

theorem ternary_to_decimal :
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0 : ℕ) = 16 := by sorry

end ternary_to_decimal_l1754_175467


namespace bus_motion_time_is_24_minutes_l1754_175496

/-- Represents the bus journey on a highway -/
structure BusJourney where
  distance : ℝ  -- Total distance in km
  num_stops : ℕ -- Number of intermediate stops
  stop_duration : ℝ -- Duration of each stop in minutes
  speed_difference : ℝ -- Difference in km/h between non-stop speed and average speed with stops

/-- Calculates the time the bus is in motion -/
def motion_time (journey : BusJourney) : ℝ :=
  sorry

/-- The main theorem stating that the motion time is 24 minutes for the given conditions -/
theorem bus_motion_time_is_24_minutes (journey : BusJourney) 
  (h1 : journey.distance = 10)
  (h2 : journey.num_stops = 6)
  (h3 : journey.stop_duration = 1)
  (h4 : journey.speed_difference = 5) :
  motion_time journey = 24 :=
sorry

end bus_motion_time_is_24_minutes_l1754_175496


namespace train_overtake_time_l1754_175407

/-- The time taken for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed : ℝ) (train_length : ℝ) :
  train_speed = 100 →
  motorbike_speed = 64 →
  train_length = 400.032 →
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 40.0032 := by
  sorry

end train_overtake_time_l1754_175407


namespace inequality_problem_l1754_175433

theorem inequality_problem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1/a > b^2 - 1/b) := by
  sorry

end inequality_problem_l1754_175433


namespace handshakes_count_l1754_175492

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- People who know each other
  group2_size : ℕ  -- People who don't know anyone
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_count :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end handshakes_count_l1754_175492


namespace vector_linear_combination_l1754_175425

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b. -/
theorem vector_linear_combination (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, 2) → c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
by sorry

end vector_linear_combination_l1754_175425


namespace ones_digit_of_largest_power_of_three_dividing_18_factorial_l1754_175440

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_power_of_three_dividing (n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc x => acc + (if x % 3 = 0 then 1 else 0) + (if x % 9 = 0 then 1 else 0)) 0

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_largest_power_of_three_dividing_18_factorial :
  ones_digit (3^(largest_power_of_three_dividing (factorial 18))) = 1 := by
  sorry

end ones_digit_of_largest_power_of_three_dividing_18_factorial_l1754_175440


namespace diagonals_parity_iff_n_parity_l1754_175406

/-- The number of diagonals in a regular polygon with 2n+1 sides. -/
def num_diagonals (n : ℕ) : ℕ := (2 * n + 1).choose 2 - (2 * n + 1)

/-- Theorem: The number of diagonals in a regular polygon with 2n+1 sides is odd if and only if n is even. -/
theorem diagonals_parity_iff_n_parity (n : ℕ) (h : n > 1) :
  Odd (num_diagonals n) ↔ Even n := by
  sorry

end diagonals_parity_iff_n_parity_l1754_175406


namespace parabola_coefficient_sum_l1754_175454

/-- A parabola with equation y = dx^2 + ex + f, vertex (-3, 2), and passing through (-5, 10) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 2 = d * (-3)^2 + e * (-3) + f
  point_condition : 10 = d * (-5)^2 + e * (-5) + f

/-- The sum of coefficients d, e, and f equals 10 -/
theorem parabola_coefficient_sum (p : Parabola) : p.d + p.e + p.f = 10 := by
  sorry

end parabola_coefficient_sum_l1754_175454


namespace sports_club_size_l1754_175446

/-- The number of members in a sports club -/
def sports_club_members (B T BT N : ℕ) : ℕ :=
  B + T - BT + N

/-- Theorem: The sports club has 30 members -/
theorem sports_club_size :
  ∃ (B T BT N : ℕ),
    B = 17 ∧
    T = 17 ∧
    BT = 6 ∧
    N = 2 ∧
    sports_club_members B T BT N = 30 := by
  sorry

end sports_club_size_l1754_175446


namespace exponential_function_fixed_point_l1754_175437

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^x + 1 passes through the point (0, 2) -/
theorem exponential_function_fixed_point 
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
  sorry

end exponential_function_fixed_point_l1754_175437


namespace solve_system_for_y_l1754_175435

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - 3 * y = 18) 
  (eq2 : x + 2 * y = 8) : 
  y = -2 / 7 := by
sorry

end solve_system_for_y_l1754_175435


namespace modified_prism_edge_count_l1754_175423

/-- Represents a modified rectangular prism with intersecting corner cuts -/
structure ModifiedPrism where
  original_edges : Nat
  vertex_count : Nat
  new_edges_per_vertex : Nat
  intersections_per_vertex : Nat
  additional_edges_per_intersection : Nat

/-- Calculates the total number of edges in the modified prism -/
def total_edges (p : ModifiedPrism) : Nat :=
  p.original_edges + 
  (p.vertex_count * p.new_edges_per_vertex) + 
  (p.vertex_count * p.intersections_per_vertex * p.additional_edges_per_intersection)

/-- Theorem stating that the modified prism has 52 edges -/
theorem modified_prism_edge_count :
  ∃ (p : ModifiedPrism), total_edges p = 52 :=
sorry

end modified_prism_edge_count_l1754_175423


namespace taxi_trip_length_l1754_175477

/-- Calculates the trip length in miles given the taxi fare parameters and total charge -/
def trip_length (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let segments := (total_charge - initial_fee) / charge_per_segment
  segments * segment_length

theorem taxi_trip_length :
  let initial_fee : ℚ := 225/100
  let charge_per_segment : ℚ := 35/100
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 54/10
  trip_length initial_fee charge_per_segment segment_length total_charge = 36/10 := by
  sorry

end taxi_trip_length_l1754_175477


namespace union_P_complement_Q_l1754_175449

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_R_Q : Set ℝ := {x | ¬(x ∈ Q)}

-- State the theorem
theorem union_P_complement_Q : P ∪ C_R_Q = Ioc (-2) 3 := by sorry

end union_P_complement_Q_l1754_175449


namespace geometric_sequence_properties_l1754_175403

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℝ := sorry

/-- Represents the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common difference when inserting n numbers between a_n and a_{n+1} -/
def d (n : ℕ) : ℝ := sorry

/-- Main theorem encompassing both parts of the problem -/
theorem geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * S n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (¬ ∃ m k p : ℕ,
    m < k ∧ k < p ∧
    (k - m = p - k) ∧
    (d m * d p = d k * d k)) :=
sorry

end geometric_sequence_properties_l1754_175403


namespace volunteer_distribution_theorem_l1754_175468

/-- The number of ways to distribute volunteers among exits -/
def distribute_volunteers (num_volunteers : ℕ) (num_exits : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements -/
theorem volunteer_distribution_theorem :
  distribute_volunteers 5 4 = 240 :=
sorry

end volunteer_distribution_theorem_l1754_175468


namespace faye_candy_count_l1754_175480

/-- Calculates the remaining candy count after a given number of days -/
def remaining_candy (initial : ℕ) (daily_consumption : ℕ) (daily_addition : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_addition - days * daily_consumption

/-- Theorem: Faye's remaining candy count after y days -/
theorem faye_candy_count :
  ∀ (x y z : ℕ), remaining_candy 47 x z y = 47 + y * z - y * x :=
by
  sorry

#check faye_candy_count

end faye_candy_count_l1754_175480


namespace length_of_CD_length_of_CD_explicit_l1754_175421

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    this theorem proves the length of CD. -/
theorem length_of_CD (a : ℝ) (h : a ≥ Real.sqrt 7) : ℝ :=
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (a^2 + 9)
  let BD : ℝ := Real.sqrt (a^2 - 7)
  |AD - BD|

/-- The length of CD is |4 - √(a² - 7)| -/
theorem length_of_CD_explicit (a : ℝ) (h : a ≥ Real.sqrt 7) :
  length_of_CD a h = |4 - Real.sqrt (a^2 - 7)| :=
sorry

end length_of_CD_length_of_CD_explicit_l1754_175421


namespace hyperbola_intersection_x_coordinate_l1754_175418

theorem hyperbola_intersection_x_coordinate :
  ∀ x y : ℝ,
  (Real.sqrt ((x - 5)^2 + y^2) - Real.sqrt ((x + 5)^2 + y^2) = 6) →
  (y = 4) →
  (x = -3 * Real.sqrt 2) :=
by sorry

end hyperbola_intersection_x_coordinate_l1754_175418


namespace students_playing_soccer_l1754_175455

theorem students_playing_soccer 
  (total_students : ℕ) 
  (boy_students : ℕ) 
  (girls_not_playing : ℕ) 
  (soccer_boys_percentage : ℚ) :
  total_students = 420 →
  boy_students = 320 →
  girls_not_playing = 65 →
  soccer_boys_percentage = 86/100 →
  ∃ (students_playing_soccer : ℕ), 
    students_playing_soccer = 250 ∧
    (total_students - boy_students - girls_not_playing : ℚ) = 
      (1 - soccer_boys_percentage) * students_playing_soccer := by
  sorry

end students_playing_soccer_l1754_175455


namespace sixth_term_is_46_l1754_175451

/-- The sequence of small circles in each figure -/
def circleSequence (n : ℕ) : ℕ := n * (n + 1) + 4

/-- The theorem stating that the 6th term of the sequence is 46 -/
theorem sixth_term_is_46 : circleSequence 6 = 46 := by
  sorry

end sixth_term_is_46_l1754_175451


namespace intersection_distance_l1754_175487

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 2.5

-- Define the shared focus
def shared_focus : ℝ × ℝ := (5, 0)

-- Define the directrix of the parabola
def parabola_directrix : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}

-- Theorem statement
theorem intersection_distance :
  ∃ p1 p2 : ℝ × ℝ,
    hyperbola p1.1 p1.2 ∧
    hyperbola p2.1 p2.2 ∧
    parabola p1.1 p1.2 ∧
    parabola p2.1 p2.2 ∧
    p1 ≠ p2 ∧
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 4 * Real.sqrt 218 / 15 :=
sorry

end intersection_distance_l1754_175487


namespace smallest_n_divisibility_l1754_175444

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (¬(24 ∣ m^2) ∨ ¬(900 ∣ m^3) ∨ ¬(1024 ∣ m^4))) ∧
  24 ∣ n^2 ∧ 900 ∣ n^3 ∧ 1024 ∣ n^4 ∧ n = 120 := by
  sorry

end smallest_n_divisibility_l1754_175444


namespace p_sufficient_not_necessary_for_q_l1754_175447

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 1 < x ∧ x < 2 → Real.log x < 1) ∧
  (∃ x : ℝ, Real.log x < 1 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end p_sufficient_not_necessary_for_q_l1754_175447


namespace road_width_calculation_l1754_175471

/-- Given a rectangular lawn with two roads running through the middle,
    calculate the width of each road based on the cost of traveling. -/
theorem road_width_calculation (lawn_length lawn_width total_cost cost_per_sqm : ℝ)
    (h1 : lawn_length = 80)
    (h2 : lawn_width = 60)
    (h3 : total_cost = 5625)
    (h4 : cost_per_sqm = 3)
    (h5 : total_cost = (lawn_length + lawn_width) * road_width * cost_per_sqm) :
    road_width = total_cost / (cost_per_sqm * (lawn_length + lawn_width)) :=
by sorry

end road_width_calculation_l1754_175471


namespace total_marbles_is_72_l1754_175426

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of yellow:blue:green marbles -/
def marbleRatio : MarbleCounts := ⟨2, 3, 4⟩

/-- The actual number of green marbles in the bag -/
def greenMarbleCount : ℕ := 32

/-- Calculate the total number of marbles in the bag -/
def totalMarbles (mc : MarbleCounts) : ℕ :=
  mc.yellow + mc.blue + mc.green

/-- Theorem stating that the total number of marbles is 72 -/
theorem total_marbles_is_72 :
  ∃ (factor : ℕ), 
    factor * marbleRatio.green = greenMarbleCount ∧
    totalMarbles (MarbleCounts.mk 
      (factor * marbleRatio.yellow)
      (factor * marbleRatio.blue)
      greenMarbleCount) = 72 := by
  sorry

end total_marbles_is_72_l1754_175426


namespace coupon_probability_l1754_175402

theorem coupon_probability (n m k : ℕ) (h1 : n = 17) (h2 : m = 9) (h3 : k = 6) : 
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end coupon_probability_l1754_175402


namespace nearest_integer_to_sum_l1754_175448

theorem nearest_integer_to_sum (x y : ℝ) 
  (h1 : abs x - y = 5)
  (h2 : abs x * y - x^2 = -12) : 
  round (x + y) = -5 := by
  sorry

end nearest_integer_to_sum_l1754_175448


namespace age_sum_is_23_l1754_175411

/-- The ages of Al, Bob, and Carl satisfy the given conditions and their sum is 23 -/
theorem age_sum_is_23 (a b c : ℕ) : 
  a = 10 * b * c ∧ 
  a^3 = 8000 + 8 * b^3 * c^3 → 
  a + b + c = 23 := by
sorry

end age_sum_is_23_l1754_175411


namespace smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l1754_175413

def N : ℕ := 2329089562800

theorem smallest_valid_integer (k : ℕ) (h : k < N) : 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

theorem N_divisible_by_1_to_28 : 
  ∀ i ∈ Finset.range 28, N % (i + 1) = 0 :=
sorry

theorem N_not_divisible_by_29_or_30 : 
  N % 29 ≠ 0 ∨ N % 30 ≠ 0 :=
sorry

theorem N_is_smallest_valid_integer : 
  ∀ k < N, 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

end smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l1754_175413


namespace sigma_inequality_l1754_175420

/-- Sum of positive divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Theorem: If σ(n) > 2n, then σ(mn) > 2mn for any m -/
theorem sigma_inequality (n : ℕ+) (h : sigma n > 2 * n) :
  ∀ m : ℕ+, sigma (m * n) > 2 * m * n := by
  sorry

end sigma_inequality_l1754_175420


namespace smallest_fd_minus_de_is_eight_l1754_175499

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given triangle satisfies the triangle inequality -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- The main theorem stating the smallest difference between FD and DE -/
theorem smallest_fd_minus_de_is_eight :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 3009 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    satisfies_triangle_inequality t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 3009 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      satisfies_triangle_inequality t' →
      t'.fd - t'.de ≥ t.fd - t.de) →
    t.fd - t.de = 8 := by
  sorry

#check smallest_fd_minus_de_is_eight

end smallest_fd_minus_de_is_eight_l1754_175499


namespace fourth_term_coefficient_l1754_175469

theorem fourth_term_coefficient (a : ℝ) : 
  (Nat.choose 6 3) * a^3 * (-1)^3 = 160 → a = -2 := by
  sorry

end fourth_term_coefficient_l1754_175469


namespace equal_area_if_equal_midpoints_l1754_175466

/-- A polygon with an even number of sides -/
structure EvenPolygon where
  vertices : List (ℝ × ℝ)
  even_sides : Even vertices.length

/-- The midpoints of the sides of a polygon -/
def midpoints (p : EvenPolygon) : List (ℝ × ℝ) :=
  sorry

/-- The area of a polygon -/
def area (p : EvenPolygon) : ℝ :=
  sorry

/-- Theorem: If two even-sided polygons have the same midpoints, their areas are equal -/
theorem equal_area_if_equal_midpoints (p q : EvenPolygon) 
  (h : midpoints p = midpoints q) : area p = area q :=
  sorry

end equal_area_if_equal_midpoints_l1754_175466


namespace curve_C_left_of_x_equals_2_l1754_175419

/-- The curve C is defined by the equation x³ + 2y² = 8 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^3 + 2 * p.2^2 = 8}

/-- Theorem: All points on curve C have x-coordinate less than or equal to 2 -/
theorem curve_C_left_of_x_equals_2 : ∀ p ∈ C, p.1 ≤ 2 := by
  sorry

end curve_C_left_of_x_equals_2_l1754_175419


namespace complex_product_l1754_175445

theorem complex_product (z₁ z₂ : ℂ) :
  Complex.abs z₁ = 1 →
  Complex.abs z₂ = 1 →
  z₁ + z₂ = (-7/5 : ℂ) + (1/5 : ℂ) * Complex.I →
  z₁ * z₂ = (24/25 : ℂ) - (7/25 : ℂ) * Complex.I :=
by
  sorry

end complex_product_l1754_175445


namespace problem_solution_l1754_175493

theorem problem_solution : ∃ x : ℚ, (x + x/4 = 80 * 3/4) ∧ (x = 48) := by
  sorry

end problem_solution_l1754_175493


namespace principal_amount_l1754_175417

-- Define the interest rate and time
def r : ℝ := 0.07
def t : ℝ := 2

-- Define the difference between C.I. and S.I.
def difference : ℝ := 49

-- State the theorem
theorem principal_amount (P : ℝ) :
  P * ((1 + r)^t - 1 - t * r) = difference → P = 10000 := by
  sorry

end principal_amount_l1754_175417


namespace radio_loss_percentage_l1754_175429

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for the given cost and selling prices is 17% -/
theorem radio_loss_percentage : 
  loss_percentage 1500 1245 = 17 := by sorry

end radio_loss_percentage_l1754_175429


namespace box_dimensions_l1754_175404

def is_valid_box (a b : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ b ∧ a^2 * b = a^2 + 4*a*b

theorem box_dimensions :
  ∀ a b : ℕ, is_valid_box a b ↔ (a = 8 ∧ b = 2) ∨ (a = 5 ∧ b = 5) :=
sorry

end box_dimensions_l1754_175404


namespace gcd_of_factorials_l1754_175484

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem gcd_of_factorials : Nat.gcd (factorial 8) (Nat.gcd (factorial 10) (factorial 11)) = factorial 8 := by
  sorry

end gcd_of_factorials_l1754_175484


namespace equilateral_triangle_side_length_l1754_175416

/-- An equilateral triangle with a point inside --/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle --/
  side_length : ℝ
  /-- The perpendicular distance from the point to the first side --/
  dist1 : ℝ
  /-- The perpendicular distance from the point to the second side --/
  dist2 : ℝ
  /-- The perpendicular distance from the point to the third side --/
  dist3 : ℝ
  /-- The side length is positive --/
  side_positive : side_length > 0
  /-- All distances are positive --/
  dist_positive : dist1 > 0 ∧ dist2 > 0 ∧ dist3 > 0

/-- The theorem stating the relationship between the side length and the perpendicular distances --/
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.dist1 = 2) 
  (h2 : t.dist2 = 3) 
  (h3 : t.dist3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_side_length_l1754_175416


namespace overlap_area_l1754_175422

theorem overlap_area (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) :
  total_length = left_length + right_length →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end overlap_area_l1754_175422


namespace largest_multiple_of_9_under_100_l1754_175405

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n * 9 = 99 ∧ 
  99 < 100 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
sorry

end largest_multiple_of_9_under_100_l1754_175405


namespace lawn_mowing_problem_l1754_175452

theorem lawn_mowing_problem (mary_rate tom_rate : ℚ) (tom_work_time : ℚ) 
  (h1 : mary_rate = 1 / 4)
  (h2 : tom_rate = 1 / 5)
  (h3 : tom_work_time = 2) :
  1 - tom_rate * tom_work_time = 3 / 5 := by
  sorry

end lawn_mowing_problem_l1754_175452


namespace root_product_sum_l1754_175434

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 4046 * x₁ - 1 = 0) ∧
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 4046 * x₂ - 1 = 0) ∧
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 4046 * x₃ - 1 = 0) →
  x₂ * (x₁ + x₃) = 2 + 1 / 2023 := by
sorry

end root_product_sum_l1754_175434


namespace sum_ac_equals_eight_l1754_175415

theorem sum_ac_equals_eight 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end sum_ac_equals_eight_l1754_175415


namespace min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l1754_175494

-- Part 1
theorem min_value_of_f (a : ℝ) (ha : a > 0) :
  a^2 + 2/a ≥ 3 :=
sorry

theorem min_value_of_f_equality (a : ℝ) (ha : a > 0) :
  a^2 + 2/a = 3 ↔ a = 1 :=
sorry

-- Part 2
def m_range (m : ℝ) : Prop :=
  m ≤ -3 ∨ m ≥ -1

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ a : ℝ, a > 0 → a^3 + 2 ≥ 3*a*(|m - 1| - |2*m + 3|)) ↔ m_range m :=
sorry

end min_value_of_f_min_value_of_f_equality_inequality_holds_iff_m_in_range_l1754_175494


namespace problem_solution_l1754_175424

theorem problem_solution (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 1) : 
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 := by
  sorry

end problem_solution_l1754_175424


namespace angle_measure_proof_l1754_175470

theorem angle_measure_proof (x : ℝ) : 
  (90 - x = 3 * x - 10) → x = 25 := by
  sorry

end angle_measure_proof_l1754_175470


namespace cylinder_volume_equality_l1754_175442

/-- Proves that for two cylinders with initial radius 5 inches and height 4 inches, 
    if the radius of one and the height of the other are increased by y inches, 
    and their volumes become equal, then y = 5/4 inches. -/
theorem cylinder_volume_equality (y : ℚ) : 
  y ≠ 0 → 
  π * (5 + y)^2 * 4 = π * 5^2 * (4 + y) → 
  y = 5/4 := by sorry

end cylinder_volume_equality_l1754_175442


namespace common_root_of_polynomials_l1754_175489

theorem common_root_of_polynomials :
  let p₁ (x : ℚ) := 3*x^4 + 13*x^3 + 20*x^2 + 17*x + 7
  let p₂ (x : ℚ) := 3*x^4 + x^3 - 8*x^2 + 11*x - 7
  p₁ (-7/3) = 0 ∧ p₂ (-7/3) = 0 :=
by sorry

end common_root_of_polynomials_l1754_175489


namespace rotation_equivalence_l1754_175450

/-- 
Given a point P rotated about a center Q:
1. 510 degrees clockwise rotation reaches point R
2. y degrees counterclockwise rotation also reaches point R
3. y < 360

Prove that y = 210
-/
theorem rotation_equivalence (y : ℝ) 
  (h1 : y < 360)
  (h2 : (510 % 360 : ℝ) = (360 - y) % 360) : y = 210 := by
  sorry

end rotation_equivalence_l1754_175450


namespace number_problem_l1754_175482

theorem number_problem (x : ℚ) : (54/2 : ℚ) + 3 * x = 75 → x = 16 := by
  sorry

end number_problem_l1754_175482


namespace arithmetic_sequence_sum_l1754_175427

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) := by
sorry

end arithmetic_sequence_sum_l1754_175427


namespace consecutive_integers_with_square_factors_l1754_175443

theorem consecutive_integers_with_square_factors (n : ℕ) :
  ∃ x : ℤ, ∀ k : ℕ, k ≥ 1 → k ≤ n →
    ∃ m : ℕ, m > 1 ∧ ∃ y : ℤ, x + k = m^2 * y := by
  sorry

end consecutive_integers_with_square_factors_l1754_175443


namespace graph_number_example_intersection_condition_l1754_175498

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a function to check if a quadratic function intersects x-axis at one point
def intersectsAtOnePoint (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

-- Theorem 1: The graph number of y = (1/3)x^2 - x - 1
theorem graph_number_example : getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) := by
  sorry

-- Theorem 2: For [m, m+1, m+1] intersecting x-axis at one point, m = -1 or m = 1/3
theorem intersection_condition (m : ℝ) :
  intersectsAtOnePoint m (m+1) (m+1) → m = -1 ∨ m = 1/3 := by
  sorry

end graph_number_example_intersection_condition_l1754_175498


namespace trig_identity_l1754_175479

/-- Proves that sin 69° cos 9° - sin 21° cos 81° = √3/2 -/
theorem trig_identity : Real.sin (69 * π / 180) * Real.cos (9 * π / 180) - 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end trig_identity_l1754_175479


namespace weight_calculation_l1754_175430

/-- Given a box containing 16 equal weights with a total weight of 17.88 kg,
    and an empty box weighing 0.6 kg, the weight of 7 such weights is 7.56 kg. -/
theorem weight_calculation (total_weight : ℝ) (box_weight : ℝ) (num_weights : ℕ) (target_weights : ℕ)
    (hw : total_weight = 17.88)
    (hb : box_weight = 0.6)
    (hn : num_weights = 16)
    (ht : target_weights = 7) :
    (total_weight - box_weight) / num_weights * target_weights = 7.56 := by
  sorry

end weight_calculation_l1754_175430


namespace not_always_possible_to_equalize_l1754_175460

/-- Represents a board with integers -/
def Board := Matrix (Fin 2018) (Fin 2019) Int

/-- Checks if two positions are neighbors on the board -/
def is_neighbor (i j i' j' : Nat) : Prop :=
  (i = i' ∧ (j = j' + 1 ∨ j + 1 = j')) ∨ 
  (j = j' ∧ (i = i' + 1 ∨ i + 1 = i'))

/-- Represents a single turn of the averaging operation -/
def average_turn (b : Board) (chosen : Set (Fin 2018 × Fin 2019)) : Board :=
  sorry

/-- Represents a sequence of turns -/
def sequence_of_turns (b : Board) (turns : Nat) : Board :=
  sorry

/-- Checks if all numbers on the board are the same -/
def all_same (b : Board) : Prop :=
  ∀ i j i' j', b i j = b i' j'

theorem not_always_possible_to_equalize : ∃ (initial : Board), 
  ∀ (turns : Nat), ¬(all_same (sequence_of_turns initial turns)) :=
sorry

end not_always_possible_to_equalize_l1754_175460


namespace majority_can_play_and_ride_l1754_175481

/-- Represents a person's location and height -/
structure Person where
  location : ℝ × ℝ
  height : ℝ

/-- The population of the country -/
def Population := List Person

/-- Checks if a person is taller than the majority within a given radius -/
def isTallerThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person is shorter than the majority within a given radius -/
def isShorterThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person can play basketball (i.e., can choose a radius to be taller than majority) -/
def canPlayBasketball (p : Person) (pop : Population) : Bool :=
  sorry

/-- Checks if a person is entitled to free transportation (i.e., can choose a radius to be shorter than majority) -/
def hasFreeTrans (p : Person) (pop : Population) : Bool :=
  sorry

/-- Calculates the percentage of people satisfying a given condition -/
def percentageSatisfying (pop : Population) (condition : Person → Population → Bool) : ℝ :=
  sorry

theorem majority_can_play_and_ride (pop : Population) :
  percentageSatisfying pop canPlayBasketball ≥ 90 ∧
  percentageSatisfying pop hasFreeTrans ≥ 90 :=
sorry

end majority_can_play_and_ride_l1754_175481


namespace prob_red_or_black_prob_not_green_l1754_175456

/-- Represents the colors of balls in the box -/
inductive Color
  | Red
  | Black
  | White
  | Green

/-- The total number of balls in the box -/
def totalBalls : ℕ := 12

/-- The number of balls of each color -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 5
  | Color.Black => 4
  | Color.White => 2
  | Color.Green => 1

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing either a red or black ball is 3/4 -/
theorem prob_red_or_black :
  probability Color.Red + probability Color.Black = 3/4 := by sorry

/-- Theorem: The probability of drawing a ball that is not green is 11/12 -/
theorem prob_not_green :
  1 - probability Color.Green = 11/12 := by sorry

end prob_red_or_black_prob_not_green_l1754_175456


namespace tan_sum_three_angles_l1754_175473

theorem tan_sum_three_angles (α β γ : ℝ) : 
  Real.tan (α + β + γ) = (Real.tan α + Real.tan β + Real.tan γ - Real.tan α * Real.tan β * Real.tan γ) / 
                         (1 - Real.tan α * Real.tan β - Real.tan β * Real.tan γ - Real.tan γ * Real.tan α) :=
by sorry

end tan_sum_three_angles_l1754_175473


namespace crude_oil_temperature_l1754_175412

-- Define the function f(x) = x^2 - 7x + 15
def f (x : ℝ) : ℝ := x^2 - 7*x + 15

-- Define the domain of f
def domain : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

theorem crude_oil_temperature (x : ℝ) (h : x ∈ domain) :
  -- The derivative of f at x = 4 is 1
  deriv f 4 = 1 ∧
  -- The function is increasing at x = 4
  0 < deriv f 4 := by
  sorry

end crude_oil_temperature_l1754_175412


namespace certain_number_exists_l1754_175476

theorem certain_number_exists : ∃ x : ℝ, 0.35 * x - (1/3) * (0.35 * x) = 42 := by
  sorry

end certain_number_exists_l1754_175476


namespace system_solution_l1754_175461

theorem system_solution (x y k : ℝ) : 
  x + y - 5 * k = 0 → 
  x - y - 9 * k = 0 → 
  2 * x + 3 * y = 6 → 
  4 * k - 1 = 2 := by
sorry

end system_solution_l1754_175461


namespace function_properties_l1754_175465

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3

theorem function_properties :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f x ≤ f (Real.exp 1)) ∧
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f 1 ≤ f x) ∧
  (∀ x ≥ 1, f x ≤ g x) := by
  sorry

end function_properties_l1754_175465


namespace simplify_complex_expression_l1754_175410

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end simplify_complex_expression_l1754_175410


namespace circle_condition_l1754_175472

/-- 
Given a real number a and the equation ax^2 + ay^2 - 4(a-1)x + 4y = 0,
this theorem states that the equation represents a circle if and only if a ≠ 0.
-/
theorem circle_condition (a : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ 
    ∀ (x y : ℝ), ax^2 + ay^2 - 4*(a-1)*x + 4*y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  a ≠ 0 :=
sorry

end circle_condition_l1754_175472


namespace desk_lamp_profit_maximization_l1754_175414

/-- Represents the profit function for desk lamp sales -/
def profit_function (original_price cost_price initial_sales price_increase : ℝ) : ℝ → ℝ :=
  λ x => (original_price + x - cost_price) * (initial_sales - 10 * x)

theorem desk_lamp_profit_maximization 
  (original_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_range_min : ℝ) 
  (price_range_max : ℝ) 
  (h1 : original_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : price_range_min = 40)
  (h5 : price_range_max = 60)
  (h6 : price_range_min ≤ price_range_max) :
  (∃ x : ℝ, x = 10 ∧ profit_function original_price cost_price initial_sales x = 10000) ∧
  (∀ y : ℝ, price_range_min ≤ original_price + y ∧ original_price + y ≤ price_range_max →
    profit_function original_price cost_price initial_sales y ≤ 
    profit_function original_price cost_price initial_sales (price_range_max - original_price)) :=
by sorry

end desk_lamp_profit_maximization_l1754_175414


namespace regular_triangular_pyramid_volume_l1754_175462

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) -- lateral edge length
  (α : ℝ) -- angle between lateral edge and base plane
  (h1 : l > 0) -- lateral edge length is positive
  (h2 : 0 < α ∧ α < π/2) -- angle is between 0 and π/2
  : ∃ (V : ℝ), V = (Real.sqrt 3 * l^3 * Real.cos α^2 * Real.sin α) / 4 :=
by
  sorry

end regular_triangular_pyramid_volume_l1754_175462


namespace initial_money_correct_l1754_175478

/-- The amount of money Little John had initially -/
def initial_money : ℚ := 10.50

/-- The amount Little John spent on sweets -/
def sweets_cost : ℚ := 2.25

/-- The amount Little John gave to each friend -/
def money_per_friend : ℚ := 2.20

/-- The number of friends Little John gave money to -/
def number_of_friends : ℕ := 2

/-- The amount of money Little John had left -/
def money_left : ℚ := 3.85

/-- Theorem stating that the initial amount of money is correct given the conditions -/
theorem initial_money_correct : 
  initial_money = sweets_cost + (money_per_friend * number_of_friends) + money_left :=
by sorry

end initial_money_correct_l1754_175478


namespace inequality_solution_l1754_175486

theorem inequality_solution (x : ℝ) : (x - 5) / ((x - 2) * (x^2 - 1)) < 0 ↔ x < -1 ∨ (1 < x ∧ x < 5) := by
  sorry

end inequality_solution_l1754_175486


namespace common_factor_is_gcf_l1754_175400

noncomputable def p1 (x y z : ℝ) : ℝ := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
noncomputable def p2 (x y z : ℝ) : ℝ := 6 * x^4 * y * z^2
noncomputable def common_factor (x y z : ℝ) : ℝ := 3 * x^2 * y * z

theorem common_factor_is_gcf :
  ∀ x y z : ℝ,
  (∃ k1 k2 : ℝ, p1 x y z = common_factor x y z * k1 ∧ p2 x y z = common_factor x y z * k2) ∧
  (∀ f : ℝ → ℝ → ℝ → ℝ, (∃ l1 l2 : ℝ, p1 x y z = f x y z * l1 ∧ p2 x y z = f x y z * l2) →
    ∃ m : ℝ, f x y z = common_factor x y z * m) :=
by sorry

end common_factor_is_gcf_l1754_175400


namespace inequality_solution_set_l1754_175463

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end inequality_solution_set_l1754_175463


namespace tenth_term_of_arithmetic_sequence_l1754_175436

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_arithmetic_sequence 
  (a₁ d : ℝ) 
  (h₁ : arithmetic_sequence a₁ d 3 = 10) 
  (h₂ : arithmetic_sequence a₁ d 8 = 30) : 
  arithmetic_sequence a₁ d 10 = 38 := by
  sorry

end tenth_term_of_arithmetic_sequence_l1754_175436


namespace quadratic_roots_l1754_175458

theorem quadratic_roots (d : ℚ) : 
  (∀ x : ℚ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt (2*d))/2 ∨ x = (-7 - Real.sqrt (2*d))/2) → 
  d = 49/6 := by
sorry

end quadratic_roots_l1754_175458


namespace quadratic_sum_equations_l1754_175441

/-- Given two quadratic equations and their roots, prove the equations for the sums of roots -/
theorem quadratic_sum_equations 
  (a b c α β γ : ℝ) 
  (h1 : a * α ≠ 0) 
  (h2 : b^2 - 4*a*c ≥ 0) 
  (h3 : β^2 - 4*α*γ ≥ 0) 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (hx : x₁ ≤ x₂) 
  (hy : y₁ ≤ y₂) 
  (hx_roots : a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) 
  (hy_roots : α * y₁^2 + β * y₁ + γ = 0 ∧ α * y₂^2 + β * y₂ + γ = 0) :
  ∃ (d δ : ℝ), 
    d^2 = b^2 - 4*a*c ∧ 
    δ^2 = β^2 - 4*α*γ ∧
    (∀ z, 2*a*α*z^2 + 2*(a*β + α*b)*z + (2*a*γ + 2*α*c + b*β - d*δ) = 0 ↔ 
      (z = x₁ + y₁ ∨ z = x₂ + y₂)) ∧
    (∀ u, 2*a*α*u^2 + 2*(a*β + α*b)*u + (2*a*γ + 2*α*c + b*β + d*δ) = 0 ↔ 
      (u = x₁ + y₂ ∨ u = x₂ + y₁)) :=
sorry

end quadratic_sum_equations_l1754_175441


namespace parabola_hyperbola_shared_focus_l1754_175474

/-- The value of p for which the focus of the parabola y^2 = 2px (p > 0) 
    is also a focus of the hyperbola x^2 - y^2 = 8 -/
theorem parabola_hyperbola_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 8 ∧ 
    ((x - p)^2 + y^2 = p^2 ∨ (x + p)^2 + y^2 = p^2)) → 
  p = 8 := by
sorry

end parabola_hyperbola_shared_focus_l1754_175474


namespace cube_root_equation_solution_l1754_175408

theorem cube_root_equation_solution :
  ∃! x : ℝ, (3 - x)^(1/3 : ℝ) = -(3/2) ∧ x = 51/8 := by
  sorry

end cube_root_equation_solution_l1754_175408


namespace range_of_a_l1754_175491

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(q x) → ¬(p x a)) ∧ 
  (∃ x, ¬(q x) ∧ p x a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a → (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end range_of_a_l1754_175491


namespace binomial_coefficient_9_5_l1754_175439

theorem binomial_coefficient_9_5 : Nat.choose 9 5 = 126 := by sorry

end binomial_coefficient_9_5_l1754_175439


namespace coefficient_of_inverse_x_l1754_175457

theorem coefficient_of_inverse_x (x : ℝ) : 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = c / x + (terms_without_inverse_x : ℝ)) → 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = -20 / x + (terms_without_inverse_x : ℝ)) :=
by sorry

end coefficient_of_inverse_x_l1754_175457


namespace binary_10011_equals_19_l1754_175438

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 10011₂ -/
def binary_10011 : List Bool := [true, true, false, false, true]

/-- Theorem stating that 10011₂ is equal to 19 in decimal -/
theorem binary_10011_equals_19 : binary_to_decimal binary_10011 = 19 := by
  sorry

end binary_10011_equals_19_l1754_175438


namespace sqrt_172_01_l1754_175464

theorem sqrt_172_01 (h1 : Real.sqrt 1.7201 = 1.311) (h2 : Real.sqrt 17.201 = 4.147) :
  Real.sqrt 172.01 = 13.11 ∨ Real.sqrt 172.01 = -13.11 :=
by sorry

end sqrt_172_01_l1754_175464


namespace gcd_lcm_sum_divisibility_l1754_175495

theorem gcd_lcm_sum_divisibility (a b : ℕ) (h : a > 0 ∧ b > 0) :
  Nat.gcd a b + Nat.lcm a b = a + b → a ∣ b ∨ b ∣ a := by
  sorry

end gcd_lcm_sum_divisibility_l1754_175495


namespace prove_late_time_l1754_175459

def late_time_problem (charlize_late : ℕ) (classmates_extra : ℕ) (num_classmates : ℕ) : Prop :=
  let classmate_late := charlize_late + classmates_extra
  let total_classmates_late := num_classmates * classmate_late
  let total_late := total_classmates_late + charlize_late
  total_late = 140

theorem prove_late_time : late_time_problem 20 10 4 := by
  sorry

end prove_late_time_l1754_175459


namespace students_walking_home_l1754_175431

theorem students_walking_home (bus auto bike scooter : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/6)
  (h_scooter : scooter = 1/10)
  (h_total : bus + auto + bike + scooter < 1) :
  1 - (bus + auto + bike + scooter) = 1/5 := by
  sorry

end students_walking_home_l1754_175431


namespace range_of_a_l1754_175488

-- Define the functions f and g
def f (x : ℝ) := 3 * abs (x - 1) + abs (3 * x + 1)
def g (a : ℝ) (x : ℝ) := abs (x + 2) + abs (x - a)

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, f x = y}
def B (a : ℝ) : Set ℝ := {y | ∃ x, g a x = y}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = B a) → (a ∈ Set.Icc (-6) 2) :=
by sorry

end range_of_a_l1754_175488


namespace expression_simplification_l1754_175485

theorem expression_simplification (x : ℝ) : 
  3 * x + 10 * x^2 - 7 - (1 + 5 * x - 10 * x^2) = 20 * x^2 - 2 * x - 8 := by
  sorry

end expression_simplification_l1754_175485


namespace single_point_conic_section_l1754_175428

/-- If the graph of 3x^2 + y^2 + 6x - 6y + d = 0 consists of a single point, then d = 12 -/
theorem single_point_conic_section (d : ℝ) :
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) →
  d = 12 := by
  sorry

end single_point_conic_section_l1754_175428


namespace simplify_trig_expression_l1754_175490

theorem simplify_trig_expression (x : ℝ) (h : 5 * Real.pi / 2 < x ∧ x < 3 * Real.pi) :
  Real.sqrt ((1 - Real.sin (3 * Real.pi / 2 - x)) / 2) = -Real.cos (x / 2) := by
  sorry

end simplify_trig_expression_l1754_175490


namespace vector_magnitude_proof_l1754_175409

/-- Given vectors a and b, prove that the magnitude of a - 2b is √3 -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a.1 = Real.cos (15 * π / 180) ∧
  a.2 = Real.sin (15 * π / 180) ∧
  b.1 = Real.cos (75 * π / 180) ∧
  b.2 = Real.sin (75 * π / 180) →
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 := by
  sorry

end vector_magnitude_proof_l1754_175409


namespace pentagon_perimeter_is_49_l1754_175475

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (x y z : ℝ) : ℝ :=
  3*x + 5*y + 6*z + 4*x + 7*y

/-- Theorem: The perimeter of the specified pentagon is 49 cm -/
theorem pentagon_perimeter_is_49 :
  pentagon_perimeter 1 2 3 = 49 := by
  sorry

end pentagon_perimeter_is_49_l1754_175475


namespace green_ducks_percentage_in_larger_pond_l1754_175432

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 15

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 20
  let larger_pond_ducks : ℕ := 80
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 16
  larger_pond_green_percentage = 
    (total_green_percentage * (smaller_pond_ducks + larger_pond_ducks) - 
     smaller_pond_green_percentage * smaller_pond_ducks) / larger_pond_ducks := by
  sorry

end green_ducks_percentage_in_larger_pond_l1754_175432
