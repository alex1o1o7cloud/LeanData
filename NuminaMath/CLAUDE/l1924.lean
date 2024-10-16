import Mathlib

namespace NUMINAMATH_CALUDE_candidate_a_votes_l1924_192439

/-- Proves that given a ratio of 2:1 for votes between two candidates and a total of 21 votes,
    the candidate with the higher number of votes received 14 votes. -/
theorem candidate_a_votes (total_votes : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : 
  total_votes = 21 → ratio_a = 2 → ratio_b = 1 → 
  (ratio_a * total_votes) / (ratio_a + ratio_b) = 14 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l1924_192439


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l1924_192482

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l1924_192482


namespace NUMINAMATH_CALUDE_sum_of_ninth_powers_of_roots_l1924_192478

theorem sum_of_ninth_powers_of_roots (u v w : ℂ) : 
  (u^3 - 3*u - 1 = 0) → 
  (v^3 - 3*v - 1 = 0) → 
  (w^3 - 3*w - 1 = 0) → 
  u^9 + v^9 + w^9 = 246 := by sorry

end NUMINAMATH_CALUDE_sum_of_ninth_powers_of_roots_l1924_192478


namespace NUMINAMATH_CALUDE_equidistant_points_bound_l1924_192483

/-- A set of points in a plane where no three points are collinear -/
structure PointSet where
  S : Set (ℝ × ℝ)
  noncollinear : ∀ (p q r : ℝ × ℝ), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r →
    (p.1 - q.1) * (r.2 - q.2) ≠ (r.1 - q.1) * (p.2 - q.2)

/-- The property that for each point, there are k equidistant points -/
def has_k_equidistant (PS : PointSet) (k : ℕ) : Prop :=
  ∀ p ∈ PS.S, ∃ (T : Set (ℝ × ℝ)), T ⊆ PS.S ∧ T.ncard = k ∧
    ∀ q ∈ T, q ≠ p → ∃ d : ℝ, d > 0 ∧ (p.1 - q.1)^2 + (p.2 - q.2)^2 = d^2

theorem equidistant_points_bound (n k : ℕ) (h_pos : 0 < n ∧ 0 < k) (PS : PointSet)
    (h_card : PS.S.ncard = n) (h_equi : has_k_equidistant PS k) :
    k ≤ (1 : ℝ)/2 + Real.sqrt (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_points_bound_l1924_192483


namespace NUMINAMATH_CALUDE_finishing_order_equals_starting_order_l1924_192429

/-- Represents an athlete in the race -/
inductive Athlete : Type
  | Grisha : Athlete
  | Sasha : Athlete
  | Lena : Athlete

/-- Represents the order of athletes -/
def AthleteOrder := List Athlete

/-- The starting order of the race -/
def startingOrder : AthleteOrder := [Athlete.Grisha, Athlete.Sasha, Athlete.Lena]

/-- The number of overtakes by each athlete -/
def overtakes : Athlete → Nat
  | Athlete.Grisha => 10
  | Athlete.Sasha => 4
  | Athlete.Lena => 6

/-- No three athletes were at the same position simultaneously -/
axiom no_triple_overtake : True

/-- All athletes finished at different times -/
axiom different_finish_times : True

/-- The finishing order of the race -/
def finishingOrder : AthleteOrder := sorry

/-- Theorem stating that the finishing order is the same as the starting order -/
theorem finishing_order_equals_starting_order : 
  finishingOrder = startingOrder := by sorry

end NUMINAMATH_CALUDE_finishing_order_equals_starting_order_l1924_192429


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l1924_192485

/-- Proves that if the difference between compound interest and simple interest 
    on a sum at 10% per annum for 2 years is Rs. 61, then the sum (principal) is Rs. 6100. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 61 → P = 6100 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l1924_192485


namespace NUMINAMATH_CALUDE_trajectory_equation_l1924_192416

/-- Given a fixed point A(1, 2) and a moving point P(x, y) in a Cartesian coordinate system,
    if OP · OA = 4, then the equation of the trajectory of P is x + 2y - 4 = 0. -/
theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (1, 2)
  let P : ℝ × ℝ := (x, y)
  let O : ℝ × ℝ := (0, 0)
  (P.1 - O.1) * (A.1 - O.1) + (P.2 - O.2) * (A.2 - O.2) = 4 →
  x + 2 * y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1924_192416


namespace NUMINAMATH_CALUDE_triangle_area_l1924_192415

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

theorem triangle_area : 
  let area := (1/2) * |a.1 * b.2 - a.2 * b.1|
  area = 9/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1924_192415


namespace NUMINAMATH_CALUDE_mean_proportional_sum_l1924_192453

/-- Mean proportional of two numbers -/
def mean_proportional (a b c : ℝ) : Prop := a / b = b / c

/-- Find x such that 0.9 : 0.6 = 0.6 : x -/
def find_x : ℝ := 0.4

/-- Find y such that 1/2 : 1/5 = 1/5 : y -/
def find_y : ℝ := 0.08

theorem mean_proportional_sum :
  mean_proportional 0.9 0.6 find_x ∧ 
  mean_proportional (1/2) (1/5) find_y ∧
  find_x + find_y = 0.48 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_sum_l1924_192453


namespace NUMINAMATH_CALUDE_regular_polygon_assembly_l1924_192456

theorem regular_polygon_assembly (interior_angle : ℝ) (h1 : interior_angle = 150) :
  ∃ (n : ℕ) (m : ℕ), n * interior_angle + m * 60 = 360 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_assembly_l1924_192456


namespace NUMINAMATH_CALUDE_additional_planes_needed_l1924_192489

def current_planes : ℕ := 29
def row_size : ℕ := 8

theorem additional_planes_needed :
  (row_size - (current_planes % row_size)) % row_size = 3 := by sorry

end NUMINAMATH_CALUDE_additional_planes_needed_l1924_192489


namespace NUMINAMATH_CALUDE_clock_initial_time_l1924_192405

/-- Represents a time of day with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the total minutes from midnight for a given time -/
def totalMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Represents the properties of the clock in the problem -/
structure Clock where
  initialTime : Time
  gainPerHour : ℕ
  totalGainBy6PM : ℕ

/-- The theorem to be proved -/
theorem clock_initial_time (c : Clock)
  (morning : c.initialTime.hours < 12)
  (gain_rate : c.gainPerHour = 5)
  (total_gain : c.totalGainBy6PM = 35) :
  c.initialTime.hours = 11 ∧ c.initialTime.minutes = 55 := by
  sorry


end NUMINAMATH_CALUDE_clock_initial_time_l1924_192405


namespace NUMINAMATH_CALUDE_edward_rides_l1924_192481

def max_rides (initial_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (initial_tickets - spent_tickets) / tickets_per_ride

theorem edward_rides : max_rides 325 115 13 = 16 := by
  sorry

end NUMINAMATH_CALUDE_edward_rides_l1924_192481


namespace NUMINAMATH_CALUDE_stock_market_investment_l1924_192448

theorem stock_market_investment (x : ℝ) (h : x > 0) : 
  x * (1 + 0.8) * (1 - 0.3) > x := by
  sorry

end NUMINAMATH_CALUDE_stock_market_investment_l1924_192448


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l1924_192499

theorem divisibility_by_twelve (n : Nat) : n < 10 → (516 * 10 + n) % 12 = 0 ↔ n = 0 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l1924_192499


namespace NUMINAMATH_CALUDE_square_perimeter_inequality_l1924_192404

theorem square_perimeter_inequality (t₁ t₂ t₃ t₄ k₁ k₂ k₃ k₄ : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₃ > 0) (h₄ : t₄ > 0)
  (hk₁ : k₁ = 4 * Real.sqrt t₁)
  (hk₂ : k₂ = 4 * Real.sqrt t₂)
  (hk₃ : k₃ = 4 * Real.sqrt t₃)
  (hk₄ : k₄ = 4 * Real.sqrt t₄)
  (ht : t₁ + t₂ + t₃ = t₄) :
  k₁ + k₂ + k₃ ≤ k₄ * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_inequality_l1924_192404


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1924_192460

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, |x| < 0) ↔ (∃ x : ℝ, |x| ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1924_192460


namespace NUMINAMATH_CALUDE_truck_wheels_count_l1924_192433

/-- Calculates the toll for a truck given the number of axles -/
def toll (x : ℕ) : ℚ := 3.50 + 0.50 * (x - 2)

/-- Calculates the total number of wheels on a truck given the number of axles -/
def totalWheels (x : ℕ) : ℕ := 2 + 4 * (x - 1)

theorem truck_wheels_count :
  ∃ (x : ℕ), 
    x > 0 ∧
    toll x = 5 ∧
    totalWheels x = 18 :=
by sorry

end NUMINAMATH_CALUDE_truck_wheels_count_l1924_192433


namespace NUMINAMATH_CALUDE_justice_plants_l1924_192445

theorem justice_plants (ferns palms succulents total_wanted : ℕ) : 
  ferns = 3 → palms = 5 → succulents = 7 → total_wanted = 24 →
  total_wanted - (ferns + palms + succulents) = 9 := by
sorry

end NUMINAMATH_CALUDE_justice_plants_l1924_192445


namespace NUMINAMATH_CALUDE_expression_equality_l1924_192459

theorem expression_equality : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1924_192459


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1924_192414

theorem negation_of_universal_proposition 
  (f : ℝ → ℝ) (m : ℝ) : 
  (¬ ∀ x, f x ≥ m) ↔ ∃ x, f x < m :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1924_192414


namespace NUMINAMATH_CALUDE_inequality_proof_l1924_192400

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1924_192400


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1924_192451

/-- The function f(x) = ax + 3 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 3

/-- The zero point of f(x) is in the interval (-1, 2) --/
def has_zero_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 < x ∧ x < 2 ∧ f a x = 0

/-- The statement is a necessary but not sufficient condition --/
theorem necessary_but_not_sufficient :
  (∀ a : ℝ, 3 < a ∧ a < 4 → has_zero_in_interval a) ∧
  (∃ a : ℝ, has_zero_in_interval a ∧ (a ≤ 3 ∨ 4 ≤ a)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1924_192451


namespace NUMINAMATH_CALUDE_square_sum_geq_linear_l1924_192473

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_linear_l1924_192473


namespace NUMINAMATH_CALUDE_tim_speed_proof_l1924_192491

/-- Represents the initial distance between Tim and Élan in miles -/
def initial_distance : ℝ := 180

/-- Represents Élan's initial speed in mph -/
def elan_initial_speed : ℝ := 5

/-- Represents the distance Tim travels until meeting Élan in miles -/
def tim_travel_distance : ℝ := 120

/-- Represents Tim's initial speed in mph -/
def tim_initial_speed : ℝ := 40

theorem tim_speed_proof :
  ∃ (t : ℝ), 
    t > 0 ∧
    t + 2*t = tim_travel_distance ∧
    t = tim_initial_speed :=
by sorry

end NUMINAMATH_CALUDE_tim_speed_proof_l1924_192491


namespace NUMINAMATH_CALUDE_lunch_packing_ratio_l1924_192421

/-- The ratio of lunch-packing days between two students -/
theorem lunch_packing_ratio 
  (total_days : ℕ) 
  (aliyah_fraction : ℚ) 
  (becky_days : ℕ) 
  (h1 : total_days = 180) 
  (h2 : aliyah_fraction = 1/2) 
  (h3 : becky_days = 45) : 
  (becky_days : ℚ) / (aliyah_fraction * total_days) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_lunch_packing_ratio_l1924_192421


namespace NUMINAMATH_CALUDE_initial_walking_time_l1924_192438

/-- Proves that given a person walking at 5 kilometers per hour, if they need 3 more hours to reach a total of 30 kilometers, then they have already walked for 3 hours. -/
theorem initial_walking_time (speed : ℝ) (additional_hours : ℝ) (total_distance : ℝ) 
  (h1 : speed = 5)
  (h2 : additional_hours = 3)
  (h3 : total_distance = 30) :
  (total_distance - additional_hours * speed) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_walking_time_l1924_192438


namespace NUMINAMATH_CALUDE_wire_connections_count_l1924_192447

/-- The number of wire segments --/
def n : ℕ := 5

/-- The number of possible orientations for each segment --/
def orientations : ℕ := 2

/-- The total number of ways to connect the wire segments --/
def total_connections : ℕ := n.factorial * orientations ^ n

theorem wire_connections_count : total_connections = 3840 := by
  sorry

end NUMINAMATH_CALUDE_wire_connections_count_l1924_192447


namespace NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_range_l1924_192496

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  m * x - y + 1 - m = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 = 5

-- Define points D and E
def point_D : ℝ × ℝ := (-2, 0)
def point_E : ℝ × ℝ := (2, 0)

-- Define the condition for P being inside C
def inside_circle (x y : ℝ) : Prop :=
  x^2 + (y-1)^2 < 5

-- Define the geometric sequence condition
def geometric_sequence (x y : ℝ) : Prop :=
  ((x+2)^2 + y^2) * ((x-2)^2 + y^2) = (x^2 + y^2)^2

-- Theorem statement
theorem line_circle_intersection_and_dot_product_range :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), line_l m x y ∧ circle_C x y) ∧
  (∀ (x y : ℝ), 
    inside_circle x y → 
    geometric_sequence x y → 
    -2 ≤ ((x+2)*(-x+2) + y*(-y)) ∧ 
    ((x+2)*(-x+2) + y*(-y)) < 1 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_and_dot_product_range_l1924_192496


namespace NUMINAMATH_CALUDE_only_one_chooses_course_a_l1924_192464

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of course selection combinations -/
def total_combinations (n k : ℕ) : ℕ := (choose n k) * (choose n k)

/-- The number of combinations where both people choose course A -/
def both_choose_a (n k : ℕ) : ℕ := (choose (n - 1) (k - 1)) * (choose (n - 1) (k - 1))

/-- The number of ways in which only one person chooses course A -/
def only_one_chooses_a (n k : ℕ) : ℕ := (total_combinations n k) - (both_choose_a n k)

theorem only_one_chooses_course_a :
  only_one_chooses_a 4 2 = 27 := by sorry

end NUMINAMATH_CALUDE_only_one_chooses_course_a_l1924_192464


namespace NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l1924_192426

theorem power_two_plus_two_gt_square (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l1924_192426


namespace NUMINAMATH_CALUDE_complement_union_equality_l1924_192403

-- Define the sets M, N, and U
variable (M N U : Set α)

-- Define the conditions
variable (hM : M.Nonempty)
variable (hN : N.Nonempty)
variable (hU : U.Nonempty)
variable (hMN : M ⊆ N)
variable (hNU : N ⊆ U)

-- State the theorem
theorem complement_union_equality :
  (U \ M) ∪ (U \ N) = U \ M :=
sorry

end NUMINAMATH_CALUDE_complement_union_equality_l1924_192403


namespace NUMINAMATH_CALUDE_farm_heads_count_l1924_192495

theorem farm_heads_count (num_hens : ℕ) (total_feet : ℕ) : 
  num_hens = 30 → total_feet = 140 → 
  ∃ (num_cows : ℕ), 
    num_hens + num_cows = 50 ∧
    num_hens * 2 + num_cows * 4 = total_feet :=
by sorry

end NUMINAMATH_CALUDE_farm_heads_count_l1924_192495


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_l1924_192468

theorem prime_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d → 
  d % 6 = 0 := by
  sorry

#check prime_arithmetic_progression

end NUMINAMATH_CALUDE_prime_arithmetic_progression_l1924_192468


namespace NUMINAMATH_CALUDE_eraser_distribution_l1924_192474

theorem eraser_distribution (total_erasers : ℕ) (num_friends : ℕ) (erasers_per_friend : ℕ) :
  total_erasers = 3840 →
  num_friends = 48 →
  erasers_per_friend = total_erasers / num_friends →
  erasers_per_friend = 80 :=
by sorry

end NUMINAMATH_CALUDE_eraser_distribution_l1924_192474


namespace NUMINAMATH_CALUDE_family_size_problem_l1924_192435

theorem family_size_problem (avg_age_before avg_age_now baby_age : ℝ) 
  (h1 : avg_age_before = 17)
  (h2 : avg_age_now = 17)
  (h3 : baby_age = 2) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_age_before + (n : ℝ) * 3 + baby_age = (n + 1 : ℝ) * avg_age_now ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_family_size_problem_l1924_192435


namespace NUMINAMATH_CALUDE_complex_power_2018_l1924_192431

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_2018 : ((1 + i) / (1 - i)) ^ 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2018_l1924_192431


namespace NUMINAMATH_CALUDE_monotonic_increasing_implies_a_eq_neg_six_l1924_192461

/-- The function f(x) defined as the absolute value of 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + a|

/-- The property of f being monotonically increasing on [3, +∞) -/
def monotonic_increasing_from_three (a : ℝ) : Prop :=
  ∀ x y, 3 ≤ x → x ≤ y → f a x ≤ f a y

/-- Theorem stating that a must be -6 for f to be monotonically increasing on [3, +∞) -/
theorem monotonic_increasing_implies_a_eq_neg_six :
  ∃ a, monotonic_increasing_from_three a ↔ a = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_implies_a_eq_neg_six_l1924_192461


namespace NUMINAMATH_CALUDE_quadratic_term_zero_l1924_192449

theorem quadratic_term_zero (a : ℝ) : 
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (18 - 2 * a) * x^2 + (a - 6) * x + 3) →
  (∀ x : ℝ, (a * x + 3) * (6 * x^2 - 2 * x + 1) = 6 * a * x^3 + (a - 6) * x + 3) →
  a = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_term_zero_l1924_192449


namespace NUMINAMATH_CALUDE_custom_op_example_l1924_192444

-- Define the custom operation
def custom_op (m n p q : ℚ) : ℚ := m * p * ((q + n) / n)

-- State the theorem
theorem custom_op_example :
  custom_op 5 9 7 4 = 455 / 9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l1924_192444


namespace NUMINAMATH_CALUDE_largest_multiple_80_correct_l1924_192441

/-- Returns true if all digits of n are either 8 or 0 -/
def allDigits80 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

/-- The largest multiple of 20 with all digits 8 or 0 -/
def largestMultiple80 : ℕ := 8880

theorem largest_multiple_80_correct :
  largestMultiple80 % 20 = 0 ∧
  allDigits80 largestMultiple80 ∧
  ∀ n : ℕ, n > largestMultiple80 → ¬(n % 20 = 0 ∧ allDigits80 n) :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_80_correct_l1924_192441


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1924_192486

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 7b₃ is -16/7 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ x y : ℝ, x = b₂ ∧ y = b₃ → 3 * x + 7 * y ≥ -16/7) ∧
  (∃ x y : ℝ, x = b₂ ∧ y = b₃ ∧ 3 * x + 7 * y = -16/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1924_192486


namespace NUMINAMATH_CALUDE_job_completion_time_l1924_192469

/-- If m men can do a job in d days, and n men can do a different job in k days,
    then m+n men can do both jobs in (m * d + n * k) / (m + n) days. -/
theorem job_completion_time
  (m n d k : ℕ) (hm : m > 0) (hn : n > 0) (hd : d > 0) (hk : k > 0) :
  let total_time := (m * d + n * k) / (m + n)
  ∃ (time : ℚ), time = total_time ∧ time > 0 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1924_192469


namespace NUMINAMATH_CALUDE_min_alpha_value_l1924_192497

/-- Definition of α-level quasi-periodic function -/
def is_alpha_quasi_periodic (f : ℝ → ℝ) (D : Set ℝ) (α : ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x ∈ D, α * f x = f (x + T)

/-- The function f on the domain [1,+∞) -/
noncomputable def f : ℝ → ℝ
| x => if 1 ≤ x ∧ x < 2 then 2^x * (2*x + 1) else 0  -- We define f only for [1,2) as given

/-- Theorem statement -/
theorem min_alpha_value :
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) →  -- Monotonically increasing
  (∀ α, is_alpha_quasi_periodic f (Set.Ici 1) α → α ≥ 10/3) ∧
  (is_alpha_quasi_periodic f (Set.Ici 1) (10/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_alpha_value_l1924_192497


namespace NUMINAMATH_CALUDE_third_year_cost_l1924_192434

def total_first_year_cost : ℝ := 10000
def tuition_percentage : ℝ := 0.40
def room_and_board_percentage : ℝ := 0.35
def tuition_increase_rate : ℝ := 0.06
def room_and_board_increase_rate : ℝ := 0.03
def initial_financial_aid_percentage : ℝ := 0.25
def financial_aid_increase_rate : ℝ := 0.02

def tuition (year : ℕ) : ℝ :=
  total_first_year_cost * tuition_percentage * (1 + tuition_increase_rate) ^ (year - 1)

def room_and_board (year : ℕ) : ℝ :=
  total_first_year_cost * room_and_board_percentage * (1 + room_and_board_increase_rate) ^ (year - 1)

def textbooks_and_transportation : ℝ :=
  total_first_year_cost * (1 - tuition_percentage - room_and_board_percentage)

def financial_aid (year : ℕ) : ℝ :=
  tuition year * (initial_financial_aid_percentage + financial_aid_increase_rate * (year - 1))

def total_cost (year : ℕ) : ℝ :=
  tuition year + room_and_board year + textbooks_and_transportation - financial_aid year

theorem third_year_cost :
  total_cost 3 = 9404.17 := by
  sorry

end NUMINAMATH_CALUDE_third_year_cost_l1924_192434


namespace NUMINAMATH_CALUDE_divisor_inequality_l1924_192425

theorem divisor_inequality (n : ℕ) (a b c d : ℕ) : 
  (1 < a) → (a < b) → (b < c) → (c < d) → (d < n) →
  (∀ k : ℕ, k ∣ n → (k = 1 ∨ k = a ∨ k = b ∨ k = c ∨ k = d ∨ k = n)) →
  (a ∣ n) → (b ∣ n) → (c ∣ n) → (d ∣ n) →
  b - a ≤ d - c := by
  sorry

#check divisor_inequality

end NUMINAMATH_CALUDE_divisor_inequality_l1924_192425


namespace NUMINAMATH_CALUDE_books_on_shelf_l1924_192475

def initial_books : ℕ := 38
def marta_removes : ℕ := 10
def tom_removes : ℕ := 5
def tom_adds : ℕ := 12

theorem books_on_shelf : 
  initial_books - marta_removes - tom_removes + tom_adds = 35 := by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l1924_192475


namespace NUMINAMATH_CALUDE_morning_eggs_count_l1924_192413

/-- The number of eggs used in a day at the Wafting Pie Company -/
def total_eggs : ℕ := 1339

/-- The number of eggs used in the afternoon at the Wafting Pie Company -/
def afternoon_eggs : ℕ := 523

/-- The number of eggs used in the morning at the Wafting Pie Company -/
def morning_eggs : ℕ := total_eggs - afternoon_eggs

theorem morning_eggs_count : morning_eggs = 816 := by sorry

end NUMINAMATH_CALUDE_morning_eggs_count_l1924_192413


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l1924_192470

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 20) : 
  r - p = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l1924_192470


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_5_7_l1924_192422

theorem least_three_digit_multiple_of_3_5_7 : 
  ∀ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 105 :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_5_7_l1924_192422


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1924_192458

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1924_192458


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l1924_192409

theorem left_handed_rock_lovers (total : Nat) (left_handed : Nat) (rock_lovers : Nat) (right_handed_non_rock : Nat)
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : rock_lovers = 18)
  (h4 : right_handed_non_rock = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : Nat, x = left_handed + rock_lovers - total + right_handed_non_rock ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l1924_192409


namespace NUMINAMATH_CALUDE_average_string_length_l1924_192446

theorem average_string_length : 
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 7
  let total_length : ℚ := string1 + string2 + string3
  let num_strings : ℕ := 3
  (total_length / num_strings) = 14 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l1924_192446


namespace NUMINAMATH_CALUDE_dogs_adopted_twenty_dogs_adopted_l1924_192457

/-- The number of dogs adopted from a pet center --/
theorem dogs_adopted (initial_dogs : ℕ) (initial_cats : ℕ) (additional_cats : ℕ) (final_total : ℕ) : ℕ :=
  let remaining_dogs := initial_dogs - (initial_dogs + initial_cats + additional_cats - final_total)
  initial_dogs - remaining_dogs

/-- Proof that 20 dogs were adopted given the problem conditions --/
theorem twenty_dogs_adopted : dogs_adopted 36 29 12 57 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogs_adopted_twenty_dogs_adopted_l1924_192457


namespace NUMINAMATH_CALUDE_stratified_sample_size_is_15_l1924_192420

/-- Represents the number of workers in each age group -/
structure WorkerGroups where
  young : Nat
  middle_aged : Nat
  older : Nat

/-- Calculates the total sample size for a stratified sample -/
def stratified_sample_size (workers : WorkerGroups) (young_sample : Nat) : Nat :=
  let total_workers := workers.young + workers.middle_aged + workers.older
  let sampling_ratio := workers.young / young_sample
  total_workers / sampling_ratio

/-- Theorem: The stratified sample size for the given worker distribution is 15 -/
theorem stratified_sample_size_is_15 :
  let workers : WorkerGroups := ⟨35, 25, 15⟩
  stratified_sample_size workers 7 = 15 := by
  sorry

#eval stratified_sample_size ⟨35, 25, 15⟩ 7

end NUMINAMATH_CALUDE_stratified_sample_size_is_15_l1924_192420


namespace NUMINAMATH_CALUDE_weeds_never_cover_entire_field_l1924_192428

/-- Represents a 10x10 grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- The initial state of the grid with 9 occupied cells -/
def initial_state : Grid := sorry

/-- Checks if a cell is adjacent to at least two occupied cells -/
def has_two_adjacent_occupied (g : Grid) (i j : Fin 10) : Bool := sorry

/-- The next state of the grid after one step of spreading -/
def next_state (g : Grid) : Grid := sorry

/-- The state of the grid after n steps of spreading -/
def state_after_n_steps (n : ℕ) : Grid := sorry

/-- Counts the number of occupied cells in the grid -/
def count_occupied (g : Grid) : ℕ := sorry

theorem weeds_never_cover_entire_field :
  ∀ n : ℕ, count_occupied (state_after_n_steps n) < 100 := by sorry

end NUMINAMATH_CALUDE_weeds_never_cover_entire_field_l1924_192428


namespace NUMINAMATH_CALUDE_cone_base_radius_l1924_192465

-- Define the surface area of the cone
def surface_area (a : ℝ) : ℝ := a

-- Define the property that the lateral surface unfolds into a semicircle
def lateral_surface_is_semicircle (r l : ℝ) : Prop := 2 * Real.pi * r = Real.pi * l

-- Theorem statement
theorem cone_base_radius (a : ℝ) (h : a > 0) :
  ∃ (r : ℝ), r > 0 ∧ 
    (∃ (l : ℝ), l > 0 ∧ 
      lateral_surface_is_semicircle r l ∧ 
      surface_area a = Real.pi * r^2 + Real.pi * r * l) ∧
    r = Real.sqrt (a / (3 * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1924_192465


namespace NUMINAMATH_CALUDE_triangle_larger_segment_l1924_192407

theorem triangle_larger_segment (a b c h x : ℝ) : 
  a = 30 → b = 70 → c = 80 → 
  a^2 = x^2 + h^2 → 
  b^2 = (c - x)^2 + h^2 → 
  c - x = 65 :=
by sorry

end NUMINAMATH_CALUDE_triangle_larger_segment_l1924_192407


namespace NUMINAMATH_CALUDE_student_distribution_l1924_192492

theorem student_distribution (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) :
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = (Nat.choose n m) * (Nat.choose (n - m) m) * 1 :=
sorry

end NUMINAMATH_CALUDE_student_distribution_l1924_192492


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1924_192430

theorem polynomial_remainder_theorem (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^5 - 8*x^4 + 15*x^3 + 30*x^2 - 47*x + 20
  (f 2) = 104 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1924_192430


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1924_192454

/-- Returns true if n is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- Returns true if n starts with 2 -/
def startsWith2 (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 29

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The smallest two-digit prime number starting with 2 such that 
    reversing its digits produces a composite number is 23 -/
theorem smallest_two_digit_prime_with_composite_reverse : 
  ∃ (n : ℕ), 
    isTwoDigit n ∧ 
    startsWith2 n ∧ 
    Nat.Prime n ∧ 
    ¬(Nat.Prime (reverseDigits n)) ∧
    (∀ m, m < n → ¬(isTwoDigit m ∧ startsWith2 m ∧ Nat.Prime m ∧ ¬(Nat.Prime (reverseDigits m)))) ∧
    n = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_composite_reverse_l1924_192454


namespace NUMINAMATH_CALUDE_divided_triangle_area_l1924_192477

/-- Represents a triangle divided into six smaller triangles -/
structure DividedTriangle where
  /-- Areas of four known smaller triangles -/
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- The theorem stating that if a triangle is divided as described, with the given areas, its total area is 380 -/
theorem divided_triangle_area (t : DividedTriangle) 
  (h1 : t.area1 = 84) 
  (h2 : t.area2 = 70) 
  (h3 : t.area3 = 35) 
  (h4 : t.area4 = 65) : 
  ∃ (area5 area6 : ℝ), t.area1 + t.area2 + t.area3 + t.area4 + area5 + area6 = 380 := by
  sorry

end NUMINAMATH_CALUDE_divided_triangle_area_l1924_192477


namespace NUMINAMATH_CALUDE_share_multiple_l1924_192455

theorem share_multiple (total : ℝ) (c_share : ℝ) (k : ℝ) :
  total = 427 →
  c_share = 84 →
  (∃ (a_share b_share : ℝ), 
    total = a_share + b_share + c_share ∧
    3 * a_share = 4 * b_share ∧
    3 * a_share = k * c_share) →
  k = 7 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l1924_192455


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1924_192440

theorem quadratic_roots_problem (a b n r s : ℝ) : 
  a^2 - n*a + 6 = 0 →
  b^2 - n*b + 6 = 0 →
  (a + 1/b)^2 - r*(a + 1/b) + s = 0 →
  (b + 1/a)^2 - r*(b + 1/a) + s = 0 →
  s = 49/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1924_192440


namespace NUMINAMATH_CALUDE_c_paisa_per_a_rupee_l1924_192488

/-- Represents the share of money for each person in rupees -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : Shares) : Prop :=
  s.b = 0.65 * s.a ∧  -- For each Rs. A has, B has 65 paisa
  s.c = 32 ∧  -- C's share is Rs. 32
  s.a + s.b + s.c = 164  -- The total sum of money is Rs. 164

/-- The theorem to be proved -/
theorem c_paisa_per_a_rupee (s : Shares) 
  (h : problem_conditions s) : (s.c * 100) / s.a = 40 := by
  sorry


end NUMINAMATH_CALUDE_c_paisa_per_a_rupee_l1924_192488


namespace NUMINAMATH_CALUDE_school_outing_problem_l1924_192472

theorem school_outing_problem (x : ℕ) : 
  (3 * x + 16 = 5 * (x - 1) + 1) → (3 * x + 16 = 46) := by
  sorry

end NUMINAMATH_CALUDE_school_outing_problem_l1924_192472


namespace NUMINAMATH_CALUDE_max_x_squared_y_value_l1924_192408

theorem max_x_squared_y_value (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^3 + y^3 + 3*x*y = 1) :
  ∃ (M : ℝ), M = 4/27 ∧ x^2 * y ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_x_squared_y_value_l1924_192408


namespace NUMINAMATH_CALUDE_sine_squares_sum_l1924_192442

theorem sine_squares_sum (α : Real) : 
  (Real.sin (α - π/3))^2 + (Real.sin α)^2 + (Real.sin (α + π/3))^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_squares_sum_l1924_192442


namespace NUMINAMATH_CALUDE_no_hall_with_101_people_l1924_192490

/-- Represents a person in the hall -/
inductive Person
| knight : Person
| liar : Person

/-- Represents the hall with people and their pointing relationships -/
structure Hall :=
  (people : Finset Nat)
  (type : Nat → Person)
  (points_to : Nat → Nat)
  (in_hall : ∀ n, n ∈ people → points_to n ∈ people)
  (all_pointed_at : ∀ n ∈ people, ∃ m ∈ people, points_to m = n)
  (knight_points_to_liar : ∀ n ∈ people, type n = Person.knight → type (points_to n) = Person.liar)
  (liar_points_to_knight : ∀ n ∈ people, type n = Person.liar → type (points_to n) = Person.knight)

/-- Theorem stating that it's impossible to have exactly 101 people in the hall -/
theorem no_hall_with_101_people : ¬ ∃ (h : Hall), Finset.card h.people = 101 := by
  sorry

end NUMINAMATH_CALUDE_no_hall_with_101_people_l1924_192490


namespace NUMINAMATH_CALUDE_solve_system_l1924_192480

theorem solve_system (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 10) 
  (eq2 : 5 * p + 2 * q = 20) : 
  q = 10 / 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1924_192480


namespace NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l1924_192493

/-- Represents the number of family members -/
def family_size : ℕ := 5

/-- Represents the number of car seats -/
def car_seats : ℕ := 5

/-- Represents the number of eligible drivers -/
def eligible_drivers : ℕ := 3

/-- Calculates the number of seating arrangements -/
def seating_arrangements (f s d : ℕ) : ℕ :=
  d * (f - 1) * Nat.factorial (f - 2)

/-- Theorem stating the number of seating arrangements for the Lopez family -/
theorem lopez_family_seating_arrangements :
  seating_arrangements family_size car_seats eligible_drivers = 72 :=
by sorry

end NUMINAMATH_CALUDE_lopez_family_seating_arrangements_l1924_192493


namespace NUMINAMATH_CALUDE_valid_systematic_sampling_l1924_192467

/-- Represents a systematic sampling selection -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Generates the set of selected numbers for a systematic sampling -/
def generateSelection (s : SystematicSampling) : Finset Nat :=
  let interval := s.totalStudents / s.sampleSize
  Finset.image (fun i => s.startingNumber + i * interval) (Finset.range s.sampleSize)

/-- Theorem stating that {3, 13, 23, 33, 43} is a valid systematic sampling selection -/
theorem valid_systematic_sampling :
  ∃ (s : SystematicSampling),
    s.totalStudents = 50 ∧
    s.sampleSize = 5 ∧
    1 ≤ s.startingNumber ∧
    s.startingNumber ≤ s.totalStudents ∧
    generateSelection s = {3, 13, 23, 33, 43} :=
sorry

end NUMINAMATH_CALUDE_valid_systematic_sampling_l1924_192467


namespace NUMINAMATH_CALUDE_fourth_root_equality_l1924_192479

theorem fourth_root_equality (x : ℝ) (hx : x > 0) : 
  (x * (x^3)^(1/4))^(1/4) = x^(7/16) := by sorry

end NUMINAMATH_CALUDE_fourth_root_equality_l1924_192479


namespace NUMINAMATH_CALUDE_expression_value_l1924_192476

theorem expression_value (a : ℝ) (h : a^2 + 2*a - 1 = 0) : 
  ((a^2 - 1)/(a^2 - 2*a + 1) - 1/(1-a)) / (1/(a^2 - a)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1924_192476


namespace NUMINAMATH_CALUDE_complex_number_equality_l1924_192463

theorem complex_number_equality : ∀ (i : ℂ), i * i = -1 → (2 - i) * i = -1 + 2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1924_192463


namespace NUMINAMATH_CALUDE_f_of_g_10_l1924_192419

/-- The function g(x) = 4x + 10 -/
def g (x : ℝ) : ℝ := 4 * x + 10

/-- The function f(x) = 6x - 12 -/
def f (x : ℝ) : ℝ := 6 * x - 12

/-- Theorem: f(g(10)) = 288 -/
theorem f_of_g_10 : f (g 10) = 288 := by sorry

end NUMINAMATH_CALUDE_f_of_g_10_l1924_192419


namespace NUMINAMATH_CALUDE_polynomial_mapping_l1924_192406

def polynomial_equation (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) : Prop :=
  ∀ x, x^4 + a₁*x^3 + a₂*x^2 + a₃*x + a₄ = (x+1)^4 + b₁*(x+1)^3 + b₂*(x+1)^2 + b₃*(x+1) + b₄

def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let b₁ := 0
  let b₂ := -3
  let b₃ := 4
  let b₄ := -1
  (b₁, b₂, b₃, b₄)

theorem polynomial_mapping :
  polynomial_equation 4 3 2 1 0 (-3) 4 (-1) → f 4 3 2 1 = (0, -3, 4, -1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_mapping_l1924_192406


namespace NUMINAMATH_CALUDE_part1_part2_l1924_192487

-- Define the concept of l-increasing function
def is_l_increasing (f : ℝ → ℝ) (D : Set ℝ) (M : Set ℝ) (l : ℝ) : Prop :=
  l ≠ 0 ∧ (∀ x ∈ M, x + l ∈ D ∧ f (x + l) ≥ f x)

-- Part 1
theorem part1 (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Ici (-1), f x = x^2) →
  is_l_increasing f (Set.Ici (-1)) (Set.Ici (-1)) m →
  m ≥ 2 := by sorry

-- Part 2
theorem part2 (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f x) →
  (∀ x ≥ 0, f x = |x - a^2| - a^2) →
  is_l_increasing f Set.univ Set.univ 8 →
  -2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1924_192487


namespace NUMINAMATH_CALUDE_unique_solution_on_sphere_l1924_192498

theorem unique_solution_on_sphere (x y : ℝ) :
  (x - 8)^2 + (y - 9)^2 + (x - y)^2 = 1/3 →
  x = 8 + 1/3 ∧ y = 8 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_on_sphere_l1924_192498


namespace NUMINAMATH_CALUDE_function_behavior_l1924_192462

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) :
  is_even_function f →
  has_period f 2 →
  is_decreasing_on f (-1) 0 →
  (is_increasing_on f 6 7 ∧ is_decreasing_on f 7 8) :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_l1924_192462


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1924_192418

/-- Sum of interior angles of a polygon with n sides -/
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

theorem polygon_interior_angles :
  (∀ n : ℕ, n ≥ 3 → sumInteriorAngles n = (n - 2) * 180) ∧
  sumInteriorAngles 6 = 720 ∧
  (∃ n : ℕ, n ≥ 3 ∧ (1/3) * sumInteriorAngles n = 300 ∧ n = 7) :=
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1924_192418


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1924_192412

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 4 + Real.sqrt 4 / Real.sqrt 5) * (Real.sqrt 5 / Real.sqrt 6) =
  (Real.sqrt 10 + 2 * Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1924_192412


namespace NUMINAMATH_CALUDE_sufficient_condition_l1924_192423

theorem sufficient_condition (a : ℝ) : a ≥ 0 → a^2 + a ≥ 0 := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_l1924_192423


namespace NUMINAMATH_CALUDE_probability_of_marked_section_on_top_l1924_192427

theorem probability_of_marked_section_on_top (n : ℕ) (h : n = 8) : 
  (1 : ℚ) / n = (1 : ℚ) / 8 := by
  sorry

#check probability_of_marked_section_on_top

end NUMINAMATH_CALUDE_probability_of_marked_section_on_top_l1924_192427


namespace NUMINAMATH_CALUDE_min_value_T_l1924_192401

/-- Given a quadratic inequality with no real solutions and constraints on its coefficients,
    prove that a certain expression T has a minimum value of 4. -/
theorem min_value_T (a b c : ℝ) : 
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →  -- No real solutions to the inequality
  a > 0 →
  a * b > 1 → 
  (∀ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) → T ≥ 4) ∧ 
  (∃ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) ∧ T = 4) :=
by sorry


end NUMINAMATH_CALUDE_min_value_T_l1924_192401


namespace NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1924_192432

theorem simplify_tan_cot_expression :
  let tan_45 : Real := 1
  let cot_45 : Real := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_tan_cot_expression_l1924_192432


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1924_192443

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 < x ∧ x < 2}

def B : Set ℝ := {x | abs x ≤ 1}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1924_192443


namespace NUMINAMATH_CALUDE_range_of_m_l1924_192450

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ (4 - x) / 3 ∧ (4 - x) / 3 ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, ¬(p x) → ¬(q x m)) ∧  -- ¬p is necessary for ¬q
  (∃ x, ¬(p x) ∧ q x m) ∧     -- ¬p is not sufficient for ¬q
  (m > 0) →                   -- m is positive
  m ≥ 9 :=                    -- The range of m
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1924_192450


namespace NUMINAMATH_CALUDE_ticket_cost_calculation_l1924_192494

/-- Calculates the total amount spent on tickets given the prices and quantities -/
def total_ticket_cost (adult_price child_price : ℚ) (total_tickets child_tickets : ℕ) : ℚ :=
  let adult_tickets := total_tickets - child_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem stating that the total amount spent on tickets is $83.50 -/
theorem ticket_cost_calculation :
  total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16 = (83.50 : ℚ) := by
  sorry

#eval total_ticket_cost (5.50 : ℚ) (3.50 : ℚ) 21 16

end NUMINAMATH_CALUDE_ticket_cost_calculation_l1924_192494


namespace NUMINAMATH_CALUDE_solve_luncheon_problem_l1924_192417

def luncheon_problem (no_shows : ℕ) (table_capacity : ℕ) (tables_needed : ℕ) : Prop :=
  let attendees := tables_needed * table_capacity
  let total_invited := no_shows + attendees
  total_invited = 18

theorem solve_luncheon_problem :
  luncheon_problem 12 3 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_luncheon_problem_l1924_192417


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1924_192411

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1924_192411


namespace NUMINAMATH_CALUDE_stability_comparison_lower_variance_more_stable_shooting_competition_result_l1924_192436

/-- Represents a shooter in the competition -/
structure Shooter where
  name : String
  variance : ℝ

/-- Defines the stability of a shooter based on their variance -/
def moreStable (a b : Shooter) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : Shooter) 
  (h : a.variance ≠ b.variance) : 
  moreStable a b ∨ moreStable b a :=
sorry

theorem lower_variance_more_stable (a b : Shooter) 
  (h : a.variance < b.variance) : 
  moreStable a b :=
sorry

theorem shooting_competition_result (a b : Shooter)
  (ha : a.name = "A" ∧ a.variance = 0.25)
  (hb : b.name = "B" ∧ b.variance = 0.12) :
  moreStable b a :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_lower_variance_more_stable_shooting_competition_result_l1924_192436


namespace NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l1924_192402

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Properties of the cylinder -/
def CylinderProperties (c : RightCircularCylinder) : Prop :=
  let lateral_area := 2 * Real.pi * c.radius * c.height
  let base_area := Real.pi * c.radius ^ 2
  lateral_area / base_area = 5 / 3 ∧
  (4 * c.radius ^ 2 + c.height ^ 2) = 39 ^ 2

/-- Theorem statement -/
theorem cylinder_surface_area_and_volume 
  (c : RightCircularCylinder) 
  (h : CylinderProperties c) : 
  (2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius ^ 2 = 1188 * Real.pi) ∧
  (Real.pi * c.radius ^ 2 * c.height = 4860 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l1924_192402


namespace NUMINAMATH_CALUDE_transylvanian_logic_l1924_192452

/-- Represents the possible types of beings in Transylvania -/
inductive Being
| Human
| Vampire

/-- Represents the possible responses to questions -/
inductive Response
| Yes
| No

/-- A function that determines how a being responds to a question about another being's type -/
def respond (respondent : Being) (subject : Being) : Response :=
  match respondent, subject with
  | Being.Human, Being.Human => Response.Yes
  | Being.Human, Being.Vampire => Response.No
  | Being.Vampire, Being.Human => Response.No
  | Being.Vampire, Being.Vampire => Response.Yes

theorem transylvanian_logic (A B : Being) 
  (h1 : respond A B = Response.Yes) : 
  respond B A = Response.Yes := by
  sorry

end NUMINAMATH_CALUDE_transylvanian_logic_l1924_192452


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1924_192466

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 6
  let θ : ℝ := (7 * π) / 4
  let φ : ℝ := π / 3
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) := by
  sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1924_192466


namespace NUMINAMATH_CALUDE_two_intersection_points_l1924_192410

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem: There are exactly two distinct intersection points
theorem two_intersection_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_intersection_point x₁ y₁ ∧
    is_intersection_point x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∀ (x y : ℝ), is_intersection_point x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l1924_192410


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1924_192437

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + 8*y - x*y = 0 ∧ x + y = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1924_192437


namespace NUMINAMATH_CALUDE_infinitely_many_nondivisible_l1924_192424

theorem infinitely_many_nondivisible (a b : ℕ) : 
  Set.Infinite {n : ℕ | ¬(n^b + 1 ∣ a^n + 1)} := by
sorry

end NUMINAMATH_CALUDE_infinitely_many_nondivisible_l1924_192424


namespace NUMINAMATH_CALUDE_expected_weekly_rainfall_l1924_192484

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def prob_sun : ℝ := 0.3

/-- The probability of 3 inches of rain --/
def prob_rain_3 : ℝ := 0.4

/-- The probability of 7 inches of rain --/
def prob_rain_7 : ℝ := 0.3

/-- The amount of rain in inches for the sunny scenario --/
def rain_sun : ℝ := 0

/-- The amount of rain in inches for the 3-inch rain scenario --/
def rain_3 : ℝ := 3

/-- The amount of rain in inches for the 7-inch rain scenario --/
def rain_7 : ℝ := 7

/-- The expected value of rainfall for a single day --/
def expected_daily_rainfall : ℝ :=
  prob_sun * rain_sun + prob_rain_3 * rain_3 + prob_rain_7 * rain_7

theorem expected_weekly_rainfall :
  days * expected_daily_rainfall = 23.1 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rainfall_l1924_192484


namespace NUMINAMATH_CALUDE_play_recording_distribution_l1924_192471

theorem play_recording_distribution (play_duration : ℕ) (disc_capacity : ℕ) 
  (h1 : play_duration = 385)
  (h2 : disc_capacity = 75) : 
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * disc_capacity ≥ play_duration ∧
    (num_discs - 1) * disc_capacity < play_duration ∧
    play_duration / num_discs = 64 := by
  sorry

end NUMINAMATH_CALUDE_play_recording_distribution_l1924_192471
