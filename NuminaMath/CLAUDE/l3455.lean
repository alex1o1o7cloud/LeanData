import Mathlib

namespace NUMINAMATH_CALUDE_stone_count_is_odd_l3455_345568

/-- Represents the assembly of stones along a road -/
structure StoneAssembly where
  stone_interval : ℝ  -- Distance between stones in meters
  total_distance : ℝ  -- Total distance covered in meters

/-- Theorem: The total number of stones is odd given the conditions -/
theorem stone_count_is_odd (assembly : StoneAssembly) 
  (h_interval : assembly.stone_interval = 10)
  (h_distance : assembly.total_distance = 4800) : 
  ∃ (n : ℕ), (2 * n + 1) * assembly.stone_interval = assembly.total_distance / 2 := by
  sorry

#check stone_count_is_odd

end NUMINAMATH_CALUDE_stone_count_is_odd_l3455_345568


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l3455_345517

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to f(x) at x = 1 is 2 -/
theorem tangent_slope_at_one : 
  (deriv f) 1 = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l3455_345517


namespace NUMINAMATH_CALUDE_circle_sequence_theorem_circle_sequence_theorem_proof_l3455_345506

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a structure for a circle
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

-- Define a function to check if a circle passes through two points
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  (c.center.x - p1.x)^2 + (c.center.y - p1.y)^2 = c.radius^2 ∧
  (c.center.x - p2.x)^2 + (c.center.y - p2.y)^2 = c.radius^2

-- Define a function to check if two circles are tangent
def areTangent (c1 c2 : Circle) : Prop :=
  (c1.center.x - c2.center.x)^2 + (c1.center.y - c2.center.y)^2 = (c1.radius + c2.radius)^2

-- Define the main theorem
theorem circle_sequence_theorem (t : Triangle) 
  (C1 C2 C3 C4 C5 C6 C7 : Circle) : Prop :=
  passesThrough C1 t.A t.B ∧
  passesThrough C2 t.B t.C ∧ areTangent C1 C2 ∧
  passesThrough C3 t.C t.A ∧ areTangent C2 C3 ∧
  passesThrough C4 t.A t.B ∧ areTangent C3 C4 ∧
  passesThrough C5 t.B t.C ∧ areTangent C4 C5 ∧
  passesThrough C6 t.C t.A ∧ areTangent C5 C6 ∧
  passesThrough C7 t.A t.B ∧ areTangent C6 C7
  →
  C7 = C1

-- The proof would go here
theorem circle_sequence_theorem_proof : ∀ t C1 C2 C3 C4 C5 C6 C7, 
  circle_sequence_theorem t C1 C2 C3 C4 C5 C6 C7 :=
sorry

end NUMINAMATH_CALUDE_circle_sequence_theorem_circle_sequence_theorem_proof_l3455_345506


namespace NUMINAMATH_CALUDE_quarters_needed_for_final_soda_l3455_345545

theorem quarters_needed_for_final_soda (total_quarters : ℕ) (soda_cost : ℕ) : 
  total_quarters = 855 → soda_cost = 7 → 
  (soda_cost - (total_quarters % soda_cost)) = 6 := by
sorry

end NUMINAMATH_CALUDE_quarters_needed_for_final_soda_l3455_345545


namespace NUMINAMATH_CALUDE_alyssa_kittens_l3455_345577

/-- The number of kittens Alyssa initially had -/
def initial_kittens : ℕ := 8

/-- The number of kittens Alyssa gave to her friends -/
def given_away_kittens : ℕ := 4

/-- The number of kittens Alyssa now has -/
def remaining_kittens : ℕ := initial_kittens - given_away_kittens

theorem alyssa_kittens : remaining_kittens = 4 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_kittens_l3455_345577


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_l3455_345531

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2*y ≤ 3) (h2 : y ≥ 0) :
  (∀ x' y', x' + 2*y' ≤ 3 → y' ≥ 0 → 2*x' + y' ≤ 6) ∧ 
  (∃ x₀ y₀, x₀ + 2*y₀ ≤ 3 ∧ y₀ ≥ 0 ∧ 2*x₀ + y₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_l3455_345531


namespace NUMINAMATH_CALUDE_no_divisor_square_sum_l3455_345562

theorem no_divisor_square_sum (n : ℕ+) :
  ¬∃ d : ℕ+, (d ∣ 2 * n^2) ∧ ∃ x : ℕ, d^2 * n^2 + d^3 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_divisor_square_sum_l3455_345562


namespace NUMINAMATH_CALUDE_original_fraction_l3455_345592

theorem original_fraction (x y : ℚ) : 
  x > 0 ∧ y > 0 →
  (120 / 100 * x) / (75 / 100 * y) = 2 / 15 →
  x / y = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_original_fraction_l3455_345592


namespace NUMINAMATH_CALUDE_rescue_mission_analysis_l3455_345514

def daily_distances : List Int := [14, -9, 8, -7, 13, -6, 10, -5]
def fuel_consumption : Rat := 1/2
def fuel_capacity : Nat := 29

theorem rescue_mission_analysis :
  let net_distance := daily_distances.sum
  let max_distance := daily_distances.scanl (· + ·) 0 |>.map abs |>.maximum
  let total_distance := daily_distances.map abs |>.sum
  let fuel_needed := fuel_consumption * total_distance - fuel_capacity
  (net_distance = 18 ∧ 
   max_distance = some 23 ∧ 
   fuel_needed = 7) := by sorry

end NUMINAMATH_CALUDE_rescue_mission_analysis_l3455_345514


namespace NUMINAMATH_CALUDE_different_color_probability_l3455_345566

/-- The probability of drawing two balls of different colors from a box containing 2 red balls and 3 black balls -/
theorem different_color_probability (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) : 
  total_balls = 5 →
  red_balls = 2 →
  black_balls = 3 →
  (red_balls * black_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_different_color_probability_l3455_345566


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3455_345513

theorem quadratic_factorization (c d : ℕ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  (∀ x, x^2 - 18*x + 72 = (x - c)*(x - d)) →
  4*d - c = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3455_345513


namespace NUMINAMATH_CALUDE_right_triangle_from_number_and_reciprocal_l3455_345518

theorem right_triangle_from_number_and_reciprocal (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let s := (a + 1/a) / 2
  let d := (a - 1/a) / 2
  let p := a * (1/a)
  s^2 = d^2 + p^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_from_number_and_reciprocal_l3455_345518


namespace NUMINAMATH_CALUDE_lcm_of_48_and_64_l3455_345538

theorem lcm_of_48_and_64 :
  let a := 48
  let b := 64
  let hcf := 16
  lcm a b = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_48_and_64_l3455_345538


namespace NUMINAMATH_CALUDE_total_distributions_l3455_345579

def number_of_balls : ℕ := 8
def number_of_boxes : ℕ := 3

def valid_distribution (d : List ℕ) : Prop :=
  d.length = number_of_boxes ∧
  d.sum = number_of_balls ∧
  d.all (· > 0) ∧
  d.Pairwise (· ≠ ·)

def count_distributions : ℕ := sorry

theorem total_distributions :
  count_distributions = 2688 := by sorry

end NUMINAMATH_CALUDE_total_distributions_l3455_345579


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l3455_345557

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the distance squared between two points
def DistanceSquared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- Define the set of points P satisfying the condition
def SatisfyingPoints (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | p ∈ Circle center radius ∧
               let (a, b) := DiameterEndpoints center radius
               DistanceSquared p a + DistanceSquared p b = 10}

-- Theorem statement
theorem infinitely_many_satisfying_points (center : ℝ × ℝ) :
  Set.Infinite (SatisfyingPoints center 2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l3455_345557


namespace NUMINAMATH_CALUDE_basketball_tournament_l3455_345580

theorem basketball_tournament (x : ℕ) : 
  (3 * x / 4 : ℚ) = x - x / 4 ∧ 
  (2 * (x + 4) / 3 : ℚ) = (x + 4) - (x + 4) / 3 ∧ 
  (2 * (x + 4) / 3 : ℚ) = 3 * x / 4 + 9 ∧ 
  ((x + 4) / 3 : ℚ) = x / 4 + 5 → 
  x = 76 := by
sorry

end NUMINAMATH_CALUDE_basketball_tournament_l3455_345580


namespace NUMINAMATH_CALUDE_line_parabola_single_intersection_l3455_345542

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line passing through (-3, 1) with slope k
def line (k x y : ℝ) : Prop := y - 1 = k * (x + 3)

-- Define the condition for the line to intersect the parabola at exactly one point
def single_intersection (k : ℝ) : Prop :=
  (k = 0 ∨ k = -1 ∨ k = 2/3) ∧
  (∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line k p.1 p.2)

-- Theorem statement
theorem line_parabola_single_intersection (k : ℝ) :
  single_intersection k ↔ k = 0 ∨ k = -1 ∨ k = 2/3 :=
sorry

end NUMINAMATH_CALUDE_line_parabola_single_intersection_l3455_345542


namespace NUMINAMATH_CALUDE_shelly_has_enough_thread_l3455_345593

/-- Represents the keychain making scenario for Shelly's friends --/
structure KeychainScenario where
  class_friends : Nat
  club_friends : Nat
  sports_friends : Nat
  class_thread : Nat
  club_thread : Nat
  sports_thread : Nat
  available_thread : Nat

/-- Calculates the total thread needed and checks if it's sufficient --/
def thread_calculation (scenario : KeychainScenario) : 
  (Bool × Nat) :=
  let total_needed := 
    scenario.class_friends * scenario.class_thread +
    scenario.club_friends * scenario.club_thread +
    scenario.sports_friends * scenario.sports_thread
  let is_sufficient := total_needed ≤ scenario.available_thread
  let remaining := scenario.available_thread - total_needed
  (is_sufficient, remaining)

/-- Theorem stating that Shelly has enough thread and calculates the remaining amount --/
theorem shelly_has_enough_thread (scenario : KeychainScenario) 
  (h1 : scenario.class_friends = 10)
  (h2 : scenario.club_friends = 20)
  (h3 : scenario.sports_friends = 5)
  (h4 : scenario.class_thread = 18)
  (h5 : scenario.club_thread = 24)
  (h6 : scenario.sports_thread = 30)
  (h7 : scenario.available_thread = 1200) :
  thread_calculation scenario = (true, 390) := by
  sorry

end NUMINAMATH_CALUDE_shelly_has_enough_thread_l3455_345593


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3455_345555

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) ≤ 1/2) ↔ (∃ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3455_345555


namespace NUMINAMATH_CALUDE_system_of_equations_l3455_345573

/-- Given a system of equations, prove the values of x, y, and z. -/
theorem system_of_equations : 
  let x := 80 * (1 + 0.11)
  let y := 120 * (1 - 0.15)
  let z := (0.4 * (x + y)) * (1 + 0.2)
  (x = 88.8) ∧ (y = 102) ∧ (z = 91.584) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l3455_345573


namespace NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3455_345503

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithProgressionPoly where
  j : ℝ
  k : ℝ
  roots : Fin 4 → ℝ
  distinct : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + j * (roots i)^2 + k * (roots i) + 400 = 0

/-- The value of j in an ArithProgressionPoly is -40 -/
theorem arith_progression_poly_j_value (p : ArithProgressionPoly) : p.j = -40 := by
  sorry

end NUMINAMATH_CALUDE_arith_progression_poly_j_value_l3455_345503


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3455_345543

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2004 - 1) (2^1995 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3455_345543


namespace NUMINAMATH_CALUDE_evaluate_expression_l3455_345559

theorem evaluate_expression : 
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3455_345559


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_ratio_l3455_345569

theorem complex_pure_imaginary_ratio (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∃ (y : ℝ), (5 - 3 * Complex.I) * (m + n * Complex.I) = y * Complex.I) : 
  m / n = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_ratio_l3455_345569


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3455_345585

/-- Calculates the simple interest rate given principal, amount, and time -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem stating that the interest rate is approximately 1.11% -/
theorem interest_rate_calculation (principal amount : ℚ) (time : ℕ) 
  (h_principal : principal = 900)
  (h_amount : amount = 950)
  (h_time : time = 5) :
  abs (calculate_interest_rate principal amount time - 1.11) < 0.01 := by
  sorry

#eval calculate_interest_rate 900 950 5

end NUMINAMATH_CALUDE_interest_rate_calculation_l3455_345585


namespace NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3455_345546

/-- If rowing against a stream takes twice as long as rowing with the stream for the same distance,
    then the ratio of the boat's speed in still water to the stream's speed is 3:1. -/
theorem boat_stream_speed_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed > stream_speed) 
  (h2 : stream_speed > 0) 
  (h3 : distance > 0) 
  (h4 : distance / (boat_speed - stream_speed) = 2 * (distance / (boat_speed + stream_speed))) : 
  boat_speed / stream_speed = 3 := by
sorry


end NUMINAMATH_CALUDE_boat_stream_speed_ratio_l3455_345546


namespace NUMINAMATH_CALUDE_june_greatest_drop_l3455_345587

/-- Represents the months in the first half of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => 1.50
  | Month.February => -2.25
  | Month.March => 0.75
  | Month.April => -3.00
  | Month.May => 1.00
  | Month.June => -4.00

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.June

theorem june_greatest_drop :
  ∀ m : Month, price_change m ≥ price_change greatest_drop → m = greatest_drop :=
by sorry

end NUMINAMATH_CALUDE_june_greatest_drop_l3455_345587


namespace NUMINAMATH_CALUDE_environmental_policy_support_l3455_345551

theorem environmental_policy_support (men_support_rate : ℚ) (women_support_rate : ℚ)
  (men_count : ℕ) (women_count : ℕ) 
  (h1 : men_support_rate = 75 / 100)
  (h2 : women_support_rate = 70 / 100)
  (h3 : men_count = 200)
  (h4 : women_count = 800) :
  (men_support_rate * men_count + women_support_rate * women_count) / (men_count + women_count) = 71 / 100 :=
by sorry

end NUMINAMATH_CALUDE_environmental_policy_support_l3455_345551


namespace NUMINAMATH_CALUDE_line_mb_value_l3455_345510

/-- Given a line y = mx + b passing through the points (0, -3) and (1, -1), prove that mb = 6 -/
theorem line_mb_value (m b : ℝ) : 
  (0 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-3 : ℝ) = m * 0 + b →  -- The line passes through (0, -3)
  (-1 : ℝ) = m * 1 + b →  -- The line passes through (1, -1)
  m * b = 6 := by
sorry

end NUMINAMATH_CALUDE_line_mb_value_l3455_345510


namespace NUMINAMATH_CALUDE_blisters_on_rest_eq_80_l3455_345536

/-- Represents the number of blisters on one arm -/
def blisters_per_arm : ℕ := 60

/-- Represents the total number of blisters -/
def total_blisters : ℕ := 200

/-- Calculates the number of blisters on the rest of the body -/
def blisters_on_rest : ℕ := total_blisters - 2 * blisters_per_arm

theorem blisters_on_rest_eq_80 : blisters_on_rest = 80 := by
  sorry

end NUMINAMATH_CALUDE_blisters_on_rest_eq_80_l3455_345536


namespace NUMINAMATH_CALUDE_three_zeros_a_range_l3455_345539

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def g (x : ℝ) : ℝ := -Real.log x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ := min (f a x) (g x)

theorem three_zeros_a_range (a : ℝ) :
  (∃ x y z : ℝ, x < y ∧ y < z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    (∀ w : ℝ, w > 0 → h a w = 0 → w = x ∨ w = y ∨ w = z)) →
  -5/4 < a ∧ a < -3/4 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_a_range_l3455_345539


namespace NUMINAMATH_CALUDE_max_distinct_dance_counts_29_15_l3455_345590

/-- Represents the maximum number of distinct dance counts that can be reported
    given a number of boys and girls at a ball, where each boy can dance with
    each girl at most once. -/
def max_distinct_dance_counts (num_boys num_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 29 boys and 15 girls, the maximum number of
    distinct dance counts is 29. -/
theorem max_distinct_dance_counts_29_15 :
  max_distinct_dance_counts 29 15 = 29 := by sorry

end NUMINAMATH_CALUDE_max_distinct_dance_counts_29_15_l3455_345590


namespace NUMINAMATH_CALUDE_real_number_groups_ratio_l3455_345584

theorem real_number_groups_ratio (k : ℝ) (hk : k > 0) : 
  ∃ (group : Set ℝ) (a b c : ℝ), 
    (group ∪ (Set.univ \ group) = Set.univ) ∧ 
    (group ∩ (Set.univ \ group) = ∅) ∧
    (a ∈ group ∧ b ∈ group ∧ c ∈ group) ∧
    (a < b ∧ b < c) ∧
    ((c - b) / (b - a) = k) :=
sorry

end NUMINAMATH_CALUDE_real_number_groups_ratio_l3455_345584


namespace NUMINAMATH_CALUDE_modified_goldbach_for_2024_l3455_345500

theorem modified_goldbach_for_2024 :
  ∃ (p q : ℕ), p ≠ q ∧ Prime p ∧ Prime q ∧ p + q = 2024 :=
by
  sorry

#check modified_goldbach_for_2024

end NUMINAMATH_CALUDE_modified_goldbach_for_2024_l3455_345500


namespace NUMINAMATH_CALUDE_prob_second_science_example_l3455_345549

/-- Represents a set of questions with science and humanities subjects -/
structure QuestionSet where
  total : Nat
  science : Nat
  humanities : Nat
  h_total : total = science + humanities

/-- Calculates the probability of drawing a science question on the second draw,
    given that the first drawn question was a science question -/
def prob_second_science (qs : QuestionSet) : Rat :=
  if qs.science > 0 then
    (qs.science - 1) / (qs.total - 1)
  else
    0

theorem prob_second_science_example :
  let qs : QuestionSet := ⟨5, 3, 2, rfl⟩
  prob_second_science qs = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_second_science_example_l3455_345549


namespace NUMINAMATH_CALUDE_find_B_l3455_345541

theorem find_B (A B : ℕ) (h1 : A < 10) (h2 : B < 10) (h3 : 6 * 100 * A + 5 + 100 * B + 3 = 748) : B = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3455_345541


namespace NUMINAMATH_CALUDE_michelle_score_l3455_345576

/-- Michelle's basketball game record --/
theorem michelle_score (total_score : ℕ) (num_players : ℕ) (other_players : ℕ) (avg_other_score : ℕ) : 
  total_score = 72 →
  num_players = 8 →
  other_players = 7 →
  avg_other_score = 6 →
  total_score - (other_players * avg_other_score) = 30 := by
sorry

end NUMINAMATH_CALUDE_michelle_score_l3455_345576


namespace NUMINAMATH_CALUDE_tangent_condition_l3455_345501

/-- The equation of the curve -/
def curve_eq (x y : ℝ) : Prop := y^2 - 4*x - 2*y + 1 = 0

/-- The equation of the line -/
def line_eq (k x y : ℝ) : Prop := y = k*x + 2

/-- The line is tangent to the curve -/
def is_tangent (k : ℝ) : Prop :=
  ∃! x y, curve_eq x y ∧ line_eq k x y

/-- The main theorem -/
theorem tangent_condition :
  ∀ k, is_tangent k ↔ (k = -2 + 2*Real.sqrt 2 ∨ k = -2 - 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_condition_l3455_345501


namespace NUMINAMATH_CALUDE_correct_product_l3455_345516

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →
  (reverse_digits a * b + 5 = 266) →
  (a * b = 828) :=
by sorry

end NUMINAMATH_CALUDE_correct_product_l3455_345516


namespace NUMINAMATH_CALUDE_f_max_min_range_l3455_345532

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The condition for f to have both a maximum and a minimum -/
def has_max_and_min (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (∀ x, f a x ≤ f a x₁) ∧
  (∀ x, f a x ≥ f a x₂)

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem f_max_min_range :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -3 ∨ a > 6) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_range_l3455_345532


namespace NUMINAMATH_CALUDE_megan_markers_theorem_l3455_345591

/-- The number of markers Megan had initially -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers plus the markers given by Robert equals the total number of markers Megan has now -/
theorem megan_markers_theorem : initial_markers + roberts_markers = total_markers := by sorry

end NUMINAMATH_CALUDE_megan_markers_theorem_l3455_345591


namespace NUMINAMATH_CALUDE_arcsin_cos_4pi_over_7_l3455_345524

theorem arcsin_cos_4pi_over_7 : 
  Real.arcsin (Real.cos (4 * π / 7)) = -π / 14 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_4pi_over_7_l3455_345524


namespace NUMINAMATH_CALUDE_function_evaluation_l3455_345558

/-- Given a function f(x) = x^2 + 1, prove that f(a+1) = a^2 + 2a + 2 -/
theorem function_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 1
  f (a + 1) = a^2 + 2*a + 2 := by
sorry

end NUMINAMATH_CALUDE_function_evaluation_l3455_345558


namespace NUMINAMATH_CALUDE_min_value_of_permutation_sum_l3455_345560

theorem min_value_of_permutation_sum :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℕ),
  (x₁ :: x₂ :: x₃ :: x₄ :: x₅ :: []).Perm [1, 2, 3, 4, 5] →
  (∀ (y₁ y₂ y₃ y₄ y₅ : ℕ),
    (y₁ :: y₂ :: y₃ :: y₄ :: y₅ :: []).Perm [1, 2, 3, 4, 5] →
    x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ ≤ y₁ + 2*y₂ + 3*y₃ + 4*y₄ + 5*y₅) →
  x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = 35 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_permutation_sum_l3455_345560


namespace NUMINAMATH_CALUDE_puppies_adoption_time_l3455_345544

theorem puppies_adoption_time (initial_puppies : ℕ) (additional_puppies : ℕ) (adoption_rate : ℕ) :
  initial_puppies = 10 →
  additional_puppies = 15 →
  adoption_rate = 7 →
  (∃ (days : ℕ), days = 4 ∧ days * adoption_rate ≥ initial_puppies + additional_puppies ∧
   (days - 1) * adoption_rate < initial_puppies + additional_puppies) :=
by sorry

end NUMINAMATH_CALUDE_puppies_adoption_time_l3455_345544


namespace NUMINAMATH_CALUDE_house_purchase_l3455_345525

/-- Represents a number in base s -/
def BaseS (n : ℕ) (s : ℕ) : ℕ → ℕ
| 0 => 0
| (k+1) => (n % s) + s * BaseS (n / s) s k

theorem house_purchase (s : ℕ) 
  (h1 : BaseS 530 s 2 + BaseS 450 s 2 = BaseS 1100 s 3) : s = 8 :=
sorry

end NUMINAMATH_CALUDE_house_purchase_l3455_345525


namespace NUMINAMATH_CALUDE_find_numbers_with_difference_and_quotient_equal_l3455_345529

theorem find_numbers_with_difference_and_quotient_equal (x y : ℚ) :
  x - y = 5 ∧ x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_find_numbers_with_difference_and_quotient_equal_l3455_345529


namespace NUMINAMATH_CALUDE_tank_width_is_six_l3455_345554

/-- Represents the properties of a rectangular tank being filled with water. -/
structure Tank where
  fill_rate : ℝ  -- Cubic feet per hour
  fill_time : ℝ  -- Hours
  length : ℝ     -- Feet
  depth : ℝ      -- Feet

/-- Calculates the volume of a rectangular tank. -/
def tank_volume (t : Tank) (width : ℝ) : ℝ :=
  t.length * width * t.depth

/-- Calculates the volume of water filled in the tank. -/
def filled_volume (t : Tank) : ℝ :=
  t.fill_rate * t.fill_time

/-- Theorem stating that the width of the tank is 6 feet. -/
theorem tank_width_is_six (t : Tank) 
  (h1 : t.fill_rate = 5)
  (h2 : t.fill_time = 60)
  (h3 : t.length = 10)
  (h4 : t.depth = 5) :
  ∃ (w : ℝ), w = 6 ∧ tank_volume t w = filled_volume t :=
sorry

end NUMINAMATH_CALUDE_tank_width_is_six_l3455_345554


namespace NUMINAMATH_CALUDE_simplify_cube_roots_product_l3455_345528

theorem simplify_cube_roots_product : 
  (1 + 27) ^ (1/3 : ℝ) * (1 + 27 ^ (1/3 : ℝ)) ^ (1/3 : ℝ) * (4 : ℝ) ^ (1/2 : ℝ) = 2 * 112 ^ (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_product_l3455_345528


namespace NUMINAMATH_CALUDE_missing_number_proof_l3455_345578

theorem missing_number_proof (x : ℝ) : x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3455_345578


namespace NUMINAMATH_CALUDE_sum_of_squares_l3455_345586

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3455_345586


namespace NUMINAMATH_CALUDE_conflict_graph_contains_k4_l3455_345527

/-- A graph representing conflicts between mafia clans -/
structure ConflictGraph where
  /-- The set of vertices (clans) in the graph -/
  vertices : Finset Nat
  /-- The set of edges (conflicts) in the graph -/
  edges : Finset (Nat × Nat)
  /-- The number of vertices is 20 -/
  vertex_count : vertices.card = 20
  /-- Each vertex has a degree of at least 14 -/
  min_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 14

/-- A complete subgraph of size 4 -/
def CompleteSubgraph4 (g : ConflictGraph) : Prop :=
  ∃ (a b c d : Nat), a ∈ g.vertices ∧ b ∈ g.vertices ∧ c ∈ g.vertices ∧ d ∈ g.vertices ∧
    (a, b) ∈ g.edges ∧ (a, c) ∈ g.edges ∧ (a, d) ∈ g.edges ∧
    (b, c) ∈ g.edges ∧ (b, d) ∈ g.edges ∧
    (c, d) ∈ g.edges

/-- Theorem: Every ConflictGraph contains a complete subgraph of size 4 -/
theorem conflict_graph_contains_k4 (g : ConflictGraph) : CompleteSubgraph4 g := by
  sorry

end NUMINAMATH_CALUDE_conflict_graph_contains_k4_l3455_345527


namespace NUMINAMATH_CALUDE_possible_values_of_d_l3455_345571

theorem possible_values_of_d (a b c d : ℕ) 
  (h : (a * d - 1) / (a + 1) + (b * d - 1) / (b + 1) + (c * d - 1) / (c + 1) = d) :
  d = 1 ∨ d = 2 ∨ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_d_l3455_345571


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l3455_345512

theorem unique_solution_linear_system :
  ∃! (x y z : ℝ), 
    2*x - 3*y + z = -4 ∧
    5*x - 2*y - 3*z = 7 ∧
    x + y - 4*z = -6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l3455_345512


namespace NUMINAMATH_CALUDE_variable_value_l3455_345502

theorem variable_value : 
  ∀ (a n some_variable : ℤ) (x : ℝ),
  (3 * x + 2) * (2 * x - 7) = a * x^2 + some_variable * x + n →
  a - n + some_variable = 3 →
  some_variable = -17 := by
sorry

end NUMINAMATH_CALUDE_variable_value_l3455_345502


namespace NUMINAMATH_CALUDE_allison_bought_28_items_l3455_345526

/-- The number of glue sticks Marie bought -/
def marie_glue_sticks : ℕ := 15

/-- The number of construction paper packs Marie bought -/
def marie_paper_packs : ℕ := 30

/-- The difference in glue sticks between Allison and Marie -/
def glue_stick_difference : ℕ := 8

/-- The ratio of construction paper packs between Marie and Allison -/
def paper_pack_ratio : ℕ := 6

/-- The total number of craft supply items Allison bought -/
def allison_total_items : ℕ := marie_glue_sticks + glue_stick_difference + marie_paper_packs / paper_pack_ratio

theorem allison_bought_28_items : allison_total_items = 28 := by
  sorry

end NUMINAMATH_CALUDE_allison_bought_28_items_l3455_345526


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l3455_345596

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := (saturday_sandwiches + sunday_sandwiches) * bread_per_sandwich

theorem sally_bread_consumption :
  total_bread = 6 :=
by sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l3455_345596


namespace NUMINAMATH_CALUDE_range_of_a_l3455_345548

/-- Given a real number a, we define the following propositions: -/
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0

def q (x a : ℝ) : Prop := x > a

/-- The main theorem stating the range of values for a -/
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(p x) → ¬(q x a)) →  -- Sufficient condition
  ¬(∀ x, ¬(q x a) → ¬(p x)) →  -- Not necessary condition
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3455_345548


namespace NUMINAMATH_CALUDE_power_sum_integer_l3455_345520

theorem power_sum_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m := by
  sorry

end NUMINAMATH_CALUDE_power_sum_integer_l3455_345520


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_monic_cubic_integer_coeffs_l3455_345582

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 5 (1/3) + 2 →
  x^3 - 6*x^2 + 12*x - 13 = 0 := by sorry

theorem monic_cubic_integer_coeffs :
  ∃ (a b c : ℤ), ∀ (x : ℝ), x^3 - 6*x^2 + 12*x - 13 = x^3 + a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_monic_cubic_integer_coeffs_l3455_345582


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l3455_345598

/-- A proportional function passing through quadrants II and IV -/
theorem proportional_function_quadrants :
  ∀ (x y : ℝ), y = -2 * x →
  (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l3455_345598


namespace NUMINAMATH_CALUDE_expression_value_l3455_345509

theorem expression_value (a b c : ℝ) (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3)) / (a + b) + (b * (c^2 + 3)) / (b + c) + (c * (a^2 + 3)) / (c + a) = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3455_345509


namespace NUMINAMATH_CALUDE_total_fish_l3455_345564

-- Define the number of gold fish and blue fish
def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

-- State the theorem
theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l3455_345564


namespace NUMINAMATH_CALUDE_not_integer_fraction_l3455_345515

theorem not_integer_fraction (a b : ℕ) (ha : a > b) (hb : b > 2) :
  ¬ (∃ k : ℤ, (2^a + 1 : ℤ) = k * (2^b - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_integer_fraction_l3455_345515


namespace NUMINAMATH_CALUDE_triangle_angle_B_l3455_345519

theorem triangle_angle_B (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  3 * a * Real.cos C = 2 * c * Real.cos A →
  Real.tan A = 1 / 3 →
  B = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l3455_345519


namespace NUMINAMATH_CALUDE_solutions_to_equation_l3455_345594

theorem solutions_to_equation : 
  {(m, n) : ℕ × ℕ | 7^m - 3 * 2^n = 1} = {(1, 1), (2, 4)} := by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_l3455_345594


namespace NUMINAMATH_CALUDE_lukes_stickers_l3455_345574

theorem lukes_stickers (initial_stickers birthday_stickers given_to_sister used_on_card final_stickers : ℕ) :
  initial_stickers = 20 →
  birthday_stickers = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  final_stickers = 39 →
  ∃ (bought_stickers : ℕ),
    bought_stickers = 12 ∧
    initial_stickers + birthday_stickers + bought_stickers = final_stickers + given_to_sister + used_on_card :=
by sorry

end NUMINAMATH_CALUDE_lukes_stickers_l3455_345574


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3455_345504

def factory_whistle_period : ℕ := 18
def train_bell_period : ℕ := 30
def foghorn_period : ℕ := 45

def start_time : ℕ := 360  -- 6:00 a.m. in minutes since midnight

theorem next_simultaneous_occurrence :
  ∃ (t : ℕ), t > start_time ∧
  t % factory_whistle_period = 0 ∧
  t % train_bell_period = 0 ∧
  t % foghorn_period = 0 ∧
  t - start_time = 90 :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3455_345504


namespace NUMINAMATH_CALUDE_binomial_60_3_l3455_345595

theorem binomial_60_3 : Nat.choose 60 3 = 57020 := by sorry

end NUMINAMATH_CALUDE_binomial_60_3_l3455_345595


namespace NUMINAMATH_CALUDE_student_calculation_l3455_345523

theorem student_calculation (x : ℕ) (h : x = 121) : 2 * x - 140 = 102 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l3455_345523


namespace NUMINAMATH_CALUDE_house_rent_fraction_l3455_345507

theorem house_rent_fraction (salary : ℝ) (house_rent food conveyance left : ℝ) : 
  food = (3/10) * salary →
  conveyance = (1/8) * salary →
  food + conveyance = 3400 →
  left = 1400 →
  house_rent = salary - (food + conveyance + left) →
  house_rent / salary = 2/5 := by
sorry

end NUMINAMATH_CALUDE_house_rent_fraction_l3455_345507


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l3455_345563

theorem book_arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let math_arrangements : ℕ := Nat.factorial (math_books - 1)
  let english_arrangements : ℕ := Nat.factorial (english_books - 1)
  math_arrangements * english_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l3455_345563


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l3455_345575

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  heart_rate * pace * race_distance

/-- Proves that the athlete's heart beats 28800 times during the 30-mile race. -/
theorem athlete_heartbeats :
  let heart_rate : ℕ := 160  -- heartbeats per minute
  let pace : ℕ := 6          -- minutes per mile
  let race_distance : ℕ := 30 -- miles
  total_heartbeats heart_rate pace race_distance = 28800 := by
  sorry


end NUMINAMATH_CALUDE_athlete_heartbeats_l3455_345575


namespace NUMINAMATH_CALUDE_f_30_value_l3455_345547

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : is_valid_f f) : f 30 = 900 := by
  sorry

end NUMINAMATH_CALUDE_f_30_value_l3455_345547


namespace NUMINAMATH_CALUDE_conference_handshakes_l3455_345553

theorem conference_handshakes (n : ℕ) (h : n = 30) : (n * (n - 1)) / 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3455_345553


namespace NUMINAMATH_CALUDE_limit_of_f_difference_quotient_l3455_345550

def f (x : ℝ) := x^2

theorem limit_of_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f Δx) - (f 0)) / Δx - 0| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_f_difference_quotient_l3455_345550


namespace NUMINAMATH_CALUDE_reflect_x_three_two_l3455_345537

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system. -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The coordinates of (3,2) with respect to the x-axis are (3,-2). -/
theorem reflect_x_three_two :
  reflect_x (3, 2) = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_three_two_l3455_345537


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3455_345599

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 ∧ 
  q^3 - 8*q^2 + 10*q - 3 = 0 ∧ 
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 8/69 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3455_345599


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3455_345581

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Nat := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def AdjacentVertices : Nat := 2

/-- The probability of choosing two distinct adjacent vertices in a decagon -/
theorem decagon_adjacent_vertex_probability : 
  (AdjacentVertices : ℚ) / (Decagon - 1 : ℚ) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertex_probability_l3455_345581


namespace NUMINAMATH_CALUDE_min_inverse_sum_min_inverse_sum_achieved_l3455_345583

theorem min_inverse_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  (1 / x + 1 / y) ≥ 3 / 5 := by
  sorry

theorem min_inverse_sum_achieved (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 12) (h_prod : x * y = 20) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 12 ∧ x * y = 20 ∧ 1 / x + 1 / y = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_min_inverse_sum_min_inverse_sum_achieved_l3455_345583


namespace NUMINAMATH_CALUDE_polynomial_factor_d_value_l3455_345533

theorem polynomial_factor_d_value :
  ∀ d : ℚ,
  (∀ x : ℚ, (3 * x + 4 = 0) → (5 * x^3 + 17 * x^2 + d * x + 28 = 0)) →
  d = 233 / 9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_d_value_l3455_345533


namespace NUMINAMATH_CALUDE_sixteen_points_divide_square_into_ten_equal_triangles_l3455_345597

/-- Represents a point inside a unit square -/
structure PointInSquare where
  x : Real
  y : Real
  inside : 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1

/-- Represents the areas of the four triangles formed by a point and the square's sides -/
structure TriangleAreas where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  sum_is_ten : a₁ + a₂ + a₃ + a₄ = 10
  all_positive : 1 ≤ a₁ ∧ 1 ≤ a₂ ∧ 1 ≤ a₃ ∧ 1 ≤ a₄
  all_at_most_four : a₁ ≤ 4 ∧ a₂ ≤ 4 ∧ a₃ ≤ 4 ∧ a₄ ≤ 4

/-- The main theorem stating that there are exactly 16 points satisfying the condition -/
theorem sixteen_points_divide_square_into_ten_equal_triangles :
  ∃ (points : Finset PointInSquare),
    points.card = 16 ∧
    (∀ p ∈ points, ∃ (areas : TriangleAreas), True) ∧
    (∀ p : PointInSquare, p ∉ points → ¬∃ (areas : TriangleAreas), True) := by
  sorry


end NUMINAMATH_CALUDE_sixteen_points_divide_square_into_ten_equal_triangles_l3455_345597


namespace NUMINAMATH_CALUDE_japanese_turtle_crane_problem_l3455_345522

/-- Represents the number of cranes in the cage. -/
def num_cranes : ℕ := sorry

/-- Represents the number of turtles in the cage. -/
def num_turtles : ℕ := sorry

/-- The total number of heads in the cage. -/
def total_heads : ℕ := 35

/-- The total number of feet in the cage. -/
def total_feet : ℕ := 94

/-- The number of feet a crane has. -/
def crane_feet : ℕ := 2

/-- The number of feet a turtle has. -/
def turtle_feet : ℕ := 4

/-- Theorem stating that the system of equations correctly represents the Japanese turtle and crane problem. -/
theorem japanese_turtle_crane_problem :
  (num_cranes + num_turtles = total_heads) ∧
  (crane_feet * num_cranes + turtle_feet * num_turtles = total_feet) :=
sorry

end NUMINAMATH_CALUDE_japanese_turtle_crane_problem_l3455_345522


namespace NUMINAMATH_CALUDE_power_product_exponent_l3455_345567

theorem power_product_exponent (a b : ℝ) : (a^2 * b^3)^2 = a^4 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_product_exponent_l3455_345567


namespace NUMINAMATH_CALUDE_prudence_sleep_weeks_l3455_345511

/-- Represents Prudence's sleep schedule and total sleep time --/
structure SleepSchedule where
  weekdayNights : Nat  -- Number of weekday nights (Sun-Thurs)
  weekendNights : Nat  -- Number of weekend nights (Fri-Sat)
  napDays : Nat        -- Number of days with naps
  weekdaySleep : Nat   -- Hours of sleep on weekday nights
  weekendSleep : Nat   -- Hours of sleep on weekend nights
  napDuration : Nat    -- Duration of naps in hours
  totalSleep : Nat     -- Total hours of sleep

/-- Calculates the number of weeks required to reach the total sleep time --/
def weeksToReachSleep (schedule : SleepSchedule) : Nat :=
  let weeklySleeep := 
    schedule.weekdayNights * schedule.weekdaySleep +
    schedule.weekendNights * schedule.weekendSleep +
    schedule.napDays * schedule.napDuration
  schedule.totalSleep / weeklySleeep

/-- Theorem: Given Prudence's sleep schedule, it takes 4 weeks to reach 200 hours of sleep --/
theorem prudence_sleep_weeks : 
  weeksToReachSleep {
    weekdayNights := 5,
    weekendNights := 2,
    napDays := 2,
    weekdaySleep := 6,
    weekendSleep := 9,
    napDuration := 1,
    totalSleep := 200
  } = 4 := by
  sorry


end NUMINAMATH_CALUDE_prudence_sleep_weeks_l3455_345511


namespace NUMINAMATH_CALUDE_inequality_proof_l3455_345589

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a * b) ≤ (1 / 3) * Real.sqrt ((a^2 + b^2) / 2) + (2 / 3) * (2 / (1 / a + 1 / b)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3455_345589


namespace NUMINAMATH_CALUDE_train_passing_time_l3455_345534

/-- Proves that a train of given length and speed takes a specific time to pass a stationary point -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 63 →
  passing_time = 16 →
  train_length / (train_speed_kmh * 1000 / 3600) = passing_time := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3455_345534


namespace NUMINAMATH_CALUDE_horse_feeding_amount_l3455_345508

/-- Calculates the amount of food each horse receives at each feeding --/
def food_per_horse_per_feeding (num_horses : ℕ) (feedings_per_day : ℕ) (days : ℕ) (bags_bought : ℕ) (pounds_per_bag : ℕ) : ℕ :=
  (bags_bought * pounds_per_bag) / (num_horses * feedings_per_day * days)

/-- Theorem stating the amount of food each horse receives at each feeding --/
theorem horse_feeding_amount :
  food_per_horse_per_feeding 25 2 60 60 1000 = 20 := by
  sorry

#eval food_per_horse_per_feeding 25 2 60 60 1000

end NUMINAMATH_CALUDE_horse_feeding_amount_l3455_345508


namespace NUMINAMATH_CALUDE_function_identity_l3455_345570

theorem function_identity (f : ℕ → ℕ) : 
  (∀ m n : ℕ, f (m + f n) = f (f m) + f n) → 
  (∀ n : ℕ, f n = n) := by
sorry

end NUMINAMATH_CALUDE_function_identity_l3455_345570


namespace NUMINAMATH_CALUDE_integer_fraction_condition_l3455_345572

theorem integer_fraction_condition (n : ℤ) : 
  (∃ k : ℤ, 16 * (n^2 - n - 1)^2 = k * (2*n - 1)) ↔ 
  n = -12 ∨ n = -2 ∨ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end NUMINAMATH_CALUDE_integer_fraction_condition_l3455_345572


namespace NUMINAMATH_CALUDE_brand_w_households_l3455_345540

theorem brand_w_households (total : ℕ) (neither : ℕ) (both : ℕ) : 
  total = 200 →
  neither = 80 →
  both = 40 →
  ∃ (w b : ℕ), w + b + both + neither = total ∧ b = 3 * both ∧ w = 40 :=
by sorry

end NUMINAMATH_CALUDE_brand_w_households_l3455_345540


namespace NUMINAMATH_CALUDE_range_of_G_l3455_345565

/-- The function G(x) defined as |x+1|-|x-1| for all real x -/
def G (x : ℝ) : ℝ := |x + 1| - |x - 1|

/-- The range of G(x) is [-2,2] -/
theorem range_of_G : Set.range G = Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_range_of_G_l3455_345565


namespace NUMINAMATH_CALUDE_trivia_contest_probability_l3455_345505

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 5

/-- The number of possible answers for each question -/
def num_answers : ℕ := 5

/-- The probability of guessing a single question correctly -/
def p_correct : ℚ := 1 / num_answers

/-- The probability of guessing a single question incorrectly -/
def p_incorrect : ℚ := 1 - p_correct

/-- The probability of guessing all questions incorrectly -/
def p_all_incorrect : ℚ := p_incorrect ^ num_questions

/-- The probability of guessing at least one question correctly -/
def p_at_least_one_correct : ℚ := 1 - p_all_incorrect

theorem trivia_contest_probability :
  p_at_least_one_correct = 2101 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_trivia_contest_probability_l3455_345505


namespace NUMINAMATH_CALUDE_band_members_formation_l3455_345530

theorem band_members_formation :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
  (∃ k : ℕ, n = 10 * k + 4) ∧
  (∃ m : ℕ, n = 12 * m + 6) := by
  sorry

end NUMINAMATH_CALUDE_band_members_formation_l3455_345530


namespace NUMINAMATH_CALUDE_banana_profit_calculation_l3455_345535

/-- Calculates the profit from selling bananas given the purchase and selling rates and the total quantity purchased. -/
theorem banana_profit_calculation 
  (purchase_rate_pounds : ℚ) 
  (purchase_rate_dollars : ℚ) 
  (sell_rate_pounds : ℚ) 
  (sell_rate_dollars : ℚ) 
  (total_pounds : ℚ) : 
  purchase_rate_pounds = 3 →
  purchase_rate_dollars = 1/2 →
  sell_rate_pounds = 4 →
  sell_rate_dollars = 1 →
  total_pounds = 72 →
  (sell_rate_dollars / sell_rate_pounds * total_pounds) - 
  (purchase_rate_dollars / purchase_rate_pounds * total_pounds) = 6 := by
sorry

end NUMINAMATH_CALUDE_banana_profit_calculation_l3455_345535


namespace NUMINAMATH_CALUDE_impossibility_theorem_l3455_345556

theorem impossibility_theorem (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬(((1 - a) * b > 1/4) ∧ ((1 - b) * c > 1/4) ∧ ((1 - c) * a > 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l3455_345556


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3455_345561

theorem polar_to_rectangular_conversion :
  let r : ℝ := 6
  let θ : ℝ := 5 * π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -3 * Real.sqrt 2) ∧ (y = -3 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l3455_345561


namespace NUMINAMATH_CALUDE_cafeteria_pies_l3455_345588

theorem cafeteria_pies (total_apples : ℕ) (handout_percentage : ℚ) (apples_per_pie : ℕ) : 
  total_apples = 800 →
  handout_percentage = 65 / 100 →
  apples_per_pie = 15 →
  (total_apples - (total_apples * handout_percentage).floor) / apples_per_pie = 18 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l3455_345588


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3455_345521

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hc_def : c = (a + b) / 2) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3455_345521


namespace NUMINAMATH_CALUDE_angle_ratio_not_sufficient_for_right_triangle_l3455_345552

theorem angle_ratio_not_sufficient_for_right_triangle 
  (A B C : ℝ) (h_sum : A + B + C = 180) (h_ratio : A / 9 = B / 12 ∧ B / 12 = C / 15) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_sufficient_for_right_triangle_l3455_345552
