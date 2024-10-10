import Mathlib

namespace profit_calculation_l1225_122576

/-- The number of pencils John needs to sell to make a profit of $120 -/
def pencils_to_sell : ℕ := 1200

/-- The cost of buying 5 pencils in dollars -/
def buy_cost : ℚ := 7

/-- The number of pencils John buys at the given cost -/
def buy_quantity : ℕ := 5

/-- The selling price of 4 pencils in dollars -/
def sell_price : ℚ := 6

/-- The number of pencils John sells at the given price -/
def sell_quantity : ℕ := 4

/-- The desired profit in dollars -/
def target_profit : ℚ := 120

/-- Theorem stating that the number of pencils John needs to sell to make a profit of $120 is correct -/
theorem profit_calculation (p : ℕ) (h : p = pencils_to_sell) :
  (p : ℚ) * (sell_price / sell_quantity - buy_cost / buy_quantity) = target_profit :=
sorry

end profit_calculation_l1225_122576


namespace crayons_count_l1225_122526

/-- The number of crayons in a box with specific color relationships -/
def total_crayons (blue : ℕ) : ℕ :=
  let red := 4 * blue
  let green := 2 * red
  let yellow := green / 2
  blue + red + green + yellow

/-- Theorem stating that the total number of crayons is 51 when there are 3 blue crayons -/
theorem crayons_count : total_crayons 3 = 51 := by
  sorry

end crayons_count_l1225_122526


namespace fisherman_catch_l1225_122577

theorem fisherman_catch (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (salmon : ℕ) (pike : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  blue_gill = 2 * bass →
  salmon = bass + bass / 3 →
  pike = (bass + trout + blue_gill + salmon) / 5 →
  bass + trout + blue_gill + salmon + pike = 138 := by
  sorry

end fisherman_catch_l1225_122577


namespace largest_n_satisfying_inequality_l1225_122592

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1/2 : ℚ) + n/9 < 1 ↔ n ≤ 4 :=
by sorry

end largest_n_satisfying_inequality_l1225_122592


namespace triangle_altitude_l1225_122549

/-- Given a rectangle with length 3s and width s, and a triangle inside with one side
    along the diagonal and area half of the rectangle's area, the altitude of the
    triangle to the diagonal base is 3s√10/10 -/
theorem triangle_altitude (s : ℝ) (h : s > 0) :
  let l := 3 * s
  let w := s
  let diagonal := Real.sqrt (l^2 + w^2)
  let rectangle_area := l * w
  let triangle_area := rectangle_area / 2
  triangle_area = (1 / 2) * diagonal * (3 * s * Real.sqrt 10 / 10) :=
by sorry

end triangle_altitude_l1225_122549


namespace cubic_function_property_l1225_122510

/-- Given a cubic function f(x) = mx³ + nx + 1 where mn ≠ 0 and f(-1) = 5, prove that f(1) = 7 -/
theorem cubic_function_property (m n : ℝ) (h1 : m * n ≠ 0) :
  let f := fun x : ℝ => m * x^3 + n * x + 1
  f (-1) = 5 → f 1 = 7 := by
sorry

end cubic_function_property_l1225_122510


namespace circle_points_condition_l1225_122552

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (2, 1)

-- Define the condition for a point being inside or outside the circle
def is_inside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 < 0
def is_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 + a*x - 1 > 0

-- Theorem statement
theorem circle_points_condition (a : ℝ) :
  (is_inside_circle point_A.1 point_A.2 a ∧ is_outside_circle point_B.1 point_B.2 a) ∨
  (is_outside_circle point_A.1 point_A.2 a ∧ is_inside_circle point_B.1 point_B.2 a) →
  -4 < a ∧ a < -2 := by
  sorry

end circle_points_condition_l1225_122552


namespace sqrt_sum_equals_three_sqrt_two_over_two_l1225_122535

theorem sqrt_sum_equals_three_sqrt_two_over_two 
  (a b : ℝ) (h1 : a + b = -6) (h2 : a * b = 8) :
  Real.sqrt (b / a) + Real.sqrt (a / b) = 3 * Real.sqrt 2 / 2 := by
  sorry

end sqrt_sum_equals_three_sqrt_two_over_two_l1225_122535


namespace intersection_point_y_axis_l1225_122567

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem intersection_point_y_axis :
  ∃ (y : ℝ), f 0 = y ∧ (0, y) = (0, -2) := by sorry

end intersection_point_y_axis_l1225_122567


namespace greatest_third_side_length_l1225_122554

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 10 cm. -/
theorem greatest_third_side_length : ℕ :=
  let a : ℝ := 7
  let b : ℝ := 10
  let c : ℝ := 16
  have triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b := by sorry
  have c_less_than_sum : c < a + b := by sorry
  have c_greatest_integer : ∀ n : ℕ, (n : ℝ) > c → (n : ℝ) ≥ a + b := by sorry
  16


end greatest_third_side_length_l1225_122554


namespace magic_card_profit_100_l1225_122522

/-- Calculates the profit from selling a Magic card that has tripled in value --/
def magic_card_profit (purchase_price : ℝ) : ℝ :=
  3 * purchase_price - purchase_price

theorem magic_card_profit_100 :
  magic_card_profit 100 = 200 := by
  sorry

#eval magic_card_profit 100

end magic_card_profit_100_l1225_122522


namespace train_passing_jogger_l1225_122571

/-- Theorem: Train passing jogger
  Given:
  - Jogger's speed: 9 kmph
  - Train's speed: 45 kmph
  - Train's length: 120 meters
  - Initial distance between jogger and train engine: 240 meters
  Prove: The time for the train to pass the jogger is 36 seconds
-/
theorem train_passing_jogger 
  (jogger_speed : Real) 
  (train_speed : Real) 
  (train_length : Real) 
  (initial_distance : Real) 
  (h1 : jogger_speed = 9) 
  (h2 : train_speed = 45) 
  (h3 : train_length = 120) 
  (h4 : initial_distance = 240) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 36 := by
  sorry

end train_passing_jogger_l1225_122571


namespace fermat_number_prime_factor_l1225_122598

theorem fermat_number_prime_factor (n : ℕ) (hn : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end fermat_number_prime_factor_l1225_122598


namespace least_exponent_sum_for_500_l1225_122517

def isPowerOfTwo (n : ℕ) : Prop := ∃ k, n = 2^k

def isDistinctPowersOfTwoSum (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (λ e => 2^e)).sum ∧ 
  exponents.length ≥ 2 ∧
  exponents.Nodup

theorem least_exponent_sum_for_500 :
  ∃ (exponents : List ℕ),
    isDistinctPowersOfTwoSum 500 exponents ∧
    exponents.sum = 32 ∧
    ∀ (other_exponents : List ℕ),
      isDistinctPowersOfTwoSum 500 other_exponents →
      other_exponents.sum ≥ 32 :=
by sorry

end least_exponent_sum_for_500_l1225_122517


namespace student_number_problem_l1225_122580

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 102 → x = 120 := by
  sorry

end student_number_problem_l1225_122580


namespace smallest_solution_quadratic_l1225_122543

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 18 * x - 90 = x * (x + 10)) → x ≥ -9 :=
by sorry

end smallest_solution_quadratic_l1225_122543


namespace ten_player_tournament_decided_in_seven_rounds_l1225_122544

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  rounds : ℕ

/-- The scoring system for the tournament -/
def score_system : ℕ → ℚ
  | 0 => 0     -- Loss
  | 1 => 1/2   -- Draw
  | _ => 1     -- Win

/-- The maximum possible score for a player after a given number of rounds -/
def max_score (t : ChessTournament) : ℚ := t.rounds

/-- The total points distributed after a given number of rounds -/
def total_points (t : ChessTournament) : ℚ := (t.num_players * t.rounds) / 2

/-- A tournament is decided if the maximum score is greater than the average of the remaining points -/
def is_decided (t : ChessTournament) : Prop :=
  max_score t > (total_points t - max_score t) / (t.num_players - 1)

/-- The main theorem: A 10-player tournament is decided after 7 rounds -/
theorem ten_player_tournament_decided_in_seven_rounds :
  let t : ChessTournament := ⟨10, 7⟩
  is_decided t ∧ ∀ r < 7, ¬is_decided ⟨10, r⟩ := by sorry

end ten_player_tournament_decided_in_seven_rounds_l1225_122544


namespace cara_age_is_40_l1225_122579

-- Define the ages as natural numbers
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Theorem statement
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end cara_age_is_40_l1225_122579


namespace buses_meet_at_two_pm_l1225_122585

/-- Represents a bus with departure and arrival times -/
structure Bus where
  departure : ℕ
  arrival : ℕ

/-- The time when two buses meet given their schedules -/
def meeting_time (bus1 bus2 : Bus) : ℕ :=
  sorry

theorem buses_meet_at_two_pm (bus1 bus2 : Bus)
  (h1 : bus1.departure = 11 ∧ bus1.arrival = 16)
  (h2 : bus2.departure = 12 ∧ bus2.arrival = 17) :
  meeting_time bus1 bus2 = 14 :=
sorry

end buses_meet_at_two_pm_l1225_122585


namespace certain_value_calculation_l1225_122582

theorem certain_value_calculation (N : ℝ) : 
  (0.4 * N = 180) → ((1/4) * (1/3) * (2/5) * N = 15) := by
  sorry

end certain_value_calculation_l1225_122582


namespace min_value_of_quadratic_l1225_122566

theorem min_value_of_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), min_y = 9 ∧ ∀ y : ℝ, y = 5 * x^2 - 10 * x + 14 → y ≥ min_y :=
by sorry

end min_value_of_quadratic_l1225_122566


namespace path_equivalence_arrow_sequence_equivalence_l1225_122599

/-- Represents the cyclic pattern of points in the path -/
def cycle_length : ℕ := 5

/-- Maps a point to its equivalent position in the cycle -/
def cycle_position (n : ℕ) : ℕ := n % cycle_length

/-- Theorem: The path from point 520 to 523 is equivalent to 0 to 3 in the cycle -/
theorem path_equivalence : 
  (cycle_position 520 = cycle_position 0) ∧ 
  (cycle_position 523 = cycle_position 3) := by
  sorry

/-- The sequence of arrows from 520 to 523 is the same as 0 to 3 -/
theorem arrow_sequence_equivalence : 
  ∀ (i : ℕ), i < 3 → 
  cycle_position (520 + i) = cycle_position i := by
  sorry

end path_equivalence_arrow_sequence_equivalence_l1225_122599


namespace slower_speed_calculation_l1225_122578

/-- Given a person who walks a certain distance at two different speeds, 
    prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 50) 
  (h2 : faster_speed = 14) 
  (h3 : additional_distance = 20) :
  let total_distance := actual_distance + additional_distance
  let time := total_distance / faster_speed
  let slower_speed := actual_distance / time
  slower_speed = 10 := by
sorry

end slower_speed_calculation_l1225_122578


namespace james_hives_l1225_122590

theorem james_hives (honey_per_hive : ℝ) (jar_capacity : ℝ) (jars_to_buy : ℕ) :
  honey_per_hive = 20 →
  jar_capacity = 0.5 →
  jars_to_buy = 100 →
  (honey_per_hive * (jars_to_buy : ℝ) * jar_capacity) / honey_per_hive = 5 :=
by sorry

end james_hives_l1225_122590


namespace miles_trumpets_l1225_122594

-- Define the number of body parts (as per typical human attributes)
def hands : Nat := 2
def head : Nat := 1
def fingers : Nat := 10

-- Define the number of each instrument based on the conditions
def guitars : Nat := hands + 2
def trombones : Nat := head + 2
def french_horns : Nat := guitars - 1
def trumpets : Nat := fingers - 3

-- Define the total number of instruments
def total_instruments : Nat := 17

-- Theorem to prove
theorem miles_trumpets :
  guitars + trombones + french_horns + trumpets = total_instruments ∧ trumpets = 7 := by
  sorry

end miles_trumpets_l1225_122594


namespace angle_C_is_pi_over_three_max_area_equilateral_l1225_122532

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesConditions (t : Triangle) : Prop :=
  t.c * Real.cos t.B + (t.b - 2 * t.a) * Real.cos t.C = 0 ∧ t.c = 2 * Real.sqrt 3

-- Theorem 1: Angle C is π/3
theorem angle_C_is_pi_over_three (t : Triangle) (h : satisfiesConditions t) : 
  t.C = π / 3 := by sorry

-- Theorem 2: Maximum area is 3√3 and occurs when the triangle is equilateral
theorem max_area_equilateral (t : Triangle) (h : satisfiesConditions t) :
  (∃ (area : ℝ), area = 3 * Real.sqrt 3 ∧ 
    area = (1/2) * t.a * t.b * Real.sin t.C ∧
    t.a = t.b ∧ t.b = t.c) := by sorry

end angle_C_is_pi_over_three_max_area_equilateral_l1225_122532


namespace vector_q_in_terms_of_c_and_d_l1225_122547

/-- Given a line segment CD and points P and Q, where P divides CD internally
    in the ratio 3:5 and Q divides DP externally in the ratio 1:2,
    prove that vector Q can be expressed in terms of vectors C and D. -/
theorem vector_q_in_terms_of_c_and_d
  (C D P Q : EuclideanSpace ℝ (Fin 3))
  (h_P_on_CD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • C + t • D)
  (h_CP_PD : ∃ k : ℝ, k > 0 ∧ dist C P = k * (3 / 8) ∧ dist P D = k * (5 / 8))
  (h_Q_external : ∃ s : ℝ, s < 0 ∧ Q = (1 - s) • D + s • P ∧ abs s = 2) :
  Q = (5 / 8) • C + (-13 / 8) • D :=
by sorry

end vector_q_in_terms_of_c_and_d_l1225_122547


namespace bananas_left_l1225_122587

/-- The number of bananas initially in the jar -/
def initial_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem: The number of bananas left in the jar is 41 -/
theorem bananas_left : initial_bananas - removed_bananas = 41 := by
  sorry

end bananas_left_l1225_122587


namespace vanessa_age_proof_l1225_122595

def guesses : List Nat := [32, 34, 36, 40, 42, 45, 48, 52, 55, 58]

def vanessaAge : Nat := 53

theorem vanessa_age_proof :
  -- At least half of the guesses are too low
  (guesses.filter (· < vanessaAge)).length ≥ guesses.length / 2 ∧
  -- Three guesses are off by one
  (guesses.filter (fun x => x = vanessaAge - 1 ∨ x = vanessaAge + 1)).length = 3 ∧
  -- Vanessa's age is a prime number
  Nat.Prime vanessaAge ∧
  -- One guess is exactly correct
  guesses.contains vanessaAge ∧
  -- Vanessa's age is 53
  vanessaAge = 53 := by
  sorry

#eval vanessaAge

end vanessa_age_proof_l1225_122595


namespace students_in_both_activities_l1225_122509

theorem students_in_both_activities (total : ℕ) (band : ℕ) (sports : ℕ) (either : ℕ) 
  (h1 : total = 320)
  (h2 : band = 85)
  (h3 : sports = 200)
  (h4 : either = 225) :
  band + sports - either = 60 :=
by sorry

end students_in_both_activities_l1225_122509


namespace arithmetic_sequence_ratio_l1225_122563

/-- Two arithmetic sequences a and b with their respective sums A and B -/
def arithmetic_sequences (a b : ℕ → ℚ) (A B : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, 
    (∃ d₁ d₂ : ℚ, ∀ k : ℕ, a (k + 1) = a k + d₁ ∧ b (k + 1) = b k + d₂) ∧
    A n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1) ∧
    B n = n * b 1 + n * (n - 1) / 2 * (b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (A B : ℕ → ℚ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end arithmetic_sequence_ratio_l1225_122563


namespace lcm_24_30_40_50_l1225_122591

theorem lcm_24_30_40_50 : Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 50)) = 600 := by
  sorry

end lcm_24_30_40_50_l1225_122591


namespace complex_calculation_l1225_122593

theorem complex_calculation : (1 - Complex.I)^2 - (4 + 2 * Complex.I) / (1 - 2 * Complex.I) = -4 * Complex.I := by
  sorry

end complex_calculation_l1225_122593


namespace triangle_and_vector_theorem_l1225_122503

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The given condition for the triangle -/
def triangleCondition (t : Triangle) : Prop :=
  (2 * t.a - t.c) * cos t.B = t.b * cos t.C

/-- The vector m -/
def m (A : ℝ) : ℝ × ℝ := (sin A, cos (2 * A))

/-- The vector n -/
def n : ℝ × ℝ := (6, 1)

/-- The dot product of m and n -/
def dotProduct (A : ℝ) : ℝ := 6 * sin A + cos (2 * A)

/-- The main theorem -/
theorem triangle_and_vector_theorem (t : Triangle) 
  (h : triangleCondition t) : 
  t.B = π / 3 ∧ 
  (∀ A, dotProduct A ≤ 5) ∧ 
  (∃ A, dotProduct A = 5) := by
  sorry

end

end triangle_and_vector_theorem_l1225_122503


namespace constant_theta_is_plane_l1225_122528

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the condition θ = c
def constant_theta (c : ℝ) (p : SphericalCoord) : Prop :=
  p.θ = c

-- Define a plane in 3D space
def is_plane (S : Set SphericalCoord) : Prop :=
  ∃ (a b d : ℝ), ∀ (p : SphericalCoord), p ∈ S ↔ 
    a * (p.ρ * Real.sin p.φ * Real.cos p.θ) + 
    b * (p.ρ * Real.sin p.φ * Real.sin p.θ) + 
    d * (p.ρ * Real.cos p.φ) = 0

-- Theorem statement
theorem constant_theta_is_plane (c : ℝ) :
  is_plane {p : SphericalCoord | constant_theta c p} :=
sorry

end constant_theta_is_plane_l1225_122528


namespace angle_in_first_or_third_quadrant_l1225_122574

def is_acute (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

def in_first_quadrant (θ : Real) : Prop :=
  0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2

def in_third_quadrant (θ : Real) : Prop :=
  Real.pi < θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2

theorem angle_in_first_or_third_quadrant (α : Real) (k : Int) 
  (h_acute : is_acute α) :
  in_first_quadrant (k * Real.pi + α) ∨ in_third_quadrant (k * Real.pi + α) := by
  sorry

end angle_in_first_or_third_quadrant_l1225_122574


namespace infinite_m_exist_l1225_122562

/-- A(n) is the number of subsets of {1,2,...,n} with sum of elements divisible by p -/
def A (p : ℕ) (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem infinite_m_exist (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p)
  (h_not_div : ¬(p^2 ∣ (2^(p-1) - 1))) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ (m : ℕ), m ∈ S →
      ∀ (k : ℤ), ∃ (q : ℤ), A p m - k = p * q := by
  sorry

end infinite_m_exist_l1225_122562


namespace f_neg_two_eq_eleven_l1225_122575

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end f_neg_two_eq_eleven_l1225_122575


namespace sum_consecutive_odd_iff_multiple_of_four_l1225_122500

/-- An even composite number can be represented as the sum of consecutive odd numbers
    if and only if it is a multiple of 4. -/
theorem sum_consecutive_odd_iff_multiple_of_four (m : ℕ) :
  (∃ (n : ℕ) (a : ℤ), m = n * (2 * a + n) ∧ n > 1) ↔ 4 ∣ m ∧ m > 2 :=
sorry

end sum_consecutive_odd_iff_multiple_of_four_l1225_122500


namespace real_gdp_change_omega_l1225_122507

/-- Represents the production and price data for Omega in a given year -/
structure YearData where
  vegetable_production : ℕ
  fruit_production : ℕ
  vegetable_price : ℕ
  fruit_price : ℕ

/-- Calculates the nominal GDP for a given year -/
def nominalGDP (data : YearData) : ℕ :=
  data.vegetable_production * data.vegetable_price + data.fruit_production * data.fruit_price

/-- Calculates the real GDP for a given year using base year prices -/
def realGDP (data : YearData) (base : YearData) : ℕ :=
  data.vegetable_production * base.vegetable_price + data.fruit_production * base.fruit_price

/-- Calculates the percentage change in GDP -/
def percentageChange (old : ℕ) (new : ℕ) : ℚ :=
  100 * (new - old : ℚ) / old

/-- The main theorem stating the percentage change in real GDP -/
theorem real_gdp_change_omega :
  let data2014 : YearData := {
    vegetable_production := 1200,
    fruit_production := 750,
    vegetable_price := 90000,
    fruit_price := 75000
  }
  let data2015 : YearData := {
    vegetable_production := 900,
    fruit_production := 900,
    vegetable_price := 100000,
    fruit_price := 70000
  }
  let nominal2014 := nominalGDP data2014
  let real2015 := realGDP data2015 data2014
  let change := percentageChange nominal2014 real2015
  ∃ ε > 0, |change + 9.59| < ε :=
by sorry

end real_gdp_change_omega_l1225_122507


namespace left_handed_jazz_lovers_l1225_122546

/-- Represents a club with members of different handedness and music preferences -/
structure Club where
  total_members : ℕ
  left_handed : ℕ
  jazz_lovers : ℕ
  right_handed_non_jazz : ℕ

/-- Theorem stating the number of left-handed jazz lovers in the club -/
theorem left_handed_jazz_lovers (c : Club)
  (h1 : c.total_members = 20)
  (h2 : c.left_handed = 8)
  (h3 : c.jazz_lovers = 15)
  (h4 : c.right_handed_non_jazz = 2)
  (h5 : c.left_handed + (c.total_members - c.left_handed) = c.total_members) :
  c.left_handed + c.jazz_lovers - c.total_members + c.right_handed_non_jazz = 5 := by
  sorry

#check left_handed_jazz_lovers

end left_handed_jazz_lovers_l1225_122546


namespace rectangle_width_decrease_l1225_122589

theorem rectangle_width_decrease (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end rectangle_width_decrease_l1225_122589


namespace cos_sum_of_complex_exponentials_l1225_122556

theorem cos_sum_of_complex_exponentials (α β : ℝ) : 
  Complex.exp (Complex.I * α) = (4:ℝ)/5 + Complex.I * (3:ℝ)/5 →
  Complex.exp (Complex.I * β) = -(5:ℝ)/13 + Complex.I * (12:ℝ)/13 →
  Real.cos (α + β) = -(7:ℝ)/13 := by sorry

end cos_sum_of_complex_exponentials_l1225_122556


namespace box_volume_increase_l1225_122521

/-- Proves that for a rectangular box with given dimensions, the value of x that
    satisfies the equation for equal volume increase when increasing length or height is 0. -/
theorem box_volume_increase (l w h : ℝ) (x : ℝ) : 
  l = 6 → w = 4 → h = 5 → ((l + x) * w * h = l * w * (h + x)) → x = 0 := by
  sorry

end box_volume_increase_l1225_122521


namespace fraction_equality_l1225_122553

theorem fraction_equality (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end fraction_equality_l1225_122553


namespace tuesday_temperature_l1225_122508

/-- Given the average temperatures for three consecutive days and the temperature of the last day,
    this theorem proves the temperature of the first day. -/
theorem tuesday_temperature
  (avg_tues_wed_thurs : ℝ)
  (avg_wed_thurs_fri : ℝ)
  (temp_friday : ℝ)
  (h1 : avg_tues_wed_thurs = 32)
  (h2 : avg_wed_thurs_fri = 34)
  (h3 : temp_friday = 44) :
  ∃ (temp_tuesday temp_wednesday temp_thursday : ℝ),
    (temp_tuesday + temp_wednesday + temp_thursday) / 3 = avg_tues_wed_thurs ∧
    (temp_wednesday + temp_thursday + temp_friday) / 3 = avg_wed_thurs_fri ∧
    temp_tuesday = 38 := by
  sorry


end tuesday_temperature_l1225_122508


namespace sum_of_naturals_l1225_122542

theorem sum_of_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end sum_of_naturals_l1225_122542


namespace circumscribed_circle_area_l1225_122527

theorem circumscribed_circle_area (s : ℝ) (h : s = 12) :
  let triangle_side := s
  let triangle_height := (Real.sqrt 3 / 2) * triangle_side
  let circle_radius := (2 / 3) * triangle_height
  let circle_area := π * circle_radius ^ 2
  circle_area = 48 * π := by sorry

end circumscribed_circle_area_l1225_122527


namespace square_of_prime_divisibility_l1225_122506

theorem square_of_prime_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (n ∣ p - 1) → 
  (p ∣ n^3 - 1) → 
  ∃ k : ℕ, 4*p - 3 = k^2 :=
sorry

end square_of_prime_divisibility_l1225_122506


namespace imaginary_part_of_z_l1225_122541

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 - i) / (2 * i)) = -1/2 := by
  sorry

end imaginary_part_of_z_l1225_122541


namespace equation_solution_l1225_122520

theorem equation_solution :
  ∃ (y₁ y₂ : ℝ), 
    (4 * (-1)^2 + 3 * y₁^2 + 8 * (-1) - 6 * y₁ + 30 = 50) ∧
    (4 * (-1)^2 + 3 * y₂^2 + 8 * (-1) - 6 * y₂ + 30 = 50) ∧
    (y₁ = 1 + Real.sqrt (29/3)) ∧
    (y₂ = 1 - Real.sqrt (29/3)) := by
  sorry

end equation_solution_l1225_122520


namespace pump_fill_time_solution_l1225_122533

def pump_fill_time (P : ℝ) : Prop :=
  P > 0 ∧ (1 / P - 1 / 14 = 3 / 7)

theorem pump_fill_time_solution :
  ∃ P, pump_fill_time P ∧ P = 2 := by sorry

end pump_fill_time_solution_l1225_122533


namespace min_modulus_of_complex_l1225_122531

theorem min_modulus_of_complex (t : ℝ) : 
  let z : ℂ := (t - 1) + (t + 1) * I
  ∃ (m : ℝ), (∀ t : ℝ, Complex.abs z ≥ m) ∧ (∃ t₀ : ℝ, Complex.abs (((t₀ - 1) : ℂ) + (t₀ + 1) * I) = m) ∧ m = Real.sqrt 2 :=
by sorry

end min_modulus_of_complex_l1225_122531


namespace route_down_length_l1225_122560

/-- Proves that the length of the route down the mountain is 12 miles given the specified conditions. -/
theorem route_down_length (time_up time_down : ℝ) (rate_up : ℝ) (rate_down_factor : ℝ) :
  time_up = time_down →
  rate_down_factor = 1.5 →
  rate_up = 4 →
  time_up = 2 →
  rate_up * time_up = 8 →
  rate_down_factor * rate_up * time_down = 12 :=
by
  sorry

end route_down_length_l1225_122560


namespace shape_division_count_l1225_122536

/-- A shape with 17 cells -/
def Shape : Type := Unit

/-- A rectangle of size 1 × 2 -/
def Rectangle : Type := Unit

/-- A square of size 1 × 1 -/
def Square : Type := Unit

/-- A division of the shape into rectangles and a square -/
def Division : Type := List Rectangle × Square

/-- The number of ways to divide the shape -/
def numDivisions (s : Shape) : ℕ := 10

/-- Theorem: There are 10 ways to divide the shape into 8 rectangles and 1 square -/
theorem shape_division_count (s : Shape) :
  (numDivisions s = 10) ∧
  (∀ d : Division, List.length (d.1) = 8) :=
sorry

end shape_division_count_l1225_122536


namespace absolute_value_equation_solution_l1225_122529

theorem absolute_value_equation_solution :
  ∃! n : ℚ, |n + 4| = 3 - n :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l1225_122529


namespace equation_solution_l1225_122569

theorem equation_solution : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 := by
  sorry

end equation_solution_l1225_122569


namespace certain_number_problem_l1225_122583

theorem certain_number_problem (x : ℝ) : 0.7 * x = (4/5 * 25) + 8 → x = 40 := by
  sorry

end certain_number_problem_l1225_122583


namespace kite_diagonal_sum_less_than_largest_sides_sum_l1225_122523

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite where
  sides : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  side_positive : ∀ i, sides i > 0
  diagonal_positive : ∀ i, diagonals i > 0
  adjacent_equal : sides 0 = sides 1 ∧ sides 2 = sides 3

theorem kite_diagonal_sum_less_than_largest_sides_sum (k : Kite) :
  k.diagonals 0 + k.diagonals 1 < 
  (max (k.sides 0) (k.sides 2)) + (max (k.sides 1) (k.sides 3)) + 
  (min (max (k.sides 0) (k.sides 2)) (max (k.sides 1) (k.sides 3))) :=
sorry

end kite_diagonal_sum_less_than_largest_sides_sum_l1225_122523


namespace quadratic_equation_roots_l1225_122558

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * x^2 + b * x + c = 0) → (
    (|r₁ - r₂| = 1 ∧ max r₁ r₂ = 4) ↔ (a = 1 ∧ b = -7 ∧ c = 12)
  ) := by sorry

end quadratic_equation_roots_l1225_122558


namespace kyro_debt_payment_percentage_l1225_122581

/-- Proves that Kyro paid 80% of her debt to Fernanda given the problem conditions -/
theorem kyro_debt_payment_percentage (aryan_debt : ℝ) (kyro_debt : ℝ) 
  (aryan_payment_percentage : ℝ) (initial_savings : ℝ) (final_savings : ℝ) :
  aryan_debt = 1200 →
  aryan_debt = 2 * kyro_debt →
  aryan_payment_percentage = 0.6 →
  initial_savings = 300 →
  final_savings = 1500 →
  (kyro_debt - (final_savings - initial_savings - aryan_payment_percentage * aryan_debt)) / kyro_debt = 0.2 := by
  sorry

#check kyro_debt_payment_percentage

end kyro_debt_payment_percentage_l1225_122581


namespace divisible_by_35_l1225_122502

theorem divisible_by_35 (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end divisible_by_35_l1225_122502


namespace no_zero_root_l1225_122511

-- Define the three equations
def equation1 (x : ℝ) : Prop := 4 * x^2 - 4 = 36
def equation2 (x : ℝ) : Prop := (2*x + 1)^2 = (x + 2)^2
def equation3 (x : ℝ) : Prop := (x^2 - 9 : ℝ) = x + 2

-- Theorem statement
theorem no_zero_root :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end no_zero_root_l1225_122511


namespace rockville_baseball_league_members_l1225_122557

/-- The cost of a pair of cleats in dollars -/
def cleatCost : ℕ := 6

/-- The additional cost of a jersey compared to cleats in dollars -/
def jerseyAdditionalCost : ℕ := 7

/-- The total cost for all members in dollars -/
def totalCost : ℕ := 3360

/-- The number of sets (cleats and jersey) each member needs -/
def setsPerMember : ℕ := 2

/-- The cost of one set (cleats and jersey) for a member -/
def setCost : ℕ := cleatCost + (cleatCost + jerseyAdditionalCost)

/-- The total cost for one member -/
def memberCost : ℕ := setsPerMember * setCost

/-- The number of members in the Rockville Baseball League -/
def numberOfMembers : ℕ := totalCost / memberCost

theorem rockville_baseball_league_members :
  numberOfMembers = 88 := by sorry

end rockville_baseball_league_members_l1225_122557


namespace trouser_original_price_l1225_122597

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 10 →
  discount_percentage = 90 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 100 :=
by sorry

end trouser_original_price_l1225_122597


namespace triangle_side_length_l1225_122540

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S_ABC : ℝ) :
  a = 4 →
  B = π / 3 →
  S_ABC = 6 * Real.sqrt 3 →
  b = 2 * Real.sqrt 7 :=
by sorry

end triangle_side_length_l1225_122540


namespace exists_special_number_l1225_122565

/-- A function that checks if all digits in a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- A function that swaps two digits in a natural number at given positions -/
def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

theorem exists_special_number :
  ∃ (N : ℕ),
    N % 2020 = 0 ∧
    has_distinct_digits N ∧
    num_digits N = 6 ∧
    ∀ (i j : ℕ), i ≠ j → (swap_digits N i j) % 2020 ≠ 0 ∧
    ∀ (M : ℕ), M % 2020 = 0 → has_distinct_digits M →
      (∀ (i j : ℕ), i ≠ j → (swap_digits M i j) % 2020 ≠ 0) →
      num_digits M ≥ 6 :=
by sorry

end exists_special_number_l1225_122565


namespace rectangle_length_calculation_l1225_122525

theorem rectangle_length_calculation (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 :=
by sorry

end rectangle_length_calculation_l1225_122525


namespace cube_root_of_zero_l1225_122518

theorem cube_root_of_zero (x : ℝ) : x^3 = 0 → x = 0 := by
  sorry

end cube_root_of_zero_l1225_122518


namespace undergrad_sample_count_l1225_122505

/-- Represents the number of undergraduate students in a stratified sample -/
def undergrad_sample_size (total_population : ℕ) (undergrad_population : ℕ) (sample_size : ℕ) : ℕ :=
  (undergrad_population * sample_size) / total_population

/-- Theorem stating the number of undergraduate students in the stratified sample -/
theorem undergrad_sample_count :
  undergrad_sample_size 5600 3000 280 = 150 := by
  sorry

end undergrad_sample_count_l1225_122505


namespace bells_toll_together_l1225_122584

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
sorry

end bells_toll_together_l1225_122584


namespace runners_visibility_probability_l1225_122537

/-- Represents a runner on a circular track -/
structure Runner where
  name : String
  lapTime : ℕ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the circular track -/
structure Track where
  circumference : ℝ
  photoCoverage : ℝ
  shadowInterval : ℕ
  shadowDuration : ℕ

/-- Calculates the probability of both runners being visible in the photo -/
def calculateVisibilityProbability (sarah : Runner) (sam : Runner) (track : Track) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem runners_visibility_probability :
  let sarah : Runner := ⟨"Sarah", 120, true⟩
  let sam : Runner := ⟨"Sam", 100, false⟩
  let track : Track := ⟨1, 1/3, 45, 15⟩
  calculateVisibilityProbability sarah sam track = 1333/6000 := by
  sorry

end runners_visibility_probability_l1225_122537


namespace cookie_boxes_theorem_l1225_122551

theorem cookie_boxes_theorem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ),
    mark_sold = n - 8 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) → 
  n = 9 := by
sorry

end cookie_boxes_theorem_l1225_122551


namespace yan_position_ratio_l1225_122512

/-- Yan's position between home and stadium -/
structure Position where
  home_dist : ℝ     -- distance from home
  stadium_dist : ℝ  -- distance to stadium
  home_dist_nonneg : 0 ≤ home_dist
  stadium_dist_nonneg : 0 ≤ stadium_dist

/-- Yan's walking speed -/
def walking_speed : ℝ := 1

/-- Yan's cycling speed -/
def cycling_speed : ℝ := 4 * walking_speed

theorem yan_position_ratio (pos : Position) : 
  (pos.home_dist / pos.stadium_dist = 3 / 5) ↔ 
  (pos.stadium_dist / walking_speed = 
   pos.home_dist / walking_speed + (pos.home_dist + pos.stadium_dist) / cycling_speed) := by
  sorry

end yan_position_ratio_l1225_122512


namespace nested_function_evaluation_l1225_122555

-- Define the functions a and b
def a (k : ℕ) : ℕ := (k + 1) ^ 2
def b (k : ℕ) : ℕ := k ^ 3 - 2 * k + 1

-- State the theorem
theorem nested_function_evaluation :
  b (a (a (a (a 1)))) = 95877196142432 :=
by sorry

end nested_function_evaluation_l1225_122555


namespace more_seventh_graders_l1225_122539

theorem more_seventh_graders (n m : ℕ) 
  (h1 : n > 0) 
  (h2 : m > 0) 
  (h3 : 7 * n = 6 * m) : 
  m > n :=
by
  sorry

end more_seventh_graders_l1225_122539


namespace cally_shorts_count_l1225_122586

/-- Represents the number of clothing items a person has. -/
structure ClothingItems where
  whiteShirts : Nat
  coloredShirts : Nat
  shorts : Nat
  pants : Nat

/-- Calculate the total number of clothing items. -/
def totalItems (items : ClothingItems) : Nat :=
  items.whiteShirts + items.coloredShirts + items.shorts + items.pants

theorem cally_shorts_count (totalWashed : Nat) (cally : ClothingItems) (danny : ClothingItems)
    (h1 : totalWashed = 58)
    (h2 : cally.whiteShirts = 10)
    (h3 : cally.coloredShirts = 5)
    (h4 : cally.pants = 6)
    (h5 : danny.whiteShirts = 6)
    (h6 : danny.coloredShirts = 8)
    (h7 : danny.shorts = 10)
    (h8 : danny.pants = 6)
    (h9 : totalWashed = totalItems cally + totalItems danny) :
  cally.shorts = 7 := by
  sorry


end cally_shorts_count_l1225_122586


namespace unequal_gender_probability_l1225_122570

/-- The number of children in the family -/
def num_children : ℕ := 6

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The probability of having an unequal number of sons and daughters -/
def unequal_gender_prob : ℚ := 11/16

theorem unequal_gender_probability :
  (1 : ℚ) - (Nat.choose num_children (num_children / 2) : ℚ) / (2 ^ num_children) = unequal_gender_prob :=
sorry

end unequal_gender_probability_l1225_122570


namespace systematic_sampling_theorem_l1225_122514

theorem systematic_sampling_theorem (total_workers : ℕ) (sample_size : ℕ) (start_num : ℕ) (interval_start : ℕ) (interval_end : ℕ) : 
  total_workers = 840 →
  sample_size = 42 →
  start_num = 21 →
  interval_start = 421 →
  interval_end = 720 →
  (interval_end - interval_start + 1) / (total_workers / sample_size) = 15 :=
by sorry

end systematic_sampling_theorem_l1225_122514


namespace fixed_point_coordinates_l1225_122572

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line equation y - 2 = k(x + 1) -/
def lineEquation (k : ℝ) (p : Point) : Prop :=
  p.y - 2 = k * (p.x + 1)

/-- The theorem statement -/
theorem fixed_point_coordinates :
  (∃ M : Point, ∀ k : ℝ, lineEquation k M) →
  ∃ M : Point, M.x = -1 ∧ M.y = 2 :=
by sorry

end fixed_point_coordinates_l1225_122572


namespace root_differences_ratio_l1225_122561

open Real

/-- Given quadratic trinomials and their root differences, prove the ratio of differences squared is 3 -/
theorem root_differences_ratio (a b : ℝ) : 
  let f₁ := fun x : ℝ => x^2 + a*x + 3
  let f₂ := fun x : ℝ => x^2 + 2*x - b
  let f₃ := fun x : ℝ => x^2 + 2*(a-1)*x + b + 6
  let f₄ := fun x : ℝ => x^2 + (4-a)*x - 2*b - 3
  let A := sqrt (a^2 - 12)
  let B := sqrt (4 + 4*b)
  let C := sqrt (4*a^2 - 8*a - 4*b - 20)
  let D := sqrt (a^2 - 8*a + 8*b + 28)
  A^2 ≠ B^2 →
  (C^2 - D^2) / (A^2 - B^2) = 3 :=
by
  sorry


end root_differences_ratio_l1225_122561


namespace yun_lost_paperclips_l1225_122538

theorem yun_lost_paperclips : ∀ (yun_current : ℕ),
  yun_current ≤ 20 →
  (1 + 1/4 : ℚ) * yun_current + 7 = 9 →
  20 - yun_current = 19 :=
by
  sorry

end yun_lost_paperclips_l1225_122538


namespace tangent_line_at_y_axis_l1225_122545

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4) / (x - 2)

theorem tangent_line_at_y_axis (x y : ℝ) :
  (f 0 = -2) →
  (∀ x, deriv f x = (x^2 - 4*x - 4) / (x - 2)^2) →
  (y = -x - 2) ↔ (y - f 0 = deriv f 0 * (x - 0)) :=
by sorry

end tangent_line_at_y_axis_l1225_122545


namespace first_day_charge_l1225_122573

/-- Represents the charge and attendance for a three-day show -/
structure ShowData where
  day1_charge : ℝ
  day2_charge : ℝ
  day3_charge : ℝ
  attendance_ratio : Fin 3 → ℝ
  average_charge : ℝ

/-- Theorem stating the charge on the first day given the show data -/
theorem first_day_charge (s : ShowData)
  (h1 : s.day2_charge = 7.5)
  (h2 : s.day3_charge = 2.5)
  (h3 : s.attendance_ratio 0 = 2)
  (h4 : s.attendance_ratio 1 = 5)
  (h5 : s.attendance_ratio 2 = 13)
  (h6 : s.average_charge = 5)
  (h7 : (s.attendance_ratio 0 * s.day1_charge + 
         s.attendance_ratio 1 * s.day2_charge + 
         s.attendance_ratio 2 * s.day3_charge) / 
        (s.attendance_ratio 0 + s.attendance_ratio 1 + s.attendance_ratio 2) = s.average_charge) :
  s.day1_charge = 15 := by
  sorry

end first_day_charge_l1225_122573


namespace unique_number_theorem_l1225_122568

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Generates the three numbers obtained by replacing one digit with 1 -/
def generateReplacedNumbers (n : ThreeDigitNumber) : List Nat :=
  [100 + 10 * n.tens + n.ones,
   100 * n.hundreds + 10 + n.ones,
   100 * n.hundreds + 10 * n.tens + 1]

/-- The main theorem stating that if the sum of replaced numbers is 1243,
    then the original number must be 566 -/
theorem unique_number_theorem (n : ThreeDigitNumber) :
  (generateReplacedNumbers n).sum = 1243 → n.toNat = 566 := by
  sorry

end unique_number_theorem_l1225_122568


namespace valid_sequences_of_length_20_l1225_122515

/-- Counts valid binary sequences of given length -/
def countValidSequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else countValidSequences (n - 4) + 2 * countValidSequences (n - 5) + countValidSequences (n - 6)

/-- Theorem stating the number of valid sequences of length 20 -/
theorem valid_sequences_of_length_20 :
  countValidSequences 20 = 86 := by sorry

end valid_sequences_of_length_20_l1225_122515


namespace operation_result_l1225_122534

-- Define the set of operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an operation
def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_result (star mul : Operation) 
  (h : apply_op star 12 2 / apply_op mul 9 3 = 2) :
  apply_op star 7 3 / apply_op mul 12 6 = 7 / 6 := by
  sorry

end operation_result_l1225_122534


namespace cupcakes_per_event_l1225_122501

theorem cupcakes_per_event (total_cupcakes : ℕ) (num_events : ℕ) 
  (h1 : total_cupcakes = 768) 
  (h2 : num_events = 8) :
  total_cupcakes / num_events = 96 := by
  sorry

end cupcakes_per_event_l1225_122501


namespace prob_two_cards_two_suits_l1225_122550

/-- The probability of drawing a card of a specific suit from a standard deck -/
def prob_specific_suit : ℚ := 1 / 4

/-- The number of cards drawn -/
def num_draws : ℕ := 6

/-- The number of suits we're interested in -/
def num_suits : ℕ := 2

/-- The number of cards needed from each suit -/
def cards_per_suit : ℕ := 2

/-- The probability of getting the desired outcome when drawing six cards with replacement -/
def prob_desired_outcome : ℚ := (prob_specific_suit ^ (num_draws : ℕ))

theorem prob_two_cards_two_suits :
  prob_desired_outcome = 1 / 4096 := by
  sorry

end prob_two_cards_two_suits_l1225_122550


namespace max_candy_leftover_l1225_122504

theorem max_candy_leftover (x : ℕ) (h1 : x > 120) : 
  ∃ (q : ℕ), x = 12 * (10 + q) + 11 ∧ 
  ∀ (r : ℕ), r < 11 → ∃ (q' : ℕ), x ≠ 12 * (10 + q') + r :=
by sorry

end max_candy_leftover_l1225_122504


namespace soup_feeding_theorem_l1225_122588

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_used_for_children := children_fed / can_capacity.children
  let remaining_cans := total_cans - cans_used_for_children
  remaining_cans * can_capacity.adults

/-- Theorem: Given 7 cans of soup, where each can feeds 4 adults or 7 children,
    if 21 children are fed, the remaining soup can feed 16 adults -/
theorem soup_feeding_theorem :
  let can_capacity : SoupCan := { adults := 4, children := 7 }
  let total_cans : ℕ := 7
  let children_fed : ℕ := 21
  remaining_adults_fed total_cans can_capacity children_fed = 16 := by
  sorry


end soup_feeding_theorem_l1225_122588


namespace unique_positive_integer_solution_l1225_122513

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 3 * x = 1764 := by
  sorry

end unique_positive_integer_solution_l1225_122513


namespace square_side_length_range_l1225_122596

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ side : ℝ, side^2 = area ∧ 3 < side ∧ side < 4 := by
  sorry

end square_side_length_range_l1225_122596


namespace digit_sum_possibilities_l1225_122519

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Predicate to check if four digits are all different -/
def all_different (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- The theorem stating the possible sums of four different digits -/
theorem digit_sum_possibilities (a b c d : Digit) 
  (h : all_different a b c d) :
  (a.val + b.val + c.val + d.val = 10) ∨ 
  (a.val + b.val + c.val + d.val = 18) ∨ 
  (a.val + b.val + c.val + d.val = 19) := by
  sorry

end digit_sum_possibilities_l1225_122519


namespace sum_of_sequences_l1225_122559

def sequence1 : List ℕ := [1, 12, 23, 34, 45]
def sequence2 : List ℕ := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum) = 265 := by
  sorry

end sum_of_sequences_l1225_122559


namespace cats_dogs_ratio_l1225_122530

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
theorem cats_dogs_ratio (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) (num_dogs : ℕ) : 
  cat_ratio * num_dogs = dog_ratio * num_cats → 
  cat_ratio = 3 → 
  dog_ratio = 4 → 
  num_cats = 18 → 
  num_dogs = 24 := by
sorry

end cats_dogs_ratio_l1225_122530


namespace propositions_correctness_l1225_122516

theorem propositions_correctness :
  (∃ a b, a > b ∧ b > 0 ∧ (1 / a ≥ 1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a^2 - a > b^2 - b) ∧
  (∃ a b, a > b ∧ b > 0 ∧ a^3 ≤ b^3) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2*a + b = 1 → 
    (∀ x y, x > 0 ∧ y > 0 ∧ 2*x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
    a^2 + b^2 = 1/9) :=
by sorry

end propositions_correctness_l1225_122516


namespace unique_solution_system_l1225_122524

theorem unique_solution_system (x y : ℝ) : 
  (x - 2*y = 1 ∧ 3*x + 4*y = 23) ↔ (x = 5 ∧ y = 2) := by
sorry

end unique_solution_system_l1225_122524


namespace daughters_age_l1225_122548

/-- Given a mother and daughter whose combined age is 60 years this year,
    and ten years ago the mother's age was seven times the daughter's age,
    prove that the daughter's age this year is 15 years. -/
theorem daughters_age (mother_age daughter_age : ℕ) : 
  mother_age + daughter_age = 60 →
  mother_age - 10 = 7 * (daughter_age - 10) →
  daughter_age = 15 := by
sorry

end daughters_age_l1225_122548


namespace triangle_left_side_value_l1225_122564

/-- Given a triangle with sides L, R, and B satisfying certain conditions, prove that L = 12 -/
theorem triangle_left_side_value (L R B : ℝ) 
  (h1 : L + R + B = 50)
  (h2 : R = L + 2)
  (h3 : B = 24) : 
  L = 12 := by
  sorry

end triangle_left_side_value_l1225_122564
