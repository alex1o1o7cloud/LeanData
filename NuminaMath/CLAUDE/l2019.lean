import Mathlib

namespace NUMINAMATH_CALUDE_determinant_sum_l2019_201943

theorem determinant_sum (x y : ℝ) (h1 : x ≠ y) 
  (h2 : Matrix.det ![![2, 6, 12], ![4, x, y], ![4, y, x]] = 0) : 
  x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_determinant_sum_l2019_201943


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2019_201924

/-- The speed of a boat in still water given its travel distances with and against a stream -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 16) 
  (h2 : against_stream = 6) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 11 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2019_201924


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l2019_201983

/-- Represents the savings from a coupon given a price -/
def CouponSavings (price : ℝ) : (ℝ → ℝ) → ℝ := fun f => f price

/-- Coupon A: 20% off the listed price -/
def CouponA (price : ℝ) : ℝ := 0.2 * price

/-- Coupon B: $50 off the listed price -/
def CouponB (_ : ℝ) : ℝ := 50

/-- Coupon C: 30% off the amount by which the listed price exceeds $120 -/
def CouponC (price : ℝ) : ℝ := 0.3 * (price - 120)

theorem coupon_savings_difference (price_min price_max : ℝ) :
  (price_min > 120) →
  (price_max > 120) →
  (∀ p : ℝ, p ≥ price_min → p ≤ price_max → 
    CouponSavings p CouponA ≥ max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  (∀ p : ℝ, p < price_min ∨ p > price_max → 
    CouponSavings p CouponA < max (CouponSavings p CouponB) (CouponSavings p CouponC)) →
  price_max - price_min = 110 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l2019_201983


namespace NUMINAMATH_CALUDE_total_ants_l2019_201997

def ants_problem (abe beth cece duke emily frances : ℕ) : Prop :=
  abe = 4 ∧
  beth = 2 * abe ∧
  cece = 3 * abe ∧
  duke = abe / 2 ∧
  emily = abe + (75 * abe) / 100 ∧
  frances = 2 * cece ∧
  abe + beth + cece + duke + emily + frances = 57

theorem total_ants : ∃ (abe beth cece duke emily frances : ℕ),
  ants_problem abe beth cece duke emily frances :=
sorry

end NUMINAMATH_CALUDE_total_ants_l2019_201997


namespace NUMINAMATH_CALUDE_prob_nine_successes_possible_l2019_201961

/-- The number of trials -/
def n : ℕ := 10

/-- The success probability -/
def p : ℝ := 0.9

/-- The binomial probability mass function -/
def binomial_pmf (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that the probability of exactly 9 successes is between 0 and 1 -/
theorem prob_nine_successes_possible :
  0 < binomial_pmf 9 ∧ binomial_pmf 9 < 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_nine_successes_possible_l2019_201961


namespace NUMINAMATH_CALUDE_hyperbola_points_m_range_l2019_201957

/-- Given points A(-1, y₁) and B(2, y₂) on the hyperbola y = (3+m)/x with y₁ > y₂, 
    the range of values for m is m < -3 -/
theorem hyperbola_points_m_range (m : ℝ) (y₁ y₂ : ℝ) : 
  y₁ = (3 + m) / (-1) → 
  y₂ = (3 + m) / 2 → 
  y₁ > y₂ → 
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_points_m_range_l2019_201957


namespace NUMINAMATH_CALUDE_count_divisible_numbers_eq_179_l2019_201944

/-- The count of five-digit numbers exactly divisible by 6, 7, 8, and 9 -/
def count_divisible_numbers : ℕ :=
  let lcm := Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 9))
  let lower_bound := ((10000 + lcm - 1) / lcm : ℕ)
  let upper_bound := (99999 / lcm : ℕ)
  upper_bound - lower_bound + 1

theorem count_divisible_numbers_eq_179 : count_divisible_numbers = 179 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_eq_179_l2019_201944


namespace NUMINAMATH_CALUDE_problem_statement_l2019_201948

theorem problem_statement (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, x₁^2 ≥ (1/2)^x₂ - m) → 
  m ≥ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2019_201948


namespace NUMINAMATH_CALUDE_c_wins_probability_l2019_201927

/-- Represents a player in the backgammon tournament -/
inductive Player := | A | B | C

/-- Represents the state of the tournament -/
structure TournamentState where
  lastWinner : Player
  lastLoser : Player

/-- The probability of a player winning a single game -/
def winProbability : ℚ := 1 / 2

/-- The probability of player C winning the tournament -/
def probCWins : ℚ := 2 / 7

/-- Theorem stating that the probability of player C winning the tournament is 2/7 -/
theorem c_wins_probability : 
  probCWins = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_c_wins_probability_l2019_201927


namespace NUMINAMATH_CALUDE_base_representation_comparison_l2019_201937

theorem base_representation_comparison (n : ℕ) (h : n = 1357) :
  (Nat.log 3 n + 1) = (Nat.log 5 n + 1) + (Nat.log 8 n + 1) - 2 :=
by sorry

end NUMINAMATH_CALUDE_base_representation_comparison_l2019_201937


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l2019_201980

/-- Calculates the time required to mow a rectangular lawn -/
theorem lawn_mowing_time 
  (length width : ℝ) 
  (effective_swath : ℝ) 
  (mowing_speed : ℝ) : 
  length = 120 → 
  width = 200 → 
  effective_swath = 2 → 
  mowing_speed = 4000 → 
  (width / effective_swath) * length / mowing_speed = 3 := by
  sorry

#check lawn_mowing_time

end NUMINAMATH_CALUDE_lawn_mowing_time_l2019_201980


namespace NUMINAMATH_CALUDE_inequality_implication_l2019_201936

theorem inequality_implication (x y : ℝ) : x > y → -2*x < -2*y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2019_201936


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2019_201934

theorem arithmetic_and_geometric_sequence (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence condition
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l2019_201934


namespace NUMINAMATH_CALUDE_new_average_age_l2019_201964

def initial_people : ℕ := 8
def initial_average_age : ℚ := 25
def leaving_person_age : ℕ := 20
def remaining_people : ℕ := 7

theorem new_average_age :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l2019_201964


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l2019_201971

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f(-x) = f(x) for all x
variable (h : ∀ x, f (-x) = f x)

-- Define g as the derivative of f
variable (g : ℝ → ℝ)
variable (hg : ∀ x, HasDerivAt f (g x) x)

-- State the theorem
theorem derivative_of_even_function :
  ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l2019_201971


namespace NUMINAMATH_CALUDE_investment_ratio_l2019_201900

/-- Represents the business investment scenario of Krishan and Nandan -/
structure BusinessInvestment where
  nandan_investment : ℝ
  nandan_time : ℝ
  krishan_investment_multiplier : ℝ
  total_gain : ℝ
  nandan_gain : ℝ

/-- The ratio of Krishan's investment to Nandan's investment is 4:1 -/
theorem investment_ratio (b : BusinessInvestment) : 
  b.nandan_time > 0 ∧ 
  b.nandan_investment > 0 ∧ 
  b.total_gain = 26000 ∧ 
  b.nandan_gain = 2000 ∧ 
  b.total_gain = b.nandan_gain + b.krishan_investment_multiplier * b.nandan_investment * (3 * b.nandan_time) →
  b.krishan_investment_multiplier = 4 := by
  sorry

end NUMINAMATH_CALUDE_investment_ratio_l2019_201900


namespace NUMINAMATH_CALUDE_tip_percentage_is_ten_percent_l2019_201988

/-- Calculates the tip percentage given the total bill, number of people, and amount paid per person. -/
def calculate_tip_percentage (total_bill : ℚ) (num_people : ℕ) (amount_per_person : ℚ) : ℚ :=
  let total_paid := num_people * amount_per_person
  let tip_amount := total_paid - total_bill
  (tip_amount / total_bill) * 100

/-- Proves that for a bill of $139.00 split among 8 people, if each pays $19.1125, the tip is 10%. -/
theorem tip_percentage_is_ten_percent :
  calculate_tip_percentage 139 8 (19 + 9/80) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_ten_percent_l2019_201988


namespace NUMINAMATH_CALUDE_circles_intersection_properties_l2019_201913

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x - y = 0

-- Define the perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Define a point P on circle O1
def P : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the distance from a point to a line
def distance_to_line (p : ℝ × ℝ) (l : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Theorem statement
theorem circles_intersection_properties :
  (∀ x y, line_AB x y ↔ x = y) ∧
  (∀ x y, perp_bisector_AB x y ↔ x + y = 1) ∧
  (∃ P, circle_O1 P.1 P.2 ∧ 
    distance_to_line P line_AB = Real.sqrt 2 / 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_properties_l2019_201913


namespace NUMINAMATH_CALUDE_B_equals_roster_l2019_201975

def A : Set Int := {-2, 2, 3, 4}

def B : Set Int := {x | ∃ t ∈ A, x = t^2}

theorem B_equals_roster : B = {4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_B_equals_roster_l2019_201975


namespace NUMINAMATH_CALUDE_delta_flight_price_l2019_201930

theorem delta_flight_price (delta_discount : Real) (united_price : Real) (united_discount : Real) (price_difference : Real) :
  delta_discount = 0.20 →
  united_price = 1100 →
  united_discount = 0.30 →
  price_difference = 90 →
  ∃ (original_delta_price : Real),
    original_delta_price * (1 - delta_discount) = 
    united_price * (1 - united_discount) - price_difference ∧
    original_delta_price = 850 := by
  sorry

end NUMINAMATH_CALUDE_delta_flight_price_l2019_201930


namespace NUMINAMATH_CALUDE_integer_solutions_eq1_integer_solutions_eq2_l2019_201919

-- Equation 1
theorem integer_solutions_eq1 :
  ∀ x y : ℤ, 11 * x + 5 * y = 7 ↔ ∃ t : ℤ, x = 2 - 5 * t ∧ y = -3 + 11 * t :=
sorry

-- Equation 2
theorem integer_solutions_eq2 :
  ∀ x y : ℤ, 4 * x + y = 3 * x * y ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_eq1_integer_solutions_eq2_l2019_201919


namespace NUMINAMATH_CALUDE_line_MN_tangent_to_circle_l2019_201996

-- Define the necessary types
variable (Point Line Circle : Type)

-- Define the necessary relations and functions
variable (on_line : Point → Line → Prop)
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (tangent_line : Line → Circle → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Point)
variable (line_through : Point → Point → Line)

-- Define the given points, lines, and circle
variable (A B C O E F R P N M : Point)
variable (AB AC PR EF MN : Line)
variable (ω : Circle)

-- State the theorem
theorem line_MN_tangent_to_circle (h1 : ¬ on_line A (line_through B C))
  (h2 : center ω = O)
  (h3 : tangent_line AC ω)
  (h4 : tangent_line AB ω)
  (h5 : on_line E AC)
  (h6 : on_line F AB)
  (h7 : on_line R EF)
  (h8 : parallel (line_through O P) EF)
  (h9 : on_line P AB)
  (h10 : N = intersect PR AC)
  (h11 : M = intersect AB (line_through R C))
  (h12 : parallel (line_through R C) AC) :
  tangent_line MN ω :=
sorry

end NUMINAMATH_CALUDE_line_MN_tangent_to_circle_l2019_201996


namespace NUMINAMATH_CALUDE_sum_first_eight_primes_mod_ninth_prime_l2019_201921

def first_nine_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23]

theorem sum_first_eight_primes_mod_ninth_prime : 
  (List.sum (List.take 8 first_nine_primes)) % (List.get! first_nine_primes 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_eight_primes_mod_ninth_prime_l2019_201921


namespace NUMINAMATH_CALUDE_unique_solution_congruence_system_l2019_201951

theorem unique_solution_congruence_system :
  ∀ x y z : ℤ,
  2 ≤ x ∧ x ≤ y ∧ y ≤ z →
  (x * y) % z = 1 →
  (x * z) % y = 1 →
  (y * z) % x = 1 →
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_congruence_system_l2019_201951


namespace NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2019_201929

theorem quadratic_equation_two_distinct_roots : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 3*x₁ + 1 = 0 ∧ x₂^2 - 3*x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_distinct_roots_l2019_201929


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_equation_l2019_201952

theorem perpendicular_lines_slope_equation (k₁ k₂ n : ℝ) : 
  (2 * k₁^2 + 8 * k₁ + n = 0) →
  (2 * k₂^2 + 8 * k₂ + n = 0) →
  (k₁ * k₂ = -1) →
  n = -2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_equation_l2019_201952


namespace NUMINAMATH_CALUDE_distance_after_four_hours_l2019_201922

/-- The distance between two students walking in opposite directions -/
def distance_between_students (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem: The distance between two students walking in opposite directions for 4 hours,
    with speeds of 6 km/hr and 9 km/hr respectively, is 60 km. -/
theorem distance_after_four_hours :
  distance_between_students 6 9 4 = 60 := by
  sorry

#eval distance_between_students 6 9 4

end NUMINAMATH_CALUDE_distance_after_four_hours_l2019_201922


namespace NUMINAMATH_CALUDE_sticks_difference_l2019_201954

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14) 
  (h2 : left = 4) : 
  picked_up - left = 10 := by
sorry

end NUMINAMATH_CALUDE_sticks_difference_l2019_201954


namespace NUMINAMATH_CALUDE_riverside_academy_statistics_l2019_201946

/-- The number of students taking statistics at Riverside Academy -/
def students_taking_statistics (total_students : ℕ) (physics_students : ℕ) (both_subjects : ℕ) : ℕ :=
  total_students - (physics_students - both_subjects)

/-- Theorem: The number of students taking statistics is 21 -/
theorem riverside_academy_statistics :
  let total_students : ℕ := 25
  let physics_students : ℕ := 10
  let both_subjects : ℕ := 6
  students_taking_statistics total_students physics_students both_subjects = 21 := by
  sorry

end NUMINAMATH_CALUDE_riverside_academy_statistics_l2019_201946


namespace NUMINAMATH_CALUDE_rectangle_length_l2019_201928

/-- Given a rectangle where the length is three times the breadth and the area is 6075 square meters,
    prove that the length of the rectangle is 135 meters. -/
theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 6075 → 
  length = 135 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l2019_201928


namespace NUMINAMATH_CALUDE_hannah_savings_l2019_201932

theorem hannah_savings (a₁ : ℕ) (r : ℕ) (n : ℕ) (last_term : ℕ) :
  a₁ = 4 → r = 2 → n = 4 → last_term = 20 →
  (a₁ * (r^n - 1) / (r - 1)) + last_term = 80 := by
  sorry

end NUMINAMATH_CALUDE_hannah_savings_l2019_201932


namespace NUMINAMATH_CALUDE_parking_arrangement_equality_parking_spaces_count_l2019_201967

/-- Number of arrangements of k elements from n elements -/
def A (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of parking spaces -/
def n : ℕ := sorry

/-- Theorem stating the equality of probabilities for different parking arrangements -/
theorem parking_arrangement_equality : A (n - 2) 3 = A 3 2 * A (n - 2) 2 := by sorry

/-- Theorem proving that n equals 10 -/
theorem parking_spaces_count : n = 10 := by sorry

end NUMINAMATH_CALUDE_parking_arrangement_equality_parking_spaces_count_l2019_201967


namespace NUMINAMATH_CALUDE_two_primes_equal_l2019_201920

theorem two_primes_equal (a b c : ℕ) 
  (hp : Nat.Prime (b^c + a))
  (hq : Nat.Prime (a^b + c))
  (hr : Nat.Prime (c^a + b)) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    ((x = b^c + a ∧ y = a^b + c) ∨
     (x = b^c + a ∧ y = c^a + b) ∨
     (x = a^b + c ∧ y = c^a + b)) ∧
    x = y :=
sorry

end NUMINAMATH_CALUDE_two_primes_equal_l2019_201920


namespace NUMINAMATH_CALUDE_intersection_M_N_l2019_201969

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2019_201969


namespace NUMINAMATH_CALUDE_classroom_fundraising_l2019_201984

/-- The amount each classroom needs to raise -/
def classroom_goal : ℕ := 200

/-- The number of families that contributed $10 each -/
def num_families_10 : ℕ := 8

/-- The number of families that contributed $5 each -/
def num_families_5 : ℕ := 10

/-- The contribution from families giving $10 each -/
def contribution_10 : ℕ := 10 * num_families_10

/-- The contribution from families giving $5 each -/
def contribution_5 : ℕ := 5 * num_families_5

/-- The amount still needed to reach the goal -/
def amount_needed : ℕ := 30

/-- The number of families with unknown contribution -/
def num_unknown_families : ℕ := 2

theorem classroom_fundraising (x : ℕ) : 
  x * num_unknown_families + contribution_10 + contribution_5 = classroom_goal - amount_needed →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_classroom_fundraising_l2019_201984


namespace NUMINAMATH_CALUDE_exponential_regression_model_l2019_201940

/-- Given a model y = ce^(kx) and a linear equation z = 0.3x + 4 where z = ln y,
    prove that c = e^4 and k = 0.3 -/
theorem exponential_regression_model (c k : ℝ) :
  (∀ x y : ℝ, y = c * Real.exp (k * x)) →
  (∀ x z : ℝ, z = 0.3 * x + 4) →
  (∀ y : ℝ, z = Real.log y) →
  c = Real.exp 4 ∧ k = 0.3 := by
sorry

end NUMINAMATH_CALUDE_exponential_regression_model_l2019_201940


namespace NUMINAMATH_CALUDE_evelyn_initial_skittles_l2019_201958

/-- The number of Skittles Evelyn shared with Christine -/
def shared_skittles : ℕ := 72

/-- The number of Skittles Evelyn had left after sharing -/
def remaining_skittles : ℕ := 4

/-- The initial number of Skittles Evelyn had -/
def initial_skittles : ℕ := shared_skittles + remaining_skittles

theorem evelyn_initial_skittles : initial_skittles = 76 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_initial_skittles_l2019_201958


namespace NUMINAMATH_CALUDE_small_prob_event_cannot_occur_is_false_l2019_201945

-- Define a probability space
variable (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)

-- Define an event as a measurable set
def Event (Ω : Type) [MeasurableSpace Ω] := {A : Set Ω // MeasurableSet A}

-- Define a very small probability
def VerySmallProbability (ε : ℝ) : Prop := 0 < ε ∧ ε < 1/1000000

-- Statement: An event with a very small probability cannot occur
theorem small_prob_event_cannot_occur_is_false :
  ∃ (A : Event Ω) (ε : ℝ), VerySmallProbability ε ∧ P A < ε ∧ ¬(P A = 0) :=
sorry

end NUMINAMATH_CALUDE_small_prob_event_cannot_occur_is_false_l2019_201945


namespace NUMINAMATH_CALUDE_expression_evaluation_l2019_201985

theorem expression_evaluation : -1^2008 + (-1)^2009 + 1^2010 - 1^2011 = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2019_201985


namespace NUMINAMATH_CALUDE_fish_catching_average_l2019_201904

theorem fish_catching_average (aang_fish sokka_fish toph_fish : ℕ) 
  (h1 : aang_fish = 7)
  (h2 : sokka_fish = 5)
  (h3 : toph_fish = 12) :
  (aang_fish + sokka_fish + toph_fish) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_catching_average_l2019_201904


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2019_201911

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∀ x : ℝ, x^2 + a*x + b = 0 → (2*x)^2 + b*(2*x) + c = 0) →
  a / c = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2019_201911


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2019_201999

theorem sqrt_sum_equals_seven (y : ℝ) 
  (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) : 
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l2019_201999


namespace NUMINAMATH_CALUDE_sphere_radius_in_unit_cube_l2019_201938

/-- The radius of a sphere satisfying specific conditions in a unit cube -/
theorem sphere_radius_in_unit_cube : ∃ r : ℝ,
  (r > 0) ∧ 
  (r^4 - 4*r^3 + 6*r^2 - 8*r + 4 = 0) ∧
  ((0 - r)^2 + (0 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through A(0,0,0)
  ((1 - r)^2 + (1 - r)^2 + (0 - (1 - r))^2 = r^2) ∧ -- Sphere passes through C(1,1,0)
  ((1 - r)^2 + (0 - r)^2 = r^2) ∧                   -- Sphere touches edge through B(1,0,0)
  (1 - (1 - r) = r)                                 -- Sphere touches top face (z=1)
  := by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_unit_cube_l2019_201938


namespace NUMINAMATH_CALUDE_dan_destroyed_balloons_l2019_201989

/-- The number of red balloons destroyed by Dan -/
def balloons_destroyed (fred_balloons sam_balloons remaining_balloons : ℝ) : ℝ :=
  fred_balloons + sam_balloons - remaining_balloons

theorem dan_destroyed_balloons :
  balloons_destroyed 10.0 46.0 40 = 16.0 := by
  sorry

end NUMINAMATH_CALUDE_dan_destroyed_balloons_l2019_201989


namespace NUMINAMATH_CALUDE_geoffrey_money_left_l2019_201947

-- Define the given amounts
def grandmother_gift : ℕ := 20
def aunt_gift : ℕ := 25
def uncle_gift : ℕ := 30
def total_money : ℕ := 125
def game_cost : ℕ := 35
def num_games : ℕ := 3

-- Theorem to prove
theorem geoffrey_money_left :
  total_money - (grandmother_gift + aunt_gift + uncle_gift + num_games * game_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_money_left_l2019_201947


namespace NUMINAMATH_CALUDE_exists_palindromic_product_l2019_201973

/-- A natural number is palindromic in base 10 if it reads the same forward and backward. -/
def IsPalindromic (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (λ acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number not divisible by 10, there exists another natural number
    such that their product is palindromic in base 10. -/
theorem exists_palindromic_product (x : ℕ) (hx : ¬ 10 ∣ x) :
  ∃ y : ℕ, IsPalindromic (x * y) := by
  sorry

end NUMINAMATH_CALUDE_exists_palindromic_product_l2019_201973


namespace NUMINAMATH_CALUDE_last_colored_cell_position_l2019_201979

/-- Represents a position in the grid --/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the dimensions of the rectangle --/
structure Dimensions :=
  (width : Nat)
  (height : Nat)

/-- Represents the direction of movement in the spiral --/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Function to determine the next position in the spiral --/
def nextPosition (pos : Position) (dir : Direction) : Position :=
  match dir with
  | Direction.Right => { row := pos.row,     col := pos.col + 1 }
  | Direction.Down  => { row := pos.row + 1, col := pos.col }
  | Direction.Left  => { row := pos.row,     col := pos.col - 1 }
  | Direction.Up    => { row := pos.row - 1, col := pos.col }

/-- Function to determine if a position is within the rectangle --/
def isWithinBounds (pos : Position) (dim : Dimensions) : Bool :=
  pos.row ≥ 1 && pos.row ≤ dim.height && pos.col ≥ 1 && pos.col ≤ dim.width

/-- Theorem stating that the last colored cell in a 200x100 rectangle,
    colored in a spiral pattern, is at position (51, 50) --/
theorem last_colored_cell_position :
  ∃ (coloringProcess : Nat → Position),
    (coloringProcess 0 = { row := 1, col := 1 }) →
    (∀ n, isWithinBounds (coloringProcess n) { width := 200, height := 100 }) →
    (∀ n, ∃ dir, nextPosition (coloringProcess n) dir = coloringProcess (n + 1)) →
    (∃ lastStep, ∀ m > lastStep, ¬isWithinBounds (coloringProcess m) { width := 200, height := 100 }) →
    (coloringProcess lastStep = { row := 51, col := 50 }) :=
by sorry


end NUMINAMATH_CALUDE_last_colored_cell_position_l2019_201979


namespace NUMINAMATH_CALUDE_find_a_l2019_201986

/-- The value of a that satisfies the given inequality system -/
def a : ℝ := 4

/-- The system of inequalities -/
def inequality_system (x a : ℝ) : Prop :=
  2 * x + 1 > 3 ∧ a - x > 1

/-- The solution set of the inequality system -/
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x < 3

theorem find_a :
  (∀ x, inequality_system x a ↔ solution_set x) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_find_a_l2019_201986


namespace NUMINAMATH_CALUDE_problem_statement_l2019_201960

theorem problem_statement (x y : ℝ) 
  (hx : x > 4) 
  (hy : y > 9) 
  (h : (Real.log x / Real.log 4)^4 + (Real.log y / Real.log 9)^4 + 18 = 18 * (Real.log x / Real.log 4) * (Real.log y / Real.log 9)) : 
  x^2 + y^2 = 4^(2 * Real.sqrt 3) + 9^(2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2019_201960


namespace NUMINAMATH_CALUDE_find_k_l2019_201941

theorem find_k : ∃ k : ℚ, (2 * 2 - 3 * k * (-1) = 1) ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2019_201941


namespace NUMINAMATH_CALUDE_modular_inverse_40_mod_61_l2019_201925

theorem modular_inverse_40_mod_61 :
  (∃ x : ℤ, 21 * x ≡ 1 [ZMOD 61] ∧ x ≡ 15 [ZMOD 61]) →
  (∃ y : ℤ, 40 * y ≡ 1 [ZMOD 61] ∧ y ≡ 46 [ZMOD 61]) :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_40_mod_61_l2019_201925


namespace NUMINAMATH_CALUDE_triangle_problem_l2019_201987

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  (4 * a * Real.cos B = c^2 - 4 * b * Real.cos A) →
  (C = π / 3) →
  (a + b = 4 * Real.sqrt 2) →
  -- Conclusions
  (c = 4) ∧
  (1/2 * a * b * Real.sin C = (4 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l2019_201987


namespace NUMINAMATH_CALUDE_valid_basis_l2019_201923

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (3, -1)
def a : ℝ × ℝ := (3, 4)

theorem valid_basis :
  ∃ (x y : ℝ), x • e₁ + y • e₂ = a ∧ ¬(∃ (k : ℝ), e₁ = k • e₂) :=
sorry

end NUMINAMATH_CALUDE_valid_basis_l2019_201923


namespace NUMINAMATH_CALUDE_more_women_than_men_l2019_201963

theorem more_women_than_men (total : ℕ) (ratio : ℚ) : 
  total = 18 → ratio = 7/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 4 :=
sorry

end NUMINAMATH_CALUDE_more_women_than_men_l2019_201963


namespace NUMINAMATH_CALUDE_right_triangles_count_l2019_201998

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a line of points -/
def Line := List Point

/-- Creates a line of points with given y-coordinate -/
def createLine (y : ℕ) : Line :=
  List.map (fun x => ⟨x, y⟩) (List.range 73)

/-- Checks if three points form a right triangle -/
def isRightTriangle (p1 p2 p3 : Point) : Bool :=
  -- Implementation omitted for brevity
  sorry

/-- Counts the number of right triangles formed by points from two lines -/
def countRightTriangles (line1 line2 : Line) : ℕ :=
  -- Implementation omitted for brevity
  sorry

/-- The main theorem to prove -/
theorem right_triangles_count :
  let line1 := createLine 3
  let line2 := createLine 4
  countRightTriangles line1 line2 = 10654 := by
  sorry

end NUMINAMATH_CALUDE_right_triangles_count_l2019_201998


namespace NUMINAMATH_CALUDE_wendys_final_tally_l2019_201908

/-- Calculates Wendy's final point tally for recycling --/
def wendys_points (cans_points newspaper_points cans_total cans_recycled newspapers_recycled penalty_points bonus_points bonus_cans_threshold bonus_newspapers_threshold : ℕ) : ℕ :=
  let points_earned := cans_points * cans_recycled + newspaper_points * newspapers_recycled
  let points_lost := penalty_points * (cans_total - cans_recycled)
  let bonus := if cans_recycled ≥ bonus_cans_threshold ∧ newspapers_recycled ≥ bonus_newspapers_threshold then bonus_points else 0
  points_earned - points_lost + bonus

/-- Theorem stating that Wendy's final point tally is 69 --/
theorem wendys_final_tally :
  wendys_points 5 10 11 9 3 3 15 10 2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_wendys_final_tally_l2019_201908


namespace NUMINAMATH_CALUDE_boys_speed_l2019_201955

/-- The speed of a boy traveling from home to school on the first day, given certain conditions. -/
theorem boys_speed (distance : ℝ) (late_time : ℝ) (early_time : ℝ) (second_day_speed : ℝ) : 
  distance = 2.5 ∧ 
  late_time = 7 / 60 ∧ 
  early_time = 8 / 60 ∧ 
  second_day_speed = 10 → 
  ∃ (first_day_speed : ℝ), first_day_speed = 9.375 := by
  sorry

#eval (9.375 : Float)

end NUMINAMATH_CALUDE_boys_speed_l2019_201955


namespace NUMINAMATH_CALUDE_additional_people_for_faster_mowing_l2019_201906

/-- Represents the number of people needed to mow a lawn in a given time -/
structure LawnMowing where
  people : ℕ
  hours : ℕ

/-- The work rate (people × hours) for mowing the lawn -/
def workRate (l : LawnMowing) : ℕ := l.people * l.hours

theorem additional_people_for_faster_mowing 
  (initial : LawnMowing) 
  (target : LawnMowing) 
  (h1 : initial.people = 4) 
  (h2 : initial.hours = 6) 
  (h3 : target.hours = 3) 
  (h4 : workRate initial = workRate target) : 
  target.people - initial.people = 4 := by
  sorry

end NUMINAMATH_CALUDE_additional_people_for_faster_mowing_l2019_201906


namespace NUMINAMATH_CALUDE_round_trip_speed_l2019_201914

/-- Given a round trip between two points A and B, this theorem proves
    that if the distance is 120 miles, the speed from A to B is 60 mph,
    and the average speed for the entire trip is 45 mph, then the speed
    from B to A must be 36 mph. -/
theorem round_trip_speed (d : ℝ) (v_ab : ℝ) (v_avg : ℝ) (v_ba : ℝ) : 
  d = 120 → v_ab = 60 → v_avg = 45 → 
  (2 * d) / (d / v_ab + d / v_ba) = v_avg →
  v_ba = 36 := by
  sorry

#check round_trip_speed

end NUMINAMATH_CALUDE_round_trip_speed_l2019_201914


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2019_201918

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (2 + a) * x + 2 * a > 0}
  (a < 2 → S = {x : ℝ | x < a ∨ x > 2}) ∧
  (a = 2 → S = {x : ℝ | x ≠ 2}) ∧
  (a > 2 → S = {x : ℝ | x > a ∨ x < 2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2019_201918


namespace NUMINAMATH_CALUDE_remainder_3_167_mod_11_l2019_201977

theorem remainder_3_167_mod_11 : 3^167 % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_167_mod_11_l2019_201977


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l2019_201907

/-- Converts a number from base 5 to base 10 -/
def base5To10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The loot values in base 5 -/
def jewelry : List Nat := [4, 2, 1, 3]
def goldCoins : List Nat := [2, 2, 1, 3]
def rubbingAlcohol : List Nat := [4, 2, 1]

theorem pirate_loot_sum :
  base5To10 jewelry + base5To10 goldCoins + base5To10 rubbingAlcohol = 865 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l2019_201907


namespace NUMINAMATH_CALUDE_computer_pricing_l2019_201909

theorem computer_pricing (C : ℝ) : 
  C + 0.60 * C = 2560 → C + 0.40 * C = 2240 := by sorry

end NUMINAMATH_CALUDE_computer_pricing_l2019_201909


namespace NUMINAMATH_CALUDE_gcd_547_323_l2019_201978

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_547_323_l2019_201978


namespace NUMINAMATH_CALUDE_calculate_expression_l2019_201966

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2019_201966


namespace NUMINAMATH_CALUDE_tan_15_degrees_l2019_201976

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_degrees_l2019_201976


namespace NUMINAMATH_CALUDE_production_average_problem_l2019_201990

theorem production_average_problem (n : ℕ) : 
  (∀ (past_total : ℕ), past_total = n * 50 →
   (past_total + 90) / (n + 1) = 58) →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_production_average_problem_l2019_201990


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l2019_201972

/-- Given a circle D defined by the equation x^2 - 20x + y^2 + 6y + 25 = 0,
    prove that the sum of its center coordinates and radius is 7 + √66 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), x^2 - 20*x + y^2 + 6*y + 25 = 0 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = 7 + Real.sqrt 66 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l2019_201972


namespace NUMINAMATH_CALUDE_distance_to_point_l2019_201959

theorem distance_to_point : Real.sqrt (9^2 + (-40)^2) = 41 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l2019_201959


namespace NUMINAMATH_CALUDE_proposition_and_related_l2019_201931

theorem proposition_and_related (a b : ℝ) : 
  (a + b = 1 → a * b ≤ 1/4) ∧ 
  (a * b > 1/4 → a + b ≠ 1) ∧ 
  ¬(a * b ≤ 1/4 → a + b = 1) ∧ 
  ¬(a + b ≠ 1 → a * b > 1/4) := by
sorry

end NUMINAMATH_CALUDE_proposition_and_related_l2019_201931


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l2019_201991

/-- The squared distance from the center to a focus of a hyperbola -/
def hyperbola_c_squared (a b : ℝ) : ℝ := a^2 + b^2

/-- The squared distance from the center to a focus of an ellipse -/
def ellipse_c_squared (a b : ℝ) : ℝ := a^2 - b^2

theorem ellipse_hyperbola_foci_coincide :
  let ellipse_a_squared : ℝ := 16
  let hyperbola_a_squared : ℝ := 144 / 25
  let hyperbola_b_squared : ℝ := 81 / 25
  ∀ b_squared : ℝ,
    hyperbola_c_squared hyperbola_a_squared hyperbola_b_squared =
    ellipse_c_squared ellipse_a_squared b_squared →
    b_squared = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l2019_201991


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2019_201965

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := Complex.mk (a^2 - a - 2) (a + 1)
  (z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2019_201965


namespace NUMINAMATH_CALUDE_profit_distribution_l2019_201915

theorem profit_distribution (share_a share_b share_c : ℕ) (total_profit : ℕ) : 
  share_a + share_b + share_c = total_profit →
  2 * share_a = 3 * share_b →
  3 * share_b = 5 * share_c →
  share_c - share_b = 4000 →
  total_profit = 20000 := by
sorry

end NUMINAMATH_CALUDE_profit_distribution_l2019_201915


namespace NUMINAMATH_CALUDE_find_x_l2019_201912

theorem find_x (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a) ^ (2 * b) = (a ^ b * x ^ b) ^ 2 → x = 4 := by
sorry

end NUMINAMATH_CALUDE_find_x_l2019_201912


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l2019_201953

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) > a n

theorem geometric_sequence_proof (a : ℕ → ℝ) 
  (h1 : geometric_sequence a)
  (h2 : increasing_sequence a)
  (h3 : a 5 ^ 2 = a 10)
  (h4 : ∀ (n : ℕ), 2 * (a n + a (n + 2)) = 5 * a (n + 1)) :
  ∀ (n : ℕ), a n = 2^n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l2019_201953


namespace NUMINAMATH_CALUDE_solve_star_equation_l2019_201916

/-- Custom binary operation -/
def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

/-- Theorem statement -/
theorem solve_star_equation :
  ∀ y : ℝ, star 4 y = 80 → y = 84 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_star_equation_l2019_201916


namespace NUMINAMATH_CALUDE_basketball_team_selection_count_l2019_201981

/-- The number of ways to select a basketball team lineup with specific roles. -/
theorem basketball_team_selection_count :
  let total_members : ℕ := 15
  let leadership_material : ℕ := 6
  let positions_to_fill : ℕ := 5
  
  -- Number of ways to select captain and vice-captain
  let leadership_selection : ℕ := leadership_material * (leadership_material - 1)
  
  -- Number of ways to select 5 position players from remaining members
  let position_selection : ℕ := 
    (total_members - 2) * (total_members - 3) * (total_members - 4) * 
    (total_members - 5) * (total_members - 6)
  
  leadership_selection * position_selection = 3326400 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_count_l2019_201981


namespace NUMINAMATH_CALUDE_product_equals_2010_l2019_201968

def sequence_product (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (n + 1 : ℚ) / n * sequence_product (n - 1)

theorem product_equals_2010 :
  sequence_product 2009 = 2010 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_2010_l2019_201968


namespace NUMINAMATH_CALUDE_jake_audrey_ball_difference_l2019_201933

theorem jake_audrey_ball_difference :
  ∀ (jake_balls audrey_balls : ℕ),
    jake_balls = 7 →
    audrey_balls = 41 →
    audrey_balls - jake_balls = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_jake_audrey_ball_difference_l2019_201933


namespace NUMINAMATH_CALUDE_a_5_equals_9_l2019_201974

-- Define the sequence and its sum
def S (n : ℕ) := n^2

-- Define the general term of the sequence
def a (n : ℕ) : ℕ := S n - S (n-1)

-- Theorem statement
theorem a_5_equals_9 : a 5 = 9 := by sorry

end NUMINAMATH_CALUDE_a_5_equals_9_l2019_201974


namespace NUMINAMATH_CALUDE_quadrilateral_fixed_point_theorem_l2019_201950

-- Define the plane
variable (Plane : Type)

-- Define points in the plane
variable (Point : Type)
variable (A B C D P : Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

-- Define the angle function
variable (angle : Point → Point → Point → ℝ)

-- Define the line through two points
variable (line_through : Point → Point → Set Point)

-- Define the "lies on" relation
variable (lies_on : Point → Set Point → Prop)

-- Theorem statement
theorem quadrilateral_fixed_point_theorem :
  ∃ P : Point,
    ∀ C D : Point,
      distance A B = distance B C →
      distance A D = distance D C →
      angle A D C = Real.pi / 2 →
      lies_on P (line_through C D) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_fixed_point_theorem_l2019_201950


namespace NUMINAMATH_CALUDE_female_employees_count_l2019_201905

/-- Represents the number of employees in a company -/
structure Company where
  total_employees : ℕ
  female_managers : ℕ
  male_employees : ℕ
  female_employees : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 200 ∧
  c.total_employees * 2 = (c.female_managers + (c.male_employees * 2 / 5)) * 5 ∧
  c.total_employees = c.male_employees + c.female_employees

/-- Theorem stating that under the given conditions, the number of female employees is 500 -/
theorem female_employees_count (c : Company) :
  company_conditions c → c.female_employees = 500 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l2019_201905


namespace NUMINAMATH_CALUDE_simplify_expression_l2019_201926

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  let M := a - b
  (2 * a) / (a^2 - b^2) - 1 / M = 1 / (a + b) := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2019_201926


namespace NUMINAMATH_CALUDE_hockey_league_games_l2019_201970

/-- Represents a hockey league with two divisions -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games played in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let intra_games := league.divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_games := league.divisions * league.teams_per_division * league.teams_per_division * league.inter_division_games
  intra_games + inter_games

/-- Theorem stating that the total number of games in the described hockey league is 192 -/
theorem hockey_league_games :
  let league : HockeyLeague := {
    divisions := 2,
    teams_per_division := 6,
    intra_division_games := 4,
    inter_division_games := 2
  }
  total_games league = 192 := by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l2019_201970


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2019_201910

/-- Given two vectors in 2D Euclidean space with specific magnitudes and angle between them,
    prove that the magnitude of their difference is √7. -/
theorem vector_difference_magnitude
  (a b : ℝ × ℝ)  -- Two vectors in 2D real space
  (h1 : ‖a‖ = 2)  -- Magnitude of a is 2
  (h2 : ‖b‖ = 3)  -- Magnitude of b is 3
  (h3 : a • b = 3)  -- Dot product of a and b (equivalent to 60° angle)
  : ‖a - b‖ = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2019_201910


namespace NUMINAMATH_CALUDE_smith_family_buffet_cost_l2019_201994

/-- Calculates the total cost of a family buffet given the pricing structure and family composition. -/
def familyBuffetCost (adultPrice childPrice : ℚ) (seniorDiscount : ℚ) 
  (numAdults numChildren numSeniors : ℕ) : ℚ :=
  (numAdults : ℚ) * adultPrice + 
  (numChildren : ℚ) * childPrice + 
  (numSeniors : ℚ) * (adultPrice * (1 - seniorDiscount))

/-- Theorem stating that Mr. Smith's family buffet cost is $162 -/
theorem smith_family_buffet_cost : 
  familyBuffetCost 30 15 (1/10) 3 3 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_buffet_cost_l2019_201994


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2019_201993

theorem quadratic_roots_product (a b : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + b = 0 → (x = a ∨ x = b)) → 
  a + b = 5 → 
  a * b = 6 → 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2019_201993


namespace NUMINAMATH_CALUDE_max_k_for_f_greater_than_k_l2019_201982

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + x * Real.log x + b

theorem max_k_for_f_greater_than_k (b : ℝ) :
  (∀ x > 0, (3 * x - f x 1 - 4 = 0)) →
  (∃ k : ℤ, ∀ x > 0, f x b > k) →
  (∀ k : ℤ, (∀ x > 0, f x b > k) → k ≤ -3) ∧
  (∀ x > 0, f x b > -3) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_f_greater_than_k_l2019_201982


namespace NUMINAMATH_CALUDE_two_cos_sixty_degrees_l2019_201902

theorem two_cos_sixty_degrees : 2 * Real.cos (π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_cos_sixty_degrees_l2019_201902


namespace NUMINAMATH_CALUDE_trapezoid_area_l2019_201903

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  let total_trapezoid_area := outer_area - inner_area
  let num_trapezoids := 4
  (total_trapezoid_area / num_trapezoids : ℝ) = 8 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2019_201903


namespace NUMINAMATH_CALUDE_tv_count_indeterminate_l2019_201956

structure GroupInfo where
  total : ℕ
  married : ℕ
  radio : ℕ
  ac : ℕ
  tv_radio_ac_married : ℕ

def has_tv (info : GroupInfo) : Set ℕ :=
  { n | n ≥ info.tv_radio_ac_married ∧ n ≤ info.total }

theorem tv_count_indeterminate (info : GroupInfo) 
  (h_total : info.total = 100)
  (h_married : info.married = 81)
  (h_radio : info.radio = 85)
  (h_ac : info.ac = 70)
  (h_tram : info.tv_radio_ac_married = 11) :
  ∃ (n : ℕ), n ∈ has_tv info ∧ 
  ∀ (m : ℕ), m ≠ n → (m ∈ has_tv info ↔ n ∈ has_tv info) :=
sorry

end NUMINAMATH_CALUDE_tv_count_indeterminate_l2019_201956


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_product_8_and_even_digit_l2019_201917

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def has_even_digit (n : ℕ) : Prop :=
  (n / 100) % 2 = 0 ∨ ((n / 10) % 10) % 2 = 0 ∨ (n % 10) % 2 = 0

theorem smallest_three_digit_with_product_8_and_even_digit :
  ∀ n : ℕ, is_three_digit n → digit_product n = 8 → has_even_digit n → 124 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_product_8_and_even_digit_l2019_201917


namespace NUMINAMATH_CALUDE_not_rhombus_from_equal_adjacent_sides_l2019_201992

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- A rhombus is a quadrilateral with all sides equal -/
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
                 dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- Two sides are adjacent if they share a vertex -/
def are_adjacent_sides (q : Quadrilateral) (i j : Fin 4) : Prop :=
  (j = (i + 1) % 4) ∨ (i = (j + 1) % 4)

/-- A pair of adjacent sides are equal -/
def has_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  ∃ i j : Fin 4, are_adjacent_sides q i j ∧ 
    dist (q.vertices i) (q.vertices ((i + 1) % 4)) = 
    dist (q.vertices j) (q.vertices ((j + 1) % 4))

/-- The statement to be proved false -/
theorem not_rhombus_from_equal_adjacent_sides :
  ¬(∀ q : Quadrilateral, has_equal_adjacent_sides q → is_rhombus q) :=
sorry

end NUMINAMATH_CALUDE_not_rhombus_from_equal_adjacent_sides_l2019_201992


namespace NUMINAMATH_CALUDE_f_behavior_l2019_201962

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

def has_min_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → c ≤ f x

-- State the theorem
theorem f_behavior :
  is_even f →
  increasing_on f 5 7 →
  has_min_value f 5 7 6 →
  decreasing_on f (-7) (-5) ∧ has_min_value f (-7) (-5) 6 :=
sorry

end NUMINAMATH_CALUDE_f_behavior_l2019_201962


namespace NUMINAMATH_CALUDE_equal_savings_time_l2019_201939

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem equal_savings_time (sara_initial : ℕ) (sara_weekly : ℕ) (jim_initial : ℕ) (jim_weekly : ℕ)
  (h1 : sara_initial = 4100)
  (h2 : sara_weekly = 10)
  (h3 : jim_initial = 0)
  (h4 : jim_weekly = 15) :
  ∃ w : ℕ, w = 820 ∧ sara_initial + w * sara_weekly = jim_initial + w * jim_weekly :=
by
  sorry

end NUMINAMATH_CALUDE_equal_savings_time_l2019_201939


namespace NUMINAMATH_CALUDE_min_buses_needed_l2019_201901

/-- The capacity of each bus -/
def bus_capacity : ℕ := 45

/-- The total number of students to be transported -/
def total_students : ℕ := 540

/-- The minimum number of buses needed -/
def min_buses : ℕ := 12

/-- Theorem: The minimum number of buses needed to transport all students is 12 -/
theorem min_buses_needed : 
  (∀ n : ℕ, n * bus_capacity ≥ total_students → n ≥ min_buses) ∧ 
  (min_buses * bus_capacity ≥ total_students) :=
sorry

end NUMINAMATH_CALUDE_min_buses_needed_l2019_201901


namespace NUMINAMATH_CALUDE_area_of_triangle_PAB_l2019_201942

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

-- Define the tangent line
def tangent_line (x y m : ℝ) : Prop := y = Real.sqrt 3 * x + m ∧ m > 0

-- Define the point of tangency P
def point_P (x y : ℝ) : Prop := circle_O x y ∧ ∃ m, tangent_line x y m

-- Define points A and B as intersections of circle O and line y = x
def point_A_B (xa ya xb yb : ℝ) : Prop :=
  circle_O xa ya ∧ line_y_eq_x xa ya ∧
  circle_O xb yb ∧ line_y_eq_x xb yb ∧
  (xa ≠ xb ∨ ya ≠ yb)

-- Theorem statement
theorem area_of_triangle_PAB :
  ∀ (xa ya xb yb xp yp : ℝ),
  point_A_B xa ya xb yb →
  point_P xp yp →
  ∃ (area : ℝ), area = Real.sqrt 6 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_PAB_l2019_201942


namespace NUMINAMATH_CALUDE_molecular_weight_N2O3_l2019_201949

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms in N2O3
def N_count : ℕ := 2
def O_count : ℕ := 3

-- Define the number of moles
def moles : ℝ := 4

-- Theorem statement
theorem molecular_weight_N2O3 : 
  moles * (N_count * atomic_weight_N + O_count * atomic_weight_O) = 304.08 := by
  sorry


end NUMINAMATH_CALUDE_molecular_weight_N2O3_l2019_201949


namespace NUMINAMATH_CALUDE_log_cube_exp_inequality_l2019_201995

theorem log_cube_exp_inequality (x : ℝ) (h : 0 < x ∧ x < 1) :
  Real.log x / Real.log 3 < x^3 ∧ x^3 < 3^x := by
  sorry

end NUMINAMATH_CALUDE_log_cube_exp_inequality_l2019_201995


namespace NUMINAMATH_CALUDE_rainfall_depth_calculation_l2019_201935

/-- Calculates the approximate rainfall depth given container dimensions and collected water depth -/
theorem rainfall_depth_calculation (container_side : ℝ) (container_height : ℝ) (water_depth : ℝ) 
  (h1 : container_side = 20)
  (h2 : container_height = 40)
  (h3 : water_depth = 10) : 
  ∃ (rainfall_depth : ℝ), abs (rainfall_depth - 12.7) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_depth_calculation_l2019_201935
