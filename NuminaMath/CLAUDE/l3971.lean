import Mathlib

namespace NUMINAMATH_CALUDE_dividend_problem_l3971_397163

theorem dividend_problem (M D Q : ℕ) (h1 : M = 6 * D) (h2 : D = 4 * Q) : M = 144 := by
  sorry

end NUMINAMATH_CALUDE_dividend_problem_l3971_397163


namespace NUMINAMATH_CALUDE_circle_radius_l3971_397180

/-- Given a circle with equation x^2 + y^2 - 4x - 2y - 5 = 0, its radius is √10 -/
theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 2*y - 5 = 0) → 
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3971_397180


namespace NUMINAMATH_CALUDE_shark_difference_l3971_397137

theorem shark_difference (cape_may_sharks daytona_beach_sharks : ℕ) 
  (h1 : cape_may_sharks = 32) 
  (h2 : daytona_beach_sharks = 12) : 
  cape_may_sharks - 2 * daytona_beach_sharks = 8 := by
  sorry

end NUMINAMATH_CALUDE_shark_difference_l3971_397137


namespace NUMINAMATH_CALUDE_power_function_through_point_l3971_397123

/-- A power function passing through (2, √2/2) has f(9) = 1/3 -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x ^ α) →  -- f is a power function
  f 2 = Real.sqrt 2 / 2 →            -- f passes through (2, √2/2)
  f 9 = 1 / 3 :=                     -- f(9) = 1/3
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3971_397123


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l3971_397147

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x - a - 1 < 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem set_operations_and_inclusion (a : ℝ) : 
  (Set.compl A ∪ Set.compl B = {x | x ≤ 3 ∨ x ≥ 6}) ∧
  (B ⊆ C a ↔ a ≥ 8) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l3971_397147


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3971_397121

theorem quadratic_function_properties (b : ℝ) : 
  (∃ x : ℝ, x^2 - 2*b*x + b^2 + b - 5 = 0) →
  (∀ x < 3.5, (2*x - 2*b) < 0) →
  3.5 ≤ b ∧ b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3971_397121


namespace NUMINAMATH_CALUDE_servant_cash_received_l3971_397144

/-- Calculates the cash received by a servant after working for a partial year --/
theorem servant_cash_received
  (annual_cash : ℕ)
  (turban_price : ℕ)
  (months_worked : ℕ)
  (h1 : annual_cash = 90)
  (h2 : turban_price = 50)
  (h3 : months_worked = 9) :
  (months_worked * (annual_cash + turban_price) / 12) - turban_price = 55 :=
sorry

end NUMINAMATH_CALUDE_servant_cash_received_l3971_397144


namespace NUMINAMATH_CALUDE_average_hamburgers_is_nine_l3971_397148

-- Define the total number of hamburgers sold
def total_hamburgers : ℕ := 63

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the average number of hamburgers sold per day
def average_hamburgers : ℚ := total_hamburgers / days_in_week

-- Theorem to prove
theorem average_hamburgers_is_nine : average_hamburgers = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_hamburgers_is_nine_l3971_397148


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3971_397136

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: For an arithmetic sequence with first term a₁ and common difference d,
    if S₈ - S₃ = 20, then S₁₁ = 44 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) :
  S 8 a₁ d - S 3 a₁ d = 20 → S 11 a₁ d = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3971_397136


namespace NUMINAMATH_CALUDE_min_value_when_a_2_range_of_a_l3971_397150

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 1|

-- Theorem for the minimum value when a = 2
theorem min_value_when_a_2 :
  ∃ (min : ℝ), min = 3 ∧ ∀ x, f 2 x ≥ min :=
sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∃ x, f a x < 2) ↔ -3 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_2_range_of_a_l3971_397150


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3971_397175

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = (2 + x) / (2 - x) ∧
  (2 + 0) / (2 - 0) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3971_397175


namespace NUMINAMATH_CALUDE_field_trip_students_l3971_397170

theorem field_trip_students (adult_chaperones : ℕ) (student_fee adult_fee total_cost : ℚ) : 
  adult_chaperones = 4 →
  student_fee = 5 →
  adult_fee = 6 →
  total_cost = 199 →
  ∃ (num_students : ℕ), (num_students : ℚ) * student_fee + (adult_chaperones : ℚ) * adult_fee = total_cost ∧ 
    num_students = 35 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_students_l3971_397170


namespace NUMINAMATH_CALUDE_cubic_inequality_l3971_397149

theorem cubic_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3971_397149


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_circle_intersection_chord_length_l3971_397103

/-- The length of the chord formed by the intersection of an asymptote of a hyperbola with a specific circle -/
theorem hyperbola_asymptote_circle_intersection_chord_length 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 5  -- eccentricity
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 3)^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x ∨ y = -(b / a) * x}
  ∀ (A B : ℝ × ℝ), A ∈ circle → B ∈ circle → A ∈ asymptote → B ∈ asymptote →
  e^2 = 1 + b^2 / a^2 →
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_circle_intersection_chord_length_l3971_397103


namespace NUMINAMATH_CALUDE_least_number_with_remainders_l3971_397156

theorem least_number_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 34 = 4 ∧ 
  n % 5 = 4 ∧
  ∀ m : ℕ, m > 0 ∧ m % 34 = 4 ∧ m % 5 = 4 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainders_l3971_397156


namespace NUMINAMATH_CALUDE_parabola_parameter_l3971_397191

/-- Given a circle C₁ and a parabola C₂ intersecting at two points with a specific chord length,
    prove that the parameter of the parabola has a specific value. -/
theorem parabola_parameter (p : ℝ) (h_p : p > 0) : 
  ∃ A B : ℝ × ℝ,
    (A.1^2 + (A.2 - 2)^2 = 4) ∧ 
    (B.1^2 + (B.2 - 2)^2 = 4) ∧
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (8*Real.sqrt 5/5)^2) →
    p = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_l3971_397191


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l3971_397140

theorem cone_vertex_angle (α β αf : Real) : 
  β = 2 * Real.arcsin (1/4) →
  2 * α = αf →
  2 * α = Real.pi/6 + Real.arcsin (1/4) :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l3971_397140


namespace NUMINAMATH_CALUDE_polyhedron_edges_existence_l3971_397112

/-- The number of edges in the initial polyhedra we can start with -/
def initial_edges : List Nat := [8, 9, 10]

/-- The number of edges added when slicing off a triangular angle -/
def edges_per_slice : Nat := 3

/-- Proposition: For any natural number n ≥ 8, there exists a polyhedron with exactly n edges -/
theorem polyhedron_edges_existence (n : Nat) (h : n ≥ 8) :
  ∃ (k : Nat) (m : Nat), k ∈ initial_edges ∧ n = k + m * edges_per_slice :=
sorry

end NUMINAMATH_CALUDE_polyhedron_edges_existence_l3971_397112


namespace NUMINAMATH_CALUDE_zsigmondy_prime_l3971_397100

theorem zsigmondy_prime (n : ℕ+) (p : ℕ) (k : ℕ) :
  3^(n : ℕ) - 2^(n : ℕ) = p^k → Nat.Prime p → Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_zsigmondy_prime_l3971_397100


namespace NUMINAMATH_CALUDE_inequality_proof_l3971_397141

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c ≤ 3) :
  a / (1 + a^2) + b / (1 + b^2) + c / (1 + c^2) ≤ 3/2 ∧ 3/2 ≤ 1/(1+a) + 1/(1+b) + 1/(1+c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3971_397141


namespace NUMINAMATH_CALUDE_m_range_l3971_397124

/-- Proposition p: The quadratic equation with real coefficients x^2 + mx + 2 = 0 has imaginary roots -/
def prop_p (m : ℝ) : Prop := m^2 - 8 < 0

/-- Proposition q: For the equation 2x^2 - 4(m-1)x + m^2 + 7 = 0 (m ∈ ℝ), 
    the sum of the moduli of its two imaginary roots does not exceed 4√2 -/
def prop_q (m : ℝ) : Prop := 16*(m-1)^2 - 8*(m^2 + 7) < 0

/-- The range of m when both propositions p and q are true -/
theorem m_range (m : ℝ) : prop_p m ∧ prop_q m ↔ -1 < m ∧ m < 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l3971_397124


namespace NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l3971_397177

theorem remainder_of_55_power_55_plus_55_mod_56 :
  (55^55 + 55) % 56 = 54 := by sorry

end NUMINAMATH_CALUDE_remainder_of_55_power_55_plus_55_mod_56_l3971_397177


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3971_397160

/-- The area of a quadrilateral with given sides and one angle -/
theorem quadrilateral_area (a b c d : Real) (α : Real) : 
  a = 52 →
  b = 56 →
  c = 33 →
  d = 39 →
  α = 112 + 37 / 60 + 12 / 3600 →
  ∃ (area : Real), abs (area - 1774) < 1 ∧ 
  area = (1/2) * a * d * Real.sin α + 
          Real.sqrt ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - b) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - c) * 
                     ((1/2) * (b + c + Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α)) - Real.sqrt (a^2 + d^2 - 2*a*d*Real.cos α))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3971_397160


namespace NUMINAMATH_CALUDE_largest_number_l3971_397165

def a : ℚ := 24680 + 1 / 1357
def b : ℚ := 24680 - 1 / 1357
def c : ℚ := 24680 * (1 / 1357)
def d : ℚ := 24680 / (1 / 1357)
def e : ℚ := 24680.1357

theorem largest_number : 
  d > a ∧ d > b ∧ d > c ∧ d > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3971_397165


namespace NUMINAMATH_CALUDE_sin_negative_330_degrees_l3971_397120

theorem sin_negative_330_degrees : Real.sin ((-330 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_330_degrees_l3971_397120


namespace NUMINAMATH_CALUDE_inequalities_solution_sets_l3971_397106

def inequality1 (x : ℝ) : Prop := x^2 + 3*x + 2 ≤ 0

def inequality2 (x : ℝ) : Prop := -3*x^2 + 2*x + 2 < 0

def solution_set1 : Set ℝ := {x | -2 ≤ x ∧ x ≤ -1}

def solution_set2 : Set ℝ := {x | x < (1 - Real.sqrt 7) / 3 ∨ x > (1 + Real.sqrt 7) / 3}

theorem inequalities_solution_sets :
  (∀ x, x ∈ solution_set1 ↔ inequality1 x) ∧
  (∀ x, x ∈ solution_set2 ↔ inequality2 x) := by sorry

end NUMINAMATH_CALUDE_inequalities_solution_sets_l3971_397106


namespace NUMINAMATH_CALUDE_tennis_tournament_player_count_l3971_397192

/-- Represents a valid number of players in a tennis tournament with 2 vs 2 matches -/
def ValidPlayerCount (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 8 * k + 1

/-- Each player plays against every other player exactly once -/
def EachPlayerPlaysAllOthers (n : ℕ) : Prop :=
  (n - 1) % 2 = 0

/-- The total number of games is an integer -/
def TotalGamesInteger (n : ℕ) : Prop :=
  (n * (n - 1)) % 8 = 0

/-- Main theorem: Characterization of valid player counts in the tennis tournament -/
theorem tennis_tournament_player_count (n : ℕ) :
  (EachPlayerPlaysAllOthers n ∧ TotalGamesInteger n) ↔ ValidPlayerCount n :=
sorry

end NUMINAMATH_CALUDE_tennis_tournament_player_count_l3971_397192


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l3971_397188

theorem henrys_cd_collection :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l3971_397188


namespace NUMINAMATH_CALUDE_factorial_loop_condition_l3971_397151

/-- A function that calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The theorem stating that if a factorial program outputs 720,
    then the loop condition must be i <= 6 -/
theorem factorial_loop_condition (output : ℕ) (loop_condition : ℕ → Bool) :
  output = 720 →
  (∀ n : ℕ, factorial n = output → loop_condition = fun i => i ≤ n) →
  loop_condition = fun i => i ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_factorial_loop_condition_l3971_397151


namespace NUMINAMATH_CALUDE_base_five_digits_of_1837_l3971_397184

theorem base_five_digits_of_1837 (n : Nat) (h : n = 1837) :
  (Nat.log 5 n + 1 : Nat) = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1837_l3971_397184


namespace NUMINAMATH_CALUDE_collinear_points_on_cubic_curve_l3971_397116

/-- Three points on a cubic curve that are collinear satisfy a specific relation -/
theorem collinear_points_on_cubic_curve
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_curve₁ : y₁^2 = x₁^3)
  (h_curve₂ : y₂^2 = x₂^3)
  (h_curve₃ : y₃^2 = x₃^3)
  (h_collinear : (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁))
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  x₁ / y₁ + x₂ / y₂ + x₃ / y₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_on_cubic_curve_l3971_397116


namespace NUMINAMATH_CALUDE_goldfinch_percentage_l3971_397135

/-- The number of goldfinches -/
def goldfinches : ℕ := 6

/-- The number of sparrows -/
def sparrows : ℕ := 9

/-- The number of grackles -/
def grackles : ℕ := 5

/-- The total number of birds -/
def total_birds : ℕ := goldfinches + sparrows + grackles

/-- The fraction of goldfinches -/
def goldfinch_fraction : ℚ := goldfinches / total_birds

/-- Theorem: The percentage of goldfinches is 30% -/
theorem goldfinch_percentage :
  goldfinch_fraction * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_goldfinch_percentage_l3971_397135


namespace NUMINAMATH_CALUDE_jacoby_lottery_ticket_cost_l3971_397118

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def sisters_count : ℕ := 2
def remaining_needed : ℕ := 3214

theorem jacoby_lottery_ticket_cost :
  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let gifts := sister_gift * sisters_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + gifts
  let actual_total := trip_cost - remaining_needed
  total_earned - actual_total = 10
  := by sorry

end NUMINAMATH_CALUDE_jacoby_lottery_ticket_cost_l3971_397118


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3971_397122

theorem rationalize_denominator :
  (Real.sqrt 2) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5) = 
  (3 + Real.sqrt 6 + Real.sqrt 15) / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3971_397122


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_foci_l3971_397142

-- Define the hyperbola equation
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define that the hyperbola and ellipse share the same foci
def same_foci (m : ℝ) : Prop := ∃ (a b : ℝ), (hyperbola m a b ∧ ellipse a b)

-- Theorem statement
theorem hyperbola_ellipse_foci (m : ℝ) (h : same_foci m) : m = 1/3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_foci_l3971_397142


namespace NUMINAMATH_CALUDE_translation_left_2_units_l3971_397158

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateLeft (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x - units, y := p.y }

/-- The problem statement -/
theorem translation_left_2_units :
  let P : Point2D := { x := -2, y := -1 }
  let A' : Point2D := translateLeft P 2
  A' = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_translation_left_2_units_l3971_397158


namespace NUMINAMATH_CALUDE_platform_length_l3971_397161

/-- Calculates the length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- Train speed in kmph
  (h2 : platform_crossing_time = 30)  -- Time to cross platform in seconds
  (h3 : man_crossing_time = 15)  -- Time to cross man in seconds
  : ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3971_397161


namespace NUMINAMATH_CALUDE_weight_of_B_l3971_397104

theorem weight_of_B (A B C : ℝ) :
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 41 →
  (B + C) / 2 = 43 →
  B = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l3971_397104


namespace NUMINAMATH_CALUDE_only_D_is_certain_l3971_397101

structure Event where
  name : String
  is_certain : Bool

def A : Event := { name := "Moonlight in front of the bed", is_certain := false }
def B : Event := { name := "Lonely smoke in the desert", is_certain := false }
def C : Event := { name := "Reach for the stars with your hand", is_certain := false }
def D : Event := { name := "Yellow River flows into the sea", is_certain := true }

def events : List Event := [A, B, C, D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by
  sorry

end NUMINAMATH_CALUDE_only_D_is_certain_l3971_397101


namespace NUMINAMATH_CALUDE_advertising_department_size_l3971_397186

/-- Proves that given a company with 1000 total employees, using stratified sampling
    to draw 80 employees, if 4 employees are sampled from the advertising department,
    then the number of employees in the advertising department is 50. -/
theorem advertising_department_size
  (total_employees : ℕ)
  (sample_size : ℕ)
  (sampled_from_advertising : ℕ)
  (h_total : total_employees = 1000)
  (h_sample : sample_size = 80)
  (h_ad_sample : sampled_from_advertising = 4)
  : (sampled_from_advertising : ℚ) / sample_size * total_employees = 50 := by
  sorry

end NUMINAMATH_CALUDE_advertising_department_size_l3971_397186


namespace NUMINAMATH_CALUDE_trains_meet_time_trains_meet_time_approx_l3971_397128

/-- Calculates the time for two trains to meet given their lengths, initial distance, and speeds. -/
theorem trains_meet_time (length1 length2 initial_distance : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  let total_distance := initial_distance + length1 + length2
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  total_distance / relative_speed

/-- The time for two trains to meet is approximately 6.69 seconds. -/
theorem trains_meet_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |trains_meet_time 90 95 250 64 92 - 6.69| < ε :=
sorry

end NUMINAMATH_CALUDE_trains_meet_time_trains_meet_time_approx_l3971_397128


namespace NUMINAMATH_CALUDE_product_cost_change_l3971_397119

theorem product_cost_change (initial_cost : ℝ) (h : initial_cost > 0) : 
  initial_cost * (1 + 0.2)^2 * (1 - 0.2)^2 < initial_cost := by
  sorry

end NUMINAMATH_CALUDE_product_cost_change_l3971_397119


namespace NUMINAMATH_CALUDE_parabola_c_value_l3971_397195

/-- A parabola with equation x = ay² + by + c, vertex at (5, -3), and passing through (7, 1) has c = 49/8 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * (-3)^2 + b * (-3) + c) →  -- vertex condition
  (7 = a * 1^2 + b * 1 + c) →                 -- point condition
  (c = 49/8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3971_397195


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3971_397189

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3971_397189


namespace NUMINAMATH_CALUDE_original_number_theorem_l3971_397162

theorem original_number_theorem (x : ℝ) : 
  12 * ((x * 0.5 - 10) / 6) = 15 → x = 35 := by
sorry

end NUMINAMATH_CALUDE_original_number_theorem_l3971_397162


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3971_397178

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalScore : Nat
  notOutCount : Nat

/-- Calculate the batting average -/
def battingAverage (b : Batsman) : Rat :=
  b.totalScore / (b.innings - b.notOutCount)

/-- The increase in average after a new innings -/
def averageIncrease (b : Batsman) (newScore : Nat) : Rat :=
  battingAverage { innings := b.innings + 1, totalScore := b.totalScore + newScore, notOutCount := b.notOutCount } -
  battingAverage b

/-- Theorem: The batsman's average increase is 2 runs -/
theorem batsman_average_increase :
  ∀ (b : Batsman),
    b.innings = 11 →
    b.notOutCount = 0 →
    (battingAverage { innings := b.innings + 1, totalScore := b.totalScore + 80, notOutCount := b.notOutCount } = 58) →
    averageIncrease b 80 = 2 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l3971_397178


namespace NUMINAMATH_CALUDE_inequality_proof_l3971_397155

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * b + b * c + c * a = 1) : 
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 
    2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3971_397155


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l3971_397152

def f (c x : ℝ) : ℝ := c*x^4 + 15*x^3 - 5*c*x^2 - 45*x + 55

theorem factor_implies_c_value (c : ℝ) :
  (∀ x : ℝ, (x + 5) ∣ f c x) → c = 319 / 100 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l3971_397152


namespace NUMINAMATH_CALUDE_halloween_candy_duration_l3971_397117

/-- Calculates the number of full days candy will last given initial amounts, trades, losses, and daily consumption. -/
def candy_duration (neighbors : ℕ) (sister : ℕ) (traded : ℕ) (lost : ℕ) (daily_consumption : ℕ) : ℕ :=
  ((neighbors + sister - traded - lost) / daily_consumption : ℕ)

/-- Theorem stating that under the given conditions, the candy will last for 23 full days. -/
theorem halloween_candy_duration :
  candy_duration 75 130 25 15 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_duration_l3971_397117


namespace NUMINAMATH_CALUDE_aiNK_probability_l3971_397185

/-- The number of distinct cards labeled with the letters of "NanKai" -/
def total_cards : ℕ := 6

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The number of ways to form "aiNK" from the drawn cards -/
def successful_outcomes : ℕ := 1

/-- The total number of ways to draw 4 cards from 6 -/
def total_outcomes : ℕ := Nat.choose total_cards drawn_cards

/-- The probability of drawing four cards that can form "aiNK" -/
def probability : ℚ := successful_outcomes / total_outcomes

theorem aiNK_probability : probability = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_aiNK_probability_l3971_397185


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_value_l3971_397105

/-- Given vectors a and b in R², if the magnitude of their sum equals the magnitude of their difference, then the second component of b is 3. -/
theorem vector_equality_implies_x_value (a b : ℝ × ℝ) :
  a = (2, -4) →
  b.1 = 6 →
  ‖a + b‖ = ‖a - b‖ →
  b.2 = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_equality_implies_x_value_l3971_397105


namespace NUMINAMATH_CALUDE_triangle_perimeter_21_l3971_397114

-- Define the triangle
def Triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter_21 :
  ∀ c : ℝ,
  Triangle 10 3 c →
  (Perimeter 10 3 c = 18 ∨ Perimeter 10 3 c = 19 ∨ Perimeter 10 3 c = 20 ∨ Perimeter 10 3 c = 21) →
  Perimeter 10 3 c = 21 :=
by
  sorry

#check triangle_perimeter_21

end NUMINAMATH_CALUDE_triangle_perimeter_21_l3971_397114


namespace NUMINAMATH_CALUDE_symbol_set_has_14_plus_l3971_397199

/-- A set of symbols consisting of plus and minus signs -/
structure SymbolSet where
  total : ℕ
  plus : ℕ
  minus : ℕ
  sum_eq : total = plus + minus
  plus_constraint : ∀ (n : ℕ), n ≤ total - plus → n < 10
  minus_constraint : ∀ (n : ℕ), n ≤ total - minus → n < 15

/-- The theorem stating that a SymbolSet with 23 total symbols has 14 plus signs -/
theorem symbol_set_has_14_plus (s : SymbolSet) (h : s.total = 23) : s.plus = 14 := by
  sorry

end NUMINAMATH_CALUDE_symbol_set_has_14_plus_l3971_397199


namespace NUMINAMATH_CALUDE_diamond_three_four_l3971_397130

def diamond (a b : ℝ) : ℝ := 4 * a + 3 * b - 2 * a * b

theorem diamond_three_four : diamond 3 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3971_397130


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l3971_397146

theorem right_triangle_squares_area (XY YZ XZ : ℝ) :
  XY = 5 →
  XZ = 13 →
  XY^2 + YZ^2 = XZ^2 →
  XY^2 + YZ^2 = 169 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l3971_397146


namespace NUMINAMATH_CALUDE_square_diff_theorem_l3971_397196

theorem square_diff_theorem : (25 + 9)^2 - (25^2 + 9^2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_theorem_l3971_397196


namespace NUMINAMATH_CALUDE_cats_remaining_after_missions_l3971_397133

/-- The number of cats remaining on Tatoosh Island after two relocation missions -/
def cats_remaining (initial : ℕ) (first_relocation : ℕ) : ℕ :=
  let after_first := initial - first_relocation
  let second_relocation := after_first / 2
  after_first - second_relocation

/-- Theorem stating that 600 cats remain on the island after the relocation missions -/
theorem cats_remaining_after_missions :
  cats_remaining 1800 600 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cats_remaining_after_missions_l3971_397133


namespace NUMINAMATH_CALUDE_overall_loss_percentage_is_about_2_09_percent_l3971_397113

/-- Represents an appliance with its cost price and profit/loss percentage -/
structure Appliance where
  costPrice : ℕ
  profitLossPercentage : ℤ

/-- Calculates the selling price of an appliance -/
def sellingPrice (a : Appliance) : ℚ :=
  a.costPrice * (1 + a.profitLossPercentage / 100)

/-- The list of appliances with their cost prices and profit/loss percentages -/
def appliances : List Appliance := [
  ⟨15000, -5⟩,
  ⟨8000, 10⟩,
  ⟨12000, -8⟩,
  ⟨10000, 15⟩,
  ⟨5000, 7⟩,
  ⟨20000, -12⟩
]

/-- The total cost price of all appliances -/
def totalCostPrice : ℕ := (appliances.map (·.costPrice)).sum

/-- The total selling price of all appliances -/
def totalSellingPrice : ℚ := (appliances.map sellingPrice).sum

/-- The overall loss percentage -/
def overallLossPercentage : ℚ :=
  (totalCostPrice - totalSellingPrice) / totalCostPrice * 100

/-- Theorem stating that the overall loss percentage is approximately 2.09% -/
theorem overall_loss_percentage_is_about_2_09_percent :
  abs (overallLossPercentage - 2.09) < 0.01 := by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_is_about_2_09_percent_l3971_397113


namespace NUMINAMATH_CALUDE_third_face_area_l3971_397107

-- Define the properties of the cuboidal box
def cuboidal_box (l w h : ℝ) : Prop :=
  l > 0 ∧ w > 0 ∧ h > 0 ∧
  l * w = 72 ∧
  w * h = 60 ∧
  l * w * h = 720

-- Theorem statement
theorem third_face_area (l w h : ℝ) :
  cuboidal_box l w h → l * h = 120 := by
  sorry

end NUMINAMATH_CALUDE_third_face_area_l3971_397107


namespace NUMINAMATH_CALUDE_angle_four_times_complement_l3971_397110

theorem angle_four_times_complement (x : ℝ) : 
  (x = 4 * (90 - x)) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_four_times_complement_l3971_397110


namespace NUMINAMATH_CALUDE_car_race_bet_l3971_397168

theorem car_race_bet (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  ∃ w : ℝ, w = 8 / 3 ∧ 
    karen_speed * (w / tom_speed - karen_delay) = w + winning_margin :=
by sorry

end NUMINAMATH_CALUDE_car_race_bet_l3971_397168


namespace NUMINAMATH_CALUDE_rectangle_length_l3971_397164

theorem rectangle_length (l w : ℝ) (h1 : l = 4 * w) (h2 : l * w = 100) : l = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3971_397164


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3971_397179

theorem min_values_ab_and_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2 * a + b) :
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → x * y = 2 * x + y → a + 2 * b ≤ x + 2 * y) ∧
  a * b = 8 ∧ a + 2 * b = 9 := by
sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l3971_397179


namespace NUMINAMATH_CALUDE_fish_count_proof_l3971_397125

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken released -/
def kens_released : ℕ := 3

/-- The number of fish Ken brought home -/
def kens_brought_home : ℕ := kens_catch - kens_released

/-- The number of fish Kendra brought home (same as caught) -/
def kendras_brought_home : ℕ := kendras_catch

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := kens_brought_home + kendras_brought_home

theorem fish_count_proof : total_brought_home = 87 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_proof_l3971_397125


namespace NUMINAMATH_CALUDE_system_solution_l3971_397176

theorem system_solution (x y z : ℝ) : 
  x^2 + y^2 = -x + 3*y + z ∧ 
  y^2 + z^2 = x + 3*y - z ∧ 
  x^2 + z^2 = 2*x + 2*y - z ∧ 
  x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ((x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = -3/2 ∧ y = 5/2 ∧ z = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3971_397176


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3971_397129

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^5 + 3 * x^4 - 5 * x^3 + 2 * x^2 - 10 * x + 8) + 
  (-3 * x^5 - x^4 + 4 * x^3 - 2 * x^2 + 15 * x - 12) = 
  -x^5 + 2 * x^4 - x^3 + 5 * x - 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3971_397129


namespace NUMINAMATH_CALUDE_distance_between_points_l3971_397139

theorem distance_between_points (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let angleABC := Real.arccos ((AB^2 + BC^2 - AC^2) / (2 * AB * BC))
  AB = 20 ∧ BC = 30 ∧ angleABC = 2 * Real.pi / 3 →
  AC = 10 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3971_397139


namespace NUMINAMATH_CALUDE_onion_price_per_pound_l3971_397157

/-- Represents the price and quantity of ingredients --/
structure Ingredient where
  name : String
  quantity : ℝ
  price_per_unit : ℝ

/-- Represents the ratatouille recipe --/
def Recipe : List Ingredient := [
  ⟨"eggplants", 5, 2⟩,
  ⟨"zucchini", 4, 2⟩,
  ⟨"tomatoes", 4, 3.5⟩,
  ⟨"basil", 1, 5⟩  -- Price adjusted for 1 pound
]

def onion_quantity : ℝ := 3
def quart_yield : ℕ := 4
def price_per_quart : ℝ := 10

/-- Calculates the total cost of ingredients excluding onions --/
def total_cost_without_onions : ℝ :=
  Recipe.map (fun i => i.quantity * i.price_per_unit) |>.sum

/-- Calculates the target total cost --/
def target_total_cost : ℝ := quart_yield * price_per_quart

/-- Theorem: The price per pound of onions is $1.00 --/
theorem onion_price_per_pound :
  (target_total_cost - total_cost_without_onions) / onion_quantity = 1 := by
  sorry

end NUMINAMATH_CALUDE_onion_price_per_pound_l3971_397157


namespace NUMINAMATH_CALUDE_factor_x12_minus_4096_l3971_397182

theorem factor_x12_minus_4096 (x : ℝ) :
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_4096_l3971_397182


namespace NUMINAMATH_CALUDE_product_zero_l3971_397111

theorem product_zero (b : ℤ) (h : b = 4) : 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l3971_397111


namespace NUMINAMATH_CALUDE_partial_fraction_A_value_l3971_397198

-- Define the polynomial in the denominator
def p (x : ℝ) : ℝ := x^4 - 2*x^3 - 29*x^2 + 70*x + 120

-- Define the partial fraction decomposition
def partial_fraction (x A B C D : ℝ) : Prop :=
  1 / p x = A / (x + 4) + B / (x - 2) + C / (x - 2)^2 + D / (x - 3)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction x A B C D) → A = -1/252 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_A_value_l3971_397198


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3971_397167

theorem gcd_lcm_product_24_36 : 
  (Nat.gcd 24 36) * (Nat.lcm 24 36) = 864 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_36_l3971_397167


namespace NUMINAMATH_CALUDE_range_of_m_l3971_397193

theorem range_of_m (x m : ℝ) : 
  (m > 0) →
  (∀ x, ((x - 4) / 3)^2 > 4 → x^2 - 2*x + 1 - m^2 > 0) →
  (∃ x, ((x - 4) / 3)^2 > 4 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3971_397193


namespace NUMINAMATH_CALUDE_remaining_gift_cards_value_l3971_397115

/-- Represents the types of gift cards --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents a gift card with its type and value --/
structure GiftCard where
  type : GiftCardType
  value : Nat

def initial_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 },
  { type := GiftCardType.Amazon, value := 1000 }
]

def sent_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 }
]

theorem remaining_gift_cards_value : 
  (List.sum (initial_gift_cards.map (λ g => g.value)) - 
   List.sum (sent_gift_cards.map (λ g => g.value))) = 4250 := by
  sorry

end NUMINAMATH_CALUDE_remaining_gift_cards_value_l3971_397115


namespace NUMINAMATH_CALUDE_rectangle_center_sum_l3971_397126

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- Rectangle is in the first quadrant
  rect.A.1 ≥ 0 ∧ rect.A.2 ≥ 0 ∧
  rect.B.1 ≥ 0 ∧ rect.B.2 ≥ 0 ∧
  rect.C.1 ≥ 0 ∧ rect.C.2 ≥ 0 ∧
  rect.D.1 ≥ 0 ∧ rect.D.2 ≥ 0 ∧
  -- Points on the lines
  (2 : ℝ) ∈ Set.Icc rect.D.1 rect.A.1 ∧
  (6 : ℝ) ∈ Set.Icc rect.C.1 rect.B.1 ∧
  (10 : ℝ) ∈ Set.Icc rect.A.1 rect.B.1 ∧
  (18 : ℝ) ∈ Set.Icc rect.C.1 rect.D.1 ∧
  -- Ratio of AB to BC is 2:1
  2 * (rect.B.1 - rect.C.1) = rect.B.1 - rect.A.1

-- Theorem statement
theorem rectangle_center_sum (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  (rect.A.1 + rect.C.1) / 2 + (rect.A.2 + rect.C.2) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_center_sum_l3971_397126


namespace NUMINAMATH_CALUDE_points_on_line_y_relation_l3971_397131

/-- Given two points A(1, y₁) and B(-1, y₂) on the line y = -3x + 2, 
    prove that y₁ < y₂ -/
theorem points_on_line_y_relation (y₁ y₂ : ℝ) : 
  (1 : ℝ) > (-1 : ℝ) → -- x₁ > x₂
  y₁ = -3 * (1 : ℝ) + 2 → -- Point A satisfies the line equation
  y₂ = -3 * (-1 : ℝ) + 2 → -- Point B satisfies the line equation
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_points_on_line_y_relation_l3971_397131


namespace NUMINAMATH_CALUDE_coffee_price_increase_percentage_l3971_397159

def first_quarter_price : ℝ := 40
def fourth_quarter_price : ℝ := 60

theorem coffee_price_increase_percentage : 
  (fourth_quarter_price - first_quarter_price) / first_quarter_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_coffee_price_increase_percentage_l3971_397159


namespace NUMINAMATH_CALUDE_original_tree_count_l3971_397187

/-- The number of leaves each tree drops during fall. -/
def leaves_per_tree : ℕ := 100

/-- The total number of fallen leaves. -/
def total_fallen_leaves : ℕ := 1400

/-- The current number of trees is twice the original plan. -/
def current_trees_twice_original (original : ℕ) : Prop :=
  2 * original = total_fallen_leaves / leaves_per_tree

/-- Theorem stating the original number of trees the town council intended to plant. -/
theorem original_tree_count : ∃ (original : ℕ), 
  current_trees_twice_original original ∧ original = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_original_tree_count_l3971_397187


namespace NUMINAMATH_CALUDE_floor_sum_opposite_l3971_397172

theorem floor_sum_opposite (x : ℝ) (h : x = 15.8) : 
  ⌊x⌋ + ⌊-x⌋ = -1 := by sorry

end NUMINAMATH_CALUDE_floor_sum_opposite_l3971_397172


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3971_397166

/-- Given a sphere with an inscribed triangle on its section, prove its surface area --/
theorem sphere_surface_area (a b c : ℝ) (r R : ℝ) : 
  a = 6 → b = 8 → c = 10 →  -- Triangle side lengths
  r = 5 →  -- Radius of section's circle
  R^2 - (R/2)^2 = r^2 →  -- Relation between sphere radius and section
  4 * π * R^2 = 400 * π / 3 := by
  sorry

#check sphere_surface_area

end NUMINAMATH_CALUDE_sphere_surface_area_l3971_397166


namespace NUMINAMATH_CALUDE_find_m_l3971_397132

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

theorem find_m : 
  ∃ (m : ℝ), (U \ A m = {1, 2}) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3971_397132


namespace NUMINAMATH_CALUDE_car_speed_problem_l3971_397102

/-- Proves that the speed of the first car is 60 mph given the problem conditions -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  2.5 * v + 2.5 * 64 = 310 → 
  v = 60 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3971_397102


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3971_397138

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a structure for quadratic radicals
structure QuadraticRadical where
  coefficient : ℚ
  radicand : ℕ

-- Define a function to determine if a QuadraticRadical is in its simplest form
def is_simplest_form (qr : QuadraticRadical) : Prop :=
  qr.coefficient ≠ 0 ∧ 
  ¬(is_perfect_square qr.radicand) ∧ 
  is_prime qr.radicand

-- Define the given options
def option_A : QuadraticRadical := ⟨1, 2⟩ -- We represent √(2/3) as √2 / √3
def option_B : QuadraticRadical := ⟨2, 2⟩
def option_C : QuadraticRadical := ⟨1, 24⟩
def option_D : QuadraticRadical := ⟨1, 81⟩

-- Theorem statement
theorem simplest_quadratic_radical :
  is_simplest_form option_B ∧
  ¬(is_simplest_form option_A) ∧
  ¬(is_simplest_form option_C) ∧
  ¬(is_simplest_form option_D) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3971_397138


namespace NUMINAMATH_CALUDE_sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l3971_397108

-- Define the first and second quadrants
def first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def second_quadrant (θ : ℝ) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem sin_inequality_implies_angle_inequality (α β : ℝ) :
  Real.sin α ≠ Real.sin β → α ≠ β :=
sorry

theorem sin_positive_in_first_and_second_quadrant (θ : ℝ) :
  (first_quadrant θ ∨ second_quadrant θ) → Real.sin θ > 0 :=
sorry

end NUMINAMATH_CALUDE_sin_inequality_implies_angle_inequality_sin_positive_in_first_and_second_quadrant_l3971_397108


namespace NUMINAMATH_CALUDE_toy_store_spending_l3971_397173

/-- Proof of student's spending at toy store -/
theorem toy_store_spending (total_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_spending : ℚ) :
  total_allowance = 4.5 →
  arcade_fraction = 3/5 →
  candy_spending = 1.2 →
  let remaining_after_arcade := total_allowance - (arcade_fraction * total_allowance)
  let toy_store_spending := remaining_after_arcade - candy_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_toy_store_spending_l3971_397173


namespace NUMINAMATH_CALUDE_evaluate_expression_l3971_397197

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 2 * y^x = 533 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3971_397197


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_quotient_isosceles_right_triangle_max_quotient_l3971_397127

/-- For an isosceles right triangle with legs of length a, 
    the value of 2a / √(a^2 + a^2) is equal to √2 -/
theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  2 * a / Real.sqrt (a^2 + a^2) = Real.sqrt 2 := by
  sorry

/-- The maximum quotient (a + b) / c for an isosceles right triangle 
    with legs of length a is √2 -/
theorem isosceles_right_triangle_max_quotient (a : ℝ) (h : a > 0) :
  (a + a) / Real.sqrt (2 * a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_quotient_isosceles_right_triangle_max_quotient_l3971_397127


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_a_greater_than_one_l3971_397181

/-- Represents a quadratic function of the form f(x) = 2ax^2 - x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

/-- Condition for exactly one solution in (0, 1) -/
def has_exactly_one_solution_in_interval (a : ℝ) : Prop :=
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f a x = 0

theorem quadratic_solution_implies_a_greater_than_one :
  ∀ a : ℝ, has_exactly_one_solution_in_interval a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_a_greater_than_one_l3971_397181


namespace NUMINAMATH_CALUDE_integral_x_squared_l3971_397134

theorem integral_x_squared : ∫ x in (-1)..1, x^2 = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l3971_397134


namespace NUMINAMATH_CALUDE_cyclist_distance_difference_l3971_397143

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance (c : Cyclist) (time : ℝ) : ℝ := c.speed * time

theorem cyclist_distance_difference 
  (clara : Cyclist) 
  (david : Cyclist) 
  (h1 : clara.speed = 14.4) 
  (h2 : david.speed = 10.8) 
  (time : ℝ) 
  (h3 : time = 5) : 
  distance clara time - distance david time = 18 := by
sorry

end NUMINAMATH_CALUDE_cyclist_distance_difference_l3971_397143


namespace NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l3971_397169

/-- Definition of a Benelux couple -/
def is_benelux_couple (m n : ℕ) : Prop :=
  1 < m ∧ m < n ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)) ∧
  (∀ p : ℕ, Nat.Prime p → (p ∣ (m + 1) ↔ p ∣ (n + 1)))

/-- Theorem: There exist infinitely many Benelux couples -/
theorem infinitely_many_benelux_couples :
  ∀ N : ℕ, ∃ m n : ℕ, N < m ∧ is_benelux_couple m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_benelux_couples_l3971_397169


namespace NUMINAMATH_CALUDE_solve_for_A_l3971_397190

theorem solve_for_A (x : ℝ) (A : ℝ) (h : (5 : ℝ) / (x + 1) = A - ((2 * x - 3) / (x + 1))) :
  A = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l3971_397190


namespace NUMINAMATH_CALUDE_parabola_opens_upwards_l3971_397171

/-- For a parabola y = (2-m)x^2 + 1 to open upwards, m must be less than 2 -/
theorem parabola_opens_upwards (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (2 - m) * x^2 + 1) → 
  (∀ a b : ℝ, a < b → ((2 - m) * a^2 + 1) < ((2 - m) * b^2 + 1)) →
  m < 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_opens_upwards_l3971_397171


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3971_397154

/-- The focal length of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_focal_length (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let equation := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

/-- The focal length of the hyperbola x^2/4 - y^2/9 = 1 is 2√13 -/
theorem specific_hyperbola_focal_length :
  let equation := fun (x y : ℝ) => x^2 / 4 - y^2 / 9 = 1
  let focal_length := 2 * Real.sqrt (4 + 9)
  equation 2 3 → focal_length = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l3971_397154


namespace NUMINAMATH_CALUDE_expansion_coefficients_l3971_397183

/-- The coefficient of x^n in the expansion of (1 + x^5 + x^7)^20 -/
def coeff (n : ℕ) : ℕ :=
  (Finset.range 21).sum (fun k =>
    (Finset.range (21 - k)).sum (fun m =>
      if 5 * k + 7 * m == n && k + m ≤ 20
      then Nat.choose 20 k * Nat.choose (20 - k) m
      else 0))

theorem expansion_coefficients :
  coeff 17 = 3420 ∧ coeff 18 = 0 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l3971_397183


namespace NUMINAMATH_CALUDE_yoga_studio_men_count_l3971_397109

theorem yoga_studio_men_count :
  ∀ (num_men : ℕ) (avg_weight_men avg_weight_women avg_weight_all : ℝ),
    avg_weight_men = 190 →
    avg_weight_women = 120 →
    num_men + 6 = 14 →
    (num_men * avg_weight_men + 6 * avg_weight_women) / 14 = avg_weight_all →
    avg_weight_all = 160 →
    num_men = 8 := by
  sorry

end NUMINAMATH_CALUDE_yoga_studio_men_count_l3971_397109


namespace NUMINAMATH_CALUDE_regularity_lemma_l3971_397194

/-- A graph represented as a set of vertices and a set of edges -/
structure Graph (V : Type) where
  vertices : Set V
  edges : Set (V × V)

/-- The maximum degree of a graph -/
def max_degree (G : Graph V) : ℕ := sorry

/-- A regularity graph with parameters ε, ℓ, and d -/
structure RegularityGraph (V : Type) extends Graph V where
  ε : ℝ
  ℓ : ℕ
  d : ℝ

/-- The s-closure of a regularity graph -/
def s_closure (R : RegularityGraph V) (s : ℕ) : Graph V := sorry

/-- Subgraph relation -/
def is_subgraph (H G : Graph V) : Prop := sorry

theorem regularity_lemma {V : Type} (d : ℝ) (Δ : ℕ) 
  (hd : d ∈ Set.Icc 0 1) (hΔ : Δ ≥ 1) :
  ∃ ε₀ > 0, ∀ (G H : Graph V) (s : ℕ) (R : RegularityGraph V),
    max_degree H ≤ Δ →
    R.ε ≤ ε₀ →
    R.ℓ ≥ 2 * s / d^Δ →
    R.d = d →
    is_subgraph H (s_closure R s) →
    is_subgraph H G :=
sorry

end NUMINAMATH_CALUDE_regularity_lemma_l3971_397194


namespace NUMINAMATH_CALUDE_possible_value_less_than_five_l3971_397153

theorem possible_value_less_than_five : ∃ x : ℝ, x < 5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_possible_value_less_than_five_l3971_397153


namespace NUMINAMATH_CALUDE_lukes_final_balance_l3971_397174

/-- Calculates Luke's final balance after six months of financial activities --/
def lukesFinalBalance (initialAmount : ℝ) (februarySpendingRate : ℝ) 
  (marchSpending marchIncome : ℝ) (monthlyPiggyBankRate : ℝ) : ℝ :=
  let afterFebruary := initialAmount * (1 - februarySpendingRate)
  let afterMarch := afterFebruary - marchSpending + marchIncome
  let afterApril := afterMarch * (1 - monthlyPiggyBankRate)
  let afterMay := afterApril * (1 - monthlyPiggyBankRate)
  let afterJune := afterMay * (1 - monthlyPiggyBankRate)
  afterJune

/-- Theorem stating Luke's final balance after six months --/
theorem lukes_final_balance :
  lukesFinalBalance 48 0.3 11 21 0.1 = 31.79 := by
  sorry

end NUMINAMATH_CALUDE_lukes_final_balance_l3971_397174


namespace NUMINAMATH_CALUDE_exists_small_triangle_area_l3971_397145

-- Define a lattice point type
structure LatticePoint where
  x : Int
  y : Int

-- Define the condition for a point to be within the given bounds
def withinBounds (p : LatticePoint) : Prop :=
  abs p.x ≤ 2 ∧ abs p.y ≤ 2

-- Define the condition for three points to be non-collinear
def nonCollinear (p q r : LatticePoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

-- Calculate the area of a triangle formed by three points
def triangleArea (p q r : LatticePoint) : ℚ :=
  let a := (q.x - p.x) * (r.y - p.y) - (r.x - p.x) * (q.y - p.y)
  (abs a : ℚ) / 2

-- Main theorem
theorem exists_small_triangle_area 
  (P : Fin 6 → LatticePoint)
  (h_bounds : ∀ i, withinBounds (P i))
  (h_noncollinear : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → nonCollinear (P i) (P j) (P k)) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ triangleArea (P i) (P j) (P k) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_small_triangle_area_l3971_397145
