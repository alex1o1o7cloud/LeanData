import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l302_30289

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℤ) (q : ℤ) :
  geometric_sequence a q ∧ a 1 = 1 ∧ q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l302_30289


namespace NUMINAMATH_CALUDE_tangent_line_equations_l302_30297

/-- The curve to which the line is tangent -/
def f (x : ℝ) : ℝ := x^2 * (x + 1)

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * x

/-- A line that passes through (3/5, 0) and is tangent to f at point t -/
def tangent_line (t : ℝ) (x : ℝ) : ℝ :=
  f' t * (x - t) + f t

/-- The point (3/5, 0) lies on the tangent line -/
def point_condition (t : ℝ) : Prop :=
  tangent_line t (3/5) = 0

/-- The possible equations for the tangent line -/
def possible_equations (x : ℝ) : Prop :=
  (∃ t, point_condition t ∧ tangent_line t x = 0) ∨
  (∃ t, point_condition t ∧ tangent_line t x = -3/2 * x + 9/125) ∨
  (∃ t, point_condition t ∧ tangent_line t x = 5 * x - 3)

theorem tangent_line_equations :
  ∀ x, possible_equations x :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l302_30297


namespace NUMINAMATH_CALUDE_hash_prehash_eighteen_l302_30260

-- Define the # operator
def hash (x : ℝ) : ℝ := x + 5

-- Define the # prefix operator
def prehash (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem hash_prehash_eighteen : prehash (hash 18) = 18 := by
  sorry

end NUMINAMATH_CALUDE_hash_prehash_eighteen_l302_30260


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_equals_3_sqrt_2_l302_30285

theorem sqrt_3_times_sqrt_6_equals_3_sqrt_2 : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_6_equals_3_sqrt_2_l302_30285


namespace NUMINAMATH_CALUDE_total_diagonals_50_75_l302_30283

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem total_diagonals_50_75 : diagonals 50 + diagonals 75 = 3875 := by
  sorry

end NUMINAMATH_CALUDE_total_diagonals_50_75_l302_30283


namespace NUMINAMATH_CALUDE_greater_number_is_eighteen_l302_30281

theorem greater_number_is_eighteen (x y : ℝ) 
  (sum : x + y = 30)
  (diff : x - y = 6)
  (y_lower_bound : y ≥ 10)
  (x_greater : x > y) :
  x = 18 := by
sorry

end NUMINAMATH_CALUDE_greater_number_is_eighteen_l302_30281


namespace NUMINAMATH_CALUDE_woodworker_legs_count_l302_30206

/-- The number of furniture legs made by a woodworker -/
def total_furniture_legs (chairs tables : ℕ) : ℕ :=
  4 * chairs + 4 * tables

/-- Theorem: A woodworker who has built 6 chairs and 4 tables has made 40 furniture legs in total -/
theorem woodworker_legs_count : total_furniture_legs 6 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_legs_count_l302_30206


namespace NUMINAMATH_CALUDE_red_cards_after_turning_l302_30249

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

def count_red_cards (n : ℕ) : ℕ :=
  let initial_red := n
  let turned_by_2 := n / 2
  let odd_turned_by_3 := (n / 3 + 1) / 2
  let even_turned_by_3 := n / 6
  initial_red - turned_by_2 - odd_turned_by_3 + even_turned_by_3

theorem red_cards_after_turning (n : ℕ) (h : n = 100) : count_red_cards n = 49 := by
  sorry

end NUMINAMATH_CALUDE_red_cards_after_turning_l302_30249


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l302_30288

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ initial_avg = 184 ∧ actual_height = 106 ∧ actual_avg = 182 →
  ∃ wrong_height : ℝ, wrong_height = 176 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l302_30288


namespace NUMINAMATH_CALUDE_only_lottery_is_random_l302_30299

-- Define the events
inductive Event
| BasketballFall
| LotteryWin
| BirthdayMatch
| DrawBlackBall

-- Define the properties of events
def isCertain (e : Event) : Prop :=
  match e with
  | Event.BasketballFall => true
  | _ => false

def isImpossible (e : Event) : Prop :=
  match e with
  | Event.DrawBlackBall => true
  | _ => false

def isRandom (e : Event) : Prop :=
  ¬(isCertain e) ∧ ¬(isImpossible e)

-- Define the given conditions
axiom gravity_exists : isCertain Event.BasketballFall
axiom pigeonhole_principle : isCertain Event.BirthdayMatch
axiom bag_contents : isImpossible Event.DrawBlackBall

-- State the theorem
theorem only_lottery_is_random :
  ∀ e : Event, isRandom e ↔ e = Event.LotteryWin :=
sorry

end NUMINAMATH_CALUDE_only_lottery_is_random_l302_30299


namespace NUMINAMATH_CALUDE_expression_value_l302_30261

theorem expression_value (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  (x - 2*y) * (x + 2*y) - (x - 2*y)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l302_30261


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_equals_volume_l302_30294

theorem rectangular_parallelepiped_surface_area_equals_volume :
  ∃ (a b c : ℕ+), 2 * (a * b + b * c + a * c) = a * b * c :=
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_equals_volume_l302_30294


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l302_30219

/-- Given a principal amount where the simple interest for 2 years at 5% per annum is 52,
    prove that the compound interest at 5% per annum for 2 years is 53.30 -/
theorem compound_interest_calculation (P : ℝ) : 
  (P * 5 * 2) / 100 = 52 →
  P * ((1 + 5/100)^2 - 1) = 53.30 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l302_30219


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l302_30232

theorem binomial_square_coefficient (x : ℝ) : ∃ (a : ℝ), 
  (∃ (r s : ℝ), (r * x + s)^2 = a * x^2 + 20 * x + 9) ∧ 
  a = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l302_30232


namespace NUMINAMATH_CALUDE_sons_age_l302_30276

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l302_30276


namespace NUMINAMATH_CALUDE_magnitude_AB_is_5_l302_30202

def A : ℝ × ℝ := (-1, -6)
def B : ℝ × ℝ := (2, -2)

def vector_AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem magnitude_AB_is_5 : 
  Real.sqrt ((vector_AB.1)^2 + (vector_AB.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_magnitude_AB_is_5_l302_30202


namespace NUMINAMATH_CALUDE_inequality_properties_l302_30256

theorem inequality_properties (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a * b > b ^ 2) ∧ (1 / a < 1 / b) ∧ (a ^ 2 > a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l302_30256


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_geq_two_l302_30223

/-- The function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The theorem stating that if f(x+a) is increasing on [0, +∞), then a ≥ 2 -/
theorem f_increasing_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f (x + a) < f (y + a)) →
  a ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_geq_two_l302_30223


namespace NUMINAMATH_CALUDE_arrangementsWithOneNotAtEndFour_l302_30209

/-- The number of ways to arrange n people in a row. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with one specific person not at either end. -/
def arrangementsWithOneNotAtEnd (n : ℕ) : ℕ :=
  (n - 1) * Nat.factorial (n - 1)

/-- Theorem stating that for 4 people, there are 18 ways to arrange them with one specific person not at either end. -/
theorem arrangementsWithOneNotAtEndFour :
  arrangementsWithOneNotAtEnd 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arrangementsWithOneNotAtEndFour_l302_30209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l302_30270

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  a_def : ∀ n, a n = a 1 + (n - 1) * d

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence)
  (h1 : seq.sum 2 = 10)
  (h2 : seq.sum 5 = 55) :
  ∀ n : ℕ, n ≥ 1 → (seq.a (n + 2) - seq.a n) / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l302_30270


namespace NUMINAMATH_CALUDE_bethany_saw_80_paintings_l302_30266

/-- The number of paintings Bethany saw at the museum -/
structure MuseumVisit where
  portraits : ℕ
  stillLifes : ℕ

/-- Bethany's visit to the museum satisfies the given conditions -/
def bethanysVisit : MuseumVisit where
  portraits := 16
  stillLifes := 4 * 16

/-- The total number of paintings Bethany saw -/
def totalPaintings (visit : MuseumVisit) : ℕ :=
  visit.portraits + visit.stillLifes

/-- Theorem stating that Bethany saw 80 paintings in total -/
theorem bethany_saw_80_paintings :
  totalPaintings bethanysVisit = 80 := by
  sorry

end NUMINAMATH_CALUDE_bethany_saw_80_paintings_l302_30266


namespace NUMINAMATH_CALUDE_incorrect_logical_statement_l302_30265

theorem incorrect_logical_statement : 
  ¬(∀ (p q : Prop), (¬p ∨ ¬q) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_logical_statement_l302_30265


namespace NUMINAMATH_CALUDE_gas_mixture_ratio_l302_30295

-- Define the gases and elements
inductive Gas : Type
| A : Gas  -- CO2
| B : Gas  -- O2

inductive Element : Type
| C : Element
| O : Element

-- Define the molar mass function
def molarMass : Gas → ℝ
| Gas.A => 44  -- Molar mass of CO2
| Gas.B => 32  -- Molar mass of O2

-- Define the number of atoms of each element in each gas molecule
def atomCount : Gas → Element → ℕ
| Gas.A, Element.C => 1
| Gas.A, Element.O => 2
| Gas.B, Element.C => 0
| Gas.B, Element.O => 2

-- Define the mass ratio of C to O in the mixed gas
def massRatio (x y : ℝ) : Prop :=
  (12 * x) / (16 * (2 * x + 2 * y)) = 1 / 8

-- Define the volume ratio of A to B
def volumeRatio (x y : ℝ) : Prop :=
  x / y = 1 / 2

-- The theorem to prove
theorem gas_mixture_ratio : 
  ∀ (x y : ℝ), x > 0 → y > 0 → massRatio x y → volumeRatio x y :=
sorry

end NUMINAMATH_CALUDE_gas_mixture_ratio_l302_30295


namespace NUMINAMATH_CALUDE_probability_not_red_marble_l302_30252

theorem probability_not_red_marble (orange purple red yellow : ℕ) : 
  orange = 4 → purple = 7 → red = 8 → yellow = 5 → 
  (orange + purple + yellow) / (orange + purple + red + yellow) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_red_marble_l302_30252


namespace NUMINAMATH_CALUDE_x_power_x_power_x_at_3_l302_30298

theorem x_power_x_power_x_at_3 :
  let x : ℝ := 3
  (x^x)^(x^x) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_x_power_x_power_x_at_3_l302_30298


namespace NUMINAMATH_CALUDE_ant_walk_probability_l302_30274

/-- Represents a point on the lattice -/
structure Point where
  x : Int
  y : Int

/-- Determines if a point is red (even x+y) or blue (odd x+y) -/
def isRed (p : Point) : Bool :=
  (p.x + p.y) % 2 == 0

/-- Represents the ant's position and the number of steps taken -/
structure AntState where
  position : Point
  steps : Nat

/-- Defines the possible directions the ant can move -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Returns the new point after moving in a given direction -/
def move (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Down => ⟨p.x, p.y - 1⟩
  | Direction.Left => ⟨p.x - 1, p.y⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Defines the probability of the ant being at point C after 4 steps -/
def probAtCAfter4Steps (startPoint : Point) (endPoint : Point) : Real :=
  sorry

/-- The main theorem to prove -/
theorem ant_walk_probability :
  probAtCAfter4Steps ⟨0, 0⟩ ⟨1, 0⟩ = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ant_walk_probability_l302_30274


namespace NUMINAMATH_CALUDE_middle_number_proof_l302_30245

theorem middle_number_proof (x y z : ℕ) (h1 : x < y) (h2 : y < z) 
  (h3 : x + y = 16) (h4 : x + z = 21) (h5 : y + z = 23) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l302_30245


namespace NUMINAMATH_CALUDE_base_k_conversion_l302_30227

/-- Given that 44 in base k equals 36 in base 10, prove that 67 in base 10 equals 103 in base k. -/
theorem base_k_conversion (k : ℕ) (h : 4 * k + 4 = 36) : 
  (67 : ℕ).digits k = [3, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_k_conversion_l302_30227


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_equation_l302_30238

/-- A triangle ABC with side lengths a, b, and c is equilateral if and only if
    a^4 + b^4 + c^4 - a^2b^2 - b^2c^2 - a^2c^2 = 0 -/
theorem triangle_equilateral_iff_equation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  a^4 + b^4 + c^4 - a^2*b^2 - b^2*c^2 - a^2*c^2 = 0 ↔ a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_equation_l302_30238


namespace NUMINAMATH_CALUDE_nils_geese_count_l302_30262

/-- Represents the number of geese on Nils' farm -/
def n : ℕ := sorry

/-- Represents the number of days the feed lasts initially -/
def k : ℕ := sorry

/-- The amount of feed consumed by one goose per day -/
def x : ℝ := sorry

/-- The total amount of feed available -/
def A : ℝ := sorry

/-- The feed lasts k days for n geese -/
axiom initial_feed : A = k * x * n

/-- The feed lasts (k + 20) days for (n - 75) geese -/
axiom sell_75_geese : A = (k + 20) * x * (n - 75)

/-- The feed lasts (k - 15) days for (n + 100) geese -/
axiom buy_100_geese : A = (k - 15) * x * (n + 100)

theorem nils_geese_count : n = 300 := by sorry

end NUMINAMATH_CALUDE_nils_geese_count_l302_30262


namespace NUMINAMATH_CALUDE_nineteenth_triangular_number_l302_30211

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- The 19th triangular number is 210 -/
theorem nineteenth_triangular_number : triangular_number 19 = 210 := by
  sorry

end NUMINAMATH_CALUDE_nineteenth_triangular_number_l302_30211


namespace NUMINAMATH_CALUDE_circumcircle_radius_l302_30230

/-- Given a triangle ABC with side length a = 2 and sin A = 1/3, 
    the radius R of its circumcircle is 3. -/
theorem circumcircle_radius (A B C : ℝ × ℝ) (a : ℝ) (sin_A : ℝ) :
  a = 2 →
  sin_A = 1/3 →
  let R := (a / 2) / sin_A
  R = 3 := by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_l302_30230


namespace NUMINAMATH_CALUDE_nested_sqrt_calculation_l302_30234

theorem nested_sqrt_calculation : Real.sqrt (32 * Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4))) = 16 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_calculation_l302_30234


namespace NUMINAMATH_CALUDE_ranch_feed_corn_cost_l302_30273

/-- Represents the ranch with its animals and pasture. -/
structure Ranch where
  sheep : ℕ
  cattle : ℕ
  pasture_acres : ℕ

/-- Represents the feed requirements and costs. -/
structure FeedInfo where
  cow_grass_per_month : ℕ
  sheep_grass_per_month : ℕ
  corn_bag_cost : ℕ
  cow_corn_months_per_bag : ℕ
  sheep_corn_months_per_bag : ℕ

/-- Calculates the annual cost of feed corn for the ranch. -/
def annual_feed_corn_cost (ranch : Ranch) (feed : FeedInfo) : ℕ :=
  sorry

/-- Theorem stating the annual feed corn cost for the given ranch and feed information. -/
theorem ranch_feed_corn_cost :
  let ranch := Ranch.mk 8 5 144
  let feed := FeedInfo.mk 2 1 10 1 2
  annual_feed_corn_cost ranch feed = 360 :=
sorry

end NUMINAMATH_CALUDE_ranch_feed_corn_cost_l302_30273


namespace NUMINAMATH_CALUDE_fraction_of_seniors_studying_japanese_l302_30207

theorem fraction_of_seniors_studying_japanese 
  (num_juniors : ℝ) 
  (num_seniors : ℝ) 
  (fraction_juniors_studying : ℝ) 
  (fraction_total_studying : ℝ) :
  num_seniors = 3 * num_juniors →
  fraction_juniors_studying = 3 / 4 →
  fraction_total_studying = 0.4375 →
  (fraction_total_studying * (num_juniors + num_seniors) - fraction_juniors_studying * num_juniors) / num_seniors = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_seniors_studying_japanese_l302_30207


namespace NUMINAMATH_CALUDE_parallelogram_with_equilateral_triangles_l302_30282

-- Define the points
variable (A B C D P Q : ℝ × ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2 ∧
  A.1 - D.1 = B.1 - C.1 ∧ A.2 - D.2 = B.2 - C.2

-- Define an equilateral triangle
def is_equilateral_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 ∧
  (Y.1 - Z.1)^2 + (Y.2 - Z.2)^2 = (Z.1 - X.1)^2 + (Z.2 - X.2)^2

-- State the theorem
theorem parallelogram_with_equilateral_triangles
  (h1 : is_parallelogram A B C D)
  (h2 : is_equilateral_triangle B C P)
  (h3 : is_equilateral_triangle C D Q) :
  is_equilateral_triangle A P Q :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_equilateral_triangles_l302_30282


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l302_30247

/-- A quadratic function passing through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

/-- The theorem stating that if the quadratic function passes through the origin, then m = -4 -/
theorem quadratic_through_origin (m : ℝ) :
  (∀ x, quadratic_function m x = 0 → x = 0) →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l302_30247


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_system_l302_30215

theorem no_solution_to_inequality_system :
  ¬∃ (x y z t : ℝ), 
    (abs x > abs (y - z + t)) ∧
    (abs y > abs (x - z + t)) ∧
    (abs z > abs (x - y + t)) ∧
    (abs t > abs (x - y + z)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_system_l302_30215


namespace NUMINAMATH_CALUDE_stadium_attendance_l302_30203

/-- Given a stadium with initial attendees and girls, calculate remaining attendees after some leave --/
def remaining_attendees (total : ℕ) (girls : ℕ) : ℕ :=
  let boys := total - girls
  let boys_left := boys / 4
  let girls_left := girls / 8
  total - (boys_left + girls_left)

/-- Theorem stating that 480 people remain given the initial conditions --/
theorem stadium_attendance : remaining_attendees 600 240 = 480 := by
  sorry

end NUMINAMATH_CALUDE_stadium_attendance_l302_30203


namespace NUMINAMATH_CALUDE_cindy_envelope_distribution_l302_30225

theorem cindy_envelope_distribution (initial_envelopes : ℕ) (friends : ℕ) (remaining_envelopes : ℕ) 
  (h1 : initial_envelopes = 37)
  (h2 : friends = 5)
  (h3 : remaining_envelopes = 22) :
  (initial_envelopes - remaining_envelopes) / friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelope_distribution_l302_30225


namespace NUMINAMATH_CALUDE_great_eighteen_league_games_l302_30290

/-- Calculates the number of games in a soccer league with specified structure -/
def soccer_league_games (divisions : Nat) (teams_per_division : Nat) 
  (intra_division_games : Nat) (inter_division_games : Nat) : Nat :=
  let intra_games := divisions * (teams_per_division.choose 2) * intra_division_games
  let inter_games := divisions.choose 2 * teams_per_division^2 * inter_division_games
  intra_games + inter_games

/-- The Great Eighteen Soccer League game count theorem -/
theorem great_eighteen_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_great_eighteen_league_games_l302_30290


namespace NUMINAMATH_CALUDE_negation_of_all_politicians_are_loyal_l302_30222

universe u

def Politician (α : Type u) := α → Prop
def Loyal (α : Type u) := α → Prop

theorem negation_of_all_politicians_are_loyal 
  {α : Type u} (politician : Politician α) (loyal : Loyal α) :
  (¬ ∀ (x : α), politician x → loyal x) ↔ (∃ (x : α), politician x ∧ ¬ loyal x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_politicians_are_loyal_l302_30222


namespace NUMINAMATH_CALUDE_remainder_problem_l302_30239

theorem remainder_problem (m : ℤ) (k : ℤ) (h : m = 100 * k - 2) : 
  (m^2 + 4*m + 6) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l302_30239


namespace NUMINAMATH_CALUDE_star_example_l302_30205

-- Define the ★ operation
def star (m n p q : ℚ) : ℚ := (m + 1) * (p + 1) * ((q + 1) / (n + 1))

-- Theorem statement
theorem star_example : star (5/11) (11/1) (7/2) (2/1) = 12 := by
  sorry

end NUMINAMATH_CALUDE_star_example_l302_30205


namespace NUMINAMATH_CALUDE_fifty_percent_of_2002_l302_30253

theorem fifty_percent_of_2002 : (50 : ℚ) / 100 * 2002 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_of_2002_l302_30253


namespace NUMINAMATH_CALUDE_percent_of_percent_equality_l302_30277

theorem percent_of_percent_equality (y : ℝ) (h : y ≠ 0) :
  (18 / 100) * y = (30 / 100) * ((60 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_equality_l302_30277


namespace NUMINAMATH_CALUDE_min_time_proof_l302_30235

-- Define the quantities of honey and milk
def honey_pots : ℕ := 10
def milk_cans : ℕ := 22

-- Define the time taken by Pooh and Piglet for honey and milk
def pooh_honey_time : ℕ := 2
def pooh_milk_time : ℕ := 1
def piglet_honey_time : ℕ := 5
def piglet_milk_time : ℕ := 3

-- Define the function to calculate the minimum time
def min_consumption_time : ℕ :=
  -- The actual calculation is not implemented here
  30

-- State the theorem
theorem min_time_proof :
  min_consumption_time = 30 :=
sorry

end NUMINAMATH_CALUDE_min_time_proof_l302_30235


namespace NUMINAMATH_CALUDE_max_salary_cricket_team_l302_30291

/-- Represents a cricket team -/
structure CricketTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player in a cricket team -/
def maxSalary (team : CricketTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem: The maximum possible salary for the highest-paid player in the given cricket team is 416000 -/
theorem max_salary_cricket_team :
  ∃ (team : CricketTeam),
    team.players = 18 ∧
    team.minSalary = 12000 ∧
    team.salaryCap = 620000 ∧
    maxSalary team = 416000 := by
  sorry

end NUMINAMATH_CALUDE_max_salary_cricket_team_l302_30291


namespace NUMINAMATH_CALUDE_max_log_product_l302_30259

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  ∃ (max_val : ℝ), max_val = 4 ∧ ∀ (a b : ℝ), a > 1 → b > 1 → Real.log a / Real.log 10 + Real.log b / Real.log 10 = 4 →
    Real.log x / Real.log 10 * Real.log y / Real.log 10 ≥ Real.log a / Real.log 10 * Real.log b / Real.log 10 :=
by
  sorry

#check max_log_product

end NUMINAMATH_CALUDE_max_log_product_l302_30259


namespace NUMINAMATH_CALUDE_tangent_sum_product_l302_30244

theorem tangent_sum_product (α β γ : Real) (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) = 
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_product_l302_30244


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l302_30279

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (2 * a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → z = -2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l302_30279


namespace NUMINAMATH_CALUDE_coefficient_of_x_l302_30224

/-- Given that for some natural number n:
    1) M = 4^n is the sum of coefficients in (5x - 1/√x)^n
    2) N = 2^n is the sum of binomial coefficients
    3) M - N = 240
    Then the coefficient of x in the expansion of (5x - 1/√x)^n is 150 -/
theorem coefficient_of_x (n : ℕ) (M N : ℝ) 
  (hM : M = 4^n)
  (hN : N = 2^n)
  (hDiff : M - N = 240) :
  ∃ (coeff : ℝ), coeff = 150 ∧ 
  coeff = (-1)^2 * (n.choose 2) * 5^(n-2) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l302_30224


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l302_30213

theorem quadratic_complete_square (x : ℝ) : ∃ (a k : ℝ), 
  3 * x^2 + 8 * x + 15 = a * (x - (-4/3))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l302_30213


namespace NUMINAMATH_CALUDE_max_k_value_l302_30248

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x + 2 * log x

noncomputable def g (k : ℤ) (x : ℝ) : ℝ := (1/2) * x^2 + k*x + (2-x) * log x - k

theorem max_k_value :
  ∃ (k_max : ℤ),
    (∀ (k : ℤ), (∀ (x : ℝ), x > 1 → g k x < f 1 x) → k ≤ k_max) ∧
    (∀ (x : ℝ), x > 1 → g k_max x < f 1 x) ∧
    k_max = 3 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l302_30248


namespace NUMINAMATH_CALUDE_solve_system_l302_30292

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - y = 7) 
  (eq2 : x + 3 * y = 2) : 
  x = 23 / 10 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l302_30292


namespace NUMINAMATH_CALUDE_candy_remaining_l302_30228

theorem candy_remaining (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) 
  (h1 : initial = 21)
  (h2 : first_eaten = 5)
  (h3 : second_eaten = 9) : 
  initial - first_eaten - second_eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l302_30228


namespace NUMINAMATH_CALUDE_m_equals_eight_m_uniqueness_l302_30242

/-- The value of m for which the given conditions are satisfied -/
def find_m : ℝ → Prop := λ m =>
  m ≠ 0 ∧
  ∃ A B : ℝ × ℝ,
    -- Circle equation
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    -- Points A and B are on the directrix of the parabola
    A.1 = -m/4 ∧
    B.1 = -m/4 ∧
    -- Distance between A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 ∧
    -- Parabola equation (not directly used, but implied by the directrix)
    ∀ x y, y^2 = m*x → x ≥ -m/4

/-- Theorem stating that m = 8 satisfies the given conditions -/
theorem m_equals_eight : find_m 8 := by sorry

/-- Theorem stating that 8 is the only value of m that satisfies the given conditions -/
theorem m_uniqueness : ∀ m, find_m m → m = 8 := by sorry

end NUMINAMATH_CALUDE_m_equals_eight_m_uniqueness_l302_30242


namespace NUMINAMATH_CALUDE_unique_b_for_two_integer_solutions_l302_30250

theorem unique_b_for_two_integer_solutions :
  ∃! b : ℤ, ∃! (s : Finset ℤ), s.card = 2 ∧ ∀ x : ℤ, x ∈ s ↔ x^2 + b*x - 2 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_b_for_two_integer_solutions_l302_30250


namespace NUMINAMATH_CALUDE_transmission_time_approx_five_minutes_l302_30264

/-- Represents the data transmission problem --/
structure DataTransmission where
  numBlocks : ℕ
  chunksPerBlock : ℕ
  errorCorrectionRate : ℚ
  transmissionRate : ℕ

/-- Calculates the total number of chunks including error correction --/
def totalChunks (d : DataTransmission) : ℚ :=
  (d.numBlocks * d.chunksPerBlock : ℚ) * (1 + d.errorCorrectionRate)

/-- Calculates the transmission time in seconds --/
def transmissionTimeSeconds (d : DataTransmission) : ℚ :=
  totalChunks d / d.transmissionRate

/-- Calculates the transmission time in minutes --/
def transmissionTimeMinutes (d : DataTransmission) : ℚ :=
  transmissionTimeSeconds d / 60

/-- The main theorem stating that the transmission time is approximately 5 minutes --/
theorem transmission_time_approx_five_minutes
  (d : DataTransmission)
  (h1 : d.numBlocks = 50)
  (h2 : d.chunksPerBlock = 500)
  (h3 : d.errorCorrectionRate = 1/10)
  (h4 : d.transmissionRate = 100) :
  ∃ ε > 0, |transmissionTimeMinutes d - 5| < ε :=
sorry


end NUMINAMATH_CALUDE_transmission_time_approx_five_minutes_l302_30264


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l302_30208

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The x-axis symmetry operation -/
def xAxisSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_wrt_x_axis :
  let original := Point3D.mk (-2) 1 9
  xAxisSymmetry original = Point3D.mk (-2) (-1) (-9) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_x_axis_l302_30208


namespace NUMINAMATH_CALUDE_angle_inequality_l302_30212

theorem angle_inequality : 
  let a := (2 * Real.tan (22.5 * π / 180)) / (1 - Real.tan (22.5 * π / 180) ^ 2)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_angle_inequality_l302_30212


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l302_30246

/-- The radius of a circle inscribed in a rectangle and tangent to four circles -/
theorem inscribed_circle_radius (AB BC : ℝ) (h_AB : AB = 8) (h_BC : BC = 6) : ∃ r : ℝ,
  r > 0 ∧ r < 6 ∧
  (r + 4)^2 = r^2 + r^2 ∧
  (r + 3)^2 = (8 - r)^2 + r^2 ∧
  r = 11 - Real.sqrt 66 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l302_30246


namespace NUMINAMATH_CALUDE_three_fifths_square_specific_number_l302_30233

theorem three_fifths_square_specific_number : 
  (3 / 5 : ℝ) * (14.500000000000002 ^ 2) = 126.15000000000002 := by
  sorry

end NUMINAMATH_CALUDE_three_fifths_square_specific_number_l302_30233


namespace NUMINAMATH_CALUDE_milly_boas_count_l302_30257

theorem milly_boas_count (tail_feathers_per_flamingo : ℕ) 
                         (safe_pluck_percentage : ℚ)
                         (feathers_per_boa : ℕ)
                         (flamingoes_to_harvest : ℕ) :
  tail_feathers_per_flamingo = 20 →
  safe_pluck_percentage = 1/4 →
  feathers_per_boa = 200 →
  flamingoes_to_harvest = 480 →
  (↑flamingoes_to_harvest * safe_pluck_percentage * ↑tail_feathers_per_flamingo) / ↑feathers_per_boa = 12 :=
by sorry

end NUMINAMATH_CALUDE_milly_boas_count_l302_30257


namespace NUMINAMATH_CALUDE_quadratic_vertex_l302_30286

/-- The quadratic function f(x) = 2(x - 4)^2 + 5 -/
def f (x : ℝ) : ℝ := 2 * (x - 4)^2 + 5

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (4, 5)

theorem quadratic_vertex :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l302_30286


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_and_minimum_value_l302_30296

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_and_inequality_and_minimum_value :
  -- 1. The tangent line to f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. f(x) ≥ x - 1 for all x > 0
  (∀ x > 0, f x ≥ x - 1) ∧
  -- 3. The minimum value of a such that f(x) ≥ ax² + 2/a for all x > 0 and a ≠ 0 is -e³
  (∀ a ≠ 0, (∀ x > 0, f x ≥ a * x^2 + 2/a) ↔ a ≥ -Real.exp 3) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_and_minimum_value_l302_30296


namespace NUMINAMATH_CALUDE_reading_difference_l302_30272

theorem reading_difference (min_assigned : ℕ) (harrison_extra : ℕ) (sam_pages : ℕ) :
  min_assigned = 25 →
  harrison_extra = 10 →
  sam_pages = 100 →
  ∃ (pam_pages : ℕ) (harrison_pages : ℕ),
    pam_pages = sam_pages / 2 ∧
    harrison_pages = min_assigned + harrison_extra ∧
    pam_pages > harrison_pages ∧
    pam_pages - harrison_pages = 15 :=
by sorry

end NUMINAMATH_CALUDE_reading_difference_l302_30272


namespace NUMINAMATH_CALUDE_five_people_booth_arrangements_l302_30221

/-- The number of ways to arrange n people in a booth with at most k people on each side -/
def boothArrangements (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange 5 people in a booth with at most 3 people on each side -/
theorem five_people_booth_arrangements :
  boothArrangements 5 3 = 240 := by sorry

end NUMINAMATH_CALUDE_five_people_booth_arrangements_l302_30221


namespace NUMINAMATH_CALUDE_cantor_set_removal_operations_l302_30231

theorem cantor_set_removal_operations (n : ℕ) : 
  (((2 : ℝ) / 3) ^ (n - 1) * (1 / 3) ≥ 1 / 60) ↔ n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_cantor_set_removal_operations_l302_30231


namespace NUMINAMATH_CALUDE_katie_earnings_l302_30220

/-- Calculates the total money earned from selling necklaces -/
def total_money_earned (bead_necklaces gem_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_necklaces) * price_per_necklace

/-- Proves that Katie earned 21 dollars from selling her necklaces -/
theorem katie_earnings : 
  let bead_necklaces : ℕ := 4
  let gem_necklaces : ℕ := 3
  let price_per_necklace : ℕ := 3
  total_money_earned bead_necklaces gem_necklaces price_per_necklace = 21 := by
sorry

end NUMINAMATH_CALUDE_katie_earnings_l302_30220


namespace NUMINAMATH_CALUDE_square_side_length_l302_30263

theorem square_side_length (diagonal : ℝ) (h : diagonal = 2 * Real.sqrt 2) :
  ∃ (side : ℝ), side * side * 2 = diagonal * diagonal ∧ side = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l302_30263


namespace NUMINAMATH_CALUDE_one_third_of_five_times_seven_l302_30243

theorem one_third_of_five_times_seven :
  (1/3 : ℚ) * (5 * 7) = 35/3 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_five_times_seven_l302_30243


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l302_30217

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l302_30217


namespace NUMINAMATH_CALUDE_refrigerator_payment_proof_l302_30284

def refrigerator_problem (first_payment second_payment third_payment : ℝ)
  (first_percent second_percent third_percent sales_tax_rate : ℝ)
  (delivery_fee : ℝ) : Prop :=
  let total_cost := first_payment / first_percent
  let sales_tax := sales_tax_rate * total_cost
  let total_with_tax_and_fee := total_cost + sales_tax + delivery_fee
  let total_payments := first_payment + second_payment + third_payment
  let remaining_payment := total_with_tax_and_fee - total_payments
  remaining_payment = 1137.50

theorem refrigerator_payment_proof :
  refrigerator_problem 875 650 1200 0.25 0.15 0.35 0.075 100 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_payment_proof_l302_30284


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l302_30236

theorem sum_of_roots_cubic : ∃ (A B C : ℝ),
  (3 * A^3 - 9 * A^2 + 6 * A - 4 = 0) ∧
  (3 * B^3 - 9 * B^2 + 6 * B - 4 = 0) ∧
  (3 * C^3 - 9 * C^2 + 6 * C - 4 = 0) ∧
  (A + B + C = 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l302_30236


namespace NUMINAMATH_CALUDE_water_consumption_problem_l302_30269

/-- The water consumption problem -/
theorem water_consumption_problem 
  (total_water : ℝ) 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (later_people : ℕ) 
  (later_days : ℕ) 
  (h1 : total_water = 18.9)
  (h2 : initial_people = 6)
  (h3 : initial_days = 4)
  (h4 : later_people = 7)
  (h5 : later_days = 2) :
  ∃ (x : ℝ), 
    x = 6 ∧ 
    (initial_people * (total_water / (initial_people * initial_days)) * later_days + 
     x * (total_water / (initial_people * initial_days)) * later_days = total_water) := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_problem_l302_30269


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l302_30268

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - x = 0 ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l302_30268


namespace NUMINAMATH_CALUDE_cone_properties_l302_30240

-- Define the cone
structure Cone where
  generatrix : ℝ
  base_diameter : ℝ

-- Define the theorem
theorem cone_properties (c : Cone) 
  (h1 : c.generatrix = 2 * Real.sqrt 5)
  (h2 : c.base_diameter = 4) :
  -- 1. Volume of the cone
  let volume := (1/3) * Real.pi * (c.base_diameter/2)^2 * Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2)
  volume = (16/3) * Real.pi ∧
  -- 2. Minimum distance from any point on a parallel section to the vertex
  let min_distance := (4/5) * Real.sqrt 5
  (∀ r : ℝ, r > 0 → r < c.base_diameter/2 → 
    ∀ p : ℝ × ℝ, p.1^2 + p.2^2 = r^2 → 
      p.1^2 + (Real.sqrt ((2*Real.sqrt 5)^2 - (c.base_diameter/2)^2) - (c.base_diameter/2 - r))^2 ≥ min_distance^2) ∧
  -- 3. Area of the section when it's the center of the circumscribed sphere
  let section_radius := (3/5) * c.base_diameter/2
  Real.pi * section_radius^2 = (36/25) * Real.pi := by
sorry

end NUMINAMATH_CALUDE_cone_properties_l302_30240


namespace NUMINAMATH_CALUDE_quadratic_real_root_l302_30275

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l302_30275


namespace NUMINAMATH_CALUDE_path_length_of_rotating_triangle_l302_30278

/-- Represents a square with side length 4 inches -/
def Square := {s : ℝ // s = 4}

/-- Represents an equilateral triangle with side length 2 inches -/
def EquilateralTriangle := {t : ℝ // t = 2}

/-- Calculates the path length of vertex P during rotations -/
noncomputable def pathLength (square : Square) (triangle : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem stating the path length of vertex P -/
theorem path_length_of_rotating_triangle 
  (square : Square) 
  (triangle : EquilateralTriangle) : 
  pathLength square triangle = (40 * Real.pi) / 3 := by
  sorry

end NUMINAMATH_CALUDE_path_length_of_rotating_triangle_l302_30278


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersection_l302_30293

/-- Given a right triangular prism with base edges a, b, and c,
    when intersected by a plane to form an equilateral triangle with side length d,
    prove that d satisfies the equation: 3d^4 - 100d^2 + 576 = 0 -/
theorem equilateral_triangle_intersection (a b c d : ℝ) : 
  a = 3 → b = 4 → c = 5 → 
  3 * d^4 - 100 * d^2 + 576 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersection_l302_30293


namespace NUMINAMATH_CALUDE_sum_of_u_and_v_l302_30216

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u + 3 * v = -1) : 
  u + v = -41 / 29 := by
sorry

end NUMINAMATH_CALUDE_sum_of_u_and_v_l302_30216


namespace NUMINAMATH_CALUDE_functional_equation_implies_even_l302_30255

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b

theorem functional_equation_implies_even (f : ℝ → ℝ) (k : ℝ) 
    (h_eq : FunctionalEquation f k) (h_nonzero : ∃ x, f x ≠ 0) : 
    ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_even_l302_30255


namespace NUMINAMATH_CALUDE_labor_tools_problem_l302_30267

/-- The price per tool of the first batch of type A labor tools -/
def first_batch_price (total_cost : ℕ) (quantity : ℕ) : ℕ :=
  total_cost / quantity

/-- The price per tool of the second batch of type A labor tools -/
def second_batch_price (first_price : ℕ) (increase : ℕ) : ℕ :=
  first_price + increase

/-- The total cost of the second batch -/
def second_batch_total_cost (price : ℕ) (quantity : ℕ) : ℕ :=
  price * quantity

/-- The maximum number of type A tools in the third batch -/
def max_type_a_tools (type_a_price : ℕ) (type_b_price : ℕ) (total_tools : ℕ) (max_cost : ℕ) : ℕ :=
  min (total_tools) ((max_cost - type_b_price * total_tools) / (type_a_price - type_b_price))

theorem labor_tools_problem (first_total_cost second_total_cost : ℕ) 
  (price_increase : ℕ) (third_batch_total : ℕ) (type_b_price : ℕ) (third_batch_max_cost : ℕ) :
  first_total_cost = 2000 ∧ 
  second_total_cost = 2200 ∧ 
  price_increase = 5 ∧
  third_batch_total = 50 ∧
  type_b_price = 40 ∧
  third_batch_max_cost = 2500 →
  ∃ quantity : ℕ,
    first_batch_price first_total_cost quantity = 50 ∧
    second_batch_price (first_batch_price first_total_cost quantity) price_increase = 
      first_batch_price second_total_cost quantity ∧
    max_type_a_tools 
      (second_batch_price (first_batch_price first_total_cost quantity) price_increase)
      type_b_price third_batch_total third_batch_max_cost = 33 := by
  sorry

end NUMINAMATH_CALUDE_labor_tools_problem_l302_30267


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_five_l302_30280

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_five_l302_30280


namespace NUMINAMATH_CALUDE_real_roots_range_roots_condition_m_value_l302_30271

/-- Quadratic equation parameters -/
def a : ℝ := 1
def b (m : ℝ) : ℝ := 2 * (m - 1)
def c (m : ℝ) : ℝ := m^2 + 2

/-- Discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := (b m)^2 - 4 * a * (c m)

/-- Theorem stating the range of m for real roots -/
theorem real_roots_range (m : ℝ) :
  (∃ x : ℝ, a * x^2 + (b m) * x + (c m) = 0) ↔ m ≤ -1/2 := by sorry

/-- Theorem stating the value of m when the roots satisfy the given condition -/
theorem roots_condition_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (hroots : a * x₁^2 + (b m) * x₁ + (c m) = 0 ∧ a * x₂^2 + (b m) * x₂ + (c m) = 0)
  (hcond : (x₁ - x₂)^2 = 18 - x₁ * x₂) :
  m = -2 := by sorry

end NUMINAMATH_CALUDE_real_roots_range_roots_condition_m_value_l302_30271


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l302_30237

theorem P_sufficient_not_necessary_Q :
  (∀ x y : ℝ, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) :=
by sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_Q_l302_30237


namespace NUMINAMATH_CALUDE_nonconsecutive_choose_18_5_l302_30201

def nonconsecutive_choose (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

theorem nonconsecutive_choose_18_5 :
  nonconsecutive_choose 18 5 = Nat.choose 14 5 :=
sorry

end NUMINAMATH_CALUDE_nonconsecutive_choose_18_5_l302_30201


namespace NUMINAMATH_CALUDE_square_difference_pattern_l302_30229

theorem square_difference_pattern (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l302_30229


namespace NUMINAMATH_CALUDE_probability_ace_two_three_four_l302_30214

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (Ace, 2, 3, 4) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of four cards from a standard deck -/
def probability_four_card_sequence : ℚ :=
  (cards_per_rank : ℚ) / deck_size *
  (cards_per_rank : ℚ) / (deck_size - 1) *
  (cards_per_rank : ℚ) / (deck_size - 2) *
  (cards_per_rank : ℚ) / (deck_size - 3)

/-- The probability of drawing an Ace, 2, 3, and 4 in that order from a standard deck of 52 cards, without replacement, is equal to 16/405525 -/
theorem probability_ace_two_three_four : probability_four_card_sequence = 16 / 405525 := by
  sorry

end NUMINAMATH_CALUDE_probability_ace_two_three_four_l302_30214


namespace NUMINAMATH_CALUDE_remainder_8437_div_9_l302_30204

theorem remainder_8437_div_9 : 8437 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8437_div_9_l302_30204


namespace NUMINAMATH_CALUDE_class_average_score_l302_30251

theorem class_average_score (total_students : ℕ) 
  (assigned_day_percent : ℚ) (makeup_date_percent : ℚ) (later_date_percent : ℚ)
  (assigned_day_score : ℚ) (makeup_date_score : ℚ) (later_date_score : ℚ) :
  total_students = 100 →
  assigned_day_percent = 60 / 100 →
  makeup_date_percent = 30 / 100 →
  later_date_percent = 10 / 100 →
  assigned_day_score = 60 / 100 →
  makeup_date_score = 80 / 100 →
  later_date_score = 75 / 100 →
  (assigned_day_percent * assigned_day_score * total_students +
   makeup_date_percent * makeup_date_score * total_students +
   later_date_percent * later_date_score * total_students) / total_students = 675 / 1000 := by
  sorry

#eval (60 * 60 + 30 * 80 + 10 * 75) / 100  -- Expected output: 67.5

end NUMINAMATH_CALUDE_class_average_score_l302_30251


namespace NUMINAMATH_CALUDE_distribution_problem_l302_30200

theorem distribution_problem (total : ℕ) (a b c : ℕ) : 
  total = 370 →
  total = a + b + c →
  b + c = a + 50 →
  (a : ℚ) / b = (b : ℚ) / c →
  a = 160 ∧ b = 120 ∧ c = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribution_problem_l302_30200


namespace NUMINAMATH_CALUDE_store_b_cheaper_for_40_l302_30254

-- Define the rental fee functions
def y₁ (x : ℕ) : ℝ := 96 * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ 6 then 160 * x else 80 * x + 480

-- Theorem statement
theorem store_b_cheaper_for_40 :
  y₂ 40 < y₁ 40 := by
  sorry

end NUMINAMATH_CALUDE_store_b_cheaper_for_40_l302_30254


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l302_30258

theorem least_subtraction_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬(∃ k : ℕ, 9671 - y = 385 * k)) ∧ 
  (∃ k : ℕ, 9671 - x = 385 * k) → 
  x = 46 := by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l302_30258


namespace NUMINAMATH_CALUDE_fraction_problem_l302_30210

theorem fraction_problem (x y : ℚ) (h : (x / y) * 6 + 6 = 10) : x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l302_30210


namespace NUMINAMATH_CALUDE_catch_game_end_state_l302_30226

/-- Represents the state of the game at each throw -/
structure GameState where
  throw_number : ℕ
  distance : ℕ

/-- Calculates the game state for a given throw number -/
def game_state (n : ℕ) : GameState :=
  { throw_number := n,
    distance := (n + 1) / 2 }

/-- Determines if Pat is throwing based on the throw number -/
def is_pat_throwing (n : ℕ) : Prop :=
  n % 2 = 1

theorem catch_game_end_state :
  let final_throw := 29
  let final_state := game_state final_throw
  final_state.distance = 15 ∧ is_pat_throwing final_throw := by
sorry

end NUMINAMATH_CALUDE_catch_game_end_state_l302_30226


namespace NUMINAMATH_CALUDE_complex_number_equality_l302_30287

theorem complex_number_equality (z : ℂ) (h : z * Complex.I = 2 - 2 * Complex.I) : z = -2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l302_30287


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l302_30218

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 6 + a 8 + a 10 = 72) :
  2 * a 10 - a 12 = 24 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l302_30218


namespace NUMINAMATH_CALUDE_not_both_nonstandard_l302_30241

def IntegerFunction (G : ℤ → ℤ) : Prop :=
  ∀ c : ℤ, ∃ x : ℤ, G x ≠ c

def NonStandard (G : ℤ → ℤ) (a : ℤ) : Prop :=
  ∀ x : ℤ, G x = G (a - x)

theorem not_both_nonstandard (G : ℤ → ℤ) (h : IntegerFunction G) :
  ¬(NonStandard G 267 ∧ NonStandard G 269) := by
  sorry

end NUMINAMATH_CALUDE_not_both_nonstandard_l302_30241
