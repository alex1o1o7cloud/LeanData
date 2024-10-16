import Mathlib

namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3792_379278

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (a b : ℝ) : ℝ := b

/-- Given a line with equation y = 2x - 1, prove that its y-intercept is -1. -/
theorem y_intercept_of_line (x y : ℝ) (h : y = 2 * x - 1) : y_intercept 2 (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3792_379278


namespace NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l3792_379262

theorem x_zero_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0) ∧
  ¬(∀ x y : ℝ, x = 0 → x^2 + y^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_x_zero_necessary_not_sufficient_l3792_379262


namespace NUMINAMATH_CALUDE_product_plus_number_equals_93_l3792_379248

theorem product_plus_number_equals_93 : ∃ x : ℤ, (-11 * -8) + x = 93 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_product_plus_number_equals_93_l3792_379248


namespace NUMINAMATH_CALUDE_probability_two_females_one_male_l3792_379290

theorem probability_two_females_one_male (total : ℕ) (females : ℕ) (males : ℕ) (chosen : ℕ) :
  total = females + males →
  total = 8 →
  females = 5 →
  males = 3 →
  chosen = 3 →
  (Nat.choose females 2 * Nat.choose males 1 : ℚ) / Nat.choose total chosen = 15 / 28 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_females_one_male_l3792_379290


namespace NUMINAMATH_CALUDE_valid_combination_exists_l3792_379203

/-- Represents a combination of cards -/
structure CardCombination where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Checks if a card combination is valid according to the given conditions -/
def isValidCombination (c : CardCombination) : Prop :=
  c.red + c.blue + c.green = 20 ∧
  c.red ≥ 2 ∧
  c.blue ≥ 3 ∧
  c.green ≥ 1 ∧
  3 * c.red + 5 * c.blue + 7 * c.green = 84

/-- There exists a valid card combination that satisfies all conditions -/
theorem valid_combination_exists : ∃ c : CardCombination, isValidCombination c := by
  sorry

#check valid_combination_exists

end NUMINAMATH_CALUDE_valid_combination_exists_l3792_379203


namespace NUMINAMATH_CALUDE_factorial_equation_l3792_379272

theorem factorial_equation : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l3792_379272


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3792_379264

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of three consecutive terms in an arithmetic sequence -/
def sum_three_consecutive (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n + a (n + 1) + a (n + 2)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  sum_three_consecutive a 4 = 36 →
  a 1 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3792_379264


namespace NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l3792_379230

theorem mrs_hilt_pizza_slices :
  ∀ (num_pizzas : ℕ) (slices_per_pizza : ℕ),
    num_pizzas = 5 →
    slices_per_pizza = 12 →
    num_pizzas * slices_per_pizza = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizza_slices_l3792_379230


namespace NUMINAMATH_CALUDE_part_i_part_ii_l3792_379247

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, -x^2 + 3*m - 1 ≤ 0

def Q (m a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, m - a*x ≤ 0

-- Part (i)
theorem part_i (m : ℝ) : 
  (¬(P m) ∧ ¬(Q m 1) ∧ (P m ∨ Q m 1)) → (1/3 < m ∧ m ≤ 1) :=
sorry

-- Part (ii)
theorem part_ii (m a : ℝ) :
  ((P m → Q m a) ∧ ¬(Q m a → P m)) → (a ≥ 1/3 ∨ a ≤ -1/3) :=
sorry

end NUMINAMATH_CALUDE_part_i_part_ii_l3792_379247


namespace NUMINAMATH_CALUDE_fraction_of_A_students_l3792_379265

/-- In a class where some students received A's, prove that the fraction of students 
    who received A's is 0.7, given the following conditions:
    - 0.2 fraction of students received B's
    - 0.9 fraction of students received either A's or B's -/
theorem fraction_of_A_students (fractionB : Real) (fractionAorB : Real) 
  (hB : fractionB = 0.2) 
  (hAorB : fractionAorB = 0.9) : 
  fractionAorB - fractionB = 0.7 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_A_students_l3792_379265


namespace NUMINAMATH_CALUDE_mandatory_work_effect_l3792_379202

/-- Represents the labor market for doctors -/
structure DoctorLaborMarket where
  state_supply : ℝ → ℝ  -- Supply function for state sector
  state_demand : ℝ → ℝ  -- Demand function for state sector
  private_supply : ℝ → ℝ  -- Supply function for private sector
  private_demand : ℝ → ℝ  -- Demand function for private sector

/-- Represents the policy of mandatory work in public healthcare -/
structure MandatoryWorkPolicy where
  years_required : ℕ  -- Number of years required in public healthcare

/-- The equilibrium wage in the state sector -/
def state_equilibrium_wage (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The equilibrium price in the private healthcare sector -/
def private_equilibrium_price (market : DoctorLaborMarket) : ℝ :=
  sorry

/-- The effect of the mandatory work policy on the labor market -/
def apply_policy (market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) : DoctorLaborMarket :=
  sorry

theorem mandatory_work_effect (initial_market : DoctorLaborMarket) (policy : MandatoryWorkPolicy) :
  let final_market := apply_policy initial_market policy
  state_equilibrium_wage final_market > state_equilibrium_wage initial_market ∧
  private_equilibrium_price final_market < private_equilibrium_price initial_market :=
sorry

end NUMINAMATH_CALUDE_mandatory_work_effect_l3792_379202


namespace NUMINAMATH_CALUDE_molar_mass_not_unique_l3792_379281

/-- Represents a solution with a solute -/
structure Solution :=
  (mass_fraction : ℝ)
  (mass : ℝ)

/-- Represents the result of mixing two solutions and evaporating water -/
structure MixedSolution :=
  (solution1 : Solution)
  (solution2 : Solution)
  (evaporated_water : ℝ)
  (final_molarity : ℝ)

/-- Function to calculate molar mass given additional information -/
noncomputable def calculate_molar_mass (mixed : MixedSolution) (additional_info : ℝ) : ℝ :=
  sorry

/-- Theorem stating that molar mass cannot be uniquely determined without additional information -/
theorem molar_mass_not_unique (mixed : MixedSolution) :
  ∃ (info1 info2 : ℝ), info1 ≠ info2 ∧ 
  calculate_molar_mass mixed info1 ≠ calculate_molar_mass mixed info2 :=
sorry

end NUMINAMATH_CALUDE_molar_mass_not_unique_l3792_379281


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l3792_379298

/-- The line x = my + 1 intersects the parabola y² = x at two distinct points for any real m -/
theorem line_parabola_intersection (m : ℝ) : 
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ 
  (y₁^2 = m * y₁ + 1) ∧ 
  (y₂^2 = m * y₂ + 1) := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l3792_379298


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3792_379233

theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a/b + b/c + c/a) + (b/a + c/b + a/c) = 9) :
  (a/b + b/c + c/a)^2 + (b/a + c/b + a/c)^2 ≥ 45 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3792_379233


namespace NUMINAMATH_CALUDE_trail_mix_composition_l3792_379288

/-- The weight of peanuts used in the trail mix -/
def peanuts : ℚ := 0.16666666666666666

/-- The weight of raisins used in the trail mix -/
def raisins : ℚ := 0.08333333333333333

/-- The total weight of the trail mix -/
def total_mix : ℚ := 0.4166666666666667

/-- The weight of chocolate chips used in the trail mix -/
def chocolate_chips : ℚ := total_mix - (peanuts + raisins)

theorem trail_mix_composition :
  chocolate_chips = 0.1666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_composition_l3792_379288


namespace NUMINAMATH_CALUDE_sprint_tournament_races_l3792_379226

/-- Calculates the minimum number of races needed to determine a champion -/
def minimumRaces (totalSprinters : Nat) (lanesPerRace : Nat) : Nat :=
  let eliminationsNeeded := totalSprinters - 1
  let eliminationsPerRace := lanesPerRace - 1
  (eliminationsNeeded + eliminationsPerRace - 1) / eliminationsPerRace

theorem sprint_tournament_races : 
  minimumRaces 256 8 = 37 := by
  sorry

#eval minimumRaces 256 8

end NUMINAMATH_CALUDE_sprint_tournament_races_l3792_379226


namespace NUMINAMATH_CALUDE_sheep_buying_problem_l3792_379209

theorem sheep_buying_problem (x : ℝ) : 
  (∃ n : ℕ, n * 5 + 45 = x ∧ n * 7 + 3 = x) → (x - 45) / 5 = (x - 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_sheep_buying_problem_l3792_379209


namespace NUMINAMATH_CALUDE_karen_average_speed_l3792_379211

/-- Calculates the time difference in hours between two times given in hours and minutes -/
def timeDifference (start_hour start_minute end_hour end_minute : ℕ) : ℚ :=
  (end_hour - start_hour : ℚ) + (end_minute - start_minute : ℚ) / 60

/-- Calculates the average speed given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ :=
  distance / time

theorem karen_average_speed :
  let start_time : ℕ × ℕ := (9, 40)  -- (hour, minute)
  let end_time : ℕ × ℕ := (13, 20)   -- (hour, minute)
  let distance : ℚ := 198
  let time := timeDifference start_time.1 start_time.2 end_time.1 end_time.2
  averageSpeed distance time = 54 := by sorry

end NUMINAMATH_CALUDE_karen_average_speed_l3792_379211


namespace NUMINAMATH_CALUDE_max_value_fraction_l3792_379204

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2 + x*y) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3792_379204


namespace NUMINAMATH_CALUDE_cuboid_faces_at_vertex_l3792_379238

/-- A cuboid is a three-dimensional geometric shape -/
structure Cuboid where

/-- A vertex is a point where edges of a geometric shape meet -/
structure Vertex where

/-- Represents a face of a geometric shape -/
structure Face where

/-- The number of faces meeting at a vertex of a cuboid -/
def faces_at_vertex (c : Cuboid) (v : Vertex) : ℕ := sorry

/-- Theorem stating that the number of faces meeting at a vertex of a cuboid is 3 -/
theorem cuboid_faces_at_vertex (c : Cuboid) (v : Vertex) : 
  faces_at_vertex c v = 3 := by sorry

end NUMINAMATH_CALUDE_cuboid_faces_at_vertex_l3792_379238


namespace NUMINAMATH_CALUDE_inequality_and_maximum_l3792_379241

theorem inequality_and_maximum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 3) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3) ∧
  (c = a * b → ∀ c', c' = a * b → c' ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_maximum_l3792_379241


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3792_379243

theorem polynomial_simplification (q : ℝ) :
  (5 * q^3 - 7 * q + 8) + (3 - 9 * q^2 + 3 * q) = 5 * q^3 - 9 * q^2 - 4 * q + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3792_379243


namespace NUMINAMATH_CALUDE_max_visitable_halls_is_91_l3792_379240

/-- Represents a triangular castle divided into smaller triangular halls. -/
structure TriangularCastle where
  total_halls : ℕ
  side_length : ℝ
  hall_side_length : ℝ

/-- Represents a path through the castle halls. -/
def VisitPath (castle : TriangularCastle) := List ℕ

/-- Checks if a path is valid (no repeated visits). -/
def is_valid_path (castle : TriangularCastle) (path : VisitPath castle) : Prop :=
  path.length ≤ castle.total_halls ∧ path.Nodup

/-- The maximum number of halls that can be visited. -/
def max_visitable_halls (castle : TriangularCastle) : ℕ :=
  91

/-- Theorem stating that the maximum number of visitable halls is 91. -/
theorem max_visitable_halls_is_91 (castle : TriangularCastle) 
  (h1 : castle.total_halls = 100)
  (h2 : castle.side_length = 100)
  (h3 : castle.hall_side_length = 10) :
  ∀ (path : VisitPath castle), is_valid_path castle path → path.length ≤ max_visitable_halls castle :=
by sorry

end NUMINAMATH_CALUDE_max_visitable_halls_is_91_l3792_379240


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l3792_379206

/-- Represents the odometer reading as a three-digit number -/
structure OdometerReading where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds + tens + ones ≤ 9

/-- Represents a car trip -/
structure CarTrip where
  hours : Nat
  speed : Nat
  initial : OdometerReading
  final : OdometerReading
  valid : speed = 65 ∧
          final.hundreds = initial.ones ∧
          final.tens = initial.tens ∧
          final.ones = initial.hundreds

theorem odometer_sum_squares (trip : CarTrip) :
  trip.initial.hundreds ^ 2 + trip.initial.tens ^ 2 + trip.initial.ones ^ 2 = 41 :=
sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l3792_379206


namespace NUMINAMATH_CALUDE_inequality_solution_l3792_379249

-- Define the given inequality and its solution set
def given_inequality (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3*x + 2 > 0
def solution_set (b : ℝ) (x : ℝ) : Prop := x < 1 ∨ x > b

-- Define the values to be proven
def a_value : ℝ := 1
def b_value : ℝ := 2

-- Define the new inequality
def new_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 - (2*m + 1)*x + 2 < 0

-- Define the solution sets for different m values
def solution_set_m_zero (x : ℝ) : Prop := x > 2
def solution_set_m_gt_half (m : ℝ) (x : ℝ) : Prop := 1/m < x ∧ x < 2
def solution_set_m_half : Set ℝ := ∅
def solution_set_m_between_zero_half (m : ℝ) (x : ℝ) : Prop := 2 < x ∧ x < 1/m
def solution_set_m_neg (m : ℝ) (x : ℝ) : Prop := x < 1/m ∨ x > 2

-- State the theorem
theorem inequality_solution :
  (∀ x, given_inequality a_value x ↔ solution_set b_value x) ∧
  (∀ m x, new_inequality m x ↔
    (m = 0 ∧ solution_set_m_zero x) ∨
    (m > 1/2 ∧ solution_set_m_gt_half m x) ∨
    (m = 1/2 ∧ x ∈ solution_set_m_half) ∨
    (0 < m ∧ m < 1/2 ∧ solution_set_m_between_zero_half m x) ∨
    (m < 0 ∧ solution_set_m_neg m x)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3792_379249


namespace NUMINAMATH_CALUDE_election_votes_l3792_379270

theorem election_votes (total_votes : ℕ) (winner_percentage : ℚ) (vote_difference : ℕ) :
  total_votes > 0 →
  winner_percentage = 62 / 100 →
  vote_difference = 300 →
  ⌊total_votes * winner_percentage⌋ - ⌊total_votes * (1 - winner_percentage)⌋ = vote_difference →
  ⌊total_votes * winner_percentage⌋ = 775 :=
by
  sorry

end NUMINAMATH_CALUDE_election_votes_l3792_379270


namespace NUMINAMATH_CALUDE_abc_is_cube_l3792_379220

theorem abc_is_cube (a b c : ℕ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → ¬(n ∣ a) ∧ ¬(n ∣ b) ∧ ¬(n ∣ c)) →
  (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∀ n : ℕ, 2 ≤ n ∧ n ≤ 100 → ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z)) →
    a ≤ x ∧ b ≤ y ∧ c ≤ z) →
  ∃ k : ℕ, a * b * c = k^3 :=
by sorry

end NUMINAMATH_CALUDE_abc_is_cube_l3792_379220


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3792_379225

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 + 11 - 7) = Real.sqrt (5 + 11) - Real.sqrt x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3792_379225


namespace NUMINAMATH_CALUDE_sin_inequality_solution_set_l3792_379234

theorem sin_inequality_solution_set (a : ℝ) (θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (h3 : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2 * n - 1 : ℝ) * π - θ < x ∧ x < (2 * n : ℝ) * π + θ} =
  {x : ℝ | Real.sin x < a} :=
by sorry

end NUMINAMATH_CALUDE_sin_inequality_solution_set_l3792_379234


namespace NUMINAMATH_CALUDE_rulers_placed_l3792_379229

theorem rulers_placed (initial_rulers final_rulers : ℕ) (h : final_rulers = initial_rulers + 14) :
  final_rulers - initial_rulers = 14 := by
  sorry

end NUMINAMATH_CALUDE_rulers_placed_l3792_379229


namespace NUMINAMATH_CALUDE_fruit_arrangement_problem_l3792_379207

def number_of_arrangements (n : ℕ) (a b c d : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial a * Nat.factorial b * Nat.factorial c * Nat.factorial d)

theorem fruit_arrangement_problem : number_of_arrangements 10 4 3 2 1 = 12600 := by
  sorry

end NUMINAMATH_CALUDE_fruit_arrangement_problem_l3792_379207


namespace NUMINAMATH_CALUDE_l₂_slope_l3792_379218

-- Define the slope and y-intercept of line l₁
def m₁ : ℝ := 2
def b₁ : ℝ := 3

-- Define the equation of line l₁
def l₁ (x y : ℝ) : Prop := y = m₁ * x + b₁

-- Define the equation of the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry relation between two points
def symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  symmetry_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Define line l₂ as symmetric to l₁ with respect to y = -x
def l₂ (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), l₁ x₁ y₁ ∧ symmetric x₁ y₁ x y

-- State the theorem
theorem l₂_slope :
  ∃ (m₂ : ℝ), m₂ = 1/2 ∧ ∀ (x y : ℝ), l₂ x y → ∃ (b₂ : ℝ), y = m₂ * x + b₂ :=
sorry

end NUMINAMATH_CALUDE_l₂_slope_l3792_379218


namespace NUMINAMATH_CALUDE_problem_solution_l3792_379222

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x * y = 2 * a) 
  (h2 : x * z = 3 * b) 
  (h3 : y * z = 4 * c) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (3*a*b)/(2*c) + (8*a*c)/(3*b) + (6*b*c)/a ∧ 
  x*y*z = 2 * Real.sqrt (6*a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3792_379222


namespace NUMINAMATH_CALUDE_solution_set_characterization_l3792_379260

def valid_digit (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def base_10_value (x y z : ℕ) : ℕ := 100 * x + 10 * y + z

def base_7_value (x y z : ℕ) : ℕ := 49 * x + 7 * y + z

def satisfies_equation (x y z : ℕ) : Prop :=
  base_10_value x y z = 2 * base_7_value x y z

def valid_triple (x y z : ℕ) : Prop :=
  valid_digit x ∧ valid_digit y ∧ valid_digit z ∧ satisfies_equation x y z

theorem solution_set_characterization :
  {t : ℕ × ℕ × ℕ | valid_triple t.1 t.2.1 t.2.2} =
  {(3,1,2), (5,2,2), (4,1,4), (6,2,4), (5,1,6)} := by sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l3792_379260


namespace NUMINAMATH_CALUDE_sara_balloons_l3792_379285

/-- Given that Sara initially had 31 red balloons and gave away 24 red balloons,
    prove that she is left with 7 red balloons. -/
theorem sara_balloons (initial_red : Nat) (given_away : Nat) (remaining : Nat) 
    (h1 : initial_red = 31)
    (h2 : given_away = 24)
    (h3 : remaining = initial_red - given_away) : 
  remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_sara_balloons_l3792_379285


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3792_379263

theorem triangle_angle_measure (A B C : Real) (a b c : Real) :
  A ∈ Set.Ioo 0 π →
  B ∈ Set.Ioo 0 π →
  b * Real.sin A + a * Real.cos B = 0 →
  B = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3792_379263


namespace NUMINAMATH_CALUDE_job_completion_time_work_completion_time_jose_raju_l3792_379256

/-- The time required for two workers to complete a job together, given their individual completion times -/
theorem job_completion_time (jose_time raju_time : ℝ) (jose_time_pos : jose_time > 0) (raju_time_pos : raju_time > 0) :
  (1 / jose_time + 1 / raju_time)⁻¹ = (jose_time * raju_time) / (jose_time + raju_time) :=
by sorry

theorem work_completion_time_jose_raju :
  let jose_time : ℝ := 10
  let raju_time : ℝ := 40
  (1 / jose_time + 1 / raju_time)⁻¹ = 8 :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_work_completion_time_jose_raju_l3792_379256


namespace NUMINAMATH_CALUDE_cycle_price_proof_l3792_379232

/-- Proves that a cycle sold at a 12% loss for Rs. 1232 had an original price of Rs. 1400 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1232)
  (h2 : loss_percentage = 12) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1400 := by
  sorry

#check cycle_price_proof

end NUMINAMATH_CALUDE_cycle_price_proof_l3792_379232


namespace NUMINAMATH_CALUDE_decaf_coffee_percentage_l3792_379236

/-- Proves that the percentage of decaffeinated coffee in the initial stock is 40% --/
theorem decaf_coffee_percentage
  (initial_stock : ℝ)
  (additional_purchase : ℝ)
  (decaf_percent_additional : ℝ)
  (decaf_percent_total : ℝ)
  (h1 : initial_stock = 400)
  (h2 : additional_purchase = 100)
  (h3 : decaf_percent_additional = 60)
  (h4 : decaf_percent_total = 44)
  (h5 : decaf_percent_total / 100 * (initial_stock + additional_purchase) =
        (initial_stock * x / 100) + (additional_purchase * decaf_percent_additional / 100)) :
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_decaf_coffee_percentage_l3792_379236


namespace NUMINAMATH_CALUDE_x_greater_than_half_l3792_379279

theorem x_greater_than_half (x : ℝ) (h1 : 1 / x^2 < 4) (h2 : 1 / x > -2) (h3 : x ≠ 0) : x > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l3792_379279


namespace NUMINAMATH_CALUDE_cindy_marbles_l3792_379286

theorem cindy_marbles (initial_marbles : Nat) (friends : Nat) (marbles_per_friend : Nat) : 
  initial_marbles = 500 → friends = 4 → marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l3792_379286


namespace NUMINAMATH_CALUDE_infinite_cube_differences_l3792_379212

theorem infinite_cube_differences (n : ℕ+) : 
  (∃ p : ℕ+, 3 * p + 1 = (n + 1)^3 - n^3) ∧ 
  (∃ q : ℕ+, 5 * q + 1 = (5 * n + 1)^3 - (5 * n)^3) := by
  sorry

end NUMINAMATH_CALUDE_infinite_cube_differences_l3792_379212


namespace NUMINAMATH_CALUDE_mangoes_quantity_l3792_379289

/-- The quantity of mangoes purchased by Harkamal -/
def mangoes_kg : ℕ := sorry

/-- The price of grapes per kg -/
def grapes_price : ℕ := 70

/-- The price of mangoes per kg -/
def mangoes_price : ℕ := 45

/-- The quantity of grapes purchased in kg -/
def grapes_kg : ℕ := 8

/-- The total amount paid -/
def total_paid : ℕ := 965

theorem mangoes_quantity :
  grapes_kg * grapes_price + mangoes_kg * mangoes_price = total_paid ∧
  mangoes_kg = 9 := by sorry

end NUMINAMATH_CALUDE_mangoes_quantity_l3792_379289


namespace NUMINAMATH_CALUDE_students_liking_sports_l3792_379266

theorem students_liking_sports (B C : Finset Nat) 
  (hB : B.card = 10)
  (hC : C.card = 8)
  (hBC : (B ∩ C).card = 4) :
  (B ∪ C).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l3792_379266


namespace NUMINAMATH_CALUDE_super_rare_snake_price_multiplier_l3792_379255

/-- Represents the snake selling scenario --/
structure SnakeScenario where
  num_snakes : Nat
  eggs_per_snake : Nat
  regular_price : Nat
  total_revenue : Nat

/-- Calculates the price multiplier of the super rare snake --/
def super_rare_price_multiplier (scenario : SnakeScenario) : Nat :=
  let total_eggs := scenario.num_snakes * scenario.eggs_per_snake
  let num_regular_snakes := total_eggs - 1
  let regular_revenue := num_regular_snakes * scenario.regular_price
  let super_rare_price := scenario.total_revenue - regular_revenue
  super_rare_price / scenario.regular_price

/-- Theorem stating the price multiplier of the super rare snake --/
theorem super_rare_snake_price_multiplier :
  let scenario := SnakeScenario.mk 3 2 250 2250
  super_rare_price_multiplier scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_super_rare_snake_price_multiplier_l3792_379255


namespace NUMINAMATH_CALUDE_congruence_systems_solvability_l3792_379201

theorem congruence_systems_solvability :
  (∃ x : ℤ, x ≡ 2 [ZMOD 3] ∧ x ≡ 6 [ZMOD 14]) ∧
  (¬ ∃ x : ℤ, x ≡ 5 [ZMOD 12] ∧ x ≡ 7 [ZMOD 15]) ∧
  (∃ x : ℤ, x ≡ 10 [ZMOD 12] ∧ x ≡ 16 [ZMOD 21]) :=
by sorry

end NUMINAMATH_CALUDE_congruence_systems_solvability_l3792_379201


namespace NUMINAMATH_CALUDE_percentage_problem_l3792_379292

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 780 →
  (25 / 100) * x = (p / 100) * 1500 - 30 →
  p = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l3792_379292


namespace NUMINAMATH_CALUDE_triangle_side_length_l3792_379297

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : c = Real.sqrt 2)
  (h2 : b = Real.sqrt 6)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : A + B + C = π) -- sum of angles in a triangle
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) -- positive side lengths
  (h6 : a / (Real.sin A) = b / (Real.sin B)) -- sine rule
  (h7 : b / (Real.sin B) = c / (Real.sin C)) -- sine rule
  : a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3792_379297


namespace NUMINAMATH_CALUDE_max_value_of_f_l3792_379268

-- Define the function f(x)
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- State the theorem
theorem max_value_of_f (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 2, f b x ≥ 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = 1) →
  (∃ x ∈ Set.Icc (-1) 2, f b x = max 13 (4 + 2*Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3792_379268


namespace NUMINAMATH_CALUDE_angle_equivalence_l3792_379210

/-- Proves that 2023° is equivalent to -137° in the context of angle measurements -/
theorem angle_equivalence : ∃ (k : ℤ), 2023 = -137 + 360 * k := by sorry

end NUMINAMATH_CALUDE_angle_equivalence_l3792_379210


namespace NUMINAMATH_CALUDE_age_ratio_is_four_thirds_l3792_379217

-- Define the current ages of Arun and Deepak
def arun_current_age : ℕ := 26 - 6
def deepak_current_age : ℕ := 15

-- Define the ratio of their ages
def age_ratio : ℚ := arun_current_age / deepak_current_age

-- Theorem to prove
theorem age_ratio_is_four_thirds : age_ratio = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_four_thirds_l3792_379217


namespace NUMINAMATH_CALUDE_copperfield_numbers_l3792_379287

theorem copperfield_numbers : ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
  (∃ (k₁ k₂ k₃ : ℕ+), 
    x₁ * (3 ^ k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * (3 ^ k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * (3 ^ k₃.val) = x₃ + 2500 * k₃.val) :=
by sorry

end NUMINAMATH_CALUDE_copperfield_numbers_l3792_379287


namespace NUMINAMATH_CALUDE_toys_cost_price_gained_l3792_379294

/-- Calculates the number of toys' cost price gained in a sale --/
theorem toys_cost_price_gained
  (num_toys : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_toy : ℕ)
  (h1 : num_toys = 18)
  (h2 : total_selling_price = 16800)
  (h3 : cost_price_per_toy = 800) :
  (total_selling_price - num_toys * cost_price_per_toy) / cost_price_per_toy = 3 :=
by sorry

end NUMINAMATH_CALUDE_toys_cost_price_gained_l3792_379294


namespace NUMINAMATH_CALUDE_line_intercept_l3792_379221

/-- A line with slope 2 and y-intercept m passing through (1,1) has m = -1 -/
theorem line_intercept (m : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + m)  -- Equation of the line
  → 1 = 2 * 1 + m             -- Line passes through (1,1)
  → m = -1                    -- The y-intercept is -1
:= by sorry

end NUMINAMATH_CALUDE_line_intercept_l3792_379221


namespace NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3792_379245

theorem percentage_of_boys_studying_science (total_boys : ℕ) (boys_from_A : ℕ) (boys_A_not_science : ℕ) :
  total_boys = 150 →
  boys_from_A = (20 : ℕ) * total_boys / 100 →
  boys_A_not_science = 21 →
  (boys_from_A - boys_A_not_science) * 100 / boys_from_A = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_studying_science_l3792_379245


namespace NUMINAMATH_CALUDE_taxi_speed_taxi_speed_is_45_l3792_379258

/-- The speed of a taxi that overtakes a bus under specific conditions. -/
theorem taxi_speed : ℝ → Prop :=
  fun v =>
    (∀ (bus_distance : ℝ),
      bus_distance = 4 * (v - 30) →  -- Distance covered by bus in 4 hours
      bus_distance + 2 * (v - 30) = 2 * v) →  -- Taxi covers bus distance in 2 hours
    v = 45

/-- Proof of the taxi speed theorem. -/
theorem taxi_speed_is_45 : taxi_speed 45 := by
  sorry

end NUMINAMATH_CALUDE_taxi_speed_taxi_speed_is_45_l3792_379258


namespace NUMINAMATH_CALUDE_digit_count_l3792_379244

theorem digit_count (n : ℕ) 
  (h1 : (n : ℚ) * 18 = n * 18) 
  (h2 : 4 * 8 = 32) 
  (h3 : 5 * 26 = 130) 
  (h4 : n * 18 = 32 + 130) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_l3792_379244


namespace NUMINAMATH_CALUDE_fraction_chain_l3792_379213

theorem fraction_chain (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_fraction_chain_l3792_379213


namespace NUMINAMATH_CALUDE_plane_sphere_sum_l3792_379253

-- Define the origin
def O : ℝ × ℝ × ℝ := (0, 0, 0)

-- Define the fixed point (2a, 2b, 2c)
def fixed_point (a b c : ℝ) : ℝ × ℝ × ℝ := (2*a, 2*b, 2*c)

-- Define the points A, B, C on the axes
def A (α : ℝ) : ℝ × ℝ × ℝ := (α, 0, 0)
def B (β : ℝ) : ℝ × ℝ × ℝ := (0, β, 0)
def C (γ : ℝ) : ℝ × ℝ × ℝ := (0, 0, γ)

-- Define the center of the sphere
def sphere_center (p q r : ℝ) : ℝ × ℝ × ℝ := (p, q, r)

-- State the theorem
theorem plane_sphere_sum (a b c p q r α β γ : ℝ) 
  (h1 : A α ≠ O) (h2 : B β ≠ O) (h3 : C γ ≠ O)
  (h4 : sphere_center p q r ≠ O)
  (h5 : ∃ (t : ℝ), t * (2*a) / α + t * (2*b) / β + t * (2*c) / γ = t) 
  (h6 : ∀ (x y z : ℝ), (x - p)^2 + (y - q)^2 + (z - r)^2 = p^2 + q^2 + r^2 → 
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = α ∧ y = 0 ∧ z = 0) ∨ 
    (x = 0 ∧ y = β ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = γ)) :
  (2*a)/p + (2*b)/q + (2*c)/r = 2 := by
sorry

end NUMINAMATH_CALUDE_plane_sphere_sum_l3792_379253


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3792_379267

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 - 3*x - 9 = 0) :
  x^3 - 3*x^2 - 9*x + 27 = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3792_379267


namespace NUMINAMATH_CALUDE_adam_tshirts_correct_l3792_379282

/-- The number of t-shirts Adam initially took out -/
def adam_tshirts : ℕ := 20

/-- The total number of clothing items donated -/
def total_donated : ℕ := 126

/-- The number of Adam's friends who donated -/
def friends_donating : ℕ := 3

/-- The number of pants Adam took out -/
def adam_pants : ℕ := 4

/-- The number of jumpers Adam took out -/
def adam_jumpers : ℕ := 4

/-- The number of pajama sets Adam took out -/
def adam_pajamas : ℕ := 4

/-- Theorem stating that the number of t-shirts Adam initially took out is correct -/
theorem adam_tshirts_correct : 
  (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) / 2 + 
  friends_donating * (adam_pants + adam_jumpers + 2 * adam_pajamas + adam_tshirts) = 
  total_donated :=
sorry

end NUMINAMATH_CALUDE_adam_tshirts_correct_l3792_379282


namespace NUMINAMATH_CALUDE_concentric_circles_k_value_l3792_379250

/-- Two concentric circles with center at the origin --/
structure ConcentricCircles where
  largeRadius : ℝ
  smallRadius : ℝ

/-- The point P on the larger circle --/
def P : ℝ × ℝ := (10, 6)

/-- The point S on the smaller circle --/
def S (k : ℝ) : ℝ × ℝ := (0, k)

/-- The distance QR --/
def QR : ℝ := 4

theorem concentric_circles_k_value (circles : ConcentricCircles) 
  (h1 : circles.largeRadius ^ 2 = P.1 ^ 2 + P.2 ^ 2)
  (h2 : circles.smallRadius = circles.largeRadius - QR)
  (h3 : (S k).2 = circles.smallRadius) :
  k = 2 * Real.sqrt 34 - 4 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_k_value_l3792_379250


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l3792_379208

theorem factorization_of_polynomial (x : ℝ) :
  29 * 40 * x^4 + 64 = 29 * 40 * ((x^2 - 4*x + 8) * (x^2 + 4*x + 8)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l3792_379208


namespace NUMINAMATH_CALUDE_aarons_age_l3792_379277

theorem aarons_age (julie aaron : ℕ) (h1 : julie = 4 * aaron) 
  (h2 : julie + 10 = 2 * (aaron + 10)) : aaron = 5 := by
  sorry

end NUMINAMATH_CALUDE_aarons_age_l3792_379277


namespace NUMINAMATH_CALUDE_factory_bulb_supply_percentage_l3792_379261

theorem factory_bulb_supply_percentage 
  (prob_x : ℝ) 
  (prob_y : ℝ) 
  (prob_total : ℝ) 
  (h1 : prob_x = 0.59) 
  (h2 : prob_y = 0.65) 
  (h3 : prob_total = 0.62) : 
  ∃ (p : ℝ), p * prob_x + (1 - p) * prob_y = prob_total ∧ p = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_factory_bulb_supply_percentage_l3792_379261


namespace NUMINAMATH_CALUDE_triangle_inequality_l3792_379224

/-- Given a triangle with side lengths a, b, c and area T, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 T, 
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c T : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : T > 0)
  (h_T : T = Real.sqrt ((a + b + c) * (a + b - c) * (b + c - a) * (c + a - b)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * T ∧ 
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * T ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3792_379224


namespace NUMINAMATH_CALUDE_jane_initial_crayons_l3792_379280

/-- The number of crayons Jane started with -/
def initial_crayons : ℕ := sorry

/-- The number of crayons eaten by the hippopotamus -/
def eaten_crayons : ℕ := 7

/-- The number of crayons Jane ended with -/
def final_crayons : ℕ := 80

/-- Theorem stating that Jane started with 87 crayons -/
theorem jane_initial_crayons : initial_crayons = 87 := by
  sorry

end NUMINAMATH_CALUDE_jane_initial_crayons_l3792_379280


namespace NUMINAMATH_CALUDE_abs_neg_six_equals_six_l3792_379235

theorem abs_neg_six_equals_six : abs (-6 : ℤ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_six_equals_six_l3792_379235


namespace NUMINAMATH_CALUDE_prop_p_and_q_implies_range_of_a_l3792_379227

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the range of a
def range_of_a (a : ℝ) : Prop := a ≤ -2 ∨ a = 1

-- Theorem statement
theorem prop_p_and_q_implies_range_of_a :
  ∀ a : ℝ, p a ∧ q a → range_of_a a :=
by sorry

end NUMINAMATH_CALUDE_prop_p_and_q_implies_range_of_a_l3792_379227


namespace NUMINAMATH_CALUDE_existence_of_n_l3792_379215

theorem existence_of_n (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h_cd : c * d = 1) :
  ∃ n : ℤ, (a * b : ℝ) ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_n_l3792_379215


namespace NUMINAMATH_CALUDE_coefficient_of_minus_two_ab_l3792_379205

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℤ) (x : String) : ℤ := m

/-- Given monomial -2ab, prove its coefficient is -2 -/
theorem coefficient_of_minus_two_ab :
  coefficient (-2) "ab" = -2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_minus_two_ab_l3792_379205


namespace NUMINAMATH_CALUDE_smallest_with_2023_divisors_l3792_379231

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n can be written as m * 6^k where 6 is not a divisor of m -/
def has_form (n m k : ℕ) : Prop :=
  n = m * 6^k ∧ ¬(6 ∣ m)

theorem smallest_with_2023_divisors :
  ∃ (m k : ℕ),
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    has_form (m * 6^k) m k ∧
    num_divisors (m * 6^k) = 2023 ∧
    m + k = 59055 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_2023_divisors_l3792_379231


namespace NUMINAMATH_CALUDE_set_D_forms_triangle_l3792_379200

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem set_D_forms_triangle :
  can_form_triangle 10 10 5 := by
  sorry

end NUMINAMATH_CALUDE_set_D_forms_triangle_l3792_379200


namespace NUMINAMATH_CALUDE_multiply_by_nine_l3792_379216

theorem multiply_by_nine (A B : ℕ) (h1 : 1 ≤ A ∧ A ≤ 9) (h2 : B ≤ 9) :
  (10 * A + B) * 9 = ((10 * A + B) - (A + 1)) * 10 + (10 - B) := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_nine_l3792_379216


namespace NUMINAMATH_CALUDE_blocks_remaining_l3792_379257

theorem blocks_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 78 → used = 19 → remaining = initial - used → remaining = 59 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_l3792_379257


namespace NUMINAMATH_CALUDE_monotonic_f_range_l3792_379293

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 3 else (a + 2) * Real.exp (a * x)

/-- The theorem stating the range of a for which f is monotonic -/
theorem monotonic_f_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_monotonic_f_range_l3792_379293


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3792_379273

theorem solution_to_system_of_equations :
  let x : ℚ := -49/3
  let y : ℚ := -17/6
  (3 * x - 18 * y = 2) ∧ (4 * y - x = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3792_379273


namespace NUMINAMATH_CALUDE_orange_ribbons_l3792_379276

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow + purple + orange + silver = total →
  4 * yellow = total →
  3 * purple = total →
  8 * orange = total →
  silver = 45 →
  orange = 19 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l3792_379276


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_two_l3792_379295

/-- For a complex number z = (a^2 - 4) + (a + 2)i where a is real,
    z is purely imaginary if and only if a = 2 -/
theorem purely_imaginary_iff_a_eq_two (a : ℝ) :
  let z : ℂ := (a^2 - 4) + (a + 2)*I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_eq_two_l3792_379295


namespace NUMINAMATH_CALUDE_marble_exchange_ratio_l3792_379291

theorem marble_exchange_ratio : 
  ∀ (ben_initial john_initial ben_final john_final marbles_given : ℕ),
    ben_initial = 18 →
    john_initial = 17 →
    ben_final = ben_initial - marbles_given →
    john_final = john_initial + marbles_given →
    john_final = ben_final + 17 →
    marbles_given * 2 = ben_initial :=
by
  sorry

end NUMINAMATH_CALUDE_marble_exchange_ratio_l3792_379291


namespace NUMINAMATH_CALUDE_vector_at_negative_two_l3792_379252

/-- A parameterized line in 2D space. -/
structure ParameterizedLine where
  vector : ℝ → (ℝ × ℝ)

/-- Given conditions for the parameterized line. -/
def line_conditions (L : ParameterizedLine) : Prop :=
  L.vector 1 = (2, 5) ∧ L.vector 4 = (5, -7)

/-- The theorem stating the vector at t = -2 given the conditions. -/
theorem vector_at_negative_two
  (L : ParameterizedLine)
  (h : line_conditions L) :
  L.vector (-2) = (-1, 17) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_two_l3792_379252


namespace NUMINAMATH_CALUDE_base7_351_to_base6_l3792_379296

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base7_351_to_base6 :
  base10ToBase6 (base7ToBase10 351) = 503 := by sorry

end NUMINAMATH_CALUDE_base7_351_to_base6_l3792_379296


namespace NUMINAMATH_CALUDE_sane_person_identified_l3792_379299

/-- Represents the types of individuals in Transylvania -/
inductive PersonType
| Sane
| Transylvanian

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| No

/-- A function that determines how a person of a given type would answer the question -/
def wouldAnswer (t : PersonType) : Answer :=
  match t with
  | PersonType.Sane => Answer.No
  | PersonType.Transylvanian => Answer.Yes

/-- Theorem stating that if an answer allows immediate identification, the person must be sane -/
theorem sane_person_identified
  (answer : Answer)
  (h_immediate : ∃ (t : PersonType), wouldAnswer t = answer) :
  answer = Answer.No ∧ wouldAnswer PersonType.Sane = answer :=
sorry

end NUMINAMATH_CALUDE_sane_person_identified_l3792_379299


namespace NUMINAMATH_CALUDE_remaining_distance_l3792_379254

theorem remaining_distance (total_distance driven_distance : ℕ) 
  (h1 : total_distance = 1200)
  (h2 : driven_distance = 384) :
  total_distance - driven_distance = 816 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_l3792_379254


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3792_379284

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 7 * y = 5) ∧ (x = 62 / 3) ∧ (y = 17) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3792_379284


namespace NUMINAMATH_CALUDE_interest_calculation_l3792_379275

/-- Given a principal amount, calculate the compound interest for 2 years at 3% rate -/
def compoundInterest (principal : ℝ) : ℝ :=
  principal * (1 + 0.03)^2 - principal

/-- Given a principal amount, calculate the simple interest for 2 years at 3% rate -/
def simpleInterest (principal : ℝ) : ℝ :=
  principal * 0.03 * 2

/-- Theorem stating that if the compound interest is $609, then the simple interest is $600 -/
theorem interest_calculation (P : ℝ) (h : compoundInterest P = 609) : 
  simpleInterest P = 600 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3792_379275


namespace NUMINAMATH_CALUDE_sin_C_equals_half_l3792_379246

theorem sin_C_equals_half (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- b = 2c * sin(B)
  (b = 2 * c * Real.sin B) →
  -- Then sin(C) = 1/2
  Real.sin C = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_C_equals_half_l3792_379246


namespace NUMINAMATH_CALUDE_tv_show_duration_l3792_379269

theorem tv_show_duration (seasons_15 seasons_20 seasons_12 : ℕ)
  (episodes_15 episodes_20 episodes_12 : ℕ)
  (avg_episodes_per_year : ℕ) :
  seasons_15 = 8 →
  seasons_20 = 4 →
  seasons_12 = 2 →
  episodes_15 = 15 →
  episodes_20 = 20 →
  episodes_12 = 12 →
  avg_episodes_per_year = 16 →
  (seasons_15 * episodes_15 + seasons_20 * episodes_20 + seasons_12 * episodes_12) /
    avg_episodes_per_year = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_show_duration_l3792_379269


namespace NUMINAMATH_CALUDE_chocolate_problem_l3792_379214

/-- The number of chocolates in the cost price -/
def cost_chocolates : ℕ := 24

/-- The gain percentage -/
def gain_percent : ℚ := 1/2

/-- The number of chocolates in the selling price -/
def selling_chocolates : ℕ := 16

theorem chocolate_problem (C S : ℚ) (n : ℕ) 
  (h1 : C > 0) 
  (h2 : S > 0) 
  (h3 : n > 0) 
  (h4 : cost_chocolates * C = n * S) 
  (h5 : gain_percent = (S - C) / C) : 
  n = selling_chocolates := by
  sorry

#check chocolate_problem

end NUMINAMATH_CALUDE_chocolate_problem_l3792_379214


namespace NUMINAMATH_CALUDE_sqrt_fourth_root_approx_l3792_379219

theorem sqrt_fourth_root_approx : 
  ∃ (x : ℝ), x^2 = (0.000625)^(1/4) ∧ |x - 0.4| < 0.05 := by sorry

end NUMINAMATH_CALUDE_sqrt_fourth_root_approx_l3792_379219


namespace NUMINAMATH_CALUDE_construction_possible_l3792_379239

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Check if a line is tangent to a circle -/
def tangent_to_circle (l : Line) (c : Circle) : Prop :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

/-- Check if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

/-- The main theorem -/
theorem construction_possible 
  (E F G H : ℝ × ℝ) : 
  ∃ (e f g : Line) (k : Circle),
    perpendicular e f ∧
    tangent_to_circle e k ∧
    tangent_to_circle f k ∧
    point_on_line E g ∧
    point_on_line F g ∧
    point_on_circle G k ∧
    point_on_circle H k ∧
    point_on_line G g ∧
    point_on_line H g :=
  sorry

end NUMINAMATH_CALUDE_construction_possible_l3792_379239


namespace NUMINAMATH_CALUDE_lettuce_types_count_l3792_379274

/-- The number of lunch combo options given the number of lettuce types -/
def lunch_combos (lettuce_types : ℕ) : ℕ :=
  lettuce_types * 3 * 4 * 2

/-- Theorem stating that there are 2 types of lettuce -/
theorem lettuce_types_count : ∃ (n : ℕ), n = 2 ∧ lunch_combos n = 48 := by
  sorry

end NUMINAMATH_CALUDE_lettuce_types_count_l3792_379274


namespace NUMINAMATH_CALUDE_hearty_beads_count_l3792_379242

/-- The number of packages of blue beads -/
def blue_packages : ℕ := 4

/-- The number of packages of red beads -/
def red_packages : ℕ := 5

/-- The number of packages of green beads -/
def green_packages : ℕ := 2

/-- The number of beads in each blue package -/
def blue_beads_per_package : ℕ := 30

/-- The number of beads in each red package -/
def red_beads_per_package : ℕ := 45

/-- The number of additional beads in each green package compared to a blue package -/
def green_extra_beads : ℕ := 15

/-- The total number of beads Hearty has -/
def total_beads : ℕ := blue_packages * blue_beads_per_package + 
                        red_packages * red_beads_per_package + 
                        green_packages * (blue_beads_per_package + green_extra_beads)

theorem hearty_beads_count : total_beads = 435 := by
  sorry

end NUMINAMATH_CALUDE_hearty_beads_count_l3792_379242


namespace NUMINAMATH_CALUDE_area_triangle_ABC_l3792_379271

/-- Linear function f(x) = ax + b -/
def linear_function (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- The y-intercept of a linear function f(x) = ax + b -/
def y_intercept (f : ℝ → ℝ) : ℝ := f 0

theorem area_triangle_ABC : 
  ∀ (m n : ℝ),
  let f := linear_function (3/2) m
  let g := linear_function (-1/2) n
  (f (-4) = 0) →
  (g (-4) = 0) →
  let B := (0, y_intercept f)
  let C := (0, y_intercept g)
  let A := (-4, 0)
  (1/2 * |A.1| * |B.2 - C.2| = 16) :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_ABC_l3792_379271


namespace NUMINAMATH_CALUDE_tripled_division_l3792_379251

theorem tripled_division (a b q r : ℤ) 
  (h1 : a = b * q + r) 
  (h2 : 0 ≤ r ∧ r < b) : 
  ∃ (r' : ℤ), 3 * a = (3 * b) * q + r' ∧ r' = 3 * r := by
sorry

end NUMINAMATH_CALUDE_tripled_division_l3792_379251


namespace NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l3792_379228

theorem squared_difference_of_quadratic_roots : ∀ Φ φ : ℝ, 
  Φ ≠ φ →
  Φ^2 = 2*Φ + 1 →
  φ^2 = 2*φ + 1 →
  (Φ - φ)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_squared_difference_of_quadratic_roots_l3792_379228


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l3792_379237

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 3^x = x + 50 := by sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l3792_379237


namespace NUMINAMATH_CALUDE_quadrilateral_Q₁PNF_is_cyclic_l3792_379283

/-- Two circles with points on them and their intersections -/
structure TwoCirclesConfig where
  /-- The first circle -/
  circle1 : Set (ℝ × ℝ)
  /-- The second circle -/
  circle2 : Set (ℝ × ℝ)
  /-- Point Q₁, an intersection of the two circles -/
  Q₁ : ℝ × ℝ
  /-- Point Q₂, another intersection of the two circles -/
  Q₂ : ℝ × ℝ
  /-- Point A on the first circle -/
  A : ℝ × ℝ
  /-- Point B on the first circle -/
  B : ℝ × ℝ
  /-- Point C, where AQ₂ intersects circle2 again -/
  C : ℝ × ℝ
  /-- Point F on arc Q₁Q₂ of circle1, inside circle2 -/
  F : ℝ × ℝ
  /-- Point P, intersection of AF and BQ₁ -/
  P : ℝ × ℝ
  /-- Point N, where PC intersects circle2 again -/
  N : ℝ × ℝ

  /-- Q₁ and Q₂ are on both circles -/
  h1 : Q₁ ∈ circle1 ∧ Q₁ ∈ circle2
  h2 : Q₂ ∈ circle1 ∧ Q₂ ∈ circle2
  /-- A and B are on circle1 -/
  h3 : A ∈ circle1
  h4 : B ∈ circle1
  /-- C is on circle2 -/
  h5 : C ∈ circle2
  /-- F is on arc Q₁Q₂ of circle1, inside circle2 -/
  h6 : F ∈ circle1
  /-- N is on circle2 -/
  h7 : N ∈ circle2

/-- The main theorem: Quadrilateral Q₁PNF is cyclic -/
theorem quadrilateral_Q₁PNF_is_cyclic (config : TwoCirclesConfig) :
  ∃ (circle : Set (ℝ × ℝ)), config.Q₁ ∈ circle ∧ config.P ∈ circle ∧ config.N ∈ circle ∧ config.F ∈ circle :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_Q₁PNF_is_cyclic_l3792_379283


namespace NUMINAMATH_CALUDE_ava_watched_hours_l3792_379259

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of minutes Ava watched television
def ava_watched_minutes : ℕ := 240

-- Theorem to prove
theorem ava_watched_hours : ava_watched_minutes / minutes_per_hour = 4 := by
  sorry

end NUMINAMATH_CALUDE_ava_watched_hours_l3792_379259


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3792_379223

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  b * Real.sin (2 * A) = Real.sqrt 3 * a * Real.sin B →
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 →
  b / c = 3 * Real.sqrt 3 / 4 →
  A = π / 6 ∧ a = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3792_379223
