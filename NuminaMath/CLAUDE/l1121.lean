import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_remainder_l1121_112178

def f (x : ℝ) : ℝ := x^5 + 2*x^3 + x^2 + 4

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), f = λ x => (x - 2) * q x + 56 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1121_112178


namespace NUMINAMATH_CALUDE_smallest_number_of_ducks_l1121_112135

/-- Represents the number of birds in a flock for each type --/
structure FlockSize where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- Represents the number of flocks for each type of bird --/
structure FlockCount where
  duck : ℕ
  crane : ℕ
  heron : ℕ

/-- The main theorem stating the smallest number of ducks observed --/
theorem smallest_number_of_ducks 
  (flock_size : FlockSize)
  (flock_count : FlockCount)
  (h1 : flock_size.duck = 13)
  (h2 : flock_size.crane = 17)
  (h3 : flock_size.heron = 11)
  (h4 : flock_size.duck * flock_count.duck = flock_size.crane * flock_count.crane)
  (h5 : 6 * (flock_size.duck * flock_count.duck) = 5 * (flock_size.heron * flock_count.heron))
  (h6 : 3 * (flock_size.crane * flock_count.crane) = 8 * (flock_size.heron * flock_count.heron))
  (h7 : ∀ c : FlockCount, 
    (c.duck < flock_count.duck ∨ c.crane < flock_count.crane ∨ c.heron < flock_count.heron) →
    (flock_size.duck * c.duck ≠ flock_size.crane * c.crane ∨
     6 * (flock_size.duck * c.duck) ≠ 5 * (flock_size.heron * c.heron) ∨
     3 * (flock_size.crane * c.crane) ≠ 8 * (flock_size.heron * c.heron))) :
  flock_size.duck * flock_count.duck = 520 := by
  sorry


end NUMINAMATH_CALUDE_smallest_number_of_ducks_l1121_112135


namespace NUMINAMATH_CALUDE_player_field_time_l1121_112190

/-- Given a sports tournament with the following conditions:
  * The team has 10 players
  * 8 players are always on the field
  * The match lasts 45 minutes
  * All players must play the same amount of time
This theorem proves that each player will be on the field for 36 minutes. -/
theorem player_field_time 
  (total_players : ℕ) 
  (field_players : ℕ) 
  (match_duration : ℕ) 
  (h1 : total_players = 10)
  (h2 : field_players = 8)
  (h3 : match_duration = 45) :
  (field_players * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_field_time_l1121_112190


namespace NUMINAMATH_CALUDE_ducks_in_lake_l1121_112174

theorem ducks_in_lake (initial_ducks additional_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : additional_ducks = 20) :
  initial_ducks + additional_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l1121_112174


namespace NUMINAMATH_CALUDE_cuboid_height_l1121_112123

/-- The height of a rectangular cuboid given its surface area, length, and width -/
theorem cuboid_height (surface_area length width height : ℝ) : 
  surface_area = 2 * length * width + 2 * length * height + 2 * width * height →
  surface_area = 442 →
  length = 7 →
  width = 8 →
  height = 11 := by
  sorry


end NUMINAMATH_CALUDE_cuboid_height_l1121_112123


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1121_112199

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Theorem statement
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 3 + a 7 + a 8 = 8 →
  a 4 + a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1121_112199


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1121_112154

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + 3*m + 1 = 0 → n^2 + 3*n + 1 = 0 → m * n = 1 → (3*m + 1) / (m^3 * n) = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1121_112154


namespace NUMINAMATH_CALUDE_uncle_ben_farm_l1121_112114

def farm_problem (total_chickens : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) (total_eggs : ℕ) : Prop :=
  ∃ (roosters hens : ℕ),
    roosters + hens = total_chickens ∧
    3 * (hens - non_laying_hens) = total_eggs ∧
    roosters = 39

theorem uncle_ben_farm :
  farm_problem 440 15 3 1158 :=
sorry

end NUMINAMATH_CALUDE_uncle_ben_farm_l1121_112114


namespace NUMINAMATH_CALUDE_greatest_number_l1121_112193

theorem greatest_number : ∀ (a b c : ℝ), 
  a = 43.23 ∧ b = 2/5 ∧ c = 21.23 →
  a > b ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_l1121_112193


namespace NUMINAMATH_CALUDE_unique_triangle_solution_l1121_112168

/-- Represents the assignment of numbers to letters in the triangle puzzle -/
structure TriangleAssignment where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat

/-- The set of numbers used in the puzzle -/
def puzzleNumbers : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- Checks if the given assignment satisfies all conditions of the puzzle -/
def isValidAssignment (assignment : TriangleAssignment) : Prop :=
  assignment.A ∈ puzzleNumbers ∧
  assignment.B ∈ puzzleNumbers ∧
  assignment.C ∈ puzzleNumbers ∧
  assignment.D ∈ puzzleNumbers ∧
  assignment.E ∈ puzzleNumbers ∧
  assignment.F ∈ puzzleNumbers ∧
  assignment.D + assignment.E + assignment.B = 14 ∧
  assignment.A + assignment.C = 3 ∧
  assignment.A ≠ assignment.B ∧ assignment.A ≠ assignment.C ∧ assignment.A ≠ assignment.D ∧
  assignment.A ≠ assignment.E ∧ assignment.A ≠ assignment.F ∧
  assignment.B ≠ assignment.C ∧ assignment.B ≠ assignment.D ∧ assignment.B ≠ assignment.E ∧
  assignment.B ≠ assignment.F ∧
  assignment.C ≠ assignment.D ∧ assignment.C ≠ assignment.E ∧ assignment.C ≠ assignment.F ∧
  assignment.D ≠ assignment.E ∧ assignment.D ≠ assignment.F ∧
  assignment.E ≠ assignment.F

/-- The unique solution to the triangle puzzle -/
def triangleSolution : TriangleAssignment :=
  { A := 1, B := 3, C := 2, D := 5, E := 6, F := 4 }

/-- Theorem stating that the triangleSolution is the only valid assignment -/
theorem unique_triangle_solution :
  ∀ assignment : TriangleAssignment,
    isValidAssignment assignment → assignment = triangleSolution := by
  sorry

end NUMINAMATH_CALUDE_unique_triangle_solution_l1121_112168


namespace NUMINAMATH_CALUDE_angle_ADF_measure_l1121_112137

-- Define the circle O and points A, B, C, D, E, F
variable (O : ℝ × ℝ) (A B C D E F : ℝ × ℝ)

-- Define the circle's radius
variable (r : ℝ)

-- Define the angle measure function
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- State the given conditions
axiom C_on_BE_extension : sorry
axiom CA_tangent : sorry
axiom DC_bisects_ACB : angle_measure C A D = angle_measure C B D
axiom DC_intersects_AE : sorry
axiom DC_intersects_AB : sorry

-- Define the theorem
theorem angle_ADF_measure :
  angle_measure A D F = 67.5 := sorry

end NUMINAMATH_CALUDE_angle_ADF_measure_l1121_112137


namespace NUMINAMATH_CALUDE_basketball_scores_theorem_l1121_112112

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  total_games : Nat
  sixth_game_score : Nat
  seventh_game_score : Nat
  eighth_game_score : Nat
  ninth_game_score : Nat
  first_five_avg : ℝ
  first_nine_avg : ℝ

/-- Theorem about basketball scores -/
theorem basketball_scores_theorem (scores : BasketballScores) 
  (h1 : scores.total_games = 10)
  (h2 : scores.sixth_game_score = 22)
  (h3 : scores.seventh_game_score = 15)
  (h4 : scores.eighth_game_score = 12)
  (h5 : scores.ninth_game_score = 19) :
  (scores.first_nine_avg = (5 * scores.first_five_avg + 68) / 9) ∧
  (∃ (min_y : ℝ), min_y = 12 ∧ ∀ y, y = scores.first_nine_avg → y ≥ min_y) ∧
  (scores.first_nine_avg > scores.first_five_avg → 
    ∃ (max_score : ℕ), max_score = 84 ∧ 
    ∀ s, s = (5 : ℝ) * scores.first_five_avg → s ≤ max_score) := by
  sorry

end NUMINAMATH_CALUDE_basketball_scores_theorem_l1121_112112


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l1121_112175

/-- Given a triangle ABC with sides a, b, and c satisfying b^2 + c^2 ≥ 5a^2, 
    BC is the smallest side of the triangle. -/
theorem smallest_side_of_triangle (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_condition : b^2 + c^2 ≥ 5*a^2) : 
    c ≤ a ∧ c ≤ b := by
  sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l1121_112175


namespace NUMINAMATH_CALUDE_dixon_passing_students_l1121_112179

theorem dixon_passing_students (collins_total : ℕ) (collins_passed : ℕ) (dixon_total : ℕ) 
  (h1 : collins_total = 30) 
  (h2 : collins_passed = 18) 
  (h3 : dixon_total = 45) :
  (dixon_total * collins_passed) / collins_total = 27 := by
  sorry

end NUMINAMATH_CALUDE_dixon_passing_students_l1121_112179


namespace NUMINAMATH_CALUDE_expression_value_l1121_112145

theorem expression_value (m n : ℤ) (h : m - n = 2) : 2*m^2 - 4*m*n + 2*n^2 - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1121_112145


namespace NUMINAMATH_CALUDE_tanning_time_proof_l1121_112136

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
def remaining_tanning_time (monthly_limit : ℕ) (week1_time : ℕ) (week2_time : ℕ) : ℕ :=
  monthly_limit - (week1_time + week2_time)

/-- Proves that given the specified tanning times, the remaining time is 45 minutes. -/
theorem tanning_time_proof : remaining_tanning_time 200 75 80 = 45 := by
  sorry

end NUMINAMATH_CALUDE_tanning_time_proof_l1121_112136


namespace NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l1121_112121

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = 12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_trig_sum_l1121_112121


namespace NUMINAMATH_CALUDE_max_value_on_unit_circle_l1121_112117

theorem max_value_on_unit_circle (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + b^2 = 1 → a^2 + 4*b + 3 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_unit_circle_l1121_112117


namespace NUMINAMATH_CALUDE_eliminate_x_l1121_112105

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The system of equations -/
def system : (LinearEquation × LinearEquation) :=
  ({ a := 6, b := 2, c := 4 },
   { a := 3, b := -3, c := -6 })

/-- Operation that combines two equations -/
def combineEquations (eq1 eq2 : LinearEquation) (k : ℝ) : LinearEquation :=
  { a := eq1.a - k * eq2.a,
    b := eq1.b - k * eq2.b,
    c := eq1.c - k * eq2.c }

/-- Theorem stating that the specified operation eliminates x -/
theorem eliminate_x :
  let (eq1, eq2) := system
  let result := combineEquations eq1 eq2 2
  result.a = 0 := by sorry

end NUMINAMATH_CALUDE_eliminate_x_l1121_112105


namespace NUMINAMATH_CALUDE_triangle_properties_l1121_112106

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 = t.c^2 + t.a * t.b)
  (h2 : Real.sqrt 3 * t.c = 14 * Real.sin t.C)
  (h3 : t.a + t.b = 13) :
  t.C = π/3 ∧ t.c = 7 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1121_112106


namespace NUMINAMATH_CALUDE_john_star_wars_spending_l1121_112148

/-- Calculates the total money spent on Star Wars toys --/
def total_spent (group_a_cost group_b_cost : ℝ) 
                (group_a_discount group_b_discount : ℝ) 
                (group_a_tax group_b_tax lightsaber_tax : ℝ) : ℝ :=
  let group_a_discounted := group_a_cost * (1 - group_a_discount)
  let group_b_discounted := group_b_cost * (1 - group_b_discount)
  let group_a_total := group_a_discounted * (1 + group_a_tax)
  let group_b_total := group_b_discounted * (1 + group_b_tax)
  let other_toys_total := group_a_total + group_b_total
  let lightsaber_cost := 2 * other_toys_total
  let lightsaber_total := lightsaber_cost * (1 + lightsaber_tax)
  other_toys_total + lightsaber_total

/-- The total amount John spent on Star Wars toys is $4008.312 --/
theorem john_star_wars_spending :
  total_spent 900 600 0.15 0.25 0.06 0.09 0.04 = 4008.312 := by
  sorry

end NUMINAMATH_CALUDE_john_star_wars_spending_l1121_112148


namespace NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l1121_112115

theorem square_equality_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  ¬(∀ x y : ℝ, x^2 = y^2 → x = y) :=
by sorry

end NUMINAMATH_CALUDE_square_equality_necessary_not_sufficient_l1121_112115


namespace NUMINAMATH_CALUDE_next_simultaneous_activation_l1121_112195

/-- Represents the time interval in minutes for each location's signal -/
structure SignalIntervals :=
  (fire : ℕ)
  (police : ℕ)
  (hospital : ℕ)

/-- Calculates the time in minutes until the next simultaneous activation -/
def timeUntilNextSimultaneous (intervals : SignalIntervals) : ℕ :=
  Nat.lcm (Nat.lcm intervals.fire intervals.police) intervals.hospital

/-- Theorem stating that for the given intervals, the next simultaneous activation occurs after 180 minutes -/
theorem next_simultaneous_activation (intervals : SignalIntervals)
  (h1 : intervals.fire = 12)
  (h2 : intervals.police = 18)
  (h3 : intervals.hospital = 30) :
  timeUntilNextSimultaneous intervals = 180 := by
  sorry

#eval timeUntilNextSimultaneous ⟨12, 18, 30⟩

end NUMINAMATH_CALUDE_next_simultaneous_activation_l1121_112195


namespace NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1121_112194

theorem largest_common_divisor_408_340 : ∃ (n : ℕ), n = 68 ∧ 
  n ∣ 408 ∧ n ∣ 340 ∧ ∀ (m : ℕ), m ∣ 408 ∧ m ∣ 340 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_408_340_l1121_112194


namespace NUMINAMATH_CALUDE_coin_collection_value_l1121_112155

theorem coin_collection_value (total_coins : ℕ) (two_dollar_coins : ℕ) : 
  total_coins = 275 →
  two_dollar_coins = 148 →
  (total_coins - two_dollar_coins) * 1 + two_dollar_coins * 2 = 423 := by
sorry

end NUMINAMATH_CALUDE_coin_collection_value_l1121_112155


namespace NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l1121_112140

theorem experienced_sailors_monthly_earnings :
  let total_sailors : ℕ := 17
  let inexperienced_sailors : ℕ := 5
  let experienced_sailors : ℕ := total_sailors - inexperienced_sailors
  let inexperienced_hourly_wage : ℚ := 10
  let wage_increase_ratio : ℚ := 1 / 5
  let experienced_hourly_wage : ℚ := inexperienced_hourly_wage * (1 + wage_increase_ratio)
  let weekly_hours : ℕ := 60
  let weeks_per_month : ℕ := 4
  
  experienced_sailors * experienced_hourly_wage * weekly_hours * weeks_per_month = 34560 :=
by sorry

end NUMINAMATH_CALUDE_experienced_sailors_monthly_earnings_l1121_112140


namespace NUMINAMATH_CALUDE_largest_multiple_of_eight_less_than_neg_63_l1121_112104

theorem largest_multiple_of_eight_less_than_neg_63 :
  ∀ n : ℤ, n * 8 < -63 → n * 8 ≤ -64 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_eight_less_than_neg_63_l1121_112104


namespace NUMINAMATH_CALUDE_hash_difference_l1121_112167

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x

-- Theorem statement
theorem hash_difference : (hash 7 4) - (hash 4 7) = -9 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1121_112167


namespace NUMINAMATH_CALUDE_car_speed_problem_l1121_112186

/-- The speed of Car A in miles per hour -/
def speed_A : ℝ := 58

/-- The speed of Car B in miles per hour -/
def speed_B : ℝ := 50

/-- The initial distance between Car A and Car B in miles -/
def initial_distance : ℝ := 16

/-- The final distance between Car A and Car B in miles -/
def final_distance : ℝ := 8

/-- The time taken for Car A to overtake Car B in hours -/
def time : ℝ := 3

theorem car_speed_problem :
  speed_A * time = speed_B * time + initial_distance + final_distance := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1121_112186


namespace NUMINAMATH_CALUDE_average_percent_change_population_l1121_112196

-- Define the initial and final population
def initial_population : ℕ := 175000
def final_population : ℕ := 297500

-- Define the time period in years
def years : ℕ := 10

-- Define the theorem
theorem average_percent_change_population (initial_pop : ℕ) (final_pop : ℕ) (time : ℕ) :
  initial_pop = initial_population →
  final_pop = final_population →
  time = years →
  (((final_pop - initial_pop : ℝ) / initial_pop) * 100) / time = 7 :=
by sorry

end NUMINAMATH_CALUDE_average_percent_change_population_l1121_112196


namespace NUMINAMATH_CALUDE_second_equation_result_l1121_112197

theorem second_equation_result (x y : ℤ) 
  (eq1 : 3 * x + y = 40) 
  (eq2 : 3 * y^2 = 48) : 
  2 * x - y = 20 := by
sorry

end NUMINAMATH_CALUDE_second_equation_result_l1121_112197


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1121_112147

/-- The equation of the reflected light ray given an incident ray and a reflecting line. -/
theorem reflected_ray_equation :
  -- Incident ray: y = 2x + 1
  let incident_ray (x y : ℝ) := y = 2 * x + 1
  -- Reflecting line: y = x
  let reflecting_line (x y : ℝ) := y = x
  -- Reflected ray equation
  let reflected_ray (x y : ℝ) := x - 2 * y - 1 = 0
  -- The theorem
  ∀ x y : ℝ, reflected_ray x y ↔ 
    ∃ m : ℝ, incident_ray m (2 * m + 1) ∧ 
             reflecting_line ((x + y) / 2) ((x + y) / 2) ∧
             (x - m = y - (2 * m + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1121_112147


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1121_112125

theorem right_triangle_third_side : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a = 6 → b = 8 →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 10 ∨ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1121_112125


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l1121_112173

-- Define an arithmetic sequence of positive integers
def arithmetic_sequence (a₁ : ℕ+) (d : ℕ) : ℕ → ℕ+
  | 0 => a₁
  | n + 1 => ⟨(arithmetic_sequence a₁ d n).val + d, by sorry⟩

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic (a₁ : ℕ+) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁.val + (n - 1) * d) / 2

-- Theorem statement
theorem greatest_common_divisor_of_sum (a₁ : ℕ+) (d : ℕ) :
  6 = Nat.gcd (sum_arithmetic a₁ d 12) (Nat.gcd (sum_arithmetic (⟨a₁.val + 1, by sorry⟩) d 12)
    (sum_arithmetic (⟨a₁.val + 2, by sorry⟩) d 12)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_sum_l1121_112173


namespace NUMINAMATH_CALUDE_classroom_size_l1121_112129

theorem classroom_size :
  ∀ (initial_students : ℕ),
  (0.4 * initial_students : ℝ) = (0.32 * (initial_students + 5) : ℝ) →
  initial_students = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_classroom_size_l1121_112129


namespace NUMINAMATH_CALUDE_base9_4318_equals_3176_l1121_112134

/-- Converts a base-9 digit to its decimal (base-10) value. -/
def base9ToDecimal (digit : ℕ) : ℕ := digit

/-- Converts a base-9 number to its decimal (base-10) equivalent. -/
def convertBase9ToDecimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

theorem base9_4318_equals_3176 :
  convertBase9ToDecimal [4, 3, 1, 8] = 3176 := by
  sorry

#eval convertBase9ToDecimal [4, 3, 1, 8]

end NUMINAMATH_CALUDE_base9_4318_equals_3176_l1121_112134


namespace NUMINAMATH_CALUDE_book_pages_digits_l1121_112177

/-- Given a book with n pages, calculate the total number of digits used to number all pages. -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigits := min n 9
  let doubleDigits := max (min n 99 - 9) 0
  let tripleDigits := max (n - 99) 0
  singleDigits + 2 * doubleDigits + 3 * tripleDigits

/-- Theorem stating that a book with 360 pages requires exactly 972 digits to number all its pages. -/
theorem book_pages_digits : totalDigits 360 = 972 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_digits_l1121_112177


namespace NUMINAMATH_CALUDE_abc_inequality_l1121_112138

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1121_112138


namespace NUMINAMATH_CALUDE_power_sum_problem_l1121_112101

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 := by
sorry

end NUMINAMATH_CALUDE_power_sum_problem_l1121_112101


namespace NUMINAMATH_CALUDE_circle_radius_secant_l1121_112141

theorem circle_radius_secant (center P Q R : ℝ × ℝ) : 
  let distance := λ (a b : ℝ × ℝ) => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let radius := distance center Q
  distance center P = 15 ∧ 
  distance P Q = 10 ∧ 
  distance Q R = 8 ∧
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ R = (1 - t) • P + t • Q) →
  radius = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_secant_l1121_112141


namespace NUMINAMATH_CALUDE_total_routes_is_seven_l1121_112191

/-- The number of routes from A to C -/
def total_routes (highways_AB : ℕ) (paths_BC : ℕ) (direct_waterway : ℕ) : ℕ :=
  highways_AB * paths_BC + direct_waterway

/-- Theorem: Given the specified number of routes, the total number of routes from A to C is 7 -/
theorem total_routes_is_seven :
  total_routes 2 3 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_is_seven_l1121_112191


namespace NUMINAMATH_CALUDE_cherry_pie_count_l1121_112162

theorem cherry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h1 : total_pies = 36)
  (h2 : apple_ratio = 2)
  (h3 : blueberry_ratio = 5)
  (h4 : cherry_ratio = 4) :
  (cherry_ratio : ℚ) * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 144 / 11 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pie_count_l1121_112162


namespace NUMINAMATH_CALUDE_equation_equivalence_l1121_112160

theorem equation_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = 1 ∧ c = -7 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1121_112160


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1121_112171

/-- Represents a parabola of the form y = x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the conditions and conclusion about the parabola -/
theorem parabola_intersection_theorem (p : Parabola) 
  (A B C : Point) :
  (A.x = 0) →  -- A is on y-axis
  (B.x > 0 ∧ C.x > 0) →  -- B and C are on positive x-axis
  (B.y = 0 ∧ C.y = 0) →  -- B and C are on x-axis
  (A.y = p.c) →  -- A is the y-intercept
  (C.x - B.x = 2) →  -- BC = 2
  (1/2 * A.y * (C.x - B.x) = 3) →  -- Area of triangle ABC is 3
  (p.b = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1121_112171


namespace NUMINAMATH_CALUDE_smallest_m_inequality_l1121_112198

theorem smallest_m_inequality (a b c : ℝ) :
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N * (x^2 + y^2 + z^2)^2) → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_inequality_l1121_112198


namespace NUMINAMATH_CALUDE_fraction_equality_l1121_112100

theorem fraction_equality (a b c : ℝ) (h1 : b ≠ c) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  (a - b) / (b - c) = a / c ↔ 1 / b = (1 / a + 1 / c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1121_112100


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l1121_112188

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l1121_112188


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_unique_greatest_integer_l1121_112166

theorem greatest_integer_inequality (x : ℤ) : (7 : ℚ) / 9 > (x : ℚ) / 15 ↔ x ≤ 11 := by sorry

theorem unique_greatest_integer : ∃! x : ℤ, x = (Nat.floor ((7 : ℚ) / 9 * 15) : ℤ) ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_unique_greatest_integer_l1121_112166


namespace NUMINAMATH_CALUDE_colombian_coffee_amount_l1121_112189

/-- Proves the amount of Colombian coffee in a specific coffee mix -/
theorem colombian_coffee_amount
  (total_mix : ℝ)
  (colombian_price : ℝ)
  (brazilian_price : ℝ)
  (mix_price : ℝ)
  (h1 : total_mix = 100)
  (h2 : colombian_price = 8.75)
  (h3 : brazilian_price = 3.75)
  (h4 : mix_price = 6.35) :
  ∃ (colombian_amount : ℝ),
    colombian_amount = 52 ∧
    colombian_amount ≥ 0 ∧
    colombian_amount ≤ total_mix ∧
    ∃ (brazilian_amount : ℝ),
      brazilian_amount = total_mix - colombian_amount ∧
      colombian_price * colombian_amount + brazilian_price * brazilian_amount = mix_price * total_mix :=
by sorry

end NUMINAMATH_CALUDE_colombian_coffee_amount_l1121_112189


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1121_112192

theorem quadratic_one_solution (k : ℝ) : 
  (k > 0) → (∃! x, 4 * x^2 + k * x + 4 = 0) ↔ k = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1121_112192


namespace NUMINAMATH_CALUDE_negation_equivalence_l1121_112128

theorem negation_equivalence : 
  (¬(∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1121_112128


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_l1121_112164

/-- The cost of paint per kg given specific conditions -/
theorem paint_cost_per_kg (coverage : Real) (total_cost : Real) (side_length : Real) :
  coverage = 15 →
  total_cost = 200 →
  side_length = 5 →
  (total_cost / (6 * side_length^2 / coverage)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_l1121_112164


namespace NUMINAMATH_CALUDE_f_max_value_f_min_value_f_touches_x_axis_l1121_112183

/-- A cubic function that touches the x-axis at (1,0) -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x

/-- The maximum value of f(x) is 4/27 -/
theorem f_max_value : ∃ (x : ℝ), f x = 4/27 ∧ ∀ (y : ℝ), f y ≤ 4/27 :=
sorry

/-- The minimum value of f(x) is 0 -/
theorem f_min_value : ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y ≥ 0 :=
sorry

/-- The function f(x) touches the x-axis at (1,0) -/
theorem f_touches_x_axis : f 1 = 0 ∧ ∀ (x : ℝ), x ≠ 1 → f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_max_value_f_min_value_f_touches_x_axis_l1121_112183


namespace NUMINAMATH_CALUDE_cos_240_deg_l1121_112184

/-- Cosine of 240 degrees is equal to -1/2 -/
theorem cos_240_deg : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_deg_l1121_112184


namespace NUMINAMATH_CALUDE_remainder_problem_l1121_112120

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (4 * n + 2) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1121_112120


namespace NUMINAMATH_CALUDE_total_candy_collected_l1121_112142

/-- The number of candy pieces collected by Travis and his brother -/
def total_candy : ℕ := 68

/-- The number of people who collected candy -/
def num_people : ℕ := 2

/-- The number of candy pieces each person ate -/
def candy_eaten_per_person : ℕ := 4

/-- The number of candy pieces left after eating -/
def candy_left : ℕ := 60

/-- Theorem stating that the total candy collected equals 68 -/
theorem total_candy_collected :
  total_candy = candy_left + (num_people * candy_eaten_per_person) :=
by sorry

end NUMINAMATH_CALUDE_total_candy_collected_l1121_112142


namespace NUMINAMATH_CALUDE_chip_exits_at_A2_l1121_112116

-- Define the grid size
def gridSize : Nat := 4

-- Define the possible directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a cell position
structure Position where
  row : Nat
  col : Nat

-- Define the state of the game
structure GameState where
  chipPosition : Position
  arrows : Array (Array Direction)

-- Define the initial state
def initialState : GameState := sorry

-- Define a function to get the next position based on current position and direction
def nextPosition (pos : Position) (dir : Direction) : Position := sorry

-- Define a function to flip the direction
def flipDirection (dir : Direction) : Direction := sorry

-- Define a function to make a move
def makeMove (state : GameState) : GameState := sorry

-- Define a function to check if a position is out of bounds
def isOutOfBounds (pos : Position) : Bool := sorry

-- Define a function to simulate the game until the chip exits
def simulateUntilExit (state : GameState) : Position := sorry

-- The main theorem to prove
theorem chip_exits_at_A2 :
  let finalPos := simulateUntilExit initialState
  finalPos = Position.mk 0 1 := sorry

end NUMINAMATH_CALUDE_chip_exits_at_A2_l1121_112116


namespace NUMINAMATH_CALUDE_greatest_valid_integer_l1121_112146

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 145) ∧ is_valid 145 :=
sorry

end NUMINAMATH_CALUDE_greatest_valid_integer_l1121_112146


namespace NUMINAMATH_CALUDE_basketball_winning_percentage_l1121_112122

theorem basketball_winning_percentage (total_games season_games remaining_games first_wins : ℕ)
  (h1 : total_games = season_games + remaining_games)
  (h2 : season_games = 75)
  (h3 : remaining_games = 45)
  (h4 : first_wins = 60)
  (h5 : total_games = 120) :
  (∃ x : ℕ, x = 36 ∧ (first_wins + x : ℚ) / total_games = 4/5) :=
sorry

end NUMINAMATH_CALUDE_basketball_winning_percentage_l1121_112122


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l1121_112180

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l1121_112180


namespace NUMINAMATH_CALUDE_periodic_sin_and_empty_subset_l1121_112149

-- Define a periodic function
def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = f x

-- Define the sine function
noncomputable def sin : ℝ → ℝ := Real.sin

-- Define set A
variable (A : Set ℝ)

-- Theorem statement
theorem periodic_sin_and_empty_subset (A : Set ℝ) : 
  (isPeriodic sin) ∧ (∅ ⊆ A) := by sorry

end NUMINAMATH_CALUDE_periodic_sin_and_empty_subset_l1121_112149


namespace NUMINAMATH_CALUDE_first_day_is_sunday_l1121_112161

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of week after a given number of days -/
def afterDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (afterDays start m)

/-- Theorem: If the 21st day of a month is a Saturday, then the 1st day of that month is a Sunday -/
theorem first_day_is_sunday (d : DayOfWeek) :
  afterDays d 20 = DayOfWeek.Saturday → d = DayOfWeek.Sunday :=
by
  sorry


end NUMINAMATH_CALUDE_first_day_is_sunday_l1121_112161


namespace NUMINAMATH_CALUDE_sugar_pack_weight_l1121_112127

/-- Given the total sugar, number of packs, and leftover sugar, calculates the weight of each pack. -/
def packWeight (totalSugar : ℕ) (numPacks : ℕ) (leftoverSugar : ℕ) : ℕ :=
  (totalSugar - leftoverSugar) / numPacks

/-- Proves that given 3020 grams of total sugar, 12 packs, and 20 grams of leftover sugar, 
    the weight of each pack is 250 grams. -/
theorem sugar_pack_weight :
  packWeight 3020 12 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_sugar_pack_weight_l1121_112127


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1121_112133

theorem sqrt_equality_implies_t_value :
  ∀ t : ℝ, (Real.sqrt (3 * Real.sqrt (t - 3)) = (8 - t) ^ (1/4)) → t = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_t_value_l1121_112133


namespace NUMINAMATH_CALUDE_unique_determination_l1121_112102

-- Define the triangle types
inductive TriangleType
  | Isosceles
  | Equilateral
  | Right
  | Scalene

-- Define the given parts
inductive GivenParts
  | BaseAngleVertexAngle
  | VertexAngleBase
  | CircumscribedRadius
  | ArmInscribedRadius
  | TwoAnglesOneSide

-- Function to check if a combination uniquely determines a triangle
def uniquelyDetermines (t : TriangleType) (p : GivenParts) : Prop :=
  match t, p with
  | TriangleType.Isosceles, GivenParts.BaseAngleVertexAngle => False
  | TriangleType.Isosceles, GivenParts.VertexAngleBase => True
  | TriangleType.Equilateral, GivenParts.CircumscribedRadius => True
  | TriangleType.Right, GivenParts.ArmInscribedRadius => False
  | TriangleType.Scalene, GivenParts.TwoAnglesOneSide => True
  | _, _ => False

theorem unique_determination :
  ∀ (t : TriangleType) (p : GivenParts),
    (t = TriangleType.Isosceles ∧ p = GivenParts.BaseAngleVertexAngle) ↔ ¬(uniquelyDetermines t p) :=
sorry

end NUMINAMATH_CALUDE_unique_determination_l1121_112102


namespace NUMINAMATH_CALUDE_stating_chess_tournament_players_l1121_112144

/-- The number of players in the chess tournament. -/
def num_players : ℕ := 11

/-- The total number of games played in the tournament. -/
def total_games : ℕ := 132

/-- 
Theorem stating that the number of players in the chess tournament is 11,
given the conditions of the problem.
-/
theorem chess_tournament_players :
  (∀ n : ℕ, n > 0 → 2 * n * (n - 1) = total_games) → num_players = 11 :=
by sorry

end NUMINAMATH_CALUDE_stating_chess_tournament_players_l1121_112144


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l1121_112118

/-- The jumping distances of animals in a contest -/
structure JumpingContest where
  mouse_jump : ℕ
  frog_jump : ℕ
  grasshopper_jump : ℕ
  mouse_frog_diff : frog_jump = mouse_jump + 12
  grasshopper_frog_diff : grasshopper_jump = frog_jump + 19

/-- Theorem: In a jumping contest where the mouse jumped 8 inches, 
    the mouse jumped 12 inches less than the frog, 
    and the grasshopper jumped 19 inches farther than the frog, 
    the grasshopper jumped 39 inches. -/
theorem grasshopper_jump_distance (contest : JumpingContest) 
  (h_mouse_jump : contest.mouse_jump = 8) : 
  contest.grasshopper_jump = 39 := by
  sorry


end NUMINAMATH_CALUDE_grasshopper_jump_distance_l1121_112118


namespace NUMINAMATH_CALUDE_complex_absolute_value_squared_l1121_112107

theorem complex_absolute_value_squared (z : ℂ) (h : z + Complex.abs z = 3 + 7*I) : Complex.abs z ^ 2 = 841 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_squared_l1121_112107


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1121_112181

theorem smallest_number_divisible (n : ℕ) : n = 32127 ↔ 
  (∀ m : ℕ, m < n → ¬(((m + 3) % 510 = 0) ∧ ((m + 3) % 4590 = 0) ∧ ((m + 3) % 105 = 0))) ∧
  ((n + 3) % 510 = 0) ∧ ((n + 3) % 4590 = 0) ∧ ((n + 3) % 105 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1121_112181


namespace NUMINAMATH_CALUDE_geometric_series_sum_8_terms_l1121_112109

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_8_terms :
  geometric_series_sum (1/4) (1/4) 8 = 65535/196608 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_8_terms_l1121_112109


namespace NUMINAMATH_CALUDE_Q_roots_nature_l1121_112130

/-- The polynomial Q(x) = x^6 - 4x^5 + 3x^4 - 7x^3 - x^2 + x + 10 -/
def Q (x : ℝ) : ℝ := x^6 - 4*x^5 + 3*x^4 - 7*x^3 - x^2 + x + 10

/-- Theorem stating that Q(x) has at least one negative root and at least two positive roots -/
theorem Q_roots_nature :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ Q x = 0 ∧ Q y = 0) :=
sorry

end NUMINAMATH_CALUDE_Q_roots_nature_l1121_112130


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l1121_112185

def total_balls : ℕ := 7 + 5 + 4

def red_balls : ℕ := 7

theorem probability_two_red_balls :
  (Nat.choose red_balls 2 : ℚ) / (Nat.choose total_balls 2) = 7 / 40 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l1121_112185


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_2018_2013_l1121_112150

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio_2018_2013 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : GeometricSequence a q)
  (h_sum : a 1 + a 6 = 8)
  (h_prod : a 3 * a 4 = 12) :
  a 2018 / a 2013 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_2018_2013_l1121_112150


namespace NUMINAMATH_CALUDE_lcm_of_prime_and_nondivisor_l1121_112132

theorem lcm_of_prime_and_nondivisor (p n : ℕ) (hp : Nat.Prime p) (hn : ¬(n ∣ p)) :
  Nat.lcm p n = p * n :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_prime_and_nondivisor_l1121_112132


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1121_112187

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Nat
  edges_per_vertex : Nat

/-- Calculates the number of interior diagonals in a dodecahedron -/
def interior_diagonals (d : Dodecahedron) : Nat :=
  (d.vertices * (d.vertices - 1 - d.edges_per_vertex)) / 2

/-- Theorem stating that a dodecahedron has 160 interior diagonals -/
theorem dodecahedron_interior_diagonals :
  ∃ d : Dodecahedron, d.vertices = 20 ∧ d.edges_per_vertex = 3 ∧ interior_diagonals d = 160 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l1121_112187


namespace NUMINAMATH_CALUDE_intersection_M_N_l1121_112165

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1121_112165


namespace NUMINAMATH_CALUDE_units_digit_problem_l1121_112113

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l1121_112113


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1121_112153

theorem geometric_sequence_common_ratio 
  (a₁ : ℝ) (r : ℝ) 
  (h₁ : a₁ ≠ 0) 
  (h₂ : r > 0) 
  (h₃ : ∀ n m : ℕ, n ≠ m → a₁ * r^n ≠ a₁ * r^m) 
  (h₄ : ∃ d : ℝ, a₁ * r^3 - a₁ * r = d ∧ a₁ * r^4 - a₁ * r^3 = d) : 
  r = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1121_112153


namespace NUMINAMATH_CALUDE_dave_white_tshirt_packs_l1121_112151

/-- The number of T-shirts in a pack of white T-shirts -/
def white_pack_size : ℕ := 6

/-- The number of T-shirts in a pack of blue T-shirts -/
def blue_pack_size : ℕ := 4

/-- The number of packs of blue T-shirts Dave bought -/
def blue_packs : ℕ := 2

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := 26

/-- The number of packs of white T-shirts Dave bought -/
def white_packs : ℕ := 3

theorem dave_white_tshirt_packs :
  white_packs * white_pack_size + blue_packs * blue_pack_size = total_tshirts :=
by sorry

end NUMINAMATH_CALUDE_dave_white_tshirt_packs_l1121_112151


namespace NUMINAMATH_CALUDE_grapefruit_orchards_l1121_112108

/-- Represents the number of orchards for each type of citrus fruit -/
structure CitrusOrchards where
  total : ℕ
  lemons : ℕ
  oranges : ℕ
  limes : ℕ
  grapefruits : ℕ
  mandarins : ℕ

/-- Theorem stating the number of grapefruit orchards given the conditions -/
theorem grapefruit_orchards (c : CitrusOrchards) : c.grapefruits = 6 :=
  by
  have h1 : c.total = 40 := sorry
  have h2 : c.lemons = 15 := sorry
  have h3 : c.oranges = 2 * c.lemons / 3 := sorry
  have h4 : c.limes = c.grapefruits := sorry
  have h5 : c.mandarins = c.grapefruits / 2 := sorry
  have h6 : c.total = c.lemons + c.oranges + c.limes + c.grapefruits + c.mandarins := sorry
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_l1121_112108


namespace NUMINAMATH_CALUDE_black_shirts_per_pack_l1121_112163

/-- Given:
  * 3 packs of black shirts and 3 packs of yellow shirts were bought
  * Yellow shirts come in packs of 2
  * Total number of shirts is 21
Prove that the number of black shirts in each pack is 5 -/
theorem black_shirts_per_pack (black_packs yellow_packs : ℕ) 
  (yellow_per_pack total_shirts : ℕ) (black_per_pack : ℕ) :
  black_packs = 3 →
  yellow_packs = 3 →
  yellow_per_pack = 2 →
  total_shirts = 21 →
  black_packs * black_per_pack + yellow_packs * yellow_per_pack = total_shirts →
  black_per_pack = 5 := by
  sorry

#check black_shirts_per_pack

end NUMINAMATH_CALUDE_black_shirts_per_pack_l1121_112163


namespace NUMINAMATH_CALUDE_no_solution_l1121_112131

def equation1 (x₁ x₂ x₃ : ℝ) : Prop := 2 * x₁ + 5 * x₂ - 4 * x₃ = 8
def equation2 (x₁ x₂ x₃ : ℝ) : Prop := 3 * x₁ + 15 * x₂ - 9 * x₃ = 5
def equation3 (x₁ x₂ x₃ : ℝ) : Prop := 5 * x₁ + 5 * x₂ - 7 * x₃ = 1

theorem no_solution : ¬∃ x₁ x₂ x₃ : ℝ, equation1 x₁ x₂ x₃ ∧ equation2 x₁ x₂ x₃ ∧ equation3 x₁ x₂ x₃ := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1121_112131


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1121_112152

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x - 2) = 0) ∧ 
  (∃ x : ℝ, (x - 1) * (x - 2) = 0 ∧ x - 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1121_112152


namespace NUMINAMATH_CALUDE_farm_animals_problem_l1121_112126

theorem farm_animals_problem :
  ∃! (s c : ℕ), s > 0 ∧ c > 0 ∧ 28 * s + 27 * c = 1200 ∧ c > s :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_problem_l1121_112126


namespace NUMINAMATH_CALUDE_coefficient_a3_value_l1121_112157

/-- Given a polynomial expansion and sum of coefficients condition, prove a₃ = -5 -/
theorem coefficient_a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_a3_value_l1121_112157


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l1121_112169

theorem circle_diameter_ratio (C D : Real) (h1 : D = 20) 
  (h2 : C > 0 ∧ C < D) (h3 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 5) : 
  C = 10 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l1121_112169


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1121_112176

theorem fraction_equals_zero (x : ℝ) (h : x ≠ 0) :
  (x - 3) / (4 * x) = 0 ↔ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1121_112176


namespace NUMINAMATH_CALUDE_lines_intersect_l1121_112103

/-- Two lines in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define when two lines are intersecting -/
def are_intersecting (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- The problem statement -/
theorem lines_intersect : 
  let line1 : Line2D := ⟨3, -2, 5⟩
  let line2 : Line2D := ⟨1, 3, 10⟩
  are_intersecting line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_l1121_112103


namespace NUMINAMATH_CALUDE_divisibility_of_power_difference_l1121_112143

theorem divisibility_of_power_difference (a b c k q : ℕ) (n : ℤ) :
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  k ≥ 1 →
  n = a^(c^k) - b^(c^k) →
  (∃ (p : List ℕ), (∀ x ∈ p, Nat.Prime x) ∧ p.length ≥ q ∧ (∀ x ∈ p, c % x = 0)) →
  ∃ (r : List ℕ), (∀ x ∈ r, Nat.Prime x) ∧ r.length ≥ q * k ∧ (∀ x ∈ r, n % x = 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_power_difference_l1121_112143


namespace NUMINAMATH_CALUDE_enclosed_area_semicircles_l1121_112111

/-- Given a semicircle with radius R and its diameter divided into parts 2r and 2(R-r),
    the area enclosed between the three semicircles (the original and two smaller ones)
    is equal to π r(R-r) -/
theorem enclosed_area_semicircles (R r : ℝ) (h1 : 0 < R) (h2 : 0 < r) (h3 : r < R) :
  let original_area := π * R^2 / 2
  let small_area1 := π * r^2 / 2
  let small_area2 := π * (R-r)^2 / 2
  original_area - small_area1 - small_area2 = π * r * (R-r) :=
by sorry

end NUMINAMATH_CALUDE_enclosed_area_semicircles_l1121_112111


namespace NUMINAMATH_CALUDE_train_speed_problem_l1121_112172

/-- Prove that given two trains of equal length 62.5 meters, where the faster train
    travels at 46 km/hr and passes the slower train in 45 seconds, the speed of
    the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 62.5 →
  faster_speed = 46 →
  passing_time = 45 →
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * (1000 / 3600) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1121_112172


namespace NUMINAMATH_CALUDE_marsha_second_package_distance_l1121_112182

/-- Represents the distance Marsha drives for her second package delivery -/
def second_package_distance : ℝ := 28

/-- Represents Marsha's total payment for the day -/
def total_payment : ℝ := 104

/-- Represents Marsha's payment per mile -/
def payment_per_mile : ℝ := 2

/-- Represents the distance Marsha drives for her first package delivery -/
def first_package_distance : ℝ := 10

theorem marsha_second_package_distance :
  second_package_distance = 28 ∧
  total_payment = payment_per_mile * (first_package_distance + second_package_distance + second_package_distance / 2) :=
by sorry

end NUMINAMATH_CALUDE_marsha_second_package_distance_l1121_112182


namespace NUMINAMATH_CALUDE_people_who_got_off_train_l1121_112158

theorem people_who_got_off_train (initial_people : ℕ) (people_who_got_on : ℕ) (final_people : ℕ) :
  initial_people = 82 →
  people_who_got_on = 17 →
  final_people = 73 →
  ∃ (people_who_got_off : ℕ), 
    initial_people - people_who_got_off + people_who_got_on = final_people ∧
    people_who_got_off = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_train_l1121_112158


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1121_112170

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 15) :
  a^4 + b^4 + c^4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1121_112170


namespace NUMINAMATH_CALUDE_total_hamburgers_calculation_l1121_112119

/-- Calculates the total number of hamburgers bought given the total amount spent,
    costs of single and double burgers, and the number of double burgers bought. -/
theorem total_hamburgers_calculation 
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_bought : ℕ)
  (h1 : total_spent = 66.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_bought = 33) :
  ∃ (single_burgers_bought : ℕ),
    single_burgers_bought + double_burgers_bought = 50 ∧
    total_spent = single_burger_cost * single_burgers_bought + double_burger_cost * double_burgers_bought :=
by sorry


end NUMINAMATH_CALUDE_total_hamburgers_calculation_l1121_112119


namespace NUMINAMATH_CALUDE_line_outside_plane_iff_at_most_one_point_l1121_112124

-- Define the basic types
variable (L : Type*) -- Type for lines
variable (P : Type*) -- Type for planes

-- Define the relationships between lines and planes
variable (parallel : L → P → Prop)
variable (intersects : L → P → Prop)
variable (within : L → P → Prop)
variable (outside : L → P → Prop)

-- Define the number of common points
variable (common_points : L → P → ℕ)

-- Theorem statement
theorem line_outside_plane_iff_at_most_one_point 
  (l : L) (p : P) : 
  outside l p ↔ common_points l p ≤ 1 := by sorry

end NUMINAMATH_CALUDE_line_outside_plane_iff_at_most_one_point_l1121_112124


namespace NUMINAMATH_CALUDE_prob_red_white_red_is_7_66_l1121_112156

-- Define the number of red and white marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 7

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles

-- Define the probability of drawing red, white, and red marbles in order
def prob_red_white_red : ℚ := (red_marbles : ℚ) / total_marbles *
                              (white_marbles : ℚ) / (total_marbles - 1) *
                              (red_marbles - 1 : ℚ) / (total_marbles - 2)

-- Theorem statement
theorem prob_red_white_red_is_7_66 :
  prob_red_white_red = 7 / 66 := by sorry

end NUMINAMATH_CALUDE_prob_red_white_red_is_7_66_l1121_112156


namespace NUMINAMATH_CALUDE_fourth_task_end_time_l1121_112110

-- Define the start time of the first task
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes

-- Define the end time of the second task
def end_second_task : Nat := 10 * 60 + 20  -- 10:20 AM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem fourth_task_end_time :
  let total_time := end_second_task - start_time
  let task_duration := total_time / 2
  let end_time := end_second_task + task_duration * 2
  end_time = 12 * 60 + 40  -- 12:40 PM in minutes
  := by sorry

end NUMINAMATH_CALUDE_fourth_task_end_time_l1121_112110


namespace NUMINAMATH_CALUDE_min_sum_squares_l1121_112159

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 ≥ m) ∧
             m = 3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1121_112159


namespace NUMINAMATH_CALUDE_function_inequality_l1121_112139

-- Define the function f on the non-zero real numbers
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition that f''(x) - f(x)/x > 0 for all non-zero x
variable (h : ∀ x : ℝ, x ≠ 0 → (deriv^[2] f) x - f x / x > 0)

-- State the theorem
theorem function_inequality : 3 * f 4 > 4 * f 3 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l1121_112139
