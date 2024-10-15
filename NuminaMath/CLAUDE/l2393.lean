import Mathlib

namespace NUMINAMATH_CALUDE_more_valid_placements_diff_intersections_l2393_239383

/-- Represents the number of radial streets in city N -/
def radial_streets : ℕ := 7

/-- Represents the number of parallel streets in city N -/
def parallel_streets : ℕ := 7

/-- Total number of intersections in the city -/
def total_intersections : ℕ := radial_streets * parallel_streets

/-- Calculates the number of valid store placements when stores must not be at the same intersection -/
def valid_placements_diff_intersections : ℕ := total_intersections * (total_intersections - 1)

/-- Calculates the number of valid store placements when stores must not be on the same street -/
def valid_placements_diff_streets : ℕ := 
  valid_placements_diff_intersections - 2 * (radial_streets * (total_intersections - radial_streets))

/-- Theorem stating that the condition of different intersections allows more valid placements -/
theorem more_valid_placements_diff_intersections : 
  valid_placements_diff_intersections > valid_placements_diff_streets :=
sorry

end NUMINAMATH_CALUDE_more_valid_placements_diff_intersections_l2393_239383


namespace NUMINAMATH_CALUDE_min_covering_size_l2393_239319

def X : Finset Nat := {1, 2, 3, 4, 5}

def is_covering (F : Finset (Finset Nat)) : Prop :=
  ∀ B ∈ Finset.powerset X, B.card = 3 → ∃ A ∈ F, A ⊆ B

theorem min_covering_size :
  ∃ F : Finset (Finset Nat),
    (∀ A ∈ F, A ⊆ X ∧ A.card = 2) ∧
    is_covering F ∧
    F.card = 10 ∧
    (∀ G : Finset (Finset Nat),
      (∀ A ∈ G, A ⊆ X ∧ A.card = 2) →
      is_covering G →
      G.card ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_min_covering_size_l2393_239319


namespace NUMINAMATH_CALUDE_initial_rows_count_l2393_239324

theorem initial_rows_count (chairs_per_row : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) 
  (h1 : chairs_per_row = 12)
  (h2 : extra_chairs = 11)
  (h3 : total_chairs = 95) :
  ∃ (initial_rows : ℕ), initial_rows * chairs_per_row + extra_chairs = total_chairs ∧ initial_rows = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_rows_count_l2393_239324


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l2393_239391

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) = 1 / Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l2393_239391


namespace NUMINAMATH_CALUDE_ages_sum_l2393_239315

theorem ages_sum (a b c : ℕ) : 
  a = 16 + b + c → 
  a^2 = 1632 + (b + c)^2 → 
  a + b + c = 102 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l2393_239315


namespace NUMINAMATH_CALUDE_aron_dusting_days_l2393_239305

/-- Represents the cleaning schedule and durations for Aron -/
structure CleaningSchedule where
  vacuumingTimePerDay : ℕ
  vacuumingDaysPerWeek : ℕ
  dustingTimePerDay : ℕ
  totalCleaningTimePerWeek : ℕ

/-- Calculates the number of days Aron spends dusting per week -/
def dustingDaysPerWeek (schedule : CleaningSchedule) : ℕ :=
  let totalVacuumingTime := schedule.vacuumingTimePerDay * schedule.vacuumingDaysPerWeek
  let totalDustingTime := schedule.totalCleaningTimePerWeek - totalVacuumingTime
  totalDustingTime / schedule.dustingTimePerDay

/-- Theorem stating that Aron spends 2 days a week dusting -/
theorem aron_dusting_days (schedule : CleaningSchedule)
    (h1 : schedule.vacuumingTimePerDay = 30)
    (h2 : schedule.vacuumingDaysPerWeek = 3)
    (h3 : schedule.dustingTimePerDay = 20)
    (h4 : schedule.totalCleaningTimePerWeek = 130) :
    dustingDaysPerWeek schedule = 2 := by
  sorry

#eval dustingDaysPerWeek {
  vacuumingTimePerDay := 30,
  vacuumingDaysPerWeek := 3,
  dustingTimePerDay := 20,
  totalCleaningTimePerWeek := 130
}

end NUMINAMATH_CALUDE_aron_dusting_days_l2393_239305


namespace NUMINAMATH_CALUDE_log_equation_solution_l2393_239348

theorem log_equation_solution (k c p : ℝ) (h : k > 0) (hp : p > 0) :
  Real.log k^2 / Real.log 10 = c - 2 * Real.log p / Real.log 10 →
  k = 10^c / p := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2393_239348


namespace NUMINAMATH_CALUDE_field_trip_buses_l2393_239332

/-- The number of students in the school -/
def total_students : ℕ := 11210

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 118

/-- The number of school buses needed for the field trip -/
def buses_needed : ℕ := total_students / seats_per_bus

theorem field_trip_buses : buses_needed = 95 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_buses_l2393_239332


namespace NUMINAMATH_CALUDE_tenth_group_sample_l2393_239308

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_sample : ℕ

/-- The number drawn from a specific group in systematic sampling -/
def group_sample (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + s.first_sample

/-- Theorem stating the relationship between samples from different groups -/
theorem tenth_group_sample (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : group_sample s 5 = 22) :
  group_sample s 10 = 47 := by
  sorry

#check tenth_group_sample

end NUMINAMATH_CALUDE_tenth_group_sample_l2393_239308


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l2393_239314

theorem triangle_area_theorem (A B C : Real) (a b c : Real) :
  c = 2 →
  C = π / 3 →
  let m : Real × Real := (Real.sin C + Real.sin (B - A), 4)
  let n : Real × Real := (Real.sin (2 * A), 1)
  (∃ (k : Real), m.1 = k * n.1 ∧ m.2 = k * n.2) →
  let S := (1 / 2) * a * c * Real.sin B
  (S = 4 * Real.sqrt 13 / 13 ∨ S = 2 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l2393_239314


namespace NUMINAMATH_CALUDE_birgit_travel_time_l2393_239322

def hiking_time : ℝ := 3.5
def distance_traveled : ℝ := 21
def birgit_speed_difference : ℝ := 4
def birgit_travel_distance : ℝ := 8

theorem birgit_travel_time : 
  let total_minutes := hiking_time * 60
  let average_speed := total_minutes / distance_traveled
  let birgit_speed := average_speed - birgit_speed_difference
  birgit_speed * birgit_travel_distance = 48 := by sorry

end NUMINAMATH_CALUDE_birgit_travel_time_l2393_239322


namespace NUMINAMATH_CALUDE_quadratic_roots_min_value_l2393_239341

theorem quadratic_roots_min_value (m : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 - 2*m*x₁ + m + 6 = 0 →
  x₂^2 - 2*m*x₂ + m + 6 = 0 →
  x₁ ≠ x₂ →
  ∀ y : ℝ, y = (x₁ - 1)^2 + (x₂ - 1)^2 → y ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_min_value_l2393_239341


namespace NUMINAMATH_CALUDE_win_sector_area_l2393_239397

theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_area : ℝ) : 
  radius = 8 → win_probability = 1/4 → win_area = 16 * Real.pi → 
  win_area = win_probability * Real.pi * radius^2 := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l2393_239397


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2393_239338

/-- The circle C is defined by the equation x^2 + y^2 + ax - 2y + b = 0 -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- The line of symmetry is defined by the equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Point P has coordinates (2,1) -/
def point_P : ℝ × ℝ := (2, 1)

/-- The symmetric point of P with respect to the line x + y - 1 = 0 -/
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.2 - 1, P.1 - 1)

theorem circle_center_coordinates (a b : ℝ) :
  (circle_equation 2 1 a b) ∧ 
  (circle_equation (symmetric_point point_P).1 (symmetric_point point_P).2 a b) →
  ∃ (h k : ℝ), h = 0 ∧ k = 1 ∧ 
    ∀ (x y : ℝ), circle_equation x y a b ↔ (x - h)^2 + (y - k)^2 = h^2 + k^2 - b :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2393_239338


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_example_l2393_239331

/-- The sum of an arithmetic sequence with given parameters. -/
def arithmetic_sequence_sum (n : ℕ) (a l : ℤ) : ℤ :=
  n * (a + l) / 2

/-- Theorem stating that the sum of the given arithmetic sequence is 175. -/
theorem arithmetic_sequence_sum_example :
  arithmetic_sequence_sum 10 (-5) 40 = 175 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_example_l2393_239331


namespace NUMINAMATH_CALUDE_r_fraction_of_total_l2393_239369

/-- Given a total amount of 6000 and r having 2400, prove that the fraction of the total amount that r has is 2/5 -/
theorem r_fraction_of_total (total : ℕ) (r_amount : ℕ) 
  (h_total : total = 6000) (h_r : r_amount = 2400) : 
  (r_amount : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_r_fraction_of_total_l2393_239369


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2393_239320

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 2 * a 9 = -8) →
  (a 1 + a 13 = 17 ∨ a 1 + a 13 = -17/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2393_239320


namespace NUMINAMATH_CALUDE_problem_statement_l2393_239307

theorem problem_statement (x Q : ℝ) (h : 2 * (5 * x + 3 * Real.pi) = Q) :
  4 * (10 * x + 6 * Real.pi + 2) = 4 * Q + 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2393_239307


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l2393_239357

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpendicular l α)
  (h3 : parallel l m) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l2393_239357


namespace NUMINAMATH_CALUDE_bridge_length_l2393_239399

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 265 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2393_239399


namespace NUMINAMATH_CALUDE_football_victory_points_l2393_239339

/-- Represents the points system in a football competition -/
structure FootballPoints where
  victory : ℕ
  draw : ℕ := 1
  defeat : ℕ := 0

/-- Represents the state of a team in the competition -/
structure TeamState where
  totalMatches : ℕ := 20
  playedMatches : ℕ := 5
  currentPoints : ℕ := 8
  targetPoints : ℕ := 40
  minRemainingWins : ℕ := 9

/-- The minimum number of points for a victory that satisfies the given conditions -/
def minVictoryPoints (points : FootballPoints) (state : TeamState) : Prop :=
  points.victory = 3 ∧
  points.victory * state.minRemainingWins + 
    (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw ≥ 
    state.targetPoints - state.currentPoints ∧
  ∀ v : ℕ, v < points.victory → 
    v * state.minRemainingWins + 
      (state.totalMatches - state.playedMatches - state.minRemainingWins) * points.draw < 
      state.targetPoints - state.currentPoints

theorem football_victory_points :
  ∃ (points : FootballPoints) (state : TeamState), minVictoryPoints points state := by
  sorry

end NUMINAMATH_CALUDE_football_victory_points_l2393_239339


namespace NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l2393_239329

theorem arithmetic_progression_implies_equal_numbers 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_arithmetic : (Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2 = (a + b) / 2) : 
  a = b := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_implies_equal_numbers_l2393_239329


namespace NUMINAMATH_CALUDE_expression_simplification_l2393_239347

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^2 - 9 * b^2) / (3 * a * b) - (6 * a * b - 9 * b^2) / (4 * a * b - 3 * a^2) =
  2 * (a^2 - 9 * b^2) / (3 * a * b) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2393_239347


namespace NUMINAMATH_CALUDE_base8_of_215_l2393_239325

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : ℕ := sorry

theorem base8_of_215 : toBase8 215 = 327 := by sorry

end NUMINAMATH_CALUDE_base8_of_215_l2393_239325


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2393_239304

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 21 * k)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 21 * k₄) := by
  sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l2393_239304


namespace NUMINAMATH_CALUDE_largest_inscribed_right_triangle_area_l2393_239398

/-- The area of the largest inscribed right triangle in a circle -/
theorem largest_inscribed_right_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (diameter * r) / 2
  max_triangle_area = 64 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_right_triangle_area_l2393_239398


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2393_239374

/-- Theorem: If a fruit seller sells 40% of his apples and has 420 apples remaining, 
    then he originally had 700 apples. -/
theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℚ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2393_239374


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2393_239351

theorem scientific_notation_equivalence : 
  1400000000 = (1.4 : ℝ) * (10 : ℝ) ^ 9 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2393_239351


namespace NUMINAMATH_CALUDE_smartphone_savings_plan_l2393_239362

theorem smartphone_savings_plan (smartphone_cost initial_savings : ℕ) 
  (saving_months weeks_per_month : ℕ) : 
  smartphone_cost = 160 →
  initial_savings = 40 →
  saving_months = 2 →
  weeks_per_month = 4 →
  (smartphone_cost - initial_savings) / (saving_months * weeks_per_month) = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_savings_plan_l2393_239362


namespace NUMINAMATH_CALUDE_divisor_property_solutions_l2393_239312

/-- The number of positive divisors of a positive integer n -/
def num_divisors (n : ℕ+) : ℕ+ :=
  sorry

/-- The property that the fourth power of the number of divisors equals the number itself -/
def has_divisor_property (m : ℕ+) : Prop :=
  (num_divisors m) ^ 4 = m

/-- Theorem stating that only 625, 6561, and 4100625 satisfy the divisor property -/
theorem divisor_property_solutions :
  ∀ m : ℕ+, has_divisor_property m ↔ m ∈ ({625, 6561, 4100625} : Set ℕ+) :=
sorry

end NUMINAMATH_CALUDE_divisor_property_solutions_l2393_239312


namespace NUMINAMATH_CALUDE_remainder_4015_div_32_l2393_239360

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4015_div_32_l2393_239360


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2393_239395

/-- A primitive third root of unity -/
noncomputable def α : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def P (C D E : ℂ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

theorem polynomial_divisibility (C D E : ℂ) :
  (∀ x, x^2 - x + 1 = 0 → P C D E x = 0) →
  C + D + E = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2393_239395


namespace NUMINAMATH_CALUDE_fraction_equality_l2393_239316

theorem fraction_equality (x y : ℝ) (h : x / y = 5 / 3) :
  x / (y - x) = -3 / 2 ∧ x / (y - x) ≠ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2393_239316


namespace NUMINAMATH_CALUDE_father_son_ages_l2393_239323

/-- Proves that given the conditions about the ages of a father and son, their present ages are 36 and 12 years respectively. -/
theorem father_son_ages (father_age son_age : ℕ) : 
  father_age = 3 * son_age ∧ 
  father_age + 12 = 2 * (son_age + 12) →
  father_age = 36 ∧ son_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_father_son_ages_l2393_239323


namespace NUMINAMATH_CALUDE_intersection_empty_condition_l2393_239378

theorem intersection_empty_condition (a : ℝ) : 
  let A : Set ℝ := Set.Iio (2 * a)
  let B : Set ℝ := Set.Ioi (3 - a^2)
  A ∩ B = ∅ → 2 * a ≤ 3 - a^2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_empty_condition_l2393_239378


namespace NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l2393_239370

-- Define the conditions
def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0

def condition_q (x : ℝ) : Prop := |x - 2| < 1

-- Theorem statement
theorem condition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, condition_p x → condition_q x) ∧
  ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by sorry


end NUMINAMATH_CALUDE_condition_p_sufficient_not_necessary_for_q_l2393_239370


namespace NUMINAMATH_CALUDE_notebooks_distribution_l2393_239306

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Condition 1
  (N / (C / 2) = 16) →  -- Condition 2
  N = 512 := by sorry

end NUMINAMATH_CALUDE_notebooks_distribution_l2393_239306


namespace NUMINAMATH_CALUDE_orange_juice_profit_l2393_239358

-- Define the tree types and their properties
structure TreeType where
  name : String
  trees : ℕ
  orangesPerTree : ℕ
  pricePerCup : ℚ

-- Define the additional costs
def additionalCosts : ℚ := 180

-- Define the number of oranges needed to make one cup of juice
def orangesPerCup : ℕ := 3

-- Define the tree types
def valencia : TreeType := ⟨"Valencia", 150, 400, 4⟩
def navel : TreeType := ⟨"Navel", 120, 650, 9/2⟩
def bloodOrange : TreeType := ⟨"Blood Orange", 160, 500, 5⟩

-- Calculate profit for a single tree type
def calculateProfit (t : TreeType) : ℚ :=
  let totalOranges := t.trees * t.orangesPerTree
  let totalCups := totalOranges / orangesPerCup
  totalCups * t.pricePerCup - additionalCosts

-- Calculate total profit
def totalProfit : ℚ :=
  calculateProfit valencia + calculateProfit navel + calculateProfit bloodOrange

-- Theorem statement
theorem orange_juice_profit : totalProfit = 329795 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_profit_l2393_239358


namespace NUMINAMATH_CALUDE_total_clothes_donated_l2393_239345

/-- Proves that the total number of clothes donated is 87 given the specified conditions --/
theorem total_clothes_donated (shirts : ℕ) (pants : ℕ) (shorts : ℕ) : 
  shirts = 12 →
  pants = 5 * shirts →
  shorts = pants / 4 →
  shirts + pants + shorts = 87 := by
  sorry

end NUMINAMATH_CALUDE_total_clothes_donated_l2393_239345


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2393_239367

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) - 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2393_239367


namespace NUMINAMATH_CALUDE_evaluate_y_l2393_239337

theorem evaluate_y (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) - 2 = |x - 2| + |x + 3| - 2 :=
by sorry

end NUMINAMATH_CALUDE_evaluate_y_l2393_239337


namespace NUMINAMATH_CALUDE_max_servings_emily_l2393_239380

/-- Represents the recipe requirements for 4 servings -/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (water : ℚ)
  (milk : ℚ)

/-- Represents Emily's available ingredients -/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

def recipe : Recipe :=
  { chocolate := 3
  , sugar := 1/2
  , water := 2
  , milk := 3 }

def emily : Available :=
  { chocolate := 9
  , sugar := 3
  , milk := 10 }

/-- Calculates the number of servings possible for a given ingredient -/
def servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

theorem max_servings_emily :
  let chocolate_servings := servings_for_ingredient recipe.chocolate emily.chocolate
  let sugar_servings := servings_for_ingredient recipe.sugar emily.sugar
  let milk_servings := servings_for_ingredient recipe.milk emily.milk
  min chocolate_servings (min sugar_servings milk_servings) = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_emily_l2393_239380


namespace NUMINAMATH_CALUDE_a_formula_T_formula_l2393_239361

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom S3_eq_0 : S 3 = 0
axiom S5_eq_neg5 : S 5 = -5

-- Theorem 1: General formula for a_n
theorem a_formula (n : ℕ) : a n = -n + 2 := sorry

-- Define the sequence 1 / (a_{2n-1} * a_{2n+1})
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of b
def T (n : ℕ) : ℚ := sorry

-- Theorem 2: Sum of the first n terms of b
theorem T_formula (n : ℕ) : T n = n / (1 - 2*n) := sorry

end NUMINAMATH_CALUDE_a_formula_T_formula_l2393_239361


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_100_l2393_239356

theorem largest_multiple_of_8_less_than_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_100_l2393_239356


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2393_239303

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ 
  x ∈ Set.Ici 5 ∪ Set.Iic (-6) ∪ Set.Icc (-2) 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2393_239303


namespace NUMINAMATH_CALUDE_gcf_of_36_and_60_l2393_239300

theorem gcf_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_36_and_60_l2393_239300


namespace NUMINAMATH_CALUDE_max_angle_point_is_tangency_point_l2393_239385

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A structure representing a circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Function to calculate the angle between three points -/
def angle (A B M : Point) : ℝ := sorry

/-- Function to check if a point is on a line -/
def pointOnLine (P : Point) (l : Line) : Prop := sorry

/-- Function to check if a line intersects a segment -/
def lineIntersectsSegment (l : Line) (A B : Point) : Prop := sorry

/-- Function to check if a circle passes through two points -/
def circlePassesThroughPoints (C : Circle) (A B : Point) : Prop := sorry

/-- Function to check if a circle is tangent to a line -/
def circleTangentToLine (C : Circle) (l : Line) : Prop := sorry

/-- Theorem stating that the point M on line (d) that maximizes the angle ∠AMB
    is the point of tangency of the smallest circumcircle passing through A and B
    with the line (d) -/
theorem max_angle_point_is_tangency_point
  (A B : Point) (d : Line) 
  (h : ¬ lineIntersectsSegment d A B) :
  ∃ (M : Point) (C : Circle),
    pointOnLine M d ∧
    circlePassesThroughPoints C A B ∧
    circleTangentToLine C d ∧
    (∀ (M' : Point), pointOnLine M' d → angle A M' B ≤ angle A M B) :=
sorry

end NUMINAMATH_CALUDE_max_angle_point_is_tangency_point_l2393_239385


namespace NUMINAMATH_CALUDE_smallest_values_for_equation_l2393_239344

theorem smallest_values_for_equation (a b c : ℤ) 
  (ha : a > 2) (hb : b < 10) (hc : c ≥ 0) 
  (heq : 32 = a + 2*b + 3*c) : 
  a = 4 ∧ b = 8 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_values_for_equation_l2393_239344


namespace NUMINAMATH_CALUDE_a_equals_negative_one_l2393_239342

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (a - 1) + Complex.I^4 * (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a equals -1 -/
theorem a_equals_negative_one : 
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_a_equals_negative_one_l2393_239342


namespace NUMINAMATH_CALUDE_smallest_ending_9_div_13_proof_l2393_239353

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def smallest_ending_9_div_13 : ℕ := 129

theorem smallest_ending_9_div_13_proof :
  (ends_in_9 smallest_ending_9_div_13) ∧
  (smallest_ending_9_div_13 % 13 = 0) ∧
  (∀ m : ℕ, m < smallest_ending_9_div_13 → ¬(ends_in_9 m ∧ m % 13 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_ending_9_div_13_proof_l2393_239353


namespace NUMINAMATH_CALUDE_final_time_sum_l2393_239364

def initial_time : Nat := 15 * 60 * 60  -- 3:00:00 PM in seconds
def elapsed_time : Nat := 137 * 60 * 60 + 58 * 60 + 59  -- 137h 58m 59s in seconds

def final_time : Nat := (initial_time + elapsed_time) % (24 * 60 * 60)

def hours (t : Nat) : Nat := (t / 3600) % 12
def minutes (t : Nat) : Nat := (t / 60) % 60
def seconds (t : Nat) : Nat := t % 60

theorem final_time_sum :
  hours final_time + minutes final_time + seconds final_time = 125 := by
sorry

end NUMINAMATH_CALUDE_final_time_sum_l2393_239364


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2393_239318

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2393_239318


namespace NUMINAMATH_CALUDE_circle_ranges_l2393_239394

/-- The equation of a circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

/-- The range of m for which the equation represents a circle -/
def m_range (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

/-- The range of the radius r of the circle -/
def r_range (r : ℝ) : Prop :=
  0 < r ∧ r ≤ 4/Real.sqrt 7

/-- Theorem stating the ranges of m and r for the given circle equation -/
theorem circle_ranges :
  (∃ x y : ℝ, circle_equation x y m) → m_range m ∧ (∃ r : ℝ, r_range r) :=
by sorry

end NUMINAMATH_CALUDE_circle_ranges_l2393_239394


namespace NUMINAMATH_CALUDE_ivy_cupcakes_l2393_239302

/-- The number of cupcakes Ivy baked in the morning -/
def morning_cupcakes : ℕ := 20

/-- The additional number of cupcakes Ivy baked in the afternoon compared to the morning -/
def afternoon_extra : ℕ := 15

/-- The total number of cupcakes Ivy baked -/
def total_cupcakes : ℕ := morning_cupcakes + (morning_cupcakes + afternoon_extra)

theorem ivy_cupcakes : total_cupcakes = 55 := by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l2393_239302


namespace NUMINAMATH_CALUDE_mnp_value_l2393_239349

theorem mnp_value (a b x z : ℝ) (m n p : ℤ) 
  (h : a^12 * x * z - a^10 * z - a^9 * x = a^8 * (b^6 - 1)) 
  (h_equiv : (a^m * x - a^n) * (a^p * z - a^3) = a^8 * b^6) : 
  m * n * p = 4 := by
sorry

end NUMINAMATH_CALUDE_mnp_value_l2393_239349


namespace NUMINAMATH_CALUDE_lucas_1364_units_digit_l2393_239335

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Property that Lucas numbers' units digits repeat every 12 terms -/
axiom lucas_units_period (n : ℕ) : lucas n % 10 = lucas (n % 12) % 10

/-- L₁₅ equals 1364 -/
axiom L_15 : lucas 15 = 1364

/-- Theorem: The units digit of L₁₃₆₄ is 7 -/
theorem lucas_1364_units_digit : lucas 1364 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_1364_units_digit_l2393_239335


namespace NUMINAMATH_CALUDE_point_on_line_for_all_k_l2393_239381

/-- The point (-2, -1) lies on the line kx + y + 2k + 1 = 0 for all values of k. -/
theorem point_on_line_for_all_k :
  ∀ (k : ℝ), k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_for_all_k_l2393_239381


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2393_239386

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂ ∧ b₁ ≠ b₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = -(a/b)x - (c/b) -/
def slope_intercept_form (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -(a/b) * x - (c/b) :=
  sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x + 2 * y - 1 = 0 ↔ 8 * x + a * y + (2 - a) = 0) →
  a = -4 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2393_239386


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2393_239330

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

theorem parabola_focus_distance (para : Parabola) 
  (A : PointOnParabola para) (h : A.y = Real.sqrt 2) :
  (3 * A.x = A.x + para.p / 2) → para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2393_239330


namespace NUMINAMATH_CALUDE_problem_solution_l2393_239346

theorem problem_solution (a : ℝ) : 3 ∈ ({a, a^2 - 2*a} : Set ℝ) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2393_239346


namespace NUMINAMATH_CALUDE_max_value_and_constraint_optimization_l2393_239373

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - 2*|x + 1|

-- State the theorem
theorem max_value_and_constraint_optimization :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧ 
  m = 2 ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + 2*b^2 + c^2 = m → 
    ab + bc ≤ 1 ∧ ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀^2 + 2*b₀^2 + c₀^2 = m ∧ a₀*b₀ + b₀*c₀ = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_and_constraint_optimization_l2393_239373


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2393_239384

theorem cubic_equation_roots (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2393_239384


namespace NUMINAMATH_CALUDE_plot_width_l2393_239371

/-- Proves that a rectangular plot with given conditions has a width of 47.5 meters -/
theorem plot_width (length : ℝ) (poles : ℕ) (pole_distance : ℝ) (width : ℝ) :
  length = 90 →
  poles = 56 →
  pole_distance = 5 →
  (poles - 1 : ℝ) * pole_distance = 2 * (length + width) →
  width = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_plot_width_l2393_239371


namespace NUMINAMATH_CALUDE_max_eccentricity_ellipse_l2393_239301

/-- The maximum eccentricity of an ellipse with given properties -/
theorem max_eccentricity_ellipse :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (2, 0)
  let P : ℝ → ℝ × ℝ := λ x => (x, x + 3)
  let dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let c : ℝ := dist A B / 2
  let a (x : ℝ) : ℝ := (dist (P x) A + dist (P x) B) / 2
  let e (x : ℝ) : ℝ := c / a x
  ∃ (x : ℝ), ∀ (y : ℝ), e y ≤ e x ∧ e x = 2 * Real.sqrt 26 / 13 :=
sorry

end NUMINAMATH_CALUDE_max_eccentricity_ellipse_l2393_239301


namespace NUMINAMATH_CALUDE_only_paintable_integer_l2393_239334

/-- Represents a painting configuration for the fence. -/
structure PaintingConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval
  v : ℕ+  -- Victor's interval

/-- Checks if a picket is painted by Harold. -/
def paintedByHarold (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.h.val = 1

/-- Checks if a picket is painted by Tanya. -/
def paintedByTanya (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.t.val = 2

/-- Checks if a picket is painted by Ulysses. -/
def paintedByUlysses (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.u.val = 3

/-- Checks if a picket is painted by Victor. -/
def paintedByVictor (config : PaintingConfig) (picket : ℕ) : Prop :=
  picket % config.v.val = 4

/-- Checks if a picket is painted by exactly one person. -/
def paintedOnce (config : PaintingConfig) (picket : ℕ) : Prop :=
  (paintedByHarold config picket ∨ paintedByTanya config picket ∨
   paintedByUlysses config picket ∨ paintedByVictor config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByTanya config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByHarold config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByUlysses config picket) ∧
  ¬(paintedByTanya config picket ∧ paintedByVictor config picket) ∧
  ¬(paintedByUlysses config picket ∧ paintedByVictor config picket)

/-- Checks if a configuration is paintable. -/
def isPaintable (config : PaintingConfig) : Prop :=
  ∀ picket : ℕ, picket > 0 → paintedOnce config picket

/-- Calculates the paintable integer for a configuration. -/
def paintableInteger (config : PaintingConfig) : ℕ :=
  1000 * config.h.val + 100 * config.t.val + 10 * config.u.val + config.v.val

/-- The main theorem stating that 4812 is the only paintable integer. -/
theorem only_paintable_integer :
  ∀ config : PaintingConfig, isPaintable config → paintableInteger config = 4812 := by
  sorry


end NUMINAMATH_CALUDE_only_paintable_integer_l2393_239334


namespace NUMINAMATH_CALUDE_bus_stoppage_time_l2393_239309

/-- Proves that a bus with given speeds stops for 30 minutes per hour -/
theorem bus_stoppage_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 32)
  (h2 : speed_with_stops = 16) : 
  (1 - speed_with_stops / speed_without_stops) * 60 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bus_stoppage_time_l2393_239309


namespace NUMINAMATH_CALUDE_rice_and_flour_consumption_l2393_239350

theorem rice_and_flour_consumption (initial_rice initial_flour consumed : ℕ) : 
  initial_rice = 500 →
  initial_flour = 200 →
  initial_rice - consumed = 7 * (initial_flour - consumed) →
  consumed = 150 := by
sorry

end NUMINAMATH_CALUDE_rice_and_flour_consumption_l2393_239350


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2393_239321

theorem shaded_area_percentage (square_side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * square_side_length - rectangle_length) * square_side_length) / (rectangle_width * rectangle_length)) * 100 = 14.29 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2393_239321


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2393_239333

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The x-coordinate of one focus -/
  focus_x : ℝ
  /-- The y-coordinate of one focus -/
  focus_y : ℝ
  /-- The slope of one asymptote -/
  asymptote_slope : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with one focus at (5,0) and one asymptote with slope 3/4 is 5/4 -/
theorem hyperbola_eccentricity :
  let h : Hyperbola := { focus_x := 5, focus_y := 0, asymptote_slope := 3/4 }
  eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2393_239333


namespace NUMINAMATH_CALUDE_largest_n_for_product_1764_l2393_239390

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 1764)
  : (∀ m : ℕ, (∃ n : ℕ, a n * b n = 1764) → m ≤ 44) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_product_1764_l2393_239390


namespace NUMINAMATH_CALUDE_petya_wins_l2393_239382

/-- Represents the game state -/
structure GameState where
  total_players : Nat
  vasya_turn : Bool

/-- Represents the result of the game -/
inductive GameResult
  | VasyaWins
  | PetyaWins

/-- Optimal play function -/
def optimal_play (state : GameState) : GameResult :=
  sorry

/-- The main theorem -/
theorem petya_wins :
  ∀ (initial_state : GameState),
    initial_state.total_players = 2022 →
    initial_state.vasya_turn = true →
    optimal_play initial_state = GameResult.PetyaWins :=
  sorry

end NUMINAMATH_CALUDE_petya_wins_l2393_239382


namespace NUMINAMATH_CALUDE_complete_square_constant_l2393_239379

theorem complete_square_constant (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
sorry

end NUMINAMATH_CALUDE_complete_square_constant_l2393_239379


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l2393_239392

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 = 1 →
  (a 5 + (3/2) * a 4) / 2 = 1/2 →
  a 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a1_l2393_239392


namespace NUMINAMATH_CALUDE_range_of_x_for_sqrt_4_minus_x_l2393_239354

theorem range_of_x_for_sqrt_4_minus_x : 
  ∀ x : ℝ, (∃ y : ℝ, y^2 = 4 - x) ↔ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_x_for_sqrt_4_minus_x_l2393_239354


namespace NUMINAMATH_CALUDE_roots_of_equation_l2393_239376

def f (x : ℝ) : ℝ := x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {-2, -1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2393_239376


namespace NUMINAMATH_CALUDE_tangent_lines_not_always_same_l2393_239375

-- Define a curve as a function from ℝ to ℝ
def Curve := ℝ → ℝ

-- Define a point in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in ℝ² as a pair of slope and y-intercept
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a tangent line to a curve at a point
def tangentLineToCurve (f : Curve) (p : Point) : Line := sorry

-- Define a tangent line passing through a point
def tangentLineThroughPoint (p : Point) : Line := sorry

-- The theorem to prove
theorem tangent_lines_not_always_same (f : Curve) (p : Point) : 
  ¬ ∀ (f : Curve) (p : Point), tangentLineToCurve f p = tangentLineThroughPoint p :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_not_always_same_l2393_239375


namespace NUMINAMATH_CALUDE_unique_a_value_l2393_239317

def A (a : ℝ) : Set ℝ := {1, a}
def B : Set ℝ := {1, 3}

theorem unique_a_value (a : ℝ) (h : A a ∪ B = {1, 2, 3}) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2393_239317


namespace NUMINAMATH_CALUDE_smallest_integer_y_l2393_239377

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, (z : ℚ) / 4 + 3 / 7 > 2 / 3 → y ≤ z :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l2393_239377


namespace NUMINAMATH_CALUDE_marks_departure_time_correct_l2393_239326

-- Define the problem parameters
def robs_normal_time : ℝ := 1
def robs_additional_time : ℝ := 0.5
def marks_normal_time_factor : ℝ := 3
def marks_time_reduction : ℝ := 0.2
def time_zone_difference : ℝ := 2
def robs_departure_time : ℝ := 11

-- Define the function to calculate Mark's departure time
def calculate_marks_departure_time : ℝ :=
  let robs_travel_time := robs_normal_time + robs_additional_time
  let marks_normal_time := marks_normal_time_factor * robs_normal_time
  let marks_travel_time := marks_normal_time * (1 - marks_time_reduction)
  let robs_arrival_time := robs_departure_time + robs_travel_time
  let marks_arrival_time := robs_arrival_time + time_zone_difference
  marks_arrival_time - marks_travel_time

-- Theorem statement
theorem marks_departure_time_correct :
  calculate_marks_departure_time = 11 + 36 / 60 :=
sorry

end NUMINAMATH_CALUDE_marks_departure_time_correct_l2393_239326


namespace NUMINAMATH_CALUDE_total_oranges_picked_l2393_239328

/-- The total number of oranges picked by Joan and Sara -/
def total_oranges (joan_oranges sara_oranges : ℕ) : ℕ :=
  joan_oranges + sara_oranges

/-- Theorem: Given that Joan picked 37 oranges and Sara picked 10 oranges,
    the total number of oranges picked is 47 -/
theorem total_oranges_picked :
  total_oranges 37 10 = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l2393_239328


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l2393_239352

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  35 * C = 21 * S →
  ((S - C) / C) * 100 = 200 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l2393_239352


namespace NUMINAMATH_CALUDE_unique_delivery_exists_l2393_239310

/-- Represents the amount of cargo delivered to each warehouse -/
structure Delivery where
  first : Int
  second : Int
  third : Int

/-- Checks if a delivery satisfies the given conditions -/
def satisfiesConditions (d : Delivery) : Prop :=
  d.first + d.second = 400 ∧
  d.second + d.third = -300 ∧
  d.first + d.third = -440

/-- The theorem stating that there is a unique delivery satisfying the conditions -/
theorem unique_delivery_exists : ∃! d : Delivery, satisfiesConditions d ∧ 
  d.first = -130 ∧ d.second = -270 ∧ d.third = 230 := by
  sorry

end NUMINAMATH_CALUDE_unique_delivery_exists_l2393_239310


namespace NUMINAMATH_CALUDE_contradiction_proof_l2393_239393

theorem contradiction_proof (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end NUMINAMATH_CALUDE_contradiction_proof_l2393_239393


namespace NUMINAMATH_CALUDE_dot_product_range_l2393_239368

-- Define the points O and A
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)

-- Define the set of points P on the right branch of the hyperbola
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 - p.2^2 = 1 ∧ p.1 > 0}

-- Define the dot product of OA and OP
def dot_product (p : ℝ × ℝ) : ℝ := p.1 + p.2

-- Theorem statement
theorem dot_product_range :
  ∀ p ∈ P, dot_product p > 0 ∧ ∀ M : ℝ, ∃ q ∈ P, dot_product q > M :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l2393_239368


namespace NUMINAMATH_CALUDE_no_natural_solution_equation_l2393_239343

theorem no_natural_solution_equation : ∀ x y : ℕ, 2^x + 21^x ≠ y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_equation_l2393_239343


namespace NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l2393_239359

theorem sixth_grade_homework_forgetfulness (students_A : ℕ) (students_B : ℕ) 
  (forgot_A_percent : ℚ) (forgot_B_percent : ℚ) (total_forgot_percent : ℚ) :
  students_A = 20 →
  forgot_A_percent = 20 / 100 →
  forgot_B_percent = 15 / 100 →
  total_forgot_percent = 16 / 100 →
  (students_A : ℚ) * forgot_A_percent + (students_B : ℚ) * forgot_B_percent = 
    total_forgot_percent * ((students_A : ℚ) + (students_B : ℚ)) →
  students_B = 80 := by
sorry

end NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l2393_239359


namespace NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l2393_239365

theorem sqrt_fraction_equivalence (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x - 1) / x)) = -x := by sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equivalence_l2393_239365


namespace NUMINAMATH_CALUDE_ice_cube_melting_l2393_239336

theorem ice_cube_melting (V : ℝ) : 
  V > 0 →
  (1/5) * (3/4) * (2/3) * (1/2) * V = 30 →
  V = 150 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_melting_l2393_239336


namespace NUMINAMATH_CALUDE_rectangle_fit_impossibility_l2393_239396

theorem rectangle_fit_impossibility : 
  ∀ (a b c d : ℝ), 
    a = 5 ∧ b = 6 ∧ c = 3 ∧ d = 8 → 
    (c^2 + d^2 : ℝ) > (a^2 + b^2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_fit_impossibility_l2393_239396


namespace NUMINAMATH_CALUDE_simplify_fraction_l2393_239372

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 0) :
  (1 - 1 / (x - 3)) / ((x^2 - 4*x) / (x^2 - 9)) = (x + 3) / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2393_239372


namespace NUMINAMATH_CALUDE_jake_has_eight_peaches_l2393_239387

-- Define the number of peaches each person has
def steven_peaches : ℕ := 15
def jill_peaches : ℕ := steven_peaches - 14
def jake_peaches : ℕ := steven_peaches - 7

-- Theorem statement
theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eight_peaches_l2393_239387


namespace NUMINAMATH_CALUDE_correct_observation_value_l2393_239355

/-- Proves that the correct value of an incorrectly recorded observation is 58,
    given the initial and corrected means of a set of observations. -/
theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℚ)
  (incorrect_value : ℚ)
  (corrected_mean : ℚ)
  (h_n : n = 40)
  (h_initial : initial_mean = 36)
  (h_incorrect : incorrect_value = 20)
  (h_corrected : corrected_mean = 36.45) :
  (n : ℚ) * corrected_mean - ((n : ℚ) * initial_mean - incorrect_value) = 58 :=
sorry

end NUMINAMATH_CALUDE_correct_observation_value_l2393_239355


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l2393_239366

def age_difference : ℕ := 20
def younger_present_age : ℕ := 35
def years_ago : ℕ := 15

def elder_present_age : ℕ := younger_present_age + age_difference

def younger_past_age : ℕ := younger_present_age - years_ago
def elder_past_age : ℕ := elder_present_age - years_ago

theorem age_ratio_theorem :
  (elder_past_age % younger_past_age = 0) →
  (elder_past_age / younger_past_age = 2) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l2393_239366


namespace NUMINAMATH_CALUDE_vector_representation_l2393_239327

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, -2)
def a : ℝ × ℝ := (-4, 0)

theorem vector_representation :
  a = (-1 : ℝ) • e₁ + (-1 : ℝ) • e₂ := by sorry

end NUMINAMATH_CALUDE_vector_representation_l2393_239327


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2393_239389

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  ∃ r : ℝ, r ≠ 0 ∧ b = 10 * r ∧ (3/4) = b * r → 
  b = 5 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2393_239389


namespace NUMINAMATH_CALUDE_hezekiahs_age_l2393_239340

theorem hezekiahs_age (hezekiah_age : ℕ) (ryanne_age : ℕ) : 
  (ryanne_age = hezekiah_age + 7) → 
  (hezekiah_age + ryanne_age = 15) → 
  (hezekiah_age = 4) := by
sorry

end NUMINAMATH_CALUDE_hezekiahs_age_l2393_239340


namespace NUMINAMATH_CALUDE_base_seven_1732_equals_709_l2393_239388

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_1732_equals_709 :
  base_seven_to_ten [2, 3, 7, 1] = 709 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_1732_equals_709_l2393_239388


namespace NUMINAMATH_CALUDE_exists_divisible_by_four_l2393_239313

def collatz_sequence (a₁ : ℕ+) : ℕ → ℕ
  | 0 => a₁.val
  | n + 1 => 
    let prev := collatz_sequence a₁ n
    if prev % 2 = 0 then prev / 2 else 3 * prev + 1

theorem exists_divisible_by_four (a₁ : ℕ+) : 
  ∃ n : ℕ, (collatz_sequence a₁ n) % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_divisible_by_four_l2393_239313


namespace NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2393_239311

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_x : x > 1)
  (h_cos : Real.cos (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = Real.sqrt (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_in_terms_of_x_l2393_239311


namespace NUMINAMATH_CALUDE_conference_games_count_l2393_239363

/-- The number of teams in the basketball conference -/
def num_teams : ℕ := 10

/-- The number of times each team plays every other team in the conference -/
def intra_conference_games : ℕ := 2

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season for the basketball conference -/
def total_games : ℕ :=
  (num_teams.choose 2 * intra_conference_games) + (num_teams * non_conference_games)

theorem conference_games_count : total_games = 150 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l2393_239363
