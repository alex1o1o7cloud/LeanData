import Mathlib

namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2526_252625

/-- The quadratic function f(x) = x^2 - ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a

/-- The discriminant of f(x) -/
def discriminant (a : ℝ) : ℝ := a^2 - 4*a

/-- f(x) has two distinct zeros -/
def has_two_distinct_zeros (a : ℝ) : Prop := discriminant a > 0

/-- Condition "a > 4" is sufficient for f(x) to have two distinct zeros -/
theorem sufficient_condition (a : ℝ) (h : a > 4) : has_two_distinct_zeros a := by
  sorry

/-- Condition "a > 4" is not necessary for f(x) to have two distinct zeros -/
theorem not_necessary_condition : ∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a := by
  sorry

/-- "a > 4" is a sufficient but not necessary condition for f(x) to have two distinct zeros -/
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a > 4 → has_two_distinct_zeros a) ∧
  (∃ a : ℝ, a ≤ 4 ∧ has_two_distinct_zeros a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2526_252625


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2526_252607

theorem sin_negative_31pi_over_6 : Real.sin (-31 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l2526_252607


namespace NUMINAMATH_CALUDE_pets_remaining_l2526_252695

theorem pets_remaining (initial_pets : ℕ) (lost_pets : ℕ) (death_rate : ℚ) : 
  initial_pets = 16 → 
  lost_pets = 6 → 
  death_rate = 1/5 → 
  initial_pets - lost_pets - (death_rate * (initial_pets - lost_pets : ℚ)).floor = 8 := by
  sorry

end NUMINAMATH_CALUDE_pets_remaining_l2526_252695


namespace NUMINAMATH_CALUDE_swim_time_ratio_l2526_252679

/-- The ratio of time taken to swim upstream vs downstream -/
theorem swim_time_ratio 
  (Vm : ℝ) 
  (Vs : ℝ) 
  (h1 : Vm = 5) 
  (h2 : Vs = 1.6666666666666667) : 
  (Vm + Vs) / (Vm - Vs) = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l2526_252679


namespace NUMINAMATH_CALUDE_playerB_is_best_choice_l2526_252635

-- Define a structure for a player
structure Player where
  name : String
  average : Float
  variance : Float

-- Define the players
def playerA : Player := { name := "A", average := 9.2, variance := 3.6 }
def playerB : Player := { name := "B", average := 9.5, variance := 3.6 }
def playerC : Player := { name := "C", average := 9.5, variance := 7.4 }
def playerD : Player := { name := "D", average := 9.2, variance := 8.1 }

def players : List Player := [playerA, playerB, playerC, playerD]

-- Function to determine if a player is the best choice
def isBestChoice (p : Player) (players : List Player) : Prop :=
  (∀ q ∈ players, p.average ≥ q.average) ∧
  (∀ q ∈ players, p.variance ≤ q.variance) ∧
  (∃ q ∈ players, p.average > q.average ∨ p.variance < q.variance)

-- Theorem stating that playerB is the best choice
theorem playerB_is_best_choice : isBestChoice playerB players := by
  sorry

end NUMINAMATH_CALUDE_playerB_is_best_choice_l2526_252635


namespace NUMINAMATH_CALUDE_perpendicular_line_theorem_l2526_252614

structure Plane where
  -- Define a plane structure

structure Point where
  -- Define a point structure

structure Line where
  -- Define a line structure

-- Define perpendicularity between planes
def perpendicular (α β : Plane) : Prop := sorry

-- Define a point lying in a plane
def lies_in (A : Point) (α : Plane) : Prop := sorry

-- Define a line passing through a point
def passes_through (l : Line) (A : Point) : Prop := sorry

-- Define a line perpendicular to a plane
def perpendicular_to_plane (l : Line) (β : Plane) : Prop := sorry

-- Define a line lying in a plane
def line_in_plane (l : Line) (α : Plane) : Prop := sorry

theorem perpendicular_line_theorem (α β : Plane) (A : Point) :
  perpendicular α β →
  lies_in A α →
  ∃! l : Line, passes_through l A ∧ perpendicular_to_plane l β ∧ line_in_plane l α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_theorem_l2526_252614


namespace NUMINAMATH_CALUDE_subset_divisibility_property_l2526_252684

theorem subset_divisibility_property (A : Finset ℕ) (hA : A.card = 3) :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card = 2 ∧
  ∀ (x y : ℕ) (hx : x ∈ B) (hy : y ∈ B) (m n : ℕ) (hm : Odd m) (hn : Odd n),
  (10 : ℕ) ∣ (x^m * y^n - x^n * y^m) :=
sorry

end NUMINAMATH_CALUDE_subset_divisibility_property_l2526_252684


namespace NUMINAMATH_CALUDE_series_divergence_l2526_252677

theorem series_divergence (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 0 < a n)
  (h2 : ∀ n : ℕ, a n ≤ a (2 * n) + a (2 * n + 1)) :
  ¬ (Summable a) :=
sorry

end NUMINAMATH_CALUDE_series_divergence_l2526_252677


namespace NUMINAMATH_CALUDE_scale_division_l2526_252637

/-- Represents a length in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ

/-- Converts a Length to total inches -/
def Length.to_inches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- Converts total inches to a Length -/
def inches_to_length (total_inches : ℕ) : Length :=
  { feet := total_inches / 12, inches := total_inches % 12 }

theorem scale_division (scale : Length) (parts : ℕ) 
    (h1 : scale.feet = 6 ∧ scale.inches = 8) 
    (h2 : parts = 4) : 
  inches_to_length (scale.to_inches / parts) = { feet := 1, inches := 8 } := by
sorry

end NUMINAMATH_CALUDE_scale_division_l2526_252637


namespace NUMINAMATH_CALUDE_sample_size_is_100_l2526_252631

/-- Represents a city with its total sales and number of cars selected for investigation. -/
structure City where
  name : String
  totalSales : Nat
  selected : Nat

/-- Represents the sampling data for the car manufacturer's investigation. -/
def samplingData : List City :=
  [{ name := "A", totalSales := 420, selected := 30 },
   { name := "B", totalSales := 280, selected := 20 },
   { name := "C", totalSales := 700, selected := 50 }]

/-- Checks if the sampling is proportional to the total sales. -/
def isProportionalSampling (data : List City) : Prop :=
  ∀ i j, i ∈ data → j ∈ data → 
    i.totalSales * j.selected = j.totalSales * i.selected

/-- The total sample size is the sum of all selected cars. -/
def totalSampleSize (data : List City) : Nat :=
  (data.map (·.selected)).sum

/-- Theorem stating that the total sample size is 100 given the conditions. -/
theorem sample_size_is_100 (h : isProportionalSampling samplingData) :
  totalSampleSize samplingData = 100 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_100_l2526_252631


namespace NUMINAMATH_CALUDE_factorization_equality_l2526_252633

theorem factorization_equality (a : ℝ) : -9 - a^2 + 6*a = -(a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2526_252633


namespace NUMINAMATH_CALUDE_average_marks_combined_l2526_252660

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 70 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l2526_252660


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2526_252673

theorem min_value_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  r₁ + r₂ = r₁^2 + r₂^2 →
  r₁^2 + r₂^2 = r₁^4 + r₂^4 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (s t : ℝ), 
    (∀ x, x^2 - s*x + t = 0 ↔ x = r₁ ∨ x = r₂) →
    r₁ + r₂ = r₁^2 + r₂^2 →
    r₁^2 + r₂^2 = r₁^4 + r₂^4 →
    min ≤ 1/r₁^5 + 1/r₂^5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2526_252673


namespace NUMINAMATH_CALUDE_complex_magnitude_l2526_252601

theorem complex_magnitude (z : ℂ) : z = -2 - I → Complex.abs (z + I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2526_252601


namespace NUMINAMATH_CALUDE_quadratic_problem_l2526_252605

def quadratic_function (a b x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 + b

theorem quadratic_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : a > 0) 
  (h3 : 4 < a + |b| ∧ a + |b| < 9) 
  (h4 : quadratic_function a b 1 = 3) :
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), quadratic_function a b (x - y) = quadratic_function a b (x + y)) ∧
  (a = 2 ∧ b = 6) ∧
  (∃ (t : ℝ), (t = 1/2 ∨ t = 5/2) ∧
    ∀ (x : ℝ), t ≤ x ∧ x ≤ t + 1 → quadratic_function a b x ≥ 3/2 ∧
    ∃ (x₀ : ℝ), t ≤ x₀ ∧ x₀ ≤ t + 1 ∧ quadratic_function a b x₀ = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2526_252605


namespace NUMINAMATH_CALUDE_side_c_value_l2526_252670

/-- Given an acute triangle ABC with sides a = 4, b = 5, and area 5√3, 
    prove that the length of side c is √21 -/
theorem side_c_value (A B C : ℝ) (a b c : ℝ) (h_acute : A + B + C = π) 
  (h_a : a = 4) (h_b : b = 5) (h_area : (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3) :
  c = Real.sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_side_c_value_l2526_252670


namespace NUMINAMATH_CALUDE_bertha_initial_balls_l2526_252634

def tennis_balls (initial_balls : ℕ) : Prop :=
  let worn_out := 20 / 10
  let lost := 20 / 5
  let bought := 20 / 4 * 3
  let final_balls := initial_balls - 1 - worn_out - lost + bought
  final_balls = 10

theorem bertha_initial_balls : 
  ∃ (x : ℕ), tennis_balls x ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_bertha_initial_balls_l2526_252634


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_2017_l2526_252672

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define the pattern for the last two digits of powers of 7
def powerOf7Pattern (k : ℕ) : ℕ :=
  match k % 4 with
  | 0 => 01
  | 1 => 07
  | 2 => 49
  | 3 => 43
  | _ => 0  -- This case should never occur

theorem last_two_digits_of_7_2017 :
  lastTwoDigits (7^2017) = 07 :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_2017_l2526_252672


namespace NUMINAMATH_CALUDE_derivative_ln_2x_plus_1_l2526_252661

open Real

theorem derivative_ln_2x_plus_1 (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 1)) x = 2 / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_plus_1_l2526_252661


namespace NUMINAMATH_CALUDE_sample_represents_knowledge_l2526_252663

/-- Represents the population of teachers and students -/
def Population : ℕ := 1500

/-- Represents the sample size -/
def SampleSize : ℕ := 150

/-- Represents an individual in the population -/
structure Individual where
  id : ℕ
  isTeacher : Bool

/-- Represents the survey sample -/
structure Sample where
  individuals : Finset Individual
  size : ℕ
  h_size : size = SampleSize

/-- Represents the national security knowledge of an individual -/
def NationalSecurityKnowledge : Type := ℕ

/-- The theorem stating what the sample represents in the survey -/
theorem sample_represents_knowledge (sample : Sample) :
  ∃ (knowledge : Individual → NationalSecurityKnowledge),
    (∀ i ∈ sample.individuals, knowledge i ∈ Set.range knowledge) ∧
    (∀ i ∉ sample.individuals, knowledge i ∉ Set.range knowledge) :=
sorry

end NUMINAMATH_CALUDE_sample_represents_knowledge_l2526_252663


namespace NUMINAMATH_CALUDE_interest_difference_l2526_252685

def principal : ℚ := 250
def rate : ℚ := 4
def time : ℚ := 8

def simple_interest (p r t : ℚ) : ℚ := (p * r * t) / 100

theorem interest_difference :
  principal - simple_interest principal rate time = 170 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l2526_252685


namespace NUMINAMATH_CALUDE_team_a_championship_probability_l2526_252608

-- Define the game state
structure GameState where
  team_a_wins_needed : ℕ
  team_b_wins_needed : ℕ

-- Define the probability of Team A winning
def prob_team_a_wins (state : GameState) : ℚ :=
  if state.team_a_wins_needed = 0 then 1
  else if state.team_b_wins_needed = 0 then 0
  else sorry

-- Theorem statement
theorem team_a_championship_probability :
  let initial_state : GameState := ⟨1, 2⟩
  prob_team_a_wins initial_state = 3/4 :=
sorry

end NUMINAMATH_CALUDE_team_a_championship_probability_l2526_252608


namespace NUMINAMATH_CALUDE_berries_to_buy_l2526_252655

def total_needed : ℕ := 36
def strawberries : ℕ := 4
def blueberries : ℕ := 8
def raspberries : ℕ := 3
def blackberries : ℕ := 5

theorem berries_to_buy (total_needed strawberries blueberries raspberries blackberries : ℕ) :
  total_needed - (strawberries + blueberries + raspberries + blackberries) = 16 := by
  sorry

end NUMINAMATH_CALUDE_berries_to_buy_l2526_252655


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l2526_252680

theorem prime_divisors_of_50_factorial (n : ℕ) : n = 50 → 
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l2526_252680


namespace NUMINAMATH_CALUDE_two_dressers_capacity_l2526_252611

/-- The total number of pieces of clothing that can be held by two dressers -/
def total_clothing_capacity (first_dresser_drawers : ℕ) (first_dresser_capacity : ℕ) 
  (second_dresser_drawers : ℕ) (second_dresser_capacity : ℕ) : ℕ :=
  first_dresser_drawers * first_dresser_capacity + second_dresser_drawers * second_dresser_capacity

/-- Theorem stating the total clothing capacity of two specific dressers -/
theorem two_dressers_capacity : 
  total_clothing_capacity 12 8 6 10 = 156 := by
  sorry

end NUMINAMATH_CALUDE_two_dressers_capacity_l2526_252611


namespace NUMINAMATH_CALUDE_percentage_calculation_l2526_252627

theorem percentage_calculation (x : ℝ) : 
  (x / 100) * (25 / 100 * 1600) = 20 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2526_252627


namespace NUMINAMATH_CALUDE_stair_climbing_time_l2526_252644

theorem stair_climbing_time (n : ℕ) (a d : ℝ) (h : n = 7 ∧ a = 25 ∧ d = 10) :
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 385 :=
by sorry

end NUMINAMATH_CALUDE_stair_climbing_time_l2526_252644


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l2526_252669

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    Real.sqrt 2 * (2*a + 3) * Real.cos (θ - π/4) + 6 / (Real.sin θ + Real.cos θ) - 2 * Real.sin (2*θ) < 3*a + 6) 
  → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l2526_252669


namespace NUMINAMATH_CALUDE_triangle_midpoints_x_sum_l2526_252604

theorem triangle_midpoints_x_sum (p q r : ℝ) : 
  p + q + r = 15 → 
  (p + q) / 2 + (q + r) / 2 + (r + p) / 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoints_x_sum_l2526_252604


namespace NUMINAMATH_CALUDE_equidistant_points_on_line_l2526_252699

theorem equidistant_points_on_line :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (4 * x₁ + 3 * y₁ = 12) ∧
    (4 * x₂ + 3 * y₂ = 12) ∧
    (|x₁| = |y₁|) ∧
    (|x₂| = |y₂|) ∧
    (x₁ > 0 ∧ y₁ > 0) ∧
    (x₂ > 0 ∧ y₂ < 0) ∧
    ¬∃ (x₃ y₃ : ℝ),
      (4 * x₃ + 3 * y₃ = 12) ∧
      (|x₃| = |y₃|) ∧
      ((x₃ < 0 ∧ y₃ > 0) ∨ (x₃ < 0 ∧ y₃ < 0)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_on_line_l2526_252699


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l2526_252649

def num_sides : ℕ := 5
def num_dice : ℕ := 5

theorem distinct_numbers_probability :
  (Nat.factorial num_dice : ℚ) / (num_sides ^ num_dice : ℚ) = 120 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l2526_252649


namespace NUMINAMATH_CALUDE_jim_gas_cost_l2526_252657

/-- The total amount spent on gas by Jim -/
def total_gas_cost (nc_gallons : ℕ) (nc_price : ℚ) (va_gallons : ℕ) (va_price_increase : ℚ) : ℚ :=
  (nc_gallons : ℚ) * nc_price + (va_gallons : ℚ) * (nc_price + va_price_increase)

/-- Theorem stating that Jim's total gas cost is $50.00 -/
theorem jim_gas_cost :
  total_gas_cost 10 2 10 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jim_gas_cost_l2526_252657


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2526_252646

/-- The value of p for which the focus of the parabola y² = 2px (p > 0) 
    coincides with the right focus of the hyperbola x² - y² = 2 -/
theorem parabola_hyperbola_focus_coincidence (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 2 ∧ x = p/2 ∧ x = 2) → 
  p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l2526_252646


namespace NUMINAMATH_CALUDE_bus_seating_capacity_bus_total_capacity_l2526_252639

/-- The number of people that can sit in a bus given the seating arrangement --/
theorem bus_seating_capacity 
  (left_seats : ℕ) 
  (right_seats_difference : ℕ) 
  (people_per_seat : ℕ) 
  (back_seat_capacity : ℕ) : ℕ :=
  let right_seats := left_seats - right_seats_difference
  let left_capacity := left_seats * people_per_seat
  let right_capacity := right_seats * people_per_seat
  left_capacity + right_capacity + back_seat_capacity

/-- The total number of people that can sit in the bus is 90 --/
theorem bus_total_capacity : 
  bus_seating_capacity 15 3 3 9 = 90 := by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_bus_total_capacity_l2526_252639


namespace NUMINAMATH_CALUDE_walk_distance_difference_l2526_252616

theorem walk_distance_difference (total_distance susan_distance : ℕ) 
  (h1 : total_distance = 15)
  (h2 : susan_distance = 9) :
  susan_distance - (total_distance - susan_distance) = 3 :=
by sorry

end NUMINAMATH_CALUDE_walk_distance_difference_l2526_252616


namespace NUMINAMATH_CALUDE_square_difference_equals_690_l2526_252676

theorem square_difference_equals_690 : (23 + 15)^2 - (23^2 + 15^2) = 690 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_690_l2526_252676


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l2526_252692

/-- Given a rectangle where the length is twice the width and the perimeter in inches
    equals the area in square inches, prove that the width is 3 inches and the length is 6 inches. -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) :
  (6 * w = 2 * w^2) → (w = 3 ∧ 2 * w = 6) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l2526_252692


namespace NUMINAMATH_CALUDE_equation_solutions_l2526_252682

theorem equation_solutions :
  (∃ (x : ℝ), (1/3) * (x - 3)^2 = 12 ↔ x = 9 ∨ x = -3) ∧
  (∃ (x : ℝ), (2*x - 1)^2 = (1 - x)^2 ↔ x = 2/3 ∨ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2526_252682


namespace NUMINAMATH_CALUDE_parabola_tangent_circle_problem_l2526_252650

-- Define the parabola T₀: y = x²
def T₀ (x : ℝ) : ℝ := x^2

-- Define point P
def P : ℝ × ℝ := (1, -1)

-- Define the tangent line passing through P and intersecting T₀
def tangent_line (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ 
  T₀ x₁ = (x₁ - 1) * 2 * x₁ + (-1) ∧
  T₀ x₂ = (x₂ - 1) * 2 * x₂ + (-1)

-- Define circle E with center at P and tangent to line MN
def circle_E (r : ℝ) : Prop :=
  r = (4 : ℝ) / Real.sqrt 5

-- Define chords AC and BD passing through origin and perpendicular in circle E
def chords_ABCD (d₁ d₂ : ℝ) : Prop :=
  d₁^2 + d₂^2 = 2

-- Main theorem
theorem parabola_tangent_circle_problem :
  ∃ (x₁ x₂ r d₁ d₂ : ℝ),
    tangent_line x₁ x₂ ∧
    circle_E r ∧
    chords_ABCD d₁ d₂ ∧
    x₁ = 1 - Real.sqrt 2 ∧
    x₂ = 1 + Real.sqrt 2 ∧
    r^2 * Real.pi = (16 : ℝ) / 5 ∧
    2 * r^2 - (d₁^2 + d₂^2) ≤ (22 : ℝ) / 5 :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_circle_problem_l2526_252650


namespace NUMINAMATH_CALUDE_winning_pair_probability_l2526_252643

/-- Represents the color of a card -/
inductive Color
| Blue
| Purple

/-- Represents the letter on a card -/
inductive Letter
| A | B | C | D | E | F

/-- Represents a card with a color and a letter -/
structure Card where
  color : Color
  letter : Letter

/-- The deck of cards -/
def deck : List Card := sorry

/-- Checks if two cards form a winning pair -/
def is_winning_pair (c1 c2 : Card) : Bool := sorry

/-- Calculates the probability of drawing a winning pair -/
def probability_winning_pair : ℚ := sorry

/-- Theorem stating the probability of drawing a winning pair -/
theorem winning_pair_probability : 
  probability_winning_pair = 29 / 45 := by sorry

end NUMINAMATH_CALUDE_winning_pair_probability_l2526_252643


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2526_252651

-- Define the property that the function f must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x^2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2526_252651


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_l2526_252609

theorem greatest_three_digit_number : ∃ (n : ℕ), 
  n = 970 ∧ 
  n < 1000 ∧ 
  n ≥ 100 ∧ 
  ∃ (k : ℕ), n = 8 * k + 2 ∧ 
  ∃ (m : ℕ), n = 7 * m + 4 ∧ 
  ∀ (x : ℕ), x < 1000 ∧ x ≥ 100 ∧ (∃ (a : ℕ), x = 8 * a + 2) ∧ (∃ (b : ℕ), x = 7 * b + 4) → x ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_l2526_252609


namespace NUMINAMATH_CALUDE_skirt_cut_amount_l2526_252688

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end NUMINAMATH_CALUDE_skirt_cut_amount_l2526_252688


namespace NUMINAMATH_CALUDE_floor_calculation_l2526_252621

/-- The floor of (2011^3 / (2009 * 2010)) - (2009^3 / (2010 * 2011)) is 8 -/
theorem floor_calculation : 
  ⌊(2011^3 : ℝ) / (2009 * 2010) - (2009^3 : ℝ) / (2010 * 2011)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_calculation_l2526_252621


namespace NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2526_252613

theorem sum_reciprocals_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) : 
  1/a + 1/b + 1/c ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_inequality_l2526_252613


namespace NUMINAMATH_CALUDE_problem_solution_l2526_252652

theorem problem_solution (x y z w : ℝ) 
  (h1 : x * w > 0)
  (h2 : y * z > 0)
  (h3 : 1 / x + 1 / w = 20)
  (h4 : 1 / y + 1 / z = 25)
  (h5 : 1 / (x * w) = 6)
  (h6 : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2526_252652


namespace NUMINAMATH_CALUDE_veranda_area_l2526_252697

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 m². -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 132 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l2526_252697


namespace NUMINAMATH_CALUDE_min_total_length_l2526_252687

/-- A set of arcs on a circle -/
structure ArcSet :=
  (n : ℕ)                    -- number of arcs
  (arcs : Fin n → ℝ)         -- length of each arc in degrees
  (total_length : ℝ)         -- total length of all arcs
  (rotation_overlap : ∀ θ : ℝ, ∃ i : Fin n, ∃ j : Fin n, (arcs i + θ) % 360 = arcs j)
                             -- for any rotation, there's an overlap

/-- The minimum total length of arcs in an ArcSet is 360/n -/
theorem min_total_length (F : ArcSet) : F.total_length ≥ 360 / F.n :=
sorry

end NUMINAMATH_CALUDE_min_total_length_l2526_252687


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2526_252641

/-- Given a sphere with surface area 16π cm², prove its volume is 32π/3 cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 4 * π * r^2 = 16 * π → (4/3) * π * r^3 = (32 * π)/3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2526_252641


namespace NUMINAMATH_CALUDE_donation_ratio_l2526_252686

theorem donation_ratio (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300)
  (h2 : julie_donation = 4700) :
  (julie_donation - margo_donation) / (margo_donation + julie_donation) = 2 / 45 := by
  sorry

end NUMINAMATH_CALUDE_donation_ratio_l2526_252686


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l2526_252629

/-- The number of tennis balls originally ordered -/
def total_balls : ℕ := 114

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow : ℕ := 50

/-- The number of white balls received -/
def white_balls : ℕ := total_balls / 2

/-- The number of yellow balls received -/
def yellow_balls : ℕ := total_balls / 2 + extra_yellow

/-- The ratio of white balls to yellow balls after the error -/
def ball_ratio : ℚ := white_balls / yellow_balls

theorem tennis_ball_ratio : ball_ratio = 57 / 107 := by sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l2526_252629


namespace NUMINAMATH_CALUDE_x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l2526_252624

theorem x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two 
  (x : ℝ) (h : x + 1/x = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_ten_equals_fifty_plus_twenty_five_sqrt_five_over_two_l2526_252624


namespace NUMINAMATH_CALUDE_banana_group_size_l2526_252668

theorem banana_group_size 
  (total_bananas : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_group_size_l2526_252668


namespace NUMINAMATH_CALUDE_equation_solution_l2526_252612

theorem equation_solution (x : ℝ) : 
  (24 : ℝ) / 36 = Real.sqrt (x / 36) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2526_252612


namespace NUMINAMATH_CALUDE_price_change_theorem_l2526_252653

theorem price_change_theorem (initial_price : ℝ) (h : initial_price > 0) :
  let egg_price_new := initial_price * (1 - 0.02)
  let apple_price_new := initial_price * (1 + 0.10)
  let total_price_old := 2 * initial_price
  let total_price_new := egg_price_new + apple_price_new
  let price_increase := total_price_new - total_price_old
  let percentage_increase := price_increase / total_price_old * 100
  percentage_increase = 4 := by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l2526_252653


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l2526_252698

theorem cement_mixture_weight (sand_ratio : ℚ) (water_ratio : ℚ) (gravel_weight : ℚ) 
  (h1 : sand_ratio = 1/2)
  (h2 : water_ratio = 1/5)
  (h3 : gravel_weight = 15) :
  ∃ (total_weight : ℚ), 
    sand_ratio * total_weight + water_ratio * total_weight + gravel_weight = total_weight ∧
    total_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l2526_252698


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2526_252683

theorem modulus_of_complex_fraction : 
  Complex.abs (2 / (1 + Complex.I * Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2526_252683


namespace NUMINAMATH_CALUDE_points_in_quadrant_I_l2526_252647

theorem points_in_quadrant_I (x y : ℝ) : y > 3*x ∧ y > 5 - 2*x → x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_points_in_quadrant_I_l2526_252647


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l2526_252619

/-- Calculates the cost price per meter of cloth given the total meters sold,
    the total selling price, and the profit per meter. -/
def costPricePerMeter (totalMeters : ℕ) (sellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (sellingPrice - profitPerMeter * totalMeters) / totalMeters

/-- Proves that given the specified conditions, the cost price per meter of cloth is 95 Rs. -/
theorem cloth_cost_price_theorem (totalMeters sellingPrice profitPerMeter : ℕ)
    (h1 : totalMeters = 85)
    (h2 : sellingPrice = 8925)
    (h3 : profitPerMeter = 10) :
    costPricePerMeter totalMeters sellingPrice profitPerMeter = 95 := by
  sorry

#eval costPricePerMeter 85 8925 10

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l2526_252619


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2526_252675

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l2526_252675


namespace NUMINAMATH_CALUDE_sixth_salary_l2526_252674

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := salaries.sum
  salaries.length = 5 ∧
  mean * n = total + (n - salaries.length) * (mean * n - total)

theorem sixth_salary :
  ∀ (salaries : List ℝ) (mean : ℝ),
  salary_problem salaries mean →
  (mean * (salaries.length + 1) - salaries.sum) = 2500 :=
by sorry

#check sixth_salary

end NUMINAMATH_CALUDE_sixth_salary_l2526_252674


namespace NUMINAMATH_CALUDE_cos_difference_equals_half_l2526_252667

theorem cos_difference_equals_half : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (66 * π / 180) * Real.cos (54 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_equals_half_l2526_252667


namespace NUMINAMATH_CALUDE_pipe_sale_result_l2526_252623

theorem pipe_sale_result : 
  ∀ (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ),
    price = 1.20 →
    profit_percent = 20 →
    loss_percent = 20 →
    let profit_pipe_cost := price / (1 + profit_percent / 100)
    let loss_pipe_cost := price / (1 - loss_percent / 100)
    let total_cost := profit_pipe_cost + loss_pipe_cost
    let total_revenue := 2 * price
    total_revenue - total_cost = -0.10 := by
  sorry

end NUMINAMATH_CALUDE_pipe_sale_result_l2526_252623


namespace NUMINAMATH_CALUDE_cubic_congruence_solutions_l2526_252681

theorem cubic_congruence_solutions :
  ∀ (a b : ℤ),
    (a^3 ≡ b^3 [ZMOD 121] ↔ (a ≡ b [ZMOD 121] ∨ 11 ∣ a ∧ 11 ∣ b)) ∧
    (a^3 ≡ b^3 [ZMOD 169] ↔ (a ≡ b [ZMOD 169] ∨ a ≡ 22*b [ZMOD 169] ∨ a ≡ 146*b [ZMOD 169] ∨ 13 ∣ a ∧ 13 ∣ b)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_congruence_solutions_l2526_252681


namespace NUMINAMATH_CALUDE_bobby_shoes_count_l2526_252632

theorem bobby_shoes_count (bonny_shoes becky_shoes bobby_shoes : ℕ) : 
  bonny_shoes = 13 →
  bonny_shoes = 2 * becky_shoes - 5 →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 := by
sorry

end NUMINAMATH_CALUDE_bobby_shoes_count_l2526_252632


namespace NUMINAMATH_CALUDE_temperature_at_night_l2526_252645

/-- Given the temperature changes throughout a day, prove the final temperature at night. -/
theorem temperature_at_night 
  (noon_temp : ℤ) 
  (afternoon_temp : ℤ) 
  (temp_drop : ℤ) 
  (h1 : noon_temp = 5)
  (h2 : afternoon_temp = 7)
  (h3 : temp_drop = 9) : 
  afternoon_temp - temp_drop = -2 := by
  sorry

end NUMINAMATH_CALUDE_temperature_at_night_l2526_252645


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l2526_252617

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 14 15) = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l2526_252617


namespace NUMINAMATH_CALUDE_friend_meeting_distance_l2526_252615

theorem friend_meeting_distance (trail_length : ℝ) (rate_difference : ℝ) : 
  trail_length = 36 → rate_difference = 0.25 → 
  let faster_friend_distance : ℝ := trail_length * (1 + rate_difference) / (2 + rate_difference)
  faster_friend_distance = 20 := by
sorry

end NUMINAMATH_CALUDE_friend_meeting_distance_l2526_252615


namespace NUMINAMATH_CALUDE_minimum_groups_l2526_252628

/-- A function that determines if a number belongs to the set G_k -/
def in_G_k (n : ℕ) (k : ℕ) : Prop :=
  n % 6 = k ∧ 1 ≤ n ∧ n ≤ 600

/-- A function that checks if two numbers can be in the same group -/
def can_be_in_same_group (a b : ℕ) : Prop :=
  (a + b) % 6 = 0

/-- A valid grouping of numbers -/
def valid_grouping (groups : List (List ℕ)) : Prop :=
  (∀ group ∈ groups, ∀ a ∈ group, ∀ b ∈ group, a ≠ b → can_be_in_same_group a b) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 600 → ∃ group ∈ groups, n ∈ group)

theorem minimum_groups :
  ∃ (groups : List (List ℕ)), valid_grouping groups ∧
    (∀ (other_groups : List (List ℕ)), valid_grouping other_groups →
      groups.length ≤ other_groups.length) ∧
    groups.length = 202 := by
  sorry

end NUMINAMATH_CALUDE_minimum_groups_l2526_252628


namespace NUMINAMATH_CALUDE_triangle_area_with_given_sides_l2526_252690

theorem triangle_area_with_given_sides :
  let a : ℝ := 65
  let b : ℝ := 60
  let c : ℝ := 25
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 750 := by sorry

end NUMINAMATH_CALUDE_triangle_area_with_given_sides_l2526_252690


namespace NUMINAMATH_CALUDE_equal_perimeters_rectangle_square_l2526_252656

/-- Given two equal lengths of wire, one formed into a rectangle and one formed into a square,
    the perimeters of the resulting shapes are equal. -/
theorem equal_perimeters_rectangle_square (wire_length : ℝ) (h : wire_length > 0) :
  ∃ (rect_width rect_height square_side : ℝ),
    rect_width > 0 ∧ rect_height > 0 ∧ square_side > 0 ∧
    2 * (rect_width + rect_height) = wire_length ∧
    4 * square_side = wire_length ∧
    2 * (rect_width + rect_height) = 4 * square_side :=
by sorry

end NUMINAMATH_CALUDE_equal_perimeters_rectangle_square_l2526_252656


namespace NUMINAMATH_CALUDE_a_2k_minus_1_has_three_prime_factors_l2526_252603

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 6
  | (n + 2) => 4 * a (n + 1) - a n + 2

theorem a_2k_minus_1_has_three_prime_factors (k : ℕ) (h : k > 3) :
  ∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  (p * q * r : ℤ) ∣ a (2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_a_2k_minus_1_has_three_prime_factors_l2526_252603


namespace NUMINAMATH_CALUDE_jason_music_store_spending_l2526_252638

/-- The cost of Jason's flute -/
def flute_cost : ℚ := 142.46

/-- The cost of Jason's music tool -/
def music_tool_cost : ℚ := 8.89

/-- The cost of Jason's song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

theorem jason_music_store_spending :
  total_spent = 158.35 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_spending_l2526_252638


namespace NUMINAMATH_CALUDE_same_type_ab_squared_and_neg_two_ab_squared_l2526_252689

/-- A polynomial type representing terms of the form c * a^m * b^n where c is a constant -/
structure PolynomialTerm (α : Type*) [CommRing α] where
  coeff : α
  a_exp : ℕ
  b_exp : ℕ

/-- The degree of a polynomial term -/
def PolynomialTerm.degree {α : Type*} [CommRing α] (term : PolynomialTerm α) : ℕ :=
  term.a_exp + term.b_exp

/-- Check if two polynomial terms are of the same type -/
def same_type {α : Type*} [CommRing α] (t1 t2 : PolynomialTerm α) : Prop :=
  t1.a_exp = t2.a_exp ∧ t1.b_exp = t2.b_exp

theorem same_type_ab_squared_and_neg_two_ab_squared 
  {α : Type*} [CommRing α] (a b : α) : 
  let ab_squared : PolynomialTerm α := ⟨1, 1, 2⟩
  let neg_two_ab_squared : PolynomialTerm α := ⟨-2, 1, 2⟩
  same_type ab_squared neg_two_ab_squared ∧ 
  ab_squared.degree = 3 :=
by sorry

end NUMINAMATH_CALUDE_same_type_ab_squared_and_neg_two_ab_squared_l2526_252689


namespace NUMINAMATH_CALUDE_optimal_solution_is_valid_and_minimal_l2526_252600

/-- Represents a nail in the painting hanging problem -/
inductive Nail
| a₁ : Nail
| a₂ : Nail
| a₃ : Nail
| a₄ : Nail

/-- Represents a sequence of nails and their inverses -/
inductive NailSequence
| empty : NailSequence
| cons : Nail → NailSequence → NailSequence
| inv : Nail → NailSequence → NailSequence

/-- Counts the number of symbols in a nail sequence -/
def symbolCount : NailSequence → Nat
| NailSequence.empty => 0
| NailSequence.cons _ s => 1 + symbolCount s
| NailSequence.inv _ s => 1 + symbolCount s

/-- Checks if a nail sequence falls when a given nail is removed -/
def fallsWhenRemoved (s : NailSequence) (n : Nail) : Prop := sorry

/-- Represents the optimal solution [[a₁, a₂], [a₃, a₄]] -/
def optimalSolution : NailSequence := sorry

/-- Theorem: The optimal solution is valid and minimal -/
theorem optimal_solution_is_valid_and_minimal :
  (∀ n : Nail, fallsWhenRemoved optimalSolution n) ∧
  (∀ s : NailSequence, (∀ n : Nail, fallsWhenRemoved s n) → symbolCount optimalSolution ≤ symbolCount s) := by
  sorry

end NUMINAMATH_CALUDE_optimal_solution_is_valid_and_minimal_l2526_252600


namespace NUMINAMATH_CALUDE_function_properties_l2526_252666

/-- Given two real numbers p1 and p2, where p1 ≠ p2, we define two functions f and g. -/
theorem function_properties (p1 p2 : ℝ) (h : p1 ≠ p2) :
  let f := fun x : ℝ => (3 : ℝ) ^ (|x - p1|)
  let g := fun x : ℝ => (3 : ℝ) ^ (|x - p2|)
  -- 1. f can be obtained by translating g
  (∃ k : ℝ, ∀ x : ℝ, f x = g (x + k)) ∧
  -- 2. f + g is symmetric about x = (p1 + p2) / 2
  (∀ x : ℝ, f x + g x = f (p1 + p2 - x) + g (p1 + p2 - x)) ∧
  -- 3. f - g is symmetric about the point ((p1 + p2) / 2, 0)
  (∀ x : ℝ, f x - g x = -(f (p1 + p2 - x) - g (p1 + p2 - x))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2526_252666


namespace NUMINAMATH_CALUDE_quadratic_no_solution_b_range_l2526_252602

theorem quadratic_no_solution_b_range (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_solution_b_range_l2526_252602


namespace NUMINAMATH_CALUDE_linda_coloring_books_l2526_252610

/-- Represents Linda's purchase --/
structure Purchase where
  coloringBookPrice : ℝ
  coloringBookCount : ℕ
  peanutPackPrice : ℝ
  peanutPackCount : ℕ
  stuffedAnimalPrice : ℝ
  totalPaid : ℝ

/-- Theorem stating the number of coloring books Linda bought --/
theorem linda_coloring_books (p : Purchase) 
  (h1 : p.coloringBookPrice = 4)
  (h2 : p.peanutPackPrice = 1.5)
  (h3 : p.peanutPackCount = 4)
  (h4 : p.stuffedAnimalPrice = 11)
  (h5 : p.totalPaid = 25)
  (h6 : p.coloringBookPrice * p.coloringBookCount + 
        p.peanutPackPrice * p.peanutPackCount + 
        p.stuffedAnimalPrice = p.totalPaid) :
  p.coloringBookCount = 2 := by
  sorry

end NUMINAMATH_CALUDE_linda_coloring_books_l2526_252610


namespace NUMINAMATH_CALUDE_parabola_translation_l2526_252664

def original_parabola (x : ℝ) : ℝ := x^2 + 1

def transformed_parabola (x : ℝ) : ℝ := x^2 + 4*x + 5

def translation_distance : ℝ := 2

theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola (x + translation_distance) = original_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2526_252664


namespace NUMINAMATH_CALUDE_projection_matrix_values_l2526_252626

/-- A 2x2 matrix is a projection matrix if and only if Q^2 = Q -/
def is_projection_matrix (Q : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  Q * Q = Q

/-- The specific form of our matrix Q -/
def Q (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 20/49; c, 29/49]

theorem projection_matrix_values :
  ∀ a c : ℚ, is_projection_matrix (Q a c) → a = 20/49 ∧ c = 29/49 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l2526_252626


namespace NUMINAMATH_CALUDE_circle_equation_proof_l2526_252630

theorem circle_equation_proof (x y : ℝ) : 
  let center := (1, -2)
  let radius := Real.sqrt 2
  let circle_eq := (x - 1)^2 + (y + 2)^2 = 2
  let center_line_eq := -2 * center.1 = center.2
  let tangent_line_eq := x + y = 1
  let tangent_point := (2, -1)
  (
    center_line_eq ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ circle_eq ∧
    (center.1 - tangent_point.1)^2 + (center.2 - tangent_point.2)^2 = radius^2 ∧
    (tangent_point.1 + tangent_point.2 = 1)
  ) := by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l2526_252630


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2526_252694

theorem fraction_decomposition (x C D : ℚ) : 
  (7 * x - 15) / (3 * x^2 - x - 4) = C / (x - 1) + D / (3 * x + 4) →
  3 * x^2 - x - 4 = (3 * x + 4) * (x - 1) →
  C = -8/7 ∧ D = 73/7 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l2526_252694


namespace NUMINAMATH_CALUDE_calculation_proof_l2526_252678

theorem calculation_proof : (-36) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2526_252678


namespace NUMINAMATH_CALUDE_platform_length_l2526_252620

/-- The length of a train platform given crossing times and lengths -/
theorem platform_length 
  (train_length : ℝ) 
  (first_time : ℝ) 
  (second_time : ℝ) 
  (second_platform : ℝ) 
  (h1 : train_length = 190) 
  (h2 : first_time = 15) 
  (h3 : second_time = 20) 
  (h4 : second_platform = 250) : 
  ∃ (first_platform : ℝ), 
    first_platform = 140 ∧ 
    (train_length + first_platform) / first_time = 
    (train_length + second_platform) / second_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l2526_252620


namespace NUMINAMATH_CALUDE_caitlin_bracelets_l2526_252662

/-- The number of bracelets Caitlin can make -/
def num_bracelets : ℕ := 11

/-- The total number of beads Caitlin has -/
def total_beads : ℕ := 528

/-- The number of large beads per bracelet -/
def large_beads_per_bracelet : ℕ := 12

/-- The ratio of small beads to large beads in each bracelet -/
def small_to_large_ratio : ℕ := 2

theorem caitlin_bracelets :
  (total_beads / 2) / (large_beads_per_bracelet * small_to_large_ratio) = num_bracelets :=
sorry

end NUMINAMATH_CALUDE_caitlin_bracelets_l2526_252662


namespace NUMINAMATH_CALUDE_min_red_chips_is_76_l2526_252665

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies the given conditions -/
def isValidChipCount (c : ChipCount) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 75

/-- The minimum number of red chips that satisfies the conditions -/
def minRedChips : ℕ := 76

/-- Theorem stating that the minimum number of red chips is 76 -/
theorem min_red_chips_is_76 :
  ∀ c : ChipCount, isValidChipCount c → c.red ≥ minRedChips :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_is_76_l2526_252665


namespace NUMINAMATH_CALUDE_f5_computation_l2526_252658

/-- A function that represents a boolean operation (AND or OR) -/
def BoolOp : Type := Bool → Bool → Bool

/-- Compute f₅ using only 5 boolean operations -/
def compute_f5 (x₁ x₂ x₃ x₄ x₅ : Bool) (op₁ op₂ op₃ op₄ op₅ : BoolOp) : Bool :=
  let x₆ := op₁ x₁ x₃
  let x₇ := op₂ x₂ x₆
  let x₈ := op₃ x₃ x₅
  let x₉ := op₄ x₄ x₈
  op₅ x₇ x₉

/-- Theorem: f₅ can be computed using only 5 operations of conjunctions and disjunctions -/
theorem f5_computation (x₁ x₂ x₃ x₄ x₅ : Bool) :
  ∃ (op₁ op₂ op₃ op₄ op₅ : BoolOp),
    (∀ a b, op₁ a b = a ∨ b ∨ op₁ a b = a ∧ b) ∧
    (∀ a b, op₂ a b = a ∨ b ∨ op₂ a b = a ∧ b) ∧
    (∀ a b, op₃ a b = a ∨ b ∨ op₃ a b = a ∧ b) ∧
    (∀ a b, op₄ a b = a ∨ b ∨ op₄ a b = a ∧ b) ∧
    (∀ a b, op₅ a b = a ∨ b ∨ op₅ a b = a ∧ b) :=
by
  sorry


end NUMINAMATH_CALUDE_f5_computation_l2526_252658


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2526_252696

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem statement
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - 1)^2 + (y + 2)^2 = 16) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2526_252696


namespace NUMINAMATH_CALUDE_largest_class_size_l2526_252654

theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) : 
  total_students = 100 → 
  num_classes = 5 → 
  diff = 2 → 
  (∃ x : ℕ, total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff)) → 
  ∃ x : ℕ, x = 24 ∧ total_students = x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2526_252654


namespace NUMINAMATH_CALUDE_count_two_digit_primes_ending_in_3_l2526_252642

def two_digit_primes_ending_in_3 : List Nat := [13, 23, 33, 43, 53, 63, 73, 83, 93]

theorem count_two_digit_primes_ending_in_3 : 
  (two_digit_primes_ending_in_3.filter Nat.Prime).length = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_primes_ending_in_3_l2526_252642


namespace NUMINAMATH_CALUDE_initial_water_amount_l2526_252606

/-- The amount of water initially in the bucket, in gallons. -/
def initial_water : ℝ := sorry

/-- The amount of water remaining in the bucket, in gallons. -/
def remaining_water : ℝ := 0.5

/-- The amount of water that leaked out of the bucket, in gallons. -/
def leaked_water : ℝ := 0.25

/-- Theorem stating that the initial amount of water is equal to 0.75 gallon. -/
theorem initial_water_amount : initial_water = 0.75 := by sorry

end NUMINAMATH_CALUDE_initial_water_amount_l2526_252606


namespace NUMINAMATH_CALUDE_set_intersection_problem_l2526_252691

def A (a : ℝ) : Set ℝ := {3, 4, a^2 - 3*a - 1}
def B (a : ℝ) : Set ℝ := {2*a, -3}

theorem set_intersection_problem (a : ℝ) :
  (A a ∩ B a = {-3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l2526_252691


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l2526_252648

theorem product_of_x_and_y (x y : ℝ) : 
  3 * x + 4 * y = 60 → 6 * x - 4 * y = 12 → x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_x_and_y_l2526_252648


namespace NUMINAMATH_CALUDE_school_attendance_l2526_252636

/-- Calculates the number of years a student attends school given the cost per semester,
    number of semesters per year, and total cost. -/
def years_of_school (cost_per_semester : ℕ) (semesters_per_year : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / (cost_per_semester * semesters_per_year)

/-- Theorem stating that given the specific costs and duration, the student attends 13 years of school. -/
theorem school_attendance : years_of_school 20000 2 520000 = 13 := by
  sorry

end NUMINAMATH_CALUDE_school_attendance_l2526_252636


namespace NUMINAMATH_CALUDE_solve_for_y_l2526_252659

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 3*x + 6 = y - 2) (h2 : x = -5) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2526_252659


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l2526_252693

theorem largest_positive_root_bound (b c : ℝ) (hb : |b| ≤ 3) (hc : |c| ≤ 3) :
  let r := (3 + Real.sqrt 21) / 2
  ∀ x : ℝ, x > 0 → x^2 + b*x + c = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l2526_252693


namespace NUMINAMATH_CALUDE_min_value_of_arithmetic_geometric_seq_l2526_252640

/-- A positive arithmetic-geometric sequence -/
def ArithGeomSeq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ q : ℝ, q > 0 ∧ ∀ k, a (k + 1) = a k * q

theorem min_value_of_arithmetic_geometric_seq
  (a : ℕ → ℝ)
  (h_seq : ArithGeomSeq a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) :
  (∃ m n : ℕ, 1 / m + 4 / n = 3 / 2) ∧
  (∀ m n : ℕ, 1 / m + 4 / n ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_arithmetic_geometric_seq_l2526_252640


namespace NUMINAMATH_CALUDE_value_of_expression_l2526_252622

theorem value_of_expression (x : ℝ) (h : x^2 - 3*x - 12 = 0) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2526_252622


namespace NUMINAMATH_CALUDE_linear_function_intersection_l2526_252671

/-- A linear function y = kx + 2 intersects the x-axis at a point 2 units away from the origin if and only if k = ±1 -/
theorem linear_function_intersection (k : ℝ) : 
  (∃ x : ℝ, k * x + 2 = 0 ∧ |x| = 2) ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_intersection_l2526_252671


namespace NUMINAMATH_CALUDE_perfect_cube_pair_solution_l2526_252618

theorem perfect_cube_pair_solution : ∀ a b : ℕ+,
  (∃ k : ℕ+, (a ^ 3 + 6 * a * b + 1 : ℕ) = k ^ 3) →
  (∃ m : ℕ+, (b ^ 3 + 6 * a * b + 1 : ℕ) = m ^ 3) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_cube_pair_solution_l2526_252618
