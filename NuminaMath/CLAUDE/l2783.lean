import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_from_square_perimeter_l2783_278375

/-- The area of a circle that shares a center with a square of perimeter 40 feet -/
theorem circle_area_from_square_perimeter : ∃ (circle_area : ℝ), 
  circle_area = 50 * Real.pi ∧ 
  ∃ (square_side : ℝ), 
    4 * square_side = 40 ∧
    circle_area = Real.pi * (square_side * Real.sqrt 2 / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_square_perimeter_l2783_278375


namespace NUMINAMATH_CALUDE_square_land_side_length_l2783_278305

/-- Given a square-shaped land plot with an area of 625 square units,
    prove that the length of one side is 25 units. -/
theorem square_land_side_length :
  ∀ (side : ℝ), side > 0 → side * side = 625 → side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l2783_278305


namespace NUMINAMATH_CALUDE_cake_chord_length_squared_l2783_278329

theorem cake_chord_length_squared (d : ℝ) (n : ℕ) (l : ℝ) : 
  d = 18 → n = 4 → l = (d / 2) * Real.sqrt 2 → l^2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_cake_chord_length_squared_l2783_278329


namespace NUMINAMATH_CALUDE_solve_equation_l2783_278390

theorem solve_equation : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2783_278390


namespace NUMINAMATH_CALUDE_smallest_prime_angle_in_inscribed_triangle_l2783_278349

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement -/
theorem smallest_prime_angle_in_inscribed_triangle :
  ∀ q : ℕ,
  q > 0 →
  isPrime q →
  isPrime (2 * q) →
  isPrime (180 - 3 * q) →
  (∀ p : ℕ, p < q → p > 0 → ¬(isPrime p ∧ isPrime (2 * p) ∧ isPrime (180 - 3 * p))) →
  q = 7 := by
  sorry

#check smallest_prime_angle_in_inscribed_triangle

end NUMINAMATH_CALUDE_smallest_prime_angle_in_inscribed_triangle_l2783_278349


namespace NUMINAMATH_CALUDE_circle_area_with_special_condition_l2783_278368

theorem circle_area_with_special_condition (r : ℝ) (h : r > 0) :
  (5 : ℝ) * (1 / (2 * Real.pi * r)) = r / 2 → π * r^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_special_condition_l2783_278368


namespace NUMINAMATH_CALUDE_star_equation_solution_l2783_278380

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem star_equation_solution :
  ∀ x : ℝ, star 3 x = 15 → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l2783_278380


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2783_278398

/-- Given a rectangle where one side is measured 8% in excess and the other side is measured 5% in deficit, 
    the error percentage in the calculated area is 2.6%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L' W' : ℝ) (h1 : L' = 1.08 * L) (h2 : W' = 0.95 * W) :
  (L' * W' - L * W) / (L * W) * 100 = 2.6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l2783_278398


namespace NUMINAMATH_CALUDE_paths_from_A_to_E_l2783_278324

/-- The number of paths between two consecutive points -/
def paths_between_consecutive : ℕ := 2

/-- The number of direct paths from A to E -/
def direct_paths : ℕ := 1

/-- The number of intermediate points between A and E -/
def intermediate_points : ℕ := 4

/-- The total number of paths from A to E -/
def total_paths : ℕ := paths_between_consecutive ^ intermediate_points + direct_paths

theorem paths_from_A_to_E : total_paths = 17 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_E_l2783_278324


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_lines_theorem_l2783_278327

-- Define the line l1
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the parallel line l2
def l2_parallel (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

-- Define the perpendicular lines l2
def l2_perp_pos (x y : ℝ) : Prop := 4 * x - 3 * y + 4 * Real.sqrt 6 = 0
def l2_perp_neg (x y : ℝ) : Prop := 4 * x - 3 * y - 4 * Real.sqrt 6 = 0

-- Theorem for parallel line
theorem parallel_line_theorem :
  (∀ x y, l2_parallel x y ↔ ∃ k, 3 * x + 4 * y = k) ∧
  l2_parallel (-1) 3 := by sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines_theorem :
  (∀ x y, (l2_perp_pos x y ∨ l2_perp_neg x y) → 
    (3 * 4 + 4 * (-3) = 0)) ∧
  (∀ x y, l2_perp_pos x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) ∧
  (∀ x y, l2_perp_neg x y → 
    (1/2 * |x| * |y| = 4 ∧ 4 * x = 0 → y = Real.sqrt 6 ∧ 3 * y = 0 → x = 4/3 * Real.sqrt 6)) := by sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_lines_theorem_l2783_278327


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2783_278341

theorem arithmetic_calculation : 5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2783_278341


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l2783_278308

theorem polynomial_expansion_properties 
  (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) : 
  (a₀ = -1) ∧ (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l2783_278308


namespace NUMINAMATH_CALUDE_derivative_y_wrt_x_l2783_278385

noncomputable section

variable (t : ℝ)

def x : ℝ := Real.arcsin (Real.sin t)
def y : ℝ := Real.arccos (Real.cos t)

theorem derivative_y_wrt_x : 
  deriv (fun x => y x) (x t) = 1 :=
sorry

end NUMINAMATH_CALUDE_derivative_y_wrt_x_l2783_278385


namespace NUMINAMATH_CALUDE_characterization_of_k_l2783_278355

theorem characterization_of_k (k m n : ℕ+) (h : m * (m + k) = n * (n + 1)) :
  k = 1 ∨ k ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_k_l2783_278355


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l2783_278318

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

/-- Given that point A(-2, b) is symmetric to point B(a, 3) with respect to the origin, prove that a - b = 5 -/
theorem symmetric_points_difference (a b : ℝ) 
  (h : symmetric_wrt_origin (-2, b) (a, 3)) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l2783_278318


namespace NUMINAMATH_CALUDE_arabella_first_step_time_l2783_278384

/-- Represents the time spent learning dance steps -/
structure DanceSteps where
  first : ℝ
  second : ℝ
  third : ℝ

/-- The conditions for Arabella's dance step learning -/
def arabella_dance_conditions (steps : DanceSteps) : Prop :=
  steps.second = steps.first / 2 ∧
  steps.third = steps.first + steps.second ∧
  steps.first + steps.second + steps.third = 90

/-- Theorem stating that under the given conditions, the time spent on the first step is 30 minutes -/
theorem arabella_first_step_time (steps : DanceSteps) 
  (h : arabella_dance_conditions steps) : steps.first = 30 := by
  sorry

end NUMINAMATH_CALUDE_arabella_first_step_time_l2783_278384


namespace NUMINAMATH_CALUDE_henry_correct_answers_l2783_278391

/-- Represents a mathematics contest with given scoring rules and a participant's performance. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the number of correct answers given a MathContest instance. -/
def correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest conditions, Henry had 10 correct answers. -/
theorem henry_correct_answers : 
  let contest : MathContest := {
    total_problems := 15,
    correct_points := 6,
    incorrect_points := -3,
    total_score := 45
  }
  correct_answers contest = 10 := by
  sorry

end NUMINAMATH_CALUDE_henry_correct_answers_l2783_278391


namespace NUMINAMATH_CALUDE_inequality_proof_l2783_278331

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (b / a + c / b + a / c) ≥ (1 / 3) * (a + b + c) * (1 / a + 1 / b + 1 / c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2783_278331


namespace NUMINAMATH_CALUDE_particle_position_at_2004_l2783_278306

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Defines the movement pattern of the particle -/
def next_position (p : Position) : Position :=
  if p.x = p.y then Position.mk (p.x + 1) p.y
  else if p.x > p.y then Position.mk p.x (p.y + 1)
  else Position.mk (p.x + 1) p.y

/-- Calculates the position of the particle after n seconds -/
def position_at_time (n : ℕ) : Position :=
  match n with
  | 0 => Position.mk 0 0
  | n + 1 => next_position (position_at_time n)

/-- The main theorem stating the position of the particle after 2004 seconds -/
theorem particle_position_at_2004 :
  position_at_time 2004 = Position.mk 20 44 := by
  sorry


end NUMINAMATH_CALUDE_particle_position_at_2004_l2783_278306


namespace NUMINAMATH_CALUDE_probability_adjacent_ascending_five_cds_l2783_278346

/-- The probability of two specific CDs being adjacent in ascending order when n CDs are randomly arranged -/
def probability_adjacent_ascending (n : ℕ) : ℚ :=
  if n ≥ 2 then (4 * (n - 2).factorial) / n.factorial else 0

/-- Theorem: The probability of CDs 1 and 2 being next to each other in ascending order 
    when 5 CDs are randomly placed in a cassette holder is 1/5 -/
theorem probability_adjacent_ascending_five_cds : 
  probability_adjacent_ascending 5 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_ascending_five_cds_l2783_278346


namespace NUMINAMATH_CALUDE_probability_exactly_once_l2783_278330

theorem probability_exactly_once (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - (1 - p)^3 = 26/27) →
  3 * p * (1 - p)^2 = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_probability_exactly_once_l2783_278330


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2783_278323

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ a*x - b*y + 2 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁^2 + y₁^2 + 2*x₁ - 4*y₁ + 1 = 0 ∧ 
                         x₂^2 + y₂^2 + 2*x₂ - 4*y₂ + 1 = 0 ∧
                         a*x₁ - b*y₁ + 2 = 0 ∧ a*x₂ - b*y₂ + 2 = 0 ∧
                         (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16) →
  (∀ c d : ℝ, c > 0 → d > 0 → 
    (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 1 = 0 ∧ c*x - d*y + 2 = 0) →
    1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 3/2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2783_278323


namespace NUMINAMATH_CALUDE_intersection_distance_to_pole_l2783_278351

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ

-- Define the distance function in polar coordinates
def distance_to_pole (p : PolarCoordinate) : ℝ := p.1

-- Define the curves
def curve1 (ρ θ : ℝ) : Prop := ρ = 2 * θ + 1
def curve2 (ρ θ : ℝ) : Prop := ρ * θ = 1

theorem intersection_distance_to_pole :
  ∀ (p : PolarCoordinate),
    p.1 > 0 →
    curve1 p.1 p.2 →
    curve2 p.1 p.2 →
    distance_to_pole p = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_to_pole_l2783_278351


namespace NUMINAMATH_CALUDE_parabola_intersection_l2783_278307

theorem parabola_intersection (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x, a * (x - b) * (x - 1) = 0 → x = 3 ∨ x = 1) ∧
  a * (3 - b) * (3 - 1) = 0 →
  ∃ x, x ≠ 3 ∧ a * (x - b) * (x - 1) = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2783_278307


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2783_278319

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    left focus F₁, right focus F₂, and a point P on the hyperbola,
    if PF₂ is perpendicular to the x-axis, |F₁F₂| = 12, and |PF₂| = 5,
    then the eccentricity of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity 
  (a b : ℝ) (F₁ F₂ P : ℝ × ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_foci : F₁.1 < F₂.1 ∧ F₁.2 = 0 ∧ F₂.2 = 0)
  (h_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1)
  (h_perpendicular : P.1 = F₂.1)
  (h_distance_foci : Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2) = 12)
  (h_distance_PF₂ : Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 5) :
  (Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) / (2 * a) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2783_278319


namespace NUMINAMATH_CALUDE_intersection_line_circle_l2783_278348

/-- Given a line y = kx + 3 intersecting the circle (x-1)^2 + (y-2)^2 = 4 at points M and N,
    if |MN| ≥ 2√3, then k ≤ 0. -/
theorem intersection_line_circle (k : ℝ) (M N : ℝ × ℝ) : 
  (∀ x y, y = k * x + 3 → (x - 1)^2 + (y - 2)^2 = 4) →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12 →
  k ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l2783_278348


namespace NUMINAMATH_CALUDE_stratified_sampling_l2783_278334

theorem stratified_sampling (total_male : ℕ) (total_female : ℕ) (selected_male : ℕ) :
  total_male = 56 →
  total_female = 42 →
  selected_male = 8 →
  (total_male : ℚ) / total_female = 4 / 3 →
  ∃ selected_female : ℕ, 
    (selected_female : ℚ) / selected_male = total_female / total_male ∧
    selected_female = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2783_278334


namespace NUMINAMATH_CALUDE_buses_in_parking_lot_l2783_278387

theorem buses_in_parking_lot (initial_buses additional_buses : ℕ) : 
  initial_buses = 7 → additional_buses = 6 → initial_buses + additional_buses = 13 :=
by sorry

end NUMINAMATH_CALUDE_buses_in_parking_lot_l2783_278387


namespace NUMINAMATH_CALUDE_no_simultaneous_greater_value_l2783_278340

theorem no_simultaneous_greater_value : ¬∃ x : ℝ, (x + 3) / 5 > 2 * x + 3 ∧ (x + 3) / 5 > 1 - x := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_greater_value_l2783_278340


namespace NUMINAMATH_CALUDE_sum_not_prime_l2783_278372

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a + b + c + d = x * y :=
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2783_278372


namespace NUMINAMATH_CALUDE_cube_equation_solution_l2783_278301

theorem cube_equation_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 25 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l2783_278301


namespace NUMINAMATH_CALUDE_trapezoid_larger_base_length_l2783_278373

/-- A trapezoid with a midline of length 10 and a diagonal that divides the midline
    into two parts with a difference of 3 has a larger base of length 13. -/
theorem trapezoid_larger_base_length (x y : ℝ) 
  (h1 : (x + y) / 2 = 10)  -- midline length is 10
  (h2 : x - y = 6)         -- difference between parts of divided midline is 3 * 2
  : x = 13 := by  -- x represents the larger base
  sorry

end NUMINAMATH_CALUDE_trapezoid_larger_base_length_l2783_278373


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l2783_278378

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality : 
  7^2 - 16*7 + 63 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 ∧ 
  (∃ ε > 0, (7 - ε)^2 - 16*(7 - ε) + 63 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_seven_satisfies_inequality_seven_is_smallest_l2783_278378


namespace NUMINAMATH_CALUDE_intersection_distance_l2783_278354

/-- The distance between the intersection points of a line and a circle --/
theorem intersection_distance (x y : ℝ) : 
  (x - y + 1 = 0) → -- Line equation
  (x^2 + (y-2)^2 = 4) → -- Circle equation
  ∃ A B : ℝ × ℝ, -- Two intersection points
    A ≠ B ∧
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + (A.2-2)^2 = 4) ∧ -- A satisfies both equations
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + (B.2-2)^2 = 4) ∧ -- B satisfies both equations
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14 -- Distance between A and B is √14
  := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l2783_278354


namespace NUMINAMATH_CALUDE_matts_baseball_cards_value_l2783_278393

theorem matts_baseball_cards_value (n : ℕ) (x : ℚ) : 
  n = 8 →  -- Matt has 8 baseball cards
  2 * x + 3 = (3 * 2 + 9) →  -- He trades 2 cards for 3 $2 cards and 1 $9 card, making a $3 profit
  x = 6 :=  -- Each of Matt's baseball cards is worth $6
by
  sorry

end NUMINAMATH_CALUDE_matts_baseball_cards_value_l2783_278393


namespace NUMINAMATH_CALUDE_min_groups_for_photography_class_l2783_278335

theorem min_groups_for_photography_class (total_students : ℕ) (max_group_size : ℕ) 
  (h1 : total_students = 30) (h2 : max_group_size = 6) : 
  Nat.ceil (total_students / max_group_size) = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_groups_for_photography_class_l2783_278335


namespace NUMINAMATH_CALUDE_nine_team_league_games_l2783_278352

/-- The total number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Theorem: In a 9-team league where each team plays 3 games with every other team, 
    the total number of games played is 108 -/
theorem nine_team_league_games :
  total_games 9 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_nine_team_league_games_l2783_278352


namespace NUMINAMATH_CALUDE_fraction_simplification_l2783_278314

theorem fraction_simplification 
  (a b x y : ℝ) : 
  (3*b*x*(a^3*x^3 + 3*a^2*y^2 + 2*b^2*y^2) + 2*a*y*(2*a^2*x^2 + 3*b^2*x^2 + b^3*y^3)) / (3*b*x + 2*a*y) 
  = a^3*x^3 + 3*a^2*x*y + 2*b^2*y^2 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2783_278314


namespace NUMINAMATH_CALUDE_average_service_hours_l2783_278313

theorem average_service_hours (n : ℕ) (h1 h2 h3 : ℕ) (s1 s2 s3 : ℕ) :
  n = 10 →
  h1 = 15 →
  h2 = 16 →
  h3 = 20 →
  s1 = 2 →
  s2 = 5 →
  s3 = 3 →
  s1 + s2 + s3 = n →
  (h1 * s1 + h2 * s2 + h3 * s3) / n = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_average_service_hours_l2783_278313


namespace NUMINAMATH_CALUDE_farm_animals_l2783_278362

theorem farm_animals (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →
  (initial_horses - 15) - (initial_cows + 15) = 30 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l2783_278362


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l2783_278326

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l2783_278326


namespace NUMINAMATH_CALUDE_principal_calculation_l2783_278394

/-- Proves that given specific conditions, the principal amount is 1600 --/
theorem principal_calculation (rate : ℚ) (time : ℚ) (amount : ℚ) :
  rate = 5 / 100 →
  time = 12 / 5 →
  amount = 1792 →
  amount = (1600 : ℚ) * (1 + rate * time) :=
by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l2783_278394


namespace NUMINAMATH_CALUDE_max_f_sum_l2783_278321

/-- A permutation of 4n letters consisting of n occurrences each of A, B, C, and D -/
def Permutation (n : ℕ) := Fin (4 * n) → Fin 4

/-- The number of B's to the right of each A in the permutation -/
def f_AB (σ : Permutation n) : ℕ := sorry

/-- The number of C's to the right of each B in the permutation -/
def f_BC (σ : Permutation n) : ℕ := sorry

/-- The number of D's to the right of each C in the permutation -/
def f_CD (σ : Permutation n) : ℕ := sorry

/-- The number of A's to the right of each D in the permutation -/
def f_DA (σ : Permutation n) : ℕ := sorry

/-- The sum of f_AB, f_BC, f_CD, and f_DA for a given permutation -/
def f_sum (σ : Permutation n) : ℕ := f_AB σ + f_BC σ + f_CD σ + f_DA σ

theorem max_f_sum (n : ℕ) : (∀ σ : Permutation n, f_sum σ ≤ 3 * n^2) ∧ (∃ σ : Permutation n, f_sum σ = 3 * n^2) := by sorry

end NUMINAMATH_CALUDE_max_f_sum_l2783_278321


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l2783_278396

theorem parallelogram_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 108) 
  (h2 : height = 9) :
  area / height = 12 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l2783_278396


namespace NUMINAMATH_CALUDE_smallest_w_l2783_278345

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 → 
  is_factor (2^5) (936 * w) → 
  is_factor (3^3) (936 * w) → 
  is_factor (14^2) (936 * w) → 
  w ≥ 1764 :=
sorry

end NUMINAMATH_CALUDE_smallest_w_l2783_278345


namespace NUMINAMATH_CALUDE_houses_with_neither_feature_l2783_278386

theorem houses_with_neither_feature (total : ℕ) (garage : ℕ) (pool : ℕ) (both : ℕ) :
  total = 90 →
  garage = 50 →
  pool = 40 →
  both = 35 →
  total - (garage + pool - both) = 35 := by
sorry

end NUMINAMATH_CALUDE_houses_with_neither_feature_l2783_278386


namespace NUMINAMATH_CALUDE_star_equation_equiv_two_distinct_real_roots_l2783_278316

/-- The star operation defined as m ☆ n = mn² - mn - 1 -/
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

/-- The equation 1 ☆ x = 0 is equivalent to x² - x - 1 = 0 -/
theorem star_equation_equiv (x : ℝ) : star 1 x = 0 ↔ x^2 - x - 1 = 0 := by sorry

/-- The equation x² - x - 1 = 0 has two distinct real roots -/
theorem two_distinct_real_roots :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ r₁^2 - r₁ - 1 = 0 ∧ r₂^2 - r₂ - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_star_equation_equiv_two_distinct_real_roots_l2783_278316


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2783_278333

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2783_278333


namespace NUMINAMATH_CALUDE_girls_in_school_l2783_278365

/-- Proves that in a school with 1600 students, if a stratified sample of 200 students
    contains 10 fewer girls than boys, then the total number of girls in the school is 760. -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_in_sample : ℕ) :
  total_students = 1600 →
  sample_size = 200 →
  girls_in_sample = sample_size / 2 - 5 →
  (girls_in_sample : ℚ) / (total_students : ℚ) = (sample_size : ℚ) / (total_students : ℚ) →
  girls_in_sample * (total_students / sample_size) = 760 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l2783_278365


namespace NUMINAMATH_CALUDE_lcm_of_12_and_15_l2783_278320

theorem lcm_of_12_and_15 : 
  let a := 12
  let b := 15
  let hcf := 3
  let lcm := Nat.lcm a b
  lcm = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_and_15_l2783_278320


namespace NUMINAMATH_CALUDE_product_of_integers_l2783_278342

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 22)
  (diff_squares_eq : x^2 - y^2 = 44) :
  x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2783_278342


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l2783_278312

theorem abs_neg_five_eq_five : |(-5 : ℤ)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l2783_278312


namespace NUMINAMATH_CALUDE_function_minimum_at_one_l2783_278392

/-- The function f(x) = (x^2 + 1) / (x + a) has a minimum at x = 1 -/
theorem function_minimum_at_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => (x^2 + 1) / (x + a)
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), x ≠ -a → f x ≥ f 1 :=
by
  sorry

end NUMINAMATH_CALUDE_function_minimum_at_one_l2783_278392


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l2783_278371

theorem smallest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l2783_278371


namespace NUMINAMATH_CALUDE_total_lives_l2783_278376

theorem total_lives (num_friends : ℕ) (lives_per_friend : ℕ) 
  (h1 : num_friends = 8) (h2 : lives_per_friend = 8) : 
  num_friends * lives_per_friend = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_l2783_278376


namespace NUMINAMATH_CALUDE_square_tiles_count_l2783_278381

/-- Represents the number of edges for each tile type -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- Triangle
| 1 => 4  -- Square
| 2 => 5  -- Rectangle

/-- Proves that given the conditions, the number of square tiles is 10 -/
theorem square_tiles_count 
  (total_tiles : ℕ) 
  (total_edges : ℕ) 
  (h_total_tiles : total_tiles = 32)
  (h_total_edges : total_edges = 114) :
  ∃ (triangles squares rectangles : ℕ),
    triangles + squares + rectangles = total_tiles ∧
    3 * triangles + 4 * squares + 5 * rectangles = total_edges ∧
    squares = 10 :=
by sorry

end NUMINAMATH_CALUDE_square_tiles_count_l2783_278381


namespace NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_l2783_278339

theorem triangle_angle_arithmetic_sequence (A B C : ℝ) (AC BC : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Given side lengths
  AC = Real.sqrt 6 →
  BC = 2 →
  -- A is positive and less than π/3
  0 < A →
  A < Real.pi / 3 →
  -- Conclusion: A equals π/4 (45°)
  A = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_l2783_278339


namespace NUMINAMATH_CALUDE_car_cost_l2783_278347

/-- The cost of a car given an initial payment and monthly installments -/
theorem car_cost (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment + num_installments * installment_amount = 18000 :=
by
  sorry

#check car_cost 3000 6 2500

end NUMINAMATH_CALUDE_car_cost_l2783_278347


namespace NUMINAMATH_CALUDE_wendys_bouquets_l2783_278374

/-- Calculates the number of bouquets that can be made given the initial number of flowers,
    number of wilted flowers, and number of flowers per bouquet. -/
def calculateBouquets (initialFlowers : ℕ) (wiltedFlowers : ℕ) (flowersPerBouquet : ℕ) : ℕ :=
  (initialFlowers - wiltedFlowers) / flowersPerBouquet

/-- Proves that Wendy can make 2 bouquets with the given conditions. -/
theorem wendys_bouquets :
  calculateBouquets 45 35 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendys_bouquets_l2783_278374


namespace NUMINAMATH_CALUDE_tree_planting_temperature_reduction_l2783_278311

theorem tree_planting_temperature_reduction 
  (initial_temp : ℝ) 
  (cost_per_tree : ℝ) 
  (temp_reduction_per_tree : ℝ) 
  (total_cost : ℝ) 
  (h1 : initial_temp = 80)
  (h2 : cost_per_tree = 6)
  (h3 : temp_reduction_per_tree = 0.1)
  (h4 : total_cost = 108) :
  initial_temp - (total_cost / cost_per_tree * temp_reduction_per_tree) = 78.2 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_temperature_reduction_l2783_278311


namespace NUMINAMATH_CALUDE_triangle_vector_properties_l2783_278344

/-- Given a triangle ABC with internal angles A, B, C, this theorem proves
    properties related to vectors m and n, and the side lengths of the triangle. -/
theorem triangle_vector_properties (A B C : Real) (m n : Real × Real) :
  let m : Real × Real := (2 * Real.sqrt 3, 1)
  let n : Real × Real := (Real.cos (A / 2) ^ 2, Real.sin A)
  C = 2 * Real.pi / 3 →
  ‖(1, 0) - (Real.cos A, Real.sin A)‖ = 3 →
  (A = Real.pi / 2 → ‖n‖ = Real.sqrt 5 / 2) ∧
  (∀ θ, m.1 * (Real.cos (θ / 2) ^ 2) + m.2 * Real.sin θ ≤ m.1 * (Real.cos (Real.pi / 12) ^ 2) + m.2 * Real.sin (Real.pi / 6)) ∧
  (‖(Real.cos (Real.pi / 6), Real.sin (Real.pi / 6)) - (Real.cos (5 * Real.pi / 6), Real.sin (5 * Real.pi / 6))‖ = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_vector_properties_l2783_278344


namespace NUMINAMATH_CALUDE_franks_final_score_l2783_278328

/-- Calculates the final score in a trivia competition given the number of correct and incorrect answers in each half. -/
def final_score (first_half_correct first_half_incorrect second_half_correct second_half_incorrect : ℕ) : ℤ :=
  let points_per_correct : ℤ := 3
  let points_per_incorrect : ℤ := -1
  (first_half_correct * points_per_correct + first_half_incorrect * points_per_incorrect) +
  (second_half_correct * points_per_correct + second_half_incorrect * points_per_incorrect)

/-- Theorem stating that Frank's final score in the trivia competition is 39 points. -/
theorem franks_final_score :
  final_score 6 4 10 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_franks_final_score_l2783_278328


namespace NUMINAMATH_CALUDE_initial_average_weight_l2783_278370

theorem initial_average_weight 
  (initial_count : ℕ) 
  (new_student_weight : ℝ) 
  (new_average : ℝ) : 
  initial_count = 29 →
  new_student_weight = 10 →
  new_average = 27.4 →
  ∃ (initial_average : ℝ),
    initial_average * initial_count + new_student_weight = 
    new_average * (initial_count + 1) ∧
    initial_average = 28 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l2783_278370


namespace NUMINAMATH_CALUDE_special_function_value_l2783_278336

/-- A monotonic function on (0, +∞) satisfying f(f(x) - 1/x) = 2 for all x > 0 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 < x ∧ 0 < y ∧ x < y → f x < f y) ∧ 
  (∀ x, 0 < x → f (f x - 1/x) = 2)

/-- Theorem stating that for a special function f, f(1/5) = 6 -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f (1/5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l2783_278336


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2783_278317

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (2 - 5 * x) = 5 → x = -4.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2783_278317


namespace NUMINAMATH_CALUDE_complex_cube_real_iff_l2783_278338

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_cube_real_iff (z : ℂ) : 
  is_real (z^3) ↔ z.im = 0 ∨ z.im = Real.sqrt 3 * z.re ∨ z.im = -Real.sqrt 3 * z.re :=
sorry

end NUMINAMATH_CALUDE_complex_cube_real_iff_l2783_278338


namespace NUMINAMATH_CALUDE_jerry_current_average_l2783_278366

/-- Jerry's current average score on the first 3 tests -/
def current_average : ℝ := sorry

/-- Jerry's score on the fourth test -/
def fourth_test_score : ℝ := 93

/-- The increase in average score after the fourth test -/
def average_increase : ℝ := 2

theorem jerry_current_average : 
  (current_average * 3 + fourth_test_score) / 4 = current_average + average_increase ∧ 
  current_average = 85 := by sorry

end NUMINAMATH_CALUDE_jerry_current_average_l2783_278366


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2783_278302

/-- The number of arrangements of n distinct objects taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of students -/
def num_students : ℕ := 8

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of gaps between and around students -/
def num_gaps : ℕ := num_students + 1

theorem teachers_not_adjacent_arrangements :
  (A num_students num_students) * (A num_gaps num_teachers) =
  (A num_students num_students) * (A 9 2) := by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2783_278302


namespace NUMINAMATH_CALUDE_total_heads_is_48_l2783_278332

/-- Represents the number of feet an animal has -/
def feet_count (animal : String) : ℕ :=
  if animal = "hen" then 2 else 4

/-- The total number of animals -/
def total_animals (hens cows : ℕ) : ℕ := hens + cows

/-- The total number of feet -/
def total_feet (hens cows : ℕ) : ℕ := feet_count "hen" * hens + feet_count "cow" * cows

/-- Theorem stating that the total number of heads is 48 -/
theorem total_heads_is_48 (hens cows : ℕ) 
  (h1 : total_feet hens cows = 140) 
  (h2 : hens = 26) : 
  total_animals hens cows = 48 := by sorry

end NUMINAMATH_CALUDE_total_heads_is_48_l2783_278332


namespace NUMINAMATH_CALUDE_three_from_seven_combination_l2783_278337

theorem three_from_seven_combination : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_three_from_seven_combination_l2783_278337


namespace NUMINAMATH_CALUDE_soda_survey_l2783_278322

/-- Given a survey of 600 people and a central angle of 108° for the "Soda" sector,
    prove that the number of people who chose "Soda" is 180. -/
theorem soda_survey (total_people : ℕ) (soda_angle : ℕ) :
  total_people = 600 →
  soda_angle = 108 →
  (total_people * soda_angle) / 360 = 180 := by
  sorry

end NUMINAMATH_CALUDE_soda_survey_l2783_278322


namespace NUMINAMATH_CALUDE_ferry_passengers_with_hats_l2783_278359

theorem ferry_passengers_with_hats (total_passengers : ℕ) 
  (percent_men : ℚ) (percent_women_with_hats : ℚ) (percent_men_with_hats : ℚ) :
  total_passengers = 1500 →
  percent_men = 2/5 →
  percent_women_with_hats = 3/20 →
  percent_men_with_hats = 3/25 →
  ∃ (total_with_hats : ℕ), total_with_hats = 207 :=
by
  sorry

end NUMINAMATH_CALUDE_ferry_passengers_with_hats_l2783_278359


namespace NUMINAMATH_CALUDE_tax_increase_proof_l2783_278300

theorem tax_increase_proof (item_cost : ℝ) (old_rate new_rate : ℝ) 
  (h1 : item_cost = 1000)
  (h2 : old_rate = 0.07)
  (h3 : new_rate = 0.075) :
  new_rate * item_cost - old_rate * item_cost = 5 :=
by sorry

end NUMINAMATH_CALUDE_tax_increase_proof_l2783_278300


namespace NUMINAMATH_CALUDE_iteration_convergence_l2783_278379

theorem iteration_convergence (a b : ℝ) (h : a > b) :
  ∃ k : ℕ, (2 : ℝ)^(-k : ℤ) * (a - b) < 1 / 2002 := by
  sorry

end NUMINAMATH_CALUDE_iteration_convergence_l2783_278379


namespace NUMINAMATH_CALUDE_polynomial_solution_l2783_278358

theorem polynomial_solution (a b c : ℤ) : 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  (∀ X : ℤ, a^3 + a*a*X + b*X + c = a^3) ∧
  (∀ X : ℤ, b^3 + a*b*X + b*X + c = b^3) →
  a = 1 ∧ b = -1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_solution_l2783_278358


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2783_278356

theorem geometric_series_ratio (a r : ℝ) (hr : 0 < r) (hr1 : r < 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2783_278356


namespace NUMINAMATH_CALUDE_clock_strikes_ten_l2783_278397

/-- A clock that strikes at regular intervals -/
structure StrikingClock where
  /-- The time it takes to complete a given number of strikes -/
  strike_time : ℕ → ℝ
  /-- The number of strikes at a given hour -/
  strikes_at_hour : ℕ → ℕ

/-- Our specific clock that takes 7 seconds to strike 7 times at 7 o'clock -/
def our_clock : StrikingClock where
  strike_time := fun n => if n = 7 then 7 else 0  -- We only know about 7 strikes
  strikes_at_hour := fun h => if h = 7 then 7 else 0  -- We only know about 7 o'clock

/-- The theorem stating that our clock takes 10.5 seconds to strike 10 times -/
theorem clock_strikes_ten (c : StrikingClock) (h : c.strike_time 7 = 7) :
  c.strike_time 10 = 10.5 := by
  sorry

#check clock_strikes_ten our_clock (by rfl)

end NUMINAMATH_CALUDE_clock_strikes_ten_l2783_278397


namespace NUMINAMATH_CALUDE_count_random_events_l2783_278399

-- Define the set of events
inductive Event
| DiceRoll
| Rain
| Lottery
| SumGreaterThanTwo
| WaterBoiling

-- Define a function to check if an event is random
def isRandomEvent : Event → Bool
| Event.DiceRoll => true
| Event.Rain => true
| Event.Lottery => true
| Event.SumGreaterThanTwo => false
| Event.WaterBoiling => false

-- Theorem: The number of random events is 3
theorem count_random_events :
  (List.filter isRandomEvent [Event.DiceRoll, Event.Rain, Event.Lottery, Event.SumGreaterThanTwo, Event.WaterBoiling]).length = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_random_events_l2783_278399


namespace NUMINAMATH_CALUDE_triangles_in_regular_decagon_l2783_278343

def regular_decagon_vertices : ℕ := 10

theorem triangles_in_regular_decagon :
  (regular_decagon_vertices.choose 3) = 120 :=
by sorry

end NUMINAMATH_CALUDE_triangles_in_regular_decagon_l2783_278343


namespace NUMINAMATH_CALUDE_sum_divisible_by_1987_l2783_278303

def odd_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ := (List.range 993).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem sum_divisible_by_1987 : 
  ∃ k : ℤ, (odd_product : ℤ) + (even_product : ℤ) = 1987 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_1987_l2783_278303


namespace NUMINAMATH_CALUDE_arwen_tulips_l2783_278350

/-- Proves that Arwen picked 20 tulips, given the conditions of the problem -/
theorem arwen_tulips : 
  ∀ (a e : ℕ), 
    e = 2 * a →  -- Elrond picked twice as many tulips as Arwen
    a + e = 60 →  -- They picked 60 tulips in total
    a = 20  -- Arwen picked 20 tulips
    := by sorry

end NUMINAMATH_CALUDE_arwen_tulips_l2783_278350


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2783_278325

theorem rectangular_to_polar_conversion :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 3 * Real.sqrt 2 ∧ θ = π / 4 ∧
  r * Real.cos θ = 3 ∧ r * Real.sin θ = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l2783_278325


namespace NUMINAMATH_CALUDE_quadratic_function_positive_range_l2783_278382

theorem quadratic_function_positive_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 3 → a * x^2 - 2 * a * x + 3 > 0) ↔ 
  (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_positive_range_l2783_278382


namespace NUMINAMATH_CALUDE_mrs_petersons_tumblers_l2783_278363

/-- The number of tumblers bought given the price per tumbler, 
    the amount paid, and the change received. -/
def number_of_tumblers (price_per_tumbler : ℕ) (amount_paid : ℕ) (change : ℕ) : ℕ :=
  (amount_paid - change) / price_per_tumbler

/-- Theorem stating that Mrs. Petersons bought 10 tumblers -/
theorem mrs_petersons_tumblers : 
  number_of_tumblers 45 500 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mrs_petersons_tumblers_l2783_278363


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2783_278367

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).re = 0 ∧ 
  (((m^2 - m - 2) : ℂ) + (m + 1) * Complex.I).im ≠ 0 → 
  m = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2783_278367


namespace NUMINAMATH_CALUDE_elizabeths_husband_weight_l2783_278357

/-- Represents a married couple -/
structure Couple where
  husband_weight : ℝ
  wife_weight : ℝ

/-- The problem setup -/
def cannibal_problem (couples : Fin 3 → Couple) : Prop :=
  let wives_weights := (couples 0).wife_weight + (couples 1).wife_weight + (couples 2).wife_weight
  let total_weight := (couples 0).husband_weight + (couples 0).wife_weight +
                      (couples 1).husband_weight + (couples 1).wife_weight +
                      (couples 2).husband_weight + (couples 2).wife_weight
  ∃ (leon victor maurice : Fin 3),
    leon ≠ victor ∧ leon ≠ maurice ∧ victor ≠ maurice ∧
    wives_weights = 171 ∧
    ¬ ∃ n : ℤ, total_weight = n ∧
    (couples leon).husband_weight = (couples leon).wife_weight ∧
    (couples victor).husband_weight = 1.5 * (couples victor).wife_weight ∧
    (couples maurice).husband_weight = 2 * (couples maurice).wife_weight ∧
    (couples 0).wife_weight = (couples 1).wife_weight + 10 ∧
    (couples 1).wife_weight = (couples 2).wife_weight - 5 ∧
    (couples victor).husband_weight = 85.5

/-- The main theorem to prove -/
theorem elizabeths_husband_weight (couples : Fin 3 → Couple) :
  cannibal_problem couples → ∃ i : Fin 3, (couples i).husband_weight = 85.5 :=
sorry

end NUMINAMATH_CALUDE_elizabeths_husband_weight_l2783_278357


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2783_278315

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₁ * r^(n - 1)

theorem fifth_term_of_geometric_sequence
  (a₁ a₂ : ℚ)
  (h₁ : a₁ = 2)
  (h₂ : a₂ = 1/4)
  (h₃ : a₂ = a₁ * (a₂ / a₁)) :
  geometric_sequence a₁ (a₂ / a₁) 5 = 1/2048 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l2783_278315


namespace NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l2783_278360

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem min_colors_for_distribution_centers : 
  (∃ (n : ℕ), n ≥ 6 ∧ combinations n 3 ≥ 20) ∧
  (∀ (m : ℕ), m < 6 → combinations m 3 < 20) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_for_distribution_centers_l2783_278360


namespace NUMINAMATH_CALUDE_sqrt_of_point_zero_nine_equals_point_three_l2783_278377

theorem sqrt_of_point_zero_nine_equals_point_three : 
  Real.sqrt 0.09 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_point_zero_nine_equals_point_three_l2783_278377


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2783_278364

theorem complex_number_simplification :
  (6 - 3*Complex.I) + 3*(2 - 7*Complex.I) = 12 - 24*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2783_278364


namespace NUMINAMATH_CALUDE_wait_hare_is_random_l2783_278388

-- Define the type for events
inductive Event
| StrongYouth
| ScoopMoon
| WaitHare
| GreenMountains

-- Define what it means for an event to be random
def isRandom (e : Event) : Prop :=
  match e with
  | Event.WaitHare => True
  | _ => False

-- Theorem statement
theorem wait_hare_is_random :
  isRandom Event.WaitHare :=
sorry

end NUMINAMATH_CALUDE_wait_hare_is_random_l2783_278388


namespace NUMINAMATH_CALUDE_temperature_conversion_l2783_278353

theorem temperature_conversion (t k some_number : ℝ) :
  t = 5 / 9 * (k - some_number) →
  t = 105 →
  k = 221 →
  some_number = 32 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2783_278353


namespace NUMINAMATH_CALUDE_clock_solution_l2783_278304

/-- The time in minutes after 9:00 when the minute hand will be exactly opposite
    the place where the hour hand was two minutes ago, five minutes from now. -/
def clock_problem : ℝ → Prop := λ t =>
  0 < t ∧ t < 60 ∧  -- Time is between 9:00 and 10:00
  abs (6 * (t + 5) - (270 + 0.5 * (t - 2))) = 180  -- Opposite hands condition

theorem clock_solution : ∃ t, clock_problem t ∧ t = 10.75 := by
  sorry

end NUMINAMATH_CALUDE_clock_solution_l2783_278304


namespace NUMINAMATH_CALUDE_min_value_xyz_one_min_value_achievable_l2783_278395

theorem min_value_xyz_one (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 ≥ 243 / Real.rpow 4 (1/9) :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y * z = 1 ∧
  3 * x^2 + 12 * x * y + 9 * y^2 + 15 * y * z + 3 * z^2 = 243 / Real.rpow 4 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_one_min_value_achievable_l2783_278395


namespace NUMINAMATH_CALUDE_annuity_duration_exists_l2783_278389

/-- The duration of the original annuity in years -/
def original_duration : ℝ := 20

/-- The interest rate as a decimal -/
def interest_rate : ℝ := 0.04

/-- The equation that the new duration must satisfy -/
def annuity_equation (x : ℝ) : Prop :=
  Real.exp x = 2 * Real.exp original_duration / (Real.exp original_duration + 1)

/-- Theorem stating the existence of a solution to the annuity equation -/
theorem annuity_duration_exists :
  ∃ x : ℝ, annuity_equation x :=
sorry

end NUMINAMATH_CALUDE_annuity_duration_exists_l2783_278389


namespace NUMINAMATH_CALUDE_exists_positive_value_for_expression_l2783_278383

theorem exists_positive_value_for_expression : ∃ n : ℕ+, n.val^2 - 8*n.val + 7 > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_value_for_expression_l2783_278383


namespace NUMINAMATH_CALUDE_raccoons_pepper_sprayed_l2783_278361

theorem raccoons_pepper_sprayed (num_raccoons : ℕ) (num_squirrels : ℕ) : 
  num_squirrels = 6 * num_raccoons →
  num_raccoons + num_squirrels = 84 →
  num_raccoons = 12 := by
sorry

end NUMINAMATH_CALUDE_raccoons_pepper_sprayed_l2783_278361


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2783_278310

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2783_278310


namespace NUMINAMATH_CALUDE_value_of_c_l2783_278309

theorem value_of_c (a c : ℝ) (h1 : 3 * a + 2 = 2) (h2 : c - a = 3) : c = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l2783_278309


namespace NUMINAMATH_CALUDE_trapezium_area_and_shorter_side_l2783_278369

theorem trapezium_area_and_shorter_side (a b h : ℝ) :
  a = 24 ∧ b = 18 ∧ h = 15 →
  (1/2 : ℝ) * (a + b) * h = 315 ∧ min a b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_and_shorter_side_l2783_278369
