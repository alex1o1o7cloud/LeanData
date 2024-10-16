import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2620_262060

/-- A quadratic equation with parameter k -/
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 + 6 * x + k^2 - k

theorem quadratic_root_zero (k : ℝ) :
  (quadratic_equation k 0 = 0) ∧ (k - 1 ≠ 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2620_262060


namespace NUMINAMATH_CALUDE_determinant_of_roots_l2620_262046

theorem determinant_of_roots (s p q : ℝ) (a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 → 
  b^3 + s*b^2 + p*b + q = 0 → 
  c^3 + s*c^2 + p*c + q = 0 → 
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -s*(s^2 - 3*p) := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l2620_262046


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l2620_262073

theorem complex_expression_evaluation :
  let a : ℝ := 3.67
  let b : ℝ := 4.83
  let c : ℝ := 2.57
  let d : ℝ := -0.12
  let x : ℝ := 7.25
  let y : ℝ := -0.55
  
  let expression : ℝ := (3*a * (4*b - 2*y)^2) / (5*c * d^3 * 0.5*x) - (2*x * y^3) / (a * b^2 * c)
  
  ∃ ε > 0, |expression - (-57.179729)| < ε ∧ ε < 0.000001 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l2620_262073


namespace NUMINAMATH_CALUDE_average_sequence_problem_l2620_262047

theorem average_sequence_problem (a b c d e : ℝ) : 
  a = 8 ∧ 
  d = 26 ∧
  b = (a + c) / 2 ∧
  c = (b + d) / 2 ∧
  d = (c + e) / 2 
  → e = 32 := by sorry

end NUMINAMATH_CALUDE_average_sequence_problem_l2620_262047


namespace NUMINAMATH_CALUDE_sector_area_l2620_262013

/-- Given a circular sector with central angle 3 radians and perimeter 5, its area is 3/2. -/
theorem sector_area (θ : Real) (p : Real) (S : Real) : 
  θ = 3 → p = 5 → S = (θ * (p - θ)) / (2 * (2 + θ)) → S = 3/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l2620_262013


namespace NUMINAMATH_CALUDE_base_6_number_identification_l2620_262083

def is_base_6_digit (d : ℕ) : Prop := d < 6

def is_base_6_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_base_6_digit d

theorem base_6_number_identification :
  ¬ is_base_6_number 66 ∧
  ¬ is_base_6_number 207 ∧
  ¬ is_base_6_number 652 ∧
  is_base_6_number 3142 :=
sorry

end NUMINAMATH_CALUDE_base_6_number_identification_l2620_262083


namespace NUMINAMATH_CALUDE_range_of_a_l2620_262020

def p (x : ℝ) : Prop := |4 * x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∃ x, ¬(p x) ∧ q x a) ∧
  (∀ x, ¬(q x a) → ¬(p x)) →
  ∃ S : Set ℝ, S = {a : ℝ | 0 ≤ a ∧ a ≤ 1/2} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2620_262020


namespace NUMINAMATH_CALUDE_sorting_inequality_l2620_262019

theorem sorting_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sorting_inequality_l2620_262019


namespace NUMINAMATH_CALUDE_sodium_in_salt_calculation_l2620_262005

/-- The amount of sodium (in mg) per teaspoon of salt -/
def sodium_per_tsp_salt : ℝ := 50

/-- The amount of sodium (in mg) per oz of parmesan cheese -/
def sodium_per_oz_parmesan : ℝ := 25

/-- The number of teaspoons of salt in the recipe -/
def tsp_salt_in_recipe : ℝ := 2

/-- The number of oz of parmesan cheese in the original recipe -/
def oz_parmesan_in_recipe : ℝ := 8

/-- The reduction in oz of parmesan cheese to achieve 1/3 sodium reduction -/
def oz_parmesan_reduction : ℝ := 4

theorem sodium_in_salt_calculation : 
  sodium_per_tsp_salt = 50 := by sorry

end NUMINAMATH_CALUDE_sodium_in_salt_calculation_l2620_262005


namespace NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l2620_262089

/-- The curve represented by x²sin(α) - y²cos(α) = 1 is an ellipse with foci on the y-axis when α is between π/2 and 3π/4 -/
theorem curve_is_ellipse_with_foci_on_y_axis (α : Real) 
  (h_α_range : α ∈ Set.Ioo (π / 2) (3 * π / 4)) :
  ∃ (a b : Real), a > 0 ∧ b > 0 ∧ a > b ∧
  ∀ (x y : Real), x^2 * Real.sin α - y^2 * Real.cos α = 1 ↔ 
    (x^2 / b^2) + (y^2 / a^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l2620_262089


namespace NUMINAMATH_CALUDE_intersection_A_B_l2620_262084

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}
def B : Set ℝ := {x : ℝ | |x| ≤ 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2620_262084


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2620_262053

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation y = -2x + 1 in slope-intercept form. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = -2 * x + 1
  (∀ x y, L1 x y ↔ y = (1/2) * x - (3/2)) →
  (L2 P.1 P.2) →
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) * (x₂ - x₁) = -1 / ((y₂ - y₁) / (x₂ - x₁))) →
  ∀ x y, L2 x y ↔ y = -2 * x + 1 :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2620_262053


namespace NUMINAMATH_CALUDE_cross_pollinated_percentage_l2620_262007

/-- Represents the apple orchard with Fuji and Gala trees -/
structure Orchard where
  totalTrees : ℕ
  pureFuji : ℕ
  pureGala : ℕ
  crossPollinated : ℕ

/-- The percentage of cross-pollinated trees in the orchard is 2/3 -/
theorem cross_pollinated_percentage (o : Orchard) : 
  o.totalTrees = o.pureFuji + o.pureGala + o.crossPollinated →
  o.pureFuji + o.crossPollinated = 170 →
  o.pureFuji = 3 * o.totalTrees / 4 →
  o.pureGala = 30 →
  o.crossPollinated * 3 = o.totalTrees * 2 := by
  sorry

#check cross_pollinated_percentage

end NUMINAMATH_CALUDE_cross_pollinated_percentage_l2620_262007


namespace NUMINAMATH_CALUDE_number_of_cars_in_race_l2620_262056

/-- The number of cars in a race where:
  1. Each car starts with 3 people.
  2. After the halfway point, each car has 4 people.
  3. At the end of the race, there are 80 people in total. -/
theorem number_of_cars_in_race : ℕ :=
  let initial_people_per_car : ℕ := 3
  let final_people_per_car : ℕ := 4
  let total_people_at_end : ℕ := 80
  20

#check number_of_cars_in_race

end NUMINAMATH_CALUDE_number_of_cars_in_race_l2620_262056


namespace NUMINAMATH_CALUDE_soap_box_width_theorem_l2620_262064

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The partial dimensions of the soap box (width is unknown) -/
def soapBoxPartialDimensions (w : ℝ) : BoxDimensions :=
  { length := 7, width := w, height := 5 }

/-- The maximum number of soap boxes that can fit in the carton -/
def maxSoapBoxes : ℕ := 300

/-- Theorem stating the width of the soap box that allows exactly 300 to fit in the carton -/
theorem soap_box_width_theorem (w : ℝ) :
  (boxVolume cartonDimensions = maxSoapBoxes * boxVolume (soapBoxPartialDimensions w)) ↔ w = 6 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_width_theorem_l2620_262064


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l2620_262036

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = {x : ℝ | x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l2620_262036


namespace NUMINAMATH_CALUDE_ship_length_proof_l2620_262034

/-- The length of the ship in terms of Emily's normal steps -/
def ship_length : ℕ := 120

/-- The number of steps Emily takes with wind behind her -/
def steps_with_wind : ℕ := 300

/-- The number of steps Emily takes against the wind -/
def steps_against_wind : ℕ := 75

/-- The number of extra steps the wind allows Emily to take in the direction it blows -/
def wind_effect : ℕ := 20

theorem ship_length_proof :
  ∀ (E S : ℝ),
  E > 0 ∧ S > 0 →
  (steps_with_wind + wind_effect : ℝ) * E = ship_length + (steps_with_wind + wind_effect) * S →
  (steps_against_wind - wind_effect : ℝ) * E = ship_length - (steps_against_wind - wind_effect) * S →
  ship_length = 120 := by
  sorry

end NUMINAMATH_CALUDE_ship_length_proof_l2620_262034


namespace NUMINAMATH_CALUDE_twelve_boys_handshakes_l2620_262025

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 12 boys, the total number of handshakes is 66 -/
theorem twelve_boys_handshakes :
  handshakes 12 = 66 := by
  sorry

end NUMINAMATH_CALUDE_twelve_boys_handshakes_l2620_262025


namespace NUMINAMATH_CALUDE_no_real_solutions_l2620_262003

theorem no_real_solutions :
  ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 12*y + 28 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2620_262003


namespace NUMINAMATH_CALUDE_function_range_l2620_262098

/-- The function y = (3 * sin(x) + 1) / (sin(x) + 2) has a range of [-2, 4/3] -/
theorem function_range (x : ℝ) : 
  let y := (3 * Real.sin x + 1) / (Real.sin x + 2)
  ∃ (a b : ℝ), a = -2 ∧ b = 4/3 ∧ a ≤ y ∧ y ≤ b ∧
  (∃ (x₁ x₂ : ℝ), 
    (3 * Real.sin x₁ + 1) / (Real.sin x₁ + 2) = a ∧
    (3 * Real.sin x₂ + 1) / (Real.sin x₂ + 2) = b) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l2620_262098


namespace NUMINAMATH_CALUDE_circle_properties_l2620_262010

/-- Given a circle with equation x^2 + y^2 = 10x - 8y + 4, prove its properties --/
theorem circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 = 10*x - 8*y + 4
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 = 5 ∧ center.2 = -4) ∧  -- Center is (5, -4)
    (radius = 3 * Real.sqrt 5) ∧     -- Radius is 3√5
    (center.1 + center.2 = 1) ∧      -- Sum of center coordinates is 1
    ∀ (x y : ℝ), equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l2620_262010


namespace NUMINAMATH_CALUDE_construction_delay_l2620_262057

/-- Represents the construction project -/
structure ConstructionProject where
  total_days : ℕ
  initial_workers : ℕ
  additional_workers : ℕ
  days_before_addition : ℕ

/-- Calculates the delay in days if additional workers were not added -/
def calculate_delay (project : ConstructionProject) : ℕ :=
  let total_work := project.total_days * project.initial_workers
  let work_done_before_addition := project.days_before_addition * project.initial_workers
  let remaining_work := total_work - work_done_before_addition
  let days_with_additional_workers := project.total_days - project.days_before_addition
  let work_done_after_addition := days_with_additional_workers * (project.initial_workers + project.additional_workers)
  (remaining_work + project.initial_workers - 1) / project.initial_workers - days_with_additional_workers

theorem construction_delay (project : ConstructionProject) 
  (h1 : project.total_days = 100)
  (h2 : project.initial_workers = 100)
  (h3 : project.additional_workers = 100)
  (h4 : project.days_before_addition = 20) :
  calculate_delay project = 80 := by
  sorry

#eval calculate_delay { total_days := 100, initial_workers := 100, additional_workers := 100, days_before_addition := 20 }

end NUMINAMATH_CALUDE_construction_delay_l2620_262057


namespace NUMINAMATH_CALUDE_exists_universal_program_l2620_262054

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Fin 8) (y : Fin 8)

/-- Represents a labyrinth configuration --/
def Labyrinth := Fin 8 → Fin 8 → Bool

/-- Represents a move command --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a program as a list of commands --/
def Program := List Command

/-- Checks if a square is accessible in the labyrinth --/
def isAccessible (lab : Labyrinth) (pos : Position) : Bool :=
  lab pos.x pos.y

/-- Executes a single command on a position in a labyrinth --/
def executeCommand (lab : Labyrinth) (pos : Position) (cmd : Command) : Position :=
  sorry

/-- Executes a program on a position in a labyrinth --/
def executeProgram (lab : Labyrinth) (pos : Position) (prog : Program) : Position :=
  sorry

/-- Checks if a program visits all accessible squares in a labyrinth from a given starting position --/
def visitsAllAccessible (lab : Labyrinth) (start : Position) (prog : Program) : Prop :=
  sorry

/-- The main theorem: there exists a program that visits all accessible squares
    for any labyrinth and starting position --/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (lab : Labyrinth) (start : Position),
      visitsAllAccessible lab start prog :=
sorry

end NUMINAMATH_CALUDE_exists_universal_program_l2620_262054


namespace NUMINAMATH_CALUDE_double_average_l2620_262078

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let new_avg := 2 * original_avg
  new_avg = 160 := by sorry

end NUMINAMATH_CALUDE_double_average_l2620_262078


namespace NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l2620_262043

theorem smallest_y_in_arithmetic_series (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 →  -- all terms are positive
  ∃ d : ℝ, x = y - d ∧ z = y + d →  -- arithmetic series condition
  x * y * z = 125 →  -- product condition
  y ≥ 5 ∧ ∀ y' : ℝ, (∃ x' z' : ℝ, x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    (∃ d' : ℝ, x' = y' - d' ∧ z' = y' + d') ∧ 
    x' * y' * z' = 125) → y' ≥ 5 := by
  sorry

#check smallest_y_in_arithmetic_series

end NUMINAMATH_CALUDE_smallest_y_in_arithmetic_series_l2620_262043


namespace NUMINAMATH_CALUDE_four_player_tournament_games_l2620_262055

/-- The number of games in a round-robin tournament with n players -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against 
    every other player exactly once, the total number of games is 6 -/
theorem four_player_tournament_games : 
  num_games 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_player_tournament_games_l2620_262055


namespace NUMINAMATH_CALUDE_wang_gang_seat_location_l2620_262016

/-- Represents a seat in a classroom -/
structure Seat where
  row : Nat
  column : Nat

/-- Represents a classroom -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Checks if a seat is valid for a given classroom -/
def is_valid_seat (c : Classroom) (s : Seat) : Prop :=
  s.row ≤ c.rows ∧ s.column ≤ c.columns

theorem wang_gang_seat_location (c : Classroom) (s : Seat) :
  c.rows = 7 ∧ c.columns = 8 ∧ s = Seat.mk 5 8 ∧ is_valid_seat c s →
  s.row = 5 ∧ s.column = 8 := by
  sorry

end NUMINAMATH_CALUDE_wang_gang_seat_location_l2620_262016


namespace NUMINAMATH_CALUDE_expression_simplification_l2620_262066

theorem expression_simplification (m n x : ℚ) :
  (5 * m + 3 * n - 7 * m - n = -2 * m + 2 * n) ∧
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2) = 2 * x^2 - 5 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2620_262066


namespace NUMINAMATH_CALUDE_time_ratio_in_countries_l2620_262075

/- Given conditions -/
def total_trip_duration : ℕ := 10
def time_in_first_country : ℕ := 2

/- Theorem to prove -/
theorem time_ratio_in_countries :
  (total_trip_duration - time_in_first_country) / time_in_first_country = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_ratio_in_countries_l2620_262075


namespace NUMINAMATH_CALUDE_min_disks_is_fifteen_l2620_262042

/-- Represents the storage problem with given file sizes and disk capacity. -/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  files_09mb : ℕ
  files_08mb : ℕ
  files_05mb : ℕ
  h_total : total_files = files_09mb + files_08mb + files_05mb

/-- Calculates the minimum number of disks required for the given storage problem. -/
def min_disks_required (p : StorageProblem) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of disks required for the given problem is 15. -/
theorem min_disks_is_fifteen :
  let p : StorageProblem := {
    total_files := 35,
    disk_capacity := 8/5,
    files_09mb := 5,
    files_08mb := 10,
    files_05mb := 20,
    h_total := by rfl
  }
  min_disks_required p = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_disks_is_fifteen_l2620_262042


namespace NUMINAMATH_CALUDE_max_regions_intersected_by_line_l2620_262037

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The number of regions that the planes of a tetrahedron divide space into -/
def num_regions_tetrahedron : ℕ := 15

/-- The maximum number of regions a line can intersect -/
def max_intersected_regions (t : Tetrahedron) (l : Line) : ℕ := sorry

/-- Theorem stating the maximum number of regions a line can intersect -/
theorem max_regions_intersected_by_line (t : Tetrahedron) :
  ∃ l : Line, max_intersected_regions t l = 5 ∧
  ∀ l' : Line, max_intersected_regions t l' ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_regions_intersected_by_line_l2620_262037


namespace NUMINAMATH_CALUDE_probability_of_drawing_balls_l2620_262096

theorem probability_of_drawing_balls (prob_A prob_B : ℝ) 
  (h_prob_A : prob_A = 1/3) (h_prob_B : prob_B = 1/2) :
  let prob_both_red := prob_A * prob_B
  let prob_exactly_one_red := prob_A * (1 - prob_B) + (1 - prob_A) * prob_B
  let prob_both_not_red := (1 - prob_A) * (1 - prob_B)
  let prob_at_least_one_red := 1 - prob_both_not_red
  (prob_both_red = 1/6) ∧
  (prob_exactly_one_red = 1/2) ∧
  (prob_both_not_red = 5/6) ∧
  (prob_at_least_one_red = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_drawing_balls_l2620_262096


namespace NUMINAMATH_CALUDE_barn_paint_area_l2620_262015

/-- Calculates the total area to be painted in a rectangular barn -/
def total_paint_area (width length height : ℝ) : ℝ :=
  2 * (2 * width * height + 2 * length * height) + 2 * width * length

/-- Theorem stating the total area to be painted for a specific barn -/
theorem barn_paint_area :
  let width : ℝ := 15
  let length : ℝ := 20
  let height : ℝ := 8
  total_paint_area width length height = 1720 := by
  sorry

end NUMINAMATH_CALUDE_barn_paint_area_l2620_262015


namespace NUMINAMATH_CALUDE_max_at_two_implies_a_geq_neg_half_l2620_262000

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem max_at_two_implies_a_geq_neg_half (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ f a 2) →
  a ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_at_two_implies_a_geq_neg_half_l2620_262000


namespace NUMINAMATH_CALUDE_combined_discount_optimal_l2620_262097

/-- Represents the cost calculation for a clothing purchase with discount options -/
def ClothingPurchase (x : ℕ) : Prop :=
  x > 30 ∧
  let jacket_price : ℕ := 100
  let tshirt_price : ℕ := 60
  let option1_cost : ℕ := 3000 + 60 * (x - 30)
  let option2_cost : ℕ := 2400 + 48 * x
  let combined_cost : ℕ := 3000 + 48 * (x - 30)
  combined_cost ≤ min option1_cost option2_cost

/-- Theorem stating that the combined discount strategy is optimal for any valid x -/
theorem combined_discount_optimal (x : ℕ) : ClothingPurchase x := by
  sorry

end NUMINAMATH_CALUDE_combined_discount_optimal_l2620_262097


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2620_262068

theorem sum_of_roots_quadratic : 
  let a : ℝ := 1
  let b : ℝ := -8
  let c : ℝ := -7
  let sum_of_roots := -b / a
  sum_of_roots = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2620_262068


namespace NUMINAMATH_CALUDE_greatest_power_of_eleven_l2620_262069

theorem greatest_power_of_eleven (n : ℕ+) : 
  (Finset.card (Nat.divisors n) = 72) →
  (Finset.card (Nat.divisors (11 * n)) = 96) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) →
  (∃ k : ℕ, 11^k ∣ n ∧ ∀ m : ℕ, 11^m ∣ n → m ≤ k) ∧ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_eleven_l2620_262069


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2620_262001

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a₁ = 3
  a 4 = 24 →                                 -- a₄ = 24
  a 3 + a 4 + a 5 = 84 :=                    -- prove a₃ + a₄ + a₅ = 84
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2620_262001


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2620_262027

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + y = 3) :
  2^x + 2^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 3 ∧ 2^x₀ + 2^y₀ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2620_262027


namespace NUMINAMATH_CALUDE_amusement_park_running_cost_l2620_262008

/-- The amusement park problem -/
theorem amusement_park_running_cost 
  (initial_cost : ℝ) 
  (daily_tickets : ℕ) 
  (ticket_price : ℝ) 
  (days_to_breakeven : ℕ) 
  (h1 : initial_cost = 100000)
  (h2 : daily_tickets = 150)
  (h3 : ticket_price = 10)
  (h4 : days_to_breakeven = 200) :
  let daily_revenue := daily_tickets * ticket_price
  let total_revenue := daily_revenue * days_to_breakeven
  let daily_running_cost_percentage := 
    (total_revenue - initial_cost) / (initial_cost * days_to_breakeven) * 100
  daily_running_cost_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_amusement_park_running_cost_l2620_262008


namespace NUMINAMATH_CALUDE_sin_2017pi_over_6_l2620_262022

theorem sin_2017pi_over_6 : Real.sin ((2017 * π) / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_2017pi_over_6_l2620_262022


namespace NUMINAMATH_CALUDE_gcd_128_144_360_l2620_262086

theorem gcd_128_144_360 : Nat.gcd 128 (Nat.gcd 144 360) = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_128_144_360_l2620_262086


namespace NUMINAMATH_CALUDE_sum_divisibility_l2620_262023

theorem sum_divisibility (y : ℕ) : 
  y = 36 + 48 + 72 + 144 + 216 + 432 + 1296 →
  3 ∣ y ∧ 4 ∣ y ∧ 6 ∣ y ∧ 12 ∣ y :=
by sorry

end NUMINAMATH_CALUDE_sum_divisibility_l2620_262023


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2620_262085

theorem rationalize_denominator : 
  (30 : ℝ) / Real.sqrt 15 = 2 * Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2620_262085


namespace NUMINAMATH_CALUDE_male_honor_roll_fraction_l2620_262052

theorem male_honor_roll_fraction (total : ℝ) (h1 : total > 0) :
  let female_ratio : ℝ := 2 / 5
  let female_honor_ratio : ℝ := 5 / 6
  let total_honor_ratio : ℝ := 22 / 30
  let male_ratio : ℝ := 1 - female_ratio
  let male_honor_ratio : ℝ := (total_honor_ratio * total - female_honor_ratio * female_ratio * total) / (male_ratio * total)
  male_honor_ratio = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_male_honor_roll_fraction_l2620_262052


namespace NUMINAMATH_CALUDE_apple_pie_servings_l2620_262062

theorem apple_pie_servings 
  (guests : ℕ) 
  (apples_per_guest : ℝ) 
  (num_pies : ℕ) 
  (apples_per_serving : ℝ) 
  (h1 : guests = 12) 
  (h2 : apples_per_guest = 3) 
  (h3 : num_pies = 3) 
  (h4 : apples_per_serving = 1.5) : 
  (guests * apples_per_guest) / (num_pies * apples_per_serving) = 8 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_servings_l2620_262062


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2620_262094

theorem right_triangle_perimeter (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_area : (1/2) * a * b = 150) (h_leg : a = 30) : 
  a + b + c = 40 + 10 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2620_262094


namespace NUMINAMATH_CALUDE_trapezoid_ab_length_l2620_262091

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- The ratio of the areas of triangles ABC and ADC
  area_ratio : ℚ
  -- The combined length of bases AB and CD
  total_base_length : ℝ
  -- The length of base AB
  ab_length : ℝ

/-- Theorem: If the area ratio is 8:2 and the total base length is 120,
    then the length of AB is 96 -/
theorem trapezoid_ab_length (t : Trapezoid) :
  t.area_ratio = 8 / 2 ∧ t.total_base_length = 120 → t.ab_length = 96 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ab_length_l2620_262091


namespace NUMINAMATH_CALUDE_max_value_theorem_l2620_262080

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2620_262080


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l2620_262070

theorem absolute_value_equation_solution_count : 
  ∃! x : ℝ, |x - 5| = |x + 3| := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_count_l2620_262070


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l2620_262039

theorem geometric_progression_ratio (x y z r : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x ≠ y → y ≠ z → x ≠ z →
  (∃ a : ℝ, a ≠ 0 ∧ 
    x * (y - z) = a ∧ 
    y * (z - x) = a * r ∧ 
    z * (x - y) = a * r^2) →
  x * (y - z) * y * (z - x) * z * (x - y) = (y * (z - x))^2 →
  r = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l2620_262039


namespace NUMINAMATH_CALUDE_total_vertices_eq_21_l2620_262012

/-- The number of vertices in a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of triangles -/
def num_triangles : ℕ := 1

/-- The number of hexagons -/
def num_hexagons : ℕ := 3

/-- The total number of vertices in all shapes -/
def total_vertices : ℕ := num_triangles * triangle_vertices + num_hexagons * hexagon_vertices

theorem total_vertices_eq_21 : total_vertices = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_vertices_eq_21_l2620_262012


namespace NUMINAMATH_CALUDE_total_earnings_l2620_262063

/-- Proves that the total amount earned by 5 men, W women, and 8 boys is 210 rupees -/
theorem total_earnings (W : ℕ) (mens_wage : ℕ) 
  (h1 : 5 = W)  -- 5 men are equal to W women
  (h2 : W = 8)  -- W women are equal to 8 boys
  (h3 : mens_wage = 14)  -- Men's wages are Rs. 14 each
  : 5 * mens_wage + W * mens_wage + 8 * mens_wage = 210 := by
  sorry

#eval 5 * 14 + 5 * 14 + 8 * 14  -- Evaluates to 210

end NUMINAMATH_CALUDE_total_earnings_l2620_262063


namespace NUMINAMATH_CALUDE_range_of_x_l2620_262033

theorem range_of_x (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ 2 * Real.pi)
  (h2 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2620_262033


namespace NUMINAMATH_CALUDE_cube_root_abs_sqrt_squared_sum_l2620_262093

theorem cube_root_abs_sqrt_squared_sum (π : ℝ) : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) + |1 - π| + (9 : ℝ).sqrt - (-1 : ℝ)^2 = π - 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_abs_sqrt_squared_sum_l2620_262093


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l2620_262045

theorem high_school_math_club_payment (B : ℕ) : 
  B < 10 → (∃ k : ℤ, 200 + 10 * B + 5 = 13 * k) → B = 1 :=
by sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l2620_262045


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2620_262038

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3*p - 4) * (6*q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2620_262038


namespace NUMINAMATH_CALUDE_min_value_theorem_l2620_262014

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (1 / x + 4 / y) ≥ 3 ∧ ∃ x0 y0 : ℝ, x0 > 0 ∧ y0 > 0 ∧ x0 + y0 = 3 ∧ 1 / x0 + 4 / y0 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2620_262014


namespace NUMINAMATH_CALUDE_correct_aprons_tomorrow_l2620_262017

def aprons_to_sew_tomorrow (total : ℕ) (already_sewn : ℕ) (today_multiplier : ℕ) : ℕ :=
  let today_sewn := already_sewn * today_multiplier
  let total_sewn := already_sewn + today_sewn
  let remaining := total - total_sewn
  remaining / 2

theorem correct_aprons_tomorrow :
  aprons_to_sew_tomorrow 150 13 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_correct_aprons_tomorrow_l2620_262017


namespace NUMINAMATH_CALUDE_initial_water_amount_l2620_262072

/-- 
Given a bucket with an initial amount of water, prove that this amount is 3 gallons
when adding 6.8 gallons results in a total of 9.8 gallons.
-/
theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l2620_262072


namespace NUMINAMATH_CALUDE_zeros_after_one_in_100_pow_250_l2620_262050

/-- The number of zeros following the digit '1' in the expanded form of 100^250 -/
def zeros_after_one : ℕ := 500

/-- The exponent in the expression 100^250 -/
def exponent : ℕ := 250

theorem zeros_after_one_in_100_pow_250 : zeros_after_one = 2 * exponent := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_100_pow_250_l2620_262050


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2620_262058

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 6*x + 3 = 0) ↔ ((x + 3)^2 = 6) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2620_262058


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2620_262032

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 4) = 256 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2620_262032


namespace NUMINAMATH_CALUDE_total_books_l2620_262035

theorem total_books (tim_books sam_books : ℕ) 
  (h1 : tim_books = 44) 
  (h2 : sam_books = 52) : 
  tim_books + sam_books = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2620_262035


namespace NUMINAMATH_CALUDE_polycarp_multiplication_l2620_262006

theorem polycarp_multiplication (a b : ℕ) : 
  100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 →
  1000 * a + b = 7 * a * b →
  a = 143 ∧ b = 143 := by
sorry

end NUMINAMATH_CALUDE_polycarp_multiplication_l2620_262006


namespace NUMINAMATH_CALUDE_square_sum_ge_double_product_l2620_262024

theorem square_sum_ge_double_product (a b : ℝ) : (a^2 + b^2 > 2*a*b) ∨ (a^2 + b^2 = 2*a*b) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_double_product_l2620_262024


namespace NUMINAMATH_CALUDE_equation_solutions_l2620_262009

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (3 - x)^4 = 146 ↔ 
  x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2620_262009


namespace NUMINAMATH_CALUDE_flagpole_shadow_length_l2620_262040

theorem flagpole_shadow_length 
  (flagpole_height : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : building_height = 28)
  (h3 : building_shadow = 70)
  : ∃ (flagpole_shadow : ℝ), 
    flagpole_height / flagpole_shadow = building_height / building_shadow ∧ 
    flagpole_shadow = 45 := by
  sorry

end NUMINAMATH_CALUDE_flagpole_shadow_length_l2620_262040


namespace NUMINAMATH_CALUDE_equivalent_statements_l2620_262092

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l2620_262092


namespace NUMINAMATH_CALUDE_expected_heads_value_l2620_262059

/-- The probability of a coin landing heads -/
def p_heads : ℚ := 1/3

/-- The number of coins -/
def num_coins : ℕ := 100

/-- The maximum number of flips allowed for each coin -/
def max_flips : ℕ := 4

/-- The probability of a coin showing heads after up to four flips -/
def p_heads_after_four_flips : ℚ :=
  p_heads + (1 - p_heads) * p_heads + (1 - p_heads)^2 * p_heads + (1 - p_heads)^3 * p_heads

/-- The expected number of coins showing heads after all flips -/
def expected_heads : ℚ := num_coins * p_heads_after_four_flips

theorem expected_heads_value : expected_heads = 6500/81 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_value_l2620_262059


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2620_262099

theorem quadratic_expression_value (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 24 * x * y + 13 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2620_262099


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_six_l2620_262049

theorem sum_of_x_and_y_equals_six (x y : ℝ) (h : x^2 + y^2 = 8*x + 4*y - 20) : x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_six_l2620_262049


namespace NUMINAMATH_CALUDE_m_range_l2620_262087

theorem m_range (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) (h_ineq : x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2620_262087


namespace NUMINAMATH_CALUDE_correct_operation_l2620_262088

theorem correct_operation (x y : ℝ) : y * x - 2 * x * y = -x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2620_262088


namespace NUMINAMATH_CALUDE_valid_parameterizations_l2620_262061

def is_valid_parameterization (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, let (x, y) := p₀ + t • d
           y = x - 1

theorem valid_parameterizations :
  (is_valid_parameterization (1, 0) (1, 1)) ∧
  (is_valid_parameterization (0, -1) (-1, -1)) ∧
  (is_valid_parameterization (2, 1) (0.5, 0.5)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l2620_262061


namespace NUMINAMATH_CALUDE_intersection_sum_l2620_262011

theorem intersection_sum (c d : ℚ) : 
  (3 = (1/3) * (-1) + c) → 
  (-1 = (1/3) * 3 + d) → 
  c + d = 4/3 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l2620_262011


namespace NUMINAMATH_CALUDE_complex_moduli_product_l2620_262041

theorem complex_moduli_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l2620_262041


namespace NUMINAMATH_CALUDE_christmas_decorations_distribution_l2620_262030

theorem christmas_decorations_distribution :
  let total_decorations : ℕ := 455
  let valid_student_count (n : ℕ) : Prop := 10 < n ∧ n < 70
  let valid_distribution (students : ℕ) (per_student : ℕ) : Prop :=
    valid_student_count students ∧
    students * per_student = total_decorations

  (∀ students per_student, valid_distribution students per_student →
    (students = 65 ∧ per_student = 7) ∨
    (students = 35 ∧ per_student = 13) ∨
    (students = 13 ∧ per_student = 35)) ∧
  (valid_distribution 65 7 ∧
   valid_distribution 35 13 ∧
   valid_distribution 13 35) :=
by sorry

end NUMINAMATH_CALUDE_christmas_decorations_distribution_l2620_262030


namespace NUMINAMATH_CALUDE_smallest_number_l2620_262028

def number_set : Finset ℤ := {0, -3, 2, -2}

theorem smallest_number : 
  ∀ x ∈ number_set, -3 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2620_262028


namespace NUMINAMATH_CALUDE_ellipse_ratio_l2620_262029

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-latus rectum c,
    if a² + b² - 3c² = 0, then (a + c) / (a - c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_l2620_262029


namespace NUMINAMATH_CALUDE_stating_meeting_handshakes_l2620_262090

/-- 
Given a group of people at a meeting, where each person shakes hands with at least
a certain number of others, this function calculates the minimum possible number of handshakes.
-/
def min_handshakes (n : ℕ) (min_shakes_per_person : ℕ) : ℕ :=
  (n * min_shakes_per_person) / 2

/-- 
Theorem stating that for a meeting of 30 people where each person shakes hands with
at least 3 others, the minimum possible number of handshakes is 45.
-/
theorem meeting_handshakes :
  min_handshakes 30 3 = 45 := by
  sorry

#eval min_handshakes 30 3

end NUMINAMATH_CALUDE_stating_meeting_handshakes_l2620_262090


namespace NUMINAMATH_CALUDE_exactly_four_triples_l2620_262076

/-- The number of ordered triples (a, b, c) of positive integers satisfying the given LCM conditions -/
def count_triples : ℕ := 4

/-- Predicate to check if a triple (a, b, c) satisfies the LCM conditions -/
def satisfies_conditions (a b c : ℕ+) : Prop :=
  Nat.lcm a b = 90 ∧ Nat.lcm a c = 980 ∧ Nat.lcm b c = 630

/-- The main theorem stating that there are exactly 4 triples satisfying the conditions -/
theorem exactly_four_triples :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = count_triples ∧
    ∀ t, t ∈ s ↔ satisfies_conditions t.1 t.2.1 t.2.2) :=
sorry

end NUMINAMATH_CALUDE_exactly_four_triples_l2620_262076


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l2620_262082

/-- Exchange rate between bahs and rahs -/
def bah_to_rah_rate : ℚ := 16 / 10

/-- Exchange rate between rahs and yahs -/
def rah_to_yah_rate : ℚ := 15 / 9

/-- The number of yahs we want to convert -/
def yah_amount : ℕ := 2000

/-- The expected number of bahs after conversion -/
def expected_bah_amount : ℕ := 375

theorem yah_to_bah_conversion :
  (yah_amount : ℚ) / (rah_to_yah_rate * bah_to_rah_rate) = expected_bah_amount := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l2620_262082


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2620_262031

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan (b / a) = π / 6) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2620_262031


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2620_262081

theorem arithmetic_progression_first_term
  (a : ℕ → ℝ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_arithmetic : ∃ d, ∀ n, a (n + 1) - a n = d)
  (h_sum : a 0 + a 1 + a 2 = 12)
  (h_product : a 0 * a 1 * a 2 = 48) :
  a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l2620_262081


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2620_262004

theorem complex_equation_solution (z z₁ z₂ : ℂ) : 
  z₁ = 5 + 10*I ∧ z₂ = 3 - 4*I ∧ (1 : ℂ)/z = 1/z₁ + 1/z₂ → z = 5 - (5/2)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2620_262004


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2620_262067

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs x = abs y) ∧                             -- Equally distant from x-axis and y-axis
    (abs x = abs (x + y - 3) / Real.sqrt 2) ∧     -- Equally distant from the line x + y = 3
    (x = 3/2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2620_262067


namespace NUMINAMATH_CALUDE_purely_imaginary_m_eq_3_second_quadrant_m_range_l2620_262048

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := (m^2 - m - 6 : ℝ) + (m^2 + 5*m + 6 : ℝ) * Complex.I

/-- Theorem: If z is purely imaginary, then m = 3 -/
theorem purely_imaginary_m_eq_3 :
  (∀ m : ℝ, z m = Complex.I * Complex.im (z m)) → (∃ m : ℝ, m = 3) :=
sorry

/-- Theorem: If z is in the second quadrant, then -2 < m < 3 -/
theorem second_quadrant_m_range :
  (∀ m : ℝ, Complex.re (z m) < 0 ∧ Complex.im (z m) > 0) → (∀ m : ℝ, -2 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_m_eq_3_second_quadrant_m_range_l2620_262048


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2620_262018

/-- Alice's number -/
def alice_number : ℕ := 45

/-- Bob's number is a natural number -/
def bob_number : ℕ := sorry

/-- Every prime factor of Alice's number is also a prime factor of Bob's number -/
axiom bob_has_alice_prime_factors :
  ∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ bob_number

/-- Bob's number is the smallest possible given the conditions -/
axiom bob_number_is_smallest :
  ∀ n : ℕ, (∀ p : ℕ, Prime p → p ∣ alice_number → p ∣ n) → bob_number ≤ n

theorem smallest_bob_number : bob_number = 15 := by sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2620_262018


namespace NUMINAMATH_CALUDE_triangle_side_length_l2620_262065

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  c = Real.sqrt 2 →
  b = Real.sqrt 6 →
  B = 2 * π / 3 →  -- 120° in radians
  a^2 + a * Real.sqrt 2 - 4 = 0 →
  a = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2620_262065


namespace NUMINAMATH_CALUDE_triangle_area_solution_l2620_262079

/-- Given a triangle with vertices (0, 0), (x, 3x), and (x, 0), where x > 0,
    if the area of this triangle is 100 square units, then x = 10√6/3 -/
theorem triangle_area_solution (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 100 → x = (10 * Real.sqrt 6) / 3 := by
  sorry

#check triangle_area_solution

end NUMINAMATH_CALUDE_triangle_area_solution_l2620_262079


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_y_eq_4x_squared_l2620_262044

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
theorem parabola_focus (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1 / (4 * a))^2 :=
sorry

/-- The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem focus_of_y_eq_4x_squared :
  let f : ℝ × ℝ := (0, 1/16)
  ∀ x y : ℝ, y = 4 * x^2 → (x - f.1)^2 + (y - f.2)^2 = (y - f.2 + 1/16)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_y_eq_4x_squared_l2620_262044


namespace NUMINAMATH_CALUDE_n_minus_m_not_odd_l2620_262021

theorem n_minus_m_not_odd (n m : ℤ) (h : Even (n^2 - m^2)) : ¬Odd (n - m) := by
  sorry

end NUMINAMATH_CALUDE_n_minus_m_not_odd_l2620_262021


namespace NUMINAMATH_CALUDE_rain_given_east_wind_l2620_262077

/-- Given that:
    1. The probability of an east wind in April is 8/30
    2. The probability of both an east wind and rain in April is 7/30
    Prove that the probability of rain in April given an east wind is 7/8 -/
theorem rain_given_east_wind (p_east : ℚ) (p_east_and_rain : ℚ) 
  (h1 : p_east = 8/30) (h2 : p_east_and_rain = 7/30) :
  p_east_and_rain / p_east = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_rain_given_east_wind_l2620_262077


namespace NUMINAMATH_CALUDE_product_mod_seven_l2620_262071

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2620_262071


namespace NUMINAMATH_CALUDE_mrs_hilt_spent_74_cents_l2620_262095

/-- Calculates the total amount spent by Mrs. Hilt at the school store -/
def school_store_total (notebook_cost ruler_cost pencil_cost : ℕ) (num_pencils : ℕ) : ℕ :=
  notebook_cost + ruler_cost + (pencil_cost * num_pencils)

/-- Proves that Mrs. Hilt spent 74 cents at the school store -/
theorem mrs_hilt_spent_74_cents :
  school_store_total 35 18 7 3 = 74 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_spent_74_cents_l2620_262095


namespace NUMINAMATH_CALUDE_not_perfect_square_l2620_262026

theorem not_perfect_square (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd (a - b) (a * b + 1) = 1) (h3 : Nat.gcd (a + b) (a * b - 1) = 1) :
  ¬ ∃ k : ℕ, (a - b)^2 + (a * b + 1)^2 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2620_262026


namespace NUMINAMATH_CALUDE_remainder_theorem_l2620_262074

theorem remainder_theorem (n : ℤ) (h : n % 11 = 5) : (4 * n - 9) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2620_262074


namespace NUMINAMATH_CALUDE_fuel_tank_ethanol_percentage_l2620_262002

/-- Fuel tank problem -/
theorem fuel_tank_ethanol_percentage
  (tank_capacity : ℝ)
  (fuel_a_ethanol_percentage : ℝ)
  (total_ethanol : ℝ)
  (fuel_a_volume : ℝ)
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_ethanol_percentage = 12 / 100)
  (h3 : total_ethanol = 30)
  (h4 : fuel_a_volume = 106) :
  (total_ethanol - fuel_a_ethanol_percentage * fuel_a_volume) / (tank_capacity - fuel_a_volume) = 16 / 100 := by
sorry

end NUMINAMATH_CALUDE_fuel_tank_ethanol_percentage_l2620_262002


namespace NUMINAMATH_CALUDE_rational_function_sum_l2620_262051

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  h_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  h_horiz_asymp : ∀ ε > 0, ∃ M, ∀ x, |x| > M → |p x / q x| < ε
  h_vert_asymp : ContinuousAt q (-2) ∧ q (-2) = 0
  h_p3 : p 3 = 1
  h_q3 : q 3 = 4

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) : 
  ∀ x, f.p x + f.q x = (4 * x^2 + 7 * x - 9) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_sum_l2620_262051
