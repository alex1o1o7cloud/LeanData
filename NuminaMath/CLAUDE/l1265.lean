import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_pattern_l1265_126591

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_pattern (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 - 2 * a 2 + a 3 = 0) →
  (a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0) →
  (a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0) →
  (a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_pattern_l1265_126591


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l1265_126541

theorem cubic_expansion_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x, (2*x + 1)^3 = a₀*x^3 + a₁*x^2 + a₂*x + a₃) →
  a₁ + a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l1265_126541


namespace NUMINAMATH_CALUDE_f_period_f_definition_f_negative_one_l1265_126501

def f (x : ℝ) : ℝ := sorry

theorem f_period (x : ℝ) : f (x + 2) = f x := sorry

theorem f_definition (x : ℝ) (h : x ∈ Set.Icc 1 3) : f x = x - 2 := sorry

theorem f_negative_one : f (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_f_period_f_definition_f_negative_one_l1265_126501


namespace NUMINAMATH_CALUDE_keaton_apple_earnings_l1265_126505

/-- Represents Keaton's farm earnings -/
structure FarmEarnings where
  orangeHarvestFrequency : ℕ  -- Number of orange harvests per year
  orangeHarvestValue : ℕ      -- Value of each orange harvest in dollars
  totalAnnualEarnings : ℕ     -- Total annual earnings in dollars

/-- Calculates the annual earnings from apple harvest -/
def appleEarnings (f : FarmEarnings) : ℕ :=
  f.totalAnnualEarnings - (f.orangeHarvestFrequency * f.orangeHarvestValue)

/-- Theorem: Keaton's annual earnings from apple harvest is $120 -/
theorem keaton_apple_earnings :
  ∃ (f : FarmEarnings),
    f.orangeHarvestFrequency = 6 ∧
    f.orangeHarvestValue = 50 ∧
    f.totalAnnualEarnings = 420 ∧
    appleEarnings f = 120 := by
  sorry

end NUMINAMATH_CALUDE_keaton_apple_earnings_l1265_126505


namespace NUMINAMATH_CALUDE_A_power_95_l1265_126521

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_95 : A^95 = !![0, 0, 0; 0, 0, 1; 0, -1, 0] := by
  sorry

end NUMINAMATH_CALUDE_A_power_95_l1265_126521


namespace NUMINAMATH_CALUDE_min_q_value_l1265_126564

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_right_triangle (p q : ℕ) : Prop := p + q = 90

theorem min_q_value (p q : ℕ) (h1 : is_right_triangle p q) (h2 : is_prime p) (h3 : p > q) :
  q ≥ 7 := by sorry

end NUMINAMATH_CALUDE_min_q_value_l1265_126564


namespace NUMINAMATH_CALUDE_horner_rule_evaluation_l1265_126597

/-- Horner's Rule evaluation for a polynomial --/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x² + 79x³ + 6x⁴ + 5x⁵ + 3x⁶ --/
def f : List ℤ := [12, 35, -8, 79, 6, 5, 3]

/-- Theorem: The value of f(-4) using Horner's Rule is 220 --/
theorem horner_rule_evaluation :
  horner_eval f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_evaluation_l1265_126597


namespace NUMINAMATH_CALUDE_max_prob_with_C_second_l1265_126579

/-- Represents the probability of winning against a player -/
structure WinProbability (α : Type) where
  prob : α → ℝ
  pos : ∀ x, prob x > 0

variable {α : Type}

/-- The players A, B, and C -/
inductive Player : Type where
  | A : Player
  | B : Player
  | C : Player

/-- The probabilities of winning against each player -/
def win_prob (p : WinProbability Player) : Prop :=
  p.prob Player.A < p.prob Player.B ∧ p.prob Player.B < p.prob Player.C

/-- The probability of winning two consecutive games when player x is in the second game -/
def prob_two_consec_wins (p : WinProbability Player) (x : Player) : ℝ :=
  2 * (p.prob Player.A * p.prob x + p.prob Player.B * p.prob x + p.prob Player.C * p.prob x
     - 2 * p.prob Player.A * p.prob Player.B * p.prob Player.C)

/-- The theorem stating that the probability is maximized when C is in the second game -/
theorem max_prob_with_C_second (p : WinProbability Player) (h : win_prob p) :
    ∀ x : Player, prob_two_consec_wins p Player.C ≥ prob_two_consec_wins p x :=
  sorry

end NUMINAMATH_CALUDE_max_prob_with_C_second_l1265_126579


namespace NUMINAMATH_CALUDE_bus_average_speed_l1265_126523

/-- The average speed of a bus catching up to a bicycle -/
theorem bus_average_speed (bicycle_speed : ℝ) (initial_distance : ℝ) (catch_up_time : ℝ) :
  bicycle_speed = 15 →
  initial_distance = 195 →
  catch_up_time = 3 →
  (initial_distance + bicycle_speed * catch_up_time) / catch_up_time = 80 :=
by sorry

end NUMINAMATH_CALUDE_bus_average_speed_l1265_126523


namespace NUMINAMATH_CALUDE_arflaser_wavelength_scientific_notation_l1265_126536

theorem arflaser_wavelength_scientific_notation :
  ∀ (wavelength : ℝ),
  wavelength = 0.000000193 →
  ∃ (a : ℝ) (n : ℤ),
    wavelength = a * (10 : ℝ) ^ n ∧
    1 ≤ a ∧ a < 10 ∧
    a = 1.93 ∧ n = -7 :=
by sorry

end NUMINAMATH_CALUDE_arflaser_wavelength_scientific_notation_l1265_126536


namespace NUMINAMATH_CALUDE_condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1265_126518

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the planes α and β
variable (α β : Plane)

-- Theorem for condition ②
theorem condition_2 
  (h : ∀ l : Line, contains α l → line_parallel_plane l β) :
  parallel α β :=
sorry

-- Theorem for condition ④
theorem condition_4 
  (a b : Line)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : line_parallel a b) :
  parallel α β :=
sorry

-- Theorem for condition ①
theorem condition_1_not_sufficient 
  (h : ∃ S : Set Line, (∀ l ∈ S, contains α l ∧ line_parallel_plane l β) ∧ Set.Infinite S) :
  ¬(parallel α β → True) :=
sorry

-- Theorem for condition ③
theorem condition_3_not_sufficient 
  (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : line_parallel_plane a β)
  (h4 : line_parallel_plane b α) :
  ¬(parallel α β → True) :=
sorry

end NUMINAMATH_CALUDE_condition_2_condition_4_condition_1_not_sufficient_condition_3_not_sufficient_l1265_126518


namespace NUMINAMATH_CALUDE_probability_same_color_top_three_l1265_126547

def total_cards : ℕ := 52
def cards_per_color : ℕ := 26

theorem probability_same_color_top_three (total : ℕ) (per_color : ℕ) 
  (h1 : total = 52) 
  (h2 : per_color = 26) 
  (h3 : total = 2 * per_color) :
  (2 * (per_color.choose 3)) / (total.choose 3) = 12 / 51 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_top_three_l1265_126547


namespace NUMINAMATH_CALUDE_equation_solutions_l1265_126508

/-- The equation we want to solve -/
def equation (x : ℝ) : Prop :=
  (13*x - x^2) / (x + 1) * (x + (13 - x) / (x + 1)) = 42

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 1 ∨ x = 6 ∨ x = 3 + Real.sqrt 2 ∨ x = 3 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1265_126508


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1265_126583

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^3 > x^2) ↔ ∃ x : ℝ, x^3 ≤ x^2 := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1265_126583


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_10_l1265_126576

theorem least_subtraction_for_divisibility_by_10 :
  ∃ (n : ℕ), n = 2 ∧ 
  (427398 - n) % 10 = 0 ∧
  ∀ (m : ℕ), m < n → (427398 - m) % 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_10_l1265_126576


namespace NUMINAMATH_CALUDE_total_sheets_required_l1265_126566

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0 to 9) -/
def digit_count : ℕ := 10

/-- The number of sheets required for one character -/
def sheets_per_char : ℕ := 1

/-- Theorem: The total number of sheets required to write all uppercase and lowercase 
    English alphabets and digits from 0 to 9 is 62 -/
theorem total_sheets_required : 
  sheets_per_char * (2 * alphabet_size + digit_count) = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_required_l1265_126566


namespace NUMINAMATH_CALUDE_zoo_animals_ratio_l1265_126561

theorem zoo_animals_ratio (snakes monkeys lions pandas dogs : ℕ) : 
  snakes = 15 →
  monkeys = 2 * snakes →
  lions = monkeys - 5 →
  pandas = lions + 8 →
  snakes + monkeys + lions + pandas + dogs = 114 →
  dogs * 3 = pandas := by
sorry

end NUMINAMATH_CALUDE_zoo_animals_ratio_l1265_126561


namespace NUMINAMATH_CALUDE_thirtieth_term_is_351_l1265_126537

/-- Arithmetic sequence with first term 3 and common difference 12 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  3 + (n - 1) * 12

/-- The 30th term of the arithmetic sequence is 351 -/
theorem thirtieth_term_is_351 : arithmeticSequence 30 = 351 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_term_is_351_l1265_126537


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1265_126524

theorem solution_set_inequality (x : ℝ) : 
  (x * (x - 1) < 0) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1265_126524


namespace NUMINAMATH_CALUDE_grid_sum_invariant_l1265_126574

/-- Represents a 5x5 grid where each cell contains a natural number -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Represents a sequence of 25 moves to fill the grid -/
def MoveSequence := Fin 25 → Fin 5 × Fin 5

/-- Checks if two cells are adjacent in the grid -/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Generates a grid based on a move sequence -/
def generateGrid (moves : MoveSequence) : Grid :=
  sorry

/-- Calculates the sum of all numbers in a grid -/
def gridSum (g : Grid) : ℕ :=
  sorry

/-- The main theorem: the sum of all numbers in the grid is always 40 -/
theorem grid_sum_invariant (moves : MoveSequence) :
  gridSum (generateGrid moves) = 40 :=
  sorry

end NUMINAMATH_CALUDE_grid_sum_invariant_l1265_126574


namespace NUMINAMATH_CALUDE_chord_length_squared_l1265_126565

/-- Given three circles with radii 4, 7, and 9, where the circles with radii 4 and 7 
    are externally tangent to each other and internally tangent to the circle with radius 9, 
    the square of the length of the chord of the circle with radius 9 that is a common 
    external tangent to the other two circles is equal to 224. -/
theorem chord_length_squared (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : r₃ = 9) 
  (h_ext_tangent : r₃ = r₁ + r₂) 
  (h_int_tangent₁ : r₃ - r₁ = r₂) (h_int_tangent₂ : r₃ - r₂ = r₁) : 
  ∃ (chord_length : ℝ), chord_length^2 = 224 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l1265_126565


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l1265_126554

-- Define a decreasing function
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f) (h_inequality : f (m - 1) > f (2 * m - 1)) :
  m > 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l1265_126554


namespace NUMINAMATH_CALUDE_conference_games_count_l1265_126599

/-- The number of teams in Division A -/
def teams_a : Nat := 7

/-- The number of teams in Division B -/
def teams_b : Nat := 5

/-- The number of games each team plays against others in its division -/
def intra_division_games : Nat := 2

/-- The number of games each team plays against teams in the other division (excluding rivalry game) -/
def inter_division_games : Nat := 1

/-- The number of special pre-season rivalry games per team -/
def rivalry_games : Nat := 1

/-- The total number of conference games scheduled -/
def total_games : Nat := 
  -- Games within Division A
  (teams_a * (teams_a - 1) / 2) * intra_division_games +
  -- Games within Division B
  (teams_b * (teams_b - 1) / 2) * intra_division_games +
  -- Regular inter-division games
  teams_a * teams_b * inter_division_games +
  -- Special pre-season rivalry games
  teams_a * rivalry_games

theorem conference_games_count : total_games = 104 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_count_l1265_126599


namespace NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_line_l1265_126515

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (3, 2)

theorem parabola_midpoint_trajectory_and_line :
  -- Part 1: Prove that the trajectory E is y² = 4x
  (∀ x y : ℝ, (∃ x₀ y₀ : ℝ, parabola x₀ y₀ ∧ x = x₀ ∧ y = y₀/2) → trajectory_E x y) ∧
  -- Part 2: Prove that the line l passing through P and intersecting E at A and B (where P is the midpoint of AB) has the equation x - y - 1 = 0
  (∀ A B : ℝ × ℝ,
    let (x₁, y₁) := A
    let (x₂, y₂) := B
    trajectory_E x₁ y₁ ∧ 
    trajectory_E x₂ y₂ ∧ 
    x₁ + x₂ = 2 * point_P.1 ∧
    y₁ + y₂ = 2 * point_P.2 →
    line_l x₁ y₁ ∧ line_l x₂ y₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_trajectory_and_line_l1265_126515


namespace NUMINAMATH_CALUDE_cube_volume_after_cylinder_removal_l1265_126534

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_volume_after_cylinder_removal (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  let cube_volume := cube_side ^ 3
  let cylinder_volume := π * cylinder_radius ^ 2 * cube_side
  cube_volume - cylinder_volume = 216 - 54 * π := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_after_cylinder_removal_l1265_126534


namespace NUMINAMATH_CALUDE_even_function_extension_l1265_126555

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem even_function_extension
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x - x^4) :
  ∀ x > 0, f x = -x^4 - x :=
by sorry

end NUMINAMATH_CALUDE_even_function_extension_l1265_126555


namespace NUMINAMATH_CALUDE_f_greater_than_one_iff_l1265_126577

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_iff (x₀ : ℝ) :
  f x₀ > 1 ↔ x₀ < -1 ∨ x₀ > 1 := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_iff_l1265_126577


namespace NUMINAMATH_CALUDE_range_not_real_l1265_126517

/-- Given real numbers a and b satisfying ab = a + b + 3, 
    the range of (a-1)b is not equal to R. -/
theorem range_not_real : ¬ (∀ (y : ℝ), ∃ (a b : ℝ), a * b = a + b + 3 ∧ (a - 1) * b = y) := by
  sorry

end NUMINAMATH_CALUDE_range_not_real_l1265_126517


namespace NUMINAMATH_CALUDE_expression_rationality_expression_rationality_iff_l1265_126526

theorem expression_rationality (x : ℚ) : ∃ (k : ℚ), 
  x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k := by
  sorry

theorem expression_rationality_iff : 
  ∀ x : ℝ, (∃ k : ℚ, x^2 + (Real.sqrt (x^2 + 1))^2 - 1 / (x^2 + (Real.sqrt (x^2 + 1))^2) = k) ↔ 
  ∃ q : ℚ, x = q := by
  sorry

end NUMINAMATH_CALUDE_expression_rationality_expression_rationality_iff_l1265_126526


namespace NUMINAMATH_CALUDE_triangle_problem_l1265_126595

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) (h1 : t.a * Real.cos t.C - t.c * Real.sin t.A = 0)
    (h2 : t.b = 4) (h3 : (1/2) * t.a * t.b * Real.sin t.C = 6) :
    t.C = π/3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1265_126595


namespace NUMINAMATH_CALUDE_total_yardage_progress_l1265_126527

def team_a_moves : List Int := [-5, 8, -3, 6]
def team_b_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress : 
  (team_a_moves.sum + team_b_moves.sum) = 10 := by sorry

end NUMINAMATH_CALUDE_total_yardage_progress_l1265_126527


namespace NUMINAMATH_CALUDE_speedster_convertibles_l1265_126590

/-- The number of Speedster convertibles given the total number of vehicles,
    non-Speedsters, and the fraction of Speedsters that are convertibles. -/
theorem speedster_convertibles
  (total_vehicles : ℕ)
  (non_speedsters : ℕ)
  (speedster_convertible_fraction : ℚ)
  (h1 : total_vehicles = 80)
  (h2 : non_speedsters = 50)
  (h3 : speedster_convertible_fraction = 4/5) :
  (total_vehicles - non_speedsters) * speedster_convertible_fraction = 24 := by
  sorry

#eval (80 - 50) * (4/5 : ℚ)

end NUMINAMATH_CALUDE_speedster_convertibles_l1265_126590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1265_126510

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 50)
  (h_fifth : a 5 = 30) :
  a 7 = 10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l1265_126510


namespace NUMINAMATH_CALUDE_karens_cookies_l1265_126507

/-- Theorem: Karen's Cookies --/
theorem karens_cookies (
  kept_for_self : ℕ)
  (given_to_grandparents : ℕ)
  (class_size : ℕ)
  (cookies_per_person : ℕ)
  (h1 : kept_for_self = 10)
  (h2 : given_to_grandparents = 8)
  (h3 : class_size = 16)
  (h4 : cookies_per_person = 2)
  : kept_for_self + given_to_grandparents + class_size * cookies_per_person = 50 := by
  sorry

end NUMINAMATH_CALUDE_karens_cookies_l1265_126507


namespace NUMINAMATH_CALUDE_count_special_numbers_proof_l1265_126582

/-- The number of five-digit numbers with two pairs of adjacent equal digits,
    where digits from different pairs are different, and the remaining digit
    is different from all other digits. -/
def count_special_numbers : ℕ := 1944

/-- The set of valid configurations for the special five-digit numbers. -/
inductive Configuration : Type
  | AABBC : Configuration
  | AACBB : Configuration
  | CAABB : Configuration

/-- The number of possible choices for the first digit of the number. -/
def first_digit_choices : ℕ := 9

/-- The number of possible choices for the second digit of the number. -/
def second_digit_choices : ℕ := 9

/-- The number of possible choices for the third digit of the number. -/
def third_digit_choices : ℕ := 8

/-- The number of valid configurations. -/
def num_configurations : ℕ := 3

theorem count_special_numbers_proof :
  count_special_numbers =
    num_configurations * first_digit_choices * second_digit_choices * third_digit_choices :=
by sorry

end NUMINAMATH_CALUDE_count_special_numbers_proof_l1265_126582


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1265_126587

theorem negation_of_universal_statement :
  (¬ ∀ x ∈ Set.Icc 0 2, x^2 - 2*x ≤ 0) ↔ (∃ x ∈ Set.Icc 0 2, x^2 - 2*x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1265_126587


namespace NUMINAMATH_CALUDE_eight_digit_rotation_l1265_126525

def is_coprime (a b : Nat) : Prop := Nat.gcd a b = 1

def rotate_last_to_first (n : Nat) : Nat :=
  let d := n % 10
  let k := n / 10
  d * 10^7 + k

theorem eight_digit_rotation (A B : Nat) :
  (∃ B : Nat, 
    B > 44444444 ∧ 
    is_coprime B 12 ∧ 
    A = rotate_last_to_first B) →
  (A ≤ 99999998 ∧ A ≥ 14444446) :=
by sorry

end NUMINAMATH_CALUDE_eight_digit_rotation_l1265_126525


namespace NUMINAMATH_CALUDE_ratio_chain_l1265_126586

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hbc : b / c = 2 / 3)
  (hcd : c / d = 3 / 5) :
  a / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l1265_126586


namespace NUMINAMATH_CALUDE_infinitely_many_m_with_coprime_binomial_l1265_126511

theorem infinitely_many_m_with_coprime_binomial (k l : ℕ+) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ m ∈ S, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_m_with_coprime_binomial_l1265_126511


namespace NUMINAMATH_CALUDE_cos_difference_special_case_l1265_126533

theorem cos_difference_special_case (x₁ x₂ : Real) 
  (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 2 * Real.pi)
  (h4 : Real.sin x₁ = 1/3) (h5 : Real.sin x₂ = 1/3) : 
  Real.cos (x₁ - x₂) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_special_case_l1265_126533


namespace NUMINAMATH_CALUDE_base_6_addition_l1265_126589

def to_base_10 (n : ℕ) (base : ℕ) : ℕ := sorry

def from_base_10 (n : ℕ) (base : ℕ) : ℕ := sorry

theorem base_6_addition : 
  from_base_10 (to_base_10 5 6 + to_base_10 21 6) 6 = 30 := by sorry

end NUMINAMATH_CALUDE_base_6_addition_l1265_126589


namespace NUMINAMATH_CALUDE_digit_sum_19_or_20_l1265_126514

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_different (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def equation_holds (a b c d : ℕ) : Prop :=
  ∃ (x y z : ℕ), is_digit x ∧ is_digit y ∧ is_digit z ∧
  (a * 100 + 50 + b) + (400 + c * 10 + d) = x * 100 + y * 10 + z

theorem digit_sum_19_or_20 (a b c d : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
  are_different a b c d ∧
  equation_holds a b c d →
  a + b + c + d = 19 ∨ a + b + c + d = 20 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_19_or_20_l1265_126514


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l1265_126543

theorem smaller_root_of_equation : 
  let f (x : ℝ) := (x - 1/3)^2 + (x - 1/3)*(x - 2/3)
  ∃ y, f y = 0 ∧ y ≤ 1/3 ∧ ∀ z, f z = 0 → z ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l1265_126543


namespace NUMINAMATH_CALUDE_regression_line_equation_l1265_126512

/-- Regression line parameters -/
structure RegressionParams where
  x_bar : ℝ
  y_bar : ℝ
  slope : ℝ

/-- Regression line equation -/
def regression_line (params : RegressionParams) (x : ℝ) : ℝ :=
  params.slope * x + (params.y_bar - params.slope * params.x_bar)

/-- Theorem: Given the slope, x̄, and ȳ, prove the regression line equation -/
theorem regression_line_equation (params : RegressionParams)
  (h1 : params.x_bar = 4)
  (h2 : params.y_bar = 5)
  (h3 : params.slope = 2) :
  ∀ x, regression_line params x = 2 * x - 3 := by
  sorry

#check regression_line_equation

end NUMINAMATH_CALUDE_regression_line_equation_l1265_126512


namespace NUMINAMATH_CALUDE_cos_alpha_proof_l1265_126535

def angle_alpha : ℝ := sorry

def point_P : ℝ × ℝ := (-4, 3)

theorem cos_alpha_proof :
  point_P.1 = -4 ∧ point_P.2 = 3 →
  point_P ∈ {p : ℝ × ℝ | ∃ r : ℝ, r > 0 ∧ p = (r * Real.cos angle_alpha, r * Real.sin angle_alpha)} →
  Real.cos angle_alpha = -4/5 := by
  sorry

#check cos_alpha_proof

end NUMINAMATH_CALUDE_cos_alpha_proof_l1265_126535


namespace NUMINAMATH_CALUDE_cone_volume_l1265_126578

/-- The volume of a cone with slant height 15 cm and height 13 cm is (728/3)π cubic centimeters. -/
theorem cone_volume (π : ℝ) (slant_height height : ℝ) 
  (h1 : slant_height = 15)
  (h2 : height = 13) :
  (1/3 : ℝ) * π * (slant_height^2 - height^2) * height = (728/3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1265_126578


namespace NUMINAMATH_CALUDE_max_n_is_eleven_l1265_126516

/-- A coloring of integers from 1 to 14 with two colors -/
def Coloring := Fin 14 → Bool

/-- Check if there exist pairs of numbers with the same color and given difference -/
def hasPairsWithDifference (c : Coloring) (k : Nat) (color : Bool) : Prop :=
  ∃ i j, i < j ∧ j ≤ 14 ∧ j - i = k ∧ c i = color ∧ c j = color

/-- The property that a coloring satisfies the conditions for a given n -/
def validColoring (c : Coloring) (n : Nat) : Prop :=
  ∀ k, k ≤ n → hasPairsWithDifference c k true ∧ hasPairsWithDifference c k false

/-- The main theorem: the maximum possible n is 11 -/
theorem max_n_is_eleven :
  (∃ c : Coloring, validColoring c 11) ∧
  (∀ c : Coloring, ¬validColoring c 12) :=
sorry

end NUMINAMATH_CALUDE_max_n_is_eleven_l1265_126516


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1265_126532

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1265_126532


namespace NUMINAMATH_CALUDE_second_red_probability_l1265_126572

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ
  black : ℕ
  red : ℕ
  green : ℕ

/-- The probability of drawing a red marble as the second marble -/
def second_red_prob (bagA bagB bagC : Bag) : ℚ :=
  let total_A := bagA.white + bagA.black
  let total_B := bagB.red + bagB.green
  let total_C := bagC.red + bagC.green
  let prob_white_A := bagA.white / total_A
  let prob_black_A := bagA.black / total_A
  let prob_red_B := bagB.red / total_B
  let prob_red_C := bagC.red / total_C
  prob_white_A * prob_red_B + prob_black_A * prob_red_C

theorem second_red_probability :
  let bagA : Bag := { white := 4, black := 5, red := 0, green := 0 }
  let bagB : Bag := { white := 0, black := 0, red := 3, green := 7 }
  let bagC : Bag := { white := 0, black := 0, red := 5, green := 3 }
  second_red_prob bagA bagB bagC = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_second_red_probability_l1265_126572


namespace NUMINAMATH_CALUDE_petes_flag_shapes_petes_flag_total_shapes_l1265_126584

/-- Calculates the total number of shapes on Pete's flag based on US flag specifications -/
theorem petes_flag_shapes (us_stars : Nat) (us_stripes : Nat) : Nat :=
  let circles := us_stars / 2 - 3
  let squares := us_stripes * 2 + 6
  circles + squares

/-- Proves that the total number of shapes on Pete's flag is 54 -/
theorem petes_flag_total_shapes : 
  petes_flag_shapes 50 13 = 54 := by
  sorry

end NUMINAMATH_CALUDE_petes_flag_shapes_petes_flag_total_shapes_l1265_126584


namespace NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l1265_126522

-- Define the quadratic functions
def f (x : ℝ) := x^2 - 3*x - 4
def g (x : ℝ) := x^2 - x - 6

-- Define the solution sets
def S₁ : Set ℝ := {x | -1 < x ∧ x < 4}
def S₂ : Set ℝ := {x | x < -2 ∨ x > 3}

-- Theorem statements
theorem solution_set_f : {x : ℝ | f x < 0} = S₁ := by sorry

theorem solution_set_g : {x : ℝ | g x > 0} = S₂ := by sorry

end NUMINAMATH_CALUDE_solution_set_f_solution_set_g_l1265_126522


namespace NUMINAMATH_CALUDE_age_ratio_problem_l1265_126559

theorem age_ratio_problem (alma_age melina_age alma_score : ℕ) : 
  alma_age + melina_age = 2 * alma_score →
  melina_age = 60 →
  alma_score = 40 →
  melina_age / alma_age = 3 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l1265_126559


namespace NUMINAMATH_CALUDE_man_speed_against_current_l1265_126544

/-- A river with three sections and a man traveling along it -/
structure River :=
  (current_speed1 : ℝ)
  (current_speed2 : ℝ)
  (current_speed3 : ℝ)
  (man_speed_with_current1 : ℝ)

/-- Calculate the man's speed against the current in each section -/
def speed_against_current (r : River) : ℝ × ℝ × ℝ :=
  let speed_still_water := r.man_speed_with_current1 - r.current_speed1
  (speed_still_water - r.current_speed1,
   speed_still_water - r.current_speed2,
   speed_still_water - r.current_speed3)

/-- Theorem stating the man's speed against the current in each section -/
theorem man_speed_against_current (r : River) 
  (h1 : r.current_speed1 = 1.5)
  (h2 : r.current_speed2 = 2.5)
  (h3 : r.current_speed3 = 3.5)
  (h4 : r.man_speed_with_current1 = 25) :
  speed_against_current r = (22, 21, 20) :=
sorry


end NUMINAMATH_CALUDE_man_speed_against_current_l1265_126544


namespace NUMINAMATH_CALUDE_carnation_fraction_l1265_126563

def flower_bouquet (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) : Prop :=
  pink_roses + red_roses + pink_carnations + red_carnations = total ∧
  pink_roses + pink_carnations = (7/10) * total ∧
  pink_roses = (1/2) * (pink_roses + pink_carnations) ∧
  red_carnations = (5/6) * (red_roses + red_carnations)

theorem carnation_fraction (total : ℝ) (pink_roses red_roses pink_carnations red_carnations : ℝ) 
  (h : flower_bouquet total pink_roses red_roses pink_carnations red_carnations) :
  (pink_carnations + red_carnations) / total = 3/5 :=
sorry

end NUMINAMATH_CALUDE_carnation_fraction_l1265_126563


namespace NUMINAMATH_CALUDE_count_arrangements_l1265_126581

/-- The number of arrangements of 5 students (2 male and 3 female) in a line formation,
    where one specific male student does not stand at either end and only two of the
    three female students stand next to each other. -/
def num_arrangements : ℕ := 48

/-- Proves that the number of different possible arrangements is 48. -/
theorem count_arrangements :
  let total_students : ℕ := 5
  let male_students : ℕ := 2
  let female_students : ℕ := 3
  let specific_male_not_at_ends : Bool := true
  let two_females_adjacent : Bool := true
  num_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_count_arrangements_l1265_126581


namespace NUMINAMATH_CALUDE_investment_growth_l1265_126539

/-- Calculates the final amount after simple interest is applied --/
def final_amount (principal : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

/-- Proves that an investment of $1000 at 10% simple interest for 3 years results in $1300 --/
theorem investment_growth :
  final_amount 1000 (1/10) 3 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l1265_126539


namespace NUMINAMATH_CALUDE_equal_share_problem_l1265_126551

theorem equal_share_problem (total_amount : ℚ) (num_people : ℕ) :
  total_amount = 3.75 →
  num_people = 3 →
  total_amount / num_people = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_share_problem_l1265_126551


namespace NUMINAMATH_CALUDE_white_balls_count_l1265_126519

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  green = 30 ∧
  yellow = 10 ∧
  red = 47 ∧
  purple = 3 ∧
  prob_not_red_purple = 1/2 →
  total - (green + yellow + red + purple) = 10 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1265_126519


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1265_126556

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * (x + 5) + 3 * (y + 5) = 3 * S + 30 := by sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1265_126556


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1265_126570

/-- Given two regular polygons with the same perimeter, where the first polygon has 45 sides
    and a side length three times as long as the second, prove that the second polygon has 135 sides. -/
theorem second_polygon_sides (s : ℝ) (sides_second : ℕ) : 
  s > 0 →  -- Assume positive side length
  45 * (3 * s) = sides_second * s →  -- Same perimeter condition
  sides_second = 135 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1265_126570


namespace NUMINAMATH_CALUDE_initial_children_on_bus_l1265_126575

theorem initial_children_on_bus (children_off : ℕ) (children_on : ℕ) (final_children : ℕ) :
  children_off = 10 →
  children_on = 5 →
  final_children = 16 →
  final_children + (children_off - children_on) = 21 :=
by sorry

end NUMINAMATH_CALUDE_initial_children_on_bus_l1265_126575


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l1265_126552

theorem contrapositive_odd_sum_even :
  (¬(∃ (a b : ℤ), Odd a ∧ Odd b ∧ ¬(Even (a + b))) ↔
   (∀ (a b : ℤ), ¬(Even (a + b)) → ¬(Odd a ∧ Odd b))) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_even_l1265_126552


namespace NUMINAMATH_CALUDE_angle_cosine_in_3d_space_l1265_126545

/-- Given a point P(x, y, z) in the first octant of 3D space, if the cosines of the angles between OP
    and the x-axis (α) and y-axis (β) are 1/3 and 1/5 respectively, then the cosine of the angle
    between OP and the z-axis (γ) is √(191)/15. -/
theorem angle_cosine_in_3d_space (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) :
  let magnitude := Real.sqrt (x^2 + y^2 + z^2)
  (x / magnitude = 1 / 3) → (y / magnitude = 1 / 5) → (z / magnitude = Real.sqrt 191 / 15) := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_in_3d_space_l1265_126545


namespace NUMINAMATH_CALUDE_biology_class_percentage_l1265_126503

theorem biology_class_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 880 →
  not_enrolled = 572 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_biology_class_percentage_l1265_126503


namespace NUMINAMATH_CALUDE_polynomial_d_value_l1265_126594

/-- Represents a polynomial of degree 4 -/
structure Polynomial4 (α : Type) [Field α] where
  a : α
  b : α
  c : α
  d : α

/-- Calculates the sum of coefficients for a polynomial of degree 4 -/
def sumCoefficients {α : Type} [Field α] (p : Polynomial4 α) : α :=
  1 + p.a + p.b + p.c + p.d

/-- Calculates the mean of zeros for a polynomial of degree 4 -/
def meanZeros {α : Type} [Field α] (p : Polynomial4 α) : α :=
  -p.a / 4

/-- The main theorem -/
theorem polynomial_d_value
  {α : Type} [Field α]
  (p : Polynomial4 α)
  (h1 : meanZeros p = p.d)
  (h2 : p.d = sumCoefficients p)
  (h3 : p.d = 3) :
  p.d = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_d_value_l1265_126594


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l1265_126529

theorem christmas_tree_lights (T : ℝ) : ∃ (R Y G B : ℝ),
  R = 0.30 * T ∧
  Y = 0.45 * T ∧
  G = 110 ∧
  T = R + Y + G + B ∧
  B = 0.25 * T - 110 :=
by sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l1265_126529


namespace NUMINAMATH_CALUDE_lily_bushes_theorem_l1265_126513

theorem lily_bushes_theorem (bushes : Fin 19 → ℕ) : 
  ∃ i : Fin 19, Even ((bushes i) + (bushes ((i + 1) % 19))) := by
  sorry

end NUMINAMATH_CALUDE_lily_bushes_theorem_l1265_126513


namespace NUMINAMATH_CALUDE_race_completion_time_l1265_126502

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- The race scenario -/
def Race (a b : Runner) : Prop :=
  -- The race is 1000 meters long
  1000 = a.speed * a.time ∧
  -- A beats B by 40 meters
  960 = b.speed * a.time ∧
  -- A beats B by 10 seconds
  b.time = a.time + 10

/-- The theorem stating A's completion time -/
theorem race_completion_time (a b : Runner) (h : Race a b) : a.time = 250 := by
  sorry

end NUMINAMATH_CALUDE_race_completion_time_l1265_126502


namespace NUMINAMATH_CALUDE_monica_milk_amount_l1265_126558

-- Define the initial amount of milk Don has
def dons_milk : ℚ := 3/4

-- Define the fraction of milk Don gives to Rachel
def fraction_to_rachel : ℚ := 1/2

-- Define the fraction of Rachel's milk that Monica drinks
def fraction_monica_drinks : ℚ := 1/3

-- Theorem statement
theorem monica_milk_amount :
  fraction_monica_drinks * (fraction_to_rachel * dons_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_monica_milk_amount_l1265_126558


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1265_126560

theorem simultaneous_equations_solution :
  ∃ (x y : ℚ), 
    (3 * x - 2 * y = 12) ∧ 
    (9 * y - 6 * x = -18) ∧ 
    (x = 24/5) ∧ 
    (y = 6/5) := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1265_126560


namespace NUMINAMATH_CALUDE_TU_length_l1265_126585

-- Define the points
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry
def S : ℝ × ℝ := (16, 0)
def T : ℝ × ℝ := (16, 9.6)
def U : ℝ × ℝ := sorry

-- Define the triangles
def triangle_PQR : Set (ℝ × ℝ) := {P, Q, R}
def triangle_STU : Set (ℝ × ℝ) := {S, T, U}

-- Define the similarity of triangles
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem TU_length :
  similar_triangles triangle_PQR triangle_STU →
  distance Q R = 24 →
  distance P Q = 16 →
  distance P R = 19.2 →
  distance T U = 12.8 := by sorry

end NUMINAMATH_CALUDE_TU_length_l1265_126585


namespace NUMINAMATH_CALUDE_local_maximum_at_one_l1265_126500

/-- The function y = (x+1)/(x^2+3) has a local maximum at x = 1 -/
theorem local_maximum_at_one :
  ∃ δ > 0, ∀ x : ℝ, x ≠ 1 → |x - 1| < δ →
    (x + 1) / (x^2 + 3) ≤ (1 + 1) / (1^2 + 3) := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_one_l1265_126500


namespace NUMINAMATH_CALUDE_saline_mixture_proof_l1265_126568

def initial_volume : ℝ := 50
def initial_concentration : ℝ := 0.4
def added_concentration : ℝ := 0.1
def final_concentration : ℝ := 0.25
def added_volume : ℝ := 50

theorem saline_mixture_proof :
  (initial_volume * initial_concentration + added_volume * added_concentration) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_saline_mixture_proof_l1265_126568


namespace NUMINAMATH_CALUDE_larger_integer_value_l1265_126573

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  max a b = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l1265_126573


namespace NUMINAMATH_CALUDE_unique_assignment_l1265_126592

/-- Represents the digits assigned to letters --/
structure Assignment where
  A : Fin 5
  M : Fin 5
  E : Fin 5
  H : Fin 5
  Z : Fin 5
  N : Fin 5

/-- Checks if all assigned digits are different --/
def Assignment.allDifferent (a : Assignment) : Prop :=
  a.A ≠ a.M ∧ a.A ≠ a.E ∧ a.A ≠ a.H ∧ a.A ≠ a.Z ∧ a.A ≠ a.N ∧
  a.M ≠ a.E ∧ a.M ≠ a.H ∧ a.M ≠ a.Z ∧ a.M ≠ a.N ∧
  a.E ≠ a.H ∧ a.E ≠ a.Z ∧ a.E ≠ a.N ∧
  a.H ≠ a.Z ∧ a.H ≠ a.N ∧
  a.Z ≠ a.N

/-- Checks if the assignment satisfies the given inequalities --/
def Assignment.satisfiesInequalities (a : Assignment) : Prop :=
  3 > a.A.val + 1 ∧ 
  a.A.val + 1 > a.M.val + 1 ∧ 
  a.M.val + 1 < a.E.val + 1 ∧ 
  a.E.val + 1 < a.H.val + 1 ∧ 
  a.H.val + 1 < a.A.val + 1

/-- Checks if the assignment results in the correct ZAMENA number --/
def Assignment.correctZAMENA (a : Assignment) : Prop :=
  a.Z.val + 1 = 5 ∧
  a.A.val + 1 = 4 ∧
  a.M.val + 1 = 1 ∧
  a.E.val + 1 = 2 ∧
  a.N.val + 1 = 4 ∧
  a.H.val + 1 = 3

theorem unique_assignment :
  ∀ a : Assignment,
    a.allDifferent ∧ a.satisfiesInequalities → a.correctZAMENA :=
by sorry

end NUMINAMATH_CALUDE_unique_assignment_l1265_126592


namespace NUMINAMATH_CALUDE_equation_represents_parabola_l1265_126540

/-- The equation represents a parabola if it can be transformed into the form y = ax² + bx + c, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation |y - 3| = √((x+1)² + y²) -/
def given_equation (x y : ℝ) : Prop :=
  |y - 3| = Real.sqrt ((x + 1)^2 + y^2)

theorem equation_represents_parabola :
  ∃ f : ℝ → ℝ, (∀ x y, given_equation x y ↔ y = f x) ∧ is_parabola f :=
sorry

end NUMINAMATH_CALUDE_equation_represents_parabola_l1265_126540


namespace NUMINAMATH_CALUDE_partnership_capital_share_l1265_126542

theorem partnership_capital_share (Y : ℚ) : 
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y + (1 - ((1 / 3 : ℚ) + (1 / 4 : ℚ) + Y)) = 1 →
  (1 / 3 : ℚ) + (1 / 4 : ℚ) + Y = 1 →
  Y = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l1265_126542


namespace NUMINAMATH_CALUDE_xy_value_l1265_126550

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 8) : x * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1265_126550


namespace NUMINAMATH_CALUDE_smallest_value_operation_l1265_126596

theorem smallest_value_operation (a b : ℤ) (h1 : a = -3) (h2 : b = -6) :
  a + b ≤ min (a - b) (min (a * b) (a / b)) := by sorry

end NUMINAMATH_CALUDE_smallest_value_operation_l1265_126596


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1265_126509

theorem rational_equation_solution :
  ∃ y : ℝ, y ≠ 3 ∧ y ≠ (1/3) ∧
  (y^2 - 7*y + 12)/(y - 3) + (3*y^2 + 5*y - 8)/(3*y - 1) = -8 ∧
  y = -6 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1265_126509


namespace NUMINAMATH_CALUDE_x_value_theorem_l1265_126562

theorem x_value_theorem (x y : ℝ) :
  (x / (x + 2) = (y^2 + 3*y - 2) / (y^2 + 3*y + 1)) →
  x = (2*y^2 + 6*y - 4) / 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_theorem_l1265_126562


namespace NUMINAMATH_CALUDE_philips_school_days_l1265_126549

/-- Given the following conditions:
  - The distance from Philip's house to school is 2.5 miles
  - The distance from Philip's house to the market is 2 miles
  - Philip makes two round trips to school each day he goes to school
  - Philip makes one round trip to the market during weekends
  - Philip's car's mileage for a typical week is 44 miles

  Prove that Philip makes round trips to school 4 days a week. -/
theorem philips_school_days :
  ∀ (school_distance market_distance : ℚ)
    (daily_school_trips weekly_market_trips : ℕ)
    (weekly_mileage : ℚ),
  school_distance = 5/2 →
  market_distance = 2 →
  daily_school_trips = 2 →
  weekly_market_trips = 1 →
  weekly_mileage = 44 →
  ∃ (days : ℕ),
    days = 4 ∧
    weekly_mileage = (2 * school_distance * daily_school_trips * days : ℚ) + (2 * market_distance * weekly_market_trips) :=
by sorry

end NUMINAMATH_CALUDE_philips_school_days_l1265_126549


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1265_126588

-- Define the given line
def given_line (x y : ℝ) (c : ℝ) : Prop := x - 2 * y + c = 0

-- Define the point through which the perpendicular line passes
def point : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ (c : ℝ),
  (perpendicular_line point.1 point.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y →
    ∃ (m : ℝ), m * (x - point.1) = y - point.2 ∧
    m * (-1/2) = -1) ∧
  (∀ (x y : ℝ), given_line x y c →
    ∃ (m : ℝ), y = m * x + c / 2 ∧ m = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1265_126588


namespace NUMINAMATH_CALUDE_centers_form_square_l1265_126530

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Represents a square -/
structure Square :=
  (center : Point)
  (side_length : ℝ)

/-- Function to construct squares on the sides of a parallelogram -/
def construct_squares (p : Parallelogram) : Square × Square × Square × Square :=
  sorry

/-- Function to check if four points form a square -/
def is_square (p q r s : Point) : Prop :=
  sorry

/-- Theorem: The centers of squares constructed on the sides of a parallelogram form a square -/
theorem centers_form_square (p : Parallelogram) :
  let (sq1, sq2, sq3, sq4) := construct_squares p
  is_square sq1.center sq2.center sq3.center sq4.center :=
sorry

end NUMINAMATH_CALUDE_centers_form_square_l1265_126530


namespace NUMINAMATH_CALUDE_tank_capacity_l1265_126598

theorem tank_capacity (initial_fraction : ℚ) (added_gallons : ℚ) (final_fraction : ℚ) :
  initial_fraction = 3/4 →
  added_gallons = 9 →
  final_fraction = 7/8 →
  initial_fraction * C + added_gallons = final_fraction * C →
  C = 72 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1265_126598


namespace NUMINAMATH_CALUDE_permutations_of_four_l1265_126538

theorem permutations_of_four (n : ℕ) (h : n = 4) : Nat.factorial n = 24 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_four_l1265_126538


namespace NUMINAMATH_CALUDE_swimmer_speed_l1265_126504

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  man : ℝ  -- Speed of the man in still water (km/h)
  stream : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.man + s.stream else s.man - s.stream

/-- Theorem stating that given the conditions, the man's speed in still water is 15.5 km/h. -/
theorem swimmer_speed (s : SwimmerSpeeds) 
  (h1 : effectiveSpeed s true * 2 = 36)  -- Downstream condition
  (h2 : effectiveSpeed s false * 2 = 26) -- Upstream condition
  : s.man = 15.5 := by
  sorry

#check swimmer_speed

end NUMINAMATH_CALUDE_swimmer_speed_l1265_126504


namespace NUMINAMATH_CALUDE_negation_equivalence_l1265_126548

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1265_126548


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l1265_126557

/-- The transformation f that maps (x, y) to (x+2y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that the preimage of (3, 1) under f is (1, 1) -/
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l1265_126557


namespace NUMINAMATH_CALUDE_race_participants_race_result_l1265_126569

theorem race_participants (group_size : ℕ) (start_position : ℕ) (end_position : ℕ) : ℕ :=
  let total_groups := start_position + end_position - 1
  total_groups * group_size

theorem race_result : race_participants 3 7 5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_race_participants_race_result_l1265_126569


namespace NUMINAMATH_CALUDE_equation_solution_l1265_126593

theorem equation_solution :
  ∃ y : ℚ, (1 : ℚ) / 3 + 1 / y = 7 / 9 ↔ y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1265_126593


namespace NUMINAMATH_CALUDE_john_pennies_l1265_126571

/-- Given that Kate has 223 pennies and John has 165 more pennies than Kate,
    prove that John has 388 pennies. -/
theorem john_pennies (kate_pennies : ℕ) (john_extra : ℕ) 
    (h1 : kate_pennies = 223)
    (h2 : john_extra = 165) :
    kate_pennies + john_extra = 388 := by
  sorry

end NUMINAMATH_CALUDE_john_pennies_l1265_126571


namespace NUMINAMATH_CALUDE_vasya_number_digits_l1265_126506

theorem vasya_number_digits (x : ℝ) (h_pos : x > 0) 
  (h_kolya : 10^8 ≤ x^3 ∧ x^3 < 10^9) (h_petya : 10^10 ≤ x^4 ∧ x^4 < 10^11) :
  10^32 ≤ x^12 ∧ x^12 < 10^33 := by
  sorry

end NUMINAMATH_CALUDE_vasya_number_digits_l1265_126506


namespace NUMINAMATH_CALUDE_quadratic_inverse_sum_l1265_126567

/-- A quadratic function with real coefficients -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The inverse of a quadratic function -/
def InverseQuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ c * x^2 + b * x + a

theorem quadratic_inverse_sum (a b c : ℝ) :
  (∀ x, QuadraticFunction a b c (InverseQuadraticFunction a b c x) = x) →
  a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inverse_sum_l1265_126567


namespace NUMINAMATH_CALUDE_berry_collection_theorem_l1265_126546

def berry_collection (total_berries : ℕ) (sergey_speed_ratio : ℕ) : Prop :=
  let sergey_picked := (2 * total_berries) / 3
  let dima_picked := total_berries / 3
  let sergey_collected := sergey_picked / 2
  let dima_collected := (2 * dima_picked) / 3
  sergey_collected - dima_collected = 100

theorem berry_collection_theorem :
  berry_collection 900 2 := by
  sorry

end NUMINAMATH_CALUDE_berry_collection_theorem_l1265_126546


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_is_one_l1265_126528

theorem fraction_zero_implies_x_is_one (x : ℝ) :
  (x - 1) / (x - 3) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_is_one_l1265_126528


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1265_126580

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 → 
  (∃ (s : ℝ), 81 * s = b ∧ b * s = 8/27) → 
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1265_126580


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l1265_126553

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_powers_of_i_equals_zero :
  i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_powers_of_i_equals_zero_l1265_126553


namespace NUMINAMATH_CALUDE_max_distance_sum_l1265_126520

/-- Given m ∈ R, for points A on the line x + my = 0 and B on the line mx - y - m + 3 = 0,
    where these lines intersect at point P, the maximum value of |PA| + |PB| is 2√5. -/
theorem max_distance_sum (m : ℝ) : 
  ∃ (A B P : ℝ × ℝ), 
    (A.1 + m * A.2 = 0) ∧ 
    (m * B.1 - B.2 - m + 3 = 0) ∧ 
    (P.1 + m * P.2 = 0) ∧ 
    (m * P.1 - P.2 - m + 3 = 0) ∧
    (∀ (A' B' : ℝ × ℝ), 
      (A'.1 + m * A'.2 = 0) → 
      (m * B'.1 - B'.2 - m + 3 = 0) → 
      Real.sqrt ((P.1 - A'.1)^2 + (P.2 - A'.2)^2) + Real.sqrt ((P.1 - B'.1)^2 + (P.2 - B'.2)^2) ≤ 2 * Real.sqrt 5) ∧
    (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_sum_l1265_126520


namespace NUMINAMATH_CALUDE_probability_white_ball_l1265_126531

theorem probability_white_ball (n : ℕ) : 
  (2 : ℚ) / (n + 2) = 2 / 5 → (n : ℚ) / (n + 2) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l1265_126531
