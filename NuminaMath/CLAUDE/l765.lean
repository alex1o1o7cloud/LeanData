import Mathlib

namespace unique_solution_iff_n_eleven_l765_76589

/-- The equation x^2 - 3x + 5 = 0 has a unique solution in (ℤ_n, +, ·) if and only if n = 11 -/
theorem unique_solution_iff_n_eleven (n : ℕ) (hn : n ≥ 2) :
  (∃! x : ZMod n, x^2 - 3*x + 5 = 0) ↔ n = 11 := by sorry

end unique_solution_iff_n_eleven_l765_76589


namespace max_area_of_three_rectangles_l765_76506

/-- Given two rectangles with dimensions 9x12 and 10x15, 
    prove that the maximum area of a rectangle that can be formed 
    by arranging these two rectangles along with a third rectangle is 330. -/
theorem max_area_of_three_rectangles : 
  let rect1_width : ℝ := 9
  let rect1_height : ℝ := 12
  let rect2_width : ℝ := 10
  let rect2_height : ℝ := 15
  ∃ (rect3_width rect3_height : ℝ),
    (max 
      (max rect1_width rect2_width * (rect1_height + rect2_height))
      (max rect1_height rect2_height * (rect1_width + rect2_width))
    ) = 330 := by
  sorry

end max_area_of_three_rectangles_l765_76506


namespace closed_path_count_l765_76592

/-- The number of distinct closed paths on a grid with total length 2n -/
def num_closed_paths (n : ℕ) : ℕ := (Nat.choose (2 * n) n) ^ 2

/-- Theorem stating that the number of distinct closed paths on a grid
    with total length 2n is equal to (C_{2n}^n)^2 -/
theorem closed_path_count (n : ℕ) : 
  num_closed_paths n = (Nat.choose (2 * n) n) ^ 2 := by
  sorry

end closed_path_count_l765_76592


namespace daily_savings_amount_l765_76544

/-- Represents the number of days Ian saves money -/
def savingDays : ℕ := 40

/-- Represents the total amount saved in dimes -/
def totalSavedDimes : ℕ := 4

/-- Represents the value of a dime in cents -/
def dimeValueInCents : ℕ := 10

/-- Theorem: If Ian saves for 40 days and accumulates 4 dimes, his daily savings is 1 cent -/
theorem daily_savings_amount : 
  (totalSavedDimes * dimeValueInCents) / savingDays = 1 := by
  sorry

end daily_savings_amount_l765_76544


namespace line_through_circle_center_parallel_to_given_line_l765_76566

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y = 0

/-- The equation of the given line -/
def given_line (x y : ℝ) : Prop :=
  2*x - y = 0

/-- The equation of the line we need to prove -/
def target_line (x y : ℝ) : Prop :=
  2*x - y - 3 = 0

/-- Theorem stating that the line passing through the center of the circle
    and parallel to the given line has the equation 2x - y - 3 = 0 -/
theorem line_through_circle_center_parallel_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ (x y : ℝ), circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = cx^2 + cy^2) ∧
    (given_line cx cy → target_line cx cy) ∧
    (∀ (x y : ℝ), given_line x y → ∃ (k : ℝ), target_line (x + k) (y + 2*k)) :=
sorry

end line_through_circle_center_parallel_to_given_line_l765_76566


namespace parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l765_76574

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the conditions
def opposite_sides_parallel (q : Quadrilateral) : Prop := sorry
def opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def one_pair_parallel_and_equal (q : Quadrilateral) : Prop := sorry
def diagonals_equal (q : Quadrilateral) : Prop := sorry

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statements
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_parallel q → is_parallelogram q := by sorry

theorem equal_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_equal q → is_parallelogram q := by sorry

theorem one_pair_parallel_equal_implies_parallelogram (q : Quadrilateral) :
  one_pair_parallel_and_equal q → is_parallelogram q := by sorry

theorem equal_diagonals_implies_parallelogram (q : Quadrilateral) :
  diagonals_equal q → is_parallelogram q := by sorry

end parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l765_76574


namespace statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l765_76568

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (intersect_line : Line → Line → Prop)
variable (intersect_plane : Plane → Plane → Line)

-- Statement A
theorem statement_a_incorrect 
  (a b : Line) (α : Plane) :
  ∃ a b α, subset b α ∧ parallel a b ∧ ¬(parallel_line_plane a α) := by sorry

-- Statement B
theorem statement_b_correct 
  (a b : Line) (α β : Plane) :
  parallel_line_plane a α → intersect_plane α β = b → subset a β → parallel a b := by sorry

-- Statement C
theorem statement_c_correct 
  (a b : Line) (α β : Plane) (p : Line) :
  subset a α → subset b α → intersect_line a b → 
  parallel_line_plane a β → parallel_line_plane b β → 
  parallel_plane α β := by sorry

-- Statement D
theorem statement_d_correct 
  (a b : Line) (α β γ : Plane) :
  parallel_plane α β → intersect_plane α γ = a → intersect_plane β γ = b → 
  parallel a b := by sorry

end statement_a_incorrect_statement_b_correct_statement_c_correct_statement_d_correct_l765_76568


namespace common_divisors_8400_7560_l765_76570

theorem common_divisors_8400_7560 : Nat.card {d : ℕ | d ∣ 8400 ∧ d ∣ 7560} = 32 := by
  sorry

end common_divisors_8400_7560_l765_76570


namespace train_crossing_time_specific_train_crossing_time_l765_76573

/-- The time (in seconds) it takes for a train to cross a man walking in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let train_speed_ms := train_speed * 1000 / 3600
  let man_speed_ms := man_speed * 1000 / 3600
  let relative_speed := train_speed_ms - man_speed_ms
  train_length / relative_speed

/-- The specific problem instance --/
theorem specific_train_crossing_time :
  train_crossing_time 900 63 3 = 54 := by sorry

end train_crossing_time_specific_train_crossing_time_l765_76573


namespace virus_spread_l765_76536

def infection_rate : ℕ → ℕ
  | 0 => 1
  | n + 1 => infection_rate n * 9

theorem virus_spread (x : ℕ) :
  (∃ n : ℕ, infection_rate n = 81) →
  (∀ n : ℕ, infection_rate (n + 1) = infection_rate n * 9) →
  infection_rate 2 = 81 →
  infection_rate 3 > 700 :=
by sorry

#check virus_spread

end virus_spread_l765_76536


namespace quadratic_solution_set_implies_coefficients_l765_76551

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 + 5 * x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x : ℝ, f a c x > 0 ↔ 1/3 < x ∧ x < 1/2

-- Theorem statement
theorem quadratic_solution_set_implies_coefficients :
  ∀ a c : ℝ, solution_set a c → a = -6 ∧ c = -1 := by sorry

end quadratic_solution_set_implies_coefficients_l765_76551


namespace cubic_polynomial_with_arithmetic_progression_roots_l765_76522

/-- A cubic polynomial with coefficients in ℂ -/
structure CubicPolynomial where
  a : ℂ
  b : ℂ
  c : ℂ
  d : ℂ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_in_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), (r - d) * (r) * (r + d) = -p.d ∧
                (r - d) + r + (r + d) = p.b ∧
                (r - d) * r + (r - d) * (r + d) + r * (r + d) = p.c

/-- The roots of a cubic polynomial are not all real -/
def roots_not_all_real (p : CubicPolynomial) : Prop :=
  ∃ (r : ℂ), r.im ≠ 0 ∧ (r^3 + p.a * r^2 + p.b * r + p.c = 0)

theorem cubic_polynomial_with_arithmetic_progression_roots (a : ℝ) :
  let p := CubicPolynomial.mk 1 (-9) 42 a
  roots_in_arithmetic_progression p ∧ roots_not_all_real p → a = -72 := by
  sorry

end cubic_polynomial_with_arithmetic_progression_roots_l765_76522


namespace geometric_sequence_sixth_term_l765_76503

/-- Given a geometric sequence where the first term is 512 and the 8th term is 2,
    prove that the 6th term is 16. -/
theorem geometric_sequence_sixth_term
  (a : ℝ) -- First term
  (r : ℝ) -- Common ratio
  (h1 : a = 512) -- First term is 512
  (h2 : a * r^7 = 2) -- 8th term is 2
  : a * r^5 = 16 := by
  sorry


end geometric_sequence_sixth_term_l765_76503


namespace polynomial_simplification_l765_76581

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end polynomial_simplification_l765_76581


namespace license_plate_theorem_l765_76516

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of consonants (including Y) -/
def consonant_count : ℕ := alphabet_size - vowel_count

/-- The number of digits -/
def digit_count : ℕ := 10

/-- The number of possible license plates -/
def license_plate_count : ℕ := consonant_count * vowel_count * consonant_count * digit_count * vowel_count

theorem license_plate_theorem : license_plate_count = 110250 := by
  sorry

end license_plate_theorem_l765_76516


namespace geoffreys_birthday_money_l765_76548

/-- The amount of money Geoffrey received from his grandmother -/
def grandmothers_gift : ℤ := 70

/-- The amount of money Geoffrey received from his aunt -/
def aunts_gift : ℤ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncles_gift : ℤ := 30

/-- The total amount Geoffrey had in his wallet after receiving gifts -/
def total_in_wallet : ℤ := 125

/-- The cost of each video game -/
def game_cost : ℤ := 35

/-- The number of games Geoffrey bought -/
def number_of_games : ℤ := 3

/-- The amount of money Geoffrey had left after buying the games -/
def money_left : ℤ := 20

theorem geoffreys_birthday_money :
  grandmothers_gift + aunts_gift + uncles_gift = total_in_wallet - (game_cost * number_of_games - money_left) :=
by sorry

end geoffreys_birthday_money_l765_76548


namespace f_monotonicity_l765_76590

noncomputable def f (x : ℝ) : ℝ := -2 * x / (1 + x^2)

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < -1 → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) := by
  sorry

end f_monotonicity_l765_76590


namespace unique_solution_equation_l765_76578

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x - 6 / (x - 2) = 4 - 6 / (x - 2) :=
by sorry

end unique_solution_equation_l765_76578


namespace sin_cos_identity_l765_76591

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_identity_l765_76591


namespace fraction_equality_l765_76509

theorem fraction_equality : (2 - 4 + 8 - 16 + 32 + 64) / (4 - 8 + 16 - 32 + 64 + 128) = 1/2 := by
  sorry

end fraction_equality_l765_76509


namespace students_pets_difference_l765_76535

theorem students_pets_difference (num_classrooms : ℕ) (students_per_class : ℕ) (pets_per_class : ℕ)
  (h1 : num_classrooms = 5)
  (h2 : students_per_class = 20)
  (h3 : pets_per_class = 3) :
  num_classrooms * students_per_class - num_classrooms * pets_per_class = 85 := by
  sorry

end students_pets_difference_l765_76535


namespace sequence_general_term_l765_76587

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = n^2 - 4n,
    prove that the general term a_n is equal to 2n - 5. -/
theorem sequence_general_term (a : ℕ → ℤ) (S : ℕ → ℤ)
    (h : ∀ n : ℕ, S n = n^2 - 4*n) :
  ∀ n : ℕ, a n = 2*n - 5 :=
by sorry

end sequence_general_term_l765_76587


namespace special_polyhedron_sum_l765_76560

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex

/-- The theorem about the special polyhedron -/
theorem special_polyhedron_sum (p : SpecialPolyhedron) : 
  p.F = 30 ∧ 
  p.F = p.t + p.h ∧
  p.T = 3 ∧ 
  p.H = 2 ∧
  p.V - p.E + p.F = 2 ∧ 
  p.E = (3 * p.t + 6 * p.h) / 2 →
  100 * p.H + 10 * p.T + p.V = 262 := by
  sorry

end special_polyhedron_sum_l765_76560


namespace gold_copper_ratio_l765_76526

/-- Proves that the ratio of gold to copper in an alloy that is 17 times as heavy as water is 4:1,
    given that gold is 19 times as heavy as water and copper is 9 times as heavy as water. -/
theorem gold_copper_ratio (g c : ℝ) 
  (h1 : g > 0) 
  (h2 : c > 0) 
  (h_gold : 19 * g = 17 * (g + c)) 
  (h_copper : 9 * c = 17 * (g + c) - 19 * g) : 
  g / c = 4 := by
sorry

end gold_copper_ratio_l765_76526


namespace cherry_pie_degree_is_48_l765_76571

/-- Represents the pie preferences in a class --/
structure PiePreferences where
  total : ℕ
  chocolate : ℕ
  apple : ℕ
  blueberry : ℕ
  cherry_lemon_equal : Bool

/-- Calculates the degree for cherry pie in a pie chart --/
def cherry_pie_degree (prefs : PiePreferences) : ℕ :=
  let remaining := prefs.total - (prefs.chocolate + prefs.apple + prefs.blueberry)
  let cherry := (remaining + 1) / 2  -- Round up for cherry
  (cherry * 360) / prefs.total

/-- The main theorem stating the degree for cherry pie --/
theorem cherry_pie_degree_is_48 (prefs : PiePreferences) 
  (h1 : prefs.total = 45)
  (h2 : prefs.chocolate = 15)
  (h3 : prefs.apple = 10)
  (h4 : prefs.blueberry = 9)
  (h5 : prefs.cherry_lemon_equal = true) :
  cherry_pie_degree prefs = 48 := by
  sorry

#eval cherry_pie_degree ⟨45, 15, 10, 9, true⟩

end cherry_pie_degree_is_48_l765_76571


namespace fraction_meaningful_l765_76525

theorem fraction_meaningful (x : ℝ) : (1 : ℝ) / (x - 4) ≠ 0 ↔ x ≠ 4 := by
  sorry

end fraction_meaningful_l765_76525


namespace cleaner_flow_rate_l765_76508

/-- Represents the rate of cleaner flow through a pipe over time --/
structure CleanerFlow where
  initial_rate : ℝ
  middle_rate : ℝ
  final_rate : ℝ
  total_time : ℝ
  first_change_time : ℝ
  second_change_time : ℝ
  total_amount : ℝ

/-- The cleaner flow satisfies the problem conditions --/
def satisfies_conditions (flow : CleanerFlow) : Prop :=
  flow.initial_rate = 2 ∧
  flow.final_rate = 4 ∧
  flow.total_time = 30 ∧
  flow.first_change_time = 15 ∧
  flow.second_change_time = 25 ∧
  flow.total_amount = 80 ∧
  flow.initial_rate * flow.first_change_time +
  flow.middle_rate * (flow.second_change_time - flow.first_change_time) +
  flow.final_rate * (flow.total_time - flow.second_change_time) = flow.total_amount

theorem cleaner_flow_rate (flow : CleanerFlow) :
  satisfies_conditions flow → flow.middle_rate = 3 := by
  sorry


end cleaner_flow_rate_l765_76508


namespace negative_a_squared_times_b_over_a_squared_l765_76517

theorem negative_a_squared_times_b_over_a_squared (a b : ℝ) (h : a ≠ 0) :
  ((-a)^2 * b) / (a^2) = b := by sorry

end negative_a_squared_times_b_over_a_squared_l765_76517


namespace smallest_four_digit_divisible_by_digits_l765_76557

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10).filter (· ≠ 0) → n % d = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_digits :
  ∀ n, is_four_digit n → is_divisible_by_digits n → n ≥ 1362 :=
by sorry

end smallest_four_digit_divisible_by_digits_l765_76557


namespace complement_of_A_l765_76597

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

theorem complement_of_A : (Aᶜ : Set ℝ) = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

end complement_of_A_l765_76597


namespace testicular_cell_properties_l765_76534

-- Define the possible bases
inductive Base
| A
| C
| T

-- Define the possible cell cycle periods
inductive Period
| Interphase
| EarlyMitosis
| LateMitosis
| EarlyMeiosis1
| LateMeiosis1
| EarlyMeiosis2
| LateMeiosis2

-- Define the structure of a testicular cell
structure TesticularCell where
  nucleotideTypes : Finset (List Base)
  lowestStabilityPeriod : Period
  dnaSeperationPeriod : Period

-- Define the theorem
theorem testicular_cell_properties : ∃ (cell : TesticularCell),
  (cell.nucleotideTypes.card = 3) ∧
  (cell.lowestStabilityPeriod = Period.Interphase) ∧
  (cell.dnaSeperationPeriod = Period.LateMeiosis1 ∨ cell.dnaSeperationPeriod = Period.LateMeiosis2) :=
by
  sorry

end testicular_cell_properties_l765_76534


namespace m_range_l765_76554

open Set

def A : Set ℝ := {x : ℝ | |x - 1| + |x + 1| ≤ 3}

def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - (2*m + 1)*x + m^2 + m < 0}

theorem m_range (m : ℝ) : (A ∩ B m).Nonempty → m ∈ Set.Ioo (-5/2) (3/2) := by
  sorry

end m_range_l765_76554


namespace box_volume_increase_l765_76539

-- Define the properties of the rectangular box
def rectangular_box (l w h : ℝ) : Prop :=
  l * w * h = 5400 ∧
  2 * (l * w + w * h + h * l) = 2352 ∧
  4 * (l + w + h) = 240

-- State the theorem
theorem box_volume_increase (l w h : ℝ) :
  rectangular_box l w h →
  (l + 2) * (w + 2) * (h + 2) = 8054 :=
by
  sorry

end box_volume_increase_l765_76539


namespace x_intercept_distance_l765_76577

/-- Given two lines intersecting at (8, 20) with slopes 4 and -2,
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -2 * x + 36) →
  line1 8 = 20 →
  line2 8 = 20 →
  |line1⁻¹ 0 - line2⁻¹ 0| = 15 :=
sorry

end x_intercept_distance_l765_76577


namespace short_bar_length_l765_76523

theorem short_bar_length (total_length long_short_diff : ℝ) 
  (h1 : total_length = 950)
  (h2 : long_short_diff = 150) :
  let short_bar := (total_length - long_short_diff) / 2
  short_bar = 400 := by
sorry

end short_bar_length_l765_76523


namespace total_cost_is_180_l765_76530

/-- The cost to fill all planter pots at the corners of a rectangle-shaped pool -/
def total_cost : ℝ :=
  let palm_fern_cost : ℝ := 15.00
  let creeping_jenny_cost : ℝ := 4.00
  let geranium_cost : ℝ := 3.50
  let plants_per_pot : ℕ := 1 + 4 + 4
  let cost_per_pot : ℝ := palm_fern_cost + 4 * creeping_jenny_cost + 4 * geranium_cost
  let corners : ℕ := 4
  corners * cost_per_pot

/-- Theorem stating that the total cost to fill all planter pots is $180.00 -/
theorem total_cost_is_180 : total_cost = 180.00 := by
  sorry

end total_cost_is_180_l765_76530


namespace reverse_difference_for_253_l765_76501

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  h_range := n.o_range
  t_range := n.t_range
  o_range := n.h_range

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem reverse_difference_for_253 (n : ThreeDigitNumber) 
    (h_253 : n.toNat = 253)
    (h_sum : n.sumOfDigits = 10)
    (h_middle : n.tens = n.hundreds + n.ones) :
    (n.reverse.toNat - n.toNat) = 99 := by
  sorry

#check reverse_difference_for_253

end reverse_difference_for_253_l765_76501


namespace cooper_savings_l765_76585

theorem cooper_savings (total_savings : ℕ) (days_in_year : ℕ) (daily_savings : ℕ) :
  total_savings = 12410 →
  days_in_year = 365 →
  daily_savings * days_in_year = total_savings →
  daily_savings = 34 := by
  sorry

end cooper_savings_l765_76585


namespace prime_square_minus_one_div_24_l765_76595

theorem prime_square_minus_one_div_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) : 
  24 ∣ (p^2 - 1) := by
sorry

end prime_square_minus_one_div_24_l765_76595


namespace moss_pollen_scientific_notation_l765_76559

/-- The diameter of a moss flower's pollen in meters -/
def moss_pollen_diameter : ℝ := 0.0000084

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Theorem stating that the moss pollen diameter is equal to its scientific notation representation -/
theorem moss_pollen_scientific_notation :
  ∃ (sn : ScientificNotation), moss_pollen_diameter = sn.coefficient * (10 : ℝ) ^ sn.exponent ∧
  sn.coefficient = 8.4 ∧ sn.exponent = -6 := by
  sorry

end moss_pollen_scientific_notation_l765_76559


namespace cos_pi_4_plus_alpha_l765_76504

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (α - π/4) = 1/3) : 
  Real.cos (π/4 + α) = -1/3 := by
sorry

end cos_pi_4_plus_alpha_l765_76504


namespace edward_picked_three_l765_76569

/-- The number of pieces of paper Olivia picked up -/
def olivia_pieces : ℕ := 16

/-- The total number of pieces of paper picked up by Olivia and Edward -/
def total_pieces : ℕ := 19

/-- The number of pieces of paper Edward picked up -/
def edward_pieces : ℕ := total_pieces - olivia_pieces

theorem edward_picked_three : edward_pieces = 3 := by
  sorry

end edward_picked_three_l765_76569


namespace equation_roots_l765_76552

def equation (x : ℝ) : ℝ := x * (x + 2)^2 * (3 - x) * (5 + x)

theorem equation_roots : 
  {x : ℝ | equation x = 0} = {0, -2, 3, -5} := by sorry

end equation_roots_l765_76552


namespace min_a_value_l765_76582

/-- Set A defined by the quadratic inequality -/
def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (1 - a) * p.1^2 + 2 * p.1 * p.2 - a * p.2^2 ≤ 0}

/-- Set B defined by the linear inequality and positivity conditions -/
def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 ≥ 0 ∧ p.1 > 0 ∧ p.2 > 0}

/-- Theorem stating the minimum value of a given the subset relationship -/
theorem min_a_value (h : set_B ⊆ set_A a) : a ≥ 55 / 34 := by
  sorry

#check min_a_value

end min_a_value_l765_76582


namespace diagonal_length_of_courtyard_l765_76593

/-- Represents a rectangular courtyard with sides in ratio 4:3 -/
structure Courtyard where
  length : ℝ
  width : ℝ
  ratio_constraint : length = (4/3) * width

/-- The cost of paving in Rupees per square meter -/
def paving_cost_per_sqm : ℝ := 0.5

/-- The total cost of paving the courtyard in Rupees -/
def total_paving_cost : ℝ := 600

theorem diagonal_length_of_courtyard (c : Courtyard) : 
  c.length * c.width * paving_cost_per_sqm = total_paving_cost →
  Real.sqrt (c.length^2 + c.width^2) = 50 := by
  sorry

end diagonal_length_of_courtyard_l765_76593


namespace second_coin_value_l765_76575

/-- Proves that the value of the second type of coin is 0.5 rupees -/
theorem second_coin_value (total_value : ℝ) (num_coins : ℕ) (coin1_value : ℝ) (coin3_value : ℝ) :
  total_value = 35 →
  num_coins = 20 →
  coin1_value = 1 →
  coin3_value = 0.25 →
  ∃ (coin2_value : ℝ), 
    coin2_value = 0.5 ∧
    num_coins * (coin1_value + coin2_value + coin3_value) = total_value :=
by sorry

end second_coin_value_l765_76575


namespace smallest_number_proof_l765_76515

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 30 →
  b = 25 →
  c = b + 7 →
  a ≤ b ∧ b ≤ c →
  a = 33 := by
sorry

end smallest_number_proof_l765_76515


namespace work_completion_time_l765_76563

/-- The time it takes to complete a work given two workers with different rates and a specific work schedule. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_solo_days : ℝ) 
  (hp : p_rate = total_work / 10) 
  (hq : q_rate = total_work / 6) 
  (hp_solo : p_solo_days = 2) : 
  p_solo_days + (total_work - p_solo_days * p_rate) / (p_rate + q_rate) = 5 := by
  sorry

end work_completion_time_l765_76563


namespace other_focus_coordinates_l765_76519

/-- A hyperbola with given axes of symmetry and one focus on the y-axis -/
structure Hyperbola where
  x_axis : ℝ
  y_axis : ℝ
  focus_on_y_axis : ℝ × ℝ

/-- The other focus of the hyperbola -/
def other_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem stating that the other focus has coordinates (-2, 2) -/
theorem other_focus_coordinates (h : Hyperbola) 
  (hx : h.x_axis = -1)
  (hy : h.y_axis = 2)
  (hf : h.focus_on_y_axis.1 = 0 ∧ h.focus_on_y_axis.2 = 2) :
  other_focus h = (-2, 2) := by sorry

end other_focus_coordinates_l765_76519


namespace polynomial_factorization_l765_76583

theorem polynomial_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * ((a + b) * a^2 * b^2 + (b + c) * b^2 * c^2 + (a + c) * c^2 * a) := by
  sorry

end polynomial_factorization_l765_76583


namespace imaginary_part_of_complex_fraction_l765_76537

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9/5 := by sorry

end imaginary_part_of_complex_fraction_l765_76537


namespace trigonometric_identity_l765_76562

theorem trigonometric_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.cos (x + y) ^ 2 + 2 * Real.sin x * Real.sin y * Real.cos (x + y) = 1 + Real.cos y ^ 2 := by
  sorry

end trigonometric_identity_l765_76562


namespace green_square_area_percentage_l765_76561

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSide : ℝ

/-- The cross is symmetric and occupies 49% of the flag's area -/
def isValidCrossFlag (flag : CrossFlag) : Prop :=
  flag.crossArea = 0.49 * flag.side^2 ∧
  flag.greenSide = 2 * flag.crossWidth

/-- Theorem: The green square occupies 6.01% of the flag's area -/
theorem green_square_area_percentage (flag : CrossFlag) 
  (h : isValidCrossFlag flag) : 
  (flag.greenSide^2) / (flag.side^2) = 0.0601 := by
  sorry

end green_square_area_percentage_l765_76561


namespace pet_store_cats_l765_76528

theorem pet_store_cats (white_cats black_cats total_cats : ℕ) 
  (h1 : white_cats = 2)
  (h2 : black_cats = 10)
  (h3 : total_cats = 15)
  : total_cats - (white_cats + black_cats) = 3 := by
  sorry

end pet_store_cats_l765_76528


namespace system_three_solutions_l765_76532

/-- The system of equations has exactly three solutions if and only if a = 9 or a = 23 + 4√15 -/
theorem system_three_solutions (a : ℝ) :
  (∃! x y z : ℝ × ℝ, 
    ((abs (y.2 + 9) + abs (x.1 + 2) - 2) * (x.1^2 + x.2^2 - 3) = 0 ∧
     (x.1 + 2)^2 + (x.2 + 4)^2 = a) ∧
    ((abs (y.2 + 9) + abs (y.1 + 2) - 2) * (y.1^2 + y.2^2 - 3) = 0 ∧
     (y.1 + 2)^2 + (y.2 + 4)^2 = a) ∧
    ((abs (z.2 + 9) + abs (z.1 + 2) - 2) * (z.1^2 + z.2^2 - 3) = 0 ∧
     (z.1 + 2)^2 + (z.2 + 4)^2 = a) ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔
  (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) :=
by sorry

end system_three_solutions_l765_76532


namespace fraction_equality_l765_76521

theorem fraction_equality (a b : ℚ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end fraction_equality_l765_76521


namespace cube_edge_length_l765_76567

theorem cube_edge_length (l w h : ℝ) (cube_edge : ℝ) : 
  l = 2 → w = 4 → h = 8 → l * w * h = cube_edge^3 → cube_edge = 4 := by
  sorry

end cube_edge_length_l765_76567


namespace investment_interest_rate_l765_76543

theorem investment_interest_rate 
  (total_investment : ℝ) 
  (rate1 rate2 : ℝ) 
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.05)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (part1 part2 : ℝ), 
    part1 + part2 = total_investment ∧ 
    part1 * rate1 = part2 * rate2) :
  (rate1 * (total_investment - (rate2 * total_investment) / (rate1 + rate2)) + 
   rate2 * ((rate1 * total_investment) / (rate1 + rate2))) / total_investment = 0.05833 :=
by sorry

end investment_interest_rate_l765_76543


namespace carmen_total_sales_l765_76531

/-- Represents the sales to a house -/
structure HouseSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ
  samoasPrice : ℚ
  thinMintsPrice : ℚ
  fudgeDelightsPrice : ℚ
  sugarCookiesPrice : ℚ

/-- Calculates the total sales for a house -/
def houseSalesTotal (sales : HouseSales) : ℚ :=
  sales.samoas * sales.samoasPrice +
  sales.thinMints * sales.thinMintsPrice +
  sales.fudgeDelights * sales.fudgeDelightsPrice +
  sales.sugarCookies * sales.sugarCookiesPrice

/-- Represents Carmen's total sales -/
def carmenSales : List HouseSales :=
  [
    { samoas := 3, thinMints := 0, fudgeDelights := 0, sugarCookies := 0,
      samoasPrice := 4, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 2, fudgeDelights := 1, sugarCookies := 0,
      samoasPrice := 0, thinMintsPrice := 7/2, fudgeDelightsPrice := 5, sugarCookiesPrice := 0 },
    { samoas := 0, thinMints := 0, fudgeDelights := 0, sugarCookies := 9,
      samoasPrice := 0, thinMintsPrice := 0, fudgeDelightsPrice := 0, sugarCookiesPrice := 2 }
  ]

theorem carmen_total_sales :
  (carmenSales.map houseSalesTotal).sum = 42 := by
  sorry

end carmen_total_sales_l765_76531


namespace interest_difference_approx_l765_76540

-- Define the initial deposit
def initial_deposit : ℝ := 12000

-- Define the interest rates
def compound_rate : ℝ := 0.06
def simple_rate : ℝ := 0.08

-- Define the time period
def years : ℕ := 20

-- Define the compound interest function
def compound_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + r) ^ n

-- Define the simple interest function
def simple_balance (p r : ℝ) (n : ℕ) : ℝ := p * (1 + n * r)

-- State the theorem
theorem interest_difference_approx :
  ∃ (ε : ℝ), ε < 1 ∧ 
  |round (compound_balance initial_deposit compound_rate years - 
          simple_balance initial_deposit simple_rate years) - 7286| ≤ ε :=
sorry

end interest_difference_approx_l765_76540


namespace cubic_monotonicity_l765_76527

def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem cubic_monotonicity 
  (a b c d : ℝ) 
  (h1 : f a b c d 0 = -4)
  (h2 : f' a b c 0 = 12)
  (h3 : f a b c d 2 = 0)
  (h4 : f' a b c 2 = 0) :
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = 2 ∧
  (∀ x < x₁, f' a b c x > 0) ∧
  (∀ x ∈ Set.Ioo x₁ x₂, f' a b c x < 0) ∧
  (∀ x > x₂, f' a b c x > 0) :=
sorry

end cubic_monotonicity_l765_76527


namespace arithmetic_sequence_formula_l765_76584

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

-- State the theorem
theorem arithmetic_sequence_formula 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h1 : a 3 + a 4 = 4)
  (h2 : a 5 + a 7 = 6) :
  ∃ C : ℚ, ∀ n : ℕ, a n = (2 * n + C) / 5 :=
sorry

end arithmetic_sequence_formula_l765_76584


namespace mod_congruence_unique_solution_l765_76512

theorem mod_congruence_unique_solution : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ -300 ≡ n [ZMOD 23] := by
  sorry

end mod_congruence_unique_solution_l765_76512


namespace f_min_and_inequality_l765_76555

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := x / exp x - 2 / exp 1

theorem f_min_and_inequality :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x) ∧
  (∀ (x : ℝ), x > 0 → f x ≥ -1 / exp 1) ∧
  (∀ (m n : ℝ), m > 0 → n > 0 → f m ≥ g n) :=
sorry

end f_min_and_inequality_l765_76555


namespace carpet_fits_rooms_l765_76518

/-- Represents a rectangular room --/
structure Room where
  width : ℕ
  length : ℕ

/-- Represents a rectangular carpet --/
structure Carpet where
  width : ℕ
  length : ℕ

/-- Checks if a carpet fits perfectly in a room --/
def fitsPerectly (c : Carpet) (r : Room) : Prop :=
  c.width ^ 2 + c.length ^ 2 = r.width ^ 2 + r.length ^ 2

theorem carpet_fits_rooms :
  ∃ (c : Carpet) (r1 r2 : Room),
    c.width = 25 ∧
    c.length = 50 ∧
    r1.width = 38 ∧
    r2.width = 50 ∧
    r1.length = r2.length ∧
    fitsPerectly c r1 ∧
    fitsPerectly c r2 := by
  sorry

end carpet_fits_rooms_l765_76518


namespace cyclist_meeting_oncoming_buses_l765_76507

/-- The time interval between a cyclist meeting oncoming buses, given constant speeds and specific time intervals -/
theorem cyclist_meeting_oncoming_buses 
  (overtake_interval : ℝ) 
  (bus_interval : ℝ) 
  (h1 : overtake_interval > 0)
  (h2 : bus_interval > 0)
  (h3 : bus_interval = overtake_interval / 2) :
  overtake_interval / 2 = bus_interval := by
sorry

end cyclist_meeting_oncoming_buses_l765_76507


namespace exponent_identities_l765_76586

theorem exponent_identities (x a : ℝ) (h : a ≠ 0) : 
  (3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6) ∧ 
  (a^3 * a + (-a^2)^3 / a^2 = 0) := by sorry

end exponent_identities_l765_76586


namespace power_seven_mod_nineteen_l765_76547

theorem power_seven_mod_nineteen : 7^2023 ≡ 4 [ZMOD 19] := by
  sorry

end power_seven_mod_nineteen_l765_76547


namespace work_completion_time_l765_76579

/-- Represents the time it takes for A to complete the work alone -/
def time_A : ℝ := 15

/-- Represents the time it takes for B to complete the work alone -/
def time_B : ℝ := 27

/-- Represents the total amount of work -/
def total_work : ℝ := 1

/-- Represents the number of days A works before leaving -/
def days_A_worked : ℝ := 5

/-- Represents the number of days B works to complete the remaining work -/
def days_B_worked : ℝ := 18

theorem work_completion_time :
  (days_A_worked / time_A) + (days_B_worked / time_B) = total_work ∧
  time_A = total_work / ((total_work - (days_B_worked / time_B)) / days_A_worked) :=
sorry

end work_completion_time_l765_76579


namespace sum_of_three_numbers_l765_76599

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149) 
  (sum_of_products : a*b + b*c + a*c = 70) : 
  a + b + c = 17 := by
  sorry

end sum_of_three_numbers_l765_76599


namespace logarithm_equality_l765_76594

theorem logarithm_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 4*y^2 = 12*x*y) :
  Real.log (x + 2*y) / Real.log 10 - 2 * (Real.log 2 / Real.log 10) = 
  (1/2) * (Real.log x / Real.log 10 + Real.log y / Real.log 10) := by
  sorry

end logarithm_equality_l765_76594


namespace percent_exceeding_speed_limit_l765_76524

theorem percent_exceeding_speed_limit 
  (total_motorists : ℕ) 
  (h_total_positive : total_motorists > 0)
  (percent_ticketed : ℝ) 
  (h_percent_ticketed : percent_ticketed = 10)
  (percent_unticketed_speeders : ℝ) 
  (h_percent_unticketed : percent_unticketed_speeders = 50) : 
  (percent_ticketed * total_motorists / 100 + 
   percent_ticketed * total_motorists / 100) / total_motorists * 100 = 20 := by
  sorry

#check percent_exceeding_speed_limit

end percent_exceeding_speed_limit_l765_76524


namespace minimum_spend_equal_fruits_l765_76513

/-- Represents a fruit set with apples, oranges, and cost -/
structure FruitSet where
  apples : ℕ
  oranges : ℕ
  cost : ℕ

/-- Calculates the total cost of buying multiple fruit sets -/
def totalCost (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.cost * quantity

/-- Calculates the total number of apples in multiple fruit sets -/
def totalApples (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.apples * quantity

/-- Calculates the total number of oranges in multiple fruit sets -/
def totalOranges (set : FruitSet) (quantity : ℕ) : ℕ :=
  set.oranges * quantity

theorem minimum_spend_equal_fruits : 
  let set1 : FruitSet := ⟨3, 15, 360⟩
  let set2 : FruitSet := ⟨20, 5, 500⟩
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    totalApples set1 x + totalApples set2 y = totalOranges set1 x + totalOranges set2 y ∧
    ∀ (a b : ℕ), 
      (a > 0 ∧ b > 0 ∧ 
       totalApples set1 a + totalApples set2 b = totalOranges set1 a + totalOranges set2 b) →
      totalCost set1 x + totalCost set2 y ≤ totalCost set1 a + totalCost set2 b ∧
    totalCost set1 x + totalCost set2 y = 3800 :=
by
  sorry


end minimum_spend_equal_fruits_l765_76513


namespace siblings_selection_probability_l765_76564

/-- The probability of three siblings being selected simultaneously -/
theorem siblings_selection_probability (px py pz : ℚ) 
  (hx : px = 1/7) (hy : py = 2/9) (hz : pz = 3/11) : 
  px * py * pz = 1/115.5 := by
  sorry

end siblings_selection_probability_l765_76564


namespace ellipse_point_properties_l765_76500

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-4, 0)
def right_focus : ℝ × ℝ := (4, 0)

-- Define the angle between PF₁ and PF₂
def angle_PF1F2 (P : ℝ × ℝ) : ℝ := 60

-- Theorem statement
theorem ellipse_point_properties (P : ℝ × ℝ) 
  (h_on_ellipse : is_on_ellipse P.1 P.2) 
  (h_angle : angle_PF1F2 P = 60) :
  (∃ (S : ℝ), S = 3 * Real.sqrt 3 ∧ 
    S = (1/2) * Real.sqrt ((P.1 - left_focus.1)^2 + (P.2 - left_focus.2)^2) *
              Real.sqrt ((P.1 - right_focus.1)^2 + (P.2 - right_focus.2)^2) *
              Real.sin (angle_PF1F2 P * π / 180)) ∧
  (P.1 = 5 * Real.sqrt 13 / 4 ∨ P.1 = -5 * Real.sqrt 13 / 4) ∧
  (P.2 = 4 * Real.sqrt 3 / 4 ∨ P.2 = -4 * Real.sqrt 3 / 4) := by
sorry

end ellipse_point_properties_l765_76500


namespace sum_of_altitudes_is_23_and_one_seventh_l765_76514

/-- A triangle formed by the line 18x + 9y = 108 and the coordinate axes -/
structure Triangle where
  -- The line equation
  line_eq : ℝ → ℝ → Prop := fun x y => 18 * x + 9 * y = 108
  -- The triangle is formed with coordinate axes
  forms_triangle : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ line_eq a 0 ∧ line_eq 0 b

/-- The sum of the lengths of the altitudes of the triangle -/
def sum_of_altitudes (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the sum of the altitudes is 23 1/7 -/
theorem sum_of_altitudes_is_23_and_one_seventh (t : Triangle) :
  sum_of_altitudes t = 23 + 1 / 7 := by
  sorry

end sum_of_altitudes_is_23_and_one_seventh_l765_76514


namespace trivia_team_groups_l765_76520

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : not_picked = 10)
  (h3 : students_per_group = 6) :
  (total_students - not_picked) / students_per_group = 8 := by
  sorry

end trivia_team_groups_l765_76520


namespace largest_non_sum_30_composite_l765_76588

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n > 93, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 93 :=
sorry

end largest_non_sum_30_composite_l765_76588


namespace solve_aunt_gift_problem_l765_76558

def aunt_gift_problem (jade_initial : ℕ) (julia_initial : ℕ) (total_final : ℕ) : Prop :=
  let total_initial := jade_initial + julia_initial
  let total_gift := total_final - total_initial
  let gift_per_person := total_gift / 2
  (jade_initial = 38) ∧
  (julia_initial = jade_initial / 2) ∧
  (total_final = 97) ∧
  (gift_per_person = 20)

theorem solve_aunt_gift_problem :
  ∃ (jade_initial julia_initial total_final : ℕ),
    aunt_gift_problem jade_initial julia_initial total_final :=
by
  sorry

end solve_aunt_gift_problem_l765_76558


namespace percent_equality_l765_76538

theorem percent_equality (x y : ℝ) (P : ℝ) (h1 : y = 0.25 * x) 
  (h2 : (P / 100) * (x - y) = 0.15 * (x + y)) : P = 25 := by
  sorry

end percent_equality_l765_76538


namespace income_ratio_l765_76572

def monthly_income_C : ℕ := 17000
def annual_income_A : ℕ := 571200

def monthly_income_B : ℕ := monthly_income_C + (12 * monthly_income_C) / 100
def monthly_income_A : ℕ := annual_income_A / 12

theorem income_ratio :
  (monthly_income_A : ℚ) / monthly_income_B = 5 / 2 := by sorry

end income_ratio_l765_76572


namespace paise_to_rupees_l765_76545

/-- 
If 0.5% of a quantity is equal to 65 paise, then the quantity is equal to 130 rupees.
-/
theorem paise_to_rupees (a : ℝ) : (0.005 * a = 65) → (a = 130 * 100) := by
  sorry

end paise_to_rupees_l765_76545


namespace normal_distribution_probability_l765_76502

/-- A normally distributed random variable -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Probability function for a normal random variable -/
noncomputable def P (X : NormalRV) (f : ℝ → Prop) : ℝ := sorry

theorem normal_distribution_probability 
  (X : NormalRV)
  (h1 : P X (λ x => x > 5) = 0.2)
  (h2 : P X (λ x => x < -1) = 0.2) :
  P X (λ x => 2 < x ∧ x < 5) = 0.3 := by sorry

end normal_distribution_probability_l765_76502


namespace second_order_arithmetic_sequence_property_l765_76546

/-- Second-order arithmetic sequence -/
def SecondOrderArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ x y z : ℝ, ∀ n : ℕ, a n = x * n^2 + y * n + z

/-- First-order difference sequence -/
def FirstOrderDifference (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a (n + 1) - a n

/-- Second-order difference sequence -/
def SecondOrderDifference (b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = b (n + 1) - b n

theorem second_order_arithmetic_sequence_property
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (c : ℕ → ℝ)
  (h1 : SecondOrderArithmeticSequence a)
  (h2 : FirstOrderDifference a b)
  (h3 : SecondOrderDifference b c)
  (h4 : ∀ n : ℕ, c n = 20)
  (h5 : a 10 = 23)
  (h6 : a 20 = 23) :
  a 30 = 2023 := by
  sorry

end second_order_arithmetic_sequence_property_l765_76546


namespace expression_simplification_l765_76511

theorem expression_simplification (y : ℝ) :
  3 * y - 7 * y^2 + 15 - (2 + 6 * y - 7 * y^2) = -3 * y + 13 := by
  sorry

end expression_simplification_l765_76511


namespace absolute_sum_a_b_l765_76556

theorem absolute_sum_a_b : ∀ a b : ℝ, 
  (∀ x : ℝ, (7*x - a)^2 = 49*x^2 - b*x + 9) → 
  |a + b| = 45 := by
sorry

end absolute_sum_a_b_l765_76556


namespace sum_interior_angles_regular_polygon_l765_76541

theorem sum_interior_angles_regular_polygon (n : ℕ) (h : n > 2) :
  (360 / 45 : ℝ) = n →
  (180 * (n - 2) : ℝ) = 1080 :=
by sorry

end sum_interior_angles_regular_polygon_l765_76541


namespace digit_product_theorem_l765_76553

theorem digit_product_theorem (A M C : ℕ) : 
  A < 10 → M < 10 → C < 10 →
  (100 * A + 10 * M + C) * (A + M + C) = 2244 →
  A = 3 := by
sorry

end digit_product_theorem_l765_76553


namespace sector_max_area_l765_76542

/-- Given a circular sector with perimeter 40 units, prove that the area is maximized
    when the central angle is 2 radians and the maximum area is 100 square units. -/
theorem sector_max_area (R : ℝ) (α : ℝ) (h : R * α + 2 * R = 40) :
  (R * α * R / 2 ≤ 100) ∧
  (R * α * R / 2 = 100 ↔ α = 2 ∧ R = 10) :=
sorry

end sector_max_area_l765_76542


namespace union_of_specific_sets_l765_76510

theorem union_of_specific_sets :
  let A : Set ℕ := {0, 1, 2}
  let B : Set ℕ := {2, 4}
  A ∪ B = {0, 1, 2, 4} := by
sorry

end union_of_specific_sets_l765_76510


namespace equal_savings_l765_76550

/-- Represents the financial situation of Uma and Bala -/
structure FinancialSituation where
  uma_income : ℝ
  bala_income : ℝ
  uma_expenditure : ℝ
  bala_expenditure : ℝ

/-- The conditions given in the problem -/
def problem_conditions (fs : FinancialSituation) : Prop :=
  fs.uma_income / fs.bala_income = 8 / 7 ∧
  fs.uma_expenditure / fs.bala_expenditure = 7 / 6 ∧
  fs.uma_income = 16000

/-- The savings of Uma and Bala -/
def savings (fs : FinancialSituation) : ℝ × ℝ :=
  (fs.uma_income - fs.uma_expenditure, fs.bala_income - fs.bala_expenditure)

/-- The theorem to be proved -/
theorem equal_savings (fs : FinancialSituation) :
  problem_conditions fs → savings fs = (2000, 2000) := by
  sorry


end equal_savings_l765_76550


namespace distribute_five_balls_four_boxes_l765_76576

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls 5 4 = 56 := by
  sorry

#eval distribute_balls 5 4

end distribute_five_balls_four_boxes_l765_76576


namespace binomial_1300_2_l765_76549

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binomial_1300_2_l765_76549


namespace road_trip_time_calculation_l765_76596

/-- Represents the road trip problem -/
theorem road_trip_time_calculation 
  (freeway_distance : ℝ) 
  (mountain_distance : ℝ) 
  (mountain_time : ℝ) 
  (speed_ratio : ℝ) :
  freeway_distance = 120 →
  mountain_distance = 25 →
  mountain_time = 75 →
  speed_ratio = 4 →
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  freeway_time + mountain_time = 165 :=
by sorry

end road_trip_time_calculation_l765_76596


namespace max_xy_given_constraint_l765_76580

theorem max_xy_given_constraint (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4 * y = 4) :
  ∀ x' y' : ℝ, 0 < x' → 0 < y' → x' + 4 * y' = 4 → x' * y' ≤ x * y ∧ x * y = 1 :=
sorry

end max_xy_given_constraint_l765_76580


namespace overlap_area_is_75_l765_76529

-- Define a 30-60-90 triangle
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = hypotenuse / 2
  longLeg_eq : longLeg = shortLeg * Real.sqrt 3

-- Define the overlapping configuration
def overlapArea (t : Triangle30_60_90) : ℝ :=
  t.longLeg * t.longLeg

-- Theorem statement
theorem overlap_area_is_75 (t : Triangle30_60_90) :
  overlapArea t = 75 := by
  sorry

end overlap_area_is_75_l765_76529


namespace count_six_digit_numbers_middle_same_is_90000_l765_76533

/-- Counts the number of six-digit numbers where only the middle two digits are the same -/
def count_six_digit_numbers_middle_same : ℕ :=
  -- First digit: 9 choices (1-9)
  9 * 
  -- Second digit: 10 choices (0-9)
  10 * 
  -- Third digit: 10 choices (0-9)
  10 * 
  -- Fourth digit: 1 choice (same as third)
  1 * 
  -- Fifth digit: 10 choices (0-9)
  10 * 
  -- Sixth digit: 10 choices (0-9)
  10

/-- Theorem stating that the count of six-digit numbers with only middle digits the same is 90000 -/
theorem count_six_digit_numbers_middle_same_is_90000 :
  count_six_digit_numbers_middle_same = 90000 := by
  sorry

end count_six_digit_numbers_middle_same_is_90000_l765_76533


namespace travis_annual_cereal_cost_l765_76565

/-- Calculates the annual cereal cost for Travis --/
theorem travis_annual_cereal_cost :
  let box_a_cost : ℝ := 3.50
  let box_b_cost : ℝ := 4.00
  let box_c_cost : ℝ := 5.25
  let box_a_consumption : ℝ := 1
  let box_b_consumption : ℝ := 0.5
  let box_c_consumption : ℝ := 1/3
  let discount_rate : ℝ := 0.1
  let weeks_per_year : ℕ := 52

  let weekly_cost : ℝ := 
    box_a_cost * box_a_consumption + 
    box_b_cost * box_b_consumption + 
    box_c_cost * box_c_consumption

  let discounted_weekly_cost : ℝ := weekly_cost * (1 - discount_rate)

  let annual_cost : ℝ := discounted_weekly_cost * weeks_per_year

  annual_cost = 339.30 := by sorry

end travis_annual_cereal_cost_l765_76565


namespace larger_integer_problem_l765_76505

theorem larger_integer_problem (x y : ℕ+) 
  (h1 : x.val - y.val = 8) 
  (h2 : x.val * y.val = 198) : 
  x.val = 18 := by
sorry

end larger_integer_problem_l765_76505


namespace white_tiles_count_l765_76598

theorem white_tiles_count (total : ℕ) (yellow : ℕ) (purple : ℕ) 
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6) :
  total - (yellow + (yellow + 1) + purple) = 7 := by
  sorry

end white_tiles_count_l765_76598
