import Mathlib

namespace ten_apples_left_l1130_113084

/-- The number of apples left after Frank's dog eats some -/
def apples_left (on_tree : ℕ) (on_ground : ℕ) (eaten : ℕ) : ℕ :=
  on_tree + (on_ground - eaten)

/-- Theorem: Given the initial conditions, there are 10 apples left -/
theorem ten_apples_left : apples_left 5 8 3 = 10 := by
  sorry

end ten_apples_left_l1130_113084


namespace sum_first_150_remainder_l1130_113025

theorem sum_first_150_remainder (n : ℕ) (h : n = 150) : 
  (n * (n + 1) / 2) % 12000 = 11325 := by
  sorry

end sum_first_150_remainder_l1130_113025


namespace binary_sequence_equiv_powerset_nat_l1130_113046

/-- The type of infinite binary sequences -/
def BinarySequence := ℕ → Bool

/-- The theorem stating the equinumerosity of binary sequences and subsets of naturals -/
theorem binary_sequence_equiv_powerset_nat :
  ∃ (f : BinarySequence → Set ℕ), Function.Bijective f :=
sorry

end binary_sequence_equiv_powerset_nat_l1130_113046


namespace circle_equation_polar_l1130_113097

/-- The equation of a circle in polar coordinates with center at (√2, π) passing through the pole -/
theorem circle_equation_polar (ρ θ : ℝ) : 
  (ρ = -2 * Real.sqrt 2 * Real.cos θ) ↔ 
  (∃ (x y : ℝ), 
    -- Convert polar to Cartesian coordinates
    (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
    -- Circle equation in Cartesian coordinates
    ((x + Real.sqrt 2)^2 + y^2 = 2) ∧
    -- Circle passes through the pole (origin in Cartesian)
    (∃ (θ₀ : ℝ), ρ * Real.cos θ₀ = 0 ∧ ρ * Real.sin θ₀ = 0)) := by
  sorry


end circle_equation_polar_l1130_113097


namespace isosceles_trajectory_equation_l1130_113061

/-- An isosceles triangle ABC with vertices A(3,20) and B(3,5) -/
structure IsoscelesTriangle where
  C : ℝ × ℝ
  isIsosceles : (C.1 - 3)^2 + (C.2 - 20)^2 = (3 - 3)^2 + (5 - 20)^2
  notCollinear : C.1 ≠ 3

/-- The trajectory equation of point C in an isosceles triangle ABC -/
def trajectoryEquation (t : IsoscelesTriangle) : Prop :=
  (t.C.1 - 3)^2 + (t.C.2 - 20)^2 = 225

/-- Theorem: The trajectory equation holds for any isosceles triangle satisfying the given conditions -/
theorem isosceles_trajectory_equation (t : IsoscelesTriangle) : trajectoryEquation t := by
  sorry


end isosceles_trajectory_equation_l1130_113061


namespace mcgregor_books_finished_l1130_113029

theorem mcgregor_books_finished (total_books : ℕ) (floyd_finished : ℕ) (books_left : ℕ) : 
  total_books = 89 → floyd_finished = 32 → books_left = 23 → 
  total_books - floyd_finished - books_left = 34 := by
sorry

end mcgregor_books_finished_l1130_113029


namespace john_lap_time_improvement_l1130_113090

theorem john_lap_time_improvement :
  let initial_laps : ℚ := 15
  let initial_time : ℚ := 40
  let current_laps : ℚ := 18
  let current_time : ℚ := 36
  let initial_lap_time := initial_time / initial_laps
  let current_lap_time := current_time / current_laps
  let improvement := initial_lap_time - current_lap_time
  improvement = 2/3
:= by sorry

end john_lap_time_improvement_l1130_113090


namespace captainSelection_l1130_113059

/-- The number of ways to select a captain and a vice-captain from a team of 11 people -/
def selectCaptains : ℕ :=
  11 * 10

/-- Theorem stating that the number of ways to select a captain and a vice-captain
    from a team of 11 people is equal to 110 -/
theorem captainSelection : selectCaptains = 110 := by
  sorry

end captainSelection_l1130_113059


namespace cubic_equation_solution_l1130_113052

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cubic_equation_solution_l1130_113052


namespace diamond_equation_solution_l1130_113024

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

-- Theorem statement
theorem diamond_equation_solution :
  ∃ A : ℝ, diamond A 3 = 85 ∧ A = 17.25 := by
  sorry

end diamond_equation_solution_l1130_113024


namespace thabo_hardcover_nonfiction_l1130_113038

/-- The number of books Thabo owns -/
def total_books : ℕ := 280

/-- The number of paperback nonfiction books -/
def paperback_nonfiction (hardcover_nonfiction : ℕ) : ℕ := hardcover_nonfiction + 20

/-- The number of paperback fiction books -/
def paperback_fiction (hardcover_nonfiction : ℕ) : ℕ := 2 * (paperback_nonfiction hardcover_nonfiction)

/-- Theorem stating the number of hardcover nonfiction books Thabo owns -/
theorem thabo_hardcover_nonfiction :
  ∃ (hardcover_nonfiction : ℕ),
    hardcover_nonfiction + paperback_nonfiction hardcover_nonfiction + paperback_fiction hardcover_nonfiction = total_books ∧
    hardcover_nonfiction = 55 := by
  sorry

end thabo_hardcover_nonfiction_l1130_113038


namespace bicycle_journey_l1130_113075

theorem bicycle_journey (t₁ t₂ : ℝ) (h₁ : t₁ > 0) (h₂ : t₂ > 0) : 
  (5 * t₁ + 15 * t₂) / (t₁ + t₂) = 10 → t₂ / (t₁ + t₂) = 1/2 := by
  sorry

end bicycle_journey_l1130_113075


namespace max_value_sqrt_sum_l1130_113099

theorem max_value_sqrt_sum (x : ℝ) (h : x ∈ Set.Icc (-49 : ℝ) 49) :
  ∃ (M : ℝ), M = 14 ∧ Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ M ∧
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-49 : ℝ) 49 ∧ Real.sqrt (49 + x₀) + Real.sqrt (49 - x₀) = M :=
sorry

end max_value_sqrt_sum_l1130_113099


namespace metallic_sheet_width_l1130_113030

/-- Given a rectangular metallic sheet with length 48 m, from which squares of side 8 m
    are cut from each corner to form an open box with volume 5120 m³,
    prove that the width of the original metallic sheet is 36 m. -/
theorem metallic_sheet_width :
  ∀ (w : ℝ),
  let length : ℝ := 48
  let cut_side : ℝ := 8
  let box_volume : ℝ := 5120
  let box_length : ℝ := length - 2 * cut_side
  let box_width : ℝ := w - 2 * cut_side
  let box_height : ℝ := cut_side
  box_volume = box_length * box_width * box_height →
  w = 36 :=
by
  sorry

end metallic_sheet_width_l1130_113030


namespace intersection_of_three_lines_l1130_113001

theorem intersection_of_three_lines (k : ℝ) : 
  (∃! p : ℝ × ℝ, 
    (p.1 + k * p.2 = 0) ∧ 
    (2 * p.1 + 3 * p.2 + 8 = 0) ∧ 
    (p.1 - p.2 - 1 = 0)) → 
  k = -1/2 := by
sorry

end intersection_of_three_lines_l1130_113001


namespace range_of_expression_l1130_113092

theorem range_of_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z : ℝ), z = 4 * Real.arcsin x - Real.arccos y ∧ 
  -5 * π / 2 ≤ z ∧ z ≤ 3 * π / 2 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 + y₁^2 = 1 ∧ 4 * Real.arcsin x₁ - Real.arccos y₁ = -5 * π / 2) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 + y₂^2 = 1 ∧ 4 * Real.arcsin x₂ - Real.arccos y₂ = 3 * π / 2) :=
by sorry

end range_of_expression_l1130_113092


namespace least_number_with_remainder_l1130_113072

theorem least_number_with_remainder (n : ℕ) : n = 266 ↔ 
  (n > 0 ∧ 
   n % 33 = 2 ∧ 
   n % 8 = 2 ∧ 
   ∀ m : ℕ, m > 0 → m % 33 = 2 → m % 8 = 2 → m ≥ n) := by
  sorry

end least_number_with_remainder_l1130_113072


namespace find_m_l1130_113086

theorem find_m : ∃ m : ℝ, (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 := by sorry

end find_m_l1130_113086


namespace softball_team_composition_l1130_113028

theorem softball_team_composition (total : ℕ) (ratio : ℚ) : 
  total = 16 ∧ ratio = 5/11 → ∃ (men women : ℕ), 
    men + women = total ∧ 
    (men : ℚ) / (women : ℚ) = ratio ∧ 
    women - men = 6 :=
by sorry

end softball_team_composition_l1130_113028


namespace largest_prime_divisor_l1130_113033

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ 
    p ∣ (2^(p+1) + 3^(p+1) + 5^(p+1) + 7^(p+1)) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (2^(q+1) + 3^(q+1) + 5^(q+1) + 7^(q+1)) → q ≤ p :=
by
  use 29
  sorry

end largest_prime_divisor_l1130_113033


namespace range_of_a_l1130_113089

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| - |x + 2| ≤ 3) → -5 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l1130_113089


namespace key_sequence_produces_desired_output_l1130_113015

/-- Represents the mapping of keys to displayed letters on the magical keyboard. -/
def keyboard_mapping : Char → Char
| 'Q' => 'A'
| 'S' => 'D'
| 'D' => 'S'
| 'J' => 'H'
| 'K' => 'O'
| 'L' => 'P'
| 'R' => 'E'
| 'N' => 'M'
| 'Y' => 'T'
| c => c  -- For all other characters, map to themselves

/-- The sequence of key presses -/
def key_sequence : List Char := ['J', 'K', 'L', 'R', 'N', 'Q', 'Y', 'J']

/-- The desired display output -/
def desired_output : List Char := ['H', 'O', 'P', 'E', 'M', 'A', 'T', 'H']

/-- Theorem stating that the key sequence produces the desired output -/
theorem key_sequence_produces_desired_output :
  key_sequence.map keyboard_mapping = desired_output := by
  sorry

#eval key_sequence.map keyboard_mapping

end key_sequence_produces_desired_output_l1130_113015


namespace import_tax_calculation_l1130_113077

/-- Given an item with a total value V, subject to a 7% import tax on the portion
    exceeding $1,000, prove that if the tax paid is $109.90, then V = $2,567. -/
theorem import_tax_calculation (V : ℝ) : 
  (0.07 * (V - 1000) = 109.90) → V = 2567 := by
  sorry

end import_tax_calculation_l1130_113077


namespace rectangle_area_l1130_113034

theorem rectangle_area (width : ℝ) (length : ℝ) : 
  width = 7 → length = 4 * width → width * length = 196 := by
  sorry

end rectangle_area_l1130_113034


namespace board_numbers_theorem_l1130_113093

theorem board_numbers_theorem (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a - 2, b + 2, c^2} → 
  a + b + c = 2005 → 
  a = 1003 ∨ a = 1002 := by
  sorry

end board_numbers_theorem_l1130_113093


namespace letter_digit_impossibility_l1130_113039

theorem letter_digit_impossibility :
  ¬ ∃ (f : Fin 7 → Fin 10),
    Function.Injective f ∧
    (f 0 * f 1 * 0 : ℕ) = (f 2 * f 3 * f 4 * f 5 * f 6 : ℕ) := by
  sorry

end letter_digit_impossibility_l1130_113039


namespace system_solutions_l1130_113098

theorem system_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (x₁ + x₂ = x₃^2 ∧
   x₂ + x₃ = x₄^2 ∧
   x₃ + x₁ = x₅^2 ∧
   x₄ + x₅ = x₁^2 ∧
   x₅ + x₁ = x₂^2) →
  ((x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2) ∨
   (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0)) :=
by sorry

end system_solutions_l1130_113098


namespace inverse_square_relation_l1130_113066

theorem inverse_square_relation (k : ℝ) (a b c : ℝ) :
  (∀ a b c, a^2 * b^2 / c = k) →
  (4^2 * 2^2 / 3 = k) →
  (a^2 * 4^2 / 6 = k) →
  a^2 = 8 := by
  sorry

end inverse_square_relation_l1130_113066


namespace inequalities_proof_l1130_113000

theorem inequalities_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (b / a + a / b > 2) := by
  sorry

end inequalities_proof_l1130_113000


namespace circle_line_distance_l1130_113063

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  equation : ℝ → ℝ → Prop

/-- Calculates the distance from a point to a line --/
def distancePointToLine (point : ℝ × ℝ) (line : Line) : ℝ := sorry

/-- The main theorem --/
theorem circle_line_distance (c : Circle) (l : Line) :
  c.equation = fun x y => x^2 + y^2 - 2*x - 8*y + 1 = 0 →
  l.equation = fun x y => l.a*x - y + 1 = 0 →
  c.center = (1, 4) →
  distancePointToLine c.center l = 1 →
  l.a = 4/3 := by sorry

end circle_line_distance_l1130_113063


namespace initial_lives_calculation_l1130_113054

/-- Proves that the initial number of lives equals the current number of lives plus the number of lives lost -/
theorem initial_lives_calculation (current_lives lost_lives : ℕ) 
  (h1 : current_lives = 70) 
  (h2 : lost_lives = 13) : 
  current_lives + lost_lives = 83 := by
  sorry

end initial_lives_calculation_l1130_113054


namespace power_mod_29_l1130_113088

theorem power_mod_29 : 17^2003 % 29 = 26 := by
  sorry

end power_mod_29_l1130_113088


namespace milk_packets_average_price_l1130_113074

theorem milk_packets_average_price 
  (total_packets : ℕ) 
  (kept_packets : ℕ) 
  (returned_packets : ℕ) 
  (kept_avg_price : ℚ) 
  (returned_avg_price : ℚ) :
  total_packets = kept_packets + returned_packets →
  kept_packets = 3 →
  returned_packets = 2 →
  kept_avg_price = 12 →
  returned_avg_price = 32 →
  (kept_packets * kept_avg_price + returned_packets * returned_avg_price) / total_packets = 20 :=
by sorry

end milk_packets_average_price_l1130_113074


namespace unique_base_solution_l1130_113069

/-- Converts a base-10 number to its representation in base b -/
def toBase (n : ℕ) (b : ℕ) : List ℕ := sorry

/-- Converts a number represented as a list of digits in base b to base 10 -/
def fromBase (digits : List ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if the equation 742_b - 305_b = 43C_b holds for a given base b -/
def equationHolds (b : ℕ) : Prop :=
  let lhs := fromBase (toBase 742 b) b - fromBase (toBase 305 b) b
  let rhs := fromBase (toBase 43 b) b * 12
  lhs = rhs

theorem unique_base_solution :
  ∃! b : ℕ, b > 1 ∧ equationHolds b :=
sorry

end unique_base_solution_l1130_113069


namespace minuend_not_integer_l1130_113073

theorem minuend_not_integer (M S : ℝ) : M + S + (M - S) = 555 → ¬(∃ n : ℤ, M = n) := by
  sorry

end minuend_not_integer_l1130_113073


namespace janes_mean_score_l1130_113081

def janes_scores : List ℝ := [85, 90, 95, 80, 100]

theorem janes_mean_score : 
  (janes_scores.sum / janes_scores.length : ℝ) = 90 := by
  sorry

end janes_mean_score_l1130_113081


namespace negation_of_all_nonnegative_l1130_113062

theorem negation_of_all_nonnegative (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by sorry

end negation_of_all_nonnegative_l1130_113062


namespace sqrt_product_equality_l1130_113083

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l1130_113083


namespace complex_division_result_l1130_113041

theorem complex_division_result : (4 - 2*I) / (1 + I) = 1 - 3*I := by
  sorry

end complex_division_result_l1130_113041


namespace center_value_theorem_l1130_113016

/-- Represents a 6x6 matrix with arithmetic sequences in rows and columns -/
def ArithmeticMatrix := Matrix (Fin 6) (Fin 6) ℝ

/-- Checks if a sequence is arithmetic -/
def is_arithmetic_sequence (seq : Fin 6 → ℝ) : Prop :=
  ∀ i j k : Fin 6, i < j ∧ j < k → seq j - seq i = seq k - seq j

/-- The matrix has arithmetic sequences in all rows and columns -/
def matrix_arithmetic (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 6, is_arithmetic_sequence (λ j => M i j)) ∧
  (∀ j : Fin 6, is_arithmetic_sequence (λ i => M i j))

theorem center_value_theorem (M : ArithmeticMatrix) 
  (h_arithmetic : matrix_arithmetic M)
  (h_first_row : M 0 1 = 3 ∧ M 0 4 = 27)
  (h_last_row : M 5 1 = 25 ∧ M 5 4 = 85) :
  M 2 2 = 30 ∧ M 2 3 = 30 ∧ M 3 2 = 30 ∧ M 3 3 = 30 := by
  sorry

end center_value_theorem_l1130_113016


namespace perfect_square_triples_l1130_113022

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def satisfies_condition (a b c : ℕ) : Prop :=
  is_perfect_square (2^a + 2^b + 2^c + 3)

theorem perfect_square_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
sorry

end perfect_square_triples_l1130_113022


namespace no_integer_solutions_l1130_113003

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x ≠ 1 ∧ (x^7 - 1) / (x - 1) = y^5 - 1 := by
  sorry

end no_integer_solutions_l1130_113003


namespace max_value_trig_sum_l1130_113014

theorem max_value_trig_sum (a b φ : ℝ) :
  ∃ (max : ℝ), ∀ θ : ℝ, a * Real.cos (θ + φ) + b * Real.sin (θ + φ) ≤ max ∧
  ∃ θ₀ : ℝ, a * Real.cos (θ₀ + φ) + b * Real.sin (θ₀ + φ) = max ∧
  max = Real.sqrt (a^2 + b^2) :=
sorry

end max_value_trig_sum_l1130_113014


namespace marbles_lost_l1130_113017

theorem marbles_lost (initial : ℕ) (final : ℕ) (h1 : initial = 38) (h2 : final = 23) :
  initial - final = 15 := by
  sorry

end marbles_lost_l1130_113017


namespace football_lineup_combinations_l1130_113045

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem football_lineup_combinations :
  lineup_combinations = 31680 := by
  sorry

end football_lineup_combinations_l1130_113045


namespace isosceles_trapezoid_area_l1130_113009

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 25,
    diagonal_length := 34,
    longer_base := 40
  }
  ∃ ε > 0, |area t - 569.275| < ε :=
sorry

end isosceles_trapezoid_area_l1130_113009


namespace rectangular_prism_volume_l1130_113060

/-- A rectangular prism with given conditions -/
structure RectangularPrism where
  length : ℝ
  breadth : ℝ
  height : ℝ
  length_breadth_diff : length - breadth = 23
  perimeter : 2 * length + 2 * breadth = 166

/-- The volume of a rectangular prism is 1590h cubic meters -/
theorem rectangular_prism_volume (prism : RectangularPrism) : 
  prism.length * prism.breadth * prism.height = 1590 * prism.height := by
  sorry

end rectangular_prism_volume_l1130_113060


namespace earliest_meeting_time_l1130_113005

def anna_lap_time : ℕ := 4
def stephanie_lap_time : ℕ := 7
def james_lap_time : ℕ := 6

theorem earliest_meeting_time :
  let meeting_time := lcm (lcm anna_lap_time stephanie_lap_time) james_lap_time
  meeting_time = 84 := by
  sorry

end earliest_meeting_time_l1130_113005


namespace sum_of_z_values_l1130_113027

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 3*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 3/4) := by
  sorry

end sum_of_z_values_l1130_113027


namespace sum_reciprocal_squares_l1130_113044

theorem sum_reciprocal_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end sum_reciprocal_squares_l1130_113044


namespace product_sum_digits_base7_l1130_113010

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Sums the digits of a number in base-7 --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_sum_digits_base7 :
  let a := 35
  let b := 52
  sumDigitsBase7 (toBase7 (toBase10 a * toBase10 b)) = 16 := by sorry

end product_sum_digits_base7_l1130_113010


namespace rectangle_division_even_triangles_l1130_113035

theorem rectangle_division_even_triangles 
  (a b c d : ℕ) 
  (h_rect : a > 0 ∧ b > 0) 
  (h_tri : c > 0 ∧ d > 0) 
  (h_div : (a * b) % (c * d / 2) = 0) :
  ∃ k : ℕ, k % 2 = 0 ∧ k * (c * d / 2) = a * b :=
sorry

end rectangle_division_even_triangles_l1130_113035


namespace hyperbola_focal_distance_l1130_113050

/-- The focal distance of the hyperbola 2x^2 - y^2 = 6 is 6 -/
theorem hyperbola_focal_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | 2 * x^2 - y^2 = 6}
  ∃ f : ℝ, f = 6 ∧ ∀ (x y : ℝ), (x, y) ∈ hyperbola →
    ∃ (F₁ F₂ : ℝ × ℝ), abs (x - F₁.1) + abs (x - F₂.1) = 2 * f :=
by sorry

end hyperbola_focal_distance_l1130_113050


namespace parallel_intersection_lines_l1130_113068

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallelism relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the intersection operation for a plane and a line
variable (intersect : Plane → Plane → Line)

-- Define the parallelism relation for lines
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_intersection_lines
  (m n : Line)
  (α β γ : Plane)
  (h1 : α ≠ β)
  (h2 : α ≠ γ)
  (h3 : β ≠ γ)
  (h4 : parallel_planes α β)
  (h5 : intersect α γ = m)
  (h6 : intersect β γ = n) :
  parallel_lines m n :=
sorry

end parallel_intersection_lines_l1130_113068


namespace square_area_with_circles_l1130_113042

/-- The area of a square containing a 3x3 grid of circles with radius 3 inches -/
theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  (3 * (2 * r))^2 = 324 := by
  sorry

end square_area_with_circles_l1130_113042


namespace perpendicular_tangents_theorem_l1130_113076

noncomputable def f (x : ℝ) : ℝ := abs x / Real.exp x

def is_perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

theorem perpendicular_tangents_theorem (x₀ : ℝ) (m : ℤ) :
  x₀ > 0 ∧
  x₀ ∈ Set.Ioo (m / 4 : ℝ) ((m + 1) / 4 : ℝ) ∧
  is_perpendicular ((deriv f) (-1)) ((deriv f) x₀) →
  m = 2 :=
sorry

end perpendicular_tangents_theorem_l1130_113076


namespace triangle_angle_A_l1130_113031

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hab : a > b) 
  (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = π / 4) :
  ∃ A : ℝ, (A = π / 3 ∨ A = 2 * π / 3) ∧ 
    Real.sin A = (a * Real.sin B) / b :=
sorry

end triangle_angle_A_l1130_113031


namespace share_difference_l1130_113002

def money_distribution (total : ℕ) (faruk vasim ranjith : ℕ) : Prop :=
  faruk + vasim + ranjith = total ∧ 3 * ranjith = 7 * faruk ∧ faruk = vasim

theorem share_difference (total : ℕ) (faruk vasim ranjith : ℕ) :
  money_distribution total faruk vasim ranjith → vasim = 1500 → ranjith - faruk = 2000 := by
  sorry

end share_difference_l1130_113002


namespace seating_solution_l1130_113019

/-- A seating arrangement with rows of 6 or 7 people. -/
structure SeatingArrangement where
  rows_with_7 : ℕ
  rows_with_6 : ℕ
  total_people : ℕ
  h1 : total_people = 7 * rows_with_7 + 6 * rows_with_6
  h2 : total_people = 59

/-- The solution to the seating arrangement problem. -/
theorem seating_solution (s : SeatingArrangement) : s.rows_with_7 = 5 := by
  sorry

#check seating_solution

end seating_solution_l1130_113019


namespace estate_division_l1130_113040

theorem estate_division (estate : ℝ) 
  (wife_share son_share daughter_share cook_share : ℝ) : 
  (daughter_share + son_share = estate / 2) →
  (daughter_share = 4 * son_share / 3) →
  (wife_share = 2 * son_share) →
  (cook_share = 500) →
  (estate = wife_share + son_share + daughter_share + cook_share) →
  estate = 7000 := by
  sorry

#check estate_division

end estate_division_l1130_113040


namespace return_trip_time_l1130_113058

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against_wind : ℝ  -- time taken against wind
  time_diff : ℝ  -- time difference between still air and with wind

/-- The main theorem about the return trip time -/
theorem return_trip_time (fs : FlightScenario) 
  (h1 : fs.time_against_wind = 90)
  (h2 : fs.d = fs.time_against_wind * (fs.p - fs.w))
  (h3 : fs.d / (fs.p + fs.w) = fs.d / fs.p - fs.time_diff)
  (h4 : fs.time_diff = 12) :
  fs.d / (fs.p + fs.w) = 18 ∨ fs.d / (fs.p + fs.w) = 60 := by
  sorry

end return_trip_time_l1130_113058


namespace population_reaches_max_capacity_l1130_113020

/-- The number of acres on the island of Nisos -/
def island_acres : ℕ := 36000

/-- The number of acres required per person -/
def acres_per_person : ℕ := 2

/-- The initial population in 2040 -/
def initial_population : ℕ := 300

/-- The number of years it takes for the population to quadruple -/
def quadruple_period : ℕ := 30

/-- The maximum capacity of the island -/
def max_capacity : ℕ := island_acres / acres_per_person

/-- The population after n periods -/
def population (n : ℕ) : ℕ := initial_population * 4^n

/-- The number of years from 2040 until the population reaches or exceeds the maximum capacity -/
theorem population_reaches_max_capacity : 
  ∃ n : ℕ, n * quadruple_period = 90 ∧ population n ≥ max_capacity ∧ population (n - 1) < max_capacity :=
sorry

end population_reaches_max_capacity_l1130_113020


namespace wilted_flowers_count_l1130_113013

def initial_flowers : ℕ := 88
def flowers_per_bouquet : ℕ := 5
def bouquets_made : ℕ := 8

theorem wilted_flowers_count : 
  initial_flowers - (flowers_per_bouquet * bouquets_made) = 48 := by
  sorry

end wilted_flowers_count_l1130_113013


namespace problem_statement_l1130_113049

theorem problem_statement (A B : ℝ) : 
  A^2 = 0.012345678987654321 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1) →
  B^2 = 0.012345679 →
  9 * 10^9 * (1 - |A|) * B = 1 ∨ 9 * 10^9 * (1 - |A|) * B = -1 :=
by sorry

end problem_statement_l1130_113049


namespace unique_pair_sum_28_l1130_113080

theorem unique_pair_sum_28 (a b : ℕ) : 
  a ≠ b → a > 11 → b > 11 → a + b = 28 → (Even a ∨ Even b) → 
  ((a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12)) := by
sorry

end unique_pair_sum_28_l1130_113080


namespace completing_square_sum_l1130_113055

theorem completing_square_sum (d e f : ℤ) : 
  (100 : ℤ) * (x : ℚ)^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f →
  d > 0 →
  d + e + f = 112 :=
by sorry

end completing_square_sum_l1130_113055


namespace no_valid_balanced_coloring_l1130_113023

/-- A chessboard is a 2D grid of squares that can be colored black or white -/
def Chessboard := Fin 1900 → Fin 1900 → Bool

/-- A point on the chessboard -/
def Point := Fin 1900 × Fin 1900

/-- The center point of the chessboard -/
def center : Point := (949, 949)

/-- Two points are symmetric if they are equidistant from the center in opposite directions -/
def symmetric (p q : Point) : Prop :=
  p.1 + q.1 = 2 * center.1 ∧ p.2 + q.2 = 2 * center.2

/-- A valid coloring satisfies the symmetry condition -/
def valid_coloring (c : Chessboard) : Prop :=
  ∀ p q : Point, symmetric p q → c p.1 p.2 ≠ c q.1 q.2

/-- A balanced coloring has an equal number of black and white squares in each row and column -/
def balanced_coloring (c : Chessboard) : Prop :=
  (∀ i : Fin 1900, (Finset.filter (λ j => c i j) Finset.univ).card = 950) ∧
  (∀ j : Fin 1900, (Finset.filter (λ i => c i j) Finset.univ).card = 950)

/-- The main theorem: it's impossible to have a valid and balanced coloring -/
theorem no_valid_balanced_coloring :
  ¬∃ c : Chessboard, valid_coloring c ∧ balanced_coloring c := by
  sorry

end no_valid_balanced_coloring_l1130_113023


namespace division_theorem_l1130_113056

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 125 →
  divisor = 15 →
  remainder = 5 →
  quotient = (dividend - remainder) / divisor →
  quotient = 8 := by
sorry

end division_theorem_l1130_113056


namespace farmer_land_usage_l1130_113032

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0  -- unreachable

/-- The total parts in the ratio -/
def total_parts : ℕ := (land_ratio 0) + (land_ratio 1) + (land_ratio 2)

/-- The number of acres used for corn -/
def corn_acres : ℕ := 376

theorem farmer_land_usage :
  let total_acres := (total_parts * corn_acres) / (land_ratio 2)
  total_acres = 1034 := by sorry

end farmer_land_usage_l1130_113032


namespace problem_solution_l1130_113008

theorem problem_solution : ∀ M N X : ℕ,
  M = 2022 / 3 →
  N = M / 3 →
  X = M + N →
  X = 898 := by
sorry

end problem_solution_l1130_113008


namespace unique_ages_l1130_113067

/-- Represents the ages of Gala, Vova, and Katya -/
structure Ages where
  gala : ℕ
  vova : ℕ
  katya : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.gala < 6 ∧
  ages.vova + ages.katya = 112 ∧
  (ages.vova / ages.gala : ℚ) = (ages.katya / ages.vova : ℚ)

/-- Theorem stating that the only ages satisfying all conditions are 2, 14, and 98 -/
theorem unique_ages : ∃! ages : Ages, satisfies_conditions ages ∧ ages.gala = 2 ∧ ages.vova = 14 ∧ ages.katya = 98 := by
  sorry

end unique_ages_l1130_113067


namespace no_prime_cubic_polynomial_l1130_113095

theorem no_prime_cubic_polynomial :
  ¬ ∃ (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) := by
  sorry

end no_prime_cubic_polynomial_l1130_113095


namespace possible_values_of_y_l1130_113047

theorem possible_values_of_y (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 5)
  y = 0 ∨ y = 144 ∨ y = -24 := by sorry

end possible_values_of_y_l1130_113047


namespace tully_age_proof_l1130_113006

def kate_current_age : ℕ := 29

theorem tully_age_proof (tully_age_year_ago : ℕ) : 
  (tully_age_year_ago + 4 = 2 * (kate_current_age + 3)) → tully_age_year_ago = 60 :=
by
  sorry

end tully_age_proof_l1130_113006


namespace circle_center_coordinates_l1130_113079

/-- Given a circle with polar equation ρ = 5cos(θ) - 5√3sin(θ), 
    its center coordinates in polar form are (5, 5π/3) -/
theorem circle_center_coordinates (θ : Real) (ρ : Real) :
  ρ = 5 * Real.cos θ - 5 * Real.sqrt 3 * Real.sin θ →
  ∃ (r : Real) (φ : Real),
    r = 5 ∧ φ = 5 * Real.pi / 3 ∧
    r * Real.cos φ = 5 / 2 ∧
    r * Real.sin φ = -5 * Real.sqrt 3 / 2 :=
by sorry

end circle_center_coordinates_l1130_113079


namespace polynomial_product_expansion_l1130_113051

theorem polynomial_product_expansion (x : ℝ) :
  (x^3 - 3*x^2 + 3*x - 1) * (x^2 + 3*x + 3) = x^5 - 3*x^2 + 6*x - 3 := by
  sorry

end polynomial_product_expansion_l1130_113051


namespace tan_sum_pi_twelfths_l1130_113007

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by sorry

end tan_sum_pi_twelfths_l1130_113007


namespace max_square_plots_l1130_113082

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def available_fencing : ℕ := 1994

/-- Calculates the number of square plots given the side length -/
def num_plots (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length) * (dims.length / side_length)

/-- Calculates the required internal fencing for a given configuration -/
def required_fencing (dims : FieldDimensions) (side_length : ℕ) : ℕ :=
  (dims.width / side_length - 1) * dims.length + (dims.length / side_length - 1) * dims.width

/-- Theorem stating that 78 is the maximum number of square plots -/
theorem max_square_plots (dims : FieldDimensions) 
    (h_width : dims.width = 24) 
    (h_length : dims.length = 52) : 
    ∀ side_length : ℕ, 
      side_length > 0 → 
      dims.width % side_length = 0 → 
      dims.length % side_length = 0 → 
      required_fencing dims side_length ≤ available_fencing → 
      num_plots dims side_length ≤ 78 :=
  sorry

#check max_square_plots

end max_square_plots_l1130_113082


namespace mother_age_is_55_l1130_113085

/-- The mother's age in years -/
def mother_age : ℕ := 55

/-- The daughter's age in years -/
def daughter_age : ℕ := mother_age - 27

theorem mother_age_is_55 :
  (mother_age = daughter_age + 27) ∧
  (mother_age - 1 = 2 * (daughter_age - 1)) →
  mother_age = 55 := by
  sorry

#check mother_age_is_55

end mother_age_is_55_l1130_113085


namespace quadratic_equations_solutions_l1130_113053

theorem quadratic_equations_solutions :
  (∀ x, x * (x - 3) + x = 3 ↔ x = 3 ∨ x = -1) ∧
  (∀ x, 3 * x^2 - 1 = 4 * x ↔ x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) :=
by sorry

end quadratic_equations_solutions_l1130_113053


namespace actual_average_height_l1130_113012

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initial average height in centimeters -/
def initial_avg : ℚ := 182

/-- The incorrectly recorded height in centimeters -/
def incorrect_height : ℚ := 166

/-- The correct height in centimeters -/
def correct_height : ℚ := 106

/-- The actual average height after correction -/
def actual_avg : ℚ := (num_boys * initial_avg - (incorrect_height - correct_height)) / num_boys

theorem actual_average_height :
  ∃ ε > 0, abs (actual_avg - 180.29) < ε :=
sorry

end actual_average_height_l1130_113012


namespace circle_passes_through_points_l1130_113094

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - 3*y - 3

/-- Point A coordinates -/
def A : ℝ × ℝ := (-1, 0)

/-- Point B coordinates -/
def B : ℝ × ℝ := (3, 0)

/-- Point C coordinates -/
def C : ℝ × ℝ := (1, 4)

/-- Theorem: The circle_equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation A.1 A.2 = 0 ∧
  circle_equation B.1 B.2 = 0 ∧
  circle_equation C.1 C.2 = 0 := by
  sorry

end circle_passes_through_points_l1130_113094


namespace a_squared_b_plus_ab_squared_l1130_113078

theorem a_squared_b_plus_ab_squared (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) :
  a^2 * b + a * b^2 = 42 := by
  sorry

end a_squared_b_plus_ab_squared_l1130_113078


namespace inequality_proof_l1130_113048

theorem inequality_proof (a b : ℝ) (h : a < b) : 7 * a - 7 * b < 0 := by
  sorry

end inequality_proof_l1130_113048


namespace add_518_276_base_12_l1130_113071

/-- Addition in base 12 --/
def add_base_12 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 12 to base 10 --/
def base_12_to_10 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 12 --/
def base_10_to_12 (n : ℕ) : ℕ :=
  sorry

theorem add_518_276_base_12 :
  add_base_12 (base_10_to_12 518) (base_10_to_12 276) = base_10_to_12 792 :=
sorry

end add_518_276_base_12_l1130_113071


namespace reciprocal_of_repeating_third_l1130_113043

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- Theorem statement
theorem reciprocal_of_repeating_third :
  (repeating_third⁻¹ : ℚ) = 3 := by sorry

end reciprocal_of_repeating_third_l1130_113043


namespace sum_of_sqrt_products_gt_sum_of_numbers_l1130_113096

theorem sum_of_sqrt_products_gt_sum_of_numbers 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) : 
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z := by
  sorry

end sum_of_sqrt_products_gt_sum_of_numbers_l1130_113096


namespace display_rows_l1130_113026

/-- Represents the number of cans in a row given its position from the top. -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- Represents the total number of cans in the first n rows. -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display is 10, given the conditions. -/
theorem display_rows :
  ∃ (n : ℕ), n > 0 ∧ total_cans n = 145 ∧ n = 10 :=
sorry

end display_rows_l1130_113026


namespace pasta_sauce_free_percentage_l1130_113021

/-- Given a pasta dish weighing 200 grams with 50 grams of sauce,
    prove that 75% of the dish is sauce-free. -/
theorem pasta_sauce_free_percentage
  (total_weight : ℝ)
  (sauce_weight : ℝ)
  (h_total : total_weight = 200)
  (h_sauce : sauce_weight = 50) :
  (total_weight - sauce_weight) / total_weight * 100 = 75 := by
  sorry

end pasta_sauce_free_percentage_l1130_113021


namespace conic_is_ellipse_l1130_113070

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
    ∀ (x y : ℝ), conic_equation x y ↔
      (x - (point1.1 + point2.1)/2)^2/a^2 + (y - (point1.2 + point2.2)/2)^2/b^2 = 1 :=
sorry

end conic_is_ellipse_l1130_113070


namespace linear_function_k_value_l1130_113065

/-- Proves that for the linear function y = kx + 3 passing through the point (2, 5), the value of k is 1. -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3) → -- Condition 1: The function is y = kx + 3
  (5 : ℝ) = k * 2 + 3 →        -- Condition 2: The function passes through the point (2, 5)
  k = 1 :=                     -- Conclusion: The value of k is 1
by sorry

end linear_function_k_value_l1130_113065


namespace total_credits_proof_l1130_113036

theorem total_credits_proof (emily_credits : ℕ) 
  (h1 : emily_credits = 20)
  (h2 : ∃ aria_credits : ℕ, aria_credits = 2 * emily_credits)
  (h3 : ∃ spencer_credits : ℕ, spencer_credits = emily_credits / 2)
  (h4 : ∃ hannah_credits : ℕ, hannah_credits = 3 * (emily_credits / 2)) :
  2 * (emily_credits + 2 * emily_credits + emily_credits / 2 + 3 * (emily_credits / 2)) = 200 := by
  sorry

end total_credits_proof_l1130_113036


namespace sons_age_l1130_113037

theorem sons_age (son_age woman_age : ℕ) : 
  woman_age = 2 * son_age + 3 →
  woman_age + son_age = 84 →
  son_age = 27 := by
sorry

end sons_age_l1130_113037


namespace integer_divisibility_l1130_113057

theorem integer_divisibility (m : ℕ) : 
  Prime m → 
  ∃ k : ℕ+, m = 13 * k + 1 → 
  m ≠ 8191 → 
  ∃ n : ℤ, (2^(m-1) - 1) = 8191 * m * n :=
sorry

end integer_divisibility_l1130_113057


namespace parallel_vectors_m_value_l1130_113011

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (2, m) (m, 2) → m = 2 ∨ m = -2 := by
  sorry

end parallel_vectors_m_value_l1130_113011


namespace prob_rain_all_days_l1130_113091

def prob_rain_friday : ℚ := 40 / 100
def prob_rain_saturday : ℚ := 50 / 100
def prob_rain_sunday : ℚ := 30 / 100

def events_independent : Prop := True

theorem prob_rain_all_days : 
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 6 / 100 :=
by sorry

end prob_rain_all_days_l1130_113091


namespace not_perfect_square_l1130_113004

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬∃ (m : ℕ), 9*n^2 - 9*n + 9 = m^2 := by
  sorry

end not_perfect_square_l1130_113004


namespace quadratic_one_solution_l1130_113064

theorem quadratic_one_solution (q : ℝ) : 
  q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 18 * x + 8 = 0) ↔ q = 81/8 := by
  sorry

end quadratic_one_solution_l1130_113064


namespace range_of_m_l1130_113018

/-- The equation |(x-1)(x-3)| = m*x has four distinct real roots -/
def has_four_distinct_roots (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (∀ (x : ℝ), |((x - 1) * (x - 3))| = m * x ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

/-- The theorem stating the range of m -/
theorem range_of_m : 
  ∀ (m : ℝ), has_four_distinct_roots m ↔ 0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
sorry

end range_of_m_l1130_113018


namespace gretzky_street_length_proof_l1130_113087

/-- The length of Gretzky Street in kilometers -/
def gretzky_street_length : ℝ := 5.95

/-- The number of numbered intersecting streets -/
def num_intersections : ℕ := 15

/-- The distance between each intersecting street in meters -/
def intersection_distance : ℝ := 350

/-- The number of additional segments at the beginning and end -/
def additional_segments : ℕ := 2

/-- Theorem stating that the length of Gretzky Street is 5.95 kilometers -/
theorem gretzky_street_length_proof :
  gretzky_street_length = 
    (intersection_distance * (num_intersections + additional_segments)) / 1000 := by
  sorry

end gretzky_street_length_proof_l1130_113087
