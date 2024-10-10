import Mathlib

namespace complex_power_difference_l668_66853

theorem complex_power_difference (x : ℂ) : 
  x - (1 / x) = 3 * Complex.I → x^3375 - (1 / x^3375) = -18 * Complex.I := by
  sorry

end complex_power_difference_l668_66853


namespace exist_numbers_with_digit_sum_property_l668_66864

/-- Sum of digits function -/
def S (x : ℕ) : ℕ := sorry

/-- Theorem stating the existence of numbers satisfying the given conditions -/
theorem exist_numbers_with_digit_sum_property : 
  ∃ (a b c : ℕ), 
    S (a + b) < 5 ∧ 
    S (a + c) < 5 ∧ 
    S (b + c) < 5 ∧ 
    S (a + b + c) > 50 := by
  sorry

end exist_numbers_with_digit_sum_property_l668_66864


namespace updated_average_weight_average_weight_proof_l668_66811

theorem updated_average_weight (initial_avg : ℝ) (second_avg : ℝ) (third_avg : ℝ) 
  (correction1 : ℝ) (correction2 : ℝ) (correction3 : ℝ) : ℝ :=
  let initial_total := initial_avg * 5
  let second_total := second_avg * 9
  let third_total := third_avg * 12
  let corrected_total := third_total + correction1 + correction2 + correction3
  corrected_total / 12

theorem average_weight_proof :
  updated_average_weight 60 63 64 5 5 5 = 64.4167 := by
  sorry

end updated_average_weight_average_weight_proof_l668_66811


namespace intersection_A_B_values_a_b_l668_66840

-- Define sets A and B
def A : Set ℝ := {x | 4 - x^2 > 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | -2 < x ∧ x < 1} := by sorry

-- Define the quadratic inequality
def quadratic_inequality (a b : ℝ) : Set ℝ := {x | 2*x^2 + a*x + b < 0}

-- Theorem for the values of a and b
theorem values_a_b : 
  ∃ a b : ℝ, quadratic_inequality a b = B ∧ a = 4 ∧ b = -6 := by sorry

end intersection_A_B_values_a_b_l668_66840


namespace line_parallel_plane_perpendicular_implies_perpendicular_l668_66850

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Parallel relation between a line and a plane -/
def parallel (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop := sorry

/-- Theorem: If a line is parallel to a plane and another line is perpendicular to the same plane, 
    then the two lines are perpendicular to each other -/
theorem line_parallel_plane_perpendicular_implies_perpendicular 
  (m n : Line3D) (α : Plane3D) 
  (h1 : parallel m α) 
  (h2 : perpendicular_line_plane n α) : 
  perpendicular_lines m n := by
  sorry

end line_parallel_plane_perpendicular_implies_perpendicular_l668_66850


namespace inverse_proportion_problem_l668_66897

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, (x = a ∧ y = b) ∨ (x = b ∧ y = a) → x * y = k

theorem inverse_proportion_problem (a b : ℝ) :
  InverselyProportional a b →
  (∃ a₀ b₀ : ℝ, a₀ + b₀ = 60 ∧ a₀ = 3 * b₀ ∧ InverselyProportional a₀ b₀) →
  (a = -12 → b = -225/4) :=
by sorry

end inverse_proportion_problem_l668_66897


namespace total_amount_earned_l668_66876

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold. -/
theorem total_amount_earned (avg_price : ℝ) (num_pairs : ℕ) : avg_price = 9.8 → num_pairs = 50 → avg_price * (num_pairs : ℝ) = 490 := by
  sorry

end total_amount_earned_l668_66876


namespace cone_volume_ratio_l668_66815

theorem cone_volume_ratio : 
  let r_C : ℝ := 20
  let h_C : ℝ := 50
  let r_D : ℝ := 25
  let h_D : ℝ := 40
  let V_C := (1/3) * π * r_C^2 * h_C
  let V_D := (1/3) * π * r_D^2 * h_D
  V_C / V_D = 4/5 := by sorry

end cone_volume_ratio_l668_66815


namespace line_passes_through_point_l668_66879

/-- The line equation kx - y - 3k + 3 = 0 passes through the point (3,3) for all values of k. -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 : ℝ) * k - 3 - 3 * k + 3 = 0 := by sorry

end line_passes_through_point_l668_66879


namespace universal_set_determination_l668_66885

universe u

theorem universal_set_determination (U : Set ℕ) (A : Set ℕ) (h1 : A = {1, 3, 5})
  (h2 : Set.compl A = {2, 4, 6}) : U = {1, 2, 3, 4, 5, 6} := by
  sorry

end universal_set_determination_l668_66885


namespace original_rectangle_perimeter_l668_66812

/-- Given a rectangle with sides a and b, prove that if it's cut diagonally
    and then one piece is cut parallel to its shorter sides at the midpoints,
    resulting in a rectangle with perimeter 129 cm, then the perimeter of the
    original rectangle was 258 cm. -/
theorem original_rectangle_perimeter
  (a b : ℝ) 
  (h_positive : a > 0 ∧ b > 0)
  (h_final_perimeter : 2 * (a / 2 + b / 2) = 129) :
  2 * (a + b) = 258 :=
sorry

end original_rectangle_perimeter_l668_66812


namespace latest_score_is_68_l668_66810

def scores : List Int := [68, 75, 83, 94]

def is_integer_average (subset : List Int) : Prop :=
  subset.sum % subset.length = 0

theorem latest_score_is_68 :
  (∀ subset : List Int, subset ⊆ scores → is_integer_average subset) →
  scores.head? = some 68 :=
by sorry

end latest_score_is_68_l668_66810


namespace last_digit_is_three_l668_66819

/-- Represents a four-digit number -/
structure FourDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  h1 : d1 < 10
  h2 : d2 < 10
  h3 : d3 < 10
  h4 : d4 < 10

/-- Predicate for the first clue -/
def clue1 (n : FourDigitNumber) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (n.d1 = (Vector.get ⟨[1,3,5,9], rfl⟩ i) ∧ i ≠ 0) ∧
    (n.d2 = (Vector.get ⟨[1,3,5,9], rfl⟩ j) ∧ j ≠ 1)

/-- Predicate for the second clue -/
def clue2 (n : FourDigitNumber) : Prop :=
  n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3

/-- Predicate for the third clue -/
def clue3 (n : FourDigitNumber) : Prop :=
  (n.d1 = 9 ∧ (n.d2 = 0 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d2 = 0 ∧ (n.d1 = 9 ∨ n.d3 = 1 ∨ n.d4 = 3)) ∨
  (n.d3 = 1 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d4 = 3)) ∨
  (n.d4 = 3 ∧ (n.d1 = 9 ∨ n.d2 = 0 ∨ n.d3 = 1))

/-- Predicate for the fourth clue -/
def clue4 (n : FourDigitNumber) : Prop :=
  (n.d2 = 1 ∨ n.d3 = 1 ∨ n.d4 = 1) ∧ n.d1 ≠ 1

/-- Predicate for the fifth clue -/
def clue5 (n : FourDigitNumber) : Prop :=
  n.d1 ≠ 7 ∧ n.d1 ≠ 6 ∧ n.d1 ≠ 4 ∧ n.d1 ≠ 2 ∧
  n.d2 ≠ 7 ∧ n.d2 ≠ 6 ∧ n.d2 ≠ 4 ∧ n.d2 ≠ 2 ∧
  n.d3 ≠ 7 ∧ n.d3 ≠ 6 ∧ n.d3 ≠ 4 ∧ n.d3 ≠ 2 ∧
  n.d4 ≠ 7 ∧ n.d4 ≠ 6 ∧ n.d4 ≠ 4 ∧ n.d4 ≠ 2

theorem last_digit_is_three (n : FourDigitNumber) 
  (h1 : clue1 n) (h2 : clue2 n) (h3 : clue3 n) (h4 : clue4 n) (h5 : clue5 n) : 
  n.d4 = 3 := by
  sorry

end last_digit_is_three_l668_66819


namespace geometric_sequence_problem_l668_66866

/-- A geometric sequence is a sequence where the ratio of any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n with a_3 = 2 and a_5 = 8, prove that a_7 = 32 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a3 : a 3 = 2) 
    (h_a5 : a 5 = 8) : 
  a 7 = 32 := by
sorry

end geometric_sequence_problem_l668_66866


namespace youngest_not_first_or_last_l668_66816

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line with one specific person at the start or end -/
def restrictedArrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

/-- The number of people in the line -/
def n : ℕ := 5

theorem youngest_not_first_or_last :
  totalArrangements n - restrictedArrangements n = 72 := by
  sorry

#eval totalArrangements n - restrictedArrangements n

end youngest_not_first_or_last_l668_66816


namespace larger_number_puzzle_l668_66877

theorem larger_number_puzzle (x y : ℕ) : 
  x * y = 18 → x + y = 13 → max x y = 9 := by
  sorry

end larger_number_puzzle_l668_66877


namespace supplement_double_complement_30_l668_66880

def original_angle : ℝ := 30

-- Define complement
def complement (angle : ℝ) : ℝ := 90 - angle

-- Define double
def double (x : ℝ) : ℝ := 2 * x

-- Define supplement
def supplement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem supplement_double_complement_30 : 
  supplement (double (complement original_angle)) = 60 := by sorry

end supplement_double_complement_30_l668_66880


namespace matt_current_age_is_65_l668_66802

def james_age_3_years_ago : ℕ := 27
def years_since_james_27 : ℕ := 3
def years_until_matt_twice_james : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_since_james_27

def james_age_in_5_years : ℕ := james_current_age + years_until_matt_twice_james

def matt_age_in_5_years : ℕ := 2 * james_age_in_5_years

theorem matt_current_age_is_65 : matt_age_in_5_years - years_until_matt_twice_james = 65 := by
  sorry

end matt_current_age_is_65_l668_66802


namespace household_spending_theorem_l668_66856

/-- The number of households that did not spend at least $150 per month on electricity, natural gas, or water -/
def x : ℕ := 46

/-- The total number of households surveyed -/
def total_households : ℕ := 500

/-- Households spending ≥$150 on both electricity and gas -/
def both_elec_gas : ℕ := 160

/-- Households spending ≥$150 on electricity but not gas -/
def elec_not_gas : ℕ := 75

/-- Households spending ≥$150 on gas but not electricity -/
def gas_not_elec : ℕ := 80

theorem household_spending_theorem :
  x + 3 * x + both_elec_gas + elec_not_gas + gas_not_elec = total_households :=
sorry

end household_spending_theorem_l668_66856


namespace polynomial_identity_solutions_l668_66883

variable (x : ℝ)

noncomputable def p (x : ℝ) : ℝ := x^2 + x + 1

theorem polynomial_identity_solutions :
  ∃! (q₁ q₂ : ℝ → ℝ), 
    (∀ x, q₁ x = x^2 + 2*x) ∧ 
    (∀ x, q₂ x = x^2 - 1) ∧ 
    (∀ q : ℝ → ℝ, (∀ x, (p x)^2 - 2*(p x)*(q x) + (q x)^2 - 4*(p x) + 3*(q x) + 3 = 0) → 
      (q = q₁ ∨ q = q₂)) :=
by sorry

end polynomial_identity_solutions_l668_66883


namespace exists_valid_formula_l668_66882

def uses_five_twos (formula : ℕ → ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    (a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2) ∧
    ∀ n, formula n = f a b c d e
  where f := λ a b c d e => sorry -- placeholder for the actual formula

def is_valid_formula (formula : ℕ → ℕ) : Prop :=
  uses_five_twos formula ∧
  (∀ n, n ∈ Finset.range 10 → formula (n + 11) = n + 11)

theorem exists_valid_formula : ∃ formula, is_valid_formula formula := by
  sorry

#check exists_valid_formula

end exists_valid_formula_l668_66882


namespace polynomial_equivalence_l668_66831

/-- Given y = x^2 + 1/x^2, prove that x^6 + x^4 - 5x^3 + x^2 + 1 = 0 is equivalent to x^3(y+1) + y = -5x^3 -/
theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x^2 + 1/x^2) :
  x^6 + x^4 - 5*x^3 + x^2 + 1 = 0 ↔ x^3*(y+1) + y = -5*x^3 := by
  sorry

end polynomial_equivalence_l668_66831


namespace inner_triangle_area_l668_66844

/-- Given a triangle with area T, the area of the smaller triangle formed by
    joining the points that divide each side into three equal segments is 4/9 * T -/
theorem inner_triangle_area (T : ℝ) (h : T > 0) :
  ∃ (inner_area : ℝ), inner_area = (4 / 9) * T := by
  sorry

end inner_triangle_area_l668_66844


namespace rationalize_sqrt3_plus_1_l668_66889

theorem rationalize_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end rationalize_sqrt3_plus_1_l668_66889


namespace movie_ticket_price_l668_66839

/-- The price of a movie ticket and nachos, where the nachos cost half the ticket price and the total is $24. -/
def MovieTheaterVisit : Type :=
  {ticket : ℚ // ∃ (nachos : ℚ), nachos = ticket / 2 ∧ ticket + nachos = 24}

/-- Theorem stating that the price of the movie ticket is $16. -/
theorem movie_ticket_price (visit : MovieTheaterVisit) : visit.val = 16 := by
  sorry

end movie_ticket_price_l668_66839


namespace right_triangle_shorter_leg_l668_66878

theorem right_triangle_shorter_leg :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 65 →
  a ≤ b →
  a = 25 :=
by
  sorry

end right_triangle_shorter_leg_l668_66878


namespace winning_strategy_extends_l668_66841

/-- Represents the winning player for a given game state -/
inductive Winner : Type
  | Player1 : Winner
  | Player2 : Winner

/-- Represents the game state -/
structure GameState :=
  (t : ℕ)  -- Current number on the blackboard
  (a : ℕ)  -- First subtraction option
  (b : ℕ)  -- Second subtraction option

/-- Determines the winner of the game given a game state -/
def winningPlayer (state : GameState) : Winner :=
  sorry

/-- Theorem stating that if Player 1 wins for x, they also win for x + 2005k -/
theorem winning_strategy_extends (x k a b : ℕ) :
  (1 ≤ x) →
  (x ≤ 2005) →
  (0 < a) →
  (0 < b) →
  (a + b = 2005) →
  (winningPlayer { t := x, a := a, b := b } = Winner.Player1) →
  (winningPlayer { t := x + 2005 * k, a := a, b := b } = Winner.Player1) :=
by
  sorry

end winning_strategy_extends_l668_66841


namespace first_three_consecutive_fives_l668_66858

/-- The sequence of digits formed by concatenating natural numbers -/
def digitSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => sorry  -- Definition of the sequence

/-- The function that returns the digit at a given position in the sequence -/
def digitAt (position : ℕ) : ℕ := sorry

/-- Theorem stating the positions of the first occurrence of three consecutive '5' digits -/
theorem first_three_consecutive_fives :
  ∃ (start : ℕ), start = 100 ∧
    digitAt start = 5 ∧
    digitAt (start + 1) = 5 ∧
    digitAt (start + 2) = 5 ∧
    (∀ (pos : ℕ), pos < start → ¬(digitAt pos = 5 ∧ digitAt (pos + 1) = 5 ∧ digitAt (pos + 2) = 5)) :=
  sorry


end first_three_consecutive_fives_l668_66858


namespace complex_sum_equals_eleven_l668_66847

/-- Given complex numbers a and b, prove that a + 3b = 11 -/
theorem complex_sum_equals_eleven (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + I) :
  a + 3*b = 11 := by
  sorry

end complex_sum_equals_eleven_l668_66847


namespace smallest_base_perfect_square_l668_66843

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∀ k : ℕ, k > 5 ∧ k < b → ¬∃ n : ℕ, 4 * k + 5 = n^2) → 
  ∃ n : ℕ, 4 * b + 5 = n^2 → b = 11 := by
sorry

end smallest_base_perfect_square_l668_66843


namespace cubic_equation_solutions_no_solution_for_2891_l668_66848

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n ∧
                (y-x)^3 - 3*(y-x)*(-x)^2 + (-x)^3 = n ∧
                (-y)^3 - 3*(-y)*(x-y)^2 + (x-y)^3 = n) :=
sorry

theorem no_solution_for_2891 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 :=
sorry

end cubic_equation_solutions_no_solution_for_2891_l668_66848


namespace revenue_change_after_price_and_quantity_change_l668_66886

theorem revenue_change_after_price_and_quantity_change 
  (P Q : ℝ) (P_new Q_new R R_new : ℝ) 
  (h1 : P_new = 0.8 * P) 
  (h2 : Q_new = 1.6 * Q) 
  (h3 : R = P * Q) 
  (h4 : R_new = P_new * Q_new) : 
  R_new = 1.28 * R := by sorry

end revenue_change_after_price_and_quantity_change_l668_66886


namespace shortest_track_length_l668_66890

theorem shortest_track_length (melanie_piece_length martin_piece_length : ℕ) 
  (h1 : melanie_piece_length = 8)
  (h2 : martin_piece_length = 20) :
  Nat.lcm melanie_piece_length martin_piece_length = 40 := by
sorry

end shortest_track_length_l668_66890


namespace sin_cos_product_given_tan_l668_66808

theorem sin_cos_product_given_tan (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end sin_cos_product_given_tan_l668_66808


namespace smallest_positive_largest_negative_smallest_abs_rational_l668_66871

theorem smallest_positive_largest_negative_smallest_abs_rational :
  ∃ (x y : ℤ) (z : ℚ),
    (∀ n : ℤ, n > 0 → x ≤ n) ∧
    (∀ n : ℤ, n < 0 → y ≥ n) ∧
    (∀ q : ℚ, |z| ≤ |q|) ∧
    2 * x + 3 * y + 4 * z = -1 :=
by sorry

end smallest_positive_largest_negative_smallest_abs_rational_l668_66871


namespace quadratic_function_proof_l668_66899

/-- A quadratic function passing through three given points -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_proof (a b c : ℝ) :
  (quadratic_function a b c 1 = 5) ∧
  (quadratic_function a b c 0 = 3) ∧
  (quadratic_function a b c (-1) = -3) →
  (∀ x, quadratic_function a b c x = -2 * x^2 + 4 * x + 3) ∧
  (∃ x y, x = 1 ∧ y = 5 ∧ ∀ t, quadratic_function a b c t ≤ quadratic_function a b c x) :=
by sorry

end quadratic_function_proof_l668_66899


namespace max_perpendicular_pairs_l668_66872

/-- A line in a plane -/
structure Line

/-- A perpendicular pair of lines -/
structure PerpendicularPair (Line : Type) where
  line1 : Line
  line2 : Line

/-- A configuration of lines in a plane -/
structure PlaneConfiguration where
  lines : Finset Line
  perpendicular_pairs : Finset (PerpendicularPair Line)
  line_count : lines.card = 20

/-- The theorem stating the maximum number of perpendicular pairs -/
theorem max_perpendicular_pairs (config : PlaneConfiguration) :
  ∃ (max_config : PlaneConfiguration), 
    ∀ (c : PlaneConfiguration), c.perpendicular_pairs.card ≤ max_config.perpendicular_pairs.card ∧
    max_config.perpendicular_pairs.card = 100 :=
  sorry

end max_perpendicular_pairs_l668_66872


namespace complex_real_condition_l668_66828

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (2 + m * Complex.I) / (1 + Complex.I)
  (z.im = 0) → m = 2 := by
  sorry

end complex_real_condition_l668_66828


namespace expression_simplification_l668_66894

theorem expression_simplification (y : ℝ) : 3*y + 4*y^2 + 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 6 := by
  sorry

end expression_simplification_l668_66894


namespace angle_D_measure_l668_66884

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angle_E : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angle_F_eq_three_E : True

/-- Theorem: In the given isosceles triangle, the measure of angle D is 77 1/7 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) : 
  (3 * t.angle_E : ℝ) = 77 + 1/7 := by
  sorry

end angle_D_measure_l668_66884


namespace sum_85_to_100_l668_66893

def sum_consecutive_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_85_to_100 :
  sum_consecutive_integers 85 100 = 1480 :=
by sorry

end sum_85_to_100_l668_66893


namespace min_sum_squares_l668_66834

def S : Finset Int := {-6, -4, -3, -1, 1, 3, 5, 8}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 5) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 5 :=
sorry

end min_sum_squares_l668_66834


namespace exists_square_between_consecutive_prime_sums_l668_66842

-- Define S_n as the sum of the first n prime numbers
def S (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_square_between_consecutive_prime_sums : 
  ∃ k : ℕ, S 2023 < k^2 ∧ k^2 < S 2024 := by sorry

end exists_square_between_consecutive_prime_sums_l668_66842


namespace parabola_properties_l668_66830

/-- A parabola with coefficient a < 0 intersecting x-axis at (-3,0) and (1,0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neg : a < 0
  h_root1 : a * (-3)^2 + b * (-3) + c = 0
  h_root2 : a * 1^2 + b * 1 + c = 0

theorem parabola_properties (p : Parabola) :
  (p.b^2 - 4 * p.a * p.c > 0) ∧ (3 * p.b + 2 * p.c = 0) := by
  sorry

end parabola_properties_l668_66830


namespace roots_difference_is_one_l668_66887

-- Define the polynomial
def f (x : ℝ) : ℝ := 64 * x^3 - 144 * x^2 + 92 * x - 15

-- Define the roots
def roots : Set ℝ := {x : ℝ | f x = 0}

-- Define the arithmetic progression property
def is_arithmetic_progression (s : Set ℝ) : Prop :=
  ∃ (a d : ℝ), s = {a - d, a, a + d}

-- Theorem statement
theorem roots_difference_is_one :
  is_arithmetic_progression roots →
  ∃ (r₁ r₂ r₃ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₃ ∈ roots ∧
  r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ - r₁ = 1 :=
sorry

end roots_difference_is_one_l668_66887


namespace second_largest_of_5_8_4_l668_66857

def second_largest (a b c : ℕ) : ℕ :=
  if a ≥ b ∧ b ≥ c then b
  else if a ≥ c ∧ c ≥ b then c
  else if b ≥ a ∧ a ≥ c then a
  else if b ≥ c ∧ c ≥ a then c
  else if c ≥ a ∧ a ≥ b then a
  else b

theorem second_largest_of_5_8_4 : second_largest 5 8 4 = 5 := by
  sorry

end second_largest_of_5_8_4_l668_66857


namespace division_problem_l668_66809

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 1375 → 
  divisor = 66 → 
  remainder = 55 → 
  dividend = divisor * quotient + remainder → 
  quotient = 20 := by
sorry

end division_problem_l668_66809


namespace zoo_zebra_count_l668_66860

theorem zoo_zebra_count :
  ∀ (penguins zebras tigers zookeepers : ℕ),
    penguins = 30 →
    tigers = 8 →
    zookeepers = 12 →
    (penguins + zebras + tigers + zookeepers) = 
      (2 * penguins + 4 * zebras + 4 * tigers + 2 * zookeepers) - 132 →
    zebras = 22 := by
  sorry

end zoo_zebra_count_l668_66860


namespace cubic_polynomials_common_roots_l668_66862

theorem cubic_polynomials_common_roots :
  ∃! (c d : ℝ), ∃ (r s : ℝ),
    r ≠ s ∧
    (r^3 + c*r^2 + 17*r + 10 = 0) ∧
    (r^3 + d*r^2 + 22*r + 14 = 0) ∧
    (s^3 + c*s^2 + 17*s + 10 = 0) ∧
    (s^3 + d*s^2 + 22*s + 14 = 0) ∧
    (∀ (x : ℝ), x ≠ r ∧ x ≠ s →
      (x^3 + c*x^2 + 17*x + 10 ≠ 0) ∨
      (x^3 + d*x^2 + 22*x + 14 ≠ 0)) ∧
    c = 8 ∧
    d = 9 := by
  sorry

end cubic_polynomials_common_roots_l668_66862


namespace baseball_cards_per_pack_l668_66807

theorem baseball_cards_per_pack : 
  ∀ (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    (num_people * cards_per_person) / total_packs = 20 := by
  sorry

end baseball_cards_per_pack_l668_66807


namespace vector_magnitude_l668_66829

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 5)
  (h2 : ‖2 • a + b‖ = 5 * Real.sqrt 3)
  (h3 : ‖a - b‖ = 5 * Real.sqrt 2) :
  ‖a‖ = 5 * Real.sqrt 6 / 3 := by
  sorry

end vector_magnitude_l668_66829


namespace functional_equation_solution_l668_66833

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) + x * y = f x * f y) →
  (∀ x : ℝ, f x = 1 - x) ∨ (∀ x : ℝ, f x = x + 1) :=
by sorry

end functional_equation_solution_l668_66833


namespace brown_eyes_light_brown_skin_l668_66813

/-- Represents the characteristics of the group of girls -/
structure GirlGroup where
  total : Nat
  blue_eyes_fair_skin : Nat
  light_brown_skin : Nat
  brown_eyes : Nat

/-- Theorem stating the number of girls with brown eyes and light brown skin -/
theorem brown_eyes_light_brown_skin (g : GirlGroup) 
  (h1 : g.total = 50)
  (h2 : g.blue_eyes_fair_skin = 14)
  (h3 : g.light_brown_skin = 31)
  (h4 : g.brown_eyes = 18) :
  g.brown_eyes - (g.total - g.light_brown_skin - g.blue_eyes_fair_skin) = 13 := by
  sorry

#check brown_eyes_light_brown_skin

end brown_eyes_light_brown_skin_l668_66813


namespace final_sum_after_transformation_l668_66837

theorem final_sum_after_transformation (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end final_sum_after_transformation_l668_66837


namespace shifted_line_not_in_third_quadrant_l668_66814

/-- The original line equation -/
def original_line (x : ℝ) : ℝ := -2 * x - 1

/-- The shifted line equation -/
def shifted_line (x : ℝ) : ℝ := -2 * x + 5

/-- The shift amount -/
def shift : ℝ := 3

/-- Theorem: The shifted line does not intersect the third quadrant -/
theorem shifted_line_not_in_third_quadrant :
  ∀ x y : ℝ, y = shifted_line x → ¬(x < 0 ∧ y < 0) :=
sorry

end shifted_line_not_in_third_quadrant_l668_66814


namespace optimal_dimensions_maximize_volume_unique_maximum_volume_l668_66852

/-- Represents the volume of a rectangular frame as a function of its width. -/
def volume (x : ℝ) : ℝ := 2 * x^2 * (4.5 - 3*x)

/-- The maximum volume of the rectangular frame. -/
def max_volume : ℝ := 3

/-- The width that maximizes the volume. -/
def optimal_width : ℝ := 1

/-- The length that maximizes the volume. -/
def optimal_length : ℝ := 2

/-- The height that maximizes the volume. -/
def optimal_height : ℝ := 1.5

/-- Theorem stating that the given dimensions maximize the volume of the rectangular frame. -/
theorem optimal_dimensions_maximize_volume :
  (∀ x, 0 < x → x < 3/2 → volume x ≤ max_volume) ∧
  volume optimal_width = max_volume ∧
  optimal_length = 2 * optimal_width ∧
  optimal_height = 4.5 - 3 * optimal_width :=
sorry

/-- Theorem stating that the maximum volume is unique. -/
theorem unique_maximum_volume :
  ∀ x, 0 < x → x < 3/2 → volume x = max_volume → x = optimal_width :=
sorry

end optimal_dimensions_maximize_volume_unique_maximum_volume_l668_66852


namespace cross_product_perpendicular_l668_66804

/-- The cross product of two 3D vectors -/
def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => match i with
    | 0 => v 1 * w 2 - v 2 * w 1
    | 1 => v 2 * w 0 - v 0 * w 2
    | 2 => v 0 * w 1 - v 1 * w 0

/-- The dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

theorem cross_product_perpendicular (v w : Fin 3 → ℝ) :
  let v1 : Fin 3 → ℝ := fun i => match i with
    | 0 => 3
    | 1 => -2
    | 2 => 4
  let v2 : Fin 3 → ℝ := fun i => match i with
    | 0 => 1
    | 1 => 5
    | 2 => -3
  let cp := cross_product v1 v2
  cp 0 = -14 ∧ cp 1 = 13 ∧ cp 2 = 17 ∧
  dot_product cp v1 = 0 ∧ dot_product cp v2 = 0 :=
by
  sorry

end cross_product_perpendicular_l668_66804


namespace expression_equals_sum_l668_66801

theorem expression_equals_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let numerator := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c := by
  sorry

end expression_equals_sum_l668_66801


namespace new_tax_rate_is_28_percent_l668_66888

/-- Calculates the new tax rate given the initial conditions --/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  100 * (initial_rate * income - savings) / income

/-- Proves that the new tax rate is 28% given the initial conditions --/
theorem new_tax_rate_is_28_percent :
  let initial_rate : ℚ := 42 / 100
  let income : ℚ := 34500
  let savings : ℚ := 4830
  calculate_new_tax_rate initial_rate income savings = 28 := by
  sorry

#eval calculate_new_tax_rate (42/100) 34500 4830

end new_tax_rate_is_28_percent_l668_66888


namespace min_value_sum_of_squares_l668_66870

theorem min_value_sum_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (m : ℝ), m = 16 - 2 * Real.sqrt 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m :=
by sorry

end min_value_sum_of_squares_l668_66870


namespace polygon_diagonals_l668_66865

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) : (n - 3 = 4) → n = 7 := by
  sorry

end polygon_diagonals_l668_66865


namespace little_red_journey_l668_66803

/-- The distance from Little Red's house to school in kilometers -/
def distance_to_school : ℝ := 1.5

/-- Little Red's average speed uphill in kilometers per hour -/
def speed_uphill : ℝ := 2

/-- Little Red's average speed downhill in kilometers per hour -/
def speed_downhill : ℝ := 3

/-- The total time taken for the journey in minutes -/
def total_time : ℝ := 18

/-- The system of equations describing Little Red's journey to school -/
def journey_equations (x y : ℝ) : Prop :=
  (speed_uphill / 60 * x + speed_downhill / 60 * y = distance_to_school) ∧
  (x + y = total_time)

theorem little_red_journey :
  ∀ x y : ℝ, journey_equations x y ↔
    (2 / 60 * x + 3 / 60 * y = 1.5) ∧ (x + y = 18) :=
sorry

end little_red_journey_l668_66803


namespace negation_of_existence_negation_of_proposition_l668_66846

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ ∀ n, ¬p n :=
by sorry

theorem negation_of_proposition : 
  (¬ ∃ n : ℕ, 3^n > 2018) ↔ (∀ n : ℕ, 3^n ≤ 2018) :=
by sorry

end negation_of_existence_negation_of_proposition_l668_66846


namespace remaining_pencils_l668_66851

/-- The number of pencils remaining in a drawer after some are taken. -/
def pencils_remaining (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem stating that 12 pencils remain in the drawer. -/
theorem remaining_pencils :
  pencils_remaining 34 22 = 12 := by
  sorry

end remaining_pencils_l668_66851


namespace overhead_cost_reduction_l668_66825

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- Calculates the total cost of manufacturing a car --/
def total_cost (cost : CarCost) : ℝ :=
  cost.raw_material + cost.labor + cost.overhead

theorem overhead_cost_reduction 
  (initial_cost : CarCost) 
  (new_cost : CarCost) 
  (h1 : initial_cost.raw_material = (4/9) * total_cost initial_cost)
  (h2 : initial_cost.labor = (3/9) * total_cost initial_cost)
  (h3 : initial_cost.overhead = (2/9) * total_cost initial_cost)
  (h4 : new_cost.raw_material = 1.1 * initial_cost.raw_material)
  (h5 : new_cost.labor = 1.08 * initial_cost.labor)
  (h6 : total_cost new_cost = 1.06 * total_cost initial_cost) :
  new_cost.overhead = 0.95 * initial_cost.overhead :=
sorry

end overhead_cost_reduction_l668_66825


namespace percentage_same_grade_l668_66892

def total_students : ℕ := 50

def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 8
def same_grade_D : ℕ := 2
def same_grade_F : ℕ := 1

def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D + same_grade_F

theorem percentage_same_grade : 
  (total_same_grade : ℚ) / total_students * 100 = 40 := by
  sorry

end percentage_same_grade_l668_66892


namespace gcd_sum_lcm_l668_66855

theorem gcd_sum_lcm (a b : ℤ) : Nat.gcd (a + b).natAbs (Nat.lcm a.natAbs b.natAbs) = Nat.gcd a.natAbs b.natAbs := by
  sorry

end gcd_sum_lcm_l668_66855


namespace gcd_of_three_numbers_l668_66827

theorem gcd_of_three_numbers : Nat.gcd 7254 (Nat.gcd 10010 22554) = 26 := by
  sorry

end gcd_of_three_numbers_l668_66827


namespace greatest_divisor_four_consecutive_integers_l668_66835

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  (∃ k : ℕ, k > 12 ∧ (∀ m : ℕ, m > 0 → k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) →
  False :=
sorry

end greatest_divisor_four_consecutive_integers_l668_66835


namespace marys_speed_l668_66845

theorem marys_speed (mary_hill_length ann_hill_length ann_speed time_difference : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : ann_hill_length = 800)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13)
  (h5 : ann_hill_length / ann_speed = mary_hill_length / mary_speed + time_difference) :
  mary_speed = 90 :=
by
  sorry

#check marys_speed

end marys_speed_l668_66845


namespace geometric_sequence_common_ratio_l668_66875

/-- Given a geometric sequence {a_n} where a_2010 = 8a_2007, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h : a 2010 = 8 * a 2007) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 := by
  sorry

end geometric_sequence_common_ratio_l668_66875


namespace tangent_ratio_inequality_l668_66822

theorem tangent_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) :
  Real.tan α / α < Real.tan β / β := by
  sorry

end tangent_ratio_inequality_l668_66822


namespace robin_gum_count_l668_66824

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 9

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 135 := by
  sorry

end robin_gum_count_l668_66824


namespace technician_salary_l668_66868

theorem technician_salary (total_workers : Nat) (technicians : Nat) (avg_salary : ℕ) 
  (non_tech_avg : ℕ) (h1 : total_workers = 12) (h2 : technicians = 6) 
  (h3 : avg_salary = 9000) (h4 : non_tech_avg = 6000) : 
  (total_workers * avg_salary - (total_workers - technicians) * non_tech_avg) / technicians = 12000 :=
sorry

end technician_salary_l668_66868


namespace solution_count_l668_66820

-- Define the equations
def equation1 (x y : ℂ) : Prop := y = (x + 2)^3
def equation2 (x y : ℂ) : Prop := x * y + 2 * y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the count of real and imaginary solutions
def real_solution_count : ℕ := 2
def imaginary_solution_count : ℕ := 2

-- Theorem statement
theorem solution_count :
  (∃ (s : Finset (ℂ × ℂ)), s.card = real_solution_count + imaginary_solution_count ∧
    (∀ (p : ℂ × ℂ), p ∈ s ↔ is_solution p.1 p.2) ∧
    (∃ (r : Finset (ℂ × ℂ)), r ⊆ s ∧ r.card = real_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ r → p.1.im = 0 ∧ p.2.im = 0)) ∧
    (∃ (i : Finset (ℂ × ℂ)), i ⊆ s ∧ i.card = imaginary_solution_count ∧
      (∀ (p : ℂ × ℂ), p ∈ i → p.1.im ≠ 0 ∨ p.2.im ≠ 0))) :=
sorry

end solution_count_l668_66820


namespace magnitude_relationship_l668_66854

theorem magnitude_relationship
  (a b c d : ℝ)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_positive : d > 0)
  (x : ℝ)
  (h_x : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ)
  (h_y : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ)
  (h_z : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end magnitude_relationship_l668_66854


namespace suit_price_increase_l668_66800

theorem suit_price_increase (original_price : ℝ) (discounted_price : ℝ) :
  original_price = 160 →
  discounted_price = 150 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 25 ∧
    discounted_price = (original_price * (1 + increase_percentage / 100)) * 0.75 :=
by sorry

end suit_price_increase_l668_66800


namespace line_slopes_problem_l668_66838

theorem line_slopes_problem (k₁ k₂ b : ℝ) : 
  (2 * k₁^2 - 3 * k₁ - b = 0) → 
  (2 * k₂^2 - 3 * k₂ - b = 0) → 
  ((k₁ * k₂ = -1) → b = 2) ∧ 
  ((k₁ = k₂) → b = -9/8) := by
sorry

end line_slopes_problem_l668_66838


namespace g_is_odd_and_f_negative_two_l668_66898

/-- The function f(x) -/
noncomputable def f (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n + x^2 - 1

/-- The function g(x) -/
noncomputable def g (x m n : ℝ) : ℝ := (2^x - 2^(-x)) * m + (x^3 + x) * n

theorem g_is_odd_and_f_negative_two (m n : ℝ) :
  (∀ x, g (-x) m n = -g x m n) ∧ (f 2 m n = 8 → f (-2) m n = -2) :=
sorry

end g_is_odd_and_f_negative_two_l668_66898


namespace total_grapes_is_83_l668_66832

/-- The number of grapes in Rob's bowl -/
def rob_grapes : ℕ := 25

/-- The number of grapes in Allie's bowl -/
def allie_grapes : ℕ := rob_grapes + 2

/-- The number of grapes in Allyn's bowl -/
def allyn_grapes : ℕ := allie_grapes + 4

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := rob_grapes + allie_grapes + allyn_grapes

theorem total_grapes_is_83 : total_grapes = 83 := by
  sorry

end total_grapes_is_83_l668_66832


namespace flagstaff_height_is_correct_l668_66891

/-- The height of the flagstaff in meters -/
def flagstaff_height : ℝ := 17.5

/-- The length of the flagstaff's shadow in meters -/
def flagstaff_shadow : ℝ := 40.25

/-- The height of the building in meters -/
def building_height : ℝ := 12.5

/-- The length of the building's shadow in meters -/
def building_shadow : ℝ := 28.75

/-- Theorem stating that the calculated flagstaff height is correct -/
theorem flagstaff_height_is_correct :
  flagstaff_height = (building_height * flagstaff_shadow) / building_shadow :=
by sorry

end flagstaff_height_is_correct_l668_66891


namespace equation_with_integer_roots_l668_66867

/-- Given that the equation (x-a)(x-8) - 1 = 0 has two integer roots, prove that a = 8 -/
theorem equation_with_integer_roots (a : ℤ) :
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ (x₁ - a) * (x₁ - 8) - 1 = 0 ∧ (x₂ - a) * (x₂ - 8) - 1 = 0) →
  a = 8 := by
  sorry


end equation_with_integer_roots_l668_66867


namespace expression_simplification_l668_66895

theorem expression_simplification (a : ℤ) (h : a = 2020) : 
  (a^4 - 3*a^3*(a+1) + 4*a^2*(a+1)^2 - (a+1)^4 + 1) / (a*(a+1)) = a^2 - 2 := by
  sorry

end expression_simplification_l668_66895


namespace right_triangle_sine_roots_l668_66863

theorem right_triangle_sine_roots (A B C : Real) (p q : Real) :
  0 < A ∧ A < Real.pi / 2 →
  0 < B ∧ B < Real.pi / 2 →
  C = Real.pi / 2 →
  A + B + C = Real.pi →
  (∀ x, x^2 + p*x + q = 0 ↔ x = Real.sin A ∨ x = Real.sin B) →
  (p^2 - 2*q = 1 ∧ -Real.sqrt 2 ≤ p ∧ p < -1 ∧ 0 < q ∧ q ≤ 1/2) ∧
  (∀ x, x^2 + p*x + q = 0 → (x = Real.sin A ∨ x = Real.sin B)) :=
by sorry

end right_triangle_sine_roots_l668_66863


namespace towel_sets_cost_l668_66896

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.2

def total_cost : ℚ := guest_sets * guest_price + master_sets * master_price
def discount_amount : ℚ := total_cost * discount_rate
def final_cost : ℚ := total_cost - discount_amount

theorem towel_sets_cost : final_cost = 224 := by
  sorry

end towel_sets_cost_l668_66896


namespace smallest_right_triangle_area_l668_66821

/-- The smallest area of a right triangle with one side 6 and another side x < 6 -/
theorem smallest_right_triangle_area :
  ∀ x : ℝ, x < 6 →
  (5 * Real.sqrt 11) / 2 ≤ min (3 * x) ((x * Real.sqrt (36 - x^2)) / 2) :=
by sorry

end smallest_right_triangle_area_l668_66821


namespace translation_of_point_l668_66861

/-- Given a point A with coordinates (-2, 3) in a Cartesian coordinate system,
    prove that translating it 3 units right and 5 units down results in
    point B with coordinates (1, -2). -/
theorem translation_of_point (A B : ℝ × ℝ) :
  A = (-2, 3) →
  B.1 = A.1 + 3 →
  B.2 = A.2 - 5 →
  B = (1, -2) := by
  sorry

end translation_of_point_l668_66861


namespace discount_order_difference_l668_66823

/-- The difference in final price when applying discounts in different orders -/
theorem discount_order_difference : 
  let original_price : ℚ := 25
  let flat_discount : ℚ := 4
  let percentage_discount : ℚ := 0.2
  let price_flat_then_percent : ℚ := (original_price - flat_discount) * (1 - percentage_discount)
  let price_percent_then_flat : ℚ := (original_price * (1 - percentage_discount)) - flat_discount
  (price_flat_then_percent - price_percent_then_flat) * 100 = 80
  := by sorry

end discount_order_difference_l668_66823


namespace imaginary_part_of_z_l668_66826

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l668_66826


namespace geometric_sequence_property_l668_66818

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 2 * a 3 = 5 →
  a 7 * a 8 * a 9 = 10 →
  a 4 * a 5 * a 6 = 5 * Real.sqrt 2 := by
  sorry

end geometric_sequence_property_l668_66818


namespace janine_read_150_pages_l668_66873

/-- The number of pages Janine read in two months -/
def pages_read_in_two_months (books_last_month : ℕ) (books_this_month_factor : ℕ) (pages_per_book : ℕ) : ℕ :=
  (books_last_month + books_last_month * books_this_month_factor) * pages_per_book

/-- Theorem stating that Janine read 150 pages in two months -/
theorem janine_read_150_pages :
  pages_read_in_two_months 5 2 10 = 150 := by
  sorry

#eval pages_read_in_two_months 5 2 10

end janine_read_150_pages_l668_66873


namespace min_distance_between_curves_l668_66836

noncomputable def f (x : ℝ) : ℝ := x^3
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_between_curves :
  ∃ (min_val : ℝ), min_val = (1/3 : ℝ) + (1/3 : ℝ) * Real.log 3 ∧
  ∀ (x : ℝ), x > 0 → |f x - g x| ≥ min_val :=
sorry

end min_distance_between_curves_l668_66836


namespace least_positive_integer_with_remainders_l668_66874

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ (N : ℕ), N > 0 ∧ 
    N % 7 = 6 ∧
    N % 8 = 7 ∧
    N % 9 = 8 ∧
    N % 10 = 9 ∧
    N % 11 = 10 ∧
    N % 12 = 11 → M ≤ N) ∧
  M = 27719 :=
by sorry

end least_positive_integer_with_remainders_l668_66874


namespace derivative_log2_l668_66805

open Real

theorem derivative_log2 (x : ℝ) (h : x > 0) :
  deriv (fun x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end derivative_log2_l668_66805


namespace power_calculation_l668_66869

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end power_calculation_l668_66869


namespace road_trip_days_l668_66859

/-- Proves that the number of days of the road trip is 3, given the driving hours of Jade and Krista and the total hours driven. -/
theorem road_trip_days (jade_hours krista_hours total_hours : ℕ) 
  (h1 : jade_hours = 8)
  (h2 : krista_hours = 6)
  (h3 : total_hours = 42) :
  (total_hours : ℚ) / (jade_hours + krista_hours : ℚ) = 3 := by
  sorry

end road_trip_days_l668_66859


namespace H_composition_equals_neg_one_l668_66849

/-- The function H defined as H(x) = x^2 - 2x - 1 -/
def H (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that H(H(H(H(H(2))))) = -1 -/
theorem H_composition_equals_neg_one : H (H (H (H (H 2)))) = -1 := by
  sorry

end H_composition_equals_neg_one_l668_66849


namespace greg_age_l668_66806

/-- Given the ages and relationships of siblings, prove Greg's age -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
sorry

end greg_age_l668_66806


namespace max_product_under_constraint_l668_66881

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_constraint : 6 * a + 5 * b = 45) :
  a * b ≤ 135 / 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 5 * b₀ = 45 ∧ a₀ * b₀ = 135 / 8 :=
by sorry

end max_product_under_constraint_l668_66881


namespace equation_solution_l668_66817

theorem equation_solution :
  let f (n : ℝ) := (3 - 2*n) / (n + 2) + (3*n - 9) / (3 - 2*n)
  let n₁ := (25 + Real.sqrt 13) / 18
  let n₂ := (25 - Real.sqrt 13) / 18
  f n₁ = 2 ∧ f n₂ = 2 := by sorry

end equation_solution_l668_66817
