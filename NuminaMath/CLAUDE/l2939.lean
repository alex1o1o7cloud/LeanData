import Mathlib

namespace x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l2939_293926

-- Define propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Theorem 1
theorem x_range_when_a_is_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem 2
theorem a_range_when_q_necessary_not_sufficient (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l2939_293926


namespace cookies_with_three_cups_l2939_293935

/- Define the rate of cookies per cup of flour -/
def cookies_per_cup (total_cookies : ℕ) (total_cups : ℕ) : ℚ :=
  total_cookies / total_cups

/- Define the function to calculate cookies from cups of flour -/
def cookies_from_cups (rate : ℚ) (cups : ℕ) : ℚ :=
  rate * cups

/- Theorem statement -/
theorem cookies_with_three_cups 
  (h1 : cookies_per_cup 24 4 = 6) 
  (h2 : cookies_from_cups (cookies_per_cup 24 4) 3 = 18) : 
  ℕ := by
  sorry

end cookies_with_three_cups_l2939_293935


namespace min_value_expression_l2939_293989

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * c) / (3 * a + b) + (6 * a) / (b + 3 * c) + (2 * b) / (a + c) ≥ 12 := by
  sorry

end min_value_expression_l2939_293989


namespace triangle_must_be_obtuse_l2939_293988

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (t.a = 2 * t.b ∨ t.b = 2 * t.c ∨ t.c = 2 * t.a) ∧
  (t.A = Real.pi / 6 ∨ t.B = Real.pi / 6 ∨ t.C = Real.pi / 6)

-- Define an obtuse triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

-- Theorem statement
theorem triangle_must_be_obtuse (t : Triangle) (h : TriangleProperties t) : IsObtuseTriangle t :=
sorry

end triangle_must_be_obtuse_l2939_293988


namespace triangle_segment_length_l2939_293982

/-- Triangle ABC with points D and E on BC -/
structure TriangleABC where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Length of CD -/
  CD : ℝ
  /-- Ratio of BE to EC -/
  BE_EC_ratio : ℝ
  /-- Equality of angles BAE and CAD -/
  angle_equality : Bool

/-- The main theorem -/
theorem triangle_segment_length 
  (t : TriangleABC) 
  (h1 : t.AB = 12) 
  (h2 : t.BC = 16) 
  (h3 : t.CA = 15) 
  (h4 : t.CD = 5) 
  (h5 : t.BE_EC_ratio = 3) 
  (h6 : t.angle_equality = true) : 
  ∃ (BE : ℝ), BE = 5.5 := by
  sorry

end triangle_segment_length_l2939_293982


namespace crow_eating_time_l2939_293925

/-- The time it takes for a crow to eat a fraction of nuts -/
def eat_time (total_fraction : ℚ) (time : ℚ) : ℚ := total_fraction / time

theorem crow_eating_time :
  let quarter_time : ℚ := 5
  let quarter_fraction : ℚ := 1/4
  let fifth_fraction : ℚ := 1/5
  let rate := eat_time quarter_fraction quarter_time
  eat_time fifth_fraction rate = 4 := by sorry

end crow_eating_time_l2939_293925


namespace function_identity_l2939_293946

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h_continuous : Continuous f)
variable (h_inequality : ∀ (a b c : ℝ) (x : ℝ), f (a * x^2 + b * x + c) ≥ a * (f x)^2 + b * (f x) + c)

-- Theorem statement
theorem function_identity : f = id := by
  sorry

end function_identity_l2939_293946


namespace derivative_at_three_l2939_293992

/-- Given a function f with f(x) = 3x^2 + 2xf'(1) for all x, prove that f'(3) = 6 -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 1)) :
  deriv f 3 = 6 := by
  sorry

end derivative_at_three_l2939_293992


namespace election_invalid_votes_l2939_293961

theorem election_invalid_votes 
  (total_polled : ℕ) 
  (vote_difference : ℕ) 
  (losing_percentage : ℚ) :
  total_polled = 850 →
  vote_difference = 500 →
  losing_percentage = 1/5 →
  (∃ (invalid_votes : ℕ), invalid_votes = 17) :=
by sorry

end election_invalid_votes_l2939_293961


namespace root_in_interval_l2939_293931

def f (x : ℝ) : ℝ := x^3 + x + 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo (-2 : ℝ) (-1), f x = 0 :=
sorry

end root_in_interval_l2939_293931


namespace tom_waits_six_months_l2939_293910

/-- Represents Tom's medication and doctor visit costs --/
structure MedicationCosts where
  pills_per_day : ℕ
  doctor_visit_cost : ℕ
  pill_cost : ℕ
  insurance_coverage : ℚ
  total_annual_cost : ℕ

/-- Calculates the number of months between doctor visits --/
def months_between_visits (costs : MedicationCosts) : ℚ :=
  let annual_medication_cost := costs.pills_per_day * 365 * costs.pill_cost * (1 - costs.insurance_coverage)
  let annual_doctor_cost := costs.total_annual_cost - annual_medication_cost
  let visits_per_year := annual_doctor_cost / costs.doctor_visit_cost
  12 / visits_per_year

/-- Theorem stating that Tom waits 6 months between doctor visits --/
theorem tom_waits_six_months (costs : MedicationCosts) 
  (h1 : costs.pills_per_day = 2)
  (h2 : costs.doctor_visit_cost = 400)
  (h3 : costs.pill_cost = 5)
  (h4 : costs.insurance_coverage = 4/5)
  (h5 : costs.total_annual_cost = 1530) :
  months_between_visits costs = 6 := by
  sorry


end tom_waits_six_months_l2939_293910


namespace all_cells_equal_l2939_293921

/-- Represents a 10x10 board with integer values -/
def Board := Fin 10 → Fin 10 → ℤ

/-- Predicate to check if a board satisfies the given conditions -/
def satisfies_conditions (b : Board) : Prop :=
  ∃ d : ℤ,
    (∀ i : Fin 10, b i i = d) ∧
    (∀ i j : Fin 10, b i j ≤ d)

/-- Theorem stating that if a board satisfies the conditions, all cells are equal -/
theorem all_cells_equal (b : Board) (h : satisfies_conditions b) :
    ∃ d : ℤ, ∀ i j : Fin 10, b i j = d := by
  sorry


end all_cells_equal_l2939_293921


namespace pipe_cutting_time_l2939_293949

/-- The time needed to cut a pipe into sections -/
def cut_time (sections : ℕ) (time_per_cut : ℕ) : ℕ :=
  (sections - 1) * time_per_cut

/-- Theorem: The time needed to cut a pipe into 5 sections is 24 minutes -/
theorem pipe_cutting_time : cut_time 5 6 = 24 := by
  sorry

end pipe_cutting_time_l2939_293949


namespace initial_speed_problem_l2939_293929

theorem initial_speed_problem (v : ℝ) : 
  (0.5 * v + 1 * (2 * v) = 75) → v = 30 := by
  sorry

end initial_speed_problem_l2939_293929


namespace carnation_percentage_l2939_293976

/-- Represents a bouquet of flowers -/
structure Bouquet where
  total : ℝ
  pink : ℝ
  red : ℝ
  pink_roses : ℝ
  pink_carnations : ℝ
  red_roses : ℝ
  red_carnations : ℝ

/-- The theorem stating the percentage of carnations in the bouquet -/
theorem carnation_percentage (b : Bouquet) : 
  b.pink + b.red = b.total →
  b.pink_roses + b.pink_carnations = b.pink →
  b.red_roses + b.red_carnations = b.red →
  b.pink_roses = b.pink / 2 →
  b.red_carnations = b.red * 2 / 3 →
  b.pink = b.total * 7 / 10 →
  (b.pink_carnations + b.red_carnations) / b.total = 11 / 20 := by
sorry

end carnation_percentage_l2939_293976


namespace prob_all_same_color_is_34_455_l2939_293980

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 6
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

def prob_all_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem prob_all_same_color_is_34_455 : prob_all_same_color = 34 / 455 := by
  sorry

end prob_all_same_color_is_34_455_l2939_293980


namespace twenty_percent_greater_than_80_l2939_293956

theorem twenty_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 0.2) → x = 96 := by
  sorry

end twenty_percent_greater_than_80_l2939_293956


namespace min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2939_293957

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_reciprocal_sum_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 
    (1 / a + 1 / b) < 4 + 2 * Real.sqrt 3 + ε := by
  sorry

end min_value_reciprocal_sum_min_value_reciprocal_sum_achieved_l2939_293957


namespace solution_set_implies_a_bound_l2939_293936

theorem solution_set_implies_a_bound (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |x + 1| ≥ a) → a ≤ 3 := by
  sorry

end solution_set_implies_a_bound_l2939_293936


namespace line_equation_equivalence_l2939_293963

/-- Given a line expressed as a dot product of vectors, prove it can be rewritten in slope-intercept form -/
theorem line_equation_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-5)) = 0 ↔ y = 2 * x - 11 := by
  sorry

end line_equation_equivalence_l2939_293963


namespace binomial_coefficient_20_10_l2939_293999

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
  sorry

end binomial_coefficient_20_10_l2939_293999


namespace percent_of_x_l2939_293962

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end percent_of_x_l2939_293962


namespace smallest_four_digit_multiple_of_3_and_5_l2939_293938

theorem smallest_four_digit_multiple_of_3_and_5 : ∃ n : ℕ,
  (n ≥ 1000 ∧ n < 10000) ∧  -- 4-digit number
  n % 3 = 0 ∧               -- multiple of 3
  n % 5 = 0 ∧               -- multiple of 5
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000 ∧ m % 3 = 0 ∧ m % 5 = 0) → n ≤ m) ∧
  n = 1005 :=
by sorry

end smallest_four_digit_multiple_of_3_and_5_l2939_293938


namespace inequality_proof_equality_condition_l2939_293923

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ ∃ (k : ℝ), k > 0 ∧ x = 2*k ∧ y = k ∧ z = k :=
by sorry

end inequality_proof_equality_condition_l2939_293923


namespace reyn_placed_25_pieces_l2939_293965

/-- Represents the puzzle distribution and placement problem --/
structure PuzzleProblem where
  total_pieces : Nat
  num_sons : Nat
  pieces_left : Nat
  rhys_multiplier : Nat
  rory_multiplier : Nat

/-- Calculates the number of pieces Reyn placed --/
def reyn_pieces (p : PuzzleProblem) : Nat :=
  let pieces_per_son := p.total_pieces / p.num_sons
  let total_placed := p.total_pieces - p.pieces_left
  total_placed / (1 + p.rhys_multiplier + p.rory_multiplier)

/-- Theorem stating that Reyn placed 25 pieces --/
theorem reyn_placed_25_pieces : 
  let p : PuzzleProblem := {
    total_pieces := 300,
    num_sons := 3,
    pieces_left := 150,
    rhys_multiplier := 2,
    rory_multiplier := 3
  }
  reyn_pieces p = 25 := by
  sorry

end reyn_placed_25_pieces_l2939_293965


namespace sum_to_k_perfect_square_l2939_293922

theorem sum_to_k_perfect_square (k : ℕ) :
  (∃ n : ℕ, n < 100 ∧ k * (k + 1) / 2 = n^2) → k = 1 ∨ k = 8 ∨ k = 49 := by
  sorry

end sum_to_k_perfect_square_l2939_293922


namespace smallest_special_number_after_3429_l2939_293997

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  n = a * 1000 + b * 100 + c * 10 + d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

theorem smallest_special_number_after_3429 :
  ∀ k : ℕ, k > 3429 ∧ k < 3450 → ¬(uses_four_different_digits k) ∧
  uses_four_different_digits 3450 :=
by sorry

end smallest_special_number_after_3429_l2939_293997


namespace max_x_value_l2939_293973

theorem max_x_value (x : ℝ) : 
  ((5*x - 25)/(4*x - 5))^2 + ((5*x - 25)/(4*x - 5)) = 18 → x ≤ 55/29 := by
  sorry

end max_x_value_l2939_293973


namespace numberOfWaysTheorem_l2939_293942

/-- The number of ways to choose sets S_ij satisfying the given conditions -/
def numberOfWays (n : ℕ) : ℕ :=
  (Nat.factorial (2 * n)) * (2 ^ (n ^ 2))

/-- The theorem stating the number of ways to choose sets S_ij -/
theorem numberOfWaysTheorem (n : ℕ) :
  numberOfWays n = (Nat.factorial (2 * n)) * (2 ^ (n ^ 2)) :=
by sorry

end numberOfWaysTheorem_l2939_293942


namespace parabola_focus_l2939_293955

/-- A parabola is defined by the equation y = 4x^2 -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2

/-- The focus of a parabola y = ax^2 is at (0, 1/(4a)) -/
def is_focus (a : ℝ) (p : ℝ × ℝ) : Prop :=
  p.1 = 0 ∧ p.2 = 1 / (4 * a)

/-- Theorem: The focus of the parabola y = 4x^2 is at (0, 1/16) -/
theorem parabola_focus :
  ∃ (f : ℝ × ℝ), (∀ x y, parabola x y → is_focus 4 f) ∧ f = (0, 1/16) :=
sorry

end parabola_focus_l2939_293955


namespace sevenPointFourSix_eq_fraction_l2939_293993

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℕ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def toRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / (99 : ℚ)

/-- The repeating decimal 7.464646... -/
def sevenPointFourSix : RepeatingDecimal :=
  { integerPart := 7, repeatingPart := 46 }

theorem sevenPointFourSix_eq_fraction :
  toRational sevenPointFourSix = 739 / 99 := by
  sorry

end sevenPointFourSix_eq_fraction_l2939_293993


namespace max_diagonal_length_l2939_293933

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_diagonal_length (PQRS : Quadrilateral) : 
  distance PQRS.P PQRS.Q = 7 →
  distance PQRS.Q PQRS.R = 13 →
  distance PQRS.R PQRS.S = 7 →
  distance PQRS.S PQRS.P = 10 →
  ∃ (pr : ℕ), pr ≤ 19 ∧ 
    distance PQRS.P PQRS.R = pr ∧
    ∀ (x : ℕ), distance PQRS.P PQRS.R = x → x ≤ pr :=
by sorry

end max_diagonal_length_l2939_293933


namespace rectangular_box_surface_area_l2939_293948

/-- Given a rectangular box with dimensions a, b, and c, if the sum of the lengths
    of the twelve edges is 140 and the distance from one corner to the farthest
    corner is 21, then the total surface area of the box is 784. -/
theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 35)
  (diagonal : a^2 + b^2 + c^2 = 441) :
  2 * (a * b + b * c + c * a) = 784 := by
sorry

end rectangular_box_surface_area_l2939_293948


namespace square_difference_l2939_293914

theorem square_difference (m n : ℕ+) 
  (h : (2001 : ℕ) * m ^ 2 + m = (2002 : ℕ) * n ^ 2 + n) : 
  ∃ k : ℕ, m - n = k ^ 2 := by
  sorry

end square_difference_l2939_293914


namespace converse_not_always_true_l2939_293996

theorem converse_not_always_true : 
  ¬ (∀ (a b m : ℝ), a < b → a * m^2 < b * m^2) :=
by sorry

end converse_not_always_true_l2939_293996


namespace polynomial_equality_l2939_293968

/-- Given two polynomials p(x) = 2x^2 + 5x - 2 and q(x) = 2x^2 + 5x + 4,
    prove that the polynomial r(x) = 10x + 6 satisfies p(x) + r(x) = q(x) for all x. -/
theorem polynomial_equality (x : ℝ) :
  (2 * x^2 + 5 * x - 2) + (10 * x + 6) = 2 * x^2 + 5 * x + 4 := by
  sorry

end polynomial_equality_l2939_293968


namespace CaBr2_molecular_weight_l2939_293943

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.904

/-- The number of calcium atoms in CaBr2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of bromine atoms in CaBr2 -/
def num_Br_atoms : ℕ := 2

/-- The molecular weight of CaBr2 in g/mol -/
def molecular_weight_CaBr2 : ℝ :=
  atomic_weight_Ca * num_Ca_atoms + atomic_weight_Br * num_Br_atoms

theorem CaBr2_molecular_weight :
  molecular_weight_CaBr2 = 199.888 := by
  sorry

end CaBr2_molecular_weight_l2939_293943


namespace expression_simplification_l2939_293939

theorem expression_simplification (x y : ℝ) : 
  2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 := by
  sorry

end expression_simplification_l2939_293939


namespace percentage_of_C_grades_l2939_293954

def gradeC (score : ℕ) : Bool :=
  76 ≤ score ∧ score ≤ 85

def scores : List ℕ := [93, 71, 55, 98, 81, 89, 77, 72, 78, 62, 87, 80, 68, 82, 91, 67, 76, 84, 70, 95]

theorem percentage_of_C_grades (scores : List ℕ) : 
  (100 * (scores.filter gradeC).length) / scores.length = 35 :=
sorry

end percentage_of_C_grades_l2939_293954


namespace root_comparison_l2939_293986

noncomputable def f (x : ℝ) : ℝ := Real.log x - 6 + 2 * x

theorem root_comparison (x₀ : ℝ) (hx₀ : f x₀ = 0) :
  Real.log (Real.log x₀) < Real.log (Real.sqrt x₀) ∧
  Real.log (Real.sqrt x₀) < Real.log x₀ ∧
  Real.log x₀ < (Real.log x₀)^2 := by
  sorry

end root_comparison_l2939_293986


namespace imaginary_part_of_complex_fraction_l2939_293906

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (3 + I) / (2 - I) → z.im = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2939_293906


namespace blackboard_numbers_l2939_293960

theorem blackboard_numbers (n : ℕ) (h1 : n = 2004) (h2 : (List.range n).sum % 167 = 0)
  (x : ℕ) (h3 : x ≤ 166) (h4 : (x + 999) % 167 = 0) : x = 3 := by
  sorry

end blackboard_numbers_l2939_293960


namespace function_inequality_l2939_293945

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, 3 * f x - f' x > 0) :
  f 1 < Real.exp 3 * f 0 := by
sorry

end function_inequality_l2939_293945


namespace center_is_five_l2939_293901

/-- Represents a 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if the grid satisfies the consecutive number condition -/
def isConsecutive (g : Grid) : Prop :=
  ∀ p1 p2 : Fin 3 × Fin 3, sharesEdge p1 p2 →
    (g p1.1 p1.2).val + 1 = (g p2.1 p2.2).val ∨
    (g p2.1 p2.2).val + 1 = (g p1.1 p1.2).val

/-- The sum of corner numbers in the grid -/
def cornerSum (g : Grid) : Nat :=
  (g 0 0).val + (g 0 2).val + (g 2 0).val + (g 2 2).val

/-- All numbers from 1 to 9 are used in the grid -/
def usesAllNumbers (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n

theorem center_is_five (g : Grid)
  (h1 : isConsecutive g)
  (h2 : cornerSum g = 20)
  (h3 : usesAllNumbers g) :
  g 1 1 = 5 :=
sorry

end center_is_five_l2939_293901


namespace largest_product_of_three_l2939_293916

def S : Finset Int := {-4, -3, -1, 5, 6}

theorem largest_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
    x ≠ y ∧ y ≠ z ∧ x ≠ z → x * y * z ≤ 72) ∧
  (∃ x y z : Int, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 72) :=
by sorry

end largest_product_of_three_l2939_293916


namespace geometric_series_sum_is_four_thirds_l2939_293920

/-- The sum of the infinite geometric series with first term 1 and common ratio 1/4 -/
def geometric_series_sum : ℚ := 4/3

/-- The first term of the geometric series -/
def a : ℚ := 1

/-- The common ratio of the geometric series -/
def r : ℚ := 1/4

/-- Theorem stating that the sum of the infinite geometric series
    1 + (1/4) + (1/4)² + (1/4)³ + ... is equal to 4/3 -/
theorem geometric_series_sum_is_four_thirds :
  geometric_series_sum = (a / (1 - r)) := by sorry

end geometric_series_sum_is_four_thirds_l2939_293920


namespace inequality_property_equivalence_l2939_293924

theorem inequality_property_equivalence (t : ℝ) (ht : t > 0) :
  (∃ X : Set ℝ, Set.Infinite X ∧
    ∀ (x y z : ℝ) (a : ℝ) (d : ℝ), x ∈ X → y ∈ X → z ∈ X → d > 0 →
      max (|x - (a - d)|) (max (|y - a|) (|z - (a + d)|)) > t * d) ↔
  t < (1 : ℝ) / 2 :=
by sorry

end inequality_property_equivalence_l2939_293924


namespace min_value_quadratic_l2939_293974

theorem min_value_quadratic (y : ℝ) : 
  (5 * y^2 + 5 * y + 4 = 9) → 
  (∀ z : ℝ, 5 * z^2 + 5 * z + 4 = 9 → y ≤ z) → 
  y = (-1 - Real.sqrt 5) / 2 :=
by sorry

end min_value_quadratic_l2939_293974


namespace intersection_A_complement_B_l2939_293994

open Set

def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 6}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 7}

theorem intersection_A_complement_B : A ∩ (𝒰 \ B) = Ioo (-3) 2 ∪ {2} := by sorry

end intersection_A_complement_B_l2939_293994


namespace tricycle_wheels_l2939_293952

theorem tricycle_wheels (num_bicycles num_tricycles bicycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 16)
  (h2 : num_tricycles = 7)
  (h3 : bicycle_wheels = 2)
  (h4 : total_wheels = 53)
  : (total_wheels - num_bicycles * bicycle_wheels) / num_tricycles = 3 := by
  sorry

end tricycle_wheels_l2939_293952


namespace prime_numbers_existence_l2939_293995

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem prime_numbers_existence : 
  ∃ (a : ℕ), 
    a < 10 ∧ 
    is_prime (11*a - 1) ∧ 
    is_prime (10*a + 1) ∧ 
    is_prime (10*a + 7) ∧ 
    a = 4 :=
sorry

end prime_numbers_existence_l2939_293995


namespace earliest_meet_time_proof_l2939_293985

def charlie_lap_time : ℕ := 5
def alex_lap_time : ℕ := 8
def taylor_lap_time : ℕ := 10

def earliest_meet_time : ℕ := 40

theorem earliest_meet_time_proof :
  lcm (lcm charlie_lap_time alex_lap_time) taylor_lap_time = earliest_meet_time :=
by sorry

end earliest_meet_time_proof_l2939_293985


namespace subset_implies_a_equals_three_l2939_293900

theorem subset_implies_a_equals_three (A B : Set ℕ) (a : ℕ) 
  (hA : A = {1, 3}) 
  (hB : B = {1, 2, a}) 
  (hSubset : A ⊆ B) : 
  a = 3 := by
sorry

end subset_implies_a_equals_three_l2939_293900


namespace supermarket_prices_l2939_293998

/-- The price of sugar per kilogram -/
def sugar_price : ℝ := sorry

/-- The price of salt per kilogram -/
def salt_price : ℝ := sorry

/-- The price of rice per kilogram -/
def rice_price : ℝ := sorry

/-- The total price of given quantities of sugar, salt, and rice -/
def total_price (sugar_kg salt_kg rice_kg : ℝ) : ℝ :=
  sugar_kg * sugar_price + salt_kg * salt_price + rice_kg * rice_price

theorem supermarket_prices :
  (total_price 5 3 2 = 28) ∧
  (total_price 4 2 1 = 22) ∧
  (sugar_price = 2 * salt_price) ∧
  (rice_price = 3 * salt_price) →
  total_price 6 4 3 = 36.75 := by
sorry

end supermarket_prices_l2939_293998


namespace equation_solution_l2939_293958

theorem equation_solution (a : ℕ) : 
  (∃ x y : ℕ, (x + y)^2 + 3*x + y = 2*a) ↔ a = 4 :=
by sorry

end equation_solution_l2939_293958


namespace cylinder_volume_increase_l2939_293932

/-- Given a cylindrical tank with radius 5 inches and height 6 inches, 
    this theorem proves that increasing the radius by x inches 
    results in the same volume increase as increasing the height by 2x inches 
    when x = 10/3. -/
theorem cylinder_volume_increase (x : ℝ) : x = 10/3 ↔ 
  π * (5 + x)^2 * 6 = π * 5^2 * (6 + 2*x) := by
  sorry

#check cylinder_volume_increase

end cylinder_volume_increase_l2939_293932


namespace candy_distribution_l2939_293930

/-- Given the number of candies for each type and the number of cousins,
    calculates the number of candies left after equal distribution. -/
def candies_left (apple orange lemon grape cousins : ℕ) : ℕ :=
  (apple + orange + lemon + grape) % cousins

theorem candy_distribution (apple orange lemon grape cousins : ℕ) 
    (h : cousins > 0) : 
  candies_left apple orange lemon grape cousins = 
  (apple + orange + lemon + grape) % cousins := by
  sorry

end candy_distribution_l2939_293930


namespace power_four_2024_mod_11_l2939_293944

theorem power_four_2024_mod_11 : 4^2024 % 11 = 3 := by
  sorry

end power_four_2024_mod_11_l2939_293944


namespace pauls_crayons_l2939_293941

/-- Paul's crayon problem -/
theorem pauls_crayons (initial given lost broken traded : ℕ) : 
  initial = 250 → 
  given = 150 → 
  lost = 512 → 
  broken = 75 → 
  traded = 35 → 
  lost - (given + broken + traded) = 252 := by
  sorry

end pauls_crayons_l2939_293941


namespace sixty_has_twelve_divisors_l2939_293928

/-- The number of positive divisors of 60 -/
def num_divisors_60 : ℕ := Finset.card (Nat.divisors 60)

/-- Theorem stating that 60 has exactly 12 positive divisors -/
theorem sixty_has_twelve_divisors : num_divisors_60 = 12 := by
  sorry

end sixty_has_twelve_divisors_l2939_293928


namespace tiffany_cans_l2939_293913

theorem tiffany_cans (bags_monday : ℕ) : bags_monday = 12 :=
  by
    have h1 : bags_monday + 12 = 2 * bags_monday := by sorry
    -- The number of bags on Tuesday (bags_monday + 12) is double the number of bags on Monday (2 * bags_monday)
    sorry

end tiffany_cans_l2939_293913


namespace ramesh_refrigerator_price_l2939_293991

/-- The price Ramesh paid for a refrigerator given specific conditions -/
theorem ramesh_refrigerator_price (P : ℝ) 
  (h1 : 1.1 * P = 17600)  -- Selling price for 10% profit without discount
  (h2 : 0.2 * P = P - 0.8 * P)  -- 20% discount on labelled price
  (h3 : 125 = 125)  -- Transport cost
  (h4 : 250 = 250)  -- Installation cost
  : 0.8 * P + 125 + 250 = 13175 := by
  sorry

end ramesh_refrigerator_price_l2939_293991


namespace inverse_of_two_mod_185_l2939_293987

theorem inverse_of_two_mod_185 : Int.ModEq 1 185 (2 * 93) := by sorry

end inverse_of_two_mod_185_l2939_293987


namespace calculate_expression_l2939_293915

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 := by
  sorry

end calculate_expression_l2939_293915


namespace a_18_value_l2939_293966

/-- An equal sum sequence with common sum c -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = c

theorem a_18_value (a : ℕ → ℝ) (h : EqualSumSequence a 5) (h1 : a 1 = 2) :
  a 18 = 3 := by
  sorry

end a_18_value_l2939_293966


namespace solve_equation_l2939_293904

theorem solve_equation : ∃ y : ℝ, 4 * y + 6 * y = 450 - 10 * (y - 5) ∧ y = 25 := by
  sorry

end solve_equation_l2939_293904


namespace irreducible_fraction_l2939_293977

theorem irreducible_fraction (n : ℤ) : Int.gcd (39*n + 4) (26*n + 3) = 1 := by sorry

end irreducible_fraction_l2939_293977


namespace geese_ratio_l2939_293940

/-- Represents the number of ducks and geese bought by a person -/
structure DucksAndGeese where
  ducks : ℕ
  geese : ℕ

/-- The problem setup -/
def market_problem (lily rayden : DucksAndGeese) : Prop :=
  rayden.ducks = 3 * lily.ducks ∧
  lily.ducks = 20 ∧
  lily.geese = 10 ∧
  rayden.ducks + rayden.geese = lily.ducks + lily.geese + 70

/-- The theorem to prove -/
theorem geese_ratio (lily rayden : DucksAndGeese) 
  (h : market_problem lily rayden) : 
  rayden.geese = 4 * lily.geese := by
  sorry


end geese_ratio_l2939_293940


namespace spiral_staircase_handrail_length_l2939_293927

/-- The length of a spiral staircase handrail -/
theorem spiral_staircase_handrail_length 
  (turn_angle : Real) 
  (rise : Real) 
  (radius : Real) 
  (handrail_length : Real) : 
  turn_angle = 315 ∧ 
  rise = 12 ∧ 
  radius = 4 → 
  abs (handrail_length - Real.sqrt (rise^2 + (turn_angle / 360 * 2 * Real.pi * radius)^2)) < 0.1 :=
by sorry

end spiral_staircase_handrail_length_l2939_293927


namespace equation_represents_intersecting_lines_l2939_293981

theorem equation_represents_intersecting_lines (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 3*x*y ↔ x*y = 0 :=
by sorry

end equation_represents_intersecting_lines_l2939_293981


namespace lee_proposal_time_l2939_293918

/-- Calculates the number of months needed to save for an engagement ring based on annual salary and monthly savings. -/
def months_to_save_for_ring (annual_salary : ℕ) (monthly_savings : ℕ) : ℕ :=
  let monthly_salary := annual_salary / 12
  let ring_cost := 2 * monthly_salary
  ring_cost / monthly_savings

/-- Proves that given the specified conditions, it takes 10 months to save for the ring. -/
theorem lee_proposal_time : months_to_save_for_ring 60000 1000 = 10 := by
  sorry

end lee_proposal_time_l2939_293918


namespace traces_bag_weight_is_two_l2939_293950

/-- The weight of one of Trace's shopping bags -/
def traces_bag_weight (
  trace_bag_count : ℕ
  ) (
  gordon_bag1_weight : ℕ
  ) (
  gordon_bag2_weight : ℕ
  ) : ℚ :=
  (gordon_bag1_weight + gordon_bag2_weight) / trace_bag_count

/-- Theorem stating the weight of one of Trace's shopping bags -/
theorem traces_bag_weight_is_two :
  traces_bag_weight 5 3 7 = 2 := by
  sorry

#eval traces_bag_weight 5 3 7

end traces_bag_weight_is_two_l2939_293950


namespace demand_exceeds_15000_only_in_7_and_8_l2939_293979

def S (n : ℕ) : ℚ := (n : ℚ) / 90 * (21 * n - n^2 - 5)

def a (n : ℕ) : ℚ := S n - S (n-1)

theorem demand_exceeds_15000_only_in_7_and_8 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 →
    (a n > (3/2) ↔ n = 7 ∨ n = 8) :=
sorry

end demand_exceeds_15000_only_in_7_and_8_l2939_293979


namespace tv_production_last_period_avg_l2939_293970

/-- Represents the production of TVs in a factory over a month -/
structure TVProduction where
  totalDays : Nat
  firstPeriodDays : Nat
  firstPeriodAvg : Nat
  monthlyAvg : Nat

/-- Calculates the average production for the last period of the month -/
def lastPeriodAvg (p : TVProduction) : Rat :=
  let lastPeriodDays := p.totalDays - p.firstPeriodDays
  let totalProduction := p.totalDays * p.monthlyAvg
  let firstPeriodProduction := p.firstPeriodDays * p.firstPeriodAvg
  (totalProduction - firstPeriodProduction) / lastPeriodDays

/-- Theorem stating that given the conditions, the average production for the last 5 days is 20 TVs per day -/
theorem tv_production_last_period_avg 
  (p : TVProduction) 
  (h1 : p.totalDays = 30) 
  (h2 : p.firstPeriodDays = 25) 
  (h3 : p.firstPeriodAvg = 50) 
  (h4 : p.monthlyAvg = 45) : 
  lastPeriodAvg p = 20 := by
  sorry

end tv_production_last_period_avg_l2939_293970


namespace least_time_six_horses_meet_l2939_293969

def horse_lap_time (k : ℕ) : ℕ := k + 1

def is_at_start (t : ℕ) (k : ℕ) : Prop :=
  t % (horse_lap_time k) = 0

def at_least_six_at_start (t : ℕ) : Prop :=
  ∃ (s : Finset ℕ), s.card ≥ 6 ∧ s ⊆ Finset.range 8 ∧ ∀ k ∈ s, is_at_start t k

theorem least_time_six_horses_meet :
  ∃ (T : ℕ), T > 0 ∧ at_least_six_at_start T ∧
  ∀ (t : ℕ), t > 0 ∧ t < T → ¬(at_least_six_at_start t) ∧
  T = 420 :=
sorry

end least_time_six_horses_meet_l2939_293969


namespace enrollment_increase_l2939_293978

/-- Theorem: Enrollment Increase Calculation

Given:
- Enrollment at the beginning of 1992 was 20% greater than at the beginning of 1991
- Enrollment at the beginning of 1993 was 26% greater than at the beginning of 1991

Prove:
The percent increase in enrollment from the beginning of 1992 to the beginning of 1993 is 5%
-/
theorem enrollment_increase (e : ℝ) : 
  let e_1992 := 1.20 * e
  let e_1993 := 1.26 * e
  (e_1993 - e_1992) / e_1992 * 100 = 5 := by
  sorry

end enrollment_increase_l2939_293978


namespace walking_problem_l2939_293953

/-- Represents the ratio of steps taken by the good walker to the bad walker in the same time -/
def step_ratio : ℚ := 100 / 60

/-- Represents the head start of the bad walker in steps -/
def head_start : ℕ := 100

/-- Represents the walking problem described in "Nine Chapters on the Mathematical Art" -/
theorem walking_problem (x y : ℚ) :
  (x - y = head_start) ∧ (x = step_ratio * y) ↔
  x - y = head_start ∧ x = (100 : ℚ) / 60 * y :=
sorry

end walking_problem_l2939_293953


namespace cube_plus_reciprocal_cube_l2939_293983

theorem cube_plus_reciprocal_cube (x : ℝ) (h1 : x > 0) (h2 : (x + 1/x)^2 = 25) :
  x^3 + 1/x^3 = 110 := by
  sorry

end cube_plus_reciprocal_cube_l2939_293983


namespace hyperbola_asymptote_l2939_293911

/-- The equation of the asymptotes of a hyperbola -/
def asymptote_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y = (b / a) * x ∨ y = -(b / a) * x}

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - y^2 / 3 = 1

theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y →
  (x, y) ∈ asymptote_equation 1 (Real.sqrt 3) :=
sorry

end hyperbola_asymptote_l2939_293911


namespace negative_one_is_root_l2939_293937

/-- The polynomial f(x) = x^3 + x^2 - 6x - 6 -/
def f (x : ℝ) : ℝ := x^3 + x^2 - 6*x - 6

/-- Theorem: -1 is a root of the polynomial f(x) = x^3 + x^2 - 6x - 6 -/
theorem negative_one_is_root : f (-1) = 0 := by
  sorry

end negative_one_is_root_l2939_293937


namespace product_divisible_by_four_probability_l2939_293934

/-- The set of integers from 6 to 18, inclusive -/
def IntegerRange : Set ℤ := {n : ℤ | 6 ≤ n ∧ n ≤ 18}

/-- The set of integers in IntegerRange that are divisible by 4 -/
def DivisibleBy4 : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 4 = 0}

/-- The set of even integers in IntegerRange -/
def EvenInRange : Set ℤ := {n : ℤ | n ∈ IntegerRange ∧ n % 2 = 0}

/-- The number of ways to choose 2 distinct integers from IntegerRange -/
def TotalChoices : ℕ := Nat.choose (Finset.card (Finset.range 13)) 2

/-- The number of ways to choose 2 distinct integers from IntegerRange 
    such that their product is divisible by 4 -/
def FavorableChoices : ℕ := 33

theorem product_divisible_by_four_probability : 
  (FavorableChoices : ℚ) / TotalChoices = 33 / 78 := by sorry

end product_divisible_by_four_probability_l2939_293934


namespace sum_of_segment_lengths_divisible_by_four_l2939_293947

/-- Represents a square sheet of graph paper -/
structure GraphPaper where
  sideLength : ℕ

/-- The sum of lengths of all segments in the graph paper -/
def sumOfSegmentLengths (paper : GraphPaper) : ℕ :=
  2 * paper.sideLength * (paper.sideLength + 1)

/-- Theorem stating that the sum of segment lengths is divisible by 4 -/
theorem sum_of_segment_lengths_divisible_by_four (paper : GraphPaper) :
  4 ∣ sumOfSegmentLengths paper := by
  sorry

end sum_of_segment_lengths_divisible_by_four_l2939_293947


namespace emilee_earnings_l2939_293984

/-- Proves that Emilee earns $25 given the conditions of the problem -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  total = terrence_earnings + (terrence_earnings + jermaine_extra) + (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) →
  (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) = 25 := by
  sorry

#check emilee_earnings

end emilee_earnings_l2939_293984


namespace pqr_value_l2939_293907

theorem pqr_value (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 390 / (p * q * r) = 1) :
  p * q * r = 1680 := by
sorry

end pqr_value_l2939_293907


namespace jill_weekly_earnings_l2939_293959

/-- Calculates Jill's earnings as a waitress for a week --/
def jill_earnings (hourly_wage : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) 
                  (shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ) : ℝ :=
  let total_hours := shifts * hours_per_shift
  let wage_earnings := hourly_wage * total_hours
  let total_orders := avg_orders_per_hour * total_hours
  let orders_with_tax := total_orders * (1 + sales_tax)
  let tip_earnings := orders_with_tax * tip_rate
  wage_earnings + tip_earnings

/-- Theorem stating Jill's earnings for the week --/
theorem jill_weekly_earnings : 
  jill_earnings 4 0.15 0.1 3 8 40 = 254.4 := by
  sorry

end jill_weekly_earnings_l2939_293959


namespace rhinoceros_preserve_watering_area_l2939_293964

theorem rhinoceros_preserve_watering_area 
  (initial_population : ℕ)
  (grazing_area_per_rhino : ℕ)
  (population_increase_percent : ℚ)
  (total_preserve_area : ℕ) :
  initial_population = 8000 →
  grazing_area_per_rhino = 100 →
  population_increase_percent = 1/10 →
  total_preserve_area = 890000 →
  let increased_population := initial_population + (initial_population * population_increase_percent).floor
  let total_grazing_area := increased_population * grazing_area_per_rhino
  let watering_area := total_preserve_area - total_grazing_area
  watering_area = 10000 := by
sorry

end rhinoceros_preserve_watering_area_l2939_293964


namespace exists_vertex_with_positive_product_l2939_293971

-- Define a polyhedron type
structure Polyhedron where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  marks : (Nat × Nat) → Int
  vertex_count : vertices.card = 101
  edge_marks : ∀ e ∈ edges, marks e = 1 ∨ marks e = -1

-- Define the product of marks at a vertex
def product_at_vertex (p : Polyhedron) (v : Nat) : Int :=
  (p.edges.filter (λ e => e.1 = v ∨ e.2 = v)).prod p.marks

-- Theorem statement
theorem exists_vertex_with_positive_product (p : Polyhedron) :
  ∃ v ∈ p.vertices, product_at_vertex p v = 1 := by
  sorry

end exists_vertex_with_positive_product_l2939_293971


namespace opposite_number_problem_l2939_293951

theorem opposite_number_problem (x : ℤ) : (x + 1 = -(-10)) → x = 9 := by
  sorry

end opposite_number_problem_l2939_293951


namespace remaining_credit_after_call_prove_remaining_credit_l2939_293905

/-- Calculates the remaining credit on a prepaid phone card after a call. -/
theorem remaining_credit_after_call 
  (initial_value : ℝ) 
  (cost_per_minute : ℝ) 
  (call_duration : ℕ) 
  (remaining_credit : ℝ) : Prop :=
  initial_value = 30 ∧ 
  cost_per_minute = 0.16 ∧ 
  call_duration = 22 ∧ 
  remaining_credit = initial_value - (cost_per_minute * call_duration) → 
  remaining_credit = 26.48

/-- Proof of the remaining credit calculation. -/
theorem prove_remaining_credit : 
  ∃ (initial_value cost_per_minute : ℝ) (call_duration : ℕ) (remaining_credit : ℝ),
    remaining_credit_after_call initial_value cost_per_minute call_duration remaining_credit :=
by
  sorry

end remaining_credit_after_call_prove_remaining_credit_l2939_293905


namespace special_function_is_negation_l2939_293909

/-- A function satisfying the given functional equation -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x - y) = f x + f (f y - f (-x)) + x

/-- The main theorem: if f satisfies the functional equation, then f(x) = -x for all x -/
theorem special_function_is_negation (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f x = -x :=
sorry

end special_function_is_negation_l2939_293909


namespace negation_of_existence_negation_of_quadratic_inequality_l2939_293975

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l2939_293975


namespace concentric_circles_radius_l2939_293919

theorem concentric_circles_radius (r₁ r₂ AB : ℝ) : 
  r₁ > 0 → r₂ > 0 →
  r₂ / r₁ = 7 / 3 →
  AB = 20 →
  ∃ (AC BC : ℝ), 
    AC = 2 * r₂ ∧
    BC^2 + AB^2 = AC^2 ∧
    BC^2 = r₂^2 - r₁^2 →
  r₂ = 70 / 3 := by
sorry

end concentric_circles_radius_l2939_293919


namespace binomial_expansion_coefficient_l2939_293967

theorem binomial_expansion_coefficient (a : ℝ) (h : a ≠ 0) :
  let expansion := fun (x : ℝ) ↦ (x - a / x)^6
  let B := expansion 1  -- Constant term when x = 1
  B = 44 → a = -22/5 := by
sorry

end binomial_expansion_coefficient_l2939_293967


namespace cube_root_of_product_l2939_293990

theorem cube_root_of_product (x y z : ℕ) : 
  (5^9 * 7^6 * 13^3 : ℝ)^(1/3) = 79625 := by
  sorry

end cube_root_of_product_l2939_293990


namespace water_tank_problem_l2939_293917

/-- Calculates the water volume in a tank after a given number of hours, 
    with specified initial volume, loss rate, and water additions. -/
def water_volume (initial_volume : ℝ) (loss_rate : ℝ) (additions : List ℝ) : ℝ :=
  initial_volume - loss_rate * additions.length + additions.sum

/-- The water volume problem -/
theorem water_tank_problem : 
  let initial_volume : ℝ := 40
  let loss_rate : ℝ := 2
  let additions : List ℝ := [0, 0, 1, 3]
  water_volume initial_volume loss_rate additions = 36 := by
  sorry

end water_tank_problem_l2939_293917


namespace diamond_value_l2939_293908

/-- Given that ◇5₉ = ◇3₁₁ where ◇ represents a digit, prove that ◇ = 1 -/
theorem diamond_value : ∃ (d : ℕ), d < 10 ∧ d * 9 + 5 = d * 11 + 3 ∧ d = 1 := by sorry

end diamond_value_l2939_293908


namespace symmetry_axis_implies_p_plus_s_zero_l2939_293902

/-- Given a curve y = (px + q)/(rx + s) with y = 2x as its axis of symmetry,
    where p, q, r, s are nonzero real numbers, prove that p + s = 0. -/
theorem symmetry_axis_implies_p_plus_s_zero
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_symmetry : ∀ (x y : ℝ), y = (p * x + q) / (r * x + s) → 2 * x = (p * (2 * y) + q) / (r * (2 * y) + s)) :
  p + s = 0 := by
  sorry

end symmetry_axis_implies_p_plus_s_zero_l2939_293902


namespace coin_machine_theorem_l2939_293903

/-- Represents the coin-changing machine's rules --/
structure CoinMachine where
  quarter_to_nickels : ℕ → ℕ
  nickel_to_pennies : ℕ → ℕ
  penny_to_quarters : ℕ → ℕ

/-- Represents the possible amounts in cents --/
def possible_amounts (m : CoinMachine) (n : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, x = 1 + 74 * k}

/-- The set of given options in cents --/
def given_options : Set ℕ := {175, 325, 449, 549, 823}

theorem coin_machine_theorem (m : CoinMachine) 
  (h1 : m.quarter_to_nickels 1 = 5)
  (h2 : m.nickel_to_pennies 1 = 5)
  (h3 : m.penny_to_quarters 1 = 3) :
  given_options ∩ (possible_amounts m 1) = {823} := by
  sorry

end coin_machine_theorem_l2939_293903


namespace problem_statement_l2939_293912

theorem problem_statement (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end problem_statement_l2939_293912


namespace negative_root_implies_a_less_than_negative_three_l2939_293972

theorem negative_root_implies_a_less_than_negative_three (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (a - 3)) → a < -3 := by
  sorry

end negative_root_implies_a_less_than_negative_three_l2939_293972
