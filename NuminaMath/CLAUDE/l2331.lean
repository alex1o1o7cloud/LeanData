import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2331_233155

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 + a 5 = 105)
  (h_sum2 : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l2331_233155


namespace NUMINAMATH_CALUDE_spool_length_problem_l2331_233111

/-- Calculates the length of each spool of wire -/
def spool_length (total_spools : ℕ) (wire_per_necklace : ℕ) (total_necklaces : ℕ) : ℕ :=
  (wire_per_necklace * total_necklaces) / total_spools

theorem spool_length_problem :
  let total_spools : ℕ := 3
  let wire_per_necklace : ℕ := 4
  let total_necklaces : ℕ := 15
  spool_length total_spools wire_per_necklace total_necklaces = 20 := by
  sorry

end NUMINAMATH_CALUDE_spool_length_problem_l2331_233111


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_12_l2331_233137

theorem cos_alpha_plus_pi_12 (α : Real) (h : Real.tan (α + π/3) = -2) :
  Real.cos (α + π/12) = Real.sqrt 10 / 10 ∨ Real.cos (α + π/12) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_12_l2331_233137


namespace NUMINAMATH_CALUDE_candy_boxes_problem_l2331_233147

/-- Given that Paul bought 6 boxes of chocolate candy and 4 boxes of caramel candy,
    with a total of 90 candies, and each box contains the same number of pieces,
    prove that there are 9 pieces of candy in each box. -/
theorem candy_boxes_problem (pieces_per_box : ℕ) : 
  (6 * pieces_per_box + 4 * pieces_per_box = 90) → pieces_per_box = 9 := by
sorry

end NUMINAMATH_CALUDE_candy_boxes_problem_l2331_233147


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2331_233100

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 6*x^10 + 3*x^9) =
  18*x^13 - 3*x^12 + 12*x^11 - 3*x^10 - 6*x^9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2331_233100


namespace NUMINAMATH_CALUDE_vector_at_minus_2_l2331_233195

/-- A line in a plane parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector at t = 5 is (0, 5) -/
axiom vector_at_5 : line 5 = (0, 5)

/-- The vector at t = 8 is (9, 1) -/
axiom vector_at_8 : line 8 = (9, 1)

/-- The theorem to prove -/
theorem vector_at_minus_2 : line (-2) = (21, -23/3) := by sorry

end NUMINAMATH_CALUDE_vector_at_minus_2_l2331_233195


namespace NUMINAMATH_CALUDE_sqrt_360000_equals_600_l2331_233169

theorem sqrt_360000_equals_600 : Real.sqrt 360000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360000_equals_600_l2331_233169


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2331_233101

/-- An arithmetic sequence with given second and fifth terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  second_term : a 2 = 2
  fifth_term : a 5 = 5

/-- The general term of the arithmetic sequence is n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2331_233101


namespace NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l2331_233124

-- Equation 1
theorem solutions_eq1 : 
  ∀ x : ℝ, x^2 - 6*x - 7 = 0 ↔ x = 7 ∨ x = -1 := by sorry

-- Equation 2
theorem solutions_eq2 : 
  ∀ x : ℝ, 3*x^2 - 1 = 2*x ↔ x = 1 ∨ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l2331_233124


namespace NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l2331_233107

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 18500000 = a * (10 : ℝ) ^ n ∧ a = 1.85 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l2331_233107


namespace NUMINAMATH_CALUDE_panda_survival_probability_l2331_233117

theorem panda_survival_probability (p_10 p_15 : ℝ) 
  (h1 : p_10 = 0.8) 
  (h2 : p_15 = 0.6) : 
  p_15 / p_10 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_panda_survival_probability_l2331_233117


namespace NUMINAMATH_CALUDE_trajectory_is_hyperbola_l2331_233118

-- Define the complex plane
def ComplexPlane := ℂ

-- Define the condition for the trajectory
def TrajectoryCondition (z : ℂ) : Prop :=
  Complex.abs (Complex.abs (z - 1) - Complex.abs (z + Complex.I)) = 1

-- Define a hyperbola in the complex plane
def IsHyperbola (S : Set ℂ) : Prop :=
  ∃ (F₁ F₂ : ℂ) (a : ℝ), a > 0 ∧ Complex.abs (F₁ - F₂) > 2 * a ∧
    S = {z : ℂ | Complex.abs (Complex.abs (z - F₁) - Complex.abs (z - F₂)) = 2 * a}

-- Theorem statement
theorem trajectory_is_hyperbola :
  IsHyperbola {z : ℂ | TrajectoryCondition z} :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_hyperbola_l2331_233118


namespace NUMINAMATH_CALUDE_cards_distribution_l2331_233185

/-- 
Given 52 cards dealt to 8 people as evenly as possible, 
this theorem proves that 4 people will have fewer than 7 cards.
-/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 52)
  (h2 : num_people = 8) :
  (num_people - (total_cards % num_people)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2331_233185


namespace NUMINAMATH_CALUDE_cosine_intersection_theorem_l2331_233152

theorem cosine_intersection_theorem (f : ℝ → ℝ) (θ : ℝ) : 
  (∀ x ≥ 0, f x = |Real.cos x|) →
  (∃ l : ℝ → ℝ, l 0 = 0 ∧ (∃ a b c d : ℝ, 0 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d = θ ∧
    f a = l a ∧ f b = l b ∧ f c = l c ∧ f d = l d ∧
    ∀ x : ℝ, x ≥ 0 → x ≠ a → x ≠ b → x ≠ c → x ≠ d → f x ≠ l x)) →
  ((1 + θ^2) * Real.sin (2*θ)) / θ = -2 := by
sorry

end NUMINAMATH_CALUDE_cosine_intersection_theorem_l2331_233152


namespace NUMINAMATH_CALUDE_train_distance_l2331_233163

/-- The distance covered by a train traveling at a constant speed for a given time. -/
theorem train_distance (speed : ℝ) (time : ℝ) (h1 : speed = 150) (h2 : time = 8) :
  speed * time = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2331_233163


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2331_233143

/-- Represents the number of tokens Alex has -/
structure Tokens where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules for the booths -/
structure ExchangeRule where
  redCost : ℕ
  blueCost : ℕ
  redGain : ℕ
  blueGain : ℕ
  silverGain : ℕ

/-- Defines if an exchange is possible given the current tokens and an exchange rule -/
def canExchange (t : Tokens) (r : ExchangeRule) : Prop :=
  t.red ≥ r.redCost ∧ t.blue ≥ r.blueCost

/-- Applies an exchange rule to the current tokens -/
def applyExchange (t : Tokens) (r : ExchangeRule) : Tokens :=
  { red := t.red - r.redCost + r.redGain,
    blue := t.blue - r.blueCost + r.blueGain,
    silver := t.silver + r.silverGain }

/-- Theorem: The maximum number of silver tokens Alex can obtain is 23 -/
theorem max_silver_tokens :
  ∀ (initial : Tokens)
    (rule1 rule2 : ExchangeRule),
  initial.red = 60 ∧ initial.blue = 90 ∧ initial.silver = 0 →
  rule1 = { redCost := 3, blueCost := 0, redGain := 0, blueGain := 2, silverGain := 1 } →
  rule2 = { redCost := 0, blueCost := 4, redGain := 1, blueGain := 0, silverGain := 1 } →
  ∃ (final : Tokens),
    (∀ t, (canExchange t rule1 ∨ canExchange t rule2) → t.silver ≤ final.silver) ∧
    final.silver = 23 :=
by sorry

end NUMINAMATH_CALUDE_max_silver_tokens_l2331_233143


namespace NUMINAMATH_CALUDE_chessboard_inner_square_probability_l2331_233166

/-- Represents a square chessboard -/
structure Chessboard where
  size : ℕ

/-- Calculates the number of squares on the perimeter of the chessboard -/
def perimeterSquares (board : Chessboard) : ℕ :=
  4 * board.size - 4

/-- Calculates the number of squares not on the perimeter of the chessboard -/
def innerSquares (board : Chessboard) : ℕ :=
  board.size * board.size - perimeterSquares board

/-- The probability of choosing an inner square on the chessboard -/
def innerSquareProbability (board : Chessboard) : ℚ :=
  innerSquares board / (board.size * board.size)

theorem chessboard_inner_square_probability :
  let board := Chessboard.mk 10
  innerSquareProbability board = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_inner_square_probability_l2331_233166


namespace NUMINAMATH_CALUDE_two_in_A_l2331_233179

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem two_in_A : 2 ∈ A := by sorry

end NUMINAMATH_CALUDE_two_in_A_l2331_233179


namespace NUMINAMATH_CALUDE_range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l2331_233158

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, x^2 - m ≤ 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧
  ∀ x y : ℝ, x^2 / m^2 + y^2 / 4 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Theorem 1
theorem range_m_when_not_p_false (m : ℝ) :
  ¬(¬(p m)) → m ≥ 1 := by sorry

-- Theorem 2
theorem range_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  m ∈ Set.Ioi (-2) ∪ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l2331_233158


namespace NUMINAMATH_CALUDE_museum_wings_l2331_233121

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  painting_wings : ℕ
  artifact_wings : ℕ
  large_paintings : ℕ
  small_paintings : ℕ
  artifacts_per_wing : ℕ

/-- Calculates the total number of paintings in the museum -/
def total_paintings (m : Museum) : ℕ :=
  m.large_paintings + m.small_paintings

/-- Calculates the total number of artifacts in the museum -/
def total_artifacts (m : Museum) : ℕ :=
  m.artifact_wings * m.artifacts_per_wing

/-- Theorem stating the total number of wings in the museum -/
theorem museum_wings (m : Museum) 
  (h1 : m.painting_wings = 3)
  (h2 : m.large_paintings = 1)
  (h3 : m.small_paintings = 24)
  (h4 : m.artifacts_per_wing = 20)
  (h5 : total_artifacts m = 4 * total_paintings m) :
  m.painting_wings + m.artifact_wings = 8 := by
  sorry

#check museum_wings

end NUMINAMATH_CALUDE_museum_wings_l2331_233121


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2331_233140

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 3003 → 
  A + B + C ≤ 105 ∧ (∃ (P Q R : ℕ+), P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ P * Q * R = 3003 ∧ P + Q + R = 105) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2331_233140


namespace NUMINAMATH_CALUDE_intersection_distance_implies_a_bound_l2331_233132

/-- Given a line and a circle with parameter a, if the distance between
    their intersection points is at least 2√3, then a ≤ -4/3 -/
theorem intersection_distance_implies_a_bound
  (a : ℝ)
  (line : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (M N : ℝ × ℝ)
  (h_line : ∀ x y, line x y ↔ a * x - y + 3 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - a)^2 = 4)
  (h_intersection : line M.1 M.2 ∧ circle M.1 M.2 ∧ line N.1 N.2 ∧ circle N.1 N.2)
  (h_distance : (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) :
  a ≤ -4/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_implies_a_bound_l2331_233132


namespace NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l2331_233198

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s - 18 ≤ 94 := by sorry

theorem quadratic_maximum_achieved : ∃ s : ℝ, -7 * s^2 + 56 * s - 18 = 94 := by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_quadratic_maximum_achieved_l2331_233198


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l2331_233170

open Real

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (π / 2), StrictMono (fun x => (sin x + a) / cos x)) →
  a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l2331_233170


namespace NUMINAMATH_CALUDE_planted_fraction_for_specific_field_l2331_233134

/-- Represents a right triangle field with an unplanted square at the right angle -/
structure TriangleField where
  /-- Length of the first leg of the triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the triangle -/
  leg2 : ℝ
  /-- Side length of the unplanted square -/
  square_side : ℝ
  /-- Shortest distance from the square to the hypotenuse -/
  distance_to_hypotenuse : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (field : TriangleField) : ℝ :=
  sorry

/-- Theorem stating the planted fraction for the specific field described in the problem -/
theorem planted_fraction_for_specific_field :
  let field : TriangleField := {
    leg1 := 5,
    leg2 := 12,
    square_side := 60 / 49,
    distance_to_hypotenuse := 3
  }
  planted_fraction field = 11405 / 12005 := by
  sorry

end NUMINAMATH_CALUDE_planted_fraction_for_specific_field_l2331_233134


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_73_l2331_233165

def p₁ (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1
def p₂ (x : ℝ) : ℝ := 2 * x^2 + x + 4
def p₃ (x : ℝ) : ℝ := x^2 + 2 * x + 3

def product (x : ℝ) : ℝ := p₁ x * p₂ x * p₃ x

theorem coefficient_x_cubed_is_73 :
  ∃ (a b c d : ℝ), product = fun x ↦ 73 * x^3 + a * x^4 + b * x^2 + c * x + d :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_73_l2331_233165


namespace NUMINAMATH_CALUDE_linear_function_decreasing_implies_negative_slope_l2331_233151

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Property that y decreases as x increases -/
def decreasing (f : LinearFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f.m * x₁ + f.b > f.m * x₂ + f.b

theorem linear_function_decreasing_implies_negative_slope (f : LinearFunction) 
    (h : f.b = 5) (dec : decreasing f) : f.m < 0 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_implies_negative_slope_l2331_233151


namespace NUMINAMATH_CALUDE_number_problem_l2331_233125

theorem number_problem (x : ℝ) : 
  (0.3 * x = 0.6 * 150 + 120) → x = 700 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2331_233125


namespace NUMINAMATH_CALUDE_factor_tree_product_l2331_233106

theorem factor_tree_product : ∀ (X F G H : ℕ),
  X = F * G →
  F = 11 * 7 →
  G = 7 * H →
  H = 17 * 2 →
  X = 57556 := by
sorry

end NUMINAMATH_CALUDE_factor_tree_product_l2331_233106


namespace NUMINAMATH_CALUDE_circle_diameter_and_circumference_l2331_233115

theorem circle_diameter_and_circumference (A : ℝ) (h : A = 16 * Real.pi) :
  ∃ (d c : ℝ), d = 8 ∧ c = 8 * Real.pi ∧ A = Real.pi * (d / 2)^2 ∧ c = Real.pi * d := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_and_circumference_l2331_233115


namespace NUMINAMATH_CALUDE_mixed_fraction_power_product_l2331_233191

theorem mixed_fraction_power_product (n : ℕ) (m : ℕ) :
  (-(3 : ℚ) / 2) ^ (2021 : ℕ) * (2 : ℚ) / 3 ^ (2023 : ℕ) = -(4 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_power_product_l2331_233191


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2331_233164

/-- The line equation passing through a fixed point for any real k -/
def line_equation (k x y : ℝ) : ℝ := (2*k - 1)*x - (k + 3)*y - (k - 11)

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2331_233164


namespace NUMINAMATH_CALUDE_range_of_m_l2331_233150

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : ∀ (x y : ℝ), x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) : 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2331_233150


namespace NUMINAMATH_CALUDE_solution_difference_l2331_233194

theorem solution_difference (x₀ y₀ : ℝ) : 
  (x₀^3 - 2023*x₀ = y₀^3 - 2023*y₀ + 2020) →
  (x₀^2 + x₀*y₀ + y₀^2 = 2022) →
  (x₀ - y₀ = -2020) := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2331_233194


namespace NUMINAMATH_CALUDE_golden_section_proportion_l2331_233188

/-- Golden section point of a line segment -/
def is_golden_section_point (A B C : ℝ) : Prop :=
  (B - A) / (C - A) = (C - A) / (B - C)

theorem golden_section_proportion (A B C : ℝ) 
  (h1 : is_golden_section_point A B C) 
  (h2 : C - A > B - C) : 
  (B - A) / (C - A) = (C - A) / (B - C) := by
  sorry

end NUMINAMATH_CALUDE_golden_section_proportion_l2331_233188


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2331_233172

theorem binomial_expansion_constant_term (x : ℝ) (n : ℕ) :
  (∀ k : ℕ, k ≤ n → (n.choose k) ≤ (n.choose 4)) →
  (∃ k : ℕ, (8 : ℝ) - (4 * k) / 3 = 0) →
  (∃ c : ℝ, c = (n.choose 6) * (1/2)^2 * (-1)^6) →
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l2331_233172


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l2331_233161

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l2331_233161


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l2331_233160

theorem sum_mod_thirteen : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l2331_233160


namespace NUMINAMATH_CALUDE_expression_evaluation_l2331_233136

theorem expression_evaluation (x : ℝ) (h : x = 4) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 2*x) / (x + 1)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2331_233136


namespace NUMINAMATH_CALUDE_temperature_conversion_l2331_233153

theorem temperature_conversion (C F : ℝ) : 
  C = (4/7) * (F - 40) → C = 28 → F = 89 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2331_233153


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2331_233178

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : ∃ x₀ y₀ : ℝ, x₀ + y₀ = 60 ∧ x₀ = 3 * y₀ ∧ InverselyProportional x₀ y₀) :
  x = 12 → y = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2331_233178


namespace NUMINAMATH_CALUDE_factorial_difference_l2331_233159

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 / Nat.factorial 3 = 3568320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2331_233159


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2331_233186

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ = 0 ∧ x₂^2 + 2*x₂ = 0 ∧ x₁ = 0 ∧ x₂ = -2) ∧
  (∃ x₁ x₂ : ℝ, (x₁+1)^2 - 144 = 0 ∧ (x₂+1)^2 - 144 = 0 ∧ x₁ = 11 ∧ x₂ = -13) ∧
  (∃ x₁ x₂ : ℝ, 3*(x₁-2)^2 = x₁*(x₁-2) ∧ 3*(x₂-2)^2 = x₂*(x₂-2) ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, x₁^2 + 5*x₁ - 1 = 0 ∧ x₂^2 + 5*x₂ - 1 = 0 ∧ 
    x₁ = (-5 + Real.sqrt 29) / 2 ∧ x₂ = (-5 - Real.sqrt 29) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2331_233186


namespace NUMINAMATH_CALUDE_circular_sector_properties_l2331_233199

/-- A circular sector with given area and perimeter -/
structure CircularSector where
  area : ℝ
  perimeter : ℝ

/-- The central angle of a circular sector -/
def central_angle (s : CircularSector) : ℝ := sorry

/-- The chord length of a circular sector -/
def chord_length (s : CircularSector) : ℝ := sorry

/-- Theorem stating the properties of a specific circular sector -/
theorem circular_sector_properties :
  let s : CircularSector := { area := 1, perimeter := 4 }
  (central_angle s = 2) ∧ (chord_length s = 2 * Real.sin 1) := by sorry

end NUMINAMATH_CALUDE_circular_sector_properties_l2331_233199


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2331_233148

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_nonneg : x ≥ 0)
  (y_geq : y ≥ -3/2)
  (z_geq : z ≥ -1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
    ∀ a b c : ℝ, a + b + c = 3 → a ≥ 0 → b ≥ -3/2 → c ≥ -1 →
      Real.sqrt (2 * a) + Real.sqrt (2 * b + 3) + Real.sqrt (2 * c + 2) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2331_233148


namespace NUMINAMATH_CALUDE_tyrones_pennies_l2331_233197

/-- The number of pennies Tyrone found -/
def pennies : ℕ := sorry

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The total value of Tyrone's money in dollars -/
def total_value : ℚ := 13

/-- The value of Tyrone's money excluding pennies -/
def value_without_pennies : ℚ :=
  2 * 1 + -- two $1 bills
  1 * 5 + -- one $5 bill
  13 * (1 / 4) + -- 13 quarters
  20 * (1 / 10) + -- 20 dimes
  8 * (1 / 20) -- 8 nickels

theorem tyrones_pennies :
  pennies * penny_value = total_value - value_without_pennies ∧
  pennies = 35 := by sorry

end NUMINAMATH_CALUDE_tyrones_pennies_l2331_233197


namespace NUMINAMATH_CALUDE_median_list_i_is_eight_l2331_233127

def list_i : List ℕ := [9, 2, 4, 7, 10, 11]
def list_ii : List ℕ := [3, 3, 4, 6, 7, 10]

def median (l : List ℕ) : ℚ := sorry
def mode (l : List ℕ) : ℕ := sorry

theorem median_list_i_is_eight :
  median list_i = 8 :=
by
  have h1 : median list_ii + mode list_ii = 8 := by sorry
  have h2 : median list_i = median list_ii + mode list_ii := by sorry
  sorry

end NUMINAMATH_CALUDE_median_list_i_is_eight_l2331_233127


namespace NUMINAMATH_CALUDE_jet_bar_sales_difference_l2331_233130

def weekly_target : ℕ := 90
def monday_sales : ℕ := 45
def remaining_sales : ℕ := 16

theorem jet_bar_sales_difference : 
  monday_sales - (weekly_target - remaining_sales - monday_sales) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jet_bar_sales_difference_l2331_233130


namespace NUMINAMATH_CALUDE_trigonometric_expressions_l2331_233120

theorem trigonometric_expressions (θ : ℝ) 
  (h : (4 * Real.sin θ - 2 * Real.cos θ) / (3 * Real.sin θ + 5 * Real.cos θ) = 6 / 11) : 
  (5 * (Real.cos θ)^2) / ((Real.sin θ)^2 + 2 * Real.sin θ * Real.cos θ - 3 * (Real.cos θ)^2) = 1 ∧ 
  1 - 4 * Real.sin θ * Real.cos θ + 2 * (Real.cos θ)^2 = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expressions_l2331_233120


namespace NUMINAMATH_CALUDE_t_range_for_strictly_decreasing_function_l2331_233142

theorem t_range_for_strictly_decreasing_function 
  (f : ℝ → ℝ) (h_decreasing : ∀ x y, x < y → f y < f x) :
  ∀ t : ℝ, f (t^2) - f t < 0 → t < 0 ∨ t > 1 :=
by sorry

end NUMINAMATH_CALUDE_t_range_for_strictly_decreasing_function_l2331_233142


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2331_233174

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function

/-- Properties of the specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 6 > seq.S 7 ∧ seq.S 7 > seq.S 5

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  seq.d < 0 ∧ 
  seq.S 11 > 0 ∧ 
  seq.S 12 > 0 ∧ 
  seq.S 8 < seq.S 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2331_233174


namespace NUMINAMATH_CALUDE_unique_solution_l2331_233128

theorem unique_solution (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_squares_eq : x^2 + y^2 + z^2 = 3)
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2331_233128


namespace NUMINAMATH_CALUDE_complex_cube_plus_one_in_first_quadrant_l2331_233141

theorem complex_cube_plus_one_in_first_quadrant : 
  let z : ℂ := 1 / Complex.I
  (z^3 + 1).re > 0 ∧ (z^3 + 1).im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_plus_one_in_first_quadrant_l2331_233141


namespace NUMINAMATH_CALUDE_closest_to_standard_weight_l2331_233193

def quality_errors : List ℝ := [-0.02, 0.1, -0.23, -0.3, 0.2]

theorem closest_to_standard_weight :
  ∀ x ∈ quality_errors, |(-0.02)| ≤ |x| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_standard_weight_l2331_233193


namespace NUMINAMATH_CALUDE_price_restoration_l2331_233139

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (increase_percentage : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  reduced_price * (1 + increase_percentage) = original_price →
  increase_percentage = 0.25 := by
  sorry

#check price_restoration

end NUMINAMATH_CALUDE_price_restoration_l2331_233139


namespace NUMINAMATH_CALUDE_ring_element_equality_l2331_233102

variable {A : Type*} [Ring A] [Finite A]

theorem ring_element_equality (a b : A) (h : (a * b - 1) * b = 0) : 
  b * (a * b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ring_element_equality_l2331_233102


namespace NUMINAMATH_CALUDE_bert_stamps_correct_l2331_233162

/-- The number of stamps Bert bought -/
def stamps_bought : ℕ := 300

/-- The number of stamps Bert had before the purchase -/
def stamps_before : ℕ := stamps_bought / 2

/-- The total number of stamps Bert has after the purchase -/
def total_stamps : ℕ := 450

/-- Theorem stating that the number of stamps Bert bought is correct -/
theorem bert_stamps_correct :
  stamps_bought = 300 ∧
  stamps_before = stamps_bought / 2 ∧
  total_stamps = stamps_before + stamps_bought :=
by sorry

end NUMINAMATH_CALUDE_bert_stamps_correct_l2331_233162


namespace NUMINAMATH_CALUDE_valid_sampling_interval_l2331_233154

def total_population : ℕ := 102
def removed_individuals : ℕ := 2
def sampling_interval : ℕ := 10

theorem valid_sampling_interval :
  (total_population - removed_individuals) % sampling_interval = 0 := by
  sorry

end NUMINAMATH_CALUDE_valid_sampling_interval_l2331_233154


namespace NUMINAMATH_CALUDE_phone_answer_probability_l2331_233182

theorem phone_answer_probability : 
  let p1 : ℚ := 1/10  -- Probability of answering on the first ring
  let p2 : ℚ := 3/10  -- Probability of answering on the second ring
  let p3 : ℚ := 2/5   -- Probability of answering on the third ring
  let p4 : ℚ := 1/10  -- Probability of answering on the fourth ring
  p1 + p2 + p3 + p4 = 9/10 := by
sorry

end NUMINAMATH_CALUDE_phone_answer_probability_l2331_233182


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2331_233123

theorem no_real_roots_quadratic : 
  {x : ℝ | x^2 - x + 1 = 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2331_233123


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2331_233144

theorem quadratic_equations_solutions :
  (∀ x, x^2 - 7*x - 18 = 0 ↔ x = 9 ∨ x = -2) ∧
  (∀ x, 4*x^2 + 1 = 4*x ↔ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2331_233144


namespace NUMINAMATH_CALUDE_total_purchase_options_l2331_233145

/-- The number of oreo flavors --/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors --/
def num_milk_flavors : ℕ := 4

/-- The total number of product types --/
def total_product_types : ℕ := num_oreo_flavors + num_milk_flavors

/-- The total number of products they purchase --/
def total_purchases : ℕ := 4

/-- Charlie's purchase options --/
def charlie_options (k : ℕ) : ℕ := Nat.choose total_product_types k

/-- Delta's oreo purchase options when buying k oreos --/
def delta_options (k : ℕ) : ℕ :=
  Nat.choose num_oreo_flavors k +
  if k ≥ 2 then num_oreo_flavors * Nat.choose (num_oreo_flavors - 1) (k - 2) else 0 +
  if k = 3 then num_oreo_flavors else 0

/-- The main theorem stating the total number of ways to purchase --/
theorem total_purchase_options : 
  (charlie_options 3 * num_oreo_flavors) +
  (charlie_options 2 * delta_options 2) +
  (charlie_options 1 * delta_options 3) = 2225 := by
  sorry

end NUMINAMATH_CALUDE_total_purchase_options_l2331_233145


namespace NUMINAMATH_CALUDE_range_of_a_for_non_negative_x_l2331_233196

theorem range_of_a_for_non_negative_x (a x : ℝ) : 
  (x - a = 1 - 2*x ∧ x ≥ 0) → a ≥ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_non_negative_x_l2331_233196


namespace NUMINAMATH_CALUDE_desiree_age_proof_l2331_233168

/-- Desiree's current age -/
def desiree_age : ℕ := 6

/-- Desiree's cousin's current age -/
def cousin_age : ℕ := 3

/-- Proves that Desiree's current age is 6 years old -/
theorem desiree_age_proof :
  (desiree_age = 2 * cousin_age) ∧
  (desiree_age + 30 = (2/3 : ℚ) * (cousin_age + 30) + 14) →
  desiree_age = 6 := by
sorry

end NUMINAMATH_CALUDE_desiree_age_proof_l2331_233168


namespace NUMINAMATH_CALUDE_polynomial_division_l2331_233110

-- Define the theorem
theorem polynomial_division (a b : ℝ) (h : b ≠ 2*a) : 
  (4*a^2 - b^2) / (b - 2*a) = -2*a - b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l2331_233110


namespace NUMINAMATH_CALUDE_plains_total_area_l2331_233157

def plain_problem (region_B region_A total : ℕ) : Prop :=
  (region_B = 200) ∧
  (region_A = region_B - 50) ∧
  (total = region_A + region_B)

theorem plains_total_area : 
  ∃ (region_B region_A total : ℕ), 
    plain_problem region_B region_A total ∧ total = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_plains_total_area_l2331_233157


namespace NUMINAMATH_CALUDE_set_equality_l2331_233190

theorem set_equality (A B X : Set α) 
  (h1 : A ∪ B ∪ X = A ∪ B) 
  (h2 : A ∩ X = A ∩ B) 
  (h3 : B ∩ X = A ∩ B) : 
  X = A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2331_233190


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2331_233112

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then k = 1 -/
theorem perpendicular_vectors_k_value (k : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, k)
  let a : ℝ × ℝ := (-1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 = 0) → k = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l2331_233112


namespace NUMINAMATH_CALUDE_seating_arrangements_l2331_233135

/-- The number of boys -/
def num_boys : Nat := 5

/-- The number of girls -/
def num_girls : Nat := 4

/-- The total number of chairs -/
def total_chairs : Nat := 9

/-- The number of odd-numbered chairs -/
def odd_chairs : Nat := (total_chairs + 1) / 2

/-- The number of even-numbered chairs -/
def even_chairs : Nat := total_chairs / 2

/-- Factorial function -/
def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem seating_arrangements :
  factorial num_boys * factorial num_girls = 2880 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2331_233135


namespace NUMINAMATH_CALUDE_max_value_xyz_l2331_233177

theorem max_value_xyz (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  x * y^2 * z^3 ≤ 1 / 432 :=
sorry

end NUMINAMATH_CALUDE_max_value_xyz_l2331_233177


namespace NUMINAMATH_CALUDE_sqrt_of_negative_nine_l2331_233113

theorem sqrt_of_negative_nine :
  (3 * Complex.I)^2 = -9 ∧ (-3 * Complex.I)^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_nine_l2331_233113


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2331_233122

/-- An arithmetic sequence with first four terms a, y, b, 3y has a/b = 0 -/
theorem arithmetic_sequence_ratio (a y b : ℝ) : 
  (∃ d : ℝ, y = a + d ∧ b = y + d ∧ 3*y = b + d) → a / b = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2331_233122


namespace NUMINAMATH_CALUDE_b_value_l2331_233104

theorem b_value (a b : ℚ) : 
  (let x := 2 + Real.sqrt 3
   x^3 + a*x^2 + b*x - 20 = 0) →
  b = 81 := by
sorry

end NUMINAMATH_CALUDE_b_value_l2331_233104


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l2331_233105

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two points are symmetric with respect to a line -/
def symmetric_points (P Q : ℝ × ℝ) (l : Line) : Prop := sorry

/-- A point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

theorem circle_symmetry_line (C : Circle) (P Q : ℝ × ℝ) (m : ℝ) :
  C.center = (-1, 3) →
  C.radius = 3 →
  on_circle P C →
  on_circle Q C →
  symmetric_points P Q (Line.mk 1 m 4) →
  m = -1 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l2331_233105


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2331_233126

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 = 12) →
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2331_233126


namespace NUMINAMATH_CALUDE_brian_commission_l2331_233181

/-- Calculates the commission for a given sale price and commission rate -/
def calculate_commission (sale_price : ℝ) (commission_rate : ℝ) : ℝ :=
  sale_price * commission_rate

/-- Calculates the total commission for multiple sales -/
def total_commission (sale_prices : List ℝ) (commission_rate : ℝ) : ℝ :=
  sale_prices.map (λ price => calculate_commission price commission_rate) |>.sum

theorem brian_commission :
  let commission_rate : ℝ := 0.02
  let sale_prices : List ℝ := [157000, 499000, 125000]
  total_commission sale_prices commission_rate = 15620 := by
  sorry

end NUMINAMATH_CALUDE_brian_commission_l2331_233181


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l2331_233187

/-- A rectangular box inscribed in a sphere --/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  r : ℝ
  h_surface_area : 2 * (x*y + x*z + y*z) = 432
  h_edge_sum : 4 * (x + y + z) = 104
  h_inscribed : (2*r)^2 = x^2 + y^2 + z^2

/-- Theorem: If a rectangular box Q is inscribed in a sphere, with surface area 432,
    sum of edge lengths 104, and one dimension 8, then the radius of the sphere is 7 --/
theorem inscribed_box_radius (Q : InscribedBox) (h_x : Q.x = 8) : Q.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l2331_233187


namespace NUMINAMATH_CALUDE_custom_op_solution_l2331_233109

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 10 = 90 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l2331_233109


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l2331_233176

theorem complex_modulus_squared (w : ℂ) (h : w + Complex.abs w = 4 + 5*I) : 
  Complex.abs w ^ 2 = 1681 / 64 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l2331_233176


namespace NUMINAMATH_CALUDE_third_chest_silver_excess_l2331_233192

/-- Represents the number of coins in each chest -/
structure ChestContents where
  gold : ℕ
  silver : ℕ

/-- Problem setup -/
def coin_problem (chest1 chest2 chest3 : ChestContents) : Prop :=
  let total_gold := chest1.gold + chest2.gold + chest3.gold
  let total_silver := chest1.silver + chest2.silver + chest3.silver
  total_gold = 40 ∧
  total_silver = 40 ∧
  chest1.gold = chest1.silver + 7 ∧
  chest2.gold = chest2.silver + 15

/-- Theorem statement -/
theorem third_chest_silver_excess 
  (chest1 chest2 chest3 : ChestContents) 
  (h : coin_problem chest1 chest2 chest3) : 
  chest3.silver = chest3.gold + 22 := by
  sorry

#check third_chest_silver_excess

end NUMINAMATH_CALUDE_third_chest_silver_excess_l2331_233192


namespace NUMINAMATH_CALUDE_square_of_101_l2331_233129

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l2331_233129


namespace NUMINAMATH_CALUDE_farmers_market_sales_l2331_233183

/-- The farmers' market sales problem -/
theorem farmers_market_sales
  (total_earnings : ℕ)
  (broccoli_sales : ℕ)
  (carrot_sales : ℕ)
  (spinach_sales : ℕ)
  (cauliflower_sales : ℕ)
  (h1 : total_earnings = 380)
  (h2 : broccoli_sales = 57)
  (h3 : carrot_sales = 2 * broccoli_sales)
  (h4 : spinach_sales = carrot_sales / 2 + 16)
  (h5 : total_earnings = broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales) :
  cauliflower_sales = 136 := by
  sorry


end NUMINAMATH_CALUDE_farmers_market_sales_l2331_233183


namespace NUMINAMATH_CALUDE_compute_expression_l2331_233156

theorem compute_expression : 10 + 4 * (5 + 3)^3 = 2058 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2331_233156


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2331_233171

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2^x - 1 < 0) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2331_233171


namespace NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l2331_233116

theorem product_repeating_decimal_and_seven (x : ℚ) : 
  (x = 1/3) → (x * 7 = 7/3) := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_decimal_and_seven_l2331_233116


namespace NUMINAMATH_CALUDE_exists_non_negative_sums_l2331_233149

/-- Represents a sign change operation on a matrix -/
inductive SignChange
| Row (i : Nat)
| Col (j : Nat)

/-- Apply a sequence of sign changes to a matrix -/
def applySignChanges (A : Matrix (Fin m) (Fin n) ℝ) (changes : List SignChange) : Matrix (Fin m) (Fin n) ℝ :=
  sorry

/-- Check if all row and column sums are non-negative -/
def allSumsNonNegative (A : Matrix (Fin m) (Fin n) ℝ) : Prop :=
  sorry

/-- Main theorem: For any real matrix, there exists a sequence of sign changes
    that results in all row and column sums being non-negative -/
theorem exists_non_negative_sums (A : Matrix (Fin m) (Fin n) ℝ) :
  ∃ (changes : List SignChange), allSumsNonNegative (applySignChanges A changes) :=
  sorry

end NUMINAMATH_CALUDE_exists_non_negative_sums_l2331_233149


namespace NUMINAMATH_CALUDE_andy_position_after_2023_turns_l2331_233133

/-- Andy's position on the coordinate plane -/
structure Position where
  x : Int
  y : Int

/-- Direction Andy is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Andy's state, including position and direction -/
structure AndyState where
  pos : Position
  dir : Direction

/-- Function to update Andy's state after one move -/
def move (state : AndyState) (distance : Int) : AndyState :=
  sorry

/-- Function to turn Andy 90° right -/
def turnRight (dir : Direction) : Direction :=
  sorry

/-- Function to simulate Andy's movement for a given number of turns -/
def simulateAndy (initialState : AndyState) (turns : Nat) : Position :=
  sorry

theorem andy_position_after_2023_turns :
  let initialState : AndyState := { pos := { x := 10, y := -10 }, dir := Direction.North }
  let finalPosition := simulateAndy initialState 2023
  finalPosition = { x := 1022, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_andy_position_after_2023_turns_l2331_233133


namespace NUMINAMATH_CALUDE_equivalent_operations_l2331_233167

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (7/4) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2331_233167


namespace NUMINAMATH_CALUDE_square_division_theorem_l2331_233103

-- Define the square and point P
def Square (E F G H : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := E
  let (x₂, y₂) := F
  let (x₃, y₃) := G
  let (x₄, y₄) := H
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 16 ∧
  (x₃ - x₂)^2 + (y₃ - y₂)^2 = 16 ∧
  (x₄ - x₃)^2 + (y₄ - y₃)^2 = 16 ∧
  (x₁ - x₄)^2 + (y₁ - y₄)^2 = 16

def PointOnSide (P : ℝ × ℝ) (E H : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * E.1 + (1 - t) * H.1, t * E.2 + (1 - t) * H.2)

-- Define the area division property
def DivideAreaEqually (P : ℝ × ℝ) (E F G H : ℝ × ℝ) : Prop :=
  let area_EFP := abs ((F.1 - E.1) * (P.2 - E.2) - (P.1 - E.1) * (F.2 - E.2)) / 2
  let area_FGP := abs ((G.1 - F.1) * (P.2 - F.2) - (P.1 - F.1) * (G.2 - F.2)) / 2
  let area_GHP := abs ((H.1 - G.1) * (P.2 - G.2) - (P.1 - G.1) * (H.2 - G.2)) / 2
  let area_HEP := abs ((E.1 - H.1) * (P.2 - H.2) - (P.1 - H.1) * (E.2 - H.2)) / 2
  area_EFP = area_FGP ∧ area_FGP = area_GHP ∧ area_GHP = area_HEP

-- State the theorem
theorem square_division_theorem (E F G H P : ℝ × ℝ) :
  Square E F G H →
  PointOnSide P E H →
  DivideAreaEqually P E F G H →
  (F.1 - P.1)^2 + (F.2 - P.2)^2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l2331_233103


namespace NUMINAMATH_CALUDE_stock_rise_amount_l2331_233119

/-- Represents the daily change in stock value -/
structure StockChange where
  morning_rise : ℝ
  afternoon_fall : ℝ

/-- Calculates the stock value after n days given initial value and daily change -/
def stock_value_after_days (initial_value : ℝ) (daily_change : StockChange) (n : ℕ) : ℝ :=
  initial_value + n * (daily_change.morning_rise - daily_change.afternoon_fall)

theorem stock_rise_amount (initial_value : ℝ) (daily_change : StockChange) :
  initial_value = 100 →
  daily_change.afternoon_fall = 1 →
  stock_value_after_days initial_value daily_change 100 = 200 →
  daily_change.morning_rise = 2 := by
  sorry

#eval stock_value_after_days 100 ⟨2, 1⟩ 100

end NUMINAMATH_CALUDE_stock_rise_amount_l2331_233119


namespace NUMINAMATH_CALUDE_max_volume_rect_frame_l2331_233108

/-- Represents the dimensions of a rectangular frame. -/
structure RectFrame where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular frame. -/
def volume (frame : RectFrame) : ℝ :=
  frame.length * frame.width * frame.height

/-- Calculates the perimeter of the base of a rectangular frame. -/
def basePerimeter (frame : RectFrame) : ℝ :=
  2 * (frame.length + frame.width)

/-- Calculates the total length of steel bar used for a rectangular frame. -/
def totalBarLength (frame : RectFrame) : ℝ :=
  basePerimeter frame + 4 * frame.height

/-- Theorem: The maximum volume of a rectangular frame enclosed by an 18m steel bar,
    where the ratio of length to width is 2:1, is equal to the correct maximum volume. -/
theorem max_volume_rect_frame :
  ∃ (frame : RectFrame),
    frame.length = 2 * frame.width ∧
    totalBarLength frame = 18 ∧
    ∀ (other : RectFrame),
      other.length = 2 * other.width →
      totalBarLength other = 18 →
      volume frame ≥ volume other :=
by sorry


end NUMINAMATH_CALUDE_max_volume_rect_frame_l2331_233108


namespace NUMINAMATH_CALUDE_bens_old_car_cost_l2331_233114

/-- Proves that Ben's old car cost $1900 given the problem conditions -/
theorem bens_old_car_cost :
  ∀ (old_car_cost new_car_cost : ℕ),
  new_car_cost = 2 * old_car_cost →
  old_car_cost = 1800 →
  new_car_cost = 1800 + 2000 →
  old_car_cost = 1900 := by
  sorry

#check bens_old_car_cost

end NUMINAMATH_CALUDE_bens_old_car_cost_l2331_233114


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_thirds_l2331_233175

theorem sin_thirteen_pi_thirds : Real.sin (13 * π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_thirds_l2331_233175


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l2331_233138

/-- The degree of (5x^3 + 7)^10 is 30 -/
theorem degree_of_polynomial (x : ℝ) : 
  Polynomial.degree ((5 * X + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l2331_233138


namespace NUMINAMATH_CALUDE_root_coincidence_problem_l2331_233131

theorem root_coincidence_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧
   (2*r + s = -a) ∧ (r^2 + 2*r*s = b) ∧ (r^2 * s = -16*a)) →
  |a*b| = 2128 :=
sorry

end NUMINAMATH_CALUDE_root_coincidence_problem_l2331_233131


namespace NUMINAMATH_CALUDE_cars_without_ac_l2331_233173

theorem cars_without_ac (total : ℕ) (min_racing : ℕ) (max_ac_no_racing : ℕ) 
  (h_total : total = 100)
  (h_min_racing : min_racing = 41)
  (h_max_ac_no_racing : max_ac_no_racing = 59) :
  total - (max_ac_no_racing + 0) = 41 := by
  sorry

end NUMINAMATH_CALUDE_cars_without_ac_l2331_233173


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2331_233146

theorem negative_fraction_comparison : -3/5 < -4/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2331_233146


namespace NUMINAMATH_CALUDE_xy_equation_solution_l2331_233189

theorem xy_equation_solution (x y : ℕ+) (p q : ℕ) :
  x ≥ y →
  x * y - (x + y) = 2 * p + q →
  p = Nat.gcd x y →
  q = Nat.lcm x y →
  ((x = 9 ∧ y = 3) ∨ (x = 5 ∧ y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_xy_equation_solution_l2331_233189


namespace NUMINAMATH_CALUDE_total_walking_hours_l2331_233184

/-- Represents the types of dogs Charlotte walks -/
inductive DogType
  | Poodle
  | Chihuahua
  | Labrador

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday

/-- Returns the number of hours it takes to walk a dog of a given type -/
def walkingTime (d : DogType) : Nat :=
  match d with
  | DogType.Poodle => 2
  | DogType.Chihuahua => 1
  | DogType.Labrador => 3

/-- Returns the number of dogs of a given type walked on a specific day -/
def dogsWalked (day : Day) (dogType : DogType) : Nat :=
  match day, dogType with
  | Day.Monday, DogType.Poodle => 4
  | Day.Monday, DogType.Chihuahua => 2
  | Day.Monday, DogType.Labrador => 0
  | Day.Tuesday, DogType.Poodle => 4
  | Day.Tuesday, DogType.Chihuahua => 2
  | Day.Tuesday, DogType.Labrador => 0
  | Day.Wednesday, DogType.Poodle => 0
  | Day.Wednesday, DogType.Chihuahua => 0
  | Day.Wednesday, DogType.Labrador => 4

/-- Calculates the total hours spent walking dogs on a given day -/
def hoursPerDay (day : Day) : Nat :=
  (dogsWalked day DogType.Poodle * walkingTime DogType.Poodle) +
  (dogsWalked day DogType.Chihuahua * walkingTime DogType.Chihuahua) +
  (dogsWalked day DogType.Labrador * walkingTime DogType.Labrador)

/-- Theorem stating that the total hours for dog-walking this week is 32 -/
theorem total_walking_hours :
  hoursPerDay Day.Monday + hoursPerDay Day.Tuesday + hoursPerDay Day.Wednesday = 32 := by
  sorry


end NUMINAMATH_CALUDE_total_walking_hours_l2331_233184


namespace NUMINAMATH_CALUDE_sector_area_l2331_233180

/-- Given a sector with arc length and radius both equal to 2, its area is 2. -/
theorem sector_area (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) :
  (1 / 2) * arc_length * radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2331_233180
