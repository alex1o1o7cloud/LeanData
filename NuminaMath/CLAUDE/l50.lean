import Mathlib

namespace NUMINAMATH_CALUDE_derivative_at_negative_one_l50_5058

/-- Given a function f(x) = ax^4 + bx^2 + c, if f'(1) = 2, then f'(-1) = -2 -/
theorem derivative_at_negative_one
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_one_l50_5058


namespace NUMINAMATH_CALUDE_sum_of_p_and_q_l50_5023

theorem sum_of_p_and_q (p q : ℝ) (h_distinct : p ≠ q) (h_greater : p > q) :
  let M := !![2, -5, 8; 1, p, q; 1, q, p]
  Matrix.det M = 0 → p + q = -13/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_p_and_q_l50_5023


namespace NUMINAMATH_CALUDE_swimming_problem_l50_5068

/-- Represents the daily swimming distances of Jamir, Sarah, and Julien -/
structure SwimmingDistances where
  julien : ℕ
  sarah : ℕ
  jamir : ℕ

/-- Calculates the total distance swam by all three swimmers in a week -/
def weeklyTotalDistance (d : SwimmingDistances) : ℕ :=
  7 * (d.julien + d.sarah + d.jamir)

/-- The swimming problem statement -/
theorem swimming_problem (d : SwimmingDistances) : 
  d.sarah = 2 * d.julien →
  d.jamir = d.sarah + 20 →
  weeklyTotalDistance d = 1890 →
  d.julien = 50 := by
  sorry

end NUMINAMATH_CALUDE_swimming_problem_l50_5068


namespace NUMINAMATH_CALUDE_system_solution_l50_5072

theorem system_solution :
  ∃ (x y : ℝ),
    (10 * x^2 + 5 * y^2 - 2 * x * y - 38 * x - 6 * y + 41 = 0) ∧
    (3 * x^2 - 2 * y^2 + 5 * x * y - 17 * x - 6 * y + 20 = 0) ∧
    (x = 2) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l50_5072


namespace NUMINAMATH_CALUDE_min_value_A_over_C_l50_5070

theorem min_value_A_over_C (x : ℝ) (A C : ℝ) (h1 : x^2 + 1/x^2 = A) (h2 : x + 1/x = C)
  (h3 : A > 0) (h4 : C > 0) (h5 : ∀ y : ℝ, y > 0 → y + 1/y ≥ 2) :
  A / C ≥ 1 ∧ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀^2 + 1/x₀^2) / (x₀ + 1/x₀) = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_A_over_C_l50_5070


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l50_5028

/-- Given points A, B, C, and the conditions that A' and B' lie on y = x,
    prove that the length of A'B' is 3√2/28 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 10) →
  C = (3, 6) →
  (∃ t : ℝ, A' = (t, t)) →
  (∃ s : ℝ, B' = (s, s)) →
  (∃ k : ℝ, A'.1 = k * (C.1 - A.1) + A.1 ∧ A'.2 = k * (C.2 - A.2) + A.2) →
  (∃ m : ℝ, B'.1 = m * (C.1 - B.1) + B.1 ∧ B'.2 = m * (C.2 - B.2) + B.2) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 3 * Real.sqrt 2 / 28 := by
sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l50_5028


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_first_six_l50_5042

/-- A geometric sequence with positive terms satisfying a_{n+2} + 2a_{n+1} = 8a_n -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (a 1 = 1) ∧
  (∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n)

/-- The sum of the first 6 terms of the geometric sequence -/
def SumFirstSixTerms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem geometric_sequence_sum_first_six (a : ℕ → ℝ) 
  (h : GeometricSequence a) : SumFirstSixTerms a = 63 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_first_six_l50_5042


namespace NUMINAMATH_CALUDE_athlete_A_second_day_prob_l50_5087

-- Define the probabilities
def prob_A_first_day : ℝ := 0.5
def prob_B_first_day : ℝ := 0.5
def prob_A_second_day_given_A_first : ℝ := 0.6
def prob_A_second_day_given_B_first : ℝ := 0.5

-- State the theorem
theorem athlete_A_second_day_prob :
  prob_A_first_day * prob_A_second_day_given_A_first +
  prob_B_first_day * prob_A_second_day_given_B_first = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_athlete_A_second_day_prob_l50_5087


namespace NUMINAMATH_CALUDE_percentage_to_pass_l50_5045

def max_marks : ℕ := 780
def passing_marks : ℕ := 234

theorem percentage_to_pass : 
  (passing_marks : ℝ) / max_marks * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l50_5045


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l50_5099

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l50_5099


namespace NUMINAMATH_CALUDE_sine_product_inequality_l50_5050

theorem sine_product_inequality :
  1/8 < Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) ∧
  Real.sin (20 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sine_product_inequality_l50_5050


namespace NUMINAMATH_CALUDE_correct_statements_l50_5007

-- Define the proof methods
inductive ProofMethod
| Synthetic
| Analytic
| Contradiction

-- Define the characteristics of proof methods
def isCauseAndEffect (m : ProofMethod) : Prop := sorry
def isResultToCause (m : ProofMethod) : Prop := sorry
def isDirectProof (m : ProofMethod) : Prop := sorry

-- Define the statements
def statement1 : Prop := isCauseAndEffect ProofMethod.Synthetic
def statement2 : Prop := ¬(isDirectProof ProofMethod.Analytic)
def statement3 : Prop := isResultToCause ProofMethod.Analytic
def statement4 : Prop := isDirectProof ProofMethod.Contradiction

-- Theorem stating which statements are correct
theorem correct_statements :
  statement1 ∧ statement3 ∧ ¬statement2 ∧ ¬statement4 := by sorry

end NUMINAMATH_CALUDE_correct_statements_l50_5007


namespace NUMINAMATH_CALUDE_root_sum_l50_5005

theorem root_sum (n m : ℝ) (h1 : n ≠ 0) (h2 : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_l50_5005


namespace NUMINAMATH_CALUDE_distance_X_to_Y_l50_5079

/-- The distance between points X and Y -/
def D : ℝ := sorry

/-- Yolanda's walking rate in miles per hour -/
def yolanda_rate : ℝ := 3

/-- Bob's walking rate in miles per hour -/
def bob_rate : ℝ := 4

/-- Time difference between Yolanda and Bob's start in hours -/
def time_difference : ℝ := 1

/-- Distance Bob walked when they met -/
def bob_distance : ℝ := 4

/-- Theorem stating the distance between X and Y -/
theorem distance_X_to_Y : D = 10 := by sorry

end NUMINAMATH_CALUDE_distance_X_to_Y_l50_5079


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l50_5004

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x - 3

theorem quadratic_solution_set (m : ℝ) : 
  (∀ x : ℝ, f m x ≤ 0 ↔ -1 < x ∧ x < 3) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l50_5004


namespace NUMINAMATH_CALUDE_root_range_implies_m_range_l50_5088

theorem root_range_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - 2*m*x + m^2 - 1 = 0 → x > -2) →
  m > -1 :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_m_range_l50_5088


namespace NUMINAMATH_CALUDE_quadratic_inequality_l50_5046

theorem quadratic_inequality (x : ℝ) :
  9 * x^2 - 6 * x + 1 > 0 ↔ x < 1/3 ∨ x > 1/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l50_5046


namespace NUMINAMATH_CALUDE_chess_board_configurations_l50_5017

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct items -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place pawns on the board -/
def num_pawn_placements : ℕ := factorial 5

/-- The number of ways to assign distinct pawns to positions -/
def num_pawn_assignments : ℕ := factorial 5

/-- The total number of valid configurations -/
def total_configurations : ℕ := num_pawn_placements * num_pawn_assignments

/-- Theorem stating the total number of valid configurations -/
theorem chess_board_configurations :
  total_configurations = 14400 := by sorry

end NUMINAMATH_CALUDE_chess_board_configurations_l50_5017


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l50_5016

def num_flips : ℕ := 8
def min_heads : ℕ := 6

-- Probability of getting at least min_heads in num_flips flips of a fair coin
def prob_at_least_heads : ℚ :=
  (Finset.sum (Finset.range (num_flips - min_heads + 1))
    (λ i => Nat.choose num_flips (num_flips - i))) / 2^num_flips

theorem prob_at_least_six_heads_in_eight_flips :
  prob_at_least_heads = 37 / 256 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l50_5016


namespace NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l50_5067

theorem shortest_altitude_of_special_triangle :
  ∀ (a b c h : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = (1/2) * c * h →
  h = 7.2 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_special_triangle_l50_5067


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l50_5041

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ

/-- Properties of the isosceles triangle -/
def IsoscelesTriangle.properties (t : IsoscelesTriangle) : Prop :=
  t.base = 16 ∧ t.side = 10

/-- Inradius of the triangle -/
def inradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Circumradius of the triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Distance between the centers of inscribed and circumscribed circles -/
def centerDistance (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem about the properties of the isosceles triangle -/
theorem isosceles_triangle_properties (t : IsoscelesTriangle) 
  (h : t.properties) : 
  inradius t = 8/3 ∧ 
  circumradius t = 25/3 ∧ 
  centerDistance t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_properties_l50_5041


namespace NUMINAMATH_CALUDE_simon_is_ten_l50_5044

def alvin_age : ℕ := 30

def simon_age : ℕ := alvin_age / 2 - 5

theorem simon_is_ten : simon_age = 10 := by
  sorry

end NUMINAMATH_CALUDE_simon_is_ten_l50_5044


namespace NUMINAMATH_CALUDE_ed_doug_marble_difference_l50_5089

theorem ed_doug_marble_difference (ed_initial : ℕ) (doug_initial : ℕ) (ed_lost : ℕ) (ed_final : ℕ) :
  ed_initial = doug_initial + 30 →
  ed_initial = ed_final + ed_lost →
  ed_lost = 21 →
  ed_final = 91 →
  ed_final - doug_initial = 9 :=
by sorry

end NUMINAMATH_CALUDE_ed_doug_marble_difference_l50_5089


namespace NUMINAMATH_CALUDE_inequality_problem_l50_5036

theorem inequality_problem (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr : p * r > q * r) :
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l50_5036


namespace NUMINAMATH_CALUDE_unique_scenario_l50_5084

/-- Represents the type of islander -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents the possible responses to the question -/
inductive Response
  | Yes
  | No

/-- Represents the scenario of two islanders -/
structure IslandScenario where
  askedIslander : IslanderType
  otherIslander : IslanderType
  response : Response

/-- Determines if a given scenario is consistent with the rules of knights and liars -/
def isConsistentScenario (scenario : IslandScenario) : Prop :=
  match scenario.askedIslander, scenario.response with
  | IslanderType.Knight, Response.Yes => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight
  | IslanderType.Knight, Response.No => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.Yes => scenario.askedIslander ≠ IslanderType.Knight ∧ scenario.otherIslander ≠ IslanderType.Knight
  | IslanderType.Liar, Response.No => scenario.askedIslander = IslanderType.Knight ∨ scenario.otherIslander = IslanderType.Knight

/-- Determines if a given scenario provides definitive information about both islanders -/
def providesDefinitiveInfo (scenario : IslandScenario) : Prop :=
  isConsistentScenario scenario ∧
  ∀ (altScenario : IslandScenario),
    isConsistentScenario altScenario →
    scenario.askedIslander = altScenario.askedIslander ∧
    scenario.otherIslander = altScenario.otherIslander

/-- The main theorem: The only scenario that satisfies all conditions is when the asked islander is a liar and the other is a knight -/
theorem unique_scenario :
  ∃! (scenario : IslandScenario),
    isConsistentScenario scenario ∧
    providesDefinitiveInfo scenario ∧
    scenario.askedIslander = IslanderType.Liar ∧
    scenario.otherIslander = IslanderType.Knight :=
  sorry

end NUMINAMATH_CALUDE_unique_scenario_l50_5084


namespace NUMINAMATH_CALUDE_scatter_diagram_placement_l50_5076

/-- Represents a variable in a scatter diagram -/
inductive ScatterVariable
| Explanatory
| Predictor

/-- Represents an axis in a scatter diagram -/
inductive Axis
| X
| Y

/-- Determines the correct axis for a given scatter variable -/
def correct_axis_placement (v : ScatterVariable) : Axis :=
  match v with
  | ScatterVariable.Explanatory => Axis.X
  | ScatterVariable.Predictor => Axis.Y

/-- Theorem stating the correct placement of variables in a scatter diagram -/
theorem scatter_diagram_placement :
  (correct_axis_placement ScatterVariable.Explanatory = Axis.X) ∧
  (correct_axis_placement ScatterVariable.Predictor = Axis.Y) :=
by sorry

end NUMINAMATH_CALUDE_scatter_diagram_placement_l50_5076


namespace NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l50_5053

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_fourth_and_fifth_terms
  (a : ℕ → ℕ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_second : a 2 = 10)
  (h_third : a 3 = 17)
  (h_sixth : a 6 = 38) :
  a 4 + a 5 = 55 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_and_fifth_terms_l50_5053


namespace NUMINAMATH_CALUDE_inequality_solution_l50_5037

theorem inequality_solution (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 ∧ 
  x ≠ -5/3 ∧ 
  (4*x^2 + 2) / (5 + 3*x) ≥ 1 ↔ 
  x ∈ Set.Icc (-3 : ℝ) (-3/4) ∪ Set.Icc 1 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l50_5037


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l50_5055

theorem units_digit_sum_of_powers : (42^5 + 24^5 + 2^5) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l50_5055


namespace NUMINAMATH_CALUDE_max_candy_leftover_l50_5086

theorem max_candy_leftover (x : ℕ+) : ∃ (q r : ℕ), x = 7 * q + r ∧ r ≤ 6 ∧ ∀ (r' : ℕ), x = 7 * q + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l50_5086


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l50_5027

-- Define the points
def P : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (-2, 3)
def N : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + y = 1

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 5

-- Theorem for the line
theorem line_passes_through_P_and_forms_triangle :
  line_equation P.1 P.2 ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (1/2 : ℝ) * a * b = 1/2) :=
sorry

-- Theorem for the circle
theorem circle_passes_through_M_and_N_with_center_on_y_axis :
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 ∧
  (∃ y : ℝ, circle_equation 0 y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_forms_triangle_circle_passes_through_M_and_N_with_center_on_y_axis_l50_5027


namespace NUMINAMATH_CALUDE_sqrt_inequality_l50_5021

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d)
  (h5 : a + d = b + c) : 
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l50_5021


namespace NUMINAMATH_CALUDE_knight_statements_count_l50_5073

/-- Represents the type of islanders -/
inductive IslanderType
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : ℕ := 28

/-- The number of times "You are a liar!" was said -/
def liar_statements : ℕ := 230

/-- Function to calculate the number of "You are a knight!" statements -/
def knight_statements (knights : ℕ) (liars : ℕ) : ℕ :=
  knights * (knights - 1) / 2 + liars * (liars - 1) / 2

theorem knight_statements_count :
  ∃ (knights liars : ℕ),
    knights ≥ 2 ∧
    liars ≥ 2 ∧
    knights + liars = total_islanders ∧
    knights * liars = liar_statements / 2 ∧
    knight_statements knights liars + liar_statements = total_islanders * (total_islanders - 1) ∧
    knight_statements knights liars = 526 :=
by
  sorry

end NUMINAMATH_CALUDE_knight_statements_count_l50_5073


namespace NUMINAMATH_CALUDE_mod_37_5_l50_5047

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_37_5_l50_5047


namespace NUMINAMATH_CALUDE_corrected_mean_l50_5002

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong1 wrong2 correct1 correct2 : ℝ) 
  (h1 : n = 100)
  (h2 : original_mean = 56)
  (h3 : wrong1 = 38)
  (h4 : wrong2 = 27)
  (h5 : correct1 = 89)
  (h6 : correct2 = 73) :
  let incorrect_sum := n * original_mean
  let difference := (correct1 + correct2) - (wrong1 + wrong2)
  let corrected_sum := incorrect_sum + difference
  corrected_sum / n = 56.97 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_l50_5002


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l50_5033

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum : a 6 + a 8 = 4) : 
  a 8 * (a 4 + 2 * a 6 + a 8) = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l50_5033


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l50_5056

theorem ticket_price_possibilities : ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, x > 0 ∧ 72 % x = 0 ∧ 90 % x = 0 ∧ 150 % x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l50_5056


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l50_5054

/-- The number of female students --/
def total_students : ℕ := 6

/-- The number of students to be selected for the relay race --/
def selected_students : ℕ := 4

/-- A function to calculate the number of arrangements when only one of A or B participates --/
def one_participates : ℕ := 2 * (total_students - 2).choose (selected_students - 1) * (selected_students).factorial

/-- A function to calculate the number of arrangements when both A and B participate --/
def both_participate : ℕ := selected_students.choose 2 * (selected_students - 1).factorial

/-- The total number of different arrangements --/
def total_arrangements : ℕ := one_participates + both_participate

theorem relay_race_arrangements :
  total_arrangements = 264 := by sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l50_5054


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l50_5038

theorem greatest_integer_solution : 
  ∃ (x : ℤ), (8 - 6*x > 26) ∧ (∀ (y : ℤ), y > x → 8 - 6*y ≤ 26) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l50_5038


namespace NUMINAMATH_CALUDE_negation_equivalence_l50_5034

theorem negation_equivalence :
  (¬ ∃ x₀ > 0, x₀^2 - 5*x₀ + 6 > 0) ↔ (∀ x > 0, x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l50_5034


namespace NUMINAMATH_CALUDE_intersecting_circles_chord_length_l50_5011

/-- Given two circles with radii 10 and 7, whose centers are 15 units apart,
    and a point P where the circles intersect, if a line is drawn through P
    such that QP = PR, then QP^2 = 10800/35. -/
theorem intersecting_circles_chord_length 
  (O₁ O₂ P Q R : ℝ × ℝ) -- Points in 2D plane
  (h₁ : dist O₁ O₂ = 15) -- Centers are 15 units apart
  (h₂ : dist O₁ P = 10) -- Radius of first circle
  (h₃ : dist O₂ P = 7)  -- Radius of second circle
  (h₄ : dist Q P = dist P R) -- QP = PR
  : (dist Q P)^2 = 10800/35 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_chord_length_l50_5011


namespace NUMINAMATH_CALUDE_triangle_problem_l50_5031

open Real

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  cos (A - π/3) = 2 * cos A →
  b = 2 →
  (1/2) * b * c * sin A = 3 * sqrt 3 →
  a^2 = b^2 + c^2 - 2*b*c * cos A →
  cos (2*C) = 1 - a^2 / (6 * b^2) →
  (a = 2 * sqrt 7 ∧ (B = π/12 ∨ B = 7*π/12)) := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l50_5031


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l50_5008

theorem tennis_tournament_matches (n : ℕ) (byes : ℕ) (wildcard : ℕ) : 
  n = 128 → byes = 36 → wildcard = 1 →
  ∃ (total_matches : ℕ), 
    total_matches = n - 1 + wildcard ∧ 
    total_matches = 128 ∧
    total_matches % 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l50_5008


namespace NUMINAMATH_CALUDE_three_squares_balance_l50_5075

/-- A balance system with three symbols: triangle, square, and circle. -/
structure BalanceSystem where
  triangle : ℚ
  square : ℚ
  circle : ℚ

/-- The balance rules for the system. -/
def balance_rules (s : BalanceSystem) : Prop :=
  5 * s.triangle + 2 * s.square = 21 * s.circle ∧
  2 * s.triangle = s.square + 3 * s.circle

/-- The theorem to prove. -/
theorem three_squares_balance (s : BalanceSystem) :
  balance_rules s → 3 * s.square = 9 * s.circle :=
by
  sorry

end NUMINAMATH_CALUDE_three_squares_balance_l50_5075


namespace NUMINAMATH_CALUDE_chocolate_bar_difference_l50_5080

theorem chocolate_bar_difference :
  let first_friend_portion : ℚ := 5 / 6
  let second_friend_portion : ℚ := 2 / 3
  first_friend_portion - second_friend_portion = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_difference_l50_5080


namespace NUMINAMATH_CALUDE_train_length_calculation_l50_5020

/-- Given a train that crosses a platform and a post, calculate its length. -/
theorem train_length_calculation (platform_length : ℝ) (platform_time : ℝ) (post_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : post_time = 18) :
  ∃ (train_length : ℝ), train_length = 300 ∧ 
    (train_length + platform_length) / platform_time = train_length / post_time := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l50_5020


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_relation_l50_5094

theorem smallest_integer_gcd_lcm_relation (m : ℕ) (h : m > 0) :
  (Nat.gcd 60 m * 20 = Nat.lcm 60 m) →
  (∀ k : ℕ, k > 0 ∧ k < m → Nat.gcd 60 k * 20 ≠ Nat.lcm 60 k) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_relation_l50_5094


namespace NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l50_5043

theorem decimal_equivalent_one_fourth_power_one : (1 / 4 : ℚ) ^ 1 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_one_fourth_power_one_l50_5043


namespace NUMINAMATH_CALUDE_factor_polynomial_l50_5085

theorem factor_polynomial (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l50_5085


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l50_5064

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ a b c : ℕ+, a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 52 := by
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l50_5064


namespace NUMINAMATH_CALUDE_quadratic_coefficient_determination_l50_5040

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_determination
  (a b c : ℝ)
  (f : ℝ → ℝ)
  (h_f : f = QuadraticFunction a b c)
  (h_point : f 0 = 3)
  (h_vertex : ∃ (k : ℝ), f 2 = -1 ∧ ∀ x, f x ≥ f 2) :
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_determination_l50_5040


namespace NUMINAMATH_CALUDE_water_needed_for_punch_l50_5006

/-- Represents the recipe ratios and calculates the required amount of water -/
def water_needed (lemon_juice : ℝ) : ℝ :=
  let sugar := 3 * lemon_juice
  let water := 3 * sugar
  water

/-- Proves that 36 cups of water are needed given the recipe ratios and 4 cups of lemon juice -/
theorem water_needed_for_punch : water_needed 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_punch_l50_5006


namespace NUMINAMATH_CALUDE_intersection_triangle_is_right_angled_l50_5051

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- Two points representing the foci -/
structure Foci where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point representing the intersection of the ellipse and hyperbola -/
def IntersectionPoint (e : Ellipse) (h : Hyperbola) := ℝ × ℝ

/-- Theorem stating that the triangle formed by the foci and intersection point is right-angled -/
theorem intersection_triangle_is_right_angled
  (e : Ellipse) (h : Hyperbola) (f : Foci) (P : IntersectionPoint e h)
  (h_same_foci : e.m - 1 = h.n + 1) :
  -- The triangle F₁PF₂ is right-angled
  ∃ (x y : ℝ), x^2 + y^2 = (f.F₁.1 - f.F₂.1)^2 + (f.F₁.2 - f.F₂.2)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_triangle_is_right_angled_l50_5051


namespace NUMINAMATH_CALUDE_kopek_payment_l50_5062

theorem kopek_payment (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_kopek_payment_l50_5062


namespace NUMINAMATH_CALUDE_computer_price_l50_5093

theorem computer_price (P : ℝ) 
  (h1 : 1.30 * P = 351)
  (h2 : 2 * P = 540) :
  P = 270 := by
sorry

end NUMINAMATH_CALUDE_computer_price_l50_5093


namespace NUMINAMATH_CALUDE_gunny_bag_fill_proof_l50_5003

/-- Conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2200

/-- Conversion factor from pounds to ounces -/
def pounds_to_ounces : ℝ := 16

/-- Conversion factor from grams to ounces -/
def grams_to_ounces : ℝ := 0.035274

/-- Capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℝ := 13.5

/-- Weight of a packet in pounds -/
def packet_weight_pounds : ℝ := 16

/-- Weight of a packet in additional ounces -/
def packet_weight_extra_ounces : ℝ := 4

/-- Weight of a packet in additional grams -/
def packet_weight_extra_grams : ℝ := 350

/-- The number of packets needed to fill the gunny bag -/
def packets_needed : ℕ := 1745

theorem gunny_bag_fill_proof : 
  ⌈(gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / 
   (packet_weight_pounds * pounds_to_ounces + packet_weight_extra_ounces + 
    packet_weight_extra_grams * grams_to_ounces)⌉ = packets_needed := by
  sorry

end NUMINAMATH_CALUDE_gunny_bag_fill_proof_l50_5003


namespace NUMINAMATH_CALUDE_park_area_l50_5060

/-- Given a rectangular park with length to breadth ratio of 1:2, where a cyclist completes one round along the boundary in 6 minutes at an average speed of 6 km/hr, prove that the area of the park is 20,000 square meters. -/
theorem park_area (length width : ℝ) (average_speed : ℝ) (time_taken : ℝ) : 
  length > 0 ∧ 
  width > 0 ∧ 
  length = (1/2) * width ∧ 
  average_speed = 6 ∧ 
  time_taken = 1/10 ∧ 
  2 * (length + width) = average_speed * time_taken * 1000 →
  length * width = 20000 := by sorry

end NUMINAMATH_CALUDE_park_area_l50_5060


namespace NUMINAMATH_CALUDE_greater_number_in_ratio_l50_5091

theorem greater_number_in_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a / b = 3 / 4 → a + b = 21 → max a b = 12 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_in_ratio_l50_5091


namespace NUMINAMATH_CALUDE_B_subset_A_l50_5078

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem B_subset_A : B ⊆ A := by sorry

end NUMINAMATH_CALUDE_B_subset_A_l50_5078


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l50_5081

theorem complex_expression_simplification :
  3 * (4 - 2 * Complex.I) - 2 * (2 * Complex.I - 3) = 18 - 10 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l50_5081


namespace NUMINAMATH_CALUDE_chess_players_per_game_l50_5024

theorem chess_players_per_game (total_players : Nat) (total_games : Nat) (players_per_game : Nat) : 
  total_players = 8 → 
  total_games = 28 → 
  (total_players.choose players_per_game) = total_games → 
  players_per_game = 2 := by
sorry

end NUMINAMATH_CALUDE_chess_players_per_game_l50_5024


namespace NUMINAMATH_CALUDE_isosceles_triangle_solution_l50_5095

-- Define the triangle properties
def isIsoscelesTriangle (x : ℝ) : Prop :=
  ∃ (side1 side2 side3 : ℝ),
    side1 = Real.tan x ∧ 
    side2 = Real.tan x ∧ 
    side3 = Real.tan (5 * x) ∧
    side1 = side2

-- Define the vertex angle condition
def hasVertexAngle4x (x : ℝ) : Prop :=
  ∃ (vertexAngle : ℝ),
    vertexAngle = 4 * x

-- Define the theorem
theorem isosceles_triangle_solution :
  ∀ x : ℝ,
    isIsoscelesTriangle x →
    hasVertexAngle4x x →
    0 < x →
    x < 90 →
    x = 20 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_solution_l50_5095


namespace NUMINAMATH_CALUDE_cookie_count_l50_5001

theorem cookie_count (bags : ℕ) (cookies_per_bag : ℕ) (h1 : bags = 37) (h2 : cookies_per_bag = 19) :
  bags * cookies_per_bag = 703 := by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l50_5001


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l50_5022

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 6*x + 36

-- Define the center and radius of the circle
def is_center_and_radius (a b r : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_center_radius_sum :
  ∃ (a b r : ℝ), is_center_and_radius a b r ∧ a + b + r = 5 + Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l50_5022


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l50_5090

theorem max_value_sqrt_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c))) ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l50_5090


namespace NUMINAMATH_CALUDE_square_side_length_l50_5065

theorem square_side_length (area : ℝ) (h : area = 9/16) :
  ∃ (side : ℝ), side > 0 ∧ side^2 = area ∧ side = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l50_5065


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l50_5098

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 165 → (n : ℝ) * interior_angle = (n - 2 : ℝ) * 180 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l50_5098


namespace NUMINAMATH_CALUDE_sum_product_ratio_l50_5063

theorem sum_product_ratio (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z) (h4 : x + y + z = 1) :
  (x * y + y * z + z * x) / (x^2 + y^2 + z^2) = (x * y + y * z + z * x) / (1 - 2 * (x * y + y * z + z * x)) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_ratio_l50_5063


namespace NUMINAMATH_CALUDE_ellipse_m_value_l50_5014

/-- Given an ellipse with equation x²/25 + y²/m² = 1 (m > 0) and left focus point at (-4, 0), 
    prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1) → 
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ (x + 5)^2/25 + y^2/m^2 < 1) → 
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l50_5014


namespace NUMINAMATH_CALUDE_tan_theta_for_pure_imaginary_l50_5026

theorem tan_theta_for_pure_imaginary (θ : Real) :
  let z : ℂ := Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5)
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_for_pure_imaginary_l50_5026


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l50_5015

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c ≥ Real.sqrt 161 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l50_5015


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l50_5012

/-- Calculates the gain percentage of a shopkeeper using a false weight --/
theorem shopkeeper_gain_percentage 
  (true_weight : ℝ) 
  (false_weight : ℝ) 
  (h1 : true_weight = 1000) 
  (h2 : false_weight = 980) : 
  (true_weight - false_weight) / true_weight * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l50_5012


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l50_5049

theorem quadratic_inequality_always_positive (r : ℝ) :
  (∀ x : ℝ, (r^2 - 1) * x^2 + 2 * (r - 1) * x + 1 > 0) ↔ r > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l50_5049


namespace NUMINAMATH_CALUDE_max_value_quadratic_l50_5066

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  ∃ (M : ℝ), M = 30 + 20*Real.sqrt 3 ∧ 
  ∀ (z w : ℝ), z > 0 → w > 0 → z^2 - 2*z*w + 3*w^2 = 10 → 
  z^2 + 2*z*w + 3*w^2 ≤ M := by
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l50_5066


namespace NUMINAMATH_CALUDE_city_mileage_problem_l50_5059

theorem city_mileage_problem (n : ℕ) : n * (n - 1) / 2 = 15 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_mileage_problem_l50_5059


namespace NUMINAMATH_CALUDE_max_integer_squared_inequality_l50_5061

theorem max_integer_squared_inequality : ∃ (n : ℕ),
  n = 30499 ∧ 
  n^2 ≤ 160 * 170 * 180 * 190 ∧
  ∀ (m : ℕ), m > n → m^2 > 160 * 170 * 180 * 190 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_squared_inequality_l50_5061


namespace NUMINAMATH_CALUDE_square_division_exists_l50_5019

/-- Represents a trapezoid with a given height -/
structure Trapezoid where
  height : ℝ

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ

/-- Represents a division of a square into trapezoids -/
structure SquareDivision where
  square : Square
  trapezoids : List Trapezoid

/-- Checks if a list of trapezoids has the required heights -/
def has_required_heights (trapezoids : List Trapezoid) : Prop :=
  trapezoids.length = 4 ∧
  (∃ (h₁ h₂ h₃ h₄ : Trapezoid),
    trapezoids = [h₁, h₂, h₃, h₄] ∧
    h₁.height = 1 ∧ h₂.height = 2 ∧ h₃.height = 3 ∧ h₄.height = 4)

/-- Checks if a square division is valid -/
def is_valid_division (div : SquareDivision) : Prop :=
  div.square.side_length = 4 ∧
  has_required_heights div.trapezoids

/-- Theorem: A square with side length 4 can be divided into four trapezoids with heights 1, 2, 3, and 4 -/
theorem square_division_exists : ∃ (div : SquareDivision), is_valid_division div := by
  sorry

end NUMINAMATH_CALUDE_square_division_exists_l50_5019


namespace NUMINAMATH_CALUDE_square_area_ratio_l50_5082

theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (3 * y)^2 / (9 * y)^2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l50_5082


namespace NUMINAMATH_CALUDE_max_roots_of_abs_sum_eq_abs_l50_5096

/-- A quadratic polynomial of the form ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given point x -/
def evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The number of roots of the equation |p₁(x)| + |p₂(x)| = |p₃(x)| -/
def numRoots (p₁ p₂ p₃ : QuadraticPolynomial) : ℕ :=
  sorry

/-- Theorem: The equation |p₁(x)| + |p₂(x)| = |p₃(x)| has at most 8 roots -/
theorem max_roots_of_abs_sum_eq_abs (p₁ p₂ p₃ : QuadraticPolynomial) :
  numRoots p₁ p₂ p₃ ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_roots_of_abs_sum_eq_abs_l50_5096


namespace NUMINAMATH_CALUDE_value_of_expression_l50_5018

theorem value_of_expression (a b : ℝ) (h : 2 * a - b = -1) : 
  2021 + 4 * a - 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l50_5018


namespace NUMINAMATH_CALUDE_cubes_form_name_l50_5000

/-- Represents a cube with letters on its faces -/
structure Cube where
  faces : Fin 6 → Char

/-- Represents the visible face of a cube -/
inductive VisibleFace
  | front
  | right

/-- Returns the letter on the visible face of a cube -/
def visibleLetter (c : Cube) (f : VisibleFace) : Char :=
  match f with
  | VisibleFace.front => c.faces 0
  | VisibleFace.right => c.faces 1

/-- Represents the arrangement of four cubes -/
structure CubeArrangement where
  cubes : Fin 4 → Cube
  visibleFaces : Fin 4 → VisibleFace

/-- The name formed by the visible letters in the cube arrangement -/
def formName (arr : CubeArrangement) : String :=
  String.mk (List.ofFn fun i => visibleLetter (arr.cubes i) (arr.visibleFaces i))

/-- The theorem stating that the given cube arrangement forms the name "Ника" -/
theorem cubes_form_name (arr : CubeArrangement) 
  (h1 : visibleLetter (arr.cubes 0) (arr.visibleFaces 0) = 'Н')
  (h2 : visibleLetter (arr.cubes 1) (arr.visibleFaces 1) = 'И')
  (h3 : visibleLetter (arr.cubes 2) (arr.visibleFaces 2) = 'К')
  (h4 : visibleLetter (arr.cubes 3) (arr.visibleFaces 3) = 'А') :
  formName arr = "Ника" := by
  sorry


end NUMINAMATH_CALUDE_cubes_form_name_l50_5000


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l50_5025

theorem algebraic_expression_value : ∀ x : ℝ, x^2 - 4*x = 5 → 2*x^2 - 8*x - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l50_5025


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l50_5009

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (h_geometric_mean : 4 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l50_5009


namespace NUMINAMATH_CALUDE_odd_sum_prob_is_five_thirteenths_l50_5035

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even sides on each die -/
def num_even_sides : ℕ := 3

/-- The number of odd sides on each die -/
def num_odd_sides : ℕ := 3

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_sides ^ num_dice

/-- The number of outcomes where all dice show odd numbers -/
def all_odd_outcomes : ℕ := num_odd_sides ^ num_dice

/-- The number of outcomes where the product of dice values is even -/
def even_product_outcomes : ℕ := total_outcomes - all_odd_outcomes

/-- The probability of rolling an odd sum given an even product -/
def prob_odd_sum_given_even_product : ℚ := 5 / 13

theorem odd_sum_prob_is_five_thirteenths :
  prob_odd_sum_given_even_product = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_odd_sum_prob_is_five_thirteenths_l50_5035


namespace NUMINAMATH_CALUDE_min_distance_is_zero_l50_5029

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := x^2 - 5*x + 4

-- Define the distance function between the two graphs
def distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_distance_is_zero :
  ∃ (x : ℝ), distance x = 0 ∧ ∀ (y : ℝ), distance y ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_min_distance_is_zero_l50_5029


namespace NUMINAMATH_CALUDE_pen_purchase_ratio_l50_5010

/-- The ratio of fountain pens to ballpoint pens in a purchase scenario --/
theorem pen_purchase_ratio (x y : ℕ) (h1 : (2 * x + y) * 3 = 3 * (2 * y + x)) :
  y = 4 * x := by
  sorry

#check pen_purchase_ratio

end NUMINAMATH_CALUDE_pen_purchase_ratio_l50_5010


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l50_5097

-- Define sets A and B
def A : Set ℝ := {x | |x + 1| < 4}
def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - 2*a) < 0}

-- Theorem statement
theorem intersection_equality_implies_a_range (a : ℝ) :
  A ∩ B a = B a → a ∈ Set.Icc (-2.5) 1.5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_range_l50_5097


namespace NUMINAMATH_CALUDE_intersection_point_correct_l50_5013

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -1

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 4

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 13 / 10

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 29 / 10

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct : 
  (m₁ * x_intersect + b₁ = y_intersect) ∧ 
  (m₂ * (x_intersect - x₀) = y_intersect - y₀) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l50_5013


namespace NUMINAMATH_CALUDE_trigonometric_identity_l50_5048

theorem trigonometric_identity : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let cos60 : ℝ := 1/2
  2 * sin30 - cos45^2 + cos60 = 1 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l50_5048


namespace NUMINAMATH_CALUDE_factorial_product_simplification_l50_5039

theorem factorial_product_simplification (a : ℝ) :
  (1 * 1) * (2 * 1 * a) * (3 * 2 * 1 * a^3) * (4 * 3 * 2 * 1 * a^6) * (5 * 4 * 3 * 2 * 1 * a^10) = 34560 * a^20 := by
  sorry

end NUMINAMATH_CALUDE_factorial_product_simplification_l50_5039


namespace NUMINAMATH_CALUDE_quadratic_factorization_l50_5083

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x, x^2 - 18*x + 72 = (x - a)*(x - b)) : 
  2*b - a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l50_5083


namespace NUMINAMATH_CALUDE_subset_divisibility_subset_1000_500_divisibility_l50_5052

theorem subset_divisibility (n : ℕ) (k : ℕ) (p : ℕ) : Prop :=
  p ∣ Nat.choose n k

theorem subset_1000_500_divisibility :
  subset_divisibility 1000 500 3 ∧
  subset_divisibility 1000 500 5 ∧
  ¬(subset_divisibility 1000 500 11) ∧
  subset_divisibility 1000 500 13 ∧
  subset_divisibility 1000 500 17 :=
by sorry

end NUMINAMATH_CALUDE_subset_divisibility_subset_1000_500_divisibility_l50_5052


namespace NUMINAMATH_CALUDE_balls_satisfy_conditions_l50_5071

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sporting event -/
inductive SportingEvent
| FigureSkating
| RhythmicGymnastics
| Other

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  calculationMethod : String
  yearIntroduced : Nat

/-- Checks if a word sounds similar to a festive dance event -/
def soundsSimilarTo (w : RussianWord) (e : FestiveDanceEvent) : Prop :=
  sorry

/-- Checks if a word is used in a sporting event -/
def usedInSportingEvent (w : RussianWord) (e : SportingEvent) : Prop :=
  sorry

/-- Checks if a word is used in the Russian pension system -/
def usedInPensionSystem (w : RussianWord) (p : RussianPensionSystem) : Prop :=
  sorry

/-- The main theorem stating that "баллы" satisfies all conditions -/
theorem balls_satisfy_conditions :
  ∃ (w : RussianWord) (e : FestiveDanceEvent) (p : RussianPensionSystem),
    w.word = "баллы" ∧
    soundsSimilarTo w e ∧
    usedInSportingEvent w SportingEvent.FigureSkating ∧
    usedInSportingEvent w SportingEvent.RhythmicGymnastics ∧
    usedInPensionSystem w p ∧
    p.yearIntroduced = 2015 :=
  sorry


end NUMINAMATH_CALUDE_balls_satisfy_conditions_l50_5071


namespace NUMINAMATH_CALUDE_compare_star_operations_l50_5032

-- Define the new operation
def star (a b : ℤ) : ℚ := (a * b : ℚ) - (a : ℚ) / (b : ℚ)

-- Theorem statement
theorem compare_star_operations : star 6 (-3) < star 4 (-4) := by
  sorry

end NUMINAMATH_CALUDE_compare_star_operations_l50_5032


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l50_5069

/-- Proves that the walking speed is 4 km/hr given the problem conditions -/
theorem walking_speed_calculation (total_distance : ℝ) (total_time : ℝ) (running_speed : ℝ) :
  total_distance = 8 →
  total_time = 1.5 →
  running_speed = 8 →
  ∃ (walking_speed : ℝ),
    walking_speed > 0 ∧
    (total_distance / 2) / walking_speed + (total_distance / 2) / running_speed = total_time ∧
    walking_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l50_5069


namespace NUMINAMATH_CALUDE_parallelogram_area_l50_5074

/-- The area of a parallelogram with vertices at (1, 1), (7, 1), (4, 9), and (10, 9) is 48 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices
  let v1 : ℝ × ℝ := (1, 1)
  let v2 : ℝ × ℝ := (7, 1)
  let v3 : ℝ × ℝ := (4, 9)
  let v4 : ℝ × ℝ := (10, 9)

  -- Define the parallelogram
  let parallelogram := [v1, v2, v3, v4]

  -- Calculate the area
  let area := 48

  -- Prove that the area of the parallelogram is 48 square units
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l50_5074


namespace NUMINAMATH_CALUDE_seven_lines_angle_l50_5057

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a function to check if two lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Define a function to measure the angle between two lines
def angle_between (l1 l2 : Line) : ℝ := sorry

-- The main theorem
theorem seven_lines_angle (lines : Fin 7 → Line) :
  (∀ i j, i ≠ j → ¬ parallel (lines i) (lines j)) →
  ∃ i j, i ≠ j ∧ angle_between (lines i) (lines j) < 26 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_seven_lines_angle_l50_5057


namespace NUMINAMATH_CALUDE_brandon_skittles_count_l50_5077

/-- Given Brandon's initial Skittles count and the number of Skittles he loses,
    prove that his final Skittles count is the difference between the initial count and the number lost. -/
theorem brandon_skittles_count (initial_count lost_count : ℕ) :
  initial_count - lost_count = initial_count - lost_count :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_count_l50_5077


namespace NUMINAMATH_CALUDE_point_location_l50_5092

theorem point_location (m n : ℝ) : 2^m + 2^n < 2 * Real.sqrt 2 → m + n < 1 := by
  sorry

end NUMINAMATH_CALUDE_point_location_l50_5092


namespace NUMINAMATH_CALUDE_max_value_of_f_l50_5030

-- Define the function f
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l50_5030
