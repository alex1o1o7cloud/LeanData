import Mathlib

namespace distinct_collections_biology_l1582_158208

def biology : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

def num_o : ℕ := (biology.filter (· = 'O')).card

theorem distinct_collections_biology :
  let total_selections := (Finset.powerset biology).filter (λ s => 
    (s.filter (λ c => c ∈ vowels)).card = 3 ∧ 
    (s.filter (λ c => c ∈ consonants)).card = 2)
  (Finset.powerset total_selections).card = 18 :=
sorry

end distinct_collections_biology_l1582_158208


namespace f_composite_value_l1582_158245

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else a^x + b

theorem f_composite_value (a b : ℝ) :
  f 0 a b = 2 →
  f (-1) a b = 3 →
  f (f (-3) a b) a b = 2 := by
sorry

end f_composite_value_l1582_158245


namespace f_non_monotonic_iff_l1582_158204

/-- A piecewise function f(x) depending on parameters a and t -/
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

/-- The theorem stating the condition for f to be non-monotonic for all t -/
theorem f_non_monotonic_iff (a : ℝ) :
  (∀ t : ℝ, ¬Monotone (f a t)) ↔ a ≤ 3/4 := by sorry

end f_non_monotonic_iff_l1582_158204


namespace alternating_sum_equals_three_to_seven_l1582_158256

theorem alternating_sum_equals_three_to_seven (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ + a₆ - a₇ = 3^7 := by
sorry

end alternating_sum_equals_three_to_seven_l1582_158256


namespace two_students_know_same_number_l1582_158254

/-- Represents the number of students a given student knows -/
def StudentsKnown := Fin 81

/-- The set of all students in the course -/
def Students := Fin 81

theorem two_students_know_same_number (f : Students → StudentsKnown) :
  ∃ (i j : Students), i ≠ j ∧ f i = f j :=
sorry

end two_students_know_same_number_l1582_158254


namespace modular_arithmetic_problem_l1582_158214

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), 
    (7 * a) % 60 = 1 ∧ 
    (13 * b) % 60 = 1 ∧ 
    ((3 * a + 6 * b) % 60) = 51 := by
  sorry

end modular_arithmetic_problem_l1582_158214


namespace peach_to_apricot_ratio_l1582_158274

/-- Given a total number of trees and a number of apricot trees, 
    calculate the ratio of peach trees to apricot trees. -/
def tree_ratio (total : ℕ) (apricot : ℕ) : ℚ × ℚ :=
  let peach := total - apricot
  (peach, apricot)

/-- The theorem states that for 232 total trees and 58 apricot trees,
    the ratio of peach trees to apricot trees is 3:1. -/
theorem peach_to_apricot_ratio :
  tree_ratio 232 58 = (3, 1) := by sorry

end peach_to_apricot_ratio_l1582_158274


namespace max_qed_value_l1582_158262

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number -/
def ThreeDigitNumber := Fin 1000

theorem max_qed_value 
  (D E L M Q : Digit) 
  (h_distinct : D ≠ E ∧ D ≠ L ∧ D ≠ M ∧ D ≠ Q ∧ 
                E ≠ L ∧ E ≠ M ∧ E ≠ Q ∧ 
                L ≠ M ∧ L ≠ Q ∧ 
                M ≠ Q)
  (h_equation : 91 * E.val + 10 * L.val + 101 * M.val = 100 * Q.val + D.val) :
  (∀ (D' E' Q' : Digit), 
    D' ≠ E' ∧ D' ≠ Q' ∧ E' ≠ Q' → 
    100 * Q'.val + 10 * E'.val + D'.val ≤ 893) :=
sorry

end max_qed_value_l1582_158262


namespace ring_toss_earnings_l1582_158273

/-- The number of days a ring toss game earned money, given total earnings and daily earnings. -/
def days_earned (total_earnings daily_earnings : ℕ) : ℕ :=
  total_earnings / daily_earnings

/-- Theorem stating that the ring toss game earned money for 5 days. -/
theorem ring_toss_earnings : days_earned 165 33 = 5 := by
  sorry

end ring_toss_earnings_l1582_158273


namespace medicine_price_reduction_l1582_158287

/-- Represents the average percentage decrease in price per reduction -/
def average_decrease : ℝ := 0.25

/-- The original price of the medicine in yuan -/
def original_price : ℝ := 16

/-- The current price of the medicine in yuan -/
def current_price : ℝ := 9

/-- The number of successive price reductions -/
def num_reductions : ℕ := 2

theorem medicine_price_reduction :
  current_price = original_price * (1 - average_decrease) ^ num_reductions :=
by sorry

end medicine_price_reduction_l1582_158287


namespace largest_package_size_l1582_158295

theorem largest_package_size (john_markers alice_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alice_markers = 60) : 
  Nat.gcd john_markers alice_markers = 12 := by
  sorry

end largest_package_size_l1582_158295


namespace smaller_rss_better_fit_regression_line_passes_through_center_l1582_158286

/-- Represents a linear regression model -/
structure LinearRegression where
  x : List ℝ  -- Independent variable data
  y : List ℝ  -- Dependent variable data
  β : ℝ       -- Slope of the regression line
  α : ℝ       -- Intercept of the regression line

/-- Calculates the residual sum of squares for a linear regression model -/
def residualSumOfSquares (model : LinearRegression) : ℝ :=
  sorry

/-- Calculates the mean of a list of real numbers -/
def mean (data : List ℝ) : ℝ :=
  sorry

/-- Theorem stating that a smaller residual sum of squares indicates a better fitting effect -/
theorem smaller_rss_better_fit (model1 model2 : LinearRegression) :
  residualSumOfSquares model1 < residualSumOfSquares model2 →
  -- The fitting effect of model1 is better than model2
  sorry :=
sorry

/-- Theorem stating that the linear regression equation passes through the center point (x̄, ȳ) of the sample -/
theorem regression_line_passes_through_center (model : LinearRegression) :
  let x_mean := mean model.x
  let y_mean := mean model.y
  model.α + model.β * x_mean = y_mean :=
sorry

end smaller_rss_better_fit_regression_line_passes_through_center_l1582_158286


namespace money_distribution_l1582_158220

theorem money_distribution (a b c : ℕ) (total : ℕ) : 
  a + b + c = 9 → 
  b = 3 → 
  900 * b = 2700 → 
  900 * (a + b + c) = 2700 * 3 :=
by sorry

end money_distribution_l1582_158220


namespace simplify_expression_l1582_158213

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b) - 2 * b^2 = 9 * b^3 + 4 * b^2 := by
  sorry

end simplify_expression_l1582_158213


namespace pyramid_edge_length_l1582_158227

/-- A pyramid with 8 edges of equal length -/
structure Pyramid where
  edge_count : ℕ
  edge_length : ℝ
  total_length : ℝ
  edge_count_eq : edge_count = 8
  total_eq : total_length = edge_count * edge_length

/-- Theorem: If a pyramid has 8 edges of equal length, and the sum of all edges is 14.8 meters,
    then the length of each edge is 1.85 meters. -/
theorem pyramid_edge_length (p : Pyramid) (h : p.total_length = 14.8) :
  p.edge_length = 1.85 := by
  sorry

end pyramid_edge_length_l1582_158227


namespace line_no_intersection_slope_range_l1582_158271

/-- Given points A(-2,3) and B(3,2), and a line l: y = kx - 2, 
    if l has no intersection with line segment AB, 
    then the slope k of line l is in the range (-5/2, 4/3). -/
theorem line_no_intersection_slope_range (k : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (3, 2)
  let l (x : ℝ) := k * x - 2
  (∀ x y, (x, y) ∈ Set.Icc A B → y ≠ l x) →
  k ∈ Set.Ioo (-5/2 : ℝ) (4/3 : ℝ) := by
sorry

end line_no_intersection_slope_range_l1582_158271


namespace exists_prescription_with_four_potent_l1582_158283

/-- Represents a type of medicine -/
structure Medicine :=
  (isPotent : Bool)

/-- Represents a prescription -/
structure Prescription :=
  (medicines : Finset Medicine)

/-- The set of all available medicines -/
def AllMedicines : Finset Medicine := sorry

/-- The set of all prescriptions -/
def AllPrescriptions : Finset Prescription := sorry

/-- Conditions of the problem -/
axiom total_prescriptions : Finset.card AllPrescriptions = 68

axiom medicines_per_prescription : 
  ∀ p : Prescription, p ∈ AllPrescriptions → Finset.card p.medicines = 5

axiom at_least_one_potent : 
  ∀ p : Prescription, p ∈ AllPrescriptions → 
    ∃ m : Medicine, m ∈ p.medicines ∧ m.isPotent

axiom three_medicines_in_one_prescription : 
  ∀ m₁ m₂ m₃ : Medicine, m₁ ∈ AllMedicines → m₂ ∈ AllMedicines → m₃ ∈ AllMedicines →
    m₁ ≠ m₂ → m₂ ≠ m₃ → m₁ ≠ m₃ →
    ∃! p : Prescription, p ∈ AllPrescriptions ∧ m₁ ∈ p.medicines ∧ m₂ ∈ p.medicines ∧ m₃ ∈ p.medicines

/-- The main theorem to prove -/
theorem exists_prescription_with_four_potent : 
  ∃ p : Prescription, p ∈ AllPrescriptions ∧ 
    (Finset.filter (fun m => m.isPotent) p.medicines).card ≥ 4 := by
  sorry

end exists_prescription_with_four_potent_l1582_158283


namespace polynomial_product_equality_l1582_158226

theorem polynomial_product_equality (x y z : ℝ) :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2) =
  27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by
sorry

end polynomial_product_equality_l1582_158226


namespace perpendicular_lines_slope_l1582_158279

/-- 
Given two lines in the xy-plane:
- Line 1 with equation y = mx + 1
- Line 2 with equation y = 4x - 8
If Line 1 is perpendicular to Line 2, then m = -1/4
-/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 1) →  -- Line 1 exists
  (∃ (x y : ℝ), y = 4 * x - 8) →  -- Line 2 exists
  (∀ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m * x₁ + 1 → y₂ = 4 * x₂ - 8 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (x₂ - x₁)) →  -- Lines are perpendicular
  m = -1/4 :=
by sorry

end perpendicular_lines_slope_l1582_158279


namespace rotation_theorem_l1582_158288

/-- Represents a square board with side length 2^n -/
structure Board (n : Nat) where
  size : Nat := 2^n
  elements : Fin (size * size) → Nat

/-- Represents the state of the board after rotations -/
def rotatedBoard (n : Nat) : Board n → Board n :=
  sorry

/-- The main diagonal of a board -/
def mainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- The other main diagonal (bottom-left to top-right) of a board -/
def otherMainDiagonal (n : Nat) (b : Board n) : List Nat :=
  sorry

/-- Initial board setup -/
def initialBoard : Board 5 :=
  { elements := λ i => i.val + 1 }

theorem rotation_theorem :
  mainDiagonal 5 (rotatedBoard 5 initialBoard) =
    (otherMainDiagonal 5 initialBoard).reverse := by
  sorry

end rotation_theorem_l1582_158288


namespace sundae_cost_l1582_158209

theorem sundae_cost (cherry_jubilee : ℝ) (peanut_butter : ℝ) (royal_banana : ℝ) 
  (tip_percentage : ℝ) (final_bill : ℝ) :
  cherry_jubilee = 9 →
  peanut_butter = 7.5 →
  royal_banana = 10 →
  tip_percentage = 0.2 →
  final_bill = 42 →
  ∃ (death_by_chocolate : ℝ),
    death_by_chocolate = 8.5 ∧
    (cherry_jubilee + peanut_butter + royal_banana + death_by_chocolate) * (1 + tip_percentage) = final_bill :=
by sorry

end sundae_cost_l1582_158209


namespace magpie_call_not_correlation_l1582_158255

/-- Represents a statement that may or may not indicate a correlation. -/
inductive Statement
| A : Statement  -- A timely snow promises a good harvest
| B : Statement  -- A good teacher produces outstanding students
| C : Statement  -- Smoking is harmful to health
| D : Statement  -- The magpie's call is a sign of happiness

/-- Predicate to determine if a statement represents a correlation. -/
def is_correlation (s : Statement) : Prop :=
  match s with
  | Statement.A => True
  | Statement.B => True
  | Statement.C => True
  | Statement.D => False

/-- Theorem stating that Statement D does not represent a correlation. -/
theorem magpie_call_not_correlation :
  ¬ (is_correlation Statement.D) :=
sorry

end magpie_call_not_correlation_l1582_158255


namespace power_of_five_l1582_158251

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^4 * 625^3 → m = 21 := by
  sorry

end power_of_five_l1582_158251


namespace q_squared_minus_one_div_fifteen_l1582_158281

/-- The largest prime with 2011 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2011 digits -/
axiom q_digits : ∃ (n : ℕ), 10^2010 ≤ q ∧ q < 10^2011

/-- q is the largest prime with 2011 digits -/
axiom q_largest : ∀ (p : ℕ), Nat.Prime p → (∃ (n : ℕ), 10^2010 ≤ p ∧ p < 10^2011) → p ≤ q

theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end q_squared_minus_one_div_fifteen_l1582_158281


namespace horizontal_distance_on_line_l1582_158269

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem horizontal_distance_on_line (m n p : ℝ) : 
  (m = n / 7 - 2 / 5) → 
  (m + p = (n + 21) / 7 - 2 / 5) → 
  p = 3 := by
sorry

end horizontal_distance_on_line_l1582_158269


namespace marble_drawing_probability_l1582_158250

/-- The probability of drawing marbles consecutively by color --/
theorem marble_drawing_probability : 
  let total_marbles : ℕ := 12
  let blue_marbles : ℕ := 4
  let orange_marbles : ℕ := 3
  let green_marbles : ℕ := 5
  let favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial blue_marbles * 
                                 Nat.factorial orange_marbles * Nat.factorial green_marbles
  let total_outcomes : ℕ := Nat.factorial total_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4620 := by
  sorry

end marble_drawing_probability_l1582_158250


namespace robot_models_properties_l1582_158261

/-- Represents the cost and quantity information for robot models --/
structure RobotModels where
  cost_A : ℕ  -- Cost of model A in yuan
  cost_B : ℕ  -- Cost of model B in yuan
  total_A : ℕ  -- Total spent on model A in yuan
  total_B : ℕ  -- Total spent on model B in yuan
  total_units : ℕ  -- Total units to be purchased

/-- Calculates the maximum number of model A units that can be purchased --/
def max_model_A (r : RobotModels) : ℕ :=
  min ((2 * r.total_units) / 3) r.total_units

/-- Theorem stating the properties of the robot models --/
theorem robot_models_properties (r : RobotModels) 
  (h1 : r.cost_B = 2 * r.cost_A - 400)
  (h2 : r.total_A = 96000)
  (h3 : r.total_B = 168000)
  (h4 : r.total_units = 100) :
  r.cost_A = 1600 ∧ r.cost_B = 2800 ∧ max_model_A r = 66 := by
  sorry

#eval max_model_A ⟨1600, 2800, 96000, 168000, 100⟩

end robot_models_properties_l1582_158261


namespace probability_at_least_one_woman_l1582_158267

def total_group_size : ℕ := 15
def men_count : ℕ := 9
def women_count : ℕ := 6
def selection_size : ℕ := 4

theorem probability_at_least_one_woman :
  let total_combinations := Nat.choose total_group_size selection_size
  let all_men_combinations := Nat.choose men_count selection_size
  (total_combinations - all_men_combinations : ℚ) / total_combinations = 137 / 151 := by
  sorry

end probability_at_least_one_woman_l1582_158267


namespace unique_triple_solution_l1582_158221

theorem unique_triple_solution : 
  ∃! (x y z : ℝ), x + y = 2 ∧ x * y - z^2 = 1 :=
by sorry

end unique_triple_solution_l1582_158221


namespace minutes_in_year_scientific_notation_l1582_158232

/-- The number of days in a year -/
def days_in_year : ℕ := 360

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Converts a natural number to a real number -/
def to_real (n : ℕ) : ℝ := n

/-- Rounds a real number to three significant figures -/
noncomputable def round_to_three_sig_figs (x : ℝ) : ℝ := 
  sorry

/-- The main theorem stating that the number of minutes in a year,
    when expressed in scientific notation with three significant figures,
    is equal to 5.18 × 10^5 -/
theorem minutes_in_year_scientific_notation :
  round_to_three_sig_figs (to_real (days_in_year * hours_in_day * minutes_in_hour)) = 5.18 * 10^5 := by
  sorry

end minutes_in_year_scientific_notation_l1582_158232


namespace systematic_sampling_last_id_l1582_158265

theorem systematic_sampling_last_id 
  (total_students : Nat) 
  (sample_size : Nat) 
  (first_id : Nat) 
  (h1 : total_students = 2000) 
  (h2 : sample_size = 50) 
  (h3 : first_id = 3) :
  let interval := total_students / sample_size
  let last_id := first_id + interval * (sample_size - 1)
  last_id = 1963 := by
sorry

end systematic_sampling_last_id_l1582_158265


namespace geometric_sequence_problem_l1582_158289

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) 
  (h_seq : is_positive_geometric_sequence a)
  (h_1 : a 1 * a 2 * a 3 = 4)
  (h_2 : a 4 * a 5 * a 6 = 12)
  (h_3 : a (n-1) * a n * a (n+1) = 324) :
  n = 14 := by
  sorry

end geometric_sequence_problem_l1582_158289


namespace stock_market_value_l1582_158216

theorem stock_market_value 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_yield : ℝ) 
  (h1 : dividend_rate = 0.07) 
  (h2 : market_yield = 0.10) : 
  (dividend_rate * face_value) / market_yield = 0.7 * face_value := by
  sorry

end stock_market_value_l1582_158216


namespace equation_represents_two_lines_l1582_158215

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop :=
  3 * x^2 - 36 * y^2 - 18 * x + 27 = 0

/-- The two lines represented by the equation -/
def line1 (x y : ℝ) : Prop :=
  x = 3 + 2 * Real.sqrt 3 * y

def line2 (x y : ℝ) : Prop :=
  x = 3 - 2 * Real.sqrt 3 * y

/-- Theorem stating that the equation represents two lines -/
theorem equation_represents_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
by sorry

end equation_represents_two_lines_l1582_158215


namespace circles_intersection_l1582_158238

def circle1_center : ℝ × ℝ := (0, 0)
def circle2_center : ℝ × ℝ := (-3, 4)
def circle2_radius : ℝ := 2

theorem circles_intersection (m : ℝ) :
  (∃ (x y : ℝ), (x - circle1_center.1)^2 + (y - circle1_center.2)^2 = m ∧
                (x - circle2_center.1)^2 + (y - circle2_center.2)^2 = circle2_radius^2) ↔
  9 < m ∧ m < 49 := by
  sorry

end circles_intersection_l1582_158238


namespace inequality_implication_l1582_158229

theorem inequality_implication (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 := by
  sorry

end inequality_implication_l1582_158229


namespace parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l1582_158253

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define the properties
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def verticalAngles (a1 a2 : Angle) : Prop := sorry
def rightTriangle (t : Triangle) : Prop := sorry
def acuteAngles (t : Triangle) (a1 a2 : Angle) : Prop := sorry
def complementaryAngles (a1 a2 : Angle) : Prop := sorry
def supplementaryAngles (a1 a2 : Angle) : Prop := sorry

-- Theorem A
theorem parallel_implies_alternate_interior_angles (l1 l2 : Line) (a1 a2 : Angle) :
  parallel l1 l2 → alternateInteriorAngles a1 a2 l1 l2 := sorry

-- Theorem B
theorem vertical_angles_are_equal (a1 a2 : Angle) :
  verticalAngles a1 a2 → a1 = a2 := sorry

-- Theorem C
theorem right_triangle_acute_angles_complementary (t : Triangle) (a1 a2 : Angle) :
  rightTriangle t → acuteAngles t a1 a2 → complementaryAngles a1 a2 := sorry

-- Theorem D
theorem supplements_of_same_angle_are_equal (a1 a2 a3 : Angle) :
  supplementaryAngles a1 a3 → supplementaryAngles a2 a3 → a1 = a2 := sorry

-- The main theorem: inverse of B is false, while inverses of A, C, and D are true
theorem inverse_of_vertical_angles_false_others_true :
  (∃ a1 a2 : Angle, a1 = a2 ∧ ¬verticalAngles a1 a2) ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) ∧
  (∀ a1 a2 : Angle, complementaryAngles a1 a2 → ∃ t : Triangle, rightTriangle t ∧ acuteAngles t a1 a2) ∧
  (∀ a1 a2 a3 : Angle, a1 = a2 → supplementaryAngles a1 a3 → supplementaryAngles a2 a3) := sorry

end parallel_implies_alternate_interior_angles_vertical_angles_are_equal_right_triangle_acute_angles_complementary_supplements_of_same_angle_are_equal_inverse_of_vertical_angles_false_others_true_l1582_158253


namespace andrew_toast_count_l1582_158217

/-- The cost of breakfast for Dale and Andrew -/
def total_cost : ℕ := 15

/-- The cost of a slice of toast -/
def toast_cost : ℕ := 1

/-- The cost of an egg -/
def egg_cost : ℕ := 3

/-- The number of slices of toast Dale had -/
def dale_toast : ℕ := 2

/-- The number of eggs Dale had -/
def dale_eggs : ℕ := 2

/-- The number of eggs Andrew had -/
def andrew_eggs : ℕ := 2

/-- The number of slices of toast Andrew had -/
def andrew_toast : ℕ := 1

theorem andrew_toast_count :
  total_cost = 
    dale_toast * toast_cost + dale_eggs * egg_cost + 
    andrew_toast * toast_cost + andrew_eggs * egg_cost :=
by sorry

end andrew_toast_count_l1582_158217


namespace donation_sum_l1582_158206

theorem donation_sum : 
  let donation1 : ℝ := 245.00
  let donation2 : ℝ := 225.00
  let donation3 : ℝ := 230.00
  donation1 + donation2 + donation3 = 700.00 := by
  sorry

end donation_sum_l1582_158206


namespace circle_to_octagon_area_ratio_l1582_158211

/-- The ratio of the area of a circle inscribed in a regular octagon
    (where the circle's radius equals the octagon's apothem)
    to the area of the octagon itself. -/
theorem circle_to_octagon_area_ratio : ∃ (a b : ℕ), (a : ℝ).sqrt / b * π = (π / (4 * Real.sqrt 2)) := by
  sorry

end circle_to_octagon_area_ratio_l1582_158211


namespace third_student_number_l1582_158212

theorem third_student_number (A B C D : ℤ) 
  (sum_eq : A + B + C + D = 531)
  (diff_eq : A + B = C + D + 31)
  (third_fourth_diff : C = D + 22) :
  C = 136 := by
  sorry

end third_student_number_l1582_158212


namespace sonita_stamp_purchase_l1582_158292

theorem sonita_stamp_purchase (two_q_stamps : ℕ) 
  (h1 : two_q_stamps > 0)
  (h2 : two_q_stamps < 9)
  (h3 : two_q_stamps % 5 = 0) :
  2 * two_q_stamps + 10 * two_q_stamps + (100 - 12 * two_q_stamps) / 5 = 63 := by
  sorry

#check sonita_stamp_purchase

end sonita_stamp_purchase_l1582_158292


namespace figure_tiling_iff_multiple_of_three_l1582_158272

/-- Represents a figure Φ consisting of three n×n squares. -/
structure Figure (n : ℕ) where
  squares : Fin 3 → Fin n → Fin n → Bool

/-- Represents a 1×3 or 3×1 tile. -/
inductive Tile
  | horizontal : Tile
  | vertical : Tile

/-- A tiling of the figure Φ using 1×3 and 3×1 tiles. -/
def Tiling (n : ℕ) := Set (ℕ × ℕ × Tile)

/-- Predicate to check if a tiling is valid for the given figure. -/
def isValidTiling (n : ℕ) (φ : Figure n) (t : Tiling n) : Prop := sorry

/-- The main theorem stating that a valid tiling exists if and only if n is a multiple of 3. -/
theorem figure_tiling_iff_multiple_of_three (n : ℕ) (φ : Figure n) :
  (n > 1) → (∃ t : Tiling n, isValidTiling n φ t) ↔ ∃ k : ℕ, n = 3 * k :=
sorry

end figure_tiling_iff_multiple_of_three_l1582_158272


namespace quadrilateral_area_is_4014_l1582_158299

/-- The area of a quadrilateral with vertices at (1, 1), (1, 5), (3, 5), and (2006, 2003) -/
def quadrilateralArea : ℝ :=
  let A := (1, 1)
  let B := (1, 5)
  let C := (3, 5)
  let D := (2006, 2003)
  -- Area calculation goes here
  0 -- Placeholder

/-- Theorem stating that the area of the quadrilateral is 4014 square units -/
theorem quadrilateral_area_is_4014 : quadrilateralArea = 4014 := by
  sorry

end quadrilateral_area_is_4014_l1582_158299


namespace johns_donation_l1582_158239

/-- Given 10 initial contributions, if a new donation causes the average
    contribution to increase by 80% to $90, then the new donation must be $490. -/
theorem johns_donation (initial_count : ℕ) (increase_percentage : ℚ) (new_average : ℚ) :
  initial_count = 10 →
  increase_percentage = 80 / 100 →
  new_average = 90 →
  let initial_average := new_average / (1 + increase_percentage)
  let initial_total := initial_count * initial_average
  let new_total := (initial_count + 1) * new_average
  new_total - initial_total = 490 := by
sorry

end johns_donation_l1582_158239


namespace four_Z_three_equals_127_l1582_158297

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a*b^2 + 3*a^2*b + b^3

-- Theorem statement
theorem four_Z_three_equals_127 : Z 4 3 = 127 := by
  sorry

end four_Z_three_equals_127_l1582_158297


namespace union_equality_implies_m_values_l1582_158233

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_values (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 :=
by sorry

end union_equality_implies_m_values_l1582_158233


namespace union_subset_intersection_implies_a_equals_one_l1582_158258

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

-- State the theorem
theorem union_subset_intersection_implies_a_equals_one (a : ℝ) :
  (A ∪ B a) ⊆ (A ∩ B a) → a = 1 := by
  sorry

end union_subset_intersection_implies_a_equals_one_l1582_158258


namespace distribute_balls_eq_partitions_six_balls_four_boxes_l1582_158218

/-- Number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing n indistinguishable balls into k indistinguishable boxes
    is equivalent to finding partitions of n into at most k parts -/
theorem distribute_balls_eq_partitions (n k : ℕ) :
  distribute_balls n k = partitions n k := by sorry

/-- The specific case for 6 balls and 4 boxes -/
theorem six_balls_four_boxes :
  distribute_balls 6 4 = 9 := by sorry

end distribute_balls_eq_partitions_six_balls_four_boxes_l1582_158218


namespace parabola_circle_tangent_to_yaxis_l1582_158285

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y² = 4x -/
def focus : Point := ⟨1, 0⟩

/-- A circle with center c and radius r -/
structure Circle where
  center : Point
  radius : ℝ

/-- The y-axis -/
def yAxis := {p : Point | p.x = 0}

/-- Predicate to check if a circle is tangent to the y-axis -/
def isTangentToYAxis (c : Circle) : Prop :=
  c.center.x = c.radius

/-- Theorem: For any point P on the parabola y² = 4x, 
    the circle with diameter PF (where F is the focus) 
    is tangent to the y-axis -/
theorem parabola_circle_tangent_to_yaxis 
  (P : Point) (h : P ∈ Parabola) : 
  ∃ (c : Circle), c.center = ⟨(P.x + focus.x) / 2, P.y / 2⟩ ∧ 
                  c.radius = (P.x + focus.x) / 2 ∧
                  isTangentToYAxis c :=
sorry

end parabola_circle_tangent_to_yaxis_l1582_158285


namespace max_integer_difference_l1582_158294

theorem max_integer_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) :
  (∀ (a b : ℤ), 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 → y - x ≥ b - a) ∧ y - x ≤ 5 :=
sorry

end max_integer_difference_l1582_158294


namespace positive_number_problem_l1582_158243

theorem positive_number_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x - 4 = 21 / x)
  (eq2 : x + y^2 = 45)
  (eq3 : y * z = x^3) :
  x = 7 ∧ y = Real.sqrt 38 ∧ z = 343 * Real.sqrt 38 / 38 := by
  sorry

end positive_number_problem_l1582_158243


namespace sqrt_of_sqrt_four_equals_sqrt_two_l1582_158234

theorem sqrt_of_sqrt_four_equals_sqrt_two : Real.sqrt (Real.sqrt 4) = Real.sqrt 2 := by
  sorry

end sqrt_of_sqrt_four_equals_sqrt_two_l1582_158234


namespace expression_equality_l1582_158257

theorem expression_equality : 6 * 1000 + 5 * 100 + 6 * 1 = 6506 := by
  sorry

end expression_equality_l1582_158257


namespace range_of_r_l1582_158242

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^4 + 6*x^2 + 9 - 2*x

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, y ≥ 9 ↔ ∃ x : ℝ, x ≥ 0 ∧ r x = y :=
by sorry

end range_of_r_l1582_158242


namespace parallelepiped_to_cube_l1582_158210

/-- Represents a rectangular parallelepiped with side lengths (a, b, c) -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a cube with side length s -/
structure Cube where
  s : ℝ

/-- Predicate to check if a parallelepiped can be divided into four parts
    that can be reassembled to form a cube -/
def can_form_cube (p : Parallelepiped) : Prop :=
  ∃ (cube : Cube), 
    cube.s ^ 3 = p.a * p.b * p.c ∧ 
    (∃ (x : ℝ), p.a = 8*x ∧ p.b = 8*x ∧ p.c = 27*x ∧ cube.s = 12*x)

/-- Theorem stating that a rectangular parallelepiped with side ratio 8:8:27
    can be divided into four parts that can be reassembled to form a cube -/
theorem parallelepiped_to_cube : 
  ∀ (p : Parallelepiped), p.a / p.b = 1 ∧ p.b / p.c = 8 / 27 → can_form_cube p :=
by sorry

end parallelepiped_to_cube_l1582_158210


namespace square_vertex_B_l1582_158222

/-- A square in a 2D Cartesian plane -/
structure Square where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- Theorem: Given a square OABC with O(0,0) and A(4,3), and C in the fourth quadrant, B is at (7,-1) -/
theorem square_vertex_B (s : Square) : 
  s.O = (0, 0) → 
  s.A = (4, 3) → 
  isInFourthQuadrant s.C → 
  s.B = (7, -1) := by
  sorry


end square_vertex_B_l1582_158222


namespace friendly_pairs_complete_l1582_158276

def FriendlyPair (a b c d : ℕ+) : Prop :=
  2 * (a.val + b.val) = c.val * d.val ∧ 2 * (c.val + d.val) = a.val * b.val

def AllFriendlyPairs : Set (ℕ+ × ℕ+ × ℕ+ × ℕ+) :=
  {⟨22, 5, 54, 1⟩, ⟨13, 6, 38, 1⟩, ⟨10, 7, 34, 1⟩, ⟨10, 3, 13, 2⟩,
   ⟨6, 4, 10, 2⟩, ⟨6, 3, 6, 3⟩, ⟨4, 4, 4, 4⟩}

theorem friendly_pairs_complete :
  ∀ a b c d : ℕ+, FriendlyPair a b c d ↔ (a, b, c, d) ∈ AllFriendlyPairs :=
sorry

end friendly_pairs_complete_l1582_158276


namespace shirt_cost_l1582_158291

theorem shirt_cost (J S : ℝ) 
  (eq1 : 3 * J + 2 * S = 69) 
  (eq2 : 2 * J + 3 * S = 76) : 
  S = 18 := by
  sorry

end shirt_cost_l1582_158291


namespace mod_congruence_unique_solution_l1582_158264

theorem mod_congruence_unique_solution : 
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 9 ∧ n ≡ -245 [ZMOD 10] ∧ n = 5 := by
  sorry

end mod_congruence_unique_solution_l1582_158264


namespace third_level_lamps_l1582_158296

/-- Represents a pagoda with a given number of stories and lamps -/
structure Pagoda where
  stories : ℕ
  total_lamps : ℕ

/-- Calculates the number of lamps on a specific level of the pagoda -/
def lamps_on_level (p : Pagoda) (level : ℕ) : ℕ :=
  let first_level := p.total_lamps * (1 - 1 / 2^p.stories) / (2^p.stories - 1)
  first_level / 2^(level - 1)

theorem third_level_lamps (p : Pagoda) (h1 : p.stories = 7) (h2 : p.total_lamps = 381) :
  lamps_on_level p 5 = 12 := by
  sorry

#eval lamps_on_level ⟨7, 381⟩ 5

end third_level_lamps_l1582_158296


namespace xyz_equals_27_l1582_158241

theorem xyz_equals_27 
  (a b c x y z : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (nonzero_x : x ≠ 0)
  (nonzero_y : y ≠ 0)
  (nonzero_z : z ≠ 0)
  (eq_a : a = b * c * (x - 2))
  (eq_b : b = a * c * (y - 2))
  (eq_c : c = a * b * (z - 2))
  (sum_product : x * y + x * z + y * z = 10)
  (sum : x + y + z = 6) :
  x * y * z = 27 := by
  sorry


end xyz_equals_27_l1582_158241


namespace triangle_side_length_l1582_158270

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 2 → c = Real.sqrt 2 → Real.cos A = -(Real.sqrt 2) / 4 → b = 1 := by
  sorry

end triangle_side_length_l1582_158270


namespace solution_set_for_a_eq_1_range_of_a_l1582_158293

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| - a

-- Theorem 1: Solution set for f(x) > x + 1 when a = 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x > x + 1} = {x : ℝ | x > 3 ∨ x < -1/3} :=
sorry

-- Theorem 2: Range of a for which ∃x : f(x) < 0.5 * f(x + 1)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x < 0.5 * f a (x + 1)} = {a : ℝ | a > -2} :=
sorry

end solution_set_for_a_eq_1_range_of_a_l1582_158293


namespace equation_holds_iff_m_equals_168_l1582_158259

theorem equation_holds_iff_m_equals_168 :
  ∀ m : ℤ, (4^4 : ℤ) - 7 = 9^2 + m ↔ m = 168 := by
  sorry

end equation_holds_iff_m_equals_168_l1582_158259


namespace triangle_perimeter_l1582_158205

-- Define the triangle sides
def a : ℝ := 10
def b : ℝ := 6
def c : ℝ := 7

-- Define the perimeter
def perimeter : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter : perimeter = 23 := by
  sorry

end triangle_perimeter_l1582_158205


namespace michael_crates_tuesday_l1582_158248

/-- The number of crates Michael bought on Tuesday -/
def T : ℕ := sorry

/-- The number of crates Michael gave out -/
def crates_given_out : ℕ := 2

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday : ℕ := 5

/-- The number of eggs each crate holds -/
def eggs_per_crate : ℕ := 30

/-- The total number of eggs Michael has now -/
def total_eggs : ℕ := 270

theorem michael_crates_tuesday : T = 6 := by
  sorry

end michael_crates_tuesday_l1582_158248


namespace emily_sixth_quiz_score_l1582_158280

def emily_scores : List ℕ := [85, 90, 88, 92, 98]
def desired_mean : ℕ := 92
def num_quizzes : ℕ := 6

theorem emily_sixth_quiz_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = desired_mean ∧
    sixth_score = 99 := by
  sorry

end emily_sixth_quiz_score_l1582_158280


namespace farm_equation_correct_l1582_158268

/-- Represents the farm problem with chickens and pigs --/
structure FarmProblem where
  total_heads : ℕ
  total_legs : ℕ
  chicken_count : ℕ
  pig_count : ℕ

/-- The equation correctly represents the farm problem --/
theorem farm_equation_correct (farm : FarmProblem)
  (head_sum : farm.chicken_count + farm.pig_count = farm.total_heads)
  (head_count : farm.total_heads = 70)
  (leg_count : farm.total_legs = 196) :
  2 * farm.chicken_count + 4 * (70 - farm.chicken_count) = 196 := by
  sorry

#check farm_equation_correct

end farm_equation_correct_l1582_158268


namespace final_sum_after_operations_l1582_158298

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * ((x + 5) + (y + 5)) = 3 * S + 30 := by
  sorry

end final_sum_after_operations_l1582_158298


namespace arithmetic_sequence_2014_l1582_158244

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2014 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 2014 ∧ n = 672 := by
  sorry

end arithmetic_sequence_2014_l1582_158244


namespace fraction_sum_l1582_158252

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_sum_l1582_158252


namespace polynomial_division_remainder_l1582_158240

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 2 * X^6 - X^4 + 4 * X^2 - 7 = (X^2 + 4*X + 3) * q + (-704*X - 706) :=
by sorry

end polynomial_division_remainder_l1582_158240


namespace inverse_proportion_problem_l1582_158235

/-- Given that x and y are inversely proportional, prove that when x + y = 60, x = 3y, 
    and x = -6, then y = -112.5 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) : 
  (x * y = k) →  -- x and y are inversely proportional
  (x + y = 60) →  -- sum condition
  (x = 3 * y) →  -- proportion condition
  (x = -6) →  -- given x value
  y = -112.5 := by sorry

end inverse_proportion_problem_l1582_158235


namespace eh_length_l1582_158263

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 7
  (ex - fx)^2 + (ey - fy)^2 = 7^2 ∧
  -- FG = 21
  (fx - gx)^2 + (fy - gy)^2 = 21^2 ∧
  -- GH = 7
  (gx - hx)^2 + (gy - hy)^2 = 7^2 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 13^2 ∧
  -- Angle at H is a right angle
  (ex - hx) * (gx - hx) + (ey - hy) * (gy - hy) = 0

-- Theorem statement
theorem eh_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  let (ex, ey) := q.E
  let (hx, hy) := q.H
  (ex - hx)^2 + (ey - hy)^2 = 24^2 :=
sorry

end eh_length_l1582_158263


namespace train_bridge_crossing_time_l1582_158224

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 110 → 
  train_speed_kmph = 36 → 
  bridge_length = 170 → 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 28 := by
  sorry


end train_bridge_crossing_time_l1582_158224


namespace max_gcd_triangular_number_l1582_158228

def triangular_number (n : ℕ+) : ℕ := (n : ℕ) * (n + 1) / 2

theorem max_gcd_triangular_number :
  ∃ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n + 2) = 6 ∧
  ∀ (m : ℕ+), Nat.gcd (6 * triangular_number m) (m + 2) ≤ 6 := by
  sorry

end max_gcd_triangular_number_l1582_158228


namespace remainder_82460_div_8_l1582_158284

theorem remainder_82460_div_8 : 82460 % 8 = 4 := by
  sorry

end remainder_82460_div_8_l1582_158284


namespace monic_quartic_polynomial_problem_l1582_158201

-- Define a monic quartic polynomial
def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the polynomial p with given conditions
def p : ℝ → ℝ := sorry

-- State the theorem
theorem monic_quartic_polynomial_problem :
  is_monic_quartic p ∧ 
  p 1 = 2 ∧ 
  p 2 = 7 ∧ 
  p 3 = 10 ∧ 
  p 4 = 17 → 
  p 5 = 26 := by sorry

end monic_quartic_polynomial_problem_l1582_158201


namespace complete_square_quadratic_l1582_158219

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (c d : ℝ), x^2 + 6*x - 4 = 0 ↔ (x + c)^2 = d ∧ d = 13 := by
  sorry

end complete_square_quadratic_l1582_158219


namespace base_2_representation_of_96_l1582_158247

theorem base_2_representation_of_96 :
  ∃ (a b c d e f g : Nat),
    96 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 ∧
    a = 1 ∧ b = 1 ∧ c = 0 ∧ d = 0 ∧ e = 0 ∧ f = 0 ∧ g = 0 :=
by sorry

end base_2_representation_of_96_l1582_158247


namespace problem_statement_l1582_158275

theorem problem_statement (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + 2 = e + Real.sqrt (a + b + c + d - e)) :
  e = 3/4 := by
  sorry

end problem_statement_l1582_158275


namespace path_area_calculation_l1582_158249

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Proves that the area of the path around the given field is 675 sq m -/
theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5) :
  path_area field_length field_width path_width = 675 := by
  sorry

#eval path_area 75 55 2.5

end path_area_calculation_l1582_158249


namespace inequality_proof_l1582_158282

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a * 2^(-b) > b * 2^(-a) := by
  sorry

end inequality_proof_l1582_158282


namespace luisa_apples_taken_l1582_158236

/-- Proves that Luisa took out 2 apples from the bag -/
theorem luisa_apples_taken (initial_apples initial_oranges initial_mangoes : ℕ)
  (remaining_fruits : ℕ) :
  initial_apples = 7 →
  initial_oranges = 8 →
  initial_mangoes = 15 →
  remaining_fruits = 14 →
  ∃ (apples_taken : ℕ),
    apples_taken + 2 * apples_taken + (2 * initial_mangoes / 3) =
      initial_apples + initial_oranges + initial_mangoes - remaining_fruits ∧
    apples_taken = 2 :=
by sorry

end luisa_apples_taken_l1582_158236


namespace vanessa_video_files_l1582_158231

theorem vanessa_video_files :
  ∀ (initial_music_files initial_video_files deleted_files remaining_files : ℕ),
    initial_music_files = 16 →
    deleted_files = 30 →
    remaining_files = 34 →
    initial_music_files + initial_video_files = deleted_files + remaining_files →
    initial_video_files = 48 :=
by
  sorry

end vanessa_video_files_l1582_158231


namespace car_speed_l1582_158237

/-- Represents a kilometer marker with two digits -/
structure Marker where
  tens : ℕ
  ones : ℕ
  h_digits : tens < 10 ∧ ones < 10

/-- Represents the time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24 ∧ minutes < 60

/-- Represents an observation of a marker at a specific time -/
structure Observation where
  time : Time
  marker : Marker

def speed_kmh (start_obs end_obs : Observation) : ℚ :=
  let time_diff := (end_obs.time.hours - start_obs.time.hours : ℚ) + 
                   ((end_obs.time.minutes - start_obs.time.minutes : ℚ) / 60)
  let distance := (end_obs.marker.tens * 10 + end_obs.marker.ones) - 
                  (start_obs.marker.tens * 10 + start_obs.marker.ones)
  distance / time_diff

theorem car_speed 
  (obs1 obs2 obs3 : Observation)
  (h_time1 : obs1.time = ⟨12, 0, by norm_num⟩)
  (h_time2 : obs2.time = ⟨12, 42, by norm_num⟩)
  (h_time3 : obs3.time = ⟨13, 0, by norm_num⟩)
  (h_marker1 : obs1.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_marker2 : obs2.marker = ⟨obs1.marker.ones, obs1.marker.tens, by sorry⟩)
  (h_marker3 : obs3.marker = ⟨obs1.marker.tens, obs1.marker.ones, by sorry⟩)
  (h_constant_speed : speed_kmh obs1 obs2 = speed_kmh obs2 obs3) :
  speed_kmh obs1 obs3 = 90 := by
  sorry


end car_speed_l1582_158237


namespace union_of_intervals_l1582_158246

open Set

theorem union_of_intervals (M N : Set ℝ) : 
  M = {x : ℝ | -1 < x ∧ x < 3} → 
  N = {x : ℝ | x ≥ 1} → 
  M ∪ N = {x : ℝ | x > -1} := by
  sorry

end union_of_intervals_l1582_158246


namespace sum_in_base6_l1582_158223

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- The main theorem --/
theorem sum_in_base6 :
  let a := base6ToBase10 [4, 3, 2, 1]  -- 1234₆
  let b := base6ToBase10 [4, 5, 6]     -- 654₆
  let c := base6ToBase10 [2, 1]        -- 12₆
  base10ToBase6 (a + b + c) = [4, 4, 3, 2] -- 2344₆
:= by sorry

end sum_in_base6_l1582_158223


namespace not_monomial_two_over_a_l1582_158260

/-- Definition of a monomial -/
def is_monomial (e : ℤ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ x, e x = c * x^n

/-- The expression 2/a is not a monomial -/
theorem not_monomial_two_over_a : ¬ is_monomial (λ a => 2 / a) := by
  sorry

end not_monomial_two_over_a_l1582_158260


namespace klinker_double_age_in_15_years_l1582_158266

/-- The number of years it will take for Mr. Klinker to be twice as old as his daughter -/
def years_until_double_age (klinker_age : ℕ) (daughter_age : ℕ) : ℕ :=
  (klinker_age - 2 * daughter_age)

/-- Proof that it will take 15 years for Mr. Klinker to be twice as old as his daughter -/
theorem klinker_double_age_in_15_years :
  years_until_double_age 35 10 = 15 := by
  sorry

#eval years_until_double_age 35 10

end klinker_double_age_in_15_years_l1582_158266


namespace min_filters_correct_l1582_158277

/-- The minimum number of filters required to reduce impurities -/
def min_filters : ℕ := 5

/-- The initial impurity concentration -/
def initial_impurity : ℝ := 0.2

/-- The fraction of impurities remaining after each filter -/
def filter_efficiency : ℝ := 0.2

/-- The maximum allowed final impurity concentration -/
def max_final_impurity : ℝ := 0.0001

/-- Theorem stating that min_filters is the minimum number of filters required -/
theorem min_filters_correct :
  (initial_impurity * filter_efficiency ^ min_filters ≤ max_final_impurity) ∧
  (∀ k : ℕ, k < min_filters → initial_impurity * filter_efficiency ^ k > max_final_impurity) :=
sorry

end min_filters_correct_l1582_158277


namespace larry_cards_remaining_l1582_158203

/-- Given that Larry initially has 67 cards and Dennis takes 9 cards away,
    prove that Larry now has 58 cards. -/
theorem larry_cards_remaining (initial_cards : ℕ) (cards_taken : ℕ) : 
  initial_cards = 67 → cards_taken = 9 → initial_cards - cards_taken = 58 := by
  sorry

end larry_cards_remaining_l1582_158203


namespace smallest_difference_is_one_l1582_158207

/-- Represents a triangle with integer side lengths -/
structure IntegerTriangle where
  de : ℕ
  ef : ℕ
  df : ℕ

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IntegerTriangle) : Prop :=
  t.de + t.ef > t.df ∧ t.de + t.df > t.ef ∧ t.ef + t.df > t.de

/-- Theorem: The smallest possible difference between EF and DE in the given conditions is 1 -/
theorem smallest_difference_is_one :
  ∃ (t : IntegerTriangle),
    t.de + t.ef + t.df = 3005 ∧
    t.de < t.ef ∧
    t.ef ≤ t.df ∧
    is_valid_triangle t ∧
    (∀ (u : IntegerTriangle),
      u.de + u.ef + u.df = 3005 →
      u.de < u.ef →
      u.ef ≤ u.df →
      is_valid_triangle u →
      u.ef - u.de ≥ t.ef - t.de) ∧
    t.ef - t.de = 1 := by
  sorry

end smallest_difference_is_one_l1582_158207


namespace sum_of_proportional_values_l1582_158278

theorem sum_of_proportional_values (a b c d e f : ℝ) 
  (h1 : a / b = 4 / 3)
  (h2 : c / d = 4 / 3)
  (h3 : e / f = 4 / 3)
  (h4 : b + d + f = 15) :
  a + c + e = 20 := by
  sorry

end sum_of_proportional_values_l1582_158278


namespace male_to_total_ratio_l1582_158200

/-- Represents the population of alligators on Lagoon island -/
structure AlligatorPopulation where
  maleCount : ℕ
  adultFemaleCount : ℕ
  juvenileFemaleRatio : ℚ

/-- The ratio of male alligators to total alligators is 1:2 -/
theorem male_to_total_ratio (pop : AlligatorPopulation)
    (h1 : pop.maleCount = 25)
    (h2 : pop.adultFemaleCount = 15)
    (h3 : pop.juvenileFemaleRatio = 2/5) :
    pop.maleCount / (pop.maleCount + pop.adultFemaleCount / (1 - pop.juvenileFemaleRatio)) = 1/2 := by
  sorry


end male_to_total_ratio_l1582_158200


namespace collinear_vectors_x_value_l1582_158202

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, 6)
  collinear a b → x = 3 :=
by sorry

end collinear_vectors_x_value_l1582_158202


namespace jessica_seashells_count_l1582_158290

/-- The number of seashells Mary found -/
def mary_seashells : ℕ := 18

/-- The total number of seashells Mary and Jessica found together -/
def total_seashells : ℕ := 59

/-- The number of seashells Jessica found -/
def jessica_seashells : ℕ := total_seashells - mary_seashells

theorem jessica_seashells_count : jessica_seashells = 41 := by
  sorry

end jessica_seashells_count_l1582_158290


namespace problem_statement_l1582_158225

theorem problem_statement (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + 2*x + 2*y = 10) :
  x^2*y + x*y^2 = 500/121 := by
  sorry

end problem_statement_l1582_158225


namespace correct_oranges_to_put_back_l1582_158230

/-- Represents the fruit selection problem --/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back --/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back --/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 15)
  (h4 : fs.initial_avg_price = 48/100)
  (h5 : fs.desired_avg_price = 45/100) :
  oranges_to_put_back fs = 3 := by
  sorry

end correct_oranges_to_put_back_l1582_158230
