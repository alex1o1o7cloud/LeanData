import Mathlib

namespace NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_l53_5338

/-- A line with slope -3 passing through points A and B -/
def line1 (x y : ℝ) : Prop := y = -3 * x + 13

/-- A line passing through points C and D -/
def line2 (x y : ℝ) : Prop := y = -x + 7

/-- Point A on the x-axis -/
def A : ℝ × ℝ := (5, 0)

/-- Point B on the y-axis -/
def B : ℝ × ℝ := (0, 13)

/-- Point C on the x-axis -/
def C : ℝ × ℝ := (5, 0)

/-- Point D on the y-axis -/
def D : ℝ × ℝ := (0, 7)

/-- Point E where the lines intersect -/
def E : ℝ × ℝ := (3, 4)

/-- The area of quadrilateral OBEC -/
def area_OBEC : ℝ := 67.5

theorem area_of_quadrilateral_OBEC :
  line1 E.1 E.2 ∧ line2 E.1 E.2 →
  area_OBEC = (B.2 * E.1 + C.1 * E.2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_l53_5338


namespace NUMINAMATH_CALUDE_parallelogram_area_from_boards_l53_5373

/-- The area of a parallelogram formed by two boards crossing at a 45-degree angle -/
theorem parallelogram_area_from_boards (board1_width board2_width : ℝ) 
  (h1 : board1_width = 5)
  (h2 : board2_width = 8)
  (h3 : Real.pi / 4 = 45 * Real.pi / 180) :
  board2_width * (board1_width * Real.sin (Real.pi / 4)) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_from_boards_l53_5373


namespace NUMINAMATH_CALUDE_loan_years_approx_eight_l53_5351

/-- Calculates the number of years for which the first part of a loan is lent, given the total sum,
    the second part, and interest rates for both parts. -/
def calculate_years (total : ℚ) (second_part : ℚ) (rate1 : ℚ) (rate2 : ℚ) : ℚ :=
  let first_part := total - second_part
  let n := (second_part * rate2 * 3) / (first_part * rate1)
  n

/-- Proves that given the specified conditions, the number of years for which
    the first part is lent is approximately 8. -/
theorem loan_years_approx_eight :
  let total := 2691
  let second_part := 1656
  let rate1 := 3 / 100
  let rate2 := 5 / 100
  let years := calculate_years total second_part rate1 rate2
  ∃ ε > 0, abs (years - 8) < ε := by
  sorry


end NUMINAMATH_CALUDE_loan_years_approx_eight_l53_5351


namespace NUMINAMATH_CALUDE_bookstore_sales_l53_5336

/-- Calculates the number of bookmarks sold given the number of books sold and the ratio of books to bookmarks. -/
def bookmarks_sold (books : ℕ) (book_ratio : ℕ) (bookmark_ratio : ℕ) : ℕ :=
  (books * bookmark_ratio) / book_ratio

/-- Theorem stating that given 72 books sold and a 9:2 ratio of books to bookmarks, 16 bookmarks were sold. -/
theorem bookstore_sales : bookmarks_sold 72 9 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_l53_5336


namespace NUMINAMATH_CALUDE_cereal_eating_time_l53_5328

/-- The time required for two people to eat a certain amount of cereal together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (quick_rate + slow_rate)

/-- Theorem: Mr. Quick and Mr. Slow eat 5 pounds of cereal in 600/11 minutes -/
theorem cereal_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  eating_time quick_rate slow_rate total_amount = 600 / 11 := by
  sorry

#eval eating_time (1/15 : ℚ) (1/40 : ℚ) 5

end NUMINAMATH_CALUDE_cereal_eating_time_l53_5328


namespace NUMINAMATH_CALUDE_survey_total_is_120_l53_5301

/-- Represents the survey results of parents' ratings on their children's online class experience -/
structure SurveyResults where
  total : ℕ
  excellent : ℕ
  verySatisfactory : ℕ
  satisfactory : ℕ
  needsImprovement : ℕ

/-- The conditions of the survey results -/
def surveyConditions (s : SurveyResults) : Prop :=
  s.excellent = (15 * s.total) / 100 ∧
  s.verySatisfactory = (60 * s.total) / 100 ∧
  s.satisfactory = (80 * (s.total - s.excellent - s.verySatisfactory)) / 100 ∧
  s.needsImprovement = s.total - s.excellent - s.verySatisfactory - s.satisfactory ∧
  s.needsImprovement = 6

/-- Theorem stating that the total number of parents who answered the survey is 120 -/
theorem survey_total_is_120 (s : SurveyResults) (h : surveyConditions s) : s.total = 120 := by
  sorry

end NUMINAMATH_CALUDE_survey_total_is_120_l53_5301


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_y_eq_x_l53_5398

def is_symmetric_point (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = midpoint.2 ∧ (p2.2 - p1.2) / (p2.1 - p1.1) = -1

theorem symmetric_point_wrt_y_eq_x : 
  is_symmetric_point (3, 1) (1, 3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_y_eq_x_l53_5398


namespace NUMINAMATH_CALUDE_conference_attendees_l53_5315

theorem conference_attendees (men : ℕ) : 
  (men : ℝ) * 0.1 + 300 * 0.6 + 500 * 0.7 = (men + 300 + 500 : ℝ) * (1 - 0.5538461538461539) →
  men = 500 := by
sorry

end NUMINAMATH_CALUDE_conference_attendees_l53_5315


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l53_5391

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l53_5391


namespace NUMINAMATH_CALUDE_work_increase_with_absence_l53_5337

/-- Given a total work W distributed among p persons, if 1/5 of the members are absent,
    the increase in work for each remaining person is W/(4p). -/
theorem work_increase_with_absence (W p : ℝ) (h : p > 0) :
  let original_work_per_person := W / p
  let remaining_persons := (4 / 5) * p
  let new_work_per_person := W / remaining_persons
  new_work_per_person - original_work_per_person = W / (4 * p) :=
by sorry

end NUMINAMATH_CALUDE_work_increase_with_absence_l53_5337


namespace NUMINAMATH_CALUDE_expand_product_l53_5319

theorem expand_product (x y : ℝ) : (3*x + 4*y)*(2*x - 5*y + 7) = 6*x^2 - 7*x*y + 21*x - 20*y^2 + 28*y := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l53_5319


namespace NUMINAMATH_CALUDE_grandparents_count_l53_5325

/-- Represents the amount of money each grandparent gave to John -/
def money_per_grandparent : ℕ := 50

/-- Represents the total amount of money John received -/
def total_money : ℕ := 100

/-- The number of grandparents who gave John money -/
def num_grandparents : ℕ := 2

/-- Theorem stating that the number of grandparents who gave John money is 2 -/
theorem grandparents_count :
  num_grandparents = 2 ∧ total_money = num_grandparents * money_per_grandparent :=
sorry

end NUMINAMATH_CALUDE_grandparents_count_l53_5325


namespace NUMINAMATH_CALUDE_sin_45_degrees_l53_5314

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  -- Define the properties of the unit circle and 45° angle
  have unit_circle : ∀ θ, Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := by sorry
  have symmetry_45 : Real.sin (π / 4) = Real.cos (π / 4) := by sorry

  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l53_5314


namespace NUMINAMATH_CALUDE_solve_equation_l53_5371

theorem solve_equation (x y : ℝ) : y = 1 / (2 * x + 2) → y = 2 → x = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l53_5371


namespace NUMINAMATH_CALUDE_total_filets_meeting_limit_l53_5329

/- Define fish species -/
inductive Species
| Bluefish
| Yellowtail
| RedSnapper

/- Define a structure for a fish -/
structure Fish where
  species : Species
  length : Nat

/- Define the minimum size limits -/
def minSizeLimit (s : Species) : Nat :=
  match s with
  | Species.Bluefish => 7
  | Species.Yellowtail => 6
  | Species.RedSnapper => 8

/- Define a function to check if a fish meets the size limit -/
def meetsLimit (f : Fish) : Bool :=
  f.length ≥ minSizeLimit f.species

/- Define the list of all fish caught -/
def allFish : List Fish := [
  {species := Species.Bluefish, length := 5},
  {species := Species.Bluefish, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.RedSnapper, length := 11},
  {species := Species.Bluefish, length := 6},
  {species := Species.Yellowtail, length := 6},
  {species := Species.Yellowtail, length := 10},
  {species := Species.RedSnapper, length := 4},
  {species := Species.Bluefish, length := 8},
  {species := Species.RedSnapper, length := 3},
  {species := Species.Yellowtail, length := 7},
  {species := Species.Yellowtail, length := 12},
  {species := Species.Bluefish, length := 12},
  {species := Species.Bluefish, length := 12}
]

/- Define the number of filets per fish -/
def filetsPerFish : Nat := 2

/- Theorem: The total number of filets from fish meeting size limits is 22 -/
theorem total_filets_meeting_limit : 
  (allFish.filter meetsLimit).length * filetsPerFish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_filets_meeting_limit_l53_5329


namespace NUMINAMATH_CALUDE_square_sum_theorem_l53_5377

theorem square_sum_theorem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : x*y + x + y = 11) : 
  x^2 + y^2 = 2893/36 := by
sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l53_5377


namespace NUMINAMATH_CALUDE_pressure_change_l53_5382

/-- Given a relationship between pressure (P), area (A), and velocity (V),
    prove that doubling the area and increasing velocity from 20 to 30
    results in a specific pressure change. -/
theorem pressure_change (k : ℝ) :
  (∃ (P₀ A₀ V₀ : ℝ), P₀ = k * A₀ * V₀^2 ∧ P₀ = 0.5 ∧ A₀ = 1 ∧ V₀ = 20) →
  (∃ (P₁ A₁ V₁ : ℝ), P₁ = k * A₁ * V₁^2 ∧ A₁ = 2 ∧ V₁ = 30 ∧ P₁ = 2.25) :=
by sorry

end NUMINAMATH_CALUDE_pressure_change_l53_5382


namespace NUMINAMATH_CALUDE_molecular_weight_aluminum_iodide_l53_5343

/-- Given that the molecular weight of 7 moles of aluminum iodide is 2856 grams,
    prove that the molecular weight of one mole of aluminum iodide is 408 grams/mole. -/
theorem molecular_weight_aluminum_iodide :
  let total_weight : ℝ := 2856
  let num_moles : ℝ := 7
  total_weight / num_moles = 408 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_aluminum_iodide_l53_5343


namespace NUMINAMATH_CALUDE_solve_system_l53_5346

theorem solve_system (a b x y : ℝ) 
  (eq1 : a * x + b * y = 16)
  (eq2 : b * x - a * y = -12)
  (sol_x : x = 2)
  (sol_y : y = 4) : 
  a = 4 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l53_5346


namespace NUMINAMATH_CALUDE_angle4_is_70_l53_5352

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 : ℝ)

-- Define the conditions
axiom angle1_plus_angle2 : angle1 + angle2 = 180
axiom angle4_eq_angle5 : angle4 = angle5
axiom triangle_sum : angle1 + angle3 + angle5 = 180
axiom angle1_value : angle1 = 50
axiom angle3_value : angle3 = 60

-- Theorem to prove
theorem angle4_is_70 : angle4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle4_is_70_l53_5352


namespace NUMINAMATH_CALUDE_friend_has_five_balloons_l53_5300

/-- The number of balloons you have -/
def your_balloons : ℕ := 7

/-- The difference between your balloons and your friend's balloons -/
def difference : ℕ := 2

/-- The number of balloons your friend has -/
def friend_balloons : ℕ := your_balloons - difference

theorem friend_has_five_balloons : friend_balloons = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_has_five_balloons_l53_5300


namespace NUMINAMATH_CALUDE_cookie_tin_weight_is_9_l53_5333

/-- The weight of a tin of cookies in ounces -/
def cookie_tin_weight (chip_bag_weight : ℕ) (num_chip_bags : ℕ) (cookie_tin_multiplier : ℕ) (total_weight_pounds : ℕ) : ℕ :=
  let total_weight_ounces : ℕ := total_weight_pounds * 16
  let total_chip_weight : ℕ := chip_bag_weight * num_chip_bags
  let num_cookie_tins : ℕ := num_chip_bags * cookie_tin_multiplier
  let total_cookie_weight : ℕ := total_weight_ounces - total_chip_weight
  total_cookie_weight / num_cookie_tins

/-- Theorem stating that a tin of cookies weighs 9 ounces under the given conditions -/
theorem cookie_tin_weight_is_9 :
  cookie_tin_weight 20 6 4 21 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_tin_weight_is_9_l53_5333


namespace NUMINAMATH_CALUDE_expected_value_binomial_li_expected_traffic_jams_l53_5312

/-- The number of intersections Mr. Li passes through -/
def n : ℕ := 6

/-- The probability of a traffic jam at each intersection -/
def p : ℚ := 1/6

/-- The expected value of a binomial distribution is n * p -/
theorem expected_value_binomial (n : ℕ) (p : ℚ) :
  n * p = 1 → n = 6 ∧ p = 1/6 := by sorry

/-- The expected number of traffic jams Mr. Li encounters is 1 -/
theorem li_expected_traffic_jams :
  n * p = 1 := by sorry

end NUMINAMATH_CALUDE_expected_value_binomial_li_expected_traffic_jams_l53_5312


namespace NUMINAMATH_CALUDE_smallest_n_correct_l53_5394

/-- The smallest positive integer n for which (x^3 - 1/x^2)^n contains a non-zero constant term -/
def smallest_n : ℕ := 5

/-- Predicate to check if (x^3 - 1/x^2)^n has a non-zero constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ (k : ℕ), k ≠ 0 ∧ (3 * n = 5 * k)

theorem smallest_n_correct :
  (has_constant_term smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_constant_term m)) :=
by sorry

#check smallest_n_correct

end NUMINAMATH_CALUDE_smallest_n_correct_l53_5394


namespace NUMINAMATH_CALUDE_product_325_67_base_7_units_digit_l53_5381

theorem product_325_67_base_7_units_digit : 
  (325 * 67) % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_product_325_67_base_7_units_digit_l53_5381


namespace NUMINAMATH_CALUDE_pyramid_volume_l53_5388

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) : 
  base_side = 1/3 → height = 1 → volume = (1/3) * (base_side^2) * height → volume = 1/27 := by
  sorry

#check pyramid_volume

end NUMINAMATH_CALUDE_pyramid_volume_l53_5388


namespace NUMINAMATH_CALUDE_expected_outcome_is_correct_l53_5376

/-- Represents the possible outcomes of rolling a die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six

/-- The probability of rolling a specific outcome -/
def probability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 1/3
  | DieOutcome.Four | DieOutcome.Five | DieOutcome.Six => 1/6

/-- The monetary value associated with each outcome -/
def monetaryValue (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 4
  | DieOutcome.Four => -2
  | DieOutcome.Five => -5
  | DieOutcome.Six => -7

/-- The expected monetary outcome of a roll -/
def expectedMonetaryOutcome : ℚ :=
  (probability DieOutcome.One * monetaryValue DieOutcome.One) +
  (probability DieOutcome.Two * monetaryValue DieOutcome.Two) +
  (probability DieOutcome.Three * monetaryValue DieOutcome.Three) +
  (probability DieOutcome.Four * monetaryValue DieOutcome.Four) +
  (probability DieOutcome.Five * monetaryValue DieOutcome.Five) +
  (probability DieOutcome.Six * monetaryValue DieOutcome.Six)

theorem expected_outcome_is_correct :
  expectedMonetaryOutcome = 167/100 := by sorry

end NUMINAMATH_CALUDE_expected_outcome_is_correct_l53_5376


namespace NUMINAMATH_CALUDE_example_linear_equation_l53_5384

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants and at least one of a or b is non-zero. --/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y = a * x + b * y - c

/-- The equation x + 4y = 6 is a linear equation in two variables. --/
theorem example_linear_equation :
  IsLinearEquationInTwoVariables (fun x y ↦ x + 4 * y - 6) := by
  sorry

end NUMINAMATH_CALUDE_example_linear_equation_l53_5384


namespace NUMINAMATH_CALUDE_young_inequality_l53_5317

theorem young_inequality (x y α β : ℝ) 
  (hx : x > 0) (hy : y > 0) (hα : α > 0) (hβ : β > 0) (hsum : α + β = 1) :
  x^α * y^β ≤ α*x + β*y :=
sorry

end NUMINAMATH_CALUDE_young_inequality_l53_5317


namespace NUMINAMATH_CALUDE_find_point_B_l53_5324

/-- Given vector a, point A, and a line y = 2x, find point B on the line such that AB is parallel to a -/
theorem find_point_B (a : ℝ × ℝ) (A : ℝ × ℝ) :
  a = (1, 1) →
  A = (-3, -1) →
  ∃ B : ℝ × ℝ,
    B.2 = 2 * B.1 ∧
    ∃ k : ℝ, k • a = (B.1 - A.1, B.2 - A.2) ∧
    B = (2, 4) := by
  sorry


end NUMINAMATH_CALUDE_find_point_B_l53_5324


namespace NUMINAMATH_CALUDE_xyz_value_l53_5326

theorem xyz_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30 * Real.rpow 4 (1/3))
  (hxz : x * z = 45 * Real.rpow 4 (1/3))
  (hyz : y * z = 18 * Real.rpow 4 (1/3)) :
  x * y * z = 540 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l53_5326


namespace NUMINAMATH_CALUDE_power_of_three_expression_l53_5390

theorem power_of_three_expression : ∀ (a b c d e f g h : ℕ), 
  a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 4 ∧ e = 8 ∧ f = 16 ∧ g = 32 ∧ h = 64 →
  3^a * 3^b / 3^c / 3^d / 3^e * 3^f * 3^g * 3^h = 3^99 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l53_5390


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_two_one_l53_5370

/-- 
Given two positive integers r and s with r > s, and two distinct non-constant polynomials P and Q 
with real coefficients such that P(x)^r - P(x)^s = Q(x)^r - Q(x)^s for all real x, 
prove that r = 2 and s = 1.
-/
theorem polynomial_equality_implies_two_one (r s : ℕ) (P Q : ℝ → ℝ) : 
  r > s → 
  s > 0 →
  (∀ x : ℝ, P x ≠ Q x) → 
  (∃ a b c d : ℝ, a ≠ 0 ∧ c ≠ 0 ∧ ∀ x : ℝ, P x = a * x + b ∧ Q x = c * x + d) →
  (∀ x : ℝ, (P x)^r - (P x)^s = (Q x)^r - (Q x)^s) →
  r = 2 ∧ s = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_two_one_l53_5370


namespace NUMINAMATH_CALUDE_lunch_break_duration_l53_5313

-- Define the painting rates and lunch break duration
structure PaintingScenario where
  joseph_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the conditions from the problem
def monday_condition (s : PaintingScenario) : Prop :=
  (8 - s.lunch_break) * (s.joseph_rate + s.helpers_rate) = 0.6

def tuesday_condition (s : PaintingScenario) : Prop :=
  (5 - s.lunch_break) * s.helpers_rate = 0.3

def wednesday_condition (s : PaintingScenario) : Prop :=
  (6 - s.lunch_break) * s.joseph_rate = 0.1

-- Theorem stating that the lunch break is 45 minutes
theorem lunch_break_duration :
  ∃ (s : PaintingScenario),
    monday_condition s ∧
    tuesday_condition s ∧
    wednesday_condition s ∧
    s.lunch_break = 0.75 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l53_5313


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l53_5393

-- Define the universal set I as ℝ
def I : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^(Real.sqrt (3 + 2*x - x^2))}

-- Define set N
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_complement_theorem : M ∩ (I \ N) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l53_5393


namespace NUMINAMATH_CALUDE_june_found_17_eggs_l53_5334

/-- The total number of bird eggs June found -/
def total_eggs (tree1_nests tree1_eggs_per_nest tree2_eggs frontyard_eggs : ℕ) : ℕ :=
  tree1_nests * tree1_eggs_per_nest + tree2_eggs + frontyard_eggs

/-- Theorem stating that June found 17 eggs in total -/
theorem june_found_17_eggs : 
  total_eggs 2 5 3 4 = 17 := by
  sorry

end NUMINAMATH_CALUDE_june_found_17_eggs_l53_5334


namespace NUMINAMATH_CALUDE_inequality_proof_l53_5386

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem inequality_proof (a s t : ℝ) (h1 : s > 0) (h2 : t > 0) 
  (h3 : 2 * s + t = a) 
  (h4 : Set.Icc (-1) 7 = {x | f a x ≤ 4}) : 
  1 / s + 8 / t ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l53_5386


namespace NUMINAMATH_CALUDE_playground_area_l53_5375

/-- A rectangular playground with perimeter 72 feet and length three times the width has an area of 243 square feet. -/
theorem playground_area : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  2 * (w + l) = 72 →
  l = 3 * w →
  w * l = 243 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l53_5375


namespace NUMINAMATH_CALUDE_at_least_one_side_not_exceeding_double_l53_5340

-- Define a structure for a parallelogram
structure Parallelogram :=
  (side1 : ℝ)
  (side2 : ℝ)
  (area : ℝ)

-- Define the problem setup
def parallelogram_inscriptions (P1 P2 P3 : Parallelogram) : Prop :=
  -- P2 is inscribed in P1
  P2.area < P1.area ∧
  -- P3 is inscribed in P2
  P3.area < P2.area ∧
  -- The sides of P3 are parallel to the sides of P1
  (P3.side1 < P1.side1 ∧ P3.side2 < P1.side2)

-- Theorem statement
theorem at_least_one_side_not_exceeding_double :
  ∀ (P1 P2 P3 : Parallelogram),
  parallelogram_inscriptions P1 P2 P3 →
  (P1.side1 ≤ 2 * P3.side1 ∨ P1.side2 ≤ 2 * P3.side2) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_side_not_exceeding_double_l53_5340


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l53_5302

theorem units_digit_of_expression : 
  (30 * 32 * 34 * 36 * 38 * 40) / 2000 ≡ 6 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l53_5302


namespace NUMINAMATH_CALUDE_g_sum_symmetric_l53_5306

/-- Given a function g(x) = ax^8 + bx^6 - cx^4 + 5 where g(10) = 3,
    prove that g(10) + g(-10) = 6 -/
theorem g_sum_symmetric (a b c : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^8 + b * x^6 - c * x^4 + 5
  g 10 = 3 → g 10 + g (-10) = 6 := by sorry

end NUMINAMATH_CALUDE_g_sum_symmetric_l53_5306


namespace NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l53_5399

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumOfFirstNPrimes (n : ℕ) : ℕ := (List.range n).map (nthPrime ∘ (· + 1)) |>.sum

/-- There exists a perfect square between the sum of the first n primes and the sum of the first n+1 primes -/
theorem perfect_square_between_prime_sums (n : ℕ) :
  ∃ k : ℕ, sumOfFirstNPrimes n < k^2 ∧ k^2 < sumOfFirstNPrimes (n + 1) := by sorry

end NUMINAMATH_CALUDE_perfect_square_between_prime_sums_l53_5399


namespace NUMINAMATH_CALUDE_problem_statement_l53_5397

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * (6 - a) ≤ 9) ∧ 
  (a * b = a + b + 3 → a * b ≥ 9) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l53_5397


namespace NUMINAMATH_CALUDE_sandbox_area_calculation_l53_5323

/-- The area of a rectangular sandbox in square centimeters -/
def sandbox_area (length_meters : ℝ) (width_cm : ℝ) : ℝ :=
  (length_meters * 100) * width_cm

/-- Theorem: The area of a rectangular sandbox with length 3.12 meters and width 146 centimeters is 45552 square centimeters -/
theorem sandbox_area_calculation :
  sandbox_area 3.12 146 = 45552 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_area_calculation_l53_5323


namespace NUMINAMATH_CALUDE_min_value_theorem_l53_5303

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.sqrt (b / a) + Real.sqrt (a / b) - 2 = (Real.sqrt (a * b) - 4 * a * b) / (2 * a * b)) :
  ∃ (min : ℝ), min = 4 * Real.sqrt 2 + 6 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → 
  Real.sqrt (y / x) + Real.sqrt (x / y) - 2 = (Real.sqrt (x * y) - 4 * x * y) / (2 * x * y) →
  1 / x + 2 / y ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l53_5303


namespace NUMINAMATH_CALUDE_cubic_polynomial_remainder_l53_5383

/-- A cubic polynomial of the form ax³ - 6x² + bx - 5 -/
def f (a b x : ℝ) : ℝ := a * x^3 - 6 * x^2 + b * x - 5

theorem cubic_polynomial_remainder (a b : ℝ) :
  (f a b 1 = -5) ∧ (f a b (-2) = -53) → a = 7 ∧ b = -7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_remainder_l53_5383


namespace NUMINAMATH_CALUDE_difference_of_squares_l53_5308

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l53_5308


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l53_5320

/-- An isosceles triangle with congruent sides of 6 cm and perimeter of 20 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base : ℝ), 
    base > 0 → 
    6 + 6 + base = 20 → 
    base = 8 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l53_5320


namespace NUMINAMATH_CALUDE_exchange_of_segments_is_structure_variation_l53_5380

-- Define the basic concepts
def ChromosomalVariation : Type := sorry
def NonHomologousChromosome : Type := sorry
def ChromosomeStructure : Type := sorry
def Translocation : Type := sorry

-- Define the exchange of partial segments between non-homologous chromosomes
def PartialSegmentExchange (c1 c2 : NonHomologousChromosome) : Translocation := sorry

-- Define what constitutes a variation in chromosome structure
def IsChromosomeStructureVariation (t : Translocation) : Prop := sorry

-- Theorem to prove
theorem exchange_of_segments_is_structure_variation 
  (c1 c2 : NonHomologousChromosome) : 
  IsChromosomeStructureVariation (PartialSegmentExchange c1 c2) := by
  sorry

end NUMINAMATH_CALUDE_exchange_of_segments_is_structure_variation_l53_5380


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l53_5359

/-- Given real numbers a, b, c, d satisfying the conditions,
    prove that the minimum value of (a-c)^2 + (b-d)^2 is (9/5) * (ln(e/3))^2 -/
theorem min_distance_curve_line (a b c d : ℝ) 
    (h1 : (a + 3 * Real.log a) / b = 1)
    (h2 : (d - 3) / (2 * c) = 1) :
    ∃ (min : ℝ), min = (9/5) * (Real.log (Real.exp 1 / 3))^2 ∧
    ∀ (x y z w : ℝ), 
    (x + 3 * Real.log x) / y = 1 → 
    (w - 3) / (2 * z) = 1 → 
    (x - z)^2 + (y - w)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l53_5359


namespace NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l53_5357

theorem only_negative_one_squared_is_negative :
  ((-1 : ℝ)^0 < 0 ∨ |-1| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1^2) < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |-1| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1^2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l53_5357


namespace NUMINAMATH_CALUDE_fraction_problem_l53_5321

theorem fraction_problem (a b : ℚ) (h1 : a + b = 100) (h2 : b = 60) : 
  (3 / 10) * a = (1 / 5) * b := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l53_5321


namespace NUMINAMATH_CALUDE_divisibility_problem_l53_5345

theorem divisibility_problem (a b c : ℕ) 
  (ha : a > 1) 
  (hb : b > c) 
  (hc : c > 1) 
  (hdiv : (a * b * c + 1) % (a * b - b + 1) = 0) : 
  b % a = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l53_5345


namespace NUMINAMATH_CALUDE_game_results_l53_5389

/-- A game between two players A and B with specific winning conditions -/
structure Game where
  pA : ℝ  -- Probability of A winning a single game
  pB : ℝ  -- Probability of B winning a single game
  hpA : pA = 2/3
  hpB : pB = 1/3
  hprob : pA + pB = 1

/-- The number of games played when the match is decided -/
def num_games (g : Game) : ℕ → ℝ
  | 2 => g.pA^2 + g.pB^2
  | 3 => g.pB * g.pA^2 + g.pA * g.pB^2
  | 4 => g.pA * g.pB * g.pA^2 + g.pB * g.pA * g.pB^2
  | 5 => g.pB * g.pA * g.pB * g.pA + g.pA * g.pB * g.pA * g.pB
  | _ => 0

/-- The probability that B wins exactly one game and A wins the match -/
def prob_B_wins_one (g : Game) : ℝ :=
  g.pB * g.pA^2 + g.pA * g.pB * g.pA^2

/-- The expected number of games played -/
def expected_games (g : Game) : ℝ :=
  2 * (num_games g 2) + 3 * (num_games g 3) + 4 * (num_games g 4) + 5 * (num_games g 5)

theorem game_results (g : Game) :
  prob_B_wins_one g = 20/81 ∧
  num_games g 2 = 5/9 ∧
  num_games g 3 = 2/9 ∧
  num_games g 4 = 10/81 ∧
  num_games g 5 = 8/81 ∧
  expected_games g = 224/81 := by
  sorry

end NUMINAMATH_CALUDE_game_results_l53_5389


namespace NUMINAMATH_CALUDE_license_plate_count_l53_5327

/-- The number of digits in a license plate -/
def num_digits : ℕ := 5

/-- The number of letters in a license plate -/
def num_letters : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_choices : ℕ := 10

/-- The number of possible letters -/
def letter_choices : ℕ := 26

/-- The number of non-vowel letters -/
def non_vowel_choices : ℕ := 21

/-- The number of positions where the letter block can be placed -/
def block_positions : ℕ := num_digits + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * digit_choices^num_digits * (letter_choices^num_letters - non_vowel_choices^num_letters)

theorem license_plate_count : total_license_plates = 4989000000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l53_5327


namespace NUMINAMATH_CALUDE_stating_club_officer_selection_count_l53_5349

/-- Represents the number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of offices to be filled -/
def num_offices : ℕ := 3

/-- 
Theorem stating that the number of ways to choose a president, vice-president, and secretary 
from a club of 24 members (12 boys and 12 girls) is 5808, given that the president and 
vice-president must be of the same gender, the secretary can be of any gender, and no one 
can hold more than one office.
-/
theorem club_officer_selection_count : 
  (num_boys * (num_boys - 1) + num_girls * (num_girls - 1)) * (total_members - 2) = 5808 := by
  sorry

end NUMINAMATH_CALUDE_stating_club_officer_selection_count_l53_5349


namespace NUMINAMATH_CALUDE_complex_equation_solution_l53_5362

/-- Given a complex number Z satisfying (1+i)Z = 2, prove that Z = 1 - i -/
theorem complex_equation_solution (Z : ℂ) (h : (1 + Complex.I) * Z = 2) : Z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l53_5362


namespace NUMINAMATH_CALUDE_a_investment_value_l53_5363

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- The theorem states that given the specific conditions of the partnership,
    a's investment must be 24000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 32000)
  (hc : p.c_investment = 36000)
  (hp : p.total_profit = 92000)
  (hcs : p.c_profit_share = 36000)
  (h_profit_distribution : p.c_profit_share = p.c_investment * p.total_profit / (p.a_investment + p.b_investment + p.c_investment)) :
  p.a_investment = 24000 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_value_l53_5363


namespace NUMINAMATH_CALUDE_yoongi_has_bigger_number_l53_5356

theorem yoongi_has_bigger_number : ∀ (yoongi_number jungkook_number : ℕ),
  yoongi_number = 4 →
  jungkook_number = 6 / 3 →
  yoongi_number > jungkook_number :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_bigger_number_l53_5356


namespace NUMINAMATH_CALUDE_second_player_wins_alice_wins_l53_5395

/-- Represents the frequency of each letter in the string -/
def LetterFrequency := Char → Nat

/-- The game state -/
structure GameState where
  frequencies : LetterFrequency
  playerTurn : Bool -- true for first player, false for second player

/-- Checks if all frequencies are even -/
def allEven (freq : LetterFrequency) : Prop :=
  ∀ c, Even (freq c)

/-- Represents a valid move in the game -/
inductive Move where
  | erase (c : Char) (n : Nat)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.erase c n =>
      { frequencies := λ x => if x = c then state.frequencies x - n else state.frequencies x,
        playerTurn := ¬state.playerTurn }

/-- Checks if the game is over (all frequencies are zero) -/
def isGameOver (state : GameState) : Prop :=
  ∀ c, state.frequencies c = 0

/-- The winning strategy for the second player -/
def secondPlayerStrategy (state : GameState) : Move :=
  sorry -- Implementation not required for the statement

/-- The main theorem stating that the second player can always win -/
theorem second_player_wins (initialState : GameState) :
  ¬initialState.playerTurn →
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

/-- The specific game instance from the problem -/
def initialGameState : GameState :=
  { frequencies := λ c =>
      if c = 'А' then 3
      else if c = 'О' then 3
      else if c = 'Д' then 2
      else if c = 'Я' then 2
      else if c ∈ ['Г', 'Р', 'С', 'К', 'У', 'Т', 'Н', 'Л', 'И', 'М', 'П'] then 1
      else 0,
    playerTurn := false }

/-- Theorem specific to the given problem instance -/
theorem alice_wins : 
  ∃ (strategy : GameState → Move),
    ∀ (moves : List Move),
      let finalState := (moves.foldl applyMove initialGameState)
      (isGameOver finalState ∧ ¬finalState.playerTurn) ∨
      (¬isGameOver finalState ∧ allEven finalState.frequencies) :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_alice_wins_l53_5395


namespace NUMINAMATH_CALUDE_part_one_part_two_l53_5318

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| - |x - a|

-- Part I
theorem part_one (a : ℝ) : f a 1 > 1 ↔ a ∈ Set.Iic (-1) ∪ Set.Ioi 1 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a < 0) :
  (∀ x y : ℝ, x ≤ a → y ≤ a → f a x ≤ |y + 2020| + |y - a|) ↔
  a ∈ Set.Icc (-1010) 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l53_5318


namespace NUMINAMATH_CALUDE_min_value_of_x_l53_5307

theorem min_value_of_x (x : ℝ) (h1 : x > 0) (h2 : Real.log x ≥ Real.log 2 + (1/2) * Real.log x) : x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_x_l53_5307


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l53_5365

theorem complex_number_quadrant : ∃ (z : ℂ), z = (3 + 4*I)*I ∧ (z.re < 0 ∧ z.im > 0) :=
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l53_5365


namespace NUMINAMATH_CALUDE_sin_neg_ten_thirds_pi_l53_5361

theorem sin_neg_ten_thirds_pi : Real.sin (-10/3 * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_ten_thirds_pi_l53_5361


namespace NUMINAMATH_CALUDE_product_of_integers_l53_5364

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_squares_eq : x^2 - y^2 = 52) :
  x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l53_5364


namespace NUMINAMATH_CALUDE_witch_clock_theorem_l53_5387

def clock_cycle (t : ℕ) : ℕ :=
  (5 * (t / 8 + 1) - 3 * (t / 8)) % 60

theorem witch_clock_theorem (t : ℕ) (h : t = 2022) :
  clock_cycle t = 28 := by
  sorry

end NUMINAMATH_CALUDE_witch_clock_theorem_l53_5387


namespace NUMINAMATH_CALUDE_product_is_three_l53_5368

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The product of the repeating decimal 0.333... and 9 --/
def product : ℚ := repeating_third * 9

/-- Theorem stating that the product of 0.333... and 9 is equal to 3 --/
theorem product_is_three : product = 3 := by sorry

end NUMINAMATH_CALUDE_product_is_three_l53_5368


namespace NUMINAMATH_CALUDE_no_dual_integer_root_quadratics_l53_5360

theorem no_dual_integer_root_quadratics : 
  ¬ ∃ (a b c : ℤ), 
    (∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℤ), y₁ ≠ y₂ ∧ (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_dual_integer_root_quadratics_l53_5360


namespace NUMINAMATH_CALUDE_min_value_on_line_equality_condition_l53_5396

theorem min_value_on_line (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b - 1 = 0 → 4/(a + b) + 1/b ≥ 9 :=
by sorry

theorem equality_condition (a b : ℝ) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + 2*b - 1 = 0 ∧ 4/(a + b) + 1/b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_equality_condition_l53_5396


namespace NUMINAMATH_CALUDE_amount_received_by_B_l53_5369

/-- Theorem: Given a total amount of 1440, if A receives 1/3 as much as B, and B receives 1/4 as much as C, then B receives 202.5. -/
theorem amount_received_by_B (total : ℝ) (a b c : ℝ) : 
  total = 1440 →
  a = (1/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  b = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_amount_received_by_B_l53_5369


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l53_5339

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (5 * x - 3) / (x^2 - 3*x - 18) = C / (x - 6) + D / (x + 3)) →
  C = 3 ∧ D = 2 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l53_5339


namespace NUMINAMATH_CALUDE_bottle_height_l53_5367

/-- Represents a bottle composed of two cylinders -/
structure Bottle where
  r1 : ℝ  -- radius of the first cylinder
  r2 : ℝ  -- radius of the second cylinder
  h_right : ℝ  -- water height when right side up
  h_upside : ℝ  -- water height when upside down

/-- The total height of the bottle -/
def total_height (b : Bottle) : ℝ :=
  29

/-- Theorem stating that the total height of the bottle is 29 cm -/
theorem bottle_height (b : Bottle) 
  (h_r1 : b.r1 = 1) 
  (h_r2 : b.r2 = 3) 
  (h_right : b.h_right = 20) 
  (h_upside : b.h_upside = 28) : 
  total_height b = 29 := by
  sorry

end NUMINAMATH_CALUDE_bottle_height_l53_5367


namespace NUMINAMATH_CALUDE_distribution_combinations_l53_5305

/-- The number of ways to distribute 2 objects among 4 categories -/
def distributionCount : ℕ := 10

/-- The number of categories -/
def categoryCount : ℕ := 4

/-- The number of objects to distribute -/
def objectCount : ℕ := 2

theorem distribution_combinations :
  (categoryCount : ℕ) + (categoryCount * (categoryCount - 1) / 2) = distributionCount :=
sorry

end NUMINAMATH_CALUDE_distribution_combinations_l53_5305


namespace NUMINAMATH_CALUDE_action_figure_price_l53_5392

/-- Given the cost of sneakers, initial savings, number of action figures sold, and money left after purchase, 
    prove the price per action figure. -/
theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (figures_sold : ℕ) 
  (money_left : ℕ) 
  (h1 : sneaker_cost = 90)
  (h2 : initial_savings = 15)
  (h3 : figures_sold = 10)
  (h4 : money_left = 25) :
  (sneaker_cost - initial_savings + money_left) / figures_sold = 10 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_price_l53_5392


namespace NUMINAMATH_CALUDE_jim_sara_savings_equality_l53_5335

/-- Proves that Jim and Sara will have saved the same amount after 820 weeks -/
theorem jim_sara_savings_equality :
  let sara_initial : ℕ := 4100
  let sara_weekly : ℕ := 10
  let jim_weekly : ℕ := 15
  let weeks : ℕ := 820
  sara_initial + sara_weekly * weeks = jim_weekly * weeks :=
by sorry

end NUMINAMATH_CALUDE_jim_sara_savings_equality_l53_5335


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l53_5347

/-- Given a geometric sequence {a_n} where a₁ = 3 and a₄ = 24, prove that a₃ + a₄ + a₅ = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Condition for geometric sequence
  a 1 = 3 →
  a 4 = 24 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l53_5347


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l53_5378

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

theorem perpendicular_vectors (t : ℝ) : 
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l53_5378


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l53_5372

theorem decimal_to_fraction (x : ℚ) : x = 3.675 → x = 147 / 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l53_5372


namespace NUMINAMATH_CALUDE_complex_equation_proof_l53_5341

theorem complex_equation_proof (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^2005 + (y / (x + y))^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l53_5341


namespace NUMINAMATH_CALUDE_bug_crawl_distance_l53_5342

-- Define the positions of the bug
def start_pos : ℤ := 3
def pos1 : ℤ := -5
def pos2 : ℤ := 8
def end_pos : ℤ := 0

-- Define the function to calculate distance between two points
def distance (a b : ℤ) : ℕ := (a - b).natAbs

-- Define the total distance
def total_distance : ℕ := 
  distance start_pos pos1 + distance pos1 pos2 + distance pos2 end_pos

-- Theorem to prove
theorem bug_crawl_distance : total_distance = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_crawl_distance_l53_5342


namespace NUMINAMATH_CALUDE_student_arrangements_l53_5358

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def adjacent_arrangement (n : ℕ) : ℕ := 2 * factorial (n - 1)

def non_adjacent_arrangement (n : ℕ) : ℕ := factorial (n - 2) * (n * (n - 1))

def special_arrangement (n : ℕ) : ℕ := 
  factorial n - 3 * factorial (n - 1) + 2 * factorial (n - 2)

theorem student_arrangements :
  adjacent_arrangement 5 = 48 ∧
  non_adjacent_arrangement 5 = 72 ∧
  special_arrangement 5 = 60 := by
  sorry

#eval adjacent_arrangement 5
#eval non_adjacent_arrangement 5
#eval special_arrangement 5

end NUMINAMATH_CALUDE_student_arrangements_l53_5358


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l53_5350

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 3) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (3 * n) / S a n = c) →
  a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l53_5350


namespace NUMINAMATH_CALUDE_diophantine_equation_only_trivial_solution_l53_5385

theorem diophantine_equation_only_trivial_solution (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_only_trivial_solution_l53_5385


namespace NUMINAMATH_CALUDE_emma_henry_weight_l53_5309

theorem emma_henry_weight (e f g h : ℝ) 
  (ef_sum : e + f = 310)
  (fg_sum : f + g = 265)
  (gh_sum : g + h = 280) :
  e + h = 325 := by
sorry

end NUMINAMATH_CALUDE_emma_henry_weight_l53_5309


namespace NUMINAMATH_CALUDE_double_thrice_one_is_eight_l53_5322

def double (n : ℕ) : ℕ := 2 * n

def iterate_double (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => iterate_double (double n) k

theorem double_thrice_one_is_eight :
  iterate_double 1 3 = 8 := by sorry

end NUMINAMATH_CALUDE_double_thrice_one_is_eight_l53_5322


namespace NUMINAMATH_CALUDE_geometric_mean_problem_l53_5355

theorem geometric_mean_problem (k : ℝ) : (2 * k)^2 = k * (k + 3) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_problem_l53_5355


namespace NUMINAMATH_CALUDE_correct_counting_error_l53_5316

/-- The error in cents to be subtracted when quarters are mistakenly counted as half dollars
    and nickels are mistakenly counted as dimes. -/
def counting_error (x y : ℕ) : ℕ := 25 * x + 5 * y

/-- The value of a quarter in cents. -/
def quarter_value : ℕ := 25

/-- The value of a half dollar in cents. -/
def half_dollar_value : ℕ := 50

/-- The value of a nickel in cents. -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents. -/
def dime_value : ℕ := 10

theorem correct_counting_error (x y : ℕ) :
  counting_error x y = (half_dollar_value - quarter_value) * x + (dime_value - nickel_value) * y :=
by sorry

end NUMINAMATH_CALUDE_correct_counting_error_l53_5316


namespace NUMINAMATH_CALUDE_estimate_households_with_three_plus_houses_l53_5332

/-- Estimate the number of households owning 3 or more houses -/
theorem estimate_households_with_three_plus_houses
  (total_households : ℕ)
  (ordinary_households : ℕ)
  (high_income_households : ℕ)
  (sample_ordinary : ℕ)
  (sample_high_income : ℕ)
  (sample_ordinary_with_three_plus : ℕ)
  (sample_high_income_with_three_plus : ℕ)
  (h1 : total_households = 100000)
  (h2 : ordinary_households = 99000)
  (h3 : high_income_households = 1000)
  (h4 : sample_ordinary = 990)
  (h5 : sample_high_income = 100)
  (h6 : sample_ordinary_with_three_plus = 50)
  (h7 : sample_high_income_with_three_plus = 70)
  (h8 : total_households = ordinary_households + high_income_households) :
  ⌊(sample_ordinary_with_three_plus : ℚ) / sample_ordinary * ordinary_households +
   (sample_high_income_with_three_plus : ℚ) / sample_high_income * high_income_households⌋ = 5700 :=
by sorry


end NUMINAMATH_CALUDE_estimate_households_with_three_plus_houses_l53_5332


namespace NUMINAMATH_CALUDE_correct_system_l53_5331

/-- Represents the money owned by person A -/
def money_A : ℝ := sorry

/-- Represents the money owned by person B -/
def money_B : ℝ := sorry

/-- Condition 1: If B gives half of his money to A, then A will have 50 units of money -/
axiom condition1 : money_A + (1/2 : ℝ) * money_B = 50

/-- Condition 2: If A gives two-thirds of his money to B, then B will have 50 units of money -/
axiom condition2 : (2/3 : ℝ) * money_A + money_B = 50

/-- The system of equations correctly represents the given conditions -/
theorem correct_system : 
  (money_A + (1/2 : ℝ) * money_B = 50) ∧ 
  ((2/3 : ℝ) * money_A + money_B = 50) := by sorry

end NUMINAMATH_CALUDE_correct_system_l53_5331


namespace NUMINAMATH_CALUDE_sqrt_product_division_problem_statement_l53_5344

theorem sqrt_product_division (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt a * Real.sqrt b / (1 / Real.sqrt c) = c → a * b = c :=
by sorry

theorem problem_statement : 
  Real.sqrt 2 * Real.sqrt 3 / (1 / Real.sqrt 6) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_division_problem_statement_l53_5344


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l53_5348

/-- Given a line with slope m and a point P(x1, y1), prove that the line
    y = mx + (y1 - mx1) passes through P and is parallel to the original line. -/
theorem parallel_line_through_point (m x1 y1 : ℝ) :
  let L2 : ℝ → ℝ := λ x => m * x + (y1 - m * x1)
  (L2 x1 = y1) ∧ (∀ x y, y = L2 x ↔ y - y1 = m * (x - x1)) := by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l53_5348


namespace NUMINAMATH_CALUDE_customer_payment_percentage_l53_5353

theorem customer_payment_percentage (savings_percentage : ℝ) (payment_percentage : ℝ) :
  savings_percentage = 14.5 →
  payment_percentage = 100 - savings_percentage →
  payment_percentage = 85.5 :=
by sorry

end NUMINAMATH_CALUDE_customer_payment_percentage_l53_5353


namespace NUMINAMATH_CALUDE_integral_of_polynomial_l53_5374

theorem integral_of_polynomial : ∫ (x : ℝ) in (0)..(2), (3*x^2 + 4*x^3) = 24 := by sorry

end NUMINAMATH_CALUDE_integral_of_polynomial_l53_5374


namespace NUMINAMATH_CALUDE_min_copies_discount_proof_l53_5310

/-- The minimum number of photocopies required for a discount -/
def min_copies_for_discount : ℕ := 160

/-- The cost of one photocopy in dollars -/
def cost_per_copy : ℚ := 2 / 100

/-- The discount rate offered -/
def discount_rate : ℚ := 25 / 100

/-- The total savings when ordering 160 copies -/
def total_savings : ℚ := 80 / 100

theorem min_copies_discount_proof :
  (min_copies_for_discount : ℚ) * cost_per_copy * (1 - discount_rate) =
  (min_copies_for_discount : ℚ) * cost_per_copy - total_savings :=
by sorry

end NUMINAMATH_CALUDE_min_copies_discount_proof_l53_5310


namespace NUMINAMATH_CALUDE_apple_cost_proof_l53_5304

theorem apple_cost_proof (original_price : ℝ) (price_increase : ℝ) (family_size : ℕ) (pounds_per_person : ℝ) : 
  original_price = 1.6 → 
  price_increase = 0.25 → 
  family_size = 4 → 
  pounds_per_person = 2 → 
  (original_price + original_price * price_increase) * (family_size : ℝ) * pounds_per_person = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_proof_l53_5304


namespace NUMINAMATH_CALUDE_simplify_negative_fraction_power_l53_5354

theorem simplify_negative_fraction_power :
  (-1 / 343 : ℝ) ^ (-3/5 : ℝ) = -343 := by sorry

end NUMINAMATH_CALUDE_simplify_negative_fraction_power_l53_5354


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l53_5330

theorem three_digit_divisible_by_nine :
  ∀ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
    n % 10 = 2 ∧          -- Units digit is 2
    n / 100 = 4 ∧         -- Hundreds digit is 4
    n % 9 = 0             -- Divisible by 9
    → n = 432 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l53_5330


namespace NUMINAMATH_CALUDE_not_P_sufficient_for_not_q_l53_5366

-- Define the propositions P and q
def P (x : ℝ) : Prop := |5*x - 2| > 3
def q (x : ℝ) : Prop := 1 / (x^2 + 4*x - 5) > 0

-- State the theorem
theorem not_P_sufficient_for_not_q :
  (∀ x : ℝ, ¬(P x) → ¬(q x)) ∧
  ¬(∀ x : ℝ, ¬(q x) → ¬(P x)) :=
sorry

end NUMINAMATH_CALUDE_not_P_sufficient_for_not_q_l53_5366


namespace NUMINAMATH_CALUDE_blue_to_red_light_ratio_l53_5379

/-- Proves that the ratio of blue lights to red lights is 3:1 given the problem conditions -/
theorem blue_to_red_light_ratio :
  let initial_white_lights : ℕ := 59
  let red_lights : ℕ := 12
  let green_lights : ℕ := 6
  let remaining_to_buy : ℕ := 5
  let total_colored_lights : ℕ := initial_white_lights - remaining_to_buy
  let blue_lights : ℕ := total_colored_lights - (red_lights + green_lights)
  (blue_lights : ℚ) / red_lights = 3 / 1 := by
sorry

end NUMINAMATH_CALUDE_blue_to_red_light_ratio_l53_5379


namespace NUMINAMATH_CALUDE_probability_a_equals_one_l53_5311

theorem probability_a_equals_one (a b c : ℕ+) (sum_constraint : a + b + c = 6) :
  (Finset.filter (fun x => x.1 = 1) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card /
  (Finset.filter (fun x => x.1 + x.2.1 + x.2.2 = 6) (Finset.product (Finset.range 6) (Finset.product (Finset.range 6) (Finset.range 6)))).card
  = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_a_equals_one_l53_5311
