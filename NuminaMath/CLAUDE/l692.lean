import Mathlib

namespace NUMINAMATH_CALUDE_set_operations_l692_69254

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1 ∨ x ≤ -3}
def B : Set ℝ := {x | -4 < x ∧ x < 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -4 < x ∧ x ≤ -3}) ∧
  (A ∪ B = {x | x < 0 ∨ x ≥ 1}) ∧
  (A ∪ (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l692_69254


namespace NUMINAMATH_CALUDE_sam_has_46_balloons_l692_69281

/-- Given the number of red balloons Fred and Dan have, and the total number of red balloons,
    calculate the number of red balloons Sam has. -/
def sams_balloons (fred_balloons dan_balloons total_balloons : ℕ) : ℕ :=
  total_balloons - (fred_balloons + dan_balloons)

/-- Theorem stating that under the given conditions, Sam has 46 red balloons. -/
theorem sam_has_46_balloons :
  sams_balloons 10 16 72 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_46_balloons_l692_69281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l692_69217

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence b d →
  increasing_sequence b →
  b 3 * b 4 = 21 →
  b 2 * b 5 = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l692_69217


namespace NUMINAMATH_CALUDE_complex_number_problem_l692_69239

theorem complex_number_problem (a : ℝ) (z : ℂ) : 
  z = (1 + a * Complex.I) / Complex.I → 
  z.re = 1 → 
  a = 1 ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l692_69239


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l692_69272

theorem discount_percentage_proof (total_cost : ℝ) (num_shirts : ℕ) (discounted_price : ℝ) :
  total_cost = 60 ∧ num_shirts = 3 ∧ discounted_price = 12 →
  (1 - discounted_price / (total_cost / num_shirts)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l692_69272


namespace NUMINAMATH_CALUDE_nancy_antacid_intake_l692_69213

/-- Represents the number of antacids Nancy takes per day for different food types -/
structure AntacidIntake where
  indian : ℕ
  mexican : ℕ
  other : ℝ

/-- Represents Nancy's weekly food consumption -/
structure WeeklyConsumption where
  indian_days : ℕ
  mexican_days : ℕ

/-- Calculates Nancy's monthly antacid intake based on her eating habits -/
def monthly_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) : ℝ :=
  4 * (intake.indian * consumption.indian_days + intake.mexican * consumption.mexican_days) +
  intake.other * (30 - 4 * (consumption.indian_days + consumption.mexican_days))

/-- Theorem stating Nancy's antacid intake for non-Indian and non-Mexican food days -/
theorem nancy_antacid_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) :
  intake.indian = 3 →
  intake.mexican = 2 →
  consumption.indian_days = 3 →
  consumption.mexican_days = 2 →
  monthly_intake intake consumption = 60 →
  intake.other = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_nancy_antacid_intake_l692_69213


namespace NUMINAMATH_CALUDE_simplify_expression_l692_69283

theorem simplify_expression (y : ℝ) : 3 * y - 5 * y^2 + 12 - (7 - 3 * y + 5 * y^2) = -10 * y^2 + 6 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l692_69283


namespace NUMINAMATH_CALUDE_smallest_ends_in_7_div_by_5_l692_69298

def ends_in_7 (n : ℕ) : Prop := n % 10 = 7

theorem smallest_ends_in_7_div_by_5 : 
  ∃ (n : ℕ), n > 0 ∧ ends_in_7 n ∧ n % 5 = 0 ∧ 
  ∀ (m : ℕ), m > 0 → ends_in_7 m → m % 5 = 0 → m ≥ n :=
by
  use 37
  sorry

end NUMINAMATH_CALUDE_smallest_ends_in_7_div_by_5_l692_69298


namespace NUMINAMATH_CALUDE_self_repeating_mod_1000_numbers_l692_69270

/-- A three-digit number that remains unchanged when raised to any natural power modulo 1000 -/
def self_repeating_mod_1000 (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∀ k : ℕ, k > 0 → n^k ≡ n [MOD 1000]

/-- The only three-digit numbers that remain unchanged when raised to any natural power modulo 1000 are 625 and 376 -/
theorem self_repeating_mod_1000_numbers :
  ∀ n : ℕ, self_repeating_mod_1000 n ↔ n = 625 ∨ n = 376 := by sorry

end NUMINAMATH_CALUDE_self_repeating_mod_1000_numbers_l692_69270


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l692_69233

/-- Sequence c_n satisfying the given recurrence relation -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

/-- Definition of a_n based on c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ (k : ℤ), a n = k^2 := by
  sorry


end NUMINAMATH_CALUDE_a_is_perfect_square_l692_69233


namespace NUMINAMATH_CALUDE_no_97_points_l692_69221

/-- Represents the score on a test with the given scoring system -/
structure TestScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total : correct + unanswered + incorrect = 20

/-- Calculates the total points for a given TestScore -/
def calculatePoints (score : TestScore) : ℕ :=
  5 * score.correct + score.unanswered

/-- Theorem stating that 97 points is not possible on the test -/
theorem no_97_points : ¬ ∃ (score : TestScore), calculatePoints score = 97 := by
  sorry


end NUMINAMATH_CALUDE_no_97_points_l692_69221


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l692_69277

/-- Given information about students taking geometry and history classes,
    prove that the number of students taking either geometry or history
    but not both is 35. -/
theorem students_taking_one_subject (total_geometry : ℕ)
                                    (both_subjects : ℕ)
                                    (history_only : ℕ)
                                    (h1 : total_geometry = 40)
                                    (h2 : both_subjects = 20)
                                    (h3 : history_only = 15) :
  (total_geometry - both_subjects) + history_only = 35 := by
  sorry


end NUMINAMATH_CALUDE_students_taking_one_subject_l692_69277


namespace NUMINAMATH_CALUDE_tennis_cost_calculation_l692_69262

/-- Represents the cost of tennis equipment under different purchasing options -/
def TennisCost (x : ℕ) : Prop :=
  let racketPrice : ℕ := 200
  let ballPrice : ℕ := 40
  let racketQuantity : ℕ := 20
  let option1Cost : ℕ := racketPrice * racketQuantity + ballPrice * (x - racketQuantity)
  let option2Cost : ℕ := (racketPrice * racketQuantity + ballPrice * x) * 9 / 10
  x > 20 ∧ option1Cost = 40 * x + 3200 ∧ option2Cost = 3600 + 36 * x

theorem tennis_cost_calculation (x : ℕ) : TennisCost x := by
  sorry

end NUMINAMATH_CALUDE_tennis_cost_calculation_l692_69262


namespace NUMINAMATH_CALUDE_f_properties_l692_69246

def f (x : ℝ) := x^2 + x - 2

theorem f_properties :
  (∀ y : ℝ, y ∈ Set.Icc (-1) 1 → ∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x > f y) ∧
  (∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x = -9/4 ∧ ∀ y : ℝ, y ∈ Set.Ico (-1) 1 → f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l692_69246


namespace NUMINAMATH_CALUDE_some_number_value_l692_69204

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 45 * 49) : n = 3 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l692_69204


namespace NUMINAMATH_CALUDE_rational_sum_and_power_integers_l692_69251

theorem rational_sum_and_power_integers (n : ℕ) : 
  (Odd n) ↔ 
  (∃ (a b : ℚ), 
    0 < a ∧ 0 < b ∧ 
    ¬(∃ (i : ℤ), a = i) ∧ ¬(∃ (j : ℤ), b = j) ∧
    (∃ (k : ℤ), (a + b : ℚ) = k) ∧ 
    (∃ (l : ℤ), (a^n + b^n : ℚ) = l)) :=
sorry

end NUMINAMATH_CALUDE_rational_sum_and_power_integers_l692_69251


namespace NUMINAMATH_CALUDE_total_balloons_l692_69235

-- Define the number of red and green balloons
def red_balloons : ℕ := 8
def green_balloons : ℕ := 9

-- Theorem stating that the total number of balloons is 17
theorem total_balloons : red_balloons + green_balloons = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l692_69235


namespace NUMINAMATH_CALUDE_cube_root_cubed_equals_identity_l692_69265

theorem cube_root_cubed_equals_identity (x : ℝ) : (x^(1/3))^3 = x := by sorry

end NUMINAMATH_CALUDE_cube_root_cubed_equals_identity_l692_69265


namespace NUMINAMATH_CALUDE_yard_length_theorem_l692_69273

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 14 trees planted at equal distances, 
    with one tree at each end, and a distance of 21 meters between consecutive trees, 
    is equal to 273 meters. -/
theorem yard_length_theorem : 
  yard_length 14 21 = 273 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_theorem_l692_69273


namespace NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_quadratic_l692_69276

-- Theorem 1
theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

-- Theorem 2
theorem factorize_quadratic (x : ℝ) : 2*x^2 - 20*x + 50 = 2*(x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorize_difference_of_squares_factorize_quadratic_l692_69276


namespace NUMINAMATH_CALUDE_john_remaining_money_l692_69253

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- The amount John has saved in base 8 --/
def john_savings : ℕ := 5555

/-- The cost of the round-trip airline ticket in base 10 --/
def ticket_cost : ℕ := 1200

/-- The amount John will have left after buying the ticket --/
def remaining_money : ℕ := base8_to_base10 john_savings - ticket_cost

theorem john_remaining_money :
  remaining_money = 1725 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l692_69253


namespace NUMINAMATH_CALUDE_root_product_theorem_l692_69293

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ x₆ : ℂ) : 
  (x₁^6 - x₁^3 + 1 = 0) → 
  (x₂^6 - x₂^3 + 1 = 0) → 
  (x₃^6 - x₃^3 + 1 = 0) → 
  (x₄^6 - x₄^3 + 1 = 0) → 
  (x₅^6 - x₅^3 + 1 = 0) → 
  (x₆^6 - x₆^3 + 1 = 0) → 
  (x₁^2 - 3) * (x₂^2 - 3) * (x₃^2 - 3) * (x₄^2 - 3) * (x₅^2 - 3) * (x₆^2 - 3) = 757 := by
  sorry

end NUMINAMATH_CALUDE_root_product_theorem_l692_69293


namespace NUMINAMATH_CALUDE_only_lottery_is_random_l692_69296

-- Define the events
def event_A := "No moisture, seed germination"
def event_B := "At least 2 people out of 367 have the same birthday"
def event_C := "Melting of ice at -1°C under standard pressure"
def event_D := "Xiao Ying bought a lottery ticket and won a 5 million prize"

-- Define a predicate for random events
def is_random_event (e : String) : Prop := sorry

-- Theorem stating that only event_D is a random event
theorem only_lottery_is_random :
  ¬(is_random_event event_A) ∧
  ¬(is_random_event event_B) ∧
  ¬(is_random_event event_C) ∧
  is_random_event event_D :=
by sorry

end NUMINAMATH_CALUDE_only_lottery_is_random_l692_69296


namespace NUMINAMATH_CALUDE_regression_equation_equivalence_l692_69299

/-- Conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Conversion factor from pounds to kilograms -/
def pound_to_kg : ℝ := 0.45

/-- Slope of the regression equation in imperial units (pounds per inch) -/
def imperial_slope : ℝ := 4

/-- Intercept of the regression equation in imperial units (pounds) -/
def imperial_intercept : ℝ := -130

/-- Predicted weight in imperial units (pounds) given height in inches -/
def predicted_weight_imperial (height : ℝ) : ℝ :=
  imperial_slope * height + imperial_intercept

/-- Predicted weight in metric units (kg) given height in cm -/
def predicted_weight_metric (height : ℝ) : ℝ :=
  0.72 * height - 58.5

theorem regression_equation_equivalence :
  ∀ height_inch : ℝ,
  let height_cm := height_inch * inch_to_cm
  predicted_weight_metric height_cm =
    predicted_weight_imperial height_inch * pound_to_kg := by
  sorry

end NUMINAMATH_CALUDE_regression_equation_equivalence_l692_69299


namespace NUMINAMATH_CALUDE_am_gm_inequality_l692_69206

theorem am_gm_inequality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≤ b) :
  (b - a)^3 / (8 * b) > (a + b) / 2 - Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l692_69206


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l692_69241

/-- The equation is quadratic if and only if the exponent of x in the first term is 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The coefficient of the highest degree term should not be zero -/
def coeff_nonzero (m : ℝ) : Prop := m ≠ 2

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ coeff_nonzero m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l692_69241


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l692_69282

/-- Definition of the ellipse C -/
def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focal length -/
def focal_length (c : ℝ) : Prop :=
  c = 2

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse 2 (-Real.sqrt 2) a b

/-- Definition of the line intersecting the ellipse -/
def intersecting_line (x y m : ℝ) : Prop :=
  y = x + m

/-- Definition of the circle where the midpoint lies -/
def midpoint_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- Main theorem -/
theorem ellipse_intersection_theorem (a b c m : ℝ) : 
  a > b ∧ b > 0 ∧
  focal_length c ∧
  point_on_ellipse a b →
  (∀ x y, ellipse x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    ellipse A.1 A.2 a b ∧
    ellipse B.1 B.2 a b ∧
    intersecting_line A.1 A.2 m ∧
    intersecting_line B.1 B.2 m ∧
    midpoint_circle ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) →
    m = 3 * Real.sqrt 5 / 5 ∨ m = -3 * Real.sqrt 5 / 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l692_69282


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l692_69200

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 14*x - 2*y + 14 = 0

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (3, -2)
def center2 : ℝ × ℝ := (7, 1)
def radius1 : ℝ := 1
def radius2 : ℝ := 6

-- Theorem stating that the circles are internally tangent
theorem circles_internally_tangent :
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  d = radius2 - radius1 :=
sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l692_69200


namespace NUMINAMATH_CALUDE_triangle_problem_l692_69291

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_problem (a b c : ℝ) 
  (h : Real.sqrt (8 - a) + Real.sqrt (a - 8) = abs (c - 17) + b^2 - 30*b + 225) :
  a = 8 ∧ 
  b = 15 ∧ 
  c = 17 ∧
  triangle_inequality a b c ∧
  a^2 + b^2 = c^2 ∧
  a + b + c = 40 ∧
  a * b / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l692_69291


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l692_69267

theorem complex_number_quadrant : 
  let z : ℂ := (2 - 3*I) / (I^3)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l692_69267


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l692_69285

/-- Given that N(4,9) is the midpoint of CD and C has coordinates (10,5),
    prove that the sum of the coordinates of D is 11. -/
theorem midpoint_coordinate_sum :
  let N : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (10, 5)
  ∀ D : ℝ × ℝ,
  (N.1 = (C.1 + D.1) / 2 ∧ N.2 = (C.2 + D.2) / 2) →
  D.1 + D.2 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l692_69285


namespace NUMINAMATH_CALUDE_fred_paper_count_l692_69257

theorem fred_paper_count (initial_sheets received_sheets given_sheets : ℕ) :
  initial_sheets + received_sheets - given_sheets =
  initial_sheets + received_sheets - given_sheets :=
by sorry

end NUMINAMATH_CALUDE_fred_paper_count_l692_69257


namespace NUMINAMATH_CALUDE_polynomial_existence_l692_69275

theorem polynomial_existence : ∃ (f : ℝ → ℝ), 
  (∃ (a b c d e g h : ℝ), ∀ x, f x = a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + g*x + h) ∧ 
  (∀ x, f (Real.sin x) + f (Real.cos x) = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l692_69275


namespace NUMINAMATH_CALUDE_lcm_problem_l692_69208

theorem lcm_problem (n : ℕ+) : ∃ n, 
  n > 0 ∧ 
  216 % n = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≤ 9 ∧
  Nat.lcm (Nat.lcm (Nat.lcm 8 24) 36) n = 216 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l692_69208


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l692_69229

open Complex

theorem min_value_of_complex_expression (z : ℂ) (h : abs (z - (3 - 3*I)) = 3) :
  ∃ (min : ℝ), min = 100 ∧ ∀ (w : ℂ), abs (w - (3 - 3*I)) = 3 → 
    abs (w - (2 + 2*I))^2 + abs (w - (6 - 6*I))^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l692_69229


namespace NUMINAMATH_CALUDE_sum_of_angles_with_tangent_roots_l692_69249

theorem sum_of_angles_with_tangent_roots (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (∃ x y : Real, x^2 - 5*x + 6 = 0 ∧ y^2 - 5*y + 6 = 0 ∧ Real.tan α = x ∧ Real.tan β = y) →
  α + β = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_with_tangent_roots_l692_69249


namespace NUMINAMATH_CALUDE_p_costs_more_after_10_years_l692_69202

/-- Represents the yearly price increase in paise -/
structure PriceIncrease where
  p : ℚ  -- Price increase for commodity P
  q : ℚ  -- Price increase for commodity Q

/-- Represents the initial prices in rupees -/
structure InitialPrice where
  p : ℚ  -- Initial price for commodity P
  q : ℚ  -- Initial price for commodity Q

/-- Calculates the year when commodity P costs 40 paise more than commodity Q -/
def yearWhenPCostsMoreThanQ (increase : PriceIncrease) (initial : InitialPrice) : ℕ :=
  sorry

/-- The theorem stating that P costs 40 paise more than Q after 10 years -/
theorem p_costs_more_after_10_years 
  (increase : PriceIncrease) 
  (initial : InitialPrice) 
  (h1 : increase.p = 40/100) 
  (h2 : increase.q = 15/100) 
  (h3 : initial.p = 420/100) 
  (h4 : initial.q = 630/100) : 
  yearWhenPCostsMoreThanQ increase initial = 10 := by sorry

end NUMINAMATH_CALUDE_p_costs_more_after_10_years_l692_69202


namespace NUMINAMATH_CALUDE_largest_square_area_l692_69245

theorem largest_square_area (X Y Z : ℝ) (h_right_angle : X^2 + Y^2 = Z^2)
  (h_equal_sides : X = Y) (h_sum_areas : X^2 + Y^2 + Z^2 + (2*Y)^2 = 650) :
  Z^2 = 650/3 := by
sorry

end NUMINAMATH_CALUDE_largest_square_area_l692_69245


namespace NUMINAMATH_CALUDE_fraction_equality_l692_69238

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l692_69238


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l692_69288

/-- The general form equation of a line passing through (1, 1) with slope -3 -/
theorem line_equation_through_point_with_slope :
  ∃ (A B C : ℝ), A ≠ 0 ∨ B ≠ 0 ∧
  (∀ x y : ℝ, A * x + B * y + C = 0 ↔ y - 1 = -3 * (x - 1)) ∧
  A = 3 ∧ B = 1 ∧ C = -4 := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l692_69288


namespace NUMINAMATH_CALUDE_bobby_candy_consumption_l692_69209

/-- The number of candies Bobby eats per day from Monday through Friday -/
def daily_candies : ℕ := 2

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of candies in each packet -/
def candies_per_packet : ℕ := 18

/-- The number of weeks it takes Bobby to finish the packets -/
def num_weeks : ℕ := 3

/-- The number of candies Bobby eats on weekend days -/
def weekend_candies : ℕ := 1

theorem bobby_candy_consumption :
  daily_candies * 5 * num_weeks + weekend_candies * 2 * num_weeks = num_packets * candies_per_packet :=
sorry

end NUMINAMATH_CALUDE_bobby_candy_consumption_l692_69209


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l692_69227

theorem sqrt_equation_solution :
  ∃! x : ℝ, 4 * x - 3 ≥ 0 ∧ Real.sqrt (4 * x - 3) + 16 / Real.sqrt (4 * x - 3) = 8 :=
by
  -- The unique solution is x = 19/4
  use 19/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l692_69227


namespace NUMINAMATH_CALUDE_inequality_equivalence_l692_69214

theorem inequality_equivalence (x : ℝ) : 
  -1 < (x^2 - 12*x + 35) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 12*x + 35) / (x^2 - 4*x + 8) < 1 ↔ 
  x > 27/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l692_69214


namespace NUMINAMATH_CALUDE_tan_alpha_value_l692_69232

theorem tan_alpha_value (α : Real) : 
  Real.tan (π / 4 + α) = 1 / 2 → Real.tan α = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l692_69232


namespace NUMINAMATH_CALUDE_paraboloid_surface_area_l692_69264

/-- The paraboloid of revolution --/
def paraboloid (x y z : ℝ) : Prop := 3 * y = x^2 + z^2

/-- The bounding plane --/
def bounding_plane (y : ℝ) : Prop := y = 6

/-- The first octant --/
def first_octant (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

/-- The surface area of the part of the paraboloid --/
noncomputable def surface_area : ℝ := sorry

/-- The theorem stating the surface area of the specified part of the paraboloid --/
theorem paraboloid_surface_area :
  surface_area = 39 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_paraboloid_surface_area_l692_69264


namespace NUMINAMATH_CALUDE_series_sum_equals_two_l692_69236

/-- Given a real number k > 1 such that the infinite sum of (7n-3)/k^n from n=1 to infinity equals 2,
    prove that k = 2 + (3√2)/2 -/
theorem series_sum_equals_two (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : k = 2 + 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_two_l692_69236


namespace NUMINAMATH_CALUDE_barn_paint_area_l692_69242

/-- Calculates the total area to be painted for a rectangular barn with given dimensions and windows. -/
def total_paint_area (width length height window_width window_height window_count : ℕ) : ℕ :=
  let wall_area_1 := 2 * (width * height)
  let wall_area_2 := 2 * (length * height - window_width * window_height * window_count)
  let ceiling_area := width * length
  2 * (wall_area_1 + wall_area_2) + ceiling_area

/-- The total area to be painted for the given barn is 780 sq yd. -/
theorem barn_paint_area :
  total_paint_area 12 15 6 2 3 2 = 780 :=
by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l692_69242


namespace NUMINAMATH_CALUDE_min_value_problem_l692_69230

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) : 
  1/x + 1/(3*y) ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 2 ∧ 1/x₀ + 1/(3*y₀) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l692_69230


namespace NUMINAMATH_CALUDE_distance_AB_bounds_l692_69261

/-- Given six points in space with specific distance relationships, 
    prove that the distance between two of the points lies within a certain range. -/
theorem distance_AB_bounds 
  (A B C D E F : EuclideanSpace ℝ (Fin 3)) 
  (h1 : dist A C = 10 ∧ dist A D = 10 ∧ dist B E = 10 ∧ dist B F = 10)
  (h2 : dist A E = 12 ∧ dist A F = 12 ∧ dist B C = 12 ∧ dist B D = 12)
  (h3 : dist C D = 11 ∧ dist E F = 11)
  (h4 : dist C E = 5 ∧ dist D F = 5) : 
  8.8 < dist A B ∧ dist A B < 19.2 := by
  sorry


end NUMINAMATH_CALUDE_distance_AB_bounds_l692_69261


namespace NUMINAMATH_CALUDE_smallest_z_satisfying_conditions_l692_69258

theorem smallest_z_satisfying_conditions : ∃ (z : ℕ), z = 10 ∧ 
  (∀ (x y : ℕ), x > 0 ∧ y > 0 →
    (27 ^ z) * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
    x + y = z ∧
    x * y < z ^ 2) ∧
  (∀ (z' : ℕ), z' < z →
    ¬(∃ (x y : ℕ), x > 0 ∧ y > 0 ∧
      (27 ^ z') * (5 ^ x) > (3 ^ 24) * (2 ^ y) ∧
      x + y = z' ∧
      x * y < z' ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_z_satisfying_conditions_l692_69258


namespace NUMINAMATH_CALUDE_edwards_initial_money_l692_69247

/-- Given that Edward spent $16 and has $2 left, his initial amount of money was $18. -/
theorem edwards_initial_money :
  ∀ (initial spent left : ℕ),
    spent = 16 →
    left = 2 →
    initial = spent + left →
    initial = 18 := by
  sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l692_69247


namespace NUMINAMATH_CALUDE_root_in_interval_l692_69290

def f (x : ℝ) := x^3 - x - 1

theorem root_in_interval :
  ∃ r ∈ Set.Icc 1.25 1.5, f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l692_69290


namespace NUMINAMATH_CALUDE_rectangle_difference_l692_69212

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  breadth : ℕ

/-- The perimeter of the rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.breadth)

/-- The area of the rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.breadth

/-- The difference between length and breadth -/
def Rectangle.difference (r : Rectangle) : ℕ := r.length - r.breadth

theorem rectangle_difference (r : Rectangle) :
  r.perimeter = 266 ∧ r.area = 4290 → r.difference = 23 := by
  sorry

#eval Rectangle.difference { length := 78, breadth := 55 }

end NUMINAMATH_CALUDE_rectangle_difference_l692_69212


namespace NUMINAMATH_CALUDE_percentage_increase_l692_69205

theorem percentage_increase (initial final : ℝ) (h : initial > 0) :
  let increase := (final - initial) / initial * 100
  initial = 200 ∧ final = 250 → increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l692_69205


namespace NUMINAMATH_CALUDE_min_gumballs_for_three_same_color_l692_69295

/-- Represents the colors of gumballs in the machine -/
inductive GumballColor
| Red
| Blue
| White
| Green

/-- Represents the gumball machine -/
structure GumballMachine where
  red : Nat
  blue : Nat
  white : Nat
  green : Nat

/-- Returns the minimum number of gumballs needed to guarantee 3 of the same color -/
def minGumballsForThreeSameColor (machine : GumballMachine) : Nat :=
  sorry

/-- Theorem stating that for the given gumball machine, 
    the minimum number of gumballs needed to guarantee 3 of the same color is 8 -/
theorem min_gumballs_for_three_same_color :
  let machine : GumballMachine := { red := 13, blue := 5, white := 1, green := 9 }
  minGumballsForThreeSameColor machine = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_gumballs_for_three_same_color_l692_69295


namespace NUMINAMATH_CALUDE_school_sections_theorem_l692_69271

/-- Calculates the total number of sections needed in a school with given constraints -/
def totalSections (numBoys numGirls : ℕ) (maxBoysPerSection maxGirlsPerSection : ℕ) (numSubjects : ℕ) : ℕ :=
  let boySections := (numBoys + maxBoysPerSection - 1) / maxBoysPerSection * numSubjects
  let girlSections := (numGirls + maxGirlsPerSection - 1) / maxGirlsPerSection * numSubjects
  boySections + girlSections

/-- Theorem stating that the total number of sections is 87 under the given constraints -/
theorem school_sections_theorem :
  totalSections 408 192 24 16 3 = 87 := by
  sorry

end NUMINAMATH_CALUDE_school_sections_theorem_l692_69271


namespace NUMINAMATH_CALUDE_perfect_correlation_l692_69201

/-- A sample point in a 2D plane -/
structure SamplePoint where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The correlation coefficient -/
def correlationCoefficient (points : List SamplePoint) : ℝ :=
  sorry

/-- Theorem: If all sample points lie on a straight line with non-zero slope, 
    then the correlation coefficient R^2 is 1 -/
theorem perfect_correlation 
  (points : List SamplePoint) 
  (line : Line) 
  (h1 : line.slope ≠ 0) 
  (h2 : ∀ p ∈ points, p.y = line.slope * p.x + line.intercept) : 
  correlationCoefficient points = 1 :=
sorry

end NUMINAMATH_CALUDE_perfect_correlation_l692_69201


namespace NUMINAMATH_CALUDE_work_completion_time_l692_69292

/-- 
Given:
- A can complete the work in 60 days
- A and B together can complete the work in 15 days

Prove that B can complete the work alone in 20 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = 60) (h2 : 1 / a + 1 / b = 1 / 15) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l692_69292


namespace NUMINAMATH_CALUDE_no_one_common_tangent_l692_69216

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles have different radii --/
def hasDifferentRadii (c1 c2 : Circle) : Prop :=
  c1.radius ≠ c2.radius

/-- Counts the number of common tangents between two circles --/
def commonTangentsCount (c1 c2 : Circle) : ℕ := sorry

/-- Theorem stating that two circles with different radii cannot have exactly one common tangent --/
theorem no_one_common_tangent (c1 c2 : Circle) 
  (h : hasDifferentRadii c1 c2) : 
  commonTangentsCount c1 c2 ≠ 1 := by sorry

end NUMINAMATH_CALUDE_no_one_common_tangent_l692_69216


namespace NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l692_69294

theorem square_sum_difference (n : ℕ) : 
  (2*n + 1)^2 - (2*n - 1)^2 + (2*n - 1)^2 - (2*n - 3)^2 + 
  (2*n - 3)^2 - (2*n - 5)^2 + (2*n - 5)^2 - (2*n - 7)^2 + 
  (2*n - 7)^2 - (2*n - 9)^2 + (2*n - 9)^2 - (2*n - 11)^2 = 24 * n :=
by
  sorry

theorem specific_square_sum_difference : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_specific_square_sum_difference_l692_69294


namespace NUMINAMATH_CALUDE_custom_op_example_l692_69260

-- Define the custom operation
def custom_op (a b : Int) : Int := a * (b + 1) + a * b

-- State the theorem
theorem custom_op_example : custom_op (-3) 4 = -27 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l692_69260


namespace NUMINAMATH_CALUDE_common_roots_product_l692_69243

-- Define the polynomials
def p (C : ℝ) (x : ℝ) : ℝ := x^3 + C*x^2 - 20
def q (D : ℝ) (x : ℝ) : ℝ := x^3 + D*x - 80

-- Define the theorem
theorem common_roots_product (C D : ℝ) :
  ∃ (r₁ r₂ : ℝ) (a b c : ℕ),
    (p C r₁ = 0 ∧ q D r₁ = 0) ∧
    (p C r₂ = 0 ∧ q D r₂ = 0) ∧
    r₁ ≠ r₂ ∧
    (r₁ * r₂ = a * (c ^ (1 / b : ℝ))) ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 25 :=
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l692_69243


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l692_69207

theorem polar_to_cartesian (M : ℝ × ℝ) :
  M.1 = 3 ∧ M.2 = π / 6 →
  (M.1 * Real.cos M.2 = 3 * Real.sqrt 3 / 2) ∧
  (M.1 * Real.sin M.2 = 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l692_69207


namespace NUMINAMATH_CALUDE_volume_of_enlarged_box_l692_69279

/-- Represents a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Theorem: Volume of enlarged box -/
theorem volume_of_enlarged_box (box : Box) 
  (volume_eq : box.l * box.w * box.h = 5000)
  (surface_area_eq : 2 * (box.l * box.w + box.w * box.h + box.l * box.h) = 1800)
  (edge_sum_eq : 4 * (box.l + box.w + box.h) = 210) :
  (box.l + 2) * (box.w + 2) * (box.h + 2) = 7018 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_enlarged_box_l692_69279


namespace NUMINAMATH_CALUDE_polynomial_division_l692_69218

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (x^4 - 3*x^2) / x^2 = x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l692_69218


namespace NUMINAMATH_CALUDE_max_soda_bottles_problem_l692_69224

/-- Represents the maximum number of soda bottles that can be consumed given a certain amount of money, cost per bottle, and exchange rate for empty bottles. -/
def max_soda_bottles (total_money : ℚ) (cost_per_bottle : ℚ) (exchange_rate : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given 30 yuan, a soda cost of 2.5 yuan per bottle, and the ability to exchange 3 empty bottles for 1 new bottle, the maximum number of soda bottles that can be consumed is 18. -/
theorem max_soda_bottles_problem :
  max_soda_bottles 30 2.5 3 = 18 :=
sorry

end NUMINAMATH_CALUDE_max_soda_bottles_problem_l692_69224


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l692_69219

theorem perfect_square_trinomial (x y : ℝ) :
  x^2 - x*y + (1/4)*y^2 = (x - (1/2)*y)^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l692_69219


namespace NUMINAMATH_CALUDE_log_equation_solution_l692_69289

theorem log_equation_solution (x : ℝ) (h : x > 0) : 
  2 * (Real.log x / Real.log 6) = 1 - (Real.log 3 / Real.log 6) ↔ x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l692_69289


namespace NUMINAMATH_CALUDE_polynomial_value_at_2_l692_69234

-- Define the polynomial coefficients
def a₃ : ℝ := 7
def a₂ : ℝ := 3
def a₁ : ℝ := -5
def a₀ : ℝ := 11

-- Define the point at which to evaluate the polynomial
def x : ℝ := 2

-- Define Horner's method for a cubic polynomial
def horner_cubic (a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₃ * x + a₂) * x + a₁) * x + a₀

-- Theorem statement
theorem polynomial_value_at_2 :
  horner_cubic a₃ a₂ a₁ a₀ x = 69 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_2_l692_69234


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l692_69223

theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 50 →           -- One of the equal angles is 50°
  max α (max β γ) = 80 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l692_69223


namespace NUMINAMATH_CALUDE_expression_values_l692_69255

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let expr := a / abs a + b / abs b + c / abs c + d / abs d + (a * b * c * d) / abs (a * b * c * d)
  expr = 5 ∨ expr = 1 ∨ expr = -3 ∨ expr = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l692_69255


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l692_69286

theorem divisible_by_twelve (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l692_69286


namespace NUMINAMATH_CALUDE_multiplication_and_exponentiation_l692_69226

theorem multiplication_and_exponentiation : 121 * (5^4) = 75625 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_and_exponentiation_l692_69226


namespace NUMINAMATH_CALUDE_no_integer_solutions_l692_69244

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ),
    x^6 + x^3 + x^3 * y + y = 147^157 ∧
    x^3 + x^3 * y + y^2 + y + z^9 = 157^1177 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l692_69244


namespace NUMINAMATH_CALUDE_income_data_mean_difference_l692_69278

/-- The difference between the mean of incorrect data and the mean of actual data -/
theorem income_data_mean_difference (T : ℝ) : 
  (T + 1200000) / 500 - (T + 120000) / 500 = 2160 := by sorry

end NUMINAMATH_CALUDE_income_data_mean_difference_l692_69278


namespace NUMINAMATH_CALUDE_combined_degrees_l692_69266

theorem combined_degrees (summer_degrees jolly_degrees : ℕ) : 
  summer_degrees = 150 → 
  summer_degrees = jolly_degrees + 5 → 
  summer_degrees + jolly_degrees = 295 := by
sorry

end NUMINAMATH_CALUDE_combined_degrees_l692_69266


namespace NUMINAMATH_CALUDE_min_value_of_t_l692_69211

theorem min_value_of_t (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → 1 / x + 1 / y ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 2 * y = 1 ∧ 1 / x + 1 / y = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_t_l692_69211


namespace NUMINAMATH_CALUDE_product_cube_l692_69222

theorem product_cube (a b c : ℕ+) (h : a * b * c = 180) : (a * b) ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_product_cube_l692_69222


namespace NUMINAMATH_CALUDE_m_plus_n_equals_one_l692_69225

theorem m_plus_n_equals_one (m n : ℤ) (h : |m - 2| + (n + 1)^2 = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_n_equals_one_l692_69225


namespace NUMINAMATH_CALUDE_mean_height_is_68_25_l692_69231

def heights : List ℕ := [57, 59, 62, 64, 64, 65, 65, 68, 69, 70, 71, 73, 75, 75, 77, 78]

theorem mean_height_is_68_25 : 
  let total_height : ℕ := heights.sum
  let num_players : ℕ := heights.length
  (total_height : ℚ) / num_players = 68.25 := by
sorry

end NUMINAMATH_CALUDE_mean_height_is_68_25_l692_69231


namespace NUMINAMATH_CALUDE_lcm_16_24_l692_69210

theorem lcm_16_24 : Nat.lcm 16 24 = 48 := by sorry

end NUMINAMATH_CALUDE_lcm_16_24_l692_69210


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l692_69220

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l692_69220


namespace NUMINAMATH_CALUDE_expression_simplification_l692_69297

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (((x + 2) / (x - 2) + (x - x^2) / (x^2 - 4*x + 4)) / ((x - 4) / (x - 2))) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l692_69297


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l692_69269

theorem consecutive_negative_integers_product_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_sum_l692_69269


namespace NUMINAMATH_CALUDE_repair_time_30_workers_l692_69228

/-- Represents the time taken to complete a road repair job given the number of workers -/
def repair_time (num_workers : ℕ) : ℚ :=
  3 * 45 / num_workers

/-- Proves that 30 workers would take 4.5 days to complete the road repair -/
theorem repair_time_30_workers :
  repair_time 30 = 4.5 := by sorry

end NUMINAMATH_CALUDE_repair_time_30_workers_l692_69228


namespace NUMINAMATH_CALUDE_dislike_sector_angle_l692_69259

-- Define the ratios for the four categories
def ratio_extremely_like : ℕ := 6
def ratio_like : ℕ := 9
def ratio_somewhat_like : ℕ := 2
def ratio_dislike : ℕ := 1

-- Define the total ratio
def total_ratio : ℕ := ratio_extremely_like + ratio_like + ratio_somewhat_like + ratio_dislike

-- Define the central angle of the dislike sector
def central_angle_dislike : ℚ := (ratio_dislike : ℚ) / (total_ratio : ℚ) * 360

-- Theorem statement
theorem dislike_sector_angle :
  central_angle_dislike = 20 := by sorry

end NUMINAMATH_CALUDE_dislike_sector_angle_l692_69259


namespace NUMINAMATH_CALUDE_margarets_mean_score_l692_69284

def scores : List ℝ := [78, 81, 85, 87, 90, 92]

theorem margarets_mean_score 
  (h1 : scores.length = 6)
  (h2 : ∃ (cyprian_scores margaret_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    margaret_scores.length = 2 ∧ 
    cyprian_scores ++ margaret_scores = scores)
  (h3 : ∃ (cyprian_scores : List ℝ), 
    cyprian_scores.length = 4 ∧ 
    cyprian_scores.sum / cyprian_scores.length = 84) :
  ∃ (margaret_scores : List ℝ), 
    margaret_scores.length = 2 ∧ 
    margaret_scores.sum / margaret_scores.length = 88.5 := by
sorry

end NUMINAMATH_CALUDE_margarets_mean_score_l692_69284


namespace NUMINAMATH_CALUDE_class_test_average_l692_69280

theorem class_test_average (class_size : ℝ) (h_positive : class_size > 0) : 
  let group_a := 0.15 * class_size
  let group_b := 0.50 * class_size
  let group_c := class_size - group_a - group_b
  let score_a := 100
  let score_b := 78
  ∃ score_c : ℝ,
    (group_a * score_a + group_b * score_b + group_c * score_c) / class_size = 76.05 ∧
    score_c = 63 :=
by sorry

end NUMINAMATH_CALUDE_class_test_average_l692_69280


namespace NUMINAMATH_CALUDE_eleanor_childless_descendants_l692_69263

/-- Eleanor's family structure -/
structure EleanorFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Eleanor's daughters and granddaughters with no daughters -/
def childless_descendants (f : EleanorFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Eleanor's daughters and granddaughters with no daughters -/
theorem eleanor_childless_descendants :
  ∀ f : EleanorFamily,
  f.daughters = 8 →
  f.total_descendants = 43 →
  f.daughters_with_children * 7 = f.total_descendants - f.daughters →
  childless_descendants f = 38 := by
  sorry

end NUMINAMATH_CALUDE_eleanor_childless_descendants_l692_69263


namespace NUMINAMATH_CALUDE_larger_number_proof_l692_69252

theorem larger_number_proof (L S : ℕ) (h1 : L - S = 1375) (h2 : L = 6 * S + 15) : L = 1647 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l692_69252


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_l692_69248

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ 4, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_l692_69248


namespace NUMINAMATH_CALUDE_amy_video_files_l692_69287

/-- Proves that Amy had 21 video files initially -/
theorem amy_video_files :
  ∀ (initial_music_files deleted_files remaining_files : ℕ),
    initial_music_files = 4 →
    deleted_files = 23 →
    remaining_files = 2 →
    initial_music_files + (deleted_files + remaining_files) - initial_music_files = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_video_files_l692_69287


namespace NUMINAMATH_CALUDE_reflection_about_y_axis_example_l692_69274

/-- Given a point in 3D space, return its reflection about the y-axis -/
def reflect_about_y_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, -z)

/-- The reflection of point (3, -2, 1) about the y-axis is (-3, -2, -1) -/
theorem reflection_about_y_axis_example : 
  reflect_about_y_axis (3, -2, 1) = (-3, -2, -1) := by
  sorry

#check reflection_about_y_axis_example

end NUMINAMATH_CALUDE_reflection_about_y_axis_example_l692_69274


namespace NUMINAMATH_CALUDE_ellipse_parabola_triangle_area_l692_69250

/-- Definition of the ellipse C₁ -/
def ellipse (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

/-- Definition of the parabola C₂ -/
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

/-- The focus F of the parabola, which is also the vertex of the ellipse -/
def F : ℝ × ℝ := (0, 2)

/-- Definition of a point being on the ellipse -/
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

/-- Definition of two vectors being orthogonal -/
def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

/-- Definition of a line being tangent to the parabola -/
def tangent_to_parabola (P Q : ℝ × ℝ) : Prop :=
  ∃ k m : ℝ, (∀ x y : ℝ, y = k * x + m → (x^2 = 8 * y ↔ x = P.1 ∧ y = P.2))

theorem ellipse_parabola_triangle_area :
  ∀ P Q : ℝ × ℝ,
  on_ellipse P → on_ellipse Q →
  orthogonal (P.1 - F.1, P.2 - F.2) (Q.1 - F.1, Q.2 - F.2) →
  tangent_to_parabola P Q →
  P ≠ F → Q ≠ F → P ≠ Q →
  abs ((P.1 - F.1) * (Q.2 - F.2) - (P.2 - F.2) * (Q.1 - F.1)) / 2 = 18 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_triangle_area_l692_69250


namespace NUMINAMATH_CALUDE_candle_burning_theorem_l692_69240

theorem candle_burning_theorem (n : ℕ) (h : n > 0) :
  (∃ k : ℕ, k > 0 ∧ n * k = n * (n + 1) / 2) → Odd n :=
by
  sorry

#check candle_burning_theorem

end NUMINAMATH_CALUDE_candle_burning_theorem_l692_69240


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l692_69268

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 1) * x^2 + (a - 1) * x + 1 > 0) ↔ a ∈ Set.Icc 1 5 \ {5} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l692_69268


namespace NUMINAMATH_CALUDE_middle_term_value_l692_69215

/-- An arithmetic sequence with 9 terms -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ n, a n = a₁ + (n - 1) * d

theorem middle_term_value
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_sum_first_4 : (a 1) + (a 2) + (a 3) + (a 4) = 3)
  (h_sum_last_3 : (a 7) + (a 8) + (a 9) = 4) :
  a 5 = 19 / 148 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_value_l692_69215


namespace NUMINAMATH_CALUDE_v_2008_value_l692_69256

-- Define the sequence v_n
def v : ℕ → ℕ 
| n => sorry  -- The exact definition would be complex to write out

-- Define the function g(n) for the last term in a group with n terms
def g (n : ℕ) : ℕ := 2 * n^2 - 3 * n + 2

-- Define the function for the total number of terms up to and including group n
def totalTerms (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to prove
theorem v_2008_value : v 2008 = 7618 := by sorry

end NUMINAMATH_CALUDE_v_2008_value_l692_69256


namespace NUMINAMATH_CALUDE_min_tests_correct_l692_69237

/-- Represents the result of a test between two balls -/
inductive TestResult
| Same
| Different

/-- Represents a ball -/
structure Ball :=
  (id : Nat)
  (metal : Bool)  -- True for copper, False for zinc

/-- Represents a test between two balls -/
structure Test :=
  (ball1 : Ball)
  (ball2 : Ball)
  (result : TestResult)

/-- The minimum number of tests required to determine the material of each ball -/
def min_tests (n : Nat) (copper_count : Nat) (zinc_count : Nat) : Nat :=
  n - 1

theorem min_tests_correct (n : Nat) (copper_count : Nat) (zinc_count : Nat) 
  (h1 : n = 99)
  (h2 : copper_count = 50)
  (h3 : zinc_count = 49)
  (h4 : copper_count + zinc_count = n) :
  min_tests n copper_count zinc_count = 98 := by
  sorry

#eval min_tests 99 50 49

end NUMINAMATH_CALUDE_min_tests_correct_l692_69237


namespace NUMINAMATH_CALUDE_number_of_factors_of_power_l692_69203

theorem number_of_factors_of_power (b n : ℕ+) (hb : b = 8) (hn : n = 15) :
  (Finset.range ((n * (Nat.factorization b).sum (fun _ e => e)) + 1)).card = 46 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_power_l692_69203
