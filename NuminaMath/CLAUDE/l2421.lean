import Mathlib

namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_l2421_242166

-- Define the function f(x) = x^3 - x + a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x + a

-- Define the property of being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem not_necessary_not_sufficient :
  ¬(∀ a : ℝ, (a^2 - a = 0 ↔ is_increasing (f a))) :=
sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_l2421_242166


namespace NUMINAMATH_CALUDE_games_last_year_l2421_242131

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 25

/-- The difference in games attended between last year and this year -/
def games_difference : ℕ := 11

/-- Theorem stating the number of games Fred attended last year -/
theorem games_last_year : games_this_year + games_difference = 36 := by
  sorry

end NUMINAMATH_CALUDE_games_last_year_l2421_242131


namespace NUMINAMATH_CALUDE_only_setA_cannot_form_triangle_l2421_242181

-- Define a function to check if three line segments can form a triangle
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the sets of line segments
def setA : List ℝ := [4, 4, 9]
def setB : List ℝ := [3, 5, 6]
def setC : List ℝ := [6, 8, 10]
def setD : List ℝ := [5, 12, 13]

-- State the theorem
theorem only_setA_cannot_form_triangle :
  (¬ canFormTriangle setA[0] setA[1] setA[2]) ∧
  (canFormTriangle setB[0] setB[1] setB[2]) ∧
  (canFormTriangle setC[0] setC[1] setC[2]) ∧
  (canFormTriangle setD[0] setD[1] setD[2]) := by
  sorry

end NUMINAMATH_CALUDE_only_setA_cannot_form_triangle_l2421_242181


namespace NUMINAMATH_CALUDE_sum_of_ages_l2421_242148

/-- Given the ages of siblings and cousins, calculate the sum of their ages. -/
theorem sum_of_ages (juliet ralph maggie nicky lucy lily alex : ℕ) : 
  juliet = 10 ∧ 
  juliet = maggie + 3 ∧ 
  ralph = juliet + 2 ∧ 
  nicky * 2 = ralph ∧ 
  lucy = ralph + 1 ∧ 
  lily = ralph + 1 ∧ 
  alex + 5 = lucy → 
  maggie + ralph + nicky + lucy + lily + alex = 59 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2421_242148


namespace NUMINAMATH_CALUDE_solve_equation_l2421_242154

theorem solve_equation (b : ℚ) (h : b + b / 4 = 10 / 4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2421_242154


namespace NUMINAMATH_CALUDE_difference_local_face_value_65793_l2421_242133

/-- The difference between the local value and face value of a digit in a numeral -/
def local_face_value_difference (numeral : ℕ) (digit : ℕ) (place : ℕ) : ℕ :=
  digit * (10 ^ place) - digit

/-- The hundreds place in a decimal number system -/
def hundreds_place : ℕ := 2

theorem difference_local_face_value_65793 :
  local_face_value_difference 65793 7 hundreds_place = 693 := by
  sorry

end NUMINAMATH_CALUDE_difference_local_face_value_65793_l2421_242133


namespace NUMINAMATH_CALUDE_movies_to_watch_l2421_242145

theorem movies_to_watch (total_movies : ℕ) (watched_movies : ℕ) 
  (h1 : total_movies = 35) (h2 : watched_movies = 18) :
  total_movies - watched_movies = 17 := by
  sorry

end NUMINAMATH_CALUDE_movies_to_watch_l2421_242145


namespace NUMINAMATH_CALUDE_pentagon_perimeter_division_l2421_242101

/-- Given a regular pentagon with perimeter 125 and side length 25,
    prove that the perimeter divided by the side length equals 5. -/
theorem pentagon_perimeter_division (perimeter : ℝ) (side_length : ℝ) :
  perimeter = 125 →
  side_length = 25 →
  perimeter / side_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_division_l2421_242101


namespace NUMINAMATH_CALUDE_water_in_large_bottle_sport_formulation_l2421_242185

/-- Represents a flavored drink formulation -/
structure Formulation where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation of the drink -/
def standard_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : Formulation :=
  { flavoring := 1, corn_syrup := 4, water := 60 }

/-- The amount of corn syrup in the large bottle (in ounces) -/
def large_bottle_corn_syrup : ℚ := 8

/-- Theorem stating the amount of water in the large bottle of sport formulation -/
theorem water_in_large_bottle_sport_formulation :
  (large_bottle_corn_syrup * sport_formulation.water) / sport_formulation.corn_syrup = 120 := by
  sorry

end NUMINAMATH_CALUDE_water_in_large_bottle_sport_formulation_l2421_242185


namespace NUMINAMATH_CALUDE_sqrt_identity_l2421_242125

theorem sqrt_identity (a b : ℝ) (h : a^2 ≥ b ∧ a ≥ 0 ∧ b ≥ 0) :
  (∀ (s : Bool), Real.sqrt (a + (if s then 1 else -1) * Real.sqrt b) = 
    Real.sqrt ((a + Real.sqrt (a^2 - b)) / 2) + 
    (if s then 1 else -1) * Real.sqrt ((a - Real.sqrt (a^2 - b)) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l2421_242125


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2421_242156

theorem complex_expression_equality : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)
  M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2421_242156


namespace NUMINAMATH_CALUDE_intersection_condition_l2421_242114

theorem intersection_condition (a : ℝ) : 
  (∃ x y : ℝ, a * x + 2 * y = 3 ∧ x + (a - 1) * y = 1) → a ≠ 2 ∧ 
  ∃ b : ℝ, b ≠ 2 ∧ ¬(∃ x y : ℝ, b * x + 2 * y = 3 ∧ x + (b - 1) * y = 1) :=
by sorry

#check intersection_condition

end NUMINAMATH_CALUDE_intersection_condition_l2421_242114


namespace NUMINAMATH_CALUDE_frog_hop_ratio_l2421_242118

/-- The number of hops taken by each frog -/
structure FrogHops where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the frog hopping problem -/
def frog_hop_conditions (h : FrogHops) : Prop :=
  ∃ (m : ℕ), 
    h.first = m * h.second ∧
    h.second = 2 * h.third ∧
    h.first + h.second + h.third = 99 ∧
    h.second = 18

/-- The theorem stating the ratio of hops between the first and second frog -/
theorem frog_hop_ratio (h : FrogHops) (hc : frog_hop_conditions h) :
  h.first / h.second = 4 := by
  sorry

#check frog_hop_ratio

end NUMINAMATH_CALUDE_frog_hop_ratio_l2421_242118


namespace NUMINAMATH_CALUDE_function_property_l2421_242157

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_periodic_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = -f x

def is_monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_periodic_neg_one f) 
  (h3 : is_monotone_increasing_on f (-1) 0) :
  f 2 > f (Real.sqrt 2) ∧ f (Real.sqrt 2) > f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2421_242157


namespace NUMINAMATH_CALUDE_tan_315_degrees_l2421_242178

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_degrees_l2421_242178


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2421_242170

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 15 - 20 * I ∧ Complex.abs w = 5 → Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2421_242170


namespace NUMINAMATH_CALUDE_xy_range_l2421_242192

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = x * y * (2 * Real.log x + Real.log y)) : 
  x * y ≥ Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_xy_range_l2421_242192


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2421_242117

/-- Given the ages of three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 17 →  -- total age is 17
  b = 6 →  -- b is 6 years old
  b = 2 * c  -- the ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2421_242117


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2421_242151

/-- The circle described by (x-a)^2 + (y-a)^2 = 4 always has two distinct points
    at distance 1 from the origin if and only if a is in the given range -/
theorem circle_intersection_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + (y₁ - a)^2 = 4 ∧ 
    (x₂ - a)^2 + (y₂ - a)^2 = 4 ∧
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁, y₁) ≠ (x₂, y₂)) ↔ 
  (a ∈ Set.Ioo (-3 * Real.sqrt 2 / 2) (-Real.sqrt 2 / 2) ∪ 
       Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_range_l2421_242151


namespace NUMINAMATH_CALUDE_sun_division_problem_l2421_242134

/-- Prove that the total amount is 105 given the conditions of the sun division problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (y = 0.45 * x) →  -- For each rupee x gets, y gets 45 paisa
  (z = 0.30 * x) →  -- For each rupee x gets, z gets 30 paisa
  (y = 27) →        -- y's share is Rs. 27
  (x + y + z = 105) -- The total amount is 105
  := by sorry

end NUMINAMATH_CALUDE_sun_division_problem_l2421_242134


namespace NUMINAMATH_CALUDE_find_a_and_b_l2421_242152

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 :=
by sorry

end NUMINAMATH_CALUDE_find_a_and_b_l2421_242152


namespace NUMINAMATH_CALUDE_acid_mixture_concentration_l2421_242193

/-- Calculates the final acid concentration when replacing part of a solution with another -/
def finalAcidConcentration (initialConcentration replacementConcentration : ℚ) 
  (replacementFraction : ℚ) : ℚ :=
  (1 - replacementFraction) * initialConcentration + 
  replacementFraction * replacementConcentration

/-- Proves that replacing half of a 50% acid solution with a 30% acid solution results in a 40% solution -/
theorem acid_mixture_concentration : 
  finalAcidConcentration (1/2) (3/10) (1/2) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_acid_mixture_concentration_l2421_242193


namespace NUMINAMATH_CALUDE_optimal_pill_count_l2421_242161

/-- Represents the vitamin content of a single pill -/
structure PillContent where
  vitaminA : ℕ
  vitaminB : ℕ
  vitaminC : ℕ

/-- Represents the recommended weekly servings for vitamins -/
structure WeeklyRecommendation where
  vitaminA : ℕ
  vitaminB : ℕ
  vitaminC : ℕ

/-- Checks if the given number of pills meets or exceeds all vitamin requirements -/
def meetsRequirements (pillContent : PillContent) (weeklyRecommendation : WeeklyRecommendation) (numPills : ℕ) : Prop :=
  numPills * pillContent.vitaminA ≥ weeklyRecommendation.vitaminA ∧
  numPills * pillContent.vitaminB ≥ weeklyRecommendation.vitaminB ∧
  numPills * pillContent.vitaminC ≥ weeklyRecommendation.vitaminC

theorem optimal_pill_count 
  (pillContent : PillContent)
  (weeklyRecommendation : WeeklyRecommendation)
  (h1 : pillContent.vitaminA = 50)
  (h2 : pillContent.vitaminB = 20)
  (h3 : pillContent.vitaminC = 10)
  (h4 : weeklyRecommendation.vitaminA = 1400)
  (h5 : weeklyRecommendation.vitaminB = 700)
  (h6 : weeklyRecommendation.vitaminC = 280) :
  meetsRequirements pillContent weeklyRecommendation 35 :=
by sorry

end NUMINAMATH_CALUDE_optimal_pill_count_l2421_242161


namespace NUMINAMATH_CALUDE_line_passes_through_point_min_length_AB_min_dot_product_l2421_242142

-- Define the line l: mx + y - 1 - 2m = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 - 2 * m = 0

-- Define the circle O: x^2 + y^2 = r^2
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem 1: The line l passes through the point (2, 1) for all m
theorem line_passes_through_point (m : ℝ) : line_l m 2 1 := by sorry

-- Theorem 2: When r = 4, the minimum length of AB is 2√11
theorem min_length_AB (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 11 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min_length := by sorry

-- Theorem 3: When r = 4, the minimum value of OA · OB is -16
theorem min_dot_product (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_dot : ℝ, min_dot = -16 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  A.1 * B.1 + A.2 * B.2 ≥ min_dot := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_min_length_AB_min_dot_product_l2421_242142


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2421_242106

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/12. -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2421_242106


namespace NUMINAMATH_CALUDE_probability_three_same_color_l2421_242183

def total_balls : ℕ := 27
def green_balls : ℕ := 15
def white_balls : ℕ := 12

def probability_same_color : ℚ := 3 / 13

theorem probability_three_same_color :
  let total_combinations := Nat.choose total_balls 3
  let green_combinations := Nat.choose green_balls 3
  let white_combinations := Nat.choose white_balls 3
  (green_combinations + white_combinations : ℚ) / total_combinations = probability_same_color :=
by sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l2421_242183


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2421_242186

theorem arithmetic_calculations :
  (23 + (-13) + (-17) + 8 = 1) ∧
  (-2^3 - (1 + 0.5) / (1/3) * (-3) = 11/2) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2421_242186


namespace NUMINAMATH_CALUDE_perimeter_of_figure_C_l2421_242167

/-- Represents the dimensions of a rectangle in terms of small rectangles -/
structure RectDimension where
  width : ℕ
  height : ℕ

/-- Calculates the perimeter of a figure given its dimensions and the size of small rectangles -/
def perimeter (dim : RectDimension) (x y : ℝ) : ℝ :=
  2 * (dim.width * x + dim.height * y)

theorem perimeter_of_figure_C (x y : ℝ) : 
  perimeter ⟨6, 1⟩ x y = 56 →
  perimeter ⟨2, 3⟩ x y = 56 →
  perimeter ⟨1, 3⟩ x y = 40 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_figure_C_l2421_242167


namespace NUMINAMATH_CALUDE_difference_ones_zeros_253_l2421_242121

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem difference_ones_zeros_253 :
  let binary := binary_representation 253
  let ones := count_ones binary
  let zeros := count_zeros binary
  ones - zeros = 6 :=
sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_253_l2421_242121


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l2421_242130

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 27 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l2421_242130


namespace NUMINAMATH_CALUDE_dental_bill_theorem_l2421_242173

/-- The cost of a dental filling -/
def filling_cost : ℕ := sorry

/-- The cost of a dental cleaning -/
def cleaning_cost : ℕ := 70

/-- The cost of a tooth extraction -/
def extraction_cost : ℕ := 290

/-- The total bill for dental services -/
def total_bill : ℕ := 5 * filling_cost

theorem dental_bill_theorem : 
  filling_cost = 120 ∧ 
  total_bill = cleaning_cost + 2 * filling_cost + extraction_cost := by
  sorry

end NUMINAMATH_CALUDE_dental_bill_theorem_l2421_242173


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l2421_242111

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the probability space for two dice rolls
def TwoDice : Type := Die × Die

-- Define event A: sum of dice is even
def eventA (roll : TwoDice) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

-- Define event B: sum of dice is less than 7
def eventB (roll : TwoDice) : Prop :=
  roll.1.val + 1 + roll.2.val + 1 < 7

-- Define the probability measure
def P : Set TwoDice → ℝ := sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {roll : TwoDice | eventB roll ∧ eventA roll} / P {roll : TwoDice | eventA roll} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l2421_242111


namespace NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l2421_242113

-- Define the cost of one piece of gum in cents
def cost_per_gum : ℕ := 5

-- Define the number of pieces of gum
def num_gums : ℕ := 2000

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_gums_in_dollars : 
  (cost_per_gum * num_gums) / cents_per_dollar = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_gums_in_dollars_l2421_242113


namespace NUMINAMATH_CALUDE_meaningful_range_l2421_242143

def is_meaningful (x : ℝ) : Prop :=
  x + 1 ≥ 0 ∧ x ≠ 0

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -1 ∧ x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l2421_242143


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2421_242129

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds_bound : hundreds < 10
  h_tens_bound : tens < 10
  h_ones_bound : ones < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The reverse of a three-digit number -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.ones + 10 * n.tens + n.hundreds

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber,
    (n.hundreds + n.tens + n.ones = 10) ∧
    (n.tens = n.hundreds + n.ones) ∧
    (n.reverse = n.value + 99) ∧
    (n.value = 253) := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2421_242129


namespace NUMINAMATH_CALUDE_children_after_addition_l2421_242176

-- Define the event parameters
def total_guests : Nat := 80
def num_men : Nat := 40
def num_women : Nat := num_men / 2
def added_children : Nat := 10

-- Theorem statement
theorem children_after_addition : 
  total_guests - (num_men + num_women) + added_children = 30 := by
  sorry

end NUMINAMATH_CALUDE_children_after_addition_l2421_242176


namespace NUMINAMATH_CALUDE_f_composition_result_l2421_242165

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3 else 2^x

theorem f_composition_result : f (f (1/9)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_result_l2421_242165


namespace NUMINAMATH_CALUDE_john_work_hours_l2421_242122

def hours_per_day : ℕ := 8
def start_day : ℕ := 3
def end_day : ℕ := 8

def total_days : ℕ := end_day - start_day

theorem john_work_hours : hours_per_day * total_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_john_work_hours_l2421_242122


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l2421_242120

theorem cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) :
  selling_price = 720 →
  num_balls_sold = 20 →
  num_balls_loss = 5 →
  ∃ (cost_price : ℕ),
    cost_price * num_balls_sold - selling_price = cost_price * num_balls_loss ∧
    cost_price = 48 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l2421_242120


namespace NUMINAMATH_CALUDE_quadratic_ratio_l2421_242196

/-- The quadratic function f(x) = x^2 + 1600x + 1607 -/
def f (x : ℝ) : ℝ := x^2 + 1600*x + 1607

/-- The constant b in the completed square form (x+b)^2 + c -/
def b : ℝ := 800

/-- The constant c in the completed square form (x+b)^2 + c -/
def c : ℝ := -638393

/-- Theorem stating that c/b equals -797.99125 for the given quadratic -/
theorem quadratic_ratio : c / b = -797.99125 := by sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2421_242196


namespace NUMINAMATH_CALUDE_allowance_increase_l2421_242109

theorem allowance_increase (base_amount : ℝ) (middle_school_extra : ℝ) (percentage_increase : ℝ) : 
  let middle_school_allowance := base_amount + middle_school_extra
  let senior_year_allowance := middle_school_allowance * (1 + percentage_increase / 100)
  base_amount = 8 ∧ middle_school_extra = 2 ∧ percentage_increase = 150 →
  senior_year_allowance - 2 * middle_school_allowance = 5 := by sorry

end NUMINAMATH_CALUDE_allowance_increase_l2421_242109


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2421_242136

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x + 2| > 3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2421_242136


namespace NUMINAMATH_CALUDE_quadratic_polynomial_divisibility_l2421_242116

theorem quadratic_polynomial_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p)
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_divisible : ∀ x : ℕ, x > 0 → p ∣ (a * x^2 + b * x + c)) :
  (p = 2 ∧ a + b + c = 4) ∨ (p > 2 ∧ a + b + c = 3 * p) := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_divisibility_l2421_242116


namespace NUMINAMATH_CALUDE_probability_problem_l2421_242108

theorem probability_problem (P : Set α → ℝ) (A B : Set α)
  (h1 : P (A ∩ B) = 1/6)
  (h2 : P (Aᶜ) = 2/3)
  (h3 : P B = 1/2) :
  (P (A ∩ B) ≠ 0 ∧ P A * P B = P (A ∩ B)) :=
by sorry

end NUMINAMATH_CALUDE_probability_problem_l2421_242108


namespace NUMINAMATH_CALUDE_sum_of_coordinates_on_inverse_graph_l2421_242199

-- Define the function f
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem sum_of_coordinates_on_inverse_graph : 
  (f 2 = 6) → -- This condition is derived from (2,3) being on y=f(x)/2
  ∃ x y : ℝ, (y = 2 * (f⁻¹ x)) ∧ (x + y = 10) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_on_inverse_graph_l2421_242199


namespace NUMINAMATH_CALUDE_min_value_fraction_l2421_242144

theorem min_value_fraction (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  1 / x + 4 / (1 - x) ≥ 9 ∧
  (1 / x + 4 / (1 - x) = 9 ↔ x = 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2421_242144


namespace NUMINAMATH_CALUDE_photo_border_area_l2421_242194

/-- The area of the border around a rectangular photograph -/
theorem photo_border_area (photo_height photo_width border_width : ℝ) 
  (h_height : photo_height = 9)
  (h_width : photo_width = 12)
  (h_border : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 162 := by
  sorry

#check photo_border_area

end NUMINAMATH_CALUDE_photo_border_area_l2421_242194


namespace NUMINAMATH_CALUDE_angle_A_range_l2421_242150

theorem angle_A_range (a b c : ℝ) (A : ℝ) :
  a = 2 →
  b = 2 * Real.sqrt 2 →
  c^2 = a^2 + b^2 - 2*a*b * Real.cos A →
  0 < A ∧ A ≤ π/4 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_range_l2421_242150


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l2421_242182

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 8*x - 1 = 0 ↔ (x - 4)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l2421_242182


namespace NUMINAMATH_CALUDE_x_equals_six_l2421_242158

theorem x_equals_six (a b x : ℝ) 
  (h1 : 2^a = x) 
  (h2 : 3^b = x) 
  (h3 : 1/a + 1/b = 1) : 
  x = 6 := by sorry

end NUMINAMATH_CALUDE_x_equals_six_l2421_242158


namespace NUMINAMATH_CALUDE_part_one_part_two_l2421_242127

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A

-- Theorem for part I
theorem part_one (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : t.a = 2 * Real.sqrt 6) 
  (h3 : t.b = 3) : 
  t.c = 5 := by sorry

-- Theorem for part II
theorem part_two (t : Triangle) 
  (h1 : validTriangle t) 
  (h2 : t.C = Real.pi / 2) : 
  t.B = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2421_242127


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l2421_242190

theorem smallest_multiple_of_45_and_75_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 45 ∣ n ∧ 75 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 45 ∣ m → 75 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  use 225
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l2421_242190


namespace NUMINAMATH_CALUDE_ball_game_proof_l2421_242123

theorem ball_game_proof (total_balls : ℕ) (red_prob_1 : ℚ) (black_prob_2 red_prob_2 : ℚ) 
  (green_balls : ℕ) (red_prob_3 : ℚ) :
  total_balls = 10 →
  red_prob_1 = 1 →
  black_prob_2 = 1/2 →
  red_prob_2 = 1/2 →
  green_balls = 2 →
  red_prob_3 = 7/10 →
  ∃ (black_balls : ℕ), black_balls = 1 := by
  sorry

#check ball_game_proof

end NUMINAMATH_CALUDE_ball_game_proof_l2421_242123


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_property_l2421_242115

/-- Given a geometric sequence with first term a₁ and common ratio q,
    the sum of the 4th, 5th, and 6th terms squared equals the product of
    the sum of the 1st, 2nd, and 3rd terms and the sum of the 7th, 8th, and 9th terms. -/
theorem geometric_sequence_sum_property (a₁ q : ℝ) :
  (a₁ * q^3 + a₁ * q^4 + a₁ * q^5)^2 = (a₁ + a₁ * q + a₁ * q^2) * (a₁ * q^6 + a₁ * q^7 + a₁ * q^8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_property_l2421_242115


namespace NUMINAMATH_CALUDE_root_of_quadratic_equation_l2421_242164

theorem root_of_quadratic_equation : 
  ∃ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_root_of_quadratic_equation_l2421_242164


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l2421_242155

/-- Represents the number of valid arrangements for n coins where no three consecutive coins are face to face to face -/
def validArrangements : Nat → Nat
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => validArrangements (n + 2) + validArrangements (n + 1) + validArrangements n

/-- The number of ways to choose 5 positions out of 10 for gold coins -/
def colorDistributions : Nat := Nat.choose 10 5

/-- The total number of distinguishable arrangements of 5 gold and 5 silver coins
    with the given face-to-face constraint -/
def totalArrangements : Nat := colorDistributions * validArrangements 10

theorem coin_stack_arrangements :
  totalArrangements = 69048 := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l2421_242155


namespace NUMINAMATH_CALUDE_sum_of_squares_nonzero_iff_one_nonzero_l2421_242179

theorem sum_of_squares_nonzero_iff_one_nonzero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_nonzero_iff_one_nonzero_l2421_242179


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_twice_sum_iff_zero_l2421_242110

theorem sqrt_sum_squares_equals_twice_sum_iff_zero (a b : ℝ) : 
  a ≥ 0 → b ≥ 0 → (Real.sqrt (a^2 + b^2) = 2 * (a + b) ↔ a = 0 ∧ b = 0) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_twice_sum_iff_zero_l2421_242110


namespace NUMINAMATH_CALUDE_dice_probabilities_l2421_242184

-- Define the sample space and events
def Ω : ℕ := 216  -- Total number of possible outcomes
def A : ℕ := 120  -- Number of outcomes where all dice show different numbers
def AB : ℕ := 75  -- Number of outcomes satisfying both A and B

-- Define the probabilities
def P_AB : ℚ := AB / Ω
def P_A : ℚ := A / Ω
def P_B_given_A : ℚ := P_AB / P_A

-- State the theorem
theorem dice_probabilities :
  P_AB = 75 / 216 ∧ P_B_given_A = 5 / 8 := by
  sorry


end NUMINAMATH_CALUDE_dice_probabilities_l2421_242184


namespace NUMINAMATH_CALUDE_variance_binomial_8_3_4_l2421_242175

/-- The variance of a binomial distribution B(n, p) with n trials and probability p of success. -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Proof that the variance of X ~ B(8, 3/4) is 3/2 -/
theorem variance_binomial_8_3_4 :
  binomialVariance 8 (3/4) = 3/2 := by
  sorry

#check variance_binomial_8_3_4

end NUMINAMATH_CALUDE_variance_binomial_8_3_4_l2421_242175


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2421_242169

theorem inequality_equivalence (x : ℝ) :
  (7 * x - 2 < 3 * (x + 2) ↔ x < 2) ∧
  ((x - 1) / 3 ≥ (x - 3) / 12 + 1 ↔ x ≥ 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2421_242169


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2421_242168

/-- Given two functions f and g, and a condition on their composition, prove the value of b. -/
theorem composition_equation_solution (f g : ℝ → ℝ) (b : ℝ) 
  (hf : ∀ x, f x = 3 * x - 2)
  (hg : ∀ x, g x = 7 - 2 * x)
  (h_comp : g (f b) = 1) : 
  b = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2421_242168


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_iff_all_zero_l2421_242104

theorem sum_of_squares_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_iff_all_zero_l2421_242104


namespace NUMINAMATH_CALUDE_numerical_puzzle_solutions_l2421_242198

/-- A function that checks if a number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that extracts the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that extracts the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem stating the solutions to the numerical puzzle -/
theorem numerical_puzzle_solutions :
  ∀ n : ℕ, is_two_digit n →
    (∃ b v : ℕ, 
      n = b^v ∧ 
      tens_digit n ≠ ones_digit n ∧
      b = ones_digit n) ↔ 
    (n = 32 ∨ n = 36 ∨ n = 64) :=
sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solutions_l2421_242198


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l2421_242191

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 15)
  (h2 : z + x = 16)
  (h3 : x + y = 17) :
  Real.sqrt (x * y * z * (x + y + z)) = 24 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l2421_242191


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l2421_242180

/-- Represents a solid constructed from unit cubes -/
structure CubeSolid where
  bottomRow : ℕ
  middleColumn : ℕ
  leftColumns : ℕ
  leftColumnHeight : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surfaceArea (solid : CubeSolid) : ℕ :=
  let bottomArea := solid.bottomRow + 2 * (solid.bottomRow + 1)
  let middleColumnArea := 4 + (solid.middleColumn - 1)
  let leftColumnsArea := 2 * (2 * solid.leftColumnHeight + 1)
  bottomArea + middleColumnArea + leftColumnsArea

/-- The specific solid described in the problem -/
def problemSolid : CubeSolid :=
  { bottomRow := 5
  , middleColumn := 5
  , leftColumns := 2
  , leftColumnHeight := 3 }

theorem problem_solid_surface_area :
  surfaceArea problemSolid = 34 := by
  sorry

#eval surfaceArea problemSolid

end NUMINAMATH_CALUDE_problem_solid_surface_area_l2421_242180


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2421_242160

theorem complex_fraction_simplification :
  (1 - 2*Complex.I) / (2 + Complex.I) = -Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2421_242160


namespace NUMINAMATH_CALUDE_det_trig_matrix_zero_l2421_242107

/-- The determinant of a specific 3x3 matrix involving trigonometric functions is zero. -/
theorem det_trig_matrix_zero (θ φ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2 * Real.sin θ, - Real.cos θ;
                                       -2 * Real.sin θ, 0, Real.sin φ;
                                       Real.cos θ, - Real.sin φ, 0]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_det_trig_matrix_zero_l2421_242107


namespace NUMINAMATH_CALUDE_points_form_circle_l2421_242103

theorem points_form_circle :
  ∀ (t : ℝ), (∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t) →
  ∃ (r : ℝ), x^2 + y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_points_form_circle_l2421_242103


namespace NUMINAMATH_CALUDE_largest_square_size_for_rectangle_l2421_242162

theorem largest_square_size_for_rectangle (width height : ℕ) 
  (h_width : width = 63) (h_height : height = 42) :
  Nat.gcd width height = 21 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_size_for_rectangle_l2421_242162


namespace NUMINAMATH_CALUDE_cupcake_packages_l2421_242159

/-- Given the initial number of cupcakes, the number of cupcakes eaten, and the number of cupcakes per package,
    calculate the number of complete packages that can be made. -/
def packages_made (initial : ℕ) (eaten : ℕ) (per_package : ℕ) : ℕ :=
  (initial - eaten) / per_package

/-- Theorem stating that with 18 initial cupcakes, 8 eaten, and 2 cupcakes per package,
    the number of packages that can be made is 5. -/
theorem cupcake_packages : packages_made 18 8 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cupcake_packages_l2421_242159


namespace NUMINAMATH_CALUDE_kirsten_stole_14_meatballs_l2421_242188

/-- The number of meatballs Kirsten stole -/
def meatballs_stolen (initial final : ℕ) : ℕ := initial - final

/-- Proof that Kirsten stole 14 meatballs -/
theorem kirsten_stole_14_meatballs (initial final : ℕ) 
  (h_initial : initial = 25)
  (h_final : final = 11) : 
  meatballs_stolen initial final = 14 := by
  sorry

end NUMINAMATH_CALUDE_kirsten_stole_14_meatballs_l2421_242188


namespace NUMINAMATH_CALUDE_magic_square_x_value_l2421_242197

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e f g h i : ℤ)
  (row_sum : a + b + c = d + e + f ∧ d + e + f = g + h + i)
  (col_sum : a + d + g = b + e + h ∧ b + e + h = c + f + i)
  (diag_sum : a + e + i = c + e + g)

/-- The theorem stating the value of x in the given magic square -/
theorem magic_square_x_value (ms : MagicSquare) 
  (h1 : ms.a = x)
  (h2 : ms.b = 19)
  (h3 : ms.c = 96)
  (h4 : ms.d = 1) :
  x = 200 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_x_value_l2421_242197


namespace NUMINAMATH_CALUDE_equation_solution_l2421_242195

theorem equation_solution : ∃ x : ℝ, 61 + x * 12 / (180 / 3) = 62 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2421_242195


namespace NUMINAMATH_CALUDE_magic_box_solution_l2421_242132

def magic_box (a b : ℝ) : ℝ := a^2 + 2*b - 3

theorem magic_box_solution (m : ℝ) : 
  magic_box m (-3*m) = 4 ↔ m = 7 ∨ m = -1 := by sorry

end NUMINAMATH_CALUDE_magic_box_solution_l2421_242132


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_cosine_l2421_242139

open Real

theorem min_max_abs_quadratic_cosine :
  (∃ y₀ : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 + x*y₀ + cos y₀| ≤ 2)) ∧
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 + x*y + cos y| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_cosine_l2421_242139


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2421_242112

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

theorem sum_of_roots_specific_equation :
  let equation := fun x : ℝ => x^2 + 2023 * x - 2024
  let sum_of_roots := -2023
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2421_242112


namespace NUMINAMATH_CALUDE_translation_proof_l2421_242171

-- Define a translation of the complex plane
def translation (z w : ℂ) := z + w

-- Theorem statement
theorem translation_proof (t : ℂ → ℂ) :
  (∃ w : ℂ, ∀ z : ℂ, t z = translation z w) →
  (t (1 + 3*I) = 4 + 7*I) →
  (t (2 - I) = 5 + 3*I) :=
by sorry

end NUMINAMATH_CALUDE_translation_proof_l2421_242171


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l2421_242163

theorem honey_harvest_increase (last_year harvest_this_year increase : ℕ) : 
  last_year = 2479 → 
  harvest_this_year = 8564 → 
  increase = harvest_this_year - last_year → 
  increase = 6085 := by
  sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l2421_242163


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l2421_242100

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l2421_242100


namespace NUMINAMATH_CALUDE_composition_equation_solution_l2421_242137

theorem composition_equation_solution (α β : ℝ → ℝ) (h1 : ∀ x, α x = 4 * x + 9) 
  (h2 : ∀ x, β x = 9 * x + 6) (h3 : α (β x) = 8) : x = -25/36 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l2421_242137


namespace NUMINAMATH_CALUDE_largest_minus_smallest_difference_l2421_242147

def digits : List Nat := [3, 9, 6, 0, 5, 1, 7]

def largest_number (ds : List Nat) : Nat :=
  sorry

def smallest_number (ds : List Nat) : Nat :=
  sorry

theorem largest_minus_smallest_difference :
  largest_number digits - smallest_number digits = 8729631 := by
  sorry

end NUMINAMATH_CALUDE_largest_minus_smallest_difference_l2421_242147


namespace NUMINAMATH_CALUDE_double_magic_result_l2421_242141

/-- Magic box function that takes two rational numbers and produces a new rational number -/
def magic_box (a b : ℚ) : ℚ := a^2 + b + 1

/-- The result of applying the magic box function twice -/
def double_magic (a b c : ℚ) : ℚ :=
  let m := magic_box a b
  magic_box m c

/-- Theorem stating that the double application of the magic box function
    with inputs (-2, 3) and then (m, 1) results in 66 -/
theorem double_magic_result : double_magic (-2) 3 1 = 66 := by
  sorry

end NUMINAMATH_CALUDE_double_magic_result_l2421_242141


namespace NUMINAMATH_CALUDE_runners_meeting_time_l2421_242102

/-- Represents a runner with their lap time and start time offset -/
structure Runner where
  lap_time : ℕ
  start_offset : ℕ

/-- Calculates the earliest meeting time for multiple runners -/
def earliest_meeting_time (runners : List Runner) : ℕ :=
  sorry

/-- The main theorem stating the earliest meeting time for the given runners -/
theorem runners_meeting_time :
  let ben := Runner.mk 5 0
  let emily := Runner.mk 8 2
  let nick := Runner.mk 9 4
  earliest_meeting_time [ben, emily, nick] = 360 :=
sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l2421_242102


namespace NUMINAMATH_CALUDE_contest_age_fraction_l2421_242189

theorem contest_age_fraction (total_participants : ℕ) (F : ℚ) : 
  total_participants = 500 →
  (F + F / 8 : ℚ) = 0.5625 →
  F = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_contest_age_fraction_l2421_242189


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l2421_242172

theorem factorial_equation_solution :
  ∃ (n : ℕ), (4 * 3 * 2 * 1) / (Nat.factorial (4 - n)) = 24 ∧ n = 3 := by
sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l2421_242172


namespace NUMINAMATH_CALUDE_box_height_proof_l2421_242138

theorem box_height_proof (length width cube_volume num_cubes : ℝ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : cube_volume = 3)
  (h4 : num_cubes = 384) :
  (num_cubes * cube_volume) / (length * width) = 6 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l2421_242138


namespace NUMINAMATH_CALUDE_total_presents_l2421_242149

theorem total_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) (halloween : ℕ) : 
  christmas = 60 →
  birthday = 3 * easter →
  easter = christmas / 2 - 10 →
  halloween = birthday - easter →
  christmas + easter + birthday + halloween = 180 := by
sorry

end NUMINAMATH_CALUDE_total_presents_l2421_242149


namespace NUMINAMATH_CALUDE_chord_equation_l2421_242140

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 24

-- Define point P
def P : ℝ × ℝ := (1, -2)

-- Define a chord AB that passes through P and is bisected by P
structure Chord :=
  (A B : ℝ × ℝ)
  (passes_through_P : (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2)
  (on_ellipse : ellipse A.1 A.2 ∧ ellipse B.1 B.2)

-- Theorem statement
theorem chord_equation (AB : Chord) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ (x y : ℝ), 
    ((x, y) = AB.A ∨ (x, y) = AB.B) → a * x + b * y + c = 0) ∧
    a = 3 ∧ b = -2 ∧ c = -7 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l2421_242140


namespace NUMINAMATH_CALUDE_negation_of_statement_l2421_242177

theorem negation_of_statement : 
  (¬(∀ a : ℝ, a ≠ 0 → a^2 > 0)) ↔ (∃ a : ℝ, a = 0 ∧ a^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_statement_l2421_242177


namespace NUMINAMATH_CALUDE_rotation_effect_l2421_242174

-- Define a type for the shapes
inductive Shape
  | Triangle
  | Circle
  | Square
  | Pentagon

-- Define a function to represent the initial arrangement
def initial_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 0
  | Shape.Circle => 1
  | Shape.Square => 2
  | Shape.Pentagon => 3

-- Define a function to represent the position after rotation
def rotated_position (s : Shape) : ℕ :=
  match s with
  | Shape.Triangle => 1
  | Shape.Circle => 2
  | Shape.Square => 3
  | Shape.Pentagon => 0

-- Theorem stating that each shape moves to the next position after rotation
theorem rotation_effect :
  ∀ s : Shape, (rotated_position s) = ((initial_position s) + 1) % 4 :=
by sorry

end NUMINAMATH_CALUDE_rotation_effect_l2421_242174


namespace NUMINAMATH_CALUDE_goods_train_speed_l2421_242146

/-- Proves that the speed of a goods train is 100 km/h given specific conditions --/
theorem goods_train_speed (man_train_speed : ℝ) (passing_time : ℝ) (goods_train_length : ℝ) :
  man_train_speed = 80 →
  passing_time = 8 →
  goods_train_length = 400 →
  ∃ (goods_train_speed : ℝ),
    goods_train_speed = 100 ∧
    (goods_train_speed + man_train_speed) * (5 / 18) * passing_time = goods_train_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_speed_l2421_242146


namespace NUMINAMATH_CALUDE_total_cats_l2421_242119

/-- The number of cats in a pet store -/
def num_cats (white black gray : ℕ) : ℕ := white + black + gray

/-- Theorem stating that the total number of cats is 15 -/
theorem total_cats : num_cats 2 10 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_l2421_242119


namespace NUMINAMATH_CALUDE_unique_number_from_remainders_l2421_242126

theorem unique_number_from_remainders 
  (x : ℕ) 
  (h1 : x ≥ 1 ∧ x ≤ 100) 
  (r1 : ℕ) 
  (r2 : ℕ) 
  (r3 : ℕ) 
  (h2 : x % 3 = r1) 
  (h3 : x % 5 = r2) 
  (h4 : x % 7 = r3) : 
  ∃! y : ℕ, 
    y ≥ 1 ∧ y ≤ 100 ∧ 
    y % 3 = r1 ∧ 
    y % 5 = r2 ∧ 
    y % 7 = r3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_from_remainders_l2421_242126


namespace NUMINAMATH_CALUDE_rectangle_area_l2421_242135

/-- The area of a rectangle with width 10 meters and length 2 meters is 20 square meters. -/
theorem rectangle_area : 
  ∀ (width length area : ℝ), 
  width = 10 → 
  length = 2 → 
  area = width * length → 
  area = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2421_242135


namespace NUMINAMATH_CALUDE_open_box_volume_calculation_l2421_242124

/-- Given a rectangular sheet and squares cut from corners, calculates the volume of the resulting open box. -/
def openBoxVolume (sheetLength sheetWidth squareSide : ℝ) : ℝ :=
  (sheetLength - 2 * squareSide) * (sheetWidth - 2 * squareSide) * squareSide

/-- Theorem: The volume of the open box formed from a 48m x 36m sheet with 5m squares cut from corners is 9880 m³. -/
theorem open_box_volume_calculation :
  openBoxVolume 48 36 5 = 9880 := by
  sorry

#eval openBoxVolume 48 36 5

end NUMINAMATH_CALUDE_open_box_volume_calculation_l2421_242124


namespace NUMINAMATH_CALUDE_solve_system_l2421_242187

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l2421_242187


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2421_242128

theorem inequality_equivalence (x : ℝ) :
  (5 * x^2 + 20 * x - 34) / ((3 * x - 2) * (x - 5) * (x + 1)) < 2 ↔
  (-6 * x^3 + 27 * x^2 + 33 * x - 44) / ((3 * x - 2) * (x - 5) * (x + 1)) < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2421_242128


namespace NUMINAMATH_CALUDE_fundraiser_total_l2421_242105

/-- Calculates the total money raised from a class fundraiser --/
def totalMoneyRaised (numStudentsBrownies : ℕ) (numBrowniesPerStudent : ℕ) 
                     (numStudentsCookies : ℕ) (numCookiesPerStudent : ℕ)
                     (numStudentsDonuts : ℕ) (numDonutsPerStudent : ℕ)
                     (priceBrownie : ℚ) (priceCookie : ℚ) (priceDonut : ℚ) : ℚ :=
  (numStudentsBrownies * numBrowniesPerStudent : ℚ) * priceBrownie +
  (numStudentsCookies * numCookiesPerStudent : ℚ) * priceCookie +
  (numStudentsDonuts * numDonutsPerStudent : ℚ) * priceDonut

theorem fundraiser_total : 
  totalMoneyRaised 50 20 30 36 25 18 (3/2) (9/4) 3 = 5280 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l2421_242105


namespace NUMINAMATH_CALUDE_percent_palindromes_with_seven_l2421_242153

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (x y : Fin 10)

/-- Checks if a palindrome contains at least one 7 -/
def containsSeven (p : Palindrome) : Prop :=
  p.x = 7 ∨ p.y = 7

/-- The set of all palindromes between 1000 and 2000 -/
def allPalindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 7 -/
def palindromesWithSeven : Finset Palindrome :=
  sorry

theorem percent_palindromes_with_seven :
  (palindromesWithSeven.card : ℚ) / allPalindromes.card = 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_palindromes_with_seven_l2421_242153
