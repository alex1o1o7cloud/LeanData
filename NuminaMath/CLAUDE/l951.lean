import Mathlib

namespace NUMINAMATH_CALUDE_second_largest_of_5_8_4_l951_95112

def second_largest (a b c : ℕ) : ℕ :=
  if a ≥ b ∧ b ≥ c then b
  else if a ≥ c ∧ c ≥ b then c
  else if b ≥ a ∧ a ≥ c then a
  else if b ≥ c ∧ c ≥ a then c
  else if c ≥ a ∧ a ≥ b then a
  else b

theorem second_largest_of_5_8_4 : second_largest 5 8 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_of_5_8_4_l951_95112


namespace NUMINAMATH_CALUDE_complex_fraction_magnitude_l951_95110

/-- Given that i is the imaginary unit, prove that |((5+3i)/(4-i))| = √2 -/
theorem complex_fraction_magnitude : 
  Complex.abs ((5 + 3 * Complex.I) / (4 - Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_magnitude_l951_95110


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l951_95153

theorem solve_fraction_equation :
  ∀ y : ℚ, (3 / 4 : ℚ) - (5 / 8 : ℚ) = 1 / y → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l951_95153


namespace NUMINAMATH_CALUDE_smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l951_95177

def N : ℕ := 2329089562800

theorem smallest_valid_integer (k : ℕ) (h : k < N) : 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

theorem N_divisible_by_1_to_28 : 
  ∀ i ∈ Finset.range 28, N % (i + 1) = 0 :=
sorry

theorem N_not_divisible_by_29_or_30 : 
  N % 29 ≠ 0 ∨ N % 30 ≠ 0 :=
sorry

theorem N_is_smallest_valid_integer : 
  ∀ k < N, 
  (∀ i ∈ Finset.range 28, k % (i + 1) = 0) → 
  (k % 29 ≠ 0 ∨ k % 30 ≠ 0) → 
  False :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_integer_N_divisible_by_1_to_28_N_not_divisible_by_29_or_30_N_is_smallest_valid_integer_l951_95177


namespace NUMINAMATH_CALUDE_high_low_game_combinations_l951_95183

/-- Represents the types of cards in the high-low game -/
inductive CardType
| High
| Low

/-- The high-low card game -/
structure HighLowGame where
  totalCards : Nat
  highCards : Nat
  lowCards : Nat
  highCardPoints : Nat
  lowCardPoints : Nat
  targetPoints : Nat

/-- Calculates the total points for a given combination of high and low cards -/
def calculatePoints (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Nat :=
  highCount * game.highCardPoints + lowCount * game.lowCardPoints

/-- Checks if a given combination of high and low cards achieves the target points -/
def isValidCombination (game : HighLowGame) (highCount : Nat) (lowCount : Nat) : Prop :=
  calculatePoints game highCount lowCount = game.targetPoints

/-- Theorem: In the high-low game, to earn exactly 5 points, 
    the number of low cards drawn must be either 1, 3, or 5 -/
theorem high_low_game_combinations (game : HighLowGame) 
    (h1 : game.totalCards = 52)
    (h2 : game.highCards = game.lowCards)
    (h3 : game.highCards + game.lowCards = game.totalCards)
    (h4 : game.highCardPoints = 2)
    (h5 : game.lowCardPoints = 1)
    (h6 : game.targetPoints = 5) :
    ∀ (highCount lowCount : Nat), 
      isValidCombination game highCount lowCount → 
      lowCount = 1 ∨ lowCount = 3 ∨ lowCount = 5 :=
  sorry


end NUMINAMATH_CALUDE_high_low_game_combinations_l951_95183


namespace NUMINAMATH_CALUDE_circle_ratio_invariant_l951_95109

theorem circle_ratio_invariant (r : ℝ) (h : r > 2) : 
  let new_radius := r - 2
  let new_diameter := 2 * r - 4
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := by
sorry

end NUMINAMATH_CALUDE_circle_ratio_invariant_l951_95109


namespace NUMINAMATH_CALUDE_norma_cards_l951_95125

/-- The number of cards Norma has after losing some -/
def cards_remaining (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem: Norma has 18 cards remaining -/
theorem norma_cards : cards_remaining 88 70 = 18 := by
  sorry

end NUMINAMATH_CALUDE_norma_cards_l951_95125


namespace NUMINAMATH_CALUDE_principal_amount_l951_95164

-- Define the interest rate and time
def r : ℝ := 0.07
def t : ℝ := 2

-- Define the difference between C.I. and S.I.
def difference : ℝ := 49

-- State the theorem
theorem principal_amount (P : ℝ) :
  P * ((1 + r)^t - 1 - t * r) = difference → P = 10000 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_l951_95164


namespace NUMINAMATH_CALUDE_radical_simplification_l951_95168

theorem radical_simplification : 
  Real.sqrt ((16^12 + 8^14) / (16^5 + 8^16 + 2^24)) = 2^11 * Real.sqrt (65/17) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l951_95168


namespace NUMINAMATH_CALUDE_complex_first_quadrant_a_range_l951_95171

theorem complex_first_quadrant_a_range (a : ℝ) :
  let z : ℂ := Complex.mk a (1 - a)
  (0 < z.re ∧ 0 < z.im) → (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_first_quadrant_a_range_l951_95171


namespace NUMINAMATH_CALUDE_addition_and_subtraction_of_integers_and_fractions_l951_95140

theorem addition_and_subtraction_of_integers_and_fractions :
  (1 : ℤ) * 17 + (-12) = 5 ∧ -((1 : ℚ) / 7) - (-(6 / 7)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_addition_and_subtraction_of_integers_and_fractions_l951_95140


namespace NUMINAMATH_CALUDE_inequality_proof_l951_95120

theorem inequality_proof (a b c : ℝ) (h : Real.sqrt a ≥ Real.sqrt (b * c) ∧ Real.sqrt (b * c) ≥ Real.sqrt a - c) : b * c ≥ b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l951_95120


namespace NUMINAMATH_CALUDE_sum_product_quadratic_l951_95181

theorem sum_product_quadratic (S P x y : ℝ) :
  x + y = S ∧ x * y = P →
  ∃ t : ℝ, t ^ 2 - S * t + P = 0 ∧ (t = x ∨ t = y) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_quadratic_l951_95181


namespace NUMINAMATH_CALUDE_student_scores_l951_95185

theorem student_scores (M P C : ℝ) 
  (h1 : C = P + 20) 
  (h2 : (M + C) / 2 = 30) : 
  M + P = 40 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l951_95185


namespace NUMINAMATH_CALUDE_desk_lamp_profit_maximization_l951_95178

/-- Represents the profit function for desk lamp sales -/
def profit_function (original_price cost_price initial_sales price_increase : ℝ) : ℝ → ℝ :=
  λ x => (original_price + x - cost_price) * (initial_sales - 10 * x)

theorem desk_lamp_profit_maximization 
  (original_price : ℝ) 
  (cost_price : ℝ) 
  (initial_sales : ℝ) 
  (price_range_min : ℝ) 
  (price_range_max : ℝ) 
  (h1 : original_price = 40)
  (h2 : cost_price = 30)
  (h3 : initial_sales = 600)
  (h4 : price_range_min = 40)
  (h5 : price_range_max = 60)
  (h6 : price_range_min ≤ price_range_max) :
  (∃ x : ℝ, x = 10 ∧ profit_function original_price cost_price initial_sales x = 10000) ∧
  (∀ y : ℝ, price_range_min ≤ original_price + y ∧ original_price + y ≤ price_range_max →
    profit_function original_price cost_price initial_sales y ≤ 
    profit_function original_price cost_price initial_sales (price_range_max - original_price)) :=
by sorry

end NUMINAMATH_CALUDE_desk_lamp_profit_maximization_l951_95178


namespace NUMINAMATH_CALUDE_abes_budget_l951_95135

/-- Abe's restaurant budget problem -/
theorem abes_budget (B : ℚ) 
  (food_expense : B / 3 = B - (B / 4 + 1250))
  (supplies_expense : B / 4 = B - (B / 3 + 1250))
  (wages_expense : 1250 = B - (B / 3 + B / 4))
  (total_expense : B = B / 3 + B / 4 + 1250) :
  B = 3000 := by
  sorry

end NUMINAMATH_CALUDE_abes_budget_l951_95135


namespace NUMINAMATH_CALUDE_min_degree_for_horizontal_asymptote_l951_95147

/-- The numerator of our rational function -/
def numerator (x : ℝ) : ℝ := -3 * x^6 + 5 * x^4 - 4 * x^3 + 2

/-- The degree of the numerator -/
def numerator_degree : ℕ := 6

/-- A proposition stating that for any polynomial p(x) of degree less than 6,
    the rational function numerator(x) / p(x) does not have a horizontal asymptote -/
def no_horizontal_asymptote_below_6 (p : ℝ → ℝ) : Prop :=
  ∀ d : ℕ, d < 6 → ¬∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, x > M → |numerator x / p x - L| < ε

/-- A proposition stating that there exists a polynomial p(x) of degree 6
    such that the rational function numerator(x) / p(x) has a horizontal asymptote -/
def horizontal_asymptote_at_6 : Prop :=
  ∃ (p : ℝ → ℝ), ∃ (L : ℝ), 
    (∀ x, p x = x^6) ∧ 
    (∀ ε > 0, ∃ M, ∀ x, x > M → |numerator x / p x - L| < ε)

/-- The main theorem stating that 6 is the minimum degree for p(x) to have a horizontal asymptote -/
theorem min_degree_for_horizontal_asymptote :
  (∀ p : ℝ → ℝ, no_horizontal_asymptote_below_6 p) ∧ horizontal_asymptote_at_6 :=
sorry

end NUMINAMATH_CALUDE_min_degree_for_horizontal_asymptote_l951_95147


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l951_95134

theorem ice_cream_flavors (total_flavors : ℕ) (tried_two_years_ago : ℕ) (tried_last_year : ℕ) : 
  total_flavors = 100 →
  tried_two_years_ago = total_flavors / 4 →
  tried_last_year = 2 * tried_two_years_ago →
  total_flavors - (tried_two_years_ago + tried_last_year) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l951_95134


namespace NUMINAMATH_CALUDE_length_of_CD_length_of_CD_explicit_l951_95190

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    this theorem proves the length of CD. -/
theorem length_of_CD (a : ℝ) (h : a ≥ Real.sqrt 7) : ℝ :=
  let BC : ℝ := 3
  let AC : ℝ := a
  let AD : ℝ := 4
  let AB : ℝ := Real.sqrt (a^2 + 9)
  let BD : ℝ := Real.sqrt (a^2 - 7)
  |AD - BD|

/-- The length of CD is |4 - √(a² - 7)| -/
theorem length_of_CD_explicit (a : ℝ) (h : a ≥ Real.sqrt 7) :
  length_of_CD a h = |4 - Real.sqrt (a^2 - 7)| :=
sorry

end NUMINAMATH_CALUDE_length_of_CD_length_of_CD_explicit_l951_95190


namespace NUMINAMATH_CALUDE_last_colored_number_l951_95116

/-- The number of columns in the table -/
def num_columns : ℕ := 8

/-- The triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of a number in the table -/
def position (n : ℕ) : ℕ := n % num_columns

/-- Predicate to check if a number is colored -/
def is_colored (n : ℕ) : Prop := ∃ k : ℕ, n = triangular_number k

/-- Predicate to check if all columns are colored up to a certain number -/
def all_columns_colored (n : ℕ) : Prop :=
  ∀ col : ℕ, col < num_columns → ∃ m : ℕ, m ≤ n ∧ is_colored m ∧ position m = col

/-- The main theorem -/
theorem last_colored_number :
  ∃ n : ℕ, n = 120 ∧ is_colored n ∧ all_columns_colored n ∧
  ∀ m : ℕ, m < n → ¬(all_columns_colored m) :=
sorry

end NUMINAMATH_CALUDE_last_colored_number_l951_95116


namespace NUMINAMATH_CALUDE_circle_properties_l951_95156

/-- 
Given an equation x^2 + y^2 - 2x + 4y + m = 0 representing a circle,
prove that the center coordinates are (1, -2) and the range of m is (-∞, 5)
-/
theorem circle_properties (x y m : ℝ) :
  (x^2 + y^2 - 2*x + 4*y + m = 0) →
  (∃ r : ℝ, r > 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2) →
  ((1, -2) = (1, -2) ∧ m < 5) := by
sorry

end NUMINAMATH_CALUDE_circle_properties_l951_95156


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l951_95163

/-- An equilateral triangle with a point inside --/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle --/
  side_length : ℝ
  /-- The perpendicular distance from the point to the first side --/
  dist1 : ℝ
  /-- The perpendicular distance from the point to the second side --/
  dist2 : ℝ
  /-- The perpendicular distance from the point to the third side --/
  dist3 : ℝ
  /-- The side length is positive --/
  side_positive : side_length > 0
  /-- All distances are positive --/
  dist_positive : dist1 > 0 ∧ dist2 > 0 ∧ dist3 > 0

/-- The theorem stating the relationship between the side length and the perpendicular distances --/
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangleWithPoint) 
  (h1 : t.dist1 = 2) 
  (h2 : t.dist2 = 3) 
  (h3 : t.dist3 = 4) : 
  t.side_length = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l951_95163


namespace NUMINAMATH_CALUDE_coupon_probability_l951_95141

theorem coupon_probability (n m k : ℕ) (h1 : n = 17) (h2 : m = 9) (h3 : k = 6) : 
  (Nat.choose k k * Nat.choose (n - k) (m - k)) / Nat.choose n m = 3 / 442 := by
  sorry

end NUMINAMATH_CALUDE_coupon_probability_l951_95141


namespace NUMINAMATH_CALUDE_g_minus_two_equals_eleven_l951_95197

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 1

-- State the theorem
theorem g_minus_two_equals_eleven : g (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_g_minus_two_equals_eleven_l951_95197


namespace NUMINAMATH_CALUDE_roger_money_theorem_l951_95103

def roger_money_problem (initial_amount gift_amount spent_amount : ℕ) : Prop :=
  initial_amount + gift_amount - spent_amount = 19

theorem roger_money_theorem :
  roger_money_problem 16 28 25 := by
  sorry

end NUMINAMATH_CALUDE_roger_money_theorem_l951_95103


namespace NUMINAMATH_CALUDE_katies_journey_distance_l951_95170

/-- The total distance of Katie's journey to the island -/
def total_distance (leg1 leg2 leg3 : ℕ) : ℕ :=
  leg1 + leg2 + leg3

/-- Theorem stating that the total distance of Katie's journey is 436 miles -/
theorem katies_journey_distance :
  total_distance 132 236 68 = 436 := by
  sorry

end NUMINAMATH_CALUDE_katies_journey_distance_l951_95170


namespace NUMINAMATH_CALUDE_prescription_final_cost_l951_95139

/-- Calculates the final cost of a prescription after cashback and rebate --/
theorem prescription_final_cost 
  (original_price : ℝ) 
  (cashback_percent : ℝ) 
  (rebate : ℝ) 
  (h1 : original_price = 150)
  (h2 : cashback_percent = 0.1)
  (h3 : rebate = 25) :
  original_price - (cashback_percent * original_price) - rebate = 110 := by
  sorry

#check prescription_final_cost

end NUMINAMATH_CALUDE_prescription_final_cost_l951_95139


namespace NUMINAMATH_CALUDE_function_identity_l951_95192

def is_strictly_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ m n : ℕ+, m > n → f m > f n

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : is_strictly_increasing f) :
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l951_95192


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l951_95136

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 3 = 0 ∧ m % 4 = 1 ∧ m % 5 = 2 → n ≤ m :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l951_95136


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l951_95132

theorem max_product_under_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_constraint : 6 * a + 5 * b = 45) :
  a * b ≤ 135 / 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 5 * b₀ = 45 ∧ a₀ * b₀ = 135 / 8 :=
by sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l951_95132


namespace NUMINAMATH_CALUDE_balls_in_boxes_l951_95176

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

-- State the theorem
theorem balls_in_boxes : 
  distribute_balls num_balls num_boxes = 21 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l951_95176


namespace NUMINAMATH_CALUDE_arctan_special_angle_combination_l951_95184

/-- Proves that arctan(tan 75° - 3tan 15° + tan 45°) = 30° --/
theorem arctan_special_angle_combination :
  Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180) + Real.tan (45 * π / 180)) = 30 * π / 180 := by
  sorry

#check arctan_special_angle_combination

end NUMINAMATH_CALUDE_arctan_special_angle_combination_l951_95184


namespace NUMINAMATH_CALUDE_perry_recipe_fat_per_serving_l951_95193

/-- Calculates the grams of fat per serving in a recipe --/
def fatPerServing (servings : ℕ) (totalMixture : ℚ) (creamRatio cheesRatio butterRatio : ℕ) 
  (creamFat cheeseFat butterFat : ℚ) : ℚ :=
  let totalRatio := creamRatio + cheesRatio + butterRatio
  let partSize := totalMixture / totalRatio
  let creamAmount := partSize * creamRatio
  let cheeseAmount := partSize * cheesRatio
  let butterAmount := partSize * butterRatio * 2  -- Convert half-cups to cups
  let totalFat := creamAmount * creamFat + cheeseAmount * cheeseFat + butterAmount * butterFat
  totalFat / servings

/-- The grams of fat per serving in Perry's recipe --/
theorem perry_recipe_fat_per_serving :
  fatPerServing 6 (3/2) 5 3 2 88 110 184 = 37.65 := by
  sorry

end NUMINAMATH_CALUDE_perry_recipe_fat_per_serving_l951_95193


namespace NUMINAMATH_CALUDE_tangent_length_equals_hypotenuse_leg_l951_95113

-- Define the triangle DEF
structure RightTriangle where
  DE : ℝ
  DF : ℝ
  rightAngleAtE : True

-- Define the circle
structure TangentCircle where
  centerOnDE : True
  tangentToDF : True
  tangentToEF : True

-- Define the theorem
theorem tangent_length_equals_hypotenuse_leg 
  (triangle : RightTriangle) 
  (circle : TangentCircle) 
  (h1 : triangle.DE = 7) 
  (h2 : triangle.DF = Real.sqrt 85) : 
  ∃ Q : ℝ × ℝ, ∃ FQ : ℝ, FQ = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_length_equals_hypotenuse_leg_l951_95113


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l951_95133

/-- The number of times Terrell lifts the weights in his initial routine -/
def initial_lifts : ℕ := 10

/-- The weight of each dumbbell in Terrell's initial routine (in pounds) -/
def initial_weight : ℕ := 25

/-- The number of dumbbells Terrell uses -/
def num_dumbbells : ℕ := 3

/-- The weight of each dumbbell in Terrell's new routine (in pounds) -/
def new_weight : ℕ := 20

/-- The total weight Terrell lifts in his initial routine -/
def total_initial_weight : ℕ := num_dumbbells * initial_weight * initial_lifts

/-- The minimum number of times Terrell needs to lift the new weights to match or exceed the initial total weight -/
def min_new_lifts : ℕ := 13

theorem terrell_weight_lifting :
  num_dumbbells * new_weight * min_new_lifts ≥ total_initial_weight :=
sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l951_95133


namespace NUMINAMATH_CALUDE_sum_ac_equals_eight_l951_95179

theorem sum_ac_equals_eight 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_ac_equals_eight_l951_95179


namespace NUMINAMATH_CALUDE_newspaper_delivery_totals_l951_95106

/-- Represents the delivery schedule for a newspaper --/
structure DeliverySchedule where
  weekdayHouses : Nat
  weekdayDays : Nat
  weekendHouses : Nat
  weekendDays : Nat

/-- Calculates the total newspapers delivered in a week for a given schedule --/
def totalNewspapers (schedule : DeliverySchedule) : Nat :=
  schedule.weekdayHouses * schedule.weekdayDays + schedule.weekendHouses * schedule.weekendDays

/-- The delivery schedule for Newspaper A --/
def scheduleA : DeliverySchedule :=
  { weekdayHouses := 100, weekdayDays := 6, weekendHouses := 120, weekendDays := 1 }

/-- The delivery schedule for Newspaper B --/
def scheduleB : DeliverySchedule :=
  { weekdayHouses := 80, weekdayDays := 2, weekendHouses := 70, weekendDays := 2 }

/-- The delivery schedule for Newspaper C --/
def scheduleC : DeliverySchedule :=
  { weekdayHouses := 70, weekdayDays := 3, weekendHouses := 40, weekendDays := 1 }

theorem newspaper_delivery_totals :
  totalNewspapers scheduleA = 720 ∧
  totalNewspapers scheduleB = 300 ∧
  totalNewspapers scheduleC = 250 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_delivery_totals_l951_95106


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l951_95154

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Proposition: 108 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 108 → num_factors m ≠ 12) ∧ num_factors 108 = 12 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l951_95154


namespace NUMINAMATH_CALUDE_percentage_of_quarters_l951_95121

theorem percentage_of_quarters (dimes quarters half_dollars : ℕ) : 
  dimes = 75 → quarters = 35 → half_dollars = 15 →
  (quarters * 25 : ℚ) / (dimes * 10 + quarters * 25 + half_dollars * 50) = 368 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_quarters_l951_95121


namespace NUMINAMATH_CALUDE_g_one_equals_three_l951_95187

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g as an even function
def even_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem g_one_equals_three
  (f g : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_even : even_function g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_g_one_equals_three_l951_95187


namespace NUMINAMATH_CALUDE_davids_initial_money_l951_95129

/-- 
Given that David has $800 less than he spent after spending money on a trip,
and he now has $500 left, prove that he had $1800 at the beginning of his trip.
-/
theorem davids_initial_money :
  ∀ (initial_money spent_money remaining_money : ℕ),
  remaining_money = spent_money - 800 →
  remaining_money = 500 →
  initial_money = spent_money + remaining_money →
  initial_money = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_initial_money_l951_95129


namespace NUMINAMATH_CALUDE_household_spending_theorem_l951_95111

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

end NUMINAMATH_CALUDE_household_spending_theorem_l951_95111


namespace NUMINAMATH_CALUDE_add_6666_seconds_to_3pm_l951_95119

/-- Represents a time of day in hours, minutes, and seconds -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Converts seconds to a TimeOfDay structure -/
def secondsToTime (totalSeconds : Nat) : TimeOfDay :=
  let hours := totalSeconds / 3600
  let remainingSeconds := totalSeconds % 3600
  let minutes := remainingSeconds / 60
  let seconds := remainingSeconds % 60
  { hours := hours, minutes := minutes, seconds := seconds }

/-- Adds a TimeOfDay to another TimeOfDay, handling overflow -/
def addTime (t1 t2 : TimeOfDay) : TimeOfDay :=
  let totalSeconds := (t1.hours * 3600 + t1.minutes * 60 + t1.seconds) +
                      (t2.hours * 3600 + t2.minutes * 60 + t2.seconds)
  secondsToTime totalSeconds

theorem add_6666_seconds_to_3pm (startTime : TimeOfDay) (elapsedSeconds : Nat) :
  startTime.hours = 15 ∧ startTime.minutes = 0 ∧ startTime.seconds = 0 ∧ 
  elapsedSeconds = 6666 →
  let endTime := addTime startTime (secondsToTime elapsedSeconds)
  endTime.hours = 16 ∧ endTime.minutes = 51 ∧ endTime.seconds = 6 := by
  sorry

end NUMINAMATH_CALUDE_add_6666_seconds_to_3pm_l951_95119


namespace NUMINAMATH_CALUDE_rower_distance_l951_95162

/-- Represents the problem of calculating the distance traveled by a rower in a river --/
theorem rower_distance (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 10 →
  river_speed = 1.2 →
  total_time = 1 →
  let upstream_speed := man_speed - river_speed
  let downstream_speed := man_speed + river_speed
  let one_way_distance := (upstream_speed * downstream_speed * total_time) / (2 * (upstream_speed + downstream_speed))
  2 * one_way_distance = 9.856 := by
sorry

#eval (10 - 1.2) * (10 + 1.2) / (2 * ((10 - 1.2) + (10 + 1.2)))

end NUMINAMATH_CALUDE_rower_distance_l951_95162


namespace NUMINAMATH_CALUDE_points_always_odd_l951_95150

/-- Represents the number of points on the line after a certain number of operations -/
def num_points (initial : ℕ) (operations : ℕ) : ℕ :=
  if operations = 0 then
    initial
  else
    2 * num_points initial (operations - 1) - 1

/-- Theorem stating that the number of points is always odd after any number of operations -/
theorem points_always_odd (initial : ℕ) (operations : ℕ) :
  Odd (num_points initial operations) :=
by
  sorry


end NUMINAMATH_CALUDE_points_always_odd_l951_95150


namespace NUMINAMATH_CALUDE_square_of_two_digit_number_ending_in_five_l951_95173

theorem square_of_two_digit_number_ending_in_five (d : ℕ) 
  (h : d ≥ 1 ∧ d ≤ 9) : 
  (10 * d + 5)^2 = 100 * d * (d + 1) + 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_digit_number_ending_in_five_l951_95173


namespace NUMINAMATH_CALUDE_max_abs_z3_l951_95180

theorem max_abs_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂)) :
  Complex.abs z₃ ≤ Real.sqrt 2 ∧ 
  ∃ w₁ w₂ w₃ : ℂ, Complex.abs w₁ ≤ 1 ∧ 
               Complex.abs w₂ ≤ 1 ∧ 
               Complex.abs (2 * w₃ - (w₁ + w₂)) ≤ Complex.abs (w₁ - w₂) ∧
               Complex.abs w₃ = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_z3_l951_95180


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l951_95198

theorem right_triangle_third_side (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ((a = Real.sqrt 2 ∧ b = Real.sqrt 3) ∨ (a = Real.sqrt 3 ∧ b = Real.sqrt 2)) →
  c = Real.sqrt 5 ∨ c = 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l951_95198


namespace NUMINAMATH_CALUDE_bricks_to_fill_road_l951_95194

/-- Calculates the number of bricks needed to fill a rectangular road without overlapping -/
theorem bricks_to_fill_road (road_width road_length brick_width brick_height : ℝ) :
  road_width = 6 →
  road_length = 4 →
  brick_width = 0.6 →
  brick_height = 0.2 →
  (road_width * road_length) / (brick_width * brick_height) = 200 := by
  sorry

end NUMINAMATH_CALUDE_bricks_to_fill_road_l951_95194


namespace NUMINAMATH_CALUDE_actual_speed_is_22_5_l951_95143

/-- Proves that the actual average speed is 22.5 mph given the conditions of the problem -/
theorem actual_speed_is_22_5 (v t : ℝ) (h : v > 0) (h' : t > 0) :
  (v * t = (v + 37.5) * (3/8 * t)) → v = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_22_5_l951_95143


namespace NUMINAMATH_CALUDE_pentagonal_prism_vertices_l951_95118

/-- Definition of a pentagonal prism -/
structure PentagonalPrism :=
  (bases : ℕ)
  (rectangular_faces : ℕ)
  (h_bases : bases = 2)
  (h_faces : rectangular_faces = 5)

/-- The number of vertices in a pentagonal prism -/
def num_vertices (p : PentagonalPrism) : ℕ := 10

/-- Theorem stating that a pentagonal prism has 10 vertices -/
theorem pentagonal_prism_vertices (p : PentagonalPrism) : num_vertices p = 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_vertices_l951_95118


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l951_95151

theorem imaginary_part_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  Complex.im ((3 + i) / i^2 * i) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l951_95151


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l951_95186

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^5 - 24*x^4 + 5*x^3 + 15*x^2 - 18*x + 12 = 
  (x - 3) * (x^5 + 5*x^4 - 9*x^3 - 22*x^2 - 51*x - 171) - 501 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l951_95186


namespace NUMINAMATH_CALUDE_system_equation_result_l951_95172

theorem system_equation_result (a b A B C : ℝ) (x : ℝ) 
  (h1 : a * Real.sin x + b * Real.cos x = 0)
  (h2 : A * Real.sin (2 * x) + B * Real.cos (2 * x) = C)
  (h3 : a ≠ 0) :
  2 * a * b * A + (b^2 - a^2) * B + (a^2 + b^2) * C = 0 :=
by sorry

end NUMINAMATH_CALUDE_system_equation_result_l951_95172


namespace NUMINAMATH_CALUDE_x_squared_less_than_one_iff_l951_95122

theorem x_squared_less_than_one_iff (x : ℝ) : -1 < x ∧ x < 1 ↔ x^2 < 1 := by sorry

end NUMINAMATH_CALUDE_x_squared_less_than_one_iff_l951_95122


namespace NUMINAMATH_CALUDE_overlap_area_l951_95161

theorem overlap_area (total_length width : ℝ) 
  (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) :
  total_length = left_length + right_length →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    (left_only_area + overlap_area) / (right_only_area + overlap_area) = left_length / right_length :=
by sorry

end NUMINAMATH_CALUDE_overlap_area_l951_95161


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l951_95142

/-- Represents the nth term of the geometric sequence -/
def a (n : ℕ) : ℝ := sorry

/-- Represents the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The common difference when inserting n numbers between a_n and a_{n+1} -/
def d (n : ℕ) : ℝ := sorry

/-- Main theorem encompassing both parts of the problem -/
theorem geometric_sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * S n + 2) →
  (∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1)) ∧
  (¬ ∃ m k p : ℕ,
    m < k ∧ k < p ∧
    (k - m = p - k) ∧
    (d m * d p = d k * d k)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l951_95142


namespace NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l951_95149

theorem freshmen_in_liberal_arts (total_students : ℝ) 
  (freshmen_percent : ℝ) 
  (psych_majors_percent : ℝ) 
  (freshmen_psych_liberal_arts_percent : ℝ) :
  freshmen_percent = 0.6 →
  psych_majors_percent = 0.2 →
  freshmen_psych_liberal_arts_percent = 0.048 →
  (freshmen_psych_liberal_arts_percent * total_students) / 
  (psych_majors_percent * freshmen_percent * total_students) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_freshmen_in_liberal_arts_l951_95149


namespace NUMINAMATH_CALUDE_binomial_15_12_times_3_l951_95144

theorem binomial_15_12_times_3 : 3 * (Nat.choose 15 12) = 2730 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_times_3_l951_95144


namespace NUMINAMATH_CALUDE_mooney_ate_four_brownies_l951_95128

/-- The number of brownies in a dozen -/
def dozen : ℕ := 12

/-- The number of brownies Mother initially made -/
def initial_brownies : ℕ := 2 * dozen

/-- The number of brownies Father ate -/
def father_ate : ℕ := 8

/-- The number of brownies Mother made the next morning -/
def new_batch : ℕ := 2 * dozen

/-- The total number of brownies after adding the new batch -/
def final_count : ℕ := 36

/-- Theorem: Mooney ate 4 brownies -/
theorem mooney_ate_four_brownies :
  initial_brownies - father_ate - (final_count - new_batch) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mooney_ate_four_brownies_l951_95128


namespace NUMINAMATH_CALUDE_quadratic_properties_l951_95188

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc 1 4

theorem quadratic_properties :
  ∃ (x_min : ℝ), x_min ∈ interval ∧ 
  (∀ (x : ℝ), x ∈ interval → f x_min ≤ f x) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 1.5 → y ∈ Set.Icc 1 1.5 → x ≤ y → f x ≥ f y) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1.5 4 → y ∈ Set.Icc 1.5 4 → x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l951_95188


namespace NUMINAMATH_CALUDE_product_102_108_l951_95145

theorem product_102_108 : 102 * 108 = 11016 := by
  sorry

end NUMINAMATH_CALUDE_product_102_108_l951_95145


namespace NUMINAMATH_CALUDE_complete_set_is_reals_l951_95126

def is_complete (A : Set ℝ) : Prop :=
  A.Nonempty ∧ ∀ a b : ℝ, (a + b) ∈ A → (a * b) ∈ A

theorem complete_set_is_reals (A : Set ℝ) : is_complete A → A = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_complete_set_is_reals_l951_95126


namespace NUMINAMATH_CALUDE_multiply_58_62_l951_95130

theorem multiply_58_62 : 58 * 62 = 3596 := by
  sorry

end NUMINAMATH_CALUDE_multiply_58_62_l951_95130


namespace NUMINAMATH_CALUDE_total_tulips_is_308_l951_95105

/-- The number of tulips needed for Anna's smiley face design --/
def total_tulips : ℕ :=
  let red_eye := 8
  let purple_eyebrow := 5
  let red_nose := 12
  let red_smile := 18
  let yellow_background := 9 * red_smile
  let purple_eyebrows := 4 * (2 * red_eye)
  let yellow_nose := 3 * red_nose
  
  let total_red := 2 * red_eye + red_nose + red_smile
  let total_purple := 2 * purple_eyebrow + (purple_eyebrows - 2 * purple_eyebrow)
  let total_yellow := yellow_background + yellow_nose
  
  total_red + total_purple + total_yellow

theorem total_tulips_is_308 : total_tulips = 308 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_is_308_l951_95105


namespace NUMINAMATH_CALUDE_first_number_proof_l951_95124

theorem first_number_proof (x : ℕ) : 
  (∃ k : ℤ, x = 2 * k + 7) ∧ 
  (∃ l : ℤ, 2037 = 2 * l + 5) ∧ 
  (∀ y : ℕ, y < x → ¬(∃ m : ℤ, y = 2 * m + 7)) → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_first_number_proof_l951_95124


namespace NUMINAMATH_CALUDE_T_properties_l951_95107

theorem T_properties (n : ℕ) : 
  let T := (10 * (10^n - 1)) / 81 - n / 9
  ∃ (k : ℕ), T = k ∧ T % 11 = ((n + 1) / 2) % 11 := by
  sorry

end NUMINAMATH_CALUDE_T_properties_l951_95107


namespace NUMINAMATH_CALUDE_our_ellipse_correct_l951_95115

/-- An ellipse with foci at (-2, 0) and (2, 0) passing through (2, 3) -/
structure Ellipse where
  -- The equation of the ellipse
  equation : ℝ → ℝ → Prop
  -- The foci are at (-2, 0) and (2, 0)
  foci_x : equation (-2) 0 ∧ equation 2 0
  -- The ellipse passes through (2, 3)
  passes_through : equation 2 3
  -- The equation is of the form x^2/a^2 + y^2/b^2 = 1 for some a, b
  is_standard_form : ∃ a b : ℝ, ∀ x y : ℝ, equation x y ↔ x^2/a^2 + y^2/b^2 = 1

/-- The specific ellipse we're interested in -/
def our_ellipse : Ellipse where
  equation := fun x y => x^2/16 + y^2/12 = 1
  foci_x := sorry
  passes_through := sorry
  is_standard_form := sorry

/-- The theorem stating that our_ellipse satisfies all the conditions -/
theorem our_ellipse_correct : 
  our_ellipse.equation = fun x y => x^2/16 + y^2/12 = 1 := by sorry

end NUMINAMATH_CALUDE_our_ellipse_correct_l951_95115


namespace NUMINAMATH_CALUDE_common_roots_product_l951_95160

-- Define the cubic equations
def cubic1 (x C : ℝ) : ℝ := x^3 + 3*x^2 + C*x + 15
def cubic2 (x D : ℝ) : ℝ := x^3 + D*x^2 + 70

-- Define the condition of having two common roots
def has_two_common_roots (C D : ℝ) : Prop :=
  ∃ (p q : ℝ), p ≠ q ∧ cubic1 p C = 0 ∧ cubic1 q C = 0 ∧ cubic2 p D = 0 ∧ cubic2 q D = 0

-- The main theorem
theorem common_roots_product (C D : ℝ) : 
  has_two_common_roots C D → 
  ∃ (p q : ℝ), p * q = 10 * (7/2)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_common_roots_product_l951_95160


namespace NUMINAMATH_CALUDE_supplement_double_complement_30_l951_95131

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

end NUMINAMATH_CALUDE_supplement_double_complement_30_l951_95131


namespace NUMINAMATH_CALUDE_jakes_birdhouse_width_l951_95152

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℝ := 1
def sara_height : ℝ := 2
def sara_depth : ℝ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_height : ℝ := 20
def jake_depth : ℝ := 18

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume difference between Jake's and Sara's birdhouses in cubic inches -/
def volume_difference : ℝ := 1152

/-- Theorem stating that Jake's birdhouse width is 22.4 inches -/
theorem jakes_birdhouse_width :
  ∃ (jake_width : ℝ),
    jake_width * jake_height * jake_depth -
    (sara_width * sara_height * sara_depth * feet_to_inches^3) =
    volume_difference ∧
    jake_width = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_jakes_birdhouse_width_l951_95152


namespace NUMINAMATH_CALUDE_sin_equation_condition_l951_95114

theorem sin_equation_condition (α β : Real) :
  (7 * 15 * Real.sin α + Real.sin β = Real.sin (α + β)) ↔
  (∃ k : ℤ, α = 2 * k * Real.pi ∨ β = 2 * k * Real.pi ∨ α + β = 2 * k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_condition_l951_95114


namespace NUMINAMATH_CALUDE_abc_inequality_l951_95102

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 11/6 * c < a + b ∧ a + b < 2 * c)
  (h2 : 3/2 * a < b + c ∧ b + c < 5/3 * a)
  (h3 : 5/2 * b < a + c ∧ a + c < 11/4 * b) :
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l951_95102


namespace NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l951_95155

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : 
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) := by sorry

end NUMINAMATH_CALUDE_a_perpendicular_to_a_minus_b_l951_95155


namespace NUMINAMATH_CALUDE_propositions_true_l951_95182

theorem propositions_true :
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → (a - c) / c > (b - c) / b) ∧
  (∀ a b : ℝ, a > |b| → a^2 > b^2) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l951_95182


namespace NUMINAMATH_CALUDE_space_station_cost_share_l951_95117

/-- Proves that if a total cost of 50 billion dollars is shared equally among 500 million people,
    then each person's share is 100 dollars. -/
theorem space_station_cost_share :
  let total_cost : ℝ := 50 * 10^9  -- 50 billion dollars
  let num_people : ℝ := 500 * 10^6  -- 500 million people
  let share_per_person : ℝ := total_cost / num_people
  share_per_person = 100 := by sorry

end NUMINAMATH_CALUDE_space_station_cost_share_l951_95117


namespace NUMINAMATH_CALUDE_min_box_value_l951_95191

theorem min_box_value (a b Box : ℤ) : 
  (∀ x, (a*x + b)*(b*x + a) = 36*x^2 + Box*x + 36) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  Box = a^2 + b^2 →
  ∃ (min_Box : ℤ), (∀ Box', (∃ a' b' : ℤ, 
    (∀ x, (a'*x + b')*(b'*x + a') = 36*x^2 + Box'*x + 36) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' = a'^2 + b'^2) → 
    min_Box ≤ Box') ∧
  min_Box = 72 :=
sorry

end NUMINAMATH_CALUDE_min_box_value_l951_95191


namespace NUMINAMATH_CALUDE_solve_system_l951_95174

theorem solve_system (x y : ℝ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 6) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l951_95174


namespace NUMINAMATH_CALUDE_library_rearrangement_l951_95138

theorem library_rearrangement (total_books : ℕ) (initial_shelves : ℕ) (books_per_new_shelf : ℕ)
  (h1 : total_books = 1500)
  (h2 : initial_shelves = 50)
  (h3 : books_per_new_shelf = 28)
  (h4 : total_books % initial_shelves = 0) : -- Ensures equally-filled initial shelves
  (total_books % books_per_new_shelf : ℕ) = 14 := by
sorry

end NUMINAMATH_CALUDE_library_rearrangement_l951_95138


namespace NUMINAMATH_CALUDE_sum_of_ratios_l951_95165

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem sum_of_ratios (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f 1 = 2) : 
  f 2 / f 1 + f 4 / f 3 + f 6 / f 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_l951_95165


namespace NUMINAMATH_CALUDE_log_reciprocal_l951_95100

theorem log_reciprocal (M : ℝ) (a : ℤ) (b : ℝ) 
  (h_pos : M > 0) 
  (h_log : Real.log M / Real.log 10 = a + b) 
  (h_b : 0 < b ∧ b < 1) : 
  Real.log (1 / M) / Real.log 10 = (-a - 1) + (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_log_reciprocal_l951_95100


namespace NUMINAMATH_CALUDE_smoothie_ratio_l951_95101

/-- Given two juices P and V, and two smoothies A and Y, prove that the ratio of P to V in smoothie A is 4:1 -/
theorem smoothie_ratio :
  -- Total amounts of juices
  ∀ (total_p total_v : ℚ),
  -- Amounts in smoothie A
  ∀ (a_p a_v : ℚ),
  -- Amounts in smoothie Y
  ∀ (y_p y_v : ℚ),
  -- Conditions
  total_p = 24 →
  total_v = 25 →
  a_p = 20 →
  total_p = a_p + y_p →
  total_v = a_v + y_v →
  y_p * 5 = y_v →
  -- Conclusion
  a_p * 1 = a_v * 4 :=
by sorry

end NUMINAMATH_CALUDE_smoothie_ratio_l951_95101


namespace NUMINAMATH_CALUDE_sigma_inequality_l951_95189

/-- Sum of positive divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Theorem: If σ(n) > 2n, then σ(mn) > 2mn for any m -/
theorem sigma_inequality (n : ℕ+) (h : sigma n > 2 * n) :
  ∀ m : ℕ+, sigma (m * n) > 2 * m * n := by
  sorry

end NUMINAMATH_CALUDE_sigma_inequality_l951_95189


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l951_95159

/-- Calculates the interest rate for a purchase with a payment plan. -/
theorem interest_rate_calculation (purchase_price down_payment monthly_payment num_months : ℚ)
  (h_purchase : purchase_price = 112)
  (h_down : down_payment = 12)
  (h_monthly : monthly_payment = 10)
  (h_months : num_months = 12) :
  ∃ (interest_rate : ℚ), 
    (abs (interest_rate - 17.9) < 0.05) ∧ 
    (interest_rate = (((down_payment + monthly_payment * num_months) - purchase_price) / purchase_price) * 100) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l951_95159


namespace NUMINAMATH_CALUDE_num_triangles_on_circle_l951_95104

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle --/
def num_points : ℕ := 10

/-- The number of points needed to form a triangle --/
def points_per_triangle : ℕ := 3

/-- Theorem: The number of different triangles that can be formed
    by choosing 3 points from 10 distinct points on a circle's circumference
    is equal to 120 --/
theorem num_triangles_on_circle :
  choose num_points points_per_triangle = 120 := by sorry

end NUMINAMATH_CALUDE_num_triangles_on_circle_l951_95104


namespace NUMINAMATH_CALUDE_triangle_properties_l951_95127

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The main theorem
theorem triangle_properties :
  (∀ t, is_equilateral t → is_isosceles t) ∧
  (∃ t, is_isosceles t ∧ ¬is_equilateral t) ∧
  (∃ t, ¬is_equilateral t ∧ is_isosceles t) := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l951_95127


namespace NUMINAMATH_CALUDE_min_value_problem_equality_condition_l951_95196

theorem min_value_problem (x : ℝ) (h : x > 0) : 6 * x + 1 / x^6 ≥ 7 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 
  (6 * x + 1 / x^6 = 7) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_equality_condition_l951_95196


namespace NUMINAMATH_CALUDE_base_nine_digits_of_2048_l951_95169

theorem base_nine_digits_of_2048 : ∃ n : ℕ, n > 0 ∧ 9^(n-1) ≤ 2048 ∧ 2048 < 9^n :=
by sorry

end NUMINAMATH_CALUDE_base_nine_digits_of_2048_l951_95169


namespace NUMINAMATH_CALUDE_min_sum_of_bases_l951_95123

theorem min_sum_of_bases (a b : ℕ+) : 
  (3 * a + 5 = 5 * b + 3) → 
  (∀ (x y : ℕ+), (3 * x + 5 = 5 * y + 3) → (x + y ≥ a + b)) →
  a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_bases_l951_95123


namespace NUMINAMATH_CALUDE_janeles_cats_average_weight_is_correct_l951_95108

/-- The combined average weight of Janele's cats -/
def janeles_cats_average_weight : ℝ := by sorry

/-- The weights of Janele's first 7 cats -/
def first_seven_cats_weights : List ℝ := [12, 12, 14.7, 9.3, 14.9, 15.6, 8.7]

/-- Lily's weights over 4 days -/
def lily_weights : List ℝ := [14, 15.3, 13.2, 14.7]

/-- The number of Janele's cats -/
def num_cats : ℕ := 8

theorem janeles_cats_average_weight_is_correct :
  janeles_cats_average_weight = 
    (List.sum first_seven_cats_weights + List.sum lily_weights / 4) / num_cats := by sorry

end NUMINAMATH_CALUDE_janeles_cats_average_weight_is_correct_l951_95108


namespace NUMINAMATH_CALUDE_representation_625_ends_with_1_l951_95199

def base_count : ℕ := 4

theorem representation_625_ends_with_1 :
  (∃ (S : Finset ℕ), (∀ b ∈ S, 3 ≤ b ∧ b ≤ 10) ∧
   (∀ b ∈ S, (625 : ℕ) % b = 1) ∧
   S.card = base_count) :=
by sorry

end NUMINAMATH_CALUDE_representation_625_ends_with_1_l951_95199


namespace NUMINAMATH_CALUDE_range_of_fraction_l951_95167

theorem range_of_fraction (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3/2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l951_95167


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l951_95195

theorem magical_red_knights_fraction :
  ∀ (total_knights : ℕ) (red_knights blue_knights magical_knights : ℕ) 
    (red_magical blue_magical : ℚ),
    red_knights = total_knights / 3 →
    blue_knights = total_knights - red_knights →
    magical_knights = total_knights / 5 →
    blue_magical = (2/3) * red_magical →
    red_knights * red_magical + blue_knights * blue_magical = magical_knights →
    red_magical = 9/35 := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l951_95195


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l951_95137

theorem sum_and_ratio_to_difference (x y : ℝ) : 
  x + y = 520 → x / y = 0.75 → y - x = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l951_95137


namespace NUMINAMATH_CALUDE_demokhar_life_span_l951_95148

theorem demokhar_life_span (x : ℝ) 
  (boy : x / 4 = x * (1 / 4))
  (young_man : x / 5 = x * (1 / 5))
  (adult_man : x / 3 = x * (1 / 3))
  (old_man : 13 = x - (x / 4 + x / 5 + x / 3)) :
  x = 60 := by
    sorry

end NUMINAMATH_CALUDE_demokhar_life_span_l951_95148


namespace NUMINAMATH_CALUDE_fraction_equation_l951_95175

theorem fraction_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (2 * a + 5 * b) / (b + 5 * a) = 3) : 
  a / b = 1.2 - 0.4 * Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_l951_95175


namespace NUMINAMATH_CALUDE_circle_through_three_points_l951_95146

/-- A circle in a 2D plane --/
structure Circle where
  /-- The coefficient of x^2 (always 1 for a standard form circle equation) --/
  a : ℝ := 1
  /-- The coefficient of y^2 (always 1 for a standard form circle equation) --/
  b : ℝ := 1
  /-- The coefficient of x --/
  d : ℝ
  /-- The coefficient of y --/
  e : ℝ
  /-- The constant term --/
  f : ℝ

/-- Check if a point (x, y) lies on the circle --/
def Circle.contains (c : Circle) (x y : ℝ) : Prop :=
  c.a * x^2 + c.b * y^2 + c.d * x + c.e * y + c.f = 0

/-- The theorem stating that there exists a unique circle passing through three given points --/
theorem circle_through_three_points :
  ∃! c : Circle,
    c.contains 0 0 ∧
    c.contains 1 1 ∧
    c.contains 4 2 ∧
    c.a = 1 ∧
    c.b = 1 ∧
    c.d = -8 ∧
    c.e = 6 ∧
    c.f = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l951_95146


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l951_95158

theorem line_intercepts_sum (x y : ℝ) : 
  x / 3 - y / 4 = 1 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l951_95158


namespace NUMINAMATH_CALUDE_jerry_aunt_money_l951_95166

/-- The amount of money Jerry received from his aunt -/
def aunt_money : ℝ := 9.05

/-- The amount of money Jerry received from his uncle -/
def uncle_money : ℝ := aunt_money

/-- The amount of money Jerry received from his friends -/
def friends_money : ℝ := 22 + 23 + 22 + 22

/-- The amount of money Jerry received from his sister -/
def sister_money : ℝ := 7

/-- The mean of all the money Jerry received -/
def mean_money : ℝ := 16.3

/-- The number of sources Jerry received money from -/
def num_sources : ℕ := 7

theorem jerry_aunt_money :
  (friends_money + sister_money + aunt_money + uncle_money) / num_sources = mean_money :=
sorry

end NUMINAMATH_CALUDE_jerry_aunt_money_l951_95166


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l951_95157

structure Ball :=
  (color : String)

def Bag := List Ball

def draw (bag : Bag) : Ball × Bag :=
  match bag with
  | [] => ⟨Ball.mk "empty", []⟩
  | (b::bs) => ⟨b, bs⟩

def Event := Bag → Prop

def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ∀ bag, ¬(e1 bag ∧ e2 bag)

def complementary (e1 e2 : Event) : Prop :=
  ∀ bag, e1 bag ↔ ¬(e2 bag)

def initialBag : Bag :=
  [Ball.mk "red", Ball.mk "blue", Ball.mk "black", Ball.mk "white"]

def ADrawsWhite : Event :=
  λ bag => (draw bag).1.color = "white"

def BDrawsWhite : Event :=
  λ bag => let (_, remainingBag) := draw bag
           (draw remainingBag).1.color = "white"

theorem events_mutually_exclusive_but_not_complementary :
  mutuallyExclusive ADrawsWhite BDrawsWhite ∧
  ¬(complementary ADrawsWhite BDrawsWhite) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_but_not_complementary_l951_95157
