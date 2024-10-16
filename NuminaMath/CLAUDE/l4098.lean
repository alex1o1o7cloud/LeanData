import Mathlib

namespace NUMINAMATH_CALUDE_inequality_condition_l4098_409897

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l4098_409897


namespace NUMINAMATH_CALUDE_base_subtraction_l4098_409887

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The problem statement -/
theorem base_subtraction : 
  let base_9_number := [4, 2, 3]  -- 324 in base 9 (least significant digit first)
  let base_7_number := [5, 6, 1]  -- 165 in base 7 (least significant digit first)
  (to_base_10 base_9_number 9) - (to_base_10 base_7_number 7) = 169 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_l4098_409887


namespace NUMINAMATH_CALUDE_age_difference_equals_first_ratio_l4098_409881

/-- Represents the age ratio of four siblings -/
structure AgeRatio :=
  (a b c d : ℕ)

/-- Calculates the age difference between the first two siblings given their age ratio and total future age -/
def ageDifference (ratio : AgeRatio) (totalFutureAge : ℕ) : ℚ :=
  let x : ℚ := (totalFutureAge - 20 : ℚ) / (ratio.a + ratio.b + ratio.c + ratio.d : ℚ)
  ratio.a * x - ratio.b * x

/-- Theorem: The age difference between the first two siblings is equal to the first number in the ratio -/
theorem age_difference_equals_first_ratio 
  (ratio : AgeRatio) 
  (totalFutureAge : ℕ) 
  (h1 : ratio.a = 4) 
  (h2 : ratio.b = 3) 
  (h3 : ratio.c = 7) 
  (h4 : ratio.d = 5) 
  (h5 : totalFutureAge = 230) : 
  ageDifference ratio totalFutureAge = ratio.a := by
  sorry

#eval ageDifference ⟨4, 3, 7, 5⟩ 230

end NUMINAMATH_CALUDE_age_difference_equals_first_ratio_l4098_409881


namespace NUMINAMATH_CALUDE_min_x_given_inequality_l4098_409845

theorem min_x_given_inequality (x : ℝ) :
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) →
  x ≥ -1 ∧ ∀ y : ℝ, (∀ a : ℝ, a > 0 → y^2 ≤ 1 + a) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_x_given_inequality_l4098_409845


namespace NUMINAMATH_CALUDE_count_solution_pairs_l4098_409851

/-- The number of distinct ordered pairs of positive integers (x,y) satisfying x^4y^2 - 10x^2y + 9 = 0 -/
def solution_count : ℕ := 3

/-- A predicate that checks if a pair of positive integers satisfies the equation -/
def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 2) - 10 * (x.val ^ 2) * y.val + 9 = 0

theorem count_solution_pairs :
  (∃! (s : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ s, satisfies_equation p.1 p.2) ∧ 
    s.card = solution_count) := by sorry

end NUMINAMATH_CALUDE_count_solution_pairs_l4098_409851


namespace NUMINAMATH_CALUDE_parabola_translation_l4098_409830

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 2 0 1
  let translated := translate original (-1) (-3)
  translated = Parabola.mk 2 4 (-2) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4098_409830


namespace NUMINAMATH_CALUDE_alpha_range_l4098_409888

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l4098_409888


namespace NUMINAMATH_CALUDE_runners_in_quarter_segment_time_l4098_409876

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℕ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the time both runners spend simultaneously in a quarter segment of the track -/
def timeInQuarterSegment (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem runners_in_quarter_segment_time :
  let runner1 : Runner := { lapTime := 72, direction := true }
  let runner2 : Runner := { lapTime := 80, direction := false }
  timeInQuarterSegment runner1 runner2 = 46 := by sorry

end NUMINAMATH_CALUDE_runners_in_quarter_segment_time_l4098_409876


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4098_409838

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : z * (1 + i) = 2) : z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4098_409838


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l4098_409813

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + bc + c², then the measure of angle A is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  a^2 = b^2 + b*c + c^2 →
  A = 2 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l4098_409813


namespace NUMINAMATH_CALUDE_sunflower_height_l4098_409843

-- Define the height of Marissa's sister in inches
def sister_height_inches : ℕ := 4 * 12 + 3

-- Define the height difference between the sunflower and Marissa's sister
def height_difference : ℕ := 21

-- Theorem to prove the height of the sunflower
theorem sunflower_height :
  (sister_height_inches + height_difference) / 12 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_height_l4098_409843


namespace NUMINAMATH_CALUDE_paint_usage_l4098_409831

theorem paint_usage (initial_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) :
  initial_paint = 360 ∧
  first_week_fraction = 1/4 ∧
  second_week_fraction = 1/2 →
  let first_week_usage := first_week_fraction * initial_paint
  let remaining_paint := initial_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 225 :=
by sorry

end NUMINAMATH_CALUDE_paint_usage_l4098_409831


namespace NUMINAMATH_CALUDE_triangle_side_length_l4098_409857

-- Define a triangle XYZ
structure Triangle :=
  (x y z : ℝ)
  (X Y Z : ℝ)

-- State the theorem
theorem triangle_side_length (t : Triangle) 
  (hy : t.y = 7)
  (hz : t.z = 6)
  (hcos : Real.cos (t.Y - t.Z) = 1/2) :
  t.x = Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4098_409857


namespace NUMINAMATH_CALUDE_divergent_series_convergent_combination_l4098_409834

/-- Two positive sequences with divergent series but convergent combined series -/
theorem divergent_series_convergent_combination :
  ∃ (a b : ℕ → ℝ),
    (∀ n, a n > 0) ∧
    (∀ n, b n > 0) ∧
    (¬ Summable a) ∧
    (¬ Summable b) ∧
    Summable (λ n ↦ (2 * a n * b n) / (a n + b n)) := by
  sorry

end NUMINAMATH_CALUDE_divergent_series_convergent_combination_l4098_409834


namespace NUMINAMATH_CALUDE_john_uber_profit_l4098_409895

/-- Calculates the net profit of an Uber driver given their income and expenses --/
def uberDriverNetProfit (grossIncome : ℕ) (carPurchasePrice : ℕ) (monthlyMaintenance : ℕ) 
  (maintenancePeriod : ℕ) (annualInsurance : ℕ) (tireReplacement : ℕ) (tradeInValue : ℕ) 
  (taxRate : ℚ) : ℤ :=
  let totalMaintenance := monthlyMaintenance * maintenancePeriod
  let taxAmount := (grossIncome : ℚ) * taxRate
  let totalExpenses := carPurchasePrice + totalMaintenance + annualInsurance + tireReplacement + taxAmount.ceil
  (grossIncome : ℤ) - (totalExpenses : ℤ) + (tradeInValue : ℤ)

/-- Theorem stating that John's net profit as an Uber driver is $6,300 --/
theorem john_uber_profit : 
  uberDriverNetProfit 30000 20000 300 12 1200 400 6000 (15/100) = 6300 := by
  sorry

end NUMINAMATH_CALUDE_john_uber_profit_l4098_409895


namespace NUMINAMATH_CALUDE_village_population_growth_l4098_409863

theorem village_population_growth (
  adult_percentage : Real)
  (child_percentage : Real)
  (employed_adult_percentage : Real)
  (unemployed_adult_percentage : Real)
  (employed_adult_population : ℕ)
  (adult_growth_rate : Real)
  (h1 : adult_percentage = 0.6)
  (h2 : child_percentage = 0.4)
  (h3 : employed_adult_percentage = 0.7)
  (h4 : unemployed_adult_percentage = 0.3)
  (h5 : employed_adult_population = 18000)
  (h6 : adult_growth_rate = 0.05)
  (h7 : adult_percentage + child_percentage = 1)
  (h8 : employed_adult_percentage + unemployed_adult_percentage = 1) :
  ∃ (new_total_population : ℕ), new_total_population = 45000 := by
  sorry

#check village_population_growth

end NUMINAMATH_CALUDE_village_population_growth_l4098_409863


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4098_409896

theorem simplify_trig_expression (x : ℝ) :
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4098_409896


namespace NUMINAMATH_CALUDE_megan_songs_count_l4098_409802

/-- The number of songs Megan bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Theorem stating the total number of songs Megan bought -/
theorem megan_songs_count :
  total_songs 2 8 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_megan_songs_count_l4098_409802


namespace NUMINAMATH_CALUDE_apple_products_total_cost_l4098_409886

/-- Calculates the total cost of an iPhone and iWatch after discounts and cashback -/
theorem apple_products_total_cost 
  (iphone_price : ℝ) 
  (iwatch_price : ℝ) 
  (iphone_discount : ℝ) 
  (iwatch_discount : ℝ) 
  (cashback_rate : ℝ) 
  (h1 : iphone_price = 800) 
  (h2 : iwatch_price = 300) 
  (h3 : iphone_discount = 0.15) 
  (h4 : iwatch_discount = 0.10) 
  (h5 : cashback_rate = 0.02) : 
  ℝ := by
  sorry

#check apple_products_total_cost

end NUMINAMATH_CALUDE_apple_products_total_cost_l4098_409886


namespace NUMINAMATH_CALUDE_card_selection_ways_l4098_409822

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of cards per suit in a standard deck
def cards_per_suit : ℕ := 13

-- Define the total number of cards in a standard deck
def total_cards : ℕ := num_suits * cards_per_suit

-- Define the number of cards to choose
def cards_to_choose : ℕ := 4

-- Define the number of cards to keep after discarding
def cards_to_keep : ℕ := 3

-- Theorem statement
theorem card_selection_ways :
  (num_suits.choose cards_to_choose) * (cards_per_suit ^ cards_to_choose) * cards_to_choose = 114244 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_ways_l4098_409822


namespace NUMINAMATH_CALUDE_ratio_equivalence_l4098_409849

theorem ratio_equivalence (x : ℚ) : (3 / x = 3 / 16) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l4098_409849


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_minimum_value_equality_l4098_409844

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ (2/3 ≤ x ∧ x < 2) ∨ x > 2 :=
sorry

-- Problem 2
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_minimum_value_equality_l4098_409844


namespace NUMINAMATH_CALUDE_mandy_coin_value_l4098_409815

/-- Represents the number of cents in a coin -/
inductive Coin
| Dime : Coin
| Quarter : Coin

def coin_value : Coin → Nat
| Coin.Dime => 10
| Coin.Quarter => 25

/-- Represents Mandy's coin collection -/
structure CoinCollection where
  dimes : Nat
  quarters : Nat
  total_coins : Nat
  coin_balance : dimes + quarters = total_coins
  dime_quarter_relation : dimes + 2 = quarters

def collection_value (c : CoinCollection) : Nat :=
  c.dimes * coin_value Coin.Dime + c.quarters * coin_value Coin.Quarter

theorem mandy_coin_value :
  ∃ c : CoinCollection, c.total_coins = 17 ∧ collection_value c = 320 := by
  sorry

end NUMINAMATH_CALUDE_mandy_coin_value_l4098_409815


namespace NUMINAMATH_CALUDE_pie_eating_difference_l4098_409818

theorem pie_eating_difference :
  let first_participant : ℚ := 5/6
  let second_participant : ℚ := 2/3
  first_participant - second_participant = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_difference_l4098_409818


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l4098_409889

theorem prop_a_necessary_not_sufficient :
  (∃ a : ℝ, a < 2 ∧ a^2 ≥ 4) ∧
  (∀ a : ℝ, a^2 < 4 → a < 2) :=
by sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_l4098_409889


namespace NUMINAMATH_CALUDE_perfume_price_decrease_l4098_409854

theorem perfume_price_decrease (original_price increased_price final_price : ℝ) : 
  original_price = 1200 →
  increased_price = original_price * 1.1 →
  final_price = original_price - 78 →
  (increased_price - final_price) / increased_price = 0.15 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_decrease_l4098_409854


namespace NUMINAMATH_CALUDE_cos_unique_identifier_l4098_409870

open Real

theorem cos_unique_identifier (x : ℝ) (h1 : π / 2 < x) (h2 : x < π) :
  (sin x > 0 ∧ cos x < 0 ∧ cot x < 0) ∧
  (∀ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = cot →
    (f x < 0 → f = cos)) :=
by sorry

end NUMINAMATH_CALUDE_cos_unique_identifier_l4098_409870


namespace NUMINAMATH_CALUDE_not_in_second_quadrant_l4098_409864

-- Define the linear function
def f (x : ℝ) : ℝ := x - 1

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) := by
  sorry

end NUMINAMATH_CALUDE_not_in_second_quadrant_l4098_409864


namespace NUMINAMATH_CALUDE_elberta_money_l4098_409882

theorem elberta_money (granny_smith : ℕ) (anjou : ℕ) (elberta : ℕ) : 
  granny_smith = 64 →
  anjou = granny_smith / 4 →
  elberta = anjou + 3 →
  elberta = 19 := by
  sorry

end NUMINAMATH_CALUDE_elberta_money_l4098_409882


namespace NUMINAMATH_CALUDE_trajectory_and_angle_property_l4098_409811

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the trajectory Γ
def trajectory_Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define point T
def point_T : ℝ × ℝ := (4, 0)

-- Define the line y = k(x-1)
def line_k (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the statement
theorem trajectory_and_angle_property :
  ∀ (P Q : ℝ × ℝ),
  (∃ (x y : ℝ), P = (x, y) ∧ circle_E x y) →
  (∃ (x y : ℝ), Q = (x, y) ∧ 
    (∃ (m b : ℝ), y - P.2 = m * (x - P.1) ∧ 
      y - point_F.2 = -1/m * (x - point_F.1) ∧
      2 * x = P.1 + point_F.1 ∧ 2 * y = P.2 + point_F.2) ∧
    (∃ (t : ℝ), x = t * P.1 ∧ y = t * P.2 ∧ 0 ≤ t ∧ t ≤ 1)) →
  (∃ (x y : ℝ), Q = (x, y) ∧ trajectory_Γ x y) ∧
  (∀ (k : ℝ) (R S : ℝ × ℝ),
    (∃ (x y : ℝ), R = (x, y) ∧ trajectory_Γ x y ∧ line_k k x y) →
    (∃ (x y : ℝ), S = (x, y) ∧ trajectory_Γ x y ∧ line_k k x y) →
    (R.2 / (R.1 - point_T.1) + S.2 / (S.1 - point_T.1) = 0)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_angle_property_l4098_409811


namespace NUMINAMATH_CALUDE_opposite_sign_implications_l4098_409817

theorem opposite_sign_implications (a b : ℝ) 
  (h1 : |2*a + b| * Real.sqrt (3*b + 12) ≤ 0) 
  (h2 : |2*a + b| + Real.sqrt (3*b + 12) > 0) : 
  (Real.sqrt (2*a - 3*b) = 4 ∨ Real.sqrt (2*a - 3*b) = -4) ∧ 
  (∀ x : ℝ, a*x^2 + 4*b - 2 = 0 ↔ x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_implications_l4098_409817


namespace NUMINAMATH_CALUDE_paco_cookie_difference_l4098_409847

/-- Given Paco's cookie situation, prove that he ate 9 more cookies than he gave away -/
theorem paco_cookie_difference (initial_cookies : ℕ) (cookies_given : ℕ) (cookies_eaten : ℕ)
  (h1 : initial_cookies = 41)
  (h2 : cookies_given = 9)
  (h3 : cookies_eaten = 18) :
  cookies_eaten - cookies_given = 9 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookie_difference_l4098_409847


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l4098_409855

/-- Given two points are symmetric with respect to the origin, 
    their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  B.1 = -A.1 ∧ B.2 = -A.2

theorem symmetry_coordinates :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l4098_409855


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l4098_409801

def m : ℕ := 2023^2 + 3^2023

theorem units_digit_of_m_squared_plus_3_to_m (m : ℕ) : (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_3_to_m_l4098_409801


namespace NUMINAMATH_CALUDE_center_of_mass_distance_to_line_l4098_409850

/-- Two material points in a plane -/
structure MaterialPoint where
  position : ℝ × ℝ
  mass : ℝ

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Distance from a point to a line -/
def distanceToLine (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Center of mass of two material points -/
def centerOfMass (p1 p2 : MaterialPoint) : ℝ × ℝ := sorry

theorem center_of_mass_distance_to_line 
  (P Q : MaterialPoint) (MN : Line) 
  (a b : ℝ) 
  (h1 : distanceToLine P.position MN = a) 
  (h2 : distanceToLine Q.position MN = b) :
  let Z := centerOfMass P Q
  distanceToLine Z MN = (P.mass * a + Q.mass * b) / (P.mass + Q.mass) := by
  sorry

end NUMINAMATH_CALUDE_center_of_mass_distance_to_line_l4098_409850


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l4098_409856

/-- A function f is decreasing on ℝ if for all x y, x < y implies f x > f y -/
def DecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : DecreasingOn f) (h_inequality : f (3 * a) < f (-2 * a + 10)) : 
  a > 2 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l4098_409856


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l4098_409816

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 4) :
  w / y = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l4098_409816


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l4098_409890

/-- Given two circles with equations (x^2 + y^2 + 2ax + a^2 - 4 = 0) and (x^2 + y^2 - 4by - 1 + 4b^2 = 0),
    where a ∈ ℝ, ab ≠ 0, and the circles have exactly three common tangents,
    prove that the minimum value of (1/a^2 + 1/b^2) is 1. -/
theorem min_value_sum_reciprocal_squares (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∨ x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0) →
  (∃! (t1 t2 t3 : ℝ × ℝ → ℝ), t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    (∀ x y : ℝ, (t1 (x, y) = 0 ∨ t2 (x, y) = 0 ∨ t3 (x, y) = 0) ↔ 
      ((x^2 + y^2 + 2*a*x + a^2 - 4 = 0) ∨ (x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0)))) →
  a ≠ 0 →
  b ≠ 0 →
  ∃ (m : ℝ), m = 1 ∧ ∀ (k : ℝ), k ≥ 0 → (1 / a^2 + 1 / b^2) ≥ m + k :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_squares_l4098_409890


namespace NUMINAMATH_CALUDE_celias_rent_l4098_409807

/-- Celia's monthly budget -/
structure MonthlyBudget where
  food : ℕ
  streaming : ℕ
  cellPhone : ℕ
  rent : ℕ
  savings : ℕ

/-- Celia's budget satisfies the given conditions -/
def validBudget (b : MonthlyBudget) : Prop :=
  b.food = 400 ∧
  b.streaming = 30 ∧
  b.cellPhone = 50 ∧
  b.savings = 198 ∧
  b.savings * 10 = b.food + b.streaming + b.cellPhone + b.rent

/-- Theorem: Celia's rent is $1500 -/
theorem celias_rent (b : MonthlyBudget) (h : validBudget b) : b.rent = 1500 := by
  sorry


end NUMINAMATH_CALUDE_celias_rent_l4098_409807


namespace NUMINAMATH_CALUDE_complement_event_A_equiv_l4098_409877

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: there are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- Theorem: The complement of "at least 2 defective products" is "at most 1 defective product" -/
theorem complement_event_A_equiv :
  ∀ defective : ℕ, defective ≤ sample_size →
    complement_A defective ↔ defective ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_event_A_equiv_l4098_409877


namespace NUMINAMATH_CALUDE_quadratic_factorization_l4098_409841

theorem quadratic_factorization (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l4098_409841


namespace NUMINAMATH_CALUDE_x_powers_sum_l4098_409862

theorem x_powers_sum (x : ℝ) (h : x + 1/x = 10) : 
  x^2 + 1/x^2 = 98 ∧ x^3 + 1/x^3 = 970 := by
  sorry

end NUMINAMATH_CALUDE_x_powers_sum_l4098_409862


namespace NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l4098_409860

def initial_bales : ℕ := 15
def initial_cost_per_bale : ℕ := 20
def new_cost_per_bale : ℕ := 27

theorem additional_cost_for_new_requirements :
  (initial_bales * 3 * new_cost_per_bale) - (initial_bales * initial_cost_per_bale) = 915 := by
  sorry

end NUMINAMATH_CALUDE_additional_cost_for_new_requirements_l4098_409860


namespace NUMINAMATH_CALUDE_prism_on_sphere_surface_area_l4098_409852

/-- A right prism with all vertices on a sphere -/
structure PrismOnSphere where
  /-- The height of the prism -/
  height : ℝ
  /-- The volume of the prism -/
  volume : ℝ
  /-- The surface area of the sphere -/
  sphereSurfaceArea : ℝ

/-- Theorem: If a right prism with all vertices on a sphere has height 4 and volume 64,
    then the surface area of the sphere is 48π -/
theorem prism_on_sphere_surface_area (p : PrismOnSphere) 
    (h_height : p.height = 4)
    (h_volume : p.volume = 64) :
    p.sphereSurfaceArea = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_prism_on_sphere_surface_area_l4098_409852


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l4098_409800

/-- Represents the number of residents in different age groups and the sampling size for one group -/
structure CommunityData where
  residents_35_to_45 : ℕ
  residents_46_to_55 : ℕ
  residents_56_to_65 : ℕ
  sampled_46_to_55 : ℕ

/-- Calculates the total number of people selected in a stratified sampling survey -/
def totalSampled (data : CommunityData) : ℕ :=
  (data.residents_35_to_45 + data.residents_46_to_55 + data.residents_56_to_65) / 
  (data.residents_46_to_55 / data.sampled_46_to_55)

/-- Theorem: Given the community data, the total number of people selected in the sampling survey is 140 -/
theorem stratified_sampling_result (data : CommunityData) 
  (h1 : data.residents_35_to_45 = 450)
  (h2 : data.residents_46_to_55 = 750)
  (h3 : data.residents_56_to_65 = 900)
  (h4 : data.sampled_46_to_55 = 50) :
  totalSampled data = 140 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l4098_409800


namespace NUMINAMATH_CALUDE_largest_base5_to_decimal_l4098_409828

/-- Converts a base-5 digit to its decimal (base-10) value --/
def base5ToDecimal (digit : Nat) : Nat := digit

/-- Calculates the value of a base-5 digit in its positional notation --/
def digitValue (digit : Nat) (position : Nat) : Nat := 
  base5ToDecimal digit * (5 ^ position)

/-- Represents a five-digit base-5 number --/
structure FiveDigitBase5Number where
  digit1 : Nat
  digit2 : Nat
  digit3 : Nat
  digit4 : Nat
  digit5 : Nat
  all_digits_valid : digit1 < 5 ∧ digit2 < 5 ∧ digit3 < 5 ∧ digit4 < 5 ∧ digit5 < 5

/-- Converts a five-digit base-5 number to its decimal (base-10) equivalent --/
def toDecimal (n : FiveDigitBase5Number) : Nat :=
  digitValue n.digit1 4 + digitValue n.digit2 3 + digitValue n.digit3 2 + 
  digitValue n.digit4 1 + digitValue n.digit5 0

/-- The largest five-digit base-5 number --/
def largestBase5 : FiveDigitBase5Number where
  digit1 := 4
  digit2 := 4
  digit3 := 4
  digit4 := 4
  digit5 := 4
  all_digits_valid := by simp

theorem largest_base5_to_decimal : 
  toDecimal largestBase5 = 3124 := by sorry

end NUMINAMATH_CALUDE_largest_base5_to_decimal_l4098_409828


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l4098_409833

theorem largest_integer_inequality : ∀ y : ℤ, y ≤ 7 ↔ (y : ℚ) / 4 + 3 / 7 < 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l4098_409833


namespace NUMINAMATH_CALUDE_total_insects_count_l4098_409835

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of lacewings -/
def lacewings : ℕ := 23250

/-- The total number of insects on the fields -/
def total_insects : ℕ := ladybugs_with_spots + ladybugs_without_spots + lacewings

theorem total_insects_count : total_insects = 90332 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_count_l4098_409835


namespace NUMINAMATH_CALUDE_triangle_angle_value_max_side_sum_l4098_409883

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

theorem triangle_angle_value (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

theorem max_side_sum (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 4) :
  ∃ (b c : ℝ), t.b = b ∧ t.c = c ∧ b + c ≤ 8 * Real.sqrt 3 / 3 ∧
  ∀ (b' c' : ℝ), t.b = b' ∧ t.c = c' → b' + c' ≤ 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_max_side_sum_l4098_409883


namespace NUMINAMATH_CALUDE_parabola_intersection_l4098_409812

/-- Given a parabola y = ax² + x + c that intersects the x-axis at x-coordinate 1,
    prove that a + c = -1 -/
theorem parabola_intersection (a c : ℝ) : 
  (∀ x, a * x^2 + x + c = 0 → x = 1) → a + c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l4098_409812


namespace NUMINAMATH_CALUDE_number_of_bags_l4098_409806

theorem number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 52) (h2 : cookies_per_bag = 2) :
  total_cookies / cookies_per_bag = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bags_l4098_409806


namespace NUMINAMATH_CALUDE_optimal_stamp_combination_l4098_409840

/-- The minimum number of stamps needed to make 50 cents using only 5-cent and 7-cent stamps -/
def min_stamps : ℕ := 8

/-- The number of 5-cent stamps used in the optimal solution -/
def num_5cent : ℕ := 3

/-- The number of 7-cent stamps used in the optimal solution -/
def num_7cent : ℕ := 5

theorem optimal_stamp_combination :
  (∀ x y : ℕ, 5 * x + 7 * y = 50 → x + y ≥ min_stamps) ∧
  5 * num_5cent + 7 * num_7cent = 50 ∧
  num_5cent + num_7cent = min_stamps := by
  sorry

end NUMINAMATH_CALUDE_optimal_stamp_combination_l4098_409840


namespace NUMINAMATH_CALUDE_product_evaluation_l4098_409878

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l4098_409878


namespace NUMINAMATH_CALUDE_spice_difference_total_l4098_409892

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.35

theorem spice_difference_total : 
  abs (cinnamon - nutmeg) + abs (nutmeg - ginger) + abs (cinnamon - ginger) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_spice_difference_total_l4098_409892


namespace NUMINAMATH_CALUDE_cube_volume_l4098_409803

/-- Given a cube with side perimeter 32 cm, its volume is 512 cubic cm. -/
theorem cube_volume (side_perimeter : ℝ) (h : side_perimeter = 32) : 
  (side_perimeter / 4)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l4098_409803


namespace NUMINAMATH_CALUDE_division_of_decimals_l4098_409875

theorem division_of_decimals : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l4098_409875


namespace NUMINAMATH_CALUDE_number_properties_l4098_409853

def is_even (n : ℕ) := n % 2 = 0
def is_odd (n : ℕ) := n % 2 ≠ 0
def is_prime (n : ℕ) := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → n % m ≠ 0)
def is_composite (n : ℕ) := n > 1 ∧ ¬(is_prime n)

theorem number_properties :
  (∀ n : ℕ, n ≤ 10 → (is_even n ∧ ¬is_composite n) → n = 2) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ ¬is_prime n) → n = 1) ∧
  (∀ n : ℕ, n ≤ 10 → (is_odd n ∧ is_composite n) → n = 9) ∧
  (∀ n : ℕ, is_prime n → n ≥ 2) ∧
  (∀ n : ℕ, is_composite n → n ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_number_properties_l4098_409853


namespace NUMINAMATH_CALUDE_marble_probability_l4098_409814

/-- Given a bag of marbles with 5 red, 4 blue, and 6 yellow marbles,
    the probability of drawing one marble that is either red or blue is 3/5. -/
theorem marble_probability : 
  let red : ℕ := 5
  let blue : ℕ := 4
  let yellow : ℕ := 6
  let total : ℕ := red + blue + yellow
  let target : ℕ := red + blue
  (target : ℚ) / total = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l4098_409814


namespace NUMINAMATH_CALUDE_exists_coverable_prism_l4098_409868

/-- A regular triangular prism with side edge length √3 times the base edge length -/
structure RegularTriangularPrism where
  base_edge : ℝ
  side_edge : ℝ
  side_edge_eq : side_edge = base_edge * Real.sqrt 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- A covering of a prism by equilateral triangles -/
structure PrismCovering where
  prism : RegularTriangularPrism
  triangles : Set EquilateralTriangle
  covers_prism : Bool
  no_overlaps : Bool

/-- Theorem stating the existence of a regular triangular prism that can be covered by equilateral triangles -/
theorem exists_coverable_prism : ∃ (p : RegularTriangularPrism) (c : PrismCovering), 
  c.prism = p ∧ c.covers_prism ∧ c.no_overlaps := by
  sorry

end NUMINAMATH_CALUDE_exists_coverable_prism_l4098_409868


namespace NUMINAMATH_CALUDE_cubic_integer_roots_l4098_409819

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a cubic polynomial, counting multiplicity -/
def num_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- The theorem stating the possible values for the number of integer roots -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  num_integer_roots p = 0 ∨ num_integer_roots p = 1 ∨ num_integer_roots p = 2 ∨ num_integer_roots p = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_integer_roots_l4098_409819


namespace NUMINAMATH_CALUDE_ladder_distance_l4098_409874

theorem ladder_distance (angle : Real) (length : Real) (distance : Real) : 
  angle = 60 * π / 180 →
  length = 19 →
  distance = length * Real.cos angle →
  distance = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l4098_409874


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l4098_409869

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 2*α - 2005 = 0) → 
  (β^2 + 2*β - 2005 = 0) → 
  (α^2 + 3*α + β = 2003) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l4098_409869


namespace NUMINAMATH_CALUDE_shorter_worm_length_l4098_409894

/-- Given two worms where one is 0.8 inches long and the other is 0.7 inches longer,
    prove that the length of the shorter worm is 0.8 inches. -/
theorem shorter_worm_length (worm1 worm2 : ℝ) 
  (h1 : worm1 = 0.8)
  (h2 : worm2 = worm1 + 0.7) :
  min worm1 worm2 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_shorter_worm_length_l4098_409894


namespace NUMINAMATH_CALUDE_fraction_change_l4098_409893

/-- Given a fraction 3/4, if we increase the numerator by 12% and decrease the denominator by 2%,
    the resulting fraction is approximately 0.8571. -/
theorem fraction_change (ε : ℝ) (h_ε : ε > 0) :
  ∃ (new_fraction : ℝ),
    (3 * (1 + 0.12)) / (4 * (1 - 0.02)) = new_fraction ∧
    |new_fraction - 0.8571| < ε :=
by sorry

end NUMINAMATH_CALUDE_fraction_change_l4098_409893


namespace NUMINAMATH_CALUDE_total_teachers_l4098_409810

/-- The total number of teachers in a school with stratified sampling -/
theorem total_teachers (senior_teachers : ℕ) (intermediate_teachers : ℕ) 
  (total_selected : ℕ) (other_selected : ℕ) 
  (h1 : senior_teachers = 26)
  (h2 : intermediate_teachers = 104)
  (h3 : total_selected = 56)
  (h4 : other_selected = 16)
  (h_stratified : (total_selected - other_selected) / (senior_teachers + intermediate_teachers) = 
                  total_selected / (senior_teachers + intermediate_teachers + (total_selected - other_selected) * (senior_teachers + intermediate_teachers) / (total_selected - other_selected))) :
  senior_teachers + intermediate_teachers + (total_selected - other_selected) * (senior_teachers + intermediate_teachers) / (total_selected - other_selected) = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_teachers_l4098_409810


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l4098_409861

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

-- Define the property of being monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_monotone_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → 2 < a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l4098_409861


namespace NUMINAMATH_CALUDE_custom_chess_pieces_l4098_409825

theorem custom_chess_pieces (num_players : Nat) (std_pieces_per_player : Nat)
  (missing_queens : Nat) (missing_knights : Nat) (missing_pawns : Nat)
  (h1 : num_players = 3)
  (h2 : std_pieces_per_player = 16)
  (h3 : missing_queens = 2)
  (h4 : missing_knights = 5)
  (h5 : missing_pawns = 8) :
  let total_missing := missing_queens + missing_knights + missing_pawns
  let total_original := num_players * std_pieces_per_player
  let pieces_per_player := (total_original - total_missing) / num_players
  (pieces_per_player = 11) ∧ (total_original - total_missing = 33) := by
  sorry

end NUMINAMATH_CALUDE_custom_chess_pieces_l4098_409825


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l4098_409808

theorem eight_digit_divisibility (n : ℕ) (h : 1000 ≤ n ∧ n < 10000) :
  ∃ k : ℕ, 10001 * n = k * (10000 * n + n) :=
sorry

end NUMINAMATH_CALUDE_eight_digit_divisibility_l4098_409808


namespace NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l4098_409867

theorem nested_sqrt_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l4098_409867


namespace NUMINAMATH_CALUDE_translated_minimum_point_l4098_409859

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- Theorem statement
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ g x_min = 0 ∧ x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l4098_409859


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l4098_409858

theorem smallest_n_for_sqrt_inequality :
  ∀ n : ℕ, n > 0 → (Real.sqrt (5 * n) - Real.sqrt (5 * n - 4) < 0.01) ↔ n ≥ 8001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l4098_409858


namespace NUMINAMATH_CALUDE_pencil_cartons_l4098_409884

/-- Given an order of pencils and erasers, prove the number of cartons of pencils -/
theorem pencil_cartons (pencil_cost eraser_cost total_cartons total_cost : ℕ) 
  (h1 : pencil_cost = 6)
  (h2 : eraser_cost = 3)
  (h3 : total_cartons = 100)
  (h4 : total_cost = 360) :
  ∃ (pencil_cartons eraser_cartons : ℕ),
    pencil_cartons + eraser_cartons = total_cartons ∧
    pencil_cost * pencil_cartons + eraser_cost * eraser_cartons = total_cost ∧
    pencil_cartons = 20 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cartons_l4098_409884


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_9450_l4098_409899

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem least_n_factorial_divisible_by_9450 :
  ∃ (n : ℕ), n > 0 ∧ is_factor 9450 (Nat.factorial n) ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬is_factor 9450 (Nat.factorial m) :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_9450_l4098_409899


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l4098_409898

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  ¬ is_factor (x * y) 48 →
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l4098_409898


namespace NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l4098_409823

/-- The number of three-digit whole numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The number of digits available for the first position (excluding 0, 8, and 9) -/
def first_digit_options : ℕ := 7

/-- The number of digits available for the second and third positions (excluding 8 and 9) -/
def other_digit_options : ℕ := 8

/-- The number of three-digit numbers without 8 or 9 -/
def numbers_without_eight_or_nine : ℕ := first_digit_options * other_digit_options * other_digit_options

theorem three_digit_numbers_with_eight_or_nine :
  total_three_digit_numbers - numbers_without_eight_or_nine = 452 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_with_eight_or_nine_l4098_409823


namespace NUMINAMATH_CALUDE_fairGame_l4098_409866

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  yellow : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (count : BallCount) : ℕ :=
  count.yellow + count.black + count.red

/-- Determines if the game is fair given the current ball count -/
def isFair (count : BallCount) : Prop :=
  count.yellow = count.black

/-- Represents the action of replacing black balls with yellow balls -/
def replaceBalls (count : BallCount) (n : ℕ) : BallCount :=
  { yellow := count.yellow + n
    black := count.black - n
    red := count.red }

/-- The main theorem stating that replacing 4 black balls with yellow balls makes the game fair -/
theorem fairGame (initialCount : BallCount)
    (h1 : initialCount.yellow = 5)
    (h2 : initialCount.black = 13)
    (h3 : initialCount.red = 22) :
    isFair (replaceBalls initialCount 4) := by
  sorry

end NUMINAMATH_CALUDE_fairGame_l4098_409866


namespace NUMINAMATH_CALUDE_umars_age_l4098_409879

/-- Given the ages of Ali, Yusaf, and Umar, prove Umar's age -/
theorem umars_age (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  umar_age = 2 * yusaf_age →
  umar_age = 10 := by
sorry

end NUMINAMATH_CALUDE_umars_age_l4098_409879


namespace NUMINAMATH_CALUDE_total_roses_is_109_l4098_409804

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_is_109 : total_roses_needed = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_roses_is_109_l4098_409804


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l4098_409865

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x, f a c x > 0 ↔ 1 < x ∧ x < 3

-- Define the sufficient condition
def sufficient_condition (a c m : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * x + 4 * c > 0 → x + m > 0

-- Define the not necessary condition
def not_necessary_condition (a c m : ℝ) : Prop :=
  ∃ x, x + m > 0 ∧ ¬(a * x^2 + 2 * x + 4 * c > 0)

theorem quadratic_inequality_problem (a c m : ℝ) :
  solution_set a c →
  sufficient_condition a c m →
  not_necessary_condition a c m →
  (a = -1/4 ∧ c = -3/4) ∧ (m ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l4098_409865


namespace NUMINAMATH_CALUDE_sum_of_roots_l4098_409871

theorem sum_of_roots (p q r : ℕ+) : 
  4 * (7^(1/4) - 6^(1/4) : ℝ) = p^(1/4) + q^(1/4) - r^(1/4) → p + q + r = 122 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4098_409871


namespace NUMINAMATH_CALUDE_eight_div_repeating_third_l4098_409832

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- The result of 8 divided by 0.3333... --/
theorem eight_div_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_div_repeating_third_l4098_409832


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l4098_409848

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + a

-- Define the discriminant
def discriminant (a : ℝ) : ℝ := 9 - 4*a

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (∃ x : ℝ, quadratic_equation 1 x = 0) ∧
  (∃ a : ℝ, a ≠ 1 ∧ ∃ x : ℝ, quadratic_equation a x = 0) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l4098_409848


namespace NUMINAMATH_CALUDE_sweets_distribution_l4098_409891

theorem sweets_distribution (total_sweets : ℕ) (sweets_per_child : ℕ) : 
  total_sweets = 288 →
  sweets_per_child = 4 →
  ∃ (num_children : ℕ), 
    (num_children * sweets_per_child + total_sweets / 3 = total_sweets) ∧
    num_children = 48 := by
  sorry

end NUMINAMATH_CALUDE_sweets_distribution_l4098_409891


namespace NUMINAMATH_CALUDE_discount_calculation_l4098_409846

theorem discount_calculation (CP : ℝ) (MP SP discount : ℝ) : 
  MP = 1.1 * CP → 
  SP = 0.99 * CP → 
  discount = MP - SP → 
  discount = 0.11 * CP :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l4098_409846


namespace NUMINAMATH_CALUDE_cloud_ratio_is_twelve_to_one_l4098_409824

/-- The ratio of cumulus clouds to cumulonimbus clouds -/
def cloud_ratio (cirrus cumulus cumulonimbus : ℕ) : ℚ :=
  (cumulus : ℚ) / cumulonimbus

theorem cloud_ratio_is_twelve_to_one :
  ∀ (cirrus cumulus cumulonimbus : ℕ),
    cirrus = 4 * cumulus →
    cumulus = cloud_ratio cumulus cumulonimbus * cumulonimbus →
    cumulonimbus = 3 →
    cirrus = 144 →
    cloud_ratio cumulus cumulonimbus = 12 := by
  sorry

end NUMINAMATH_CALUDE_cloud_ratio_is_twelve_to_one_l4098_409824


namespace NUMINAMATH_CALUDE_policy_effect_l4098_409826

-- Define the labor market for teachers
structure TeacherMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_wage : ℝ  -- Equilibrium wage

-- Define the commercial education market
structure CommercialEducationMarket where
  supply : ℝ → ℝ  -- Supply function
  demand : ℝ → ℝ  -- Demand function
  equilibrium_price : ℝ  -- Equilibrium price

-- Define the government policy
def government_policy (min_years : ℕ) (locality : String) : Prop :=
  ∃ (requirement : Prop), requirement

-- Theorem statement
theorem policy_effect 
  (teacher_market : TeacherMarket)
  (commercial_market : CommercialEducationMarket)
  (min_years : ℕ)
  (locality : String) :
  government_policy min_years locality →
  ∃ (new_teacher_market : TeacherMarket)
    (new_commercial_market : CommercialEducationMarket),
    new_teacher_market.equilibrium_wage > teacher_market.equilibrium_wage ∧
    new_commercial_market.equilibrium_price < commercial_market.equilibrium_price :=
by
  sorry

end NUMINAMATH_CALUDE_policy_effect_l4098_409826


namespace NUMINAMATH_CALUDE_A_mod_126_l4098_409805

/-- A function that generates the number A by concatenating all three-digit numbers from 100 to 799 -/
def generate_A : ℕ := sorry

/-- Theorem stating that the number A is congruent to 91 modulo 126 -/
theorem A_mod_126 : generate_A % 126 = 91 := by sorry

end NUMINAMATH_CALUDE_A_mod_126_l4098_409805


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l4098_409873

theorem min_value_expression (x : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l4098_409873


namespace NUMINAMATH_CALUDE_f_composition_value_l4098_409880

def f (x : ℝ) : ℝ := 4 * x^3 - 6 * x + 2

theorem f_composition_value : f (f 2) = 42462 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l4098_409880


namespace NUMINAMATH_CALUDE_dividend_calculation_l4098_409820

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 47.5)
  (h2 : quotient = 24.3)
  (h3 : remainder = 32.4) :
  divisor * quotient + remainder = 1186.15 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4098_409820


namespace NUMINAMATH_CALUDE_zou_win_probability_l4098_409842

/-- Represents the outcome of a race -/
inductive RaceOutcome
| Win
| Loss

/-- Calculates the probability of winning a race given the previous outcome -/
def winProbability (previousOutcome : RaceOutcome) : ℚ :=
  match previousOutcome with
  | RaceOutcome.Win => 2/3
  | RaceOutcome.Loss => 1/3

/-- Represents a sequence of race outcomes -/
def RaceSequence := List RaceOutcome

/-- Calculates the probability of a given race sequence -/
def sequenceProbability (sequence : RaceSequence) : ℚ :=
  sequence.foldl (fun acc outcome => acc * winProbability outcome) 1

/-- Generates all possible race sequences where Zou wins exactly 5 out of 6 races -/
def winningSequences : List RaceSequence := sorry

theorem zou_win_probability :
  let totalProbability := (winningSequences.map sequenceProbability).sum
  totalProbability = 80/243 := by sorry

end NUMINAMATH_CALUDE_zou_win_probability_l4098_409842


namespace NUMINAMATH_CALUDE_farm_animals_l4098_409829

theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) : 
  total_heads = 60 →
  total_feet = 200 →
  hen_heads = 1 →
  hen_feet = 2 →
  cow_heads = 1 →
  cow_feet = 4 →
  ∃ (num_hens : ℕ) (num_cows : ℕ),
    num_hens + num_cows = total_heads ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 20 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_l4098_409829


namespace NUMINAMATH_CALUDE_equation_solutions_l4098_409839

theorem equation_solutions :
  (∃ x : ℚ, 3 * x + 20 = 4 * x - 25 ∧ x = 45) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6 ∧ x = 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4098_409839


namespace NUMINAMATH_CALUDE_total_classes_taught_l4098_409827

-- Define the number of classes Eduardo taught
def eduardo_classes : ℕ := 3

-- Define Frankie's classes as double Eduardo's
def frankie_classes : ℕ := 2 * eduardo_classes

-- Theorem to prove
theorem total_classes_taught : eduardo_classes + frankie_classes = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_classes_taught_l4098_409827


namespace NUMINAMATH_CALUDE_canteen_distance_l4098_409821

theorem canteen_distance (girls_camp_distance boys_camp_distance : ℝ) 
  (h1 : girls_camp_distance = 600)
  (h2 : boys_camp_distance = 800) :
  let hypotenuse := Real.sqrt (girls_camp_distance ^ 2 + boys_camp_distance ^ 2)
  let canteen_distance := Real.sqrt ((girls_camp_distance ^ 2 + (hypotenuse / 2) ^ 2))
  ⌊canteen_distance⌋ = 781 := by
  sorry

end NUMINAMATH_CALUDE_canteen_distance_l4098_409821


namespace NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l4098_409809

/-- Given two sets A and B with specific elements, prove that if A = B, then a = 1 -/
theorem set_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_a_equals_one_l4098_409809


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l4098_409836

theorem hyperbola_foci_distance : 
  ∀ (x y : ℝ), 
  (y^2 / 75) - (x^2 / 11) = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 86 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l4098_409836


namespace NUMINAMATH_CALUDE_sqrt_sum_upper_bound_l4098_409885

theorem sqrt_sum_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_upper_bound_l4098_409885


namespace NUMINAMATH_CALUDE_maria_friends_money_l4098_409837

/-- The amount of money Maria gave to her three friends -/
def total_given (maria_money : ℝ) (isha_share : ℝ) (florence_share : ℝ) (rene_share : ℝ) : ℝ :=
  isha_share + florence_share + rene_share

/-- Theorem stating the total amount Maria gave to her friends -/
theorem maria_friends_money :
  ∀ (maria_money : ℝ),
  maria_money > 0 →
  let isha_share := (1/3) * maria_money
  let florence_share := (1/2) * isha_share
  let rene_share := 300
  florence_share = 3 * rene_share →
  total_given maria_money isha_share florence_share rene_share = 3000 := by
  sorry

end NUMINAMATH_CALUDE_maria_friends_money_l4098_409837


namespace NUMINAMATH_CALUDE_triangle_inequality_l4098_409872

theorem triangle_inequality (x y z : ℝ) (A B C : ℝ) :
  x > 0 → y > 0 → z > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  x * Real.sin A + y * Real.sin B + z * Real.sin C ≤ 
  (1/2) * (x*y + y*z + z*x) * Real.sqrt ((x + y + z)/(x*y*z)) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4098_409872
