import Mathlib

namespace NUMINAMATH_CALUDE_prob_red_then_black_our_deck_l3689_368911

/-- A customized deck of cards -/
structure CustomDeck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (deck : CustomDeck) : ℚ :=
  (deck.red_cards : ℚ) * (deck.black_cards : ℚ) / ((deck.total_cards : ℚ) * (deck.total_cards - 1 : ℚ))

/-- Our specific deck -/
def our_deck : CustomDeck :=
  { total_cards := 78
  , red_cards := 39
  , black_cards := 39 }

theorem prob_red_then_black_our_deck :
  prob_red_then_black our_deck = 507 / 2002 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_our_deck_l3689_368911


namespace NUMINAMATH_CALUDE_frank_cookie_fraction_l3689_368982

/-- Given the number of cookies for Millie, calculate Mike's cookies -/
def mikeCookies (millieCookies : ℕ) : ℕ := 3 * millieCookies

/-- Calculate the fraction of Frank's cookies compared to Mike's -/
def frankFraction (frankCookies millieCookies : ℕ) : ℚ :=
  frankCookies / (mikeCookies millieCookies)

/-- Theorem: Frank's fraction of cookies compared to Mike's is 1/4 -/
theorem frank_cookie_fraction :
  frankFraction 3 4 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_cookie_fraction_l3689_368982


namespace NUMINAMATH_CALUDE_faye_team_size_l3689_368983

def team_size (total_points : ℕ) (faye_points : ℕ) (others_points : ℕ) : ℕ :=
  (total_points - faye_points) / others_points + 1

theorem faye_team_size :
  team_size 68 28 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_faye_team_size_l3689_368983


namespace NUMINAMATH_CALUDE_no_resident_claims_to_be_liar_l3689_368926

-- Define the types of residents on the island
inductive Resident
| Knight
| Liar

-- Define the statement made by a resident
def makes_statement (r : Resident) : Prop :=
  match r with
  | Resident.Knight => True   -- Knights always tell the truth
  | Resident.Liar => False    -- Liars always lie

-- Define the statement "I am a liar"
def claims_to_be_liar (r : Resident) : Prop :=
  makes_statement r = (r = Resident.Liar)

-- Theorem: No resident can claim to be a liar
theorem no_resident_claims_to_be_liar :
  ∀ r : Resident, ¬(claims_to_be_liar r) :=
by sorry

end NUMINAMATH_CALUDE_no_resident_claims_to_be_liar_l3689_368926


namespace NUMINAMATH_CALUDE_inequality_proof_l3689_368940

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (a^2 + a*c + c^2) ≥ 
  4 * Real.sqrt ((a*b / (a+b))^2 + (a*b / (a+b)) * (a*c / (a+c)) + (a*c / (a+c))^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3689_368940


namespace NUMINAMATH_CALUDE_sqrt_27_minus_3tan60_plus_power_equals_1_l3689_368955

theorem sqrt_27_minus_3tan60_plus_power_equals_1 :
  Real.sqrt 27 - 3 * Real.tan (60 * π / 180) + (π - Real.sqrt 2) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_3tan60_plus_power_equals_1_l3689_368955


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_5_subset_condition_l3689_368929

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_eq_5 :
  (A ∩ B 5 = A) ∧ (Aᶜ ∪ B 5 = Set.univ) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ m, A ⊆ B m ↔ m > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_eq_5_subset_condition_l3689_368929


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l3689_368997

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_3_4_5 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l3689_368997


namespace NUMINAMATH_CALUDE_product_sum_inequality_l3689_368966

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l3689_368966


namespace NUMINAMATH_CALUDE_initial_amount_proof_l3689_368934

/-- Prove that given an initial amount P, after applying 5% interest for the first year
    and 6% interest for the second year, if the final amount is 5565, then P must be 5000. -/
theorem initial_amount_proof (P : ℝ) : 
  P * (1 + 0.05) * (1 + 0.06) = 5565 → P = 5000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l3689_368934


namespace NUMINAMATH_CALUDE_second_train_speed_l3689_368938

/-- Proves that the speed of the second train is 36 kmph given the conditions of the problem -/
theorem second_train_speed (first_train_speed : ℝ) (time_difference : ℝ) (meeting_distance : ℝ) :
  first_train_speed = 30 →
  time_difference = 5 →
  meeting_distance = 1050 →
  ∃ (second_train_speed : ℝ),
    second_train_speed * (meeting_distance / second_train_speed) =
    meeting_distance - first_train_speed * time_difference +
    first_train_speed * (meeting_distance / second_train_speed) ∧
    second_train_speed = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_second_train_speed_l3689_368938


namespace NUMINAMATH_CALUDE_grandmas_brownie_pan_l3689_368957

/-- Represents a rectangular brownie pan with cuts -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width
  length_cuts : ℕ
  width_cuts : ℕ

/-- Calculates the number of interior pieces -/
def interior_pieces (pan : BrowniePan) : ℕ :=
  (pan.m.val - pan.length_cuts - 1) * (pan.n.val - pan.width_cuts - 1)

/-- Calculates the number of perimeter pieces -/
def perimeter_pieces (pan : BrowniePan) : ℕ :=
  2 * (pan.m.val + pan.n.val) - 4

/-- The main theorem about Grandma's brownie pan -/
theorem grandmas_brownie_pan :
  ∃ (pan : BrowniePan),
    pan.length_cuts = 3 ∧
    pan.width_cuts = 5 ∧
    interior_pieces pan = 2 * perimeter_pieces pan ∧
    pan.m = 6 ∧
    pan.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_brownie_pan_l3689_368957


namespace NUMINAMATH_CALUDE_sum_of_squares_fourth_degree_equation_l3689_368958

-- Part 1
theorem sum_of_squares (x y : ℝ) :
  (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7 → x^2 + y^2 = 5 :=
by sorry

-- Part 2
theorem fourth_degree_equation (x : ℝ) :
  x^4 - 6*x^2 + 8 = 0 → x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_fourth_degree_equation_l3689_368958


namespace NUMINAMATH_CALUDE_custom_operation_calculation_l3689_368968

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- Theorem statement
theorem custom_operation_calculation :
  star (star (star 2 3) 4) 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_calculation_l3689_368968


namespace NUMINAMATH_CALUDE_min_value_theorem_l3689_368904

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ ∃ y > 1, y + 4 / (y - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3689_368904


namespace NUMINAMATH_CALUDE_words_with_e_count_l3689_368974

/-- The number of letters in the alphabet we're using -/
def n : ℕ := 5

/-- The length of the words we're creating -/
def k : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def m : ℕ := 4

/-- The number of 4-letter words that can be made from the letters A, B, C, D, and E, 
    allowing repetition and using the letter E at least once -/
def words_with_e : ℕ := n^k - m^k

theorem words_with_e_count : words_with_e = 369 := by sorry

end NUMINAMATH_CALUDE_words_with_e_count_l3689_368974


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3689_368961

theorem max_sum_of_factors (clubsuit heartsuit : ℕ) : 
  clubsuit * heartsuit = 48 → 
  Even clubsuit → 
  ∃ (a b : ℕ), a * b = 48 ∧ Even a ∧ a + b ≤ clubsuit + heartsuit ∧ a + b = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3689_368961


namespace NUMINAMATH_CALUDE_midpoint_of_intersection_l3689_368984

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  intersecting_line A.1 A.2 ∧ intersecting_line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  (A.1 + B.1) / 2 = -9/5 ∧ (A.2 + B.2) / 2 = 1/5 :=
sorry

end NUMINAMATH_CALUDE_midpoint_of_intersection_l3689_368984


namespace NUMINAMATH_CALUDE_problem_statement_l3689_368912

theorem problem_statement (a b : ℝ) : 
  (a + b + 1 = -2) → (a + b - 1) * (1 - a - b) = -16 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3689_368912


namespace NUMINAMATH_CALUDE_josh_initial_marbles_l3689_368916

/-- The number of marbles Josh found -/
def marbles_found : ℕ := 7

/-- The current total number of marbles Josh has -/
def current_total : ℕ := 28

/-- The initial number of marbles in Josh's collection -/
def initial_marbles : ℕ := current_total - marbles_found

theorem josh_initial_marbles :
  initial_marbles = 21 := by sorry

end NUMINAMATH_CALUDE_josh_initial_marbles_l3689_368916


namespace NUMINAMATH_CALUDE_no_real_roots_l3689_368993

theorem no_real_roots : 
  ¬∃ x : ℝ, (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_l3689_368993


namespace NUMINAMATH_CALUDE_machine_does_not_require_repair_l3689_368919

/-- Represents a weighing machine for food portions --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.max_deviation < m.unreadable_deviation_bound

/-- Theorem: The weighing machine does not require repair --/
theorem machine_does_not_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) := by
  sorry

#check machine_does_not_require_repair

end NUMINAMATH_CALUDE_machine_does_not_require_repair_l3689_368919


namespace NUMINAMATH_CALUDE_arc_minutes_to_degrees_l3689_368950

theorem arc_minutes_to_degrees :
  ∀ (arc_minutes : ℝ) (degrees : ℝ),
  (arc_minutes = 1200) →
  (degrees = 20) →
  (arc_minutes * (1 / 60) = degrees) :=
by
  sorry

end NUMINAMATH_CALUDE_arc_minutes_to_degrees_l3689_368950


namespace NUMINAMATH_CALUDE_coffee_shop_weekly_production_l3689_368939

/-- A coffee shop that brews a certain number of coffee cups per day -/
structure CoffeeShop where
  weekday_cups_per_hour : ℕ
  weekend_total_cups : ℕ
  hours_open_per_day : ℕ

/-- Calculate the total number of coffee cups brewed in one week -/
def weekly_coffee_cups (shop : CoffeeShop) : ℕ :=
  let weekday_cups := shop.weekday_cups_per_hour * shop.hours_open_per_day * 5
  let weekend_cups := shop.weekend_total_cups
  weekday_cups + weekend_cups

/-- Theorem stating that a coffee shop with given parameters brews 370 cups in a week -/
theorem coffee_shop_weekly_production :
  ∀ (shop : CoffeeShop),
    shop.weekday_cups_per_hour = 10 →
    shop.weekend_total_cups = 120 →
    shop.hours_open_per_day = 5 →
    weekly_coffee_cups shop = 370 :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_weekly_production_l3689_368939


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3689_368989

/-- Given a point P(x, -9) where the distance from the x-axis to P is half the distance
    from the y-axis to P, prove that the distance from P to the y-axis is 18 units. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -9)
  (abs (P.2) = (1/2 : ℝ) * abs P.1) →
  abs P.1 = 18 := by
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3689_368989


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_price_l3689_368975

/-- The price Ramesh paid for a refrigerator given specific conditions -/
theorem ramesh_refrigerator_price (P : ℝ) 
  (h1 : 1.1 * P = 17600)  -- Selling price for 10% profit without discount
  (h2 : 0.2 * P = P - 0.8 * P)  -- 20% discount on labelled price
  (h3 : 125 = 125)  -- Transport cost
  (h4 : 250 = 250)  -- Installation cost
  : 0.8 * P + 125 + 250 = 13175 := by
  sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_price_l3689_368975


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3689_368987

/-- 
Given an equation (x^2)/(15-k) + (y^2)/(k-9) = 1 that represents an ellipse with foci on the y-axis,
prove that k is in the open interval (12, 15).
-/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) →  -- equation represents an ellipse
  (∃ c : ℝ, ∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1 ↔ 
    y^2 / (k - 9) + x^2 / (15 - k) = 1 ∧ 
    y^2 / c^2 - x^2 / (k - 9 - c^2) = 1) →  -- foci are on y-axis
  k > 12 ∧ k < 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3689_368987


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3689_368999

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hθ : θ = Real.pi / 3) : 
  Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos θ)) = 5 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3689_368999


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3689_368917

/-- Given a geometric sequence {a_n} with first term a and common ratio q,
    if a_2 * a_5 = 2 * a_3 and (a_4 + 2 * a_7) / 2 = 5/4,
    then the sum of the first 5 terms (S_5) is equal to 31. -/
theorem geometric_sequence_sum (a q : ℝ) : 
  (a * q * (a * q^4) = 2 * (a * q^2)) →
  ((a * q^3 + 2 * (a * q^6)) / 2 = 5/4) →
  (a * (1 - q^5)) / (1 - q) = 31 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3689_368917


namespace NUMINAMATH_CALUDE_percentage_no_conditions_is_13_33_l3689_368913

/-- Represents the survey results of teachers' health conditions -/
structure TeacherSurvey where
  total : ℕ
  highBP : ℕ
  heartTrouble : ℕ
  diabetes : ℕ
  highBP_heartTrouble : ℕ
  heartTrouble_diabetes : ℕ
  highBP_diabetes : ℕ
  all_three : ℕ

/-- Calculates the percentage of teachers with no health conditions -/
def percentageWithNoConditions (survey : TeacherSurvey) : ℚ :=
  let withConditions := 
    survey.highBP + survey.heartTrouble + survey.diabetes -
    survey.highBP_heartTrouble - survey.heartTrouble_diabetes - survey.highBP_diabetes +
    survey.all_three
  let withoutConditions := survey.total - withConditions
  (withoutConditions : ℚ) / survey.total * 100

/-- The main theorem stating that the percentage of teachers with no health conditions is 13.33% -/
theorem percentage_no_conditions_is_13_33 (survey : TeacherSurvey) 
  (h1 : survey.total = 150)
  (h2 : survey.highBP = 80)
  (h3 : survey.heartTrouble = 60)
  (h4 : survey.diabetes = 30)
  (h5 : survey.highBP_heartTrouble = 20)
  (h6 : survey.heartTrouble_diabetes = 10)
  (h7 : survey.highBP_diabetes = 15)
  (h8 : survey.all_three = 5) :
  percentageWithNoConditions survey = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_no_conditions_is_13_33_l3689_368913


namespace NUMINAMATH_CALUDE_derivative_at_three_l3689_368976

/-- Given a function f with f(x) = 3x^2 + 2xf'(1) for all x, prove that f'(3) = 6 -/
theorem derivative_at_three (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 2 * x * (deriv f 1)) :
  deriv f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_three_l3689_368976


namespace NUMINAMATH_CALUDE_simplify_expression_l3689_368965

theorem simplify_expression (x : ℝ) : 
  3*x + 5*x^2 + 12 - (6 - 3*x - 10*x^2) = 15*x^2 + 6*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3689_368965


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l3689_368907

/-- Given 18 bottle caps shared among 6 friends, prove that each friend receives 3 bottle caps. -/
theorem bottle_cap_distribution (total_caps : ℕ) (num_friends : ℕ) (caps_per_friend : ℕ) : 
  total_caps = 18 → num_friends = 6 → caps_per_friend = total_caps / num_friends → caps_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l3689_368907


namespace NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_three_l3689_368914

theorem m_squared_plus_inverse_squared_plus_three (m : ℝ) (h : m + 1/m = 6) :
  m^2 + 1/m^2 + 3 = 37 := by sorry

end NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_three_l3689_368914


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l3689_368971

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
structure Point where
  x : ℝ
  y : ℝ

def P₀ : Point := ⟨-1, -4⟩

-- Define the third quadrant
def in_third_quadrant (p : Point) : Prop := p.x < 0 ∧ p.y < 0

-- Define the tangent line slope
def tangent_slope : ℝ := 4

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x + 4 * y + 17 = 0

theorem tangent_point_and_perpendicular_line :
  f P₀.x = P₀.y ∧
  f' P₀.x = tangent_slope ∧
  in_third_quadrant P₀ →
  perpendicular_line P₀.x P₀.y :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l3689_368971


namespace NUMINAMATH_CALUDE_intersection_M_N_l3689_368947

def M : Set ℝ := {-3, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3689_368947


namespace NUMINAMATH_CALUDE_x_value_theorem_l3689_368918

theorem x_value_theorem (x y z a b c : ℝ) 
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : a ≠ 0)
  (h5 : b ≠ 0)
  (h6 : c ≠ 0)
  (h7 : x + y + z = a * b * c) :
  x = 2 * a * b * c / (a * b + b * c + a * c) := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l3689_368918


namespace NUMINAMATH_CALUDE_sevenPointFourSix_eq_fraction_l3689_368977

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

end NUMINAMATH_CALUDE_sevenPointFourSix_eq_fraction_l3689_368977


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3689_368978

theorem cube_plus_reciprocal_cube (x : ℝ) (h1 : x > 0) (h2 : (x + 1/x)^2 = 25) :
  x^3 + 1/x^3 = 110 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3689_368978


namespace NUMINAMATH_CALUDE_milk_left_over_problem_l3689_368973

/-- Calculates the amount of milk left over given the total milk production,
    percentage consumed by kids, and percentage of remainder used for cooking. -/
def milk_left_over (total_milk : ℝ) (kids_consumption_percent : ℝ) (cooking_percent : ℝ) : ℝ :=
  let remaining_after_kids := total_milk * (1 - kids_consumption_percent)
  let used_for_cooking := remaining_after_kids * cooking_percent
  remaining_after_kids - used_for_cooking

/-- Proves that given 16 cups of milk, with 75% consumed by kids and 50% of the remainder
    used for cooking, the amount of milk left over is 2 cups. -/
theorem milk_left_over_problem :
  milk_left_over 16 0.75 0.50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_left_over_problem_l3689_368973


namespace NUMINAMATH_CALUDE_plant_species_numbering_not_unique_l3689_368910

theorem plant_species_numbering_not_unique : ∃ a b : ℕ, 
  2 ≤ a ∧ a < b ∧ b ≤ 20000 ∧ 
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 20000 → Nat.gcd a k = Nat.gcd b k) :=
sorry

end NUMINAMATH_CALUDE_plant_species_numbering_not_unique_l3689_368910


namespace NUMINAMATH_CALUDE_f_properties_l3689_368945

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 else x + 6/x - 6

theorem f_properties :
  (f (f (-2)) = -1/2) ∧
  (∀ x, f x ≥ 2 * Real.sqrt 6 - 6) ∧
  (∃ x, f x = 2 * Real.sqrt 6 - 6) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3689_368945


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3689_368960

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3689_368960


namespace NUMINAMATH_CALUDE_team_selection_count_l3689_368948

/-- The number of ways to select a team of 4 boys from 10 boys and 4 girls from 12 girls -/
def select_team : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem team_selection_count : select_team = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l3689_368948


namespace NUMINAMATH_CALUDE_intersection_uniqueness_l3689_368930

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := -2 * y = 7 * x - 3

/-- The intersection point -/
def intersection_point : ℚ × ℚ := (-3/17, 36/17)

theorem intersection_uniqueness :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_uniqueness_l3689_368930


namespace NUMINAMATH_CALUDE_value_of_x_l3689_368963

theorem value_of_x (n : ℝ) (x : ℝ) 
  (h1 : x = 3 * n) 
  (h2 : 2 * n + 3 = 0.20 * 25) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l3689_368963


namespace NUMINAMATH_CALUDE_inequality_system_solvability_l3689_368922

theorem inequality_system_solvability (n : ℕ) : 
  (∃ x : ℝ, 
    (1 < x ∧ x < 2) ∧
    (2 < x^2 ∧ x^2 < 3) ∧
    (∀ k : ℕ, 3 ≤ k ∧ k ≤ n → k < x^k ∧ x^k < k + 1)) ↔
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solvability_l3689_368922


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_angle_ratio_l3689_368932

/-- A triangle with angles in the ratio 2:2:1 is isosceles -/
theorem isosceles_triangle_from_angle_ratio (A B C : ℝ) 
  (h_sum : A + B + C = 180) 
  (h_ratio : ∃ (k : ℝ), A = 2*k ∧ B = 2*k ∧ C = k) : 
  A = B ∨ B = C ∨ A = C := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_angle_ratio_l3689_368932


namespace NUMINAMATH_CALUDE_rectangle_width_proof_l3689_368943

/-- Proves that the width of a rectangle is 14 cm given specific conditions -/
theorem rectangle_width_proof (length width perimeter : ℝ) (triangle_side : ℝ) : 
  length = 10 →
  perimeter = 2 * (length + width) →
  perimeter = 3 * triangle_side →
  triangle_side = 16 →
  width = 14 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_proof_l3689_368943


namespace NUMINAMATH_CALUDE_complex_magnitude_eighth_power_l3689_368981

theorem complex_magnitude_eighth_power : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_eighth_power_l3689_368981


namespace NUMINAMATH_CALUDE_james_fish_purchase_l3689_368996

theorem james_fish_purchase (fish_per_roll : ℕ) (bad_fish_percent : ℚ) (rolls_made : ℕ) :
  fish_per_roll = 40 →
  bad_fish_percent = 1/5 →
  rolls_made = 8 →
  ∃ (total_fish : ℕ), total_fish = 400 ∧ 
    (total_fish : ℚ) * (1 - bad_fish_percent) = (fish_per_roll * rolls_made : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_james_fish_purchase_l3689_368996


namespace NUMINAMATH_CALUDE_flower_city_theorem_l3689_368941

/-- A bipartite graph representing the relationship between short men and little girls -/
structure FlowerCityGraph where
  A : Type -- Set of short men
  B : Type -- Set of little girls
  edge : A → B → Prop -- Edge relation

/-- The property that each short man knows exactly 6 little girls -/
def each_man_knows_six_girls (G : FlowerCityGraph) : Prop :=
  ∀ a : G.A, (∃! (b1 b2 b3 b4 b5 b6 : G.B), 
    G.edge a b1 ∧ G.edge a b2 ∧ G.edge a b3 ∧ G.edge a b4 ∧ G.edge a b5 ∧ G.edge a b6 ∧
    (∀ b : G.B, G.edge a b → (b = b1 ∨ b = b2 ∨ b = b3 ∨ b = b4 ∨ b = b5 ∨ b = b6)))

/-- The property that each little girl knows exactly 6 short men -/
def each_girl_knows_six_men (G : FlowerCityGraph) : Prop :=
  ∀ b : G.B, (∃! (a1 a2 a3 a4 a5 a6 : G.A), 
    G.edge a1 b ∧ G.edge a2 b ∧ G.edge a3 b ∧ G.edge a4 b ∧ G.edge a5 b ∧ G.edge a6 b ∧
    (∀ a : G.A, G.edge a b → (a = a1 ∨ a = a2 ∨ a = a3 ∨ a = a4 ∨ a = a5 ∨ a = a6)))

/-- The theorem stating that the number of short men equals the number of little girls -/
theorem flower_city_theorem (G : FlowerCityGraph) 
  (h1 : each_man_knows_six_girls G) 
  (h2 : each_girl_knows_six_men G) : 
  Nonempty (Equiv G.A G.B) :=
sorry

end NUMINAMATH_CALUDE_flower_city_theorem_l3689_368941


namespace NUMINAMATH_CALUDE_math_only_count_l3689_368935

/-- Represents the number of students in various class combinations -/
structure ClassCounts where
  total : ℕ
  math : ℕ
  science : ℕ
  foreignLang : ℕ
  mathOnly : ℕ
  mathScience : ℕ
  mathForeign : ℕ
  scienceForeign : ℕ
  allThree : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem math_only_count (c : ClassCounts) : 
  c.total = 120 ∧ 
  c.math = 85 ∧ 
  c.science = 70 ∧ 
  c.foreignLang = 54 ∧ 
  c.total = c.math + c.science + c.foreignLang - c.mathScience - c.mathForeign - c.scienceForeign + c.allThree →
  c.mathOnly = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_only_count_l3689_368935


namespace NUMINAMATH_CALUDE_angle_C_measure_l3689_368964

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.b^2 + t.c^2 - Real.sqrt 3 * t.b * t.c = t.a^2 ∧
  t.b * t.c = Real.sqrt 3 * t.a^2

-- Theorem statement
theorem angle_C_measure (t : Triangle) 
  (h : satisfiesConditions t) : t.angleC = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l3689_368964


namespace NUMINAMATH_CALUDE_center_numbers_l3689_368962

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 4 × Fin 4) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if a grid satisfies the conditions of the problem -/
def validGrid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 17) ∧
  (∀ n : ℕ, n ∈ Finset.range 16 → ∃ p1 p2, g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ sharesEdge p1 p2) ∧
  (g 0 0 + g 0 3 + g 3 0 + g 3 3 = 34)

/-- The center 2x2 grid -/
def centerGrid (g : Grid) : Finset ℕ :=
  {g 1 1, g 1 2, g 2 1, g 2 2}

theorem center_numbers (g : Grid) (h : validGrid g) :
  centerGrid g = {9, 10, 11, 12} :=
sorry

end NUMINAMATH_CALUDE_center_numbers_l3689_368962


namespace NUMINAMATH_CALUDE_half_of_four_power_2022_l3689_368994

theorem half_of_four_power_2022 : (4 ^ 2022) / 2 = 2 ^ 4043 := by
  sorry

end NUMINAMATH_CALUDE_half_of_four_power_2022_l3689_368994


namespace NUMINAMATH_CALUDE_apple_cost_per_kg_l3689_368970

theorem apple_cost_per_kg (p q : ℚ) : 
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 := by sorry

end NUMINAMATH_CALUDE_apple_cost_per_kg_l3689_368970


namespace NUMINAMATH_CALUDE_cube_product_three_four_l3689_368903

theorem cube_product_three_four : (3 : ℕ)^3 * (4 : ℕ)^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_three_four_l3689_368903


namespace NUMINAMATH_CALUDE_square_area_equals_triangle_perimeter_l3689_368959

/-- Given a right-angled triangle with sides 6 cm and 8 cm, 
    a square with the same perimeter as this triangle has an area of 36 cm². -/
theorem square_area_equals_triangle_perimeter : 
  ∃ (triangle_hypotenuse : ℝ) (square_side : ℝ),
    triangle_hypotenuse^2 = 6^2 + 8^2 ∧ 
    6 + 8 + triangle_hypotenuse = 4 * square_side ∧
    square_side^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_triangle_perimeter_l3689_368959


namespace NUMINAMATH_CALUDE_sword_length_difference_is_23_l3689_368969

/-- The length difference between June's and Christopher's swords -/
def sword_length_difference : ℕ → ℕ → ℕ → ℕ
  | christopher_length, jameson_diff, june_diff =>
    let jameson_length := 2 * christopher_length + jameson_diff
    let june_length := jameson_length + june_diff
    june_length - christopher_length

theorem sword_length_difference_is_23 :
  sword_length_difference 15 3 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sword_length_difference_is_23_l3689_368969


namespace NUMINAMATH_CALUDE_square_of_1007_l3689_368988

theorem square_of_1007 : (1007 : ℕ)^2 = 1014049 := by sorry

end NUMINAMATH_CALUDE_square_of_1007_l3689_368988


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3689_368933

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - Real.log x - a*x

theorem min_value_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f x 1 ≥ min := by sorry

theorem f_greater_than_x_iff_a_negative :
  (∀ x > 0, f x a > x) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3689_368933


namespace NUMINAMATH_CALUDE_total_distance_two_wheels_l3689_368944

/-- The total distance covered by two wheels with different radii -/
theorem total_distance_two_wheels 
  (r1 r2 N : ℝ) 
  (h_positive : r1 > 0 ∧ r2 > 0 ∧ N > 0) : 
  let wheel1_revolutions : ℝ := 1500
  let wheel2_revolutions : ℝ := N * wheel1_revolutions
  let distance_wheel1 : ℝ := 2 * Real.pi * r1 * wheel1_revolutions
  let distance_wheel2 : ℝ := 2 * Real.pi * r2 * wheel2_revolutions
  let total_distance : ℝ := distance_wheel1 + distance_wheel2
  total_distance = 3000 * Real.pi * (r1 + N * r2) :=
by sorry

end NUMINAMATH_CALUDE_total_distance_two_wheels_l3689_368944


namespace NUMINAMATH_CALUDE_power_of_six_seven_equals_product_of_seven_sixes_l3689_368972

theorem power_of_six_seven_equals_product_of_seven_sixes :
  6^7 = (List.replicate 7 6).prod := by
  sorry

end NUMINAMATH_CALUDE_power_of_six_seven_equals_product_of_seven_sixes_l3689_368972


namespace NUMINAMATH_CALUDE_opposite_of_2024_l3689_368928

theorem opposite_of_2024 : -(2024 : ℤ) = -2024 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2024_l3689_368928


namespace NUMINAMATH_CALUDE_max_piece_length_and_total_pieces_l3689_368909

-- Define the lengths of the two pipes
def pipe1_length : ℕ := 42
def pipe2_length : ℕ := 63

-- Define the theorem
theorem max_piece_length_and_total_pieces :
  ∃ (max_length : ℕ) (total_pieces : ℕ),
    max_length = Nat.gcd pipe1_length pipe2_length ∧
    max_length = 21 ∧
    total_pieces = pipe1_length / max_length + pipe2_length / max_length ∧
    total_pieces = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_piece_length_and_total_pieces_l3689_368909


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3689_368951

theorem amount_after_two_years 
  (present_value : ℝ) 
  (yearly_increase_rate : ℝ) 
  (h1 : present_value = 57600) 
  (h2 : yearly_increase_rate = 1/8) 
  (h3 : (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900) : 
  (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3689_368951


namespace NUMINAMATH_CALUDE_jake_lawn_mowing_earnings_l3689_368946

/-- Jake's desired hourly rate in dollars -/
def desired_hourly_rate : ℝ := 20

/-- Time taken to mow the lawn in hours -/
def lawn_mowing_time : ℝ := 1

/-- Time taken to plant flowers in hours -/
def flower_planting_time : ℝ := 2

/-- Total charge for planting flowers in dollars -/
def flower_planting_charge : ℝ := 45

/-- Earnings for mowing the lawn in dollars -/
def lawn_mowing_earnings : ℝ := desired_hourly_rate * lawn_mowing_time

theorem jake_lawn_mowing_earnings :
  lawn_mowing_earnings = 20 := by sorry

end NUMINAMATH_CALUDE_jake_lawn_mowing_earnings_l3689_368946


namespace NUMINAMATH_CALUDE_optimal_price_l3689_368967

/-- Represents the daily sales profit function for an agricultural product. -/
def W (x : ℝ) : ℝ := -2 * x^2 + 120 * x - 1600

/-- Represents the daily sales quantity function for an agricultural product. -/
def y (x : ℝ) : ℝ := -2 * x + 80

/-- The cost price per kilogram of the agricultural product. -/
def cost_price : ℝ := 20

/-- The maximum allowed selling price per kilogram. -/
def max_price : ℝ := 30

/-- The desired daily sales profit. -/
def target_profit : ℝ := 150

/-- Theorem stating that a selling price of 25 achieves the target profit
    while satisfying the given conditions. -/
theorem optimal_price :
  W 25 = target_profit ∧
  25 ≤ max_price ∧
  y 25 > 0 :=
sorry

end NUMINAMATH_CALUDE_optimal_price_l3689_368967


namespace NUMINAMATH_CALUDE_adjacent_numbers_to_10000_l3689_368915

theorem adjacent_numbers_to_10000 :
  let adjacent_numbers (n : ℤ) := (n - 1, n + 1)
  adjacent_numbers 10000 = (9999, 10001) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_numbers_to_10000_l3689_368915


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3689_368956

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 3*y + m = 0 → y = x) → 
  m = 9/4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3689_368956


namespace NUMINAMATH_CALUDE_marys_bedrooms_l3689_368942

/-- Represents the number of rooms in a house -/
structure House where
  bedrooms : ℕ
  kitchen : Unit
  livingRoom : Unit

/-- Represents a vacuum cleaner -/
structure VacuumCleaner where
  batteryLife : ℕ  -- in minutes
  chargingTimes : ℕ

/-- Represents the time it takes to vacuum a room -/
def roomVacuumTime : ℕ := 4

theorem marys_bedrooms (h : House) (v : VacuumCleaner)
    (hv : v.batteryLife = 10 ∧ v.chargingTimes = 2) :
    h.bedrooms = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_bedrooms_l3689_368942


namespace NUMINAMATH_CALUDE_quadratic_root_range_l3689_368952

theorem quadratic_root_range (a : ℝ) : 
  (∃ x y : ℝ, x > 1 ∧ y < -1 ∧ 
   x^2 + (a^2 + 1)*x + a - 2 = 0 ∧
   y^2 + (a^2 + 1)*y + a - 2 = 0) →
  -1 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l3689_368952


namespace NUMINAMATH_CALUDE_vet_fees_for_cats_l3689_368953

theorem vet_fees_for_cats (dog_fee : ℝ) (num_dogs : ℕ) (num_cats : ℕ) (donation_fraction : ℝ) (total_donation : ℝ) :
  dog_fee = 15 →
  num_dogs = 8 →
  num_cats = 3 →
  donation_fraction = 1/3 →
  total_donation = 53 →
  ∃ (cat_fee : ℝ), cat_fee = 13 ∧ 
    donation_fraction * (num_dogs * dog_fee + num_cats * cat_fee) = total_donation :=
by sorry

end NUMINAMATH_CALUDE_vet_fees_for_cats_l3689_368953


namespace NUMINAMATH_CALUDE_M_when_a_is_one_M_union_N_equals_N_l3689_368998

-- Define the set M as a function of a
def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}

-- Define the set N
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem 1: When a = 1, M is the open interval (0, 2)
theorem M_when_a_is_one : M 1 = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem 2: M ∪ N = N if and only if a ∈ [-1, 2]
theorem M_union_N_equals_N (a : ℝ) : M a ∪ N = N ↔ -1 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_M_when_a_is_one_M_union_N_equals_N_l3689_368998


namespace NUMINAMATH_CALUDE_solve_selinas_shirt_sales_l3689_368924

/-- Represents the problem of determining how many shirts Selina sold. -/
def SelinasShirtSales : Prop :=
  let pants_price : ℕ := 5
  let shorts_price : ℕ := 3
  let shirt_price : ℕ := 4
  let pants_sold : ℕ := 3
  let shorts_sold : ℕ := 5
  let shirts_bought : ℕ := 2
  let shirt_buy_price : ℕ := 10
  let remaining_money : ℕ := 30
  ∃ (shirts_sold : ℕ),
    shirts_sold * shirt_price + 
    pants_sold * pants_price + 
    shorts_sold * shorts_price = 
    remaining_money + shirts_bought * shirt_buy_price ∧
    shirts_sold = 5

theorem solve_selinas_shirt_sales : SelinasShirtSales := by
  sorry

#check solve_selinas_shirt_sales

end NUMINAMATH_CALUDE_solve_selinas_shirt_sales_l3689_368924


namespace NUMINAMATH_CALUDE_right_triangle_3_4_5_l3689_368995

theorem right_triangle_3_4_5 (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 :=
by
  sorry

#check right_triangle_3_4_5

end NUMINAMATH_CALUDE_right_triangle_3_4_5_l3689_368995


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l3689_368931

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) :
  let semicircle_square_area := (4 / 5) * r^2
  let equilateral_triangle_side := 2 * r
  let triangle_square_area := r^2
  semicircle_square_area / triangle_square_area = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l3689_368931


namespace NUMINAMATH_CALUDE_anya_hair_growth_l3689_368991

/-- The number of hairs Anya washes down the drain -/
def washed_hairs : ℕ := 32

/-- The number of hairs Anya brushes out -/
def brushed_hairs : ℕ := washed_hairs / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := washed_hairs + brushed_hairs + 1

theorem anya_hair_growth :
  hairs_to_grow = 49 :=
by sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l3689_368991


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3689_368920

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750)
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l3689_368920


namespace NUMINAMATH_CALUDE_equation_solutions_l3689_368923

theorem equation_solutions : 
  {x : ℝ | (1 / (x^2 + 11*x - 8) + 1 / (x^2 + 2*x - 8) + 1 / (x^2 - 13*x - 8) = 0)} = 
  {8, 1, -1, -8} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3689_368923


namespace NUMINAMATH_CALUDE_absolute_value_greater_than_two_l3689_368949

theorem absolute_value_greater_than_two (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_greater_than_two_l3689_368949


namespace NUMINAMATH_CALUDE_mass_of_apples_left_correct_l3689_368937

/-- Calculates the mass of apples left after sales -/
def mass_of_apples_left (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) : ℕ :=
  (kidney_apples + golden_apples + canada_apples) - apples_sold

/-- Proves that the mass of apples left is correct given the initial masses and the mass of apples sold -/
theorem mass_of_apples_left_correct 
  (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) :
  mass_of_apples_left kidney_apples golden_apples canada_apples apples_sold =
  (kidney_apples + golden_apples + canada_apples) - apples_sold :=
by
  sorry

/-- Verifies the specific case in the problem -/
example : mass_of_apples_left 23 37 14 36 = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_mass_of_apples_left_correct_l3689_368937


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l3689_368906

/-- Given a man's rowing speeds with and against a stream, calculates his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 20) 
  (h2 : speed_against_stream = 8) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l3689_368906


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l3689_368980

/-- Parabola with focus F and equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * C.p * x

/-- Circle intersecting y-axis -/
structure IntersectingCircle (M : PointOnParabola C) where
  radius : ℝ
  chord_length : ℝ
  h_chord : chord_length = 2 * Real.sqrt 5
  h_radius_eq : radius^2 = M.x^2 + 5

/-- Line intersecting parabola -/
structure IntersectingLine (C : Parabola) where
  slope : ℝ
  x_intercept : ℝ
  h_slope : slope = Real.pi / 4
  h_intercept : x_intercept = 2

/-- Intersection points of line and parabola -/
structure IntersectionPoints (C : Parabola) (l : IntersectingLine C) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h_on_parabola₁ : y₁^2 = 2 * C.p * x₁
  h_on_parabola₂ : y₂^2 = 2 * C.p * x₂
  h_on_line₁ : y₁ = l.slope * (x₁ - l.x_intercept)
  h_on_line₂ : y₂ = l.slope * (x₂ - l.x_intercept)

theorem parabola_circle_intersection
  (C : Parabola)
  (M : PointOnParabola C)
  (circle : IntersectingCircle M)
  (l : IntersectingLine C)
  (points : IntersectionPoints C l) :
  circle.radius = 3 ∧ x₁ * x₂ + y₁ * y₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l3689_368980


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3689_368925

def problem1 (p q : ℝ × ℝ) : Prop :=
  p = (1, 2) ∧ 
  ∃ m : ℝ, q = (m, 1) ∧ 
  p.1 * q.1 + p.2 * q.2 = 0 →
  ‖q‖ = Real.sqrt 5

theorem vector_perpendicular : problem1 (1, 2) (-2, 1) := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3689_368925


namespace NUMINAMATH_CALUDE_phone_number_probability_l3689_368921

theorem phone_number_probability : 
  ∀ (n : ℕ) (p : ℚ),
    n = 10 →  -- There are 10 possible digits
    p = 1 / n →  -- Probability of correct guess on each attempt
    (p + p + p) = 3 / 10  -- Probability of success in no more than 3 attempts
    := by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l3689_368921


namespace NUMINAMATH_CALUDE_divisible_by_four_count_l3689_368900

theorem divisible_by_four_count :
  (∃! (n : Nat), n = (Finset.filter (fun d => (10 * d + 4) % 4 = 0) (Finset.range 10)).card ∧ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_count_l3689_368900


namespace NUMINAMATH_CALUDE_max_triangle_sum_l3689_368990

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9, 10}

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
  d ∈ triangle_numbers ∧ e ∈ triangle_numbers ∧ f ∈ triangle_numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def side_sum (a b c : ℕ) : ℕ := a + b + c

def equal_sums (a b c d e f : ℕ) : Prop :=
  side_sum a b c = side_sum c d e ∧
  side_sum c d e = side_sum e f a

theorem max_triangle_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    equal_sums a b c d e f →
    side_sum a b c ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_sum_l3689_368990


namespace NUMINAMATH_CALUDE_perpendicular_tangents_point_l3689_368985

/-- The point on the line y = x from which two perpendicular tangents 
    can be drawn to the parabola y = x^2 -/
theorem perpendicular_tangents_point :
  ∃! P : ℝ × ℝ, 
    (P.1 = P.2) ∧ 
    (∃ m₁ m₂ : ℝ, 
      (m₁ * m₂ = -1) ∧
      (∀ x y : ℝ, y = m₁ * (x - P.1) + P.2 → y = x^2 → x = P.1) ∧
      (∀ x y : ℝ, y = m₂ * (x - P.1) + P.2 → y = x^2 → x = P.1)) ∧
    P = (-1/4, -1/4) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_point_l3689_368985


namespace NUMINAMATH_CALUDE_polynomial_sum_l3689_368992

/-- Two distinct polynomials with real coefficients -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

/-- The theorem statement -/
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ x, f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (∃ x, g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∀ x, f a b x ≥ -144) →          -- minimum value of f is -144
  (∀ x, g c d x ≥ -144) →          -- minimum value of g is -144
  (∃ x, f a b x = -144) →          -- f achieves its minimum
  (∃ x, g c d x = -144) →          -- g achieves its minimum
  f a b 150 = -200 →               -- f(150) = -200
  g c d 150 = -200 →               -- g(150) = -200
  a + c = -300 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3689_368992


namespace NUMINAMATH_CALUDE_blood_type_sample_size_l3689_368927

/-- Given a population of students with known blood types, calculate the number of students
    with a specific blood type that should be drawn in a stratified sample. -/
theorem blood_type_sample_size (total_students sample_size blood_type_O : ℕ)
    (h1 : total_students = 500)
    (h2 : blood_type_O = 200)
    (h3 : sample_size = 40) :
    (blood_type_O : ℚ) / total_students * sample_size = 16 := by
  sorry


end NUMINAMATH_CALUDE_blood_type_sample_size_l3689_368927


namespace NUMINAMATH_CALUDE_emilee_earnings_l3689_368979

/-- Proves that Emilee earns $25 given the conditions of the problem -/
theorem emilee_earnings (total : ℕ) (terrence_earnings : ℕ) (jermaine_extra : ℕ) :
  total = 90 →
  terrence_earnings = 30 →
  jermaine_extra = 5 →
  total = terrence_earnings + (terrence_earnings + jermaine_extra) + (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) →
  (total - (terrence_earnings + (terrence_earnings + jermaine_extra))) = 25 := by
  sorry

#check emilee_earnings

end NUMINAMATH_CALUDE_emilee_earnings_l3689_368979


namespace NUMINAMATH_CALUDE_probability_third_ball_white_l3689_368905

-- Define the problem setup
theorem probability_third_ball_white (n : ℕ) (h : n > 2) :
  let bags := Finset.range n
  let balls_in_bag (k : ℕ) := k + (n - k)
  let prob_choose_bag := 1 / n
  let prob_white_third (k : ℕ) := (n - k) / n
  (bags.sum (λ k => prob_choose_bag * prob_white_third k)) = (n - 1) / (2 * n) :=
by sorry


end NUMINAMATH_CALUDE_probability_third_ball_white_l3689_368905


namespace NUMINAMATH_CALUDE_largest_circle_radius_is_b_l3689_368986

/-- An ellipsoid with semi-axes a > b > c -/
structure Ellipsoid where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a > b
  h2 : b > c

/-- The radius of the largest circle on an ellipsoid -/
def largest_circle_radius (e : Ellipsoid) : ℝ := e.b

/-- Theorem: The radius of the largest circle on an ellipsoid with semi-axes a > b > c is b -/
theorem largest_circle_radius_is_b (e : Ellipsoid) :
  largest_circle_radius e = e.b :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_radius_is_b_l3689_368986


namespace NUMINAMATH_CALUDE_kira_downloaded_songs_l3689_368902

/-- The size of each song in megabytes -/
def song_size : ℕ := 5

/-- The total size of new songs in megabytes -/
def total_new_size : ℕ := 140

/-- The number of songs downloaded later on that day -/
def songs_downloaded : ℕ := total_new_size / song_size

theorem kira_downloaded_songs :
  songs_downloaded = 28 := by sorry

end NUMINAMATH_CALUDE_kira_downloaded_songs_l3689_368902


namespace NUMINAMATH_CALUDE_point_not_outside_implies_on_or_inside_l3689_368901

-- Define a circle in a 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the possible relationships between a point and a circle
inductive PointCircleRelation
  | Inside
  | On
  | Outside

-- Function to determine the relation between a point and a circle
def pointCircleRelation (p : ℝ × ℝ) (c : Circle) : PointCircleRelation :=
  sorry

-- Theorem statement
theorem point_not_outside_implies_on_or_inside
  (p : ℝ × ℝ) (c : Circle) :
  pointCircleRelation p c ≠ PointCircleRelation.Outside →
  (pointCircleRelation p c = PointCircleRelation.On ∨
   pointCircleRelation p c = PointCircleRelation.Inside) :=
by sorry

end NUMINAMATH_CALUDE_point_not_outside_implies_on_or_inside_l3689_368901


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l3689_368908

def QuadraticFunction (a h k : ℝ) : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k

theorem quadratic_function_theorem (a : ℝ) (h k : ℝ) :
  (∀ x, QuadraticFunction a h k x ≤ 2) ∧
  QuadraticFunction a h k 2 = 1 ∧
  QuadraticFunction a h k 4 = 1 →
  h = 3 ∧ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l3689_368908


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_asymptotes_l3689_368954

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 - y^2/25 = 1

-- Define the hyperbola
def hyperbola (K : ℝ) (x y : ℝ) : Prop := x^2/K + y^2/25 = 1

-- Define the asymptote condition
def same_asymptotes (K : ℝ) : Prop := ∀ (x y : ℝ), y = (5/4)*x ↔ y = (5/Real.sqrt K)*x

-- Theorem statement
theorem ellipse_hyperbola_asymptotes (K : ℝ) : 
  (∀ (x y : ℝ), ellipse x y ∧ hyperbola K x y) → same_asymptotes K → K = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_asymptotes_l3689_368954


namespace NUMINAMATH_CALUDE_all_two_digit_numbers_appear_l3689_368936

/-- Represents a sequence of numbers from 1 to 1,000,000 in arbitrary order -/
def ArbitrarySequence := Fin 1000000 → Fin 1000000

/-- Represents a two-digit number (from 10 to 99) -/
def TwoDigitNumber := Fin 90

/-- A function that checks if a given two-digit number appears in the sequence when cut into two-digit pieces -/
def appearsInSequence (seq : ArbitrarySequence) (n : TwoDigitNumber) : Prop :=
  ∃ i : Fin 999999, (seq i).val / 100 % 100 = n.val + 10 ∨ (seq i).val % 100 = n.val + 10

/-- The main theorem statement -/
theorem all_two_digit_numbers_appear (seq : ArbitrarySequence) :
  ∀ n : TwoDigitNumber, appearsInSequence seq n :=
sorry

end NUMINAMATH_CALUDE_all_two_digit_numbers_appear_l3689_368936
