import Mathlib

namespace NUMINAMATH_CALUDE_marys_average_speed_l3832_383209

/-- Mary's round trip walking problem -/
theorem marys_average_speed (uphill_distance downhill_distance : ℝ)
                             (uphill_time downhill_time : ℝ)
                             (h1 : uphill_distance = 1.5)
                             (h2 : downhill_distance = 1.5)
                             (h3 : uphill_time = 45 / 60)
                             (h4 : downhill_time = 15 / 60) :
  (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end NUMINAMATH_CALUDE_marys_average_speed_l3832_383209


namespace NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l3832_383218

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a polygon is inside a triangle --/
def isInside (p : Polygon) (t : Triangle) : Prop := sorry

/-- Theorem: The largest possible area of a centrally symmetric polygon inside a triangle is 2/3 of the triangle's area --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) :
  ∃ (p : Polygon), isCentrallySymmetric p ∧ isInside p t ∧
    ∀ (q : Polygon), isCentrallySymmetric q → isInside q t →
      area p ≥ area q ∧ area p = (2/3) * area (Polygon.mk [t.A, t.B, t.C]) :=
sorry

end NUMINAMATH_CALUDE_largest_centrally_symmetric_polygon_area_l3832_383218


namespace NUMINAMATH_CALUDE_expression_simplification_l3832_383225

theorem expression_simplification :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3832_383225


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3832_383204

theorem largest_gold_coins_distribution (total : ℕ) : 
  (∃ (k : ℕ), total = 13 * k + 3) →
  total < 150 →
  (∀ n : ℕ, (∃ (k : ℕ), n = 13 * k + 3) → n < 150 → n ≤ total) →
  total = 146 :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3832_383204


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3832_383254

theorem fraction_multiplication (x : ℝ) : 
  (1 : ℝ) / 3 * 2 / 7 * 9 / 13 * x / 17 = 18 * x / 4911 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3832_383254


namespace NUMINAMATH_CALUDE_jackson_school_supplies_cost_l3832_383293

/-- Calculates the total cost of school supplies for a class, given the number of students,
    item quantities per student, item costs, and a teacher discount. -/
def totalCostOfSupplies (students : ℕ) 
                        (penPerStudent notebookPerStudent binderPerStudent highlighterPerStudent : ℕ)
                        (penCost notebookCost binderCost highlighterCost : ℚ)
                        (teacherDiscount : ℚ) : ℚ :=
  let totalPens := students * penPerStudent
  let totalNotebooks := students * notebookPerStudent
  let totalBinders := students * binderPerStudent
  let totalHighlighters := students * highlighterPerStudent
  let totalCost := totalPens * penCost + totalNotebooks * notebookCost + 
                   totalBinders * binderCost + totalHighlighters * highlighterCost
  totalCost - teacherDiscount

/-- Theorem stating that the total cost of school supplies for Jackson's class is $858.25 -/
theorem jackson_school_supplies_cost : 
  totalCostOfSupplies 45 6 4 2 3 (65/100) (145/100) (480/100) (85/100) 125 = 85825/100 := by
  sorry

end NUMINAMATH_CALUDE_jackson_school_supplies_cost_l3832_383293


namespace NUMINAMATH_CALUDE_cody_chocolate_boxes_cody_bought_seven_boxes_l3832_383269

theorem cody_chocolate_boxes : ℕ → Prop :=
  fun x =>
    -- x is the number of boxes of chocolate candy
    -- 3 is the number of boxes of caramel candy
    -- 8 is the number of pieces in each box
    -- 80 is the total number of pieces
    x * 8 + 3 * 8 = 80 →
    x = 7

-- The proof
theorem cody_bought_seven_boxes : cody_chocolate_boxes 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_chocolate_boxes_cody_bought_seven_boxes_l3832_383269


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l3832_383275

theorem quadratic_perfect_square (x : ℝ) : ∃ (a b : ℝ), x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l3832_383275


namespace NUMINAMATH_CALUDE_daylight_duration_l3832_383299

/-- Given a day with 24 hours and a daylight to nighttime ratio of 9:7, 
    the duration of daylight is 13.5 hours. -/
theorem daylight_duration (total_hours : ℝ) (daylight_ratio nighttime_ratio : ℕ) 
    (h1 : total_hours = 24)
    (h2 : daylight_ratio = 9)
    (h3 : nighttime_ratio = 7) :
  (daylight_ratio : ℝ) / (daylight_ratio + nighttime_ratio : ℝ) * total_hours = 13.5 := by
sorry

end NUMINAMATH_CALUDE_daylight_duration_l3832_383299


namespace NUMINAMATH_CALUDE_unique_base_conversion_l3832_383208

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : ℕ) (b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion : 
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 45 = baseBToBase10 113 b :=
by sorry

end NUMINAMATH_CALUDE_unique_base_conversion_l3832_383208


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_2999_l3832_383283

theorem largest_prime_factor_of_2999 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2999 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2999 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_2999_l3832_383283


namespace NUMINAMATH_CALUDE_parabola_vertex_in_first_quadrant_l3832_383220

/-- Given a parabola y = -x^2 + (a+1)x + (a+2) where a > 1, 
    its vertex lies in the first quadrant -/
theorem parabola_vertex_in_first_quadrant (a : ℝ) (h : a > 1) :
  let f (x : ℝ) := -x^2 + (a+1)*x + (a+2)
  let vertex_x := (a+1)/2
  let vertex_y := f vertex_x
  vertex_x > 0 ∧ vertex_y > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_in_first_quadrant_l3832_383220


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3832_383246

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3832_383246


namespace NUMINAMATH_CALUDE_maya_total_pages_l3832_383229

/-- The total number of pages Maya read in two weeks -/
def total_pages (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

/-- Theorem stating that Maya read 4500 pages in total -/
theorem maya_total_pages :
  total_pages 5 300 2 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_maya_total_pages_l3832_383229


namespace NUMINAMATH_CALUDE_remaining_money_after_bike_purchase_l3832_383205

/-- Calculates the remaining money after buying a bike with quarters from jars --/
theorem remaining_money_after_bike_purchase (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℕ) : 
  num_jars = 5 → 
  quarters_per_jar = 160 → 
  bike_cost = 180 → 
  (num_jars * quarters_per_jar * 25 - bike_cost * 100) / 100 = 20 := by
  sorry

#check remaining_money_after_bike_purchase

end NUMINAMATH_CALUDE_remaining_money_after_bike_purchase_l3832_383205


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_l3832_383265

/-- The locus of the midpoint of chord MN on a parabola -/
theorem parabola_midpoint_locus (p : ℝ) (x y : ℝ) :
  let parabola := fun (x y : ℝ) => y^2 - 2*p*x = 0
  let normal_intersection := fun (x y m : ℝ) => y - m*x + p*(m + m^3/2) = 0
  let conjugate_diameter := fun (y m : ℝ) => m*y - p = 0
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    normal_intersection x₂ y₂ m ∧
    conjugate_diameter y₁ m ∧
    x = (x₁ + x₂) / 2 ∧
    y = (y₁ + y₂) / 2
  →
  y^4 - (p*x)*y^2 + p^4/2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_l3832_383265


namespace NUMINAMATH_CALUDE_equation_solution_l3832_383292

theorem equation_solution : ∃ (q r : ℝ), 
  q ≠ r ∧ 
  q > r ∧
  ((5 * q - 15) / (q^2 + q - 20) = q + 3) ∧
  ((5 * r - 15) / (r^2 + r - 20) = r + 3) ∧
  q - r = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3832_383292


namespace NUMINAMATH_CALUDE_estimate_smaller_than_actual_l3832_383279

theorem estimate_smaller_than_actual (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  (x - z) - (y + z) < x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_smaller_than_actual_l3832_383279


namespace NUMINAMATH_CALUDE_circle_center_sum_l3832_383231

theorem circle_center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 = 6*X - 8*Y + 24 ↔ (X - x)^2 + (Y - y)^2 = (x^2 + y^2 - 6*x + 8*y - 24)) →
  x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3832_383231


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3832_383282

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ x ≤ 23 ∧ (1055 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1055 + y) % 23 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3832_383282


namespace NUMINAMATH_CALUDE_charity_donation_percentage_l3832_383298

theorem charity_donation_percentage 
  (total_raised : ℝ)
  (num_organizations : ℕ)
  (amount_per_org : ℝ)
  (h1 : total_raised = 2500)
  (h2 : num_organizations = 8)
  (h3 : amount_per_org = 250) :
  (num_organizations * amount_per_org) / total_raised * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_charity_donation_percentage_l3832_383298


namespace NUMINAMATH_CALUDE_max_product_sum_l3832_383256

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) ∧
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_l3832_383256


namespace NUMINAMATH_CALUDE_trig_equality_l3832_383237

theorem trig_equality (θ : ℝ) (h : Real.sin (θ + π/3) = 2/3) : 
  Real.cos (θ - π/6) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_equality_l3832_383237


namespace NUMINAMATH_CALUDE_truncated_pyramid_volume_l3832_383234

/-- Given a truncated pyramid with base areas S₁ and S₂ (S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume 
  (S₁ S₂ V : ℝ) 
  (h₁ : 0 < S₁) 
  (h₂ : 0 < S₂) 
  (h₃ : S₁ < S₂) 
  (h₄ : 0 < V) : 
  ∃ (V_full : ℝ), V_full = (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁) := by
  sorry

end NUMINAMATH_CALUDE_truncated_pyramid_volume_l3832_383234


namespace NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_fifty_l3832_383238

theorem largest_multiple_of_seven_less_than_negative_fifty :
  ∀ n : ℤ, 7 ∣ n ∧ n < -50 → n ≤ -56 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_seven_less_than_negative_fifty_l3832_383238


namespace NUMINAMATH_CALUDE_fruit_salad_weight_l3832_383212

/-- The amount of melon in pounds used in the fruit salad -/
def melon_weight : ℚ := 0.25

/-- The amount of berries in pounds used in the fruit salad -/
def berries_weight : ℚ := 0.38

/-- The total amount of fruit in pounds used in the fruit salad -/
def total_fruit_weight : ℚ := melon_weight + berries_weight

theorem fruit_salad_weight : total_fruit_weight = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_weight_l3832_383212


namespace NUMINAMATH_CALUDE_total_amount_is_fifteen_l3832_383296

/-- Represents the share distribution among three people -/
structure ShareDistribution where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the total amount given a share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: Given the specified share distribution and the first person's share, 
    the total amount is 15 rupees -/
theorem total_amount_is_fifteen 
  (s : ShareDistribution) 
  (h1 : s.first = 10)
  (h2 : s.second = 0.3 * s.first)
  (h3 : s.third = 0.2 * s.first) : 
  totalAmount s = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_is_fifteen_l3832_383296


namespace NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3832_383268

/-- Calculates the percentage of weight loss during beef processing. -/
theorem beef_weight_loss_percentage 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 876.9230769230769) 
  (h2 : weight_after = 570) : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_loss_percentage_l3832_383268


namespace NUMINAMATH_CALUDE_jesses_room_width_l3832_383291

/-- Proves that the width of Jesse's room is 12 feet -/
theorem jesses_room_width (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) :
  length = 2 →
  tile_area = 4 →
  num_tiles = 6 →
  (length * (tile_area * num_tiles / length : ℝ) = length * 12) :=
by
  sorry

end NUMINAMATH_CALUDE_jesses_room_width_l3832_383291


namespace NUMINAMATH_CALUDE_pqu_theorem_l3832_383240

/-- A structure representing the relationship between P, Q, and U -/
structure PQU where
  P : ℝ
  Q : ℝ
  U : ℝ
  k : ℝ
  h : P = k * Q / U

/-- The theorem statement -/
theorem pqu_theorem (x y : PQU) (h1 : x.P = 6) (h2 : x.U = 4) (h3 : x.Q = 8)
                    (h4 : y.P = 18) (h5 : y.U = 9) : y.Q = 54 := by
  sorry

end NUMINAMATH_CALUDE_pqu_theorem_l3832_383240


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3832_383262

theorem arithmetic_calculation : 12 - 10 + 15 / 5 * 8 + 7 - 6 * 4 + 3 - 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3832_383262


namespace NUMINAMATH_CALUDE_intersection_sum_is_eight_l3832_383228

noncomputable def P : ℝ × ℝ := (0, 8)
noncomputable def Q : ℝ × ℝ := (0, 0)
noncomputable def R : ℝ × ℝ := (10, 0)

noncomputable def G : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
noncomputable def H : ℝ × ℝ := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

noncomputable def line_PH (x : ℝ) : ℝ := 
  (H.2 - P.2) / (H.1 - P.1) * (x - P.1) + P.2

theorem intersection_sum_is_eight : 
  ∃ (I : ℝ × ℝ), I.1 = G.1 ∧ I.2 = line_PH I.1 ∧ I.1 + I.2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_is_eight_l3832_383228


namespace NUMINAMATH_CALUDE_save_sign_white_area_l3832_383255

/-- Represents the area covered by a letter on the sign -/
structure LetterArea where
  s : ℕ
  a : ℕ
  v : ℕ
  e : ℕ

/-- The sign with the word "SAVE" painted on it -/
structure Sign where
  width : ℕ
  height : ℕ
  letterAreas : LetterArea

/-- Calculate the white area of the sign -/
def whiteArea (sign : Sign) : ℕ :=
  sign.width * sign.height - (sign.letterAreas.s + sign.letterAreas.a + sign.letterAreas.v + sign.letterAreas.e)

/-- Theorem stating the white area of the sign is 86 square units -/
theorem save_sign_white_area :
  ∀ (sign : Sign),
    sign.width = 20 ∧
    sign.height = 7 ∧
    sign.letterAreas.s = 14 ∧
    sign.letterAreas.a = 16 ∧
    sign.letterAreas.v = 12 ∧
    sign.letterAreas.e = 12 →
    whiteArea sign = 86 := by
  sorry

end NUMINAMATH_CALUDE_save_sign_white_area_l3832_383255


namespace NUMINAMATH_CALUDE_initial_bonus_is_500_l3832_383263

/-- Represents the bonus calculation for a teacher based on student test scores. -/
structure BonusCalculation where
  numStudents : Nat
  baseAverage : Nat
  bonusThreshold : Nat
  bonusPerPoint : Nat
  maxScore : Nat
  gradedTests : Nat
  gradedAverage : Nat
  lastTwoTestsScore : Nat
  totalBonus : Nat

/-- Calculates the initial bonus amount given the bonus calculation parameters. -/
def initialBonusAmount (bc : BonusCalculation) : Nat :=
  sorry

/-- Theorem stating that given the specific conditions, the initial bonus amount is $500. -/
theorem initial_bonus_is_500 (bc : BonusCalculation) 
  (h1 : bc.numStudents = 10)
  (h2 : bc.baseAverage = 75)
  (h3 : bc.bonusThreshold = 75)
  (h4 : bc.bonusPerPoint = 10)
  (h5 : bc.maxScore = 150)
  (h6 : bc.gradedTests = 8)
  (h7 : bc.gradedAverage = 70)
  (h8 : bc.lastTwoTestsScore = 290)
  (h9 : bc.totalBonus = 600) :
  initialBonusAmount bc = 500 :=
sorry

end NUMINAMATH_CALUDE_initial_bonus_is_500_l3832_383263


namespace NUMINAMATH_CALUDE_nigels_initial_amount_l3832_383217

theorem nigels_initial_amount 
  (olivia_initial : ℕ) 
  (ticket_price : ℕ) 
  (num_tickets : ℕ) 
  (amount_left : ℕ) 
  (h1 : olivia_initial = 112)
  (h2 : ticket_price = 28)
  (h3 : num_tickets = 6)
  (h4 : amount_left = 83) :
  olivia_initial + (ticket_price * num_tickets - (olivia_initial - amount_left)) = 251 :=
by sorry

end NUMINAMATH_CALUDE_nigels_initial_amount_l3832_383217


namespace NUMINAMATH_CALUDE_company_employees_l3832_383271

/-- If a company has 15% more employees in December than in January,
    and it has 490 employees in December, then it had 426 employees in January. -/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 490 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 426 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_l3832_383271


namespace NUMINAMATH_CALUDE_number_of_divisors_5005_l3832_383252

theorem number_of_divisors_5005 : Nat.card (Nat.divisors 5005) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_5005_l3832_383252


namespace NUMINAMATH_CALUDE_square_of_difference_l3832_383215

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3832_383215


namespace NUMINAMATH_CALUDE_marcus_earnings_l3832_383223

/-- Represents Marcus's work and earnings over two weeks -/
structure MarcusWork where
  hourly_wage : ℝ
  hours_week1 : ℝ
  hours_week2 : ℝ
  earnings_difference : ℝ

/-- Calculates the total earnings for two weeks given Marcus's work data -/
def total_earnings (w : MarcusWork) : ℝ :=
  w.hourly_wage * (w.hours_week1 + w.hours_week2)

theorem marcus_earnings :
  ∀ w : MarcusWork,
  w.hours_week1 = 12 ∧
  w.hours_week2 = 18 ∧
  w.earnings_difference = 36 ∧
  w.hourly_wage * (w.hours_week2 - w.hours_week1) = w.earnings_difference →
  total_earnings w = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_marcus_earnings_l3832_383223


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l3832_383207

theorem min_value_of_quadratic_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l3832_383207


namespace NUMINAMATH_CALUDE_village_population_l3832_383247

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 23040 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 25600 := by
sorry

end NUMINAMATH_CALUDE_village_population_l3832_383247


namespace NUMINAMATH_CALUDE_rectangle_side_problem_l3832_383274

theorem rectangle_side_problem (side1 : ℝ) (side2 : ℕ) (unknown_side : ℝ) : 
  side1 = 5 →
  side2 = 12 →
  (side1 * side2 = side1 * unknown_side + 25 ∨ side1 * unknown_side = side1 * side2 + 25) →
  unknown_side = 7 ∨ unknown_side = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_problem_l3832_383274


namespace NUMINAMATH_CALUDE_quadratic_integer_conjugate_theorem_l3832_383288

/-- A structure representing a quadratic integer of the form a + b√d -/
structure QuadraticInteger (d : ℕ) where
  a : ℤ
  b : ℤ

/-- The conjugate of a quadratic integer -/
def conjugate {d : ℕ} (z : QuadraticInteger d) : QuadraticInteger d :=
  ⟨z.a, -z.b⟩

theorem quadratic_integer_conjugate_theorem
  (d : ℕ) (x₀ y₀ x y X Y : ℤ) (r : ℕ) 
  (h_d : ¬ ∃ (n : ℕ), n ^ 2 = d)
  (h_pos : x₀ > 0 ∧ y₀ > 0 ∧ x > 0 ∧ y > 0)
  (h_eq : X + Y * d^(1/2) = (x + y * d^(1/2)) * (x₀ - y₀ * d^(1/2))^r) :
  X - Y * d^(1/2) = (x - y * d^(1/2)) * (x₀ + y₀ * d^(1/2))^r := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_conjugate_theorem_l3832_383288


namespace NUMINAMATH_CALUDE_power_two_plus_one_div_three_l3832_383295

theorem power_two_plus_one_div_three (n : ℕ+) :
  3 ∣ (2^n.val + 1) ↔ n.val % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_power_two_plus_one_div_three_l3832_383295


namespace NUMINAMATH_CALUDE_sin_240_degrees_l3832_383287

theorem sin_240_degrees : 
  Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l3832_383287


namespace NUMINAMATH_CALUDE_f_increasing_m_range_l3832_383280

/-- A function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x * |x - m| + 2 * x - 3

/-- Theorem stating that if f is increasing on ℝ, then m is in the interval [-2, 2] -/
theorem f_increasing_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_m_range_l3832_383280


namespace NUMINAMATH_CALUDE_gervais_driving_days_l3832_383216

theorem gervais_driving_days :
  let gervais_avg_miles_per_day : ℝ := 315
  let henri_total_miles : ℝ := 1250
  let difference_in_miles : ℝ := 305
  let gervais_days : ℝ := (henri_total_miles - difference_in_miles) / gervais_avg_miles_per_day
  gervais_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_gervais_driving_days_l3832_383216


namespace NUMINAMATH_CALUDE_unique_zero_location_l3832_383203

def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem unique_zero_location (f : ℝ → ℝ) :
  has_unique_zero_in f 0 16 ∧
  has_unique_zero_in f 0 8 ∧
  has_unique_zero_in f 0 6 ∧
  has_unique_zero_in f 2 4 →
  ¬ ∃ x, 0 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_location_l3832_383203


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l3832_383281

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℕ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l3832_383281


namespace NUMINAMATH_CALUDE_inequality_one_inequality_two_l3832_383249

-- Statement 1
theorem inequality_one (a : ℝ) (h : a > 3) : a + 4 / (a - 3) ≥ 7 := by
  sorry

-- Statement 2
theorem inequality_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_inequality_two_l3832_383249


namespace NUMINAMATH_CALUDE_central_angle_alice_bob_l3832_383260

/-- Represents a point on the Earth's surface with latitude and longitude -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on a spherical Earth -/
noncomputable def centralAngle (a b : EarthPoint) : Real :=
  sorry

/-- The location of Alice near Quito, Ecuador -/
def alice : EarthPoint :=
  { latitude := 0, longitude := -78 }

/-- The location of Bob near Vladivostok, Russia -/
def bob : EarthPoint :=
  { latitude := 43, longitude := 132 }

/-- Theorem stating that the central angle between Alice and Bob is 150 degrees -/
theorem central_angle_alice_bob :
  centralAngle alice bob = 150 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_alice_bob_l3832_383260


namespace NUMINAMATH_CALUDE_count_numbers_with_three_l3832_383294

/-- Count of numbers from 1 to 800 without digit 3 -/
def count_without_three : ℕ := 729

/-- Count of numbers from 1 to 800 with at least one digit 3 -/
def count_with_three : ℕ := 800 - count_without_three

theorem count_numbers_with_three : count_with_three = 71 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_three_l3832_383294


namespace NUMINAMATH_CALUDE_tan_difference_angle_sum_l3832_383200

-- Problem 1
theorem tan_difference (A B : Real) (h : 2 * Real.tan A = 3 * Real.tan B) :
  Real.tan (A - B) = Real.sin (2 * B) / (5 - Real.cos (2 * B)) := by sorry

-- Problem 2
theorem angle_sum (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan α = 1/7) 
  (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + 2*β = π/4 := by sorry

end NUMINAMATH_CALUDE_tan_difference_angle_sum_l3832_383200


namespace NUMINAMATH_CALUDE_two_power_ten_minus_one_factors_l3832_383270

theorem two_power_ten_minus_one_factors : 
  ∃ (p q r : Nat), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (2^10 - 1 : Nat) = p * q * r ∧
    p + q + r = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_two_power_ten_minus_one_factors_l3832_383270


namespace NUMINAMATH_CALUDE_power_equation_solution_l3832_383202

theorem power_equation_solution (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3832_383202


namespace NUMINAMATH_CALUDE_sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l3832_383239

def S (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

def forbidden_digits : Finset ℕ := {2, 4, 7, 9}

theorem sum_of_forbidden_digits : (forbidden_digits.sum id) = 22 := by sorry

theorem units_digit_not_in_forbidden (n : ℕ+) :
  (S n) % 10 ∉ forbidden_digits := by sorry

theorem sum_forbidden_digits_correct :
  ∃ (digits : Finset ℕ), 
    (∀ (n : ℕ+), (S n) % 10 ∉ digits) ∧
    (digits.sum id = 22) ∧
    (∀ (d : ℕ), d ∉ digits → ∃ (n : ℕ+), (S n) % 10 = d) := by sorry

end NUMINAMATH_CALUDE_sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l3832_383239


namespace NUMINAMATH_CALUDE_trig_identity_l3832_383219

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3832_383219


namespace NUMINAMATH_CALUDE_smallest_quotient_l3832_383285

def digit_sum_of_squares (n : ℕ) : ℕ :=
  if n < 10 then n * n else (n % 10) * (n % 10) + digit_sum_of_squares (n / 10)

theorem smallest_quotient (n : ℕ) (h : n > 0) :
  (n : ℚ) / (digit_sum_of_squares n) ≥ 1 / 9 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℚ) / (digit_sum_of_squares m) = 1 / 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_quotient_l3832_383285


namespace NUMINAMATH_CALUDE_scientific_notation_28400_l3832_383261

theorem scientific_notation_28400 : 28400 = 2.84 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_28400_l3832_383261


namespace NUMINAMATH_CALUDE_odd_function_zeros_and_equation_root_l3832_383248

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zeros_and_equation_root (f : ℝ → ℝ) (zeros : Finset ℝ) :
  isOdd f →
  zeros.card = 2017 →
  (∀ x ∈ zeros, f x = 0) →
  ∃ r ∈ Set.Ioo 0 1, 2^r + r - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_function_zeros_and_equation_root_l3832_383248


namespace NUMINAMATH_CALUDE_f_zero_equals_three_l3832_383276

-- Define the function f
noncomputable def f (t : ℝ) : ℝ :=
  let x := (t + 1) / 2
  (1 - x^2) / x^2

-- Theorem statement
theorem f_zero_equals_three :
  f 0 = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_zero_equals_three_l3832_383276


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l3832_383201

/-- Given two functions f and g with identical symmetry axes, 
    prove that [0, π/8] is an increasing interval of f on [0, π] -/
theorem increasing_interval_of_f (ω : ℝ) (h_ω : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * x - π / 4)
  (∀ x y : ℝ, f x = f y ↔ g x = g y) →  -- Symmetry axes are identical
  (∀ x y : ℝ, x ∈ Set.Icc 0 (π / 8) → y ∈ Set.Icc 0 (π / 8) → x < y → f x < f y) ∧
  Set.Icc 0 (π / 8) ⊆ Set.Icc 0 π :=
by sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l3832_383201


namespace NUMINAMATH_CALUDE_min_value_of_max_function_l3832_383259

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > 2*y) :
  ∃ (t : ℝ), t = max (x^2/2) (4/(y*(x-2*y))) ∧ t ≥ 4 ∧ 
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 > 2*y0 ∧ 
    max (x0^2/2) (4/(y0*(x0-2*y0))) = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_max_function_l3832_383259


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l3832_383273

theorem sum_of_fractions_zero (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1) :
  a / (b - c) + b / (c - a) + c / (a - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l3832_383273


namespace NUMINAMATH_CALUDE_first_day_cost_l3832_383206

/-- The cost of a hamburger -/
def hamburger_cost : ℚ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℚ := 1

/-- The cost of 2 hamburgers and 3 hot dogs -/
def second_day_cost : ℚ := 7

theorem first_day_cost : 3 * hamburger_cost + 4 * hot_dog_cost = 10 :=
  by sorry

end NUMINAMATH_CALUDE_first_day_cost_l3832_383206


namespace NUMINAMATH_CALUDE_no_infinite_set_with_divisibility_property_l3832_383227

theorem no_infinite_set_with_divisibility_property :
  ¬ ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ (a b : ℤ), a ∈ S → b ∈ S → (a^2 + b^2 - a*b) ∣ (a*b)^2 :=
sorry

end NUMINAMATH_CALUDE_no_infinite_set_with_divisibility_property_l3832_383227


namespace NUMINAMATH_CALUDE_salary_increase_proof_l3832_383290

def new_salary : ℝ := 90000
def percent_increase : ℝ := 38.46153846153846

theorem salary_increase_proof :
  let old_salary := new_salary / (1 + percent_increase / 100)
  let increase := new_salary - old_salary
  increase = 25000 := by sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l3832_383290


namespace NUMINAMATH_CALUDE_hiking_equipment_cost_l3832_383250

/-- Calculates the total cost of hiking equipment --/
theorem hiking_equipment_cost (hoodie_cost : ℚ) (boot_cost : ℚ) (flashlight_percentage : ℚ) (discount_percentage : ℚ) : 
  hoodie_cost = 80 →
  flashlight_percentage = 20 / 100 →
  boot_cost = 110 →
  discount_percentage = 10 / 100 →
  hoodie_cost + (flashlight_percentage * hoodie_cost) + (boot_cost - discount_percentage * boot_cost) = 195 := by
  sorry

end NUMINAMATH_CALUDE_hiking_equipment_cost_l3832_383250


namespace NUMINAMATH_CALUDE_circle_equation_l3832_383226

/-- The standard equation of a circle with center (0, -2) and radius 4 is x^2 + (y+2)^2 = 16 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (0, -2)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + (y + 2)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3832_383226


namespace NUMINAMATH_CALUDE_smallest_upper_bound_l3832_383272

def S : Set (ℕ+ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n : ℕ+, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1 : ℝ) * f (2 * n)}

theorem smallest_upper_bound :
  ∃ M : ℕ+, (∀ f ∈ S, ∀ n : ℕ+, f n < M) ∧
  (∀ M' : ℕ+, M' < M → ∃ f ∈ S, ∃ n : ℕ+, f n ≥ M') :=
by sorry

end NUMINAMATH_CALUDE_smallest_upper_bound_l3832_383272


namespace NUMINAMATH_CALUDE_log_seven_forty_eight_l3832_383211

theorem log_seven_forty_eight (a b : ℝ) (h1 : Real.log 3 / Real.log 7 = a) (h2 : Real.log 4 / Real.log 7 = b) :
  Real.log 48 / Real.log 7 = a + 2 * b := by
  sorry

end NUMINAMATH_CALUDE_log_seven_forty_eight_l3832_383211


namespace NUMINAMATH_CALUDE_distance_between_points_l3832_383277

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 3)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3832_383277


namespace NUMINAMATH_CALUDE_sum_of_products_l3832_383253

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 57) :
  x*y + y*z + x*z = 24 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3832_383253


namespace NUMINAMATH_CALUDE_scale_division_l3832_383222

/-- Converts feet and inches to total inches -/
def feetInchesToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet and inches -/
def inchesToFeetInches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division (totalFeet : ℕ) (totalInches : ℕ) (parts : ℕ) 
    (h1 : totalFeet = 7) 
    (h2 : totalInches = 6) 
    (h3 : parts = 5) : 
  inchesToFeetInches (feetInchesToInches totalFeet totalInches / parts) = (1, 6) := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l3832_383222


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3832_383242

theorem inequality_system_solution_set :
  let S := {x : ℝ | 2 * x - 1 ≥ x + 1 ∧ x + 8 ≤ 4 * x - 1}
  S = {x : ℝ | x ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3832_383242


namespace NUMINAMATH_CALUDE_dividend_calculation_l3832_383236

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3832_383236


namespace NUMINAMATH_CALUDE_initial_amount_calculation_l3832_383286

/-- Represents the simple interest scenario --/
structure SimpleInterest where
  initialAmount : ℝ
  rate : ℝ
  time : ℝ
  finalAmount : ℝ

/-- The simple interest calculation is correct --/
def isValidSimpleInterest (si : SimpleInterest) : Prop :=
  si.finalAmount = si.initialAmount * (1 + si.rate * si.time / 100)

/-- Theorem stating the initial amount given the conditions --/
theorem initial_amount_calculation (si : SimpleInterest) 
  (h1 : si.finalAmount = 1050)
  (h2 : si.rate = 8)
  (h3 : si.time = 5)
  (h4 : isValidSimpleInterest si) : 
  si.initialAmount = 750 := by
  sorry

#check initial_amount_calculation

end NUMINAMATH_CALUDE_initial_amount_calculation_l3832_383286


namespace NUMINAMATH_CALUDE_largest_number_l3832_383221

theorem largest_number (a b c d e : ℝ) : 
  a = 13579 + 1 / 2468 →
  b = 13579 - 1 / 2468 →
  c = 13579 * (1 / 2468) →
  d = 13579 / (1 / 2468) →
  e = 13579.2468 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3832_383221


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3832_383245

theorem modulus_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l3832_383245


namespace NUMINAMATH_CALUDE_fourth_month_sales_l3832_383224

def sales_month1 : ℕ := 6535
def sales_month2 : ℕ := 6927
def sales_month3 : ℕ := 6855
def sales_month5 : ℕ := 6562
def sales_month6 : ℕ := 4891
def required_average : ℕ := 6500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = required_average ∧
    sales_month4 = 7230 := by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l3832_383224


namespace NUMINAMATH_CALUDE_train_length_l3832_383232

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 24 →
  train_speed = 50 →
  train_speed * crossing_time - bridge_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3832_383232


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3832_383230

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n * 2^(n + 1) + 1 = k^2) ↔ n = 0 ∨ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3832_383230


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l3832_383241

/-- Calculates the number of items drawn from a category in stratified sampling -/
def items_drawn (category_size : ℕ) (total_size : ℕ) (sample_size : ℕ) : ℕ :=
  (category_size * sample_size) / total_size

/-- Represents the stratified sampling problem -/
theorem stratified_sampling_sum (grains : ℕ) (vegetable_oil : ℕ) (animal_products : ℕ) (fruits_vegetables : ℕ) 
  (sample_size : ℕ) (h1 : grains = 40) (h2 : vegetable_oil = 10) (h3 : animal_products = 30) 
  (h4 : fruits_vegetables = 20) (h5 : sample_size = 20) :
  items_drawn vegetable_oil (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size + 
  items_drawn fruits_vegetables (grains + vegetable_oil + animal_products + fruits_vegetables) sample_size = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_sum_l3832_383241


namespace NUMINAMATH_CALUDE_shopping_money_l3832_383214

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 3500 → initial_amount = 5000 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l3832_383214


namespace NUMINAMATH_CALUDE_monkey_reaches_top_monkey_reaches_top_in_19_minutes_l3832_383278

def pole_height : ℕ := 10
def ascend_distance : ℕ := 2
def slip_distance : ℕ := 1

def monkey_position (minutes : ℕ) : ℕ :=
  let full_cycles := minutes / 2
  let remainder := minutes % 2
  if remainder = 0 then
    full_cycles * (ascend_distance - slip_distance)
  else
    full_cycles * (ascend_distance - slip_distance) + ascend_distance

theorem monkey_reaches_top :
  ∃ (minutes : ℕ), monkey_position minutes ≥ pole_height ∧
                   ∀ (m : ℕ), m < minutes → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

theorem monkey_reaches_top_in_19_minutes :
  monkey_position 19 ≥ pole_height ∧
  ∀ (m : ℕ), m < 19 → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_monkey_reaches_top_monkey_reaches_top_in_19_minutes_l3832_383278


namespace NUMINAMATH_CALUDE_enhanced_mindmaster_codes_l3832_383258

/-- The number of colors available in the enhanced Mindmaster game -/
def num_colors : ℕ := 7

/-- The number of slots in a secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the enhanced Mindmaster game -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the number of possible secret codes is 16807 -/
theorem enhanced_mindmaster_codes :
  num_codes = 16807 := by sorry

end NUMINAMATH_CALUDE_enhanced_mindmaster_codes_l3832_383258


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l3832_383289

theorem complex_expression_simplification (p q : ℝ) (hp : p > 0) (hpq : p > q) :
  let numerator := Real.sqrt ((p^4 + q^4) / (p^4 - p^2 * q^2) + 2 * q^2 / (p^2 - q^2) * (p^3 - p * q^2)) - 2 * q * Real.sqrt p
  let denominator := Real.sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)
  numerator / denominator = Real.sqrt (p^2 - q^2) / Real.sqrt p :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l3832_383289


namespace NUMINAMATH_CALUDE_book_ratio_is_three_to_one_l3832_383235

-- Define the number of books for each person
def elmo_books : ℕ := 24
def stu_books : ℕ := 4
def laura_books : ℕ := 2 * stu_books

-- Define the ratio of Elmo's books to Laura's books
def book_ratio : ℚ := elmo_books / laura_books

-- Theorem to prove
theorem book_ratio_is_three_to_one : book_ratio = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_book_ratio_is_three_to_one_l3832_383235


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3832_383257

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem matrix_equation_solution :
  ∃ (a b c : ℚ), 
    B^3 + a • B^2 + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
    a = 0 ∧ b = -283/13 ∧ c = 902/13 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3832_383257


namespace NUMINAMATH_CALUDE_voronovich_inequality_l3832_383243

theorem voronovich_inequality {a b c : ℝ} (ha : 0 < a) (hab : a < b) (hbc : b < c) :
  a^20 * b^12 + b^20 * c^12 + c^20 * a^12 < b^20 * a^12 + a^20 * c^12 + c^20 * b^12 :=
by sorry

end NUMINAMATH_CALUDE_voronovich_inequality_l3832_383243


namespace NUMINAMATH_CALUDE_rectangular_box_diagonals_l3832_383213

/-- A rectangular box with given surface area and edge length sum has a specific sum of interior diagonal lengths. -/
theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 206)
  (h_edge_sum : 4 * (a + b + c) = 64) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonals_l3832_383213


namespace NUMINAMATH_CALUDE_parabola_shift_left_l3832_383244

/-- The analytical expression of a parabola shifted to the left -/
theorem parabola_shift_left (x y : ℝ) :
  (∀ x, y = x^2) →  -- Original parabola
  (∀ x, y = (x + 1)^2) -- Parabola shifted 1 unit left
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_left_l3832_383244


namespace NUMINAMATH_CALUDE_buttons_per_shirt_proof_l3832_383233

/-- The number of shirts Sally sews on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sews on Tuesday -/
def tuesday_shirts : ℕ := 3

/-- The number of shirts Sally sews on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The total number of buttons Sally needs for all shirts -/
def total_buttons : ℕ := 45

/-- The number of buttons per shirt -/
def buttons_per_shirt : ℕ := 5

theorem buttons_per_shirt_proof :
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt = total_buttons :=
by sorry

end NUMINAMATH_CALUDE_buttons_per_shirt_proof_l3832_383233


namespace NUMINAMATH_CALUDE_exchange_divisibility_l3832_383210

theorem exchange_divisibility (p a d : ℤ) : 
  p = 4*a + d ∧ p = a + 5*d → 
  ∃ (t : ℤ), p = 19*t ∧ a = 4*t ∧ d = 3*t ∧ p + a + d = 26*t :=
by sorry

end NUMINAMATH_CALUDE_exchange_divisibility_l3832_383210


namespace NUMINAMATH_CALUDE_triangle_theorem_l3832_383251

/-- Triangle ABC with sides a, b, c opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b + t.c) * (t.a - t.b + t.c) = t.a * t.c)
  (h2 : Real.sin t.A * Real.sin t.C = (Real.sqrt 3 - 1) / 4) :
  t.B = 2 * Real.pi / 3 ∧ (t.C = Real.pi / 12 ∨ t.C = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3832_383251


namespace NUMINAMATH_CALUDE_traveler_distance_l3832_383284

/-- The straight-line distance from start to end point given net northward and westward distances -/
theorem traveler_distance (north west : ℝ) (h_north : north = 12) (h_west : west = 12) :
  Real.sqrt (north ^ 2 + west ^ 2) = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_traveler_distance_l3832_383284


namespace NUMINAMATH_CALUDE_two_number_problem_l3832_383264

theorem two_number_problem (x y : ℝ) 
  (h1 : 0.35 * x = 0.50 * x - 24)
  (h2 : 0.30 * y = 0.55 * x - 36) : 
  x = 160 ∧ y = 520/3 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l3832_383264


namespace NUMINAMATH_CALUDE_matrix_det_times_two_l3832_383267

def matrix_det (a b c d : ℤ) : ℤ := a * d - b * c

theorem matrix_det_times_two :
  2 * (matrix_det 5 7 2 3) = 2 := by sorry

end NUMINAMATH_CALUDE_matrix_det_times_two_l3832_383267


namespace NUMINAMATH_CALUDE_inequality_proof_l3832_383297

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 1/b^3 - 1) * (b^3 + 1/c^3 - 1) * (c^3 + 1/a^3 - 1) ≤ (a*b*c + 1/(a*b*c) - 1)^3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3832_383297


namespace NUMINAMATH_CALUDE_f_minimum_l3832_383266

/-- The polynomial f(x) = x^2 + 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 10

/-- The point where f(x) reaches its minimum -/
def min_point : ℝ := -3

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f min_point := by sorry

end NUMINAMATH_CALUDE_f_minimum_l3832_383266
