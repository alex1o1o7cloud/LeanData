import Mathlib

namespace NUMINAMATH_CALUDE_two_false_propositions_l1259_125946

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > -3 → a > -6

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > -6 → a > -3

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ -6

-- Define the contrapositive proposition
def contrapositive_prop (a : ℝ) : Prop := a ≤ -6 → a ≤ -3

-- Theorem statement
theorem two_false_propositions :
  ∃ (f : Fin 4 → Prop), 
    (∀ a : ℝ, f 0 = original_prop a ∧ 
              f 1 = converse_prop a ∧ 
              f 2 = inverse_prop a ∧ 
              f 3 = contrapositive_prop a) ∧
    (∃! (i j : Fin 4), i ≠ j ∧ ¬(f i) ∧ ¬(f j) ∧ 
      ∀ (k : Fin 4), k ≠ i ∧ k ≠ j → f k) :=
by
  sorry

end NUMINAMATH_CALUDE_two_false_propositions_l1259_125946


namespace NUMINAMATH_CALUDE_sum_equals_5000_minus_N_l1259_125957

theorem sum_equals_5000_minus_N (N : ℕ) : 
  988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_5000_minus_N_l1259_125957


namespace NUMINAMATH_CALUDE_inscribed_triangle_radius_l1259_125953

/-- Given a regular triangle inscribed in a circular segment with the following properties:
    - The arc of the segment has a central angle α
    - One vertex of the triangle coincides with the midpoint of the arc
    - The other two vertices lie on the chord
    - The area of the triangle is S
    Then the radius R of the circle is given by R = (√(S√3)) / (2 sin²(α/4)) -/
theorem inscribed_triangle_radius (S α : ℝ) (h_S : S > 0) (h_α : 0 < α ∧ α < 2 * Real.pi) :
  ∃ R : ℝ, R > 0 ∧ R = (Real.sqrt (S * Real.sqrt 3)) / (2 * (Real.sin (α / 4))^2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_radius_l1259_125953


namespace NUMINAMATH_CALUDE_cherry_pits_sprouted_percentage_l1259_125934

theorem cherry_pits_sprouted_percentage (total_pits : ℕ) (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  saplings_sold = 6 →
  saplings_left = 14 →
  (((saplings_sold + saplings_left : ℚ) / total_pits) * 100 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pits_sprouted_percentage_l1259_125934


namespace NUMINAMATH_CALUDE_intersection_points_slope_l1259_125947

theorem intersection_points_slope :
  ∀ (s x y : ℝ), 
    (2 * x + 3 * y = 8 * s + 5) →
    (x + 2 * y = 3 * s + 2) →
    y = -(7/2) * x + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_slope_l1259_125947


namespace NUMINAMATH_CALUDE_ball_probability_and_replacement_l1259_125940

/-- Given a bag with red, yellow, and blue balls, this theorem proves:
    1. The initial probability of drawing a red ball.
    2. The number of red balls replaced to achieve a specific probability of drawing a yellow ball. -/
theorem ball_probability_and_replacement 
  (initial_red : ℕ) 
  (initial_yellow : ℕ) 
  (initial_blue : ℕ) 
  (replaced : ℕ) :
  initial_red = 10 → 
  initial_yellow = 2 → 
  initial_blue = 8 → 
  (initial_red : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 1/2 ∧
  (initial_yellow + replaced : ℚ) / (initial_red + initial_yellow + initial_blue : ℚ) = 2/5 →
  replaced = 6 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_and_replacement_l1259_125940


namespace NUMINAMATH_CALUDE_saras_house_difference_l1259_125976

theorem saras_house_difference (sara_house : ℕ) (nada_house : ℕ) : 
  sara_house = 1000 → nada_house = 450 → sara_house - 2 * nada_house = 100 := by
  sorry

end NUMINAMATH_CALUDE_saras_house_difference_l1259_125976


namespace NUMINAMATH_CALUDE_jet_ski_time_to_dock_b_l1259_125914

/-- Represents the scenario of a jet ski and a canoe traveling on a river --/
structure RiverTravel where
  distance : ℝ  -- Distance between dock A and dock B
  speed_difference : ℝ  -- Speed difference between jet ski and current
  total_time : ℝ  -- Total time until jet ski meets canoe

/-- 
Calculates the time taken by the jet ski to reach dock B.
Returns the time in hours.
-/
def time_to_dock_b (rt : RiverTravel) : ℝ :=
  sorry

/-- Theorem stating that the time taken by the jet ski to reach dock B is 3 hours --/
theorem jet_ski_time_to_dock_b (rt : RiverTravel) 
  (h1 : rt.distance = 60) 
  (h2 : rt.speed_difference = 10) 
  (h3 : rt.total_time = 8) : 
  time_to_dock_b rt = 3 :=
  sorry

end NUMINAMATH_CALUDE_jet_ski_time_to_dock_b_l1259_125914


namespace NUMINAMATH_CALUDE_nine_digit_sum_l1259_125912

-- Define the type for digits 1-9
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

-- Define the structure for the nine-digit number
structure NineDigitNumber where
  digits : Fin 9 → Digit
  distinct : ∀ i j, i ≠ j → digits i ≠ digits j

-- Define the property that each two-digit segment is a product of two single-digit numbers
def validSegments (n : NineDigitNumber) : Prop :=
  ∀ i : Fin 8, ∃ (x y : Digit), 
    (n.digits i).val * 10 + (n.digits (i + 1)).val = x.val * y.val

-- Define the function to calculate the sum of ABC + DEF + GHI
def sumSegments (n : NineDigitNumber) : ℕ :=
  ((n.digits 0).val * 100 + (n.digits 1).val * 10 + (n.digits 2).val) +
  ((n.digits 3).val * 100 + (n.digits 4).val * 10 + (n.digits 5).val) +
  ((n.digits 6).val * 100 + (n.digits 7).val * 10 + (n.digits 8).val)

-- State the theorem
theorem nine_digit_sum (n : NineDigitNumber) (h : validSegments n) : 
  sumSegments n = 1440 :=
sorry

end NUMINAMATH_CALUDE_nine_digit_sum_l1259_125912


namespace NUMINAMATH_CALUDE_prob_one_good_product_prob_one_good_product_proof_l1259_125930

/-- The probability of selecting exactly one good product when randomly selecting
    two products from a set of five products, where three are good and two are defective. -/
theorem prob_one_good_product : ℚ :=
  let total_products : ℕ := 5
  let good_products : ℕ := 3
  let defective_products : ℕ := 2
  let selected_products : ℕ := 2
  3 / 5

/-- Proof that the probability of selecting exactly one good product is 3/5. -/
theorem prob_one_good_product_proof :
  prob_one_good_product = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_good_product_prob_one_good_product_proof_l1259_125930


namespace NUMINAMATH_CALUDE_triangle_division_exists_l1259_125950

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry

/-- A division of a triangle into four convex shapes -/
structure TriangleDivision where
  original : ConvexPolygon 3
  triangle : ConvexPolygon 3
  quadrilateral : ConvexPolygon 4
  pentagon : ConvexPolygon 5
  hexagon : ConvexPolygon 6
  valid_division : sorry

/-- Any triangle can be divided into a triangle, quadrilateral, pentagon, and hexagon -/
theorem triangle_division_exists : ∀ (t : ConvexPolygon 3), ∃ (d : TriangleDivision), d.original = t :=
sorry

end NUMINAMATH_CALUDE_triangle_division_exists_l1259_125950


namespace NUMINAMATH_CALUDE_tree_planting_event_l1259_125905

theorem tree_planting_event (boys girls : ℕ) : 
  girls = boys + 400 →
  girls > boys →
  (60 : ℚ) / 100 * (boys + girls : ℚ) = 960 →
  boys = 600 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_event_l1259_125905


namespace NUMINAMATH_CALUDE_chandler_bike_savings_l1259_125918

/-- The number of weeks Chandler needs to save to buy a mountain bike -/
def weeks_to_save (bike_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  (bike_cost - birthday_money) / weekly_earnings

theorem chandler_bike_savings : 
  let bike_cost : ℕ := 600
  let grandparents_gift : ℕ := 60
  let aunt_gift : ℕ := 40
  let cousin_gift : ℕ := 20
  let weekly_earnings : ℕ := 20
  let total_birthday_money : ℕ := grandparents_gift + aunt_gift + cousin_gift
  weeks_to_save bike_cost total_birthday_money weekly_earnings = 24 := by
  sorry

#eval weeks_to_save 600 (60 + 40 + 20) 20

end NUMINAMATH_CALUDE_chandler_bike_savings_l1259_125918


namespace NUMINAMATH_CALUDE_equation_solution_property_l1259_125996

theorem equation_solution_property (m n : ℝ) : 
  (∃ x : ℝ, m * x + n - 2 = 0 ∧ x = 2) → 2 * m + n + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_property_l1259_125996


namespace NUMINAMATH_CALUDE_awake_cats_l1259_125942

theorem awake_cats (total : ℕ) (asleep : ℕ) (awake : ℕ) : 
  total = 98 → asleep = 92 → awake = total - asleep → awake = 6 := by
  sorry

end NUMINAMATH_CALUDE_awake_cats_l1259_125942


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1259_125901

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the condition for any point P on the ellipse
def point_condition (PF1 PF2 : ℝ) : Prop :=
  PF1 + PF2 = 2 * Real.sqrt 2

-- Define the focal distance
def focal_distance : ℝ := 2

-- Define the intersecting line
def intersecting_line (x y t : ℝ) : Prop :=
  x - y + t = 0

-- Define the circle condition for the midpoint of AB
def midpoint_condition (x y : ℝ) : Prop :=
  x^2 + y^2 > 10/9

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  (∀ (x y : ℝ), ellipse_C x y a b → ∃ (PF1 PF2 : ℝ), point_condition PF1 PF2) →
  (a^2 - b^2 = focal_distance^2) →
  (∀ (x y : ℝ), ellipse_C x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ (t : ℝ),
    (∃ (x1 y1 x2 y2 : ℝ),
      ellipse_C x1 y1 a b ∧
      ellipse_C x2 y2 a b ∧
      intersecting_line x1 y1 t ∧
      intersecting_line x2 y2 t ∧
      x1 ≠ x2 ∧
      midpoint_condition ((x1 + x2) / 2) ((y1 + y2) / 2)) →
    (-Real.sqrt 3 < t ∧ t ≤ -Real.sqrt 2) ∨ (Real.sqrt 2 ≤ t ∧ t < Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1259_125901


namespace NUMINAMATH_CALUDE_right_triangle_sin_d_l1259_125928

theorem right_triangle_sin_d (D E F : ℝ) (h1 : 0 < D) (h2 : D < π / 2) : 
  5 * Real.sin D = 12 * Real.cos D → Real.sin D = 12 / 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sin_d_l1259_125928


namespace NUMINAMATH_CALUDE_coffee_shop_spending_coffee_shop_spending_proof_l1259_125988

theorem coffee_shop_spending : ℝ → ℝ → Prop :=
  fun b d =>
    (d = 0.6 * b) →  -- David spent 40 cents less for each dollar Ben spent
    (b = d + 14) →   -- Ben paid $14 more than David
    (b + d = 56)     -- Their total spending

-- The proof is omitted
theorem coffee_shop_spending_proof : ∃ b d : ℝ, coffee_shop_spending b d := by sorry

end NUMINAMATH_CALUDE_coffee_shop_spending_coffee_shop_spending_proof_l1259_125988


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1259_125945

/-- 
A point M with coordinates (m-1, 2m) lies on the x-axis if and only if 
its coordinates are (-1, 0).
-/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1, 2 * m) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ (m - 1, 2 * m) = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1259_125945


namespace NUMINAMATH_CALUDE_gcd_143_100_l1259_125979

theorem gcd_143_100 : Nat.gcd 143 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_100_l1259_125979


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1259_125975

-- Define the constants for the foci locations
def ellipse_focus : ℝ := 5
def hyperbola_focus : ℝ := 8

-- Define the theorem
theorem ellipse_hyperbola_product (c d : ℝ) : 
  (d^2 - c^2 = ellipse_focus^2) →   -- Condition for ellipse foci
  (c^2 + d^2 = hyperbola_focus^2) → -- Condition for hyperbola foci
  |c * d| = Real.sqrt ((39 * 89) / 4) := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1259_125975


namespace NUMINAMATH_CALUDE_gcd_power_three_l1259_125978

theorem gcd_power_three : Nat.gcd (3^600 - 1) (3^612 - 1) = 3^12 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_three_l1259_125978


namespace NUMINAMATH_CALUDE_profit_is_27000_l1259_125949

/-- Represents the profit sharing problem between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  let tom_total := ps.tom_investment * ps.tom_months
  let jose_total := ps.jose_investment * ps.jose_months
  let ratio_sum := (tom_total / (tom_total.gcd jose_total)) + (jose_total / (tom_total.gcd jose_total))
  (ratio_sum * ps.jose_profit) / (jose_total / (tom_total.gcd jose_total))

/-- Theorem stating that the total profit is 27000 for the given conditions -/
theorem profit_is_27000 (ps : ProfitSharing)
  (h1 : ps.tom_investment = 30000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 45000)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 15000) :
  total_profit ps = 27000 := by
  sorry

#eval total_profit { tom_investment := 30000, tom_months := 12, jose_investment := 45000, jose_months := 10, jose_profit := 15000 }

end NUMINAMATH_CALUDE_profit_is_27000_l1259_125949


namespace NUMINAMATH_CALUDE_angle_bisector_vector_l1259_125987

/-- Given points A and B in a Cartesian coordinate system, 
    and a point C on the angle bisector of ∠AOB with |OC| = 2, 
    prove that OC has specific coordinates. -/
theorem angle_bisector_vector (A B C : ℝ × ℝ) : 
  A = (0, 1) →
  B = (-3, 4) →
  (C.1 * A.2 = C.2 * A.1 ∧ C.1 * B.2 = C.2 * B.1) → -- C is on angle bisector
  C.1^2 + C.2^2 = 4 → -- |OC| = 2
  C = (-Real.sqrt 10 / 5, 3 * Real.sqrt 10 / 5) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_vector_l1259_125987


namespace NUMINAMATH_CALUDE_tiger_distance_is_160_l1259_125904

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The total distance traveled by the escaped tiger -/
def tiger_distance : ℝ :=
  distance 25 1 + distance 35 2 + distance 20 1.5 + distance 10 1 + distance 50 0.5

/-- Theorem stating that the tiger traveled 160 miles -/
theorem tiger_distance_is_160 : tiger_distance = 160 := by
  sorry

end NUMINAMATH_CALUDE_tiger_distance_is_160_l1259_125904


namespace NUMINAMATH_CALUDE_coworker_repair_ratio_l1259_125974

/-- The ratio of phones a coworker fixes to the total number of damaged phones -/
theorem coworker_repair_ratio : 
  ∀ (initial_phones repaired_phones new_phones phones_per_person : ℕ),
    initial_phones = 15 →
    repaired_phones = 3 →
    new_phones = 6 →
    phones_per_person = 9 →
    (phones_per_person : ℚ) / ((initial_phones - repaired_phones + new_phones) : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_coworker_repair_ratio_l1259_125974


namespace NUMINAMATH_CALUDE_subset_union_theorem_l1259_125911

theorem subset_union_theorem (n : ℕ) (X : Finset ℕ) (m : ℕ) 
  (A : Fin m → Finset ℕ) :
  n > 6 →
  X.card = n →
  (∀ i : Fin m, (A i).card = 5) →
  (∀ i j : Fin m, i ≠ j → A i ≠ A j) →
  m > n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15) / 600 →
  ∃ (i₁ i₂ i₃ i₄ i₅ i₆ : Fin m), 
    i₁ < i₂ ∧ i₂ < i₃ ∧ i₃ < i₄ ∧ i₄ < i₅ ∧ i₅ < i₆ ∧
    (A i₁ ∪ A i₂ ∪ A i₃ ∪ A i₄ ∪ A i₅ ∪ A i₆) = X :=
by sorry

end NUMINAMATH_CALUDE_subset_union_theorem_l1259_125911


namespace NUMINAMATH_CALUDE_percent_of_percent_l1259_125961

theorem percent_of_percent (y : ℝ) : (21 / 100) * y = (30 / 100) * ((70 / 100) * y) := by
  sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1259_125961


namespace NUMINAMATH_CALUDE_waiter_tip_earnings_l1259_125900

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  (total_customers - non_tipping_customers) * tip_amount = 15 := by
sorry

end NUMINAMATH_CALUDE_waiter_tip_earnings_l1259_125900


namespace NUMINAMATH_CALUDE_complex_power_sum_l1259_125995

theorem complex_power_sum (z : ℂ) (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1259_125995


namespace NUMINAMATH_CALUDE_intersection_A_B_l1259_125903

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 6}

-- Define set B
def B : Set ℝ := {x | 3 * x^2 + x - 8 ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 4/3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1259_125903


namespace NUMINAMATH_CALUDE_pokemon_card_collection_l1259_125926

def cards_needed (michael_cards : ℕ) (mark_diff : ℕ) (lloyd_ratio : ℕ) (total_goal : ℕ) : ℕ :=
  let mark_cards := michael_cards - mark_diff
  let lloyd_cards := mark_cards / lloyd_ratio
  total_goal - (michael_cards + mark_cards + lloyd_cards)

theorem pokemon_card_collection : 
  cards_needed 100 10 3 300 = 80 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_card_collection_l1259_125926


namespace NUMINAMATH_CALUDE_fifth_term_zero_l1259_125954

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_zero
  (a : ℕ → ℚ)
  (x y : ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = x^2 - 4*y^2)
  (h_fourth : a 3 = x / (2*y))
  (h_x : x = 1/2)
  (h_y : y = 1/4)
  : a 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_zero_l1259_125954


namespace NUMINAMATH_CALUDE_dans_remaining_money_l1259_125992

/-- Calculates the remaining money after purchases. -/
def remaining_money (initial : ℕ) (candy_price : ℕ) (chocolate_price : ℕ) : ℕ :=
  initial - (candy_price + chocolate_price)

/-- Proves that Dan has $2 left after his purchases. -/
theorem dans_remaining_money :
  remaining_money 7 2 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l1259_125992


namespace NUMINAMATH_CALUDE_certain_amount_problem_l1259_125971

theorem certain_amount_problem (first_number : ℕ) (certain_amount : ℕ) : 
  first_number = 5 →
  first_number + (11 + certain_amount) = 19 →
  certain_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_problem_l1259_125971


namespace NUMINAMATH_CALUDE_max_discarded_apples_l1259_125951

theorem max_discarded_apples (n : ℕ) : ∃ (q : ℕ), n = 7 * q + 6 ∧ 
  ∀ (r : ℕ), r < 6 → ∃ (q' : ℕ), n ≠ 7 * q' + r :=
sorry

end NUMINAMATH_CALUDE_max_discarded_apples_l1259_125951


namespace NUMINAMATH_CALUDE_solution_replacement_l1259_125919

theorem solution_replacement (initial_conc : ℚ) (replacing_conc : ℚ) (final_conc : ℚ) 
  (h1 : initial_conc = 70/100)
  (h2 : replacing_conc = 25/100)
  (h3 : final_conc = 35/100) :
  ∃ (x : ℚ), x = 7/9 ∧ initial_conc * (1 - x) + replacing_conc * x = final_conc :=
by sorry

end NUMINAMATH_CALUDE_solution_replacement_l1259_125919


namespace NUMINAMATH_CALUDE_replacement_results_in_four_terms_l1259_125920

-- Define the expression as a function of x and the replacement term
def expression (x : ℝ) (replacement : ℝ → ℝ) : ℝ := 
  (x^3 - 2)^2 + (x^2 + replacement x)^2

-- Define the expansion of the expression
def expanded_expression (x : ℝ) : ℝ := 
  x^6 + x^4 + 4*x^2 + 4

-- Theorem statement
theorem replacement_results_in_four_terms :
  ∀ x : ℝ, expression x (λ y => 2*y) = expanded_expression x :=
by sorry

end NUMINAMATH_CALUDE_replacement_results_in_four_terms_l1259_125920


namespace NUMINAMATH_CALUDE_rectangle_from_right_triangle_l1259_125994

theorem rectangle_from_right_triangle (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0) :
  ∃ x y : ℝ, 
    x + y = c ∧ 
    x * y = a * b / 2 ∧
    x = (c + a - b) / 2 ∧ 
    y = (c - a + b) / 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_from_right_triangle_l1259_125994


namespace NUMINAMATH_CALUDE_circle_area_8m_diameter_circle_area_8m_diameter_proof_l1259_125972

/-- The area of a circle with diameter 8 meters, in square centimeters -/
theorem circle_area_8m_diameter (π : ℝ) : ℝ :=
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area_sq_meters : ℝ := π * radius ^ 2
  let sq_cm_per_sq_meter : ℝ := 10000
  160000 * π

/-- Proof that the area of a circle with diameter 8 meters is 160000π square centimeters -/
theorem circle_area_8m_diameter_proof (π : ℝ) :
  circle_area_8m_diameter π = 160000 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_8m_diameter_circle_area_8m_diameter_proof_l1259_125972


namespace NUMINAMATH_CALUDE_profit_calculation_l1259_125916

/-- Given that the cost price of 30 articles equals the selling price of x articles,
    and the profit is 25%, prove that x = 24. -/
theorem profit_calculation (x : ℝ) 
  (h1 : 30 * cost_price = x * selling_price)
  (h2 : selling_price = 1.25 * cost_price) : 
  x = 24 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l1259_125916


namespace NUMINAMATH_CALUDE_prime_average_count_l1259_125986

theorem prime_average_count : 
  ∃ (p₁ p₂ p₃ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
    p₁ > 20 ∧ p₂ > 20 ∧ p₃ > 20 ∧
    (p₁ + p₂ + p₃) / 3 = 83 / 3 ∧
    ∀ (q₁ q₂ q₃ q₄ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
      q₁ > 20 ∧ q₂ > 20 ∧ q₃ > 20 ∧ q₄ > 20 →
      (q₁ + q₂ + q₃ + q₄) / 4 ≠ 83 / 3 :=
by sorry

end NUMINAMATH_CALUDE_prime_average_count_l1259_125986


namespace NUMINAMATH_CALUDE_not_prime_p_l1259_125965

theorem not_prime_p (x k : ℕ) (p : ℕ) (h : x^5 + 2*x + 3 = p*k) : ¬ Nat.Prime p := by
  sorry

end NUMINAMATH_CALUDE_not_prime_p_l1259_125965


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1259_125970

theorem unique_modular_congruence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ 100000 [ZMOD 9] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1259_125970


namespace NUMINAMATH_CALUDE_five_digit_divisibility_l1259_125907

def is_valid_digit (d : ℕ) : Prop := d ∈ ({2, 3, 4, 5, 6} : Set ℕ)

def digits_to_number (p q r s t : ℕ) : ℕ := p * 10000 + q * 1000 + r * 100 + s * 10 + t

theorem five_digit_divisibility (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t →
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t →
  (p * 100 + q * 10 + r) % 6 = 0 →
  (q * 100 + r * 10 + s) % 3 = 0 →
  (r * 100 + s * 10 + t) % 9 = 0 →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisibility_l1259_125907


namespace NUMINAMATH_CALUDE_adam_received_one_smiley_l1259_125943

/-- Represents the number of smileys each friend received -/
structure SmileyCounts where
  adam : ℕ
  mojmir : ℕ
  petr : ℕ
  pavel : ℕ

/-- The conditions of the problem -/
def validSmileyCounts (counts : SmileyCounts) : Prop :=
  counts.adam + counts.mojmir + counts.petr + counts.pavel = 52 ∧
  counts.adam ≥ 1 ∧
  counts.mojmir ≥ 1 ∧
  counts.petr ≥ 1 ∧
  counts.pavel ≥ 1 ∧
  counts.petr + counts.pavel = 33 ∧
  counts.mojmir > counts.adam ∧
  counts.mojmir > counts.petr ∧
  counts.mojmir > counts.pavel

theorem adam_received_one_smiley (counts : SmileyCounts) 
  (h : validSmileyCounts counts) : counts.adam = 1 := by
  sorry

end NUMINAMATH_CALUDE_adam_received_one_smiley_l1259_125943


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_h_l1259_125964

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when written in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_vertex_form_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_h_l1259_125964


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1259_125962

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1259_125962


namespace NUMINAMATH_CALUDE_budget_projection_l1259_125909

/-- Given the equation fp - w = 15000, where f = 7 and w = 70 + 210i, prove that p = 2153 + 30i -/
theorem budget_projection (f : ℝ) (w p : ℂ) 
  (eq : f * p - w = 15000)
  (hf : f = 7)
  (hw : w = 70 + 210 * Complex.I) : 
  p = 2153 + 30 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_budget_projection_l1259_125909


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1259_125984

theorem arithmetic_calculations :
  (0.25 + (-9) + (-1/4) - 11 = -20) ∧
  (-15 + 5 + 1/3 * (-6) = -12) ∧
  ((-3/8 - 1/6 + 3/4) * 24 = 5) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1259_125984


namespace NUMINAMATH_CALUDE_negative_deviation_notation_l1259_125973

/-- Represents a height deviation from the average. -/
structure HeightDeviation where
  value : ℝ

/-- The average height of the team. -/
def averageHeight : ℝ := 175

/-- Notation for height deviations. -/
def denoteDeviation (d : HeightDeviation) : ℝ := d.value

/-- Axiom: Positive deviation is denoted by a positive number. -/
axiom positive_deviation_notation (d : HeightDeviation) :
  d.value > 0 → denoteDeviation d > 0

/-- Theorem: Negative deviation should be denoted by a negative number. -/
theorem negative_deviation_notation (d : HeightDeviation) :
  d.value < 0 → denoteDeviation d < 0 :=
sorry

end NUMINAMATH_CALUDE_negative_deviation_notation_l1259_125973


namespace NUMINAMATH_CALUDE_system1_solution_system2_solution_l1259_125902

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), x + y = 2 ∧ 5 * x - 2 * (x + y) = 6 ∧ x = 2 ∧ y = 0 := by sorry

-- System 2
theorem system2_solution :
  ∃ (a b c : ℝ), a + b = 3 ∧ 5 * a + 3 * c = 1 ∧ a + b + c = 0 ∧ a = 2 ∧ b = 1 ∧ c = -3 := by sorry

end NUMINAMATH_CALUDE_system1_solution_system2_solution_l1259_125902


namespace NUMINAMATH_CALUDE_triangle_perimeter_when_area_equals_four_inradius_l1259_125966

/-- Given a triangle with an inscribed circle, if the area of the triangle is numerically equal to
    four times the radius of the inscribed circle, then the perimeter of the triangle is 8. -/
theorem triangle_perimeter_when_area_equals_four_inradius (A r s p : ℝ) :
  A > 0 → r > 0 → s > 0 → p > 0 →
  A = r * s →  -- Area formula using inradius and semiperimeter
  A = 4 * r →  -- Given condition
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  p = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_when_area_equals_four_inradius_l1259_125966


namespace NUMINAMATH_CALUDE_quadrilateral_circle_condition_l1259_125983

-- Define the lines
def line1 (a x y : ℝ) : Prop := (a + 2) * x + (1 - a) * y - 3 = 0
def line2 (a x y : ℝ) : Prop := (a - 1) * x + (2 * a + 3) * y + 2 = 0

-- Define the property of forming a quadrilateral with coordinate axes
def forms_quadrilateral (a : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    line1 a x1 0 ∧ line1 a 0 y1 ∧ line2 a x2 0 ∧ line2 a 0 y2

-- Define the property of having a circumscribed circle
def has_circumscribed_circle (a : ℝ) : Prop :=
  forms_quadrilateral a → 
    (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0

-- The theorem to prove
theorem quadrilateral_circle_condition (a : ℝ) :
  forms_quadrilateral a → has_circumscribed_circle a → (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_circle_condition_l1259_125983


namespace NUMINAMATH_CALUDE_incorrect_statement_proof_l1259_125956

structure VisionSurvey where
  total_students : Nat
  sample_size : Nat
  is_about_vision : Bool

def is_correct_statement (s : VisionSurvey) (statement : String) : Prop :=
  match statement with
  | "The sample size is correct" => s.sample_size = 40
  | "The sample is about vision of selected students" => s.is_about_vision
  | "The population is about vision of all students" => s.is_about_vision
  | "The individual refers to each student" => false
  | _ => false

theorem incorrect_statement_proof (s : VisionSurvey) 
  (h1 : s.total_students = 400) 
  (h2 : s.sample_size = 40) 
  (h3 : s.is_about_vision = true) :
  ¬(is_correct_statement s "The individual refers to each student") := by
  sorry

end NUMINAMATH_CALUDE_incorrect_statement_proof_l1259_125956


namespace NUMINAMATH_CALUDE_divya_age_l1259_125935

theorem divya_age (divya_age nacho_age : ℝ) : 
  nacho_age + 5 = 3 * (divya_age + 5) →
  nacho_age + divya_age = 40 →
  divya_age = 7.5 := by
sorry

end NUMINAMATH_CALUDE_divya_age_l1259_125935


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l1259_125915

/-- The equation 4(3x-b) = 3(4x + 16) has infinitely many solutions x if and only if b = -12 -/
theorem infinite_solutions_iff_b_eq_neg_twelve (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_b_eq_neg_twelve_l1259_125915


namespace NUMINAMATH_CALUDE_abc_inequality_l1259_125982

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 - a*b = c^2) : (a - c) * (b - c) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1259_125982


namespace NUMINAMATH_CALUDE_percentage_calculation_l1259_125981

theorem percentage_calculation : 
  let initial_value : ℝ := 180
  let percentage : ℝ := 1/3
  let divisor : ℝ := 6
  (initial_value * (percentage / 100)) / divisor = 0.1 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1259_125981


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_square_extensions_l1259_125998

/-- The radius of a circle tangent to the extensions of two sides of a square,
    where two tangents from the opposite corner form a specific angle. -/
theorem circle_radius_tangent_to_square_extensions 
  (side_length : ℝ) 
  (tangent_angle : ℝ) 
  (sin_half_angle : ℝ) :
  side_length = 6 + 2 * Real.sqrt 5 →
  tangent_angle = 36 →
  sin_half_angle = (Real.sqrt 5 - 1) / 4 →
  ∃ (radius : ℝ), 
    radius = 2 * (2 * Real.sqrt 2 + Real.sqrt 5 - 1) ∧
    radius = side_length * Real.sqrt 2 / 
      ((4 / (Real.sqrt 5 - 1)) - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_square_extensions_l1259_125998


namespace NUMINAMATH_CALUDE_max_value_product_l1259_125921

theorem max_value_product (a b c x y z : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_product_l1259_125921


namespace NUMINAMATH_CALUDE_fraction_equality_l1259_125999

theorem fraction_equality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h1 : (a + b + c) / (a + b - c) = 7)
  (h2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1259_125999


namespace NUMINAMATH_CALUDE_a_values_l1259_125917

/-- The set of real numbers x such that x^2 - 2x - 8 = 0 -/
def A : Set ℝ := {x | x^2 - 2*x - 8 = 0}

/-- The set of real numbers x such that x^2 + a*x + a^2 - 12 = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a^2 - 12 = 0}

/-- The set of all possible values for a given the conditions -/
def possible_a : Set ℝ := {a | a < -4 ∨ a = -2 ∨ a ≥ 4}

theorem a_values (h : A ∪ B a = A) : a ∈ possible_a := by
  sorry

end NUMINAMATH_CALUDE_a_values_l1259_125917


namespace NUMINAMATH_CALUDE_cycle_original_price_l1259_125925

/-- Proves that given a cycle sold at a loss of 18% with a selling price of 1148, the original price of the cycle was 1400. -/
theorem cycle_original_price (loss_percentage : ℝ) (selling_price : ℝ) (original_price : ℝ) : 
  loss_percentage = 18 →
  selling_price = 1148 →
  selling_price = (1 - loss_percentage / 100) * original_price →
  original_price = 1400 := by
sorry

end NUMINAMATH_CALUDE_cycle_original_price_l1259_125925


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1259_125932

theorem tangent_line_to_parabola (x y : ℝ) :
  let f : ℝ → ℝ := λ t => t^2
  let tangent_point : ℝ × ℝ := (1, 1)
  let slope : ℝ := 2 * tangent_point.1
  2 * x - y - 1 = 0 ↔ y = slope * (x - tangent_point.1) + tangent_point.2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1259_125932


namespace NUMINAMATH_CALUDE_prob_two_blue_balls_l1259_125990

/-- The probability of drawing two blue balls from an urn --/
theorem prob_two_blue_balls (total : ℕ) (blue : ℕ) (h1 : total = 10) (h2 : blue = 5) :
  (blue.choose 2 : ℚ) / total.choose 2 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_blue_balls_l1259_125990


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1259_125924

def A (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ⊆ B) ↔ (a = -2 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1259_125924


namespace NUMINAMATH_CALUDE_solution_set_correct_l1259_125968

/-- The solution set of the inequality -x^2 + 2x > 0 -/
def SolutionSet : Set ℝ := {x | 0 < x ∧ x < 2}

/-- Theorem stating that SolutionSet is the correct solution to the inequality -/
theorem solution_set_correct :
  ∀ x : ℝ, x ∈ SolutionSet ↔ -x^2 + 2*x > 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_correct_l1259_125968


namespace NUMINAMATH_CALUDE_triangle_angles_l1259_125977

theorem triangle_angles (a b c : ℝ) (ha : a = 3) (hb : b = Real.sqrt 11) (hc : c = 2 + Real.sqrt 5) :
  ∃ (A B C : ℝ), 
    (0 < A ∧ A < π) ∧ 
    (0 < B ∧ B < π) ∧ 
    (0 < C ∧ C < π) ∧ 
    A + B + C = π ∧
    B = C ∧
    A = π - 2*B := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_l1259_125977


namespace NUMINAMATH_CALUDE_karl_net_income_l1259_125985

/-- Represents the sale of boots and subsequent transactions -/
structure BootSale where
  initial_price : ℚ
  actual_sale_price : ℚ
  reduced_price : ℚ
  refund_amount : ℚ
  candy_expense : ℚ
  actual_refund : ℚ

/-- Calculates the net income from a boot sale -/
def net_income (sale : BootSale) : ℚ :=
  sale.actual_sale_price * 2 - sale.refund_amount

/-- Theorem stating that Karl's net income is 20 talers -/
theorem karl_net_income (sale : BootSale) 
  (h1 : sale.initial_price = 25)
  (h2 : sale.actual_sale_price = 12.5)
  (h3 : sale.reduced_price = 10)
  (h4 : sale.refund_amount = 5)
  (h5 : sale.candy_expense = 3)
  (h6 : sale.actual_refund = 1) :
  net_income sale = 20 := by
  sorry


end NUMINAMATH_CALUDE_karl_net_income_l1259_125985


namespace NUMINAMATH_CALUDE_joes_test_count_l1259_125991

/-- Given Joe's test scores, prove the number of initial tests --/
theorem joes_test_count (initial_avg : ℚ) (lowest_score : ℚ) (new_avg : ℚ) :
  initial_avg = 40 →
  lowest_score = 25 →
  new_avg = 45 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * initial_avg = ((n : ℚ) - 1) * new_avg + lowest_score ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_joes_test_count_l1259_125991


namespace NUMINAMATH_CALUDE_sphere_surface_area_equal_volume_cone_l1259_125997

/-- Given a cone with radius 2 inches and height 6 inches, 
    prove that the surface area of a sphere with the same volume 
    is 4π(6^(2/3)) square inches. -/
theorem sphere_surface_area_equal_volume_cone (π : ℝ) : 
  let cone_radius : ℝ := 2
  let cone_height : ℝ := 6
  let cone_volume : ℝ := (1/3) * π * cone_radius^2 * cone_height
  let sphere_radius : ℝ := (3 * cone_volume / (4 * π))^(1/3)
  let sphere_surface_area : ℝ := 4 * π * sphere_radius^2
  sphere_surface_area = 4 * π * 6^(2/3) := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_equal_volume_cone_l1259_125997


namespace NUMINAMATH_CALUDE_valid_rental_plans_l1259_125993

/-- Represents a bus rental plan --/
structure RentalPlan where
  typeA : Nat  -- Number of Type A buses
  typeB : Nat  -- Number of Type B buses

/-- Checks if a rental plan can accommodate exactly the given number of students --/
def isValidPlan (plan : RentalPlan) (totalStudents : Nat) (typeACapacity : Nat) (typeBCapacity : Nat) : Prop :=
  plan.typeA * typeACapacity + plan.typeB * typeBCapacity = totalStudents

/-- Theorem stating that the three given rental plans are valid for 37 students --/
theorem valid_rental_plans :
  let totalStudents := 37
  let typeACapacity := 8
  let typeBCapacity := 4
  let plan1 : RentalPlan := ⟨2, 6⟩
  let plan2 : RentalPlan := ⟨3, 4⟩
  let plan3 : RentalPlan := ⟨4, 2⟩
  isValidPlan plan1 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan2 totalStudents typeACapacity typeBCapacity ∧
  isValidPlan plan3 totalStudents typeACapacity typeBCapacity :=
by sorry


end NUMINAMATH_CALUDE_valid_rental_plans_l1259_125993


namespace NUMINAMATH_CALUDE_euler_line_parallel_l1259_125910

/-- Triangle ABC with vertices A(-3,0), B(3,0), and C(3,3) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨-3, 0⟩, ⟨3, 0⟩, ⟨3, 3⟩}

/-- The Euler line of a triangle -/
def euler_line (t : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  sorry

/-- A line with equation ax + (a^2 - 3)y - 9 = 0 -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + (a^2 - 3) * p.2 - 9 = 0}

/-- Two lines are parallel -/
def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  sorry

theorem euler_line_parallel :
  ∀ a : ℝ, parallel (line_l a) (euler_line triangle_ABC) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_euler_line_parallel_l1259_125910


namespace NUMINAMATH_CALUDE_masked_digits_unique_solution_l1259_125923

def is_valid_pair (d : Nat) : Bool :=
  let product := d * d
  product ≥ 10 ∧ product < 100 ∧ product % 10 ≠ d

def get_last_digit (n : Nat) : Nat :=
  n % 10

theorem masked_digits_unique_solution :
  ∃! (elephant mouse pig panda : Nat),
    elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧
    mouse ≠ pig ∧ mouse ≠ panda ∧
    pig ≠ panda ∧
    is_valid_pair mouse ∧
    get_last_digit (mouse * mouse) = elephant ∧
    elephant = 6 ∧ mouse = 4 ∧ pig = 8 ∧ panda = 1 :=
by sorry

end NUMINAMATH_CALUDE_masked_digits_unique_solution_l1259_125923


namespace NUMINAMATH_CALUDE_rectangle_with_equal_sums_l1259_125906

/-- A regular polygon with 2004 sides -/
structure RegularPolygon2004 where
  vertices : Fin 2004 → ℕ
  vertex_range : ∀ i, 1 ≤ vertices i ∧ vertices i ≤ 501

/-- Four vertices form a rectangle in a regular 2004-sided polygon -/
def isRectangle (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  (b - a) % 2004 = (d - c) % 2004 ∧ (c - b) % 2004 = (a - d) % 2004

/-- The sums of numbers assigned to opposite vertices are equal -/
def equalOppositeSums (p : RegularPolygon2004) (a b c d : Fin 2004) : Prop :=
  p.vertices a + p.vertices c = p.vertices b + p.vertices d

/-- Main theorem: There exist four vertices forming a rectangle with equal opposite sums -/
theorem rectangle_with_equal_sums (p : RegularPolygon2004) :
  ∃ a b c d : Fin 2004, isRectangle p a b c d ∧ equalOppositeSums p a b c d := by
  sorry


end NUMINAMATH_CALUDE_rectangle_with_equal_sums_l1259_125906


namespace NUMINAMATH_CALUDE_eliana_refills_l1259_125980

theorem eliana_refills (total_spent : ℕ) (cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) :
  total_spent / cost_per_refill = 3 := by
  sorry

end NUMINAMATH_CALUDE_eliana_refills_l1259_125980


namespace NUMINAMATH_CALUDE_symmetry_y_axis_values_l1259_125963

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetry_y_axis_values :
  ∀ a b : ℝ, symmetric_y_axis a (-3) 2 b → a = -2 ∧ b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_y_axis_values_l1259_125963


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l1259_125938

theorem parametric_to_standard_equation (x y θ : ℝ) 
  (h1 : x = 1 + 2 * Real.cos θ) 
  (h2 : y = 2 * Real.sin θ) : 
  (x - 1)^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l1259_125938


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1259_125948

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a > 0) (h2 : c < 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1259_125948


namespace NUMINAMATH_CALUDE_tangerines_taken_l1259_125927

/-- Represents the contents of Tina's fruit bag -/
structure FruitBag where
  apples : Nat
  oranges : Nat
  tangerines : Nat

/-- Represents the number of fruits taken away -/
structure FruitsTaken where
  oranges : Nat
  tangerines : Nat

def initialBag : FruitBag := { apples := 9, oranges := 5, tangerines := 17 }

def orangesTaken : Nat := 2

theorem tangerines_taken (bag : FruitBag) (taken : FruitsTaken) : 
  bag.oranges - taken.oranges + 4 = bag.tangerines - taken.tangerines →
  taken.tangerines = 10 := by
  sorry

#check tangerines_taken initialBag { oranges := orangesTaken, tangerines := 10 }

end NUMINAMATH_CALUDE_tangerines_taken_l1259_125927


namespace NUMINAMATH_CALUDE_haley_concert_spending_l1259_125929

/-- The amount spent on concert tickets -/
def concert_spending (ticket_price : ℕ) (tickets_for_self : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_for_self + extra_tickets)

theorem haley_concert_spending :
  concert_spending 4 3 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_haley_concert_spending_l1259_125929


namespace NUMINAMATH_CALUDE_max_xy_value_l1259_125958

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 6 * x + 8 * y = 72) :
  x * y ≤ 27 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 6 * x₀ + 8 * y₀ = 72 ∧ x₀ * y₀ = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l1259_125958


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1259_125955

/-- Given a line L1 with equation 4x + 5y = 10 and a perpendicular line L2 with y-intercept -3,
    the x-intercept of L2 is 12/5 -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 4 * x + 5 * y = 10) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 3) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₂ - y₁) * (4 * (x + 1) + 5 * y₁ - 10) = 0) →
  (0, -3) ∈ L2 →
  (12/5, 0) ∈ L2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1259_125955


namespace NUMINAMATH_CALUDE_next_birthday_age_is_56_l1259_125933

/-- Represents a person's age in years, months, weeks, and days -/
structure Age where
  years : ℕ
  months : ℕ
  weeks : ℕ
  days : ℕ

/-- Calculates the age on the next birthday given a current age -/
def nextBirthdayAge (currentAge : Age) : ℕ :=
  sorry

/-- Theorem stating that given the specific age, the next birthday age will be 56 -/
theorem next_birthday_age_is_56 :
  let currentAge : Age := { years := 50, months := 50, weeks := 50, days := 50 }
  nextBirthdayAge currentAge = 56 := by
  sorry

end NUMINAMATH_CALUDE_next_birthday_age_is_56_l1259_125933


namespace NUMINAMATH_CALUDE_remainder_18_pow_63_mod_5_l1259_125941

theorem remainder_18_pow_63_mod_5 : 18^63 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_18_pow_63_mod_5_l1259_125941


namespace NUMINAMATH_CALUDE_S_bounds_l1259_125960

def S : Set ℝ := { y | ∃ x : ℝ, x ≥ 0 ∧ y = (2 * x + 3) / (x + 2) }

theorem S_bounds :
  ∃ (m M : ℝ),
    (∀ y ∈ S, m ≤ y) ∧
    (∀ y ∈ S, y ≤ M) ∧
    m ∈ S ∧
    M ∉ S ∧
    m = 3/2 ∧
    M = 2 := by
  sorry


end NUMINAMATH_CALUDE_S_bounds_l1259_125960


namespace NUMINAMATH_CALUDE_camden_swim_count_l1259_125913

/-- The number of weeks in March -/
def weeks_in_march : ℕ := 4

/-- The number of times Susannah went swimming in March -/
def susannah_swims : ℕ := 24

/-- The difference in weekly swims between Susannah and Camden -/
def weekly_swim_difference : ℕ := 2

/-- Camden's total number of swims in March -/
def camden_swims : ℕ := 16

theorem camden_swim_count :
  (susannah_swims / weeks_in_march - weekly_swim_difference) * weeks_in_march = camden_swims := by
  sorry

end NUMINAMATH_CALUDE_camden_swim_count_l1259_125913


namespace NUMINAMATH_CALUDE_weekly_egg_supply_l1259_125989

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of dozens of eggs supplied to the first store daily -/
def store1_supply : ℕ := 5

/-- The number of eggs supplied to the second store daily -/
def store2_supply : ℕ := 30

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem weekly_egg_supply : 
  (store1_supply * dozen + store2_supply) * days_in_week = 630 := by
  sorry

end NUMINAMATH_CALUDE_weekly_egg_supply_l1259_125989


namespace NUMINAMATH_CALUDE_shortest_paths_count_julia_paths_count_l1259_125922

theorem shortest_paths_count : Nat → Nat → Nat
| m, n => Nat.choose (m + n) m

theorem julia_paths_count : shortest_paths_count 8 5 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_shortest_paths_count_julia_paths_count_l1259_125922


namespace NUMINAMATH_CALUDE_sine_function_vertical_shift_l1259_125967

/-- Given a sine function y = a * sin(b * x) + d with positive constants a, b, and d,
    if the maximum value of y is 4 and the minimum value of y is -2, then d = 1. -/
theorem sine_function_vertical_shift 
  (a b d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hmax : ∀ x, a * Real.sin (b * x) + d ≤ 4)
  (hmin : ∀ x, a * Real.sin (b * x) + d ≥ -2)
  (hex_max : ∃ x, a * Real.sin (b * x) + d = 4)
  (hex_min : ∃ x, a * Real.sin (b * x) + d = -2) : 
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_vertical_shift_l1259_125967


namespace NUMINAMATH_CALUDE_candy_bar_sales_ratio_l1259_125936

theorem candy_bar_sales_ratio :
  ∀ (price : ℚ) (marvin_sales : ℕ) (tina_extra_earnings : ℚ),
    price = 2 →
    marvin_sales = 35 →
    tina_extra_earnings = 140 →
    ∃ (tina_sales : ℕ),
      tina_sales * price = marvin_sales * price + tina_extra_earnings ∧
      tina_sales = 3 * marvin_sales :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_sales_ratio_l1259_125936


namespace NUMINAMATH_CALUDE_injective_function_property_l1259_125952

theorem injective_function_property (f : ℕ → ℕ) :
  (∀ m n : ℕ, m > 0 → n > 0 → f (n * f m) ≤ n * m) →
  Function.Injective f →
  ∀ x : ℕ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_injective_function_property_l1259_125952


namespace NUMINAMATH_CALUDE_guppies_per_day_l1259_125937

/-- The number of guppies Jason's moray eel eats per day -/
def moray_eel_guppies : ℕ := 20

/-- The number of betta fish Jason has -/
def num_betta_fish : ℕ := 5

/-- The number of guppies each betta fish eats per day -/
def betta_fish_guppies : ℕ := 7

/-- Theorem: Jason needs to buy 55 guppies per day -/
theorem guppies_per_day : 
  moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 := by
  sorry

end NUMINAMATH_CALUDE_guppies_per_day_l1259_125937


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1259_125969

/-- Given an arithmetic sequence a with S₃ = 6, prove that 5a₁ + a₇ = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : 
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  (3 * a 1 + 3 * d = 6) →       -- S₃ = 6 condition
  5 * a 1 + a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1259_125969


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1259_125908

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ i : ℕ, i ∈ Finset.range 10 → i.succ ∣ m) → n ≤ m) :=
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l1259_125908


namespace NUMINAMATH_CALUDE_inequality_preservation_l1259_125931

theorem inequality_preservation (a b c : ℝ) (h : a < b) (h' : b < 0) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1259_125931


namespace NUMINAMATH_CALUDE_sequence_a_formula_l1259_125944

def sequence_a (n : ℕ+) : ℝ := sorry

def sum_S (n : ℕ+) : ℝ := sorry

axiom sum_S_2 : sum_S 2 = 4

axiom sequence_a_next (n : ℕ+) : sequence_a (n + 1) = 2 * sum_S n + 1

theorem sequence_a_formula (n : ℕ+) : sequence_a n = 3^(n.val - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l1259_125944


namespace NUMINAMATH_CALUDE_product_max_value_l1259_125939

theorem product_max_value (x y z u : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ u ≥ 0) 
  (h_constraint : 2*x + x*y + z + y*z*u = 1) : 
  x^2 * y^2 * z^2 * u ≤ 1/512 := by
  sorry

end NUMINAMATH_CALUDE_product_max_value_l1259_125939


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1259_125959

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 8)) →
  p = -67/3 ∧ q = -158/3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1259_125959
