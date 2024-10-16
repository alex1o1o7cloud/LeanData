import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l914_91417

theorem equation_solution : ∃ x : ℝ, 
  6 * ((1/2) * x - 4) + 2 * x = 7 - ((1/3) * x - 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l914_91417


namespace NUMINAMATH_CALUDE_tan_sum_ratio_l914_91474

theorem tan_sum_ratio : 
  (Real.tan (20 * π / 180) + Real.tan (40 * π / 180) + Real.tan (120 * π / 180)) / 
  (Real.tan (20 * π / 180) * Real.tan (40 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_ratio_l914_91474


namespace NUMINAMATH_CALUDE_largest_common_term_l914_91456

def is_in_ap1 (a : ℕ) : Prop := ∃ k : ℕ, a = 4 + 5 * k

def is_in_ap2 (a : ℕ) : Prop := ∃ k : ℕ, a = 7 + 11 * k

def is_common_term (a : ℕ) : Prop := is_in_ap1 a ∧ is_in_ap2 a

theorem largest_common_term :
  ∃ a : ℕ, a = 984 ∧ is_common_term a ∧ a < 1000 ∧
  ∀ b : ℕ, is_common_term b ∧ b < 1000 → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l914_91456


namespace NUMINAMATH_CALUDE_stair_cleaning_problem_l914_91430

theorem stair_cleaning_problem (a b c : ℕ) (h1 : a > c) (h2 : 101 * (a + c) + 20 * b = 746) :
  let n := 100 * a + 10 * b + c
  (2 * n = 944) ∨ (2 * n = 1142) := by
  sorry

end NUMINAMATH_CALUDE_stair_cleaning_problem_l914_91430


namespace NUMINAMATH_CALUDE_max_value_on_interval_l914_91429

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧
  (∀ x y, x > 0 → y > 0 → f x < f y) ∧
  (∀ x y, f (x + y) = f x + f y)

theorem max_value_on_interval 
  (f : ℝ → ℝ) 
  (h : f_properties f) :
  ∃ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), f y ≤ f x ∧ f x = -4 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_l914_91429


namespace NUMINAMATH_CALUDE_ball_max_height_l914_91437

/-- The height function of the ball's path -/
def h (t : ℝ) : ℝ := -16 * t^2 + 80 * t + 21

/-- Theorem stating that the maximum height of the ball is 121 feet -/
theorem ball_max_height :
  ∃ (t : ℝ), ∀ (s : ℝ), h s ≤ h t ∧ h t = 121 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l914_91437


namespace NUMINAMATH_CALUDE_hyperbola_focus_l914_91442

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - 4*y^2 - 6*x + 24*y - 11 = 0

-- Define the foci coordinates
def focus_coord (x y : ℝ) : Prop :=
  (x = 3 ∧ y = 3 + 2 * Real.sqrt 5) ∨ (x = 3 ∧ y = 3 - 2 * Real.sqrt 5)

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ focus_coord x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l914_91442


namespace NUMINAMATH_CALUDE_olivias_groceries_cost_l914_91480

/-- The total cost of Olivia's groceries is $42 -/
theorem olivias_groceries_cost (banana_cost bread_cost milk_cost apple_cost : ℕ)
  (h1 : banana_cost = 12)
  (h2 : bread_cost = 9)
  (h3 : milk_cost = 7)
  (h4 : apple_cost = 14) :
  banana_cost + bread_cost + milk_cost + apple_cost = 42 := by
  sorry

end NUMINAMATH_CALUDE_olivias_groceries_cost_l914_91480


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l914_91463

/-- Represents the game setup with total boxes and high-value boxes -/
structure GameSetup where
  total_boxes : ℕ
  high_value_boxes : ℕ

/-- Calculates the probability of holding a high-value box -/
def probability_high_value (g : GameSetup) (eliminated : ℕ) : ℚ :=
  g.high_value_boxes / (g.total_boxes - eliminated)

/-- Theorem stating that eliminating 7 boxes results in at least 50% chance of high-value box -/
theorem deal_or_no_deal_probability 
  (g : GameSetup) 
  (h1 : g.total_boxes = 30) 
  (h2 : g.high_value_boxes = 8) : 
  probability_high_value g 7 ≥ 1/2 := by
  sorry

#eval probability_high_value ⟨30, 8⟩ 7

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l914_91463


namespace NUMINAMATH_CALUDE_junior_score_l914_91461

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_ratio : ℝ := 0.2
  let senior_ratio : ℝ := 0.8
  let class_average : ℝ := 78
  let senior_average : ℝ := 75
  let junior_count : ℝ := junior_ratio * n
  let senior_count : ℝ := senior_ratio * n
  let total_score : ℝ := class_average * n
  let senior_total_score : ℝ := senior_average * senior_count
  let junior_total_score : ℝ := total_score - senior_total_score
  junior_total_score / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l914_91461


namespace NUMINAMATH_CALUDE_triangle_max_area_l914_91427

theorem triangle_max_area (a b c : ℝ) (h1 : a = 75) (h2 : c = 2 * b) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (∀ x : ℝ, x > 0 → area ≤ 1100) ∧ (∃ x : ℝ, x > 0 ∧ area = 1100) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l914_91427


namespace NUMINAMATH_CALUDE_train_length_calculation_l914_91496

/-- Calculates the length of a train given its speed, the length of a bridge it passes, and the time it takes to pass the bridge. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (passing_time : ℝ) : 
  train_speed = 50 * (1000 / 3600) → 
  bridge_length = 140 →
  passing_time = 36 →
  (train_speed * passing_time) - bridge_length = 360 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l914_91496


namespace NUMINAMATH_CALUDE_equation_solution_l914_91426

theorem equation_solution (a : ℚ) : -3 / (a - 3) = 3 / (a + 2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l914_91426


namespace NUMINAMATH_CALUDE_cos_22_5_squared_minus_sin_22_5_squared_l914_91451

theorem cos_22_5_squared_minus_sin_22_5_squared : 
  Real.cos (22.5 * π / 180) ^ 2 - Real.sin (22.5 * π / 180) ^ 2 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_22_5_squared_minus_sin_22_5_squared_l914_91451


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l914_91454

theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 3)^2 + 4*(a 3) + 1 = 0 →  -- a_3 is a root of x^2 + 4x + 1 = 0
  (a 15)^2 + 4*(a 15) + 1 = 0 →  -- a_15 is a root of x^2 + 4x + 1 = 0
  a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l914_91454


namespace NUMINAMATH_CALUDE_tim_income_percentage_l914_91450

theorem tim_income_percentage (tim juan mary : ℝ) 
  (h1 : mary = 1.7 * tim) 
  (h2 : mary = 1.02 * juan) : 
  (juan - tim) / juan = 0.4 := by
sorry

end NUMINAMATH_CALUDE_tim_income_percentage_l914_91450


namespace NUMINAMATH_CALUDE_quadratic_roots_l914_91482

theorem quadratic_roots (a b c : ℝ) (h : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁^3 - x₂^3 = 2011) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ a * y₁^2 + 2 * b * y₁ + 4 * c = 0 ∧ a * y₂^2 + 2 * b * y₂ + 4 * c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l914_91482


namespace NUMINAMATH_CALUDE_tank_filled_at_10pm_l914_91475

/-- Represents the rainfall rate at a given hour after 1 pm -/
def rainfall_rate (hour : ℕ) : ℝ :=
  if hour = 0 then 2
  else if hour ≤ 4 then 1
  else 3

/-- Calculates the total rainfall up to a given hour after 1 pm -/
def total_rainfall (hour : ℕ) : ℝ :=
  (Finset.range (hour + 1)).sum rainfall_rate

/-- The height of the fish tank in inches -/
def tank_height : ℝ := 18

/-- Theorem stating that the fish tank will be filled at 10 pm -/
theorem tank_filled_at_10pm :
  ∃ (h : ℕ), h = 9 ∧ total_rainfall h ≥ tank_height ∧ total_rainfall (h - 1) < tank_height :=
sorry

end NUMINAMATH_CALUDE_tank_filled_at_10pm_l914_91475


namespace NUMINAMATH_CALUDE_congruence_solution_l914_91428

theorem congruence_solution (n : ℤ) : 13 * n ≡ 8 [ZMOD 47] ↔ n ≡ 29 [ZMOD 47] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l914_91428


namespace NUMINAMATH_CALUDE_max_pads_purchase_existence_of_max_purchase_l914_91473

def cost_pin : ℕ := 2
def cost_pen : ℕ := 3
def cost_pad : ℕ := 9
def total_budget : ℕ := 60

def is_valid_purchase (pins pens pads : ℕ) : Prop :=
  pins ≥ 1 ∧ pens ≥ 1 ∧ pads ≥ 1 ∧
  cost_pin * pins + cost_pen * pens + cost_pad * pads = total_budget

theorem max_pads_purchase :
  ∀ pins pens pads : ℕ, is_valid_purchase pins pens pads → pads ≤ 5 :=
by sorry

theorem existence_of_max_purchase :
  ∃ pins pens : ℕ, is_valid_purchase pins pens 5 :=
by sorry

end NUMINAMATH_CALUDE_max_pads_purchase_existence_of_max_purchase_l914_91473


namespace NUMINAMATH_CALUDE_division_problem_l914_91490

theorem division_problem :
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  (quotient = 84) ∧ 
  (remainder = 232) ∧ 
  (remainder + 375 = 607) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l914_91490


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l914_91400

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l914_91400


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l914_91452

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: If a point satisfies x + y < 0 and xy > 0, then it's in the third quadrant -/
theorem point_in_third_quadrant (A : Point) 
    (sum_condition : A.x + A.y < 0) 
    (product_condition : A.x * A.y > 0) : 
    isInThirdQuadrant A := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l914_91452


namespace NUMINAMATH_CALUDE_smaller_prime_factor_l914_91483

theorem smaller_prime_factor : ∃ p : ℕ, 
  Prime p ∧ 
  p > 4002001 ∧ 
  316990099009901 = 4002001 * p ∧
  316990099009901 = 32016000000000001 / 101 := by
sorry

end NUMINAMATH_CALUDE_smaller_prime_factor_l914_91483


namespace NUMINAMATH_CALUDE_shopkeeper_cards_l914_91489

/-- The number of cards in a complete deck of standard playing cards -/
def standard_deck : ℕ := 52

/-- The number of cards in a complete deck of Uno cards -/
def uno_deck : ℕ := 108

/-- The number of cards in a complete deck of tarot cards -/
def tarot_deck : ℕ := 78

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 4

/-- The number of complete decks of Uno cards -/
def uno_decks : ℕ := 3

/-- The number of complete decks of tarot cards -/
def tarot_decks : ℕ := 5

/-- The number of additional standard playing cards -/
def extra_standard : ℕ := 12

/-- The number of additional Uno cards -/
def extra_uno : ℕ := 7

/-- The number of additional tarot cards -/
def extra_tarot : ℕ := 9

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * standard_deck + extra_standard +
  uno_decks * uno_deck + extra_uno +
  tarot_decks * tarot_deck + extra_tarot

theorem shopkeeper_cards : total_cards = 950 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cards_l914_91489


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l914_91476

theorem quadratic_root_sqrt5_minus3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 + 1) ∧
  a = 1 ∧ b = 2 ∧ c = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt5_minus3_l914_91476


namespace NUMINAMATH_CALUDE_cubic_expression_property_l914_91402

theorem cubic_expression_property (a b : ℝ) :
  a * (3^3) + b * 3 - 5 = 20 → a * ((-3)^3) + b * (-3) - 5 = -30 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_property_l914_91402


namespace NUMINAMATH_CALUDE_square_root_properties_l914_91420

theorem square_root_properties (x c d e f : ℝ) : 
  (x^3 - x^2 - 6*x + 2 = 0 → (x^2)^3 - 13*(x^2)^2 + 40*(x^2) - 4 = 0) ∧
  (x^4 + c*x^3 + d*x^2 + e*x + f = 0 → 
    (x^2)^4 + (2*d - c^2)*(x^2)^3 + (d^2 - 2*c*e + 2*f)*(x^2)^2 + (2*d*f - e^2)*(x^2) + f^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_square_root_properties_l914_91420


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_union_complement_B_A_l914_91481

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- State the theorems
theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≥ 6 ∨ x < 3} := by sorry

theorem union_complement_B_A : 
  Bᶜ ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_union_complement_B_A_l914_91481


namespace NUMINAMATH_CALUDE_unique_root_implies_specific_function_max_min_on_interval_l914_91471

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- Theorem 1
theorem unique_root_implies_specific_function (a b : ℝ) (h1 : a ≠ 0) (h2 : f a b 2 = 0) 
  (h3 : ∃! x, f a b x - x = 0) : 
  ∀ x, f (-1/2) 1 x = f a b x := by sorry

-- Theorem 2
theorem max_min_on_interval (x : ℝ) (h : x ∈ Set.Icc (-1) 2) : 
  f 1 (-2) x ≤ 3 ∧ f 1 (-2) x ≥ -1 ∧ 
  (∃ x₁ ∈ Set.Icc (-1) 2, f 1 (-2) x₁ = 3) ∧ 
  (∃ x₂ ∈ Set.Icc (-1) 2, f 1 (-2) x₂ = -1) := by sorry

end NUMINAMATH_CALUDE_unique_root_implies_specific_function_max_min_on_interval_l914_91471


namespace NUMINAMATH_CALUDE_part_one_part_two_l914_91472

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one : 
  (A (-1) ∪ B = {x | x < 2 ∨ x > 5}) ∧ 
  ((Set.univ \ A (-1)) ∩ B = {x | x < -2 ∨ x > 5}) := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (a ≥ 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l914_91472


namespace NUMINAMATH_CALUDE_arithmetic_computation_l914_91495

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l914_91495


namespace NUMINAMATH_CALUDE_adultChildRatioIsTwo_l914_91432

/-- Represents the ticket prices and attendance information for a show -/
structure ShowInfo where
  adultTicketPrice : ℚ
  childTicketPrice : ℚ
  totalReceipts : ℚ
  numAdults : ℕ

/-- Calculates the ratio of adults to children given show information -/
def adultChildRatio (info : ShowInfo) : ℚ :=
  let numChildren := (info.totalReceipts - info.adultTicketPrice * info.numAdults) / info.childTicketPrice
  info.numAdults / numChildren

/-- Theorem stating that the ratio of adults to children is 2:1 for the given show information -/
theorem adultChildRatioIsTwo (info : ShowInfo) 
    (h1 : info.adultTicketPrice = 11/2)
    (h2 : info.childTicketPrice = 5/2)
    (h3 : info.totalReceipts = 1026)
    (h4 : info.numAdults = 152) : 
  adultChildRatio info = 2 := by
  sorry

#eval adultChildRatio {
  adultTicketPrice := 11/2,
  childTicketPrice := 5/2,
  totalReceipts := 1026,
  numAdults := 152
}

end NUMINAMATH_CALUDE_adultChildRatioIsTwo_l914_91432


namespace NUMINAMATH_CALUDE_total_cans_count_l914_91458

-- Define the given conditions
def total_oil : ℕ := 290
def small_cans : ℕ := 10
def small_can_volume : ℕ := 8
def large_can_volume : ℕ := 15

-- State the theorem
theorem total_cans_count : 
  ∃ (large_cans : ℕ), 
    small_cans * small_can_volume + large_cans * large_can_volume = total_oil ∧
    small_cans + large_cans = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_count_l914_91458


namespace NUMINAMATH_CALUDE_circle_equation_l914_91478

theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-2, 3)
  let is_tangent_to_x_axis : Prop := ∃ (x_0 : ℝ), (x_0 + 2)^2 + 3^2 = (x + 2)^2 + (y - 3)^2
  is_tangent_to_x_axis →
  (x + 2)^2 + (y - 3)^2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l914_91478


namespace NUMINAMATH_CALUDE_rectangle_fold_theorem_l914_91419

theorem rectangle_fold_theorem : ∃ (a b : ℕ+), 
  a ≤ b ∧ 
  (a.val : ℝ) / (b.val : ℝ) * Real.sqrt ((a.val : ℝ)^2 + (b.val : ℝ)^2) = 65 ∧
  2 * (a.val + b.val) = 408 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_theorem_l914_91419


namespace NUMINAMATH_CALUDE_f_extremum_l914_91418

/-- The function f(x, y) -/
def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

/-- Theorem stating the extremum of f(x, y) -/
theorem f_extremum :
  (∃ (x y : ℝ), f x y = 10 ∧ ∀ (a b : ℝ), f a b ≥ 10) ∧
  (∃ (x y : ℝ), f x y = 570 ∧ ∀ (a b : ℝ), f a b ≤ 570) :=
sorry

end NUMINAMATH_CALUDE_f_extremum_l914_91418


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l914_91460

theorem simplify_and_rationalize : 
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 
  (2 * Real.sqrt 7) / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l914_91460


namespace NUMINAMATH_CALUDE_wine_exchange_equation_l914_91421

/-- Represents the value of clear wine in terms of grain -/
def clear_wine_value : ℝ := 10

/-- Represents the value of turbid wine in terms of grain -/
def turbid_wine_value : ℝ := 3

/-- Represents the total amount of grain used -/
def total_grain : ℝ := 30

/-- Represents the total amount of wine obtained -/
def total_wine : ℝ := 5

/-- Proves that the equation 10x + 3(5-x) = 30 correctly represents the problem -/
theorem wine_exchange_equation (x : ℝ) : 
  x ≥ 0 ∧ x ≤ total_wine → 
  clear_wine_value * x + turbid_wine_value * (total_wine - x) = total_grain := by
sorry

end NUMINAMATH_CALUDE_wine_exchange_equation_l914_91421


namespace NUMINAMATH_CALUDE_move_right_result_l914_91466

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Move a point horizontally in the Cartesian coordinate system -/
def moveHorizontal (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The initial point (-1, 2) -/
def initialPoint : Point :=
  { x := -1, y := 2 }

/-- The number of units to move right -/
def moveRightUnits : ℝ := 3

/-- The final point after moving -/
def finalPoint : Point := moveHorizontal initialPoint moveRightUnits

/-- Theorem: Moving the initial point 3 units to the right results in (2, 2) -/
theorem move_right_result :
  finalPoint = { x := 2, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_move_right_result_l914_91466


namespace NUMINAMATH_CALUDE_min_k_value_k_range_l914_91444

/-- Given that for all a ∈ (-∞, 0) and all x ∈ (0, +∞), 
    the inequality x^2 + (3-a)x + 3 - 2a^2 < ke^x holds,
    prove that the minimum value of k is 3. -/
theorem min_k_value (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) → 
  k ≥ 3 := by
  sorry

/-- The range of k is [3, +∞) -/
theorem k_range (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) ↔ 
  k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_k_value_k_range_l914_91444


namespace NUMINAMATH_CALUDE_sally_weekend_pages_l914_91449

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- Theorem: Sally reads 20 pages on each weekend day -/
theorem sally_weekend_pages : 
  (total_pages - weekday_pages * weekdays_per_week * weeks_to_finish) / (weekend_days_per_week * weeks_to_finish) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_weekend_pages_l914_91449


namespace NUMINAMATH_CALUDE_nanning_gdp_scientific_notation_l914_91423

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem nanning_gdp_scientific_notation :
  let gdp : ℝ := 1060 * 10^9  -- 1060 billion
  let scientific_form := toScientificNotation gdp
  scientific_form.coefficient = 1.06 ∧ scientific_form.exponent = 11 :=
by sorry

end NUMINAMATH_CALUDE_nanning_gdp_scientific_notation_l914_91423


namespace NUMINAMATH_CALUDE_fraction_inequality_l914_91457

theorem fraction_inequality (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l914_91457


namespace NUMINAMATH_CALUDE_stationery_sales_distribution_l914_91499

theorem stationery_sales_distribution (pen_sales pencil_sales eraser_sales : ℝ) 
  (h_pen : pen_sales = 42)
  (h_pencil : pencil_sales = 25)
  (h_eraser : eraser_sales = 12)
  (h_total : pen_sales + pencil_sales + eraser_sales + (100 - pen_sales - pencil_sales - eraser_sales) = 100) :
  100 - pen_sales - pencil_sales - eraser_sales = 21 := by
sorry

end NUMINAMATH_CALUDE_stationery_sales_distribution_l914_91499


namespace NUMINAMATH_CALUDE_polygon_sides_l914_91425

/-- A polygon with equal internal angles and external angles equal to 2/3 of the adjacent internal angles has 5 sides. -/
theorem polygon_sides (n : ℕ) (internal_angle : ℝ) (external_angle : ℝ) : 
  n > 2 →
  internal_angle > 0 →
  external_angle > 0 →
  (n : ℝ) * internal_angle = (n - 2 : ℝ) * 180 →
  external_angle = (2 / 3) * internal_angle →
  internal_angle + external_angle = 180 →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l914_91425


namespace NUMINAMATH_CALUDE_paint_used_after_four_weeks_l914_91407

def initial_paint : ℚ := 520

def week1_fraction : ℚ := 1/4
def week2_fraction : ℚ := 1/3
def week3_fraction : ℚ := 3/8
def week4_fraction : ℚ := 1/5

def paint_used (initial : ℚ) (f1 f2 f3 f4 : ℚ) : ℚ :=
  let remaining1 := initial * (1 - f1)
  let remaining2 := remaining1 * (1 - f2)
  let remaining3 := remaining2 * (1 - f3)
  let remaining4 := remaining3 * (1 - f4)
  initial - remaining4

theorem paint_used_after_four_weeks :
  paint_used initial_paint week1_fraction week2_fraction week3_fraction week4_fraction = 390 := by
  sorry

end NUMINAMATH_CALUDE_paint_used_after_four_weeks_l914_91407


namespace NUMINAMATH_CALUDE_special_triangle_properties_l914_91470

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the specified conditions. -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧ t.c = Real.sqrt 7 ∧ Real.cos t.A + (1/2) * t.a = t.b

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.C = π/3 ∧ (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

#check special_triangle_properties

end NUMINAMATH_CALUDE_special_triangle_properties_l914_91470


namespace NUMINAMATH_CALUDE_total_full_spots_l914_91491

/-- Calculates the number of full parking spots in a multi-story parking garage -/
def fullParkingSpots : ℕ :=
  let totalLevels : ℕ := 7
  let firstLevelSpots : ℕ := 100
  let spotIncrease : ℕ := 50
  let firstLevelOpenSpots : ℕ := 58
  let openSpotDecrease : ℕ := 3
  let openSpotIncrease : ℕ := 10
  let switchLevel : ℕ := 4

  let totalFullSpots : ℕ := (List.range totalLevels).foldl
    (fun acc level =>
      let totalSpots := firstLevelSpots + level * spotIncrease
      let openSpots := if level < switchLevel - 1
        then firstLevelOpenSpots - level * openSpotDecrease
        else firstLevelOpenSpots - (switchLevel - 1) * openSpotDecrease + (level - switchLevel + 1) * openSpotIncrease
      acc + (totalSpots - openSpots))
    0

  totalFullSpots

/-- The theorem stating that the total number of full parking spots is 1329 -/
theorem total_full_spots : fullParkingSpots = 1329 := by
  sorry

end NUMINAMATH_CALUDE_total_full_spots_l914_91491


namespace NUMINAMATH_CALUDE_theodore_sturgeon_books_l914_91485

theorem theodore_sturgeon_books (h p : ℕ) : 
  h + p = 10 →
  30 * h + 20 * p = 250 →
  h = 5 :=
by sorry

end NUMINAMATH_CALUDE_theodore_sturgeon_books_l914_91485


namespace NUMINAMATH_CALUDE_bus_journey_l914_91436

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 6) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_l914_91436


namespace NUMINAMATH_CALUDE_retirement_fund_decrease_l914_91412

/-- Proves that the decrease in Kate's retirement fund is $12 --/
theorem retirement_fund_decrease (previous_value current_value : ℕ) 
  (h1 : previous_value = 1472)
  (h2 : current_value = 1460) : 
  previous_value - current_value = 12 := by
  sorry

end NUMINAMATH_CALUDE_retirement_fund_decrease_l914_91412


namespace NUMINAMATH_CALUDE_ratio_problem_l914_91453

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : e / f = 1 / 6)
  (h5 : a * b * c / (d * e * f) = 1 / 4) :
  d / e = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l914_91453


namespace NUMINAMATH_CALUDE_family_spent_38_dollars_l914_91415

def regular_ticket_price : ℝ := 5
def popcorn_price : ℝ := 0.8 * regular_ticket_price
def ticket_discount_rate : ℝ := 0.1
def soda_discount_rate : ℝ := 0.5
def num_tickets : ℕ := 4
def num_popcorn : ℕ := 2
def num_sodas : ℕ := 4

def discounted_ticket_price : ℝ := regular_ticket_price * (1 - ticket_discount_rate)
def soda_price : ℝ := popcorn_price  -- Assuming soda price is the same as popcorn price
def discounted_soda_price : ℝ := soda_price * (1 - soda_discount_rate)

theorem family_spent_38_dollars :
  let total_ticket_cost := num_tickets * discounted_ticket_price
  let total_popcorn_cost := num_popcorn * popcorn_price
  let total_soda_cost := num_popcorn * discounted_soda_price + (num_sodas - num_popcorn) * soda_price
  total_ticket_cost + total_popcorn_cost + total_soda_cost = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_spent_38_dollars_l914_91415


namespace NUMINAMATH_CALUDE_investment_problem_l914_91439

/-- The investment problem with three partners A, B, and C. -/
theorem investment_problem (investment_B investment_C : ℕ) 
  (profit_B : ℕ) (profit_diff_A_C : ℕ) (investment_A : ℕ) : 
  investment_B = 8000 →
  investment_C = 10000 →
  profit_B = 1000 →
  profit_diff_A_C = 500 →
  (investment_A : ℚ) / investment_B = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  (investment_A : ℚ) / investment_C = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  investment_A = 12000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l914_91439


namespace NUMINAMATH_CALUDE_parabola_focus_l914_91424

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  let (x, y) := p
  parabola x y ∧ x = 0 ∧ y = -1

-- Theorem statement
theorem parabola_focus :
  ∃ p : ℝ × ℝ, focus p parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l914_91424


namespace NUMINAMATH_CALUDE_problem_solution_l914_91492

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 5 + (2829/27) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l914_91492


namespace NUMINAMATH_CALUDE_equations_same_graph_l914_91405

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x^2 - 1
def equation_II (x y : ℝ) : Prop := x ≠ 1 → y = (x^3 - x) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^3 - x

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem equations_same_graph :
  (same_graph equation_II equation_III) ∧
  (¬ same_graph equation_I equation_II) ∧
  (¬ same_graph equation_I equation_III) :=
sorry

end NUMINAMATH_CALUDE_equations_same_graph_l914_91405


namespace NUMINAMATH_CALUDE_ordering_abc_l914_91446

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → 
  b = Real.cos (1/4) → 
  c = 4 * Real.sin (1/4) → 
  c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l914_91446


namespace NUMINAMATH_CALUDE_remainder_twelve_remainder_107_is_least_unique_divisor_l914_91431

def least_number : ℕ := 540

-- 540 leaves a remainder of 5 when divided by 12
theorem remainder_twelve : least_number % 12 = 5 := by sorry

-- 107 leaves a remainder of 5 when 540 is divided by it
theorem remainder_107 : least_number % 107 = 5 := by sorry

-- 540 is the least number that leaves a remainder of 5 when divided by some numbers
theorem is_least (n : ℕ) : n < least_number → ¬(∃ m : ℕ, m > 1 ∧ n % m = 5) := by sorry

-- 107 is the only number (other than 12) that leaves a remainder of 5 when 540 is divided by it
theorem unique_divisor (n : ℕ) : n ≠ 12 → n ≠ 107 → least_number % n ≠ 5 := by sorry

end NUMINAMATH_CALUDE_remainder_twelve_remainder_107_is_least_unique_divisor_l914_91431


namespace NUMINAMATH_CALUDE_right_triangle_legs_l914_91469

/-- A right-angled triangle with a point inside it -/
structure RightTriangleWithPoint where
  /-- Length of one leg of the triangle -/
  x : ℝ
  /-- Length of the other leg of the triangle -/
  y : ℝ
  /-- Distance from the point to one side -/
  d1 : ℝ
  /-- Distance from the point to the other side -/
  d2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : x > 0 ∧ y > 0
  /-- The point is inside the triangle -/
  point_inside : d1 > 0 ∧ d2 > 0 ∧ d1 < y ∧ d2 < x
  /-- The area of the triangle is 100 -/
  area : x * y / 2 = 100
  /-- The distances from the point to the sides are 4 and 8 -/
  distances : d1 = 4 ∧ d2 = 8

/-- The theorem stating the possible leg lengths of the triangle -/
theorem right_triangle_legs (t : RightTriangleWithPoint) :
  (t.x = 40 ∧ t.y = 5) ∨ (t.x = 10 ∧ t.y = 20) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l914_91469


namespace NUMINAMATH_CALUDE_terry_current_age_l914_91406

/-- Terry's current age in years -/
def terry_age : ℕ := sorry

/-- Nora's current age in years -/
def nora_age : ℕ := 10

/-- The number of years in the future when Terry's age will be 4 times Nora's current age -/
def years_future : ℕ := 10

theorem terry_current_age : 
  terry_age = 30 :=
by
  have h1 : terry_age + years_future = 4 * nora_age := sorry
  sorry

end NUMINAMATH_CALUDE_terry_current_age_l914_91406


namespace NUMINAMATH_CALUDE_volume_of_rotated_region_l914_91467

/-- The volume of the solid formed by rotating the region bounded by y = 2x - x^2 and y = 2x^2 - 4x around the x-axis. -/
theorem volume_of_rotated_region : ∃ V : ℝ,
  (∀ x y : ℝ, (y = 2*x - x^2 ∨ y = 2*x^2 - 4*x) → 
    V = π * ∫ x in (0)..(2), ((2*x^2 - 4*x)^2 - (2*x - x^2)^2)) ∧
  V = (16 * π) / 5 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_region_l914_91467


namespace NUMINAMATH_CALUDE_marble_group_size_l914_91438

theorem marble_group_size : 
  ∀ (x : ℕ), 
  (144 / x : ℚ) = (144 / (x + 2) : ℚ) + 1 → 
  x = 16 := by
sorry

end NUMINAMATH_CALUDE_marble_group_size_l914_91438


namespace NUMINAMATH_CALUDE_exponent_multiplication_l914_91411

theorem exponent_multiplication (a : ℝ) : 2 * a^2 * a^4 = 2 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l914_91411


namespace NUMINAMATH_CALUDE_number_wall_theorem_l914_91416

/-- Represents a number wall with 5 numbers in the bottom row -/
structure NumberWall :=
  (bottom : Fin 5 → ℕ)
  (second_row : Fin 4 → ℕ)
  (third_row : Fin 3 → ℕ)
  (fourth_row : Fin 2 → ℕ)
  (top : ℕ)

/-- The rule for constructing a number wall -/
def valid_wall (w : NumberWall) : Prop :=
  (∀ i : Fin 4, w.second_row i = w.bottom i + w.bottom (i + 1)) ∧
  (∀ i : Fin 3, w.third_row i = w.second_row i + w.second_row (i + 1)) ∧
  (∀ i : Fin 2, w.fourth_row i = w.third_row i + w.third_row (i + 1)) ∧
  (w.top = w.fourth_row 0 + w.fourth_row 1)

theorem number_wall_theorem (w : NumberWall) (h : valid_wall w) :
  w.bottom 1 = 5 ∧ w.bottom 2 = 9 ∧ w.bottom 3 = 7 ∧ w.bottom 4 = 12 ∧
  w.top = 54 ∧ w.third_row 1 = 34 →
  w.bottom 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_theorem_l914_91416


namespace NUMINAMATH_CALUDE_complex_equation_solution_l914_91486

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + (b : ℂ) * Complex.I = (1 - Complex.I) / (2 * Complex.I) → 
  a = -1/2 ∧ b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l914_91486


namespace NUMINAMATH_CALUDE_car_dealer_problem_l914_91498

theorem car_dealer_problem (X Y : ℚ) (h1 : X > 0) (h2 : Y > 0) : 
  1.54 * (X + Y) = 1.4 * X + 1.6 * Y →
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ X * b = Y * a ∧ Nat.gcd a b = 1 ∧ 11 * a + 13 * b = 124 :=
by sorry

end NUMINAMATH_CALUDE_car_dealer_problem_l914_91498


namespace NUMINAMATH_CALUDE_quadratic_roots_constraint_l914_91403

/-- A quadratic function f(x) = x^2 + 2bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

/-- The equation f(x) + x + b = 0 -/
def g (b c x : ℝ) : ℝ := f b c x + x + b

theorem quadratic_roots_constraint (b c : ℝ) :
  f b c 1 = 0 ∧
  (∃ x₁ x₂, x₁ ∈ Set.Ioo (-3) (-2) ∧ x₂ ∈ Set.Ioo 0 1 ∧
    g b c x₁ = 0 ∧ g b c x₂ = 0) →
  b ∈ Set.Ioo (-5/2) (-1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_constraint_l914_91403


namespace NUMINAMATH_CALUDE_distinct_ratios_theorem_l914_91410

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then
    1/2 - |x - 3/2|
  else
    Real.exp (x - 2) * (-x^2 + 8*x - 12)

theorem distinct_ratios_theorem (n : ℕ) (x : Fin n → ℝ) :
  n ≥ 2 →
  (∀ i : Fin n, x i > 1) →
  (∀ i j : Fin n, i ≠ j → x i ≠ x j) →
  (∀ i j : Fin n, f (x i) / (x i) = f (x j) / (x j)) →
  n ∈ ({2, 3, 4} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_distinct_ratios_theorem_l914_91410


namespace NUMINAMATH_CALUDE_triangle_base_length_l914_91443

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) :
  height = 10 →
  area = 50 →
  area = (base * height) / 2 →
  base = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l914_91443


namespace NUMINAMATH_CALUDE_abs_diff_sqrt_two_l914_91445

theorem abs_diff_sqrt_two : ∀ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 = 2 → |3 - x| - |x - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_sqrt_two_l914_91445


namespace NUMINAMATH_CALUDE_unique_functional_equation_solution_l914_91409

theorem unique_functional_equation_solution :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_solution_l914_91409


namespace NUMINAMATH_CALUDE_anns_age_l914_91435

theorem anns_age (a b : ℕ) : 
  a + b = 50 → 
  b = (2 * a / 3 : ℚ) + 2 * (a - b) → 
  a = 26 := by
sorry

end NUMINAMATH_CALUDE_anns_age_l914_91435


namespace NUMINAMATH_CALUDE_problem_solution_l914_91440

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Define the range of k
def k_range (k : ℝ) : Prop := (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)

-- Theorem statement
theorem problem_solution (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k_range k := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l914_91440


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l914_91434

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l914_91434


namespace NUMINAMATH_CALUDE_inverse_direct_variation_l914_91488

/-- Given that a²c varies inversely with b³ and c varies directly as b², 
    prove that a² = 25/128 when b = 4, given initial conditions. -/
theorem inverse_direct_variation (a b c : ℝ) (k k' : ℝ) : 
  (∀ a b c, a^2 * c * b^3 = k) →  -- a²c varies inversely with b³
  (∀ b c, c = k' * b^2) →         -- c varies directly as b²
  (5^2 * 12 * 2^3 = k) →          -- initial condition for k
  (12 = k' * 2^2) →               -- initial condition for k'
  (∀ a, a^2 * (k' * 4^2) * 4^3 = k) →  -- condition for b = 4
  (∃ a, a^2 = 25 / 128) :=
by sorry

end NUMINAMATH_CALUDE_inverse_direct_variation_l914_91488


namespace NUMINAMATH_CALUDE_john_ben_difference_l914_91447

/-- Represents the marble transfer problem --/
structure MarbleTransfer where
  ben_initial : ℝ
  john_initial : ℝ
  lisa_initial : ℝ
  max_initial : ℝ
  ben_to_john_percent : ℝ
  ben_to_lisa_percent : ℝ
  john_to_max_percent : ℝ
  lisa_to_john_percent : ℝ

/-- Calculates the final marble counts after all transfers --/
def finalCounts (mt : MarbleTransfer) : ℝ × ℝ × ℝ × ℝ :=
  let ben_to_john := mt.ben_initial * mt.ben_to_john_percent
  let ben_to_lisa := mt.ben_initial * mt.ben_to_lisa_percent
  let ben_final := mt.ben_initial - ben_to_john - ben_to_lisa
  let john_from_ben := ben_to_john
  let john_to_max := john_from_ben * mt.john_to_max_percent
  let lisa_with_ben := mt.lisa_initial + ben_to_lisa
  let lisa_to_john := mt.lisa_initial * mt.lisa_to_john_percent + ben_to_lisa
  let john_final := mt.john_initial + john_from_ben - john_to_max + lisa_to_john
  let max_final := mt.max_initial + john_to_max
  let lisa_final := lisa_with_ben - lisa_to_john
  (ben_final, john_final, lisa_final, max_final)

/-- Theorem stating the difference in marbles between John and Ben after transfers --/
theorem john_ben_difference (mt : MarbleTransfer) 
  (h1 : mt.ben_initial = 18)
  (h2 : mt.john_initial = 17)
  (h3 : mt.lisa_initial = 12)
  (h4 : mt.max_initial = 9)
  (h5 : mt.ben_to_john_percent = 0.5)
  (h6 : mt.ben_to_lisa_percent = 0.25)
  (h7 : mt.john_to_max_percent = 0.65)
  (h8 : mt.lisa_to_john_percent = 0.2) :
  (finalCounts mt).2.1 - (finalCounts mt).1 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_john_ben_difference_l914_91447


namespace NUMINAMATH_CALUDE_min_value_a_l914_91459

theorem min_value_a (a b : ℕ) (h : 1998 * a = b^4) : 1215672 ≤ a := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l914_91459


namespace NUMINAMATH_CALUDE_square_one_on_top_l914_91413

/-- Represents the possible positions of a square after folding and rotation. -/
inductive Position
  | TopLeft | TopMiddle | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomMiddle | BottomRight

/-- Represents the state of the grid after each operation. -/
structure GridState :=
  (positions : Fin 9 → Position)

/-- Folds the right half over the left half. -/
def foldRightOverLeft (state : GridState) : GridState := sorry

/-- Folds the top half over the bottom half. -/
def foldTopOverBottom (state : GridState) : GridState := sorry

/-- Folds the left half over the right half. -/
def foldLeftOverRight (state : GridState) : GridState := sorry

/-- Rotates the entire grid 90 degrees clockwise. -/
def rotateClockwise (state : GridState) : GridState := sorry

/-- The initial state of the grid. -/
def initialState : GridState :=
  { positions := λ i => match i with
    | 0 => Position.TopLeft
    | 1 => Position.TopMiddle
    | 2 => Position.TopRight
    | 3 => Position.MiddleLeft
    | 4 => Position.Center
    | 5 => Position.MiddleRight
    | 6 => Position.BottomLeft
    | 7 => Position.BottomMiddle
    | 8 => Position.BottomRight }

theorem square_one_on_top :
  (rotateClockwise (foldLeftOverRight (foldTopOverBottom (foldRightOverLeft initialState)))).positions 0 = Position.TopLeft := by
  sorry

end NUMINAMATH_CALUDE_square_one_on_top_l914_91413


namespace NUMINAMATH_CALUDE_tan_theta_negative_three_l914_91477

theorem tan_theta_negative_three (θ : Real) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_negative_three_l914_91477


namespace NUMINAMATH_CALUDE_f_difference_bound_l914_91487

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x + 15

-- State the theorem
theorem f_difference_bound (x a : ℝ) (h : |x - a| < 1) : 
  |f x - f a| < 2 * (|a| + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_bound_l914_91487


namespace NUMINAMATH_CALUDE_three_number_problem_l914_91404

theorem three_number_problem (x y z : ℚ) : 
  x + (1/3) * z = y ∧ 
  y + (1/3) * x = z ∧ 
  z - x = 10 → 
  x = 10 ∧ y = 50/3 ∧ z = 20 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l914_91404


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l914_91479

theorem system_of_equations_solution :
  ∃! (x y : ℚ), (5 * x - 3 * y = -7) ∧ (2 * x + 7 * y = -26) ∧ 
  (x = -127 / 41) ∧ (y = -116 / 41) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l914_91479


namespace NUMINAMATH_CALUDE_teddy_pillow_count_l914_91448

/-- The amount of fluffy foam material used for each pillow in pounds -/
def material_per_pillow : ℝ := 5 - 3

/-- The amount of fluffy foam material Teddy has in tons -/
def total_material_tons : ℝ := 3

/-- The number of pounds in a ton -/
def pounds_per_ton : ℝ := 2000

/-- The theorem stating how many pillows Teddy can make -/
theorem teddy_pillow_count : 
  (total_material_tons * pounds_per_ton) / material_per_pillow = 3000 := by
  sorry

end NUMINAMATH_CALUDE_teddy_pillow_count_l914_91448


namespace NUMINAMATH_CALUDE_equation_solutions_l914_91494

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3 * x - 2|

-- Define the domain of f
def domain_f (x : ℝ) : Prop := x ≠ 3 ∧ x ≠ 0

-- Define the equation to be solved
def equation (x a : ℝ) : Prop := |3 * x - 2| = |x + a|

-- Theorem statement
theorem equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧
  (∀ x, domain_f x → equation x a₁) ∧
  (∀ x, domain_f x → equation x a₂) ∧
  (∀ a, (∀ x, domain_f x → equation x a) → (a = a₁ ∨ a = a₂)) ∧
  a₁ = -2/3 ∧ a₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l914_91494


namespace NUMINAMATH_CALUDE_tile_ratio_l914_91433

/-- Given a square pattern with black and white tiles and a white border added, 
    calculate the ratio of black to white tiles. -/
theorem tile_ratio (initial_black initial_white border_width : ℕ) 
  (h1 : initial_black = 5)
  (h2 : initial_white = 20)
  (h3 : border_width = 1)
  (h4 : initial_black + initial_white = (initial_black + initial_white).sqrt ^ 2) :
  let total_side := (initial_black + initial_white).sqrt + 2 * border_width
  let total_tiles := total_side ^ 2
  let added_white := total_tiles - (initial_black + initial_white)
  let final_white := initial_white + added_white
  (initial_black : ℚ) / final_white = 5 / 44 := by sorry

end NUMINAMATH_CALUDE_tile_ratio_l914_91433


namespace NUMINAMATH_CALUDE_gcd_459_357_l914_91484

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l914_91484


namespace NUMINAMATH_CALUDE_goat_feed_theorem_l914_91414

/-- Represents the number of days feed lasts for a given number of goats -/
def feed_duration (num_goats : ℕ) (days : ℕ) : Prop := True

theorem goat_feed_theorem (D : ℕ) :
  feed_duration 20 D →
  feed_duration 30 (D - 3) →
  feed_duration 15 (D + D) :=
by
  sorry

#check goat_feed_theorem

end NUMINAMATH_CALUDE_goat_feed_theorem_l914_91414


namespace NUMINAMATH_CALUDE_tent_setup_plans_l914_91493

theorem tent_setup_plans : 
  let total_students : ℕ := 50
  let valid_setup (x y : ℕ) := 3 * x + 2 * y = total_students
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => valid_setup p.1 p.2) (Finset.product (Finset.range (total_students + 1)) (Finset.range (total_students + 1)))).card ∧ n = 8
  := by sorry

end NUMINAMATH_CALUDE_tent_setup_plans_l914_91493


namespace NUMINAMATH_CALUDE_raghu_investment_l914_91455

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6069 →
  raghu = 2100 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l914_91455


namespace NUMINAMATH_CALUDE_complex_product_theorem_l914_91465

theorem complex_product_theorem (α₁ α₂ α₃ : ℝ) : 
  let z₁ : ℂ := Complex.exp (I * α₁)
  let z₂ : ℂ := Complex.exp (I * α₂)
  let z₃ : ℂ := Complex.exp (I * α₃)
  z₁ * z₂ = Complex.exp (I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (I * (α₁ + α₂ + α₃)) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l914_91465


namespace NUMINAMATH_CALUDE_smallest_angle_EBC_l914_91497

theorem smallest_angle_EBC (ABC ABD DBE : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) (h3 : DBE = 10) : 
  ∃ (EBC : ℝ), EBC = 20 ∧ ∀ (x : ℝ), x ≥ 20 → x ≥ EBC := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_EBC_l914_91497


namespace NUMINAMATH_CALUDE_tax_calculation_l914_91401

theorem tax_calculation (total_earnings deductions tax_paid : ℝ) 
  (h1 : total_earnings = 100000)
  (h2 : deductions = 30000)
  (h3 : tax_paid = 12000) : 
  ∃ (taxed_at_10_percent : ℝ),
    taxed_at_10_percent = 20000 ∧
    tax_paid = 0.1 * taxed_at_10_percent + 
               0.2 * (total_earnings - deductions - taxed_at_10_percent) :=
by sorry

end NUMINAMATH_CALUDE_tax_calculation_l914_91401


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l914_91422

theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ 
  (∃ (k : ℕ), (5 * n) / 3 = k + 2/3) ∧
  (∃ (k : ℕ), (7 * n) / 5 = k + 2/5) ∧
  (∃ (k : ℕ), (9 * n) / 7 = k + 2/7) ∧
  (∃ (k : ℕ), (11 * n) / 9 = k + 2/9) ∧
  (∀ (m : ℕ), m > 1 → 
    ((∃ (k : ℕ), (5 * m) / 3 = k + 2/3) ∧
     (∃ (k : ℕ), (7 * m) / 5 = k + 2/5) ∧
     (∃ (k : ℕ), (9 * m) / 7 = k + 2/7) ∧
     (∃ (k : ℕ), (11 * m) / 9 = k + 2/9)) → m ≥ n) ∧
  n = 316 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_conditions_l914_91422


namespace NUMINAMATH_CALUDE_f_satisfies_all_points_l914_91441

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := -50 * x + 200

/-- The set of points from the given table --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 150), (2, 100), (3, 50), (4, 0)]

/-- Theorem stating that the function f satisfies all points in the given table --/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_all_points_l914_91441


namespace NUMINAMATH_CALUDE_diminishing_allocation_problem_l914_91408

/-- Represents a diminishing allocation sequence -/
def DiminishingAllocation (a : ℝ) (b : ℝ) : ℕ → ℝ
  | 0 => b
  | n + 1 => DiminishingAllocation a b n * (1 - a)

theorem diminishing_allocation_problem (a b : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : 0 < b) 
  (h3 : DiminishingAllocation a b 2 = 80) 
  (h4 : DiminishingAllocation a b 1 + DiminishingAllocation a b 3 = 164) :
  a = 0.2 ∧ 
  DiminishingAllocation a b 0 + DiminishingAllocation a b 1 + 
  DiminishingAllocation a b 2 + DiminishingAllocation a b 3 = 369 := by
  sorry


end NUMINAMATH_CALUDE_diminishing_allocation_problem_l914_91408


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l914_91468

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 5*a + 7 = 0) → 
  (b^3 - 2*b^2 + 5*b + 7 = 0) → 
  (c^3 - 2*c^2 + 5*c + 7 = 0) → 
  a^2 + b^2 + c^2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l914_91468


namespace NUMINAMATH_CALUDE_stationery_prices_l914_91462

theorem stationery_prices (pen_price notebook_price : ℝ) : 
  pen_price + notebook_price = 3.6 →
  pen_price + 4 * notebook_price = 10.5 →
  pen_price = 1.3 ∧ notebook_price = 2.3 := by
sorry

end NUMINAMATH_CALUDE_stationery_prices_l914_91462


namespace NUMINAMATH_CALUDE_power_of_128_l914_91464

theorem power_of_128 : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  have h1 : (128 : ℝ) = 2^7 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_128_l914_91464
