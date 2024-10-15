import Mathlib

namespace NUMINAMATH_CALUDE_largest_integer_fraction_l523_52330

theorem largest_integer_fraction (n : ℤ) : (n / 11 : ℚ) < 2/3 ↔ n ≤ 7 :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_fraction_l523_52330


namespace NUMINAMATH_CALUDE_davids_math_marks_l523_52383

def english_marks : ℕ := 76
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / num_subjects = average_marks ∧
    math_marks = 65 :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l523_52383


namespace NUMINAMATH_CALUDE_inequality_solution_range_of_a_l523_52394

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x - 3|

-- Theorem for the solution of the inequality
theorem inequality_solution :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -1/4 ≤ x ∧ x ≤ 9/4} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∀ x, ∃ y, Real.log y = f x + a} = {a : ℝ | a > -2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_of_a_l523_52394


namespace NUMINAMATH_CALUDE_tile_coverage_proof_l523_52370

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the ceiling of a fraction represented as a numerator and denominator -/
def ceilingDiv (n d : ℕ) : ℕ := (n + d - 1) / d

theorem tile_coverage_proof (tile : Dimensions) (room : Dimensions) : 
  tile.length = 2 → 
  tile.width = 5 → 
  room.length = feetToInches 3 → 
  room.width = feetToInches 8 → 
  ceilingDiv (area room) (area tile) = 346 := by
  sorry

end NUMINAMATH_CALUDE_tile_coverage_proof_l523_52370


namespace NUMINAMATH_CALUDE_triangle_f_sign_l523_52369

/-- Triangle ABC with sides a ≤ b ≤ c, circumradius R, and inradius r -/
structure Triangle where
  a : Real
  b : Real
  c : Real
  R : Real
  r : Real
  h_sides : a ≤ b ∧ b ≤ c
  h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0

/-- The function f defined for the triangle -/
def f (t : Triangle) : Real := t.a + t.b - 2 * t.R - 2 * t.r

/-- Angle C of the triangle -/
noncomputable def angle_C (t : Triangle) : Real := Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

theorem triangle_f_sign (t : Triangle) :
  (f t > 0 ↔ angle_C t < Real.pi / 2) ∧
  (f t = 0 ↔ angle_C t = Real.pi / 2) ∧
  (f t < 0 ↔ angle_C t > Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_triangle_f_sign_l523_52369


namespace NUMINAMATH_CALUDE_abc_sum_mod_five_l523_52325

theorem abc_sum_mod_five (a b c : ℕ) : 
  a < 5 → b < 5 → c < 5 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 5 = 1 →
  (4 * c) % 5 = 3 →
  (3 * b) % 5 = (2 + b) % 5 →
  (a + b + c) % 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_five_l523_52325


namespace NUMINAMATH_CALUDE_value_of_2a_plus_b_l523_52340

theorem value_of_2a_plus_b (a b : ℝ) (h : |a + 2| + (b - 5)^2 = 0) : 2*a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_2a_plus_b_l523_52340


namespace NUMINAMATH_CALUDE_parallelogram_height_l523_52368

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) (h1 : area = 704) (h2 : base = 32) 
  (h3 : area = base * height) : height = 22 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l523_52368


namespace NUMINAMATH_CALUDE_correct_freshman_count_l523_52353

/-- The number of students the college needs to admit into the freshman class each year
    to maintain a total enrollment of 3400 students, given specific dropout rates for each class. -/
def requiredFreshmen : ℕ :=
  let totalEnrollment : ℕ := 3400
  let freshmanDropoutRate : ℚ := 1/3
  let sophomoreDropouts : ℕ := 40
  let juniorDropoutRate : ℚ := 1/10
  5727

/-- Theorem stating that the required number of freshmen is 5727 -/
theorem correct_freshman_count :
  requiredFreshmen = 5727 :=
by sorry

end NUMINAMATH_CALUDE_correct_freshman_count_l523_52353


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l523_52387

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Blue

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing 2 white balls and 2 blue balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.White} + 2 • {BallColor.Blue}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: All drawn balls are blue -/
def allBlue (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Blue ∧ outcome.second = BallColor.Blue

/-- The probability of an event occurring when drawing two balls from the bag -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ := sorry

/-- Theorem: "At least one white ball" and "All are blue balls" are mutually exclusive -/
theorem mutually_exclusive_events :
  probability (λ outcome => atLeastOneWhite outcome ∧ allBlue outcome) = 0 :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l523_52387


namespace NUMINAMATH_CALUDE_quadratic_equations_sum_l523_52354

theorem quadratic_equations_sum (x y : ℝ) : 
  9 * x^2 - 36 * x - 81 = 0 → 
  y^2 + 6 * y + 9 = 0 → 
  (x + y = -1 + Real.sqrt 13) ∨ (x + y = -1 - Real.sqrt 13) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equations_sum_l523_52354


namespace NUMINAMATH_CALUDE_remainder_product_l523_52334

theorem remainder_product (n : ℕ) (d : ℕ) (m : ℕ) (h : d ≠ 0) :
  (n % d) * m = 33 ↔ n = 2345678 ∧ d = 128 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_l523_52334


namespace NUMINAMATH_CALUDE_miranda_pillows_l523_52388

/-- Calculates the number of pillows Miranda can stuff given the following conditions:
  * Each pillow needs 2 pounds of feathers
  * 1 pound of goose feathers is approximately 300 feathers
  * Miranda's goose has approximately 3600 feathers
-/
def pillows_from_goose (feathers_per_pillow : ℕ) (feathers_per_pound : ℕ) (goose_feathers : ℕ) : ℕ :=
  (goose_feathers / feathers_per_pound) / feathers_per_pillow

/-- Proves that Miranda can stuff 6 pillows given the conditions -/
theorem miranda_pillows : 
  pillows_from_goose 2 300 3600 = 6 := by
  sorry

end NUMINAMATH_CALUDE_miranda_pillows_l523_52388


namespace NUMINAMATH_CALUDE_graduating_class_size_l523_52360

theorem graduating_class_size 
  (geometry_count : ℕ) 
  (biology_count : ℕ) 
  (overlap_difference : ℕ) 
  (h1 : geometry_count = 144) 
  (h2 : biology_count = 119) 
  (h3 : overlap_difference = 88) :
  geometry_count + biology_count - (biology_count - overlap_difference) = 232 := by
  sorry

end NUMINAMATH_CALUDE_graduating_class_size_l523_52360


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l523_52355

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 1)^8 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3 + 
    a₄*(x-2)^4 + a₅*(x-2)^5 + a₆*(x-2)^6 + a₇*(x-2)^7 + a₈*(x-2)^8 + a₉*(x-2)^9 + a₁₀*(x-2)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 2555 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l523_52355


namespace NUMINAMATH_CALUDE_max_ski_trips_l523_52332

/-- Represents the time in minutes for a single trip up and down the mountain -/
def trip_time : ℕ := 15 + 5

/-- Represents the total available time in minutes -/
def total_time : ℕ := 2 * 60

/-- Theorem stating the maximum number of times a person can ski down the mountain in 2 hours -/
theorem max_ski_trips : (total_time / trip_time : ℕ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_ski_trips_l523_52332


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l523_52352

-- Problem 1
theorem problem_1 (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a > 0) (h4 : b > 0) :
  a + b = 8 := by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) (h : |a - 2| + |b - 3| + |c - 4| = 0) :
  a + b + c = 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l523_52352


namespace NUMINAMATH_CALUDE_initial_girls_count_l523_52300

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 3 / 10 →
  ((initial_girls : ℚ) - 3) / total = 1 / 5 →
  initial_girls = 9 :=
sorry

end NUMINAMATH_CALUDE_initial_girls_count_l523_52300


namespace NUMINAMATH_CALUDE_yoongi_stack_higher_l523_52345

/-- The height of Box A in centimeters -/
def box_a_height : ℝ := 3

/-- The height of Box B in centimeters -/
def box_b_height : ℝ := 3.5

/-- The number of Box A stacked by Taehyung -/
def taehyung_boxes : ℕ := 16

/-- The number of Box B stacked by Yoongi -/
def yoongi_boxes : ℕ := 14

/-- The total height of Taehyung's stack in centimeters -/
def taehyung_stack_height : ℝ := box_a_height * taehyung_boxes

/-- The total height of Yoongi's stack in centimeters -/
def yoongi_stack_height : ℝ := box_b_height * yoongi_boxes

theorem yoongi_stack_higher :
  yoongi_stack_height > taehyung_stack_height ∧
  yoongi_stack_height - taehyung_stack_height = 1 :=
by sorry

end NUMINAMATH_CALUDE_yoongi_stack_higher_l523_52345


namespace NUMINAMATH_CALUDE_power_of_power_l523_52317

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l523_52317


namespace NUMINAMATH_CALUDE_simplify_and_abs_l523_52341

theorem simplify_and_abs (a : ℝ) (h : a = -2) : 
  |12 * a^5 / (72 * a^3)| = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_abs_l523_52341


namespace NUMINAMATH_CALUDE_power_difference_l523_52397

theorem power_difference (a : ℝ) (m n : ℤ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m-n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l523_52397


namespace NUMINAMATH_CALUDE_ratio_proof_l523_52313

theorem ratio_proof (N : ℝ) 
  (h1 : (1 / 1) * (1 / 3) * (2 / 5) * N = 20)
  (h2 : 0.4 * N = 240) :
  20 / ((1 / 3) * (2 / 5) * N) = 2 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_proof_l523_52313


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l523_52338

theorem y_in_terms_of_x (x y : ℝ) (h : y - 2*x = 6) : y = 2*x + 6 := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l523_52338


namespace NUMINAMATH_CALUDE_octal_digit_reversal_difference_l523_52349

theorem octal_digit_reversal_difference (A B : Nat) : 
  A ≠ B → 
  A < 8 → 
  B < 8 → 
  ∃ k : Int, (8 * A + B) - (8 * B + A) = 7 * k ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_octal_digit_reversal_difference_l523_52349


namespace NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l523_52385

theorem abs_sum_equals_sum_abs_necessary_not_sufficient (a b : ℝ) :
  (∀ a b : ℝ, a * b > 0 → |a + b| = |a| + |b|) ∧
  (∃ a b : ℝ, |a + b| = |a| + |b| ∧ a * b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_equals_sum_abs_necessary_not_sufficient_l523_52385


namespace NUMINAMATH_CALUDE_road_trip_cost_l523_52316

theorem road_trip_cost (initial_friends : ℕ) (additional_friends : ℕ) (cost_decrease : ℚ) :
  initial_friends = 5 →
  additional_friends = 3 →
  cost_decrease = 15 →
  ∃ total_cost : ℚ,
    total_cost / initial_friends - total_cost / (initial_friends + additional_friends) = cost_decrease ∧
    total_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_cost_l523_52316


namespace NUMINAMATH_CALUDE_trigonometric_identity_l523_52365

theorem trigonometric_identity : 
  2 * Real.sin (390 * π / 180) - Real.tan (-45 * π / 180) + 5 * Real.cos (360 * π / 180) = 7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l523_52365


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l523_52329

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l523_52329


namespace NUMINAMATH_CALUDE_expression_evaluation_l523_52348

theorem expression_evaluation (x z : ℝ) (hz : z ≠ 0) (hx : x = 1 / z^2) :
  (x + 1/x) * (z^2 - 1/z^2) = z^4 - 1/z^4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l523_52348


namespace NUMINAMATH_CALUDE_donnas_card_shop_wage_l523_52358

/-- Calculates Donna's hourly wage at the card shop based on her weekly earnings --/
theorem donnas_card_shop_wage (
  dog_walking_hours : ℕ)
  (dog_walking_rate : ℚ)
  (dog_walking_days : ℕ)
  (card_shop_hours : ℕ)
  (card_shop_days : ℕ)
  (babysitting_hours : ℕ)
  (babysitting_rate : ℚ)
  (total_earnings : ℚ)
  (h1 : dog_walking_hours = 2)
  (h2 : dog_walking_rate = 10)
  (h3 : dog_walking_days = 5)
  (h4 : card_shop_hours = 2)
  (h5 : card_shop_days = 5)
  (h6 : babysitting_hours = 4)
  (h7 : babysitting_rate = 10)
  (h8 : total_earnings = 305) :
  (total_earnings - (dog_walking_hours * dog_walking_rate * dog_walking_days + babysitting_hours * babysitting_rate)) / (card_shop_hours * card_shop_days) = 33/2 := by
  sorry

#eval (33 : ℚ) / 2

end NUMINAMATH_CALUDE_donnas_card_shop_wage_l523_52358


namespace NUMINAMATH_CALUDE_factorial_division_l523_52343

theorem factorial_division : 
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 :=
by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l523_52343


namespace NUMINAMATH_CALUDE_solution_to_system_l523_52393

theorem solution_to_system : ∃ (x y : ℚ), 
  (7 * x - 50 * y = 3) ∧ (3 * y - x = 5) ∧ 
  (x = -259/29) ∧ (y = -38/29) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_l523_52393


namespace NUMINAMATH_CALUDE_product_sum_difference_l523_52398

theorem product_sum_difference (a b : ℤ) (h1 : b = 8) (h2 : b - a = 3) :
  a * b - 2 * (a + b) = 14 := by sorry

end NUMINAMATH_CALUDE_product_sum_difference_l523_52398


namespace NUMINAMATH_CALUDE_line_proof_l523_52328

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x - y + 2 = 0
def line3 (x y : ℝ) : Prop := x + y = 0
def line4 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem line_proof :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    line4 x y ∧
    perpendicular (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_line_proof_l523_52328


namespace NUMINAMATH_CALUDE_vacation_animals_l523_52376

/-- The total number of animals bought on the last vacation --/
def total_animals (rainbowfish clowns tetras guppies angelfish cichlids : ℕ) : ℕ :=
  rainbowfish + clowns + tetras + guppies + angelfish + cichlids

/-- Theorem stating the total number of animals bought on the last vacation --/
theorem vacation_animals :
  ∃ (rainbowfish clowns tetras guppies angelfish cichlids : ℕ),
    rainbowfish = 40 ∧
    cichlids = rainbowfish / 2 ∧
    angelfish = cichlids + 10 ∧
    guppies = 3 * angelfish ∧
    clowns = 2 * guppies ∧
    tetras = 5 * clowns ∧
    total_animals rainbowfish clowns tetras guppies angelfish cichlids = 1260 := by
  sorry


end NUMINAMATH_CALUDE_vacation_animals_l523_52376


namespace NUMINAMATH_CALUDE_max_puns_purchase_l523_52344

/-- Represents the cost of each item --/
structure ItemCosts where
  pin : ℕ
  pon : ℕ
  pun : ℕ

/-- Represents the quantity of each item purchased --/
structure Purchase where
  pins : ℕ
  pons : ℕ
  puns : ℕ

/-- Calculates the total cost of a purchase --/
def totalCost (costs : ItemCosts) (purchase : Purchase) : ℕ :=
  costs.pin * purchase.pins + costs.pon * purchase.pons + costs.pun * purchase.puns

/-- Checks if a purchase is valid (at least one of each item) --/
def isValidPurchase (purchase : Purchase) : Prop :=
  purchase.pins ≥ 1 ∧ purchase.pons ≥ 1 ∧ purchase.puns ≥ 1

/-- The main theorem statement --/
theorem max_puns_purchase (costs : ItemCosts) (budget : ℕ) : 
  costs.pin = 3 → costs.pon = 4 → costs.pun = 9 → budget = 108 →
  ∃ (max_puns : ℕ), 
    (∃ (p : Purchase), isValidPurchase p ∧ totalCost costs p = budget ∧ p.puns = max_puns) ∧
    (∀ (p : Purchase), isValidPurchase p → totalCost costs p = budget → p.puns ≤ max_puns) ∧
    max_puns = 10 :=
sorry

end NUMINAMATH_CALUDE_max_puns_purchase_l523_52344


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l523_52309

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4)) ∧ 
  (∃ a b : ℝ, (a + b > 4 ∧ a * b > 4) ∧ ¬(a > 2 ∧ b > 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l523_52309


namespace NUMINAMATH_CALUDE_stop_after_seventh_shot_probability_value_l523_52314

/-- The maximum number of shots allowed -/
def max_shots : ℕ := 10

/-- The probability of making a shot for student A -/
def shot_probability : ℚ := 2/3

/-- Calculate the score based on the shot number when the student stops -/
def score (n : ℕ) : ℕ := 12 - n

/-- The probability of the specific sequence of shots leading to stopping after the 7th shot -/
def stop_after_seventh_shot_probability : ℚ :=
  (1 - shot_probability) * shot_probability * (1 - shot_probability) *
  1 * (1 - shot_probability) * shot_probability * shot_probability

theorem stop_after_seventh_shot_probability_value :
  stop_after_seventh_shot_probability = 8/729 :=
sorry

end NUMINAMATH_CALUDE_stop_after_seventh_shot_probability_value_l523_52314


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l523_52311

theorem imaginary_part_of_complex_fraction : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l523_52311


namespace NUMINAMATH_CALUDE_combined_sale_price_l523_52337

/-- Calculate the sale price given the purchase cost and profit percentage -/
def calculateSalePrice (purchaseCost : ℚ) (profitPercentage : ℚ) : ℚ :=
  purchaseCost * (1 + profitPercentage)

/-- The problem statement -/
theorem combined_sale_price :
  let itemA_cost : ℚ := 650
  let itemB_cost : ℚ := 350
  let itemC_cost : ℚ := 400
  let itemA_profit : ℚ := 0.40
  let itemB_profit : ℚ := 0.25
  let itemC_profit : ℚ := 0.30
  let itemA_sale := calculateSalePrice itemA_cost itemA_profit
  let itemB_sale := calculateSalePrice itemB_cost itemB_profit
  let itemC_sale := calculateSalePrice itemC_cost itemC_profit
  itemA_sale + itemB_sale + itemC_sale = 1867.50 := by
  sorry

end NUMINAMATH_CALUDE_combined_sale_price_l523_52337


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l523_52351

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.isPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation :
  ∃ (l : Line),
    l.passesThrough ⟨1, -2⟩ ∧
    l.isPerpendicular ⟨2, 3, -1⟩ ∧
    l = ⟨3, -2, -7⟩ := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l523_52351


namespace NUMINAMATH_CALUDE_seeds_per_row_l523_52307

/-- Given a garden with potatoes planted in rows, this theorem proves
    the number of seeds in each row when the total number of potatoes
    and the number of rows are known. -/
theorem seeds_per_row (total_potatoes : ℕ) (num_rows : ℕ) 
    (h1 : total_potatoes = 54) 
    (h2 : num_rows = 6) 
    (h3 : total_potatoes % num_rows = 0) : 
  total_potatoes / num_rows = 9 := by
  sorry

end NUMINAMATH_CALUDE_seeds_per_row_l523_52307


namespace NUMINAMATH_CALUDE_first_number_is_24_l523_52356

theorem first_number_is_24 (x : ℝ) : 
  (x + 35 + 58) / 3 = (19 + 51 + 29) / 3 + 6 → x = 24 := by
sorry

end NUMINAMATH_CALUDE_first_number_is_24_l523_52356


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l523_52379

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | |x| < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l523_52379


namespace NUMINAMATH_CALUDE_unique_digit_multiplication_l523_52362

theorem unique_digit_multiplication :
  ∃! (A B C D E : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    A ≠ 0 ∧
    (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
    E * 10000 + D * 1000 + C * 100 + B * 10 + A ∧
    A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 ∧ E = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_multiplication_l523_52362


namespace NUMINAMATH_CALUDE_complex_product_theorem_l523_52374

theorem complex_product_theorem (z₁ z₂ : ℂ) : 
  z₁.re = 2 ∧ z₁.im = 1 ∧ z₂.re = 0 ∧ z₂.im = -1 → z₁ * z₂ = 1 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l523_52374


namespace NUMINAMATH_CALUDE_baker_sales_change_l523_52339

/-- A baker's weekly pastry sales problem --/
theorem baker_sales_change (price : ℕ) (days_per_week : ℕ) (monday_sales : ℕ) (avg_sales : ℕ) 
  (h1 : price = 5)
  (h2 : days_per_week = 7)
  (h3 : monday_sales = 2)
  (h4 : avg_sales = 5) :
  ∃ (daily_change : ℕ),
    daily_change = 1 ∧
    monday_sales + 
    (monday_sales + daily_change) + 
    (monday_sales + 2 * daily_change) + 
    (monday_sales + 3 * daily_change) + 
    (monday_sales + 4 * daily_change) + 
    (monday_sales + 5 * daily_change) + 
    (monday_sales + 6 * daily_change) = days_per_week * avg_sales :=
by
  sorry

end NUMINAMATH_CALUDE_baker_sales_change_l523_52339


namespace NUMINAMATH_CALUDE_inequality_implication_l523_52306

theorem inequality_implication (x y : ℝ) : x < y → -x/2 > -y/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l523_52306


namespace NUMINAMATH_CALUDE_percent_relation_l523_52319

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) : 
  y = (3/7) * x := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l523_52319


namespace NUMINAMATH_CALUDE_chord_equation_l523_52331

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 1)

-- Define a chord of the ellipse
def is_chord (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2

-- Define M as the midpoint of the chord
def M_bisects_chord (A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1)/2, (A.2 + B.2)/2)

-- The theorem to prove
theorem chord_equation :
  ∀ A B : ℝ × ℝ,
  is_chord A B →
  M_bisects_chord A B →
  ∃ k m : ℝ, k = -1/2 ∧ m = 4 ∧ 
    ∀ x y : ℝ, (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) → y = k*x + m :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_l523_52331


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l523_52320

/-- Given a parabola y^2 = 2px bounded by x = a, the maximum area of an inscribed rectangle 
    with its midline on the parabola's axis is (4a/3) * sqrt(2ap/3) -/
theorem max_area_inscribed_rectangle (p a : ℝ) (hp : p > 0) (ha : a > 0) :
  let parabola := fun y : ℝ => y^2 / (2*p)
  let bound := a
  let inscribed_rectangle_area := fun x : ℝ => 2 * (a - x) * Real.sqrt (2*p*x)
  ∃ max_area : ℝ, max_area = (4*a/3) * Real.sqrt (2*a*p/3) ∧
    ∀ x, 0 < x ∧ x < a → inscribed_rectangle_area x ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_l523_52320


namespace NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l523_52363

/-- Represents a chessboard -/
structure Chessboard where
  rows : Nat
  cols : Nat

/-- Represents a triomino -/
structure Triomino where
  length : Nat
  width : Nat

/-- Function to check if a chessboard can be tiled with triominoes -/
def canBeTiled (board : Chessboard) (triomino : Triomino) : Prop :=
  (board.rows * board.cols) % (triomino.length * triomino.width) = 0

/-- Function to check if a chessboard with one corner removed can be tiled with triominoes -/
def canBeTiledWithCornerRemoved (board : Chessboard) (triomino : Triomino) : Prop :=
  ∃ (colorA colorB colorC : Nat),
    colorA + colorB + colorC = board.rows * board.cols - 1 ∧
    colorA = colorB ∧ colorB = colorC

/-- Theorem: An 8x8 chessboard cannot be tiled with 3x1 triominoes -/
theorem chessboard_cannot_be_tiled :
  ¬ canBeTiled ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

/-- Theorem: An 8x8 chessboard with one corner removed cannot be tiled with 3x1 triominoes -/
theorem chessboard_with_corner_removed_cannot_be_tiled :
  ¬ canBeTiledWithCornerRemoved ⟨8, 8⟩ ⟨3, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_chessboard_cannot_be_tiled_chessboard_with_corner_removed_cannot_be_tiled_l523_52363


namespace NUMINAMATH_CALUDE_wax_needed_for_SUV_l523_52382

/-- The amount of wax needed to detail Kellan's SUV -/
def wax_for_SUV : ℕ := by sorry

/-- The amount of wax needed to detail Kellan's car -/
def wax_for_car : ℕ := 3

/-- The amount of wax in the bottle Kellan bought -/
def wax_bought : ℕ := 11

/-- The amount of wax Kellan spilled -/
def wax_spilled : ℕ := 2

/-- The amount of wax left after detailing both vehicles -/
def wax_left : ℕ := 2

theorem wax_needed_for_SUV : 
  wax_for_SUV = 4 := by sorry

end NUMINAMATH_CALUDE_wax_needed_for_SUV_l523_52382


namespace NUMINAMATH_CALUDE_fourth_term_equals_seven_l523_52312

/-- Given a sequence {a_n} where the sum of the first n terms S_n = n^2, prove that a_4 = 7 -/
theorem fourth_term_equals_seven (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = n^2) : 
    a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_equals_seven_l523_52312


namespace NUMINAMATH_CALUDE_tan_plus_4sin_30_deg_l523_52308

theorem tan_plus_4sin_30_deg :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  tan_30 + 4 * sin_30 = (Real.sqrt 3 + 6) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_4sin_30_deg_l523_52308


namespace NUMINAMATH_CALUDE_coefficient_x_cube_in_expansion_l523_52326

theorem coefficient_x_cube_in_expansion : ∃ (c : ℤ), c = -10 ∧ 
  ∀ (x : ℝ), x * (x - 1)^5 = x^6 - 5*x^5 + 10*x^4 + c*x^3 + 5*x^2 - x := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cube_in_expansion_l523_52326


namespace NUMINAMATH_CALUDE_tangent_lines_existence_l523_52350

/-- The range of a for which there exist two different lines tangent to both f(x) and g(x) -/
theorem tangent_lines_existence (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₃ ∧ x₁ > 0 ∧ x₃ > 0 ∧
    (2 + a * Real.log x₁) + (a / x₁) * (x₂ - x₁) = (a * x₂^2 + 1) ∧
    (2 + a * Real.log x₃) + (a / x₃) * (x₄ - x₃) = (a * x₄^2 + 1)) ↔
  (a < 0 ∨ a > 2 / (1 + Real.log 2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_existence_l523_52350


namespace NUMINAMATH_CALUDE_union_and_subset_conditions_l523_52399

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem union_and_subset_conditions :
  (∀ m : ℝ, m = 4 → A ∪ B m = {x | -2 ≤ x ∧ x ≤ 7}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ 3) := by sorry

end NUMINAMATH_CALUDE_union_and_subset_conditions_l523_52399


namespace NUMINAMATH_CALUDE_success_permutations_count_l523_52389

/-- The number of distinct permutations of the multiset {S, S, S, U, C, C, E} -/
def successPermutations : ℕ :=
  Nat.factorial 7 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of SUCCESS is 420 -/
theorem success_permutations_count : successPermutations = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_count_l523_52389


namespace NUMINAMATH_CALUDE_cubes_volume_percentage_l523_52367

def box_length : ℕ := 8
def box_width : ℕ := 5
def box_height : ℕ := 12
def cube_side : ℕ := 4

def cubes_per_dimension (box_dim : ℕ) (cube_dim : ℕ) : ℕ :=
  box_dim / cube_dim

def total_cubes : ℕ :=
  (cubes_per_dimension box_length cube_side) *
  (cubes_per_dimension box_width cube_side) *
  (cubes_per_dimension box_height cube_side)

def cube_volume : ℕ := cube_side ^ 3
def total_cubes_volume : ℕ := total_cubes * cube_volume
def box_volume : ℕ := box_length * box_width * box_height

theorem cubes_volume_percentage :
  (total_cubes_volume : ℚ) / (box_volume : ℚ) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cubes_volume_percentage_l523_52367


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l523_52318

theorem arithmetic_geometric_mean_problem (p q r s : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 22 →
  (p * q * s)^(1/3) = 20 →
  r - p = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l523_52318


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l523_52323

theorem sum_of_coefficients (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 → a + b = 70 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l523_52323


namespace NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l523_52342

theorem cos_2alpha_minus_pi_3 (α : ℝ) 
  (h : Real.sin (α + π/6) - Real.cos α = 1/3) : 
  Real.cos (2*α - π/3) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_minus_pi_3_l523_52342


namespace NUMINAMATH_CALUDE_min_value_is_zero_l523_52371

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 9*x^2 - 24*x*y + 19*y^2 - 6*x - 9*y + 12

/-- The minimum value of f over all real x and y is 0 -/
theorem min_value_is_zero : 
  ∀ x y : ℝ, f x y ≥ 0 ∧ ∃ x₀ y₀ : ℝ, f x₀ y₀ = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_zero_l523_52371


namespace NUMINAMATH_CALUDE_amoeba_count_after_10_days_l523_52303

/-- The number of amoebas in the puddle after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- Theorem stating that after 10 days, there will be 59049 amoebas in the puddle -/
theorem amoeba_count_after_10_days : amoeba_count 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_after_10_days_l523_52303


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l523_52310

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 7 ∧ x₂ = 3 - Real.sqrt 7 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l523_52310


namespace NUMINAMATH_CALUDE_complementary_event_is_at_most_one_wins_l523_52395

-- Define the sample space
inductive Outcome
  | BothWin
  | AWinsBLoses
  | ALosesBWins
  | BothLose

-- Define the event A
def eventA (outcome : Outcome) : Prop :=
  outcome = Outcome.BothWin

-- Define the complementary event
def complementaryEventA (outcome : Outcome) : Prop :=
  outcome = Outcome.AWinsBLoses ∨ outcome = Outcome.ALosesBWins ∨ outcome = Outcome.BothLose

-- Theorem statement
theorem complementary_event_is_at_most_one_wins :
  ∀ (outcome : Outcome), ¬(eventA outcome) ↔ complementaryEventA outcome :=
sorry

end NUMINAMATH_CALUDE_complementary_event_is_at_most_one_wins_l523_52395


namespace NUMINAMATH_CALUDE_two_integer_b_values_l523_52378

theorem two_integer_b_values : 
  ∃! (S : Finset ℤ), 
    (Finset.card S = 2) ∧ 
    (∀ b ∈ S, ∃! (T : Finset ℤ), 
      (Finset.card T = 2) ∧ 
      (∀ x ∈ T, x^2 + b*x + 3 ≤ 0) ∧
      (∀ x : ℤ, x^2 + b*x + 3 ≤ 0 → x ∈ T)) := by
sorry

end NUMINAMATH_CALUDE_two_integer_b_values_l523_52378


namespace NUMINAMATH_CALUDE_hassans_orange_trees_l523_52373

/-- Represents the number of trees in an orchard --/
structure Orchard :=
  (orange : ℕ)
  (apple : ℕ)

/-- The total number of trees in an orchard --/
def Orchard.total (o : Orchard) : ℕ := o.orange + o.apple

theorem hassans_orange_trees :
  ∀ (ahmed hassan : Orchard),
  ahmed.orange = 8 →
  ahmed.apple = 4 * hassan.apple →
  hassan.apple = 1 →
  ahmed.total = hassan.total + 9 →
  hassan.orange = 2 := by
sorry

end NUMINAMATH_CALUDE_hassans_orange_trees_l523_52373


namespace NUMINAMATH_CALUDE_caravan_hens_l523_52305

def caravan (num_hens : ℕ) : Prop :=
  let num_goats : ℕ := 45
  let num_camels : ℕ := 8
  let num_keepers : ℕ := 15
  let total_heads : ℕ := num_hens + num_goats + num_camels + num_keepers
  let total_feet : ℕ := 2 * num_hens + 4 * num_goats + 4 * num_camels + 2 * num_keepers
  total_feet = total_heads + 224

theorem caravan_hens : ∃ (num_hens : ℕ), caravan num_hens ∧ num_hens = 50 := by
  sorry

end NUMINAMATH_CALUDE_caravan_hens_l523_52305


namespace NUMINAMATH_CALUDE_no_four_digit_perfect_square_palindromes_l523_52390

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_four_digit_perfect_square_palindromes_l523_52390


namespace NUMINAMATH_CALUDE_integral_sqrt_plus_x_equals_pi_over_two_l523_52346

open Set
open MeasureTheory
open Interval

/-- The definite integral of √(1-x²) + x from -1 to 1 equals π/2 -/
theorem integral_sqrt_plus_x_equals_pi_over_two :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_plus_x_equals_pi_over_two_l523_52346


namespace NUMINAMATH_CALUDE_birdhouse_earnings_l523_52386

/-- The price of a large birdhouse in dollars -/
def large_price : ℕ := 22

/-- The price of a medium birdhouse in dollars -/
def medium_price : ℕ := 16

/-- The price of a small birdhouse in dollars -/
def small_price : ℕ := 7

/-- The number of large birdhouses sold -/
def large_sold : ℕ := 2

/-- The number of medium birdhouses sold -/
def medium_sold : ℕ := 2

/-- The number of small birdhouses sold -/
def small_sold : ℕ := 3

/-- The total money earned from selling birdhouses -/
def total_earned : ℕ := large_price * large_sold + medium_price * medium_sold + small_price * small_sold

theorem birdhouse_earnings : total_earned = 97 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_earnings_l523_52386


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l523_52375

theorem consecutive_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (n + 2)^2 - n^2 = 4 * k :=
sorry

theorem consecutive_odd_integers_square_difference (n : ℤ) : 
  ∃ k : ℤ, (2*n + 3)^2 - (2*n - 1)^2 = 8 * k :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_consecutive_odd_integers_square_difference_l523_52375


namespace NUMINAMATH_CALUDE_probability_all_green_apples_l523_52364

def total_apples : ℕ := 10
def red_apples : ℕ := 6
def green_apples : ℕ := 4
def chosen_apples : ℕ := 3

theorem probability_all_green_apples :
  (Nat.choose green_apples chosen_apples : ℚ) / (Nat.choose total_apples chosen_apples) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_green_apples_l523_52364


namespace NUMINAMATH_CALUDE_aaron_cards_found_l523_52372

/-- Given that Aaron initially had 5 cards and ended up with 67 cards,
    prove that he found 62 cards. -/
theorem aaron_cards_found :
  let initial_cards : ℕ := 5
  let final_cards : ℕ := 67
  let cards_found := final_cards - initial_cards
  cards_found = 62 := by sorry

end NUMINAMATH_CALUDE_aaron_cards_found_l523_52372


namespace NUMINAMATH_CALUDE_escalator_standing_time_l523_52396

/-- Represents the time it takes Clea to ride down an escalator under different conditions -/
def escalator_time (non_operating_time walking_time standing_time : ℝ) : Prop :=
  -- Distance of the escalator
  ∃ d : ℝ,
  -- Speed of Clea walking down the escalator
  ∃ c : ℝ,
  -- Speed of the escalator
  ∃ s : ℝ,
  -- Conditions
  (d = 70 * c) ∧  -- Time to walk down non-operating escalator
  (d = 28 * (c + s)) ∧  -- Time to walk down operating escalator
  (standing_time = d / s) ∧  -- Time to stand on operating escalator
  (standing_time = 47)  -- The result we want to prove

/-- Theorem stating that given the conditions, the standing time on the operating escalator is 47 seconds -/
theorem escalator_standing_time :
  escalator_time 70 28 47 :=
sorry

end NUMINAMATH_CALUDE_escalator_standing_time_l523_52396


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l523_52336

/-- Given an arithmetic sequence {aₙ} with S₃ = 6 and a₃ = 4, prove that the common difference d = 2 -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sequence of partial sums
  (h1 : S 3 = 6)  -- Given S₃ = 6
  (h2 : a 3 = 4)  -- Given a₃ = 4
  (h3 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)  -- Sum formula for arithmetic sequence
  (h4 : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l523_52336


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l523_52333

/-- Two 2D vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (-3, 2)
  let b : ℝ × ℝ := (x, 4)
  are_parallel a b → x = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l523_52333


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l523_52327

/-- Represents the ratio of paints (red:blue:yellow:white) -/
structure PaintRatio :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (white : ℕ)

/-- Calculates the amount of blue paint needed given a paint ratio and the amount of white paint used -/
def blue_paint_needed (ratio : PaintRatio) (white_paint : ℕ) : ℕ :=
  (ratio.blue * white_paint) / ratio.white

/-- Theorem stating that given the specific paint ratio and 16 quarts of white paint, 12 quarts of blue paint are needed -/
theorem blue_paint_calculation (ratio : PaintRatio) (h1 : ratio.red = 2) (h2 : ratio.blue = 3) 
    (h3 : ratio.yellow = 1) (h4 : ratio.white = 4) (white_paint : ℕ) (h5 : white_paint = 16) : 
  blue_paint_needed ratio white_paint = 12 := by
  sorry

#eval blue_paint_needed {red := 2, blue := 3, yellow := 1, white := 4} 16

end NUMINAMATH_CALUDE_blue_paint_calculation_l523_52327


namespace NUMINAMATH_CALUDE_complex_equation_solution_l523_52361

theorem complex_equation_solution (z : ℂ) : (1 - I)^2 * z = 3 + 2*I → z = -1 + (3/2)*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l523_52361


namespace NUMINAMATH_CALUDE_find_number_l523_52392

/-- Given the equation (47% of 1442 - 36% of N) + 66 = 6, prove that N = 2049.28 --/
theorem find_number (N : ℝ) : (0.47 * 1442 - 0.36 * N) + 66 = 6 → N = 2049.28 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l523_52392


namespace NUMINAMATH_CALUDE_hyperbola_equation_l523_52366

/-- Given a hyperbola and an ellipse with shared foci and related eccentricities,
    prove that the hyperbola has the equation x²/4 - y²/3 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) ∧ 
  (∃ (x y : ℝ), x^2/16 + y^2/9 = 1) ∧
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c^2 = 16 - 9) ∧
  (∃ (e_h e_e : ℝ), e_h = c/a ∧ e_e = c/4 ∧ e_h = 2*e_e) →
  a^2 = 4 ∧ b^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l523_52366


namespace NUMINAMATH_CALUDE_base_8_45327_equals_19159_l523_52301

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_45327_equals_19159 :
  base_8_to_10 [7, 2, 3, 5, 4] = 19159 := by
  sorry

end NUMINAMATH_CALUDE_base_8_45327_equals_19159_l523_52301


namespace NUMINAMATH_CALUDE_total_stairs_climbed_l523_52391

def samir_stairs : ℕ := 318

def veronica_stairs : ℕ := samir_stairs / 2 + 18

theorem total_stairs_climbed : samir_stairs + veronica_stairs = 495 := by
  sorry

end NUMINAMATH_CALUDE_total_stairs_climbed_l523_52391


namespace NUMINAMATH_CALUDE_parameterized_line_problem_l523_52315

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  -- The vector on the line at parameter t
  vector : ℝ → (Fin 3 → ℝ)

/-- The problem statement as a theorem -/
theorem parameterized_line_problem :
  ∀ (line : ParameterizedLine),
    (line.vector 1 = ![2, 4, 9]) →
    (line.vector 3 = ![1, 1, 2]) →
    (line.vector 4 = ![0.5, -0.5, -1.5]) := by
  sorry

end NUMINAMATH_CALUDE_parameterized_line_problem_l523_52315


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l523_52304

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l523_52304


namespace NUMINAMATH_CALUDE_functions_identical_functions_not_identical_l523_52359

-- Part 1
theorem functions_identical (x : ℝ) (h : x ≠ 0) : x / x^2 = 1 / x := by sorry

-- Part 2
theorem functions_not_identical : ∃ x : ℝ, x ≠ Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_functions_identical_functions_not_identical_l523_52359


namespace NUMINAMATH_CALUDE_point_on_line_value_l523_52357

theorem point_on_line_value (a b : ℝ) : 
  b = 3 * a - 2 → 2 * b - 6 * a + 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l523_52357


namespace NUMINAMATH_CALUDE_count_numbers_with_square_factor_eq_41_l523_52302

def perfect_squares : List Nat := [4, 9, 16, 25, 36, 49, 64, 100]

def is_divisible_by_square (n : Nat) : Bool :=
  perfect_squares.any (λ s => n % s = 0)

def count_numbers_with_square_factor : Nat :=
  (List.range 100).filter is_divisible_by_square |>.length

theorem count_numbers_with_square_factor_eq_41 :
  count_numbers_with_square_factor = 41 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_square_factor_eq_41_l523_52302


namespace NUMINAMATH_CALUDE_largest_special_square_l523_52321

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := (n - last_two_digits n) / 100

theorem largest_special_square : 
  ∀ n : ℕ, 
    (is_square n ∧ 
     n % 100 ≠ 0 ∧ 
     is_square (remove_last_two_digits n)) →
    n ≤ 1681 :=
sorry

end NUMINAMATH_CALUDE_largest_special_square_l523_52321


namespace NUMINAMATH_CALUDE_movie_ratio_is_half_l523_52384

/-- The ratio of movies Theresa saw in 2009 to movies Timothy saw in 2009 -/
def movie_ratio (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ) : ℚ :=
  theresa_2009 / timothy_2009

theorem movie_ratio_is_half :
  ∀ (timothy_2009 timothy_2010 theresa_2009 theresa_2010 : ℕ),
    timothy_2009 = 24 →
    timothy_2010 = timothy_2009 + 7 →
    theresa_2010 = 2 * timothy_2010 →
    timothy_2009 + theresa_2009 + timothy_2010 + theresa_2010 = 129 →
    movie_ratio timothy_2009 timothy_2010 theresa_2009 theresa_2010 = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_movie_ratio_is_half_l523_52384


namespace NUMINAMATH_CALUDE_andreys_stamps_l523_52347

theorem andreys_stamps :
  ∃ (x : ℕ), x > 0 ∧ x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andreys_stamps_l523_52347


namespace NUMINAMATH_CALUDE_rubiks_cube_return_to_original_state_l523_52335

theorem rubiks_cube_return_to_original_state 
  {S : Type} [Finite S] (f : S → S) : 
  ∃ n : ℕ+, ∀ x : S, (f^[n] x = x) := by
  sorry

end NUMINAMATH_CALUDE_rubiks_cube_return_to_original_state_l523_52335


namespace NUMINAMATH_CALUDE_sum_of_cubes_l523_52322

theorem sum_of_cubes (a b c d e : ℕ) : 
  a ∈ ({0, 1, 2} : Set ℕ) → 
  b ∈ ({0, 1, 2} : Set ℕ) → 
  c ∈ ({0, 1, 2} : Set ℕ) → 
  d ∈ ({0, 1, 2} : Set ℕ) → 
  e ∈ ({0, 1, 2} : Set ℕ) → 
  a + b + c + d + e = 6 → 
  a^2 + b^2 + c^2 + d^2 + e^2 = 10 → 
  a^3 + b^3 + c^3 + d^3 + e^3 = 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l523_52322


namespace NUMINAMATH_CALUDE_congruence_power_l523_52324

theorem congruence_power (a b m n : ℕ) (h : a ≡ b [MOD m]) : a^n ≡ b^n [MOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_power_l523_52324


namespace NUMINAMATH_CALUDE_at_most_two_solutions_l523_52380

theorem at_most_two_solutions (a b c : ℝ) (ha : a > 2000) :
  ¬∃ (x₁ x₂ x₃ : ℤ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧
    (|a * x₁^2 + b * x₁ + c| ≤ 1000) ∧
    (|a * x₂^2 + b * x₂ + c| ≤ 1000) ∧
    (|a * x₃^2 + b * x₃ + c| ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_at_most_two_solutions_l523_52380


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l523_52377

/-- Given an arithmetic sequence of 6 terms where the first term is 11 and the last term is 51,
    prove that the third term is 27. -/
theorem arithmetic_sequence_third_term :
  ∀ (seq : Fin 6 → ℝ),
    (∀ i j : Fin 6, seq (i + 1) - seq i = seq (j + 1) - seq j) →  -- arithmetic sequence
    seq 0 = 11 →  -- first term is 11
    seq 5 = 51 →  -- last term is 51
    seq 2 = 27 :=  -- third term is 27
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l523_52377


namespace NUMINAMATH_CALUDE_cookies_sold_proof_l523_52381

/-- The number of packs of cookies sold by Robyn -/
def robyn_sales : ℕ := 47

/-- The number of packs of cookies sold by Lucy -/
def lucy_sales : ℕ := 29

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_sales : ℕ := robyn_sales + lucy_sales

theorem cookies_sold_proof : total_sales = 76 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_proof_l523_52381
