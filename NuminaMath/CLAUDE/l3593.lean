import Mathlib

namespace NUMINAMATH_CALUDE_johns_weekly_water_intake_l3593_359397

/-- Proves that John drinks 42 quarts of water in a week -/
theorem johns_weekly_water_intake :
  let daily_intake_gallons : ℚ := 3/2
  let gallons_to_quarts : ℚ := 4
  let days_in_week : ℕ := 7
  let weekly_intake_quarts : ℚ := daily_intake_gallons * gallons_to_quarts * days_in_week
  weekly_intake_quarts = 42 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_water_intake_l3593_359397


namespace NUMINAMATH_CALUDE_distance_on_parametric_line_l3593_359382

/-- The distance between two points on a parametric line --/
theorem distance_on_parametric_line :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + 3 * t, 1 + t)
  let point1 := line 0
  let point2 := line 1
  (point1.1 - point2.1)^2 + (point1.2 - point2.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_parametric_line_l3593_359382


namespace NUMINAMATH_CALUDE_points_per_question_l3593_359311

theorem points_per_question (first_half : ℕ) (second_half : ℕ) (final_score : ℕ) :
  first_half = 8 →
  second_half = 2 →
  final_score = 80 →
  final_score / (first_half + second_half) = 8 := by
sorry

end NUMINAMATH_CALUDE_points_per_question_l3593_359311


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l3593_359383

theorem fraction_sum_simplification :
  18 / 462 + 35 / 77 = 38 / 77 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l3593_359383


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l3593_359384

/-- The modulus of the complex number z = (1-2i)(3+i) is equal to 5√2 -/
theorem modulus_of_complex_product : 
  let z : ℂ := (1 - 2*I) * (3 + I)
  ‖z‖ = 5 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l3593_359384


namespace NUMINAMATH_CALUDE_given_number_eq_scientific_repr_l3593_359362

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The given number -0.000032 -/
def given_number : ℝ := -0.000032

/-- The scientific notation representation of the given number -/
def scientific_repr : ScientificNotation :=
  { coefficient := -3.2
    exponent := -5
    property := by sorry }

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_eq_scientific_repr :
  given_number = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_eq_scientific_repr_l3593_359362


namespace NUMINAMATH_CALUDE_not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l3593_359355

/-- Definition of sum-product equal number pair -/
def is_sum_product_equal (a b : ℝ) : Prop := a + b = a * b

/-- Theorem 1: (-2, 4) is not a sum-product equal number pair -/
theorem not_sum_product_equal_neg_two_four : ¬ is_sum_product_equal (-2) 4 := by sorry

/-- Theorem 2: (√2+2, √2) is a sum-product equal number pair -/
theorem sum_product_equal_sqrt_two_plus_two_sqrt_two : is_sum_product_equal (Real.sqrt 2 + 2) (Real.sqrt 2) := by sorry

/-- Theorem 3: For (m,n) where m,n ≠ 1, if it's a sum-product equal number pair, then m = n / (n-1) -/
theorem sum_product_equal_relation (m n : ℝ) (hm : m ≠ 1) (hn : n ≠ 1) :
  is_sum_product_equal m n → m = n / (n - 1) := by sorry

end NUMINAMATH_CALUDE_not_sum_product_equal_neg_two_four_sum_product_equal_sqrt_two_plus_two_sqrt_two_sum_product_equal_relation_l3593_359355


namespace NUMINAMATH_CALUDE_average_monthly_income_l3593_359335

/-- Given a person's expenses and savings over a year, calculate their average monthly income. -/
theorem average_monthly_income
  (expense_first_3_months : ℕ)
  (expense_next_4_months : ℕ)
  (expense_last_5_months : ℕ)
  (yearly_savings : ℕ)
  (h1 : expense_first_3_months = 1700)
  (h2 : expense_next_4_months = 1550)
  (h3 : expense_last_5_months = 1800)
  (h4 : yearly_savings = 5200) :
  (3 * expense_first_3_months + 4 * expense_next_4_months + 5 * expense_last_5_months + yearly_savings) / 12 = 2125 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_income_l3593_359335


namespace NUMINAMATH_CALUDE_largest_integer_less_than_sqrt7_plus_sqrt3_power6_l3593_359304

theorem largest_integer_less_than_sqrt7_plus_sqrt3_power6 :
  ⌊(Real.sqrt 7 + Real.sqrt 3)^6⌋ = 7039 := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_sqrt7_plus_sqrt3_power6_l3593_359304


namespace NUMINAMATH_CALUDE_pen_pencil_distribution_l3593_359389

theorem pen_pencil_distribution (P : ℕ) : 
  (∃ (k : ℕ), P = 20 * k) ↔ 
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 1340 / x = y ∧ P / x = y ∧ x ≤ 20 ∧ 
   ∀ (z : ℕ), z > x → (1340 / z ≠ P / z ∨ 1340 % z ≠ 0 ∨ P % z ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_pen_pencil_distribution_l3593_359389


namespace NUMINAMATH_CALUDE_f_minimum_value_range_l3593_359358

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * (x + 1)

theorem f_minimum_value_range (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀ ∧ f a x₀ > a^2 + a) ↔ -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_range_l3593_359358


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3593_359316

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if it has an asymptote y = √5 x, then its eccentricity is √6. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.sqrt 5) : 
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3593_359316


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l3593_359375

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) : 
  length = 6 → width = 4.75 → rate = 900 → length * width * rate = 25650 := by sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l3593_359375


namespace NUMINAMATH_CALUDE_multiplication_error_factors_l3593_359324

theorem multiplication_error_factors : ∃ (x y z : ℕ), 
  x = y + 10 ∧ 
  x * y = z + 40 ∧ 
  z = 39 * y + 22 ∧ 
  x = 41 ∧ 
  y = 31 := by
sorry

end NUMINAMATH_CALUDE_multiplication_error_factors_l3593_359324


namespace NUMINAMATH_CALUDE_mixed_number_properties_l3593_359352

/-- Represents a mixed number as a pair of integers (whole, numerator, denominator) -/
structure MixedNumber where
  whole : ℤ
  numerator : ℕ
  denominator : ℕ
  h_pos : denominator > 0
  h_proper : numerator < denominator

/-- The smallest composite number -/
def smallest_composite : ℕ := 4

/-- Converts a mixed number to a rational number -/
def mixed_to_rational (m : MixedNumber) : ℚ :=
  m.whole + (m.numerator : ℚ) / m.denominator

theorem mixed_number_properties (m : MixedNumber) 
  (h_m : m = ⟨3, 2, 7, by norm_num, by norm_num⟩) : 
  ∃ (fractional_unit : ℚ) (num_units : ℕ) (units_to_add : ℕ),
    fractional_unit = 1 / 7 ∧ 
    num_units = 23 ∧
    units_to_add = 5 ∧
    mixed_to_rational m = num_units * fractional_unit ∧
    mixed_to_rational m + units_to_add * fractional_unit = smallest_composite := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_properties_l3593_359352


namespace NUMINAMATH_CALUDE_bianca_cupcakes_theorem_l3593_359302

/-- Represents the number of cupcakes Bianca made after selling the first batch -/
def cupcakes_made_after (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Bianca made 17 cupcakes after selling the first batch -/
theorem bianca_cupcakes_theorem (initial : ℕ) (sold : ℕ) (final : ℕ)
  (h1 : initial = 14)
  (h2 : sold = 6)
  (h3 : final = 25) :
  cupcakes_made_after initial sold final = 17 := by
  sorry

end NUMINAMATH_CALUDE_bianca_cupcakes_theorem_l3593_359302


namespace NUMINAMATH_CALUDE_unique_common_solution_coefficient_l3593_359332

theorem unique_common_solution_coefficient : 
  ∃! a : ℝ, ∃ x : ℝ, (x^2 + a*x + 1 = 0) ∧ (x^2 - x - a = 0) ∧ (a = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_common_solution_coefficient_l3593_359332


namespace NUMINAMATH_CALUDE_mikes_candies_l3593_359363

theorem mikes_candies (initial_candies : ℕ) : 
  (initial_candies > 0) →
  (initial_candies % 4 = 0) →
  (∃ (sister_took : ℕ), 1 ≤ sister_took ∧ sister_took ≤ 4 ∧
    5 + sister_took = initial_candies * 3 / 4 * 2 / 3 - 24) →
  initial_candies = 64 := by
sorry

end NUMINAMATH_CALUDE_mikes_candies_l3593_359363


namespace NUMINAMATH_CALUDE_problem_statement_l3593_359340

theorem problem_statement (x : ℝ) : 
  let a := x^2 - 1
  let b := 2*x + 2
  (a + b ≥ 0) ∧ (max a b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3593_359340


namespace NUMINAMATH_CALUDE_subtract_three_five_l3593_359336

theorem subtract_three_five : 3 - 5 = -2 := by sorry

end NUMINAMATH_CALUDE_subtract_three_five_l3593_359336


namespace NUMINAMATH_CALUDE_baseball_card_count_l3593_359386

def final_card_count (initial_count : ℕ) : ℕ :=
  let after_maria := initial_count - (initial_count + 1) / 2
  let after_peter := after_maria - 1
  let final_count := after_peter * 3
  final_count

theorem baseball_card_count : final_card_count 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_count_l3593_359386


namespace NUMINAMATH_CALUDE_ab_max_and_inequality_l3593_359305

theorem ab_max_and_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + a = 15 - b) :
  (∃ (max : ℝ), max = 9 ∧ a * b ≤ max) ∧ b ≥ 6 - a := by
  sorry

end NUMINAMATH_CALUDE_ab_max_and_inequality_l3593_359305


namespace NUMINAMATH_CALUDE_alice_next_birthday_age_l3593_359337

theorem alice_next_birthday_age :
  ∀ (a b c : ℝ),
  a = 1.25 * b →                -- Alice is 25% older than Bob
  b = 0.7 * c →                 -- Bob is 30% younger than Carlos
  a + b + c = 30 →              -- Sum of their ages is 30 years
  ⌊a⌋ + 1 = 11 :=               -- Alice's age on her next birthday
by
  sorry


end NUMINAMATH_CALUDE_alice_next_birthday_age_l3593_359337


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3593_359385

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_7 = 12, the sum of a_3 and a_11 is 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a7 : a 7 = 12) : 
  a 3 + a 11 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3593_359385


namespace NUMINAMATH_CALUDE_simplify_expression_l3593_359333

theorem simplify_expression (x : ℝ) : 
  3 * x^3 + 4 * x^2 + 2 - (7 - 3 * x^3 - 4 * x^2) = 6 * x^3 + 8 * x^2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3593_359333


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3593_359308

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.30) 
  (h2 : ingrid_tax_rate = 0.40) 
  (h3 : john_income = 56000) 
  (h4 : ingrid_income = 74000) : 
  ∃ (combined_rate : ℝ), 
    combined_rate = (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) :=
by
  sorry

#eval (0.30 * 56000 + 0.40 * 74000) / (56000 + 74000)

end NUMINAMATH_CALUDE_combined_tax_rate_l3593_359308


namespace NUMINAMATH_CALUDE_triangle_area_l3593_359370

theorem triangle_area (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → 
  b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 → 
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3593_359370


namespace NUMINAMATH_CALUDE_f_difference_l3593_359354

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3593_359354


namespace NUMINAMATH_CALUDE_two_thirds_of_45_plus_10_l3593_359376

theorem two_thirds_of_45_plus_10 : ((2 : ℚ) / 3) * 45 + 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_45_plus_10_l3593_359376


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3593_359390

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3593_359390


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l3593_359348

/-- Given a 6-8-10 right triangle, prove that the sum of the areas of right isosceles triangles
    constructed on the two shorter sides is equal to the area of the right isosceles triangle
    constructed on the hypotenuse. -/
theorem isosceles_triangle_areas_sum (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10)
  (h4 : a^2 + b^2 = c^2) : (1/2 * a^2) + (1/2 * b^2) = 1/2 * c^2 := by
  sorry

#check isosceles_triangle_areas_sum

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l3593_359348


namespace NUMINAMATH_CALUDE_count_closest_to_two_sevenths_l3593_359365

def is_closest_to_two_sevenths (r : ℚ) : Prop :=
  ∀ n d : ℕ, n ≤ 2 → d > 0 → |r - 2/7| ≤ |r - (n : ℚ)/d|

def is_four_place_decimal (r : ℚ) : Prop :=
  ∃ a b c d : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    r = (a * 1000 + b * 100 + c * 10 + d) / 10000

theorem count_closest_to_two_sevenths :
  ∃! (s : Finset ℚ), 
    (∀ r ∈ s, is_four_place_decimal r ∧ is_closest_to_two_sevenths r) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_closest_to_two_sevenths_l3593_359365


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3593_359379

theorem circle_y_axis_intersection_sum : 
  ∀ (x y : ℝ), 
  ((x + 8)^2 + (y - 5)^2 = 13^2) →  -- Circle equation
  (x = 0) →                        -- Points on y-axis
  ∃ (y1 y2 : ℝ),
    ((0 + 8)^2 + (y1 - 5)^2 = 13^2) ∧
    ((0 + 8)^2 + (y2 - 5)^2 = 13^2) ∧
    y1 + y2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3593_359379


namespace NUMINAMATH_CALUDE_triangle_side_difference_l3593_359318

theorem triangle_side_difference (a b : ℕ) (ha : a = 8) (hb : b = 13) : 
  (∃ (x_max x_min : ℕ), 
    (∀ x : ℕ, (x + a > b ∧ x + b > a ∧ a + b > x) → x_min ≤ x ∧ x ≤ x_max) ∧
    (x_max + a > b ∧ x_max + b > a ∧ a + b > x_max) ∧
    (x_min + a > b ∧ x_min + b > a ∧ a + b > x_min) ∧
    (∀ y : ℕ, y > x_max ∨ y < x_min → ¬(y + a > b ∧ y + b > a ∧ a + b > y)) ∧
    x_max - x_min = 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l3593_359318


namespace NUMINAMATH_CALUDE_negative_y_positive_l3593_359327

theorem negative_y_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_y_positive_l3593_359327


namespace NUMINAMATH_CALUDE_lunch_cost_proof_l3593_359392

theorem lunch_cost_proof (adam_cost rick_cost jose_cost : ℝ) :
  adam_cost = (2/3) * rick_cost →
  rick_cost = jose_cost →
  jose_cost = 45 →
  adam_cost + rick_cost + jose_cost = 120 := by
sorry

end NUMINAMATH_CALUDE_lunch_cost_proof_l3593_359392


namespace NUMINAMATH_CALUDE_restaurant_glasses_count_l3593_359381

theorem restaurant_glasses_count :
  ∀ (x y : ℕ),
  -- x is the number of 12-glass boxes, y is the number of 16-glass boxes
  y = x + 16 →
  -- The average number of glasses per box is 15
  (12 * x + 16 * y) / (x + y) = 15 →
  -- The total number of glasses is 480
  12 * x + 16 * y = 480 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_glasses_count_l3593_359381


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3593_359398

theorem adult_ticket_cost (student_price : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_amount : ℕ) : 
  student_price = 6 →
  num_students = 20 →
  num_adults = 12 →
  total_amount = 216 →
  ∃ (adult_price : ℕ), 
    student_price * num_students + adult_price * num_adults = total_amount ∧ 
    adult_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3593_359398


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3593_359343

/-- Given an inverse proportion function y = k/x passing through (-2, 3), prove k = -6 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3593_359343


namespace NUMINAMATH_CALUDE_playground_to_landscape_ratio_l3593_359345

/-- A rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  length_breadth_relation : length = 4 * breadth
  length_value : length = 120
  playground_size : playground_area = 1200

/-- The ratio of playground area to total landscape area is 1:3 -/
theorem playground_to_landscape_ratio (L : Landscape) :
  L.playground_area / (L.length * L.breadth) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_playground_to_landscape_ratio_l3593_359345


namespace NUMINAMATH_CALUDE_slope_angle_sqrt3_l3593_359394

/-- The slope angle of the line y = √3x + 1 is 60° -/
theorem slope_angle_sqrt3 : 
  let l : ℝ → ℝ := λ x => Real.sqrt 3 * x + 1
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_sqrt3_l3593_359394


namespace NUMINAMATH_CALUDE_carls_cupcake_goal_l3593_359357

/-- Carl's cupcake selling problem -/
theorem carls_cupcake_goal (goal : ℕ) (days : ℕ) (payment : ℕ) (cupcakes_per_day : ℕ) : 
  goal = 96 → days = 2 → payment = 24 → cupcakes_per_day * days = goal + payment → cupcakes_per_day = 60 := by
  sorry

end NUMINAMATH_CALUDE_carls_cupcake_goal_l3593_359357


namespace NUMINAMATH_CALUDE_team_loss_percentage_l3593_359353

/-- Represents the ratio of games won to games lost -/
def winLossRatio : Rat := 7 / 3

/-- The total number of games played -/
def totalGames : ℕ := 50

/-- Calculates the percentage of games lost -/
def percentLost : ℚ :=
  let gamesLost := totalGames / (1 + winLossRatio)
  (gamesLost / totalGames) * 100

theorem team_loss_percentage :
  ⌊percentLost⌋ = 30 :=
sorry

end NUMINAMATH_CALUDE_team_loss_percentage_l3593_359353


namespace NUMINAMATH_CALUDE_sqrt_two_irrationality_proof_assumption_l3593_359371

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define what it means for a real number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_two_irrationality_proof_assumption :
  (IsIrrational (Real.sqrt 2)) ↔ 
  (¬IsRational (Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrationality_proof_assumption_l3593_359371


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3593_359346

theorem inequality_system_solution (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 1 ↔ x - a > 2 ∧ 2*x - b < 0) →
  a^(-b) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3593_359346


namespace NUMINAMATH_CALUDE_square_sum_equals_nine_billion_four_million_l3593_359338

theorem square_sum_equals_nine_billion_four_million : (300000 : ℕ)^2 + (20000 : ℕ)^2 = 9004000000 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_nine_billion_four_million_l3593_359338


namespace NUMINAMATH_CALUDE_composite_numbers_with_special_divisors_l3593_359350

theorem composite_numbers_with_special_divisors :
  ∀ n : ℕ, n > 1 →
    (∀ d : ℕ, d ∣ n → d ≠ 1 → d ≠ n → n - 20 ≤ d ∧ d ≤ n - 12) →
    n = 21 ∨ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_composite_numbers_with_special_divisors_l3593_359350


namespace NUMINAMATH_CALUDE_blue_section_damage_probability_l3593_359307

/-- The probability of k successes in n Bernoulli trials -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- The number of trials -/
def n : ℕ := 7

/-- The number of successes -/
def k : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

theorem blue_section_damage_probability :
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_damage_probability_l3593_359307


namespace NUMINAMATH_CALUDE_rice_price_reduction_l3593_359399

theorem rice_price_reduction (P : ℝ) (h : P > 0) :
  49 * P = 50 * (P * (1 - 2/100)) :=
by sorry

end NUMINAMATH_CALUDE_rice_price_reduction_l3593_359399


namespace NUMINAMATH_CALUDE_calculate_expression_l3593_359366

theorem calculate_expression (y : ℝ) (h : y = 3) : y + y * (y^y)^2 = 2190 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3593_359366


namespace NUMINAMATH_CALUDE_population_changes_l3593_359334

/-- Enumeration of possible population number changes --/
inductive PopulationChange
  | Increase
  | Decrease
  | Fluctuation
  | Extinction

/-- Theorem stating that population changes can be increase, decrease, fluctuation, or extinction --/
theorem population_changes : 
  ∀ (change : PopulationChange), 
    change = PopulationChange.Increase ∨
    change = PopulationChange.Decrease ∨
    change = PopulationChange.Fluctuation ∨
    change = PopulationChange.Extinction :=
by
  sorry

#check population_changes

end NUMINAMATH_CALUDE_population_changes_l3593_359334


namespace NUMINAMATH_CALUDE_flag_designs_count_l3593_359313

/-- The number of colors available for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes in the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27 -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l3593_359313


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3593_359300

theorem cyclic_fraction_inequality (x y z : ℝ) :
  (x^2 / (x^2 + 2*y*z)) + (y^2 / (y^2 + 2*z*x)) + (z^2 / (z^2 + 2*x*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3593_359300


namespace NUMINAMATH_CALUDE_rsa_congruence_l3593_359380

theorem rsa_congruence (p q e d M : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  p ≠ q → 
  (e * d) % ((p - 1) * (q - 1)) = 1 → 
  ((M ^ e) ^ d) % (p * q) = M % (p * q) := by
sorry

end NUMINAMATH_CALUDE_rsa_congruence_l3593_359380


namespace NUMINAMATH_CALUDE_tangent_lines_max_value_min_value_l3593_359368

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x - 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x - 9

-- Theorem for tangent lines
theorem tangent_lines :
  ∃ (x₀ y₀ : ℝ), 
    (f' x₀ = -9 ∧ y₀ = f x₀ ∧ (y₀ = -9*x₀ - 3 ∨ y₀ = -9*x₀ + 19)) :=
sorry

-- Theorem for maximum value
theorem max_value :
  ∃ (x : ℝ), f x = 24 ∧ ∀ y, f y ≤ f x :=
sorry

-- Theorem for minimum value
theorem min_value :
  ∃ (x : ℝ), f x = -8 ∧ ∀ y, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_max_value_min_value_l3593_359368


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3593_359347

/-- An arithmetic sequence with positive first term and a_3/a_4 = 7/5 reaches maximum sum at n = 6 -/
theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 > 0 →  -- positive first term
  a 3 / a 4 = 7 / 5 →  -- given ratio
  ∃ S : ℕ → ℝ, ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2 ∧  -- sum formula
  (∀ m, S m ≤ S 6) :=  -- maximum sum at n = 6
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3593_359347


namespace NUMINAMATH_CALUDE_line_intersects_equidistant_points_in_first_and_second_quadrants_l3593_359319

/-- The line equation 4x + 6y = 24 -/
def line_equation (x y : ℝ) : Prop := 4 * x + 6 * y = 24

/-- A point (x, y) is equidistant from coordinate axes if |x| = |y| -/
def equidistant (x y : ℝ) : Prop := abs x = abs y

/-- A point (x, y) is in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The line 4x + 6y = 24 intersects with y = x and y = -x only in the first and second quadrants -/
theorem line_intersects_equidistant_points_in_first_and_second_quadrants :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_equation x₁ y₁ ∧ equidistant x₁ y₁ ∧ first_quadrant x₁ y₁ ∧
    line_equation x₂ y₂ ∧ equidistant x₂ y₂ ∧ second_quadrant x₂ y₂ ∧
    (∀ (x y : ℝ), line_equation x y ∧ equidistant x y →
      first_quadrant x y ∨ second_quadrant x y) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_equidistant_points_in_first_and_second_quadrants_l3593_359319


namespace NUMINAMATH_CALUDE_binary_representation_properties_l3593_359315

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def count_ones (n : ℕ) : ℕ := sorry

/-- A function that counts the number of 0s in the binary representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number n that is a multiple of 17 and has exactly three 1s in its binary representation:
    1) The binary representation of n has at least six 0s
    2) If the binary representation of n has exactly 7 0s, then n is even -/
theorem binary_representation_properties (n : ℕ) 
  (h1 : n % 17 = 0) 
  (h2 : count_ones n = 3) : 
  (count_zeros n ≥ 6) ∧ 
  (count_zeros n = 7 → Even n) := by sorry

end NUMINAMATH_CALUDE_binary_representation_properties_l3593_359315


namespace NUMINAMATH_CALUDE_two_triangles_with_perimeter_8_l3593_359310

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 8
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- The set of all valid IntTriangles -/
def validTriangles : Set IntTriangle :=
  {t : IntTriangle | t.a > 0 ∧ t.b > 0 ∧ t.c > 0}

/-- Two triangles are considered the same if they have the same multiset of side lengths -/
def sameTriangle (t1 t2 : IntTriangle) : Prop :=
  Multiset.ofList [t1.a, t1.b, t1.c] = Multiset.ofList [t2.a, t2.b, t2.c]

theorem two_triangles_with_perimeter_8 :
    ∃ (t1 t2 : IntTriangle),
      t1 ∈ validTriangles ∧ 
      t2 ∈ validTriangles ∧ 
      ¬(sameTriangle t1 t2) ∧
      ∀ (t : IntTriangle), t ∈ validTriangles → 
        (sameTriangle t t1 ∨ sameTriangle t t2) :=
  sorry

end NUMINAMATH_CALUDE_two_triangles_with_perimeter_8_l3593_359310


namespace NUMINAMATH_CALUDE_flower_count_l3593_359339

theorem flower_count (vase_capacity : ℝ) (carnation_count : ℝ) (vases_needed : ℝ) :
  vase_capacity = 6.0 →
  carnation_count = 7.0 →
  vases_needed = 6.666666667 →
  (vases_needed * vase_capacity + carnation_count : ℝ) = 47.0 := by
  sorry

end NUMINAMATH_CALUDE_flower_count_l3593_359339


namespace NUMINAMATH_CALUDE_julie_school_year_work_hours_l3593_359314

/-- Julie's summer work scenario -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work scenario -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate required hours per week for school year -/
def requiredHoursPerWeek (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyWage := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let totalHours := schoolYear.targetEarnings / hourlyWage
  totalHours / schoolYear.weeks

/-- Theorem: Julie needs to work 15 hours per week during school year -/
theorem julie_school_year_work_hours 
  (summer : SummerWork) 
  (schoolYear : SchoolYearWork) 
  (h1 : summer.hoursPerWeek = 60)
  (h2 : summer.weeks = 10)
  (h3 : summer.earnings = 6000)
  (h4 : schoolYear.weeks = 40)
  (h5 : schoolYear.targetEarnings = 6000) : 
  requiredHoursPerWeek summer schoolYear = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_work_hours_l3593_359314


namespace NUMINAMATH_CALUDE_baseball_card_ratio_l3593_359351

/-- Proves the ratio of cards Maria took to initial cards is 8:5 --/
theorem baseball_card_ratio : 
  ∀ (initial final maria_taken peter_given : ℕ),
  initial = 15 →
  peter_given = 1 →
  final = 18 →
  maria_taken = 3 * (initial - peter_given) - final →
  (maria_taken : ℚ) / initial = 8 / 5 := by
    sorry

end NUMINAMATH_CALUDE_baseball_card_ratio_l3593_359351


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3593_359328

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 = 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 0 ∧ x < 1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | x = -2 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3593_359328


namespace NUMINAMATH_CALUDE_z_value_and_quadrant_l3593_359323

def z : ℂ := (1 + Complex.I) * (3 - 2 * Complex.I)

theorem z_value_and_quadrant :
  z = 5 + Complex.I ∧ Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_value_and_quadrant_l3593_359323


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3593_359349

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 by 24 inches
    and the other has a length of 2 inches, prove that the width of the second rectangle is 60 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length : ℕ)
    (jordan_width : ℕ) (h1 : carol_length = 5) (h2 : carol_width = 24) (h3 : jordan_length = 2)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 60 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3593_359349


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_edge_sum_equality_l3593_359312

/-- 
Given a rectangular parallelepiped with edge lengths a, b, and c,
the sum of the squares of its four space diagonals is equal to
the sum of the squares of all its edges.
-/
theorem parallelepiped_diagonal_edge_sum_equality 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  4 * (a^2 + b^2 + c^2) = 4 * a^2 + 4 * b^2 + 4 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_diagonal_edge_sum_equality_l3593_359312


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3593_359317

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference :
  ∀ a₁ : ℝ, ∃ d : ℝ,
    let a := arithmeticSequence a₁ d
    (a 4 = 6) ∧ (a 3 + a 5 = a 10) → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3593_359317


namespace NUMINAMATH_CALUDE_sin_comparison_l3593_359309

theorem sin_comparison : 
  (∀ x ∈ Set.Icc (-π/2) 0, Monotone fun y ↦ Real.sin y) →
  -π/18 ∈ Set.Icc (-π/2) 0 →
  -π/10 ∈ Set.Icc (-π/2) 0 →
  Real.sin (-π/18) > Real.sin (-π/10) := by
  sorry

end NUMINAMATH_CALUDE_sin_comparison_l3593_359309


namespace NUMINAMATH_CALUDE_solution_sum_l3593_359391

/-- The solutions of the quadratic equation 2x(5x-11) = -10 -/
def solutions (x : ℝ) : Prop :=
  2 * x * (5 * x - 11) = -10

/-- The rational form of the solutions -/
def rational_form (m n p : ℤ) (x : ℝ) : Prop :=
  (x = (m + Real.sqrt n) / p) ∨ (x = (m - Real.sqrt n) / p)

/-- The theorem statement -/
theorem solution_sum (m n p : ℤ) :
  (∀ x, solutions x → rational_form m n p x) →
  Int.gcd m (Int.gcd n p) = 1 →
  m + n + p = 242 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l3593_359391


namespace NUMINAMATH_CALUDE_fuel_distribution_l3593_359393

def total_fuel : ℝ := 60

theorem fuel_distribution (second_third : ℝ) (final_third : ℝ) 
  (h1 : second_third = total_fuel / 3)
  (h2 : final_third = second_third / 2)
  (h3 : second_third + final_third + (total_fuel - second_third - final_third) = total_fuel) :
  total_fuel - second_third - final_third = 30 := by
sorry

end NUMINAMATH_CALUDE_fuel_distribution_l3593_359393


namespace NUMINAMATH_CALUDE_number_of_violas_proof_l3593_359373

/-- The number of violas in a music store, given the following conditions:
  * There are 800 cellos in the store
  * There are 70 cello-viola pairs made from the same tree
  * The probability of randomly choosing a cello-viola pair from the same tree is 0.00014583333333333335
-/
def number_of_violas : ℕ :=
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  600

theorem number_of_violas_proof :
  let total_cellos : ℕ := 800
  let same_tree_pairs : ℕ := 70
  let probability : ℚ := 70 / (800 * 600)
  number_of_violas = 600 := by
  sorry

end NUMINAMATH_CALUDE_number_of_violas_proof_l3593_359373


namespace NUMINAMATH_CALUDE_line_equation_problem_1_line_equation_problem_2_l3593_359374

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statements
theorem line_equation_problem_1 (l : Line) (A : Point) :
  A.x = 0 ∧ A.y = 2 ∧ 
  (l.a^2 / (l.a^2 + l.b^2) = 1/4) →
  ∃ (k : ℝ), k > 0 ∧ l.a = k * Real.sqrt 3 ∧ l.b = -3 * k ∧ l.c = 6 * k :=
sorry

theorem line_equation_problem_2 (l l₁ : Line) (A : Point) :
  A.x = 2 ∧ A.y = 1 ∧
  l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = 5 ∧
  (l.a / l.b = (l₁.a / l₁.b) / 2) →
  ∃ (k : ℝ), k > 0 ∧ l.a = 3 * k ∧ l.b = -k ∧ l.c = -5 * k :=
sorry

end NUMINAMATH_CALUDE_line_equation_problem_1_line_equation_problem_2_l3593_359374


namespace NUMINAMATH_CALUDE_real_part_zero_necessary_not_sufficient_l3593_359342

/-- A complex number is purely imaginary if and only if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition "real part is zero" is necessary but not sufficient for a complex number to be purely imaginary. -/
theorem real_part_zero_necessary_not_sufficient :
  ∀ (a b : ℝ), 
    (∀ (z : ℂ), is_purely_imaginary z → z.re = 0) ∧
    ¬(∀ (z : ℂ), z.re = 0 → is_purely_imaginary z) :=
by sorry

end NUMINAMATH_CALUDE_real_part_zero_necessary_not_sufficient_l3593_359342


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l3593_359360

theorem arithmetic_sequence_count (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = 2014) (h2 : m = 315) (h3 : k = 5490) :
  (∃ (sequences : Finset (Finset ℕ)),
    sequences.card = k ∧
    (∀ seq ∈ sequences,
      seq.card = m ∧
      (∃ d : ℕ, d > 0 ∧ d ≤ 6 ∧
        (∀ i j : ℕ, i < j → i ∈ seq → j ∈ seq →
          ∃ k : ℕ, j - i = k * d)) ∧
      1 ∈ seq ∧
      (∀ x ∈ seq, 1 ≤ x ∧ x ≤ n))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l3593_359360


namespace NUMINAMATH_CALUDE_rebuild_points_l3593_359303

-- Define a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define symmetry with respect to a point
def symmetric (p1 p2 center : Point) : Prop :=
  center.x = (p1.x + p2.x) / 2 ∧ center.y = (p1.y + p2.y) / 2

theorem rebuild_points (A' B' C' D' : Point) :
  ∃! (A B C D : Point),
    symmetric A A' B ∧
    symmetric B B' C ∧
    symmetric C C' D ∧
    symmetric D D' A :=
  sorry

end NUMINAMATH_CALUDE_rebuild_points_l3593_359303


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3593_359367

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 40 * x - 56 = (d * x + e)^2 + f) →
  d * e = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3593_359367


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3593_359361

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x + Real.sqrt (x + 6) = 12 → x = 529 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3593_359361


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3593_359377

theorem min_reciprocal_sum (x y : ℝ) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ a b : ℝ, Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3593_359377


namespace NUMINAMATH_CALUDE_covered_number_is_eight_l3593_359395

/-- A circular arrangement of 15 numbers -/
def CircularArrangement := Fin 15 → ℕ

/-- The property that the sum of any six consecutive numbers is 50 -/
def SumProperty (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 15, (arr i + arr (i + 1) + arr (i + 2) + arr (i + 3) + arr (i + 4) + arr (i + 5)) = 50

/-- The property that two adjacent numbers are 7 and 10 with a number between them -/
def AdjacentProperty (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 2) = 10

theorem covered_number_is_eight (arr : CircularArrangement) 
  (h1 : SumProperty arr) (h2 : AdjacentProperty arr) : 
  ∃ i : Fin 15, arr i = 7 ∧ arr (i + 1) = 8 ∧ arr (i + 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_covered_number_is_eight_l3593_359395


namespace NUMINAMATH_CALUDE_work_absence_problem_l3593_359320

theorem work_absence_problem (total_days : ℕ) (daily_wage : ℕ) (daily_fine : ℕ) (total_received : ℕ) :
  total_days = 30 →
  daily_wage = 10 →
  daily_fine = 2 →
  total_received = 216 →
  ∃ (absent_days : ℕ),
    absent_days = 7 ∧
    total_received = daily_wage * (total_days - absent_days) - daily_fine * absent_days :=
by sorry

end NUMINAMATH_CALUDE_work_absence_problem_l3593_359320


namespace NUMINAMATH_CALUDE_prob_not_shaded_is_500_1001_l3593_359378

/-- Represents a 2 by 1001 rectangle with middle squares shaded -/
structure ShadedRectangle where
  width : ℕ := 2
  length : ℕ := 1001
  middle_shaded : ℕ := (length + 1) / 2

/-- Calculates the total number of rectangles in the figure -/
def total_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * (r.length * (r.length + 1)) / 2

/-- Calculates the number of rectangles that include a shaded square -/
def shaded_rectangles (r : ShadedRectangle) : ℕ :=
  r.width * r.middle_shaded * (r.length - r.middle_shaded + 1)

/-- The probability of choosing a rectangle that doesn't include a shaded square -/
def prob_not_shaded (r : ShadedRectangle) : ℚ :=
  1 - (shaded_rectangles r : ℚ) / (total_rectangles r : ℚ)

theorem prob_not_shaded_is_500_1001 (r : ShadedRectangle) :
  prob_not_shaded r = 500 / 1001 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_shaded_is_500_1001_l3593_359378


namespace NUMINAMATH_CALUDE_thirteen_times_fifty_in_tens_l3593_359372

theorem thirteen_times_fifty_in_tens : 13 * 50 = 65 * 10 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_times_fifty_in_tens_l3593_359372


namespace NUMINAMATH_CALUDE_sin_shift_left_l3593_359329

/-- Shifting a sinusoidal function to the left -/
theorem sin_shift_left (x : ℝ) :
  let f (t : ℝ) := Real.sin (2 * t)
  let g (t : ℝ) := f (t + π / 6)
  g x = Real.sin (2 * x + π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sin_shift_left_l3593_359329


namespace NUMINAMATH_CALUDE_heart_ratio_l3593_359326

def heart (n m : ℝ) : ℝ := n^4 * m^3

theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l3593_359326


namespace NUMINAMATH_CALUDE_derivative_of_exp_sin_l3593_359325

theorem derivative_of_exp_sin (x : ℝ) :
  deriv (fun x => Real.exp x * Real.sin x) x = Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_sin_l3593_359325


namespace NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l3593_359321

/-- Given a sphere with initial surface area 400π cm² and radius increased by 2 cm, 
    prove that the new volume is 2304π cm³ -/
theorem sphere_volume_after_radius_increase :
  ∀ (r : ℝ), 
    (4 * π * r^2 = 400 * π) →  -- Initial surface area condition
    ((4 / 3) * π * (r + 2)^3 = 2304 * π) -- New volume after radius increase
:= by sorry

end NUMINAMATH_CALUDE_sphere_volume_after_radius_increase_l3593_359321


namespace NUMINAMATH_CALUDE_mountain_climb_time_l3593_359359

/-- Proves that the time to go up the mountain is 2 hours given the specified conditions -/
theorem mountain_climb_time 
  (total_time : ℝ) 
  (uphill_speed downhill_speed : ℝ) 
  (route_difference : ℝ) :
  total_time = 4 →
  uphill_speed = 3 →
  downhill_speed = 4 →
  route_difference = 2 →
  ∃ (uphill_time : ℝ),
    uphill_time = 2 ∧
    ∃ (downhill_time uphill_distance downhill_distance : ℝ),
      uphill_time + downhill_time = total_time ∧
      uphill_distance / uphill_speed = uphill_time ∧
      downhill_distance / downhill_speed = downhill_time ∧
      downhill_distance = uphill_distance + route_difference :=
by sorry

end NUMINAMATH_CALUDE_mountain_climb_time_l3593_359359


namespace NUMINAMATH_CALUDE_weight_of_three_moles_CaI2_l3593_359356

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The weight of n moles of CaI2 in grams -/
def weight_CaI2 (n : ℝ) : ℝ := n * molecular_weight_CaI2

theorem weight_of_three_moles_CaI2 : 
  weight_CaI2 3 = 881.64 := by sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_CaI2_l3593_359356


namespace NUMINAMATH_CALUDE_linear_correlation_classification_l3593_359388

-- Define the relationships
def parent_child_height : ℝ → ℝ := sorry
def cylinder_volume_radius : ℝ → ℝ := sorry
def car_weight_fuel_efficiency : ℝ → ℝ := sorry
def household_income_expenditure : ℝ → ℝ := sorry

-- Define linear correlation
def is_linearly_correlated (f : ℝ → ℝ) : Prop := 
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

-- Theorem statement
theorem linear_correlation_classification :
  is_linearly_correlated parent_child_height ∧
  is_linearly_correlated car_weight_fuel_efficiency ∧
  is_linearly_correlated household_income_expenditure ∧
  ¬ is_linearly_correlated cylinder_volume_radius :=
sorry

end NUMINAMATH_CALUDE_linear_correlation_classification_l3593_359388


namespace NUMINAMATH_CALUDE_amount_of_c_l3593_359369

theorem amount_of_c (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 := by
sorry

end NUMINAMATH_CALUDE_amount_of_c_l3593_359369


namespace NUMINAMATH_CALUDE_paths_amc9_count_l3593_359322

/-- Represents the number of paths to spell "AMC9" in the grid -/
def pathsAMC9 (m_from_a : Nat) (c_from_m : Nat) (nine_from_c : Nat) : Nat :=
  m_from_a * c_from_m * nine_from_c

/-- Theorem stating that the number of paths to spell "AMC9" is 36 -/
theorem paths_amc9_count :
  pathsAMC9 4 3 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_paths_amc9_count_l3593_359322


namespace NUMINAMATH_CALUDE_solution_y_percent_a_l3593_359396

/-- Represents a chemical solution with a given percentage of chemical A -/
structure Solution where
  percent_a : ℝ
  h_percent_range : 0 ≤ percent_a ∧ percent_a ≤ 1

/-- Represents a mixture of two solutions -/
structure Mixture where
  solution_x : Solution
  solution_y : Solution
  proportion_x : ℝ
  h_proportion_range : 0 ≤ proportion_x ∧ proportion_x ≤ 1

/-- Calculates the percentage of chemical A in a mixture -/
def mixture_percent_a (m : Mixture) : ℝ :=
  m.proportion_x * m.solution_x.percent_a + (1 - m.proportion_x) * m.solution_y.percent_a

theorem solution_y_percent_a (x : Solution) (y : Solution) (m : Mixture) 
  (h_x : x.percent_a = 0.3)
  (h_m : m.solution_x = x ∧ m.solution_y = y ∧ m.proportion_x = 0.8)
  (h_mixture : mixture_percent_a m = 0.32) :
  y.percent_a = 0.4 := by
  sorry


end NUMINAMATH_CALUDE_solution_y_percent_a_l3593_359396


namespace NUMINAMATH_CALUDE_product_mod_25_l3593_359387

theorem product_mod_25 : (43 * 67 * 92) % 25 = 2 := by
  sorry

#check product_mod_25

end NUMINAMATH_CALUDE_product_mod_25_l3593_359387


namespace NUMINAMATH_CALUDE_bee_multiple_l3593_359341

theorem bee_multiple (bees_day1 bees_day2 : ℕ) (h1 : bees_day1 = 144) (h2 : bees_day2 = 432) :
  bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l3593_359341


namespace NUMINAMATH_CALUDE_power_zero_plus_power_division_l3593_359331

theorem power_zero_plus_power_division (x y : ℕ) : 3^0 + 9^5 / 9^3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_plus_power_division_l3593_359331


namespace NUMINAMATH_CALUDE_decimal_digits_divisibility_l3593_359330

def repeatedDigits (a b c : ℕ) : ℕ :=
  a * (10^4006 - 10^2004) / 99 + b * 10^2002 + c * (10^2002 - 1) / 99

theorem decimal_digits_divisibility (a b c : ℕ) 
  (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) 
  (h_div : 37 ∣ repeatedDigits a b c) : 
  b = a + c := by sorry

end NUMINAMATH_CALUDE_decimal_digits_divisibility_l3593_359330


namespace NUMINAMATH_CALUDE_line_slope_l3593_359344

/-- A line in the xy-plane with y-intercept 20 and passing through (150, 600) has slope 580/150 -/
theorem line_slope (line : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ line ↔ y = (580/150) * x + 20) →
  (0, 20) ∈ line →
  (150, 600) ∈ line →
  ∃ (m : ℝ), ∀ (x y : ℝ), (x, y) ∈ line ↔ y = m * x + 20 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3593_359344


namespace NUMINAMATH_CALUDE_inequality_proof_l3593_359301

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h : a + b < c + d) : 
  a * c + b * d > a * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3593_359301


namespace NUMINAMATH_CALUDE_parallelogram_count_l3593_359364

/-- The number of ways to choose 2 items from 4 -/
def choose_2_from_4 : ℕ := 6

/-- The number of horizontal lines -/
def horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def vertical_lines : ℕ := 4

/-- The number of parallelograms formed -/
def num_parallelograms : ℕ := choose_2_from_4 * choose_2_from_4

theorem parallelogram_count : num_parallelograms = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_count_l3593_359364


namespace NUMINAMATH_CALUDE_circles_properties_l3593_359306

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the tangent line
def tangent_line (x : ℝ) : Prop := x = -2

theorem circles_properties :
  (∀ x y : ℝ, C1 x y ∧ C2 x y → common_chord x y) ∧
  (∀ x y : ℝ, (C1 x y ∧ tangent_line x) ∨ (C2 x y ∧ tangent_line x) →
    ∃! t : ℝ, (x = -2 ∧ y = t) ∧ (C1 x y ∨ C2 x y)) :=
sorry

end NUMINAMATH_CALUDE_circles_properties_l3593_359306
