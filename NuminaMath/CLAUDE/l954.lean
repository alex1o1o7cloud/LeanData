import Mathlib

namespace NUMINAMATH_CALUDE_probability_ratio_l954_95413

def total_cards : ℕ := 60
def numbers_per_card : ℕ := 12
def cards_per_number : ℕ := 5
def cards_drawn : ℕ := 5

def probability_same_number (n : ℕ) : ℚ :=
  (numbers_per_card : ℚ) / (total_cards.choose cards_drawn)

def probability_four_and_one (n : ℕ) : ℚ :=
  ((numbers_per_card * (numbers_per_card - 1) * (cards_per_number.choose 4) * cards_per_number) : ℚ) / 
  (total_cards.choose cards_drawn)

theorem probability_ratio : 
  (probability_four_and_one total_cards) / (probability_same_number total_cards) = 275 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l954_95413


namespace NUMINAMATH_CALUDE_negative_one_odd_power_l954_95490

theorem negative_one_odd_power (n : ℕ) (h : Odd n) : (-1 : ℤ) ^ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_odd_power_l954_95490


namespace NUMINAMATH_CALUDE_contest_result_l954_95400

/-- The total number of baskets made by Alex, Sandra, and Hector -/
def totalBaskets (alex sandra hector : ℕ) : ℕ := alex + sandra + hector

/-- Theorem: Given the conditions, the total number of baskets is 80 -/
theorem contest_result : ∃ (sandra hector : ℕ),
  sandra = 3 * 8 ∧ 
  hector = 2 * sandra ∧
  totalBaskets 8 sandra hector = 80 := by
  sorry

end NUMINAMATH_CALUDE_contest_result_l954_95400


namespace NUMINAMATH_CALUDE_phone_purchase_problem_l954_95464

/-- Represents the purchase price of phone models -/
structure PhonePrices where
  modelA : ℕ
  modelB : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  modelACount : ℕ
  modelBCount : ℕ

def totalCost (prices : PhonePrices) (plan : PurchasePlan) : ℕ :=
  prices.modelA * plan.modelACount + prices.modelB * plan.modelBCount

theorem phone_purchase_problem (prices : PhonePrices) : 
  (prices.modelA * 2 + prices.modelB = 5000) →
  (prices.modelA * 3 + prices.modelB * 2 = 8000) →
  (prices.modelA = 2000 ∧ prices.modelB = 1000) ∧
  (∃ (plans : List PurchasePlan), 
    plans.length = 3 ∧
    (∀ plan ∈ plans, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000) ∧
    (∀ plan : PurchasePlan, 
      plan.modelACount + plan.modelBCount = 20 ∧
      24000 ≤ totalCost prices plan ∧
      totalCost prices plan ≤ 26000 →
      plan ∈ plans)) :=
by sorry

end NUMINAMATH_CALUDE_phone_purchase_problem_l954_95464


namespace NUMINAMATH_CALUDE_expression_simplification_l954_95466

/-- Proves that the simplification of 7y + 8 - 3y + 15 + 2x is equivalent to 4y + 2x + 23 -/
theorem expression_simplification (x y : ℝ) :
  7 * y + 8 - 3 * y + 15 + 2 * x = 4 * y + 2 * x + 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l954_95466


namespace NUMINAMATH_CALUDE_man_and_son_work_time_l954_95406

/-- The time taken for a man and his son to complete a task together, given their individual completion times -/
theorem man_and_son_work_time (man_time son_time : ℝ) (h1 : man_time = 5) (h2 : son_time = 20) :
  1 / (1 / man_time + 1 / son_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_man_and_son_work_time_l954_95406


namespace NUMINAMATH_CALUDE_jack_age_l954_95418

/-- Given that Jack's age is 20 years less than twice Jane's age,
    and the sum of their ages is 60, prove that Jack is 33 years old. -/
theorem jack_age (j a : ℕ) 
  (h1 : j = 2 * a - 20)  -- Jack's age is 20 years less than twice Jane's age
  (h2 : j + a = 60)      -- The sum of their ages is 60
  : j = 33 := by
  sorry

end NUMINAMATH_CALUDE_jack_age_l954_95418


namespace NUMINAMATH_CALUDE_triangle_problem_l954_95445

theorem triangle_problem (A B C : Real) (a b c : Real) :
  b = 3 * Real.sqrt 2 →
  Real.cos A = Real.sqrt 6 / 3 →
  B = A + π / 2 →
  a = 3 ∧ Real.cos (2 * C) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l954_95445


namespace NUMINAMATH_CALUDE_total_crayons_correct_l954_95481

/-- The total number of crayons in a box -/
def total_crayons : ℕ := 24

/-- The number of red crayons -/
def red_crayons : ℕ := 8

/-- The number of blue crayons -/
def blue_crayons : ℕ := 6

/-- The number of green crayons -/
def green_crayons : ℕ := 4

/-- The number of pink crayons -/
def pink_crayons : ℕ := 6

/-- Theorem stating that the total number of crayons is correct -/
theorem total_crayons_correct :
  total_crayons = red_crayons + blue_crayons + green_crayons + pink_crayons ∧
  green_crayons = (2 * blue_crayons) / 3 :=
by sorry

end NUMINAMATH_CALUDE_total_crayons_correct_l954_95481


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l954_95408

theorem janabel_widget_sales :
  let a : ℕ → ℕ := fun n => 2 * n - 1
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 20 = 400 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l954_95408


namespace NUMINAMATH_CALUDE_only_set_C_forms_triangle_l954_95486

/-- A function that checks if three numbers can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The sets of lengths given in the problem --/
def set_A : List ℝ := [3, 4, 8]
def set_B : List ℝ := [8, 7, 15]
def set_C : List ℝ := [13, 12, 20]
def set_D : List ℝ := [5, 5, 11]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
by sorry

end NUMINAMATH_CALUDE_only_set_C_forms_triangle_l954_95486


namespace NUMINAMATH_CALUDE_simplify_squared_terms_l954_95478

theorem simplify_squared_terms (a : ℝ) : 2 * a^2 - 3 * a^2 = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_squared_terms_l954_95478


namespace NUMINAMATH_CALUDE_linear_inequalities_solution_sets_l954_95454

theorem linear_inequalities_solution_sets :
  (∀ x : ℝ, (4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3) ↔ (-2 ≤ x ∧ x < 7 / 2)) ∧
  (∀ x : ℝ, (x - 3 * (x - 2) ≥ 4 ∧ (2 * x - 1) / 5 ≥ (x + 1) / 2) ↔ x ≤ -7) :=
by sorry

end NUMINAMATH_CALUDE_linear_inequalities_solution_sets_l954_95454


namespace NUMINAMATH_CALUDE_farm_animals_product_l954_95425

theorem farm_animals_product (pigs chickens : ℕ) : 
  chickens = pigs + 12 →
  chickens + pigs = 52 →
  pigs * chickens = 640 :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_product_l954_95425


namespace NUMINAMATH_CALUDE_modulus_of_z_l954_95470

theorem modulus_of_z (i z : ℂ) (hi : i * i = -1) (hz : i * z = 3 + 4 * i) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l954_95470


namespace NUMINAMATH_CALUDE_pencils_purchased_l954_95446

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : num_pens = 30)
  (h2 : total_cost = 510)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 12) :
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l954_95446


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l954_95483

theorem largest_of_seven_consecutive_integers (n : ℕ) : 
  (n > 0) →
  (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) = 3010) →
  (n+6 = 433) :=
by sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l954_95483


namespace NUMINAMATH_CALUDE_sphere_wedge_volume_l954_95447

theorem sphere_wedge_volume (c : ℝ) (h1 : c = 16 * Real.pi) : 
  let r := c / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * r^3
  let wedge_volume := sphere_volume / 8
  wedge_volume = (256 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_wedge_volume_l954_95447


namespace NUMINAMATH_CALUDE_part1_part2_part3_l954_95440

-- Define the functions f and g
def f (x : ℝ) := x - 2
def g (m : ℝ) (x : ℝ) := x^2 - 2*m*x + 4

-- Part 1
theorem part1 (m : ℝ) :
  (∀ x, g m x > f x) ↔ m ∈ Set.Ioo (-Real.sqrt 6 - 1/2) (Real.sqrt 6 - 1/2) :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g m x₁ = f x₂) ↔ 
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
sorry

-- Part 3
theorem part3 :
  (∀ n : ℝ, ∃ x₀ ∈ Set.Icc (-2) 2, |g (-1) x₀ - x₀^2 + n| ≥ k) ↔
  k ∈ Set.Iic 4 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l954_95440


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l954_95496

theorem decimal_to_fraction : (2.375 : ℚ) = 19 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l954_95496


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_quartic_equation_solutions_l954_95444

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ 2*x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 - Real.sqrt 6 / 2 ∧ 
             x₂ = -1 + Real.sqrt 6 / 2 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

theorem quartic_equation_solutions :
  let g : ℝ → ℝ := λ x ↦ 4*(2*x - 1)^2 - 9*(x + 4)^2
  ∃ x₁ x₂ : ℝ, x₁ = -8/11 ∧ 
             x₂ = 16/5 ∧ 
             g x₁ = 0 ∧ g x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_quartic_equation_solutions_l954_95444


namespace NUMINAMATH_CALUDE_first_group_size_l954_95465

/-- Represents the rate of work in meters of cloth colored per man per day -/
def rate_of_work (men : ℕ) (length : ℕ) (days : ℕ) : ℚ :=
  (length : ℚ) / ((men : ℚ) * (days : ℚ))

theorem first_group_size (m : ℕ) : 
  rate_of_work m 48 2 = rate_of_work 2 36 3 → m = 4 := by
  sorry

#check first_group_size

end NUMINAMATH_CALUDE_first_group_size_l954_95465


namespace NUMINAMATH_CALUDE_parallelogram_area_l954_95449

theorem parallelogram_area (a b : ℝ) (θ : ℝ) 
  (ha : a = 20) (hb : b = 10) (hθ : θ = 150 * π / 180) : 
  a * b * Real.sin θ = 100 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l954_95449


namespace NUMINAMATH_CALUDE_midpoint_after_translation_l954_95460

/-- Given a triangle DJH with vertices D(2, 3), J(3, 7), and H(7, 3),
    prove that the midpoint of D'H' after translating the triangle
    3 units right and 1 unit down is (7.5, 2). -/
theorem midpoint_after_translation :
  let D : ℝ × ℝ := (2, 3)
  let J : ℝ × ℝ := (3, 7)
  let H : ℝ × ℝ := (7, 3)
  let translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 - 1)
  let D' := translate D
  let H' := translate H
  let midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
  midpoint D' H' = (7.5, 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_after_translation_l954_95460


namespace NUMINAMATH_CALUDE_library_book_count_l954_95471

/-- The number of books in the library after a series of transactions --/
def books_in_library (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (taken_thursday : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - taken_thursday

/-- Theorem: The number of books in the library after the given transactions is 150 --/
theorem library_book_count : 
  books_in_library 250 120 35 15 = 150 := by
  sorry

#eval books_in_library 250 120 35 15

end NUMINAMATH_CALUDE_library_book_count_l954_95471


namespace NUMINAMATH_CALUDE_granola_bar_cost_l954_95438

/-- Calculates the total cost of granola bars after applying a discount --/
def total_cost_after_discount (
  oatmeal_quantity : ℕ)
  (oatmeal_price : ℚ)
  (peanut_quantity : ℕ)
  (peanut_price : ℚ)
  (chocolate_quantity : ℕ)
  (chocolate_price : ℚ)
  (mixed_quantity : ℕ)
  (mixed_price : ℚ)
  (discount_percentage : ℚ) : ℚ :=
  let total_before_discount := 
    oatmeal_quantity * oatmeal_price +
    peanut_quantity * peanut_price +
    chocolate_quantity * chocolate_price +
    mixed_quantity * mixed_price
  let discount_amount := (discount_percentage / 100) * total_before_discount
  total_before_discount - discount_amount

/-- Theorem stating the total cost after discount for the given problem --/
theorem granola_bar_cost : 
  total_cost_after_discount 6 (25/20) 8 (3/2) 5 (7/4) 3 2 15 = 2911/100 :=
sorry

end NUMINAMATH_CALUDE_granola_bar_cost_l954_95438


namespace NUMINAMATH_CALUDE_marble_ratio_l954_95491

theorem marble_ratio (total : ℕ) (red : ℕ) (dark_blue : ℕ) 
  (h1 : total = 63) 
  (h2 : red = 38) 
  (h3 : dark_blue = 6) :
  (total - red - dark_blue) / red = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l954_95491


namespace NUMINAMATH_CALUDE_inequality_system_solution_l954_95463

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 6 < 4*x - 3 ∧ x > m) ↔ x > 3) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l954_95463


namespace NUMINAMATH_CALUDE_simplify_expression_l954_95461

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.75) ^ 2 = (87 - 12 * Real.sqrt 51) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l954_95461


namespace NUMINAMATH_CALUDE_burger_share_length_l954_95409

-- Define the length of a foot in inches
def foot_in_inches : ℕ := 12

-- Define the burger length in feet
def burger_length_feet : ℕ := 1

-- Define the number of people sharing the burger
def num_people : ℕ := 2

-- Theorem to prove
theorem burger_share_length :
  (burger_length_feet * foot_in_inches) / num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_burger_share_length_l954_95409


namespace NUMINAMATH_CALUDE_sum_and_product_identities_l954_95495

theorem sum_and_product_identities (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -1) : 
  (a^2 + b^2 = 6) ∧ ((a - b)^2 = 8) := by sorry

end NUMINAMATH_CALUDE_sum_and_product_identities_l954_95495


namespace NUMINAMATH_CALUDE_sum_c_d_equals_three_l954_95401

theorem sum_c_d_equals_three (a b c d : ℝ)
  (h1 : a + b = 12)
  (h2 : b + c = 9)
  (h3 : a + d = 6) :
  c + d = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_c_d_equals_three_l954_95401


namespace NUMINAMATH_CALUDE_seven_lines_intersections_l954_95456

/-- The maximum number of intersection points for n lines in a plane -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- The set of possible numbers of intersection points for 7 lines in a plane -/
def possible_intersections : Set ℕ :=
  {0, 1} ∪ Set.Icc 6 21

theorem seven_lines_intersections :
  (max_intersections 7 = 21) ∧
  (possible_intersections = {0, 1} ∪ Set.Icc 6 21) :=
sorry

end NUMINAMATH_CALUDE_seven_lines_intersections_l954_95456


namespace NUMINAMATH_CALUDE_cube_root_inequality_l954_95424

theorem cube_root_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.rpow (a * b) (1/3) + Real.rpow (c * d) (1/3) ≤ Real.rpow ((a + b + c) * (b + c + d)) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l954_95424


namespace NUMINAMATH_CALUDE_sin_240_degrees_l954_95467

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l954_95467


namespace NUMINAMATH_CALUDE_data_properties_l954_95426

def data : List ℕ := [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ := sorry

def fractionLessThanMode (l : List ℕ) : ℚ := sorry

def firstQuartile (l : List ℕ) : ℕ := sorry

def medianWithinFirstQuartile (l : List ℕ) : ℚ := sorry

theorem data_properties :
  fractionLessThanMode data = 4/11 ∧
  medianWithinFirstQuartile data = 4 := by sorry

end NUMINAMATH_CALUDE_data_properties_l954_95426


namespace NUMINAMATH_CALUDE_calculate_face_value_l954_95451

/-- The relationship between banker's discount, true discount, and face value -/
def bankers_discount_relation (bd td fv : ℚ) : Prop :=
  bd = td + td^2 / fv

/-- Given the banker's discount and true discount, calculate the face value -/
theorem calculate_face_value (bd td : ℚ) (h : bankers_discount_relation bd td 300) :
  bd = 72 ∧ td = 60 → 300 = 300 := by sorry

end NUMINAMATH_CALUDE_calculate_face_value_l954_95451


namespace NUMINAMATH_CALUDE_paytons_score_l954_95488

theorem paytons_score (total_students : ℕ) (students_without_payton : ℕ) 
  (avg_without_payton : ℚ) (new_avg_with_payton : ℚ) :
  total_students = 15 →
  students_without_payton = 14 →
  avg_without_payton = 80 →
  new_avg_with_payton = 81 →
  (students_without_payton * avg_without_payton + payton_score) / total_students = new_avg_with_payton →
  payton_score = 95 := by
  sorry

end NUMINAMATH_CALUDE_paytons_score_l954_95488


namespace NUMINAMATH_CALUDE_polynomial_multiplication_expansion_l954_95487

theorem polynomial_multiplication_expansion :
  ∀ (z : ℝ), (5 * z^3 + 4 * z^2 - 3 * z + 7) * (2 * z^4 - z^3 + z - 2) =
    10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_expansion_l954_95487


namespace NUMINAMATH_CALUDE_previous_year_profit_percentage_l954_95494

/-- Given a company's financial data over two years, calculate the profit percentage in the previous year. -/
theorem previous_year_profit_percentage
  (R : ℝ)  -- Revenues in the previous year
  (P : ℝ)  -- Profits in the previous year
  (h1 : 0.8 * R = R - 0.2 * R)  -- Revenues fell by 20% in 2009
  (h2 : 0.09 * (0.8 * R) = 0.072 * R)  -- Profits were 9% of revenues in 2009
  (h3 : 0.072 * R = 0.72 * P)  -- Profits in 2009 were 72% of previous year's profits
  : P / R = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_previous_year_profit_percentage_l954_95494


namespace NUMINAMATH_CALUDE_combined_average_score_l954_95412

theorem combined_average_score (score_a score_b score_c : ℝ) 
  (ratio_a ratio_b ratio_c : ℕ) : 
  score_a = 65 → score_b = 90 → score_c = 77 →
  ratio_a = 4 → ratio_b = 6 → ratio_c = 5 →
  (score_a * ratio_a + score_b * ratio_b + score_c * ratio_c) / (ratio_a + ratio_b + ratio_c) = 79 := by
sorry

end NUMINAMATH_CALUDE_combined_average_score_l954_95412


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l954_95441

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) :
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l954_95441


namespace NUMINAMATH_CALUDE_short_trees_calculation_l954_95452

/-- The number of short trees currently in the park -/
def current_short_trees : ℕ := 112

/-- The number of short trees to be planted -/
def trees_to_plant : ℕ := 105

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 217

/-- Theorem stating that the current number of short trees plus the number of trees to be planted equals the total number of short trees after planting -/
theorem short_trees_calculation :
  current_short_trees + trees_to_plant = total_short_trees := by sorry

end NUMINAMATH_CALUDE_short_trees_calculation_l954_95452


namespace NUMINAMATH_CALUDE_couples_after_dance_l954_95403

/-- The number of initial couples at the ball. -/
def n : ℕ := 2018

/-- The function that determines the source area for a couple at minute i. -/
def s (i : ℕ) : ℕ := i % n + 1

/-- The function that determines the destination area for a couple at minute i. -/
def r (i : ℕ) : ℕ := (2 * i) % n + 1

/-- Predicate to determine if a couple in area k survives after t minutes. -/
def survives (k t : ℕ) : Prop := sorry

/-- The number of couples remaining after t minutes. -/
def remaining_couples (t : ℕ) : ℕ := sorry

/-- The main theorem stating that after n² minutes, 505 couples remain. -/
theorem couples_after_dance : remaining_couples (n^2) = 505 := by sorry

end NUMINAMATH_CALUDE_couples_after_dance_l954_95403


namespace NUMINAMATH_CALUDE_abs_sum_inequality_iff_range_l954_95462

theorem abs_sum_inequality_iff_range (x : ℝ) : 
  (abs (x + 1) + abs (x - 2) ≤ 5) ↔ (-2 ≤ x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_iff_range_l954_95462


namespace NUMINAMATH_CALUDE_skylar_current_age_l954_95404

/-- Represents Skylar's donation history and age calculation -/
def skylar_age (start_age : ℕ) (annual_donation : ℕ) (total_donated : ℕ) : ℕ :=
  start_age + total_donated / annual_donation

/-- Theorem stating Skylar's current age -/
theorem skylar_current_age :
  skylar_age 13 5 105 = 34 := by
  sorry

end NUMINAMATH_CALUDE_skylar_current_age_l954_95404


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l954_95410

theorem quadratic_expression_value (x y z : ℝ) 
  (eq1 : 4*x + 2*y + z = 20)
  (eq2 : x + 4*y + 2*z = 26)
  (eq3 : 2*x + y + 4*z = 28) :
  20*x^2 + 24*x*y + 20*y^2 + 12*z^2 = 500 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l954_95410


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l954_95442

theorem sum_of_five_consecutive_squares_not_perfect_square (x : ℤ) : 
  ¬∃ (k : ℤ), 5 * (x^2 + 2) = k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_squares_not_perfect_square_l954_95442


namespace NUMINAMATH_CALUDE_trampoline_jumps_l954_95415

theorem trampoline_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) : 
  ronald_jumps = 157 → rupert_extra_jumps = 86 → 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l954_95415


namespace NUMINAMATH_CALUDE_triangle_tangent_segment_length_l954_95428

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point on a line segment -/
structure PointOnSegment where
  segment : ℝ
  position : ℝ

/-- Checks if a point is on the incircle of a triangle -/
def isOnIncircle (t : Triangle) (p : PointOnSegment) : Prop := sorry

/-- Checks if a line segment is tangent to the incircle of a triangle -/
def isTangentToIncircle (t : Triangle) (p1 p2 : PointOnSegment) : Prop := sorry

/-- Main theorem -/
theorem triangle_tangent_segment_length 
  (t : Triangle) 
  (x y : PointOnSegment) :
  t.a = 19 ∧ t.b = 20 ∧ t.c = 21 →
  x.segment = t.a ∧ y.segment = t.c →
  x.position + y.position = t.a →
  isTangentToIncircle t x y →
  x.position = 67 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_segment_length_l954_95428


namespace NUMINAMATH_CALUDE_quadrilateral_angle_equality_l954_95411

-- Define the points
variable (A B C D E F P : Point)

-- Define the quadrilateral
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that a point is on a line segment
def on_segment (X Y Z : Point) : Prop := sorry

-- Define that two lines intersect at a point
def lines_intersect_at (W X Y Z P : Point) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point) : Prop := sorry

-- State the theorem
theorem quadrilateral_angle_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment B E C)
  (h3 : on_segment C F D)
  (h4 : lines_intersect_at B F D E P)
  (h5 : angle_eq B A E F A D) :
  angle_eq B A P C A D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_equality_l954_95411


namespace NUMINAMATH_CALUDE_problem_solution_l954_95414

theorem problem_solution (x : ℝ) : (0.5 * x - (1/3) * x = 110) → x = 660 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l954_95414


namespace NUMINAMATH_CALUDE_smallest_integer_bound_l954_95427

theorem smallest_integer_bound (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d  -- Four different integers
  → d = 90  -- Largest integer is 90
  → (a + b + c + d) / 4 = 68  -- Average is 68
  → a ≥ 5  -- Smallest integer is at least 5
:= by sorry

end NUMINAMATH_CALUDE_smallest_integer_bound_l954_95427


namespace NUMINAMATH_CALUDE_no_partition_sum_product_l954_95433

theorem no_partition_sum_product (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  x * y ≠ 120 - x - y := by
sorry

end NUMINAMATH_CALUDE_no_partition_sum_product_l954_95433


namespace NUMINAMATH_CALUDE_game_sequence_repeats_a_2009_equals_65_l954_95405

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the game rules -/
def game_sequence (i : ℕ) : ℕ := 
  match i with
  | 0 => 5  -- n₁ = 5
  | i + 1 => sum_of_digits ((game_sequence i)^2 + 1)

/-- The a_i values in the sequence -/
def a_sequence (i : ℕ) : ℕ := (game_sequence i)^2 + 1

theorem game_sequence_repeats : 
  ∀ k : ℕ, k ≥ 3 → game_sequence k = game_sequence (k % 3) := sorry

theorem a_2009_equals_65 : a_sequence 2009 = 65 := by sorry

end NUMINAMATH_CALUDE_game_sequence_repeats_a_2009_equals_65_l954_95405


namespace NUMINAMATH_CALUDE_product_is_112015_l954_95474

/-- Represents a three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  distinct : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones
  non_zero : hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0
  valid_range : hundreds < 10 ∧ tens < 10 ∧ ones < 10

def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

theorem product_is_112015 (iks ksi : ThreeDigitNumber) 
  (h1 : iks.hundreds = ksi.ones ∧ iks.tens = ksi.hundreds ∧ iks.ones = ksi.tens)
  (h2 : ∃ (c i k : Nat), c ≠ i ∧ c ≠ k ∧ i ≠ k ∧ 
    c = max iks.hundreds (max iks.tens iks.ones) ∧
    c = max ksi.hundreds (max ksi.tens ksi.ones))
  (h3 : ∃ (p : Nat), p = to_nat iks * to_nat ksi ∧ 
    (∃ (d1 d2 d3 d4 d5 d6 : Nat),
      p = 100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6 ∧
      d1 = c ∧ d2 = c ∧ d3 = c ∧
      ((d4 = i ∧ d5 = k ∧ d6 = 0) ∨ 
       (d4 = i ∧ d5 = 0 ∧ d6 = k) ∨ 
       (d4 = k ∧ d5 = i ∧ d6 = 0) ∨ 
       (d4 = k ∧ d5 = 0 ∧ d6 = i) ∨ 
       (d4 = 0 ∧ d5 = i ∧ d6 = k) ∨ 
       (d4 = 0 ∧ d5 = k ∧ d6 = i))))
  : to_nat iks * to_nat ksi = 112015 := by
  sorry

end NUMINAMATH_CALUDE_product_is_112015_l954_95474


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l954_95448

/-- The number of posts in the fence -/
def num_posts : ℕ := 36

/-- The width of each post in inches -/
def post_width_inches : ℕ := 6

/-- The space between adjacent posts in feet -/
def space_between_posts : ℕ := 4

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- Conversion factor from inches to feet -/
def inches_to_feet : ℚ := 1 / 12

theorem square_fence_perimeter :
  let posts_per_side : ℕ := num_posts / num_sides
  let post_width_feet : ℚ := post_width_inches * inches_to_feet
  let gaps_per_side : ℕ := posts_per_side - 1
  let side_length : ℚ := gaps_per_side * space_between_posts + posts_per_side * post_width_feet
  num_sides * side_length = 130 := by sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l954_95448


namespace NUMINAMATH_CALUDE_simplify_expressions_l954_95479

theorem simplify_expressions :
  (- (99 + 71 / 72) * 36 = - (3599 + 1 / 2)) ∧
  ((-3) * (1 / 4) - 2.5 * (-2.45) + (3 + 1 / 2) * (25 / 100) = 6 + 1 / 4) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l954_95479


namespace NUMINAMATH_CALUDE_balls_left_after_removal_l954_95482

/-- The number of balls initially in the box -/
def initial_balls : ℕ := 10

/-- The number of balls Jungkook removes -/
def removed_balls : ℕ := 3

/-- The number of balls left in the box after Jungkook's action -/
def remaining_balls : ℕ := initial_balls - removed_balls

theorem balls_left_after_removal : remaining_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_balls_left_after_removal_l954_95482


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l954_95493

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - 2 * |x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x > 1} = {x : ℝ | 2/3 < x ∧ x < 2} :=
sorry

-- Part II
theorem a_value_when_f_always_nonpositive :
  (∀ x : ℝ, f a x ≤ 0) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_f_always_nonpositive_l954_95493


namespace NUMINAMATH_CALUDE_stating_cubic_factorization_condition_l954_95498

/-- Represents a cubic equation of the form x^3 + ax^2 + bx + c = 0 -/
structure CubicEquation (α : Type) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the factored form (x^2 + m)(x + n) = 0 -/
structure FactoredForm (α : Type) [Field α] where
  m : α
  n : α

/-- 
Theorem stating the necessary and sufficient condition for a cubic equation 
to be factored into the given form
-/
theorem cubic_factorization_condition {α : Type} [Field α] (eq : CubicEquation α) :
  (∃ (ff : FactoredForm α), 
    ∀ (x : α), x^3 + eq.a * x^2 + eq.b * x + eq.c = 0 ↔ 
    (x^2 + ff.m) * (x + ff.n) = 0) ↔ 
  eq.c = eq.a * eq.b :=
sorry

end NUMINAMATH_CALUDE_stating_cubic_factorization_condition_l954_95498


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l954_95499

theorem triangle_radius_inequality (a b c R r : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hR : 0 < R) (hr : 0 < r)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circum : 4 * R * (a * b * c) = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))
  (h_inradius : r * (a + b + c) = 2 * (a * b * c) / (a + b + c)) :
  1 / R^2 ≤ 1 / a^2 + 1 / b^2 + 1 / c^2 ∧ 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≤ 1 / (2 * r)^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l954_95499


namespace NUMINAMATH_CALUDE_magic_mike_calculation_l954_95458

/-- The problem statement --/
theorem magic_mike_calculation (p q r s t : ℝ) : 
  p = 3 ∧ q = 4 ∧ r = 5 ∧ s = 6 →
  (p - q + r * s - t = p - (q - (r * (s - t)))) →
  t = 0 := by
sorry

end NUMINAMATH_CALUDE_magic_mike_calculation_l954_95458


namespace NUMINAMATH_CALUDE_equation_solution_l954_95439

theorem equation_solution :
  let f := fun x : ℝ => 2 / x - (3 / x) * (5 / x) + 1 / 2
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧
    x₁ = -2 + Real.sqrt 34 ∧ 
    x₂ = -2 - Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l954_95439


namespace NUMINAMATH_CALUDE_negation_of_existence_squared_nonpositive_l954_95476

theorem negation_of_existence_squared_nonpositive :
  (¬ ∃ x : ℝ, x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_squared_nonpositive_l954_95476


namespace NUMINAMATH_CALUDE_bags_filled_on_sunday_l954_95402

/-- Given the total number of cans collected, cans per bag, and bags filled on Saturday,
    calculate the number of bags filled on Sunday. -/
theorem bags_filled_on_sunday
  (total_cans : ℕ)
  (cans_per_bag : ℕ)
  (bags_on_saturday : ℕ)
  (h1 : total_cans = 63)
  (h2 : cans_per_bag = 9)
  (h3 : bags_on_saturday = 3) :
  total_cans / cans_per_bag - bags_on_saturday = 4 := by
  sorry

end NUMINAMATH_CALUDE_bags_filled_on_sunday_l954_95402


namespace NUMINAMATH_CALUDE_missing_figure_proof_l954_95420

theorem missing_figure_proof (x : ℝ) : (0.1 / 100) * x = 0.24 → x = 240 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l954_95420


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l954_95455

def coin_flips : ℕ := 12
def desired_heads : ℕ := 9

theorem probability_nine_heads_in_twelve_flips :
  (Nat.choose coin_flips desired_heads : ℚ) / (2 ^ coin_flips) = 220 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_l954_95455


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l954_95419

theorem imaginary_part_of_reciprocal (z : ℂ) (h : z = 1 - 2*I) : 
  Complex.im (z⁻¹) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l954_95419


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l954_95421

theorem cubic_equation_roots (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, x^3 - 9*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 50 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l954_95421


namespace NUMINAMATH_CALUDE_power_fraction_equality_l954_95459

theorem power_fraction_equality : (88888 ^ 5 : ℚ) / (22222 ^ 5) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l954_95459


namespace NUMINAMATH_CALUDE_first_worker_load_time_l954_95436

/-- The time it takes for two workers to load a truck together -/
def combined_time : ℝ := 3.0769230769230766

/-- The time it takes for the second worker to load a truck alone -/
def second_worker_time : ℝ := 8

/-- The time it takes for the first worker to load a truck alone -/
def first_worker_time : ℝ := 5

/-- Theorem stating that given the combined time and the second worker's time, 
    the first worker's time to load the truck alone is 5 hours -/
theorem first_worker_load_time : 
  1 / first_worker_time + 1 / second_worker_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_first_worker_load_time_l954_95436


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l954_95443

theorem probability_at_least_one_correct (p : ℝ) (h1 : p = 1/2) :
  1 - (1 - p)^3 = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l954_95443


namespace NUMINAMATH_CALUDE_probability_of_event_A_l954_95430

/-- A tetrahedron with faces numbered 0, 1, 2, and 3 -/
inductive TetrahedronFace
| Zero
| One
| Two
| Three

/-- The result of throwing the tetrahedron twice -/
structure ThrowResult where
  first : TetrahedronFace
  second : TetrahedronFace

/-- Convert TetrahedronFace to a natural number -/
def faceToNat (face : TetrahedronFace) : Nat :=
  match face with
  | TetrahedronFace.Zero => 0
  | TetrahedronFace.One => 1
  | TetrahedronFace.Two => 2
  | TetrahedronFace.Three => 3

/-- Event A: m^2 + n^2 ≤ 4 -/
def eventA (result : ThrowResult) : Prop :=
  let m := faceToNat result.first
  let n := faceToNat result.second
  m^2 + n^2 ≤ 4

/-- The probability of event A occurring -/
def probabilityOfEventA : ℚ := 3/8

theorem probability_of_event_A :
  probabilityOfEventA = 3/8 := by sorry

end NUMINAMATH_CALUDE_probability_of_event_A_l954_95430


namespace NUMINAMATH_CALUDE_pencil_cost_l954_95407

/-- Calculates the cost of a pencil given shopping information -/
theorem pencil_cost (initial_amount : ℚ) (hat_cost : ℚ) (num_cookies : ℕ) (cookie_cost : ℚ) (remaining_amount : ℚ) : 
  initial_amount = 20 →
  hat_cost = 10 →
  num_cookies = 4 →
  cookie_cost = 5/4 →
  remaining_amount = 3 →
  initial_amount - (hat_cost + num_cookies * cookie_cost + remaining_amount) = 2 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l954_95407


namespace NUMINAMATH_CALUDE_base_equality_l954_95416

/-- Given a positive integer b, converts the base-b number 101ᵦ to base 10 -/
def base_b_to_decimal (b : ℕ) : ℕ := b^2 + 1

/-- Converts 24₅ to base 10 -/
def base_5_to_decimal : ℕ := 2 * 5 + 4

/-- The theorem states that 4 is the unique positive integer b that satisfies 24₅ = 101ᵦ -/
theorem base_equality : ∃! (b : ℕ), b > 0 ∧ base_5_to_decimal = base_b_to_decimal b :=
sorry

end NUMINAMATH_CALUDE_base_equality_l954_95416


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l954_95469

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x - 1) - 2

-- Theorem statement
theorem shifted_parabola_vertex :
  ∃ (vertex_x vertex_y : ℝ),
    vertex_x = 0 ∧
    vertex_y = 2 ∧
    ∀ (x : ℝ), shifted_parabola x ≤ shifted_parabola vertex_x :=
by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l954_95469


namespace NUMINAMATH_CALUDE_product_evaluation_l954_95437

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l954_95437


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l954_95435

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (total_sales : ℚ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_oranges = 30)
  (h4 : afternoon_apples = 50)
  (h5 : afternoon_oranges = 40)
  (h6 : total_sales = 205) :
  ∃ (morning_apples : ℕ), 
    apple_price * (morning_apples + afternoon_apples) + 
    orange_price * (morning_oranges + afternoon_oranges) = total_sales ∧
    morning_apples = 40 := by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l954_95435


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l954_95480

theorem same_solution_implies_c_value (c : ℝ) :
  (∃ x : ℝ, 3 * x + 8 = 5 ∧ c * x + 15 = 3) → c = 12 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l954_95480


namespace NUMINAMATH_CALUDE_equation_solution_exists_l954_95453

theorem equation_solution_exists : ∃ x : ℝ, 
  (x^3 - (0.1)^3) / (x^2 + 0.066 + (0.1)^2) = 0.5599999999999999 ∧ 
  abs (x - 0.8) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l954_95453


namespace NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_l954_95429

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (2 * a - 2, a + 5)

-- Define the point Q
def Q : ℝ × ℝ := (4, 5)

-- Theorem for part 1
theorem P_on_x_axis (a : ℝ) : 
  P a = (-12, 0) ↔ (P a).2 = 0 :=
sorry

-- Theorem for part 2
theorem P_parallel_to_y_axis (a : ℝ) :
  (P a).1 = Q.1 → P a = (4, 8) ∧ (P a).1 > 0 ∧ (P a).2 > 0 :=
sorry

end NUMINAMATH_CALUDE_P_on_x_axis_P_parallel_to_y_axis_l954_95429


namespace NUMINAMATH_CALUDE_coin_game_probability_l954_95434

def coin_game (n : ℕ) : ℝ :=
  sorry

theorem coin_game_probability : coin_game 5 = 1521 / 2^15 := by
  sorry

end NUMINAMATH_CALUDE_coin_game_probability_l954_95434


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l954_95432

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and a point on the line x - √2 y - 5 = 0 is √3/3 -/
theorem min_distance_ellipse_line : 
  ∃ (d : ℝ), d = Real.sqrt 3 / 3 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / 8 + P.2^2 / 4 = 1) → 
    (Q.1 - Real.sqrt 2 * Q.2 - 5 = 0) → 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≥ d ∧
    ∃ (P₀ Q₀ : ℝ × ℝ), 
      (P₀.1^2 / 8 + P₀.2^2 / 4 = 1) ∧
      (Q₀.1 - Real.sqrt 2 * Q₀.2 - 5 = 0) ∧
      Real.sqrt ((P₀.1 - Q₀.1)^2 + (P₀.2 - Q₀.2)^2) = d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l954_95432


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l954_95472

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b) → (1/4 : ℝ)^a > (1/4 : ℝ)^b ∧
  ∃ a' b' : ℝ, (1/4 : ℝ)^a' > (1/4 : ℝ)^b' ∧ ¬(0 < a' ∧ a' < b') :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l954_95472


namespace NUMINAMATH_CALUDE_cost_of_tea_cake_eclair_l954_95484

/-- Given the costs of tea and a cake, tea and an éclair, and a cake and an éclair,
    prove that the sum of the costs of tea, a cake, and an éclair
    is equal to half the sum of all three given costs. -/
theorem cost_of_tea_cake_eclair
  (t c e : ℝ)  -- t: cost of tea, c: cost of cake, e: cost of éclair
  (h1 : t + c = 4.5)  -- cost of tea and cake
  (h2 : t + e = 4)    -- cost of tea and éclair
  (h3 : c + e = 6.5)  -- cost of cake and éclair
  : t + c + e = (4.5 + 4 + 6.5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tea_cake_eclair_l954_95484


namespace NUMINAMATH_CALUDE_quadratic_inequality_l954_95468

/-- Quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  (q.a : ℝ) * x^2 + (q.b : ℝ) * x + (q.c : ℝ)

/-- A quadratic trinomial is positive for all real x -/
def IsAlwaysPositive (q : QuadraticTrinomial) : Prop :=
  ∀ x : ℝ, evaluate q x > 0

theorem quadratic_inequality {f g : QuadraticTrinomial} 
  (hf : IsAlwaysPositive f) 
  (hg : IsAlwaysPositive g)
  (h : ∀ x : ℝ, evaluate f x / evaluate g x ≥ Real.sqrt 2) :
  ∀ x : ℝ, evaluate f x / evaluate g x > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l954_95468


namespace NUMINAMATH_CALUDE_student_circle_circumference_l954_95489

/-- The circumference of a circle formed by people standing with overlapping arms -/
def circle_circumference (n : ℕ) (arm_span : ℝ) (overlap : ℝ) : ℝ :=
  n * (arm_span - overlap)

/-- Proof that the circumference of the circle formed by 16 students is 110.4 cm -/
theorem student_circle_circumference :
  circle_circumference 16 10.4 3.5 = 110.4 := by
  sorry

end NUMINAMATH_CALUDE_student_circle_circumference_l954_95489


namespace NUMINAMATH_CALUDE_cistern_width_l954_95417

/-- Given a cistern with the following properties:
  * length: 10 meters
  * water depth: 1.35 meters
  * total wet surface area: 103.2 square meters
  Prove that the width of the cistern is 6 meters. -/
theorem cistern_width (length : ℝ) (water_depth : ℝ) (wet_surface_area : ℝ) :
  length = 10 →
  water_depth = 1.35 →
  wet_surface_area = 103.2 →
  ∃ (width : ℝ), 
    wet_surface_area = length * width + 2 * length * water_depth + 2 * width * water_depth ∧
    width = 6 :=
by sorry

end NUMINAMATH_CALUDE_cistern_width_l954_95417


namespace NUMINAMATH_CALUDE_percentage_of_non_roses_l954_95423

theorem percentage_of_non_roses (roses tulips daisies : ℕ) 
  (h_roses : roses = 25)
  (h_tulips : tulips = 40)
  (h_daisies : daisies = 35) :
  (100 : ℚ) * (tulips + daisies : ℚ) / (roses + tulips + daisies : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_roses_l954_95423


namespace NUMINAMATH_CALUDE_even_function_implies_m_zero_l954_95485

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + 4

theorem even_function_implies_m_zero (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_zero_l954_95485


namespace NUMINAMATH_CALUDE_R_when_S_is_9_l954_95473

-- Define the equation R = gS - 6
def R (g S : ℚ) : ℚ := g * S - 6

-- State the theorem
theorem R_when_S_is_9 :
  ∀ g : ℚ, R g 7 = 18 → R g 9 = 174 / 7 := by
  sorry

end NUMINAMATH_CALUDE_R_when_S_is_9_l954_95473


namespace NUMINAMATH_CALUDE_ones_more_frequent_than_fives_l954_95457

-- Define the upper bound of the sequence
def upperBound : ℕ := 1000000000

-- Define a function that computes the digital root of a number
def digitalRoot (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

-- Define a function that counts occurrences of a digit in the final sequence
def countDigit (d : ℕ) : ℕ :=
  (upperBound / 9) + if d = 1 then 1 else 0

-- Theorem statement
theorem ones_more_frequent_than_fives :
  countDigit 1 > countDigit 5 := by
sorry

end NUMINAMATH_CALUDE_ones_more_frequent_than_fives_l954_95457


namespace NUMINAMATH_CALUDE_coaches_schedule_lcm_l954_95497

theorem coaches_schedule_lcm : Nat.lcm 5 (Nat.lcm 3 (Nat.lcm 9 8)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_coaches_schedule_lcm_l954_95497


namespace NUMINAMATH_CALUDE_min_value_fraction_l954_95492

theorem min_value_fraction (x : ℝ) (h : x > 10) :
  x^2 / (x - 10) ≥ 40 ∧ ∃ y > 10, y^2 / (y - 10) = 40 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l954_95492


namespace NUMINAMATH_CALUDE_orange_cost_l954_95450

/-- If 4 dozen oranges cost $24.00, then 6 dozen oranges at the same rate will cost $36.00 -/
theorem orange_cost (cost_four_dozen : ℝ) (h : cost_four_dozen = 24) :
  (6 : ℝ) / 4 * cost_four_dozen = 36 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_l954_95450


namespace NUMINAMATH_CALUDE_sequence_zero_l954_95477

/-- A sequence of real numbers indexed by positive integers. -/
def RealSequence := ℕ+ → ℝ

/-- The property that b_n ≤ c_n for all n. -/
def LessEqProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, b n ≤ c n

/-- The property that b_{n+1} and c_{n+1} are roots of x^2 + b_n*x + c_n = 0. -/
def RootProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧
            (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0

theorem sequence_zero (b c : RealSequence) 
  (h1 : LessEqProperty b c) (h2 : RootProperty b c) :
  (∀ n : ℕ+, b n = 0 ∧ c n = 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_zero_l954_95477


namespace NUMINAMATH_CALUDE_total_profit_is_135000_l954_95422

/-- Represents an investor in the partnership business -/
structure Investor where
  name : String
  investment : ℕ
  months : ℕ

/-- Calculates the total profit given the investors and C's profit share -/
def calculateTotalProfit (investors : List Investor) (cProfit : ℕ) : ℕ :=
  let totalInvestmentMonths := investors.map (λ i => i.investment * i.months) |>.sum
  let cInvestmentMonths := (investors.find? (λ i => i.name = "C")).map (λ i => i.investment * i.months)
  match cInvestmentMonths with
  | some im => cProfit * totalInvestmentMonths / im
  | none => 0

/-- Theorem stating that the total profit is 135000 given the specified conditions -/
theorem total_profit_is_135000 (investors : List Investor) (h1 : investors.length = 5)
    (h2 : investors.any (λ i => i.name = "A" ∧ i.investment = 12000 ∧ i.months = 6))
    (h3 : investors.any (λ i => i.name = "B" ∧ i.investment = 16000 ∧ i.months = 12))
    (h4 : investors.any (λ i => i.name = "C" ∧ i.investment = 20000 ∧ i.months = 12))
    (h5 : investors.any (λ i => i.name = "D" ∧ i.investment = 24000 ∧ i.months = 12))
    (h6 : investors.any (λ i => i.name = "E" ∧ i.investment = 18000 ∧ i.months = 6))
    (h7 : calculateTotalProfit investors 36000 = 135000) : 
  calculateTotalProfit investors 36000 = 135000 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_is_135000_l954_95422


namespace NUMINAMATH_CALUDE_f_neg_three_equals_six_l954_95475

-- Define the function f with the given property
def f : ℝ → ℝ := sorry

-- State the main theorem
theorem f_neg_three_equals_six :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  f 1 = 2 →
  f (-3) = 6 := by sorry

end NUMINAMATH_CALUDE_f_neg_three_equals_six_l954_95475


namespace NUMINAMATH_CALUDE_special_rectangle_area_l954_95431

/-- Represents a rectangle with a diagonal of length y and length three times its width -/
structure SpecialRectangle where
  y : ℝ  -- diagonal length
  w : ℝ  -- width
  h : ℝ  -- height (length)
  h_eq : h = 3 * w  -- length is three times the width
  diag_eq : y^2 = h^2 + w^2  -- Pythagorean theorem for the diagonal

/-- The area of a SpecialRectangle is 3y^2/10 -/
theorem special_rectangle_area (rect : SpecialRectangle) :
  rect.w * rect.h = (3 * rect.y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l954_95431
