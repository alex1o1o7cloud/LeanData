import Mathlib

namespace NUMINAMATH_CALUDE_rogers_second_bag_pieces_l3443_344340

/-- Represents the number of candy bags each person has -/
def num_bags : ℕ := 2

/-- Represents the number of candy pieces in each of Sandra's bags -/
def sandra_bag_pieces : ℕ := 6

/-- Represents the number of candy pieces in Roger's first bag -/
def roger_first_bag_pieces : ℕ := 11

/-- Represents the difference in total candy pieces between Roger and Sandra -/
def difference : ℕ := 2

/-- Theorem stating the number of candy pieces in Roger's second bag -/
theorem rogers_second_bag_pieces :
  ∃ (x : ℕ), x = num_bags * sandra_bag_pieces + difference - roger_first_bag_pieces :=
sorry

end NUMINAMATH_CALUDE_rogers_second_bag_pieces_l3443_344340


namespace NUMINAMATH_CALUDE_rice_bag_weight_l3443_344300

theorem rice_bag_weight (rice_bags : ℕ) (flour_bags : ℕ) (total_weight : ℕ) :
  rice_bags = 20 →
  flour_bags = 50 →
  total_weight = 2250 →
  (∃ (rice_weight flour_weight : ℕ),
    rice_weight * rice_bags + flour_weight * flour_bags = total_weight ∧
    rice_weight = 2 * flour_weight) →
  ∃ (rice_weight : ℕ), rice_weight = 50 :=
by sorry

end NUMINAMATH_CALUDE_rice_bag_weight_l3443_344300


namespace NUMINAMATH_CALUDE_constant_triangle_area_l3443_344323

noncomputable section

-- Define the curve C: xy = 1, x > 0
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 = 1 ∧ p.1 > 0}

-- Define a point P on the curve C
def P (a : ℝ) : ℝ × ℝ := (a, 1/a)

-- Define the tangent line l at point P
def tangent_line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - 1/a = -(1/a^2) * (p.1 - a)}

-- Define points A and B as intersections of tangent line with axes
def A (a : ℝ) : ℝ × ℝ := (0, 2/a)
def B (a : ℝ) : ℝ × ℝ := (2*a, 0)

-- Define the area of triangle OAB
def triangle_area (a : ℝ) : ℝ := (1/2) * (2/a) * (2*a)

-- Theorem statement
theorem constant_triangle_area (a : ℝ) (h : a > 0) :
  P a ∈ C → triangle_area a = 2 := by sorry

end

end NUMINAMATH_CALUDE_constant_triangle_area_l3443_344323


namespace NUMINAMATH_CALUDE_specific_gold_cube_profit_l3443_344368

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (buy_price : ℝ) (sell_multiplier : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := volume * density
  let cost := mass * buy_price
  let sell_price := cost * sell_multiplier
  sell_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end NUMINAMATH_CALUDE_specific_gold_cube_profit_l3443_344368


namespace NUMINAMATH_CALUDE_product_zero_l3443_344308

theorem product_zero (b : ℤ) (h : b = 3) : 
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * 
  (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_product_zero_l3443_344308


namespace NUMINAMATH_CALUDE_only_54_65_rounds_differently_l3443_344384

def round_to_nearest_tenth (x : Float) : Float :=
  (x * 10).round / 10

theorem only_54_65_rounds_differently : 
  let numbers := [54.56, 54.63, 54.64, 54.65, 54.59]
  ∀ x ∈ numbers, x ≠ 54.65 → round_to_nearest_tenth x = 54.6 ∧
  round_to_nearest_tenth 54.65 ≠ 54.6 := by
  sorry

end NUMINAMATH_CALUDE_only_54_65_rounds_differently_l3443_344384


namespace NUMINAMATH_CALUDE_fries_ratio_l3443_344303

def sally_initial_fries : ℕ := 14
def mark_initial_fries : ℕ := 36
def sally_final_fries : ℕ := 26

def fries_mark_gave_sally : ℕ := sally_final_fries - sally_initial_fries

theorem fries_ratio :
  (fries_mark_gave_sally : ℚ) / mark_initial_fries = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fries_ratio_l3443_344303


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_theorem_l3443_344341

/-- The radius of the smallest circle from which a triangle with sides 2, 3, and 4 can be cut out --/
def smallest_enclosing_circle_radius : ℝ := 2

/-- The three sides of the triangle --/
def triangle_sides : Fin 3 → ℝ
| 0 => 2
| 1 => 3
| 2 => 4
| _ => 0  -- This line is necessary due to Lean's totality requirement

theorem smallest_enclosing_circle_theorem :
  ∀ r : ℝ, (∀ i : Fin 3, triangle_sides i ≤ 2 * r) → r ≥ smallest_enclosing_circle_radius :=
by sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_theorem_l3443_344341


namespace NUMINAMATH_CALUDE_tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l3443_344354

/-- Represents the tax reduction process for a commodity -/
def tax_reduction (initial_tax : ℝ) : ℝ :=
  let after_first_reduction := initial_tax * (1 - 0.25)
  let after_second_reduction := after_first_reduction * (1 - 0.20)
  after_second_reduction

/-- Theorem stating that the tax reduction process results in 60% of the initial tax -/
theorem tax_reduction_is_sixty_percent (a : ℝ) :
  tax_reduction a = 0.60 * a := by
  sorry

/-- Corollary for the specific case where the initial tax is 1000 million euros -/
theorem tax_reduction_for_thousand_million :
  tax_reduction 1000 = 600 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_is_sixty_percent_tax_reduction_for_thousand_million_l3443_344354


namespace NUMINAMATH_CALUDE_gcd_459_357_l3443_344389

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l3443_344389


namespace NUMINAMATH_CALUDE_fraction_square_decimal_equivalent_l3443_344367

theorem fraction_square_decimal_equivalent : (1 / 9 : ℚ)^2 = 0.012345679012345678 := by
  sorry

end NUMINAMATH_CALUDE_fraction_square_decimal_equivalent_l3443_344367


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3443_344370

/-- The cost per kilogram of mangos -/
def mango_cost : ℝ := sorry

/-- The cost per kilogram of rice -/
def rice_cost : ℝ := sorry

/-- The cost per kilogram of flour -/
def flour_cost : ℝ := 22

theorem total_cost_calculation : 
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  (4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 941.6) :=
by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3443_344370


namespace NUMINAMATH_CALUDE_sum_squared_odd_l3443_344396

theorem sum_squared_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_odd_l3443_344396


namespace NUMINAMATH_CALUDE_fundraising_goal_l3443_344390

/-- Fundraising problem -/
theorem fundraising_goal (ken mary scott goal : ℕ) : 
  ken = 600 →
  mary = 5 * ken →
  mary = 3 * scott →
  ken + mary + scott = goal + 600 →
  goal = 4000 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_goal_l3443_344390


namespace NUMINAMATH_CALUDE_milk_mixture_problem_l3443_344316

/-- Proves that the volume removed and replaced with water is 50 litres -/
theorem milk_mixture_problem (total_volume : ℝ) (initial_milk : ℝ) (final_concentration : ℝ) :
  total_volume = 100 →
  initial_milk = 36 →
  final_concentration = 0.09 →
  ∃ (V : ℝ), V = 50 ∧
    (initial_milk / total_volume) * (1 - V / total_volume)^2 = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_milk_mixture_problem_l3443_344316


namespace NUMINAMATH_CALUDE_morse_high_school_students_l3443_344398

/-- The number of students in the other three grades at Morse High School -/
def other_grades_students : ℕ := 1500

/-- The number of seniors at Morse High School -/
def seniors : ℕ := 300

/-- The percentage of seniors who have cars -/
def senior_car_percentage : ℚ := 40 / 100

/-- The percentage of students in other grades who have cars -/
def other_grades_car_percentage : ℚ := 10 / 100

/-- The percentage of all students who have cars -/
def total_car_percentage : ℚ := 15 / 100

theorem morse_high_school_students :
  (seniors * senior_car_percentage + other_grades_students * other_grades_car_percentage : ℚ) =
  (seniors + other_grades_students) * total_car_percentage :=
sorry

end NUMINAMATH_CALUDE_morse_high_school_students_l3443_344398


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3443_344317

/-- Proves that the interest rate at which A lends to B is 8% per annum -/
theorem interest_rate_calculation (principal : ℝ) (rate_C : ℝ) (time : ℝ) (gain_B : ℝ)
  (h1 : principal = 3150)
  (h2 : rate_C = 12.5)
  (h3 : time = 2)
  (h4 : gain_B = 283.5) :
  let interest_C := principal * rate_C / 100 * time
  let rate_A := (interest_C - gain_B) / (principal * time) * 100
  rate_A = 8 := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3443_344317


namespace NUMINAMATH_CALUDE_two_evaluations_determine_sequence_l3443_344349

/-- A finite sequence of natural numbers -/
def Sequence := List Nat

/-- Evaluate the polynomial at a given point -/
def evaluatePolynomial (s : Sequence) (β : Nat) : Nat :=
  s.enum.foldl (fun acc (i, a) => acc + a * β ^ i) 0

/-- Theorem stating that two evaluations are sufficient to determine the sequence -/
theorem two_evaluations_determine_sequence (s : Sequence) :
  ∃ β₁ β₂ : Nat, β₁ ≠ β₂ ∧
  ∀ t : Sequence, t.length = s.length →
    evaluatePolynomial s β₁ = evaluatePolynomial t β₁ ∧
    evaluatePolynomial s β₂ = evaluatePolynomial t β₂ →
    s = t :=
  sorry

end NUMINAMATH_CALUDE_two_evaluations_determine_sequence_l3443_344349


namespace NUMINAMATH_CALUDE_factory_output_doubling_time_l3443_344375

theorem factory_output_doubling_time (growth_rate : ℝ) (doubling_time : ℝ) :
  growth_rate = 0.1 →
  (1 + growth_rate) ^ doubling_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factory_output_doubling_time_l3443_344375


namespace NUMINAMATH_CALUDE_complementary_and_supplementary_angles_l3443_344356

/-- Given an angle of 46 degrees, its complementary angle is 44 degrees and its supplementary angle is 134 degrees. -/
theorem complementary_and_supplementary_angles (angle : ℝ) : 
  angle = 46 → 
  (90 - angle = 44) ∧ (180 - angle = 134) := by
  sorry

end NUMINAMATH_CALUDE_complementary_and_supplementary_angles_l3443_344356


namespace NUMINAMATH_CALUDE_two_to_700_gt_five_to_300_l3443_344386

theorem two_to_700_gt_five_to_300 : 2^700 > 5^300 := by
  sorry

end NUMINAMATH_CALUDE_two_to_700_gt_five_to_300_l3443_344386


namespace NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l3443_344394

/-- The equation of a potential ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (3 - m) + y^2 / (m - 1) = 1

/-- The condition on m -/
def m_condition (m : ℝ) : Prop :=
  1 < m ∧ m < 3

/-- The equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  3 - m > 0 ∧ m - 1 > 0 ∧ m ≠ 2

/-- The m_condition is necessary but not sufficient for the equation to represent an ellipse -/
theorem m_condition_necessary_not_sufficient :
  (∀ m, is_ellipse m → m_condition m) ∧
  ¬(∀ m, m_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_m_condition_necessary_not_sufficient_l3443_344394


namespace NUMINAMATH_CALUDE_ecu_distribution_l3443_344352

theorem ecu_distribution (x y z : ℤ) : 
  (x - y - z = 8) ∧ 
  (y - (x - y - z) - z = 8) ∧ 
  (z - (x - y - z) - (y - (x - y - z)) = 8) → 
  x = 13 ∧ y = 7 ∧ z = 4 := by
sorry

end NUMINAMATH_CALUDE_ecu_distribution_l3443_344352


namespace NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3443_344359

theorem tens_digit_of_N_power_20 (N : ℕ) 
  (h1 : Even N) 
  (h2 : ¬ (10 ∣ N)) : 
  (N^20 / 10) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3443_344359


namespace NUMINAMATH_CALUDE_exam_mistakes_l3443_344353

theorem exam_mistakes (total_students : ℕ) (total_mistakes : ℕ) 
  (h1 : total_students = 333) (h2 : total_mistakes = 1000) : 
  ∀ (x y z : ℕ), 
    (x + y + z = total_students) → 
    (4 * y + 6 * z ≤ total_mistakes) → 
    (z ≤ x) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_mistakes_l3443_344353


namespace NUMINAMATH_CALUDE_conference_beverages_l3443_344319

theorem conference_beverages (total participants : ℕ) (coffee_drinkers : ℕ) (tea_drinkers : ℕ) (both_drinkers : ℕ) :
  total = 30 →
  coffee_drinkers = 15 →
  tea_drinkers = 18 →
  both_drinkers = 8 →
  total - (coffee_drinkers + tea_drinkers - both_drinkers) = 5 := by
sorry

end NUMINAMATH_CALUDE_conference_beverages_l3443_344319


namespace NUMINAMATH_CALUDE_jim_flour_on_counter_l3443_344364

/-- The amount of flour Jim has on the kitchen counter -/
def flour_on_counter (flour_in_cupboard flour_in_pantry flour_per_loaf : ℕ) (loaves_can_bake : ℕ) : ℕ :=
  loaves_can_bake * flour_per_loaf - (flour_in_cupboard + flour_in_pantry)

/-- Theorem stating that Jim has 100g of flour on the kitchen counter -/
theorem jim_flour_on_counter :
  flour_on_counter 200 100 200 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jim_flour_on_counter_l3443_344364


namespace NUMINAMATH_CALUDE_sin45_plus_sqrt2_half_l3443_344388

theorem sin45_plus_sqrt2_half (h : Real.sin (π / 4) = Real.sqrt 2 / 2) :
  Real.sin (π / 4) + Real.sqrt 2 / 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin45_plus_sqrt2_half_l3443_344388


namespace NUMINAMATH_CALUDE_series_evaluation_l3443_344318

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

theorem series_evaluation : series_sum = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_series_evaluation_l3443_344318


namespace NUMINAMATH_CALUDE_equation_equals_twentyfour_l3443_344325

theorem equation_equals_twentyfour : 6 / (1 - 3 / 10) = 24 := by sorry

end NUMINAMATH_CALUDE_equation_equals_twentyfour_l3443_344325


namespace NUMINAMATH_CALUDE_factor_expression_l3443_344376

theorem factor_expression (x : ℝ) : 60 * x^2 + 45 * x = 15 * x * (4 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3443_344376


namespace NUMINAMATH_CALUDE_golden_retriever_pup_difference_l3443_344387

/-- Represents the number of pups each golden retriever had more than each husky -/
def pup_difference : ℕ := 2

theorem golden_retriever_pup_difference :
  let num_huskies : ℕ := 5
  let num_pitbulls : ℕ := 2
  let num_golden_retrievers : ℕ := 4
  let pups_per_husky : ℕ := 3
  let pups_per_pitbull : ℕ := 3
  let total_adult_dogs : ℕ := num_huskies + num_pitbulls + num_golden_retrievers
  let total_pups : ℕ := total_adult_dogs + 30
  total_pups = num_huskies * pups_per_husky + num_pitbulls * pups_per_pitbull + 
               num_golden_retrievers * (pups_per_husky + pup_difference) →
  pup_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_retriever_pup_difference_l3443_344387


namespace NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_five_l3443_344355

theorem two_numbers_with_difference_and_quotient_five :
  ∀ x y : ℝ, x - y = 5 → x / y = 5 → x = 25 / 4 ∧ y = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_difference_and_quotient_five_l3443_344355


namespace NUMINAMATH_CALUDE_function_characterization_l3443_344358

def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ n p, is_prime p → (f n)^p ≡ n [MOD f p]

def is_identity (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = n

def is_constant_one_on_primes (f : ℕ → ℕ) : Prop :=
  ∀ p, is_prime p → f p = 1

def is_special_function (f : ℕ → ℕ) : Prop :=
  f 2 = 2 ∧
  (∀ p, is_prime p → p > 2 → f p = 1) ∧
  (∀ n, f n ≡ n [MOD 2])

theorem function_characterization (f : ℕ → ℕ) :
  satisfies_condition f →
  (is_identity f ∨ is_constant_one_on_primes f ∨ is_special_function f) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3443_344358


namespace NUMINAMATH_CALUDE_bill_value_l3443_344383

theorem bill_value (total_money : ℕ) (num_bills : ℕ) (h1 : total_money = 45) (h2 : num_bills = 9) :
  total_money / num_bills = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_value_l3443_344383


namespace NUMINAMATH_CALUDE_first_business_donation_is_half_dollar_l3443_344305

/-- Represents the fundraising scenario for Didi's soup kitchen --/
structure FundraisingScenario where
  num_cakes : ℕ
  slices_per_cake : ℕ
  price_per_slice : ℚ
  second_business_donation : ℚ
  total_raised : ℚ

/-- Calculates the donation per slice from the first business owner --/
def first_business_donation_per_slice (scenario : FundraisingScenario) : ℚ :=
  let total_slices := scenario.num_cakes * scenario.slices_per_cake
  let sales_revenue := total_slices * scenario.price_per_slice
  let total_business_donations := scenario.total_raised - sales_revenue
  let second_business_total := total_slices * scenario.second_business_donation
  let first_business_total := total_business_donations - second_business_total
  first_business_total / total_slices

/-- Theorem stating that the first business owner's donation per slice is $0.50 --/
theorem first_business_donation_is_half_dollar (scenario : FundraisingScenario) 
  (h1 : scenario.num_cakes = 10)
  (h2 : scenario.slices_per_cake = 8)
  (h3 : scenario.price_per_slice = 1)
  (h4 : scenario.second_business_donation = 1/4)
  (h5 : scenario.total_raised = 140) :
  first_business_donation_per_slice scenario = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_first_business_donation_is_half_dollar_l3443_344305


namespace NUMINAMATH_CALUDE_car_ac_price_ratio_l3443_344362

/-- Given a car that costs $500 more than an AC, and the AC costs $1500,
    prove that the ratio of the car's price to the AC's price is 4:3. -/
theorem car_ac_price_ratio :
  ∀ (car_price ac_price : ℕ),
  ac_price = 1500 →
  car_price = ac_price + 500 →
  (car_price : ℚ) / ac_price = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_car_ac_price_ratio_l3443_344362


namespace NUMINAMATH_CALUDE_quadratic_properties_l3443_344330

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, f x < f y → x < y ∨ x > y) ∧  -- Opens downwards
  (∀ x : ℝ, f (x + (-2)) = f ((-2) - x)) ∧  -- Axis of symmetry is x = -2
  (∀ x : ℝ, f x < 0) ∧                      -- Does not intersect x-axis
  (∀ x y : ℝ, x > -1 → y > x → f y < f x)   -- Decreases for x > -1
  :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3443_344330


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3443_344328

/-- The number of ways to place 5 numbered balls into 5 numbered boxes -/
def ball_placement_count : ℕ := 20

/-- A function that returns the number of ways to place n numbered balls into n numbered boxes
    such that exactly k balls match their box numbers -/
def place_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that the number of ways to place 5 numbered balls into 5 numbered boxes,
    where each box contains one ball and exactly two balls match their box numbers, is 20 -/
theorem ball_placement_theorem : place_balls 5 2 = ball_placement_count := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3443_344328


namespace NUMINAMATH_CALUDE_least_positive_even_congruence_l3443_344324

theorem least_positive_even_congruence : ∃ x : ℕ, 
  (x + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ 
  Even x ∧
  x = 2 ∧
  ∀ y : ℕ, y < x → ¬((y + 3721 : ℤ) ≡ 1547 [ZMOD 12] ∧ Even y) := by
  sorry

end NUMINAMATH_CALUDE_least_positive_even_congruence_l3443_344324


namespace NUMINAMATH_CALUDE_max_cfriendly_diff_l3443_344346

-- Define c-friendly function
def CFriendly (c : ℝ) (f : ℝ → ℝ) : Prop :=
  c > 1 ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → |f x - f y| ≤ c * |x - y|

-- State the theorem
theorem max_cfriendly_diff (c : ℝ) (f : ℝ → ℝ) (x y : ℝ) :
  CFriendly c f →
  x ∈ Set.Icc 0 1 →
  y ∈ Set.Icc 0 1 →
  |f x - f y| ≤ (c + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_cfriendly_diff_l3443_344346


namespace NUMINAMATH_CALUDE_cleaning_payment_l3443_344338

theorem cleaning_payment (payment_rate rooms_cleaned : ℚ) : 
  payment_rate = 13 / 3 → rooms_cleaned = 8 / 5 → payment_rate * rooms_cleaned = 104 / 15 := by
  sorry

end NUMINAMATH_CALUDE_cleaning_payment_l3443_344338


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3443_344373

theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a < 0) → a < 11 ∧
  ∃ a : ℝ, a < 11 ∧ ∀ x : ℝ, x^2 - 2*x + a ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3443_344373


namespace NUMINAMATH_CALUDE_skirt_width_is_four_l3443_344313

-- Define the width of the rectangle for each skirt
def width : ℝ := sorry

-- Define the length of the rectangle for each skirt
def length : ℝ := 12

-- Define the number of skirts
def num_skirts : ℕ := 3

-- Define the area of material for the bodice
def bodice_area : ℝ := 2 + 2 * 5

-- Define the cost per square foot of material
def cost_per_sqft : ℝ := 3

-- Define the total cost of material
def total_cost : ℝ := 468

-- Theorem statement
theorem skirt_width_is_four :
  width = 4 ∧
  length * width * num_skirts + bodice_area = total_cost / cost_per_sqft :=
sorry

end NUMINAMATH_CALUDE_skirt_width_is_four_l3443_344313


namespace NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l3443_344357

-- Define the triangle
def triangle_side1 : ℝ := 15
def triangle_side2 : ℝ := 36
def triangle_side3 : ℝ := 39

-- Define the rectangle's area formula
def rectangle_area (α β ω : ℝ) : ℝ := α * ω - β * ω^2

-- State the theorem
theorem inscribed_rectangle_coefficient :
  ∃ (α β : ℝ),
    (∀ ω, rectangle_area α β ω ≥ 0) ∧
    (rectangle_area α β triangle_side2 = 0) ∧
    (rectangle_area α β (triangle_side2 / 2) = 
      (triangle_side1 + triangle_side2 + triangle_side3) * 
      (triangle_side1 + triangle_side2 - triangle_side3) * 
      (triangle_side1 - triangle_side2 + triangle_side3) * 
      (-triangle_side1 + triangle_side2 + triangle_side3) / 
      (4 * 16)) ∧
    β = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_coefficient_l3443_344357


namespace NUMINAMATH_CALUDE_f_6_equals_37_l3443_344385

def f : ℕ → ℤ
| 0 => 0  -- Arbitrary base case
| n + 1 =>
  if 1 ≤ n + 1 ∧ n + 1 ≤ 4 then f n - (n + 1)
  else if 5 ≤ n + 1 ∧ n + 1 ≤ 8 then f n + 2 * (n + 1)
  else f n * (n + 1)

theorem f_6_equals_37 (h : f 4 = 15) : f 6 = 37 := by
  sorry

end NUMINAMATH_CALUDE_f_6_equals_37_l3443_344385


namespace NUMINAMATH_CALUDE_difference_of_bounds_l3443_344382

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := m * x + 2
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

-- Define the theorem
theorem difference_of_bounds (m : ℝ) :
  ∃ (a b : ℤ), (∀ x : ℝ, a ≤ f m x - g m x ∧ f m x - g m x ≤ b) ∧
  (∀ y : ℝ, a ≤ y ∧ y ≤ b → ∃ x : ℝ, f m x - g m x = y) →
  a - b = -2 :=
sorry

end NUMINAMATH_CALUDE_difference_of_bounds_l3443_344382


namespace NUMINAMATH_CALUDE_usual_bus_time_l3443_344315

/-- The usual time to catch the bus, given that walking at 4/5 of the usual speed results in missing the bus by 3 minutes, is 12 minutes. -/
theorem usual_bus_time (usual_speed : ℝ) (usual_time : ℝ) : 
  (4 / 5 * usual_speed * (usual_time + 3) = usual_speed * usual_time) → 
  usual_time = 12 := by
sorry

end NUMINAMATH_CALUDE_usual_bus_time_l3443_344315


namespace NUMINAMATH_CALUDE_time_before_first_rewind_is_35_l3443_344311

/-- Represents the viewing time of a movie with interruptions -/
structure MovieViewing where
  totalTime : ℕ
  firstRewindTime : ℕ
  timeBetweenRewinds : ℕ
  secondRewindTime : ℕ
  timeAfterSecondRewind : ℕ

/-- Calculates the time watched before the first rewind -/
def timeBeforeFirstRewind (mv : MovieViewing) : ℕ :=
  mv.totalTime - (mv.firstRewindTime + mv.timeBetweenRewinds + mv.secondRewindTime + mv.timeAfterSecondRewind)

/-- Theorem stating that for the given movie viewing scenario, 
    the time watched before the first rewind is 35 minutes -/
theorem time_before_first_rewind_is_35 : 
  let mv : MovieViewing := {
    totalTime := 120,
    firstRewindTime := 5,
    timeBetweenRewinds := 45,
    secondRewindTime := 15,
    timeAfterSecondRewind := 20
  }
  timeBeforeFirstRewind mv = 35 := by
  sorry

end NUMINAMATH_CALUDE_time_before_first_rewind_is_35_l3443_344311


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l3443_344378

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_in_pitcher := C / 2
  let cups := 4
  let juice_per_cup := juice_in_pitcher / cups
  juice_per_cup / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l3443_344378


namespace NUMINAMATH_CALUDE_tile_size_calculation_l3443_344332

theorem tile_size_calculation (length width : ℝ) (num_tiles : ℕ) (h1 : length = 2) (h2 : width = 12) (h3 : num_tiles = 6) :
  (length * width) / num_tiles = 4 := by
  sorry

end NUMINAMATH_CALUDE_tile_size_calculation_l3443_344332


namespace NUMINAMATH_CALUDE_ellipse_focus_implies_k_l3443_344310

/-- Represents an ellipse with equation kx^2 + 5y^2 = 5 -/
structure Ellipse (k : ℝ) where
  equation : ∀ x y : ℝ, k * x^2 + 5 * y^2 = 5

/-- A focus of an ellipse -/
def Focus := ℝ × ℝ

/-- Theorem: For the ellipse kx^2 + 5y^2 = 5, if one of its foci is (2, 0), then k = 1 -/
theorem ellipse_focus_implies_k (k : ℝ) (e : Ellipse k) (f : Focus) :
  f = (2, 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_implies_k_l3443_344310


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l3443_344320

/-- Represents a point on the perimeter of the block area -/
structure Point where
  distance : ℝ  -- Distance from the starting point A
  mk_point_valid : 0 ≤ distance ∧ distance < 24

/-- Represents a walker -/
structure Walker where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The scenario of the two walkers -/
def scenario : Prop :=
  ∃ (jane hector : Walker) (meeting_point : Point),
    jane.speed = 2 * hector.speed ∧
    jane.direction ≠ hector.direction ∧
    (jane.direction = true → meeting_point.distance = 16) ∧
    (jane.direction = false → meeting_point.distance = 8)

/-- The theorem to be proved -/
theorem meeting_point_theorem :
  scenario → 
  ∃ (meeting_point : Point), 
    (meeting_point.distance = 8 ∨ meeting_point.distance = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l3443_344320


namespace NUMINAMATH_CALUDE_vector_magnitude_l3443_344365

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (norm a = 1) → 
  (norm (a - 2 • b) = Real.sqrt 21) → 
  (a.1 * b.1 + a.2 * b.2 = - (1/2) * norm a * norm b) →
  norm b = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3443_344365


namespace NUMINAMATH_CALUDE_dividend_calculation_l3443_344304

theorem dividend_calculation (divisor quotient remainder : ℕ) :
  divisor = 15 →
  quotient = 8 →
  remainder = 5 →
  (divisor * quotient + remainder) = 125 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3443_344304


namespace NUMINAMATH_CALUDE_custom_op_equality_l3443_344302

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality for the given expression -/
theorem custom_op_equality (a b : ℝ) :
  custom_op a b + custom_op (b - a) b = b^2 - b :=
by sorry

end NUMINAMATH_CALUDE_custom_op_equality_l3443_344302


namespace NUMINAMATH_CALUDE_mass_of_man_on_boat_l3443_344306

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under given conditions. -/
theorem mass_of_man_on_boat :
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sink_depth : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sink_depth water_density = 60 := by
sorry


end NUMINAMATH_CALUDE_mass_of_man_on_boat_l3443_344306


namespace NUMINAMATH_CALUDE_total_distance_rowed_l3443_344334

/-- The total distance traveled by a man rowing upstream and downstream -/
theorem total_distance_rowed (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  man_speed = 6 →
  river_speed = 1.2 →
  total_time = 1 →
  2 * (total_time * man_speed * river_speed) / (man_speed + river_speed) = 5.76 :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_rowed_l3443_344334


namespace NUMINAMATH_CALUDE_f_at_2_l3443_344342

-- Define the polynomial function
def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

-- Theorem statement
theorem f_at_2 : f 2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l3443_344342


namespace NUMINAMATH_CALUDE_range_of_a_l3443_344363

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ≤ 0 ∨ a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3443_344363


namespace NUMINAMATH_CALUDE_sum_of_squares_over_products_l3443_344337

theorem sum_of_squares_over_products (a b c : ℝ) (h : a + b + c = 0) :
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_over_products_l3443_344337


namespace NUMINAMATH_CALUDE_arthur_winning_strategy_l3443_344391

theorem arthur_winning_strategy (n : ℕ) (hn : n ≥ 2) :
  ∃ (A B : Finset ℕ), 
    A.card = n ∧ B.card = n ∧
    ∀ (k : ℕ) (hk : k ≤ 2*n - 2),
      ∃ (x : Fin k → ℕ),
        (∀ i : Fin k, ∃ (a : A) (b : B), x i = a * b) ∧
        (∀ i j l : Fin k, i ≠ j ∧ j ≠ l ∧ i ≠ l → ¬(x i ∣ x j * x l)) :=
by sorry

end NUMINAMATH_CALUDE_arthur_winning_strategy_l3443_344391


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l3443_344322

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 - Complex.I) * (2 + 3 * Complex.I) ∧ z.re = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l3443_344322


namespace NUMINAMATH_CALUDE_circus_ticket_price_l3443_344361

/-- Proves that the price of an upper seat ticket is $20 given the conditions of the circus ticket sales. -/
theorem circus_ticket_price :
  let lower_seat_price : ℕ := 30
  let total_tickets : ℕ := 80
  let total_revenue : ℕ := 2100
  let lower_seats_sold : ℕ := 50
  let upper_seats_sold : ℕ := total_tickets - lower_seats_sold
  let upper_seat_price : ℕ := (total_revenue - lower_seat_price * lower_seats_sold) / upper_seats_sold
  upper_seat_price = 20 := by sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l3443_344361


namespace NUMINAMATH_CALUDE_strawberry_jelly_amount_l3443_344344

/-- Given the total amount of jelly and the amount of blueberry jelly, 
    calculate the amount of strawberry jelly. -/
theorem strawberry_jelly_amount 
  (total_jelly : ℕ) 
  (blueberry_jelly : ℕ) 
  (h1 : total_jelly = 6310)
  (h2 : blueberry_jelly = 4518) : 
  total_jelly - blueberry_jelly = 1792 := by
sorry

end NUMINAMATH_CALUDE_strawberry_jelly_amount_l3443_344344


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3443_344366

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^51 + 51) % (x + 1) = 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3443_344366


namespace NUMINAMATH_CALUDE_orange_bin_problem_l3443_344347

theorem orange_bin_problem (initial_oranges : ℕ) : 
  initial_oranges - 2 + 28 = 31 → initial_oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_problem_l3443_344347


namespace NUMINAMATH_CALUDE_unique_solution_l3443_344329

/-- Prove that 7 is the only positive integer solution to the equation -/
theorem unique_solution : ∃! (x : ℕ), x > 0 ∧ (1/4 : ℚ) * (10*x + 7 - x^2) - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3443_344329


namespace NUMINAMATH_CALUDE_min_period_sin_2x_cos_2x_l3443_344350

/-- The minimum positive period of the function y = sin(2x) cos(2x) is π/2 -/
theorem min_period_sin_2x_cos_2x :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x) * Real.cos (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧
    (∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_period_sin_2x_cos_2x_l3443_344350


namespace NUMINAMATH_CALUDE_greatest_number_l3443_344301

theorem greatest_number (x : ℤ) (n : ℕ) : 
  (x ≤ 4) → 
  (2.134 * (n : ℝ)^(x : ℝ) < 210000) → 
  (∀ m : ℕ, m > n → 2.134 * (m : ℝ)^(4 : ℝ) ≥ 210000) → 
  n = 17 := by
sorry

end NUMINAMATH_CALUDE_greatest_number_l3443_344301


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3443_344392

theorem fraction_evaluation : (18 : ℝ) / (4.9 * 106) = 18 / 519.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3443_344392


namespace NUMINAMATH_CALUDE_area_of_pentagon_l3443_344336

-- Define the square ABCD
def ABCD : Set (ℝ × ℝ) := sorry

-- Define that BD is a diagonal of ABCD
def BD_is_diagonal (ABCD : Set (ℝ × ℝ)) : Prop := sorry

-- Define the length of BD
def BD_length : ℝ := 20

-- Define the rectangle BDFE
def BDFE : Set (ℝ × ℝ) := sorry

-- Define the pentagon ABEFD
def ABEFD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Theorem statement
theorem area_of_pentagon (h1 : BD_is_diagonal ABCD) (h2 : BD_length = 20) : 
  area ABEFD = 300 := by sorry

end NUMINAMATH_CALUDE_area_of_pentagon_l3443_344336


namespace NUMINAMATH_CALUDE_runner_time_difference_l3443_344326

theorem runner_time_difference (total_distance : ℝ) (second_half_time : ℝ) : 
  total_distance = 40 ∧ second_half_time = 24 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    (total_distance / 2) / initial_speed + (total_distance / 2) / (initial_speed / 2) = second_half_time ∧
    (total_distance / 2) / (initial_speed / 2) - (total_distance / 2) / initial_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_runner_time_difference_l3443_344326


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l3443_344312

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l3443_344312


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l3443_344369

theorem equidistant_point_on_y_axis : ∃ y : ℚ, 
  let A : ℚ × ℚ := (-3, 1)
  let B : ℚ × ℚ := (-2, 5)
  let P : ℚ × ℚ := (0, y)
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2 ∧ y = 19/8 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l3443_344369


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l3443_344381

theorem simple_interest_time_calculation 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (rate : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 160) 
  (h3 : rate = 20) : 
  (simple_interest * 100) / (principal * rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l3443_344381


namespace NUMINAMATH_CALUDE_detergent_for_nine_pounds_l3443_344379

/-- The amount of detergent needed for a given weight of clothes -/
def detergent_needed (rate : ℝ) (weight : ℝ) : ℝ := rate * weight

/-- Theorem: Given a rate of 2 ounces of detergent per pound of clothes,
    the amount of detergent needed for 9 pounds of clothes is 18 ounces -/
theorem detergent_for_nine_pounds :
  detergent_needed 2 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_detergent_for_nine_pounds_l3443_344379


namespace NUMINAMATH_CALUDE_three_solutions_l3443_344393

/-- Represents a solution to the equation A^B = BA --/
structure Solution :=
  (A B : Nat)
  (h1 : A ≠ B)
  (h2 : A ≥ 1 ∧ A ≤ 9)
  (h3 : B ≥ 1 ∧ B ≤ 9)
  (h4 : A^B = 10*B + A)
  (h5 : 10*B + A ≠ A*B)

/-- The set of all valid solutions --/
def validSolutions : Set Solution := {s | s.A^s.B = 10*s.B + s.A ∧ s.A ≠ s.B ∧ s.A ≥ 1 ∧ s.A ≤ 9 ∧ s.B ≥ 1 ∧ s.B ≤ 9 ∧ 10*s.B + s.A ≠ s.A*s.B}

/-- Theorem stating that there are exactly three solutions --/
theorem three_solutions :
  ∃ (s1 s2 s3 : Solution),
    validSolutions = {s1, s2, s3} ∧
    ((s1.A = 2 ∧ s1.B = 5) ∨ (s1.A = 6 ∧ s1.B = 2) ∨ (s1.A = 4 ∧ s1.B = 3)) ∧
    ((s2.A = 2 ∧ s2.B = 5) ∨ (s2.A = 6 ∧ s2.B = 2) ∨ (s2.A = 4 ∧ s2.B = 3)) ∧
    ((s3.A = 2 ∧ s3.B = 5) ∨ (s3.A = 6 ∧ s3.B = 2) ∨ (s3.A = 4 ∧ s3.B = 3)) ∧
    s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 :=
sorry

end NUMINAMATH_CALUDE_three_solutions_l3443_344393


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3443_344314

/-- Given a rectangle with integer side lengths and a perimeter of 150 feet,
    the maximum possible area is 1406 square feet. -/
theorem max_area_rectangle (x y : ℕ) : 
  x + y = 75 → x * y ≤ 1406 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3443_344314


namespace NUMINAMATH_CALUDE_president_and_committee_10_people_l3443_344307

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be part of the committee -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (Nat.choose (n - 1) 3)

/-- Theorem stating that choosing a president and a 3-person committee from 10 people,
    where the president cannot be part of the committee, can be done in 840 ways -/
theorem president_and_committee_10_people :
  choose_president_and_committee 10 = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_10_people_l3443_344307


namespace NUMINAMATH_CALUDE_team_scoring_problem_l3443_344345

theorem team_scoring_problem (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) 
  (h1 : player1_score = 20)
  (h2 : player2_score = player1_score / 2)
  (h3 : ∃ X : ℕ, player3_score = X * player2_score)
  (h4 : (player1_score + player2_score + player3_score) / 3 = 30) :
  ∃ X : ℕ, player3_score = X * player2_score ∧ X = 6 := by
  sorry

end NUMINAMATH_CALUDE_team_scoring_problem_l3443_344345


namespace NUMINAMATH_CALUDE_unique_solution_mn_l3443_344377

theorem unique_solution_mn : ∀ m n : ℕ, 
  m < n → 
  (∃ k : ℕ, m^2 + 1 = k * n) → 
  (∃ l : ℕ, n^2 + 1 = l * m) → 
  m = 1 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_mn_l3443_344377


namespace NUMINAMATH_CALUDE_bike_price_calculation_l3443_344321

theorem bike_price_calculation (current_price : ℝ) : 
  (current_price * 1.1 = 82500) → current_price = 75000 := by
  sorry

end NUMINAMATH_CALUDE_bike_price_calculation_l3443_344321


namespace NUMINAMATH_CALUDE_marble_problem_l3443_344360

theorem marble_problem :
  ∃ n : ℕ,
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 2 → n ≤ m) ∧
    n % 8 = 5 ∧
    n % 7 = 2 ∧
    n % 9 = 1 ∧
    n = 37 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l3443_344360


namespace NUMINAMATH_CALUDE_trig_sum_equality_l3443_344371

theorem trig_sum_equality : 
  (Real.cos (2 * π / 180)) / (Real.sin (47 * π / 180)) + 
  (Real.cos (88 * π / 180)) / (Real.sin (133 * π / 180)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_equality_l3443_344371


namespace NUMINAMATH_CALUDE_pencil_count_theorem_pencils_in_drawer_l3443_344333

/-- The total number of pencils after adding more to the drawer -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of pencils is the sum of initial and added pencils -/
theorem pencil_count_theorem (initial : ℕ) (added : ℕ) :
  total_pencils initial added = initial + added := by
  sorry

/-- Given information about the pencils in the drawer -/
def initial_pencils : ℕ := 41
def pencils_added : ℕ := 30

/-- The result we want to prove -/
theorem pencils_in_drawer :
  total_pencils initial_pencils pencils_added = 71 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_theorem_pencils_in_drawer_l3443_344333


namespace NUMINAMATH_CALUDE_vehicle_ownership_l3443_344399

theorem vehicle_ownership (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (bicycle_owners : ℕ)
  (car_and_motorcycle : ℕ) (motorcycle_and_bicycle : ℕ) (car_and_bicycle : ℕ)
  (h1 : total_adults = 500)
  (h2 : car_owners = 400)
  (h3 : motorcycle_owners = 200)
  (h4 : bicycle_owners = 150)
  (h5 : car_and_motorcycle = 100)
  (h6 : motorcycle_and_bicycle = 50)
  (h7 : car_and_bicycle = 30)
  (h8 : total_adults ≤ car_owners + motorcycle_owners + bicycle_owners - car_and_motorcycle - motorcycle_and_bicycle - car_and_bicycle) :
  car_owners - car_and_motorcycle - car_and_bicycle = 270 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_ownership_l3443_344399


namespace NUMINAMATH_CALUDE_river_length_estimate_l3443_344339

/-- Represents a measurement with an associated error probability -/
structure Measurement where
  value : ℝ
  error : ℝ
  errorProb : ℝ

/-- Calculates the best estimate and error probability given two measurements -/
def calculateEstimate (m1 m2 : Measurement) : ℝ × ℝ :=
  sorry

theorem river_length_estimate 
  (gsa awra : Measurement)
  (h1 : gsa.value = 402)
  (h2 : gsa.error = 0.5)
  (h3 : gsa.errorProb = 0.04)
  (h4 : awra.value = 403)
  (h5 : awra.error = 0.5)
  (h6 : awra.errorProb = 0.04) :
  calculateEstimate gsa awra = (402.5, 0.04) :=
sorry

end NUMINAMATH_CALUDE_river_length_estimate_l3443_344339


namespace NUMINAMATH_CALUDE_square_puzzle_l3443_344309

/-- Given a square with side length n satisfying the equation n^2 + 20 = (n + 1)^2 - 9,
    prove that the total number of small squares is 216. -/
theorem square_puzzle (n : ℕ) (h : n^2 + 20 = (n + 1)^2 - 9) : n^2 + 20 = 216 := by
  sorry

end NUMINAMATH_CALUDE_square_puzzle_l3443_344309


namespace NUMINAMATH_CALUDE_league_games_l3443_344331

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 20 → 
  total_games = 1900 → 
  total_games = (num_teams * (num_teams - 1) * games_per_matchup) / 2 → 
  games_per_matchup = 10 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l3443_344331


namespace NUMINAMATH_CALUDE_two_dogs_walking_time_l3443_344343

def dog_walking_earnings (base_charge : ℕ) (per_minute_charge : ℕ) 
  (dogs_1 : ℕ) (time_1 : ℕ) (dogs_2 : ℕ) (time_2 : ℕ) 
  (dogs_3 : ℕ) (time_3 : ℕ) : ℕ :=
  (dogs_1 * (base_charge + per_minute_charge * time_1)) +
  (dogs_2 * (base_charge + per_minute_charge * time_2)) +
  (dogs_3 * (base_charge + per_minute_charge * time_3))

theorem two_dogs_walking_time : 
  ∃ (time_2 : ℕ), 
    dog_walking_earnings 20 1 1 10 2 time_2 3 9 = 171 ∧ 
    time_2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_walking_time_l3443_344343


namespace NUMINAMATH_CALUDE_monotonic_function_implies_a_le_e_l3443_344335

/-- Given f(x) = 2xe^x - ax^2 - 2ax is monotonically increasing on [1, +∞), prove that a ≤ e -/
theorem monotonic_function_implies_a_le_e (a : ℝ) :
  (∀ x ≥ 1, Monotone (fun x => 2 * x * Real.exp x - a * x^2 - 2 * a * x)) →
  a ≤ Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_function_implies_a_le_e_l3443_344335


namespace NUMINAMATH_CALUDE_company_profit_achievable_l3443_344395

/-- Represents the company's financial model -/
structure CompanyModel where
  investment : ℝ
  production_cost : ℝ
  advertising_fee : ℝ
  min_price : ℝ
  max_price : ℝ
  sales_function : ℝ → ℝ

/-- Calculates the profit for a given price -/
def profit (model : CompanyModel) (price : ℝ) : ℝ :=
  let sales := model.sales_function price
  sales * (price - model.production_cost) - model.investment - model.advertising_fee

/-- Theorem stating that the company can achieve a total profit of 35 million over two years -/
theorem company_profit_achievable (model : CompanyModel) 
  (h1 : model.investment = 25)
  (h2 : model.production_cost = 20)
  (h3 : model.advertising_fee = 5)
  (h4 : model.min_price = 30)
  (h5 : model.max_price = 70)
  (h6 : ∀ x, model.sales_function x = -x + 150)
  (h7 : ∀ x, model.min_price ≤ x ∧ x ≤ model.max_price → 0 ≤ model.sales_function x) :
  ∃ (price1 price2 : ℝ), 
    model.min_price ≤ price1 ∧ price1 ≤ model.max_price ∧
    model.min_price ≤ price2 ∧ price2 ≤ model.max_price ∧
    profit model price1 + profit model price2 = 35 :=
  sorry


end NUMINAMATH_CALUDE_company_profit_achievable_l3443_344395


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3443_344372

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  (y^2 / a^2) - (x^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

-- Define the focus
def focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = -2

-- Define the asymptote slope
def asymptote_slope (slope : ℝ) : Prop :=
  slope = Real.sqrt 3

-- Theorem statement
theorem hyperbola_equation 
  (a b x y : ℝ) 
  (h1 : hyperbola a b x y) 
  (h2 : focus 0 (-2)) 
  (h3 : asymptote_slope (a/b)) : 
  (y^2 / 3) - x^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3443_344372


namespace NUMINAMATH_CALUDE_megan_total_songs_l3443_344397

/-- Represents the number of albums bought in each genre -/
structure AlbumCounts where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Represents the number of songs per album in each genre -/
structure SongsPerAlbum where
  country : ℕ
  pop : ℕ
  rock : ℕ
  jazz : ℕ

/-- Calculates the total number of songs bought -/
def totalSongs (counts : AlbumCounts) (songsPerAlbum : SongsPerAlbum) : ℕ :=
  counts.country * songsPerAlbum.country +
  counts.pop * songsPerAlbum.pop +
  counts.rock * songsPerAlbum.rock +
  counts.jazz * songsPerAlbum.jazz

/-- Theorem stating that Megan bought 160 songs in total -/
theorem megan_total_songs :
  let counts : AlbumCounts := ⟨2, 8, 5, 2⟩
  let songsPerAlbum : SongsPerAlbum := ⟨12, 7, 10, 15⟩
  totalSongs counts songsPerAlbum = 160 := by
  sorry

end NUMINAMATH_CALUDE_megan_total_songs_l3443_344397


namespace NUMINAMATH_CALUDE_brian_trip_distance_l3443_344327

/-- Calculates the distance traveled given car efficiency, initial tank capacity, and fuel used --/
def distanceTraveled (efficiency : ℝ) (initialTank : ℝ) (fuelUsed : ℝ) : ℝ :=
  efficiency * fuelUsed

/-- Represents Brian's trip --/
structure BrianTrip where
  efficiency : ℝ
  initialTank : ℝ
  remainingFuelFraction : ℝ
  drivingTime : ℝ

/-- Theorem stating the distance Brian traveled --/
theorem brian_trip_distance (trip : BrianTrip) 
  (h1 : trip.efficiency = 20)
  (h2 : trip.initialTank = 15)
  (h3 : trip.remainingFuelFraction = 3/7)
  (h4 : trip.drivingTime = 2) :
  ∃ (distance : ℝ), abs (distance - distanceTraveled trip.efficiency trip.initialTank (trip.initialTank * (1 - trip.remainingFuelFraction))) < 0.1 :=
by sorry

#check brian_trip_distance

end NUMINAMATH_CALUDE_brian_trip_distance_l3443_344327


namespace NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3443_344374

def g (x : ℝ) : ℝ := (x - 3)^2 - 4

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y : ℝ, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3443_344374


namespace NUMINAMATH_CALUDE_student_selection_probability_l3443_344348

/-- Given a set of 3 students where 2 are to be selected, 
    the probability of a specific student being selected is 2/3 -/
theorem student_selection_probability 
  (S : Finset Nat) 
  (h_card : S.card = 3) 
  (A : Nat) 
  (h_A_in_S : A ∈ S) : 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2 ∧ A ∈ pair} / 
  Nat.card {pair : Finset Nat | pair ⊆ S ∧ pair.card = 2} = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_probability_l3443_344348


namespace NUMINAMATH_CALUDE_inequality_proof_l3443_344351

theorem inequality_proof (p : ℝ) (hp : p > 1) :
  ∃ (K_p : ℝ), K_p > 0 ∧
  ∀ (x y : ℝ), (|x|^p + |y|^p = 2) →
  (x - y)^2 ≤ K_p * (4 - (x + y)^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3443_344351


namespace NUMINAMATH_CALUDE_susan_ate_six_candies_l3443_344380

/-- Represents Susan's candy consumption and purchases over a week -/
structure CandyWeek where
  dailyLimit : ℕ
  tuesdayBought : ℕ
  thursdayBought : ℕ
  fridayBought : ℕ
  leftAtEndOfWeek : ℕ
  totalSpent : ℕ

/-- Calculates the number of candies Susan ate during the week -/
def candiesEaten (week : CandyWeek) : ℕ :=
  week.tuesdayBought + week.thursdayBought + week.fridayBought - week.leftAtEndOfWeek

/-- Theorem stating that Susan ate 6 candies during the week -/
theorem susan_ate_six_candies (week : CandyWeek)
  (h1 : week.dailyLimit = 3)
  (h2 : week.tuesdayBought = 3)
  (h3 : week.thursdayBought = 5)
  (h4 : week.fridayBought = 2)
  (h5 : week.leftAtEndOfWeek = 4)
  (h6 : week.totalSpent = 9) :
  candiesEaten week = 6 := by
  sorry

end NUMINAMATH_CALUDE_susan_ate_six_candies_l3443_344380
