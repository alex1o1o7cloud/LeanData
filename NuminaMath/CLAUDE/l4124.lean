import Mathlib

namespace unique_solution_cubic_equation_l4124_412430

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 4 :=
by sorry

end unique_solution_cubic_equation_l4124_412430


namespace power_function_through_point_l4124_412491

theorem power_function_through_point (a : ℝ) : 
  (2 : ℝ) ^ a = (1 / 2 : ℝ) → a = -1 := by
sorry

end power_function_through_point_l4124_412491


namespace one_fourth_of_8_4_l4124_412497

theorem one_fourth_of_8_4 : (8.4 : ℚ) / 4 = 21 / 10 := by
  sorry

end one_fourth_of_8_4_l4124_412497


namespace even_monotone_inequality_l4124_412432

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem even_monotone_inequality (h1 : is_even f) (h2 : monotone_increasing_on_positive f) :
  f (-1) < f 2 ∧ f 2 < f (-3) :=
sorry

end even_monotone_inequality_l4124_412432


namespace tan_570_degrees_l4124_412420

theorem tan_570_degrees : Real.tan (570 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_570_degrees_l4124_412420


namespace count_four_digit_numbers_divisible_by_five_l4124_412423

theorem count_four_digit_numbers_divisible_by_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 :=
by
  sorry

end count_four_digit_numbers_divisible_by_five_l4124_412423


namespace floor_length_percentage_l4124_412471

theorem floor_length_percentage (length width area : ℝ) : 
  length = 23 ∧ 
  area = 529 / 3 ∧ 
  area = length * width → 
  length = width * 3 :=
by sorry

end floor_length_percentage_l4124_412471


namespace rubber_band_difference_l4124_412459

theorem rubber_band_difference (justine bailey ylona : ℕ) : 
  ylona = 24 →
  justine = ylona - 2 →
  bailey + 4 = 8 →
  justine > bailey →
  justine - bailey = 10 :=
by
  sorry

end rubber_band_difference_l4124_412459


namespace problem_solution_l4124_412473

def A (a : ℝ) : Set ℝ := {x | a - 2 < x ∧ x < a + 2}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a = 0}

theorem problem_solution :
  (∀ x, x ∈ (A 0 ∪ B 0) ↔ -2 < x ∧ x ≤ 2) ∧
  (∀ a, (Aᶜ a ∩ B a).Nonempty ↔ a ≤ 0 ∨ a ≥ 4) :=
sorry

end problem_solution_l4124_412473


namespace f_continuous_iff_a_eq_5_l4124_412480

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + 2 else 2*x + a

-- State the theorem
theorem f_continuous_iff_a_eq_5 (a : ℝ) :
  Continuous (f a) ↔ a = 5 := by
  sorry

end f_continuous_iff_a_eq_5_l4124_412480


namespace article_cost_l4124_412477

/-- 
Given an article that can be sold at two different prices, prove that the cost of the article
is 140 if the higher price yields a 5% greater gain than the lower price.
-/
theorem article_cost (selling_price_high selling_price_low : ℕ) 
  (h1 : selling_price_high = 350)
  (h2 : selling_price_low = 340)
  (h3 : selling_price_high - selling_price_low = 10) :
  ∃ (cost gain : ℕ),
    selling_price_low = cost + gain ∧
    selling_price_high = cost + gain + (gain * 5 / 100) ∧
    cost = 140 := by
  sorry

end article_cost_l4124_412477


namespace equation_solution_expression_result_l4124_412488

-- Problem 1
theorem equation_solution :
  ∃ y : ℝ, 4 * (y - 1) = 1 - 3 * (y - 3) ∧ y = 2 := by sorry

-- Problem 2
theorem expression_result :
  (-2)^3 / 4 + 6 * |1/3 - 1| - 1/2 * 14 = -5 := by sorry

end equation_solution_expression_result_l4124_412488


namespace percentage_caught_sampling_candy_l4124_412403

/-- The percentage of customers caught sampling candy -/
def percentage_caught (total_sample_percent : ℝ) (not_caught_ratio : ℝ) : ℝ :=
  total_sample_percent - (not_caught_ratio * total_sample_percent)

/-- Theorem stating the percentage of customers caught sampling candy -/
theorem percentage_caught_sampling_candy :
  let total_sample_percent : ℝ := 23.913043478260867
  let not_caught_ratio : ℝ := 0.08
  percentage_caught total_sample_percent not_caught_ratio = 22 := by
  sorry


end percentage_caught_sampling_candy_l4124_412403


namespace xiao_ming_age_problem_l4124_412411

/-- Proves that Xiao Ming was 7 years old when his father's age was 5 times Xiao Ming's age -/
theorem xiao_ming_age_problem (current_age : ℕ) (father_current_age : ℕ) 
  (h1 : current_age = 12) (h2 : father_current_age = 40) : 
  ∃ (past_age : ℕ), past_age = 7 ∧ father_current_age - (current_age - past_age) = 5 * past_age :=
by
  sorry

end xiao_ming_age_problem_l4124_412411


namespace teacup_box_ratio_l4124_412401

theorem teacup_box_ratio : 
  let total_boxes : ℕ := 26
  let pan_boxes : ℕ := 6
  let cups_per_box : ℕ := 5 * 4
  let broken_cups_per_box : ℕ := 2
  let remaining_cups : ℕ := 180
  
  let non_pan_boxes : ℕ := total_boxes - pan_boxes
  let teacup_boxes : ℕ := remaining_cups / (cups_per_box - broken_cups_per_box)
  let decoration_boxes : ℕ := non_pan_boxes - teacup_boxes
  
  (decoration_boxes : ℚ) / total_boxes = 5 / 13 :=
by sorry

end teacup_box_ratio_l4124_412401


namespace complex_equation_solution_l4124_412492

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.I * z = z + a * Complex.I) 
  (h2 : Complex.abs z = Real.sqrt 5) 
  (h3 : a > 0) : 
  a = 2 := by
  sorry

end complex_equation_solution_l4124_412492


namespace trajectory_of_moving_point_l4124_412427

/-- The trajectory of a point M(x, y) that is twice as far from A(-4, 0) as it is from B(2, 0) -/
theorem trajectory_of_moving_point (x y : ℝ) : 
  (((x + 4)^2 + y^2).sqrt = 2 * ((x - 2)^2 + y^2).sqrt) ↔ 
  (x^2 + y^2 - 8*x = 0) :=
sorry

end trajectory_of_moving_point_l4124_412427


namespace at_least_two_satisfying_functions_l4124_412400

/-- A function satisfying the given equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + (f y)^2) = x + y^2

/-- Theorem stating that there are at least two distinct functions satisfying the equation -/
theorem at_least_two_satisfying_functions :
  ∃ f g : ℝ → ℝ, f ≠ g ∧ SatisfyingFunction f ∧ SatisfyingFunction g :=
sorry

end at_least_two_satisfying_functions_l4124_412400


namespace metallic_sheet_width_l4124_412416

/-- Represents the dimensions and properties of a metallic sheet and the box formed from it. -/
structure MetallicSheet where
  length : ℝ
  width : ℝ
  cutSize : ℝ
  boxVolume : ℝ

/-- Theorem stating the width of the metallic sheet given the conditions -/
theorem metallic_sheet_width (sheet : MetallicSheet)
  (h1 : sheet.length = 50)
  (h2 : sheet.cutSize = 8)
  (h3 : sheet.boxVolume = 5440)
  (h4 : sheet.boxVolume = (sheet.length - 2 * sheet.cutSize) * (sheet.width - 2 * sheet.cutSize) * sheet.cutSize) :
  sheet.width = 36 := by
  sorry


end metallic_sheet_width_l4124_412416


namespace first_interest_rate_is_ten_percent_l4124_412468

/-- Proves that the first interest rate is 10% given the problem conditions -/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (second_rate : ℚ)
  (total_profit : ℕ)
  (h1 : total_amount = 70000)
  (h2 : first_part = 60000)
  (h3 : second_part = 10000)
  (h4 : total_amount = first_part + second_part)
  (h5 : second_rate = 20 / 100)
  (h6 : total_profit = 8000)
  (h7 : total_profit = first_part * (first_rate / 100) + second_part * (second_rate / 100)) :
  first_rate = 10 := by
  sorry

end first_interest_rate_is_ten_percent_l4124_412468


namespace perpendicular_lines_coefficient_l4124_412466

theorem perpendicular_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, ax + y - 1 = 0 ↔ x + ay - 1 = 0) → a = 0 := by
  sorry

end perpendicular_lines_coefficient_l4124_412466


namespace quadratic_inequality_solution_range_l4124_412495

theorem quadratic_inequality_solution_range (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 :=
sorry

end quadratic_inequality_solution_range_l4124_412495


namespace total_pools_calculation_l4124_412478

/-- The number of Pat's Pool Supply stores -/
def pool_supply_stores : ℕ := 4

/-- The number of Pat's Ark & Athletic Wear stores -/
def ark_athletic_stores : ℕ := 6

/-- The ratio of pools between Pat's Pool Supply and Pat's Ark & Athletic Wear stores -/
def pool_ratio : ℕ := 5

/-- The initial number of pools at one Pat's Ark & Athletic Wear store -/
def initial_pools : ℕ := 200

/-- The number of pools sold at one Pat's Ark & Athletic Wear store -/
def pools_sold : ℕ := 8

/-- The number of pools returned to one Pat's Ark & Athletic Wear store -/
def pools_returned : ℕ := 3

/-- The total number of swimming pools across all stores -/
def total_pools : ℕ := 5070

theorem total_pools_calculation :
  let current_pools := initial_pools - pools_sold + pools_returned
  let supply_store_pools := pool_ratio * current_pools
  total_pools = ark_athletic_stores * current_pools + pool_supply_stores * supply_store_pools := by
  sorry

end total_pools_calculation_l4124_412478


namespace largest_consecutive_even_integer_l4124_412463

theorem largest_consecutive_even_integer : ∃ n : ℕ,
  (n - 8) + (n - 6) + (n - 4) + (n - 2) + n = 2 * (25 * 26 / 2) ∧
  n % 2 = 0 ∧
  n = 134 :=
by
  sorry

end largest_consecutive_even_integer_l4124_412463


namespace max_p_plus_q_l4124_412446

theorem max_p_plus_q (p q : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → 2*p*x^2 + q*x - p + 1 ≥ 0) → 
  p + q ≤ 2 :=
by sorry

end max_p_plus_q_l4124_412446


namespace fiftieth_number_is_fourteen_l4124_412457

/-- Defines the cumulative sum of elements up to the nth row -/
def cumulativeSum (n : ℕ) : ℕ := 
  (List.range n).foldl (fun acc i => acc + 2 * (i + 1)) 0

/-- Defines the value of each element in the nth row -/
def rowValue (n : ℕ) : ℕ := 2 * n

theorem fiftieth_number_is_fourteen : 
  ∃ (n : ℕ), cumulativeSum n < 50 ∧ 50 ≤ cumulativeSum (n + 1) ∧ rowValue (n + 1) = 14 := by
  sorry

end fiftieth_number_is_fourteen_l4124_412457


namespace table_stool_equation_correctness_l4124_412483

/-- Represents a scenario with tables and stools -/
structure TableStoolScenario where
  numTables : ℕ
  numStools : ℕ
  totalItems : ℕ
  totalLegs : ℕ
  h_totalItems : numTables + numStools = totalItems
  h_totalLegs : 4 * numTables + 3 * numStools = totalLegs

/-- The correct system of equations for the given scenario -/
def correctSystem (x y : ℕ) : Prop :=
  x + y = 12 ∧ 4 * x + 3 * y = 40

theorem table_stool_equation_correctness :
  ∀ (scenario : TableStoolScenario),
    scenario.totalItems = 12 →
    scenario.totalLegs = 40 →
    correctSystem scenario.numTables scenario.numStools :=
by sorry

end table_stool_equation_correctness_l4124_412483


namespace fruit_seller_apples_l4124_412402

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples * 40 / 100 = 560) → initial_apples = 1400 := by
  sorry

end fruit_seller_apples_l4124_412402


namespace frequency_of_eighth_group_l4124_412476

theorem frequency_of_eighth_group 
  (num_rectangles : ℕ) 
  (sample_size : ℕ) 
  (area_last_rectangle : ℝ) 
  (sum_area_other_rectangles : ℝ) :
  num_rectangles = 8 →
  sample_size = 200 →
  area_last_rectangle = (1/4 : ℝ) * sum_area_other_rectangles →
  (area_last_rectangle / (area_last_rectangle + sum_area_other_rectangles)) * sample_size = 40 :=
by sorry

end frequency_of_eighth_group_l4124_412476


namespace min_value_of_expression_equality_attained_l4124_412465

theorem min_value_of_expression (x : ℝ) : 
  (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008 :=
by sorry

theorem equality_attained : 
  ∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008 :=
by sorry

end min_value_of_expression_equality_attained_l4124_412465


namespace reading_time_calculation_l4124_412472

def total_homework_time : ℕ := 120
def math_time : ℕ := 25
def spelling_time : ℕ := 30
def history_time : ℕ := 20
def science_time : ℕ := 15

theorem reading_time_calculation :
  total_homework_time - (math_time + spelling_time + history_time + science_time) = 30 := by
  sorry

end reading_time_calculation_l4124_412472


namespace c_share_of_rent_l4124_412499

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a given usage -/
def calculateShare (u : Usage) (totalRent : ℕ) (totalUsage : ℕ) : ℚ :=
  (u.oxen * u.months : ℚ) / totalUsage * totalRent

/-- The main theorem stating C's share of the rent -/
theorem c_share_of_rent :
  let a := Usage.mk 10 7
  let b := Usage.mk 12 5
  let c := Usage.mk 15 3
  let totalRent := 210
  let totalUsage := a.oxen * a.months + b.oxen * b.months + c.oxen * c.months
  calculateShare c totalRent totalUsage = 54 := by
  sorry


end c_share_of_rent_l4124_412499


namespace smallest_dual_base_representation_l4124_412417

theorem smallest_dual_base_representation : ∃ n : ℕ, ∃ a b : ℕ, 
  (a > 2 ∧ b > 2) ∧ 
  (1 * a + 3 = n) ∧ 
  (3 * b + 1 = n) ∧
  (∀ m : ℕ, ∀ c d : ℕ, 
    (c > 2 ∧ d > 2) → 
    (1 * c + 3 = m) → 
    (3 * d + 1 = m) → 
    m ≥ n) ∧
  n = 13 :=
by sorry

end smallest_dual_base_representation_l4124_412417


namespace triangle_vector_dot_product_l4124_412470

/-- Given a triangle ABC with vectors AB and AC, prove that the dot product of AB and BC equals 5 -/
theorem triangle_vector_dot_product (A B C : ℝ × ℝ) : 
  let AB : ℝ × ℝ := (2, 3)
  let AC : ℝ × ℝ := (3, 4)
  let BC : ℝ × ℝ := AC - AB
  (AB.1 * BC.1 + AB.2 * BC.2) = 5 := by
  sorry

end triangle_vector_dot_product_l4124_412470


namespace calculation_proofs_l4124_412447

theorem calculation_proofs (x y a b : ℝ) :
  (((1/2) * x * y)^2 * (6 * x^2 * y) = (3/2) * x^4 * y^3) ∧
  ((2*a + b)^2 = 4*a^2 + 4*a*b + b^2) := by sorry

end calculation_proofs_l4124_412447


namespace expand_expression_l4124_412434

theorem expand_expression (x : ℝ) : (7*x + 11 - 3) * 4*x = 28*x^2 + 32*x := by
  sorry

end expand_expression_l4124_412434


namespace brick_width_is_10cm_l4124_412433

/-- Proves that the width of each brick is 10 centimeters, given the courtyard dimensions,
    brick length, and total number of bricks. -/
theorem brick_width_is_10cm 
  (courtyard_length : ℝ) 
  (courtyard_width : ℝ) 
  (brick_length : ℝ) 
  (total_bricks : ℕ) 
  (h1 : courtyard_length = 18) 
  (h2 : courtyard_width = 16) 
  (h3 : brick_length = 0.2) 
  (h4 : total_bricks = 14400) : 
  ∃ (brick_width : ℝ), brick_width = 0.1 ∧ 
    courtyard_length * courtyard_width * 100 * 100 = 
    brick_length * brick_width * total_bricks * 10000 :=
by sorry

end brick_width_is_10cm_l4124_412433


namespace hundredth_term_value_l4124_412448

/-- A geometric sequence with first term 5 and second term -15 -/
def geometric_sequence (n : ℕ) : ℚ :=
  5 * (-3)^(n - 1)

/-- The 100th term of the geometric sequence -/
def a_100 : ℚ := geometric_sequence 100

theorem hundredth_term_value : a_100 = -5 * 3^99 := by
  sorry

end hundredth_term_value_l4124_412448


namespace ice_cream_combinations_l4124_412422

theorem ice_cream_combinations :
  (5 : ℕ) * (Nat.choose 7 3) = 175 := by
  sorry

end ice_cream_combinations_l4124_412422


namespace tangency_point_satisfies_equations_tangency_point_is_unique_l4124_412409

/-- The point of tangency for two parabolas -/
def point_of_tangency : ℝ × ℝ := (-7, -25)

/-- First parabola equation -/
def parabola1 (x y : ℝ) : Prop := y = x^2 + 17*x + 40

/-- Second parabola equation -/
def parabola2 (x y : ℝ) : Prop := x = y^2 + 51*y + 650

/-- Theorem stating that the point_of_tangency satisfies both parabola equations -/
theorem tangency_point_satisfies_equations :
  parabola1 point_of_tangency.1 point_of_tangency.2 ∧
  parabola2 point_of_tangency.1 point_of_tangency.2 :=
by sorry

/-- Theorem stating that the point_of_tangency is the unique point satisfying both equations -/
theorem tangency_point_is_unique :
  ∀ (x y : ℝ), parabola1 x y ∧ parabola2 x y → (x, y) = point_of_tangency :=
by sorry

end tangency_point_satisfies_equations_tangency_point_is_unique_l4124_412409


namespace recreation_spending_comparison_l4124_412418

theorem recreation_spending_comparison (wages_last_week : ℝ) : 
  let recreation_last_week := 0.60 * wages_last_week
  let wages_this_week := 0.90 * wages_last_week
  let recreation_this_week := 0.70 * wages_this_week
  recreation_this_week / recreation_last_week = 1.05 := by
sorry

end recreation_spending_comparison_l4124_412418


namespace p_or_q_is_true_l4124_412496

theorem p_or_q_is_true : 
  let p : Prop := 2 + 3 = 5
  let q : Prop := 5 < 4
  p ∨ q := by sorry

end p_or_q_is_true_l4124_412496


namespace laura_change_l4124_412474

/-- The change Laura received after purchasing pants and shirts -/
theorem laura_change (pants_cost shirt_cost : ℕ) (pants_quantity shirt_quantity : ℕ) (amount_given : ℕ) : 
  pants_cost = 54 → 
  pants_quantity = 2 → 
  shirt_cost = 33 → 
  shirt_quantity = 4 → 
  amount_given = 250 → 
  amount_given - (pants_cost * pants_quantity + shirt_cost * shirt_quantity) = 10 := by
  sorry

end laura_change_l4124_412474


namespace compare_with_negative_three_l4124_412450

theorem compare_with_negative_three : 
  let numbers : List ℝ := [-3, 2, 0, -4]
  numbers.filter (λ x => x < -3) = [-4] := by sorry

end compare_with_negative_three_l4124_412450


namespace outfits_with_restrictions_l4124_412436

/-- The number of unique outfits that can be made with shirts and pants, with restrictions -/
def uniqueOutfits (shirts : ℕ) (pants : ℕ) (restrictedShirts : ℕ) (restrictedPants : ℕ) : ℕ :=
  shirts * pants - restrictedShirts * restrictedPants

/-- Theorem stating the number of unique outfits under given conditions -/
theorem outfits_with_restrictions :
  uniqueOutfits 5 6 1 2 = 28 := by
  sorry

#eval uniqueOutfits 5 6 1 2

end outfits_with_restrictions_l4124_412436


namespace abs_neg_2023_l4124_412412

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by sorry

end abs_neg_2023_l4124_412412


namespace floor_product_inequality_l4124_412444

theorem floor_product_inequality (m n : ℕ+) :
  ⌊Real.sqrt 2 * m⌋ * ⌊Real.sqrt 7 * n⌋ < ⌊Real.sqrt 14 * (m * n)⌋ := by
  sorry

end floor_product_inequality_l4124_412444


namespace fourth_power_nested_root_l4124_412414

theorem fourth_power_nested_root : 
  (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) := by
  sorry

end fourth_power_nested_root_l4124_412414


namespace complex_calculation_l4124_412493

theorem complex_calculation : (2 - I) / (1 - I) - I = 3/2 - 1/2 * I := by
  sorry

end complex_calculation_l4124_412493


namespace f_increasing_l4124_412445

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := by sorry

end f_increasing_l4124_412445


namespace complexity_theorem_l4124_412454

-- Define complexity of a positive integer
def complexity (n : ℕ) : ℕ := sorry

-- Define the property for part (a)
def property_a (n : ℕ) : Prop :=
  ∀ m : ℕ, n ≤ m → m ≤ 2*n → complexity m ≤ complexity n

-- Define the property for part (b)
def property_b (n : ℕ) : Prop :=
  ∀ m : ℕ, n < m → m < 2*n → complexity m < complexity n

theorem complexity_theorem :
  (∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n = 2^k → property_a n) ∧
  (¬ ∃ n : ℕ, n > 1 ∧ property_b n) := by sorry

end complexity_theorem_l4124_412454


namespace negative_three_squared_l4124_412479

theorem negative_three_squared : (-3 : ℤ)^2 = 9 := by
  sorry

end negative_three_squared_l4124_412479


namespace product_equals_eight_l4124_412494

theorem product_equals_eight : 
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := by
  sorry

end product_equals_eight_l4124_412494


namespace bathroom_flooring_area_l4124_412408

/-- The total area of hardwood flooring installed in Nancy's bathroom -/
def total_area (central_length central_width hallway_length hallway_width : ℝ) : ℝ :=
  central_length * central_width + hallway_length * hallway_width

/-- Proof that the total area of hardwood flooring installed in Nancy's bathroom is 124 square feet -/
theorem bathroom_flooring_area :
  total_area 10 10 6 4 = 124 := by
  sorry

end bathroom_flooring_area_l4124_412408


namespace easter_egg_hunt_l4124_412481

theorem easter_egg_hunt (kevin bonnie cheryl george : ℕ) 
  (h1 : kevin = 5)
  (h2 : bonnie = 13)
  (h3 : cheryl = 56)
  (h4 : cheryl = kevin + bonnie + george + 29) :
  george = 9 := by
  sorry

end easter_egg_hunt_l4124_412481


namespace power_value_from_condition_l4124_412451

theorem power_value_from_condition (x y : ℝ) : 
  |x - 3| + (y + 3)^2 = 0 → y^x = -27 := by sorry

end power_value_from_condition_l4124_412451


namespace solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l4124_412484

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set_f_leq_x_plus_1 :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, f x ≥ k :=
sorry

-- Theorem 3: Non-existence of positive a, b satisfying conditions
theorem no_positive_a_b :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 :=
sorry

end solution_set_f_leq_x_plus_1_min_value_f_no_positive_a_b_l4124_412484


namespace rational_with_smallest_abs_value_l4124_412464

theorem rational_with_smallest_abs_value : ∃ q : ℚ, |q| < |1| := by
  -- The proof would go here, but we're only writing the statement
  sorry

end rational_with_smallest_abs_value_l4124_412464


namespace minuend_subtrahend_difference_problem_l4124_412490

theorem minuend_subtrahend_difference_problem :
  ∃ (a b c : ℤ),
    (a + b + c = 1024) ∧
    (c = b - 88) ∧
    (a = b + c) ∧
    (a = 712) ∧
    (b = 400) ∧
    (c = 312) := by
  sorry

end minuend_subtrahend_difference_problem_l4124_412490


namespace ratio_equality_l4124_412424

theorem ratio_equality (x y z : ℝ) 
  (h1 : x * y * z ≠ 0) 
  (h2 : 2 * x * y = 3 * y * z) 
  (h3 : 3 * y * z = 5 * x * z) : 
  (x + 3 * y - 3 * z) / (x + 3 * y - 6 * z) = 2 := by
  sorry

end ratio_equality_l4124_412424


namespace fixed_charge_is_28_l4124_412475

-- Define the variables
def fixed_charge : ℝ := sorry
def january_call_charge : ℝ := sorry
def february_call_charge : ℝ := sorry

-- Define the conditions
axiom january_bill : fixed_charge + january_call_charge = 52
axiom february_bill : fixed_charge + february_call_charge = 76
axiom february_double_january : february_call_charge = 2 * january_call_charge

-- Theorem to prove
theorem fixed_charge_is_28 : fixed_charge = 28 := by sorry

end fixed_charge_is_28_l4124_412475


namespace jenny_distance_relationship_l4124_412453

/-- Given Jenny's running and walking speeds and times, prove the relationship between distances -/
theorem jenny_distance_relationship 
  (x : ℝ) -- Jenny's running speed in miles per hour
  (y : ℝ) -- Jenny's walking speed in miles per hour
  (r : ℝ) -- Time spent running in minutes
  (w : ℝ) -- Time spent walking in minutes
  (d : ℝ) -- Difference in distance (running - walking) in miles
  (hx : x > 0) -- Assumption: running speed is positive
  (hy : y > 0) -- Assumption: walking speed is positive
  (hr : r ≥ 0) -- Assumption: time spent running is non-negative
  (hw : w ≥ 0) -- Assumption: time spent walking is non-negative
  : x * r - y * w = 60 * d :=
by sorry

end jenny_distance_relationship_l4124_412453


namespace min_value_expression_l4124_412426

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hc : c > 1) :
  (((a^2 + 1) / (2*a*b) - 1) * c + (Real.sqrt 2 / (c - 1))) ≥ 3 * Real.sqrt 2 :=
by sorry

end min_value_expression_l4124_412426


namespace three_year_officer_pays_51_l4124_412421

/-- The price of duty shoes for an officer who has served at least three years -/
def price_for_three_year_officer : ℝ :=
  let full_price : ℝ := 85
  let first_year_discount : ℝ := 0.20
  let three_year_discount : ℝ := 0.25
  let discounted_price : ℝ := full_price * (1 - first_year_discount)
  discounted_price * (1 - three_year_discount)

/-- Theorem stating that an officer who has served at least three years pays $51 for duty shoes -/
theorem three_year_officer_pays_51 :
  price_for_three_year_officer = 51 := by
  sorry

end three_year_officer_pays_51_l4124_412421


namespace keystone_arch_angle_l4124_412425

/-- Represents a keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_congruent : Bool
  trapezoids_isosceles : Bool
  bottom_sides_horizontal : Bool

/-- Calculates the smaller interior angle of a trapezoid in a keystone arch -/
def smaller_interior_angle (arch : KeystoneArch) : ℝ :=
  if arch.num_trapezoids = 8 ∧ 
     arch.trapezoids_congruent ∧ 
     arch.trapezoids_isosceles ∧ 
     arch.bottom_sides_horizontal
  then 78.75
  else 0

/-- Theorem stating that the smaller interior angle of each trapezoid in the specified keystone arch is 78.75° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 8 ∧ 
  arch.trapezoids_congruent ∧ 
  arch.trapezoids_isosceles ∧ 
  arch.bottom_sides_horizontal →
  smaller_interior_angle arch = 78.75 := by
  sorry

end keystone_arch_angle_l4124_412425


namespace pentagon_area_l4124_412437

/-- The area of a pentagon with specific dimensions -/
theorem pentagon_area : 
  ∀ (right_triangle_base right_triangle_height trapezoid_base1 trapezoid_base2 trapezoid_height : ℝ),
  right_triangle_base = 28 →
  right_triangle_height = 30 →
  trapezoid_base1 = 25 →
  trapezoid_base2 = 18 →
  trapezoid_height = 39 →
  (1/2 * right_triangle_base * right_triangle_height) + 
  (1/2 * (trapezoid_base1 + trapezoid_base2) * trapezoid_height) = 1257 :=
by sorry

end pentagon_area_l4124_412437


namespace max_value_x_plus_1000y_l4124_412469

theorem max_value_x_plus_1000y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x + 2018 / y = 1000) (eq2 : 9 / x + y = 1) :
  ∃ (x' y' : ℝ), x' + 2018 / y' = 1000 ∧ 9 / x' + y' = 1 ∧
  ∀ (a b : ℝ), a + 2018 / b = 1000 → 9 / a + b = 1 → x' + 1000 * y' ≥ a + 1000 * b ∧
  x' + 1000 * y' = 1991 :=
sorry

end max_value_x_plus_1000y_l4124_412469


namespace two_digit_number_property_l4124_412487

def P (n : Nat) : Nat :=
  (n / 10) * (n % 10)

def S (n : Nat) : Nat :=
  (n / 10) + (n % 10)

theorem two_digit_number_property : ∃! N : Nat, 
  10 ≤ N ∧ N < 100 ∧ N = P N + 2 * S N :=
by
  sorry

end two_digit_number_property_l4124_412487


namespace quadratic_function_m_range_l4124_412410

/-- A quadratic function f(x) = a + bx - x^2 satisfying certain conditions -/
def f (a b : ℝ) (x : ℝ) : ℝ := a + b * x - x^2

/-- The theorem stating the range of m for the given conditions -/
theorem quadratic_function_m_range (a b m : ℝ) :
  (∀ x, f a b (1 + x) = f a b (1 - x)) →
  (∀ x ≤ 4, Monotone (fun x => f a b (x + m))) →
  m ≤ -3 :=
sorry

end quadratic_function_m_range_l4124_412410


namespace complex_expression_equality_l4124_412406

theorem complex_expression_equality : 
  let x := (3 + 3/8)^(2/3) - (5 + 4/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  x / 0.0625^(1/4) = 23/150 := by
  sorry

end complex_expression_equality_l4124_412406


namespace zhang_fei_probabilities_l4124_412404

/-- The set of events Zhang Fei can participate in -/
inductive Event : Type
  | LongJump : Event
  | Meters100 : Event
  | Meters200 : Event
  | Meters400 : Event

/-- The probability of selecting an event -/
def selectProbability (e : Event) : ℚ :=
  1 / 4

/-- The probability of selecting two specific events when choosing at most two events -/
def selectTwoEventsProbability (e1 e2 : Event) : ℚ :=
  2 / 12

theorem zhang_fei_probabilities :
  (selectProbability Event.LongJump = 1 / 4) ∧
  (selectTwoEventsProbability Event.LongJump Event.Meters100 = 1 / 6) := by
  sorry

end zhang_fei_probabilities_l4124_412404


namespace cake_muffin_probability_l4124_412482

theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 := by
sorry

end cake_muffin_probability_l4124_412482


namespace probability_of_valid_triangle_l4124_412438

-- Define a regular 15-gon
def regular_15gon : Set (ℝ × ℝ) := sorry

-- Define a function to get all segments in the 15-gon
def all_segments (polygon : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define a function to check if three segments form a triangle with positive area
def forms_triangle (s1 s2 s3 : Set (ℝ × ℝ)) : Prop := sorry

-- Define the total number of ways to choose 3 segments
def total_combinations : ℕ := Nat.choose 105 3

-- Define the number of valid triangles
def valid_triangles : ℕ := sorry

-- Theorem statement
theorem probability_of_valid_triangle :
  (valid_triangles : ℚ) / total_combinations = 713 / 780 := by sorry

end probability_of_valid_triangle_l4124_412438


namespace exponent_multiplication_l4124_412428

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l4124_412428


namespace rational_sqrt_equation_zero_l4124_412435

theorem rational_sqrt_equation_zero (a b c : ℚ) 
  (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end rational_sqrt_equation_zero_l4124_412435


namespace min_colors_correct_l4124_412431

/-- The number of distribution centers to be represented -/
def num_centers : ℕ := 12

/-- Calculates the number of unique representations possible with n colors -/
def num_representations (n : ℕ) : ℕ := n + n.choose 2

/-- Checks if a given number of colors is sufficient to represent all centers -/
def is_sufficient (n : ℕ) : Prop := num_representations n ≥ num_centers

/-- The minimum number of colors needed -/
def min_colors : ℕ := 5

/-- Theorem stating that min_colors is the minimum number of colors needed -/
theorem min_colors_correct :
  is_sufficient min_colors ∧ ∀ k < min_colors, ¬is_sufficient k :=
sorry

end min_colors_correct_l4124_412431


namespace white_balls_count_l4124_412419

theorem white_balls_count (a : ℕ) : 
  (a : ℝ) / (a + 3) = 4/5 → a = 12 :=
by
  sorry

end white_balls_count_l4124_412419


namespace smallest_n_for_g_greater_than_15_l4124_412462

def g (n : ℕ+) : ℕ := 
  (Nat.digits 10 ((10^n.val) / (7^n.val))).sum

theorem smallest_n_for_g_greater_than_15 : 
  ∀ k : ℕ+, k < 12 → g k ≤ 15 ∧ g 12 > 15 := by sorry

end smallest_n_for_g_greater_than_15_l4124_412462


namespace planes_parallel_implies_lines_parallel_l4124_412452

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the parallel relation for planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_implies_lines_parallel
  (α β γ : Plane) (m n : Line)
  (h1 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h2 : intersect α γ = m)
  (h3 : intersect β γ = n)
  (h4 : parallel_planes α β) :
  parallel_lines m n :=
sorry

end planes_parallel_implies_lines_parallel_l4124_412452


namespace complement_of_A_wrt_U_l4124_412407

def U : Set Nat := {1, 2, 3}
def A : Set Nat := {1, 2}

theorem complement_of_A_wrt_U : (U \ A) = {3} := by sorry

end complement_of_A_wrt_U_l4124_412407


namespace sugar_price_proof_l4124_412456

/-- Proves that given the initial price of sugar as 6 Rs/kg, a new price of 7.50 Rs/kg, 
    and a reduction in consumption of 19.999999999999996%, the initial price of sugar is 6 Rs/kg. -/
theorem sugar_price_proof (initial_price : ℝ) (new_price : ℝ) (consumption_reduction : ℝ) : 
  initial_price = 6 ∧ new_price = 7.5 ∧ consumption_reduction = 19.999999999999996 → initial_price = 6 := by
  sorry

end sugar_price_proof_l4124_412456


namespace function_relationship_l4124_412467

/-- Given functions f and g, and constants A, B, and C, prove the relationship between A, B, and C. -/
theorem function_relationship (A B C : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 2 * B^2 * x + 3
  let g := fun x => B * x + 1
  f (g 1) = C →
  A = (C + 2 * B^3 + 2 * B^2 - 3) / (B^2 + 2 * B + 1) := by
  sorry

end function_relationship_l4124_412467


namespace camera_price_difference_l4124_412458

-- Define the list price
def list_price : ℚ := 49.95

-- Define the price at Budget Buys
def budget_buys_price : ℚ := list_price - 10

-- Define the price at Value Mart
def value_mart_price : ℚ := list_price * (1 - 0.2)

-- Theorem to prove
theorem camera_price_difference :
  (max budget_buys_price value_mart_price - min budget_buys_price value_mart_price) * 100 = 1 := by
  sorry


end camera_price_difference_l4124_412458


namespace x_squared_mod_20_l4124_412405

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 20]) 
  (h2 : 4 * x ≡ 12 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
  sorry

end x_squared_mod_20_l4124_412405


namespace sufficient_not_necessary_l4124_412440

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + (a + 1) * y + 1
  l2 : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- Perpendicularity condition for two lines -/
def isPerpendicular (lines : TwoLines) : Prop :=
  lines.a * 1 + (lines.a + 1) * lines.a = 0

/-- Theorem stating that a = -2 is a sufficient but not necessary condition for perpendicularity -/
theorem sufficient_not_necessary (lines : TwoLines) :
  (lines.a = -2 → isPerpendicular lines) ∧
  ¬(isPerpendicular lines → lines.a = -2) := by
  sorry

end sufficient_not_necessary_l4124_412440


namespace initial_balance_is_20_l4124_412498

def football_club_balance (initial_balance : ℝ) : Prop :=
  let players_sold := 2
  let price_per_sold_player := 10
  let players_bought := 4
  let price_per_bought_player := 15
  let final_balance := 60
  
  initial_balance + players_sold * price_per_sold_player - 
  players_bought * price_per_bought_player = final_balance

theorem initial_balance_is_20 : 
  ∃ (initial_balance : ℝ), football_club_balance initial_balance ∧ initial_balance = 20 :=
by
  sorry

end initial_balance_is_20_l4124_412498


namespace expression_simplification_l4124_412415

theorem expression_simplification (a b c x : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  (x + a)^2 / ((a - b) * (a - c)) + (x + b)^2 / ((b - a) * (b - c)) + (x + c)^2 / ((c - a) * (c - b)) =
  a * x + b * x + c * x - a - b - c :=
by sorry

end expression_simplification_l4124_412415


namespace intersection_range_l4124_412489

theorem intersection_range (k : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ = Real.sqrt (y₁^2 + 6) ∧ 
    x₂ = Real.sqrt (y₂^2 + 6)) →
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end intersection_range_l4124_412489


namespace reflected_ray_equation_l4124_412461

/-- The equation of a reflected ray given an incident ray and a reflecting line -/
theorem reflected_ray_equation 
  (incident_ray : ℝ → ℝ → Prop) 
  (reflecting_line : ℝ → ℝ → Prop) 
  (reflected_ray : ℝ → ℝ → Prop) : 
  (∀ x y, incident_ray x y ↔ x - 2*y + 3 = 0) →
  (∀ x y, reflecting_line x y ↔ y = x) →
  (∀ x y, reflected_ray x y ↔ 2*x - y - 3 = 0) := by
sorry

end reflected_ray_equation_l4124_412461


namespace flow_rates_theorem_l4124_412441

/-- Represents an irrigation channel in the system -/
inductive Channel
| AB | BC | CD | DE | BG | GD | GF | FE

/-- Represents a node in the irrigation system -/
inductive Node
| A | B | C | D | E | F | G | H

/-- The flow rate in a channel -/
def flow_rate (c : Channel) : ℝ := sorry

/-- The total input flow rate -/
def q₀ : ℝ := sorry

/-- The irrigation system is symmetric -/
axiom symmetric_system : ∀ c₁ c₂ : Channel, flow_rate c₁ = flow_rate c₂

/-- The sum of flow rates remains constant along any path -/
axiom constant_flow_sum : ∀ path : List Channel, 
  (∀ c ∈ path, c ∈ [Channel.AB, Channel.BC, Channel.CD, Channel.DE, Channel.BG, Channel.GD, Channel.GF, Channel.FE]) →
  (List.sum (path.map flow_rate) = q₀)

/-- Theorem stating the flow rates in channels DE, BC, and GF -/
theorem flow_rates_theorem :
  flow_rate Channel.DE = (4/7) * q₀ ∧
  flow_rate Channel.BC = (2/7) * q₀ ∧
  flow_rate Channel.GF = (3/7) * q₀ := by
  sorry

end flow_rates_theorem_l4124_412441


namespace complex_integer_sum_of_squares_l4124_412443

theorem complex_integer_sum_of_squares (x y : ℤ) :
  (∃ (a b c d : ℤ), x + y * I = (a + b * I)^2 + (c + d * I)^2) ↔ Even y := by
  sorry

end complex_integer_sum_of_squares_l4124_412443


namespace workshop_average_salary_l4124_412442

/-- Proves that the average salary of all workers is 8000 Rs given the specified conditions -/
theorem workshop_average_salary
  (total_workers : ℕ)
  (technician_count : ℕ)
  (technician_avg_salary : ℕ)
  (non_technician_avg_salary : ℕ)
  (h1 : total_workers = 14)
  (h2 : technician_count = 7)
  (h3 : technician_avg_salary = 10000)
  (h4 : non_technician_avg_salary = 6000) :
  (technician_count * technician_avg_salary + (total_workers - technician_count) * non_technician_avg_salary) / total_workers = 8000 :=
by sorry

end workshop_average_salary_l4124_412442


namespace disjoint_subsets_remainder_l4124_412429

def T : Finset Nat := Finset.range 15

def disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) (h : S = T) : 
  disjoint_subsets S % 500 = 186 := by
  sorry

end disjoint_subsets_remainder_l4124_412429


namespace third_day_sales_formula_l4124_412460

/-- Represents the sales of sportswear over three days -/
structure SportswearSales where
  /-- Sales on the first day -/
  first_day : ℕ
  /-- Parameter m used in calculations -/
  m : ℕ

/-- Calculates the sales on the second day -/
def second_day_sales (s : SportswearSales) : ℤ :=
  3 * s.first_day - 3 * s.m

/-- Calculates the sales on the third day -/
def third_day_sales (s : SportswearSales) : ℤ :=
  second_day_sales s + s.m

/-- Theorem stating that the third day sales equal 3a - 2m -/
theorem third_day_sales_formula (s : SportswearSales) :
  third_day_sales s = 3 * s.first_day - 2 * s.m :=
by
  sorry

end third_day_sales_formula_l4124_412460


namespace three_cubes_of_27_equals_3_to_10_l4124_412449

theorem three_cubes_of_27_equals_3_to_10 : ∃ x : ℕ, 27^3 + 27^3 + 27^3 = 3^x ∧ x = 10 := by
  sorry

end three_cubes_of_27_equals_3_to_10_l4124_412449


namespace eggs_leftover_l4124_412455

def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + emma_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end eggs_leftover_l4124_412455


namespace solution_set_of_inequality_l4124_412413

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) / (2 * x + 1) ≤ 0 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 1 ∪ {1} :=
sorry

end solution_set_of_inequality_l4124_412413


namespace remainder_problem_l4124_412485

theorem remainder_problem (a b : ℤ) 
  (ha : a % 98 = 92) 
  (hb : b % 147 = 135) : 
  (3 * a + b) % 49 = 19 := by
  sorry

end remainder_problem_l4124_412485


namespace certain_number_problem_l4124_412486

theorem certain_number_problem (x N : ℝ) :
  625^(-x) + N^(-2*x) + 5^(-4*x) = 11 →
  x = 0.25 →
  N = 25/2809 := by
sorry

end certain_number_problem_l4124_412486


namespace system_one_solution_system_two_solution_l4124_412439

-- System 1
theorem system_one_solution (x y : ℚ) : 
  2 * x - y = 5 ∧ x - 1 = (2 * y - 1) / 2 → x = 9/2 ∧ y = 4 := by sorry

-- System 2
theorem system_two_solution (x y : ℚ) :
  3 * x + 2 * y = 1 ∧ 2 * x - 3 * y = 5 → x = 1 ∧ y = -1 := by sorry

end system_one_solution_system_two_solution_l4124_412439
