import Mathlib

namespace NUMINAMATH_CALUDE_light_travel_distance_l1489_148967

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- The expected distance light travels in 50 years -/
def expected_distance : ℝ := 473.04 * (10 ^ 12)

/-- Theorem stating that the distance light travels in 50 years is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * (years : ℝ) = expected_distance := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l1489_148967


namespace NUMINAMATH_CALUDE_fraction_reduction_l1489_148931

theorem fraction_reduction (b y : ℝ) (h : 4 * b^2 + y^4 ≠ 0) :
  ((Real.sqrt (4 * b^2 + y^4) - (y^4 - 4 * b^2) / Real.sqrt (4 * b^2 + y^4)) / (4 * b^2 + y^4)) ^ (2/3) = 
  (8 * b^2) / (4 * b^2 + y^4) :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l1489_148931


namespace NUMINAMATH_CALUDE_simplify_expression_l1489_148959

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ -1) :
  ((a^2 + 1) / a - 2) / ((a^2 - 1) / (a^2 + a)) = a - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1489_148959


namespace NUMINAMATH_CALUDE_partner_investment_duration_l1489_148917

/-- Given two partners P and Q with investments and profits, calculate Q's investment duration -/
theorem partner_investment_duration
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (p_duration : ℕ)
  (h_investment : investment_ratio_p = 7 ∧ investment_ratio_q = 5)
  (h_profit : profit_ratio_p = 7 ∧ profit_ratio_q = 14)
  (h_p_duration : p_duration = 5) :
  ∃ q_duration : ℕ,
    q_duration = 14 ∧
    (investment_ratio_p * p_duration) / (investment_ratio_q * q_duration) =
    profit_ratio_p / profit_ratio_q :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_duration_l1489_148917


namespace NUMINAMATH_CALUDE_jasons_grade_difference_l1489_148922

/-- Given the grades of Jenny and Bob, and the relationship between Bob's and Jason's grades,
    prove that Jason's grade is 25 points less than Jenny's. -/
theorem jasons_grade_difference (jenny_grade : ℕ) (bob_grade : ℕ) :
  jenny_grade = 95 →
  bob_grade = 35 →
  bob_grade * 2 = jenny_grade - 25 :=
by sorry

end NUMINAMATH_CALUDE_jasons_grade_difference_l1489_148922


namespace NUMINAMATH_CALUDE_range_of_absolute_linear_function_l1489_148974

theorem range_of_absolute_linear_function 
  (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  let f : ℝ → ℝ := fun x ↦ |a * x + b|
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ max (|b|) (|a + b|)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = 0) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = max (|b|) (|a + b|)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_absolute_linear_function_l1489_148974


namespace NUMINAMATH_CALUDE_andrews_numbers_l1489_148927

theorem andrews_numbers (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end NUMINAMATH_CALUDE_andrews_numbers_l1489_148927


namespace NUMINAMATH_CALUDE_stone_fall_time_exists_stone_fall_time_approx_l1489_148920

theorem stone_fall_time_exists : ∃ s : ℝ, s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0 := by
  sorry

theorem stone_fall_time_approx (s : ℝ) (hs : s > 0 ∧ -4.5 * s^2 - 12 * s + 48 = 0) : 
  ∃ ε > 0, |s - 3.82| < ε := by
  sorry

end NUMINAMATH_CALUDE_stone_fall_time_exists_stone_fall_time_approx_l1489_148920


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1489_148998

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for f to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  is_increasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l1489_148998


namespace NUMINAMATH_CALUDE_goats_bought_l1489_148969

theorem goats_bought (total_cost : ℕ) (cow_price goat_price : ℕ) (num_cows : ℕ) :
  total_cost = 1400 →
  cow_price = 460 →
  goat_price = 60 →
  num_cows = 2 →
  ∃ (num_goats : ℕ), num_goats = 8 ∧ total_cost = num_cows * cow_price + num_goats * goat_price :=
by sorry

end NUMINAMATH_CALUDE_goats_bought_l1489_148969


namespace NUMINAMATH_CALUDE_jeans_discount_impossibility_total_price_calculation_l1489_148932

/-- Represents the prices and discount rates for jeans --/
structure JeansSale where
  fox_price : ℝ
  pony_price : ℝ
  fox_quantity : ℕ
  pony_quantity : ℕ
  total_discount_rate : ℝ
  pony_discount_rate : ℝ

/-- Theorem stating the impossibility of the given discount rates --/
theorem jeans_discount_impossibility (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2)
  (h5 : sale.total_discount_rate = 0.18)
  (h6 : sale.pony_discount_rate = 0.5667) :
  False := by
  sorry

/-- Function to calculate the total regular price --/
def total_regular_price (sale : JeansSale) : ℝ :=
  sale.fox_price * sale.fox_quantity + sale.pony_price * sale.pony_quantity

/-- Theorem stating the total regular price for the given quantities --/
theorem total_price_calculation (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2) :
  total_regular_price sale = 81 := by
  sorry

end NUMINAMATH_CALUDE_jeans_discount_impossibility_total_price_calculation_l1489_148932


namespace NUMINAMATH_CALUDE_rogue_trader_goods_value_l1489_148996

def base7ToBase10 (n : ℕ) : ℕ := sorry

def spiceValue : ℕ := 5213
def metalValue : ℕ := 1653
def fruitValue : ℕ := 202

theorem rogue_trader_goods_value :
  base7ToBase10 spiceValue + base7ToBase10 metalValue + base7ToBase10 fruitValue = 2598 := by
  sorry

end NUMINAMATH_CALUDE_rogue_trader_goods_value_l1489_148996


namespace NUMINAMATH_CALUDE_proposition_counterexample_l1489_148924

theorem proposition_counterexample : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_counterexample_l1489_148924


namespace NUMINAMATH_CALUDE_bookstore_profit_rate_l1489_148984

/-- Calculates the overall rate of profit for three books given their cost and selling prices -/
theorem bookstore_profit_rate 
  (cost_A selling_A cost_B selling_B cost_C selling_C : ℚ) 
  (h1 : cost_A = 50) (h2 : selling_A = 70)
  (h3 : cost_B = 80) (h4 : selling_B = 100)
  (h5 : cost_C = 150) (h6 : selling_C = 180) :
  (selling_A - cost_A + selling_B - cost_B + selling_C - cost_C) / 
  (cost_A + cost_B + cost_C) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_profit_rate_l1489_148984


namespace NUMINAMATH_CALUDE_problem_statement_l1489_148953

theorem problem_statement :
  (¬(∀ x : ℝ, x > 0 → Real.log x ≥ 0)) ∧ (∃ x₀ : ℝ, Real.sin x₀ = Real.cos x₀) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1489_148953


namespace NUMINAMATH_CALUDE_line_passes_through_point_l1489_148943

/-- The line equation mx - y + 1 - m = 0 passes through the point (1,1) for all real m -/
theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l1489_148943


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1489_148900

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y) / Real.sqrt ((x^2 + x*z + z^2) * (y^2 + y*z + z^2)) +
  (y * z) / Real.sqrt ((y^2 + y*x + x^2) * (z^2 + z*x + x^2)) +
  (z * x) / Real.sqrt ((z^2 + z*y + y^2) * (x^2 + x*y + y^2)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1489_148900


namespace NUMINAMATH_CALUDE_eight_S_three_l1489_148958

-- Define the operation §
def S (a b : ℤ) : ℤ := 4*a + 7*b

-- Theorem to prove
theorem eight_S_three : S 8 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_eight_S_three_l1489_148958


namespace NUMINAMATH_CALUDE_chicken_buying_equation_l1489_148934

/-- Represents the scenario of a group buying chickens -/
structure ChickenBuying where
  people : ℕ
  cost : ℕ

/-- The excess when each person contributes 9 coins -/
def excess (cb : ChickenBuying) : ℤ :=
  9 * cb.people - cb.cost

/-- The shortage when each person contributes 6 coins -/
def shortage (cb : ChickenBuying) : ℤ :=
  cb.cost - 6 * cb.people

/-- The theorem representing the chicken buying scenario -/
theorem chicken_buying_equation (cb : ChickenBuying) 
  (h1 : excess cb = 11) 
  (h2 : shortage cb = 16) : 
  9 * cb.people - 11 = 6 * cb.people + 16 := by
  sorry

end NUMINAMATH_CALUDE_chicken_buying_equation_l1489_148934


namespace NUMINAMATH_CALUDE_fraction_sum_l1489_148919

theorem fraction_sum : (3 : ℚ) / 8 + (9 : ℚ) / 14 = (33 : ℚ) / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1489_148919


namespace NUMINAMATH_CALUDE_rainfall_solution_l1489_148935

def rainfall_problem (day1 day2 day3 : ℝ) : Prop :=
  day1 = 4 ∧
  day2 = 5 * day1 ∧
  day3 = day1 + day2 - 6

theorem rainfall_solution :
  ∀ day1 day2 day3 : ℝ,
  rainfall_problem day1 day2 day3 →
  day3 = 18 := by
sorry

end NUMINAMATH_CALUDE_rainfall_solution_l1489_148935


namespace NUMINAMATH_CALUDE_insulated_cups_problem_l1489_148914

-- Define the cost prices and quantities
def cost_A : ℝ := 110
def cost_B : ℝ := 88
def quantity_A : ℕ := 30
def quantity_B : ℕ := 50

-- Define the selling prices
def sell_A : ℝ := 160
def sell_B : ℝ := 140

-- Define the total number of cups and profit
def total_cups : ℕ := 80
def total_profit : ℝ := 4100

-- Theorem statement
theorem insulated_cups_problem :
  -- Condition 1: 4 type A cups cost the same as 5 type B cups
  4 * cost_A = 5 * cost_B ∧
  -- Condition 2: 3 type A cups cost $154 more than 2 type B cups
  3 * cost_A = 2 * cost_B + 154 ∧
  -- Condition 3: Total cups purchased is 80
  quantity_A + quantity_B = total_cups ∧
  -- Condition 4: Profit calculation
  (sell_A - cost_A) * quantity_A + (sell_B - cost_B) * quantity_B = total_profit :=
by
  sorry


end NUMINAMATH_CALUDE_insulated_cups_problem_l1489_148914


namespace NUMINAMATH_CALUDE_intersection_M_N_l1489_148949

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (p.1 - 1)}

noncomputable def N : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}

theorem intersection_M_N :
  ∃! a : ℝ, a > 1 ∧ Real.sqrt (a - 1) = Real.log a ∧
  M ∩ N = {(a, Real.log a)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1489_148949


namespace NUMINAMATH_CALUDE_monthly_salary_is_7600_l1489_148942

/-- Represents the monthly salary allocation problem --/
def SalaryAllocation (x : ℝ) : Prop :=
  let bank := x / 2
  let remaining := x / 2
  let mortgage := remaining / 2 - 300
  let meals := (remaining - mortgage) / 2 + 300
  let leftover := remaining - mortgage - meals
  (bank = x / 2) ∧
  (mortgage = remaining / 2 - 300) ∧
  (meals = (remaining - mortgage) / 2 + 300) ∧
  (leftover = 800)

/-- Theorem stating that the monthly salary satisfying the given conditions is 7600 --/
theorem monthly_salary_is_7600 :
  ∃ x : ℝ, SalaryAllocation x ∧ x = 7600 :=
sorry

end NUMINAMATH_CALUDE_monthly_salary_is_7600_l1489_148942


namespace NUMINAMATH_CALUDE_janabel_widget_sales_l1489_148939

theorem janabel_widget_sales (n : ℕ) (a₁ : ℕ) (d : ℕ) : 
  n = 15 → a₁ = 2 → d = 2 → (n * (2 * a₁ + (n - 1) * d)) / 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_janabel_widget_sales_l1489_148939


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_properties_l1489_148903

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral where
  R : ℝ  -- circumradius
  a : ℝ  -- side length
  b : ℝ  -- side length
  c : ℝ  -- side length
  d : ℝ  -- side length
  S : ℝ  -- area
  positive_R : R > 0
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  positive_S : S > 0

-- Define what it means for a cyclic quadrilateral to be a square
def is_square (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_properties (q : CyclicQuadrilateral) :
  (16 * q.R^2 * q.S^2 = (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)) ∧
  (q.R * q.S * Real.sqrt 2 ≥ (q.a * q.b * q.c * q.d)^(3/4)) ∧
  (q.R * q.S * Real.sqrt 2 = (q.a * q.b * q.c * q.d)^(3/4) ↔ is_square q) :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_properties_l1489_148903


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1489_148910

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) : 
  x^2 + 10*x = 34 → 
  x = Real.sqrt a - b → 
  a + b = 64 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1489_148910


namespace NUMINAMATH_CALUDE_luke_final_sticker_count_l1489_148941

/-- Calculates the final number of stickers Luke has after various transactions -/
def final_sticker_count (initial : ℕ) (bought : ℕ) (from_friend : ℕ) (birthday : ℕ) 
                        (traded_out : ℕ) (traded_in : ℕ) (to_sister : ℕ) 
                        (for_card : ℕ) (to_charity : ℕ) : ℕ :=
  initial + bought + from_friend + birthday - traded_out + traded_in - to_sister - for_card - to_charity

/-- Theorem stating that Luke ends up with 67 stickers -/
theorem luke_final_sticker_count :
  final_sticker_count 20 12 25 30 10 15 5 8 12 = 67 := by
  sorry

end NUMINAMATH_CALUDE_luke_final_sticker_count_l1489_148941


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1489_148990

/-- A function f(x) = ax^2 + b satisfies f(xy) + f(x + y) ≥ f(x)f(y) for all real x and y
    if and only if 0 < a < 1, 0 < b ≤ 1, and 2a + b - 2 ≤ 0 -/
theorem function_inequality_condition (a b : ℝ) :
  (∀ x y : ℝ, a * (x * y)^2 + b + a * (x + y)^2 + b ≥ (a * x^2 + b) * (a * y^2 + b)) ↔
  (0 < a ∧ a < 1 ∧ 0 < b ∧ b ≤ 1 ∧ 2 * a + b - 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1489_148990


namespace NUMINAMATH_CALUDE_bird_increase_l1489_148977

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds : ℕ := 1300

/-- The number of fish-eater birds Cohen saw on the first day -/
def first_day_birds : ℕ := 300

/-- The decrease in the number of birds from the first day to the third day -/
def third_day_decrease : ℕ := 200

/-- Theorem stating the increase in the number of birds from the first to the second day -/
theorem bird_increase : 
  ∃ (second_day_birds third_day_birds : ℕ), 
    first_day_birds + second_day_birds + third_day_birds = total_birds ∧
    third_day_birds = first_day_birds - third_day_decrease ∧
    second_day_birds = first_day_birds + 600 :=
by sorry

end NUMINAMATH_CALUDE_bird_increase_l1489_148977


namespace NUMINAMATH_CALUDE_power_division_addition_l1489_148975

theorem power_division_addition (a : ℝ) : a^4 / a^2 + a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_addition_l1489_148975


namespace NUMINAMATH_CALUDE_visitation_problem_l1489_148929

/-- Represents the visitation schedule of a friend --/
structure VisitSchedule where
  period : ℕ+

/-- Calculates the number of days in a given period when exactly two friends visit --/
def exactlyTwoVisits (alice beatrix claire : VisitSchedule) (totalDays : ℕ) : ℕ :=
  sorry

/-- Theorem statement for the visitation problem --/
theorem visitation_problem :
  let alice : VisitSchedule := ⟨1⟩
  let beatrix : VisitSchedule := ⟨5⟩
  let claire : VisitSchedule := ⟨7⟩
  let totalDays : ℕ := 180
  exactlyTwoVisits alice beatrix claire totalDays = 51 := by sorry

end NUMINAMATH_CALUDE_visitation_problem_l1489_148929


namespace NUMINAMATH_CALUDE_train_crossing_time_l1489_148978

/-- Proves that a train 75 meters long, traveling at 54 km/hr, will take 5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 75 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 5 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1489_148978


namespace NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1489_148988

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_quadratic_equation_l1489_148988


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1489_148997

/-- The area of a rectangular field with one side 16 m and a diagonal of 17 m is 16 * √33 square meters. -/
theorem rectangular_field_area (a b : ℝ) (h1 : a = 16) (h2 : a^2 + b^2 = 17^2) :
  a * b = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1489_148997


namespace NUMINAMATH_CALUDE_victors_total_money_l1489_148907

/-- Victor's initial money in dollars -/
def initial_money : ℕ := 10

/-- Victor's allowance in dollars -/
def allowance : ℕ := 8

/-- Theorem: Victor's total money is $18 -/
theorem victors_total_money : initial_money + allowance = 18 := by
  sorry

end NUMINAMATH_CALUDE_victors_total_money_l1489_148907


namespace NUMINAMATH_CALUDE_triangle_equality_l1489_148982

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l1489_148982


namespace NUMINAMATH_CALUDE_campaign_fliers_l1489_148950

theorem campaign_fliers (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 600 → initial_fliers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_campaign_fliers_l1489_148950


namespace NUMINAMATH_CALUDE_cottage_rent_division_l1489_148973

/-- The total rent for the cottage -/
def total_rent : ℤ := 300

/-- The amount paid by the first friend -/
def first_friend_payment (f2 f3 f4 : ℤ) : ℤ := (f2 + f3 + f4) / 2

/-- The amount paid by the second friend -/
def second_friend_payment (f1 f3 f4 : ℤ) : ℤ := (f1 + f3 + f4) / 3

/-- The amount paid by the third friend -/
def third_friend_payment (f1 f2 f4 : ℤ) : ℤ := (f1 + f2 + f4) / 4

/-- The amount paid by the fourth friend -/
def fourth_friend_payment (f1 f2 f3 : ℤ) : ℤ := total_rent - (f1 + f2 + f3)

theorem cottage_rent_division :
  ∃ (f1 f2 f3 f4 : ℤ),
    f1 = first_friend_payment f2 f3 f4 ∧
    f2 = second_friend_payment f1 f3 f4 ∧
    f3 = third_friend_payment f1 f2 f4 ∧
    f4 = fourth_friend_payment f1 f2 f3 ∧
    f1 + f2 + f3 + f4 = total_rent ∧
    f4 = 65 :=
by sorry

end NUMINAMATH_CALUDE_cottage_rent_division_l1489_148973


namespace NUMINAMATH_CALUDE_weight_difference_l1489_148951

/-- Proves that Heather is 53.4 pounds lighter than Emily, Elizabeth, and George combined -/
theorem weight_difference (heather emily elizabeth george : ℝ) 
  (h1 : heather = 87.5)
  (h2 : emily = 45.3)
  (h3 : elizabeth = 38.7)
  (h4 : george = 56.9) :
  heather - (emily + elizabeth + george) = -53.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l1489_148951


namespace NUMINAMATH_CALUDE_smith_children_age_l1489_148968

theorem smith_children_age (age1 age2 age3 age4 : ℕ) 
  (h1 : age1 = 5)
  (h2 : age2 = 7)
  (h3 : age3 = 10)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = 8) :
  age4 = 10 := by
sorry

end NUMINAMATH_CALUDE_smith_children_age_l1489_148968


namespace NUMINAMATH_CALUDE_printer_price_ratio_printer_price_ratio_proof_l1489_148904

/-- The ratio of the printer price to the total price of enhanced computer and printer -/
theorem printer_price_ratio : ℚ :=
let basic_computer_price : ℕ := 2000
let basic_total_price : ℕ := 2500
let price_difference : ℕ := 500
let printer_price : ℕ := basic_total_price - basic_computer_price
let enhanced_computer_price : ℕ := basic_computer_price + price_difference
let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
1 / 6

theorem printer_price_ratio_proof :
  let basic_computer_price : ℕ := 2000
  let basic_total_price : ℕ := 2500
  let price_difference : ℕ := 500
  let printer_price : ℕ := basic_total_price - basic_computer_price
  let enhanced_computer_price : ℕ := basic_computer_price + price_difference
  let enhanced_total_price : ℕ := enhanced_computer_price + printer_price
  (printer_price : ℚ) / enhanced_total_price = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_printer_price_ratio_printer_price_ratio_proof_l1489_148904


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l1489_148911

theorem intersection_point_coordinates
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let line1 := {(x, y) : ℝ × ℝ | a * x + b * y = c}
  let line2 := {(x, y) : ℝ × ℝ | b * x + c * y = a}
  let line3 := {(x, y) : ℝ × ℝ | y = 2 * x}
  (∀ (p q : ℝ × ℝ), p ∈ line1 ∧ q ∈ line2 → (p.1 - q.1) * (p.2 - q.2) = -1) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3) →
  (∃ (P : ℝ × ℝ), P ∈ line1 ∧ P ∈ line2 ∧ P ∈ line3 ∧ P = (-3/5, -6/5)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l1489_148911


namespace NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt_two_l1489_148915

theorem modulus_of_z_equals_sqrt_two :
  let z : ℂ := (Complex.I + 1) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_equals_sqrt_two_l1489_148915


namespace NUMINAMATH_CALUDE_solve_equation_l1489_148926

theorem solve_equation (x : ℚ) (h : (1/3 - 1/4) * 2 = 1/x) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1489_148926


namespace NUMINAMATH_CALUDE_plane_speed_theorem_l1489_148979

/-- Given a plane's speed against a tailwind and the tailwind speed, 
    calculate the plane's speed with the tailwind. -/
def plane_speed_with_tailwind (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) : ℝ :=
  speed_against_tailwind + 2 * tailwind_speed

/-- Theorem: The plane's speed with the tailwind is 460 mph given the conditions. -/
theorem plane_speed_theorem (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) 
  (h1 : speed_against_tailwind = 310)
  (h2 : tailwind_speed = 75) :
  plane_speed_with_tailwind speed_against_tailwind tailwind_speed = 460 := by
  sorry

#eval plane_speed_with_tailwind 310 75

end NUMINAMATH_CALUDE_plane_speed_theorem_l1489_148979


namespace NUMINAMATH_CALUDE_marks_score_ratio_l1489_148909

theorem marks_score_ratio (highest_score range marks_score : ℕ) : 
  highest_score = 98 →
  range = 75 →
  marks_score = 46 →
  marks_score % (highest_score - range) = 0 →
  marks_score / (highest_score - range) = 2 :=
by sorry

end NUMINAMATH_CALUDE_marks_score_ratio_l1489_148909


namespace NUMINAMATH_CALUDE_equidistant_points_l1489_148991

def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_l1489_148991


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l1489_148981

def f (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 4

def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    g (r₁^3) b c d = 0 ∧ g (r₂^3) b c d = 0 ∧ g (r₃^3) b c d = 0) →
  b = 24 ∧ c = 32 ∧ d = 64 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l1489_148981


namespace NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1489_148937

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 4) (Fin 4) ℤ := !![3, 0, 2, 0;
                                       2, 3, -1, 4;
                                       0, 4, -2, 3;
                                       5, 2, 0, 1]
  Matrix.det A = -84 := by sorry

end NUMINAMATH_CALUDE_determinant_of_specific_matrix_l1489_148937


namespace NUMINAMATH_CALUDE_new_tires_cost_calculation_l1489_148930

/-- The amount spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The total amount spent on car parts -/
def total_car_parts_cost : ℝ := 224.87

/-- The amount spent on new tires -/
def new_tires_cost : ℝ := total_car_parts_cost - speakers_cost

theorem new_tires_cost_calculation : 
  new_tires_cost = 106.33 := by sorry

end NUMINAMATH_CALUDE_new_tires_cost_calculation_l1489_148930


namespace NUMINAMATH_CALUDE_wang_elevator_problem_l1489_148963

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_problem :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 33.6) := by
  sorry

end NUMINAMATH_CALUDE_wang_elevator_problem_l1489_148963


namespace NUMINAMATH_CALUDE_total_wrapping_paper_l1489_148957

/-- The amount of wrapping paper needed for three presents -/
def wrapping_paper (first_present second_present third_present : ℝ) : ℝ :=
  first_present + second_present + third_present

/-- Theorem: The total amount of wrapping paper needed is 7 square feet -/
theorem total_wrapping_paper :
  let first_present := 2
  let second_present := 3/4 * first_present
  let third_present := first_present + second_present
  wrapping_paper first_present second_present third_present = 7 :=
by sorry

end NUMINAMATH_CALUDE_total_wrapping_paper_l1489_148957


namespace NUMINAMATH_CALUDE_f_2019_eq_zero_l1489_148965

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_3 (f : ℝ → ℝ) : Prop := ∀ x, f (3 - x) = f x

theorem f_2019_eq_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_3 f) : 
  f 2019 = 0 := by sorry

end NUMINAMATH_CALUDE_f_2019_eq_zero_l1489_148965


namespace NUMINAMATH_CALUDE_eight_digit_divisibility_l1489_148994

-- Define a four-digit number
def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

-- Define the eight-digit number formed by repeating the four-digit number
def eight_digit_number (a b c d : ℕ) : ℕ := four_digit_number a b c d * 10000 + four_digit_number a b c d

-- Theorem statement
theorem eight_digit_divisibility (a b c d : ℕ) :
  (a < 10) → (b < 10) → (c < 10) → (d < 10) →
  (∃ k₁ k₂ : ℕ, eight_digit_number a b c d = 73 * k₁ ∧ eight_digit_number a b c d = 137 * k₂) := by
  sorry


end NUMINAMATH_CALUDE_eight_digit_divisibility_l1489_148994


namespace NUMINAMATH_CALUDE_geometric_condition_implies_a_equals_two_l1489_148946

/-- The value of a for which the given geometric conditions are satisfied -/
def geometric_a : ℝ := 2

/-- The line equation y = 2x + 2 -/
def line (x : ℝ) : ℝ := 2 * x + 2

/-- The parabola equation y = ax^2 -/
def parabola (a x : ℝ) : ℝ := a * x^2

/-- Theorem stating that under the given geometric conditions, a = 2 -/
theorem geometric_condition_implies_a_equals_two (a : ℝ) 
  (h_pos : a > 0)
  (h_intersect : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ line x₁ = parabola a x₁ ∧ line x₂ = parabola a x₂)
  (h_midpoint : ∃ (x_mid : ℝ), x_mid = (x₁ + x₂) / 2 ∧ 
    parabola a x_mid = a * x_mid^2 ∧ 
    ∀ (y : ℝ), y ≠ a * x_mid^2 → |x_mid - x₁| + |y - line x₁| = |x_mid - x₂| + |y - line x₂|)
  (h_vector_condition : ∀ (A P Q : ℝ × ℝ), 
    P.1 ≠ Q.1 → 
    line P.1 = P.2 → line Q.1 = Q.2 → 
    parabola a A.1 = A.2 → 
    |(A.1 - P.1, A.2 - P.2)| + |(A.1 - Q.1, A.2 - Q.2)| = 
    |(A.1 - P.1, A.2 - P.2)| - |(A.1 - Q.1, A.2 - Q.2)|)
  : a = geometric_a := by sorry

end NUMINAMATH_CALUDE_geometric_condition_implies_a_equals_two_l1489_148946


namespace NUMINAMATH_CALUDE_remainder_sum_factorials_25_l1489_148954

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem remainder_sum_factorials_25 :
  (sum_factorials 50) % 25 = (sum_factorials 4) % 25 :=
by sorry

end NUMINAMATH_CALUDE_remainder_sum_factorials_25_l1489_148954


namespace NUMINAMATH_CALUDE_equation_solution_l1489_148923

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x - 3) = 3 / (x + 1)) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1489_148923


namespace NUMINAMATH_CALUDE_y_at_40_l1489_148970

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The line passing through the given points -/
def exampleLine : Line :=
  { point1 := (2, 5)
  , point2 := (6, 17)
  , point3 := (10, 29) }

/-- Function to calculate y-coordinate for a given x-coordinate on the line -/
def yCoordinate (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem y_at_40 (l : Line) : l = exampleLine → yCoordinate l 40 = 119 := by
  sorry

end NUMINAMATH_CALUDE_y_at_40_l1489_148970


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l1489_148901

theorem polynomial_sum_simplification :
  let p₁ : Polynomial ℚ := 2 * X^5 - 3 * X^3 + 5 * X^2 - 4 * X + 6
  let p₂ : Polynomial ℚ := -X^5 + 4 * X^4 - 2 * X^3 - X^2 + 3 * X - 8
  p₁ + p₂ = X^5 + 4 * X^4 - 5 * X^3 + 4 * X^2 - X - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l1489_148901


namespace NUMINAMATH_CALUDE_expression_value_l1489_148947

theorem expression_value (x y : ℝ) (h : (x - y) / (x + y) = 3) :
  2 * (x - y) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1489_148947


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1489_148928

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x + 2)*(4 - x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for the range of a when B ∪ C = B
theorem range_of_a (a : ℝ) (h : B ∪ C a = B) : -2 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1489_148928


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1489_148952

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (Real.sqrt x)
  ∀ x : ℝ, (x > 0 ∧ f x = 16 / (9 - f x)) ↔ (x = 4096 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1489_148952


namespace NUMINAMATH_CALUDE_fraction_inequality_l1489_148925

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1489_148925


namespace NUMINAMATH_CALUDE_lindas_savings_l1489_148960

theorem lindas_savings (savings : ℝ) : 
  savings > 0 →
  (0.9 * (3/8) * savings) + (0.85 * (1/4) * savings) + 450 = savings →
  savings = 1000 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l1489_148960


namespace NUMINAMATH_CALUDE_smallest_non_existent_count_l1489_148933

/-- The number of terms in the arithmetic progression -/
def progression_length : ℕ := 1999

/-- 
  Counts the number of integer terms in an arithmetic progression 
  with 'progression_length' terms and common difference 1/m
-/
def count_integer_terms (m : ℕ) : ℕ :=
  1 + (progression_length - 1) / m

/-- 
  Checks if there exists an arithmetic progression of 'progression_length' 
  real numbers containing exactly n integers
-/
def exists_progression_with_n_integers (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ count_integer_terms m = n

theorem smallest_non_existent_count : 
  (∀ k < 70, exists_progression_with_n_integers k) ∧
  ¬exists_progression_with_n_integers 70 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_existent_count_l1489_148933


namespace NUMINAMATH_CALUDE_dodecagon_triangles_l1489_148983

/-- A regular dodecagon is a 12-sided polygon. -/
def regular_dodecagon : ℕ := 12

/-- The number of triangles that can be formed using the vertices of a regular dodecagon. -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem dodecagon_triangles :
  num_triangles regular_dodecagon = 220 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_triangles_l1489_148983


namespace NUMINAMATH_CALUDE_simplify_expression_l1489_148956

theorem simplify_expression (x : ℝ) : 120 * x - 75 * x = 45 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1489_148956


namespace NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_abs_l1489_148971

theorem cube_root_minus_square_root_plus_abs : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) - Real.sqrt ((-3 : ℝ)^2) + |Real.sqrt 2 - 1| = Real.sqrt 2 - 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_minus_square_root_plus_abs_l1489_148971


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_approx_l1489_148962

theorem one_third_minus_decimal_approx : 
  (1 : ℚ) / 3 - 33333333 / 100000000 = 1 / (3 * 100000000) := by sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_approx_l1489_148962


namespace NUMINAMATH_CALUDE_crossout_theorem_l1489_148944

/-- The process of crossing out numbers and writing sums -/
def crossOutProcess (n : ℕ) : ℕ → ℕ
| 0 => n
| (m + 1) => let prev := crossOutProcess n m
             if prev > 4 then prev - 3 else prev

/-- The condition for n to be reduced to one number -/
def reducesToOne (n : ℕ) : Prop :=
  ∃ k, crossOutProcess n k = 1

/-- The sum of all numbers written during the process -/
def totalSum (n : ℕ) : ℕ :=
  sorry  -- Definition of totalSum would go here

/-- Main theorem combining both parts of the problem -/
theorem crossout_theorem :
  (∀ n : ℕ, reducesToOne n ↔ n % 3 = 1) ∧
  totalSum 2002 = 12881478 :=
sorry

end NUMINAMATH_CALUDE_crossout_theorem_l1489_148944


namespace NUMINAMATH_CALUDE_complex_fraction_real_minus_imag_l1489_148938

theorem complex_fraction_real_minus_imag (z : ℂ) (a b : ℝ) : 
  z = 5 / (-3 - Complex.I) → 
  a = z.re → 
  b = z.im → 
  a - b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_minus_imag_l1489_148938


namespace NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1489_148948

/-- The area of the largest equilateral triangle inscribed in a circle with radius 10 cm,
    where one side of the triangle is a diameter of the circle. -/
theorem largest_inscribed_equilateral_triangle_area :
  let r : ℝ := 10  -- radius of the circle in cm
  let d : ℝ := 2 * r  -- diameter of the circle in cm
  let h : ℝ := r * Real.sqrt 3  -- height of the equilateral triangle
  let area : ℝ := (1 / 2) * d * h  -- area of the triangle
  area = 100 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_equilateral_triangle_area_l1489_148948


namespace NUMINAMATH_CALUDE_article_price_before_discount_l1489_148993

/-- 
Given an article whose price after a 24% decrease is 988 rupees, 
prove that its original price was 1300 rupees.
-/
theorem article_price_before_discount (price_after_discount : ℝ) 
  (h1 : price_after_discount = 988) 
  (h2 : price_after_discount = 0.76 * (original_price : ℝ)) : 
  original_price = 1300 := by
  sorry

end NUMINAMATH_CALUDE_article_price_before_discount_l1489_148993


namespace NUMINAMATH_CALUDE_news_watching_probability_l1489_148905

/-- Represents a survey conducted in a town -/
structure TownSurvey where
  total_population : ℕ
  sample_size : ℕ
  news_watchers : ℕ

/-- Calculates the probability of a random person watching the news based on survey results -/
def probability_watch_news (survey : TownSurvey) : ℚ :=
  survey.news_watchers / survey.sample_size

/-- Theorem stating the probability of watching news for the given survey -/
theorem news_watching_probability (survey : TownSurvey) 
  (h1 : survey.total_population = 100000)
  (h2 : survey.sample_size = 2000)
  (h3 : survey.news_watchers = 250) :
  probability_watch_news survey = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_news_watching_probability_l1489_148905


namespace NUMINAMATH_CALUDE_cosine_value_l1489_148989

theorem cosine_value (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin (α + π / 6) = 3 / 5) : 
  Real.cos (α - π / 6) = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_l1489_148989


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1489_148980

/-- Given a principal P at simple interest for 6 years, if increasing the
    interest rate by 4% results in $144 more interest, then P = $600. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 4) * 6) / 100 - (P * R * 6) / 100 = 144 → P = 600 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l1489_148980


namespace NUMINAMATH_CALUDE_odd_integers_equality_l1489_148916

theorem odd_integers_equality (a b : ℕ) (ha : Odd a) (hb : Odd b) (hpos_a : 0 < a) (hpos_b : 0 < b)
  (h_div : (2 * a * b + 1) ∣ (a^2 + b^2 + 1)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_equality_l1489_148916


namespace NUMINAMATH_CALUDE_sequence_conjecture_l1489_148966

theorem sequence_conjecture (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, a (n + 1) - a n > 0) ∧
  (∀ n : ℕ, (a (n + 1) - a n)^2 - 2 * (a (n + 1) + a n) + 1 = 0) →
  ∀ n : ℕ, a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_conjecture_l1489_148966


namespace NUMINAMATH_CALUDE_limit_to_e_l1489_148986

theorem limit_to_e (x : ℕ → ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| > 1/ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + 1 / x n) ^ (x n) - Real.exp 1| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_to_e_l1489_148986


namespace NUMINAMATH_CALUDE_swimming_pool_receipts_l1489_148921

/-- Calculates the total receipts for a public swimming pool given the number of children and adults, and their respective admission prices. -/
def total_receipts (total_people : ℕ) (children : ℕ) (child_price : ℚ) (adult_price : ℚ) : ℚ :=
  let adults := total_people - children
  let children_total := child_price * children
  let adults_total := adult_price * adults
  children_total + adults_total

/-- Proves that the total receipts for the given scenario is $1405.50 -/
theorem swimming_pool_receipts :
  total_receipts 754 388 (3/2) (9/4) = 2811/2 :=
by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_receipts_l1489_148921


namespace NUMINAMATH_CALUDE_roots_properties_l1489_148913

theorem roots_properties (x₁ x₂ : ℝ) (h : x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) : 
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l1489_148913


namespace NUMINAMATH_CALUDE_distinct_towers_count_l1489_148999

/-- Represents the number of cubes of each color -/
structure CubeCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct towers -/
def countDistinctTowers (cubes : CubeCount) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct towers of height 10 that can be built
    using 3 red cubes, 4 blue cubes, and 5 yellow cubes, with two cubes
    not being used, is equal to 6,812 -/
theorem distinct_towers_count :
  let cubes : CubeCount := { red := 3, blue := 4, yellow := 5 }
  let towerHeight : Nat := 10
  countDistinctTowers cubes towerHeight = 6812 := by
  sorry

end NUMINAMATH_CALUDE_distinct_towers_count_l1489_148999


namespace NUMINAMATH_CALUDE_additional_hovering_time_l1489_148912

/-- Represents the hovering time of a plane in different time zones over two days. -/
structure PlaneHoveringTime where
  mountain_day1 : ℕ
  central_day1 : ℕ
  eastern_day1 : ℕ
  mountain_day2 : ℕ
  central_day2 : ℕ
  eastern_day2 : ℕ

/-- Theorem stating that given the conditions of the problem, the additional hovering time
    in each time zone on the second day is 5 hours. -/
theorem additional_hovering_time
  (h : PlaneHoveringTime)
  (h_mountain_day1 : h.mountain_day1 = 3)
  (h_central_day1 : h.central_day1 = 4)
  (h_eastern_day1 : h.eastern_day1 = 2)
  (h_total_time : h.mountain_day1 + h.central_day1 + h.eastern_day1 +
                  h.mountain_day2 + h.central_day2 + h.eastern_day2 = 24)
  (h_equal_additional : h.mountain_day2 = h.central_day2 ∧ h.central_day2 = h.eastern_day2) :
  h.mountain_day2 = 5 ∧ h.central_day2 = 5 ∧ h.eastern_day2 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_hovering_time_l1489_148912


namespace NUMINAMATH_CALUDE_symmetric_function_theorem_l1489_148908

/-- A function f: ℝ → ℝ is symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem symmetric_function_theorem (f : ℝ → ℝ) 
  (h_sym : SymmetricAboutOrigin f)
  (h_nonneg : ∀ x ≥ 0, f x = x * (1 - x)) :
  ∀ x ≤ 0, f x = x * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_theorem_l1489_148908


namespace NUMINAMATH_CALUDE_adam_miles_l1489_148961

/-- Adam ran 25 miles more than Katie, and Katie ran 10 miles. -/
theorem adam_miles (katie_miles : ℕ) (adam_miles : ℕ) : 
  katie_miles = 10 → adam_miles = katie_miles + 25 → adam_miles = 35 := by
  sorry

end NUMINAMATH_CALUDE_adam_miles_l1489_148961


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l1489_148936

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 8) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 + l^2 + h^2 = d^2 ∧ w = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l1489_148936


namespace NUMINAMATH_CALUDE_room_length_is_five_l1489_148992

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5 meters. -/
theorem room_length_is_five (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 4.75 →
  total_cost = 21375 →
  rate_per_sqm = 900 →
  (total_cost / rate_per_sqm) / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_is_five_l1489_148992


namespace NUMINAMATH_CALUDE_slope_of_l₃_l1489_148995

-- Define the lines and points
def l₁ : Set (ℝ × ℝ) := {(x, y) | 5 * x - 3 * y = 2}
def l₂ : Set (ℝ × ℝ) := {(x, y) | y = 2}
def A : ℝ × ℝ := (2, -2)

-- Define the existence of point B
def B_exists : Prop := ∃ B : ℝ × ℝ, B ∈ l₁ ∧ B ∈ l₂

-- Define the existence of point C and line l₃
def C_and_l₃_exist : Prop := ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
  C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) > 0)

-- Define the area of triangle ABC
def area_ABC : ℝ := 5

-- Theorem statement
theorem slope_of_l₃ (h₁ : A ∈ l₁) (h₂ : B_exists) (h₃ : C_and_l₃_exist) (h₄ : area_ABC = 5) :
  ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
    C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
    (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 20 / 9) :=
sorry

end NUMINAMATH_CALUDE_slope_of_l₃_l1489_148995


namespace NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l1489_148976

theorem sphere_radius_from_surface_area (S : ℝ) (r : ℝ) (h : S = 4 * Real.pi) :
  S = 4 * Real.pi * r^2 → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_surface_area_l1489_148976


namespace NUMINAMATH_CALUDE_properties_of_negative_three_halves_l1489_148987

def x : ℚ := -3/2

theorem properties_of_negative_three_halves :
  (- x = 3/2) ∧ 
  (x⁻¹ = -2/3) ∧ 
  (|x| = 3/2) := by sorry

end NUMINAMATH_CALUDE_properties_of_negative_three_halves_l1489_148987


namespace NUMINAMATH_CALUDE_mason_bricks_used_l1489_148902

/-- Calculates the total number of bricks used by a mason given the following conditions:
  * The mason needs to build 6 courses per wall
  * Each course has 10 bricks
  * He needs to build 4 walls
  * He can't finish two courses of the last wall due to lack of bricks
-/
def total_bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (unfinished_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course * complete_walls
  let incomplete_wall_bricks := (courses_per_wall - unfinished_courses) * bricks_per_course
  complete_wall_bricks + incomplete_wall_bricks

theorem mason_bricks_used :
  total_bricks_used 6 10 4 2 = 220 := by
  sorry

end NUMINAMATH_CALUDE_mason_bricks_used_l1489_148902


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1489_148985

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (46 * x + 8) % 24 = 4 ∧ ∀ (y : ℕ), y > 0 ∧ (46 * y + 8) % 24 = 4 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l1489_148985


namespace NUMINAMATH_CALUDE_lattice_points_form_square_l1489_148955

-- Define a structure for a point in the plane
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a function to calculate the squared distance between two points
def squaredDistance (p q : Point) : ℤ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define a function to calculate the area of a triangle given three points
def areaOfTriangle (p q r : Point) : ℚ :=
  let a := squaredDistance p q
  let b := squaredDistance q r
  let c := squaredDistance r p
  ((a + b + c)^2 - 2 * (a^2 + b^2 + c^2)) / 16

-- Theorem statement
theorem lattice_points_form_square (p q r : Point) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_inequality : (squaredDistance p q).sqrt + (squaredDistance q r).sqrt < (8 * areaOfTriangle p q r + 1).sqrt) :
  ∃ s : Point, s ≠ p ∧ s ≠ q ∧ s ≠ r ∧ 
    squaredDistance p q = squaredDistance q r ∧
    squaredDistance r s = squaredDistance s p ∧
    squaredDistance p q = squaredDistance r s :=
sorry

end NUMINAMATH_CALUDE_lattice_points_form_square_l1489_148955


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1489_148964

theorem two_digit_number_property (a b j m : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = j * (a^2 + b^2)) (h4 : 10 * b + a = m * (a^2 + b^2)) : m = j := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1489_148964


namespace NUMINAMATH_CALUDE_trapezoid_area_l1489_148945

/-- The area of a trapezoid with height x, one base 4x, and the other base 3x is 7x²/2 -/
theorem trapezoid_area (x : ℝ) : 
  x * ((4 * x + 3 * x) / 2) = 7 * x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l1489_148945


namespace NUMINAMATH_CALUDE_f_equals_x_l1489_148972

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem f_equals_x (a b c : ℝ) :
  (∀ x, f a b c x + f a b c (-x) = 0) →  -- f is odd
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f a b c x < f a b c y) →  -- f is strictly increasing on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) →  -- conditions on a, b, c
  ∀ x ≥ 1, f a b c x ≥ 1 →  -- f(x) ≥ 1 for x ≥ 1
  (∀ x ≥ 1, f a b c (f a b c x) = x) →  -- f(f(x)) = x for x ≥ 1
  ∀ x ≥ 1, f a b c x = x :=  -- conclusion: f(x) = x for x ≥ 1
by sorry


end NUMINAMATH_CALUDE_f_equals_x_l1489_148972


namespace NUMINAMATH_CALUDE_remainder_b_96_mod_50_l1489_148940

theorem remainder_b_96_mod_50 : (7^96 + 9^96) % 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_b_96_mod_50_l1489_148940


namespace NUMINAMATH_CALUDE_calf_grazing_area_increase_calf_grazing_area_increase_value_l1489_148918

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m -/
theorem calf_grazing_area_increase : ℝ :=
  let initial_length : ℝ := 12
  let final_length : ℝ := 25
  let initial_area := Real.pi * initial_length ^ 2
  let final_area := Real.pi * final_length ^ 2
  final_area - initial_area

/-- The additional area a calf can graze when its rope is increased from 12 m to 25 m is 481π m² -/
theorem calf_grazing_area_increase_value : 
  calf_grazing_area_increase = 481 * Real.pi := by sorry

end NUMINAMATH_CALUDE_calf_grazing_area_increase_calf_grazing_area_increase_value_l1489_148918


namespace NUMINAMATH_CALUDE_mean_temperature_is_84_l1489_148906

def temperatures : List ℝ := [82, 84, 83, 85, 86]

theorem mean_temperature_is_84 :
  (temperatures.sum / temperatures.length) = 84 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_84_l1489_148906
