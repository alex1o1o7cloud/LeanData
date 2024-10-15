import Mathlib

namespace NUMINAMATH_CALUDE_angle4_is_60_l1822_182231

/-- Represents a quadrilateral with specific angle properties -/
structure SpecialQuadrilateral where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_property : angle1 + angle2 + angle3 = 360
  equal_angles : angle3 = angle4 ∧ angle3 = angle5
  angle1_value : angle1 = 100
  angle2_value : angle2 = 80

/-- Theorem: In a SpecialQuadrilateral, angle4 equals 60 degrees -/
theorem angle4_is_60 (q : SpecialQuadrilateral) : q.angle4 = 60 := by
  sorry


end NUMINAMATH_CALUDE_angle4_is_60_l1822_182231


namespace NUMINAMATH_CALUDE_children_share_sum_l1822_182228

theorem children_share_sum (total_money : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_money = 4500 → 
  ratio_a = 2 → 
  ratio_b = 4 → 
  ratio_c = 5 → 
  ratio_d = 4 → 
  (ratio_a + ratio_b) * total_money / (ratio_a + ratio_b + ratio_c + ratio_d) = 1800 := by
sorry

end NUMINAMATH_CALUDE_children_share_sum_l1822_182228


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1822_182259

theorem solution_set_quadratic_inequality (x : ℝ) :
  2 * x + 3 - x^2 > 0 ↔ -1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1822_182259


namespace NUMINAMATH_CALUDE_four_digit_divisibility_sum_l1822_182281

/-- The number of four-digit numbers divisible by 3 -/
def C : ℕ := 3000

/-- The number of four-digit multiples of 7 -/
def D : ℕ := 1286

/-- Theorem stating that the sum of four-digit numbers divisible by 3 and multiples of 7 is 4286 -/
theorem four_digit_divisibility_sum : C + D = 4286 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisibility_sum_l1822_182281


namespace NUMINAMATH_CALUDE_even_function_implies_b_zero_l1822_182236

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = x² + bx -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

theorem even_function_implies_b_zero (b : ℝ) :
  IsEven (f b) → b = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_b_zero_l1822_182236


namespace NUMINAMATH_CALUDE_cosine_equation_solutions_l1822_182254

open Real

theorem cosine_equation_solutions :
  let f := fun (x : ℝ) => 3 * (cos x)^4 - 6 * (cos x)^3 + 4 * (cos x)^2 - 1
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_cosine_equation_solutions_l1822_182254


namespace NUMINAMATH_CALUDE_range_of_a_l1822_182297

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) → 
  -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1822_182297


namespace NUMINAMATH_CALUDE_noah_sales_revenue_l1822_182245

-- Define constants
def large_painting_price : ℝ := 60
def small_painting_price : ℝ := 30
def last_month_large_sales : ℕ := 8
def last_month_small_sales : ℕ := 4
def large_painting_discount : ℝ := 0.1
def small_painting_commission : ℝ := 0.05
def sales_tax_rate : ℝ := 0.07

-- Define the theorem
theorem noah_sales_revenue :
  let this_month_large_sales := 2 * last_month_large_sales
  let this_month_small_sales := 2 * last_month_small_sales
  let discounted_large_price := large_painting_price * (1 - large_painting_discount)
  let commissioned_small_price := small_painting_price * (1 - small_painting_commission)
  let total_sales_before_tax := 
    this_month_large_sales * discounted_large_price +
    this_month_small_sales * commissioned_small_price
  let sales_tax := total_sales_before_tax * sales_tax_rate
  let total_sales_revenue := total_sales_before_tax + sales_tax
  total_sales_revenue = 1168.44 := by
  sorry

end NUMINAMATH_CALUDE_noah_sales_revenue_l1822_182245


namespace NUMINAMATH_CALUDE_cupcakes_leftover_l1822_182269

/-- Proves that given 40 cupcakes, after distributing to two classes and four individuals, 2 cupcakes remain. -/
theorem cupcakes_leftover (total : ℕ) (class1 : ℕ) (class2 : ℕ) (additional : ℕ) : 
  total = 40 → class1 = 18 → class2 = 16 → additional = 4 → 
  total - (class1 + class2 + additional) = 2 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_leftover_l1822_182269


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l1822_182232

/-- Line represented by a linear equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_iff_a_eq_neg_one :
  let l1 : Line := { a := 1, b := -1, c := -1 }
  let l2 : Line := { a := 1, b := a, c := -2 }
  parallel l1 l2 ↔ a = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l1822_182232


namespace NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l1822_182286

theorem square_plus_minus_one_divisible_by_five (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^2 + 1)) ∨ (5 ∣ (a^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l1822_182286


namespace NUMINAMATH_CALUDE_chennys_friends_l1822_182243

theorem chennys_friends (initial_candies : ℕ) (additional_candies : ℕ) (candies_per_friend : ℕ) :
  initial_candies = 10 →
  additional_candies = 4 →
  candies_per_friend = 2 →
  (initial_candies + additional_candies) / candies_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_chennys_friends_l1822_182243


namespace NUMINAMATH_CALUDE_add_2023_minutes_to_midnight_l1822_182239

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime and returns the resulting DateTime -/
def addMinutes (start : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- The starting DateTime (midnight on December 31, 2020) -/
def startTime : DateTime :=
  { day := 0, hour := 0, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : Nat := 2023

/-- The expected result DateTime (January 1 at 9:43 AM) -/
def expectedResult : DateTime :=
  { day := 1, hour := 9, minute := 43 }

/-- Theorem stating that adding 2023 minutes to midnight on December 31, 2020,
    results in January 1 at 9:43 AM -/
theorem add_2023_minutes_to_midnight :
  addMinutes startTime minutesToAdd = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_add_2023_minutes_to_midnight_l1822_182239


namespace NUMINAMATH_CALUDE_not_perfect_square_9999xxxx_l1822_182212

theorem not_perfect_square_9999xxxx : 
  ∀ n : ℕ, 99990000 ≤ n ∧ n ≤ 99999999 → ¬∃ m : ℕ, n = m * m := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_9999xxxx_l1822_182212


namespace NUMINAMATH_CALUDE_a_equals_3y_l1822_182280

theorem a_equals_3y (a b y : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * y^3) 
  (h3 : a - b = 3 * y) : 
  a = 3 * y := by
sorry

end NUMINAMATH_CALUDE_a_equals_3y_l1822_182280


namespace NUMINAMATH_CALUDE_arithmetic_equations_correctness_l1822_182287

theorem arithmetic_equations_correctness : 
  (-2 + 8 ≠ 10) ∧ 
  (-1 - 3 = -4) ∧ 
  (-2 * 2 ≠ 4) ∧ 
  (-8 / -1 ≠ -1/8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_equations_correctness_l1822_182287


namespace NUMINAMATH_CALUDE_anton_number_is_729_l1822_182262

-- Define a three-digit number type
def ThreeDigitNumber := {n : ℕ // n ≥ 100 ∧ n ≤ 999}

-- Define a function to check if two numbers match in exactly one digit place
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.val / 100 = b.val / 100 ∧ a.val % 100 ≠ b.val % 100) ∨
  (a.val % 100 / 10 = b.val % 100 / 10 ∧ a.val / 100 ≠ b.val / 100 ∧ a.val % 10 ≠ b.val % 10) ∨
  (a.val % 10 = b.val % 10 ∧ a.val / 10 ≠ b.val / 10)

theorem anton_number_is_729 (x : ThreeDigitNumber) :
  matchesOneDigit x ⟨109, by norm_num⟩ ∧
  matchesOneDigit x ⟨704, by norm_num⟩ ∧
  matchesOneDigit x ⟨124, by norm_num⟩ →
  x = ⟨729, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_anton_number_is_729_l1822_182262


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1822_182253

/-- Given a train and tunnel with specified lengths and crossing time, calculate the train's speed in km/hr -/
theorem train_speed_calculation (train_length tunnel_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 415)
  (h2 : tunnel_length = 285)
  (h3 : crossing_time = 40) :
  (train_length + tunnel_length) / crossing_time * 3.6 = 63 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1822_182253


namespace NUMINAMATH_CALUDE_f_properties_l1822_182298

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a*x + a)

theorem f_properties (a : ℝ) (h : a > 2) :
  let f' := deriv (f a)
  ∃ (S₁ S₂ S₃ : Set ℝ),
    (f' 0 = a) ∧
    (S₁ = Set.Iio 0) ∧
    (S₂ = Set.Ioi (a - 2)) ∧
    (S₃ = Set.Ioo 0 (a - 2)) ∧
    (StrictMonoOn (f a) S₁) ∧
    (StrictMonoOn (f a) S₂) ∧
    (StrictAntiOn (f a) S₃) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1822_182298


namespace NUMINAMATH_CALUDE_freshman_class_size_l1822_182223

theorem freshman_class_size (N : ℕ) 
  (h1 : N > 0) 
  (h2 : 90 ≤ N) 
  (h3 : 100 ≤ N) :
  (90 : ℝ) / N * (20 : ℝ) / 100 = (20 : ℝ) / N → N = 450 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1822_182223


namespace NUMINAMATH_CALUDE_waiter_tips_l1822_182241

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Theorem stating the total tips earned by the waiter --/
theorem waiter_tips : total_tips 7 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tips_l1822_182241


namespace NUMINAMATH_CALUDE_number_ratio_l1822_182261

/-- Given three numbers satisfying specific conditions, prove their ratio -/
theorem number_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  B = 30 →
  8 * B = 5 * C →
  A * 3 = B * 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1822_182261


namespace NUMINAMATH_CALUDE_clouddale_total_rainfall_l1822_182238

/-- Calculates the total annual rainfall given the average monthly rainfall -/
def annual_rainfall (average_monthly : ℝ) : ℝ := average_monthly * 12

/-- Represents the rainfall data for Clouddale -/
structure ClouddaleRainfall where
  avg_2003 : ℝ  -- Average monthly rainfall in 2003
  increase_rate : ℝ  -- Percentage increase in 2004

/-- Theorem stating the total rainfall for both years in Clouddale -/
theorem clouddale_total_rainfall (data : ClouddaleRainfall) 
  (h1 : data.avg_2003 = 45)
  (h2 : data.increase_rate = 0.05) : 
  (annual_rainfall data.avg_2003 = 540) ∧ 
  (annual_rainfall (data.avg_2003 * (1 + data.increase_rate)) = 567) := by
  sorry

#eval annual_rainfall 45
#eval annual_rainfall (45 * 1.05)

end NUMINAMATH_CALUDE_clouddale_total_rainfall_l1822_182238


namespace NUMINAMATH_CALUDE_mario_flower_count_l1822_182284

/-- The number of flowers on Mario's first hibiscus plant -/
def first_plant_flowers : ℕ := 2

/-- The number of flowers on Mario's second hibiscus plant -/
def second_plant_flowers : ℕ := 2 * first_plant_flowers

/-- The number of flowers on Mario's third hibiscus plant -/
def third_plant_flowers : ℕ := 4 * second_plant_flowers

/-- The total number of flowers on all of Mario's hibiscus plants -/
def total_flowers : ℕ := first_plant_flowers + second_plant_flowers + third_plant_flowers

theorem mario_flower_count : total_flowers = 22 := by
  sorry

end NUMINAMATH_CALUDE_mario_flower_count_l1822_182284


namespace NUMINAMATH_CALUDE_expand_product_l1822_182288

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1822_182288


namespace NUMINAMATH_CALUDE_divisibility_by_1947_l1822_182219

theorem divisibility_by_1947 (n : ℕ) : 
  (46 * 2^(n+1) + 296 * 13 * 2^(n+1)) % 1947 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_1947_l1822_182219


namespace NUMINAMATH_CALUDE_wages_problem_l1822_182256

/-- Given a sum of money that can pay b's wages for 28 days and both a's and b's wages for 12 days,
    prove that it can pay a's wages for 21 days. -/
theorem wages_problem (S : ℝ) (Wa Wb : ℝ) (S_pays_b_28_days : S = 28 * Wb) 
    (S_pays_both_12_days : S = 12 * (Wa + Wb)) : S = 21 * Wa := by
  sorry

end NUMINAMATH_CALUDE_wages_problem_l1822_182256


namespace NUMINAMATH_CALUDE_problem_statement_l1822_182215

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- Theorem stating the properties of p and q
theorem problem_statement :
  ¬p ∧ q ∧ (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬p ∧ ¬¬q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1822_182215


namespace NUMINAMATH_CALUDE_angle_A_value_range_of_b_squared_plus_c_squared_l1822_182213

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  (t.a + t.b) * (Real.sin t.A - Real.sin t.B) = t.c * (Real.sin t.C - Real.sin t.B)

-- Theorem 1: Prove that A = π/3
theorem angle_A_value (t : Triangle) (h : given_condition t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Prove the range of b² + c² when a = 4
theorem range_of_b_squared_plus_c_squared (t : Triangle) (h1 : given_condition t) (h2 : t.a = 4) :
  16 < t.b^2 + t.c^2 ∧ t.b^2 + t.c^2 ≤ 32 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_range_of_b_squared_plus_c_squared_l1822_182213


namespace NUMINAMATH_CALUDE_sum_first_four_is_sixty_l1822_182248

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum_first_two : a + a * r = 15
  sum_first_six : a * (1 - r^6) / (1 - r) = 93

/-- The sum of the first 4 terms of the geometric sequence is 60 -/
theorem sum_first_four_is_sixty (seq : GeometricSequence) :
  seq.a * (1 - seq.r^4) / (1 - seq.r) = 60 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_four_is_sixty_l1822_182248


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1822_182208

/-- Given a geometric sequence with first term 3 and common ratio 5/2, 
    the tenth term is equal to 5859375/512. -/
theorem tenth_term_of_geometric_sequence : 
  let a : ℚ := 3
  let r : ℚ := 5/2
  let n : ℕ := 10
  let a_n := a * r^(n - 1)
  a_n = 5859375/512 := by sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l1822_182208


namespace NUMINAMATH_CALUDE_range_of_fraction_l1822_182205

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  1/4 < a/b ∧ a/b < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1822_182205


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1822_182264

/-- Calculate the compound interest given principal, rate, and time -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Calculate the total amount returned after compound interest -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.05 2 = 492 →
  total_amount P 492 = 5292 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1822_182264


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l1822_182206

/-- The quadratic function f(x) = -3x^2 + 24x - 45 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 24 * x - 45

/-- The same function in completed square form a(x+b)^2 + c -/
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

theorem quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, f x = g x a b c) ∧ (a + b + c = 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l1822_182206


namespace NUMINAMATH_CALUDE_c_investment_amount_l1822_182276

/-- A business partnership between C and D -/
structure Business where
  c_investment : ℕ
  d_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- The business scenario as described in the problem -/
def scenario : Business where
  c_investment := 0  -- Unknown, to be proved
  d_investment := 1500
  total_profit := 500
  d_profit_share := 100

/-- Theorem stating C's investment amount -/
theorem c_investment_amount (b : Business) (h1 : b = scenario) :
  b.c_investment = 6000 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_amount_l1822_182276


namespace NUMINAMATH_CALUDE_f_fixed_point_l1822_182285

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

-- Define the repeated application of f
def repeat_f (p q : ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f p q (repeat_f p q n x)

-- State the theorem
theorem f_fixed_point (p q : ℝ) :
  (∀ x ∈ Set.Icc 2 4, |f p q x| ≤ 1/2) →
  repeat_f p q 2017 ((5 - Real.sqrt 11) / 2) = (5 + Real.sqrt 11) / 2 :=
by sorry

end NUMINAMATH_CALUDE_f_fixed_point_l1822_182285


namespace NUMINAMATH_CALUDE_nancy_tuition_ratio_l1822_182274

/-- Calculates the ratio of student loan to scholarship for Nancy's university tuition --/
theorem nancy_tuition_ratio :
  let tuition : ℚ := 22000
  let parents_contribution : ℚ := tuition / 2
  let scholarship : ℚ := 3000
  let work_hours : ℚ := 200
  let hourly_rate : ℚ := 10
  let work_earnings : ℚ := work_hours * hourly_rate
  let total_available : ℚ := parents_contribution + scholarship + work_earnings
  let loan_amount : ℚ := tuition - total_available
  loan_amount / scholarship = 2 := by sorry

end NUMINAMATH_CALUDE_nancy_tuition_ratio_l1822_182274


namespace NUMINAMATH_CALUDE_triangle_count_for_2016_30_triangle_count_formula_l1822_182220

/-- Represents the number of non-overlapping triangles in a mesh region --/
def f (m n : ℕ) : ℕ := 2 * m - n - 2

/-- The theorem states that for 2016 points forming a 30-gon convex hull, 
    the number of non-overlapping triangles is 4000 --/
theorem triangle_count_for_2016_30 :
  f 2016 30 = 4000 := by
  sorry

/-- A more general theorem about the formula for f(m, n) --/
theorem triangle_count_formula {m n : ℕ} (h_m : m > 2) (h_n : 3 ≤ n ∧ n ≤ m) :
  f m n = 2 * m - n - 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_for_2016_30_triangle_count_formula_l1822_182220


namespace NUMINAMATH_CALUDE_factory_production_excess_l1822_182252

theorem factory_production_excess (monthly_plan : ℝ) :
  let january_production := 1.05 * monthly_plan
  let february_production := 1.04 * january_production
  let two_month_plan := 2 * monthly_plan
  let total_production := january_production + february_production
  (total_production - two_month_plan) / two_month_plan = 0.071 := by
sorry

end NUMINAMATH_CALUDE_factory_production_excess_l1822_182252


namespace NUMINAMATH_CALUDE_hexagons_from_circle_points_l1822_182209

/-- The number of points on the circle -/
def n : ℕ := 15

/-- The number of vertices in a hexagon -/
def k : ℕ := 6

/-- A function to calculate binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := (Nat.choose n k)

/-- Theorem: The number of distinct convex hexagons formed from 15 points on a circle is 5005 -/
theorem hexagons_from_circle_points : binomial_coefficient n k = 5005 := by
  sorry

#eval binomial_coefficient n k  -- This should output 5005

end NUMINAMATH_CALUDE_hexagons_from_circle_points_l1822_182209


namespace NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l1822_182226

/-- Calculates the teacher's earnings for piano lessons over a given number of weeks -/
def teacher_earnings (rate_per_half_hour : ℕ) (lesson_duration_hours : ℕ) (lessons_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  rate_per_half_hour * (lesson_duration_hours * 2) * lessons_per_week * num_weeks

/-- Proves that the teacher earns $100 in 5 weeks under the given conditions -/
theorem teacher_earnings_five_weeks :
  teacher_earnings 10 1 1 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_teacher_earnings_five_weeks_l1822_182226


namespace NUMINAMATH_CALUDE_bookshelf_problem_l1822_182233

theorem bookshelf_problem (x : ℕ) 
  (h1 : (4 * x : ℚ) / (5 * x + 35 + 6 * x + 4 * x) = 22 / 100) : 
  4 * x = 44 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_problem_l1822_182233


namespace NUMINAMATH_CALUDE_katie_miles_run_l1822_182242

/-- Given that Adam ran 125 miles and Adam ran 80 miles more than Katie, prove that Katie ran 45 miles. -/
theorem katie_miles_run (adam_miles : ℕ) (difference : ℕ) (katie_miles : ℕ) 
  (h1 : adam_miles = 125)
  (h2 : adam_miles = katie_miles + difference)
  (h3 : difference = 80) : 
  katie_miles = 45 := by
sorry

end NUMINAMATH_CALUDE_katie_miles_run_l1822_182242


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l1822_182290

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 6000 → 
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 1800 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l1822_182290


namespace NUMINAMATH_CALUDE_inequality_count_l1822_182229

theorem inequality_count (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : x^2 < a^2) (hyb : |y| < |b|) : 
  ∃! (n : ℕ), n = 2 ∧ 
  (n = (if x^2 + y^2 < a^2 + b^2 then 1 else 0) +
       (if x^2 - y^2 < a^2 - b^2 then 1 else 0) +
       (if x^2 * y^2 < a^2 * b^2 then 1 else 0) +
       (if x^2 / y^2 < a^2 / b^2 then 1 else 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_count_l1822_182229


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l1822_182278

/-- The weight that a toy bridge must support given the following conditions:
  * There are 6 cans of soda, each containing 12 ounces of soda
  * Each empty can weighs 2 ounces
  * There are 2 additional empty cans
-/
theorem toy_bridge_weight (soda_cans : ℕ) (soda_per_can : ℕ) (empty_can_weight : ℕ) (additional_cans : ℕ) :
  soda_cans = 6 →
  soda_per_can = 12 →
  empty_can_weight = 2 →
  additional_cans = 2 →
  (soda_cans * soda_per_can) + ((soda_cans + additional_cans) * empty_can_weight) = 88 := by
  sorry

end NUMINAMATH_CALUDE_toy_bridge_weight_l1822_182278


namespace NUMINAMATH_CALUDE_grandfather_gift_problem_l1822_182293

theorem grandfather_gift_problem (x y : ℕ) : 
  x + y = 30 → 
  5 * x * (x + 1) + 5 * y * (y + 1) = 2410 → 
  (x = 16 ∧ y = 14) ∨ (x = 14 ∧ y = 16) := by
sorry

end NUMINAMATH_CALUDE_grandfather_gift_problem_l1822_182293


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1822_182275

theorem triangle_angle_calculation (A B C : ℝ) :
  A + B + C = 180 →
  B = 4 * A →
  C - B = 27 →
  A = 17 ∧ B = 68 ∧ C = 95 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1822_182275


namespace NUMINAMATH_CALUDE_square_difference_l1822_182222

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 45) (h2 : x * y = 10) :
  (x - y)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1822_182222


namespace NUMINAMATH_CALUDE_gift_box_volume_l1822_182235

/-- The volume of a rectangular box. -/
def boxVolume (length width height : ℝ) : ℝ :=
  length * width * height

/-- Theorem: The volume of a gift box with dimensions 9 cm wide, 4 cm long, and 7 cm high is 252 cubic centimeters. -/
theorem gift_box_volume :
  boxVolume 4 9 7 = 252 := by
  sorry

end NUMINAMATH_CALUDE_gift_box_volume_l1822_182235


namespace NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l1822_182272

-- Problem 1
theorem factorization_problem1 (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by sorry

-- Problem 2
theorem factorization_problem2 (a b x y : ℝ) : 
  a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_factorization_problem2_l1822_182272


namespace NUMINAMATH_CALUDE_last_digit_of_2021_2021_l1822_182255

-- Define the table size
def n : Nat := 2021

-- Define the cell value function
def cellValue (x y : Nat) : Nat :=
  if x = 1 ∧ y = 1 then 0
  else 2^(x + y - 2) - 1

-- State the theorem
theorem last_digit_of_2021_2021 :
  (cellValue n n) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_2021_2021_l1822_182255


namespace NUMINAMATH_CALUDE_lauren_pencils_l1822_182258

/-- Proves that Lauren received 6 pencils given the conditions of the problem -/
theorem lauren_pencils (initial_pencils : ℕ) (remaining_pencils : ℕ) (matt_extra : ℕ) :
  initial_pencils = 24 →
  remaining_pencils = 9 →
  matt_extra = 3 →
  ∃ (lauren_pencils : ℕ),
    lauren_pencils + (lauren_pencils + matt_extra) = initial_pencils - remaining_pencils ∧
    lauren_pencils = 6 :=
by sorry

end NUMINAMATH_CALUDE_lauren_pencils_l1822_182258


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_l1822_182211

def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem hyperbola_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ ∧
    ¬∃ (x₁' y₁' x₂' y₂' : ℝ),
      hyperbola x₁' y₁' ∧
      hyperbola x₂' y₂' ∧
      (is_midpoint 1 1 x₁' y₁' x₂' y₂' ∨
       is_midpoint (-1) 2 x₁' y₁' x₂' y₂' ∨
       is_midpoint 1 3 x₁' y₁' x₂' y₂') :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_l1822_182211


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1822_182200

/-- The quadratic function f(x) = x^2 - 8x + 18 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 18

theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (x_min : ℝ), f x_min = 2) ∧
  (∀ (x : ℝ), f x = 2 → x = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1822_182200


namespace NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l1822_182207

theorem greatest_integer_for_integer_fraction : 
  ∃ (x : ℤ), x = 53 ∧ 
  (∀ (y : ℤ), y > 53 → ¬(∃ (z : ℤ), (y^2 + 2*y + 13) / (y - 5) = z)) ∧
  (∃ (z : ℤ), (x^2 + 2*x + 13) / (x - 5) = z) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_for_integer_fraction_l1822_182207


namespace NUMINAMATH_CALUDE_find_m_value_l1822_182295

theorem find_m_value (n m : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + n) = x^2 + m*x - 21) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_find_m_value_l1822_182295


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1822_182214

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence where a_7 + a_9 = 16, a_8 = 8 -/
theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 7 + a 9 = 16) :
  a 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1822_182214


namespace NUMINAMATH_CALUDE_even_function_range_l1822_182267

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∃ y ∈ Set.Icc (1 + a) 2, -y ∈ Set.Icc (1 + a) 2) →
  (Set.range (f a b) = Set.Icc (-10) 2) :=
sorry

end NUMINAMATH_CALUDE_even_function_range_l1822_182267


namespace NUMINAMATH_CALUDE_shaded_triangle_probability_l1822_182216

/-- Given a set of triangles with equal selection probability, 
    this function calculates the probability of selecting a shaded triangle -/
def probability_shaded_triangle (total_triangles : ℕ) (shaded_triangles : ℕ) : ℚ :=
  shaded_triangles / total_triangles

/-- Theorem: The probability of selecting a shaded triangle 
    given 6 total triangles and 2 shaded triangles is 1/3 -/
theorem shaded_triangle_probability :
  probability_shaded_triangle 6 2 = 1/3 := by
  sorry

#eval probability_shaded_triangle 6 2

end NUMINAMATH_CALUDE_shaded_triangle_probability_l1822_182216


namespace NUMINAMATH_CALUDE_dog_food_calculation_l1822_182247

theorem dog_food_calculation (num_dogs : ℕ) (total_food_kg : ℕ) (num_days : ℕ) 
  (h1 : num_dogs = 4)
  (h2 : total_food_kg = 14)
  (h3 : num_days = 14) :
  (total_food_kg * 1000) / (num_dogs * num_days) = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_dog_food_calculation_l1822_182247


namespace NUMINAMATH_CALUDE_percentage_difference_l1822_182270

theorem percentage_difference (x y : ℝ) (h : x = 4 * y) :
  (x - y) / x * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1822_182270


namespace NUMINAMATH_CALUDE_function_inequality_l1822_182230

open Real

/-- A function satisfying the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧
  (∀ x, (x - 1) * (deriv f x - f x) > 0) ∧
  (∀ x, f (2 - x) = f x * Real.exp (2 - 2*x))

/-- The main theorem -/
theorem function_inequality (f : ℝ → ℝ) (h : satisfies_conditions f) :
  f 3 < Real.exp 3 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1822_182230


namespace NUMINAMATH_CALUDE_hidden_dots_count_l1822_182237

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6

/-- The total number of dots on three dice -/
def total_dots : ℕ := 3 * die_sum

/-- The sum of visible numbers on the dice -/
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 5 + 6

/-- The number of hidden dots on the dice -/
def hidden_dots : ℕ := total_dots - visible_dots

theorem hidden_dots_count : hidden_dots = 41 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l1822_182237


namespace NUMINAMATH_CALUDE_trip_cost_equals_bills_cost_l1822_182265

/-- Proves that the cost of Liam's trip to Paris is equal to the cost of his bills. -/
theorem trip_cost_equals_bills_cost (
  monthly_savings : ℕ)
  (savings_duration_years : ℕ)
  (bills_cost : ℕ)
  (money_left_after_bills : ℕ)
  (h1 : monthly_savings = 500)
  (h2 : savings_duration_years = 2)
  (h3 : bills_cost = 3500)
  (h4 : money_left_after_bills = 8500)
  : bills_cost = monthly_savings * savings_duration_years * 12 - money_left_after_bills :=
by sorry

end NUMINAMATH_CALUDE_trip_cost_equals_bills_cost_l1822_182265


namespace NUMINAMATH_CALUDE_impossible_to_raise_average_l1822_182203

def current_scores : List ℝ := [82, 75, 88, 91, 78]
def max_score : ℝ := 100
def target_increase : ℝ := 5

theorem impossible_to_raise_average (scores : List ℝ) (max_score : ℝ) (target_increase : ℝ) :
  let current_avg := scores.sum / scores.length
  let new_sum := scores.sum + max_score
  let new_avg := new_sum / (scores.length + 1)
  new_avg < current_avg + target_increase :=
by sorry

end NUMINAMATH_CALUDE_impossible_to_raise_average_l1822_182203


namespace NUMINAMATH_CALUDE_total_food_is_point_nine_l1822_182266

/-- The amount of cat food Jake needs to serve each day for one cat -/
def food_for_one_cat : ℝ := 0.5

/-- The extra amount of cat food needed for the second cat -/
def extra_food_for_second_cat : ℝ := 0.4

/-- The total amount of cat food Jake needs to serve each day for two cats -/
def total_food_for_two_cats : ℝ := food_for_one_cat + extra_food_for_second_cat

theorem total_food_is_point_nine :
  total_food_for_two_cats = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_total_food_is_point_nine_l1822_182266


namespace NUMINAMATH_CALUDE_hypergeometric_distribution_proof_l1822_182218

def hypergeometric_prob (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

theorem hypergeometric_distribution_proof (N n m : ℕ) 
  (h1 : N = 10) (h2 : n = 8) (h3 : m = 2) : 
  (hypergeometric_prob N n m 0 = 1/45) ∧
  (hypergeometric_prob N n m 1 = 16/45) ∧
  (hypergeometric_prob N n m 2 = 28/45) := by
  sorry

end NUMINAMATH_CALUDE_hypergeometric_distribution_proof_l1822_182218


namespace NUMINAMATH_CALUDE_orchid_bushes_planted_l1822_182202

/-- The number of orchid bushes planted in the park -/
theorem orchid_bushes_planted (initial : ℕ) (final : ℕ) (planted : ℕ) : 
  initial = 2 → final = 6 → planted = final - initial → planted = 4 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_planted_l1822_182202


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l1822_182249

theorem unique_quadratic_pair : ∃! (b c : ℕ+), 
  (∃! x : ℝ, x^2 + b*x + c = 0) ∧ 
  (∃! x : ℝ, x^2 + c*x + b = 0) := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l1822_182249


namespace NUMINAMATH_CALUDE_c_percentage_less_than_d_l1822_182282

def full_marks : ℕ := 500
def d_marks : ℕ := (80 * full_marks) / 100
def a_marks : ℕ := 360

def b_marks : ℕ := a_marks * 100 / 90
def c_marks : ℕ := b_marks * 100 / 125

theorem c_percentage_less_than_d :
  (d_marks - c_marks) * 100 / d_marks = 20 := by sorry

end NUMINAMATH_CALUDE_c_percentage_less_than_d_l1822_182282


namespace NUMINAMATH_CALUDE_domino_coverage_l1822_182271

theorem domino_coverage (n k : ℕ+) :
  (∃ (coverage : Fin n × Fin n → Fin k × Bool),
    (∀ (i j : Fin n), ∃ (x : Fin k) (b : Bool),
      coverage (i, j) = (x, b) ∧
      (b = true → coverage (i, j.succ) = (x, false)) ∧
      (b = false → coverage (i.succ, j) = (x, true))))
  ↔ k ∣ n := by sorry

end NUMINAMATH_CALUDE_domino_coverage_l1822_182271


namespace NUMINAMATH_CALUDE_gcd_1821_2993_l1822_182296

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1821_2993_l1822_182296


namespace NUMINAMATH_CALUDE_square_difference_equality_l1822_182294

theorem square_difference_equality : (15 + 12)^2 - (15 - 12)^2 = 720 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1822_182294


namespace NUMINAMATH_CALUDE_product_one_minus_reciprocals_l1822_182234

theorem product_one_minus_reciprocals : (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_product_one_minus_reciprocals_l1822_182234


namespace NUMINAMATH_CALUDE_A_is_uncountable_l1822_182240

-- Define the set A as the closed interval [0, 1]
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Theorem stating that A is uncountable
theorem A_is_uncountable : ¬ (Countable A) := by
  sorry

end NUMINAMATH_CALUDE_A_is_uncountable_l1822_182240


namespace NUMINAMATH_CALUDE_letians_estimate_l1822_182268

/-- Given x and y are positive real numbers with x > y, and z and w are small positive real numbers with z > w,
    prove that (x + z) - (y - w) > x - y. -/
theorem letians_estimate (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
    (hz : z > 0) (hw : w > 0) (hzw : z > w) : 
  (x + z) - (y - w) > x - y := by
  sorry

end NUMINAMATH_CALUDE_letians_estimate_l1822_182268


namespace NUMINAMATH_CALUDE_problem_1_l1822_182260

theorem problem_1 (x y : ℝ) (h : |x + 2| + |y - 3| = 0) : x - y + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1822_182260


namespace NUMINAMATH_CALUDE_hyperbolas_M_value_l1822_182224

/-- Two hyperbolas with the same asymptotes -/
def hyperbolas_same_asymptotes (M : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 → y = k*x ∨ y = -k*x) ∧
  (∀ (x y : ℝ), y^2/25 - x^2/M = 1 → y = k*x ∨ y = -k*x)

/-- The theorem stating that M must equal 225/16 for the hyperbolas to have the same asymptotes -/
theorem hyperbolas_M_value :
  hyperbolas_same_asymptotes (225/16) ∧
  ∀ M : ℝ, hyperbolas_same_asymptotes M → M = 225/16 :=
sorry

end NUMINAMATH_CALUDE_hyperbolas_M_value_l1822_182224


namespace NUMINAMATH_CALUDE_vector_magnitude_equivalence_l1822_182210

/-- Given non-zero, non-collinear vectors a and b, prove that |a| = |b| if and only if |a + 2b| = |2a + b| -/
theorem vector_magnitude_equivalence {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n] 
  (a b : n) (ha : a ≠ 0) (hb : b ≠ 0) (hnc : ¬ Collinear ℝ {0, a, b}) :
  ‖a‖ = ‖b‖ ↔ ‖a + 2 • b‖ = ‖2 • a + b‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_equivalence_l1822_182210


namespace NUMINAMATH_CALUDE_greatest_x_value_l1822_182201

theorem greatest_x_value (x : ℕ+) (y : ℕ) (b : ℚ) 
  (h1 : y.Prime)
  (h2 : y = 2)
  (h3 : b = 3.56)
  (h4 : (b * y^x.val : ℚ) < 600000) :
  x.val ≤ 17 ∧ ∃ (x' : ℕ+), x'.val = 17 ∧ (b * y^x'.val : ℚ) < 600000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1822_182201


namespace NUMINAMATH_CALUDE_total_hair_cut_equals_41_l1822_182225

/-- Represents the hair length of a person before and after a haircut. -/
structure Haircut where
  original : ℕ  -- Original hair length in inches
  cut : ℕ       -- Amount of hair cut off in inches

/-- Calculates the total amount of hair cut off from multiple haircuts. -/
def total_hair_cut (haircuts : List Haircut) : ℕ :=
  haircuts.map (·.cut) |>.sum

/-- Theorem stating that the total hair cut off from Isabella, Damien, and Ella is 41 inches. -/
theorem total_hair_cut_equals_41 : 
  let isabella : Haircut := { original := 18, cut := 9 }
  let damien : Haircut := { original := 24, cut := 12 }
  let ella : Haircut := { original := 30, cut := 20 }
  total_hair_cut [isabella, damien, ella] = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_hair_cut_equals_41_l1822_182225


namespace NUMINAMATH_CALUDE_complement_union_problem_l1822_182250

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem complement_union_problem (A B : Set ℕ) 
  (h1 : (U \ A) ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : (U \ A) ∩ (U \ B) = {2}) :
  U \ (A ∪ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l1822_182250


namespace NUMINAMATH_CALUDE_math_contest_score_difference_l1822_182217

theorem math_contest_score_difference (score60 score75 score85 score95 : ℝ)
  (percent60 percent75 percent85 percent95 : ℝ)
  (h1 : score60 = 60)
  (h2 : score75 = 75)
  (h3 : score85 = 85)
  (h4 : score95 = 95)
  (h5 : percent60 = 0.2)
  (h6 : percent75 = 0.4)
  (h7 : percent85 = 0.25)
  (h8 : percent95 = 0.15)
  (h9 : percent60 + percent75 + percent85 + percent95 = 1) :
  let median := score75
  let mean := percent60 * score60 + percent75 * score75 + percent85 * score85 + percent95 * score95
  median - mean = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_math_contest_score_difference_l1822_182217


namespace NUMINAMATH_CALUDE_translation_result_l1822_182292

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Translates a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  ⟨p.x, p.y + dy⟩

/-- The initial point M -/
def M : Point := ⟨2, 5⟩

/-- The resulting point after translations -/
def resultingPoint : Point :=
  translateVertical (translateHorizontal M (-2)) (-3)

theorem translation_result :
  resultingPoint = ⟨0, 2⟩ := by sorry

end NUMINAMATH_CALUDE_translation_result_l1822_182292


namespace NUMINAMATH_CALUDE_vasya_can_win_l1822_182251

/-- Represents the state of the water pots -/
structure PotState :=
  (pot3 : Nat)
  (pot5 : Nat)
  (pot7 : Nat)

/-- Represents a move by Vasya -/
inductive VasyaMove
  | FillPot3
  | FillPot5
  | FillPot7
  | TransferPot3ToPot5
  | TransferPot3ToPot7
  | TransferPot5ToPot3
  | TransferPot5ToPot7
  | TransferPot7ToPot3
  | TransferPot7ToPot5

/-- Represents a move by Dima -/
inductive DimaMove
  | EmptyPot3
  | EmptyPot5
  | EmptyPot7

/-- Applies Vasya's move to the current state -/
def applyVasyaMove (state : PotState) (move : VasyaMove) : PotState :=
  sorry

/-- Applies Dima's move to the current state -/
def applyDimaMove (state : PotState) (move : DimaMove) : PotState :=
  sorry

/-- Checks if the game is won (1 liter in any pot) -/
def isGameWon (state : PotState) : Bool :=
  state.pot3 = 1 || state.pot5 = 1 || state.pot7 = 1

/-- Theorem: Vasya can win the game -/
theorem vasya_can_win :
  ∃ (moves : List (VasyaMove × VasyaMove)),
    ∀ (dimaMoves : List DimaMove),
      let finalState := (moves.zip dimaMoves).foldl
        (fun state (vasyaMoves, dimaMove) =>
          let s1 := applyVasyaMove state vasyaMoves.1
          let s2 := applyVasyaMove s1 vasyaMoves.2
          applyDimaMove s2 dimaMove)
        { pot3 := 0, pot5 := 0, pot7 := 0 }
      isGameWon finalState :=
by
  sorry


end NUMINAMATH_CALUDE_vasya_can_win_l1822_182251


namespace NUMINAMATH_CALUDE_conditional_probability_equal_marginal_l1822_182283

-- Define the sample space and events
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)
variable (A B : Set Ω)

-- Define the probabilities and independence
variable (hA : P A = 1/6)
variable (hB : P B = 1/2)
variable (hInd : P.Independent A B)

-- State the theorem
theorem conditional_probability_equal_marginal
  (h_prob_B_pos : P B > 0) :
  P.condProb A B = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_equal_marginal_l1822_182283


namespace NUMINAMATH_CALUDE_sum_of_specific_pair_l1822_182204

theorem sum_of_specific_pair : 
  ∃ (pairs : List (ℕ × ℕ)), 
    (pairs.length = 300) ∧ 
    (∀ (p : ℕ × ℕ), p ∈ pairs → 
      p.1 < 1500 ∧ p.2 < 1500 ∧ 
      p.2 = p.1 + 1 ∧ 
      (p.1 + p.2) % 5 = 0) ∧
    (57, 58) ∈ pairs →
    57 + 58 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_pair_l1822_182204


namespace NUMINAMATH_CALUDE_expression_decrease_l1822_182227

theorem expression_decrease (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 0.9 * x
  let y' := 0.7 * y
  (x' ^ 2 * y' ^ 3) / (x ^ 2 * y ^ 3) = 0.27783 := by
sorry

end NUMINAMATH_CALUDE_expression_decrease_l1822_182227


namespace NUMINAMATH_CALUDE_max_garden_area_l1822_182263

/-- The maximum area of a rectangular garden with 150 feet of fencing and natural number side lengths -/
theorem max_garden_area : 
  ∃ (l w : ℕ), 
    2 * (l + w) = 150 ∧ 
    l * w = 1406 ∧
    ∀ (a b : ℕ), 2 * (a + b) = 150 → a * b ≤ 1406 := by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l1822_182263


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1822_182291

theorem two_digit_number_problem (a b : ℕ) : 
  b = 2 * a → 
  (10 * a + b) - (10 * b + a) = 36 → 
  (a + b) - (b - a) = 8 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1822_182291


namespace NUMINAMATH_CALUDE_students_liking_sports_l1822_182277

theorem students_liking_sports (basketball : Finset ℕ) (cricket : Finset ℕ) 
  (h1 : basketball.card = 7)
  (h2 : cricket.card = 5)
  (h3 : (basketball ∩ cricket).card = 3) :
  (basketball ∪ cricket).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l1822_182277


namespace NUMINAMATH_CALUDE_intersection_A_B_l1822_182221

def A : Set ℕ := {1, 2, 3, 4, 5}

def B : Set ℕ := {x : ℕ | (x - 1) * (x - 4) < 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1822_182221


namespace NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l1822_182279

theorem triangle_circumcircle_diameter 
  (a b c : ℝ) 
  (ha : a = 25) 
  (hb : b = 39) 
  (hc : c = 40) : 
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * (a * b * c) / (4 * area) = 125 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_diameter_l1822_182279


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1822_182246

theorem solution_set_inequality (x : ℝ) :
  (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1822_182246


namespace NUMINAMATH_CALUDE_product_of_roots_l1822_182257

theorem product_of_roots (p q r : ℂ) : 
  (3 * p^3 - 9 * p^2 + 5 * p - 15 = 0) →
  (3 * q^3 - 9 * q^2 + 5 * q - 15 = 0) →
  (3 * r^3 - 9 * r^2 + 5 * r - 15 = 0) →
  p * q * r = 5 := by sorry

end NUMINAMATH_CALUDE_product_of_roots_l1822_182257


namespace NUMINAMATH_CALUDE_ten_stairs_ways_l1822_182273

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2) + stair_ways (m + 3)

theorem ten_stairs_ways : stair_ways 10 = 401 := by
  sorry

end NUMINAMATH_CALUDE_ten_stairs_ways_l1822_182273


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1822_182289

/-- Given a telescope that increases the visual range by 66.67% to reach 150 kilometers,
    prove that the initial visual range without the telescope is 90 kilometers. -/
theorem telescope_visual_range (initial_range : ℝ) : 
  (initial_range + initial_range * (2/3) = 150) → initial_range = 90 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1822_182289


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1822_182299

theorem quadratic_equation_roots (p q : ℝ) : 
  (∀ x, x^2 + p*x + q = 0 → (x + 1)^2 - p^2*(x + 1) + p*q = 0) →
  ((p = 1 ∧ ∃ q : ℝ, True) ∨ (p = -2 ∧ q = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1822_182299


namespace NUMINAMATH_CALUDE_pentagon_count_l1822_182244

/-- The number of distinct points on the circumference of a circle -/
def n : ℕ := 15

/-- The number of vertices in each polygon -/
def k : ℕ := 5

/-- The number of distinct convex pentagons that can be formed -/
def num_pentagons : ℕ := Nat.choose n k

theorem pentagon_count :
  num_pentagons = 3003 :=
sorry

end NUMINAMATH_CALUDE_pentagon_count_l1822_182244
