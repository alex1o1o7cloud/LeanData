import Mathlib

namespace NUMINAMATH_CALUDE_customers_remaining_l1499_149929

theorem customers_remaining (initial : ℕ) (difference : ℕ) (final : ℕ) : 
  initial = 19 → difference = 15 → final = initial - difference → final = 4 := by
  sorry

end NUMINAMATH_CALUDE_customers_remaining_l1499_149929


namespace NUMINAMATH_CALUDE_waiter_customer_count_l1499_149924

theorem waiter_customer_count :
  let num_tables : ℕ := 9
  let women_per_table : ℕ := 7
  let men_per_table : ℕ := 3
  let total_customers := num_tables * (women_per_table + men_per_table)
  total_customers = 90 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l1499_149924


namespace NUMINAMATH_CALUDE_triangle_mn_length_l1499_149943

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  AB = 5 ∧ AC = 4 ∧ BC = 6

-- Define the angle bisector and point X
def angleBisector (t : Triangle) : ℝ × ℝ → Prop := sorry

-- Define points M and N
def pointM (t : Triangle) : ℝ × ℝ := sorry
def pointN (t : Triangle) : ℝ × ℝ := sorry

-- Define parallel lines
def isParallel (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_mn_length (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : ∃ X, angleBisector t X ∧ X.1 ∈ Set.Icc t.A.1 t.B.1 ∧ X.2 = t.A.2)
  (h3 : isParallel (X, pointM t) (t.A, t.C))
  (h4 : isParallel (X, pointN t) (t.B, t.C)) :
  let MN := Real.sqrt ((pointM t).1 - (pointN t).1)^2 + ((pointM t).2 - (pointN t).2)^2
  MN = 3 * Real.sqrt 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_mn_length_l1499_149943


namespace NUMINAMATH_CALUDE_last_digit_of_product_l1499_149972

theorem last_digit_of_product (n : ℕ) : 
  (3^2001 * 7^2002 * 13^2003) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l1499_149972


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_50_l1499_149913

/-- The cost of paint per kg, given the coverage rate and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage_rate : ℝ) (cube_side : ℝ) (total_cost : ℝ) : ℝ :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage_rate
  total_cost / paint_needed

/-- The cost of paint per kg is 50, given the specified conditions. -/
theorem paint_cost_is_50 : paint_cost_per_kg 20 20 6000 = 50 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_50_l1499_149913


namespace NUMINAMATH_CALUDE_intersection_coordinate_sum_l1499_149908

/-- Given a triangle ABC with vertices A(2,8), B(2,2), C(10,2), 
    D is the midpoint of AB, E is the midpoint of BC, 
    and F is the intersection point of AE and CD. -/
theorem intersection_coordinate_sum (A B C D E F : ℝ × ℝ) : 
  A = (2, 8) → B = (2, 2) → C = (10, 2) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (F.2 - A.2) / (F.1 - A.1) = (E.2 - A.2) / (E.1 - A.1) →
  (F.2 - C.2) / (F.1 - C.1) = (D.2 - C.2) / (D.1 - C.1) →
  F.1 + F.2 = 13 := by
sorry

end NUMINAMATH_CALUDE_intersection_coordinate_sum_l1499_149908


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l1499_149985

theorem rectangular_plot_dimensions :
  ∀ (width length area : ℕ),
    length = width + 1 →
    area = width * length →
    1000 ≤ area ∧ area < 10000 →
    (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ area = 1000 * a + 100 * a + 10 * b + b) →
    width ∈ ({33, 66, 99} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l1499_149985


namespace NUMINAMATH_CALUDE_bobs_family_adults_l1499_149940

/-- The number of adults in Bob's family -/
def num_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) : ℕ :=
  (total_apples - num_children * apples_per_child) / apples_per_adult

/-- Theorem stating that the number of adults in Bob's family is 40 -/
theorem bobs_family_adults :
  num_adults 450 33 10 3 = 40 := by
  sorry

#eval num_adults 450 33 10 3

end NUMINAMATH_CALUDE_bobs_family_adults_l1499_149940


namespace NUMINAMATH_CALUDE_ice_cube_volume_l1499_149974

theorem ice_cube_volume (initial_volume : ℝ) : 
  initial_volume > 0 →
  (initial_volume * (1/4) * (1/4) = 0.75) →
  initial_volume = 12 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l1499_149974


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1499_149934

theorem quadratic_roots_to_coefficients :
  ∀ (p q : ℝ),
    (∀ x : ℝ, x^2 + p*x + q = 0 ↔ x = -2 ∨ x = 3) →
    p = -1 ∧ q = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l1499_149934


namespace NUMINAMATH_CALUDE_probability_under_20_is_7_16_l1499_149956

/-- Represents a group of people with age categories --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : over30 + under20 = total

/-- The probability of selecting a person under 20 years old --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

theorem probability_under_20_is_7_16 (group : AgeGroup) 
  (h2 : group.total = 160) 
  (h3 : group.over30 = 90) : 
  probabilityUnder20 group = 7 / 16 := by
  sorry

#check probability_under_20_is_7_16

end NUMINAMATH_CALUDE_probability_under_20_is_7_16_l1499_149956


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1499_149911

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + b > 0) ↔ (0 < b ∧ b < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1499_149911


namespace NUMINAMATH_CALUDE_geometric_arithmetic_relation_l1499_149978

/-- A geometric sequence with positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic sequence -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_relation (a b : ℕ → ℝ) :
  geometric_sequence a
  → a 2 = 4
  → a 4 = 16
  → arithmetic_sequence b
  → b 3 = a 3
  → b 5 = a 5
  → ∀ n : ℕ, b n = 12 * n - 28 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_relation_l1499_149978


namespace NUMINAMATH_CALUDE_journey_speed_l1499_149999

/-- Proves the required speed for the second part of a journey given the total distance, total time, initial speed, and initial time. -/
theorem journey_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (h1 : total_distance = 24) 
  (h2 : total_time = 8) 
  (h3 : initial_speed = 4) 
  (h4 : initial_time = 4) 
  : 
  (total_distance - initial_speed * initial_time) / (total_time - initial_time) = 2 := by
  sorry

#check journey_speed

end NUMINAMATH_CALUDE_journey_speed_l1499_149999


namespace NUMINAMATH_CALUDE_a_share_is_3690_l1499_149926

/-- Calculates the share of profit for an investor in a partnership business. -/
def calculate_share_of_profit (investment_a investment_b investment_c total_profit : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  ratio_a * total_profit

/-- Theorem stating that A's share of the profit is 3690 given the specified investments and total profit. -/
theorem a_share_is_3690 :
  calculate_share_of_profit 6300 4200 10500 12300 = 3690 := by
  sorry

end NUMINAMATH_CALUDE_a_share_is_3690_l1499_149926


namespace NUMINAMATH_CALUDE_jellybean_theorem_l1499_149921

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let scarlett_took := 2 * shelby_ate
  let scarlett_returned := (scarlett_took * 2) / 5  -- 40% rounded down
  let shannon_refilled := (samantha_took + shelby_ate) / 2
  initial - samantha_took - shelby_ate + scarlett_returned + shannon_refilled

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 81. -/
theorem jellybean_theorem : final_jellybean_count 90 24 12 = 81 := by
  sorry

#eval final_jellybean_count 90 24 12

end NUMINAMATH_CALUDE_jellybean_theorem_l1499_149921


namespace NUMINAMATH_CALUDE_simplify_radicals_l1499_149923

theorem simplify_radicals : 
  Real.sqrt 18 * Real.sqrt 72 - Real.sqrt 32 = 36 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radicals_l1499_149923


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1499_149901

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, y = 2^x}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 < y ∧ y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1499_149901


namespace NUMINAMATH_CALUDE_largest_consecutive_composites_l1499_149938

theorem largest_consecutive_composites : ∃ (n : ℕ), 
  (n ≤ 36) ∧ 
  (∀ i ∈ Finset.range 7, 30 ≤ n - i ∧ n - i < 40 ∧ ¬(Nat.Prime (n - i))) ∧
  (∀ m : ℕ, m > n → 
    ¬(∀ i ∈ Finset.range 7, 30 ≤ m - i ∧ m - i < 40 ∧ ¬(Nat.Prime (m - i)))) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_composites_l1499_149938


namespace NUMINAMATH_CALUDE_star_operation_result_l1499_149917

-- Define the operation *
def star : Fin 4 → Fin 4 → Fin 4
| 1, 1 => 1 | 1, 2 => 2 | 1, 3 => 3 | 1, 4 => 4
| 2, 1 => 2 | 2, 2 => 4 | 2, 3 => 1 | 2, 4 => 3
| 3, 1 => 3 | 3, 2 => 1 | 3, 3 => 4 | 3, 4 => 2
| 4, 1 => 4 | 4, 2 => 3 | 4, 3 => 2 | 4, 4 => 1

-- State the theorem
theorem star_operation_result : star 4 (star 3 2) = 4 := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l1499_149917


namespace NUMINAMATH_CALUDE_persistent_iff_two_l1499_149962

/-- A number T is persistent if for any a, b, c, d ∈ ℝ \ {0, 1} satisfying
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T,
    we also have 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 ∧ a ≠ 1 ∧ b ≠ 0 ∧ b ≠ 1 ∧ c ≠ 0 ∧ c ≠ 1 ∧ d ≠ 0 ∧ d ≠ 1 →
    a + b + c + d = T →
    1/a + 1/b + 1/c + 1/d = T →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- The only persistent number is 2 -/
theorem persistent_iff_two : ∀ T : ℝ, isPersistent T ↔ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_persistent_iff_two_l1499_149962


namespace NUMINAMATH_CALUDE_permutation_equation_solution_l1499_149984

/-- Permutation function: number of ways to arrange k items out of m items -/
def A (m : ℕ) (k : ℕ) : ℕ := m.factorial / (m - k).factorial

/-- The theorem states that the equation 3A₈ⁿ⁻¹ = 4A₉ⁿ⁻² is satisfied when n = 9 -/
theorem permutation_equation_solution :
  ∃ n : ℕ, 3 * A 8 (n - 1) = 4 * A 9 (n - 2) ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_permutation_equation_solution_l1499_149984


namespace NUMINAMATH_CALUDE_total_entertainment_hours_l1499_149936

/-- Represents the hours spent on an activity for each day of the week -/
structure WeeklyHours :=
  (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ)
  (friday : ℕ) (saturday : ℕ) (sunday : ℕ)

/-- Calculates the total hours spent on an activity throughout the week -/
def totalHours (hours : WeeklyHours) : ℕ :=
  hours.monday + hours.tuesday + hours.wednesday + hours.thursday +
  hours.friday + hours.saturday + hours.sunday

/-- Haley's TV watching hours -/
def tvHours : WeeklyHours :=
  { monday := 0, tuesday := 2, wednesday := 0, thursday := 4,
    friday := 0, saturday := 6, sunday := 3 }

/-- Haley's video game playing hours -/
def gameHours : WeeklyHours :=
  { monday := 3, tuesday := 0, wednesday := 5, thursday := 0,
    friday := 1, saturday := 0, sunday := 0 }

theorem total_entertainment_hours :
  totalHours tvHours + totalHours gameHours = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_entertainment_hours_l1499_149936


namespace NUMINAMATH_CALUDE_ratio_equals_three_tenths_l1499_149946

-- Define the system of equations
def system (k x y z w : ℝ) : Prop :=
  x + 2*k*y + 4*z - w = 0 ∧
  4*x + k*y + 2*z + w = 0 ∧
  3*x + 5*y - 3*z + 2*w = 0 ∧
  2*x + 3*y + z - 4*w = 0

-- Theorem statement
theorem ratio_equals_three_tenths :
  ∃ (k x y z w : ℝ), 
    system k x y z w ∧ 
    x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0 ∧
    x * y / (z * w) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_three_tenths_l1499_149946


namespace NUMINAMATH_CALUDE_product_max_for_square_l1499_149967

/-- A quadrilateral inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The product of sums of opposite sides pairs -/
def productOfSums (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Theorem: The product of sums is maximum when the quadrilateral is a square -/
theorem product_max_for_square (q : CyclicQuadrilateral) :
  productOfSums q ≤ productOfSums { a := (q.a + q.b + q.c + q.d) / 4,
                                    b := (q.a + q.b + q.c + q.d) / 4,
                                    c := (q.a + q.b + q.c + q.d) / 4,
                                    d := (q.a + q.b + q.c + q.d) / 4,
                                    inscribed := sorry } := by
  sorry

end NUMINAMATH_CALUDE_product_max_for_square_l1499_149967


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1499_149961

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 25 (2*x) = Nat.choose 25 (x+4)) → (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1499_149961


namespace NUMINAMATH_CALUDE_all_parameterizations_valid_l1499_149996

/-- The line equation y = 2x - 4 -/
def line_equation (x y : ℝ) : Prop := y = 2 * x - 4

/-- Parameterization A -/
def param_A (t : ℝ) : ℝ × ℝ := (3 + t, -2 + 2*t)

/-- Parameterization B -/
def param_B (t : ℝ) : ℝ × ℝ := (4 + 2*t, 4*t)

/-- Parameterization C -/
def param_C (t : ℝ) : ℝ × ℝ := (t, -4 + 2*t)

/-- Parameterization D -/
def param_D (t : ℝ) : ℝ × ℝ := (1 + 0.5*t, -1 + t)

/-- Parameterization E -/
def param_E (t : ℝ) : ℝ × ℝ := (-1 - 2*t, -6 - 4*t)

/-- Theorem stating that all parameterizations are valid for the given line -/
theorem all_parameterizations_valid :
  (∀ t, line_equation (param_A t).1 (param_A t).2) ∧
  (∀ t, line_equation (param_B t).1 (param_B t).2) ∧
  (∀ t, line_equation (param_C t).1 (param_C t).2) ∧
  (∀ t, line_equation (param_D t).1 (param_D t).2) ∧
  (∀ t, line_equation (param_E t).1 (param_E t).2) :=
sorry

end NUMINAMATH_CALUDE_all_parameterizations_valid_l1499_149996


namespace NUMINAMATH_CALUDE_original_price_l1499_149905

theorem original_price (p q d : ℝ) (h_d_pos : d > 0) :
  let x := d / (1 + (p - q) / 100 - p * q / 10000)
  let price_after_increase := x * (1 + p / 100)
  let final_price := price_after_increase * (1 - q / 100)
  final_price = d :=
by sorry

end NUMINAMATH_CALUDE_original_price_l1499_149905


namespace NUMINAMATH_CALUDE_winter_hamburger_sales_l1499_149991

/-- Given the total annual sales and percentages for spring and summer,
    calculate the number of hamburgers sold in winter. -/
theorem winter_hamburger_sales
  (total_sales : ℝ)
  (spring_percent : ℝ)
  (summer_percent : ℝ)
  (h_total : total_sales = 20)
  (h_spring : spring_percent = 0.3)
  (h_summer : summer_percent = 0.35) :
  total_sales - (spring_percent * total_sales + summer_percent * total_sales + (1 - spring_percent - summer_percent) / 2 * total_sales) = 3.5 :=
sorry

end NUMINAMATH_CALUDE_winter_hamburger_sales_l1499_149991


namespace NUMINAMATH_CALUDE_combined_cost_price_is_430_95_l1499_149998

-- Define the parameters for each stock
def stock1_face_value : ℝ := 100
def stock1_discount_rate : ℝ := 0.04
def stock1_brokerage_rate : ℝ := 0.002

def stock2_face_value : ℝ := 200
def stock2_discount_rate : ℝ := 0.06
def stock2_brokerage_rate : ℝ := 0.0025

def stock3_face_value : ℝ := 150
def stock3_discount_rate : ℝ := 0.03
def stock3_brokerage_rate : ℝ := 0.005

-- Define a function to calculate the cost price of a stock
def cost_price (face_value discount_rate brokerage_rate : ℝ) : ℝ :=
  (face_value - face_value * discount_rate) + face_value * brokerage_rate

-- Define the total cost price
def total_cost_price : ℝ :=
  cost_price stock1_face_value stock1_discount_rate stock1_brokerage_rate +
  cost_price stock2_face_value stock2_discount_rate stock2_brokerage_rate +
  cost_price stock3_face_value stock3_discount_rate stock3_brokerage_rate

-- Theorem statement
theorem combined_cost_price_is_430_95 :
  total_cost_price = 430.95 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_price_is_430_95_l1499_149998


namespace NUMINAMATH_CALUDE_gcd_180_294_l1499_149965

theorem gcd_180_294 : Nat.gcd 180 294 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_294_l1499_149965


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1499_149944

def p (x : ℝ) : ℝ := x^3 + x^2 + 3
def q (x : ℝ) : ℝ := 2*x^4 + x^2 + 7

theorem constant_term_expansion :
  (p 0) * (q 0) = 21 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1499_149944


namespace NUMINAMATH_CALUDE_cubic_equation_product_l1499_149930

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2015 ∧ y₁^3 - 3*x₁^2*y₁ = 2014)
  (h₂ : x₂^3 - 3*x₂*y₂^2 = 2015 ∧ y₂^3 - 3*x₂^2*y₂ = 2014)
  (h₃ : x₃^3 - 3*x₃*y₃^2 = 2015 ∧ y₃^3 - 3*x₃^2*y₃ = 2014) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -4/1007 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l1499_149930


namespace NUMINAMATH_CALUDE_leah_coin_value_l1499_149945

/-- Represents the types of coins Leah has --/
inductive Coin
| Penny
| Nickel
| Dime

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

/-- Leah's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  total_coins : pennies + nickels + dimes = 15
  dime_nickel_relation : dimes - 1 = nickels

theorem leah_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + 
  c.nickels * coinValue Coin.Nickel + 
  c.dimes * coinValue Coin.Dime = 89 := by
  sorry

#check leah_coin_value

end NUMINAMATH_CALUDE_leah_coin_value_l1499_149945


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l1499_149919

/-- The sum of the infinite series ∑(n=1 to ∞) 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' (n : ℕ), 1 / (n * (n + 3)) = 11 / 18 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l1499_149919


namespace NUMINAMATH_CALUDE_kevin_bought_three_muffins_l1499_149920

/-- The number of muffins Kevin bought -/
def num_muffins : ℕ := 3

/-- The cost of juice in dollars -/
def juice_cost : ℚ := 145/100

/-- The total amount paid in dollars -/
def total_paid : ℚ := 370/100

/-- The cost of each muffin in dollars -/
def muffin_cost : ℚ := 75/100

/-- Theorem stating that the number of muffins Kevin bought is 3 -/
theorem kevin_bought_three_muffins :
  num_muffins = 3 ∧
  juice_cost + (num_muffins : ℚ) * muffin_cost = total_paid :=
sorry

end NUMINAMATH_CALUDE_kevin_bought_three_muffins_l1499_149920


namespace NUMINAMATH_CALUDE_unique_prime_power_of_four_minus_one_l1499_149915

theorem unique_prime_power_of_four_minus_one :
  ∃! (n : ℕ), n > 0 ∧ Nat.Prime (4^n - 1) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_power_of_four_minus_one_l1499_149915


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1499_149981

/-- Calculate the total cost for a group to eat at a restaurant -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℕ) 
  (total_people : ℕ) 
  (kids_count : ℕ) 
  (h1 : adult_meal_cost = 3)
  (h2 : total_people = 12)
  (h3 : kids_count = 7) :
  (total_people - kids_count) * adult_meal_cost = 15 := by
  sorry

#check restaurant_bill_calculation

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1499_149981


namespace NUMINAMATH_CALUDE_unit_digit_of_x_is_six_l1499_149992

theorem unit_digit_of_x_is_six :
  let x : ℤ := (-2)^1988
  ∃ k : ℤ, x = 10 * k + 6 :=
by sorry

end NUMINAMATH_CALUDE_unit_digit_of_x_is_six_l1499_149992


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l1499_149960

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 5

def ring_arrangements (total_rings : ℕ) (arranged_rings : ℕ) (fingers : ℕ) : ℕ :=
  (Nat.choose total_rings arranged_rings) * fingers * (Nat.factorial arranged_rings)

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange number_of_fingers = 756000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l1499_149960


namespace NUMINAMATH_CALUDE_alcohol_solution_percentage_l1499_149906

/-- Given a solution with initial volume and alcohol percentage, proves that adding pure alcohol to reach a target percentage results in the correct initial alcohol percentage. -/
theorem alcohol_solution_percentage
  (initial_volume : ℝ)
  (pure_alcohol_added : ℝ)
  (target_percentage : ℝ)
  (h1 : initial_volume = 6)
  (h2 : pure_alcohol_added = 3)
  (h3 : target_percentage = 0.5)
  : ∃ (initial_percentage : ℝ),
    initial_percentage * initial_volume + pure_alcohol_added =
    target_percentage * (initial_volume + pure_alcohol_added) ∧
    initial_percentage = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_percentage_l1499_149906


namespace NUMINAMATH_CALUDE_candy_cost_problem_l1499_149933

/-- The cost per pound of the first type of candy -/
def first_candy_cost : ℝ := sorry

/-- The weight of the first type of candy in pounds -/
def first_candy_weight : ℝ := 10

/-- The weight of the second type of candy in pounds -/
def second_candy_weight : ℝ := 20

/-- The cost per pound of the second type of candy -/
def second_candy_cost : ℝ := 5

/-- The cost per pound of the mixture -/
def mixture_cost : ℝ := 6

/-- The total weight of the mixture in pounds -/
def total_weight : ℝ := 30

theorem candy_cost_problem :
  first_candy_cost * first_candy_weight + 
  second_candy_cost * second_candy_weight = 
  mixture_cost * total_weight ∧
  first_candy_cost = 8 := by sorry

end NUMINAMATH_CALUDE_candy_cost_problem_l1499_149933


namespace NUMINAMATH_CALUDE_triangle_area_and_angle_B_l1499_149925

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_area_and_angle_B 
  (a b c : ℝ) 
  (h_triangle : Triangle a b c)
  (h_eq : b^2 = c^2 + a^2 - Real.sqrt 2 * a * c)
  (h_a : a = Real.sqrt 2)
  (h_cos_A : Real.cos A = 4/5)
  : Real.cos B = Real.sqrt 2 / 2 ∧ 
    ∃ (S : ℝ), S = 7/6 ∧ S = 1/2 * a * b * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_and_angle_B_l1499_149925


namespace NUMINAMATH_CALUDE_flour_needed_l1499_149902

theorem flour_needed (total : ℝ) (added : ℝ) (needed : ℝ) :
  total = 8.5 ∧ added = 2.25 ∧ needed = total - added → needed = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_flour_needed_l1499_149902


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1499_149980

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) :
  arithmetic_sequence a (-2) →
  (a 1 + a 5) / 2 = -1 →
  a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1499_149980


namespace NUMINAMATH_CALUDE_power_ranger_stickers_l1499_149997

theorem power_ranger_stickers (total : ℕ) (first_box : ℕ) : 
  total = 58 → first_box = 23 → (total - first_box) - first_box = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_ranger_stickers_l1499_149997


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_range_of_a_l1499_149969

-- Part I
theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16/3 := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, |x - a| + |2*x - 1| ≥ 2) :
  a ≤ -3/2 ∨ a ≥ 5/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_range_of_a_l1499_149969


namespace NUMINAMATH_CALUDE_tangent_line_sum_range_l1499_149900

theorem tangent_line_sum_range (m n : ℝ) :
  (∀ x y : ℝ, m * x + n * y - 2 = 0 → x^2 + y^2 ≠ 1) ∧
  (∃ x y : ℝ, m * x + n * y - 2 = 0 ∧ x^2 + y^2 = 1) →
  -2 * Real.sqrt 2 ≤ m + n ∧ m + n ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_sum_range_l1499_149900


namespace NUMINAMATH_CALUDE_bridge_length_l1499_149928

theorem bridge_length 
  (left_bank : ℚ) 
  (right_bank : ℚ) 
  (river_width : ℚ) :
  left_bank = 1/4 →
  right_bank = 1/3 →
  river_width = 120 →
  (1 - left_bank - right_bank) * (288 : ℚ) = river_width :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1499_149928


namespace NUMINAMATH_CALUDE_house_height_difference_l1499_149968

/-- Given three houses with heights 80 feet, 70 feet, and 99 feet,
    prove that the difference between the average height and 80 feet is 3 feet. -/
theorem house_height_difference (h₁ h₂ h₃ : ℝ) 
  (h₁_height : h₁ = 80)
  (h₂_height : h₂ = 70)
  (h₃_height : h₃ = 99) :
  (h₁ + h₂ + h₃) / 3 - h₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_house_height_difference_l1499_149968


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1499_149918

theorem inequality_solution_set (x : ℝ) : 
  (abs (x - 1) + abs (x - 2) < 2) ↔ (1/2 < x ∧ x < 5/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1499_149918


namespace NUMINAMATH_CALUDE_min_value_theorem_l1499_149939

theorem min_value_theorem (x y : ℝ) : (x + y)^2 + (x - 2/y)^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1499_149939


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1499_149955

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = a n * q) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1499_149955


namespace NUMINAMATH_CALUDE_price_change_theorem_l1499_149927

theorem price_change_theorem (initial_price : ℝ) (initial_price_positive : 0 < initial_price) :
  let price_after_increase := initial_price * (1 + 0.35)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  (final_price - initial_price) / initial_price = 0.03275 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l1499_149927


namespace NUMINAMATH_CALUDE_unique_solution_system_l1499_149914

theorem unique_solution_system : 
  ∃! (x y : ℝ), (x + y = (7 - x) + (7 - y)) ∧ (x - y = (x - 2) + (y - 2)) ∧ x = 5 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1499_149914


namespace NUMINAMATH_CALUDE_range_of_m_l1499_149987

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b : ℝ, a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) :
  m ≤ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1499_149987


namespace NUMINAMATH_CALUDE_plan_b_more_economical_l1499_149903

/-- Proves that Plan B (fixed money spent) is more economical than Plan A (fixed amount of gasoline) for two refuelings with different prices. -/
theorem plan_b_more_economical (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) :
  (2 * x * y) / (x + y) < (x + y) / 2 := by
  sorry

#check plan_b_more_economical

end NUMINAMATH_CALUDE_plan_b_more_economical_l1499_149903


namespace NUMINAMATH_CALUDE_certain_number_proof_l1499_149973

def w : ℕ := 132

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem certain_number_proof :
  ∃ (n : ℕ), 
    (is_factor (2^5) (n * w)) ∧ 
    (is_factor (3^3) (n * w)) ∧ 
    (is_factor (11^2) (n * w)) ∧
    (∀ (m : ℕ), m < w → ¬(is_factor (2^5) (n * m) ∧ is_factor (3^3) (n * m) ∧ is_factor (11^2) (n * m))) →
  n = 792 :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1499_149973


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1499_149989

theorem sin_alpha_value (α : Real) 
  (h1 : Real.sin (α - π/4) = 7 * Real.sqrt 2 / 10)
  (h2 : Real.cos (2 * α) = 7/25) : 
  Real.sin α = 3/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1499_149989


namespace NUMINAMATH_CALUDE_gcd_divisibility_l1499_149916

theorem gcd_divisibility (p q r s : ℕ+) : 
  (Nat.gcd p.val q.val = 21) →
  (Nat.gcd q.val r.val = 45) →
  (Nat.gcd r.val s.val = 75) →
  (120 < Nat.gcd s.val p.val) →
  (Nat.gcd s.val p.val < 180) →
  9 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_l1499_149916


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1499_149951

theorem binomial_expansion_properties :
  let f := fun x => (2 * x + 1) ^ 4
  ∃ (a b c d e : ℤ),
    f x = a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
    c = 24 ∧
    a + b + c + d + e = 81 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1499_149951


namespace NUMINAMATH_CALUDE_saltwater_solution_l1499_149963

/-- Represents the saltwater tank problem --/
def saltwater_problem (x : ℝ) : Prop :=
  let original_salt := 0.2 * x
  let volume_after_evaporation := 0.75 * x
  let salt_after_addition := original_salt + 14
  let final_volume := salt_after_addition / (1/3)
  let water_added := final_volume - volume_after_evaporation
  (x = 104.99999999999997) ∧ (water_added = 26.25)

/-- Theorem stating the solution to the saltwater problem --/
theorem saltwater_solution :
  ∃ (x : ℝ), saltwater_problem x :=
sorry

end NUMINAMATH_CALUDE_saltwater_solution_l1499_149963


namespace NUMINAMATH_CALUDE_range_of_a_l1499_149942

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, (Set.compl A ∪ B a = U) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1499_149942


namespace NUMINAMATH_CALUDE_problem_statement_l1499_149994

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) :
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1499_149994


namespace NUMINAMATH_CALUDE_arithmetic_progression_five_digit_terms_l1499_149975

/-- 
Given an arithmetic progression with first term a₁ = -1 and common difference d = 19,
this theorem states that the terms consisting only of the digit 5 are given by the formula:
n = (5 * (10^(171k+1) + 35)) / 171, where k is a non-negative integer
-/
theorem arithmetic_progression_five_digit_terms 
  (k : ℕ) : 
  ∃ (n : ℕ), 
    ((-1 : ℤ) + (n - 1) * 19 = 5 * ((10 ^ (171 * k + 1) - 1) / 9)) ∧ 
    (n = (5 * (10 ^ (171 * k + 1) + 35)) / 171) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_five_digit_terms_l1499_149975


namespace NUMINAMATH_CALUDE_solve_problem_l1499_149964

-- Define the sets A and B as functions of m
def A (m : ℤ) : Set ℤ := {-4, 2*m-1, m^2}
def B (m : ℤ) : Set ℤ := {9, m-5, 1-m}

-- Define the universal set U
def U : Set ℤ := Set.univ

-- State the theorem
theorem solve_problem (m : ℤ) 
  (h_intersection : A m ∩ B m = {9}) : 
  m = -3 ∧ A m ∩ (U \ B m) = {-4, -7} := by
  sorry


end NUMINAMATH_CALUDE_solve_problem_l1499_149964


namespace NUMINAMATH_CALUDE_workshop_pairing_probability_l1499_149971

/-- The probability of a specific pairing in a group of participants. -/
def specific_pairing_probability (total_participants : ℕ) : ℚ :=
  if total_participants ≤ 1 then 0
  else 1 / (total_participants - 1 : ℚ)

/-- Theorem: In a workshop with 24 participants, the probability of John pairing with Alice is 1/23. -/
theorem workshop_pairing_probability :
  specific_pairing_probability 24 = 1 / 23 := by
  sorry


end NUMINAMATH_CALUDE_workshop_pairing_probability_l1499_149971


namespace NUMINAMATH_CALUDE_f_neg_two_equals_ten_l1499_149982

/-- Given a function f(x) = x^2 - 3x, prove that f(-2) = 10 -/
theorem f_neg_two_equals_ten (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 3*x) : f (-2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_ten_l1499_149982


namespace NUMINAMATH_CALUDE_ratio_evaluation_l1499_149941

theorem ratio_evaluation : (2^121 * 3^123) / 6^122 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l1499_149941


namespace NUMINAMATH_CALUDE_combined_average_score_l1499_149957

theorem combined_average_score (score1 score2 : ℝ) (ratio1 ratio2 : ℕ) : 
  score1 = 88 →
  score2 = 75 →
  ratio1 = 2 →
  ratio2 = 3 →
  (ratio1 * score1 + ratio2 * score2) / (ratio1 + ratio2) = 80 :=
by sorry

end NUMINAMATH_CALUDE_combined_average_score_l1499_149957


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1499_149922

/-- Given a hyperbola with equation x²/a² - y²/2 = 1 where a > 0 and eccentricity is 2,
    prove that a = √6/3 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  a = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1499_149922


namespace NUMINAMATH_CALUDE_tan_theta_value_l1499_149931

theorem tan_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2)
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2 * θ) = 3) :
  Real.tan θ = 2 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_theta_value_l1499_149931


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1499_149993

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1499_149993


namespace NUMINAMATH_CALUDE_blown_out_dune_probability_l1499_149935

/-- The probability that a sand dune remains after being formed -/
def prob_dune_remains : ℚ := 1 / 3

/-- The probability that a blown-out sand dune contains treasure -/
def prob_treasure : ℚ := 1 / 5

/-- The probability that a formed sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and a lucky coupon -/
def prob_both : ℚ := prob_treasure * prob_lucky_coupon

theorem blown_out_dune_probability : prob_both = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_blown_out_dune_probability_l1499_149935


namespace NUMINAMATH_CALUDE_certain_number_proof_l1499_149947

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 6) 
  (h2 : p - q = 0.3) : 
  3 / q = 15 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1499_149947


namespace NUMINAMATH_CALUDE_initial_bucket_capacity_is_5_l1499_149937

/-- The capacity of the initially filled bucket -/
def initial_bucket_capacity : ℝ := 5

/-- The capacity of the small bucket -/
def small_bucket_capacity : ℝ := 3

/-- The capacity of the large bucket -/
def large_bucket_capacity : ℝ := 6

/-- The amount of additional water the large bucket can hold -/
def additional_capacity : ℝ := 4

theorem initial_bucket_capacity_is_5 :
  initial_bucket_capacity = small_bucket_capacity + (large_bucket_capacity - additional_capacity) :=
by
  sorry

#check initial_bucket_capacity_is_5

end NUMINAMATH_CALUDE_initial_bucket_capacity_is_5_l1499_149937


namespace NUMINAMATH_CALUDE_square_sum_nonzero_iff_exists_nonzero_l1499_149954

theorem square_sum_nonzero_iff_exists_nonzero (a b : ℝ) : 
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_square_sum_nonzero_iff_exists_nonzero_l1499_149954


namespace NUMINAMATH_CALUDE_bus_speed_is_45_l1499_149986

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

def initial_number : TwoDigitNumber := sorry

def one_hour_later : ThreeDigitNumber := sorry

def two_hours_later : ThreeDigitNumber := sorry

/-- The speed of the bus in km/h -/
def bus_speed : Nat := sorry

theorem bus_speed_is_45 :
  (one_hour_later.hundreds = initial_number.ones) ∧
  (one_hour_later.tens = 0) ∧
  (one_hour_later.ones = initial_number.tens) ∧
  (two_hours_later.hundreds = one_hour_later.hundreds) ∧
  (two_hours_later.ones = one_hour_later.ones) ∧
  (two_hours_later.tens ≠ 0) →
  bus_speed = 45 := by sorry

end NUMINAMATH_CALUDE_bus_speed_is_45_l1499_149986


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1499_149970

/-- Represents a two-digit number as a pair of natural numbers -/
def TwoDigitNumber := Nat × Nat

/-- Converts a two-digit number to its decimal representation -/
def toDecimal (n : TwoDigitNumber) : ℚ :=
  (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100

/-- Converts a two-digit number to its repeating decimal representation -/
def toRepeatingDecimal (n : TwoDigitNumber) : ℚ :=
  1 + (n.1 : ℚ) / 10 + (n.2 : ℚ) / 100 + (n.1 : ℚ) / 1000 + (n.2 : ℚ) / 10000

theorem two_digit_number_problem (cd : TwoDigitNumber) :
  72 * toRepeatingDecimal cd - 72 * (1 + toDecimal cd) = 0.8 → cd = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1499_149970


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1499_149948

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1499_149948


namespace NUMINAMATH_CALUDE_same_hours_october_september_l1499_149912

/-- Represents Julie's landscaping business earnings --/
structure LandscapingEarnings where
  mowing_rate : ℕ
  weeding_rate : ℕ
  sept_mowing_hours : ℕ
  sept_weeding_hours : ℕ
  total_earnings : ℕ

/-- Theorem stating that Julie worked the same hours in October as in September --/
theorem same_hours_october_september (j : LandscapingEarnings)
  (h1 : j.mowing_rate = 4)
  (h2 : j.weeding_rate = 8)
  (h3 : j.sept_mowing_hours = 25)
  (h4 : j.sept_weeding_hours = 3)
  (h5 : j.total_earnings = 248) :
  j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours =
  j.total_earnings - (j.mowing_rate * j.sept_mowing_hours + j.weeding_rate * j.sept_weeding_hours) :=
by
  sorry

end NUMINAMATH_CALUDE_same_hours_october_september_l1499_149912


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_equals_one_l1499_149958

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

/-- Given vectors a and b, prove that if they are parallel, then m = 1 -/
theorem parallel_vectors_imply_m_equals_one (m : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (m, m + 1)
  are_parallel a b → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_equals_one_l1499_149958


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l1499_149983

/-- The curve defined by the equation x^2 + y^2 = 2(|x| + |y|) -/
def curve (x y : ℝ) : Prop := x^2 + y^2 = 2 * (abs x + abs y)

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ curve x y}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of the region enclosed by the curve x^2 + y^2 = 2(|x| + |y|) is 2π -/
theorem area_of_enclosed_region : area enclosed_region = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l1499_149983


namespace NUMINAMATH_CALUDE_partners_capital_time_l1499_149950

/-- A proof that under given business conditions, A's capital was used for 15 months -/
theorem partners_capital_time (C P : ℝ) : 
  C > 0 → P > 0 →
  let a_capital := C / 4
  let b_capital := 3 * C / 4
  let b_time := 10
  let b_profit := 2 * P / 3
  let a_profit := P / 3
  ∃ (a_time : ℝ),
    a_time * a_capital / (b_time * b_capital) = a_profit / b_profit ∧
    a_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_partners_capital_time_l1499_149950


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1499_149966

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -300 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1499_149966


namespace NUMINAMATH_CALUDE_olympic_items_problem_l1499_149979

/-- Olympic Commemorative Items Problem -/
theorem olympic_items_problem 
  (total_items : ℕ) 
  (figurine_cost pendant_cost : ℚ) 
  (total_spent : ℚ) 
  (figurine_price pendant_price : ℚ) 
  (min_profit : ℚ) 
  (h1 : total_items = 180)
  (h2 : figurine_cost = 80)
  (h3 : pendant_cost = 50)
  (h4 : total_spent = 11400)
  (h5 : figurine_price = 100)
  (h6 : pendant_price = 60)
  (h7 : min_profit = 2900) :
  ∃ (figurines pendants max_pendants : ℕ),
    figurines + pendants = total_items ∧
    figurine_cost * figurines + pendant_cost * pendants = total_spent ∧
    figurines = 80 ∧
    pendants = 100 ∧
    max_pendants = 70 ∧
    ∀ m : ℕ, m ≤ max_pendants →
      (pendant_price - pendant_cost) * m + 
      (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit :=
by
  sorry


end NUMINAMATH_CALUDE_olympic_items_problem_l1499_149979


namespace NUMINAMATH_CALUDE_intersection_M_N_l1499_149953

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1499_149953


namespace NUMINAMATH_CALUDE_angle_C_is_120_max_area_condition_l1499_149990

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : (2 * a + b) * Real.cos C + c * Real.cos B = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Part 1: Prove that angle C is 120°
theorem angle_C_is_120 : C = 2 * π / 3 := by sorry

-- Part 2: Prove that when c = 4, area is maximized when a = b = (4√3)/3
theorem max_area_condition (h : c = 4) :
  (∀ a' b', a' > 0 → b' > 0 → a' * b' * Real.sin C ≤ a * b * Real.sin C) →
  a = 4 * Real.sqrt 3 / 3 ∧ b = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_max_area_condition_l1499_149990


namespace NUMINAMATH_CALUDE_distance_to_line_rational_l1499_149932

/-- The distance from any lattice point to the line 3x - 4y + 4 = 0 is rational -/
theorem distance_to_line_rational (a b : ℤ) : ∃ (q : ℚ), q = |4 * b - 3 * a - 4| / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_rational_l1499_149932


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l1499_149959

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 2 + 8*I) : Complex.abs z ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l1499_149959


namespace NUMINAMATH_CALUDE_proportionality_statements_l1499_149907

-- Define the basic concepts
def is_direct_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * g x

def is_inverse_proportion (f g : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * g x = k

def is_not_proportional (f g : ℝ → ℝ) : Prop :=
  ¬(is_direct_proportion f g) ∧ ¬(is_inverse_proportion f g)

-- Define the specific relationships
def brick_area (n : ℝ) : ℝ := sorry
def brick_count (n : ℝ) : ℝ := sorry

def walk_speed (t : ℝ) : ℝ := sorry
def walk_time (t : ℝ) : ℝ := sorry

def circle_area (r : ℝ) : ℝ := sorry
def circle_radius (r : ℝ) : ℝ := sorry

-- State the theorem
theorem proportionality_statements :
  (is_direct_proportion brick_area brick_count) ∧
  (is_inverse_proportion walk_speed walk_time) ∧
  (is_not_proportional circle_area circle_radius) :=
by sorry

end NUMINAMATH_CALUDE_proportionality_statements_l1499_149907


namespace NUMINAMATH_CALUDE_right_triangle_area_l1499_149976

theorem right_triangle_area (h : ℝ) (h_pos : h > 0) : ∃ (a b : ℝ),
  a > 0 ∧ b > 0 ∧
  a / b = 3 / 4 ∧
  a^2 + b^2 = h^2 ∧
  (1/2) * a * b = (6/25) * h^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1499_149976


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1499_149909

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1499_149909


namespace NUMINAMATH_CALUDE_unknown_number_solution_l1499_149977

theorem unknown_number_solution : 
  ∃ x : ℝ, (4.7 * 13.26 + 4.7 * x + 4.7 * 77.31 = 470) ∧ (abs (x - 9.43) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_solution_l1499_149977


namespace NUMINAMATH_CALUDE_alice_max_plates_l1499_149995

/-- Represents the shopping problem with pans, pots, and plates. -/
structure Shopping where
  pan_price : ℕ
  pot_price : ℕ
  plate_price : ℕ
  total_budget : ℕ
  min_pans : ℕ
  min_pots : ℕ

/-- Calculates the maximum number of plates that can be bought. -/
def max_plates (s : Shopping) : ℕ :=
  sorry

/-- The shopping problem instance as described in the question. -/
def alice_shopping : Shopping :=
  { pan_price := 3
  , pot_price := 5
  , plate_price := 11
  , total_budget := 100
  , min_pans := 2
  , min_pots := 2
  }

/-- Theorem stating that the maximum number of plates Alice can buy is 7. -/
theorem alice_max_plates :
  max_plates alice_shopping = 7 := by
  sorry

end NUMINAMATH_CALUDE_alice_max_plates_l1499_149995


namespace NUMINAMATH_CALUDE_degree_of_g_l1499_149904

/-- Given polynomials f and g, where h(x) = f(g(x)) + g(x), 
    the degree of h(x) is 6, and the degree of f(x) is 3, 
    then the degree of g(x) is 2. -/
theorem degree_of_g (f g h : Polynomial ℝ) :
  (∀ x, h.eval x = (f.comp g).eval x + g.eval x) →
  h.degree = 6 →
  f.degree = 3 →
  g.degree = 2 := by
sorry

end NUMINAMATH_CALUDE_degree_of_g_l1499_149904


namespace NUMINAMATH_CALUDE_heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l1499_149910

/-- The sum of internal angles of a heptagon, excluding the central point when divided into triangles -/
theorem heptagon_internal_angles_sum : ℝ :=
  let n : ℕ := 7  -- number of vertices in the heptagon
  let polygon_angle_sum : ℝ := (n - 2) * 180
  let central_angle_sum : ℝ := 360
  polygon_angle_sum - central_angle_sum

/-- Proof that the sum of internal angles of a heptagon, excluding the central point, is 540 degrees -/
theorem heptagon_internal_angles_sum_is_540 :
  heptagon_internal_angles_sum = 540 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_internal_angles_sum_heptagon_internal_angles_sum_is_540_l1499_149910


namespace NUMINAMATH_CALUDE_large_planks_nails_l1499_149988

/-- The number of nails needed for large planks in John's house wall construction -/
def nails_for_large_planks (total_nails : ℕ) (nails_for_small_planks : ℕ) : ℕ :=
  total_nails - nails_for_small_planks

/-- Theorem stating that the number of nails for large planks is 15 -/
theorem large_planks_nails :
  nails_for_large_planks 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_large_planks_nails_l1499_149988


namespace NUMINAMATH_CALUDE_graduating_students_average_score_l1499_149949

theorem graduating_students_average_score 
  (total_students : ℕ) 
  (overall_average : ℝ) 
  (graduating_students : ℕ) 
  (non_graduating_students : ℕ) 
  (graduating_average : ℝ) 
  (non_graduating_average : ℝ) :
  total_students = 100 →
  overall_average = 100 →
  non_graduating_students = (3 : ℝ) / 2 * graduating_students →
  graduating_average = (3 : ℝ) / 2 * non_graduating_average →
  total_students = graduating_students + non_graduating_students →
  (graduating_students : ℝ) * graduating_average + 
    (non_graduating_students : ℝ) * non_graduating_average = 
    (total_students : ℝ) * overall_average →
  graduating_average = 125 := by
sorry


end NUMINAMATH_CALUDE_graduating_students_average_score_l1499_149949


namespace NUMINAMATH_CALUDE_lesser_fraction_l1499_149952

theorem lesser_fraction (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_sum : x + y = 13/14) (h_product : x * y = 1/5) : 
  min x y = 87/700 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1499_149952
