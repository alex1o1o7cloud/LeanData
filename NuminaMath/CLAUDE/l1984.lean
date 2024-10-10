import Mathlib

namespace jack_euros_calculation_l1984_198435

/-- Calculates the number of euros Jack has given his total currency amounts -/
theorem jack_euros_calculation (pounds : ℕ) (yen : ℕ) (total_yen : ℕ) 
  (h1 : pounds = 42)
  (h2 : yen = 3000)
  (h3 : total_yen = 9400)
  (h4 : ∀ (e : ℕ), e * 2 * 100 + yen + pounds * 100 = total_yen) :
  ∃ (euros : ℕ), euros = 11 ∧ euros * 2 * 100 + yen + pounds * 100 = total_yen :=
by sorry

end jack_euros_calculation_l1984_198435


namespace shaded_area_calculation_l1984_198497

/-- Given a rectangular grid and two unshaded shapes within it, calculate the area of the shaded region. -/
theorem shaded_area_calculation (grid_width grid_height : ℝ)
  (triangle_base triangle_height : ℝ)
  (trapezoid_height trapezoid_top_base trapezoid_bottom_base : ℝ) :
  grid_width = 10 ∧ 
  grid_height = 5 ∧ 
  triangle_base = 3 ∧ 
  triangle_height = 2 ∧ 
  trapezoid_height = 3 ∧ 
  trapezoid_top_base = 3 ∧ 
  trapezoid_bottom_base = 6 →
  grid_width * grid_height - 
  (1/2 * triangle_base * triangle_height) - 
  (1/2 * (trapezoid_top_base + trapezoid_bottom_base) * trapezoid_height) = 33.5 := by
  sorry


end shaded_area_calculation_l1984_198497


namespace rectangle_area_l1984_198412

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the diagonal BD
def diagonal (rect : Rectangle) : ℝ × ℝ := (rect.B.1 - rect.D.1, rect.B.2 - rect.D.2)

-- Define points E and F on the diagonal
structure PerpendicularPoints (rect : Rectangle) :=
  (E F : ℝ × ℝ)

-- Define the perpendicularity condition
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- State the theorem
theorem rectangle_area (rect : Rectangle) (perp : PerpendicularPoints rect) :
  isPerpendicular (rect.A.1 - perp.E.1, rect.A.2 - perp.E.2) (diagonal rect) →
  isPerpendicular (rect.C.1 - perp.F.1, rect.C.2 - perp.F.2) (diagonal rect) →
  (perp.E.1 - rect.B.1)^2 + (perp.E.2 - rect.B.2)^2 = 1 →
  (perp.F.1 - perp.E.1)^2 + (perp.F.2 - perp.E.2)^2 = 4 →
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2) = 4 * Real.sqrt 3 :=
by sorry

end rectangle_area_l1984_198412


namespace volunteer_distribution_l1984_198411

/-- The number of ways to distribute n girls and m boys into two groups,
    where each group must have at least one girl and one boy. -/
def distribution_schemes (n m : ℕ) : ℕ :=
  if n < 2 ∨ m < 2 then 0
  else (Nat.choose n 1 + Nat.choose n 2) * Nat.factorial m

/-- The problem statement -/
theorem volunteer_distribution : distribution_schemes 5 2 = 60 := by
  sorry

end volunteer_distribution_l1984_198411


namespace correct_operation_l1984_198447

theorem correct_operation (a b : ℝ) : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  sorry

end correct_operation_l1984_198447


namespace reciprocal_of_negative_2023_l1984_198464

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end reciprocal_of_negative_2023_l1984_198464


namespace circle_area_ratio_l1984_198448

theorem circle_area_ratio (Q P R : ℝ) (hP : P = 0.5 * Q) (hR : R = 0.75 * Q) :
  (π * (R / 2)^2) / (π * (Q / 2)^2) = 0.140625 := by
  sorry

end circle_area_ratio_l1984_198448


namespace september_march_ratio_is_two_to_one_l1984_198400

/-- Vacation policy and Andrew's work record --/
structure VacationRecord where
  workRatio : ℕ  -- Number of work days required for 1 vacation day
  workDays : ℕ   -- Number of days worked
  marchDays : ℕ  -- Vacation days taken in March
  remainingDays : ℕ  -- Remaining vacation days

/-- Calculate the ratio of September vacation days to March vacation days --/
def septemberToMarchRatio (record : VacationRecord) : ℚ :=
  let totalVacationDays := record.workDays / record.workRatio
  let septemberDays := totalVacationDays - record.remainingDays - record.marchDays
  septemberDays / record.marchDays

/-- Theorem stating the ratio of September to March vacation days is 2:1 --/
theorem september_march_ratio_is_two_to_one 
  (record : VacationRecord)
  (h1 : record.workRatio = 10)
  (h2 : record.workDays = 300)
  (h3 : record.marchDays = 5)
  (h4 : record.remainingDays = 15) :
  septemberToMarchRatio record = 2 := by
  sorry

#eval septemberToMarchRatio ⟨10, 300, 5, 15⟩

end september_march_ratio_is_two_to_one_l1984_198400


namespace max_profit_is_120_l1984_198432

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := L₁ x + L₂ x

/-- Total sales volume constraint -/
def sales_constraint : ℝ := 15

theorem max_profit_is_120 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ sales_constraint ∧
  ∀ y : ℝ, y ≥ 0 ∧ y ≤ sales_constraint →
  total_profit x ≥ total_profit y ∧
  total_profit x = 120 :=
by sorry

end max_profit_is_120_l1984_198432


namespace interest_rate_cut_l1984_198407

theorem interest_rate_cut (x : ℝ) : 
  (2.25 / 100 : ℝ) * (1 - x)^2 = (1.98 / 100 : ℝ) → 
  (∃ (initial_rate final_rate : ℝ), 
    initial_rate = 2.25 / 100 ∧ 
    final_rate = 1.98 / 100 ∧ 
    final_rate = initial_rate * (1 - x)^2) :=
by
  sorry

end interest_rate_cut_l1984_198407


namespace sugar_solution_replacement_l1984_198477

theorem sugar_solution_replacement (initial_sugar_percent : ℝ) 
                                   (final_sugar_percent : ℝ) 
                                   (second_sugar_percent : ℝ) 
                                   (replaced_portion : ℝ) : 
  initial_sugar_percent = 10 →
  final_sugar_percent = 16 →
  second_sugar_percent = 34 →
  (100 - replaced_portion) * initial_sugar_percent / 100 + 
    replaced_portion * second_sugar_percent / 100 = 
    final_sugar_percent →
  replaced_portion = 25 := by
sorry

end sugar_solution_replacement_l1984_198477


namespace S_is_infinite_l1984_198465

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- The set of points satisfying the given conditions -/
def S : Set RationalPoint :=
  {p : RationalPoint | p.x > 0 ∧ p.y > 0 ∧ p.x * p.y ≤ 12}

/-- Theorem stating that the set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end S_is_infinite_l1984_198465


namespace weekly_average_rainfall_l1984_198444

/-- Calculates the daily average rainfall for a week given specific conditions. -/
theorem weekly_average_rainfall : 
  let monday_rain : ℝ := 2 + 1
  let tuesday_rain : ℝ := 2 * monday_rain
  let wednesday_rain : ℝ := 0
  let thursday_rain : ℝ := 1
  let friday_rain : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain
  let total_rainfall : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain
  let days_in_week : ℕ := 7
  total_rainfall / days_in_week = 20 / 7 := by
  sorry

end weekly_average_rainfall_l1984_198444


namespace sqrt_meaningful_range_l1984_198454

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2023) ↔ x ≥ 2023 := by
  sorry

end sqrt_meaningful_range_l1984_198454


namespace shift_theorem_l1984_198480

/-- Given two functions f and g, where f is shifted by φ to obtain g, prove that sinφ = 24/25 -/
theorem shift_theorem (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 3 * Real.sin x + 4 * Real.cos x) →
  (∀ x, g x = 3 * Real.sin x - 4 * Real.cos x) →
  (∀ x, g x = f (x - φ)) →
  Real.sin φ = 24 / 25 := by
sorry

end shift_theorem_l1984_198480


namespace red_crayon_boxes_l1984_198419

/-- The number of boxes of red crayons given the following conditions:
  * 6 boxes of 8 orange crayons each
  * 7 boxes of 5 blue crayons each
  * Each box of red crayons contains 11 crayons
  * Total number of crayons is 94
-/
theorem red_crayon_boxes : ℕ := by
  sorry

#check red_crayon_boxes

end red_crayon_boxes_l1984_198419


namespace root_relation_l1984_198476

theorem root_relation (k : ℤ) : 
  (∃ x₁ x₂ : ℝ, x₁ = x₂ / 3 ∧ 
   4 * x₁^2 - (3*k + 2) * x₁ + (k^2 - 1) = 0 ∧
   4 * x₂^2 - (3*k + 2) * x₂ + (k^2 - 1) = 0) ↔ 
  k = 2 :=
sorry

end root_relation_l1984_198476


namespace polygon_area_l1984_198440

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the polygon
def polygon : List Point := [
  ⟨0, 0⟩, ⟨12, 0⟩, ⟨24, 12⟩, ⟨24, 0⟩, ⟨36, 0⟩,
  ⟨36, 24⟩, ⟨24, 36⟩, ⟨12, 36⟩, ⟨0, 36⟩, ⟨0, 24⟩
]

-- Function to calculate the area of the polygon
def calculateArea (vertices : List Point) : ℤ :=
  sorry

-- Theorem stating that the area of the polygon is 1008 square units
theorem polygon_area : calculateArea polygon = 1008 :=
  sorry

end polygon_area_l1984_198440


namespace scale_division_l1984_198405

/-- Given a scale of length 7 feet and 12 inches divided into 4 equal parts,
    the length of each part is 24 inches. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 :=
by sorry

end scale_division_l1984_198405


namespace new_person_weight_l1984_198498

/-- Given a group of 8 people where one person weighing 55 kg is replaced by a new person,
    and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 55 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 := by
  sorry

end new_person_weight_l1984_198498


namespace derivative_sqrt_at_one_l1984_198433

theorem derivative_sqrt_at_one :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  HasDerivAt f (1/2) 1 := by sorry

end derivative_sqrt_at_one_l1984_198433


namespace tan_alpha_plus_pi_fourth_l1984_198422

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/7)
  (h2 : Real.tan (β - π/4) = -1/3) :
  Real.tan (α + π/4) = 8/9 := by
  sorry

end tan_alpha_plus_pi_fourth_l1984_198422


namespace roberts_extra_chocolates_l1984_198413

/-- Given that Robert ate 12 chocolates and Nickel ate 3 chocolates,
    prove that Robert ate 9 more chocolates than Nickel. -/
theorem roberts_extra_chocolates (robert : Nat) (nickel : Nat)
    (h1 : robert = 12) (h2 : nickel = 3) :
    robert - nickel = 9 := by
  sorry

end roberts_extra_chocolates_l1984_198413


namespace parallelogram_vertex_C_l1984_198426

/-- Represents a parallelogram in the complex plane -/
structure ComplexParallelogram where
  O : ℂ
  A : ℂ
  B : ℂ
  C : ℂ
  is_origin : O = 0
  is_parallelogram : C - O = B - A

/-- The complex number corresponding to vertex C in the given parallelogram -/
def vertex_C (p : ComplexParallelogram) : ℂ := p.B + p.A

/-- Theorem stating that for the given parallelogram, vertex C corresponds to 3+5i -/
theorem parallelogram_vertex_C :
  ∀ (p : ComplexParallelogram),
    p.O = 0 ∧ p.A = 1 - 3*I ∧ p.B = 4 + 2*I →
    vertex_C p = 3 + 5*I := by
  sorry

end parallelogram_vertex_C_l1984_198426


namespace system_solution_ratio_l1984_198499

theorem system_solution_ratio (k x y z : ℝ) : 
  x + k*y + 2*z = 0 →
  2*x + k*y + 3*z = 0 →
  3*x + 5*y + 4*z = 0 →
  x ≠ 0 →
  y ≠ 0 →
  z ≠ 0 →
  x*z / (y^2) = -25 := by sorry

end system_solution_ratio_l1984_198499


namespace optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l1984_198415

/-- Represents the daily profit function for a merchant's goods -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price that maximizes daily profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), x ≠ optimal_price → profit_function x < profit_function optimal_price :=
by sorry

/-- Verifies that the optimal price satisfies the given conditions -/
theorem optimal_price_satisfies_conditions :
  let initial_price : ℝ := 10
  let initial_sales : ℝ := 100
  let cost_per_item : ℝ := 8
  let price_increase : ℝ := optimal_price - initial_price
  let sales_decrease : ℝ := 10 * price_increase
  (initial_sales - sales_decrease) * (optimal_price - cost_per_item) = profit_function optimal_price :=
by sorry

end optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l1984_198415


namespace inequality_solution_l1984_198468

theorem inequality_solution (x m : ℝ) : 
  (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 
  (∀ x, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + m < 0) → 
  m < 9 :=
by sorry

end inequality_solution_l1984_198468


namespace smallest_integer_with_remainders_l1984_198441

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 6 = 3) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m ≥ n) ∧
  (n = 21) := by
sorry

end smallest_integer_with_remainders_l1984_198441


namespace smallest_a_for_two_zeros_l1984_198446

/-- The function f(x) = x^2 - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

/-- The function g(x) = (a-2)*x -/
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

/-- The function F(x) = f(x) - g(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- The theorem stating that 3 is the smallest positive integer value of a 
    for which F(x) has exactly two zeros -/
theorem smallest_a_for_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ F 3 x₁ = 0 ∧ F 3 x₂ = 0 ∧
  ∀ (a : ℕ), a < 3 → ¬∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ F (a : ℝ) y₁ = 0 ∧ F (a : ℝ) y₂ = 0 :=
by sorry

end smallest_a_for_two_zeros_l1984_198446


namespace linear_function_point_value_l1984_198420

theorem linear_function_point_value (m n : ℝ) : 
  n = 3 - 5 * m → 10 * m + 2 * n - 3 = 3 := by sorry

end linear_function_point_value_l1984_198420


namespace intersection_range_length_AB_l1984_198456

-- Define the hyperbola C: x^2 - y^2 = 1
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem for the range of k
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k > -Real.sqrt 2 ∧ k < -1) ∨ 
    (k > -1 ∧ k < 1) ∨ 
    (k > 1 ∧ k < Real.sqrt 2) :=
sorry

-- Define the condition for the midpoint x-coordinate
def midpoint_x_is_sqrt2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem for the length of AB
theorem length_AB (k : ℝ) :
  midpoint_x_is_sqrt2 k → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 :=
sorry

end intersection_range_length_AB_l1984_198456


namespace polynomial_square_decomposition_l1984_198495

theorem polynomial_square_decomposition (P : Polynomial ℝ) 
  (R : Polynomial ℝ) (h : P^2 = R.comp (Polynomial.X^2)) :
  ∃ Q : Polynomial ℝ, P = Q.comp (Polynomial.X^2) ∨ 
    P = Polynomial.X * Q.comp (Polynomial.X^2) := by
  sorry

end polynomial_square_decomposition_l1984_198495


namespace quadratic_inequality_solution_set_l1984_198492

/-- Given that the solution set of ax² + bx + c > 0 is (-1, 3), 
    prove that the solution set of ax² - bx + c > 0 is (-3, 1) -/
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  Set.Ioo (-3 : ℝ) 1 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry


end quadratic_inequality_solution_set_l1984_198492


namespace apple_count_l1984_198436

theorem apple_count (apples oranges : ℕ) : 
  oranges = 20 → 
  (apples : ℚ) / (apples + (oranges - 14 : ℚ)) = 7/10 → 
  apples = 14 := by
sorry

end apple_count_l1984_198436


namespace expand_product_l1984_198445

theorem expand_product (x : ℝ) : 3 * (2 * x - 7) * (x + 9) = 6 * x^2 + 33 * x - 189 := by
  sorry

end expand_product_l1984_198445


namespace acute_angle_between_l1_l2_l1984_198482

/-- The acute angle formed by the intersection of two lines in a 2D plane. -/
def acuteAngleBetweenLines (l1 l2 : ℝ → ℝ → Prop) : ℝ := sorry

/-- Line l1: √3x - y + 1 = 0 -/
def l1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

/-- Line l2: x + 5 = 0 -/
def l2 (x y : ℝ) : Prop := x + 5 = 0

/-- The acute angle formed by the intersection of l1 and l2 is 30° -/
theorem acute_angle_between_l1_l2 : acuteAngleBetweenLines l1 l2 = 30 * Real.pi / 180 := by sorry

end acute_angle_between_l1_l2_l1984_198482


namespace pens_count_l1984_198478

/-- Given a ratio of pens to markers as 2:5 and 25 markers, prove that the number of pens is 10 -/
theorem pens_count (markers : ℕ) (h1 : markers = 25) : 
  (2 : ℚ) / 5 * markers = 10 := by
  sorry

#check pens_count

end pens_count_l1984_198478


namespace first_year_payment_is_20_l1984_198425

/-- Represents the payment structure over four years -/
structure PaymentStructure where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Defines the conditions of the payment structure -/
def valid_payment_structure (p : PaymentStructure) : Prop :=
  p.second_year = p.first_year + 2 ∧
  p.third_year = p.second_year + 3 ∧
  p.fourth_year = p.third_year + 4 ∧
  p.first_year + p.second_year + p.third_year + p.fourth_year = 96

/-- Theorem stating that the first year's payment is 20 rupees -/
theorem first_year_payment_is_20 :
  ∀ (p : PaymentStructure), valid_payment_structure p → p.first_year = 20 := by
  sorry

end first_year_payment_is_20_l1984_198425


namespace bank_savings_exceed_target_l1984_198467

/-- Geometric sequence sum function -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- Starting amount in cents -/
def initial_deposit : ℚ := 2

/-- Daily multiplication factor -/
def daily_factor : ℚ := 2

/-- Target amount in cents -/
def target_amount : ℚ := 400

theorem bank_savings_exceed_target :
  ∀ n : ℕ, n < 8 → geometric_sum initial_deposit daily_factor n < target_amount ∧
  geometric_sum initial_deposit daily_factor 8 ≥ target_amount :=
by sorry

end bank_savings_exceed_target_l1984_198467


namespace production_days_calculation_l1984_198472

/-- Proves that the number of days is 4, given the conditions from the problem -/
theorem production_days_calculation (n : ℕ) : 
  (∀ (average_past : ℝ) (production_today : ℝ) (new_average : ℝ),
    average_past = 50 ∧
    production_today = 90 ∧
    new_average = 58 ∧
    (n : ℝ) * average_past + production_today = (n + 1 : ℝ) * new_average) →
  n = 4 := by
sorry

end production_days_calculation_l1984_198472


namespace n_pow_half_n_eq_eight_l1984_198459

theorem n_pow_half_n_eq_eight (n : ℝ) : n = 2^Real.sqrt 6 → n^(n/2) = 8 := by
  sorry

end n_pow_half_n_eq_eight_l1984_198459


namespace share_ratio_a_to_b_l1984_198484

/-- Proof of the ratio of shares between A and B --/
theorem share_ratio_a_to_b (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 510 →
  a_share = 360 →
  b_share = 90 →
  c_share = 60 →
  b_share = c_share / 4 →
  a_share / b_share = 4 :=
by sorry

end share_ratio_a_to_b_l1984_198484


namespace find_third_number_l1984_198401

def third_number (a b n : ℕ) : Prop :=
  (Nat.gcd a (Nat.gcd b n) = 8) ∧
  (Nat.lcm a (Nat.lcm b n) = 2^4 * 3^2 * 17 * 7)

theorem find_third_number :
  third_number 136 144 7 :=
by sorry

end find_third_number_l1984_198401


namespace cake_change_calculation_l1984_198403

/-- Calculates the change received when buying cake slices -/
theorem cake_change_calculation (single_price double_price single_quantity double_quantity payment : ℕ) :
  single_price = 4 →
  double_price = 7 →
  single_quantity = 7 →
  double_quantity = 5 →
  payment = 100 →
  payment - (single_price * single_quantity + double_price * double_quantity) = 37 := by
  sorry

#check cake_change_calculation

end cake_change_calculation_l1984_198403


namespace pentagon_rectangle_ratio_l1984_198421

/-- Given a regular pentagon and a rectangle with the same perimeter,
    where the rectangle's length is twice its width,
    prove that the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w : ℝ) (h1 : 5 * p = 30) (h2 : 6 * w = 30) : p / w = 6 / 5 := by
  sorry

end pentagon_rectangle_ratio_l1984_198421


namespace ava_mia_difference_l1984_198491

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava > counts.mia ∧
  counts.alice = counts.ava / 2 ∧
  counts.david + counts.mia + counts.ava + counts.alice = 195

/-- The theorem to prove -/
theorem ava_mia_difference (counts : ShellCounts) :
  problem_conditions counts → counts.ava - counts.mia = 20 :=
by
  sorry

end ava_mia_difference_l1984_198491


namespace overtaking_time_l1984_198423

/-- The problem of determining when person B starts walking to overtake person A --/
theorem overtaking_time (speed_A speed_B overtake_time : ℝ) (h1 : speed_A = 5)
  (h2 : speed_B = 5.555555555555555) (h3 : overtake_time = 1.8) :
  let start_time_diff := overtake_time * speed_B / speed_A - overtake_time
  start_time_diff = 0.2 := by sorry

end overtaking_time_l1984_198423


namespace negation_of_universal_proposition_l1984_198471

theorem negation_of_universal_proposition (a : ℝ) :
  (¬ ∀ x > 0, Real.log x = a) ↔ (∃ x > 0, Real.log x ≠ a) := by sorry

end negation_of_universal_proposition_l1984_198471


namespace min_value_theorem_l1984_198416

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (heq : a + 2*b = 2) :
  ∃ (min_val : ℝ), min_val = 4*(1 + Real.sqrt 2) ∧
  ∀ (x : ℝ), x = 2/(a - 1) + a/b → x ≥ min_val :=
sorry

end min_value_theorem_l1984_198416


namespace total_friends_l1984_198449

def friends_in_line (front : ℕ) (back : ℕ) : ℕ :=
  (front - 1) + 1 + (back - 1)

theorem total_friends (seokjin_front : ℕ) (seokjin_back : ℕ) 
  (h1 : seokjin_front = 8) (h2 : seokjin_back = 6) : 
  friends_in_line seokjin_front seokjin_back = 13 := by
  sorry

end total_friends_l1984_198449


namespace arithmetic_computation_l1984_198481

theorem arithmetic_computation : 2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 := by
  sorry

end arithmetic_computation_l1984_198481


namespace triangle_area_product_l1984_198450

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2) * (8/a) * (8/b) = 8 → a * b = 4 := by sorry

end triangle_area_product_l1984_198450


namespace cubic_root_sum_power_l1984_198404

theorem cubic_root_sum_power (p q r t : ℝ) : 
  (p + q + r = 7) → 
  (p * q + q * r + r * p = 8) → 
  (p * q * r = 1) → 
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) → 
  t^4 - 14 * t^2 - 8 * t = -18 := by
sorry

end cubic_root_sum_power_l1984_198404


namespace hash_four_negative_three_l1984_198424

-- Define the # operation
def hash (x y : Int) : Int := x * (y - 1) + x * y

-- Theorem statement
theorem hash_four_negative_three : hash 4 (-3) = -28 := by
  sorry

end hash_four_negative_three_l1984_198424


namespace only_A_in_first_quadrant_l1984_198485

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def point_A : ℝ × ℝ := (3, 2)
def point_B : ℝ × ℝ := (-3, 2)
def point_C : ℝ × ℝ := (3, -2)
def point_D : ℝ × ℝ := (-3, -2)

theorem only_A_in_first_quadrant :
  first_quadrant point_A.1 point_A.2 ∧
  ¬first_quadrant point_B.1 point_B.2 ∧
  ¬first_quadrant point_C.1 point_C.2 ∧
  ¬first_quadrant point_D.1 point_D.2 := by
  sorry

end only_A_in_first_quadrant_l1984_198485


namespace dog_food_cans_per_package_l1984_198474

/-- Given the following conditions:
  * Chad bought 6 packages of cat food, each containing 9 cans
  * Chad bought 2 packages of dog food
  * The total number of cat food cans is 48 more than the total number of dog food cans
  Prove that each package of dog food contains 3 cans -/
theorem dog_food_cans_per_package :
  let cat_packages : ℕ := 6
  let cat_cans_per_package : ℕ := 9
  let dog_packages : ℕ := 2
  let total_cat_cans : ℕ := cat_packages * cat_cans_per_package
  let dog_cans_per_package : ℕ := total_cat_cans / dog_packages - 24
  dog_cans_per_package = 3 := by sorry

end dog_food_cans_per_package_l1984_198474


namespace complex_equation_solution_l1984_198438

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l1984_198438


namespace range_of_2a_plus_3b_l1984_198458

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) : 
  ∃ (x : ℝ), -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 ∧ 
  ∀ (y : ℝ), -9/2 < y ∧ y < 13/2 → ∃ (a' b' : ℝ), 
    -1 < a' + b' ∧ a' + b' < 3 ∧ 
    2 < a' - b' ∧ a' - b' < 4 ∧ 
    2*a' + 3*b' = y :=
sorry

end range_of_2a_plus_3b_l1984_198458


namespace cone_intersection_volume_ratio_l1984_198409

/-- A cone with a circular base -/
structure Cone :=
  (radius : ℝ)
  (height : ℝ)

/-- A plane passing through the vertex of the cone -/
structure IntersectingPlane :=
  (chord_length : ℝ)

/-- The theorem stating the ratio of volumes when a plane intersects a cone -/
theorem cone_intersection_volume_ratio
  (c : Cone)
  (p : IntersectingPlane)
  (h1 : p.chord_length = c.radius) :
  ∃ (v1 v2 : ℝ),
    v1 > 0 ∧ v2 > 0 ∧
    (v1 / v2 = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3)) :=
sorry

end cone_intersection_volume_ratio_l1984_198409


namespace complex_magnitude_proof_l1984_198431

theorem complex_magnitude_proof : Complex.abs (8/7 + 3*I) = Real.sqrt 505 / 7 := by
  sorry

end complex_magnitude_proof_l1984_198431


namespace equation_solution_l1984_198453

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end equation_solution_l1984_198453


namespace probability_not_snow_l1984_198406

theorem probability_not_snow (p : ℚ) (h : p = 2/5) : 1 - p = 3/5 := by
  sorry

end probability_not_snow_l1984_198406


namespace fruit_selection_problem_l1984_198408

theorem fruit_selection_problem (apple_price orange_price : ℚ)
  (initial_avg_price new_avg_price : ℚ) (oranges_removed : ℕ) :
  apple_price = 40 / 100 →
  orange_price = 60 / 100 →
  initial_avg_price = 54 / 100 →
  new_avg_price = 48 / 100 →
  oranges_removed = 5 →
  ∃ (apples oranges : ℕ),
    (apple_price * apples + orange_price * oranges) / (apples + oranges) = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_removed)) / (apples + oranges - oranges_removed) = new_avg_price ∧
    apples + oranges = 10 :=
by sorry

end fruit_selection_problem_l1984_198408


namespace nancy_zoo_pictures_nancy_zoo_pictures_proof_l1984_198463

theorem nancy_zoo_pictures : ℕ → Prop :=
  fun zoo_pictures =>
    let museum_pictures := 8
    let deleted_pictures := 38
    let remaining_pictures := 19
    zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures →
    zoo_pictures = 49

-- Proof
theorem nancy_zoo_pictures_proof : nancy_zoo_pictures 49 := by
  sorry

end nancy_zoo_pictures_nancy_zoo_pictures_proof_l1984_198463


namespace max_value_cos_sin_l1984_198457

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end max_value_cos_sin_l1984_198457


namespace no_integer_solutions_l1984_198429

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 8*m^2 + 17*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end no_integer_solutions_l1984_198429


namespace triangle_centroid_property_l1984_198466

/-- Triangle with centroid -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  h_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance squared between two points -/
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances from a point to triangle vertices -/
def sum_dist_sq (t : Triangle) (M : ℝ × ℝ) : ℝ :=
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C

/-- Theorem statement -/
theorem triangle_centroid_property (t : Triangle) :
  (∀ M : ℝ × ℝ, sum_dist_sq t M ≥ sum_dist_sq t t.G) ∧
  (∀ M : ℝ × ℝ, sum_dist_sq t M = sum_dist_sq t t.G ↔ M = t.G) ∧
  (∀ k : ℝ, k > sum_dist_sq t t.G →
    ∃ r : ℝ, r = Real.sqrt ((k - sum_dist_sq t t.G) / 3) ∧
      {M : ℝ × ℝ | sum_dist_sq t M = k} = {M : ℝ × ℝ | dist_sq M t.G = r^2}) :=
by sorry

end triangle_centroid_property_l1984_198466


namespace subset_implies_m_equals_one_l1984_198414

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}

theorem subset_implies_m_equals_one (m : ℝ) : A m ⊆ B m → m = 1 := by
  sorry

end subset_implies_m_equals_one_l1984_198414


namespace range_of_k_l1984_198461

-- Define the equation
def equation (x k : ℝ) : Prop := |x| / (x - 2) = k * x

-- Define the property of having three distinct real roots
def has_three_distinct_roots (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation x₁ k ∧ equation x₂ k ∧ equation x₃ k

-- Theorem statement
theorem range_of_k (k : ℝ) :
  has_three_distinct_roots k ↔ 0 < k ∧ k < 1/2 :=
sorry

end range_of_k_l1984_198461


namespace solve_equation_l1984_198486

theorem solve_equation : ∃ x : ℝ, (5 - x = 8) ∧ (x = -3) := by sorry

end solve_equation_l1984_198486


namespace systematic_sampling_proof_l1984_198437

theorem systematic_sampling_proof (N n : ℕ) (hN : N = 92) (hn : n = 30) :
  let k := N / n
  (k = 3) ∧ (k - 1 = 2) :=
by
  sorry

end systematic_sampling_proof_l1984_198437


namespace union_of_A_and_complement_of_B_l1984_198428

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {3, 4}

-- Define set B
def B : Set ℕ := {1, 4, 5}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (U \ B) = {2, 3, 4} := by
  sorry

end union_of_A_and_complement_of_B_l1984_198428


namespace polynomial_value_theorem_l1984_198402

theorem polynomial_value_theorem (m n : ℝ) 
  (h1 : 2*m + n + 2 = m + 2*n) 
  (h2 : m - n + 2 ≠ 0) : 
  let x := 3*(m + n + 1)
  (x^2 + 4*x + 6 : ℝ) = 3 := by sorry

end polynomial_value_theorem_l1984_198402


namespace average_score_is_68_l1984_198410

/-- Represents a score and the number of students who received it -/
structure ScoreData where
  score : ℕ
  count : ℕ

/-- Calculates the average score given a list of ScoreData -/
def averageScore (data : List ScoreData) : ℚ :=
  let totalStudents := data.map (·.count) |>.sum
  let weightedSum := data.map (fun sd => sd.score * sd.count) |>.sum
  weightedSum / totalStudents

/-- The given score data from Mrs. Thompson's test -/
def testScores : List ScoreData := [
  ⟨95, 10⟩,
  ⟨85, 15⟩,
  ⟨75, 20⟩,
  ⟨65, 25⟩,
  ⟨55, 15⟩,
  ⟨45, 10⟩,
  ⟨35, 5⟩
]

theorem average_score_is_68 :
  averageScore testScores = 68 := by
  sorry

end average_score_is_68_l1984_198410


namespace least_integer_with_divisibility_pattern_l1984_198417

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern : 
  ∃ (p q : ℕ), 18 ≤ p ∧ p < 25 ∧ consecutive_pair p q ∧
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ p ∧ k ≠ q → is_divisible 659375723440 k) ∧
  ¬(is_divisible 659375723440 p) ∧ ¬(is_divisible 659375723440 q) ∧
  (∀ (n : ℕ), n < 659375723440 → 
    ¬(∃ (r s : ℕ), 18 ≤ r ∧ r < 25 ∧ consecutive_pair r s ∧
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ r ∧ k ≠ s → is_divisible n k) ∧
    ¬(is_divisible n r) ∧ ¬(is_divisible n s))) :=
by sorry

end least_integer_with_divisibility_pattern_l1984_198417


namespace exponential_linear_critical_point_l1984_198496

/-- A function with a positive critical point -/
def has_positive_critical_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (deriv f) x = 0

/-- The main theorem -/
theorem exponential_linear_critical_point (a : ℝ) :
  has_positive_critical_point (fun x => Real.exp x + a * x) → a < -1 := by
  sorry

end exponential_linear_critical_point_l1984_198496


namespace square_plus_reciprocal_square_l1984_198469

theorem square_plus_reciprocal_square (a : ℝ) (h : a + 1/a = Real.sqrt 5) :
  a^2 + 1/a^2 = 3 := by
  sorry

end square_plus_reciprocal_square_l1984_198469


namespace production_scaling_l1984_198439

/-- Given that x men working x hours a day for x days produce x^2 articles,
    prove that z men working z hours a day for z days produce z^3/x articles. -/
theorem production_scaling (x z : ℝ) (hx : x > 0) :
  (x * x * x * x^2 = x^3 * x^2) →
  (z * z * z * (z^3 / x) = z^3 * (z^3 / x)) :=
by sorry

end production_scaling_l1984_198439


namespace quadratic_inequality_empty_solution_set_l1984_198493

/-- The solution set of a quadratic inequality is empty iff the coefficient of x^2 is positive and the discriminant is non-positive -/
theorem quadratic_inequality_empty_solution_set 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 0) ↔ (a > 0 ∧ b^2 - 4*a*c ≤ 0) :=
sorry

end quadratic_inequality_empty_solution_set_l1984_198493


namespace tangent_line_y_intercept_l1984_198494

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 1), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  ∀ l : Line,
    (∃ p1 p2 : ℝ × ℝ,
      isTangent l c1 ∧
      isTangent l c2 ∧
      isInFirstQuadrant p1 ∧
      isInFirstQuadrant p2 ∧
      (p1.1 - 3)^2 + (p1.2 - 1)^2 = 3^2 ∧
      (p2.1 - 7)^2 + p2.2^2 = 2^2) →
    l.yIntercept = 5 := by
  sorry

end tangent_line_y_intercept_l1984_198494


namespace number_of_girls_l1984_198452

/-- Given a group of kids with boys and girls, prove the number of girls. -/
theorem number_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : total = 9 ∧ boys = 6 → girls = 3 := by
  sorry

end number_of_girls_l1984_198452


namespace min_distance_complex_unit_circle_l1984_198489

theorem min_distance_complex_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 4*I) ≥ min_val :=
sorry

end min_distance_complex_unit_circle_l1984_198489


namespace prism_18_edges_has_8_faces_l1984_198488

/-- Represents a prism -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let base_sides := p.edges / 3
  base_sides + 2

/-- Theorem: A prism with 18 edges has 8 faces -/
theorem prism_18_edges_has_8_faces :
  ∀ p : Prism, p.edges = 18 → num_faces p = 8 := by
  sorry

end prism_18_edges_has_8_faces_l1984_198488


namespace ivan_share_increase_l1984_198473

theorem ivan_share_increase (p v s i : ℝ) 
  (h1 : p + v + s + i > 0)
  (h2 : 2*p + v + s + i = 1.3*(p + v + s + i))
  (h3 : p + 2*v + s + i = 1.25*(p + v + s + i))
  (h4 : p + v + 3*s + i = 1.5*(p + v + s + i)) :
  ∃ k : ℝ, k > 6 ∧ k*i > 0.6*(p + v + s + k*i) := by
  sorry

end ivan_share_increase_l1984_198473


namespace piggy_bank_coins_l1984_198479

/-- The number of dimes in a piggy bank containing quarters and dimes -/
def num_dimes : ℕ := sorry

/-- The number of quarters in the piggy bank -/
def num_quarters : ℕ := sorry

/-- The total number of coins in the piggy bank -/
def total_coins : ℕ := 100

/-- The total value of coins in the piggy bank in cents -/
def total_value : ℕ := 1975

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

theorem piggy_bank_coins :
  num_dimes = 35 ∧
  num_quarters + num_dimes = total_coins ∧
  num_quarters * quarter_value + num_dimes * dime_value = total_value :=
sorry

end piggy_bank_coins_l1984_198479


namespace triangle_area_formulas_l1984_198460

theorem triangle_area_formulas (R r : ℝ) (A B C : ℝ) :
  let T := R * r * (Real.sin A + Real.sin B + Real.sin C)
  T = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end triangle_area_formulas_l1984_198460


namespace sequence_properties_l1984_198490

def sequence_term (n : ℕ) : ℕ :=
  3 * (n^2 - n + 1)

theorem sequence_properties : 
  (∃ k, sequence_term k = 48 ∧ sequence_term (k + 1) = 63) ∧ 
  sequence_term 8 = 168 ∧
  sequence_term 2013 = 9120399 := by
  sorry

end sequence_properties_l1984_198490


namespace shaded_area_is_thirty_l1984_198434

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- The theorem to be proved -/
theorem shaded_area_is_thirty 
  (t : IsoscelesRightTriangle) 
  (h_partitions : num_partitions = 25) 
  (h_shaded : num_shaded = 15) : 
  (t.leg_length * t.leg_length / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end shaded_area_is_thirty_l1984_198434


namespace halloween_candy_problem_l1984_198487

theorem halloween_candy_problem (debby_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ) :
  debby_candy = 32 →
  eaten_candy = 35 →
  remaining_candy = 39 →
  ∃ (sister_candy : ℕ), 
    debby_candy + sister_candy = eaten_candy + remaining_candy ∧
    sister_candy = 42 :=
by sorry

end halloween_candy_problem_l1984_198487


namespace equal_interior_angles_decagon_l1984_198442

/-- The measure of an interior angle in a regular decagon -/
def regular_decagon_angle : ℝ := 144

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: In a decagon where all interior angles are equal, each interior angle measures 144° -/
theorem equal_interior_angles_decagon : 
  ∀ (angles : Fin decagon_sides → ℝ), 
    (∀ (i j : Fin decagon_sides), angles i = angles j) →
    (∀ (i : Fin decagon_sides), angles i = regular_decagon_angle) :=
by sorry

end equal_interior_angles_decagon_l1984_198442


namespace set_operations_l1984_198470

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x ≥ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0}) ∧
  (A ∪ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x < 2}) := by
  sorry

end set_operations_l1984_198470


namespace cloak_change_theorem_l1984_198462

/-- Represents the price of an invisibility cloak and the change received in different scenarios -/
structure CloakTransaction where
  silverPaid : ℕ
  goldChange : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silverChangeForGoldPurchase (transaction1 transaction2 : CloakTransaction) (goldPaid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak for 14 gold coins -/
theorem cloak_change_theorem (transaction1 transaction2 : CloakTransaction) 
  (h1 : transaction1.silverPaid = 20 ∧ transaction1.goldChange = 4)
  (h2 : transaction2.silverPaid = 15 ∧ transaction2.goldChange = 1) :
  silverChangeForGoldPurchase transaction1 transaction2 14 = 10 := by
  sorry

end cloak_change_theorem_l1984_198462


namespace f_seven_halves_eq_neg_sqrt_two_l1984_198483

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_halves_eq_neg_sqrt_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_exp : ∀ x ∈ Set.Ioo 0 1, f x = 2^x) :
  f (7/2) = -Real.sqrt 2 := by
  sorry

end f_seven_halves_eq_neg_sqrt_two_l1984_198483


namespace sum_of_ages_l1984_198418

/-- Given the age relationships between Paula, Karl, and Jane at different points in time, 
    prove that the sum of their current ages is 63 years. -/
theorem sum_of_ages (P K J : ℚ) : 
  (P - 7 = 4 * (K - 7)) →  -- 7 years ago, Paula was 4 times as old as Karl
  (J - 7 = (P - 7) / 2) →  -- 7 years ago, Jane was half as old as Paula
  (P + 8 = 2 * (K + 8)) →  -- In 8 years, Paula will be twice as old as Karl
  (J + 8 = K + 5) →        -- In 8 years, Jane will be 3 years younger than Karl
  P + K + J = 63 :=        -- The sum of their current ages is 63
by sorry

end sum_of_ages_l1984_198418


namespace perpendicular_iff_m_eq_half_l1984_198455

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def l1 : Line :=
  { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def l2 (m : ℝ) : Line :=
  { a := m, b := 1, c := 1 }

/-- The theorem stating the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_iff_m_eq_half :
  ∀ m : ℝ, perpendicular l1 (l2 m) ↔ m = 1/2 := by
  sorry

end perpendicular_iff_m_eq_half_l1984_198455


namespace puppies_given_sandy_friend_puppies_l1984_198475

/-- Given the initial number of puppies and the total number of puppies after receiving more,
    calculate the number of puppies Sandy's friend gave her. -/
theorem puppies_given (initial : ℝ) (total : ℕ) : ℝ :=
  total - initial

/-- Prove that the number of puppies Sandy's friend gave her is 4. -/
theorem sandy_friend_puppies : puppies_given 8 12 = 4 := by
  sorry

end puppies_given_sandy_friend_puppies_l1984_198475


namespace clara_alice_pen_ratio_l1984_198427

def alice_pens : ℕ := 60
def alice_age : ℕ := 20
def clara_future_age : ℕ := 61
def years_to_future : ℕ := 5

theorem clara_alice_pen_ratio :
  ∃ (clara_pens : ℕ) (clara_age : ℕ),
    clara_age > alice_age ∧
    clara_age + years_to_future = clara_future_age ∧
    clara_age - alice_age = alice_pens - clara_pens ∧
    clara_pens * 5 = alice_pens * 2 :=
by sorry

end clara_alice_pen_ratio_l1984_198427


namespace min_value_x2_minus_xy_plus_y2_l1984_198430

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end min_value_x2_minus_xy_plus_y2_l1984_198430


namespace R_is_top_right_l1984_198443

/-- Represents a rectangle with integer labels at its corners -/
structure Rectangle where
  a : Int  -- left-top
  b : Int  -- right-top
  c : Int  -- right-bottom
  d : Int  -- left-bottom

/-- The set of four rectangles -/
def rectangles : Finset Rectangle := sorry

/-- P is one of the rectangles -/
def P : Rectangle := ⟨5, 1, 8, 2⟩

/-- Q is one of the rectangles -/
def Q : Rectangle := ⟨2, 8, 10, 4⟩

/-- R is one of the rectangles -/
def R : Rectangle := ⟨4, 5, 1, 7⟩

/-- S is one of the rectangles -/
def S : Rectangle := ⟨8, 3, 7, 5⟩

/-- The rectangles are arranged in a 2x2 matrix -/
def isArranged2x2 (rects : Finset Rectangle) : Prop := sorry

/-- A rectangle is at the top-right position -/
def isTopRight (rect : Rectangle) (rects : Finset Rectangle) : Prop := sorry

/-- Main theorem: R is at the top-right position -/
theorem R_is_top_right : isTopRight R rectangles := by sorry

end R_is_top_right_l1984_198443


namespace third_year_compound_interest_l1984_198451

/-- Calculates compound interest for a given principal, rate, and number of years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

theorem third_year_compound_interest (P : ℝ) (r : ℝ) :
  r = 0.06 →
  compoundInterest P r 2 = 1200 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |compoundInterest P r 3 - 1858.03| < ε :=
sorry

end third_year_compound_interest_l1984_198451
