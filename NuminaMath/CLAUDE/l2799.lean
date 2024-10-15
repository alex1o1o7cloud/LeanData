import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_sum_l2799_279951

-- Define the hyperbola parameters
def center : ℝ × ℝ := (3, -2)
def focus : ℝ × ℝ := (3, 5)
def vertex : ℝ × ℝ := (3, 0)

-- Define h and k from the center
def h : ℝ := center.1
def k : ℝ := center.2

-- Define a as the distance from center to vertex
def a : ℝ := |center.2 - vertex.2|

-- Define c as the distance from center to focus
def c : ℝ := |center.2 - focus.2|

-- Define b using the relationship c^2 = a^2 + b^2
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

-- Theorem statement
theorem hyperbola_sum : h + k + a + b = 3 + 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2799_279951


namespace NUMINAMATH_CALUDE_equation_solution_l2799_279963

theorem equation_solution : 
  ∃! y : ℝ, 7 * (2 * y - 3) + 5 = -3 * (4 - 5 * y) ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2799_279963


namespace NUMINAMATH_CALUDE_new_student_weight_new_student_weight_is_46_l2799_279942

/-- The weight of a new student who replaces an 86 kg student in a group of 8,
    resulting in an average weight decrease of 5 kg. -/
theorem new_student_weight : ℝ :=
  let n : ℕ := 8 -- number of students
  let avg_decrease : ℝ := 5 -- average weight decrease in kg
  let replaced_weight : ℝ := 86 -- weight of the replaced student in kg
  replaced_weight - n * avg_decrease

/-- Proof that the new student's weight is 46 kg -/
theorem new_student_weight_is_46 : new_student_weight = 46 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_new_student_weight_is_46_l2799_279942


namespace NUMINAMATH_CALUDE_chlorine_original_cost_l2799_279900

/-- The original cost of a liter of chlorine -/
def chlorine_cost : ℝ := sorry

/-- The sale price of chlorine as a percentage of its original price -/
def chlorine_sale_percent : ℝ := 0.80

/-- The original price of a box of soap -/
def soap_original_price : ℝ := 16

/-- The sale price of a box of soap -/
def soap_sale_price : ℝ := 12

/-- The number of liters of chlorine bought -/
def chlorine_quantity : ℕ := 3

/-- The number of boxes of soap bought -/
def soap_quantity : ℕ := 5

/-- The total savings when buying chlorine and soap at sale prices -/
def total_savings : ℝ := 26

theorem chlorine_original_cost :
  chlorine_cost = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_chlorine_original_cost_l2799_279900


namespace NUMINAMATH_CALUDE_divisor_problem_l2799_279947

theorem divisor_problem :
  ∃! d : ℕ+, d > 5 ∧
  (∃ x q : ℤ, x = q * d.val + 5) ∧
  (∃ x p : ℤ, 4 * x = p * d.val + 6) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2799_279947


namespace NUMINAMATH_CALUDE_total_money_l2799_279960

theorem total_money (mark : ℚ) (carolyn : ℚ) (david : ℚ)
  (h1 : mark = 5 / 6)
  (h2 : carolyn = 4 / 9)
  (h3 : david = 7 / 12) :
  mark + carolyn + david = 67 / 36 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2799_279960


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achievable_l2799_279964

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  1/a + 3/b ≥ 16 :=
sorry

theorem min_value_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3*b = 1 ∧ 1/a + 3/b < 16 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_min_value_achievable_l2799_279964


namespace NUMINAMATH_CALUDE_inverse_inequality_l2799_279976

theorem inverse_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l2799_279976


namespace NUMINAMATH_CALUDE_all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l2799_279925

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  vertex_angle : ℝ

-- Define a right-angled isosceles triangle
structure RightAngledIsoscelesTriangle extends IsoscelesTriangle where
  is_right_angled : vertex_angle = 90

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.vertex_angle = t2.vertex_angle

-- Theorem 1: All isosceles right-angled triangles are similar
theorem all_right_angled_isosceles_similar (t1 t2 : RightAngledIsoscelesTriangle) :
  are_similar t1.toIsoscelesTriangle t2.toIsoscelesTriangle :=
sorry

-- Theorem 2: Two isosceles triangles with equal vertex angles are similar
theorem isosceles_equal_vertex_angle_similar (t1 t2 : IsoscelesTriangle)
  (h : t1.vertex_angle = t2.vertex_angle) :
  are_similar t1 t2 :=
sorry

end NUMINAMATH_CALUDE_all_right_angled_isosceles_similar_isosceles_equal_vertex_angle_similar_l2799_279925


namespace NUMINAMATH_CALUDE_domain_of_g_l2799_279956

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 4

-- Define the function g
def g (x : ℝ) : ℝ := f x + f (-x)

-- Define the domain of g
def domain_g : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem domain_of_g :
  ∀ x, x ∈ domain_g ↔ (x ∈ domain_f ∧ (-x) ∈ domain_f) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l2799_279956


namespace NUMINAMATH_CALUDE_parabola_properties_l2799_279966

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_c_neg : c < 0
  h_n_ge_3 : ∃ (n : ℝ), n ≥ 3 ∧ a * n^2 + b * n + c = 0
  h_passes_1_1 : a + b + c = 1
  h_passes_m_0 : ∃ (m : ℝ), a * m^2 + b * m + c = 0

/-- Main theorem -/
theorem parabola_properties (p : Parabola) :
  (p.b > 0) ∧
  (4 * p.a * p.c - p.b^2 < 4 * p.a) ∧
  (∀ (t : ℝ), p.a * 2^2 + p.b * 2 + p.c = t → t > 1) ∧
  (∃ (x : ℝ), p.a * x^2 + p.b * x + p.c = x ∧
    (∃ (m : ℝ), p.a * m^2 + p.b * m + p.c = 0 ∧ 0 < m ∧ m ≤ 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2799_279966


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l2799_279946

theorem midpoint_coordinate_product (p1 p2 : ℝ × ℝ) :
  let m := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  p1 = (10, -3) → p2 = (-4, 9) → m.1 * m.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l2799_279946


namespace NUMINAMATH_CALUDE_smallest_change_l2799_279984

def original : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.823456
  else if n = 2 then 0.183456
  else if n = 3 then 0.128456
  else if n = 4 then 0.123856
  else if n = 6 then 0.123458
  else original

theorem smallest_change :
  ∀ n : ℕ, n ≠ 6 → change_digit 6 8 < change_digit n 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_change_l2799_279984


namespace NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_zero_product_l2799_279968

theorem negation_of_implication_for_all (P Q : ℝ → ℝ → Prop) :
  (¬ ∀ a b : ℝ, P a b → Q a b) ↔ (∃ a b : ℝ, P a b ∧ ¬ Q a b) :=
by sorry

theorem negation_of_zero_product :
  (¬ ∀ a b : ℝ, a = 0 → a * b = 0) ↔ (∃ a b : ℝ, a = 0 ∧ a * b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_zero_product_l2799_279968


namespace NUMINAMATH_CALUDE_purple_balls_count_l2799_279932

/-- Represents the number of green balls in the bin -/
def green_balls : ℕ := 5

/-- Represents the win amount for drawing a green ball -/
def green_win : ℚ := 2

/-- Represents the loss amount for drawing a purple ball -/
def purple_loss : ℚ := 2

/-- Represents the expected winnings -/
def expected_win : ℚ := (1 : ℚ) / 2

/-- 
Given a bin with 5 green balls and k purple balls, where k is a positive integer,
and a game where drawing a green ball wins 2 dollars and drawing a purple ball loses 2 dollars,
if the expected amount won is 50 cents, then k must equal 3.
-/
theorem purple_balls_count (k : ℕ+) : 
  (green_balls : ℚ) / (green_balls + k) * green_win + 
  (k : ℚ) / (green_balls + k) * (-purple_loss) = expected_win → 
  k = 3 := by
  sorry


end NUMINAMATH_CALUDE_purple_balls_count_l2799_279932


namespace NUMINAMATH_CALUDE_fraction_sum_l2799_279953

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 2) : (a + b) / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2799_279953


namespace NUMINAMATH_CALUDE_root_expression_value_l2799_279934

theorem root_expression_value (p m n : ℝ) : 
  (m^2 + (p - 2) * m + 1 = 0) → 
  (n^2 + (p - 2) * n + 1 = 0) → 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 := by
sorry

end NUMINAMATH_CALUDE_root_expression_value_l2799_279934


namespace NUMINAMATH_CALUDE_f_at_negative_three_l2799_279905

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem f_at_negative_three : f (-3) = 110 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_three_l2799_279905


namespace NUMINAMATH_CALUDE_complex_calculation_l2799_279936

theorem complex_calculation : (26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l2799_279936


namespace NUMINAMATH_CALUDE_scientific_notation_of_149000000_l2799_279998

theorem scientific_notation_of_149000000 :
  149000000 = 1.49 * (10 : ℝ)^8 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_149000000_l2799_279998


namespace NUMINAMATH_CALUDE_combined_weight_calculation_l2799_279992

/-- The combined weight of Leo and Kendra -/
def combinedWeight (leoWeight kenWeight : ℝ) : ℝ := leoWeight + kenWeight

/-- Leo's weight after gaining 10 pounds -/
def leoWeightGained (leoWeight : ℝ) : ℝ := leoWeight + 10

/-- Condition that Leo's weight after gaining 10 pounds is 50% more than Kendra's weight -/
def weightCondition (leoWeight kenWeight : ℝ) : Prop :=
  leoWeightGained leoWeight = kenWeight * 1.5

theorem combined_weight_calculation (leoWeight kenWeight : ℝ) 
  (h1 : leoWeight = 98) 
  (h2 : weightCondition leoWeight kenWeight) : 
  combinedWeight leoWeight kenWeight = 170 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_calculation_l2799_279992


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2799_279999

theorem solve_linear_equation (x : ℝ) (h : 3*x - 5*x + 7*x = 150) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2799_279999


namespace NUMINAMATH_CALUDE_largest_lcm_with_18_l2799_279921

theorem largest_lcm_with_18 : 
  (Nat.lcm 18 4).max 
    ((Nat.lcm 18 6).max 
      ((Nat.lcm 18 9).max 
        ((Nat.lcm 18 14).max 
          (Nat.lcm 18 18)))) = 126 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_18_l2799_279921


namespace NUMINAMATH_CALUDE_smallest_b_for_composite_l2799_279957

theorem smallest_b_for_composite (b : ℕ+) (h : b = 9) :
  (∀ x : ℤ, ∃ a c : ℤ, a > 1 ∧ c > 1 ∧ x^4 + b^2 = a * c) ∧
  (∀ b' : ℕ+, b' < b → ∃ x : ℤ, ∀ a c : ℤ, (a > 1 ∧ c > 1 → x^4 + b'^2 ≠ a * c)) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_composite_l2799_279957


namespace NUMINAMATH_CALUDE_max_true_statements_l2799_279983

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (trueStatements : Finset (Fin 5)), 
    (∀ i ∈ trueStatements, statements[i] = true) ∧ 
    trueStatements.card ≤ 3 ∧
    ∀ (otherStatements : Finset (Fin 5)), 
      (∀ i ∈ otherStatements, statements[i] = true) →
      otherStatements.card ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l2799_279983


namespace NUMINAMATH_CALUDE_max_value_fraction_l2799_279920

theorem max_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 9 * a^2 + b^2 = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x * y) / (3 * x + y) ≤ (a * b) / (3 * a + b)) →
  (a * b) / (3 * a + b) = Real.sqrt 2 / 12 :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l2799_279920


namespace NUMINAMATH_CALUDE_expense_reduction_equation_l2799_279967

/-- Represents the average monthly reduction rate as a real number between 0 and 1 -/
def reduction_rate : ℝ := sorry

/-- The initial monthly expenses in yuan -/
def initial_expenses : ℝ := 2500

/-- The final monthly expenses after two months in yuan -/
def final_expenses : ℝ := 1600

/-- The number of months over which the reduction occurred -/
def num_months : ℕ := 2

theorem expense_reduction_equation :
  initial_expenses * (1 - reduction_rate) ^ num_months = final_expenses :=
sorry

end NUMINAMATH_CALUDE_expense_reduction_equation_l2799_279967


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2799_279929

theorem algebraic_expression_value :
  let x : ℚ := 4
  let y : ℚ := -1/5
  ((x + 2*y)^2 - y*(x + 4*y) - x^2) / (-2*y) = -6 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2799_279929


namespace NUMINAMATH_CALUDE_points_on_line_l2799_279908

/-- Given three points (8, 10), (0, m), and (-8, 6) on a straight line, prove that m = 8 -/
theorem points_on_line (m : ℝ) : 
  (∀ (t : ℝ), ∃ (s : ℝ), (8 * (1 - t) + 0 * t, 10 * (1 - t) + m * t) = 
    (0 * (1 - s) + (-8) * s, m * (1 - s) + 6 * s)) → 
  m = 8 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l2799_279908


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2799_279962

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 + 10*x = 56 ↔ x = 4 ∨ x = -14) ∧
  (∀ x : ℝ, 4*x^2 + 48 = 32*x ↔ x = 6 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 + 20 = 12*x ↔ x = 10 ∨ x = 2) ∧
  (∀ x : ℝ, 3*x^2 - 36 = 32*x - x^2 ↔ x = 9 ∨ x = -1) ∧
  (∀ x : ℝ, x^2 + 8*x = 20) ∧
  (∀ x : ℝ, 3*x^2 = 12*x + 63) ∧
  (∀ x : ℝ, x^2 + 16 = 8*x) ∧
  (∀ x : ℝ, 6*x^2 + 12*x = 90) ∧
  (∀ x : ℝ, (1/2)*x^2 + x = 7.5) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2799_279962


namespace NUMINAMATH_CALUDE_max_students_planting_trees_l2799_279913

theorem max_students_planting_trees :
  ∀ (a b : ℕ),
  3 * a + 5 * b = 115 →
  ∀ (x y : ℕ),
  3 * x + 5 * y = 115 →
  a + b ≥ x + y →
  a + b = 37 :=
by sorry

end NUMINAMATH_CALUDE_max_students_planting_trees_l2799_279913


namespace NUMINAMATH_CALUDE_oranges_remaining_l2799_279961

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after Michaela and Cassandra have eaten until they are full -/
theorem oranges_remaining : total_oranges - (michaela_oranges + cassandra_oranges) = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_l2799_279961


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2799_279933

theorem smallest_factorization_coefficient : 
  ∃ (c : ℕ), c > 0 ∧ 
  (∃ (r s : ℤ), x^2 + c*x + 2016 = (x + r) * (x + s)) ∧ 
  (∀ (c' : ℕ), 0 < c' ∧ c' < c → 
    ¬∃ (r' s' : ℤ), x^2 + c'*x + 2016 = (x + r') * (x + s')) ∧
  c = 108 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2799_279933


namespace NUMINAMATH_CALUDE_A_power_101_l2799_279965

def A : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 1; 1, 0, 0; 0, 1, 0]

theorem A_power_101 : A ^ 101 = !![0, 1, 0; 0, 0, 1; 1, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_A_power_101_l2799_279965


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2799_279989

theorem quadratic_polynomial_satisfies_conditions :
  ∃ p : ℝ → ℝ,
    (∀ x, p x = x^2 + 1) ∧
    p (-3) = 10 ∧
    p 0 = 1 ∧
    p 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l2799_279989


namespace NUMINAMATH_CALUDE_negative_three_hash_six_l2799_279996

/-- The '#' operation for rational numbers -/
def hash (a b : ℚ) : ℚ := a^2 + a*b - 5

/-- Theorem: (-3)#6 = -14 -/
theorem negative_three_hash_six : hash (-3) 6 = -14 := by sorry

end NUMINAMATH_CALUDE_negative_three_hash_six_l2799_279996


namespace NUMINAMATH_CALUDE_three_tangent_lines_l2799_279952

/-- A line that passes through the point (0, 2) and has only one common point with the parabola y^2 = 8x -/
structure TangentLine where
  -- The slope of the line (None if vertical)
  slope : Option ℝ
  -- Condition that the line passes through (0, 2)
  passes_through_point : True
  -- Condition that the line has only one common point with y^2 = 8x
  single_intersection : True

/-- The number of lines passing through (0, 2) with only one common point with y^2 = 8x -/
def count_tangent_lines : ℕ := sorry

/-- Theorem stating that there are exactly 3 such lines -/
theorem three_tangent_lines : count_tangent_lines = 3 := by sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l2799_279952


namespace NUMINAMATH_CALUDE_fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l2799_279937

/-- Represents the remaining fuel in a car's tank as a function of time. -/
def remaining_fuel (x : ℝ) : ℝ :=
  80 - 10 * x

theorem fuel_consumption_model (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x = 80 - 10 * x :=
by
  sorry

/-- Verifies that the remaining fuel is 80 at time 0. -/
theorem initial_fuel : remaining_fuel 0 = 80 :=
by
  sorry

/-- Proves that the fuel decreases by 10 units for each unit of time. -/
theorem fuel_decrease_rate (x : ℝ) :
  remaining_fuel (x + 1) = remaining_fuel x - 10 :=
by
  sorry

/-- Confirms that the remaining fuel is non-negative for non-negative time. -/
theorem non_negative_fuel (x : ℝ) (hx : x ≥ 0) :
  remaining_fuel x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_model_initial_fuel_fuel_decrease_rate_non_negative_fuel_l2799_279937


namespace NUMINAMATH_CALUDE_state_fair_revenue_l2799_279982

/-- Represents the revenue calculation for a state fair -/
theorem state_fair_revenue
  (ticket_price : ℝ)
  (total_ticket_revenue : ℝ)
  (food_price : ℝ)
  (ride_price : ℝ)
  (souvenir_price : ℝ)
  (game_price : ℝ)
  (h1 : ticket_price = 8)
  (h2 : total_ticket_revenue = 8000)
  (h3 : food_price = 10)
  (h4 : ride_price = 6)
  (h5 : souvenir_price = 18)
  (h6 : game_price = 5) :
  ∃ (total_revenue : ℝ),
    total_revenue = total_ticket_revenue +
      (3/5 * (total_ticket_revenue / ticket_price) * food_price) +
      (1/3 * (total_ticket_revenue / ticket_price) * ride_price) +
      (1/6 * (total_ticket_revenue / ticket_price) * souvenir_price) +
      (1/10 * (total_ticket_revenue / ticket_price) * game_price) ∧
    total_revenue = 19486 := by
  sorry


end NUMINAMATH_CALUDE_state_fair_revenue_l2799_279982


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l2799_279973

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  midline_difference : ℝ
  longer_base : ℝ
  shorter_base : ℝ

/-- The theorem stating the properties of the specific trapezoid -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.midline = 10)
  (h2 : t.midline_difference = 3)
  (h3 : t.midline = (t.longer_base + t.shorter_base) / 2)
  (h4 : t.midline_difference = (t.longer_base - t.shorter_base) / 2) :
  t.longer_base = 13 := by
    sorry


end NUMINAMATH_CALUDE_trapezoid_longer_base_l2799_279973


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2799_279941

theorem divisibility_theorem (n : ℕ) (a : ℝ) (h : n > 0) :
  ∃ k : ℤ, a^(2*n + 1) + (a - 1)^(n + 2) = k * (a^2 - a + 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2799_279941


namespace NUMINAMATH_CALUDE_existence_of_sequence_l2799_279938

/-- Given positive integers a and b where b > a > 1 and a does not divide b,
    as well as a sequence of positive integers b_n such that b_{n+1} ≥ 2b_n for all n,
    there exists a sequence of positive integers a_n satisfying certain conditions. -/
theorem existence_of_sequence (a b : ℕ) (b_seq : ℕ → ℕ) 
  (h_a_pos : a > 0) (h_b_pos : b > 0) (h_b_gt_a : b > a) (h_a_gt_1 : a > 1)
  (h_a_not_div_b : ¬ (b % a = 0))
  (h_b_seq_growth : ∀ n : ℕ, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ, 
    (∀ n : ℕ, (a_seq (n + 1) - a_seq n = a) ∨ (a_seq (n + 1) - a_seq n = b)) ∧
    (∀ m l : ℕ, ∀ n : ℕ, a_seq m + a_seq l ≠ b_seq n) :=
sorry

end NUMINAMATH_CALUDE_existence_of_sequence_l2799_279938


namespace NUMINAMATH_CALUDE_car_B_speed_l2799_279935

/-- Proves that the speed of car B is 90 km/h given the problem conditions -/
theorem car_B_speed (distance : ℝ) (time : ℝ) (speed_ratio : ℝ × ℝ) :
  distance = 88 →
  time = 32 / 60 →
  speed_ratio = (5, 6) →
  ∃ (speed_A speed_B : ℝ),
    speed_A / speed_B = speed_ratio.1 / speed_ratio.2 ∧
    distance = (speed_A + speed_B) * time ∧
    speed_B = 90 := by
  sorry

end NUMINAMATH_CALUDE_car_B_speed_l2799_279935


namespace NUMINAMATH_CALUDE_village_panic_percentage_l2799_279991

theorem village_panic_percentage (original_population : ℕ) 
  (initial_disappearance_rate : ℚ) (final_population : ℕ) :
  original_population = 7200 →
  initial_disappearance_rate = 1/10 →
  final_population = 4860 →
  (1 - (final_population : ℚ) / ((1 - initial_disappearance_rate) * original_population)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_village_panic_percentage_l2799_279991


namespace NUMINAMATH_CALUDE_find_boys_in_first_group_l2799_279990

/-- Represents the daily work done by a single person -/
structure WorkRate :=
  (amount : ℝ)

/-- Represents a group of workers -/
structure WorkGroup :=
  (men : ℕ)
  (boys : ℕ)

/-- Represents the time taken to complete a job -/
def completeJob (g : WorkGroup) (d : ℕ) (m : WorkRate) (b : WorkRate) : ℝ :=
  d * (g.men * m.amount + g.boys * b.amount)

theorem find_boys_in_first_group :
  ∀ (m b : WorkRate) (x : ℕ),
    m.amount = 2 * b.amount →
    completeJob ⟨12, x⟩ 5 m b = completeJob ⟨13, 24⟩ 4 m b →
    x = 16 := by
  sorry

end NUMINAMATH_CALUDE_find_boys_in_first_group_l2799_279990


namespace NUMINAMATH_CALUDE_factor_polynomial_l2799_279958

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 300 * x^7 = 75 * x^3 * (1 - 4 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2799_279958


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l2799_279980

/-- Given a cubic polynomial with real coefficients and three real roots,
    prove the inequality involving the difference between the largest and smallest roots. -/
theorem cubic_root_inequality (a b c : ℝ) (α β γ : ℝ) : 
  let p : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (∀ x, p x = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  α < β →
  β < γ →
  Real.sqrt (a^2 - 3*b) < γ - α ∧ γ - α ≤ 2 * Real.sqrt ((a^2 / 3) - b) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_inequality_l2799_279980


namespace NUMINAMATH_CALUDE_guard_skipped_circles_l2799_279945

def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400
def intended_circles : ℕ := 10
def actual_distance : ℕ := 16000

def perimeter : ℕ := 2 * (warehouse_length + warehouse_width)
def intended_distance : ℕ := intended_circles * perimeter
def skipped_distance : ℕ := intended_distance - actual_distance
def times_skipped : ℕ := skipped_distance / perimeter

theorem guard_skipped_circles :
  times_skipped = 2 := by sorry

end NUMINAMATH_CALUDE_guard_skipped_circles_l2799_279945


namespace NUMINAMATH_CALUDE_scale_model_height_l2799_279948

/-- The scale ratio of the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 1063

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height * scale_ratio

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 43 := by sorry

end NUMINAMATH_CALUDE_scale_model_height_l2799_279948


namespace NUMINAMATH_CALUDE_min_value_expression_l2799_279959

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3 * a * b - 6 * b + a * (1 - a))^2 + (9 * b^2 + 2 * a + 3 * b * (1 - a))^2) / (a^2 + 9 * b^2) ≥ 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    ((3 * a₀ * b₀ - 6 * b₀ + a₀ * (1 - a₀))^2 + (9 * b₀^2 + 2 * a₀ + 3 * b₀ * (1 - a₀))^2) / (a₀^2 + 9 * b₀^2) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2799_279959


namespace NUMINAMATH_CALUDE_joe_lifts_l2799_279977

theorem joe_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift = 400) :
  total_weight - first_lift = first_lift + 100 := by
  sorry

end NUMINAMATH_CALUDE_joe_lifts_l2799_279977


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2799_279917

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_k_for_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p + q = 78 ∧ 
    p * q = k ∧ 
    p^2 - 78*p + k = 0 ∧ 
    q^2 - 78*q + k = 0 ∧
    k = 146 :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l2799_279917


namespace NUMINAMATH_CALUDE_stratified_sampling_group_size_l2799_279994

theorem stratified_sampling_group_size 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (group_a_sample : ℕ) :
  total_population = 200 →
  sample_size = 40 →
  group_a_sample = 16 →
  (total_population - (total_population * group_a_sample / sample_size) : ℕ) = 120 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_size_l2799_279994


namespace NUMINAMATH_CALUDE_f_greater_than_one_f_monotonicity_f_non_negative_iff_l2799_279988

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x^2) - k * x + 2 * k * Real.log x

-- State the theorems to be proved
theorem f_greater_than_one (x : ℝ) (hx : x > 0) : f 0 x > 1 := by sorry

theorem f_monotonicity (x : ℝ) (hx : x > 0) :
  (x > 2 → (∀ y > x, f 1 y > f 1 x)) ∧
  (x < 2 → (∀ y ∈ Set.Ioo 0 x, f 1 y > f 1 x)) := by sorry

theorem f_non_negative_iff (k : ℝ) :
  (∀ x > 0, f k x ≥ 0) ↔ k ≤ Real.exp 1 := by sorry

end

end NUMINAMATH_CALUDE_f_greater_than_one_f_monotonicity_f_non_negative_iff_l2799_279988


namespace NUMINAMATH_CALUDE_correct_num_arrangements_l2799_279954

/-- The number of different arrangements for 7 students in a row,
    where one student must stand in the center and two other students must stand together. -/
def num_arrangements : ℕ := 192

/-- The number of students -/
def total_students : ℕ := 7

/-- The number of students that must stand together (excluding the center student) -/
def students_together : ℕ := 2

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_num_arrangements :
  num_arrangements = 
    2 * (Nat.factorial students_together) * 
    (Nat.choose (total_students - 3) 1) * 
    (Nat.factorial 2) * 
    (Nat.factorial 3) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_num_arrangements_l2799_279954


namespace NUMINAMATH_CALUDE_consecutive_even_sum_representation_l2799_279918

theorem consecutive_even_sum_representation (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ m : ℕ, n * (n - 1)^(k - 1) = n * (2 * m + (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_representation_l2799_279918


namespace NUMINAMATH_CALUDE_equation_solution_l2799_279978

noncomputable def f (x : ℝ) : ℝ := x + Real.arctan x * Real.sqrt (x^2 + 1)

theorem equation_solution :
  ∃! x : ℝ, 2*x + 2 + f x + f (x + 2) = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2799_279978


namespace NUMINAMATH_CALUDE_tuesday_equals_friday_l2799_279910

def total_weekly_time : ℝ := 5
def monday_time : ℝ := 1.5
def wednesday_time : ℝ := 1.5
def friday_time : ℝ := 1

def tuesday_time : ℝ := total_weekly_time - (monday_time + wednesday_time + friday_time)

theorem tuesday_equals_friday : tuesday_time = friday_time := by
  sorry

end NUMINAMATH_CALUDE_tuesday_equals_friday_l2799_279910


namespace NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2799_279940

/-- The number of people moving to Alaska in 5 days -/
def total_people : ℕ := 4000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculate the average number of people moving to Alaska per hour -/
def average_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem average_people_per_hour_rounded : 
  round_to_nearest average_per_hour = 33 := by
  sorry

end NUMINAMATH_CALUDE_average_people_per_hour_rounded_l2799_279940


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l2799_279979

theorem smallest_n_for_exact_tax : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * m) ∧
  (∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * n) ∧
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l2799_279979


namespace NUMINAMATH_CALUDE_parallel_vectors_expression_l2799_279923

noncomputable def θ : ℝ := Real.arctan (3 : ℝ)

theorem parallel_vectors_expression (a b : ℝ × ℝ) :
  a = (3, 1) →
  b = (Real.sin θ, Real.cos θ) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  2 + Real.sin θ * Real.cos θ - Real.cos θ ^ 2 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_expression_l2799_279923


namespace NUMINAMATH_CALUDE_three_cell_corners_count_l2799_279943

theorem three_cell_corners_count (total_cells : ℕ) (x y : ℕ) : 
  total_cells = 22 → 
  3 * x + 4 * y = total_cells → 
  (x = 2 ∨ x = 6) ∧ (y = 4 ∨ y = 1) :=
sorry

end NUMINAMATH_CALUDE_three_cell_corners_count_l2799_279943


namespace NUMINAMATH_CALUDE_expression_evaluation_l2799_279969

theorem expression_evaluation (m : ℝ) (h : m = 2) : 
  (m^2 - 9) / (m^2 - 6*m + 9) / (1 - 2/(m-3)) = -5/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2799_279969


namespace NUMINAMATH_CALUDE_total_goats_l2799_279944

def washington_herd : ℕ := 5000
def paddington_difference : ℕ := 220

theorem total_goats : washington_herd + (washington_herd + paddington_difference) = 10220 := by
  sorry

end NUMINAMATH_CALUDE_total_goats_l2799_279944


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l2799_279939

theorem quadratic_function_minimum (a b c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let f' : ℝ → ℝ := λ x ↦ 2 * a * x + b
  (f' 0 > 0) →
  (∀ x : ℝ, f x ≥ 0) →
  f 1 / f' 0 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l2799_279939


namespace NUMINAMATH_CALUDE_gcd_and_prime_check_l2799_279906

theorem gcd_and_prime_check : 
  (Nat.gcd 7854 15246 = 6) ∧ ¬(Nat.Prime 6) := by sorry

end NUMINAMATH_CALUDE_gcd_and_prime_check_l2799_279906


namespace NUMINAMATH_CALUDE_gcd_b_always_one_l2799_279930

def b (n : ℕ) : ℤ := (8^n - 1) / 7

theorem gcd_b_always_one (n : ℕ) : Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_always_one_l2799_279930


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2799_279915

theorem gcd_lcm_product (a b : ℕ) (ha : a = 150) (hb : b = 90) :
  (Nat.gcd a b) * (Nat.lcm a b) = 13500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2799_279915


namespace NUMINAMATH_CALUDE_system_solution_l2799_279986

def solution_set : Set (ℝ × ℝ) :=
  {(-2/Real.sqrt 5, 1/Real.sqrt 5), (-2/Real.sqrt 5, -1/Real.sqrt 5),
   (2/Real.sqrt 5, -1/Real.sqrt 5), (2/Real.sqrt 5, 1/Real.sqrt 5)}

def satisfies_system (x y : ℝ) : Prop :=
  x^2 + y^2 ≤ 1 ∧
  16*x^4 - 8*x^2*y^2 + y^4 - 40*x^2 - 10*y^2 + 25 = 0

theorem system_solution :
  ∀ x y : ℝ, satisfies_system x y ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2799_279986


namespace NUMINAMATH_CALUDE_line_tangent_to_fixed_circle_l2799_279922

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a line -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- Function to check if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to get the circumcircle of a triangle -/
def circumcircle (t : Triangle) : Circle := sorry

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Function to check if a point is in a half-plane relative to a line -/
def isInHalfPlane (p : Point) (l : Line) : Prop := sorry

/-- Function to get the perpendicular bisector of a line segment -/
def perpendicularBisector (p1 : Point) (p2 : Point) : Line := sorry

/-- Function to get the intersection of two lines -/
def lineIntersection (l1 : Line) (l2 : Line) : Point := sorry

/-- Function to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Main theorem -/
theorem line_tangent_to_fixed_circle 
  (A B C : Point) 
  (h1 : isAcuteAngled (Triangle.mk A B C))
  (h2 : ∀ C', isOnCircle C' (circumcircle (Triangle.mk A B C)) → 
              isInHalfPlane C' (Line.mk A B) → 
              ∃ M N : Point,
                M = lineIntersection (perpendicularBisector B C') (Line.mk A C') ∧
                N = lineIntersection (perpendicularBisector A C') (Line.mk B C') ∧
                ∃ fixedCircle : Circle, isTangent (Line.mk M N) fixedCircle) :
  ∃ fixedCircle : Circle, ∀ C' M N : Point,
    isOnCircle C' (circumcircle (Triangle.mk A B C)) →
    isInHalfPlane C' (Line.mk A B) →
    M = lineIntersection (perpendicularBisector B C') (Line.mk A C') →
    N = lineIntersection (perpendicularBisector A C') (Line.mk B C') →
    isTangent (Line.mk M N) fixedCircle :=
by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_fixed_circle_l2799_279922


namespace NUMINAMATH_CALUDE_track_length_proof_l2799_279909

/-- The length of the circular track in meters -/
def track_length : ℝ := 180

/-- The distance Brenda runs before their first meeting in meters -/
def brenda_first_meeting : ℝ := 120

/-- The additional distance Sally runs after their first meeting before their second meeting in meters -/
def sally_additional : ℝ := 180

theorem track_length_proof :
  ∃ (brenda_speed sally_speed : ℝ),
    brenda_speed > 0 ∧ sally_speed > 0 ∧
    brenda_speed ≠ sally_speed ∧
    (sally_speed * track_length = brenda_speed * (track_length + brenda_first_meeting)) ∧
    (sally_speed * (track_length + sally_additional) = brenda_speed * (2 * track_length + brenda_first_meeting)) :=
sorry

end NUMINAMATH_CALUDE_track_length_proof_l2799_279909


namespace NUMINAMATH_CALUDE_apple_distribution_l2799_279950

theorem apple_distribution (total_apples : ℕ) (num_friends : ℕ) (apples_per_friend : ℕ) : 
  total_apples = 9 → num_friends = 3 → total_apples / num_friends = apples_per_friend → apples_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l2799_279950


namespace NUMINAMATH_CALUDE_aaron_can_lids_l2799_279949

theorem aaron_can_lids (num_boxes : ℕ) (lids_per_box : ℕ) (total_lids : ℕ) :
  num_boxes = 3 →
  lids_per_box = 13 →
  total_lids = 53 →
  total_lids - (num_boxes * lids_per_box) = 14 := by
  sorry

end NUMINAMATH_CALUDE_aaron_can_lids_l2799_279949


namespace NUMINAMATH_CALUDE_f_g_3_equals_28_l2799_279916

-- Define the functions f and g
def g (x : ℝ) : ℝ := x^2 + 1
def f (x : ℝ) : ℝ := 3*x - 2

-- State the theorem
theorem f_g_3_equals_28 : f (g 3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_f_g_3_equals_28_l2799_279916


namespace NUMINAMATH_CALUDE_salary_increase_l2799_279974

theorem salary_increase (S : ℝ) (h1 : S > 0) : 
  0.08 * (S + S * (10 / 100)) = 1.4667 * (0.06 * S) := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l2799_279974


namespace NUMINAMATH_CALUDE_product_equals_243_l2799_279970

theorem product_equals_243 : 
  (1 / 3 : ℚ) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_243_l2799_279970


namespace NUMINAMATH_CALUDE_range_of_m_l2799_279926

def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def is_increasing_on_reals (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (m^2 - m + 1)^x < (m^2 - m + 1)^y

def p (m : ℝ) : Prop := has_two_distinct_real_roots m

def q (m : ℝ) : Prop := is_increasing_on_reals m

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m) →
  ((-2 ≤ m ∧ m < 0) ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2799_279926


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2799_279993

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = 1/4) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2799_279993


namespace NUMINAMATH_CALUDE_max_a_value_l2799_279901

theorem max_a_value (a : ℝ) : (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l2799_279901


namespace NUMINAMATH_CALUDE_rachel_made_18_dollars_l2799_279995

/-- The amount of money Rachel made selling chocolate bars -/
def rachel_money (total_bars : ℕ) (unsold_bars : ℕ) (price_per_bar : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem stating that Rachel made $18 -/
theorem rachel_made_18_dollars :
  rachel_money 13 4 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_rachel_made_18_dollars_l2799_279995


namespace NUMINAMATH_CALUDE_pears_cost_l2799_279931

theorem pears_cost (initial_amount : ℕ) (banana_cost : ℕ) (banana_packs : ℕ) (asparagus_cost : ℕ) (chicken_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 55 →
  banana_cost = 4 →
  banana_packs = 2 →
  asparagus_cost = 6 →
  chicken_cost = 11 →
  remaining_amount = 28 →
  initial_amount - (banana_cost * banana_packs + asparagus_cost + chicken_cost + remaining_amount) = 2 :=
by
  sorry

#check pears_cost

end NUMINAMATH_CALUDE_pears_cost_l2799_279931


namespace NUMINAMATH_CALUDE_equation_solution_l2799_279975

theorem equation_solution : ∃ x : ℚ, (2/7) * (1/8) * x - 4 = 12 ∧ x = 448 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2799_279975


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l2799_279911

theorem smallest_resolvable_debt (pig_value chicken_value : ℕ) 
  (h_pig : pig_value = 250) (h_chicken : chicken_value = 175) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∃ (p c : ℤ), debt = pig_value * p + chicken_value * c) ∧
  (∀ (d : ℕ), d > 0 → d < debt → 
    ¬∃ (p c : ℤ), d = pig_value * p + chicken_value * c) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l2799_279911


namespace NUMINAMATH_CALUDE_ilwoong_drive_files_l2799_279955

theorem ilwoong_drive_files (num_folders : ℕ) (subfolders_per_folder : ℕ) (files_per_subfolder : ℕ) :
  num_folders = 25 →
  subfolders_per_folder = 10 →
  files_per_subfolder = 8 →
  num_folders * subfolders_per_folder * files_per_subfolder = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ilwoong_drive_files_l2799_279955


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt343_l2799_279981

theorem rationalize_denominator_sqrt343 : 
  7 / Real.sqrt 343 = Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt343_l2799_279981


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2799_279924

theorem inequality_equivalence (x : ℝ) : -1/2 * x - 1 < 0 ↔ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2799_279924


namespace NUMINAMATH_CALUDE_jordan_rectangle_length_l2799_279902

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_length : 
  ∀ (carol jordan : Rectangle),
    carol.length = 5 →
    carol.width = 24 →
    jordan.width = 60 →
    area carol = area jordan →
    jordan.length = 2 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_length_l2799_279902


namespace NUMINAMATH_CALUDE_pencils_with_eraser_count_l2799_279987

/-- The number of pencils with an eraser sold in a stationery store -/
def pencils_with_eraser : ℕ := sorry

/-- The price of a pencil with an eraser -/
def price_eraser : ℚ := 8/10

/-- The price of a regular pencil -/
def price_regular : ℚ := 1/2

/-- The price of a short pencil -/
def price_short : ℚ := 4/10

/-- The number of regular pencils sold -/
def regular_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_sold : ℕ := 35

/-- The total revenue from all pencil sales -/
def total_revenue : ℚ := 194

/-- Theorem stating that the number of pencils with an eraser sold is 200 -/
theorem pencils_with_eraser_count : pencils_with_eraser = 200 :=
  by sorry

end NUMINAMATH_CALUDE_pencils_with_eraser_count_l2799_279987


namespace NUMINAMATH_CALUDE_initially_tagged_fish_count_l2799_279914

/-- The number of fish initially caught and tagged in a pond -/
def initially_tagged_fish (total_fish : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) : ℕ :=
  (tagged_in_second * total_fish) / second_catch

/-- Theorem stating that the number of initially tagged fish is 50 -/
theorem initially_tagged_fish_count :
  initially_tagged_fish 250 50 10 = 50 := by
  sorry

#eval initially_tagged_fish 250 50 10

end NUMINAMATH_CALUDE_initially_tagged_fish_count_l2799_279914


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2799_279919

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2799_279919


namespace NUMINAMATH_CALUDE_linear_function_range_l2799_279907

theorem linear_function_range (x y : ℝ) :
  y = -2 * x + 3 →
  y ≤ 6 →
  x ≥ -3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_range_l2799_279907


namespace NUMINAMATH_CALUDE_sqrt_floor_impossibility_l2799_279972

theorem sqrt_floor_impossibility (x : ℝ) (h1 : 100 ≤ x ∧ x ≤ 200) (h2 : ⌊Real.sqrt x⌋ = 14) : 
  ⌊Real.sqrt (50 * x)⌋ ≠ 140 := by
sorry

end NUMINAMATH_CALUDE_sqrt_floor_impossibility_l2799_279972


namespace NUMINAMATH_CALUDE_square_length_CD_l2799_279928

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -3 * x^2 + 2 * x + 5

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the problem statement
theorem square_length_CD (C D : PointOnParabola) : 
  (C.x = -D.x ∧ C.y = -D.y) → (C.x - D.x)^2 + (C.y - D.y)^2 = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_square_length_CD_l2799_279928


namespace NUMINAMATH_CALUDE_crank_slider_motion_l2799_279927

/-- Crank-slider mechanism -/
structure CrankSlider where
  oa : ℝ
  ab : ℝ
  mb : ℝ
  ω : ℝ

/-- Point coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Velocity vector -/
structure Velocity where
  vx : ℝ
  vy : ℝ

/-- Motion equations for point M -/
def motionEquations (cs : CrankSlider) (t : ℝ) : Point :=
  sorry

/-- Trajectory equation for point M -/
def trajectoryEquation (cs : CrankSlider) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Velocity of point M -/
def velocityM (cs : CrankSlider) (t : ℝ) : Velocity :=
  sorry

theorem crank_slider_motion 
  (cs : CrankSlider) 
  (h1 : cs.oa = 90) 
  (h2 : cs.ab = 90) 
  (h3 : cs.mb = cs.ab / 3) 
  (h4 : cs.ω = 10) :
  ∃ (me : ℝ → Point) (te : ℝ → ℝ → Prop) (ve : ℝ → Velocity),
    me = motionEquations cs ∧
    te = trajectoryEquation cs ∧
    ve = velocityM cs :=
  sorry

end NUMINAMATH_CALUDE_crank_slider_motion_l2799_279927


namespace NUMINAMATH_CALUDE_log_expression_equality_l2799_279912

theorem log_expression_equality : 
  Real.sqrt (Real.log 18 / Real.log 4 - Real.log 18 / Real.log 9 + Real.log 9 / Real.log 2) = 
  (3 * Real.log 3 - Real.log 2) / Real.sqrt (2 * Real.log 3 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l2799_279912


namespace NUMINAMATH_CALUDE_probability_even_balls_correct_l2799_279904

def probability_even_balls (n : ℕ) : ℚ :=
  1/2 - 1/(2*(2^n - 1))

theorem probability_even_balls_correct (n : ℕ) :
  probability_even_balls n = 1/2 - 1/(2*(2^n - 1)) :=
sorry

end NUMINAMATH_CALUDE_probability_even_balls_correct_l2799_279904


namespace NUMINAMATH_CALUDE_integer_root_values_l2799_279903

def polynomial (x b : ℤ) : ℤ := x^3 + 6*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-217, -74, -43, -31, -22, -19, 19, 22, 31, 43, 74, 217} :=
by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2799_279903


namespace NUMINAMATH_CALUDE_marblesPerJar_eq_five_l2799_279997

/-- The number of marbles in each jar, given the conditions of the problem -/
def marblesPerJar : ℕ :=
  let numJars : ℕ := 16
  let numPots : ℕ := numJars / 2
  let totalMarbles : ℕ := 200
  let marblesPerPot : ℕ → ℕ := fun x ↦ 3 * x
  (totalMarbles / (numJars + numPots * 3))

theorem marblesPerJar_eq_five : marblesPerJar = 5 := by
  sorry

end NUMINAMATH_CALUDE_marblesPerJar_eq_five_l2799_279997


namespace NUMINAMATH_CALUDE_oscars_bus_ride_l2799_279985

/-- Oscar's bus ride to school problem -/
theorem oscars_bus_ride (charlie_ride : ℝ) (oscar_difference : ℝ) :
  charlie_ride = 0.25 →
  oscar_difference = 0.5 →
  charlie_ride + oscar_difference = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_oscars_bus_ride_l2799_279985


namespace NUMINAMATH_CALUDE_max_value_theorem_l2799_279971

/-- Given a quadratic function y = ax² + x - b where a > 0 and b > 1,
    if the solution set P of y > 0 intersects with Q = {x | -2-t < x < -2+t}
    for all positive t, then the maximum value of 1/a - 1/b is 1/2. -/
theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, (a * x^2 + x - b > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, (a' * x^2 + x - b' > 0 ∧ -2 - t < x ∧ x < -2 + t)) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2799_279971
