import Mathlib

namespace NUMINAMATH_CALUDE_square_of_powers_l1711_171183

theorem square_of_powers (n : ℕ) : 
  (∃ k : ℕ, 2^10 + 2^13 + 2^14 + 3 * 2^n = k^2) ↔ n = 13 ∨ n = 15 := by
sorry

end NUMINAMATH_CALUDE_square_of_powers_l1711_171183


namespace NUMINAMATH_CALUDE_max_value_fraction_l1711_171150

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ a b : ℝ, -3 ≤ a ∧ a ≤ -1 → 1 ≤ b ∧ b ≤ 3 → (a + b) / (a - b) ≤ (x + y) / (x - y)) →
  (x + y) / (x - y) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1711_171150


namespace NUMINAMATH_CALUDE_triangle_4_5_6_l1711_171117

/-- A triangle can be formed from three line segments if the sum of any two sides is greater than the third side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that line segments of lengths 4, 5, and 6 can form a triangle. -/
theorem triangle_4_5_6 : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_4_5_6_l1711_171117


namespace NUMINAMATH_CALUDE_factor_sum_l1711_171196

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1711_171196


namespace NUMINAMATH_CALUDE_complex_number_problem_l1711_171132

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ = 1 + Complex.I * Real.sqrt 3 →
  Complex.abs z₂ = 2 →
  ∃ (r : ℝ), r > 0 ∧ z₁ * z₂ = r →
  z₂ = 1 - Complex.I * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1711_171132


namespace NUMINAMATH_CALUDE_sum_of_y_values_l1711_171154

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ y₁ y₂ : ℝ, (x^2 + x^2*y₁^2 + x^2*y₁^4 = 525 ∧ x + x*y₁ + x*y₁^2 = 35) ∧
             (x^2 + x^2*y₂^2 + x^2*y₂^4 = 525 ∧ x + x*y₂ + x*y₂^2 = 35) ∧
             y₁ + y₂ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l1711_171154


namespace NUMINAMATH_CALUDE_absolute_difference_simplification_l1711_171110

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_simplification_l1711_171110


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1711_171142

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : ∀ x : ℝ, x^2 + b*x + a = 0 ↔ x = a ∨ x = -b) : 
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1711_171142


namespace NUMINAMATH_CALUDE_base6_addition_l1711_171180

/-- Represents a number in base 6 as a list of digits (least significant first) -/
def Base6 := List Nat

/-- Addition of two base 6 numbers -/
def add_base6 (a b : Base6) : Base6 :=
  sorry

/-- Conversion of a natural number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- Conversion of a base 6 number to a natural number -/
def from_base6 (b : Base6) : Nat :=
  sorry

theorem base6_addition :
  add_base6 [2, 3, 5, 4] [6, 4, 3, 5, 2] = [5, 2, 5, 2, 3] :=
sorry

end NUMINAMATH_CALUDE_base6_addition_l1711_171180


namespace NUMINAMATH_CALUDE_horners_rule_polynomial_l1711_171138

theorem horners_rule_polynomial (x : ℝ) : 
  x^3 + 2*x^2 + x - 1 = ((x + 2)*x + 1)*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_horners_rule_polynomial_l1711_171138


namespace NUMINAMATH_CALUDE_money_value_difference_l1711_171194

def euro_to_dollar : ℝ := 1.5
def diana_dollars : ℝ := 600
def etienne_euros : ℝ := 450

theorem money_value_difference : 
  let etienne_dollars := etienne_euros * euro_to_dollar
  let percentage_diff := (diana_dollars - etienne_dollars) / etienne_dollars * 100
  ∀ ε > 0, |percentage_diff + 11.11| < ε :=
sorry

end NUMINAMATH_CALUDE_money_value_difference_l1711_171194


namespace NUMINAMATH_CALUDE_sum_sub_fixed_points_ln_exp_zero_l1711_171174

/-- A real number t is a sub-fixed point of function f if f(t) = -t -/
def IsSubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The sum of sub-fixed points of ln and exp -/
def SumSubFixedPoints : ℝ := sorry

theorem sum_sub_fixed_points_ln_exp_zero : SumSubFixedPoints = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_sub_fixed_points_ln_exp_zero_l1711_171174


namespace NUMINAMATH_CALUDE_second_month_sale_l1711_171139

def sale_first : ℕ := 6435
def sale_third : ℕ := 6855
def sale_fourth : ℕ := 7230
def sale_fifth : ℕ := 6562
def sale_sixth : ℕ := 7391
def average_sale : ℕ := 6900
def num_months : ℕ := 6

theorem second_month_sale :
  sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth +
  (average_sale * num_months - (sale_first + sale_third + sale_fourth + sale_fifth + sale_sixth)) = 
  average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l1711_171139


namespace NUMINAMATH_CALUDE_journey_time_proof_l1711_171112

/-- Proves that given a round trip where the outbound journey is at 60 km/h, 
    the return journey is at 90 km/h, and the total time is 2 hours, 
    the time taken for the outbound journey is 72 minutes. -/
theorem journey_time_proof (distance : ℝ) 
    (h1 : distance / 60 + distance / 90 = 2) : 
    distance / 60 * 60 = 72 := by
  sorry

#check journey_time_proof

end NUMINAMATH_CALUDE_journey_time_proof_l1711_171112


namespace NUMINAMATH_CALUDE_no_students_in_both_l1711_171190

/-- Represents the number of students in different language classes -/
structure LanguageClasses where
  total : ℕ
  onlyFrench : ℕ
  onlySpanish : ℕ
  neither : ℕ

/-- Calculates the number of students taking both French and Spanish -/
def studentsInBoth (classes : LanguageClasses) : ℕ :=
  classes.total - (classes.onlyFrench + classes.onlySpanish + classes.neither)

/-- Theorem: In the given scenario, no students are taking both French and Spanish -/
theorem no_students_in_both (classes : LanguageClasses)
  (h_total : classes.total = 28)
  (h_french : classes.onlyFrench = 5)
  (h_spanish : classes.onlySpanish = 10)
  (h_neither : classes.neither = 13) :
  studentsInBoth classes = 0 := by
  sorry

#eval studentsInBoth { total := 28, onlyFrench := 5, onlySpanish := 10, neither := 13 }

end NUMINAMATH_CALUDE_no_students_in_both_l1711_171190


namespace NUMINAMATH_CALUDE_evaluate_expression_l1711_171198

theorem evaluate_expression : (1 / ((-5^4)^2)) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1711_171198


namespace NUMINAMATH_CALUDE_multiples_of_eight_range_l1711_171131

theorem multiples_of_eight_range (end_num : ℕ) (num_multiples : ℚ) : 
  end_num = 200 →
  num_multiples = 13.5 →
  ∃ (start_num : ℕ), 
    start_num = 84 ∧
    (end_num - start_num) / 8 + 1 = num_multiples ∧
    start_num ≤ end_num ∧
    ∀ n : ℕ, start_num ≤ n ∧ n ≤ end_num → (n - start_num) % 8 = 0 → n ≤ end_num :=
by sorry


end NUMINAMATH_CALUDE_multiples_of_eight_range_l1711_171131


namespace NUMINAMATH_CALUDE_perpendicular_to_same_line_relationships_l1711_171186

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Define the relationships between two lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def intersect (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

def skew (l1 l2 : Line3D) : Prop :=
  -- We assume this definition exists in the library
  sorry

-- Theorem statement
theorem perpendicular_to_same_line_relationships 
  (l1 l2 p : Line3D) (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) :
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_line_relationships_l1711_171186


namespace NUMINAMATH_CALUDE_library_repacking_l1711_171111

theorem library_repacking (initial_boxes : Nat) (books_per_initial_box : Nat) (books_per_new_box : Nat) :
  initial_boxes = 1584 →
  books_per_initial_box = 45 →
  books_per_new_box = 47 →
  (initial_boxes * books_per_initial_box) % books_per_new_box = 28 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l1711_171111


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1711_171135

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x > 3 ∧ y > 3 → x + y > 6) ∧
  (∃ x y : ℝ, x + y > 6 ∧ ¬(x > 3 ∧ y > 3)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1711_171135


namespace NUMINAMATH_CALUDE_statue_cost_l1711_171118

theorem statue_cost (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 25 →
  selling_price = original_cost * (1 + profit_percentage / 100) →
  original_cost = 496 := by
sorry

end NUMINAMATH_CALUDE_statue_cost_l1711_171118


namespace NUMINAMATH_CALUDE_problem_solution_l1711_171172

theorem problem_solution (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1711_171172


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l1711_171101

theorem greatest_integer_with_gcd_eight (n : ℕ) : n < 150 ∧ n.gcd 16 = 8 → n ≤ 136 :=
by sorry

theorem gcd_of_136_and_16 : Nat.gcd 136 16 = 8 :=
by sorry

theorem is_136_less_than_150 : 136 < 150 :=
by sorry

theorem main_result : ∃ (n : ℕ), n < 150 ∧ n.gcd 16 = 8 ∧ 
  ∀ (m : ℕ), m < 150 ∧ m.gcd 16 = 8 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_eight_gcd_of_136_and_16_is_136_less_than_150_main_result_l1711_171101


namespace NUMINAMATH_CALUDE_liz_additional_money_needed_l1711_171133

def original_price : ℝ := 32500
def new_car_price : ℝ := 30000
def sale_percentage : ℝ := 0.8

theorem liz_additional_money_needed :
  new_car_price - sale_percentage * original_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_liz_additional_money_needed_l1711_171133


namespace NUMINAMATH_CALUDE_total_flowers_in_gardens_l1711_171114

/-- Given 10 gardens, each with 544 pots, and 32 flowers per pot,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_in_gardens_l1711_171114


namespace NUMINAMATH_CALUDE_longest_tape_measure_l1711_171130

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 100) 
  (hb : b = 225) 
  (hc : c = 780) : 
  Nat.gcd a (Nat.gcd b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_measure_l1711_171130


namespace NUMINAMATH_CALUDE_min_value_product_l1711_171170

theorem min_value_product (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 6) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l1711_171170


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l1711_171185

theorem max_value_of_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 3 ≤ y' ∧ y' ≤ 5 → (x' + y' + 1) / x' ≤ (x + y + 1) / x) →
  (x + y + 1) / x = -1/5 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l1711_171185


namespace NUMINAMATH_CALUDE_postage_calculation_l1711_171195

/-- Calculates the postage cost for a letter given its weight and rate structure -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  base_rate + additional_rate * (⌈weight - 1⌉ : ℚ)

/-- The postage for a 5.75 ounce letter is $1.00 given the specified rates -/
theorem postage_calculation :
  let weight : ℚ := 5.75
  let base_rate : ℕ := 25
  let additional_rate : ℕ := 15
  calculate_postage weight base_rate additional_rate = 100 := by
sorry

#eval calculate_postage (5.75 : ℚ) 25 15

end NUMINAMATH_CALUDE_postage_calculation_l1711_171195


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l1711_171181

theorem tangent_equation_solution (x : Real) :
  5.30 * Real.tan x * Real.tan (20 * π / 180) +
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) +
  Real.tan (40 * π / 180) * Real.tan x = 1 →
  ∃ k : ℤ, x = (30 + 180 * k) * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l1711_171181


namespace NUMINAMATH_CALUDE_inequality_proof_l1711_171173

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.sqrt (x₁ * x₂) < (x₁ - x₂) / (Real.log x₁ - Real.log x₂) ∧
  (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < (x₁ + x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1711_171173


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1711_171124

def f (x : ℝ) : ℝ := x^2 + 22*x + 105

theorem quadratic_function_properties :
  (∀ x, f x = x^2 + 22*x + 105) ∧
  (∃ a b : ℤ, ∀ x, f x = x^2 + a*x + b) ∧
  (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0) ∧
  (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
    f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0 ∧ f (f r₄) = 0 ∧
    r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) ∧
  (∀ g : ℝ → ℝ, (∃ a b : ℤ, ∀ x, g x = x^2 + a*x + b) →
    (∃ r₁ r₂ r₃ r₄ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄ ∧
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0) →
    (∃ d : ℝ, d ≠ 0 ∧ ∃ r₁ r₂ r₃ r₄ : ℝ,
      g (g r₁) = 0 ∧ g (g r₂) = 0 ∧ g (g r₃) = 0 ∧ g (g r₄) = 0 ∧
      r₂ = r₁ + d ∧ r₃ = r₂ + d ∧ r₄ = r₃ + d) →
    (1 + a + b ≥ 1 + 22 + 105)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1711_171124


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1711_171102

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + 2

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | 2/a ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) (h : a < 0) :
  {x : ℝ | f a x ≥ 0} = solution_set a :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1711_171102


namespace NUMINAMATH_CALUDE_lucas_investment_l1711_171197

theorem lucas_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : alpha_rate = 0.04)
  (h3 : beta_rate = 0.06)
  (h4 : final_amount = 1584.50) :
  ∃ (alpha_investment : ℝ),
    alpha_investment * (1 + alpha_rate) + (total_investment - alpha_investment) * (1 + beta_rate) = final_amount ∧
    alpha_investment = 275 :=
by sorry

end NUMINAMATH_CALUDE_lucas_investment_l1711_171197


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1711_171162

theorem division_remainder_proof (dividend : ℕ) (divisor : ℚ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 16698 →
  divisor = 187.46067415730337 →
  quotient = 89 →
  dividend = (divisor * quotient).floor + remainder →
  remainder = 14 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1711_171162


namespace NUMINAMATH_CALUDE_rosa_pages_called_l1711_171143

/-- The number of pages Rosa called last week -/
def last_week_pages : ℝ := 10.2

/-- The total number of pages Rosa called -/
def total_pages : ℝ := 18.8

/-- The number of pages Rosa called this week -/
def this_week_pages : ℝ := total_pages - last_week_pages

theorem rosa_pages_called : this_week_pages = 8.6 := by
  sorry

end NUMINAMATH_CALUDE_rosa_pages_called_l1711_171143


namespace NUMINAMATH_CALUDE_stella_spent_40_l1711_171104

/-- Represents the inventory and financial data of Stella's antique shop --/
structure AntiqueShop where
  dolls : ℕ
  clocks : ℕ
  glasses : ℕ
  doll_price : ℕ
  clock_price : ℕ
  glass_price : ℕ
  profit : ℕ

/-- Calculates the total revenue from selling all items --/
def total_revenue (shop : AntiqueShop) : ℕ :=
  shop.dolls * shop.doll_price + shop.clocks * shop.clock_price + shop.glasses * shop.glass_price

/-- Theorem stating that Stella spent $40 to buy everything --/
theorem stella_spent_40 (shop : AntiqueShop) 
    (h1 : shop.dolls = 3)
    (h2 : shop.clocks = 2)
    (h3 : shop.glasses = 5)
    (h4 : shop.doll_price = 5)
    (h5 : shop.clock_price = 15)
    (h6 : shop.glass_price = 4)
    (h7 : shop.profit = 25) : 
  total_revenue shop - shop.profit = 40 := by
  sorry

end NUMINAMATH_CALUDE_stella_spent_40_l1711_171104


namespace NUMINAMATH_CALUDE_al_sewing_time_l1711_171156

/-- Represents the time it takes for Al to sew dresses individually -/
def al_time : ℝ := 12

/-- Represents the time it takes for Allison to sew dresses individually -/
def allison_time : ℝ := 9

/-- Represents the time Allison and Al work together -/
def together_time : ℝ := 3

/-- Represents the additional time Allison needs after Al leaves -/
def allison_additional_time : ℝ := 3.75

/-- Theorem stating that Al's individual sewing time is 12 hours -/
theorem al_sewing_time : 
  (together_time * (1 / allison_time + 1 / al_time)) + 
  (allison_additional_time * (1 / allison_time)) = 1 := by
sorry

end NUMINAMATH_CALUDE_al_sewing_time_l1711_171156


namespace NUMINAMATH_CALUDE_train_length_calculation_l1711_171165

/-- Proves that given two trains of equal length running on parallel lines in the same direction,
    with speeds of 45 km/hr and 36 km/hr respectively, if the faster train passes the slower train
    in 36 seconds, then the length of each train is 45 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) 
  (h1 : faster_speed = 45) 
  (h2 : slower_speed = 36) 
  (h3 : passing_time = 36) : 
  let relative_speed := (faster_speed - slower_speed) * (5 / 18)
  let distance_covered := relative_speed * passing_time
  let train_length := distance_covered / 2
  train_length = 45 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1711_171165


namespace NUMINAMATH_CALUDE_chord_equation_l1711_171187

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given circle, point P, prove that the line AB passing through P has the specified equation -/
theorem chord_equation (c : Circle) (p : ℝ × ℝ) : 
  c.center = (2, 0) → 
  c.radius = 4 → 
  p = (3, 1) → 
  ∃ (l : Line), l.a = 1 ∧ l.b = 1 ∧ l.c = -4 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l1711_171187


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1711_171191

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = 64 ∧ d = s * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1711_171191


namespace NUMINAMATH_CALUDE_fuel_mixture_proof_l1711_171103

def tank_capacity : ℝ := 200
def ethanol_percentage_A : ℝ := 0.12
def ethanol_percentage_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem fuel_mixture_proof (x : ℝ) 
  (hx : x ≥ 0 ∧ x ≤ 100) 
  (h_ethanol : ethanol_percentage_A * x + ethanol_percentage_B * (tank_capacity - x) = total_ethanol) :
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_fuel_mixture_proof_l1711_171103


namespace NUMINAMATH_CALUDE_ratio_equality_l1711_171109

theorem ratio_equality (a b : ℝ) (h : 7 * a = 8 * b) : (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1711_171109


namespace NUMINAMATH_CALUDE_rectangle_area_l1711_171127

/-- Given a rectangle with length thrice its breadth and diagonal 26 meters,
    prove that its area is 202.8 square meters. -/
theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let d := 26
  d^2 = l^2 + b^2 → b * l = 202.8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1711_171127


namespace NUMINAMATH_CALUDE_red_beads_in_necklace_l1711_171105

/-- Represents the number of red beads in each group -/
def redBeadsInGroup (n : ℕ) : ℕ := 2 * n

/-- Represents the total number of red beads up to the nth group -/
def totalRedBeads (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the total number of beads (red and white) up to the nth group -/
def totalBeads (n : ℕ) : ℕ := n + totalRedBeads n

theorem red_beads_in_necklace :
  ∃ n : ℕ, totalBeads n ≤ 99 ∧ totalBeads (n + 1) > 99 ∧ totalRedBeads n = 90 := by
  sorry

end NUMINAMATH_CALUDE_red_beads_in_necklace_l1711_171105


namespace NUMINAMATH_CALUDE_table_length_is_77_l1711_171155

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents the placement of a sheet on the table -/
structure SheetPlacement where
  horizontal_offset : ℕ
  vertical_offset : ℕ

/-- Calculates the final dimensions of a table covered by sheets -/
def calculateTableDimensions (sheet_size : Dimensions) (table_width : ℕ) : Dimensions :=
  let total_sheets := table_width - sheet_size.width
  { length := sheet_size.length + total_sheets,
    width := table_width }

theorem table_length_is_77 (sheet_size : Dimensions) (table_width : ℕ) :
  sheet_size.length = 5 →
  sheet_size.width = 8 →
  table_width = 80 →
  (calculateTableDimensions sheet_size table_width).length = 77 := by
  sorry

#eval (calculateTableDimensions { length := 5, width := 8 } 80).length

end NUMINAMATH_CALUDE_table_length_is_77_l1711_171155


namespace NUMINAMATH_CALUDE_mollys_current_age_l1711_171125

/-- Given the ratio of Sandy's age to Molly's age and Sandy's age after 6 years, 
    calculate Molly's current age. -/
theorem mollys_current_age 
  (sandy_age : ℕ) 
  (molly_age : ℕ) 
  (h1 : sandy_age / molly_age = 4 / 3)  -- Ratio of ages
  (h2 : sandy_age + 6 = 30)             -- Sandy's age after 6 years
  : molly_age = 18 :=
by sorry

end NUMINAMATH_CALUDE_mollys_current_age_l1711_171125


namespace NUMINAMATH_CALUDE_sin_RPS_equals_sin_RPQ_l1711_171178

-- Define the angles
variable (RPQ RPS : Real)

-- Define the supplementary angle relationship
axiom supplementary_angles : RPQ + RPS = Real.pi

-- Define the given sine value
axiom sin_RPQ : Real.sin RPQ = 7/25

-- Theorem to prove
theorem sin_RPS_equals_sin_RPQ : Real.sin RPS = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_RPS_equals_sin_RPQ_l1711_171178


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1711_171192

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 3 / 2 + Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ 1 / x₀ + 1 / y₀ = 3 / 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1711_171192


namespace NUMINAMATH_CALUDE_nicholas_crackers_l1711_171122

theorem nicholas_crackers (marcus_crackers : ℕ) (mona_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  nicholas_crackers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l1711_171122


namespace NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_q_l1711_171113

theorem negation_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ (x < -1 ∨ x > 1)) := by sorry

end NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_q_l1711_171113


namespace NUMINAMATH_CALUDE_opposite_greater_implies_negative_l1711_171184

theorem opposite_greater_implies_negative (x : ℝ) : -x > x → x < 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_greater_implies_negative_l1711_171184


namespace NUMINAMATH_CALUDE_stating_tom_initial_investment_l1711_171145

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 45000  -- Jose's investment
  total_profit : ℕ := 72000  -- Total profit after one year
  jose_profit : ℕ := 40000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom was in business
  jose_months : ℕ := 10  -- Months Jose was in business

/-- 
Theorem stating that given the conditions of the profit sharing scenario, 
Tom's initial investment was 30000.
-/
theorem tom_initial_investment (ps : ProfitSharing) : ps.tom_investment = 30000 := by
  sorry

#check tom_initial_investment

end NUMINAMATH_CALUDE_stating_tom_initial_investment_l1711_171145


namespace NUMINAMATH_CALUDE_seeds_per_can_l1711_171159

def total_seeds : ℕ := 54
def num_cans : ℕ := 9

theorem seeds_per_can :
  total_seeds / num_cans = 6 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_can_l1711_171159


namespace NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_seven_l1711_171123

theorem cube_root_of_three_times_two_to_seven (x : ℝ) :
  x = Real.rpow 2 7 + Real.rpow 2 7 + Real.rpow 2 7 →
  Real.rpow x (1/3) = 4 * Real.rpow 6 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_three_times_two_to_seven_l1711_171123


namespace NUMINAMATH_CALUDE_solve_candy_problem_l1711_171171

def candy_problem (kit_kat : ℕ) (nerds : ℕ) (lollipops : ℕ) (baby_ruth : ℕ) (remaining : ℕ) : Prop :=
  let hershey := 3 * kit_kat
  let reese := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reese
  let given_away := total - remaining
  given_away = 5

theorem solve_candy_problem :
  candy_problem 5 8 11 10 49 := by sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l1711_171171


namespace NUMINAMATH_CALUDE_difference_of_squares_l1711_171151

theorem difference_of_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ ¬(∃ k : ℤ, m = 4*k + 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1711_171151


namespace NUMINAMATH_CALUDE_circle_radius_tripled_area_l1711_171116

theorem circle_radius_tripled_area (r : ℝ) : r > 0 →
  (π * (r + 3)^2 = 3 * π * r^2) → r = (3 * (1 + Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tripled_area_l1711_171116


namespace NUMINAMATH_CALUDE_no_matrix_sine_exists_l1711_171128

open Matrix

/-- Definition of matrix sine function -/
noncomputable def matrixSine (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ∑' n, ((-1)^n / (2*n+1).factorial : ℝ) • (A^(2*n+1))

/-- The statement to be proved -/
theorem no_matrix_sine_exists : 
  ¬ ∃ A : Matrix (Fin 2) (Fin 2) ℝ, matrixSine A = ![![1, 1996], ![0, 1]] :=
sorry

end NUMINAMATH_CALUDE_no_matrix_sine_exists_l1711_171128


namespace NUMINAMATH_CALUDE_area_KLMQ_is_ten_l1711_171152

structure Rectangle where
  width : ℝ
  height : ℝ

def area (r : Rectangle) : ℝ := r.width * r.height

theorem area_KLMQ_is_ten (JLMR JKQR : Rectangle) 
  (h1 : JLMR.width = 2)
  (h2 : JKQR.height = 3)
  (h3 : JLMR.height = 8) :
  ∃ KLMQ : Rectangle, area KLMQ = 10 :=
sorry

end NUMINAMATH_CALUDE_area_KLMQ_is_ten_l1711_171152


namespace NUMINAMATH_CALUDE_novel_pages_count_l1711_171193

/-- Represents the number of pages in the novel -/
def total_pages : ℕ := 420

/-- Pages read on the first day -/
def pages_read_day1 (x : ℕ) : ℕ := x / 4 + 10

/-- Pages read on the second day -/
def pages_read_day2 (x : ℕ) : ℕ := (x - pages_read_day1 x) / 3 + 20

/-- Pages read on the third day -/
def pages_read_day3 (x : ℕ) : ℕ := (x - pages_read_day1 x - pages_read_day2 x) / 2 + 40

/-- Pages remaining after the third day -/
def pages_remaining (x : ℕ) : ℕ := x - pages_read_day1 x - pages_read_day2 x - pages_read_day3 x

theorem novel_pages_count : pages_remaining total_pages = 50 := by sorry

end NUMINAMATH_CALUDE_novel_pages_count_l1711_171193


namespace NUMINAMATH_CALUDE_opposite_of_negative_1009_opposite_of_negative_1009_proof_l1711_171166

theorem opposite_of_negative_1009 : Int → Prop :=
  fun x => x + (-1009) = 0 → x = 1009

-- The proof is omitted
theorem opposite_of_negative_1009_proof : opposite_of_negative_1009 1009 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_1009_opposite_of_negative_1009_proof_l1711_171166


namespace NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l1711_171120

/-- The measure of each interior angle in a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ := by
  -- Define a regular hexagon
  let regular_hexagon : Nat := 6

  -- Define the formula for the sum of interior angles of a polygon
  let sum_of_interior_angles (n : Nat) : ℝ := (n - 2) * 180

  -- Calculate the sum of interior angles for a hexagon
  let total_angle_sum : ℝ := sum_of_interior_angles regular_hexagon

  -- Calculate the measure of each interior angle
  let interior_angle : ℝ := total_angle_sum / regular_hexagon

  -- Prove that the interior angle is 120 degrees
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_interior_angle_l1711_171120


namespace NUMINAMATH_CALUDE_odd_function_property_l1711_171157

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Main theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_f_1 : f 1 = 1/2)
  (h_f_shift : ∀ x, f (x + 2) = f x + f 2) :
  f 5 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l1711_171157


namespace NUMINAMATH_CALUDE_seven_story_pagoda_top_lights_verify_total_lights_l1711_171137

/-- Represents a pagoda with a given number of stories and a total number of lights -/
structure Pagoda where
  stories : ℕ
  total_lights : ℕ
  lights_ratio : ℕ

/-- Calculates the number of lights at the top of the pagoda -/
def top_lights (p : Pagoda) : ℕ :=
  p.total_lights / (2^p.stories - 1)

/-- Theorem stating that a 7-story pagoda with 381 total lights and a doubling ratio has 3 lights at the top -/
theorem seven_story_pagoda_top_lights :
  let p := Pagoda.mk 7 381 2
  top_lights p = 3 := by
  sorry

/-- Verifies that the sum of lights across all stories equals the total lights -/
theorem verify_total_lights (p : Pagoda) :
  (top_lights p) * (2^p.stories - 1) = p.total_lights := by
  sorry

end NUMINAMATH_CALUDE_seven_story_pagoda_top_lights_verify_total_lights_l1711_171137


namespace NUMINAMATH_CALUDE_original_mean_calculation_l1711_171160

theorem original_mean_calculation (n : ℕ) (decrease : ℝ) (updated_mean : ℝ) :
  n = 50 →
  decrease = 6 →
  updated_mean = 194 →
  (updated_mean + decrease : ℝ) = 200 := by
  sorry

end NUMINAMATH_CALUDE_original_mean_calculation_l1711_171160


namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1711_171176

-- Define a type for colors
inductive Color
| Red
| Blue
| Green

-- Define a type for the graph
def Graph := Fin 17 → Fin 17 → Color

-- Statement of the theorem
theorem monochromatic_triangle_exists (g : Graph) : 
  ∃ (a b c : Fin 17), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  g a b = g b c ∧ g b c = g a c :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l1711_171176


namespace NUMINAMATH_CALUDE_cereal_eating_time_l1711_171134

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eating_time (rate1 rate2 amount : ℚ) : ℚ :=
  amount / (rate1 + rate2)

/-- Mr. Fat's eating rate in pounds per minute -/
def mr_fat_rate : ℚ := 1 / 20

/-- Mr. Thin's eating rate in pounds per minute -/
def mr_thin_rate : ℚ := 1 / 30

/-- The amount of cereal to be eaten in pounds -/
def cereal_amount : ℚ := 3

theorem cereal_eating_time :
  eating_time mr_fat_rate mr_thin_rate cereal_amount = 36 := by
  sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l1711_171134


namespace NUMINAMATH_CALUDE_logarithm_equality_l1711_171199

theorem logarithm_equality (c d : ℝ) : 
  c = Real.log 400 / Real.log 4 → d = Real.log 20 / Real.log 2 → c = d := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l1711_171199


namespace NUMINAMATH_CALUDE_store_pants_price_l1711_171115

theorem store_pants_price (selling_price : ℝ) (price_difference : ℝ) (store_price : ℝ) : 
  selling_price = 34 →
  price_difference = 8 →
  store_price = selling_price - price_difference →
  store_price = 26 := by
sorry

end NUMINAMATH_CALUDE_store_pants_price_l1711_171115


namespace NUMINAMATH_CALUDE_linear_function_through_point_l1711_171163

theorem linear_function_through_point (k : ℝ) : 
  (∀ x : ℝ, (k * x = k * 3) → (k * x = 1)) → k = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_point_l1711_171163


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1711_171164

theorem triangle_perimeter (a b x : ℝ) : 
  a = 1 → b = 2 → x^2 - 3*x + 2 = 0 → 
  (a + b > x ∧ a + x > b ∧ b + x > a) →
  a + b + x = 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1711_171164


namespace NUMINAMATH_CALUDE_complex_real_condition_l1711_171177

theorem complex_real_condition (m : ℝ) :
  (∃ (x : ℝ), Complex.mk (m - 1) (m + 1) = x) ↔ m = -1 := by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l1711_171177


namespace NUMINAMATH_CALUDE_pizza_cost_is_twelve_l1711_171169

/-- Calculates the cost of each pizza given the number of people, people per pizza, 
    earnings per night, and number of nights worked. -/
def pizza_cost (total_people : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) (nights_worked : ℕ) : ℚ :=
  let total_pizzas := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_earnings := earnings_per_night * nights_worked
  total_earnings / total_pizzas

/-- Proves that the cost of each pizza is $12 under the given conditions. -/
theorem pizza_cost_is_twelve :
  pizza_cost 15 3 4 15 = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_is_twelve_l1711_171169


namespace NUMINAMATH_CALUDE_difference_of_squares_81_49_l1711_171141

theorem difference_of_squares_81_49 : 81^2 - 49^2 = 4160 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_81_49_l1711_171141


namespace NUMINAMATH_CALUDE_percent_democrats_voters_l1711_171121

theorem percent_democrats_voters (d r : ℝ) : 
  d + r = 100 →
  0.75 * d + 0.2 * r = 53 →
  d = 60 :=
by sorry

end NUMINAMATH_CALUDE_percent_democrats_voters_l1711_171121


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1711_171167

theorem no_integer_solutions : 
  ¬∃ (y z : ℤ), (2*y^2 - 2*y*z - z^2 = 15) ∧ 
                (6*y*z + 2*z^2 = 60) ∧ 
                (y^2 + 8*z^2 = 90) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1711_171167


namespace NUMINAMATH_CALUDE_problem_29_AHSME_1978_l1711_171146

theorem problem_29_AHSME_1978 (a b c x : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : (a + b - c) / c = (a - b + c) / b)
  (h2 : (a + b - c) / c = (-a + b + c) / a)
  (h3 : x = ((a + b) * (b + c) * (c + a)) / (a * b * c))
  (h4 : x < 0) :
  x = -1 := by sorry

end NUMINAMATH_CALUDE_problem_29_AHSME_1978_l1711_171146


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l1711_171168

theorem students_liking_both_desserts 
  (total : Nat) 
  (like_apple : Nat) 
  (like_chocolate : Nat) 
  (like_neither : Nat) 
  (h1 : total = 40)
  (h2 : like_apple = 18)
  (h3 : like_chocolate = 15)
  (h4 : like_neither = 12) :
  like_apple + like_chocolate - (total - like_neither) = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l1711_171168


namespace NUMINAMATH_CALUDE_initials_probability_l1711_171182

/-- The number of students in the class -/
def class_size : ℕ := 30

/-- The number of consonants in the alphabet (excluding Y) -/
def num_consonants : ℕ := 21

/-- The number of consonants we're interested in (B, C, D) -/
def target_consonants : ℕ := 3

/-- The probability of selecting a student with initials starting with B, C, or D -/
def probability : ℚ := 1 / 21

theorem initials_probability :
  probability = (min class_size (target_consonants * (num_consonants - 1))) / (class_size * num_consonants) :=
sorry

end NUMINAMATH_CALUDE_initials_probability_l1711_171182


namespace NUMINAMATH_CALUDE_smallest_root_between_3_and_4_l1711_171119

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x + 5 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_root_between_3_and_4 :
  ∃ s, is_smallest_positive_root s ∧ 3 ≤ s ∧ s < 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_between_3_and_4_l1711_171119


namespace NUMINAMATH_CALUDE_equal_distances_exist_l1711_171189

/-- Represents a position on an 8x8 grid -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Calculates the squared Euclidean distance between two positions -/
def squaredDistance (p1 p2 : Position) : ℕ :=
  (p1.row - p2.row).val ^ 2 + (p1.col - p2.col).val ^ 2

/-- Represents a configuration of 8 rooks on a chessboard -/
structure RookConfiguration where
  positions : Fin 8 → Position
  no_attack : ∀ i j, i ≠ j → positions i ≠ positions j

theorem equal_distances_exist (config : RookConfiguration) :
  ∃ i j k l : Fin 8, i < j ∧ k < l ∧ (i, j) ≠ (k, l) ∧
    squaredDistance (config.positions i) (config.positions j) =
    squaredDistance (config.positions k) (config.positions l) :=
sorry

end NUMINAMATH_CALUDE_equal_distances_exist_l1711_171189


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1711_171106

theorem solve_exponential_equation :
  ∃ w : ℝ, (2 : ℝ)^(2*w) = 8^(w-4) → w = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1711_171106


namespace NUMINAMATH_CALUDE_product_remainder_l1711_171108

theorem product_remainder (a b c : ℕ) (ha : a % 10 = 3) (hb : b % 10 = 2) (hc : c % 10 = 4) :
  (a * b * c) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1711_171108


namespace NUMINAMATH_CALUDE_sams_book_count_l1711_171126

theorem sams_book_count :
  let used_adventure_books : ℝ := 13.0
  let used_mystery_books : ℝ := 17.0
  let new_crime_books : ℝ := 15.0
  let total_books := used_adventure_books + used_mystery_books + new_crime_books
  total_books = 45.0 := by sorry

end NUMINAMATH_CALUDE_sams_book_count_l1711_171126


namespace NUMINAMATH_CALUDE_volleyball_count_l1711_171136

theorem volleyball_count (total : ℕ) (soccer : ℕ) (basketball : ℕ) (tennis : ℕ) (baseball : ℕ) (hockey : ℕ) (volleyball : ℕ) :
  total = 180 →
  soccer = 20 →
  basketball = soccer + 5 →
  tennis = 2 * soccer →
  baseball = soccer + 10 →
  hockey = tennis / 2 →
  volleyball = total - (soccer + basketball + tennis + baseball + hockey) →
  volleyball = 45 := by
sorry

end NUMINAMATH_CALUDE_volleyball_count_l1711_171136


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1711_171161

theorem triangle_angle_theorem (a b c : ℝ) : 
  (a = 2 * b) →                 -- One angle is twice the second angle
  (c = b + 30) →                -- The third angle is 30° more than the second angle
  (a + b + c = 180) →           -- Sum of angles in a triangle is 180°
  (a = 75 ∧ b = 37.5 ∧ c = 67.5) -- The measures of the angles are 75°, 37.5°, and 67.5°
  := by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1711_171161


namespace NUMINAMATH_CALUDE_expression_change_l1711_171144

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ t ↦ t^2 - 3
  (f (x + a) - f x = 2*a*x + a^2) ∧ (f (x - a) - f x = -2*a*x + a^2) :=
sorry

end NUMINAMATH_CALUDE_expression_change_l1711_171144


namespace NUMINAMATH_CALUDE_gcd_5616_11609_l1711_171147

theorem gcd_5616_11609 : Nat.gcd 5616 11609 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5616_11609_l1711_171147


namespace NUMINAMATH_CALUDE_park_rose_bushes_l1711_171140

/-- Calculate the final number of rose bushes in the park -/
def final_rose_bushes (initial : ℕ) (planned : ℕ) (rate : ℕ) (removed : ℕ) : ℕ :=
  initial + planned * rate - removed

/-- Theorem stating the final number of rose bushes in the park -/
theorem park_rose_bushes : final_rose_bushes 2 4 3 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_park_rose_bushes_l1711_171140


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l1711_171107

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2010WithSumOfDigits10 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2010_with_sum_of_digits_10 :
  isFirstYearAfter2010WithSumOfDigits10 2035 := by sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_sum_of_digits_10_l1711_171107


namespace NUMINAMATH_CALUDE_solve_for_B_l1711_171129

theorem solve_for_B : ∀ (A B : ℕ), 
  (A ≥ 1 ∧ A ≤ 9) →  -- Ensure A is a single digit
  (B ≥ 0 ∧ B ≤ 9) →  -- Ensure B is a single digit
  632 - (100 * A + 10 * B + 1) = 41 → 
  B = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_for_B_l1711_171129


namespace NUMINAMATH_CALUDE_short_trees_planted_count_l1711_171153

/-- The number of short trees planted in the park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is 64 -/
theorem short_trees_planted_count : short_trees_planted 31 95 = 64 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_planted_count_l1711_171153


namespace NUMINAMATH_CALUDE_age_difference_l1711_171148

theorem age_difference (x y z : ℕ) : 
  x + y = y + z + 18 → (x - z : ℚ) / 10 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1711_171148


namespace NUMINAMATH_CALUDE_convergence_and_bound_l1711_171149

def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 3 * u k - 3 * (u k)^2

theorem convergence_and_bound :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1/2| < ε) ∧
  (∀ k < 9, |u k - 1/2| > 1/2^500) ∧
  |u 9 - 1/2| ≤ 1/2^500 :=
sorry

end NUMINAMATH_CALUDE_convergence_and_bound_l1711_171149


namespace NUMINAMATH_CALUDE_weighted_sum_inequality_l1711_171188

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_cond : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_weighted_sum_inequality_l1711_171188


namespace NUMINAMATH_CALUDE_base_7_divisibility_l1711_171100

def is_base_7_digit (x : ℕ) : Prop := x ≤ 6

def base_7_to_decimal (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

theorem base_7_divisibility (x : ℕ) : 
  is_base_7_digit x → (base_7_to_decimal x) % 29 = 0 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_base_7_divisibility_l1711_171100


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1711_171175

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x^2 - 2*x < 0 → abs (x - 2) < 2) ∧ 
  (∃ x : ℝ, abs (x - 2) < 2 ∧ ¬(x^2 - 2*x < 0)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1711_171175


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l1711_171158

def A : Set ℝ := {2, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 6 = 0}

theorem subset_implies_m_values (m : ℝ) (h : B m ⊆ A) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l1711_171158


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1711_171179

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ninth_term
  (a : ℕ → ℚ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1/2)
  (h_relation : a 2 * a 8 = 2 * a 5 + 3) :
  a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l1711_171179
