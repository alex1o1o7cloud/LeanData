import Mathlib

namespace absolute_value_difference_l2316_231690

theorem absolute_value_difference : |(8-(3^2))| - |((4^2) - (6*3))| = -1 := by
  sorry

end absolute_value_difference_l2316_231690


namespace complex_fraction_simplification_l2316_231693

theorem complex_fraction_simplification :
  let z₁ : ℂ := 5 + 7 * I
  let z₂ : ℂ := 2 + 3 * I
  z₁ / z₂ = (31 : ℚ) / 13 - (1 : ℚ) / 13 * I :=
by sorry

end complex_fraction_simplification_l2316_231693


namespace max_value_m_inequality_solution_l2316_231674

theorem max_value_m (a b : ℝ) (h : a ≠ b) :
  (∃ m : ℝ, ∀ M : ℝ, (∀ a b : ℝ, a ≠ b → M * |a - b| ≤ |2*a + b| + |a + 2*b|) → M ≤ m) ∧
  (∀ a b : ℝ, a ≠ b → 1 * |a - b| ≤ |2*a + b| + |a + 2*b|) :=
by sorry

theorem inequality_solution (x : ℝ) :
  |x - 1| < 1 * (2*x + 1) ↔ x > 0 :=
by sorry

end max_value_m_inequality_solution_l2316_231674


namespace partnership_investment_ratio_l2316_231653

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  a_investment : ℝ
  b_investment_multiple : ℝ
  annual_gain : ℝ
  a_share : ℝ

/-- Calculates the ratio of B's investment to A's investment --/
def investment_ratio (p : Partnership) : ℝ := p.b_investment_multiple

theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : p.a_investment > 0) :
  investment_ratio p = 3 := by
  sorry

#check partnership_investment_ratio

end partnership_investment_ratio_l2316_231653


namespace magical_red_knights_fraction_l2316_231667

theorem magical_red_knights_fraction (total : ℕ) (red blue magical : ℕ) :
  total > 0 →
  red + blue = total →
  red = (3 * total) / 8 →
  magical = total / 4 →
  ∃ (p q : ℕ), q > 0 ∧ 
    red * p * 3 * blue = blue * q * 3 * red ∧
    red * p * q + blue * p * q = magical * q * 3 →
  (3 : ℚ) / 7 = p / q := by
  sorry

end magical_red_knights_fraction_l2316_231667


namespace first_four_digits_1973_l2316_231655

theorem first_four_digits_1973 (n : ℕ) (h : ∀ k : ℕ, n ≠ 10^k) :
  ∃ j k : ℕ, j > 0 ∧ k > 0 ∧ 1973 ≤ (n^j : ℝ) / (10^k : ℝ) ∧ (n^j : ℝ) / (10^k : ℝ) < 1974 :=
sorry

end first_four_digits_1973_l2316_231655


namespace largest_unachievable_sum_l2316_231623

theorem largest_unachievable_sum (a : ℕ) (ha : Odd a) (ha_pos : 0 < a) :
  let n := (a^2 + 5*a + 4) / 2
  (∀ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z → a*x + (a+1)*y + (a+2)*z ≠ n) ∧
  (∀ m : ℕ, n < m → ∃ x y z : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ a*x + (a+1)*y + (a+2)*z = m) :=
by sorry

end largest_unachievable_sum_l2316_231623


namespace perimeter_difference_is_one_l2316_231633

-- Define the figures
def figure1_width : ℕ := 4
def figure1_height : ℕ := 2
def figure1_extra_square : ℕ := 1

def figure2_width : ℕ := 6
def figure2_height : ℕ := 2

-- Define the perimeter calculation functions
def perimeter_figure1 (w h e : ℕ) : ℕ :=
  2 * (w + h) + 3 * e

def perimeter_figure2 (w h : ℕ) : ℕ :=
  2 * (w + h)

-- Theorem statement
theorem perimeter_difference_is_one :
  Int.natAbs (perimeter_figure1 figure1_width figure1_height figure1_extra_square -
              perimeter_figure2 figure2_width figure2_height) = 1 := by
  sorry

end perimeter_difference_is_one_l2316_231633


namespace correlation_coefficient_properties_l2316_231687

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

-- Define a positive relationship between two variables
def positive_relationship (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b → y a < y b

-- Define a perfect linear relationship between two variables
def perfect_linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ m b, ∀ t, y t = m * x t + b

-- Theorem statement
theorem correlation_coefficient_properties
  (x y : ℝ → ℝ) (r : ℝ) (h : r = correlation_coefficient x y) :
  (r > 0 → positive_relationship x y) ∧
  (r = 1 ∨ r = -1 → perfect_linear_relationship x y) :=
sorry

end correlation_coefficient_properties_l2316_231687


namespace real_roots_condition_specific_condition_l2316_231652

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Part 1: Real roots condition
theorem real_roots_condition (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ↔ m ≤ 1/4 :=
sorry

-- Part 2: Specific condition leading to m = -1
theorem specific_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁*x₂ + x₁ + x₂ = 4) →
  m = -1 :=
sorry

end real_roots_condition_specific_condition_l2316_231652


namespace ellipse_equation_l2316_231670

/-- An ellipse with center at the origin, coordinate axes as axes of symmetry,
    and passing through points (√6, 1) and (-√3, -√2) has the equation x²/9 + y²/3 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
    x^2 / m + y^2 / n = 1 ∧
    6 / m + 1 / n = 1 ∧
    3 / m + 2 / n = 1) →
  x^2 / 9 + y^2 / 3 = 1 := by
sorry

end ellipse_equation_l2316_231670


namespace limit_implies_a_and_b_l2316_231638

/-- Given that the limit of (ln(2-x))^2 / (x^2 + ax + b) as x approaches 1 is equal to 1,
    prove that a = -2 and b = 1. -/
theorem limit_implies_a_and_b (a b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → 
    |((Real.log (2 - x))^2) / (x^2 + a*x + b) - 1| < ε) →
  a = -2 ∧ b = 1 := by
sorry

end limit_implies_a_and_b_l2316_231638


namespace smallest_prime_after_six_nonprimes_l2316_231678

theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), 
    (∀ k ∈ Finset.range 6, ¬ Nat.Prime (n + k + 1)) ∧ 
    Nat.Prime (n + 7) ∧
    (∀ m < n, ¬(∀ k ∈ Finset.range 6, ¬ Nat.Prime (m + k + 1)) ∨ ¬Nat.Prime (m + 7)) ∧
    n + 7 = 37 :=
by sorry

end smallest_prime_after_six_nonprimes_l2316_231678


namespace intersection_and_perpendicular_lines_l2316_231634

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  line x y

-- Define a function to check if a line passes through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  point_on_line x₁ y₁ line ∧ point_on_line x₂ y₂ line

-- Define a function to check if two lines are perpendicular
def perpendicular (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧
  (∀ x y : ℝ, line1 x y → y = m₁ * x + (P.2 - m₁ * P.1)) ∧
  (∀ x y : ℝ, line2 x y → y = m₂ * x + (P.2 - m₂ * P.1))

-- Theorem statement
theorem intersection_and_perpendicular_lines :
  (point_on_line P.1 P.2 l₁ ∧ point_on_line P.1 P.2 l₂) →
  (∃ line1 : ℝ → ℝ → Prop, line_through_points P.1 P.2 0 0 line1 ∧
    ∀ x y : ℝ, line1 x y ↔ 3*x + 2*y = 0) ∧
  (∃ line2 : ℝ → ℝ → Prop, line_through_points P.1 P.2 P.1 P.2 line2 ∧
    perpendicular line2 l₃ ∧
    ∀ x y : ℝ, line2 x y ↔ 3*x + y + 6 = 0) :=
sorry

end intersection_and_perpendicular_lines_l2316_231634


namespace defective_shipped_percentage_l2316_231622

theorem defective_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (shipped_rate : ℝ)
  (h1 : defective_rate = 0.07)
  (h2 : shipped_rate = 0.05) :
  (defective_rate * shipped_rate) * 100 = 0.35 := by
sorry

end defective_shipped_percentage_l2316_231622


namespace nonagon_dissection_l2316_231698

/-- Represents a rhombus with unit side length and a specific angle -/
structure Rhombus :=
  (angle : ℝ)

/-- Represents an isosceles triangle with unit side length and a specific vertex angle -/
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)

/-- Represents a regular polygon with a specific number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- The original 9-gon composed of specific shapes -/
def original_nonagon : RegularPolygon :=
  { sides := 9 }

/-- The set of rhombuses with 40° angles -/
def rhombuses_40 : Finset Rhombus :=
  sorry

/-- The set of rhombuses with 80° angles -/
def rhombuses_80 : Finset Rhombus :=
  sorry

/-- The set of isosceles triangles with 120° vertex angles -/
def triangles_120 : Finset IsoscelesTriangle :=
  sorry

/-- Represents the dissection of the original nonagon into three congruent regular nonagons -/
def dissection (original : RegularPolygon) (parts : Finset RegularPolygon) : Prop :=
  sorry

/-- The theorem stating that the original nonagon can be dissected into three congruent regular nonagons -/
theorem nonagon_dissection :
  ∃ (parts : Finset RegularPolygon),
    (parts.card = 3) ∧
    (∀ p ∈ parts, p.sides = 9) ∧
    (dissection original_nonagon parts) :=
sorry

end nonagon_dissection_l2316_231698


namespace seed_packet_combinations_l2316_231637

/-- Represents the cost of a sunflower seed packet -/
def sunflower_cost : ℕ := 4

/-- Represents the cost of a lavender seed packet -/
def lavender_cost : ℕ := 1

/-- Represents the cost of a marigold seed packet -/
def marigold_cost : ℕ := 3

/-- Represents the total budget -/
def total_budget : ℕ := 60

/-- Counts the number of non-negative integer solutions to the equation -/
def count_solutions : ℕ := sorry

/-- Theorem stating that there are exactly 72 different combinations of seed packets -/
theorem seed_packet_combinations : count_solutions = 72 := by sorry

end seed_packet_combinations_l2316_231637


namespace correct_remaining_money_l2316_231694

/-- Calculates the remaining money after shopping --/
def remaining_money (initial_amount : ℕ) (banana_price : ℕ) (banana_quantity : ℕ) 
  (pear_price : ℕ) (asparagus_price : ℕ) (chicken_price : ℕ) : ℕ :=
  initial_amount - (banana_price * banana_quantity + pear_price + asparagus_price + chicken_price)

/-- Proves that the remaining money is correct given the initial amount and purchases --/
theorem correct_remaining_money :
  remaining_money 55 4 2 2 6 11 = 28 := by
  sorry

end correct_remaining_money_l2316_231694


namespace cost_solution_l2316_231604

/-- The cost of electronic whiteboards and projectors -/
def CostProblem (projector_cost : ℕ) (whiteboard_cost : ℕ) : Prop :=
  (whiteboard_cost = projector_cost + 4000) ∧
  (4 * whiteboard_cost + 3 * projector_cost = 44000)

/-- Theorem stating the correct costs for the whiteboard and projector -/
theorem cost_solution :
  ∃ (projector_cost whiteboard_cost : ℕ),
    CostProblem projector_cost whiteboard_cost ∧
    projector_cost = 4000 ∧
    whiteboard_cost = 8000 := by
  sorry

end cost_solution_l2316_231604


namespace fraction_increase_l2316_231658

theorem fraction_increase (m n a : ℝ) (h1 : m > n) (h2 : n > 0) (h3 : a > 0) :
  (n + a) / (m + a) > n / m := by
  sorry

end fraction_increase_l2316_231658


namespace sum_even_factors_720_l2316_231685

def even_factor_sum (n : ℕ) : ℕ := sorry

theorem sum_even_factors_720 : even_factor_sum 720 = 2340 := by sorry

end sum_even_factors_720_l2316_231685


namespace circle_center_is_zero_one_l2316_231697

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition of circle passing through a point
def passes_through (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the condition of circle being tangent to parabola at a point
def tangent_to_parabola (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  y = parabola x ∧ passes_through c p ∧
  ∀ q : ℝ × ℝ, q ≠ p → parabola q.1 = q.2 → ¬passes_through c q

theorem circle_center_is_zero_one :
  ∃ c : Circle,
    passes_through c (0, 2) ∧
    tangent_to_parabola c (1, 1) ∧
    c.center = (0, 1) := by
  sorry

end circle_center_is_zero_one_l2316_231697


namespace infinitely_many_perfect_squares_in_sequence_l2316_231601

theorem infinitely_many_perfect_squares_in_sequence :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ ∃ m : ℕ, ⌊n * Real.sqrt 2⌋ = m^2 := by
  sorry

end infinitely_many_perfect_squares_in_sequence_l2316_231601


namespace exist_five_naturals_sum_product_ten_l2316_231691

theorem exist_five_naturals_sum_product_ten : 
  ∃ (a b c d e : ℕ), a + b + c + d + e = 10 ∧ a * b * c * d * e = 10 :=
by sorry

end exist_five_naturals_sum_product_ten_l2316_231691


namespace triangle_sides_expression_l2316_231675

theorem triangle_sides_expression (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle_ineq_1 : a + b > c)
  (h_triangle_ineq_2 : a + c > b)
  (h_triangle_ineq_3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c := by
  sorry

end triangle_sides_expression_l2316_231675


namespace min_sum_a_b_l2316_231611

theorem min_sum_a_b (a b : ℕ+) (h : (20 : ℚ) / 19 = 1 + 1 / (1 + a / b)) :
  ∃ (a' b' : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + a' / b') ∧ a' + b' = 19 ∧ 
  ∀ (c d : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + c / d) → a' + b' ≤ c + d :=
sorry

end min_sum_a_b_l2316_231611


namespace prime_pair_perfect_square_sum_theorem_l2316_231612

/-- A pair of prime numbers (p, q) such that p^2 + 5pq + 4q^2 is a perfect square -/
def PrimePairWithPerfectSquareSum (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∃ k : ℕ, p^2 + 5*p*q + 4*q^2 = k^2

/-- The theorem stating that only three specific pairs of prime numbers satisfy the condition -/
theorem prime_pair_perfect_square_sum_theorem :
  ∀ p q : ℕ, PrimePairWithPerfectSquareSum p q ↔ 
    ((p = 13 ∧ q = 3) ∨ (p = 5 ∧ q = 11) ∨ (p = 7 ∧ q = 5)) :=
by sorry

end prime_pair_perfect_square_sum_theorem_l2316_231612


namespace worker_c_left_days_l2316_231619

def work_rate (days : ℕ) : ℚ := 1 / days

theorem worker_c_left_days 
  (rate_a rate_b rate_c : ℚ)
  (total_days : ℕ)
  (h1 : rate_a = work_rate 30)
  (h2 : rate_b = work_rate 30)
  (h3 : rate_c = work_rate 40)
  (h4 : total_days = 12)
  : ∃ (x : ℕ), 
    (total_days - x) * (rate_a + rate_b + rate_c) + x * (rate_a + rate_b) = 1 ∧ 
    x = 8 := by
  sorry

end worker_c_left_days_l2316_231619


namespace first_scenario_solution_second_scenario_solution_l2316_231624

/-- Represents the purchase scenarios of a company buying noodles -/
structure NoodlePurchase where
  /-- Total cost in yuan -/
  total_cost : ℕ
  /-- Total number of portions -/
  total_portions : ℕ
  /-- Price of mixed sauce noodles in yuan -/
  mixed_sauce_price : ℕ
  /-- Price of beef noodles in yuan -/
  beef_price : ℕ

/-- Represents the updated purchase scenario -/
structure UpdatedNoodlePurchase where
  /-- Cost of mixed sauce noodles in yuan -/
  mixed_sauce_cost : ℕ
  /-- Cost of beef noodles in yuan -/
  beef_cost : ℕ
  /-- Price difference between beef and mixed sauce noodles in yuan -/
  price_difference : ℕ

/-- Theorem for the first scenario -/
theorem first_scenario_solution (purchase : NoodlePurchase)
  (h1 : purchase.total_cost = 3000)
  (h2 : purchase.total_portions = 170)
  (h3 : purchase.mixed_sauce_price = 15)
  (h4 : purchase.beef_price = 20) :
  ∃ (mixed_sauce beef : ℕ),
    mixed_sauce = 80 ∧
    beef = 90 ∧
    mixed_sauce + beef = purchase.total_portions ∧
    mixed_sauce * purchase.mixed_sauce_price + beef * purchase.beef_price = purchase.total_cost :=
  sorry

/-- Theorem for the second scenario -/
theorem second_scenario_solution (purchase : UpdatedNoodlePurchase)
  (h1 : purchase.mixed_sauce_cost = 1260)
  (h2 : purchase.beef_cost = 1200)
  (h3 : purchase.price_difference = 6) :
  ∃ (beef : ℕ),
    beef = 60 ∧
    (3 * beef : ℚ) / 2 * (purchase.beef_cost / beef - purchase.price_difference) = purchase.mixed_sauce_cost :=
  sorry

end first_scenario_solution_second_scenario_solution_l2316_231624


namespace endpoint_coordinate_sum_l2316_231644

/-- Given a line segment with one endpoint (6, -2) and midpoint (3, 5),
    the sum of the coordinates of the other endpoint is 12. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (6 + x) / 2 = 3 →
  (-2 + y) / 2 = 5 →
  x + y = 12 := by
  sorry

end endpoint_coordinate_sum_l2316_231644


namespace remainder_5031_div_28_l2316_231643

theorem remainder_5031_div_28 : 5031 % 28 = 19 := by
  sorry

end remainder_5031_div_28_l2316_231643


namespace perfect_square_theorem_l2316_231681

theorem perfect_square_theorem (a b c d : ℤ) : 
  d = (a + Real.rpow 2 (1/3 : ℝ) * b + Real.rpow 4 (1/3 : ℝ) * c)^2 → 
  ∃ k : ℤ, d = k^2 := by
sorry

end perfect_square_theorem_l2316_231681


namespace trigonometric_identity_l2316_231620

theorem trigonometric_identity (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (Real.sin x)^4 / a^2 + (Real.cos x)^4 / b^2 = 1 / (a^2 + b^2)) :
  (Real.sin x)^100 / a^100 + (Real.cos x)^100 / b^100 = 2 / (a^2 + b^2)^100 := by
  sorry

end trigonometric_identity_l2316_231620


namespace probability_of_selecting_male_student_l2316_231618

theorem probability_of_selecting_male_student 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (h1 : total_students = male_students + female_students)
  (h2 : male_students = 2)
  (h3 : female_students = 3) :
  (male_students : ℚ) / total_students = 2 / 5 := by
sorry

end probability_of_selecting_male_student_l2316_231618


namespace positive_values_of_f_l2316_231699

open Set

noncomputable def f (a : ℝ) : ℝ := a + (-1 + 9*a + 4*a^2) / (a^2 - 3*a - 10)

theorem positive_values_of_f :
  {a : ℝ | f a > 0} = Ioo (-2 : ℝ) (-1) ∪ Ioo (-1 : ℝ) 1 ∪ Ioi 5 :=
sorry

end positive_values_of_f_l2316_231699


namespace min_values_theorem_l2316_231673

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + 3 = a * b) :
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a + b ≤ x + y) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → a^2 + b^2 ≤ x^2 + y^2) ∧
  (∀ x y, x > 0 → y > 0 → x + y + 3 = x * y → 1/a + 1/b ≤ 1/x + 1/y) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y + 3 = x * y ∧ a + b = x + y ∧ a^2 + b^2 = x^2 + y^2 ∧ 1/a + 1/b = 1/x + 1/y) :=
by
  sorry

#check min_values_theorem

end min_values_theorem_l2316_231673


namespace total_cost_of_supplies_l2316_231651

/-- Calculates the total cost of supplies for a class project -/
theorem total_cost_of_supplies (num_students : ℕ) 
  (bow_cost vinegar_cost baking_soda_cost : ℕ) : 
  num_students = 23 → 
  bow_cost = 5 → 
  vinegar_cost = 2 → 
  baking_soda_cost = 1 → 
  num_students * (bow_cost + vinegar_cost + baking_soda_cost) = 184 := by
  sorry

#check total_cost_of_supplies

end total_cost_of_supplies_l2316_231651


namespace exchange_problem_l2316_231649

def exchange_rate : ℚ := 11 / 8
def spent_amount : ℕ := 70

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem exchange_problem (d : ℕ) :
  (exchange_rate * d : ℚ) - spent_amount = d →
  sum_of_digits d = 10 := by
  sorry

end exchange_problem_l2316_231649


namespace quadrilateral_area_l2316_231666

/-- The area of a quadrilateral with a diagonal of length 40 and offsets 11 and 9 is 400 -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 offset2 : ℝ) : 
  diagonal = 40 → offset1 = 11 → offset2 = 9 → 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 400 := by
sorry

end quadrilateral_area_l2316_231666


namespace hexagon_side_length_squared_l2316_231610

/-- A regular hexagon inscribed in an ellipse -/
structure InscribedHexagon where
  /-- The ellipse equation is x^2 + 9y^2 = 9 -/
  ellipse : ∀ (x y : ℝ), x^2 + 9*y^2 = 9 → ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y
  /-- One vertex of the hexagon is (0,1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the hexagon is aligned along the y-axis -/
  diagonal : ∃ (d : ℝ × ℝ) (e : ℝ × ℝ), d.1 = 0 ∧ e.1 = 0 ∧ d.2 = -e.2
  /-- The hexagon is regular -/
  regular : ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ‖s1 - s2‖ = ‖s2 - s1‖

/-- The square of the length of each side of the hexagon is 729/98 -/
theorem hexagon_side_length_squared (h : InscribedHexagon) : 
  ∃ (s1 s2 : ℝ × ℝ), s1 ≠ s2 ∧ ‖s1 - s2‖^2 = 729/98 :=
sorry

end hexagon_side_length_squared_l2316_231610


namespace count_right_triangles_with_leg_15_l2316_231603

/-- The number of right triangles with integer side lengths and one leg equal to 15 -/
def rightTrianglesWithLeg15 : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 
    let (a, b, c) := t
    a = 15 ∧ a^2 + b^2 = c^2 ∧ a < b ∧ b < c) (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card

/-- Theorem stating that there are exactly 4 right triangles with integer side lengths and one leg equal to 15 -/
theorem count_right_triangles_with_leg_15 : rightTrianglesWithLeg15 = 4 := by
  sorry

end count_right_triangles_with_leg_15_l2316_231603


namespace A_subset_complement_B_l2316_231602

-- Define the universe set S
def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Finset Char := {'a', 'c'}

-- Define set B
def B : Finset Char := {'b', 'e'}

-- Theorem statement
theorem A_subset_complement_B : A ⊆ S \ B := by sorry

end A_subset_complement_B_l2316_231602


namespace dans_car_fuel_efficiency_l2316_231662

/-- Represents the fuel efficiency of Dan's car in miles per gallon. -/
def fuel_efficiency : ℝ := 32

/-- The cost of gas in dollars per gallon. -/
def gas_cost : ℝ := 4

/-- The distance Dan's car can travel on $42 of gas, in miles. -/
def distance : ℝ := 336

/-- The amount spent on gas, in dollars. -/
def gas_spent : ℝ := 42

/-- Theorem stating that Dan's car's fuel efficiency is 32 miles per gallon. -/
theorem dans_car_fuel_efficiency :
  fuel_efficiency = distance / (gas_spent / gas_cost) := by
  sorry

end dans_car_fuel_efficiency_l2316_231662


namespace wall_height_l2316_231665

-- Define the width of the wall
def wall_width : ℝ := 4

-- Define the area of the wall
def wall_area : ℝ := 16

-- Theorem: The height of the wall is 4 feet
theorem wall_height : 
  wall_area / wall_width = 4 :=
by sorry

end wall_height_l2316_231665


namespace polynomial_coefficient_sum_l2316_231648

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
sorry

end polynomial_coefficient_sum_l2316_231648


namespace pole_intersection_height_l2316_231636

/-- Given two poles with heights 30 and 60 units, placed 50 units apart,
    the height of the intersection of the lines joining the top of each pole
    to the foot of the opposite pole is 20 units. -/
theorem pole_intersection_height :
  let h₁ : ℝ := 30  -- Height of the first pole
  let h₂ : ℝ := 60  -- Height of the second pole
  let d : ℝ := 50   -- Distance between the poles
  let m₁ : ℝ := (0 - h₁) / d  -- Slope of the first line
  let m₂ : ℝ := (0 - h₂) / (-d)  -- Slope of the second line
  let x : ℝ := (h₁ - 0) / (m₂ - m₁)  -- x-coordinate of intersection
  let y : ℝ := m₁ * x + h₁  -- y-coordinate of intersection
  y = 20 := by sorry

end pole_intersection_height_l2316_231636


namespace average_study_time_difference_l2316_231615

def daily_differences : List ℤ := [15, -5, 25, 35, -15, 10, 20]

def days_in_week : ℕ := 7

theorem average_study_time_difference : 
  (daily_differences.sum : ℚ) / days_in_week = 12 := by
  sorry

end average_study_time_difference_l2316_231615


namespace odd_square_minus_one_div_eight_l2316_231621

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k :=
by sorry

end odd_square_minus_one_div_eight_l2316_231621


namespace average_of_six_numbers_l2316_231632

theorem average_of_six_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (subset : ℕ)
  (avg_subset : ℚ)
  (h_total : total = 10)
  (h_avg_all : avg_all = 80)
  (h_subset : subset = 4)
  (h_avg_subset : avg_subset = 113) :
  let remaining := total - subset
  let sum_all := total * avg_all
  let sum_subset := subset * avg_subset
  let sum_remaining := sum_all - sum_subset
  (sum_remaining : ℚ) / remaining = 58 := by sorry

end average_of_six_numbers_l2316_231632


namespace monday_rainfall_calculation_l2316_231609

def total_rainfall : ℝ := 0.67
def tuesday_rainfall : ℝ := 0.42
def wednesday_rainfall : ℝ := 0.08

theorem monday_rainfall_calculation :
  ∃ (monday_rainfall : ℝ),
    monday_rainfall + tuesday_rainfall + wednesday_rainfall = total_rainfall ∧
    monday_rainfall = 0.17 := by
  sorry

end monday_rainfall_calculation_l2316_231609


namespace double_area_square_exists_l2316_231682

/-- A point on the grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid --/
structure GridSquare where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  d : GridPoint

/-- The area of a grid square --/
def area (s : GridSquare) : ℕ := sorry

/-- A square is legal if its vertices are grid points --/
def is_legal (s : GridSquare) : Prop := sorry

theorem double_area_square_exists (n : ℕ) (h : ∃ s : GridSquare, is_legal s ∧ area s = n) :
  ∃ t : GridSquare, is_legal t ∧ area t = 2 * n := by sorry

end double_area_square_exists_l2316_231682


namespace max_b_value_l2316_231679

-- Define a lattice point
def is_lattice_point (x y : ℤ) : Prop := true

-- Define the line equation
def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

-- Define the condition for not passing through lattice points
def no_lattice_intersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → is_lattice_point x y → line_equation m x ≠ y

-- State the theorem
theorem max_b_value :
  ∀ b : ℚ, (∀ m : ℚ, 1/3 < m ∧ m < b → no_lattice_intersection m) →
  b ≤ 68/203 :=
sorry

end max_b_value_l2316_231679


namespace parallelepiped_net_removal_l2316_231683

/-- Represents a parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a parallelepiped -/
structure Net where
  squares : ℕ

/-- Represents the number of possible positions to remove a square from a net -/
def possible_removals (n : Net) : ℕ := sorry

theorem parallelepiped_net_removal 
  (p : Parallelepiped) 
  (n : Net) :
  p.length = 2 ∧ p.width = 1 ∧ p.height = 1 →
  n.squares = 10 →
  possible_removals { squares := n.squares - 1 } = 5 :=
sorry

end parallelepiped_net_removal_l2316_231683


namespace complement_intersection_theorem_l2316_231608

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A ∩ Set.compl B) = {0, 5} := by
  sorry

end complement_intersection_theorem_l2316_231608


namespace cartesian_plane_problem_l2316_231654

-- Define the points
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, 0)

-- Define vectors
def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B

-- Define the length of OC
def OC_length : ℝ := 1

-- Theorem statement
theorem cartesian_plane_problem :
  -- Part 1: Angle between OA and OB is 45°
  let angle := Real.arccos ((OA.1 * OB.1 + OA.2 * OB.2) / (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2)))
  angle = Real.pi / 4 ∧
  -- Part 2: If OC ⊥ OA, then C has coordinates (±√2/2, ±√2/2)
  (∀ C : ℝ × ℝ, (C.1 * OA.1 + C.2 * OA.2 = 0 ∧ C.1^2 + C.2^2 = OC_length^2) →
    (C = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ C = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2))) ∧
  -- Part 3: Range of |OA + OB + OC|
  (∀ C : ℝ × ℝ, C.1^2 + C.2^2 = OC_length^2 →
    Real.sqrt 10 - 1 ≤ Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ∧
    Real.sqrt ((OA.1 + OB.1 + C.1)^2 + (OA.2 + OB.2 + C.2)^2) ≤ Real.sqrt 10 + 1) :=
by
  sorry

end cartesian_plane_problem_l2316_231654


namespace geometric_sequence_product_l2316_231606

/-- A positive term geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a 2 / a 1

/-- The theorem statement -/
theorem geometric_sequence_product (seq : GeometricSequence) 
  (h1 : 2 * (seq.a 1)^2 - 7 * (seq.a 1) + 6 = 0)
  (h2 : 2 * (seq.a 48)^2 - 7 * (seq.a 48) + 6 = 0) :
  seq.a 1 * seq.a 2 * seq.a 25 * seq.a 48 * seq.a 49 = 9 * Real.sqrt 3 := by
  sorry

#check geometric_sequence_product

end geometric_sequence_product_l2316_231606


namespace num_biology_books_is_14_l2316_231642

/-- The number of ways to choose 2 books from n books -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of chemistry books -/
def num_chemistry_books : ℕ := 8

/-- The total number of ways to choose 2 biology and 2 chemistry books -/
def total_ways : ℕ := 2548

/-- The number of biology books satisfies the given conditions -/
theorem num_biology_books_is_14 : 
  ∃ (n : ℕ), n > 0 ∧ choose_two n * choose_two num_chemistry_books = total_ways ∧ n = 14 :=
sorry

end num_biology_books_is_14_l2316_231642


namespace negation_equivalence_l2316_231686

theorem negation_equivalence (f g : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x * g x = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) := by
  sorry

end negation_equivalence_l2316_231686


namespace train_platform_crossing_time_l2316_231626

/-- Given a train of length 2000 m that crosses a tree in 200 sec,
    the time it takes to pass a platform of length 2500 m is 450 sec. -/
theorem train_platform_crossing_time :
  ∀ (train_length platform_length tree_crossing_time : ℝ),
    train_length = 2000 →
    platform_length = 2500 →
    tree_crossing_time = 200 →
    (train_length + platform_length) / (train_length / tree_crossing_time) = 450 := by
  sorry

end train_platform_crossing_time_l2316_231626


namespace rent_calculation_l2316_231659

def monthly_budget (rent : ℚ) : Prop :=
  let food := (3/5) * rent
  let mortgage := 3 * food
  let savings := 2000
  let taxes := (2/5) * savings
  rent + food + mortgage + savings + taxes = 4840

theorem rent_calculation :
  ∃ (rent : ℚ), monthly_budget rent ∧ rent = 600 := by
sorry

end rent_calculation_l2316_231659


namespace monomial_combination_l2316_231645

/-- 
Given two monomials that can be combined, this theorem proves 
the values of their exponents.
-/
theorem monomial_combination (m n : ℕ) : 
  (∃ (a b : ℝ), 3 * a^(m+1) * b = -b^(n-1) * a^3) → 
  (m = 2 ∧ n = 2) := by
  sorry

#check monomial_combination

end monomial_combination_l2316_231645


namespace work_completion_equality_prove_men_first_group_l2316_231629

/-- The number of days it takes the first group to complete the work -/
def days_first_group : ℕ := 30

/-- The number of men in the second group -/
def men_second_group : ℕ := 10

/-- The number of days it takes the second group to complete the work -/
def days_second_group : ℕ := 36

/-- The number of men in the first group -/
def men_first_group : ℕ := 12

theorem work_completion_equality :
  men_first_group * days_first_group = men_second_group * days_second_group :=
by sorry

theorem prove_men_first_group :
  men_first_group = (men_second_group * days_second_group) / days_first_group :=
by sorry

end work_completion_equality_prove_men_first_group_l2316_231629


namespace shortest_path_on_specific_floor_l2316_231696

/-- Represents a rectangular floor with a missing tile -/
structure RectangularFloor :=
  (width : Nat)
  (length : Nat)
  (missingTileX : Nat)
  (missingTileY : Nat)

/-- Calculates the shortest path length for a bug traversing the floor -/
def shortestPathLength (floor : RectangularFloor) : Nat :=
  floor.width + floor.length - Nat.gcd floor.width floor.length + 1

/-- Theorem stating the shortest path length for the given floor configuration -/
theorem shortest_path_on_specific_floor :
  let floor : RectangularFloor := {
    width := 12,
    length := 20,
    missingTileX := 6,
    missingTileY := 10
  }
  shortestPathLength floor = 29 := by
  sorry


end shortest_path_on_specific_floor_l2316_231696


namespace rectangle_triangle_equality_l2316_231663

theorem rectangle_triangle_equality (AB AD DC : ℝ) (h1 : AB = 4) (h2 : AD = 8) (h3 : DC = 4) :
  let ABCD_area := AB * AD
  let DCE_area := (1 / 2) * DC * CE
  let CE := 2 * ABCD_area / DC
  ABCD_area = DCE_area → DE = 4 * Real.sqrt 17 :=
by
  sorry

end rectangle_triangle_equality_l2316_231663


namespace largest_among_four_l2316_231641

theorem largest_among_four : ∀ (a b c d : ℝ), 
  a = 0 → b = -1 → c = -2 → d = Real.sqrt 3 →
  d = max a (max b (max c d)) :=
by sorry

end largest_among_four_l2316_231641


namespace wang_liang_age_l2316_231639

def is_valid_age (age : ℕ) : Prop :=
  ∃ (birth_year : ℕ),
    (2012 - birth_year = age) ∧
    (age = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10))

theorem wang_liang_age :
  (is_valid_age 7 ∨ is_valid_age 25) ∧
  ∀ (age : ℕ), is_valid_age age → (age = 7 ∨ age = 25) :=
sorry

end wang_liang_age_l2316_231639


namespace total_cantaloupes_l2316_231625

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44
def susan_cantaloupes : ℕ := 57
def nancy_cantaloupes : ℕ := 25

theorem total_cantaloupes : 
  fred_cantaloupes + tim_cantaloupes + susan_cantaloupes + nancy_cantaloupes = 164 := by
  sorry

end total_cantaloupes_l2316_231625


namespace seat_swapping_arrangements_l2316_231650

def number_of_students : ℕ := 7
def students_to_swap : ℕ := 3

theorem seat_swapping_arrangements :
  (number_of_students.choose students_to_swap) * (students_to_swap.factorial) = 70 := by
  sorry

end seat_swapping_arrangements_l2316_231650


namespace rebecca_marbles_l2316_231616

theorem rebecca_marbles (group_size : ℕ) (num_groups : ℕ) (total_marbles : ℕ) : 
  group_size = 4 → num_groups = 5 → total_marbles = group_size * num_groups → total_marbles = 20 := by
  sorry

end rebecca_marbles_l2316_231616


namespace distance_specific_point_to_line_l2316_231684

/-- The distance from a point to a line in 3D space --/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point1 line_point2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem: The distance from (2, 3, -1) to the line passing through (3, -1, 4) and (5, 0, 1) is √3667/14 --/
theorem distance_specific_point_to_line :
  let point : ℝ × ℝ × ℝ := (2, 3, -1)
  let line_point1 : ℝ × ℝ × ℝ := (3, -1, 4)
  let line_point2 : ℝ × ℝ × ℝ := (5, 0, 1)
  distance_point_to_line point line_point1 line_point2 = Real.sqrt 3667 / 14 := by
  sorry

end distance_specific_point_to_line_l2316_231684


namespace two_distinct_roots_range_l2316_231677

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m+3)

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

-- Theorem statement
theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < -2 ∨ m > 6 :=
sorry

end two_distinct_roots_range_l2316_231677


namespace range_of_a_for_two_negative_roots_l2316_231692

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + |a|

-- Define the condition for two negative roots
def has_two_negative_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
  quadratic a x₁ = 0 ∧ quadratic a x₂ = 0

-- State the theorem
theorem range_of_a_for_two_negative_roots :
  ∃ l u : ℝ, ∀ a : ℝ, has_two_negative_roots a ↔ l < a ∧ a < u :=
sorry

end range_of_a_for_two_negative_roots_l2316_231692


namespace f_2x_l2316_231613

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2*x) = 4*x^2 - 1 := by
  sorry

end f_2x_l2316_231613


namespace K33_not_planar_l2316_231672

/-- A bipartite graph with two sets of three vertices each --/
structure BipartiteGraph :=
  (left : Finset ℕ)
  (right : Finset ℕ)
  (edges : Set (ℕ × ℕ))

/-- The K₃,₃ graph --/
def K33 : BipartiteGraph :=
  { left := {1, 2, 3},
    right := {4, 5, 6},
    edges := {(1,4), (1,5), (1,6), (2,4), (2,5), (2,6), (3,4), (3,5), (3,6)} }

/-- A graph is planar if it can be drawn on a plane without edge crossings --/
def isPlanar (G : BipartiteGraph) : Prop := sorry

/-- Theorem: K₃,₃ is not planar --/
theorem K33_not_planar : ¬ isPlanar K33 := by
  sorry

end K33_not_planar_l2316_231672


namespace tom_marble_pairs_l2316_231661

/-- Represents the set of marbles Tom has -/
structure MarbleSet where
  distinct_colors : Nat  -- Number of distinct colored marbles
  yellow_marbles : Nat   -- Number of identical yellow marbles

/-- Calculates the number of ways to choose 2 marbles from a given MarbleSet -/
def count_marble_pairs (ms : MarbleSet) : Nat :=
  let yellow_pair := if ms.yellow_marbles ≥ 2 then 1 else 0
  let distinct_pairs := Nat.choose ms.distinct_colors 2
  yellow_pair + distinct_pairs

/-- Theorem: Given Tom's marble set, the number of different groups of two marbles is 7 -/
theorem tom_marble_pairs :
  let toms_marbles : MarbleSet := ⟨4, 5⟩
  count_marble_pairs toms_marbles = 7 := by
  sorry

end tom_marble_pairs_l2316_231661


namespace total_hamburgers_bought_l2316_231689

/-- Proves that the total number of hamburgers bought is 50 given the specified conditions. -/
theorem total_hamburgers_bought (total_spent : ℚ) (single_cost : ℚ) (double_cost : ℚ) (double_count : ℕ) : ℕ :=
  if total_spent = 70.5 ∧ single_cost = 1 ∧ double_cost = 1.5 ∧ double_count = 41 then
    50
  else
    0

#check total_hamburgers_bought

end total_hamburgers_bought_l2316_231689


namespace arithmetic_sequence_fourth_term_l2316_231617

/-- 
Given an arithmetic sequence where the sum of the third and fifth terms is 10,
prove that the fourth term is 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) -- a is the arithmetic sequence
  (h : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) -- definition of arithmetic sequence
  (sum_condition : a 3 + a 5 = 10) -- sum of third and fifth terms is 10
  : a 4 = 5 := by
  sorry

end arithmetic_sequence_fourth_term_l2316_231617


namespace problem_2023_l2316_231607

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end problem_2023_l2316_231607


namespace smallest_value_w_cube_plus_z_cube_l2316_231647

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 1)
  (h2 : Complex.abs (w^2 + z^2) = 14) :
  ∃ (min_val : ℝ), min_val = 41/2 ∧ 
    ∀ (w' z' : ℂ), Complex.abs (w' + z') = 1 → Complex.abs (w'^2 + z'^2) = 14 → 
      Complex.abs (w'^3 + z'^3) ≥ min_val :=
sorry

end smallest_value_w_cube_plus_z_cube_l2316_231647


namespace quadratic_root_difference_squares_l2316_231635

theorem quadratic_root_difference_squares (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) → 
  x₁^2 - x₂^2 = c^2 / a^2 → 
  b^4 - c^4 = 4 * a^3 * b * c := by
  sorry

end quadratic_root_difference_squares_l2316_231635


namespace vector_subtraction_l2316_231640

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a - b = (5, -3) := by sorry

end vector_subtraction_l2316_231640


namespace smallest_product_is_zero_l2316_231614

def S : Set Int := {-10, -6, 0, 2, 5}

theorem smallest_product_is_zero :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = 0 ∧ 
  ∀ (a b : Int), a ∈ S → b ∈ S → x * y ≤ a * b :=
by sorry

end smallest_product_is_zero_l2316_231614


namespace geometric_sequence_problem_l2316_231605

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The main theorem -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 5 * a 11 = 3 →
  a 3 + a 13 = 4 →
  (a 15 / a 5 = 1/3 ∨ a 15 / a 5 = 3) :=
by sorry

end geometric_sequence_problem_l2316_231605


namespace congruence_solution_count_l2316_231600

theorem congruence_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 43 = 75 % 43 := by sorry

end congruence_solution_count_l2316_231600


namespace edith_books_count_edith_books_count_proof_l2316_231631

theorem edith_books_count : ℕ → Prop :=
  fun total : ℕ =>
    ∃ (x y : ℕ),
      x = (120 * 56) / 100 ∧  -- 20% more than 56
      y = (x + 56) / 2 ∧      -- half of total novels
      total = x + 56 + y ∧    -- total books
      total = 185             -- correct answer

-- The proof goes here
theorem edith_books_count_proof : edith_books_count 185 := by
  sorry

end edith_books_count_edith_books_count_proof_l2316_231631


namespace bowl_score_theorem_l2316_231657

def noa_score : ℕ := 30

def phillip_score (noa : ℕ) : ℕ := 2 * noa

def total_score (noa phillip : ℕ) : ℕ := noa + phillip

theorem bowl_score_theorem : 
  total_score noa_score (phillip_score noa_score) = 90 := by
  sorry

end bowl_score_theorem_l2316_231657


namespace baseball_gear_cost_l2316_231664

theorem baseball_gear_cost (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 67)
  (h2 : remaining_amount = 33) :
  initial_amount - remaining_amount = 34 := by
  sorry

end baseball_gear_cost_l2316_231664


namespace unpainted_cubes_5x5x5_l2316_231680

/-- Given a cube of size n x n x n, where the outer layer is painted,
    calculate the number of unpainted inner cubes. -/
def unpaintedCubes (n : ℕ) : ℕ :=
  (n - 2)^3

/-- The number of unpainted cubes in a 5x5x5 painted cube is 27. -/
theorem unpainted_cubes_5x5x5 :
  unpaintedCubes 5 = 27 := by
  sorry

end unpainted_cubes_5x5x5_l2316_231680


namespace fifth_month_sale_l2316_231688

def sales_problem (sales1 sales2 sales3 sales4 sales6 average : ℕ) : Prop :=
  let total_sales := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales4 + sales6
  total_sales - known_sales = 3560

theorem fifth_month_sale :
  sales_problem 3435 3920 3855 4230 2000 3500 := by
  sorry

end fifth_month_sale_l2316_231688


namespace race_distances_l2316_231646

/-- In a 100 m race, if B beats C by 4 m and A beats C by 28 m, then A beats B by 24 m. -/
theorem race_distances (x : ℝ) : 
  (100 : ℝ) - x - 4 = 100 - 28 → x = 24 := by sorry

end race_distances_l2316_231646


namespace profit_is_333_l2316_231671

/-- Represents the candy bar sales scenario -/
structure CandyBarSales where
  totalBars : ℕ
  firstBatchCost : ℚ
  secondBatchCost : ℚ
  firstBatchSell : ℚ
  secondBatchSell : ℚ

/-- Calculates the profit from candy bar sales -/
def calculateProfit (sales : CandyBarSales) : ℚ :=
  let costPrice := (800 / 3) + 100
  let sellingPrice := 300 + (600 * 2 / 3)
  sellingPrice - costPrice

/-- Theorem stating that the profit is $333 -/
theorem profit_is_333 (sales : CandyBarSales) 
    (h1 : sales.totalBars = 1200)
    (h2 : sales.firstBatchCost = 1/3)
    (h3 : sales.secondBatchCost = 1/4)
    (h4 : sales.firstBatchSell = 1/2)
    (h5 : sales.secondBatchSell = 2/3) :
  Int.floor (calculateProfit sales) = 333 := by
  sorry

#eval Int.floor (calculateProfit { 
  totalBars := 1200, 
  firstBatchCost := 1/3, 
  secondBatchCost := 1/4, 
  firstBatchSell := 1/2, 
  secondBatchSell := 2/3
})

end profit_is_333_l2316_231671


namespace erased_numbers_l2316_231660

def has_digit (n : ℕ) (d : ℕ) : Prop := ∃ k m : ℕ, n = k * 10 + d + m * 10

theorem erased_numbers (remaining_with_one : ℕ) (remaining_with_two : ℕ) (remaining_without_one_or_two : ℕ) :
  remaining_with_one = 20 →
  remaining_with_two = 19 →
  remaining_without_one_or_two = 30 →
  (∀ n : ℕ, n ≤ 100 → (has_digit n 1 ∨ has_digit n 2 ∨ (¬ has_digit n 1 ∧ ¬ has_digit n 2))) →
  100 - (remaining_with_one + remaining_with_two + remaining_without_one_or_two - 2) = 33 := by
  sorry

end erased_numbers_l2316_231660


namespace triangle_ratio_l2316_231656

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition: sides form an arithmetic sequence -/
def is_arithmetic_sequence (t : Triangle) : Prop :=
  2 * t.b = t.a + t.c

/-- Condition: C = 2(A + B) -/
def angle_condition (t : Triangle) : Prop :=
  t.C = 2 * (t.A + t.B)

/-- Theorem: If sides form an arithmetic sequence and C = 2(A + B), then b/a = 5/3 -/
theorem triangle_ratio (t : Triangle) 
    (h1 : is_arithmetic_sequence t) 
    (h2 : angle_condition t) : 
    t.b / t.a = 5 / 3 := by
  sorry

end triangle_ratio_l2316_231656


namespace half_AB_equals_neg_two_two_l2316_231630

def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![-3, 2]

theorem half_AB_equals_neg_two_two : 
  (1 / 2 : ℝ) • (OB - OA) = ![(-2 : ℝ), (2 : ℝ)] := by sorry

end half_AB_equals_neg_two_two_l2316_231630


namespace james_out_of_pocket_l2316_231628

def initial_purchase : ℝ := 3000
def returned_tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost_increase : ℝ := 0.2
def sold_bike_price_ratio : ℝ := 0.8
def toaster_cost : ℝ := 100

theorem james_out_of_pocket :
  let remaining_after_returns := initial_purchase - returned_tv_cost - returned_bike_cost
  let sold_bike_cost := returned_bike_cost * (1 + sold_bike_cost_increase)
  let sold_bike_price := sold_bike_cost * sold_bike_price_ratio
  let final_amount := remaining_after_returns - sold_bike_price + toaster_cost
  final_amount = 1420 := by sorry

end james_out_of_pocket_l2316_231628


namespace three_fourths_to_fifth_power_l2316_231695

theorem three_fourths_to_fifth_power : (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by
  sorry

end three_fourths_to_fifth_power_l2316_231695


namespace product_of_solutions_l2316_231668

theorem product_of_solutions : ∃ (y₁ y₂ : ℝ), 
  (abs y₁ = 3 * (abs y₁ - 2)) ∧ 
  (abs y₂ = 3 * (abs y₂ - 2)) ∧ 
  (y₁ ≠ y₂) ∧ 
  (y₁ * y₂ = -9) := by
  sorry

end product_of_solutions_l2316_231668


namespace age_difference_l2316_231669

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 30 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 32 := by
  sorry

end age_difference_l2316_231669


namespace product_14_sum_9_l2316_231627

theorem product_14_sum_9 :
  ∀ a b : ℕ, 
    1 ≤ a ∧ a ≤ 10 →
    1 ≤ b ∧ b ≤ 10 →
    a * b = 14 →
    a + b = 9 := by
  sorry

end product_14_sum_9_l2316_231627


namespace rationalize_sqrt_five_twelfths_l2316_231676

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l2316_231676
