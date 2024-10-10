import Mathlib

namespace arcsin_neg_one_l2455_245596

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end arcsin_neg_one_l2455_245596


namespace integer_solution_abc_l2455_245586

theorem integer_solution_abc : 
  ∀ a b c : ℤ, 1 < a ∧ a < b ∧ b < c ∧ ((a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1)) → 
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) := by
  sorry

end integer_solution_abc_l2455_245586


namespace trajectory_is_parabola_l2455_245593

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (x^2 + (y + 3)^2)^(1/2) = |y - 3| → x^2 = -12*y :=
by sorry

end trajectory_is_parabola_l2455_245593


namespace dans_marbles_l2455_245549

/-- 
Given that Dan gave some marbles to Mary and has some marbles left,
prove that the initial number of marbles is equal to the sum of
the marbles given and the marbles left.
-/
theorem dans_marbles (marbles_given : ℕ) (marbles_left : ℕ) :
  marbles_given + marbles_left = 64 → marbles_given = 14 → marbles_left = 50 := by
  sorry

#check dans_marbles

end dans_marbles_l2455_245549


namespace coin_flip_probability_l2455_245553

theorem coin_flip_probability (p : ℝ) (h : p = 1 / 2) :
  p * (1 - p) * (1 - p) = 1 / 8 := by
  sorry

end coin_flip_probability_l2455_245553


namespace cloth_cost_price_l2455_245541

/-- Proves that the cost price of one meter of cloth is 140 Rs. given the conditions -/
theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_meter = 10) :
  (selling_price - total_length * profit_per_meter) / total_length = 140 :=
by
  sorry

#check cloth_cost_price

end cloth_cost_price_l2455_245541


namespace divisible_by_72_digits_l2455_245583

theorem divisible_by_72_digits (a b : Nat) : 
  a < 10 → b < 10 → 
  (42000 + 1000 * a + 40 + b) % 72 = 0 → 
  ((a = 8 ∧ b = 0) ∨ (a = 0 ∧ b = 8)) :=
by sorry

end divisible_by_72_digits_l2455_245583


namespace a_range_l2455_245506

/-- Given points A and B, if line AB is symmetric about x-axis and intersects a circle, then a is in [1/4, 2] -/
theorem a_range (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (a, 0)
  let line_AB_symmetric : Prop := ∃ (k : ℝ), ∀ (x y : ℝ), y = k * (x - a) ↔ -y = k * (x - (-2)) + 3
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}
  let line_AB_intersects_circle : Prop := ∃ (p : ℝ × ℝ), p ∈ circle ∧ ∃ (k : ℝ), p.2 - 0 = k * (p.1 - a)
  line_AB_symmetric → line_AB_intersects_circle → a ∈ Set.Icc (1/4 : ℝ) 2 :=
by
  sorry

end a_range_l2455_245506


namespace green_green_pairs_l2455_245560

/-- Represents the distribution of students and pairs in a math competition. -/
structure Competition :=
  (total_students : ℕ)
  (blue_shirts : ℕ)
  (green_shirts : ℕ)
  (total_pairs : ℕ)
  (blue_blue_pairs : ℕ)

/-- The main theorem about the number of green-green pairs in the competition. -/
theorem green_green_pairs (c : Competition) 
  (h1 : c.total_students = 150)
  (h2 : c.blue_shirts = 68)
  (h3 : c.green_shirts = 82)
  (h4 : c.total_pairs = 75)
  (h5 : c.blue_blue_pairs = 30)
  (h6 : c.total_students = c.blue_shirts + c.green_shirts)
  (h7 : c.total_students = 2 * c.total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 37 ∧ 
    c.total_pairs = c.blue_blue_pairs + green_green_pairs + (c.blue_shirts - 2 * c.blue_blue_pairs) :=
sorry

end green_green_pairs_l2455_245560


namespace right_triangle_hypotenuse_l2455_245543

/-- A right triangle with medians m₁ and m₂ drawn from the vertices of the acute angles has hypotenuse of length 3√(336/13) when m₁ = 6 and m₂ = √48 -/
theorem right_triangle_hypotenuse (m₁ m₂ : ℝ) (h₁ : m₁ = 6) (h₂ : m₂ = Real.sqrt 48) :
  ∃ (a b c : ℝ), 
    a^2 + b^2 = c^2 ∧  -- right triangle condition
    (b^2 + (3*a/2)^2 = m₁^2) ∧  -- first median condition
    (a^2 + (3*b/2)^2 = m₂^2) ∧  -- second median condition
    c = 3 * Real.sqrt (336/13) :=
by sorry

end right_triangle_hypotenuse_l2455_245543


namespace value_of_M_l2455_245561

theorem value_of_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1500) ∧ (M = 4500) := by
  sorry

end value_of_M_l2455_245561


namespace smallest_multiple_of_2019_l2455_245548

/-- A number of the form abcabcabc... where a, b, and c are digits -/
def RepeatingDigitNumber (a b c : ℕ) : ℕ := 
  a * 100000000 + b * 10000000 + c * 1000000 +
  a * 100000 + b * 10000 + c * 1000 +
  a * 100 + b * 10 + c

/-- The smallest multiple of 2019 of the form abcabcabc... -/
def SmallestMultiple : ℕ := 673673673

theorem smallest_multiple_of_2019 :
  (∀ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 →
    RepeatingDigitNumber a b c % 2019 = 0 →
    RepeatingDigitNumber a b c ≥ SmallestMultiple) ∧
  SmallestMultiple % 2019 = 0 ∧
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧
    RepeatingDigitNumber a b c = SmallestMultiple :=
by sorry

end smallest_multiple_of_2019_l2455_245548


namespace police_emergency_number_prime_divisor_l2455_245594

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 100 * k + 133) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end police_emergency_number_prime_divisor_l2455_245594


namespace problem_solution_l2455_245554

def f (x : ℝ) := |x + 1| + |x - 2|
def g (a x : ℝ) := |x + 1| - |x - a| + a

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 5 ↔ x ∈ Set.Icc (-2) 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ 1) := by sorry

end problem_solution_l2455_245554


namespace test_examination_l2455_245501

/-- The number of boys examined in a test --/
def num_boys : ℕ := 50

/-- The number of girls examined in the test --/
def num_girls : ℕ := 100

/-- The percentage of boys who pass the test --/
def boys_pass_rate : ℚ := 1/2

/-- The percentage of girls who pass the test --/
def girls_pass_rate : ℚ := 2/5

/-- The percentage of total students who fail the test --/
def total_fail_rate : ℚ := 5667/10000

theorem test_examination :
  num_boys = 50 ∧
  (num_boys * (1 - boys_pass_rate) + num_girls * (1 - girls_pass_rate)) / (num_boys + num_girls) = total_fail_rate :=
by sorry

end test_examination_l2455_245501


namespace a_less_than_neg_one_l2455_245565

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem a_less_than_neg_one (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by sorry

end a_less_than_neg_one_l2455_245565


namespace ball_cost_price_l2455_245517

theorem ball_cost_price (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) :
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 60 := by
  sorry

end ball_cost_price_l2455_245517


namespace market_equilibrium_change_l2455_245526

-- Define the demand and supply functions
def demand (p : ℝ) : ℝ := 150 - p
def supply (p : ℝ) : ℝ := 3 * p - 10

-- Define the new demand function after increase
def new_demand (α : ℝ) (p : ℝ) : ℝ := α * (150 - p)

-- Define the equilibrium condition
def is_equilibrium (p : ℝ) : Prop := demand p = supply p

-- Define the new equilibrium condition
def is_new_equilibrium (α : ℝ) (p : ℝ) : Prop := new_demand α p = supply p

-- State the theorem
theorem market_equilibrium_change (α : ℝ) :
  (∃ p₀ : ℝ, is_equilibrium p₀) →
  (∃ p₁ : ℝ, is_new_equilibrium α p₁ ∧ p₁ = 1.25 * p₀) →
  α = 1.4 := by sorry

end market_equilibrium_change_l2455_245526


namespace vegan_nut_free_menu_fraction_l2455_245546

theorem vegan_nut_free_menu_fraction :
  let total_vegan_dishes : ℕ := 8
  let vegan_menu_fraction : ℚ := 1/4
  let nut_containing_vegan_dishes : ℕ := 5
  let nut_free_vegan_dishes : ℕ := total_vegan_dishes - nut_containing_vegan_dishes
  let nut_free_vegan_fraction : ℚ := nut_free_vegan_dishes / total_vegan_dishes
  nut_free_vegan_fraction * vegan_menu_fraction = 3/32 :=
by sorry

end vegan_nut_free_menu_fraction_l2455_245546


namespace circle_and_tangent_line_l2455_245515

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 
                 a^2 + b^2 = r^2 ∧
                 (a - 7)^2 + (b - 7)^2 = r^2 ∧
                 b = 4/3 * a

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop :=
  (y = -3/4 * x) ∨ (x + y + 5 * Real.sqrt 2 - 7 = 0) ∨ (x + y - 5 * Real.sqrt 2 - 7 = 0)

theorem circle_and_tangent_line :
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 25) ∧
  (∀ x y, tangent_line_l x y ↔ 
    ((x + y = 0 ∨ x = y) ∧ 
     ∃ (t : ℝ), (x - 3 + t)^2 + (y - 4 + 3/4 * t)^2 = 25 ∧
                ((x + t)^2 + (y + 3/4 * t)^2 > 25 ∨ (x - t)^2 + (y - 3/4 * t)^2 > 25))) :=
by sorry

end circle_and_tangent_line_l2455_245515


namespace total_spending_calculation_l2455_245562

def shirt_price : ℝ := 13.04
def shirt_tax_rate : ℝ := 0.07
def jacket_price : ℝ := 12.27
def jacket_tax_rate : ℝ := 0.085
def scarf_price : ℝ := 7.90
def hat_price : ℝ := 9.13
def scarf_hat_tax_rate : ℝ := 0.065

def total_cost (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

theorem total_spending_calculation :
  total_cost shirt_price shirt_tax_rate +
  total_cost jacket_price jacket_tax_rate +
  total_cost scarf_price scarf_hat_tax_rate +
  total_cost hat_price scarf_hat_tax_rate =
  45.4027 := by sorry

end total_spending_calculation_l2455_245562


namespace absolute_value_equals_sqrt_square_l2455_245529

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end absolute_value_equals_sqrt_square_l2455_245529


namespace decimal_addition_l2455_245568

theorem decimal_addition : (0.0935 : ℚ) + (0.007 : ℚ) + (0.2 : ℚ) = (0.3005 : ℚ) := by
  sorry

end decimal_addition_l2455_245568


namespace tax_free_items_cost_l2455_245521

/-- Calculates the cost of tax-free items given total purchase, sales tax, and tax rate -/
def cost_of_tax_free_items (total_purchase : ℚ) (sales_tax : ℚ) (tax_rate : ℚ) : ℚ :=
  total_purchase - sales_tax / tax_rate

/-- Theorem stating that given the problem conditions, the cost of tax-free items is 20 -/
theorem tax_free_items_cost : 
  let total_purchase : ℚ := 25
  let sales_tax : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100    -- 6%
  cost_of_tax_free_items total_purchase sales_tax tax_rate = 20 := by
  sorry


end tax_free_items_cost_l2455_245521


namespace solution_set_of_equation_l2455_245559

theorem solution_set_of_equation (x y : ℝ) : 
  (|x*y| + |x - y + 1| = 0) ↔ ((x = 0 ∧ y = 1) ∨ (x = -1 ∧ y = 0)) := by
  sorry

end solution_set_of_equation_l2455_245559


namespace minimize_f_minimum_l2455_245573

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

/-- The theorem stating that 82/43 minimizes the minimum value of f(x) -/
theorem minimize_f_minimum (a : ℝ) :
  (∀ x, f (82/43) x ≤ f a x) → a = 82/43 := by
  sorry

end minimize_f_minimum_l2455_245573


namespace square_plus_reciprocal_square_l2455_245595

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3.5) : x^2 + 1/x^2 = 10.25 := by
  sorry

end square_plus_reciprocal_square_l2455_245595


namespace triangle_area_theorem_l2455_245518

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * 3*x = 108 → x = 6 * Real.sqrt 2 := by
  sorry

end triangle_area_theorem_l2455_245518


namespace green_or_purple_probability_l2455_245557

/-- The probability of drawing a green or purple marble from a bag -/
theorem green_or_purple_probability 
  (green : ℕ) (purple : ℕ) (white : ℕ) 
  (h_green : green = 4) 
  (h_purple : purple = 3) 
  (h_white : white = 6) : 
  (green + purple : ℚ) / (green + purple + white) = 7 / 13 := by
  sorry

end green_or_purple_probability_l2455_245557


namespace sixth_term_of_arithmetic_progression_l2455_245547

def arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_progression
  (a : ℕ → ℝ)
  (h_ap : arithmetic_progression a)
  (h_sum : a 1 + a 2 + a 3 = 168)
  (h_diff : a 2 - a 5 = 42) :
  a 6 = 3 := by
sorry

end sixth_term_of_arithmetic_progression_l2455_245547


namespace lakeside_volleyball_club_players_l2455_245516

/-- The number of players in the Lakeside Volleyball Club -/
def num_players : ℕ := 80

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 10

/-- The additional cost of a uniform compared to a pair of shoes in dollars -/
def uniform_additional_cost : ℕ := 15

/-- The total expenditure for all gear in dollars -/
def total_expenditure : ℕ := 5600

/-- Theorem stating that the number of players in the Lakeside Volleyball Club is 80 -/
theorem lakeside_volleyball_club_players :
  num_players = (total_expenditure / (2 * (shoe_cost + (shoe_cost + uniform_additional_cost)))) :=
by sorry

end lakeside_volleyball_club_players_l2455_245516


namespace decimal_to_fraction_l2455_245563

theorem decimal_to_fraction :
  (2.75 : ℚ) = 11 / 4 := by sorry

end decimal_to_fraction_l2455_245563


namespace museum_paintings_l2455_245525

theorem museum_paintings (removed : ℕ) (remaining : ℕ) (initial : ℕ) : 
  removed = 3 → remaining = 95 → initial = remaining + removed → initial = 98 := by
  sorry

end museum_paintings_l2455_245525


namespace seven_thirteenths_repeating_length_l2455_245591

def repeating_decimal_length (n m : ℕ) : ℕ :=
  sorry

theorem seven_thirteenths_repeating_length :
  repeating_decimal_length 7 13 = 6 := by sorry

end seven_thirteenths_repeating_length_l2455_245591


namespace complex_equation_solution_l2455_245556

theorem complex_equation_solution (a : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = -2 * Complex.I → a = -2 := by
  sorry

end complex_equation_solution_l2455_245556


namespace root_sum_reciprocal_l2455_245512

theorem root_sum_reciprocal (p : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁^2 - 6*p*x₁ + p^2 = 0)
  (h2 : x₂^2 - 6*p*x₂ + p^2 = 0)
  (h3 : x₁ ≠ x₂)
  (h4 : p ≠ 0) :
  1 / (x₁ + p) + 1 / (x₂ + p) = 1 / p :=
sorry

end root_sum_reciprocal_l2455_245512


namespace complex_modulus_equality_l2455_245508

theorem complex_modulus_equality (n : ℝ) (hn : 0 < n) :
  Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 → n = 25 := by
  sorry

end complex_modulus_equality_l2455_245508


namespace cube_sum_expression_l2455_245576

theorem cube_sum_expression (x y z w a b c d : ℝ) 
  (hxy : x * y = a)
  (hxz : x * z = b)
  (hyz : y * z = c)
  (hxw : x * w = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + d^3 * b^3) / (a * b * c * d) := by
  sorry

end cube_sum_expression_l2455_245576


namespace triangle_side_bounds_l2455_245523

/-- Given a triangle ABC with side lengths a, b, c forming an arithmetic sequence
    and satisfying a² + b² + c² = 21, prove that √6 < b ≤ √7 -/
theorem triangle_side_bounds (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : ∃ d : ℝ, a = b - d ∧ c = b + d)  -- arithmetic sequence
  (h5 : a^2 + b^2 + c^2 = 21) :
  Real.sqrt 6 < b ∧ b ≤ Real.sqrt 7 := by
  sorry

end triangle_side_bounds_l2455_245523


namespace third_term_5_4_l2455_245550

/-- Decomposition function that returns the n-th term in the decomposition of m^k -/
def decomposition (m : ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  2 * m * k - 1 + 2 * (n - 1)

/-- Theorem stating that the third term in the decomposition of 5^4 is 125 -/
theorem third_term_5_4 : decomposition 5 4 3 = 125 := by
  sorry

end third_term_5_4_l2455_245550


namespace complex_modulus_range_l2455_245540

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5) / 5) ((Real.sqrt 5) / 5) := by sorry

end complex_modulus_range_l2455_245540


namespace polynomial_equality_l2455_245567

theorem polynomial_equality (a b : ℝ) :
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) →
  (a = 1 ∧ b = -6) := by
sorry

end polynomial_equality_l2455_245567


namespace highest_power_of_three_in_M_l2455_245530

def M : ℕ := sorry  -- Definition of M as concatenation of 2-digit integers from 10 to 81

theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), (3^2 ∣ M) ∧ ¬(3^(2+1) ∣ M) :=
sorry

end highest_power_of_three_in_M_l2455_245530


namespace zero_point_in_interval_l2455_245520

noncomputable def f (x : ℝ) : ℝ := 2 * x - 1 + Real.log x / Real.log 2

theorem zero_point_in_interval :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 := by sorry

end zero_point_in_interval_l2455_245520


namespace line_ellipse_intersections_l2455_245566

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The ellipse equation (x-1)^2 + 4y^2 = 4 -/
def ellipse_eq (x y : ℝ) : Prop := (x - 1)^2 + 4 * y^2 = 4

/-- The number of intersections between the line and the ellipse -/
def num_intersections : ℕ := 0

/-- Theorem stating that the number of intersections between the line and the ellipse is 0 -/
theorem line_ellipse_intersections :
  ∀ x y : ℝ, line_eq x y ∧ ellipse_eq x y → num_intersections = 0 :=
by sorry

end line_ellipse_intersections_l2455_245566


namespace cubic_equation_roots_l2455_245535

theorem cubic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ x = 3) →
  (∃ x y : ℝ, x^3 - 6*x^2 + a*x - 6 = 0 ∧ y^3 - 6*y^2 + a*y - 6 = 0 ∧ x = 1 ∧ y = 2) :=
by sorry

end cubic_equation_roots_l2455_245535


namespace cupcakes_for_classes_l2455_245538

/-- The number of fourth-grade classes for which Jessa needs to make cupcakes -/
def num_fourth_grade_classes : ℕ := 3

/-- The number of students in each fourth-grade class -/
def students_per_fourth_grade : ℕ := 30

/-- The number of students in the P.E. class -/
def students_in_pe : ℕ := 50

/-- The total number of cupcakes Jessa needs to make -/
def total_cupcakes : ℕ := 140

theorem cupcakes_for_classes :
  num_fourth_grade_classes * students_per_fourth_grade + students_in_pe = total_cupcakes :=
by sorry

end cupcakes_for_classes_l2455_245538


namespace mail_delivery_l2455_245511

theorem mail_delivery (total : ℕ) (johann : ℕ) (friends : ℕ) : 
  total = 180 → 
  johann = 98 → 
  friends = 2 → 
  (total - johann) % friends = 0 → 
  (total - johann) / friends = 41 :=
by sorry

end mail_delivery_l2455_245511


namespace triangle_properties_l2455_245558

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) (h1 : (t.a + t.c) * Real.sin t.A = Real.sin t.A + Real.sin t.C)
    (h2 : t.c^2 + t.c = t.b^2 - 1) (h3 : t.a = 1) (h4 : t.c = 2) :
    t.B = 2 * Real.pi / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2) := by
  sorry

#check triangle_properties

end triangle_properties_l2455_245558


namespace min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2455_245531

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - a|

-- Theorem for part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 ∧ ∃ y : ℝ, f 1 y = 2 :=
sorry

-- Theorem for part II
theorem range_of_a_when_two_not_in_solution_set :
  ∀ a : ℝ, (f a 2 > 5) ↔ (a < -5/2 ∨ a > 5/2) :=
sorry

end min_value_when_a_is_one_range_of_a_when_two_not_in_solution_set_l2455_245531


namespace prime_square_divisibility_l2455_245545

theorem prime_square_divisibility (p : Nat) (h_prime : Prime p) (h_gt_3 : p > 3) :
  96 ∣ (4 * p^2 - 100) ↔ p^2 % 24 = 25 := by
  sorry

end prime_square_divisibility_l2455_245545


namespace circle_plus_k_circle_plus_k_k_l2455_245599

-- Define the ⊕ operation
def circle_plus (x y : ℝ) : ℝ := x^3 + x - y

-- Theorem statement
theorem circle_plus_k_circle_plus_k_k (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end circle_plus_k_circle_plus_k_k_l2455_245599


namespace number_divided_by_6_multiplied_by_12_l2455_245575

theorem number_divided_by_6_multiplied_by_12 :
  ∃ x : ℝ, (x / 6) * 12 = 15 ∧ x = 7.5 := by
  sorry

end number_divided_by_6_multiplied_by_12_l2455_245575


namespace quadratic_decreasing_l2455_245582

/-- Theorem: For a quadratic function y = ax² + 2ax + c where a < 0,
    and points A(1, y₁) and B(2, y₂) on this function, y₁ - y₂ > 0. -/
theorem quadratic_decreasing (a c y₁ y₂ : ℝ) (ha : a < 0) 
  (h1 : y₁ = a + 2*a + c) 
  (h2 : y₂ = 4*a + 4*a + c) : 
  y₁ - y₂ > 0 := by
  sorry

end quadratic_decreasing_l2455_245582


namespace maple_trees_after_planting_l2455_245510

/-- The number of maple trees in the park after planting -/
def total_maple_trees (initial_maple_trees planted_maple_trees : ℕ) : ℕ :=
  initial_maple_trees + planted_maple_trees

/-- Theorem: The total number of maple trees after planting is equal to
    the sum of the initial number of maple trees and the number of maple trees being planted -/
theorem maple_trees_after_planting 
  (initial_maple_trees planted_maple_trees : ℕ) : 
  total_maple_trees initial_maple_trees planted_maple_trees = 
  initial_maple_trees + planted_maple_trees := by
  sorry

#eval total_maple_trees 2 9

end maple_trees_after_planting_l2455_245510


namespace distance_on_line_l2455_245539

/-- The distance between two points on a line --/
theorem distance_on_line (m k a b c d : ℝ) :
  b = m * a + k →
  d = m * c + k →
  Real.sqrt ((a - c)^2 + (b - d)^2) = |a - c| * Real.sqrt (1 + m^2) := by
  sorry

end distance_on_line_l2455_245539


namespace joan_music_store_spending_l2455_245524

/-- The amount Joan spent at the music store -/
def total_spent (trumpet_cost music_tool_cost song_book_cost : ℚ) : ℚ :=
  trumpet_cost + music_tool_cost + song_book_cost

/-- Proof that Joan spent $163.28 at the music store -/
theorem joan_music_store_spending :
  total_spent 149.16 9.98 4.14 = 163.28 := by
  sorry

end joan_music_store_spending_l2455_245524


namespace initial_water_percentage_l2455_245509

theorem initial_water_percentage
  (V₁ : ℝ) (V₂ : ℝ) (P_f : ℝ)
  (h₁ : V₁ = 20)
  (h₂ : V₂ = 20)
  (h₃ : P_f = 5)
  : ∃ P_i : ℝ, P_i = 10 ∧ (P_i / 100) * V₁ = (P_f / 100) * (V₁ + V₂) := by
  sorry

end initial_water_percentage_l2455_245509


namespace functional_equation_solutions_l2455_245533

theorem functional_equation_solutions (f : ℕ → ℕ) 
  (h : ∀ n m : ℕ, f (3 * n + 2 * m) = f n * f m) : 
  (∀ n, f n = 0) ∨ 
  (∀ n, f n = 1) ∨ 
  ((∀ n, n ≠ 0 → f n = 0) ∧ f 0 = 1) := by
sorry

end functional_equation_solutions_l2455_245533


namespace equation_solutions_l2455_245577

theorem equation_solutions :
  (∃ x : ℝ, 0.5 * x + 1.1 = 6.5 - 1.3 * x ∧ x = 3) ∧
  (∃ x : ℝ, (1/6) * (3 * x - 9) = (2/5) * x - 3 ∧ x = -15) := by
  sorry

end equation_solutions_l2455_245577


namespace point_q_midpoint_l2455_245587

/-- Given five points on a line, prove that Q is the midpoint of A and B -/
theorem point_q_midpoint (O A B C D Q : ℝ) (l m n p : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D →  -- Points are in order
  A - O = l →  -- OA = l
  B - O = m →  -- OB = m
  C - O = n →  -- OC = n
  D - O = p →  -- OD = p
  A ≤ Q ∧ Q ≤ B →  -- Q is between A and B
  (C - Q) / (Q - D) = (B - Q) / (Q - A) →  -- CQ : QD = BQ : QA
  Q - O = (l + m) / 2 :=  -- OQ = (l + m) / 2
by sorry

end point_q_midpoint_l2455_245587


namespace tea_trader_profit_percentage_l2455_245534

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage
  (tea1_weight : ℝ)
  (tea1_cost : ℝ)
  (tea2_weight : ℝ)
  (tea2_cost : ℝ)
  (sale_price : ℝ)
  (h1 : tea1_weight = 80)
  (h2 : tea1_cost = 15)
  (h3 : tea2_weight = 20)
  (h4 : tea2_cost = 20)
  (h5 : sale_price = 20) :
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_weight := tea1_weight + tea2_weight
  let cost_per_kg := total_cost / total_weight
  let profit_per_kg := sale_price - cost_per_kg
  let profit_percentage := (profit_per_kg / cost_per_kg) * 100
  profit_percentage = 25 := by
sorry


end tea_trader_profit_percentage_l2455_245534


namespace susie_score_l2455_245544

/-- Calculates the total score in a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * (correct : ℤ) - (incorrect : ℤ)

/-- Theorem stating that Susie's score in the math contest is 20 points. -/
theorem susie_score : calculateScore 15 10 5 = 20 := by
  sorry

end susie_score_l2455_245544


namespace hyperbola_eccentricity_l2455_245542

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (where a > 0, b > 0),
    if one of its asymptotes is tangent to the curve y = √(x - 1),
    then its eccentricity is √5/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y = b / a * x ∧ y = Real.sqrt (x - 1) ∧
   (∀ (x' y' : ℝ), y' = b / a * x' → y' ≠ Real.sqrt (x' - 1) ∨ (x' = x ∧ y' = y))) →
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 5 / 2 :=
sorry

end hyperbola_eccentricity_l2455_245542


namespace train_crossing_time_l2455_245519

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 1050 →
  train_speed_kmh = 126 →
  (2 * train_length) / (train_speed_kmh * 1000 / 3600) = 60 := by
  sorry

end train_crossing_time_l2455_245519


namespace f_increasing_and_range_l2455_245597

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc 0 1 = Set.Icc 0 (1/3)) := by sorry

end f_increasing_and_range_l2455_245597


namespace brother_siblings_sibling_product_l2455_245537

/-- Represents a family with sisters and brothers -/
structure Family where
  sisters : Nat
  brothers : Nat

/-- Theorem: In a family where one sister has 4 sisters and 6 brothers,
    her brother has 5 sisters and 6 brothers -/
theorem brother_siblings (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s = 5 ∧ b = 6 := by
  sorry

/-- Corollary: The product of the number of sisters and brothers
    that the brother has is 30 -/
theorem sibling_product (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s * b = 30 := by
  sorry

end brother_siblings_sibling_product_l2455_245537


namespace range_of_a_a_upper_bound_range_of_a_characterization_l2455_245532

def A : Set ℝ := {x | x - 1 > 0}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) : (A ∩ B a).Nonempty → a > 1 := by sorry

theorem a_upper_bound : ∀ a : ℝ, a > 1 → (A ∩ B a).Nonempty := by sorry

theorem range_of_a_characterization :
  ∀ a : ℝ, (A ∩ B a).Nonempty ↔ a > 1 := by sorry

end range_of_a_a_upper_bound_range_of_a_characterization_l2455_245532


namespace target_practice_probabilities_l2455_245570

/-- Represents a shooter in the target practice scenario -/
structure Shooter where
  hit_probability : ℝ
  num_shots : ℕ

/-- Calculates the probability of the given event -/
def calculate_probability (s1 s2 : Shooter) (event : Shooter → Shooter → ℝ) : ℝ :=
  event s1 s2

/-- The scenario with two shooters -/
def target_practice_scenario : Prop :=
  ∃ (s1 s2 : Shooter),
    s1.hit_probability = 0.8 ∧
    s2.hit_probability = 0.6 ∧
    s1.num_shots = 2 ∧
    s2.num_shots = 3 ∧
    (calculate_probability s1 s2 (λ _ _ => 0.99744) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability)^s1.num_shots * (1 - s2.hit_probability)^s2.num_shots)) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.13824) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)) * 
                                             (Nat.choose s2.num_shots 2 * s2.hit_probability^2 * (1 - s2.hit_probability)))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.87328) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability^2) * 
                                             (1 - s2.hit_probability^2 - s2.hit_probability^3))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.032) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)^(s1.num_shots - 1) * (1 - s2.hit_probability)^s2.num_shots) + 
                                             ((1 - s1.hit_probability)^s1.num_shots * s2.num_shots * s2.hit_probability * (1 - s2.hit_probability)^(s2.num_shots - 1))))

theorem target_practice_probabilities : target_practice_scenario := sorry

end target_practice_probabilities_l2455_245570


namespace parabola_translation_l2455_245580

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 2 1
  translated = Parabola.mk 1 (-4) 5  -- y = (x-2)^2 + 1
  := by sorry

end parabola_translation_l2455_245580


namespace interest_difference_theorem_l2455_245551

/-- The difference between compound and simple interest over 2 years at 10% per annum -/
def interestDifference (P : ℝ) : ℝ :=
  P * ((1 + 0.1)^2 - 1) - P * 0.1 * 2

/-- The problem statement -/
theorem interest_difference_theorem (P : ℝ) :
  interestDifference P = 18 → P = 1800 := by
  sorry

end interest_difference_theorem_l2455_245551


namespace binomial_difference_squares_l2455_245504

theorem binomial_difference_squares (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end binomial_difference_squares_l2455_245504


namespace max_constant_term_l2455_245552

def p₁ (a : ℝ) (x : ℝ) : ℝ := x - a

def p₂ (r s t : ℕ) (x : ℝ) : ℝ := (x - 1) ^ r * (x - 2) ^ s * (x + 3) ^ t

def constant_term (a : ℝ) (r s t : ℕ) : ℝ :=
  (-1) ^ (r + s) * 2 ^ s * 3 ^ t - a

theorem max_constant_term :
  ∀ a : ℝ, ∀ r s t : ℕ,
    r ≥ 1 → s ≥ 1 → t ≥ 1 → r + s + t = 4 →
    constant_term a r s t ≤ 21 ∧
    (constant_term (-3) 1 1 2 = 21) :=
sorry

end max_constant_term_l2455_245552


namespace elevation_equals_depression_l2455_245500

/-- The elevation angle from point a to point b -/
def elevation_angle (a b : Point) : ℝ := sorry

/-- The depression angle from point b to point a -/
def depression_angle (b a : Point) : ℝ := sorry

/-- Theorem stating that the elevation angle from a to b equals the depression angle from b to a -/
theorem elevation_equals_depression (a b : Point) :
  elevation_angle a b = depression_angle b a := by sorry

end elevation_equals_depression_l2455_245500


namespace kishore_saved_ten_percent_l2455_245513

/-- Represents Mr. Kishore's financial situation --/
structure KishoreFinances where
  rent : ℕ
  milk : ℕ
  groceries : ℕ
  education : ℕ
  petrol : ℕ
  miscellaneous : ℕ
  savings : ℕ

/-- Calculates the total expenses --/
def totalExpenses (k : KishoreFinances) : ℕ :=
  k.rent + k.milk + k.groceries + k.education + k.petrol + k.miscellaneous

/-- Calculates the total monthly salary --/
def totalSalary (k : KishoreFinances) : ℕ :=
  totalExpenses k + k.savings

/-- Calculates the percentage saved --/
def percentageSaved (k : KishoreFinances) : ℚ :=
  (k.savings : ℚ) / (totalSalary k : ℚ) * 100

/-- Theorem: Mr. Kishore saved 10% of his monthly salary --/
theorem kishore_saved_ten_percent (k : KishoreFinances)
    (h1 : k.rent = 5000)
    (h2 : k.milk = 1500)
    (h3 : k.groceries = 4500)
    (h4 : k.education = 2500)
    (h5 : k.petrol = 2000)
    (h6 : k.miscellaneous = 3940)
    (h7 : k.savings = 2160) :
    percentageSaved k = 10 := by
  sorry

end kishore_saved_ten_percent_l2455_245513


namespace fraction_inequality_l2455_245527

theorem fraction_inequality (a b c d e : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) 
  (h5 : e < 0) : 
  e / (a - c)^2 > e / (b - d)^2 := by
  sorry

end fraction_inequality_l2455_245527


namespace decimal_point_shift_l2455_245502

theorem decimal_point_shift (x : ℝ) :
  (x * 10 = 760.8) → (x = 76.08) :=
by sorry

end decimal_point_shift_l2455_245502


namespace circle_radius_comparison_l2455_245555

-- Define the structure for a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of three circles being pairwise disjoint and collinear
def pairwiseDisjointCollinear (c1 c2 c3 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (x2 - x1) * (y3 - y1) = (y2 - y1) * (x3 - x1) ∧ 
  (c1.radius + c2.radius < Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)) ∧
  (c2.radius + c3.radius < Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)) ∧
  (c3.radius + c1.radius < Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2))

-- Define the property of a circle touching three other circles externally
def touchesExternally (c : Circle) (c1 c2 c3 : Circle) : Prop :=
  let (x, y) := c.center
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let (x3, y3) := c3.center
  (Real.sqrt ((x - x1)^2 + (y - y1)^2) = c.radius + c1.radius) ∧
  (Real.sqrt ((x - x2)^2 + (y - y2)^2) = c.radius + c2.radius) ∧
  (Real.sqrt ((x - x3)^2 + (y - y3)^2) = c.radius + c3.radius)

-- The main theorem
theorem circle_radius_comparison 
  (c1 c2 c3 c : Circle) 
  (h1 : pairwiseDisjointCollinear c1 c2 c3)
  (h2 : touchesExternally c c1 c2 c3) :
  c.radius > c2.radius :=
sorry

end circle_radius_comparison_l2455_245555


namespace expenditure_ratio_proof_l2455_245585

/-- Represents the financial data of a person -/
structure PersonFinance where
  income : ℕ
  savings : ℕ
  expenditure : ℕ

/-- The problem statement -/
theorem expenditure_ratio_proof 
  (p1 p2 : PersonFinance)
  (h1 : p1.income = 3000)
  (h2 : p1.income * 4 = p2.income * 5)
  (h3 : p1.savings = 1200)
  (h4 : p2.savings = 1200)
  (h5 : p1.expenditure = p1.income - p1.savings)
  (h6 : p2.expenditure = p2.income - p2.savings)
  : p1.expenditure * 2 = p2.expenditure * 3 := by
  sorry

end expenditure_ratio_proof_l2455_245585


namespace work_completion_time_l2455_245584

/-- Given workers a, b, and c who can complete a work in 16, x, and 12 days respectively,
    and together they complete the work in 3.2 days, prove that x = 6. -/
theorem work_completion_time (x : ℝ) 
  (h1 : 1/16 + 1/x + 1/12 = 1/3.2) : x = 6 := by
  sorry

end work_completion_time_l2455_245584


namespace max_abc_value_l2455_245579

theorem max_abc_value (a b c : ℝ) (sum_eq : a + b + c = 5) (prod_sum_eq : a * b + b * c + c * a = 7) :
  ∀ x y z : ℝ, x + y + z = 5 → x * y + y * z + z * x = 7 → a * b * c ≥ x * y * z ∧ ∃ p q r : ℝ, p + q + r = 5 ∧ p * q + q * r + r * p = 7 ∧ p * q * r = 3 :=
sorry

end max_abc_value_l2455_245579


namespace solution_set_implies_sum_l2455_245598

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, 1 < x ∧ x < b ↔ f a x < 0) →
  a + b = 3 := by
sorry

end solution_set_implies_sum_l2455_245598


namespace range_of_m_l2455_245536

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x - floor x}

-- Define set B
def B (m : ℝ) : Set ℝ := {y | 0 ≤ y ∧ y ≤ m}

-- State the theorem
theorem range_of_m (m : ℝ) :
  (A ⊂ B m) ↔ m ∈ Set.Ici 1 :=
sorry

end range_of_m_l2455_245536


namespace complex_number_simplification_l2455_245581

theorem complex_number_simplification :
  (-5 - 3 * Complex.I) * 2 - (2 + 5 * Complex.I) = -12 - 11 * Complex.I := by
  sorry

end complex_number_simplification_l2455_245581


namespace upstream_speed_l2455_245592

/-- 
Given a man's rowing speed in still water and his speed downstream, 
this theorem proves that his speed upstream can be calculated as the 
difference between his speed in still water and half the difference 
between his downstream speed and his speed in still water.
-/
theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0) 
  (h2 : speed_downstream > speed_still) : 
  speed_still - (speed_downstream - speed_still) / 2 = 
  speed_still - (speed_downstream - speed_still) / 2 :=
by sorry

end upstream_speed_l2455_245592


namespace pentagon_sum_l2455_245503

/-- Given integers u and v with 0 < v < u, and points A, B, C, D, E defined as follows:
    A = (u, v)
    B is the reflection of A across y = -x
    C is the reflection of B across the y-axis
    D is the reflection of C across the x-axis
    E is the reflection of D across the y-axis
    If the area of pentagon ABCDE is 500, then u + v = 21 -/
theorem pentagon_sum (u v : ℤ) (hu : u > 0) (hv : v > 0) (huv : u > v)
  (harea : 6 * u * v - 2 * v^2 = 500) : u + v = 21 := by
  sorry

end pentagon_sum_l2455_245503


namespace divisibility_theorem_l2455_245588

theorem divisibility_theorem (n : ℕ) (h : ∃ m : ℕ, 2^n - 2 = n * m) :
  ∃ k : ℕ, 2^(2^n - 1) - 2 = (2^n - 1) * k := by
  sorry

end divisibility_theorem_l2455_245588


namespace negation_of_existence_quadratic_inequality_negation_l2455_245564

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) := by sorry

end negation_of_existence_quadratic_inequality_negation_l2455_245564


namespace derivative_of_f_l2455_245572

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 2 * x - 1 := by sorry

end derivative_of_f_l2455_245572


namespace apple_picking_problem_l2455_245569

theorem apple_picking_problem (x : ℝ) : 
  x + (3/4) * x + 600 = 2600 → x = 1142 := by
  sorry

end apple_picking_problem_l2455_245569


namespace quadratic_root_difference_l2455_245514

theorem quadratic_root_difference (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + 12 = 0 ∧ 
                x₂^2 + p*x₂ + 12 = 0 ∧ 
                x₁ - x₂ = 1) → 
  p = 7 ∨ p = -7 :=
by sorry

end quadratic_root_difference_l2455_245514


namespace shaded_area_square_with_quarter_circles_l2455_245571

/-- The area of the shaded region formed by a square with side length 10 cm
    and four quarter circles drawn at its corners is equal to 100 - 25π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_radius : ℝ := square_side / 2
  let full_circle_area : ℝ := π * quarter_circle_radius ^ 2
  let shaded_area : ℝ := square_area - full_circle_area
  shaded_area = 100 - 25 * π := by sorry

end shaded_area_square_with_quarter_circles_l2455_245571


namespace greatest_divisor_with_remainders_l2455_245590

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end greatest_divisor_with_remainders_l2455_245590


namespace solution_to_equation_l2455_245507

theorem solution_to_equation (x y : ℝ) : 
  (x - 7)^2 + (y - 8)^2 + (x - y)^2 = 1/3 ↔ x = 7 + 1/3 ∧ y = 7 + 2/3 := by
  sorry

end solution_to_equation_l2455_245507


namespace notebook_cost_l2455_245574

theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (h1 : notebook_cost + pencil_cost = 2.20)
  (h2 : notebook_cost = pencil_cost + 2) : 
  notebook_cost = 2.10 := by
  sorry

end notebook_cost_l2455_245574


namespace admission_ratio_theorem_l2455_245578

def admission_problem (a c : ℕ+) : Prop :=
  30 * a.val + 15 * c.val = 2550

def ratio_closest_to_one (a c : ℕ+) : Prop :=
  ∀ (x y : ℕ+), admission_problem x y →
    |((a:ℚ) / c) - 1| ≤ |((x:ℚ) / y) - 1|

theorem admission_ratio_theorem :
  ∃ (a c : ℕ+), admission_problem a c ∧ ratio_closest_to_one a c ∧ a.val = 57 ∧ c.val = 56 :=
sorry

end admission_ratio_theorem_l2455_245578


namespace ratio_lcm_problem_l2455_245589

theorem ratio_lcm_problem (a b x : ℕ+) (h_ratio : a.val * x.val = 8 * b.val) 
  (h_lcm : Nat.lcm a.val b.val = 432) (h_a : a = 48) : b = 72 := by
  sorry

end ratio_lcm_problem_l2455_245589


namespace max_earnings_l2455_245522

def max_work_hours : ℕ := 80
def regular_hours : ℕ := 20
def regular_wage : ℚ := 8
def regular_tips : ℚ := 2
def overtime_wage_multiplier : ℚ := 1.25
def overtime_tips : ℚ := 3
def bonus_per_5_hours : ℚ := 20

def overtime_hours : ℕ := max_work_hours - regular_hours

def regular_earnings : ℚ := regular_hours * (regular_wage + regular_tips)
def overtime_wage : ℚ := regular_wage * overtime_wage_multiplier
def overtime_earnings : ℚ := overtime_hours * (overtime_wage + overtime_tips)
def bonus : ℚ := (overtime_hours / 5) * bonus_per_5_hours

def total_earnings : ℚ := regular_earnings + overtime_earnings + bonus

theorem max_earnings :
  total_earnings = 1220 := by sorry

end max_earnings_l2455_245522


namespace difference_of_squares_l2455_245528

theorem difference_of_squares : 550^2 - 450^2 = 100000 := by
  sorry

end difference_of_squares_l2455_245528


namespace license_plate_palindrome_probability_final_probability_sum_l2455_245505

/-- Probability of a palindrome in a four-letter sequence -/
def letter_palindrome_prob : ℚ := 1 / 676

/-- Probability of a palindrome in a four-digit sequence -/
def digit_palindrome_prob : ℚ := 1 / 100

/-- Total number of possible license plates -/
def total_plates : ℕ := 26^4 * 10^4

/-- Number of favorable outcomes (license plates with at least one palindrome) -/
def favorable_outcomes : ℕ := 155

/-- Denominator of the final probability fraction -/
def prob_denominator : ℕ := 13520

theorem license_plate_palindrome_probability :
  (favorable_outcomes : ℚ) / prob_denominator =
  letter_palindrome_prob + digit_palindrome_prob - letter_palindrome_prob * digit_palindrome_prob :=
by sorry

theorem final_probability_sum :
  favorable_outcomes + prob_denominator = 13675 :=
by sorry

end license_plate_palindrome_probability_final_probability_sum_l2455_245505
