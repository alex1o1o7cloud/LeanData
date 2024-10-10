import Mathlib

namespace inequality_system_solution_l3158_315805

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  x + 1 < 4 ∧ 1 - 3*x ≥ -5

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x ≤ 2}

-- Theorem statement
theorem inequality_system_solution :
  {x : ℝ | inequality_system x} = solution_set :=
sorry

end inequality_system_solution_l3158_315805


namespace square_root_divided_by_13_equals_4_l3158_315819

theorem square_root_divided_by_13_equals_4 :
  ∃ x : ℝ, (Real.sqrt x) / 13 = 4 ∧ x = 2704 := by
  sorry

end square_root_divided_by_13_equals_4_l3158_315819


namespace chair_cost_l3158_315815

/-- Given that Ellen spent $180 for 12 chairs, prove that each chair costs $15. -/
theorem chair_cost (total_spent : ℕ) (num_chairs : ℕ) (h1 : total_spent = 180) (h2 : num_chairs = 12) :
  total_spent / num_chairs = 15 := by
sorry

end chair_cost_l3158_315815


namespace omega_value_l3158_315804

/-- Given a function f(x) = sin(ωx) + cos(ωx) where ω > 0 and x ∈ ℝ,
    if f(x) is monotonically increasing on (-ω, ω) and
    the graph of y = f(x) is symmetric with respect to x = ω,
    then ω = √π / 2 -/
theorem omega_value (ω : ℝ) (h_pos : ω > 0) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x) + Real.cos (ω * x)
  (∀ x ∈ Set.Ioo (-ω) ω, Monotone f) →
  (∀ x : ℝ, f (ω + x) = f (ω - x)) →
  ω = Real.sqrt π / 2 := by
  sorry

end omega_value_l3158_315804


namespace total_red_peaches_l3158_315883

-- Define the number of baskets
def num_baskets : ℕ := 6

-- Define the number of red peaches per basket
def red_peaches_per_basket : ℕ := 16

-- Theorem to prove
theorem total_red_peaches : 
  num_baskets * red_peaches_per_basket = 96 := by
  sorry

end total_red_peaches_l3158_315883


namespace p_minimum_value_l3158_315829

/-- The quadratic function p in terms of a and b -/
def p (a b : ℝ) : ℝ := 2*a^2 - 8*a*b + 17*b^2 - 16*a - 4*b + 2044

/-- The theorem stating the minimum value of p and the values of a and b at which it occurs -/
theorem p_minimum_value :
  ∃ (a b : ℝ), p a b = 1976 ∧ 
  (∀ (x y : ℝ), p x y ≥ 1976) ∧
  a = 2*b + 4 ∧ b = 2 := by sorry

end p_minimum_value_l3158_315829


namespace odd_function_derivative_even_l3158_315896

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Theorem statement
theorem odd_function_derivative_even
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hodd : odd_function f) :
  even_function (deriv f) :=
sorry

end odd_function_derivative_even_l3158_315896


namespace divisor_ratio_of_M_l3158_315894

def M : ℕ := 126 * 36 * 187

/-- Sum of odd divisors of a natural number -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- Sum of even divisors of a natural number -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

/-- The ratio of sum of even divisors to sum of odd divisors -/
def divisor_ratio (n : ℕ) : ℚ :=
  (sum_even_divisors n : ℚ) / (sum_odd_divisors n : ℚ)

theorem divisor_ratio_of_M :
  divisor_ratio M = 14 := by sorry

end divisor_ratio_of_M_l3158_315894


namespace nina_money_proof_l3158_315862

/-- The amount of money Nina has -/
def nina_money : ℕ := 48

/-- The original cost of each widget -/
def original_cost : ℕ := 8

/-- The reduced cost of each widget -/
def reduced_cost : ℕ := original_cost - 2

theorem nina_money_proof :
  (nina_money = 6 * original_cost) ∧
  (nina_money = 8 * reduced_cost) :=
by sorry

end nina_money_proof_l3158_315862


namespace expected_digits_is_31_20_l3158_315801

/-- A fair 20-sided die with numbers 1 through 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits for a given number on the die -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum (λ i => numDigits (i + 1))) / icosahedralDie.card

theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end expected_digits_is_31_20_l3158_315801


namespace natasha_money_l3158_315827

/-- Represents the amount of money each person has -/
structure Money where
  cosima : ℕ
  carla : ℕ
  natasha : ℕ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.carla = 2 * m.cosima ∧
  m.natasha = 3 * m.carla ∧
  (7 * (m.cosima + m.carla + m.natasha) - 5 * (m.cosima + m.carla + m.natasha)) = 180

/-- The theorem to prove -/
theorem natasha_money (m : Money) : 
  problem_conditions m → m.natasha = 60 :=
by
  sorry


end natasha_money_l3158_315827


namespace classroom_addition_problem_l3158_315858

theorem classroom_addition_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 6) (h3 : x * y = 45) : 
  x = 11 ∧ y = 5 := by
sorry

end classroom_addition_problem_l3158_315858


namespace x_over_y_value_l3158_315817

theorem x_over_y_value (x y a b : ℝ) 
  (h1 : (2 * a - x) / (3 * b - y) = 3)
  (h2 : a / b = 4.5) :
  x / y = 3 := by
sorry

end x_over_y_value_l3158_315817


namespace square_sum_101_99_l3158_315880

theorem square_sum_101_99 : 101 * 101 + 99 * 99 = 20200 := by
  sorry

end square_sum_101_99_l3158_315880


namespace expression_evaluation_l3158_315882

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (x - 3)^2 + (x + 3)*(x - 3) + 2*x*(2 - x) = -1 := by sorry

end expression_evaluation_l3158_315882


namespace diamond_equation_solution_l3158_315873

/-- Definition of the diamond operation -/
def diamond (A B : ℝ) : ℝ := 4 * A + 3 * B + 7

/-- Theorem stating the solution to the equation A ◇ 5 = 71 -/
theorem diamond_equation_solution :
  ∃ A : ℝ, diamond A 5 = 71 ∧ A = 12.25 := by
  sorry

end diamond_equation_solution_l3158_315873


namespace lydia_plant_count_l3158_315847

/-- Represents the total number of plants Lydia has -/
def total_plants : ℕ := sorry

/-- Represents the number of flowering plants Lydia has -/
def flowering_plants : ℕ := sorry

/-- Represents the number of flowering plants on the porch -/
def porch_plants : ℕ := sorry

/-- The percentage of flowering plants among all plants -/
def flowering_percentage : ℚ := 2/5

/-- The fraction of flowering plants on the porch -/
def porch_fraction : ℚ := 1/4

/-- The number of flowers each flowering plant produces -/
def flowers_per_plant : ℕ := 5

/-- The total number of flowers on the porch -/
def total_porch_flowers : ℕ := 40

theorem lydia_plant_count :
  (flowering_plants = flowering_percentage * total_plants) ∧
  (porch_plants = porch_fraction * flowering_plants) ∧
  (total_porch_flowers = porch_plants * flowers_per_plant) →
  total_plants = 80 := by sorry

end lydia_plant_count_l3158_315847


namespace sqrt_sum_approximation_l3158_315893

theorem sqrt_sum_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |((Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 1.00) / (Real.sqrt 0.49)) - 2.6507| < ε :=
sorry

end sqrt_sum_approximation_l3158_315893


namespace quadratic_sets_l3158_315867

/-- A quadratic function with a minimum value -/
structure QuadraticWithMinimum where
  a : ℝ
  f : ℝ → ℝ
  hf : f = fun x ↦ a * x^2 + x
  ha : a > 0

/-- The set A where f(x) < 0 -/
def setA (q : QuadraticWithMinimum) : Set ℝ :=
  {x | q.f x < 0}

/-- The set B defined by |x+4| < a -/
def setB (a : ℝ) : Set ℝ :=
  {x | |x + 4| < a}

/-- The main theorem -/
theorem quadratic_sets (q : QuadraticWithMinimum) :
  (setA q = Set.Ioo (-1 / q.a) 0) ∧
  (setB q.a ⊆ setA q ↔ 0 < q.a ∧ q.a ≤ Real.sqrt 5 - 2) := by
  sorry

end quadratic_sets_l3158_315867


namespace at_least_one_greater_than_one_l3158_315839

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by sorry

end at_least_one_greater_than_one_l3158_315839


namespace dot_product_from_norms_l3158_315875

theorem dot_product_from_norms (a b : ℝ × ℝ) :
  ‖a + b‖ = Real.sqrt 10 → ‖a - b‖ = Real.sqrt 6 → a • b = 1 := by
  sorry

end dot_product_from_norms_l3158_315875


namespace remainder_theorem_l3158_315860

-- Define the polynomial x^3 + x^2 + x + 1
def f (x : ℂ) : ℂ := x^3 + x^2 + x + 1

-- Define the polynomial x^60 + x^45 + x^30 + x^15 + 1
def g (x : ℂ) : ℂ := x^60 + x^45 + x^30 + x^15 + 1

theorem remainder_theorem :
  ∃ (q : ℂ → ℂ), ∀ x, g x = f x * q x + 5 :=
sorry

end remainder_theorem_l3158_315860


namespace sara_balloons_count_l3158_315877

/-- The number of yellow balloons that Tom has -/
def tom_balloons : ℕ := 9

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 17

/-- The number of yellow balloons that Sara has -/
def sara_balloons : ℕ := total_balloons - tom_balloons

theorem sara_balloons_count : sara_balloons = 8 := by
  sorry

end sara_balloons_count_l3158_315877


namespace good_number_theorem_l3158_315831

def is_good (n : ℕ) : Prop :=
  ∃ (k₁ k₂ k₃ k₄ : ℕ), 
    k₁ > 0 ∧ k₂ > 0 ∧ k₃ > 0 ∧ k₄ > 0 ∧
    k₁ ≠ k₂ ∧ k₁ ≠ k₃ ∧ k₁ ≠ k₄ ∧ k₂ ≠ k₃ ∧ k₂ ≠ k₄ ∧ k₃ ≠ k₄ ∧
    (n + k₁ ∣ n + k₁^2) ∧ (n + k₂ ∣ n + k₂^2) ∧ (n + k₃ ∣ n + k₃^2) ∧ (n + k₄ ∣ n + k₄^2) ∧
    ∀ (k : ℕ), k > 0 ∧ k ≠ k₁ ∧ k ≠ k₂ ∧ k ≠ k₃ ∧ k ≠ k₄ → ¬(n + k ∣ n + k^2)

theorem good_number_theorem :
  is_good 58 ∧ 
  ∀ (p : ℕ), p > 2 → (is_good (2 * p) ↔ Nat.Prime p ∧ Nat.Prime (2 * p + 1)) :=
sorry

end good_number_theorem_l3158_315831


namespace cubic_root_property_l3158_315810

/-- Given a cubic polynomial with roots α, β, and γ, 
    there exist constants A, B, and C such that Aα² + Bα + C = β or γ -/
theorem cubic_root_property (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  ∃ A B C : ℝ, A*α^2 + B*α + C = β ∨ A*α^2 + B*α + C = γ := by
  sorry

end cubic_root_property_l3158_315810


namespace tan_alpha_minus_pi_fourth_l3158_315874

theorem tan_alpha_minus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β + π/4) = 1/4) :
  Real.tan (α - π/4) = 3/22 := by
  sorry

end tan_alpha_minus_pi_fourth_l3158_315874


namespace equation_solution_l3158_315814

theorem equation_solution (x : ℝ) (h1 : x^2 + x ≠ 0) (h2 : x + 1 ≠ 0) :
  (2 - 1 / (x^2 + x) = (2*x + 1) / (x + 1)) ↔ x = 1 := by
  sorry

end equation_solution_l3158_315814


namespace line_plane_infinite_intersection_l3158_315885

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- A point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk 
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

/-- A point lies on a plane -/
def pointOnPlane (p : Point3D) (pl : Plane3D) : Prop :=
  (p.x - pl.point.x) * pl.normal.x +
  (p.y - pl.point.y) * pl.normal.y +
  (p.z - pl.point.z) * pl.normal.z = 0

/-- The theorem: There exists a line and a plane that have infinite common points -/
theorem line_plane_infinite_intersection :
  ∃ (l : Line3D) (pl : Plane3D), ∀ (p : Point3D), 
    pointOnLine p l → pointOnPlane p pl :=
sorry

end line_plane_infinite_intersection_l3158_315885


namespace smallest_dual_base_representation_l3158_315818

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 3 ∧ b > 3 ∧ 
    2 * a + 3 = n ∧ 
    3 * b + 2 = n

theorem smallest_dual_base_representation :
  (is_valid_representation 17) ∧ 
  (∀ m : ℕ, m < 17 → ¬(is_valid_representation m)) :=
sorry

end smallest_dual_base_representation_l3158_315818


namespace h_max_at_3_l3158_315826

/-- Linear function f(x) = -2x + 8 -/
def f (x : ℝ) : ℝ := -2 * x + 8

/-- Linear function g(x) = 2x - 4 -/
def g (x : ℝ) : ℝ := 2 * x - 4

/-- Product function h(x) = f(x) * g(x) -/
def h (x : ℝ) : ℝ := f x * g x

/-- Theorem stating that h(x) reaches its maximum at x = 3 -/
theorem h_max_at_3 : ∀ x : ℝ, h x ≤ h 3 := by sorry

end h_max_at_3_l3158_315826


namespace quadratic_sum_l3158_315879

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 5 * x^2 - 10 * x - 3 = a * (x - h)^2 + k) → 
  a + h + k = -2 := by
sorry

end quadratic_sum_l3158_315879


namespace smallest_d_value_l3158_315855

theorem smallest_d_value (d : ℝ) : 
  (5 * Real.sqrt 5)^2 + (d + 4)^2 = (5 * d)^2 → d ≥ (1 + Real.sqrt 212.5) / 6 := by
  sorry

end smallest_d_value_l3158_315855


namespace y_value_l3158_315871

theorem y_value (x y : ℝ) (h1 : x^3 - x - 2 = y + 2) (h2 : x = 3) : y = 20 := by
  sorry

end y_value_l3158_315871


namespace hyperbola_m_range_l3158_315836

/-- Given a hyperbola with equation x²/m + y²/(2m-1) = 1, prove that the range of m is 0 < m < 1/2 -/
theorem hyperbola_m_range (m : ℝ) : 
  (∃ x y : ℝ, x^2/m + y^2/(2*m-1) = 1) → 0 < m ∧ m < 1/2 := by
  sorry

end hyperbola_m_range_l3158_315836


namespace lucky_larry_problem_l3158_315837

theorem lucky_larry_problem (a b c d e : ℝ) : 
  a = 12 ∧ b = 3 ∧ c = 15 ∧ d = 2 →
  (a / b - c - d * e = a / (b - (c - (d * e)))) →
  e = 4 := by
  sorry

end lucky_larry_problem_l3158_315837


namespace expr_D_not_fraction_l3158_315803

-- Define what a fraction is
def is_fraction (expr : ℚ → ℚ) : Prop :=
  ∃ (f g : ℚ → ℚ), ∀ x, expr x = (f x) / (g x) ∧ g x ≠ 0

-- Define the expressions
def expr_A (x : ℚ) : ℚ := 1 / (x^2)
def expr_B (a b : ℚ) : ℚ := (b + 3) / a
def expr_C (x : ℚ) : ℚ := (x^2 - 1) / (x + 1)
def expr_D (a : ℚ) : ℚ := (2 / 7) * a

-- Theorem stating that expr_D is not a fraction
theorem expr_D_not_fraction : ¬ is_fraction expr_D :=
sorry

end expr_D_not_fraction_l3158_315803


namespace solution_characterization_l3158_315848

theorem solution_characterization (x y z : ℝ) 
  (h1 : x + y + z = 1/x + 1/y + 1/z) 
  (h2 : x^2 + y^2 + z^2 = 1/x^2 + 1/y^2 + 1/z^2) :
  ∃ (e : ℝ) (t : ℝ), e = 1 ∨ e = -1 ∧ t ≠ 0 ∧ 
    ((x = e ∧ y = t ∧ z = 1/t) ∨ 
     (x = e ∧ y = 1/t ∧ z = t) ∨ 
     (x = t ∧ y = e ∧ z = 1/t) ∨ 
     (x = t ∧ y = 1/t ∧ z = e) ∨ 
     (x = 1/t ∧ y = e ∧ z = t) ∨ 
     (x = 1/t ∧ y = t ∧ z = e)) :=
by sorry

end solution_characterization_l3158_315848


namespace paint_area_calculation_l3158_315887

/-- Calculates the area to be painted on a wall with given dimensions and openings. -/
def areaToPaint (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) : ℝ :=
  let wallArea := wallHeight * wallWidth
  let windowArea := windowHeight * windowWidth
  let doorArea := doorHeight * doorWidth
  wallArea - windowArea - doorArea

/-- Theorem stating that the area to be painted on the given wall is 128.5 square feet. -/
theorem paint_area_calculation :
  areaToPaint 10 15 3 5 1 6.5 = 128.5 := by
  sorry

end paint_area_calculation_l3158_315887


namespace close_interval_for_m_and_n_l3158_315833

-- Define the functions m and n
def m (x : ℝ) := x^2 - 3*x + 4
def n (x : ℝ) := 2*x - 3

-- Define what it means for two functions to be close on an interval
def are_close (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

-- Theorem statement
theorem close_interval_for_m_and_n :
  are_close m n 2 3 :=
sorry

end close_interval_for_m_and_n_l3158_315833


namespace first_square_covering_all_rows_l3158_315812

-- Define a function to calculate the row number of a square
def row_number (n : ℕ) : ℕ := (n - 1) / 10 + 1

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Theorem statement
theorem first_square_covering_all_rows :
  (∀ r : ℕ, r ≥ 1 → r ≤ 10 → ∃ n : ℕ, n ≤ 100 ∧ is_perfect_square n ∧ row_number n = r) ∧
  (∀ m : ℕ, m < 100 → ¬(∀ r : ℕ, r ≥ 1 → r ≤ 10 → ∃ n : ℕ, n ≤ m ∧ is_perfect_square n ∧ row_number n = r)) :=
by sorry

end first_square_covering_all_rows_l3158_315812


namespace light_bulb_probabilities_l3158_315878

/-- Represents the number of light bulbs in the box -/
def total_bulbs : ℕ := 6

/-- Represents the number of defective light bulbs -/
def defective_bulbs : ℕ := 2

/-- Represents the number of good light bulbs -/
def good_bulbs : ℕ := 4

/-- Represents the number of bulbs selected -/
def selected_bulbs : ℕ := 2

/-- Calculates the probability of selecting two defective bulbs -/
def prob_two_defective : ℚ := 1 / 15

/-- Calculates the probability of selecting exactly one defective bulb -/
def prob_one_defective : ℚ := 8 / 15

theorem light_bulb_probabilities :
  (total_bulbs = defective_bulbs + good_bulbs) ∧
  (selected_bulbs = 2) →
  (prob_two_defective = 1 / 15) ∧
  (prob_one_defective = 8 / 15) := by
  sorry

end light_bulb_probabilities_l3158_315878


namespace either_odd_or_even_l3158_315865

theorem either_odd_or_even (n : ℤ) : 
  (Odd (2*n - 1)) ∨ (Even (2*n + 1)) := by
  sorry

end either_odd_or_even_l3158_315865


namespace jade_handled_81_transactions_l3158_315844

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90

def anthony_transactions : ℕ := mabel_transactions + mabel_transactions / 10

def cal_transactions : ℕ := anthony_transactions * 2 / 3

def jade_transactions : ℕ := cal_transactions + 15

-- Theorem to prove
theorem jade_handled_81_transactions : jade_transactions = 81 := by
  sorry

end jade_handled_81_transactions_l3158_315844


namespace circle_tangent_to_line_l3158_315856

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

-- Define the tangent point
def tangent_point : ℝ × ℝ := (2, -1)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 9

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ (x y : ℝ),
  line x y →
  (∃ (t : ℝ), (x, y) = (2 + t * 3, -1 - t * 4)) →
  circle_equation x y :=
sorry

end circle_tangent_to_line_l3158_315856


namespace marble_game_probability_l3158_315892

theorem marble_game_probability (B R : ℕ) : 
  B + R = 21 →
  (B : ℚ) / 21 * ((B - 1) : ℚ) / 20 = 1 / 2 →
  B^2 + R^2 = 261 := by
sorry

end marble_game_probability_l3158_315892


namespace circle_equation_l3158_315859

/-- The equation of a circle passing through points A(1, -1) and B(-1, 1) with its center on the line x + y - 2 = 0 -/
theorem circle_equation :
  ∃ (h k : ℝ),
    (h + k - 2 = 0) ∧
    ((1 - h)^2 + (-1 - k)^2 = (h - 1)^2 + (k - 1)^2) ∧
    ((1 - h)^2 + (-1 - k)^2 = 4) ∧
    (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 4 ↔ (x - 1)^2 + (y - 1)^2 = 4) :=
sorry

end circle_equation_l3158_315859


namespace fashion_show_runway_time_l3158_315816

/-- Fashion Show Runway Time Calculation -/
theorem fashion_show_runway_time :
  let num_models : ℕ := 6
  let bathing_suits_per_model : ℕ := 2
  let evening_wear_per_model : ℕ := 3
  let time_per_trip : ℕ := 2

  let total_trips_per_model : ℕ := bathing_suits_per_model + evening_wear_per_model
  let total_trips : ℕ := num_models * total_trips_per_model
  let total_time : ℕ := total_trips * time_per_trip

  total_time = 60 := by sorry

end fashion_show_runway_time_l3158_315816


namespace production_period_is_seven_days_l3158_315899

def computers_per_day : ℕ := 1500
def price_per_computer : ℕ := 150
def total_revenue : ℕ := 1575000

theorem production_period_is_seven_days :
  (total_revenue / price_per_computer) / computers_per_day = 7 := by
  sorry

end production_period_is_seven_days_l3158_315899


namespace parabola_sum_l3158_315806

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := (3, -2)

/-- The parabola contains the point (0, 5) -/
def contains_point (p : Parabola) : Prop :=
  p.a * 0^2 + p.b * 0 + p.c = 5

/-- The axis of symmetry is vertical -/
def vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∃ x : ℚ, ∀ y : ℚ, p.a * (x - 3)^2 = y + 2

theorem parabola_sum (p : Parabola) 
  (h1 : vertex p = (3, -2))
  (h2 : contains_point p)
  (h3 : vertical_axis_of_symmetry p) :
  p.a + p.b + p.c = 10/9 := by
  sorry

end parabola_sum_l3158_315806


namespace difference_largest_negative_smallest_positive_not_two_l3158_315800

theorem difference_largest_negative_smallest_positive_not_two : ¬(∃ n m : ℤ, 
  (∀ k : ℤ, k < 0 → k ≤ n) ∧ 
  (∀ k : ℤ, k > 0 → m ≤ k) ∧ 
  n - m = 2) :=
sorry

end difference_largest_negative_smallest_positive_not_two_l3158_315800


namespace point_covering_theorem_l3158_315824

/-- A point in the unit square -/
structure Point where
  x : Real
  y : Real
  x_in_unit : 0 ≤ x ∧ x ≤ 1
  y_in_unit : 0 ≤ y ∧ y ≤ 1

/-- A rectangle inside the unit square with sides parallel to the square's sides -/
structure Rectangle where
  x1 : Real
  y1 : Real
  x2 : Real
  y2 : Real
  x1_le_x2 : x1 ≤ x2
  y1_le_y2 : y1 ≤ y2
  in_unit_square : 0 ≤ x1 ∧ x2 ≤ 1 ∧ 0 ≤ y1 ∧ y2 ≤ 1

/-- Check if a point is inside a rectangle -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.x1 ≤ p.x ∧ p.x ≤ r.x2 ∧ r.y1 ≤ p.y ∧ p.y ≤ r.y2

/-- The area of a rectangle -/
def rectangleArea (r : Rectangle) : Real :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

/-- The main theorem -/
theorem point_covering_theorem :
  ∃ (points : Finset Point),
    points.card = 1965 ∧
    ∀ (r : Rectangle),
      rectangleArea r = 1 / 200 →
      ∃ (p : Point), p ∈ points ∧ pointInRectangle p r :=
sorry

end point_covering_theorem_l3158_315824


namespace original_number_of_professors_l3158_315852

theorem original_number_of_professors : 
  ∃ p : ℕ+, 
    p.val > 0 ∧ 
    6480 % p.val = 0 ∧ 
    11200 % (p.val + 3) = 0 ∧ 
    (6480 : ℚ) / p.val < (11200 : ℚ) / (p.val + 3) ∧ 
    p = 5 := by
  sorry

end original_number_of_professors_l3158_315852


namespace mango_juice_savings_l3158_315832

/-- Represents the volume and cost of a bottle of mango juice -/
structure Bottle where
  volume : ℕ  -- volume in ounces
  cost : ℕ    -- cost in pesetas

/-- Calculates the savings when buying a big bottle instead of equivalent small bottles -/
def calculateSavings (bigBottle smallBottle : Bottle) : ℕ :=
  let smallBottlesNeeded := bigBottle.volume / smallBottle.volume
  smallBottlesNeeded * smallBottle.cost - bigBottle.cost

/-- Theorem stating the savings when buying a big bottle instead of equivalent small bottles -/
theorem mango_juice_savings :
  let bigBottle : Bottle := { volume := 30, cost := 2700 }
  let smallBottle : Bottle := { volume := 6, cost := 600 }
  calculateSavings bigBottle smallBottle = 300 := by
  sorry


end mango_juice_savings_l3158_315832


namespace expression_simplification_l3158_315876

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x - 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = 1 / 5 := by sorry

end expression_simplification_l3158_315876


namespace complex_system_solution_l3158_315820

theorem complex_system_solution (z₁ z₂ : ℂ) 
  (eq1 : z₁ - 2 * z₂ = 5 + Complex.I) 
  (eq2 : 2 * z₁ + z₂ = 3 * Complex.I) : 
  z₁ = 1 + (7 / 5) * Complex.I := by
  sorry

end complex_system_solution_l3158_315820


namespace room_area_calculation_l3158_315851

/-- Given a rectangular carpet covering 30% of a room's floor area,
    if the carpet measures 4 feet by 9 feet,
    then the total floor area is 120 square feet. -/
theorem room_area_calculation (carpet_length carpet_width carpet_coverage total_area : ℝ) : 
  carpet_length = 4 →
  carpet_width = 9 →
  carpet_coverage = 0.30 →
  carpet_coverage * total_area = carpet_length * carpet_width →
  total_area = 120 := by
sorry

end room_area_calculation_l3158_315851


namespace inequality_solution_set_l3158_315834

theorem inequality_solution_set : 
  {x : ℝ | (2 / x + Real.sqrt (1 - x) ≥ 1 + Real.sqrt (1 - x)) ∧ (x > 0) ∧ (x ≤ 1)} = 
  {x : ℝ | x > 0 ∧ x ≤ 1} := by
sorry

end inequality_solution_set_l3158_315834


namespace arithmetic_mean_greater_than_harmonic_mean_l3158_315850

theorem arithmetic_mean_greater_than_harmonic_mean 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) (hne : a ≠ b) : 
  (a + b) / 2 > 2 * a * b / (a + b) := by
  sorry

end arithmetic_mean_greater_than_harmonic_mean_l3158_315850


namespace new_person_weight_l3158_315808

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3 →
  replaced_weight = 70 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 94 :=
by
  sorry

end new_person_weight_l3158_315808


namespace prop_or_quadratic_always_positive_parallel_iff_l3158_315823

-- Define propositions p and q
def p : Prop := ∀ x : ℚ, (x : ℝ) = x
def q : Prop := ∀ x : ℝ, x > 0 → Real.log x < 0

-- Statement 1
theorem prop_or : p ∨ q := by sorry

-- Statement 2
theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by sorry

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 2 * a = 0

-- Define parallel lines
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y ↔ ∃ k : ℝ, line2 a (x + k) (y + k)

-- Statement 3
theorem parallel_iff : ∀ a : ℝ, parallel a ↔ a = -1 := by sorry

end prop_or_quadratic_always_positive_parallel_iff_l3158_315823


namespace equation_solution_range_l3158_315811

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (2*x - m) / (x - 3) - 1 = x / (3 - x)) → 
  (m > 3 ∧ m ≠ 9) :=
by sorry

end equation_solution_range_l3158_315811


namespace max_page_number_l3158_315888

def count_digit_2 (n : ℕ) : ℕ :=
  let ones := n % 10
  let tens := (n / 10) % 10
  let hundreds := (n / 100) % 10
  (if ones = 2 then 1 else 0) +
  (if tens = 2 then 1 else 0) +
  (if hundreds = 2 then 1 else 0)

def total_2s_up_to (n : ℕ) : ℕ :=
  (List.range (n + 1)).map count_digit_2 |> List.sum

theorem max_page_number (available_2s : ℕ) (h : available_2s = 100) : 
  ∃ (max_page : ℕ), max_page = 244 ∧ 
    total_2s_up_to max_page ≤ available_2s ∧
    ∀ (n : ℕ), n > max_page → total_2s_up_to n > available_2s :=
by sorry

end max_page_number_l3158_315888


namespace trigonometric_expression_equality_l3158_315809

theorem trigonometric_expression_equality : 
  4 * (Real.sin (49 * π / 48) ^ 3 * Real.cos (49 * π / 16) + 
       Real.cos (49 * π / 48) ^ 3 * Real.sin (49 * π / 16)) * 
       Real.cos (49 * π / 12) = 0.75 := by sorry

end trigonometric_expression_equality_l3158_315809


namespace rectangle_longer_side_l3158_315863

/-- Given a circle with radius 6 cm that is tangent to three sides of a rectangle,
    where the rectangle's area is four times the circle's area,
    prove that the length of the longer side of the rectangle is 12π cm. -/
theorem rectangle_longer_side (r : ℝ) (circle_area rectangle_area : ℝ) 
  (shorter_side longer_side : ℝ) :
  r = 6 →
  circle_area = Real.pi * r^2 →
  rectangle_area = 4 * circle_area →
  shorter_side = 2 * r →
  rectangle_area = shorter_side * longer_side →
  longer_side = 12 * Real.pi :=
by sorry

end rectangle_longer_side_l3158_315863


namespace polygon_20_vertices_has_170_diagonals_l3158_315884

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A polygon with 20 vertices has 170 diagonals -/
theorem polygon_20_vertices_has_170_diagonals :
  num_diagonals 20 = 170 := by
  sorry

end polygon_20_vertices_has_170_diagonals_l3158_315884


namespace max_abs_x2_l3158_315895

theorem max_abs_x2 (x₁ x₂ x₃ : ℝ) (h : x₁^2 + x₂^2 + x₃^2 + x₁*x₂ + x₂*x₃ = 2) : 
  ∃ (M : ℝ), M = 2 ∧ |x₂| ≤ M ∧ ∃ (y₁ y₂ y₃ : ℝ), y₁^2 + y₂^2 + y₃^2 + y₁*y₂ + y₂*y₃ = 2 ∧ |y₂| = M :=
sorry

end max_abs_x2_l3158_315895


namespace swimming_improvement_l3158_315889

/-- Represents John's swimming performance -/
structure SwimmingPerformance where
  laps : ℕ
  time : ℕ

/-- Calculates the lap time in minutes per lap -/
def lapTime (performance : SwimmingPerformance) : ℚ :=
  performance.time / performance.laps

theorem swimming_improvement 
  (initial : SwimmingPerformance) 
  (final : SwimmingPerformance) 
  (h1 : initial.laps = 15) 
  (h2 : initial.time = 35) 
  (h3 : final.laps = 18) 
  (h4 : final.time = 33) : 
  lapTime initial - lapTime final = 1/9 := by
  sorry

end swimming_improvement_l3158_315889


namespace least_multiple_with_digit_sum_l3158_315838

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem least_multiple_with_digit_sum (N : ℕ) : N = 779 ↔ 
  (∀ m : ℕ, m < N → (m % 19 = 0 → sum_of_digits m ≠ 23)) ∧ 
  (N % 19 = 0) ∧ 
  (sum_of_digits N = 23) :=
sorry

end least_multiple_with_digit_sum_l3158_315838


namespace initial_bees_in_hive_initial_bees_count_l3158_315872

theorem initial_bees_in_hive : ℕ → Prop :=
  fun initial_bees =>
    initial_bees + 7 = 23

theorem initial_bees_count : ∃ (n : ℕ), initial_bees_in_hive n ∧ n = 16 := by
  sorry

end initial_bees_in_hive_initial_bees_count_l3158_315872


namespace shopping_mall_problem_l3158_315821

/-- Represents the shopping mall's product purchasing scenario -/
structure ProductScenario where
  cost_a : ℚ  -- Cost price of product A
  cost_b : ℚ  -- Cost price of product B
  quantity_a : ℕ  -- Quantity of product A purchased
  quantity_b : ℕ  -- Quantity of product B purchased

/-- Theorem representing the shopping mall problem -/
theorem shopping_mall_problem 
  (scenario : ProductScenario) 
  (h1 : scenario.cost_a = scenario.cost_b - 2)
  (h2 : 80 / scenario.cost_a = 100 / scenario.cost_b)
  (h3 : scenario.quantity_a = 3 * scenario.quantity_b - 5)
  (h4 : scenario.quantity_a + scenario.quantity_b ≤ 95)
  (h5 : (12 - scenario.cost_a) * scenario.quantity_a + 
        (15 - scenario.cost_b) * scenario.quantity_b > 380) :
  (scenario.cost_a = 8 ∧ scenario.cost_b = 10) ∧
  (∀ s : ProductScenario, s.quantity_b ≤ 25) ∧
  (scenario.quantity_a = 67 ∧ scenario.quantity_b = 24) ∨
  (scenario.quantity_a = 70 ∧ scenario.quantity_b = 25) :=
sorry

end shopping_mall_problem_l3158_315821


namespace second_player_wins_l3158_315869

/-- Represents a position on the chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the state of the game -/
structure GameState :=
  (white_rook : Position)
  (black_rook : Position)
  (visited : Set Position)
  (current_player : Bool)  -- true for White, false for Black

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (new_pos : Position) : Bool :=
  -- Implementation details omitted
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Option Position

/-- Checks if a strategy is a winning strategy for the given player -/
def is_winning_strategy (strategy : Strategy) (player : Bool) : Prop :=
  -- Implementation details omitted
  sorry

/-- The main theorem stating that the second player (Black) has a winning strategy -/
theorem second_player_wins :
  ∃ (strategy : Strategy), 
    is_winning_strategy strategy false ∧
    strategy { 
      white_rook := { row := 1, col := 1 },  -- b2
      black_rook := { row := 3, col := 2 },  -- c4
      visited := { { row := 1, col := 1 }, { row := 3, col := 2 } },
      current_player := true
    } ≠ none :=
  sorry

end second_player_wins_l3158_315869


namespace point_outside_circle_l3158_315891

theorem point_outside_circle (a : ℝ) : 
  let P : ℝ × ℝ := (a, 10)
  let C : ℝ × ℝ := (1, 1)
  let r : ℝ := Real.sqrt 2
  let d : ℝ := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  d > r := by sorry

end point_outside_circle_l3158_315891


namespace mary_money_l3158_315854

def quarters : ℕ := 21
def dimes : ℕ := (quarters - 7) / 2

def quarter_value : ℚ := 25 / 100
def dime_value : ℚ := 10 / 100

def total_money : ℚ := quarters * quarter_value + dimes * dime_value

theorem mary_money : total_money = 595 / 100 := by
  sorry

end mary_money_l3158_315854


namespace project_hours_ratio_l3158_315841

/-- Represents the hours worked by each person on the project -/
structure ProjectHours where
  least : ℕ
  hardest : ℕ
  third : ℕ

/-- Checks if the given ProjectHours satisfies the problem conditions -/
def isValidProjectHours (hours : ProjectHours) : Prop :=
  hours.least + hours.hardest + hours.third = 90 ∧
  hours.hardest = hours.least + 20

/-- Theorem stating the ratio of hours worked -/
theorem project_hours_ratio :
  ∃ (hours : ProjectHours),
    isValidProjectHours hours ∧
    hours.least = 25 ∧
    hours.hardest = 45 ∧
    hours.third = 20 := by
  sorry

end project_hours_ratio_l3158_315841


namespace range_of_a_l3158_315881

theorem range_of_a (p q : Prop) (a : ℝ) : 
  (∀ x > -1, p → x^2 / (x + 1) ≥ a) →
  (q ↔ ∃ x : ℝ, a * x^2 - a * x + 1 = 0) →
  (¬p ∧ ¬q) →
  (p ∨ q) →
  (a = 0 ∨ a ≥ 4) :=
by sorry

end range_of_a_l3158_315881


namespace test_completion_ways_l3158_315843

/-- The number of questions in the test -/
def num_questions : ℕ := 8

/-- The number of answer choices for each question -/
def num_choices : ℕ := 7

/-- The total number of options for each question (including unanswered) -/
def total_options : ℕ := num_choices + 1

/-- The theorem stating the total number of ways to complete the test -/
theorem test_completion_ways :
  (total_options : ℕ) ^ num_questions = 16777216 := by
  sorry

end test_completion_ways_l3158_315843


namespace find_divisor_l3158_315842

theorem find_divisor (x : ℝ) (y : ℝ) 
  (h1 : (x - 5) / 7 = 7) 
  (h2 : (x - 4) / y = 5) : 
  y = 10 := by
sorry

end find_divisor_l3158_315842


namespace min_fence_length_l3158_315870

theorem min_fence_length (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 100) :
  2 * (x + y) ≥ 40 ∧ (2 * (x + y) = 40 ↔ x = 10 ∧ y = 10) := by
  sorry

end min_fence_length_l3158_315870


namespace math_test_score_difference_l3158_315866

theorem math_test_score_difference :
  ∀ (grant_score john_score hunter_score : ℕ),
    grant_score = 100 →
    john_score = 2 * hunter_score →
    hunter_score = 45 →
    grant_score - john_score = 10 :=
by
  sorry

end math_test_score_difference_l3158_315866


namespace retailer_profit_percentage_l3158_315813

/-- Calculates the actual profit percentage for a retailer given a markup and discount rate -/
def actualProfitPercentage (markup : ℝ) (discount : ℝ) : ℝ :=
  let markedPrice := 1 + markup
  let sellingPrice := markedPrice * (1 - discount)
  let profit := sellingPrice - 1
  profit * 100

/-- Theorem stating that the actual profit percentage is 5% for a 40% markup and 25% discount -/
theorem retailer_profit_percentage :
  actualProfitPercentage 0.4 0.25 = 5 := by
  sorry

end retailer_profit_percentage_l3158_315813


namespace monotonicity_and_extrema_l3158_315846

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem monotonicity_and_extrema (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 + (2*a - 1) * x - Real.log x) →
  (a = 1/2 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x, x > 0 → f a x ≥ 1/2) ∧
    f a 1 = 1/2) ∧
  (a ≤ 0 →
    ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a > 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end monotonicity_and_extrema_l3158_315846


namespace rational_inequality_solution_l3158_315861

theorem rational_inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 :=
by sorry

end rational_inequality_solution_l3158_315861


namespace ship_round_trip_tickets_l3158_315853

/-- Given a ship with passengers, prove that 80% of passengers have round-trip tickets -/
theorem ship_round_trip_tickets (total_passengers : ℝ) 
  (h1 : total_passengers > 0) 
  (h2 : (0.4 : ℝ) * total_passengers = round_trip_with_car)
  (h3 : (0.5 : ℝ) * round_trip_tickets = round_trip_without_car)
  (h4 : round_trip_tickets = round_trip_with_car + round_trip_without_car) :
  round_trip_tickets = (0.8 : ℝ) * total_passengers :=
by
  sorry

end ship_round_trip_tickets_l3158_315853


namespace waiter_new_customers_l3158_315845

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 19)
  (h2 : customers_left = 14)
  (h3 : final_customers = 41) :
  final_customers - (initial_customers - customers_left) = 36 := by
  sorry

end waiter_new_customers_l3158_315845


namespace square_property_l3158_315864

theorem square_property (n : ℕ+) : ∃ k : ℤ, (n + 1 : ℤ) * (n + 2) * (n^2 + 3*n) + 1 = k^2 := by
  sorry

end square_property_l3158_315864


namespace f_inequality_l3158_315830

/-- The number of ways to express a positive integer as a sum of ascending positive integers. -/
def f (n : ℕ+) : ℕ := sorry

/-- The theorem stating that f(n+1) ≤ (1/2)[f(n) + f(n+2)] for any positive integer n. -/
theorem f_inequality (n : ℕ+) : f (n + 1) ≤ (f n + f (n + 2)) / 2 := by sorry

end f_inequality_l3158_315830


namespace total_sequences_count_l3158_315898

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 5

/-- The total number of possible sequences of student selections for one week -/
def total_sequences : ℕ := num_students ^ meetings_per_week

/-- Theorem stating that the total number of sequences is 759,375 -/
theorem total_sequences_count : total_sequences = 759375 := by
  sorry

end total_sequences_count_l3158_315898


namespace sally_quarters_l3158_315897

def initial_quarters : ℕ := 760
def spent_quarters : ℕ := 418

theorem sally_quarters :
  initial_quarters - spent_quarters = 342 := by sorry

end sally_quarters_l3158_315897


namespace savings_calculation_l3158_315868

/-- Given a person's income and the ratio of income to expenditure, calculate their savings. -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given an income of 15000 and an income to expenditure ratio of 5:4, the savings are 3000. -/
theorem savings_calculation :
  calculate_savings 15000 5 4 = 3000 := by
  sorry

end savings_calculation_l3158_315868


namespace min_k_value_l3158_315835

/-- Given a line and a circle in a Cartesian coordinate system,
    prove that the minimum value of k satisfying the conditions is -√3 -/
theorem min_k_value (k : ℝ) : 
  (∃ P : ℝ × ℝ, P.2 = k * (P.1 - 3 * Real.sqrt 3)) →
  (∃ Q : ℝ × ℝ, Q.1^2 + (Q.2 - 1)^2 = 1) →
  (∃ P Q : ℝ × ℝ, P = (3 * Q.1, 3 * Q.2)) →
  -Real.sqrt 3 ≤ k :=
by sorry

end min_k_value_l3158_315835


namespace marks_birth_year_l3158_315849

theorem marks_birth_year (current_year : ℕ) (janice_age : ℕ) 
  (h1 : current_year = 2021)
  (h2 : janice_age = 21)
  (h3 : ∃ (graham_age : ℕ), graham_age = 2 * janice_age)
  (h4 : ∃ (mark_age : ℕ), mark_age = graham_age + 3) :
  ∃ (birth_year : ℕ), birth_year = current_year - (2 * janice_age + 3) := by
  sorry

end marks_birth_year_l3158_315849


namespace stock_b_highest_income_l3158_315822

/-- Represents a stock with its dividend rate and price per share -/
structure Stock where
  dividend_rate : Rat
  price_per_share : Nat

/-- Calculates the annual income from a stock given the total investment -/
def annual_income (stock : Stock) (total_investment : Nat) : Rat :=
  (total_investment : Rat) * stock.dividend_rate

/-- Theorem: Stock B yields the highest annual income among the three stocks -/
theorem stock_b_highest_income (total_investment : Nat) 
  (stock_a stock_b stock_c : Stock)
  (h_total : total_investment = 6800)
  (h_a : stock_a = { dividend_rate := 1/10, price_per_share := 136 })
  (h_b : stock_b = { dividend_rate := 12/100, price_per_share := 150 })
  (h_c : stock_c = { dividend_rate := 8/100, price_per_share := 100 }) :
  annual_income stock_b (150 * (total_investment / 150)) ≥ 
    max (annual_income stock_a (136 * (total_investment / 136)))
        (annual_income stock_c (100 * (total_investment / 100))) :=
by sorry


end stock_b_highest_income_l3158_315822


namespace remainder_problem_l3158_315825

theorem remainder_problem : ∃ k : ℤ, 
  2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12 = 1001 * k + 400 := by
  sorry

end remainder_problem_l3158_315825


namespace log_product_sqrt_equals_sqrt_two_l3158_315840

theorem log_product_sqrt_equals_sqrt_two : 
  Real.sqrt (Real.log 8 / Real.log 4 * Real.log 16 / Real.log 8) = Real.sqrt 2 := by
  sorry

end log_product_sqrt_equals_sqrt_two_l3158_315840


namespace clare_milk_cartons_l3158_315828

def prove_milk_cartons (initial_money : ℕ) (num_bread : ℕ) (cost_bread : ℕ) (cost_milk : ℕ) (money_left : ℕ) : Prop :=
  let money_spent : ℕ := initial_money - money_left
  let bread_cost : ℕ := num_bread * cost_bread
  let milk_cost : ℕ := money_spent - bread_cost
  let num_milk_cartons : ℕ := milk_cost / cost_milk
  num_milk_cartons = 2

theorem clare_milk_cartons :
  prove_milk_cartons 47 4 2 2 35 := by
  sorry

end clare_milk_cartons_l3158_315828


namespace mikes_weekly_pullups_l3158_315890

/-- Calculates the number of pull-ups Mike does in a week -/
theorem mikes_weekly_pullups 
  (pullups_per_visit : ℕ) 
  (office_visits_per_day : ℕ) 
  (days_in_week : ℕ) 
  (h1 : pullups_per_visit = 2) 
  (h2 : office_visits_per_day = 5) 
  (h3 : days_in_week = 7) : 
  pullups_per_visit * office_visits_per_day * days_in_week = 70 := by
  sorry

#check mikes_weekly_pullups

end mikes_weekly_pullups_l3158_315890


namespace function_lower_bound_l3158_315857

/-- Given a function f(x) = x^2 - (a+1)x + a, where a is a real number,
    if f(x) ≥ -1 for all x > 1, then a ≤ 3 -/
theorem function_lower_bound (a : ℝ) :
  (∀ x > 1, x^2 - (a + 1)*x + a ≥ -1) → a ≤ 3 := by
  sorry

end function_lower_bound_l3158_315857


namespace odd_function_property_l3158_315802

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g as a function from ℝ to ℝ
def g (x : ℝ) : ℝ := f x + 9

-- Theorem statement
theorem odd_function_property (hf_odd : ∀ x, f (-x) = -f x) (hg_value : g (-2) = 3) : f 2 = 6 := by
  sorry

end odd_function_property_l3158_315802


namespace extreme_value_and_slope_l3158_315807

/-- A function f with an extreme value at x = 1 -/
def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (x : ℝ) (a b : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_value_and_slope (a b : ℝ) :
  f 1 a b = 10 ∧ f' 1 a b = 0 → f' 2 a b = 17 :=
by sorry

end extreme_value_and_slope_l3158_315807


namespace geometric_sequence_common_ratio_l3158_315886

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a4a5 : a 4 * a 5 = 1) 
  (h_a8a9 : a 8 * a 9 = 16) : 
  q = Real.sqrt 2 := by
sorry

end geometric_sequence_common_ratio_l3158_315886
