import Mathlib

namespace polynomial_properties_l790_79091

def P (x : ℤ) : ℤ := x * (x + 1) * (x + 2)

theorem polynomial_properties :
  (∀ x : ℤ, ∃ k : ℤ, P x = 3 * k) ∧
  (∃ a b c d : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) ∧
  (∃ a b c : ℤ, ∀ x : ℤ, P x = x^3 + a*x^2 + b*x + c) :=
sorry

end polynomial_properties_l790_79091


namespace circle_through_points_l790_79069

theorem circle_through_points : ∃ (A B C D : ℝ), 
  (A * 0^2 + B * 0^2 + C * 0 + D * 0 + 1 = 0) ∧ 
  (A * 4^2 + B * 0^2 + C * 4 + D * 0 + 1 = 0) ∧ 
  (A * (-1)^2 + B * 1^2 + C * (-1) + D * 1 + 1 = 0) ∧ 
  (A = 1 ∧ B = 1 ∧ C = -4 ∧ D = -6) := by
  sorry

end circle_through_points_l790_79069


namespace town_population_problem_l790_79054

/-- The original population of the town -/
def original_population : ℕ := 1200

/-- The increase in population -/
def population_increase : ℕ := 1500

/-- The percentage decrease after the increase -/
def percentage_decrease : ℚ := 15 / 100

/-- The final difference in population compared to the original plus increase -/
def final_difference : ℕ := 45

theorem town_population_problem :
  let increased_population := original_population + population_increase
  let decreased_population := increased_population - (increased_population * percentage_decrease).floor
  decreased_population = original_population + population_increase - final_difference :=
by sorry

end town_population_problem_l790_79054


namespace inverse_computation_l790_79032

-- Define the function g
def g : ℕ → ℕ
| 1 => 4
| 2 => 9
| 3 => 11
| 5 => 3
| 7 => 6
| 12 => 2
| _ => 0  -- for other inputs, we'll return 0

-- Assume g has an inverse
axiom g_has_inverse : Function.Bijective g

-- Define g_inv as the inverse of g
noncomputable def g_inv : ℕ → ℕ := Function.invFun g

-- State the theorem
theorem inverse_computation :
  g_inv ((g_inv 2 + g_inv 11) / g_inv 3) = 5 := by sorry

end inverse_computation_l790_79032


namespace polynomial_equality_l790_79067

theorem polynomial_equality (q : ℝ → ℝ) :
  (∀ x, q x + (2 * x^6 + 4 * x^4 + 5 * x^3 + 10 * x) = 
    (12 * x^5 + 6 * x^4 + 28 * x^3 + 30 * x^2 + 3 * x + 2)) →
  (∀ x, q x = -2 * x^6 + 12 * x^5 + 2 * x^4 + 23 * x^3 + 30 * x^2 - 7 * x + 2) :=
by
  sorry

end polynomial_equality_l790_79067


namespace sum_of_2_and_odd_prime_last_digit_l790_79026

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def last_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_2_and_odd_prime_last_digit (p : ℕ) 
  (h_prime : is_prime p) 
  (h_odd : p % 2 = 1) 
  (h_greater_7 : p > 7) 
  (h_sum_not_single_digit : p + 2 ≥ 10) : 
  last_digit (p + 2) = 1 ∨ last_digit (p + 2) = 3 ∨ last_digit (p + 2) = 9 :=
sorry

end sum_of_2_and_odd_prime_last_digit_l790_79026


namespace grid_adjacent_difference_l790_79022

/-- Represents a cell in the grid -/
structure Cell where
  row : Fin 18
  col : Fin 18

/-- The type of the grid -/
def Grid := Fin 18 → Fin 18 → ℕ+

/-- Two cells are adjacent if they share an edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- The main theorem -/
theorem grid_adjacent_difference (g : Grid) 
  (h : ∀ (c1 c2 : Cell), c1 ≠ c2 → g c1.row c1.col ≠ g c2.row c2.col) :
  ∃ (c1 c2 c3 c4 : Cell), 
    adjacent c1 c2 ∧ adjacent c3 c4 ∧ 
    (c1, c2) ≠ (c3, c4) ∧
    (g c1.row c1.col).val + 10 ≤ (g c2.row c2.col).val ∧
    (g c3.row c3.col).val + 10 ≤ (g c4.row c4.col).val :=
sorry

end grid_adjacent_difference_l790_79022


namespace machine_a_production_rate_l790_79053

/-- Represents the production rate and time for a machine. -/
structure Machine where
  rate : ℝ  -- Sprockets produced per hour
  time : ℝ  -- Hours to produce 2000 sprockets

/-- Given three machines A, B, and G with specific production relationships,
    prove that machine A produces 200/11 sprockets per hour. -/
theorem machine_a_production_rate 
  (a b g : Machine)
  (total_sprockets : ℝ)
  (h1 : total_sprockets = 2000)
  (h2 : a.time = g.time + 10)
  (h3 : b.time = g.time - 5)
  (h4 : g.rate = 1.1 * a.rate)
  (h5 : b.rate = 1.15 * a.rate)
  (h6 : a.rate * a.time = total_sprockets)
  (h7 : b.rate * b.time = total_sprockets)
  (h8 : g.rate * g.time = total_sprockets) :
  a.rate = 200 / 11 := by
  sorry

#eval (200 : ℚ) / 11

end machine_a_production_rate_l790_79053


namespace possible_c_value_l790_79018

/-- An obtuse-angled triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  obtuse : c^2 > a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating that 2√5 is a possible value for c in the given obtuse triangle -/
theorem possible_c_value (t : ObtuseTriangle) 
  (ha : t.a = Real.sqrt 2)
  (hb : t.b = 2 * Real.sqrt 2)
  (hc : t.c > t.b) :
  ∃ (t' : ObtuseTriangle), t'.a = t.a ∧ t'.b = t.b ∧ t'.c = 2 * Real.sqrt 5 := by
  sorry

end possible_c_value_l790_79018


namespace subsidy_and_job_creation_l790_79036

/-- Data for SZ province's "home appliances to the countryside" program in 2008 -/
structure ProgramData2008 where
  new_shops : ℕ
  jobs_created : ℕ
  units_sold : ℕ
  sales_amount : ℝ
  consumption_increase : ℝ
  subsidy_rate : ℝ

/-- Data for the program from 2008 to 2010 -/
structure ProgramData2008To2010 where
  total_jobs : ℕ
  increase_2010_vs_2009 : ℝ
  jobs_increase_2010_vs_2009 : ℝ

/-- Theorem about the subsidy funds needed in 2008 and job creation rate -/
theorem subsidy_and_job_creation 
  (data_2008 : ProgramData2008)
  (data_2008_to_2010 : ProgramData2008To2010)
  (h1 : data_2008.new_shops = 8000)
  (h2 : data_2008.jobs_created = 75000)
  (h3 : data_2008.units_sold = 1130000)
  (h4 : data_2008.sales_amount = 1.6 * 10^9)
  (h5 : data_2008.consumption_increase = 1.7)
  (h6 : data_2008.subsidy_rate = 0.13)
  (h7 : data_2008_to_2010.total_jobs = 247000)
  (h8 : data_2008_to_2010.increase_2010_vs_2009 = 0.5)
  (h9 : data_2008_to_2010.jobs_increase_2010_vs_2009 = 10/81) :
  ∃ (subsidy_funds : ℝ) (jobs_per_point : ℝ),
    subsidy_funds = 2.08 * 10^9 ∧ 
    jobs_per_point = 20000 := by
  sorry

end subsidy_and_job_creation_l790_79036


namespace max_sum_of_factors_l790_79025

theorem max_sum_of_factors (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 2277 →
  a + b + c + d ≤ 84 :=
by sorry

end max_sum_of_factors_l790_79025


namespace samantha_birth_year_l790_79024

def first_amc_year : ℕ := 1985

def samantha_age_at_seventh_amc : ℕ := 12

theorem samantha_birth_year :
  ∃ (birth_year : ℕ),
    birth_year = first_amc_year + 6 - samantha_age_at_seventh_amc ∧
    birth_year = 1979 :=
by sorry

end samantha_birth_year_l790_79024


namespace linda_original_money_l790_79098

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_money :
  linda_original = 10 :=
by
  have h1 : lucy_original - transfer_amount = linda_original + transfer_amount :=
    sorry
  sorry

end linda_original_money_l790_79098


namespace expression_always_defined_l790_79001

theorem expression_always_defined (x : ℝ) (h : x > 12) : x^2 - 24*x + 144 ≠ 0 := by
  sorry

end expression_always_defined_l790_79001


namespace unique_prime_with_14_divisors_l790_79087

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The theorem stating that there is exactly one prime p such that p^2 + 23 has 14 positive divisors -/
theorem unique_prime_with_14_divisors :
  ∃! p : ℕ, Nat.Prime p ∧ num_divisors (p^2 + 23) = 14 :=
sorry

end unique_prime_with_14_divisors_l790_79087


namespace coordinates_of_point_B_l790_79002

/-- Given two points A and B in 2D space, this theorem proves that if the coordinates of A are (-2, -1) and the vector from A to B is (3, 4), then the coordinates of B are (1, 3). -/
theorem coordinates_of_point_B (A B : ℝ × ℝ) : 
  A = (-2, -1) → (B.1 - A.1, B.2 - A.2) = (3, 4) → B = (1, 3) := by
  sorry

end coordinates_of_point_B_l790_79002


namespace first_share_interest_rate_l790_79081

/-- Proves that the interest rate of the first type of share is 9% given the problem conditions --/
theorem first_share_interest_rate : 
  let total_investment : ℝ := 100000
  let second_share_rate : ℝ := 11
  let total_interest_rate : ℝ := 9.5
  let second_share_investment : ℝ := 25000
  let first_share_investment : ℝ := total_investment - second_share_investment
  let total_interest : ℝ := total_interest_rate / 100 * total_investment
  let second_share_interest : ℝ := second_share_rate / 100 * second_share_investment
  let first_share_interest : ℝ := total_interest - second_share_interest
  let first_share_rate : ℝ := first_share_interest / first_share_investment * 100
  first_share_rate = 9 := by
  sorry


end first_share_interest_rate_l790_79081


namespace y_derivative_l790_79043

noncomputable def y (x : ℝ) : ℝ :=
  (2 * x^2 - x + 1/2) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) - 
  x^3 / (2 * Real.sqrt 3) - (Real.sqrt 3 / 2) * x

theorem y_derivative (x : ℝ) (hx : x ≠ 0) : 
  deriv y x = (4 * x - 1) * Real.arctan ((x^2 - 1) / (x * Real.sqrt 3)) + 
  (Real.sqrt 3 * (x^2 + 1) * (3 * x^2 - 2 * x - x^4)) / (2 * (x^4 + x^2 + 1)) :=
by sorry

end y_derivative_l790_79043


namespace jorges_gifts_count_l790_79047

/-- The number of gifts Jorge gave at Rosalina's wedding --/
def jorges_gifts (total_gifts emilios_gifts pedros_gifts : ℕ) : ℕ :=
  total_gifts - (emilios_gifts + pedros_gifts)

theorem jorges_gifts_count :
  jorges_gifts 21 11 4 = 6 :=
by sorry

end jorges_gifts_count_l790_79047


namespace unique_five_digit_pair_l790_79071

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Check if each digit of b is exactly 1 greater than the corresponding digit of a -/
def digitsOneGreater (a b : ℕ) : Prop :=
  ∀ i : ℕ, i < 5 → (b / 10^i) % 10 = (a / 10^i) % 10 + 1

/-- The main theorem -/
theorem unique_five_digit_pair : 
  ∀ a b : ℕ, 
    10000 ≤ a ∧ a < 100000 ∧
    10000 ≤ b ∧ b < 100000 ∧
    isPerfectSquare a ∧
    isPerfectSquare b ∧
    b - a = 11111 ∧
    digitsOneGreater a b →
    a = 13225 ∧ b = 24336 :=
sorry

end unique_five_digit_pair_l790_79071


namespace quadratic_function_coefficients_l790_79008

/-- A quadratic function f(x) = x^2 + ax + b satisfying f(f(x) + x) / f(x) = x^2 + 2023x + 1776 
    has coefficients a = 2021 and b = -246. -/
theorem quadratic_function_coefficients (a b : ℝ) : 
  (∀ x, (((x^2 + a*x + b)^2 + a*(x^2 + a*x + b) + b) / (x^2 + a*x + b) = x^2 + 2023*x + 1776)) → 
  (a = 2021 ∧ b = -246) :=
by sorry

end quadratic_function_coefficients_l790_79008


namespace problem_solution_l790_79017

def row1 (n : ℕ) : ℤ := (-2) ^ n

def row2 (n : ℕ) : ℤ := row1 n + 2

def row3 (n : ℕ) : ℤ := (-2) ^ (n - 1)

theorem problem_solution :
  (row1 4 = 16) ∧
  (∀ n : ℕ, row2 n = row1 n + 2) ∧
  (∃ k : ℕ, row3 k + row3 (k + 1) + row3 (k + 2) = -192 ∧
            row3 k = -64 ∧ row3 (k + 1) = 128 ∧ row3 (k + 2) = -256) :=
by sorry

end problem_solution_l790_79017


namespace quadrilateral_reconstruction_l790_79057

/-- Given a quadrilateral EFGH with extended sides, prove the reconstruction formula for point E -/
theorem quadrilateral_reconstruction 
  (E F G H E' F' G' H' : ℝ × ℝ) 
  (h1 : E' - F = 2 * (E - F))
  (h2 : F' - G = 2 * (F - G))
  (h3 : G' - H = 2 * (G - H))
  (h4 : H' - E = 2 * (H - E)) :
  E = (1/79 : ℝ) • E' + (26/79 : ℝ) • F' + (26/79 : ℝ) • G' + (52/79 : ℝ) • H' := by
  sorry

end quadrilateral_reconstruction_l790_79057


namespace lawn_mowing_solution_l790_79035

/-- Represents the lawn mowing problem --/
def LawnMowingProblem (lawn_length lawn_width swath_width overlap flowerbed_diameter walking_rate : ℝ) : Prop :=
  let effective_width := (swath_width - overlap) / 12  -- Convert to feet
  let flowerbed_area := Real.pi * (flowerbed_diameter / 2) ^ 2
  let mowing_area := lawn_length * lawn_width - flowerbed_area
  let num_strips := lawn_width / effective_width
  let total_distance := num_strips * lawn_length
  let mowing_time := total_distance / walking_rate
  mowing_time = 2

/-- The main theorem stating the solution to the lawn mowing problem --/
theorem lawn_mowing_solution :
  LawnMowingProblem 100 160 30 6 20 4000 := by
  sorry

#check lawn_mowing_solution

end lawn_mowing_solution_l790_79035


namespace square_root_equation_solution_l790_79004

theorem square_root_equation_solution (A C : ℝ) (hA : A ≥ 0) (hC : C ≥ 0) :
  ∃ x : ℝ, x > 0 ∧
    Real.sqrt (2 + A * C + 2 * C * x) + Real.sqrt (A * C - 2 + 2 * A * x) =
    Real.sqrt (2 * (A + C) * x + 2 * A * C) ∧
    x = 4 := by
  sorry

end square_root_equation_solution_l790_79004


namespace point_on_curve_l790_79085

noncomputable def tangent_slope (x : ℝ) : ℝ := 1 + Real.log x

theorem point_on_curve (x y : ℝ) (h : y = x * Real.log x) :
  tangent_slope x = 2 → x = Real.exp 1 ∧ y = Real.exp 1 := by
  sorry

end point_on_curve_l790_79085


namespace households_using_neither_brand_l790_79073

/-- Given information about household soap usage, prove the number of households using neither brand. -/
theorem households_using_neither_brand (total : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  only_A = 60 →
  both = 40 →
  (total - (only_A + 3 * both + both)) = 80 := by
  sorry

end households_using_neither_brand_l790_79073


namespace gcd_40304_30213_l790_79042

theorem gcd_40304_30213 : Nat.gcd 40304 30213 = 1 := by
  sorry

end gcd_40304_30213_l790_79042


namespace magic_square_sum_l790_79089

/-- Represents a 3x3 magic square with center 7 -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  x : ℤ
  y : ℤ := 7

/-- The magic sum of the square -/
def magicSum (s : MagicSquare) : ℤ := 22 + s.c

/-- Properties of the magic square -/
def isMagicSquare (s : MagicSquare) : Prop :=
  s.a + s.y + s.d = magicSum s ∧
  s.c + s.y + s.b = magicSum s ∧
  s.x + s.y + s.a = magicSum s ∧
  s.c + s.y + s.x = magicSum s

theorem magic_square_sum (s : MagicSquare) (h : isMagicSquare s) :
  s.x + s.y + s.a + s.b + s.c + s.d = 68 := by
  sorry

#check magic_square_sum

end magic_square_sum_l790_79089


namespace ratio_invariance_l790_79077

theorem ratio_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x * y) / (y * x) = 1 → ∃ (r : ℝ), r ≠ 0 ∧ x / y = r :=
by sorry

end ratio_invariance_l790_79077


namespace proposition_relationship_l790_79030

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 3) ∧ x + y = 5) := by
  sorry

end proposition_relationship_l790_79030


namespace odd_function_extension_l790_79019

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x > 0, f x = Real.exp x) :
  ∀ x < 0, f x = -Real.exp (-x) := by
sorry

end odd_function_extension_l790_79019


namespace mistaken_divisor_l790_79038

/-- Given a division with remainder 0, correct divisor 21, correct quotient 24,
    and a mistaken quotient of 42, prove that the mistaken divisor is 12. -/
theorem mistaken_divisor (dividend : ℕ) (mistaken_divisor : ℕ) : 
  dividend % 21 = 0 ∧ 
  dividend / 21 = 24 ∧ 
  dividend / mistaken_divisor = 42 →
  mistaken_divisor = 12 := by
sorry

end mistaken_divisor_l790_79038


namespace equation_solutions_l790_79041

/-- Given two equations about x and k -/
theorem equation_solutions (x k : ℚ) : 
  (3 * (2 * x - 1) = k + 2 * x) →
  ((x - k) / 2 = x + 2 * k) →
  (
    /- Part 1 -/
    (x = 4 → (x - k) / 2 = x + 2 * k → x = -65) ∧
    /- Part 2 -/
    (∃ x, 3 * (2 * x - 1) = k + 2 * x ∧ (x - k) / 2 = x + 2 * k) → k = -1/7
  ) := by sorry

end equation_solutions_l790_79041


namespace inequality_properties_l790_79009

theorem inequality_properties (a b c : ℝ) (h : a < b) :
  (a + c < b + c) ∧
  (a - 2 < b - 2) ∧
  (2 * a < 2 * b) ∧
  (-3 * a > -3 * b) := by
  sorry

end inequality_properties_l790_79009


namespace consumption_decrease_l790_79039

theorem consumption_decrease (original_price original_quantity : ℝ) 
  (h1 : original_price > 0) (h2 : original_quantity > 0) : 
  let new_price := 1.4 * original_price
  let new_budget := 1.12 * (original_price * original_quantity)
  let new_quantity := new_budget / new_price
  new_quantity / original_quantity = 0.8 := by sorry

end consumption_decrease_l790_79039


namespace cyclic_quadrilateral_inequality_l790_79072

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)
  (distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)

/-- The theorem states that in a cyclic quadrilateral ABCD where AB is the longest side,
    the sum of AB and BD is greater than the sum of AC and CD. -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (Q : CyclicQuadrilateral P) :
  (∀ X Y : P, dist Q.A Q.B ≥ dist X Y) →
  dist Q.A Q.B + dist Q.B Q.D > dist Q.A Q.C + dist Q.C Q.D :=
sorry

end cyclic_quadrilateral_inequality_l790_79072


namespace imaginary_part_of_z_l790_79061

theorem imaginary_part_of_z (z : ℂ) : (z - 2*I) * (2 - I) = 5 → z.im = 3 := by sorry

end imaginary_part_of_z_l790_79061


namespace alcohol_mixture_theorem_l790_79034

/-- Proves that mixing 300 mL of 10% alcohol solution with 100 mL of 30% alcohol solution results in a 15% alcohol solution -/
theorem alcohol_mixture_theorem :
  let x_volume : ℝ := 300
  let x_concentration : ℝ := 0.10
  let y_volume : ℝ := 100
  let y_concentration : ℝ := 0.30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = 0.15 := by sorry

end alcohol_mixture_theorem_l790_79034


namespace construction_labor_problem_l790_79066

theorem construction_labor_problem (total_hired : ℕ) (operator_pay laborer_pay : ℚ) (total_payroll : ℚ) :
  total_hired = 35 →
  operator_pay = 140 →
  laborer_pay = 90 →
  total_payroll = 3950 →
  ∃ (operators laborers : ℕ),
    operators + laborers = total_hired ∧
    operators * operator_pay + laborers * laborer_pay = total_payroll ∧
    laborers = 19 := by
  sorry

end construction_labor_problem_l790_79066


namespace common_chords_concur_l790_79020

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three pairwise intersecting circles --/
structure ThreeIntersectingCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  intersect_12 : c1.center.1 ^ 2 + c1.center.2 ^ 2 ≠ c2.center.1 ^ 2 + c2.center.2 ^ 2 ∨ c1.center ≠ c2.center
  intersect_23 : c2.center.1 ^ 2 + c2.center.2 ^ 2 ≠ c3.center.1 ^ 2 + c3.center.2 ^ 2 ∨ c2.center ≠ c3.center
  intersect_31 : c3.center.1 ^ 2 + c3.center.2 ^ 2 ≠ c1.center.1 ^ 2 + c1.center.2 ^ 2 ∨ c3.center ≠ c1.center

/-- A line in a plane, represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The common chord of two intersecting circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Three lines concur if they all pass through a single point --/
def concur (l1 l2 l3 : Line) : Prop := sorry

/-- The theorem: The common chords of three pairwise intersecting circles concur --/
theorem common_chords_concur (circles : ThreeIntersectingCircles) :
  let chord12 := commonChord circles.c1 circles.c2
  let chord23 := commonChord circles.c2 circles.c3
  let chord31 := commonChord circles.c3 circles.c1
  concur chord12 chord23 chord31 := by sorry

end common_chords_concur_l790_79020


namespace least_addition_for_divisibility_l790_79056

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), (1052 + y) % 23 = 0 → y ≥ x) ∧ 
  (1052 + x) % 23 = 0 := by
sorry

end least_addition_for_divisibility_l790_79056


namespace younger_brother_age_after_30_years_l790_79006

/-- Given two brothers with an age difference of 10 years, where the elder is 40 years old now,
    prove that the younger brother will be 60 years old after 30 years. -/
theorem younger_brother_age_after_30_years
  (age_difference : ℕ)
  (elder_brother_current_age : ℕ)
  (years_from_now : ℕ)
  (h1 : age_difference = 10)
  (h2 : elder_brother_current_age = 40)
  (h3 : years_from_now = 30) :
  elder_brother_current_age - age_difference + years_from_now = 60 :=
by sorry

end younger_brother_age_after_30_years_l790_79006


namespace probability_two_colored_is_four_ninths_l790_79007

/-- Represents a cube divided into smaller cubes -/
structure DividedCube where
  total_small_cubes : ℕ
  two_colored_faces : ℕ

/-- The probability of selecting a cube with exactly 2 colored faces -/
def probability_two_colored (cube : DividedCube) : ℚ :=
  cube.two_colored_faces / cube.total_small_cubes

/-- Theorem stating the probability of selecting a cube with exactly 2 colored faces -/
theorem probability_two_colored_is_four_ninths (cube : DividedCube) 
    (h1 : cube.total_small_cubes = 27)
    (h2 : cube.two_colored_faces = 12) : 
  probability_two_colored cube = 4/9 := by
  sorry

#check probability_two_colored_is_four_ninths

end probability_two_colored_is_four_ninths_l790_79007


namespace rowing_problem_solution_l790_79060

/-- Represents the problem of calculating the distance to a destination given rowing speeds and time. -/
def RowingProblem (stillWaterSpeed currentVelocity totalTime : ℝ) : Prop :=
  let downstreamSpeed := stillWaterSpeed + currentVelocity
  let upstreamSpeed := stillWaterSpeed - currentVelocity
  ∃ (distance : ℝ),
    distance > 0 ∧
    distance / downstreamSpeed + distance / upstreamSpeed = totalTime

/-- Theorem stating that given the specific conditions of the problem, the distance to the destination is 2.4 km. -/
theorem rowing_problem_solution :
  RowingProblem 5 1 1 →
  ∃ (distance : ℝ), distance = 2.4 := by
  sorry

#check rowing_problem_solution

end rowing_problem_solution_l790_79060


namespace complement_probability_l790_79084

theorem complement_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end complement_probability_l790_79084


namespace andrea_jim_age_sum_l790_79044

theorem andrea_jim_age_sum : 
  ∀ (A J x y : ℕ),
  A = J + 29 →                   -- Andrea is 29 years older than Jim
  A - x + J - x = 47 →           -- Sum of their ages x years ago was 47
  J - y = 2 * (J - x) →          -- Jim's age y years ago was twice his age x years ago
  A = 3 * (J - y) →              -- Andrea's current age is three times Jim's age y years ago
  A + J = 79 :=                  -- The sum of their current ages is 79
by
  sorry

end andrea_jim_age_sum_l790_79044


namespace lcm_six_fifteen_l790_79048

theorem lcm_six_fifteen : Nat.lcm 6 15 = 30 := by
  sorry

end lcm_six_fifteen_l790_79048


namespace perfect_square_condition_l790_79040

theorem perfect_square_condition (x y : ℕ) :
  (∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2) ↔ x = y :=
by sorry

end perfect_square_condition_l790_79040


namespace external_tangent_y_intercept_l790_79010

/-- The y-intercept of the common external tangent line with positive slope to two circles -/
theorem external_tangent_y_intercept 
  (center1 : ℝ × ℝ) (radius1 : ℝ) (center2 : ℝ × ℝ) (radius2 : ℝ) 
  (h1 : center1 = (1, 5)) 
  (h2 : radius1 = 3) 
  (h3 : center2 = (15, 10)) 
  (h4 : radius2 = 10) : 
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨ 
       (x - center2.1)^2 + (y - center2.2)^2 = radius2^2)) ∧ 
    b = 7416 / 1000 := by
  sorry

end external_tangent_y_intercept_l790_79010


namespace convex_polygon_sides_l790_79055

/-- A convex polygon with the sum of all angles except two equal to 3240° has 22 sides. -/
theorem convex_polygon_sides (n : ℕ) (sum_except_two : ℝ) : 
  sum_except_two = 3240 → (∃ (a b : ℝ), 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 
    180 * (n - 2) = sum_except_two + a + b) → n = 22 :=
by sorry

end convex_polygon_sides_l790_79055


namespace tangent_and_normal_equations_l790_79005

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 5

/-- The x-coordinate of the point of interest -/
def x₀ : ℝ := 2

/-- The y-coordinate of the point of interest -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line at x₀ -/
def m : ℝ := 2*x₀ - 2

theorem tangent_and_normal_equations :
  (∀ x y, 2*x - y + 1 = 0 ↔ y = m*(x - x₀) + y₀) ∧
  (∀ x y, x + 2*y - 12 = 0 ↔ y = -1/(2*m)*(x - x₀) + y₀) := by
  sorry


end tangent_and_normal_equations_l790_79005


namespace even_function_implies_f_2_eq_3_l790_79092

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_eq_3 :
  (∀ x : ℝ, f a x = f a (-x)) → f a 2 = 3 := by
  sorry

end even_function_implies_f_2_eq_3_l790_79092


namespace poetry_competition_results_l790_79099

-- Define the contingency table
def a : ℕ := 6
def b : ℕ := 9
def c : ℕ := 4
def d : ℕ := 1
def n : ℕ := 20

-- Define K^2 calculation
def K_squared : ℚ := (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d) : ℚ)

-- Define probabilities for student C
def prob_buzz : ℚ := 3/5
def prob_correct_buzz : ℚ := 4/5

-- Define the score variable X
inductive Score
| neg_one : Score
| zero : Score
| two : Score

-- Define the probability distribution of X
def prob_X : Score → ℚ
| Score.neg_one => prob_buzz * (1 - prob_correct_buzz)
| Score.zero => 1 - prob_buzz
| Score.two => prob_buzz * prob_correct_buzz

-- Define the expected value of X
def E_X : ℚ := -1 * prob_X Score.neg_one + 0 * prob_X Score.zero + 2 * prob_X Score.two

-- Define the condition for p
def p_condition (p : ℚ) : Prop := 
  |3 * p + 2.52 - (4 * p + 1.68)| ≤ 1/10 ∧ 0 < p ∧ p < 1

theorem poetry_competition_results :
  K_squared < 3841/1000 ∧
  prob_X Score.neg_one = 12/100 ∧
  prob_X Score.zero = 2/5 ∧
  prob_X Score.two = 24/50 ∧
  E_X = 21/25 ∧
  ∀ p, p_condition p ↔ 37/50 ≤ p ∧ p ≤ 47/50 :=
sorry

end poetry_competition_results_l790_79099


namespace exponent_calculation_l790_79074

theorem exponent_calculation (a : ℝ) : a^3 * a * a^4 + (-3 * a^4)^2 = 10 * a^8 := by
  sorry

end exponent_calculation_l790_79074


namespace nearest_integer_to_3_plus_sqrt2_to_6_l790_79070

theorem nearest_integer_to_3_plus_sqrt2_to_6 :
  ∃ n : ℤ, ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - n| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - m| ∧ n = 3707 :=
by sorry

end nearest_integer_to_3_plus_sqrt2_to_6_l790_79070


namespace sum_of_squares_l790_79068

theorem sum_of_squares (a b c d : ℝ) : 
  a + b = -3 →
  a * b + b * c + c * a = -4 →
  a * b * c + b * c * d + c * d * a + d * a * b = 14 →
  a * b * c * d = 30 →
  a^2 + b^2 + c^2 + d^2 = 141 / 4 := by sorry

end sum_of_squares_l790_79068


namespace max_daily_sales_revenue_l790_79015

def P (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def Q (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 30 then -t + 40
  else 0

def dailySalesRevenue (t : ℕ) : ℝ := P t * Q t

theorem max_daily_sales_revenue :
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ dailySalesRevenue t = 1125) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → dailySalesRevenue t ≤ 1125) ∧
  (dailySalesRevenue 25 = 1125) :=
sorry

end max_daily_sales_revenue_l790_79015


namespace discontinuity_at_three_l790_79050

/-- The function f(x) = 6 / (x-3)² is discontinuous at x = 3 -/
theorem discontinuity_at_three (f : ℝ → ℝ) (h : ∀ x ≠ 3, f x = 6 / (x - 3)^2) :
  ¬ ContinuousAt f 3 := by
  sorry

end discontinuity_at_three_l790_79050


namespace inequality_solution_set_l790_79090

theorem inequality_solution_set :
  let S : Set ℝ := {x | (3 - x) / (2 * x - 4) < 1}
  S = {x | x < 2 ∨ x > 7/3} := by
  sorry

end inequality_solution_set_l790_79090


namespace marbles_probability_l790_79063

theorem marbles_probability (total : ℕ) (red : ℕ) (h1 : total = 48) (h2 : red = 12) :
  let p := (total - red) / total
  p * p = 9 / 16 := by
  sorry

end marbles_probability_l790_79063


namespace tribe_leadership_combinations_l790_79094

def tribe_size : ℕ := 12
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_combinations :
  (tribe_size) *
  (tribe_size - num_chiefs) *
  (tribe_size - num_chiefs - 1) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 221760 :=
by sorry

end tribe_leadership_combinations_l790_79094


namespace two_number_difference_l790_79075

theorem two_number_difference (a b : ℕ) : 
  a + b = 20460 → 
  b % 12 = 0 → 
  a = b / 10 → 
  b - a = 17314 := by sorry

end two_number_difference_l790_79075


namespace dress_price_difference_l790_79082

theorem dress_price_difference (discounted_price : ℝ) (discount_rate : ℝ) (increase_rate : ℝ) : 
  discounted_price = 61.2 ∧ discount_rate = 0.15 ∧ increase_rate = 0.25 →
  (discounted_price / (1 - discount_rate) * (1 + increase_rate)) - (discounted_price / (1 - discount_rate)) = 4.5 := by
sorry

end dress_price_difference_l790_79082


namespace square_max_perimeter_l790_79079

/-- A right-angled quadrilateral inscribed in a circle --/
structure InscribedRightQuadrilateral (r : ℝ) where
  x : ℝ
  y : ℝ
  right_angled : x^2 + y^2 = (2*r)^2
  inscribed : x > 0 ∧ y > 0

/-- The perimeter of an inscribed right-angled quadrilateral --/
def perimeter (r : ℝ) (q : InscribedRightQuadrilateral r) : ℝ :=
  2 * (q.x + q.y)

/-- The statement that the square has the largest perimeter --/
theorem square_max_perimeter (r : ℝ) (hr : r > 0) :
  ∀ q : InscribedRightQuadrilateral r,
    perimeter r q ≤ 4 * r * Real.sqrt 2 :=
sorry

end square_max_perimeter_l790_79079


namespace emily_savings_l790_79021

def shoe_price : ℕ := 50
def promotion_b_discount : ℕ := 20

def cost_promotion_a (price : ℕ) : ℕ := price + price / 2

def cost_promotion_b (price : ℕ) (discount : ℕ) : ℕ := price + (price - discount)

theorem emily_savings : 
  cost_promotion_b shoe_price promotion_b_discount - cost_promotion_a shoe_price = 5 := by
sorry

end emily_savings_l790_79021


namespace local_extremum_sum_l790_79049

/-- A function f with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_sum (a b : ℝ) :
  f a b 1 = 10 ∧ f' a b 1 = 0 → a + b = -7 := by
  sorry

end local_extremum_sum_l790_79049


namespace triangle_area_is_eight_l790_79093

-- Define the slopes and intersection point
def slope1 : ℝ := -1
def slope2 : ℝ := 3
def intersection : ℝ × ℝ := (1, 3)

-- Define the lines
def line1 (x : ℝ) : ℝ := slope1 * (x - intersection.1) + intersection.2
def line2 (x : ℝ) : ℝ := slope2 * (x - intersection.1) + intersection.2
def line3 (x y : ℝ) : Prop := x - y = 2

-- Define the points of the triangle
def pointA : ℝ × ℝ := intersection
def pointB : ℝ × ℝ := (-1, -3)  -- Intersection of line2 and line3
def pointC : ℝ × ℝ := (3, 1)    -- Intersection of line1 and line3

-- Theorem statement
theorem triangle_area_is_eight :
  let area := (1/2) * abs (
    pointA.1 * (pointB.2 - pointC.2) +
    pointB.1 * (pointC.2 - pointA.2) +
    pointC.1 * (pointA.2 - pointB.2)
  )
  area = 8 := by sorry

end triangle_area_is_eight_l790_79093


namespace tangent_circle_problem_l790_79028

theorem tangent_circle_problem (center_to_intersection : ℚ) (radius : ℚ) (center_to_line : ℚ) (x : ℚ) :
  center_to_intersection = 3/8 →
  radius = 3/16 →
  center_to_line = 1/2 →
  x = center_to_intersection + radius - center_to_line →
  x = 1/16 := by
sorry

end tangent_circle_problem_l790_79028


namespace matrix_equation_l790_79076

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 3, 4]

def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]

theorem matrix_equation (a b c d : ℝ) (h1 : A * B a b c d = B a b c d * A) 
  (h2 : 4 * b ≠ c) : (a - 2 * d) / (c - 4 * b) = 3 / 10 := by
  sorry

end matrix_equation_l790_79076


namespace function_characterization_l790_79064

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem function_characterization :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end function_characterization_l790_79064


namespace length_AB_l790_79088

/-- A line passing through (2,0) with slope 2 -/
def line_l (x y : ℝ) : Prop := y = 2 * x - 4

/-- The curve y^2 - 4x = 0 -/
def curve (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A is on both the line and the curve -/
def point_A (x y : ℝ) : Prop := line_l x y ∧ curve x y

/-- Point B is on both the line and the curve, and is different from A -/
def point_B (x y : ℝ) : Prop := line_l x y ∧ curve x y ∧ (x, y) ≠ (1, -2)

/-- The main theorem: the length of AB is 3√5 -/
theorem length_AB :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 3 * Real.sqrt 5 :=
sorry

end length_AB_l790_79088


namespace complex_multiplication_l790_79096

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end complex_multiplication_l790_79096


namespace shortest_distance_parabola_to_line_l790_79062

/-- The shortest distance from a point on the parabola y = x^2 to the line x - y - 2 = 0 is 7√2/8 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | p.1 - p.2 - 2 = 0}
  ∃ d : ℝ, d = 7 * Real.sqrt 2 / 8 ∧
    ∀ p ∈ parabola, ∀ q ∈ line,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end shortest_distance_parabola_to_line_l790_79062


namespace evenness_condition_l790_79097

/-- Given a real number ω, prove that there exists a real number a such that 
    f(x+a) is an even function, where f(x) = (x-6)^2 * sin(ωx), 
    if and only if ω = π/4 -/
theorem evenness_condition (ω : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, (((x + a - 6)^2 * Real.sin (ω * (x + a))) = 
                      (((-x) + a - 6)^2 * Real.sin (ω * ((-x) + a)))))
  ↔ 
  ω = π / 4 := by
sorry

end evenness_condition_l790_79097


namespace girls_average_score_l790_79000

theorem girls_average_score (num_boys num_girls : ℕ) (boys_avg class_avg girls_avg : ℚ) : 
  num_boys = 12 → 
  num_girls = 4 → 
  boys_avg = 84 → 
  class_avg = 86 → 
  (num_boys * boys_avg + num_girls * girls_avg) / (num_boys + num_girls) = class_avg → 
  girls_avg = 92 := by
  sorry

end girls_average_score_l790_79000


namespace car_meeting_points_distance_prove_car_meeting_points_distance_l790_79080

/-- Given two cars starting from points A and B, if they meet at a point 108 km from B, 
    then continue to each other's starting points and return, meeting again at a point 84 km from A, 
    the distance between their two meeting points is 48 km. -/
theorem car_meeting_points_distance : ℝ → Prop :=
  fun d =>
    let first_meeting := d - 108
    let second_meeting := 84
    first_meeting - second_meeting = 48

/-- Proof of the theorem -/
theorem prove_car_meeting_points_distance : ∃ d : ℝ, car_meeting_points_distance d :=
sorry

end car_meeting_points_distance_prove_car_meeting_points_distance_l790_79080


namespace cubic_polynomial_roots_l790_79046

/-- Given a cubic polynomial with two equal integer roots, prove |ab| = 5832 -/
theorem cubic_polynomial_roots (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (∃ r s : ℤ, (∀ x : ℝ, x^3 + a*x^2 + b*x + 16*a = (x - r)^2 * (x - s)) ∧ 
   (r ≠ s)) → 
  |a * b| = 5832 := by
  sorry

end cubic_polynomial_roots_l790_79046


namespace subtracted_value_l790_79052

theorem subtracted_value (N : ℕ) (V : ℕ) (h1 : N = 800) (h2 : N / 5 - V = 6) : V = 154 := by
  sorry

end subtracted_value_l790_79052


namespace wallpaper_removal_time_l790_79029

/-- Proves that the time taken to remove wallpaper from the first wall is 2 hours -/
theorem wallpaper_removal_time (total_walls : ℕ) (walls_removed : ℕ) (remaining_time : ℕ) :
  total_walls = 8 →
  walls_removed = 1 →
  remaining_time = 14 →
  remaining_time / (total_walls - walls_removed) = 2 := by
  sorry

end wallpaper_removal_time_l790_79029


namespace simplify_expression_evaluate_expression_l790_79027

-- Part 1
theorem simplify_expression (a : ℝ) : -2*a^2 + 3 - (3*a^2 - 6*a + 1) + 3 = -5*a^2 + 6*a + 5 := by
  sorry

-- Part 2
theorem evaluate_expression (x y : ℝ) (hx : x = -2) (hy : y = -3) :
  1/2*x - 2*(x - 1/3*y^2) + (-3/2*x + 1/3*y^2) = 15 := by
  sorry

end simplify_expression_evaluate_expression_l790_79027


namespace intersection_point_l790_79058

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 7 = (y - 2) / 1 ∧ (y - 2) / 1 = (z - 6) / (-1)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  4 * x + y - 6 * z - 5 = 0

/-- The theorem stating that (8, 3, 5) is the unique point of intersection -/
theorem intersection_point :
  ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 8 ∧ y = 3 ∧ z = 5 := by sorry

end intersection_point_l790_79058


namespace sevenDigitIntegers_eq_630_l790_79059

/-- The number of different positive, seven-digit integers that can be formed
    using the digits 2, 2, 3, 5, 5, 9, and 9 -/
def sevenDigitIntegers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, seven-digit integers
    that can be formed using the digits 2, 2, 3, 5, 5, 9, and 9 is 630 -/
theorem sevenDigitIntegers_eq_630 : sevenDigitIntegers = 630 := by
  sorry

end sevenDigitIntegers_eq_630_l790_79059


namespace odd_divides_power_factorial_minus_one_l790_79033

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h : n > 0) (hodd : Odd n) :
  n ∣ 2^(n!) - 1 :=
by sorry

end odd_divides_power_factorial_minus_one_l790_79033


namespace g_difference_l790_79003

/-- The function g defined as g(n) = n^3 + 3n^2 + 3n + 1 -/
def g (n : ℝ) : ℝ := n^3 + 3*n^2 + 3*n + 1

/-- Theorem stating that g(s) - g(s-2) = 6s^2 + 2 for any real number s -/
theorem g_difference (s : ℝ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end g_difference_l790_79003


namespace small_mold_radius_l790_79037

theorem small_mold_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 2 → n = 64 → (2 / 3 * Real.pi * R^3) = (n * (2 / 3 * Real.pi * r^3)) → r = 1 / 2 := by
  sorry

end small_mold_radius_l790_79037


namespace min_chocolate_cookies_l790_79078

theorem min_chocolate_cookies (chocolate_batch_size peanut_batch_size total_cookies : ℕ)
  (chocolate_ratio peanut_ratio : ℕ) :
  chocolate_batch_size = 5 →
  peanut_batch_size = 6 →
  chocolate_ratio = 3 →
  peanut_ratio = 2 →
  total_cookies = 94 →
  ∃ (chocolate_batches peanut_batches : ℕ),
    chocolate_batches * chocolate_batch_size + peanut_batches * peanut_batch_size = total_cookies ∧
    chocolate_batches * chocolate_batch_size * peanut_ratio = peanut_batches * peanut_batch_size * chocolate_ratio ∧
    chocolate_batches * chocolate_batch_size ≥ 60 ∧
    ∀ (c p : ℕ), c * chocolate_batch_size + p * peanut_batch_size = total_cookies →
      c * chocolate_batch_size * peanut_ratio = p * peanut_batch_size * chocolate_ratio →
      c * chocolate_batch_size ≥ chocolate_batches * chocolate_batch_size :=
by sorry

end min_chocolate_cookies_l790_79078


namespace right_triangle_third_side_l790_79011

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h : Real.sqrt (a^2 - 6*a + 9) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) :=
by sorry

end right_triangle_third_side_l790_79011


namespace degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l790_79031

/-- Represents a monomial in variables x and y -/
structure Monomial :=
  (coeff : ℤ)
  (x_power : ℕ)
  (y_power : ℕ)

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ :=
  m.x_power + m.y_power

/-- The monomial -x³y -/
def mono : Monomial :=
  { coeff := -1, x_power := 3, y_power := 1 }

/-- Theorem stating that the degree of -x³y is 4 -/
theorem degree_of_minus_x_cubed_y_is_four :
  degree mono = 4 :=
sorry

/-- Theorem stating that the degree of -x³y is not 3 -/
theorem degree_of_minus_x_cubed_y_is_not_three :
  degree mono ≠ 3 :=
sorry

end degree_of_minus_x_cubed_y_is_four_degree_of_minus_x_cubed_y_is_not_three_l790_79031


namespace tobias_daily_hours_l790_79051

/-- Proves that Tobias plays 5 hours per day given the conditions of the problem -/
theorem tobias_daily_hours (nathan_daily_hours : ℕ) (nathan_days : ℕ) (tobias_days : ℕ) (total_hours : ℕ) :
  nathan_daily_hours = 3 →
  nathan_days = 14 →
  tobias_days = 7 →
  total_hours = 77 →
  ∃ (tobias_daily_hours : ℕ), 
    tobias_daily_hours * tobias_days + nathan_daily_hours * nathan_days = total_hours ∧
    tobias_daily_hours = 5 :=
by sorry

end tobias_daily_hours_l790_79051


namespace solution_greater_than_two_l790_79023

theorem solution_greater_than_two (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 6 := by
  sorry

end solution_greater_than_two_l790_79023


namespace subtracted_value_l790_79083

theorem subtracted_value (x : ℝ) (h1 : (x - 5) / 7 = 7) : 
  ∃ y : ℝ, (x - y) / 10 = 4 ∧ y = 14 := by
  sorry

end subtracted_value_l790_79083


namespace decimal_expansion_222nd_digit_l790_79016

/-- The decimal expansion of 47/777 -/
def decimal_expansion : ℚ := 47 / 777

/-- The length of the repeating block in the decimal expansion -/
def repeat_length : ℕ := 6

/-- The position we're interested in -/
def position : ℕ := 222

/-- The function that returns the nth digit after the decimal point in the decimal expansion -/
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

theorem decimal_expansion_222nd_digit :
  nth_digit position = 5 := by sorry

end decimal_expansion_222nd_digit_l790_79016


namespace arithmetic_computation_l790_79086

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 7 * 2 / 2 = 20 := by
  sorry

end arithmetic_computation_l790_79086


namespace negative_three_times_inequality_l790_79045

theorem negative_three_times_inequality (m n : ℝ) (h : m > n) : -3*m < -3*n := by
  sorry

end negative_three_times_inequality_l790_79045


namespace symmetric_difference_properties_l790_79014

open Set

variable {α : Type*} [MeasurableSpace α]

def symmetricDifference (A B : Set α) : Set α := (A \ B) ∪ (B \ A)

theorem symmetric_difference_properties 
  (A B : ℕ → Set α) : 
  (symmetricDifference (A 1) (B 1) = symmetricDifference (Aᶜ 1) (Bᶜ 1)) ∧ 
  (symmetricDifference (⋃ n, A n) (⋃ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) ∧
  (symmetricDifference (⋂ n, A n) (⋂ n, B n) ⊆ ⋃ n, symmetricDifference (A n) (B n)) := by
  sorry


end symmetric_difference_properties_l790_79014


namespace isosceles_trapezoid_tangent_ratio_l790_79065

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid :=
  (A B C D : Point)

/-- Checks if a point lies on a given line segment -/
def pointOnSegment (P Q R : Point) : Prop :=
  sorry

/-- Checks if a line is tangent to a circle -/
def isTangent (P Q : Point) (circle : Circle) : Prop :=
  sorry

/-- Checks if a trapezoid is circumscribed around a circle -/
def isCircumscribed (trapezoid : IsoscelesTrapezoid) (circle : Circle) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ :=
  sorry

theorem isosceles_trapezoid_tangent_ratio 
  (trapezoid : IsoscelesTrapezoid) 
  (circle : Circle) 
  (P Q R S : Point) :
  isCircumscribed trapezoid circle →
  isTangent P S circle →
  pointOnSegment P Q R →
  pointOnSegment P S R →
  distance P Q / distance Q R = distance R S / distance S R :=
sorry

end isosceles_trapezoid_tangent_ratio_l790_79065


namespace line_slope_intercept_sum_l790_79095

/-- Given a line passing through the points (2, -1) and (5, 2), 
    prove that the sum of its slope and y-intercept is -2. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (2 : ℝ) * m + b = -1 ∧ 
  (5 : ℝ) * m + b = 2 → 
  m + b = -2 := by
  sorry

end line_slope_intercept_sum_l790_79095


namespace code_number_correspondence_exists_l790_79012

-- Define the set of codes
def Codes : Type := Fin 5 → Fin 3 → Char

-- Define the set of numbers
def Numbers : Type := Fin 5 → Nat

-- Define the given codes
def given_codes : Codes := λ i j ↦ 
  match i, j with
  | 0, 0 => 'R' | 0, 1 => 'W' | 0, 2 => 'Q'
  | 1, 0 => 'S' | 1, 1 => 'X' | 1, 2 => 'W'
  | 2, 0 => 'P' | 2, 1 => 'S' | 2, 2 => 'T'
  | 3, 0 => 'X' | 3, 1 => 'N' | 3, 2 => 'Y'
  | 4, 0 => 'N' | 4, 1 => 'X' | 4, 2 => 'Y'
  | _, _ => 'A' -- Default case, should never be reached

-- Define the given and solution numbers
def given_and_solution_numbers : Numbers := λ i ↦
  match i with
  | 0 => 286
  | 1 => 540
  | 2 => 793
  | 3 => 948
  | 4 => 450

-- Define a bijection type between Codes and Numbers
def CodeNumberBijection := {f : Codes → Numbers // Function.Bijective f}

theorem code_number_correspondence_exists : ∃ (f : CodeNumberBijection), 
  ∀ (i : Fin 5), f.val given_codes i = given_and_solution_numbers i :=
sorry

end code_number_correspondence_exists_l790_79012


namespace min_treasures_is_15_l790_79013

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried." -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried." -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried." -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried." -/
def signs_3 : ℕ := 3

/-- Predicate to check if a sign is truthful given the number of treasures -/
def is_truthful (sign_value : ℕ) (num_treasures : ℕ) : Prop :=
  sign_value ≠ num_treasures

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ),
    n = 15 ∧
    (∀ m : ℕ, m < n →
      ¬(is_truthful signs_15 m ∧
        is_truthful signs_8 m ∧
        is_truthful signs_4 m ∧
        is_truthful signs_3 m)) :=
by
  sorry

end min_treasures_is_15_l790_79013
